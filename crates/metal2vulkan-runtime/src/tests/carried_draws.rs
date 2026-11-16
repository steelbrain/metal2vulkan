//! A render encoder's kept draw carried into the next render encoder of its command buffer, or of
//! a later command buffer of its queue (`render/carried.rs`): the next encoder's first draw of the
//! same state is rebound against it (`WorkCounts::draws_rebound_carried`), with the pixels of the
//! same passes drawn every step, the values its own calls bound, a texture slot its state does not
//! read left unbound there stopping nothing; a draw that reads its pass's color attachment through
//! the texture's copy carries too, and the next encoder's draw reads that encoder's copy; a carry
//! its queue holds keeps no object the program released alive, and the next command buffer's
//! draw takes no dead one; command buffers encoded side by side, each two committed once both
//! encoded, each open from a carry their queue holds from earlier commits (`queue::CARRIES`), a
//! buffer one of them names freed while it is held; it stops where the bind state holds a slot the new encoder did not bind
//! (`draws_carry_stopped_stale`) or where the draw reads an attachment of its pass through a slot
//! narrowed to other levels (`draws_carry_stopped_attached`), or where it reads its pass's color
//! attachment through the texture's copy and every step would read it through a snapshot
//! (`draws_carry_stopped_snapshot`); and the carry is let go where a call
//! comes between the encoders (`kept_draws_carry_dropped_between`) or the next pass is of another
//! layout (`kept_draws_carry_dropped_layout`). An encoder that keeps none counts what a kept draw
//! of it would have come to (`encoders_none_kept_next_*`). In builds with debug assertions the
//! first carried draw, and every second one after it, takes every step and is checked against the
//! rebound (the module's check), so these tests run that check too.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixtures
//! `validation/fixtures/public/engine/tob_renders.ll` and `engine/alv_levels.ll` (`llvm-as`).

use super::blit_holds::{over_host, until, Host, Scratch, WAIT};
use super::draw_again::{adding_state, device, plan_device, warm, Heard};
use super::encoder_holds::{bytes_of, library, words_of};
use super::shadow_standing_device::reads_through_snapshots;
use crate::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandQueue, Device, LoadAction, Origin,
    PipelineOptions, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, SamplerState, ScissorRect, Size, StoreAction, Texture, TextureDescriptor,
    WorkCounts,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{IndexType, PixelFormat, PrimitiveType, SamplerMipFilter, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// The side of the target.
const SIDE: u64 = 16;

/// The offset of a buffer's second color: a constant buffer's offset alignment on every device.
const SECOND: u64 = 256;

/// The places for colors in a buffer of them.
const COLORS: usize = 3;

/// A pass's calls.
type Pass<'a> = &'a dyn Fn(&mut RenderCommandEncoder);

/// How a frame's passes are encoded: in one command buffer, one after another; in one command
/// buffer with an empty blit encoder between each two; each in a command buffer of its own, of one
/// queue, which holds the carry of each for the next (`kept_draws_carried_across`); or each in a
/// command buffer of a queue of its own, after the one before completed, which carries nothing.
#[derive(Clone, Copy)]
enum Apart {
    Together,
    Blits,
    CommandBuffers,
    Queues,
}

impl Apart {
    /// The command buffer a frame's first pass encodes into, of `queue`, or of a new queue where
    /// each pass has its own: kept in `queue`.
    fn first(self, queue: &mut CommandQueue) -> CommandBuffer {
        if let Apart::Queues = self {
            *queue = queue.device().new_command_queue().unwrap();
        }
        queue.command_buffer().unwrap()
    }

    /// Encode what comes between two passes into `command_buffer`, the one the pass before
    /// encoded into, of `queue`: where the next pass has a command buffer of its own, it is
    /// committed, and the next one, of `queue` or of a new queue kept there, takes its place.
    fn between(self, queue: &mut CommandQueue, command_buffer: &mut CommandBuffer) {
        match self {
            Apart::Together => {}
            Apart::Blits => {
                let blit = command_buffer.blit_command_encoder().unwrap();
                blit.end_encoding().unwrap();
            }
            Apart::CommandBuffers => {
                command_buffer.commit().unwrap();
                *command_buffer = queue.command_buffer().unwrap();
            }
            Apart::Queues => {
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                *command_buffer = self.first(queue);
            }
        }
    }
}

/// What a frame drew, and the device's counts' change over it, but over the spans of it `left_out`
/// (`before` and `after` each), which ran no command of the frame's.
struct Frame {
    words: Vec<u32>,
    before: WorkCounts,
    after: WorkCounts,
    left_out: Vec<(WorkCounts, WorkCounts)>,
}

impl Frame {
    /// How the count `of` changed over the frame.
    fn counted(&self, of: fn(&WorkCounts) -> u64) -> u64 {
        let left_out = self.left_out.iter();
        let left_out: u64 = left_out.map(|(before, after)| of(after) - of(before)).sum();
        of(&self.after) - of(&self.before) - left_out
    }

    /// The carry's counts over the frame: carried, rebound carried, stopped stale, stopped
    /// attached, stopped otherwise, dropped between, dropped of another layout, dropped under.
    fn carries(&self) -> [u64; 8] {
        [
            self.counted(|c| c.kept_draws_carried),
            self.counted(|c| c.draws_rebound_carried),
            self.counted(|c| c.draws_carry_stopped_stale),
            self.counted(|c| c.draws_carry_stopped_attached),
            self.counted(|c| c.draws_carry_stopped_other),
            self.counted(|c| c.kept_draws_carry_dropped_between),
            self.counted(|c| c.kept_draws_carry_dropped_layout),
            self.counted(|c| c.kept_draws_carry_dropped_under),
        ]
    }

    /// The reads of the frame's stage tables its draws' settles read whole, by why
    /// (`stage_tables_read_whole_*`, in their order: unread, moved, interface, maps, views, spare,
    /// carried, unlisted).
    fn read_whole(&self) -> [u64; 8] {
        [
            self.counted(|c| c.stage_tables_read_whole_unread),
            self.counted(|c| c.stage_tables_read_whole_moved),
            self.counted(|c| c.stage_tables_read_whole_interface),
            self.counted(|c| c.stage_tables_read_whole_maps),
            self.counted(|c| c.stage_tables_read_whole_views),
            self.counted(|c| c.stage_tables_read_whole_spare),
            self.counted(|c| c.stage_tables_read_whole_carried),
            self.counted(|c| c.stage_tables_read_whole_unlisted),
        ]
    }

    /// The render encoders that opened from a carry their queue held (`kept_draws_carried_across`).
    fn across(&self) -> u64 {
        self.counted(|c| c.kept_draws_carried_across)
    }

    /// What the render encoders that ended keeping no draw came to for their command buffer's
    /// next render encoder (`encoders_none_kept_next_*`): split, other ends, between, last,
    /// another layout, under, fits.
    fn none_kept_next(&self) -> [u64; 7] {
        [
            self.counted(|c| c.encoders_none_kept_next_split),
            self.counted(|c| c.encoders_none_kept_next_ends),
            self.counted(|c| c.encoders_none_kept_next_between),
            self.counted(|c| c.encoders_none_kept_next_last),
            self.counted(|c| c.encoders_none_kept_next_layout),
            self.counted(|c| c.encoders_none_kept_next_under),
            self.counted(|c| c.encoders_none_kept_next_fits),
        ]
    }

    /// The render encoders that ended carrying nothing, by reason (`encoders_uncarried_*`, in
    /// their order): drawless, none kept, and the eleven others together.
    fn uncarried(&self) -> [u64; 3] {
        let others: [fn(&WorkCounts) -> u64; 11] = [
            |c| c.encoders_uncarried_unsettled,
            |c| c.encoders_uncarried_split,
            |c| c.encoders_uncarried_pass,
            |c| c.encoders_uncarried_replay,
            |c| c.encoders_uncarried_commands,
            |c| c.encoders_uncarried_shader_tests,
            |c| c.encoders_uncarried_imageblock,
            |c| c.encoders_uncarried_visibility,
            |c| c.encoders_uncarried_instance,
            |c| c.encoders_uncarried_unrebound,
            |c| c.encoders_uncarried_failing_state,
        ];
        [
            self.counted(|c| c.encoders_uncarried_drawless),
            self.counted(|c| c.encoders_uncarried_none_kept),
            others.into_iter().map(|of| self.counted(of)).sum(),
        ]
    }
}

/// The carry's counts ([`Frame::carries`]) of a frame whose carried draws are `carried`, the
/// rebinds of which all go on, the first and every second one after it taking every step in a
/// build with debug assertions (the module docs); the frame's read back, a blit encoder after the
/// last pass, lets that pass's kept draw go, counted as one between.
fn rebound(carried: u64) -> [u64; 8] {
    let rebound = match cfg!(debug_assertions) {
        true => carried / 2,
        false => carried,
    };
    [carried, rebound, 0, 0, 0, 1, 0, 0]
}

/// The carry's counts of a frame that carried nothing: but the last pass's kept draw, which the
/// read back lets go (`rebound`), and one let go at each of `blits` blit encoders between passes.
fn dropped(blits: u64) -> [u64; 8] {
    [0, 0, 0, 0, 0, blits + 1, 0, 0]
}

/// A pass over `level` of `target`, cleared where `first`, else loaded, and stored.
fn pass_over((target, level): (&Texture, u64), first: bool) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.level = level;
    color.load_action = match first {
        true => LoadAction::Clear,
        false => LoadAction::Load,
    };
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    pass
}

/// A private `RGBA8Unorm` target of [`SIDE`] square.
fn target(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// Encode `passes` into a new target on `queue`, encoded as `apart` says, commit and wait: what
/// the frame drew.
fn frame(queue: &CommandQueue, passes: &[Pass<'_>], apart: Apart) -> Frame {
    let device = queue.device();
    let target = target(device);
    let back = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let before = device.work_counts();
    let mut queue = queue.clone();
    let mut command_buffer = apart.first(&mut queue);
    for (at, pass) in passes.iter().enumerate() {
        if at > 0 {
            apart.between(&mut queue, &mut command_buffer);
        }
        let descriptor = pass_over((&target, 0), at == 0);
        let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
        pass(&mut encoder);
        encoder.end_encoding().unwrap();
    }
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        SIDE * 4,
        SIDE * SIDE * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    Frame {
        words: words_of(&back),
        before,
        after: device.work_counts(),
        left_out: Vec::new(),
    }
}

/// A shared buffer of [`COLORS`] places of [`SECOND`] bytes, `colors` in the first ones: every
/// buffer of one length, its bound length's class (`bind::length_class`, of 512 to 1023 bytes)
/// the same at offsets 0 and [`SECOND`] alike, so a pass's other buffer or offset keeps the bind
/// state's shape, as a draw rebound needs.
fn colors(device: &Device, colors: &[[f32; 4]]) -> Buffer {
    assert!(colors.len() <= COLORS);
    let mut bytes = vec![0; COLORS * SECOND as usize];
    for (at, color) in colors.iter().enumerate() {
        let color = bytes_of(color, f32::to_le_bytes);
        let at = at * SECOND as usize;
        bytes[at..at + color.len()].copy_from_slice(&color);
    }
    device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap()
}

/// Set `state` and the rectangle over the whole target, bind the color at `offset` of `color` at
/// fragment buffer 0, and draw the rectangle.
fn rectangle(
    encoder: &mut RenderCommandEncoder,
    state: &RenderPipelineState,
    (color, offset): (&Buffer, u64),
) {
    encoder.set_render_pipeline_state(state).unwrap();
    let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
    encoder.set_vertex_bytes(&corners, 0).unwrap();
    (encoder.set_fragment_buffer(Some(color), offset, 0)).unwrap();
    let drawn = encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0);
    assert_eq!(drawn, Ok(()));
}

/// The color channels of a pixel `word` of the target, red first.
fn channels(word: u32) -> [u8; 4] {
    word.to_le_bytes()
}

/// Passes of the same state in one command buffer, each binding as the one before: each but the
/// first rebinds the kept draw its encoder's last one carried, with the pixels of the same passes
/// in command buffers of queues of their own, which carry nothing. In command buffers of their own
/// of one queue, each but the first rebinds the kept draw the queue held from the commit before
/// (`kept_draws_carried_across`), with the same pixels. A blit encoder between each two lets the
/// carried draw go, counted, with the same pixels.
#[test]
fn passes_of_the_same_state_rebind_the_carried_draw() {
    let scratch = Scratch::new("carried-same");
    let heard = Heard::default();
    let device = plan_device(&scratch.0, &[], &heard);
    let queue = device.new_command_queue().unwrap();
    let state = adding_state(&device);
    let color = colors(&device, &[[1.0 / 16.0; 4]]);
    let pass = |encoder: &mut RenderCommandEncoder| rectangle(encoder, &state, (&color, 0));
    warm(&device, pass);
    const PASSES: usize = 5;
    let passes: [Pass<'_>; PASSES] = [&pass; PASSES];
    let together = frame(&queue, &passes, Apart::Together);
    let across = frame(&queue, &passes, Apart::CommandBuffers);
    let alone = frame(&queue, &passes, Apart::Queues);
    let blits = frame(&queue, &passes, Apart::Blits);
    let carried = PASSES as u64 - 1;
    assert_eq!(together.carries(), rebound(carried), "together");
    assert_eq!(
        across.carries(),
        rebound(carried),
        "in command buffers of their own"
    );
    assert_eq!(across.across(), carried, "carried across");
    assert_eq!(
        alone.carries(),
        dropped(0),
        "in command buffers of queues of their own"
    );
    assert_eq!(blits.carries(), dropped(carried), "with blits between");
    assert_eq!(together.words, alone.words, "the pixels of the passes");
    assert_eq!(
        across.words, alone.words,
        "the pixels of the passes, carried across"
    );
    assert_eq!(
        blits.words, alone.words,
        "the pixels of the passes, blits between"
    );
    // Each pass added its color, about 16 in each channel (as the device rounds each blend).
    let pixel = channels(together.words[0]);
    let least = 15 * PASSES as u8;
    assert!(
        pixel.iter().all(|&c| c >= least),
        "every pass's color added: {pixel:?}"
    );
}

/// A render encoder that ends carrying nothing counts the first reason
/// (`encoders_uncarried_*`): passes that each carry their kept draw into the next count none, and
/// passes that draw nothing count each as drawless.
#[test]
fn an_encoder_that_carries_nothing_counts_why() {
    let scratch = Scratch::new("carried-uncarried");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let queue = device.new_command_queue().unwrap();
    let state = adding_state(&device);
    let color = colors(&device, &[[1.0 / 16.0; 4]]);
    let pass = |encoder: &mut RenderCommandEncoder| rectangle(encoder, &state, (&color, 0));
    let drawless = |_: &mut RenderCommandEncoder| {};
    warm(&device, pass);
    const PASSES: usize = 3;
    let passes: [Pass<'_>; PASSES] = [&pass; PASSES];
    let drawn = frame(&queue, &passes, Apart::Together);
    let passes: [Pass<'_>; PASSES] = [&drawless; PASSES];
    let cleared = frame(&queue, &passes, Apart::Together);
    assert_eq!(drawn.carries(), rebound(PASSES as u64 - 1), "drawn");
    assert_eq!(drawn.uncarried(), [0, 0, 0], "drawn");
    assert_eq!(cleared.uncarried(), [PASSES as u64, 0, 0], "drawless");
}

/// Passes that each draw through three pipeline states, each pass after the first starting from a
/// state its encoder's last draw did not draw: every state's kept draw carries with the encoder's
/// last one (`kept_draws_carried_aside`), and the next encoder's first draw of each state is
/// rebound against its own (`draws_rebound_carried`), the later ones after a draw of another
/// state began the pass's instance, with no draw unlike its kept draw or stopped, and the pixels
/// of the same passes in command buffers of queues of their own; in command buffers of their own
/// of one queue, the same, each state's kept draw carried across. Passes of eight states, the most
/// a bind state holds (its last draw's and seven set aside, `live_bind::SET_ASIDE`), carry every
/// one: the next pass's first draw of each is rebound, none takes every step.
#[test]
fn passes_of_several_states_rebind_each_state_s_carried_draw() {
    let scratch = Scratch::new("carried-states");
    let heard = Heard::default();
    let device = plan_device(&scratch.0, &[], &heard);
    let queue = device.new_command_queue().unwrap();
    let states: [RenderPipelineState; 8] = std::array::from_fn(|_| adding_state(&device));
    let color = colors(&device, &[[1.0 / 64.0; 4]]);
    let draw = |encoder: &mut RenderCommandEncoder, order: &[usize]| {
        for &at in order {
            rectangle(encoder, &states[at], (&color, 0));
        }
    };
    let first = |encoder: &mut RenderCommandEncoder| draw(encoder, &[0, 1, 2]);
    let turned = |encoder: &mut RenderCommandEncoder| draw(encoder, &[1, 2, 0]);
    let eight = |encoder: &mut RenderCommandEncoder| draw(encoder, &[0, 1, 2, 3, 4, 5, 6, 7]);
    warm(&device, eight);
    let aside = |frame: &Frame| frame.counted(|c| c.kept_draws_carried_aside);
    let missed = |c: &WorkCounts| c.draws_unlike_kept + c.draws_rebind_stopped;
    // Two carried passes of three states: each state's first draw rebound.
    let passes: [Pass<'_>; 3] = [&first, &turned, &turned];
    let together = frame(&queue, &passes, Apart::Together);
    let across = frame(&queue, &passes, Apart::CommandBuffers);
    let alone = frame(&queue, &passes, Apart::Queues);
    let mut expected = rebound(2 * 3);
    expected[0] = 2;
    for (case, frame) in [("together", &together), ("across", &across)] {
        assert_eq!(
            frame.carries(),
            expected,
            "three states, {case}: the counts"
        );
        assert_eq!(aside(frame), 2 * 2, "three states, {case}");
        assert_eq!(
            frame.counted(missed),
            0,
            "three states, {case}: no draw missed"
        );
        assert_eq!(frame.words, alone.words, "three states, {case}: the pixels");
    }
    assert_eq!(across.across(), 2, "three states: carried across");
    // A carried pass of eight states: all eight rebound, none taking every step but the first
    // pass's.
    let passes: [Pass<'_>; 2] = [&eight, &eight];
    let together = frame(&queue, &passes, Apart::Together);
    let alone = frame(&queue, &passes, Apart::Queues);
    let mut expected = rebound(8);
    expected[0] = 1;
    assert_eq!(together.carries(), expected, "eight states: the counts");
    assert_eq!(aside(&together), 7, "eight states: carried aside");
    let none_kept = |frame: &Frame| frame.counted(|c| c.draws_none_kept);
    assert_eq!(
        none_kept(&together),
        8,
        "eight states: the first pass's draws alone"
    );
    assert_eq!(together.counted(missed), 0, "eight states: no draw missed");
    assert_eq!(together.words, alone.words, "eight states: the pixels");
}

/// The next encoder binds the same state and buffers but for one slot's value: the same buffer at
/// another offset, or another buffer. Its carried draw is rebound with the value its own call
/// bound, which the pixels show: each pass adds the color it bound, in a channel of its own. Each
/// form is the first carried draw of one frame (checked against every step in a build with debug
/// assertions) and the second of another; and so in command buffers of their own of one queue,
/// carried across.
#[test]
fn a_carried_draw_rebinds_the_values_its_encoder_bound() {
    let scratch = Scratch::new("carried-values");
    let heard = Heard::default();
    let device = plan_device(&scratch.0, &[], &heard);
    let queue = device.new_command_queue().unwrap();
    let state = adding_state(&device);
    let red = [0.25, 0.0, 0.0, 0.25];
    let green = [0.0, 0.25, 0.0, 0.25];
    let blue = [0.0, 0.0, 0.25, 0.25];
    let first = colors(&device, &[red, green]);
    let other = colors(&device, &[blue]);
    let pass_red = |encoder: &mut RenderCommandEncoder| rectangle(encoder, &state, (&first, 0));
    let offset = |encoder: &mut RenderCommandEncoder| rectangle(encoder, &state, (&first, SECOND));
    let buffer = |encoder: &mut RenderCommandEncoder| rectangle(encoder, &state, (&other, 0));
    warm(&device, pass_red);
    let frames: [(&str, [Pass<'_>; 3]); 2] = [
        ("offset, then buffer", [&pass_red, &offset, &buffer]),
        ("buffer, then offset", [&pass_red, &buffer, &offset]),
    ];
    for (case, passes) in frames {
        let together = frame(&queue, &passes, Apart::Together);
        let across = frame(&queue, &passes, Apart::CommandBuffers);
        let alone = frame(&queue, &passes, Apart::Queues);
        assert_eq!(together.carries(), rebound(2), "{case}: the counts");
        assert_eq!(across.carries(), rebound(2), "{case}, across: the counts");
        assert_eq!(across.across(), 2, "{case}: carried across");
        assert_eq!(together.words, alone.words, "{case}: the pixels");
        assert_eq!(across.words, alone.words, "{case}, across: the pixels");
        // Red, green and blue each a quarter, the alpha about three quarters (as the device
        // rounds each blend): each pass's own color, none another's again.
        let [r, g, b, a] = channels(together.words[0]);
        let each = r > 0 && r == g && g == b && a > 2 * r;
        assert!(each, "{case}: each pass's color: {:?}", [r, g, b, a]);
    }
}

/// The next encoder binds every slot its draw reads but leaves unbound one the encoder before
/// bound (fragment buffer 1, which no function reads): the carried draw's bind state holds every
/// buffer slot (`argument_table::ReadSlots`), so it holds a slot that encoder's calls did not
/// bind, its rebind stops (stale) and the draw takes every step, with the pixels of the passes in
/// command buffers of queues of their own.
#[test]
fn a_slot_the_next_encoder_leaves_unbound_stops_the_carry() {
    let scratch = Scratch::new("carried-stale");
    let heard = Heard::default();
    let device = plan_device(&scratch.0, &[], &heard);
    let queue = device.new_command_queue().unwrap();
    let state = adding_state(&device);
    let color = colors(&device, &[[1.0 / 16.0; 4]]);
    let unread = colors(&device, &[[1.0; 4]]);
    let with_extra = |encoder: &mut RenderCommandEncoder| {
        (encoder.set_fragment_buffer(Some(&unread), 0, 1)).unwrap();
        rectangle(encoder, &state, (&color, 0));
    };
    let without = |encoder: &mut RenderCommandEncoder| rectangle(encoder, &state, (&color, 0));
    warm(&device, with_extra);
    let passes: [Pass<'_>; 2] = [&with_extra, &without];
    let together = frame(&queue, &passes, Apart::Together);
    let alone = frame(&queue, &passes, Apart::Queues);
    // The second pass's draw took every step and keeps itself, which the read back lets go.
    assert_eq!(together.carries(), [1, 0, 1, 0, 0, 1, 0, 0], "the counts");
    // Each pass's first draw read its four stage tables whole (`stage_tables_read_whole_*`): the
    // first pass's from the state the encoder before it left (a spare), the second's from the
    // state the first carried in. In queues of their own, the first pass's from a spare too, and
    // the second's from a new state that never read its tables: the first's state went with its
    // carry to its queue's place, which no encoder of the second's queue takes.
    assert_eq!(
        together.read_whole(),
        [0, 0, 0, 0, 0, 4, 4, 0],
        "read whole"
    );
    assert_eq!(
        alone.read_whole(),
        [4, 0, 0, 0, 0, 4, 0, 0],
        "read whole apart"
    );
    assert_eq!(together.words, alone.words, "the pixels");
    let pixel = channels(together.words[0]);
    assert!(
        pixel.iter().all(|&c| c >= 30),
        "both passes' colors: {pixel:?}"
    );
}

/// The next encoder binds every slot its draw reads but leaves unbound a texture slot the encoder
/// before bound (fragment texture 1, which no function reads): the carried draw's bind state holds
/// no texture slot its state does not read (`argument_table::ReadSlots`), so nothing in it is
/// stale and the carried draw is rebound, with the pixels of the passes in command buffers of
/// queues of their own.
#[test]
fn a_texture_slot_no_function_reads_left_unbound_keeps_the_carry() {
    let scratch = Scratch::new("carried-unread");
    let heard = Heard::default();
    let device = plan_device(&scratch.0, &[], &heard);
    let queue = device.new_command_queue().unwrap();
    let state = adding_state(&device);
    let color = colors(&device, &[[1.0 / 16.0; 4]]);
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let unread = device.new_texture(&descriptor).unwrap();
    let with_extra = |encoder: &mut RenderCommandEncoder| {
        (encoder.set_fragment_texture(Some(&unread), 1)).unwrap();
        rectangle(encoder, &state, (&color, 0));
    };
    let without = |encoder: &mut RenderCommandEncoder| rectangle(encoder, &state, (&color, 0));
    warm(&device, with_extra);
    let passes: [Pass<'_>; 2] = [&with_extra, &without];
    let together = frame(&queue, &passes, Apart::Together);
    let alone = frame(&queue, &passes, Apart::Queues);
    assert_eq!(together.carries(), rebound(1), "the counts");
    assert_eq!(together.words, alone.words, "the pixels");
    let pixel = channels(together.words[0]);
    assert!(
        pixel.iter().all(|&c| c >= 30),
        "both passes' colors: {pixel:?}"
    );
}

/// A state of `tob_rect` (a rectangle from vertex bytes 0's corners) and `fragment` of the owned
/// fixture `engine/tob_renders.ll` into one `RGBA8Unorm` color.
fn rect_state(device: &Device, fragment: &str) -> RenderPipelineState {
    let renders = library(device, "engine/tob_renders.ll");
    let function = |name| {
        let function = renders.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(function(fragment));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    (device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())).unwrap()
}

/// The corners, in normalized device coordinates, of the target's pixels `[x0, x1) x [y0, y1)`.
fn corners([x0, y0, x1, y1]: [u64; 4]) -> Vec<u8> {
    let x = |p: u64| p as f32 / SIDE as f32 * 2.0 - 1.0;
    let y = |p: u64| 1.0 - p as f32 / SIDE as f32 * 2.0;
    bytes_of(&[x(x0), y(y1), x(x1), y(y0)], f32::to_le_bytes)
}

/// The target's quarters and its left half, `[x0, y0, x1, y1]`.
const HALF: u64 = SIDE / 2;
const TOP_LEFT: [u64; 4] = [0, 0, HALF, HALF];
const TOP_RIGHT: [u64; 4] = [HALF, 0, SIDE, HALF];
const BOTTOM_LEFT: [u64; 4] = [0, HALF, HALF, SIDE];
const BOTTOM_RIGHT: [u64; 4] = [HALF, HALF, SIDE, SIDE];
const LEFT: [u64; 4] = [0, 0, HALF, SIDE];

/// The colors the solid draws write, and the clear: each channel 0 or 1, so each is its bytes.
const SOLID: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0, 1.0],
];
const CLEAR: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

/// The word of `color` in an `RGBA8Unorm` texel.
fn word_of(color: [f32; 4]) -> u32 {
    u32::from_le_bytes(color.map(|channel| (channel * 255.0) as u8))
}

/// The backdrop frame (the test's docs) over a new target that shaders read, its passes encoded as
/// `apart` says, each a draw of `read` (reading the
/// target at each pixel of a rectangle) and of `solid` (a color of `colors` over another), the
/// first only of `solid`, clearing the target: what it drew.
fn backdrop_frame(
    queue: &CommandQueue,
    (solid, read): (&RenderPipelineState, &RenderPipelineState),
    colors: &[(&Buffer, u64); 4],
    apart: Apart,
) -> Frame {
    let device = queue.device();
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let target = device.new_texture(&descriptor).unwrap();
    let back = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    // Each pass's read rectangle (none in the first), and its solid one.
    let passes = [
        (None, TOP_LEFT),
        (Some(TOP_RIGHT), BOTTOM_RIGHT),
        (Some(BOTTOM_RIGHT), BOTTOM_LEFT),
        (Some(LEFT), TOP_RIGHT),
    ];
    let before = device.work_counts();
    let mut queue = queue.clone();
    let mut command_buffer = apart.first(&mut queue);
    for (at, (reads, solid_over)) in passes.into_iter().enumerate() {
        if at > 0 {
            apart.between(&mut queue, &mut command_buffer);
        }
        let mut pass = pass_over((&target, 0), at == 0);
        let [r, g, b, a] = CLEAR.map(f64::from);
        pass.color_attachments[0].clear_color = ClearColor::new(r, g, b, a);
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        if let Some(reads) = reads {
            encoder.set_render_pipeline_state(read).unwrap();
            encoder.set_vertex_bytes(&corners(reads), 0).unwrap();
            (encoder.set_fragment_texture(Some(&target), 0)).unwrap();
            let drawn = encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0);
            assert_eq!(drawn, Ok(()));
        }
        encoder.set_render_pipeline_state(solid).unwrap();
        encoder.set_vertex_bytes(&corners(solid_over), 0).unwrap();
        let (color, offset) = colors[at];
        (encoder.set_fragment_buffer(Some(color), offset, 0)).unwrap();
        let drawn = encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0);
        assert_eq!(drawn, Ok(()));
        encoder.end_encoding().unwrap();
    }
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        SIDE * 4,
        SIDE * SIDE * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    Frame {
        words: words_of(&back),
        before,
        after: device.work_counts(),
        left_out: Vec::new(),
    }
}

/// Passes over one target loaded and stored, each after the first drawing a backdrop (`tob_read`,
/// which reads its pass's color attachment's texture at the fragment's own pixel, through the
/// texture's copy taken before the pass: `texture/attached_reads.rs`) over a rectangle, then a
/// solid color over another (`tob_solid`). The backdrop's kept draw carries into the next pass
/// (`render/carried.rs`), whose backdrop is rebound against it and must read that pass's copy,
/// which the pass before changed: each backdrop reads the solid color the pass before drew (the
/// bottom right quarter, then the bottom left one), which a copy from before that pass does not
/// hold, and writes it back, so the target ends with each pass's solid color in place. The same
/// frame in command buffers of its own, of one queue, carries each pass's kept draws across and
/// counts and draws the same; in command buffers of queues of their own, which carries nothing, it
/// draws the same pixels, with as many reads through copies.
#[test]
fn a_carried_backdrop_reads_its_own_pass_s_copy() {
    let scratch = Scratch::new("carried-backdrop");
    let heard = Heard::default();
    let device = plan_device(&scratch.0, &[], &heard);
    let queue = device.new_command_queue().unwrap();
    let solid = rect_state(&device, "tob_solid");
    let read = rect_state(&device, "tob_read");
    let firsts = colors(&device, &SOLID[..3]);
    let last = colors(&device, &SOLID[3..]);
    let colors = [
        (&firsts, 0),
        (&firsts, SECOND),
        (&firsts, 2 * SECOND),
        (&last, 0),
    ];
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    let sampled = device.new_texture(&descriptor).unwrap();
    warm(&device, |encoder| {
        rectangle(encoder, &solid, colors[0]);
        encoder.set_render_pipeline_state(&read).unwrap();
        encoder.set_vertex_bytes(&corners(TOP_LEFT), 0).unwrap();
        (encoder.set_fragment_texture(Some(&sampled), 0)).unwrap();
        let drawn = encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0);
        assert_eq!(drawn, Ok(()));
    });
    let states = (&solid, &read);
    let together = backdrop_frame(&queue, states, &colors, Apart::Together);
    let across = backdrop_frame(&queue, states, &colors, Apart::CommandBuffers);
    let alone = backdrop_frame(&queue, states, &colors, Apart::Queues);
    // Three passes opened from a carry; five carried draws rebound (three solid, two backdrops),
    // the backdrop's kept draw carried aside twice.
    let mut expected = rebound(5);
    expected[0] = 3;
    assert_eq!(together.carries(), expected, "the counts");
    assert_eq!(across.carries(), expected, "across: the counts");
    assert_eq!(across.across(), 3, "carried across");
    let aside = |frame: &Frame| frame.counted(|c| c.kept_draws_carried_aside);
    assert_eq!((aside(&together), aside(&across)), (2, 2), "carried aside");
    // Every step: the first pass's solid draw and the second's backdrop together and across;
    // each pass's first draw, and the second's and later passes' second, alone.
    let none_kept = |frame: &Frame| frame.counted(|c| c.draws_none_kept);
    assert_eq!(
        [&together, &across, &alone].map(none_kept),
        [2, 2, 7],
        "none kept"
    );
    let through = |frame: &Frame| frame.counted(|c| c.attachment_reads_through_copies);
    assert_eq!(
        [&together, &across, &alone].map(through),
        [3, 3, 3],
        "reads through copies"
    );
    assert_eq!(together.words, alone.words, "the pixels");
    assert_eq!(across.words, alone.words, "across: the pixels");
    let quarter = |[x0, y0, x1, y1]: [u64; 4], color: [f32; 4]| {
        let mut texels = (y0..y1).flat_map(|y| (x0..x1).map(move |x| (y * SIDE + x) as usize));
        let wrong = texels.find(|&at| together.words[at] != word_of(color));
        assert_eq!(wrong, None, "{color:?} over {:?}", [x0, y0, x1, y1]);
    };
    quarter(TOP_LEFT, SOLID[0]);
    quarter(BOTTOM_RIGHT, SOLID[1]);
    quarter(BOTTOM_LEFT, SOLID[2]);
    quarter(TOP_RIGHT, SOLID[3]);
}

/// The side of the targets of [`a_carried_copy_reader_stops_where_every_step_snapshots`]: two
/// tiles of 32 each way, its rows of 256 bytes over a buffer.
const PAGES_SIDE: u64 = 64;

/// A pass over a private target, then one over a texture over a buffer's bytes, each in a command
/// buffer of its own of one queue, each one draw of `tob_read` over the whole target through a
/// scissor of one tile's quarter, which reads the target at the fragment's own pixel. The first
/// reads it through the texture's copy taken before the pass (`texture/attached_reads.rs`, its
/// area too small for a snapshot), and keeps its draw, which the queue holds for the next command
/// buffer's pass (`kept_draws_carried_across`). Over a buffer's bytes, every step reads through a
/// snapshot whatever the area (`render/snapshot.rs`) where the device reads so
/// (`shadow_standing_device::reads_through_snapshots`): the carried draw's rebind, which reads
/// through the copy, stops (`draws_carry_stopped_snapshot`), and the draw takes every step,
/// counted as a read through a snapshot (`reading_draws_snapshotted`). The device's in-place
/// checks settle before the passes: a check running beside them changes what the buffer lies
/// under (`Under`), which lets the carry go stale first.
#[test]
fn a_carried_copy_reader_stops_where_every_step_snapshots() {
    let scratch = Scratch::new("carried-copy-reader");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let host = Host::new();
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(buffer) = over_host(&device, &host, &runs) else {
        return;
    };
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, PAGES_SIDE, PAGES_SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let over_pages = (buffer.new_texture(&descriptor, 0, PAGES_SIDE * 4)).unwrap();
    if !reads_through_snapshots(&device, &over_pages) {
        eprintln!(
            "CARRIED-COPY-READER on {}: no snapshot over a buffer's bytes",
            device.name()
        );
        return;
    }
    device.settle_in_place_checks(WAIT);
    descriptor.resource_options = ResourceOptions::private();
    let private = device.new_texture(&descriptor).unwrap();
    let read = rect_state(&device, "tob_read");
    let queue = device.new_command_queue().unwrap();
    let pass = |target: &Texture| {
        let pass = pass_over((target, 0), false);
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&read).unwrap();
        let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
        encoder.set_vertex_bytes(&corners, 0).unwrap();
        (encoder.set_fragment_texture(Some(target), 0)).unwrap();
        let scissor = ScissorRect {
            x: 8,
            y: 8,
            width: 8,
            height: 8,
        };
        encoder.set_scissor_rect(scissor).unwrap();
        let drawn = encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0);
        assert_eq!(drawn, Ok(()));
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer
    };
    let counted = |before: &WorkCounts, after: &WorkCounts| {
        let frame = Frame {
            words: Vec::new(),
            before: *before,
            after: *after,
            left_out: Vec::new(),
        };
        let reads = [
            frame.counted(|c| c.attachment_reads_through_copies),
            frame.counted(|c| c.reading_draws_snapshotted),
        ];
        let carry = [
            frame.across(),
            frame.counted(|c| c.draws_rebound_carried),
            frame.counted(|c| c.draws_carry_stopped_snapshot),
            frame.counted(|c| c.draws_carry_stopped),
        ];
        (reads, carry)
    };
    let before = device.work_counts();
    let first = pass(&private);
    let between = device.work_counts();
    let second = pass(&over_pages);
    for command_buffer in [first, second] {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }
    let after = device.work_counts();
    assert_eq!(
        counted(&before, &between),
        ([1, 0], [0, 0, 0, 0]),
        "the private pass: reads through copies and snapshots; carried across, rebound, stopped \
         at the snapshot, stopped"
    );
    assert_eq!(
        counted(&between, &after),
        ([0, 1], [1, 0, 1, 1]),
        "the pass over a buffer's bytes"
    );
}

/// The rectangle's indices: both its triangles, and its second alone, twice.
const BOTH_TRIANGLES: [u16; 6] = [0, 1, 2, 3, 4, 5];
const SECOND_TRIANGLE: [u16; 6] = [3, 4, 5, 3, 4, 5];

/// Four command buffers, each one pass over one target with one indexed draw of the rectangle
/// over the whole target, in its color of [`SOLID`], on `queue` where `apart` is
/// [`Apart::CommandBuffers`], each on a queue of its own where it is [`Apart::Queues`] (the
/// test's docs): the first of both triangles, from indices in `host`'s pages, which the program
/// releases after the first command buffer's commit and which is freed (its deallocator run)
/// before the second command buffer's pass opens; the second and third of the second triangle,
/// from a buffer of the program's; the fourth likewise through a pipeline state made again after
/// the first one was released and its last command buffer completed. What it drew, `None` where
/// the device imports no host memory.
fn released_frame(queue: &CommandQueue, host: &Host, apart: Apart) -> Option<Frame> {
    let device = queue.device();
    let mut solid = Some(rect_state(device, "tob_solid"));
    let firsts = colors(device, &SOLID[..3]);
    let last = colors(device, &SOLID[3..]);
    let colors = [
        (&firsts, 0),
        (&firsts, SECOND),
        (&firsts, 2 * SECOND),
        (&last, 0),
    ];
    let pairs = BOTH_TRIANGLES.as_chunks::<2>().0.iter();
    let words: Vec<u32> = pairs
        .map(|&[a, b]| u32::from(a) | u32::from(b) << 16)
        .collect();
    host.set(0, &words);
    let freed = Arc::new(AtomicUsize::new(0));
    let mut both = Some(over_host(device, host, &freed)?);
    let second = bytes_of(&SECOND_TRIANGLE, u16::to_le_bytes);
    let second = (device.new_buffer_with_bytes(&second, ResourceOptions::shared())).unwrap();
    let draw = |encoder: &mut RenderCommandEncoder,
                state: &RenderPipelineState,
                indices: &Buffer,
                at: usize| {
        encoder.set_render_pipeline_state(state).unwrap();
        encoder
            .set_vertex_bytes(&corners([0, 0, SIDE, SIDE]), 0)
            .unwrap();
        let (color, offset) = colors[at];
        (encoder.set_fragment_buffer(Some(color), offset, 0)).unwrap();
        let drawn = encoder.draw_indexed_primitives(
            PrimitiveType::Triangle,
            BOTH_TRIANGLES.len() as u32,
            IndexType::UInt16,
            indices,
            0,
            1,
            0,
            0,
        );
        assert_eq!(drawn, Ok(()));
    };
    let target = target(device);
    let back = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let before = device.work_counts();
    let mut left_out = Vec::new();
    let mut queue = queue.clone();
    let mut command_buffer = apart.first(&mut queue);
    for at in 0..colors.len() {
        if at > 0 {
            let committed = command_buffer.clone();
            apart.between(&mut queue, &mut command_buffer);
            match at {
                // Released after the commit that left the carry: freed once its command buffer
                // completed, the carry its queue holds naming it by key and handle alone.
                1 => {
                    drop(both.take());
                    let freed = || freed.load(Ordering::SeqCst) == 1;
                    until("the released index buffer freed", freed);
                }
                // Released, and made again once nothing holds the one released, its own pipeline
                // made by runs on queues of their own, whose counts are left out.
                3 => {
                    drop(solid.take());
                    committed.wait_until_completed_for(WAIT).unwrap();
                    let again = rect_state(device, "tob_solid");
                    let warming = device.work_counts();
                    warm(device, |encoder| draw(encoder, &again, &second, at));
                    left_out.push((warming, device.work_counts()));
                    solid = Some(again);
                }
                _ => {}
            }
        }
        let pass = pass_over((&target, 0), at == 0);
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        let state = solid.as_ref().unwrap();
        draw(&mut encoder, state, both.as_ref().unwrap_or(&second), at);
        encoder.end_encoding().unwrap();
    }
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        SIDE * 4,
        SIDE * SIDE * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    Some(Frame {
        words: words_of(&back),
        before,
        after: device.work_counts(),
        left_out,
    })
}

/// A carry its queue holds from a commit (`render/carried.rs`) names the program's objects by key
/// and handle alone: an index buffer the program releases after the commit that left the carry is
/// freed once that command buffer completed, before the next command buffer's pass opens (its
/// deallocator run), and that pass's draw, of another index buffer, opens from the carry
/// (`kept_draws_carried_across`) and draws that buffer's indices, as the next one's does again,
/// rebound. With the pipeline state released and made again before the fourth command buffer, its
/// pass opens from the carry too, but its draw finds no kept draw of its state
/// (`draws_none_kept`) and takes every step. The pixels are those of the same command buffers
/// each on a queue of its own, which carries nothing: the first triangle of the first draw's
/// color, the second of the last draw's.
#[test]
fn a_carry_its_queue_holds_keeps_nothing_released_alive() {
    let host = Host::new();
    let scratch = Scratch::new("carried-released");
    let heard = Heard::default();
    let device = plan_device(&scratch.0, &[], &heard);
    let queue = device.new_command_queue().unwrap();
    let Some(across) = released_frame(&queue, &host, Apart::CommandBuffers) else {
        return;
    };
    let alone = released_frame(&queue, &host, Apart::Queues).unwrap();
    // Three passes opened from the carry their queue held, and the second's and third's draws
    // rebound (the second's checked against every step in a build with debug assertions); the
    // fourth's draw, of another pipeline state, rebinds nothing.
    let rebound = match cfg!(debug_assertions) {
        true => 1,
        false => 2,
    };
    assert_eq!(
        across.carries(),
        [3, rebound, 0, 0, 0, 1, 0, 0],
        "the counts"
    );
    assert_eq!(across.across(), 3, "carried across");
    let none_kept = |frame: &Frame| frame.counted(|c| c.draws_none_kept);
    assert_eq!((none_kept(&across), none_kept(&alone)), (2, 4), "none kept");
    assert_eq!(across.words, alone.words, "the pixels");
    let (first, last) = (word_of(SOLID[0]), word_of(SOLID[3]));
    let count = |word| across.words.iter().filter(|&&at| at == word).count();
    assert_eq!(
        count(first) + count(last),
        across.words.len(),
        "the first and last draws' colors alone"
    );
    assert!(count(first) > 0 && count(last) > 0, "both triangles");
}

/// Four passes over a new target, the first cleared, each one draw of `draw` (given the pass's
/// place), in four command buffers. Where `side_by_side`, of one new queue, made two by two, each
/// two encoded before either commits, then committed in order (as a program's threads encode
/// theirs and commit them): the first two open with no carry (their queue held none), and each of
/// the last two opens from one its queue holds from the first two's commits (`queue::CARRIES`).
/// Else each of a queue of its own, committed and completed before the next is made, which carries
/// nothing. `between` runs once the first two committed, given the first. What the frame drew.
fn side_by_side_frame(
    device: &Device,
    side_by_side: bool,
    draw: &dyn Fn(&mut RenderCommandEncoder, usize),
    between: &mut dyn FnMut(&CommandBuffer),
) -> Frame {
    let target = target(device);
    let back = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let before = device.work_counts();
    let queue = device.new_command_queue().unwrap();
    for pair in [0, 2] {
        let mut encoded = Vec::new();
        for at in pair..pair + 2 {
            let queue = match side_by_side {
                true => queue.clone(),
                false => device.new_command_queue().unwrap(),
            };
            let command_buffer = queue.command_buffer().unwrap();
            let pass = pass_over((&target, 0), at == 0);
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            draw(&mut encoder, at);
            encoder.end_encoding().unwrap();
            if !side_by_side {
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
            }
            encoded.push(command_buffer);
        }
        for command_buffer in encoded.iter().filter(|_| side_by_side) {
            command_buffer.commit().unwrap();
        }
        if pair == 0 {
            between(&encoded[0]);
        }
    }
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        SIDE * 4,
        SIDE * SIDE * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    Frame {
        words: words_of(&back),
        before,
        after: device.work_counts(),
        left_out: Vec::new(),
    }
}

/// The draws of a frame that took every step as no kept draw of their state was kept
/// (`draws_none_kept`), and of them those whose encoder opened from no carry
/// (`draws_none_kept_uncarried`).
fn none_kept_uncarried(frame: &Frame) -> [u64; 2] {
    [
        frame.counted(|c| c.draws_none_kept),
        frame.counted(|c| c.draws_none_kept_uncarried),
    ]
}

/// Command buffers of one queue encoded side by side (`side_by_side_frame`): the queue holds the
/// carries of the first two from their commits, the latest first (`queue::CARRIES`), so each of
/// the next two opens from one (`kept_draws_carried_across`) and none is let go at a commit
/// (`kept_draws_carry_dropped_commit`); only the first two draws, of encoders no carry reached,
/// take every step (`draws_none_kept_uncarried`). The pixels are those of the same passes on
/// queues of their own, which carry nothing and whose every draw takes every step, each pass's
/// color added.
#[test]
fn command_buffers_encoded_side_by_side_each_open_from_a_carry_their_queue_holds() {
    let scratch = Scratch::new("carried-side-by-side");
    let heard = Heard::default();
    let device = plan_device(&scratch.0, &[], &heard);
    let state = adding_state(&device);
    let color = colors(&device, &[[1.0 / 16.0; 4]]);
    let draw = |encoder: &mut RenderCommandEncoder, _: usize| {
        rectangle(encoder, &state, (&color, 0));
    };
    warm(&device, |encoder| draw(encoder, 0));
    let side_by_side = side_by_side_frame(&device, true, &draw, &mut |_| {});
    let alone = side_by_side_frame(&device, false, &draw, &mut |_| {});
    // Each carried draw is the first of its carry's, which a build with debug assertions checks
    // against every step (the module docs).
    let rebound = match cfg!(debug_assertions) {
        true => 0,
        false => 2,
    };
    let counts = side_by_side.carries();
    assert_eq!(
        counts,
        [2, rebound, 0, 0, 0, 0, 0, 0],
        "side by side: the counts"
    );
    assert_eq!(side_by_side.across(), 2, "both carried across");
    let dropped = |frame: &Frame| frame.counted(|c| c.kept_draws_carry_dropped_commit);
    assert_eq!(dropped(&side_by_side), 0, "none let go at a commit");
    assert_eq!(
        none_kept_uncarried(&side_by_side),
        [2, 2],
        "side by side: every step, the first two draws"
    );
    assert_eq!(none_kept_uncarried(&alone), [4, 4], "alone: every draw");
    assert_eq!(side_by_side.words, alone.words, "the pixels");
    let pixel = channels(alone.words[0]);
    assert!(
        pixel.iter().all(|&c| c >= 15 * 4),
        "every pass's color added: {pixel:?}"
    );
}

/// Command buffers of one queue encoded side by side (`side_by_side_frame`), the first's draw of
/// both triangles of the rectangle from indices in a host page's buffer (no copy), the others' of
/// its second triangle from a buffer of the program's, each in its color of [`SOLID`]. The
/// program releases the host page's buffer once the first two committed: it is freed (its
/// deallocator run) once the first completed, while its queue holds that command buffer's carry,
/// which names it by key and handle alone, beside the second's. The third pass opens from the
/// second's carry and the fourth from the first's, whose draw takes the program's buffer's
/// indices, rebound. The pixels are those of the same command buffers each on a queue of its own:
/// the first triangle of the first draw's color, the second of the last's.
#[test]
fn a_carry_held_beside_another_keeps_no_released_buffer_alive() {
    let host = Host::new();
    let scratch = Scratch::new("carried-side-by-side-released");
    let heard = Heard::default();
    let device = plan_device(&scratch.0, &[], &heard);
    let state = rect_state(&device, "tob_solid");
    let firsts = colors(&device, &SOLID[..3]);
    let last = colors(&device, &SOLID[3..]);
    let colors = [
        (&firsts, 0),
        (&firsts, SECOND),
        (&firsts, 2 * SECOND),
        (&last, 0),
    ];
    let pairs = BOTH_TRIANGLES.as_chunks::<2>().0.iter();
    let words: Vec<u32> = pairs
        .map(|&[a, b]| u32::from(a) | u32::from(b) << 16)
        .collect();
    host.set(0, &words);
    let second = bytes_of(&SECOND_TRIANGLE, u16::to_le_bytes);
    let second = (device.new_buffer_with_bytes(&second, ResourceOptions::shared())).unwrap();
    let both = RefCell::new(None);
    let draw = |encoder: &mut RenderCommandEncoder, at: usize| {
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder
            .set_vertex_bytes(&corners([0, 0, SIDE, SIDE]), 0)
            .unwrap();
        let (color, offset) = colors[at];
        (encoder.set_fragment_buffer(Some(color), offset, 0)).unwrap();
        let both = both.borrow();
        let indices = match at {
            0 => both.as_ref().unwrap(),
            _ => &second,
        };
        let drawn = encoder.draw_indexed_primitives(
            PrimitiveType::Triangle,
            BOTH_TRIANGLES.len() as u32,
            IndexType::UInt16,
            indices,
            0,
            1,
            0,
            0,
        );
        assert_eq!(drawn, Ok(()));
    };
    let Some(made) = over_host(&device, &host, &Arc::default()) else {
        return;
    };
    *both.borrow_mut() = Some(made);
    warm(&device, |encoder| draw(encoder, 0));
    warm(&device, |encoder| draw(encoder, 1));
    // Each frame's host page buffer made anew, its own deallocator counted, the one before it
    // released first.
    let frame = |side_by_side: bool| {
        drop(both.borrow_mut().take());
        let freed = Arc::new(AtomicUsize::new(0));
        let made = over_host(&device, &host, &freed).unwrap();
        *both.borrow_mut() = Some(made);
        let mut release = |first: &CommandBuffer| {
            drop(both.borrow_mut().take());
            first.wait_until_completed_for(WAIT).unwrap();
            let freed = || freed.load(Ordering::SeqCst) == 1;
            until("the released index buffer freed, its carry held", freed);
        };
        side_by_side_frame(&device, side_by_side, &draw, &mut release)
    };
    let side_by_side = frame(true);
    let alone = frame(false);
    assert_eq!(side_by_side.across(), 2, "both carried across");
    let rebound = match cfg!(debug_assertions) {
        true => 0,
        false => 2,
    };
    let counts = side_by_side.carries();
    assert_eq!(
        counts,
        [2, rebound, 0, 0, 0, 0, 0, 0],
        "side by side: the counts"
    );
    assert_eq!(side_by_side.words, alone.words, "the pixels");
    let (first, last) = (word_of(SOLID[0]), word_of(SOLID[3]));
    let count = |word| side_by_side.words.iter().filter(|&&at| at == word).count();
    assert_eq!(
        count(first) + count(last),
        side_by_side.words.len(),
        "the first and last draws' colors alone"
    );
    assert!(count(first) > 0 && count(last) > 0, "both triangles");
}

/// A pass of another layout (an `R8Unorm` target) after a pass that carries its kept draw lets
/// it go, counted.
#[test]
fn a_pass_of_another_layout_drops_the_carry() {
    let scratch = Scratch::new("carried-layout");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let queue = device.new_command_queue().unwrap();
    let state = adding_state(&device);
    let color = colors(&device, &[[1.0 / 16.0; 4]]);
    let pass = |encoder: &mut RenderCommandEncoder| rectangle(encoder, &state, (&color, 0));
    warm(&device, pass);
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::R8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let narrow = device.new_texture(&descriptor).unwrap();
    let target = target(&device);
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder =
        (command_buffer.render_command_encoder(&pass_over((&target, 0), true))).unwrap();
    pass(&mut encoder);
    encoder.end_encoding().unwrap();
    let other = pass_over((&narrow, 0), true);
    let encoder = command_buffer.render_command_encoder(&other).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let ran = Frame {
        words: Vec::new(),
        before,
        after: device.work_counts(),
        left_out: Vec::new(),
    };
    assert_eq!(ran.carries(), [0, 0, 0, 0, 0, 0, 1, 0], "the counts");
}

/// Passes whose one draw is indirect, which keeps nothing (its use reads indirect arguments), end
/// carrying nothing as none kept (`encoders_uncarried_none_kept`), each counting what a kept draw
/// of it would have come to (`encoders_none_kept_next_*`): in one command buffer each but the last
/// fits the next pass, and the read back's blit encoder comes after the last; with a blit encoder
/// between each two, each is one between; in command buffers of their own, of one queue or of
/// queues of their own, each but the last ends its command buffer; and a pass of another layout
/// (an `R8Unorm` target) after one is of another layout. The pixels are the same each way.
#[test]
fn an_encoder_that_keeps_none_counts_what_its_carry_would_come_to() {
    let scratch = Scratch::new("carried-none-kept");
    let heard = Heard::default();
    let device = plan_device(&scratch.0, &[], &heard);
    let queue = device.new_command_queue().unwrap();
    let state = adding_state(&device);
    let color = colors(&device, &[[1.0 / 16.0; 4]]);
    // `MTLDrawPrimitivesIndirectArguments`: six vertices, one instance.
    let arguments = bytes_of(&[6u32, 1, 0, 0], u32::to_le_bytes);
    let arguments = (device.new_buffer_with_bytes(&arguments, ResourceOptions::shared())).unwrap();
    let pass = |encoder: &mut RenderCommandEncoder| {
        encoder.set_render_pipeline_state(&state).unwrap();
        let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
        encoder.set_vertex_bytes(&corners, 0).unwrap();
        (encoder.set_fragment_buffer(Some(&color), 0, 0)).unwrap();
        let drawn = encoder.draw_primitives_indirect(PrimitiveType::Triangle, &arguments, 0);
        assert_eq!(drawn, Ok(()));
    };
    warm(&device, pass);
    const PASSES: usize = 3;
    let passes: [Pass<'_>; PASSES] = [&pass; PASSES];
    let together = frame(&queue, &passes, Apart::Together);
    let blits = frame(&queue, &passes, Apart::Blits);
    let across = frame(&queue, &passes, Apart::CommandBuffers);
    let alone = frame(&queue, &passes, Apart::Queues);
    let (ended, next) = (PASSES as u64, PASSES as u64 - 1);
    let frames = [
        (&together, "together"),
        (&blits, "blits between"),
        (&across, "in command buffers of their own"),
        (&alone, "in command buffers of queues of their own"),
    ];
    for (frame, name) in frames {
        assert_eq!(frame.uncarried(), [0, ended, 0], "{name}");
        assert_eq!(frame.words, alone.words, "{name}: the pixels of the passes");
    }
    assert_eq!(
        together.none_kept_next(),
        [0, 0, 1, 0, 0, 0, next],
        "together"
    );
    assert_eq!(blits.none_kept_next(), [0, 0, ended, 0, 0, 0, 0], "blits");
    let apart = [0, 0, 1, next, 0, 0, 0];
    assert_eq!(across.none_kept_next(), apart, "in command buffers");
    assert_eq!(alone.none_kept_next(), apart, "in queues");
    // Each pass added its color, about 16 in each channel (as the device rounds each blend).
    let pixel = channels(together.words[0]);
    let least = 15 * PASSES as u8;
    assert!(
        pixel.iter().all(|&c| c >= least),
        "every pass's color added: {pixel:?}"
    );
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::R8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let narrow = device.new_texture(&descriptor).unwrap();
    let target = target(&device);
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder =
        (command_buffer.render_command_encoder(&pass_over((&target, 0), true))).unwrap();
    pass(&mut encoder);
    encoder.end_encoding().unwrap();
    let other = pass_over((&narrow, 0), true);
    let encoder = command_buffer.render_command_encoder(&other).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let ran = Frame {
        words: Vec::new(),
        before,
        after: device.work_counts(),
        left_out: Vec::new(),
    };
    assert_eq!(
        ran.none_kept_next(),
        [0, 0, 0, 0, 1, 0, 0],
        "another layout"
    );
}

/// The texture's levels, and every byte of level `l`'s texels.
const LEVELS: usize = 5;
const LEVEL_BYTES: [u8; LEVELS] = [0x10, 0x40, 0x80, 0xc0, 0xf0];

/// A pass's draw that samples a texture an attachment of its pass renders (`alv_level1`, which
/// reads level 1 of fragment texture 0, the pass rendering level 2 of it), after a pass of the
/// same state that samples it into another target: the carried draw's rebind stops (attached: its
/// slot is narrowed to level 1, which its kept draw's was not), and the draw takes every step,
/// reading the texture's level 1 into both targets.
#[test]
fn a_carried_draw_that_samples_its_attachment_stops() {
    let scratch = Scratch::new("carried-attached");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let queue = device.new_command_queue().unwrap();
    let levels = library(&device, "engine/alv_levels.ll");
    let function = |name| {
        let function = levels.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("alv_rect"));
    descriptor.fragment = Some(function("alv_level1"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let state =
        (device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())).unwrap();
    let mut sampler = SamplerDescriptor::new();
    sampler.mip_filter = SamplerMipFilter::Nearest;
    let sampler = device.new_sampler_state(&sampler).unwrap();
    let texture = mipmapped(&queue);
    let draw = |encoder: &mut RenderCommandEncoder, sampled: &Texture, sampler: &SamplerState| {
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder.set_fragment_texture(Some(sampled), 0).unwrap();
        (encoder.set_fragment_sampler_state(Some(sampler), 0)).unwrap();
        let drawn = encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0);
        assert_eq!(drawn, Ok(()));
    };
    warm(&device, |encoder| draw(encoder, &texture, &sampler));
    let target = target(&device);
    // The target's texels, then level 2's.
    let level_two = (SIDE >> 2) * (SIDE >> 2);
    let length = (SIDE * SIDE + level_two) * 4;
    let back = (device.new_buffer(length, ResourceOptions::shared())).unwrap();
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for (into, first) in [((&target, 0), true), ((&texture, 2), false)] {
        let pass = pass_over(into, first);
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        draw(&mut encoder, &texture, &sampler);
        encoder.end_encoding().unwrap();
    }
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let read = [(&target, 0, 0), (&texture, 2, SIDE * SIDE * 4)];
    for (from, level, at) in read {
        let side = SIDE >> level;
        blit.copy_from_texture_to_buffer(
            from,
            0,
            level,
            Origin::new(0, 0, 0),
            Size::new(side, side, 1),
            &back,
            at,
            side * 4,
            side * side * 4,
            BlitOption::None,
        )
        .unwrap();
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let ran = Frame {
        words: words_of(&back),
        before,
        after: device.work_counts(),
        left_out: Vec::new(),
    };
    assert_eq!(ran.carries(), [1, 0, 0, 1, 0, 0, 0, 0], "the counts");
    let (target_words, level_words) = ran.words.split_at((SIDE * SIDE) as usize);
    let one = u32::from_le_bytes([LEVEL_BYTES[1]; 4]);
    for (what, words) in [("the target", target_words), ("level 2", level_words)] {
        let wrong = words.iter().position(|&word| word != one);
        assert_eq!(wrong, None, "{what}: every texel level 1's color");
    }
}

/// A private `RGBA8Unorm` texture of [`SIDE`] square and [`LEVELS`] levels, each level `l` holding
/// [`LEVEL_BYTES`]`[l]` in every byte, filled on `queue`.
fn mipmapped(queue: &CommandQueue) -> Texture {
    let device = queue.device();
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, true);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let texture = device.new_texture(&descriptor).unwrap();
    assert_eq!(texture.mipmap_level_count(), LEVELS as u64);
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let mut sources = Vec::new();
    for (level, byte) in LEVEL_BYTES.iter().enumerate() {
        let side = SIDE >> level;
        let bytes = vec![*byte; (side * side * 4) as usize];
        let source = (device.new_buffer_with_bytes(&bytes, ResourceOptions::shared())).unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            side * 4,
            side * side * 4,
            Size::new(side, side, 1),
            &texture,
            0,
            level as u64,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
        sources.push(source);
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    texture
}
