//! A render encoder's kept draw carried into the next render encoder of its command buffer
//! (`render/carried.rs`): the next encoder's first draw of the same state is rebound against it
//! (`WorkCounts::draws_rebound_carried`), with the pixels of the same passes drawn every step, the
//! values its own calls bound; it stops where the bind state holds a slot the new encoder did not
//! bind (`draws_carry_stopped_stale`) or where the draw samples an attachment of its pass
//! (`draws_carry_stopped_attached`); and the carry is let go where a call comes between the
//! encoders (`kept_draws_carry_dropped_between`) or the next pass is of another layout
//! (`kept_draws_carry_dropped_layout`). In builds with debug assertions the first carried draw of
//! a command buffer, and every second one after it, takes every step and is checked against the
//! rebound (the module's check), so these tests run that check too.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixtures
//! `validation/fixtures/public/engine/tob_renders.ll` and `engine/alv_levels.ll` (`llvm-as`).

use super::blit_holds::{Scratch, WAIT};
use super::draw_again::{adding_state, device, warm, Heard};
use super::encoder_holds::{bytes_of, library, words_of};
use crate::{
    BlitOption, Buffer, ClearColor, CommandQueue, Device, LoadAction, Origin, PipelineOptions,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState,
    Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, SamplerMipFilter, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// The side of the target.
const SIDE: u64 = 16;

/// The offset of a buffer's second color: a constant buffer's offset alignment on every device.
const SECOND: u64 = 256;

/// The places for colors in a buffer of them.
const COLORS: usize = 3;

/// A pass's calls.
type Pass<'a> = &'a dyn Fn(&mut RenderCommandEncoder);

/// How a frame's passes are encoded: in one command buffer, one after another; in one command
/// buffer with an empty blit encoder between each two; or each in a command buffer of its own.
#[derive(Clone, Copy)]
enum Apart {
    Together,
    Blits,
    CommandBuffers,
}

/// What a frame drew, and the device's counts' change over it.
struct Frame {
    words: Vec<u32>,
    before: WorkCounts,
    after: WorkCounts,
}

impl Frame {
    /// How the count `of` changed over the frame.
    fn counted(&self, of: fn(&WorkCounts) -> u64) -> u64 {
        of(&self.after) - of(&self.before)
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
    let mut command_buffer = queue.command_buffer().unwrap();
    for (at, pass) in passes.iter().enumerate() {
        if at > 0 {
            match apart {
                Apart::Together => {}
                Apart::Blits => {
                    let blit = command_buffer.blit_command_encoder().unwrap();
                    blit.end_encoding().unwrap();
                }
                Apart::CommandBuffers => {
                    command_buffer.commit().unwrap();
                    command_buffer = queue.command_buffer().unwrap();
                }
            }
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
/// in command buffers of their own, which carry nothing. A blit encoder between each two lets the
/// carried draw go, counted, with the same pixels.
#[test]
fn passes_of_the_same_state_rebind_the_carried_draw() {
    let scratch = Scratch::new("carried-same");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let queue = device.new_command_queue().unwrap();
    let state = adding_state(&device);
    let color = colors(&device, &[[1.0 / 16.0; 4]]);
    let pass = |encoder: &mut RenderCommandEncoder| rectangle(encoder, &state, (&color, 0));
    warm(&device, pass);
    const PASSES: usize = 5;
    let passes: [Pass<'_>; PASSES] = [&pass; PASSES];
    let together = frame(&queue, &passes, Apart::Together);
    let alone = frame(&queue, &passes, Apart::CommandBuffers);
    let blits = frame(&queue, &passes, Apart::Blits);
    let carried = PASSES as u64 - 1;
    assert_eq!(together.carries(), rebound(carried), "together");
    assert_eq!(
        alone.carries(),
        dropped(0),
        "in command buffers of their own"
    );
    assert_eq!(blits.carries(), dropped(carried), "with blits between");
    assert_eq!(together.words, alone.words, "the pixels of the passes");
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

/// Passes that each draw through three pipeline states, each pass after the first starting from a
/// state its encoder's last draw did not draw: every state's kept draw carries with the encoder's
/// last one (`kept_draws_carried_aside`), and the next encoder's first draw of each state is
/// rebound against its own (`draws_rebound_carried`), the later ones after a draw of another
/// state began the pass's instance, with no draw unlike its kept draw or stopped, and the pixels
/// of the same passes in command buffers of their own. Passes of five states, more than an encoder
/// carries (four), let go of the kept draw of the state drawn longest before the last
/// (`kept_draws_carry_capped`), whose next draw takes every step; the other four are rebound.
#[test]
fn passes_of_several_states_rebind_each_state_s_carried_draw() {
    let scratch = Scratch::new("carried-states");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let queue = device.new_command_queue().unwrap();
    let states: [RenderPipelineState; 5] = std::array::from_fn(|_| adding_state(&device));
    let color = colors(&device, &[[1.0 / 64.0; 4]]);
    let draw = |encoder: &mut RenderCommandEncoder, order: &[usize]| {
        for &at in order {
            rectangle(encoder, &states[at], (&color, 0));
        }
    };
    let first = |encoder: &mut RenderCommandEncoder| draw(encoder, &[0, 1, 2]);
    let turned = |encoder: &mut RenderCommandEncoder| draw(encoder, &[1, 2, 0]);
    let five = |encoder: &mut RenderCommandEncoder| draw(encoder, &[0, 1, 2, 3, 4]);
    warm(&device, five);
    let aside = |frame: &Frame| frame.counted(|c| c.kept_draws_carried_aside);
    let capped = |frame: &Frame| frame.counted(|c| c.kept_draws_carry_capped);
    let missed = |c: &WorkCounts| c.draws_unlike_kept + c.draws_rebind_stopped;
    // Two carried passes of three states: each state's first draw rebound.
    let passes: [Pass<'_>; 3] = [&first, &turned, &turned];
    let together = frame(&queue, &passes, Apart::Together);
    let alone = frame(&queue, &passes, Apart::CommandBuffers);
    let mut expected = rebound(2 * 3);
    expected[0] = 2;
    assert_eq!(together.carries(), expected, "three states: the counts");
    assert_eq!(
        (aside(&together), capped(&together)),
        (2 * 2, 0),
        "three states"
    );
    assert_eq!(together.counted(missed), 0, "three states: no draw missed");
    assert_eq!(together.words, alone.words, "three states: the pixels");
    // A carried pass of five states: four rebound, the fifth (drawn first, longest before the
    // last) let go at each pass's end.
    let passes: [Pass<'_>; 2] = [&five, &five];
    let together = frame(&queue, &passes, Apart::Together);
    let alone = frame(&queue, &passes, Apart::CommandBuffers);
    let mut expected = rebound(4);
    expected[0] = 1;
    assert_eq!(together.carries(), expected, "five states: the counts");
    assert_eq!((aside(&together), capped(&together)), (3, 2), "five states");
    assert_eq!(together.words, alone.words, "five states: the pixels");
}

/// The next encoder binds the same state and buffers but for one slot's value: the same buffer at
/// another offset, or another buffer. Its carried draw is rebound with the value its own call
/// bound, which the pixels show: each pass adds the color it bound, in a channel of its own. Each
/// form is the first carried draw of one frame (checked against every step in a build with debug
/// assertions) and the second of another.
#[test]
fn a_carried_draw_rebinds_the_values_its_encoder_bound() {
    let scratch = Scratch::new("carried-values");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
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
        let alone = frame(&queue, &passes, Apart::CommandBuffers);
        assert_eq!(together.carries(), rebound(2), "{case}: the counts");
        assert_eq!(together.words, alone.words, "{case}: the pixels");
        // Red, green and blue each a quarter, the alpha about three quarters (as the device
        // rounds each blend): each pass's own color, none another's again.
        let [r, g, b, a] = channels(together.words[0]);
        let each = r > 0 && r == g && g == b && a > 2 * r;
        assert!(each, "{case}: each pass's color: {:?}", [r, g, b, a]);
    }
}

/// The next encoder binds every slot its draw reads but leaves unbound one the encoder before
/// bound (fragment buffer 1, which no function reads): the carried draw's bind state holds a slot
/// that encoder's calls did not bind, so its rebind stops (stale) and the draw takes every step,
/// with the pixels of the passes in command buffers of their own.
#[test]
fn a_slot_the_next_encoder_leaves_unbound_stops_the_carry() {
    let scratch = Scratch::new("carried-stale");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
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
    let alone = frame(&queue, &passes, Apart::CommandBuffers);
    // The second pass's draw took every step and keeps itself, which the read back lets go.
    assert_eq!(together.carries(), [1, 0, 1, 0, 0, 1, 0, 0], "the counts");
    assert_eq!(together.words, alone.words, "the pixels");
    let pixel = channels(together.words[0]);
    assert!(
        pixel.iter().all(|&c| c >= 30),
        "both passes' colors: {pixel:?}"
    );
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
    };
    assert_eq!(ran.carries(), [0, 0, 0, 0, 0, 0, 1, 0], "the counts");
}

/// The texture's levels, and every byte of level `l`'s texels.
const LEVELS: usize = 5;
const LEVEL_BYTES: [u8; LEVELS] = [0x10, 0x40, 0x80, 0xc0, 0xf0];

/// A pass's draw that samples a texture an attachment of its pass renders (`alv_level1`, which
/// reads level 1 of fragment texture 0, the pass rendering level 2 of it), after a pass of the
/// same state that samples it into another target: the carried draw's rebind stops (attached), and
/// the draw takes every step, reading the texture's level 1 into both targets.
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
