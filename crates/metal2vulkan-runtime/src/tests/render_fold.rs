//! Render encoders in groups of encoders recorded into one `VkCommandBuffer` (`encoder_fold.rs`,
//! **Render encoders**): one whose pass records at its end joins the group before it, one whose
//! first instance records inline only begins one, each after one barrier from everything of the
//! group's earlier encoders, and what a pass leaves for later ends the group after it; one whose
//! first instance records inline over what the render pass instance the group left open renders
//! continues that instance (**Continued passes**). A render
//! encoder takes a group only with a log of its calls, which the device keeps where it has a
//! buffer over host pages: each lane but [`a_render_encoder_without_a_log_takes_no_group`] makes
//! one first.
//!
//! The hazards a missing barrier leaves are checked by what each encoder reads, a draw's colors
//! that the compute encoder before it wrote and a copy of the target the draw wrote; a CPU device
//! may run them in order without the barrier, so each lane also counts the barrier at each render
//! encoder's open (`WorkCounts::command_barriers`).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `scale_and_offset`
//! (`output[i] = input[i] * scale + i`), `colored_vertex` and `colored_fragment` of the owned
//! fixture `validation/fixtures/public/examples/runtime_examples.ll`.

use super::blit_holds::{options, Scratch, WAIT};
use super::dispatch_pending::state;
use super::encoder_holds::{render_state, words_of};
use super::rebind_behind_a_wait::{Page, BYTES};
use crate::pipeline::pool::held;
use crate::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferError, CommandBufferStatus,
    CommitFailure, ComputePipelineState, Device, EncoderKind, Error, EventRef, FailedEncoders,
    LoadAction, Origin, PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResolveTarget,
    ResourceOptions, Size, StoreAction, Submission, SubmissionCheck, Texture, TextureDescriptor,
    WorkCounts,
};
use metal2vulkan::device::RenderLoweringPolicy;
use metal2vulkan::metal::{DispatchType, PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::collections::BTreeSet;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

/// The target's side, in texels.
const SIDE: u64 = 16;

/// The color the draws take, and the bytes an `RGBA8Unorm` texel stores of it.
const DRAWN: [f32; 4] = [1.0, 0.2, 0.0, 1.0];
const DRAWN_TEXEL: [u8; 4] = [255, 51, 0, 255];

/// The quad's vertices, and the floats of their colors.
const VERTICES: u32 = 6;
const COLOR_WORDS: u64 = VERTICES as u64 * 4;

/// The steps of a chain.
const STEPS: u64 = 6;

/// The fixture's functions and the buffers every lane draws and dispatches with, on a device with
/// a buffer over host pages, so it logs its render encoders' calls.
struct Lane {
    device: Device,
    compute: ComputePipelineState,
    render: RenderPipelineState,
    scale: Buffer,
    /// Two triangles over the whole target.
    positions: Buffer,
    /// The input from which `scale_and_offset` writes [`DRAWN`] for every vertex.
    from: Buffer,
    /// [`DRAWN`] for every vertex, as the CPU wrote it.
    colors: Buffer,
    /// The buffer over host pages; `None` on a lane that makes none.
    _host: Option<Buffer>,
    /// The host pages, dropped after the buffer over them and the device.
    _page: Page,
}

impl Lane {
    /// A lane on a device of `options`, with a buffer over host pages where `host`; `None`,
    /// printed, where the device imports no host memory.
    fn new(scratch: &Scratch, lowering: RenderLoweringPolicy, host: bool) -> Option<Lane> {
        let mut options = options(&scratch.0);
        options.policy.render_lowering = lowering;
        let device = Device::system_default_with(&options).unwrap();
        let page = Page::new();
        let host = match host {
            false => None,
            // SAFETY: the page outlives the buffer (dropped after it and the device), and no
            // command uses it.
            true => match unsafe {
                device.new_buffer_with_bytes_no_copy(
                    page.memory(),
                    BYTES as u64,
                    ResourceOptions::shared(),
                    None,
                )
            } {
                Ok(buffer) => Some(buffer),
                Err(Error::Unsupported(gap)) => {
                    eprintln!("RENDER-FOLD: the device imports no host memory: {gap:?}");
                    return None;
                }
                Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
            },
        };
        let corners: [[f32; 2]; 6] = [
            [-1.0, 1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, -1.0],
        ];
        let lane = Lane {
            compute: state(&device),
            render: render_state(
                &device,
                "examples/runtime_examples.ll",
                "colored_vertex",
                "colored_fragment",
            ),
            scale: floats(&device, [1.0]),
            positions: floats(&device, corners.into_iter().flatten()),
            from: floats(
                &device,
                (0..COLOR_WORDS).map(|i| DRAWN[(i % 4) as usize] - i as f32),
            ),
            colors: floats(&device, (0..VERTICES).flat_map(|_| DRAWN)),
            _host: host,
            _page: page,
            device,
        };
        Some(lane)
    }

    /// A lane on the default device, which lowers render calls at the call.
    fn at_call(scratch: &Scratch) -> Option<Lane> {
        Lane::new(scratch, RenderLoweringPolicy::AtCall, true)
    }

    /// A `Private` target, cleared to zero and warm: each pass shape the lanes draw, and a
    /// dispatch and a copy, ran once, so the chains below record each at once (no draw waits
    /// for its pipeline, which would leave work for later).
    fn target(&self) -> Texture {
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::private();
        let target = self.device.new_texture(&descriptor).unwrap();
        let queue = self.device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let colors = self.zeros(COLOR_WORDS);
        self.dispatch(&command_buffer, &colors);
        self.draw(&command_buffer, &target, LoadAction::DontCare, &colors);
        self.draw(&command_buffer, &target, LoadAction::Load, &colors);
        let back = self.back();
        self.copy_back(&command_buffer, &target, &back);
        run(&command_buffer);
        let command_buffer = queue.command_buffer().unwrap();
        let mut cleared = pass(&target, LoadAction::Clear);
        cleared.color_attachments[0].clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        let encoder = command_buffer.render_command_encoder(&cleared).unwrap();
        encoder.end_encoding().unwrap();
        run(&command_buffer);
        target
    }

    /// A `Shared` buffer of `words` zero floats.
    fn zeros(&self, words: u64) -> Buffer {
        floats(&self.device, (0..words).map(|_| 0.0))
    }

    /// A `Shared` buffer the target's texels are copied into.
    fn back(&self) -> Buffer {
        self.zeros(SIDE * SIDE)
    }

    /// A compute encoder that writes [`DRAWN`] for every vertex into `colors`, ended.
    fn dispatch(&self, command_buffer: &CommandBuffer, colors: &Buffer) {
        self.dispatch_of(command_buffer, DispatchType::Serial, colors);
    }

    /// [`dispatch`](Self::dispatch) in an encoder of `dispatch_type`.
    fn dispatch_of(
        &self,
        command_buffer: &CommandBuffer,
        dispatch_type: DispatchType,
        colors: &Buffer,
    ) {
        let mut encoder = command_buffer
            .compute_command_encoder_with_dispatch_type(dispatch_type)
            .unwrap();
        encoder.set_compute_pipeline_state(&self.compute).unwrap();
        encoder.set_buffer(Some(&self.from), 0, 0).unwrap();
        encoder.set_buffer(Some(colors), 0, 1).unwrap();
        encoder.set_buffer(Some(&self.scale), 0, 2).unwrap();
        encoder
            .dispatch_threads(Size::new(COLOR_WORDS, 1, 1), Size::new(COLOR_WORDS, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
    }

    /// A render encoder that draws the quad over `target`, loaded as `load`, with the vertices'
    /// colors from `colors`, ended: `DontCare` records its pass at its end, `Load` its first
    /// instance inline.
    fn draw(
        &self,
        command_buffer: &CommandBuffer,
        target: &Texture,
        load: LoadAction,
        colors: &Buffer,
    ) {
        let mut encoder = (command_buffer)
            .render_command_encoder(&pass(target, load))
            .unwrap();
        encoder.set_render_pipeline_state(&self.render).unwrap();
        encoder
            .set_vertex_buffer(Some(&self.positions), 0, 0)
            .unwrap();
        encoder.set_vertex_buffer(Some(colors), 0, 1).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, VERTICES, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
    }

    /// A blit encoder that copies `target`'s texels into `back`, ended.
    fn copy_back(&self, command_buffer: &CommandBuffer, target: &Texture, back: &Buffer) {
        let row = SIDE * 4;
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            target,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            back,
            0,
            row,
            row * SIDE,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
    }
}

/// A pass over `target`, loaded as `load` and stored.
fn pass(target: &Texture, load: LoadAction) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = load;
    color.store_action = StoreAction::Store;
    pass
}

/// A `Shared` buffer of `values`' bytes.
fn floats(device: &Device, values: impl IntoIterator<Item = f32>) -> Buffer {
    let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
    device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap()
}

/// Commit `command_buffer` and wait for it: it completes with no error.
fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        command_buffer.error()
    );
}

/// How many of the texels copied into `back` are not [`DRAWN_TEXEL`].
fn wrong_texels(back: &Buffer) -> usize {
    (words_of(back).into_iter())
        .filter(|word| word.to_le_bytes() != DRAWN_TEXEL)
        .count()
}

/// What moved of the counts the groups name.
#[derive(Debug, PartialEq, Eq)]
struct Moved {
    /// `encoders_folded`.
    folded: u64,
    /// `command_buffers_begun_encoders`.
    begun: u64,
    /// `command_buffers_begun_secondaries`.
    secondaries: u64,
    /// `command_barriers`.
    barriers: u64,
}

impl Moved {
    fn of(before: &WorkCounts, after: &WorkCounts) -> Moved {
        Moved {
            folded: after.encoders_folded - before.encoders_folded,
            begun: after.command_buffers_begun_encoders - before.command_buffers_begun_encoders,
            secondaries: after.command_buffers_begun_secondaries
                - before.command_buffers_begun_secondaries,
            barriers: after.command_barriers - before.command_barriers,
        }
    }
}

/// A chain of [`STEPS`] steps on `lane`, each a compute encoder that writes a draw's colors, a
/// render encoder that draws with them over a target of its own, its pass recorded at its end,
/// and a blit encoder that copies the target's texels back: what moved, and how many texels of
/// each step came back wrong.
fn chain(lane: &Lane) -> (Moved, Vec<usize>) {
    let queue = lane.device.new_command_queue().unwrap();
    let targets: Vec<Texture> = (0..STEPS).map(|_| lane.target()).collect();
    let colors: Vec<Buffer> = (0..STEPS).map(|_| lane.zeros(COLOR_WORDS)).collect();
    let backs: Vec<Buffer> = (0..STEPS).map(|_| lane.back()).collect();
    let before = lane.device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for ((target, colors), back) in targets.iter().zip(&colors).zip(&backs) {
        lane.dispatch(&command_buffer, colors);
        lane.draw(&command_buffer, target, LoadAction::DontCare, colors);
        lane.copy_back(&command_buffer, target, back);
    }
    run(&command_buffer);
    let moved = Moved::of(&before, &lane.device.work_counts());
    (moved, backs.iter().map(wrong_texels).collect())
}

/// Render encoders whose passes record at their ends join the group of the compute and blit
/// encoders around them: one `VkCommandBuffer` for the chain, and each draw's secondary. Each
/// draw reads the colors the compute encoder before it wrote, and each copy the texels the draw
/// before it wrote: each render encoder's open recorded one barrier, and each copy's one.
#[test]
fn render_encoders_whose_pass_records_at_its_end_join_their_group() {
    let scratch = Scratch::new("render-fold-chain");
    let Some(lane) = Lane::at_call(&scratch) else {
        return;
    };
    let (moved, wrong) = chain(&lane);
    println!("RENDER-FOLD-CHAIN {moved:?} wrong texels by step {wrong:?}");
    assert_eq!(wrong, vec![0; STEPS as usize], "every texel is the draw's");
    assert_eq!(
        (moved.folded, moved.begun, moved.secondaries),
        (3 * STEPS - 1, 1, STEPS),
        "every encoder after the first joined the first's group: {moved:?}"
    );
    assert!(
        moved.barriers >= 2 * STEPS,
        "each render encoder ordered after the group at its open, each copy after its draw: \
         {moved:?}"
    );
}

/// The same chain where the device lowers render calls after they return
/// (`RenderLoweringPolicy::Worker`): each render encoder's end, lowered on the device's lowering
/// thread or settled by the next encoder's open, takes the group as one lowered at its call does.
#[test]
fn render_encoders_lowered_after_their_calls_join_their_group() {
    let scratch = Scratch::new("render-fold-worker");
    let Some(lane) = Lane::new(&scratch, RenderLoweringPolicy::Worker, true) else {
        return;
    };
    let (moved, wrong) = chain(&lane);
    println!("RENDER-FOLD-WORKER {moved:?} wrong texels by step {wrong:?}");
    assert_eq!(wrong, vec![0; STEPS as usize], "every texel is the draw's");
    assert_eq!(
        (moved.folded, moved.begun, moved.secondaries),
        (3 * STEPS - 1, 1, STEPS),
        "{moved:?}"
    );
    assert!(moved.barriers >= 2 * STEPS, "{moved:?}");
}

/// A render encoder whose first instance records inline joins no group but begins one: the
/// copy and the compute encoder after it record into its `VkCommandBuffer`, and the next such
/// render encoder begins its own, which the copy after it joins.
#[test]
fn a_render_encoder_that_records_inline_begins_a_group_and_joins_none() {
    let scratch = Scratch::new("render-fold-inline");
    let Some(lane) = Lane::at_call(&scratch) else {
        return;
    };
    let queue = lane.device.new_command_queue().unwrap();
    let targets = [lane.target(), lane.target()];
    let backs = [lane.back(), lane.back()];
    let colors = lane.zeros(COLOR_WORDS);
    let before = lane.device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    lane.draw(&command_buffer, &targets[0], LoadAction::Load, &lane.colors);
    lane.copy_back(&command_buffer, &targets[0], &backs[0]);
    lane.dispatch(&command_buffer, &colors);
    lane.draw(&command_buffer, &targets[1], LoadAction::Load, &colors);
    lane.copy_back(&command_buffer, &targets[1], &backs[1]);
    run(&command_buffer);
    let after = lane.device.work_counts();
    let moved = Moved::of(&before, &after);
    let inline = after.render_passes_begun_inline - before.render_passes_begun_inline;
    let wrong: Vec<usize> = backs.iter().map(wrong_texels).collect();
    println!("RENDER-FOLD-INLINE {moved:?} inline {inline} wrong texels {wrong:?}");
    assert_eq!(wrong, [0, 0], "every texel is the draw's");
    assert_eq!(
        inline, 2,
        "both passes recorded their first instance inline"
    );
    assert_eq!(
        (moved.folded, moved.begun),
        (3, 2),
        "each inline render encoder began a group, which the encoders after it joined: {moved:?}"
    );
}

/// What moved of the counts a continued render pass instance names (`encoder_fold.rs`'s
/// **Continued passes**).
#[derive(Debug, PartialEq, Eq)]
struct Continued {
    /// `render_passes_begun_inline`.
    begun: u64,
    /// `render_passes_continued`.
    continued: u64,
    /// `commands_recorded.cmd_begin_render_pass`.
    begins: u64,
    /// `command_buffers_submitted`.
    submitted: u64,
}

impl Continued {
    fn of(before: &WorkCounts, after: &WorkCounts) -> Continued {
        Continued {
            begun: after.render_passes_begun_inline - before.render_passes_begun_inline,
            continued: after.render_passes_continued - before.render_passes_continued,
            begins: after.commands_recorded.cmd_begin_render_pass
                - before.commands_recorded.cmd_begin_render_pass,
            submitted: after.command_buffers_submitted - before.command_buffers_submitted,
        }
    }
}

/// A `Shared` buffer of the colors of every vertex the quad takes, each `color`.
fn colored(lane: &Lane, color: [f32; 4]) -> Buffer {
    floats(&lane.device, (0..VERTICES).flat_map(|_| color))
}

/// [`STEPS`] render encoders over one target, the first clearing it and the rest loading it,
/// each storing it and drawing the quad over all of it in a color of its own, the last
/// [`DRAWN`]: each after the first continues the first's render pass instance, which begins
/// once, and the texels are the last draw's, as Metal's passes in order leave them. Without the
/// continuation each encoder begins an instance of its own, in a `VkCommandBuffer` of its own.
#[test]
fn render_encoders_over_one_target_continue_one_render_pass_instance() {
    let scratch = Scratch::new("render-fold-continued");
    let Some(lane) = Lane::at_call(&scratch) else {
        return;
    };
    let queue = lane.device.new_command_queue().unwrap();
    let target = lane.target();
    let back = lane.back();
    let colors: Vec<Buffer> = (0..STEPS - 1)
        .map(|step| colored(&lane, [0.0, 0.1 * step as f32, 1.0, 1.0]))
        .chain([lane.colors.clone()])
        .collect();
    let before = lane.device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for (step, colors) in colors.iter().enumerate() {
        let load = match step {
            0 => LoadAction::Clear,
            _ => LoadAction::Load,
        };
        lane.draw(&command_buffer, &target, load, colors);
    }
    lane.copy_back(&command_buffer, &target, &back);
    run(&command_buffer);
    let continued = Continued::of(&before, &lane.device.work_counts());
    let wrong = wrong_texels(&back);
    println!("RENDER-FOLD-CONTINUED {continued:?} wrong texels {wrong}");
    assert_eq!(wrong, 0, "every texel is the last draw's");
    assert_eq!(
        (continued.begun, continued.continued, continued.begins),
        (1, STEPS - 1, 1),
        "one render pass instance began, every later encoder continued it: {continued:?}"
    );
    assert!(
        continued.submitted < STEPS,
        "the encoders went out in fewer command buffers than one each: {continued:?}"
    );
}

/// The same chain with a compute encoder after its third render encoder, which writes the
/// colors the fourth draws with: the compute encoder joins the group and ends the open render
/// pass instance first, the fourth render encoder begins one of its own, which the rest continue,
/// and every texel is the last draw's: the colors the dispatch wrote.
#[test]
fn a_dispatch_between_render_encoders_ends_the_continued_instance() {
    let scratch = Scratch::new("render-fold-continued-dispatch");
    let Some(lane) = Lane::at_call(&scratch) else {
        return;
    };
    let queue = lane.device.new_command_queue().unwrap();
    let target = lane.target();
    let back = lane.back();
    let written = lane.zeros(COLOR_WORDS);
    let blue = colored(&lane, [0.0, 0.0, 1.0, 1.0]);
    let before = lane.device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for _ in 0..3 {
        lane.draw(&command_buffer, &target, LoadAction::Load, &blue);
    }
    lane.dispatch(&command_buffer, &written);
    for _ in 0..3 {
        lane.draw(&command_buffer, &target, LoadAction::Load, &written);
    }
    lane.copy_back(&command_buffer, &target, &back);
    run(&command_buffer);
    let continued = Continued::of(&before, &lane.device.work_counts());
    let wrong = wrong_texels(&back);
    println!("RENDER-FOLD-CONTINUED-DISPATCH {continued:?} wrong texels {wrong}");
    assert_eq!(
        wrong, 0,
        "every texel is the last draw's, of the dispatch's colors"
    );
    assert_eq!(
        (continued.begun, continued.continued, continued.begins),
        (2, 4, 2),
        "the dispatch ended the first instance: {continued:?}"
    );
}

/// Render encoders that continue one render pass instance, committed behind an event wait only
/// the host signals, the last drawing with colors from a buffer over host pages that is rebound
/// before the signal: the rebind encodes the group again, each encoder apart, refusing none, and
/// the last draw takes its colors from the new pages (`encoder_fold.rs`'s **Encoded again**).
#[test]
fn a_rebind_encodes_continued_render_encoders_again() {
    // Declared before the lane, so dropped after it.
    let pages = [Page::new(), Page::new()];
    let scratch = Scratch::new("render-fold-continued-rebind");
    let Some(lane) = Lane::at_call(&scratch) else {
        return;
    };
    let queue = lane.device.new_command_queue().unwrap();
    let target = lane.target();
    let words: Vec<u8> = (0..VERTICES)
        .flat_map(|_| DRAWN)
        .flat_map(f32::to_le_bytes)
        .collect();
    pages[1].write(&words);
    // SAFETY: each page outlives every buffer over it (dropped after the lane), and only the
    // buffer's commands use it while it lives.
    let colors = unsafe {
        lane.device.new_buffer_with_bytes_no_copy(
            pages[0].memory(),
            BYTES as u64,
            ResourceOptions::shared(),
            None,
        )
    }
    .unwrap();
    let blue = colored(&lane, [0.0, 0.0, 1.0, 1.0]);
    let gate = lane.device.new_shared_event().unwrap();
    let before = lane.device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::Shared(&gate), 1)
        .unwrap();
    lane.draw(&command_buffer, &target, LoadAction::Load, &blue);
    lane.draw(&command_buffer, &target, LoadAction::Load, &blue);
    lane.draw(&command_buffer, &target, LoadAction::Load, &colors);
    command_buffer.commit().unwrap();
    assert_eq!(colors.detach_backing(), Ok(true));
    // SAFETY: as above.
    unsafe { colors.rebind_bytes_no_copy(pages[1].memory(), None, None) }.unwrap();
    // Read before the signal: the rebind encodes again in its own call.
    let after = lane.device.work_counts();
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        command_buffer.error()
    );
    let back = lane.back();
    let copy = queue.command_buffer().unwrap();
    lane.copy_back(&copy, &target, &back);
    run(&copy);
    let continued = after.render_passes_continued - before.render_passes_continued;
    let reencoded = after.encoders_reencoded - before.encoders_reencoded;
    let refused = after.reencodes_refused - before.reencodes_refused;
    println!(
        "RENDER-FOLD-CONTINUED-REBIND continued {continued} reencoded {reencoded} refused \
         {refused}"
    );
    assert_eq!(
        continued, 2,
        "both later encoders continued the first's instance"
    );
    assert_eq!(
        (reencoded, refused),
        (3, 0),
        "the rebind encoded the group's three encoders again"
    );
    assert_eq!(
        wrong_texels(&back),
        0,
        "the last draw took its colors from the new pages"
    );
}

/// On a device with no buffer over host pages a render encoder logs no call, so it takes no
/// group: the compute encoder's, its own and the copy's `VkCommandBuffer`s, as before.
#[test]
fn a_render_encoder_without_a_log_takes_no_group() {
    let scratch = Scratch::new("render-fold-logless");
    let Some(lane) = Lane::new(&scratch, RenderLoweringPolicy::AtCall, false) else {
        return;
    };
    let queue = lane.device.new_command_queue().unwrap();
    let target = lane.target();
    let (colors, back) = (lane.zeros(COLOR_WORDS), lane.back());
    let before = lane.device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    lane.dispatch(&command_buffer, &colors);
    lane.draw(&command_buffer, &target, LoadAction::DontCare, &colors);
    lane.copy_back(&command_buffer, &target, &back);
    run(&command_buffer);
    let moved = Moved::of(&before, &lane.device.work_counts());
    println!("RENDER-FOLD-LOGLESS {moved:?}");
    assert_eq!(wrong_texels(&back), 0, "every texel is the draw's");
    assert_eq!((moved.folded, moved.begun), (0, 3), "{moved:?}");
}

/// A compute encoder, a render encoder that joins its group and whose pass `encode` leaves work
/// for later, and a copy of the texture the pass rendered: the render encoder joins, and the
/// group ends after it, so the copy begins a `VkCommandBuffer` of its own. What moved.
fn ends_its_group(
    lane: &Lane,
    encode: impl FnOnce(&CommandBuffer) -> Texture,
    after_encoding: impl FnOnce(),
    back: &Buffer,
) -> Moved {
    let queue = lane.device.new_command_queue().unwrap();
    let colors = lane.zeros(COLOR_WORDS);
    let before = lane.device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    lane.dispatch(&command_buffer, &colors);
    let target = encode(&command_buffer);
    lane.copy_back(&command_buffer, &target, back);
    let moved = Moved::of(&before, &lane.device.work_counts());
    command_buffer.commit().unwrap();
    after_encoding();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        command_buffer.error()
    );
    moved
}

/// A draw whose pipeline is still being made (the pool held) records its pass later, as a
/// segment: the render encoder joined the compute encoder's group, which ends after it.
#[test]
fn a_draw_whose_pipeline_is_being_made_ends_its_group() {
    let scratch = Scratch::new("render-fold-pending");
    let Some(lane) = Lane::at_call(&scratch) else {
        return;
    };
    let target = lane.target();
    let back = lane.back();
    let row_center = 1.0 - 1.0 / SIDE as f32;
    let line = floats(&lane.device, [-2.0, row_center, 2.0, row_center]);
    let hold = held::hold();
    let pooled = lane.device.work_counts().pipelines_pooled;
    let moved = ends_its_group(
        &lane,
        |command_buffer| {
            let mut encoder = (command_buffer)
                .render_command_encoder(&pass(&target, LoadAction::DontCare))
                .unwrap();
            encoder.set_render_pipeline_state(&lane.render).unwrap();
            encoder.set_vertex_buffer(Some(&line), 0, 0).unwrap();
            encoder.set_vertex_buffer(Some(&lane.colors), 0, 1).unwrap();
            // A line, of another class than the triangles the state's creation made the
            // pipeline of: only its draw makes its pipeline.
            encoder
                .draw_primitives(PrimitiveType::Line, 0, 2, 1, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
            target.clone()
        },
        || drop(hold),
        &back,
    );
    let pooled = lane.device.work_counts().pipelines_pooled - pooled;
    let top = words_of(&back)[..SIDE as usize].to_vec();
    println!("RENDER-FOLD-PENDING {moved:?} pooled {pooled}");
    assert_eq!(pooled, 1, "the line's making waited in the pool");
    assert!(
        top.iter().all(|word| word.to_le_bytes() == DRAWN_TEXEL),
        "the line drew the top row: {top:x?}"
    );
    assert_eq!(
        (moved.folded, moved.begun),
        (1, 2),
        "the render encoder joined, the copy began its own: {moved:?}"
    );
}

/// A pass that executes an indirect command buffer's draw records later, as a segment: the
/// render encoder joined the compute encoder's group, which ends after it.
#[test]
fn an_execution_of_render_commands_ends_its_group() {
    use super::indirect_command_buffer_device::render_state as icb_state;
    use crate::IndirectCommandBufferDescriptor;
    use metal2vulkan::metal::{IndirectCommandType, RenderStages, ResourceUsage};
    let scratch = Scratch::new("render-fold-execution");
    let Some(lane) = Lane::at_call(&scratch) else {
        return;
    };
    let target = lane.target();
    let back = lane.back();
    let state = icb_state(&lane.device, true);
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.max_vertex_buffer_bind_count = 2;
    let icb = (lane.device)
        .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
        .unwrap();
    let command = icb.indirect_render_command_at(0).unwrap();
    command.set_render_pipeline_state(&state).unwrap();
    command
        .set_vertex_buffer(Some(&lane.positions), 0, 0)
        .unwrap();
    command.set_vertex_buffer(Some(&lane.colors), 0, 1).unwrap();
    command
        .draw_primitives(PrimitiveType::Triangle, 0, VERTICES.into(), 1, 0)
        .unwrap();
    let moved = ends_its_group(
        &lane,
        |command_buffer| {
            let mut encoder = (command_buffer)
                .render_command_encoder(&pass(&target, LoadAction::DontCare))
                .unwrap();
            for buffer in [&lane.positions, &lane.colors] {
                encoder
                    .use_resource(
                        buffer.as_resource(),
                        ResourceUsage::READ,
                        RenderStages::VERTEX,
                    )
                    .unwrap();
            }
            encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            encoder.end_encoding().unwrap();
            target.clone()
        },
        || (),
        &back,
    );
    println!("RENDER-FOLD-EXECUTION {moved:?}");
    assert_eq!(
        wrong_texels(&back),
        0,
        "every texel is the execution's draw's"
    );
    assert_eq!(
        (moved.folded, moved.begun),
        (1, 2),
        "the render encoder joined, the copy began its own: {moved:?}"
    );
}

/// A tile dispatch whose pipeline is still being made (the pool held) records its pass later, as
/// a segment: the render encoder joined the compute encoder's group, which ends after it. Its
/// pass's last attachment, every pixel of which the dispatch writes, is loaded `DontCare`, so the
/// pass records at its end: one whose first instance records inline joins no group
/// (`a_render_encoder_that_records_inline_begins_a_group_and_joins_none`).
#[test]
fn a_tile_dispatch_whose_pipeline_is_being_made_ends_its_group() {
    use super::tile_pending::{tile_state, Targets};
    let scratch = Scratch::new("render-fold-tile");
    let Some(lane) = Lane::at_call(&scratch) else {
        return;
    };
    let state = tile_state(&lane.device);
    let targets = Targets::new(&lane.device);
    // Made, and the lane's pipelines with it, before the pool is held.
    let copied = lane.target();
    let back = lane.back();
    let hold = held::hold();
    let before = lane.device.work_counts();
    let moved = ends_its_group(
        &lane,
        |command_buffer| {
            let mut pass = RenderPassDescriptor::new();
            for (attachment, target) in pass.color_attachments.iter_mut().zip(&targets.colors) {
                attachment.texture = Some(target.clone());
                attachment.load_action = LoadAction::Load;
                attachment.store_action = StoreAction::Store;
            }
            pass.color_attachments[2].load_action = LoadAction::DontCare;
            pass.tile_width = 16;
            pass.tile_height = 16;
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&state).unwrap();
            encoder.set_tile_buffer(Some(&targets.out), 0, 0).unwrap();
            encoder
                .dispatch_threads_per_tile(Size::new(8, 8, 1))
                .unwrap();
            encoder.end_encoding().unwrap();
            copied.clone()
        },
        || drop(hold),
        &back,
    );
    let after = lane.device.work_counts();
    let pooled = after.tile_pipelines_pooled - before.tile_pipelines_pooled;
    let inline = after.render_passes_begun_inline - before.render_passes_begun_inline;
    println!("RENDER-FOLD-TILE {moved:?} pooled {pooled} inline {inline}");
    assert_eq!(pooled, 1, "the dispatch's making waited in the pool");
    assert_eq!(inline, 0, "the pass recorded at its end");
    assert!(targets.stored(), "the dispatch stored what Metal's stores");
    assert_eq!(
        (moved.folded, moved.begun),
        (1, 2),
        "the render encoder joined, the copy began its own: {moved:?}"
    );
}

/// A four-sample pass whose color is resolved into a target set late
/// (`set_color_resolve_target`, `render/late_resolve.rs`) records its pass, the resolve with it,
/// at its end in the group's `VkCommandBuffer`: it joins, and so does the copy of the resolved
/// target after it, which reads every texel the resolve wrote.
#[test]
fn a_late_resolve_records_in_its_group() {
    let scratch = Scratch::new("render-fold-resolve");
    let Some(lane) = Lane::at_call(&scratch) else {
        return;
    };
    let library = super::encoder_holds::library(&lane.device, "examples/runtime_examples.ll");
    let function = |name: &str| {
        let function = library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("colored_vertex"));
    descriptor.fragment = Some(function("colored_fragment"));
    descriptor.raster_sample_count = 4;
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let multisampled = (lane.device)
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let mut msaa = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    msaa.texture_type = TextureType::Type2DMultisample;
    msaa.sample_count = 4;
    msaa.usage = TextureUsage::RENDER_TARGET;
    msaa.resource_options = ResourceOptions::private();
    let samples = lane.device.new_texture(&msaa).unwrap();
    let resolved = lane.target();
    let encode = |command_buffer: &CommandBuffer, colors: &Buffer| {
        let mut pass = pass(&samples, LoadAction::DontCare);
        pass.color_attachments[0].store_action = StoreAction::Unknown;
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&multisampled).unwrap();
        encoder
            .set_vertex_buffer(Some(&lane.positions), 0, 0)
            .unwrap();
        encoder.set_vertex_buffer(Some(colors), 0, 1).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, VERTICES, 1, 0)
            .unwrap();
        (encoder.set_color_resolve_target(Some(ResolveTarget::new(resolved.clone())), 0)).unwrap();
        (encoder.set_color_store_action(StoreAction::MultisampleResolve, 0)).unwrap();
        encoder.end_encoding().unwrap();
    };
    // Warm: the four-sample pass's pipeline made, so the pass below records at its end.
    let queue = lane.device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    encode(&command_buffer, &lane.colors);
    run(&command_buffer);

    let (colors, back) = (lane.zeros(COLOR_WORDS), lane.back());
    let before = lane.device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    lane.dispatch(&command_buffer, &colors);
    encode(&command_buffer, &colors);
    lane.copy_back(&command_buffer, &resolved, &back);
    run(&command_buffer);
    let moved = Moved::of(&before, &lane.device.work_counts());
    println!("RENDER-FOLD-RESOLVE {moved:?}");
    assert_eq!(wrong_texels(&back), 0, "every resolved texel is the draw's");
    assert_eq!(
        (moved.folded, moved.begun),
        (2, 1),
        "the render encoder and the copy joined the compute encoder's group: {moved:?}"
    );
}

/// A compute encoder, a render encoder, a compute encoder and a blit encoder's fill of one group,
/// and a concurrent compute encoder after it, which begins a group of its own, committed as
/// failed with `encoders`: which of the five ran (each compute encoder writes the colors of a
/// buffer of its own, the render encoder draws over a target the copy after the commit reads),
/// and how many encoders the commit encoded again (`WorkCounts::encoders_reencoded`).
fn failed_with(encoders: FailedEncoders) -> ([bool; 5], u64) {
    let scratch = Scratch::new("render-fold-failed");
    let lane = Lane::at_call(&scratch).expect("the device imports host memory");
    let queue = lane.device.new_command_queue().unwrap();
    let target = lane.target();
    let outputs: Vec<Buffer> = (0..5).map(|_| lane.zeros(COLOR_WORDS)).collect();
    let before = lane.device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    lane.dispatch(&command_buffer, &outputs[0]);
    lane.draw(&command_buffer, &target, LoadAction::DontCare, &lane.colors);
    lane.dispatch(&command_buffer, &outputs[2]);
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(&outputs[3], 0..COLOR_WORDS * 4, 0x5a)
        .unwrap();
    blit.end_encoding().unwrap();
    lane.dispatch_of(&command_buffer, DispatchType::Concurrent, &outputs[4]);
    let moved = Moved::of(&before, &lane.device.work_counts());
    assert_eq!((moved.folded, moved.begun), (3, 2), "{moved:?}");
    let failure = CommitFailure {
        code: 2,
        encoders,
        signals: true,
    };
    command_buffer.commit_failed(failure).unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
    assert_eq!(
        command_buffer.error(),
        Some(CommandBufferError::Injected(2))
    );
    let after = lane.device.work_counts();
    let back = lane.back();
    let copy = queue.command_buffer().unwrap();
    lane.copy_back(&copy, &target, &back);
    run(&copy);
    let ran = |at: usize| match at {
        1 => wrong_texels(&back) == 0,
        _ => words_of(&outputs[at]).iter().any(|&word| word != 0),
    };
    let reencoded = after.encoders_reencoded - before.encoders_reencoded;
    ([ran(0), ran(1), ran(2), ran(3), ran(4)], reencoded)
}

/// A failed set that names a group's render encoder fails it alone: the group's other three are
/// encoded again apart at the commit, and the encoder after it runs.
#[test]
fn a_failed_render_encoder_inside_a_group_fails_alone() {
    let set = FailedEncoders::Set(BTreeSet::from([1]));
    assert_eq!(failed_with(set), ([true, false, true, true, true], 3));
}

/// A failed set that names the compute encoder before a group's render encoder fails it alone:
/// the render encoder is encoded again from its log, apart, with the other two.
#[test]
fn a_failed_encoder_before_a_render_encoder_of_its_group_fails_alone() {
    let set = FailedEncoders::Set(BTreeSet::from([0]));
    assert_eq!(failed_with(set), ([false, true, true, true, true], 3));
}

/// A first failed encoder past a group's render encoder fails from there on: the compute and
/// render encoders before it are encoded again apart at the commit, and run.
#[test]
fn a_first_failed_encoder_after_a_render_encoder_runs_it() {
    let from = FailedEncoders::From(2);
    assert_eq!(failed_with(from), ([true, true, false, false, false], 2));
}

/// A compute encoder, a render encoder whose pass records at its end and a compute encoder, of
/// one group, are three of Metal's submissions (a render pass is one of its own). With a
/// submission check that fails the render encoder's, the commit encodes them again apart
/// (`State::split_failure`; the pass not where the check failed it first), as the group's
/// `VkCommandBuffer` holds commands of each: the pass runs nothing and the dispatches around it
/// run, as on Metal; the check sees each submission once.
#[test]
fn a_check_failing_a_group_s_render_pass_fails_it_alone() {
    let scratch = Scratch::new("render-fold-checked");
    let lane = Lane::at_call(&scratch).expect("the device imports host memory");
    let queue = lane.device.new_command_queue().unwrap();
    let target = lane.target();
    let outputs: Vec<Buffer> = (0..3).map(|_| lane.zeros(COLOR_WORDS)).collect();
    let before = lane.device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    lane.dispatch(&command_buffer, &outputs[0]);
    lane.draw(&command_buffer, &target, LoadAction::DontCare, &lane.colors);
    lane.dispatch(&command_buffer, &outputs[2]);
    let moved = Moved::of(&before, &lane.device.work_counts());
    assert_eq!((moved.folded, moved.begun), (2, 1), "{moved:?}");
    let seen = Arc::new(Mutex::new(Vec::new()));
    let noted = Arc::clone(&seen);
    let check: SubmissionCheck = Arc::new(move |submission: &Submission<'_>| {
        let kind = submission.kind();
        noted.lock().unwrap().push((kind, submission.encoders()));
        (kind == EncoderKind::Render).then(|| NonZeroU64::new(8).unwrap())
    });
    command_buffer.set_submission_check(check).unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
    assert_eq!(
        command_buffer.error(),
        Some(CommandBufferError::Injected(8))
    );
    let after = lane.device.work_counts();
    // The pass's encoding again takes its place too, unless the check, on the scheduler's
    // thread, failed its submission first (`command_buffer/reencode.rs`, **Split at the commit**).
    let reencoded = after.encoders_reencoded - before.encoders_reencoded;
    assert!(matches!(reencoded, 2 | 3), "{reencoded}");
    assert_eq!(
        *seen.lock().unwrap(),
        [
            (EncoderKind::Compute, 0..1),
            (EncoderKind::Render, 1..2),
            (EncoderKind::Compute, 2..3)
        ]
    );
    let back = lane.back();
    let copy = queue.command_buffer().unwrap();
    lane.copy_back(&copy, &target, &back);
    run(&copy);
    assert_eq!(
        wrong_texels(&back),
        words_of(&back).len(),
        "the pass drew nothing"
    );
    for at in [0, 2] {
        assert!(
            words_of(&outputs[at]).iter().any(|&word| word != 0),
            "the dispatch of encoder {at} ran"
        );
    }
}

/// A group of a compute encoder and a render encoder that draws with colors from a buffer over
/// host pages, committed behind an event wait only the host signals, the buffer rebound to other
/// pages before the signal: the rebind encodes both again (`command_buffer/reencode.rs`, groups
/// whole), refusing none, and the draw takes its colors from the new pages.
#[test]
fn a_rebind_encodes_a_group_with_a_render_encoder_again() {
    // Declared before the lane, so dropped after it.
    let pages = [Page::new(), Page::new()];
    let scratch = Scratch::new("render-fold-rebind");
    let Some(lane) = Lane::at_call(&scratch) else {
        return;
    };
    let queue = lane.device.new_command_queue().unwrap();
    let target = lane.target();
    let words: Vec<u8> = (0..VERTICES)
        .flat_map(|_| DRAWN)
        .flat_map(f32::to_le_bytes)
        .collect();
    pages[1].write(&words);
    // SAFETY: each page outlives every buffer over it (dropped after the lane), and only the
    // buffer's commands use it while it lives.
    let colors = unsafe {
        lane.device.new_buffer_with_bytes_no_copy(
            pages[0].memory(),
            BYTES as u64,
            ResourceOptions::shared(),
            None,
        )
    }
    .unwrap();
    let written = lane.zeros(COLOR_WORDS);
    let gate = lane.device.new_shared_event().unwrap();
    let before = lane.device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::Shared(&gate), 1)
        .unwrap();
    lane.dispatch(&command_buffer, &written);
    lane.draw(&command_buffer, &target, LoadAction::DontCare, &colors);
    command_buffer.commit().unwrap();
    assert_eq!(colors.detach_backing(), Ok(true));
    // SAFETY: as above.
    unsafe { colors.rebind_bytes_no_copy(pages[1].memory(), None, None) }.unwrap();
    // Read before the signal: the rebind encodes again in its own call.
    let after = lane.device.work_counts();
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        command_buffer.error()
    );
    let back = lane.back();
    let copy = queue.command_buffer().unwrap();
    lane.copy_back(&copy, &target, &back);
    run(&copy);
    let moved = Moved::of(&before, &after);
    println!(
        "RENDER-FOLD-REBIND {moved:?} reencoded {} refused {}",
        after.encoders_reencoded - before.encoders_reencoded,
        after.reencodes_refused - before.reencodes_refused
    );
    assert_eq!(
        moved.folded, 1,
        "the render encoder joined the compute encoder's group"
    );
    assert_eq!(
        (
            after.encoders_reencoded - before.encoders_reencoded,
            after.reencodes_refused - before.reencodes_refused
        ),
        (2, 0),
        "the rebind encoded both encoders of the group again"
    );
    assert_eq!(
        wrong_texels(&back),
        0,
        "the draw took its colors from the new pages"
    );
}
