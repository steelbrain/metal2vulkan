//! What a window server's frame costs the thread that encodes it, on the device the cfg selects,
//! without the validation layer: report only, in release builds, to weigh the runtime's encode
//! path (a pass's opening, its draws, its end, the commit) before and after a change to it.
//!
//! The workload is a guest window server's frame as the consumer (macOS guests over the runtime)
//! replays it, on one thread: [`BUFFERS`] command buffers committed back to back, each one render
//! pass of [`DRAWS`] indexed draws of six `UInt16` indices, with the state changes a compositor
//! makes between draws. A pass draws, in turn: a textured rectangle from a vertex buffer
//! (`tob_rect`, `tob_sample`; its rectangle at an offset of its own, one of [`TEXTURES`]
//! textures, a linear sampler); another of them; an antialiased mask (`zwd_quad`, `zwd_masked`:
//! its shape as vertex and fragment bytes, a mask texture, the sampler); another mask, of other
//! bytes; a rounded panel (`zwd_panel`: its shape's bytes alone); and a rectangle again. Every
//! pipeline blends premultiplied colors into `BGRA8Unorm_sRGB`. Each pass loads one of
//! [`TARGETS`] window layers and stores it, but every [`RESOLVED`]th, which clears a 4-sample
//! texture and resolves it into its layer. The frame waits for its last command buffer at its end,
//! outside the times below.
//!
//! The device is made as the consumer makes its own: the default policy (the fast forms), GPU
//! times from the host's clock, no execution record, no layer. The `announced` runs add every
//! precision kind to `DeviceOptions::announce_each`, delivered to a sink that counts each kind and
//! keeps its last text, as the consumer's does. The `checked` runs give each command buffer a
//! submission check of its own before its commit (`CommandBuffer::set_submission_check`), as the
//! consumer commits every command buffer: a check that notes the submission under a lock and
//! lets it run, as the consumer's does while nothing is detached. The device's scheduler thread
//! then runs each check and submits each batch, so the commit hands its batches off to it.
//!
//! [`WARM`] frames run untimed (every pipeline is made), the first of them reported alone (its CPU
//! and wall time, its blocks, its commits' median and longest wall time), then [`TIMED`] frames are
//! printed as `FRAME-COST` lines: the encoding thread's CPU time and wall time per frame (from the
//! first pass's opening through the last commit), and how often it blocked, inside its commits too,
//! and was preempted meanwhile (on Linux, its voluntary and involuntary context switches, from
//! `getrusage`: a commit that waits for the scheduler's lock blocks); the 50th and 90th percentiles
//! of the wall time of a pass's opening (`render_command_encoder`), of a draw (its state changes
//! and its draw call), of a pass's end (`end_encoding`) and of a commit; where the lib test binary
//! counts its allocations (a build without `--cfg m2v_system_alloc`), the encoding thread's
//! allocations of each; where the kernel lets a thread count its own (Linux), the user-space
//! instructions the encoding thread retired in each, on average and their 50th percentile, and in
//! a frame's steps together (steady where the times are not, so a change shows run to run),
//! counted in [`COUNTED`] frames of their own after the timed ones: each read of the counter is a
//! system call, which a timed frame's CPU time would hold; the notices the sink took per frame;
//! and a frame's share of the device's work counts (`Device::work_counts`) the draw and commit
//! paths keep, the command buffers begun (by what for: `WorkCounts::command_buffers_begun_encoders`
//! and kin) and submitted among them, the set-0 writes pushed and those of the switches between
//! layouts of equal shapes (`WorkCounts::set_zero_equal_shape_switches` and kin), the passes whose
//! first instance recorded inline and the others by the check that ruled it out
//! (`WorkCounts::inline_passes`, `WorkCounts::passes_not_inline_actions` and kin)
//! with the render pass begins and secondary executions recorded
//! (`WorkCounts::commands_replayed`), and the device's live
//! `VkDeviceMemory` by kind after the timed frames (ANV on i915 lists each in every kernel
//! submission). Then the CPU time of the
//! process's other threads over the timed frames, their waits included, per command buffer (on
//! Linux, from each thread's `schedstat`), with its user-space part (from its `stat`, in clock
//! ticks) and how often they slept (their voluntary context switches, from their `status`: the
//! scheduler's are the wakes it took, a commit's among them): the runtime's scheduler, queue-wait
//! (which make the completion turns) and handler threads by name, and the rest (the driver's) together; with their
//! allocations per command buffer where they are counted, and the submission checks run per frame.
//!
//! `frame_costs` measures the plain device, the checked one, then the announced one, each in turn
//! on one device of its own; `frame_costs_plain`, `frame_costs_checked` and
//! `frame_costs_announced` measure one alone, in a process of its own, for a profile of it
//! (`perf record -g` of the test's process).
//!
//! `frame_costs_over_imports` measures the checked device holding host imports live beside its
//! own memory, as the consumer's guest does, each in every kernel submission on ANV and RADV, for
//! each count of [`IMPORTS`], then with the more in the guest's shape (a second queue taking every
//! [`SECOND_QUEUE_EVERY`]th command buffer, every command buffer carrying a submission handler, as
//! the consumer's queue gives each one, and signalling a shared event at its end, every
//! [`PAGING_WAIT_EVERY`]th of the first queue's waiting for one the second signals):
//! under the default `CompletionPolicy::Grouped`, then under `CompletionPolicy::Exact`, twice each
//! in turn, each on a device of its own. Its lines also count the `vkQueueSubmit` calls
//! that held several batches (`WorkCounts::grouped_submissions`) and their batches, the batches
//! that ended in their work's last command buffer or in one of their own
//! (`WorkCounts::finales_folded`, `WorkCounts::finales_apart`), and the groups
//! by what sent them (`WorkCounts::group_ends_full` and kin), those kept for their queue's earlier
//! work and those of them a wait sent (`WorkCounts::groups_kept_pending`,
//! `WorkCounts::groups_sent_for_waits`), the batches by how many earlier
//! batches of their queue were pending as each went (`WorkCounts::batches_with_none_pending` and
//! kin), and the scheduler thread's sleeps
//! and wakings by what woke it (`WorkCounts::scheduler_sleeps` and kin), and the
//! condition-variable notifies the runtime's threads made and spared (`WorkCounts::wakes_spared`
//! and kin). Where the
//! kernel-submission counter is loaded (`scripts/i915-execbuffers`, `LD_PRELOAD`), every
//! measurement's lines count the kernel submissions of its timed frames, the objects each lists
//! and the time each spent in the ioctl.
//!
//! `frame_costs_over_buffers` measures the plain device with window layers that are textures over
//! `Shared` buffers at tight rows (2048 bytes, which RADV and ANV render in place, as linear images
//! over the buffers' memory) in place of private textures, their passes loaded and stored, loaded
//! and stored `DontCare`, and cleared and stored `DontCare`, each on a device of its own: what a
//! pass that does not store such a layer costs, rendered through the image its texture keeps so
//! that the buffer keeps its bytes. Its lines also give a frame's wall time through its last
//! command buffer's completion, the GPU's work included.
//!
//! `frame_costs_repeating` measures the plain device with each of a pass's draws but its panel
//! drawn again right after itself, with no call before it ([`REPEATED`]): 11 draws a pass, 5 of
//! which (45%) follow a draw with no call between them, about the share of the consumer's guest's
//! draws on the Iris Xe (43%). Its draws bind what the draws before them bound, which every draw
//! of the other runs changes.
//!
//! `frame_costs_guest_passes` measures the plain device with frames of the consumer's guest's
//! shapes ([`Shape`]), each pass loading its layer as the guest's passes do ([`GUEST_LOADS`]: most
//! clear it) and its draws the window server's in turn from its first: the old image's 100 passes
//! of one draw, in 100 command buffers of one pass and in one command buffer of all 100; and the
//! new image's frame ([`NEW_IMAGE`]): 41 command buffers, 9 of no pass, 32 of one pass, most of
//! one or two draws but five of 20 to 105, which hold most of its 277 draws. A single-draw pass's
//! own costs (its opening, end and commit, its instance's state) are a draw's there, where the
//! window server's passes spread them over six draws.
//!
//! `frame_costs_guest_composite` measures the plain device with frames of the guest's composite
//! passes ([`Composite`]), one a command buffer of a plain draw and then two through the fetching
//! CoreAnimation composite: a `BGRA8Unorm` layer loaded as [`GUEST_LOADS`] and stored beside a
//! memoryless `RGBA16Float` scratch loaded and stored `DontCare`, the shape of the guest's passes
//! whose first instance does not record inline; then with the scratch a private texture cleared
//! and stored, whose actions the inline path takes. On a device that fetches through the pixel
//! interlock the fetching draws leave the layer out of their subpass and the plain draw keeps it
//! in its own, so each pass splits there, as the guest's do, or runs whole where the device runs
//! such a pass faster so (`render/reroute.rs`). Its lines count the passes by why their first
//! instance did not record inline, the split instances recorded inline, the instances ended
//! unrun, and the replays' render pass begins and secondary executions.
//!
//! `frame_costs_alternating` measures the plain device with frames of passes whose draws switch
//! between two pipeline states of one layout at every draw ([`Shape::Alternating`]): the mask's,
//! and a gathering mask's of the same interface (`zwd_mask_gather`), each draw its shape's bytes
//! and the mask texture and sampler set once a pass, the passes loading their layers as
//! [`GUEST_LOADS`]: 640 draws, 600 switches a frame. The
//! consumer's guest's passes switch so between states whose layouts are defined alike, which share
//! one layout (`WorkCounts::pipeline_layouts_shared`): its lines give what a draw's push of set 0
//! carries there.
//!
//! `frame_costs_part_area` measures the plain device with the old image's passes over a render
//! area of 500x300 of each 512x512 layer ([`Shape::Part`]), whose clears reach past it to whole
//! tiles, 512x320 (`clear_reach.rs`).
//!
//! `frame_costs_host_pages` measures the plain device with the index and rectangle buffers over
//! host pages ([`Workload::host_pages`]), as every buffer of the consumer's guest lies over its RAM,
//! in the window server's frame and in the new image's: each draw's uses of them then take the
//! uses of the keys their bytes share, and its keys are named and their pages held, as the guest's
//! draws' are.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`, in a
//! build with `--cfg m2v_system_alloc` for the times, whose lib test binary keeps the system
//! allocator as a consumer's process does (the counting one costs the draw path about a sixth of
//! its time), and in one without it for the allocations.
//!
//! Three checks run by default where the allocations are counted:
//! `a_frame_s_draws_and_pass_ends_allocate_within_their_bounds` holds a frame's draws to
//! [`DRAW_ALLOCATIONS`] on the encoding thread, a draw's state changes and its draw call
//! together ([`DRAW_ALLOCATIONS_DEBUG`] in a build with debug assertions), and its passes' ends
//! to [`PASS_END_ALLOCATIONS`], each on average;
//! `a_frame_s_pass_opens_allocate_within_their_bound` its pass opens to [`OPEN_ALLOCATIONS`]
//! ([`FIRST_LOCK_ALLOCATIONS`] more on Apple's platforms); and
//! `a_frame_s_commits_allocate_within_their_bound` holds a checked frame's commits to
//! [`COMMIT_ALLOCATIONS`] on the encoding thread ([`FIRST_LOCK_ALLOCATIONS`] more on Apple's
//! platforms) and to [`SUBMIT_ALLOCATIONS`] on the others (the scheduler's passes, checks and
//! submissions, the completion), per command buffer on average.
//! Another runs by default on every device,
//! `a_frame_s_passes_and_commits_begin_their_command_buffers_within_their_bound`: each pass's
//! encoder, from its opening through its end, begins at most [`PASS_BEGINS`] command buffers, each
//! commit none, and the frame's submissions none but the barrier command buffers they record to
//! keep, on a device that keeps them (`Device::work_counts`). They count, and time nothing.

use super::host_memory::{host_page, no_copy, Mapping};
use super::memoryless::memoryless_options;
use super::render_arguments::{Members, ORDERED};
use super::renders::public_render;
use super::submit_costs::Execbuffers;
use super::{buffer_with, loader, open_device, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::candidate::engine::graphics::Assembly;
use crate::candidate::engine::resources::pixel_format;
use crate::candidate::runtime_case::render_pipeline_state;
use crate::case::indirect_commands::metal_primitive_type;
use crate::gpu::TEST_DEVICE;
use crate::worker::alloc;
use metal2vulkan::bind::{DrawCall, SamplerDescriptor};
use metal2vulkan::device::CompletionPolicy;
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{
    BlendFactor, ColorWriteMask, DataType, IndexType, PixelFormat, PrimitiveType, RenderStages,
    ResourceUsage, SamplerMinMagFilter, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferStatus, CommandQueue, Degradation,
    DegradationCategory, Device, DeviceOptions, EventRef, Library, LoadAction, MessageSink,
    PipelineOptions, Region, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, SamplerState, ShaderCacheOptions, SharedEvent, StoreAction, SubmissionCheck,
    Texture, TextureDescriptor, WorkCounts,
};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The untimed and the timed frames of a run, and the frames whose instructions it counts.
const WARM: usize = 20;
const TIMED: usize = 100;
const COUNTED: usize = 20;
/// The command buffers of a frame, each one pass.
const BUFFERS: usize = 67;
/// The draws of a pass.
const DRAWS: usize = 6;
/// In a repeating run ([`Run::REPEATING`]), the draws of a pass drawn again right after
/// themselves, with no call between the two: every one but the panel.
const REPEATED: [bool; DRAWS] = [true, true, true, true, false, true];
/// Every `RESOLVED`th pass draws into a 4-sample texture resolved into its layer.
const RESOLVED: usize = 4;
/// The window layers the passes draw into, and their side.
const TARGETS: usize = 4;
const TARGET_SIDE: u64 = 512;
/// The render area of [`Shape::Part`]'s passes, of whole 32-pixel tiles on neither axis: a clear
/// reaches 512x320.
const PART: [u64; 2] = [500, 300];
/// The sampled textures the rectangles take in turn, and their side.
const TEXTURES: usize = 16;
const TEXTURE_SIDE: u64 = 64;
/// The bytes of `zwd_shapes`' `ZwdShape`: rectangle, fill, outline and shape.
const SHAPE_BYTES: usize = 64;

/// The load actions of the passes of the guest's shapes ([`Shape::Passes`], [`Shape::NewImage`]),
/// in turn: of every 20 passes, 15 clear their layer, 4 load it and 1 leaves it undefined, about
/// the consumer's guest's mix (74 clear, 20 load and 5 don't-care of 99, reims-vgpu's counts of
/// its window server's passes).
const GUEST_LOADS: [LoadAction; 20] = {
    use LoadAction::{Clear as C, DontCare as D, Load as L};
    [C, C, L, C, C, C, L, C, C, D, C, C, L, C, C, C, L, C, C, C]
};

/// The command buffers of a frame in [`Shape::NewImage`], in turn: each the draws of its one pass,
/// or `None` for one with no pass. As the consumer's guest's new image commits them (a frame of
/// its trace: 41 command buffers, none of more than one pass; 32 passes of 275 draws, on average a
/// pass of none, 15 of one, 4.6 of two, 5 of 3 to 5, 1.5 of 6 to 10, 3.8 of 11 to 50 and one of
/// more): here 41, 32 passes of 277 draws.
const NEW_IMAGE: [Option<usize>; 41] = [
    Some(1),
    Some(20),
    Some(1),
    None,
    Some(2),
    Some(1),
    Some(3),
    Some(1),
    Some(105),
    None,
    Some(1),
    Some(2),
    Some(6),
    Some(1),
    None,
    Some(3),
    Some(1),
    Some(30),
    Some(2),
    None,
    Some(1),
    Some(0),
    Some(4),
    Some(1),
    None,
    Some(35),
    Some(1),
    Some(2),
    Some(3),
    None,
    Some(1),
    Some(6),
    Some(1),
    None,
    Some(3),
    Some(2),
    Some(35),
    None,
    Some(1),
    Some(1),
    None,
];
/// The host imports a run over imports holds live: the consumer's guest drag on the Iris Xe held
/// 259 to 556, each in every kernel submission. With the fewer, the Iris Xe's scheduler thread
/// keeps up with the encoding thread; with the more, its submissions fall behind the commits, as
/// the guest's do (its commits' handlers ran 4 to 9 ms after them).
const IMPORTS: [usize; 2] = [260, 520];
/// In the guest's shape ([`Run::guest`]), the share of command buffers the second queue takes
/// (every 20th, about 5%, as the guest's blit queue against its exec queue), and of the first
/// queue's that wait for the second's paging event first (every 10th, about 10%).
const SECOND_QUEUE_EVERY: usize = 20;
const PAGING_WAIT_EVERY: usize = 10;
/// The bytes of each, over pages of its own.
const IMPORT_BYTES: usize = 512 << 10;
/// The most allocations a frame's draw makes on the encoding thread on average, its state changes
/// and its draw call together, once every pipeline is made: the binder writes each draw's commands
/// over the last draw's (`BindingPlan::instantiate_draw_into`), and the draw its uses, descriptor
/// writes, pushed set 0 and vertex buffers over the last draw's (`LiveBind`), so none of them
/// allocate, and a pass's draws hold the first rectangle they reach (`clear_reach::DrawnArea`) and
/// each color's box (`linear_attachment::Coverage`) in place. Measured: 6.05 on ANV (UHD 770).
#[cfg(not(m2v_system_alloc))]
const DRAW_ALLOCATIONS: f64 = 9.0;
/// [`DRAW_ALLOCATIONS`] in a build with debug assertions, which CI's tests are: the draw path's
/// allocations and those of `LiveBind::shape`'s checks there, which build the state's whole bind
/// shape at each draw to compare the kept or found one with it. Measured: 17.88 on lavapipe
/// (llvmpipe, LLVM 23.1.1, Mesa 26.2.4).
#[cfg(not(m2v_system_alloc))]
const DRAW_ALLOCATIONS_DEBUG: f64 = 20.0;
/// The most allocations a frame's pass open (`render_command_encoder`) makes on the encoding
/// thread on average, once every pipeline is made, [`FIRST_LOCK_ALLOCATIONS`] more on Apple's
/// platforms: its shape's lists (`color_list.rs`), its encoder's default viewport and scissor
/// rectangle (`render_state::ViewportList`), its attachments' extents and its tracker's first part
/// (`tracker::EncoderUses`) are held in place, and its lowering is boxed once where it is built
/// (`render.rs`), which the encoder then moves as a pointer. Measured: 4.00 on lavapipe in a build
/// with debug assertions; 5.00 on the M3 Max through MoltenVK.
#[cfg(not(m2v_system_alloc))]
const OPEN_ALLOCATIONS: f64 = 4.5;
/// The most allocations a frame's pass end (`end_encoding`) makes on the encoding thread on
/// average, once every pipeline is made: its instances' shapes' lists are held in place
/// (`color_list.rs`), its attachments' uses are listed in one list sized at once, and it lists
/// no extended-range shadows' settings, stand-ins' copies or tile dispatches' pipelines where
/// it has none, nor its instances' draws and moved attachments where every instance recorded
/// inline. Measured: 16.91 on ANV (UHD 770), and on lavapipe in a build with debug assertions.
#[cfg(not(m2v_system_alloc))]
const PASS_END_ALLOCATIONS: f64 = 19.0;
/// The most command buffers a pass begins, from its opening through its end, once every pipeline
/// is made: its primary and its draws' secondary. A pass whose first instance records inline, and
/// that does not split, begins its primary alone: its draws' secondary never begins.
const PASS_BEGINS: u64 = 2;
/// The most allocations a checked frame's commit makes on the encoding thread on average, once
/// every pipeline is made, [`FIRST_LOCK_ALLOCATIONS`] more on Apple's platforms: a commit hands its
/// batches to the scheduler's thread and returns, making no pass (`scheduler/inbox.rs`), so this
/// is the hand-off's cost. Measured: 3.00 on RADV (Radeon 780M), NVIDIA (RTX 3090, RTX 5080
/// Laptop GPU), ANV (Iris Xe) and lavapipe, 4.00 on the M3 Max through MoltenVK, every commit of
/// the frames handing its batches off.
#[cfg(not(m2v_system_alloc))]
const COMMIT_ALLOCATIONS: f64 = 3.0;
/// The allocations a commit makes on Apple's platforms beyond [`COMMIT_ALLOCATIONS`], and a pass
/// open beyond [`OPEN_ALLOCATIONS`]: there Rust's standard library (1.98.1) boxes a `Mutex`'s
/// pthread mutex at its first lock and a `Condvar`'s at its first notify. A commit takes its
/// command buffer's gate's lock for the first time (as it holds the command buffer for the
/// scheduler), and each pass of these frames, the only one of its command buffer, opens the
/// command buffer's first encoder, which takes the command buffer's state lock for the first time
/// (`CommandBuffer::open_render`, the M3 Max's allocation sites). Linux's locks allocate nothing.
#[cfg(not(m2v_system_alloc))]
const FIRST_LOCK_ALLOCATIONS: f64 = if cfg!(target_vendor = "apple") {
    1.0
} else {
    0.0
};
/// The most allocations the process's other threads make per command buffer of a checked frame:
/// the scheduler thread's pass, its check and its submission, its completion and its handlers.
/// Measured: 14.9 to 15.4 on lavapipe, its batches' grouped submissions allocating nothing
/// (`command_buffer/group.rs`); more on the M3 Max through MoltenVK, where the queues' waiters
/// wait for each command buffer's fence too
/// (`RuntimeDriverDefect::ReadsCommandBuffersAfterTheirSignal`) and whose locks allocate at their
/// first use.
#[cfg(not(m2v_system_alloc))]
const SUBMIT_ALLOCATIONS: f64 = 20.0;

/// One run's device, commits and layers: whether the device announces every precision kind,
/// whether each command buffer commits with a submission check ([`Checks`]), what the window
/// layers are, how many host imports the device holds live beside them ([`IMPORT_BYTES`] each),
/// and its policy's `completion`.
#[derive(Clone, Copy)]
struct Run {
    announced: bool,
    checked: bool,
    layers: Layers,
    imports: usize,
    completion: CompletionPolicy,
    guest: bool,
    /// Whether the [`REPEATED`] draws of each pass are drawn again with no call before them.
    repeats: bool,
    shape: Shape,
    /// Whether the index and rectangle buffers lie over host pages ([`Workload::host_pages`]).
    host: bool,
}

impl Run {
    const PLAIN: Run = Run {
        announced: false,
        checked: false,
        layers: Layers::Private,
        imports: 0,
        completion: CompletionPolicy::Grouped,
        guest: false,
        repeats: false,
        shape: Shape::Window,
        host: false,
    };
    const CHECKED: Run = Run {
        checked: true,
        ..Run::PLAIN
    };
    const ANNOUNCED: Run = Run {
        announced: true,
        ..Run::PLAIN
    };
    const REPEATING: Run = Run {
        repeats: true,
        ..Run::PLAIN
    };

    /// The plain device's run of a frame of `shape`.
    const fn shaped(shape: Shape) -> Run {
        Run {
            shape,
            ..Run::PLAIN
        }
    }

    /// [`shaped`](Self::shaped)'s, its index and rectangle buffers over host pages.
    const fn hosted(shape: Shape) -> Run {
        Run {
            host: true,
            ..Run::shaped(shape)
        }
    }

    /// The checked device's run with `imports` host imports live, under `completion`, in the
    /// guest's shape where `guest` ([`Workload::guest`]).
    const fn over_imports(imports: usize, completion: CompletionPolicy, guest: bool) -> Run {
        Run {
            imports,
            completion,
            guest,
            ..Run::CHECKED
        }
    }

    /// The plain device's run with layers over buffers, its passes loading and storing them so.
    const fn over_buffers(load: LoadAction, store: StoreAction) -> Run {
        Run {
            layers: Layers::OverBuffers(load, store),
            ..Run::PLAIN
        }
    }

    fn label(self) -> String {
        let device = match (self.announced, self.checked) {
            (false, false) => "plain",
            (false, true) => "checked",
            (true, false) => "announced",
            (true, true) => "announced checked",
        };
        let device = match self.layers {
            Layers::Private => device.to_owned(),
            Layers::OverBuffers(load, store) => {
                format!("{device}, layers over buffers, {load:?} load, {store:?} store")
            }
        };
        let device = match self.guest {
            false => device,
            true => format!("{device}, the guest's queues and events"),
        };
        let device = match self.repeats {
            false => device,
            true => format!("{device}, draws repeated with no call before them"),
        };
        let device = match self.shape {
            Shape::Window => device,
            Shape::Passes { buffers, passes } => format!(
                "{device}, {buffers} command buffers of {passes} passes of one draw, the guest's \
                 loads"
            ),
            Shape::NewImage => {
                format!("{device}, the guest's new image's frame, the guest's loads")
            }
            Shape::Composite {
                buffers,
                draws,
                kept,
            } => format!(
                "{device}, {buffers} command buffers of 1 composite pass of a plain draw and {} \
                 fetching draws, the guest's loads, {}",
                draws - 1,
                match kept {
                    false => "a memoryless DontCare scratch",
                    true => "a private scratch cleared and stored",
                }
            ),
            Shape::Alternating { buffers, draws } => format!(
                "{device}, {buffers} command buffers of 1 pass of {draws} masked shapes \
                 alternating between two pipeline states of one layout, the guest's loads"
            ),
            Shape::Part { buffers, passes } => format!(
                "{device}, {buffers} command buffers of {passes} passes of one draw over {}x{} of \
                 each layer, the guest's loads",
                PART[0], PART[1]
            ),
        };
        let device = match self.host {
            false => device,
            true => format!("{device}, buffers over host pages"),
        };
        match self.imports {
            0 => device,
            imports => format!(
                "{device}, {imports} host imports, {:?} completion",
                self.completion
            ),
        }
    }
}

/// How a frame's draws group into passes and command buffers.
#[derive(Clone, Copy)]
enum Shape {
    /// The window server's (the module docs): [`BUFFERS`] command buffers of one pass of
    /// [`DRAWS`] draws, every [`RESOLVED`]th pass resolved.
    Window,
    /// The guest's old image's: `buffers` command buffers of `passes` passes of one draw each,
    /// loading as [`GUEST_LOADS`].
    Passes { buffers: usize, passes: usize },
    /// The guest's new image's: the command buffers of [`NEW_IMAGE`], loading as
    /// [`GUEST_LOADS`].
    NewImage,
    /// The guest's composite passes ([`Composite`]): `buffers` command buffers of one pass of
    /// `draws` draws each, a plain draw and then fetching ones, the scratch memoryless and
    /// `DontCare`, or, where `kept`, a private texture cleared and stored (actions the inline path
    /// takes).
    Composite {
        buffers: usize,
        draws: usize,
        kept: bool,
    },
    /// `buffers` command buffers of one pass of `draws` masked shapes, which switch between the
    /// mask's pipeline state and the gathering mask's ([`Workload::gather`]) at every draw,
    /// loading as [`GUEST_LOADS`].
    Alternating { buffers: usize, draws: usize },
    /// [`Shape::Passes`]'s passes over a render area of [`PART`] of each layer, whose clears reach
    /// past it, to whole tiles (`clear_reach.rs`).
    Part { buffers: usize, passes: usize },
}

impl Shape {
    /// Each command buffer's passes, as their draws.
    fn layout(self) -> Vec<Vec<usize>> {
        match self {
            Shape::Window => vec![vec![DRAWS]; BUFFERS],
            Shape::Passes { buffers, passes } => vec![vec![1; passes]; buffers],
            Shape::NewImage => (NEW_IMAGE.iter())
                .map(|draws| draws.iter().copied().collect())
                .collect(),
            Shape::Composite { buffers, draws, .. } => vec![vec![draws]; buffers],
            Shape::Alternating { buffers, draws } => vec![vec![draws]; buffers],
            Shape::Part { buffers, passes } => vec![vec![1; passes]; buffers],
        }
    }
}

/// What the window layers are, and how a pass that is not resolved loads and stores its layer.
#[derive(Clone, Copy, Debug)]
enum Layers {
    /// Private textures, loaded and stored.
    Private,
    /// Textures over `Shared` buffers at tight rows, loaded and stored by these.
    OverBuffers(LoadAction, StoreAction),
}

/// The submissions the checks of a checked run saw: each the encoders it named, noted under a
/// lock, as the consumer's check notes each submission it lets run.
#[derive(Default)]
struct Checks {
    seen: Mutex<(u64, Range<usize>)>,
}

impl Checks {
    /// A command buffer's check: note the submission, and let it run.
    fn check(self: &Arc<Self>) -> SubmissionCheck {
        let checks = Arc::clone(self);
        Arc::new(move |submission| {
            let mut seen = checks.seen.lock().unwrap();
            seen.0 += 1;
            seen.1 = submission.encoders();
            None
        })
    }

    /// The checks run so far.
    fn count(&self) -> u64 {
        self.seen.lock().unwrap().0
    }
}

/// The pipelines of one raster sample count: the rectangle's, the mask's and the panel's.
struct Pipelines {
    rect: RenderPipelineState,
    masked: RenderPipelineState,
    panel: RenderPipelineState,
}

/// The device, its queues, and what the frames draw with.
struct Workload {
    device: Device,
    layers: Layers,
    /// The buffers the layers lie over, where they do.
    layer_buffers: Vec<Buffer>,
    /// The command queues a frame's command buffers are committed on: the first alone, or, in
    /// the guest's shape, the second taking every [`SECOND_QUEUE_EVERY`]th.
    queues: Vec<CommandQueue>,
    /// In the guest's shape, the shared events every command buffer signals at its end with the
    /// next value (the guest's FIFO event), and the second queue's command buffers signal too,
    /// for every [`PAGING_WAIT_EVERY`]th of the first queue's to wait for first (its paging
    /// event), with the values signalled so far.
    guest: Option<GuestEvents>,
    /// Whether each pass draws its [`REPEATED`] draws again with no call before them.
    repeats: bool,
    /// The frame's shape, and each command buffer's passes as their draws ([`Shape::layout`]).
    shape: Shape,
    layout: Vec<Vec<usize>>,
    /// A checked run's submissions checked.
    checks: Option<Arc<Checks>>,
    /// One sample, and four.
    pipelines: [Pipelines; 2],
    sampler: SamplerState,
    textures: Vec<Texture>,
    mask: Texture,
    /// The window layers, and the 4-sample texture a resolved pass draws into.
    targets: Vec<Texture>,
    multisampled: Texture,
    /// Six `UInt16` indices, 0 to 5, and the rectangles' corners, one `float4` a draw of a frame.
    indices: Buffer,
    rects: Buffer,
    /// Where they lie over host pages: a no-copy buffer over all of them, and the pages, which
    /// outlive the three buffers (declared after them).
    _host_pages: Option<(Buffer, Mapping)>,
    /// What a composite frame draws with, in a run of that shape.
    composite: Option<Composite>,
    /// In an alternating run ([`Shape::Alternating`]), a gathering mask's pipeline state of one
    /// sample (`zwd_quad`, `zwd_mask_gather`): the mask's interface, so a layout defined as the
    /// mask's, and another variant.
    gather: Option<RenderPipelineState>,
}

/// The guest's composite pass, as its passes the inline path takes none of
/// (`WorkCounts::passes_not_inline_actions`) are: color 0 one of [`TARGETS`] private
/// `BGRA8Unorm` layers, loaded as [`GUEST_LOADS`] and stored, color 1 a memoryless `RGBA16Float`
/// scratch loaded and stored `DontCare`; each draw through the CoreAnimation composite of
/// `runtime_lane::framebuffer_fetch::blend8` (the owned fixture `fbl_fetch_blend8_safe`'s
/// `fbl8_safe_loaded`): color 0 fetched and blended one over one minus the source alpha, color 1
/// written under an empty mask; each pass's first draw a plain one before them, white
/// (`rt6_resolves`'s `rt6_resolve_white`) through the same blend and mask.
struct Composite {
    /// The plain draw's pipeline.
    plain: RenderPipelineState,
    /// The fetching draws' pipeline.
    pipeline: RenderPipelineState,
    layers: Vec<Texture>,
    scratch: Texture,
    /// The `half4` sources the fragment function reads, at `64 y + x` and 4096 past it.
    sources: Buffer,
    /// Whether the scratch is a private texture its passes clear and store.
    kept: bool,
}

impl Composite {
    fn new(device: &Device, kept: bool) -> Composite {
        let library = device
            .new_library_with_air(
                &llvm_assemble(&engine_dir().join("fbl_fetch_blend8_safe.ll")).unwrap(),
            )
            .unwrap();
        let function = |name| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("fbl8_safe_vertex"));
        descriptor.fragment = Some(function("fbl8_safe_loaded"));
        let white = device
            .new_library_with_air(&llvm_assemble(&engine_dir().join("rt6_resolves.ll")).unwrap())
            .unwrap();
        let white = white
            .new_function_with_name("rt6_resolve_white")
            .unwrap()
            .function_descriptor();
        let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm);
        color.blending_enabled = true;
        color.source_rgb_blend_factor = BlendFactor::One;
        color.source_alpha_blend_factor = BlendFactor::One;
        color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        let mut scratch = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
        scratch.write_mask = ColorWriteMask::NONE;
        descriptor.color_attachments[0] = Some(color);
        descriptor.color_attachments[1] = Some(scratch);
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        descriptor.fragment = Some(StageSource::Function(white));
        let plain = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let layer = || {
            let mut layer = TextureDescriptor::texture_2d(
                PixelFormat::BGRA8Unorm,
                TARGET_SIDE,
                TARGET_SIDE,
                false,
            );
            layer.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            layer.resource_options = ResourceOptions::private();
            device.new_texture(&layer).unwrap()
        };
        let mut scratch = TextureDescriptor::texture_2d(
            PixelFormat::RGBA16Float,
            TARGET_SIDE,
            TARGET_SIDE,
            false,
        );
        scratch.usage = TextureUsage::RENDER_TARGET;
        scratch.resource_options = match kept {
            false => memoryless_options(),
            true => ResourceOptions::private(),
        };
        // A half 0.5 in every channel, past every index the fragment reads.
        let count = (64 * (TARGET_SIDE - 1) + TARGET_SIDE + 4096) as usize;
        let half = 0x3800u16.to_le_bytes();
        let sources: Vec<u8> = (0..4 * count).flat_map(|_| half).collect();
        Composite {
            plain,
            pipeline,
            layers: (0..TARGETS).map(|_| layer()).collect(),
            scratch: device.new_texture(&scratch).unwrap(),
            sources: buffer_with(device, &sources).unwrap(),
            kept,
        }
    }
}

/// A function of `library`, its constant 0 (`zwd_variant`) 0.
fn function(library: &Library, name: &str) -> StageSource {
    let constants = FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(DataType::UInt, 0u32.to_le_bytes().to_vec()),
    );
    StageSource::Function(
        library
            .new_function_with_name_constant_values(name, &constants)
            .unwrap()
            .function_descriptor(),
    )
}

/// A pipeline state of `vertex` and `fragment` blending premultiplied colors into
/// `BGRA8Unorm_sRGB`, rasterizing `samples` samples.
fn pipeline(
    device: &Device,
    vertex: StageSource,
    fragment: StageSource,
    samples: u32,
) -> RenderPipelineState {
    let mut descriptor = RenderPipelineDescriptor::new(vertex);
    descriptor.fragment = Some(fragment);
    descriptor.raster_sample_count = samples;
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA8UnormSRGB);
    color.blending_enabled = true;
    color.source_rgb_blend_factor = BlendFactor::One;
    color.source_alpha_blend_factor = BlendFactor::One;
    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    descriptor.color_attachments[0] = Some(color);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A `TEXTURE_SIDE` square `BGRA8Unorm` texture, every texel `texel`.
fn sampled(device: &Device, texel: [u8; 4]) -> Texture {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, TEXTURE_SIDE, TEXTURE_SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let texels: Vec<u8> = (0..TEXTURE_SIDE * TEXTURE_SIDE)
        .flat_map(|_| texel)
        .collect();
    let region = Region::new_2d(0, 0, TEXTURE_SIDE, TEXTURE_SIDE);
    texture
        .replace_region(region, 0, 0, &texels, 4 * TEXTURE_SIDE, 0)
        .unwrap();
    texture
}

/// A `TARGET_SIDE` square `BGRA8Unorm_sRGB` render target of `samples` samples.
fn target(device: &Device, samples: u64) -> Texture {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8UnormSRGB, TARGET_SIDE, TARGET_SIDE, false);
    if samples > 1 {
        descriptor.texture_type = TextureType::Type2DMultisample;
        descriptor.sample_count = samples;
    }
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// A `TARGET_SIDE` square `BGRA8Unorm_sRGB` render target over a new `Shared` buffer at tight
/// rows, and the buffer.
fn target_over_buffer(device: &Device) -> (Texture, Buffer) {
    let rows = 4 * TARGET_SIDE;
    let buffer = device
        .new_buffer(rows * TARGET_SIDE, ResourceOptions::shared())
        .unwrap();
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8UnormSRGB, TARGET_SIDE, TARGET_SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = buffer.resource_options();
    (buffer.new_texture(&descriptor, 0, rows).unwrap(), buffer)
}

/// `ZwdShape`'s bytes: draw `n`'s box in a pass of the target's size, a fill and an outline.
fn shape(n: usize) -> [u8; SHAPE_BYTES] {
    let x = (n % 8) as f32 * 60.0;
    let y = (n / 8 % 8) as f32 * 60.0;
    let words: [f32; 16] = [
        x,
        y,
        x + 48.0,
        y + 32.0,
        0.5,
        0.25,
        0.125,
        0.5,
        0.25,
        0.25,
        0.25,
        0.25,
        6.0,
        TARGET_SIDE as f32,
        TARGET_SIDE as f32,
        0.0,
    ];
    let mut bytes = [0u8; SHAPE_BYTES];
    for (at, word) in words.iter().enumerate() {
        bytes[4 * at..4 * at + 4].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

impl Workload {
    fn new(
        device: Device,
        checked: bool,
        layers: Layers,
        (guest, repeats, shape, host): (bool, bool, Shape, bool),
    ) -> Workload {
        let library = |name: &str| {
            device.new_library_with_air(&llvm_assemble(&engine_dir().join(name)).unwrap())
        };
        let tob = library("tob_renders.ll").unwrap();
        let zwd = library("zwd_shapes.ll").unwrap();
        let pipelines = [1, 4].map(|samples| Pipelines {
            rect: pipeline(
                &device,
                function(&tob, "tob_rect"),
                function(&tob, "tob_sample"),
                samples,
            ),
            masked: pipeline(
                &device,
                function(&zwd, "zwd_quad"),
                function(&zwd, "zwd_masked"),
                samples,
            ),
            panel: pipeline(
                &device,
                function(&zwd, "zwd_quad"),
                function(&zwd, "zwd_panel"),
                samples,
            ),
        });
        let mut sampler = SamplerDescriptor::new();
        sampler.min_filter = SamplerMinMagFilter::Linear;
        sampler.mag_filter = SamplerMinMagFilter::Linear;
        let sampler = device.new_sampler_state(&sampler).unwrap();
        let textures = (0..TEXTURES)
            .map(|n| sampled(&device, [n as u8 * 16, 128, 255 - n as u8 * 16, 255]))
            .collect();
        let indices: Vec<u8> = (0u16..6).flat_map(u16::to_le_bytes).collect();
        // A rectangle a draw: a 25 by 20 grid of them over the target.
        let rects: Vec<u8> = (0..BUFFERS * DRAWS)
            .flat_map(|n| {
                let (x, y) = ((n % 25) as f32, (n / 25 % 20) as f32);
                let at = |v: f32, cells: f32| v * 2.0 / cells - 1.0;
                [
                    at(x, 25.0),
                    at(y, 20.0),
                    at(x + 1.0, 25.0),
                    at(y + 1.0, 20.0),
                ]
            })
            .flat_map(f32::to_le_bytes)
            .collect();
        let (targets, layer_buffers) = match layers {
            Layers::Private => (
                (0..TARGETS).map(|_| target(&device, 1)).collect(),
                Vec::new(),
            ),
            Layers::OverBuffers(..) => (0..TARGETS).map(|_| target_over_buffer(&device)).unzip(),
        };
        let (indices, rects, host_pages) = match host {
            false => (
                buffer_with(&device, &indices).unwrap(),
                buffer_with(&device, &rects).unwrap(),
                None,
            ),
            true => {
                let (indices, rects, whole) = Workload::host_pages(&device, &indices, &rects);
                (indices, rects, Some(whole))
            }
        };
        Workload {
            layers,
            layer_buffers,
            queues: (0..1 + usize::from(guest))
                .map(|_| device.new_command_queue().unwrap())
                .collect(),
            guest: guest.then(|| GuestEvents {
                fifo: device.new_shared_event().unwrap(),
                paging: device.new_shared_event().unwrap(),
                values: Cell::new((0, 0)),
            }),
            repeats,
            shape,
            layout: shape.layout(),
            checks: checked.then(Arc::default),
            pipelines,
            sampler,
            textures,
            mask: sampled(&device, [255, 255, 255, 128]),
            targets,
            multisampled: target(&device, 4),
            indices,
            rects,
            _host_pages: host_pages,
            composite: match shape {
                Shape::Composite { kept, .. } => Some(Composite::new(&device, kept)),
                _ => None,
            },
            gather: matches!(shape, Shape::Alternating { .. }).then(|| {
                pipeline(
                    &device,
                    function(&zwd, "zwd_quad"),
                    function(&zwd, "zwd_mask_gather"),
                    1,
                )
            }),
            device,
        }
    }

    /// The index and rectangle buffers over host pages, as the consumer's guest's buffers all
    /// lie over its RAM: each a no-copy buffer over pages of one mapping holding `indices` and
    /// `rects`, and a third no-copy buffer over the whole mapping, whose bytes meet both, so a
    /// use of either takes the uses of the keys its bytes share (`HostAliases::augment`), as the
    /// guest's overlapping buffers do.
    fn host_pages(
        device: &Device,
        indices: &[u8],
        rects: &[u8],
    ) -> (Buffer, Buffer, (Buffer, Mapping)) {
        let page = host_page();
        let (index_len, rect_len) = (
            indices.len().next_multiple_of(page),
            rects.len().next_multiple_of(page),
        );
        let pages = Mapping::anonymous(index_len + rect_len);
        pages.write(0, indices);
        pages.write(index_len, rects);
        let over = |offset: usize, len: usize| {
            // SAFETY: the pages are this process's, mapped read and write, and outlive the
            // buffers (`Workload::host_pages`'s order).
            let at = unsafe { pages.pointer().add(offset) };
            // SAFETY: as above.
            unsafe { no_copy(device, at, len, ResourceOptions::shared(), None) }.unwrap()
        };
        let (indices, rects) = (over(0, index_len), over(index_len, rect_len));
        let whole = over(0, index_len + rect_len);
        (indices, rects, (whole, pages))
    }

    /// Pass `pass`'s descriptor: its layer loaded and stored (by the layers' actions where they
    /// lie over buffers), or, for a resolved pass, the 4-sample texture cleared and resolved into
    /// the layer.
    fn pass(&self, pass: usize) -> RenderPassDescriptor {
        let mut descriptor = RenderPassDescriptor::new();
        if let Some(composite) = &self.composite {
            let color = &mut descriptor.color_attachments[0];
            color.texture = Some(composite.layers[pass % TARGETS].clone());
            color.load_action = GUEST_LOADS[pass % GUEST_LOADS.len()];
            color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            color.store_action = StoreAction::Store;
            let scratch = &mut descriptor.color_attachments[1];
            scratch.texture = Some(composite.scratch.clone());
            (scratch.load_action, scratch.store_action) = match composite.kept {
                false => (LoadAction::DontCare, StoreAction::DontCare),
                true => (LoadAction::Clear, StoreAction::Store),
            };
            scratch.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            return descriptor;
        }
        let color = &mut descriptor.color_attachments[0];
        let layer = self.targets[pass % TARGETS].clone();
        if !matches!(self.shape, Shape::Window) {
            color.texture = Some(layer);
            color.load_action = GUEST_LOADS[pass % GUEST_LOADS.len()];
            color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            color.store_action = StoreAction::Store;
            if let Shape::Part { .. } = self.shape {
                [
                    descriptor.render_target_width,
                    descriptor.render_target_height,
                ] = PART;
            }
        } else if self.resolved(pass) {
            color.texture = Some(self.multisampled.clone());
            color.resolve_texture = Some(layer);
            color.load_action = LoadAction::Clear;
            color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            color.store_action = StoreAction::MultisampleResolve;
        } else {
            let (load, store) = match self.layers {
                Layers::Private => (LoadAction::Load, StoreAction::Store),
                Layers::OverBuffers(load, store) => (load, store),
            };
            color.texture = Some(layer);
            color.load_action = load;
            color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            color.store_action = store;
        }
        descriptor
    }

    /// Whether pass `pass` draws into the 4-sample texture and resolves it: every [`RESOLVED`]th
    /// of the window server's.
    fn resolved(&self, pass: usize) -> bool {
        matches!(self.shape, Shape::Window) && pass % RESOLVED == RESOLVED - 1
    }

    /// Draw `draw` of a pass (the module docs; `draw` past them as `draw % DRAWS`), its state
    /// changes first: pass `pass`'s, the `n`th draw of the frame.
    fn draw(&self, encoder: &mut RenderCommandEncoder, (pass, n): (usize, usize), draw: usize) {
        if let Some(composite) = &self.composite {
            match draw {
                0 => encoder.set_render_pipeline_state(&composite.plain).unwrap(),
                1 => {
                    encoder
                        .set_render_pipeline_state(&composite.pipeline)
                        .unwrap();
                    encoder
                        .set_fragment_buffer(Some(&composite.sources), 0, 0)
                        .unwrap();
                }
                _ => {}
            }
            self.draw_call(encoder);
            return;
        }
        if let Some(gather) = &self.gather {
            let state = match draw % 2 {
                0 => &self.pipelines[0].masked,
                _ => gather,
            };
            encoder.set_render_pipeline_state(state).unwrap();
            if draw == 0 {
                encoder.set_fragment_texture(Some(&self.mask), 0).unwrap();
                encoder
                    .set_fragment_sampler_state(Some(&self.sampler), 0)
                    .unwrap();
            }
            let shape = shape(n);
            encoder.set_vertex_bytes(&shape, 0).unwrap();
            encoder.set_fragment_bytes(&shape, 0).unwrap();
            self.draw_call(encoder);
            return;
        }
        let pipelines = &self.pipelines[usize::from(self.resolved(pass))];
        match draw % DRAWS {
            0 | 1 | 5 => {
                if draw != 1 {
                    encoder.set_render_pipeline_state(&pipelines.rect).unwrap();
                    encoder
                        .set_fragment_sampler_state(Some(&self.sampler), 0)
                        .unwrap();
                }
                encoder
                    .set_vertex_buffer(Some(&self.rects), 16 * n as u64, 0)
                    .unwrap();
                encoder
                    .set_fragment_texture(Some(&self.textures[n % TEXTURES]), 0)
                    .unwrap();
            }
            2 | 3 => {
                if draw == 2 {
                    encoder
                        .set_render_pipeline_state(&pipelines.masked)
                        .unwrap();
                    encoder.set_fragment_texture(Some(&self.mask), 0).unwrap();
                    encoder
                        .set_fragment_sampler_state(Some(&self.sampler), 0)
                        .unwrap();
                }
                let shape = shape(n);
                encoder.set_vertex_bytes(&shape, 0).unwrap();
                encoder.set_fragment_bytes(&shape, 0).unwrap();
            }
            _ => {
                encoder.set_render_pipeline_state(&pipelines.panel).unwrap();
                let shape = shape(n);
                encoder.set_vertex_bytes(&shape, 0).unwrap();
                encoder.set_fragment_bytes(&shape, 0).unwrap();
            }
        }
        self.draw_call(encoder);
    }

    /// The draw call of every draw: its rectangle's six indices.
    fn draw_call(&self, encoder: &mut RenderCommandEncoder) {
        encoder
            .draw_indexed_primitives(
                PrimitiveType::Triangle,
                6,
                IndexType::UInt16,
                &self.indices,
                0,
                1,
                0,
                0,
            )
            .unwrap();
    }

    /// The draws of a window server's pass: [`DRAWS`], and the [`REPEATED`] ones again where it
    /// repeats them.
    fn draws(&self) -> usize {
        DRAWS
            + REPEATED
                .iter()
                .filter(|&&again| again && self.repeats)
                .count()
    }

    /// A frame's command buffers, passes and draws.
    fn buffers(&self) -> usize {
        self.layout.len()
    }

    fn passes(&self) -> usize {
        self.layout.iter().map(Vec::len).sum()
    }

    fn draws_a_frame(&self) -> usize {
        match self.shape {
            Shape::Window => BUFFERS * self.draws(),
            _ => self.layout.iter().flatten().sum(),
        }
    }

    /// Encode and commit one frame, then wait for its last command buffer: what each step cost.
    fn frame(&self) -> Frame {
        self.frame_reading(Reads::default())
    }

    /// [`frame`](Self::frame), reading `reads` around each step too (so its times are not the
    /// plain frame's).
    fn frame_reading(&self, reads: Reads) -> Frame {
        let mut frame = Frame::default();
        // Each queue's last command buffer of the frame.
        let mut lasts: Vec<Option<CommandBuffer>> = self.queues.iter().map(|_| None).collect();
        let counts = || reads.work_counts.then(|| self.device.work_counts());
        let switches = thread_switches();
        let cpu = thread_cpu();
        let start = Instant::now();
        let at_start = counts();
        let mut begun = [Begun::default(); 4];
        let (mut pass, mut n) = (0, 0);
        for (buffer, passes) in self.layout.iter().enumerate() {
            let queue = usize::from(
                self.queues.len() > 1 && buffer % SECOND_QUEUE_EVERY == SECOND_QUEUE_EVERY - 1,
            );
            let command_buffer = self.queues[queue].command_buffer().unwrap();
            if let Some(guest) = &self.guest {
                guest.begin(&command_buffer, queue, buffer);
            }
            for &draws in passes {
                let descriptor = self.pass(pass);
                let before = counts();
                let (mut encoder, open) = step(reads.instructions, || {
                    command_buffer.render_command_encoder(&descriptor).unwrap()
                });
                frame.open.push(open);
                if let Some(before) = &before {
                    begun[0].add(Begun::since(&self.device, before));
                }
                for draw in 0..draws {
                    let first = match self.shape {
                        Shape::Window => pass * DRAWS + draw,
                        _ => n,
                    };
                    n += 1;
                    let again = self.repeats && REPEATED[draw % DRAWS];
                    for repeat in [false, true].into_iter().take(1 + usize::from(again)) {
                        let before = counts();
                        let ((), drawn) = step(reads.instructions, || match repeat {
                            false => self.draw(&mut encoder, (pass, first), draw),
                            true => self.draw_call(&mut encoder),
                        });
                        frame.draws.push(drawn);
                        if let Some(before) = &before {
                            begun[1].add(Begun::since(&self.device, before));
                        }
                    }
                }
                let before = counts();
                let ((), ended) = step(reads.instructions, || encoder.end_encoding().unwrap());
                frame.end.push(ended);
                if let Some(before) = &before {
                    begun[2].add(Begun::since(&self.device, before));
                }
                pass += 1;
            }
            if let Some(guest) = &self.guest {
                guest.end(&command_buffer, queue);
            }
            // The consumer makes each command buffer's check before its commit call.
            let check = self.checks.as_ref().map(Checks::check);
            let before = counts();
            let switches = thread_switches();
            let ((), committed) = step(reads.instructions, || {
                if let Some(check) = check {
                    command_buffer.set_submission_check(check).unwrap();
                }
                command_buffer.commit().unwrap()
            });
            if let Some((before, after)) = switches.zip(thread_switches()) {
                let blocked = after.since(before).voluntary;
                *frame.commit_blocks.get_or_insert(0) += blocked;
            }
            frame.commit.push(committed);
            if let Some(before) = &before {
                begun[3].add(Begun::since(&self.device, before));
                let after = self.device.work_counts();
                frame
                    .handed
                    .push(after.commits_handed_off > before.commits_handed_off);
            }
            lasts[queue] = Some(command_buffer);
        }
        frame.begun = begun;
        frame.wall = start.elapsed();
        frame.cpu = thread_cpu().saturating_sub(cpu);
        frame.switches = switches
            .zip(thread_switches())
            .map(|(before, after)| after.since(before));
        for last in lasts.iter().flatten() {
            last.wait_until_completed_for(WAIT).unwrap();
        }
        frame.done = start.elapsed();
        frame.whole = at_start.map(|before| Begun::since(&self.device, &before));
        for last in lasts.iter().flatten() {
            assert_eq!(last.status(), CommandBufferStatus::Completed);
        }
        frame
    }
}

/// One step's wall time, and the encoding thread's allocations and user-space instructions in it
/// where they are counted.
#[derive(Clone, Copy)]
struct Step {
    wall: Duration,
    allocations: Option<u64>,
    instructions: Option<u64>,
}

/// Run `body` as one step, counting its instructions where `counted`.
fn step<R>(counted: bool, body: impl FnOnce() -> R) -> (R, Step) {
    let allocations = alloc::thread_allocations();
    let instructions = counted.then(instructions::read).flatten();
    let start = Instant::now();
    let result = body();
    let wall = start.elapsed();
    let instructions = instructions.map(|before| instructions::read().unwrap() - before);
    let allocations = allocations.map(|before| alloc::thread_allocations().unwrap() - before);
    (
        result,
        Step {
            wall,
            allocations,
            instructions,
        },
    )
}

/// The calling thread's count of the instructions it retired in user space, where the kernel lets
/// a thread count its own (Linux's `perf_event_open`, `kernel.perf_event_paranoid` at most 2):
/// steady where a step's time is not, so a change to the encode path shows in it run to run. On a
/// hybrid CPU (Intel's P and E cores) it counts on the P cores alone (`cpu_core`), where a run is
/// pinned (`taskset`).
#[cfg(target_os = "linux")]
mod instructions {
    use std::cell::Cell;

    /// `struct perf_event_attr` as its first published size (`PERF_ATTR_SIZE_VER0`).
    #[repr(C)]
    struct Attr {
        kind: u32,
        size: u32,
        config: u64,
        sample_period: u64,
        sample_type: u64,
        read_format: u64,
        flags: u64,
        wakeup_events: u32,
        bp_type: u32,
        config1: u64,
    }

    /// `PERF_TYPE_HARDWARE`, `PERF_COUNT_HW_INSTRUCTIONS`, and the `exclude_kernel` and
    /// `exclude_hv` flags.
    const HARDWARE: u32 = 0;
    const INSTRUCTIONS: u64 = 1;
    const EXCLUDE_KERNEL_HV: u64 = (1 << 5) | (1 << 6);

    thread_local! {
        /// The thread's counter: its descriptor, or -1 where none opened; opened at its first read.
        static COUNTER: Cell<Option<i32>> = const { Cell::new(None) };
    }

    /// The counter's descriptor for the calling thread, opened at its first call.
    fn counter() -> i32 {
        COUNTER.with(|cell| {
            if let Some(fd) = cell.get() {
                return fd;
            }
            // A hybrid CPU's P cores' PMU, in the config's upper half; none elsewhere.
            let pmu = std::fs::read_to_string("/sys/bus/event_source/devices/cpu_core/type")
                .ok()
                .and_then(|text| text.trim().parse::<u64>().ok())
                .unwrap_or(0);
            let attr = Attr {
                kind: HARDWARE,
                size: std::mem::size_of::<Attr>() as u32,
                config: INSTRUCTIONS | (pmu << 32),
                sample_period: 0,
                sample_type: 0,
                read_format: 0,
                flags: EXCLUDE_KERNEL_HV,
                wakeup_events: 0,
                bp_type: 0,
                config1: 0,
            };
            // SAFETY: `attr` is a valid `perf_event_attr` of its stated size, live for the call;
            // the calling thread (0), any CPU (-1), no group (-1), no flags.
            let fd = unsafe {
                libc::syscall(
                    libc::SYS_perf_event_open,
                    &attr as *const Attr,
                    0,
                    -1,
                    -1,
                    0,
                )
            };
            let fd = i32::try_from(fd).unwrap_or(-1);
            cell.set(Some(fd));
            fd
        })
    }

    /// The instructions the calling thread retired in user space since its counter opened.
    pub(super) fn read() -> Option<u64> {
        let fd = counter();
        if fd < 0 {
            return None;
        }
        let mut count = 0u64;
        // SAFETY: `count` is 8 writable bytes, the size of the counter's value.
        let read = unsafe { libc::read(fd, (&mut count as *mut u64).cast(), 8) };
        (read == 8).then_some(count)
    }
}

/// Elsewhere none is counted.
#[cfg(not(target_os = "linux"))]
mod instructions {
    pub(super) fn read() -> Option<u64> {
        None
    }
}

/// What a frame reads around each step beside its wall time ([`Workload::frame_reading`]): none
/// in a timed frame, as each read costs the encoding thread.
#[derive(Clone, Copy, Default)]
struct Reads {
    /// The device's work counts: the command buffers each step began ([`Frame::begun`]).
    work_counts: bool,
    /// The encoding thread's user-space instructions in each ([`Step::instructions`]).
    instructions: bool,
}

/// The command buffers the device began, of those the ones encoders began as they recorded
/// (`WorkCounts::encoder_command_buffers_begun`) and the barrier command buffers it recorded to
/// keep, in a span a frame that read the work counts ran ([`Workload::frame_reading`]); and the
/// batches it submitted with no command buffer of barriers before their first encoder
/// (`WorkCounts::bare_starts`).
#[derive(Clone, Copy, Default)]
struct Begun {
    all: u64,
    encoder: u64,
    kept: u64,
    bare: u64,
}

impl Begun {
    /// What `device` began since `before`, its work counts then.
    fn since(device: &Device, before: &WorkCounts) -> Begun {
        let now = device.work_counts();
        Begun {
            all: now.command_buffers_begun - before.command_buffers_begun,
            encoder: now.encoder_command_buffers_begun - before.encoder_command_buffers_begun,
            kept: now.barrier_buffers_kept - before.barrier_buffers_kept,
            bare: now.bare_starts - before.bare_starts,
        }
    }

    fn add(&mut self, other: Begun) {
        self.all += other.all;
        self.encoder += other.encoder;
        self.kept += other.kept;
        self.bare += other.bare;
    }
}

/// A frame's steps of one kind ([`Frame`]'s `open`, `draws`, `end` or `commit`).
type StepsOf = fn(&Frame) -> &[Step];

/// One frame's steps and times ([`Workload::frame`]).
#[derive(Default)]
struct Frame {
    open: Vec<Step>,
    draws: Vec<Step>,
    end: Vec<Step>,
    commit: Vec<Step>,
    wall: Duration,
    cpu: Duration,
    /// Through the last command buffer's completion.
    done: Duration,
    /// The command buffers its passes' openings, their draws, their ends and its commits began,
    /// in that order, where it read the work counts ([`Workload::frame_reading`]); none
    /// elsewhere.
    begun: [Begun; 4],
    /// The command buffers begun from its first opening through its last command buffer's
    /// completion, on every thread, where it read the work counts: its batches' submissions, which
    /// the scheduler's thread makes while the encoding thread goes on, begin theirs in whichever
    /// step runs then.
    whole: Option<Begun>,
    /// The encoding thread's context switches over the same span as `cpu`, where they are read.
    switches: Option<Switches>,
    /// Its voluntary ones inside its commits.
    commit_blocks: Option<u64>,
    /// Whether each commit handed its batches to the scheduler's thread
    /// (`WorkCounts::commits_handed_off`), where it read the work counts; none elsewhere.
    handed: Vec<bool>,
}

/// A thread's context switches: the voluntary ones (it blocked: on a lock, a condition variable, a
/// sleep, I/O) and the involuntary ones (it was preempted).
#[derive(Clone, Copy, Default)]
struct Switches {
    voluntary: u64,
    involuntary: u64,
}

impl Switches {
    /// The switches from `before` to these.
    fn since(self, before: Switches) -> Switches {
        Switches {
            voluntary: self.voluntary.saturating_sub(before.voluntary),
            involuntary: self.involuntary.saturating_sub(before.involuntary),
        }
    }

    /// A thread's switches so far, from its `status` text.
    fn of_status(status: &str) -> Option<Switches> {
        let field = |name: &str| {
            status
                .lines()
                .find_map(|line| line.strip_prefix(name)?.trim().parse().ok())
        };
        Some(Switches {
            voluntary: field("voluntary_ctxt_switches:")?,
            involuntary: field("nonvoluntary_ctxt_switches:")?,
        })
    }
}

/// The calling thread's context switches so far ([`Switches`]), from `getrusage`: a system call,
/// which a frame makes twice a commit, outside its times. `None` off Linux.
#[cfg(target_os = "linux")]
fn thread_switches() -> Option<Switches> {
    // SAFETY: `rusage` is plain integers, for which zeros are a value.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `getrusage` writes the struct it is given and has no other effect.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut usage) };
    (rc == 0).then(|| Switches {
        voluntary: u64::try_from(usage.ru_nvcsw).unwrap_or(0),
        involuntary: u64::try_from(usage.ru_nivcsw).unwrap_or(0),
    })
}

/// The calling thread's context switches so far: not read off Linux.
#[cfg(not(target_os = "linux"))]
fn thread_switches() -> Option<Switches> {
    None
}

/// The calling thread's CPU time so far.
fn thread_cpu() -> Duration {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `clock_gettime` writes the timespec it is given and has no other effect.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut now) };
    assert_eq!(rc, 0, "clock_gettime(CLOCK_THREAD_CPUTIME_ID)");
    Duration::new(now.tv_sec as u64, now.tv_nsec as u32)
}

/// The 50th and 90th percentiles of `values`.
fn percentiles(mut values: Vec<f64>) -> [f64; 2] {
    values.sort_by(f64::total_cmp);
    let at = |p: usize| values[(values.len() * p / 100).min(values.len() - 1)];
    [at(50), at(90)]
}

/// The kinds announced, each with its count and last text: the consumer's sink.
type Heard = Arc<Mutex<BTreeMap<Degradation, (u64, String)>>>;

/// A thread's CPU time: all of it, from its `schedstat` (nanoseconds), and the user-space part,
/// from its `stat` (`utime`, in clock ticks: 10 ms on a common kernel, a coarse share); and how
/// often it slept, its voluntary context switches, from its `status`.
#[derive(Clone, Copy, Default)]
struct Cpu {
    all: Duration,
    user: Duration,
    sleeps: u64,
}

/// The process's other threads' CPU time so far ([`Cpu`]), each by its name (cut to 15 bytes, as
/// Linux keeps it): every thread but the calling one. `None` off Linux.
fn other_threads_cpu() -> Option<BTreeMap<String, Cpu>> {
    // `<pid>/task/<tid>`: the calling thread's own.
    let own = std::fs::read_link("/proc/thread-self").ok()?;
    // SAFETY: `sysconf` reads a constant of the system and has no other effect.
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    let tick = Duration::from_secs(1) / u32::try_from(ticks).ok().filter(|&t| t > 0)?;
    let mut threads = BTreeMap::new();
    for task in std::fs::read_dir("/proc/self/task").ok()? {
        let task = task.ok()?.path();
        if task.file_name() == own.file_name() {
            continue;
        }
        // A thread that ended meanwhile is left out.
        let (Ok(name), Ok(schedstat), Ok(stat), Ok(status)) = (
            std::fs::read_to_string(task.join("comm")),
            std::fs::read_to_string(task.join("schedstat")),
            std::fs::read_to_string(task.join("stat")),
            std::fs::read_to_string(task.join("status")),
        ) else {
            continue;
        };
        let nanos: u64 = schedstat.split_whitespace().next()?.parse().ok()?;
        // The fields after the name's closing parenthesis start at the third, the state;
        // `utime` is the fourteenth.
        let utime: u32 = stat
            .rsplit_once(')')?
            .1
            .split_whitespace()
            .nth(11)?
            .parse()
            .ok()?;
        let sleeps = Switches::of_status(&status)?.voluntary;
        let cpu: &mut Cpu = threads.entry(name.trim().to_owned()).or_default();
        cpu.all += Duration::from_nanos(nanos);
        cpu.user += tick * utime;
        cpu.sleeps += sleeps;
    }
    Some(threads)
}

/// The groups of the process's other threads [`other_threads_cpu`] reports: the runtime's by the
/// prefix of their names, and the rest.
const THREAD_GROUPS: [(&str, &str); 3] = [
    ("scheduler", "metal2vulkan-sc"),
    ("queue waits", "metal2vulkan-qu"),
    ("handlers", "metal2vulkan-ha"),
];

/// What the other threads spent between `before` and `after` ([`other_threads_cpu`]), by
/// [`THREAD_GROUPS`] and the rest, `other`: a thread that ended meanwhile is left out, one that
/// started counts from 0.
fn spent_by_group(
    before: &BTreeMap<String, Cpu>,
    after: &BTreeMap<String, Cpu>,
) -> Vec<(&'static str, Cpu)> {
    let mut groups: Vec<(&'static str, Cpu)> = THREAD_GROUPS
        .iter()
        .map(|(group, _)| (*group, Cpu::default()))
        .chain([("other", Cpu::default())])
        .collect();
    for (name, now) in after {
        let then = before.get(name).copied().unwrap_or_default();
        let at = THREAD_GROUPS
            .iter()
            .position(|(_, prefix)| name.starts_with(prefix))
            .unwrap_or(THREAD_GROUPS.len());
        groups[at].1.all += now.all.saturating_sub(then.all);
        groups[at].1.user += now.user.saturating_sub(then.user);
        groups[at].1.sleeps += now.sleeps.saturating_sub(then.sleeps);
    }
    groups
}

/// A device as the consumer makes its own (the module docs), announcing every precision kind at
/// every occurrence where `announced`, into `heard`.
fn device(
    cache: &ScratchCache,
    announced: bool,
    heard: &Heard,
    completion: CompletionPolicy,
) -> Device {
    let mut options = DeviceOptions::default();
    options.policy.completion = completion;
    options.shader_cache = ShaderCacheOptions::in_directory(cache.path());
    options.loader = loader();
    options.gpu_times_from_host = true;
    let sink = Arc::clone(heard);
    options.messages = Some(MessageSink::new(move |message| {
        if let Some(notice) = &message.degradation {
            let mut heard = sink.lock().unwrap();
            let entry = heard.entry(notice.degradation).or_default();
            entry.0 += 1;
            entry.1.clone_from(&message.text);
        }
    }));
    if announced {
        options.announce_each = Degradation::ALL
            .iter()
            .copied()
            .filter(|kind| kind.category() == DegradationCategory::Precision)
            .collect();
    }
    open_device(TEST_DEVICE, options).unwrap()
}

/// The frame's costs, printed (the module docs): a run of each of `runs` in turn, each on a device
/// of its own.
fn measure(runs: &[Run]) {
    for &run in runs {
        let cache = ScratchCache::new().unwrap();
        let heard = Heard::default();
        let work = Workload::new(
            device(&cache, run.announced, &heard, run.completion),
            run.checked,
            run.layers,
            (run.guest, run.repeats, run.shape, run.host),
        );
        let _imports = imports(&work.device, run.imports);
        let label = run.label();
        let name = work.device.name().to_owned();
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let first = work.frame();
        let [commit, _] = percentiles(
            first
                .commit
                .iter()
                .map(|s| s.wall.as_secs_f64() * 1e6)
                .collect(),
        );
        let longest = first
            .commit
            .iter()
            .map(|s| s.wall)
            .max()
            .unwrap_or_default();
        eprintln!(
            "FRAME-COST {name} {label}: the first frame: the encoding thread's CPU {:.2} ms, wall \
             {:.2} ms, {}; its commits {commit:.2} us (longest {:.2})",
            ms(first.cpu),
            ms(first.wall),
            blocks([&first]),
            longest.as_secs_f64() * 1e6,
        );
        for _ in 1..WARM {
            work.frame();
        }
        let notices_before: u64 = heard.lock().unwrap().values().map(|(n, _)| n).sum();
        let counts_before = work.device.work_counts();
        let checks_before = work.checks.as_ref().map_or(0, |checks| checks.count());
        let threads_before = other_threads_cpu();
        let allocations_before = others_allocations();
        let kernel_before = Execbuffers::read();
        let frames: Vec<Frame> = (0..TIMED).map(|_| work.frame()).collect();
        let kernel = Execbuffers::read();
        let allocations = others_allocations();
        let threads = other_threads_cpu();
        let checks = work.checks.as_ref().map_or(0, |checks| checks.count());
        let counts = work.device.work_counts();
        let notices: u64 = heard.lock().unwrap().values().map(|(n, _)| n).sum();
        let reads = Reads {
            instructions: true,
            ..Reads::default()
        };
        let counted: Vec<Frame> = (0..COUNTED).map(|_| work.frame_reading(reads)).collect();
        let (buffers, passes) = (work.buffers(), work.passes());
        let draws = work.draws_a_frame();
        let [cpu, cpu_high] = percentiles(frames.iter().map(|f| ms(f.cpu)).collect());
        let [wall, wall_high] = percentiles(frames.iter().map(|f| ms(f.wall)).collect());
        let [done, done_high] = percentiles(frames.iter().map(|f| ms(f.done)).collect());
        eprintln!(
            "FRAME-COST {name} {label}: {buffers} command buffers of {passes} passes of {draws} \
             indexed draws a frame, {WARM} untimed and {TIMED} timed frames: the encoding thread's CPU {cpu:.2} ms a frame (p90 {cpu_high:.2}), wall \
             {wall:.2} ms (p90 {wall_high:.2}), through completion {done:.2} ms (p90 \
             {done_high:.2}), {} a frame; {:.1} us of CPU a draw over the frame; {:.1} notices \
             a frame",
            blocks(&frames),
            cpu * 1e3 / draws as f64,
            (notices - notices_before) as f64 / TIMED as f64,
        );
        let steps: [(&str, StepsOf); 4] = [
            ("pass open", |f| &f.open),
            ("draw", |f| &f.draws),
            ("pass end", |f| &f.end),
            ("commit", |f| &f.commit),
        ];
        for (step, of) in steps {
            let all: Vec<Step> = frames.iter().flat_map(|f| of(f).iter().copied()).collect();
            let [p50, p90] = percentiles(all.iter().map(|s| s.wall.as_secs_f64() * 1e6).collect());
            let allocations = match all.iter().map(|s| s.allocations).sum::<Option<u64>>() {
                Some(total) => format!("{:.2} allocations each", total as f64 / all.len() as f64),
                None => "allocations not counted".to_owned(),
            };
            let read: Vec<Step> = counted.iter().flat_map(|f| of(f).iter().copied()).collect();
            let instructions = match read.iter().map(|s| s.instructions).sum::<Option<u64>>() {
                Some(total) => {
                    let [p50, _] = percentiles(
                        read.iter()
                            .filter_map(|s| s.instructions)
                            .map(|n| n as f64)
                            .collect(),
                    );
                    format!(
                        "{:.0} user-space instructions each (p50 {p50:.0})",
                        total as f64 / read.len() as f64
                    )
                }
                None => "instructions not counted".to_owned(),
            };
            eprintln!(
                "FRAME-COST {name} {label}: {step}: {p50:.2} us (p90 {p90:.2}); {allocations}; \
                 {instructions}"
            );
        }
        let instructions: Option<u64> = counted
            .iter()
            .flat_map(|f| f.open.iter().chain(&f.draws).chain(&f.end).chain(&f.commit))
            .map(|s| s.instructions)
            .sum();
        if let Some(total) = instructions {
            eprintln!(
                "FRAME-COST {name} {label}: the encoding thread's user-space instructions in a \
                 frame's steps, over {COUNTED} frames of their own: {:.0} a frame, {:.0} a pass",
                total as f64 / COUNTED as f64,
                total as f64 / (COUNTED * passes) as f64,
            );
        }
        let per_frame = |count: fn(&WorkCounts) -> u64| {
            (count(&counts) - count(&counts_before)) as f64 / TIMED as f64
        };
        eprintln!(
            "FRAME-COST {name} {label}: a frame's work: {:.0} bind shapes made, {:.0} found, \
             {:.0} draws settled, {:.0} argument sampler lists taken, {:.0} bind views worked \
             out, {:.0} stage tables read, {:.0} draws' uses noted as before, {:.0} pipelines \
             looked up, {:.0} draw states recorded, {:.0} draws pushing set 0, {:.0} argument \
             tables retained, {:.1} command pools made, \
             {:.1} timeline waits, {:.1} addressable buffers made, {:.1} queue submissions, {:.1} \
             commits handed off, {:.1} submission checks, {:.1} render passes made",
            per_frame(|c| c.draw_shapes_made),
            per_frame(|c| c.draw_shapes_found),
            per_frame(|c| c.draws_settled),
            per_frame(|c| c.argument_samplers_taken),
            per_frame(|c| c.bind_views_worked_out),
            per_frame(|c| c.stage_tables_read),
            per_frame(|c| c.commands_noted_as_before),
            per_frame(|c| c.graphics_pipelines_looked_up),
            per_frame(|c| c.draw_states_recorded),
            per_frame(|c| c.draws_pushing_set_zero),
            per_frame(|c| c.argument_tables_retained),
            per_frame(|c| c.command_pools_made),
            per_frame(|c| c.timeline_waits),
            per_frame(|c| c.addressable_buffers_made),
            per_frame(|c| c.queue_submissions),
            per_frame(|c| c.commits_handed_off),
            (checks - checks_before) as f64 / TIMED as f64,
            per_frame(|c| c.render_passes_made),
        );
        eprintln!(
            "FRAME-COST {name} {label}: a frame's set 0: {:.1} writes pushed; {:.1} switches \
             between layouts of equal shapes, which pushed {:.1} of them and would push {:.1} if \
             those layouts were one",
            per_frame(|c| c.set_zero_writes_pushed),
            per_frame(|c| c.set_zero_equal_shape_switches),
            per_frame(|c| c.set_zero_writes_on_equal_shape_switches),
            per_frame(|c| c.set_zero_writes_if_shapes_shared),
        );
        eprintln!(
            "FRAME-COST {name} {label}: a frame's passes: {:.1} recorded their first instance \
             inline; not inline for the encoder {:.1}, the layout {:.1}, a memoryless color {:.1}, \
             a color's actions {:.1} (a DontCare load {:.1}, a reach not shared {:.1}, \
             a resolve {:.1}, a store {:.1}), a texture over a buffer \
             {:.1}, another attachment kind \
             {:.1}, a storage color {:.1}, a stand-in {:.1}, a draw {:.1}; {:.1} split instances \
             recorded inline, {:.1} instances ended unrun; the replays' {:.1} render pass begins \
             and {:.1} secondary executions",
            per_frame(|c| c.inline_passes),
            per_frame(|c| c.passes_not_inline_encoders),
            per_frame(|c| c.passes_not_inline_layouts),
            per_frame(|c| c.passes_not_inline_memoryless),
            per_frame(|c| c.passes_not_inline_actions),
            per_frame(|c| c.passes_not_inline_actions_dont_care_load),
            per_frame(|c| c.passes_not_inline_actions_unequal_reach),
            per_frame(|c| c.passes_not_inline_actions_resolve),
            per_frame(|c| c.passes_not_inline_actions_store),
            per_frame(|c| c.passes_not_inline_buffer_textures),
            per_frame(|c| c.passes_not_inline_attachments),
            per_frame(|c| c.passes_not_inline_storage),
            per_frame(|c| c.passes_not_inline_stand_ins),
            per_frame(|c| c.passes_not_inline_draws),
            per_frame(|c| c.inline_split_instances),
            per_frame(|c| c.unrun_instances_ended),
            per_frame(|c| c.commands_replayed.cmd_begin_render_pass),
            per_frame(|c| c.commands_replayed.cmd_execute_commands),
        );
        eprintln!(
            "FRAME-COST {name} {label}: a frame's command buffers: {:.1} begun ({:.2} a pass), \
             {:.1} of them barrier command buffers recorded to keep, {:.1} encoders' own begun \
             ahead in their spare pools; {:.1} submitted ({:.2} a pass) in {:.1} vkQueueSubmit \
             calls, {:.1} of which held {:.1} batches together; {:.1} batches ended in their \
             work, {:.1} in a command buffer of their own",
            per_frame(|c| c.command_buffers_begun),
            per_frame(|c| c.command_buffers_begun) / passes as f64,
            per_frame(|c| c.barrier_buffers_kept),
            per_frame(|c| c.encoder_primaries_begun_ahead),
            per_frame(|c| c.command_buffers_submitted),
            per_frame(|c| c.command_buffers_submitted) / passes as f64,
            per_frame(|c| c.queue_submissions),
            per_frame(|c| c.grouped_submissions),
            per_frame(|c| c.batches_grouped),
            per_frame(|c| c.finales_folded),
            per_frame(|c| c.finales_apart),
        );
        eprintln!(
            "FRAME-COST {name} {label}: a frame's command buffers begun by what for: {:.1} \
             encoders' own, {:.1} segments, {:.1} continuations, {:.1} draws' secondaries, {:.1} \
             batches' work, {:.1} batches' barriers, {:.1} barriers kept, {:.1} internal",
            per_frame(|c| c.command_buffers_begun_encoders),
            per_frame(|c| c.command_buffers_begun_segments),
            per_frame(|c| c.command_buffers_begun_continuations),
            per_frame(|c| c.command_buffers_begun_secondaries),
            per_frame(|c| c.command_buffers_begun_batch_work),
            per_frame(|c| c.command_buffers_begun_batch_barriers),
            per_frame(|c| c.command_buffers_begun_kept_barriers),
            per_frame(|c| c.command_buffers_begun_internal),
        );
        eprintln!(
            "FRAME-COST {name} {label}: a frame's groups of batches by what sent them: {:.1} \
             full, {:.1} before a wait for a signal held, {:.1} before a batch apart, {:.1} with \
             first work, {:.1} with no batch left, {:.1} with checks or steps left, {:.1} with the \
             rest held back; {:.1} of them kept for their queue's earlier work, {:.1} of those \
             sent by a wait; its batches by the earlier ones of their queue pending as each went: \
             {:.1} none, {:.1} one, {:.1} two or three, {:.1} four or more",
            per_frame(|c| c.group_ends_full),
            per_frame(|c| c.group_ends_wait),
            per_frame(|c| c.group_ends_apart),
            per_frame(|c| c.group_ends_first_work),
            per_frame(|c| c.group_ends_drained),
            per_frame(|c| c.group_ends_checks),
            per_frame(|c| c.group_ends_held),
            per_frame(|c| c.groups_kept_pending),
            per_frame(|c| c.groups_sent_for_waits),
            per_frame(|c| c.batches_with_none_pending),
            per_frame(|c| c.batches_with_one_pending),
            per_frame(|c| c.batches_with_two_or_three_pending),
            per_frame(|c| c.batches_with_four_or_more_pending),
        );
        let per_buffer = |count: fn(&WorkCounts) -> u64| per_frame(count) / buffers as f64;
        eprintln!(
            "FRAME-COST {name} {label}: the scheduler's thread a command buffer: {:.2} sleeps, \
             {:.2} of them on the GPU, woken {:.2} times by commits, {:.2} by completions, {:.2} \
             by host signals, {:.2} at its longest wait",
            per_buffer(|c| c.scheduler_sleeps),
            per_buffer(|c| c.scheduler_gpu_waits),
            per_buffer(|c| c.scheduler_wakes_commit),
            per_buffer(|c| c.scheduler_wakes_completion),
            per_buffer(|c| c.scheduler_wakes_host),
            per_buffer(|c| c.scheduler_sleep_slices),
        );
        eprintln!(
            "FRAME-COST {name} {label}: condition-variable notifies a command buffer (each a \
             system call): {:.2} to the queue waiters, {:.2} to the handler threads, {:.2} on \
             command buffers, {:.2} to events' host waits; {:.2} spared as no thread slept",
            per_buffer(|c| c.wakes_completion_waiters),
            per_buffer(|c| c.wakes_handler_threads),
            per_buffer(|c| c.wakes_command_buffer_waiters),
            per_buffer(|c| c.wakes_event_waiters),
            per_buffer(|c| c.wakes_spared),
        );
        eprintln!(
            "FRAME-COST {name} {label}: a frame's host remap reports {:.1}, {:.1} of them over \
             live imports, touching {:.1} imports; {:.1} imports made by pointer, {:.1} KiB of \
             pages (each taken at its first kernel submission on i915)",
            per_frame(|c| c.host_remaps_reported),
            per_frame(|c| c.host_remaps_over_imports),
            per_frame(|c| c.host_imports_remapped),
            per_frame(|c| c.host_pointer_imports_made),
            per_frame(|c| c.host_pointer_import_bytes_made) / 1024.0,
        );
        eprintln!(
            "FRAME-COST {name} {label}: the device's live VkDeviceMemory after the timed frames \
             (what ANV on i915 lists in each kernel submission, beside its own buffer objects): \
             {} pooled blocks in use, {} empty, {} dedicated, {} host imports, {} dma-buf \
             imports of buffers, {} of images",
            counts.memory_blocks_live,
            counts.empty_memory_blocks_live,
            counts.dedicated_memory_blocks_live,
            counts.host_imports_live,
            counts.dma_buf_memories_live,
            counts.dma_buf_image_memories_live,
        );
        if let (Some(before), Some(after)) = (kernel_before, kernel) {
            let execbuffers = after.execbuffers - before.execbuffers;
            let each = |count: u64| count as f64 / execbuffers.max(1) as f64;
            eprintln!(
                "FRAME-COST {name} {label}: the kernel submissions (scripts/i915-execbuffers): \
                 {:.1} a frame, {:.1} objects each ({:.1} host imports), {:.1} us each in the \
                 ioctl, {:.3} ms a frame",
                execbuffers as f64 / TIMED as f64,
                each(after.objects - before.objects),
                each(after.userptr - before.userptr),
                (after.nanos - before.nanos) as f64 / 1e3 / execbuffers.max(1) as f64,
                (after.nanos - before.nanos) as f64 / 1e6 / TIMED as f64,
            );
        }
        let buffers = (TIMED * buffers) as f64;
        let threads = match (&threads_before, &threads) {
            (Some(before), Some(after)) => spent_by_group(before, after)
                .iter()
                .map(|(group, spent)| {
                    let us = |d: Duration| d.as_secs_f64() * 1e6 / buffers;
                    format!(
                        "{group} {:.1} us ({:.1} user, {:.2} sleeps)",
                        us(spent.all),
                        us(spent.user),
                        spent.sleeps as f64 / buffers
                    )
                })
                .collect::<Vec<_>>()
                .join(", "),
            _ => "not measured".to_owned(),
        };
        let allocations = match (allocations_before, allocations) {
            (Some(before), Some(after)) => {
                format!("{:.2} allocations", (after - before) as f64 / buffers)
            }
            _ => "allocations not counted".to_owned(),
        };
        eprintln!(
            "FRAME-COST {name} {label}: the other threads a command buffer: {threads}; \
             {allocations}"
        );
        if run.announced || work.layer_buffers.len() == TARGETS {
            for (kind, (count, _)) in heard.lock().unwrap().iter() {
                eprintln!(
                    "FRAME-COST {name} {label}: heard {} {count} times",
                    kind.code()
                );
            }
        }
    }
}

/// `count` host imports of `device`, each [`IMPORT_BYTES`] over pages of its own, live until
/// dropped: the buffers, then the pages.
fn imports(device: &Device, count: usize) -> Option<(Vec<Buffer>, Mapping)> {
    if count == 0 {
        return None;
    }
    let pages = Mapping::anonymous(count * IMPORT_BYTES);
    pages.fill(0x5a);
    let buffers = (0..count)
        .map(|i| {
            // SAFETY: the pages are this process's, mapped read and write, and outlive the
            // buffers (the tuple's order).
            let at = unsafe { pages.pointer().add(i * IMPORT_BYTES) };
            // SAFETY: as above.
            unsafe { no_copy(device, at, IMPORT_BYTES, ResourceOptions::shared(), None) }.unwrap()
        })
        .collect();
    Some((buffers, pages))
}

/// How often the encoding thread blocked (in its commits too) and was preempted in `frames`, on
/// average, where it was read.
fn blocks<'a>(frames: impl IntoIterator<Item = &'a Frame>) -> String {
    let (mut count, mut total, mut in_commits) = (0u64, Switches::default(), 0);
    for frame in frames {
        let (Some(switches), Some(commits)) = (frame.switches, frame.commit_blocks) else {
            return "blocks not read".to_owned();
        };
        count += 1;
        total.voluntary += switches.voluntary;
        total.involuntary += switches.involuntary;
        in_commits += commits;
    }
    let each = |n: u64| n as f64 / count.max(1) as f64;
    format!(
        "blocked {:.2} times ({:.2} in commits), preempted {:.2}",
        each(total.voluntary),
        each(in_commits),
        each(total.involuntary)
    )
}

/// The allocations the process's threads but this one made so far, where they are counted.
fn others_allocations() -> Option<u64> {
    Some(alloc::process_allocations()? - alloc::thread_allocations()?)
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs() {
    measure(&[Run::PLAIN, Run::CHECKED, Run::ANNOUNCED]);
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs_plain() {
    measure(&[Run::PLAIN]);
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs_checked() {
    measure(&[Run::CHECKED]);
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs_announced() {
    measure(&[Run::ANNOUNCED]);
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs_repeating() {
    measure(&[Run::REPEATING]);
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs_guest_passes() {
    measure(&[
        Run::shaped(Shape::Passes {
            buffers: 100,
            passes: 1,
        }),
        Run::shaped(Shape::Passes {
            buffers: 1,
            passes: 100,
        }),
        Run::shaped(Shape::NewImage),
    ]);
}

/// Passes whose draws switch between two pipeline states of one layout at every draw
/// ([`Shape::Alternating`]): 40 command buffers of one pass of 16 draws, 600 switches a frame.
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs_alternating() {
    measure(&[Run::shaped(Shape::Alternating {
        buffers: 40,
        draws: 16,
    })]);
}

/// The guest's old image's passes over part of each layer ([`Shape::Part`]): 100 command buffers
/// of one pass of one draw, over 500x300 of a 512x512 layer, loading as [`GUEST_LOADS`]; each
/// clear reaches 512x320, which the inline path takes (`render/inline_first.rs`).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs_part_area() {
    measure(&[Run::shaped(Shape::Part {
        buffers: 100,
        passes: 1,
    })]);
}

/// The guest's composite passes ([`Composite`]), one a command buffer of a plain draw and two
/// fetching ones: the shape of the guest's passes that do not record inline, and the same with a
/// scratch the inline path's action rules take, measured for those rules
/// (`render/inline_first.rs`, `render/inline_splits.rs`).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs_guest_composite() {
    measure(&[false, true].map(|kept| {
        Run::shaped(Shape::Composite {
            buffers: 40,
            draws: 3,
            kept,
        })
    }));
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs_host_pages() {
    measure(&[Run::hosted(Shape::Window), Run::hosted(Shape::NewImage)]);
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs_over_imports() {
    let legs = IMPORTS
        .into_iter()
        .map(|imports| (imports, false))
        .chain([(IMPORTS[1], true)]);
    let runs: Vec<Run> = legs
        .flat_map(|(imports, guest)| {
            [CompletionPolicy::Grouped, CompletionPolicy::Exact]
                .repeat(2)
                .into_iter()
                .map(move |completion| Run::over_imports(imports, completion, guest))
        })
        .collect();
    measure(&runs);
}

/// The guest's events in a run in its shape ([`Workload::guest`]).
struct GuestEvents {
    fifo: SharedEvent,
    paging: SharedEvent,
    /// The last values signalled to each.
    values: Cell<(u64, u64)>,
}

impl GuestEvents {
    /// Before its encoders: every command buffer carries a submission handler, as the consumer's
    /// queue gives each one, and every [`PAGING_WAIT_EVERY`]th of the first queue waits for the
    /// paging event's last value.
    fn begin(&self, command_buffer: &CommandBuffer, queue: usize, pass: usize) {
        command_buffer
            .add_submission_handler(Box::new(|_: &CommandBuffer| {}))
            .unwrap();
        let (_, paging) = self.values.get();
        if queue == 0 && pass % PAGING_WAIT_EVERY == PAGING_WAIT_EVERY / 2 && paging > 0 {
            command_buffer
                .encode_wait_for_event(EventRef::from(&self.paging), paging)
                .unwrap();
        }
    }

    /// After its encoders: every command buffer signals the FIFO event's next value, the second
    /// queue's the paging event's next one first.
    fn end(&self, command_buffer: &CommandBuffer, queue: usize) {
        let (fifo, mut paging) = self.values.get();
        if queue == 1 {
            paging += 1;
            command_buffer
                .encode_signal_event(EventRef::from(&self.paging), paging)
                .unwrap();
        }
        command_buffer
            .encode_signal_event(EventRef::from(&self.fifo), fifo + 1)
            .unwrap();
        self.values.set((fifo + 1, paging));
    }
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs_over_buffers() {
    measure(&[
        Run::over_buffers(LoadAction::Load, StoreAction::Store),
        Run::over_buffers(LoadAction::Load, StoreAction::DontCare),
        Run::over_buffers(LoadAction::Clear, StoreAction::DontCare),
    ]);
}

/// One frame of a plain device, once every pipeline is made and every pool piece done, its
/// allocations counted; and the device's name.
#[cfg(not(m2v_system_alloc))]
fn counted() -> (String, Frame) {
    let cache = ScratchCache::new().unwrap();
    let heard = Heard::default();
    let work = Workload::new(
        device(&cache, false, &heard, CompletionPolicy::default()),
        false,
        Layers::Private,
        (false, false, Shape::Window, false),
    );
    for _ in 0..WARM {
        work.frame();
    }
    let frame = work.frame();
    (work.device.name().to_string(), frame)
}

/// The allocations each step of `steps` of `frame` made on the encoding thread, on average.
#[cfg(not(m2v_system_alloc))]
fn allocations_each(frame: &Frame, steps: StepsOf) -> f64 {
    let steps = steps(frame);
    let allocations: u64 = steps.iter().map(|s| s.allocations.unwrap()).sum();
    allocations as f64 / steps.len() as f64
}

/// A frame's draws allocate at most [`DRAW_ALLOCATIONS`] each on average on the encoding thread
/// ([`DRAW_ALLOCATIONS_DEBUG`] with debug assertions), once every pipeline is made: what the draw
/// path keeps of what it found before (each texture's storage view formats, the pipeline state's
/// routing into the pass's colors and the colors it zeroes and drops, the settles of its latest
/// bind shapes and the shapes themselves, the bind state the device's last ended render encoder
/// left, the format table) is not made again at each draw, and what a draw retains, binds
/// and records it holds and builds without allocating where the object or the list allows. Its
/// passes' ends allocate at most [`PASS_END_ALLOCATIONS`] each: a pass end retains what its
/// commands use as one object, and makes no tile grid where a draw reaches every tile, no sample
/// planes' steps in a pass without planes, and no scrub's list for a texture that is not sparse.
#[cfg(not(m2v_system_alloc))]
#[test]
fn a_frame_s_draws_and_pass_ends_allocate_within_their_bounds() {
    let (device, frame) = counted();
    let each = allocations_each(&frame, |f| &f.draws);
    let bound = match cfg!(debug_assertions) {
        true => DRAW_ALLOCATIONS_DEBUG,
        false => DRAW_ALLOCATIONS,
    };
    eprintln!("FRAME-COST {device}: {each:.2} allocations a draw, at most {bound} kept");
    assert!(
        each <= bound,
        "{each:.2} allocations a draw, more than {bound}"
    );
    let each = allocations_each(&frame, |f| &f.end);
    eprintln!(
        "FRAME-COST {device}: {each:.2} allocations a pass end, at most {PASS_END_ALLOCATIONS} kept"
    );
    assert!(
        each <= PASS_END_ALLOCATIONS,
        "{each:.2} allocations a pass end, more than {PASS_END_ALLOCATIONS}"
    );
}

/// A frame's passes' encoders each begin at most [`PASS_BEGINS`] command buffers, from the
/// opening through the end, once every pipeline is made, and its commits none
/// (`WorkCounts::encoder_command_buffers_begun`, counted in each step: the scheduler's thread
/// submits earlier command buffers meanwhile, and what their submissions begin lands in whichever
/// step runs then). Its batches' submissions, counted over the whole frame through its last
/// command buffer's completion, begin none but the barrier command buffers they record to keep
/// (`command_buffer/kept_barriers.rs`), where the device keeps them: every device whose driver
/// reads no command buffer after its timeline's signal
/// (`RuntimeDriverDefect::ReadsCommandBuffersAfterTheirSignal`), where a submission records its
/// batch's barrier command buffer before its pass unless its batch starts bare, with nothing
/// planned before the pass (`WorkCounts::bare_starts`): its end goes into the pass's own last
/// command buffer (`command_buffer/submit.rs`). It prints the passes whose first instance
/// recorded inline, how many of them split, and their instances after a split that recorded
/// inline too (`WorkCounts::inline_passes`, `WorkCounts::inline_passes_split`,
/// `WorkCounts::inline_split_instances`): each inline pass that did not split begins one fewer,
/// its draws' secondary never begun. By what they were begun for
/// (`WorkCounts::command_buffers_begun_encoders` and kin), the encoders' are each command
/// buffer's one encoder's own and their draws' secondaries, and the device's own submissions begin
/// none. Each command buffer's one batch counts once by the earlier batches of its queue pending
/// as it went (`WorkCounts::batches_with_none_pending` and kin). Each group kept for its queue's
/// earlier work (`WorkCounts::groups_kept_pending`) went within the frame, counted once among the
/// groups by what sent them, and only a kept one counts as sent by a wait
/// (`WorkCounts::groups_sent_for_waits`).
#[test]
fn a_frame_s_passes_and_commits_begin_their_command_buffers_within_their_bound() {
    let cache = ScratchCache::new().unwrap();
    let heard = Heard::default();
    let work = Workload::new(
        device(&cache, false, &heard, CompletionPolicy::default()),
        false,
        Layers::Private,
        (false, false, Shape::Window, false),
    );
    // Every pipeline made, and every pool piece done, before the frame that counts.
    for _ in 0..WARM {
        work.frame();
    }
    let counted = work.device.work_counts();
    let frame = work.frame_reading(Reads {
        work_counts: true,
        ..Reads::default()
    });
    let [open, draws, end, commit] = frame.begun;
    let whole = frame.whole.expect("the work counts read over the frame");
    let passes = BUFFERS as u64;
    let late_reads = (work.device.properties().driver_defects)
        .contains(&RuntimeDriverDefect::ReadsCommandBuffersAfterTheirSignal);
    // The passes whose first instance recorded inline (`render/inline_first.rs`).
    let after = work.device.work_counts();
    let inline = after.inline_passes - counted.inline_passes;
    let split = after.inline_passes_split - counted.inline_passes_split;
    eprintln!(
        "FRAME-COST {}: {:.2} passes a pass with their first instance inline, {:.2} of them split, \
         {:.2} instances after a split inline",
        work.device.name(),
        inline as f64 / passes as f64,
        split as f64 / passes as f64,
        (after.inline_split_instances - counted.inline_split_instances) as f64 / passes as f64,
    );
    let each = |n: u64| n as f64 / passes as f64;
    eprintln!(
        "FRAME-COST {}: command buffers encoders began a pass: {:.2} opening, {:.2} drawing, \
         {:.2} ending, {:.2} committing, the bound {PASS_BEGINS} a pass; the frame's submissions \
         began {:.2} a pass ({:.2} of them barrier command buffers recorded to keep), {:.2} bare \
         starts; every step's command buffers, the submissions' landing in them: {:.2} opening, \
         {:.2} drawing, {:.2} ending, {:.2} committing; the driver reads command buffers after \
         their signal: {late_reads}",
        work.device.name(),
        each(open.encoder),
        each(draws.encoder),
        each(end.encoder),
        each(commit.encoder),
        each(whole.all - whole.encoder),
        each(whole.kept),
        each(whole.bare),
        each(open.all),
        each(draws.all),
        each(end.all),
        each(commit.all),
    );
    let pass = open.encoder + draws.encoder + end.encoder;
    let bound = PASS_BEGINS * passes - (inline - split);
    assert!(
        pass <= bound,
        "{pass} command buffers encoders began in {passes} passes, more than {PASS_BEGINS} each \
         but one fewer for each of the {} inline passes that did not split",
        inline - split
    );
    assert_eq!(
        commit.encoder, 0,
        "command buffers encoders began in commits"
    );
    assert_eq!(
        whole.encoder, pass,
        "command buffers encoders began outside the frame's steps"
    );
    // What they were begun for: each command buffer's one encoder its own once, and the rest
    // its draws' secondaries; no segment, continuation or device submission of its own.
    let begun = |by: fn(&WorkCounts) -> u64| by(&after) - by(&counted);
    let own = begun(|c| c.command_buffers_begun_encoders);
    let secondaries = begun(|c| c.command_buffers_begun_secondaries);
    assert_eq!(own, passes, "encoders' own command buffers begun");
    assert_eq!(
        own + secondaries,
        pass,
        "command buffers encoders began besides their own and their draws' secondaries"
    );
    assert_eq!(
        begun(|c| c.command_buffers_begun_internal),
        0,
        "the device's own submissions began command buffers"
    );
    // Each command buffer's one batch counted once by the earlier batches pending as it went.
    let by_pending = begun(|c| c.batches_with_none_pending)
        + begun(|c| c.batches_with_one_pending)
        + begun(|c| c.batches_with_two_or_three_pending)
        + begun(|c| c.batches_with_four_or_more_pending);
    assert_eq!(
        by_pending, passes,
        "batches counted by the earlier ones pending"
    );
    let ends = begun(|c| c.group_ends_full)
        + begun(|c| c.group_ends_wait)
        + begun(|c| c.group_ends_apart)
        + begun(|c| c.group_ends_first_work)
        + begun(|c| c.group_ends_drained)
        + begun(|c| c.group_ends_checks)
        + begun(|c| c.group_ends_held);
    let (kept, sent) = (
        begun(|c| c.groups_kept_pending),
        begun(|c| c.groups_sent_for_waits),
    );
    eprintln!(
        "FRAME-COST {}: {ends} groups went, {kept} kept for their queue's earlier work, {sent} \
         of those sent by a wait",
        work.device.name()
    );
    assert!(
        kept <= ends && sent <= kept,
        "{kept} groups kept and {sent} sent by a wait, of {ends} that went"
    );
    // The frame's batches' submissions, on whichever thread made them.
    let own = whole.all - whole.encoder - whole.kept;
    match late_reads {
        true => assert_eq!(
            own + whole.bare,
            passes,
            "a submission's barrier command buffer before its pass, or a bare start"
        ),
        false => assert_eq!(
            own, 0,
            "{own} command buffers begun by {passes} command buffers' submissions, not kept"
        ),
    }
}

/// A frame's pass opens (`render_command_encoder`) allocate at most [`OPEN_ALLOCATIONS`] each on
/// average on the encoding thread ([`FIRST_LOCK_ALLOCATIONS`] more on Apple's platforms), once
/// every pipeline is made: an encoder takes the compatible render pass the device keeps for its
/// shape instead of making one.
#[cfg(not(m2v_system_alloc))]
#[test]
fn a_frame_s_pass_opens_allocate_within_their_bound() {
    let (device, frame) = counted();
    let each = allocations_each(&frame, |f| &f.open);
    let bound = OPEN_ALLOCATIONS + FIRST_LOCK_ALLOCATIONS;
    eprintln!("FRAME-COST {device}: {each:.2} allocations a pass open, at most {bound} kept");
    assert!(
        each <= bound,
        "{each:.2} allocations a pass open, more than {bound}"
    );
}

/// A checked frame's commits allocate at most [`COMMIT_ALLOCATIONS`] each on average on the
/// encoding thread ([`FIRST_LOCK_ALLOCATIONS`] more on Apple's platforms), and the process's other
/// threads at most [`SUBMIT_ALLOCATIONS`] per command buffer (the scheduler thread's pass, the
/// check, the submission, the completion), once every pipeline is made. Every commit hands its
/// batches to the scheduler's thread (`WorkCounts::commits_handed_off`), which the test checks: no
/// commit submits, so which thread submits moves nothing between the two counts.
/// The other threads' count is the process's: the test needs a process of its own, as nextest
/// runs each test in.
#[cfg(not(m2v_system_alloc))]
#[test]
fn a_frame_s_commits_allocate_within_their_bound() {
    let cache = ScratchCache::new().unwrap();
    let heard = Heard::default();
    let work = Workload::new(
        device(&cache, false, &heard, CompletionPolicy::default()),
        true,
        Layers::Private,
        (false, false, Shape::Window, false),
    );
    // Every pipeline made, and every pool piece done, before the frames that count.
    for _ in 0..WARM {
        work.frame();
    }
    const FRAMES: usize = 4;
    let before = others_allocations().unwrap();
    // The work counts read around each step tell which commits handed their batches off.
    let frames: Vec<Frame> = (0..FRAMES)
        .map(|_| {
            work.frame_reading(Reads {
                work_counts: true,
                ..Reads::default()
            })
        })
        .collect();
    let others = others_allocations().unwrap() - before;
    let commits: Vec<u64> = (frames.iter())
        .flat_map(|frame| &frame.commit)
        .map(|step| step.allocations.unwrap())
        .collect();
    let handed = (frames.iter())
        .flat_map(|frame| &frame.handed)
        .filter(|&&handed| handed)
        .count();
    let buffers = commits.len() as f64;
    let each = commits.iter().sum::<u64>() as f64 / buffers;
    let other = others as f64 / buffers;
    let bound = COMMIT_ALLOCATIONS + FIRST_LOCK_ALLOCATIONS;
    eprintln!(
        "FRAME-COST {}: {each:.2} allocations a commit ({handed} of {} handed their batches to \
         the scheduler's thread), at most {bound} kept; {other:.2} on the other threads a \
         command buffer, at most {SUBMIT_ALLOCATIONS} kept",
        work.device.name(),
        commits.len(),
    );
    assert_eq!(
        handed,
        commits.len(),
        "every commit hands its batches to the scheduler's thread"
    );
    assert!(
        each <= bound,
        "{each:.2} allocations a commit, more than {bound}"
    );
    assert!(
        other <= SUBMIT_ALLOCATIONS,
        "{other:.2} allocations on the other threads a command buffer, more than \
         {SUBMIT_ALLOCATIONS}"
    );
}

/// Draws in a pass of [`frame_costs_resident`].
const RESIDENT_DRAWS: usize = 32;

/// The plain device's draws that reach a buffer through the resident set alone, as the consumer's
/// guest's draws reach its argument buffers' buffers: [`RESIDENT_DRAWS`] draws a pass of
/// `render_arguments`' [`ORDERED`] case (a fragment function updating the buffer its argument
/// buffer points to, made resident with `use_resources`), each of the same uses, a pass a command
/// buffer, 20 untimed and 100 timed frames: a draw's user-space instructions and wall time, and
/// how many of a pass's draws its encoder noted as before (`WorkCounts::commands_noted_as_before`).
/// A device that cannot make the case's pipeline prints its refusal.
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs_resident() {
    let cache = ScratchCache::new().unwrap();
    let heard = Heard::default();
    let device = device(&cache, false, &heard, CompletionPolicy::default());
    let case = public_render(ORDERED);
    let state = match render_pipeline_state(&device, &case.inputs()) {
        Ok(state) => state,
        Err(error) => {
            println!(
                "FRAME-COST {}: resident draws refused: {error}",
                device.name()
            );
            return;
        }
    };
    let render = &case.render;
    let members = Members::of(&device, render.fragment.as_ref().unwrap());
    let target = &render.color_targets[&0];
    let [width, height] = target.dimensions;
    let format = pixel_format(target.format);
    let mut descriptor = TextureDescriptor::texture_2d(format, width.into(), height.into(), false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    let texture = device.new_texture(&descriptor).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(texture);
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    let DrawCall::Primitives {
        vertex_start,
        vertex_count,
        instance_count,
        base_instance,
    } = render.draw
    else {
        panic!("{ORDERED}: a draw of primitives");
    };
    let &Assembly::Primitives(primitive) = &render.assembly else {
        panic!("{ORDERED}: a draw of primitives");
    };
    let queue = device.new_command_queue().unwrap();
    let resources = members.resources();
    let stages = RenderStages::VERTEX | RenderStages::FRAGMENT;
    let usage = ResourceUsage::READ | ResourceUsage::WRITE;
    let frame = |counted: bool, draws: &mut Vec<Step>| {
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&state).unwrap();
        for (&index, owner) in &members.owners {
            encoder.set_fragment_buffer(Some(owner), 0, index).unwrap();
        }
        encoder.use_resources(&resources, usage, stages).unwrap();
        for _ in 0..RESIDENT_DRAWS {
            let ((), drawn) = step(counted, || {
                encoder
                    .draw_primitives(
                        metal_primitive_type(primitive),
                        vertex_start,
                        vertex_count,
                        instance_count,
                        base_instance,
                    )
                    .unwrap()
            });
            draws.push(drawn);
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer
    };
    let mut draws = Vec::new();
    for _ in 0..20 {
        frame(false, &mut draws)
            .wait_until_completed_for(WAIT)
            .unwrap();
    }
    draws.clear();
    let noted = || device.work_counts().commands_noted_as_before;
    let before = noted();
    const FRAMES: usize = 100;
    for _ in 0..FRAMES {
        let last = frame(true, &mut draws);
        last.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(last.status(), CommandBufferStatus::Completed);
    }
    let noted = (noted() - before) as f64 / FRAMES as f64;
    let walls: Vec<f64> = draws.iter().map(|d| d.wall.as_secs_f64() * 1e6).collect();
    let [wall, wall_p90] = percentiles(walls);
    let instructions: Vec<f64> = draws
        .iter()
        .filter_map(|d| d.instructions)
        .map(|n| n as f64)
        .collect();
    let counted = match instructions.is_empty() {
        true => "instructions not counted".to_owned(),
        false => {
            let mean = instructions.iter().sum::<f64>() / instructions.len() as f64;
            let [p50, _] = percentiles(instructions);
            format!("{mean:.0} user-space instructions each (p50 {p50:.0})")
        }
    };
    println!(
        "FRAME-COST {}: resident draws, {RESIDENT_DRAWS} a pass of the same uses: draw: \
         {wall:.2} us (p50; p90 {wall_p90:.2}); {counted}; {noted:.1} of a pass's draws noted as \
         before",
        device.name()
    );
}
