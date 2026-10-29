//! The host A/B benchmark: the same Metal workloads run two ways on one machine, natively on
//! Metal (macOS only, through objc2-metal) and through this runtime on the Vulkan device its
//! loader finds (on a Mac, MoltenVK), from one metallib that `xcrun metal` compiles from
//! `host_ab.metal`. The runtime takes it through `new_library_with_data`, so its functions run as
//! the translator's SPIR-V.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example host_ab -- [options]
//! ```
//!
//! Without `--side`, each round runs one process per side, native first, then prints a table of
//! each metric: the median of the rounds' medians per side, the ratio runtime / native, and the
//! spread of the rounds. `--side native|runtime` runs one side in this process and prints its
//! result lines (for a profiler).
//!
//! Workloads (`--only a,b,...`; all by default):
//!
//! - `encode`: CPU cost of encoding. A render pass of `--draws` small textured draws into a
//!   512x512 `BGRA8Unorm` target, `static` (nothing set between draws), `bind` (each draw another
//!   vertex buffer and offset, and another fragment texture), `pipeline` (as `bind`, and the
//!   pipeline state switched between two) and `passes` (as `pipeline`, in render passes of three
//!   draws, as a window server draws), and a compute encoder of as many single-threadgroup
//!   dispatches, `static`, `offsets` (both buffers' offsets each dispatch) and `churn` (the
//!   pipeline state and both buffers' offsets each dispatch). Per draw and dispatch: the encoding thread's wall time from the command buffer's
//!   creation through `endEncoding`, and the whole process's CPU time until the wait returned.
//!   Then `dispatch_resident_*`: as many dispatches of the tiny kernel after one
//!   `useResources:count:usage:` of 0, 1, 256 or 4096 buffers, or of one buffer 4096 times
//!   (`same_4096`); per variant the `useResources` call, and per dispatch the encoding thread's
//!   wall time after it and the process's CPU time.
//! - `fixed`: a command buffer's creation, an encoder of one command each (render, compute,
//!   blit), and the commit call of a command buffer of 30 such encoders.
//! - `calls`: the frames of a guest as reims-vgpu's `frame_cost` measures them, call by call: one
//!   command buffer of long encoders (`wide`: a render pass of 512 draws, two compute encoders of
//!   64 dispatches, a blit encoder of 32 copies) and 64 command buffers of short ones (`many`:
//!   16 draws, one dispatch each, one copy), each with a completed handler and committed, then
//!   each handler awaited and the command buffer waited for. Each draw sets two vertex buffers,
//!   the second's offset moving, then a fragment texture and sampler or a fragment buffer at a
//!   moving offset, by the pipeline state, which changes every eighth draw; the second compute
//!   encoder makes 16 buffers resident first. Per step (`command_buffer`, `render_encoder`,
//!   `set_state`, ..., `draw`, `end_render`, ..., `handler_commit`): the mean a call of each of
//!   `--timed` frames, in ns, after 5 frames not timed; `*_frame_calls`, the frame's timed calls
//!   summed; and `*_allocs`, the heap allocations a call of the step made on the encoding thread
//!   over one more frame (`allocs`); `*_frame_wall`, the frame from its first call until its last
//!   wait returned. `--variants wide` or `many` runs one shape alone. Only where named,
//!   `resident_0`, `_1`, `_256` and `_4096`: a render encoder's 512 draws, nothing changed between
//!   them, after one `useResources:count:usage:stages:` of 0, 1, 256 or 4096 buffers; the call,
//!   each draw after it, and the command buffer to its wait's return.
//! - `latency`: commit to `waitUntilCompleted`'s return, and commit to the completed handler, of
//!   a command buffer that is empty, one blit fill, one dispatch, or one draw; one in flight.
//! - `gpu`: a full-screen heavy fragment (`heavy_fragment`, 2048x2048 `RGBA16Float`) and a heavy
//!   kernel (`heavy_kernel`, 4096 threadgroups of 256), one command buffer in flight: commit to
//!   the wait's return, and native's `GPUEndTime - GPUStartTime`. `--heavy-iterations` sets
//!   the fragment's loop, and 16 times it the kernel's. Then each feature kernel (`k_*`, one
//!   feature of the heavy kernel each: fma, sin, half, constant buffer, device buffer, texture
//!   read as `half` and as `float`, threadgroup float, threadgroup half) on the kernel's grid
//!   and loop. With both sides, each round also compares every kernel's output buffer after its
//!   last run between the sides, and the last table counts the 32-bit words that differ.
//! - `signals`: a command buffer of one tiny fill with a completed handler, committed alone and
//!   waited for, as the consumer's isolated execution is: `plain`; `scheduled_handler`, with a
//!   scheduled handler that sets a shared event's value (how a program without
//!   `encodeSignalEventScheduled:value:` signals as it is scheduled); `scheduled_signal`, with that
//!   private call (the runtime's `encode_signal_event_scheduled`); and the runtime's alone,
//!   `checked`, with a submission check that answers none (`set_submission_check`). Per command
//!   buffer: commit to the completed handler, commit to the wait's return, the process's CPU time,
//!   and on the runtime's side its work counts (`rt_*`: handler-thread and scheduler wakes, and
//!   the command buffers scheduled where their last batch was submitted, the scheduler's passes).
//!   Then bursts of 64 such command buffers committed back to back, `plain` and `checked`, to
//!   the last one's completion, per command buffer. Then a shared event listener made and dropped
//!   unused, one made, and its first notification, asked for a value already reached, from the
//!   ask to the block's start.
//! - `latency_split` (with `latency`): the same command buffers' host-clock timeline, commit to
//!   `kernelStartTime` to `kernelEndTime` to `GPUStartTime` to `GPUEndTime` to the wait's return.
//!   The runtime's GPU times on MoltenVK are the host's (its docs): its GPU start is its kernel
//!   end and its GPU end the moment it saw the command buffer complete.
//! - `pipeline`: the library, then each of four states' creation and first use, then each made
//!   again in the process. Each round runs it in two processes per side over a metallib of a
//!   fresh `NONCE` (`host_ab.metal`): `cold_*` (no cache holds its functions) and `warm_*` (a
//!   second process over the same runtime shader cache directory, after Metal's own caches took
//!   the first).
//! - `frame`: a window server's frame of heap textures: eight `BGRA8Unorm` textures of 256x256
//!   (`allowGPUOptimizedContents` off) made at their offsets of a `Placement` heap of `Shared`
//!   storage, each rendered by a pass of
//!   three textured draws, then each sampled into the target by one draw of a last pass, the
//!   command buffer committed and waited for, and the textures dropped. Per frame: the encoding
//!   thread's wall time from the first texture's creation through the commit, the textures'
//!   creation alone, their drop, and the process's CPU time until the drop returned. The
//!   runtime's alone (`frame_no_copy_*`): the same over a heap of host pages
//!   (`new_heap_with_bytes_no_copy`), as a guest's heap is.
//! - `many_draws`: a frame of `--many-draws` small draws, shaped as m2vbench's draws mode: one
//!   render pass into a 1920x1080 `BGRA8Unorm` target with a cleared `Depth32Float` attachment,
//!   each draw a textured rectangle of its own uniforms (a vertex and a fragment buffer at the
//!   draw's offset of one of four buffers), two fragment textures of sixteen, 16 bytes of
//!   `setVertexBytes`, alternately indexed and not; the pipeline state (four) every 8 draws, the
//!   depth-stencil state (two) every 16, the fragment sampler (two) every 32 and the cull mode
//!   every 64. Per frame: the encoding thread's wall time from the command buffer's creation
//!   through `endEncoding`, the commit call's, from its return to the wait's, the whole frame's,
//!   the process's CPU time, the encoding thread's CPU time through the commit's return, and
//!   native's GPU time. Where the side makes a heap over host pages (the runtime's), the frames
//!   again with the four uniform buffers no-copy buffers over host pages that such a heap lies
//!   over too, as a guest's buffers lie in its memory beside its heaps (the `pages_` rows).
//! - `layers`: a frame shaped as m2vbench's composite mode (CoreAnimation-like layers): one render
//!   pass into a 1920x1080 `BGRA8Unorm` target, a wallpaper, then 800 textured rounded-corner
//!   layers back to front, each one draw of a strip whose uniforms are its `setVertexBytes` and
//!   whose texture (of 48) is bound before it, with a soft shadow's draw under every eighth (its
//!   own pipeline state, set before it and back after): 901 draws. Per frame as `many_draws`.
//! - `drag`: a frame shaped as the guest's window drag (a macOS guest's WindowServer, measured in
//!   a VM's work counts): 40 command buffers, each committed, of which the first 31 hold a render
//!   pass of three textured draws into one of eight 256x256 `BGRA8Unorm` window surfaces in a
//!   `Placement` heap, and the first 22 a dispatch of the tiny kernel; the last waited for. Per
//!   frame: the encoding thread's wall time through the last commit, the whole frame's, the
//!   encoding thread's CPU time and the process's. The runtime's alone (`drag_no_copy_*`): the
//!   same over a heap of host pages, as a guest's heap is, and (`drag_heap_surfaces_*`) again
//!   with textures Metal may optimize, each pass's draws sampling three other surfaces, as a
//!   compositor's do, the frame's first command buffer uploading a window's contents into every
//!   other surface in one blit encoder first. Then (`drag_surfaces_*`) the surfaces as textures
//!   over host pages, a no-copy buffer each, their rows padded, as a guest's window surfaces
//!   are, composited and uploaded into alike.
//! - `memory`: `newBufferWithLength:` and `newBufferWithBytes:` of 4 KiB, 1 MiB and 64 MiB,
//!   `newBufferWithBytesNoCopy:` of 1 MiB and 64 MiB, each released before the next is made;
//!   `newBufferWithLength:` of 1 MiB and 64 MiB held (`held`: 48 and 6 of them, none made in
//!   memory an earlier one used), and the first touch of all of each one's bytes: a CPU write
//!   (`first_write_held`), or, of a second set, a blit fill from commit to the wait's return
//!   (`first_fill_held`) and that fill again (`second_fill_held`); and a 1024x1024 `RGBA8Unorm`
//!   `Shared` texture's `replaceRegion` and `getBytes`.
//! - `churn`: each frame makes eight 1280x800 `Private` render targets, clears and releases them,
//!   the same size each frame (`same`) or one of 64 widths in turn (`varied`); and sixteen 64-KiB
//!   buffers, filled and released (`buffers`) (`workloads::churn`).
//! - `compositing`: frames of the consumer's compositing shape, paced at 60 Hz: 20 command
//!   buffers a frame, and 10 new textures made in a `Placement` heap of tracked `Shared` storage
//!   (as the guest makes heap textures each frame): 10 command buffers each clear one new
//!   texture, made just before it, and draw 2 quads into it, 8 each draw 8 blended quads into one
//!   of two 1024x1024 layers from 4 new textures and the other layer (each depending on the
//!   command buffers before it), one dispatches 15 small kernels and fills a buffer, and the last
//!   draws both layers and 4 new textures onto a screen texture. Per frame: the wall time from the
//!   first command buffer's creation to the last one's completion (its median, its 99th
//!   percentile and the frames over 16.7 ms), the encoding thread's time to encode and commit,
//!   the process's CPU time, and on the runtime's side its work counts per frame (`rt_*`:
//!   `vkQueueSubmit` calls, the `VkSubmitInfo`s they held, each a Metal command buffer under
//!   MoltenVK, and the rest).
//! - `burst`: a window server's frames paced at 60 Hz, each period of 3 s ending in a burst of
//!   1 s, shaped as a guest's bursts in a VM's work counts (`workloads::burst`). Calm, a frame
//!   makes 15 `Private` render targets of the same sizes each frame and 10 heap textures, clears
//!   and draws into each over 20 command buffers, draws the heap textures onto a screen texture,
//!   and waits for its command buffers. In a burst it also makes 45 render targets of widths made
//!   once, its heap textures take 128 sides in turn, it spreads the work over 60 command buffers,
//!   and it leaves them in flight until the next frame starts. Calm and burst frames apart: the
//!   interval from a frame's start to the next one's (its median, its 99th percentile, and those
//!   over 16.7 and 25 ms), the encoding thread's time to encode and commit, the process's CPU
//!   time, and on the runtime's side its work counts per frame (`rt_*`).
//! - `icb`: indirect command buffers a kernel writes. Per frame, a kernel encodes 16, 64, 256 or
//!   4096 compute commands into an indirect command buffer (one a thread: one of 1 to 21 live
//!   states made with `supportIndirectCommandBuffers`, of which the commands name four at most,
//!   its own 64-word window of a buffer, one threadgroup of 64 threads, no barrier), and a
//!   second compute encoder, serial or concurrent, executes them
//!   (`executeCommandsInBuffer:withRange:`), as a GPU culling pass hands its kept work on. Per
//!   frame: the wall time from the command buffer's creation to the wait's return, the encoding
//!   thread's through the commit, the process's CPU time, and on the runtime's side its queue
//!   submissions, barriers, dispatches and indirect dispatches (`rt_*`). The first frame's output
//!   is checked (`workloads::icb`).
//! - `icb_draws`: a frame of Apple's "Encoding Indirect Command Buffers on the GPU": a blit
//!   resets an indirect command buffer of 65,536 draws whose commands inherit the pipeline state,
//!   a kernel culls 65,536 objects on a grid a little wider than the view and encodes a command for
//!   each visible one (three vertex buffers and a draw of its mesh's 18 vertices whose base
//!   instance is its index), and a render pass into a 1920x1080 `BGRA8Unorm` target executes them
//!   all. Per frame, as `many_draws`; `icb_draws_any_type_*`: the same, the kernel reading the
//!   draws' primitive type (triangles) from a buffer, where the runtime cannot tell the type from
//!   the kernel's code and so reads the records on the host. On the runtime's side, the commands
//!   drawn on the GPU alone, the records read on the host, and the draws and pipeline barriers
//!   recorded per frame (`rt_*`).
//! - `percall`: the device API's CPU cost per call, Metal against the runtime: one group a call,
//!   each timed alone, warm, with no making inside a timed window, and attributed three ways (the
//!   calling thread, our own other threads, and what the caller blocks for). It prints its own
//!   table after the metric table (`percall.rs`).
//! - `icb_cpu`: the CPU's encoding of indirect commands, each call's time: of 4096 commands of a
//!   buffer of dispatches and one of draws, each setter (pipeline state, buffer, dispatch or draw,
//!   barrier) on every command in turn, then Apple's private getters of a buffer's address and of
//!   the barrier (the runtime's read-back of them), each call taking its command first
//!   (`workloads::icb_cpu`).
//!
//! - `provoking`: an OpenGL guest's flat-shaded draws, which Apple's GL draws at
//!   `setProvokingVertexMode:` 1 (the last vertex): one render pass into a 512x512 `BGRA8Unorm`
//!   target of `--draws` draws of two small triangles each, six vertices of their own, under a
//!   pipeline whose fragment function reads a flat input; direct, indexed (16-bit, each draw its
//!   own six indices of one buffer), indexed strips (each draw a strip of four of its six
//!   vertices, its own four indices of another buffer), indirect and indexed indirect, each at
//!   mode 0 and at mode 1.
//!   Per draw: the encoding thread's wall time through `endEncoding`, the process's CPU time, and
//!   the frame from the command buffer's creation to the wait's return; and the commit to the
//!   wait's return. On the runtime's side, its draws through last-vertex pipelines, its listed
//!   draws, dispatches and pipeline barriers per frame (`rt_*`).
//!
//! Options: `--rounds N` (3), `--draws N` (2000), `--timed N` (30), `--latency N` (200),
//! `--heavy-iterations N` (64), `--heavy-runs N` (15), `--many-draws N` (6000), `--frames N`
//! (120, the compositing workload's timed frames), `--scratch DIR` (metallibs and the runtime's
//! shader caches; a directory under the OS's temporary directory by default, removed at the end),
//! `--metallib PATH` (skip the compile; not with `pipeline`), `--cache DIR` (the runtime's shader
//! cache, with `--side`), `--variants a,b,...` (the encode workload's variants alone:
//! `draw_static`, `draw_bind`, `draw_pipeline`, `draw_passes`, `dispatch_static`,
//! `dispatch_offsets`, `dispatch_churn`, `dispatch_resident_0`, `_1`, `_256`, `_4096`,
//! `_same_4096`; for a profile of one; and the memory workload's parts alone: `create`,
//! `no_copy`, `held_1m`, `held_64m`, `texture`), `--announce-each a,b,...` (the runtime's device
//! announces these degradation kinds, by code with or without its `metal2vulkan-degrade-` prefix,
//! or `all` of them, at every occurrence, `DeviceOptions::announce_each`, as a consumer that hears
//! some kinds at each call opens it), `--sink-least verbose|info|warning|error` (the runtime's
//! sink hears no other message below it, `MessageSink::with_least_severity`; `verbose` by
//! default).
//!
//! On a Mac, the runtime's device opens the Vulkan loader (`/opt/homebrew/lib` among the places
//! it looks) and MoltenVK's `MVK_CONFIG_*` variables apply. Run timings on an otherwise idle GPU.

mod allocs;
mod gpu;
#[cfg(target_os = "macos")]
mod native;
mod percall;
mod runtime;
mod vkfloor;
mod workloads;

use gpu::Gpu;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;
use workloads::Sizes;

const WORKLOADS: [&str; 20] = [
    "encode",
    "fixed",
    "calls",
    "latency",
    "gpu",
    "pipeline",
    "memory",
    "frame",
    "many_draws",
    "layers",
    "churn",
    "compositing",
    "drag",
    "burst",
    "icb",
    "icb_draws",
    "signals",
    "provoking",
    "icb_cpu",
    "percall",
];

struct Options {
    side: Option<String>,
    only: Vec<String>,
    rounds: usize,
    sizes: Sizes,
    scratch: Option<PathBuf>,
    metallib: Option<PathBuf>,
    cache: Option<PathBuf>,
    /// Where a side writes the gpu workload's kernel outputs (`--outputs`).
    outputs: Option<PathBuf>,
    /// The encode and memory workloads' variants to run (`--variants`); all when empty.
    variants: Vec<String>,
    /// The kinds the runtime's device announces at every occurrence (`--announce-each`).
    announce_each: Vec<metal2vulkan_runtime::Degradation>,
    /// The least severity of the messages the runtime's sink hears (`--sink-least`).
    sink_least: metal2vulkan_runtime::MessageSeverity,
    /// Arguments passed on to each child.
    passed: Vec<String>,
}

fn options() -> Options {
    let mut options = Options {
        side: None,
        only: WORKLOADS.iter().map(|w| w.to_string()).collect(),
        rounds: 3,
        sizes: Sizes {
            draws: 2000,
            timed: 30,
            latency: 200,
            heavy_iterations: 64,
            heavy_runs: 15,
            many_draws: 6000,
            frames: 120,
        },
        scratch: None,
        metallib: None,
        cache: None,
        outputs: None,
        variants: vec![],
        announce_each: vec![],
        sink_least: metal2vulkan_runtime::MessageSeverity::Verbose,
        passed: vec![],
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || {
            args.next()
                .unwrap_or_else(|| panic!("{flag} takes a value"))
        };
        let mut number = |passed: &mut Vec<String>| {
            let v = value();
            passed.extend([flag.clone(), v.clone()]);
            v.parse::<usize>()
                .unwrap_or_else(|_| panic!("{flag}: a number"))
        };
        match flag.as_str() {
            "--side" => options.side = Some(value()),
            "--only" => options.only = value().split(',').map(str::to_owned).collect(),
            "--rounds" => options.rounds = value().parse().expect("--rounds: a number"),
            "--draws" => options.sizes.draws = number(&mut options.passed),
            "--timed" => options.sizes.timed = number(&mut options.passed),
            "--latency" => options.sizes.latency = number(&mut options.passed),
            "--heavy-iterations" => {
                options.sizes.heavy_iterations = number(&mut options.passed) as u32
            }
            "--heavy-runs" => options.sizes.heavy_runs = number(&mut options.passed),
            "--many-draws" => options.sizes.many_draws = number(&mut options.passed),
            "--frames" => options.sizes.frames = number(&mut options.passed),
            "--scratch" => options.scratch = Some(PathBuf::from(value())),
            "--metallib" => options.metallib = Some(PathBuf::from(value())),
            "--cache" => options.cache = Some(PathBuf::from(value())),
            "--outputs" => options.outputs = Some(PathBuf::from(value())),
            "--variants" => {
                let v = value();
                options.variants = v.split(',').map(str::to_owned).collect();
                options.passed.extend(["--variants".to_owned(), v]);
            }
            "--announce-each" => {
                let v = value();
                options.announce_each = match v.as_str() {
                    "all" => metal2vulkan_runtime::Degradation::ALL.to_vec(),
                    _ => v.split(',').map(degradation).collect(),
                };
                options.passed.extend(["--announce-each".to_owned(), v]);
            }
            "--sink-least" => {
                let v = value();
                use metal2vulkan_runtime::MessageSeverity as S;
                options.sink_least = match v.as_str() {
                    "verbose" => S::Verbose,
                    "info" => S::Info,
                    "warning" => S::Warning,
                    "error" => S::Error,
                    other => panic!("--sink-least: verbose, info, warning or error, not {other}"),
                };
                options.passed.extend(["--sink-least".to_owned(), v]);
            }
            "--help" | "-h" => {
                eprintln!("see the module docs of examples/host_ab/main.rs");
                std::process::exit(0);
            }
            other => panic!("unknown option {other}"),
        }
    }
    for workload in &options.only {
        assert!(
            WORKLOADS.contains(&workload.as_str()),
            "unknown workload {workload}"
        );
    }
    options
}

/// The degradation kind of `code`, with or without its `metal2vulkan-degrade-` prefix.
fn degradation(code: &str) -> metal2vulkan_runtime::Degradation {
    let prefixed = match code.starts_with("metal2vulkan-degrade-") {
        true => code.to_owned(),
        false => format!("metal2vulkan-degrade-{code}"),
    };
    metal2vulkan_runtime::Degradation::from_code(&prefixed)
        .unwrap_or_else(|| panic!("--announce-each: no degradation kind {code}"))
}

fn main() {
    let options = options();
    match options.side.as_deref() {
        None => parent(&options),
        Some(side) => child(side, &options),
    }
}

/// One side in this process: open its device, run the selected workloads, print result lines.
fn child(side: &str, options: &Options) {
    let metallib = options.metallib.as_deref().expect("--metallib with --side");
    let t = Instant::now();
    match side {
        #[cfg(target_os = "macos")]
        "native" => {
            let native = native::Native::open();
            run(&native, t, metallib, options);
        }
        "runtime" => {
            let cache = options
                .cache
                .clone()
                .unwrap_or_else(|| scratch_default().join("cache"));
            let runtime =
                runtime::Runtime::open(&cache, &options.announce_each, options.sink_least);
            run(&runtime, t, metallib, options);
            let c = runtime.device().work_counts();
            eprintln!(
                "runtime work counts: queue_submissions {} command_buffers_submitted {} \
                 pipeline_barriers {} draws {} dispatches {} variant_pipelines_made {} \
                 device_memory_allocated {} host_pointer_imports_made {} \
                 fresh_page_buffers_made {} fresh_page_bytes {} new_buffer_bytes_cleared {} \
                 requirement_images_made {} arena_pools_made {} owned_descriptor_pools {} \
                 draws_settled {} stage_tables_read {} graphics_pipelines_looked_up {} \
                 draws_pushing_set_zero {} set_zero_writes_pushed {} set_zero_sets_reused {} \
                 set_zero_sets_found {} draw_shapes_reused {} \
                 completed_waits {} bare_starts {} submit_infos {} finales_folded {} \
                 finales_apart {} pending_none {} pending_one {} pending_two_three {} \
                 pending_four_more {}",
                c.queue_submissions,
                c.command_buffers_submitted,
                c.commands_recorded.cmd_pipeline_barrier,
                c.commands_recorded.cmd_draw,
                c.commands_recorded.cmd_dispatch,
                c.variant_pipelines_made,
                c.device_memory_allocated,
                c.host_pointer_imports_made,
                c.fresh_page_buffers_made,
                c.fresh_page_bytes,
                c.new_buffer_bytes_cleared,
                c.requirement_images_made,
                c.arena_pools_made,
                c.owned_descriptor_pools,
                c.draws_settled,
                c.stage_tables_read,
                c.graphics_pipelines_looked_up,
                c.draws_pushing_set_zero,
                c.set_zero_writes_pushed,
                c.set_zero_sets_reused,
                c.set_zero_sets_found,
                c.draw_shapes_reused,
                c.completed_waits,
                c.bare_starts,
                c.submit_infos,
                c.finales_folded,
                c.finales_apart,
                c.batches_with_none_pending,
                c.batches_with_one_pending,
                c.batches_with_two_or_three_pending,
                c.batches_with_four_or_more_pending,
            );
            eprintln!(
                "runtime scheduler: scheduler_sleeps {} scheduler_wakes_commit {}",
                c.scheduler_sleeps, c.scheduler_wakes_commit,
            );
            eprintln!(
                "runtime submit stalls: idle_submits {} submit_stalls_idle {} \
                 submit_stalls_on_work {} submits_behind_idle_stalls {} \
                 wait_behind_idle_stall_longest_nanos {}",
                c.idle_submits,
                c.submit_stalls_idle,
                c.submit_stalls_on_work,
                c.submits_behind_idle_stalls,
                c.wait_behind_idle_stall_longest_nanos,
            );
            eprintln!("runtime commands recorded: {:?}", c.commands_recorded);
            eprintln!(
                "runtime notices: {} (bound notices made {}), other messages heard {}",
                runtime::NOTICES.load(std::sync::atomic::Ordering::Relaxed),
                c.bound_notices_made,
                runtime::MESSAGES.load(std::sync::atomic::Ordering::Relaxed)
            );
        }
        other => panic!("--side native|runtime, not {other}"),
    }
}

fn run<G: Gpu>(g: &G, opened: Instant, metallib: &Path, options: &Options) {
    workloads::emit(
        "meta",
        "device_open",
        "ms",
        &mut [opened.elapsed().as_secs_f64() * 1e3],
    );
    eprintln!("{}", g.label());
    let sizes = &options.sizes;
    if options.only.iter().any(|w| w == "pipeline") {
        workloads::pipeline(g, metallib, sizes);
    }
    let rest: Vec<&str> = options
        .only
        .iter()
        .map(String::as_str)
        .filter(|w| *w != "pipeline")
        .collect();
    if rest.is_empty() {
        return;
    }
    let t = Instant::now();
    let library = g.library(metallib);
    let scene = workloads::Scene::new(g, &library, sizes);
    workloads::emit(
        "meta",
        "scene_setup",
        "ms",
        &mut [t.elapsed().as_secs_f64() * 1e3],
    );
    for workload in rest {
        match workload {
            "encode" => workloads::encode(g, &scene, sizes, &options.variants),
            "fixed" => workloads::fixed(g, &scene, sizes),
            "calls" => workloads::calls(g, &scene, &library, sizes, &options.variants),
            "latency" => workloads::latency(g, &scene, sizes),
            "gpu" => workloads::gpu(g, &scene, sizes, options.outputs.as_deref()),
            "memory" => workloads::memory(g, sizes, &options.variants),
            "frame" => workloads::frame(g, &scene, sizes),
            "many_draws" => workloads::many_draws(g, &library, sizes),
            "layers" => workloads::layers(g, &library, sizes),
            "churn" => workloads::churn(g, sizes),
            "compositing" => workloads::compositing(g, &scene, sizes),
            "drag" => workloads::drag(g, &scene, sizes),
            "burst" => workloads::burst(g, &scene),
            "icb" => workloads::icb(g, &library, sizes),
            "icb_draws" => workloads::icb_draws(g, &library, sizes),
            "signals" => workloads::signals(g, &scene, sizes),
            "provoking" => workloads::provoking(g, &library, sizes),
            "icb_cpu" => workloads::icb_cpu(g, &library, sizes),
            "percall" => percall::percall(g, &scene, &library, sizes),
            _ => unreachable!(),
        }
    }
}

/// The host clock Metal's and the runtime's command buffer times count: `CLOCK_UPTIME_RAW`
/// (`mach_absolute_time`) on macOS.
pub fn host_seconds() -> f64 {
    metal2vulkan_runtime::host_time()
}

fn scratch_default() -> PathBuf {
    std::env::temp_dir().join(format!("host-ab-{}", std::process::id()))
}

/// The shaders' source, beside this file.
fn source() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/host_ab/host_ab.metal")
}

/// Compile the source with `NONCE` into `scratch`.
fn compile(scratch: &Path, nonce: u64) -> PathBuf {
    let out = scratch.join(format!("host_ab-{nonce}.metallib"));
    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", &format!("-DNONCE={nonce}"), "-o"])
        .arg(&out)
        .arg(source())
        .status()
        .expect("run xcrun metal");
    assert!(status.success(), "xcrun metal failed");
    out
}

/// One side's results: (workload, metric) to (unit, median).
type Results = BTreeMap<(String, String), (String, f64)>;

/// Run this binary as one side and collect its result lines; with `outputs`, the side writes its
/// kernel outputs there.
fn spawn(
    side: &str,
    metallib: &Path,
    (cache, outputs): (&Path, Option<&Path>),
    only: &[&str],
    options: &Options,
) -> Results {
    let mut command = Command::new(std::env::current_exe().expect("this executable"));
    command
        .args(["--side", side, "--only", &only.join(",")])
        .arg("--metallib")
        .arg(metallib)
        .arg("--cache")
        .arg(cache);
    if let Some(outputs) = outputs {
        command.arg("--outputs").arg(outputs);
    }
    let output = command
        .args(&options.passed)
        .stderr(Stdio::inherit())
        .output()
        .expect("run a side");
    assert!(
        output.status.success(),
        "the {side} side failed: {}",
        output.status
    );
    let mut results = Results::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.first() == Some(&"R") && fields.len() >= 5 {
            let median = fields[4].parse().expect("a median");
            results.insert(
                (fields[1].to_owned(), fields[2].to_owned()),
                (fields[3].to_owned(), median),
            );
        }
    }
    results
}

/// Every round's results per side: (workload, metric) to (unit, [native medians], [runtime
/// medians]).
type Table = BTreeMap<(String, String), (String, Vec<f64>, Vec<f64>)>;

/// Each kernel output's 32-bit words, and per round how many differ between the sides.
type Compared = BTreeMap<String, (usize, Vec<usize>)>;

/// Compare the outputs both sides wrote (`<name>.bin` in each directory), then remove them.
fn compare(dirs: &[PathBuf], compared: &mut Compared) {
    let mut names: Vec<_> = std::fs::read_dir(&dirs[0])
        .expect("read the outputs")
        .map(|entry| entry.expect("an output").file_name())
        .collect();
    names.sort();
    for name in names {
        let [a, b] = [0, 1].map(|s| std::fs::read(dirs[s].join(&name)).expect("read an output"));
        assert_eq!(a.len(), b.len(), "{name:?}: the sides' lengths");
        let differ = a.chunks(4).zip(b.chunks(4)).filter(|(x, y)| x != y).count();
        let key = name.to_string_lossy().trim_end_matches(".bin").to_owned();
        let entry = compared.entry(key).or_insert((a.len() / 4, vec![]));
        entry.1.push(differ);
    }
    for dir in dirs {
        let _ = std::fs::remove_dir_all(dir);
    }
}

fn add(table: &mut Table, side: usize, results: Results, prefix: &str) {
    for ((workload, metric), (unit, median)) in results {
        let key = (workload, format!("{prefix}{metric}"));
        let entry = table.entry(key).or_insert_with(|| (unit, vec![], vec![]));
        [&mut entry.1, &mut entry.2][side].push(median);
    }
}

fn parent(options: &Options) {
    let scratch = options.scratch.clone().unwrap_or_else(scratch_default);
    let removes = options.scratch.is_none();
    std::fs::create_dir_all(&scratch).expect("make the scratch directory");
    let sides: &[&str] = if cfg!(target_os = "macos") {
        &["native", "runtime"]
    } else {
        &["runtime"]
    };
    let rest: Vec<&str> = options
        .only
        .iter()
        .map(String::as_str)
        .filter(|w| *w != "pipeline")
        .collect();
    let mut table = Table::new();
    let mut compared = Compared::new();
    // Both sides' kernel outputs, compared each round.
    let outputs: Vec<PathBuf> = if sides.len() == 2 && rest.contains(&"gpu") {
        sides
            .iter()
            .map(|side| scratch.join(format!("outputs-{side}")))
            .collect()
    } else {
        vec![]
    };
    let metallib = match &options.metallib {
        Some(path) => path.clone(),
        None => compile(&scratch, 0),
    };
    // Below 2^24 (host_ab.metal), and new each second.
    let nonce_base = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("a clock after 1970")
        .as_secs()
        % 100_000;
    for round in 0..options.rounds {
        eprintln!("round {} of {}", round + 1, options.rounds);
        if !rest.is_empty() {
            for (s, side) in sides.iter().enumerate() {
                let out = outputs.get(s).map(PathBuf::as_path);
                if let Some(dir) = out {
                    std::fs::create_dir_all(dir).expect("make an outputs directory");
                }
                let cache = scratch.join("cache-main");
                let results = spawn(side, &metallib, (&cache, out), &rest, options);
                add(&mut table, s, results, "");
            }
            if !outputs.is_empty() {
                compare(&outputs, &mut compared);
            }
        }
        if options.only.iter().any(|w| w == "pipeline") {
            for (s, side) in sides.iter().enumerate() {
                let nonce = nonce_base * 100 + (round * 2 + s) as u64 + 1;
                let fresh = compile(&scratch, nonce);
                let cache = scratch.join(format!("cache-{nonce}"));
                let cold = spawn(side, &fresh, (&cache, None), &["pipeline"], options);
                add(&mut table, s, cold, "cold_");
                let warm = spawn(side, &fresh, (&cache, None), &["pipeline"], options);
                add(&mut table, s, warm, "warm_");
                let _ = std::fs::remove_dir_all(&cache);
                let _ = std::fs::remove_file(&fresh);
            }
        }
    }
    print_table(&table, options.rounds);
    print_percall(&table, options.rounds);
    print_compared(&compared);
    if removes {
        let _ = std::fs::remove_dir_all(&scratch);
    }
}

fn median(values: &[f64]) -> Option<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted.get(sorted.len() / 2).copied()
}

fn print_table(table: &Table, rounds: usize) {
    println!(
        "| workload | metric | unit | native | runtime | runtime/native | native rounds | runtime rounds |"
    );
    println!("|---|---|---|---:|---:|---:|---|---|");
    let spread = |values: &[f64]| match (
        values.iter().copied().reduce(f64::min),
        values.iter().copied().reduce(f64::max),
    ) {
        (Some(low), Some(high)) => format!("{low:.3}..{high:.3} ({})", values.len()),
        _ => "-".to_owned(),
    };
    for ((workload, metric), (unit, native, runtime)) in table {
        let (n, r) = (median(native), median(runtime));
        let show = |v: Option<f64>| v.map_or("-".to_owned(), |v| format!("{v:.3}"));
        let ratio = match (n, r) {
            (Some(n), Some(r)) if n > 0.0 => format!("{:.2}x", r / n),
            _ => "-".to_owned(),
        };
        println!(
            "| {workload} | {metric} | {unit} | {} | {} | {ratio} | {} | {} |",
            show(n),
            show(r),
            spread(native),
            spread(runtime)
        );
    }
    println!("\n{rounds} rounds; each cell is the median of the rounds' medians.");
}

/// The per-call table of the percall workload: each call's cost on each side, their ratio, where
/// we are worst, the three attributions, and the runtime's own counts of what each call made
/// (`percall.rs`). Nothing is printed where the workload did not run.
fn print_percall(table: &Table, rounds: usize) {
    let of = |call: &str, what: &str| -> (Option<f64>, Option<f64>) {
        match table.get(&("percall".to_owned(), format!("{call}_{what}"))) {
            Some((_, native, runtime)) => (median(native), median(runtime)),
            None => (None, None),
        }
    };
    let calls: Vec<&str> = table
        .keys()
        .filter(|(workload, _)| workload == "percall")
        .filter_map(|(_, metric)| metric.strip_suffix("_self"))
        .collect();
    if calls.is_empty() {
        return;
    }
    let show = |v: Option<f64>| v.map_or("-".to_owned(), |v| format!("{v:.1}"));
    let ratio = |(n, r): (Option<f64>, Option<f64>)| match (n, r) {
        (Some(n), Some(r)) if n > 0.0 => Some(r / n),
        _ => None,
    };
    println!("\n### The per-call cost of the device API, Metal against the runtime\n");
    println!("Each call timed alone, warm, {rounds} rounds; ns a call.\n");
    println!("| call | Metal | runtime | runtime/Metal | runtime - Metal |");
    println!("|---|---:|---:|---:|---:|");
    // A call of the driver's own, which the runtime's side alone times, has no Metal counterpart
    // and goes in its own table below (`vkfloor.rs`).
    let (floor, metal): (Vec<&&str>, Vec<&&str>) =
        calls.iter().partition(|call| call.starts_with("vk_"));
    for call in &metal {
        let (n, r) = of(call, "self");
        let added = match (n, r) {
            (Some(n), Some(r)) => format!("{:+.1}", r - n),
            _ => "-".to_owned(),
        };
        println!(
            "| {call} | {} | {} | {} | {added} |",
            show(n),
            show(r),
            ratio((n, r)).map_or("-".to_owned(), |v| format!("{v:.2}x"))
        );
    }
    if !floor.is_empty() {
        println!(
            "\nThe driver's own calls under them, on the Vulkan device the runtime holds \
             (MoltenVK here), which Metal has no counterpart for: what a gap above costs \
             outside our code.\n"
        );
        println!("| driver call | cost |");
        println!("|---|---:|");
        for call in &floor {
            println!("| {call} | {} |", show(of(call, "self").1));
        }
    }
    let mut worst: Vec<(&str, f64, f64, f64)> = calls
        .iter()
        .filter_map(|call| {
            let (n, r) = of(call, "self");
            Some((*call, ratio((n, r))?, r? - n?, n?))
        })
        .collect();
    let top = |rows: &[(&str, f64, f64, f64)], what: &str| {
        println!("\n{what}\n");
        println!("| call | Metal ns | runtime ns | runtime/Metal | runtime - Metal |");
        println!("|---|---:|---:|---:|---:|");
        for (call, ratio, added, native) in rows.iter().take(12) {
            println!(
                "| {call} | {native:.1} | {:.1} | {ratio:.2}x | {added:+.1} |",
                native + added
            );
        }
    };
    worst.sort_by(|a, b| b.1.total_cmp(&a.1));
    top(&worst, "Worst by ratio:");
    worst.sort_by(|a, b| b.2.total_cmp(&a.2));
    top(&worst, "Worst by the nanoseconds a call adds:");
    println!("\n### Where each call's cost falls\n");
    println!(
        "`self` is the calling thread's wall time a call, `blocked` how much of it the thread \
         was off a CPU, `others` our own other threads' CPU time while the calls ran, `drain` \
         the calling thread's wall time after them (the commit and wait of the command buffer \
         the calls filled, where the group has one), `drain blocked` how much of that drain the \
         thread was off a CPU, and `drain others` our other threads' CPU time in it. ns a call, \
         and allocations a call on the calling thread.\n"
    );
    println!(
        "| call | Metal self | Metal others | Metal blocked | Metal drain | Metal drain \
         blocked | Metal drain others | Metal allocs | runtime self | runtime others | runtime \
         blocked | runtime drain | runtime drain blocked | runtime drain others | runtime \
         allocs |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for call in &calls {
        let parts = [
            "self",
            "others",
            "blocked",
            "drain",
            "drain_blocked",
            "drain_others",
            "allocs",
        ];
        let cells: Vec<String> = [0, 1]
            .into_iter()
            .flat_map(|side| {
                parts.map(|what| {
                    let both = of(call, what);
                    show([both.0, both.1][side])
                })
            })
            .collect();
        println!("| {call} | {} |", cells.join(" | "));
    }
    let counts: Vec<(&String, &str, Option<f64>)> = table
        .iter()
        .filter(|((workload, _), _)| workload == "percall")
        .filter_map(|((_, metric), (_, _, runtime))| {
            let (call, counted) = metric.split_once("_rt_")?;
            let _ = call;
            Some((metric, counted, median(runtime)))
        })
        .collect();
    if counts.is_empty() {
        return;
    }
    println!("\n### What the runtime's calls made it do, per call\n");
    println!("Counts, which are the same under any load; none natively.\n");
    println!("| call and count | per call |");
    println!("|---|---:|");
    for (metric, counted, value) in counts {
        let call = metric.split_once("_rt_").map_or("", |(call, _)| call);
        println!("| {call} / {counted} | {} |", show(value));
    }
}

fn print_compared(compared: &Compared) {
    if compared.is_empty() {
        return;
    }
    println!("\n| kernel output | 32-bit words | words differing (native vs runtime), per round |");
    println!("|---|---:|---|");
    for (name, (words, differ)) in compared {
        let rounds: Vec<String> = differ.iter().map(usize::to_string).collect();
        println!("| {name} | {words} | {} |", rounds.join(", "));
    }
}
