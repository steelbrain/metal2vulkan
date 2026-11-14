//! A compute encoder's dispatch whose `VkPipeline`s are being made (`pipeline/dispatch_making.rs`,
//! `pending_dispatch.rs`): Metal makes a compute state's pipeline at `newComputePipelineState`, so
//! a dispatch never waits for a compile, its command buffer completes in its queue's order, and a
//! command buffer of another queue that shares nothing with it runs meanwhile. The device's pool's
//! test hold (`pool::held`) keeps every making waiting here, so a making is as slow as the test
//! makes it.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `scale_and_offset` of the
//! owned fixture `validation/fixtures/public/examples/runtime_examples.ll` (`llvm-as`), as the
//! examples run it: `output[i] = input[i] * scale + i`.

use crate::pipeline::pool::held;
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Device,
    DeviceKind, DeviceOptions, Fence, PipelineOptions, ResourceOptions, ShaderCacheOptions, Size,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The longest a command buffer that runs may take.
const WAIT: Duration = Duration::from_secs(60);

/// How long a command buffer that waits for a making is watched not to complete.
const WATCHED: Duration = Duration::from_millis(500);

/// The threads a dispatch runs, and the scale it multiplies by.
const COUNT: u64 = 1000;
const SCALE: f32 = 2.0;

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        let path = std::env::temp_dir().join(format!(
            "m2v-dispatch-pending-{name}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache
/// in `cache`.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// A state of the fixture's `scale_and_offset`.
pub(super) fn state(device: &Device) -> ComputePipelineState {
    let library = super::encoder_holds::library(device, "examples/runtime_examples.ll");
    let function = library.new_function_with_name("scale_and_offset").unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

/// The kernel's buffers: its input `0, 1, 2, ...`, its output, and its scale.
pub(super) struct Buffers {
    input: Buffer,
    output: Buffer,
    scale: Buffer,
}

impl Buffers {
    pub(super) fn new(device: &Device) -> Buffers {
        let output = device
            .new_buffer(COUNT * 4, ResourceOptions::shared())
            .unwrap();
        Buffers::writing(device, output)
    }

    /// The kernel's buffers, its output `output`, a `Shared` buffer of `COUNT * 4` bytes at least
    /// (a no-copy buffer over host pages, for a leg of `hold_sites.rs`).
    pub(super) fn writing(device: &Device, output: Buffer) -> Buffers {
        let input: Vec<u8> = (0..COUNT).flat_map(|i| (i as f32).to_le_bytes()).collect();
        Buffers {
            input: device
                .new_buffer_with_bytes(&input, ResourceOptions::shared())
                .unwrap(),
            output,
            scale: device
                .new_buffer_with_bytes(&SCALE.to_le_bytes(), ResourceOptions::shared())
                .unwrap(),
        }
    }

    /// The buffer the dispatch writes, for a leg that commits a command buffer sharing it
    /// (`hold_sites.rs`).
    pub(super) fn output(&self) -> &Buffer {
        &self.output
    }

    /// Whether the output's first `n` words are `input[i] * SCALE + i`.
    pub(super) fn written(&self, n: u64) -> bool {
        let contents = self.output.contents().unwrap();
        // SAFETY: the buffer's `COUNT * 4` bytes, which no command buffer writes now.
        let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), (COUNT * 4) as usize) };
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .take(n as usize)
            .enumerate()
            .all(|(i, word)| f32::from_le_bytes(*word) == i as f32 * SCALE + i as f32)
    }
}

/// How a command buffer dispatches.
pub(super) enum Grid {
    /// `dispatchThreads`: this many threads, in threadgroups of this many.
    Threads(Size, Size),
    /// `dispatchThreadgroups`: one threadgroup of this many threads.
    One(Size),
}

/// A command buffer of `queue` that dispatches `state` over `buffers` as `grid` says.
pub(super) fn dispatch(
    queue: &CommandQueue,
    state: &ComputePipelineState,
    buffers: &Buffers,
    grid: Grid,
) -> CommandBuffer {
    encoded(queue, state, buffers, grid, None)
}

/// [`dispatch`], its encoder updating `fence` after the dispatch (`hold_sites.rs`).
pub(super) fn dispatch_updating(
    queue: &CommandQueue,
    state: &ComputePipelineState,
    buffers: &Buffers,
    grid: Grid,
    fence: &Fence,
) -> CommandBuffer {
    encoded(queue, state, buffers, grid, Some(fence))
}

/// [`dispatch`], its encoder updating `fence` after the dispatch where one is given.
fn encoded(
    queue: &CommandQueue,
    state: &ComputePipelineState,
    buffers: &Buffers,
    grid: Grid,
    fence: Option<&Fence>,
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&buffers.input), 0, 0).unwrap();
    encoder.set_buffer(Some(&buffers.output), 0, 1).unwrap();
    encoder.set_buffer(Some(&buffers.scale), 0, 2).unwrap();
    match grid {
        Grid::Threads(threads, per_group) => encoder.dispatch_threads(threads, per_group),
        Grid::One(threads) => encoder.dispatch_threadgroups(Size::new(1, 1, 1), threads),
    }
    .unwrap();
    if let Some(fence) = fence {
        encoder.update_fence(fence).unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer
}

/// A command buffer of `queue` that fills a scratch buffer of its own: it needs no pipeline.
fn fill(device: &Device, queue: &CommandQueue) -> CommandBuffer {
    let scratch = device.new_buffer(256, ResourceOptions::shared()).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(&scratch, 0..256, 7).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

/// Wait until `done` holds, at most [`WAIT`]: whether it did.
fn eventually(done: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + WAIT;
    while !done() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    done()
}

/// A dispatch whose two regions' pipelines (the interior of 64 threads and the edge of 40) the
/// pool is held from making returns at once, both makings handed to the pool; its command buffer
/// waits for them. A command buffer committed after it on its queue completes after it, its
/// completed handler after its, as Metal's do; one of another queue runs meanwhile. Released, the
/// dispatch writes what Metal's writes.
#[test]
fn a_dispatch_waits_for_no_making_and_its_queue_completes_in_order() {
    let scratch = Scratch::new("order");
    let device = device(&scratch.0);
    let state = state(&device);
    let buffers = Buffers::new(&device);
    let (a, b) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let hold = held::hold();

    let pooled = device.work_counts().dispatch_pipelines_pooled;
    let pending = dispatch(
        &a,
        &state,
        &buffers,
        Grid::Threads(Size::new(COUNT, 1, 1), Size::new(64, 1, 1)),
    );
    assert_eq!(
        device.work_counts().dispatch_pipelines_pooled,
        pooled + 1,
        "the flat plan's one making waits in the pool"
    );
    let after = fill(&device, &a);
    let order = Arc::new(Mutex::new(Vec::new()));
    for (name, command_buffer) in [("pending", &pending), ("after", &after)] {
        let order = Arc::clone(&order);
        command_buffer
            .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                order.lock().unwrap().push(name);
            }))
            .unwrap();
    }
    pending.commit().unwrap();
    after.commit().unwrap();

    let other = fill(&device, &b);
    other.commit().unwrap();
    other
        .wait_until_completed_for(WAIT)
        .expect("a command buffer of another queue that shares nothing runs meanwhile");
    assert_eq!(other.status(), CommandBufferStatus::Completed);
    assert!(
        after.wait_until_completed_for(WATCHED).is_err(),
        "a command buffer after the pending dispatch's on its queue waits for it"
    );
    assert_ne!(pending.status(), CommandBufferStatus::Completed);

    drop(hold);
    for command_buffer in [&pending, &after] {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(
            command_buffer.status(),
            CommandBufferStatus::Completed,
            "{:?} {:?}",
            command_buffer.error(),
            command_buffer.failed_pipeline()
        );
    }
    assert!(eventually(|| order.lock().unwrap().len() == 2));
    assert_eq!(*order.lock().unwrap(), ["pending", "after"]);
    assert!(
        buffers.written(COUNT),
        "the dispatch wrote what Metal's writes"
    );
}

/// A state released while its dispatch's making waits in the pool stops it, returning at once:
/// its dispatch's command buffer was dropped uncommitted, so nothing else holds the state. A
/// second state of the same kernel shares its variant, and its dispatch of the same threadgroup
/// took that making as it stood, handing nothing to the pool; once the making failed, its
/// command buffer makes the pipeline itself, on the scheduler's thread while the pool is still
/// held, and completes, writing what Metal's writes.
#[test]
fn a_released_state_stops_its_dispatchs_making_and_a_dispatch_that_took_it_makes_its_own() {
    let scratch = Scratch::new("release");
    let device = device(&scratch.0);
    let hold = held::hold();
    let first = state(&device);
    let second = state(&device);
    let queue = device.new_command_queue().unwrap();
    let buffers = Buffers::new(&device);
    let threads = Size::new(64, 1, 1);
    let before = device.work_counts();
    let dropped = dispatch(&queue, &first, &buffers, Grid::One(threads));
    assert_eq!(
        device.work_counts().dispatch_pipelines_pooled,
        before.dispatch_pipelines_pooled + 1,
        "the first state's dispatch handed its making to the pool"
    );
    let pending = dispatch(&queue, &second, &buffers, Grid::One(threads));
    assert_eq!(
        device.work_counts().dispatch_pipelines_pooled,
        before.dispatch_pipelines_pooled + 1,
        "the second state's dispatch took the first's making in flight"
    );
    pending.commit().unwrap();
    let released = Instant::now();
    drop(dropped);
    drop(first);
    assert!(
        released.elapsed() < WATCHED,
        "the release waited for its making"
    );
    pending.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        pending.status(),
        CommandBufferStatus::Completed,
        "{:?} {:?}",
        pending.error(),
        pending.failed_pipeline()
    );
    assert!(
        buffers.written(64),
        "the dispatch wrote what Metal's writes"
    );
    drop(hold);
}
