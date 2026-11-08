//! Not a consumer's example: command queues sharing `VkQueue`s (`QueueSharingPolicy`), each
//! workload run under the policy the arguments name, with the device's sharing counts after it.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example queue_share -- <mode> \
//!     [--policy adaptive|per-queue] [--seconds S] [--dispatches N] [--unshared] \
//!     [--threads T --in-flight F [--one-vk-queue]]
//! ```
//!
//! Every command buffer is one of two kinds. A *draw*: a render pass over a 512x512 `RGBA8Unorm`
//! texture, cleared and one triangle drawn (`colored_vertex` and `colored_fragment` of the
//! repository's owned fixture `runtime_examples.metal`, assembled by `llvm-as` on `PATH`). A *long*
//! one: `--dispatches` dispatches of `scale_and_offset` over one buffer, sized as the run starts
//! so the command buffer's GPU time is 30 ms. Latencies are a command buffer's `commit` to the
//! start of its completed handler, on the host's clock.
//!
//! - **check**: the device's check of its GPU's idling between `VkQueue`s: its verdict and
//!   switch cost (run where no record of it is kept, read back where one is).
//! - **isolation**: a compositor thread commits draws at 4,800 a second and never waits; an
//!   application thread commits long command buffers back to back, each waited for. The
//!   compositor's latencies, and how many of them are stalls (over 10 ms). `--unshared` makes the
//!   compositor's queue with `new_command_queue_unshared`.
//! - **pipelined**: as isolation, but the application keeps two long command buffers in flight,
//!   so its queue holds no idle point.
//! - **trap**: as isolation, but the application alternates 500 ms of draws committed at 10,000 a
//!   second with 10 long command buffers.
//! - **alternating**: as trap, with 2 ms of draws between each long command buffer; and
//!   **slow-alternating**, 1,500 ms.
//! - **interleave**: two compositor threads, each its own queue, at 4,800 draws a second each; the
//!   first's latencies, and the GPU's idle time between command buffers of different queues as a
//!   share of the run (from their GPU start and end times, in commit order on the GPU).
//! - **rates**: 2 and 8 threads, each its own queue, committing draws as fast as they can with 3,
//!   then 1, in flight a thread: command buffers a second, and the GPU's idle share at switches.
//!   With `--threads` and `--in-flight`, that one shape, with each command buffer's GPU end and
//!   commit to its completed handler, the device's submission, scheduler and wake counts per
//!   command buffer, and the busiest threads' CPU (Linux); `--one-vk-queue` puts every thread's
//!   queue on one `VkQueue`, each lent once (`with_vk_queue`) so it never moves.
//! - **idle-queues**: one thread committing draws in turn over 16, then 2,048 queues, 3 in flight:
//!   command buffers a second, and the thread's time in `command_buffer` per command buffer.

use metal2vulkan::device::{DevicePolicy, QueueSharingPolicy};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandQueue, ComputePipelineState, Device, DeviceOptions,
    LoadAction, PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size,
    StoreAction, Texture, TextureDescriptor,
};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A command buffer's GPU time a long one is sized for.
const LONG: Duration = Duration::from_millis(30);

/// A compositor's latency past which it counts as a stall.
const STALL_MS: f64 = 10.0;

/// The draws a second of a compositor thread.
const COMPOSITOR_RATE: f64 = 4800.0;

/// The render target's side, in texels.
const SIDE: u64 = 512;

/// The floats the long command buffers' buffers hold at most.
const MOST_FLOATS: u64 = 96 << 20;

struct Options {
    mode: String,
    policy: QueueSharingPolicy,
    seconds: f64,
    dispatches: u64,
    unshared: bool,
    /// `rates` of this many threads alone (0: the default set).
    threads: usize,
    in_flight: usize,
    /// `rates` with every thread's queue on one `VkQueue`, each lent once so it never moves.
    one_vk_queue: bool,
}

fn options() -> Options {
    let mut args = std::env::args().skip(1);
    let mut options = Options {
        mode: args.next().unwrap_or_else(|| "isolation".to_owned()),
        policy: QueueSharingPolicy::Adaptive,
        seconds: 2.0,
        dispatches: 300,
        unshared: false,
        threads: 0,
        in_flight: 3,
        one_vk_queue: false,
    };
    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--policy" => {
                options.policy = match value().as_str() {
                    "adaptive" => QueueSharingPolicy::Adaptive,
                    "per-queue" => QueueSharingPolicy::PerQueue,
                    other => panic!("no policy {other}"),
                }
            }
            "--seconds" => options.seconds = value().parse().unwrap(),
            "--dispatches" => options.dispatches = value().parse().unwrap(),
            "--unshared" => options.unshared = true,
            "--threads" => options.threads = value().parse().unwrap(),
            "--in-flight" => options.in_flight = value().parse().unwrap(),
            "--one-vk-queue" => options.one_vk_queue = true,
            other => panic!("unknown argument {other}"),
        }
    }
    options
}

/// What every workload draws and dispatches with.
struct Kit {
    device: Device,
    draw_state: RenderPipelineState,
    positions: Buffer,
    colors: Buffer,
    compute: ComputePipelineState,
    input: Buffer,
    output: Buffer,
    scale: Buffer,
    /// The grid of one long command buffer's dispatches.
    grid: u64,
    dispatches: u64,
}

impl Kit {
    fn new(options: &Options) -> Kit {
        let device = device(options.policy);
        let settled = device.settle_queue_switch_check(Duration::from_secs(10));
        println!(
            "device: {}; policy {:?}; check settled {settled}: {}",
            device.name(),
            options.policy,
            device.queue_switch_check_report()
        );
        let library = metal2vulkan::library::Library::from_air(&fixture_bitcode()).unwrap();
        let library = device.new_library_with_compile_library(library);
        let vertex = library.new_function_with_name("colored_vertex").unwrap();
        let fragment = library.new_function_with_name("colored_fragment").unwrap();
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
        descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let draw_state = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let positions: Vec<u8> = [[-1.0f32, -1.0], [1.0, -1.0], [0.0, 1.0]]
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let colors: Vec<u8> = (0..3)
            .flat_map(|_| [1.0f32, 0.5, 0.0, 1.0])
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let positions = device
            .new_buffer_with_bytes(&positions, ResourceOptions::shared())
            .unwrap();
        let colors = device
            .new_buffer_with_bytes(&colors, ResourceOptions::shared())
            .unwrap();
        let function = library.new_function_with_name("scale_and_offset").unwrap();
        let compute = device
            .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
            .unwrap();
        let input = device
            .new_buffer(MOST_FLOATS * 4, ResourceOptions::private())
            .unwrap();
        let output = device
            .new_buffer(MOST_FLOATS * 4, ResourceOptions::private())
            .unwrap();
        let scale = device
            .new_buffer_with_bytes(&1.0f32.to_le_bytes(), ResourceOptions::shared())
            .unwrap();
        let mut kit = Kit {
            device,
            draw_state,
            positions,
            colors,
            compute,
            input,
            output,
            scale,
            grid: 1 << 20,
            dispatches: options.dispatches,
        };
        kit.size_long();
        kit
    }

    /// Size the long command buffers' grid so one's GPU time is [`LONG`].
    fn size_long(&mut self) {
        let queue = self.device.new_command_queue().unwrap();
        for _ in 0..3 {
            let command_buffer = self.long(&queue);
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed().unwrap();
            let gpu =
                command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
            let wanted = self.grid as f64 * LONG.as_secs_f64() / gpu.max(1e-6);
            self.grid = (wanted as u64).clamp(1 << 16, MOST_FLOATS) & !63;
        }
        let command_buffer = self.long(&queue);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed().unwrap();
        let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
        println!(
            "long command buffer: {} dispatches of {} threads, {:.1} ms of GPU time",
            self.dispatches,
            self.grid,
            gpu * 1e3
        );
    }

    /// A long command buffer of `queue`'s, not committed.
    fn long(&self, queue: &CommandQueue) -> CommandBuffer {
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&self.compute).unwrap();
        encoder.set_buffer(Some(&self.scale), 0, 2).unwrap();
        for dispatch in 0..self.dispatches {
            let (from, to) = match dispatch % 2 {
                0 => (&self.input, &self.output),
                _ => (&self.output, &self.input),
            };
            encoder.set_buffer(Some(from), 0, 0).unwrap();
            encoder.set_buffer(Some(to), 0, 1).unwrap();
            encoder
                .dispatch_threads(Size::new(self.grid, 1, 1), Size::new(64, 1, 1))
                .unwrap();
        }
        encoder.end_encoding().unwrap();
        command_buffer
    }

    /// A draw command buffer of `queue`'s into `target`, not committed.
    fn draw(&self, queue: &CommandQueue, target: &Texture) -> CommandBuffer {
        self.draw_into(queue.command_buffer().unwrap(), target)
    }

    /// `command_buffer` with a draw into `target` encoded, not committed.
    fn draw_into(&self, command_buffer: CommandBuffer, target: &Texture) -> CommandBuffer {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 1.0);
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.draw_state).unwrap();
        encoder
            .set_vertex_buffer(Some(&self.positions), 0, 0)
            .unwrap();
        encoder.set_vertex_buffer(Some(&self.colors), 0, 1).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer
    }

    fn target(&self) -> Texture {
        let mut texture = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        texture.usage = TextureUsage::RENDER_TARGET;
        texture.resource_options = ResourceOptions::private();
        self.device.new_texture(&texture).unwrap()
    }

    fn queue(&self, unshared: bool) -> CommandQueue {
        match unshared {
            true => self.device.new_command_queue_unshared(64).unwrap(),
            false => self.device.new_command_queue().unwrap(),
        }
    }

    fn counts(&self) -> String {
        let c = self.device.work_counts();
        format!(
            "moves {} joins {} gone long {} settles {} off unshared {} withheld {}",
            c.queue_moves,
            c.queue_joins,
            c.queues_gone_long,
            c.long_queue_settles,
            c.queue_moves_off_unshared,
            c.joins_withheld_from_unshared
        )
    }
}

/// The system default device, of the default policy but for `queue_sharing`.
fn device(queue_sharing: QueueSharingPolicy) -> Device {
    let mut options = DeviceOptions::default();
    options.policy = DevicePolicy::UNTRUSTED;
    options.policy.queue_sharing = queue_sharing;
    Device::system_default_with(&options).unwrap()
}

/// The owned fixture's LLVM text, assembled to AIR bitcode by `llvm-as`.
fn fixture_bitcode() -> Vec<u8> {
    let ll = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/examples/runtime_examples.ll");
    let output = Command::new("llvm-as")
        .arg("-o")
        .arg("-")
        .arg(&ll)
        .output()
        .expect("llvm-as on PATH");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

/// Each draw's queue number and its GPU start and end there, in seconds.
type Times = Arc<Mutex<Vec<(usize, f64, f64)>>>;

/// A queue's changes of `VkQueue` or long mark: its command buffers made by then, the
/// microseconds into the run, and its `VkQueue` and mark.
type Seen = Vec<(u64, u128, (usize, bool))>;

/// The p50, p99 and largest of `samples` (milliseconds), and how many are stalls.
fn summary(samples: &mut [f64]) -> String {
    if samples.is_empty() {
        return "no samples".to_owned();
    }
    samples.sort_by(f64::total_cmp);
    let at = |p: f64| samples[((samples.len() as f64 * p) as usize).min(samples.len() - 1)];
    let stalls = samples.iter().filter(|&&s| s > STALL_MS).count();
    format!(
        "p50 {:.3} p90 {:.3} p99 {:.3} max {:.1} ms, {} stalls over {STALL_MS} ms, {} samples",
        at(0.5),
        at(0.9),
        at(0.99),
        samples[samples.len() - 1],
        stalls,
        samples.len()
    )
}

/// A thread committing draws on `queue` at `rate` a second until `stop`, never waiting, each
/// draw's latency in `latencies` and, where `times` is given, its queue's number and its GPU start
/// and end there.
fn compositor(
    kit: &Arc<Kit>,
    queue: CommandQueue,
    rate: f64,
    stop: &Arc<AtomicBool>,
    latencies: &Arc<Mutex<Vec<f64>>>,
    times: Option<(usize, Times)>,
) -> std::thread::JoinHandle<()> {
    let kit = Arc::clone(kit);
    let stop = Arc::clone(stop);
    let latencies = Arc::clone(latencies);
    std::thread::spawn(move || {
        let target = kit.target();
        let period = Duration::from_secs_f64(1.0 / rate);
        let mut next = Instant::now();
        while !stop.load(Ordering::Relaxed) {
            let command_buffer = kit.draw(&queue, &target);
            let latencies = Arc::clone(&latencies);
            let times = times.clone();
            let committed = Arc::new(Mutex::new(Instant::now()));
            let at = Arc::clone(&committed);
            command_buffer
                .add_completed_handler(Box::new(move |done: &CommandBuffer| {
                    let latency = at.lock().unwrap().elapsed().as_secs_f64() * 1e3;
                    latencies.lock().unwrap().push(latency);
                    if let Some((queue, times)) = &times {
                        if let (Ok(start), Ok(end)) = (done.gpu_start_time(), done.gpu_end_time()) {
                            times.lock().unwrap().push((*queue, start, end));
                        }
                    }
                }))
                .unwrap();
            *committed.lock().unwrap() = Instant::now();
            command_buffer.commit().unwrap();
            drop(command_buffer);
            next += period;
            let now = Instant::now();
            match next > now {
                true => std::thread::sleep(next - now),
                false => next = now,
            }
        }
    })
}

/// The application's loop until `stop`, on its own thread: `long_every` long command buffers after
/// each `short` of draws at 10,000 a second (none where `short` is zero), each long one waited for,
/// or with `in_flight` long ones in flight where above 1.
fn application(
    kit: &Arc<Kit>,
    short: Duration,
    long_every: usize,
    in_flight: usize,
    stop: &Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    let kit = Arc::clone(kit);
    let stop = Arc::clone(stop);
    std::thread::spawn(move || {
        let queue = kit.device.new_command_queue().unwrap();
        let target = kit.target();
        let mut flying: std::collections::VecDeque<CommandBuffer> = Default::default();
        while !stop.load(Ordering::Relaxed) {
            if !short.is_zero() {
                let began = Instant::now();
                let period = Duration::from_secs_f64(1.0 / 10_000.0);
                let mut next = began;
                while began.elapsed() < short && !stop.load(Ordering::Relaxed) {
                    let command_buffer = kit.draw(&queue, &target);
                    command_buffer.commit().unwrap();
                    next += period;
                    let now = Instant::now();
                    if next > now {
                        std::thread::sleep(next - now);
                    }
                }
            }
            for _ in 0..long_every {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                let command_buffer = kit.long(&queue);
                command_buffer.commit().unwrap();
                flying.push_back(command_buffer);
                while flying.len() >= in_flight.max(1) {
                    flying.pop_front().unwrap().wait_until_completed().unwrap();
                }
            }
        }
        for command_buffer in flying {
            command_buffer.wait_until_completed().unwrap();
        }
    })
}

/// A compositor beside an application of `short` draws between `long_every` long command buffers,
/// `in_flight` of them in flight: the compositor's latencies.
fn beside(kit: &Arc<Kit>, options: &Options, short: Duration, long_every: usize, in_flight: usize) {
    let stop = Arc::new(AtomicBool::new(false));
    let latencies = Arc::new(Mutex::new(Vec::new()));
    let app = application(kit, short, long_every, in_flight, &stop);
    std::thread::sleep(Duration::from_millis(50));
    let queue = kit.queue(options.unshared);
    let comp = compositor(kit, queue, COMPOSITOR_RATE, &stop, &latencies, None);
    std::thread::sleep(Duration::from_secs_f64(options.seconds));
    stop.store(true, Ordering::Relaxed);
    comp.join().unwrap();
    app.join().unwrap();
    std::thread::sleep(Duration::from_millis(100));
    let mut samples = latencies.lock().unwrap().clone();
    println!(
        "{} (policy {:?}, compositor {}, {} dispatches, short {:?} x{} in flight {}): compositor {}; {}",
        options.mode,
        options.policy,
        if options.unshared { "unshared" } else { "shared" },
        kit.dispatches,
        short,
        long_every,
        in_flight,
        summary(&mut samples),
        kit.counts()
    );
}

/// The GPU's idle time between command buffers of different queues, as a share of the span of
/// `times` (queue, GPU start, GPU end), in GPU start order; with the switches counted.
fn switch_idle(times: &mut [(usize, f64, f64)]) -> (f64, usize) {
    if times.len() < 2 {
        return (0.0, 0);
    }
    times.sort_by(|a, b| a.1.total_cmp(&b.1));
    let span = times[times.len() - 1].2 - times[0].1;
    let mut idle = 0.0;
    let mut switches = 0;
    for pair in times.windows(2) {
        if pair[0].0 != pair[1].0 {
            switches += 1;
            idle += (pair[1].1 - pair[0].2).max(0.0);
        }
    }
    (idle / span.max(1e-9), switches)
}

fn interleave(kit: &Arc<Kit>, options: &Options) {
    let stop = Arc::new(AtomicBool::new(false));
    let first = Arc::new(Mutex::new(Vec::new()));
    let second = Arc::new(Mutex::new(Vec::new()));
    let times = Arc::new(Mutex::new(Vec::new()));
    let a = compositor(
        kit,
        kit.queue(false),
        COMPOSITOR_RATE,
        &stop,
        &first,
        Some((0, Arc::clone(&times))),
    );
    let b = compositor(
        kit,
        kit.queue(false),
        COMPOSITOR_RATE,
        &stop,
        &second,
        Some((1, Arc::clone(&times))),
    );
    std::thread::sleep(Duration::from_secs_f64(options.seconds));
    stop.store(true, Ordering::Relaxed);
    a.join().unwrap();
    b.join().unwrap();
    std::thread::sleep(Duration::from_millis(100));
    let mut samples: Vec<f64> = first.lock().unwrap().iter().map(|ms| ms * 1e3).collect();
    samples.sort_by(f64::total_cmp);
    let at = |p: f64| samples[((samples.len() as f64 * p) as usize).min(samples.len() - 1)];
    let (idle, switches) = switch_idle(&mut times.lock().unwrap());
    println!(
        "interleave (policy {:?}): first stream p50 {:.1} p99 {:.1} us; GPU idle at {switches} switches {:.1}% of the span; {}",
        options.policy,
        at(0.5),
        at(0.99),
        idle * 100.0,
        kit.counts()
    );
}

/// `threads` threads, each its own queue, committing draws with `in_flight` in flight: command
/// buffers a second over the run, and the GPU's idle share at switches.
/// The queues of `threads` threads: each its own where `one` is false; else all on the first's
/// `VkQueue`, made until the round robin comes back to it, each lent once so it never moves.
fn rate_queues(kit: &Kit, threads: usize, one: bool) -> Vec<CommandQueue> {
    if !one {
        return (0..threads).map(|_| kit.queue(false)).collect();
    }
    // SAFETY: the closures keep the handle only to compare it; nothing is submitted.
    let raw = |queue: &CommandQueue| unsafe { queue.with_vk_queue(|_, raw, _| raw) };
    let first = kit.queue(false);
    let home = raw(&first);
    let mut queues = vec![first];
    let mut others = Vec::new();
    while queues.len() < threads {
        let queue = kit.queue(false);
        match raw(&queue) == home {
            true => queues.push(queue),
            false => others.push(queue),
        }
    }
    queues
}

/// Per-thread CPU time of this process by thread name, in clock ticks (Linux; empty elsewhere).
fn thread_ticks() -> Vec<(u64, String, u64)> {
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return Vec::new();
    };
    let mut ticks = Vec::new();
    for task in tasks.flatten() {
        let Ok(stat) = std::fs::read_to_string(task.path().join("stat")) else {
            continue;
        };
        let (Some(open), Some(close)) = (stat.find('('), stat.rfind(')')) else {
            continue;
        };
        let name = stat[open + 1..close].to_owned();
        let fields: Vec<&str> = stat[close + 2..].split(' ').collect();
        let (Some(user), Some(system)) = (fields.get(11), fields.get(12)) else {
            continue;
        };
        let tid = task.file_name().to_string_lossy().parse().unwrap_or(0);
        ticks.push((
            tid,
            name,
            user.parse::<u64>().unwrap_or(0) + system.parse::<u64>().unwrap_or(0),
        ));
    }
    ticks
}

/// The busiest threads between `before` and `after`, as a share of `seconds` each.
fn busiest(before: &[(u64, String, u64)], after: &[(u64, String, u64)], seconds: f64) -> String {
    let mut used: Vec<(f64, String)> = after
        .iter()
        .map(|(tid, name, ticks)| {
            let was = before
                .iter()
                .find(|(t, _, _)| t == tid)
                .map_or(0, |(_, _, t)| *t);
            ((ticks - was) as f64 / 100.0 / seconds, name.clone())
        })
        .filter(|(share, _)| *share > 0.02)
        .collect();
    used.sort_by(|a, b| b.0.total_cmp(&a.0));
    used.iter()
        .take(8)
        .map(|(share, name)| format!("{name} {:.0}%", share * 100.0))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The p50 and p99 of `samples`, in microseconds.
fn quantiles(samples: &mut [f64]) -> String {
    if samples.is_empty() {
        return "none".to_owned();
    }
    samples.sort_by(f64::total_cmp);
    let at = |p: f64| samples[((samples.len() as f64 * p) as usize).min(samples.len() - 1)] * 1e6;
    format!("p50 {:.1} p99 {:.1} us", at(0.5), at(0.99))
}

fn rate(kit: &Arc<Kit>, options: &Options, threads: usize, in_flight: usize) {
    let stop = Arc::new(AtomicBool::new(false));
    let times = Arc::new(Mutex::new(Vec::new()));
    // Each command buffer's GPU end to its completed handler, and its commit to that handler.
    let latencies = Arc::new(Mutex::new((Vec::new(), Vec::new())));
    let queues = rate_queues(kit, threads, options.one_vk_queue);
    let counts = kit.device.work_counts();
    let cpu = thread_ticks();
    let began = Instant::now();
    let handles: Vec<_> = (queues.into_iter().enumerate())
        .map(|(thread, queue)| {
            let kit = Arc::clone(kit);
            let stop = Arc::clone(&stop);
            let times = Arc::clone(&times);
            let latencies = Arc::clone(&latencies);
            std::thread::spawn(move || {
                let target = kit.target();
                let mut flying = std::collections::VecDeque::new();
                let mut made = 0u64;
                // Each change of the queue's `VkQueue` or long mark: at which command buffer,
                // how far into the run.
                let mut seen = Vec::new();
                let started = Instant::now();
                while !stop.load(Ordering::Relaxed) {
                    let command_buffer = kit.draw(&queue, &target);
                    let now = queue.vk_queue_sharing();
                    if seen.last().is_none_or(|&(_, _, last)| last != now) {
                        seen.push((made, started.elapsed().as_micros(), now));
                    }
                    let times = Arc::clone(&times);
                    let latencies = Arc::clone(&latencies);
                    let committed = Arc::new(Mutex::new(0.0));
                    let at = Arc::clone(&committed);
                    command_buffer
                        .add_completed_handler(Box::new(move |done: &CommandBuffer| {
                            let now = metal2vulkan_runtime::host_time();
                            if let (Ok(start), Ok(end)) =
                                (done.gpu_start_time(), done.gpu_end_time())
                            {
                                times.lock().unwrap().push((thread, start, end));
                                let mut latencies = latencies.lock().unwrap();
                                latencies.0.push(now - end);
                                latencies.1.push(now - *at.lock().unwrap());
                            }
                        }))
                        .unwrap();
                    *committed.lock().unwrap() = metal2vulkan_runtime::host_time();
                    command_buffer.commit().unwrap();
                    flying.push_back(command_buffer);
                    made += 1;
                    while flying.len() >= in_flight {
                        let done: CommandBuffer = flying.pop_front().unwrap();
                        done.wait_until_completed().unwrap();
                    }
                }
                for done in flying {
                    done.wait_until_completed().unwrap();
                }
                (made, seen)
            })
        })
        .collect();
    std::thread::sleep(Duration::from_secs_f64(options.seconds));
    stop.store(true, Ordering::Relaxed);
    let each: Vec<(u64, Seen)> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let made: u64 = each.iter().map(|(made, _)| made).sum();
    let elapsed = began.elapsed().as_secs_f64();
    let cpu_after = thread_ticks();
    std::thread::sleep(Duration::from_millis(100));
    let after = kit.device.work_counts();
    let (idle, switches) = switch_idle(&mut times.lock().unwrap());
    let mut latencies = latencies.lock().unwrap();
    let per = |a: u64, b: u64| (a - b) as f64 / made.max(1) as f64;
    println!(
        "rates (policy {:?}{}): {threads} threads, {in_flight} in flight: {:.1}k command buffers/s; GPU idle at {switches} switches {:.1}%; {}",
        options.policy,
        if options.one_vk_queue { ", one VkQueue" } else { "" },
        made as f64 / elapsed / 1e3,
        idle * 100.0,
        kit.counts()
    );
    println!(
        "  GPU end to handler {}; commit to handler {}",
        quantiles(&mut latencies.0),
        quantiles(&mut latencies.1)
    );
    println!(
        "  per command buffer: submissions {:.2}, submit infos {:.2}, VkCommandBuffers {:.2}, scheduler passes {:.2} (by waiters {:.2}, their batches {:.2}), scheduler sleeps {:.2} (woken by commits {:.2}, completions {:.2}), timeline waits {:.2}, fence waits {:.2}, waiter parks {:.2} (timed out {:.2}), waiter wakes {:.2}, handler-thread wakes {:.2}, command-buffer wakes {:.2}, completed where seen {:.2}, turns taken {:.2}, submit {:.1} us",
        per(after.queue_submissions, counts.queue_submissions),
        per(after.submit_infos, counts.submit_infos),
        per(after.command_buffers_submitted, counts.command_buffers_submitted),
        per(after.scheduler_passes, counts.scheduler_passes),
        per(after.scheduler_passes_by_waiters, counts.scheduler_passes_by_waiters),
        per(after.batches_submitted_by_waiters, counts.batches_submitted_by_waiters),
        per(after.scheduler_sleeps, counts.scheduler_sleeps),
        per(after.scheduler_wakes_commit, counts.scheduler_wakes_commit),
        per(after.scheduler_wakes_completion, counts.scheduler_wakes_completion),
        per(after.timeline_waits, counts.timeline_waits),
        per(after.fence_waits, counts.fence_waits),
        per(after.completion_waiter_parks, counts.completion_waiter_parks),
        per(after.completion_waiter_parks_timed_out, counts.completion_waiter_parks_timed_out),
        per(after.wakes_completion_waiters, counts.wakes_completion_waiters),
        per(after.wakes_handler_threads, counts.wakes_handler_threads),
        per(after.wakes_command_buffer_waiters, counts.wakes_command_buffer_waiters),
        per(after.completed_where_seen, counts.completed_where_seen),
        per(after.completion_turns_where_taken, counts.completion_turns_where_taken),
        per(after.batch_submit_nanos, counts.batch_submit_nanos) / 1e3,
    );
    println!("  threads' CPU: {}", busiest(&cpu, &cpu_after, elapsed));
    for (thread, (made, seen)) in each.iter().enumerate() {
        let changes: Vec<String> = seen
            .iter()
            .map(|(at, us, (vk_queue, long))| {
                format!(
                    "#{at} {us}us VkQueue {vk_queue}{}",
                    if *long { " long" } else { "" }
                )
            })
            .collect();
        println!(
            "  thread {thread}: {made} command buffers; {}",
            changes.join(", ")
        );
    }
}

fn idle_queues(kit: &Arc<Kit>, options: &Options, count: usize) {
    let queues: Vec<CommandQueue> = (0..count).map(|_| kit.queue(false)).collect();
    let target = kit.target();
    let mut flying = std::collections::VecDeque::new();
    let mut made = 0u64;
    let mut making = Duration::ZERO;
    let began = Instant::now();
    while began.elapsed().as_secs_f64() < options.seconds {
        let queue = &queues[made as usize % count];
        let start = Instant::now();
        let command_buffer = queue.command_buffer().unwrap();
        making += start.elapsed();
        let command_buffer = kit.draw_into(command_buffer, &target);
        command_buffer.commit().unwrap();
        flying.push_back(command_buffer);
        made += 1;
        while flying.len() >= 3 {
            let done: CommandBuffer = flying.pop_front().unwrap();
            done.wait_until_completed().unwrap();
        }
    }
    for done in flying {
        done.wait_until_completed().unwrap();
    }
    let elapsed = began.elapsed().as_secs_f64();
    println!(
        "idle-queues (policy {:?}): {count} queues in turn: {:.1}k command buffers/s, command_buffer {:.0} ns each; {}",
        options.policy,
        made as f64 / elapsed / 1e3,
        making.as_nanos() as f64 / made as f64,
        kit.counts()
    );
}

fn main() {
    let options = options();
    if options.mode == "check" {
        let device = device(options.policy);
        let settled = device.settle_queue_switch_check(Duration::from_secs(10));
        println!(
            "device: {}; settled {settled}: {}",
            device.name(),
            device.queue_switch_check_report()
        );
        return;
    }
    let kit = Arc::new(Kit::new(&options));
    match options.mode.as_str() {
        "isolation" => beside(&kit, &options, Duration::ZERO, 1, 1),
        "pipelined" => beside(&kit, &options, Duration::ZERO, 1, 2),
        "trap" => beside(&kit, &options, Duration::from_millis(500), 10, 1),
        "alternating" => beside(&kit, &options, Duration::from_millis(2), 1, 1),
        "slow-alternating" => beside(&kit, &options, Duration::from_millis(1500), 1, 1),
        "interleave" => interleave(&kit, &options),
        "rates" if options.threads > 0 => {
            rate(&kit, &options, options.threads, options.in_flight);
        }
        "rates" => {
            for in_flight in [3, 1] {
                for threads in [2, 8] {
                    rate(&kit, &options, threads, in_flight);
                }
            }
        }
        "idle-queues" => {
            idle_queues(&kit, &options, 16);
            idle_queues(&kit, &options, 2048);
        }
        other => panic!("no mode {other}"),
    }
}
