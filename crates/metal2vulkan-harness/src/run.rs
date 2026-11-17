//! A plan on a device: its objects made, its first run checked, the runs after it timed.

use crate::plan::{Binding, BufferPlan, Constant, Grid, Plan, Source};
use crate::{
    clocks, Buffer, Code, CommandBuffer, CommandQueue, ComputeCommandEncoder, ComputeDevice,
    Device, Library, Result, Storage,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// The kernel of the authored library that moves bytes for the harness, 16 a thread.
const COPY_FUNCTION: &str = "copy_words";

/// The most threadgroups a dispatch of the copy kernel takes in x: the least
/// `maxComputeWorkGroupCount[0]` Vulkan allows.
const COPY_GROUPS: u64 = 65535;

/// How many runs follow the checked first one: untimed ones, then `timed`.
#[derive(Clone, Copy, Debug)]
pub struct Runs {
    /// The fewest untimed runs.
    pub warmup: u32,
    /// Whether the untimed runs go on until the time settles ([`warm_up`]).
    pub settle: bool,
    /// Timed runs.
    pub timed: u32,
}

/// The runs of a settling window.
const WINDOW: usize = 10;

/// The largest difference between the medians of the last two windows, as a share of the larger,
/// at which the time has settled.
const SETTLED: f64 = 0.03;

/// The shortest settling warm-up: an integrated GPU raises its clock over hundreds of
/// milliseconds of load.
const SETTLE_LEAST: Duration = Duration::from_millis(500);

/// The longest settling warm-up; the timed runs start then, settled or not.
const SETTLE_MOST: Duration = Duration::from_secs(5);

/// The fewest timed runs whose median is a workload's time.
pub const LEAST_TIMED: usize = 5;

/// A workload's time from its timed runs' `times`: the lowest median of their consecutive
/// windows of 10 runs, so that a clock that drops partway through the runs does not move it; with
/// fewer than 10 runs, their median; `None` with fewer than [`LEAST_TIMED`].
pub fn steady(times: &[f64]) -> Option<f64> {
    if times.len() < LEAST_TIMED {
        return None;
    }
    if times.len() < WINDOW {
        return percentile(times, 50);
    }
    (times.as_chunks::<WINDOW>().0.iter())
        .filter_map(|window| percentile(window, 50))
        .min_by(f64::total_cmp)
}

/// The untimed runs before a workload's timed ones.
#[derive(Clone, Copy, Debug, Default)]
pub struct Warmup {
    /// How many ran.
    pub runs: u32,
    /// Whether the time settled: the medians of the last two windows of 10 runs within
    /// 3% of each other.
    pub settled: bool,
}

/// The untimed runs of `run`, which returns a run's time: `runs.warmup` of them, and where
/// `runs.settle`, as many more as the time takes to settle, for at least half a second, and at
/// most five seconds.
pub fn warm_up(runs: Runs, mut run: impl FnMut() -> Result<f64>) -> Result<Warmup> {
    let started = Instant::now();
    let mut times = Vec::new();
    loop {
        let least = times.len() >= runs.warmup as usize;
        if least && !runs.settle {
            return Ok(Warmup {
                runs: runs.warmup,
                settled: false,
            });
        }
        let elapsed = started.elapsed();
        if least && elapsed >= SETTLE_LEAST && settled(&times) {
            return Ok(Warmup {
                runs: times.len() as u32,
                settled: true,
            });
        }
        if least && elapsed >= SETTLE_MOST {
            return Ok(Warmup {
                runs: times.len() as u32,
                settled: false,
            });
        }
        times.push(run()?);
    }
}

/// Whether the medians of the last two windows of `times` are within [`SETTLED`] of each other.
pub(crate) fn settled(times: &[f64]) -> bool {
    let Some(start) = times.len().checked_sub(2 * WINDOW) else {
        return false;
    };
    let window = |at: usize| percentile(&times[at..at + WINDOW], 50).unwrap_or(0.0);
    let (before, last) = (window(start), window(start + WINDOW));
    (last - before).abs() <= SETTLED * last.max(before)
}

/// What the timed runs of a plan measured.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// The dispatches of a run.
    pub dispatches: usize,
    /// Each run's GPU time, `GPUEndTime - GPUStartTime`, in microseconds.
    pub gpu_us: Vec<f64>,
    /// Each run's encoding thread's wall time from the command buffer's creation through
    /// `commit`'s return, in nanoseconds.
    pub encode_ns: Vec<f64>,
    /// Each run's process CPU time from before its command buffer's creation through the return
    /// of the wait for the run before it, in nanoseconds; empty where the host has no such clock.
    pub cpu_ns: Vec<f64>,
    /// The device's gauges after the 100th and the 200th timed run, where there were that many.
    pub gauges: Vec<(u32, Vec<(&'static str, u64)>)>,
    /// The contents of the plan's outputs after the first run, in its order.
    pub outputs: Vec<Vec<u8>>,
    /// The untimed runs.
    pub warmup: Warmup,
}

/// `Ok` where `other`'s outputs of `plan` match `reference`'s, each by its comparison.
pub fn compare_outputs(plan: &Plan, reference: &Report, other: &Report) -> Result<()> {
    let pairs = reference.outputs.iter().zip(&other.outputs);
    for (output, (want, got)) in plan.outputs.iter().zip(pairs) {
        let buffer = output.buffer;
        (output.comparison.values(want).compare(got))
            .map_err(|why| format!("{}: buffer {buffer}: {why}", plan.name))?;
    }
    Ok(())
}

/// A runtime's device with the harness on it, without its types, so that runtimes of different
/// types run each workload in turn.
pub trait Session {
    /// The device's `name`.
    fn device_name(&self) -> String;

    /// `plan`'s first run, checked against its expectations and its outputs kept, then `runs`.
    fn run(&mut self, plan: &Plan, runs: Runs) -> Result<Report>;

    /// `Shared` buffers of `length` bytes made and released in turn, so each takes the range the
    /// last one left: one checked whole, then `runs`. Each one is checked to start zero (every
    /// 4 KiB), then written all over before its release.
    fn create(&mut self, length: u64, runs: Runs) -> Result<CreateReport>;

    /// Drops the session, its device last; the device's gauges after that, where the runtime's
    /// gauges outlive the device.
    fn finish(self: Box<Self>) -> Option<Vec<(&'static str, u64)>>;

    /// The device's [`Device::pipelines_created`].
    fn pipelines_created(&self) -> u64;
}

/// What the timed creations of buffers measured.
#[derive(Clone, Debug, Default)]
pub struct CreateReport {
    /// Each timed `newBufferWithLength:options:` call's wall time, in microseconds.
    pub create_us: Vec<f64>,
    /// The device's gauges after the 100th and the 200th timed creation, where there were that
    /// many.
    pub gauges: Vec<(u32, Vec<(&'static str, u64)>)>,
    /// The untimed creations.
    pub warmup: Warmup,
}

/// The `p`th percentile of `values` by nearest rank; `None` for no values.
pub fn percentile(values: &[f64], p: u32) -> Option<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (p as usize * sorted.len()).div_ceil(100).max(1);
    sorted.get(rank - 1).copied()
}

/// The harness on a device: its own queue, the authored library and its copy kernel, and the
/// metallibs the plans named so far.
pub struct Harness<D: ComputeDevice> {
    queue: D::CommandQueue,
    copy: D::ComputePipelineState,
    authored: D::Library,
    metallibs: HashMap<PathBuf, D::Library>,
    device: D,
}

/// A plan's objects on the device.
struct Made<D: ComputeDevice> {
    buffers: Vec<D::Buffer>,
    states: Vec<D::ComputePipelineState>,
    /// The index in `states` of each dispatch's.
    state_of: Vec<usize>,
}

impl<D: ComputeDevice> Harness<D> {
    /// The harness on `device`, with the authored library of the AIR `authored`
    /// ([`authored::library_air`](crate::authored::library_air)).
    pub fn new(device: D, authored: &[u8]) -> Result<Self> {
        let authored = device.new_library(Code::Air(authored))?;
        let copy =
            device.new_compute_pipeline_state(&authored.new_function(COPY_FUNCTION, &[])?)?;
        Ok(Harness {
            queue: device.new_command_queue()?,
            copy,
            authored,
            metallibs: HashMap::new(),
            device,
        })
    }

    /// Commits a run of `plan`, then waits for `in_flight`, the run before it, which the new one
    /// replaces: that run's GPU time in microseconds, and the new run's encoding time, from its
    /// command buffer's creation through `commit`'s return.
    fn next(
        &self,
        plan: &Plan,
        made: &Made<D>,
        in_flight: &mut D::CommandBuffer,
    ) -> Result<(f64, Duration)> {
        let started = Instant::now();
        let command_buffer = self.encode(plan, made)?;
        command_buffer.commit()?;
        let encoded = started.elapsed();
        let before = std::mem::replace(in_flight, command_buffer);
        before.wait_until_completed()?;
        let gpu = before.gpu_end_time()? - before.gpu_start_time()?;
        Ok((gpu * 1e6, encoded))
    }

    /// Makes the library `source` names, where it is a metallib no plan named before.
    fn load(&mut self, source: &Source) -> Result<()> {
        if let Source::Metallib(path) = source {
            if !self.metallibs.contains_key(path) {
                let bytes =
                    std::fs::read(path).map_err(|why| format!("read {}: {why}", path.display()))?;
                let library = self.device.new_library(Code::Metallib(&bytes))?;
                self.metallibs.insert(path.clone(), library);
            }
        }
        Ok(())
    }

    /// The library `source` names, which [`load`](Self::load) made.
    fn library(&self, source: &Source) -> &D::Library {
        match source {
            Source::Authored => &self.authored,
            Source::Metallib(path) => &self.metallibs[path],
        }
    }

    /// `plan`'s pipeline states, one per kernel, from its library, which
    /// [`load`](Self::load) made, and its buffers.
    fn make(&self, plan: &Plan) -> Result<Made<D>> {
        let library = self.library(&plan.library);
        // Each function and its constants' values, made once.
        let mut functions: Vec<(&str, &[Constant])> = Vec::new();
        let mut states = Vec::new();
        let mut state_of = Vec::with_capacity(plan.dispatches.len());
        for dispatch in &plan.dispatches {
            let key = (dispatch.function.as_str(), dispatch.constants.as_slice());
            let at = match functions.iter().position(|known| *known == key) {
                Some(at) => at,
                None => {
                    let function = library.new_function(key.0, key.1)?;
                    states.push(self.device.new_compute_pipeline_state(&function)?);
                    functions.push(key);
                    functions.len() - 1
                }
            };
            state_of.push(at);
        }
        let buffers = (plan.buffers.iter())
            .map(|buffer| self.device.new_buffer(buffer.length, buffer.storage))
            .collect::<Result<Vec<_>>>()?;
        Ok(Made {
            buffers,
            states,
            state_of,
        })
    }

    /// One run of `plan`: a command buffer of one compute encoder, ended, not committed.
    fn encode(&self, plan: &Plan, made: &Made<D>) -> Result<D::CommandBuffer> {
        let command_buffer = self.queue.command_buffer()?;
        let mut encoder = command_buffer.compute_command_encoder()?;
        let mut current = None;
        for (dispatch, &state) in plan.dispatches.iter().zip(&made.state_of) {
            if current != Some(state) {
                encoder.set_compute_pipeline_state(&made.states[state])?;
                current = Some(state);
            }
            for binding in &dispatch.bindings {
                match binding {
                    Binding::Buffer {
                        index,
                        buffer,
                        offset,
                    } => encoder.set_buffer(&made.buffers[*buffer], *offset, *index)?,
                    Binding::Bytes { index, bytes } => encoder.set_bytes(bytes, *index)?,
                }
            }
            match dispatch.grid {
                Grid::Threadgroups { groups, threads } => {
                    encoder.dispatch_threadgroups(groups, threads)?
                }
                Grid::Threads { threads, per_group } => {
                    encoder.dispatch_threads(threads, per_group)?
                }
            }
        }
        encoder.end_encoding()?;
        Ok(command_buffer)
    }

    /// Every buffer's starting contents: a `Shared` one's written through `contents`, a
    /// `Private` one's copied in from a `Shared` buffer, all in one command buffer, waited for.
    fn fill(&self, plans: &[BufferPlan], buffers: &[D::Buffer]) -> Result<()> {
        let mut staged = Vec::new();
        for (plan, buffer) in plans.iter().zip(buffers) {
            let Some(contents) = &plan.contents else {
                continue;
            };
            match plan.storage {
                Storage::Shared => write(buffer, contents)?,
                Storage::Private => {
                    let stage = self.device.new_buffer(plan.length, Storage::Shared)?;
                    write(&stage, contents)?;
                    staged.push((stage, buffer, plan.length));
                }
            }
        }
        if staged.is_empty() {
            return Ok(());
        }
        let command_buffer = self.queue.command_buffer()?;
        let mut encoder = command_buffer.compute_command_encoder()?;
        encoder.set_compute_pipeline_state(&self.copy)?;
        for (stage, buffer, length) in &staged {
            copy::<D>(&mut encoder, stage, buffer, *length)?;
        }
        encoder.end_encoding()?;
        command_buffer.commit()?;
        command_buffer.wait_until_completed()
    }

    /// The first `length` bytes of `buffer`: a `Shared` one's through `contents`, a `Private`
    /// one's copied out to a `Shared` buffer first.
    fn read(&self, buffer: &D::Buffer, storage: Storage, length: u64) -> Result<Vec<u8>> {
        if storage == Storage::Shared {
            return bytes(buffer, length);
        }
        // Buffer lengths are whole multiples of 16 (`Plan::validate`), so this is within it.
        let whole = length.next_multiple_of(16);
        let stage = self.device.new_buffer(whole, Storage::Shared)?;
        let command_buffer = self.queue.command_buffer()?;
        let mut encoder = command_buffer.compute_command_encoder()?;
        encoder.set_compute_pipeline_state(&self.copy)?;
        copy::<D>(&mut encoder, buffer, &stage, whole)?;
        encoder.end_encoding()?;
        command_buffer.commit()?;
        command_buffer.wait_until_completed()?;
        bytes(&stage, length)
    }
}

impl<D: ComputeDevice> Session for Harness<D> {
    fn device_name(&self) -> String {
        self.device.name()
    }

    fn run(&mut self, plan: &Plan, runs: Runs) -> Result<Report> {
        plan.validate()?;
        self.load(&plan.library)?;
        let made = self.make(plan)?;
        self.fill(&plan.buffers, &made.buffers)?;
        let first = self.encode(plan, &made)?;
        first.commit()?;
        first.wait_until_completed()?;
        for expect in &plan.expect {
            let storage = plan.buffers[expect.buffer].storage;
            let bytes = self.read(
                &made.buffers[expect.buffer],
                storage,
                expect.values.length(),
            )?;
            let buffer = expect.buffer;
            (expect.values.compare(&bytes))
                .map_err(|why| format!("{}: buffer {buffer}: {why}", plan.name))?;
        }
        let mut outputs = Vec::with_capacity(plan.outputs.len());
        for output in &plan.outputs {
            let buffer = &plan.buffers[output.buffer];
            outputs.push(self.read(&made.buffers[output.buffer], buffer.storage, buffer.length)?);
        }
        // The runs after the first keep two command buffers in flight: each is committed while
        // the GPU runs the one before it, so the GPU does not idle between runs and its clock
        // holds, as it does under a program that keeps it busy.
        let mut in_flight = self.encode(plan, &made)?;
        in_flight.commit()?;
        let warmup = warm_up(runs, || Ok(self.next(plan, &made, &mut in_flight)?.0))?;
        let mut report = Report {
            dispatches: plan.dispatches.len(),
            outputs,
            warmup,
            ..Report::default()
        };
        for run in 1..=runs.timed {
            let cpu = clocks::process_cpu_nanos();
            let (gpu_us, encoded) = self.next(plan, &made, &mut in_flight)?;
            if let (Some(before), Some(after)) = (cpu, clocks::process_cpu_nanos()) {
                report.cpu_ns.push(after.saturating_sub(before) as f64);
            }
            report.encode_ns.push(encoded.as_nanos() as f64);
            report.gpu_us.push(gpu_us);
            if run == 100 || run == 200 {
                report.gauges.push((run, self.device.gauges()));
            }
        }
        in_flight.wait_until_completed()?;
        Ok(report)
    }

    fn create(&mut self, length: u64, runs: Runs) -> Result<CreateReport> {
        create(&self.device, length, runs)
    }

    fn finish(self: Box<Self>) -> Option<Vec<(&'static str, u64)>> {
        let reader = self.device.gauges_after_drop();
        drop(self);
        reader.map(|read| read())
    }

    fn pipelines_created(&self) -> u64 {
        self.device.pipelines_created()
    }
}

/// A device whose runtime makes buffers but runs no compute work: its creation workloads alone.
pub struct Buffers<D: Device>(pub D);

impl<D: Device> Session for Buffers<D> {
    fn device_name(&self) -> String {
        self.0.name()
    }

    fn run(&mut self, plan: &Plan, _: Runs) -> Result<Report> {
        Err(format!(
            "{}: the runtime makes buffers only, with no command queue",
            plan.name
        )
        .into())
    }

    fn create(&mut self, length: u64, runs: Runs) -> Result<CreateReport> {
        create(&self.0, length, runs)
    }

    fn finish(self: Box<Self>) -> Option<Vec<(&'static str, u64)>> {
        let reader = self.0.gauges_after_drop();
        drop(self);
        reader.map(|read| read())
    }

    fn pipelines_created(&self) -> u64 {
        self.0.pipelines_created()
    }
}

/// [`Session::create`] on `device`.
fn create<D: Device>(device: &D, length: u64, runs: Runs) -> Result<CreateReport> {
    // The range the first creation takes: a buffer of the length, written, released.
    scribble(&device.new_buffer(length, Storage::Shared)?, length)?;
    // A creation's time, its buffer checked zero at every `stride`th byte, then written all over
    // and released.
    let once = |stride| -> Result<f64> {
        let started = Instant::now();
        let buffer = device.new_buffer(length, Storage::Shared)?;
        let created = started.elapsed();
        zeroed(&buffer, length, stride)?;
        scribble(&buffer, length)?;
        Ok(created.as_secs_f64() * 1e6)
    };
    once(4)?;
    let mut report = CreateReport {
        warmup: warm_up(runs, || once(4096))?,
        ..CreateReport::default()
    };
    for run in 1..=runs.timed {
        report.create_us.push(once(4096)?);
        if run == 100 || run == 200 {
            report.gauges.push((run, device.gauges()));
        }
    }
    Ok(report)
}

/// Dispatches of the copy kernel, already set on `encoder`, moving the first `length` bytes (a
/// multiple of 16) of `from` to `to`, in as many dispatches as the threadgroup count limit needs.
fn copy<D: ComputeDevice>(
    encoder: &mut D::ComputeCommandEncoder,
    from: &D::Buffer,
    to: &D::Buffer,
    length: u64,
) -> Result<()> {
    let most = COPY_GROUPS * 256 * 16;
    let mut offset = 0;
    while offset < length {
        let bytes = (length - offset).min(most);
        encoder.set_buffer(from, offset, 0)?;
        encoder.set_buffer(to, offset, 1)?;
        encoder.dispatch_threads([bytes / 16, 1, 1], [256, 1, 1])?;
        offset += bytes;
    }
    Ok(())
}

/// `Ok` where the first `length` bytes of the `Shared` buffer `buffer` are zero at every
/// `stride`th byte (a multiple of 4), which no GPU work writes now.
pub fn zeroed(buffer: &impl Buffer, length: u64, stride: u64) -> Result<()> {
    let contents = buffer.contents().ok_or("a Shared buffer has no contents")?;
    for at in (0..length.saturating_sub(3)).step_by(stride as usize) {
        // SAFETY: `contents` holds the buffer's bytes, at least `length` of them (the caller's
        // length), and `at + 4` is within them; no command buffer writes them now.
        let word = unsafe {
            contents
                .as_ptr()
                .add(at as usize)
                .cast::<u32>()
                .read_unaligned()
        };
        if word != 0 {
            return Err(
                format!("a new buffer of {length} bytes holds {word:#x} at byte {at}").into(),
            );
        }
    }
    Ok(())
}

/// The first `length` bytes of the `Shared` buffer `buffer`, which no GPU work uses now, all set
/// to a byte other than zero.
fn scribble(buffer: &impl Buffer, length: u64) -> Result<()> {
    let contents = buffer.contents().ok_or("a Shared buffer has no contents")?;
    // SAFETY: `contents` holds the buffer's bytes, at least `length` of them (the caller's
    // length), and no command buffer uses them now.
    unsafe { std::ptr::write_bytes(contents.as_ptr(), 0xa5, length as usize) };
    Ok(())
}

/// `bytes` written at the start of the `Shared` buffer `buffer`, which no GPU work uses now.
fn write(buffer: &impl Buffer, bytes: &[u8]) -> Result<()> {
    let contents = buffer.contents().ok_or("a Shared buffer has no contents")?;
    // SAFETY: `contents` holds the buffer's bytes, at least `bytes.len()` of them (the plan's
    // length, `Plan::validate`), and no command buffer uses them now.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    Ok(())
}

/// A copy of the first `length` bytes of the `Shared` buffer `buffer`, which no GPU work writes
/// now.
fn bytes(buffer: &impl Buffer, length: u64) -> Result<Vec<u8>> {
    let contents = buffer.contents().ok_or("a Shared buffer has no contents")?;
    // SAFETY: `contents` holds the buffer's bytes, at least `length` of them (the caller's
    // length), and no command buffer writes them now.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) };
    Ok(bytes.to_vec())
}
