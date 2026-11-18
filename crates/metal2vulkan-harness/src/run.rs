//! A plan on a device: its objects made, its first run checked, the runs after it timed.

use crate::authored::{Makes, STATE_CREATIONS};
use crate::plan::{
    Binding, Blit, BufferPlan, Constant, Grid, Plan, SamplerPlan, Scaled, Source, TexturePlan,
    TextureSource,
};
use crate::{
    clocks, BlitCommandEncoder, Buffer, Code, CommandBuffer, CommandQueue, ComputeCommandEncoder,
    ComputeDevice, Device, Failed, Library, Refused, RenderCommandEncoder, Result, Storage,
    Texture, Waits,
};
use metal2vulkan_devtools::instructions::ProcessInstructions;
use std::collections::HashMap;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

/// The whole process's user-space instruction counter, once [`count_instructions`] opened it;
/// None where the kernel refused it (`perf_event_paranoid` above 2, no hardware counters, or not
/// Linux), and then the harness records no instructions.
static INSTRUCTIONS: OnceLock<Option<ProcessInstructions>> = OnceLock::new();

/// Opens the process's instruction counter. The harness calls it before it makes any device, so
/// that the threads a runtime and its driver start count on it.
pub fn count_instructions() {
    INSTRUCTIONS.get_or_init(|| ProcessInstructions::new().ok());
}

/// The process's user-space instructions so far, where its counter is open.
fn instructions() -> Option<u64> {
    INSTRUCTIONS.get()?.as_ref()?.read().ok()
}

/// The kernel of the authored library that moves bytes for the harness, 16 a thread.
const COPY_FUNCTION: &str = "copy_words";

/// The kernels of the authored library that move texels for the harness, between a buffer of
/// `float4`s and a texture, one a thread, in threadgroups of 16x16: into the texture, and out.
const TEXELS_FUNCTIONS: [&str; 2] = ["texels_in", "texels_out"];

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

/// The fewest timed runs that give a workload a time ([`steady`]).
pub const LEAST_TIMED: usize = 5;

/// A workload's time from its timed runs' `times`: the lowest mean of their consecutive windows
/// of 10 runs, the time 10 runs took over 10, so that a clock that drops partway through the runs
/// does not move it; with fewer than 10 runs, their mean; `None` with fewer than [`LEAST_TIMED`].
/// A mean, not a median: where a runtime reports two runs' completions together, one run's wait
/// takes both runs' time and the next run's none, and a median of such alternating runs is the
/// short one's, while their mean is the time each run took.
pub fn steady(times: &[f64]) -> Option<f64> {
    if times.len() < LEAST_TIMED {
        return None;
    }
    let mean = |window: &[f64]| window.iter().sum::<f64>() / window.len() as f64;
    if times.len() < WINDOW {
        return Some(mean(times));
    }
    (times.as_chunks::<WINDOW>().0.iter())
        .map(|window| mean(window))
        .min_by(f64::total_cmp)
}

/// The time the matrix judges a plan's row by, in microseconds: what a program waits on, timed
/// on the host. Where the plan times each command buffer's commit to its completed handler
/// ([`Plan::commit_to_complete`]), the [`steady`] time of each run's last command buffer's
/// commit-to-complete time: a program that waits for its run waits for its last command buffer,
/// and the earlier ones' lateness ([`Report::late_us`]) is a column of its own. Else the
/// [`steady`] time of the runs' wall times ([`Report::wall_us`]). A run's GPU time
/// ([`Report::gpu_us`]) decides no row.
pub fn matrix_time(report: &Report) -> Option<f64> {
    match report.commit_us.last() {
        None => steady(&report.wall_us),
        Some(last) => steady(last),
    }
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
    /// Each run's GPU time, from its first command buffer's `GPUStartTime` to its last's
    /// `GPUEndTime`, in microseconds: a diagnostic, which decides no row ([`matrix_time`]).
    pub gpu_us: Vec<f64>,
    /// Each timed run's host wall time, from the start of its encoding until the wait for the run
    /// before it returned, in microseconds: with two runs in flight, the time each run adds. The
    /// 100th and 200th runs' times leave out the wait for the device to go idle that reads the
    /// gauges, and an untimed run follows that wait, so the next timed run waits for a run in
    /// flight, as every other does.
    pub wall_us: Vec<f64>,
    /// The runtime's counts of its work ([`Device::counts`]) over one run after the timed ones,
    /// committed alone and waited for, so nothing else is in flight.
    pub counts: Vec<(&'static str, u64)>,
    /// The runtime's submissions ([`Device::submissions`]) over that run.
    pub submissions: u64,
    /// The runtime's submissions over the timed runs, two in flight, per run: how many its
    /// grouping made of them.
    pub timed_submissions: f64,
    /// The submit infos those submissions passed, per run ([`Device::submit_infos`]); `None`
    /// where the runtime does not count them.
    pub timed_submit_infos: Option<f64>,
    /// Each run's encoding thread's wall time from the command buffer's creation through
    /// `commit`'s return, in nanoseconds.
    pub encode_ns: Vec<f64>,
    /// Each run's process CPU time from before its command buffer's creation through the return
    /// of the wait for the run before it, in nanoseconds; empty where the host has no such clock.
    pub cpu_ns: Vec<f64>,
    /// Each run's user-space instructions in the whole process over the same span, every thread
    /// of the runtime and its driver included ([`count_instructions`]); empty where the kernel
    /// refused the counter.
    pub instructions: Vec<u64>,
    /// Each timed command buffer's time from just before its `commit` until its completed
    /// handler ran, in microseconds, by its place in its run: one list per place, the first
    /// committed first, each in the order the handlers ran. Empty unless the plan asks for it
    /// ([`Plan::commit_to_complete`]).
    pub commit_us: Vec<Vec<f64>>,
    /// Each of those command buffers' lateness, by its place in its run as
    /// [`commit_us`](Self::commit_us) is: from its `GPUEndTime` to its completed handler's start,
    /// both on the runtime's host clock, in microseconds. A command buffer reported complete only
    /// with a later one is late by the time between their ends.
    pub late_us: Vec<Vec<f64>>,
    /// Where the plan is [`Scaled`], the size it ran at and the time that size came from
    /// ([`run_each`]).
    pub scaled_to: Option<ScaledTo>,
    /// The device's gauges after the 100th and the 200th timed run, where there were that many,
    /// read once the work committed by then completed: what the runs left behind.
    pub gauges: Vec<(u32, Vec<(&'static str, u64)>)>,
    /// The contents of the plan's outputs after the first run, in its order.
    pub outputs: Vec<Vec<u8>>,
    /// The untimed runs.
    pub warmup: Warmup,
}

/// Microseconds of each timed run, one list per command buffer's place in its run.
type ByPlace = Vec<Vec<f64>>;

/// The commit-to-complete times and lateness that the completed handlers of a plan's timed
/// command buffers report ([`Plan::commit_to_complete`]).
#[derive(Default)]
struct Completions {
    /// The times reported, each with its command buffer's place in its run, in microseconds, in
    /// the order the handlers ran: its commit-to-complete time and its lateness.
    times: Mutex<Vec<(usize, f64, f64)>>,
    /// Signalled at each report.
    reported: Condvar,
}

/// How long [`Completions::wait`] waits for the handlers of command buffers already complete.
const HANDLERS_RUN: Duration = Duration::from_secs(10);

impl Completions {
    /// Reports the commit-to-complete time and the lateness, in microseconds, of the command
    /// buffer at `place` in its run.
    fn ran(&self, place: usize, time: Duration, late_us: f64) {
        let mut times = self.times.lock().unwrap_or_else(PoisonError::into_inner);
        times.push((place, time.as_secs_f64() * 1e6, late_us));
        self.reported.notify_all();
    }

    /// The commit-to-complete times and the lateness of `runs` runs of `places` command buffers
    /// each, by place, once their handlers have run: a completed handler may run after the wait
    /// for its command buffer returns. An error where fewer have run within [`HANDLERS_RUN`].
    fn wait(&self, runs: usize, places: usize) -> Result<(ByPlace, ByPlace)> {
        let count = runs * places;
        let times = self.times.lock().unwrap_or_else(PoisonError::into_inner);
        let (times, _) = (self.reported)
            .wait_timeout_while(times, HANDLERS_RUN, |times| times.len() < count)
            .unwrap_or_else(PoisonError::into_inner);
        if times.len() < count {
            return Err(format!("{} of {count} completed handlers ran", times.len()).into());
        }
        let mut by_place = vec![Vec::with_capacity(runs); places];
        let mut late = vec![Vec::with_capacity(runs); places];
        for &(place, us, late_us) in times.iter() {
            by_place[place].push(us);
            late[place].push(late_us);
        }
        Ok((by_place, late))
    }
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

    /// The objects `makes` says, made and released in turn. `Shared` buffers of a length, so each
    /// takes the range the last one left: one checked whole, then `runs`, each checked to start
    /// zero (every 4 KiB), then written all over before its release. Compute pipeline states of
    /// one content: the first, of a content the device has not seen, timed apart, then `runs`'
    /// warm-up and [`crate::authored::STATE_CREATIONS`] timed ones.
    fn create(&mut self, makes: Makes, runs: Runs) -> Result<CreateReport>;

    /// Drops the session, its device last; the device's gauges after that, where the runtime's
    /// gauges outlive the device.
    fn finish(self: Box<Self>) -> Option<Vec<(&'static str, u64)>>;

    /// The device's [`Device::pipelines_created`].
    fn pipelines_created(&self) -> u64;

    /// The device's [`Device::first_use_waits`].
    fn first_use_waits(&self) -> Option<Waits>;
}

/// The size a [`Scaled`] workload ran at, and the time it was sized from.
#[derive(Clone, Copy, Debug)]
pub struct ScaledTo {
    /// The size timed first.
    pub base: u32,
    /// The GPU time of a command buffer at `base` on the first runtime that ran it, in
    /// microseconds: the median run's over its command buffers, its dispatches timed in one
    /// command buffer. `None` where the caller gave the size ([`run_each`]).
    pub base_us: Option<f64>,
    /// The size it ran at ([`Scaled::size`]).
    pub size: u32,
}

/// `plan` on each session in turn: its report, or its runtime's typed refusal. An error where a
/// session failed ([`Failed`], naming its runtime), or where the outputs of the sessions that ran
/// it differ. A [`Scaled`] plan runs at `size` where it is given, so that a runtime timed in
/// another process times the same work as the one that sized it there; else it is timed at its
/// base size on the first session that runs it, and every session runs it at the size that time
/// gives ([`Report::scaled_to`]), so the runtimes time the same work. An error where `size` is
/// given for a plan that is not [`Scaled`].
pub fn run_each(
    plan: &Plan,
    sessions: &mut [(&str, Box<dyn Session>)],
    runs: Runs,
    size: Option<u32>,
) -> Result<Vec<std::result::Result<Report, Box<Refused>>>> {
    let mut sized = match (plan.scaled, size) {
        (_, None) => None,
        (Some(scaled), Some(size)) => Some(ScaledTo {
            base: scaled.base,
            base_us: None,
            size,
        }),
        (None, Some(_)) => return Err("a size is given for a workload of one size".into()),
    };
    let mut each = Vec::with_capacity(sessions.len());
    for (runtime, session) in sessions.iter_mut() {
        let ran = match plan.scaled {
            None => session.run(plan, runs),
            Some(scaled) => run_scaled(&mut **session, plan, scaled, &mut sized, runs),
        };
        each.push(match ran {
            Ok(report) => Ok(report),
            Err(why) => Err(why.downcast::<Refused>().map_err(|why| Failed {
                runtime: runtime.to_string(),
                why,
            })?),
        });
    }
    let mut ran = each.iter().filter_map(|run| run.as_ref().ok());
    if let Some(first) = ran.next() {
        for report in ran {
            compare_outputs(plan, first, report)?;
        }
    }
    Ok(each)
}

/// `plan`, a [`Scaled`] workload at its base size, on `session` at the size `sized` holds. Where
/// no session sized it yet, this one times it at the base first and sizes it by a command
/// buffer's GPU time there: the median of three runs over its command buffers, after a warm-up
/// that goes on until the time settles, as [`warm_up`] does for the timed workloads, since a
/// clock still rising reads the base long. The base is timed with all its dispatches in one
/// command buffer: a run of several command buffers is timed from the first's `GPUStartTime` to
/// the last's `GPUEndTime`, which can count less than their work where they complete together,
/// while one command buffer's times bound its work alone.
fn run_scaled(
    session: &mut dyn Session,
    plan: &Plan,
    scaled: Scaled,
    sized: &mut Option<ScaledTo>,
    runs: Runs,
) -> Result<Report> {
    let to = match *sized {
        Some(to) => to,
        None => {
            let timing = Runs {
                warmup: 2,
                settle: true,
                timed: 3,
            };
            let as_one = Plan {
                per_command_buffer: None,
                commit_to_complete: false,
                ..plan.clone()
            };
            let at_base = session.run(&as_one, timing)?;
            let base_us =
                percentile(&at_base.gpu_us, 50).unwrap_or(0.0) / plan.command_buffers() as f64;
            *sized.insert(ScaledTo {
                base: scaled.base,
                base_us: Some(base_us),
                size: scaled.size(base_us),
            })
        }
    };
    let mut report = session.run(&(scaled.build)(to.size), runs)?;
    report.scaled_to = Some(to);
    Ok(report)
}

/// What the timed creations measured.
#[derive(Clone, Debug, Default)]
pub struct CreateReport {
    /// Each timed creation call's wall time, in microseconds.
    pub create_us: Vec<f64>,
    /// The first creation's wall time, in microseconds, where it is timed apart: a pipeline
    /// state's of a content the device had not seen. A compute pipeline state's holds one
    /// dispatch of it through to completion, since a runtime may pay its pipeline's compile in the
    /// creation or at the first dispatch and the sum holds both, and the creation of the 16-byte
    /// buffer the dispatch writes, the guest's take of memory beside it; a render pipeline state's
    /// is its creation alone.
    pub first_us: Option<f64>,
    /// The first creation's wall time alone, in microseconds: for a pipeline state, the part of
    /// [`first_us`](Self::first_us) the creation call paid, all of it for a render pipeline
    /// state; for a buffer, the device's first take of memory for a buffer of its length.
    pub first_create_us: Option<f64>,
    /// For a buffer, the second creation's wall time, in microseconds: the first after the first
    /// buffer was released, which a runtime may serve from the memory that buffer gave back.
    pub again_us: Option<f64>,
    /// The runtime's memory allocations on the caller's thread during the first creation (with
    /// its dispatch and its buffer, for a compute pipeline state): their count and time, and the
    /// longest since the device was made ([`Device::allocations`]); `None` where it does not count
    /// them.
    pub first_allocations: Option<Waits>,
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

/// The harness on a device: its own queue, the authored library and its copy kernel, its texel
/// kernels once a plan moves texels, and the metallibs the plans named so far.
pub struct Harness<D: ComputeDevice> {
    queue: D::CommandQueue,
    copy: D::ComputePipelineState,
    /// The pipeline states of [`TEXELS_FUNCTIONS`], made for the first plan that moves texels
    /// ([`Plan::moves_texels`]), so a session that runs none makes no more pipelines than before.
    texels: Option<[D::ComputePipelineState; 2]>,
    authored: D::Library,
    /// The render library, made for the first plan that names it.
    render: Option<D::Library>,
    metallibs: HashMap<PathBuf, D::Library>,
    device: D,
}

/// A plan's objects on the device.
struct Made<D: ComputeDevice> {
    buffers: Vec<D::Buffer>,
    textures: Vec<D::Texture>,
    /// A sampler state for each descriptor the plan's bindings name.
    samplers: Vec<(SamplerPlan, D::SamplerState)>,
    states: Vec<D::ComputePipelineState>,
    /// The render pipeline state of each vertex function, fragment function and target the
    /// plan's draws name.
    render_states: Vec<D::RenderPipelineState>,
    /// The index in `states` of each dispatch's, or in `render_states` of each draw's.
    state_of: Vec<usize>,
}

impl<D: ComputeDevice> Harness<D> {
    /// The harness on `device`, with the authored library of the AIR `authored`
    /// ([`authored::library_air`](crate::authored::library_air)).
    pub fn new(device: D, authored: &[u8]) -> Result<Self> {
        let authored = device.new_library(Code::Air(authored))?;
        let copy =
            device.new_compute_pipeline_state(&authored.new_function(COPY_FUNCTION, &[])?, &[])?;
        Ok(Harness {
            queue: device.new_command_queue()?,
            copy,
            texels: None,
            authored,
            render: None,
            metallibs: HashMap::new(),
            device,
        })
    }

    /// Commits a run of `plan`, then waits for `in_flight`, the run before it, which the new one
    /// replaces: that run's GPU time in microseconds, and the new run's encoding time, from its
    /// first command buffer's creation through its last `commit`'s return.
    /// Where `completions` is given, each command buffer of the new run reports its
    /// commit-to-complete time there.
    fn next(
        &self,
        plan: &Plan,
        made: &Made<D>,
        in_flight: &mut Vec<D::CommandBuffer>,
        completions: Option<&Arc<Completions>>,
    ) -> Result<(f64, Duration)> {
        let started = Instant::now();
        let run = self.commit_run(plan, made, completions)?;
        let encoded = started.elapsed();
        let before = std::mem::replace(in_flight, run);
        Ok((wait_run::<D>(&before)?, encoded))
    }

    /// Makes the library `source` names, where it is the render library or a metallib no plan
    /// named before.
    fn load(&mut self, source: &Source) -> Result<()> {
        if *source == Source::Render && self.render.is_none() {
            let air = crate::authored::render_library_air()?;
            self.render = Some(self.device.new_library(Code::Air(&air))?);
        }
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
            Source::Render => (self.render.as_ref()).expect("`load` made the render library"),
            Source::Metallib(path) => &self.metallibs[path],
        }
    }

    /// `plan`'s textures and sampler states, its pipeline states, one per kernel, from its
    /// library, which [`load`](Self::load) made, and its buffers. The textures come first: a
    /// runtime that makes none refuses the plan for that before its kernels are translated.
    fn make(&self, plan: &Plan) -> Result<Made<D>> {
        let mut textures: Vec<D::Texture> = Vec::with_capacity(plan.textures.len());
        for texture in &plan.textures {
            let made = match texture.source {
                TextureSource::View { parent, slice } => {
                    let parent = (textures.get(parent)).ok_or("a view's parent is unmade")?;
                    let slices = slice..slice + texture.array_length;
                    self.device.new_texture_view(parent, texture, slices)?
                }
                _ => self.device.new_texture(texture)?,
            };
            textures.push(made);
        }
        let mut samplers: Vec<(SamplerPlan, D::SamplerState)> = Vec::new();
        for binding in plan
            .dispatches
            .iter()
            .flat_map(|dispatch| &dispatch.bindings)
        {
            if let Binding::Sampler { sampler, .. } = binding {
                if !samplers.iter().any(|(made, _)| made == sampler) {
                    samplers.push((*sampler, self.device.new_sampler_state(*sampler)?));
                }
            }
        }
        let library = self.library(&plan.library);
        // Each function with its constants' values and its linked functions, made once.
        let mut functions: Vec<(&str, &[Constant], &[String])> = Vec::new();
        let mut states = Vec::new();
        let mut renders: Vec<(&str, &str, usize)> = Vec::new();
        let mut render_states = Vec::new();
        let mut state_of = Vec::with_capacity(plan.dispatches.len());
        for dispatch in &plan.dispatches {
            if let Grid::Draw {
                fragment, target, ..
            } = dispatch.grid
            {
                let key = (dispatch.function.as_str(), fragment, target);
                let at = match renders.iter().position(|known| *known == key) {
                    Some(at) => at,
                    None => {
                        let vertex = library.new_function(key.0, &dispatch.constants)?;
                        let fragment = library.new_function(fragment, &dispatch.constants)?;
                        let format = plan.textures[target].pixel_format;
                        render_states.push(
                            self.device
                                .new_render_pipeline_state(&vertex, &fragment, format)?,
                        );
                        renders.push(key);
                        renders.len() - 1
                    }
                };
                state_of.push(at);
                continue;
            }
            let key = (
                dispatch.function.as_str(),
                dispatch.constants.as_slice(),
                dispatch.linked.as_slice(),
            );
            let at = match functions.iter().position(|known| *known == key) {
                Some(at) => at,
                None => {
                    let function = library.new_function(key.0, key.1)?;
                    let private = (key.2.iter())
                        .map(|name| library.new_function(name, key.1))
                        .collect::<Result<Vec<_>>>()?;
                    states.push(
                        self.device
                            .new_compute_pipeline_state(&function, &private)?,
                    );
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
            textures,
            samplers,
            states,
            render_states,
            state_of,
        })
    }

    /// A run of `plan`, committed: its command buffers in order, `per_command_buffer`
    /// dispatches each (or all in one), with no wait between. Where `completions` is given, each
    /// takes a completed handler that reports its commit-to-complete time there, with its place.
    fn commit_run(
        &self,
        plan: &Plan,
        made: &Made<D>,
        completions: Option<&Arc<Completions>>,
    ) -> Result<Vec<D::CommandBuffer>> {
        for (buffer, object) in plan.buffers.iter().zip(&made.buffers) {
            if let Some(range) = &buffer.modified {
                modify(object, buffer, range)?;
            }
        }
        parts(0..plan.dispatches.len(), plan.per_command_buffer)
            .enumerate()
            .map(|(place, range)| {
                let command_buffer = self.encode(plan, made, range)?;
                if let Some(completions) = completions {
                    let (completions, committed) = (Arc::clone(completions), Instant::now());
                    command_buffer.add_completed_handler(Box::new(move |late_us| {
                        completions.ran(place, committed.elapsed(), late_us);
                    }))?;
                }
                command_buffer.commit()?;
                Ok(command_buffer)
            })
            .collect()
    }

    /// The dispatches `range` of `plan`: a command buffer of compute encoders of `per_encoder`
    /// dispatches each (or one of them all), ended, not committed.
    fn encode(&self, plan: &Plan, made: &Made<D>, range: Range<usize>) -> Result<D::CommandBuffer> {
        let command_buffer = self.queue.command_buffer()?;
        for range in parts(range, plan.per_encoder) {
            self.encode_pass(&command_buffer, plan, made, range)?;
        }
        Ok(command_buffer)
    }

    /// The dispatches `range` of `plan` in a compute encoder of `command_buffer`, ended. A
    /// dispatch's blit commands end the compute encoder before them and go in a blit encoder; a
    /// new compute encoder takes the dispatch.
    fn encode_pass(
        &self,
        command_buffer: &D::CommandBuffer,
        plan: &Plan,
        made: &Made<D>,
        range: Range<usize>,
    ) -> Result<()> {
        let mut open: Option<D::ComputeCommandEncoder> = None;
        let mut current = None;
        // The open render encoder, its target and its pipeline state.
        let mut pass: Option<(D::RenderCommandEncoder, usize, Option<usize>)> = None;
        let states = &made.state_of[range.clone()];
        for (dispatch, &state) in plan.dispatches[range].iter().zip(states) {
            if let Grid::Draw {
                target,
                vertex_count,
                ..
            } = dispatch.grid
            {
                if let Some(encoder) = open.take() {
                    encoder.end_encoding()?;
                }
                current = None;
                if let Some((encoder, _, _)) = pass.take_if(|(_, into, _)| *into != target) {
                    encoder.end_encoding()?;
                }
                let (encoder, _, bound) = match pass {
                    Some(ref mut pass) => pass,
                    None => {
                        let texture = made.textures.get(target);
                        let texture = texture.ok_or("a texture the plan names is unmade")?;
                        pass.insert((
                            command_buffer.render_command_encoder(texture)?,
                            target,
                            None,
                        ))
                    }
                };
                if *bound != Some(state) {
                    encoder.set_render_pipeline_state(&made.render_states[state])?;
                    *bound = Some(state);
                }
                for binding in &dispatch.bindings {
                    if let Binding::Buffer {
                        index,
                        buffer,
                        offset,
                    } = binding
                    {
                        encoder.set_vertex_buffer(&made.buffers[*buffer], *offset, *index)?;
                    }
                }
                encoder.draw_triangles(0, vertex_count)?;
                continue;
            }
            if let Some((encoder, _, _)) = pass.take() {
                encoder.end_encoding()?;
            }
            if !dispatch.blits.is_empty() {
                if let Some(encoder) = open.take() {
                    encoder.end_encoding()?;
                }
                current = None;
                let mut encoder = command_buffer.blit_command_encoder()?;
                let texture =
                    |at: usize| (made.textures.get(at)).ok_or("a texture the plan names is unmade");
                let buffer =
                    |at: usize| (made.buffers.get(at)).ok_or("a buffer the plan names is unmade");
                for blit in &dispatch.blits {
                    match blit {
                        Blit::Texture(copy) => encoder.copy_from_texture(
                            copy,
                            texture(copy.source)?,
                            texture(copy.destination)?,
                        )?,
                        Blit::FromBuffer(copy) => encoder.copy_from_buffer(
                            copy,
                            buffer(copy.buffer)?,
                            texture(copy.texture)?,
                        )?,
                        Blit::ToBuffer(copy) => encoder.copy_from_texture_to_buffer(
                            copy,
                            texture(copy.texture)?,
                            buffer(copy.buffer)?,
                        )?,
                        Blit::Fill(fill) => encoder.fill_buffer(
                            buffer(fill.buffer)?,
                            fill.range.clone(),
                            fill.value,
                        )?,
                        &Blit::Synchronize(at) => encoder.synchronize_resource(buffer(at)?)?,
                    }
                }
                encoder.end_encoding()?;
            }
            let encoder = match open {
                Some(ref mut encoder) => encoder,
                None => open.insert(command_buffer.compute_command_encoder()?),
            };
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
                    Binding::Texture { index, texture } => {
                        encoder.set_texture(&made.textures[*texture], *index)?
                    }
                    Binding::Sampler { index, sampler } => {
                        // `make` made one for every descriptor a binding names.
                        let made = made.samplers.iter().find(|(made, _)| made == sampler);
                        let (_, sampler) =
                            made.ok_or("a sampler state the plan names is unmade")?;
                        encoder.set_sampler_state(sampler, *index)?
                    }
                    Binding::ThreadgroupMemory { index, length } => {
                        encoder.set_threadgroup_memory_length(*length, *index)?
                    }
                }
            }
            match dispatch.grid {
                Grid::Threadgroups { groups, threads } => {
                    encoder.dispatch_threadgroups(groups, threads)?
                }
                Grid::Threads { threads, per_group } => {
                    encoder.dispatch_threads(threads, per_group)?
                }
                Grid::Draw { .. } => unreachable!("a draw is recorded above"),
            }
        }
        if let Some((encoder, _, _)) = pass {
            encoder.end_encoding()?;
        }
        match open {
            Some(encoder) => encoder.end_encoding(),
            None => Ok(()),
        }
    }

    /// Every buffer's starting contents: a `Shared` one's written through `contents`, a
    /// `Managed` one's too, then `didModifyRange:` over them, and a `Private` one's, or a
    /// `Managed` one's without a CPU copy, copied in from a `Shared` buffer, all in one command
    /// buffer, waited for.
    fn fill(&self, plans: &[BufferPlan], buffers: &[D::Buffer]) -> Result<()> {
        let mut staged = Vec::new();
        for (plan, buffer) in plans.iter().zip(buffers) {
            let Some(contents) = &plan.contents else {
                continue;
            };
            match (plan.storage, buffer.contents()) {
                (Storage::Shared, _) => write(buffer, contents)?,
                (Storage::Managed, Some(_)) => {
                    write(buffer, contents)?;
                    buffer.did_modify_range(0, plan.length)?;
                }
                (Storage::Private | Storage::Managed, _) => {
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

    /// The texel kernels' pipeline states.
    fn make_texels(&self) -> Result<[D::ComputePipelineState; 2]> {
        let [into, out] = TEXELS_FUNCTIONS.map(|name| {
            let function = self.authored.new_function(name, &[])?;
            self.device.new_compute_pipeline_state(&function, &[])
        });
        Ok([into?, out?])
    }

    /// The texel kernels' pipeline states, which `run` makes for the first plan with a texture.
    fn texels(&self) -> Result<&[D::ComputePipelineState; 2]> {
        Ok(self.texels.as_ref().ok_or("the texel kernels are unmade")?)
    }

    /// Every texture's starting bytes, written by the CPU slice by slice, and its starting
    /// texels, written in from `Shared` buffers by the texel kernel, all in one command buffer,
    /// waited for.
    fn fill_textures(&self, plans: &[TexturePlan], textures: &[D::Texture]) -> Result<()> {
        for (plan, texture) in plans.iter().zip(textures) {
            let Some(bytes) = &plan.bytes else {
                continue;
            };
            // `Plan::validate` checked the format's size and that the bytes are whole slices.
            let row = plan
                .row_bytes()
                .ok_or("a texture's format has no known size")?;
            let image = usize::try_from(row * u64::from(plan.height))?;
            for (slice, image) in (0u32..).zip(bytes.chunks(image)) {
                texture.replace_region(slice, [plan.width, plan.height], image, row)?;
            }
        }
        let mut staged = Vec::new();
        for (plan, texture) in plans.iter().zip(textures) {
            let Some(texels) = &plan.texels else {
                continue;
            };
            let bytes: Vec<u8> = texels
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            let stage = self
                .device
                .new_buffer(bytes.len() as u64, Storage::Shared)?;
            write(&stage, &bytes)?;
            staged.push((stage, texture, plan));
        }
        if staged.is_empty() {
            return Ok(());
        }
        let [into, _] = self.texels()?;
        let command_buffer = self.queue.command_buffer()?;
        let mut encoder = command_buffer.compute_command_encoder()?;
        encoder.set_compute_pipeline_state(into)?;
        for (stage, texture, plan) in &staged {
            encoder.set_buffer(stage, 0, 0)?;
            encoder.set_bytes(&plan.width.to_le_bytes(), 1)?;
            encoder.set_texture(texture, 0)?;
            encoder.dispatch_threadgroups(tiles(plan), [16, 16, 1])?;
        }
        encoder.end_encoding()?;
        command_buffer.commit()?;
        command_buffer.wait_until_completed()
    }

    /// A texture's texels, row by row, four `float`s each: read out to a `Shared` buffer by the
    /// texel kernel.
    fn read_texture(&self, texture: &D::Texture, plan: &TexturePlan) -> Result<Vec<u8>> {
        let length = 16 * u64::from(plan.width) * u64::from(plan.height);
        let stage = self.device.new_buffer(length, Storage::Shared)?;
        let [_, out] = self.texels()?;
        let command_buffer = self.queue.command_buffer()?;
        let mut encoder = command_buffer.compute_command_encoder()?;
        encoder.set_compute_pipeline_state(out)?;
        encoder.set_texture(texture, 0)?;
        encoder.set_buffer(&stage, 0, 0)?;
        encoder.set_bytes(&plan.width.to_le_bytes(), 1)?;
        encoder.dispatch_threadgroups(tiles(plan), [16, 16, 1])?;
        encoder.end_encoding()?;
        command_buffer.commit()?;
        command_buffer.wait_until_completed()?;
        bytes(&stage, length)
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
        if plan.moves_texels() && self.texels.is_none() {
            self.texels = Some(self.make_texels()?);
        }
        self.fill(&plan.buffers, &made.buffers)?;
        for (i, (buffer, object)) in plan.buffers.iter().zip(&made.buffers).enumerate() {
            if buffer.modified.is_some() && object.contents().is_none() {
                return Err(Box::new(Refused(format!(
                    "{}: buffer {i}, which the CPU writes before each run, is {:?}, and this \
                     runtime gives it no CPU copy",
                    plan.name, buffer.storage
                ))));
            }
        }
        self.fill_textures(&plan.textures, &made.textures)?;
        wait_run::<D>(&self.commit_run(plan, &made, None)?)?;
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
        for expect in &plan.expect_textures {
            let texture = expect.texture;
            let bytes = self.read_texture(&made.textures[texture], &plan.textures[texture])?;
            (expect.values.compare(&bytes))
                .map_err(|why| format!("{}: texture {texture}: {why}", plan.name))?;
        }
        let mut outputs = Vec::with_capacity(plan.outputs.len());
        for output in &plan.outputs {
            let buffer = &plan.buffers[output.buffer];
            outputs.push(self.read(&made.buffers[output.buffer], buffer.storage, buffer.length)?);
        }
        // The runs after the first keep two command buffers in flight: each is committed while
        // the GPU runs the one before it, so the GPU does not idle between runs and its clock
        // holds, as it does under a program that keeps it busy.
        let mut in_flight = self.commit_run(plan, &made, None)?;
        let warmup = warm_up(runs, || Ok(self.next(plan, &made, &mut in_flight, None)?.0))?;
        let mut report = Report {
            dispatches: plan.dispatches.len(),
            outputs,
            warmup,
            ..Report::default()
        };
        let completions = (plan.commit_to_complete).then(|| Arc::new(Completions::default()));
        let (submitted, infos) = (self.device.submissions(), self.device.submit_infos());
        for run in 1..=runs.timed {
            let wall = Instant::now();
            let cpu = clocks::process_cpu_nanos();
            let retired = instructions();
            let (gpu_us, encoded) = self.next(plan, &made, &mut in_flight, completions.as_ref())?;
            if let (Some(before), Some(after)) = (cpu, clocks::process_cpu_nanos()) {
                report.cpu_ns.push(after.saturating_sub(before) as f64);
            }
            if let (Some(before), Some(after)) = (retired, instructions()) {
                report.instructions.push(after.saturating_sub(before));
            }
            report.wall_us.push(wall.elapsed().as_secs_f64() * 1e6);
            report.encode_ns.push(encoded.as_nanos() as f64);
            report.gpu_us.push(gpu_us);
            if run == 100 || run == 200 {
                // With the device idle, so the gauges count what the runs left behind, not the
                // work in flight at that moment, which moves with completion timing.
                wait_run::<D>(&in_flight)?;
                report.gauges.push((run, self.device.gauges()));
                // An untimed run, so two are in flight again: the next timed run's wait is then
                // for a run the GPU was running, not for this idle device's last one, whose wait
                // returns at once and would time the encoding alone.
                if run < runs.timed {
                    self.next(plan, &made, &mut in_flight, None)?;
                }
            }
        }
        wait_run::<D>(&in_flight)?;
        let timed = f64::from(runs.timed.max(1));
        report.timed_submissions = (self.device.submissions() - submitted) as f64 / timed;
        report.timed_submit_infos = (infos.zip(self.device.submit_infos()))
            .map(|(before, after)| (after - before) as f64 / timed);
        if let Some(completions) = completions {
            (report.commit_us, report.late_us) =
                completions.wait(runs.timed as usize, plan.command_buffers())?;
        }
        let (before, submissions) = (self.device.counts(), self.device.submissions());
        wait_run::<D>(&self.commit_run(plan, &made, None)?)?;
        report.counts = (before.into_iter().zip(self.device.counts()))
            .map(|((count, before), (_, after))| (count, after - before))
            .collect();
        report.submissions = self.device.submissions() - submissions;
        Ok(report)
    }

    fn create(&mut self, makes: Makes, runs: Runs) -> Result<CreateReport> {
        match makes {
            Makes::Buffers { length } => create(&self.device, length, runs),
            Makes::PrivateBuffers { length } => self.create_private(length, runs),
            Makes::States {
                function,
                constants,
            } => self.create_states(function, constants, runs),
            Makes::RenderStates {
                vertex,
                fragment,
                color,
            } => {
                let library = (self.device)
                    .new_library(Code::Air(&crate::authored::render_library_air()?))?;
                let vertex = library.new_function(vertex, &[])?;
                let fragment = library.new_function(fragment, &[])?;
                self.time_states(
                    || (self.device).new_render_pipeline_state(&vertex, &fragment, color),
                    None,
                    runs,
                )
            }
        }
    }

    fn finish(self: Box<Self>) -> Option<Vec<(&'static str, u64)>> {
        let reader = self.device.gauges_after_drop();
        drop(self);
        reader.map(|read| read())
    }

    fn pipelines_created(&self) -> u64 {
        self.device.pipelines_created()
    }

    fn first_use_waits(&self) -> Option<Waits> {
        self.device.first_use_waits()
    }
}

/// Waits for `run`, a run's command buffers in order, as a program waits once for a burst it
/// committed: for its last command buffer, then for the others, each wait returning at once for
/// one already complete. The run's GPU time in microseconds, from its first command buffer's
/// `GPUStartTime` to its last's `GPUEndTime`.
fn wait_run<D: ComputeDevice>(run: &[D::CommandBuffer]) -> Result<f64> {
    let (Some(first), Some(last)) = (run.first(), run.last()) else {
        unreachable!("a validated plan's run has a command buffer");
    };
    last.wait_until_completed()?;
    for command_buffer in run {
        command_buffer.wait_until_completed()?;
    }
    Ok((last.gpu_end_time()? - first.gpu_start_time()?) * 1e6)
}

/// `range` cut in order into parts of `per` (the last one shorter where `per` does not divide
/// it), or one part of it all where `per` is `None`. `per` and `range` hold at least 1:
/// `Plan::validate` refuses a plan of no dispatch, or a `per_command_buffer` or `per_encoder` of
/// 0.
pub(crate) fn parts(range: Range<usize>, per: Option<usize>) -> impl Iterator<Item = Range<usize>> {
    let (end, per) = (range.end, per.unwrap_or(range.len()));
    range
        .step_by(per)
        .map(move |start| start..(start + per).min(end))
}

impl<D: ComputeDevice> Harness<D> {
    /// [`Session::create`] of `Private` buffers of `length` bytes (at least 32): each made, then
    /// used once by the copy kernel, which moves its first and last 16 bytes into a `Shared`
    /// buffer, through to completion, the copy checked zero; timed from the creation to the
    /// completion.
    fn create_private(&self, length: u64, runs: Runs) -> Result<CreateReport> {
        let out = self.device.new_buffer(32, Storage::Shared)?;
        let once = || -> Result<f64> {
            scribble(&out, 32)?;
            let started = Instant::now();
            let buffer = self.device.new_buffer(length, Storage::Private)?;
            let command_buffer = self.queue.command_buffer()?;
            let mut encoder = command_buffer.compute_command_encoder()?;
            encoder.set_compute_pipeline_state(&self.copy)?;
            for (from, to) in [(0, 0), (length - 16, 16)] {
                encoder.set_buffer(&buffer, from, 0)?;
                encoder.set_buffer(&out, to, 1)?;
                encoder.dispatch_threadgroups([1, 1, 1], [1, 1, 1])?;
            }
            encoder.end_encoding()?;
            command_buffer.commit()?;
            command_buffer.wait_until_completed()?;
            let used = started.elapsed();
            zeroed(&out, 32, 4)?;
            Ok(used.as_secs_f64() * 1e6)
        };
        let before = self.device.allocations();
        let first = once()?;
        let first_allocations = since(before, self.device.allocations());
        let again = once()?;
        let mut report = CreateReport {
            first_create_us: Some(first),
            again_us: Some(again),
            first_allocations,
            warmup: warm_up(runs, &once)?,
            ..CreateReport::default()
        };
        for run in 1..=runs.timed {
            report.create_us.push(once()?);
            if run == 100 || run == 200 {
                report.gauges.push((run, self.device.gauges()));
            }
        }
        Ok(report)
    }

    /// [`Session::create`] of pipeline states of `function` with the `float` constants
    /// `constants`: each made, timed, and released. The first, of a content the device has not
    /// seen, is timed with one dispatch of it through to completion, its output checked.
    fn create_states(
        &self,
        function: &str,
        constants: &[(u32, f32)],
        runs: Runs,
    ) -> Result<CreateReport> {
        let values: Vec<Constant> = (constants.iter())
            .map(|&(index, value)| Constant {
                index,
                // `MTLDataTypeFloat`.
                data_type: 3,
                value: value.to_le_bytes().to_vec(),
            })
            .collect();
        let function = self.authored.new_function(function, &values)?;
        let before = self.device.allocations();
        let (first_create, first) = self.first_state(&function, constants)?;
        let allocations = since(before, self.device.allocations());
        let mut report = self.time_states(
            || self.device.new_compute_pipeline_state(&function, &[]),
            Some(first),
            runs,
        )?;
        report.first_create_us = Some(first_create);
        report.first_allocations = allocations;
        Ok(report)
    }

    /// [`Session::create`] of the states `make` makes: the first timed apart (`first`, where the
    /// caller timed it with more than its creation; else its creation here, which is then its
    /// [`first_create_us`](CreateReport::first_create_us) too), then `runs`' warm-up and
    /// [`STATE_CREATIONS`] timed ones, each released before the next.
    fn time_states<S>(
        &self,
        make: impl Fn() -> Result<S>,
        first: Option<f64>,
        runs: Runs,
    ) -> Result<CreateReport> {
        let once = || -> Result<f64> {
            let started = Instant::now();
            let state = make()?;
            let made = started.elapsed();
            drop(state);
            Ok(made.as_secs_f64() * 1e6)
        };
        let (first, first_create_us) = match first {
            Some(first) => (first, None),
            None => {
                let first = once()?;
                (first, Some(first))
            }
        };
        let mut report = CreateReport {
            first_us: Some(first),
            first_create_us,
            warmup: warm_up(runs, &once)?,
            ..CreateReport::default()
        };
        for run in 1..=STATE_CREATIONS {
            report.create_us.push(once()?);
            if run == 100 || run == 200 {
                report.gauges.push((run, self.device.gauges()));
            }
        }
        Ok(report)
    }

    /// The first state of `function` (`signed_zero_products`, with the `float` constants
    /// `constants` as its two products' factors) made, then dispatched once, one thread of one
    /// round, through to completion: the creation's wall time and the whole's, in microseconds.
    /// The thread's `float4` is checked against the kernel's arithmetic.
    fn first_state(&self, function: &D::Function, constants: &[(u32, f32)]) -> Result<(f64, f64)> {
        let started = Instant::now();
        let destination = self.device.new_buffer(16, Storage::Shared)?;
        let creating = Instant::now();
        let state = self.device.new_compute_pipeline_state(function, &[])?;
        let made = creating.elapsed();
        let command_buffer = self.queue.command_buffer()?;
        let mut encoder = command_buffer.compute_command_encoder()?;
        encoder.set_compute_pipeline_state(&state)?;
        encoder.set_buffer(&destination, 0, 0)?;
        encoder.set_bytes(&1u32.to_le_bytes(), 1)?;
        encoder.dispatch_threadgroups([1, 1, 1], [1, 1, 1])?;
        encoder.end_encoding()?;
        command_buffer.commit()?;
        command_buffer.wait_until_completed()?;
        let whole = started.elapsed();
        // Thread 0's start (0, -0, 0.5, -0.25), one round: the FMA step plus the products'
        // difference, a * positive - a * negative, of the start.
        let factor = |index| {
            (constants.iter())
                .find(|&&(at, _)| at == index)
                .map_or(0.0, |&(_, value)| value)
        };
        let want = [0.0f32, -0.0, 0.5, -0.25]
            .map(|a| a.mul_add(0.96875, 0.03125) + (0.0 + a * factor(0) - a * factor(1)));
        let contents = destination
            .contents()
            .ok_or("the first state's destination has no contents")?;
        let mut got = [0f32; 4];
        for (at, value) in got.iter_mut().enumerate() {
            // SAFETY: the buffer holds 16 bytes and its command buffer has completed.
            let bytes = unsafe { contents.as_ptr().add(4 * at).cast::<[u8; 4]>().read() };
            *value = f32::from_le_bytes(bytes);
        }
        if got != want {
            return Err(format!("the first state's dispatch wrote {got:?}, not {want:?}").into());
        }
        Ok((made.as_secs_f64() * 1e6, whole.as_secs_f64() * 1e6))
    }
}

/// The allocations between two reads of [`Device::allocations`]: their count and time, and
/// the longest since the device was made.
fn since(before: Option<Waits>, after: Option<Waits>) -> Option<Waits> {
    let (before, after) = (before?, after?);
    Some(Waits {
        count: after.count - before.count,
        nanos: after.nanos - before.nanos,
        longest_nanos: after.longest_nanos,
    })
}

/// [`Session::create`] of buffers on `device`.
fn create<D: Device>(device: &D, length: u64, runs: Runs) -> Result<CreateReport> {
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
    // The first take of memory for a buffer of the length, then the first after its release.
    let before = device.allocations();
    let first = once(4)?;
    let first_allocations = since(before, device.allocations());
    let again = once(4)?;
    let mut report = CreateReport {
        first_create_us: Some(first),
        again_us: Some(again),
        first_allocations,
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

/// The threadgroups of 16x16 that cover a texture (whole ones: `Plan::validate`).
fn tiles(plan: &TexturePlan) -> crate::Size {
    [u64::from(plan.width / 16), u64::from(plan.height / 16), 1]
}

/// Dispatches of the copy kernel, already set on `encoder`, moving the first `length` bytes (a
/// multiple of 16) of `from` to `to` ([`copy_dispatches`]).
fn copy<D: ComputeDevice>(
    encoder: &mut D::ComputeCommandEncoder,
    from: &D::Buffer,
    to: &D::Buffer,
    length: u64,
) -> Result<()> {
    for (offset, groups, threads) in copy_dispatches(length) {
        encoder.set_buffer(from, offset, 0)?;
        encoder.set_buffer(to, offset, 1)?;
        encoder.dispatch_threadgroups([groups, 1, 1], [threads, 1, 1])?;
    }
    Ok(())
}

/// The copy kernel's dispatches for `length` bytes (a multiple of 16), 16 bytes a thread, each
/// as `(offset, threadgroups, threads per threadgroup)`: whole threadgroups of 256 threads, as
/// many dispatches as the threadgroup count limit needs, then one threadgroup of the threads left.
/// No dispatch ends in a partial threadgroup, which not every runtime dispatches.
pub(crate) fn copy_dispatches(length: u64) -> Vec<(u64, u64, u64)> {
    let threads = length / 16;
    let whole = threads / 256;
    let mut dispatches: Vec<(u64, u64, u64)> = (0..whole)
        .step_by(COPY_GROUPS as usize)
        .map(|group| (group * 256 * 16, (whole - group).min(COPY_GROUPS), 256))
        .collect();
    if threads % 256 > 0 {
        dispatches.push((whole * 256 * 16, 1, threads % 256));
    }
    dispatches
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

/// The CPU's write of `range` of `buffer` before a run: `plan`'s starting bytes there, or zero
/// where it has none, marked with `didModifyRange:` where `Managed`. The bytes are those the range
/// already holds, so a run still in flight reads the same values.
fn modify(buffer: &impl Buffer, plan: &BufferPlan, range: &Range<u64>) -> Result<()> {
    let contents = buffer
        .contents()
        .ok_or("a modified buffer has no contents")?;
    let (at, length) = (range.start as usize, (range.end - range.start) as usize);
    // SAFETY: `contents` holds the buffer's `plan.length` bytes, `range` is within them
    // (`Plan::validate`), and so it is within the starting bytes.
    unsafe {
        let to = contents.as_ptr().add(at);
        match &plan.contents {
            Some(bytes) => std::ptr::copy_nonoverlapping(bytes[at..].as_ptr(), to, length),
            None => std::ptr::write_bytes(to, 0, length),
        }
    }
    if plan.storage == Storage::Managed {
        buffer.did_modify_range(range.start, range.end - range.start)?;
    }
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
