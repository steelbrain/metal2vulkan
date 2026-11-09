//! The scheduler's thread takes a driver's stall in one signalling submission of every 1024 to a
//! queue while it is idle (`device/submit_stalls.rs`) on a device whose driver has the cost fact
//! (`DriverFact::SignallingSubmitsStallEvery1024`), and makes no empty submission on any other.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::device::submit_stalls::{MARGIN, PERIOD};
use crate::{
    CommandQueue, Device, DeviceKind, DeviceOptions, ResourceOptions, ShaderCacheOptions,
    WorkCounts,
};
use metal2vulkan::device::DriverFact;
use std::path::PathBuf;
use std::time::Duration;

/// The frames on a driver without the stall, and each frame's command buffers, at the 25 command
/// buffers a frame `host_ab present` commits: about five of the driver's periods of 1024
/// signalling submissions where each batch takes a call of its own.
const FRAMES: u32 = 200;
const PER_FRAME: u32 = 25;

/// The device's calls to its queues after which a driver with the stall must have had one taken
/// idle: [`PERIOD`] times this. Frames go on until a stall was taken idle or the calls passed it,
/// for the calls a frame makes vary: where the scheduler merges the batches it finds ready
/// together (the RTX 5080, 615.71.09), 200 frames made 1686 to 1882 signalling calls in three
/// runs, and a stall is first due idle only a period after the second the runtime saw, which
/// follows uses of the queue it does not count (in those runs the 32nd and the 784th signalling
/// call stalled, so the first due idle was the 1808th, past the end of one run;
/// `device/submit_stalls.rs`).
const PERIODS: u64 = 6;

/// The most frames a driver with the stall runs: they reach [`PERIODS`] periods of calls at 3.1
/// calls a frame or more (the RTX 5080 made about 9).
const MOST_FRAMES: u32 = 2000;

/// The gap after each frame: longer than the idle the scheduler's thread waits for before it
/// takes a stall due (`IDLE_NANOS`), with room for the stall itself.
const GAP: Duration = Duration::from_millis(8);

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path = std::env::temp_dir().join(format!("m2v-submit-stalls-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache in
/// `scratch`.
fn device(scratch: &Scratch) -> Device {
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
            directory: Some(scratch.0.clone()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// Frames of [`PER_FRAME`] command buffers of `queue`, each a fill of 16 bytes, committed back to
/// back, the last waited for, then [`GAP`] with nothing committed: at most `most`, and none once
/// `done` answers true of the device's counts after a frame.
fn frames(device: &Device, queue: &CommandQueue, most: u32, done: impl Fn(&WorkCounts) -> bool) {
    let target = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    for frame in 0..most {
        let mut last = None;
        for _ in 0..PER_FRAME {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&target, 0..16, frame as u8).unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            last = Some(command_buffer);
        }
        let last = last.unwrap();
        last.wait_until_completed().unwrap();
        assert_eq!(last.error(), None);
        std::thread::sleep(GAP);
        if done(&device.work_counts()) {
            return;
        }
    }
}

/// Frames with idle gaps between them: on a driver with the stall, the scheduler took a stall in
/// a gap within [`PERIODS`] periods of the device's calls to its queues, by empty submissions
/// that took at most the burst and the margin a stall (`WorkCounts::submit_stalls_idle`,
/// `WorkCounts::idle_submits`); on every other driver, over [`FRAMES`] frames, it made no empty
/// submission. The counts are printed either way.
#[test]
fn the_scheduler_takes_the_driver_s_stalls_while_idle_only_where_the_driver_has_them() {
    let scratch = Scratch::new();
    let device = device(&scratch);
    let queue = device.new_command_queue().unwrap();
    let stalls =
        (device.properties().driver_facts).contains(&DriverFact::SignallingSubmitsStallEvery1024);
    let before = device.work_counts();
    let calls = |counts: &WorkCounts| counts.queue_submissions - before.queue_submissions;
    let bound = PERIODS * u64::from(PERIOD);
    match stalls {
        true => frames(&device, &queue, MOST_FRAMES, |counts| {
            counts.submit_stalls_idle > before.submit_stalls_idle || calls(counts) >= bound
        }),
        false => frames(&device, &queue, FRAMES, |_| false),
    }
    let after = device.work_counts();
    let idle_submits = after.idle_submits - before.idle_submits;
    let idle = after.submit_stalls_idle - before.submit_stalls_idle;
    let on_work = after.submit_stalls_on_work - before.submit_stalls_on_work;
    let behind = after.submits_behind_idle_stalls - before.submits_behind_idle_stalls;
    println!(
        "{}: driver stalls every 1024th signalling submit: {stalls}; over {} calls to its queues: \
         idle_submits {idle_submits} submit_stalls_idle {idle} submit_stalls_on_work {on_work} \
         submits_behind_idle_stalls {behind} wait_behind_idle_stall_longest_nanos {}",
        device.name(),
        calls(&after),
        after.wait_behind_idle_stall_longest_nanos,
    );
    match stalls {
        true => {
            assert!(
                idle >= 1,
                "no stall was taken idle over {} calls to the device's queues",
                calls(&after)
            );
            let most = idle * u64::from(PER_FRAME + MARGIN);
            assert!(idle_submits <= most, "{idle_submits} empty submissions");
        }
        false => {
            assert_eq!(
                idle_submits, 0,
                "an empty submission without the driver's stall"
            );
            assert_eq!(idle, 0);
        }
    }
}
