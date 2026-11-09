//! The scheduler's thread takes a driver's stall in one signalling submission of every 1024 to a
//! queue while it is idle (`device/submit_stalls.rs`) on a device whose driver has the cost fact
//! (`DriverFact::SignallingSubmitsStallEvery1024`), and makes no empty submission on any other.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::{CommandQueue, Device, DeviceKind, DeviceOptions, ResourceOptions, ShaderCacheOptions};
use metal2vulkan::device::DriverFact;
use std::path::PathBuf;
use std::time::Duration;

/// The frames, and each frame's command buffers, at the 25 command buffers a frame `host_ab
/// present` commits: about five of the driver's periods of 1024 signalling submissions where each
/// batch takes a call of its own, about two on the RTX 5080, where the scheduler merges the
/// batches it finds ready together (2190 to 2500 signalling calls a run there).
const FRAMES: u32 = 200;
const PER_FRAME: u32 = 25;

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

/// [`FRAMES`] frames of [`PER_FRAME`] command buffers of `queue`, each a fill of 16 bytes,
/// committed back to back, the last waited for, then [`GAP`] with nothing committed.
fn frames(device: &Device, queue: &CommandQueue) {
    let target = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    for frame in 0..FRAMES {
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
    }
}

/// Over about five periods of frames with idle gaps between them: on a driver with the stall,
/// the scheduler took at least one stall in a gap, by empty submissions that took at most the
/// burst and the margin a stall (`WorkCounts::submit_stalls_idle`, `WorkCounts::idle_submits`);
/// on every other driver it made no empty submission. The counts are printed either way.
#[test]
fn the_scheduler_takes_the_driver_s_stalls_while_idle_only_where_the_driver_has_them() {
    let scratch = Scratch::new();
    let device = device(&scratch);
    let queue = device.new_command_queue().unwrap();
    let stalls =
        (device.properties().driver_facts).contains(&DriverFact::SignallingSubmitsStallEvery1024);
    let before = device.work_counts();
    frames(&device, &queue);
    let after = device.work_counts();
    let idle_submits = after.idle_submits - before.idle_submits;
    let idle = after.submit_stalls_idle - before.submit_stalls_idle;
    let on_work = after.submit_stalls_on_work - before.submit_stalls_on_work;
    let behind = after.submits_behind_idle_stalls - before.submits_behind_idle_stalls;
    println!(
        "{}: driver stalls every 1024th signalling submit: {stalls}; over {} command buffers: \
         idle_submits {idle_submits} submit_stalls_idle {idle} submit_stalls_on_work {on_work} \
         submits_behind_idle_stalls {behind} wait_behind_idle_stall_longest_nanos {}",
        device.name(),
        FRAMES * PER_FRAME,
        after.wait_behind_idle_stall_longest_nanos,
    );
    match stalls {
        true => {
            assert!(
                idle >= 1,
                "no stall was taken idle over {} command buffers",
                FRAMES * PER_FRAME
            );
            let most = idle * u64::from(PER_FRAME + crate::device::submit_stalls::MARGIN);
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
