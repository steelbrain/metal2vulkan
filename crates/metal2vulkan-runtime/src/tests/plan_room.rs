//! A device of the queue plan in the runtime's own device tests, made again while the driver
//! lacks room for it (test tooling).
//!
//! A test whose work needs the plan's queues makes its device with `queue_limits: None`. The
//! driver shares what a device needs among every process's devices: NVIDIA's gives each
//! `VkQueue` a channel, and an RTX 2080 (NVIDIA 620.70.06) holds about 210 across processes, 11
//! devices of the plan's 18 (`create.rs`'s `made_within_resources`). Beside the suite's other
//! test processes the driver refuses the plan there, and the runtime makes the device with its
//! least plan (10 of the 12 `queue_share_device` tests ran on it in one run of 31 queue tests at
//! 16 threads), whose command queues share one `VkQueue`, or refuses that too
//! (`Error::DeviceRefused`). A test of the plan's queues would then run without them and could
//! pass on what it does not test.
//!
//! So [`device_with_room`] makes such a device again, the one made dropped first, after a pause
//! that doubles from [`FIRST_PAUSE`] up to [`LONGEST_PAUSE`], until [`ROOM_WAIT`] in all has
//! passed, each pause a `PLAN-DEVICE-ROOM` line on standard error naming the answer and the time
//! waited: the evidence that the wait ran. Past the bound a refusal is handed back, and a device
//! still of its least plan fails the test. A device dropped so has delivered its notices to the
//! options' sink, as a device the test kept would have. A test of the least plan itself makes its
//! device without this, the driver's refusal stood in for (`refuse_next_devices`).

use super::blit_holds::Scratch;
use crate::device::create::{queue_plan, refuse_next_devices};
use crate::{Device, DeviceKind, DeviceOptions, Error, ShaderCacheOptions};
use std::time::Duration;

/// The first pause before a device is asked for again.
const FIRST_PAUSE: Duration = Duration::from_millis(250);

/// The longest pause between two asks.
const LONGEST_PAUSE: Duration = Duration::from_secs(2);

/// The most a device waits for room, in all: the suite's other processes' devices of a refused
/// one's time have gone well within it.
const ROOM_WAIT: Duration = Duration::from_secs(120);

/// The device `options` make, of the `VkQueue`s they ask for (the queue plan's where
/// `queue_limits` is `None`): made again while the driver refuses it for want of what other
/// devices hold (`Error::DeviceRefused`) or it took its least plan, until [`ROOM_WAIT`]. Any other
/// error is returned at once, and a refusal past the bound; a device of its least plan past the
/// bound fails the test here.
pub(super) fn device_with_room(options: &DeviceOptions) -> Result<Device, Error> {
    let mut waited = Duration::ZERO;
    let mut next = FIRST_PAUSE;
    loop {
        let made = Device::system_default_with(options);
        let answer = match &made {
            Ok(device) if device.shared().least_plan => {
                format!("the device on {} took its least queue plan", device.name())
            }
            Err(Error::DeviceRefused(refused)) => refused.to_string(),
            _ => return made,
        };
        if waited >= ROOM_WAIT {
            eprintln!("PLAN-DEVICE-ROOM: {answer}; no room after {waited:?}");
            return match made {
                Err(error) => Err(error),
                Ok(_) => panic!(
                    "{answer} after {waited:?} of pauses: a test of the plan's queues runs on \
                     them or not at all"
                ),
            };
        }
        drop(made);
        eprintln!("PLAN-DEVICE-ROOM: {answer}; asked again in {next:?} ({waited:?} waited)");
        std::thread::sleep(next);
        waited += next;
        next = (next * 2).min(LONGEST_PAUSE);
    }
}

/// A device of the plan the driver refused once (the stand-in, `refuse_next_devices`) is made
/// again of the plan's queues: on a GPU of queues to spare its first making took the least plan,
/// and on one whose least plan takes no fewer it was refused. Made plainly, it would be the
/// first answer.
#[test]
fn a_plan_device_refused_once_is_made_again_of_the_plan_s_queues() {
    let scratch = Scratch::new("plan-room");
    let options = DeviceOptions {
        queue_limits: None,
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![DeviceKind::Discrete, DeviceKind::Integrated],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(scratch.0.clone()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    refuse_next_devices(1);
    let device = device_with_room(&options).unwrap();
    // SAFETY: the handles are read while the device lives.
    let raw = unsafe { device.raw() };
    // SAFETY: the physical device is the instance's.
    let families = unsafe {
        raw.instance
            .get_physical_device_queue_family_properties(raw.physical_device)
    };
    let plan = queue_plan(&families, 0, true).unwrap();
    let shared = device.shared();
    let took = (shared.round_robin(), shared.presenting());
    println!(
        "PLAN-ROOM: {} took {} + {} of the plan's {} + {}",
        device.name(),
        took.0,
        took.1,
        plan.queues.len(),
        plan.present.len()
    );
    assert!(!shared.least_plan);
    assert_eq!(took, (plan.queues.len(), plan.present.len()));
}
