//! A device the driver refuses for want of its resources (`create.rs`'s `made_within_resources`,
//! the driver's refusal stood in for by `refuse_next_devices`): refused once, it is made with its
//! least queue plan's `VkQueue`s where that takes fewer, a round robin of one and none reserved
//! for presenting, and announces `least-queue-plan` once as it is made, its first presenting queue
//! sharing the round robin's `VkQueue` (`shared-present-queue`, saying why); where the least plan
//! takes no fewer (RADV's and lavapipe's one graphics queue), and wherever it is refused twice,
//! it is refused (`Error::DeviceRefused`), naming each attempt. Each test prints a
//! `LEAST-QUEUE-PLAN:` line with what the device took or the refusal, the evidence of the path
//! that ran.
//!
//! The plan's queues are read from the queue plan of the device's families, never from a device
//! the driver made: where other processes' devices hold what the driver lacks, a device made with
//! the plan's options takes the least plan too (an RTX 2080 beside 17 other device tests). The
//! probe that reads the families is made that way on purpose, refused once by the stand-in, and
//! dropped before the test's own device.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects.

use super::blit_holds::Scratch;
use super::plan_room::{device_with_room, with_room};
use crate::device::create::{queue_plan, refuse_next_devices};
use crate::{
    Degradation, DegradeNotice, Device, DeviceKind, DeviceOptions, Error, MessageSink,
    ShaderCacheOptions,
};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The notices a device's sink heard.
type Heard = Arc<Mutex<Vec<DegradeNotice>>>;

/// The device the cfg selects, its shader cache in `cache` and a message sink of its own, so a
/// device of its own, whose notices go to `heard`; of its queue plan, the one its least plan
/// stands in for.
fn device(cache: &Path, heard: &Heard) -> Result<Device, Error> {
    Device::system_default_with(&options(cache, heard))
}

/// The options of [`device`].
fn options(cache: &Path, heard: &Heard) -> DeviceOptions {
    let heard = Arc::clone(heard);
    DeviceOptions {
        queue_limits: None,
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![DeviceKind::Discrete, DeviceKind::Integrated],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        messages: Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                heard.lock().unwrap().push(notice.clone());
            }
        })),
        ..DeviceOptions::default()
    }
}

/// The `VkQueue`s of `device`'s round robin and those reserved for presenting (not the copy
/// queue's, which a device with a copy engine adds after them).
fn queues(device: &Device) -> (usize, usize) {
    let shared = device.shared();
    (shared.round_robin(), shared.presenting())
}

/// The device's name, the `VkQueue`s its queue plan gives its round robin and presenting
/// (`queue_plan` over its families), and those a probe took, made with its cache in `cache` as a
/// GPU's driver makes it where other processes' devices hold what it lacks: refused its plan
/// once, so of its least plan where that takes fewer, else made again, with its plan
/// (`device_with_room`). The probe is dropped before this returns.
fn planned(cache: &Path) -> (String, (usize, usize), (usize, usize)) {
    let heard = Heard::default();
    refuse_next_devices(1);
    let probe = match device(cache, &heard) {
        Ok(probe) => probe,
        Err(Error::DeviceRefused(_)) => device_with_room(&options(cache, &heard)).unwrap(),
        Err(error) => panic!("the probe: {error}"),
    };
    // SAFETY: the handles are read while the probe lives.
    let raw = unsafe { probe.raw() };
    // SAFETY: the physical device is the instance's.
    let families = unsafe {
        raw.instance
            .get_physical_device_queue_family_properties(raw.physical_device)
    };
    let plan = queue_plan(&families, 0, true).unwrap();
    let took = queues(&probe);
    (
        probe.name().to_owned(),
        (plan.queues.len(), plan.present.len()),
        took,
    )
}

/// The kinds `heard` heard, in order.
fn kinds(heard: &Heard) -> Vec<Degradation> {
    (heard.lock().unwrap().iter())
        .map(|notice| notice.degradation)
        .collect()
}

/// Refused once by the stand-in, the device takes its least plan where that takes fewer
/// `VkQueue`s; where the least plan takes no fewer it is refused. The least plan's making reaches
/// the driver, which refuses it too where other processes' devices hold its channels (the RTX
/// 2080 beside 33 queue tests, job m3-max-33373): that making is asked again, the stand-in's
/// refusal with it, while the driver lacks room (`plan_room::with_room`).
#[test]
fn a_device_refused_once_takes_its_least_plan_where_it_takes_fewer_queues() {
    let scratch = Scratch::new("least-queue-plan-once");
    let (_, (round_robin, presenting), (probe_round_robin, probe_presenting)) =
        planned(&scratch.0.join("probe"));
    let heard = Heard::default();
    let fewer = round_robin + presenting > 1;
    let made = with_room(|| {
        refuse_next_devices(1);
        let made = device(&scratch.0.join("refused"), &heard);
        // The driver's own refusal of the least plan, after the stand-in's of the plan.
        let short = match &made {
            Err(Error::DeviceRefused(refused)) if fewer && refused.attempts.len() == 2 => {
                Some(refused.to_string())
            }
            _ => None,
        };
        (made, short)
    });
    // The stand-in's refusal is spent, or never reached the driver.
    refuse_next_devices(0);
    match made {
        Ok(device) => {
            println!(
                "LEAST-QUEUE-PLAN: {} took 1 + 0 of the plan's {round_robin} + {presenting} \
                 (the probe refused once took {probe_round_robin} + {probe_presenting})",
                device.name()
            );
            assert!(round_robin + presenting > 1, "the least plan took no fewer");
            assert_eq!(queues(&device), (1, 0));
            let least: Vec<DegradeNotice> = (heard.lock().unwrap().iter())
                .filter(|notice| notice.degradation == Degradation::LeastQueuePlan)
                .cloned()
                .collect();
            assert_eq!(least.len(), 1, "{least:?}");
            assert!(
                least[0].reason.contains("ERROR_INITIALIZATION_FAILED"),
                "{least:?}"
            );
            // A presenting queue shares the round robin's `VkQueue`, saying why.
            let _present = device.new_command_queue_for_presenting().unwrap();
            let shared: Vec<DegradeNotice> = (heard.lock().unwrap().iter())
                .filter(|notice| notice.degradation == Degradation::SharedPresentQueue)
                .cloned()
                .collect();
            assert_eq!(shared.len(), 1, "{shared:?}");
            assert!(shared[0].reason.contains("least-queue-plan"), "{shared:?}");
        }
        Err(Error::DeviceRefused(refused)) => {
            println!("LEAST-QUEUE-PLAN: refused, its least plan the plan: {refused}");
            assert_eq!(round_robin + presenting, 1, "{refused}");
            assert_eq!(refused.attempts.len(), 1, "{refused}");
            assert!(!kinds(&heard).contains(&Degradation::LeastQueuePlan));
        }
        Err(error) => panic!("the device: {error}"),
    }
}

#[test]
fn a_device_refused_twice_is_refused_naming_each_attempt() {
    let scratch = Scratch::new("least-queue-plan-twice");
    let (name, (round_robin, presenting), _) = planned(&scratch.0.join("probe"));
    let heard = Heard::default();
    refuse_next_devices(2);
    let made = device(&scratch.0.join("refused"), &heard);
    let Err(Error::DeviceRefused(refused)) = made else {
        panic!(
            "a device refused twice: {:?}",
            made.map(|device| queues(&device))
        );
    };
    println!("LEAST-QUEUE-PLAN: {refused}");
    let attempts = match round_robin + presenting > 1 {
        true => 2,
        false => 1,
    };
    assert_eq!(refused.attempts.len(), attempts, "{refused}");
    assert_eq!(refused.device, name);
    assert!(heard.lock().unwrap().is_empty(), "{:?}", kinds(&heard));
    // The refusal stood in for the driver's, which the next device meets no more.
    refuse_next_devices(0);
}
