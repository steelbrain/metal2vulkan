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
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects.

use super::blit_holds::Scratch;
use crate::device::create::refuse_next_devices;
use crate::{
    Degradation, DegradeNotice, Device, DeviceKind, DeviceOptions, Error, MessageSink,
    ShaderCacheOptions,
};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The notices a device's sink heard.
type Heard = Arc<Mutex<Vec<DegradeNotice>>>;

/// The device the cfg selects, its shader cache in `cache` and a message sink of its own, so a
/// device of its own, whose notices go to `heard`.
fn device(cache: &Path, heard: &Heard) -> Result<Device, Error> {
    let heard = Arc::clone(heard);
    let options = DeviceOptions {
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
    };
    Device::system_default_with(&options)
}

/// The `VkQueue`s of `device`'s round robin and those reserved for presenting.
fn queues(device: &Device) -> (usize, usize) {
    let shared = device.shared();
    let round_robin = shared.round_robin();
    (round_robin, shared.queues.len() - round_robin)
}

/// The kinds `heard` heard, in order.
fn kinds(heard: &Heard) -> Vec<Degradation> {
    (heard.lock().unwrap().iter())
        .map(|notice| notice.degradation)
        .collect()
}

#[test]
fn a_device_refused_once_takes_its_least_plan_where_it_takes_fewer_queues() {
    let scratch = Scratch::new("least-queue-plan-once");
    // The device as its plan makes it, beside which the refused one is made.
    let planned = device(&scratch.0.join("planned"), &Heard::default()).unwrap();
    let (round_robin, presenting) = queues(&planned);
    let heard = Heard::default();
    refuse_next_devices(1);
    let made = device(&scratch.0.join("refused"), &heard);
    match made {
        Ok(device) => {
            println!(
                "LEAST-QUEUE-PLAN: {} took 1 + 0 of the plan's {round_robin} + {presenting}",
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
    let planned = device(&scratch.0.join("planned"), &Heard::default()).unwrap();
    let (round_robin, presenting) = queues(&planned);
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
    assert_eq!(refused.device, planned.name());
    assert!(heard.lock().unwrap().is_empty(), "{:?}", kinds(&heard));
    // The refusal stood in for the driver's, which the next device meets no more.
    refuse_next_devices(0);
}
