//! A device's queue limits (`DeviceOptions::queue_limits`) on a device: it sends its
//! informational message (`metal2vulkan-queue-limits-reached`) exactly where the limits change a
//! placement from what the queue plan gives, each case as many times as the plan of the device's
//! own families says (`queue_limits.rs`): the reserved and the extra queues as it is made, a
//! command queue, an unshared one, a presenting one, and a read of its queue sharing; and without
//! limits it sends none, whatever it makes.
//!
//! On a device whose families have no queue to spare (RADV's and lavapipe's one graphics queue)
//! the limits change nothing and none is expected; on one with queues to spare (NVIDIA's graphics
//! family of 16) every case sends. Each test prints a `QUEUE-LIMITS:` line with the plan, the
//! limits' queues and the messages counted by case, the evidence of the path that ran.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects.

use super::blit_holds::{Scratch, WAIT};
use crate::device::create::{
    extra_queue_places, planned_queues, queue_plan_within, shares_past_limit, QUEUE_LIMITS_REACHED,
};
use crate::{
    CommandQueue, Device, DeviceKind, DeviceOptions, MessageSink, QueueLimits, ShaderCacheOptions,
};
use ash::vk;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The limits the runtime lane makes its devices with.
const LEAN: QueueLimits = QueueLimits {
    round_robin: 1,
    presenting: 0,
};

/// The command queues, and the presenting ones, each run makes.
const QUEUES: u64 = 3;

/// What each run asks for beside its command queues: one reserved queue and one extra graphics
/// queue, which the limits move on a device with queues to spare.
const EXTRA: [vk::QueueFlags; 1] = [vk::QueueFlags::GRAPHICS];

/// The device the cfg selects under `limits`, its shader cache in `cache`, one reserved and one
/// extra graphics queue asked for, the texts of its `metal2vulkan-queue-limits-reached`
/// messages kept in `noted`.
fn device(cache: &Path, limits: Option<QueueLimits>, noted: &Arc<Mutex<Vec<String>>>) -> Device {
    let kept = Arc::clone(noted);
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
        reserved_queues: 1,
        extra_queues: EXTRA.to_vec(),
        queue_limits: limits,
        messages: Some(MessageSink::new(move |message| {
            if message.id_name == QUEUE_LIMITS_REACHED {
                kept.lock().unwrap().push(message.text.clone());
            }
        })),
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// The device's queue family properties.
fn families(device: &Device) -> Vec<vk::QueueFamilyProperties> {
    // SAFETY: the handles are read while the device lives.
    let raw = unsafe { device.raw() };
    // SAFETY: the physical device is the instance's.
    unsafe {
        raw.instance
            .get_physical_device_queue_family_properties(raw.physical_device)
    }
}

/// The messages of each case, in the order the runs make them: the reserved queues, the extra
/// queues, command queues, an unshared one, presenting ones, reads of the queue sharing.
#[derive(Debug, Default, PartialEq, Eq)]
struct Cases {
    reserved: usize,
    extra: usize,
    command: usize,
    unshared: usize,
    presenting: usize,
    read: usize,
}

impl Cases {
    /// `texts` counted by case, by the words each case's message says.
    fn of(texts: &[String]) -> Cases {
        let count = |words: &str| texts.iter().filter(|text| text.contains(words)).count();
        Cases {
            reserved: count("the queues reserved for the caller are"),
            extra: count("the extra queues are at"),
            command: count(", a command queue took VkQueue"),
            unshared: count(", an unshared command queue took VkQueue"),
            presenting: count("shares a VkQueue, the device's queue limits reserving"),
            read: count("its queue sharing was read"),
        }
    }
}

/// What a device of `families` under `limits` makes and does in [`run`] sends: none without
/// limits, else each case where the plan of those families says the limits changed it.
fn expected(families: &[vk::QueueFamilyProperties], limits: Option<QueueLimits>) -> Cases {
    let Some(limits) = limits else {
        return Cases::default();
    };
    let mut within = queue_plan_within(families, 1, Some(limits)).unwrap();
    let places = extra_queue_places(families, &mut within, &EXTRA);
    let Some(planned) = planned_queues(families, 1, &EXTRA, &within, &places) else {
        return Cases::default();
    };
    let (round_robin, present) = (within.queues.len(), within.present.len());
    let commands = |range: std::ops::Range<u64>| {
        range
            .filter(|&made| shares_past_limit(made, round_robin, planned.round_robin))
            .count()
    };
    let sharing_changed = (round_robin, present) != (planned.round_robin, planned.presenting);
    Cases {
        reserved: usize::from(planned.reserved != within.reserved),
        extra: usize::from(planned.extra != places),
        command: commands(0..QUEUES),
        // The unshared queue is made after the command queues, on a device that does not share
        // (one round-robin `VkQueue` under the lane's limits): the round robin's next.
        unshared: commands(QUEUES..QUEUES + 1),
        presenting: (0..QUEUES)
            .filter(|&asked| shares_past_limit(asked, present, planned.presenting))
            .count(),
        read: 2 * usize::from(sharing_changed),
    }
}

/// [`QUEUES`] command queues, an unshared one, [`QUEUES`] presenting ones, and the two reads of
/// the queue sharing, on `device`; what they are, kept while the counts are read.
fn run(device: &Device) -> Vec<CommandQueue> {
    let mut queues: Vec<CommandQueue> = (0..QUEUES)
        .map(|_| device.new_command_queue().unwrap())
        .collect();
    queues.push(device.new_command_queue_unshared(64).unwrap());
    queues.extend((0..QUEUES).map(|_| device.new_command_queue_for_presenting().unwrap()));
    device.settle_queue_switch_check(WAIT);
    device.queue_switch_check_report();
    queues
}

/// Under the lane's limits, the device sends the message exactly where its own families' plan
/// says the limits changed a placement: each case as many times as expected.
#[test]
fn each_placement_the_limits_change_is_noted_and_no_other() {
    let scratch = Scratch::new("queue-limits");
    let noted = Arc::new(Mutex::new(Vec::new()));
    let device = device(&scratch.0, Some(LEAN), &noted);
    let families = families(&device);
    let expected = expected(&families, Some(LEAN));
    let _queues = run(&device);
    let texts = noted.lock().unwrap().clone();
    let got = Cases::of(&texts);
    eprintln!(
        "QUEUE-LIMITS: {}: families {:?}; expected {expected:?}, got {got:?}; {} message(s): \
         {texts:#?}",
        device.name(),
        (families.iter())
            .map(|f| (f.queue_flags.as_raw(), f.queue_count))
            .collect::<Vec<_>>(),
        texts.len()
    );
    assert_eq!(got, expected);
    assert_eq!(
        texts.len(),
        expected.reserved
            + expected.extra
            + expected.command
            + expected.unshared
            + expected.presenting
            + expected.read,
        "every message is one of the cases"
    );
}

/// Without limits the device sends none, whatever it makes and reads.
#[test]
fn without_limits_nothing_is_noted() {
    let scratch = Scratch::new("queue-limits-none");
    let noted = Arc::new(Mutex::new(Vec::new()));
    let device = device(&scratch.0, None, &noted);
    let _queues = run(&device);
    let texts = noted.lock().unwrap().clone();
    eprintln!(
        "QUEUE-LIMITS: {}: no limits, {} message(s)",
        device.name(),
        texts.len()
    );
    assert!(texts.is_empty(), "{texts:#?}");
}
