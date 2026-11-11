//! A device's queue limits (`DeviceOptions::queue_limits`) on a device: it meets its
//! informational message's case (`metal2vulkan-queue-limits-reached`) exactly where the limits
//! change a placement from what the queue plan gives, each case as many times as the plan of the
//! device's own families says (`queue_limits.rs`): the reserved and the extra queues as it is
//! made, a command queue, an unshared one, a presenting one, and a read of its queue sharing. It
//! sends the first it meets alone and counts each later one
//! (`WorkCounts::queue_limits_reached_withheld`); without limits it sends none and counts none,
//! whatever it makes.
//!
//! On a device whose families have no queue to spare (RADV's and lavapipe's one graphics queue)
//! the limits change nothing and none is expected; on one with queues to spare (NVIDIA's graphics
//! family of 16) every case sends. Under the limits the device keeps the plan's staging queue
//! (`QueueLimits`): an RX 7900 XTX's compute family's first, a copy engine elsewhere. Each test
//! prints a `QUEUE-LIMITS:` line with the plan, the limits' queues, the staging queue and the
//! messages counted by case, the evidence of the path that ran.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects.

use super::blit_holds::{Scratch, WAIT};
use crate::device::create::{
    extra_queue_places, planned_queues, queue_plan, queue_plan_within, shares_past_limit,
    QUEUE_LIMITS_REACHED,
};
use crate::memory::MemoryClass;
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
    /// How many messages of every case.
    fn total(&self) -> usize {
        self.reserved + self.extra + self.command + self.unshared + self.presenting + self.read
    }

    /// The first case met, in the order the runs meet them, once: what the device sends.
    fn first(&self) -> Cases {
        let mut met = false;
        let mut once = |count: usize| {
            let first = usize::from(count > 0 && !met);
            met |= count > 0;
            first
        };
        Cases {
            reserved: once(self.reserved),
            extra: once(self.extra),
            command: once(self.command),
            unshared: once(self.unshared),
            presenting: once(self.presenting),
            read: once(self.read),
        }
    }

    /// `texts` counted by case, by the words each case's message says.
    fn of(texts: &[String]) -> Cases {
        let count = |words: &str| texts.iter().filter(|text| text.contains(words)).count();
        Cases {
            reserved: count("the queues reserved for the caller are"),
            extra: count("the extra queues are at"),
            command: count(", a command queue, the program's command queue"),
            unshared: count(", an unshared command queue, the program's command queue"),
            presenting: count("shares a VkQueue, the device's queue limits reserving"),
            read: count("its queue sharing was read"),
        }
    }
}

/// Whether a texture's image on `device` may lie in memory the host caches, as its queue plan
/// takes it (`queue_plan`).
fn cached(device: &Device) -> bool {
    crate::device::lock(&device.shared().memory).has_host_cached(MemoryClass::Private)
}

/// What a device of `families` (`cached` as [`cached`] gives) under `limits` makes and does in
/// [`run`] sends: none without limits, else each case where the plan of those families says the
/// limits changed it.
fn expected(
    families: &[vk::QueueFamilyProperties],
    cached: bool,
    limits: Option<QueueLimits>,
) -> Cases {
    let Some(limits) = limits else {
        return Cases::default();
    };
    let mut within = queue_plan_within(families, 1, Some(limits), cached).unwrap();
    let places = extra_queue_places(families, &mut within, &EXTRA);
    let Some(planned) = planned_queues(families, 1, &EXTRA, &within, &places, cached) else {
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

/// Under the lane's limits, the device meets the message's cases exactly where its own families'
/// plan says the limits changed a placement, each as many times as expected, none for the staging
/// queue, which it keeps where the plan reserves one: it sends the first case it meets, and counts
/// every later one. A second read of its queue sharing, where the limits changed it, sends nothing
/// and counts one more.
#[test]
fn each_placement_the_limits_change_is_noted_and_no_other() {
    let scratch = Scratch::new("queue-limits");
    let noted = Arc::new(Mutex::new(Vec::new()));
    let device = device(&scratch.0, Some(LEAN), &noted);
    let families = families(&device);
    let cached = cached(&device);
    let expected = expected(&families, cached, Some(LEAN));
    let mut plan = queue_plan(&families, 1, cached).unwrap();
    extra_queue_places(&families, &mut plan, &EXTRA);
    let _queues = run(&device);
    let texts = noted.lock().unwrap().clone();
    let got = Cases::of(&texts);
    let withheld = device.work_counts().queue_limits_reached_withheld;
    eprintln!(
        "QUEUE-LIMITS: {}: families {:?}; host-cached texture memory {cached}, staging queue {:?} \
         (the plan's {:?}); expected {expected:?}, got {got:?}; {} message(s), {withheld} \
         withheld: {texts:#?}",
        device.name(),
        (families.iter())
            .map(|f| (f.queue_flags.as_raw(), f.queue_count))
            .collect::<Vec<_>>(),
        device.has_staging_queue(),
        plan.staging,
        texts.len()
    );
    assert_eq!(
        device.has_staging_queue(),
        plan.staging.is_some(),
        "the limits keep the plan's staging queue"
    );
    assert_eq!(got, expected.first());
    assert_eq!(
        texts.len(),
        expected.total().min(1),
        "every message is one of the cases"
    );
    assert_eq!(withheld as usize, expected.total().saturating_sub(1));
    // A second read of the queue sharing: the case met again, counted, not sent.
    device.queue_switch_check_report();
    assert_eq!(noted.lock().unwrap().len(), texts.len());
    let again = device.work_counts().queue_limits_reached_withheld;
    assert_eq!(again, withheld + u64::from(expected.read > 0));
}

/// Without limits the device sends none, whatever it makes and reads.
#[test]
fn without_limits_nothing_is_noted() {
    let scratch = Scratch::new("queue-limits-none");
    let noted = Arc::new(Mutex::new(Vec::new()));
    let device = device(&scratch.0, None, &noted);
    let _queues = run(&device);
    let texts = noted.lock().unwrap().clone();
    let withheld = device.work_counts().queue_limits_reached_withheld;
    eprintln!(
        "QUEUE-LIMITS: {}: no limits, {} message(s), {withheld} withheld",
        device.name(),
        texts.len()
    );
    assert!(texts.is_empty(), "{texts:#?}");
    assert_eq!(withheld, 0);
}
