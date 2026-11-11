//! The messages the runtime says once per device that are no degradation
//! (`degradation::Told`): each kind's first occurrence on a device is said, and each later one is
//! counted in the kind's `WorkCounts` field, its message not even made; another device says its
//! own first. A late reservation of resource IDs, met through its own call, does so too.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Nothing runs on the GPU.

use crate::degradation::Told;
use crate::device::lock;
use crate::texture::decoded_twin::twin_decode_failed;
use crate::{
    DebugMessage, Device, DeviceKind, DeviceOptions, MessageKind, MessageSeverity, MessageSink,
    ResourceId, ResourceIdKind, WorkCounts,
};
use metal2vulkan::metal::PixelFormat;
use std::sync::{Arc, Mutex};

/// The id names of the kinds' messages, in [`KINDS`]' order.
const KINDS: [(Told, &str); 4] = [
    (
        Told::HeapTextureRefused,
        "metal2vulkan-heap-texture-refused",
    ),
    (Told::TwinDecodeFailed, "metal2vulkan-twin-decode-failed"),
    (Told::LateReservation, "metal2vulkan-late-reservation"),
    (
        Told::QueueLimitsReached,
        "metal2vulkan-queue-limits-reached",
    ),
];

/// What the device's sink heard of the kinds' messages: each one's id name.
type Heard = Arc<Mutex<Vec<String>>>;

/// The device the cfg selects, with a sink that keeps the id name of every message of a kind of
/// [`KINDS`].
fn listening() -> (Device, Heard) {
    let heard: Heard = Arc::default();
    let kept = Arc::clone(&heard);
    let sink = MessageSink::new(move |message: &DebugMessage| {
        if KINDS.iter().any(|(_, name)| *name == message.id_name) {
            lock(&kept).push(message.id_name.clone());
        }
    });
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        messages: Some(sink),
        ..DeviceOptions::default()
    };
    (Device::system_default_with(&options).unwrap(), heard)
}

/// The `WorkCounts` field that counts `told`'s withheld messages.
fn withheld(counts: &WorkCounts, told: Told) -> u64 {
    match told {
        Told::HeapTextureRefused => counts.heap_texture_refused_withheld,
        Told::TwinDecodeFailed => counts.twin_decode_failed_withheld,
        Told::LateReservation => counts.late_reservation_withheld,
        Told::QueueLimitsReached => counts.queue_limits_reached_withheld,
    }
}

/// The message of kind `told` named `name`: a decoded twin's failed decode as that site makes it,
/// another kind's as an informational line.
fn message(told: Told, name: &str) -> DebugMessage {
    if told == Told::TwinDecodeFailed {
        let failed = crate::error::internal("a decode that failed");
        return twin_decode_failed(PixelFormat::ETC2_RGB8, [8, 8, 1], &failed);
    }
    DebugMessage {
        severity: MessageSeverity::Info,
        kind: MessageKind {
            general: true,
            ..MessageKind::default()
        },
        id_name: name.to_owned(),
        id: 0,
        text: format!("metal2vulkan: an occurrence of {name}"),
        degradation: None,
    }
}

/// Each kind, met twice on a device: the first is said, the second made no message and counted
/// one in the kind's field alone; on another device, the kind's first is said again.
#[test]
fn each_kind_is_said_once_per_device_and_counted_after() {
    let (device, heard) = listening();
    let (other, other_heard) = listening();
    for (at, &(told, name)) in KINDS.iter().enumerate() {
        let made = std::cell::Cell::new(0);
        let tell = |device: &Device| {
            device.tell(told, || {
                made.set(made.get() + 1);
                message(told, name)
            })
        };
        tell(&device);
        tell(&device);
        let names: Vec<&str> = KINDS[..=at].iter().map(|(_, name)| *name).collect();
        assert_eq!(*lock(&heard), names, "{told:?}");
        assert_eq!(made.get(), 1, "{told:?}: the later message was made");
        let counts = device.work_counts();
        for &(kind, _) in &KINDS {
            let want = u64::from(KINDS[..=at].iter().any(|(met, _)| *met == kind));
            assert_eq!(withheld(&counts, kind), want, "{told:?}: {kind:?}");
        }
        tell(&other);
        assert_eq!(*lock(&other_heard), names, "{told:?} on another device");
        assert_eq!(withheld(&other.work_counts(), told), 0, "{told:?}");
    }
}

/// A reservation of texture IDs after the device's first is ignored, as Metal ignores it: the
/// first such late call is said (`metal2vulkan-late-reservation`), the second counted
/// (`WorkCounts::late_reservation_withheld`) and not said; IDs go on as before.
#[test]
fn a_second_late_reservation_is_counted_not_said() {
    let (device, heard) = listening();
    let reserve = |id| {
        device
            .reserve_resource_ids(ResourceIdKind::Texture, &[ResourceId(id)])
            .unwrap()
    };
    let late = || device.work_counts().late_reservation_withheld;
    reserve(2);
    assert!(lock(&heard).is_empty(), "{:?}", lock(&heard));
    reserve(4);
    assert_eq!(*lock(&heard), ["metal2vulkan-late-reservation"]);
    assert_eq!(late(), 0);
    reserve(6);
    assert_eq!(*lock(&heard), ["metal2vulkan-late-reservation"]);
    assert_eq!(late(), 1);
}
