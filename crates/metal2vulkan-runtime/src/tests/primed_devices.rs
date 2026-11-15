//! Which devices are primed (`device/prime.rs`): the process's first is, and a device made again on
//! what stays loaded, after a device of its identity registered the exit hook, is not, and
//! registers nothing.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Nothing runs on the GPU but the first
//! device's priming.

use crate::exit_guard::registrations;
use crate::{Device, DeviceKind, DeviceOptions, MessageSink};
use std::sync::atomic::Ordering;

/// Options of the kind the cfg selects, with a message sink of their own: a group of devices no
/// other test makes (a sink is compared by its identity).
fn options() -> DeviceOptions {
    DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        messages: Some(MessageSink::new(|_| {})),
        ..DeviceOptions::default()
    }
}

/// Four devices made and dropped one after another after a first: each made while the process
/// loaded no shared object is primed no more and registers the hook no more. One made as the
/// process loaded a shared object is primed, rightly, and not counted. Printed: `PRIMED-DEVICES`,
/// how many were made with no load, and the registrations.
#[test]
fn a_device_made_again_on_what_stays_loaded_is_not_primed() {
    let options = options();
    drop(Device::system_default_with(&options).unwrap());
    let first = registrations();
    assert!(
        first.primings >= 1,
        "the process's first device was not primed: {first:?}"
    );
    let mut unloaded = 0;
    for round in 0..4 {
        let before = registrations();
        drop(Device::system_default_with(&options).unwrap());
        let after = registrations();
        if before.loads.is_none() || after.loads != before.loads {
            continue;
        }
        unloaded += 1;
        assert_eq!(
            after.primings, before.primings,
            "round {round}: a device made again on what stays loaded was primed: {after:?}"
        );
        assert_eq!(
            after.at_devices, before.at_devices,
            "round {round}: a device made again on what stays loaded registered the hook: \
             {after:?}"
        );
    }
    eprintln!(
        "PRIMED-DEVICES: {unloaded} of 4 devices made again with no load; {:?}",
        registrations()
    );
}

/// The drivers' keeper is made with the process's first device, before that device is primed and
/// registers the exit hook: a load its instance makes (here one forced as it is made, as a host's
/// driver the device's instance let go loads again for the keeper's) comes before the load count
/// the registration keeps, so a device made again on what stays loaded is not primed. Printed:
/// `PRIMED-AFTER-KEEPER`, whether the second device was judged (no load as it was made) and the
/// registrations.
#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
#[test]
fn a_load_as_the_drivers_are_kept_primes_no_later_device() {
    crate::device::keeper::LOAD_AS_KEPT.store(true, Ordering::SeqCst);
    let options = options();
    drop(Device::system_default_with(&options).unwrap());
    assert_eq!(
        crate::loaded::forced(),
        1,
        "the process's first device made the keeper, with its forced load"
    );
    let before = registrations();
    assert!(
        before.primings >= 1,
        "the first device was not primed: {before:?}"
    );
    drop(Device::system_default_with(&options).unwrap());
    let after = registrations();
    let judged = before.loads.is_some() && after.loads == before.loads;
    if judged {
        assert_eq!(
            after.primings, before.primings,
            "the device made after the keeper's load was primed: {after:?}"
        );
        assert_eq!(
            after.at_devices, before.at_devices,
            "the device made after the keeper's load registered the hook: {after:?}"
        );
    }
    eprintln!("PRIMED-AFTER-KEEPER: judged {judged}; {after:?}");
}
