//! Which devices are primed (`device/prime.rs`): the process's first is, and a device made again on
//! what stays loaded, after a device of its identity registered the exit hook, is not, and
//! registers nothing.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Nothing runs on the GPU but the first
//! device's priming.

use crate::exit_guard::registrations;
use crate::{Device, DeviceKind, DeviceOptions, MessageSink};

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
