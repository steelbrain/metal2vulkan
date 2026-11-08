//! Devices by registry ID (`Device::registry_ids_with`, `Device::with_registry_id`): the IDs are
//! listed with no device made, the device of an ID is the one the other creators make of its
//! physical device (the same device while it lives), and an ID no physical device has is
//! `NoDevice`. The identities of the physical devices (`Device::identities_with`), and of the one
//! the default device is made of (`Device::system_default_identity_with`), are read with no device
//! made too, and each is the one its device answers.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Nothing runs on the GPU.

use crate::capabilities::working_set_buffer_bound;
use crate::device::create::live_devices_with;
use crate::{Device, DeviceKind, DeviceOptions, Error, MessageSink};

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

#[test]
fn a_device_opens_by_its_registry_id_and_the_ids_make_none() {
    let options = options();
    let ids = Device::registry_ids_with(&options).unwrap();
    assert!(!ids.is_empty(), "no registry ID listed");
    assert_eq!(live_devices_with(&options), 0, "listing made a device");
    // The first ID is the preferred physical device's, of which the runtime makes its default.
    let opened = Device::with_registry_id(ids[0], &options).unwrap();
    assert_eq!(opened.registry_id(), ids[0]);
    assert_eq!(
        live_devices_with(&options),
        1,
        "only the device of the ID is made"
    );
    let default = Device::system_default_with(&options).unwrap();
    assert_eq!(default.registry_id(), ids[0]);
    assert_eq!(
        live_devices_with(&options),
        1,
        "the default is the device opened"
    );
    let all = Device::all_with(&options).unwrap();
    let all: Vec<u64> = all.iter().map(Device::registry_id).collect();
    assert!(
        all.iter().all(|id| ids.contains(id)),
        "{all:x?} among {ids:x?}"
    );
    // Metal never answers 0, so no physical device has it.
    assert!(matches!(
        Device::with_registry_id(0, &options),
        Err(Error::NoDevice)
    ));
}

#[test]
fn the_default_device_s_identity_is_read_with_no_device_made_and_is_the_device_s_own() {
    let options = options();
    let identity = Device::system_default_identity_with(&options).unwrap();
    assert_eq!(
        live_devices_with(&options),
        0,
        "reading the identity made a device"
    );
    let device = Device::system_default_with(&options).unwrap();
    let properties = device.properties();
    // The same physical device: its UUID and the registry ID folded from it.
    assert_eq!(identity.device_uuid, properties.device_uuid);
    assert_eq!(identity.registry_id, device.registry_id());
    // And the same answers for every property the identity carries.
    assert_eq!(identity.name, properties.name);
    assert_eq!(identity.kind, properties.kind);
    assert_eq!(identity.vendor_id, properties.vendor_id);
    assert_eq!(identity.device_id, properties.device_id);
    assert_eq!(identity.api_version, properties.api_version);
    assert_eq!(identity.pci_bus, properties.pci_bus);
    assert_eq!(identity.drm_render, properties.drm_render);
    // The identity is the first of the physical devices the creators prefer, which is the one
    // `with_registry_id` opens by its own first listed ID.
    let ids = Device::registry_ids_with(&options).unwrap();
    assert_eq!(identity.registry_id, ids[0]);
    // A PCI bus address and a DRM render node are read only where the physical device lists
    // their extension, and a render node only where the driver reports the device has one, so
    // either may be `None`; where one is answered it is the same before and after the device
    // exists, which the equalities above hold. MoltenVK lists neither extension, so on macOS
    // both are `None` here.
    if cfg!(target_os = "macos") {
        assert_eq!(
            (properties.pci_bus, properties.drm_render),
            (None, None),
            "MoltenVK lists VK_EXT_pci_bus_info and VK_EXT_physical_device_drm for no device"
        );
    }
}

#[test]
fn every_listed_identity_is_read_with_no_device_made_and_is_its_device_s_own() {
    let options = options();
    let identities = Device::identities_with(&options).unwrap();
    assert!(!identities.is_empty(), "no physical device listed");
    let ids: Vec<u64> = identities
        .iter()
        .map(|identity| identity.registry_id)
        .collect();
    assert_eq!(ids, Device::registry_ids_with(&options).unwrap());
    let first = Device::system_default_identity_with(&options).unwrap();
    assert_eq!(first, identities[0]);
    assert_eq!(
        live_devices_with(&options),
        0,
        "reading the identities made a device"
    );
    let mut opened = 0;
    for identity in &identities {
        eprintln!("listed: {identity:?}");
        // Vulkan requires a `DEVICE_LOCAL` heap of every physical device.
        let heap = identity.recommended_max_working_set_size.unwrap();
        let device = match Device::with_registry_id(identity.registry_id, &options) {
            Ok(device) => device,
            // Listed all the same where the runtime refuses its device, which this answers.
            Err(error @ (Error::Unsupported(_) | Error::NoDevice)) => {
                eprintln!("refused: {error}");
                continue;
            }
            Err(error) => panic!("{}: {error}", identity.name),
        };
        let properties = device.properties();
        assert_eq!(device.registry_id(), identity.registry_id);
        assert_eq!(properties.device_uuid, identity.device_uuid);
        assert_eq!(properties.name, identity.name);
        assert_eq!(properties.kind, identity.kind);
        assert_eq!(properties.vendor_id, identity.vendor_id);
        assert_eq!(properties.device_id, identity.device_id);
        assert_eq!(properties.api_version, identity.api_version);
        assert_eq!(properties.pci_bus, identity.pci_bus);
        assert_eq!(properties.drm_render, identity.drm_render);
        // The heap no device property answers bounds the one that does.
        assert!(
            properties.max_buffer_length <= working_set_buffer_bound(heap),
            "maxBufferLength {} past three quarters of the heap {heap}",
            properties.max_buffer_length
        );
        opened += 1;
    }
    assert!(opened > 0, "the runtime made no device of {identities:?}");
}
