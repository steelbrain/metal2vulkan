//! `Device::from_queries` decides by Vulkan queries alone: the description's identity changes
//! nothing it derives.

use super::*;
use crate::device::description::APPLE_VENDOR_ID;

/// The identities each committed profile is also read under: MoltenVK on an Apple GPU, the
/// NVIDIA driver on a GA10x, RADV, Intel's Mesa driver and none.
fn identities() -> [(Option<vk::DriverId>, u32, u32); 5] {
    [
        (Some(vk::DriverId::MOLTENVK), APPLE_VENDOR_ID, 0x1_0009),
        (
            Some(vk::DriverId::NVIDIA_PROPRIETARY),
            0x10de,
            NVIDIA_AMPERE_GA10X[0],
        ),
        (Some(vk::DriverId::MESA_RADV), 0x1002, 0x744c),
        (Some(vk::DriverId::INTEL_OPEN_SOURCE_MESA), 0x8086, 0x9a49),
        (None, 0, 0),
    ]
}

/// On every committed profile, descriptions that differ only in `driver_id`, `vendor_id` and
/// `device_id` give one profile, whose only workarounds are the two every device but Apple's
/// takes; `Device::new` reads the identity, so at least one profile differs there.
#[test]
fn from_queries_derives_one_profile_whatever_the_identity() {
    let mut new_differs = false;
    for (stem, _) in committed_profiles() {
        let desc = profile(&stem);
        let derived: Vec<Option<DeviceProfile>> = identities()
            .into_iter()
            .map(|(driver, vendor, device)| {
                let mut other = desc.clone();
                other.driver_id = driver;
                other.vendor_id = vendor;
                other.device_id = device;
                Device::from_queries(other, DevicePolicy::default())
                    .ok()
                    .map(|d| d.profile().clone())
            })
            .collect();
        for profile in &derived[1..] {
            assert_eq!(profile, &derived[0], "{stem}");
        }
        let Some(queried) = &derived[0] else {
            continue;
        };
        assert_eq!(
            queried.workarounds(),
            [
                DriverWorkaround::ClippedCoverageNotMetals,
                DriverWorkaround::TessellatorNotMetals
            ],
            "{stem}"
        );
        if let Ok(identified) = Device::new(desc, DevicePolicy::default()) {
            new_differs |= identified.profile() != queried;
        }
    }
    assert!(
        new_differs,
        "no committed profile's identity changes Device::new"
    );
}
