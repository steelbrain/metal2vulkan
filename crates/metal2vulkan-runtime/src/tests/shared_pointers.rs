//! `Device::imports_shared_host_pointers`: found once per device by importing one page of shared
//! memory, and answered from that after. MoltenVK (anonymous shared memory, handed to Metal) and
//! lavapipe (a `memfd`) import it; RADV refuses it (the 780M and the 7900 XTX, Mesa 26.2.4).
//! NVIDIA is expected to import it, as it imported such memory for no-copy buffers, and Intel's
//! Windows driver a view of a section (a UHD 630, driver 101.2145). Another driver's answer is
//! printed.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Nothing runs on the GPU.

use crate::{Device, DeviceKind, DeviceOptions, MessageSink};
use metal2vulkan::vk::DriverId;

#[test]
fn a_device_says_once_whether_it_imports_shared_memory_pointers() {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        // A group of devices no other test makes.
        messages: Some(MessageSink::new(|_| {})),
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    let expected = match device.properties().description.driver_id {
        Some(DriverId::MESA_RADV) => Some(false),
        Some(
            DriverId::MOLTENVK
            | DriverId::MESA_LLVMPIPE
            | DriverId::NVIDIA_PROPRIETARY
            | DriverId::INTEL_PROPRIETARY_WINDOWS,
        ) => Some(true),
        _ => None,
    };
    let imports = || device.work_counts().host_pointer_imports_made;
    let before = imports();
    let answer = device.imports_shared_host_pointers().unwrap();
    if let Some(expected) = expected {
        assert_eq!(answer, expected, "{}", device.name());
    }
    // The page's import, counted where the device took it.
    assert_eq!(imports() - before, u64::from(answer));
    assert_eq!(device.imports_shared_host_pointers().unwrap(), answer);
    assert_eq!(imports() - before, u64::from(answer), "asked again");
    eprintln!("shared host pointers on {}: {answer}", device.name());
}
