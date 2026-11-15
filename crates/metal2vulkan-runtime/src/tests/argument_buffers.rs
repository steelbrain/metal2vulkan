//! MoltenVK's Metal argument buffers, read from a device's limits at the releases whose source
//! was read (`device/argument_buffers.rs`).

use crate::device::argument_buffers::{
    check, setting, ArgumentBuffers, Facts, APPLE, LAYER, READ, SETTING, TIER_2_UPDATE_AFTER_BIND,
};
use crate::device::LayerSettingValue;
use crate::error::{DriverSetting, Error};
use metal2vulkan::device::DeviceDescription;
use metal2vulkan::vk::DriverId;

/// MoltenVK 1.4.1 on an Apple M3 Max, as `vulkaninfo` reported it (2026-10-07) with
/// `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS` on (the default) and off.
fn m3(release: u32, update_after_bind: u32, per_stage: u32) -> Facts {
    Facts {
        driver: Some(DriverId::MOLTENVK),
        release,
        vendor: APPLE,
        update_after_bind,
        per_stage,
    }
}

/// The table and the decode agree: at every release `READ` lists, MoltenVK's two shapes of the
/// limits (tier-2 argument buffers in use, or the per-stage limit) read as on and off; at a
/// release next to one, or one before every listed release, nothing is read, whatever the limits.
#[test]
fn the_read_releases_decode_both_states_and_no_other_release_does() {
    assert!(!READ.is_empty());
    for &release in READ {
        assert_eq!(
            m3(release, TIER_2_UPDATE_AFTER_BIND, 256).argument_buffers(),
            ArgumentBuffers::On,
            "{release}"
        );
        assert_eq!(
            m3(release, 128, 128).argument_buffers(),
            ArgumentBuffers::Off,
            "{release}"
        );
        // Limits of neither shape: MoltenVK changed what it reports, so nothing is read.
        assert_eq!(
            m3(release, 4096, 128).argument_buffers(),
            ArgumentBuffers::Unread
        );
        // A GPU of another vendor may be tier 1, whose limits are the per-stage ones with
        // argument buffers on.
        let other = Facts {
            vendor: 0x8086,
            ..m3(release, 128, 128)
        };
        assert_eq!(other.argument_buffers(), ArgumentBuffers::Unread);
        for unread in [release - 1, release + 1] {
            if READ.contains(&unread) {
                continue;
            }
            for (update_after_bind, per_stage) in [(TIER_2_UPDATE_AFTER_BIND, 256), (128, 128)] {
                assert_eq!(
                    m3(unread, update_after_bind, per_stage).argument_buffers(),
                    ArgumentBuffers::Unread,
                    "{unread}"
                );
            }
        }
    }
    assert_eq!(m3(0, 128, 128).argument_buffers(), ArgumentBuffers::Unread);
    for driver in [
        None,
        Some(DriverId::MESA_LLVMPIPE),
        Some(DriverId::MESA_RADV),
    ] {
        let facts = Facts {
            driver,
            ..m3(READ[0], 128, 128)
        };
        assert_eq!(facts.argument_buffers(), ArgumentBuffers::NotMoltenVk);
    }
}

/// Only a device without argument buffers is refused, naming the setting.
#[test]
fn a_device_without_argument_buffers_is_refused() {
    for state in [
        ArgumentBuffers::NotMoltenVk,
        ArgumentBuffers::On,
        ArgumentBuffers::Unread,
    ] {
        assert!(check(state).is_ok(), "{state:?}");
    }
    let refused = check(ArgumentBuffers::Off);
    assert_eq!(
        refused,
        Err(Error::DriverSetting(
            DriverSetting::MoltenVkArgumentBuffersOff
        ))
    );
    let message = refused.map_or_else(|error| error.to_string(), |()| String::new());
    assert!(
        message.contains("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS"),
        "{message}"
    );
}

/// The setting the runtime's instance chains is MoltenVK's, on.
#[test]
fn the_instance_asks_moltenvk_for_argument_buffers() {
    let asked = setting();
    assert_eq!(asked.layer.as_c_str(), LAYER);
    assert_eq!(asked.name.as_c_str(), SETTING);
    assert_eq!(asked.value, LayerSettingValue::Bool(true));
}

/// Every committed MoltenVK profile (each measured in MoltenVK's default configuration) reads
/// as binding through argument buffers, and every other profile as not MoltenVK.
#[test]
fn the_committed_profiles_read_as_measured() {
    let dir = metal2vulkan_devtools::validation_dir().join("profiles");
    let mut moltenvk = 0;
    for path in std::fs::read_dir(&dir).expect("validation/profiles is readable") {
        let path = path.expect("directory entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("profile is readable");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
        let expected = match desc.driver_id {
            Some(DriverId::MOLTENVK) => {
                moltenvk += 1;
                ArgumentBuffers::On
            }
            _ => ArgumentBuffers::NotMoltenVk,
        };
        assert_eq!(
            Facts::of(&desc).argument_buffers(),
            expected,
            "{}",
            path.display()
        );
    }
    assert!(moltenvk > 0, "no committed MoltenVK profile");
}
