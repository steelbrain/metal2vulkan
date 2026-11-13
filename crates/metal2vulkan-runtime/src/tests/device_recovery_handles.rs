//! What the device an object answers is across its device's recovery (`device/carry.rs`): the
//! caller's handle of the generation the object was on, kept as long as the caller holds it. A
//! recovery carries the object on and lets go of its own handle of the lost generation at that
//! generation's rest; a handle the caller took before keeps that generation, its `VkDevice`
//! included, until the caller lets it go, and its calls go on to the generation serving.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).

use super::device_recovery::{heard_device, recovery_notices, Heard, Scratch};
use super::device_recovery_carry::live_vk_devices;
use crate::{Device, ResourceOptions};
use metal2vulkan::bind::SamplerDescriptor;

/// `device`'s recovery, which `heard` hears announced once.
fn recover(device: &Device, heard: &Heard) {
    let from = heard.lock().unwrap().len();
    device.recover_for_test().unwrap();
    assert_eq!(recovery_notices(heard, from, "the device").len(), 1);
}

/// `held`, a handle of a lost generation, used: it names that generation, which serves no more,
/// and its calls go to the generation serving.
fn used(held: &Device) {
    assert!(!held.generation_serves());
    let buffer = held.new_buffer(16, ResourceOptions::shared()).unwrap();
    assert!(buffer.device().generation_serves());
}

/// Holds the device `$object`, made on `$device` before a recovery, answered, across the
/// recovery and the lost generation's rest (no command buffer records there), then uses it: the
/// lost `VkDevice` lives while the handle is held and goes with it, the object carried on.
macro_rules! held_across_a_recovery {
    ($device:expr, $heard:expr, $kind:literal, $object:expr) => {{
        let object = $object;
        let held = object.device();
        recover($device, $heard);
        assert_eq!(live_vk_devices($device, 2), 2, "{}'s device, held", $kind);
        used(&held);
        drop(held);
        assert_eq!(live_vk_devices($device, 1), 1, "{}'s device, let go", $kind);
        assert!(object.device().generation_serves(), "{}", $kind);
    }};
}

/// The device a fence, a sampler state, a `Private` buffer and an event answered before a
/// recovery each outlive the lost generation's rest while the caller holds it: the generation's
/// `VkDevice` lives until the handle goes, and the handle's calls reach the generation serving.
#[test]
fn a_held_device_of_an_object_outlives_the_lost_generation_s_rest() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    let sampler = SamplerDescriptor::new();
    held_across_a_recovery!(&device, &heard, "a fence", device.new_fence());
    held_across_a_recovery!(
        &device,
        &heard,
        "a sampler state",
        device.new_sampler_state(&sampler).unwrap()
    );
    held_across_a_recovery!(
        &device,
        &heard,
        "a Private buffer",
        device.new_buffer(256, ResourceOptions::private()).unwrap()
    );
    held_across_a_recovery!(&device, &heard, "an event", device.new_event().unwrap());
}
