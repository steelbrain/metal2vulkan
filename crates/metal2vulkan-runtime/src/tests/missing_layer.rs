//! A device opens although a layer its caller asked for is one the loader cannot provide
//! (`device/create.rs`): the layers are diagnostics, so `VK_ERROR_LAYER_NOT_PRESENT` never
//! reaches a caller that asked for a device. The device says as it is made which layers it was
//! made without, and a device whose layers the loader provides says nothing.
//!
//! Bounded by construction: each test opens one device, so at most two `vkCreateInstance` calls
//! (the layers asked for, then none) and one `VkInstance`, dropped with the device before the test
//! ends.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Nothing runs on the GPU.

use crate::device::create::{open_loader, LAYERS_DROPPED};
use crate::{DebugMessage, Device, DeviceKind, DeviceOptions, MessageSink};
use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex};

/// A layer name no loader lists, which the runtime must drop rather than fail on.
const ABSENT: &CStr = c"VK_LAYER_metal2vulkan_absent_for_this_test";

/// The validation layer, which this host may or may not provide.
const VALIDATION: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Whether the loader lists `name`, read before any instance is created: the loader's layer
/// enumeration is a two-call sequence that retries while it answers `VK_INCOMPLETE`, so it is
/// never run after a failed creation.
fn lists(name: &CStr) -> bool {
    let Ok(entry) = open_loader(None) else {
        return false;
    };
    // SAFETY: the loader is open; listing its layers needs nothing else.
    let listed = unsafe { entry.enumerate_instance_layer_properties() };
    let Ok(layers) = listed else {
        return false;
    };
    (layers.iter()).any(|layer| layer.layer_name_as_c_str().is_ok_and(|n| n == name))
}

/// The messages a device reports, and options of the kind the cfg selects that ask for `layers`
/// and report to them: a group of one device, which no other test shares (a sink is compared by
/// its identity).
fn watched(layers: Vec<CString>) -> (Arc<Mutex<Vec<DebugMessage>>>, DeviceOptions) {
    let said: Arc<Mutex<Vec<DebugMessage>>> = Arc::new(Mutex::new(Vec::new()));
    let reported = Arc::clone(&said);
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        layers,
        messages: Some(MessageSink::new(move |message: &DebugMessage| {
            let mut said = reported.lock().unwrap_or_else(|p| p.into_inner());
            // Bounded: a message list a flood cannot grow.
            if said.len() < 64 {
                said.push(message.clone());
            }
        })),
        ..DeviceOptions::default()
    };
    (said, options)
}

#[test]
fn a_device_opens_without_the_layers_the_loader_cannot_provide() {
    // One the loader cannot provide, beside one it may: the creation drops the layers it asked
    // for and creates the instance once more with none.
    let (said, options) = watched(vec![VALIDATION.to_owned(), ABSENT.to_owned()]);
    let device = Device::system_default_with(&options)
        .expect("a device opens although a layer asked for is missing");
    let name = device.name();
    let said = said.lock().unwrap_or_else(|p| p.into_inner()).clone();
    let dropped: Vec<&DebugMessage> = (said.iter())
        .filter(|message| message.id_name == LAYERS_DROPPED)
        .collect();
    assert_eq!(
        dropped.len(),
        1,
        "the device said once which layers it was made without: {said:#?}"
    );
    let text = &dropped[0].text;
    assert!(
        dropped[0].degradation.is_none(),
        "no degrade notice: nothing of Metal's answers differs"
    );
    // Every layer asked for is named, and the device.
    for layer in [VALIDATION, ABSENT] {
        let layer = layer.to_string_lossy();
        assert!(text.contains(layer.as_ref()), "{text}");
    }
    assert!(text.contains(&name), "{text}");
    drop(device);
}

/// Every layer asked for provided: the device is made with them and says nothing. Where the
/// loader lists none (this host), the device asks for none, which is the same answer.
#[test]
fn a_device_made_with_its_layers_says_nothing() {
    let layers: Vec<CString> = match lists(VALIDATION) {
        true => vec![VALIDATION.to_owned()],
        false => Vec::new(),
    };
    let (said, options) = watched(layers);
    let device = Device::system_default_with(&options).expect("a device opens");
    let said = said.lock().unwrap_or_else(|p| p.into_inner()).clone();
    assert!(
        !said.iter().any(|message| message.id_name == LAYERS_DROPPED),
        "nothing was dropped, so nothing is said: {said:#?}"
    );
    drop(device);
}
