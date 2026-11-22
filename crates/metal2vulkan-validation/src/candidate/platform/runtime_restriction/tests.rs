//! The profiles layer's settings as the runtime's instance takes them, and the refusal of a device
//! that lacks the profile, device-free.

use super::*;
use crate::check::admission::ExecutorError;

/// The layer's settings reach a runtime instance as the variables its settings library reads:
/// one per `VkLayerSettingEXT` setting the layer takes, named
/// `VK_KHRONOS_PROFILES_<SETTING>`, a flag as `true` or `false` and a list joined by commas, none
/// empty (the library reads an empty variable as unset), and the layer's reports kept off the
/// standard output a device worker reports on.
#[test]
fn the_layers_settings_reach_a_runtime_instance_as_its_variables() {
    let restriction = ProfileRestriction::committed("runtime-floor").unwrap();
    let hidden: BTreeSet<String> = [
        "VK_EXT_shader_viewport_index_layer".to_owned(),
        "VK_KHR_maintenance4".to_owned(),
    ]
    .into();
    let restrict = ProfilesSettings::new(ProfilesLayer::Restrict(&restriction, &hidden)).unwrap();
    let file = restriction.file.to_string_lossy().into_owned();
    let expected: Vec<(String, String)> = [
        ("PROFILE_EMULATION", "true"),
        ("PROFILE_FILE", file.as_str()),
        ("PROFILE_NAME", "VP_M2V_runtime_floor"),
        (
            "SIMULATE_CAPABILITIES",
            "SIMULATE_API_VERSION_BIT,SIMULATE_FEATURES_BIT,SIMULATE_PROPERTIES_BIT,\
             SIMULATE_EXTENSIONS_BIT,SIMULATE_FORMATS_BIT",
        ),
        ("DEFAULT_FEATURE_VALUES", "DEFAULT_FEATURE_VALUES_FALSE"),
        ("DEBUG_FAIL_ON_ERROR", "false"),
        ("DEBUG_ACTIONS", "DEBUG_ACTION_FILE_BIT"),
        ("DEBUG_FILENAME", "/dev/null"),
        ("EMULATE_PORTABILITY", "false"),
        (
            "EXCLUDE_DEVICE_EXTENSIONS",
            "VK_EXT_shader_viewport_index_layer,VK_KHR_maintenance4",
        ),
    ]
    .iter()
    .map(|(name, value)| (format!("VK_KHRONOS_PROFILES_{name}"), (*value).to_owned()))
    .collect();
    assert_eq!(restrict.environment(), expected);
    assert_eq!(restrict.environment().len(), restrict.settings().len());

    // Hiding nothing sets no exclusion; the native probe only turns the emulation off.
    let none = BTreeSet::new();
    let presenting = ProfilesSettings::new(ProfilesLayer::Restrict(&restriction, &none)).unwrap();
    assert!(presenting.environment().iter().all(|(name, value)| name
        != "VK_KHRONOS_PROFILES_EXCLUDE_DEVICE_EXTENSIONS"
        && !value.is_empty()));
    let probe = ProfilesSettings::new(ProfilesLayer::PassThrough).unwrap();
    assert_eq!(
        probe.environment(),
        [
            (
                "VK_KHRONOS_PROFILES_PROFILE_EMULATION".into(),
                "false".into()
            ),
            (
                "VK_KHRONOS_PROFILES_EMULATE_PORTABILITY".into(),
                "false".into()
            ),
        ]
    );
}

/// A run through the runtime restricted to a profile its device lacks capabilities of is a typed
/// device refusal, recorded with the device as it is, the restriction and the members, in the
/// restricted slot, under the runtime's fingerprint.
#[test]
fn a_restricted_run_on_a_device_without_the_profile_is_a_recorded_refusal() {
    use crate::candidate::engine::{engine_slot, ENGINE_EXECUTOR_ABI};
    use crate::refusal::{Refusal, RefusalClass, RefusalObservation};

    let restriction: &'static ProfileRestriction = Box::leak(Box::new(
        ProfileRestriction::committed("nvidia-pascal-linux").unwrap(),
    ));
    let gaps =
        vec!["limits.max_compute_work_group_invocations 1536 beyond the device's 1024".into()];
    let device = serde_json::json!({
        "device": "a GPU",
        "executor": "runtime-compute",
        "tools": [{"layer": "VK_LAYER_KHRONOS_validation", "name": "a tool", "purposes": 0,
                   "version": "1.0.0"}],
        "profiles_layer": null,
    });
    let refused = Refused {
        restriction,
        gaps: gaps.clone(),
        device,
    };
    assert!(refused
        .to_string()
        .contains("profile nvidia-pascal-linux states 1 capabilities the device lacks"));
    let attempt = refused.attempt();
    let not_supported = match attempt.result {
        Err(ExecutorError::NotSupported(not_supported)) => not_supported,
        other => panic!("{:?}", other.err()),
    };
    assert_eq!(
        not_supported,
        NotSupported::Restriction {
            profile: "nvidia-pascal-linux".into(),
            gaps: gaps.clone(),
        }
    );
    let opened = attempt.refused_on.expect("the refusal names the device");
    assert_eq!(opened.environment["device"], "a GPU");
    let layer = &opened.environment["profiles_layer"];
    assert_eq!(layer["profile"], "nvidia-pascal-linux");
    assert_eq!(layer["profile_sha256"], restriction.sha256.as_str());
    assert_eq!(layer["refused"], serde_json::json!(gaps));
    assert_eq!(opened.translator_fingerprint, runtime_fingerprint());
    let record = RefusalObservation {
        timing: Some(crate::observation::test_timing()),
        case_id: "a".repeat(64),
        air_sha256: "b".repeat(64),
        input_sha256: "c".repeat(64),
        translator_fingerprint: opened.translator_fingerprint.clone(),
        environment_id: engine_slot("nvidia-rtx5080-laptop", Some("nvidia-pascal-linux")),
        environment: opened.environment.clone(),
        executor_abi: ENGINE_EXECUTOR_ABI.into(),
        refusal: Refusal::of(&not_supported),
        artifact: None,
    };
    record.validate_content().unwrap();
    assert_eq!(record.refusal.class, RefusalClass::Device);
}
