//! The memory-model choice and its feature checks.

use super::*;
use crate::spirv::testing::committed_profiles;

fn needs(f: VkFeature) -> Error {
    Unsupported::needs(Feature::CoherentMemory, Capability::Feature(f)).into()
}

/// The committed profiles, and the runtime's floor without the Vulkan model's device scope and
/// without the model: devices below that floor, which the floor's own model leaves no committed
/// profile short of.
fn profiles() -> Vec<(String, crate::device::profile::DeviceProfile)> {
    let mut out = committed_profiles().expect("profiles");
    let scope = VkFeature::VulkanMemoryModelDeviceScope;
    for without in [&[scope][..], &[VkFeature::VulkanMemoryModel, scope]] {
        let p = crate::pipeline::testing::profile_without(
            "runtime-floor",
            without,
            crate::device::DevicePolicy::RESIDENT_TABLES,
        )
        .expect("below the floor");
        out.push((format!("runtime-floor without {without:?}"), p));
    }
    out
}

/// `GLSL450` needs nothing, with or without `Coherent` resources; the Vulkan model needs
/// `vulkanMemoryModel`, and with a `Device`-scope operation also `vulkanMemoryModelDeviceScope`,
/// on every committed profile and below the floor ([`profiles`]).
#[test]
fn the_vulkan_model_is_checked_against_the_profile() {
    let profiles = profiles();
    let (mut refused, mut accepted) = (0, 0);
    for (stem, p) in &profiles {
        for device_scope in [false, true] {
            let glsl = MemoryModelPlan {
                model: BarrierModel::Glsl450,
                device_scope,
                coherent_resources: device_scope,
            };
            assert_eq!(check(&glsl, p), Ok(()), "{stem}");
            let vulkan = MemoryModelPlan {
                model: BarrierModel::Vulkan,
                device_scope,
                coherent_resources: false,
            };
            let mm = p.fields().vulkan_memory_model;
            let ds = p
                .fields()
                .enabled
                .features
                .contains(&VkFeature::VulkanMemoryModelDeviceScope);
            let want = if !mm {
                Err(needs(VkFeature::VulkanMemoryModel))
            } else if device_scope && !ds {
                Err(needs(VkFeature::VulkanMemoryModelDeviceScope))
            } else {
                Ok(())
            };
            if want.is_ok() {
                accepted += 1;
            } else {
                refused += 1;
            }
            assert_eq!(
                check(&vulkan, p),
                want,
                "{stem} device_scope={device_scope}"
            );
        }
    }
    assert!(refused > 0 && accepted > 0, "both outcomes are exercised");
}

/// LLVM fence orderings map like `air.atomic.fence`'s; the non-fence orderings emit nothing.
#[test]
fn llvm_fence_orders_map_onto_vulkan() {
    assert_eq!(fence_order(Ordering::Unordered), None);
    assert_eq!(fence_order(Ordering::Monotonic), None);
    assert_eq!(fence_order(Ordering::Acquire), Some(MemoryOrder::Acquire));
    assert_eq!(fence_order(Ordering::Release), Some(MemoryOrder::Release));
    assert_eq!(
        fence_order(Ordering::AcqRel),
        Some(MemoryOrder::AcquireRelease)
    );
    assert_eq!(
        fence_order(Ordering::SeqCst),
        Some(MemoryOrder::AcquireRelease)
    );
}

/// The default plan is `GLSL450` (Metal's default model), without `Coherent` resources.
#[test]
fn the_default_model_is_glsl450() {
    assert_eq!(MemoryModelPlan::default().model, BarrierModel::Glsl450);
    assert!(!MemoryModelPlan::default().coherent_resources);
}

/// Coherent accesses that took the `GLSL450` form (a profile short of the Vulkan model or its
/// device scope) cannot meet the Vulkan model: its check names the feature the profile lacks.
#[test]
fn glsl450_coherent_resources_never_pass_under_the_vulkan_model() {
    let profiles = profiles();
    let mut checked = 0;
    for (stem, p) in &profiles {
        if vulkan_model(p) {
            continue;
        }
        let plan = MemoryModelPlan {
            model: BarrierModel::Vulkan,
            device_scope: false,
            coherent_resources: true,
        };
        let want = if p.fields().vulkan_memory_model {
            needs(VkFeature::VulkanMemoryModelDeviceScope)
        } else {
            needs(VkFeature::VulkanMemoryModel)
        };
        assert_eq!(check(&plan, p), Err(want), "{stem}");
        checked += 1;
    }
    assert!(checked > 0, "no profile lacks the Vulkan model");
}
