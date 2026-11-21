//! The extended-range blend's answer by the device's fetch ordering.

use super::*;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::metal::PixelFormat;

/// Committed profile `stem`.
fn profile(stem: &str) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
        .expect("device")
        .profile()
        .clone()
}

/// An extended-range format's blend (`extended_range_blend`): a padded form's taken where the device
/// orders fetch through rasterization-order input attachments, over which the function blends, and
/// refused naming that ordering elsewhere, the pixel interlock's devices included; one of Apple's
/// extended-range forms not implemented on every device (`codes_round_the_tile`); a bfloat form,
/// which Metal blends nowhere, Metal's refusal; any other format refused.
#[test]
fn an_extended_range_blend_needs_rasterization_order_fetch() {
    let xr = [
        PixelFormat::BGR10XR,
        PixelFormat::BGR10XRSRGB,
        PixelFormat::BGRA10XR,
        PixelFormat::BGRA10XRSRGB,
        PixelFormat::RGB10A82PXR10,
        PixelFormat::RGB10A82PXR10SRGB,
    ];
    let not_implemented = Err(Unsupported::not_implemented(Feature::Format));
    let blended = [
        PixelFormat::R10UnormX6,
        PixelFormat::R10UnormX6SRGB,
        PixelFormat::RG10UnormX12,
        PixelFormat::RG10UnormX12SRGB,
        PixelFormat::RGB10A2UnormSRGB,
    ];
    let bfloat = [
        PixelFormat::R16Bfloat,
        PixelFormat::RG16Bfloat,
        PixelFormat::RGBA16Bfloat,
    ];
    let unordered = Err(Unsupported::needs(
        Feature::FramebufferFetch,
        Capability::Feature(VkFeature::RasterizationOrderColorAttachmentAccess),
    ));
    for (stem, ordered) in [
        ("lavapipe", true),
        ("gate-lavapipe-x86_64", true),
        ("moltenvk-m3", false),
        ("amd-rdna3-radv", false),
        ("amd-780m-radv", false),
        ("intel-arl-anv", false),
        ("intel-irisxe-anv", false),
        ("nvidia-rtx5080-laptop", false),
        ("nvidia-pascal-linux", false),
        ("runtime-floor", false),
    ] {
        let p = profile(stem);
        assert_eq!(inputs_order(&p), ordered, "{stem}");
        for pf in blended {
            let want = if ordered { Ok(()) } else { unordered.clone() };
            assert_eq!(extended_range_blend(pf, &p), want, "{stem}: {pf:?}");
        }
        for pf in xr {
            assert_eq!(
                extended_range_blend(pf, &p),
                not_implemented,
                "{stem}: {pf:?}"
            );
        }
        for pf in bfloat {
            assert_eq!(
                extended_range_blend(pf, &p),
                Err(Unsupported::not_representable(Feature::Format)),
                "{stem}: {pf:?}"
            );
        }
        assert_eq!(
            extended_range_blend(PixelFormat::RGBA16Float, &p),
            not_implemented,
            "{stem}"
        );
    }
}
