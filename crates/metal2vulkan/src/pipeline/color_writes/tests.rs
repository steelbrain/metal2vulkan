//! [`color_writes`] of the 32-bit float attachments: a blend of factors other than one and zero
//! is the device's arithmetic (`ChannelWrite::Device`, which records
//! `PrecisionNote::DeviceColorConversion`) where the device does not store Metal's color writes,
//! and nothing otherwise.

use super::*;
use crate::metal::{BlendFactor, PixelFormat};

/// Attachment 0 of `pf`, blended source-over where `blend` is `Some(true)`, with factors one and
/// zero where `Some(false)`, not blended where `None`.
fn attachment(pf: PixelFormat, blend: Option<bool>) -> ColorAttachmentDescriptor {
    let mut c = ColorAttachmentDescriptor::new(pf);
    if let Some(over) = blend {
        c.blending_enabled = true;
        if over {
            c.source_rgb_blend_factor = BlendFactor::SourceAlpha;
            c.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
            c.source_alpha_blend_factor = BlendFactor::SourceAlpha;
            c.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        }
    }
    c
}

/// Every committed profile's map for the 32-bit float formats: each channel the device's under a
/// source-over blend, and no entry without a blend or under factors one and zero; an integer format takes none either way.
#[test]
fn a_blended_32_bit_float_channel_is_the_devices() {
    let mut device_blends = 0;
    for (stem, dev) in crate::metal::format::tests::committed_devices() {
        let p = dev.profile();
        for (pf, channels) in [
            (PixelFormat::R32Float, 1),
            (PixelFormat::RG32Float, 2),
            (PixelFormat::RGBA32Float, 4),
        ] {
            let map = |blend| {
                let colors = [Some(attachment(pf, blend))];
                color_writes(
                    [0],
                    &colors,
                    (false, TextureWriteRoundingMode::TowardZero),
                    p,
                )
            };
            let blended = map(Some(true));
            let mut want = [None; 4];
            for channel in want.iter_mut().take(channels) {
                *channel = Some(ChannelWrite::Device);
            }
            assert_eq!(blended.get(&0), Some(&want), "{stem} {pf:?}");
            device_blends += 1;
            assert!(map(None).is_empty(), "{stem} {pf:?} unblended");
            assert!(map(Some(false)).is_empty(), "{stem} {pf:?} one and zero");
        }
        let colors = [Some(attachment(PixelFormat::RGBA32Uint, None))];
        let integer = color_writes(
            [0],
            &colors,
            (false, TextureWriteRoundingMode::TowardZero),
            p,
        );
        assert!(integer.is_empty(), "{stem} RGBA32Uint");
    }
    assert!(
        device_blends > 0,
        "no committed profile blends on the device"
    );
}

/// A fragment function's map is the same over a sweep of blend states of an `RGBA8Unorm`
/// attachment -- no blend, one and zero, source-over -- on every committed profile: Metal's unorm
/// rule, which the attachment takes where it stores the output as written, is one the lowering
/// tells from the device's conversion nowhere (no profile is measured to store an 8-bit unorm
/// code as Metal does), so it is named the device's ([`lowered`]) and the sweep compiles the
/// function once. Whether the attachment takes Metal's rule still follows its blend
/// ([`takes_metal_rule`]), which link reads for the blend of one and zero.
#[test]
fn a_blend_sweep_of_a_rule_the_device_converts_compiles_under_one_map() {
    let mut converted = 0;
    for (stem, dev) in crate::metal::format::tests::committed_devices() {
        let p = dev.profile();
        let rounding = (false, TextureWriteRoundingMode::TowardZero);
        let sweep = |pf| {
            [None, Some(false), Some(true)].map(|blend| {
                let colors = [Some(attachment(pf, blend))];
                (
                    color_writes([0], &colors, rounding, p),
                    takes_metal_rule(0, &colors, rounding, p),
                )
            })
        };
        let [unblended, one_zero, over] = sweep(PixelFormat::RGBA8Unorm);
        assert_eq!(unblended.0, over.0, "{stem}: unblended and source-over");
        assert_eq!(one_zero.0, over.0, "{stem}: one and zero and source-over");
        assert_eq!(
            over.0.get(&0),
            Some(&[Some(ChannelWrite::Device); 4]),
            "{stem}"
        );
        converted += 1;
        assert_eq!(
            [unblended.1, one_zero.1, over.1],
            [true, true, false],
            "{stem}: the attachment takes Metal's rule where it stores the output as written"
        );
    }
    assert!(converted > 0, "no committed profile converts on the device");
}
