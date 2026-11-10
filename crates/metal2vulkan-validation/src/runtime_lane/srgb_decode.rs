//! An 8-bit sRGB texture's texels read and sampled on the device the cfg selects, under the
//! validation layer, against Metal's decode (each color code `c` through Metal's twelve-bit table,
//! `k / 4095` for its step `k`; alpha `c / 255`): the device's own decode wherever the device
//! samples the format (`PrecisionNote::SrgbTexelDecode`).
//!
//! A 16 by 16 `RGBA8Unorm_sRGB` texture holds every code in each channel (texel `n`'s codes `n`,
//! `255 - n`, `7n + 3` and `13n + 5`, mod 256), read by a kernel's read and by its sample through a
//! nearest runtime sampler (`fpv_rgbx_textures`' `fpv_rgbx_fetch` and `fpv_rgbx_sample`). On a
//! device that samples through Metal's own texture unit (`VkProperty::MetalSampler`: MoltenVK on
//! Apple's GPU) every lane is Metal's bit for bit and nothing is announced. Elsewhere the
//! dispatches announce `srgb-texel-decode`, every color lane is the decode of its own code (nearer
//! Metal's value of that code than of either neighbour's), alpha is `c / 255` within one `f32` ulp,
//! and the lanes that differ from Metal's are counted with the largest difference, in table steps
//! (`1/4095`), on an `SRGB-DEVICE-DECODE` line.

use super::at_plan;
use super::degrade_notices::counts;
use super::degrade_notices::on_device_with;
use super::pixel_samplers::metal_sampler;
use super::rgbx_textures::run;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    Degradation, Device, Region, ResourceOptions, Texture, TextureDescriptor,
};

/// The texture's side.
const SIDE: u64 = 16;

/// The texels.
const TEXELS: usize = (SIDE * SIDE) as usize;

/// Texel `n`'s four codes: every code in each channel.
fn codes(n: usize) -> [u8; 4] {
    let n = n as u32;
    [n, 255 - n, (7 * n + 3) % 256, (13 * n + 5) % 256].map(|c| c as u8)
}

/// Metal's value of channel `channel`'s code `c`: a color code's step over 4095, alpha's code over
/// 255, each correctly rounded to `f32`.
fn metal(channel: usize, c: u8) -> f32 {
    match channel {
        3 => (f64::from(c) / 255.0) as f32,
        _ => (f64::from(super::srgb_step(u32::from(c))) / 4095.0) as f32,
    }
}

/// A shared texture of every code, read only.
fn texture(device: &Device) -> Texture {
    let bytes: Vec<u8> = (0..TEXELS).flat_map(codes).collect();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8UnormSRGB, SIDE, SIDE, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&d).unwrap();
    texture
        .replace_region(Region::new_2d(0, 0, SIDE, SIDE), 0, 0, &bytes, SIDE * 4, 0)
        .unwrap();
    texture
}

/// A read and a nearest sample of every code are the device's decode: Metal's bit for bit through
/// Metal's own texture unit, elsewhere announced, each color lane its own code's decode (the
/// module docs).
#[test]
fn an_srgb_texture_reads_the_devices_decode() {
    on_device_with(at_plan, |device, messages| {
        let name = device.name().to_owned();
        let metals = metal_sampler(device);
        let texture = texture(device);
        let _ = messages.take_notices();
        for (kernel, sampled) in [("fpv_rgbx_fetch", false), ("fpv_rgbx_sample", true)] {
            let out = run(device, kernel, &texture, &vec![0u8; TEXELS * 16], sampled);
            let got: Vec<f32> = out
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect();
            let (mut differ, mut steps, mut misread) = (0, 0f64, Vec::new());
            for n in 0..TEXELS {
                for (channel, c) in codes(n).into_iter().enumerate() {
                    let (read, want) = (got[4 * n + channel], metal(channel, c));
                    if read.to_bits() == want.to_bits() {
                        continue;
                    }
                    differ += 1;
                    if channel == 3 {
                        if read.to_bits().abs_diff(want.to_bits()) > 1 {
                            misread.push((n, channel, c, read, want));
                        }
                        continue;
                    }
                    steps = steps.max((f64::from(read) - f64::from(want)).abs() * 4095.0);
                    if !super::srgb_decode_holds(read, want, metals) {
                        misread.push((n, channel, c, read, want));
                    }
                }
            }
            println!(
                "SRGB-DEVICE-DECODE {name} {kernel}: {differ} of {} lanes differ from Metal's, \
                 color at most {steps:.4} table steps (1/4095) off",
                4 * TEXELS
            );
            assert!(
                misread.is_empty(),
                "{kernel} on {name}: lanes (texel, channel, code, read, Metal's) not their code's \
                 decode: {:?}",
                &misread[..misread.len().min(8)]
            );
            if metals {
                assert_eq!(differ, 0, "{kernel} on {name}: Metal's own texture unit");
            }
        }
        let announced = counts(&messages.take_notices())
            .get(&Degradation::SrgbTexelDecode)
            .copied()
            .unwrap_or(0);
        assert_eq!(announced > 0, !metals, "srgb-texel-decode on {name}");
    });
}
