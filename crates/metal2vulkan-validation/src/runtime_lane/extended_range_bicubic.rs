//! Bicubic samples of extended-range textures on the device the cfg selects, under the validation
//! layer, against Metal's answers.
//!
//! The owned fixture `xr_bicubic`'s kernels sample each texture of the extended-range bicubic
//! table (`metal2vulkan_devtools::xr_bicubic`: random codes of each `BGR10_XR`, `BGRA10_XR` and
//! two-plane format and `_sRGB` twin at five sizes, written as Metal's probe wrote them) at the
//! table's coordinates, through constexpr bicubic samplers and through a runtime sampler whose
//! filters are `Cubic`. Metal's answer is the table's (an Apple M3 Max on macOS 26.7); the
//! device's is the bicubic filter in its own arithmetic (`bicubic-filter-arithmetic`, which each
//! kernel's state announces), each lane within two steps of its grid and the result's rounding of
//! Metal's (`xr_bicubic::near`). A lane past that is printed on an `XR-BICUBIC-DIFFERS` line.

use super::degrade_notices::{counts, on_device};
use super::extended_range_samples::{format, state_of, texel};
use super::two_plane_copies::{lane, Lane};
use super::{buffer_with, read_buffer};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, SamplerAddressMode, SamplerMinMagFilter, TextureUsage};
use metal2vulkan_devtools::xr_bicubic::{self, KERNELS, RAW_FORMATS, SIZES};
use metal2vulkan_runtime::{
    CommandBufferStatus, Degradation, Region, ResourceOptions, Size, Texture, TextureDescriptor,
};
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// A `w` x `h` texture of raw format `raw` holding `codes`, written as the probe wrote it: a
/// two-plane texture's alpha bytes by a kernel's `float` write, then its color words; another's
/// texels by `replace_region`.
fn texture(lane: &Lane<'_>, raw: u32, codes: &[[u32; 4]], (w, h): (u64, u64)) -> Texture {
    if let 550 | 551 = raw {
        let words: Vec<u8> = codes
            .iter()
            .flat_map(|&[r, g, b, _]| (r << 20 | g << 10 | b).to_le_bytes())
            .collect();
        let alpha: Vec<u8> = codes.iter().map(|c| c[3] as u8).collect();
        let format = match raw {
            550 => PixelFormat::RGB10A82PXR10,
            _ => PixelFormat::RGB10A82PXR10SRGB,
        };
        return lane.holding(format, (&words, &alpha), (w, h));
    }
    let bytes: Vec<u8> = codes.iter().flat_map(|&c| texel(raw, c)).collect();
    let bytes_per_texel = if raw <= 553 { 8 } else { 4 };
    let mut descriptor = TextureDescriptor::texture_2d(format(raw), w, h, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = lane.device.new_texture(&descriptor).unwrap();
    texture
        .replace_region(
            Region::new_2d(0, 0, w, h),
            0,
            0,
            &bytes,
            w * bytes_per_texel,
            0,
        )
        .unwrap();
    texture
}

/// Each sample of the table, through each of the fixture's kernels, is Metal's lanes, one step
/// from them at a marked tie, and each kernel's state announces the filter's arithmetic.
#[test]
fn bicubic_samples_of_extended_range_textures_are_metal_s() {
    on_device(|device, messages| {
        let lane = lane(device);
        let mut cubic = SamplerDescriptor::new();
        cubic.min_filter = SamplerMinMagFilter::Cubic;
        cubic.mag_filter = SamplerMinMagFilter::Cubic;
        cubic.s_address_mode = SamplerAddressMode::ClampToEdge;
        cubic.t_address_mode = SamplerAddressMode::ClampToEdge;
        let sampler = device.new_sampler_state(&cubic).unwrap();
        let table = xr_bicubic::table();
        let textures = xr_bicubic::textures();
        let (mut checked, mut apart, mut wrong) = (0, 0, Vec::new());
        for kernel in KERNELS {
            let state = state_of(device, "xr_bicubic", kernel);
            let half = kernel.ends_with("_half");
            let size = if half { 8 } else { 16 };
            for raw in RAW_FORMATS {
                for (w, h) in SIZES {
                    let samples = &table[&(raw, (w, h), kernel.to_string())];
                    let texture = texture(
                        &lane,
                        raw,
                        &textures[&(raw, (w, h))],
                        (u64::from(w), u64::from(h)),
                    );
                    let coords: Vec<u8> = samples
                        .iter()
                        .flat_map(|s| s.at.into_iter().flat_map(f32::to_le_bytes))
                        .collect();
                    let coords = buffer_with(device, &coords).unwrap();
                    let n = samples.len() as u64;
                    let out = buffer_with(device, &vec![0xab; (n * size) as usize]).unwrap();
                    let command_buffer = lane.queue.command_buffer().unwrap();
                    let mut encoder = command_buffer.compute_command_encoder().unwrap();
                    encoder.set_compute_pipeline_state(&state).unwrap();
                    encoder.set_texture(Some(&texture), 0).unwrap();
                    if kernel.starts_with("xrb_runtime") {
                        encoder.set_sampler_state(Some(&sampler), 0).unwrap();
                    }
                    encoder.set_buffer(Some(&coords), 0, 0).unwrap();
                    encoder.set_buffer(Some(&out), 0, 1).unwrap();
                    encoder
                        .dispatch_threads(Size::new(n, 1, 1), Size::new(n, 1, 1))
                        .unwrap();
                    encoder.end_encoding().unwrap();
                    command_buffer.commit().unwrap();
                    command_buffer.wait_until_completed_for(WAIT).unwrap();
                    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                    assert_eq!(command_buffer.error(), None);
                    let bytes = read_buffer(&out, 0, n * size).unwrap();
                    for (s, v) in samples.iter().zip(bytes.chunks_exact(size as usize)) {
                        let got: [u32; 4] = std::array::from_fn(|c| match half {
                            true => u32::from(u16::from_le_bytes([v[2 * c], v[2 * c + 1]])),
                            false => u32::from_le_bytes([
                                v[4 * c],
                                v[4 * c + 1],
                                v[4 * c + 2],
                                v[4 * c + 3],
                            ]),
                        });
                        for (c, (&got, &want)) in got.iter().zip(&s.lanes).enumerate() {
                            checked += 1;
                            apart += usize::from(got != want);
                            if !xr_bicubic::near(got, want, xr_bicubic::grid(raw, c), half) {
                                wrong.push(format!(
                                    "{kernel} {raw} {w}x{h} at {:?} lane {c}: {got:#x}, Metal \
                                     {want:#x}",
                                    s.at
                                ));
                            }
                        }
                    }
                }
            }
        }
        let notices = counts(&messages.take_notices());
        for line in &wrong {
            println!("XR-BICUBIC-DIFFERS {}: {line}", device.name());
        }
        println!(
            "XR-BICUBIC {}: {checked} lanes checked ({apart} apart from Metal's), {} past the \
             device's arithmetic",
            device.name(),
            wrong.len()
        );
        assert!(
            wrong.is_empty(),
            "{} lanes past the device's arithmetic",
            wrong.len()
        );
        assert!(
            notices.contains_key(&Degradation::BicubicFilterArithmetic),
            "the filter's arithmetic is announced: {notices:?}"
        );
    });
}
