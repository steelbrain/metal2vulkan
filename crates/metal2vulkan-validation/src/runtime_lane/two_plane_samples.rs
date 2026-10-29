//! Samples of Apple's two-plane extended-range textures on the device the cfg selects, under the
//! validation layer, against Metal's bytes.
//!
//! Each case replays one of a Metal probe `tx1`'s sample cases: a 16 x 16 texture of
//! `RGB10A8_2P_XR10` (`p2`) or its sRGB twin (`p2s`) holding seed 4's texels of the probe's
//! generator (`two_plane_copies::planes`), written as the probe writes them, the alpha bytes by a
//! kernel's `float` write and then the words by `replace_region` (`two_plane_copies::Lane`),
//! sampled at seed 5's 4,096 coordinates ([`coordinates`]) through a runtime sampler (the owned
//! fixture `xr_samples`'s kernels), nearest or linear, clamp to edge or repeat (`nc`, `lc`, `nr`,
//! `lr`), with a `float` and a `half` result; `lc` also through the fixture's constexpr linear
//! clamp-to-edge sampler.
//!
//! Metal's answer: the SHA-256 of each case's 4,096 results ([`METAL`]; an Apple
//! M4 under macOS 27.0.1, through the probe's constexpr samplers, plain and under Metal's
//! validation layer alike), which the probe keeps. A nearest case is Metal's bytes; a linear one
//! runs, its blend the device's `FMix` of the decoded taps (`bicubic-filter-arithmetic`), which a
//! digest cannot bound, so its values are compared by the conformance suite's
//! `xr-two-plane-samples` cases. A differing case is printed with its first results, on one
//! `XR-TWO-PLANE-SAMPLES-DIFFERS` line.
//!
//! A fragment function's samples of such a texture run the variant that takes its slot, counted
//! per draw (`WorkCounts::two_plane_draws`): [`draws_that_sample_a_two_plane_texture_are_counted`].

use super::degrade_notices::on_device;
use super::extended_range_samples::state;
use super::extended_range_two_planes::digest;
use super::read_through_views::{draw, sampling_state, Sampled};
use super::two_plane_copies::{lane, planes, Rng, FORMATS};
use super::{buffer_with, read_buffer};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, SamplerAddressMode, SamplerMinMagFilter, TextureUsage};
use metal2vulkan_runtime::{CommandBufferStatus, Size, TextureDescriptor};
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// The texture's extent.
const SIDE: u64 = 16;
/// The samples of each case.
const SAMPLES: u64 = 4096;

/// Metal's results: the probe's case and the SHA-256 of its 4,096 results.
const METAL: [(&str, &str); 16] = [
    (
        "smp.p2.nc.f",
        "86f9c37283b0079b6097e6e9ecb9671e5268ede621eb5a33dbbefd38d500b930",
    ),
    (
        "smp.p2.nc.h",
        "6b93e62120869397bd673d40a1ccb3123b452ce32b4565bde726777cd7348f79",
    ),
    (
        "smp.p2.lc.f",
        "304c290289b1c88f61c9092f05095ae4fc3250e0683c6ef849b6df7ba4072a9e",
    ),
    (
        "smp.p2.lc.h",
        "056c57a2917517418b57ec861f9c8de8ebb998d0e1a79aa78ba51f2ed865b221",
    ),
    (
        "smp.p2.nr.f",
        "bf509ad9a15dc8d1efb75623e4fe5ac2f00e279a5d62d2dd46a7a2e57f95de81",
    ),
    (
        "smp.p2.nr.h",
        "f484670908816aa1f92ba594c38183add72283a6b07a2ddc27eb92731b5dcce6",
    ),
    (
        "smp.p2.lr.f",
        "7f28042f92e7ba719bf0eae74a0b0dd88bccc3360d9f409762459f54e0eabf84",
    ),
    (
        "smp.p2.lr.h",
        "27b3a8342de65f38ffcd578a91f47c3d18d9d4bbf0f30dad35b97f5a985fa9c1",
    ),
    (
        "smp.p2s.nc.f",
        "89bb6b1fb2deadbfcd8080a0da18d82c3e7e4bb3f47210839c0015c90d6e5172",
    ),
    (
        "smp.p2s.nc.h",
        "9c9abfd12e72b156e1e8b8fbbb00eeb8c8f0bd289c817d0e99781a1afe21faef",
    ),
    (
        "smp.p2s.lc.f",
        "18e7c93583b98c9c11825886e6f0d39d8a05097e67ba2f520464eeb4bdd43350",
    ),
    (
        "smp.p2s.lc.h",
        "5dcc9940c0db2c033ef4dc9a7c9e869bcbbe5525ca5037fdaa28b7d148b01a23",
    ),
    (
        "smp.p2s.nr.f",
        "c264de965958fd80be78dacc7178cd0f20cd5ddd74de5106f2536fc53706468b",
    ),
    (
        "smp.p2s.nr.h",
        "4a767f4203aa7c3ee7de16002e378d8b4de29ff808e4b1f4d0957b5a10b4d76e",
    ),
    (
        "smp.p2s.lr.f",
        "6ff35014e270335757efb75b99ab4aa7968abe78497eec1b690161781facbdd9",
    ),
    (
        "smp.p2s.lr.h",
        "0df22c36e81b77136e19de14419c0e414bdd635f07fb54329a6d7213180ffaaa",
    ),
];

/// The probe's 4,096 normalized coordinates, seed 5's: the 256 texel centers, 512 random points
/// inside the texture, 512 within `32 / 8192` of a texel boundary on each axis, and 2,816 in
/// `[-0.3, 1.3)` on each axis, past the edges too.
fn coordinates() -> Vec<[f32; 2]> {
    let mut rng = Rng::new(5);
    let side = SIDE as f64;
    let unit = |rng: &mut Rng| (rng.next() >> 11) as f64 / 9_007_199_254_740_992.0;
    (0..SAMPLES)
        .map(|i| {
            if i < SIDE * SIDE {
                let center = |k: u64| (k as f32 + 0.5) / SIDE as f32;
                [center(i % SIDE), center(i / SIDE)]
            } else if i < 768 {
                let x = unit(&mut rng) as f32;
                [x, unit(&mut rng) as f32]
            } else if i < 1280 {
                let near = |rng: &mut Rng| {
                    let edge = rng.below(SIDE + 1) as f64 / side;
                    (edge + (rng.below(65) as f64 - 32.0) / 8192.0) as f32
                };
                let x = near(&mut rng);
                [x, near(&mut rng)]
            } else {
                let x = (-0.3 + 1.6 * unit(&mut rng)) as f32;
                [x, (-0.3 + 1.6 * unit(&mut rng)) as f32]
            }
        })
        .collect()
}

/// The sampler state of the probe's sampler `name`: nearest or linear (`n`, `l`), clamp to edge
/// or repeat (`c`, `r`), normalized coordinates.
fn sampler(name: &str) -> SamplerDescriptor {
    let mut d = SamplerDescriptor::new();
    if name.starts_with('l') {
        d.min_filter = SamplerMinMagFilter::Linear;
        d.mag_filter = SamplerMinMagFilter::Linear;
    }
    if name.ends_with('r') {
        d.s_address_mode = SamplerAddressMode::Repeat;
        d.t_address_mode = SamplerAddressMode::Repeat;
    }
    d
}

/// Each case's samples, through the runtime sampler of the probe's state (and `lc` through the
/// constexpr one too), run; a nearest case's are Metal's bytes (a linear one's blend is the
/// device's, module docs).
#[test]
fn two_plane_samples_are_metal_s() {
    on_device(|device, _| {
        let lane = lane(device);
        let coords: Vec<u8> = coordinates()
            .into_iter()
            .flatten()
            .flat_map(f32::to_le_bytes)
            .collect();
        let coords = buffer_with(device, &coords).unwrap();
        let mut wrong = Vec::new();
        let mut checked = 0;
        for (format, name) in FORMATS {
            let (texture, _, _) = lane.filled(format, 4, (SIDE, SIDE));
            for s in ["nc", "lc", "nr", "lr"] {
                let runtime = device.new_sampler_state(&sampler(s)).unwrap();
                for half in [false, true] {
                    let suffix = if half { "_half" } else { "" };
                    let case = format!("smp.{name}.{s}.{}", if half { 'h' } else { 'f' });
                    let mut kernels = vec![(format!("xr_sample_runtime{suffix}"), Some(&runtime))];
                    if s == "lc" {
                        kernels.push((format!("xr_sample_constexpr{suffix}"), None));
                    }
                    for (kernel, sampler) in kernels {
                        let state = state(device, &kernel);
                        let size = if half { 8 } else { 16 };
                        let out =
                            buffer_with(device, &vec![0xab; (SAMPLES * size) as usize]).unwrap();
                        let command_buffer = lane.queue.command_buffer().unwrap();
                        let mut encoder = command_buffer.compute_command_encoder().unwrap();
                        encoder.set_compute_pipeline_state(&state).unwrap();
                        encoder.set_texture(Some(&texture), 0).unwrap();
                        if let Some(sampler) = sampler {
                            encoder.set_sampler_state(Some(sampler), 0).unwrap();
                        }
                        encoder.set_buffer(Some(&coords), 0, 0).unwrap();
                        encoder.set_buffer(Some(&out), 0, 1).unwrap();
                        encoder
                            .dispatch_threads(Size::new(SAMPLES, 1, 1), Size::new(64, 1, 1))
                            .unwrap();
                        encoder.end_encoding().unwrap();
                        command_buffer.commit().unwrap();
                        command_buffer.wait_until_completed_for(WAIT).unwrap();
                        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                        assert_eq!(command_buffer.error(), None);
                        let bytes = read_buffer(&out, 0, SAMPLES * size).unwrap();
                        let want = METAL
                            .iter()
                            .find(|m| m.0 == case)
                            .map(|m| m.1)
                            .unwrap_or_else(|| panic!("no Metal digest of {case}"));
                        let got = digest(&bytes);
                        if s.starts_with('l') {
                            continue;
                        }
                        checked += 1;
                        if got != want {
                            let first: String = bytes[..(16 * size) as usize]
                                .iter()
                                .map(|b| format!("{b:02x}"))
                                .collect();
                            wrong.push(format!(
                                "{case} through {kernel}: {got}, Metal {want}; the first 16 results {first}"
                            ));
                        }
                    }
                }
            }
        }
        for line in &wrong {
            println!("XR-TWO-PLANE-SAMPLES-DIFFERS {}: {line}", device.name());
        }
        println!(
            "XR-TWO-PLANE-SAMPLES {}: {checked} cases of {SAMPLES} samples checked, {} differ from Metal",
            device.name(),
            wrong.len()
        );
        assert!(wrong.is_empty(), "{} cases differ from Metal", wrong.len());
    });
}

/// A draw whose fragment function samples a two-plane texture, nearest or linear, runs the variant
/// that takes its slot and counts once (`WorkCounts::two_plane_draws`); the same pipeline's draws
/// over an `RGBA8Unorm` texture count none.
#[test]
fn draws_that_sample_a_two_plane_texture_are_counted() {
    on_device(|device, _| {
        let lane = lane(device);
        let state = sampling_state(device);
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        d.usage = TextureUsage::SHADER_READ;
        let plain = device.new_texture(&d).unwrap();
        let counted = || device.work_counts().two_plane_draws;
        let before = counted();
        for sampled in [Sampled::Nearest, Sampled::Linear] {
            draw(&lane.queue, &state, &plain, sampled);
        }
        assert_eq!(counted(), before, "draws over an RGBA8Unorm texture");
        for (format, name) in FORMATS {
            let (texture, _, _) = lane.filled(format, 4, (SIDE, SIDE));
            for sampled in [Sampled::Nearest, Sampled::Linear] {
                let before = counted();
                draw(&lane.queue, &state, &texture, sampled);
                assert_eq!(counted(), before + 1, "{name}: a {sampled:?} draw");
            }
        }
    });
}

/// The SHA-256 of seed 4's planes and of seed 5's coordinates, pinned beside the FNV-1a words the
/// probe printed of them, whose 32 bits alone leave a case's inputs loosely held.
const INPUT_SHA: [&str; 3] = [
    "a763539f6746b2fc54e795303a2a02ac9727cc31382c7e3b9b03cd63c1d7b7d3",
    "96997d686459de2ca5fa082997cd27fcf23e0839edef9a3397f7f07c7457ff31",
    "508d5b75cbb92c96744c9eb9651df674316de37fdf460e1e3a9568bfa5b78cb5",
];

/// The generator gives the probe's texels and coordinates: the FNV-1a digests the probe printed
/// of seed 4's planes and of seed 5's coordinates (the probe's results), and their SHA-256.
#[test]
fn the_coordinates_are_the_probes() {
    use sha2::Digest;
    let fnv = |bytes: &[u8]| {
        bytes.iter().fold(2_166_136_261u32, |h, &b| {
            (h ^ u32::from(b)).wrapping_mul(16_777_619)
        })
    };
    let sha = |bytes: &[u8]| crate::hash::hex_encode(sha2::Sha256::digest(bytes));
    let (words, alpha) = planes(4, SIDE * SIDE);
    assert_eq!(fnv(&words), 0xa9c5_8e27);
    assert_eq!(fnv(&alpha), 0xe564_a75d);
    let coords: Vec<u8> = coordinates()
        .into_iter()
        .flatten()
        .flat_map(f32::to_le_bytes)
        .collect();
    assert_eq!(fnv(&coords), 0x1efe_3caa);
    for (bytes, want) in [&words, &alpha, &coords].into_iter().zip(INPUT_SHA) {
        assert_eq!(sha(bytes), want);
    }
}
