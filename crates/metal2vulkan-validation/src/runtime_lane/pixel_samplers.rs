//! Samples through pixel-coordinate samplers (`PrecisionNote::PixelSamplerScaling`; the consumer's
//! ask of 2026-10-09) on the device the cfg selects, under the validation layer, against plain
//! Metal's answers.
//!
//! Metal's answers (our own probe, an Apple M4 under macOS 27.0.1, two processes,
//! identical): a 5x3 `RGBA8Unorm` and a 5x3 `RGBA16Float` texture, sampled at level 0 on a grid of
//! 80 x 10 coordinates in and outside the texture (each texel edge and centre, one ulp either
//! side, quarter and 1/512 steps; [`grid`]) through a runtime sampler state of
//! `normalizedCoordinates` NO at a sampler slot (`slot`) and in an argument buffer beside its
//! texture (`arg`), and through a constexpr `coord::pixel` sampler (`const`); nearest and linear
//! filters; clamp to edge, to zero, to an opaque white border, and (runtime states) mirror clamp
//! to edge. Each configuration's `float4` results hash ([`METAL`]); a configuration names its
//! format, source, filter and address mode. Metal answered each `slot` and `arg` configuration
//! exactly as the `norm` control: an argument buffer's normalized state of the same filter and
//! address mode sampled at the coordinate divided by the size. A runtime state's clamp to zero
//! reads its border color, here opaque white, as clamp to the border does, and a constexpr one
//! transparent black.
//!
//! Each configuration here runs the same grid through the runtime. A constexpr sampler's sample
//! of this 2D texture hands the device's unnormalized sampler the coordinate itself; a nearest
//! runtime or argument buffer's sampler's hands the device's normalized sampler its texel's centre
//! over the size, where the device reads exactly that texel; a linear one hands it the coordinate
//! over the size, which the device's sampler reads in its own arithmetic: the stages record it
//! (`PrecisionNote::PixelSamplerScaling`, and `ArgumentPixelSamplerScaling` for an argument
//! buffer's sampler) and every device announces it. Each pixel configuration is held to its
//! normalized control on the same device (the runtime state's `norm`; for `const`, a constexpr
//! sampler of the same words in normalized coordinates): a cell that differs from the control at
//! the same coordinate must be the control's answer at a coordinate within [`REACH`] ulps of it on
//! each axis, and the widest shift each configuration needed is printed (`PIXEL-SAMPLER-REACH`
//! lines). A `const` linear configuration on a device whose sampler is not Metal's is the one
//! exception: it is the device's linear filter in unnormalized coordinates, whose arithmetic is
//! that path's own rather than the normalized filter's at a moved coordinate (lavapipe: 79 to 107
//! of 800 cells of clamp to zero and to the border apart from the control in their last bits, none
//! of clamp to edge), so it is printed and not held to the control. A configuration whose hash is
//! Metal's needs no control: the NVIDIA RTX 5080 Laptop's normalized nearest controls answer the
//! texel below at an edge (`(1, -0.5)` reads texel 0, where the pixel configurations and Metal
//! read texel 1), past any shift within [`REACH`] ulps, and every nearest pixel configuration
//! there is Metal's. On a device with
//! `VkProperty::MetalSampler` (MoltenVK) each `norm` control's hash, each `const` configuration's
//! and each nearest configuration's must be Metal's. Elsewhere every sample is the device's
//! sampler's, which the stages record (`PrecisionNote::DeviceSampling`) and the device announces.
//! Every configuration that is not Metal's is printed (`PIXEL-SAMPLER-DIFFERS` lines) without
//! failing where it need not be Metal's. An argument buffer's note waits for a dispatch while a
//! pixel state lives ([`the_argument_buffer_note_waits_for_a_pixel_state`]).

use super::argument_linear_samples::kernel;
use super::degrade_notices::counts;
use super::depth_compares::{pipeline_of, run_held, Held, Through};
use super::driver_deviations::stage_notes;
use super::extended_range_targets::half_bits;
use super::{layer_reported_nothing_at_all, validated_device, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::VkProperty;
use metal2vulkan::metal::{
    PixelFormat, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter,
};
use metal2vulkan::pipeline::PrecisionNote;
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    ComputePipelineState, Degradation, Device, Region, SamplerState, SamplerStateOptions, Texture,
    TextureDescriptor,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// The textures' size: neither side a power of two.
const W: u64 = 5;
const H: u64 = 3;

/// Plain Metal's answers: each configuration's results hashed (the module docs).
const METAL: [(&str, &str); 60] = [
    (
        "rgba8_slot_nearest_edge",
        "250024876cde0c2f67dccea354f400d28c94b6cd9634aae0b38c6354f35070b7",
    ),
    (
        "rgba8_arg_nearest_edge",
        "250024876cde0c2f67dccea354f400d28c94b6cd9634aae0b38c6354f35070b7",
    ),
    (
        "rgba8_norm_nearest_edge",
        "250024876cde0c2f67dccea354f400d28c94b6cd9634aae0b38c6354f35070b7",
    ),
    (
        "rgba8_const_nearest_edge",
        "250024876cde0c2f67dccea354f400d28c94b6cd9634aae0b38c6354f35070b7",
    ),
    (
        "rgba8_slot_nearest_zero",
        "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
    ),
    (
        "rgba8_arg_nearest_zero",
        "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
    ),
    (
        "rgba8_norm_nearest_zero",
        "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
    ),
    (
        "rgba8_const_nearest_zero",
        "4ce73456509dc569e5a67d724ecd5f2bea5dda4d8b7279b5f8e4a7351998c5be",
    ),
    (
        "rgba8_slot_nearest_border",
        "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
    ),
    (
        "rgba8_arg_nearest_border",
        "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
    ),
    (
        "rgba8_norm_nearest_border",
        "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
    ),
    (
        "rgba8_const_nearest_border",
        "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
    ),
    (
        "rgba8_slot_nearest_mirror",
        "27c2ee7d7782b153e1e9daa8f2e37dba6ad073e4dc1f08b6350634a0b66fdf32",
    ),
    (
        "rgba8_arg_nearest_mirror",
        "27c2ee7d7782b153e1e9daa8f2e37dba6ad073e4dc1f08b6350634a0b66fdf32",
    ),
    (
        "rgba8_norm_nearest_mirror",
        "27c2ee7d7782b153e1e9daa8f2e37dba6ad073e4dc1f08b6350634a0b66fdf32",
    ),
    (
        "rgba8_slot_linear_edge",
        "4c832fb335efa8b6f389e8a2b8a6721efff8bef2d0d933df0670336ee3841051",
    ),
    (
        "rgba8_arg_linear_edge",
        "4c832fb335efa8b6f389e8a2b8a6721efff8bef2d0d933df0670336ee3841051",
    ),
    (
        "rgba8_norm_linear_edge",
        "4c832fb335efa8b6f389e8a2b8a6721efff8bef2d0d933df0670336ee3841051",
    ),
    (
        "rgba8_const_linear_edge",
        "4c832fb335efa8b6f389e8a2b8a6721efff8bef2d0d933df0670336ee3841051",
    ),
    (
        "rgba8_slot_linear_zero",
        "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
    ),
    (
        "rgba8_arg_linear_zero",
        "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
    ),
    (
        "rgba8_norm_linear_zero",
        "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
    ),
    (
        "rgba8_const_linear_zero",
        "8c8ebb90b930331d3790933171428b8edff1c8296f8bb344a6c06362dc382c83",
    ),
    (
        "rgba8_slot_linear_border",
        "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
    ),
    (
        "rgba8_arg_linear_border",
        "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
    ),
    (
        "rgba8_norm_linear_border",
        "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
    ),
    (
        "rgba8_const_linear_border",
        "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
    ),
    (
        "rgba8_slot_linear_mirror",
        "26940521965a1456adaba62f9dc004a9f27967fd355737d6efdca7c46209b80b",
    ),
    (
        "rgba8_arg_linear_mirror",
        "26940521965a1456adaba62f9dc004a9f27967fd355737d6efdca7c46209b80b",
    ),
    (
        "rgba8_norm_linear_mirror",
        "26940521965a1456adaba62f9dc004a9f27967fd355737d6efdca7c46209b80b",
    ),
    (
        "rgba16f_slot_nearest_edge",
        "e28b6390ff83ff95e1c5d00dae0fcf2f7b51ad9c20c68e88223e25cfc18700bd",
    ),
    (
        "rgba16f_arg_nearest_edge",
        "e28b6390ff83ff95e1c5d00dae0fcf2f7b51ad9c20c68e88223e25cfc18700bd",
    ),
    (
        "rgba16f_norm_nearest_edge",
        "e28b6390ff83ff95e1c5d00dae0fcf2f7b51ad9c20c68e88223e25cfc18700bd",
    ),
    (
        "rgba16f_const_nearest_edge",
        "e28b6390ff83ff95e1c5d00dae0fcf2f7b51ad9c20c68e88223e25cfc18700bd",
    ),
    (
        "rgba16f_slot_nearest_zero",
        "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
    ),
    (
        "rgba16f_arg_nearest_zero",
        "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
    ),
    (
        "rgba16f_norm_nearest_zero",
        "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
    ),
    (
        "rgba16f_const_nearest_zero",
        "7b239607fcf4299166fc92e5300ea888e7224f8a7ed32d7e5265cb78cc62657e",
    ),
    (
        "rgba16f_slot_nearest_border",
        "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
    ),
    (
        "rgba16f_arg_nearest_border",
        "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
    ),
    (
        "rgba16f_norm_nearest_border",
        "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
    ),
    (
        "rgba16f_const_nearest_border",
        "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
    ),
    (
        "rgba16f_slot_nearest_mirror",
        "fb1a84e8f744e0bd92aab91af807fe789fdf105b5ae8bd526fb427ef0d04409d",
    ),
    (
        "rgba16f_arg_nearest_mirror",
        "fb1a84e8f744e0bd92aab91af807fe789fdf105b5ae8bd526fb427ef0d04409d",
    ),
    (
        "rgba16f_norm_nearest_mirror",
        "fb1a84e8f744e0bd92aab91af807fe789fdf105b5ae8bd526fb427ef0d04409d",
    ),
    (
        "rgba16f_slot_linear_edge",
        "7a6a5f64bc4531517ba0d4d4912c2d687ee0779777af9376c02b5e09f4af2421",
    ),
    (
        "rgba16f_arg_linear_edge",
        "7a6a5f64bc4531517ba0d4d4912c2d687ee0779777af9376c02b5e09f4af2421",
    ),
    (
        "rgba16f_norm_linear_edge",
        "7a6a5f64bc4531517ba0d4d4912c2d687ee0779777af9376c02b5e09f4af2421",
    ),
    (
        "rgba16f_const_linear_edge",
        "7a6a5f64bc4531517ba0d4d4912c2d687ee0779777af9376c02b5e09f4af2421",
    ),
    (
        "rgba16f_slot_linear_zero",
        "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
    ),
    (
        "rgba16f_arg_linear_zero",
        "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
    ),
    (
        "rgba16f_norm_linear_zero",
        "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
    ),
    (
        "rgba16f_const_linear_zero",
        "dbb87cfd8c4ab3c54dede294e5ebc42ad0a2fd84166cc1b46a33e3a3b61e1777",
    ),
    (
        "rgba16f_slot_linear_border",
        "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
    ),
    (
        "rgba16f_arg_linear_border",
        "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
    ),
    (
        "rgba16f_norm_linear_border",
        "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
    ),
    (
        "rgba16f_const_linear_border",
        "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
    ),
    (
        "rgba16f_slot_linear_mirror",
        "24724e535ae68eaccd7460b0c8948ce18d3118bb92395b690d7c8918dc521e06",
    ),
    (
        "rgba16f_arg_linear_mirror",
        "24724e535ae68eaccd7460b0c8948ce18d3118bb92395b690d7c8918dc521e06",
    ),
    (
        "rgba16f_norm_linear_mirror",
        "24724e535ae68eaccd7460b0c8948ce18d3118bb92395b690d7c8918dc521e06",
    ),
];

/// The probe's coordinates, `x` fastest: per integer `i` from -1 to 6 the texel edge `i` and one
/// ulp either side (`2^-24` at 0), `i + 1/512`, `i + 0.25`, the centre `i + 0.5` and one ulp
/// either side, `i + 0.75` and `i + 255/256`; `y` from a list of ten across and past the 3 rows.
/// The SHA-256 of [`grid`]'s coordinates, `x` then `y` little-endian.
const GRID_SHA: &str = "10064f01ccb480a13ee9b0853ba41bbc5e2dfa3d490ffaef9cbcc987f75249b9";

/// The inputs are the probe's: its coordinates, before any configuration claims Metal's samples
/// at them.
#[test]
fn the_inputs_are_the_probes() {
    let mut sha = sha2::Sha256::new();
    for cell in grid() {
        sha.update(cell[0].to_le_bytes());
        sha.update(cell[1].to_le_bytes());
    }
    assert_eq!(crate::hash::hex_encode(sha.finalize()), GRID_SHA);
}

fn grid() -> Vec<[f32; 2]> {
    let mut xs = Vec::new();
    for i in -1..=(W as i32 + 1) {
        let (b, h) = (i as f32, i as f32 + 0.5);
        let (lo, hi) = match i {
            0 => (-(2f32.powi(-24)), 2f32.powi(-24)),
            _ => (b.next_down(), b.next_up()),
        };
        xs.extend([
            lo,
            b,
            hi,
            b + 2f32.powi(-9),
            b + 0.25,
            h.next_down(),
            h,
            h.next_up(),
            b + 0.75,
            b + 255.0 / 256.0,
        ]);
    }
    let ys = [
        -0.5,
        0.0,
        0.5,
        1f32.next_down(),
        1.0,
        1.25,
        1.5f32.next_up(),
        2.5,
        3.0,
        3.25,
    ];
    ys.iter()
        .flat_map(|&y| xs.iter().map(move |&x| [x, y]))
        .collect()
}

/// The texture of `format`, its texel `(x, y)` channel `c` the probe's.
fn texture(device: &Device, format: PixelFormat) -> Texture {
    let texture = device
        .new_texture(&TextureDescriptor::texture_2d(format, W, H, false))
        .unwrap();
    let mut bytes = Vec::new();
    for y in 0..H {
        for x in 0..W {
            for c in 0..4 {
                match format {
                    PixelFormat::RGBA8Unorm => {
                        bytes.push(((x * 53 + y * 97 + c * 29 + 11) % 256) as u8)
                    }
                    _ => {
                        let k = ((x * 7 + y * 13 + c * 5 + 3) % 29) as i32 - 14;
                        bytes.extend(half_bits(k as f32 / 4.0).to_le_bytes());
                    }
                }
            }
        }
    }
    let row = W * if format == PixelFormat::RGBA8Unorm {
        4
    } else {
        8
    };
    texture
        .replace_region(Region::new_2d(0, 0, W, H), 0, 0, &bytes, row, 0)
        .unwrap();
    texture
}

/// An address mode: the probe's name, the runtime's mode, and a constexpr sampler's code and
/// border color (none for mirror clamp to edge, which a constexpr sampler cannot state).
type Address = (&'static str, SamplerAddressMode, Option<(u64, u64)>);

/// The address modes sampled.
const ADDRESSES: [Address; 4] = [
    ("edge", SamplerAddressMode::ClampToEdge, Some((1, 0))),
    ("zero", SamplerAddressMode::ClampToZero, Some((0, 0))),
    (
        "border",
        SamplerAddressMode::ClampToBorderColor,
        Some((0, 2)),
    ),
    ("mirror", SamplerAddressMode::MirrorClampToEdge, None),
];

/// A runtime sampler state of `filter` and `address` on every axis, an opaque white border, and
/// argument-buffer support, of pixel coordinates unless `normalized`.
fn sampler(
    device: &Device,
    filter: SamplerMinMagFilter,
    address: SamplerAddressMode,
    normalized: bool,
) -> SamplerState {
    let mut d = SamplerDescriptor::new();
    d.min_filter = filter;
    d.mag_filter = filter;
    d.s_address_mode = address;
    d.t_address_mode = address;
    d.r_address_mode = address;
    d.border_color = SamplerBorderColor::OpaqueWhite;
    d.normalized_coordinates = normalized;
    let mut options = SamplerStateOptions::default();
    options.support_argument_buffers = true;
    device.new_sampler_state_with_options(&d, options).unwrap()
}

/// A constexpr `coord::pixel` sampler's words: address `code` on every axis, `linear` filters or
/// nearest, no mip filter, no comparison, the frontend's LOD clamps, border color `border`.
fn pixel_words(code: u64, linear: bool, border: u64) -> u64 {
    let filters = if linear { 1 << 9 | 1 << 11 } else { 0 };
    (code | code << 3 | code << 6) | filters | 1 << 15 | 8 << 16 | 0x7bff << 40 | border << 56
}

/// The widest shift [`compare`] searches for a pixel configuration's cell, in ulps of the
/// normalized coordinate on each axis.
pub(super) const REACH: i32 = 4;

/// `v` moved by `k` ulps, up where `k` is positive.
pub(super) fn shifted(mut v: f32, k: i32) -> f32 {
    for _ in 0..k.unsigned_abs() {
        v = if k > 0 { v.next_up() } else { v.next_down() };
    }
    v
}

/// The normalized coordinate of each cell at `cells`, the probe's divided by the size.
fn normalized(cells: &[[f32; 2]]) -> Vec<[f32; 4]> {
    cells
        .iter()
        .map(|&[x, y]| [x / W as f32, y / H as f32, 0.0, 0.0])
        .collect()
}

/// A configuration's results beside Metal's and beside its normalized control.
struct Outcome {
    /// The configuration's name (the module docs).
    name: String,
    /// Whether its results hash as Metal's.
    metal: bool,
    /// Whether it is a `norm` control.
    control: bool,
    /// Its cells that differ from its control at the same coordinate (none for a control).
    differ: usize,
    /// The widest shift, in ulps on either axis, of the control's coordinate at which the control
    /// answers each such cell as the configuration does; `None` where one is not the control's
    /// within [`REACH`].
    reach: Option<u32>,
    /// Its hash where it is not Metal's, then its first cells that differ from its control.
    differs: Vec<String>,
}

/// Every configuration's results on `device` beside Metal's and beside its control.
fn compare(device: &Device) -> Vec<Outcome> {
    let make = |heap: bool, through: Through| {
        pipeline_of(
            device,
            &llvm_assemble_text(&kernel(heap, through, false, false)).unwrap(),
            "k",
        )
    };
    let (slot, argument) = (make(false, Through::Runtime), make(true, Through::Argument));
    let metal: BTreeMap<&str, &str> = METAL.into_iter().collect();
    let cells = grid();
    let pixel: Vec<[f32; 4]> = cells.iter().map(|&[x, y]| [x, y, 0.0, 0.0]).collect();
    let n = cells.len();
    let run = |texture: &Texture,
               held: Held<'_>,
               sampler: Option<&SamplerState>,
               state: &ComputePipelineState,
               jobs: &[[f32; 4]]| {
        let mut out = run_held(device, (texture, held, sampler), state, jobs, 1);
        out.truncate(4 * jobs.len());
        out
    };
    let texel =
        |out: &[f32], j: usize| -> [u32; 4] { std::array::from_fn(|c| out[4 * j + c].to_bits()) };
    let mut outcomes = Vec::new();
    for (format, fname) in [
        (PixelFormat::RGBA8Unorm, "rgba8"),
        (PixelFormat::RGBA16Float, "rgba16f"),
    ] {
        let texture = texture(device, format);
        for (filter, lname) in [
            (SamplerMinMagFilter::Nearest, "nearest"),
            (SamplerMinMagFilter::Linear, "linear"),
        ] {
            for &(aname, address, constexpr) in &ADDRESSES {
                let pixel_state = sampler(device, filter, address, false);
                let norm_state = sampler(device, filter, address, true);
                // A configuration's control at `jobs`: the normalized state through the argument
                // buffer, or a constexpr sampler of `words` in normalized coordinates.
                let control = |words: Option<u64>, jobs: &[[f32; 4]]| match words {
                    None => run(
                        &texture,
                        Held::Argument(Some(&norm_state)),
                        None,
                        &argument,
                        jobs,
                    ),
                    Some(words) => run(
                        &texture,
                        Held::Slot,
                        None,
                        &make(false, Through::Constexpr(words & !(1 << 15))),
                        jobs,
                    ),
                };
                let norm = control(None, &normalized(&cells));
                let mut results = vec![
                    ("norm", None, None),
                    (
                        "slot",
                        Some(run(&texture, Held::Slot, Some(&pixel_state), &slot, &pixel)),
                        None,
                    ),
                    (
                        "arg",
                        Some(run(
                            &texture,
                            Held::Argument(Some(&pixel_state)),
                            None,
                            &argument,
                            &pixel,
                        )),
                        None,
                    ),
                ];
                if let Some((code, border)) = constexpr {
                    let words = pixel_words(code, filter == SamplerMinMagFilter::Linear, border);
                    let state = make(false, Through::Constexpr(words));
                    let out = run(&texture, Held::Slot, None, &state, &pixel);
                    results.push(("const", Some(out), Some(words)));
                }
                for (source, out, words) in results {
                    let name = format!("{fname}_{source}_{lname}_{aname}");
                    let out = out.as_ref().unwrap_or(&norm);
                    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let hash = crate::hash::hex_encode(Sha256::digest(&bytes));
                    let is_metal = metal.get(name.as_str()) == Some(&hash.as_str());
                    let mut differs = Vec::new();
                    if !is_metal {
                        differs.push(format!("hash {hash}"));
                    }
                    if source == "norm" {
                        outcomes.push(Outcome {
                            name,
                            metal: is_metal,
                            control: true,
                            differ: 0,
                            reach: Some(0),
                            differs,
                        });
                        continue;
                    }
                    let own = match words {
                        None => norm.clone(),
                        Some(words) => control(Some(words), &normalized(&cells)),
                    };
                    let wrong: Vec<usize> = (0..n)
                        .filter(|&j| texel(out, j) != texel(&own, j))
                        .collect();
                    // Each such cell's control at every shift of its coordinate within `REACH`.
                    let shifts: Vec<(i32, i32)> = (-REACH..=REACH)
                        .flat_map(|kx| (-REACH..=REACH).map(move |ky| (kx, ky)))
                        .collect();
                    let jobs: Vec<[f32; 4]> = wrong
                        .iter()
                        .flat_map(|&j| {
                            let [u, v, ..] = normalized(&cells[j..=j])[0];
                            shifts
                                .iter()
                                .map(move |&(kx, ky)| [shifted(u, kx), shifted(v, ky), 0.0, 0.0])
                        })
                        .collect();
                    let around = match jobs.is_empty() {
                        true => Vec::new(),
                        false => control(words, &jobs),
                    };
                    let mut reach = Some(0);
                    for (w, &j) in wrong.iter().enumerate() {
                        let widest = shifts
                            .iter()
                            .enumerate()
                            .filter(|&(s, _)| texel(&around, w * shifts.len() + s) == texel(out, j))
                            .map(|(_, &(kx, ky))| kx.unsigned_abs().max(ky.unsigned_abs()))
                            .min();
                        reach = reach.zip(widest).map(|(a, b)| a.max(b));
                        if differs.len() < 7 {
                            let [x, y] = cells[j];
                            differs.push(format!(
                                "({x:e}, {y:e}): {:?}, the control {:?}, matched at a shift of {widest:?}",
                                &out[4 * j..4 * j + 4],
                                &own[4 * j..4 * j + 4]
                            ));
                        }
                    }
                    outcomes.push(Outcome {
                        name,
                        metal: is_metal,
                        control: false,
                        differ: wrong.len(),
                        reach,
                        differs,
                    });
                }
            }
        }
    }
    outcomes
}

/// Every device announces both pixel samplers' notes, and every pixel configuration's cell is its
/// normalized control's within [`REACH`] ulps of the control's coordinate, but for a `const` linear
/// one where the device's sampler is not Metal's and one whose results are Metal's (the module
/// docs). On a
/// device with Metal's own sampler each `norm` control, each `const` configuration and each
/// nearest configuration answers Metal's results; elsewhere the samples are the device's sampler's
/// and the device announces `device-sampling`. Each configuration that is not Metal's is printed
/// with the cells where it differs from its control.
#[test]
fn pixel_samplers_answer_metal_s_texels() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let (outcomes, metal_sampler, name) = {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        eprintln!("pixel samplers on {}", device.name());
        (
            compare(&device),
            metal_sampler(&device),
            device.name().to_owned(),
        )
    };
    layer_reported_nothing_at_all(&messages);
    let announced: BTreeSet<Degradation> = counts(&messages.take_notices()).into_keys().collect();
    eprintln!("announced: {announced:?}");
    for kind in [
        Degradation::PixelSamplerScaling,
        Degradation::ArgumentPixelSamplerScaling,
    ] {
        assert!(announced.contains(&kind), "{name}: {kind:?} not announced");
    }
    assert_eq!(
        announced.contains(&Degradation::DeviceSampling),
        !metal_sampler,
        "{name}: device-sampling exactly where the device's sampler is not Metal's"
    );
    assert_eq!(outcomes.len(), METAL.len());
    for o in outcomes.iter().filter(|o| !o.control) {
        eprintln!(
            "PIXEL-SAMPLER-REACH {name}: {}: {} of {} cells differ from the control, widest \
             shift {:?} ulp",
            o.name,
            o.differ,
            grid().len(),
            o.reach
        );
    }
    let wrong: BTreeMap<&str, &Outcome> = outcomes
        .iter()
        .filter(|o| !o.metal)
        .map(|o| (o.name.as_str(), o))
        .collect();
    eprintln!(
        "{name}: {} of {} configurations are not Metal's",
        wrong.len(),
        METAL.len()
    );
    for (configuration, o) in &wrong {
        eprintln!(
            "PIXEL-SAMPLER-DIFFERS {name}: {configuration}: {:?}",
            o.differs
        );
    }
    // The device's own linear filter in unnormalized coordinates, where its sampler is not Metal's:
    // not its normalized filter at a moved coordinate (the module docs). A configuration whose
    // results are Metal's is held to them, whatever its control answers.
    let own_filter = |o: &Outcome| !metal_sampler && o.name.contains("_const_linear_");
    let beyond: Vec<(&str, &[String])> = outcomes
        .iter()
        .filter(|o| o.reach.is_none() && !own_filter(o) && !o.metal)
        .map(|o| (o.name.as_str(), o.differs.as_slice()))
        .collect();
    assert!(
        beyond.is_empty(),
        "{name}: cells beyond {REACH} ulps of the control: {beyond:#?}"
    );
    if metal_sampler {
        let controls: Vec<&&str> = wrong.keys().filter(|n| n.contains("_norm_")).collect();
        assert!(
            controls.is_empty(),
            "{name}: normalized controls not Metal's: {controls:?}"
        );
        let exact: Vec<&&str> = wrong
            .keys()
            .filter(|n| n.contains("_const_") || n.contains("_nearest_"))
            .collect();
        assert!(
            exact.is_empty(),
            "{name}: constexpr and nearest configurations not Metal's: {exact:?}"
        );
    }
}

/// Whether `device` samples through Metal's own sampler (`VkProperty::MetalSampler`).
pub(super) fn metal_sampler(device: &Device) -> bool {
    device
        .compile_device()
        .profile()
        .enabled()
        .properties
        .contains(&VkProperty::MetalSampler)
}

/// Whether `device` samples by its own sampler, not Metal's (`VkProperty::MetalSampler`), checking
/// that the device announced `device-sampling` among `messages`' notices exactly then.
pub(super) fn own_sampler(device: &Device, messages: &Messages) -> bool {
    let own = !metal_sampler(device);
    let announced = counts(&messages.take_notices());
    assert_eq!(
        announced.contains_key(&Degradation::DeviceSampling),
        own,
        "{}: device-sampling exactly where the device's sampler is not Metal's",
        device.name()
    );
    own
}

/// `argument-pixel-sampler-scaling` fires at a dispatch, never at a state's making. An argument
/// buffer's sample through the sampler it holds, whose stage records the note on every device
/// (which state a program's argument buffer names, the module cannot see), dispatched while the
/// device holds only a normalized state, announces nothing; making a pixel state of nearest
/// filters announces nothing; the next dispatch, through the normalized state still, announces it
/// once.
#[test]
fn the_argument_buffer_note_waits_for_a_pixel_state() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let kind = Degradation::ArgumentPixelSamplerScaling;
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let bitcode = llvm_assemble_text(&kernel(true, Through::Argument, false, false)).unwrap();
        assert!(
            stage_notes(&device, &bitcode, "k")
                .contains(&PrecisionNote::ArgumentPixelSamplerScaling),
            "an argument buffer's sample records the note"
        );
        let argument = pipeline_of(&device, &bitcode, "k");
        let texture = texture(&device, PixelFormat::RGBA8Unorm);
        let jobs = normalized(&grid());
        let (nearest, edge) = (
            SamplerMinMagFilter::Nearest,
            SamplerAddressMode::ClampToEdge,
        );
        let normalized = sampler(&device, nearest, edge, true);
        let dispatch = || {
            run_held(
                &device,
                (&texture, Held::Argument(Some(&normalized)), None),
                &argument,
                &jobs,
                1,
            );
        };
        dispatch();
        let before = counts(&messages.take_notices());
        assert!(!before.contains_key(&kind), "{before:?}");
        let pixel = sampler(&device, nearest, edge, false);
        let made = counts(&messages.take_notices());
        assert!(!made.contains_key(&kind), "{made:?}");
        dispatch();
        let after = counts(&messages.take_notices());
        assert_eq!(after.get(&kind).copied(), Some(1), "{after:?}");
        drop(pixel);
    }
    layer_reported_nothing_at_all(&messages);
}

/// `device-denormals` of a nearest pixel sample through a runtime sampler slot, whose texel's
/// centre `floor(x) + 0.5` reads a negative subnormal coordinate by the device's `floor`, waits for
/// a dispatch through a state whose `s` or `t` mode reads texel -1 apart from texel 0: its stage
/// records the note held under the bound word (`SpecKey::BoundSamplerWord`), the pipeline's making
/// announces nothing, dispatches at subnormal coordinates through clamp to edge and mirror clamp to
/// edge announce nothing, and the next, through clamp to zero, announces it once.
#[test]
fn a_texel_centre_s_denormals_wait_for_a_mode_that_parts_texel_minus_one() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let kind = Degradation::DeviceDenormals;
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let bitcode = llvm_assemble_text(&kernel(false, Through::Runtime, false, false)).unwrap();
        assert!(
            stage_notes(&device, &bitcode, "k").contains(&PrecisionNote::DeviceDenormals),
            "a runtime slot's nearest sample records the note"
        );
        let slot = pipeline_of(&device, &bitcode, "k");
        let made = counts(&messages.take_notices());
        assert!(!made.contains_key(&kind), "made: {made:?}");
        let texture = texture(&device, PixelFormat::RGBA8Unorm);
        let below = -f32::from_bits(1);
        let jobs = [[below, 1.5, 0.0, 0.0], [2.5, below, 0.0, 0.0]];
        for (address, noted) in [
            (SamplerAddressMode::ClampToEdge, false),
            (SamplerAddressMode::MirrorClampToEdge, false),
            (SamplerAddressMode::ClampToZero, true),
        ] {
            let state = sampler(&device, SamplerMinMagFilter::Nearest, address, false);
            run_held(
                &device,
                (&texture, Held::Slot, Some(&state)),
                &slot,
                &jobs,
                1,
            );
            let after = counts(&messages.take_notices());
            eprintln!(
                "TEXEL-CENTRE-DENORMALS {}: {address:?}: {after:?}",
                device.name()
            );
            assert_eq!(
                after.get(&kind).copied(),
                noted.then_some(1),
                "{address:?}: {after:?}"
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}
