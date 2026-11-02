//! Pixel-coordinate samples and gathers at texel edges of sizes that are not powers of two (the
//! consumer's report of 2026-11-02: a nearest pixel sampler clamping to zero or to a border read
//! the texel before an edge wherever `x / N` is inexact), on the device the cfg selects, under the
//! validation layer, against plain Metal's answers.
//!
//! The owned fixture `px_edges` reads a `texture2d<float>` of `RGBA32Float` (a 1D texture and a
//! 2D array's layer 1 apart) at the coordinate each thread is given, through each way a pixel
//! sampler reaches a texture: constexpr `coord::pixel` samplers of each filter, clamping to the
//! edge, to zero and to an opaque black and an opaque white border, without an offset, with a
//! constant one and with a dynamic one (buffer 2, `(1, -2)`); the runtime sampler at slot 0
//! (states of nearest and linear filters, each of those address modes, `normalizedCoordinates`
//! NO), without an offset and with `(-1, 2)`; and the same states in an argument buffer beside the
//! texture. Texel `(x, y)` of layer `l` holds `(x + 2, y + 2, 1000 + x + 100 y + 10000 l, 0.25)`,
//! so a result names the texel it read. The sizes ([`SIZES`]) are 258x5, 21x13 and 37x3, and the
//! coordinates ([`coordinates`]) sweep each axis from two texels before the texture to two past
//! it: at each texel edge `i` and one ulp either side (`2^-24` at 0), `i + 0.25`, the centre and
//! one ulp either side, `i + 0.5 + 1/512` and `+ 3/512`, `i + 0.75` and `i + 255/256`, the other
//! axis at `floor(N / 2) + 0.25`. The 2D textures are mipmapped: a pixel sampler reads level 0
//! (Metal on an Apple M3 Max under macOS 27 answered mipmapped and single-level textures alike).
//!
//! - Every nearest access answers Metal's texel, `floor(x) + k` on each axis, its address mode's
//!   when outside: the edge's texel, a constexpr sampler's zero or border colour, and a runtime
//!   state's border colour for clamp to zero as for clamp to the border (opaque white here, but
//!   the black border's state). On every device at texel edges, quarters and centres; at every
//!   coordinate on a device that samples through Metal's own sampler (`VkProperty::MetalSampler`),
//!   elsewhere a cell off the rule is printed (`PIXEL-EDGE-DEVICE` lines), the device's
//!   unnormalized sampler's own reading of `x` (`PrecisionNote::DeviceSampling`).
//! - A constexpr pixel sampler's linear sample and gather of a 1D or 2D texture hands `x` itself
//!   to the device's unnormalized sampler: on a device with `VkProperty::MetalSampler` each
//!   kernel's results over the three sizes hash as Metal's ([`METAL`], the M3 Max), and no such
//!   kernel's stage records `PrecisionNote::PixelSamplerScaling`.
//! - A linear sample or gather through a runtime or argument buffer's sampler, or of an array, is
//!   the device's normalized sampler's at `x / N` (`PixelSamplerScaling` and
//!   `ArgumentPixelSamplerScaling`), which the stages of an array's linear sample and of the
//!   runtime slot's sample and gather record; `runtime_lane::pixel_samplers` holds such accesses
//!   to their normalized controls.

use super::depth_compares::pipeline_of;
use super::driver_deviations::stage_notes;
use super::pixel_samplers::metal_sampler;
use super::{
    buffer_with, layer_reported_nothing_at_all, validated_device, Messages, ScratchCache, WAIT,
};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    PixelFormat, ResourceUsage, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter,
    TextureType,
};
use metal2vulkan::pipeline::PrecisionNote;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, ComputePipelineState, Device, Region, ResourceOptions,
    SamplerState, SamplerStateOptions, Size, Texture, TextureDescriptor,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// The textures' sizes: no side a power of two.
const SIZES: [(u64, u64); 3] = [(258, 5), (21, 13), (37, 3)];

/// The SHA-256 of every size's [`coordinates`], `x` then `y` little-endian, in [`SIZES`]' order.
const COORDINATES_SHA: &str = "dd7886bab257a5fdc3be1e3769866298aff0fda663e2748d04c86609edbcccdc";

/// Plain Metal's answers (an Apple M3 Max under macOS 27): each constexpr linear sample's and
/// gather's `float4` results over the three sizes, in [`SIZES`]' order, hashed.
const METAL: [(&str, &str); 11] = [
    (
        "pe_linear_edge",
        "eb304d1814deb63c57685ffe31b1fbfd9b18e9e3ee2b642e776f8d8377fb7c2b",
    ),
    (
        "pe_linear_zero",
        "1bfab701932acecc12c4bec868e945d6d74723b06ad2866fb3211259b5667e30",
    ),
    (
        "pe_linear_black",
        "017c5e3df5903218e6ecb7b6f2539a42869c0748123760d3a7f679eb8317c2ec",
    ),
    (
        "pe_linear_white",
        "9b2902702a46057b070cd1945aed956e783a0079b8671eee66945b57b432e0b4",
    ),
    (
        "pe_linear_zero_offset",
        "15831a884a154274d84cd5aea592d5b94b5a1f77dd49a949540c162ae47c93e5",
    ),
    (
        "pe_linear_black_dynamic",
        "083abb9caf9838307e72b9e069eac0d4c2f60134e8ca0486c208fa90cee033db",
    ),
    (
        "pe_gather_zero",
        "d5cf30f12fcd57441c16dd7b3d1c02ea6f01adebeefe11f2ceba5aad41f74753",
    ),
    (
        "pe_gather_black",
        "c6c5b438e3e9748fd85c07ded99caf60ff77a9061fc3d81af7a2943850e49a20",
    ),
    (
        "pe_gather_white_offset",
        "84c038eccf36600db93fb19022cfa123d34394dd3c4b9551ec66c9d23f724780",
    ),
    (
        "pe_gather_edge_dynamic",
        "da2d57bc456c98ff1670002bff818fd1ec8b721e80251083d93b916bf6eded53",
    ),
    (
        "pe_linear_zero_1d",
        "82d5651673d3e9d3e76ce153b4fe2896b6cb85db41375789a2e91a7dd55fba66",
    ),
];

/// The dynamic offset the kernels read (buffer 2).
const K: [i32; 2] = [1, -2];

/// The positions on one axis around integer `i` (the module docs), in order; the rule holds on
/// every device at those [`STRICT`] names.
fn positions(i: i32) -> [f32; 11] {
    let (b, h) = (i as f32, i as f32 + 0.5);
    let (lo, hi) = match i {
        0 => (-(2f32.powi(-24)), 2f32.powi(-24)),
        _ => (b.next_down(), b.next_up()),
    };
    [
        lo,
        b,
        hi,
        b + 0.25,
        h.next_down(),
        h,
        h.next_up(),
        h + 1.0 / 512.0,
        h + 3.0 / 512.0,
        b + 0.75,
        b + 255.0 / 256.0,
    ]
}

/// The indices into [`positions`] of the edge, the quarter, the centre and the three quarters.
const STRICT: [usize; 4] = [1, 3, 5, 9];

/// The coordinates of a `w` by `h` texture: along `x` then along `y` (the module docs), with
/// whether each is at a [`STRICT`] position.
fn coordinates(w: u64, h: u64) -> Vec<([f32; 2], bool)> {
    let mut out = Vec::new();
    for axis in 0..2 {
        let n = if axis == 0 { w } else { h } as i32;
        let other = if axis == 0 { h / 2 } else { w / 2 } as f32 + 0.25;
        for i in -2..=n + 2 {
            for (at, x) in positions(i).into_iter().enumerate() {
                let p = if axis == 0 { [x, other] } else { [other, x] };
                out.push((p, STRICT.contains(&at)));
            }
        }
    }
    out
}

/// The inputs are the probe's: its coordinates, before any configuration claims Metal's answers
/// at them.
#[test]
fn the_inputs_are_the_probes() {
    let mut sha = Sha256::new();
    for (w, h) in SIZES {
        for ([x, y], _) in coordinates(w, h) {
            sha.update(x.to_le_bytes());
            sha.update(y.to_le_bytes());
        }
    }
    assert_eq!(crate::hash::hex_encode(sha.finalize()), COORDINATES_SHA);
}

/// Texel `(x, y)` of layer `layer` (the module docs).
fn texel(x: u64, y: u64, layer: u64) -> [f32; 4] {
    [
        x as f32 + 2.0,
        y as f32 + 2.0,
        (1000 + x + 100 * y + 10000 * layer) as f32,
        0.25,
    ]
}

/// Where a nearest access outside the texture reads: the edge's texel, or a colour.
#[derive(Clone, Copy, Debug)]
enum Outside {
    Edge,
    Colour([f32; 4]),
}

const TRANSPARENT: Outside = Outside::Colour([0.0; 4]);
const BLACK: Outside = Outside::Colour([0.0, 0.0, 0.0, 1.0]);
const WHITE: Outside = Outside::Colour([1.0; 4]);

/// Metal's nearest answer at `p` on a `w` by `h` texture's layer `layer`: texel `floor(p) + k`,
/// or `outside`'s where that lies outside.
fn nearest(p: [f32; 2], k: [i32; 2], (w, h): (u64, u64), outside: Outside, layer: u64) -> [f32; 4] {
    let at = |v: f32, k: i32| v.floor() as i64 + i64::from(k);
    let (i, j) = (at(p[0], k[0]), at(p[1], k[1]));
    let within = (0..w as i64).contains(&i) && (0..h as i64).contains(&j);
    match outside {
        _ if within => texel(i as u64, j as u64, layer),
        Outside::Edge => texel(
            i.clamp(0, w as i64 - 1) as u64,
            j.clamp(0, h as i64 - 1) as u64,
            layer,
        ),
        Outside::Colour(c) => c,
    }
}

/// A runtime address mode: its name, the mode, its border colour, and where its nearest access
/// outside the texture reads.
type Address = (
    &'static str,
    SamplerAddressMode,
    SamplerBorderColor,
    Outside,
);

/// The runtime states' address modes.
const ADDRESSES: [Address; 4] = [
    (
        "edge",
        SamplerAddressMode::ClampToEdge,
        SamplerBorderColor::OpaqueWhite,
        Outside::Edge,
    ),
    (
        "zero",
        SamplerAddressMode::ClampToZero,
        SamplerBorderColor::OpaqueWhite,
        WHITE,
    ),
    (
        "black",
        SamplerAddressMode::ClampToBorderColor,
        SamplerBorderColor::OpaqueBlack,
        BLACK,
    ),
    (
        "white",
        SamplerAddressMode::ClampToBorderColor,
        SamplerBorderColor::OpaqueWhite,
        WHITE,
    ),
];

/// A runtime pixel sampler state of `filter` and `address`, with argument-buffer support.
fn state(device: &Device, filter: SamplerMinMagFilter, address: &Address) -> SamplerState {
    let mut d = SamplerDescriptor::new();
    d.min_filter = filter;
    d.mag_filter = filter;
    d.s_address_mode = address.1;
    d.t_address_mode = address.1;
    d.r_address_mode = address.1;
    d.border_color = address.2;
    d.normalized_coordinates = false;
    let mut options = SamplerStateOptions::default();
    options.support_argument_buffers = true;
    device.new_sampler_state_with_options(&d, options).unwrap()
}

/// The textures of one size: the mipmapped 2D texture, the 1D texture (its row 0), and the 2D
/// array of two layers, level 0 of each holding [`texel`]s.
struct Textures {
    plane: Texture,
    line: Texture,
    array: Texture,
}

fn textures(device: &Device, (w, h): (u64, u64)) -> Textures {
    let layer = |l: u64| -> Vec<u8> {
        (0..h)
            .flat_map(|y| (0..w).flat_map(move |x| texel(x, y, l)))
            .flat_map(f32::to_le_bytes)
            .collect()
    };
    let plane = device
        .new_texture(&TextureDescriptor::texture_2d(
            PixelFormat::RGBA32Float,
            w,
            h,
            true,
        ))
        .unwrap();
    plane
        .replace_region(Region::new_2d(0, 0, w, h), 0, 0, &layer(0), w * 16, 0)
        .unwrap();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, w, 1, false);
    d.texture_type = TextureType::Type1D;
    let line = device.new_texture(&d).unwrap();
    line.replace_region(
        Region::new_2d(0, 0, w, 1),
        0,
        0,
        &layer(0)[..(w * 16) as usize],
        w * 16,
        0,
    )
    .unwrap();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, w, h, true);
    d.texture_type = TextureType::Type2DArray;
    d.array_length = 2;
    let array = device.new_texture(&d).unwrap();
    for l in 0..2 {
        array
            .replace_region(
                Region::new_2d(0, 0, w, h),
                0,
                l,
                &layer(l),
                w * 16,
                w * h * 16,
            )
            .unwrap();
    }
    Textures { plane, line, array }
}

/// How a dispatch reaches its texture and sampler.
#[derive(Clone, Copy)]
enum Bound<'a> {
    /// Texture slot 0, and the runtime sampler at slot 0 where there is one.
    Slots(&'a Texture, Option<&'a SamplerState>),
    /// An argument buffer at buffer 3 holding the texture and the sampler.
    Argument(&'a Texture, &'a SamplerState),
}

/// The `float4` results of `state` at `uv`'s coordinates through `bound`.
fn dispatch(
    device: &Device,
    state: &ComputePipelineState,
    (uv, k): (&Buffer, &Buffer),
    n: usize,
    bound: Bound<'_>,
) -> Vec<[f32; 4]> {
    let out = device
        .new_buffer(n as u64 * 16, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(uv), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    encoder.set_buffer(Some(k), 0, 2).unwrap();
    let _held = match bound {
        Bound::Slots(texture, sampler) => {
            encoder.set_texture(Some(texture), 0).unwrap();
            if sampler.is_some() {
                encoder.set_sampler_state(sampler, 0).unwrap();
            }
            None
        }
        Bound::Argument(texture, sampler) => {
            let ids: Vec<u8> = [texture.gpu_resource_id().0, sampler.gpu_resource_id().0]
                .iter()
                .flat_map(|id| id.to_le_bytes())
                .collect();
            let table = buffer_with(device, &ids).unwrap();
            encoder
                .use_resource(texture.as_resource(), ResourceUsage::READ)
                .unwrap();
            encoder.set_buffer(Some(&table), 0, 3).unwrap();
            Some(table)
        }
    };
    encoder
        .dispatch_threads(Size::new(n as u64, 1, 1), Size::new(64, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        command_buffer.error()
    );
    let contents = out.contents().unwrap();
    // SAFETY: the buffer holds `n` `float4`s, and no command buffer writes it any more.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), n * 16) };
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| {
            std::array::from_fn(|i| f32::from_le_bytes(c[4 * i..4 * i + 4].try_into().unwrap()))
        })
        .collect()
}

/// A nearest configuration: its name, the kernel, how it is bound (`None`: the constexpr
/// sampler's slots; else the runtime state's address mode, through an argument buffer where
/// `true`), its offset, where it reads outside, and the layer it reads.
struct Nearest {
    name: String,
    kernel: &'static str,
    state: Option<(usize, bool)>,
    k: [i32; 2],
    outside: Outside,
    layer: u64,
}

/// Every nearest configuration of the fixture.
fn nearest_configurations() -> Vec<Nearest> {
    let constexpr = |kernel: &'static str, k, outside, layer| Nearest {
        name: kernel.to_owned(),
        kernel,
        state: None,
        k,
        outside,
        layer,
    };
    let mut out = vec![
        constexpr("pe_nearest_zero", [0, 0], TRANSPARENT, 0),
        constexpr("pe_nearest_black", [0, 0], BLACK, 0),
        constexpr("pe_nearest_white", [0, 0], WHITE, 0),
        constexpr("pe_nearest_zero_offset", [1, -1], TRANSPARENT, 0),
        constexpr("pe_nearest_white_dynamic", K, WHITE, 0),
        constexpr("pe_nearest_edge_offset", [-2, 3], Outside::Edge, 0),
        constexpr("pe_nearest_white_array", [0, 0], WHITE, 1),
    ];
    for (kernel, argument, k) in [
        ("pe_slot_sample", false, [0, 0]),
        ("pe_slot_sample_offset", false, [-1, 2]),
        ("pe_argument_sample", true, [0, 0]),
        ("pe_argument_sample_offset", true, [-1, 2]),
    ] {
        for (a, address) in ADDRESSES.iter().enumerate() {
            out.push(Nearest {
                name: format!("{kernel}/nearest/{}", address.0),
                kernel,
                state: Some((a, argument)),
                k,
                outside: address.3,
                layer: 0,
            });
        }
    }
    out
}

/// Every nearest access answers Metal's texel at texel edges, quarters and centres, and on a device
/// with Metal's own sampler at every coordinate; the constexpr linear samples and gathers record no
/// `PixelSamplerScaling` and, on such a device, answer Metal's results (the module docs).
#[test]
fn pixel_edges_answer_metal_s_texels() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let exact = metal_sampler(&device);
        eprintln!("pixel edges on {name} (Metal's own sampler: {exact})");
        let bitcode = llvm_assemble(&engine_dir().join("px_edges.ll")).unwrap();
        // The stages: the constexpr samples and gathers of a 1D or 2D texture take `x` itself; an
        // array's linear sample and a runtime slot's are scaled.
        for (kernel, _) in METAL {
            let notes = stage_notes(&device, &bitcode, kernel);
            assert!(
                !notes.contains(&PrecisionNote::PixelSamplerScaling),
                "{name}: {kernel}: {notes:?}"
            );
        }
        for kernel in ["pe_linear_zero_array", "pe_slot_sample", "pe_slot_gather"] {
            let notes = stage_notes(&device, &bitcode, kernel);
            assert!(
                notes.contains(&PrecisionNote::PixelSamplerScaling),
                "{name}: {kernel}: {notes:?}"
            );
        }
        let configurations = nearest_configurations();
        let kernels: Vec<&str> = configurations
            .iter()
            .map(|c| c.kernel)
            .chain(METAL.iter().map(|(kernel, _)| *kernel))
            .collect();
        let pipelines: BTreeMap<&str, ComputePipelineState> = kernels
            .into_iter()
            .map(|kernel| (kernel, pipeline_of(&device, &bitcode, kernel)))
            .collect();
        let states: Vec<SamplerState> = ADDRESSES
            .iter()
            .map(|a| state(&device, SamplerMinMagFilter::Nearest, a))
            .collect();
        let k: Vec<u8> = K.iter().flat_map(|v| v.to_le_bytes()).collect();
        let k = buffer_with(&device, &k).unwrap();
        let mut constexpr: BTreeMap<&str, Vec<u8>> = BTreeMap::new();
        let mut off_rule: Vec<String> = Vec::new();
        let mut device_own = 0;
        for size in SIZES {
            let cells = coordinates(size.0, size.1);
            let n = cells.len();
            let bytes: Vec<u8> = cells
                .iter()
                .flat_map(|(p, _)| p.iter().flat_map(|v| v.to_le_bytes()))
                .collect();
            let uv = buffer_with(&device, &bytes).unwrap();
            let t = textures(&device, size);
            for c in &configurations {
                let texture = if c.layer == 1 { &t.array } else { &t.plane };
                let bound = match c.state {
                    None => Bound::Slots(texture, None),
                    Some((a, false)) => Bound::Slots(texture, Some(&states[a])),
                    Some((a, true)) => Bound::Argument(texture, &states[a]),
                };
                let out = dispatch(&device, &pipelines[c.kernel], (&uv, &k), n, bound);
                for (j, (cell, strict)) in cells.iter().enumerate() {
                    let want = nearest(*cell, c.k, size, c.outside, c.layer);
                    if out[j] == want {
                        continue;
                    }
                    let line = format!(
                        "{}x{} {}: ({:e}, {:e}): {:?}, Metal {:?}",
                        size.0, size.1, c.name, cell[0], cell[1], out[j], want
                    );
                    match *strict || exact {
                        true => off_rule.push(line),
                        false => {
                            device_own += 1;
                            eprintln!("PIXEL-EDGE-DEVICE {name}: {line}");
                        }
                    }
                }
            }
            for (kernel, _) in METAL {
                let texture = if kernel.ends_with("_1d") {
                    &t.line
                } else {
                    &t.plane
                };
                let bound = Bound::Slots(texture, None);
                let out = dispatch(&device, &pipelines[kernel], (&uv, &k), n, bound);
                let bytes = constexpr.entry(kernel).or_default();
                bytes.extend(out.iter().flatten().flat_map(|v| v.to_le_bytes()));
            }
        }
        assert!(
            off_rule.is_empty(),
            "{name}: {} nearest cells off Metal's texel: {:#?}",
            off_rule.len(),
            &off_rule[..off_rule.len().min(24)]
        );
        eprintln!("{name}: {device_own} nearest cells the device's own between texel edges");
        let mut differ = Vec::new();
        for (kernel, metal) in METAL {
            let hash = crate::hash::hex_encode(Sha256::digest(&constexpr[kernel]));
            if hash != metal {
                eprintln!("PIXEL-EDGE-UNNORMALIZED {name}: {kernel}: {hash}, Metal {metal}");
                differ.push(kernel);
            }
        }
        if exact {
            assert!(differ.is_empty(), "{name}: not Metal's: {differ:?}");
        }
    }
    layer_reported_nothing_at_all(&messages);
}
