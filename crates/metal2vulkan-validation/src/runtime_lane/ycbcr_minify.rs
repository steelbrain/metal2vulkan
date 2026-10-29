//! A 3840x2160 `YCBCR10_420_2P` texture sampled into 1440x1080 (the consumer's r25: a pass that
//! samples such a surface minified into a 1440x1080 target wrote rare ±1s in its red, green and
//! blue), on the device the cfg selects, under the validation layer, beside plain Metal's
//! answers (our own probe on an Apple M4 under macOS 27.0.1, synthetic codes).
//!
//! The surface is an `x420` one whose 10-bit codes cover `0..=1023` ([`surface_bytes`]); the
//! texture of format 505 over its plane 0 is sampled through a linear clamp-to-edge sampler
//! (normalized, not mipmapped), a runtime one at a slot and a constexpr one alike:
//! - `bandx` and `bandy`: groups of 17 consecutive `f32` coordinates around the `f32` nearest
//!   each of 2048 `1/256` steps of the position along one axis ([`band`]), where a footprint's
//!   rounding decides which side of the step a sample reads. Per group the module prints where
//!   the results switch beside where Metal's rule (`position.rs`: `X = u N` rounded with ties
//!   away from zero, `q = floor(256 X + 1/2) - 128`) and the exact product put the step;
//! - `grid`: the 1440x1080 texel centres;
//! - `draw`: a 1440x1080 `RGBA8Unorm` target drawn over by two triangles whose fragments sample
//!   at the interpolated coordinate;
//! - `read`: `read()` of every texel.
//!
//! The surface is untagged (no matrix: Metal reads it as BT.601, of the OSType's video range),
//! its planes once as CoreVideo lays them out ([`PACKED`]) and once at offsets with padded rows
//! ([`PADDED`]), over which Metal answers the same.
//!
//! Each output's hash is printed beside Metal's ([`METAL`]), which every one must equal. Metal's
//! switches follow its rule in every band group that switches (4079 of 4096; the exact product
//! misses about half), and MoltenVK on an M4 and lavapipe answered every output Metal's bytes,
//! unannounced but for lavapipe's `shader-color-conversion`. A device that samples natively
//! (not through Metal's own sampler, `VkProperty::MetalSampler`) and keeps the image the reads
//! converted filters that image's texels by its own filter (`device-sampling`) and converts the
//! draw's colours by its own conversion (`device-color-conversion`), where no hash of Metal's
//! holds: there `read` still equals Metal's, and the bands, the grid and the draw equal, bit for
//! bit, the same kernels' and draw's of an `RGBA16Float` texture of the `read()` texels
//! ([`twin_of`]), the device's filter of the same texels (NVIDIA's RTX 5080 Laptop GPU).
//!
//! A second `read` converts through the texture's refresh (`WorkCounts::ycbcr_refreshes`): the
//! planes, unchanged, are compared with the staging buffer's copy, none of their bytes copied
//! (`copy_regions_split` unchanged over [`PADDED`]'s span of 25,440,128 bytes, no multiple of
//! 4 KiB), and it reads the first one's texels. On a device whose memory is the host's the bands,
//! the grid and the draw read the planes in place (`ycbcr_in_place.rs`): they bind the planes
//! (`WorkCounts::ycbcr_planes_bound`) and convert nothing; elsewhere they bind no planes.

use super::argument_linear_samples::{kernel, LINEAR_EDGE};
use super::degrade_notices::{counts, on_device};
use super::depth_compares::{pipeline_of, run_held, Held, Through};
use super::extended_range_targets::half_bits;
use super::pixel_downscales::RENDER;
use super::pixel_samplers::metal_sampler;
use super::{buffer_with, read_buffer, ycbcr_in_place, ycbcr_textures, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    PixelFormat, PrimitiveType, SamplerAddressMode, SamplerMinMagFilter, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    ClearColor, CommandBufferStatus, Device, IoSurfacePlane, IoSurfaceYCbCr, LoadAction,
    PipelineOptions, Region, RenderPassDescriptor, ResourceOptions, SamplerState, Size,
    StoreAction, Texture, TextureDescriptor,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// The surface's size: the luma plane's rows of 16-bit codes, the chroma plane's of pairs of them,
/// half as many a row and half as many rows.
const W: u64 = 3840;
const H: u64 = 2160;
/// Where a surface's planes lie: the luma plane's and the chroma plane's offset and the bytes
/// between their rows.
#[derive(Clone, Copy)]
struct Layout {
    luma: [u64; 2],
    chroma: [u64; 2],
}

/// The luma plane at byte 0 and the chroma plane right after it, each row as long as its codes.
const PACKED: Layout = Layout {
    luma: [0, 2 * W],
    chroma: [2 * W * H, 2 * W],
};

/// Planes at offsets off any page, a gap between them, and rows padded past their codes, the
/// chroma plane's further than the luma plane's (Metal answers over an untagged
/// surface of this layout what it answers over CoreVideo's own).
const PADDED: Layout = Layout {
    luma: [20864, 7808],
    chroma: [16890368, 7936],
};
/// The target's size.
const OW: u64 = 1440;
const OH: u64 = 1080;
/// A band's groups: 64 places along the axis, 32 steps in each, 17 coordinates a step.
const GROUP: usize = 17;

/// Plain Metal's answers (an Apple M4 under macOS 27.0.1, two processes alike; the
/// runtime and the constexpr sampler alike; our probe found the same over untagged surfaces of
/// other layouts, and `read`, `read()` of every texel), by output.
const METAL: [(&str, &str); 8] = [
    (
        "read",
        "12138b4cad61c88f83cf169c7d4b36a4541f0201a53f55a109c2d1f45ec5d4ef",
    ),
    (
        "bandx sample_rt",
        "c3c7fb23a2e0967dd506a088d764b8d33cd622f645566008bebfbe1dee7974c4",
    ),
    (
        "bandx sample_cx",
        "c3c7fb23a2e0967dd506a088d764b8d33cd622f645566008bebfbe1dee7974c4",
    ),
    (
        "bandy sample_rt",
        "536bb62bdfb366a18744a10f34682339739b789ee08d08d2d6d601731baf7533",
    ),
    (
        "bandy sample_cx",
        "536bb62bdfb366a18744a10f34682339739b789ee08d08d2d6d601731baf7533",
    ),
    (
        "grid sample_rt",
        "4bc931133bc4e04810c8a6f6d36ec53daadd033f563f25daeb0b739f465b5c89",
    ),
    (
        "grid sample_cx",
        "4bc931133bc4e04810c8a6f6d36ec53daadd033f563f25daeb0b739f465b5c89",
    ),
    (
        "draw",
        "facb6fa86f409d7adae2dc225731f1f38389f85953281783998e52b9b926e98c",
    ),
];

/// `lowbias32`, the codes' hash.
fn mix(v: u32) -> u32 {
    let mut x = v;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// The surface's bytes at `layout`: luma `(x, y)`'s code `mix(i ^ 0x9e3779b9) & 1023` (`i = y W +
/// x`), and chroma pair `(cx, cy)`'s Cb `mix(j ^ 0x85ebca6b) & 1023` and Cr `mix(j ^ 0xc2b2ae35) &
/// 1023` (`j = cy W/2 + cx`), each code in the high 10 bits of a little-endian 16-bit word; every
/// other byte `0xa5`.
fn surface_bytes(layout: Layout) -> Vec<u8> {
    let word = |code: u32| (((code & 1023) as u16) << 6).to_le_bytes();
    let [luma, chroma] = [layout.luma, layout.chroma];
    let end = (luma[0] + luma[1] * H).max(chroma[0] + chroma[1] * H / 2);
    let mut bytes = vec![0xa5; end as usize];
    let mut put = |at: u64, code: u32| {
        bytes[at as usize..at as usize + 2].copy_from_slice(&word(code));
    };
    for y in 0..H {
        for x in 0..W {
            put(
                luma[0] + y * luma[1] + 2 * x,
                mix((y * W + x) as u32 ^ 0x9e37_79b9),
            );
        }
    }
    for cy in 0..H / 2 {
        for cx in 0..W / 2 {
            let j = (cy * W / 2 + cx) as u32;
            let at = chroma[0] + cy * chroma[1] + 4 * cx;
            put(at, mix(j ^ 0x85eb_ca6b));
            put(at + 2, mix(j ^ 0xc2b2_ae35));
        }
    }
    bytes
}

/// The SHA-256 of the surface's bytes at each layout, packed then padded.
const SURFACE_SHA: [&str; 2] = [
    "21f3f30e8e4b0e04e50dd27376cfe9a6a8ae2dad7fa0c46db76580fe1b1f317c",
    "c54f5ab40891acc7bd19cda7fd6aad706eaeb87ddead3cf2cd03b62c4d48b3fe",
];

/// The inputs are the probe's: the surface's bytes at both layouts, before any output claims
/// Metal's of them.
#[test]
fn the_inputs_are_the_probes() {
    for (layout, want) in [PACKED, PADDED].into_iter().zip(SURFACE_SHA) {
        assert_eq!(hash(&surface_bytes(layout)), want);
    }
}

/// The `YCBCR10_420_2P` texture over plane 0 of an untagged surface (no matrix) at `layout`.
fn texture(device: &Device, layout: Layout) -> Texture {
    let surface = buffer_with(device, &surface_bytes(layout)).unwrap();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::YCBCR104202P, W, H, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let [chroma_at, chroma_row] = layout.chroma;
    let facts = IoSurfaceYCbCr::new(
        u32::from_be_bytes(*b"x420"),
        0,
        [W, H],
        chroma_at,
        chroma_row,
    );
    let [luma_at, luma_row] = layout.luma;
    let plane = IoSurfacePlane::new(luma_at, luma_row, true).with_ycbcr(facts);
    surface.new_texture_with_iosurface_plane(&d, plane).unwrap()
}

/// `read()` of every texel of `texture`, `float4`s, rows in order (the YCbCr lane's
/// `read_texels`).
fn read_all(device: &Device, texture: &Texture) -> Vec<u8> {
    let library = device
        .new_library_with_air(&llvm_assemble_text(ycbcr_textures::KERNELS).unwrap())
        .unwrap();
    let function = library.new_function_with_name("read_texels").unwrap();
    let state = device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap();
    let out = device
        .new_buffer(16 * W * H, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder
        .dispatch_threads(Size::new(W, H, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    read_buffer(&out, 0, 16 * W * H).unwrap()
}

/// An `RGBA16Float` texture of the `float4`s `texels` (`read()` of every texel, [`read_all`]),
/// each value the `half` its image holds: the texels a natively sampling device filters where the
/// YCbCr texture keeps the image its reads converted (an `R16G16B16A16_SFLOAT` one of them).
fn twin_of(device: &Device, texels: &[u8]) -> Texture {
    let d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, W, H, false);
    let twin = device.new_texture(&d).unwrap();
    let halves: Vec<u8> = (texels.as_chunks::<4>().0.iter())
        .map(|v| half_bits(f32::from_le_bytes(*v)))
        .flat_map(u16::to_le_bytes)
        .collect();
    twin.replace_region(Region::new_2d(0, 0, W, H), 0, 0, &halves, 8 * W, 0)
        .unwrap();
    twin
}

/// The band along `axis` (0: x, 1: y): for place `m` of 64 (`7 + 59m` along x, `7 + 33m` along
/// y) and step `t` of 32, the `f32` nearest the normalized coordinate of the step `p + 1/2 +
/// (4 + 8t + 1/2) / 256` and 8 either side of it, the other axis at the fixed `(13 + 33m + 1/2 +
/// 100/256) / H` (or `(13 + 59m + ...) / W`). Each a `float4` job, the coordinate first.
fn band(axis: usize) -> Vec<[f32; 4]> {
    let (n, other_n) = if axis == 0 { (W, H) } else { (H, W) };
    let mut jobs = Vec::with_capacity(64 * 32 * GROUP);
    for m in 0..64u64 {
        let (place, fixed) = if axis == 0 {
            (7 + 59 * m, 13 + 33 * m)
        } else {
            (7 + 33 * m, 13 + 59 * m)
        };
        let other = ((fixed as f64 + 0.5 + 100.0 / 256.0) / other_n as f64) as f32;
        for t in 0..32u64 {
            let step = place as f64 + 0.5 + ((4 + 8 * t) as f64 + 0.5) / 256.0;
            let mut u = (step / n as f64) as f32;
            for _ in 0..8 {
                u = u.next_down();
            }
            for _ in 0..GROUP {
                jobs.push(if axis == 0 {
                    [u, other, 0.0, 0.0]
                } else {
                    [other, u, 0.0, 0.0]
                });
                u = u.next_up();
            }
        }
    }
    jobs
}

/// The target's texel centres, `((i + 0.5) / 1440, (j + 0.5) / 1080)` in `f32`, rows in order.
fn grid() -> Vec<[f32; 4]> {
    (0..OH)
        .flat_map(|j| {
            (0..OW).map(move |i| {
                [
                    (i as f32 + 0.5) / OW as f32,
                    (j as f32 + 0.5) / OH as f32,
                    0.0,
                    0.0,
                ]
            })
        })
        .collect()
}

/// `x` rounded to `f32` with ties away from zero.
fn ties_away(x: f64) -> f32 {
    let near = x as f32;
    if f64::from(near) == x {
        return near;
    }
    let other = if f64::from(near) < x {
        near.next_up()
    } else {
        near.next_down()
    };
    let tie = (x - f64::from(near)).abs() == (f64::from(other) - x).abs();
    if tie && other.abs() > near.abs() {
        other
    } else {
        near
    }
}

/// The `1/256` step of the linear footprint at the normalized `u` on an axis of `n` texels:
/// by Metal's rule (`exact` false: the product rounded to `f32`, ties away from zero) or of the
/// exact product.
fn step(u: f32, n: u64, exact: bool) -> i64 {
    let product = f64::from(u) * n as f64;
    let x = if exact {
        product
    } else {
        f64::from(ties_away(product))
    };
    (256.0 * x + 0.5).floor() as i64 - 128
}

/// Where a group's values first differ from its first one's (`None`: never), and whether they
/// change more than once.
fn switch<T: PartialEq>(values: &[T]) -> (Option<usize>, bool) {
    let first = values.iter().position(|v| *v != values[0]);
    let changes = values.windows(2).filter(|w| w[0] != w[1]).count();
    (first, changes > 1)
}

/// A band's switches beside the rule's and the exact product's, as lines to print.
fn analysed(jobs: &[[f32; 4]], out: &[f32], axis: usize) -> Vec<String> {
    let n = if axis == 0 { W } else { H };
    let mut against_rule: BTreeMap<i64, usize> = BTreeMap::new();
    let mut against_exact: BTreeMap<i64, usize> = BTreeMap::new();
    let (mut silent, mut several) = (0, 0);
    let mut misses = Vec::new();
    for (g, group) in jobs.chunks(GROUP).enumerate() {
        let results: Vec<[u32; 4]> = (0..GROUP)
            .map(|k| std::array::from_fn(|c| out[4 * (g * GROUP + k) + c].to_bits()))
            .collect();
        let (seen, more) = switch(&results);
        several += usize::from(more);
        let Some(seen) = seen else {
            silent += 1;
            continue;
        };
        let us: Vec<f32> = group.iter().map(|job| job[axis]).collect();
        let rule = switch(&us.iter().map(|&u| step(u, n, false)).collect::<Vec<_>>()).0;
        let exact = switch(&us.iter().map(|&u| step(u, n, true)).collect::<Vec<_>>()).0;
        let offset = |at: Option<usize>| at.map_or(i64::MAX, |at| seen as i64 - at as i64);
        *against_rule.entry(offset(rule)).or_default() += 1;
        *against_exact.entry(offset(exact)).or_default() += 1;
        if Some(seen) != rule && misses.len() < 12 {
            let u = us[seen];
            let below = us[seen - 1];
            misses.push(format!(
                "  group {g}: switches at {seen}, the rule at {rule:?}, exact at {exact:?}; u \
                 {u:e} ({:#010x}) X exact {} rounded {}; the one below X exact {} rounded {}",
                u.to_bits(),
                f64::from(u) * n as f64,
                ties_away(f64::from(u) * n as f64),
                f64::from(below) * n as f64,
                ties_away(f64::from(below) * n as f64),
            ));
        }
    }
    let mut lines = vec![
        format!(
            "  {} groups: {silent} never switch, {several} switch more than once",
            jobs.len() / GROUP
        ),
        format!("  switch minus the rule's (i64::MAX: the rule never switches): {against_rule:?}"),
        format!("  switch minus the exact product's: {against_exact:?}"),
    ];
    lines.extend(misses);
    lines
}

/// A linear clamp-to-edge sampler of normalized coordinates, not mipmapped.
fn sampler(device: &Device) -> SamplerState {
    let mut d = SamplerDescriptor::new();
    d.min_filter = SamplerMinMagFilter::Linear;
    d.mag_filter = SamplerMinMagFilter::Linear;
    d.s_address_mode = SamplerAddressMode::ClampToEdge;
    d.t_address_mode = SamplerAddressMode::ClampToEdge;
    device.new_sampler_state(&d).unwrap()
}

/// The target's bytes after two triangles over it sample `texture` through `state` at the
/// interpolated coordinate, `0..1` across and down.
fn draw(device: &Device, texture: &Texture, state: &SamplerState) -> Vec<u8> {
    let library = device
        .new_library_with_air(&llvm_assemble_text(RENDER).unwrap())
        .unwrap();
    let function = |name| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("v"));
    descriptor.fragment = Some(function("f"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let pipeline = device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, OW, OH, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&d).unwrap();
    let corners = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ];
    let vertices: Vec<u8> = corners
        .iter()
        .flat_map(|&(a, b): &(f32, f32)| {
            [
                if a == 0.0 { -1.0 } else { 1.0 },
                if b == 0.0 { 1.0 } else { -1.0 },
                0.0,
                1.0,
                a,
                b,
                0.0,
                0.0,
            ]
        })
        .flat_map(f32::to_le_bytes)
        .collect();
    let vertices = buffer_with(device, &vertices).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&pipeline).unwrap();
    encoder.set_vertex_buffer(Some(&vertices), 0, 0).unwrap();
    encoder.set_fragment_texture(Some(texture), 0).unwrap();
    encoder.set_fragment_sampler_state(Some(state), 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let mut out = vec![0; (4 * OW * OH) as usize];
    target
        .get_bytes(&mut out, 4 * OW, 0, Region::new_2d(0, 0, OW, OH), 0, 0)
        .unwrap();
    out
}

/// `bytes`' SHA-256, in hex.
fn hash(bytes: &[u8]) -> String {
    crate::hash::hex_encode(Sha256::digest(bytes))
}

/// Each output's verdict against Metal's hash: `None` where Metal's is not recorded.
fn verdict(name: &str, got: &str) -> Option<bool> {
    METAL
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, metal)| *metal == got)
}

/// Each band, the grid and the draw answer Metal's results; every output's
/// hash and each band's switches are printed.
#[test]
fn a_minified_ycbcr10_texture_samples_as_metal_does() {
    samples_as_metal_does(PACKED, "ycbcr minify");
}

/// Over an untagged surface whose planes lie at offsets with rows padded past their codes
/// ([`PADDED`]), every texel's `read()`, each band, the grid and the draw answer what Metal
/// answers over CoreVideo's layout of the same codes: the planes' places and
/// rows decide only where the codes are read, and no matrix or range is inferred beyond the
/// OSType's video range and BT.601, Metal's defaults.
#[test]
fn a_ycbcr10_texture_over_planes_at_offsets_with_padded_rows_reads_as_metal_does() {
    samples_as_metal_does(PADDED, "ycbcr padded");
}

/// The texture over a surface at `layout` read and sampled, each output's hash printed after
/// `what` and checked against Metal's, or the bands', the grid's and the draw's against the
/// [`twin_of`]'s where the device filters the kept image natively (the module docs).
fn samples_as_metal_does(layout: Layout, what: &str) {
    on_device(|device, messages| {
        eprintln!("{what} on {}", device.name());
        let texture = texture(device, layout);
        let state = sampler(device);
        let mut wrong = Vec::new();
        let texels = read_all(device, &texture);
        let got = hash(&texels);
        // A second read's conversion goes through the texture's refresh over unchanged planes:
        // no byte of them is copied (no long copy, so none split), and it reads the same texels.
        let before = device.work_counts();
        let again = hash(&read_all(device, &texture));
        let after = device.work_counts();
        assert_eq!(
            (
                after.ycbcr_refreshes - before.ycbcr_refreshes,
                after.ycbcr_whole_conversions - before.ycbcr_whole_conversions,
                after.copy_regions_split - before.copy_regions_split,
            ),
            (1, 0, 0),
            "{what}: a second read's conversion through the refresh"
        );
        assert_eq!(again, got, "{what}: a second read");
        let is = verdict("read", &got);
        eprintln!("{what} read: {is:?} hash {got}");
        if is != Some(true) {
            wrong.push("read".to_owned());
        }
        let made = |through| {
            pipeline_of(
                device,
                &llvm_assemble_text(&kernel(false, through, false, false)).unwrap(),
                "k",
            )
        };
        let (runtime, constexpr) = (
            made(Through::Runtime),
            made(Through::Constexpr(LINEAR_EDGE)),
        );
        // The samples and the draw read the planes in place on a device whose memory is the
        // host's, converting nothing; elsewhere the image the reads converted, which a natively
        // sampling device filters as it filters the twin.
        let in_place = ycbcr_in_place::reads_in_place(device);
        let twin = (!in_place && !metal_sampler(device)).then(|| twin_of(device, &texels));
        let against = match twin {
            Some(_) => "the twin's",
            None => "Metal's",
        };
        let before = device.work_counts();
        for (set, jobs) in [("bandx", band(0)), ("bandy", band(1)), ("grid", grid())] {
            for (name, kernel, sampler) in [
                ("sample_rt", &runtime, Some(&state)),
                ("sample_cx", &constexpr, None),
            ] {
                let mut out = run_held(device, (&texture, Held::Slot, sampler), kernel, &jobs, 1);
                out.truncate(4 * jobs.len());
                let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
                let label = format!("{set} {name}");
                let got = hash(&bytes);
                let is = match &twin {
                    Some(twin) => {
                        let mut plain =
                            run_held(device, (twin, Held::Slot, sampler), kernel, &jobs, 1);
                        plain.truncate(4 * jobs.len());
                        Some(
                            plain
                                .iter()
                                .map(|v| v.to_bits())
                                .eq(out.iter().map(|v| v.to_bits())),
                        )
                    }
                    None => verdict(&label, &got),
                };
                eprintln!("{what} {label}: {is:?} ({against}) hash {got}");
                if set != "grid" {
                    let axis = usize::from(set == "bandy");
                    for line in analysed(&jobs, &out, axis) {
                        eprintln!("{line}");
                    }
                }
                if is != Some(true) {
                    wrong.push(label);
                }
            }
        }
        let target = draw(device, &texture, &state);
        let work = ycbcr_in_place::ycbcr_work(&before, &device.work_counts());
        eprintln!("{what}: in place {in_place}, work {work:?}");
        match in_place {
            true => assert!(work[2] >= 7 && work[3..] == [0, 0], "{what}: in place"),
            false => assert_eq!(work[2], 0, "{what}: planes bound"),
        }
        let got = hash(&target);
        let is = match &twin {
            Some(twin) => Some(draw(device, twin, &state) == target),
            None => verdict("draw", &got),
        };
        eprintln!("{what} draw: {is:?} ({against}) hash {got}");
        if is != Some(true) {
            wrong.push("draw".to_owned());
        }
        eprintln!("announced: {:?}", counts(&messages.take_notices()));
        assert!(wrong.is_empty(), "not {against}: {wrong:?}");
    });
}
