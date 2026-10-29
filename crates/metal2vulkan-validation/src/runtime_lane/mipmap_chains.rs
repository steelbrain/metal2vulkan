//! `generate_mipmaps` on the device the cfg selects, under the validation layer: the device's blit
//! chain where its blits filter the format, the runtime's compute filter (`runtime::mipmap_filter`)
//! where they cannot, and the integer formats' texel picks.
//!
//! - **The blit chain** of formats every Vulkan device blits and filters linearly, in 2D, 2D
//!   array, cube and 3D textures: one `vkCmdBlitImage` a level (the work counts say so), the
//!   compute filter never, each level of a halving chain within a code (a half within a unit in
//!   the last place) of the bilinear filter in `float` of the level above as the device read it
//!   back, and `device-filtered-mipmaps` announced once.
//! - **The compute filter** of every format it computes that the device's blits cannot filter
//!   (the extended-range formats on every device; on lavapipe also `RGB9E5Float`, which it blits
//!   into no image, and the 8-bit sRGB formats it holds as their UNORM twins): every level the
//!   device wrote is within `bind::MipmapFilterChain::outside_bound` of the filter on the CPU of
//!   the level above as the device read it back (the device's own arithmetic: a code at a
//!   rounding boundary, a float within `2^-17` of its terms' magnitudes, a NaN any NaN), from
//!   random bytes; announced as the blit chain is.
//! - **The picks** of the integer formats: plain Metal's texel picks exactly, announced never.
//! - **The first level read in place** by the compute filter recovers every code.

use super::degrade_notices::assert_once;
use super::texture_blits::MIPMAPS;
use super::{layer_reported_nothing_at_all, validated_device, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::MipmapFilterChain;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    CommandQueue, Degradation, Device, Region, ResourceOptions, Texture, TextureDescriptor,
};

/// `n` 16-bit codes from a seed: half uniform, a quarter extremes, a quarter near the previous
/// one.
fn seeded(n: usize, seed: &mut u64) -> Vec<u16> {
    const EXTREMES: [u16; 12] = [
        0, 1, 2, 3, 0x7ffe, 0x7fff, 0x8000, 0x8001, 0x8002, 0xfffd, 0xfffe, 0xffff,
    ];
    let mut last = 0u16;
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            let v = match *seed % 4 {
                0 | 1 => (*seed >> 16) as u16,
                2 => EXTREMES[((*seed >> 8) % EXTREMES.len() as u64) as usize],
                _ => last.wrapping_add(((*seed >> 8) % 5) as u16).wrapping_sub(2),
            };
            last = v;
            v
        })
        .collect()
}

/// `n` random bytes from a seed ([`seeded`]'s codes, little-endian).
fn random_bytes(n: usize, seed: &mut u64) -> Vec<u8> {
    seeded(n / 2 + 1, seed)
        .iter()
        .flat_map(|c| c.to_le_bytes())
        .take(n)
        .collect()
}

/// A Shared texture of `pf` and `ty`, `extent`, `levels` and `slices` (cube faces included).
fn texture(
    device: &Device,
    pf: PixelFormat,
    ty: TextureType,
    extent: [usize; 3],
    levels: usize,
    slices: usize,
) -> Texture {
    let mut d = TextureDescriptor::texture_2d(pf, extent[0] as u64, extent[1] as u64, false);
    d.texture_type = ty;
    d.depth = extent[2] as u64;
    d.mipmap_level_count = levels as u64;
    if ty == TextureType::Type2DArray {
        d.array_length = slices as u64;
    }
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    device.new_texture(&d).expect("a texture")
}

/// The row and image strides of `extent`'s texels of `texel` bytes, tightly packed: an image
/// stride only for a texture of several images.
fn strides([w, h, d]: [usize; 3], texel: usize) -> (u64, u64) {
    let row = (w * texel) as u64;
    (row, if d > 1 { row * h as u64 } else { 0 })
}

/// Write `bytes` into `level` of `slice` of `t`, of `extent`, `texel` bytes a texel.
fn put(t: &Texture, slice: usize, level: usize, extent: [usize; 3], texel: usize, bytes: &[u8]) {
    let [w, h, d] = extent;
    let (row, image) = strides(extent, texel);
    let region = Region::new_3d(0, 0, 0, w as u64, h as u64, d as u64);
    t.replace_region(region, level as u64, slice as u64, bytes, row, image)
        .expect("the level written");
}

/// The bytes of `level` of `slice` of `t`, of `extent`, `texel` bytes a texel.
fn get(t: &Texture, slice: usize, level: usize, extent: [usize; 3], texel: usize) -> Vec<u8> {
    let [w, h, d] = extent;
    let (row, image) = strides(extent, texel);
    let mut out = vec![0u8; w * h * d * texel];
    let region = Region::new_3d(0, 0, 0, w as u64, h as u64, d as u64);
    t.get_bytes(&mut out, row, image, region, level as u64, slice as u64)
        .expect("the level's bytes");
    out
}

/// How a `generate_mipmaps` made its levels, by the device's work counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Made {
    /// The device's blit chain: this many `vkCmdBlitImage`.
    Blits(u64),
    /// The runtime's compute filter.
    Filter,
}

/// Generate `t`'s mipmaps in one command buffer of `queue`: how the device made them.
fn generate(device: &Device, queue: &CommandQueue, t: &Texture) -> Made {
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.generate_mipmaps(t)
        .expect("the chain runs on every device");
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let after = device.work_counts();
    let blits = after.image_blits_recorded - before.image_blits_recorded;
    let filtered = after.mipmaps_filtered - before.mipmaps_filtered;
    match (blits, filtered) {
        (0, 1) => Made::Filter,
        (n, 0) => Made::Blits(n),
        other => panic!("blits and the filter both: {other:?}"),
    }
}

/// The chain of `levels` levels of `slices` layers of `pf` from a first level of `extent`, as
/// the runtime plans it on a device of generous limits.
fn chain_of(
    pf: PixelFormat,
    ty: TextureType,
    extent: [usize; 3],
    levels: usize,
    slices: usize,
) -> Option<MipmapFilterChain> {
    MipmapFilterChain::plan(
        pf,
        extent.map(|a| a as u32),
        slices as u32,
        levels as u32,
        ty == TextureType::Type3D,
        64,
        u32::MAX,
        [65535, 65535],
        None,
    )
}

/// Level `l` of a chain from a first level of `extent`: each axis halved `l` times, at least 1.
fn level_extent(extent: [usize; 3], l: usize) -> [usize; 3] {
    extent.map(|a| (a >> l).max(1))
}

/// The bilinear filter in `float` of `above` (`slices` layers of `extent`) into the next level:
/// the compute filter's answer on the CPU.
fn filtered(
    pf: PixelFormat,
    ty: TextureType,
    extent: [usize; 3],
    slices: usize,
    above: &[u8],
) -> Vec<u8> {
    let chain = chain_of(pf, ty, extent, 2, slices).expect("a chain");
    let mut levels = chain.expected_levels(above).expect("the levels");
    levels.swap_remove(1)
}

/// The device's blit chain of formats every Vulkan device blits and filters linearly (`R8Unorm`,
/// `RGBA8Unorm`, `BGRA8Unorm`, `RGBA16Float`), in 2D textures (halving and not, one two texels
/// high), a 2D array, a cube and 3D textures: one blit a level and no compute filter, each level of a
/// halving chain within a code (a half within a unit in the last place) of the bilinear filter in
/// `float` of the level above as the device read it back, and `device-filtered-mipmaps` announced
/// once, naming the first texture.
#[test]
fn the_device_s_blit_chain_filters_each_level_and_is_announced() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        use PixelFormat as P;
        use TextureType as T;
        type Case = (
            PixelFormat,
            usize,
            TextureType,
            [usize; 3],
            usize,
            usize,
            bool,
        );
        // (format, texel bytes, type, extent, slices, levels, halving)
        let cases: [Case; 8] = [
            (P::RGBA8Unorm, 4, T::Type2D, [16, 16, 1], 1, 5, true),
            (P::RGBA8Unorm, 4, T::Type2D, [7, 5, 1], 1, 3, false),
            (P::R8Unorm, 1, T::Type2D, [32, 1, 1], 1, 6, true),
            (P::BGRA8Unorm, 4, T::Type2DArray, [8, 8, 1], 3, 4, true),
            (P::RGBA16Float, 8, T::Cube, [8, 8, 1], 6, 4, true),
            (P::RGBA8Unorm, 4, T::Type3D, [8, 8, 4], 1, 4, true),
            (P::RGBA16Float, 8, T::Type3D, [4, 4, 4], 1, 3, true),
            (P::R8Unorm, 1, T::Type2D, [64, 2, 1], 1, 7, true),
        ];
        let mut seed = 0x626c_6974_6368_6169u64;
        let mut checked = 0usize;
        for (pf, texel, ty, extent, slices, levels, halving) in cases {
            let t = texture(&device, pf, ty, extent, levels, slices);
            let texels = |l: usize| level_extent(extent, l).iter().product::<usize>();
            for slice in 0..slices {
                let bytes = match pf {
                    // Finite halves in [0, 1): the device's blit of NaNs and infinities is its
                    // own.
                    P::RGBA16Float => seeded(texels(0) * 4, &mut seed)
                        .iter()
                        .flat_map(|c| (c % 0x3c00).to_le_bytes())
                        .collect(),
                    _ => random_bytes(texels(0) * texel, &mut seed),
                };
                put(&t, slice, 0, extent, texel, &bytes);
            }
            let made = generate(&device, &queue, &t);
            assert_eq!(
                made,
                Made::Blits(levels as u64 - 1),
                "{pf:?} {ty:?} {extent:?}: a blit a level"
            );
            if !halving {
                continue;
            }
            for l in 1..levels {
                let read = |l: usize| -> Vec<u8> {
                    (0..slices)
                        .flat_map(|slice| get(&t, slice, l, level_extent(extent, l), texel))
                        .collect()
                };
                let above = read(l - 1);
                let want = filtered(pf, ty, level_extent(extent, l - 1), slices, &above);
                let got = read(l);
                let near = match pf {
                    P::RGBA16Float => got
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .zip(want.as_chunks::<2>().0)
                        .all(|(a, b)| u16::from_le_bytes(*a).abs_diff(u16::from_le_bytes(*b)) <= 1),
                    _ => got.iter().zip(&want).all(|(a, b)| a.abs_diff(*b) <= 1),
                };
                assert!(
                    near,
                    "{pf:?} {ty:?} {extent:?} level {l}: the blit {got:02x?}, the filter {want:02x?}"
                );
                checked += got.len();
            }
        }
        eprintln!(
            "the blit chain on {}: {checked} bytes of halving chains, each within a code of the \
             bilinear filter in float",
            device.name()
        );
        let notices = messages.take_notices();
        assert_once(&notices, &MIPMAPS);
        let notice = notices
            .iter()
            .filter_map(|message| message.degradation.as_ref())
            .find(|notice| notice.degradation == Degradation::DeviceFilteredMipmaps)
            .unwrap();
        assert!(notice.object.contains("RGBA8Unorm"), "{notice:?}");
        assert!(notice.reason.contains("blit chain"), "{notice:?}");
    }
    layer_reported_nothing_at_all(&messages);
}

/// Every format the compute filter computes, from random bytes (every bit pattern: NaNs,
/// infinities, subnormals, out-of-range codes), in 2D textures (one texel high, a square of a
/// power of two, one past 65 texels, halving and not), a 2D array, a cube and a 3D
/// texture: a format the device's blits filter takes its blit chain, and every other takes the
/// compute filter, each level the device wrote within `MipmapFilterChain::outside_bound` of the
/// filter on the CPU of the level above as the device read it back. The extended-range formats,
/// which no blit filters, take the filter on every device. `device-filtered-mipmaps` is
/// announced once.
#[test]
fn formats_the_blits_cannot_filter_take_the_compute_filter_level_by_level() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        let shapes: [(TextureType, [usize; 3], usize, usize); 7] = [
            (TextureType::Type2D, [7, 5, 1], 1, 3),
            (TextureType::Type2D, [17, 1, 1], 1, 5),
            (TextureType::Type2D, [8, 8, 1], 1, 4),
            (TextureType::Type2D, [66, 2, 1], 1, 3),
            (TextureType::Type2DArray, [5, 4, 1], 3, 3),
            (TextureType::Cube, [4, 4, 1], 6, 3),
            (TextureType::Type3D, [5, 4, 3], 1, 3),
        ];
        let mut seed = 0x51f1_5eed_cafe_f00du64;
        let (mut checked, mut off_model) = (0usize, 0usize);
        let (mut by_filter, mut by_blits) = (Vec::new(), Vec::new());
        for &pf in PixelFormat::ALL {
            let Some(texel) = MipmapFilterChain::texel_bytes(pf) else {
                continue;
            };
            let texel = texel as usize;
            let mut made = Vec::new();
            for (ty, extent, slices, levels) in shapes {
                let texels = level_extent(extent, 0).iter().product::<usize>();
                let t = texture(&device, pf, ty, extent, levels, slices);
                for slice in 0..slices {
                    put(
                        &t,
                        slice,
                        0,
                        extent,
                        texel,
                        &random_bytes(texels * texel, &mut seed),
                    );
                }
                let how = generate(&device, &queue, &t);
                made.push(how);
                if how != Made::Filter {
                    continue;
                }
                let read = |l: usize| -> Vec<u8> {
                    (0..slices)
                        .flat_map(|slice| get(&t, slice, l, level_extent(extent, l), texel))
                        .collect()
                };
                let chain = chain_of(pf, ty, extent, levels, slices).expect("a chain");
                for l in 1..levels {
                    let (above, got) = (read(l - 1), read(l));
                    let outside = chain.outside_bound(l, &above, &got).expect("a level");
                    assert!(
                        outside.is_empty(),
                        "{pf:?} {ty:?} {extent:?} level {l}: texels {outside:?} of {got:02x?}"
                    );
                    let model = filtered(pf, ty, level_extent(extent, l - 1), slices, &above);
                    off_model += got
                        .chunks_exact(texel)
                        .zip(model.chunks_exact(texel))
                        .filter(|(g, m)| g != m)
                        .count();
                    checked += got.len();
                }
            }
            let filter = made[0] == Made::Filter;
            assert!(
                made.iter().all(|&m| (m == Made::Filter) == filter),
                "{pf:?}: one way for every shape: {made:?}"
            );
            if filter {
                by_filter.push(pf);
            } else {
                by_blits.push(pf);
            }
        }
        eprintln!(
            "the compute filter on {}: {} formats, {checked} bytes, every one within the filter's \
             bound ({off_model} texels not the IEEE model's bytes): {by_filter:?}; the device's \
             blits filter the other {}",
            device.name(),
            by_filter.len(),
            by_blits.len()
        );
        assert_eq!(
            by_filter.len() + by_blits.len(),
            38,
            "the formats the filter computes"
        );
        for pf in [
            PixelFormat::BGR10XR,
            PixelFormat::BGR10XRSRGB,
            PixelFormat::BGRA10XR,
            PixelFormat::BGRA10XRSRGB,
        ] {
            assert!(by_filter.contains(&pf), "{pf:?}: no blit filters it");
        }
        assert_once(&messages.take_notices(), &MIPMAPS);
    }
    layer_reported_nothing_at_all(&messages);
}

/// An extended-range texture, whose image holds the raw codes no blit filters, takes the compute
/// filter on every device, not a refusal, and its `generate_mipmaps` announces
/// `device-filtered-mipmaps` naming the texture and the filter.
#[test]
fn an_extended_range_texture_takes_the_compute_filter_and_is_announced() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        let pf = PixelFormat::BGR10XR;
        let t = texture(&device, pf, TextureType::Type2D, [8, 8, 1], 4, 1);
        let mut seed = 0x7872_6d69_7073u64;
        put(&t, 0, 0, [8, 8, 1], 4, &random_bytes(8 * 8 * 4, &mut seed));
        assert_eq!(generate(&device, &queue, &t), Made::Filter);
        let read = |l: usize| get(&t, 0, l, level_extent([8, 8, 1], l), 4);
        let chain = chain_of(pf, TextureType::Type2D, [8, 8, 1], 4, 1).expect("a chain");
        for l in 1..4 {
            let outside = chain
                .outside_bound(l, &read(l - 1), &read(l))
                .expect("a level");
            assert!(outside.is_empty(), "level {l}: texels {outside:?}");
        }
        let notices = messages.take_notices();
        assert_once(&notices, &MIPMAPS);
        let notice = notices
            .iter()
            .filter_map(|message| message.degradation.as_ref())
            .find(|notice| notice.degradation == Degradation::DeviceFilteredMipmaps)
            .unwrap();
        assert!(notice.object.contains(pf.name()), "{notice:?}");
        assert!(notice.reason.contains("compute filter"), "{notice:?}");
    }
    layer_reported_nothing_at_all(&messages);
}

/// Plain Metal's integer mipmap of a level from the level above (`above`, of `above_extent`),
/// `texel` bytes a texel: texel `i` is the above's texel `floor((2i + 1) W / 2w)` on each axis,
/// `W` and `w` the two levels' sizes on it (an M4 under macOS 27.0.1).
fn picked(above: &[u8], above_extent: [usize; 3], extent: [usize; 3], texel: usize) -> Vec<u8> {
    let [aw, ah, ad] = above_extent;
    let [w, h, d] = extent;
    let pick = |i: usize, big: usize, small: usize| (2 * i + 1) * big / (2 * small);
    let mut out = Vec::with_capacity(w * h * d * texel);
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let at = ((pick(z, ad, d) * ah + pick(y, ah, h)) * aw + pick(x, aw, w)) * texel;
                out.extend_from_slice(&above[at..at + texel]);
            }
        }
    }
    out
}

/// `generate_mipmaps` of the integer formats, which Metal's API validation stops, on the device
/// the cfg selects under the validation layer: plain Metal's texel picks exactly, never refused
/// and never announced. Our probe's `R8Uint` textures, each texel the low byte of its
/// index, give its levels; 2D textures one texel high or wide, a 2D array of odd height, a cube,
/// 3D textures and a view of a 2D array's later levels and middle slices, of 1 to 16 bytes a
/// texel, give the rule applied to the level above as the device read it back.
#[test]
fn integer_mipmaps_pick_the_texels_plain_metal_picks() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        let (t2, t3) = (TextureType::Type2D, TextureType::Type3D);
        // The probe's: its levels below the first.
        type Probed<'a> = (TextureType, [usize; 3], &'a [&'a [u8]]);
        let probed: [Probed<'_>; 6] = [
            (
                t2,
                [6, 6, 1],
                &[
                    &[0x07, 0x09, 0x0b, 0x13, 0x15, 0x17, 0x1f, 0x21, 0x23],
                    &[0x15],
                ],
            ),
            (
                t2,
                [7, 5, 1],
                &[&[0x08, 0x0a, 0x0c, 0x16, 0x18, 0x1a], &[0x18]],
            ),
            (
                t2,
                [12, 10, 1],
                &[
                    &[
                        0x0d, 0x0f, 0x11, 0x13, 0x15, 0x17, 0x25, 0x27, 0x29, 0x2b, 0x2d, 0x2f,
                        0x3d, 0x3f, 0x41, 0x43, 0x45, 0x47, 0x55, 0x57, 0x59, 0x5b, 0x5d, 0x5f,
                        0x6d, 0x6f, 0x71, 0x73, 0x75, 0x77,
                    ],
                    &[0x27, 0x2b, 0x2f, 0x57, 0x5b, 0x5f],
                    &[0x5b],
                ],
            ),
            (
                t2,
                [15, 9, 1],
                &[
                    &[
                        0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x2e, 0x30, 0x32, 0x34, 0x36,
                        0x38, 0x3a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 0x6a, 0x6c, 0x6e,
                        0x70, 0x72, 0x74, 0x76,
                    ],
                    &[0x30, 0x34, 0x38, 0x6c, 0x70, 0x74],
                    &[0x70],
                ],
            ),
            (
                t3,
                [4, 4, 4],
                &[&[0x15, 0x17, 0x1d, 0x1f, 0x35, 0x37, 0x3d, 0x3f], &[0x3f]],
            ),
            (
                t3,
                [6, 4, 6],
                &[
                    &[
                        0x1f, 0x21, 0x23, 0x2b, 0x2d, 0x2f, 0x4f, 0x51, 0x53, 0x5b, 0x5d, 0x5f,
                        0x7f, 0x81, 0x83, 0x8b, 0x8d, 0x8f,
                    ],
                    &[0x5d],
                ],
            ),
        ];
        let mut checked = 0usize;
        for (ty, extent, levels) in probed {
            let full = 1 + extent.iter().max().unwrap().ilog2() as usize;
            assert_eq!(levels.len() + 1, full, "{extent:?}");
            let t = texture(&device, PixelFormat::R8Uint, ty, extent, full, 1);
            let [w, h, d] = extent;
            let first: Vec<u8> = (0..w * h * d).map(|i| i as u8).collect();
            put(&t, 0, 0, extent, 1, &first);
            generate(&device, &queue, &t);
            let at = |l: usize| extent.map(|a| (a >> l).max(1));
            let mut above = first;
            for (l, want) in (1..).zip(levels) {
                let got = get(&t, 0, l, at(l), 1);
                assert_eq!(&got, want, "{extent:?} level {l}: the probe's");
                assert_eq!(
                    got,
                    picked(&above, at(l - 1), at(l), 1),
                    "{extent:?} level {l}"
                );
                checked += got.len();
                above = got;
            }
        }
        // The other shapes, of the formats of each texel size.
        let shapes: [(PixelFormat, usize, TextureType, [usize; 3], usize); 7] = [
            (PixelFormat::RG8Sint, 2, t2, [33, 1, 1], 1),
            (PixelFormat::RG32Uint, 8, t2, [1, 9, 1], 1),
            (
                PixelFormat::RG16Uint,
                4,
                TextureType::Type2DArray,
                [9, 7, 1],
                3,
            ),
            (PixelFormat::R32Sint, 4, TextureType::Cube, [5, 5, 1], 6),
            (PixelFormat::RGBA8Uint, 4, t3, [5, 3, 7], 1),
            (PixelFormat::RGBA32Sint, 16, t2, [5, 5, 1], 1),
            (PixelFormat::RGBA16Sint, 8, t3, [8, 2, 3], 1),
        ];
        let fill = |n: usize, slice: usize| -> Vec<u8> {
            (0..n)
                .map(|i| (i * 7 + slice * 61 + i / 251) as u8)
                .collect()
        };
        for (pf, texel, ty, extent, slices) in shapes {
            let full = 1 + extent.iter().max().unwrap().ilog2() as usize;
            let t = texture(&device, pf, ty, extent, full, slices);
            let [w, h, d] = extent;
            let at = |l: usize| extent.map(|a| (a >> l).max(1));
            for slice in 0..slices {
                put(&t, slice, 0, extent, texel, &fill(w * h * d * texel, slice));
            }
            generate(&device, &queue, &t);
            for slice in 0..slices {
                let mut above = get(&t, slice, 0, extent, texel);
                for l in 1..full {
                    let got = get(&t, slice, l, at(l), texel);
                    let want = picked(&above, at(l - 1), at(l), texel);
                    assert_eq!(
                        got, want,
                        "{pf:?} {ty:?} {extent:?} slice {slice} level {l}"
                    );
                    checked += got.len();
                    above = got;
                }
            }
        }
        // A view of levels 1 to 3 and slices 1 and 2 of a 2D array of 4 levels and slices: its
        // first level is the source; level 0 and slices 0 and 3 stay.
        let (pf, texel, extent) = (PixelFormat::RGBA8Uint, 4, [11, 7, 1]);
        let t = texture(&device, pf, TextureType::Type2DArray, extent, 4, 4);
        let at = |l: usize| extent.map(|a| (a >> l).max(1));
        let mut before = Vec::new();
        for slice in 0..4 {
            for l in 0..4 {
                let [w, h, d] = at(l);
                let bytes = fill(w * h * d * texel, slice * 4 + l);
                put(&t, slice, l, at(l), texel, &bytes);
                before.push(bytes);
            }
        }
        let view = t
            .new_texture_view_with_levels_slices(pf, TextureType::Type2DArray, 1..4, 1..3)
            .unwrap();
        generate(&device, &queue, &view);
        for slice in 0..4 {
            let mut above = get(&t, slice, 1, at(1), texel);
            for l in 0..4 {
                let got = get(&t, slice, l, at(l), texel);
                if (1..3).contains(&slice) && l >= 2 {
                    assert_eq!(
                        got,
                        picked(&above, at(l - 1), at(l), texel),
                        "view {slice} {l}"
                    );
                    checked += got.len();
                    above = got;
                } else {
                    assert_eq!(got, before[slice * 4 + l], "past the view: {slice} {l}");
                }
            }
        }
        eprintln!(
            "integer mipmaps on {}: {checked} bytes, every one plain Metal's pick",
            device.name()
        );
        let announced = assert_once(&messages.take_notices(), &[]);
        assert!(announced.is_empty(), "{announced:?}");
    }
    layer_reported_nothing_at_all(&messages);
}

/// The first level read in place recovers every code: for every format the compute filter makes
/// on the device, a 2D array of two 512x256 layers whose 2x2 boxes each hold one probe texel, the
/// 65,536 values of a 16-bit pattern repeated over the texel's bytes (every code of every field:
/// each 8-, 10-, 16- or 32-bit field, every 4-, 5-, 1- and 2-bit one, every half's bits, the
/// extended-range codes), mipmapped once. Where the device read the first level in place (the
/// work count says so), each texel of the second level is the filter of one probe's four equal
/// taps, so it is within the bound of the CPU's (a box of equal taps has no rounding boundary:
/// the CPU's code) only if the fetch read back that probe's codes, up to what the filter reads
/// alike; where the level was copied in, it checks the copy. The run prints, per format,
/// which path ran and how many of its distinct probe texels the filter's answers tell apart (all
/// of them but the ones it reads alike: an snorm field's two codes of -1.0, a half's NaNs), the
/// measurement of `DriverFact::SampledFetchRecoversCodes` for the formats the device sends to the
/// filter.
#[test]
fn a_first_level_read_in_place_recovers_every_code() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        let (w, h, layers) = (512usize, 256usize, 2usize);
        let (mut in_place, mut copied, mut blitted) = (Vec::new(), Vec::new(), 0);
        for &pf in PixelFormat::ALL {
            let Some(texel) = MipmapFilterChain::texel_bytes(pf) else {
                continue;
            };
            let texel = texel as usize;
            let t = texture(&device, pf, TextureType::Type2DArray, [w, h, 1], 2, layers);
            for slice in 0..layers {
                let bytes: Vec<u8> = (0..h)
                    .flat_map(|y| {
                        (0..w).flat_map(move |x| {
                            let probe =
                                (slice * (w / 2) * (h / 2) + (y / 2) * (w / 2) + x / 2) as u16;
                            probe.to_le_bytes().into_iter().cycle().take(texel)
                        })
                    })
                    .collect();
                put(&t, slice, 0, [w, h, 1], texel, &bytes);
            }
            let before = device.work_counts().mipmap_first_levels_fetched;
            if generate(&device, &queue, &t) != Made::Filter {
                blitted += 1;
                continue;
            }
            let fetched = device.work_counts().mipmap_first_levels_fetched - before;
            assert!(fetched <= 1, "{pf:?}: one chain, one first level");
            let read = |l: usize, extent: [usize; 3]| -> Vec<u8> {
                (0..layers)
                    .flat_map(|slice| get(&t, slice, l, extent, texel))
                    .collect()
            };
            let first = read(0, [w, h, 1]);
            let chain =
                chain_of(pf, TextureType::Type2DArray, [w, h, 1], 2, layers).expect("a chain");
            let second = read(1, [w / 2, h / 2, 1]);
            let outside = chain.outside_bound(1, &first, &second).expect("a level");
            assert!(
                outside.is_empty(),
                "{pf:?}: the second level, its first {}: texels {outside:?}",
                if fetched == 1 {
                    "read in place"
                } else {
                    "copied in"
                }
            );
            // How many distinct probe texels the filter's answers tell apart.
            let probes: std::collections::BTreeSet<&[u8]> = first.chunks_exact(texel).collect();
            let answers: std::collections::BTreeSet<&[u8]> = second.chunks_exact(texel).collect();
            let row = format!("{pf:?} {}/{}", answers.len(), probes.len());
            if fetched == 1 {
                in_place.push(row);
            } else {
                copied.push(row);
            }
        }
        eprintln!(
            "first levels on {}: read in place, every code recovered (answers told apart / \
             probes): {in_place:?}; copied in: {copied:?}; {blitted} formats the device's blits \
             filter",
            device.name()
        );
        assert_eq!(
            in_place.len() + copied.len() + blitted,
            38,
            "the formats the filter computes"
        );
        assert!(
            !in_place.is_empty(),
            "the extended-range formats' integer carriers are read in place on every device"
        );
        assert_once(&messages.take_notices(), &MIPMAPS);
    }
    layer_reported_nothing_at_all(&messages);
}
