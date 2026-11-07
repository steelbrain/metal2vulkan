//! The blit encoder's texture copies: Metal's rules as probe 3 measured them on an Apple M3 Max and
//! M4, and the Vulkan regions and staging layouts that carry them. Its verdicts on
//! buffer-to-texture copies are the parity harness's records (`validation`'s scenario
//! `buffer_texture_copies`).

use super::super::blit_plan::{
    block_order_position, plan_block_order_copy, plan_buffer_copy, plan_levels_copy,
    plan_region_copy, plan_rows_copy, plan_texels_rows_copy, plan_whole_copy, texel_layout,
    texel_rows, BufferCopyPlan, BufferLayout, BufferRange, CopyPair, ImageCopyRegion, RowsCopy,
};
use super::super::copy_plan::{
    copy_unit, BlitOption, CopyProblem, CopyRefusal, CopyUnit, ImageRegion, TextureRegion,
};
use super::super::error::RuntimeGap;
use super::super::image_plan::{plan_image, ImagePlan};
use super::super::linear_plan::BufferLayout as TexelsLayout;
use super::super::pvrtc::{cpu_order_of_rows, cpu_order_position, rows_of_cpu_order};
use super::super::texture_descriptor::TextureDescriptor;
use super::super::view_plan::ViewRecord;
use super::image_plan::{committed, facts};
use super::texture_descriptor::APPLE;
use metal2vulkan::device::{Device, DeviceDescription};
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::ImageAspectFlags;

fn lavapipe() -> (DeviceDescription, Device) {
    let (_, desc, device) = committed()
        .into_iter()
        .find(|(n, ..)| n == "lavapipe")
        .unwrap();
    (desc, device)
}

/// A texture of `desc` on lavapipe: what it presents, its image, and its copy unit for `option`.
struct Texture {
    record: ViewRecord,
    image: ImagePlan,
}

fn texture(desc: TextureDescriptor) -> Texture {
    let (d, device) = lavapipe();
    let facts = facts(&d, &device);
    let layout = desc.check(APPLE).expect("a valid descriptor");
    let image = plan_image(&layout, desc.usage, false, &facts).expect("an image");
    Texture {
        record: ViewRecord::of_texture(&layout, desc.usage, desc.swizzle),
        image,
    }
}

fn tex(pf: PixelFormat, ty: TextureType, extent: [u64; 3]) -> Texture {
    texture(TextureDescriptor {
        texture_type: ty,
        pixel_format: pf,
        width: extent[0],
        height: extent[1],
        depth: extent[2],
        ..TextureDescriptor::default()
    })
}

fn unit(t: &Texture, option: BlitOption) -> Result<CopyUnit, CopyRefusal> {
    copy_unit(t.record.pixel_format, &t.image.mapping, option)
}

fn region(origin: [u32; 3], size: [u32; 3]) -> TextureRegion {
    TextureRegion {
        level: 0,
        slice: 0,
        origin,
        size,
    }
}

/// A copy into `t` from a 4096-byte buffer.
fn into(
    t: &Texture,
    option: BlitOption,
    r: TextureRegion,
    l: BufferLayout,
) -> Result<BufferCopyPlan, CopyProblem> {
    let u = unit(t, option).map_err(|e| match e {
        CopyRefusal::Invalid(p) => p,
        other => panic!("{other:?}"),
    })?;
    plan_buffer_copy(&t.record, &r, &u, &l, 4096, 0, true)
}

fn layout(offset: u64, bytes_per_row: u64, bytes_per_image: u64) -> BufferLayout {
    BufferLayout {
        offset,
        bytes_per_row,
        bytes_per_image,
    }
}

/// A depth/stencil texture's buffer copies, one aspect each: depth as 4-byte floats and stencil
/// as bytes, both straight copies. Metal's verdicts on buffer-to-texture copies (probe 3's `cbt.*`
/// cases) are the parity harness's records (`validation`'s scenario `buffer_texture_copies`), not
/// constants here.
#[test]
fn depth_stencil_buffer_copies_take_one_aspect_each() {
    let d32s8 = tex(
        PixelFormat::Depth32FloatStencil8,
        TextureType::Type2D,
        [2, 2, 1],
    );
    let full = region([0, 0, 0], [2, 2, 1]);
    let depth = into(
        &d32s8,
        BlitOption::DepthFromDepthStencil,
        full,
        layout(0, 8, 16),
    )
    .unwrap();
    assert!(depth.staging.is_none());
    assert_eq!(depth.regions[0].image.aspect, ImageAspectFlags::DEPTH);
    assert_eq!(depth.regions[0].row_length, 2);
    let stencil = into(
        &d32s8,
        BlitOption::StencilFromDepthStencil,
        full,
        layout(64, 2, 4),
    )
    .unwrap();
    assert!(stencil.staging.is_none());
    assert_eq!(stencil.regions[0].image.aspect, ImageAspectFlags::STENCIL);
    assert_eq!(
        (
            stencil.regions[0].buffer_offset,
            stencil.regions[0].row_length
        ),
        (64, 2)
    );
}

#[test]
fn layouts_vulkan_states_copy_straight() {
    let t = tex(PixelFormat::RGBA8Unorm, TextureType::Type3D, [4, 4, 3]);
    let full = region([0, 0, 0], [4, 4, 3]);
    // Row and image pitches in whole rows: one region.
    let plan = into(&t, BlitOption::None, full, layout(16, 32, 160)).unwrap();
    assert!(plan.staging.is_none());
    let r = plan.regions[0];
    assert_eq!((r.buffer_offset, r.row_length, r.image_height), (16, 8, 5));
    assert_eq!((r.image.size, r.image.layers), ([4, 4, 3], 1));
    // An image pitch that is no whole number of rows: one region per image.
    let plan = into(&t, BlitOption::None, full, layout(0, 16, 68)).unwrap();
    assert!(plan.staging.is_none());
    let offsets: Vec<(u64, [u32; 3], [u32; 3])> = plan
        .regions
        .iter()
        .map(|r| (r.buffer_offset, r.image.origin, r.image.size))
        .collect();
    assert_eq!(
        offsets,
        [
            (0, [0, 0, 0], [4, 4, 1]),
            (68, [0, 0, 1], [4, 4, 1]),
            (136, [0, 0, 2], [4, 4, 1])
        ]
    );
}

#[test]
fn a_buffer_s_base_moves_the_vulkan_offsets_and_not_metal_s_checks() {
    let t = tex(PixelFormat::RGBA8Unorm, TextureType::Type2D, [4, 4, 1]);
    let full = region([0, 0, 0], [4, 4, 1]);
    let u = unit(&t, BlitOption::None).unwrap();
    let plan = |offset, base| {
        plan_buffer_copy(
            &t.record,
            &full,
            &u,
            &layout(offset, 16, 0),
            4096,
            base,
            true,
        )
    };
    // A base of whole texels: the regions move by it; the span is Metal's.
    let at = plan(16, 8).unwrap();
    assert!(at.staging.is_none());
    assert_eq!(at.regions[0].buffer_offset, 24);
    assert_eq!(at.span, 16..80);
    // A base that misses the texel: staged, the rows read from the moved offsets.
    let staged = plan(16, 6).unwrap();
    assert_eq!(staged.span, 16..80);
    let rows: Vec<(u64, u64, u64)> = staged
        .staging
        .expect("a staged copy")
        .rows
        .iter()
        .map(|r| (r.src, r.dst, r.size))
        .collect();
    assert_eq!(rows, [(22, 0, 64)]);
    // Metal's rules hold of Metal's offset, whatever the base: an offset off the texel is
    // placed as plain Metal places it, from Metal's offset rounded down.
    let placed = plan(6, 2).unwrap();
    assert_eq!(placed.span, 4..68);
    assert_eq!(placed.staging.expect("a staged copy").rows[0].src, 6);
    assert_eq!(plan(4096 - 60, 4), Err(CopyProblem::BufferRange));
}

/// The rows' buffer offsets of a staged plan.
fn starts(plan: &BufferCopyPlan) -> Vec<u64> {
    let staging = plan.staging.as_ref().expect("a staged copy");
    staging.rows.iter().map(|r| r.src).collect()
}

/// Layouts off Metal's rules copy where plain Metal places their rows (an M4 under macOS
/// 27.0.1, our probe's sweeps): each row at its offset rounded down to the copy unit or 4
/// bytes, and Metal's aligned path into a texture from offset 0.
#[test]
fn layouts_off_metals_rules_copy_where_plain_metal_places_their_rows() {
    let copy = |pf, extent: [u64; 3], l: BufferLayout, into_texture| {
        let t = tex(pf, TextureType::Type2D, extent);
        let size = [extent[0] as u32, extent[1] as u32, 1];
        let u = unit(&t, BlitOption::None).unwrap();
        plan_buffer_copy(
            &t.record,
            &region([0, 0, 0], size),
            &u,
            &l,
            4096,
            0,
            into_texture,
        )
    };
    let at = |pf, extent, offset, bpr, into| {
        copy(pf, extent, layout(offset, bpr, 0), into).map(|p| starts(&p))
    };
    use PixelFormat as P;
    let rgba8 = (P::RGBA8Unorm, [4, 4, 1]);
    // Offsets off the unit (`sw_*`): rounded down to the unit or 4 bytes.
    for into in [true, false] {
        assert_eq!(at(rgba8.0, rgba8.1, 5, 16, into), Ok(vec![4, 20, 36, 52]));
        assert_eq!(
            at(P::RG8Unorm, [4, 4, 1], 3, 8, into),
            Ok(vec![2, 10, 18, 26])
        );
        assert_eq!(
            at(P::RGBA16Uint, [2, 4, 1], 5, 16, into),
            Ok(vec![4, 20, 36, 52])
        );
        assert_eq!(
            at(P::RGBA32Uint, [1, 4, 1], 8, 16, into),
            Ok(vec![8, 24, 40, 56])
        );
        assert_eq!(
            at(P::BC1RGBA, [8, 16, 1], 6, 16, into),
            Ok(vec![4, 20, 36, 52])
        );
        assert_eq!(
            at(P::Depth32Float, [4, 4, 1], 3, 16, into),
            Ok(vec![0, 16, 32, 48])
        );
    }
    // A `bytesPerRow` off the unit: each row rounded down on its own...
    assert_eq!(at(rgba8.0, rgba8.1, 1, 18, true), Ok(vec![0, 16, 36, 52]));
    assert_eq!(at(rgba8.0, rgba8.1, 0, 22, true), Ok(vec![0, 20, 44, 64]));
    assert_eq!(at(rgba8.0, rgba8.1, 0, 18, false), Ok(vec![0, 16, 36, 52]));
    assert_eq!(
        at(P::RGBA16Uint, [2, 4, 1], 0, 20, false),
        Ok(vec![0, 20, 40, 60])
    );
    // ...but into a texture from offset 0 with the pitch rounded to the unit a multiple of 16,
    // Metal's aligned path: rows at that pitch.
    assert_eq!(at(rgba8.0, rgba8.1, 0, 18, true), Ok(vec![0, 16, 32, 48]));
    assert_eq!(at(rgba8.0, rgba8.1, 0, 35, true), Ok(vec![0, 32, 64, 96]));
    assert_eq!(
        at(P::RGBA16Uint, [2, 4, 1], 0, 22, true),
        Ok(vec![0, 16, 32, 48])
    );
    assert_eq!(
        at(P::RGBA32Uint, [1, 4, 1], 0, 36, true),
        Ok(vec![0, 32, 64, 96])
    );
    // Rows that overlap read as each row's offset says, into a texture.
    assert_eq!(at(rgba8.0, rgba8.1, 2, 6, true), Ok(vec![0, 8, 12, 20]));
    assert_eq!(at(rgba8.0, rgba8.1, 1, 0, true), Ok(vec![0, 0, 0, 0]));
    assert_eq!(at(P::R8Uint, [4, 4, 1], 1, 3, true), Ok(vec![1, 4, 7, 10]));
    // The aligned path at a pitch below a row: each row whole from its own row's offset, the rows
    // overlapping in the buffer (an M3 Max, `probes/short_row_blits.m`), as a one-plane YCbCr
    // texture's copy at the row `MTLPixelFormatGetInfoForDevice` describes (half the row) takes.
    assert_eq!(at(rgba8.0, [8, 4, 1], 0, 16, true), Ok(vec![0, 16, 32, 48]));
    assert_eq!(
        at(P::RGBA32Uint, [4, 4, 1], 0, 32, true),
        Ok(vec![0, 32, 64, 96])
    );
    let half = at(P::YCBCR84221P, [32, 4, 1], 0, 32, true);
    assert_eq!(half, Ok(vec![0, 32, 64, 96]));
    // Refused: overlapping writes out of a texture, the aligned path's pitch of 0 (it reads
    // garbage) and offsets of 16 or more on it (not measured), and depth and stencil formats but
    // `Depth32Float` (not measured).
    let refused = Err(CopyProblem::BytesPerRow);
    assert_eq!(at(rgba8.0, rgba8.1, 0, 8, false), refused);
    assert_eq!(at(rgba8.0, rgba8.1, 0, 0, true), refused);
    assert_eq!(at(P::RGBA32Uint, [1, 4, 1], 0, 8, true), refused);
    assert_eq!(at(rgba8.0, rgba8.1, 16, 18, true), refused);
    assert_eq!(at(rgba8.0, rgba8.1, 17, 18, true), Ok(vec![16, 32, 52, 68]));
    assert_eq!(
        at(P::Depth16Unorm, [4, 4, 1], 1, 8, true),
        Err(CopyProblem::Offset)
    );
    // The images of a 3D texture (`sw_3d`): each row on its own, no aligned path.
    let t = tex(P::RGBA8Unorm, TextureType::Type3D, [4, 2, 2]);
    let u = unit(&t, BlitOption::None).unwrap();
    let r = region([0, 0, 0], [4, 2, 2]);
    for (offset, bpr, bpi, want) in [
        (0, 18, 38, [0, 16, 36, 56]),
        (2, 16, 38, [0, 16, 40, 56]),
        (2, 22, 48, [0, 24, 48, 72]),
    ] {
        for into in [true, false] {
            let l = layout(offset, bpr, bpi);
            let plan = plan_buffer_copy(&t.record, &r, &u, &l, 4096, 0, into).unwrap();
            assert_eq!(starts(&plan), want, "{offset} {bpr} {bpi} {into}");
        }
    }
}

/// Copies a staged plan's packed `texels` out to a buffer of `len` bytes of `0xab`, as a copy out
/// of the texture runs its rows back, checking that the rows' buffer bytes are disjoint (one
/// `vkCmdCopyBuffer` writes them in no order).
fn unstage(plan: &BufferCopyPlan, texels: &[u8], len: usize) -> Vec<u8> {
    let staging = plan.staging.as_ref().expect("a staged copy");
    let mut out = vec![0xabu8; len];
    let mut written = vec![false; len];
    for r in &staging.rows {
        let (s, d, n) = (r.src as usize, r.dst as usize, r.size as usize);
        assert!(n > 0, "an empty row {r:?}");
        assert!(!written[s..s + n].contains(&true), "rows overlap at {r:?}");
        written[s..s + n].fill(true);
        out[s..s + n].copy_from_slice(&texels[d..d + n]);
    }
    out
}

/// A `bytesPerImage` off Metal's rules (an M4 under macOS 27.0.1; its validation
/// stops it): a region of one image ignores it, and the images of a 3D texture are placed as rows
/// are, at `z * bytesPerImage` rounded down to the copy unit or 4 bytes; out of the texture, images
/// that overlap are written in turn, the later one's bytes staying.
#[test]
fn a_bytes_per_image_off_metals_rules_places_images_as_plain_metal_does() {
    let buffer: Vec<u8> = (0..=255).collect();
    // One image (`cbt.2d.bpi4`, `cbt.2d.bpi3`, `ctb.2d.bpi4`, `ctb.2d.bpi3`, `ctb.2d.bpi0`):
    // copied as at `bytesPerImage` 0.
    let flat = tex(PixelFormat::RGBA8Unorm, TextureType::Type2D, [2, 2, 1]);
    let u = unit(&flat, BlitOption::None).unwrap();
    let r = region([0, 0, 0], [2, 2, 1]);
    for into in [true, false] {
        let plain = plan_buffer_copy(&flat.record, &r, &u, &layout(0, 8, 0), 256, 0, into);
        for bpi in [3, 4, 18] {
            let l = layout(0, 8, bpi);
            assert_eq!(
                plan_buffer_copy(&flat.record, &r, &u, &l, 256, 0, into),
                plain,
                "{bpi} {into}"
            );
        }
    }
    // Two images of a 2x2x2 `RGBA8Unorm` texture, 16 bytes each, at `bytesPerRow` 8.
    let volume = tex(PixelFormat::RGBA8Unorm, TextureType::Type3D, [2, 2, 2]);
    let r = region([0, 0, 0], [2, 2, 2]);
    let texels: Vec<u8> = (0..0x20).collect();
    let ab = |n: usize| vec![0xabu8; n];
    for (bpi, read, written) in [
        // `cbt.3d.bpi0`, `ctb.3d.bpi0`: both images at 0; the second stays.
        (
            0,
            [(0..16).collect::<Vec<u8>>(), (0..16).collect()].concat(),
            [(0x10..0x20).collect::<Vec<u8>>(), ab(32)].concat(),
        ),
        // `bpi4`: the second image from 4.
        (
            4,
            [(0..16).collect::<Vec<u8>>(), (4..20).collect()].concat(),
            [(0..4).collect::<Vec<u8>>(), (0x10..0x20).collect(), ab(28)].concat(),
        ),
        // `bpi12`: from 12, over the first image's last 4 bytes.
        (
            12,
            [(0..16).collect::<Vec<u8>>(), (12..28).collect()].concat(),
            [(0..12).collect::<Vec<u8>>(), (0x10..0x20).collect(), ab(20)].concat(),
        ),
        // `bpi18`, off the pixel: from 18 rounded down to 16.
        (
            18,
            (0..32).collect(),
            [(0..0x20).collect::<Vec<u8>>(), ab(16)].concat(),
        ),
    ] {
        let l = layout(0, 8, bpi);
        let into = plan_buffer_copy(&volume.record, &r, &u, &l, 256, 0, true).unwrap();
        assert_eq!(stage(&into, &buffer), read, "into, bpi {bpi}");
        let out = plan_buffer_copy(&volume.record, &r, &u, &l, 48, 0, false).unwrap();
        assert_eq!(unstage(&out, &texels, 48), written, "out of, bpi {bpi}");
    }
    // Larger overlaps (`ctb.race.64x16x4.half`, `64x64x8.quarter`, `256x64x4.zero`, in order
    // in each of five runs): every image's bytes but where a later image's land.
    for (w, h, d, bpi_of) in [(64u32, 16u32, 4u32, 2u64), (64, 64, 8, 4), (256, 64, 4, 0)] {
        let t = tex(
            PixelFormat::RGBA8Unorm,
            TextureType::Type3D,
            [u64::from(w), u64::from(h), u64::from(d)],
        );
        let image = u64::from(w * h * 4);
        let bpi = image.checked_div(bpi_of).unwrap_or(0);
        let l = layout(0, u64::from(w * 4), bpi);
        let r = region([0, 0, 0], [w, h, d]);
        let len = ((u64::from(d) - 1) * bpi + image) as usize;
        let plan = plan_buffer_copy(&t.record, &r, &u, &l, len as u64, 0, false).unwrap();
        let texels: Vec<u8> = (0..image * u64::from(d))
            .map(|i| (i * 13 % 251) as u8)
            .collect();
        let mut want = ab(len);
        for z in 0..u64::from(d) {
            let (at, from) = ((z * bpi) as usize, (z * image) as usize);
            want[at..at + image as usize].copy_from_slice(&texels[from..from + image as usize]);
        }
        assert_eq!(unstage(&plan, &texels, len), want, "{w}x{h}x{d}");
    }
}

/// Applies a staging plan's row copies to `buffer`, giving the packed staging bytes.
fn stage(plan: &BufferCopyPlan, buffer: &[u8]) -> Vec<u8> {
    let staging = plan.staging.as_ref().expect("a staged copy");
    let mut out = vec![0u8; staging.length as usize];
    for r in &staging.rows {
        let (s, d, n) = (r.src as usize, r.dst as usize, r.size as usize);
        out[d..d + n].copy_from_slice(&buffer[s..s + n]);
    }
    out
}

#[test]
fn layouts_vulkan_cannot_state_copy_through_packed_staging() {
    let buffer: Vec<u8> = (0..4096u32).map(|i| (i * 7 % 251) as u8).collect();
    // A stencil copy at an offset Vulkan does not take for a depth/stencil image (Metal takes any
    // multiple of a byte).
    let d32s8 = tex(
        PixelFormat::Depth32FloatStencil8,
        TextureType::Type2D,
        [3, 2, 1],
    );
    let plan = into(
        &d32s8,
        BlitOption::StencilFromDepthStencil,
        region([0, 0, 0], [3, 2, 1]),
        layout(65, 5, 0),
    )
    .unwrap();
    assert_eq!(plan.regions[0].buffer_offset, 0);
    assert_eq!(
        stage(&plan, &buffer),
        [&buffer[65..68], &buffer[70..73]].concat()
    );
    // A 3D image pitch that is no multiple of the texel: each row moves on its own, at its
    // offset rounded down to 4 bytes, as plain Metal reads it: the second image's
    // rows at 24 and 36.
    let t = tex(PixelFormat::RGBA8Unorm, TextureType::Type3D, [2, 2, 2]);
    let plan = into(
        &t,
        BlitOption::None,
        region([0, 0, 0], [2, 2, 2]),
        layout(0, 12, 26),
    )
    .unwrap();
    let want: Vec<u8> = [0usize, 12, 24, 36]
        .iter()
        .flat_map(|&o| buffer[o..o + 8].to_vec())
        .collect();
    assert_eq!(stage(&plan, &buffer), want);
    assert_eq!(plan.regions[0].row_length, 0);
    assert_eq!(plan.regions[0].image.size, [2, 2, 2]);
}

fn pair<'a>(src: &'a Texture, dst: &'a Texture) -> CopyPair<'a> {
    CopyPair {
        src: &src.record,
        src_image: Some(&src.image),
        dst: &dst.record,
        dst_image: Some(&dst.image),
    }
}

/// Probe 3's `ct.*` cases (macOS 26.7's verdicts; macOS 27.0.1 takes only equal formats).
#[test]
fn texture_copies_follow_metals_measured_verdicts() {
    use PixelFormat as P;
    let t = |pf| tex(pf, TextureType::Type2D, [2, 2, 1]);
    let src = t(P::RGBA8Unorm);
    let full = region([0, 0, 0], [2, 2, 1]);
    for dst in [
        P::RGBA8UnormSRGB,
        P::BGRA8Unorm,
        P::R32Uint,
        P::RGBA8Uint,
        P::RGBA8Unorm,
    ] {
        let d = t(dst);
        let regions = plan_region_copy(&pair(&src, &d), &full, 0, 0, [0, 0, 0]).unwrap();
        assert_eq!(regions.len(), 1, "{dst:?}");
        assert_eq!(regions[0].dst.size, [2, 2, 1]);
    }
    let wide = t(P::RGBA16Float);
    assert_eq!(
        plan_region_copy(&pair(&src, &wide), &full, 0, 0, [0, 0, 0]),
        Err(CopyRefusal::Invalid(CopyProblem::Format))
    );
    // `ct.r32f.depth32f` and probes: Metal copies the bytes between a depth or stencil format and a
    // colour format of its unit, each texture in its own aspect.
    let aspects = |a: &Texture, b: &Texture| {
        plan_region_copy(&pair(a, b), &full, 0, 0, [0, 0, 0])
            .map(|r| (r[0].src.aspect, r[0].dst.aspect))
    };
    let (color, depth, stencil) = (
        ImageAspectFlags::COLOR,
        ImageAspectFlags::DEPTH,
        ImageAspectFlags::STENCIL,
    );
    let (r32, d32) = (t(P::R32Float), t(P::Depth32Float));
    assert_eq!(aspects(&r32, &d32), Ok((color, depth)));
    assert_eq!(aspects(&d32, &t(P::RGBA8Unorm)), Ok((depth, color)));
    assert_eq!(
        aspects(&t(P::Depth16Unorm), &t(P::RG8Unorm)),
        Ok((depth, color))
    );
    assert_eq!(
        aspects(&t(P::R8Uint), &t(P::Stencil8)),
        Ok((color, stencil))
    );
    // Units of different sizes, and a combined format with another, are Metal's to refuse.
    for (a, b) in [
        (P::Depth32Float, P::Stencil8),
        (P::Depth16Unorm, P::Depth32Float),
        (P::Depth32FloatStencil8, P::RG32Uint),
        (P::RG32Uint, P::Depth32FloatStencil8),
    ] {
        assert_eq!(
            aspects(&t(a), &t(b)),
            Err(CopyRefusal::Invalid(CopyProblem::Format)),
            "{a:?} to {b:?}"
        );
    }
    // A combined format copies into its own, both aspects.
    let both = depth | stencil;
    let d32s8 = t(P::Depth32FloatStencil8);
    assert_eq!(
        aspects(&d32s8, &t(P::Depth32FloatStencil8)),
        Ok((both, both))
    );
    // A destination region past its level.
    assert_eq!(
        plan_region_copy(&pair(&src, &src), &full, 0, 0, [1, 0, 0]),
        Err(CopyRefusal::Invalid(CopyProblem::Region))
    );
}

#[test]
fn level_and_slice_copies_pair_levels_of_one_size() {
    let array = |extent: u64, levels: u64| {
        texture(TextureDescriptor {
            texture_type: TextureType::Type2DArray,
            width: extent,
            height: extent,
            array_length: 3,
            mipmap_level_count: levels,
            ..TextureDescriptor::default()
        })
    };
    let (a, b) = (array(4, 3), array(4, 3));
    let shape = |regions: &[ImageCopyRegion]| -> Vec<(u32, u32, u32, u32, u32, [u32; 3])> {
        regions
            .iter()
            .map(|r| {
                assert_eq!(r.src.size, r.dst.size);
                (
                    r.src.level,
                    r.src.layer,
                    r.dst.level,
                    r.dst.layer,
                    r.src.layers,
                    r.src.size,
                )
            })
            .collect()
    };
    // `ct.levels.slices`: source level 1 (2x2) to destination level 0 (4x4) copies their common
    // 2x2 region, and level 2 (1x1) to level 1 (2x2) its 1x1.
    let common = plan_levels_copy(&pair(&a, &b), 1, 1, 0, 0, 2, 2).unwrap();
    assert_eq!(
        shape(&common),
        [(1, 1, 0, 0, 2, [2, 2, 1]), (2, 1, 1, 0, 2, [1, 1, 1])]
    );
    let small = array(2, 2);
    let regions = plan_levels_copy(&pair(&a, &small), 1, 1, 0, 0, 2, 2).unwrap();
    let got: Vec<(u32, u32, u32, u32, u32, [u32; 3])> = regions
        .iter()
        .map(|r| {
            (
                r.src.level,
                r.src.layer,
                r.dst.level,
                r.dst.layer,
                r.src.layers,
                r.src.size,
            )
        })
        .collect();
    assert_eq!(
        got,
        [(1, 1, 0, 0, 2, [2, 2, 1]), (2, 1, 1, 0, 2, [1, 1, 1])]
    );
    assert_eq!(
        plan_levels_copy(&pair(&a, &small), 2, 1, 0, 0, 2, 2),
        Err(CopyRefusal::Invalid(CopyProblem::Region))
    );
    // `ct.whole`: every level and slice of two textures of one shape.
    let whole = plan_whole_copy(&pair(&a, &b)).unwrap();
    assert_eq!(whole.len(), 3);
    assert!(whole.iter().all(|r| r.src.layers == 3 && r.src.layer == 0));
    // The source's level 1 is the destination's first size: levels 1 and 2 into 0 and 1.
    assert_eq!(
        shape(&plan_whole_copy(&pair(&a, &small)).unwrap()),
        [(1, 0, 0, 0, 3, [2, 2, 1]), (2, 0, 1, 0, 3, [1, 1, 1])]
    );
    // Sample counts must match.
    let ms = texture(TextureDescriptor {
        texture_type: TextureType::Type2DMultisample,
        width: 4,
        height: 4,
        sample_count: 4,
        usage: TextureUsage::RENDER_TARGET,
        ..TextureDescriptor::default()
    });
    let single = tex(PixelFormat::RGBA8Unorm, TextureType::Type2D, [4, 4, 1]);
    assert_eq!(
        plan_whole_copy(&pair(&ms, &single)),
        Err(CopyRefusal::Invalid(CopyProblem::SampleCount))
    );
}

/// Plain Metal's copies between textures of different sizes (an M4 under macOS 27.0.1, and an
/// Apple M3 Max under macOS 27.0.1 for the compressed and the shifted-pair shapes): levels of
/// different sizes copy the copy units both hold, a compressed pair's blocks included, each side
/// ending at its own level's edge; a whole copy pairs levels by extent from the first pair of one
/// extent, shifting the textures' own extents past their level counts, and copies the levels and
/// slices both textures have, none where the pair is one past a texture's last level, and aborts
/// where the extents pair at no level.
#[test]
fn copies_between_textures_of_different_sizes_follow_metal() {
    use TextureType as T;
    let made = |ty: T, extent: [u64; 3], levels: u64, slices: u64| {
        texture(TextureDescriptor {
            texture_type: ty,
            width: extent[0],
            height: extent[1],
            depth: extent[2],
            mipmap_level_count: levels,
            array_length: slices,
            ..TextureDescriptor::default()
        })
    };
    let d2 = |w: u64, h: u64, levels: u64| made(T::Type2D, [w, h, 1], levels, 1);
    // (source level, destination level, layers, size) of each region, where the two sides are
    // one size; `sided` keeps both where a compressed pair's sides end at different edges.
    let sided = |copied: Result<Vec<ImageCopyRegion>, CopyRefusal>| {
        copied.map(|regions| {
            regions
                .iter()
                .map(|r| {
                    assert_eq!((r.src.layer, r.dst.layer), (0, 0));
                    (
                        r.src.level,
                        r.dst.level,
                        r.src.layers,
                        r.src.size,
                        r.dst.size,
                    )
                })
                .collect::<Vec<_>>()
        })
    };
    let regions = |copied: Result<Vec<ImageCopyRegion>, CopyRefusal>| {
        copied.map(|regions| {
            regions
                .iter()
                .map(|r| {
                    assert_eq!((r.src.size, r.src.layer, r.dst.layer), (r.dst.size, 0, 0));
                    (r.src.level, r.dst.level, r.src.layers, r.src.size)
                })
                .collect::<Vec<_>>()
        })
    };
    // `sz_slices_8x4_to_4x8`, `sz_slices_3d_d4_to_d2`, `sz_slices_3d_d2_to_d4`.
    let (wide, tall) = (d2(8, 4, 1), d2(4, 8, 1));
    assert_eq!(
        regions(plan_levels_copy(&pair(&wide, &tall), 0, 0, 0, 0, 1, 1)),
        Ok(vec![(0, 0, 1, [4, 4, 1])])
    );
    let (deep, shallow) = (
        made(T::Type3D, [4, 2, 4], 1, 1),
        made(T::Type3D, [4, 2, 2], 1, 1),
    );
    for (a, b) in [(&deep, &shallow), (&shallow, &deep)] {
        assert_eq!(
            regions(plan_levels_copy(&pair(a, b), 0, 0, 0, 0, 1, 1)),
            Ok(vec![(0, 0, 1, [4, 2, 2])])
        );
    }
    // `sz_slices_two_levels`.
    assert_eq!(
        regions(plan_levels_copy(
            &pair(&d2(8, 8, 2), &d2(4, 4, 2)),
            0,
            0,
            0,
            0,
            1,
            2
        )),
        Ok(vec![(0, 0, 1, [4, 4, 1]), (1, 1, 1, [2, 2, 1])])
    );
    let size = Err(CopyRefusal::Invalid(CopyProblem::Size));
    let whole = |a: &Texture, b: &Texture| regions(plan_whole_copy(&pair(a, b)));
    // `sz_whole_8l4_to_4l3`, `_4l3_to_8l4`, `_8l4_to_4l1`.
    assert_eq!(
        whole(&d2(8, 8, 4), &d2(4, 4, 3)),
        Ok(vec![
            (1, 0, 1, [4, 4, 1]),
            (2, 1, 1, [2, 2, 1]),
            (3, 2, 1, [1, 1, 1])
        ])
    );
    assert_eq!(
        whole(&d2(4, 4, 3), &d2(8, 8, 4)),
        Ok(vec![
            (0, 1, 1, [4, 4, 1]),
            (1, 2, 1, [2, 2, 1]),
            (2, 3, 1, [1, 1, 1])
        ])
    );
    assert_eq!(
        whole(&d2(8, 8, 4), &d2(4, 4, 1)),
        Ok(vec![(1, 0, 1, [4, 4, 1])])
    );
    // `sz_whole_8_to_3`, `_8x4_to_4x8`, `_8x4_to_4x4`: no pair of one size; plain Metal aborts.
    assert_eq!(whole(&d2(8, 8, 4), &d2(3, 3, 2)), size);
    assert_eq!(whole(&d2(8, 4, 4), &d2(4, 8, 4)), size);
    assert_eq!(whole(&d2(8, 4, 4), &d2(4, 4, 3)), size);
    // `sz_whole_8x4_to_4x2`, `_6l3_to_3l2`, `_5l3_to_2l2`.
    assert_eq!(
        whole(&d2(8, 4, 4), &d2(4, 2, 3)),
        Ok(vec![
            (1, 0, 1, [4, 2, 1]),
            (2, 1, 1, [2, 1, 1]),
            (3, 2, 1, [1, 1, 1])
        ])
    );
    assert_eq!(
        whole(&d2(6, 6, 3), &d2(3, 3, 2)),
        Ok(vec![(1, 0, 1, [3, 3, 1]), (2, 1, 1, [1, 1, 1])])
    );
    assert_eq!(
        whole(&d2(5, 5, 3), &d2(2, 2, 2)),
        Ok(vec![(1, 0, 1, [2, 2, 1]), (2, 1, 1, [1, 1, 1])])
    );
    // `sz_whole_arr2l4_to_arr3l2`, `_arr_8to4`: the slices both have.
    let array =
        |side: u64, levels: u64, slices: u64| made(T::Type2DArray, [side, side, 1], levels, slices);
    assert_eq!(
        whole(&array(8, 4, 2), &array(8, 2, 3)),
        Ok(vec![(0, 0, 2, [8, 8, 1]), (1, 1, 2, [4, 4, 1])])
    );
    assert_eq!(
        whole(&array(8, 4, 2), &array(4, 3, 3)),
        Ok(vec![
            (1, 0, 2, [4, 4, 1]),
            (2, 1, 2, [2, 2, 1]),
            (3, 2, 2, [1, 1, 1])
        ])
    );
    // `sz_whole_3d_8to4`, `sz_whole_2d_to_arr`.
    assert_eq!(
        whole(
            &made(T::Type3D, [8, 8, 4], 2, 1),
            &made(T::Type3D, [4, 4, 2], 2, 1)
        ),
        Ok(vec![(1, 0, 1, [4, 4, 2])])
    );
    assert_eq!(
        whole(&d2(4, 4, 1), &array(4, 1, 2)),
        Ok(vec![(0, 0, 1, [4, 4, 1])])
    );
    // `wh-plain-16-to-8`, `wh-bc1-16-to-8`, `wh-bc1-16-to-32`, `wh-bc1-16l2-to-4l1`: a pair one
    // past a texture's last level copies nothing, as plain Metal copies it.
    for (name, a, b) in [
        ("16l1 into 8l1", d2(16, 16, 1), d2(8, 8, 1)),
        ("16l1 into 32l1", d2(16, 16, 1), d2(32, 32, 1)),
        ("16l2 into 4l1", d2(16, 16, 2), d2(4, 4, 1)),
    ] {
        assert_eq!(whole(&a, &b), Ok(Vec::new()), "{name}");
    }
    // `wh-bc1-16l3-to-4l1`, `wh-plain-16l5-to-1l1`: a pair past the first level, which the
    // source's own level count does reach.
    assert_eq!(
        whole(&d2(16, 16, 3), &d2(4, 4, 1)),
        Ok(vec![(2, 0, 1, [4, 4, 1])])
    );
    assert_eq!(
        whole(&d2(16, 16, 5), &d2(1, 1, 1)),
        Ok(vec![(4, 0, 1, [1, 1, 1])])
    );
    // `wh-plain-16-to-4`, `wh-bc1-16-to-4`, `wh-bc1-32-to-4`, `wh-plain-32-to-8`,
    // `wh-plain-4-to-16`, `wh-bc1-4-to-16`: a pair past the end of one texture's own levels,
    // where plain Metal reads or writes a level that texture does not hold.
    assert_eq!(whole(&d2(16, 16, 1), &d2(4, 4, 1)), size);
    assert_eq!(whole(&d2(32, 32, 1), &d2(4, 4, 1)), size);
    assert_eq!(whole(&d2(32, 32, 1), &d2(8, 8, 1)), size);
    assert_eq!(whole(&d2(4, 4, 1), &d2(16, 16, 1)), size);
    // Compressed levels of different sizes copy the blocks both levels hold, block for block,
    // from their origins: the lesser block count on each axis, each side ending at its own
    // level's edge where that block passes it (an Apple M3 Max under macOS 27.0.1, our probe
    // `lv-bc1-*`, `lv-bc3-16-to-8`, `lv-bc4-16-to-8`, `lv-bc7-16-to-8`, `lv-eac-16-to-8`,
    // `lv-etc2-16-to-8` and `lv-astc65-*`).
    let bc1 = |w: u64, h: u64, levels: u64| {
        texture(TextureDescriptor {
            texture_type: T::Type2D,
            pixel_format: PixelFormat::BC1RGBA,
            width: w,
            height: h,
            mipmap_level_count: levels,
            ..TextureDescriptor::default()
        })
    };
    let square = |side: u64| bc1(side, side, 1);
    // `lv-bc1-16-to-8`, `_to-12`, `_to-4`, `_to-20`: block counts that divide the extent.
    for (dst, size) in [
        (8, [8, 8, 1]),
        (12, [12, 12, 1]),
        (4, [4, 4, 1]),
        (20, [16, 16, 1]),
    ] {
        assert_eq!(
            sided(plan_levels_copy(
                &pair(&square(16), &square(dst)),
                0,
                0,
                0,
                0,
                1,
                1
            )),
            Ok(vec![(0, 0, 1, size, size)]),
            "BC1 16 into {dst}"
        );
    }
    // `lv-bc1-16-to-6`, `lv-bc1-18-to-6`: two blocks each way, the destination's second ending
    // inside itself, so the two sides differ in size and cover the same blocks.
    for src in [16, 18] {
        assert_eq!(
            sided(plan_levels_copy(
                &pair(&square(src), &square(6)),
                0,
                0,
                0,
                0,
                1,
                1
            )),
            Ok(vec![(0, 0, 1, [8, 8, 1], [6, 6, 1])]),
            "BC1 {src} into 6"
        );
    }
    // `lv-bc1-16x8-to-8x16`: the lesser block count on each axis on its own.
    assert_eq!(
        sided(plan_levels_copy(
            &pair(&bc1(16, 8, 1), &bc1(8, 16, 1)),
            0,
            0,
            0,
            0,
            1,
            1
        )),
        Ok(vec![(0, 0, 1, [8, 8, 1], [8, 8, 1])])
    );
    // `lv-bc1-32l2-to-16l2-level1`, `_levels2`: a level other than 0, and two at once.
    assert_eq!(
        sided(plan_levels_copy(
            &pair(&bc1(32, 32, 2), &bc1(16, 16, 2)),
            0,
            1,
            0,
            1,
            1,
            1
        )),
        Ok(vec![(1, 1, 1, [8, 8, 1], [8, 8, 1])])
    );
    assert_eq!(
        sided(plan_levels_copy(
            &pair(&bc1(32, 32, 2), &bc1(16, 16, 2)),
            0,
            0,
            0,
            0,
            1,
            2
        )),
        Ok(vec![
            (0, 0, 1, [16, 16, 1], [16, 16, 1]),
            (1, 1, 1, [8, 8, 1], [8, 8, 1])
        ])
    );
    // `wh-bc1-16l2-to-8l1`: a whole copy of compressed textures pairs levels as any other does.
    assert_eq!(
        sided(plan_whole_copy(&pair(&bc1(16, 16, 2), &bc1(8, 8, 1)))),
        Ok(vec![(1, 0, 1, [8, 8, 1], [8, 8, 1])])
    );
}

/// A whole block of a compressed level smaller than one copied between textures (an Apple M3 Max
/// under macOS 26.7, a probe): each side the blocks it starts in, ending at its level's edge, so
/// a whole block of a 4 x 2 level and one of an 8 x 8 level differ in size and cover the same
/// block, each way and from 4 x 4 or 4 x 2 (`t2t_small_to_big_*`, `t2t_big_to_small_*`), and
/// levels that end alike stay one size (`t2t_same_l9_4x4`).
#[test]
fn whole_blocks_of_small_levels_copy_between_textures() {
    let bc3 = |side: [u64; 2], levels: u64| {
        texture(TextureDescriptor {
            texture_type: TextureType::Type2D,
            pixel_format: PixelFormat::BC3RGBA,
            width: side[0],
            height: side[1],
            mipmap_level_count: levels,
            ..TextureDescriptor::default()
        })
    };
    let (mipped, other, big) = (bc3([2048, 1024], 12), bc3([2048, 1024], 12), bc3([8, 8], 1));
    let copy = |src: &Texture, level, origin, size, dst: &Texture, to_level, to| {
        let region = TextureRegion {
            level,
            slice: 0,
            origin,
            size,
        };
        plan_region_copy(&pair(src, dst), &region, to_level, 0, to).map(|regions| {
            regions
                .iter()
                .map(|r| ((r.src.origin, r.src.size), (r.dst.origin, r.dst.size)))
                .collect::<Vec<_>>()
        })
    };
    let (zero, corner) = ([0, 0, 0], [4, 4, 0]);
    let (small, whole) = ((zero, [4, 2, 1]), (corner, [4, 4, 1]));
    for size in [[4, 4, 1], [4, 2, 1]] {
        assert_eq!(
            copy(&mipped, 9, zero, size, &big, 0, corner),
            Ok(vec![(small, whole)]),
            "{size:?}"
        );
        assert_eq!(
            copy(&big, 0, corner, size, &other, 9, zero),
            Ok(vec![(whole, small)]),
            "{size:?}"
        );
    }
    assert_eq!(
        copy(&mipped, 9, zero, [4, 4, 1], &other, 9, zero),
        Ok(vec![(small, small)])
    );
    // `t2t_l9_to_l10_4x4`: level 10 is 2 x 1.
    assert_eq!(
        copy(&mipped, 9, zero, [4, 4, 1], &other, 10, zero),
        Ok(vec![(small, (zero, [2, 1, 1]))])
    );
}

/// A 4x4 `RGBA8Unorm` texture over a buffer at offset 256, rows 32 bytes apart (the layout of a
/// Metal probe on the M3 Max and an M4), two slices 128 bytes apart for an array.
fn over_buffer(slices: u64) -> (Texture, TexelsLayout) {
    let t = texture(TextureDescriptor {
        texture_type: if slices > 1 {
            TextureType::Type2DArray
        } else {
            TextureType::Type2D
        },
        pixel_format: PixelFormat::RGBA8Unorm,
        width: 4,
        height: 4,
        array_length: slices,
        ..TextureDescriptor::default()
    });
    let layout = TexelsLayout {
        offset: 256,
        row_pitch: 32,
        slice_pitch: 128,
        row_bytes: 16,
        rows: 4,
        slices,
    };
    (t, layout)
}

/// A row copy between two textures over buffers is refused where either side's GPU rows share
/// bytes, and planned where neither's do.
///
/// An IOSurface plane whose `bytesPerRow` rounded down to 16 falls below its row has GPU rows
/// that share bytes while its own rows lie a row apart. Metal leaves those bytes from whichever
/// of the two rows its own command wrote last -- an order that follows the command, the GPU pitch
/// and the row count, and that one shape of `copyFromTexture:toTexture:` does not settle from run
/// to run (`linear_plan.rs`'s **Rows that overlap**) -- so no order a row copy writes them in is
/// Metal's, and the copy is refused whichever side the plane is on. Rows a row apart or further
/// are planned, each side's from its own layout.
#[test]
fn a_row_copy_between_two_textures_over_buffers_refuses_either_sides_shared_bytes() {
    let (t, plain) = over_buffer(1);
    let unit = unit(&t, BlitOption::None).unwrap();
    // A 5x4 texture of 4-byte texels over a plane 20 bytes a row: Metal's GPU takes its rows 16
    // apart, so consecutive rows share 4 bytes, while the plane's own rows lie a row apart.
    let gpu = TexelsLayout {
        offset: 0,
        row_pitch: 16,
        slice_pitch: 64,
        row_bytes: 20,
        rows: 4,
        slices: 1,
    };
    let cpu = TexelsLayout {
        row_pitch: 20,
        slice_pitch: 80,
        ..gpu
    };
    assert_eq!((gpu.shared_bytes(), cpu.shared_bytes()), (4, 0));
    // A second plain layout, so each side's rows are told apart.
    let other = TexelsLayout {
        offset: 1024,
        ..plain
    };
    let region = ImageRegion {
        aspect: ImageAspectFlags::COLOR,
        level: 0,
        layer: 0,
        layers: 1,
        origin: [0, 0, 0],
        size: [4, 4, 1],
    };
    let copies = [ImageCopyRegion {
        src: region,
        dst: region,
    }];
    let plan = |from: (&TexelsLayout, &TexelsLayout), to: (&TexelsLayout, &TexelsLayout)| {
        plan_texels_rows_copy(from, to, &unit, &copies)
    };
    let gap = Err(RuntimeGap::OverlappingSurfaceRows);
    // The plane as the source, as the destination, and on both sides.
    assert_eq!(plan((&gpu, &cpu), (&plain, &plain)), gap);
    assert_eq!(plan((&plain, &plain), (&gpu, &cpu)), gap);
    assert_eq!(plan((&gpu, &cpu), (&gpu, &cpu)), gap);
    // Two plain textures: a row each, source rows 32 bytes apart from 256 and destination rows
    // from 1024.
    let copy = plan((&plain, &plain), (&other, &other)).expect("a plan");
    assert_eq!(
        copy.rows,
        (0..4)
            .map(|y| BufferRange {
                src: 256 + y * 32,
                dst: 1024 + y * 32,
                size: 16,
            })
            .collect::<Vec<_>>()
    );
    assert_eq!(
        (copy.src_span.clone(), copy.dst_span.clone()),
        (256..352 + 16, 1024..1120 + 16)
    );
}

/// A copy into a texture over a buffer moves its texel rows only, never the bytes between them;
/// Metal's checks of the buffer's side hold as for any texture.
#[test]
fn copies_into_textures_over_buffers_move_texel_rows() {
    let (t, texels) = over_buffer(1);
    let unit = unit(&t, BlitOption::None).unwrap();
    let layout = BufferLayout {
        offset: 16,
        bytes_per_row: 8,
        bytes_per_image: 0,
    };
    let (from, to) = plan_rows_copy(
        &t.record,
        &region([1, 1, 0], [2, 3, 1]),
        &unit,
        &layout,
        4096,
        &texels,
    )
    .unwrap();
    assert_eq!(from, vec![16..24, 24..32, 32..40]);
    assert_eq!(to, vec![256 + 32 + 4..256 + 32 + 12, 324..332, 356..364]);
    // The source rows are contiguous and the texture's are not: one row each.
    let copy = RowsCopy::new(&from, &to);
    assert_eq!(copy.rows.len(), 3);
    assert_eq!(
        (copy.src_span.clone(), copy.dst_span.clone()),
        (16..40, 292..364)
    );
    assert!(!copy.meets());
    // Metal's checks of the buffer's side: rows that do not cover the region.
    let short = BufferLayout {
        bytes_per_row: 4,
        ..layout
    };
    assert_eq!(
        plan_rows_copy(
            &t.record,
            &region([0, 0, 0], [2, 2, 1]),
            &unit,
            &short,
            4096,
            &texels
        ),
        Err(CopyProblem::BytesPerRow)
    );
}

/// A BC texture over a surface's rows (`linear_plan.rs`) copies rows of blocks: a region's texels
/// in blocks, each block its format's bytes, a row of blocks a row of the surface, a last block
/// of a row partly past the texture's texels whole (`surface_texture_envelope`'s blits on the x86
/// Macs).
#[test]
fn a_bc_texture_over_a_buffer_moves_rows_of_blocks() {
    let unit = CopyUnit {
        bytes: 8,
        extent: [4, 4],
        aspect: ImageAspectFlags::COLOR,
    };
    // A plane at 32, 64 bytes a row (a 420v surface's luma plane on the x86 Macs).
    let texels = TexelsLayout {
        offset: 32,
        row_pitch: 64,
        slice_pitch: 64 * 16,
        row_bytes: 32,
        rows: 16,
        slices: 1,
    };
    let image = |origin: [u32; 3], size: [u32; 3]| super::super::copy_plan::ImageRegion {
        aspect: ImageAspectFlags::COLOR,
        level: 0,
        layer: 0,
        layers: 1,
        origin,
        size,
    };
    // The whole 16 x 64 texture: 16 rows of 4 blocks.
    let whole = texel_rows(&texels, &unit, &image([0, 0, 0], [16, 64, 1]));
    assert_eq!(whole.len(), 16);
    assert_eq!((whole[0].clone(), whole[15].clone()), (32..64, 992..1024));
    // Blocks (1, 2) and (2, 2), the second partly past a 10-texel-wide region's texels.
    assert_eq!(
        texel_rows(&texels, &unit, &image([4, 8, 0], [6, 4, 1])),
        vec![32 + 2 * 64 + 8..32 + 2 * 64 + 24]
    );
    assert_eq!(
        texel_layout(&texels, &unit, &image([4, 8, 0], [6, 4, 1]), 0),
        BufferLayout {
            offset: 32 + 2 * 64 + 8,
            bytes_per_row: 64,
            bytes_per_image: 0,
        }
    );
}

/// Rows that are contiguous in both buffers are one copy, and a copy whose source and destination
/// rows share a byte meets (a Metal probe's overlapping copies, the M3 Max and an M4).
#[test]
fn texel_rows_merge_and_meet() {
    let (t, texels) = over_buffer(2);
    let unit = unit(&t, BlitOption::None).unwrap();
    let image = |layer, origin: [u32; 3], size: [u32; 3]| super::super::copy_plan::ImageRegion {
        aspect: ImageAspectFlags::COLOR,
        level: 0,
        layer,
        layers: 1,
        origin,
        size,
    };
    // Slice 1's rows start 128 bytes after slice 0's.
    assert_eq!(
        texel_rows(&texels, &unit, &image(1, [0, 0, 0], [4, 2, 1])),
        vec![384..400, 416..432]
    );
    let both = super::super::copy_plan::ImageRegion {
        layers: 2,
        ..image(0, [0, 3, 0], [4, 1, 1])
    };
    assert_eq!(texel_rows(&texels, &unit, &both), vec![352..368, 480..496]);
    assert_eq!(
        texel_layout(&texels, &unit, &image(0, [2, 1, 0], [1, 1, 1]), 1),
        BufferLayout {
            offset: 256 + 128 + 32 + 8,
            bytes_per_row: 32,
            bytes_per_image: 0,
        }
    );
    // A row down within one texture: row y into row y + 1 meets.
    let src = texel_rows(&texels, &unit, &image(0, [0, 0, 0], [4, 3, 1]));
    let dst = texel_rows(&texels, &unit, &image(0, [0, 1, 0], [4, 3, 1]));
    assert!(RowsCopy::new(&src, &dst).meets());
    // Two texels right within each row: the columns meet.
    let src = texel_rows(&texels, &unit, &image(0, [0, 0, 0], [3, 4, 1]));
    let dst = texel_rows(&texels, &unit, &image(0, [1, 0, 0], [3, 4, 1]));
    assert!(RowsCopy::new(&src, &dst).meets());
    // Disjoint columns of the same rows do not.
    let src = texel_rows(&texels, &unit, &image(0, [0, 0, 0], [2, 4, 1]));
    let dst = texel_rows(&texels, &unit, &image(0, [2, 0, 0], [2, 4, 1]));
    assert!(!RowsCopy::new(&src, &dst).meets());
    // Packed rows in both merge into one.
    let copy = RowsCopy::new(&[0..16, 16..32], &[64..80, 80..96]);
    assert_eq!(
        copy.rows,
        vec![BufferRange {
            src: 0,
            dst: 64,
            size: 32
        }]
    );
    assert_eq!(copy.length(), 32);
}

/// Destination rows that overlap (a rotated texture's rows longer than its plane's pitch) keep
/// the later row's bytes, as plain Metal writes them: each earlier row
/// copies only the bytes before the next row's start, the last one its whole row, and the copy
/// writes every byte once. Each row's bytes come from the same offset of its source row.
#[test]
fn overlapping_destination_rows_keep_the_later_rows_bytes() {
    // Three rows of 64 bytes, 16 bytes apart in the destination.
    let src = [0..64, 64..128, 128..192];
    let dst = [1000..1064, 1016..1080, 1032..1096];
    let copy = RowsCopy::new(&src, &dst);
    let mut written = vec![None; 96];
    for row in &copy.rows {
        for k in 0..row.size {
            let at = (row.dst + k - 1000) as usize;
            assert_eq!(written[at], None, "byte {at} written twice");
            written[at] = Some(row.src + k);
        }
    }
    let want: Vec<Option<u64>> = (0..96u64)
        .map(|at| match at {
            0..16 => Some(at),
            16..32 => Some(64 + at - 16),
            _ => Some(128 + at - 32),
        })
        .collect();
    assert_eq!(written, want);
    assert_eq!(copy.dst_span, 1000..1096);
    assert_eq!(copy.src_span, 0..192);
    // Rows apart keep their order and merge as before.
    let copy = RowsCopy::new(&[0..16, 16..32], &[64..80, 80..96]);
    assert_eq!(copy.rows.len(), 1);
}

/// Plain Metal's block order at a `bytesPerRow` of 0 as our probe measured it (an M4
/// under macOS 27.0.1, out of the texture, without the option and with it alike): for each
/// region, the level's row-major index of the block at each buffer position, `None` where the
/// copy left the buffer's bytes as they were. The region's own Morton order, counted from its
/// corner, the x bit lowest, the larger axis's extra bits on top.
#[test]
fn block_order_positions_are_metals_measured_order() {
    // (level width in blocks, the region's first block, its blocks wide and high, the measured
    // buffer positions)
    let measured: [(u64, [u64; 2], [u64; 2], &[Option<u64>]); 5] = [
        (
            4,
            [0, 0],
            [4, 4],
            &[0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15].map(Some),
        ),
        (4, [0, 1], [4, 2], &[4, 5, 8, 9, 6, 7, 10, 11].map(Some)),
        (
            8,
            [2, 1],
            [3, 2],
            &[
                Some(10),
                Some(11),
                Some(18),
                Some(19),
                Some(12),
                None,
                Some(20),
            ],
        ),
        (
            4,
            [2, 4],
            [2, 4],
            &[18, 19, 22, 23, 26, 27, 30, 31].map(Some),
        ),
        (
            8,
            [1, 2],
            [4, 4],
            &[
                17, 18, 25, 26, 19, 20, 27, 28, 33, 34, 41, 42, 35, 36, 43, 44,
            ]
            .map(Some),
        ),
    ];
    for (width, [x0, y0], blocks, order) in measured {
        for (position, index) in order.iter().enumerate() {
            let Some(index) = index else {
                continue;
            };
            let (x, y) = (index % width - x0, index / width - y0);
            assert_eq!(
                block_order_position(x, y, blocks),
                position as u64,
                "{blocks:?} at {x0},{y0}: block {x},{y}"
            );
        }
    }
}

/// Plain Metal's CPU order of a whole PVRTC level as our probe measured it (an M4 under
/// macOS 27.0.1; `replaceRegion` and `getBytes` alike, plain and validated, and the 16 x 16 level):
/// for a level's blocks across and down, the level's row-major index of the block at each position.
/// Z order with the y bit lowest, the larger axis's extra bits on top; the rows the runtime copies
/// through carry it both ways.
#[test]
fn pvrtc_cpu_order_is_metals_measured_order() {
    let measured: [([u64; 2], &[u64]); 7] = [
        ([1, 1], &[0]),
        ([1, 2], &[0, 1]),
        ([2, 2], &[0, 2, 1, 3]),
        ([2, 4], &[0, 2, 1, 3, 4, 6, 5, 7]),
        (
            [4, 4],
            &[0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15],
        ),
        (
            [4, 8],
            &[
                0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15, 16, 20, 17, 21, 24, 28, 25,
                29, 18, 22, 19, 23, 26, 30, 27, 31,
            ],
        ),
        (
            [8, 8],
            &[
                0, 8, 1, 9, 16, 24, 17, 25, 2, 10, 3, 11, 18, 26, 19, 27, 32, 40, 33, 41, 48, 56,
                49, 57, 34, 42, 35, 43, 50, 58, 51, 59, 4, 12, 5, 13, 20, 28, 21, 29, 6, 14, 7, 15,
                22, 30, 23, 31, 36, 44, 37, 45, 52, 60, 53, 61, 38, 46, 39, 47, 54, 62, 55, 63,
            ],
        ),
    ];
    for (blocks, order) in measured {
        let [across, down] = blocks;
        let n = (across * down) as usize;
        assert_eq!(order.len(), n);
        for (position, &index) in order.iter().enumerate() {
            let (x, y) = (index % across, index / across);
            assert_eq!(
                cpu_order_position(x, y, blocks),
                position as u64,
                "{blocks:?}: block {index}"
            );
        }
        // Each 8-byte block of the host bytes starts with its position.
        let host: Vec<u8> = (0..n * 8)
            .map(|i| if i % 8 == 0 { (i / 8) as u8 } else { 0xee })
            .collect();
        let rows = rows_of_cpu_order(&host, blocks);
        let firsts: Vec<u64> = rows
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| u64::from(b[0]))
            .collect();
        for (position, &index) in order.iter().enumerate() {
            assert_eq!(firsts[index as usize], position as u64, "{blocks:?}");
        }
        let mut back = vec![0; n * 8];
        cpu_order_of_rows(&rows, blocks, &mut back);
        assert_eq!(back, host, "{blocks:?}");
    }
}

/// The texture blocks `plan` copies from a buffer of `bytes` (its `VkBuffer`'s, or the staging
/// buffer's the plan fills from it): the first byte of each block's 8, by its block in the
/// image's level, in 4 x 4-texel blocks.
fn copied_blocks(plan: &BufferCopyPlan, bytes: &[u8]) -> std::collections::BTreeMap<[u32; 2], u8> {
    let staged = plan.staging.as_ref().map(|_| stage(plan, bytes));
    let from = staged.as_deref().unwrap_or(bytes);
    let mut out = std::collections::BTreeMap::new();
    for r in &plan.regions {
        let [w, h] = [r.image.size[0].div_ceil(4), r.image.size[1].div_ceil(4)];
        let row = if r.row_length == 0 {
            w
        } else {
            r.row_length / 4
        };
        for j in 0..h {
            for i in 0..w {
                let at = r.buffer_offset as usize + ((j * row + i) * 8) as usize;
                let block = [r.image.origin[0] / 4 + i, r.image.origin[1] / 4 + j];
                assert!(out.insert(block, from[at]).is_none(), "{block:?} twice");
            }
        }
    }
    out
}

/// A PVRTC copy at a `bytesPerRow` of 0 lays each block at its block-order position, from the
/// layout's offset, whole levels and regions of whole blocks alike, through staging where the
/// `VkBuffer`'s offset misses the block, which moves only the positions blocks take; a region
/// off its blocks is planned as the blocks it is rounded to; an offset off the block and a buffer
/// too short are Metal's validation's refusals.
#[test]
fn pvrtc_copies_at_a_pitch_of_0_follow_metals_block_order() {
    let t = tex(PixelFormat::PVRTCRGBA4BPP, TextureType::Type2D, [32, 32, 1]);
    let u = unit(&t, BlitOption::None).unwrap();
    // Each 8-byte block of the Metal buffer starts with its index.
    let buffer: Vec<u8> = (0..4096u32)
        .map(|i| if i % 8 == 0 { (i / 8) as u8 } else { 0xee })
        .collect();
    let plan = |r: TextureRegion, offset: u64, base: u64, length: u64| {
        plan_block_order_copy(&t.record, &r, &u, &layout(offset, 0, 0), length, base)
    };
    for (origin, blocks) in [
        ([0, 0], [8, 8]),
        ([4, 4], [2, 2]),
        ([16, 16], [4, 4]),
        ([8, 4], [3, 2]),
        ([4, 8], [4, 4]),
        ([0, 4], [8, 1]),
        ([4, 0], [1, 5]),
        ([8, 8], [5, 3]),
    ] {
        let r = region([origin[0], origin[1], 0], [blocks[0] * 4, blocks[1] * 4, 1]);
        for (offset, base) in [(0, 0), (16, 0), (8, 4)] {
            let p = plan(r, offset, base, 4096).unwrap();
            assert_eq!(p.staging.is_some(), base % 8 != 0, "{blocks:?} {base}");
            // The Metal buffer starts `base` bytes into its `VkBuffer`.
            let bytes: Vec<u8> = std::iter::repeat_n(0xdd, base as usize)
                .chain(buffer.iter().copied())
                .collect();
            let copied = copied_blocks(&p, &bytes);
            let mut last = 0;
            for y in 0..blocks[1] {
                for x in 0..blocks[0] {
                    let position = block_order_position(x.into(), y.into(), blocks.map(u64::from));
                    last = last.max(position);
                    let block = [origin[0] / 4 + x, origin[1] / 4 + y];
                    let want = ((offset / 8) + position) as u8;
                    assert_eq!(copied.get(&block), Some(&want), "{blocks:?}: {block:?}");
                }
            }
            assert_eq!(copied.len() as u32, blocks[0] * blocks[1]);
            assert_eq!(p.span, offset..offset + (last + 1) * 8);
            // Staging moves only the positions blocks take.
            if let Some(staging) = &p.staging {
                let moved: u64 = staging.rows.iter().map(|r| r.size).sum();
                assert_eq!(moved, u64::from(blocks[0] * blocks[1]) * 8);
            }
        }
    }
    // The 3 x 2 region's positions 5 and 7 are no block's: never read, never written.
    let p = plan(region([8, 4, 0], [12, 8, 1]), 0, 4, 4096).unwrap();
    let staging = p.staging.unwrap();
    let rows: Vec<(u64, u64)> = staging.rows.iter().map(|r| (r.dst, r.size)).collect();
    assert_eq!(rows, [(0, 40), (48, 8)]);
    // A region off its blocks is the blocks it is rounded to, its origin down and its size up,
    // each on its own.
    for (off, on) in [
        (([2, 0], [8, 8]), ([0, 0], [8, 8])),
        (([0, 0], [6, 8]), ([0, 0], [8, 8])),
        (([1, 1], [4, 4]), ([0, 0], [4, 4])),
        (([4, 4], [10, 12]), ([4, 4], [12, 12])),
        (([6, 0], [8, 8]), ([4, 0], [8, 8])),
        (([0, 0], [1, 1]), ([0, 0], [4, 4])),
    ] {
        let off_plan = plan(
            region([off.0[0], off.0[1], 0], [off.1[0], off.1[1], 1]),
            0,
            0,
            4096,
        );
        let on_plan = plan(
            region([on.0[0], on.0[1], 0], [on.1[0], on.1[1], 1]),
            0,
            0,
            4096,
        );
        assert!(off_plan.is_ok(), "{off:?}");
        assert_eq!(off_plan, on_plan, "{off:?} as {on:?}");
    }
    // An offset off the block, and a buffer too short.
    assert_eq!(
        plan(region([0, 0, 0], [8, 8, 1]), 4, 0, 4096),
        Err(CopyProblem::Offset)
    );
    assert_eq!(
        plan(region([8, 4, 0], [12, 8, 1]), 0, 0, 55),
        Err(CopyProblem::BufferRange)
    );
    assert!(plan(region([8, 4, 0], [12, 8, 1]), 0, 0, 56).is_ok());
}

/// Between a compressed format and an uncompressed one whose texel is its block's bytes, a region
/// copy's coordinates are the destination's texels, as our probe on an Apple M3 Max under macOS
/// 27.0.1 read them: out of BC1 into `RG32Uint` an origin of (4, 4) and a size of 2 x 2 copy the
/// blocks from the fifth across and down (texels from 16), into BC1 from `RG32Uint` an origin of
/// (1, 2) and a size of 2 x 2 copy texel (0, 0) alone into the block at (4, 4); ASTC 6x5's
/// blocks likewise. Units of other bytes stay Metal's refusal; a PVRTC format with an
/// uncompressed one, two compressed formats of different extents, and a level copy between a
/// compressed and an uncompressed format, the runtime's gap.
#[test]
fn compressed_and_uncompressed_textures_copy_in_the_destination_s_texels() {
    use PixelFormat as P;
    let t = |pf, w, h| tex(pf, TextureType::Type2D, [w, h, 1]);
    let (bc1, rg32) = (t(P::BC1RGBA, 32, 32), t(P::RG32Uint, 8, 8));
    let out = plan_region_copy(
        &pair(&bc1, &rg32),
        &region([4, 4, 0], [2, 2, 1]),
        0,
        0,
        [1, 2, 0],
    )
    .unwrap();
    assert_eq!(
        (out[0].src.origin, out[0].src.size),
        ([16, 16, 0], [8, 8, 1])
    );
    assert_eq!((out[0].dst.origin, out[0].dst.size), ([1, 2, 0], [2, 2, 1]));
    let into =
        |origin, size, at| plan_region_copy(&pair(&rg32, &bc1), &region(origin, size), 0, 0, at);
    let out = into([1, 2, 0], [2, 2, 1], [4, 4, 0]).unwrap();
    assert_eq!((out[0].src.origin, out[0].src.size), ([0, 0, 0], [1, 1, 1]));
    assert_eq!((out[0].dst.origin, out[0].dst.size), ([4, 4, 0], [4, 4, 1]));
    let out = into([1, 1, 0], [4, 2, 1], [4, 0, 0]).unwrap();
    assert_eq!((out[0].src.origin, out[0].src.size), ([0, 0, 0], [1, 1, 1]));
    assert_eq!((out[0].dst.origin, out[0].dst.size), ([4, 0, 0], [4, 4, 1]));
    let out = into([0, 0, 0], [8, 8, 1], [0, 0, 0]).unwrap();
    assert_eq!((out[0].src.size, out[0].dst.size), ([2, 2, 1], [8, 8, 1]));
    // Past the source's blocks: Metal's refusal.
    assert_eq!(
        plan_region_copy(
            &pair(&bc1, &rg32),
            &region([7, 0, 0], [2, 1, 1]),
            0,
            0,
            [0, 0, 0]
        ),
        Err(CopyRefusal::Invalid(CopyProblem::Region))
    );
    let (astc, rgba32) = (t(P::ASTC6x5LDR, 48, 40), t(P::RGBA32Uint, 8, 8));
    let out = plan_region_copy(
        &pair(&astc, &rgba32),
        &region([6, 5, 0], [2, 2, 1]),
        0,
        0,
        [1, 2, 0],
    )
    .unwrap();
    assert_eq!(
        (out[0].src.origin, out[0].src.size),
        ([36, 25, 0], [12, 10, 1])
    );
    let out = plan_region_copy(
        &pair(&rgba32, &astc),
        &region([1, 2, 0], [2, 2, 1]),
        0,
        0,
        [6, 5, 0],
    )
    .unwrap();
    assert_eq!((out[0].src.origin, out[0].src.size), ([0, 0, 0], [1, 1, 1]));
    assert_eq!((out[0].dst.origin, out[0].dst.size), ([6, 5, 0], [6, 5, 1]));
    let full = region([0, 0, 0], [2, 2, 1]);
    assert_eq!(
        plan_region_copy(&pair(&bc1, &t(P::RGBA8Unorm, 8, 8)), &full, 0, 0, [0, 0, 0]),
        Err(CopyRefusal::Invalid(CopyProblem::Format))
    );
    let gap = |r: Result<Vec<ImageCopyRegion>, CopyRefusal>| matches!(&r, Err(CopyRefusal::Unsupported(u)) if u.feature == Feature::Format);
    let pvrtc = t(P::PVRTCRGBA4BPP, 32, 32);
    assert!(gap(plan_region_copy(
        &pair(&pvrtc, &rg32),
        &full,
        0,
        0,
        [0, 0, 0]
    )));
    assert!(gap(plan_region_copy(
        &pair(&rg32, &pvrtc),
        &full,
        0,
        0,
        [0, 0, 0]
    )));
    let bc7 = t(P::BC7RGBAUnorm, 48, 40);
    assert!(gap(plan_region_copy(
        &pair(&astc, &bc7),
        &full,
        0,
        0,
        [0, 0, 0]
    )));
    assert!(gap(plan_levels_copy(&pair(&bc1, &rg32), 0, 0, 0, 0, 1, 1)));
    let small = t(P::BC1RGBA, 8, 8);
    assert!(gap(plan_whole_copy(&pair(&rg32, &small))));
}

/// A 4:2:2 texture copies into the other order, `RG8Unorm` and `R16Uint` byte for byte (macOS
/// 26.7's rule of one copy unit); a copy into one from another format of its unit is the
/// runtime's refusal (plain Metal writes `0xff` over part of each row), and
/// one between units Metal's.
#[test]
fn four_two_two_texture_copies_take_metals_verdicts() {
    use PixelFormat as P;
    let t = |pf| tex(pf, TextureType::Type2D, [4, 2, 1]);
    let full = region([0, 0, 0], [4, 2, 1]);
    let copy = |a: &Texture, b: &Texture| plan_region_copy(&pair(a, b), &full, 0, 0, [0, 0, 0]);
    for a in [P::GBGR422, P::BGRG422] {
        let pairs = t(a);
        for b in [P::GBGR422, P::BGRG422, P::RG8Unorm, P::R16Uint] {
            let regions = copy(&pairs, &t(b)).unwrap_or_else(|e| panic!("{a:?} into {b:?}: {e:?}"));
            assert_eq!(regions[0].dst.size, [4, 2, 1], "{a:?} into {b:?}");
        }
        for b in [P::RG8Unorm, P::R16Uint] {
            let r = copy(&t(b), &pairs);
            assert!(
                matches!(&r, Err(CopyRefusal::Unsupported(u)) if u.feature == Feature::Format),
                "{b:?} into {a:?}: {r:?}"
            );
        }
        let rgba = t(P::RGBA8Unorm);
        for r in [copy(&pairs, &rgba), copy(&rgba, &pairs)] {
            assert_eq!(r, Err(CopyRefusal::Invalid(CopyProblem::Format)), "{a:?}");
        }
    }
}

/// A `bytesPerRow` whose span of 4 rows wraps to `0x30` bytes past `u64::MAX`: 3 rows of it are
/// `2^64 + 0x20`.
const WRAPS: u64 = 0x5555_5555_5555_5560;

/// A guest's layout values reach the plans as they are, any `u64`. A layout whose rows pass
/// `u64::MAX` lies past every buffer, so it is refused (`BufferRange`) in both directions and on
/// every path: Metal's rules, the rows plain Metal places off them, the aligned path, and many
/// images; never a wrapped offset inside the buffer (the reims-vgpu consumer's report: a 4 KiB
/// buffer, a 16 x 16 `BGRA8Unorm` texture, 4 x 4 texels from the origin).
#[test]
fn layouts_past_u64_max_lie_past_every_buffer() {
    assert_eq!(3u64.wrapping_mul(WRAPS).wrapping_add(16), 0x30);
    let flat = tex(PixelFormat::BGRA8Unorm, TextureType::Type2D, [16, 16, 1]);
    let u = unit(&flat, BlitOption::None).unwrap();
    let r = region([0, 0, 0], [4, 4, 1]);
    for l in [
        // The report's: an offset off the unit (placed rows), a pitch off the unit (placed
        // rows), and a pitch on it whose span passes `u64::MAX`.
        layout(u64::MAX, 64, 0),
        layout(0, u64::MAX, 0),
        layout(0, 1 << 63, 0),
        // The span that wraps into the buffer, on Metal's rules and off them.
        layout(0, WRAPS, 0),
        layout(2, WRAPS, 0),
        layout(0, WRAPS + 1, 0),
        // An offset on the unit whose first row passes `u64::MAX`.
        layout(u64::MAX - 3, 64, 0),
        // Into the texture, the aligned path's pitch of 2^63 (rounded down from 2^63 + 2).
        layout(0, (1 << 63) + 2, 0),
    ] {
        for into_texture in [true, false] {
            assert_eq!(
                plan_buffer_copy(&flat.record, &r, &u, &l, 4096, 0, into_texture),
                Err(CopyProblem::BufferRange),
                "{l:?} into the texture: {into_texture}"
            );
        }
    }
    // Images of a 3D texture whose offsets pass `u64::MAX`: on Metal's rules (a span that
    // wraps, and one that passes it) and off them (`bytesPerImage` off the unit, and a
    // `bytesPerRow` whose image of rows passes `u64::MAX`).
    let volume = tex(PixelFormat::BGRA8Unorm, TextureType::Type3D, [16, 16, 4]);
    let r = region([0, 0, 0], [4, 4, 4]);
    for l in [
        layout(0, 16, WRAPS),
        layout(0, 16, 1 << 63),
        layout(0, 16, u64::MAX),
        layout(0, 1 << 62, 1 << 63),
    ] {
        for into_texture in [true, false] {
            assert_eq!(
                plan_buffer_copy(&volume.record, &r, &u, &l, 4096, 0, into_texture),
                Err(CopyProblem::BufferRange),
                "{l:?} into the texture: {into_texture}"
            );
        }
    }
    // A depth format that keeps Metal's validation's refusals off its rules.
    let depth = tex(PixelFormat::Depth16Unorm, TextureType::Type2D, [16, 16, 1]);
    let du = unit(&depth, BlitOption::None).unwrap();
    let r = region([0, 0, 0], [4, 4, 1]);
    for l in [
        layout(0, 1 << 63, 0),
        layout(0, WRAPS, 0),
        layout(u64::MAX - 1, 8, 0),
    ] {
        for into_texture in [true, false] {
            assert_eq!(
                plan_buffer_copy(&depth.record, &r, &du, &l, 4096, 0, into_texture),
                Err(CopyProblem::BufferRange),
                "{l:?}"
            );
        }
    }
    // A texture over a buffer.
    let (over, texels) = over_buffer(1);
    let ou = unit(&over, BlitOption::None).unwrap();
    let r = region([0, 0, 0], [4, 4, 1]);
    for l in [
        layout(0, 1 << 63, 0),
        layout(0, WRAPS, 0),
        layout(u64::MAX - 3, 16, 0),
    ] {
        assert_eq!(
            plan_rows_copy(&over.record, &r, &ou, &l, 4096, &texels),
            Err(CopyProblem::BufferRange),
            "{l:?}"
        );
    }
    // A PVRTC texture's blocks in Metal's block order, from an offset on the block.
    let pvrtc = tex(PixelFormat::PVRTCRGBA4BPP, TextureType::Type2D, [32, 32, 1]);
    let pu = unit(&pvrtc, BlitOption::None).unwrap();
    let r = region([0, 0, 0], [16, 16, 1]);
    assert_eq!(
        plan_block_order_copy(&pvrtc.record, &r, &pu, &layout(u64::MAX - 7, 0, 0), 4096, 0),
        Err(CopyProblem::BufferRange)
    );
}

/// A copy whose last byte is the buffer's last byte is planned, on Metal's rules and off them,
/// and one a byte further is refused (`BufferRange`): an offset off the unit's rows rounded down
/// as plain Metal places them.
#[test]
fn copies_reach_the_buffer_s_last_byte_and_no_further() {
    let t = tex(PixelFormat::BGRA8Unorm, TextureType::Type2D, [16, 16, 1]);
    let u = unit(&t, BlitOption::None).unwrap();
    let r = region([0, 0, 0], [4, 4, 1]);
    // 3 rows of 64 bytes and one of 16.
    let last = 4096 - (3 * 64 + 16);
    for into_texture in [true, false] {
        for offset in [last, last + 1, last + 3] {
            let l = layout(offset, 64, 0);
            let plan = plan_buffer_copy(&t.record, &r, &u, &l, 4096, 0, into_texture).unwrap();
            let start = offset - offset % 4;
            assert_eq!(plan.span, start..4096, "{offset}");
        }
        for offset in [last + 4, last + 5] {
            let l = layout(offset, 64, 0);
            assert_eq!(
                plan_buffer_copy(&t.record, &r, &u, &l, 4096, 0, into_texture),
                Err(CopyProblem::BufferRange),
                "{offset}"
            );
        }
    }
}

/// A copy of rows cut where a side's bytes pass its texture's tail (`buffer_texture.rs`): every
/// byte in one part, each part's rows on one side of each cut, a row across a cut split there,
/// and a side with no cut whole before it.
#[test]
fn rows_split_at_each_side_s_tail() {
    // Two 256-byte rows from a buffer into a texture whose bytes past 4096 are its tail: the
    // second row crosses it 128 bytes in.
    let copy = RowsCopy::new(&[0..256, 1024..1280], &[3712..3968, 3968..4224]);
    let parts = copy.split(None, Some(4096));
    let rows = |past: bool| -> Vec<BufferRange> {
        let part = parts.iter().find(|(s, d, _)| !*s && *d == past);
        part.map_or(Vec::new(), |(.., copy)| copy.rows.clone())
    };
    assert_eq!(parts.len(), 2);
    assert_eq!(
        rows(false),
        [
            BufferRange {
                src: 0,
                dst: 3712,
                size: 256
            },
            BufferRange {
                src: 1024,
                dst: 3968,
                size: 128
            }
        ]
    );
    assert_eq!(
        rows(true),
        [BufferRange {
            src: 1152,
            dst: 4096,
            size: 128
        }]
    );
    let (.., past) = parts
        .iter()
        .find(|(_, d, _)| *d)
        .expect("a part past the cut");
    assert_eq!(
        (past.src_span.clone(), past.dst_span.clone()),
        (1152..1280, 4096..4224)
    );
    // Both sides cut: a row crossing both cuts at different bytes is three parts, and a row
    // before both joins the first.
    let both =
        RowsCopy::new(&[0..64, 3968..4224], &[64..128, 4032..4288]).split(Some(4096), Some(4096));
    let mut sides: Vec<(bool, bool, u64)> = both
        .iter()
        .map(|(s, d, copy)| (*s, *d, copy.length()))
        .collect();
    sides.sort_unstable();
    assert_eq!(
        sides,
        [(false, false, 128), (false, true, 64), (true, true, 128)]
    );
    // No cut: the copy itself.
    let whole = copy.split(None, None);
    assert_eq!(whole.len(), 1);
    assert_eq!(whole[0].2, copy);
}
