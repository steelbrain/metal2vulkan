//! Scenario `buffer_texture_copies`: `copyFromBuffer:...toTexture:` and
//! `copyFromTexture:...toBuffer:` with row and image pitches.
//!
//! - **Pitches.** For texels of 1, 2, 4, 8 and 16 bytes, packed and sRGB among them: a 2D array
//!   texture of two slices and two levels, written from padded rows at an offset, one slice's
//!   sub-region written again, and every level and slice read back into one buffer with padded
//!   rows, its padding left as it was.
//! - **Metal's rules**, from a probe of our own: offsets, row and image pitches, a compressed
//!   region and a buffer exactly as long as the rows, each copy taken and its texels read back.
//!   The scenario holds valid usage only, since its records are made on this Mac's M3 Max, where
//!   no misuse runs. The copies plain Metal makes of offsets and pitches off
//!   the copy unit, of compressed regions off their blocks and of a combined depth/stencil
//!   texture without a blit option, which the runtime makes too, are the lane tests'
//!   (`runtime_lane::texture_blits`), and so are those of an image pitch off
//!   Metal's rules (`runtime_lane::textures`); the copies Metal's validation
//!   stops and the runtime refuses (a pitch below a row at offset 0, a region past the texture,
//!   a buffer too short) are the runtime's unit tests' (`tests::blit_plan`,
//!   `tests::copy_plan`).

use super::super::api::{Api, Layout, Observation, Region};
use super::steps::{pattern, shared_with, submit};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{BlitOption, Origin, ResourceOptions, Size, TextureDescriptor};

/// The pitch cases' formats and their texel sizes.
const PITCHES: [(&str, PixelFormat, u64); 8] = [
    ("pitches_r8unorm", PixelFormat::R8Unorm, 1),
    ("pitches_rg8unorm", PixelFormat::RG8Unorm, 2),
    ("pitches_rgba8unorm", PixelFormat::RGBA8Unorm, 4),
    ("pitches_rgba8unorm_srgb", PixelFormat::RGBA8UnormSRGB, 4),
    ("pitches_rgb10a2unorm", PixelFormat::RGB10A2Unorm, 4),
    ("pitches_rg11b10float", PixelFormat::RG11B10Float, 4),
    ("pitches_rgba16float", PixelFormat::RGBA16Float, 8),
    ("pitches_rgba32float", PixelFormat::RGBA32Float, 16),
];

/// A probe 3 copy into the whole of a texture of one level, without a blit option: the
/// texture's format, type and extent, the source's layout and length, and the layout its texels
/// are read back in.
struct Rule {
    format: PixelFormat,
    texture_type: TextureType,
    extent: [u64; 3],
    layout: [u64; 3],
    source_length: u64,
    read_back: [u64; 3],
}

/// Probe 3's cases, by name.
fn rule(case: &str) -> Option<Rule> {
    let rgba = |w: u64, h: u64, layout: [u64; 3], read_back: [u64; 3]| Rule {
        format: PixelFormat::RGBA8Unorm,
        texture_type: TextureType::Type2D,
        extent: [w, h, 1],
        layout,
        source_length: 4096,
        read_back,
    };
    let bc1 = |extent: [u64; 3], layout: [u64; 3], read_back| Rule {
        format: PixelFormat::BC1RGBA,
        texture_type: TextureType::Type2D,
        extent,
        layout,
        source_length: 4096,
        read_back,
    };
    Some(match case {
        "cbt.offset4" => rgba(2, 2, [4, 8, 0], [0, 8, 0]),
        "cbt.bpr0.1row" => rgba(2, 1, [0, 0, 0], [0, 8, 0]),
        "cbt.bpi16.2d" => rgba(2, 2, [0, 8, 16], [0, 8, 0]),
        "cbt.3d" => Rule {
            texture_type: TextureType::Type3D,
            extent: [2, 2, 2],
            ..rgba(2, 2, [0, 8, 16], [0, 8, 16])
        },
        "cbt.bc1.6x6" => bc1([6, 6, 1], [0, 16, 0], [0, 16, 0]),
        "cbt.source16" => Rule {
            source_length: 16,
            ..rgba(2, 2, [0, 8, 0], [0, 8, 0])
        },
        _ => return None,
    })
}

pub(super) fn run<A: Api>(api: &A, case: &str, obs: &mut Observation) -> Result<(), String> {
    if let Some(&(_, format, texel)) = PITCHES.iter().find(|(name, ..)| *name == case) {
        return pitches(api, obs, format, texel);
    }
    match rule(case) {
        Some(rule) => metal_rule(api, obs, &rule),
        None => Err(format!("buffer_texture_copies has no case {case}")),
    }
}

fn layout([offset, bytes_per_row, bytes_per_image]: [u64; 3]) -> Layout {
    Layout {
        offset,
        bytes_per_row,
        bytes_per_image,
    }
}

fn region(slice: u64, level: u64, origin: [u64; 3], size: [u64; 3]) -> Region {
    Region {
        slice,
        level,
        origin: Origin::new(origin[0], origin[1], origin[2]),
        size: Size::new(size[0], size[1], size[2]),
    }
}

/// A `Private` texture of one usage, `ShaderRead`.
fn descriptor(
    format: PixelFormat,
    texture_type: TextureType,
    extent: [u64; 3],
    levels: u64,
    slices: u64,
) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::default();
    descriptor.texture_type = texture_type;
    descriptor.pixel_format = format;
    [descriptor.width, descriptor.height, descriptor.depth] = extent;
    descriptor.mipmap_level_count = levels;
    descriptor.array_length = slices;
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    descriptor
}

/// The pitch case of a format with `texel`-byte texels (see the module's documentation).
fn pitches<A: Api>(
    api: &A,
    obs: &mut Observation,
    format: PixelFormat,
    texel: u64,
) -> Result<(), String> {
    let row = 6 * texel;
    let image = 4 * row;
    let Some(queue) = obs.call("queue", || api.queue()) else {
        return Ok(());
    };
    let Some(cb) = obs.call("command buffer", || api.command_buffer(&queue)) else {
        return Ok(());
    };
    let d = descriptor(format, TextureType::Type2DArray, [5, 3, 1], 2, 2);
    let Some(texture) = obs.call("texture", || api.texture(&d)) else {
        return Ok(());
    };
    let Some(rows) = shared_with(api, obs, "rows", &pattern(16 + 2 * image, 1)) else {
        return Ok(());
    };
    let Some(patch) = shared_with(api, obs, "patch", &pattern(6 * texel, 50)) else {
        return Ok(());
    };
    let Some(small) = shared_with(api, obs, "small", &pattern(4 * texel, 90)) else {
        return Ok(());
    };
    let out_length = 40 * texel;
    let Some(out) = shared_with(
        api,
        obs,
        "read-back buffer",
        &vec![0xcd; out_length as usize],
    ) else {
        return Ok(());
    };
    let Some(mut blit) = obs.call("blit", || api.blit(&cb)) else {
        return Ok(());
    };
    let none = BlitOption::None;
    let writes = [
        (
            "slice 0 level 0",
            &rows,
            [16, row, image],
            region(0, 0, [0, 0, 0], [5, 3, 1]),
        ),
        (
            "slice 1 level 0",
            &rows,
            [16 + image, row, image],
            region(1, 0, [0, 0, 0], [5, 3, 1]),
        ),
        (
            "slice 1 level 0 at (1, 1)",
            &patch,
            [0, 3 * texel, 0],
            region(1, 0, [1, 1, 0], [3, 2, 1]),
        ),
        (
            "slice 0 level 1",
            &small,
            [0, 2 * texel, 0],
            region(0, 1, [0, 0, 0], [2, 1, 1]),
        ),
        (
            "slice 1 level 1",
            &small,
            [2 * texel, 2 * texel, 0],
            region(1, 1, [0, 0, 0], [2, 1, 1]),
        ),
    ];
    for (name, from, at, to) in writes {
        let op = format!("write {name}");
        let written = obs.call(&op, || {
            api.copy_to_texture(&mut blit, (from, layout(at)), (&texture, to), none)
        });
        if written.is_none() {
            return Ok(());
        }
    }
    let reads = [
        (
            "slice 0 level 0",
            region(0, 0, [0, 0, 0], [5, 3, 1]),
            [0, 5 * texel, 0],
        ),
        (
            "slice 1 level 0",
            region(1, 0, [0, 0, 0], [5, 3, 1]),
            [16 * texel, 6 * texel, 0],
        ),
        (
            "slice 0 level 1",
            region(0, 1, [0, 0, 0], [2, 1, 1]),
            [34 * texel, 2 * texel, 0],
        ),
        (
            "slice 1 level 1",
            region(1, 1, [0, 0, 0], [2, 1, 1]),
            [36 * texel, 2 * texel, 0],
        ),
    ];
    for (name, from, at) in reads {
        let op = format!("read {name}");
        let read = obs.call(&op, || {
            api.copy_from_texture(&mut blit, (&texture, from), (&out, layout(at)), none)
        });
        if read.is_none() {
            return Ok(());
        }
    }
    if obs.call("end blit", || api.end_blit(blit)).is_none()
        || submit(api, obs, "commit", cb)?.is_none()
    {
        return Ok(());
    }
    obs.bytes("bytes", &api.read(&out));
    Ok(())
}

/// A probe 3 case: the copy into the texture, and its texels read back.
fn metal_rule<A: Api>(api: &A, obs: &mut Observation, rule: &Rule) -> Result<(), String> {
    let Some(queue) = obs.call("queue", || api.queue()) else {
        return Ok(());
    };
    let Some(cb) = obs.call("command buffer", || api.command_buffer(&queue)) else {
        return Ok(());
    };
    let d = descriptor(rule.format, rule.texture_type, rule.extent, 1, 1);
    let Some(texture) = obs.call("texture", || api.texture(&d)) else {
        return Ok(());
    };
    let Some(source) = shared_with(api, obs, "source", &pattern(rule.source_length, 1)) else {
        return Ok(());
    };
    let Some(mut blit) = obs.call("blit", || api.blit(&cb)) else {
        return Ok(());
    };
    let texels = region(0, 0, [0, 0, 0], rule.extent);
    let written = obs.call("write", || {
        api.copy_to_texture(
            &mut blit,
            (&source, layout(rule.layout)),
            (&texture, texels),
            BlitOption::None,
        )
    });
    if written.is_none() {
        return Ok(());
    }
    let Some(out) = shared_with(api, obs, "read-back buffer", &[0xcd; 64]) else {
        return Ok(());
    };
    if obs
        .call("read", || {
            api.copy_from_texture(
                &mut blit,
                (&texture, texels),
                (&out, layout(rule.read_back)),
                BlitOption::None,
            )
        })
        .is_none()
        || obs.call("end blit", || api.end_blit(blit)).is_none()
        || submit(api, obs, "commit", cb)?.is_none()
    {
        return Ok(());
    }
    obs.bytes("bytes", &api.read(&out));
    Ok(())
}
