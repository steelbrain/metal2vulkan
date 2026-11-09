//! Metal's color fills (`BlitCommandEncoder::fill_texture_color`), their placement in every texture
//! type, and `fill_buffer_pattern4`, on the device the cfg selects, under the validation layer,
//! against Metal's bytes.
//!
//! Metal's answers: an Apple M4 under macOS 27.0.1, our own probe `fil.m` (no guest data):
//!
//! - **Texels.** Into a private 128 x 64 texture of each of 70 formats, one texel a color: the
//!   probe's first 512 sweep colors (its 68 specials in every channel, then NaNs, specials and
//!   random values; the generator rebuilt here) and its 250 NaN and edge colors. Metal's answer
//!   is FNV-1a of the texels' bytes in color order, for each list. A format the device holds no
//!   texture of is skipped, with a `COLOR-FILLS-SKIPPED` line.
//! - **Placement.** The probe's 31 byte fills of the pattern `11 22 33 44` into an
//!   `RGBA8Unorm` texture of each type (2D, 2D array, 3D, cube, cube array, 1D, 1D array) at a
//!   level, slice and region each, every byte of every level and slice first filled `cd`: Metal's
//!   answer is the SHA-256 of the probe's dump (per level and slice, `[level, slice, texels, 0]`
//!   as 32-bit words, then the bytes).
//! - **`fillBuffer:range:pattern4:`** over 10 ranges of a 256-byte buffer filled `cd`, the pattern
//!   `0x04030201`: the SHA-256 of each range's `[start, length]` words and the buffer's bytes.
//!
//! - **Packed in another format** (our probe `fil5.m` on an Apple M3 Max under macOS 26.7 and an M4
//!   under macOS 27.0.1; `twins.m` on an M3 Max under macOS 27.0.1): 8 colors packed by
//!   `pixelFormat:` in each extended-range `_sRGB` format into `RGBA32` textures first set to
//!   `0x5a`, every fill in one encoder, against Metal's texels (alike one fill an encoder and one
//!   a process); then one packed in each padded 10-bit format, in which Metal packs no color.
//!
//! The packing is the runtime's (`fill_value.rs`), checked against every measured texel without a
//! device (`tests::fill_value`); this lane checks that each format's texture stores it as Metal's
//! does, and that the fills land where Metal's do. A difference prints a `COLOR-FILLS-DIFFERS`
//! line.

use super::degrade_notices::counts;
use super::degrade_notices::{assert_once, forced, shared_in_system_memory, UNPACKED_FILL};
use super::resolves::{lane, quiet, Lane};
use super::{buffer_with, read_buffer, WAIT};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, Degradation, Origin, Region, ResourceOptions, Size, Texture,
    TextureDescriptor,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

/// The probe's (`fil.m`) special values, the first colors of its sweep, every channel alike.
const SPECIALS: [f64; 68] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.0,
    -2.0,
    0.25,
    0.75,
    1.0 / 3.0,
    1e-3,
    -1e-3,
    f64::INFINITY,
    f64::NEG_INFINITY,
    5e-324,
    -5e-324,
    1e-310,
    1.401298464324817e-45,
    1.1754943508222875e-38,
    5.960464477539063e-08,
    2.9802322387695312e-08,
    4.470348358154297e-08,
    6.103515625e-05,
    65504.0,
    65519.99999999999,
    65520.0,
    65536.0,
    1e10,
    -1e10,
    3.4028234663852886e38,
    3.4028235677973366e38,
    1e39,
    -1e39,
    2147483647.0,
    2147483648.0,
    -2147483648.0,
    -2147483649.0,
    4294967295.0,
    4294967295.5,
    4294967296.0,
    255.5,
    127.5,
    -128.5,
    32767.5,
    65535.5,
    0.5 / 255.0,
    1.5 / 255.0,
    1.0 / 510.0,
    384.0 / 510.0,
    894.0 / 510.0,
    -384.0 / 510.0,
    1.2529411764705882,
    -0.7529411764705882,
    0.5 / 1023.0,
    1.5 / 1023.0,
    0.5 / 65535.0,
    64512.0,
    0.04045,
    0.0031308,
    1e-300,
    1e300,
    65408.0,
    65409.0,
    65472.0,
    4.0,
    15.5,
    31.5,
];

/// The probe's sweep: 4,096 colors, the first 68 [`SPECIALS`] in every channel, the rest each
/// channel a NaN of five patterns (1 in 7), a special (3 in 7) or a random value, from its
/// xorshift generator (its build fused the random values' multiply-adds).
fn sweep_colors() -> Vec<[f64; 4]> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    const NANS: [u64; 5] = [
        0x7ff8_0000_0000_0000,
        0xfff8_0000_0000_0000,
        0x7ff0_0000_0000_0001,
        0x7ff8_0000_0000_0123,
        0xfff0_0000_0000_0001,
    ];
    let mut colors: Vec<[f64; 4]> = (0..4096)
        .map(|_| {
            [0; 4].map(|_| {
                let r = next();
                if r % 7 == 0 {
                    return f64::from_bits(NANS[((r >> 8) % 5) as usize]);
                }
                if r % 7 < 4 {
                    return SPECIALS[((r >> 8) % SPECIALS.len() as u64) as usize];
                }
                let u = (next() >> 11) as f64 / 9007199254740992.0;
                match (r >> 20) % 4 {
                    0 => 5.0f64.mul_add(u, -2.0),
                    1 => u,
                    2 => 140000.0f64.mul_add(u, -70000.0),
                    _ => u * 2f64.powi(((r >> 32) % 160) as i32 - 80),
                }
            })
        })
        .collect();
    for (color, &special) in colors.iter_mut().zip(&SPECIALS) {
        *color = [special; 4];
    }
    colors
}

/// The probe's NaN and edge colors: each of 24 NaNs and 26 doubles at rounding
/// edges in every channel, then in each channel apart over (0.25, 0.5, 0.75, 1).
fn edge_colors() -> Vec<[f64; 4]> {
    const NAN_BITS: [u64; 24] = [
        0x7ff0_0000_0000_0001,
        0x7ff0_0000_2000_0000,
        0x7ff0_0000_4000_0000,
        0x7ff0_0400_0000_0000,
        0x7ff2_0000_0000_0000,
        0x7ff4_0000_0000_0000,
        0x7ff7_ffff_ffff_ffff,
        0x7ff0_2000_0000_0000,
        0x7ff8_0000_0000_0000,
        0x7ff8_0000_0000_0001,
        0x7ff8_0000_2000_0000,
        0x7ff8_0400_0000_0000,
        0x7ff8_2000_0000_0000,
        0x7ffa_0000_0000_0000,
        0x7ffc_0000_0000_0000,
        0x7fff_ffff_ffff_ffff,
        0xfff0_0000_0000_0001,
        0xfff4_0000_0000_0000,
        0xfff8_0000_0000_0000,
        0xfffc_0000_0000_0000,
        0xfff8_0400_0000_0000,
        0xffff_ffff_ffff_ffff,
        0x7ff0_0000_0010_0000,
        0x7ff8_0000_0010_0000,
    ];
    let p = |e: i32| 2f64.powi(e);
    let extra = [
        1.0 + p(-9) - p(-40),
        1.0 + p(-9) + p(-40),
        2.0 - p(-40),
        2.0 - p(-30),
        1.0 + p(-9),
        65408.0 + p(-30),
        65407.99999,
        65471.99999,
        65472.0,
        65472.0 + p(-30),
        p(-16) - p(-60),
        p(-17) + p(-26) - p(-60),
        p(-24) * 1.5 - p(-70),
        2147483647.5,
        2147483647.25,
        2147483648.5,
        4294967294.5,
        4294967295.25,
        4294967295.75,
        2147483647.0,
        -2147483648.5,
        -2147483648.25,
        -2147483647.5,
        0.5 - p(-40),
        0.5 + p(-40),
        1.0 / 3.0 + p(-35),
    ];
    let base = [0.25, 0.5, 0.75, 1.0];
    let values = NAN_BITS.iter().map(|&b| f64::from_bits(b)).chain(extra);
    let mut out = Vec::new();
    for v in values {
        out.push([v; 4]);
        for channel in 0..4 {
            let mut color = base;
            color[channel] = v;
            out.push(color);
        }
    }
    out
}

/// FNV-1a (64-bit) of `bytes`.
fn fnv(bytes: impl Iterator<Item = u8>) -> u64 {
    bytes.fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Metal's texels: each format's raw value and texel bytes, and FNV-1a of its texels of the
/// first 512 sweep colors and of the 250 NaN and edge colors.
const METAL: [(u32, u64, u64, u64); 70] = [
    (1, 1, 0x7480cb8d2fff4810, 0x4ddf512a557ec47b),
    (10, 1, 0x9d85de351e6e548f, 0x35a785406c3be905),
    (11, 1, 0x100063001991345e, 0x70d2e5a864890355),
    (12, 1, 0x0c2367859b16759d, 0xcd58634f4cb98c59),
    (13, 1, 0x6aa9518400d893db, 0x45af9776a4ee8641),
    (14, 1, 0xc9d9264b1d537825, 0xd32dcb82dde0a241),
    (20, 2, 0x981cb2ef498226c8, 0x21c38ea5c5aefd8d),
    (22, 2, 0x1dff8f61c87376cd, 0x939028f6625680e9),
    (23, 2, 0xe926c2bf4826de1b, 0xda4ee275f2659bc5),
    (24, 2, 0x936cfb5e3937f4da, 0xfd4f0ab5d5cef731),
    (25, 2, 0x1476122bcce7ea4e, 0x1cd2cf9cceaaf1bd),
    (26, 2, 0xef6ea982f0b83a37, 0x3d85df4cbe15e74d),
    (30, 2, 0x959cd936dbaa6099, 0xeac9a232cf46b9fd),
    (31, 2, 0xb03756ff9192aaab, 0xe00f64472d61c9f9),
    (32, 2, 0xbdb61e41a0257065, 0xa59a2b70067f495d),
    (33, 2, 0x06b7cacb82dcf1b6, 0x5df7f2d7719f1b9d),
    (34, 2, 0x806995acdaa8e799, 0x6a407230d86bca9d),
    (40, 2, 0x43d0fe3cc1d8796c, 0x4bfc32556ccb7295),
    (41, 2, 0x4545df1776d35301, 0x060ef4d3654288ed),
    (42, 2, 0xaf9b216534c2f54e, 0x7e2c4aad77c4334d),
    (43, 2, 0x4cd9cbe0e85c8177, 0x883f9e1d00751d6d),
    (53, 4, 0x3914f96321881320, 0xfa7537c714de0415),
    (54, 4, 0xb380c6cf689aa158, 0x9342ae4c6475907d),
    (55, 4, 0xaf9ff6fbeac5e14e, 0xe98a468c36e76fa5),
    (60, 4, 0xf27db468e5b87c4c, 0x0293d80143afb905),
    (62, 4, 0xcd41941c398ceb9e, 0x4749d441f3b6f99d),
    (63, 4, 0x94f5a02fa4cd8eaf, 0x3e385ea248968325),
    (64, 4, 0x559e4b7ede4e0ebd, 0xc5b22f00d0ab7a8d),
    (65, 4, 0x88d540e633e0ccbf, 0x0706281acaca526d),
    (66, 4, 0xbdf1ca4e90e878dd, 0xf242e0c7ce7e93f5),
    (70, 4, 0x42f825bb43474e24, 0xc7b16b8fae3ed2c1),
    (71, 4, 0x8a55a22a3613c4f4, 0xfcd316d7380bfe75),
    (72, 4, 0x23835bf59364e6a9, 0xda8df0edbe8ad1f9),
    (73, 4, 0xacb23dab19f81109, 0x2342da747115d435),
    (74, 4, 0x3abeb07d5b669fea, 0x81cf9d0e7b361135),
    (80, 4, 0x080d4cca1942bf94, 0x2ef6e17cb0c9a075),
    (81, 4, 0x542a0475a5132f0c, 0xfee41c37c4d929dd),
    (84, 4, 0x218e3ccc6e2ec2e2, 0xf18455f9fb6ac575),
    (90, 4, 0xa910fc747fa9db92, 0x873cd9fd7688870d),
    (91, 4, 0xbcafb94351fec5fc, 0xab0f9376948ce8ed),
    (92, 4, 0xc3460f26c8e499ad, 0xbb0e96bbde45bb51),
    (93, 4, 0x44084cddeeed1ef9, 0xbea8b5dff91f39aa),
    (94, 4, 0xbd04bb15cb6fa871, 0xa6a3991745e3a99d),
    (103, 8, 0xdacd0f4e1ec8b3db, 0x5b133a9ecb31a9e5),
    (104, 8, 0x08de33834cedd36c, 0xb3b2ed3e0907d335),
    (105, 8, 0x411a32b530735507, 0x17c92e6e79b9badd),
    (110, 8, 0xb7e9749f20e75789, 0x2babf1fa9b2d888d),
    (112, 8, 0x8e9ba87543af4feb, 0x1054b4be9c72d1dd),
    (113, 8, 0xc593ad1cb456d658, 0xc2af3fe24a976865),
    (114, 8, 0x4d35141ac6ef6691, 0x5d3ac3ba9ecada55),
    (115, 8, 0xc224c26ad3812af3, 0xd74cbcbad1b16bdd),
    (116, 8, 0x2b763858193410ec, 0x3090174849baa165),
    (123, 16, 0x7a9d86b0ae58d021, 0x6f1b190dbc2d11a5),
    (124, 16, 0x3e4b8e0cc354bfee, 0xb2bc064158c09e45),
    (125, 16, 0x1b0a60d07f2f2aad, 0x22f2f04fd290d0b5),
    (300, 4, 0x63c54f3130201a73, 0x7335203d08237579),
    (301, 4, 0xd4077a88cc9857fb, 0x96d15b39ff2001d5),
    (302, 4, 0xc0f601f3aa606fc5, 0x5e684e3b6f57626d),
    (303, 4, 0x4af59248c0918987, 0x4de9d52cfd5cba05),
    (304, 4, 0x58fe1f25b823702f, 0xb92cb053bda75805),
    (313, 8, 0x6f7921eead9572c9, 0x45af17661c96c7b5),
    (314, 8, 0x277c51150243297d, 0xc10455cbf77eab81),
    (315, 8, 0xc224c26ad3812af3, 0xd74cbcbad1b16bdd),
    (323, 16, 0xc9a778a2e04e5f30, 0x5bbfda880fac6645),
    (324, 16, 0xf0b93f9459b1db4d, 0x4f0ee3ad91a1c0bd),
    (325, 16, 0x1b0a60d07f2f2aad, 0x22f2f04fd290d0b5),
    (552, 8, 0x455966d194fe27fb, 0x173f0630d9d0e345),
    (553, 8, 0x7965cc6ab57b107e, 0x1a3a41ef34857ca5),
    (554, 4, 0x801b060b58427093, 0x7c1cd6d137e68489),
    (555, 4, 0x9077ac5b22f6d82a, 0xe5cfd31ec7cf1d25),
];

/// The probes' textures' usage, `ShaderRead | PixelFormatView`, which keeps Metal from
/// compressing them: Metal's blit into or out of a compressed `RG11B10Float`, `RGB9E5Float` or
/// `BGRA10_XR` texture rewrites its texels (`compressed_blit.rs`), which the probes' texels leave
/// out.
const USAGE: TextureUsage = TextureUsage::SHADER_READ.union(TextureUsage::PIXEL_FORMAT_VIEW);

/// The texture of the texel lists: 128 x 64.
const WIDTH: u64 = 128;
const HEIGHT: u64 = 64;

/// The bytes of each texel of `texture` (`texel` bytes each), the first `count` in row order.
fn texels(lane: &Lane, texture: &Texture, texel: u64, count: u64) -> Vec<u8> {
    let length = WIDTH * HEIGHT * texel;
    let back = buffer_with(&lane.device, &vec![0xab; length as usize]).unwrap();
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(WIDTH, HEIGHT, 1),
        &back,
        0,
        WIDTH * texel,
        length,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    read_buffer(&back, 0, count * texel).unwrap()
}

/// Each color of `colors` filled into its own texel of a new texture of `pf`, in row order, packed
/// in `packed_in` (`pixelFormat:`; a plain fill without it): the texels' bytes, or `None` where
/// the device holds no such texture. `each` runs after each fill is encoded.
fn filled(
    lane: &Lane,
    (pf, texel): (PixelFormat, u64),
    colors: &[[f64; 4]],
    packed_in: Option<PixelFormat>,
    mut each: impl FnMut(usize),
) -> Option<Vec<u8>> {
    let mut descriptor = TextureDescriptor::texture_2d(pf, WIDTH, HEIGHT, false);
    descriptor.usage = USAGE;
    descriptor.resource_options = ResourceOptions::private();
    let texture = lane.device.new_texture(&descriptor).ok()?;
    for (batch, chunk) in colors.chunks(256).enumerate() {
        let command_buffer = lane.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for (k, &[r, g, b, a]) in chunk.iter().enumerate() {
            let i = 256 * batch + k;
            let region = Region::new_2d(i as u64 % WIDTH, i as u64 / WIDTH, 1, 1);
            let color = ClearColor::new(r, g, b, a);
            blit.fill_texture_color(&texture, 0, 0, region, color, packed_in)
                .unwrap_or_else(|e| panic!("{pf:?}: a fill: {e}"));
            each(i);
        }
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }
    Some(texels(lane, &texture, texel, colors.len() as u64))
}

/// Each color of `colors` filled plain (no `pixelFormat:`) into the whole of its own slice of new
/// 2D arrays of 1 x 1 texels of `pf` (256 slices each): a fill of a whole subresource, which makes
/// no image and takes none from the recycler (asserted, [`images_of`]). The texels' bytes, in
/// color order; `each` runs after each fill is encoded.
fn filled_whole(
    lane: &Lane,
    (pf, texel): (PixelFormat, u64),
    colors: &[[f64; 4]],
    mut each: impl FnMut(usize),
) -> Vec<u8> {
    let name = lane.device.name().to_owned();
    let mut bytes = Vec::with_capacity(colors.len() * texel as usize);
    for (batch, chunk) in colors.chunks(256).enumerate() {
        let mut descriptor = TextureDescriptor::texture_2d(pf, 1, 1, false);
        descriptor.texture_type = TextureType::Type2DArray;
        descriptor.array_length = chunk.len() as u64;
        descriptor.usage = USAGE;
        descriptor.resource_options = ResourceOptions::private();
        let slices = lane.device.new_texture(&descriptor).unwrap();
        let taken = images_of(lane, |blit| {
            for (slice, &[r, g, b, a]) in chunk.iter().enumerate() {
                let (color, region) = (ClearColor::new(r, g, b, a), Region::new_2d(0, 0, 1, 1));
                blit.fill_texture_color(&slices, 0, slice as u64, region, color, None)
                    .unwrap_or_else(|e| panic!("{pf:?}: a fill: {e}"));
                each(256 * batch + slice);
            }
        });
        assert_eq!(
            taken, 0,
            "{name}: {pf:?}: fills of whole slices made or took an image"
        );
        let length = chunk.len() as u64 * texel;
        let back = buffer_with(&lane.device, &vec![0xab; length as usize]).unwrap();
        let command_buffer = lane.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for slice in 0..chunk.len() as u64 {
            blit.copy_from_texture_to_buffer(
                &slices,
                slice,
                0,
                Origin::new(0, 0, 0),
                Size::new(1, 1, 1),
                &back,
                slice * texel,
                texel,
                texel,
                BlitOption::None,
            )
            .unwrap();
        }
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        bytes.extend(read_buffer(&back, 0, length).unwrap());
    }
    bytes
}

/// The SHA-256 of the colors the probes filled with: the sweep's first 512 and the edges, each
/// color's four `f64` bits little-endian.
const SWEEP_SHA: &str = "8716c326453d2df7c2cbbc815322c70c718dba866028dafb2f0db04af4109d9d";
const EDGES_SHA: &str = "e89a983e61ef13b541eb879e3c7a0888886a5b5f8a7fa234279947e8948b5b12";

/// The inputs are the probes': both color lists, before any format claims Metal's texels of them.
#[test]
fn the_inputs_are_the_probes() {
    let bits = |colors: &[[f64; 4]]| -> Vec<u8> {
        colors
            .iter()
            .flat_map(|c| c.iter().flat_map(|v| v.to_bits().to_le_bytes()))
            .collect()
    };
    assert_eq!(
        sha256(&bits(&sweep_colors()[..512])),
        SWEEP_SHA,
        "the sweep"
    );
    assert_eq!(sha256(&bits(&edge_colors())), EDGES_SHA, "the edges");
}

/// Every format's texels of the probes' colors packed in the texture's own format by
/// `pixelFormat:` are Metal's: the packing the runtime makes on the host in every format
/// (`fill_value.rs`), the bytes Metal's plain fill writes too.
#[test]
fn color_fills_write_metal_s_texels() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let sweep = &sweep_colors()[..512];
    let edges = edge_colors();
    assert_eq!(edges.len(), 250);
    let mut wrong = Vec::new();
    for (raw, texel, swept, edged) in METAL {
        let pf = PixelFormat::try_from(raw).unwrap();
        let (Some(a), Some(b)) = (
            filled(&lane, (pf, texel), sweep, Some(pf), |_| {}),
            filled(&lane, (pf, texel), &edges, Some(pf), |_| {}),
        ) else {
            println!("COLOR-FILLS-SKIPPED {name}: {pf:?}: the device holds no such texture");
            continue;
        };
        let (a, b) = (fnv(a.into_iter()), fnv(b.into_iter()));
        if (a, b) != (swept, edged) {
            wrong.push(format!(
                "{pf:?}: sweep {a:#018x} (Metal's {swept:#018x}), edges {b:#018x} (Metal's \
                 {edged:#018x})"
            ));
        }
    }
    for line in &wrong {
        println!("COLOR-FILLS-DIFFERS {name}: {line}");
    }
    assert!(
        wrong.is_empty(),
        "{} formats differ from Metal's",
        wrong.len()
    );
    quiet(lane, &messages);
}

/// A plain color fill (no `pixelFormat:`) is the device's clear of the color where the device
/// renders the texture's format and the format's fill converts a `float` (`blit_fill.rs`; packed on
/// the host elsewhere): of the probes' colors in every format, each texel is Metal's (the fill
/// packed in the texture's own format, [`color_fills_write_metal_s_texels`]'s answer), or within
/// one code of it in each field where its fill announced `device-color-conversion` (a shared
/// exponent's within one step of the coarser exponent), or a NaN for Metal's NaN where it
/// announced `device-nan-encoding`; both are asked for at each fill. What stands apart is printed,
/// a measurement. Each color is filled twice: into a texel of a larger level (the copy of a tile
/// the device cleared), and into the whole of a 1 x 1 slice (the device's clear of the texture's
/// own image, which makes no image, [`filled_whole`]); the device's clear may store another code
/// in each (MoltenVK's of a 1 x 1 image and of a larger one), both held to Metal's alike. An
/// `RGBA8Unorm_sRGB` fill announces `device-color-conversion` (its color channels inside
/// `(0, 1)`), which the host's packing never does: the device's clear ran. How many fills
/// announced `device-color-conversion` is printed (`COLOR-FILLS-ANNOUNCED`).
#[test]
fn plain_color_fills_are_the_device_s_clear() {
    let (messages, _cache, lane) = super::resolves::lane_with(|options| {
        options.announce_each = [
            Degradation::DeviceColorConversion,
            Degradation::DeviceNanEncoding,
        ]
        .into_iter()
        .collect();
    });
    let name = lane.device.name().to_owned();
    let colors: Vec<[f64; 4]> = sweep_colors()[..512]
        .iter()
        .copied()
        .chain(edge_colors())
        .collect();
    let mut wrong = Vec::new();
    let mut srgb_converted = false;
    let (mut fills, mut announced) = (0usize, 0usize);
    for (raw, texel, _, _) in METAL {
        let pf = PixelFormat::try_from(raw).unwrap();
        let Some(metal) = filled(&lane, (pf, texel), &colors, Some(pf), |_| {}) else {
            println!("COLOR-FILLS-SKIPPED {name}: {pf:?}: the device holds no such texture");
            continue;
        };
        for whole in [false, true] {
            let placed = if whole {
                "whole 1 x 1 slices"
            } else {
                "texels of a level"
            };
            let _ = messages.take_notices();
            let mut notes = vec![BTreeSet::new(); colors.len()];
            let plain = {
                let note = |i: usize| {
                    notes[i] = counts(&messages.take_notices()).into_keys().collect();
                };
                match whole {
                    false => filled(&lane, (pf, texel), &colors, None, note).unwrap(),
                    true => filled_whole(&lane, (pf, texel), &colors, note),
                }
            };
            let (mut apart, mut nans) = (0usize, 0usize);
            let mut examples = Vec::new();
            let bytes = texel as usize;
            for (i, (got, want)) in plain.chunks(bytes).zip(metal.chunks(bytes)).enumerate() {
                let converted = notes[i].contains(&Degradation::DeviceColorConversion);
                let nan = notes[i].contains(&Degradation::DeviceNanEncoding);
                fills += 1;
                announced += usize::from(converted);
                srgb_converted |= pf == PixelFormat::RGBA8UnormSRGB && converted;
                if got == want {
                    continue;
                }
                match fields_apart(pf, got, want) {
                    Some(Apart { codes, nans: n }) if (codes == 0 || converted) && (!n || nan) => {
                        apart += usize::from(codes > 0);
                        nans += usize::from(n);
                        if examples.len() < 3 {
                            examples
                                .push(format!("{:?}: {got:02x?}, Metal's {want:02x?}", colors[i]));
                        }
                    }
                    _ => wrong.push(format!(
                        "{pf:?} ({placed}) {:?}: {got:02x?}, Metal's {want:02x?}, announced {:?}",
                        colors[i], notes[i]
                    )),
                }
            }
            if apart > 0 || nans > 0 {
                println!(
                    "COLOR-FILLS-DEVICE {name}: {pf:?} ({placed}): {apart} of {} texels a code \
                     from Metal's, {nans} a NaN the device's: {}",
                    colors.len(),
                    examples.join("; ")
                );
            }
        }
    }
    println!(
        "COLOR-FILLS-ANNOUNCED {name}: {announced} of {fills} plain fills announced \
         device-color-conversion"
    );
    for line in &wrong {
        println!("COLOR-FILLS-DIFFERS {name}: {line}");
    }
    assert!(
        wrong.is_empty(),
        "{} plain fills differ from Metal's past what they announce",
        wrong.len()
    );
    assert!(
        srgb_converted,
        "no RGBA8Unorm_sRGB fill announced device-color-conversion on {name}"
    );
    quiet(lane, &messages);
}

/// How a plain fill's texel stands from Metal's ([`fields_apart`]): how many fields stand a code
/// from Metal's, and whether one is a NaN for Metal's NaN.
pub(super) struct Apart {
    pub(super) codes: u32,
    pub(super) nans: bool,
}

/// How texel `got` of `pf` stands from Metal's `want` where each field is Metal's, a code (a
/// float's adjacent value) from it, or a half or small float's NaN for Metal's NaN; `None` where a
/// field stands further, or for a format read byte for byte. A shared exponent's texel stands a
/// code from Metal's where each channel's value is within one step of the coarser exponent's.
pub(super) fn fields_apart(pf: PixelFormat, got: &[u8], want: &[u8]) -> Option<Apart> {
    use PixelFormat as P;
    let word = |bytes: &[u8]| {
        bytes
            .iter()
            .rev()
            .fold(0u128, |w, &b| (w << 8) | u128::from(b))
    };
    if pf == P::RGB9E5Float {
        let values = |w: u128| {
            let exponent = ((w >> 27) & 0x1f) as i32;
            let step = 2f64.powi(exponent - 15 - 9);
            ([0, 9, 18].map(|at| ((w >> at) & 0x1ff) as f64 * step), step)
        };
        let ((a, sa), (b, sb)) = (values(word(got)), values(word(want)));
        let step = sa.max(sb);
        let near = a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= step);
        return near.then_some(Apart {
            codes: 1,
            nans: false,
        });
    }
    // Each field's bits from the texel's lowest, and how it reads: 0 unsigned, 1 signed, 2 a
    // float (its exponent's bits).
    let (widths, kind, exponent): (&[u32], u8, u32) = match pf {
        P::A8Unorm | P::R8Unorm | P::R8UnormSRGB => (&[8], 0, 0),
        P::R8Snorm => (&[8], 1, 0),
        P::RG8Unorm | P::RG8UnormSRGB => (&[8, 8], 0, 0),
        P::RG8Snorm => (&[8, 8], 1, 0),
        P::RGBA8Unorm
        | P::RGBA8UnormSRGB
        | P::BGRA8Unorm
        | P::BGRA8UnormSRGB
        | P::ARGB8Unorm
        | P::ARGB8UnormSRGB
        | P::RGBX8Unorm
        | P::RGBX8UnormSRGB => (&[8; 4], 0, 0),
        P::RGBA8Snorm | P::RGBX8Snorm => (&[8; 4], 1, 0),
        P::R16Unorm => (&[16], 0, 0),
        P::R16Snorm => (&[16], 1, 0),
        P::RG16Unorm => (&[16; 2], 0, 0),
        P::RG16Snorm => (&[16; 2], 1, 0),
        P::RGBA16Unorm => (&[16; 4], 0, 0),
        P::RGBA16Snorm => (&[16; 4], 1, 0),
        P::R16Float => (&[16], 2, 5),
        P::RG16Float => (&[16; 2], 2, 5),
        P::RGBA16Float | P::RGBX16Float => (&[16; 4], 2, 5),
        P::R32Float => (&[32], 2, 8),
        P::RG32Float => (&[32; 2], 2, 8),
        P::RGBA32Float | P::RGBX32Float => (&[32; 4], 2, 8),
        P::RGB10A2Unorm | P::BGR10A2Unorm => (&[10, 10, 10, 2], 0, 0),
        P::RG11B10Float => (&[11, 11, 10], 2, 5),
        P::B5G6R5Unorm => (&[5, 6, 5], 0, 0),
        P::A1BGR5Unorm => (&[1, 5, 5, 5], 0, 0),
        P::BGR5A1Unorm => (&[5, 5, 5, 1], 0, 0),
        P::ABGR4Unorm | P::BGRA4Unorm => (&[4; 4], 0, 0),
        _ => return None,
    };
    let (g, w) = (word(got), word(want));
    let mut apart = Apart {
        codes: 0,
        nans: false,
    };
    let mut at = 0;
    for &bits in widths {
        let mask = (1u128 << bits) - 1;
        let (a, b) = ((g >> at) & mask, (w >> at) & mask);
        at += bits;
        let signed = |v: u128| ((v as i128) << (128 - bits)) >> (128 - bits);
        let distance = match kind {
            1 => signed(a).abs_diff(signed(b)),
            _ => a.abs_diff(b),
        };
        // A float field's NaN: its exponent all ones and a mantissa (below a sign bit where the
        // field has one: a half's or a float's).
        let mantissa = bits - exponent - u32::from(bits >= 16);
        let nan = |v: u128| {
            kind == 2
                && (v >> mantissa) & ((1 << exponent) - 1) == (1 << exponent) - 1
                && v & ((1 << mantissa) - 1) != 0
        };
        if distance == 0 {
            continue;
        }
        if nan(a) && nan(b) {
            apart.nans = true;
        } else if distance == 1 {
            apart.codes += 1;
        } else {
            return None;
        }
    }
    Some(apart)
}

/// The probe's placement cases (`fil.m`'s `place`): name, texture type, width, height, depth,
/// array length, levels, the fill's level and slice, and its region's origin and size.
type Place = (
    &'static str,
    TextureType,
    [u64; 5],
    [u64; 2],
    [u64; 3],
    [u64; 3],
);

const PLACES: [Place; 31] = {
    use TextureType::*;
    [
        (
            "2d-base",
            Type2D,
            [16, 16, 1, 1, 3],
            [0, 0],
            [2, 3, 0],
            [5, 4, 1],
        ),
        (
            "2d-slice1",
            Type2D,
            [16, 16, 1, 1, 3],
            [0, 1],
            [2, 3, 0],
            [5, 4, 1],
        ),
        (
            "2d-slice5",
            Type2D,
            [16, 16, 1, 1, 3],
            [0, 5],
            [2, 3, 0],
            [5, 4, 1],
        ),
        (
            "2d-level1",
            Type2D,
            [16, 16, 1, 1, 3],
            [1, 0],
            [1, 1, 0],
            [4, 4, 1],
        ),
        (
            "2d-level3",
            Type2D,
            [16, 16, 1, 1, 3],
            [3, 0],
            [0, 0, 0],
            [1, 1, 1],
        ),
        (
            "2d-z1",
            Type2D,
            [16, 16, 1, 1, 3],
            [0, 0],
            [2, 3, 1],
            [5, 4, 1],
        ),
        (
            "2d-depth3",
            Type2D,
            [16, 16, 1, 1, 3],
            [0, 0],
            [2, 3, 0],
            [5, 4, 3],
        ),
        (
            "2d-past",
            Type2D,
            [16, 16, 1, 1, 3],
            [0, 0],
            [12, 13, 0],
            [8, 8, 1],
        ),
        (
            "2d-level1-past",
            Type2D,
            [16, 16, 1, 1, 3],
            [1, 0],
            [6, 6, 0],
            [8, 8, 1],
        ),
        (
            "2da-slice1",
            Type2DArray,
            [16, 16, 1, 3, 1],
            [0, 1],
            [2, 3, 0],
            [5, 4, 1],
        ),
        (
            "2da-slice3",
            Type2DArray,
            [16, 16, 1, 3, 1],
            [0, 3],
            [2, 3, 0],
            [5, 4, 1],
        ),
        (
            "2da-depth2",
            Type2DArray,
            [16, 16, 1, 3, 1],
            [0, 0],
            [2, 3, 0],
            [5, 4, 2],
        ),
        (
            "2da-z1",
            Type2DArray,
            [16, 16, 1, 3, 1],
            [0, 0],
            [2, 3, 1],
            [5, 4, 1],
        ),
        (
            "3d-z1d2",
            Type3D,
            [16, 16, 4, 1, 3],
            [0, 0],
            [2, 3, 1],
            [5, 4, 2],
        ),
        (
            "3d-slice1",
            Type3D,
            [16, 16, 4, 1, 3],
            [0, 1],
            [2, 3, 0],
            [5, 4, 1],
        ),
        (
            "3d-depth6",
            Type3D,
            [16, 16, 4, 1, 3],
            [0, 0],
            [2, 3, 1],
            [5, 4, 6],
        ),
        (
            "3d-level1",
            Type3D,
            [16, 16, 4, 1, 3],
            [1, 0],
            [1, 1, 0],
            [4, 4, 2],
        ),
        (
            "cube-slice0",
            Cube,
            [16, 16, 1, 1, 1],
            [0, 0],
            [2, 3, 0],
            [5, 4, 1],
        ),
        (
            "cube-slice5",
            Cube,
            [16, 16, 1, 1, 1],
            [0, 5],
            [2, 3, 0],
            [5, 4, 1],
        ),
        (
            "cube-slice6",
            Cube,
            [16, 16, 1, 1, 1],
            [0, 6],
            [2, 3, 0],
            [5, 4, 1],
        ),
        (
            "cube-slice8",
            Cube,
            [16, 16, 1, 1, 1],
            [0, 8],
            [2, 3, 0],
            [5, 4, 1],
        ),
        (
            "cube-depth2",
            Cube,
            [16, 16, 1, 1, 1],
            [0, 1],
            [2, 3, 0],
            [5, 4, 2],
        ),
        (
            "cubea-slice7",
            CubeArray,
            [16, 16, 1, 2, 1],
            [0, 7],
            [2, 3, 0],
            [5, 4, 1],
        ),
        (
            "cubea-slice12",
            CubeArray,
            [16, 16, 1, 2, 1],
            [0, 12],
            [2, 3, 0],
            [5, 4, 1],
        ),
        (
            "cubea-slice13",
            CubeArray,
            [16, 16, 1, 2, 1],
            [0, 13],
            [2, 3, 0],
            [5, 4, 1],
        ),
        (
            "1d-slice1",
            Type1D,
            [16, 1, 1, 1, 1],
            [0, 1],
            [2, 0, 0],
            [5, 1, 1],
        ),
        (
            "1d-h2",
            Type1D,
            [16, 1, 1, 1, 1],
            [0, 0],
            [2, 0, 0],
            [5, 2, 1],
        ),
        (
            "1d-y1",
            Type1D,
            [16, 1, 1, 1, 1],
            [0, 0],
            [2, 1, 0],
            [5, 1, 1],
        ),
        (
            "1da-slice2",
            Type1DArray,
            [16, 1, 1, 3, 1],
            [0, 2],
            [2, 0, 0],
            [5, 1, 1],
        ),
        (
            "1da-slice3",
            Type1DArray,
            [16, 1, 1, 3, 1],
            [0, 3],
            [2, 0, 0],
            [5, 1, 1],
        ),
        (
            "1da-h2",
            Type1DArray,
            [16, 1, 1, 3, 1],
            [0, 0],
            [2, 0, 0],
            [5, 2, 1],
        ),
    ]
};

/// Metal's digest of each placement case's dump.
const PLACED: [(&str, &str); 31] = [
    (
        "2d-base",
        "98732a3169289d729f689b09aef27fb8e7ce48f6d304b05c40accb8caad2a89a",
    ),
    (
        "2d-slice1",
        "98732a3169289d729f689b09aef27fb8e7ce48f6d304b05c40accb8caad2a89a",
    ),
    (
        "2d-slice5",
        "98732a3169289d729f689b09aef27fb8e7ce48f6d304b05c40accb8caad2a89a",
    ),
    (
        "2d-level1",
        "86f52bd70731626d4b1cfd7c50a02e23175144b759ce248ca61a96b8a8623668",
    ),
    (
        "2d-level3",
        "018c12cfdfd302cd82892522a015e56c9cc0adb2615e987ebed772f436146b5b",
    ),
    (
        "2d-z1",
        "98732a3169289d729f689b09aef27fb8e7ce48f6d304b05c40accb8caad2a89a",
    ),
    (
        "2d-depth3",
        "98732a3169289d729f689b09aef27fb8e7ce48f6d304b05c40accb8caad2a89a",
    ),
    (
        "2d-past",
        "08661bcae0a7ba4db0273fe2da1cd541e3ba11b4089b951c532d8f2a838132dd",
    ),
    (
        "2d-level1-past",
        "fad248f5733994464389118c39d026dfb58c2c11fa3e4bcbabe250c53f96dbc7",
    ),
    (
        "2da-slice1",
        "376efc8f8d5c96def8b307d40b8e267a2554b80e72469d15bdaab59c6fece563",
    ),
    (
        "2da-slice3",
        "f334812c00296a49dbf3c71be73b562c015b4b1e35fbc2517f2fd27016628f5c",
    ),
    (
        "2da-depth2",
        "594a465ed515c34b4c5861d22c0c12fd9edcb005b66bad04c86308707afcd47a",
    ),
    (
        "2da-z1",
        "594a465ed515c34b4c5861d22c0c12fd9edcb005b66bad04c86308707afcd47a",
    ),
    (
        "3d-z1d2",
        "f3888a6336377cd1f10836b2713bc8e4ee6f0038a4438773dc1d7eb64198bdfa",
    ),
    (
        "3d-slice1",
        "4556637647cd8ce8c294c1d6dfd9409f8ca4fffe6d428e7a88df2dda85661966",
    ),
    (
        "3d-depth6",
        "0d077686a396e62027801f0f8a4475151f144a01fda5a9e438155d17b3b128dc",
    ),
    (
        "3d-level1",
        "4e56a9ecbd3e0c80a349c9a52a46270e82f80067107876d4ec6e31149c44b687",
    ),
    (
        "cube-slice0",
        "47cd42789999febc490a3d58983ab0e00441768f32aacd7cc2a8b2a30528b750",
    ),
    (
        "cube-slice5",
        "8bdd02bc17395fbc917e7333c8f447264847a1209d4d544df5d654d64f41bef3",
    ),
    (
        "cube-slice6",
        "47cd42789999febc490a3d58983ab0e00441768f32aacd7cc2a8b2a30528b750",
    ),
    (
        "cube-slice8",
        "497280fd19be96965d03b636b947f2ca985579ade0dd379f8c68bb9a87c61065",
    ),
    (
        "cube-depth2",
        "1f67a3cf011c30cd23a8889f7d05034c84e2f6cd4b2efa82abd16c963b80ee4b",
    ),
    (
        "cubea-slice7",
        "520dc6870593a9549be6b379c08a0bf40ffee7154920ee2c78ff62382eaafdcc",
    ),
    (
        "cubea-slice12",
        "97537c0db8dfa16594992d45fd69592f9d6b4d78d694c99836ca23de0feaf3ab",
    ),
    (
        "cubea-slice13",
        "97537c0db8dfa16594992d45fd69592f9d6b4d78d694c99836ca23de0feaf3ab",
    ),
    (
        "1d-slice1",
        "d11a64a1665da0aba939cb126671175efc4b498aaa1310cc4f0f7c26753c6141",
    ),
    (
        "1d-h2",
        "d11a64a1665da0aba939cb126671175efc4b498aaa1310cc4f0f7c26753c6141",
    ),
    (
        "1d-y1",
        "6d55521f1fd2007f31ba4983d03617b5c8367bcb00287e0b65ef6ce7f4bb7a2f",
    ),
    (
        "1da-slice2",
        "22f3307dca51656b0689d8dcee270992540932035d7ef69eeea0916bc38b672e",
    ),
    (
        "1da-slice3",
        "68469347d59c720e9d774bb22ce90ec5796034efb5c3a1788d2fc3d851005043",
    ),
    (
        "1da-h2",
        "455d273451081e2c62d3ac2f05bc03d534fae5a256d8ee0f1273a29236510b07",
    ),
];

/// Metal's digest of the `pattern4` fills' dump.
const PATTERN4: &str = "603bbbdf5fe28d71d46e303671e4ff59ff59ab6c686125326f5524f65fb982b1";

/// How a placement case fills its region: the probe's byte fill of `11 22 33 44`, or a color
/// fill of a color whose codes no conversion is in question for (each channel a whole number of
/// codes), plain or packed by `pixelFormat:`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Placed {
    Bytes,
    Color { packed: bool },
}

/// The dump of placement case `place` (the module docs).
fn placed(lane: &Lane, place: &Place) -> Vec<u8> {
    placed_by(lane, place, Placed::Bytes)
}

/// [`placed`], the region filled by `by`.
fn placed_by(lane: &Lane, place: &Place, by: Placed) -> Vec<u8> {
    let &(name, ty, [w, h, d, arr, levels], [level, slice], origin, size) = place;
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, w, h, false);
    descriptor.texture_type = ty;
    descriptor.depth = d;
    descriptor.array_length = arr;
    descriptor.mipmap_level_count = levels;
    descriptor.usage = USAGE;
    descriptor.resource_options = ResourceOptions::private();
    let texture = lane
        .device
        .new_texture(&descriptor)
        .unwrap_or_else(|e| panic!("{name}: the texture: {e}"));
    let slices = match ty {
        TextureType::Cube => 6,
        TextureType::CubeArray => 6 * arr,
        _ => arr,
    };
    let extent = |l: u64| [(w >> l).max(1), (h >> l).max(1), (d >> l).max(1)];
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for l in 0..levels {
        let [lw, lh, ld] = extent(l);
        for s in 0..slices {
            blit.fill_texture(
                &texture,
                l,
                s,
                Region::new_3d(0, 0, 0, lw, lh, ld),
                &[0xcd; 4],
            )
            .unwrap();
        }
    }
    let region = Region::new_3d(origin[0], origin[1], origin[2], size[0], size[1], size[2]);
    // 51, 102, 153 and 204 of 255: the same codes whatever the rounding.
    let color = ClearColor::new(0.2, 0.4, 0.6, 0.8);
    let packed_in = Some(PixelFormat::RGBA8Unorm);
    match by {
        Placed::Bytes => {
            blit.fill_texture(&texture, level, slice, region, &[0x11, 0x22, 0x33, 0x44])
        }
        Placed::Color { packed: false } => {
            blit.fill_texture_color(&texture, level, slice, region, color, None)
        }
        Placed::Color { packed: true } => {
            blit.fill_texture_color(&texture, level, slice, region, color, packed_in)
        }
    }
    .unwrap_or_else(|e| panic!("{name}: the fill: {e}"));
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let mut dump = Vec::new();
    for l in 0..levels {
        let [lw, lh, ld] = extent(l);
        for s in 0..slices {
            let n = lw * lh * ld * 4;
            let back = buffer_with(&lane.device, &vec![0xab; n as usize]).unwrap();
            let command_buffer = lane.queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_texture_to_buffer(
                &texture,
                s,
                l,
                Origin::new(0, 0, 0),
                Size::new(lw, lh, ld),
                &back,
                0,
                lw * 4,
                lw * lh * 4,
                BlitOption::None,
            )
            .unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
            for word in [l, s, lw * lh * ld, 0] {
                dump.extend_from_slice(&(word as u32).to_le_bytes());
            }
            dump.extend(read_buffer(&back, 0, n).unwrap());
        }
    }
    dump
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Every placement case's fill lands where Metal's lands, and `fill_buffer_pattern4` writes
/// Metal's bytes.
#[test]
fn fills_land_where_metal_s_land() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let mut wrong = Vec::new();
    for (place, (case, metal)) in PLACES.iter().zip(PLACED) {
        assert_eq!(place.0, case);
        let got = sha256(&placed(&lane, place));
        if got != metal {
            wrong.push(format!("{case}: {got}, Metal's {metal}"));
        }
    }
    let ranges: [(u64, u64); 10] = [
        (0, 16),
        (41, 7),
        (3, 50),
        (16, 0),
        (5, 100),
        (0, 1),
        (15, 2),
        (17, 31),
        (32, 64),
        (1, 255),
    ];
    let mut dump = Vec::new();
    for (at, n) in ranges {
        let buffer = buffer_with(&lane.device, &[0xcd; 256]).unwrap();
        let command_buffer = lane.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer_pattern4(&buffer, at..at + n, 0x0403_0201)
            .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        dump.extend_from_slice(&(at as u32).to_le_bytes());
        dump.extend_from_slice(&(n as u32).to_le_bytes());
        dump.extend(read_buffer(&buffer, 0, 256).unwrap());
    }
    let got = sha256(&dump);
    if got != PATTERN4 {
        wrong.push(format!("pattern4: {got}, Metal's {PATTERN4}"));
    }
    for line in &wrong {
        println!("COLOR-FILLS-DIFFERS {name}: {line}");
    }
    assert!(
        wrong.is_empty(),
        "{} placements differ from Metal's",
        wrong.len()
    );
    quiet(lane, &messages);
}

/// Every placement case's plain color fill, the device's clear of the color (`blit_fill.rs`),
/// lands where the same color packed by `pixelFormat:` lands, byte for byte: a color whose codes
/// no rounding is in question for, so the two write one texel, and the packed fill lands where
/// the byte fill does ([`fills_land_where_metal_s_land`]), which is Metal's placement.
#[test]
fn plain_color_fills_land_where_packed_ones_land() {
    let (messages, _cache, lane) = lane();
    let mut wrong = Vec::new();
    for place in &PLACES {
        let plain = placed_by(&lane, place, Placed::Color { packed: false });
        let packed = placed_by(&lane, place, Placed::Color { packed: true });
        if plain != packed {
            let differ = plain.iter().zip(&packed).filter(|(a, b)| a != b).count();
            wrong.push(format!("{}: {differ} bytes", place.0));
        }
    }
    assert!(
        wrong.is_empty(),
        "plain color fills not where packed ones land: {wrong:?}"
    );
    quiet(lane, &messages);
}

/// The images the device made (`vkCreateImage`) or took from its recycler so far.
fn images_taken(lane: &Lane) -> u64 {
    let churn = lane.device.work_counts().object_churn;
    churn.vk_images_made + churn.images_recycled
}

/// The bytes of `size` texels of `texture`'s `level` and `slice`, 4 bytes each.
fn level_bytes(lane: &Lane, texture: &Texture, (level, slice): (u64, u64), size: Size) -> Vec<u8> {
    let length = size.width * size.height * size.depth * 4;
    let back = buffer_with(&lane.device, &vec![0xab; length as usize]).unwrap();
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        slice,
        level,
        Origin::new(0, 0, 0),
        size,
        &back,
        0,
        size.width * 4,
        size.width * size.height * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    read_buffer(&back, 0, length).unwrap()
}

/// Run the plain `RGBA8Unorm` fills `fills` encodes in one command buffer of `lane`'s queue: the
/// images it made or took from the recycler ([`images_taken`]).
fn images_of(
    lane: &Lane,
    fills: impl FnOnce(&mut metal2vulkan_runtime::BlitCommandEncoder),
) -> u64 {
    let before = images_taken(lane);
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    fills(&mut blit);
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    images_taken(lane) - before
}

/// A plain color fill of whole subresources is the device's clear of the texture's own image
/// (`blit_fill.rs`): it makes no image and takes none from the recycler (`WorkCounts`'
/// `vkCreateImage` calls and recycled images), into a 2D texture's second level, a 2D array's
/// slice and a 3D texture's level, each texel then the color's (`RGBA8Unorm`, codes no rounding
/// is in question for) and the other level and slices kept. A command buffer of 64 fills of parts
/// of a level takes one tile for all of them where they fill one color, and one a color where
/// they fill two in turn, each part then its color's.
#[test]
fn whole_subresource_fills_make_no_image() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let pf = PixelFormat::RGBA8Unorm;
    let color = ClearColor::new(1.0, 0.0, 0.2, 1.0);
    let texel = [255u8, 0, 51, 255];
    let other = ClearColor::new(0.0, 0.6, 1.0, 0.4);
    let other_texel = [0u8, 153, 255, 102];
    let made = |descriptor: &mut TextureDescriptor| {
        descriptor.usage = USAGE;
        descriptor.resource_options = ResourceOptions::private();
        lane.device.new_texture(descriptor).unwrap()
    };
    let solid = |bytes: &[u8], texel: [u8; 4]| bytes.chunks(4).all(|t| t == texel);
    // A texture's every byte `0x11` first.
    let set = |texture: &Texture, levels: u64, slices: u64| {
        let command_buffer = lane.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for level in 0..levels {
            for slice in 0..slices {
                let region = Region::new_3d(0, 0, 0, 64, 64, 64);
                blit.fill_texture(texture, level, slice, region, &[0x11])
                    .unwrap();
            }
        }
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    };

    let mut descriptor = TextureDescriptor::texture_2d(pf, 64, 64, false);
    descriptor.mipmap_level_count = 2;
    let mipmapped = made(&mut descriptor);
    set(&mipmapped, 2, 1);
    let taken = images_of(&lane, |blit| {
        let region = Region::new_2d(0, 0, 32, 32);
        blit.fill_texture_color(&mipmapped, 1, 0, region, color, None)
            .unwrap();
    });
    assert_eq!(
        taken, 0,
        "{name}: a 2D level's whole fill made or took an image"
    );
    assert!(solid(
        &level_bytes(&lane, &mipmapped, (1, 0), Size::new(32, 32, 1)),
        texel
    ));
    let level_0 = level_bytes(&lane, &mipmapped, (0, 0), Size::new(64, 64, 1));
    assert!(
        level_0.iter().all(|&b| b == 0x11),
        "{name}: level 0 changed"
    );

    let mut descriptor = TextureDescriptor::texture_2d(pf, 32, 32, false);
    descriptor.texture_type = TextureType::Type2DArray;
    descriptor.array_length = 3;
    let array = made(&mut descriptor);
    set(&array, 1, 3);
    let taken = images_of(&lane, |blit| {
        let region = Region::new_2d(0, 0, 32, 32);
        blit.fill_texture_color(&array, 0, 1, region, color, None)
            .unwrap();
    });
    assert_eq!(
        taken, 0,
        "{name}: a 2D array slice's whole fill made or took an image"
    );
    for slice in 0..3 {
        let bytes = level_bytes(&lane, &array, (0, slice), Size::new(32, 32, 1));
        match slice {
            1 => assert!(solid(&bytes, texel), "{name}: slice 1"),
            _ => assert!(
                bytes.iter().all(|&b| b == 0x11),
                "{name}: slice {slice} changed"
            ),
        }
    }

    let mut descriptor = TextureDescriptor::texture_2d(pf, 16, 16, false);
    descriptor.texture_type = TextureType::Type3D;
    descriptor.depth = 4;
    let volume = made(&mut descriptor);
    set(&volume, 1, 1);
    let taken = images_of(&lane, |blit| {
        let region = Region::new_3d(0, 0, 0, 16, 16, 4);
        blit.fill_texture_color(&volume, 0, 0, region, color, None)
            .unwrap();
    });
    assert_eq!(
        taken, 0,
        "{name}: a 3D level's whole fill made or took an image"
    );
    assert!(solid(
        &level_bytes(&lane, &volume, (0, 0), Size::new(16, 16, 4)),
        texel
    ));

    // Parts: 64 fills of 8 x 8 texels, one color, then two in turn; one tile a color.
    let parts = made(&mut TextureDescriptor::texture_2d(pf, 64, 64, false));
    for alternate in [false, true] {
        let taken = images_of(&lane, |blit| {
            for i in 0..64u64 {
                let region = Region::new_2d(i % 8 * 8, i / 8 * 8, 8, 8);
                let color = if alternate && i % 2 == 1 {
                    other
                } else {
                    color
                };
                blit.fill_texture_color(&parts, 0, 0, region, color, None)
                    .unwrap();
            }
        });
        let tiles = 1 + u64::from(alternate);
        assert_eq!(
            taken, tiles,
            "{name}: 64 fills of parts (alternating {alternate})"
        );
        let bytes = level_bytes(&lane, &parts, (0, 0), Size::new(64, 64, 1));
        for (at, got) in bytes.chunks(4).enumerate() {
            let (x, y) = (at as u64 % 64, at as u64 / 64);
            let i = y / 8 * 8 + x / 8;
            let want = if alternate && i % 2 == 1 {
                other_texel
            } else {
                texel
            };
            assert_eq!(
                got, want,
                "{name}: texel ({x}, {y}), alternating {alternate}"
            );
        }
    }
    quiet(lane, &messages);
}

/// The colors our probe `fil5.m` packs by `pixelFormat:` into another format's texture.
const PACKED_COLORS: [[f64; 4]; 8] = [
    [0.25, 0.5, 0.75, 1.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0],
    [-0.5, 2.0, 0.1, 0.9],
    [0.001, 0.04, 0.3, 0.5],
    [1.25, -0.75, 0.6, 0.2],
    [0.5, 0.5, 0.5, 0.5],
];

/// Metal's 16-byte texel of each of [`PACKED_COLORS`] packed in `BGRA10_XR_sRGB` and in
/// `BGR10_XR_sRGB` into an `RGBA32Uint`, `RGBA32Sint` or `RGBA32Float` texture first set to
/// `0x5a`: the packed texel repeated, alike on an Apple M3 Max under macOS 26.7 and an M4 under
/// macOS 27.0.1, a fill at a time and among 2,249 fills of one encoder.
const PACKED_INTO_RGBA32: [(PixelFormat, [&str; 8]); 2] = [
    (
        PixelFormat::BGRA10XRSRGB,
        [
            "40d0c0bd80a480df40d0c0bd80a480df",
            "0060006080df00600060006080df0060",
            "00600060006000600060006000600060",
            "80df80df80df80df80df80df80df80df",
            "808cc0ff4002c0d2808cc0ff4002c0d2",
            "80aa407cc061c09f80aa407cc061c09f",
            "c0c5000080ec8079c0c5000080ec8079",
            "c0bdc0bdc0bdc09fc0bdc0bdc0bdc09f",
        ],
    ),
    (
        PixelFormat::BGR10XRSRGB,
        [
            "41df2b2941df2b2941df2b2941df2b29",
            "8001e6378001e6378001e6378001e637",
            "80010618800106188001061880010618",
            "7efbed377efbed377efbed377efbed37",
            "32fe9f0032fe9f0032fe9f0032fe9f00",
            "aac67718aac67718aac67718aac67718",
            "1703203b1703203b1703203b1703203b",
            "f7de7b2ff7de7b2ff7de7b2ff7de7b2f",
        ],
    ),
];

/// A color packed by `pixelFormat:` in an extended-range `_sRGB` format into an `RGBA32`
/// texture repeats the packed texel, as Metal's fills write it, every fill in one encoder; one
/// packed in a padded 10-bit format, in which Metal packs no color, fills the 16 bytes the last
/// fills packed over zeros (the last `BGR10_XR_sRGB` texel over the last `BGRA10_XR_sRGB` one),
/// as Metal's consecutive fills find their packing's bytes, and the device announces
/// `unpacked-fill-color` once.
#[test]
fn packings_in_another_format_repeat_and_padded_ones_fill_the_last_packing() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let mut wrong = Vec::new();
    for target in [
        PixelFormat::RGBA32Uint,
        PixelFormat::RGBA32Sint,
        PixelFormat::RGBA32Float,
    ] {
        let mut descriptor = TextureDescriptor::texture_2d(target, WIDTH, HEIGHT, false);
        descriptor.usage = USAGE;
        descriptor.resource_options = ResourceOptions::private();
        let texture = lane.device.new_texture(&descriptor).unwrap();
        let length = WIDTH * HEIGHT * 16;
        let pattern = buffer_with(&lane.device, &vec![0x5a; length as usize]).unwrap();
        let command_buffer = lane.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &pattern,
            0,
            WIDTH * 16,
            length,
            Size::new(WIDTH, HEIGHT, 1),
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
        let mut i = 0;
        for (packed_in, _) in PACKED_INTO_RGBA32 {
            for [r, g, b, a] in PACKED_COLORS {
                let region = Region::new_2d(i % WIDTH, i / WIDTH, 1, 1);
                let color = ClearColor::new(r, g, b, a);
                blit.fill_texture_color(&texture, 0, 0, region, color, Some(packed_in))
                    .unwrap_or_else(|e| panic!("{target:?} {packed_in:?}: a fill: {e}"));
                i += 1;
            }
        }
        for padded in [
            PixelFormat::R10UnormX6,
            PixelFormat::R10UnormX6SRGB,
            PixelFormat::RG10UnormX12,
            PixelFormat::RG10UnormX12SRGB,
        ] {
            let region = Region::new_2d(i % WIDTH, i / WIDTH, 1, 1);
            let color = ClearColor::new(0.25, 0.5, 0.75, 1.0);
            blit.fill_texture_color(&texture, 0, 0, region, color, Some(padded))
                .unwrap_or_else(|e| panic!("{target:?} {padded:?}: a fill: {e}"));
        }
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let got = texels(&lane, &texture, 16, i + 1);
        let mut texel = got.chunks(16);
        for (packed_in, wants) in PACKED_INTO_RGBA32 {
            for (color, want) in PACKED_COLORS.iter().zip(wants) {
                let got = hex(texel.next().unwrap());
                if got != want {
                    wrong.push(format!(
                        "{packed_in:?} into {target:?}, {color:?}: {got} (Metal's {want})"
                    ));
                }
            }
        }
        // The padded packings' fills wrote the last packings' bytes into the texel past the
        // packed ones.
        let slot = hex(texel.next().unwrap());
        if slot != "f7de7b2fc0bdc09f0000000000000000" {
            wrong.push(format!("{target:?}: a padded packing's fill wrote {slot}"));
        }
    }
    // The `Shared` pattern buffers lie in system memory on a device without unified memory.
    let mut kinds = forced(&[(
        Degradation::SharedInSystemMemory,
        shared_in_system_memory(&lane.device),
    )]);
    kinds.extend(UNPACKED_FILL);
    assert_once(&messages.take_notices(), &Vec::from_iter(kinds));
    for line in &wrong {
        println!("COLOR-FILLS-DIFFERS {name}: {line}");
    }
    assert!(
        wrong.is_empty(),
        "{} texels differ from Metal's",
        wrong.len()
    );
    quiet(lane, &messages);
}

/// `bytes` as lowercase hex, two digits a byte.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
