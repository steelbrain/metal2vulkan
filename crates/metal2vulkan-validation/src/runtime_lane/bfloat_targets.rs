//! Render passes into Apple's bfloat formats (`R16Bfloat`, `RG16Bfloat`, `RGBA16Bfloat`) on the
//! device the cfg selects, under the validation layer, against Metal's codes: the consumer's
//! passes clear such attachments.
//!
//! Metal's answers: an Apple M4 under macOS 27.0.1, our own probe `bfp.m` (no
//! guest data):
//!
//! - **Clears.** 2,048 clear colors (the probe's generator, rebuilt here; the colors' digest is
//!   checked first), each cleared through a `Clear` load action into a 1 x 1 target: Metal's
//!   codes, as a digest over the list. The model (`ExtendedRange::clear_texel`) gives the same
//!   digest, checked here too: a 1 x 1 target's clear goes through `half`
//!   (`ClearRoute::ThroughHalf`).
//! - **Clear routes** (the consumer's M3 Max under macOS 26.7 alike). Metal clears straight to
//!   bfloat (`ClearRoute::Straight`) where it keeps the level compressed and the pass has no draw
//!   call, else through `half`: targets of each route clear their route's codes for colors whose
//!   routes differ (the consumer's, `-3e38`, `1e-40`, `70000` and a `half` tie that is no bfloat
//!   tie), the extremes, subnormals and NaNs; the consumer's codes from its M3 Max are checked
//!   against the model too.
//! - **Loads.** A 256-texel-wide target holding every bfloat code once (each channel of each
//!   texel the next code), through one pass that stores it: loaded with no draw, which keeps
//!   every code; loaded under a full-screen draw that writes another attachment alone, which
//!   rewrites every code through `half` (56,829 codes move), where the device's store keeps every
//!   code (`extended-range-bits-kept`); with a `DontCare` load under that draw, which stores
//!   zeros, or with no draw, which keeps every code; and cleared to `(0.5, -2, 3e38, 1e-30)` under
//!   that draw. Metal's answer is the SHA-256 of the target's bytes after the pass. The probe's bfloat target was its pass's color
//!   attachment 0 and the drawn one attachment 1; here they are swapped, since the owned fixture
//!   `rt6_samples`'s fill writes color 0 (Metal's pass rewrites each loaded attachment of every
//!   tile a draw reaches, whichever it writes).
//!
//! - **Draws** (an Apple M4 under macOS 27.0.1, a probe `fpv5.swift`, our own): a
//!   full-screen draw of the owned fixture's fill into a new target of each format, at the default
//!   `textureWriteRoundingMode` and at `NearestEven`, of every code's `float`, of every `half`
//!   code, and of the `float`s around each code's tie: Metal's answer is the SHA-256 of the
//!   target's bytes. A model of Metal's codes (each output rounded to `half` by the pipeline's
//!   rounding, then to the nearest bfloat, every NaN `0x7fc0`) gives every one of those digests;
//!   the runtime's codes are compared with it code by code, a NaN's bits the device's
//!   (`PrecisionNote::DevicePackConversion`).
//!
//! A difference is printed on a `BFLOAT-TARGETS-DIFFERS` line.

use super::degrade_notices::counts;
use super::extended_range_targets::digest;
use super::resolves::{lane, lane_with, quiet, Lane};
use super::{buffer_with, read_buffer, WAIT};
use crate::candidate::runtime_samples::{library, shape};
use metal2vulkan::metal::format::{ClearRoute, ExtendedRange};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage, TextureWriteRoundingMode};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, PrecisionNote, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, Degradation, LoadAction, Origin, PipelineOptions, RenderPassDescriptor,
    ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
};

/// The bfloat formats, their channels, and Metal's digests (SHA-256): the clears' codes, and
/// the target's bytes after the clear under a draw.
const FORMATS: [(PixelFormat, u64, &str, &str); 3] = [
    (
        PixelFormat::R16Bfloat,
        1,
        "8fb7c46c7f6e5f2e1900c62c4cca6900630ac1ea4c0f5b6ad8ef0420db4b34aa",
        "67e1855aed2aa0669f09fe7256b1d7cbfce6f118d8f80486bd280f460e39d0ac",
    ),
    (
        PixelFormat::RG16Bfloat,
        2,
        "5d3cda930560b938f3fabc0228099571306304351a9462480d965a91176ca2f7",
        "175e96481664506bfe385b4bf875e6c632ece4a05166fdde990eafc2d62c8756",
    ),
    (
        PixelFormat::RGBA16Bfloat,
        4,
        "0ef726b723f2e8c5951e728851819da8248b5cb6561ed9e541bfca799e0a02fc",
        "9c29ccd852bc43f77604344c5da0d867fabafcfce605d47dae626ce9fc74b4db",
    ),
];

/// Metal's digest of every code in order, which a pass that keeps them leaves.
const EVERY_CODE: &str = "68e419472d25e0b85e9917ccf692fd58245c5e95e9a46f07d1df81d2e9da246b";
/// Metal's digest after a load under a draw: every code through `half`.
const LOADED: &str = "ceb2151e45fa832676006dfa73c29488905a503964183dbe14b5fc1a1267f710";
/// Metal's digest after a `DontCare` load under a draw: zeros.
const DONT_CARE: &str = "fa43239bcee7b97ca62f007cc68487560a39e19f74f3dde7486db3f98df8e471";

/// The FNV-1a of the probe's colors' bits, as it recorded them.
const COLORS: u64 = 0xde31_e66e_0a02_d326;

/// The probe's clear colors (`bfp.m`'s generator; the module docs).
fn clear_colors() -> Vec<[f64; 4]> {
    const SPECIALS: [f64; 38] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        2.0,
        65504.0,
        65_519.999_999_999_99,
        65520.0,
        65536.0,
        1e10,
        -1e10,
        3.402_823_466_385_288_6e38,
        1e39,
        -1e39,
        1e-30,
        -1e-30,
        5.960_464_477_539_063e-8,
        2.980_232_238_769_531_2e-8,
        // The probe's `2.9802322387695315e-08`, the same double.
        2.980_232_238_769_531_2e-8,
        6.103_515_625e-5,
        6.097_555_160_522_461e-5,
        1.401_298_464_324_817e-45,
        1.0 + 1.0 / 256.0,
        1.0 + 3.0 / 256.0,
        1.0 + 1.0 / 2048.0,
        1.0 + 3.0 / 2048.0,
        1.0 + 1.0 / 2048.0 + 1.0 / 1_073_741_824.0,
        1.0 / 3.0,
        255.5,
        0.1,
        0.2,
        0.3,
        1e5,
        70000.0,
        100000.0,
        1e-8,
        1e-6,
    ];
    const NANS: [u64; 6] = [
        0x7ff8_0000_0000_0000,
        0xfff8_0000_0000_0000,
        0x7ff0_0000_0000_0001,
        0x7ff4_0000_0000_0000,
        0xfff0_0400_0000_0000,
        0x7ff8_4000_0000_0000,
    ];
    let half = |h: u16| f64::from(half_value(h));
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut value = move || -> f64 {
        let r = next();
        let signed = |v: f64| if (r >> 30) & 1 == 1 { -v } else { v };
        match r % 8 {
            0 => f64::from_bits(NANS[((r >> 8) % 6) as usize]),
            1 | 2 => SPECIALS[((r >> 8) % 38) as usize],
            3 => {
                // A `half` value, or the midpoint between it and the next, or a neighbour of the
                // midpoint.
                let h = (next() & 0x7bff) as u16;
                let m = (half(h) + half(h + 1)) / 2.0;
                signed(match (r >> 20) % 3 {
                    0 => half(h),
                    1 => m,
                    _ if (r >> 24) & 1 == 1 => m.next_up(),
                    _ => m.next_down(),
                })
            }
            4 => {
                // A bfloat midpoint, or a neighbour of it.
                let f = f64::from(f32::from_bits((next() & 0x7f7f_0000) as u32 | 0x8000));
                signed(match (r >> 20) % 3 {
                    0 => f,
                    1 => f.next_up(),
                    _ => f.next_down(),
                })
            }
            _ => {
                let u = (next() >> 11) as f64 / 9_007_199_254_740_992.0;
                match (r >> 20) % 3 {
                    0 => -2.0 + 5.0 * u,
                    1 => -70000.0 + 140_000.0 * u,
                    // `ldexp(u, e)`: the power of two exact.
                    _ => u * f64::from_bits((1023 + (r >> 32) % 200 - 100) << 52),
                }
            }
        }
    };
    let mut colors: Vec<[f64; 4]> = (0..2048)
        .map(|_| std::array::from_fn(|_| value()))
        .collect();
    for (color, special) in colors.iter_mut().zip(SPECIALS) {
        *color = [special; 4];
    }
    colors
}

/// The `float` of the `half` of bits `h` (the probe's `__fp16` conversion, exact).
fn half_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((h >> 10) & 0x1f);
    let fraction = f32::from(h & 0x3ff);
    sign * match exponent {
        0 => fraction * 2f32.powi(-24),
        31 if fraction == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        e => (1024.0 + fraction) * 2f32.powi(e - 25),
    }
}

/// FNV-1a of `bytes`.
fn fnv(bytes: impl Iterator<Item = u8>) -> u64 {
    bytes.fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// A private 2D texture of `pf`, `width` x `height`, of `usage`.
fn texture(lane: &Lane, pf: PixelFormat, width: u64, height: u64, usage: TextureUsage) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(pf, width, height, false);
    descriptor.usage = usage;
    descriptor.resource_options = ResourceOptions::private();
    lane.device
        .new_texture(&descriptor)
        .unwrap_or_else(|e| panic!("a {pf:?} texture: {e}"))
}

/// A `Clear` of a bfloat attachment stores Metal's codes for every color of the probe's.
#[test]
fn bfloat_clears_store_metal_s_codes() {
    let (messages, _cache, lane) = lane();
    let colors = clear_colors();
    let bits = colors
        .iter()
        .flat_map(|c| c.iter().flat_map(|v| v.to_le_bytes()));
    assert_eq!(fnv(bits), COLORS, "the probe's colors");
    let mut wrong = Vec::new();
    for (pf, channels, cleared, _) in FORMATS {
        let form = ExtendedRange::of(pf).expect("a bfloat format");
        let size = 2 * channels;
        let n = colors.len() as u64;
        let bytes = buffer_with(&lane.device, &vec![0xab; (size * n) as usize]).unwrap();
        let target = texture(&lane, pf, 1, 1, TextureUsage::RENDER_TARGET);
        for batch in (0..n).step_by(256) {
            let command_buffer = lane.queue.command_buffer().unwrap();
            for i in batch..(batch + 256).min(n) {
                let [r, g, b, a] = colors[i as usize];
                let mut pass = RenderPassDescriptor::new();
                let color = &mut pass.color_attachments[0];
                color.texture = Some(target.clone());
                color.load_action = LoadAction::Clear;
                color.store_action = StoreAction::Store;
                color.clear_color = ClearColor::new(r, g, b, a);
                command_buffer
                    .render_command_encoder(&pass)
                    .unwrap_or_else(|e| panic!("{pf:?}: the clear's pass: {e}"))
                    .end_encoding()
                    .unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_texture_to_buffer(
                    &target,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                    Size::new(1, 1, 1),
                    &bytes,
                    size * i,
                    size,
                    size,
                    BlitOption::None,
                )
                .unwrap();
                blit.end_encoding().unwrap();
            }
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
        }
        let model: Vec<u8> = colors
            .iter()
            .flat_map(|&c| {
                let texel = form.clear_texel(c, ClearRoute::ThroughHalf);
                texel
                    .into_iter()
                    .take(channels as usize)
                    .flat_map(|w| u16::try_from(w).expect("a code").to_le_bytes())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(digest(&model), cleared, "{pf:?}: the model's clears");
        let got = read_buffer(&bytes, 0, size * n).unwrap();
        // The bytes read again after a later command buffer of the queue completed: a texel
        // that changed reached the host after its wait returned; one that kept the fill was
        // never written (seen on the Radeon RX 7900 XTX and 780M under a full suite's load).
        let later = lane.queue.command_buffer().unwrap();
        later.commit().unwrap();
        later.wait_until_completed_for(WAIT).unwrap();
        let again = read_buffer(&bytes, 0, size * n).unwrap();
        for (i, ((g, m), a)) in got
            .chunks(size as usize)
            .zip(model.chunks(size as usize))
            .zip(again.chunks(size as usize))
            .enumerate()
            .filter(|(_, ((g, m), _))| g != m)
            .take(8)
        {
            wrong.push(format!(
                "{pf:?} color {i} {:?}: stored {g:02x?}, Metal's {m:02x?}, read again after a \
                 later command buffer {a:02x?}",
                colors[i]
            ));
        }
    }
    for line in &wrong {
        println!("BFLOAT-TARGETS-DIFFERS {}: {line}", lane.device.name());
    }
    assert!(
        wrong.is_empty(),
        "{} clears differ from Metal's",
        wrong.len()
    );
    quiet(lane, &messages);
}

/// Colors whose bfloat clears differ by route, and the extremes: the consumer's (whose codes its
/// M3 Max measured), a `half` tie that is no bfloat tie beside two that are, the largest
/// `float`, one past it and the infinity, `float` subnormals, and NaNs of either sign.
const ROUTE_COLORS: [[f64; 4]; 6] = [
    [-3e38, 1e-40, 70000.0, -2.5],
    [
        1.0 + 1.0 / 256.0,
        1.0 + 3.0 / 256.0,
        1.0 + 1.0 / 256.0 + 1.0 / 8_388_608.0,
        -0.0,
    ],
    [
        3.402_823_466_385_288_6e38,
        -3.402_823_466_385_288_6e38,
        1e39,
        f64::NEG_INFINITY,
    ],
    [
        1.401_298_464_324_817e-45,
        -1e-40,
        1.175_494_350_822_287_5e-38,
        65520.0,
    ],
    [f64::NAN, -f64::NAN, 1.0, 0.5],
    [-1e-30, 1e10, 65_519.999_999_999_99, -70000.0],
];

/// The consumer's codes from its M3 Max (macOS 26.7) for `ROUTE_COLORS`' first two colors, in
/// a 16 x 16 private `RGBA16Bfloat` target cleared by a pass with no draw call: what
/// `ClearRoute::Straight` stores.
const M3_CODES: [[u16; 4]; 2] = [
    [0xff62, 0x0001, 0x4789, 0xc020],
    [0x3f80, 0x3f82, 0x3f81, 0x8000],
];

/// A `Clear` of a bfloat attachment stores the codes of Metal's route for its texture and pass
/// (an M4 under macOS 27.0.1; the consumer's M3 Max under 26.7 alike): straight to bfloat
/// (`ClearRoute::Straight`) for a 16 x 16 private target of plain usage in a pass with no draw
/// call, and through `half` (`ClearRoute::ThroughHalf`) for an 8 x 8 one, one of `ShaderWrite` or
/// `PixelFormatView` usage, and one whose pass draws into another attachment. Every texel of the
/// target is checked. On a device that presents a Mac keeping no texture compressed
/// (`DeviceProperties::lossless_compression` false) the 16 x 16 one is not compressed either, so
/// it takes the route of the uncompressed ones, through `half` (`nan_clear::keeps_nan_sign`;
/// expected, not measured on such a Mac).
#[test]
fn bfloat_clears_take_metal_s_route() {
    let (messages, _cache, lane) = lane_with(|_| {});
    let plain = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    // The route of the shape Metal keeps compressed, where the device presents the compression.
    let compressed = match lane.device.properties().lossless_compression {
        true => ClearRoute::Straight,
        false => ClearRoute::ThroughHalf,
    };
    let library = library(&lane.device).unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut wrong = Vec::new();
    for (pf, channels, _, _) in FORMATS {
        let form = ExtendedRange::of(pf).expect("a bfloat format");
        let mut state = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
        state.fragment = Some(function("rt6_fill_rgba32float"));
        state.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
        state.color_attachments[1] = Some(ColorAttachmentDescriptor::new(pf));
        let state = lane
            .device
            .new_render_pipeline_state(&state, &PipelineOptions::default())
            .unwrap_or_else(|e| panic!("{pf:?}: the pipeline: {e}"));
        for (side, usage, draws, route) in [
            (16, plain, false, compressed),
            (8, plain, false, ClearRoute::ThroughHalf),
            (
                16,
                plain | TextureUsage::SHADER_WRITE,
                false,
                ClearRoute::ThroughHalf,
            ),
            (
                16,
                plain | TextureUsage::PIXEL_FORMAT_VIEW,
                false,
                ClearRoute::ThroughHalf,
            ),
            (16, plain, true, ClearRoute::ThroughHalf),
        ] {
            let case = format!("{pf:?} {side}x{side} {usage:?} draws {draws}");
            let size = 2 * channels;
            let length = size * side * side;
            let values: Vec<u8> = (0..side * side)
                .flat_map(|i| [(i % 7) as f32 / 7.0, 0.25, 0.5, 1.0])
                .flat_map(f32::to_le_bytes)
                .collect();
            let values = buffer_with(&lane.device, &values).unwrap();
            let shape = shape(&lane.device, side as u32, 1).unwrap();
            for (i, &[r, g, b, a]) in ROUTE_COLORS.iter().enumerate() {
                let target = texture(&lane, pf, side, side, usage);
                let drawn = texture(&lane, PixelFormat::RGBA32Float, side, side, plain);
                let mut descriptor = RenderPassDescriptor::new();
                let other = &mut descriptor.color_attachments[0];
                other.texture = Some(drawn);
                other.load_action = LoadAction::DontCare;
                other.store_action = StoreAction::Store;
                let color = &mut descriptor.color_attachments[1];
                color.texture = Some(target.clone());
                color.load_action = LoadAction::Clear;
                color.store_action = StoreAction::Store;
                color.clear_color = ClearColor::new(r, g, b, a);
                lane.run(&descriptor, |_, e, _| {
                    if draws {
                        e.set_render_pipeline_state(&state).unwrap();
                        e.set_fragment_buffer(Some(&values), 0, 0).unwrap();
                        e.set_fragment_buffer(Some(&shape), 0, 1).unwrap();
                        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                            .unwrap();
                    }
                })
                .unwrap_or_else(|e| panic!("{case}: the pass: {e}"));
                let back = buffer_with(&lane.device, &vec![0xcd; length as usize]).unwrap();
                let command_buffer = lane.queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_texture_to_buffer(
                    &target,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                    Size::new(side, side, 1),
                    &back,
                    0,
                    size * side,
                    length,
                    BlitOption::None,
                )
                .unwrap();
                blit.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None);
                let got = read_buffer(&back, 0, length).unwrap();
                let c = ROUTE_COLORS[i];
                let model: Vec<u16> = form
                    .clear_texel(c, route)
                    .into_iter()
                    .take(channels as usize)
                    .map(|w| u16::try_from(w).expect("a code"))
                    .collect();
                let texels = got.chunks(size as usize).map(|t| {
                    t.chunks(2)
                        .map(|w| u16::from_le_bytes([w[0], w[1]]))
                        .collect::<Vec<u16>>()
                });
                if let Some((at, codes)) = texels.enumerate().find(|(_, t)| *t != model) {
                    wrong.push(format!(
                        "{case} color {i} {c:?}: texel {at} stored {codes:04x?}, Metal's \
                         {model:04x?}"
                    ));
                }
                if let (ClearRoute::Straight, Some(m3), PixelFormat::RGBA16Bfloat) =
                    (route, M3_CODES.get(i), pf)
                {
                    if model != m3 {
                        wrong.push(format!(
                            "{case} color {i} {c:?}: the model's {model:04x?}, the M3 Max's \
                             {m3:04x?}"
                        ));
                    }
                }
            }
        }
    }
    for line in &wrong {
        println!("BFLOAT-TARGETS-DIFFERS {}: {line}", lane.device.name());
    }
    assert!(wrong.is_empty(), "{} clears differ", wrong.len());
    quiet(lane, &messages);
}

/// How a load case's pass treats the bfloat target.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Case {
    /// Loaded, no draw.
    Kept,
    /// Loaded, under the draw.
    Loaded,
    /// `DontCare`, under the draw.
    DontCare,
    /// `DontCare`, no draw.
    DontCareKept,
    /// Cleared to `(0.5, -2, 3e38, 1e-30)`, under the draw.
    Cleared,
}

/// The target's bytes after `case`'s pass over a target of `pf` (`channels` channels) holding
/// every code in order.
fn load_case(lane: &Lane, pf: PixelFormat, channels: u64, case: Case) -> Vec<u8> {
    let (width, height) = (256, 256 / channels);
    let row = 2 * channels * width;
    let length = row * height;
    let codes: Vec<u8> = (0..length / 2)
        .flat_map(|i| (i as u16).to_le_bytes())
        .collect();
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let target = texture(lane, pf, width, height, usage);
    let staging = buffer_with(&lane.device, &codes).unwrap();
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &staging,
        0,
        row,
        length,
        Size::new(width, height, 1),
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let draws = matches!(case, Case::Loaded | Case::DontCare | Case::Cleared);
    let library = library(&lane.device).unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut state = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
    state.fragment = Some(function("rt6_fill_rgba32float"));
    state.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
    state.color_attachments[1] = Some(ColorAttachmentDescriptor::new(pf));
    let state = lane
        .device
        .new_render_pipeline_state(&state, &PipelineOptions::default())
        .unwrap_or_else(|e| panic!("{pf:?}: the pipeline: {e}"));
    let values: Vec<u8> = (0..width * height)
        .flat_map(|i| [(i % 7) as f32 / 7.0, 0.25, 0.5, 1.0])
        .flat_map(f32::to_le_bytes)
        .collect();
    let values = buffer_with(&lane.device, &values).unwrap();
    let shape = shape(&lane.device, width as u32, 1).unwrap();
    let drawn = texture(
        lane,
        PixelFormat::RGBA32Float,
        width,
        height,
        TextureUsage::RENDER_TARGET,
    );
    let mut descriptor = RenderPassDescriptor::new();
    let other = &mut descriptor.color_attachments[0];
    other.texture = Some(drawn);
    other.load_action = LoadAction::DontCare;
    other.store_action = StoreAction::Store;
    let color = &mut descriptor.color_attachments[1];
    color.texture = Some(target.clone());
    color.store_action = StoreAction::Store;
    color.load_action = match case {
        Case::Kept | Case::Loaded => LoadAction::Load,
        Case::DontCare | Case::DontCareKept => LoadAction::DontCare,
        Case::Cleared => LoadAction::Clear,
    };
    color.clear_color = ClearColor::new(0.5, -2.0, 3e38, 1e-30);
    lane.run(&descriptor, |_, e, _| {
        if draws {
            e.set_render_pipeline_state(&state).unwrap();
            e.set_fragment_buffer(Some(&values), 0, 0).unwrap();
            e.set_fragment_buffer(Some(&shape), 0, 1).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
    })
    .unwrap_or_else(|e| panic!("{pf:?} {case:?}: the pass: {e}"));
    let back = buffer_with(&lane.device, &vec![0xcd; length as usize]).unwrap();
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(width, height, 1),
        &back,
        0,
        row,
        length,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    read_buffer(&back, 0, length).unwrap()
}

/// A pass over a bfloat target that holds every code leaves the device's codes: a load keeps them
/// where nothing draws and where a draw reaches their tiles (the device's store, where Metal's
/// takes them through `half`: its digest is printed), a `DontCare` load under a draw stores
/// zeros, and a clear under a draw stores the clear's, Metal's codes.
#[test]
fn bfloat_load_passes_leave_the_devices_codes() {
    let (messages, _cache, lane) = lane();
    let mut wrong = Vec::new();
    for (pf, channels, _, cleared) in FORMATS {
        let form = ExtendedRange::of(pf).expect("a bfloat format");
        for case in [
            Case::Kept,
            Case::Loaded,
            Case::DontCare,
            Case::DontCareKept,
            Case::Cleared,
        ] {
            // Metal's, but for a loaded target under a draw, whose codes the device's store keeps.
            let metal = match case {
                Case::Kept | Case::DontCareKept | Case::Loaded => EVERY_CODE,
                Case::DontCare => DONT_CARE,
                Case::Cleared => cleared,
            };
            let got = load_case(&lane, pf, channels, case);
            // The model's bytes: every code, zeros, or the clear's texel.
            let model: Vec<u8> = got
                .chunks(2)
                .enumerate()
                .flat_map(|(i, _)| {
                    let code = i as u32 & 0xffff;
                    let texel = form.clear_texel([0.5, -2.0, 3e38, 1e-30], ClearRoute::ThroughHalf);
                    let stored = match case {
                        Case::Kept | Case::DontCareKept | Case::Loaded => code,
                        Case::DontCare => 0,
                        Case::Cleared => texel[i % channels as usize],
                    };
                    u16::try_from(stored).expect("a code").to_le_bytes()
                })
                .collect();
            assert_eq!(digest(&model), metal, "{pf:?} {case:?}: the model's");
            if case == Case::Loaded {
                println!("BFLOAT-TARGETS-KEPT {pf:?}: every code kept, Metal's {LOADED}");
            }
            if digest(&got) != metal {
                let first = got
                    .chunks(2)
                    .zip(model.chunks(2))
                    .position(|(g, m)| g != m)
                    .unwrap_or(0);
                wrong.push(format!(
                    "{pf:?} {case:?}: {}, Metal's {metal}; first at code {first:#06x}",
                    digest(&got)
                ));
            }
        }
    }
    for line in &wrong {
        println!("BFLOAT-TARGETS-DIFFERS {}: {line}", lane.device.name());
    }
    assert!(
        wrong.is_empty(),
        "{} passes differ from Metal's",
        wrong.len()
    );
    quiet(lane, &messages);
}

/// A draw's sweep (the probe `fpv5.swift`'s): a fragment function's output per
/// texel of a full-screen draw.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Sweep {
    /// `float4`, 256 x 256: channel `c` of texel `i` the `float` of code `(i + 16411 c) mod
    /// 65536`, its bits shifted up 16.
    Codes,
    /// `half4`, 256 x 256: channel `c` of texel `i` the `half` of bits `(i + 16411 c) mod 65536`.
    Halves,
    /// `float4`, 512 wide: per code `c` the `float`s of bits `c << 16`, `| 0x7fff`, `| 0x8000`
    /// and `| 0x8001`, value `k` in channel `k % channels` of texel `k / channels`, a missing
    /// alpha 1.
    Ties,
}

/// Metal's digests (SHA-256 of the target's bytes) of each sweep's draw into each bfloat format,
/// at the default `textureWriteRoundingMode` and at `NearestEven` (an Apple M4
/// under macOS 27.0.1).
const DRAWN: [(PixelFormat, Sweep, &str, &str); 9] = [
    (
        PixelFormat::R16Bfloat,
        Sweep::Codes,
        "4ebc4cf7a7f7fdeea4bcc258bd98e21f02274883009d2feda99f6834b335a4b6",
        "ceb2151e45fa832676006dfa73c29488905a503964183dbe14b5fc1a1267f710",
    ),
    (
        PixelFormat::RG16Bfloat,
        Sweep::Codes,
        "5dc9049647ec5be636259e9836a726fe898e24bfa64d672a68a37acefc007bb7",
        "736ed3880a98880efab4a05cd54c4c7b54d44616b8b55c3fa33d608b408dc8d2",
    ),
    (
        PixelFormat::RGBA16Bfloat,
        Sweep::Codes,
        "e929b77e5348a1a6187c76c401615242d13b0d546ad91840b5f3f475ae2bcaf8",
        "e8f780bb75cb189bc8c6e1367b92b1d0666c0c2ae3dfab7edab5025dfcf5b20e",
    ),
    (
        PixelFormat::R16Bfloat,
        Sweep::Halves,
        "37de2c9beecc2633296c26c38c769266445dbdb118bf655d177cddbd3e73914c",
        "37de2c9beecc2633296c26c38c769266445dbdb118bf655d177cddbd3e73914c",
    ),
    (
        PixelFormat::RG16Bfloat,
        Sweep::Halves,
        "8a1fe4aa59d5bc0de89a666a845d444972803ea19d7e74046e0627a6a4210035",
        "8a1fe4aa59d5bc0de89a666a845d444972803ea19d7e74046e0627a6a4210035",
    ),
    (
        PixelFormat::RGBA16Bfloat,
        Sweep::Halves,
        "274607ff4f9b23e56291e9313fcf022649a82c2f269106b3d9195260d361e21f",
        "274607ff4f9b23e56291e9313fcf022649a82c2f269106b3d9195260d361e21f",
    ),
    (
        PixelFormat::R16Bfloat,
        Sweep::Ties,
        "9eae8ab2bddec31f53a91632db57688515b94b7d393b4f62371a9639e0d43bdd",
        "c0bc26e81af628350649afaffaa80513372087a6ad6110b9ab8e4acc8f50c93e",
    ),
    (
        PixelFormat::RG16Bfloat,
        Sweep::Ties,
        "9eae8ab2bddec31f53a91632db57688515b94b7d393b4f62371a9639e0d43bdd",
        "c0bc26e81af628350649afaffaa80513372087a6ad6110b9ab8e4acc8f50c93e",
    ),
    (
        PixelFormat::RGBA16Bfloat,
        Sweep::Ties,
        "9eae8ab2bddec31f53a91632db57688515b94b7d393b4f62371a9639e0d43bdd",
        "c0bc26e81af628350649afaffaa80513372087a6ad6110b9ab8e4acc8f50c93e",
    ),
];

/// The target's width and height, the fragment function, and the values buffer's bytes of
/// `sweep` into a format of `channels` channels.
fn sweep(sweep: Sweep, channels: u64) -> (u64, u64, &'static str, Vec<u8>) {
    let code = |i: u64, c: u64| ((i + 16411 * c) & 0xffff) as u32;
    match sweep {
        Sweep::Codes => {
            let values = (0..65536)
                .flat_map(|i| (0..4).map(move |c| code(i, c) << 16))
                .flat_map(u32::to_le_bytes)
                .collect();
            (256, 256, "rt6_fill_rgba32float", values)
        }
        Sweep::Halves => {
            let values = (0..65536)
                .flat_map(|i| (0..4).map(move |c| code(i, c) as u16))
                .flat_map(u16::to_le_bytes)
                .collect();
            (256, 256, "rt6_fill_rgba16float", values)
        }
        Sweep::Ties => {
            let floats: Vec<u32> = (0..65536u32)
                .flat_map(|c| [0, 0x7fff, 0x8000, 0x8001].map(|low| c << 16 | low))
                .collect();
            let texels = floats.len() as u64 / channels;
            let mut words = vec![0u32; 4 * texels as usize];
            for (k, &v) in floats.iter().enumerate() {
                let k = k as u64;
                words[(4 * (k / channels) + k % channels) as usize] = v;
            }
            if channels < 4 {
                for t in 0..texels as usize {
                    words[4 * t + 3] = 1.0f32.to_bits();
                }
            }
            let values = words.into_iter().flat_map(u32::to_le_bytes).collect();
            (512, texels / 512, "rt6_fill_rgba32float", values)
        }
    }
}

/// The bytes a full-screen draw of `sweep` leaves in a new target of `pf` (`channels` channels)
/// under a pipeline of `textureWriteRoundingMode` `rounding`.
fn drawn(
    lane: &Lane,
    (pf, channels): (PixelFormat, u64),
    sweep_of: Sweep,
    rounding: TextureWriteRoundingMode,
) -> Vec<u8> {
    let (width, height, fragment, values) = sweep(sweep_of, channels);
    let row = 2 * channels * width;
    let length = row * height;
    let library = library(&lane.device).unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut state = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
    state.fragment = Some(function(fragment));
    state.color_attachments[0] = Some(ColorAttachmentDescriptor::new(pf));
    state.texture_write_rounding_mode = rounding;
    let state = lane
        .device
        .new_render_pipeline_state(&state, &PipelineOptions::default())
        .unwrap_or_else(|e| panic!("{pf:?} {sweep_of:?} {rounding:?}: the pipeline: {e}"));
    let values = buffer_with(&lane.device, &values).unwrap();
    let shape = shape(&lane.device, width as u32, 1).unwrap();
    let target = texture(lane, pf, width, height, TextureUsage::RENDER_TARGET);
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::DontCare;
    color.store_action = StoreAction::Store;
    lane.run(&descriptor, |_, e, _| {
        e.set_render_pipeline_state(&state).unwrap();
        e.set_fragment_buffer(Some(&values), 0, 0).unwrap();
        e.set_fragment_buffer(Some(&shape), 0, 1).unwrap();
        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    })
    .unwrap_or_else(|e| panic!("{pf:?} {sweep_of:?} {rounding:?}: the pass: {e}"));
    let back = buffer_with(&lane.device, &vec![0xcd; length as usize]).unwrap();
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(width, height, 1),
        &back,
        0,
        row,
        length,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    read_buffer(&back, 0, length).unwrap()
}

/// Metal's `half` of the `float` of bits `b` as a draw into a bfloat attachment rounds a `float`
/// output: toward zero, saturating at `65504`, or to nearest even where `nearest_even`; a NaN
/// `0x7e00`.
fn metal_output_half(b: u32, nearest_even: bool) -> u16 {
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = b & 0x7fff_ffff;
    if a > 0x7f80_0000 {
        return 0x7e00;
    }
    if a == 0x7f80_0000 || a == 0 {
        return sign | ((a >> 13) as u16 & 0x7c00);
    }
    // The value is `full * 2^exp`; `e` its `half` exponent, at least the denormals' `-14`.
    let (full, exp) = match a >> 23 {
        0 => (a, -149),
        e => ((a & 0x7f_ffff) | 0x80_0000, e as i32 - 150),
    };
    let mut e = (31 - full.leading_zeros() as i32 + exp).max(-14);
    // The bits below the `half`'s last place: at least 13, as a `float` holds 24.
    let shift = (e - 10 - exp) as u32;
    let mut q = match shift {
        1..26 => {
            let (q, rem, tie) = (full >> shift, full & ((1 << shift) - 1), 1 << (shift - 1));
            match nearest_even && (rem > tie || (rem == tie && q & 1 == 1)) {
                true => q + 1,
                false => q,
            }
        }
        _ => 0,
    };
    if q >= 2048 {
        q >>= 1;
        e += 1;
    }
    let bits = match q < 1024 {
        true => q,
        false => ((e + 15) as u32) << 10 | (q & 0x3ff),
    };
    sign | match (bits >= 0x7c00, nearest_even) {
        (false, _) => bits as u16,
        (true, true) => 0x7c00,
        (true, false) => 0x7bff,
    }
}

/// Metal's bfloat code of the `half` of bits `h` a draw's tile holds: its `float` rounded to the
/// nearest bfloat, ties to even, a NaN `0x7fc0`.
fn metal_bfloat(h: u16) -> u16 {
    if h & 0x7c00 == 0x7c00 && h & 0x3ff != 0 {
        return 0x7fc0;
    }
    let b = half_value(h).to_bits();
    let (high, low) = (b >> 16, b & 0xffff);
    let up = low > 0x8000 || (low == 0x8000 && high & 1 == 1);
    (high + u32::from(up)) as u16
}

/// Whether bfloat code `c` is a NaN.
fn bfloat_nan(c: u16) -> bool {
    c & 0x7f80 == 0x7f80 && c & 0x7f != 0
}

/// The target bytes Metal's draw of `sweep_of` into a bfloat format of `channels` channels
/// leaves under `rounding` ([`metal_output_half`], then [`metal_bfloat`]): the model the test
/// holds to Metal's digests in [`DRAWN`].
fn metal_drawn(sweep_of: Sweep, channels: u64, rounding: TextureWriteRoundingMode) -> Vec<u8> {
    let (_, _, _, values) = sweep(sweep_of, channels);
    let halves = matches!(sweep_of, Sweep::Halves);
    let nearest_even = rounding == TextureWriteRoundingMode::NearestEven;
    let lane = if halves { 2 } else { 4 };
    values
        .chunks_exact(4 * lane)
        .flat_map(|texel| {
            texel.chunks_exact(lane).take(channels as usize).map(|v| {
                let h = match halves {
                    true => u16::from_le_bytes([v[0], v[1]]),
                    false => metal_output_half(
                        u32::from_le_bytes([v[0], v[1], v[2], v[3]]),
                        nearest_even,
                    ),
                };
                metal_bfloat(h)
            })
        })
        .flat_map(u16::to_le_bytes)
        .collect()
}

/// A fragment function's `float4` and `half4` outputs to a bfloat attachment store Metal's
/// codes, at the default `textureWriteRoundingMode` and at `NearestEven`: every code's `float`
/// and the `float`s around each code's tie, and every `half` code. Metal's codes are the model
/// [`metal_drawn`] gives, which the test holds to Metal's digests first: the output rounded to
/// `half` by the pipeline's rounding, then to the nearest bfloat, every NaN `0x7fc0`. Every code
/// is Metal's but a NaN, whose bits are the device's (the runtime keeps a NaN a NaN by its quiet
/// bit, `PrecisionNote::DevicePackConversion`, which the runtime then announces).
#[test]
fn bfloat_draws_store_metal_s_codes() {
    let (messages, _cache, lane) = lane();
    let (mut wrong, mut nans) = (Vec::new(), 0);
    for (pf, sweep_of, toward_zero, nearest_even) in DRAWN {
        let channels = FORMATS
            .iter()
            .find(|f| f.0 == pf)
            .map(|f| f.1)
            .expect("a bfloat format");
        for (rounding, metal) in [
            (TextureWriteRoundingMode::TowardZero, toward_zero),
            (TextureWriteRoundingMode::NearestEven, nearest_even),
        ] {
            let model = metal_drawn(sweep_of, channels, rounding);
            assert_eq!(
                digest(&model),
                metal,
                "{pf:?} {sweep_of:?} {rounding:?}: the model of Metal's codes"
            );
            let got = drawn(&lane, (pf, channels), sweep_of, rounding);
            let (mut differ, mut first) = (0, Vec::new());
            let codes = got.as_chunks::<2>().0.iter().zip(model.as_chunks::<2>().0);
            for (k, (g, w)) in codes.enumerate() {
                let (g, w) = (u16::from_le_bytes(*g), u16::from_le_bytes(*w));
                match (g == w, bfloat_nan(g) && bfloat_nan(w)) {
                    (true, _) => {}
                    (false, true) => nans += 1,
                    (false, false) => {
                        differ += 1;
                        if first.len() < 8 {
                            first.push(format!("{k}: {g:04x}, Metal's {w:04x}"));
                        }
                    }
                }
            }
            if differ > 0 {
                wrong.push(format!(
                    "{pf:?} {sweep_of:?} {rounding:?}: {differ} codes, first {first:?}"
                ));
            }
        }
    }
    for line in &wrong {
        println!("BFLOAT-TARGETS-DIFFERS {}: {line}", lane.device.name());
    }
    println!(
        "BFLOAT-TARGETS {}: {nans} NaN codes of the device's bits",
        lane.device.name()
    );
    assert!(
        wrong.is_empty(),
        "{} draws differ from Metal's",
        wrong.len()
    );
    if nans > 0 {
        let note = Degradation::from_precision_note(PrecisionNote::DevicePackConversion);
        let announced = counts(&messages.take_notices());
        assert!(
            announced.contains_key(&note),
            "{nans} NaN codes of the device's bits, unannounced: {announced:?}"
        );
    }
    quiet(lane, &messages);
}
