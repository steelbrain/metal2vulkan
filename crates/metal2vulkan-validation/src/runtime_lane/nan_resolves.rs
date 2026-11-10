//! Device tests of the resolve of NaN and infinite samples a draw writes into a `half`,
//! small-float or `float` multisampled attachment, resolved by the store action of the pass that
//! drew them, against Metal's resolve rule (an M4 under macOS 27.0.1): a `half`
//! resolve counts each NaN sample as 0 and each infinite one as 65536 of its sign, sums the samples
//! in `float` in order, multiplies by `1/n` and rounds to `half` to nearest even; a `float` resolve
//! is the float rule (a NaN anywhere `0x7fc00000`). On every set here that gives what the rules
//! the M3 Max measures under macOS 27.0.1 give (`bind::ColorResolveFormat`'s module docs: the
//! samples floored to a grid below their largest exponent, the average's tie toward +∞, and
//! `RG11B10Float`'s `half` rounded to nearest even on its grid, held to its largest finite value
//! unless infinite); outside these sets the two differ.
//!
//! Each row of one format and sample count writes a set of samples per pixel by a draw at
//! sample rate (`runtime_samples::SampleFill`) into a texture of `PixelFormatView` usage (Metal's
//! rule does not change with it), reads every sample back through its unsigned
//! view, and resolves into a plain 1x texture read by a blit; the rule is applied to the samples
//! as stored. `RG11B10Float`, which no fill writes, is drawn through its `R32Uint` view and
//! resolved by a pass that loads it. Each row prints `RESOLVE-NANINF`: how many sets resolve as
//! the rule says, the first that do not, and whether the device announced
//! `device-rounded-resolve`. A row that differs unannounced fails. (the coordinator's question:) so
//! does a drawn sample stored other than as written where the device announced neither
//! `device-nan-encoding` (the fragment function's note, a `half` output's NaN the device stores as
//! it converts it) nor `native-nan-encoding` (the fill's read of a `half` from its bits is the
//! device's widening, a NaN to the device's NaN, which the store keeps: NVIDIA's 615.71.09 on the
//! RTX 5080 Laptop GPU stores `0x7fff` for every one), or where a stored channel differs from the
//! written one other than as one NaN for another; each row prints whether the notices came.

use super::degrade_notices::counts;
use super::renders::{blit_read, run, small_nan};
use super::{at_plan, validated_device, validated_device_with, Messages, ScratchCache};
use crate::candidate::runtime_samples::{encode as samples_of, SampleFill};
use crate::case::{TextureFormat as CaseFormat, TextureType as CaseType};
use crate::executor_contract::multisample::{ReadbackView, SampleRead};
use crate::gpu::TEST_DEVICE;
use crate::literal::texture_layout;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    Degradation, Error, LoadAction, RenderPassDescriptor, ResourceOptions, StoreAction, Texture,
    TextureDescriptor,
};

/// The `half` sample sets, four samples each (the first two at 2x): NaNs of both signs and a
/// payload, the infinities, and mixtures with finite values.
const HALF_SETS: [[u16; 4]; 14] = [
    [0x7e00, 0x7e00, 0x7e00, 0x7e00],
    [0xfe00, 0xfe00, 0xfe00, 0xfe00],
    [0x7f01, 0x7f01, 0x7f01, 0x7f01],
    [0x7c00, 0x7c00, 0x7c00, 0x7c00],
    [0xfc00, 0xfc00, 0xfc00, 0xfc00],
    [0x7e00, 0x3c00, 0x3c00, 0x3c00],
    [0x3c00, 0xfe00, 0x3800, 0x3400],
    [0x7c00, 0x3c00, 0x3c00, 0x3c00],
    [0x7c00, 0xfc00, 0x3c00, 0x3c00],
    [0xfc00, 0x3c00, 0x7e00, 0x3c00],
    [0x7bff, 0x7bff, 0x7bff, 0x7bff],
    [0x7bff, 0x7c00, 0x3c00, 0x0000],
    [0x3c00, 0x3800, 0x3400, 0x3000],
    [0x0001, 0x8001, 0x3c00, 0x3c00],
];

/// The `float` sample sets, as [`HALF_SETS`].
const FLOAT_SETS: [[u32; 4]; 12] = [
    [0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000],
    [0xffc0_0000, 0xffc0_0000, 0xffc0_0000, 0xffc0_0000],
    [0x7fc0_0001, 0x7fc0_0001, 0x7fc0_0001, 0x7fc0_0001],
    [0x7f80_0000, 0x7f80_0000, 0x7f80_0000, 0x7f80_0000],
    [0xff80_0000, 0xff80_0000, 0xff80_0000, 0xff80_0000],
    [0x7fc0_0000, 0x3f80_0000, 0x3f80_0000, 0x3f80_0000],
    [0x3f80_0000, 0xffc0_0000, 0x3f00_0000, 0x3e80_0000],
    [0x7f80_0000, 0x3f80_0000, 0x3f80_0000, 0x3f80_0000],
    [0x7f80_0000, 0xff80_0000, 0x3f80_0000, 0x3f80_0000],
    [0x7f7f_ffff, 0x7f7f_ffff, 0x7f7f_ffff, 0x7f7f_ffff],
    [0x3f80_0000, 0x3f00_0000, 0x3e80_0000, 0x3e00_0000],
    [0x0000_0001, 0x8000_0001, 0x3f80_0000, 0x3f80_0000],
];

/// `RG11B10Float` sample sets, each sample one packed word: NaNs, infinities, and mixtures with
/// 1.0 (`0x3c0` in the 11-bit fields, `0x1e0` in the 10-bit one) and 0.5.
const SMALL_SETS: [[u32; 4]; 6] = [
    [pack(0x7e0, 0x7e0, 0x3f0); 4],
    [pack(0x7c0, 0x7c0, 0x3e0); 4],
    [
        pack(0x7e0, 0x7c0, 0x1e0),
        pack(0x3c0, 0x3c0, 0x1e0),
        pack(0x3c0, 0x3c0, 0x1e0),
        pack(0x3c0, 0x3c0, 0x1e0),
    ],
    [
        pack(0x3c0, 0x7e0, 0x3e0),
        pack(0x380, 0x3c0, 0x1e0),
        pack(0x3c0, 0x380, 0x1c0),
        pack(0x380, 0x380, 0x1c0),
    ],
    [pack(0x3c0, 0x380, 0x1c0); 4],
    [
        pack(0x7c0, 0x380, 0x3f0),
        pack(0x000, 0x380, 0x1e0),
        pack(0x7c0, 0x3c0, 0x3e0),
        pack(0x3c0, 0x000, 0x000),
    ],
];

/// An `RG11B10Float` word of its red, green and blue fields.
const fn pack(red: u32, green: u32, blue: u32) -> u32 {
    red | (green << 11) | (blue << 22)
}

/// A field of `bits` of a float of 5 exponent bits and `mantissa` bits (signed where `signed`,
/// its sign the bit above) as Metal's resolve counts it: a NaN 0, an infinity 65536 of its sign.
fn counted(bits: u32, mantissa: u32, signed: bool) -> f32 {
    let negative = signed && ((bits >> (mantissa + 5)) & 1) == 1;
    let exponent = (bits >> mantissa) & 0x1f;
    let fraction = bits & ((1 << mantissa) - 1);
    let magnitude = match exponent {
        0x1f if fraction != 0 => 0.0,
        0x1f => 65536.0,
        0 => fraction as f32 * 2f32.powi(-14 - mantissa as i32),
        e => (1.0 + fraction as f32 / (1u32 << mantissa) as f32) * 2f32.powi(e as i32 - 15),
    };
    if negative {
        -magnitude
    } else {
        magnitude
    }
}

/// `x` as a float of 5 exponent bits and `mantissa` bits, to nearest even, an overflow its
/// infinity; an unsigned one stores a negative `x` as 0.
fn encoded(x: f32, mantissa: u32, signed: bool) -> u32 {
    let sign = match signed && x.is_sign_negative() {
        true => 1 << (mantissa + 5),
        false => 0,
    };
    let a = f64::from(x.abs());
    if (!signed && x < 0.0) || a == 0.0 {
        return sign;
    }
    let exponent = (a.log2().floor() as i32).max(-14);
    let spacing = 2f64.powi(exponent - mantissa as i32);
    let mut steps = (a / spacing).round_ties_even() as u64;
    let mut exponent = exponent;
    if steps >= 2 << mantissa {
        exponent += 1;
        steps /= 2;
    }
    if exponent > 15 {
        return sign | (0x1f << mantissa);
    }
    match steps < 1 << mantissa {
        true => sign | steps as u32,
        false => sign | (((exponent + 15) as u32) << mantissa) | (steps as u32 - (1 << mantissa)),
    }
}

/// Metal's resolve of a field's samples: each counted, summed in `float` in order, times `1/n`,
/// to nearest even.
fn small_rule(samples: &[u32], mantissa: u32, signed: bool) -> u32 {
    let n = samples.len() as f32;
    let sum = samples
        .iter()
        .fold(0f32, |sum, &s| sum + counted(s, mantissa, signed));
    encoded(sum * (1.0 / n), mantissa, signed)
}

/// The float rule: each sample, partial sum and the result flushed to the zero of its sign, a NaN
/// anywhere `0x7fc00000`.
fn float_rule(samples: &[u32]) -> u32 {
    let flush = |x: f32| match x.is_subnormal() {
        true => 0f32.copysign(x),
        false => x,
    };
    let mut sum = 0f32;
    for (i, &s) in samples.iter().enumerate() {
        let x = flush(f32::from_bits(s));
        sum = match i {
            0 => x,
            _ => flush(sum + x),
        };
    }
    let r = flush(sum * (1.0 / samples.len() as f32));
    match r.is_nan() || samples.iter().any(|&s| f32::from_bits(s).is_nan()) {
        true => 0x7fc0_0000,
        false => r.to_bits(),
    }
}

/// One format of the lane: its pixel format, its fill's case format (`None`: through its
/// `R32Uint` view), its readback view, its texel bytes, and its sets as each pixel's samples'
/// bytes.
struct Row {
    pf: PixelFormat,
    fill: Option<CaseFormat>,
    view: ReadbackView,
    bytes: usize,
    sets: Vec<Vec<Vec<u8>>>,
}

/// Each row: `R16Float`, `RG16Float` and `RGBA16Float` with [`HALF_SETS`] in every channel,
/// `R32Float` with [`FLOAT_SETS`], `RG11B10Float` with [`SMALL_SETS`].
fn rows() -> Vec<Row> {
    let half = |channels: usize| -> Vec<Vec<Vec<u8>>> {
        HALF_SETS
            .iter()
            .map(|set| {
                set.iter()
                    .map(|h| h.to_le_bytes().repeat(channels))
                    .collect()
            })
            .collect()
    };
    let words = |sets: &[[u32; 4]]| -> Vec<Vec<Vec<u8>>> {
        sets.iter()
            .map(|set| set.iter().map(|w| w.to_le_bytes().to_vec()).collect())
            .collect()
    };
    vec![
        Row {
            pf: PixelFormat::R16Float,
            fill: Some(CaseFormat::R16Float),
            view: ReadbackView::R16Uint,
            bytes: 2,
            sets: half(1),
        },
        Row {
            pf: PixelFormat::RG16Float,
            fill: Some(CaseFormat::Rg16Float),
            view: ReadbackView::R32Uint,
            bytes: 4,
            sets: half(2),
        },
        Row {
            pf: PixelFormat::RGBA16Float,
            fill: Some(CaseFormat::Rgba16Float),
            view: ReadbackView::Rg32Uint,
            bytes: 8,
            sets: half(4),
        },
        Row {
            pf: PixelFormat::R32Float,
            fill: Some(CaseFormat::R32Float),
            view: ReadbackView::R32Uint,
            bytes: 4,
            sets: words(&FLOAT_SETS),
        },
        Row {
            pf: PixelFormat::RG11B10Float,
            fill: None,
            view: ReadbackView::R32Uint,
            bytes: 4,
            sets: words(&SMALL_SETS),
        },
    ]
}

/// Metal's resolve of `samples` (each a texel's bytes) of `pf`, by its rule.
fn metal_resolve(pf: PixelFormat, samples: &[&[u8]]) -> Vec<u8> {
    let word = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    match pf {
        PixelFormat::R32Float => float_rule(&samples.iter().map(|s| word(s)).collect::<Vec<_>>())
            .to_le_bytes()
            .to_vec(),
        PixelFormat::RG11B10Float => {
            let words: Vec<u32> = samples.iter().map(|s| word(s)).collect();
            let field = |at: u32, width: u32| {
                let fields: Vec<u32> = words
                    .iter()
                    .map(|w| (w >> at) & ((1 << width) - 1))
                    .collect();
                small_rule(&fields, width - 5, false) << at
            };
            (field(0, 11) | field(11, 11) | field(22, 10))
                .to_le_bytes()
                .to_vec()
        }
        _ => {
            let channels = samples[0].len() / 2;
            (0..channels)
                .flat_map(|c| {
                    let halves: Vec<u32> = samples
                        .iter()
                        .map(|s| u32::from(u16::from_le_bytes([s[2 * c], s[2 * c + 1]])))
                        .collect();
                    (small_rule(&halves, 10, true) as u16).to_le_bytes()
                })
                .collect()
        }
    }
}

/// Write each row's sets by a draw, resolve them by the store action (or, `RG11B10Float`, by a
/// pass that loads them), and print each row against Metal's rule (the module docs).
#[test]
fn drawn_nan_and_infinite_samples_resolve_by_metal_s_rule() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
    let name = device.name().to_owned();
    let queue = device.new_command_queue().unwrap();
    let mut silent = Vec::new();
    // The notice comes once a device: whether it came by this row.
    let mut announced = false;
    let mut nan_announced = false;
    // Whether the fill's fragment function announced `native-nan-encoding`: its read of a `half`
    // from its bits is the device's widening, a NaN to the device's NaN, which the store keeps.
    let mut widened_own = false;
    for row in rows() {
        for samples in [2u32, 4] {
            let label = format!("{:?} {samples}x", row.pf);
            let width = row.sets.len() as u32;
            let texture = |samples: u32, usage: TextureUsage| -> Result<Texture, Error> {
                let mut d = TextureDescriptor::texture_2d(row.pf, width.into(), 1, false);
                if samples > 1 {
                    d.texture_type = TextureType::Type2DMultisample;
                    d.sample_count = samples.into();
                }
                d.usage = usage;
                d.resource_options = ResourceOptions::private();
                device.new_texture(&d)
            };
            let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            // A device without the sample count refuses the texture (lavapipe at 2x).
            let target = match texture(samples, usage | TextureUsage::PIXEL_FORMAT_VIEW) {
                Ok(target) => target,
                Err(Error::Unsupported(why)) => {
                    println!("RESOLVE-NANINF {label} on {name}: unsupported {why:?}");
                    continue;
                }
                Err(error) => panic!("{label}: {error}"),
            };
            let resolved = texture(1, usage).unwrap();
            let bytes: Vec<u8> = row
                .sets
                .iter()
                .flat_map(|set| set[..samples as usize].concat())
                .collect();
            let layout = texture_layout(CaseType::D2Multisample, [width, 1, 1], samples).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let read_from = match row.fill {
                Some(format) => {
                    let fill = SampleFill::new(&device, &target, format, layout, &bytes).unwrap();
                    fill.encode_resolving(&command_buffer, &resolved).unwrap();
                    target.clone()
                }
                None => {
                    let view = target.new_texture_view(PixelFormat::R32Uint).unwrap();
                    let fill = SampleFill::new(&device, &view, CaseFormat::R32Uint, layout, &bytes)
                        .unwrap();
                    fill.encode(&command_buffer).unwrap();
                    let mut pass = RenderPassDescriptor::new();
                    let attachment = &mut pass.color_attachments[0];
                    attachment.texture = Some(target.clone());
                    attachment.load_action = LoadAction::Load;
                    attachment.resolve_texture = Some(resolved.clone());
                    attachment.store_action = StoreAction::StoreAndMultisampleResolve;
                    let encoder = command_buffer.render_command_encoder(&pass).unwrap();
                    encoder.end_encoding().unwrap();
                    view
                }
            };
            let read = samples_of(
                &device,
                &command_buffer,
                &read_from,
                SampleRead::Color(row.view),
                [width, 1],
                samples,
            )
            .unwrap();
            run(&command_buffer);
            let stored = read.bytes().unwrap();
            let got = blit_read(&device, &resolved, row.bytes as u64);
            let notices = counts(&messages.take_notices());
            announced |= notices.contains_key(&Degradation::DeviceRoundedResolve);
            nan_announced |= notices.contains_key(&Degradation::DeviceNanEncoding);
            widened_own |= notices.contains_key(&Degradation::NativeNanEncoding);
            // A channel's bytes: a `float`'s 4, a `half`'s 2 (a packed format's word is one).
            let channel = match row.pf {
                PixelFormat::R32Float | PixelFormat::RG11B10Float => 4,
                _ => 2,
            };
            let is_nan = |c: &[u8]| match c.len() {
                2 => small_nan(u32::from(u16::from_le_bytes([c[0], c[1]])) & 0x7fff, 5, 10),
                _ => f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_nan(),
            };
            let mut beyond_nans = Vec::new();
            let texel = row.bytes * samples as usize;
            let (mut exact, mut differ, mut rewritten) = (0usize, Vec::new(), 0usize);
            for (pixel, set) in row.sets.iter().enumerate() {
                let held: Vec<&[u8]> = stored[pixel * texel..(pixel + 1) * texel]
                    .chunks_exact(row.bytes)
                    .collect();
                if held.iter().zip(set).any(|(h, s)| *h != s.as_slice()) {
                    rewritten += 1;
                    // Each channel stored as written, or a written NaN stored as a NaN.
                    let nan_only = held.iter().zip(set).all(|(h, s)| {
                        h.chunks_exact(channel)
                            .zip(s.chunks_exact(channel))
                            .all(|(h, s)| h == s || (row.fill.is_some() && is_nan(h) && is_nan(s)))
                    });
                    if !nan_only {
                        beyond_nans.push(format!("{:02x?} stored {:02x?}", set, held));
                    }
                }
                let metal = metal_resolve(row.pf, &held);
                let resolve = &got[pixel * row.bytes..(pixel + 1) * row.bytes];
                match resolve == metal.as_slice() {
                    true => exact += 1,
                    false => differ.push(format!(
                        "{:02x?} -> {resolve:02x?}, rule {metal:02x?}",
                        held.concat()
                    )),
                }
            }
            println!(
                "RESOLVE-NANINF {label} on {name}: {exact}/{width} sets as Metal's rule{}; \
                 device-rounded-resolve {}{}",
                if rewritten > 0 {
                    format!(
                        " ({rewritten} sets' samples not stored as written, device-nan-encoding \
                         {}, native-nan-encoding {})",
                        if nan_announced {
                            "announced"
                        } else {
                            "not announced"
                        },
                        if widened_own {
                            "announced"
                        } else {
                            "not announced"
                        }
                    )
                } else {
                    String::new()
                },
                if announced {
                    "announced"
                } else {
                    "not announced"
                },
                match differ.is_empty() {
                    true => String::new(),
                    false => format!("; differ: {}", differ[..differ.len().min(6)].join("; ")),
                }
            );
            if !differ.is_empty() && !announced {
                silent.push(format!("{label}: {}", differ.join("; ")));
            }
            if rewritten > 0 && !(nan_announced || widened_own) {
                silent.push(format!(
                    "{label}: {rewritten} sets stored unlike written, unannounced"
                ));
            }
            if !beyond_nans.is_empty() {
                silent.push(format!(
                    "{label}: stored beyond a NaN's bits: {beyond_nans:?}"
                ));
            }
        }
    }
    drop(cache);
    assert!(
        silent.is_empty(),
        "resolves unlike Metal's rule, or samples unlike written, unannounced: {silent:#?}"
    );
}

/// A blit copy between multisampled textures keeps every sample's bits, NaNs included, as
/// Metal's copy does, or announces `device-nan-encoding`: each row's sets written
/// into a 4x texture of `PixelFormatView` usage through an unsigned view of its texel size (which
/// stores the bits on every device), copied by `copy_texture` into another, and both read back
/// through their unsigned view. The copy goes into a `PixelFormatView` texture and into a plain one
/// (render target and shader read), read through a second copy into a `PixelFormatView` texture.
/// Each row prints `BLIT-NAN`: how many sets' samples the copies kept, and whether the notice has
/// come on the device. A row whose copies changed a sample unannounced fails, as does one that
/// changed a sample beyond one NaN for another.
#[test]
fn blit_copied_nan_samples_keep_their_bits_or_announce() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let name = device.name().to_owned();
    let queue = device.new_command_queue().unwrap();
    let samples = 4u32;
    let viewed =
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
    let plain = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let (mut nan_announced, mut wrong) = (false, Vec::new());
    for row in rows() {
        for (kind, usage) in [("PixelFormatView", viewed), ("plain", plain)] {
            let label = format!("{:?} {samples}x into a {kind} texture", row.pf);
            let width = row.sets.len() as u32;
            let new = |usage| {
                let mut d = TextureDescriptor::texture_2d(row.pf, width.into(), 1, false);
                d.texture_type = TextureType::Type2DMultisample;
                d.sample_count = samples.into();
                d.usage = usage;
                d.resource_options = ResourceOptions::private();
                device.new_texture(&d)
            };
            let (source, copy, read_copy) = match (new(viewed), new(usage), new(viewed)) {
                (Ok(source), Ok(copy), Ok(read_copy)) => (source, copy, read_copy),
                (Err(Error::Unsupported(why)), ..)
                | (_, Err(Error::Unsupported(why)), _)
                | (.., Err(Error::Unsupported(why))) => {
                    println!("BLIT-NAN {label} on {name}: unsupported {why:?}");
                    continue;
                }
                (Err(error), ..) | (_, Err(error), _) | (.., Err(error)) => {
                    panic!("{label}: {error}")
                }
            };
            // The unsigned view the bits are written through, and its fill's format.
            let (view_format, fill) = match row.bytes {
                2 => (PixelFormat::R16Uint, CaseFormat::R16Uint),
                4 => (PixelFormat::R32Uint, CaseFormat::R32Uint),
                _ => (PixelFormat::RGBA16Uint, CaseFormat::Rgba16Uint),
            };
            let bytes: Vec<u8> = row
                .sets
                .iter()
                .flat_map(|set| set[..samples as usize].concat())
                .collect();
            let layout = texture_layout(CaseType::D2Multisample, [width, 1, 1], samples).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let view = source.new_texture_view(view_format).unwrap();
            SampleFill::new(&device, &view, fill, layout, &bytes)
                .unwrap()
                .encode(&command_buffer)
                .unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_texture(&source, &copy).unwrap();
            // A plain texture is read through a copy into a `PixelFormatView` one.
            let copied = match usage == plain {
                true => {
                    blit.copy_texture(&copy, &read_copy).unwrap();
                    &read_copy
                }
                false => &copy,
            };
            blit.end_encoding().unwrap();
            // A format the readback reads (each fill's) is read as it is; `RG11B10Float`
            // through its unsigned view, as it was written.
            let read = |texture: &Texture| {
                let read_from = match row.fill {
                    Some(_) => texture.clone(),
                    None => texture.new_texture_view(view_format).unwrap(),
                };
                samples_of(
                    &device,
                    &command_buffer,
                    &read_from,
                    SampleRead::Color(row.view),
                    [width, 1],
                    samples,
                )
                .unwrap()
            };
            let (written, copied) = (read(&source), read(copied));
            run(&command_buffer);
            let (written, copied) = (written.bytes().unwrap(), copied.bytes().unwrap());
            nan_announced |=
                counts(&messages.take_notices()).contains_key(&Degradation::DeviceNanEncoding);
            let texel = row.bytes * samples as usize;
            let (mut kept, mut changed, mut beyond) = (0usize, Vec::new(), Vec::new());
            for pixel in 0..width as usize {
                let at = pixel * texel..(pixel + 1) * texel;
                let (w, c) = (&written[at.clone()], &copied[at]);
                if w == c {
                    kept += 1;
                    continue;
                }
                changed.push(format!("{w:02x?} -> {c:02x?}"));
                if !nan_for_nan(row.pf, w, c) {
                    beyond.push(format!("{w:02x?} -> {c:02x?}"));
                }
            }
            println!(
                "BLIT-NAN {label} on {name}: {kept}/{width} sets' samples kept by the copies; \
                 device-nan-encoding {}{}",
                if nan_announced {
                    "announced"
                } else {
                    "not announced"
                },
                match changed.is_empty() {
                    true => String::new(),
                    false => format!("; changed: {}", changed[..changed.len().min(4)].join("; ")),
                }
            );
            if !changed.is_empty() && !nan_announced {
                wrong.push(format!(
                    "{label}: {} sets changed unannounced",
                    changed.len()
                ));
            }
            if !beyond.is_empty() {
                wrong.push(format!("{label}: changed beyond a NaN's bits: {beyond:?}"));
            }
        }
    }
    drop(cache);
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Whether texels `w` and `c` of `pf` differ only where each holds a NaN: field by field, a
/// `float`'s 32 bits, a `half`'s 16, or a packed `RG11B10Float` word's 11, 11 and 10.
fn nan_for_nan(pf: PixelFormat, w: &[u8], c: &[u8]) -> bool {
    // Each field: its bit offset in a 32-bit word, its width and its mantissa's.
    let fields: &[(u32, u32, u32)] = match pf {
        PixelFormat::R32Float => &[(0, 32, 23)],
        PixelFormat::RG11B10Float => &[(0, 11, 6), (11, 11, 6), (22, 10, 5)],
        _ => &[(0, 16, 10), (16, 16, 10)],
    };
    let is_nan = |word: u32, &(at, width, mantissa): &(u32, u32, u32)| {
        let field = (u64::from(word) >> at) & ((1u64 << width) - 1);
        let exponent = width - mantissa - u32::from(width > 11);
        let magnitude = field & ((1u64 << (mantissa + exponent)) - 1);
        magnitude >> mantissa == (1u64 << exponent) - 1 && magnitude & ((1u64 << mantissa) - 1) != 0
    };
    let words = |b: &[u8]| -> Vec<u32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    };
    words(w).iter().zip(words(c).iter()).all(|(&w, &c)| {
        fields.iter().all(|field| {
            let (at, width, _) = *field;
            let mask = ((1u64 << width) - 1) << at;
            u64::from(w) & mask == u64::from(c) & mask || (is_nan(w, field) && is_nan(c, field))
        })
    })
}
