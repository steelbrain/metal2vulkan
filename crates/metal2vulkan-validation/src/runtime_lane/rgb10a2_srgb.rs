//! `RGB10A2Unorm_sRGB` (586) textures on the device the cfg selects, under the validation layer,
//! against Metal's answers.
//!
//! Metal's answers: an Apple M3 Max under macOS 27.0.1, our own probes (`codes.m`, `s.m` and
//! `r.m`, no guest data), each the SHA-256 (first 16 hex digits) of the probe's printed lines,
//! which the lane prints alike from what the device returns:
//!
//! - **Reads.** A 1024 x 1 texture of words `c | (1023 - c) << 10 | (7 c mod 1024) << 20 |
//!   (c mod 4) << 30`, each texel read by a kernel (`fpv_bfloat_fetch`): `read <c> <r> <g> <b>
//!   <a>`, the `float` lanes' bits.
//! - **Writes.** A kernel's write (`fpv_bfloat_write`) of `k / 16384` in red, green and blue and
//!   `(k mod 4096) / 4095` in alpha, `k` from 0 to 16384, then 16 specials in every channel, into a
//!   texture 4096 texels wide: `write <k> <word>`. The grid is 4096 by 5, so it writes 20,480
//!   texels and the 4,079 past the sweep take the input's zeros; every one of them is digested.
//! - **Samples.** A 16 x 16 texture of the probe's xorshift words, sampled through a runtime
//!   sampler (clamp to edge, nearest and linear) at 4096 of its positions `(n, m) / 4096`, with
//!   `float` and `half` results (`xr_sample_runtime`, `xr_sample_runtime_half`): `s <linear> <f|h>
//!   <n> <m>` and the lanes' bits.
//! - **Clears.** 2,069 clear colors `(v, v / 2, v / 4, a)` through a `Clear` load action and no
//!   draw, into a 1 x 1 target of render target and shader read usage and a 64 x 64 one of render
//!   target usage alone: `clear <shape> <k> <word>` of its last texel. Both shapes clear through
//!   `half`.
//! - **Stores.** A full-screen draw of the owned fixture `rt6_samples`'s fill (`float4` and
//!   `half4` outputs) of the writes' values into a new 4096 x 5 target: `store <float|half> <k>
//!   <word>`. A `float` output rounds to `half` toward zero first. The draw covers all 20,480
//!   texels and every one of them is digested; the 4,075 past the sweep take the input's zeros and
//!   Metal stores `0x00000000` in each (`rgb10a2-srgb-stores-f` and `-h`).
//! - **Loads.** A 1024 x 4 target holding every color code in each channel and each alpha code,
//!   loaded and stored by a pass whose draw writes another attachment alone: `load <i> <before>
//!   <after>`. Each color code moves as `R10Unorm_X6_sRGB`'s does through Metal's table, alpha's
//!   stays; the device's store keeps every code (`extended-range-bits-kept`), so the lane holds
//!   it to the words loaded and prints Metal's digest beside it.
//!
//! The device's `isRGB10A2GammaSupported` answers as the texture's linear filter does. A
//! difference is printed on an `RGB10A2-SRGB-DIFFERS` line.

use super::degrade_notices::counts;
use super::degrade_notices::on_device_with;
use super::extended_range_samples::state_of;
use super::extended_range_targets::half_value;
use super::resolves::{lane_with, quiet, Lane};
use super::xr_codes::{
    assert_announced, assert_counted, half_within, recorded, value_within, written_within,
};
use super::{at_plan, buffer_with, read_buffer, WAIT};
use crate::candidate::runtime_samples::{library, shape};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::format::ExtendedRange;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, SamplerMinMagFilter, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, PrecisionNote, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBufferStatus, ComputePipelineState, Degradation, Device,
    LoadAction, Origin, PipelineOptions, PrivateFlag, Region, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, SamplerState, Size, StoreAction, Texture,
    TextureDescriptor,
};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;

const PF: PixelFormat = PixelFormat::RGB10A2UnormSRGB;
/// Its form.
const FORM: ExtendedRange = ExtendedRange::Rgb10A2Srgb;

/// Metal's digests: reads, writes, samples (nearest `float`, nearest `half`, linear `float`,
/// linear `half`), clears (1 x 1, 64 x 64), stores (`float`, `half`) and loads.
const READS: &str = "492c1e326a4685c8";
/// The writes' digest over the whole 4096 by 5 grid, from the 20,480 words the conformance
/// recording `rgb10a2-srgb-writes` holds of this chip's Metal run (the first 16,401 of them
/// digest to `631db99cd110dab6`, the digest this lane compared while it printed only those).
const WRITES: &str = "59cd54549c914e4e";
const SAMPLES: [[&str; 2]; 2] = [
    ["5f73627c39064d55", "4460054d432720d8"],
    ["36b79f318a83d9ac", "e1978cceb4591a05"],
];
const CLEARS: [&str; 2] = ["325c8263917a086f", "8b361b1736bc7625"];
/// The stores' digests over the whole 4096 by 5 target, from the 20,480 words each of the
/// conformance recordings `rgb10a2-srgb-stores-f` and `rgb10a2-srgb-stores-h` holds of this chip's
/// Metal run. The first 16,405 of those same words digest to `27b2e44d676d4976` and
/// `26b22082a3e6db54`, the digests this lane compared while it printed only those, and the 4,075
/// past them are `0x00000000` on Metal.
const STORES: [&str; 2] = ["92b08315643e0b2a", "cf64ce62b70ddfe3"];
const LOADS: &str = "308f150fdcbbb82a";

/// Every word `question` read back has a line in `text`, the string its digest is taken over: a
/// formatter that prints fewer lines than the readback holds digests less than it appears to, and
/// the question then passes whatever lies past the last line it printed.
fn covers(text: &str, read_back: usize, question: &str) {
    assert_eq!(
        text.lines().count(),
        read_back,
        "{question}: the digested lines do not cover every word read back"
    );
}

/// The first 16 hex digits of the SHA-256 of `text`.
fn digest(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A 2D texture of the format, `width` x `height`, of `usage`, in `options`' storage.
fn texture(
    device: &Device,
    (width, height): (u64, u64),
    usage: TextureUsage,
    options: ResourceOptions,
) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PF, width, height, false);
    descriptor.usage = usage;
    descriptor.resource_options = options;
    device
        .new_texture(&descriptor)
        .unwrap_or_else(|e| panic!("a 586 texture: {e}"))
}

/// A texture's words, row after row.
fn words_of(texture: &Texture, (width, height): (u64, u64)) -> Vec<u32> {
    let mut bytes = vec![0u8; (4 * width * height) as usize];
    texture
        .get_bytes(
            &mut bytes,
            4 * width,
            0,
            Region::new_2d(0, 0, width, height),
            0,
            0,
        )
        .unwrap();
    words(&bytes)
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// Run `state` over `texture` with `buffers` at 0 and up, `threads` (width, height) of them.
fn dispatch(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
    (sampler, buffers): (Option<&SamplerState>, &[&Buffer]),
    threads: (u64, u64),
) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    if let Some(sampler) = sampler {
        encoder.set_sampler_state(Some(sampler), 0).unwrap();
    }
    for (index, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(Some(buffer), 0, index as u32).unwrap();
    }
    encoder
        .dispatch_threads(Size::new(threads.0, threads.1, 1), Size::new(64, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
}

/// The writes' and stores' specials (`codes.m`).
const WRITE_SPECIALS: [f32; 16] = [
    -0.0,
    -1.0,
    f32::NAN,
    -f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    1.5,
    2.0,
    1e-30,
    1e-45,
    0.999_999_94,
    1.000_000_1,
    0.5 / 8192.0,
    1.5 / 8192.0,
    0.003_130_8,
    0.040_45,
];

/// The clears' and stores' specials (`r.m`).
const RENDER_SPECIALS: [f32; 20] = [
    -0.0,
    -1.0,
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    1.5,
    2.0,
    1e-30,
    0.999_999_94,
    1.000_000_1,
    0.5 / 8192.0,
    1.5 / 8192.0,
    0.003_130_8,
    0.040_45,
    65504.0,
    65520.0,
    0.333_333_34,
    0.166_666_67,
    0.5,
    0.833_333_3,
];

/// The values the probes write or draw at `k`: `k / 16384` in red, green and blue and `(k mod
/// 4096) / 4095` in alpha, then each special in every channel; zeros up to `len`.
fn values(specials: &[f32], len: usize) -> Vec<[f32; 4]> {
    let n = 16385 + specials.len();
    (0..len)
        .map(|k| match k {
            _ if k < 16385 => {
                let v = k as f32 / 16384.0;
                [v, v, v, (k % 4096) as f32 / 4095.0]
            }
            _ if k < n => [specials[k - 16385]; 4],
            _ => [0.0; 4],
        })
        .collect()
}

/// The `half` of `v`, rounded to nearest even (MSL's conversion), a NaN `0x7e00`.
fn half_bits(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7e00;
    }
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = f64::from(v.abs());
    if a >= 65520.0 {
        return sign | 0x7c00;
    }
    // The nearest multiple of the `half` ulp at `a`, ties to even, exactly in `f64`.
    let e = if a < 2f64.powi(-14) {
        -24
    } else {
        a.log2().floor() as i32 - 10
    };
    let q = (a / 2f64.powi(e)).round_ties_even() * 2f64.powi(e);
    let h = if q < 2f64.powi(-14) {
        (q / 2f64.powi(-24)) as u16
    } else {
        let e = q.log2().floor() as i32;
        let m = (q / 2f64.powi(e) - 1.0) * 1024.0;
        (((e + 15) as u16) << 10) | m as u16
    };
    sign | h
}

/// The xorshift step of the sample probe (`s.m`).
fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// Reads, writes and samples of a 586 texture by a kernel are Metal's.
#[test]
fn kernels_read_write_and_sample_rgb10a2_srgb_as_metal_does() {
    on_device_with(at_plan, |device, messages| {
        let mut wrong = Vec::new();
        // The values within the device's conversion of Metal's, and the linear samples apart.
        let (mut moved, mut filtered) = (0, 0);
        let shared = ResourceOptions::shared();
        // Reads.
        let fetch = state_of(device, "fpv_bfloat_textures", "fpv_bfloat_fetch");
        let source: Vec<u32> = (0..1024u32)
            .map(|c| c | (1023 - c) << 10 | (7 * c % 1024) << 20 | (c % 4) << 30)
            .collect();
        let t = texture(device, (1024, 1), TextureUsage::SHADER_READ, shared);
        let bytes: Vec<u8> = source.iter().flat_map(|w| w.to_le_bytes()).collect();
        t.replace_region(Region::new_2d(0, 0, 1024, 1), 0, 0, &bytes, 4096, 0)
            .unwrap();
        let out = buffer_with(device, &vec![0xab; 1024 * 16]).unwrap();
        let width = buffer_with(device, &1024u32.to_le_bytes()).unwrap();
        dispatch(device, &fetch, &t, (None, &[&out, &width]), (1024, 1));
        let got = words(&read_buffer(&out, 0, 1024 * 16).unwrap());
        let metal = words(&recorded("rgb10a2-srgb-reads", &["output", "codes"]));
        let text = |words: &[u32]| {
            let mut text = String::new();
            for (c, w) in words.as_chunks::<4>().0.iter().enumerate() {
                writeln!(
                    text,
                    "read {c} {:08x} {:08x} {:08x} {:08x}",
                    w[0], w[1], w[2], w[3]
                )
                .unwrap();
            }
            text
        };
        assert_eq!(digest(&text(&metal)), READS, "the recorded reads");
        covers(&text(&got), got.len() / 4, "reads");
        let (near, past) = lanes_within(&got, &metal, |g, w, c| {
            value_within(FORM, f32::from_bits(g), f32::from_bits(w), c == 3)
        });
        moved += near;
        if past > 0 {
            wrong.push(format!(
                "reads: {past} lanes past the device's conversion of Metal's"
            ));
        }
        // Writes.
        let write = state_of(device, "fpv_bfloat_textures", "fpv_bfloat_write");
        let n = 16385 + WRITE_SPECIALS.len();
        let height = n.div_ceil(4096) as u64;
        let input: Vec<u8> = values(&WRITE_SPECIALS, 4096 * height as usize)
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input = buffer_with(device, &input).unwrap();
        let width = buffer_with(device, &4096u32.to_le_bytes()).unwrap();
        let t = texture(device, (4096, height), TextureUsage::SHADER_WRITE, shared);
        dispatch(
            device,
            &write,
            &t,
            (None, &[&input, &width]),
            (4096, height),
        );
        let got = words_of(&t, (4096, height));
        let metal = words(&recorded("rgb10a2-srgb-writes", &["output", "words"]));
        let text = |words: &[u32]| {
            let mut text = String::new();
            for (k, w) in words.iter().enumerate() {
                writeln!(text, "write {k} {w:08x}").unwrap();
            }
            text
        };
        assert_eq!(digest(&text(&metal)), WRITES, "the recorded writes");
        covers(&text(&got), got.len(), "writes");
        let written = values(&WRITE_SPECIALS, got.len());
        let (near, past, first) = words_within(&got, &metal, &written);
        moved += near;
        if past > 0 {
            wrong.push(format!(
                "writes: {past} words past the device's encode of Metal's: {first:?}"
            ));
        }
        // Samples.
        let mut s = 0x243f_6a88_85a3_08d3u64;
        let texels: Vec<u8> = (0..256)
            .flat_map(|_| ((xorshift(&mut s) >> 17) as u32).to_le_bytes())
            .collect();
        let t = texture(device, (16, 16), TextureUsage::SHADER_READ, shared);
        t.replace_region(Region::new_2d(0, 0, 16, 16), 0, 0, &texels, 64, 0)
            .unwrap();
        let positions: Vec<(i64, i64)> = (0..4096)
            .map(|_| {
                let s = xorshift(&mut s);
                ((s % 4608) as i64 - 256, ((s >> 20) % 4608) as i64 - 256)
            })
            .collect();
        let coords: Vec<u8> = positions
            .iter()
            .flat_map(|&(n, m)| [n as f32 / 4096.0, m as f32 / 4096.0])
            .flat_map(f32::to_le_bytes)
            .collect();
        let coords = buffer_with(device, &coords).unwrap();
        // Metal's nearest `float` samples, which its `half` ones narrow.
        let mut nearest_floats = Vec::new();
        let note = Degradation::from_precision_note(PrecisionNote::BicubicFilterArithmetic);
        let mut announced = counts(&messages.take_notices());
        for (linear, filter) in [SamplerMinMagFilter::Nearest, SamplerMinMagFilter::Linear]
            .into_iter()
            .enumerate()
        {
            let mut descriptor = SamplerDescriptor::new();
            descriptor.min_filter = filter;
            descriptor.mag_filter = filter;
            let sampler = device.new_sampler_state(&descriptor).unwrap();
            for (half, name) in ["xr_sample_runtime", "xr_sample_runtime_half"]
                .into_iter()
                .enumerate()
            {
                let state = state_of(device, "xr_samples", name);
                let size = if half == 1 { 8 } else { 16 };
                let out = buffer_with(device, &vec![0xab; 4096 * size]).unwrap();
                dispatch(
                    device,
                    &state,
                    &t,
                    (Some(&sampler), &[&coords, &out]),
                    (4096, 1),
                );
                let bytes = read_buffer(&out, 0, 4096 * size as u64).unwrap();
                let case = format!(
                    "rgb10a2-srgb-samples-{}-{}",
                    ["nearest", "linear"][linear],
                    ["f", "h"][half]
                );
                let metal = recorded(&case, &["output", "samples"]);
                let text = |bytes: &[u8]| {
                    let mut text = String::new();
                    for (&(n, m), v) in positions.iter().zip(bytes.chunks_exact(size)) {
                        match half {
                            1 => {
                                let h = |c: usize| u16::from_le_bytes([v[2 * c], v[2 * c + 1]]);
                                writeln!(
                                    text,
                                    "s {linear} h {n} {m} {:04x} {:04x} {:04x} {:04x}",
                                    h(0),
                                    h(1),
                                    h(2),
                                    h(3)
                                )
                            }
                            _ => {
                                let w = words(v);
                                writeln!(
                                    text,
                                    "s {linear} f {n} {m} {:08x} {:08x} {:08x} {:08x}",
                                    w[0], w[1], w[2], w[3]
                                )
                            }
                        }
                        .unwrap();
                    }
                    text
                };
                assert_eq!(
                    digest(&text(&metal)),
                    SAMPLES[linear][half],
                    "{case}: recorded"
                );
                covers(&text(&bytes), bytes.len() / size, "samples");
                // A nearest sample is a texel's read; a linear one filters the device's reads by
                // its own arithmetic (`bicubic-filter-arithmetic`, whose class is unbounded), so it
                // runs and is counted, its values the conformance suite's `rgb10a2-srgb-samples-*`
                // cases'.
                let lanes = |b: &[u8]| -> Vec<f32> {
                    match half {
                        1 => b
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|h| half_value(u16::from_le_bytes(*h)))
                            .collect(),
                        _ => words(b).into_iter().map(f32::from_bits).collect(),
                    }
                };
                let (got, want) = (lanes(&bytes), lanes(&metal));
                if linear == 1 {
                    filtered += got
                        .iter()
                        .zip(&want)
                        .filter(|(g, w)| g.to_bits() != w.to_bits())
                        .count();
                    continue;
                }
                if half == 0 {
                    nearest_floats = want.clone();
                }
                let mut past = 0;
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    let alpha = i % 4 == 3;
                    let held = match half {
                        1 => half_within(FORM, *g, nearest_floats[i], alpha),
                        _ => value_within(FORM, *g, *w, alpha),
                    };
                    match (g.to_bits() == w.to_bits(), held) {
                        (true, _) => {}
                        (false, true) => moved += 1,
                        (false, false) => past += 1,
                    }
                }
                if past > 0 {
                    wrong.push(format!(
                        "samples nearest half {half}: {past} lanes past the device's conversion \
                         of Metal's"
                    ));
                }
            }
            // A nearest sampler's sample is the decoded texel at Metal's position: the filter's
            // note follows the bound sampler's word, so the nearest samples announce none.
            let samples = counts(&messages.take_notices());
            if linear == 0 {
                assert!(
                    !samples.contains_key(&note),
                    "the nearest samples announced bicubic-filter-arithmetic: {samples:?}"
                );
            }
            for (kind, n) in samples {
                *announced.entry(kind).or_insert(0) += n;
            }
        }
        for line in &wrong {
            println!("RGB10A2-SRGB-DIFFERS {}: {line}", device.name());
        }
        println!(
            "RGB10A2-SRGB {}: {moved} values within the device's conversion of Metal's, {filtered} \
             linear sample lanes apart",
            device.name()
        );
        assert!(
            wrong.is_empty(),
            "{} answers differ from Metal's",
            wrong.len()
        );
        assert_counted(&announced, moved, "the kernels' reads, writes and samples");
        assert!(
            filtered == 0 || announced.contains_key(&note),
            "{filtered} linear sample lanes apart, bicubic-filter-arithmetic unannounced"
        );
        let gamma = device
            .private_properties()
            .flag(PrivateFlag::IsRgb10a2GammaSupported);
        assert_eq!(gamma.value(), Some(true), "isRGB10A2GammaSupported");
    });
}

/// The pipeline of the fixture `rt6_samples`'s fill `fragment` into a 586 color attachment
/// `index` (attachment 0 an `RGBA32Float` one where `index` is 1).
fn pipeline(lane: &Lane, fragment: &str, index: usize) -> RenderPipelineState {
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
    if index == 1 {
        state.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
    }
    state.color_attachments[index] = Some(ColorAttachmentDescriptor::new(PF));
    lane.device
        .new_render_pipeline_state(&state, &PipelineOptions::default())
        .unwrap_or_else(|e| panic!("{fragment}: the pipeline: {e}"))
}

/// A private texture's words through a blit into a buffer.
fn blit_words(lane: &Lane, target: &Texture, (width, height): (u64, u64)) -> Vec<u32> {
    let length = 4 * width * height;
    let back = buffer_with(&lane.device, &vec![0xcd; length as usize]).unwrap();
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(width, height, 1),
        &back,
        0,
        4 * width,
        length,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    words(&read_buffer(&back, 0, length).unwrap())
}

/// Clears and a draw's stores of a 586 attachment are Metal's; a loading pass keeps its words.
#[test]
fn passes_clear_store_and_load_rgb10a2_srgb_as_metal_does() {
    let (messages, _cache, lane) = lane_with(|_| {});
    let mut wrong = Vec::new();
    // The stored words within a code of Metal's, the device's encode.
    let mut moved = 0;
    // Clears.
    let colors: Vec<[f64; 4]> = (0..2049 + RENDER_SPECIALS.len())
        .map(|k| match k {
            _ if k < 2049 => {
                let v = k as f64 / 2048.0;
                [v, v * 0.5, v * 0.25, (k % 64) as f64 / 63.0]
            }
            _ => {
                let v = f64::from(RENDER_SPECIALS[k - 2049]);
                [v, v * 0.5, v * 0.25, v]
            }
        })
        .collect();
    for (shape_index, (side, usage)) in [
        (
            1u64,
            TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        ),
        (64, TextureUsage::RENDER_TARGET),
    ]
    .into_iter()
    .enumerate()
    {
        let target = texture(
            &lane.device,
            (side, side),
            usage,
            ResourceOptions::private(),
        );
        let n = colors.len() as u64;
        let out = buffer_with(&lane.device, &vec![0xab; (4 * n) as usize]).unwrap();
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
                    .unwrap()
                    .end_encoding()
                    .unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_texture_to_buffer(
                    &target,
                    0,
                    0,
                    Origin::new(side - 1, side - 1, 0),
                    Size::new(1, 1, 1),
                    &out,
                    4 * i,
                    4,
                    4,
                    BlitOption::None,
                )
                .unwrap();
                blit.end_encoding().unwrap();
            }
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
        }
        let got = words(&read_buffer(&out, 0, 4 * n).unwrap());
        let mut text = String::new();
        for (k, w) in got.iter().enumerate() {
            writeln!(text, "clear {shape_index} {k} {w:08x}").unwrap();
        }
        covers(&text, got.len(), "clears");
        let want = CLEARS[shape_index];
        if digest(&text) != want {
            wrong.push(format!(
                "clears {side}x{side}: {} (Metal's {want})",
                digest(&text)
            ));
        }
    }
    // Stores.
    let n = 16385 + RENDER_SPECIALS.len();
    let height = n.div_ceil(4096) as u64;
    let all = values(&RENDER_SPECIALS, 4096 * height as usize);
    for (half, fragment) in ["rt6_fill_rgba32float", "rt6_fill_rgba16float"]
        .into_iter()
        .enumerate()
    {
        let state = pipeline(&lane, fragment, 0);
        let bytes: Vec<u8> = match half {
            1 => all
                .iter()
                .flatten()
                .flat_map(|&v| half_bits(v).to_le_bytes())
                .collect(),
            _ => all.iter().flatten().flat_map(|v| v.to_le_bytes()).collect(),
        };
        let values = buffer_with(&lane.device, &bytes).unwrap();
        let shape = shape(&lane.device, 4096, 1).unwrap();
        let target = texture(
            &lane.device,
            (4096, height),
            TextureUsage::RENDER_TARGET,
            ResourceOptions::private(),
        );
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
        .unwrap_or_else(|e| panic!("{fragment}: the pass: {e}"));
        let got = blit_words(&lane, &target, (4096, height));
        let kind = if half == 1 { "half" } else { "float" };
        let case = format!("rgb10a2-srgb-stores-{}", ["f", "h"][half]);
        let metal = words(&recorded(&case, &["output", "words"]));
        let text = |words: &[u32]| {
            let mut text = String::new();
            for (k, w) in words.iter().enumerate() {
                writeln!(text, "store {kind} {k} {w:08x}").unwrap();
            }
            text
        };
        assert_eq!(digest(&text(&metal)), STORES[half], "{case}: recorded");
        covers(&text(&got), got.len(), "stores");
        let stored: Vec<[f32; 4]> = match half {
            1 => all
                .iter()
                .map(|v| v.map(|c| half_value(half_bits(c))))
                .collect(),
            _ => all.clone(),
        };
        let (near, past, first) = words_within(&got, &metal, &stored);
        moved += near;
        if past > 0 {
            wrong.push(format!(
                "stores {kind}: {past} words past the device's encode of Metal's: {first:?}"
            ));
        }
    }
    // Loads: the 586 target is attachment 1, the fill's attachment 0.
    let before: Vec<u32> = (0..4096u32)
        .map(|i| {
            let c = i % 1024;
            c | (c * 7 % 1024) << 10 | (c * 13 % 1024) << 20 | (i / 1024) << 30
        })
        .collect();
    let target = texture(
        &lane.device,
        (1024, 4),
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        ResourceOptions::shared(),
    );
    let bytes: Vec<u8> = before.iter().flat_map(|w| w.to_le_bytes()).collect();
    target
        .replace_region(Region::new_2d(0, 0, 1024, 4), 0, 0, &bytes, 4096, 0)
        .unwrap();
    let state = pipeline(&lane, "rt6_fill_rgba32float", 1);
    let fill: Vec<u8> = (0..4096)
        .flat_map(|_| [1.0f32; 4])
        .flat_map(f32::to_le_bytes)
        .collect();
    let fill = buffer_with(&lane.device, &fill).unwrap();
    let shape = shape(&lane.device, 1024, 1).unwrap();
    let drawn = lane
        .device
        .new_texture(&{
            let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, 1024, 4, false);
            d.usage = TextureUsage::RENDER_TARGET;
            d.resource_options = ResourceOptions::private();
            d
        })
        .unwrap();
    let mut descriptor = RenderPassDescriptor::new();
    let other = &mut descriptor.color_attachments[0];
    other.texture = Some(drawn);
    other.load_action = LoadAction::DontCare;
    other.store_action = StoreAction::Store;
    let color = &mut descriptor.color_attachments[1];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    lane.run(&descriptor, |_, e, _| {
        e.set_render_pipeline_state(&state).unwrap();
        e.set_fragment_buffer(Some(&fill), 0, 0).unwrap();
        e.set_fragment_buffer(Some(&shape), 0, 1).unwrap();
        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    })
    .unwrap_or_else(|e| panic!("the load pass: {e}"));
    let after = words_of(&target, (1024, 4));
    let mut text = String::new();
    for (i, (b, a)) in before.iter().zip(&after).enumerate() {
        writeln!(text, "load {i} {b:08x} {a:08x}").unwrap();
    }
    covers(&text, after.len(), "loads");
    println!("RGB10A2-SRGB-LOADS {}: Metal's {LOADS}", digest(&text));
    if after != before {
        let moved = before.iter().zip(&after).filter(|(b, a)| b != a).count();
        wrong.push(format!(
            "loads: {moved} words moved, where the device's store keeps them"
        ));
    }
    for line in &wrong {
        println!("RGB10A2-SRGB-DIFFERS {}: {line}", lane.device.name());
    }
    println!(
        "RGB10A2-SRGB {}: {moved} stored words within a code of Metal's",
        lane.device.name()
    );
    assert!(
        wrong.is_empty(),
        "{} answers differ from Metal's",
        wrong.len()
    );
    assert_announced(&messages, moved, "the stores");
    quiet(lane, &messages);
}

/// The words of `got`, writes of `values`, that are not `want`'s but hold within the device's
/// encode's bound of each ([`written_within`]), and those past it, with the first few of those.
fn words_within(got: &[u32], want: &[u32], values: &[[f32; 4]]) -> (usize, usize, Vec<String>) {
    let (mut near, mut past, mut first) = (0, 0, Vec::new());
    for (k, ((g, w), v)) in got.iter().zip(want).zip(values).enumerate() {
        match (
            g == w,
            written_within(FORM, &g.to_le_bytes(), &w.to_le_bytes(), v),
        ) {
            (true, _) => {}
            (false, true) => near += 1,
            (false, false) => {
                past += 1;
                if first.len() < 4 {
                    first.push(format!("{k} of {v:?}: {g:08x}, Metal's {w:08x}"));
                }
            }
        }
    }
    (near, past, first)
}

/// The lanes (`float` bits, four a texel) of `got` that are not `want`'s but hold by `held` (the
/// device's, Metal's and the lane's channel), and those that do not.
fn lanes_within(
    got: &[u32],
    want: &[u32],
    held: impl Fn(u32, u32, usize) -> bool,
) -> (usize, usize) {
    let (mut near, mut past) = (0, 0);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        match (g == w, held(g, w, i % 4)) {
            (true, _) => {}
            (false, true) => near += 1,
            (false, false) => past += 1,
        }
    }
    (near, past)
}

/// The lane's `half` conversion is MSL's round to nearest even.
#[test]
fn half_bits_round_to_nearest_even() {
    assert_eq!(half_bits(1.0), 0x3c00);
    assert_eq!(half_bits(-2.0), 0xc000);
    assert_eq!(half_bits(65504.0), 0x7bff);
    assert_eq!(half_bits(65520.0), 0x7c00);
    assert_eq!(half_bits(f32::NEG_INFINITY), 0xfc00);
    assert_eq!(half_bits(1.0 + 1.0 / 2048.0), 0x3c00);
    assert_eq!(half_bits(1.0 + 3.0 / 2048.0), 0x3c02);
    assert_eq!(half_bits(2f32.powi(-24)), 0x0001);
    assert_eq!(half_bits(2f32.powi(-25)), 0x0000);
    assert_eq!(half_bits(3.0 * 2f32.powi(-25)), 0x0002);
    assert_eq!(half_bits(0.333_333_34), 0x3555);
    assert_eq!(half_bits(1e-30), 0);
}
