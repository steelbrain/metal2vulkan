//! A fetch of a blended 16-bit float attachment on the device the cfg selects, under the validation
//! layer: the owned fixture `fbl_fetch_blend16`, in the shape of a macOS 27 window-server
//! composite. Color 0 is a `SIDE`² private `RGBA16Float` target blended one over one minus the
//! source alpha for color and alpha, color 1 an `RGBA16Float` target of mask none. Each function
//! reads color 0 under fast math and returns, at pixel `i = SIDE · y + x`, a loaded source chosen
//! by the fetched alpha (`src[i + SIDE²]` where it exceeds 2, else `src[i]`): `fbl_over_loaded` a
//! `half4`, `fbl_over_loaded_float` a `float4`. One draw of the fullscreen triangle. The second
//! test draws the same functions under each blend constant's difference instead (`BlendColor` with
//! `OneMinusBlendColor`, `BlendAlpha` with `OneMinusBlendAlpha`, either as the source factor,
//! under `Subtract` and `ReverseSubtract`, for color and alpha) and the blend color
//! [`DIFFERENCE_COLOR`].
//!
//! Metal's answer (plain Metal on the M3 Max under macOS 27.0.1, our own MSL of this
//! shape, every channel of 16,384 random pixels of each of these blends, the differences under this
//! blend color): one product fused into the sum, the other rounded first, in the output's
//! arithmetic. For a `half4` output each step is rounded to `half`, nearest even, the fused
//! multiply-add once; for a `float4` output to `float`, the result then converted to `half` toward
//! zero, as Metal's unblended write of a `float` converts it. One over one minus the source alpha
//! is `h(s + d · h(1 - a))` (`s` the output's channel, `a` its alpha, `d` the texel's channel), and
//! `fma(d, f(1 - a), s)` in `float`. A difference fuses its minuend's product and rounds its
//! subtrahend's: `h(s · c - h(d · m))` under `Subtract` with `BlendColor` the source factor, its
//! `c` the blend color's channel rounded to `half` and `m` its one minus `h(1 - c)` (alpha's for
//! alpha and for `BlendAlpha`), and in `float` `c` and `f(1 - c)`. The texels are random `half`s of
//! magnitude below 8, alpha below 4 (so both loads are taken), the sources random `half`s or
//! `float`s of the same ranges, each a signed zero at times: every `half` multiply-add is exact in
//! `f64`, and `f32::mul_add` rounds the `float` one once.
//!
//! Where the draw announces no `half-output-fast-math-blend`, `half-output-precision` or
//! `device-color-conversion`, every channel is Metal's, a channel Metal stores as a NaN any NaN
//! where it announces `device-nan-encoding` (the storage image's store of the blend's NaN, the
//! device's where not measured to be Metal's: ANV stores `0xfe00`). On a device that fetches
//! through the pixel interlock (RADV, ANV, NVIDIA, MoltenVK) the function computes Metal's blend
//! itself, exactly, announcing none of the three, which is asserted. Through input attachments
//! (lavapipe) the device blends; what it announces and how many channels are Metal's is printed.
//! Each draw prints a `FETCH-BLEND16` line (`FETCH-BLEND16-DIFFERENCE` under a difference) naming
//! its function and blend. A device that has neither form (RADV on the RX 550) skips them
//! (`fetches`).

mod nans;
mod narrow;
mod zero;

use super::{fetches, form, library, through_inputs};
use crate::runtime_lane::buffer_with;
use crate::runtime_lane::degrade_notices::counts;
use crate::runtime_lane::resolves::{lane_with, quiet, Lane};
use crate::runtime_lane::WAIT;
use metal2vulkan::metal::{
    BlendFactor, BlendOperation, ColorWriteMask, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, PrecisionNote, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, Degradation, LoadAction, Origin, PipelineOptions, RenderPassDescriptor,
    ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
};

/// The target's side, as the fixture's.
const SIDE: u64 = 64;

/// The target's texels.
const TEXELS: usize = (SIDE * SIDE) as usize;

/// The notes under which a channel may be another than Metal's.
const NOTES: [PrecisionNote; 3] = [
    PrecisionNote::HalfOutputFastMathBlend,
    PrecisionNote::HalfOutputPrecision,
    PrecisionNote::DeviceColorConversion,
];

/// A deterministic stream of 64-bit words (xorshift).
struct Words(u64);

impl Words {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A random `half` of magnitude below `2^(top + 1)` and at least `2^-8`, or a signed zero one
    /// time in 32.
    fn half(&mut self, top: i32) -> f64 {
        let w = self.next();
        let sign = if w & 1 == 1 { -1.0 } else { 1.0 };
        if (w >> 1) & 31 == 0 {
            return 0.0 * sign;
        }
        let exponent = -8 + ((w >> 6) % (top as u64 + 9)) as i32;
        let mantissa = ((w >> 16) & 0x3ff) as f64;
        sign * (1.0 + mantissa / 1024.0) * 2f64.powi(exponent)
    }

    /// A random `float` of magnitude below `2^(top + 1)` and at least `2^-8`, or a signed zero one
    /// time in 32.
    fn float(&mut self, top: i32) -> f32 {
        let w = self.next();
        let sign = if w & 1 == 1 { -1.0 } else { 1.0 };
        if (w >> 1) & 31 == 0 {
            return 0.0 * sign;
        }
        let exponent = -8 + ((w >> 6) % (top as u64 + 9)) as i32;
        let mantissa = ((w >> 16) & 0x7f_ffff) as f32;
        sign * (1.0 + mantissa / 8_388_608.0) * 2f32.powi(exponent)
    }
}

/// The bits of `x` as a `half`: rounded to nearest even, or toward zero where `toward_zero` (the
/// largest finite `half` past it, as Metal's write of a `float` converts).
fn half_bits(x: f64, toward_zero: bool) -> u16 {
    let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    if x.is_nan() {
        return 0x7e00;
    }
    if a == 0.0 {
        return sign;
    }
    if a >= 65536.0 {
        return sign
            | if toward_zero && a.is_finite() {
                0x7bff
            } else {
                0x7c00
            };
    }
    let binade = |v: f64| ((v.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let q = 2f64.powi(binade(a).max(-14) - 10);
    let n = a / q;
    let v = if toward_zero {
        n.trunc()
    } else {
        n.round_ties_even()
    } * q;
    if v > 65504.0 {
        return sign | 0x7c00;
    }
    if v == 0.0 {
        return sign;
    }
    let e = binade(v);
    if e < -14 {
        return sign | (v / 2f64.powi(-24)) as u16;
    }
    sign | (((e + 15) as u16) << 10) | ((v / 2f64.powi(e - 10)) as u16 - 1024)
}

/// The value of `half` bits `h`.
fn half_value(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1f);
    let m = f64::from(h & 0x3ff);
    match e {
        0 => sign * m * 2f64.powi(-24),
        31 if m == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        _ => sign * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

/// A blend of color 0: color's and alpha's factors and operation (the same for both), and the
/// encoder's blend color.
#[derive(Clone, Copy, Debug)]
struct Blending {
    source: BlendFactor,
    destination: BlendFactor,
    operation: BlendOperation,
    color: [f32; 4],
}

/// The guest's blend: one over one minus the source alpha, added.
const OVER: Blending = Blending {
    source: BlendFactor::One,
    destination: BlendFactor::OneMinusSourceAlpha,
    operation: BlendOperation::Add,
    color: [0.0; 4],
};

/// The blend color of the differences, the probe's: blue's `h(1 - h(c))` and `h(1 - c)` differ.
const DIFFERENCE_COLOR: [f32; 4] = [0.3, 0.6, 0.9, 0.45];

/// Each blend constant's difference, either as the source factor, under each subtraction.
fn differences() -> Vec<Blending> {
    use BlendFactor as F;
    let pairs = [
        (F::BlendColor, F::OneMinusBlendColor),
        (F::OneMinusBlendColor, F::BlendColor),
        (F::BlendAlpha, F::OneMinusBlendAlpha),
        (F::OneMinusBlendAlpha, F::BlendAlpha),
    ];
    let operations = [BlendOperation::Subtract, BlendOperation::ReverseSubtract];
    (pairs.iter())
        .flat_map(|&(source, destination)| {
            operations.map(|operation| Blending {
                source,
                destination,
                operation,
                color: DIFFERENCE_COLOR,
            })
        })
        .collect()
}

/// A case: the function, and whether its output is a `float4`.
#[derive(Clone, Copy, Debug)]
struct Case {
    entry: &'static str,
    float: bool,
}

/// The fixture a sweep draws: its vertex function, its two cases, their data
/// ([`data`]'s shape) and Metal's answer ([`metal`]'s).
struct Sweep {
    fixture: &'static str,
    vertex: &'static str,
    cases: [Case; 2],
    data: fn(Case) -> Data,
    metal: fn(Case, Blending, [f64; 4], [f64; 4]) -> [u16; 4],
}

/// The fast-math fixture's sweep.
const FAST: Sweep = Sweep {
    fixture: "fbl_fetch_blend16",
    vertex: "fbl_vertex",
    cases: [
        Case {
            entry: "fbl_over_loaded",
            float: false,
        },
        Case {
            entry: "fbl_over_loaded_float",
            float: true,
        },
    ],
    data,
    metal,
};

/// A sweep's data: the texels' `half` bits, four a texel, and the sources (two a pixel), as
/// `f64`s.
type Data = (Vec<u16>, Vec<[f64; 4]>);

/// The texels and sources: random `half`s, and `half`s or `float`s (the module docs).
fn data(case: Case) -> Data {
    let mut words = Words(0x9e37_79b9_7f4a_7c15);
    let texels: Vec<u16> = (0..TEXELS * 4)
        .map(|k| {
            let top = if k % 4 == 3 { 1 } else { 2 };
            half_bits(words.half(top), false)
        })
        .collect();
    let sources = (0..TEXELS * 2)
        .map(|_| {
            let top = [2, 2, 2, 1];
            top.map(|t| match case.float {
                true => f64::from(words.float(t)),
                false => words.half(t),
            })
        })
        .collect();
    (texels, sources)
}

/// `x` rounded to `half`, nearest even.
fn h(x: f64) -> f64 {
    half_value(half_bits(x, false))
}

/// Metal's blend `b` of `s` over the texel `d`, as `half` bits: the destination's
/// product fused into the sum under `Add` and `ReverseSubtract`, the source's under `Subtract`,
/// the other product rounded first (a factor of one leaves its operand as it is).
fn metal(case: Case, b: Blending, s: [f64; 4], d: [f64; 4]) -> [u16; 4] {
    use BlendFactor as F;
    std::array::from_fn(|c| {
        let factor = |f: BlendFactor| {
            let alpha = c == 3 || matches!(f, F::BlendAlpha | F::OneMinusBlendAlpha);
            let k = f64::from(b.color[if alpha { 3 } else { c }]);
            match (f, case.float) {
                (F::One, _) => 1.0,
                (F::OneMinusSourceAlpha, false) => h(1.0 - s[3]),
                (F::OneMinusSourceAlpha, true) => f64::from(1.0f32 - s[3] as f32),
                (F::BlendColor | F::BlendAlpha, false) => h(k),
                (F::BlendColor | F::BlendAlpha, true) => k,
                (F::OneMinusBlendColor | F::OneMinusBlendAlpha, false) => h(1.0 - h(k)),
                (F::OneMinusBlendColor | F::OneMinusBlendAlpha, true) => {
                    f64::from(1.0f32 - k as f32)
                }
                _ => panic!("no factor {f:?} here"),
            }
        };
        let (ks, kd) = (factor(b.source), factor(b.destination));
        // The fused product's operand and factor, the rounded one's, and the rounded one's sign.
        let ((x, kx), (y, ky), sign) = match b.operation {
            BlendOperation::Subtract => ((s[c], ks), (d[c], kd), -1.0),
            BlendOperation::ReverseSubtract => ((d[c], kd), (s[c], ks), -1.0),
            _ => ((d[c], kd), (s[c], ks), 1.0),
        };
        match case.float {
            true => {
                let rounded = sign as f32 * (y as f32 * ky as f32);
                half_bits(f64::from((x as f32).mul_add(kx as f32, rounded)), true)
            }
            false => half_bits(x.mul_add(kx, sign * h(y * ky)), false),
        }
    })
}

/// A private `SIDE`² `RGBA16Float` texture holding `texels`.
fn filled(lane: &Lane, texels: &[u16]) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let texture = lane.device.new_texture(&descriptor).unwrap();
    let bytes: Vec<u8> = texels.iter().flat_map(|h| h.to_le_bytes()).collect();
    let source = buffer_with(&lane.device, &bytes).unwrap();
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        8 * SIDE,
        bytes.len() as u64,
        Size::new(SIDE, SIDE, 1),
        &texture,
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
    texture
}

/// Each draw's channels are Metal's where it announces none of [`NOTES`], and on a device that
/// fetches through the pixel interlock it announces none (the module docs).
#[test]
fn a_blended_sixteen_bit_float_fetch_stores_metals_blend() {
    stores_metals_blend(&FAST, &[OVER], "FETCH-BLEND16");
}

/// Each blend constant's difference: each draw's channels are Metal's where it announces none of
/// [`NOTES`], and on a device that fetches through the pixel interlock it announces none (the
/// module docs). Before the shader computed a difference in Metal's form, the color channels'
/// blend color difference took a rounded near-equivalent and every difference announced
/// `half-output-fast-math-blend` there.
#[test]
fn a_fetched_blend_constant_difference_stores_metals_blend() {
    stores_metals_blend(&FAST, &differences(), "FETCH-BLEND16-DIFFERENCE");
}

/// Each function of `sweep` drawn once under each of `blends`, each draw's line tagged `tag`.
fn stores_metals_blend(sweep: &Sweep, blends: &[Blending], tag: &str) {
    let kinds = NOTES.map(Degradation::from_precision_note);
    let device_nans = Degradation::from_precision_note(PrecisionNote::DeviceNanEncoding);
    let (messages, _cache, lane) = lane_with(|options| {
        options.announce_each = kinds.into_iter().chain([device_nans]).collect();
    });
    if !fetches(&lane.device) {
        return;
    }
    let fixture = library(&lane, sweep.fixture);
    let device = lane.device.name();
    let interlock = !through_inputs(&lane);
    let function = |name| {
        StageSource::Function(
            fixture
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    for (b, case) in blends
        .iter()
        .flat_map(|&b| sweep.cases.map(|case| (b, case)))
    {
        let what = format!("{} under {b:?} on {device}", case.entry);
        let mut descriptor = RenderPipelineDescriptor::new(function(sweep.vertex));
        descriptor.fragment = Some(function(case.entry));
        let mut blended = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
        blended.blending_enabled = true;
        blended.source_rgb_blend_factor = b.source;
        blended.destination_rgb_blend_factor = b.destination;
        blended.rgb_blend_operation = b.operation;
        blended.source_alpha_blend_factor = b.source;
        blended.destination_alpha_blend_factor = b.destination;
        blended.alpha_blend_operation = b.operation;
        let mut unwritten = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
        unwritten.write_mask = ColorWriteMask::NONE;
        descriptor.color_attachments[0] = Some(blended);
        descriptor.color_attachments[1] = Some(unwritten);
        let state = lane
            .device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap_or_else(|error| panic!("{what}: {error}"));
        let (texels, sources) = (sweep.data)(case);
        let target = filled(&lane, &texels);
        let mut second = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, SIDE, SIDE, false);
        second.usage = TextureUsage::RENDER_TARGET;
        second.resource_options = ResourceOptions::private();
        let second = lane.device.new_texture(&second).unwrap();
        let bytes: Vec<u8> = match case.float {
            true => sources
                .iter()
                .flat_map(|s| s.map(|v| v as f32))
                .flat_map(f32::to_le_bytes)
                .collect(),
            false => sources
                .iter()
                .flat_map(|s| s.map(|v| half_bits(v, false)))
                .flat_map(u16::to_le_bytes)
                .collect(),
        };
        let src = buffer_with(&lane.device, &bytes).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Load;
        color.store_action = StoreAction::Store;
        let color = &mut pass.color_attachments[1];
        color.texture = Some(second);
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        color.store_action = StoreAction::DontCare;
        let _ = messages.take_notices();
        lane.run(&pass, |_, encoder, _| {
            encoder.set_render_pipeline_state(&state).unwrap();
            let [red, green, blue, alpha] = b.color;
            encoder.set_blend_color(red, green, blue, alpha);
            encoder.set_fragment_buffer(Some(&src), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        })
        .unwrap();
        let announced = counts(&messages.take_notices());
        let noted: Vec<_> = kinds
            .iter()
            .filter(|k| announced.contains_key(*k))
            .collect();
        // Where the draw announces `device-nan-encoding`, a channel Metal stores as a NaN holds
        // the device's NaN.
        let any_nan = announced.contains_key(&device_nans);
        let nan = |h: u16| h & 0x7fff > 0x7c00;
        let alike = |got: [u16; 4], want: [u16; 4]| {
            (got.iter().zip(want)).all(|(&g, w)| g == w || (any_nan && nan(g) && nan(w)))
        };
        let read = lane.read(&target, (0, 0), 8, BlitOption::None);
        let got: Vec<u16> = read
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect();
        let mut wrong = Vec::new();
        for i in 0..TEXELS {
            let texel: [u16; 4] = got[4 * i..4 * i + 4].try_into().unwrap();
            let d: [f64; 4] = std::array::from_fn(|c| half_value(texels[4 * i + c]));
            let s = sources[if d[3] > 2.0 { i + TEXELS } else { i }];
            let want = (sweep.metal)(case, b, s, d);
            if !alike(texel, want) {
                wrong.push((i, texel, want));
            }
        }
        println!(
            "{tag} {} {:?} {:?} {:?} | {device} | through {} | Metal's {}/{} texels | announced \
             {noted:?}, device-nan-encoding {any_nan}",
            case.entry,
            b.source,
            b.destination,
            b.operation,
            form(&lane),
            TEXELS - wrong.len(),
            TEXELS,
        );
        if interlock {
            assert!(noted.is_empty(), "{what}: announced {noted:?}");
        }
        if noted.is_empty() {
            let shown = &wrong[..wrong.len().min(8)];
            assert!(
                wrong.is_empty(),
                "{what}: {} of {TEXELS} texels differ from Metal's, first (index, read, Metal's) \
                 {shown:x?}",
                wrong.len()
            );
        }
    }
    let _ = messages.take_notices();
    quiet(lane, &messages);
}
