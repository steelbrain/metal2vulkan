//! Metal's extended-range conversions run through the interpreter: every code read, and the
//! writes around every rounding boundary, against the exact rule.

use super::blend::{extended_ranges, native, NativeBlend};
use super::{
    declare, declare_padded, decode, encode, tile_read, tile_value, SOURCE_RANGE, SRGB_SOURCE_RANGE,
};
use crate::error::Error;
use crate::lower::arith::{FloatFmt, Form, Held};
use crate::lower::stage_io::extended_color::bfloat_output;
use crate::metal::format::extended_srgb::{SRGB_DECODE_STEPS, SRGB_ENCODE_FIRST};
use crate::metal::format::padded_srgb::{
    R10X6_SRGB_DECODE_STEPS, R10X6_SRGB_ENCODE_FIRST, RG10X12_SRGB_WRITE_CODES,
};
use crate::metal::format::ExtendedRange;
use crate::metal::{BlendFactor, BlendOperation, ColorWriteMask, TextureWriteRoundingMode};
use crate::pipeline::lower_inputs::{BlendEquation, ExtendedRangeColor};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::interp::{self, Buffer, Config, Resources};
use crate::spirv::ir::{
    natural_layout, BinOp, ConvOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass,
    Type, TypedModule,
};

const FORMS: [ExtendedRange; 4] = [
    ExtendedRange::Bgr10 { srgb: false },
    ExtendedRange::Bgr10 { srgb: true },
    ExtendedRange::Bgra10 { srgb: false },
    ExtendedRange::Bgra10 { srgb: true },
];

/// [`FORMS`] and the two-plane forms.
pub(crate) const EVERY_FORM: [ExtendedRange; 6] = [
    ExtendedRange::Bgr10 { srgb: false },
    ExtendedRange::Bgr10 { srgb: true },
    ExtendedRange::Bgra10 { srgb: false },
    ExtendedRange::Bgra10 { srgb: true },
    ExtendedRange::Planes { srgb: false },
    ExtendedRange::Planes { srgb: true },
];

/// What a test kernel computes of each invocation's words.
#[derive(Clone, Copy)]
enum Op {
    /// The decode of a raw texel.
    Decode,
    /// The decode of a raw texel to `half` values (as `f32` bits).
    DecodeHalf,
    /// The encode of a `vec4` of `f32` bits.
    Encode,
    /// The tile value of a raw texel.
    Tile,
    /// The blend under an equation of a write of the channels of a mask (`blend::native`) of a
    /// source (words 4 to 7, `f32` bits) over the tile value of a raw texel (words 0 to 3), the
    /// blend color words 8 to 11 (`f32` bits, as the encoder set it), the second source words 12
    /// to 15 (`f32` bits).
    Blend(BlendEquation, ColorWriteMask),
    /// Metal's read of a tile holding a `vec4` of `f32` bits ([`tile_read`]).
    TileRead,
    /// A render pass's store of a fragment function's output (`vec4` of `f32` bits, of the
    /// format, `half` values held as `f32`) to a bfloat attachment of a pipeline at
    /// `textureWriteRoundingMode` `NearestEven` where the flag is set
    /// (`stage_io::extended_color::bfloat_output`).
    Output(bool, FloatFmt),
}

impl Op {
    /// The words an invocation reads.
    fn words(self) -> u32 {
        match self {
            Op::Blend(..) => 16,
            _ => 4,
        }
    }
}

fn slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

fn storage_buffer(b: &mut ModuleBuilder, index: u32, elem: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(slot(index), StorageClass::StorageBuffer, block)
}

/// Word `k` of invocation `i`'s `n` words in `buf`.
fn word(
    f: &mut FunctionBuilder<'_>,
    buf: GlobalId,
    (i, n): (Value, u32),
    k: u32,
) -> BResult<Value> {
    let n = f.u32_const(n)?;
    let n = f.module().const_value(n)?;
    let at = f.binary(BinOp::IMul, i, n)?;
    let k = f.u32_const(k)?;
    let k = f.module().const_value(k)?;
    let at = f.binary(BinOp::IAdd, at, k)?;
    let zero = f.u32_const(0)?;
    let zero = f.module().const_value(zero)?;
    let base = f.module().global_value(buf)?;
    f.access_chain(base, &[zero, at])
}

/// A kernel whose invocation `i` reads `op`'s words from buffer 0 and writes four to buffer 1.
fn kernel(form: ExtendedRange, op: Op) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 })?;
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
    let input = storage_buffer(&mut b, 0, u32_)?;
    let output = storage_buffer(&mut b, 1, u32_)?;
    let table = match form.srgb() {
        true => Some(declare(&mut b)?),
        false => declare_padded(&mut b, form)?,
    };
    let id = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let gid = f.module().global_value(id)?;
    let gid = f.load(gid, MemAccess::default())?;
    let i = f.composite_extract(gid, &[0])?;
    let mut lanes = Vec::with_capacity(8);
    for k in 0..op.words() {
        let at = word(&mut f, input, (i, op.words()), k)?;
        lanes.push(f.load(at, MemAccess::default())?);
    }
    let x = f.composite_construct(uvec4, &lanes[..4])?;
    let out = match op {
        Op::Encode => {
            let v = f.convert(ConvOp::Bitcast, x, vec4)?;
            encode(&mut f, form, v, table)?
        }
        Op::Decode => {
            let v = decode(&mut f, form, x, table, false)?;
            f.convert(ConvOp::Bitcast, v, uvec4)?
        }
        Op::DecodeHalf => {
            let v = decode(&mut f, form, x, table, true)?;
            f.convert(ConvOp::Bitcast, v, uvec4)?
        }
        Op::Tile => {
            let v = tile_value(&mut f, form, x, table)?;
            f.convert(ConvOp::Bitcast, v, uvec4)?
        }
        Op::TileRead => {
            let held = f.convert(ConvOp::Bitcast, x, vec4)?;
            let v = tile_read(&mut f, form, held)?;
            f.convert(ConvOp::Bitcast, v, uvec4)?
        }
        Op::Output(nearest_even, fmt) => {
            let v = f.convert(ConvOp::Bitcast, x, vec4)?;
            let rounding = match nearest_even {
                true => TextureWriteRoundingMode::NearestEven,
                false => TextureWriteRoundingMode::TowardZero,
            };
            let e = ExtendedRangeColor::new(form, false).with_rounding(rounding);
            let lanes = Form {
                held: Held::Float(fmt),
                lanes: 4,
            };
            bfloat_output(&mut f, e, lanes, v)?
        }
        Op::Blend(eq, mask) => {
            let b = NativeBlend::of(&eq, mask).expect("a blend");
            let s = f.composite_construct(uvec4, &lanes[4..8])?;
            let s = f.convert(ConvOp::Bitcast, s, vec4)?;
            let c = f.composite_construct(uvec4, &lanes[8..12])?;
            let s1 = f.composite_construct(uvec4, &lanes[12..16])?;
            let s1 = f.convert(ConvOp::Bitcast, s1, vec4)?;
            let d = tile_value(&mut f, form, x, table)?;
            let ranges = Some(extended_ranges(form));
            let v = native(&mut f, &b, ranges, (s, Some(s1), d), Some(c))?;
            f.convert(ConvOp::Bitcast, v, uvec4)?
        }
    };
    for k in 0..4u32 {
        let c = f.composite_extract(out, &[k])?;
        let at = word(&mut f, output, (i, 4), k)?;
        f.store(at, c, MemAccess::default())?;
    }
    f.ret(None)?;
    let main = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

fn run<const N: usize>(form: ExtendedRange, op: Op, lanes: &[[u32; N]]) -> Vec<[u32; 4]> {
    assert_eq!(N as u32, op.words());
    let m = kernel(form, op).expect("kernel");
    let bytes: Vec<u8> = lanes
        .iter()
        .flatten()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(slot(0)),
                address: None,
                bytes,
            },
            Buffer {
                slot: Some(slot(1)),
                address: None,
                bytes: vec![0xAB; lanes.len() * 16],
            },
        ],
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [u32::try_from(lanes.len()).expect("lanes"), 1, 1],
        ..Config::default()
    };
    let (out, _) = interp::run_with_images(&m, &cfg, res, Vec::new()).expect("run");
    out.buffers[1]
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| {
            std::array::from_fn(|k| {
                u32::from_le_bytes(c[4 * k..4 * k + 4].try_into().expect("word"))
            })
        })
        .collect()
}

/// `n / 510` correctly rounded to `f32` (`|n| <= 639`): the `f64` quotient's neighbours compared
/// exactly (each `f32` times 510 is exact in `f64`).
fn ratio(n: i32) -> f32 {
    let q = f64::from(n) / 510.0;
    let y = q as f32;
    let candidates = [
        y,
        f32::from_bits(y.to_bits().wrapping_sub(1)),
        f32::from_bits(y.to_bits() + 1),
    ];
    let err = |c: f32| (f64::from(c) * 510.0 - f64::from(n)).abs();
    let best = candidates
        .into_iter()
        .filter(|c| c.is_finite())
        .min_by(|a, b| err(*a).partial_cmp(&err(*b)).expect("finite"))
        .expect("a candidate");
    if n == 0 {
        0.0
    } else {
        best
    }
}

/// Metal's value of code `c` of a color channel.
fn color_value(c: u32, srgb: bool) -> f32 {
    let n = c as i32 - 384;
    match srgb {
        false => ratio(n),
        true => {
            let s = f32::from(SRGB_DECODE_STEPS[n.unsigned_abs() as usize]) / 8192.0;
            if n < 0 {
                -s
            } else {
                s
            }
        }
    }
}

/// The raw texel of red, green, blue, and alpha codes `rgba` in `form` (a two-plane form's alpha
/// its byte), with `junk` in the bits Metal does not read.
pub(crate) fn raw(form: ExtendedRange, rgba: [u32; 4], junk: u32) -> [u32; 4] {
    match form {
        ExtendedRange::Bgr10 { .. } => [
            rgba[2] | rgba[1] << 10 | rgba[0] << 20 | (junk & 3) << 30,
            0,
            0,
            0,
        ],
        ExtendedRange::Planes { .. } => [
            rgba[2] | rgba[1] << 10 | rgba[0] << 20 | (junk & 3) << 30,
            (rgba[3] & 0xFF) | (junk & 0xFF) << 8,
            0,
            0,
        ],
        ExtendedRange::R10X6 | ExtendedRange::R10X6Srgb => [rgba[0] << 6 | (junk & 0x3F), 0, 0, 0],
        ExtendedRange::Rg10X12 | ExtendedRange::Rg10X12Srgb => [
            rgba[0] << 6 | (junk & 0x3F),
            rgba[1] << 6 | (junk >> 6 & 0x3F),
            0,
            0,
        ],
        _ => {
            let f = |c: u32| c << 6 | (junk & 0x3F);
            [f(rgba[2]), f(rgba[1]), f(rgba[0]), f(rgba[3])]
        }
    }
}

/// Every code of every channel reads Metal's value, bit for bit, whatever the bits Metal does not
/// read hold: `(c - 384) / 510` correctly rounded, an `_sRGB` color channel its table's step over
/// 8192, alpha clamped to `[0, 1]` (1 where the format has none).
#[test]
fn every_code_reads_metals_value() {
    for form in FORMS {
        let srgb = form.srgb();
        let lanes: Vec<[u32; 4]> = (0..1024u32)
            .map(|c| raw(form, [c, 1023 - c, (c * 7) % 1024, c], c * 13))
            .collect();
        let got = run(form, Op::Decode, &lanes);
        for (c, g) in (0..1024u32).zip(&got) {
            let want = [
                color_value(c, srgb),
                color_value(1023 - c, srgb),
                color_value((c * 7) % 1024, srgb),
                match form {
                    ExtendedRange::Bgr10 { .. } => 1.0,
                    _ => ratio(c.clamp(384, 894) as i32 - 384),
                },
            ];
            assert_eq!(*g, want.map(f32::to_bits), "{form:?} code {c}");
        }
    }
}

/// `x` rounded to the nearest integer, ties to even (`x` exact in `f64`).
fn round_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// The code Metal writes for color value `v` (alpha where `alpha`).
fn code(v: f32, srgb: bool, alpha: bool) -> u32 {
    let v = if v.is_nan() { 0.0 } else { v };
    let v = if alpha { v.clamp(0.0, 1.0) } else { v };
    let neg = v < 0.0;
    if srgb && !alpha {
        let q = round_even(f64::from(v.abs().min(2.0)) * 8192.0) as u32;
        let n = (0..640)
            .rev()
            .find(|&n| u32::from(SRGB_ENCODE_FIRST[n]) <= q)
            .expect("offset 0") as u32;
        return if neg { 384 - n.min(384) } else { 384 + n };
    }
    let k = round_even(f64::from(v.abs().min(1.5)) * 510.0) as i32;
    (384 + if neg { -k } else { k }).clamp(0, 1023) as u32
}

/// A write stores Metal's code around every boundary: each code's value and its neighbours, the
/// `f32`s either side of each midpoint (each an exact tie is not, the exact product rounding to
/// nearest even), each `_sRGB` index's half-point and its neighbours, and the specials; bits Metal
/// does not read are written as it writes them.
#[test]
fn writes_round_at_metals_boundaries() {
    let specials = [
        0.0f32,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1e30,
        -1e30,
        -0.75,
        0.25,
        1.25,
        1.5,
        -1.5,
        2.0,
        -2.0,
        1e-40,
    ];
    for form in FORMS {
        let srgb = form.srgb();
        let mut xs: Vec<f32> = specials.to_vec();
        for c in 0..1023u32 {
            let (a, b) = (color_value(c, srgb), color_value(c + 1, srgb));
            let mid = ((f64::from(a) + f64::from(b)) / 2.0) as f32;
            xs.extend([
                a,
                mid,
                f32::from_bits(mid.to_bits() + 1),
                f32::from_bits(mid.to_bits().wrapping_sub(1)),
            ]);
        }
        if srgb {
            for first in SRGB_ENCODE_FIRST.iter().skip(1) {
                let h = (f32::from(*first) - 0.5) / 8192.0;
                for x in [h, -h] {
                    xs.extend([
                        x,
                        f32::from_bits(x.to_bits() + 1),
                        f32::from_bits(x.to_bits() - 1),
                    ]);
                }
            }
        }
        let lanes: Vec<[u32; 4]> = xs
            .iter()
            .map(|x| [x.to_bits(), (-x).to_bits(), x.to_bits(), x.to_bits()])
            .collect();
        let got = run(form, Op::Encode, &lanes);
        for (x, g) in xs.iter().zip(&got) {
            let rgba = [
                code(*x, srgb, false),
                code(-x, srgb, false),
                code(*x, srgb, false),
                code(*x, srgb, true),
            ];
            let want = match form {
                ExtendedRange::Bgr10 { .. } => {
                    [rgba[2] | rgba[1] << 10 | rgba[0] << 20 | 3 << 30, 0, 0, 0]
                }
                _ => raw(form, rgba, 0),
            };
            assert_eq!(*g, want, "{form:?} {x:e} ({:#010x})", x.to_bits());
        }
    }
}

/// Metal's own answers on an Apple M3 Max and an M4: writes at its measured ties and
/// thresholds, and reads of its measured values.
#[test]
fn metals_measured_answers_hold() {
    // Linear: -0.75, 0.25 and 1.25 are exact ties, written as codes 2, 512 and 1022; code 0
    // reads -384/510 (0xbf40c0c1), code 512 0.25098 (0x3e808081).
    let linear = ExtendedRange::Bgra10 { srgb: false };
    let w = run(
        linear,
        Op::Encode,
        &[[
            (-0.75f32).to_bits(),
            0.25f32.to_bits(),
            1.25f32.to_bits(),
            0,
        ]],
    );
    assert_eq!(w[0], [1022 << 6, 512 << 6, 2 << 6, 384 << 6]);
    let r = run(linear, Op::Decode, &[raw(linear, [0, 512, 1023, 1023], 0)]);
    assert_eq!(r[0], [0xbf40_c0c1, 0x3e80_8081, 0x3fa0_6060, 0x3f80_0000]);
    // sRGB: code 0 reads -0.527099609375 (0xbf06f000), code 1023 1.6749267578125 (0x3fd66400);
    // 0.56634521484375 is the least float writing code 781, -0.13214111328125 the least writing
    // code 181 (bisection thresholds).
    let srgb = ExtendedRange::Bgr10 { srgb: true };
    let r = run(srgb, Op::Decode, &[raw(srgb, [0, 1023, 384, 0], 0)]);
    assert_eq!(r[0][..3], [0xbf06_f000, 0x3fd6_6400, 0]);
    let below = |x: f32| f32::from_bits(x.to_bits() - 1);
    let w = run(
        srgb,
        Op::Encode,
        &[[
            0.566_345_2f32.to_bits(),
            below(0.566_345_2).to_bits(),
            (-0.132_141_11f32).to_bits(),
            0,
        ]],
    );
    let codes = [
        (w[0][0] >> 20) & 0x3FF,
        (w[0][0] >> 10) & 0x3FF,
        w[0][0] & 0x3FF,
    ];
    assert_eq!(codes, [781, 780, 181]);
}

/// `x` (exact in `f64`) rounded to `half`: to nearest, ties away from zero where `away`, else to
/// even; held as an `f32`. No value here reaches the infinities.
fn to_half(x: f64, away: bool) -> f32 {
    if x == 0.0 {
        return x as f32;
    }
    let e = ((x.abs().to_bits() >> 52) & 0x7FF) as i32 - 1023;
    // A `half`'s spacing at `x`: `2^(e - 10)`, at least the denormals' `2^-24`.
    let ulp = 2f64.powi(e.max(-14) - 10);
    let q = x / ulp;
    let r = if away { q.round() } else { q.round_ties_even() };
    let y = r * ulp;
    assert!(y.abs() < 65520.0, "{x} rounds past the largest half");
    y as f32
}

/// The `half` value of bits `h`, as an `f32` (a NaN quiet).
fn half(h: u16) -> f32 {
    crate::float16::f16_bits_to_f32(h)
}

/// Metal's tile value of a texel of `form` whose codes are `rgba` (a two-plane form's alpha its
/// byte), as the module docs of `tile_value` state the rule.
pub(crate) fn tile_model(form: ExtendedRange, rgba: [u32; 4]) -> [f32; 4] {
    let srgb = form.srgb();
    let by_code = !srgb || matches!(form, ExtendedRange::Bgra10 { .. });
    let (lo, hi) = ((-384.0f64 / 510.0) as f32, (638.0f64 / 510.0) as f32);
    let color = |c: u32| match by_code {
        true => color_value(c.min(1022), srgb),
        false => color_value(c, srgb).clamp(lo, hi),
    };
    let alpha = match form {
        ExtendedRange::Bgr10 { .. } => 1.0,
        ExtendedRange::Bgra10 { .. } => ratio(rgba[3].clamp(384, 894) as i32 - 384),
        ExtendedRange::Planes { .. } => (f64::from(rgba[3]) / 255.0) as f32,
        ExtendedRange::R10X6
        | ExtendedRange::Rg10X12
        | ExtendedRange::R10X6Srgb
        | ExtendedRange::Rg10X12Srgb
        | ExtendedRange::Rgb10A2Srgb => {
            panic!("{form:?}: a padded form's tile value is not measured")
        }
        ExtendedRange::R16Bfloat | ExtendedRange::Rg16Bfloat | ExtendedRange::Rgba16Bfloat => {
            panic!("{form:?}: a bfloat form's tile value is not measured")
        }
    };
    [color(rgba[0]), color(rgba[1]), color(rgba[2]), alpha].map(|v| to_half(f64::from(v), true))
}

/// Every code of every channel of the six forms reads Metal's tile value, whatever the bits Metal
/// does not read hold: a linear or `BGRA10_XR_sRGB` code above 1022 as 1022, the other `_sRGB`
/// forms' values clamped to the linear values of codes 0 and 1022, and each lane rounded to `half`,
/// ties away from zero. The ends are the blend's source ranges.
#[test]
fn every_code_reads_metals_tile_value() {
    for form in EVERY_FORM {
        let alpha = |c: u32| match form {
            ExtendedRange::Planes { .. } => c % 256,
            _ => c,
        };
        let codes = |c: u32| [c, 1023 - c, (c * 7) % 1024, alpha(c)];
        let lanes: Vec<[u32; 4]> = (0..1024u32).map(|c| raw(form, codes(c), c * 13)).collect();
        let got = run(form, Op::Tile, &lanes);
        for (c, g) in (0..1024u32).zip(&got) {
            let want = tile_model(form, codes(c));
            assert_eq!(*g, want.map(f32::to_bits), "{form:?} code {c}");
        }
        // The other `_sRGB` forms' tiles end inside their source range.
        let (lo, hi) = match form {
            ExtendedRange::Bgra10 { srgb: true } => SRGB_SOURCE_RANGE,
            _ if form.srgb() => continue,
            _ => SOURCE_RANGE,
        };
        // Codes 0, 1022 and 1023: the source range's ends.
        assert_eq!(f32::from_bits(got[0][0]), lo, "{form:?}");
        assert_eq!(f32::from_bits(got[0][1]), hi, "{form:?}");
        assert_eq!(f32::from_bits(got[1][1]), hi, "{form:?}");
    }
}

/// A splitmix64 stream.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A source lane: mostly an `f32` off the `half` grid in `[-1.5, 2.5)`, sometimes a NaN, an
    /// infinity or a value far past the format's range.
    fn source(&mut self) -> f32 {
        match self.next() % 16 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            3 => 1e30,
            _ => (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 4.0 - 1.5,
        }
    }
}

/// Every blend factor.
const FACTORS: [BlendFactor; 19] = [
    BlendFactor::Zero,
    BlendFactor::One,
    BlendFactor::SourceColor,
    BlendFactor::OneMinusSourceColor,
    BlendFactor::SourceAlpha,
    BlendFactor::OneMinusSourceAlpha,
    BlendFactor::DestinationColor,
    BlendFactor::OneMinusDestinationColor,
    BlendFactor::DestinationAlpha,
    BlendFactor::OneMinusDestinationAlpha,
    BlendFactor::SourceAlphaSaturated,
    BlendFactor::BlendColor,
    BlendFactor::OneMinusBlendColor,
    BlendFactor::BlendAlpha,
    BlendFactor::OneMinusBlendAlpha,
    BlendFactor::Source1Color,
    BlendFactor::OneMinusSource1Color,
    BlendFactor::Source1Alpha,
    BlendFactor::OneMinusSource1Alpha,
];

/// Every blend operation.
const OPERATIONS: [BlendOperation; 5] = [
    BlendOperation::Add,
    BlendOperation::Subtract,
    BlendOperation::ReverseSubtract,
    BlendOperation::Min,
    BlendOperation::Max,
];

/// The device's blend (`blend::native`) of source `s`, second source `s1` and blend color `c`
/// over tile value `d` into `form`, storing the channels `mask`: each of `s`, `s1` and `c` clamped
/// to the form's range (color lanes to its source range, alpha to `[0, 1]`, a NaN to the low end),
/// then each class's `S·Fs op D·Fd` in `f32`, in the equation's order, each factor, product and
/// sum rounded once to `f32`, a factor of one its operand; `Min` and `Max` of the source and the
/// destination; a lane the mask does not store `d`'s.
fn native_model(
    form: ExtendedRange,
    eq: &BlendEquation,
    mask: ColorWriteMask,
    (s, s1, c): ([f32; 4], [f32; 4], [f32; 4]),
    d: [f32; 4],
) -> [f32; 4] {
    use BlendFactor as F;
    let (lo, hi) = match form {
        ExtendedRange::Bgra10 { srgb: true } => SRGB_SOURCE_RANGE,
        _ => SOURCE_RANGE,
    };
    let clamp = |v: [f32; 4]| -> [f32; 4] {
        std::array::from_fn(|k| {
            let (lo, hi) = if k == 3 { (0.0, 1.0) } else { (lo, hi) };
            if v[k].is_nan() {
                lo
            } else {
                v[k].clamp(lo, hi)
            }
        })
    };
    let (s, s1, c) = (clamp(s), clamp(s1), clamp(c));
    let factor = |f: BlendFactor, alpha: bool, k: usize| -> Option<f32> {
        Some(match f {
            F::Zero => 0.0,
            F::One => return None,
            F::SourceAlphaSaturated if alpha => return None,
            F::SourceColor => s[k],
            F::OneMinusSourceColor => 1.0 - s[k],
            F::SourceAlpha => s[3],
            F::OneMinusSourceAlpha => 1.0 - s[3],
            F::DestinationColor => d[k],
            F::OneMinusDestinationColor => 1.0 - d[k],
            F::DestinationAlpha => d[3],
            F::OneMinusDestinationAlpha => 1.0 - d[3],
            F::SourceAlphaSaturated => s[3].min(1.0 - d[3]),
            F::BlendColor => c[k],
            F::OneMinusBlendColor => 1.0 - c[k],
            F::BlendAlpha => c[3],
            F::OneMinusBlendAlpha => 1.0 - c[3],
            F::Source1Color => s1[k],
            F::OneMinusSource1Color => 1.0 - s1[k],
            F::Source1Alpha => s1[3],
            F::OneMinusSource1Alpha => 1.0 - s1[3],
            F::Unspecialized => unreachable!("no placeholder is drawn"),
        })
    };
    let class =
        |(fs, fd, op): (BlendFactor, BlendFactor, BlendOperation), alpha: bool, k: usize| {
            let source = factor(fs, alpha, k).map_or(s[k], |f| s[k] * f);
            let destination = factor(fd, alpha, k).map_or(d[k], |f| d[k] * f);
            match op {
                BlendOperation::Min => s[k].min(d[k]),
                BlendOperation::Max => s[k].max(d[k]),
                BlendOperation::Subtract => source - destination,
                BlendOperation::ReverseSubtract => destination - source,
                _ => source + destination,
            }
        };
    let color = [
        ColorWriteMask::RED,
        ColorWriteMask::GREEN,
        ColorWriteMask::BLUE,
    ]
    .into_iter()
    .any(|c| mask.contains(c));
    let rgb = (eq.source_rgb, eq.destination_rgb, eq.rgb_operation);
    let a = (eq.source_alpha, eq.destination_alpha, eq.alpha_operation);
    std::array::from_fn(|k| match k {
        3 if mask.contains(ColorWriteMask::ALPHA) => class(a, true, 3),
        3 => d[3],
        _ if color => class(rgb, false, k),
        _ => d[k],
    })
}

/// A blend into an extended-range attachment is the device's arithmetic (`blend::native`): over
/// sources off the `half` grid, NaNs, infinities and values past the range, random texels, blend
/// colors and second sources, every factor as each of the four factors and every operation for
/// color and for alpha, under every class of write mask, each lane holds the `f32` model's bits
/// ([`native_model`]): the clamp, then `S·Fs op D·Fd` in the equation's order, nothing rounded to
/// `half` between the steps. Metal's own forms (its lerps and common factors, its fused pairs and
/// a `half` rounding after each step) miss these bits at most lanes.
#[test]
fn every_blend_is_the_devices_arithmetic() {
    let masks = [
        ColorWriteMask::ALL,
        ColorWriteMask::RED
            .union(ColorWriteMask::GREEN)
            .union(ColorWriteMask::BLUE),
        ColorWriteMask::ALPHA,
        ColorWriteMask::GREEN,
    ];
    let mut rng = Rng(0x5EED_B1E4D);
    let mut checked = 0usize;
    for (fi, form) in EVERY_FORM.into_iter().enumerate() {
        for (i, &f) in FACTORS.iter().enumerate() {
            let k = i + fi;
            let eq = BlendEquation {
                source_rgb: f,
                destination_rgb: FACTORS[(k * 7 + 3) % FACTORS.len()],
                rgb_operation: OPERATIONS[k % OPERATIONS.len()],
                source_alpha: FACTORS[(k * 5 + 1) % FACTORS.len()],
                destination_alpha: f,
                alpha_operation: OPERATIONS[(k + 2) % OPERATIONS.len()],
            };
            let mask = masks[k % masks.len()];
            let lanes: Vec<[u32; 16]> = (0..24)
                .map(|_| {
                    let codes: [u32; 4] = std::array::from_fn(|c| match c {
                        3 if !matches!(form, ExtendedRange::Bgra10 { .. }) => {
                            (rng.next() % 256) as u32
                        }
                        _ => (rng.next() % 1024) as u32,
                    });
                    let texel = raw(form, codes, rng.next() as u32);
                    let mut w = [0u32; 16];
                    w[..4].copy_from_slice(&texel);
                    for x in w[4..].iter_mut() {
                        *x = rng.source().to_bits();
                    }
                    w
                })
                .collect();
            let tiles = run(
                form,
                Op::Tile,
                &lanes
                    .iter()
                    .map(|w| [w[0], w[1], w[2], w[3]])
                    .collect::<Vec<_>>(),
            );
            let got = run(form, Op::Blend(eq, mask), &lanes);
            for ((w, tile), got) in lanes.iter().zip(tiles).zip(got) {
                let f = |a: &[u32]| -> [f32; 4] { std::array::from_fn(|k| f32::from_bits(a[k])) };
                let d = f(&tile);
                let want = native_model(
                    form,
                    &eq,
                    mask,
                    (f(&w[4..8]), f(&w[12..16]), f(&w[8..12])),
                    d,
                );
                let same = (0..4).all(|k| {
                    want[k].to_bits() == got[k]
                        || (want[k].is_nan() && f32::from_bits(got[k]).is_nan())
                });
                assert!(
                    same,
                    "{form:?} {eq:?} {mask:?}: source {:?}, tile {d:?}: {:?} where the device's \
                     arithmetic gives {want:?}",
                    f(&w[4..8]),
                    got.map(f32::from_bits)
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, EVERY_FORM.len() * FACTORS.len() * 24);
}

/// A blend's placeholder factor or operation is refused as the placeholder it is; every other
/// factor and operation is built, under every write mask.
#[test]
fn every_blend_is_built_but_a_placeholder() {
    use super::blend::NativeBlend;
    for &fs in &FACTORS {
        for op in OPERATIONS {
            let eq = BlendEquation {
                source_rgb: fs,
                destination_rgb: BlendFactor::OneMinusSourceAlpha,
                rgb_operation: op,
                source_alpha: BlendFactor::One,
                destination_alpha: fs,
                alpha_operation: op,
            };
            for mask in [
                ColorWriteMask::ALL,
                ColorWriteMask::RED,
                ColorWriteMask::ALPHA,
            ] {
                assert!(NativeBlend::of(&eq, mask).is_ok(), "{eq:?} {mask:?}");
            }
        }
    }
    let placeholder = BlendEquation {
        source_rgb: BlendFactor::Unspecialized,
        destination_rgb: BlendFactor::Zero,
        rgb_operation: BlendOperation::Add,
        source_alpha: BlendFactor::One,
        destination_alpha: BlendFactor::Zero,
        alpha_operation: BlendOperation::Add,
    };
    assert!(matches!(
        NativeBlend::of(&placeholder, ColorWriteMask::ALL),
        Err(Error::InvalidDescriptor(_))
    ));
    let op = BlendEquation {
        rgb_operation: BlendOperation::Unspecialized,
        source_rgb: BlendFactor::One,
        ..placeholder
    };
    assert!(matches!(
        NativeBlend::of(&op, ColorWriteMask::ALL),
        Err(Error::InvalidDescriptor(_))
    ));
}

/// A padded 10-bit form reads every 16-bit word as the code in its top 10 bits over 1023, correctly
/// rounded (a `half` result to nearest even), blue 0 and alpha 1, whatever the low 6 bits hold
/// (every word); and writes the code of the exact product of the value clamped to `[0, 1]` (a
/// NaN 0) with 1023, rounded to nearest, in its word's top 10 bits, the low 6 bits zero (a grid,
/// and three `f32`s about every midpoint, each lane).
#[test]
fn padded_forms_read_and_write_metals_codes() {
    let value = |w: u32| ((w >> 6) as f64 / 1023.0) as f32;
    for form in [ExtendedRange::R10X6, ExtendedRange::Rg10X12] {
        let two = form == ExtendedRange::Rg10X12;
        // Every code under four fillings of the low 6 bits, the other lanes' words other codes.
        let lanes: Vec<[u32; 4]> = (0..4096u32)
            .map(|i| {
                let w = (i & 1023) << 6 | [0, 0x3F, 0x15, 0x2A][(i >> 10) as usize];
                [w, w.wrapping_mul(40503) & 0xFFFF, 0xFFFF, 0xFFFF]
            })
            .collect();
        let got = run(form, Op::Decode, &lanes);
        let halves = run(form, Op::DecodeHalf, &lanes);
        for ((l, g), h) in lanes.iter().zip(&got).zip(&halves) {
            let green = if two { value(l[1]) } else { 0.0 };
            let want = [value(l[0]), green, 0.0, 1.0];
            assert_eq!(*g, want.map(f32::to_bits), "{form:?} words {l:x?}");
            let want = want.map(|v| to_half(f64::from(v), false));
            assert_eq!(*h, want.map(f32::to_bits), "{form:?} half, words {l:x?}");
        }
        // Writes: about every midpoint, the codes, and the specials.
        let mut values = vec![1.0f32];
        for k in 0..1023u32 {
            values.push((f64::from(k) / 1023.0) as f32);
            let mid = ((f64::from(k) + 0.5) / 1023.0) as f32;
            values.extend((0..5).map(|d| f32::from_bits(mid.to_bits() + d - 2)));
        }
        values.extend([
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            -1.0,
            1.5,
            2.0,
            f32::from_bits(1),
            -f32::from_bits(1),
        ]);
        let code = |v: f32| -> u32 {
            match v.is_nan() {
                true => 0,
                false => (f64::from(v.clamp(0.0, 1.0)) * 1023.0).round_ties_even() as u32,
            }
        };
        let lanes: Vec<[u32; 4]> = (0..values.len())
            .map(|i| {
                let g = values[(i * 7919) % values.len()];
                [
                    values[i].to_bits(),
                    g.to_bits(),
                    0.5f32.to_bits(),
                    1f32.to_bits(),
                ]
            })
            .collect();
        let got = run(form, Op::Encode, &lanes);
        for (l, g) in lanes.iter().zip(&got) {
            let (r, gr) = (f32::from_bits(l[0]), f32::from_bits(l[1]));
            let green = if two { code(gr) << 6 } else { 0 };
            assert_eq!(*g, [code(r) << 6, green, 0, 0], "{form:?} values {r}, {gr}");
        }
    }
}

/// The code Metal's write of `v` stores to a padded `_sRGB` form's channel (the form docs): a NaN
/// 0, the value clamped to `[0, 1]`; `R10Unorm_X6_sRGB`'s the largest code whose first encode index
/// is at most the value rounded to `2^-13` (nearest even), `RG10Unorm_X12_sRGB`'s the table's code
/// of `RG10Unorm_X12`'s.
fn padded_srgb_code(form: ExtendedRange, v: f32) -> u32 {
    let v = match v.is_nan() {
        true => 0.0,
        false => f64::from(v.clamp(0.0, 1.0)),
    };
    match form {
        ExtendedRange::R10X6Srgb => {
            let i = (v * 8192.0).round_ties_even() as u16;
            let codes = R10X6_SRGB_ENCODE_FIRST.partition_point(|&f| f <= i);
            codes as u32 - 1
        }
        _ => u32::from(RG10X12_SRGB_WRITE_CODES[(v * 1023.0).round_ties_even() as usize]),
    }
}

/// `R10Unorm_X6_sRGB` reads every 16-bit word as its code's decode step over 8192 (a `half` result
/// with ties away from zero), `RG10Unorm_X12_sRGB` as `RG10Unorm_X12` does (`c / 1023`, a `half` to
/// nearest even), blue 0 and alpha 1, whatever the low 6 bits hold; and each writes Metal's code
/// through its table ([`padded_srgb_code`]) in its word's top 10 bits, the low 6 bits zero: values
/// about every threshold of `R10Unorm_X6_sRGB`'s codes and every midpoint of the linear codes, the
/// codes' values, and the specials.
#[test]
fn padded_srgb_forms_read_and_write_metals_codes() {
    let step = |w: u32| f32::from(R10X6_SRGB_DECODE_STEPS[(w >> 6) as usize]) / 8192.0;
    let linear = |w: u32| ((w >> 6) as f64 / 1023.0) as f32;
    // Values about every threshold and midpoint, the codes' values, and the specials.
    let mut values = vec![1.0f32];
    for k in 0..1024u32 {
        values.push(step(k << 6));
        values.push((f64::from(k) / 1023.0) as f32);
        let mid = ((f64::from(k) + 0.5) / 1023.0) as f32;
        values.extend((0..5).map(|d| f32::from_bits(mid.to_bits() + d - 2)));
        let first = f64::from(R10X6_SRGB_ENCODE_FIRST[k as usize]);
        if first > 0.0 {
            let edge = ((first - 0.5) / 8192.0) as f32;
            values.extend((0..5).map(|d| f32::from_bits(edge.to_bits() + d - 2)));
        }
    }
    values.extend([
        f32::NAN,
        -f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        -1.0,
        1.5,
        2.0,
        f32::from_bits(1),
        -f32::from_bits(1),
    ]);
    for form in [ExtendedRange::R10X6Srgb, ExtendedRange::Rg10X12Srgb] {
        let two = form == ExtendedRange::Rg10X12Srgb;
        let (red, rounding_away) = match two {
            false => (step as fn(u32) -> f32, true),
            true => (linear as fn(u32) -> f32, false),
        };
        // Every code under four fillings of the low 6 bits, the other lanes' words other codes.
        let lanes: Vec<[u32; 4]> = (0..4096u32)
            .map(|i| {
                let w = (i & 1023) << 6 | [0, 0x3F, 0x15, 0x2A][(i >> 10) as usize];
                [w, w.wrapping_mul(40503) & 0xFFFF, 0xFFFF, 0xFFFF]
            })
            .collect();
        let got = run(form, Op::Decode, &lanes);
        let halves = run(form, Op::DecodeHalf, &lanes);
        for ((l, g), h) in lanes.iter().zip(&got).zip(&halves) {
            let green = if two { linear(l[1]) } else { 0.0 };
            let want = [red(l[0]), green, 0.0, 1.0];
            assert_eq!(*g, want.map(f32::to_bits), "{form:?} words {l:x?}");
            let want = [
                to_half(f64::from(want[0]), rounding_away),
                to_half(f64::from(green), false),
                0.0,
                1.0,
            ];
            assert_eq!(*h, want.map(f32::to_bits), "{form:?} half, words {l:x?}");
        }
        let lanes: Vec<[u32; 4]> = (0..values.len())
            .map(|i| {
                let g = values[(i * 7919) % values.len()];
                [
                    values[i].to_bits(),
                    g.to_bits(),
                    0.5f32.to_bits(),
                    1f32.to_bits(),
                ]
            })
            .collect();
        let got = run(form, Op::Encode, &lanes);
        for (l, g) in lanes.iter().zip(&got) {
            let (r, gr) = (f32::from_bits(l[0]), f32::from_bits(l[1]));
            let green = if two {
                padded_srgb_code(form, gr) << 6
            } else {
                0
            };
            let want = [padded_srgb_code(form, r) << 6, green, 0, 0];
            assert_eq!(*g, want, "{form:?} values {r}, {gr}");
        }
    }
}

/// `a / 3` correctly rounded to `f32` (`a` below 4): the `f64` quotient's neighbours compared
/// exactly (each `f32` times 3 is exact in `f64`).
fn third(a: u32) -> f32 {
    let q = f64::from(a) / 3.0;
    let y = q as f32;
    [
        y,
        f32::from_bits(y.to_bits().wrapping_sub(1)),
        f32::from_bits(y.to_bits() + 1),
    ]
    .into_iter()
    .min_by(|x, z| {
        let d = |v: &f32| (f64::from(*v) * 3.0 - f64::from(a)).abs();
        d(x).total_cmp(&d(z))
    })
    .expect("a candidate")
}

/// `RGB10A2Unorm_sRGB` (an Apple M3 Max under macOS 27.0.1, our probe: every code of each
/// channel read, 16,385 values and the specials written in each) reads each color field's code as
/// `R10Unorm_X6_sRGB`'s red (its decode step over 8192, a `half` with ties away from zero) and
/// alpha's 2-bit code `a` as `a / 3`, and writes each color channel as `R10Unorm_X6_sRGB`'s red
/// and alpha as the exact product with 3 of the value clamped to `[0, 1]` (a NaN 0), rounded to
/// nearest, into one word: red from bit 0, green 10, blue 20, alpha 30. Its clear texels and load
/// codes follow the same rules.
#[test]
fn rgb10a2_srgb_reads_and_writes_metals_codes() {
    let form = ExtendedRange::Rgb10A2Srgb;
    let step = |c: u32| f32::from(R10X6_SRGB_DECODE_STEPS[c as usize]) / 8192.0;
    // The probe's words: red c, green 1023 - c, blue 7c mod 1024, alpha c mod 4.
    let lanes: Vec<[u32; 4]> = (0..1024u32)
        .map(|c| {
            [
                c | (1023 - c) << 10 | (7 * c % 1024) << 20 | (c % 4) << 30,
                0,
                0,
                0,
            ]
        })
        .collect();
    let got = run(form, Op::Decode, &lanes);
    let halves = run(form, Op::DecodeHalf, &lanes);
    for ((l, g), h) in lanes.iter().zip(&got).zip(&halves) {
        let w = l[0];
        let want = [
            step(w & 1023),
            step(w >> 10 & 1023),
            step(w >> 20 & 1023),
            third(w >> 30),
        ];
        assert_eq!(*g, want.map(f32::to_bits), "word {w:08x}");
        let want = want.map(|v| to_half(f64::from(v), true));
        assert_eq!(*h, want.map(f32::to_bits), "half, word {w:08x}");
    }
    let mut values: Vec<f32> = (0..=16384u32).map(|k| k as f32 / 16384.0).collect();
    for &first in &R10X6_SRGB_ENCODE_FIRST {
        let first = f64::from(first);
        if first > 0.0 {
            let edge = ((first - 0.5) / 8192.0) as f32;
            values.extend((0..5).map(|d| f32::from_bits(edge.to_bits() + d - 2)));
        }
    }
    for a in [1.0f64, 2.0] {
        let mid = ((a - 0.5) / 3.0) as f32;
        values.extend((0..5).map(|d| f32::from_bits(mid.to_bits() + d - 2)));
    }
    values.extend([
        f32::NAN,
        -f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        -1.0,
        1.5,
        f32::from_bits(1),
    ]);
    let alpha_code = |v: f32| match v.is_nan() {
        true => 0,
        false => (f64::from(v.clamp(0.0, 1.0)) * 3.0).round_ties_even() as u32,
    };
    let lanes: Vec<[u32; 4]> = (0..values.len())
        .map(|i| {
            let v = |k: usize| values[(i * k) % values.len()].to_bits();
            [v(1), v(7919), v(104_729), v(31)]
        })
        .collect();
    let got = run(form, Op::Encode, &lanes);
    for (l, g) in lanes.iter().zip(&got) {
        let v = l.map(f32::from_bits);
        let code = |c: usize| padded_srgb_code(ExtendedRange::R10X6Srgb, v[c]);
        let word = code(0) | code(1) << 10 | code(2) << 20 | alpha_code(v[3]) << 30;
        assert_eq!(*g, [word, 0, 0, 0], "values {v:?}");
    }
    // The crate's clear texels agree with the rules (the runtime's clear writes `clear_texel`).
    let half = |v: f64| crate::float16::f16_bits_to_f32(crate::float16::f32_to_f16_bits(v as f32));
    for color in [
        [0.25, 0.5, 0.75, 0.4],
        [1.0, 0.0, 0.1, 1.0],
        [-1.0, 2.0, f64::NAN, 0.2],
    ] {
        let route = crate::metal::format::ClearRoute::ThroughHalf;
        let code = |c: usize| padded_srgb_code(ExtendedRange::R10X6Srgb, half(color[c]));
        let word = code(0) | code(1) << 10 | code(2) << 20 | alpha_code(half(color[3])) << 30;
        assert_eq!(form.clear_texel(color, route), [word, 0, 0, 0], "{color:?}");
    }
}

/// Metal's `float` of bfloat code `c`, as bits (`metal::format::ExtendedRange::R16Bfloat`): the
/// code shifted up 16, any NaN code the quiet NaN.
fn bfloat_read(c: u32) -> u32 {
    match c & 0x7FFF > 0x7F80 {
        true => 0x7FC0_0000,
        false => c << 16,
    }
}

/// Metal's code of a kernel's write of the `f32` of bits `b` to a bfloat texture: its top 16 bits,
/// a NaN `0x7fc0`.
fn bfloat_write(b: u32) -> u32 {
    match b & 0x7FFF_FFFF > 0x7F80_0000 {
        true => 0x7FC0,
        false => b >> 16,
    }
}

/// Every code of each bfloat form in each channel reads Metal's `float` and its `half` (rounded to
/// nearest even, overflows to the infinities), the channels a form lacks 0 and alpha 1 whatever
/// their words hold; a write stores Metal's codes: per code its bits and its tie's neighbours,
/// every eighth `half` widened, and the specials, the components a form lacks zero (an Apple M4:
/// every code, read and written, in each bfloat format).
#[test]
fn bfloat_forms_read_and_write_metals_codes() {
    let forms = [
        (ExtendedRange::R16Bfloat, 1usize, 4096u32),
        (ExtendedRange::Rg16Bfloat, 2, 4096),
        (ExtendedRange::Rgba16Bfloat, 4, 65536),
    ];
    let one = 1.0f32.to_bits();
    for (form, ch, n) in forms {
        let step = 65536 / n;
        // Every code (every `step`-th in the narrower forms) in each channel; the words a form
        // lacks hold junk.
        let lanes: Vec<[u32; 4]> = (0..n)
            .map(|i| {
                std::array::from_fn(|c| {
                    let code = (i * step + 16411 * c as u32 + c as u32) & 0xFFFF;
                    match c < ch {
                        true => code,
                        false => code | 0xABC0_0000,
                    }
                })
            })
            .collect();
        let floats = run(form, Op::Decode, &lanes);
        let halves = run(form, Op::DecodeHalf, &lanes);
        for ((l, f), h) in lanes.iter().zip(&floats).zip(&halves) {
            let want: [u32; 4] = std::array::from_fn(|c| match (c < ch, c) {
                (true, _) => bfloat_read(l[c]),
                (false, 3) => one,
                (false, _) => 0,
            });
            assert_eq!(*f, want, "{form:?} words {l:x?}");
            let half = |b: u32| {
                let h = crate::float16::f32_to_f16_bits(f32::from_bits(b));
                crate::float16::f16_bits_to_f32(h).to_bits()
            };
            assert_eq!(*h, want.map(half), "{form:?} half, words {l:x?}");
        }
        // Writes: per code its bits and the neighbours of its tie, every eighth `half`, and the
        // specials, `value k` in channel `k % 4` of invocation `k / 4`.
        let mut values: Vec<u32> = (0..65536u32)
            .step_by(16)
            .flat_map(|c| [0, 0x7FFF, 0x8000, 0x8001, 0xFFFF].map(|low| c << 16 | low))
            .collect();
        values.extend(
            (0..65536u32)
                .step_by(8)
                .map(|h| crate::float16::f16_bits_to_f32(h as u16).to_bits()),
        );
        values.extend([
            f32::NAN.to_bits(),
            (-f32::NAN).to_bits(),
            0x7F80_0001,
            0xFFFF_FFFF,
            f32::INFINITY.to_bits(),
            f32::NEG_INFINITY.to_bits(),
            1,
            0x8000_0001,
        ]);
        while !values.len().is_multiple_of(4) {
            values.push(0);
        }
        let lanes: Vec<[u32; 4]> = values
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        let got = run(form, Op::Encode, &lanes);
        for (l, g) in lanes.iter().zip(&got) {
            let want: [u32; 4] = std::array::from_fn(|c| match c < ch {
                true => bfloat_write(l[c]),
                false => 0,
            });
            assert_eq!(*g, want, "{form:?} values {l:x?}");
        }
    }
}

/// FNV-1a of Metal's codes (each little-endian, 2 bytes) of a fragment function's `float` outputs
/// to a bfloat attachment, at the default `textureWriteRoundingMode` and at `NearestEven`: per
/// code `c` the `float`s of bits `c << 16`, and `| 0x7fff`, `| 0x8000` and `| 0x8001`, in that
/// order; every form's codes are these (an Apple M4 under macOS 27.0.1, each
/// value into `R16Bfloat`, `RG16Bfloat` and `RGBA16Bfloat`).
const METAL_FLOAT_OUTPUTS: [(bool, u64); 2] = [
    (false, 0x4334_841e_f3f2_1ba5),
    (true, 0x5142_c964_386f_8c65),
];

/// FNV-1a of Metal's codes of a fragment function's `half` outputs to a bfloat attachment, every
/// `half` code in order, at either mode.
const METAL_HALF_OUTPUTS: u64 = 0x9eb0_0b77_54b2_35e5;

/// The FNV-1a 64 of `bytes`.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xCBF2_9CE4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
    })
}

/// A fragment function's `float` and `half` outputs to a bfloat attachment store Metal's codes:
/// a `float` rounded to `half` toward zero (65504 at most) or, at `NearestEven`, to nearest even,
/// then the bfloat code nearest, ties to even, a NaN `0x7fc0`; every `float` around each code's
/// tie and every `half` code against Metal's digests, the narrower forms keeping their lanes, and
/// named answers.
#[test]
fn bfloat_attachments_store_metals_codes() {
    let digest = |got: &[[u32; 4]]| {
        assert!(got.iter().flatten().all(|&c| c <= 0xFFFF));
        let bytes: Vec<u8> = got
            .iter()
            .flatten()
            .flat_map(|&c| (c as u16).to_le_bytes())
            .collect();
        fnv(&bytes)
    };
    let floats: Vec<[u32; 4]> = (0..65536u32)
        .map(|c| [0, 0x7FFF, 0x8000, 0x8001].map(|low| c << 16 | low))
        .collect();
    let halves: Vec<[u32; 4]> = (0..16384u32)
        .map(|i| std::array::from_fn(|k| half(4 * i as u16 + k as u16).to_bits()))
        .collect();
    for (nearest_even, metal) in METAL_FLOAT_OUTPUTS {
        let form = ExtendedRange::Rgba16Bfloat;
        let got = run(form, Op::Output(nearest_even, FloatFmt::F32), &floats);
        assert_eq!(
            digest(&got),
            metal,
            "float outputs, nearest even {nearest_even}"
        );
        let got = run(form, Op::Output(nearest_even, FloatFmt::Half), &halves);
        assert_eq!(digest(&got), METAL_HALF_OUTPUTS, "half outputs");
        // The narrower forms keep their lanes of the same codes, the rest zero.
        let sample: Vec<[u32; 4]> = floats.iter().copied().step_by(61).collect();
        let wide = run(form, Op::Output(nearest_even, FloatFmt::F32), &sample);
        for (narrow, kept) in [
            (ExtendedRange::R16Bfloat, 1),
            (ExtendedRange::Rg16Bfloat, 2),
        ] {
            let got = run(narrow, Op::Output(nearest_even, FloatFmt::F32), &sample);
            for (g, w) in got.iter().zip(&wide) {
                let want: [u32; 4] = std::array::from_fn(|k| if k < kept { w[k] } else { 0 });
                assert_eq!(*g, want, "{narrow:?}");
            }
        }
    }
    // Named answers: `float` bits, the default mode's code and `NearestEven`'s.
    let named: [(u32, u32, u32); 12] = [
        (0x477F_F000, 0x4780, 0x7F80),
        (0xC77F_F000, 0xC780, 0xFF80),
        (0x477F_EFFF, 0x4780, 0x4780),
        (0x3F81_7FFF, 0x3F81, 0x3F82),
        (0x3F81_8000, 0x3F82, 0x3F82),
        (0x3F80_8000, 0x3F80, 0x3F80),
        (0x3300_0001, 0, 0x3380),
        (0x3300_0000, 0, 0),
        (0x8000_0000, 0x8000, 0x8000),
        (0xFF80_0000, 0xFF80, 0xFF80),
        (0x7FC0_0001, 0x7FC0, 0x7FC0),
        (0xFFC0_0000, 0x7FC0, 0x7FC0),
    ];
    let lanes: Vec<[u32; 4]> = named.iter().map(|&(b, ..)| [b; 4]).collect();
    for nearest_even in [false, true] {
        let got = run(
            ExtendedRange::Rgba16Bfloat,
            Op::Output(nearest_even, FloatFmt::F32),
            &lanes,
        );
        for (&(b, default, nearest), g) in named.iter().zip(&got) {
            let want = if nearest_even { nearest } else { default };
            assert_eq!(g[0], want, "{b:#x}, nearest even {nearest_even}");
        }
    }
}

/// A read of what the tile holds clamps as Metal's does (an Apple M4: unblended
/// `half` writes read back by a fetch): each color lane to the form's source range, a NaN to its
/// low end, the infinities and values past the range to its ends; alpha to `[0, 1]`, a NaN to 0;
/// a subnormal `half` kept; `BGR10_XR`'s alpha one.
#[test]
fn a_read_of_the_tile_clamps_as_metals_does() {
    let (n, inf) = (f32::NAN, f32::INFINITY);
    let subnormal = crate::float16::f16_bits_to_f32(7);
    let written: [[f32; 4]; 5] = [
        [n, -inf, inf, -0.5],
        [inf, n, -inf, n],
        [-inf, 3.0, n, 2.0],
        [10000.0, -3.0, 0.25, inf],
        [2.0, -1.0, subnormal, 0.5],
    ];
    for form in [
        ExtendedRange::Bgra10 { srgb: false },
        ExtendedRange::Bgra10 { srgb: true },
        ExtendedRange::Bgr10 { srgb: true },
        ExtendedRange::Planes { srgb: false },
    ] {
        let (lo, hi) = match form {
            ExtendedRange::Bgra10 { srgb: true } => SRGB_SOURCE_RANGE,
            _ => SOURCE_RANGE,
        };
        let a = |v: f32| match form {
            ExtendedRange::Bgr10 { .. } => 1.0,
            _ => v,
        };
        let want: [[f32; 4]; 5] = [
            [lo, lo, hi, a(0.0)],
            [hi, lo, lo, a(0.0)],
            [lo, hi, lo, a(1.0)],
            [hi, lo, 0.25, a(1.0)],
            [hi, lo, subnormal, a(0.5)],
        ];
        let lanes: Vec<[u32; 4]> = written.iter().map(|w| w.map(f32::to_bits)).collect();
        let got = run(form, Op::TileRead, &lanes);
        for ((w, g), want) in written.iter().zip(&got).zip(&want) {
            assert_eq!(*g, want.map(f32::to_bits), "{form:?}: {w:?}");
        }
    }
}
