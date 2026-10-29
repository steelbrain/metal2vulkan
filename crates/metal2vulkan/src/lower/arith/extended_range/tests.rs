//! The extended-range conversions run through the interpreter: every code read, and the writes
//! around every rounding boundary, against the device's arithmetic; the `_sRGB` forms' transfer
//! (the device's `Pow`, which the interpreter does not model) checked in the module and against
//! Metal's tables.

use super::blend::{extended_ranges, native, NativeBlend};
use super::{decode, encode, tile_read, tile_value, SOURCE_RANGE, SRGB_SOURCE_RANGE};
use crate::error::Error;
use crate::lower::arith::{FloatFmt, Form, Held};
use crate::lower::stage_io::extended_color::bfloat_output;
use crate::metal::format::extended_srgb::{SRGB_DECODE_STEPS, SRGB_ENCODE_FIRST};
use crate::metal::format::padded_srgb::{R10X6_SRGB_ENCODE_FIRST, RG10X12_SRGB_WRITE_CODES};
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

/// The linear and `_sRGB` forms of `BGR10_XR`, `BGRA10_XR` and the two-plane format.
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
            encode(&mut f, form, v)?
        }
        Op::Decode => {
            let v = decode(&mut f, form, x, false)?;
            f.convert(ConvOp::Bitcast, v, uvec4)?
        }
        Op::DecodeHalf => {
            let v = decode(&mut f, form, x, true)?;
            f.convert(ConvOp::Bitcast, v, uvec4)?
        }
        Op::Tile => {
            let v = tile_value(&mut f, form, x)?;
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
            let d = tile_value(&mut f, form, x)?;
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

/// The `f32` nearest `1 / d`.
fn reciprocal(d: f32) -> f32 {
    1.0 / d
}

/// The device's value of code `c` of a linear color channel: `float(c - 384)` times the `f32`
/// nearest `1 / 510`, rounded once.
fn color_value(c: u32) -> f32 {
    (c as i32 - 384) as f32 * reciprocal(510.0)
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

/// The device's value of the alpha code `c` of `form` (a two-plane texel's byte).
fn alpha_value(form: ExtendedRange, c: u32) -> f32 {
    match form {
        ExtendedRange::Bgr10 { .. } => 1.0,
        ExtendedRange::Planes { .. } => (c & 0xFF) as f32 * reciprocal(255.0),
        _ => color_value(c.clamp(384, 894)),
    }
}
/// The three linear forms.
const LINEAR: [ExtendedRange; 3] = [
    ExtendedRange::Bgra10 { srgb: false },
    ExtendedRange::Bgr10 { srgb: false },
    ExtendedRange::Planes { srgb: false },
];

/// The code a write of alpha value `v` stores in a two-plane texel's alpha byte: the `f32`
/// product with 255, `RoundEven`.
fn alpha_byte(v: f32) -> u32 {
    let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    round_even(f64::from(v * 255.0)) as u32
}

/// Every code of every channel of a linear form reads the device's product, `float(c - 384)`
/// times the `f32` nearest `1 / 510` (a two-plane texel's alpha byte times that of `1 / 255`),
/// alpha clamped to `[0, 1]` (1 where the format has none), whatever the bits Metal does not
/// read hold; a `half` read is that value narrowed to nearest even. The product misses Metal's
/// correctly rounded quotient at some codes, so the old quotient fails here.
#[test]
fn every_code_reads_the_devices_product() {
    let mut apart = 0;
    for form in LINEAR {
        let codes = |c: u32| [c, 1023 - c, (c * 7) % 1024, c];
        let lanes: Vec<[u32; 4]> = (0..1024u32).map(|c| raw(form, codes(c), c * 13)).collect();
        let floats = run(form, Op::Decode, &lanes);
        let halves = run(form, Op::DecodeHalf, &lanes);
        for ((c, f), h) in (0..1024u32).zip(&floats).zip(&halves) {
            let k = codes(c);
            let want = [
                color_value(k[0]),
                color_value(k[1]),
                color_value(k[2]),
                alpha_value(form, k[3]),
            ];
            assert_eq!(*f, want.map(f32::to_bits), "{form:?} code {c}");
            assert_eq!(
                *h,
                want.map(|v| to_half(f64::from(v), false).to_bits()),
                "{form:?} half, code {c}"
            );
            let quotient = ((c as i32 - 384) as f64 / 510.0) as f32;
            apart += usize::from(quotient != want[0]);
        }
    }
    assert!(apart > 0, "the product never misses the quotient");
}

/// `x` rounded to the nearest integer, ties to even (`x` exact in `f64`).
fn round_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// The code the device writes for linear color value `v` (alpha where `alpha`): the value, a NaN
/// 0, clamped to the linear values of codes 0 and 1023 (alpha to `[0, 1]`), times 510 in `f32`,
/// `RoundEven`, plus 384.
fn code(v: f32, alpha: bool) -> u32 {
    let v = if v.is_nan() { 0.0 } else { v };
    let (lo, hi) = match alpha {
        true => (0.0, 1.0),
        false => ((-384.0f64 / 510.0) as f32, (639.0f64 / 510.0) as f32),
    };
    let k = round_even(f64::from(v.clamp(lo, hi) * 510.0));
    (k + 384.0) as u32
}

/// A write of a linear form stores the device's rounded product: each code's value and its
/// neighbours, the `f32`s either side of each midpoint, and the specials (a NaN the zero code,
/// the infinities and values past the range the end codes); bits Metal does not read are written
/// as it writes them.
#[test]
fn writes_store_the_devices_rounded_product() {
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
    let mut xs: Vec<f32> = specials.to_vec();
    for c in 0..1023u32 {
        let (a, b) = (color_value(c), color_value(c + 1));
        let mid = ((f64::from(a) + f64::from(b)) / 2.0) as f32;
        xs.extend([
            a,
            mid,
            f32::from_bits(mid.to_bits() + 1),
            f32::from_bits(mid.to_bits().wrapping_sub(1)),
        ]);
    }
    for form in LINEAR {
        let lanes: Vec<[u32; 4]> = xs
            .iter()
            .map(|x| [x.to_bits(), (-x).to_bits(), x.to_bits(), x.to_bits()])
            .collect();
        let got = run(form, Op::Encode, &lanes);
        for (x, g) in xs.iter().zip(&got) {
            let alpha = match form {
                ExtendedRange::Planes { .. } => alpha_byte(*x),
                _ => code(*x, true),
            };
            let rgba = [code(*x, false), code(-x, false), code(*x, false), alpha];
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

/// Every word of the module serialized from `m`.
fn module_words(m: TypedModule) -> Vec<u32> {
    crate::spirv::serialize::serialize(m)
        .expect("serialize")
        .words
}

/// Whether `words` hold a `GLSL.std.450` `Pow` and an initialized `Private` variable (a table).
fn pow_and_table(words: &[u32]) -> (bool, bool) {
    let (mut pow, mut table) = (false, false);
    let mut at = 5;
    while at < words.len() {
        let (count, op) = ((words[at] >> 16) as usize, words[at] & 0xFFFF);
        // `OpExtInst` 12 (instruction number 26, `Pow`); `OpVariable` 59 with an initializer in
        // `Private` (6).
        pow |= op == 12 && words.get(at + 4) == Some(&26);
        table |= op == 59 && count == 5 && words.get(at + 3) == Some(&6);
        at += count.max(1);
    }
    (pow, table)
}

/// An `_sRGB` form's read, tile value and write take the MSL transfer by the device's `Pow` and
/// declare no table (Metal's tables are gone from the module), as do the padded `_sRGB` forms'.
#[test]
fn srgb_forms_take_the_transfer_by_the_devices_pow() {
    let forms = [
        ExtendedRange::Bgr10 { srgb: true },
        ExtendedRange::Bgra10 { srgb: true },
        ExtendedRange::Planes { srgb: true },
        ExtendedRange::R10X6Srgb,
        ExtendedRange::Rg10X12Srgb,
        ExtendedRange::Rgb10A2Srgb,
    ];
    for form in forms {
        let mut ops = vec![Op::Encode];
        // `RG10Unorm_X12_sRGB` reads its codes linearly.
        if form != ExtendedRange::Rg10X12Srgb {
            ops.extend([Op::Decode, Op::DecodeHalf]);
        }
        if form.srgb() {
            ops.push(Op::Tile);
        }
        for op in ops {
            let words = module_words(kernel(form, op).expect("kernel"));
            assert_eq!(pow_and_table(&words), (true, false), "{form:?}");
        }
    }
    let words = module_words(kernel(ExtendedRange::Rg10X12Srgb, Op::Decode).expect("kernel"));
    assert_eq!(pow_and_table(&words), (false, false));
}

/// The MSL transfer in `f64` (`f32` lanes of the device differ by its `Pow`'s error): the encode
/// of `x` in `[0, 1]`, and the decode.
fn srgb_encode(x: f64) -> f64 {
    match x < 0.003_130_8 {
        true => 12.92 * x,
        false => 1.055 * x.powf(1.0 / 2.4) - 0.055,
    }
}

/// See [`srgb_encode`].
fn srgb_decode(c: f64) -> f64 {
    match c <= 0.040_45 {
        true => c / 12.92,
        false => ((c + 0.055) / 1.055).powf(2.4),
    }
}

/// The transfer meets Metal's measured tables within `DevicePackConversion`'s bound, as the
/// module docs state: half a `2^-13` step of every extended-range decode step, a code of every
/// encode on the `2^-13` grid (32 of them apart), a code of `R10Unorm_X6_sRGB`'s encode, and a code
/// of `RG10Unorm_X12_sRGB`'s write of every linear code (36 apart).
#[test]
fn the_transfer_meets_metals_tables_within_the_note() {
    for (n, &s) in SRGB_DECODE_STEPS.iter().enumerate() {
        let v = srgb_decode(f64::from(n as f32 * reciprocal(510.0)));
        assert!((v * 8192.0 - f64::from(s)).abs() <= 0.5, "offset {n}");
    }
    let first = |table: &[u16], i: u32| table.partition_point(|&f| u32::from(f) <= i) as i64 - 1;
    let mut apart = 0;
    for i in 0..=2 * 8192u32 {
        let x = f64::from(i) / 8192.0;
        let ours = (srgb_encode(x) * 510.0).round_ties_even().min(639.0) as i64;
        let metal = first(&SRGB_ENCODE_FIRST, i);
        assert!((ours - metal).abs() <= 1, "{x}");
        apart += usize::from(ours != metal);
    }
    assert_eq!(apart, 32);
    for i in 0..=8192u32 {
        let x = f64::from(i) / 8192.0;
        let ours = (srgb_encode(x) * 1023.0).round_ties_even() as i64;
        assert!(
            (ours - first(&R10X6_SRGB_ENCODE_FIRST, i)).abs() <= 1,
            "{x}"
        );
    }
    let mut apart = 0;
    for (m, &metal) in RG10X12_SRGB_WRITE_CODES.iter().enumerate() {
        let t = f64::from(m as f32 * reciprocal(4092.0));
        let ours = (srgb_encode(t) * 255.0).round_ties_even() as i64;
        assert!((ours - i64::from(metal)).abs() <= 1, "linear code {m}");
        apart += usize::from(ours != i64::from(metal));
    }
    assert_eq!(apart, 36);
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

/// The tile value of a texel of linear `form` whose codes are `rgba` (a two-plane form's alpha its
/// byte), as the module docs of `tile_value` state the rule: a code above 1022 as 1022, each lane
/// the device's value narrowed to `half`, nearest even. The `_sRGB` forms' are the device's
/// (the interpreter models no `Pow`).
pub(crate) fn tile_model(form: ExtendedRange, rgba: [u32; 4]) -> [f32; 4] {
    assert!(
        !form.srgb() && !form.padded() && !form.bfloat(),
        "{form:?}: not a linear form"
    );
    let color = |c: u32| color_value(c.min(1022));
    [
        color(rgba[0]),
        color(rgba[1]),
        color(rgba[2]),
        alpha_value(form, rgba[3]),
    ]
    .map(|v| to_half(f64::from(v), false))
}

/// Every code of every channel of the linear forms reads its tile value, whatever the bits Metal
/// does not read hold: a code above 1022 as 1022, each lane narrowed to `half`. The ends are the
/// blend's source range.
#[test]
fn every_code_reads_its_tile_value() {
    for form in LINEAR {
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
        // Codes 0, 1022 and 1023: the source range's ends.
        let (lo, hi) = SOURCE_RANGE;
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

/// A padded 10-bit form reads every 16-bit word as `float(c)` of the code `c` in its top 10 bits
/// times the `f32` nearest `1 / 1023` (a `half` result that value narrowed to nearest even), blue
/// 0 and alpha 1, whatever the low 6 bits hold (every word; `RG10Unorm_X12_sRGB` reads its codes
/// so too); and writes `RoundEven(clamp(v, 0, 1) · 1023)` of the `f32` product (a NaN 0) in its
/// word's top 10 bits, the low 6 bits zero (a grid, and three `f32`s about every midpoint, each
/// lane). The product misses Metal's correctly rounded quotient at some codes.
#[test]
fn padded_forms_read_and_write_the_devices_codes() {
    let value = |w: u32| (w >> 6) as f32 * reciprocal(1023.0);
    let mut apart = 0;
    for form in [
        ExtendedRange::R10X6,
        ExtendedRange::Rg10X12,
        ExtendedRange::Rg10X12Srgb,
    ] {
        let two = form != ExtendedRange::R10X6;
        // Every code under four fillings of the low 6 bits, the other lanes' words other codes.
        let lanes: Vec<[u32; 4]> = (0..4096u32)
            .map(|i| {
                let w = (i & 1023) << 6 | [0, 0x3F, 0x15, 0x2A][(i >> 10) as usize];
                [w, w.wrapping_mul(40503) & 0xFFFF, 0xFFFF, 0xFFFF]
            })
            .collect();
        let floats = run(form, Op::Decode, &lanes);
        let halves = run(form, Op::DecodeHalf, &lanes);
        for ((l, f), h) in lanes.iter().zip(&floats).zip(&halves) {
            let green = if two { value(l[1]) } else { 0.0 };
            let want = [value(l[0]), green, 0.0, 1.0];
            assert_eq!(*f, want.map(f32::to_bits), "{form:?} words {l:x?}");
            let want = want.map(|v| to_half(f64::from(v), false).to_bits());
            assert_eq!(*h, want, "{form:?} half, words {l:x?}");
            apart += usize::from(value(l[0]) != (f64::from(l[0] >> 6) / 1023.0) as f32);
        }
        if form == ExtendedRange::Rg10X12Srgb {
            continue;
        }
        let code = |v: f32| {
            let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
            round_even(f64::from(v * 1023.0)) as u32
        };
        let mut values = vec![f32::NAN, -1.0, 2.0, f32::INFINITY, -0.0, 0.5];
        for c in 0..1023u32 {
            let mid = ((f64::from(c) + 0.5) / 1023.0) as f32;
            values.extend([
                mid,
                f32::from_bits(mid.to_bits() + 1),
                f32::from_bits(mid.to_bits() - 1),
            ]);
        }
        let lanes: Vec<[u32; 4]> = values
            .iter()
            .zip(values.iter().rev())
            .map(|(r, g)| [r.to_bits(), g.to_bits(), 0, 0])
            .collect();
        let got = run(form, Op::Encode, &lanes);
        for (l, g) in lanes.iter().zip(&got) {
            let (r, gr) = (f32::from_bits(l[0]), f32::from_bits(l[1]));
            let green = if two { code(gr) << 6 } else { 0 };
            assert_eq!(*g, [code(r) << 6, green, 0, 0], "{form:?} values {r}, {gr}");
        }
    }
    assert!(apart > 0, "the product never misses the quotient");
}

/// Metal's `float` of bfloat code `c`, as bits (`metal::format::ExtendedRange::R16Bfloat`): the
/// code shifted up 16, any NaN code the quiet NaN.
fn bfloat_read(c: u32) -> u32 {
    c << 16
}

/// The code of a kernel's write of the `f32` of bits `b` to a bfloat texture: its top 16 bits, a
/// NaN's with its quiet bit set (Metal writes `0x7fc0`).
fn bfloat_write(b: u32) -> u32 {
    match b & 0x7FFF_FFFF > 0x7F80_0000 {
        true => (b | 0x0040_0000) >> 16,
        false => b >> 16,
    }
}

/// The interpreter's device narrowing of the `float` of bits `b` to `half` (`OpQuantizeToF16`:
/// nearest even, a result below the smallest normal `half` its signed zero), a NaN kept as it is
/// (`narrow::round_texel`).
fn device_half(b: u32) -> u32 {
    let x = f32::from_bits(b);
    if x.is_nan() {
        return b;
    }
    let h = crate::float16::f32_to_f16_bits(x);
    let h = if h & 0x7C00 == 0 { h & 0x8000 } else { h };
    crate::float16::f16_bits_to_f32(h).to_bits()
}

/// Every code of each bfloat form in each channel reads its `float`, a NaN code's NaN unchanged
/// (`0x7f81` reads `0x7f810000`, where Metal reads `0x7fc00000`), and its `half` the device's
/// narrowing of it, the channels a form lacks 0 and alpha 1 whatever their words hold; a write
/// stores each value's top 16 bits, a NaN's quiet bit set (`0x7f800001` writes `0x7fc0`,
/// `0x7f810000` writes `0x7fc1`, where Metal writes `0x7fc0`): per code its bits and its tie's
/// neighbours, every eighth `half` widened, and the specials, the components a form lacks zero.
#[test]
fn bfloat_forms_read_and_write_their_bits() {
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
            assert_eq!(*h, want.map(device_half), "{form:?} half, words {l:x?}");
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
            0x7F81_0000,
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
/// then the bfloat code nearest, ties to even; a NaN stays a NaN code (`narrow::round`'s quiet
/// bit), where Metal stores `0x7fc0`. Every `float` around each code's tie and every `half` code
/// against Metal's digests with each NaN code read as Metal's, the narrower forms keeping their
/// lanes, and named answers.
#[test]
fn bfloat_attachments_store_metals_codes() {
    let digest = |got: &[[u32; 4]]| {
        assert!(got.iter().flatten().all(|&c| c <= 0xFFFF));
        let bytes: Vec<u8> = got
            .iter()
            .flatten()
            .map(|&c| match c & 0x7FFF > 0x7F80 {
                true => 0x7FC0,
                false => c,
            })
            .flat_map(|c| (c as u16).to_le_bytes())
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
            match want == 0x7FC0 {
                true => assert!(g[0] & 0x7FFF > 0x7F80, "{b:#x}: {:#x}", g[0]),
                false => assert_eq!(g[0], want, "{b:#x}, nearest even {nearest_even}"),
            }
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
