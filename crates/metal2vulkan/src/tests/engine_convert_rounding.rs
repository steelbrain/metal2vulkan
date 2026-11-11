//! `air.convert` with a rounding after its types and `.sat` after it, the forms Apple's OpenCL
//! compiler emits for every `convert_<type>[_sat][_<mode>]`, end to end.
//!
//! Each kernel converts a buffer of edge values by one intrinsic and stores the results. It
//! compiles on committed profiles with and without `shaderInt64` (the wide path), runs on the CPU
//! interpreter, and every result must be the exact IEEE answer of its rounding: integers rounded
//! to `float`, `double`-free `half`, or another integer, and `float` narrowed to `half`, computed
//! here in integer arithmetic. The interpreter's own conversions round to nearest even, so a
//! directed rounding the lowering left to them would show here.
//!
//! A conversion that rounds to nearest even and can round declares `RoundingModeRTE` for its
//! result's width (and the widths the device ties to it) where the device has them, and records
//! `NativeConversionRounding` where it does not; one that cannot round, and every directed one,
//! declares and records nothing of it.
//!
//! `llvm-as` assembles the modules; without it the tests skip visibly.

use crate::device::requirements::VkFeature;
use crate::device::{Device, DevicePolicy, DeviceProfile};
use crate::lower::arith::legal::{f16_to_f64, f64_to_f16};
use crate::lower::arith::narrow;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, compile_kept, description, link_and_validate, llvm_assemble_text, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::testing::committed_profiles;

/// A rounding a conversion names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Rte,
    Rtz,
    Rtp,
    Rtn,
}

const MODES: [(Mode, &str); 4] = [
    (Mode::Rte, "rte"),
    (Mode::Rtz, "rtz"),
    (Mode::Rtp, "rtp"),
    (Mode::Rtn, "rtn"),
];

/// A numeric type of the kernels: its `air` token, LLVM spelling and bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ty {
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
}

impl Ty {
    fn token(self) -> &'static str {
        match self {
            Ty::I8 => "i8",
            Ty::I16 => "i16",
            Ty::I32 => "i32",
            Ty::I64 => "i64",
            Ty::F16 => "f16",
            Ty::F32 => "f32",
        }
    }

    fn llvm(self) -> &'static str {
        match self {
            Ty::F16 => "half",
            Ty::F32 => "float",
            other => other.token(),
        }
    }

    fn bytes(self) -> usize {
        match self {
            Ty::I8 => 1,
            Ty::I16 | Ty::F16 => 2,
            Ty::I32 | Ty::F32 => 4,
            Ty::I64 => 8,
        }
    }

    fn bits(self) -> u32 {
        8 * self.bytes() as u32
    }
}

/// One side of a conversion: its kind letter and type.
#[derive(Clone, Copy, Debug)]
struct Side {
    kind: char,
    ty: Ty,
}

const fn s(ty: Ty) -> Side {
    Side { kind: 's', ty }
}
const fn u(ty: Ty) -> Side {
    Side { kind: 'u', ty }
}
const fn f(ty: Ty) -> Side {
    Side { kind: 'f', ty }
}

/// One intrinsic: `air.convert.<dst>.<src>.<mode>[.sat]`.
#[derive(Clone, Copy, Debug)]
struct Conv {
    dst: Side,
    src: Side,
    mode: Mode,
    sat: bool,
}

impl Conv {
    fn name(&self) -> String {
        let mode = MODES.iter().find(|(m, _)| *m == self.mode).unwrap().1;
        format!(
            "air.convert.{}.{}.{}.{}.{mode}{}",
            self.dst.kind,
            self.dst.ty.token(),
            self.src.kind,
            self.src.ty.token(),
            if self.sat { ".sat" } else { "" }
        )
    }
}

/// The edge values of each source type, as their bits.
fn inputs(ty: Ty) -> Vec<u64> {
    let words: Vec<u64> = match ty {
        Ty::I8 => vec![0, 1, 2, 0x7F, 0x80, 0x81, 0xFE, 0xFF, 0x40, 0xC0],
        Ty::I16 => vec![
            0, 1, 2047, 2048, 2049, 2050, 2051, 4097, 4098, 32767, 0x8000, 0x8001, 65504, 65519,
            65520, 65535, 0xF801, 0xF7FF, 0x1234, 0xFFFF,
        ],
        Ty::I32 => vec![
            0,
            1,
            0xFFFF_FFFF,
            16_777_216,
            16_777_217,
            16_777_218,
            16_777_219,
            0xFEFF_FFFF,
            0x7FFF_FFFF,
            0x8000_0000,
            0x7FFF_FFC0,
            0x7FFF_FF40,
            33_554_435,
            2049,
            2051,
            4097,
            65519,
            65520,
            65535,
            65536,
            0xFFFF_0010,
            100_000,
            0x8000_0001,
            0xFFFF_FF80,
            0x0100_007F,
            0xFFFE_0001,
            3,
            0x7F,
            0x80,
            0xFFF0,
            0x8001,
            0x1234_5679,
        ],
        Ty::I64 => vec![
            0,
            1,
            u64::MAX,
            (1 << 24) + 1,
            (1 << 53) + 1,
            (1 << 53) + 3,
            i64::MAX as u64,
            i64::MIN as u64,
            0x8000_0000_0000_0001,
            0x0000_0001_0000_0001,
            0x0000_0000_FFFF_FFFF,
            (1 << 40) + 3,
            (-(1i64 << 40) - 3) as u64,
            0x1234_5678_9ABC_DEF0,
            (-16_777_217i64) as u64,
            0xFFFF_FFFF_0000_0000,
            0x0000_8000_0000_0080,
            0x7FFF_FF80_0000_0000,
            0x00FF_FFFF_FFFF_FFFF,
            (1u64 << 63) + (1 << 39),
        ],
        Ty::F16 => vec![
            0x3C00, 0x3E00, 0x4100, 0xB800, 0x3800, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0x0001,
            0x8001, 0x77FF, 0x7800, 0x5640, 0xC100, 0x4248, 0x0000, 0x8000, 0x3555, 0xB555, 0x4D00,
            0xCD00,
        ],
        Ty::F32 => [
            1.0f32,
            1.000_976_6,
            1.000_488_3,
            1.0001,
            -1.0001,
            0.5,
            1.5,
            2.5,
            -0.5,
            -1.5,
            -2.5,
            0.499_999_97,
            2.7,
            -2.7,
            1e10,
            -1e10,
            2_147_483_520.0,
            2_147_483_648.0,
            -2_147_483_648.0,
            -2_147_483_904.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            f32::from_bits(0x0001_16C2),
            f32::from_bits(0x8001_16C2),
            255.5,
            256.0,
            127.5,
            -128.5,
            32767.5,
            65535.5,
            4_294_967_040.0,
            65504.0,
            65519.0,
            65520.0,
            65536.0,
            6.103_515_6e-5,
            6.0e-5,
            5.960_464_5e-8,
            2.980_232_2e-8,
            1e-8,
            -1e-8,
            3.3,
            -3.3,
            0.1,
            -0.1,
            2049.0,
            2051.0,
            0.333_333_34,
            9.223_372e18,
            -9.223_373e18,
        ]
        .iter()
        .map(|x| u64::from(x.to_bits()))
        .collect(),
    };
    words
}

/// The value of `bits` of type `ty` read with kind `kind`, as a real number (`None` for a NaN).
fn value(side: Side, bits: u64) -> Option<f64> {
    let b = side.ty.bits();
    Some(match (side.kind, side.ty) {
        ('f', Ty::F32) => {
            let x = f32::from_bits(bits as u32);
            if x.is_nan() {
                return None;
            }
            f64::from(x)
        }
        ('f', Ty::F16) => {
            let x = f16_to_f64(bits as u16);
            if x.is_nan() {
                return None;
            }
            x
        }
        ('s', _) => (((bits << (64 - b)) as i64) >> (64 - b)) as f64,
        _ => (bits & ones(b)) as f64,
    })
}

fn ones(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1 << bits) - 1
    }
}

/// The integer of `bits` read with `side`'s kind, exactly.
fn integer(side: Side, bits: u64) -> i128 {
    let b = side.ty.bits();
    if side.kind == 's' {
        i128::from(((bits << (64 - b)) as i64) >> (64 - b))
    } else {
        i128::from(bits & ones(b))
    }
}

/// Integer `v` rounded by `mode` to a float of `sig` significand bits, exactly, as an `f64`.
fn round_int(v: i128, sig: u32, mode: Mode) -> f64 {
    if v == 0 {
        return 0.0;
    }
    let neg = v < 0;
    let m = v.unsigned_abs();
    let msb = 127 - m.leading_zeros();
    let (q, sh) = if msb < sig {
        (m, 0)
    } else {
        let sh = msb - (sig - 1);
        let q = m >> sh;
        let r = m - (q << sh);
        let half = 1u128 << (sh - 1);
        let inc = match mode {
            Mode::Rte => r > half || (r == half && q & 1 == 1),
            Mode::Rtz => false,
            Mode::Rtp => !neg && r != 0,
            Mode::Rtn => neg && r != 0,
        };
        (q + u128::from(inc), sh)
    };
    let v = (q as f64) * 2f64.powi(sh as i32);
    if neg {
        -v
    } else {
        v
    }
}

/// The `half` bits of the real `x` (already a value of 11 significand bits, or past the range)
/// rounded by `mode` past 65504.
fn half_of_rounded(x: f64, mode: Mode) -> u16 {
    if x.abs() > 65504.0 {
        let away = match mode {
            Mode::Rte => true,
            Mode::Rtz => false,
            Mode::Rtp => x > 0.0,
            Mode::Rtn => x < 0.0,
        };
        let mag = if away { 0x7C00 } else { 0x7BFF };
        return if x < 0.0 { 0x8000 | mag } else { mag };
    }
    f64_to_f16(x)
}

/// `x` (finite or not, not a NaN) narrowed to `half` by `mode`: the nearest-even `half`, stepped
/// one `half` to the other side where it lies on the wrong side of `x`.
fn half_of_float(x: f64, mode: Mode) -> u16 {
    let h = f64_to_f16(x);
    let v = f16_to_f64(h);
    let step = |h: u16, up: bool| -> u16 {
        if h & 0x7FFF == 0 {
            if up {
                0x0001
            } else {
                0x8001
            }
        } else if (h & 0x8000 != 0) == up {
            h - 1
        } else {
            h + 1
        }
    };
    match mode {
        Mode::Rte => h,
        Mode::Rtz if v.abs() > x.abs() => h - 1,
        Mode::Rtz => h,
        Mode::Rtp if v < x => step(h, true),
        Mode::Rtn if v > x => step(h, false),
        Mode::Rtp | Mode::Rtn => h,
    }
}

/// The integer range of `side`.
fn range(side: Side) -> (i128, i128) {
    let b = side.ty.bits();
    if side.kind == 's' {
        (-(1i128 << (b - 1)), (1i128 << (b - 1)) - 1)
    } else {
        (0, (1i128 << b) - 1)
    }
}

/// The exact answer of `c` for the operand with bits `x`, as the destination's bits; `None` where
/// any answer is a NaN (checked as one).
fn expected(c: &Conv, x: u64) -> Option<u64> {
    let dst_bits = ones(c.dst.ty.bits());
    match (c.dst.kind, c.src.kind) {
        ('f', 'f') => {
            let v = value(c.src, x)?;
            match c.dst.ty {
                Ty::F16 => Some(u64::from(half_of_float(v, c.mode))),
                // A widening is exact.
                _ => Some(u64::from((v as f32).to_bits())),
            }
        }
        ('f', _) => {
            let v = integer(c.src, x);
            match c.dst.ty {
                Ty::F16 => Some(u64::from(half_of_rounded(round_int(v, 11, c.mode), c.mode))),
                _ => Some(u64::from((round_int(v, 24, c.mode) as f32).to_bits())),
            }
        }
        (_, 'f') => {
            let (lo, hi) = range(c.dst);
            let r = match value(c.src, x) {
                None => 0,
                Some(v) => {
                    let r = match c.mode {
                        Mode::Rte => v.round_ties_even(),
                        Mode::Rtz => v.trunc(),
                        Mode::Rtp => v.ceil(),
                        Mode::Rtn => v.floor(),
                    };
                    if r <= lo as f64 {
                        lo
                    } else if r >= hi as f64 {
                        hi
                    } else {
                        r as i128
                    }
                }
            };
            Some(r as u64 & dst_bits)
        }
        _ => {
            let v = integer(c.src, x);
            let (lo, hi) = range(c.dst);
            let r = if c.sat { v.clamp(lo, hi) } else { v };
            Some(r as u64 & dst_bits)
        }
    }
}

/// Every conversion the tests run: each rounding of the forms that can round, beside forms that
/// cannot, saturating and not.
fn conversions() -> Vec<Conv> {
    let mut out = Vec::new();
    let every = |dst: Side, src: Side, sat: bool, out: &mut Vec<Conv>| {
        for (mode, _) in MODES {
            out.push(Conv {
                dst,
                src,
                mode,
                sat,
            });
        }
    };
    every(f(Ty::F32), s(Ty::I32), false, &mut out);
    every(f(Ty::F32), u(Ty::I32), false, &mut out);
    every(f(Ty::F16), s(Ty::I32), false, &mut out);
    every(f(Ty::F16), u(Ty::I16), false, &mut out);
    every(f(Ty::F16), s(Ty::I16), false, &mut out);
    every(f(Ty::F32), s(Ty::I64), false, &mut out);
    every(f(Ty::F32), u(Ty::I64), false, &mut out);
    every(f(Ty::F16), s(Ty::I64), false, &mut out);
    every(f(Ty::F32), s(Ty::I16), false, &mut out);
    every(f(Ty::F16), u(Ty::I8), false, &mut out);
    every(f(Ty::F16), s(Ty::I8), false, &mut out);
    every(s(Ty::I32), f(Ty::F32), false, &mut out);
    every(s(Ty::I32), f(Ty::F32), true, &mut out);
    every(u(Ty::I32), f(Ty::F32), true, &mut out);
    every(u(Ty::I8), f(Ty::F32), true, &mut out);
    every(s(Ty::I8), f(Ty::F32), true, &mut out);
    every(s(Ty::I16), f(Ty::F16), true, &mut out);
    every(u(Ty::I16), f(Ty::F16), true, &mut out);
    every(s(Ty::I64), f(Ty::F32), true, &mut out);
    every(u(Ty::I64), f(Ty::F32), true, &mut out);
    every(f(Ty::F16), f(Ty::F32), false, &mut out);
    every(f(Ty::F32), f(Ty::F16), false, &mut out);
    let rtz = |dst, src, sat| Conv {
        dst,
        src,
        mode: Mode::Rtz,
        sat,
    };
    for (dst, src) in [
        (u(Ty::I8), s(Ty::I32)),
        (s(Ty::I8), s(Ty::I32)),
        (u(Ty::I8), u(Ty::I32)),
        (s(Ty::I8), u(Ty::I32)),
        (u(Ty::I16), s(Ty::I32)),
        (s(Ty::I16), u(Ty::I32)),
        (u(Ty::I32), s(Ty::I32)),
        (s(Ty::I32), u(Ty::I32)),
        (s(Ty::I32), s(Ty::I64)),
        (u(Ty::I32), s(Ty::I64)),
        (u(Ty::I64), s(Ty::I32)),
        (s(Ty::I64), u(Ty::I64)),
        (u(Ty::I64), s(Ty::I64)),
        (u(Ty::I32), s(Ty::I8)),
        (s(Ty::I32), u(Ty::I8)),
        (s(Ty::I16), s(Ty::I8)),
    ] {
        out.push(rtz(dst, src, true));
    }
    out.push(rtz(s(Ty::I16), s(Ty::I32), false));
    out.push(Conv {
        dst: u(Ty::I8),
        src: u(Ty::I32),
        mode: Mode::Rte,
        sat: false,
    });
    out
}

/// A kernel converting every input of `c`'s source type by it: buffer 0 the results, buffer 1
/// the operands, one call per operand at a constant index.
fn kernel(c: &Conv, name: &str, n: usize) -> String {
    let (src, dst) = (c.src.ty.llvm(), c.dst.ty.llvm());
    let mut body = String::new();
    for k in 0..n {
        body.push_str(&format!(
            "  %p{k} = getelementptr inbounds {src}, ptr addrspace(1) %1, i64 {k}\n  %x{k} = load {src}, ptr addrspace(1) %p{k}, align {sa}\n  %r{k} = call {dst} @{cn}({src} %x{k})\n  %q{k} = getelementptr inbounds {dst}, ptr addrspace(1) %0, i64 {k}\n  store {dst} %r{k}, ptr addrspace(1) %q{k}, align {da}\n",
            sa = c.src.ty.bytes(),
            da = c.dst.ty.bytes(),
            cn = c.name(),
        ));
    }
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1) {{\nentry:\n{body}  ret void\n}}\n\n\
declare {dst} @{cn}({src})\n\n\
!llvm.module.flags = !{{!0}}\n!air.kernel = !{{!20}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n\
!0 = !{{i32 7, !\"air.max_device_buffers\", i32 31}}\n!10 = !{{}}\n!11 = !{{!12, !13}}\n\
!12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 {db}, !\"air.arg_type_align_size\", i32 {db}, !\"air.arg_type_name\", !\"{dst}\", !\"air.arg_name\", !\"out\"}}\n\
!13 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 {sb}, !\"air.arg_type_align_size\", i32 {sb}, !\"air.arg_type_name\", !\"{src}\", !\"air.arg_name\", !\"in\"}}\n\
!17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n!20 = !{{ptr @{name}, !10, !11}}\n",
        cn = c.name(),
        db = c.dst.ty.bytes(),
        sb = c.src.ty.bytes(),
    )
}

fn assemble(name: &str, text: &str) -> Option<Vec<u8>> {
    match llvm_assemble_text(text) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

fn slot(index: u32) -> Option<ResourceSlot> {
    Some(ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    })
}

/// The committed `profile` description, with or without `shaderInt64`.
fn profile(stem: &str, int64: bool) -> DeviceProfile {
    let mut desc = description(stem).expect("description");
    if !int64 {
        assert!(desc.features.remove(&VkFeature::ShaderInt64), "{stem}");
    }
    let p = Device::new(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone();
    assert_eq!(p.fields().leaf.int64, int64, "{stem}");
    p
}

/// A half or float result's bits are a NaN.
fn is_nan(ty: Ty, bits: u64) -> bool {
    match ty {
        Ty::F16 => bits & 0x7C00 == 0x7C00 && bits & 0x3FF != 0,
        Ty::F32 => f32::from_bits(bits as u32).is_nan(),
        _ => false,
    }
}

/// Every conversion's results on the interpreter, through native 64-bit integers and through the
/// wide path, are the exact answers of its rounding, and the module passes L3.
#[test]
fn every_rounding_is_exact_on_the_interpreter() {
    let profiles = [
        ("lavapipe", profile("lavapipe", true)),
        ("moltenvk-m3 words", profile("moltenvk-m3", false)),
    ];
    let mut checked = 0usize;
    for c in conversions() {
        let name = c.name().replace('.', "_");
        let xs = inputs(c.src.ty);
        let Some(bc) = assemble(&name, &kernel(&c, &name, xs.len())) else {
            return;
        };
        for (stem, p) in &profiles {
            let what = format!("{} on {stem}", c.name());
            let (a, _, m) = compile_kept(&bc, &name, p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let input: Vec<u8> = xs
                .iter()
                .flat_map(|x| x.to_le_bytes()[..c.src.ty.bytes()].to_vec())
                .collect();
            let res = Resources {
                buffers: vec![
                    Buffer {
                        slot: slot(0),
                        address: None,
                        bytes: vec![0xAB; xs.len() * c.dst.ty.bytes()],
                    },
                    Buffer {
                        slot: slot(1),
                        address: None,
                        bytes: input,
                    },
                ],
                push: vec![0; 256],
                spec: Default::default(),
            };
            let out = run(&m, &Config::default(), res)
                .unwrap_or_else(|e| panic!("{what}: interpreter: {e}"))
                .buffers;
            let got = &out[0];
            let n = c.dst.ty.bytes();
            for (k, x) in xs.iter().enumerate() {
                let mut word = [0u8; 8];
                word[..n].copy_from_slice(&got[k * n..(k + 1) * n]);
                let g = u64::from_le_bytes(word);
                match expected(&c, *x) {
                    Some(want) if is_nan(c.dst.ty, want) => {
                        assert!(is_nan(c.dst.ty, g), "{what}: operand {x:#x}: {g:#x}")
                    }
                    Some(want) => {
                        assert_eq!(
                            g, want,
                            "{what}: operand {x:#x}: got {g:#x}, want {want:#x}"
                        )
                    }
                    None => assert!(is_nan(c.dst.ty, g), "{what}: operand {x:#x}: {g:#x}"),
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 0);
    eprintln!(
        "checked {checked} results of {} conversions on {} profiles",
        conversions().len(),
        profiles.len()
    );
}

/// The control of [`every_rounding_is_exact_on_the_interpreter`]: for every directed rounding of a
/// conversion that can round, some input's exact answer differs from the nearest-even one, which
/// the interpreter's own conversion gives; so a lowering that left the rounding to the device's
/// conversion fails that test. A conversion that cannot round (an integer to a wider float) answers
/// the same by every rounding.
#[test]
fn the_inputs_tell_each_directed_rounding_from_nearest_even() {
    for c in conversions() {
        let even = Conv {
            mode: Mode::Rte,
            ..c
        };
        let differs = inputs(c.src.ty)
            .iter()
            .filter(|x| expected(&c, **x) != expected(&even, **x))
            .count();
        let can_round = match (c.dst.kind, c.src.kind) {
            ('f', 'f') => c.dst.ty.bits() < c.src.ty.bits(),
            ('f', _) => {
                let magnitude = c.src.ty.bits() - u32::from(c.src.kind == 's');
                magnitude > if c.dst.ty == Ty::F16 { 11 } else { 24 }
            }
            (_, 'f') => true,
            _ => false,
        };
        if c.mode == Mode::Rte || !can_round {
            assert_eq!(differs, 0, "{}", c.name());
        } else {
            assert!(
                differs > 0,
                "{}: no input tells it from nearest even",
                c.name()
            );
        }
    }
}

/// The `RoundingModeRTE` widths of module `words`.
fn rounds_even_at(words: &[u32]) -> Vec<u32> {
    let mut widths = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if opcode == crate::spv::Op::ExecutionMode as u32
            && len == 4
            && words[i + 2] == crate::spv::ExecutionMode::RoundingModeRTE as u32
        {
            widths.push(words[i + 3]);
        }
        i += len;
    }
    widths.sort_unstable();
    widths
}

/// On every committed profile: an `int` to `float` by `.rte` declares `RoundingModeRTE` for 32
/// bits and the widths the device ties to it where the device has them all
/// (`narrow::rte_modes`), and records `NativeConversionRounding` where it does not; a `uchar` to
/// `float` by `.rte`, which cannot round, and an `int` to `float` by `.rtz`, computed exactly,
/// declare and record nothing of it.
#[test]
fn nearest_even_declares_the_mode_or_records_the_note() {
    let conv = |src, mode| Conv {
        dst: f(Ty::F32),
        src,
        mode,
        sat: false,
    };
    let rte = conv(s(Ty::I32), Mode::Rte);
    let exact = conv(u(Ty::I8), Mode::Rte);
    let directed = conv(s(Ty::I32), Mode::Rtz);
    let mut declared = 0;
    let mut noted = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        for (c, rounds) in [(rte, true), (exact, false), (directed, false)] {
            let name = c.name().replace('.', "_");
            let Some(bc) = assemble(&format!("{name}-{stem}"), &kernel(&c, &name, 4)) else {
                return;
            };
            let what = format!("{} on {stem}", c.name());
            let a = match compile(&bc, &name, &p) {
                Ok(a) => a,
                // A profile that cannot compile the kernel at all (no storage buffers of these
                // widths) says nothing about the rounding.
                Err(e) => {
                    eprintln!("{what}: not compiled: {e}");
                    continue;
                }
            };
            let noted_here = a
                .precision
                .contains(&PrecisionNote::NativeConversionRounding);
            let widths = rounds_even_at(&a.module.words);
            let modes = narrow::rte_modes(&p);
            if rounds {
                match &modes {
                    Some(want) => {
                        let want: Vec<u32> = want.iter().map(|w| u32::from(*w)).collect();
                        for w in &want {
                            assert!(widths.contains(w), "{what}: {widths:?} lacks {w}");
                        }
                        assert!(!noted_here, "{what}: noted where the mode is declared");
                        declared += 1;
                    }
                    None => {
                        assert!(noted_here, "{what}: no mode and no note");
                        noted += 1;
                    }
                }
            } else {
                assert!(!noted_here, "{what}: noted");
                assert!(widths.is_empty(), "{what}: declares {widths:?}");
            }
        }
        // The float controls of the profiles the venues run, read from their descriptions:
        // RADV's rounding modes are independent for 32 bits only, ANV's not at all (so 32 brings
        // 16 and 64), lavapipe's fully; MoltenVK has no 64-bit RTE and no independence, and the
        // floor states no float controls.
        let known: Option<Option<Vec<u8>>> = match stem.as_str() {
            "lavapipe" | "amd-780m-radv" => Some(Some(vec![32])),
            "intel-irisxe-anv" => Some(Some(vec![16, 32, 64])),
            "moltenvk-m3" | "vk12-floor" => Some(None),
            _ => None,
        };
        if let Some(want) = known {
            assert_eq!(narrow::rte_modes(&p), want, "{stem}");
        }
    }
    assert!(
        declared > 0 && noted > 0,
        "declared {declared}, noted {noted}"
    );
    eprintln!("RoundingModeRTE declared on {declared} profiles, noted on {noted}");
}
