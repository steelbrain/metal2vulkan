//! Metal's sRGB conversions of `air.pack`/`air.unpack` `unorm4x8_srgb`, exact on every device: both
//! go through Metal's twelve-bit tables, which A1 read in full, rather than the MSL transfer
//! functions (whose `GLSL.std.450 Pow` has the device's precision and misses Metal's table at bytes
//! where the transfer lands near a half step).
//!
//! The two tables are one `Private` array of 256 `u32` with its constant initializer, declared
//! once per module that uses it ([`declare`]): word `b` holds the decode step of byte `b` in its
//! low half and the first encode index of byte `b` in its high half.
//!
//! - Decoding byte `b` loads its step `k` ([`step`]), and `math::unpack` answers `k / 4095`
//!   correctly rounded ([`div_exact`](crate::lower::arith::exact_division::div_exact)), as
//!   Metal does.
//! - Encoding `x` rounds the exact product `x * 4095` to the index `i`
//!   ([`round_exact_product_lanes`](crate::lower::arith::math::round_exact_product_lanes)), and
//!   [`search`] finds the byte in the thresholds: eight dependent loads, branch-free.
//!
//! The quotients run once per vector in `math::unpack`, with a per-lane scale; only the table
//! reads are per channel, since a `Private` array is indexed by a scalar.

use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, Value};
use crate::spirv::ir::{BinOp, ConstBits, Constant, MemAccess, StorageClass, Type};

/// The decode step of every byte `b`: Metal's `unpack_unorm4x8_srgb_to_float` answers `k / 4095`
/// (correctly rounded) for `k = SRGB_DECODE_STEPS[b]`, which is `round(decode(b / 255) * 4095)`
/// for the MSL transfer `decode(c) = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055)^2.4`
/// (every margin from a half step is at least `1.6e-4` step). Every color word of A1's
/// `srgb-decode-of-every-byte` (bcb3682b, 768 of 768) and the old engine's
/// `kernel_srgb_unpack_all_bytes` read these steps.
pub(crate) const SRGB_DECODE_STEPS: [u16; 256] = [
    0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 20, 21, 23, 25, 27, 29, 31, 33, 35, 37, 40,
    42, 45, 48, 50, 53, 56, 59, 62, 66, 69, 72, 76, 79, 83, 87, 91, 95, 99, 103, 107, 112, 116,
    121, 126, 131, 136, 141, 146, 151, 156, 162, 168, 173, 179, 185, 191, 197, 204, 210, 216, 223,
    230, 237, 244, 251, 258, 265, 273, 280, 288, 296, 304, 312, 320, 329, 337, 346, 354, 363, 372,
    381, 390, 400, 409, 419, 428, 438, 448, 458, 469, 479, 490, 500, 511, 522, 533, 544, 555, 567,
    578, 590, 602, 614, 626, 639, 651, 664, 676, 689, 702, 715, 728, 742, 755, 769, 783, 797, 811,
    825, 840, 854, 869, 884, 899, 914, 929, 945, 960, 976, 992, 1008, 1024, 1041, 1057, 1074, 1091,
    1108, 1125, 1142, 1159, 1177, 1195, 1213, 1231, 1249, 1267, 1286, 1304, 1323, 1342, 1361, 1381,
    1400, 1420, 1440, 1459, 1480, 1500, 1520, 1541, 1562, 1582, 1603, 1625, 1646, 1668, 1689, 1711,
    1733, 1755, 1778, 1800, 1823, 1846, 1869, 1892, 1916, 1939, 1963, 1987, 2011, 2035, 2059, 2084,
    2109, 2133, 2159, 2184, 2209, 2235, 2260, 2286, 2312, 2339, 2365, 2392, 2419, 2446, 2473, 2500,
    2527, 2555, 2583, 2611, 2639, 2668, 2696, 2725, 2754, 2783, 2812, 2841, 2871, 2901, 2931, 2961,
    2991, 3022, 3052, 3083, 3114, 3146, 3177, 3209, 3240, 3272, 3304, 3337, 3369, 3402, 3435, 3468,
    3501, 3535, 3568, 3602, 3636, 3670, 3705, 3739, 3774, 3809, 3844, 3879, 3915, 3950, 3986, 4022,
    4059, 4095,
];

/// The first encode index of every byte `b >= 1` (`0` for `b = 0`, never read): Metal's
/// `pack_float_to_srgb_unorm4x8` encodes a color channel `x` (clamped to `[0, 1]`, NaN to 0) as
/// the byte `b` of the largest `SRGB_ENCODE_FIRST[b] <= i`, where `i` is the exact product
/// `x * 4095` rounded to the nearest even integer: its twelve-bit encode table. A1's
/// `srgb-encode-of-every-half-and-a-2-14-float-grid` (8f7708ff) observed a byte for every one of
/// the 4096 indices, consistently, and these are its thresholds; the MSL transfer
/// `1.055 c^(1/2.4) - 0.055` misses four of them (indices 804, 1371, 2121, and 2569).
pub(crate) const SRGB_ENCODE_FIRST: [u16; 256] = [
    0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 21, 23, 24, 26, 28, 30, 32, 34, 37, 39,
    42, 44, 47, 49, 52, 55, 58, 61, 64, 68, 71, 74, 78, 82, 85, 89, 93, 97, 102, 106, 110, 115,
    119, 124, 129, 134, 139, 144, 149, 154, 160, 165, 171, 177, 183, 189, 195, 201, 207, 214, 220,
    227, 234, 241, 248, 255, 262, 270, 277, 285, 293, 300, 308, 317, 325, 333, 342, 350, 359, 368,
    377, 386, 395, 405, 414, 424, 434, 444, 454, 464, 474, 485, 495, 506, 517, 528, 539, 550, 562,
    573, 585, 597, 609, 621, 633, 645, 658, 670, 683, 696, 709, 722, 736, 749, 763, 777, 790, 804,
    819, 833, 848, 862, 877, 892, 907, 922, 938, 953, 969, 985, 1001, 1017, 1033, 1049, 1066, 1083,
    1100, 1117, 1134, 1151, 1169, 1186, 1204, 1222, 1240, 1259, 1277, 1296, 1314, 1333, 1352, 1371,
    1391, 1410, 1430, 1450, 1470, 1490, 1510, 1531, 1552, 1572, 1593, 1615, 1636, 1657, 1679, 1701,
    1723, 1745, 1767, 1790, 1812, 1835, 1858, 1881, 1904, 1928, 1951, 1975, 1999, 2023, 2048, 2072,
    2097, 2121, 2146, 2172, 2197, 2222, 2248, 2274, 2300, 2326, 2352, 2379, 2406, 2433, 2460, 2487,
    2514, 2542, 2569, 2597, 2626, 2654, 2682, 2711, 2740, 2769, 2798, 2827, 2857, 2886, 2916, 2946,
    2977, 3007, 3038, 3068, 3099, 3130, 3162, 3193, 3225, 3257, 3289, 3321, 3354, 3386, 3419, 3452,
    3485, 3518, 3552, 3586, 3620, 3654, 3688, 3722, 3757, 3792, 3827, 3862, 3897, 3933, 3969, 4005,
    4041, 4077,
];

/// An upper estimate of the SPIR-V words of the table (its 256 constants, the composite, and the
/// variable), charged to every use.
pub const TABLE_WORDS: u32 = 256 * 4 + 262 + 8;

/// Declare the module's sRGB table: a `Private` `u32[256]` initialized to the steps (low halves)
/// and thresholds (high halves).
pub fn declare(mb: &mut ModuleBuilder) -> BResult<GlobalId> {
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let len = mb.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(256),
    })?;
    let arr = mb.ty(Type::Array {
        elem: u32_,
        len,
        stride: None,
    })?;
    let mut parts = Vec::with_capacity(256);
    for (step, first) in SRGB_DECODE_STEPS.iter().zip(SRGB_ENCODE_FIRST) {
        parts.push(mb.constant(Constant::Scalar {
            ty: u32_,
            bits: ConstBits(u64::from(*step) | (u64::from(first) << 16)),
        })?);
    }
    let init = mb.constant(Constant::Composite {
        ty: arr,
        parts: parts.into(),
    })?;
    mb.plain_global(StorageClass::Private, arr, Some(init))
}

/// Word `i` (a `u32` scalar below 256) of `table`.
fn word(fb: &mut FunctionBuilder<'_>, table: GlobalId, i: Value) -> BResult<Value> {
    let base = fb.module().global_value(table)?;
    let p = fb.access_chain(base, &[i])?;
    fb.load(p, MemAccess::default())
}

/// The decode step `k` of sRGB byte `b` (a `u32` scalar below 256): the linear value is `k /
/// 4095`, correctly rounded.
pub fn step(fb: &mut FunctionBuilder<'_>, table: GlobalId, b: Value) -> BResult<Value> {
    let w = word(fb, table, b)?;
    let low = fb.u32_const(0xFFFF)?;
    let low = fb.module().const_value(low)?;
    fb.binary(BinOp::BitwiseAnd, w, low)
}

/// The sRGB byte of encode index `i` (a `u32` scalar below 4096: the exact product of the
/// clamped linear channel and 4095, rounded to nearest even): the largest byte whose first
/// index is at most `i`.
pub fn search(fb: &mut FunctionBuilder<'_>, table: GlobalId, i: Value) -> BResult<Value> {
    let k = |fb: &mut FunctionBuilder<'_>, v: u32| -> BResult<Value> {
        let c = fb.u32_const(v)?;
        fb.module().const_value(c)
    };
    let k16 = k(fb, 16)?;
    let mut b = None;
    for step in [128u32, 64, 32, 16, 8, 4, 2, 1] {
        let s = k(fb, step)?;
        // The first step's candidate is the constant 128.
        let (j, below) = match b {
            None => (s, k(fb, 0)?),
            Some(b) => (fb.binary(BinOp::IAdd, b, s)?, b),
        };
        let w = word(fb, table, j)?;
        let first = fb.binary(BinOp::ShiftRightLogical, w, k16)?;
        let at_least = fb.binary(BinOp::UGreaterThanEqual, i, first)?;
        b = Some(fb.select(at_least, j, below)?);
    }
    b.map_or_else(|| k(fb, 0), Ok)
}
