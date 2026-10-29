//! Metal's `atan2` and `fast::atan2` on the interpreter against Metal's recorded answers (Apple M3
//! Max, probe cases 6802ab13 and c11eb8cb, every lane of the sixteen operands below squared): zeros
//! of both signs, denormals, the smallest normal, ordinary values, the largest finite values,
//! infinities, and NaNs. The interpreter does not model `GLSL.std.450 Atan2` or `Atan`, so `atan2`
//! takes its generic value as an input and `fast::atan2` a stand-in `atan`; every lane a special
//! value decides must equal Metal's answer bit for bit (a NaN any NaN, its bits the device's), and
//! every other lane must be the generic value (or the stand-in's), so the special values read
//! neither. An `f32` denormal operand is a value, as the interpreter reads it, where Metal reads a
//! zero (`PrecisionNote::DeviceDenormals` off a device measured to read it as Metal does).

use super::*;

/// The operands of both cases on `float`.
const F32_OPERANDS: [u32; 16] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x0080_0000,
    0x3F80_0000,
    0xBF80_0000,
    0x4040_0000,
    0xBF40_0000,
    0x7F7F_FFFF,
    0xFF7F_FFFF,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC0_0000,
    0xFFC0_0000,
    0x7FC1_2345,
];

/// The operands of both cases on `half`.
const HALF_OPERANDS: [u16; 16] = [
    0x0000, 0x8000, 0x0001, 0x8001, 0x0400, 0x3C00, 0xBC00, 0x4200, 0xBA00, 0x7BFF, 0xFBFF, 0x7C00,
    0xFC00, 0x7E00, 0xFE00, 0x7E45,
];

/// Metal's `atan2(Y, X)` on `float` (precise; the same in both compiles), indexed `[y][x]`.
const ATAN2_F32: [[u32; 16]; 16] = [
    [
        0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x00000000, 0x00000000, 0x40490FDB,
        0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x80000000, 0xC0490FDB, 0x80000000, 0xC0490FDB, 0x80000000, 0x80000000, 0xC0490FDB,
        0x80000000, 0xC0490FDB, 0x80000000, 0xC0490FDB, 0x80000000, 0xC0490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x00000000, 0x00000000, 0x40490FDB,
        0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x80000000, 0xC0490FDB, 0x80000000, 0xC0490FDB, 0x80000000, 0x80000000, 0xC0490FDB,
        0x80000000, 0xC0490FDB, 0x80000000, 0xC0490FDB, 0x80000000, 0xC0490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3F490FDB, 0x00800000, 0x40490FDB,
        0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3F490FDB, 0x4016CBE4,
        0x3EA4BC7E, 0x400DB70D, 0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBF490FDB, 0xC016CBE4,
        0xBEA4BC7E, 0xC00DB70D, 0x00000000, 0xC0490FDB, 0x80000000, 0xC0490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3F9FE0BC, 0x3FF23EFA,
        0x3F490FDB, 0x3FE86B51, 0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBF24BC7D, 0xC01FE0BC,
        0xBE7ADBAF, 0xC016CBE4, 0x00000000, 0xC0490FDB, 0x80000000, 0xC0490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB,
        0x3FC90FDB, 0x3FC90FDB, 0x3F490FDB, 0x4016CBE4, 0x00000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB,
        0xBFC90FDB, 0xBFC90FDB, 0xBF490FDB, 0xC016CBE4, 0x80000000, 0xC0490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB,
        0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3F490FDB, 0x4016CBE4, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB,
        0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBF490FDB, 0xC016CBE4, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
];
/// Metal's `fast::atan2(Y, X)` from a call with every fast-math flag (case c11eb8cb).
const FAST_ATAN2_F32: [[u32; 16]; 16] = [
    [
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x00000000, 0x00000000, 0x40490FDB,
        0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x80000000, 0x80000000, 0x40490FDB,
        0x80000000, 0x40490FDB, 0x80000000, 0x40490FDB, 0x80000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x00000000, 0x00000000, 0x40490FDB,
        0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x80000000, 0x80000000, 0x40490FDB,
        0x80000000, 0x40490FDB, 0x80000000, 0x40490FDB, 0x80000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0x3F490FDA, 0x00000000, 0x40490FDB,
        0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0x3F490FDA, 0x4016CBE4,
        0x3EA4BC7D, 0x400DB70D, 0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0xBF490FDA, 0xC016CBE4,
        0xBEA4BC7D, 0xC00DB70D, 0x80000000, 0xC0490FDB, 0x80000000, 0xC0490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0x3F9FE0BC, 0x3FF23EFA,
        0x3F490FDA, 0x3FE86B51, 0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0xBF24BC7D, 0xC01FE0BC,
        0xBE7ADBB1, 0xC016CBE4, 0x80000000, 0xC0490FDB, 0x80000000, 0xC0490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB,
        0x3FC90FDB, 0x3FC90FDB, 0x3F490FDA, 0x4016CBE4, 0x00000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB,
        0xBFC90FDB, 0xBFC90FDB, 0xBF490FDA, 0xC016CBE4, 0x80000000, 0xC0490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB,
        0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB,
        0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
];
/// Metal's `fast::atan2(Y, X)` from a call with no fast-math flag (case 6802ab13).
const FAST_ATAN2_F32_NO_FLAGS: [[u32; 16]; 16] = [
    [
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x00000000, 0x00000000, 0x40490FDB,
        0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x80000000, 0x80000000, 0x40490FDB,
        0x80000000, 0x40490FDB, 0x80000000, 0x40490FDB, 0x80000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x00000000, 0x00000000, 0x40490FDB,
        0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x80000000, 0x80000000, 0x40490FDB,
        0x80000000, 0x40490FDB, 0x80000000, 0x40490FDB, 0x80000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0x3F490FDA, 0x00000000, 0x40490FDB,
        0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0x3F490FDA, 0x4016CBE4,
        0x3EA4BC7D, 0x400DB70D, 0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0xBF490FDA, 0xC016CBE4,
        0xBEA4BC7D, 0xC00DB70D, 0x80000000, 0xC0490FDB, 0x80000000, 0xC0490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0x3F9FE0BC, 0x3FF23EFA,
        0x3F490FDA, 0x3FE86B51, 0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0xBF24BC7D, 0xC01FE0BC,
        0xBE7ADBB1, 0xC016CBE4, 0x80000000, 0xC0490FDB, 0x80000000, 0xC0490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB,
        0x3FC90FDB, 0x3FC90FDB, 0x00000000, 0x40490FDB, 0x00000000, 0x40490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB,
        0xBFC90FDB, 0xBFC90FDB, 0x80000000, 0xC0490FDB, 0x80000000, 0xC0490FDB, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0x3FC90FDB, 0x3FC90FDB,
        0x3FC90FDB, 0x3FC90FDB, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0x3FC90FDB, 0xBFC90FDB, 0xBFC90FDB, 0xBFC90FDB,
        0xBFC90FDB, 0xBFC90FDB, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
    [
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
        0x7FC00000, 0x7FC00000,
    ],
];
/// Metal's `atan2(Y, X)` on `half` (the same in both compiles), indexed `[y][x]`.
const ATAN2_HALF: [[u16; 16]; 16] = [
    [
        0x0000, 0x4248, 0x0000, 0x4248, 0x0000, 0x0000, 0x4248, 0x0000, 0x4248, 0x0000, 0x4248,
        0x0000, 0x4248, 0x7E00, 0x7E00, 0x7E00,
    ],
    [
        0x8000, 0xC248, 0x8000, 0xC248, 0x8000, 0x8000, 0xC248, 0x8000, 0xC248, 0x8000, 0xC248,
        0x8000, 0xC248, 0x7E00, 0x7E00, 0x7E00,
    ],
    [
        0x3E48, 0x3E48, 0x3A48, 0x40B6, 0x1400, 0x0001, 0x4248, 0x0000, 0x4248, 0x0000, 0x4248,
        0x0000, 0x4248, 0x7E00, 0x7E00, 0x7E00,
    ],
    [
        0xBE48, 0xBE48, 0xBA48, 0xC0B6, 0x9400, 0x8001, 0xC248, 0x8000, 0xC248, 0x8000, 0xC248,
        0x8000, 0xC248, 0x7E00, 0x7E00, 0x7E00,
    ],
    [
        0x3E48, 0x3E48, 0x3E47, 0x3E49, 0x3A48, 0x0400, 0x4248, 0x0155, 0x4248, 0x0000, 0x4248,
        0x0000, 0x4248, 0x7E00, 0x7E00, 0x7E00,
    ],
    [
        0x3E48, 0x3E48, 0x3E48, 0x3E48, 0x3E48, 0x3A48, 0x40B6, 0x3526, 0x406E, 0x0100, 0x4248,
        0x0000, 0x4248, 0x7E00, 0x7E00, 0x7E00,
    ],
    [
        0xBE48, 0xBE48, 0xBE48, 0xBE48, 0xBE48, 0xBA48, 0xC0B6, 0xB526, 0xC06E, 0x8100, 0xC248,
        0x8000, 0xC248, 0x7E00, 0x7E00, 0x7E00,
    ],
    [
        0x3E48, 0x3E48, 0x3E48, 0x3E48, 0x3E48, 0x3CFF, 0x3F92, 0x3A48, 0x3F43, 0x0300, 0x4248,
        0x0000, 0x4248, 0x7E00, 0x7E00, 0x7E00,
    ],
    [
        0xBE48, 0xBE48, 0xBE48, 0xBE48, 0xBE48, 0xB926, 0xC0FF, 0xB3D7, 0xC0B6, 0x80C0, 0xC248,
        0x8000, 0xC248, 0x7E00, 0x7E00, 0x7E00,
    ],
    [
        0x3E48, 0x3E48, 0x3E48, 0x3E48, 0x3E48, 0x3E48, 0x3E49, 0x3E48, 0x3E49, 0x3A48, 0x40B6,
        0x0000, 0x4248, 0x7E00, 0x7E00, 0x7E00,
    ],
    [
        0xBE48, 0xBE48, 0xBE48, 0xBE48, 0xBE48, 0xBE48, 0xBE49, 0xBE48, 0xBE49, 0xBA48, 0xC0B6,
        0x8000, 0xC248, 0x7E00, 0x7E00, 0x7E00,
    ],
    [
        0x3E48, 0x3E48, 0x3E48, 0x3E48, 0x3E48, 0x3E48, 0x3E48, 0x3E48, 0x3E48, 0x3E48, 0x3E48,
        0x3A48, 0x40B6, 0x7E00, 0x7E00, 0x7E00,
    ],
    [
        0xBE48, 0xBE48, 0xBE48, 0xBE48, 0xBE48, 0xBE48, 0xBE48, 0xBE48, 0xBE48, 0xBE48, 0xBE48,
        0xBA48, 0xC0B6, 0x7E00, 0x7E00, 0x7E00,
    ],
    [
        0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00,
        0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00,
    ],
    [
        0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00,
        0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00,
    ],
    [
        0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00,
        0x7E00, 0x7E00, 0x7E00, 0x7E00, 0x7E00,
    ],
];

/// A value no `atan2` answer of either format equals: the generic input of a lane a special value
/// decides, so a special lane that read it would fail.
const MARKER_F32: u32 = 0x3E00_0000;
const MARKER_HALF: u16 = 0x3000;

/// Whether an `f32` operand is a zero, infinite, or NaN (its denormals are values).
fn f32_special(b: u32) -> bool {
    b & 0x7FFF_FFFF == 0 || b & 0x7F80_0000 == 0x7F80_0000
}

/// Whether an `f32` operand is a denormal, which Metal reads as a zero.
fn f32_denormal(b: u32) -> bool {
    b & 0x7F80_0000 == 0 && b & 0x007F_FFFF != 0
}

/// Whether a `half` operand is a zero, infinite, or NaN (its denormals are values).
fn half_special(b: u16) -> bool {
    b & 0x7FFF == 0 || b & 0x7C00 == 0x7C00
}

/// `atan2` (`math::atan2_special` around its generic input, then the format's rounding): a lane
/// with a zero, infinite, or NaN operand is Metal's answer whatever the generic value; any other
/// lane is the generic value, a denormal one included (the device's `Atan2`, where Metal answers
/// `+0`). The generic input is the host's correctly rounded `atan2` there and [`MARKER_F32`]
/// elsewhere. Metal's own generic answers are within one ulp of it (five `float` lanes differ by
/// one, the device's precision). The `float` lanes with a denormal operand, which Metal reads as a
/// zero, are the device's reading and left out.
#[test]
fn atan2_answers_as_c99_at_every_special_operand() {
    for bits in [32u8, 16] {
        let f = float(bits);
        let mut cases: Vec<Case> = Vec::new();
        let mut want = Vec::new();
        for j in 0..16 {
            for k in 0..16 {
                if bits == 32 && (f32_denormal(F32_OPERANDS[j]) || f32_denormal(F32_OPERANDS[k])) {
                    continue;
                }
                let (y, x, metal, special) = if bits == 32 {
                    let (y, x) = (F32_OPERANDS[j], F32_OPERANDS[k]);
                    let special = f32_special(y) || f32_special(x);
                    (
                        u64::from(y),
                        u64::from(x),
                        u64::from(ATAN2_F32[j][k]),
                        special,
                    )
                } else {
                    let (y, x) = (HALF_OPERANDS[j], HALF_OPERANDS[k]);
                    let special = half_special(y) || half_special(x);
                    (
                        u64::from(y),
                        u64::from(x),
                        u64::from(ATAN2_HALF[j][k]),
                        special,
                    )
                };
                let (generic, expect) = if special {
                    let marker = if bits == 32 {
                        u64::from(MARKER_F32)
                    } else {
                        u64::from(MARKER_HALF)
                    };
                    (marker, metal)
                } else {
                    let g = if bits == 32 {
                        let r = f64::from(f32v(y)).atan2(f64::from(f32v(x)));
                        u64::from((r as f32).to_bits())
                    } else {
                        let r = f16_to_f64(y as u16).atan2(f16_to_f64(x as u16));
                        u64::from(f64_to_f16(r))
                    };
                    (g, g)
                };
                cases.push(vec![vec![y], vec![x], vec![generic]]);
                want.push(((y, x), expect));
            }
        }
        check_accept(
            &format!("atan2 f{bits} against Metal"),
            &[f, f, f],
            f,
            &|fb, a| {
                let r = math::atan2_special(fb, f, a[0], a[1], a[2])?;
                crate::lower::arith::narrow::round(fb, f, r)
            },
            &cases,
            &|c| {
                let key = (arg(c, 0), arg(c, 1));
                want.iter().find(|(k, _)| *k == key).map(|(_, w)| vec![*w])
            },
            &|g, w| g == w || (is_nan(f, g) && is_nan(f, w)),
        );
    }
}

/// The value of an `f32` operand as Metal reads it: an `f32` denormal is a zero of its sign.
fn metal_read(b: u32) -> f32 {
    if b & 0x7F80_0000 == 0 {
        f32::from_bits(b & 0x8000_0000)
    } else {
        f32::from_bits(b)
    }
}

/// `fast::atan2(y, x)` with the stand-in `atan(q) = q + 1/8`, where the formula's special values do
/// not decide it (`None` where they do): the quotient (divided over the whole range, or `y` times
/// the device's reciprocal of `x`, which the interpreter keeps where it is a denormal), the
/// stand-in, and the turn. A denormal operand or quotient is a value.
fn fast_generic(y: u32, x: u32, by_reciprocal: bool) -> Option<u32> {
    if f32_special(y) || f32_special(x) {
        return None;
    }
    let (yv, xv) = (f32::from_bits(y), f32::from_bits(x));
    let q = if by_reciprocal {
        yv * (1.0 / xv)
    } else {
        (f64::from(yv) / f64::from(xv)) as f32
    };
    if !q.is_finite() {
        return None;
    }
    Some(turned(q + 0.125, yv, xv).to_bits())
}

/// `a` turned by `±π` where `x` is below zero, by `π` where `y` is at least zero.
fn turned(a: f32, y: f32, x: f32) -> f32 {
    let pi = std::f32::consts::PI;
    if x < 0.0 {
        a + if y >= 0.0 { pi } else { -pi }
    } else {
        a
    }
}

/// `fast::atan2(y, x)` of an infinite `y` and a finite `x` beyond `2^126` in a call without every
/// flag, on the interpreter: `y` times the device's reciprocal, which the interpreter keeps as a
/// denormal, is an infinite quotient, so the answer is `±π/2` (the quotient's sign) turned, where
/// Metal, which flushes the reciprocal, answers NaN. `None` for any other pair.
fn kept_reciprocal_infinite(y: u32, x: u32) -> Option<u32> {
    let (yv, xv) = (f32::from_bits(y), f32::from_bits(x));
    if !yv.is_infinite() || !xv.is_finite() || xv.abs() <= 2f32.powi(126) {
        return None;
    }
    let half_pi = std::f32::consts::FRAC_PI_2;
    let a = if (y ^ x) & 0x8000_0000 != 0 {
        -half_pi
    } else {
        half_pi
    };
    Some(turned(a, yv, xv).to_bits())
}

/// `fast::atan2` (`math::fast_atan2`) with a stand-in `atan(q) = q + 1/8` the interpreter runs:
/// every lane of the grid that the formula's special values decide is Metal's answer for the
/// call's flags (the quotient's sign, `0 / 0` and `inf / inf`, the turn's comparisons), a NaN any
/// NaN, and every other lane is the stand-in of the quotient, turned, a denormal quotient's
/// included: the correctly rounded quotient with every flag, and without it `y` times the device's
/// reciprocal. Beyond `2^126` that reciprocal is an `f32` denormal, which the interpreter keeps
/// and Metal flushes: there a finite `y` takes the stand-in of the product and an infinite one
/// `±π/2` turned, where Metal answers a signed zero and NaN (`PrecisionNote::DeviceDenormals` off
/// a device measured to flush as Metal does). Operands beyond the grid check the scaled division
/// near `2^126` and `FLT_MAX`, and the reciprocal there. The lanes with a denormal operand, which
/// Metal reads as a zero, are the device's reading and left out.
#[test]
fn fast_atan2_answers_as_metal_at_every_special_operand() {
    let f = float(32);
    let extra: [(f32, f32); 8] = [
        (3.0, 2f32.powi(126)),
        (3.0, 1.5 * 2f32.powi(126)),
        (-1e38, f32::MAX),
        (f32::MAX, -2f32.powi(127)),
        (2e10, -3e37),
        (1.0, 2f32.powi(125)),
        (-5.0, -7.0),
        (6.5, 0.25),
    ];
    for by_reciprocal in [false, true] {
        let table = if by_reciprocal {
            &FAST_ATAN2_F32_NO_FLAGS
        } else {
            &FAST_ATAN2_F32
        };
        let mut want: Vec<((u64, u64), u64)> = Vec::new();
        let (mut decided, mut kept) = (0, 0);
        for j in 0..16 {
            for k in 0..16 {
                let (y, x) = (F32_OPERANDS[j], F32_OPERANDS[k]);
                if f32_denormal(y) || f32_denormal(x) {
                    continue;
                }
                let device = if by_reciprocal {
                    kept_reciprocal_infinite(y, x)
                } else {
                    None
                };
                let w = match (fast_generic(y, x, by_reciprocal), device) {
                    (Some(g), _) => g,
                    (None, Some(d)) => {
                        kept += 1;
                        d
                    }
                    (None, None) => {
                        decided += 1;
                        table[j][k]
                    }
                };
                want.push(((u64::from(y), u64::from(x)), u64::from(w)));
            }
        }
        // The 196 lanes of operands that are not denormals, less those with a finite quotient of
        // finite nonzero operands (and, by the reciprocal, less the four of an infinite `y` over
        // `±FLT_MAX`, whose kept reciprocal makes the quotient infinite).
        assert_eq!(
            (decided, kept),
            if by_reciprocal {
                (196 - 45 - 4, 4)
            } else {
                (196 - 45, 0)
            },
            "by reciprocal {by_reciprocal}"
        );
        for (y, x) in extra {
            let (y, x) = (y.to_bits(), x.to_bits());
            let w = fast_generic(y, x, by_reciprocal).expect("a generic lane");
            want.push(((u64::from(y), u64::from(x)), u64::from(w)));
        }
        // Metal reads denormal operands as zeros: the answers of the grid's flushed operands are
        // the answers of the zeros of their sign.
        for j in 0..16 {
            for k in 0..16 {
                let zero_of = |b: u32| {
                    F32_OPERANDS
                        .iter()
                        .position(|o| *o == metal_read(b).to_bits())
                };
                if let (Some(zj), Some(zk)) = (zero_of(F32_OPERANDS[j]), zero_of(F32_OPERANDS[k])) {
                    assert_eq!(table[j][k], table[zj][zk], "lane ({j}, {k})");
                }
            }
        }
        let cases: Vec<Case> = want
            .iter()
            .map(|((y, x), _)| vec![vec![*y], vec![*x]])
            .collect();
        check_accept(
            &format!("fast atan2 f32 (by reciprocal {by_reciprocal}) on the interpreter"),
            &[f, f],
            f,
            &|fb, a| {
                let atan = |fb: &mut FunctionBuilder<'_>, q: Value| {
                    let eighth = crate::lower::arith::ops::fsplat(fb, f, 0.125)?;
                    fb.binary(BinOp::FAdd, q, eighth)
                };
                let divide =
                    |fb: &mut FunctionBuilder<'_>, n: Value, d: Value| fb.binary(BinOp::FDiv, n, d);
                math::fast_atan2(fb, f, a[0], a[1], by_reciprocal, &atan, &divide)
            },
            &cases,
            &|c| {
                let key = (arg(c, 0), arg(c, 1));
                want.iter().find(|(k, _)| *k == key).map(|(_, w)| vec![*w])
            },
            &|g, w| g == w || (is_nan(f, g) && is_nan(f, w)),
        );
    }
}

/// [`FloatFn::words`] bounds what `atan2` and `fast::atan2` emit on every form they take, and is
/// not more than twice it.
/// The module's growth over an identity kernel is charged in full, its constants and the
/// `GLSL.std.450` import included.
#[test]
fn special_value_functions_stay_within_their_word_estimate() {
    let module_words = |f: Form, body: Body<'_>| {
        let m = kernel(&[f, f], f, body).expect("kernel");
        crate::spirv::serialize::serialize(m)
            .expect("words")
            .words
            .len()
    };
    let fns = [
        FloatFn::Atan2,
        FloatFn::FastAtan2 {
            by_reciprocal: false,
        },
        FloatFn::FastAtan2 {
            by_reciprocal: true,
        },
    ];
    let mut out = Vec::new();
    for f in [float(32), vec(float(32), 4), float(16), vec(float(16), 4)] {
        let bare = module_words(f, &|_, x| Ok(x[0]));
        for g in fns {
            let used = module_words(f, &|fb, x| {
                math::float_fn(fb, g, f, Contraction::Allowed, x, &Lib::instructions())
            });
            let (emitted, bound) = (used - bare, g.words(f) as usize);
            if emitted > bound || 2 * emitted < bound {
                out.push(format!("{g:?} {f:?}: {emitted} words against {bound}"));
            }
        }
    }
    assert!(out.is_empty(), "{out:#?}");
}
