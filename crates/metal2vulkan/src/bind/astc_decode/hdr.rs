//! The HDR half of the ASTC decode: the HDR colour endpoint modes' endpoints, and the 16-bit
//! interpolation results' half-precision values, written here from the Khronos Data Format
//! Specification's ASTC chapter (its HDR endpoint decoding and its weight application in the HDR
//! profile).

/// A 12-bit HDR endpoint value's largest.
const TOP: i32 = 0xFFF;

/// The 12-bit value of an HDR mode's alpha where the mode has none: 1.0 once widened.
const ONE: i32 = 0x780;

fn clamp12(v: i32) -> i32 {
    v.clamp(0, TOP)
}

/// `v`'s low `bits` bits as a signed value.
fn sign_extend(v: i32, bits: u32) -> i32 {
    let shift = 32 - bits;
    (v << shift) >> shift
}

/// The two endpoints of an HDR colour endpoint mode `cem` (2, 3, 7, 11, 14 or 15) from its
/// unquantized values `v`: each channel a 12-bit logarithmic value, but mode 14's alpha, an 8-bit
/// LDR one; and whether alpha is logarithmic.
pub(super) fn endpoints(cem: u32, v: &[i32; 8]) -> ([[i32; 4]; 2], bool) {
    match cem {
        2 => {
            let (y0, y1) = match v[1] >= v[0] {
                true => (v[0] << 4, v[1] << 4),
                false => ((v[1] << 4) + 8, (v[0] << 4) - 8),
            };
            ([[y0, y0, y0, ONE], [y1, y1, y1, ONE]], true)
        }
        3 => {
            let (y0, d) = match v[0] & 0x80 != 0 {
                true => (
                    ((v[1] & 0xE0) << 4) | ((v[0] & 0x7F) << 2),
                    (v[1] & 0x1F) << 2,
                ),
                false => (
                    ((v[1] & 0xF0) << 4) | ((v[0] & 0x7F) << 1),
                    (v[1] & 0x0F) << 1,
                ),
            };
            let y1 = (y0 + d).min(TOP);
            ([[y0, y0, y0, ONE], [y1, y1, y1, ONE]], true)
        }
        7 => {
            let [e0, e1] = rgb_scale(v);
            ([with_alpha(e0, ONE), with_alpha(e1, ONE)], true)
        }
        11 => {
            let [e0, e1] = rgb(v);
            ([with_alpha(e0, ONE), with_alpha(e1, ONE)], true)
        }
        14 => {
            let [e0, e1] = rgb(v);
            ([with_alpha(e0, v[6]), with_alpha(e1, v[7])], false)
        }
        _ => {
            let [e0, e1] = rgb(v);
            let [a0, a1] = alpha(v[6], v[7]);
            ([with_alpha(e0, a0), with_alpha(e1, a1)], true)
        }
    }
}

fn with_alpha([r, g, b]: [i32; 3], a: i32) -> [i32; 4] {
    [r, g, b, a]
}

/// Mode 7: HDR RGB, a base and a scale.
fn rgb_scale(v: &[i32; 8]) -> [[i32; 3]; 2] {
    let modeval = ((v[0] & 0xC0) >> 6) | ((v[1] & 0x80) >> 5) | ((v[2] & 0x80) >> 4);
    let (major, mode) = if modeval & 0xC != 0xC {
        (modeval >> 2, modeval & 3)
    } else if modeval != 0xF {
        (modeval & 3, 4)
    } else {
        (0, 5)
    };
    let mut red = v[0] & 0x3F;
    let mut green = v[1] & 0x1F;
    let mut blue = v[2] & 0x1F;
    let mut scale = v[3] & 0x1F;
    let bit = |value: i32, at: u32| (value >> at) & 1;
    let x0 = bit(v[1], 6);
    let x1 = bit(v[1], 5);
    let x2 = bit(v[2], 6);
    let x3 = bit(v[2], 5);
    let x4 = bit(v[3], 7);
    let x5 = bit(v[3], 6);
    let x6 = bit(v[3], 5);
    // Each extra bit's place by the mode: the mode is one of six, `ohm` its one-hot mask.
    let ohm = 1 << mode;
    let place = |mask: i32, x: i32, at: u32| if ohm & mask != 0 { x << at } else { 0 };
    green |= place(0x30, x0, 6) | place(0x3A, x1, 5);
    blue |= place(0x30, x2, 6) | place(0x3A, x3, 5);
    scale |= place(0x3D, x6, 5) | place(0x2D, x5, 6) | place(0x04, x4, 7);
    red |= place(0x3B, x4, 6) | place(0x04, x3, 6) | place(0x10, x5, 7) | place(0x0F, x2, 7);
    red |= place(0x05, x1, 8) | place(0x0A, x0, 8) | place(0x05, x0, 9) | place(0x02, x6, 9);
    red |= place(0x01, x3, 10) | place(0x02, x5, 10);
    let shift = [1, 1, 2, 3, 4, 5][mode as usize];
    red <<= shift;
    green <<= shift;
    blue <<= shift;
    scale <<= shift;
    if mode != 5 {
        green = red - green;
        blue = red - blue;
    }
    match major {
        1 => (red, green) = (green, red),
        2 => (red, blue) = (blue, red),
        _ => {}
    }
    [
        [red - scale, green - scale, blue - scale].map(clamp12),
        [red, green, blue].map(clamp12),
    ]
}

/// Modes 11, 14 and 15's colour: HDR RGB, direct or a base and offsets.
fn rgb(v: &[i32; 8]) -> [[i32; 3]; 2] {
    let major = ((v[4] & 0x80) >> 7) | ((v[5] & 0x80) >> 6);
    if major == 3 {
        return [
            [v[0] << 4, v[2] << 4, (v[4] & 0x7F) << 5],
            [v[1] << 4, v[3] << 4, (v[5] & 0x7F) << 5],
        ];
    }
    let mode = ((v[1] & 0x80) >> 7) | ((v[2] & 0x80) >> 6) | ((v[3] & 0x80) >> 5);
    let mut a = v[0] | ((v[1] & 0x40) << 2);
    let mut b0 = v[2] & 0x3F;
    let mut b1 = v[3] & 0x3F;
    let mut c = v[1] & 0x3F;
    let d_bits = [7, 6, 7, 6, 5, 6, 5, 6][mode as usize];
    let mut d0 = sign_extend(v[4] & 0x7F, d_bits);
    let mut d1 = sign_extend(v[5] & 0x7F, d_bits);
    let bit = |value: i32, at: u32| (value >> at) & 1;
    let x0 = bit(v[2], 6);
    let x1 = bit(v[3], 6);
    let x2 = bit(v[4], 6);
    let x3 = bit(v[5], 6);
    let x4 = bit(v[4], 5);
    let x5 = bit(v[5], 5);
    // Each extra bit's place by the mode: the mode is one of eight, `ohm` its one-hot mask.
    let ohm = 1 << mode;
    let place = |mask: i32, x: i32, at: u32| if ohm & mask != 0 { x << at } else { 0 };
    a |= place(0xA4, x0, 9) | place(0x08, x2, 9) | place(0x50, x4, 9);
    a |= place(0x50, x5, 10) | place(0xA0, x1, 10) | place(0xC0, x2, 11);
    c |= place(0x04, x1, 6) | place(0xE8, x3, 6) | place(0x20, x2, 7);
    b0 |= place(0x5B, x0, 6) | place(0x12, x2, 7);
    b1 |= place(0x5B, x1, 6) | place(0x12, x3, 7);
    let shift = (mode >> 1) ^ 3;
    a <<= shift;
    b0 <<= shift;
    b1 <<= shift;
    c <<= shift;
    d0 <<= shift;
    d1 <<= shift;
    let mut e1 = [a, a - b0, a - b1].map(clamp12);
    let mut e0 = [a - c, a - b0 - c - d0, a - b1 - c - d1].map(clamp12);
    match major {
        1 => {
            e0.swap(0, 1);
            e1.swap(0, 1);
        }
        2 => {
            e0.swap(0, 2);
            e1.swap(0, 2);
        }
        _ => {}
    }
    [e0, e1]
}

/// Mode 15's alpha: HDR, direct or a base and an offset.
fn alpha(v6: i32, v7: i32) -> [i32; 2] {
    let mode = ((v6 >> 7) & 1) | ((v7 >> 6) & 2);
    let (mut a0, mut a1) = (v6 & 0x7F, v7 & 0x7F);
    if mode == 3 {
        return [a0 << 5, a1 << 5];
    }
    a0 |= (a1 << (mode + 1)) & 0x780;
    a1 &= 0x3F >> mode;
    a1 ^= 0x20 >> mode;
    a1 -= 0x20 >> mode;
    a0 <<= 4 - mode;
    a1 <<= 4 - mode;
    [a0, clamp12(a1 + a0)]
}

/// The half-precision bits of a logarithmic 16-bit interpolation result `c`: its top five bits
/// the exponent, its eleven low bits a mantissa mapped piecewise linearly onto the half's ten,
/// the largest finite half where that would be infinite or a NaN.
pub(super) fn lns_to_half(c: u32) -> u16 {
    let e = c >> 11;
    let m = c & 0x7FF;
    let mt = if m < 512 {
        3 * m
    } else if m >= 1536 {
        5 * m - 2048
    } else {
        4 * m - 512
    };
    ((e << 10) + (mt >> 3)).min(0x7BFF) as u16
}

/// The half-precision bits of an LDR channel's 16-bit interpolation result `c` in the HDR profile,
/// `c / 65535` as Metal reads it (the module docs).
pub(super) fn unorm16_to_half(c: u32) -> u16 {
    if c == 0xFFFF {
        return 0x3C00;
    }
    if c == 0 {
        return 0;
    }
    if c < 4 {
        // Below 2^-14: a subnormal, `c · 2^-16` in steps of `2^-24`.
        return (c << 8) as u16;
    }
    // `c / 2^16`, truncated to a half: the leading one's place is the exponent.
    let lead = 31 - c.leading_zeros();
    let exponent = lead - 1;
    let mantissa = (c << (31 - lead)) >> 21 & 0x3FF;
    (exponent << 10 | mantissa) as u16
}

/// The `f32` a half's bits `h` hold, exactly.
pub(super) fn half_value(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exponent = u32::from(h >> 10) & 0x1F;
    let mantissa = u32::from(h) & 0x3FF;
    let bits = match (exponent, mantissa) {
        (0, 0) => sign,
        (0, m) => {
            // A subnormal: normalize its mantissa.
            let shift = m.leading_zeros() - 21;
            sign | (113 - shift) << 23 | ((m << shift) & 0x3FF) << 13
        }
        (0x1F, 0) => sign | 0x7F80_0000,
        (0x1F, m) => sign | 0x7FC0_0000 | m << 13,
        (e, m) => sign | (e + 112) << 23 | m << 13,
    };
    f32::from_bits(bits)
}
