//! Fused multiply-add: every route against one rounding of the exact `a * b + c`, on the
//! interpreter.
//!
//! The `float` reference is the host's `f32::mul_add` (one rounding, denormals kept); the `half`
//! reference rounds the exact sum, formed as an integer multiple of `2^-48`, once to nearest
//! even. Beside random and special operands, the inputs hold the triples where a shortcut
//! rounds twice: the exact sum within a `double` ulp of a `float` midpoint (which a `double` sum
//! rounded to `float` gets wrong without round-to-odd), with the addend dominant and with the
//! product dominant, and the `half` triples whose `float` sum lands on a `half` midpoint.

use super::*;
use crate::lower::arith::fma::{self, Fma, Route};
use std::collections::BTreeSet;

fn floats(lanes: u8) -> Form {
    vec(float(32), lanes)
}

/// `f32` bits of sign `s`, unbiased exponent `e` (a normal), and significand field `m`.
fn mk32(s: u64, e: i32, m: u64) -> u64 {
    (s << 31) | (((e + 127) as u64) << 23) | m
}

/// `half` bits of sign `s`, unbiased exponent `e` (a normal), and significand field `m`.
fn mk16(s: u64, e: i32, m: u64) -> u64 {
    (s << 15) | (((e + 15) as u64) << 10) | m
}

fn fma32(a: u64, b: u64, c: u64) -> u64 {
    u64::from(f32v(a).mul_add(f32v(b), f32v(c)).to_bits())
}

/// Whether a `double` sum rounded to `float` differs from the fused result (the hazard the
/// `double` route's round-to-odd removes).
fn double_rounds_wrong(a: u64, b: u64, c: u64) -> bool {
    let s = f64::from(f32v(a)) * f64::from(f32v(b)) + f64::from(f32v(c));
    u64::from((s as f32).to_bits()) != fma32(a, b, c)
}

/// The `float` triples: specials and every pairing of edge values, random bit patterns, nearby
/// normals, cancellations, denormal results, products past the largest float that the addend
/// brings back, and the `double` double-rounding hazards.
fn float_triples() -> Vec<[u64; 3]> {
    let edges: [u64; 19] = [
        0,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x0040_0000,
        0x007F_FFFF,
        0x0080_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x3F80_0001,
        0x3F7F_FFFF,
        0x4040_0000,
        0x7F7F_FFFF,
        0xFF7F_FFFF,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_0000,
        0x5F00_0000,
        0x1F00_0000,
    ];
    let mut out = Vec::new();
    for a in edges {
        for b in edges {
            for c in edges {
                out.push([a, b, c]);
            }
        }
    }
    let mut rng = Rng(0xF3A1_0001);
    for _ in 0..1500 {
        let r = rng.next();
        out.push([r & 0xFFFF_FFFF, r >> 32, rng.next() & 0xFFFF_FFFF]);
    }
    let normal = |rng: &mut Rng, lo: i32, span: u64| {
        let r = rng.next();
        mk32(r & 1, lo + ((r >> 1) % span) as i32, (r >> 16) & 0x7F_FFFF)
    };
    for _ in 0..1500 {
        out.push([
            normal(&mut rng, -20, 40),
            normal(&mut rng, -20, 40),
            normal(&mut rng, -30, 60),
        ]);
    }
    // Cancellations: `c` within a few ulps of `-RN(a * b)`, or some binades below it.
    for i in 0..1000u64 {
        let (a, b) = (normal(&mut rng, -60, 120), normal(&mut rng, -60, 120));
        let p = u64::from((f32v(a) * f32v(b)).to_bits());
        if (p & 0x7FFF_FFFF) < 0x0080_0000 || (p & 0x7F80_0000) == 0x7F80_0000 {
            continue;
        }
        let k = (rng.next() % 7) as i64 - 3;
        let mag = ((p & 0x7FFF_FFFF) as i64 + k) as u64;
        let c = if i % 2 == 0 {
            mag | ((p & 0x8000_0000) ^ 0x8000_0000)
        } else {
            let e = ((p >> 23) & 0xFF) as i32 - 127 - 1 - (rng.next() % 30) as i32;
            normal(&mut rng, e, 1)
        };
        out.push([a, b, c]);
    }
    // Denormal products and results.
    for _ in 0..1000 {
        let a = normal(&mut rng, -80, 20);
        let b = normal(&mut rng, -80, 20);
        let r = rng.next();
        let c = if r.is_multiple_of(3) {
            (r >> 2) & 0x807F_FFFF
        } else {
            normal(&mut rng, -130, 8)
        };
        out.push([a, b, c]);
    }
    // Products past the largest float that the addend brings back.
    for _ in 0..200 {
        let ea = 64 + (rng.next() % 64) as i32;
        let a = normal(&mut rng, ea, 1);
        let b = normal(&mut rng, 128 - ea, 1);
        let c = 0xFF7F_FFFF ^ (a & 0x8000_0000) ^ (b & 0x8000_0000);
        out.push([a, b, c]);
    }
    // The addend dominant: P = 2^(ec-24) (1 - 2^-46 u^2) just below half an ulp of an odd `c`,
    // so the `double` sum is the `float` midpoint, which ties away from `c`.
    let mut n = 0;
    while n < 200 {
        let u = 1 + rng.next() % 360;
        let ec = (rng.next() % 190) as i32 - 90;
        let c = mk32(rng.next() & 1, ec, (rng.next() & 0x7F_FFFF) | 1);
        let ka = (rng.next() % 81) as i32 - 40;
        let kb = ec - 24 - ka;
        if !(-126..=127).contains(&ka) || !(-125..=127).contains(&kb) {
            continue;
        }
        let a = mk32(rng.next() & 1, ka, u);
        let b = mk32(0, kb - 1, (1 << 23) - 2 * u);
        if double_rounds_wrong(a, b, c) {
            out.push([a, b, c]);
            n += 1;
        }
    }
    // The product dominant: its bits past the `float` midpoint bit are few, and `c` cancels
    // them but for a residue far below an ulp of the `double` sum.
    let mut n = 0;
    let mut tries = 0;
    while n < 200 && tries < 4_000_000 {
        tries += 1;
        let ma = (rng.next() & 0x7F_FFFF) | 0x80_0000;
        let mb = (rng.next() & 0x7F_FFFF) | 0x80_0000;
        let p = ma * mb;
        let top = 64 - p.leading_zeros() as i32;
        let mid = top - 25;
        if (p >> mid) & 1 == 0 {
            continue;
        }
        let low = p & ((1 << mid) - 1);
        let lowlen = 64 - low.leading_zeros() as i32;
        if low == 0 || lowlen > 17 {
            continue;
        }
        let shift = lowlen + 7 + (rng.next() % (18 - lowlen as u64).max(1)) as i32;
        if shift > 24 {
            continue;
        }
        let tiny = lowlen - shift;
        let toward = if rng.next() & 1 == 0 { 1i64 } else { -1 };
        let cint = -((low << -tiny) as i64) + toward;
        let cmag = cint.unsigned_abs();
        if 64 - cmag.leading_zeros() > 24 {
            continue;
        }
        let ea = (rng.next() % 101) as i32 - 50;
        let eb = (rng.next() % 101) as i32 - 50;
        let cexp = tiny + ea + eb - 46;
        let cv = cint as f64 * 2f64.powi(cexp);
        let c = u64::from((cv as f32).to_bits());
        if f64::from(f32v(c)) != cv {
            continue;
        }
        let a = mk32(0, ea, ma & 0x7F_FFFF);
        let b = mk32(0, eb, mb & 0x7F_FFFF);
        if double_rounds_wrong(a, b, c) {
            out.push([a, b, c]);
            n += 1;
        }
    }
    assert!(n >= 100, "only {n} product-dominant hazards");
    // The sticky bit decides: the product is exactly a `float` tie (`(1 + 2^-12)^2 = 1 + 2^-11 +
    // 2^-24` above an even last bit, `(1 + 2^-23) * 1.5` above an odd one) and an addend of
    // the sign that breaks the tie lies below the integer route's 64 bits; scaled by `2^k` and
    // negated.
    for (a, b, c_sign) in [
        (0x3F80_0800u64, 0x3F80_0800u64, 0u64),
        (0x3F80_0001, 0x3FC0_0000, 0x8000_0000),
    ] {
        for k in [-60i64, -20, 0, 20, 60] {
            for t in [-100i64, -70, -50, -42, -40, -39] {
                let ae = ((a >> 23) & 0xFF) as i64 + k;
                let ce = 127 + t + k;
                if !(1..=254).contains(&ae) || !(1..=254).contains(&ce) {
                    continue;
                }
                let a_k = (a & 0x807F_FFFF) | ((ae as u64) << 23);
                let c_k = c_sign | ((ce as u64) << 23);
                for neg in [0u64, 0x8000_0000] {
                    out.push([a_k ^ neg, b, c_k ^ neg]);
                }
            }
        }
    }
    out
}

/// The cases of `triples` on `lanes` lanes: lane `l` of case `i` is triple `i + l`.
fn lane_cases(triples: &[[u64; 3]], lanes: u8) -> Vec<Case> {
    let n = usize::from(lanes);
    (0..triples.len())
        .step_by(n)
        .map(|i| {
            (0..3)
                .map(|k| {
                    (0..n)
                        .map(|l| triples[(i + l) % triples.len()][k])
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// Bits equal, or both NaN.
fn same(f: Form) -> impl Fn(u64, u64) -> bool {
    move |g, w| g == w || (is_nan(f, g) && is_nan(f, w))
}

/// Run `fma` over `cases` and compare with `reference` per lane: bit for bit when `exact`, else
/// any NaN for a NaN.
fn check_fma(
    what: &str,
    fma: Fma,
    cases: &[Case],
    reference: &dyn Fn(u64, u64, u64) -> u64,
    exact: bool,
) {
    let f = fma.form;
    check_accept(
        what,
        &[f, f, f],
        f,
        &|fb, x| fma::emit(fb, fma, x),
        cases,
        &|c| {
            Some(
                (0..c[0].len())
                    .map(|l| reference(c[0][l], c[1][l], c[2][l]))
                    .collect(),
            )
        },
        &|g, w| if exact { g == w } else { same(f)(g, w) },
    );
}

/// `OpFmaKHR` and the device's `Fma` round the exact `a * b + c` once on the interpreter (whose
/// `Fma` is IEEE's), on scalars and vectors, specials, signed zeros, denormals, overflow, and the
/// double-rounding hazards included: the module adds nothing to the device's instruction.
#[test]
fn float_fused_multiply_add_is_the_device_s_instruction() {
    let triples = float_triples();
    for route in [Route::Khr, Route::Native] {
        for lanes in [1u8, 3] {
            let fma = Fma {
                form: floats(lanes),
                route,
            };
            check_fma(
                &format!("fma {route:?} x{lanes}"),
                fma,
                &lane_cases(&triples, lanes),
                &fma32,
                false,
            );
        }
    }
}

/// The blends' exact `float` multiply-add (`fma::float_fma_exact`, of a `float` output's blend into
/// an extended-range or 16-bit float attachment) rounds the exact `a * b + c` once on operands below
/// 2 in magnitude whose parts do not underflow, on one lane and four, and its zero is IEEE's: `-0`
/// for a negative zero product and a `-0` addend (`-0 · 1 + -0`, the zero alpha Metal stores for a
/// `float4` output's blend over a `-0` texel), `+0` for opposite zeros and for a product the addend
/// cancels. The emulation's own sum loses the first (`th + v` adds a `+0` to a `-0`). An infinite
/// operand's infinite result is that infinity, where the emulation's own steps give a NaN (a
/// `float4` output's infinity over a texel, which Metal stores as the `half` infinity).
#[test]
fn the_blends_float_multiply_add_rounds_once_and_signs_its_zeros() {
    fn operand(rng: &mut Rng) -> u64 {
        let r = rng.next();
        mk32(r & 1, ((r >> 1) % 15) as i32 - 14, (r >> 8) & 0x7F_FFFF)
    }
    let edges: [u64; 8] = [
        0,
        0x8000_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x3F00_0000,
        0xBF00_0000,
        0x3FFF_FFFF,
        0xBFFF_FFFF,
    ];
    let mut triples = Vec::new();
    for a in edges {
        for b in edges {
            for c in edges {
                triples.push([a, b, c]);
            }
        }
    }
    // Infinite operands whose results are infinite.
    triples.extend([
        [0x7F80_0000, 0x3F80_0000, 0x3F00_0000],
        [0xFF80_0000, 0x3F00_0000, 0xBF80_0000],
        [0x3F80_0000, 0x3F00_0000, 0x7F80_0000],
        [0xBFFF_FFFF, 0x3F80_0000, 0xFF80_0000],
        [0x7F80_0000, 0x3F80_0000, 0x7F80_0000],
    ]);
    let mut rng = Rng(0xB1E0_F32A);
    for _ in 0..2000 {
        triples.push([operand(&mut rng), operand(&mut rng), operand(&mut rng)]);
    }
    // Products the addend cancels: factors of 12 significand bits, whose product is a `float`.
    for _ in 0..300 {
        let (a, b) = (operand(&mut rng) & !0xFFF, operand(&mut rng) & !0xFFF);
        let p = f32v(a) * f32v(b);
        triples.push([a, b, u64::from((-p).to_bits())]);
    }
    assert!(triples.contains(&[0x8000_0000, 0x3F80_0000, 0x8000_0000]));
    for lanes in [1u8, 4] {
        let f = floats(lanes);
        check(
            &format!("the blends' float fma x{lanes}"),
            &[f, f, f],
            f,
            &|fb, x| fma::float_fma_exact(fb, f, x[0], x[1], x[2]),
            &lane_cases(&triples, lanes),
            &|c| {
                Some(
                    (0..c[0].len())
                        .map(|l| fma32(c[0][l], c[1][l], c[2][l]))
                        .collect(),
                )
            },
            false,
        );
    }
}

/// The triples reach the hazards a shortcut gets wrong: the hazard families hold triples a
/// `double` sum rounded to `float` gets wrong, and the cancellations triples where the unfused
/// result differs from the fused one.
#[test]
fn the_float_triples_reach_the_hazards() {
    let t = float_triples();
    let hazards = t
        .iter()
        .filter(|[a, b, c]| double_rounds_wrong(*a, *b, *c))
        .count();
    let unfused = t
        .iter()
        .filter(|[a, b, c]| {
            let u = u64::from((f32v(*a) * f32v(*b) + f32v(*c)).to_bits());
            let f = fma32(*a, *b, *c);
            u != f && !(is_nan(float(32), u) && is_nan(float(32), f))
        })
        .count();
    assert!(hazards >= 300, "{hazards} double-rounding hazards");
    assert!(unfused >= 1000, "{unfused} triples where unfused differs");
}

/// The exact `a * b + c` of `half` bits rounded once to `half` (nearest even): the operands
/// scaled by `2^24` are integers, so the sum is an integer multiple of `2^-48`.
fn fma16(a: u64, b: u64, c: u64) -> u64 {
    let (x, y, z) = (
        f16_to_f64(a as u16),
        f16_to_f64(b as u16),
        f16_to_f64(c as u16),
    );
    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return u64::from(f64_to_f16(x * y + z));
    }
    let scale = 2f64.powi(24);
    let (xi, yi, zi) = (
        (x * scale) as i128,
        (y * scale) as i128,
        (z * scale) as i128,
    );
    let n = xi * yi + (zi << 24);
    if n == 0 {
        // IEEE: the sum of two zeros of one sign keeps it; any other exact zero is `+0`.
        let p_neg = x.is_sign_negative() != y.is_sign_negative();
        let both = xi * yi == 0 && zi == 0 && p_neg && z.is_sign_negative();
        return if both { 0x8000 } else { 0 };
    }
    let m = n.unsigned_abs();
    let msb = 127 - m.leading_zeros() as i32;
    // Keep 11 bits, or the `2^-24` bit of a denormal (the sum's bit 24).
    let drop = (msb - 10).max(24) as u32;
    let q = m >> drop;
    let rem = m & ((1u128 << drop) - 1);
    let half = 1u128 << (drop - 1);
    let q = if rem > half || (rem == half && q & 1 == 1) {
        q + 1
    } else {
        q
    };
    let bits = if msb > 34 {
        ((((msb - 34) as u128) << 10) + q).min(0x7C00)
    } else {
        q
    };
    (bits as u64) | if n < 0 { 0x8000 } else { 0 }
}

/// The `half` triples: specials and edge pairings, random patterns, the triples whose `float`
/// sum lands on a `half` midpoint (`c` odd, `P = 2^(ec-11) (1 - 2^-20 u^2)` just inside half an
/// ulp of `c`), and products near `-c`.
fn half_triples() -> Vec<[u64; 3]> {
    let edges: [u64; 16] = [
        0, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x3C01, 0x3BFF, 0x7BFF, 0xFBFF,
        0x7C00, 0xFC00, 0x7E00, 0x4200,
    ];
    let mut out = Vec::new();
    for a in edges {
        for b in edges {
            for c in edges {
                out.push([a, b, c]);
            }
        }
    }
    let mut rng = Rng(0xF3A1_0016);
    for _ in 0..1500 {
        let r = rng.next();
        out.push([r & 0xFFFF, (r >> 16) & 0xFFFF, (r >> 32) & 0xFFFF]);
    }
    let mut n = 0;
    while n < 300 {
        let u = 1 + rng.next() % 11;
        let ec = (rng.next() % 28) as i32 - 12;
        let c = mk16(rng.next() & 1, ec, (rng.next() & 0x3FF) | 1);
        let ka = (rng.next() % 28) as i32 - 13;
        let kb = ec - 11 - ka;
        if !(-13..=15).contains(&ka) || !(-13..=15).contains(&kb) {
            continue;
        }
        let a = mk16(rng.next() & 1, ka, u);
        let b = mk16(0, kb - 1, 1024 - 2 * u);
        let s = f16_to_f64(a as u16) * f16_to_f64(b as u16) + f16_to_f64(c as u16);
        let twice = u64::from(f64_to_f16(f64::from(s as f32)));
        if twice != fma16(a, b, c) {
            out.push([a, b, c]);
            n += 1;
        }
    }
    for _ in 0..600 {
        let r = rng.next();
        let a = mk16(r & 1, ((r >> 1) % 15) as i32 - 7, (r >> 8) & 0x3FF);
        let b = mk16(
            (r >> 20) & 1,
            ((r >> 21) % 15) as i32 - 7,
            (r >> 28) & 0x3FF,
        );
        let p = f64_to_f16(f16_to_f64(a as u16) * f16_to_f64(b as u16));
        let k = (rng.next() % 7) as i64 - 3;
        let c = ((i64::from(p & 0x7FFF) + k).clamp(0, 0x7BFF) as u64)
            | (u64::from(p & 0x8000) ^ 0x8000);
        out.push([a, b, c]);
    }
    out
}

/// A `half` fused multiply-add on 16-bit floats, by the device's `Fma` ([`Route::F16Native`]),
/// rounds once on the interpreter, whose `Fma` rounds once (a device's may not, noted): the operands
/// convert to 16 bits exactly and the result widens exactly. The device's `float` `Fma` rounded to `half`
/// ([`Route::Native`]) rounds the sum twice, so it is within one `half` unit in the last place.
#[test]
fn half_fused_multiply_add_rounds_on_its_route() {
    let triples = half_triples();
    for route in [Route::F16Native] {
        for lanes in [1u8, 2] {
            let fma = Fma {
                form: vec(float(16), lanes),
                route,
            };
            check_fma(
                &format!("fma half {route:?} x{lanes}"),
                fma,
                &lane_cases(&triples, lanes),
                &fma16,
                false,
            );
        }
    }
    // Within one unit in the last place: the `float` sum's rounding may move a `half` tie.
    let near = |g: u64, w: u64| {
        same(float(16))(g, w) || ((g ^ w) & 0x8000 == 0 && (g & 0x7FFF).abs_diff(w & 0x7FFF) <= 1)
    };
    for lanes in [1u8, 2] {
        let f = vec(float(16), lanes);
        let fma = Fma {
            form: f,
            route: Route::Native,
        };
        check_accept(
            &format!("fma half native x{lanes}"),
            &[f, f, f],
            f,
            &|fb, x| fma::emit(fb, fma, x),
            &lane_cases(&triples, lanes),
            &|c| {
                Some(
                    (0..c[0].len())
                        .map(|l| fma16(c[0][l], c[1][l], c[2][l]))
                        .collect(),
                )
            },
            &near,
        );
    }
}

/// The `half` reference itself: exact sums of a few hand-checked triples, signed zeros, the
/// overflow to infinity, and a denormal result.
#[test]
fn the_half_reference_rounds_the_exact_sum() {
    // 2 * 3 - 6 = +0; -0 * 1 + -0 = -0; 1 * 1 + 2^-24 rounds to 1.
    assert_eq!(fma16(0x4000, 0x4200, 0xC600), 0);
    assert_eq!(fma16(0x8000, 0x3C00, 0x8000), 0x8000);
    assert_eq!(fma16(0x3C00, 0x3C00, 0x0001), 0x3C00);
    // 1 * 1 + 2^-11 ties to 1 (even); 1 * 1 + 2^-11 + 2^-21 rounds up to 1 + 2^-10.
    assert_eq!(fma16(0x3C00, 0x3C00, 0x1000), 0x3C00);
    assert_eq!(fma16(0x3C00, 0x3C00, 0x1001), 0x3C01);
    // 65504 * 2 - 65504 = 65504; 256 * 256 - 0 overflows.
    assert_eq!(fma16(0x7BFF, 0x4000, 0xFBFF), 0x7BFF);
    assert_eq!(fma16(0x5C00, 0x5C00, 0), 0x7C00);
    // 2^-24 * 0.5 + 0 = 2^-25 ties to +0; 3 * 2^-24 * 0.5 ties to 2 * 2^-24.
    assert_eq!(fma16(0x0001, 0x3800, 0), 0);
    assert_eq!(fma16(0x0003, 0x3800, 0), 0x0002);
}

/// The route follows the profile: `float` by `OpFmaKHR` where `shaderFmaFloat32` is enabled, else
/// the device's `Fma`; `half` by the device's 16-bit `Fma` where the profile rounds under the
/// 16-bit float controls, else the device's `float` `Fma` rounded to `half`; `bfloat` and `double`
/// keep their own lowering.
#[test]
fn the_fused_multiply_add_route_follows_the_profile() {
    let mut seen = BTreeSet::new();
    for (stem, p) in committed_profiles().expect("profiles") {
        let l = &p.fields().leaf;
        let want = if l.fma32 { Route::Khr } else { Route::Native };
        assert_eq!(
            Fma::by(fma::routes(&p), float(32)).expect("float").route,
            want,
            "{stem}"
        );
        let half = if crate::lower::arith::narrow::half_controls(&p) {
            Route::F16Native
        } else {
            Route::Native
        };
        assert_eq!(
            Fma::by(fma::routes(&p), float(16)).expect("half").route,
            half,
            "{stem}"
        );
        assert_eq!(Fma::by(fma::routes(&p), bfloat()), None, "{stem}");
        assert_eq!(Fma::by(fma::routes(&p), float(64)), None, "{stem}");
        seen.insert(want);
    }
    assert_eq!(
        seen,
        BTreeSet::from([Route::Khr, Route::Native]),
        "every float route is some committed profile's"
    );
}
