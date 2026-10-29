//! The maths routines left on the interpreter: precise `fmod`'s routine, exact against C's `fmod`
//! on `float` and on `half`, and the infinite product of Metal's fast `pow` and `powr`.

use super::*;
use crate::lower::arith::libm::{self, Lib};

/// Operands: every 2^k from `2^-126` to `2^127` and their neighbours, both signs, pseudo-random
/// bit patterns of every exponent, and the interval edges each routine reduces by.
fn operands(seed: u64, random: usize) -> Vec<u64> {
    let mut v = Vec::new();
    for e in 1u64..255 {
        for m in [0u64, 1, 0x7F_FFFF, 0x40_0000] {
            for s in [0u64, 1] {
                v.push((s << 31) | (e << 23) | m);
            }
        }
    }
    for x in [
        0.4375f32,
        0.6875,
        1.1875,
        2.4375,
        0.5,
        1.0,
        0.25,
        0.75,
        0.975,
        0.999_999_9,
        1.5,
        3.0,
        67_108_864.0,
        67_108_860.0,
    ] {
        for d in [-2i64, -1, 0, 1, 2] {
            let b = (i64::from(x.to_bits()) + d) as u64;
            v.push(b);
            v.push(b | 0x8000_0000);
        }
    }
    let mut rng = Rng(seed);
    for _ in 0..random {
        let r = rng.next() & 0xFFFF_FFFF;
        if r & 0x7F80_0000 != 0x7F80_0000 {
            v.push(r);
        }
        // Concentrated where the functions have most of their range.
        let near = (rng.next() & 0x00FF_FFFF) | 0x3E00_0000;
        v.push(near | (rng.next() & 0x8000_0000));
    }
    v
}

/// A module of precise `fmod` on lanes of form `f`, by its routine (`libm::define_fmod`): a
/// `half` call takes the `float` routine before its rounding.
fn fmod_module(f: Form) -> TypedModule {
    kernel_defs(
        &[f, f],
        f,
        &|b| Ok(vec![libm::define_fmod(b)?]),
        &|fb, x, funcs| {
            math::float_fn(
                fb,
                FloatFn::Fmod,
                f,
                Contraction::Forbidden,
                x,
                &Lib::new(funcs.first().copied()),
            )
        },
        true,
        false,
    )
    .unwrap_or_else(|e| panic!("fmod {f:?}: build: {e:?}"))
}

/// Precise `fmod` is exact (MSL Table 8.1: 0 ulp), bit for bit against C's `fmod` (Rust's `%`) of
/// the operands as Metal reads them (a denormal divisor as a zero): operands of every exponent,
/// denormal dividends included, exponent gaps up to 253, equal and near-equal magnitudes, `fmod(x,
/// ±inf) = x`, and a NaN for an infinite or NaN `x`, a NaN `y`, or a zero or denormal `y`. A zero
/// remainder is the zero of `x`'s sign.
#[test]
fn fmod_routine_is_exact() {
    let mut pool: Vec<u64> = operands(0x0F0D_D1CE, 400)
        .into_iter()
        .filter(|b| f32v(*b).is_finite())
        .collect();
    let denormals = [
        0x0000_0001u64,
        0x0000_0003,
        0x0040_0000,
        0x007F_FFFF,
        0x0000_0100,
    ];
    for d in denormals {
        pool.push(d);
        pool.push(d | 0x8000_0000);
    }
    let mut rng = Rng(0x3E3E_F00D);
    let mut pairs: Vec<Vec<u64>> = Vec::new();
    for _ in 0..12000 {
        let x = pool[(rng.next() as usize) % pool.len()];
        let y = pool[(rng.next() as usize) % pool.len()];
        pairs.push(vec![x, y]);
    }
    // Near-equal magnitudes, small multiples, and the widest gaps.
    for _ in 0..2000 {
        let y = pool[(rng.next() as usize) % pool.len()];
        let k = (rng.next() % 9) as f32 - 4.0;
        let x = f32v(y) * k + f32v(y) * 2f32.powi(-((rng.next() % 30) as i32));
        if x.is_finite() {
            pairs.push(vec![u64::from(x.to_bits()), y]);
        }
        pairs.push(vec![y, y ^ 0x8000_0000]);
    }
    for (x, y) in [
        (0x7F7F_FFFFu64, 0x0000_0001u64),
        (0xFF7F_FFFF, 0x0000_0001),
        (0x7F7F_FFFF, 0x0080_0001),
        (0x7F7F_FFFF, 0x4040_0000),
        (0x4000_0000, 0x0000_0001),
        (0xCAC9_0FDB, 0x0000_0001),
        (0x4AC9_0FDC, 0x0080_0000),
        (0x3FC0_0000, 0xBF40_0000),
        (0x8000_0000, 0x4000_0000),
        (0xFF7F_FFFF, 0xBF80_0000),
    ] {
        pairs.push(vec![x, y]);
    }
    // The special operands.
    for x in [
        0x3F80_0000u64,
        0xBFC9_0FDB,
        0x0080_0000,
        0x8000_0001,
        0x0000_0000,
    ] {
        for y in [
            0x7F80_0000u64,
            0xFF80_0000,
            0x0000_0000,
            0x8000_0000,
            0x7FC0_0000,
        ] {
            pairs.push(vec![x, y]);
        }
    }
    for x in [0x7F80_0000u64, 0xFF80_0000, 0x7FC0_0000] {
        for y in [0x3F80_0000u64, 0x7F80_0000, 0x0000_0001] {
            pairs.push(vec![x, y]);
        }
    }
    let f = float(32);
    check_module(
        "Fmod",
        fmod_module(f),
        &[f, f],
        f,
        &pairs
            .iter()
            .map(|c| c.iter().map(|b| vec![*b]).collect())
            .collect::<Vec<Case>>(),
        &|c| {
            let y = c[1][0];
            let y = if y & 0x7F80_0000 == 0 {
                y & 0x8000_0000
            } else {
                y
            };
            Some(vec![u64::from((f32v(c[0][0]) % f32v(y)).to_bits())])
        },
        &|g, w| g == w || (f32v(g).is_nan() && f32v(w).is_nan()),
    );
}

/// Metal's precise `fmod` reads a denormal dividend as itself and a denormal divisor as a zero
/// (case 087758cb, every such lane): `fmod(2, 2^-149)` and `fmod(-6588397.5, 2^-149)` are NaN,
/// `fmod(±2^-149, y)` is the dividend, and so is `fmod(2^-127, 8)`; on every device, since the
/// routine decides on the operands' bits.
#[test]
fn fmod_routine_reads_metals_denormal_operands() {
    let f = float(32);
    let nan = 0x7FC0_0000u64;
    for (x, y, want) in [
        (0x4000_0000u64, 0x0000_0001u64, nan),
        (0xCAC9_0FDB, 0x0000_0001, nan),
        (0x3F80_0000, 0x807F_FFFF, nan),
        (0x0000_0003, 0x0000_0002, nan),
        (0x0000_0000, 0x0040_0000, nan),
        (0x0000_0001, 0xBF80_0000, 0x0000_0001),
        (0x8000_0001, 0x3F00_0000, 0x8000_0001),
        (0x0040_0000, 0x4100_0000, 0x0040_0000),
        (0x007F_FFFF, 0x0080_0000, 0x007F_FFFF),
    ] {
        check_module(
            &format!("Fmod({x:#010x}, {y:#010x})"),
            fmod_module(f),
            &[f, f],
            f,
            &[vec![vec![x], vec![y]]],
            &|_| Some(vec![want]),
            &|g, w| g == w || (f32v(g).is_nan() && f32v(w).is_nan()),
        );
    }
}

/// A `half` call of `fmod` takes the `float` routine before its rounding (Metal's `half` functions
/// are the same in both modes), so it is Metal's exact `half` remainder (MSL Table 8.3) over a grid
/// of pairs: the remainder of two `half` values is one. The half reading: `fmod(x, ±inf) = x`
/// (where lavapipe, RADV, and NVIDIA answer a NaN) and `fmod(1.5, -0.75) = 0` (RADV: 0.75). A zero
/// remainder is the `float` remainder's, with the dividend's sign, where Metal's `half` answers
/// `+0` for a nonzero dividend (`PrecisionNote::NativeTranscendentals`). Part `part` of `of` of
/// the runs of 4,096 pairs.
fn half_fmod(part: usize, of: usize) {
    let h = float(16);
    let nan = |b: u64| b & 0x7C00 == 0x7C00 && b & 0x03FF != 0;
    let half_of = |x: f32| u64::from(f64_to_f16(f64::from(x)));
    let mut pairs: Vec<Case> = Vec::new();
    for x in (0..=0xFFFFu64).step_by(251) {
        for y in (3..=0xFFFFu64).step_by(241) {
            pairs.push(vec![vec![x], vec![y]]);
        }
    }
    for (x, y) in [
        (0x3E00u64, 0xBA00u64),
        (0x3C00, 0x7C00),
        (0xBC00, 0xFC00),
        (0x7BFF, 0x7C00),
        (0x0001, 0xFC00),
    ] {
        pairs.push(vec![vec![x], vec![y]]);
    }
    let runs: Vec<&[Case]> = pairs.chunks(4096).collect();
    for run in part_of(runs, part, of) {
        check_module(
            "half Fmod",
            fmod_module(h),
            &[h, h],
            h,
            run,
            &|c| {
                let x = f16_to_f64(c[0][0] as u16);
                let y = f16_to_f64(c[1][0] as u16);
                let r = (x as f32) % (y as f32);
                Some(vec![half_of(r)])
            },
            &|g, w| g == w || (nan(g) && nan(w)),
        );
    }
}

mod half_fmod_takes_the_routine {
    parts!(super::half_fmod, 4;
        part_1_of_4 = 0, part_2_of_4 = 1, part_3_of_4 = 2, part_4_of_4 = 3);
}

/// Metal's fast `pow` and `powr` (and its `half` ones, the same compositions) where `y · log2(x)`
/// is infinite answer IEEE's value whatever the composition gave there (`math::infinite_product`
/// over a composition that answered `1`, as the RTX 5080 did without `SignedZeroInfNanPreserve`):
/// Metal's word at case f6524207's special pairs and case 32931370's `half`
/// ones, and the composition's own value where the product is finite, a NaN, or an operand is one.
/// An `f32` denormal base is a zero only where the device's comparison reads it so (the
/// interpreter does not, where Metal does): there the product is finite, the composition's.
#[test]
fn an_infinite_product_answers_ieee_whatever_the_composition_gave() {
    let (f, h) = (float(32), float(16));
    let one = u64::from(1.0f32.to_bits());
    let one_h = 0x3C00;
    // (x, y, answer); `None`: the composition's value kept.
    let words: [(u64, u64, Option<u64>); 15] = [
        (0x0000_0000, 0x3F80_0000, Some(0)),
        (0x8000_0000, 0x4000_0000, Some(0)),
        (0x0000_0001, 0xBF80_0000, None),
        (0x8000_0001, 0x3F00_0000, None),
        (0x0040_0000, 0x4100_0000, None),
        (0x7F80_0000, 0xC000_0000, Some(0)),
        (0x7F80_0000, 0x4100_0000, Some(0x7F80_0000)),
        (0x3F00_0000, 0x7F80_0000, Some(0)),
        (0x4000_0000, 0x7F80_0000, Some(0x7F80_0000)),
        (0x3F00_0000, 0xFF80_0000, Some(0x7F80_0000)),
        (0x4000_0000, 0xFF80_0000, Some(0)),
        (0x3F80_0000, 0x7F80_0000, None),
        (0x0000_0000, 0x0000_0000, None),
        (0x7FC0_0000, 0x4000_0000, None),
        (0x4000_0000, 0x4040_0000, None),
    ];
    let half_words: [(u64, u64, Option<u64>); 8] = [
        (0x0000, 0x3C00, Some(0)),
        (0x8000, 0x4000, Some(0)),
        (0x7C00, 0xC000, Some(0)),
        (0x7C00, 0x4800, Some(0x7C00)),
        (0x3800, 0x7C00, Some(0)),
        (0x4000, 0xFC00, Some(0)),
        (0x0001, 0x3C00, None),
        (0x3C00, 0x7C00, None),
    ];
    for (form, words, kept) in [(f, &words[..], one), (h, &half_words[..], one_h)] {
        let module = kernel_defs(
            &[form, form],
            form,
            &|_| Ok(Vec::new()),
            &|fb, x, _| {
                let r = crate::lower::arith::ops::fsplat(fb, form, 1.0)?;
                math::infinite_product(fb, form, x[1], x[0], r)
            },
            false,
            false,
        )
        .expect("build");
        let cases: Vec<Case> = words
            .iter()
            .map(|(x, y, _)| vec![vec![*x], vec![*y]])
            .collect();
        check_module(
            &format!("infinite product {form:?}"),
            module,
            &[form, form],
            form,
            &cases,
            &|c| {
                words
                    .iter()
                    .find(|(x, y, _)| (*x, *y) == (c[0][0], c[1][0]))
                    .map(|(_, _, w)| vec![w.unwrap_or(kept)])
            },
            &|g, w| g == w,
        );
    }
}
