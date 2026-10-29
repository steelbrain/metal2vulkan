//! Interpreter differential tests of the bicubic filter's routines
//! (`lower::sampler::bicubic::routines`): the device's arithmetic, its references plain Rust `f32`
//! in the routines' order (IEEE, as the interpreter evaluates it).

use super::*;
use crate::lower::sampler::bicubic::routines::{self, Routine};
use crate::metal::bicubic::BicubicTexels as T;

fn u() -> Form {
    int(32)
}

fn v4() -> Form {
    vec(float(32), 4)
}

fn fraction(rng: &mut Rng) -> u64 {
    match rng.next() % 6 {
        0 => 0,
        1 => 128,
        2 => 255,
        _ => rng.next() % 256,
    }
}

/// A tap value in `[-1.5, 1.5]` on a fine grid, as `f32` bits.
fn tap(rng: &mut Rng) -> u64 {
    let x = (rng.next() % 6001) as f32 / 2000.0 - 1.5;
    u64::from(x.to_bits())
}

/// The Catmull-Rom weights at `k / 256` as the routine computes them: `((A f + B) f + C) f + D`
/// lane by lane, each product and sum rounded to `f32`.
fn catmull_rom(k: u32) -> [f32; 4] {
    let f = k as f32 * (1.0 / 256.0);
    let a = [-0.5f32, 1.5, -1.5, 0.5];
    let b = [1.0f32, -2.5, 2.0, -0.5];
    let c = [-0.5f32, 0.0, 0.5, 0.0];
    let d = [0.0f32, 1.0, 0.0, 0.0];
    std::array::from_fn(|i| ((a[i] * f + b[i]) * f + c[i]) * f + d[i])
}

/// `FMix(x, y, t)` as `GLSL.std.450` defines it: `x · (1 - t) + y · t`.
fn mix(x: f32, y: f32, t: f32) -> f32 {
    x * (1.0 - t) + y * t
}

/// The 2D routine's sum: each row's taps under the `s` axis's weights in order, then the rows
/// under the `t` axis's, every product and sum an `f32` operation; or where `bilinear` the inner
/// taps' `FMix` at `lkx` and `lky`; then the format's clamp.
fn plane(taps: &[[f32; 4]; 16], k: [u32; 4], bilinear: bool, flags: u32) -> [f32; 4] {
    let (wx, wy) = (catmull_rom(k[0]), catmull_rom(k[1]));
    let (fx, fy) = (k[2] as f32 / 256.0, k[3] as f32 / 256.0);
    std::array::from_fn(|l| {
        let r = if bilinear {
            let top = mix(taps[5][l], taps[6][l], fx);
            let bottom = mix(taps[9][l], taps[10][l], fx);
            mix(top, bottom, fy)
        } else {
            let mut rows = [0f32; 4];
            for (j, row) in rows.iter_mut().enumerate() {
                let mut s = taps[4 * j][l] * wx[0];
                for i in 1..4 {
                    s += taps[4 * j + i][l] * wx[i];
                }
                *row = s;
            }
            let mut s = rows[0] * wy[0];
            for j in 1..4 {
                s += rows[j] * wy[j];
            }
            s
        };
        match flags {
            f if f & T::UNORM != 0 => r.clamp(0.0, 1.0),
            f if f & T::SNORM != 0 => r.clamp(-1.0, 1.0),
            _ => r,
        }
    })
}

/// The 2D routine is the Catmull-Rom filter in the device's `f32` multiply and add (the weights
/// from the fraction, not Metal's table, and no integer sum), the inner taps' `FMix` where the
/// sample is bilinear, and the normalized formats' clamps.
#[test]
fn the_plane_routine_sums_in_the_devices_arithmetic() {
    let mut ins = vec![v4(); 16];
    ins.extend([u(), u(), u(), u(), boolean(), u()]);
    let mut rng = Rng(0x00b1_c0b1_c5ee_d0a1);
    let mut cases = Vec::new();
    for flags in [T::UNORM | T::EXACT, T::SNORM, T::HALF, T::LINEAR, 0] {
        for n in 0..400 {
            let taps: Vec<Vec<u64>> = (0..16)
                .map(|_| (0..4).map(|_| tap(&mut rng)).collect())
                .collect();
            let mut c: Case = taps;
            let k: [u64; 4] = std::array::from_fn(|_| fraction(&mut rng));
            c.extend(k.map(|k| vec![k]));
            let bilinear = flags & T::LINEAR != 0 || n % 5 == 0;
            c.push(vec![u64::from(bilinear)]);
            c.push(vec![u64::from(flags)]);
            cases.push(c);
        }
    }
    let body = |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
        let p = routines::define_plane(fb.module())?;
        Ok(fb.call(p.sum, x)?.expect("sum"))
    };
    let reference = |c: &Case| {
        let taps: [[f32; 4]; 16] =
            std::array::from_fn(|t| std::array::from_fn(|l| f32::from_bits(c[t][l] as u32)));
        let k = std::array::from_fn(|i| arg(c, 16 + i) as u32);
        let r = plane(&taps, k, arg(c, 20) != 0, arg(c, 21) as u32);
        Some(r.iter().map(|x| u64::from(x.to_bits())).collect())
    };
    check(
        "bicubic plane",
        &ins,
        v4(),
        &body,
        &cases,
        &reference,
        false,
    );
}

/// The 1D routine is the device's `FMix` of its two taps at `k / 256`, clamped as the format's
/// flags say.
#[test]
fn the_line_routine_is_the_devices_mix() {
    let mut ins = vec![v4(); 2];
    ins.extend([u(), u()]);
    let mut rng = Rng(0x00b1_c0b1_c5ee_d0a2);
    let mut cases = Vec::new();
    for flags in [T::UNORM, T::SNORM, T::LINEAR] {
        for _ in 0..400 {
            let mut c: Case = (0..2)
                .map(|_| (0..4).map(|_| tap(&mut rng)).collect())
                .collect();
            c.push(vec![fraction(&mut rng)]);
            c.push(vec![u64::from(flags)]);
            cases.push(c);
        }
    }
    let body = |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
        let l = routines::define_line(fb.module())?;
        Ok(fb.call(l.sum, x)?.expect("sum"))
    };
    let reference = |c: &Case| {
        let f = arg(c, 2) as f32 / 256.0;
        let flags = arg(c, 3) as u32;
        let lanes = (0..4).map(|l| {
            let (a, b) = (
                f32::from_bits(c[0][l] as u32),
                f32::from_bits(c[1][l] as u32),
            );
            let r = mix(a, b, f);
            let r = match flags {
                f if f & T::UNORM != 0 => r.clamp(0.0, 1.0),
                f if f & T::SNORM != 0 => r.clamp(-1.0, 1.0),
                _ => r,
            };
            u64::from(r.to_bits())
        });
        Some(lanes.collect())
    };
    check("bicubic line", &ins, v4(), &body, &cases, &reference, false);
}

/// [`Routine::words`] bounds each routine's emission.
#[test]
fn bicubic_routines_words_bound_their_emission() {
    use crate::spirv::serialize::serialize;
    let size = |body: Body<'_>| -> usize {
        let m = kernel(&[u()], u(), body).expect("build");
        serialize(m).expect("serialize").words.len()
    };
    let without = size(&|_, x| Ok(x[0]));
    let plane = size(&|fb, x| {
        routines::define_plane(fb.module())?;
        Ok(x[0])
    }) - without;
    let line = size(&|fb, x| {
        routines::define_line(fb.module())?;
        Ok(x[0])
    }) - without;
    let quad = size(&|fb, x| {
        routines::define_quad(fb.module())?;
        Ok(x[0])
    }) - without;
    eprintln!("bicubic routines: plane {plane} words, line {line} words, quad {quad} words");
    for (routine, words) in [
        (Routine::Plane, plane),
        (Routine::Line, line),
        (Routine::Quad, quad),
    ] {
        assert!(
            words <= routine.words() as usize,
            "{routine:?} is {words} words, bound {}",
            routine.words()
        );
    }
}
