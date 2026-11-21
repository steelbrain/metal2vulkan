//! `emit::cube_footprint`, on the interpreter, against Metal's footprint. Each kernel invocation
//! reads a direction's bits and writes the bits of the gradient the device is handed for a constant
//! explicit gradient `g`.
//!
//! - Metal under macOS 27.0.1 (the Apple M3 Max, the probe `emit::cube_footprint` cites) reads,
//!   on the direction's face, the face's two tangent components, `(σ_s g[s], σ_t g[t])` in
//!   Vulkan's face orientation, `s` and `t` the axes of the face's `s_c` and `t_c` and `σ` their
//!   signs, with `dPdx`'s `t` component and `dPdy`'s `s` component times `ρ = σ_s σ_t` through an
//!   anisotropic sampler. Vulkan's cube derivative rule must read that from the `Face` form over
//!   `2 |m_a|`, and nothing on the normal; and the `Length` form must hand Metal's length of the
//!   two on one tangent.
//!
//! Through a runtime sampler `emit::word_footprint` hands one of those forms or the isotropic one
//! by the sampler's word (`emit::by_anisotropy`).

use super::super::super::anisotropic::cube_face;
use super::super::super::emit::{by_anisotropy, cube_footprint};
use super::super::super::CubeFootprint;
use super::{kernel_on, run_records};
use crate::metal::sampler::SamplerWord;
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::ir::{ConstBits, Constant, Type};

/// Vulkan's face of direction `d` ("Cube map face selection"; no ties among `d`'s magnitudes):
/// its major axis, and the axis and sign of its `s_c` and of its `t_c`. Metal's faces are the
/// same.
fn face_of(d: [f32; 3]) -> (usize, (usize, f32), (usize, f32)) {
    let m = d.map(f32::abs);
    let a = if m[2] >= m[0] && m[2] >= m[1] {
        2
    } else if m[1] >= m[0] {
        1
    } else {
        0
    };
    let positive = d[a] > 0.0;
    let (s, t) = match (a, positive) {
        (0, true) => ((2, -1.0), (1, -1.0)),
        (0, false) => ((2, 1.0), (1, -1.0)),
        (1, true) => ((0, 1.0), (2, 1.0)),
        (1, false) => ((0, 1.0), (2, -1.0)),
        (_, true) => ((0, 1.0), (1, -1.0)),
        (_, false) => ((0, -1.0), (1, -1.0)),
    };
    (a, s, t)
}

/// The face gradient Metal reads under macOS 27 from gradient `g` (`dPdx` where `is_dx`, else
/// `dPdy`) at direction `d` through an anisotropic sampler, in Vulkan's face orientation.
fn metal_reading(g: [f32; 3], d: [f32; 3], is_dx: bool) -> [f32; 2] {
    let (_, (s, s_sign), (t, t_sign)) = face_of(d);
    let rho = s_sign * t_sign;
    match is_dx {
        true => [s_sign * g[s], rho * t_sign * g[t]],
        false => [rho * s_sign * g[s], t_sign * g[t]],
    }
}

/// Metal's length of a gradient of components `x` and `y`: `max(M + 3m/16, 49M/64 + 21m/32)`.
fn metal_length(x: f32, y: f32) -> f32 {
    let (big, small) = (x.abs().max(y.abs()), x.abs().min(y.abs()));
    (big + small * 0.1875).max(big * 0.765_625 + small * 0.656_25)
}

/// Directions on each of the six faces, at the face's center and off it.
fn directions() -> Vec<[f32; 3]> {
    let mut out = Vec::new();
    for a in 0..3 {
        for sign in [1.0f32, -1.0] {
            for (u, v) in [(0.0f32, 0.0f32), (0.25, -0.5), (-0.75, 0.125)] {
                let mut d = [0.0f32; 3];
                d[a] = 2.0 * sign;
                d[(a + 1) % 3] = u;
                d[(a + 2) % 3] = v;
                out.push(d);
            }
        }
    }
    out
}

/// The gradients `cube_footprint` hands the device in `form` for gradient `g` (as `dPdx` where
/// `is_dx`, else as `dPdy`), per direction of `lanes`.
fn handed_as(form: CubeFootprint, g: [f32; 3], lanes: &[[f32; 3]], is_dx: bool) -> Vec<[f32; 3]> {
    let records: Vec<u32> = lanes.iter().flat_map(|d| d.map(f32::to_bits)).collect();
    let m = kernel_on(ModuleBuilder::new(), 3, &|f, p, _| {
        let f32_ = f.module().ty(Type::Float { width: 32 })?;
        let vec3 = f.module().ty(Type::Vector { elem: f32_, len: 3 })?;
        let zero = f.module().constant(Constant::Scalar {
            ty: f32_,
            bits: ConstBits(0),
        })?;
        let zero = f.module().const_value(zero)?;
        let mut c = Vec::new();
        for x in g {
            let k = f.module().constant(Constant::Scalar {
                ty: f32_,
                bits: ConstBits(u64::from(x.to_bits())),
            })?;
            c.push(f.module().const_value(k)?);
        }
        let g = f.composite_construct(vec3, &c)?;
        let other = f.composite_construct(vec3, &[zero, zero, zero])?;
        let (dx, dy) = cube_footprint(f, p, (g, other), form)?;
        let (dx2, dy2) = cube_footprint(f, p, (other, g), form)?;
        let _ = (dy, dx2);
        Ok(match is_dx {
            true => dx,
            false => dy2,
        })
    })
    .expect("kernel");
    run_records(&m, &records, lanes.len())
}

/// The gradients `cube_footprint` hands the device in `form` for `dPdx` `g`, per direction of
/// `lanes`.
fn handed(form: CubeFootprint, g: [f32; 3], lanes: &[[f32; 3]]) -> Vec<[f32; 3]> {
    handed_as(form, g, lanes, true)
}

const GRADIENTS: [[f32; 3]; 3] = [[1.0, 2.0, 4.0], [-3.0, 0.5, -7.25], [0.0, -1.5, 6.0]];

/// Vulkan's rule reads Metal's macOS 27 face gradients from the `Face` form, for `dPdx` and
/// `dPdy` alike, the normal's component zero.
#[test]
fn a_face_footprint_keeps_metals_face_gradients() {
    let lanes = directions();
    for g in GRADIENTS {
        for is_dx in [true, false] {
            for (d, h) in lanes
                .iter()
                .zip(handed_as(CubeFootprint::Face, g, &lanes, is_dx))
            {
                let (a, (s, s_sign), (t, t_sign)) = face_of(*d);
                assert_eq!(h[a], 0.0, "{d:?} {g:?}: the normal's component {h:?}");
                let read = [s_sign * h[s], t_sign * h[t]];
                let metal = metal_reading(g, *d, is_dx);
                assert_eq!(read, metal, "{d:?} {g:?} dx {is_dx}: handed {h:?}");
            }
        }
    }
}

/// The `Length` form hands Metal's length of the face's two tangent components on the tangent `(a
/// + 2) mod 3` alone, bit for bit, which Vulkan's rule and every Metal reading read alike.
#[test]
fn a_length_footprint_hands_metals_length_on_one_tangent() {
    let lanes = directions();
    for g in GRADIENTS {
        for (d, h) in lanes.iter().zip(handed(CubeFootprint::Length, g, &lanes)) {
            let (a, (s, _), (t, _)) = face_of(*d);
            let b = (a + 2) % 3;
            let mut want = [0.0f32; 3];
            want[b] = metal_length(g[s], g[t]);
            assert_eq!(
                h.map(f32::to_bits),
                want.map(f32::to_bits),
                "{d:?} {g:?}: handed {h:?}"
            );
        }
    }
}

/// What `by_anisotropy` hands the device for gradient `g` through a sampler of word `word`, per
/// direction of `lanes`, choosing between the gradients `cube_footprint` hands in form `set` and in
/// form `clear`.
fn by_word(
    (set, clear): (CubeFootprint, CubeFootprint),
    word: u32,
    g: [f32; 3],
    lanes: &[[f32; 3]],
) -> Vec<[f32; 3]> {
    let records: Vec<u32> = lanes
        .iter()
        .flat_map(|d| [d[0].to_bits(), d[1].to_bits(), d[2].to_bits(), word])
        .collect();
    let m = kernel_on(ModuleBuilder::new(), 4, &|f, p, w| {
        let f32_ = f.module().ty(Type::Float { width: 32 })?;
        let vec3 = f.module().ty(Type::Vector { elem: f32_, len: 3 })?;
        let mut c = Vec::new();
        for x in g {
            let k = f.module().constant(Constant::Scalar {
                ty: f32_,
                bits: ConstBits(u64::from(x.to_bits())),
            })?;
            c.push(f.module().const_value(k)?);
        }
        let g = f.composite_construct(vec3, &c)?;
        let a = cube_footprint(f, p, (g, g), set)?;
        let i = cube_footprint(f, p, (g, g), clear)?;
        Ok(by_anisotropy(f, w, a, i)?.0)
    })
    .expect("kernel");
    run_records(&m, &records, lanes.len())
}

/// A runtime sampler's footprint by its word (`word_footprint`) takes, on every lane and bit for
/// bit, the anisotropic arm where the word states `SamplerWord::ANISOTROPIC`, whatever its other
/// fields, and the isotropic arm where it does not: the choice is `by_anisotropy`'s, between
/// whatever two forms it is given.
#[test]
fn a_footprint_by_the_word_takes_the_form_its_anisotropy_states() {
    let lanes = directions();
    let others = SamplerWord::MIN_LINEAR
        | SamplerWord::MAG_LINEAR
        | (2 << SamplerWord::MIP_FILTER_SHIFT)
        | SamplerWord::LEVEL_MAX;
    let bits =
        |v: &[[f32; 3]]| -> Vec<[u32; 3]> { v.iter().map(|x| x.map(f32::to_bits)).collect() };
    let (set, clear) = (CubeFootprint::Face, CubeFootprint::Length);
    for g in GRADIENTS {
        let (when_set, when_clear) = (
            bits(&handed(set, g, &lanes)),
            bits(&handed(clear, g, &lanes)),
        );
        assert_ne!(when_set, when_clear, "{g:?}");
        for (word, want) in [
            (0, &when_clear),
            (others, &when_clear),
            (SamplerWord::ANISOTROPIC, &when_set),
            (SamplerWord::ANISOTROPIC | others, &when_set),
        ] {
            let got = bits(&by_word((set, clear), word, g, &lanes));
            assert_eq!(&got, want, "{g:?} {word:#x}");
        }
    }
}

/// The emulation's face of each direction of `lanes` and gradient `g` (`anisotropic::cube_face`):
/// `out` 0 and 1, Metal's face gradient of `g` as `dPdx` and as `dPdy` (on `x` and `y`, `z` 0);
/// 2 and 3, the steps of the direction that move the face's `s` and `t` by one.
fn emulated_face(g: [f32; 3], lanes: &[[f32; 3]], out: usize) -> Vec<[f32; 3]> {
    let records: Vec<u32> = lanes.iter().flat_map(|d| d.map(f32::to_bits)).collect();
    let m = kernel_on(ModuleBuilder::new(), 3, &|f, p, _| {
        let f32_ = f.module().ty(Type::Float { width: 32 })?;
        let vec3 = f.module().ty(Type::Vector { elem: f32_, len: 3 })?;
        let zero = f.module().constant(Constant::Scalar {
            ty: f32_,
            bits: ConstBits(0),
        })?;
        let zero = f.module().const_value(zero)?;
        let mut c = Vec::new();
        for x in g {
            let k = f.module().constant(Constant::Scalar {
                ty: f32_,
                bits: ConstBits(u64::from(x.to_bits())),
            })?;
            c.push(f.module().const_value(k)?);
        }
        let g = f.composite_construct(vec3, &c)?;
        let other = f.composite_construct(vec3, &[zero, zero, zero])?;
        let (face, (gx, _)) = cube_face(f, p, (g, other))?;
        let (_, (_, gy)) = cube_face(f, p, (other, g))?;
        let pair = |f: &mut crate::spirv::builder::FunctionBuilder<'_>, v| {
            let x = f.composite_extract(v, &[0])?;
            let y = f.composite_extract(v, &[1])?;
            f.composite_construct(vec3, &[x, y, zero])
        };
        match out {
            0 => pair(f, gx),
            1 => pair(f, gy),
            2 => Ok(face.s),
            _ => Ok(face.t),
        }
    })
    .expect("kernel");
    run_records(&m, &records, lanes.len())
}

/// The anisotropic emulation's face of a cube direction takes Metal's macOS 27 face gradients
/// (`metal_reading`) over `2 |m_a|`, bit for bit (the directions' major magnitude is 2), for
/// `dPdx` and `dPdy` alike; and its steps move the direction along the face's `s` and `t` axes by
/// `2 |m_a|` times their signs, so the face coordinate Vulkan projects moves by one.
#[test]
fn the_emulations_face_takes_metals_gradients_and_steps() {
    let lanes = directions();
    for g in GRADIENTS {
        for (out, is_dx) in [(0, true), (1, false)] {
            for (d, h) in lanes.iter().zip(emulated_face(g, &lanes, out)) {
                let metal = metal_reading(g, *d, is_dx);
                let want = [metal[0] / 4.0, metal[1] / 4.0, 0.0];
                assert_eq!(
                    h.map(f32::to_bits),
                    want.map(f32::to_bits),
                    "{d:?} {g:?} dx {is_dx}: {h:?}"
                );
            }
        }
    }
    for (out, axis) in [(2, 0), (3, 1)] {
        for (d, h) in lanes.iter().zip(emulated_face(GRADIENTS[0], &lanes, out)) {
            let (_, s, t) = face_of(*d);
            let (on, sign) = [s, t][axis];
            let mut want = [0.0f32; 3];
            want[on] = 4.0 * sign;
            assert_eq!(h, want, "{d:?}: step {axis} {h:?}");
        }
    }
}
