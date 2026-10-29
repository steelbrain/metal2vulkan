//! 3D samples of an extended-range texture: the nearest texel, the 3D linear footprint's sum, and
//! the level a nearest mip filter reads.
//!
//! - On the CPU interpreter, Metal's 3D samples of the XR mip probe (an Apple M3 Max on macOS
//!   27.0.1, the fixture `xr_volumes.tsv`) are Metal's bytes, through a runtime sampler and through
//!   a constexpr one of the same state: `BGRA10_XR`, `BGR10_XR` and their `_sRGB` twins, 3D
//!   textures of five levels, nearest and linear filters under a nearest mip filter, `float` and
//!   `half` results, at explicit levels, clamped to the edge, and linear samples under mirrored
//!   repeat and repeat.
//! - What is not built is refused: a 3D texture's linear mip filter, by the lowering for a
//!   constexpr sampler and by the binder for a runtime one, and a padded form's 3D sample.

use super::levels::{constexpr_words, descriptor, level_kernel, probe_texture, run_levels, Shape};
use super::samples::{form_of, sampled_state_of, Through};
use super::*;
use crate::error::Unsupported;
use crate::metal::{SamplerAddressMode, SamplerMipFilter};

/// One line of `xr_volumes.tsv` (the XR mip probe's 3D samples): the raw format; the result (`f`,
/// `h`); the min and mag filter and the mip filter of the runtime sampler; its address mode on
/// every axis (`edge`, `mirror`, `repeat`; normalized coordinates, the default clamp); the
/// point's `u`, `v`, `w` and level as `f32` bits; and the result's lanes as bits.
struct Row {
    form: ExtendedRange,
    half: bool,
    linear: bool,
    mode: SamplerAddressMode,
    point: [f32; 4],
    bits: Vec<u32>,
}

fn rows() -> Vec<Row> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/xr_volumes.tsv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let f = |s: &str| f32::from_bits(u32::from_str_radix(s, 16).expect("f32 bits"));
    text.lines()
        .map(|l| {
            let r: Vec<&str> = l.split(' ').collect();
            assert_eq!((r[0], r[2], r[5]), ("S", "3d", "mipnearest"), "{l}");
            Row {
                form: form_of(r[1]),
                half: r[3] == "h",
                linear: r[4] == "linear",
                mode: match r[6] {
                    "edge" => SamplerAddressMode::ClampToEdge,
                    "mirror" => SamplerAddressMode::MirrorRepeat,
                    "repeat" => SamplerAddressMode::Repeat,
                    other => panic!("address mode {other}"),
                },
                point: [f(r[7]), f(r[8]), f(r[9]), f(r[10])],
                bits: r[11]
                    .split(',')
                    .map(|x| u32::from_str_radix(x, 16).expect("bits"))
                    .collect(),
            }
        })
        .collect()
}

/// The AIR address code of `mode` (`samples::words`).
fn code(mode: SamplerAddressMode) -> u64 {
    match mode {
        SamplerAddressMode::Repeat => 2,
        SamplerAddressMode::MirrorRepeat => 3,
        _ => 1,
    }
}

/// A kernel sampling `texture3d<T, sample>` slot 0 through `through` at `float4` buffer 0's lane
/// `tid` (`u`, `v`, `w`, the level), at that level, storing the texel to buffer 1.
fn volume_kernel(through: Through, half: bool) -> String {
    level_kernel(through, half, false, false)
        .replace("sample_texture_2d.", "sample_texture_3d.")
        .replace(
            "<2 x float> %uv, i1 true, <2 x i32> zeroinitializer",
            "<3 x float> %uvw, i1 true, <3 x i32> zeroinitializer",
        )
        .replace(
            "<2 x float>, i1, <2 x i32>",
            "<3 x float>, i1, <3 x i32>",
        )
        .replace(
            "%uv = shufflevector <4 x float> %pt, <4 x float> poison, <2 x i32> <i32 0, i32 1>",
            "%uvw = shufflevector <4 x float> %pt, <4 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>",
        )
        .replace(
            "%lod = extractelement <4 x float> %pt, i64 2",
            "%lod = extractelement <4 x float> %pt, i64 3",
        )
        .replace("texture2d<", "texture3d<")
}

/// The constexpr sampler of a row's state: its filters, a nearest mip filter, and `mode` on every
/// axis.
fn volume_words(linear: bool, mip: SamplerMipFilter, mode: SamplerAddressMode) -> u64 {
    let c = code(mode);
    constexpr_words(linear, mip, None) & !0x1ff | c | c << 3 | c << 6
}

/// The runtime sampler of a row's state.
fn volume_descriptor(
    linear: bool,
    mip: SamplerMipFilter,
    mode: SamplerAddressMode,
) -> crate::bind::SamplerDescriptor {
    let mut d = descriptor(linear, mip, None);
    d.s_address_mode = mode;
    d.t_address_mode = mode;
    d.r_address_mode = mode;
    d
}

/// Every 3D sample of `xr_volumes.tsv` is Metal's bytes, through the runtime sampler of its state
/// and through the constexpr sampler of that state: the level a nearest mip filter reads, the
/// nearest texel on three axes, and the 3D linear footprint's sum (each tap's weight the product
/// of the axes' weights rounded onto `2^-16`, a tie up on the `t` axis's first tap).
#[test]
fn volume_samples_are_metals_bytes() {
    let rows = rows();
    assert!(rows.len() > 120, "{} rows", rows.len());
    let mut groups: Vec<(String, Vec<&Row>)> = Vec::new();
    for r in &rows {
        let key = format!("{:?} {} {} {:?}", r.form, r.half, r.linear, r.mode);
        match groups.last_mut() {
            Some((k, g)) if *k == key => g.push(r),
            _ => groups.push((key, vec![r])),
        }
    }
    let mut checked = 0;
    for (key, group) in &groups {
        let r0 = group[0];
        // An `_sRGB` form's decode takes the device's `Pow`, which the interpreter does not model.
        if r0.form.srgb() {
            continue;
        }
        let seed = match r0.mode {
            SamplerAddressMode::ClampToEdge => 7,
            _ => 17,
        };
        let levels = probe_texture(r0.form, Shape::Volume, seed);
        let points: Vec<[f32; 4]> = group.iter().map(|r| r.point).collect();
        let nearest = SamplerMipFilter::Nearest;
        let desc = volume_descriptor(r0.linear, nearest, r0.mode);
        let words = volume_words(r0.linear, nearest, r0.mode);
        for (through, runtime) in [
            (Through::Runtime, Some(desc)),
            (Through::Constexpr(words), None),
        ] {
            let Some(bc) = assemble("volumes", &volume_kernel(through, r0.half)) else {
                return;
            };
            let mut inputs = LowerInputs::new();
            inputs.extended_range.insert(SOURCE, r0.form);
            for (stem, p) in profiles() {
                let what = format!("{key} {through:?} on {stem}");
                validate(&bc, &inputs, &p, &what);
                let (plan, m) = typed_module_with(&bc, "k", &p, &inputs)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let got = run_levels(
                    &plan,
                    &m,
                    (r0.form, &levels, Shape::Volume),
                    (&points, r0.half, false),
                    runtime,
                    &what,
                );
                for (r, lanes) in group.iter().zip(got) {
                    match r.linear {
                        true => assert!(
                            super::samples::filtered_close(&lanes, &r.bits, r.half, 0.0),
                            "{what}: point {:?}: {lanes:x?} against {:x?}",
                            r.point,
                            r.bits
                        ),
                        false => assert!(
                            super::samples::near_metals(&lanes, &r.bits),
                            "{what}: point {:?}: {lanes:x?}, Metal's {:x?}",
                            r.point,
                            r.bits
                        ),
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0);
}

/// A 3D texture's linear mip filter is not built (its blend of two levels fits no rule tried):
/// the lowering refuses a constexpr sampler's, `NotImplemented(Format)`, and builds its nearest
/// one; the binder refuses a runtime sampler's and plans a nearest one into the variant. A padded
/// form's 3D sample is refused too (not measured).
#[test]
fn volume_linear_mip_filters_and_padded_forms_are_refused() {
    let forms = [
        (PixelFormat::BGRA10XR, ExtendedRange::Bgra10 { srgb: false }),
        (
            PixelFormat::BGR10XRSRGB,
            ExtendedRange::Bgr10 { srgb: true },
        ),
    ];
    let edge = SamplerAddressMode::ClampToEdge;
    let refused = |what: &str, r: Result<(), crate::error::Error>| match r {
        Err(crate::error::Error::Unsupported(u)) => {
            let want = Unsupported::not_implemented(Feature::Format);
            assert_eq!((u.feature, u.needs), (want.feature, want.needs), "{what}");
        }
        other => panic!("{what}: {other:?}"),
    };
    for mip in [SamplerMipFilter::Nearest, SamplerMipFilter::Linear] {
        let words = volume_words(true, mip, edge);
        let Some(bc) = assemble(
            "constexpr 3D mip",
            &volume_kernel(Through::Constexpr(words), false),
        ) else {
            return;
        };
        for (stem, p) in profiles() {
            for (_, form) in forms {
                let mut i = LowerInputs::new();
                i.extended_range.insert(SOURCE, form);
                let what = format!("{mip:?} {form:?} on {stem}");
                let r = typed_module_with(&bc, "k", &p, &i).map(|_| ());
                match mip {
                    SamplerMipFilter::Nearest => r.unwrap_or_else(|e| panic!("{what}: {e}")),
                    _ => refused(&what, r),
                }
            }
            let mut i = LowerInputs::new();
            i.extended_range.insert(SOURCE, ExtendedRange::R10X6);
            let r = typed_module_with(&bc, "k", &p, &i).map(|_| ());
            refused(&format!("{mip:?} R10X6 on {stem}"), r);
        }
    }
    let Some(bc) = assemble("runtime 3D mip", &volume_kernel(Through::Runtime, false)) else {
        return;
    };
    for (stem, dev) in devices() {
        let plain = pipeline(&dev, &bc, &LowerInputs::new()).expect("a sampling kernel");
        for (pf, form) in forms {
            for (mip, planned) in [
                (SamplerMipFilter::Nearest, true),
                (SamplerMipFilter::Linear, false),
            ] {
                let what = format!("{mip:?} {pf:?} on {stem}");
                let desc = volume_descriptor(true, mip, edge);
                let state = sampled_state_of(TextureType::Type3D, pf, desc);
                match (plain.plan(&dev, &state.shape()), planned) {
                    (Ok(plan), true) => assert_eq!(
                        plan.variant.extended_range,
                        BTreeMap::from([(SOURCE, form)]),
                        "{what}"
                    ),
                    (Err(crate::error::BindError::Unsupported(u)), false) => {
                        assert_eq!(u, Unsupported::not_implemented(Feature::Format), "{what}")
                    }
                    (other, _) => panic!("{what}: {:?}", other.map(|_| ())),
                }
            }
        }
    }
}
