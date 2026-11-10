//! A device test of the level a depth comparison reads on the device the cfg selects, under the
//! validation layer: comparisons of a mipmapped texture through a runtime sampler slot, through an
//! argument buffer's sampler, and through constexpr samplers, against Metal's measured rule (an
//! Apple M4 on macOS 27.0.1, runtime, argument buffer's and constexpr samplers side by side;
//! our probe before them). Metal rounds the level to `half` (a NaN reads 0), clamps it into
//! the sampler's level clamp (a runtime or argument buffer's sampler holds its bounds as `f32`s, a
//! constexpr one as `half`s) and into the texture's levels; with `L` its whole levels and `q` its
//! fraction truncated onto sixty-fourths, a nearest mip filter reads level `L + 1` where `q > 1/2`,
//! else `L`, a linear one blends the comparisons of levels `L` and `L + 1` by `1 - q` and `q` as an
//! `r8Unorm` texture's codes 0 and 255, rounded once onto sixteenths of a code, ties up, and a
//! sampler without a mip filter reads level 0. A kernel's bias is the explicit level, and a
//! kernel's sample with neither reads level 0, clamped as a level is.
//!
//! The texture is 16x16 of five levels, every texel of level `L` one depth; each job compares
//! `equal` against one level's depth at the texture's centre, so its answer is that level's
//! weight. The samplers filter nearest under nearest and linear mip filters through clamps of
//! whole, half-way and sixty-fourth bounds, and without a mip filter. Metal's own comparison runs
//! through MoltenVK (`VkProperty::MetalSampler`), which must answer the rule exactly; elsewhere
//! the comparison is the device's, which the stages record (`PrecisionNote::DeviceSampling`) and
//! the lane counts against the rule (`DEPTH-LEVEL-DIFFERS` lines) without failing.

use super::degrade_notices::counts;
use super::depth_compares::{argument_kernel, pipeline_of, run_held, Held, Through};
use super::{
    at_plan, layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache,
};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::VkProperty;
use metal2vulkan::metal::{CompareFunction, PixelFormat, SamplerMipFilter};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{Degradation, Device, Region, SamplerState, Texture, TextureDescriptor};

/// The texture's side and levels.
const SIDE: u64 = 16;
const LEVELS: u32 = 5;

/// Each level's `Depth16Unorm` code: distinct, and each its own reference's code.
const CODES: [u16; LEVELS as usize] = [4096, 12288, 20480, 40960, 61440];

/// The level clamps `(lodMinClamp, lodMaxClamp)` of the nearest mip filter's samplers: none,
/// whole and half-way bounds, a bound of a fraction one sixty-fourth past one half, bounds
/// truncated onto the grid, an empty range of one level, and one past the texture's levels.
const CLAMPS: [(f32, f32); 9] = [
    (0.0, f32::MAX),
    (0.75, 2.5),
    (0.3, 2.7),
    (1.5, 1.5),
    (1.51, 2.51),
    (1.515625, 3.484375),
    (0.0, 0.0),
    (4.0, 100.0),
    (0.984375, 1.0078125),
];

/// How a kernel's `sample_compare` states its level: the lane's level, a kernel's bias of it (the
/// explicit level), or neither (level 0); as the call's level flag and value.
const STATED: [(&str, (&str, &str)); 3] = [
    ("level", ("true", "%lod")),
    ("bias", ("false", "%lod")),
    ("neither", ("false", "0.0")),
];

/// Synthetic AIR written for this test: `k` compares `depth2d<float>` slot 0 through the
/// constexpr sampler of `words` (runtime sampler slot 0 for `None`) at buffer 0's lane `tid` (a
/// `float4`: the coordinate, the reference in `z`, a level in `w`) at the level `lod` states
/// ([`STATED`]), writing the pass to `x` of buffer 1's lane `tid` (a `float4`).
fn slot_kernel(words: Option<u64>, (flag, level): (&str, &str)) -> String {
    let (global, param, operand, states, args, tid) = match words {
        Some(w) => (
            format!(
                "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {w}, i64 0], align 8\n"
            ),
            "",
            "@__air_sampler_state",
            "!air.sampler_states = !{!6}\n!6 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
            "!3, !4, !5, !7",
            3,
        ),
        None => (
            String::new(),
            ", ptr addrspace(2) %s",
            "%s",
            "!6 = !{i32 3, !\"air.sampler\", !\"air.location_index\", i32 0, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"s\"}\n",
            "!3, !4, !5, !6, !7",
            4,
        ),
    };
    format!(
        r#"{global}
define void @k(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out{param}, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i
  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16
  %uv = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %ref = extractelement <4 x float> %c4, i64 2
  %lod = extractelement <4 x float> %c4, i64 3
  %r = call {{ float, i8 }} @air.sample_compare_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) {operand}, i32 1, <2 x float> %uv, float %ref, i1 true, <2 x i32> zeroinitializer, i1 {flag}, float {level}, float 0.0, i32 0)
  %v = extractvalue {{ float, i8 }} %r, 0
  %v4 = insertelement <4 x float> zeroinitializer, float %v, i64 0
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v4, ptr addrspace(1) %q, align 16
  ret void
}}

declare {{ float, i8 }} @air.sample_compare_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, float, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{{!0}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{{args}}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"depth2d<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!7 = !{{i32 {tid}, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The words of a constexpr sampler of nearest filters, mip filter `mip` (AIR code: 1 nearest, 2
/// linear), clamping to the edge with normalized coordinates, comparing `equal`, with level clamp
/// `(min, max)` as `half` bits.
fn constexpr_words(mip: u64, (min, max): (u16, u16)) -> u64 {
    1 | 1 << 3 | 1 << 6 | mip << 13 | 5 << 16 | u64::from(min) << 24 | u64::from(max) << 40
}

/// The level clamps `(lod_clamp(min, max))` of the constexpr samplers, as `half` bits: none (the
/// greatest `half`), whole and half-way bounds, a bound of a sixty-fourth's fraction, and an empty
/// range of one level.
const CONSTEXPR_CLAMPS: [(u16, u16); 4] = [
    (0x0000, 0x7bff),
    (0x3a00, 0x4100),
    (0x3c10, 0x4200),
    (0x3e00, 0x3e00),
];

/// `x` rounded to `half`, to nearest even (an infinity, a NaN and a value past `half`'s range as
/// `half` holds them).
fn to_half(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let a = f64::from(x.abs());
    if a >= 65520.0 {
        return f32::INFINITY.copysign(x);
    }
    let exponent = ((x.abs().to_bits() >> 23) & 0xff) as i32 - 127;
    let ulp = 2f64.powi(exponent.max(-14) - 10);
    ((a / ulp).round_ties_even() * ulp).copysign(f64::from(x)) as f32
}

/// The weight of each level in a comparison at level `x` (`None`: level 0) through a nearest
/// sampler of mip filter `mip` and level clamp `(min, max)` (module docs): with `L + q` the level
/// clamped and truncated, a nearest mip filter's level `L + 1` where `q > 1/2`, else `L`, a linear
/// one's `L` and `L + 1` weighted `1 - q` and `q` as Metal blends the passes (onto sixteenths of
/// an 8-bit code), and level 0 without a mip filter.
fn metal_weights(
    x: Option<f32>,
    mip: SamplerMipFilter,
    (min, max): (f32, f32),
) -> [f32; LEVELS as usize] {
    let mut w = [0.0; LEVELS as usize];
    if mip == SamplerMipFilter::NotMipmapped {
        w[0] = 1.0;
        return w;
    }
    let x = match x.map(to_half) {
        Some(x) if !x.is_nan() => x,
        _ => 0.0,
    };
    let x = x.max(min).min(max).clamp(0.0, (LEVELS - 1) as f32);
    let whole = x.floor();
    let q = ((x - whole) * 64.0).floor() / 64.0;
    let l = whole as usize;
    match mip {
        // The passes as `r8Unorm` codes 0 and 255, blended and rounded once onto sixteenths of a
        // code, ties up: one level passing alone weighs `(255 (64 - m) + 2) / 4` or
        // `(255 m + 2) / 4` sixteenths of a code (q = m / 64).
        SamplerMipFilter::Linear => {
            let m = (q * 64.0) as u32;
            w[l] = ((255 * (64 - m) + 2) / 4) as f32 / 4080.0;
            if m > 0 {
                w[l + 1] = ((255 * m + 2) / 4) as f32 / 4080.0;
            }
        }
        _ => w[l + usize::from(q > 0.5)] = 1.0,
    }
    w
}

/// The value of `half` bits `h` (a finite `half`).
fn half_value(h: u16) -> f32 {
    let exponent = i32::from((h >> 10) & 0x1f);
    let fraction = f32::from(h & 0x3ff);
    match exponent {
        0 => fraction * 2f32.powi(-24),
        e => (1.0 + fraction / 1024.0) * 2f32.powi(e - 15),
    }
}

/// The levels the jobs state: every sixty-fourth in [-1, 5], the neighbourhoods of each `L + 1/2`
/// in steps of 1/4096, past the texture's levels, the infinities, and a NaN.
fn lane_levels() -> Vec<f32> {
    let mut levels: Vec<f32> = (0..=6 * 64).map(|m| -1.0 + m as f32 / 64.0).collect();
    for l in 0..4 {
        levels.extend((-32..=32).map(|m| l as f32 + 0.5 + m as f32 / 4096.0));
    }
    levels.extend([100.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);
    levels
}

/// The mipmapped texture in `format` (module docs), each level's texels its depth.
fn texture(device: &Device, format: PixelFormat) -> Texture {
    let texture = device
        .new_texture(&TextureDescriptor::texture_2d(format, SIDE, SIDE, true))
        .unwrap();
    assert_eq!(texture.mipmap_level_count(), u64::from(LEVELS));
    for l in 0..LEVELS {
        let n = SIDE >> l;
        let (bytes, row): (Vec<u8>, u64) = match format {
            PixelFormat::Depth16Unorm => (
                CODES[l as usize].to_le_bytes().repeat((n * n) as usize),
                2 * n,
            ),
            _ => (
                depth(format, l).to_le_bytes().repeat((n * n) as usize),
                4 * n,
            ),
        };
        texture
            .replace_region(Region::new_2d(0, 0, n, n), u64::from(l), 0, &bytes, row, 0)
            .unwrap();
    }
    texture
}

/// Level `l`'s depth in `format`, and the reference that compares `equal` to it alone.
fn depth(format: PixelFormat, l: u32) -> f32 {
    match format {
        PixelFormat::Depth16Unorm => f32::from(CODES[l as usize]) / 65535.0,
        _ => (l + 1) as f32 / 8.0,
    }
}

/// A runtime sampler of nearest filters, mip filter `mip` and level clamp `clamp`, comparing
/// `equal`.
fn sampler(device: &Device, mip: SamplerMipFilter, (min, max): (f32, f32)) -> SamplerState {
    let mut d = SamplerDescriptor::new();
    d.mip_filter = mip;
    d.lod_min_clamp = min;
    d.lod_max_clamp = max;
    d.compare_function = CompareFunction::Equal;
    device.new_sampler_state(&d).unwrap()
}

/// Every comparison through a runtime sampler slot (at a level, a kernel's bias of it, and
/// neither) and through an argument buffer's sampler and a runtime sampler of a texture an
/// argument buffer holds (at a level) reads the level Metal reads, and every one through a
/// constexpr sampler under a nearest or a linear mip filter (at a level, a bias, and neither)
/// reads or blends Metal's levels, on `Depth32Float` and `Depth16Unorm` (module docs), on a device
/// that samples through Metal's own sampler; elsewhere the stages announce `device-sampling` and
/// the comparisons that differ are printed.
#[test]
fn comparisons_read_the_level_metal_reads() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        let own = !device
            .compile_device()
            .profile()
            .enabled()
            .properties
            .contains(&VkProperty::MetalSampler);
        let levels = lane_levels();
        let mut compared = 0usize;
        let mut wrong = Vec::new();
        for format in [PixelFormat::Depth32Float, PixelFormat::Depth16Unorm] {
            let texture = texture(&device, format);
            let mut jobs = Vec::new();
            for &x in &levels {
                for k in 0..LEVELS {
                    jobs.push([0.5, 0.5, depth(format, k), x]);
                }
            }
            // Each answer of `out` against level `k`'s weight among `weights` of its job's level.
            let mut judge = |what: &str, out: &[f32], weights: &dyn Fn(f32) -> [f32; 5]| {
                for (j, job) in jobs.iter().enumerate() {
                    let x = job[3];
                    let k = j % LEVELS as usize;
                    let want = weights(x)[k];
                    let got = out[4 * j];
                    compared += 1;
                    if got.to_bits() != want.to_bits() {
                        wrong.push(format!(
                            "{format:?} {what} level {x:e}: level {k}'s depth {got:e}, want \
                             {want:e}"
                        ));
                    }
                }
            };
            let configs = CLAMPS
                .iter()
                .map(|&c| (SamplerMipFilter::Nearest, c))
                .chain([(SamplerMipFilter::NotMipmapped, (0.0, f32::MAX))]);
            for (mip, clamp) in configs {
                let s = sampler(&device, mip, clamp);
                let mut runs = Vec::new();
                for (stated, lod) in STATED {
                    let state = pipeline_of(
                        &device,
                        &llvm_assemble_text(&slot_kernel(None, lod)).unwrap(),
                        "k",
                    );
                    let held = (&texture, Held::Slot, Some(&s));
                    let out = run_held(&device, held, &state, &jobs, 1);
                    runs.push((format!("slot {stated}"), stated != "neither", out));
                }
                for (what, through, held) in [
                    (
                        "argument buffer's sampler",
                        Through::Argument,
                        Held::Argument(Some(&s)),
                    ),
                    (
                        "runtime sampler of an argument buffer's texture",
                        Through::Runtime,
                        Held::Argument(None),
                    ),
                ] {
                    let bitcode =
                        llvm_assemble_text(&argument_kernel(through, false, ("true", "%lod")))
                            .unwrap();
                    let state = pipeline_of(&device, &bitcode, "k");
                    let sampler = matches!(through, Through::Runtime).then_some(&s);
                    let out = run_held(&device, (&texture, held, sampler), &state, &jobs, 1);
                    runs.push((what.to_owned(), true, out));
                }
                for (what, leveled, out) in runs {
                    let what = format!("{what} {mip:?} clamp {clamp:?}");
                    judge(&what, &out, &|x| {
                        metal_weights(leveled.then_some(x), mip, clamp)
                    });
                }
            }
            // Constexpr samplers of nearest filters under a nearest and a linear mip filter, their
            // clamps `half`s.
            for (code, mip) in [
                (1, SamplerMipFilter::Nearest),
                (2, SamplerMipFilter::Linear),
            ] {
                for (min, max) in CONSTEXPR_CLAMPS {
                    let words = constexpr_words(code, (min, max));
                    let clamp = (half_value(min), half_value(max));
                    for (stated, lod) in STATED {
                        let state = pipeline_of(
                            &device,
                            &llvm_assemble_text(&slot_kernel(Some(words), lod)).unwrap(),
                            "k",
                        );
                        let out = run_held(&device, (&texture, Held::Slot, None), &state, &jobs, 1);
                        let what = format!("constexpr {stated} {mip:?} clamp {clamp:?}");
                        let leveled = stated != "neither";
                        judge(&what, &out, &|x| {
                            metal_weights(leveled.then_some(x), mip, clamp)
                        });
                    }
                }
            }
        }
        eprintln!("{name}: {} of {compared} comparisons differ", wrong.len());
        let announced = counts(&messages.take_notices());
        assert_eq!(
            announced.contains_key(&Degradation::DeviceSampling),
            own,
            "{name}: device-sampling exactly where the device's sampler is not Metal's"
        );
        if own {
            for w in &wrong {
                eprintln!("DEPTH-LEVEL-DIFFERS {name}: {w}");
            }
        } else {
            assert!(
                wrong.is_empty(),
                "{name}: {} of {compared} comparisons differ:\n{}",
                wrong.len(),
                wrong
                    .iter()
                    .take(200)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}
