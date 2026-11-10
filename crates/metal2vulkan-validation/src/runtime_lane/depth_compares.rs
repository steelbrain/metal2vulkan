//! A device test of depth comparisons against Metal's measured rule on the device the cfg selects,
//! under the validation layer: Metal's own through MoltenVK (`VkProperty::MetalSampler`), which
//! must answer it exactly; elsewhere the device's own comparisons, which the stages record
//! (`PrecisionNote::DeviceSampling`) and the lane counts against the rule
//! (`DEPTH-COMPARE-DIFFERS` lines) without failing.
//!
//! Metal compares a `Depth16Unorm` texel against the reference's code: the reference clamped into
//! [0, 1] and the exact product with 65535 rounded to the nearest code (the one half-way
//! reference, 0.5, up), a NaN failing (for cube and 2D gathers, for `sample_compare`, Apple M4 on
//! macOS 27.0.1). It compares a `Depth32Float` texel and the reference as given, in `f32`: neither
//! clamped, `-0` equal to `+0`, the infinities in order, tiny values kept, a NaN failing. Vulkan
//! clamps a UNORM format's `Dref` into [0, 1] and leaves the rest to the device.
//!
//! The kernels are `msf_compare_filter`'s (`validation/fixtures/public/engine`), over a 4x4
//! texture of distinct texels in each format, through constexpr samplers that clamp to the edge:
//! `msf_cmp_sample2d` at each texel's centre, where the linear filter weighs that texel alone, so
//! its linear and nearest comparisons are that texel's; and `msf_cmp_gather2d` at each inner
//! corner, whose plain gather states the four texels and whose comparison gathers compare each.
//! Each reference sits around a texel: a `Depth16Unorm` code's at fractions of a step and at the
//! `f32` neighbours of the half-way points, a `Depth32Float` value's at it and its `f32`
//! neighbours; and outside [0, 1], at the infinities and at a NaN.
//!
//! The same comparisons run through a runtime sampler ([`RUNTIME`]'s kernels), whose compare
//! function the shader reads from the sampler's word: `sample_compare` through a linear and a
//! nearest sampler and `gather_compare`, under each of the eight compare functions, against the
//! same rule (a reference below, equal to or above the texel, or unordered).
//!
//! And through an argument buffer that holds the texture ([`argument_kernel`]'s kernels), whose
//! depth class the shader reads from the texture's handle: `sample_compare` and `gather_compare`
//! through the constexpr nearest samplers under `less_equal` and `equal`, and through a nearest
//! runtime sampler and a nearest sampler the argument buffer holds beside the texture under each of
//! the eight functions. (A sampler that filters leaves a heap texture's comparison to the device,
//! which the module records, so it is not asked here.)
//!
//! A device whose uploads clamp a depth outside [0, 1] (`degrade_notices::clamps_depth_uploads`:
//! without `VK_EXT_depth_range_unrestricted`, MoltenVK's included) holds the `Depth32Float`
//! texels as its upload gives them ([`CLAMPED`]) and announces `restricted-depth-upload`; each
//! comparison is checked against the texel the texture holds.

use super::degrade_notices::{clamps_depth_uploads, counts};
use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with,
    Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::VkProperty;
use metal2vulkan::metal::{CompareFunction, PixelFormat, ResourceUsage, SamplerMinMagFilter};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble, llvm_assemble_text};
use metal2vulkan_runtime::{
    ComputePipelineState, Degradation, Device, Error, PipelineOptions, Region, ResourceOptions,
    SamplerState, Size, Texture, TextureDescriptor,
};
use std::cmp::Ordering;

/// The texture's side.
pub(super) const SIDE: u32 = 4;

/// The `Depth16Unorm` texels' codes, row by row: the ends, the middle, and codes between.
const CODES: [u16; 16] = [
    0, 1, 2, 16384, 32767, 32768, 32769, 49151, 65533, 65534, 65535, 100, 200, 300, 40000, 50000,
];

/// The `Depth32Float` texels, row by row: values in [0, 1], outside it, a negative zero, a
/// denormal, and neighbours of a half and of one.
pub(super) const VALUES: [f32; 16] = [
    0.0,
    0.25,
    0.5,
    1.0,
    -1.0,
    2.0,
    1000.0,
    -0.0,
    1e-40,
    f32::MIN_POSITIVE,
    0.1,
    0.3,
    0.7,
    0.9,
    0.500_000_06,
    0.999_999_94,
];

/// The references around a code `c`: `(c + f) / 65535` for each `f`, in steps.
const FRACTIONS: [f64; 13] = [
    -1.0, -0.75, -0.5, -0.49, -0.25, -0.01, 0.0, 0.01, 0.25, 0.49, 0.5, 0.75, 1.0,
];

/// References compared against every texel.
const SPECIALS: [f32; 12] = [
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    -0.0,
    0.0,
    1.0,
    -1.0,
    2.0,
    1.000_000_1,
    -1e-30,
    1e-30,
    f32::MIN_POSITIVE,
];

/// `msf_cmp_sample2d`'s four comparisons, in its order.
const SAMPLED: [&str; 4] = [
    "linear less_equal",
    "linear equal",
    "nearest less_equal",
    "nearest equal",
];

/// The threads of a threadgroup; each grid is padded to whole threadgroups.
const GROUP: usize = 64;

/// The eight compare functions, each a runtime sampler's.
const FUNCTIONS: [CompareFunction; 8] = [
    CompareFunction::Never,
    CompareFunction::Less,
    CompareFunction::Equal,
    CompareFunction::LessEqual,
    CompareFunction::Greater,
    CompareFunction::NotEqual,
    CompareFunction::GreaterEqual,
    CompareFunction::Always,
];

/// Synthetic AIR written for this test: `msf_rt_cmp_sample2d` and `msf_rt_cmp_gather2d` read
/// `depth2d<float>` slot 0 through runtime sampler slot 0 at buffer 0's lane `tid` (a `float4`:
/// the coordinate, and the reference in `z`). The first writes the `sample_compare` at level 0 to
/// `x` of buffer 1's lane `tid` (a `float4`), the second the four passes of the `gather_compare`.
const RUNTIME: &str = r#"define void @msf_rt_cmp_sample2d(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out, ptr addrspace(2) %s, i32 %tid) {
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i
  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16
  %uv = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %ref = extractelement <4 x float> %c4, i64 2
  %r = call { float, i8 } @air.sample_compare_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) %s, i32 1, <2 x float> %uv, float %ref, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)
  %v = extractvalue { float, i8 } %r, 0
  %v4 = insertelement <4 x float> zeroinitializer, float %v, i64 0
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v4, ptr addrspace(1) %q, align 16
  ret void
}

define void @msf_rt_cmp_gather2d(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out, ptr addrspace(2) %s, i32 %tid) {
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i
  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16
  %uv = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %ref = extractelement <4 x float> %c4, i64 2
  %r = call { <4 x float>, i8 } @air.gather_compare_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) %s, i32 1, <2 x float> %uv, float %ref, i1 true, <2 x i32> zeroinitializer, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v, ptr addrspace(1) %q, align 16
  ret void
}

declare { float, i8 } @air.sample_compare_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, float, i1, <2 x i32>, i1, float, float, i32)
declare { <4 x float>, i8 } @air.gather_compare_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, float, i1, <2 x i32>, i32)

!air.kernel = !{!0, !10}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @msf_rt_cmp_sample2d, !1, !2}
!10 = !{ptr @msf_rt_cmp_gather2d, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"depth2d<float, sample>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!6 = !{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!7 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The constexpr samplers `near_le` and `near_eq` of `msf_compare_filter.metal` (nearest filters,
/// no mip filter, clamped to the edge, comparing under `less_equal` and `equal`), as Apple's
/// compiler writes their words.
const NEAREST: [(u64, CompareFunction); 2] = [
    (0x007b_ff00_0002_0049, CompareFunction::LessEqual),
    (0x007b_ff00_0005_0049, CompareFunction::Equal),
];

/// What [`argument_kernel`]'s kernel compares through: a constexpr sampler of these words, runtime
/// sampler slot 0, or the sampler its argument buffer holds beside the texture.
#[derive(Clone, Copy, Debug)]
pub(super) enum Through {
    Constexpr(u64),
    Runtime,
    Argument,
}

/// Synthetic AIR written for this test: kernel `k` reads the `depth2d<float>` its argument buffer
/// (buffer 2, `{ depth2d<float> d; sampler s; }`) holds through `through` at buffer 0's lane `tid`
/// (a `float4`: the coordinate, the reference in `z`, and a level in `w`): a `sample_compare` at
/// the level `lod` states (the call's level flag and value: `("true", "0.0")` for level 0,
/// `("true", "%lod")` for the lane's level, `("false", "%lod")` for a kernel's bias of it) written
/// to `x` of buffer 1's lane `tid` (a `float4`), or where `gather` the four passes of a
/// `gather_compare`.
pub(super) fn argument_kernel(through: Through, gather: bool, lod: (&str, &str)) -> String {
    let (flag, level) = lod;
    let (global, states) = match through {
        Through::Constexpr(words) => (
            format!(
                "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8\n"
            ),
            "!air.sampler_states = !{!11}\n!11 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
        ),
        _ => (String::new(), ""),
    };
    let (param, operand, args, tid) = match through {
        Through::Constexpr(_) => ("", "@__air_sampler_state", "!3, !4, !5, !7", 3),
        Through::Runtime => (", ptr addrspace(2) %rs", "%rs", "!3, !4, !5, !6, !7", 4),
        Through::Argument => ("", "%s", "!3, !4, !5, !7", 3),
    };
    let load = match through {
        Through::Argument => "  %ps = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 1\n  %s = load ptr addrspace(2), ptr addrspace(2) %ps, align 8\n",
        _ => "",
    };
    let (call, declare) = match gather {
        true => (
            format!(
                "  %r = call {{ <4 x float>, i8 }} @air.gather_compare_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) {operand}, i32 1, <2 x float> %uv, float %ref, i1 true, <2 x i32> zeroinitializer, i32 0)\n  %v = extractvalue {{ <4 x float>, i8 }} %r, 0\n"
            ),
            "declare { <4 x float>, i8 } @air.gather_compare_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, float, i1, <2 x i32>, i32)",
        ),
        false => (
            format!(
                "  %r = call {{ float, i8 }} @air.sample_compare_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) {operand}, i32 1, <2 x float> %uv, float %ref, i1 true, <2 x i32> zeroinitializer, i1 {flag}, float {level}, float 0.0, i32 0)\n  %x = extractvalue {{ float, i8 }} %r, 0\n  %v = insertelement <4 x float> zeroinitializer, float %x, i64 0\n"
            ),
            "declare { float, i8 } @air.sample_compare_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, float, i1, <2 x i32>, i1, float, float, i32)",
        ),
    };
    format!(
        r#"%struct.M = type {{ ptr addrspace(1), ptr addrspace(2) }}
{global}
define void @k(ptr addrspace(1) %c, ptr addrspace(1) %out, ptr addrspace(2) %m{param}, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i
  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16
  %uv = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %ref = extractelement <4 x float> %c4, i64 2
  %lod = extractelement <4 x float> %c4, i64 3
  %pt = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
{load}{call}  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v, ptr addrspace(1) %q, align 16
  ret void
}}

{declare}

!air.kernel = !{{!0}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{{args}}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !8, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"M", !"air.arg_name", !"m"}}
!6 = !{{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"rs"}}
!7 = !{{i32 {tid}, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
!8 = !{{i32 0, i32 8, i32 0, !"depth2d<float, sample>", !"d", !"air.indirect_argument", !9, i32 8, i32 8, i32 0, !"sampler", !"s", !"air.indirect_argument", !10}}
!9 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"depth2d<float, sample>", !"air.arg_name", !"d"}}
!10 = !{{i32 1, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// A probed texture's texels, as Metal holds them.
#[derive(Clone, Copy)]
enum Texels {
    Unorm16(&'static [u16; 16]),
    Float32(&'static [f32; 16]),
}

impl Texels {
    fn format(self) -> PixelFormat {
        match self {
            Texels::Unorm16(_) => PixelFormat::Depth16Unorm,
            Texels::Float32(_) => PixelFormat::Depth32Float,
        }
    }

    /// The texels' bytes, row by row, and the bytes of a row.
    fn bytes(self) -> (Vec<u8>, u64) {
        match self {
            Texels::Unorm16(c) => (
                c.iter().flat_map(|c| c.to_le_bytes()).collect(),
                2 * u64::from(SIDE),
            ),
            Texels::Float32(v) => (
                v.iter().flat_map(|v| v.to_le_bytes()).collect(),
                4 * u64::from(SIDE),
            ),
        }
    }

    /// Texel `i`, shown.
    fn shown(self, i: usize) -> String {
        match self {
            Texels::Unorm16(c) => format!("code {}", c[i]),
            Texels::Float32(v) => format!("{:e}", v[i]),
        }
    }

    /// Whether reference `r` passes against texel `i`, under `less_equal` and under `equal`, as
    /// Metal compares them: `1.0` where it passes, else `0.0`.
    fn passes(self, r: f32, i: usize) -> [f32; 2] {
        [
            self.passes_under(CompareFunction::LessEqual, r, i),
            self.passes_under(CompareFunction::Equal, r, i),
        ]
    }

    /// Whether reference `r` passes against texel `i` under `f`, as Metal compares them: the
    /// function's raw value's bits 0, 1 and 2 pass a reference below, equal to and above the
    /// texel (a `Depth16Unorm` texel's code against the reference's, a `Depth32Float` texel's
    /// value against the reference, `-0` equal to `+0`), and a NaN reference passes `not_equal`
    /// and `always` alone: `1.0` where it passes, else `0.0`.
    fn passes_under(self, f: CompareFunction, r: f32, i: usize) -> f32 {
        let order = match self {
            Texels::Unorm16(c) => code(r).map(|k| k.cmp(&u32::from(c[i]))),
            Texels::Float32(v) => r.partial_cmp(&v[i]),
        };
        let bits = f.as_raw();
        let pass = match order {
            Some(Ordering::Less) => bits & 1 != 0,
            Some(Ordering::Equal) => bits & 2 != 0,
            Some(Ordering::Greater) => bits & 4 != 0,
            None => bits & 5 == 5,
        };
        if pass {
            1.0
        } else {
            0.0
        }
    }

    /// Whether a plain gather's lane `x` holds texel `i`.
    fn holds(self, x: f32, i: usize) -> bool {
        match self {
            Texels::Unorm16(c) => (f64::from(x) * 65535.0).round() == f64::from(c[i]),
            Texels::Float32(v) => x == v[i],
        }
    }

    /// The references around texel `i`: a code's at each fraction of a step and the two `f32`
    /// values on each side of each half-way point; a value's at it and its two neighbours on each
    /// side.
    fn references(self, i: usize) -> Vec<f32> {
        match self {
            Texels::Unorm16(c) => {
                let at = |f: f64| ((f64::from(c[i]) + f) / 65535.0) as f32;
                let mut refs: Vec<f32> = FRACTIONS.iter().map(|&f| at(f)).collect();
                for h in [-0.5, 0.5] {
                    let r = at(h);
                    refs.extend([
                        r.next_down().next_down(),
                        r.next_down(),
                        r.next_up(),
                        r.next_up().next_up(),
                    ]);
                }
                refs
            }
            Texels::Float32(v) => {
                let v = v[i];
                vec![
                    v,
                    v.next_down().next_down(),
                    v.next_down(),
                    v.next_up(),
                    v.next_up().next_up(),
                ]
            }
        }
    }
}

/// Metal's code of reference `r` for a `Depth16Unorm` texel: `r` clamped into [0, 1] and the
/// exact product with 65535 (exact in `f64`: 24 and 16 significant bits) rounded to nearest, the
/// half-way 0.5 up; none for a NaN.
fn code(r: f32) -> Option<u32> {
    if r.is_nan() {
        return None;
    }
    let p = f64::from(r.clamp(0.0, 1.0)) * 65535.0;
    Some((p + 0.5).floor() as u32)
}

/// The index of texel `(x, y)`.
fn index(x: u32, y: u32) -> usize {
    (y * SIDE + x) as usize
}

/// The kernel `name` of the fixture's library (or of [`RUNTIME`]'s, where `runtime`) on `device`.
pub(super) fn pipeline(device: &Device, name: &str, runtime: bool) -> ComputePipelineState {
    let bitcode = match runtime {
        true => llvm_assemble_text(RUNTIME).unwrap(),
        false => llvm_assemble(&engine_dir().join("msf_compare_filter.ll")).unwrap(),
    };
    pipeline_of(device, &bitcode, name)
}

/// The kernel `name` of the library of `bitcode` on `device`.
pub(super) fn pipeline_of(device: &Device, bitcode: &[u8], name: &str) -> ComputePipelineState {
    made_pipeline_of(device, bitcode, name).unwrap()
}

/// The kernel `name` of the library of `bitcode` on `device`, or the state's refusal.
pub(super) fn made_pipeline_of(
    device: &Device,
    bitcode: &[u8],
    name: &str,
) -> Result<ComputePipelineState, Error> {
    let library = device.new_library_with_air(bitcode).unwrap();
    let function = library.new_function_with_name(name).unwrap();
    device.new_compute_pipeline_state(
        &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
        &PipelineOptions::default(),
    )
}

/// The `float4` words `out` of kernel `state` reading `texture` at texture slot 0 (through
/// runtime sampler slot 0's `sampler`, where given) and `jobs` (each a coordinate and a
/// reference), `per_job` words per job.
pub(super) fn run(
    device: &Device,
    (texture, sampler): (&Texture, Option<&SamplerState>),
    state: &ComputePipelineState,
    jobs: &[[f32; 4]],
    per_job: usize,
) -> Vec<f32> {
    run_held(device, (texture, Held::Slot, sampler), state, jobs, per_job)
}

/// Where a kernel reads its texture: at texture slot 0, or from the argument buffer at buffer 2
/// (`{ depth2d<float> d; sampler s; }`, [`argument_kernel`]), with sampler `s` where given.
#[derive(Clone, Copy)]
pub(super) enum Held<'a> {
    Slot,
    Argument(Option<&'a SamplerState>),
}

/// [`run`], the texture read where `held` says.
pub(super) fn run_held(
    device: &Device,
    (texture, held, sampler): (&Texture, Held<'_>, Option<&SamplerState>),
    state: &ComputePipelineState,
    jobs: &[[f32; 4]],
    per_job: usize,
) -> Vec<f32> {
    let padded = jobs.len().div_ceil(GROUP) * GROUP;
    let mut bytes: Vec<u8> = jobs
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    bytes.resize(padded * 16, 0);
    let input = buffer_with(device, &bytes).unwrap();
    let length = (padded * per_job * 16) as u64;
    let output = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    // An argument buffer names the texture and its sampler by their resource IDs; it lives until
    // the command buffer completes.
    let _argument = match held {
        Held::Slot => {
            encoder.set_texture(Some(texture), 0).unwrap();
            None
        }
        Held::Argument(s) => {
            let ids = [
                texture.gpu_resource_id().0,
                s.map_or(0, |s| s.gpu_resource_id().0),
            ];
            let words: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            let argument = buffer_with(device, &words).unwrap();
            encoder
                .use_resource(texture.as_resource(), ResourceUsage::READ)
                .unwrap();
            encoder.set_buffer(Some(&argument), 0, 2).unwrap();
            Some(argument)
        }
    };
    if sampler.is_some() {
        encoder.set_sampler_state(sampler, 0).unwrap();
    }
    encoder.set_buffer(Some(&input), 0, 0).unwrap();
    encoder.set_buffer(Some(&output), 0, 1).unwrap();
    encoder
        .dispatch_threads(
            Size::new(padded as u64, 1, 1),
            Size::new(GROUP as u64, 1, 1),
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    read_buffer(&output, 0, length)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// A reference as its value and bits, with its code where `texels` compare codes.
fn shown(texels: Texels, r: f32) -> String {
    match texels {
        Texels::Unorm16(_) => format!("{r:e} ({:#010x}, code {:?})", r.to_bits(), code(r)),
        Texels::Float32(_) => format!("{r:e} ({:#010x})", r.to_bits()),
    }
}

/// Each comparison of `texels` on `device`, and the ones that are not Metal's.
fn compare(device: &Device, texels: Texels) -> (usize, Vec<String>) {
    let format = texels.format();
    let side = u64::from(SIDE);
    let texture = device
        .new_texture(&TextureDescriptor::texture_2d(format, side, side, false))
        .unwrap();
    let (bytes, row) = texels.bytes();
    texture
        .replace_region(Region::new_2d(0, 0, side, side), 0, 0, &bytes, row, 0)
        .unwrap();
    let mut wrong = Vec::new();
    let mut compared = 0usize;

    // sample_compare at each texel's centre: (linear less_equal, linear equal, nearest
    // less_equal, nearest equal).
    let (jobs, want) = sample_jobs(texels);
    let state = pipeline(device, "msf_cmp_sample2d", false);
    let out = run(device, (&texture, None), &state, &jobs, 1);
    for (j, &(i, r)) in want.iter().enumerate() {
        let [le, eq] = texels.passes(r, i);
        for (k, w) in [le, eq, le, eq].into_iter().enumerate() {
            let got = out[4 * j + k];
            compared += 1;
            if got.to_bits() != w.to_bits() {
                wrong.push(format!(
                    "{format:?} sample_compare {}: texel {}, reference {}: {got:e}, want {w:e}",
                    SAMPLED[k],
                    texels.shown(i),
                    shown(texels, r)
                ));
            }
        }
    }

    // gather and gather_compare at each inner corner.
    let (jobs, want) = gather_jobs(texels);
    let state = pipeline(device, "msf_cmp_gather2d", false);
    let out = run(device, (&texture, None), &state, &jobs, 3);
    for (j, &(taps, r)) in want.iter().enumerate() {
        for (lane, &i) in taps.iter().enumerate() {
            let gathered = out[12 * j + lane];
            if !texels.holds(gathered, i) {
                wrong.push(format!(
                    "{format:?} gather lane {lane}: {gathered:e}, want texel {}",
                    texels.shown(i)
                ));
            }
            let [le, eq] = texels.passes(r, i);
            for (k, w) in [(1, le), (2, eq)] {
                let got = out[12 * j + 4 * k + lane];
                compared += 1;
                if got.to_bits() != w.to_bits() {
                    let f = if k == 1 { "less_equal" } else { "equal" };
                    wrong.push(format!(
                        "{format:?} gather_compare {f} lane {lane}: texel {}, reference {}: \
                         {got:e}, want {w:e}",
                        texels.shown(i),
                        shown(texels, r)
                    ));
                }
            }
        }
    }

    // The same through a runtime sampler, under each compare function: sample_compare through a
    // linear and a nearest sampler, and gather_compare.
    let (jobs, want) = sample_jobs(texels);
    let state = pipeline(device, "msf_rt_cmp_sample2d", true);
    for (filter, name) in [
        (SamplerMinMagFilter::Linear, "linear"),
        (SamplerMinMagFilter::Nearest, "nearest"),
    ] {
        for f in FUNCTIONS {
            let sampler = runtime_sampler(device, filter, f);
            let out = run(device, (&texture, Some(&sampler)), &state, &jobs, 1);
            for (j, &(i, r)) in want.iter().enumerate() {
                let w = texels.passes_under(f, r, i);
                let got = out[4 * j];
                compared += 1;
                if got.to_bits() != w.to_bits() {
                    wrong.push(format!(
                        "{format:?} runtime {name} {f:?} sample_compare: texel {}, reference {}: \
                         {got:e}, want {w:e}",
                        texels.shown(i),
                        shown(texels, r)
                    ));
                }
            }
        }
    }
    let (jobs, want) = gather_jobs(texels);
    let state = pipeline(device, "msf_rt_cmp_gather2d", true);
    for f in FUNCTIONS {
        let sampler = runtime_sampler(device, SamplerMinMagFilter::Nearest, f);
        let out = run(device, (&texture, Some(&sampler)), &state, &jobs, 1);
        for (j, &(taps, r)) in want.iter().enumerate() {
            for (lane, &i) in taps.iter().enumerate() {
                let w = texels.passes_under(f, r, i);
                let got = out[4 * j + lane];
                compared += 1;
                if got.to_bits() != w.to_bits() {
                    wrong.push(format!(
                        "{format:?} runtime {f:?} gather_compare lane {lane}: texel {}, \
                         reference {}: {got:e}, want {w:e}",
                        texels.shown(i),
                        shown(texels, r)
                    ));
                }
            }
        }
    }
    let (n, w) = through_argument_buffer(device, texels, &texture);
    (compared + n, wrong.into_iter().chain(w).collect())
}

/// Each comparison of `texels` held in `texture` that kernels make through an argument buffer
/// holding it ([`argument_kernel`]), and the ones that are not Metal's: `sample_compare` and
/// `gather_compare` through the constexpr nearest samplers under `less_equal` and `equal`, and
/// through a nearest runtime sampler and a nearest sampler the argument buffer holds under each of
/// the eight functions.
fn through_argument_buffer(
    device: &Device,
    texels: Texels,
    texture: &Texture,
) -> (usize, Vec<String>) {
    let format = texels.format();
    let samples = sample_jobs(texels);
    let gathers = gather_jobs(texels);
    let mut compared = 0usize;
    let mut wrong = Vec::new();
    // The answers `out` to the jobs of a sample (its pass in `x` of its lane) or a gather (its
    // four passes in its lane) under `f`, each judged against Metal's rule.
    let mut judge = |out: &[f32], f: CompareFunction, gather: bool, what: &str| {
        let wants: Vec<(usize, usize, f32)> = match gather {
            false => samples
                .1
                .iter()
                .enumerate()
                .map(|(j, &(i, r))| (4 * j, i, r))
                .collect(),
            true => gathers
                .1
                .iter()
                .enumerate()
                .flat_map(|(j, &(taps, r))| (0..4).map(move |lane| (4 * j + lane, taps[lane], r)))
                .collect(),
        };
        let op = match gather {
            false => "sample_compare",
            true => "gather_compare",
        };
        for (at, i, r) in wants {
            let w = texels.passes_under(f, r, i);
            let got = out[at];
            compared += 1;
            if got.to_bits() != w.to_bits() {
                wrong.push(format!(
                    "{format:?} argument buffer {what} {f:?} {op} word {at}: texel {}, \
                     reference {}: {got:e}, want {w:e}",
                    texels.shown(i),
                    shown(texels, r)
                ));
            }
        }
    };
    let kernel = |through: Through, gather: bool| {
        let bitcode =
            llvm_assemble_text(&argument_kernel(through, gather, ("true", "0.0"))).unwrap();
        pipeline_of(device, &bitcode, "k")
    };
    for gather in [false, true] {
        let jobs = match gather {
            false => &samples.0,
            true => &gathers.0,
        };
        for (words, f) in NEAREST {
            let state = kernel(Through::Constexpr(words), gather);
            let held = (texture, Held::Argument(None), None);
            let out = run_held(device, held, &state, jobs, 1);
            judge(&out, f, gather, "constexpr nearest");
        }
        let runtime = kernel(Through::Runtime, gather);
        let argument = kernel(Through::Argument, gather);
        for f in FUNCTIONS {
            let sampler = runtime_sampler(device, SamplerMinMagFilter::Nearest, f);
            let held = (texture, Held::Argument(None), Some(&sampler));
            let out = run_held(device, held, &runtime, jobs, 1);
            judge(&out, f, gather, "runtime nearest");
            let held = (texture, Held::Argument(Some(&sampler)), None);
            let out = run_held(device, held, &argument, jobs, 1);
            judge(&out, f, gather, "argument nearest");
        }
    }
    (compared, wrong)
}

/// A runtime sampler of `filter` (minification and magnification) comparing under `f`, clamping
/// to the edge with normalized coordinates and no mip filter.
fn runtime_sampler(
    device: &Device,
    filter: SamplerMinMagFilter,
    f: CompareFunction,
) -> SamplerState {
    let mut d = SamplerDescriptor::new();
    d.min_filter = filter;
    d.mag_filter = filter;
    d.compare_function = f;
    device.new_sampler_state(&d).unwrap()
}

/// Jobs (each a coordinate and a reference), each with what Metal's answer to it depends on.
type Jobs<T> = (Vec<[f32; 4]>, Vec<T>);

/// [`VALUES`] as the upload of a device whose uploads clamp gives them to the depth image
/// (`bind::clamped_depth`): -1 as 0, 2 and 1000 as 1, every other one as it is.
const CLAMPED: [f32; 16] = {
    let mut clamped = VALUES;
    let mut i = 0;
    while i < VALUES.len() {
        clamped[i] = f32::from_bits(metal2vulkan::bind::clamped_depth(VALUES[i].to_bits()));
        i += 1;
    }
    clamped
};

/// The jobs of a `sample_compare` at each texel's centre, with each job's texel and reference:
/// every reference around the texel, and the special ones.
fn sample_jobs(texels: Texels) -> Jobs<(usize, f32)> {
    let step = 1.0 / SIDE as f32;
    let mut jobs = Vec::new();
    let mut want = Vec::new();
    for y in 0..SIDE {
        for x in 0..SIDE {
            let i = index(x, y);
            let (u, v) = ((x as f32 + 0.5) * step, (y as f32 + 0.5) * step);
            for r in texels.references(i).into_iter().chain(SPECIALS) {
                jobs.push([u, v, r, 0.0]);
                want.push((i, r));
            }
        }
    }
    (jobs, want)
}

/// The jobs of a gather at each inner corner, with each job's four taps' texels in gather order,
/// `(i0, j1)`, `(i1, j1)`, `(i1, j0)`, `(i0, j0)`, and its reference: every reference around each
/// tap's texel, and the special ones.
fn gather_jobs(texels: Texels) -> Jobs<([usize; 4], f32)> {
    let step = 1.0 / SIDE as f32;
    let mut jobs = Vec::new();
    let mut want = Vec::new();
    for y in 0..SIDE - 1 {
        for x in 0..SIDE - 1 {
            let taps = [
                index(x, y + 1),
                index(x + 1, y + 1),
                index(x + 1, y),
                index(x, y),
            ];
            let (u, v) = ((x + 1) as f32 * step, (y + 1) as f32 * step);
            let refs = taps.iter().flat_map(|&i| texels.references(i));
            for r in refs.chain(SPECIALS) {
                jobs.push([u, v, r, 0.0]);
                want.push((taps, r));
            }
        }
    }
    (jobs, want)
}

/// On a device that samples through Metal's own sampler, every comparison of a `Depth16Unorm`
/// and a `Depth32Float` texel is Metal's: `sample_compare` through a linear and a nearest sampler
/// at each texel's centre, and `gather_compare` at each inner corner, under `less_equal` and
/// `equal` through constexpr samplers and under each of the eight functions through a runtime
/// sampler; each plain gather states its footprint's four texels in Metal's order. And through an
/// argument buffer that holds the texture: through the constexpr nearest samplers, and through a
/// nearest runtime sampler and a nearest sampler the argument buffer holds under each of the eight
/// functions. Elsewhere the comparisons are the device's: the stages announce `device-sampling`,
/// and the comparisons that differ from Metal's are printed. Each is checked against the texel
/// the texture holds: [`CLAMPED`]'s where the device's uploads clamp, which announce
/// `restricted-depth-upload` exactly there.
#[test]
fn depth_comparisons_compare_as_metal_does() {
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
        let clamps = clamps_depth_uploads(&device);
        let float32 = match clamps {
            true => &CLAMPED,
            false => &VALUES,
        };
        let mut compared = 0;
        let mut wrong = Vec::new();
        for texels in [Texels::Unorm16(&CODES), Texels::Float32(float32)] {
            let (n, w) = compare(&device, texels);
            eprintln!(
                "{name}: {:?}: {} of {n} comparisons differ",
                texels.format(),
                w.len()
            );
            compared += n;
            wrong.extend(w);
        }
        let announced = counts(&messages.take_notices());
        assert_eq!(
            announced.contains_key(&Degradation::DeviceSampling),
            own,
            "{name}: device-sampling exactly where the device's sampler is not Metal's"
        );
        assert_eq!(
            announced.contains_key(&Degradation::RestrictedDepthUpload),
            clamps,
            "{name}: restricted-depth-upload exactly where the device's uploads clamp"
        );
        if own {
            for w in &wrong {
                eprintln!("DEPTH-COMPARE-DIFFERS {name}: {w}");
            }
        } else {
            assert!(
                wrong.is_empty(),
                "{name}: {} of {compared} comparisons differ:\n{}",
                wrong.len(),
                wrong.join("\n")
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}
