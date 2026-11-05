//! A texture an argument buffer holds, of a form a heap reads (`ExtendedRange::HEAP_READ`): the
//! binder holds it raw in the unsigned integer class of its shape (its tests in `bind::heap`), and
//! a function's float read or query of a sampled heap group reaches it there by the form its
//! handle's class index names.
//!
//! - On every committed profile with a sampled heap, a `float` or `half` read declares the float
//!   class and its unsigned integer twin, defines the decode routine once, stays inside the
//!   plan's word estimate, and passes L3; a query alone declares the twin and no routine.
//! - A read of an unsigned integer heap group declares no twin and no routine, its query its own
//!   class's; it declares the float class, which holds a member of a float format that it reads in
//!   that member's own class.
//! - The device lane `runtime_lane::heap_extended_reads` compares such reads and queries through
//!   an argument buffer with the same texture's through a slot, bit for bit, on each venue.
//! - A sample or gather through a constexpr sampler without a mip filter or anisotropy declares the
//!   twin and the routine and states no `RuntimeFact::HeapRawSamples`; a `half` gather and a
//!   runtime sampler's sample declare them and state the fact; a mip-filtered sampler's sample
//!   states it alone. The same lane compares such samples and gathers through an argument buffer
//!   with the slot's, and checks the bind refusal of the others.

use super::*;
use crate::error::{Error, Unsupported};
use crate::library::FunctionArtifact;
use crate::lower::texture::heap_raw;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{HandleClass, ImageAccess, ImageDim, ImageHandleClass, SampledType};
use crate::spirv::testing::committed_profiles;

/// What the kernel's texture holds and what it does with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Use {
    /// A `float` read and a width query.
    Float,
    /// A `half` read and a width query.
    Half,
    /// A width query alone, of a `texture2d<float>`.
    Query,
    /// A `uint` read and a width query.
    Uint,
}

/// Synthetic AIR written for this test: kernel `k` reads the `texture2d<T, access::read>` its
/// argument buffer (buffer 1, `{ texture2d<T, access::read> t; }`) holds at texel `(gid, 0)`
/// (but for [`Use::Query`]), storing the texel's first lane and its width to `out` (buffer 0).
fn kernel(u: Use) -> String {
    let (element, vec4, suffix) = match u {
        Use::Float | Use::Query => ("float", "<4 x float>", "v4f32"),
        Use::Half => ("half", "<4 x half>", "v4f16"),
        Use::Uint => ("uint", "<4 x i32>", "u.v4i32"),
    };
    let texture = format!("texture2d<{element}, read>");
    let first = match u {
        Use::Float => "  %x = extractelement <4 x float> %v, i64 0\n  %b = bitcast float %x to i32\n",
        Use::Half => "  %x = extractelement <4 x half> %v, i64 0\n  %e = fpext half %x to float\n  %b = bitcast float %e to i32\n",
        Use::Uint => "  %b = extractelement <4 x i32> %v, i64 0\n",
        Use::Query => "",
    };
    let read = match u {
        Use::Query => "  %b = add i32 0, 0\n".to_owned(),
        _ => format!(
            "  %at = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0\n  %rs = call ptr addrspace(2) @air.get_read_sampler()\n  %r = call {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %at, <2 x i32> zeroinitializer, i32 0, i32 1)\n  %v = extractvalue {{ {vec4}, i8 }} %r, 0\n{first}"
        ),
    };
    format!(
        r#"%struct.M = type {{ ptr addrspace(1) }}

define void @k(ptr addrspace(1) %out, ptr addrspace(2) %m, i32 %gid) {{
  %pt = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
{read}  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 0)
  %s = add i32 %b, %w
  %i = zext i32 %gid to i64
  %q = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %s, ptr addrspace(1) %q, align 4
  ret void
}}

declare ptr addrspace(2) @air.get_read_sampler()
declare {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)

!air.kernel = !{{!0}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !7}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !8, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"M", !"air.arg_name", !"m"}}
!7 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!8 = !{{i32 0, i32 8, i32 0, !"{texture}", !"t", !"air.indirect_argument", !9}}
!9 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"{texture}", !"air.arg_name", !"t"}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The heap image classes the module declares.
fn heap_classes(a: &FunctionArtifact) -> Vec<ImageHandleClass> {
    a.interface
        .descriptors
        .iter()
        .filter_map(|d| match d.slot.kind {
            ResourceKind::Heap(HandleClass::Image(c)) => Some(c),
            _ => None,
        })
        .collect()
}

/// The sampled 2D heap class of `ty`.
fn sampled_2d(ty: SampledType) -> impl Fn(&ImageHandleClass) -> bool {
    move |c| {
        c.dim == ImageDim::D2
            && !c.arrayed
            && !c.multisampled
            && !c.depth_compare
            && c.access == ImageAccess::Sampled
            && c.sampled_type == ty
    }
}

/// Each use on every committed profile with a sampled heap (the others refuse the argument
/// buffer's texture): the classes it declares, the routine, the word estimate, and L3.
#[test]
fn heap_reads_reach_a_raw_member_through_the_unsigned_twin() {
    let consts = FunctionConstantValues::defaults();
    let entry = EntryRef::Name("k");
    let inputs = LowerInputs::new();
    for u in [Use::Float, Use::Half, Use::Query, Use::Uint] {
        let Some(bc) = assemble("heap", &kernel(u)) else {
            return;
        };
        let mut compiled = 0;
        for (stem, p) in committed_profiles().expect("profiles") {
            let what = format!("{u:?} on {stem}");
            let a = match compile_bitcode(
                &bc,
                entry,
                &consts,
                &p,
                &inputs,
                &mut meter(),
                SPIRV_WORDS,
            ) {
                Ok(a) => a,
                Err(Error::Unsupported(Unsupported {
                    feature: Feature::ArgumentBufferTextures,
                    ..
                })) => continue,
                Err(e) => panic!("{what}: {e}"),
            };
            let plan = plan_bitcode(&bc, entry, &consts, &p, &inputs, &mut meter())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let words = a.module.words.len() as u64;
            assert!(
                plan.estimate.words >= words,
                "{what}: estimate {} < emitted {words}",
                plan.estimate.words
            );
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let classes = heap_classes(&a);
            let float = classes
                .iter()
                .filter(|c| sampled_2d(SampledType::Float)(c))
                .count();
            let uint = classes
                .iter()
                .filter(|c| sampled_2d(SampledType::Uint)(c))
                .count();
            let routine = a.strategies.entries.contains_key(heap_raw::CENSUS_NAME);
            match u {
                Use::Float | Use::Half => {
                    assert_eq!((float, uint, routine), (1, 1, true), "{what}: {classes:?}")
                }
                Use::Query => {
                    assert_eq!((float, uint, routine), (1, 1, false), "{what}: {classes:?}")
                }
                Use::Uint => {
                    assert_eq!((float, uint, routine), (1, 1, false), "{what}: {classes:?}")
                }
            }
            compiled += 1;
        }
        assert!(
            compiled > 0,
            "{u:?}: no committed profile has the sampled heap"
        );
    }
}

/// How [`sampling_kernel`] samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sampling {
    /// A sample through a constexpr sampler of these words.
    Sample(u64),
    /// A `half` sample through a constexpr sampler of these words.
    HalfSample(u64),
    /// A `float` gather of the red component through a constexpr sampler of these words.
    Gather(u64),
    /// A `half` gather of the red component through a constexpr sampler of these words.
    HalfGather(u64),
    /// A sample through the runtime sampler at sampler slot 0.
    Runtime,
}

/// A constexpr sampler's words (AIR's `air.sampler_state` layout): clamped to the edge, no mip
/// filter where `!mip` (else nearest), both filters linear where `linear`.
const fn words(linear: bool, mip: bool) -> u64 {
    let filters = if linear { 1 << 9 | 1 << 11 } else { 0 };
    let mip = if mip { 1 << 13 } else { 0 };
    0x007b_ff00_0008_0000 | 1 | 1 << 3 | 1 << 6 | filters | mip
}

/// Synthetic AIR written for this test: kernel `k` samples or gathers (`s`) the
/// `texture2d<T, access::sample>` its argument buffer (buffer 1) holds at `(gid / 8, 0.5)`,
/// storing the result's first lane's bits to `out` (buffer 0).
fn sampling_kernel(s: Sampling) -> String {
    let half = matches!(s, Sampling::HalfSample(_) | Sampling::HalfGather(_));
    let (element, vec4, suffix) = match half {
        true => ("half", "<4 x half>", "v4f16"),
        false => ("float", "<4 x float>", "v4f32"),
    };
    let texture = format!("texture2d<{element}, sample>");
    let (states, param, sampler, meta, gid, args) = match s {
        Sampling::Runtime => (
            String::new(),
            "ptr addrspace(2) %s, ",
            "%s",
            "!6 = !{i32 2, !\"air.sampler\", !\"air.location_index\", i32 0, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"s\"}\n".to_owned(),
            3,
            "!3, !4, !6, !7",
        ),
        Sampling::Sample(w) | Sampling::HalfSample(w) | Sampling::Gather(w) | Sampling::HalfGather(w) => (
            format!("@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {w}, i64 0], align 8\n"),
            "",
            "@__air_sampler_state",
            "!air.sampler_states = !{!10}\n!10 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n".to_owned(),
            2,
            "!3, !4, !7",
        ),
    };
    let (call, decl) = match s {
        Sampling::Gather(_) | Sampling::HalfGather(_) => (
            format!("@air.gather_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) {sampler}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)"),
            format!("declare {{ {vec4}, i8 }} @air.gather_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)"),
        ),
        _ => (
            format!("@air.sample_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) {sampler}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)"),
            format!("declare {{ {vec4}, i8 }} @air.sample_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)"),
        ),
    };
    let bits = match half {
        true => "  %e = fpext half %x to float\n  %b = bitcast float %e to i32\n",
        false => "  %b = bitcast float %x to i32\n",
    };
    format!(
        r#"%struct.M = type {{ ptr addrspace(1) }}
{states}
define void @k(ptr addrspace(1) %out, ptr addrspace(2) %m, {param}i32 %gid) {{
  %pt = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %g = uitofp i32 %gid to float
  %u = fmul float %g, 1.250000e-01
  %u0 = insertelement <2 x float> <float 0.0, float 5.000000e-01>, float %u, i64 0
  %uv = shufflevector <2 x float> %u0, <2 x float> poison, <2 x i32> <i32 0, i32 1>
  %r = call {{ {vec4}, i8 }} {call}
  %v = extractvalue {{ {vec4}, i8 }} %r, 0
  %x = extractelement {vec4} %v, i64 0
{bits}  %i = zext i32 %gid to i64
  %q = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %b, ptr addrspace(1) %q, align 4
  ret void
}}

{decl}

!air.kernel = !{{!0}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}
{meta}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{{args}}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !8, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"M", !"air.arg_name", !"m"}}
!7 = !{{i32 {gid}, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!8 = !{{i32 0, i32 8, i32 0, !"{texture}", !"t", !"air.indirect_argument", !9}}
!9 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{texture}", !"air.arg_name", !"t"}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#,
    )
}

/// A heap sample or gather answers a texture the heap holds by its raw texels alone from the
/// group's unsigned integer class (`sampler::heap_raw`), on every committed profile with a
/// sampled heap: a constexpr sampler's nearest or linear sample, `float` or `half`, and its
/// `float` gather declare the twin, define the decode routine, state no
/// `RuntimeFact::HeapRawSamples`, stay inside the plan's word estimate, and pass L3; a `half`
/// gather (its non-bfloat forms not measured) and a sample through a runtime sampler slot
/// declare the twin too and state the fact (with no slot, and naming slot 0); a sample through a
/// mip-filtered constexpr sampler declares no twin and states the fact with no slot.
#[test]
fn heap_samples_reach_a_raw_member_through_the_unsigned_twin() {
    let consts = FunctionConstantValues::defaults();
    let entry = EntryRef::Name("k");
    let inputs = LowerInputs::new();
    let cases = [
        (Sampling::Sample(words(false, false)), true, None),
        (Sampling::Sample(words(true, false)), true, None),
        (Sampling::HalfSample(words(true, false)), true, None),
        (Sampling::HalfSample(words(false, false)), true, None),
        (Sampling::Gather(words(true, false)), true, None),
        (Sampling::HalfGather(words(false, false)), true, Some(None)),
        (Sampling::Runtime, true, Some(Some(0))),
        (Sampling::Sample(words(false, true)), false, Some(None)),
    ];
    for (s, twin, fact) in cases {
        let Some(bc) = assemble("heap_sample", &sampling_kernel(s)) else {
            return;
        };
        let mut compiled = 0;
        for (stem, p) in committed_profiles().expect("profiles") {
            let what = format!("{s:?} on {stem}");
            let a = match compile_bitcode(
                &bc,
                entry,
                &consts,
                &p,
                &inputs,
                &mut meter(),
                SPIRV_WORDS,
            ) {
                Ok(a) => a,
                Err(Error::Unsupported(Unsupported {
                    feature: Feature::ArgumentBufferTextures,
                    ..
                })) => continue,
                Err(e) => panic!("{what}: {e}"),
            };
            let plan = plan_bitcode(&bc, entry, &consts, &p, &inputs, &mut meter())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let words = a.module.words.len() as u64;
            assert!(
                plan.estimate.words >= words,
                "{what}: estimate {} < emitted {words}",
                plan.estimate.words
            );
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let classes = heap_classes(&a);
            let uint = classes
                .iter()
                .filter(|c| sampled_2d(SampledType::Uint)(c))
                .count();
            let routine = a.strategies.entries.contains_key(heap_raw::CENSUS_NAME);
            assert_eq!((uint == 1, routine), (twin, twin), "{what}: {classes:?}");
            let stated: Vec<Option<u32>> = a
                .facts
                .iter()
                .filter_map(|f| match f {
                    RuntimeFact::HeapRawSamples { stage, sampler } => {
                        assert_eq!(*stage, Stage::Kernel, "{what}");
                        Some(*sampler)
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(stated, fact.into_iter().collect::<Vec<_>>(), "{what}");
            compiled += 1;
        }
        assert!(
            compiled > 0,
            "{s:?}: no committed profile has the sampled heap"
        );
    }
}
