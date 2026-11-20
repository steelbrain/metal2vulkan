//! Dynamic texel offsets on a profile without the `Offset` operand's features.
//!
//! Vulkan admits a dynamic `Offset` image operand on a gather with `shaderImageGatherExtended`
//! and on a sample also with `maintenance8` (VUID-RuntimeSpirv-Offset-10213); the floor has
//! neither, and drivers older than their `maintenance8` release lack it. There an access with a
//! dynamic offset is a switch over the values Metal reads on the offset's dynamic lanes (four
//! bits each), each arm the same access with its value as `ConstOffset`, which Vulkan 1.0 admits
//! on every sample and gather: both operands add the same texel offset at each level the access
//! reads, so each arm reads what the `Offset` operand of its value reads. A lane the IR states as
//! a constant is not switched on. A fragment's implicit-level sample takes its derivatives in its
//! own position and samples with them in every arm (`PrecisionNote::GradFromBias`): the switch
//! holds no implicit level. Past `OFFSET_ARMS_PER_FUNCTION` arms in one function the access is
//! refused as without the arms.
//!
//! The interpreter does not model sampling, so these are shape tests on the emitted SPIR-V, every
//! module validated by `spirv-val` under its profile. `llvm-as` assembles the kernels; without it
//! the tests skip visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DevicePolicy};
use crate::error::{Error, Feature};
use crate::lower::sampler::OFFSET_ARMS_PER_FUNCTION;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::testing::{
    compile, description, engine_dir, link_and_validate, llvm_assemble, profile, AssembleError,
};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// `a2_dynamic_texel_offset`'s constexpr sampler: nearest, clamp to edge, normalized.
const NEAREST_WORDS: u64 = 34_901_797_601_017_929;

/// A constexpr sampler filtering linearly between and within levels, repeating: the words of
/// `a2_fragment_bias_clamp`'s with a linear magnify and mip filter.
const MIPMAPPED_WORDS: u64 = 34_901_797_601_036_946;

const OP_DPDX: u16 = 207;
const OP_DPDY: u16 = 208;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
const OP_IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
const OP_IMAGE_GATHER: u16 = 96;
const OP_SWITCH: u16 = 251;

/// The image-operand mask bits of `ConstOffset` and `Offset`.
const CONST_OFFSET: u32 = 0x8;
const OFFSET: u32 = 0x10;

fn skip_missing<T>(r: Result<T, AssembleError>, what: &str) -> Option<T> {
    match r {
        Ok(x) => Some(x),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("{what}: {why}"),
    }
}

/// Bitcode of the module `body` (after the AIR header), or `None` without `llvm-as`.
fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-offset-arms-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, format!("{HEADER}{body}")).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    skip_missing(r, name)
}

/// Committed profile `stem` without `maintenance8` and `shaderImageGatherExtended`, as a driver
/// older than its `maintenance8` release (or the floor) reports them.
fn without_offset_operand(stem: &str) -> DeviceProfile {
    let mut desc = description(stem).expect("description");
    desc.features.remove(&VkFeature::Maintenance8);
    desc.features.remove(&VkFeature::ShaderImageGatherExtended);
    Device::from_queries(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// The profiles without the `Offset` operand's features: the runtime floor's and a device's with
/// them removed (the floor's below the runtime's floor, which states `shaderImageGatherExtended`).
fn armed_profiles() -> Vec<(&'static str, DeviceProfile)> {
    vec![
        (
            "runtime-floor without the operand's features",
            without_offset_operand("runtime-floor"),
        ),
        ("amd-780m-radv-m8", without_offset_operand("amd-780m-radv")),
    ]
}

/// The instructions of a SPIR-V module: (opcode, words).
fn insts(words: &[u32]) -> Vec<(u16, &[u32])> {
    let mut out = Vec::new();
    let mut at = 5;
    while let Some(w) = words.get(at) {
        let n = ((w >> 16) as usize).max(1);
        let Some(inst) = words.get(at..at + n) else {
            break;
        };
        out.push(((w & 0xffff) as u16, inst));
        at += n;
    }
    out
}

/// The number of instructions `op` whose image-operand mask (word `mask_at`) has any bit of `bits`.
fn with_operand(words: &[u32], op: u16, mask_at: usize, bits: u32) -> usize {
    insts(words)
        .iter()
        .filter(|(o, i)| *o == op && i.get(mask_at).is_some_and(|m| m & bits != 0))
        .count()
}

/// Whether the module has an `Offset` operand on any sample or gather.
fn has_offset_operand(words: &[u32]) -> bool {
    [
        (OP_IMAGE_SAMPLE_IMPLICIT_LOD, 5),
        (OP_IMAGE_SAMPLE_EXPLICIT_LOD, 5),
        (OP_IMAGE_GATHER, 6),
    ]
    .into_iter()
    .any(|(op, at)| with_operand(words, op, at, OFFSET) != 0)
}

/// The authored cases' kernel `a2_dynamic_texel_offset`: its sample and its gather each take the
/// 256 arms of their two dynamic lanes (255 with a `ConstOffset`, the zero offset's without), the
/// module has no `Offset` operand and needs neither feature, and it validates for the profile.
#[test]
fn the_fixture_takes_arms_without_the_offset_operand() {
    let path = engine_dir().join("a2_dynamic_texel_offset.ll");
    let Some(bc) = skip_missing(llvm_assemble(&path), "a2_dynamic_texel_offset") else {
        return;
    };
    for (name, p) in armed_profiles() {
        let a =
            compile(&bc, "a2_dynamic_texel_offset", &p).unwrap_or_else(|e| panic!("{name}: {e}"));
        for f in [
            VkFeature::Maintenance8,
            VkFeature::ShaderImageGatherExtended,
        ] {
            assert!(!a.requirements.features.contains(&f), "{name} needs {f:?}");
        }
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!has_offset_operand(&words), "{name}: an Offset operand");
        assert_eq!(
            (
                with_operand(&words, OP_IMAGE_SAMPLE_EXPLICIT_LOD, 5, CONST_OFFSET),
                with_operand(&words, OP_IMAGE_GATHER, 6, CONST_OFFSET),
            ),
            (255, 255),
            "{name}"
        );
        assert!(
            !a.precision.contains(&PrecisionNote::GradFromBias),
            "{name}: {:?}",
            a.precision
        );
    }
    // With the features, the `Offset` operand as before.
    let p = profile("amd-780m-radv", DevicePolicy::default()).expect("profile");
    let a = compile(&bc, "a2_dynamic_texel_offset", &p).expect("compile");
    let (_, words) = link_and_validate(&a, &p).expect("L3");
    assert!(has_offset_operand(&words));
    assert!(!insts(&words).iter().any(|(o, _)| *o == OP_SWITCH));
}

/// How [`samples_kernel`] makes each sample's `int2` offset from the `int2` `l` it loads.
#[derive(Clone, Copy, Debug)]
enum OffsetShape {
    /// `l`: both lanes dynamic.
    Loaded,
    /// `l.x` inserted into `zeroinitializer`: `y` is 0.
    XIntoZero,
    /// The constant 2 inserted into `l`'s `y`.
    TwoIntoY,
    /// A shuffle of `l`'s `x` and `y` 2 of a constant with an `undef` `x`.
    ShuffledWithTwo,
    /// `TwoIntoY` plus the constant `(-1, 3)`: `y` is 5.
    TwoIntoYPlusConstant,
    /// A shuffle of two constants' lanes, inserted over by another constant: `(3, -1)` with no
    /// lane dynamic.
    Constant,
}

/// A kernel of `n` samples of `texture2d<float, sample>` slot 0 through the nearest normalized
/// sampler at level 0, the `k`-th with an `int2` offset made from `offsets[n * tid + k]` as `shape`
/// says, summing their `x` into `out[tid]`.
fn samples_kernel(n: u32, shape: OffsetShape) -> String {
    let mut body = String::new();
    body.push_str("  %i = zext i32 %tid to i64\n");
    body.push_str(&format!("  %base = mul i64 %i, {n}\n"));
    body.push_str("  %s0 = fadd float 0.000000e+00, 0.000000e+00\n");
    for k in 0..n {
        let next = k + 1;
        body.push_str(&format!(
            "  %j{k} = add i64 %base, {k}\n  %p{k} = getelementptr inbounds <2 x i32>, ptr addrspace(1) %offs, i64 %j{k}\n  %l{k} = load <2 x i32>, ptr addrspace(1) %p{k}, align 8\n"
        ));
        body.push_str(&match shape {
            OffsetShape::Loaded => format!("  %o{k} = add <2 x i32> %l{k}, zeroinitializer\n"),
            OffsetShape::XIntoZero => format!(
                "  %x{k} = extractelement <2 x i32> %l{k}, i64 0\n  %o{k} = insertelement <2 x i32> zeroinitializer, i32 %x{k}, i64 0\n"
            ),
            OffsetShape::TwoIntoY => {
                format!("  %o{k} = insertelement <2 x i32> %l{k}, i32 2, i64 1\n")
            }
            OffsetShape::TwoIntoYPlusConstant => format!(
                "  %t{k} = insertelement <2 x i32> %l{k}, i32 2, i64 1\n  %o{k} = add <2 x i32> %t{k}, <i32 -1, i32 3>\n"
            ),
            OffsetShape::ShuffledWithTwo => format!(
                "  %o{k} = shufflevector <2 x i32> %l{k}, <2 x i32> <i32 undef, i32 2>, <2 x i32> <i32 0, i32 3>\n"
            ),
            OffsetShape::Constant => format!(
                "  %c{k} = shufflevector <2 x i32> <i32 7, i32 -1>, <2 x i32> <i32 undef, i32 5>, <2 x i32> <i32 3, i32 1>\n  %o{k} = insertelement <2 x i32> %c{k}, i32 3, i64 0\n"
            ),
        });
        body.push_str(&format!(
            "  %r{k} = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> <float 5.078125e-01, float 5.078125e-01>, i1 true, <2 x i32> %o{k}, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)\n  %v{k} = extractvalue {{ <4 x float>, i8 }} %r{k}, 0\n  %e{k} = extractelement <4 x float> %v{k}, i64 0\n  %s{next} = fadd float %s{k}, %e{k}\n"
        ));
    }
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {NEAREST_WORDS}, i64 0], align 8

define void @k(ptr addrspace(1) %t, ptr addrspace(1) %offs, ptr addrspace(1) %out, i32 %tid) {{
{body}  %q = getelementptr inbounds float, ptr addrspace(1) %out, i64 %i
  store float %s{n}, ptr addrspace(1) %q, align 4
  ret void
}}

declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{{!0}}
!air.sampler_states = !{{!7}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"int2", !"air.arg_name", !"offs"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}}
!6 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
!7 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
"#
    )
}

/// An offset whose `y` lane the IR states as a constant, through an insert, a shuffle or an add of
/// constants, switches on its `x` lane alone: 16 arms, each with a `ConstOffset` but for a zero
/// offset's. One whose every lane the IR states is a `ConstOffset` on every profile, with no
/// switch.
#[test]
fn a_constant_lane_is_not_switched_on() {
    // The arm of `x` 0 takes no `ConstOffset` where `y` is 0.
    for (shape, offsets) in [
        (OffsetShape::XIntoZero, 15),
        (OffsetShape::TwoIntoY, 16),
        (OffsetShape::ShuffledWithTwo, 16),
        (OffsetShape::TwoIntoYPlusConstant, 16),
    ] {
        let Some(bc) = assemble("one_lane", &samples_kernel(1, shape)) else {
            return;
        };
        for (name, p) in armed_profiles() {
            let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{shape:?} {name}: {e}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{shape:?} {name}: {e}"));
            assert!(!has_offset_operand(&words), "{shape:?} {name}");
            assert_eq!(
                with_operand(&words, OP_IMAGE_SAMPLE_EXPLICIT_LOD, 5, CONST_OFFSET),
                offsets,
                "{shape:?} {name}"
            );
        }
    }
    let Some(bc) = assemble("constant", &samples_kernel(1, OffsetShape::Constant)) else {
        return;
    };
    let with_operand_features = profile("amd-780m-radv", DevicePolicy::default()).expect("radv");
    let profiles = armed_profiles()
        .into_iter()
        .chain([("amd-780m-radv", with_operand_features)]);
    for (name, p) in profiles {
        let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{name}: {e}"));
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!has_offset_operand(&words), "{name}");
        assert!(
            !insts(&words).iter().any(|(o, _)| *o == OP_SWITCH),
            "{name}"
        );
        assert_eq!(
            with_operand(&words, OP_IMAGE_SAMPLE_EXPLICIT_LOD, 5, CONST_OFFSET),
            1,
            "{name}"
        );
    }
}

/// One function takes at most `OFFSET_ARMS_PER_FUNCTION` arms: two samples of two dynamic lanes
/// (512 arms), or 32 of one, translate, and a third, or a thirty-third, is refused as without the
/// arms, naming the missing features.
#[test]
fn a_function_past_the_arms_bound_is_refused() {
    assert_eq!(OFFSET_ARMS_PER_FUNCTION, 2 * 256);
    let p = without_offset_operand("amd-780m-radv");
    for (shape, within, offsets) in [
        (OffsetShape::Loaded, 2, 2 * 255),
        (OffsetShape::XIntoZero, 32, 32 * 15),
    ] {
        let Some(bc) = assemble("within", &samples_kernel(within, shape)) else {
            return;
        };
        let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{shape:?}: {e}"));
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{shape:?}: {e}"));
        assert_eq!(
            with_operand(&words, OP_IMAGE_SAMPLE_EXPLICIT_LOD, 5, CONST_OFFSET),
            offsets,
            "{shape:?}"
        );
        let Some(past) = assemble("past", &samples_kernel(within + 1, shape)) else {
            return;
        };
        match compile(&past, "k", &p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::TexelOffset, "{shape:?}: {u:?}")
            }
            other => panic!("{shape:?} past: {:?}", other.map(|_| ())),
        }
    }
}

/// A fragment function sampling `texture2d<float>` slot 0 at its position over 16, with the
/// dynamic offset `int2(position)` and shader bias `bias` (an LLVM float operand, `%bias` for one
/// read from the position's `z`), through the constexpr mipmapped sampler, or through runtime
/// sampler slot 0 with `runtime`.
fn offset_fragment(bias: &str, runtime: bool) -> String {
    let (state, sampler, param, meta) = if runtime {
        (
            String::new(),
            "ptr addrspace(2) %s".to_string(),
            ", ptr addrspace(2) %s",
            "!3 = !{!4, !5, !6}\n!6 = !{i32 2, !\"air.sampler\", !\"air.location_index\", i32 0, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"s\"}\n".to_string(),
        )
    } else {
        (
            format!("@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {MIPMAPPED_WORDS}, i64 0], align 8\n\n"),
            "ptr addrspace(2) @__air_sampler_state".to_string(),
            "",
            "!3 = !{!4, !5}\n!air.sampler_states = !{!6}\n!6 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n".to_string(),
        )
    };
    format!(
        r#"{state}define <4 x float> @k(<4 x float> %pos, ptr addrspace(1) %t{param}) {{
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %uv = fmul <2 x float> %xy, <float 6.250000e-02, float 6.250000e-02>
  %o = fptosi <2 x float> %xy to <2 x i32>
  %bias = extractelement <4 x float> %pos, i64 2
  %r = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, {sampler}, <2 x float> %uv, i1 true, <2 x i32> %o, i1 false, float {bias}, float 0.000000e+00, i32 0)
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  ret <4 x float> %v
}}

declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.fragment = !{{!0}}
!0 = !{{ptr @k, !1, !3}}
!1 = !{{!2}}
!2 = !{{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}}
{meta}!4 = !{{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}}
!5 = !{{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
"#
    )
}

/// A fragment's implicit-level sample with a dynamic offset: with the features, the implicit
/// sample with its `Offset` operand and no note; without them no implicit sample, the derivatives
/// taken before the switch and every arm sampling with them, and `GradFromBias` recorded. With
/// no bias, a constant one, one read per fragment, and through a runtime sampler.
#[test]
fn a_fragment_takes_its_derivatives_before_the_arms() {
    let shapes = [
        ("no_bias", "0.000000e+00", false),
        ("constant_bias", "1.500000e+00", false),
        ("dynamic_bias", "%bias", false),
        ("runtime_sampler", "0.000000e+00", true),
        ("runtime_dynamic_bias", "%bias", true),
    ];
    let full = profile("amd-780m-radv", DevicePolicy::default()).expect("profile");
    for (shape, bias, runtime) in shapes {
        let Some(bc) = assemble(shape, &offset_fragment(bias, runtime)) else {
            return;
        };
        let a = compile(&bc, "k", &full).unwrap_or_else(|e| panic!("{shape}: {e}"));
        let (_, words) = link_and_validate(&a, &full).unwrap_or_else(|e| panic!("{shape}: {e}"));
        assert!(has_offset_operand(&words), "{shape}: with the features");
        for (name, p) in armed_profiles() {
            let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{shape} {name}: {e}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{shape} {name}: {e}"));
            let all = insts(&words);
            assert!(!has_offset_operand(&words), "{shape} {name}");
            assert!(
                !all.iter().any(|(o, _)| *o == OP_IMAGE_SAMPLE_IMPLICIT_LOD),
                "{shape} {name}: an implicit sample"
            );
            let mut ops: Vec<u16> = all.iter().map(|(o, _)| *o).collect();
            ops.sort_unstable();
            ops.dedup();
            assert!(
                all.iter().any(|(o, _)| *o == OP_DPDX) && all.iter().any(|(o, _)| *o == OP_DPDY),
                "{shape} {name}: the derivatives: {ops:?} {:?}",
                a.strategies.entries.keys().collect::<Vec<_>>()
            );
            assert!(
                with_operand(&words, OP_IMAGE_SAMPLE_EXPLICIT_LOD, 5, CONST_OFFSET) >= 255,
                "{shape} {name}: the arms"
            );
            assert!(
                a.precision.contains(&PrecisionNote::GradFromBias),
                "{shape} {name}: {:?}",
                a.precision
            );
        }
    }
}
