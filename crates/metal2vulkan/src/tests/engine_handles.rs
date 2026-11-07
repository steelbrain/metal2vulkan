//! Texture and sampler handles as values end to end.
//!
//! Structural and L3 tests: every compiled module passes `spirv-val` under the profile (`G-val`)
//! and links through `pipeline::link`; the tests check the switch-at-use shape, the hoisted
//! gradients, and the refusals. The CPU interpreter does not model sampling, so sampled values are
//! not checked here.
//!
//! - Selected, merged (phi), null, and arrayed textures and samplers compile on every committed
//!   profile (the switch-at-use is total on the floor).
//! - The switch rule: no tangled instruction sits inside a switch-at-use. An implicit-LOD sample
//!   through a selected texture takes its gradients before the switch (`DPdx`/`DPdy`) and samples
//!   with explicit gradients inside it; an LOD query, which has no gradient form, queries every
//!   candidate before the switch, which selects one, up to a bound.
//! - A handle loaded from an argument buffer is a heap handle: a handle-table lookup and a
//!   non-uniform element of the set-1 array of its class where the profile has descriptor indexing,
//!   a typed `ArgumentBufferTextures` refusal naming the features elsewhere. A heap texture is
//!   sampled through a separate heap sampler; sampling one through a runtime sampler or with
//!   comparison is `NotImplemented`. The binder's half, and the typing of a group by its member's
//!   reads, are `src/tests/engine_heap_binder.rs`.
//! - A sampler loaded from an argument buffer is a heap handle of its own domain: a lookup in the
//!   stage's sampler table and a non-uniform element of the set-1 sampler array of the sample's
//!   use, paired with the heap image (`argument_buffer_samplers_are_heap_lookups`).
//! - A handle used as a byte pointer is `HandleAsPointer`. The pipeline state an encoded compute
//!   command is set to is a pipeline-state handle, so such a command reaches its family's refusal
//!   instead.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::{heap_classes, BindState, BindingPlan, WriteSource};
use crate::budget::Budget;
use crate::device::profile::DeviceProfile;
use crate::device::DevicePolicy;
use crate::device::{Device, DeviceDescription, Requirements, VkFeature};
use crate::error::{
    AbiError, Error, Feature, InputErrorKind, LowerFamily, Needs, RuntimeInput, Unsupported,
};
use crate::library::{FunctionArtifact, Library};
use crate::metal::sampler::SamplerWord;
use crate::metal::PixelFormat;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::descriptor::ColorAttachmentDescriptor;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::layout::BindingSource;
use crate::pipeline::link::LinkTable;
use crate::pipeline::sym::{
    CombinedSampler, HandleClass, ImageAccess, ImageDim, ImageHandleClass, ResourceKind,
    ResourceSlot, SampledType, SamplerRef, SlotClass, Stage, SyntheticBinding,
};
use crate::pipeline::testing::{
    compile, compile_given_storage_format, engine_dir, fixtures_dir, link_and_validate,
    llvm_assemble, profile, profile_with_storage_buffers, status_reach, status_reach_on,
    storage_buffers, typed_module, AssembleError, StatusReach,
};
use crate::pipeline::{
    heap_sampler_capacity, ComputePipelineDescriptor, FunctionDescriptor, RenderPipelineDescriptor,
    StageSource,
};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Resources};
use crate::spirv::testing::{committed_descriptions, committed_profiles};
use crate::vk;

const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

/// The entries of `engine/handles` that compile on every profile.
const OK: [&str; 7] = [
    "select_read",
    "phi_read",
    "select_null_write",
    "array_read",
    "sampler_select",
    "fragment_select_sample",
    "fragment_select_lod",
];

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("handles.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected handles: {why}"),
    }
}

fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let text = format!(
        "target datalayout = \"{DATALAYOUT}\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n{body}"
    );
    let dir = std::env::temp_dir().join(format!(
        "m2v-handles-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, &text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}\n{text}"),
    }
}

fn m3() -> DeviceProfile {
    profile("moltenvk-m3", DevicePolicy::default()).expect("profile")
}

fn ok(bc: &[u8], entry: &str, p: &DeviceProfile) -> std::sync::Arc<FunctionArtifact> {
    let a = compile(bc, entry, p).unwrap_or_else(|e| panic!("{entry}: compile: {e}"));
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{entry}: L3: {e}"));
    a
}

fn unsupported(e: Error) -> Unsupported {
    match e {
        Error::Unsupported(u) => u,
        other => panic!("expected Unsupported, got {other}"),
    }
}

/// The instructions of a SPIR-V module: `(opcode, word offset)`, in order.
fn insts(words: &[u32]) -> Vec<(u16, usize)> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = (words[i] >> 16) as usize;
        out.push(((words[i] & 0xFFFF) as u16, i));
        i += n.max(1);
    }
    out
}

fn count(a: &FunctionArtifact, op: u16) -> usize {
    insts(&a.module.words)
        .iter()
        .filter(|(o, _)| *o == op)
        .count()
}

/// The values of the module's one-word `OpConstant`s.
fn u32_constants(a: &FunctionArtifact) -> Vec<u32> {
    const OP_CONSTANT: u16 = 43;
    let w = &a.module.words;
    insts(w)
        .iter()
        .filter(|(o, i)| *o == OP_CONSTANT && (w[*i] >> 16) == 4)
        .map(|(_, i)| w[i + 3])
        .collect()
}

/// The values of the module's one-word integer constants that a bitwise `And`, `Or` or `Xor`
/// takes as an operand: its masks (an exact rounding's integer tie test reads `0x4000_0000`
/// as a value, not as a mask).
fn mask_constants(a: &FunctionArtifact) -> Vec<u32> {
    const OP_TYPE_INT: u16 = 21;
    const OP_CONSTANT: u16 = 43;
    const OP_BITWISE: [u16; 3] = [197, 198, 199];
    let w = &a.module.words;
    let all = insts(w);
    let ints: Vec<u32> = all
        .iter()
        .filter(|(o, _)| *o == OP_TYPE_INT)
        .map(|(_, i)| w[i + 1])
        .collect();
    let operands: Vec<u32> = all
        .iter()
        .filter(|(o, _)| OP_BITWISE.contains(o))
        .flat_map(|(_, i)| [w[i + 3], w[i + 4]])
        .collect();
    all.iter()
        .filter(|(o, i)| {
            *o == OP_CONSTANT
                && (w[*i] >> 16) == 4
                && ints.contains(&w[i + 1])
                && operands.contains(&w[i + 2])
        })
        .map(|(_, i)| w[i + 3])
        .collect()
}

/// The texture slots the function's interface names.
fn texture_slots(a: &FunctionArtifact) -> Vec<u32> {
    a.interface
        .slots
        .keys()
        .filter(|s| s.class == SlotClass::Texture)
        .map(|s| s.index)
        .collect()
}

fn uses(a: &FunctionArtifact, name: &str) -> u32 {
    a.strategies.entries.get(name).map_or(0, |u| u.uses)
}

const OP_SWITCH: u16 = 251;
const OP_SELECTION_MERGE: u16 = 247;
const OP_LABEL: u16 = 248;
const OP_DPDX: u16 = 207;
const OP_DPDY: u16 = 208;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
const OP_IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
const OP_IMAGE_QUERY_LOD: u16 = 105;

/// Every handle entry compiles, links, and validates on every committed profile.
#[test]
fn handle_entries_compile_on_every_profile() {
    let Some(bc) = fixture() else { return };
    for (stem, p) in committed_profiles().expect("profiles") {
        for entry in OK {
            let a = match compile(&bc, entry, &p) {
                Ok(a) => a,
                // A storage texture's format is a pipeline input where the profile cannot write
                // it formatless, as for a static slot.
                Err(Error::NeedsInput(RuntimeInput::StorageTextureFormat { .. }))
                    if entry == "select_null_write"
                        && !p.fields().images.storage_write_without_format =>
                {
                    continue
                }
                Err(e) => panic!("{stem} {entry}: {e}"),
            };
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {entry}: L3: {e}"));
        }
    }
}

/// A selected texture: every operation through it is a switch-at-use over its two slots.
#[test]
fn a_selected_texture_switches_at_each_use() {
    let Some(bc) = fixture() else { return };
    let a = ok(&bc, "select_read", &m3());
    assert_eq!(uses(&a, "access.handle.select"), 1);
    assert_eq!(uses(&a, "texture.read.switch"), 1);
    assert_eq!(uses(&a, "texture.query.switch"), 2);
    assert_eq!(uses(&a, "texture.is_null.switch"), 1);
    assert_eq!(count(&a, OP_SWITCH), 4);
    // Both slots are declared and read.
    let slots = texture_slots(&a);
    assert_eq!(slots, vec![0, 1]);

    let a = ok(&bc, "phi_read", &m3());
    assert_eq!(uses(&a, "texture.read.switch"), 1);
    assert_eq!(count(&a, OP_SWITCH), 1);

    // A storage texture or null: one arm, and the null handle's write is dropped.
    let a = ok(&bc, "select_null_write", &m3());
    assert_eq!(uses(&a, "texture.null_handle"), 1);
    assert_eq!(uses(&a, "texture.write.switch"), 1);
}

/// An element of a texture array is the slot `first + k`: a constant element is that slot's
/// binding, and a dynamic one a switch over the array's slots.
#[test]
fn texture_array_elements_are_static_slots() {
    let Some(bc) = fixture() else { return };
    let a = ok(&bc, "array_read", &m3());
    assert_eq!(uses(&a, "access.handle.array_load.clamped"), 1);
    assert_eq!(uses(&a, "access.handle.array_load"), 1);
    assert_eq!(uses(&a, "texture.read.switch"), 1);
    assert_eq!(uses(&a, "texture.query"), 1);
    let slots = texture_slots(&a);
    assert_eq!(slots, vec![0, 1, 2, 3]);
}

/// Selected textures and samplers: one arm per candidate combination, nested switches.
#[test]
fn selected_samplers_switch_per_combination() {
    let Some(bc) = fixture() else { return };
    let a = ok(&bc, "sampler_select", &m3());
    // `ss[gid & 1]`: one switch over two samplers; `v.sample(s)`: two textures, then two
    // samplers in each arm.
    assert_eq!(uses(&a, "sampler.sample.switch"), 2);
    assert_eq!(count(&a, OP_SWITCH), 1 + 1 + 2);
    // Combined bindings: (t, ss[0]), (t, ss[1]), and each of t, u with s0 and s1.
    assert_eq!(
        a.interface.descriptors.len(),
        6 + 1,
        "{:?}",
        a.interface.descriptors
    );
}

/// No tangled instruction (a derivative, an implicit-LOD sample, an LOD query) lies between a
/// switch-at-use's header and its merge in the emitted words (the builder refuses it; this
/// checks the words too).
fn assert_no_tangled_in_a_switch(a: &FunctionArtifact) {
    let ops = insts(&a.module.words);
    let mut merges: Vec<u32> = Vec::new();
    let words = &a.module.words;
    for (k, (op, at)) in ops.iter().enumerate() {
        match *op {
            OP_SELECTION_MERGE if ops.get(k + 1).is_some_and(|(o, _)| *o == OP_SWITCH) => {
                merges.push(words[at + 1]);
            }
            OP_LABEL if merges.last() == Some(&words[at + 1]) => {
                merges.pop();
            }
            OP_DPDX | OP_DPDY | OP_IMAGE_SAMPLE_IMPLICIT_LOD | OP_IMAGE_QUERY_LOD => {
                assert!(
                    merges.is_empty(),
                    "a tangled instruction inside a switch-at-use"
                );
            }
            _ => {}
        }
    }
}

/// The implicit-LOD rule and its switch rule: the implicit-LOD samples through a selected texture
/// take the coordinate's derivatives before the switch, in the sample's position, and sample with
/// explicit gradients inside it; no tangled instruction lies between a switch-at-use's header
/// and its merge.
#[test]
fn tangled_samples_hoist_their_gradients_out_of_the_switch() {
    let Some(bc) = fixture() else { return };
    let a = ok(&bc, "fragment_select_sample", &m3());
    assert_eq!(uses(&a, "sampler.sample_hoisted_grad.switch"), 2);
    assert!(a.precision.contains(&PrecisionNote::GradFromBias));
    assert_eq!(count(&a, OP_IMAGE_SAMPLE_IMPLICIT_LOD), 0);
    assert_eq!(count(&a, OP_IMAGE_QUERY_LOD), 0);
    assert_eq!(count(&a, OP_DPDX), 2);
    assert_eq!(count(&a, OP_DPDY), 2);
    assert_eq!(count(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD), 4);
    assert_no_tangled_in_a_switch(&a);
}

/// An LOD query through a selected texture, which has no gradient form, runs every candidate's
/// query before the switch, at the call's own position with the call's coordinate, and the switch
/// only selects one: two `OpImageQueryLod` outside the one switch-at-use, on every committed
/// profile.
#[test]
fn a_selected_lod_query_queries_every_candidate_before_its_switch() {
    let Some(bc) = fixture() else { return };
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = ok(&bc, "fragment_select_lod", &p);
        assert_eq!(uses(&a, "sampler.query_lod.switch"), 1, "{stem}");
        assert_eq!(count(&a, OP_IMAGE_QUERY_LOD), 2, "{stem}");
        assert_eq!(count(&a, OP_SWITCH), 1, "{stem}");
        assert_no_tangled_in_a_switch(&a);
    }
}

/// A fragment sample of the texture chosen per fragment (`pos.x > 8 ? a : b`, `a4`'s shape)
/// through `sampler` (`%s`, the runtime sampler argument, or `@__air_sampler_state`, a constexpr
/// one), with the bias, offset and `min_lod_clamp` operands given as AIR text.
fn chosen_sample(sampler: &str, bias: &str, offset: &str, clamp: &str) -> Option<Vec<u8>> {
    assemble(
        "chosen_sample",
        &format!(
            r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601020050, i64 0], align 8

define <4 x float> @chosen_sample(<4 x float> %pos, ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(2) %s) {{
  %x = extractelement <4 x float> %pos, i64 0
  %right = fcmp ogt float %x, 8.000000e+00
  %t = select i1 %right, ptr addrspace(1) %a, ptr addrspace(1) %b
  %p = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %uv = fmul <2 x float> %p, splat (float 6.250000e-02)
  %r = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {sampler}, <2 x float> %uv, i1 true, <2 x i32> {offset}, i1 false, float {bias}, float {clamp}, i32 0)
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  ret <4 x float> %v
}}
declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
!air.fragment = !{{!0}}
!air.sampler_states = !{{!9}}
!air.version = !{{!7}}
!air.language_version = !{{!8}}
!0 = !{{ptr @chosen_sample, !1, !3}}
!1 = !{{!2}}
!2 = !{{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}}
!3 = !{{!4, !5, !6, !10}}
!4 = !{{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}}
!5 = !{{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"a"}}
!6 = !{{i32 2, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"b"}}
!10 = !{{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
!7 = !{{i32 2, i32 8, i32 0}}
!8 = !{{!"Metal", i32 4, i32 0, i32 0}}
!9 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
"#
        ),
    )
}

/// The hoisted derivatives of a texture chosen per fragment are the device's own sample for a 2D
/// color sample without a bias, an offset or a `min_lod_clamp`, through a runtime or a constexpr
/// sampler without a bias of its own: they record no `PrecisionNote::GradFromBias` and require the
/// device's sampler (`VkProperty::MetalSampler` on a profile that samples through Metal's own,
/// MoltenVK, `VkProperty::DeviceSampling` on any other). A bias, a nonzero offset or a clamp keeps
/// the note. Every case still hoists its derivatives out of the one switch-at-use.
#[test]
fn an_unbiased_chosen_sample_is_the_device_s_own() {
    let zero = "0.000000e+00";
    let cases = [
        ("%s", zero, "zeroinitializer", zero, true),
        ("@__air_sampler_state", zero, "zeroinitializer", zero, true),
        ("%s", "5.000000e-01", "zeroinitializer", zero, false),
        ("%s", zero, "<i32 1, i32 0>", zero, false),
        ("%s", zero, "zeroinitializer", "1.000000e+00", false),
    ];
    let mut saw_metal_sampler = false;
    for (sampler, bias, offset, clamp, exact) in cases {
        let Some(bc) = chosen_sample(sampler, bias, offset, clamp) else {
            return;
        };
        for (stem, p) in committed_profiles().expect("profiles") {
            let what = format!("{stem}: {sampler} bias {bias} offset {offset} clamp {clamp}");
            let a = match compile(&bc, "chosen_sample", &p) {
                Ok(a) => a,
                // A clamp needs `shaderResourceMinLod`, an offset the device's range.
                Err(Error::Unsupported(u)) if clamp != zero || offset != "zeroinitializer" => {
                    eprintln!("{what}: {u}");
                    continue;
                }
                Err(e) => panic!("{what}: {e}"),
            };
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_eq!(uses(&a, "sampler.sample_hoisted_grad.switch"), 1, "{what}");
            assert_eq!((count(&a, OP_DPDX), count(&a, OP_DPDY)), (1, 1), "{what}");
            assert_no_tangled_in_a_switch(&a);
            use crate::device::requirements::VkProperty;
            let metal = metal_sampler(&p);
            saw_metal_sampler |= metal;
            assert_eq!(
                a.precision.contains(&PrecisionNote::GradFromBias),
                !exact,
                "{what}"
            );
            if exact {
                let sampler = match metal {
                    true => VkProperty::MetalSampler,
                    false => VkProperty::DeviceSampling,
                };
                assert!(
                    a.requirements.properties.contains(&sampler),
                    "{what}: the module requires the device's sampler"
                );
            }
        }
    }
    assert!(
        saw_metal_sampler,
        "no committed profile samples through Metal's sampler"
    );
}

/// A fragment `calculate_clamped_lod` through element `pos.x` of an `n`-texture array, one
/// runtime sampler: a switch-at-use over `n` candidates.
fn lod_query_array(n: u32) -> Option<Vec<u8>> {
    assemble(
        &format!("lod_query_array_{n}"),
        &format!(
            r#"define <4 x float> @lod_query_array(<4 x float> %pos, ptr readonly byval([{n} x ptr addrspace(1)]) %ts, ptr addrspace(2) %s) {{
  %x = extractelement <4 x float> %pos, i64 0
  %i = fptoui float %x to i32
  %k = zext i32 %i to i64
  %pt = getelementptr inbounds [{n} x ptr addrspace(1)], ptr %ts, i64 0, i64 %k
  %t = load ptr addrspace(1), ptr %pt, align 8
  %c = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %l = call float @air.calculate_clamped_lod_texture_2d(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> %c, i32 0)
  %v = insertelement <4 x float> zeroinitializer, float %l, i64 0
  ret <4 x float> %v
}}
declare float @air.calculate_clamped_lod_texture_2d(ptr addrspace(1), ptr addrspace(2), <2 x float>, i32)
!air.fragment = !{{!0}}
!air.version = !{{!7}}
!air.language_version = !{{!8}}
!0 = !{{ptr @lod_query_array, !1, !3}}
!1 = !{{!2}}
!2 = !{{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}}
!3 = !{{!4, !5, !6}}
!4 = !{{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}}
!5 = !{{i32 1, !"air.texture", !"air.location_index", i32 0, i32 {n}, !"air.sample", !"air.arg_type_name", !"array<texture2d<float, sample>, {n}>", !"air.arg_name", !"ts"}}
!6 = !{{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
!7 = !{{i32 2, i32 8, i32 0}}
!8 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
        ),
    )
}

/// The bound: a fragment runs every candidate's LOD query, so up to `LOD_QUERY_CANDIDATES` (16)
/// candidate combinations are queried before the switch, and one more is `DivergentImplicitLod`,
/// not implemented, on every committed profile.
#[test]
fn lod_query_candidates_are_bounded() {
    let (Some(at), Some(past)) = (lod_query_array(16), lod_query_array(17)) else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = ok(&at, "lod_query_array", &p);
        assert_eq!(uses(&a, "sampler.query_lod.switch"), 1, "{stem}");
        assert_eq!(count(&a, OP_IMAGE_QUERY_LOD), 16, "{stem}");
        assert_no_tangled_in_a_switch(&a);
        let u = unsupported(compile(&past, "lod_query_array", &p).expect_err(&stem));
        assert_eq!(u.feature, Feature::DivergentImplicitLod, "{stem}");
        assert_eq!(u.needs, Needs::NotImplemented, "{stem}");
    }
}

/// What a [`switched`] kernel does through the texture it selects.
#[derive(Clone, Copy, Debug)]
enum SwitchedOp {
    /// Read a `texture_buffer<float, read>`.
    TexelRead,
    /// Read a `texture2d<float, sample>` at level 0.
    ImageRead,
    /// Write a `texture_buffer<float, write>`.
    TexelWrite,
    /// Write a `texture2d<float, write>`.
    ImageWrite,
    /// Query a `texture2d<float, sample>`'s width at the level the thread names.
    WidthAtLevel,
    /// `atomic_fetch_max` on a `texture2d<uint, read_write>`.
    AtomicMax,
}

/// A kernel that selects one of two textures by thread parity and performs `op` through the
/// selection (a switch-at-use over both), storing what it reads to `[[buffer(0)]]`.
fn switched(op: SwitchedOp) -> String {
    let (ty, access, call, declaration) = match op {
        SwitchedOp::TexelRead => (
            "texture_buffer<float, read>",
            "air.read",
            "%s = call ptr addrspace(2) @air.get_read_sampler()\n  \
             %r = call { <4 x float>, i8 } @air.read_texture_buffer_1d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, i32 %gid, i32 1)\n  \
             %v = extractvalue { <4 x float>, i8 } %r, 0\n  \
             store <4 x float> %v, ptr addrspace(1) %p, align 16",
            "declare ptr addrspace(2) @air.get_read_sampler()\n\
             declare { <4 x float>, i8 } @air.read_texture_buffer_1d.v4f32(ptr addrspace(1), ptr addrspace(2), i32, i32)",
        ),
        SwitchedOp::ImageRead => (
            "texture2d<float, sample>",
            "air.sample",
            "%s = call ptr addrspace(2) @air.get_read_sampler()\n  \
             %r = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %xy, <2 x i32> zeroinitializer, i32 0, i32 0)\n  \
             %v = extractvalue { <4 x float>, i8 } %r, 0\n  \
             store <4 x float> %v, ptr addrspace(1) %p, align 16",
            "declare ptr addrspace(2) @air.get_read_sampler()\n\
             declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)",
        ),
        SwitchedOp::TexelWrite => (
            "texture_buffer<float, write>",
            "air.write",
            "call void @air.write_texture_buffer_1d.v4f32(ptr addrspace(1) %t, i32 %gid, <4 x float> splat (float 1.0), i32 2)",
            "declare void @air.write_texture_buffer_1d.v4f32(ptr addrspace(1), i32, <4 x float>, i32)",
        ),
        SwitchedOp::ImageWrite => (
            "texture2d<float, write>",
            "air.write",
            "call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %xy, <4 x float> splat (float 1.0), i32 0, i32 2)",
            "declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)",
        ),
        SwitchedOp::WidthAtLevel => (
            "texture2d<float, sample>",
            "air.sample",
            "%w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 %gid)\n  \
             %f = uitofp i32 %w to float\n  \
             %v = insertelement <4 x float> zeroinitializer, float %f, i64 0\n  \
             store <4 x float> %v, ptr addrspace(1) %p, align 16",
            "declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)",
        ),
        SwitchedOp::AtomicMax => (
            "texture2d<uint, read_write>",
            "air.read_write",
            "%m = insertelement <4 x i32> zeroinitializer, i32 %gid, i64 0
  \
             %old = call <4 x i32> @air.atomic_fetch_max_explicit_texture_2d.u.v4i32(ptr addrspace(1) %t, <2 x i32> %xy, <2 x i32> zeroinitializer, <4 x i32> %m, i32 0, i32 3)",
            "declare <4 x i32> @air.atomic_fetch_max_explicit_texture_2d.u.v4i32(ptr addrspace(1), <2 x i32>, <2 x i32>, <4 x i32>, i32, i32)",
        ),
    };
    format!(
        r#"define void @switched(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b, i32 %gid) {{
entry:
  %odd = and i32 %gid, 1
  %c = icmp eq i32 %odd, 0
  %t = select i1 %c, ptr addrspace(1) %a, ptr addrspace(1) %b
  %xy = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %gid
  {call}
  ret void
}}

{declaration}

!air.kernel = !{{!0}}
!0 = !{{ptr @switched, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"{access}", !"air.arg_type_name", !"{ty}", !"air.arg_name", !"a"}}
!5 = !{{i32 2, !"air.texture", !"air.location_index", i32 1, i32 1, !"{access}", !"air.arg_type_name", !"{ty}", !"air.arg_name", !"b"}}
!6 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
"#
    )
}

/// A switch-at-use is named by the check its arms take, as a read, write, or query of one slot is.
/// A texel buffer's clamp or guard follows the buffer encoding and an image's the image encoding: a
/// read is left to the device only where its robustness returns Metal's out-of-bounds texel
/// (`robustBufferAccess2`, `robustImageAccess2`), and a write only where it discards the write
/// (`robustBufferAccess2`, `robustImageAccess`). On `vk12-floor` (`robustBufferAccess` alone) every
/// read and write is checked in the shader; on `moltenvk-m3` (`robustBufferAccess` and
/// `robustImageAccess2`) a texel buffer's are and an image's are not. A size query at a level the
/// thread names answers from the first level on every profile, and a texture atomic is guarded
/// exactly without image
/// robustness. The floor, without formatless storage, compiles a storage slot under its format
/// (`RGBA32Float`), as the runtime compiles the variant of the bound texture's format.
#[test]
fn a_switch_is_named_by_the_check_its_arms_take() {
    use SwitchedOp::*;
    let switches = |a: &FunctionArtifact| -> Vec<(String, u32)> {
        a.strategies
            .entries
            .iter()
            .filter(|(name, _)| name.ends_with(".switch"))
            .map(|(name, u)| (name.clone(), u.uses))
            .collect()
    };
    let floor = profile("vk12-floor", DevicePolicy::default()).expect("profile");
    for (op, on_floor, on_m3) in [
        (
            TexelRead,
            "texture.read_clamped.switch",
            "texture.read_clamped.switch",
        ),
        (
            ImageRead,
            "texture.read_clamped.switch",
            "texture.read.switch",
        ),
        (
            TexelWrite,
            "texture.write_guarded.switch",
            "texture.write_guarded.switch",
        ),
        (
            ImageWrite,
            "texture.write_guarded.switch",
            "texture.write.switch",
        ),
        (
            WidthAtLevel,
            "texture.query_level.switch",
            "texture.query_level.switch",
        ),
    ] {
        let Some(bc) = assemble("switched", &switched(op)) else {
            return;
        };
        for (stem, p, want) in [
            ("vk12-floor", &floor, on_floor),
            ("moltenvk-m3", &m3(), on_m3),
        ] {
            let a = compile_given_storage_format(&bc, "switched", p, PixelFormat::RGBA32Float)
                .unwrap_or_else(|e| panic!("{op:?} on {stem}: compile: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{op:?} on {stem}: L3: {e}"));
            assert_eq!(switches(&a), [(want.into(), 1)], "{op:?} on {stem}");
        }
    }
    let Some(bc) = assemble("switched", &switched(AtomicMax)) else {
        return;
    };
    for (stem, p, want) in [
        ("moltenvk-m3", m3(), "texture.atomic_max.switch"),
        (
            "moltenvk-m3 without image robustness",
            without_robust_images("moltenvk-m3"),
            "texture.atomic_max_guarded.switch",
        ),
    ] {
        let a = ok(&bc, "switched", &p);
        assert_eq!(switches(&a), [(want.into(), 1)], "{stem}");
    }
}

/// Profile `stem` with image robustness removed (`robustImageAccess` and `robustImageAccess2`).
fn without_robust_images(stem: &str) -> DeviceProfile {
    use crate::device::requirements::VkFeature;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    desc.features.remove(&VkFeature::RobustImageAccess);
    desc.features.remove(&VkFeature::RobustImageAccess2);
    Device::new(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// A heap element's read, write, and query name the in-shader check they take, as a slot's do: a
/// read clamps and a write is guarded exactly without image robustness, and a size query at a
/// level the thread names answers from the first level on every profile.
#[test]
fn heap_operations_are_named_by_the_check_they_take() {
    let Some(bc) = assemble(
        "heap_ops",
        r#"%struct.T = type { ptr addrspace(1) }
%struct.R = type { %struct.T, %struct.T }
define void @heap_ops(ptr addrspace(2) %r, ptr addrspace(1) %out, i32 %gid) {
  %ps = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %s = load ptr addrspace(1), ptr addrspace(2) %ps, align 8
  %pw = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 1, i32 0
  %w = load ptr addrspace(1), ptr addrspace(2) %pw, align 8
  %c = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  %smp = call ptr addrspace(2) @air.get_read_sampler()
  %t = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %s, ptr addrspace(2) %smp, <2 x i32> %c, <2 x i32> zeroinitializer, i32 0, i32 0)
  %v = extractvalue { <4 x float>, i8 } %t, 0
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %w, <2 x i32> %c, <4 x float> %v, i32 0, i32 2)
  %wd = call i32 @air.get_width_texture_2d(ptr addrspace(1) %s, i32 %gid)
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %gid
  store i32 %wd, ptr addrspace(1) %o, align 4
  ret void
}
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
!air.kernel = !{!0}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @heap_ops, !1, !2}
!1 = !{}
!2 = !{!3, !7, !8}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float>", !"s", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"texture2d<float, write>", !"w", !"air.indirect_argument", !6}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float>", !"air.arg_name", !"s"}
!6 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"w"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!8 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
"#,
    ) else {
        return;
    };
    let heap = |a: &FunctionArtifact| {
        a.strategies
            .entries
            .iter()
            .filter(|(name, _)| name.starts_with("texture.") && name.ends_with(".heap"))
            .map(|(name, u)| (name.clone(), u.uses))
            .collect::<Vec<_>>()
    };
    let named = |names: [&str; 3]| {
        names
            .iter()
            .map(|name| (name.to_string(), 1))
            .collect::<Vec<_>>()
    };
    let robust = ok(&bc, "heap_ops", &m3());
    assert_eq!(
        heap(&robust),
        named([
            "texture.query_level.heap",
            "texture.read.heap",
            "texture.write.heap"
        ])
    );
    let checked = ok(&bc, "heap_ops", &without_robust_images("moltenvk-m3"));
    assert_eq!(
        heap(&checked),
        named([
            "texture.query_level.heap",
            "texture.read_clamped.heap",
            "texture.write_guarded.heap"
        ])
    );
}

/// A kernel reading texel `gid` of one `read_write` argument-buffer texture and writing zero to
/// the same texel of another, of one heap class.
const HEAP_RW: &str = r#"%struct.T = type { ptr addrspace(1) }
%struct.R = type { %struct.T, %struct.T }
define void @heap_rw(ptr addrspace(2) %r, ptr addrspace(1) %out, i32 %gid) {
  %pa = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %a = load ptr addrspace(1), ptr addrspace(2) %pa, align 8
  %pb = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 1, i32 0
  %b = load ptr addrspace(1), ptr addrspace(2) %pb, align 8
  %c = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %b, <2 x i32> %c, <4 x float> zeroinitializer, i32 0, i32 3)
  %smp = call ptr addrspace(2) @air.get_read_sampler()
  %t = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %a, ptr addrspace(2) %smp, <2 x i32> %c, <2 x i32> zeroinitializer, i32 0, i32 3)
  %v = extractvalue { <4 x float>, i8 } %t, 0
  %x = extractelement <4 x float> %v, i64 0
  %o = getelementptr inbounds float, ptr addrspace(1) %out, i32 %gid
  store float %x, ptr addrspace(1) %o, align 4
  ret void
}
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
!air.kernel = !{!0}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @heap_rw, !1, !2}
!1 = !{}
!2 = !{!3, !7, !8}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, read_write>", !"a", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"texture2d<float, read_write>", !"b", !"air.indirect_argument", !6}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"a"}
!6 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"b"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!8 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// Two `read_write` argument-buffer textures of one heap class are one array, so a read through one
/// member and a write through the other may meet at one element out of bounds. Where the device is
/// left both (`robustImageAccess2`), the class's array is `Volatile` and every group of it says so;
/// where the shader checks reads and guards writes, none is, and the module passes L3 either way.
#[test]
fn a_heap_class_read_and_written_through_two_members_is_volatile() {
    let Some(bc) = assemble("heap_rw", HEAP_RW) else {
        return;
    };
    for (p, volatile) in [(m3(), true), (without_robust_images("moltenvk-m3"), false)] {
        let (plan, _) = typed_module(&bc, "heap_rw", &p).expect("heap_rw");
        let heap = &plan.families.texture.heap;
        assert_eq!(heap.len(), 2, "two members, two groups");
        let classes: Vec<_> = heap.values().map(|h| h.class).collect();
        assert!(classes.iter().all(|c| c.is_some() && *c == classes[0]));
        assert!(heap.values().all(|h| h.volatile == volatile), "{volatile}");
        let a = ok(&bc, "heap_rw", &p);
        // The emitted module says so: the class's one heap array is decorated `Volatile`, and
        // nothing else is .
        let w = &a.module.words;
        let volatile_ids: Vec<u32> = insts(w)
            .iter()
            .filter(|(op, at)| *op == OP_DECORATE && w[at + 2] == DECORATION_VOLATILE)
            .map(|(_, at)| w[at + 1])
            .collect();
        let arrays = image_arrays(&a);
        assert_eq!(arrays.len(), 1, "one class, one array");
        let want = if volatile { arrays } else { Vec::new() };
        assert_eq!(volatile_ids, want, "{volatile}");
    }
}

/// A `read_write` argument-buffer texture's float read and write convert a member the heap views
/// through its storage choice's alias (`bind::heap::storage_conversions`): on lavapipe, which
/// stores no 8-bit sRGB format and no `A2R10G10B10`, the search carries each handle's red-blue
/// order beside its class, and the read's decode of an sRGB member's codes notes
/// `SrgbTexelDecode` and the write's encode `DevicePackConversion`; on the M3 Max's MoltenVK, which
/// stores each such format itself, the module converts nothing and notes neither. Each validates.
#[test]
fn a_heap_storage_access_converts_the_members_its_heap_views_through_an_alias() {
    let Some(bc) = assemble("heap_rw", HEAP_RW) else {
        return;
    };
    for (stem, converts) in [("lavapipe", true), ("moltenvk-m3", false)] {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        let (plan, _) = typed_module(&bc, "heap_rw", &p).expect("heap_rw");
        assert_eq!(plan.families.access.heap_red_blue, converts, "{stem}");
        let a = ok(&bc, "heap_rw", &p);
        for note in [
            PrecisionNote::SrgbTexelDecode,
            PrecisionNote::DevicePackConversion,
        ] {
            assert_eq!(a.precision.contains(&note), converts, "{stem} {note:?}");
        }
    }
}

/// The `Volatile` decoration, and the instructions of a variable's type chain.
const DECORATION_VOLATILE: u32 = 21;
const OP_VARIABLE: u16 = 59;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
/// `OpTypeVoid` to `OpTypeForwardPointer`: the instructions declaring a type.
const OP_TYPES: std::ops::RangeInclusive<u16> = 19..=39;

/// The variables of `a`'s module that are runtime arrays of images: its heap arrays.
fn image_arrays(a: &FunctionArtifact) -> Vec<u32> {
    let w = &a.module.words;
    let all = insts(w);
    let def = |id: u32| {
        all.iter()
            .find(|(op, at)| OP_TYPES.contains(op) && w.get(at + 1) == Some(&id))
            .copied()
    };
    all.iter()
        .filter(|(op, _)| *op == OP_VARIABLE)
        .filter(|(_, at)| {
            let Some((OP_TYPE_POINTER, ptr)) = def(w[at + 1]) else {
                return false;
            };
            let Some((OP_TYPE_RUNTIME_ARRAY, arr)) = def(w[ptr + 3]) else {
                return false;
            };
            matches!(def(w[arr + 2]), Some((OP_TYPE_IMAGE, _)))
        })
        .map(|(_, at)| w[at + 2])
        .collect()
}

/// A kernel reading argument-buffer texture `a` (its type, declared access, and read access are
/// `@@ATYPE@@`, `@@AACCESS@@`, and `@@READACCESS@@`), writing `b` at `@@WRITE@@` (or not), and
/// storing what it read at `@@STORE@@` (a plain or a coherent store of `%xi` to `%o`).
const HEAP_BESIDE_STORE: &str = r#"%struct.T = type { ptr addrspace(1) }
%struct.R = type { %struct.T, %struct.T }
define void @heap_store(ptr addrspace(2) %r, ptr addrspace(1) %out, i32 %gid) {
  %pa = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %a = load ptr addrspace(1), ptr addrspace(2) %pa, align 8
  %pb = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 1, i32 0
  %b = load ptr addrspace(1), ptr addrspace(2) %pb, align 8
  %c = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  @@WRITE@@
  %smp = call ptr addrspace(2) @air.get_read_sampler()
  %t = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %a, ptr addrspace(2) %smp, <2 x i32> %c, <2 x i32> zeroinitializer, i32 0, i32 @@READACCESS@@)
  %v = extractvalue { <4 x float>, i8 } %t, 0
  %x = extractelement <4 x float> %v, i64 0
  %xi = bitcast float %x to i32
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %gid
  @@STORE@@
  ret void
}
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
declare void @air.store.device_coherent.i32.p1i32(i32, ptr addrspace(1))
!air.kernel = !{!0}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @heap_store, !1, !2}
!1 = !{}
!2 = !{!3, !7, !8}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
!4 = !{i32 0, i32 8, i32 0, !"@@ATYPE@@", !"a", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"texture2d<float, read_write>", !"b", !"air.indirect_argument", !6}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"@@AACCESS@@", !"air.arg_type_name", !"@@ATYPE@@", !"air.arg_name", !"a"}
!6 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"b"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!8 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// [`HEAP_BESIDE_STORE`] with `a` `read_write` and `b` written (`storage`) or `a` sampled and
/// `b` unused, storing with a coherent store (`coherent`) or a plain one.
fn heap_beside_store(storage: bool, coherent: bool) -> String {
    let (atype, aaccess, read_access, write) = if storage {
        (
            "texture2d<float, read_write>",
            "air.read_write",
            "3",
            "call void @air.write_texture_2d.v4f32(ptr addrspace(1) %b, <2 x i32> %c, <4 x float> zeroinitializer, i32 0, i32 3)",
        )
    } else {
        ("texture2d<float>", "air.sample", "0", "")
    };
    let store = if coherent {
        "call void @air.store.device_coherent.i32.p1i32(i32 %xi, ptr addrspace(1) %o)"
    } else {
        "store i32 %xi, ptr addrspace(1) %o, align 4"
    };
    HEAP_BESIDE_STORE
        .replace("@@ATYPE@@", atype)
        .replace("@@AACCESS@@", aaccess)
        .replace("@@READACCESS@@", read_access)
        .replace("@@WRITE@@", write)
        .replace("@@STORE@@", store)
}

/// For heaps: a heap storage class's reads and writes are storage-image accesses, which the Vulkan
/// memory model a coherent access selects orders only with `NonPrivateTexel`, which is not emitted.
/// So a heap texture read and written beside a coherent store is
/// `NotImplemented{Lowering(MemoryModel)}`, as a storage slot is
/// (`a_coherent_access_beside_a_storage_image_is_refused`), whether or not the device leaves the
/// class's out-of-bounds accesses to robustness. Where it does (`robustImageAccess2`), the class is
/// `Volatile`, and the refusal is the same typed one at selection, never the builder's
/// `GlobalRule::VolatileTarget`. A sampled heap texture beside the same coherent store, and
/// either one beside a plain store, compiles and passes L3.
#[test]
fn a_coherent_access_beside_a_heap_storage_image_is_refused() {
    let want = Unsupported::not_implemented(Feature::Lowering(LowerFamily::MemoryModel));
    let profiles = [
        ("moltenvk-m3", m3()),
        (
            "moltenvk-m3 without image robustness",
            without_robust_images("moltenvk-m3"),
        ),
    ];
    for (stem, p) in &profiles {
        assert!(p.fields().vulkan_memory_model, "{stem}");
        for storage in [true, false] {
            for coherent in [false, true] {
                let Some(bc) = assemble("heap_store", &heap_beside_store(storage, coherent)) else {
                    return;
                };
                let what = format!("{stem}: storage {storage}, coherent {coherent}");
                if storage && coherent {
                    let u = match compile(&bc, "heap_store", p) {
                        Ok(_) => panic!("{what}: compiled under the Vulkan memory model"),
                        Err(e) => unsupported(e),
                    };
                    assert_eq!(
                        (u.feature, u.needs),
                        (want.feature.clone(), want.needs.clone()),
                        "{what}"
                    );
                } else {
                    let a = compile(&bc, "heap_store", p)
                        .unwrap_or_else(|e| panic!("{what}: compile: {e}"));
                    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                }
            }
        }
    }
}

/// A heap handle's lookup is named by the memory its resource-id load reaches. A `u64` atomic on
/// the argument buffer's counter member lowers the buffer by device address, so the id's two words
/// are read through it and the lookup is `access.handle.lookup.psb`, never the descriptor view's
/// `access.handle.lookup`; a profile without the heap or the `u64` atomic refuses naming it.
#[test]
fn a_heap_lookup_through_device_addresses_is_named_by_its_reach() {
    let Some(bc) = assemble(
        "lookup_psb",
        r#"%struct.R = type { ptr addrspace(1), i64 }
define void @lookup_psb(ptr addrspace(1) %r, ptr addrspace(1) %out, i32 %gid) {
  %pt = getelementptr inbounds %struct.R, ptr addrspace(1) %r, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(1) %pt, align 8
  %c = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  %s = call ptr addrspace(2) @air.get_read_sampler()
  %x = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %c, <2 x i32> zeroinitializer, i32 0, i32 0)
  %v = extractvalue { <4 x float>, i8 } %x, 0
  %o = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %gid
  store <4 x float> %v, ptr addrspace(1) %o, align 16
  %pn = getelementptr inbounds %struct.R, ptr addrspace(1) %r, i64 0, i32 1
  %n = atomicrmw add ptr addrspace(1) %pn, i64 1 monotonic, align 8
  ret void
}
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
!air.kernel = !{!0}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @lookup_psb, !1, !2}
!1 = !{}
!2 = !{!3, !7, !8}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float>", !"t", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"ulong", !"n", !"air.indirect_argument", !6}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float>", !"air.arg_name", !"t"}
!6 = !{i32 1, !"air.indirect_constant", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"ulong", !"air.arg_name", !"n"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!8 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
"#,
    ) else {
        return;
    };
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        match compile(&bc, "lookup_psb", &p) {
            Ok(a) => {
                assert!(has_sampled_heap(&p), "{stem}");
                assert!(p.fields().leaf.buffer_int64_atomics, "{stem}");
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} lookup_psb: L3: {e}"));
                assert_eq!(uses(&a, "access.handle.lookup.psb"), 1, "{stem}");
                assert_eq!(uses(&a, "access.handle.lookup"), 0, "{stem}");
                assert_eq!(uses(&a, heap_read(&p)), 1, "{stem}");
                compiled += 1;
            }
            Err(e) => {
                let u = unsupported(e);
                assert!(
                    matches!(
                        u.feature,
                        Feature::ArgumentBufferTextures | Feature::Int64Atomics
                    ),
                    "{stem}: {u:?}"
                );
                assert!(matches!(u.needs, Needs::AnyOf(_)), "{stem}: {:?}", u.needs);
            }
        }
    }
    assert!(
        compiled > 0,
        "no committed profile has a heap and u64 atomics"
    );
}

/// Nested argument buffers: `top.mid` points to an argument buffer that holds a texture `t` and
/// `leaf`, a pointer to a third holding a texture `t`, a sampler `s` and a `read_write` texture
/// `rw`; `top.leaves` is an array of two pointers to such leaves. The kernel samples
/// `top.mid->leaf->t`, `top.mid->t` and `top.leaves[g & 1]->t` through `top.mid->leaf->s`, and
/// writes their sum to `top.mid->leaf->rw` and `out[g]`. `@@LEAVES@@` is the address of the third
/// texture's leaf: `%el`, the array's element.
const NESTED_HANDLES: &str = r#"%struct.Top = type { ptr addrspace(2), [2 x ptr addrspace(2)] }
%struct.Mid = type { ptr addrspace(2), %struct.T }
%struct.Leaf = type { %struct.T, %struct.S, %struct.T }
%struct.T = type { ptr addrspace(1) }
%struct.S = type { ptr addrspace(2) }
define void @nested_handles(ptr addrspace(2) %top, ptr addrspace(1) %out, i32 %g) {
  %pm = getelementptr inbounds %struct.Top, ptr addrspace(2) %top, i64 0, i32 0
  %mid = load ptr addrspace(2), ptr addrspace(2) %pm, align 8
  %pl = getelementptr inbounds %struct.Mid, ptr addrspace(2) %mid, i64 0, i32 0
  %leaf = load ptr addrspace(2), ptr addrspace(2) %pl, align 8
  %ps = getelementptr inbounds %struct.Leaf, ptr addrspace(2) %leaf, i64 0, i32 1, i32 0
  %s = load ptr addrspace(2), ptr addrspace(2) %ps, align 8
  %pt = getelementptr inbounds %struct.Leaf, ptr addrspace(2) %leaf, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %a = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> splat (float 5.000000e-01), i1 true, <2 x i32> zeroinitializer, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)
  %av = extractvalue { <4 x float>, i8 } %a, 0
  %pmt = getelementptr inbounds %struct.Mid, ptr addrspace(2) %mid, i64 0, i32 1, i32 0
  %mt = load ptr addrspace(1), ptr addrspace(2) %pmt, align 8
  %b = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %mt, ptr addrspace(2) %s, <2 x float> splat (float 5.000000e-01), i1 true, <2 x i32> zeroinitializer, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)
  %bv = extractvalue { <4 x float>, i8 } %b, 0
  %k = and i32 %g, 1
  %kk = zext i32 %k to i64
  %pe = getelementptr inbounds %struct.Top, ptr addrspace(2) %top, i64 0, i32 1, i64 %kk
  %el = load ptr addrspace(2), ptr addrspace(2) %pe, align 8
  %pet = getelementptr inbounds %struct.Leaf, ptr addrspace(2) @@LEAVES@@, i64 0, i32 0, i32 0
  %et = load ptr addrspace(1), ptr addrspace(2) %pet, align 8
  %c = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %et, ptr addrspace(2) %s, <2 x float> splat (float 5.000000e-01), i1 true, <2 x i32> zeroinitializer, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)
  %cv = extractvalue { <4 x float>, i8 } %c, 0
  %ab = fadd <4 x float> %av, %bv
  %v = fadd <4 x float> %ab, %cv
  %prw = getelementptr inbounds %struct.Leaf, ptr addrspace(2) %leaf, i64 0, i32 2, i32 0
  %rw = load ptr addrspace(1), ptr addrspace(2) %prw, align 8
  %at = insertelement <2 x i32> zeroinitializer, i32 %g, i64 0
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %rw, <2 x i32> %at, <4 x float> %v, i32 0, i32 3)
  %o = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %g
  store <4 x float> %v, ptr addrspace(1) %o, align 16
  ret void
}
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
!air.kernel = !{!0}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @nested_handles, !1, !2}
!1 = !{}
!2 = !{!3, !18, !19}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 24, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 24, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Top", !"air.arg_name", !"top"}
!4 = !{i32 0, i32 8, i32 0, !"Mid", !"mid", !"air.indirect_argument", !5, i32 8, i32 8, i32 2, !"Leaf", !"leaves", !"air.indirect_argument", !10}
!5 = !{i32 0, !"air.indirect_buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !6, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Mid", !"air.arg_name", !"mid"}
!6 = !{i32 0, i32 8, i32 0, !"Leaf", !"leaf", !"air.indirect_argument", !7, i32 8, i32 8, i32 0, !"texture2d<float, sample>", !"t", !"air.indirect_argument", !8}
!7 = !{i32 0, !"air.indirect_buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !11, !"air.arg_type_size", i32 24, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Leaf", !"air.arg_name", !"leaf"}
!8 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!10 = !{i32 1, !"air.indirect_buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !11, !"air.arg_type_size", i32 24, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Leaf", !"air.arg_name", !"leaves"}
!11 = !{i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"t", !"air.indirect_argument", !12, i32 8, i32 8, i32 0, !"sampler", !"s", !"air.indirect_argument", !13, i32 16, i32 8, i32 0, !"texture2d<float, read_write>", !"rw", !"air.indirect_argument", !14}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!13 = !{i32 1, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!14 = !{i32 2, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"rw"}
!18 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!19 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"g"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// A texture or sampler loaded from a nested argument buffer is a heap handle like one loaded from
/// the argument buffer an entry parameter binds. [`NESTED_HANDLES`]'s textures two and three
/// buffers deep, the one in an element of an array of nested buffers, the sampler three deep and
/// the `read_write` texture three deep each read their resource id through the nested buffer's
/// device address (`access.handle.lookup.psb`, `access.handle.sampler_lookup.psb`) and search the
/// stage's tables: four texture lookups and one sampler lookup, every module L3-valid. A profile
/// without the heap or device addresses refuses typed, naming what it lacks; none refuses
/// `NotImplemented`. A texture loaded through a `select` of two nested buffers' addresses has no
/// known member and stays `NotImplemented`.
#[test]
fn nested_argument_buffer_handles_are_heap_lookups() {
    let text = NESTED_HANDLES.replace("@@LEAVES@@", "%el");
    let Some(bc) = assemble("nested_handles", &text) else {
        return;
    };
    let mut compiled = Vec::new();
    for (stem, p) in committed_profiles().expect("profiles") {
        match compile(&bc, "nested_handles", &p) {
            Ok(a) => {
                assert!(has_sampled_heap(&p), "{stem}");
                link_and_validate(&a, &p)
                    .unwrap_or_else(|e| panic!("{stem} nested_handles: L3: {e}"));
                assert_eq!(uses(&a, "access.handle.lookup.psb"), 4, "{stem}");
                assert_eq!(uses(&a, "access.handle.sampler_lookup.psb"), 1, "{stem}");
                assert_eq!(uses(&a, "access.handle.lookup"), 0, "{stem}");
                assert!(texture_slots(&a).is_empty(), "{stem}");
                compiled.push(stem);
            }
            Err(e) => {
                let u = unsupported(e);
                assert!(matches!(u.needs, Needs::AnyOf(_)), "{stem}: {u:?}");
            }
        }
    }
    for stem in [
        "lavapipe",
        "moltenvk-m3",
        "amd-780m-radv",
        "nvidia-rtx5080-laptop",
    ] {
        assert!(compiled.contains(&stem.to_string()), "{stem}: {compiled:?}");
    }

    // The third texture's leaf: `top.mid->leaf` or the array's element, as `g` says.
    let merged = NESTED_HANDLES.replace("@@LEAVES@@", "%sel").replace(
        "  %pet = getelementptr",
        "  %odd = icmp ne i32 %k, 0\n  %sel = select i1 %odd, ptr addrspace(2) %el, ptr addrspace(2) %leaf\n  %pet = getelementptr",
    );
    assert!(merged.contains("%sel = select"), "{merged}");
    let Some(bc) = assemble("nested_handles_merged", &merged) else {
        return;
    };
    let u = unsupported(compile(&bc, "nested_handles", &m3()).expect_err("refused"));
    assert_eq!(
        (u.feature, u.needs),
        (Feature::ArgumentBufferTextures, Needs::NotImplemented)
    );
}

/// The owned fixture `reflection_argument_buffers` samples the texture of its nested argument
/// buffer (`args.sub->t`), which was refused `NotImplemented` before. It compiles and passes L3
/// wherever the heap and device addresses are, reading that texture's id through the nested
/// buffer's address (`access.handle.lookup.psb`), and every other profile refuses typed.
#[test]
fn the_reflection_fixture_samples_its_nested_argument_buffer_s_texture() {
    let bc = match llvm_assemble(&fixtures_dir().join("reflection/argument_buffers.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected argument_buffers: {why}"),
    };
    let entry = "reflection_argument_buffers";
    let mut compiled = Vec::new();
    for (stem, p) in committed_profiles().expect("profiles") {
        match compile(&bc, entry, &p) {
            Ok(a) => {
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {entry}: L3: {e}"));
                assert_eq!(uses(&a, "access.handle.lookup.psb"), 1, "{stem}");
                compiled.push(stem);
            }
            Err(e) => {
                let u = unsupported(e);
                assert!(matches!(u.needs, Needs::AnyOf(_)), "{stem}: {u:?}");
            }
        }
    }
    for stem in [
        "lavapipe",
        "moltenvk-m3",
        "amd-780m-radv",
        "nvidia-rtx5080-laptop",
    ] {
        assert!(compiled.contains(&stem.to_string()), "{stem}: {compiled:?}");
    }
}

/// Whether `p` has what a sampled heap array needs: a runtime, partially bound, update-after-bind
/// array indexed non-uniformly.
fn has_sampled_heap(p: &DeviceProfile) -> bool {
    let d = &p.fields().descriptors;
    d.runtime_descriptor_array
        && d.partially_bound
        && d.non_uniform_sampled_image
        && d.update_after_bind_sampled_image
}

const OP_DECORATE: u16 = 71;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_ARRAY_LENGTH: u16 = 68;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_CALL: u16 = 57;
const OP_ULESS_THAN_EQUAL: u16 = 178;
const DECORATION_NON_UNIFORM: u32 = 5300;

/// The `NonUniform` decorations of a module.
fn non_uniform(a: &FunctionArtifact) -> usize {
    let words = &a.module.words;
    insts(words)
        .iter()
        .filter(|(op, at)| *op == OP_DECORATE && words[at + 2] == DECORATION_NON_UNIFORM)
        .count()
}

/// The `(set, binding)` of every binding symbol of `t` whose kind satisfies `f`.
fn bindings_of(t: &LinkTable, f: fn(&ResourceKind) -> bool) -> Vec<(u32, u32)> {
    t.bindings
        .iter()
        .filter(|(s, _)| f(&s.kind))
        .map(|(_, b)| *b)
        .collect()
}

fn is_heap(k: &ResourceKind) -> bool {
    matches!(k, ResourceKind::Heap(_))
}

fn is_table(k: &ResourceKind) -> bool {
    matches!(k, ResourceKind::Synthetic(SyntheticBinding::HandleTable))
}

/// Every heap array `a` declares is one of the binder's heap classes on `p`, so the binder has a
/// neutral of its type for element 0: one class, and its unsigned integer twin where a float read
/// or query of its sampled group declares one.
fn assert_binder_classes(a: &FunctionArtifact, p: &DeviceProfile, stem: &str) {
    let classes = heap_classes(p);
    let mut n = 0;
    for d in &a.interface.descriptors {
        if let ResourceKind::Heap(c) = d.slot.kind {
            assert!(classes.contains(&c), "{stem}: {c:?}");
            n += 1;
        }
    }
    assert_eq!(n, 1 + siblings(a), "{stem}");
}

/// The sibling classes `a` declares: a sampled `Sint` or `Uint` heap class beside the sampled
/// float class of its shape. A float read or query of that class's group declares the unsigned
/// integer twin (its members of a form a heap reads held there raw, `HeapPlan::raw`), and its
/// reads and queries the class of each other numeric type (a member of another numeric class
/// read in its own, `HeapPlan::cross`).
fn siblings(a: &FunctionArtifact) -> usize {
    let images = heap_image_classes(a);
    images
        .iter()
        .filter(|c| {
            c.access == ImageAccess::Sampled
                && matches!(c.sampled_type, SampledType::Sint | SampledType::Uint)
                && images.contains(&ImageHandleClass {
                    sampled_type: SampledType::Float,
                    ..**c
                })
        })
        .count()
}

/// The set-1 bindings of a module of one heap class: `(1, 0)`, then its siblings'
/// ([`siblings`]).
fn one_heap(a: &FunctionArtifact) -> Vec<(u32, u32)> {
    (0u32..).take(1 + siblings(a)).map(|b| (1, b)).collect()
}

/// A texture loaded from an argument buffer is a heap handle: one handle-table lookup of the
/// member's resource id, a non-uniform element of the set-1 array of its class, and the null
/// arm's zero texel. A profile without the heap's descriptor-indexing features refuses it naming
/// them.
#[test]
fn an_argument_buffer_texture_is_a_heap_lookup() {
    let Some(bc) = fixture() else { return };
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        match compile(&bc, "argument_buffer_read", &p) {
            Ok(a) => {
                assert!(has_sampled_heap(&p), "{stem}");
                let (t, _) = link_and_validate(&a, &p)
                    .unwrap_or_else(|e| panic!("{stem} argument_buffer_read: L3: {e}"));
                assert_eq!(uses(&a, "access.handle.lookup"), 1, "{stem}");
                assert_eq!(uses(&a, heap_read(&p)), 1, "{stem}");
                assert!(a.interface.needs_handle_translation, "{stem}");
                // The search reads the table's length once; the element's chain and its load
                // are non-uniform.
                assert_eq!(count(&a, OP_ARRAY_LENGTH), 1, "{stem}");
                assert!(count(&a, OP_ACCESS_CHAIN) >= 1, "{stem}");
                // and the twin's, which the read's raw arm loads, and the `Sint` and `Uint`
                // classes' the read's siblings load.
                assert_eq!(siblings(&a), 2, "{stem}");
                assert_eq!(non_uniform(&a), 8, "{stem}");
                // No static texture slot: the texture is the heap's.
                assert!(texture_slots(&a).is_empty(), "{stem}");
                assert_eq!(bindings_of(&t, is_heap), one_heap(&a), "{stem}");
                assert_binder_classes(&a, &p, &stem);
                let table = bindings_of(&t, is_table);
                assert_eq!(table.len(), 1, "{stem}");
                assert_eq!(table[0].0, 0, "{stem}");
                compiled += 1;
            }
            Err(e) => {
                assert!(!has_sampled_heap(&p), "{stem}: {e}");
                let u = unsupported(e);
                assert_eq!(u.feature, Feature::ArgumentBufferTextures, "{stem}");
                assert!(matches!(u.needs, Needs::AnyOf(_)), "{stem}: {:?}", u.needs);
            }
        }
    }
    assert!(compiled > 0, "no committed profile has a heap");
}

/// Through the public compile API the heap pipeline's layout is the heap's (set 1, one partially
/// bound, update-after-bind array in an update-after-bind pool), the device accepts its
/// requirements, and with nothing resident the binder plans the table's null entry and the array's
/// neutral element 0 (`src/tests/engine_heap_binder.rs` binds textures).
#[test]
fn a_heap_pipeline_compiles_and_plans() {
    let Some(bc) = fixture() else { return };
    let dev = m3_device();
    let lib = Library::from_air(&bc).expect("library");
    let f = lib.function("argument_buffer_read").expect("entry");
    let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    let pipe = dev
        .compile_compute_pipeline(&d, &Budget::standard())
        .expect("pipeline");
    let sets = &pipe.layout.sets;
    assert_eq!(sets.len(), 2);
    assert_eq!(
        sets[1].flags,
        vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL
    );
    // The float class, and its unsigned integer twin, which the read declares, and the signed
    // integer class its sibling reads.
    assert_eq!(sets[1].bindings.len(), 3);
    assert_siblings(&sets[1].bindings);
    let heap = &sets[1].bindings[0];
    assert_eq!(heap.ty, vk::DescriptorType::SAMPLED_IMAGE);
    assert!(matches!(heap.source, BindingSource::Heap(_)));
    assert!(heap.count > 1);
    assert_eq!(heap.flags, heap_flags(&pipe.requirements));
    assert!(pipe
        .requirements
        .features
        .contains(&VkFeature::DescriptorBindingUpdateUnusedWhilePending));
    assert!(sets[0].bindings.iter().any(|b| matches!(
        b.source,
        BindingSource::Synthetic {
            binding: SyntheticBinding::HandleTable,
            ..
        }
    )));
    let plan = pipe
        .plan(&dev, &BindState::default().shape())
        .expect("plan");
    assert_empty_heap(&plan, 1, 3);
}

/// Set 1 of a layout whose one sampled float class a read declares its siblings beside: the heap
/// bindings are the float class (first), then the `Sint` and `Uint` classes of its
/// shape (the twin and the siblings), all in the same stages.
fn assert_siblings(bindings: &[crate::pipeline::layout::DescriptorBinding]) {
    let class = |b: &crate::pipeline::layout::DescriptorBinding| match b.source {
        BindingSource::Heap(HandleClass::Image(c)) => c,
        other => panic!("{other:?}"),
    };
    let float = class(&bindings[0]);
    assert_eq!(float.sampled_type, SampledType::Float);
    let others: Vec<ImageHandleClass> = bindings[1..].iter().map(class).collect();
    assert_eq!(
        others,
        [SampledType::Sint, SampledType::Uint].map(|sampled_type| ImageHandleClass {
            sampled_type,
            ..float
        })
    );
    for b in &bindings[1..] {
        assert_eq!(bindings[0].stages, b.stages);
    }
}

/// With nothing resident, a heap plan writes each handle table as its null entry alone and each
/// of the `arrays` heap arrays' element 0, the class's neutral.
fn assert_empty_heap(plan: &BindingPlan, tables: usize, arrays: usize) {
    let table_writes: Vec<_> = plan
        .descriptors
        .iter()
        .filter(|d| matches!(d.source, WriteSource::HandleTable { .. }))
        .collect();
    assert_eq!(table_writes.len(), tables);
    for d in table_writes {
        assert!(matches!(
            d.source,
            WriteSource::HandleTable { bytes_len: 16, .. }
        ));
    }
    let heap: Vec<_> = plan.descriptors.iter().filter(|d| d.set == 1).collect();
    assert_eq!(heap.len(), arrays);
    for d in heap {
        assert_eq!(d.array_element, 0);
        assert!(matches!(d.source, WriteSource::Neutral(_)));
    }
}

/// The binding flags of a pipeline's set-1 arrays, by its `requirements`:
/// `PARTIALLY_BOUND | UPDATE_AFTER_BIND`, and `UPDATE_UNUSED_WHILE_PENDING` exactly where the
/// pipeline requires `descriptorBindingUpdateUnusedWhilePending` (link's
/// `heap_unused_while_pending`).
pub(super) fn heap_flags(requirements: &Requirements) -> vk::DescriptorBindingFlags {
    let flags =
        vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND;
    match requirements
        .features
        .contains(&VkFeature::DescriptorBindingUpdateUnusedWhilePending)
    {
        true => flags | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING,
        false => flags,
    }
}

/// The committed `moltenvk-m3` profile as a device.
fn m3_device() -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// A render pipeline whose vertex and fragment functions both read a texture from an argument
/// buffer: both stages share the one set-1 array of the class and each reads its own handle
/// table, which `RenderPipeline::plan` writes.
#[test]
fn a_render_heap_pipeline_shares_its_array() {
    let Some(bc) = assemble(
        "heap_render",
        r#"%struct.T = type { ptr addrspace(1) }
%struct.R = type { %struct.T }
define <{ <4 x float> }> @heap_vertex(i32 noundef %vid, ptr addrspace(2) %r) {
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %c = insertelement <2 x i32> zeroinitializer, i32 %vid, i64 0
  %v = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %c, <2 x i32> zeroinitializer, i32 0, i32 0)
  %p = extractvalue { <4 x float>, i8 } %v, 0
  %o = insertvalue <{ <4 x float> }> undef, <4 x float> %p, 0
  ret <{ <4 x float> }> %o
}
define <4 x float> @heap_fragment(<4 x float> noundef %pos, ptr addrspace(2) %r) {
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %v = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 0)
  %c = extractvalue { <4 x float>, i8 } %v, 0
  ret <4 x float> %c
}
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
!air.vertex = !{!20}
!air.fragment = !{!0}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @heap_fragment, !1, !2}
!1 = !{!11}
!11 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!12, !3}
!12 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!3 = !{i32 1, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, read>", !"t", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
!20 = !{ptr @heap_vertex, !21, !23}
!21 = !{!22}
!22 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!23 = !{!24, !25}
!24 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!25 = !{i32 1, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
"#,
    ) else {
        return;
    };
    let dev = m3_device();
    let lib = Library::from_air(&bc).expect("library");
    let v = lib.function("heap_vertex").expect("vertex");
    let f = lib.function("heap_fragment").expect("fragment");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(v)));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(f)));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let pipe = dev
        .compile_render_pipeline(&d, &Budget::standard())
        .expect("render pipeline");
    let sets = &pipe.layout.sets;
    assert_eq!(sets.len(), 2);
    // and the float class's twin, which the reads declare, and the signed integer class their
    // siblings read.
    assert_eq!(sets[1].bindings.len(), 3);
    assert_siblings(&sets[1].bindings);
    let heap = &sets[1].bindings[0];
    assert!(matches!(heap.source, BindingSource::Heap(_)));
    assert_eq!(
        heap.stages,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
    );
    let tables = sets[0]
        .bindings
        .iter()
        .filter(|b| {
            matches!(
                b.source,
                BindingSource::Synthetic {
                    binding: SyntheticBinding::HandleTable,
                    ..
                }
            )
        })
        .count();
    assert_eq!(tables, 2, "one handle table per stage");
    let plan = pipe
        .plan(&dev, &BindState::default().shape())
        .expect("plan");
    assert_empty_heap(&plan, 2, 3);
}

/// Two members of one argument buffer merged by a select form one heap group: a storage class
/// written through a guard on the handle (a null handle drops the write), queried and tested
/// for null through the same element.
#[test]
fn merged_argument_buffer_textures_share_one_heap_class() {
    let Some(bc) = assemble(
        "heap_write",
        r#"%struct.T = type { ptr addrspace(1) }
%struct.R = type { %struct.T, %struct.T }
define void @heap_write(ptr addrspace(2) %r, ptr addrspace(1) %out, i32 %gid) {
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %pu = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 1, i32 0
  %u = load ptr addrspace(1), ptr addrspace(2) %pu, align 8
  %odd = trunc i32 %gid to i1
  %h = select i1 %odd, ptr addrspace(1) %t, ptr addrspace(1) %u
  %c = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %h, <2 x i32> %c, <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, i32 0, i32 2)
  %n = call i1 @air.is_null_texture_2d(ptr addrspace(1) %h)
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %h, i32 0)
  %z = zext i1 %n to i32
  %s = add i32 %w, %z
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %gid
  store i32 %s, ptr addrspace(1) %o, align 4
  ret void
}
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
declare i1 @air.is_null_texture_2d(ptr addrspace(1))
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
!air.kernel = !{!0}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @heap_write, !1, !2}
!1 = !{}
!2 = !{!3, !7, !8}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, write>", !"t", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"texture2d<float, write>", !"u", !"air.indirect_argument", !6}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"t"}
!6 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"u"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!8 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
"#,
    ) else {
        return;
    };
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let d = &p.fields().descriptors;
        let heap = d.runtime_descriptor_array
            && d.partially_bound
            && d.non_uniform_storage_image
            && d.update_after_bind_storage_image
            && p.fields().images.storage_write_without_format;
        match compile(&bc, "heap_write", &p) {
            Ok(a) => {
                assert!(heap, "{stem}");
                let (t, _) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
                assert_eq!(uses(&a, "access.handle.lookup"), 2, "{stem}");
                // Both loads call the module's one search function, so its 16 steps (one
                // `OpULessThanEqual` each: of the `u64` ids with `shaderInt64`, else of the low
                // words) and its read of the table's length are emitted once. The module was
                // about 3,500 words with a search per load; it is within 1,600 with `u64` ids
                // and 2,300 with word ids, the bounds below.
                // The write's normalized channels are the device's conversion: noted where the
                // profile is not measured to store Metal's codes, or where it stores a texture
                // of a 1-bit alpha, which it rounds otherwise.
                let measured =
                    p.takes_driver_deviation(crate::error::DriverDeviation::MetalNormStorageWrites);
                let one_bit = crate::metal::format::one_bit_storage(
                    crate::metal::format::StorageUse::texture(false, true),
                    &p,
                );
                assert_eq!(
                    a.precision
                        .contains(&crate::pipeline::PrecisionNote::DeviceColorConversion),
                    !measured || one_bit,
                    "{stem}"
                );
                assert_eq!(count(&a, OP_FUNCTION), 2, "{stem}");
                assert_eq!(count(&a, OP_FUNCTION_CALL), 2, "{stem}");
                assert_eq!(count(&a, OP_ULESS_THAN_EQUAL), 16, "{stem}");
                assert_eq!(count(&a, OP_ARRAY_LENGTH), 1, "{stem}");
                // And an `A8Unorm` member's alpha moved into red where the device holds such a
                // texture so (`metal::format::alpha_in_red`): the class index's read and the
                // conversion.
                let a8 = crate::metal::format::alpha_in_red(crate::metal::PixelFormat::A8Unorm, &p);
                let alpha = if a8 { 100 } else { 0 };
                // And a member the heap views through its storage choice's alias converted
                // (`bind::heap::storage_conversions`): an sRGB member's encode, the device's
                // conversion, noted, and a blue-first member's exchange.
                let conversions = crate::bind::heap::storage_conversions(&p);
                let srgb = conversions.iter().any(|c| {
                    matches!(
                        c.class,
                        crate::pipeline::lower_inputs::StorageImageClass::SrgbEncode
                            | crate::pipeline::lower_inputs::StorageImageClass::SrgbEncodeBgra
                    )
                });
                assert_eq!(
                    a.precision
                        .contains(&crate::pipeline::PrecisionNote::DevicePackConversion),
                    srgb,
                    "{stem}"
                );
                let converted = if conversions.is_empty() { 0 } else { 700 };
                let bound = alpha + converted + if p.fields().leaf.int64 { 1_600 } else { 2_300 };
                assert!(
                    a.module.words.len() <= bound,
                    "{stem}: {} words over {bound} (alpha in red {a8}, {} conversions)",
                    a.module.words.len(),
                    conversions.len()
                );
                assert_eq!(uses(&a, "texture.write.heap"), 1, "{stem}");
                assert_eq!(uses(&a, "texture.query.heap"), 1, "{stem}");
                assert_eq!(uses(&a, "texture.is_null.heap"), 1, "{stem}");
                // One class: both members and the merged handle name one array.
                assert_eq!(bindings_of(&t, is_heap), one_heap(&a), "{stem}");
                assert_binder_classes(&a, &p, &stem);
                compiled += 1;
            }
            Err(e) => {
                assert!(!heap, "{stem}: {e}");
                let u = unsupported(e);
                assert_eq!(u.feature, Feature::ArgumentBufferTextures, "{stem}");
                assert!(matches!(u.needs, Needs::AnyOf(_)), "{stem}: {:?}", u.needs);
            }
        }
    }
    assert!(compiled > 0, "no committed profile has a storage heap");
}

/// A kernel writing the `half4` (or, `float`, the `float4`) of buffer 1 at its index through an
/// argument buffer's `texture2d<half, write>` (`texture2d<float, write>`).
fn heap_texel_write_kernel(float: bool) -> String {
    let (ty, elem, size, name) = match float {
        true => ("<4 x float>", "v4f32", 16, "float"),
        false => ("<4 x half>", "v4f16", 8, "half"),
    };
    format!(
        r#"%struct.T = type {{ ptr addrspace(1) }}
define void @heap_texel_write(ptr addrspace(2) %r, ptr addrspace(1) %src, i32 %gid) {{
  %pt = getelementptr inbounds %struct.T, ptr addrspace(2) %r, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %p = getelementptr inbounds {ty}, ptr addrspace(1) %src, i32 %gid
  %v = load {ty}, ptr addrspace(1) %p, align {size}
  %c = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  call void @air.write_texture_2d.{elem}(ptr addrspace(1) %t, <2 x i32> %c, {ty} %v, i32 0, i32 2)
  ret void
}}
declare void @air.write_texture_2d.{elem}(ptr addrspace(1), <2 x i32>, {ty}, i32, i32)
!air.kernel = !{{!0}}
!air.version = !{{!9}}
!air.language_version = !{{!10}}
!0 = !{{ptr @heap_texel_write, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !7, !8}}
!3 = !{{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"T", !"air.arg_name", !"r"}}
!4 = !{{i32 0, i32 8, i32 0, !"texture2d<{name}, write>", !"t", !"air.indirect_argument", !5}}
!5 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<{name}, write>", !"air.arg_name", !"t"}}
!7 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{name}4", !"air.arg_name", !"src"}}
!8 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!9 = !{{i32 2, i32 8, i32 0}}
!10 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// A `half` (or `float`) texel written through an argument buffer's texture is the device's write
/// on every committed profile: the module declares the one formatless float storage class, and
/// records `PrecisionNote::DeviceNanEncoding`, since the heap may hold a half-float, 32-bit or
/// packed small float texture, whose stored NaN is the device's; but for a `float` texel on a
/// profile whose stores of a `float` NaN into the half-float and the packed small float formats
/// are measured to be Metal's (`DriverDeviation::MetalHalfStorageNans` and
/// `DriverDeviation::MetalSmallFloatNans`: MoltenVK on the M3 Max's identity), which then
/// requires `VkProperty::MetalSmallFloatNans`.
#[test]
fn a_heap_texel_write_is_the_devices_write() {
    for float in [false, true] {
        let Some(bc) = assemble("heap_texel_write", &heap_texel_write_kernel(float)) else {
            return;
        };
        let mut compiled = 0;
        for (stem, p) in committed_profiles().expect("profiles") {
            let what = format!("{stem} float {float}");
            let a = match compile(&bc, "heap_texel_write", &p) {
                Ok(a) => a,
                Err(e) => {
                    let u = unsupported(e);
                    assert_eq!(u.feature, Feature::ArgumentBufferTextures, "{what}");
                    continue;
                }
            };
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_binder_classes(&a, &p, &what);
            let classes = heap_image_classes(&a);
            assert_eq!(classes.len(), 1, "{what}: {classes:?}");
            assert_eq!(classes[0].access, ImageAccess::Storage, "{what}");
            assert_eq!(classes[0].sampled_type, SampledType::Float, "{what}");
            let takes = |d| p.takes_driver_deviation(d);
            let small = takes(crate::error::DriverDeviation::MetalSmallFloatNans);
            let metal_s =
                float && small && takes(crate::error::DriverDeviation::MetalHalfStorageNans);
            assert_eq!(
                a.precision.contains(&PrecisionNote::DeviceNanEncoding),
                !metal_s,
                "{what}: {:?}",
                a.precision
            );
            assert_eq!(
                a.requirements
                    .properties
                    .contains(&crate::device::requirements::VkProperty::MetalSmallFloatNans),
                float && small,
                "{what}"
            );
            compiled += 1;
        }
        assert!(compiled > 0, "no committed profile writes through a heap");
    }
}

/// A kernel that reads, and queries the width of, the texture one of two adjacent argument-buffer
/// members names (`cond ? r.a : r.b`), in the frontend's form: a `select` of the two members'
/// addresses, a plain `i64` load of the resource id there, and its `inttoptr`. The second member's
/// access is `second`; with `escapes` the integer is also stored, so observed.
fn member_word_module(second: &str, escapes: bool) -> Option<Vec<u8>> {
    let access = second
        .split(", ")
        .nth(1)
        .unwrap_or("sample")
        .trim_end_matches('>');
    let store = if escapes {
        "  %q = getelementptr inbounds i64, ptr addrspace(1) %ids, i32 %gid\n  store i64 %id, ptr addrspace(1) %q, align 8\n"
    } else {
        ""
    };
    assemble(
        "heap_word",
        &format!(
            r#"%struct.T = type {{ ptr addrspace(1) }}
%struct.R = type {{ float, %struct.T, %struct.T }}
define void @heap_word(ptr addrspace(2) %r, ptr addrspace(1) %out, ptr addrspace(1) %ids, i32 %gid) {{
  %odd = and i32 %gid, 1
  %c = icmp eq i32 %odd, 0
  %pb = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 2
  %pa = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 1
  %p = select i1 %c, ptr addrspace(2) %pa, ptr addrspace(2) %pb
  %id = load i64, ptr addrspace(2) %p, align 8
  %h = inttoptr i64 %id to ptr addrspace(1)
{store}  %x = lshr i32 %gid, 1
  %xy = insertelement <2 x i32> <i32 poison, i32 0>, i32 %x, i64 0
  %smp = call ptr addrspace(2) @air.get_read_sampler()
  %v = call {{ <4 x half>, i8 }} @air.read_texture_2d.v4f16(ptr addrspace(1) %h, ptr addrspace(2) %smp, <2 x i32> %xy, <2 x i32> zeroinitializer, i32 0, i32 0)
  %t = extractvalue {{ <4 x half>, i8 }} %v, 0
  %e = extractelement <4 x half> %t, i64 0
  %f = fpext half %e to float
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %h, i32 0)
  %wf = uitofp i32 %w to float
  %s = fadd float %f, %wf
  %o = getelementptr inbounds float, ptr addrspace(1) %out, i32 %gid
  store float %s, ptr addrspace(1) %o, align 4
  ret void
}}
declare ptr addrspace(2) @air.get_read_sampler()
declare {{ <4 x half>, i8 }} @air.read_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
!air.kernel = !{{!0}}
!air.version = !{{!11}}
!air.language_version = !{{!12}}
!0 = !{{ptr @heap_word, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !8, !9, !10}}
!3 = !{{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 24, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 24, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}}
!4 = !{{i32 0, i32 4, i32 0, !"float", !"scale", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"texture2d<half, sample>", !"a", !"air.indirect_argument", !6, i32 16, i32 8, i32 0, !"{second}", !"b", !"air.indirect_argument", !7}}
!5 = !{{i32 0, !"air.indirect_constant", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"scale"}}
!6 = !{{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"a"}}
!7 = !{{i32 2, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.{access}", !"air.arg_type_name", !"{second}", !"air.arg_name", !"b"}}
!8 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}}
!9 = !{{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"ids"}}
!10 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!11 = !{{i32 2, i32 8, i32 0}}
!12 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
        ),
    )
}

/// The resource id of one of two adjacent `sample` members, loaded as an integer through a `select`
/// of their addresses and turned back into the handle, is the heap handle a load of the selected
/// member reads: one handle-table lookup of the id (either member's slot holds a `sample` texture,
/// so the class is the sampled one), read and queried as a heap texture. A profile without the
/// sampled heap refuses it naming the features. A pair of members of different accesses, whose
/// handle has no one class, and an integer the kernel also stores (the guest's bits observed) stay
/// refused.
#[test]
fn a_selected_member_resource_id_is_a_heap_lookup() {
    let Some(bc) = member_word_module("texture2d<half, sample>", false) else {
        return;
    };
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        match compile(&bc, "heap_word", &p) {
            Ok(a) => {
                assert!(has_sampled_heap(&p), "{stem}");
                let (t, _) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
                assert_eq!(uses(&a, "access.handle.lookup"), 1, "{stem}");
                assert_eq!(uses(&a, "texture.query.heap"), 1, "{stem}");
                assert_eq!(bindings_of(&t, is_heap), one_heap(&a), "{stem}");
                assert_binder_classes(&a, &p, &stem);
                compiled += 1;
            }
            Err(e) => {
                assert!(!has_sampled_heap(&p), "{stem}: {e}");
                let u = unsupported(e);
                assert_eq!(u.feature, Feature::ArgumentBufferTextures, "{stem}");
                assert!(matches!(u.needs, Needs::AnyOf(_)), "{stem}: {:?}", u.needs);
            }
        }
    }
    assert!(compiled > 0, "no committed profile has a sampled heap");
    let Some(mixed) = member_word_module("texture2d<half, write>", false) else {
        return;
    };
    let u = unsupported(compile(&mixed, "heap_word", &m3()).expect_err("mixed accesses"));
    assert_eq!(u.feature, Feature::ArgumentBufferTextures);
    assert_eq!(u.needs, Needs::NotImplemented);
    let Some(observed) = member_word_module("texture2d<half, sample>", true) else {
        return;
    };
    let e = compile(&observed, "heap_word", &m3()).expect_err("observed integer");
    assert!(matches!(&e, Error::Unsupported(_)), "observed integer: {e}");
}

/// The census name of a heap element's read on `p`: checked in the shader where the profile has no
/// `robustImageAccess2`, whose out-of-bounds read is Metal's.
fn heap_read(p: &DeviceProfile) -> &'static str {
    if p.fields().images.robust_image_access2 {
        "texture.read.heap"
    } else {
        "texture.read_clamped.heap"
    }
}

/// Whether `p` has what a storage heap array written without a format needs.
fn has_storage_heap(p: &DeviceProfile) -> bool {
    let d = &p.fields().descriptors;
    d.runtime_descriptor_array
        && d.partially_bound
        && d.non_uniform_storage_image
        && d.update_after_bind_storage_image
        && p.fields().images.storage_write_without_format
}

/// A heap entry of the fixture: its name, the profile features its heap needs, and the uses of
/// each strategy it compiles to.
type HeapCase = (
    &'static str,
    fn(&DeviceProfile) -> bool,
    &'static [(&'static str, u32)],
);

/// An argument-buffer texture reached through one dynamic element index is a heap handle of the
/// member the index walks: an element of a texture-array member (written through the storage heap),
/// and the texture of an element of an array of argument buffers (tested for null and read through
/// the sampled heap). Each is one lookup per load; a profile without the heap's features refuses it
/// naming them.
#[test]
fn dynamic_argument_buffer_elements_are_heap_lookups() {
    let Some(bc) = fixture() else { return };
    let cases: [HeapCase; 2] = [
        (
            "argument_buffer_array_write",
            has_storage_heap,
            &[("access.handle.lookup", 1), ("texture.write.heap", 1)],
        ),
        (
            "argument_buffer_struct_array",
            has_sampled_heap,
            &[
                ("access.handle.lookup", 1),
                ("texture.read.heap", 1),
                ("texture.is_null.heap", 1),
            ],
        ),
    ];
    for (entry, has_heap, strategies) in cases {
        let mut compiled = 0;
        for (stem, p) in committed_profiles().expect("profiles") {
            match compile(&bc, entry, &p) {
                Ok(a) => {
                    assert!(has_heap(&p), "{stem} {entry}");
                    let (t, _) = link_and_validate(&a, &p)
                        .unwrap_or_else(|e| panic!("{stem} {entry}: L3: {e}"));
                    for (name, n) in strategies {
                        let name = match *name {
                            "texture.read.heap" => heap_read(&p),
                            other => other,
                        };
                        assert_eq!(uses(&a, name), *n, "{stem} {entry} {name}");
                    }
                    assert!(texture_slots(&a).is_empty(), "{stem} {entry}");
                    assert_eq!(bindings_of(&t, is_heap), one_heap(&a), "{stem} {entry}");
                    assert_binder_classes(&a, &p, &stem);
                    compiled += 1;
                }
                Err(e) => {
                    assert!(!has_heap(&p), "{stem} {entry}: {e}");
                    let u = unsupported(e);
                    assert_eq!(u.feature, Feature::ArgumentBufferTextures, "{stem} {entry}");
                    assert!(matches!(u.needs, Needs::AnyOf(_)), "{stem}: {:?}", u.needs);
                }
            }
        }
        assert!(compiled > 0, "{entry}: no committed profile has its heap");
    }
}

const OP_TYPE_IMAGE: u16 = 25;

/// The `(Dim, Depth, Sampled)` operands of every `OpTypeImage` of a module.
fn image_types(a: &FunctionArtifact) -> Vec<(u32, u32, u32)> {
    let words = &a.module.words;
    insts(words)
        .iter()
        .filter(|(op, _)| *op == OP_TYPE_IMAGE)
        .map(|(_, at)| (words[at + 3], words[at + 4], words[at + 7]))
        .collect()
}

/// A depth texture read through an argument buffer is a member of the float 2D sampled heap class,
/// declared without `Depth`, beside the storage class its written texture needs: two lookups, one
/// heap read, one heap write, two set-1 arrays.
#[test]
fn an_argument_buffer_depth_texture_reads_through_the_float_class() {
    let Some(bc) = fixture() else { return };
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let heap = has_sampled_heap(&p) && has_storage_heap(&p);
        match compile(&bc, "argument_buffer_depth_read", &p) {
            Ok(a) => {
                assert!(heap, "{stem}");
                let (t, _) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
                assert_eq!(uses(&a, "access.handle.lookup"), 2, "{stem}");
                assert_eq!(uses(&a, heap_read(&p)), 1, "{stem}");
                assert_eq!(uses(&a, "texture.write.heap"), 1, "{stem}");
                assert_eq!(bindings_of(&t, is_heap).len(), 2, "{stem}");
                let classes: Vec<_> = a
                    .interface
                    .descriptors
                    .iter()
                    .filter_map(|d| match d.slot.kind {
                        ResourceKind::Heap(HandleClass::Image(c)) => Some(c),
                        _ => None,
                    })
                    .collect();
                assert!(
                    classes.iter().any(|c| c.access == ImageAccess::Sampled
                        && c.sampled_type == SampledType::Float
                        && !c.depth_compare
                        && c.dim == ImageDim::D2),
                    "{stem}: {classes:?}"
                );
                for c in &classes {
                    assert!(heap_classes(&p).contains(&HandleClass::Image(*c)), "{stem}");
                }
                // Every image type is 2D (Dim 1) and not a depth image.
                for (dim, depth, _) in image_types(&a) {
                    assert_eq!((dim, depth), (1, 0), "{stem}");
                }
                compiled += 1;
            }
            Err(e) => {
                assert!(!heap, "{stem}: {e}");
                let u = unsupported(e);
                assert_eq!(u.feature, Feature::ArgumentBufferTextures, "{stem}");
                assert!(matches!(u.needs, Needs::AnyOf(_)), "{stem}: {:?}", u.needs);
            }
        }
    }
    assert!(compiled > 0, "no committed profile has both heaps");
}

/// A fragment that samples an argument buffer's cube and queries its level count and width, and
/// only queries a cube array's array size. Both are sampled cube classes, held through the members'
/// cube views: the queries are the cube view's own (`OpImageQueryLevels`, `OpImageQuerySizeLod`),
/// with no 2D-array view and no division of the cube array's size by 6. A profile without
/// `imageCubeArray` refuses the cube array `CubeArray`; one without the sampled heap names the heap
/// features.
#[test]
fn an_argument_buffer_cube_is_queried_through_its_cube_view() {
    let Some(bc) = assemble(
        "heap_cube_queries",
        r#"%struct.T = type { ptr addrspace(1) }
%struct.R = type { %struct.T, %struct.T }
@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601020489, i64 0], align 8
define <4 x float> @heap_cube_queries(<4 x float> %pos, ptr addrspace(2) %r) {
  %pc = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %c = load ptr addrspace(1), ptr addrspace(2) %pc, align 8
  %pa = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 1, i32 0
  %a = load ptr addrspace(1), ptr addrspace(2) %pa, align 8
  %d = shufflevector <4 x float> %pos, <4 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %s = call { <4 x float>, i8 } @air.sample_texture_cube.v4f32(ptr addrspace(1) %c, ptr addrspace(2) @__air_sampler_state, <3 x float> %d, i1 false, float 0.0, float 0.0, i32 0)
  %x = extractvalue { <4 x float>, i8 } %s, 0
  %l = call i32 @air.get_num_mip_levels_texture_cube(ptr addrspace(1) %c)
  %w = call i32 @air.get_width_texture_cube(ptr addrspace(1) %c, i32 0)
  %n = call i32 @air.get_array_size_texture_cube_array(ptr addrspace(1) %a)
  %lw = add i32 %l, %w
  %q = add i32 %lw, %n
  %f = uitofp i32 %q to float
  %v = insertelement <4 x float> %x, float %f, i64 3
  ret <4 x float> %v
}
declare { <4 x float>, i8 } @air.sample_texture_cube.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, float, float, i32)
declare i32 @air.get_num_mip_levels_texture_cube(ptr addrspace(1))
declare i32 @air.get_width_texture_cube(ptr addrspace(1), i32)
declare i32 @air.get_array_size_texture_cube_array(ptr addrspace(1))
!air.fragment = !{!0}
!air.sampler_states = !{!11}
!air.version = !{!12}
!air.language_version = !{!13}
!0 = !{ptr @heap_cube_queries, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!5 = !{i32 1, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !6, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
!6 = !{i32 0, i32 8, i32 0, !"texturecube<float, sample>", !"c", !"air.indirect_argument", !7, i32 8, i32 8, i32 0, !"texturecube_array<float, sample>", !"a", !"air.indirect_argument", !8}
!7 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texturecube<float, sample>", !"air.arg_name", !"c"}
!8 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texturecube_array<float, sample>", !"air.arg_name", !"a"}
!11 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!12 = !{i32 2, i32 8, i32 0}
!13 = !{!"Metal", i32 4, i32 0, i32 0}
"#,
    ) else {
        return;
    };
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = match compile(&bc, "heap_cube_queries", &p) {
            Ok(a) => a,
            Err(e) => {
                let u = unsupported(e);
                if !has_sampled_heap(&p) {
                    assert_eq!(u.feature, Feature::ArgumentBufferTextures, "{stem}");
                    assert!(matches!(u.needs, Needs::AnyOf(_)), "{stem}: {:?}", u.needs);
                } else {
                    assert!(!p.fields().images.cube_array, "{stem}: {u:?}");
                    assert_eq!(u.feature, Feature::CubeArray, "{stem}");
                }
                continue;
            }
        };
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let mut cubes: Vec<_> = heap_image_classes(&a)
            .into_iter()
            .map(|c| (c.dim, c.arrayed, c.access, c.sampled_type))
            .collect();
        cubes.sort();
        // Each with its unsigned integer twin, which its queries declare, and the sampled cube's
        // signed integer class, which its queries query a member of that class through (the
        // only-queried cube array has no class of its own).
        let (f, i, u) = (SampledType::Float, SampledType::Sint, SampledType::Uint);
        assert_eq!(
            cubes,
            vec![
                (ImageDim::Cube, false, ImageAccess::Sampled, f),
                (ImageDim::Cube, false, ImageAccess::Sampled, i),
                (ImageDim::Cube, false, ImageAccess::Sampled, u),
                (ImageDim::Cube, true, ImageAccess::Sampled, f),
                (ImageDim::Cube, true, ImageAccess::Sampled, u)
            ],
            "{stem}"
        );
        // Every image type is a cube (Dim 3), none a 2D array.
        for (dim, _, _) in image_types(&a) {
            assert_eq!(dim, 3, "{stem}");
        }
        // Each query twice: of the float class's element and of its twin's; and the sampled cube's
        // a third time, of the signed integer class's (its twin's arm also queries an unsigned
        // integer member).
        assert_eq!(count(&a, OP_IMAGE_QUERY_LEVELS), 3, "{stem}");
        assert_eq!(count(&a, OP_IMAGE_QUERY_SIZE_LOD), 5, "{stem}");
        assert_eq!(count(&a, OP_UDIV), 0, "{stem}");
        compiled += 1;
    }
    assert!(
        compiled > 0,
        "no committed profile has the sampled heap and cube arrays"
    );
}

const OP_IMAGE_QUERY_SIZE_LOD: u16 = 103;
const OP_IMAGE_QUERY_LEVELS: u16 = 106;
const OP_UDIV: u16 = 134;

/// A dynamic index over a struct whose texture members differ names no one member: refused
/// `ArgumentBufferTextures NotImplemented`, as a span over several members is.
#[test]
fn a_dynamic_index_over_several_members_is_not_implemented() {
    let Some(bc) = assemble(
        "heap_members",
        r#"%struct.T = type { ptr addrspace(1) }
%struct.R = type { %struct.T, %struct.T }
define void @heap_members(ptr addrspace(2) %r, ptr addrspace(1) %out, i32 %gid) {
  %i = zext i32 %gid to i64
  %pt = getelementptr inbounds %struct.T, ptr addrspace(2) %r, i64 %i, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %s = call ptr addrspace(2) @air.get_read_sampler()
  %c = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  %v = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %c, <2 x i32> zeroinitializer, i32 0, i32 0)
  %x = extractvalue { <4 x float>, i8 } %v, 0
  %o = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %gid
  store <4 x float> %x, ptr addrspace(1) %o, align 16
  ret void
}
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
!air.kernel = !{!0}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @heap_members, !1, !2}
!1 = !{}
!2 = !{!3, !7, !8}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"t", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"texture2d<float, write>", !"u", !"air.indirect_argument", !6}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!6 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"u"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!8 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
"#,
    ) else {
        return;
    };
    let u = unsupported(compile(&bc, "heap_members", &m3()).expect_err("refused"));
    assert_eq!(u.feature, Feature::ArgumentBufferTextures);
    assert_eq!(u.needs, Needs::NotImplemented);
}

/// A heap texture sampled through a runtime `[[sampler(0)]]` reads a set-0 heap sampler binding of
/// that slot, with no immutable sampler (the binder writes the bound sampler), and scales a
/// pixel-coordinate sampler's coordinate behind the slot's `SamplerState` specialization, as a slot
/// texture's runtime sampler does. It passes L3 on every profile with the sampled and storage heaps
/// and names the heap features elsewhere.
#[test]
fn a_heap_texture_samples_through_a_runtime_sampler() {
    let Some(bc) = fixture() else { return };
    let entry = "argument_buffer_runtime_sampler";
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = match compile(&bc, entry, &p) {
            Ok(a) => a,
            Err(e) => {
                assert!(!has_sampled_heap(&p), "{stem}: {e}");
                let u = unsupported(e);
                assert_eq!(u.feature, Feature::ArgumentBufferTextures, "{stem}");
                assert!(matches!(u.needs, Needs::AnyOf(_)), "{stem}: {:?}", u.needs);
                continue;
            }
        };
        let (t, _) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert_eq!(uses(&a, "sampler.heap.sample"), 1, "{stem}");
        assert_eq!(
            heap_samplers(&a),
            vec![CombinedSampler::new(SamplerRef::Slot(0), false, false)],
            "{stem}"
        );
        let bound = bindings_of(&t, |k| {
            matches!(k, ResourceKind::HeapSampler(SamplerRef::Slot(0)))
        });
        assert_eq!(bound.len(), 1, "{stem}");
        assert_eq!(bound[0].0, 0, "{stem}");
        assert!(
            a.facts.iter().any(|f| matches!(
                f,
                RuntimeFact::SamplerCoordinates { slot } if slot.index == 0
            )),
            "{stem}"
        );
        // The device's sampled image, and the raw-held emulation's four filtered taps and its
        // nearest one, fetched from the unsigned twin.
        assert_eq!(count(&a, OP_SAMPLED_IMAGE), 1, "{stem}");
        assert_eq!(count(&a, OP_IMAGE_FETCH), 5, "{stem}");
        compiled += 1;
    }
    assert!(compiled > 0, "no committed profile has the heaps");
}

/// A cube's explicit gradients through a sampler loaded from an argument buffer: the sample takes
/// Metal's footprint as a slot's does (`PrecisionNote::CubeGradientFootprint`), the isotropic or
/// the anisotropic form by its table entry's word, through the set-1 sampler array of its own use,
/// and the plan takes an anisotropic listed sampler for it (`engine_heap_binder`). On a driver that
/// hands a compute pipeline's cube gradients to Metal's `gradientcube` (`CubeGradientRoute::Metal`)
/// the kernel hands Metal's own, exactly, through any sampler: no note. Every profile with the
/// sampled heap compiles it and passes L3; it was `ArgumentBufferTextures NotImplemented` before.
#[test]
fn a_cube_gradient_through_an_argument_buffer_sampler_reads_its_own_sampler_array() {
    let Some(bc) = assemble(
        "argument_sampler_cube_grad",
        r#"%struct.M = type { ptr addrspace(1), ptr addrspace(2) }
define void @argument_sampler_cube_grad(ptr addrspace(2) %m, ptr addrspace(1) %out, i32 %gid) {
  %pt = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %ps = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 1
  %s = load ptr addrspace(2), ptr addrspace(2) %ps, align 8
  %v = call { <4 x float>, i8 } @air.sample_texture_cube_grad.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <3 x float> <float 1.0, float 0.0, float 0.0>, <3 x float> <float 0.0, float 0x3F847AE140000000, float 0.0>, <3 x float> <float 0.0, float 0.0, float 0x3F847AE140000000>, float 0.0, i32 0)
  %x = extractvalue { <4 x float>, i8 } %v, 0
  %o = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %gid
  store <4 x float> %x, ptr addrspace(1) %o, align 16
  ret void
}
declare { <4 x float>, i8 } @air.sample_texture_cube_grad.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, <3 x float>, <3 x float>, float, i32)
!air.kernel = !{!0}
!air.version = !{!11}
!air.language_version = !{!12}
!0 = !{ptr @argument_sampler_cube_grad, !1, !2}
!1 = !{}
!2 = !{!3, !8, !9}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"M", !"air.arg_name", !"m"}
!4 = !{i32 0, i32 8, i32 0, !"texturecube<float, sample>", !"t", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"sampler", !"s", !"air.indirect_argument", !6}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texturecube<float, sample>", !"air.arg_name", !"t"}
!6 = !{i32 1, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!8 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!9 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!11 = !{i32 2, i32 8, i32 0}
!12 = !{!"Metal", i32 4, i32 0, i32 0}
"#,
    ) else {
        return;
    };
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = match compile(&bc, "argument_sampler_cube_grad", &p) {
            Ok(a) => a,
            Err(e) => {
                let u = unsupported(e);
                assert_eq!(u.feature, Feature::ArgumentBufferTextures, "{stem}");
                assert!(
                    matches!(u.needs, Needs::AnyOf(_)),
                    "{stem}: only the heap's device gap: {:?}",
                    u.needs
                );
                continue;
            }
        };
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let classes: Vec<HandleClass> = a
            .interface
            .descriptors
            .iter()
            .filter_map(|d| match d.slot.kind {
                ResourceKind::Heap(c @ HandleClass::Sampler { .. }) => Some(c),
                _ => None,
            })
            .collect();
        let metal =
            p.fields().cube_gradients.compute == crate::device::profile::CubeGradientRoute::Metal;
        assert_eq!(
            classes,
            vec![HandleClass::Sampler {
                compare: false,
                integer: false,
            }],
            "{stem}"
        );
        assert_eq!(
            a.precision.contains(&PrecisionNote::CubeGradientFootprint),
            !metal,
            "{stem}"
        );
        compiled += 1;
    }
    assert!(compiled > 0, "no committed profile has the sampled heap");
}

/// Several sampler candidates beside a heap texture are not built: a typed refusal.
#[test]
fn a_selected_sampler_beside_a_heap_texture_is_not_implemented() {
    let Some(bc) = assemble(
        "heap_selected_sampler",
        r#"%struct.T = type { ptr addrspace(1) }
%struct.R = type { %struct.T }
define void @heap_selected_sampler(ptr addrspace(2) %r, ptr addrspace(2) %s0, ptr addrspace(2) %s1, ptr addrspace(1) %out, i32 %gid) {
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %odd = and i32 %gid, 1
  %c = icmp eq i32 %odd, 0
  %s = select i1 %c, ptr addrspace(2) %s0, ptr addrspace(2) %s1
  %v = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> zeroinitializer, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %x = extractvalue { <4 x float>, i8 } %v, 0
  %o = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %gid
  store <4 x float> %x, ptr addrspace(1) %o, align 16
  ret void
}
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
!air.kernel = !{!0}
!air.version = !{!11}
!air.language_version = !{!12}
!0 = !{ptr @heap_selected_sampler, !1, !2}
!1 = !{}
!2 = !{!3, !6, !7, !8, !9}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"t", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!6 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s0"}
!7 = !{i32 2, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s1"}
!8 = !{i32 3, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!9 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!11 = !{i32 2, i32 8, i32 0}
!12 = !{!"Metal", i32 4, i32 0, i32 0}
"#,
    ) else {
        return;
    };
    let u = unsupported(compile(&bc, "heap_selected_sampler", &m3()).expect_err("refused"));
    assert_eq!(u.feature, Feature::ArgumentBufferTextures);
    assert_eq!(u.needs, Needs::NotImplemented);
}

/// A kernel that writes a slice of its explicit imageblock (one `elem` field per cell, `lanes`
/// lanes) to the texture of an argument buffer's one member (`texture2d<half, write>` or
/// `texture2d<short, write>`).
fn heap_slice_module(name: &str, elem: &str, ty: &str, lanes: u32, msl: &str) -> Option<Vec<u8>> {
    let suffix = if lanes == 1 {
        elem.to_string()
    } else {
        format!("v{lanes}{elem}")
    };
    let size = 2 * lanes;
    assemble(
        name,
        &format!(
            r#"%struct.T = type {{ ptr addrspace(1) }}
%struct.R = type {{ %struct.T }}
%"struct.metal::_imageblock_base" = type {{ ptr addrspace(4) }}
define void @heap_slice(ptr addrspace(2) %r, %"struct.metal::_imageblock_base" %img, <2 x i16> %lid) {{
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %base = call ptr addrspace(4) @air.imageblock_data(<2 x i16> zeroinitializer, i32 0, i16 0)
  call void @air.write_imageblock_slice_to_texture_2d.i16.{suffix}(ptr addrspace(1) %t, ptr addrspace(4) %base, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i16> zeroinitializer, i16 0, i1 false, i32 2)
  ret void
}}
declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.write_imageblock_slice_to_texture_2d.i16.{suffix}(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)
!air.kernel = !{{!0}}
!air.version = !{{!9}}
!air.language_version = !{{!10}}
!0 = !{{ptr @heap_slice, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !6, !8}}
!3 = !{{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}}
!4 = !{{i32 0, i32 8, i32 0, !"texture2d<{msl}, write>", !"u", !"air.indirect_argument", !5}}
!5 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<{msl}, write>", !"air.arg_name", !"u"}}
!6 = !{{i32 1, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 8, !"air.struct_type_info", !7, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"imageblock<Plane, layout_explicit>", !"air.arg_name", !"img"}}
!7 = !{{i32 0, i32 {size}, i32 0, !"{ty}", !"c"}}
!8 = !{{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"lid"}}
!9 = !{{i32 2, i32 8, i32 0}}
!10 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
        ),
    )
}

/// An imageblock slice written to an argument buffer's texture writes the heap element as a heap
/// `write` does: one heap lookup, the slice's loop, the float 2D storage class. Its element may be
/// narrower than the texel (Metal writes the element's lanes and zeros in every format, so the
/// member's format needs no check). A profile without the storage heap's features refuses it naming
/// them. An integer element no other operation types stays `Imageblocks NotImplemented` (its class
/// has no source).
#[test]
fn an_imageblock_slice_writes_an_argument_buffer_texture() {
    let mut compiled = 0;
    for (name, ty, lanes) in [
        ("heap_slice", "half4", 4),
        ("heap_slice_two_lanes", "half2", 2),
        ("heap_slice_one_lane", "half", 1),
    ] {
        let Some(bc) = heap_slice_module(name, "f16", ty, lanes, "half") else {
            return;
        };
        for (stem, p) in committed_profiles().expect("profiles") {
            let a = match compile(&bc, "heap_slice", &p) {
                Ok(a) => a,
                Err(e) => {
                    assert!(!has_storage_heap(&p), "{name} on {stem}: {e}");
                    let u = unsupported(e);
                    assert_eq!(
                        u.feature,
                        Feature::ArgumentBufferTextures,
                        "{name} on {stem}"
                    );
                    assert!(matches!(u.needs, Needs::AnyOf(_)), "{stem}: {:?}", u.needs);
                    continue;
                }
            };
            assert!(has_storage_heap(&p), "{name} on {stem}");
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name} on {stem}: L3: {e}"));
            assert_eq!(uses(&a, "access.handle.lookup"), 1, "{name} on {stem}");
            assert_eq!(
                uses(&a, "texture.imageblock_slice") + uses(&a, "texture.imageblock_slice_guarded"),
                1,
                "{name} on {stem}"
            );
            // The written float storage class alone.
            let classes = heap_image_classes(&a);
            assert_eq!(classes.len(), 1, "{name} on {stem}: {classes:?}");
            let c = classes[0];
            assert_eq!(
                (c.dim, c.arrayed, c.sampled_type, c.access),
                (
                    ImageDim::D2,
                    false,
                    SampledType::Float,
                    ImageAccess::Storage
                ),
                "{name} on {stem}"
            );
            assert!(texture_slots(&a).is_empty(), "{name} on {stem}");
            compiled += 1;
        }
    }
    assert!(compiled > 0, "no committed profile has the storage heap");
    let Some(bc) = heap_slice_module("heap_slice_integer", "i16", "short4", 4, "short") else {
        return;
    };
    let u = unsupported(compile(&bc, "heap_slice", &m3()).expect_err("heap_slice_integer"));
    assert_eq!(u.feature, Feature::Imageblocks);
    assert_eq!(u.needs, Needs::NotImplemented);
}

const OP_IMAGE_SAMPLE_DREF_EXPLICIT_LOD: u16 = 90;
const OP_IMAGE_GATHER: u16 = 96;
const OP_IMAGE_DREF_GATHER: u16 = 97;

/// Whether `p`'s device samples through Metal's own sampler, whose comparison is Metal's.
fn metal_sampler(p: &DeviceProfile) -> bool {
    p.fields()
        .enabled
        .properties
        .contains(&crate::device::requirements::VkProperty::MetalSampler)
}

/// What [`heap_compare_kernel`] compares through, and [`heap_linear_kernel`] samples through: a
/// constexpr sampler of these words, runtime sampler slot 0, or the sampler its argument buffer
/// holds beside the texture.
#[derive(Clone, Copy, Debug)]
enum CompareThrough {
    Constexpr(u64),
    Runtime,
    Argument,
}

/// A constexpr sampler's words: nearest filters, no mip filter, normalized coordinates clamped to
/// the edge, comparing under `less` (the `handles` fixture's `shadow_less` with nearest filters).
const NEAREST_LESS: u64 = 0x007b_ff00_0001_0049;

/// A kernel comparing the `depth2d<float>` its argument buffer (buffer 0) holds through `through`
/// at `(0.25, 0.25)` against 0.5: a `sample_compare` at level 0, or where `gather` a
/// `gather_compare`, written to buffer 1's lane `gid` (a `float4`).
fn heap_compare_kernel(through: CompareThrough, gather: bool) -> String {
    let argument = matches!(through, CompareThrough::Argument);
    let (global, states) = match through {
        CompareThrough::Constexpr(w) => (
            format!(
                "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {w}, i64 0], align 8\n"
            ),
            "!air.sampler_states = !{!10}\n!10 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
        ),
        _ => (String::new(), ""),
    };
    let (param, operand, args, gid) = match through {
        CompareThrough::Constexpr(_) => ("", "@__air_sampler_state", "!3, !6, !8", 2),
        CompareThrough::Runtime => (", ptr addrspace(2) %rs", "%rs", "!3, !6, !7, !8", 3),
        CompareThrough::Argument => ("", "%s", "!3, !6, !8", 2),
    };
    let (ty, size, member, load) = match argument {
        true => (
            "{ ptr addrspace(1), ptr addrspace(2) }",
            16,
            r#", i32 8, i32 8, i32 0, !"sampler", !"s", !"air.indirect_argument", !9"#,
            "  %ps = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 1\n  %s = load ptr addrspace(2), ptr addrspace(2) %ps, align 8\n",
        ),
        false => ("{ ptr addrspace(1) }", 8, "", ""),
    };
    let (call, declare) = match gather {
        true => (
            format!(
                "  %r = call {{ <4 x float>, i8 }} @air.gather_compare_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) {operand}, i32 1, <2 x float> <float 0.25, float 0.25>, float 0.5, i1 true, <2 x i32> zeroinitializer, i32 0)\n  %v = extractvalue {{ <4 x float>, i8 }} %r, 0\n"
            ),
            "declare { <4 x float>, i8 } @air.gather_compare_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, float, i1, <2 x i32>, i32)",
        ),
        false => (
            format!(
                "  %r = call {{ float, i8 }} @air.sample_compare_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) {operand}, i32 1, <2 x float> <float 0.25, float 0.25>, float 0.5, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)\n  %x = extractvalue {{ float, i8 }} %r, 0\n  %v = insertelement <4 x float> zeroinitializer, float %x, i64 0\n"
            ),
            "declare { float, i8 } @air.sample_compare_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, float, i1, <2 x i32>, i1, float, float, i32)",
        ),
    };
    format!(
        r#"%struct.M = type {ty}
{global}
define void @k(ptr addrspace(2) %m, ptr addrspace(1) %out{param}, i32 %gid) {{
  %pt = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
{load}{call}  %o = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %gid
  store <4 x float> %v, ptr addrspace(1) %o, align 16
  ret void
}}
{declare}
!air.kernel = !{{!0}}
!air.version = !{{!11}}
!air.language_version = !{{!12}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{{args}}}
!3 = !{{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 {size}, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"M", !"air.arg_name", !"m"}}
!4 = !{{i32 0, i32 8, i32 0, !"depth2d<float, sample>", !"d", !"air.indirect_argument", !5{member}}}
!5 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"depth2d<float, sample>", !"air.arg_name", !"d"}}
!6 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!7 = !{{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"rs"}}
!8 = !{{i32 {gid}, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!9 = !{{i32 1, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
!11 = !{{i32 2, i32 8, i32 0}}
!12 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// A heap depth comparison (a `sample_compare` or a `gather_compare` through a constexpr sampler,
/// a runtime sampler or the sampler its argument buffer holds) is the device's: one `Dref` access,
/// no plain one and no fetch, through one comparison sampler and the depth-comparison class of its
/// shape. Each validates; a profile without the sampled heap refuses each. (The interpreter models
/// no sample; the runtime lane's `depth_compares` runs these comparisons through argument buffers
/// on each device.)
#[test]
fn heap_depth_comparisons_are_the_device_s() {
    let cases = [
        (CompareThrough::Constexpr(NEAREST_LESS), false),
        (CompareThrough::Constexpr(NEAREST_LESS), true),
        (CompareThrough::Runtime, false),
        (CompareThrough::Runtime, true),
        (CompareThrough::Argument, true),
    ];
    for (through, gather) in cases {
        let name = format!("{through:?} gather {gather}");
        let Some(bc) = assemble("heap_compare", &heap_compare_kernel(through, gather)) else {
            return;
        };
        let mut compiled = 0;
        for (stem, p) in committed_profiles().expect("profiles") {
            let what = format!("{name} on {stem}");
            let a = match compile(&bc, "k", &p) {
                Ok(a) => a,
                Err(e) => {
                    assert!(!has_sampled_heap(&p), "{what}: {e}");
                    let u = unsupported(e);
                    assert_eq!(u.feature, Feature::ArgumentBufferTextures, "{what}");
                    continue;
                }
            };
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (dref, plain) = match gather {
                true => (OP_IMAGE_DREF_GATHER, OP_IMAGE_GATHER),
                false => (
                    OP_IMAGE_SAMPLE_DREF_EXPLICIT_LOD,
                    OP_IMAGE_SAMPLE_EXPLICIT_LOD,
                ),
            };
            assert_eq!(count(&a, dref), 1, "{what}");
            assert_eq!(count(&a, plain), 0, "{what}");
            assert_eq!(count(&a, OP_IMAGE_FETCH), 0, "{what}");
            assert_eq!(count(&a, OP_SAMPLED_IMAGE), 1, "{what}");
            let compares: Vec<bool> = match through {
                CompareThrough::Argument => sampler_arrays(&a).iter().map(|u| u.0).collect(),
                _ => heap_samplers(&a).iter().map(|u| u.compare).collect(),
            };
            assert_eq!(compares, [true], "{what}");
            let classes: Vec<bool> = heap_image_classes(&a)
                .iter()
                .map(|c| c.depth_compare)
                .collect();
            assert_eq!(classes, [true], "{what}");
            compiled += 1;
        }
        assert!(
            compiled > 0,
            "{name}: no committed profile has the sampled heap"
        );
    }
}

/// A heap depth texture sampled with comparison is a member of the depth-comparison class of its
/// shape (float 2D, declared `Depth`), sampled through a comparison heap sampler, the constexpr
/// one, with one `OpImageSampleDrefExplicitLod`; a profile without the sampled heap's features
/// refuses it naming them.
#[test]
fn an_argument_buffer_depth_texture_compares_through_the_depth_comparison_class() {
    let Some(bc) = fixture() else { return };
    let entry = "argument_buffer_sample_compare";
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = match compile(&bc, entry, &p) {
            Ok(a) => a,
            Err(e) => {
                assert!(!has_sampled_heap(&p), "{stem}: {e}");
                let u = unsupported(e);
                assert_eq!(u.feature, Feature::ArgumentBufferTextures, "{stem}");
                assert!(matches!(u.needs, Needs::AnyOf(_)), "{stem}: {:?}", u.needs);
                continue;
            }
        };
        assert!(has_sampled_heap(&p), "{stem}");
        let (t, _) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert_eq!(uses(&a, "access.handle.lookup"), 1, "{stem}");
        assert_eq!(uses(&a, "sampler.heap.sample"), 1, "{stem}");
        let classes = heap_image_classes(&a);
        assert_eq!(classes.len(), 1, "{stem}: {classes:?}");
        let c = classes[0];
        assert!(c.depth_compare, "{stem}: {c:?}");
        assert_eq!(
            (c.dim, c.arrayed, c.multisampled, c.sampled_type, c.access),
            (
                ImageDim::D2,
                false,
                false,
                SampledType::Float,
                ImageAccess::Sampled
            ),
            "{stem}"
        );
        assert!(heap_classes(&p).contains(&HandleClass::Image(c)), "{stem}");
        // The one image type is 2D (Dim 1) and a depth image.
        assert_eq!(
            image_types(&a)
                .into_iter()
                .map(|(dim, depth, _)| (dim, depth))
                .collect::<Vec<_>>(),
            vec![(1, 1)],
            "{stem}"
        );
        let hs = heap_samplers(&a);
        assert_eq!(hs.len(), 1, "{stem}");
        assert!(hs[0].compare, "{stem}");
        assert!(matches!(hs[0].sampler, SamplerRef::Constexpr(_)), "{stem}");
        assert_eq!(bindings_of(&t, is_heap).len(), 1, "{stem}");
        assert_eq!(count(&a, OP_IMAGE_SAMPLE_DREF_EXPLICIT_LOD), 1, "{stem}");
        compiled += 1;
    }
    assert!(compiled > 0, "no committed profile has the sampled heap");
}

const OP_SAMPLED_IMAGE: u16 = 86;
const OP_IMAGE_FETCH: u16 = 95;

/// A constexpr sampler's words: linear filters, no mip filter, normalized coordinates clamped to
/// the edge.
const LINEAR_EDGE: u64 = 0x007b_ff00_0008_0a49;

/// A kernel sampling the `texture2d<float>` (a `texture3d<float>` where `three`) its argument
/// buffer (buffer 0) holds through `through` at level 0 and a coordinate between texels, written
/// to buffer 1's lane `gid` (a `float4`).
fn heap_linear_kernel(through: CompareThrough, three: bool) -> String {
    let argument = matches!(through, CompareThrough::Argument);
    let (global, states) = match through {
        CompareThrough::Constexpr(w) => (
            format!(
                "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {w}, i64 0], align 8\n"
            ),
            "!air.sampler_states = !{!10}\n!10 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
        ),
        _ => (String::new(), ""),
    };
    let (param, operand, args, gid) = match through {
        CompareThrough::Constexpr(_) => ("", "@__air_sampler_state", "!3, !6, !8", 2),
        CompareThrough::Runtime => (", ptr addrspace(2) %rs", "%rs", "!3, !6, !7, !8", 3),
        CompareThrough::Argument => ("", "%s", "!3, !6, !8", 2),
    };
    let (ty, size, member, load) = match argument {
        true => (
            "{ ptr addrspace(1), ptr addrspace(2) }",
            16,
            r#", i32 8, i32 8, i32 0, !"sampler", !"s", !"air.indirect_argument", !9"#,
            "  %ps = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 1\n  %s = load ptr addrspace(2), ptr addrspace(2) %ps, align 8\n",
        ),
        false => ("{ ptr addrspace(1) }", 8, "", ""),
    };
    let (texture, call, declare) = match three {
        true => (
            "texture3d<float, sample>",
            format!(
                "@air.sample_texture_3d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {operand}, <3 x float> <float 0x3FD3333340000000, float 0x3FE3333340000000, float 0x3FDCCCCCC0000000>, i1 true, <3 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)"
            ),
            "declare { <4 x float>, i8 } @air.sample_texture_3d.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, <3 x i32>, i1, float, float, i32)",
        ),
        false => (
            "texture2d<float, sample>",
            format!(
                "@air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {operand}, <2 x float> <float 0x3FD3333340000000, float 0x3FE3333340000000>, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)"
            ),
            "declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)",
        ),
    };
    format!(
        r#"%struct.M = type {ty}
{global}
define void @k(ptr addrspace(2) %m, ptr addrspace(1) %out{param}, i32 %gid) {{
  %pt = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
{load}  %r = call {{ <4 x float>, i8 }} {call}
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  %o = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %gid
  store <4 x float> %v, ptr addrspace(1) %o, align 16
  ret void
}}
{declare}
!air.kernel = !{{!0}}
!air.version = !{{!11}}
!air.language_version = !{{!12}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{{args}}}
!3 = !{{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 {size}, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"M", !"air.arg_name", !"m"}}
!4 = !{{i32 0, i32 8, i32 0, !"{texture}", !"t", !"air.indirect_argument", !5{member}}}
!5 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{texture}", !"air.arg_name", !"t"}}
!6 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!7 = !{{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"rs"}}
!8 = !{{i32 {gid}, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!9 = !{{i32 1, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
!11 = !{{i32 2, i32 8, i32 0}}
!12 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// A heap texture's linear sample, through a constexpr sampler of linear filters (2D and 3D), a
/// runtime sampler slot, or the sampler its argument buffer holds, is the device's sample; the
/// census names the sample by its source, and the module states no `TexelFormat`
/// specialization. On a profile that samples through Metal's own sampler a 2D sample records the
/// filter note only under what the command's handles can name (a two-plane YCbCr texture,
/// `SpecKey::HeapYCbCrTextures`), never outright; a 3D one, whose class holds no two-plane
/// texture, records none. A 2D sample through a constexpr or runtime sampler also emulates a raw-held
/// texture's, in a branch on the class the handle carries (the search's flagged form, whose masks
/// and table the census counts under `HEAP_CLASS_CENSUS`). Each validates; a profile without the
/// sampled heap refuses each. (The interpreter models no sample; the runtime lane
/// `argument_linear_samples` reads the values on each device.)
#[test]
fn heap_linear_samples_are_the_device_s() {
    use crate::lower::texture::HEAP_CLASS_CENSUS;
    use crate::memory::handles::HANDLE_TEXEL_CLASS;
    use crate::pipeline::sym::SpecKey;
    let cases = [
        (CompareThrough::Constexpr(LINEAR_EDGE), false),
        (CompareThrough::Constexpr(LINEAR_EDGE), true),
        (CompareThrough::Runtime, false),
        (CompareThrough::Argument, false),
        (CompareThrough::Argument, true),
    ];
    for (through, three) in cases {
        let name = format!("{through:?} 3D {three}");
        let Some(bc) = assemble("heap_linear", &heap_linear_kernel(through, three)) else {
            return;
        };
        let mut compiled = 0;
        for (stem, p) in committed_profiles().expect("profiles") {
            let what = format!("{name} on {stem}");
            let a = match compile(&bc, "k", &p) {
                Ok(a) => a,
                Err(e) => {
                    assert!(!has_sampled_heap(&p), "{what}: {e}");
                    let u = unsupported(e);
                    assert_eq!(u.feature, Feature::ArgumentBufferTextures, "{what}");
                    continue;
                }
            };
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_eq!(count(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD), 1, "{what}");
            // An argument buffer's sampler's sample reads its texel's centre where its word states
            // a nearest pixel sampler: no fetch.
            let argument = matches!(through, CompareThrough::Argument);
            // The raw-held emulation: four filtered taps and a nearest one from the twin.
            let raw = !three && !argument;
            assert_eq!(count(&a, OP_IMAGE_FETCH), 5 * usize::from(raw), "{what}");
            assert_eq!(count(&a, OP_SAMPLED_IMAGE), 1, "{what}");
            assert_eq!(
                a.strategies.entries.contains_key(HEAP_CLASS_CENSUS),
                raw,
                "{what}"
            );
            let census = match argument {
                true => "sampler.argument_sampler.sample",
                false => "sampler.heap.sample",
            };
            assert_eq!(uses(&a, census), 1, "{what}");
            assert_eq!(
                mask_constants(&a).contains(&!HANDLE_TEXEL_CLASS),
                raw,
                "{what}"
            );
            let filter = PrecisionNote::FloatSampleFilter;
            assert_eq!(
                a.precision.contains(&filter),
                !three && metal_sampler(&p),
                "{what}"
            );
            let clauses: Vec<_> = (a.specialized.iter())
                .filter(|s| s.note == filter)
                .flat_map(|s| &s.when)
                .collect();
            assert_eq!(clauses.is_empty(), three || !metal_sampler(&p), "{what}");
            for c in clauses {
                let heap = SpecKey::HeapYCbCrTextures(crate::pipeline::sym::Stage::Kernel);
                assert!(c.all.iter().any(|t| t.key == heap), "{what}: {c:?}");
            }
            assert!(
                !a.interface
                    .read_keys()
                    .into_iter()
                    .any(|k| matches!(k, SpecKey::TexelFormat(_))),
                "{what}"
            );
            compiled += 1;
        }
        assert!(
            compiled > 0,
            "{name}: no committed profile has the sampled heap"
        );
    }
}

/// A constexpr sampler's words: nearest filters, no mip filter, normalized coordinates clamped to
/// the edge.
const NEAREST_EDGE: u64 = 0x007b_ff00_0008_0000 | 1 | 1 << 3 | 1 << 6;

/// [`heap_linear_kernel`]'s 2D kernel, its sample a gather of component 0 where `gather`.
fn heap_plain_kernel(through: CompareThrough, gather: bool) -> String {
    let text = heap_linear_kernel(through, false);
    if !gather {
        return text;
    }
    let mut out = text;
    for (from, to) in [
        (
            "@air.sample_texture_2d.v4f32(",
            "@air.gather_texture_2d.v4f32(",
        ),
        (
            "<2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)",
            "<2 x i32> zeroinitializer, i32 0, i32 0)",
        ),
        (
            "<2 x float>, i1, <2 x i32>, i1, float, float, i32)",
            "<2 x float>, i1, <2 x i32>, i32, i32)",
        ),
    ] {
        assert!(out.contains(from), "no {from}");
        out = out.replace(from, to);
    }
    out
}

/// [`heap_plain_kernel`] with a `half` result (`texture2d<half>`, a `half4` buffer).
fn heap_plain_half_kernel(through: CompareThrough, gather: bool) -> String {
    let mut out = heap_plain_kernel(through, gather);
    for (from, to) in [
        (".v4f32(", ".v4f16("),
        ("{ <4 x float>, i8 }", "{ <4 x half>, i8 }"),
        ("<4 x float>", "<4 x half>"),
        ("texture2d<float, sample>", "texture2d<half, sample>"),
        (
            r#"i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4""#,
            r#"i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4""#,
        ),
        ("align 16", "align 8"),
    ] {
        assert!(out.contains(from), "no {from}");
        out = out.replace(from, to);
    }
    out
}

/// A heap color texture's sample or gather with a `half` result converts the device's texel (or the
/// fetched one) by the class its handle carries, as a slot's converts by its bound format's, on a
/// profile that samples through Metal's own sampler (`VkProperty::MetalSampler`, whose `float`
/// result the class's rule rounds as Metal does): the module calls the `half` routine once, with
/// the class read from the handle's table, where the same access with a `float` result calls no
/// routine. On any other profile the `half` texel is the device's conversion: no routine. The note
/// stays: the class may be one whose conversion is not measured.
#[test]
fn heap_half_texels_convert_by_the_handle_s_class() {
    let cases = [
        (CompareThrough::Constexpr(NEAREST_EDGE), false),
        (CompareThrough::Constexpr(NEAREST_EDGE), true),
        (CompareThrough::Runtime, false),
        (CompareThrough::Argument, false),
    ];
    for (through, gather) in cases {
        let name = format!("{through:?} gather {gather}");
        let Some(bc) = assemble("heap_half", &heap_plain_half_kernel(through, gather)) else {
            return;
        };
        let float = assemble("heap_float", &heap_plain_kernel(through, gather)).expect("float");
        let mut compiled = 0;
        for (stem, p) in committed_profiles().expect("profiles") {
            let what = format!("{name} on {stem}");
            let a = match compile(&bc, "k", &p) {
                Ok(a) => a,
                Err(e) => {
                    assert!(!has_sampled_heap(&p), "{what}: {e}");
                    continue;
                }
            };
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let f = compile(&float, "k", &p).unwrap_or_else(|e| panic!("{what}: float: {e}"));
            // A sample's raw-held emulation through a constexpr or runtime sampler decodes its
            // nearest tap both ways for a `half` result (a bfloat form's toward zero, another's
            // rounded): one call more than the `float` result's.
            let twin = usize::from(!gather && !matches!(through, CompareThrough::Argument));
            let routine = usize::from(metal_sampler(&p));
            assert_eq!(
                count(&a, OP_FUNCTION_CALL),
                count(&f, OP_FUNCTION_CALL) + routine + twin,
                "{what}: the half routine is called once, the float one where it converts"
            );
            let notes = &a.precision;
            assert!(
                notes.contains(&PrecisionNote::HalfTexelConversion),
                "{what}: {notes:?}"
            );
            compiled += 1;
        }
        assert!(compiled > 0, "{name}: no profile compiled it");
    }
}

/// A heap texture's plain gather, and its nearest sample through a constexpr sampler, a runtime
/// sampler or an argument buffer's sampler, are the device's access, made once, with no fetch of
/// their own (an argument buffer's sampler's sample reads its texel's centre where its word states
/// a nearest pixel sampler). Through a constexpr or runtime sampler the raw-held
/// emulation fetches from the twin in a branch on the class the handle carries. The module states
/// no `TexelFormat` specialization.
#[test]
fn heap_gathers_and_nearest_samples_are_the_device_s() {
    use crate::lower::texture::HEAP_CLASS_CENSUS;
    use crate::memory::handles::HANDLE_TEXEL_CLASS;
    use crate::pipeline::sym::SpecKey;
    let cases = [
        (CompareThrough::Constexpr(NEAREST_EDGE), false),
        (CompareThrough::Constexpr(NEAREST_EDGE), true),
        (CompareThrough::Runtime, false),
        (CompareThrough::Runtime, true),
        (CompareThrough::Argument, false),
        (CompareThrough::Argument, true),
    ];
    for (through, gather) in cases {
        let name = format!("{through:?} gather {gather}");
        let Some(bc) = assemble("heap_plain", &heap_plain_kernel(through, gather)) else {
            return;
        };
        let mut compiled = 0;
        for (stem, p) in committed_profiles().expect("profiles") {
            let what = format!("{name} on {stem}");
            let a = match compile(&bc, "k", &p) {
                Ok(a) => a,
                Err(e) => {
                    assert!(!has_sampled_heap(&p), "{what}: {e}");
                    let u = unsupported(e);
                    assert_eq!(u.feature, Feature::ArgumentBufferTextures, "{what}");
                    continue;
                }
            };
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let word = !matches!(through, CompareThrough::Constexpr(_));
            let argument = matches!(through, CompareThrough::Argument);
            let device = if gather {
                OP_IMAGE_GATHER
            } else {
                OP_IMAGE_SAMPLE_EXPLICIT_LOD
            };
            assert_eq!(count(&a, device), 1, "{what}");
            // Through a constexpr or runtime sampler, the raw-held emulation's fetches from the
            // twin: a gather's four, a nearest sample's one, a runtime sampler's four filtered and
            // one nearest.
            let raw = !argument;
            let twin = match (raw, gather, word) {
                (false, _, _) => 0,
                (true, true, _) => 4,
                (true, false, false) => 1,
                (true, false, true) => 5,
            };
            assert_eq!(count(&a, OP_IMAGE_FETCH), twin, "{what}");
            assert_eq!(count(&a, OP_SAMPLED_IMAGE), 1, "{what}");
            assert_eq!(
                a.strategies.entries.contains_key(HEAP_CLASS_CENSUS),
                raw,
                "{what}"
            );
            assert_eq!(
                mask_constants(&a).contains(&!HANDLE_TEXEL_CLASS),
                raw,
                "{what}"
            );
            assert!(
                !a.interface
                    .read_keys()
                    .into_iter()
                    .any(|k| matches!(k, SpecKey::TexelFormat(_))),
                "{what}"
            );
            compiled += 1;
        }
        assert!(
            compiled > 0,
            "{name}: no committed profile has the sampled heap"
        );
    }
}

/// The heap classes a module declares.
fn heap_image_classes(a: &FunctionArtifact) -> Vec<ImageHandleClass> {
    a.interface
        .descriptors
        .iter()
        .filter_map(|d| match d.slot.kind {
            ResourceKind::Heap(HandleClass::Image(c)) => Some(c),
            _ => None,
        })
        .collect()
}

/// The heap samplers a module declares.
fn heap_samplers(a: &FunctionArtifact) -> Vec<CombinedSampler> {
    a.interface
        .descriptors
        .iter()
        .filter_map(|d| d.slot.heap_sampler_use())
        .collect()
}

/// A heap texture sampled through a constexpr sampler is `OpSampledImage` of its heap element and
/// a separate heap sampler: one set-0 `SAMPLER` binding per distinct sampler use, whose immutable
/// sampler is the constexpr one, beside the set-1 arrays. A kernel samples through a linear and a
/// pixel sampler (an exact fetch) and an integer texture through a nearest one; a fragment samples
/// a cube, a 2D, and a depth member and an element of a texture-array member at its implicit level,
/// four lookups sharing one sampler over a cube and a 2D class. A profile without the heap's
/// features refuses each naming them.
#[test]
fn argument_buffer_textures_sample_through_heap_samplers() {
    let Some(bc) = fixture() else { return };
    // Entry, heaps it needs, strategies, heap classes `(dim, sampled type)` sampled, heap
    // samplers `(integer)`.
    type SampleCase = (
        &'static str,
        fn(&DeviceProfile) -> bool,
        &'static [(&'static str, u32)],
        &'static [(ImageDim, SampledType)],
        &'static [bool],
    );
    fn both(p: &DeviceProfile) -> bool {
        has_sampled_heap(p) && has_storage_heap(p)
    }
    let cases: [SampleCase; 3] = [
        (
            "argument_buffer_sample",
            both,
            &[
                ("access.handle.lookup", 2),
                ("sampler.heap.sample", 1),
                ("sampler.heap.pixel_fetch", 1),
                ("texture.write.heap", 1),
            ],
            // The unsigned twin the raw-held emulation fetches from.
            &[
                (ImageDim::D2, SampledType::Float),
                (ImageDim::D2, SampledType::Uint),
            ],
            &[false, false],
        ),
        (
            "argument_buffer_sample_int",
            both,
            &[
                ("access.handle.lookup", 2),
                ("sampler.heap.sample", 1),
                ("texture.write.heap", 1),
            ],
            &[(ImageDim::D2, SampledType::Sint)],
            &[true],
        ),
        (
            "argument_buffer_fragment_sample",
            has_sampled_heap,
            &[("access.handle.lookup", 4), ("sampler.heap.sample", 4)],
            &[
                (ImageDim::D2, SampledType::Float),
                (ImageDim::D2, SampledType::Uint),
                (ImageDim::Cube, SampledType::Float),
            ],
            &[false],
        ),
    ];
    for (entry, has_heap, strategies, classes, samplers) in cases {
        let mut compiled = 0;
        for (stem, p) in committed_profiles().expect("profiles") {
            let a = match compile(&bc, entry, &p) {
                Ok(a) => a,
                Err(e) => {
                    assert!(!has_heap(&p), "{stem} {entry}: {e}");
                    let u = unsupported(e);
                    assert_eq!(u.feature, Feature::ArgumentBufferTextures, "{stem} {entry}");
                    assert!(matches!(u.needs, Needs::AnyOf(_)), "{stem}: {:?}", u.needs);
                    continue;
                }
            };
            assert!(has_heap(&p), "{stem} {entry}");
            let (t, _) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {entry}: L3: {e}"));
            for (name, n) in strategies {
                assert_eq!(uses(&a, name), *n, "{stem} {entry} {name}");
            }
            assert!(texture_slots(&a).is_empty(), "{stem} {entry}");
            let mut sampled: Vec<_> = heap_image_classes(&a)
                .into_iter()
                .filter(|c| c.access == ImageAccess::Sampled)
                .map(|c| (c.dim, c.sampled_type))
                .collect();
            sampled.sort();
            assert_eq!(sampled, classes, "{stem} {entry}");
            for c in heap_image_classes(&a) {
                assert!(
                    heap_classes(&p).contains(&HandleClass::Image(c)),
                    "{stem} {entry}"
                );
            }
            let hs = heap_samplers(&a);
            let integer: Vec<bool> = hs.iter().map(|c| c.integer).collect();
            assert_eq!(integer, samplers, "{stem} {entry}");
            for c in &hs {
                assert!(matches!(c.sampler, SamplerRef::Constexpr(_)), "{stem}");
                assert!(!c.compare, "{stem}");
            }
            // Each heap sampler is a set-0 binding; the sampled image of each sample is formed
            // at its use.
            let bound = bindings_of(&t, |k| matches!(k, ResourceKind::HeapSampler(_)));
            assert_eq!(bound.len(), samplers.len(), "{stem} {entry}");
            assert!(bound.iter().all(|(set, _)| *set == 0), "{stem} {entry}");
            let samples: u32 = strategies
                .iter()
                .filter(|(n, _)| n.starts_with("sampler.heap."))
                .map(|(_, n)| n)
                .sum();
            assert_eq!(
                count(&a, OP_SAMPLED_IMAGE) as u32,
                samples,
                "{stem} {entry}"
            );
            compiled += 1;
        }
        assert!(compiled > 0, "{entry}: no committed profile has its heap");
    }
    // The pixel sampler's sample is an exact fetch of the heap element; beside them the raw-held
    // emulation's fetches from the twin: the linear sample's four filtered taps and nearest one,
    // the pixel sampler's one.
    let a = ok(&bc, "argument_buffer_sample", &m3());
    assert_eq!(count(&a, OP_IMAGE_FETCH), 1 + 5 + 1);
    // The fragment's samples are at the implicit level, not inside a switch.
    let a = ok(&bc, "argument_buffer_fragment_sample", &m3());
    assert_eq!(count(&a, OP_IMAGE_SAMPLE_IMPLICIT_LOD), 4);
    assert_eq!(count(&a, OP_SWITCH), 0);
}

/// A texture handle used as a byte pointer (a GEP and a load through it) conflicts.
#[test]
fn a_handle_used_as_bytes_is_handle_as_pointer() {
    let Some(bc) = assemble(
        "handle_bytes",
        r#"define void @handle_bytes(ptr addrspace(1) %t, ptr addrspace(1) %out) {
  %p = getelementptr i8, ptr addrspace(1) %t, i64 4
  %v = load i32, ptr addrspace(1) %p, align 4
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 0)
  %s = add i32 %v, %w
  store i32 %s, ptr addrspace(1) %out, align 4
  ret void
}
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
!air.kernel = !{!0}
!air.version = !{!5}
!air.language_version = !{!6}
!0 = !{ptr @handle_bytes, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, i32 8, i32 0}
!6 = !{!"Metal", i32 4, i32 0, i32 0}
"#,
    ) else {
        return;
    };
    let u = unsupported(compile(&bc, "handle_bytes", &m3()).expect_err("refused"));
    assert_eq!(u.feature, Feature::HandleAsPointer);
    assert_eq!(u.needs, Needs::NotRepresentable);
}

/// A kernel encoding one compute command into the indirect command buffer of argument buffer
/// `enc` (`{command_buffer, compute_pipeline_state}`), set to the member pipeline state, then
/// storing `%v` to `out[t]`: `value` defines `%v`.
fn icb_pipeline_kernel(name: &str, value: &str) -> String {
    format!(
        r#"%enc = type {{ ptr addrspace(1), ptr addrspace(1) }}

define void @{name}(ptr addrspace(1) %enc, ptr addrspace(1) %out, i32 %t) {{
  %pc = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 0
  %cmd = load ptr addrspace(1), ptr addrspace(1) %pc, align 8
  %pp = getelementptr inbounds %enc, ptr addrspace(1) %enc, i64 0, i32 1
  %pso = load ptr addrspace(1), ptr addrspace(1) %pp, align 8
  call void @air.set_pipeline_state_compute_command(ptr addrspace(1) %cmd, i32 %t, ptr addrspace(1) %pso)
  call void @air.concurrent_dispatch_threadgroups_compute_command(ptr addrspace(1) %cmd, i32 %t, <3 x i32> <i32 1, i32 1, i32 1>, <3 x i32> <i32 32, i32 1, i32 1>)
{value}
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}}

declare void @air.set_pipeline_state_compute_command(ptr addrspace(1), i32, ptr addrspace(1))
declare void @air.concurrent_dispatch_threadgroups_compute_command(ptr addrspace(1), i32, <3 x i32>, <3 x i32>)

!air.kernel = !{{!0}}
!air.version = !{{!9}}
!air.language_version = !{{!10}}
!0 = !{{ptr @{name}, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !7, !8}}
!3 = !{{i32 0, !"air.indirect_buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Encoder", !"air.arg_name", !"enc"}}
!4 = !{{i32 0, i32 8, i32 0, !"command_buffer", !"commands", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"compute_pipeline_state", !"pipeline", !"air.indirect_argument", !6}}
!5 = !{{i32 0, !"air.command_buffer", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"command_buffer", !"air.arg_name", !"commands"}}
!6 = !{{i32 1, !"air.compute_pipeline_state", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"compute_pipeline_state", !"air.arg_name", !"pipeline"}}
!7 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!8 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}}
!9 = !{{i32 2, i32 8, i32 0}}
!10 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// `air.set_pipeline_state_compute_command`'s pipeline operand is a `compute_pipeline_state`
/// handle, the class of the argument-buffer member it is loaded from, so a GPU-encoded compute
/// command is no class conflict: with device addresses it translates (the member words are the
/// record buffer's guest address and the template's token), and without them it is its family's
/// refusal, `GpuEncodedIndirectCommands`, naming what the profile lacks. The same pipeline state
/// also read as bytes still conflicts: `HandleAsPointer`, not representable.
#[test]
fn a_command_set_to_an_argument_buffer_pipeline_state_is_its_family_refusal() {
    let Some(bc) = assemble(
        "icb_pipeline",
        &icb_pipeline_kernel("set_pipeline", "  %v = add i32 %t, 1"),
    ) else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        match compile(&bc, "set_pipeline", &p) {
            Ok(a) => {
                assert_eq!(
                    p.fields().buffer_access,
                    crate::device::profile::BufferAccess::Addressed,
                    "{stem}"
                );
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
            }
            Err(e) => {
                let u = unsupported(e);
                assert_eq!(
                    (u.feature, u.needs),
                    (
                        Feature::GpuEncodedIndirectCommands,
                        Unsupported::needs_any_of(
                            Feature::GpuEncodedIndirectCommands,
                            vec![p.addressed_needs()]
                        )
                        .needs
                    ),
                    "{stem}"
                );
            }
        }
    }
    let Some(bc) = assemble(
        "icb_pipeline_bytes",
        &icb_pipeline_kernel(
            "pipeline_bytes",
            "  %b = load i32, ptr addrspace(1) %pso, align 4\n  %v = add i32 %b, %t",
        ),
    ) else {
        return;
    };
    let u = unsupported(compile(&bc, "pipeline_bytes", &m3()).expect_err("refused"));
    assert_eq!(
        (u.feature, u.needs),
        (Feature::HandleAsPointer, Needs::NotRepresentable)
    );
}

/// A kernel `name` whose argument buffer `%ab` (Metal buffer 0, address space `space`) is a
/// pointer to `{render_pipeline_state render; compute_pipeline_state compute; uint scale; uint
/// bias;}` (24 bytes, no `air.buffer_size`, so it may be indexed past its first element), with a
/// `ulong` output buffer `%out` and the thread position `%t`; `body` precedes its `ret`.
fn pipeline_members_kernel(name: &str, space: u32, body: &str, decl: &str) -> String {
    sized_pipeline_members_kernel(name, space, body, decl, None, 24)
}

/// [`pipeline_members_kernel`] with `air.buffer_size` `buffer_size` (a reference when stated) and
/// `air.arg_type_size` `arg_type_size` (the stride between its elements when it covers the
/// layout's 24 bytes).
fn sized_pipeline_members_kernel(
    name: &str,
    space: u32,
    body: &str,
    decl: &str,
    buffer_size: Option<u32>,
    arg_type_size: u32,
) -> String {
    let buffer_size = buffer_size.map_or(String::new(), |n| {
        format!("!\"air.buffer_size\", i32 {n}, ")
    });
    format!(
        r#"%ab = type {{ ptr addrspace(1), ptr addrspace(1), i32, i32 }}

define void @{name}(ptr addrspace({space}) %ab, ptr addrspace(1) %out, i32 %t) {{
{body}
  ret void
}}
{decl}
!air.kernel = !{{!0}}
!air.version = !{{!11}}
!air.language_version = !{{!12}}
!0 = !{{ptr @{name}, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !9, !10}}
!3 = !{{i32 0, !"air.indirect_buffer", {buffer_size}!"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 {space}, !"air.struct_type_info", !4, !"air.arg_type_size", i32 {arg_type_size}, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Pipelines", !"air.arg_name", !"ab"}}
!4 = !{{i32 0, i32 8, i32 0, !"render_pipeline_state", !"render", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"compute_pipeline_state", !"compute", !"air.indirect_argument", !6, i32 16, i32 4, i32 0, !"uint", !"scale", !"air.indirect_argument", !7, i32 20, i32 4, i32 0, !"uint", !"bias", !"air.indirect_argument", !8}}
!5 = !{{i32 0, !"air.render_pipeline_state", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"render_pipeline_state", !"air.arg_name", !"render"}}
!6 = !{{i32 1, !"air.compute_pipeline_state", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"compute_pipeline_state", !"air.arg_name", !"compute"}}
!7 = !{{i32 2, !"air.indirect_constant", !"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"scale"}}
!8 = !{{i32 3, !"air.indirect_constant", !"air.location_index", i32 3, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"bias"}}
!9 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"out"}}
!10 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}}
!11 = !{{i32 2, i32 8, i32 0}}
!12 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// A kernel `name` whose argument buffer `%big` (Metal buffer 0, address space 2, no
/// `air.buffer_size`) is a pointer to `{uint data[8192]; compute_pipeline_state compute;}`
/// (32,776 bytes, the object at 32,768), with a `ulong` output `%out` and the thread position `%t`.
fn big_array_kernel(name: &str, body: &str) -> String {
    format!(
        r#"%big = type {{ [8192 x i32], ptr addrspace(1) }}

define void @{name}(ptr addrspace(2) %big, ptr addrspace(1) %out, i32 %t) {{
{body}
  ret void
}}

!air.kernel = !{{!0}}
!air.version = !{{!11}}
!air.language_version = !{{!12}}
!0 = !{{ptr @{name}, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !9, !10}}
!3 = !{{i32 0, !"air.indirect_buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 32776, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Big", !"air.arg_name", !"big"}}
!4 = !{{i32 0, i32 4, i32 8192, !"uint", !"data", !"air.indirect_argument", !5, i32 32768, i32 8, i32 0, !"compute_pipeline_state", !"compute", !"air.indirect_argument", !6}}
!5 = !{{i32 0, !"air.indirect_constant", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"data"}}
!6 = !{{i32 1, !"air.compute_pipeline_state", !"air.location_index", i32 8192, i32 1, !"air.arg_type_name", !"compute_pipeline_state", !"air.arg_name", !"compute"}}
!9 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"out"}}
!10 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}}
!11 = !{{i32 2, i32 8, i32 0}}
!12 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// A kernel `name` over Metal's layout of `struct Encoders { Pipelines stages[2]; command_buffer
/// commands; uint bias; }` with `struct Pipelines { render_pipeline_state render;
/// compute_pipeline_state compute; uint scale; }` (constant space, as `xcrun metal` writes its
/// `air.struct_type_info`: a nested struct array member, then the command buffer at 48 and `bias`
/// at 56), a `ulong` output `%out`, and the thread position `%t`.
fn nested_members_kernel(name: &str, body: &str) -> String {
    format!(
        r#"%Pipelines = type <{{ ptr addrspace(1), ptr addrspace(1), i32, [4 x i8] }}>
%Encoders = type <{{ [2 x %Pipelines], ptr addrspace(1), i32, [4 x i8] }}>

define void @{name}(ptr addrspace(2) %e, ptr addrspace(1) %out, i32 %t) {{
{body}
  ret void
}}

!air.kernel = !{{!0}}
!air.version = !{{!14}}
!air.language_version = !{{!15}}
!0 = !{{ptr @{name}, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !12, !13}}
!3 = !{{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 64, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 64, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Encoders", !"air.arg_name", !"e"}}
!4 = !{{!"air.struct_type_info", !5, i32 0, i32 24, i32 2, !"Pipelines", !"stages", !"air.indirect_argument", i32 0, i32 48, i32 8, i32 0, !"command_buffer", !"commands", !"air.indirect_argument", !9, i32 56, i32 4, i32 0, !"uint", !"bias", !"air.indirect_argument", !10}}
!5 = !{{i32 0, i32 8, i32 0, !"render_pipeline_state", !"render", !"air.indirect_argument", !6, i32 8, i32 8, i32 0, !"compute_pipeline_state", !"compute", !"air.indirect_argument", !7, i32 16, i32 4, i32 0, !"uint", !"scale", !"air.indirect_argument", !8}}
!6 = !{{i32 0, !"air.render_pipeline_state", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"render_pipeline_state", !"air.arg_name", !"render"}}
!7 = !{{i32 1, !"air.compute_pipeline_state", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"compute_pipeline_state", !"air.arg_name", !"compute"}}
!8 = !{{i32 2, !"air.indirect_constant", !"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"scale"}}
!9 = !{{i32 1, !"air.command_buffer", !"air.location_index", i32 6, i32 1, !"air.arg_type_name", !"command_buffer", !"air.arg_name", !"commands"}}
!10 = !{{i32 2, !"air.indirect_constant", !"air.location_index", i32 7, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"bias"}}
!12 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"out"}}
!13 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}}
!14 = !{{i32 2, i32 8, i32 0}}
!15 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// A kernel `name` whose constant argument buffer `%ab` (Metal buffer 0, 16 bytes) is
/// `{constant Pipelines* inner; uint x;}`, a pointer member to an argument buffer of
/// `pipeline_members_kernel`'s layout, with a `ulong` output `%out` and the thread position `%t`.
fn pointer_member_kernel(name: &str, body: &str) -> String {
    pointer_member_kernel_as(name, "air.indirect_buffer", body)
}

/// [`pointer_member_kernel`] with `%ab`'s role `role` (`air.indirect_buffer` or `air.buffer`).
fn pointer_member_kernel_as(name: &str, role: &str, body: &str) -> String {
    format!(
        r#"define void @{name}(ptr addrspace(2) %ab, ptr addrspace(1) %out, i32 %t) {{
{body}
  ret void
}}

!air.kernel = !{{!0}}
!air.version = !{{!11}}
!air.language_version = !{{!12}}
!0 = !{{ptr @{name}, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !9, !10}}
!3 = !{{i32 0, !"{role}", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Outer", !"air.arg_name", !"ab"}}
!4 = !{{i32 0, i32 8, i32 0, !"Pipelines", !"inner", !"air.indirect_argument", !13, i32 8, i32 4, i32 0, !"uint", !"x", !"air.indirect_argument", !8}}
!13 = !{{i32 0, !"air.indirect_buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !14, !"air.arg_type_size", i32 24, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Pipelines", !"air.arg_name", !"inner"}}
!14 = !{{i32 0, i32 8, i32 0, !"render_pipeline_state", !"render", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"compute_pipeline_state", !"compute", !"air.indirect_argument", !6, i32 16, i32 4, i32 0, !"uint", !"scale", !"air.indirect_argument", !7, i32 20, i32 4, i32 0, !"uint", !"bias", !"air.indirect_argument", !15}}
!5 = !{{i32 0, !"air.render_pipeline_state", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"render_pipeline_state", !"air.arg_name", !"render"}}
!6 = !{{i32 1, !"air.compute_pipeline_state", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"compute_pipeline_state", !"air.arg_name", !"compute"}}
!7 = !{{i32 2, !"air.indirect_constant", !"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"scale"}}
!15 = !{{i32 3, !"air.indirect_constant", !"air.location_index", i32 3, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"bias"}}
!8 = !{{i32 1, !"air.indirect_constant", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"x"}}
!9 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"out"}}
!10 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}}
!11 = !{{i32 2, i32 8, i32 0}}
!12 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The refusal a case expects.
#[derive(Clone, Copy, Debug)]
enum Refusal {
    /// `ArgumentBufferObjects`, not implemented.
    Objects,
    /// `HandleAsPointer`, not representable: a handle's use its family refuses.
    HandleAsPointer,
}

impl Refusal {
    fn expected(self) -> (Feature, Needs) {
        match self {
            Refusal::Objects => (Feature::ArgumentBufferObjects, Needs::NotImplemented),
            Refusal::HandleAsPointer => (Feature::HandleAsPointer, Needs::NotRepresentable),
        }
    }
}

/// A case: its entry stem, its body, and the refusal it expects at an instruction, if any.
type Case = (String, String, Option<(Refusal, u32)>);

/// Compile `entry` on every committed profile: `None` must link and validate, `Some((refusal,
/// at))` must be that refusal at instruction `at` of block 0.
fn expect_everywhere(bc: &[u8], entry: &str, want: Option<(Refusal, u32)>) {
    for (stem, p) in committed_profiles().expect("profiles") {
        match want {
            None => {
                ok(bc, entry, &p);
            }
            Some((refusal, at)) => {
                let u = unsupported(compile(bc, entry, &p).expect_err(entry));
                assert_eq!((u.feature, u.needs), refusal.expected(), "{stem} {entry}");
                assert_eq!(
                    u.location
                        .as_ref()
                        .and_then(|l| l.in_function())
                        .map(|(_, b, i)| (b, i)),
                    Some((Some(0), Some(at))),
                    "{stem} {entry}"
                );
            }
        }
    }
}

/// The bytes of an argument buffer's command-buffer and pipeline-state members are an
/// indirect-command object, never data. Metal's encoder writes the object's resource id there and
/// the binder has no class for one, so a translated pipeline must never observe a field (the
/// engine executor writes an opaque word there that is not Metal's: a null test of either is
/// true exactly when the field is unset). On every committed profile, in the device and the
/// constant space, through a pointer argument buffer that may be indexed past its first element
/// (`%ab` has no `air.buffer_size`):
/// - the data members compile, link, and validate: of the first element, of element 1, at a
///   dynamic element index, and by a memory copy of the data members alone;
/// - every other access that may reach an object member's bytes is `ArgumentBufferObjects`, not
///   implemented, at that access: an integer, a vector, or a half-word load of a field (of the
///   first element, of element 1, at a dynamic element index, at a dynamic index past the first
///   element), a load straddling a field, a memory copy over the struct, a dynamic index that may
///   land on a field, an atomic on one, a pointer selected between a field and other memory, and a
///   write of a field;
/// - only the handle load itself (a pointer load of exactly one field at a constant offset in the
///   first element) is admitted, and its integer value, null test, compare, selection, and store
///   are refused. A pointer load of a field at a dynamic offset (over the two fields, or over a
///   field and data), or of element 1's field, is no handle load and is refused at the load.
///
/// Before the rule covered them, a word of element 1 compiled on every profile, and element 1's
/// handle and each pointer load at a dynamic offset on the nine profiles with device addresses.
#[test]
fn an_indirect_command_object_members_bytes_are_never_data() {
    let data = "  %ps = getelementptr inbounds %ab, ptr addrspace(SP) %ab, i64 0, i32 2\n  %s = load i32, ptr addrspace(SP) %ps, align 4\n  %pb = getelementptr inbounds %ab, ptr addrspace(SP) %ab, i64 0, i32 3\n  %b = load i32, ptr addrspace(SP) %pb, align 4\n  %v = add i32 %s, %b\n  %z = zext i32 %v to i64\n  store i64 %z, ptr addrspace(1) %out, align 8";
    let handle = "  %pp = getelementptr inbounds %ab, ptr addrspace(SP) %ab, i64 0, i32 1\n  %h = load ptr addrspace(1), ptr addrspace(SP) %pp, align 8";
    let index = "  %a = and i32 %t, 1\n  %i = zext i32 %a to i64";
    let word_out = "  store i64 %w, ptr addrspace(1) %out, align 8";
    let bits_out = "  %w = ptrtoint ptr addrspace(1) %h to i64\n  store i64 %w, ptr addrspace(1) %out, align 8";
    let compiled: Vec<(&str, String)> = vec![
        ("data", data.into()),
        ("element_data", "  %pp = getelementptr inbounds %ab, ptr addrspace(SP) %ab, i64 1, i32 2\n  %w = load i32, ptr addrspace(SP) %pp, align 4\n  %z = zext i32 %w to i64\n  store i64 %z, ptr addrspace(1) %out, align 8".into()),
        ("dynamic_element_data", format!("{index}\n  %pp = getelementptr inbounds %ab, ptr addrspace(SP) %ab, i64 %i, i32 2\n  %w = load i32, ptr addrspace(SP) %pp, align 4\n  %z = zext i32 %w to i64\n  store i64 %z, ptr addrspace(1) %out, align 8")),
        ("copy_data", "  %pd = getelementptr inbounds i8, ptr addrspace(SP) %ab, i64 16\n  call void @llvm.memcpy.p1.pSP.i64(ptr addrspace(1) %out, ptr addrspace(SP) %pd, i64 8, i1 false)".into()),
    ];
    // (entry, body, the refused instruction, the refused feature).
    let objects = Refusal::Objects;
    let refused: Vec<(&str, String, u32, Refusal)> = vec![
        ("compute_word", format!("  %pp = getelementptr inbounds %ab, ptr addrspace(SP) %ab, i64 0, i32 1\n  %w = load i64, ptr addrspace(SP) %pp, align 8\n{word_out}"), 1, objects),
        ("render_word", format!("  %w = load i64, ptr addrspace(SP) %ab, align 8\n{word_out}"), 0, objects),
        ("half_word", "  %pp = getelementptr inbounds i8, ptr addrspace(SP) %ab, i64 12\n  %w = load i32, ptr addrspace(SP) %pp, align 4\n  %z = zext i32 %w to i64\n  store i64 %z, ptr addrspace(1) %out, align 8".into(), 1, objects),
        ("straddle", "  %pp = getelementptr inbounds i8, ptr addrspace(SP) %ab, i64 14\n  %w = load i32, ptr addrspace(SP) %pp, align 2\n  %z = zext i32 %w to i64\n  store i64 %z, ptr addrspace(1) %out, align 8".into(), 1, objects),
        ("pair", "  %w = load <2 x i64>, ptr addrspace(SP) %ab, align 8\n  %e = extractelement <2 x i64> %w, i32 1\n  store i64 %e, ptr addrspace(1) %out, align 8".into(), 0, objects),
        ("copy", "  call void @llvm.memcpy.p1.pSP.i64(ptr addrspace(1) %out, ptr addrspace(SP) %ab, i64 24, i1 false)".into(), 0, objects),
        ("dynamic", format!("  %i = zext i32 %t to i64\n  %pp = getelementptr inbounds i64, ptr addrspace(SP) %ab, i64 %i\n  %w = load i64, ptr addrspace(SP) %pp, align 8\n{word_out}"), 2, objects),
        ("element_word", format!("  %pp = getelementptr inbounds %ab, ptr addrspace(SP) %ab, i64 1, i32 1\n  %w = load i64, ptr addrspace(SP) %pp, align 8\n{word_out}"), 1, objects),
        ("dynamic_element_word", format!("  %i = zext i32 %t to i64\n  %pp = getelementptr inbounds %ab, ptr addrspace(SP) %ab, i64 %i, i32 1\n  %w = load i64, ptr addrspace(SP) %pp, align 8\n{word_out}"), 2, objects),
        ("later_element_word", format!("  %i = zext i32 %t to i64\n  %j = add i64 %i, 1\n  %pp = getelementptr inbounds %ab, ptr addrspace(SP) %ab, i64 %j, i32 1\n  %w = load i64, ptr addrspace(SP) %pp, align 8\n{word_out}"), 3, objects),
        ("handle_bits", format!("{handle}\n{bits_out}"), 1, objects),
        ("handle_null", format!("{handle}\n  %c = icmp eq ptr addrspace(1) %h, null\n  %w = zext i1 %c to i64\n{word_out}"), 1, objects),
        ("handle_store", format!("{handle}\n  store ptr addrspace(1) %h, ptr addrspace(1) %out, align 8"), 1, objects),
        ("handle_compare", format!("  %p0 = getelementptr inbounds %ab, ptr addrspace(SP) %ab, i64 0, i32 0\n  %h0 = load ptr addrspace(1), ptr addrspace(SP) %p0, align 8\n{handle}\n  %c = icmp eq ptr addrspace(1) %h, %h0\n  %w = zext i1 %c to i64\n{word_out}"), 1, objects),
        ("handle_select", format!("{handle}\n  %c = icmp eq i32 %t, 0\n  %s = select i1 %c, ptr addrspace(1) %h, ptr addrspace(1) null\n  %w = ptrtoint ptr addrspace(1) %s to i64\n{word_out}"), 1, objects),
        ("handle_offset", format!("{handle}\n  %g = getelementptr inbounds i8, ptr addrspace(1) %h, i64 0\n  %w = ptrtoint ptr addrspace(1) %g to i64\n{word_out}"), 1, Refusal::HandleAsPointer),
        ("element_handle", format!("  %pp = getelementptr inbounds %ab, ptr addrspace(SP) %ab, i64 1, i32 1\n  %h = load ptr addrspace(1), ptr addrspace(SP) %pp, align 8\n{bits_out}"), 1, objects),
        ("dynamic_handle_bits", format!("{index}\n  %pp = getelementptr inbounds ptr addrspace(1), ptr addrspace(SP) %ab, i64 %i\n  %h = load ptr addrspace(1), ptr addrspace(SP) %pp, align 8\n{bits_out}"), 3, objects),
        ("dynamic_handle_null", format!("{index}\n  %pp = getelementptr inbounds ptr addrspace(1), ptr addrspace(SP) %ab, i64 %i\n  %h = load ptr addrspace(1), ptr addrspace(SP) %pp, align 8\n  %c = icmp eq ptr addrspace(1) %h, null\n  %w = zext i1 %c to i64\n{word_out}"), 3, objects),
        ("dynamic_handle_store", format!("{index}\n  %pp = getelementptr inbounds ptr addrspace(1), ptr addrspace(SP) %ab, i64 %i\n  %h = load ptr addrspace(1), ptr addrspace(SP) %pp, align 8\n  store ptr addrspace(1) %h, ptr addrspace(1) %out, align 8"), 3, objects),
        ("mixed_handle_bits", format!("{index}\n  %j = add i64 %i, 1\n  %pp = getelementptr inbounds ptr addrspace(1), ptr addrspace(SP) %ab, i64 %j\n  %h = load ptr addrspace(1), ptr addrspace(SP) %pp, align 8\n{bits_out}"), 4, objects),
    ];
    let memcpy =
        "declare void @llvm.memcpy.p1.pSP.i64(ptr addrspace(1), ptr addrspace(SP), i64, i1)";
    for space in [1u32, 2] {
        let sp = space.to_string();
        let mut cases: Vec<Case> = compiled
            .iter()
            .map(|(name, body)| (name.to_string(), body.replace("SP", &sp), None))
            .collect();
        cases.extend(refused.iter().map(|(name, body, at, feature)| {
            (
                name.to_string(),
                body.replace("SP", &sp),
                Some((*feature, *at)),
            )
        }));
        if space == 1 {
            // A write of a field, an atomic on one, and a pointer selected between a field and
            // the output (the constant space is read-only and cannot alias `%out`).
            cases.push((
                "overwrite".into(),
                "  %pp = getelementptr inbounds %ab, ptr addrspace(1) %ab, i64 0, i32 1\n  store i64 0, ptr addrspace(1) %pp, align 8".into(),
                Some((objects, 1)),
            ));
            cases.push((
                "atomic".into(),
                format!("  %pp = getelementptr inbounds %ab, ptr addrspace(1) %ab, i64 0, i32 1\n  %w = atomicrmw add ptr addrspace(1) %pp, i64 0 monotonic, align 8\n{word_out}"),
                Some((objects, 1)),
            ));
            cases.push((
                "select".into(),
                format!("  %c = icmp eq i32 %t, 0\n  %pp = getelementptr inbounds %ab, ptr addrspace(1) %ab, i64 0, i32 1\n  %s = select i1 %c, ptr addrspace(1) %pp, ptr addrspace(1) %out\n  %w = load i64, ptr addrspace(1) %s, align 8\n{word_out}"),
                Some((objects, 3)),
            ));
        }
        for (name, body, want) in cases {
            let entry = format!("{name}_{space}");
            let decl = if name.starts_with("copy") {
                memcpy.replace("SP", &sp)
            } else {
                String::new()
            };
            let Some(bc) = assemble(
                &entry,
                &pipeline_members_kernel(&entry, space, &body, &decl),
            ) else {
                return;
            };
            expect_everywhere(&bc, &entry, want);
        }
    }

    // Metal's nested layout: `stages[1].scale` (40), `bias` (56), and `stages[t & 1].scale` (16
    // plus a multiple of 24) are data; a word at a dynamic index, `stages[1].compute` (32), and a
    // pointer load of `stages[t & 1].compute` (8 or 32, no constant offset) are not.
    let index = "  %a = and i32 %t, 1\n  %k = zext i32 %a to i64";
    let store = "  %z = zext i32 %v to i64\n  store i64 %z, ptr addrspace(1) %out, align 8";
    let nested = [
        (
            "nested_data",
            format!("{index}\n  %p1 = getelementptr inbounds %Encoders, ptr addrspace(2) %e, i64 0, i32 0, i64 1, i32 2\n  %s1 = load i32, ptr addrspace(2) %p1, align 8\n  %pb = getelementptr inbounds %Encoders, ptr addrspace(2) %e, i64 0, i32 2\n  %b = load i32, ptr addrspace(2) %pb, align 8\n  %pk = getelementptr inbounds %Encoders, ptr addrspace(2) %e, i64 0, i32 0, i64 %k, i32 2\n  %sk = load i32, ptr addrspace(2) %pk, align 8\n  %u = add i32 %s1, %b\n  %v = add i32 %u, %sk\n{store}"),
            None,
        ),
        (
            "nested_words",
            format!("{index}\n  %pk = getelementptr inbounds i64, ptr addrspace(2) %e, i64 %k\n  %w = load i64, ptr addrspace(2) %pk, align 8\n  store i64 %w, ptr addrspace(1) %out, align 8"),
            Some(3),
        ),
        (
            "nested_compute",
            "  %pc = getelementptr inbounds %Encoders, ptr addrspace(2) %e, i64 0, i32 0, i64 1, i32 1\n  %w = load i64, ptr addrspace(2) %pc, align 8\n  store i64 %w, ptr addrspace(1) %out, align 8".to_string(),
            Some(1),
        ),
        (
            "nested_dynamic_handle",
            format!("{index}\n  %pk = getelementptr inbounds %Encoders, ptr addrspace(2) %e, i64 0, i32 0, i64 %k, i32 1\n  %h = load ptr addrspace(1), ptr addrspace(2) %pk, align 8\n  %w = ptrtoint ptr addrspace(1) %h to i64\n  store i64 %w, ptr addrspace(1) %out, align 8"),
            Some(3),
        ),
    ];
    for (entry, body, at) in nested {
        let Some(bc) = assemble(entry, &nested_members_kernel(entry, &body)) else {
            return;
        };
        expect_everywhere(&bc, entry, at.map(|at| (objects, at)));
    }
}

/// An object's bytes reached through a run-time address, which the rule at each access cannot see.
/// Once an argument buffer that holds an object has its address become data (a `ptrtoint`, a store
/// into device memory), or a read loads a pointer member whose pointee holds an object, the entry's
/// first run-time dereference is `ArgumentBufferObjects`, not implemented, on every committed
/// profile: the escaped address read back through `inttoptr` or through memory, and every
/// dereference of the loaded pointer member (a word of its object, the handle at its field, and its
/// data alike). A pointer kept in a forwarding cell keeps its root, so its access is refused by the
/// rule itself. The other five compiled on the nine profiles with device addresses before this
/// rule.
#[test]
fn an_object_buffers_address_as_data_refuses_every_run_time_dereference() {
    let word_out = "  store i64 %w, ptr addrspace(1) %out, align 8";
    let escaped = [
        ("round_trip", format!("  %a = ptrtoint ptr addrspace(1) %ab to i64\n  %b = add i64 %a, 8\n  %p = inttoptr i64 %b to ptr addrspace(1)\n  %w = load i64, ptr addrspace(1) %p, align 8\n{word_out}"), 3),
        ("through_memory", format!("  %o1 = getelementptr inbounds i64, ptr addrspace(1) %out, i64 1\n  store ptr addrspace(1) %ab, ptr addrspace(1) %o1, align 8\n  %q = load ptr addrspace(1), ptr addrspace(1) %o1, align 8\n  %pp = getelementptr inbounds i8, ptr addrspace(1) %q, i64 8\n  %w = load i64, ptr addrspace(1) %pp, align 8\n{word_out}"), 4),
        ("through_cell", format!("  %cell = alloca ptr addrspace(1), align 8\n  store ptr addrspace(1) %ab, ptr %cell, align 8\n  %q = load ptr addrspace(1), ptr %cell, align 8\n  %pp = getelementptr inbounds i8, ptr addrspace(1) %q, i64 8\n  %w = load i64, ptr addrspace(1) %pp, align 8\n{word_out}"), 4),
    ];
    for (entry, body, at) in escaped {
        let Some(bc) = assemble(entry, &pipeline_members_kernel(entry, 1, &body, "")) else {
            return;
        };
        expect_everywhere(&bc, entry, Some((Refusal::Objects, at)));
    }
    let inner = "  %q = load ptr addrspace(2), ptr addrspace(2) %ab, align 8";
    let behind = [
        ("inner_word", format!("{inner}\n  %pp = getelementptr inbounds i8, ptr addrspace(2) %q, i64 8\n  %w = load i64, ptr addrspace(2) %pp, align 8\n{word_out}")),
        ("inner_handle", format!("{inner}\n  %pp = getelementptr inbounds i8, ptr addrspace(2) %q, i64 8\n  %h = load ptr addrspace(1), ptr addrspace(2) %pp, align 8\n  %w = ptrtoint ptr addrspace(1) %h to i64\n{word_out}")),
        ("inner_data", "  %q = load ptr addrspace(2), ptr addrspace(2) %ab, align 8\n  %pp = getelementptr inbounds i8, ptr addrspace(2) %q, i64 16\n  %v = load i32, ptr addrspace(2) %pp, align 4\n  %z = zext i32 %v to i64\n  store i64 %z, ptr addrspace(1) %out, align 8".to_string()),
    ];
    for (entry, body) in behind {
        let Some(bc) = assemble(entry, &pointer_member_kernel(entry, &body)) else {
            return;
        };
        expect_everywhere(&bc, entry, Some((Refusal::Objects, 2)));
    }
}

/// The bytes an object buffer holds, and how each is judged, follow its stated sizes, on every
/// committed profile, in the device and the constant space:
/// - a reference (`air.buffer_size` 40 over a 24-byte stride) holds a partial second element, whose
///   `render` (24) and `compute` (32) words are refused like the first element's;
/// - with an `air.arg_type_size` below the layout's end (20 over 24 bytes), only the bytes below
///   20 are judged by the layout, in a pointer and in a reference alike: `scale` (16) compiles,
///   and `bias` (20), which may be the next element's `render`, is refused.
///
/// Previously the element count rounded down (40 / 24 elements is one), so the partial element's
/// words compiled, and the bytes between 20 and the layout's end were judged by the first
/// element's layout, so `bias` compiled.
#[test]
fn the_bytes_an_object_buffer_holds_follow_its_stated_sizes() {
    let word = |member: u32| {
        format!("  %pp = getelementptr inbounds %ab, ptr addrspace(SP) %ab, i64 1, i32 {member}\n  %w = load i64, ptr addrspace(SP) %pp, align 8\n  store i64 %w, ptr addrspace(1) %out, align 8")
    };
    let uint = |member: u32| {
        format!("  %pp = getelementptr inbounds %ab, ptr addrspace(SP) %ab, i64 0, i32 {member}\n  %w = load i32, ptr addrspace(SP) %pp, align 4\n  %z = zext i32 %w to i64\n  store i64 %z, ptr addrspace(1) %out, align 8")
    };
    // (entry, air.buffer_size, air.arg_type_size, body, the refused instruction).
    let cases = [
        ("partial_data", Some(40), 24, uint(2), None),
        ("partial_render", Some(40), 24, word(0), Some(1)),
        ("partial_compute", Some(40), 24, word(1), Some(1)),
        ("short_scale", None, 20, uint(2), None),
        ("short_bias", None, 20, uint(3), Some(1)),
        ("short_reference_scale", Some(20), 20, uint(2), None),
        ("short_reference_bias", Some(20), 20, uint(3), Some(1)),
    ];
    for space in [1u32, 2] {
        let sp = space.to_string();
        for (name, buffer_size, arg_type_size, body, at) in &cases {
            let entry = format!("{name}_{space}");
            let body = body.replace("SP", &sp);
            let text = sized_pipeline_members_kernel(
                &entry,
                space,
                &body,
                "",
                *buffer_size,
                *arg_type_size,
            );
            let Some(bc) = assemble(&entry, &text) else {
                return;
            };
            expect_everywhere(&bc, &entry, at.map(|at| (Refusal::Objects, at)));
        }
    }
}

/// An access at least as wide as its step touches one contiguous range of bytes, which is tested
/// once, exactly. In `{uint data[8192]; compute_pipeline_state compute;}`, a word through a pointer
/// selected between `data[0]` and `data[8000]` has a joined offset shape, so it may start at any of
/// the 32,001 bytes between them, too many to list. On every committed profile it compiles: its
/// bytes are `[0, 32004)`, all data. Selected between `data[0]` and `compute` (at 32,768), it may
/// reach the object and is refused. Previously the first was checked per residue modulo the
/// 32,776-byte stride, past the 4,096 the rule lists, and was refused.
#[test]
fn an_access_as_wide_as_its_step_is_one_range() {
    let between = |far: &str| {
        format!("  %c = icmp eq i32 %t, 0\n  %p0 = getelementptr inbounds %big, ptr addrspace(2) %big, i64 0, i32 0, i64 0\n  %p1 = getelementptr inbounds %big, ptr addrspace(2) %big, i64 0, {far}\n  %pp = select i1 %c, ptr addrspace(2) %p0, ptr addrspace(2) %p1\n  %w = load i32, ptr addrspace(2) %pp, align 4\n  %z = zext i32 %w to i64\n  store i64 %z, ptr addrspace(1) %out, align 8")
    };
    let cases = [
        ("far_data", between("i32 0, i64 8000"), None),
        ("far_object", between("i32 1"), Some(4)),
    ];
    for (entry, body, at) in cases {
        let Some(bc) = assemble(entry, &big_array_kernel(entry, &body)) else {
            return;
        };
        expect_everywhere(&bc, entry, at.map(|at| (Refusal::Objects, at)));
    }
}

/// An `air.buffer` whose layout holds an argument-buffer member (`air.indirect_argument`) is
/// malformed: Metal makes any buffer whose struct holds a pointer or resource member an
/// `air.indirect_buffer`. A plain constant buffer of `{constant Pipelines* inner; uint x;}` whose
/// pointee's `compute` word a kernel reads is `InvalidInput` (`Abi(UnknownKey)`) on every committed
/// profile. Previously the object rule, which covers argument buffers, did not see it, and the read
/// compiled on the nine profiles with device addresses.
#[test]
fn a_plain_buffer_layout_with_argument_buffer_members_is_malformed() {
    let entry = "plain_inner_word";
    let body = "  %q = load ptr addrspace(2), ptr addrspace(2) %ab, align 8\n  %pp = getelementptr inbounds i8, ptr addrspace(2) %q, i64 8\n  %w = load i64, ptr addrspace(2) %pp, align 8\n  store i64 %w, ptr addrspace(1) %out, align 8";
    let Some(bc) = assemble(entry, &pointer_member_kernel_as(entry, "air.buffer", body)) else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        match compile(&bc, entry, &p).expect_err(entry) {
            Error::InvalidInput(i) => {
                assert_eq!(i.kind, InputErrorKind::Abi(AbiError::UnknownKey), "{stem}")
            }
            other => panic!("{stem}: expected an ABI error, got {other}"),
        }
    }
}

/// The metadata of a kernel `name` with a sampled texture (0), a sampler (1), an output buffer
/// (2), and the thread position (3).
const AGG_TYPES: &str = r#"%tex = type { ptr addrspace(1) }
%smp = type { ptr addrspace(2) }
%fetch = type { %tex, %smp }
"#;

fn agg_meta(name: &str) -> String {
    format!(
        r#"declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
!air.kernel = !{{!0}}
!air.version = !{{!7}}
!air.language_version = !{{!8}}
!0 = !{{ptr @{name}, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!6 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!7 = !{{i32 2, i32 8, i32 0}}
!8 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// A struct of a texture and a sampler built by `insertvalue` and passed by value (MSL's `struct {
/// texture2d<float> t; sampler s; }`, the MPS upsampler's form) is no SPIR-V value. Inlining puts
/// its `extractvalue`s in other blocks; each one resolves through the chain to its parameter, so
/// the sample binds the texture and sampler directly (no switch-at-use), through a nested
/// extraction (`extractvalue %fetch, 0` then `extractvalue %tex, 0`) too.
#[test]
fn handle_aggregates_resolve_through_their_insertions() {
    let body = format!(
        r#"{AGG_TYPES}
define void @agg_sample(ptr addrspace(1) %t, ptr addrspace(2) %s, ptr addrspace(1) %out, i32 %gid) {{
  %a = insertvalue %fetch poison, ptr addrspace(1) %t, 0, 0
  %b = insertvalue %fetch %a, ptr addrspace(2) %s, 1, 0
  %v = call <4 x float> @fetch_at(%fetch %b, i32 %gid)
  %i = zext i32 %gid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}}
define internal <4 x float> @fetch_at(%fetch %f, i32 %gid) {{
entry:
  %c = icmp eq i32 %gid, 0
  br i1 %c, label %wide, label %sample
wide:
  %inner = extractvalue %fetch %f, 0
  %tw = extractvalue %tex %inner, 0
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %tw, i32 0)
  %wf = uitofp i32 %w to float
  %wv = insertelement <4 x float> zeroinitializer, float %wf, i64 0
  ret <4 x float> %wv
sample:
  %ts = extractvalue %fetch %f, 0, 0
  %ss = extractvalue %fetch %f, 1, 0
  %r = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %ts, ptr addrspace(2) %ss, <2 x float> zeroinitializer, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %rv = extractvalue {{ <4 x float>, i8 }} %r, 0
  ret <4 x float> %rv
}}
{}"#,
        agg_meta("agg_sample")
    );
    let Some(bc) = assemble("agg_sample", &body) else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = compile(&bc, "agg_sample", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        // Two insertions and the nested extraction are no value; the three handle extractions
        // are constant handle indices.
        assert_eq!(uses(&a, "arith.aggregate.pointer"), 3, "{stem}");
        assert_eq!(uses(&a, "arith.aggregate.handle"), 3, "{stem}");
        assert_eq!(count(&a, OP_SWITCH), 0, "{stem}");
        assert_eq!(texture_slots(&a), vec![0], "{stem}");
    }
}

/// A heap texture (loaded from an argument buffer) inside a struct is forwarded: the `extractvalue`
/// in the inlined callee's block reads the lookup's element, carried there like any operand. A
/// merge of a handle struct with itself is a member-wise phi whose extraction still resolves to the
/// one texture.
#[test]
fn a_heap_handle_aggregate_forwards_its_element_and_a_merged_one_resolves() {
    let body = r#"%tex = type { ptr addrspace(1) }
%struct.Resources = type { %tex }

define void @agg_heap(ptr addrspace(2) %res, ptr addrspace(1) %out, i32 %gid) {
  %m = getelementptr inbounds %struct.Resources, ptr addrspace(2) %res, i64 0, i32 0, i32 0
  %h = load ptr addrspace(1), ptr addrspace(2) %m, align 8
  %a = insertvalue %tex poison, ptr addrspace(1) %h, 0
  %v = call <4 x float> @read_at(%tex %a, i32 %gid)
  %i = zext i32 %gid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}
define internal <4 x float> @read_at(%tex %t, i32 %gid) {
  %h = extractvalue %tex %t, 0
  %s = call ptr addrspace(2) @air.get_read_sampler()
  %c = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  %r = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %h, ptr addrspace(2) %s, <2 x i32> %c, <2 x i32> zeroinitializer, i32 0, i32 0)
  %rv = extractvalue { <4 x float>, i8 } %r, 0
  ret <4 x float> %rv
}
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
!air.kernel = !{!0}
!air.version = !{!8}
!air.language_version = !{!9}
!0 = !{ptr @agg_heap, !1, !2}
!1 = !{}
!2 = !{!3, !6, !7}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Resources", !"air.arg_name", !"res"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"t", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!6 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!7 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!8 = !{i32 2, i32 8, i32 0}
!9 = !{!"Metal", i32 4, i32 0, i32 0}
"#;
    let Some(bc) = assemble("agg_heap", body) else {
        return;
    };
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        match compile(&bc, "agg_heap", &p) {
            Ok(a) => {
                assert!(has_sampled_heap(&p), "{stem}");
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
                assert_eq!(uses(&a, "arith.aggregate.forward"), 1, "{stem}");
                assert_eq!(uses(&a, "access.handle.lookup"), 1, "{stem}");
                assert_eq!(uses(&a, heap_read(&p)), 1, "{stem}");
                compiled += 1;
            }
            Err(e) => {
                assert!(!has_sampled_heap(&p), "{stem}: {e}");
                assert_eq!(
                    unsupported(e).feature,
                    Feature::ArgumentBufferTextures,
                    "{stem}"
                );
            }
        }
    }
    assert!(compiled > 0, "no committed profile has a heap");

    let merged = format!(
        r#"{AGG_TYPES}
define void @agg_merged(ptr addrspace(1) %t, ptr addrspace(2) %s, ptr addrspace(1) %out, i32 %gid) {{
entry:
  %a = insertvalue %fetch poison, ptr addrspace(1) %t, 0, 0
  %b = insertvalue %fetch %a, ptr addrspace(2) %s, 1, 0
  %c = icmp eq i32 %gid, 0
  br i1 %c, label %x, label %y
x:
  br label %y
y:
  %f = phi %fetch [ %b, %entry ], [ %b, %x ]
  %tw = extractvalue %fetch %f, 0, 0
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %tw, i32 0)
  store i32 %w, ptr addrspace(1) %out, align 4
  ret void
}}
{}"#,
        agg_meta("agg_merged")
    );
    let Some(bc) = assemble("agg_merged", &merged) else {
        return;
    };
    let a = ok(&bc, "agg_merged", &m3());
    assert_eq!(uses(&a, "arith.aggregate.handle"), 1);
    assert_eq!(uses(&a, "arith.aggregate.member"), 0);
    assert_eq!(texture_slots(&a), vec![0]);
}

/// A phi of two texture-and-sampler structs holding different textures is a member-wise composite;
/// the texture member's extraction is a composite extract of the phi's handle index, and the query
/// switches over both candidate textures. `out[0]` picks the edge (0 takes `u`, `[[texture(1)]]`),
/// and `out[1]` receives the chosen texture's width.
#[test]
fn a_merge_of_two_handle_structs_is_member_wise() {
    let two = format!(
        r#"{AGG_TYPES}
define void @agg_two(ptr addrspace(1) %t, ptr addrspace(2) %s, ptr addrspace(1) %out, i32 %gid, ptr addrspace(1) %u) {{
entry:
  %a = insertvalue %fetch poison, ptr addrspace(1) %t, 0, 0
  %b = insertvalue %fetch %a, ptr addrspace(2) %s, 1, 0
  %k = load i32, ptr addrspace(1) %out, align 4
  %c = icmp eq i32 %k, 0
  br i1 %c, label %x, label %y
x:
  %a2 = insertvalue %fetch poison, ptr addrspace(1) %u, 0, 0
  %b2 = insertvalue %fetch %a2, ptr addrspace(2) %s, 1, 0
  br label %y
y:
  %f = phi %fetch [ %b, %entry ], [ %b2, %x ]
  %tw = extractvalue %fetch %f, 0, 0
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %tw, i32 0)
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
}}
{}"#,
        agg_meta("agg_two")
            .replace("!2 = !{!3, !4, !5, !6}", "!2 = !{!3, !4, !5, !6, !9}")
            .replace(
                "!8 = !{!\"Metal\", i32 4, i32 0, i32 0}",
                "!8 = !{!\"Metal\", i32 4, i32 0, i32 0}\n!9 = !{i32 4, !\"air.texture\", !\"air.location_index\", i32 1, i32 1, !\"air.sample\", !\"air.arg_type_name\", !\"texture2d<float, sample>\", !\"air.arg_name\", !\"u\"}"
            )
    );
    let Some(bc) = assemble("agg_two", &two) else {
        return;
    };
    let tex = |index| ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Texture,
        index,
    };
    let images = vec![
        Image::new(tex(0), [5, 3, 1], 1, 1, 1),
        Image::new(tex(1), [9, 3, 1], 1, 1, 1),
    ];
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = ok(&bc, "agg_two", &p);
        assert_eq!(uses(&a, "arith.aggregate.member"), 1, "{stem}");
        assert_eq!(uses(&a, "texture.query.switch"), 1, "{stem}");
        assert_eq!(texture_slots(&a), vec![0, 1], "{stem}");
        let (plan, m) = typed_module(&bc, "agg_two", &p).expect("module");
        for (k, width) in [(0u32, 9u32), (1, 5)] {
            let mut out = vec![0u8; 16];
            out[..4].copy_from_slice(&k.to_le_bytes());
            let res = Resources {
                buffers: vec![Buffer {
                    slot: Some(ResourceSlot {
                        stage: Stage::Kernel,
                        kind: ResourceKind::Buffer,
                        index: 0,
                    }),
                    address: None,
                    bytes: out,
                }],
                push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
                spec: Default::default(),
            };
            let (o, _) = run_with_images(&m, &Config::default(), res, images.clone())
                .unwrap_or_else(|e| panic!("{stem} {k}: {e:?}"));
            let got = u32::from_le_bytes(o.buffers[0][4..8].try_into().expect("word"));
            assert_eq!(got, width, "{stem} {k}");
        }
    }
}

/// A kernel `name` over the texture array `ts` (`array<texture2d<float>, count>
/// [[texture(0)]]`, a thread pointer to its handles) and `out` (`[[buffer(0)]]`, `uint *`):
/// `out[1]` receives the width of `ts[out[0]]`, whose handle `read` loads from the element
/// pointer `%e` into `%h`, ending its block with a branch to the query's block `use`.
fn array_width(name: &str, count: u32, read: &str) -> String {
    format!(
        r#"define void @{name}(ptr readonly byval([{count} x ptr addrspace(1)]) %ts, ptr addrspace(1) %out, i32 %gid) {{
entry:
  %k = load i32, ptr addrspace(1) %out, align 4
  %i = zext i32 %k to i64
  %e = getelementptr inbounds [{count} x ptr addrspace(1)], ptr %ts, i64 0, i64 %i
{read}
use:
  %x = call i32 @air.get_width_texture_2d(ptr addrspace(1) %h, i32 0)
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %x, ptr addrspace(1) %o, align 4
  ret void
}}
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
!air.kernel = !{{!0}}
!air.version = !{{!6}}
!air.language_version = !{{!7}}
!0 = !{{ptr @{name}, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 {count}, !"air.sample", !"air.arg_type_name", !"array<texture2d<float, sample>, {count}>", !"air.arg_name", !"ts"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!6 = !{{i32 2, i32 8, i32 0}}
!7 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// Compile `entry` of `bc` on every committed profile and run it for each `(k, width)`: texture
/// `j` of the array is `widths[j]` texels wide, and `out[1]` must be `width`.
fn check_array_width(bc: &[u8], entry: &str, widths: &[u32], runs: &[(u32, u32)]) {
    let images: Vec<Image> = (0u32..)
        .zip(widths)
        .map(|(j, w)| {
            let slot = ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Texture,
                index: j,
            };
            Image::new(slot, [*w, 2, 1], 1, 1, 1)
        })
        .collect();
    for (stem, p) in committed_profiles().expect("profiles") {
        ok(bc, entry, &p);
        let (plan, m) = typed_module(bc, entry, &p).expect("module");
        for &(k, width) in runs {
            let mut out = vec![0u8; 16];
            out[..4].copy_from_slice(&k.to_le_bytes());
            let res = Resources {
                buffers: vec![Buffer {
                    slot: Some(ResourceSlot {
                        stage: Stage::Kernel,
                        kind: ResourceKind::Buffer,
                        index: 0,
                    }),
                    address: None,
                    bytes: out,
                }],
                push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
                spec: Default::default(),
            };
            let (o, _) = run_with_images(&m, &Config::default(), res, images.clone())
                .unwrap_or_else(|e| panic!("{stem} {k}: {e:?}"));
            let got = u32::from_le_bytes(o.buffers[0][4..8].try_into().expect("word"));
            assert_eq!(got, width, "{stem} {k}");
        }
    }
}

/// A texture array of one element (an `array_ref` whose count a function constant makes one) is
/// still a thread pointer to its handles, not a texture: the IR type decides, not the location
/// count. A load of its element is texture 0.
#[test]
fn a_texture_array_of_one_is_an_array() {
    let read = "  %h = load ptr addrspace(1), ptr %e, align 8\n  br label %use";
    let Some(bc) = assemble("array_one", &array_width("array_one", 1, read)) else {
        return;
    };
    let a = ok(&bc, "array_one", &m3());
    assert_eq!(texture_slots(&a), vec![0]);
    check_array_width(&bc, "array_one", &[7], &[(0, 7)]);
}

/// A texture handle read out of a texture array as an integer and turned back into a pointer (`load
/// i64` then `inttoptr`, MPS's ASTC matrix multiply) is the element's handle: the integer load is a
/// handle word, held as the handle index across blocks, and the `inttoptr` a copy of it. Each `k`
/// reads texture `k`'s width.
#[test]
fn a_handle_read_through_an_integer_is_its_element() {
    // The integer crosses into the query's block, where `inttoptr` makes it the handle.
    let read = "  %w = load i64, ptr %e, align 8\n  %in = icmp ult i32 %k, 3\n  br i1 %in, label %use, label %done\ndone:\n  ret void";
    let text = array_width("word_read", 3, read).replace(
        "use:\n",
        "use:\n  %h = inttoptr i64 %w to ptr addrspace(1)\n",
    );
    let Some(bc) = assemble("word_read", &text) else {
        return;
    };
    let a = ok(&bc, "word_read", &m3());
    assert_eq!(uses(&a, "access.handle.array_load.clamped"), 1);
    assert_eq!(uses(&a, "access.handle.copy"), 1);
    assert_eq!(texture_slots(&a), vec![0, 1, 2]);
    check_array_width(&bc, "word_read", &[5, 7, 9], &[(0, 5), (1, 7), (2, 9)]);

    // An integer observed otherwise (stored here) would be the guest's bits of the handle, which
    // have no value: refused, never guessed.
    let observed = text.replace(
        "\n  ret void\n}\ndeclare",
        "\n  %o2 = getelementptr inbounds i64, ptr addrspace(1) %out, i64 1\n  store i64 %w, ptr addrspace(1) %o2, align 8\n  ret void\n}\ndeclare",
    );
    assert_ne!(observed, text);
    let Some(bc) = assemble("word_observed", &observed) else {
        return;
    };
    let u = unsupported(compile(&bc, "word_read", &m3()).expect_err("refused"));
    assert_eq!(u.feature, Feature::Lowering(LowerFamily::Access));
    assert_eq!(u.needs, Needs::NotImplemented);
}

/// The owned `tail_argument_samplers` fixture.
fn argument_samplers_fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("tail_argument_samplers.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => {
            panic!("llvm-as rejected tail_argument_samplers: {why}")
        }
    }
}

/// The sampler uses `(compare, integer)` of the set-1 sampler arrays a module declares.
fn sampler_arrays(a: &FunctionArtifact) -> Vec<(bool, bool)> {
    let mut out: Vec<(bool, bool)> = a
        .interface
        .descriptors
        .iter()
        .filter_map(|d| match d.slot.kind {
            ResourceKind::Heap(HandleClass::Sampler { compare, integer }) => {
                Some((compare, integer))
            }
            _ => None,
        })
        .collect();
    out.sort();
    out
}

fn is_sampler_table(k: &ResourceKind) -> bool {
    matches!(k, ResourceKind::Synthetic(SyntheticBinding::SamplerTable))
}

/// A sampler loaded from an argument buffer beside the texture it samples is a heap handle of its
/// own domain: a lookup in the stage's sampler table (set 0) whose element indexes the set-1
/// sampler array of the sample's use (comparison, integer texel), non-uniformly, and
/// `OpSampledImage` of the heap image and that sampler. Each entry declares one sampler array of
/// its use; every access in the sampler's coordinates carries the argument buffer's
/// pixel-coordinate near-equivalent (`ArgumentPixelSamplerScaling`; never `PixelSamplerScaling`),
/// a comparison or an integer texel samples through its own use, and a fragment samples at its
/// implicit level. Each sample reads its texel's centre where the sampler's word states a nearest
/// pixel sampler of one isotropic level (its test of the word's coordinates, filters, mip filter
/// and anisotropy, `SamplerWord::nearest_pixel`; an integer texel's leaves out the filters, which
/// it reads nearest whatever they say) and fetches nothing. A profile without the heap's features
/// refuses each naming them.
#[test]
fn argument_buffer_samplers_are_heap_lookups() {
    let Some(bc) = argument_samplers_fixture() else {
        return;
    };
    // Entry, strategy, sampler arrays, sample opcode.
    let cases: [(&str, &str, (bool, bool), u16); 5] = [
        (
            "argument_sampler_sample",
            "sampler.argument_sampler.sample",
            (false, false),
            OP_IMAGE_SAMPLE_EXPLICIT_LOD,
        ),
        (
            "argument_sampler_edges",
            "sampler.argument_sampler.sample",
            (false, false),
            OP_IMAGE_SAMPLE_EXPLICIT_LOD,
        ),
        (
            "argument_sampler_compare",
            "sampler.argument_sampler.sample",
            (true, false),
            OP_IMAGE_SAMPLE_DREF_EXPLICIT_LOD,
        ),
        (
            "argument_sampler_integer",
            "sampler.argument_sampler.sample",
            (false, true),
            OP_IMAGE_SAMPLE_EXPLICIT_LOD,
        ),
        (
            "argument_sampler_fragment",
            "sampler.argument_sampler.sample",
            (false, false),
            OP_IMAGE_SAMPLE_IMPLICIT_LOD,
        ),
    ];
    for (entry, strategy, use_, op) in cases {
        let mut compiled = 0;
        for (stem, p) in committed_profiles().expect("profiles") {
            let a = match compile(&bc, entry, &p) {
                Ok(a) => a,
                Err(e) => {
                    assert!(!has_sampled_heap(&p), "{stem} {entry}: {e}");
                    let u = unsupported(e);
                    assert_eq!(u.feature, Feature::ArgumentBufferTextures, "{stem} {entry}");
                    assert!(matches!(u.needs, Needs::AnyOf(_)), "{stem}: {:?}", u.needs);
                    continue;
                }
            };
            assert!(has_sampled_heap(&p), "{stem} {entry}");
            let (t, _) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {entry}: L3: {e}"));
            assert_eq!(uses(&a, strategy), 1, "{stem} {entry}");
            assert_eq!(uses(&a, "access.handle.lookup"), 1, "{stem} {entry}");
            assert_eq!(
                uses(&a, "access.handle.sampler_lookup"),
                1,
                "{stem} {entry}"
            );
            assert!(texture_slots(&a).is_empty(), "{stem} {entry}");
            assert!(heap_samplers(&a).is_empty(), "{stem} {entry}");
            assert_eq!(sampler_arrays(&a), [use_], "{stem} {entry}");
            let tables = bindings_of(&t, is_sampler_table);
            assert_eq!(tables.len(), 1, "{stem} {entry}");
            assert_eq!(tables[0].0, 0, "{stem} {entry}");
            // The image's array (its class's) and the sampler's, both in set 1.
            let heap = bindings_of(&t, is_heap);
            assert_eq!(heap.len(), 2, "{stem} {entry}");
            assert!(heap.iter().all(|(set, _)| *set == 1), "{stem} {entry}");
            for d in &a.interface.descriptors {
                if let ResourceKind::Heap(c) = d.slot.kind {
                    assert!(heap_classes(&p).contains(&c), "{stem} {entry}: {c:?}");
                }
            }
            assert_eq!(count(&a, OP_SAMPLED_IMAGE), 1, "{stem} {entry}");
            assert_eq!(count(&a, op), 1, "{stem} {entry}");
            assert_eq!(
                count(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD),
                usize::from(op == OP_IMAGE_SAMPLE_EXPLICIT_LOD),
                "{stem} {entry}"
            );
            assert_eq!(count(&a, OP_SWITCH), 0, "{stem} {entry}");
            // Non-uniform: the image's element and the sampler's (chain and load each), their
            // sampled image, and the image the pixel-coordinate scaling queries.
            assert_eq!(count(&a, OP_IMAGE_FETCH), 0, "{stem} {entry}");
            assert_eq!(non_uniform(&a), 6, "{stem} {entry}");
            assert!(
                a.precision
                    .contains(&PrecisionNote::ArgumentPixelSamplerScaling),
                "{stem} {entry}"
            );
            assert!(
                !a.precision.contains(&PrecisionNote::PixelSamplerScaling),
                "{stem} {entry}"
            );
            // The texel centre's test of the word's nearest pixel sampler.
            let centre = SamplerWord::nearest_pixel(use_.1);
            assert!(
                u32_constants(&a).contains(&centre),
                "{stem} {entry}: no test of the word's nearest pixel sampler"
            );
            compiled += 1;
        }
        assert!(compiled > 0, "{entry}: no committed profile has its heap");
    }
}

/// A kernel reading a texture and its sampler from an argument buffer has one set-1 sampler
/// array, of `heap_sampler_capacity` elements, on every committed profile that compiles it: the
/// handle table's 65536 where the update-after-bind sampler limits pass it (MoltenVK's 500000,
/// lavapipe's, NVIDIA's), the per-stage limit's 16 on the desktop baseline. On the M3's profile
/// with its per-stage update-after-bind sampler limit lowered to 100, or its set's to 200, the
/// array and the answer follow it. A profile that refuses the kernel answers 0, and so does the
/// M3's without `descriptorBindingPartiallyBound` or `shaderSampledImageArrayNonUniformIndexing`,
/// which refuses it.
#[test]
fn an_argument_sampler_array_has_the_heap_sampler_capacity() {
    let Some(bc) = argument_samplers_fixture() else {
        return;
    };
    let library = Library::from_air(&bc).expect("library");
    // The answer on `desc`, and the element count of the kernel's set-1 sampler array or the
    // kernel's refusal.
    let array = |desc: DeviceDescription| -> (u32, Result<u32, Error>) {
        let dev = Device::new(desc, DevicePolicy::default()).expect("device");
        let f = library.function("argument_sampler_sample").expect("entry");
        let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
        let count = dev
            .compile_compute_pipeline(&d, &Budget::standard())
            .map(|pipe| {
                let samplers: Vec<u32> = (pipe.layout.sets.iter())
                    .filter(|set| set.set == 1)
                    .flat_map(|set| &set.bindings)
                    .filter(|b| b.ty == vk::DescriptorType::SAMPLER)
                    .map(|b| b.count)
                    .collect();
                assert_eq!(samplers.len(), 1, "one set-1 sampler array: {samplers:?}");
                samplers[0]
            });
        (heap_sampler_capacity(dev.profile()), count)
    };
    let mut compiled = Vec::new();
    for (stem, desc) in committed_descriptions().expect("descriptions") {
        match array(desc.clone()) {
            (capacity, Ok(count)) => {
                assert_eq!(count, capacity, "{stem}");
                compiled.push((stem.as_str(), count));
            }
            (capacity, Err(e)) => assert_eq!(capacity, 0, "{stem}: refused ({e})"),
        }
    }
    eprintln!("set-1 sampler arrays: {compiled:?}");
    for stem in ["moltenvk-m3", "lavapipe", "nvidia-rtx5080-laptop"] {
        assert!(compiled.contains(&(stem, 65536)), "{stem}: {compiled:?}");
    }
    assert!(compiled.contains(&("lunarg-desktop-baseline-2023", 16)));
    let m3 = || {
        (committed_descriptions().expect("descriptions").iter())
            .find(|(stem, _)| stem == "moltenvk-m3")
            .map(|(_, desc)| desc.clone())
            .expect("moltenvk-m3")
    };
    let mut stage = m3();
    stage
        .limits
        .max_per_stage_descriptor_update_after_bind_samplers = 100;
    let (capacity, count) = array(stage);
    assert_eq!((capacity, count.ok()), (100, Some(100)), "per stage");
    let mut set = m3();
    set.limits.max_descriptor_set_update_after_bind_samplers = 200;
    let (capacity, count) = array(set);
    assert_eq!((capacity, count.ok()), (200, Some(200)), "per set");
    for lacking in [
        VkFeature::DescriptorBindingPartiallyBound,
        VkFeature::ShaderSampledImageArrayNonUniformIndexing,
    ] {
        let mut desc = m3();
        desc.features.remove(&lacking);
        let (capacity, count) = array(desc);
        assert_eq!(capacity, 0, "{lacking:?}");
        assert!(count.is_err(), "{lacking:?}: {count:?}");
    }
}

/// A sample through an argument buffer's sampler tests the sampler's entry for a cubic filter
/// (`RuntimeFact::CubicArgumentSampler`) in a kernel on every profile with the heap, marking the
/// status by address where the device addresses buffers and through the kernel's binding of it
/// elsewhere; a fragment function's never does, whose store would cost it the device's
/// hidden-surface removal, so the planner refuses a listed cubic sampler beside it.
#[test]
fn only_non_fragment_stages_test_argument_samplers_for_a_cubic_filter() {
    let Some(bc) = argument_samplers_fixture() else {
        return;
    };
    let (mut compiled, mut bound) = (0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let (Ok(kernel), Ok(fragment)) = (
            compile(&bc, "argument_sampler_sample", &p),
            compile(&bc, "argument_sampler_fragment", &p),
        ) else {
            assert!(!has_sampled_heap(&p), "{stem}");
            continue;
        };
        assert!(tests_cubic(&kernel), "{stem}: the kernel");
        let reach = status_reach(&kernel);
        assert_eq!(reach, status_reach_on(&p), "{stem}: the kernel");
        assert!(!tests_cubic(&fragment), "{stem}: the fragment function");
        assert_eq!(status_reach(&fragment), StatusReach::None, "{stem}");
        link_and_validate(&kernel, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        compiled += 1;
        bound += usize::from(reach == StatusReach::Binding);
    }
    assert!(compiled > 0, "no committed profile has the heap");
    assert!(
        bound > 0,
        "no committed profile with the heap but no addresses"
    );
}

/// Whether `a` tests an argument buffer's sampler for a cubic filter.
fn tests_cubic(a: &FunctionArtifact) -> bool {
    a.facts
        .iter()
        .any(|f| matches!(f, RuntimeFact::CubicArgumentSampler { .. }))
}

/// Where the cubic test's binding of the status would take a kernel past its storage buffers (a
/// device that does not address buffers, `lunarg-desktop-baseline-2023` with its limit lowered
/// step by step), the kernel gives the binding back: it compiles without the test, as a fragment
/// function does, so the planner refuses a listed cubic sampler beside it. Every compile stays
/// within the limit, and at the full kernel's own count of storage buffers the kernel keeps the
/// binding.
#[test]
fn the_cubic_tests_binding_gives_way_to_the_storage_buffer_limit() {
    let Some(bc) = argument_samplers_fixture() else {
        return;
    };
    let stem = "lunarg-desktop-baseline-2023";
    let full = profile(stem, DevicePolicy::default()).expect("profile");
    assert!(has_sampled_heap(&full), "{stem}");
    let kernel = compile(&bc, "argument_sampler_sample", &full).expect("kernel");
    assert_eq!(status_reach(&kernel), StatusReach::Binding);
    let with = storage_buffers(&kernel);
    let mut given_back = None;
    for limit in (0..=with).rev() {
        let p =
            profile_with_storage_buffers(stem, DevicePolicy::default(), limit).expect("profile");
        let a = match compile(&bc, "argument_sampler_sample", &p) {
            Ok(a) => a,
            Err(Error::Unsupported(u)) if u.feature == Feature::DescriptorCount => break,
            Err(e) => panic!("limit {limit}: {e}"),
        };
        assert!(storage_buffers(&a) <= limit, "limit {limit}");
        match status_reach(&a) {
            StatusReach::Binding => {
                assert!(tests_cubic(&a), "limit {limit}");
                assert!(given_back.is_none(), "limit {limit}: the binding back");
            }
            StatusReach::None => {
                assert!(!tests_cubic(&a), "limit {limit}");
                assert!(
                    !a.facts.contains(&RuntimeFact::ExecutionStatus),
                    "limit {limit}"
                );
                assert!(limit < with, "limit {limit}: given back at the full count");
                given_back.get_or_insert(limit);
            }
            StatusReach::Address => panic!("limit {limit}: an address without addresses"),
        }
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("limit {limit}: L3: {e}"));
    }
    let limit = given_back.expect("the binding is never given back");
    println!("{stem}: the cubic test's binding given back at {limit} storage buffers of {with}");
}

/// Synthetic AIR written for this test: kernel `k` samples the `texture2d<float>` its argument
/// buffer (buffer 2, `{ texture2d<float> t; sampler s; }`) holds through the sampler beside it at
/// level 0, `first` times in its first block, then, where `second`, once more in a block its
/// thread takes only where `gid` is 0, storing each sample to buffer 1.
fn repeated_argument_samples(first: usize, second: bool) -> String {
    let sample = |n: &str| {
        format!(
            "  %r{n} = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> <float 0.25, float 0.5>, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)\n  %v{n} = extractvalue {{ <4 x float>, i8 }} %r{n}, 0\n  store <4 x float> %v{n}, ptr addrspace(1) %q, align 16\n"
        )
    };
    let mut body: String = (0..first).map(|j| sample(&j.to_string())).collect();
    body += match second {
        true => "  %c = icmp eq i32 %gid, 0\n  br i1 %c, label %then, label %end\nthen:\n",
        false => "  br label %end\n",
    };
    if second {
        body += &sample("x");
        body += "  br label %end\n";
    }
    format!(
        r#"%struct.M = type {{ ptr addrspace(1), ptr addrspace(2) }}

define void @k(ptr addrspace(1) %out, ptr addrspace(2) %m, i32 %gid) {{
  %i = zext i32 %gid to i64
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  %pt = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %ps = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 1
  %s = load ptr addrspace(2), ptr addrspace(2) %ps, align 8
{body}end:
  ret void
}}

declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{{!0}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!4, !5, !7}}
!4 = !{{i32 0, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!5 = !{{i32 1, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !8, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"M", !"air.arg_name", !"m"}}
!8 = !{{i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"t", !"air.indirect_argument", !9, i32 8, i32 8, i32 0, !"sampler", !"s", !"air.indirect_argument", !10}}
!9 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!10 = !{{i32 1, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
!7 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The cubic reach test runs once per handle in a block: four samples of one argument buffer's
/// sampler in a block test it as one does, and a fifth in another block tests it again.
#[test]
fn a_block_tests_an_argument_sampler_for_a_cubic_filter_once() {
    const OP_ATOMIC_OR: u16 = 241;
    let assemble = |first, second| match metal2vulkan_devtools::llvm_assemble_text(
        &repeated_argument_samples(first, second),
    ) {
        Ok(b) => Some(b),
        Err(metal2vulkan_devtools::AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(metal2vulkan_devtools::AssembleError::Rejected(why)) => panic!("llvm-as: {why}"),
    };
    let (Some(one), Some(four), Some(five)) =
        (assemble(1, false), assemble(4, false), assemble(4, true))
    else {
        return;
    };
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let Ok(a1) = compile(&one, "k", &p) else {
            continue;
        };
        if !a1
            .facts
            .iter()
            .any(|f| matches!(f, RuntimeFact::CubicArgumentSampler { .. }))
        {
            continue;
        }
        let a4 = compile(&four, "k", &p).expect("four");
        let a5 = compile(&five, "k", &p).expect("five");
        let marks = count(&a1, OP_ATOMIC_OR);
        assert!(marks > 0, "{stem}");
        assert_eq!(count(&a4, OP_ATOMIC_OR), marks, "{stem}: one block");
        assert_eq!(count(&a5, OP_ATOMIC_OR), marks + 1, "{stem}: two blocks");
        compiled += 1;
    }
    assert!(
        compiled > 0,
        "no committed profile tests its argument samplers"
    );
}
