//! Sampler and texture completion end to end.
//!
//! Structural and L3 tests over the owned fixture `engine/sampling_completion`: every compiled
//! module passes `spirv-val` under the profile (`G-val`) and links through `pipeline::link`. The
//! CPU interpreter does not model sampling, so no texel value is checked here; the pixel fetch's
//! arithmetic (floor, clamp to edge, layer clamp) is checked structurally.
//!
//! - A sampler selected among a runtime and a constexpr sampler is a switch at the use with one
//!   combined binding per candidate; in a fragment function its implicit-LOD sample takes the
//!   hoisted gradients.
//! - A nearest, clamp-to-edge constexpr pixel sampler samples by an exact fetch, which takes a
//!   dynamic offset; a switch whose arms all fetch hoists nothing.
//! - Outside a fragment function a bias and a minimum-LOD clamp select an explicit level; an LOD
//!   query there answers Metal's measured level: the clamped one is the runtime sampler's
//!   `SamplerMinLevel` specialization bounded by the texture's last level.
//! - A dynamic offset on a sample through any other sampler is the `Offset` operand, the value
//!   Metal applies, where the profile has `maintenance8` and `shaderImageGatherExtended`, and a
//!   typed refusal naming the missing ones elsewhere (replacing the refusal).
//!
//! - The owned fixture `engine/mip_level_sampling` reaches the base-level and implicit-level
//!   strategies its authored cases run over a mipmapped texture.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkFeature;
use crate::device::DevicePolicy;
use crate::library::FunctionArtifact;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::layout::BindingSource;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::sym::{ResourceKind, SamplerRef, SlotClass, SlotRef, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, profile, AssembleError,
};
use crate::spirv::testing::committed_profiles;

/// The entries of `engine/sampling_completion` that compile on every profile.
const LOWERED: [&str; 8] = [
    "sample_mixed_samplers",
    "fragment_mixed_samplers",
    "sample_pixel_nearest",
    "fragment_select_pixel",
    "sample_pixel_zero",
    "sample_bias_kernel",
    "vertex_bias",
    "lod_query_kernel",
];

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("sampling_completion.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => {
            panic!("llvm-as rejected sampling_completion: {why}")
        }
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

/// The instructions of a SPIR-V module: opcode and word offset.
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

/// The number of `OpExtInst` of GLSL.std.450 instruction `inst`.
fn glsl(a: &FunctionArtifact, inst: u32) -> usize {
    let w = &a.module.words;
    insts(w)
        .iter()
        .filter(|(o, at)| *o == OP_EXT_INST && w.get(at + 4) == Some(&inst))
        .count()
}

const OP_EXT_INST: u16 = 12;
const OP_FDIV: u16 = 136;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
const OP_IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
const OP_IMAGE_FETCH: u16 = 95;
const OP_IMAGE_QUERY_LEVELS: u16 = 106;
const OP_IMAGE_QUERY_LOD: u16 = 105;
const OP_IMAGE_GATHER: u16 = 96;
const OP_DPDX: u16 = 207;
const OP_DPDY: u16 = 208;
const OP_BIT_FIELD_S_EXTRACT: u16 = 202;
const OP_SWITCH: u16 = 251;
const GLSL_FLOOR: u32 = 8;
const GLSL_FMIN: u32 = 37;
const GLSL_FMAX: u32 = 40;
const GLSL_UMIN: u32 = 38;
const GLSL_FCLAMP: u32 = 43;
const GLSL_SCLAMP: u32 = 45;

fn uses(a: &FunctionArtifact, name: &str) -> u32 {
    a.strategies.entries.get(name).map_or(0, |u| u.uses)
}

/// The combined bindings a module declares: (texture index, sampler).
fn combined(a: &FunctionArtifact) -> Vec<(u32, SamplerRef)> {
    let mut out: Vec<_> = a
        .interface
        .descriptors
        .iter()
        .filter_map(|d| match d.slot.kind {
            ResourceKind::CombinedTextureSampler(c) => Some((d.slot.index, c.sampler)),
            _ => None,
        })
        .collect();
    out.sort();
    out
}

/// Every entry compiles and passes L3 on every committed profile, or is refused with its typed
/// error on every profile.
#[test]
fn every_entry_compiles_or_refuses_on_every_profile() {
    let Some(bc) = fixture() else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        for entry in LOWERED {
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{entry} on {stem}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{entry} on {stem}: L3: {e}"));
        }
        // `Offset` on a sample needs `maintenance8` and `shaderImageGatherExtended`. A profile
        // without either takes the dynamic offset as `ConstOffset` arms: one switch on its dynamic
        // lane's low four bits, and 16 samples, each with its level and a `ConstOffset` and none
        // with `Offset`; the module needs neither feature (a function past the arms' bound is
        // refused, `engine_offset_arms`).
        let armed = matches!(
            stem.as_str(),
            "lunarg-desktop-baseline-2023" | "roadmap-2022" | "swiftshader" | "vk12-floor"
        );
        let a = compile(&bc, "sample_dynamic_offset", &p)
            .unwrap_or_else(|e| panic!("sample_dynamic_offset on {stem}: {e}"));
        link_and_validate(&a, &p)
            .unwrap_or_else(|e| panic!("sample_dynamic_offset on {stem}: L3: {e}"));
        let masks = operand_masks(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD, 5);
        let needs = [
            VkFeature::Maintenance8,
            VkFeature::ShaderImageGatherExtended,
        ]
        .map(|f| a.requirements.features.contains(&f));
        if armed {
            assert_eq!(count(&a, OP_SWITCH), 1, "{stem}");
            assert_eq!(masks, [MASK_LOD | MASK_CONST_OFFSET; 16], "{stem}");
            assert_eq!(needs, [false; 2], "{stem}");
        } else {
            assert_eq!(count(&a, OP_SWITCH), 0, "{stem}");
            assert_eq!(masks, [MASK_LOD | MASK_OFFSET], "{stem}");
            assert_eq!(needs, [true; 2], "{stem}");
        }
    }
}

/// A clamped LOD query in a kernel answers Metal's measured level without asking the device: the
/// runtime sampler's `SamplerMinLevel` specialization, in sixty-fourths, bounded by the texture's
/// last level (`OpImageQueryLevels` less 1). It reads neither the sampler's word nor a derivative,
/// and `OpImageQueryLod` is not made.
#[test]
fn a_kernels_clamped_lod_query_reads_the_samplers_minimum_level() {
    let Some(bc) = fixture() else {
        return;
    };
    let a = ok(&bc, "lod_query_kernel", &m3());
    assert_eq!(uses(&a, "sampler.query_level"), 1);
    assert_eq!(count(&a, OP_IMAGE_QUERY_LOD), 0);
    assert_eq!(count(&a, OP_IMAGE_QUERY_LEVELS), 1);
    assert_eq!(glsl(&a, GLSL_FMIN), 1);
    let s = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Sampler,
        index: 0,
    };
    let keys = a.interface.read_keys();
    assert!(keys.contains(&SpecKey::SamplerMinLevel(s)), "{keys:?}");
    assert!(
        !keys
            .iter()
            .any(|k| matches!(k, SpecKey::SamplerState { .. })),
        "{keys:?}"
    );
}

/// A sampler selected between a runtime and a constexpr sampler is a switch at the use, one arm and
/// one combined binding per sampler; the runtime arm keeps its coordinate specialization and the
/// constexpr binding is the layout's immutable sampler.
#[test]
fn a_sampler_selected_among_runtime_and_constexpr_switches() {
    let Some(bc) = fixture() else {
        return;
    };
    let p = m3();
    let a = ok(&bc, "sample_mixed_samplers", &p);
    assert_eq!(uses(&a, "sampler.sample.switch"), 1);
    assert_eq!(count(&a, OP_SWITCH), 1);
    assert_eq!(count(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD), 2);
    let c = combined(&a);
    assert_eq!(c.len(), 2, "{c:?}");
    assert!(c.contains(&(0, SamplerRef::Slot(0))), "{c:?}");
    assert!(
        c.iter()
            .any(|(t, s)| *t == 0 && matches!(s, SamplerRef::Constexpr(_))),
        "{c:?}"
    );
    let s = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Sampler,
        index: 0,
    };
    assert!(a
        .facts
        .contains(&RuntimeFact::SamplerCoordinates { slot: s }));
    let l = link(&[&a], &LinkInputs::compute(), &p).expect("link");
    let immutable = l.layout.sets[0]
        .bindings
        .iter()
        .filter(|b| matches!(b.source, BindingSource::CombinedTextureSampler { .. }))
        .filter(|b| b.immutable_sampler.is_some())
        .count();
    assert_eq!(immutable, 1);
}

/// In a fragment function the implicit-LOD sample through the selected sampler takes gradients
/// hoisted before the switch (one `DPdx`/`DPdy` pair, no implicit sample), and the gather through
/// it is an ordinary switch.
#[test]
fn a_selected_sampler_in_a_fragment_hoists_its_gradients() {
    let Some(bc) = fixture() else {
        return;
    };
    let a = ok(&bc, "fragment_mixed_samplers", &m3());
    assert_eq!(uses(&a, "sampler.sample_hoisted_grad.switch"), 1);
    assert_eq!(uses(&a, "sampler.sample.switch"), 1);
    assert_eq!(count(&a, OP_IMAGE_SAMPLE_IMPLICIT_LOD), 0);
    assert_eq!(count(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD), 2);
    assert_eq!(count(&a, OP_IMAGE_GATHER), 2);
    assert_eq!((count(&a, OP_DPDX), count(&a, OP_DPDY)), (1, 1));
    assert!(a.precision.contains(&PrecisionNote::GradFromBias));
}

/// A nearest, clamp-to-edge pixel sampler samples by an exact fetch: the floor of the coordinate
/// plus the offset (dynamic here), clamped to the level-0 size, and the layer clamped into the
/// array; no scaling and no scaling note.
#[test]
fn nearest_pixel_samplers_fetch_exactly() {
    let Some(bc) = fixture() else {
        return;
    };
    let a = ok(&bc, "sample_pixel_nearest", &m3());
    assert_eq!(uses(&a, "sampler.pixel_fetch"), 2);
    assert_eq!(count(&a, OP_IMAGE_FETCH), 2);
    assert_eq!(count(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD), 0);
    assert_eq!(count(&a, OP_IMAGE_SAMPLE_IMPLICIT_LOD), 0);
    assert_eq!(count(&a, OP_FDIV), 0);
    assert_eq!(glsl(&a, GLSL_FLOOR), 2);
    assert_eq!(glsl(&a, GLSL_SCLAMP), 2);
    assert_eq!(glsl(&a, GLSL_UMIN), 1, "the array layer's clamp");
    assert!(!a.precision.contains(&PrecisionNote::PixelSamplerScaling));
    // The fetch reads the combined binding's image: the constexpr sampler stays its symbol.
    assert!(combined(&a)
        .iter()
        .all(|(_, s)| matches!(s, SamplerRef::Constexpr(_))));
}

/// A switch whose arms all fetch holds nothing tangled, so no gradient is hoisted.
#[test]
fn a_switch_of_pixel_fetches_hoists_nothing() {
    let Some(bc) = fixture() else {
        return;
    };
    let a = ok(&bc, "fragment_select_pixel", &m3());
    assert_eq!(uses(&a, "sampler.sample.switch"), 1);
    assert_eq!(count(&a, OP_IMAGE_FETCH), 2);
    assert_eq!((count(&a, OP_DPDX), count(&a, OP_DPDY)), (0, 0));
    assert!(!a.precision.contains(&PrecisionNote::GradFromBias));
}

/// A nearest pixel sampler that clamps to zero keeps the sample, because its out-of-range texel
/// depends on the bound pixel format, and hands the device's unnormalized sampler the coordinate as
/// given (no division, no floor, no texel centre): the device's own nearest filter names the
/// texel from it, with no `PixelSamplerScaling`.
#[test]
fn a_clamp_to_zero_pixel_sampler_samples_its_coordinate_unnormalized() {
    let Some(bc) = fixture() else {
        return;
    };
    let a = ok(&bc, "sample_pixel_zero", &m3());
    assert_eq!(count(&a, OP_IMAGE_FETCH), 0);
    assert_eq!(uses(&a, "sampler.pixel_unnormalized"), 1);
    assert_eq!(count(&a, OP_FDIV), 0);
    assert_eq!(glsl(&a, GLSL_FLOOR), 0);
    assert!(!a.precision.contains(&PrecisionNote::PixelSamplerScaling));
}

/// The image-operand masks of every instruction `op` of the module, the mask being operand word
/// `at` of the instruction.
fn operand_masks(a: &FunctionArtifact, op: u16, at: usize) -> Vec<u32> {
    let w = &a.module.words;
    insts(w)
        .iter()
        .filter(|(o, _)| *o == op)
        .map(|(_, i)| w.get(i + at).copied().unwrap_or(0))
        .collect()
}

const MASK_LOD: u32 = 0x2;
const MASK_CONST_OFFSET: u32 = 0x8;
const MASK_OFFSET: u32 = 0x10;

/// A dynamic offset on a sample through a runtime sampler is the `Offset` operand beside its
/// explicit level, the value Metal applies (its low four bits sign-extended by one
/// `OpBitFieldSExtract`, in `[-8, 7]`); the module needs `maintenance8` and
/// `shaderImageGatherExtended`, which the profile's check admitted.
#[test]
fn a_dynamic_sample_offset_is_an_offset_operand_of_its_low_four_bits() {
    let Some(bc) = fixture() else {
        return;
    };
    let a = ok(&bc, "sample_dynamic_offset", &m3());
    assert_eq!(uses(&a, "sampler.sample"), 1);
    assert_eq!(
        operand_masks(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD, 5),
        [MASK_LOD | MASK_OFFSET]
    );
    assert_eq!(glsl(&a, GLSL_SCLAMP), 0);
    assert_eq!(count(&a, OP_BIT_FIELD_S_EXTRACT), 1);
    for f in [
        VkFeature::Maintenance8,
        VkFeature::ShaderImageGatherExtended,
    ] {
        assert!(a.requirements.features.contains(&f), "{f:?}");
    }
}

/// Outside a fragment function a bias and a minimum-LOD clamp select an explicit level: the whole
/// bias (Metal does not clamp it to `maxSamplerLodBias`), bounded below by the clamp; no `MinLod`
/// operand, so the kernel needs no `shaderResourceMinLod` (it compiles on every profile, above).
#[test]
fn bias_and_clamp_outside_a_fragment_are_explicit_levels() {
    let Some(bc) = fixture() else {
        return;
    };
    let a = ok(&bc, "sample_bias_kernel", &m3());
    assert_eq!(uses(&a, "sampler.sample_base_level"), 3);
    assert_eq!(count(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD), 3);
    assert_eq!(count(&a, OP_IMAGE_SAMPLE_IMPLICIT_LOD), 0);
    // Two biases (one dynamic, one constant), unclamped, and two minimum-LOD clamps.
    assert_eq!(glsl(&a, GLSL_FCLAMP), 0);
    assert_eq!(glsl(&a, GLSL_FMAX), 2);
    let v = ok(&bc, "vertex_bias", &m3());
    assert_eq!(uses(&v, "sampler.sample_base_level"), 1);
    assert_eq!(count(&v, OP_IMAGE_SAMPLE_IMPLICIT_LOD), 0);
}

/// The owned fixture `engine/mip_level_sampling` reaches the two strategies its authored cases
/// run over a mipmapped texture: a kernel's sample with a bias and no level is an explicit level
/// (`sampler.sample_base_level`), and a fragment function's sample with neither is the device's
/// implicit level (`sampler.sample_implicit`). On a profile that filters linearly in the shader the
/// kernel's sample is `sampler.linear` instead, so the names are the M3's; the inventory covers
/// every profile's compile.
#[test]
fn the_mip_fixture_reaches_the_base_level_and_implicit_strategies() {
    let bc = match llvm_assemble(&engine_dir().join("mip_level_sampling.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected mip_level_sampling: {why}"),
    };
    let p = m3();
    let names = |a: &FunctionArtifact| a.strategies.entries.keys().cloned().collect::<Vec<_>>();
    let k = ok(&bc, "mip_bias_kernel", &p);
    assert_eq!(uses(&k, "sampler.sample_base_level"), 1, "{:?}", names(&k));
    assert_eq!(count(&k, OP_IMAGE_SAMPLE_EXPLICIT_LOD), 1);
    assert_eq!(count(&k, OP_IMAGE_SAMPLE_IMPLICIT_LOD), 0);
    let f = ok(&bc, "mip_implicit_fragment", &p);
    assert_eq!(uses(&f, "sampler.sample_implicit"), 1, "{:?}", names(&f));
    assert_eq!(count(&f, OP_IMAGE_SAMPLE_IMPLICIT_LOD), 1);
}
