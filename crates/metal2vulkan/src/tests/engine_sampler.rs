//! Sampling, gathers, and LOD queries end to end.
//!
//! These are structural and L3 tests: every compiled module passes `spirv-val` under the profile
//! (`G-val`) and links through `pipeline::link`, and the tests check the combined bindings, the
//! strategies, the facts, the specialization constants, and the interface the lowering chose.
//! The CPU interpreter does not model sampling, so no texel value is checked here.
//!
//! - Every sampling entry of `engine/sampling` compiles and validates, or is refused with a
//!   typed error, on every committed profile.
//! - A combined binding per (texture, sampler, comparison); a constexpr sampler is the linked
//!   layout's immutable sampler.
//! - Level of detail per stage: implicit and tangled in a fragment function, the base level
//!   outside one.
//! - Pixel coordinates: scaled always for a constexpr pixel sampler, behind the runtime sampler's
//!   `SamplerCoordinates` specialization otherwise.
//! - The operand legality matrix: the refusals of unmodelled and unsupported forms.
//! - Texel offsets on gathers and implicit-LOD samples: the value Metal applies, its low four bits,
//!   as the `Offset` operand of a dynamic one, and the device's offset range containing Metal's.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::requirements::{VkFeature, VkLimit, VkProperty};
use crate::device::DevicePolicy;
use crate::device::{Device, DeviceDescription};
use crate::error::{Capability, Error, Feature, LowerFamily, Needs, Unsupported};
use crate::library::FunctionArtifact;
use crate::metal::sampler::SamplerWord;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::{NumberedSampler, SlotInterface};
use crate::pipeline::layout::BindingSource;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::sym::{ResourceKind, SamplerRef, SlotClass, SlotRef, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, profile, AssembleError,
};
use crate::spirv::testing::committed_profiles;

const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

/// The entries of `engine/sampling`.
const ENTRIES: [&str; 9] = [
    "sample_level",
    "sample_base",
    "sample_constexpr_array",
    "sample_compare_depth",
    "gather_int",
    "sample_grad",
    "sample_pixel",
    "sample_fragment",
    "sample_min_lod",
];

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("sampling.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected sampling: {why}"),
    }
}

fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let text = format!(
        "target datalayout = \"{DATALAYOUT}\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n{body}"
    );
    let dir = std::env::temp_dir().join(format!(
        "m2v-sampler-{}-{}-{name}",
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

/// The opcodes of a SPIR-V module, in order.
fn opcodes(words: &[u32]) -> Vec<u16> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = (words[i] >> 16) as usize;
        out.push((words[i] & 0xFFFF) as u16);
        i += n.max(1);
    }
    out
}

fn count(a: &FunctionArtifact, op: u16) -> usize {
    opcodes(&a.module.words)
        .iter()
        .filter(|o| **o == op)
        .count()
}

/// The `GLSL.std.450` instructions numbered `glsl` the module's `OpExtInst`s call.
fn glsl_count(a: &FunctionArtifact, glsl: u32) -> usize {
    const OP_EXT_INST: u32 = 12;
    let words = &a.module.words;
    let mut n = 0;
    let mut i = 5;
    while i < words.len() {
        let len = (words[i] >> 16) as usize;
        if words[i] & 0xFFFF == OP_EXT_INST && words.get(i + 4) == Some(&glsl) {
            n += 1;
        }
        i += len.max(1);
    }
    n
}

/// `GLSL.std.450 Floor`.
const GLSL_FLOOR: u32 = 8;

const OP_SELECT: u16 = 169;
const OP_FDIV: u16 = 136;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
const OP_IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
const OP_IMAGE_SAMPLE_DREF_EXPLICIT_LOD: u16 = 90;
const OP_IMAGE_GATHER: u16 = 96;
const OP_IMAGE_QUERY_LOD: u16 = 105;
const OP_FUNCTION_CALL: u16 = 57;

const OP_QUANTIZE_TO_F16: u16 = 116;
const OP_F_ORD_EQUAL: u16 = 180;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_F_CONVERT: u16 = 115;

/// `a`'s module rounds a `half` by one of the forms `lower::arith::narrow::round` takes, at least
/// once: each `OpQuantizeToF16` feeds an `OpFOrdEqual`, the test of the rounding that keeps half
/// subnormals where the quantize gives a zero, so none rounds a `half` alone, which flushes them;
/// or an `OpFConvert` of an `f32` to 16 bits widened back by another, the device's conversion
/// (moltenvk-m3), or the same pair on a profile whose float controls make it Metal's
/// (`narrow::controlled_modes`).
fn assert_half_rounding(a: &FunctionArtifact) {
    let w = &a.module.words;
    let mut quantized = Vec::new();
    let mut compared = Vec::new();
    // Float types by width (scalar, and vectors by their component's width), and each
    // `OpFConvert`'s (result type, result, operand).
    let mut widths = std::collections::BTreeMap::new();
    let mut converts = Vec::new();
    let mut i = 5;
    while i < w.len() {
        let n = ((w[i] >> 16) as usize).max(1);
        match (w[i] & 0xFFFF) as u16 {
            OP_QUANTIZE_TO_F16 => quantized.push(w[i + 2]),
            OP_F_ORD_EQUAL => compared.extend_from_slice(&w[i + 3..i + n]),
            OP_TYPE_FLOAT => {
                widths.insert(w[i + 1], w[i + 2]);
            }
            OP_TYPE_VECTOR => {
                if let Some(&width) = widths.get(&w[i + 2]) {
                    widths.insert(w[i + 1], width);
                }
            }
            OP_F_CONVERT => converts.push((w[i + 1], w[i + 2], w[i + 3])),
            _ => {}
        }
        i += n;
    }
    let narrowed: Vec<u32> = converts
        .iter()
        .filter(|(ty, _, _)| widths.get(ty) == Some(&16))
        .map(|&(_, result, _)| result)
        .collect();
    let converted = converts
        .iter()
        .filter(|(ty, _, operand)| widths.get(ty) == Some(&32) && narrowed.contains(operand))
        .count();
    assert!(quantized.len() + converted > 0, "no half rounding");
    for q in quantized {
        assert!(compared.contains(&q), "%{q} rounds a half alone");
    }
}

/// The fields of the `SamplerState` specialization the function declares for sampler slot `s`, if
/// it declares one.
fn sampler_fields(a: &FunctionArtifact, s: SlotRef) -> Option<u32> {
    a.interface.read_keys().into_iter().find_map(|k| match k {
        SpecKey::SamplerState { slot, fields } if slot == s => Some(fields),
        _ => None,
    })
}

fn uses(a: &FunctionArtifact, name: &str) -> u32 {
    a.strategies.entries.get(name).map_or(0, |u| u.uses)
}

fn slot(stage: Stage, class: SlotClass, index: u32) -> SlotRef {
    SlotRef {
        stage,
        class,
        index,
    }
}

/// The combined bindings a module declares: (texture index, sampler, compare, integer).
fn combined(a: &FunctionArtifact) -> Vec<(u32, SamplerRef, bool, bool)> {
    a.interface
        .descriptors
        .iter()
        .filter_map(|d| match d.slot.kind {
            ResourceKind::CombinedTextureSampler(c) => {
                Some((d.slot.index, c.sampler, c.compare, c.integer))
            }
            _ => None,
        })
        .collect()
}

/// The combined image-sampler uses of `a`'s bindings.
fn combined_kinds(a: &FunctionArtifact) -> Vec<crate::pipeline::sym::CombinedSampler> {
    a.interface
        .descriptors
        .iter()
        .filter_map(|d| match d.slot.kind {
            ResourceKind::CombinedTextureSampler(c) => Some(c),
            _ => None,
        })
        .collect()
}

/// Every sampling entry compiles and passes L3, or is refused with a typed error, on every
/// committed profile; the minimum-LOD clamp exactly where `shaderResourceMinLod` is.
#[test]
fn every_sampling_entry_compiles_or_refuses_on_every_profile() {
    let Some(bc) = fixture() else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        for entry in ENTRIES {
            match compile(&bc, entry, &p) {
                Ok(a) => {
                    link_and_validate(&a, &p)
                        .unwrap_or_else(|e| panic!("{entry} on {stem}: L3: {e}"));
                    assert!(
                        entry != "sample_min_lod" || p.fields().images.min_lod,
                        "{stem}"
                    );
                }
                Err(Error::Unsupported(u)) => {
                    if entry == "sample_min_lod" && !p.fields().images.min_lod {
                        let want = Unsupported::needs(
                            Feature::MinLodClamp,
                            Capability::Feature(VkFeature::ShaderResourceMinLod),
                        );
                        assert_eq!((u.feature, u.needs), (want.feature, want.needs), "{stem}");
                    } else {
                        assert!(
                            !matches!(u.needs, Needs::NotImplemented),
                            "{entry} on {stem}: {u}"
                        );
                    }
                }
                Err(e) => panic!("{entry} on {stem}: {e}"),
            }
        }
    }
}

/// A runtime sampler's coordinate mode is a specialization constant selecting the scaled
/// coordinate; its slot carries the coordinate and reduction facts and its interface.
#[test]
fn runtime_samplers_select_pixel_coordinates_by_specialization() {
    let Some(bc) = fixture() else {
        return;
    };
    let p = m3();
    let a = ok(&bc, "sample_level", &p);
    let s = slot(Stage::Kernel, SlotClass::Sampler, 0);
    let t = slot(Stage::Kernel, SlotClass::Texture, 0);
    assert!(a
        .facts
        .contains(&RuntimeFact::SamplerCoordinates { slot: s }));
    assert!(a.facts.contains(&RuntimeFact::SamplerReduction { slot: s }));
    // The coordinates bit alone: the device's sampler reads the scaled coordinate.
    assert_eq!(sampler_fields(&a, s), Some(SamplerWord::PIXEL_COORDINATES));
    assert!(a.precision.contains(&PrecisionNote::PixelSamplerScaling));
    assert_eq!(uses(&a, "sampler.sample"), 1);
    assert_eq!(combined(&a), vec![(0, SamplerRef::Slot(0), false, false)]);
    assert_eq!(count(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD), 1);
    assert!(count(&a, OP_SELECT) >= 1 && count(&a, OP_FDIV) >= 1);
    match a.interface.slots.get(&s) {
        Some(SlotInterface::Sampler(x)) => assert!(x.coordinate_sensitive && !x.compare),
        other => panic!("sampler slot {other:?}"),
    }
    match a.interface.slots.get(&t) {
        Some(SlotInterface::Texture(x)) => {
            assert!(x.access.sample && !x.access.read && !x.access.write)
        }
        other => panic!("texture slot {other:?}"),
    }
    // A sample-only texture has no plain texture binding.
    assert!(a
        .interface
        .descriptors
        .iter()
        .all(|d| d.slot.kind != ResourceKind::Texture));
}

/// A sample through a runtime sampler slot is the device's on every committed profile: the
/// function reads only the coordinates bit of the slot's word (`SamplerWord::PIXEL_COORDINATES`),
/// never its filter, address modes or levels, and the module requires the device's own sampler
/// (`VkProperty::MetalSampler` on MoltenVK, `VkProperty::DeviceSampling` elsewhere) and validates.
#[test]
fn a_runtime_samplers_sample_is_the_device_s() {
    let Some(bc) = fixture() else {
        return;
    };
    let s = slot(Stage::Kernel, SlotClass::Sampler, 0);
    let (mut metal, mut device) = (0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = ok(&bc, "sample_level", &p);
        assert_eq!(
            sampler_fields(&a, s),
            Some(SamplerWord::PIXEL_COORDINATES),
            "{stem}"
        );
        assert_eq!(uses(&a, "sampler.sample"), 1, "{stem}");
        let property = match p
            .fields()
            .enabled
            .properties
            .contains(&VkProperty::MetalSampler)
        {
            true => {
                metal += 1;
                VkProperty::MetalSampler
            }
            false => {
                device += 1;
                VkProperty::DeviceSampling
            }
        };
        assert!(a.requirements.properties.contains(&property), "{stem}");
    }
    assert!(metal > 0 && device > 0, "metal {metal}, device {device}");
}

/// A runtime sampler's sample at a level (a dynamic level, `1.5`, `1.0`, a kernel's dynamic bias)
/// reads none of the level fields of the slot's word on any committed profile: the device's
/// sampler takes the level, the `VkSampler` holding the sampler's clamp.
#[test]
fn a_runtime_samples_level_is_the_device_s() {
    let kernels = [
        ("dynamic_level", true, "%l"),
        ("half_level", true, "1.5"),
        ("whole_level", true, "1.0"),
        ("dynamic_bias", false, "%l"),
    ];
    let s = slot(Stage::Kernel, SlotClass::Sampler, 0);
    let levels = SamplerWord::LEVEL_MIN | SamplerWord::LEVEL_MAX | SamplerWord::MIP_FILTER;
    for (name, level, value) in kernels {
        let text = sample_kernel(
            "%s",
            "zeroinitializer",
            level,
            value,
            "%l = load float, ptr addrspace(1) %out, align 4",
            "",
            "",
        );
        let Some(bc) = assemble(name, &text) else {
            return;
        };
        for (stem, p) in committed_profiles().expect("profiles") {
            let a = ok(&bc, "k", &p);
            let fields = sampler_fields(&a, s).unwrap_or(0);
            assert_eq!(fields & levels, 0, "{name} {stem}");
        }
    }
}

/// A sample, a gather and a comparison through a runtime sampler slot read only the coordinates
/// bit of the slot's word (`SamplerWord::PIXEL_COORDINATES`), and hand the device's sampler the
/// coordinate scaled behind it, on every committed profile: each records `PixelSamplerScaling`,
/// held where the bound sampler states pixel coordinates (each clause tests that bit). A gather
/// emits no texel centre (no `Floor`). A sample and a comparison also read the slot's
/// `NearestPixelSampler` specialization, behind which they hand the device their texel's centre
/// (one `Floor`), so their note's clause also tests that it does not hold, and record
/// `DeviceDenormals` for that `Floor` held only where it holds.
#[test]
fn runtime_pixel_accesses_hand_the_device_the_scaled_coordinate() {
    use crate::pipeline::artifact::SpecializedNote;
    let s = slot(Stage::Kernel, SlotClass::Sampler, 0);
    let kernels = [
        ("pixel_sample", dynamic_offset_kernel(false), true),
        ("pixel_gather", dynamic_offset_kernel(true), false),
        ("pixel_compare", COMPARE_KERNEL.to_owned(), true),
    ];
    for (name, text, nearest) in kernels {
        let Some(bc) = assemble(name, &text) else {
            return;
        };
        for (stem, p) in committed_profiles().expect("profiles") {
            let what = format!("{name} on {stem}");
            let a = ok(&bc, "k", &p);
            assert_eq!(
                sampler_fields(&a, s),
                Some(SamplerWord::PIXEL_COORDINATES),
                "{what}"
            );
            assert!(a
                .facts
                .contains(&RuntimeFact::SamplerCoordinates { slot: s }));
            assert!(
                a.precision.contains(&PrecisionNote::PixelSamplerScaling),
                "{what}"
            );
            let key = SpecKey::SamplerState {
                slot: s,
                fields: SamplerWord::PIXEL_COORDINATES,
            };
            let held: Vec<&SpecializedNote> = a
                .specialized
                .iter()
                .filter(|n| n.note == PrecisionNote::PixelSamplerScaling)
                .collect();
            assert_eq!(held.len(), 1, "{what}: {held:?}");
            let pixel = |t: &crate::pipeline::SpecTest| {
                t.key == key && t.mask == SamplerWord::PIXEL_COORDINATES && !t.equal
            };
            let scaled = |t: &crate::pipeline::SpecTest| {
                t.key == SpecKey::NearestPixelSampler(s) && t.mask == 1 && t.bits == 0 && t.equal
            };
            assert!(
                held[0].when.iter().all(|c| c.all.iter().any(pixel)
                    && c.all.iter().any(scaled) == nearest
                    && c.all.iter().all(|t| pixel(t) || scaled(t))),
                "{what}: {held:?}"
            );
            // The centre's `floor` reads a subnormal by the device's float controls: noted only
            // under the specialization that runs it.
            let centre = |t: &crate::pipeline::SpecTest| {
                t.key == SpecKey::NearestPixelSampler(s) && t.mask == 1 && t.bits == 1 && t.equal
            };
            let denormals: Vec<&SpecializedNote> = a
                .specialized
                .iter()
                .filter(|n| n.note == PrecisionNote::DeviceDenormals)
                .collect();
            match nearest {
                true => assert!(
                    denormals.len() == 1
                        && denormals[0].when.iter().all(|c| c.all.iter().all(centre)),
                    "{what}: {denormals:?}"
                ),
                false => assert!(
                    !a.precision.contains(&PrecisionNote::DeviceDenormals),
                    "{what}"
                ),
            }
            assert_eq!(glsl_count(&a, GLSL_FLOOR), usize::from(nearest), "{what}");
            assert!(count(&a, OP_FDIV) >= 1, "{what}");
        }
    }
}

/// The notes a stage that samples through a runtime sampler slot or an argument buffer's sampler,
/// or a runtime cubic sampler's variant, records on the M3 Max under MoltenVK where its state is
/// normalized: the unconditional ones are these, as before a nearest pixel sample took its texel's
/// centre, and that centre's `DeviceDenormals` is held only where the slot's `NearestPixelSampler`
/// specialization is 1, which no normalized state's is (`Resources::nearest_pixel_sampler`), or,
/// for an integer texture's centre, where the slot's word states pixel coordinates. An argument
/// buffer's sampler records no `DeviceDenormals`: `ArgumentPixelSamplerScaling`, which the runtime
/// announces only while a pixel state lives, records its centre's reading.
#[test]
fn normalized_and_cubic_stages_record_no_note_their_states_cannot_run() {
    use crate::pipeline::sym::SpecKey;
    use crate::pipeline::testing::compile_with;
    use std::collections::BTreeSet;
    use PrecisionNote as N;
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let assembled = |name: &str| match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    };
    let s = slot(Stage::Kernel, SlotClass::Sampler, 0);
    // Each stage, the cubic variant's bound word where it is one, and its unconditional notes.
    let cubic_half = [N::BicubicFilterArithmetic, N::NativeHalfConversion];
    let stages: [(&str, &str, Option<u32>, &[PrecisionNote]); 14] = [
        ("sampling", "sample_level", None, &[]),
        ("sampling", "sample_base", None, &[]),
        ("sampling", "sample_grad", None, &[]),
        (
            "sampling",
            "sample_fragment",
            None,
            &[N::FastMathRewrites, N::CubeGradientFootprint],
        ),
        ("sampling", "sample_compare_depth", None, &[]),
        ("handles", "sampler_select", None, &[]),
        (
            "handles",
            "fragment_select_sample",
            None,
            &[N::GradFromBias],
        ),
        (
            "handles",
            "argument_buffer_runtime_sampler",
            None,
            &[N::BicubicFilterArithmetic],
        ),
        (
            "px_samplers",
            "px_sample_argument",
            None,
            &[N::ArgumentPixelSamplerScaling],
        ),
        (
            "gsm2_cubic_samplers",
            "gsm2_cubic_samples",
            None,
            &[N::BicubicFilterArithmetic],
        ),
        (
            "gsm2_cubic_samplers",
            "gsm2_cubic_half_samples",
            None,
            &cubic_half,
        ),
        (
            "gsm2_cubic_samplers",
            "gsm2_cubic_samples",
            Some(0),
            &[N::BicubicFilterArithmetic],
        ),
        (
            "gsm2_cubic_samplers",
            "gsm2_cubic_samples",
            Some(SamplerWord::PIXEL_COORDINATES),
            &[N::BicubicFilterArithmetic],
        ),
        (
            "gsm2_cubic_samplers",
            "gsm2_cubic_half_samples",
            Some(SamplerWord::PIXEL_COORDINATES),
            &cubic_half,
        ),
    ];
    for (fixture, entry, cubic, unconditional) in stages {
        let Some(bc) = assembled(fixture) else {
            return;
        };
        let what = format!("{fixture} {entry} (cubic word {cubic:?})");
        let mut inputs = crate::lower::select::LowerInputs::new();
        if let Some(word) = cubic {
            inputs.cubic_samplers.insert(s, word);
        }
        let a = compile_with(&bc, entry, &p, &inputs).unwrap_or_else(|e| panic!("{what}: {e}"));
        let held: BTreeSet<N> = a.specialized.iter().map(|n| n.note).collect();
        let always: BTreeSet<N> = a
            .precision
            .iter()
            .copied()
            .filter(|n| !held.contains(n))
            .collect();
        assert_eq!(
            always,
            unconditional.iter().copied().collect(),
            "{what}: {:?}",
            a.specialized
        );
        // The centre's note: under a nearest pixel state's specialization alone.
        let pixel_state = |t: &crate::pipeline::SpecTest| match t.key {
            SpecKey::NearestPixelSampler(_) => t.mask == 1 && t.bits == 1 && t.equal,
            SpecKey::SamplerState { .. } => {
                t.mask == SamplerWord::PIXEL_COORDINATES && t.bits == 0 && !t.equal
            }
            _ => false,
        };
        for n in a
            .specialized
            .iter()
            .filter(|n| n.note == N::DeviceDenormals)
        {
            assert!(
                !fixture.starts_with("px_") && n.when.iter().all(|c| c.all.iter().any(pixel_state)),
                "{what}: {n:?}"
            );
        }
    }
}

/// A kernel whose one comparison samples `depth2d<float, sample>` slot 0 through runtime sampler
/// slot 0 at a coordinate from its buffer.
const COMPARE_KERNEL: &str = r#"
define void @k(ptr addrspace(1) %d, ptr addrspace(2) %s, ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %gid) {
  %i = zext i32 %gid to i64
  %q = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i64 %i
  %c = load <2 x float>, ptr addrspace(1) %q, align 8
  %r = call { float, i8 } @air.sample_compare_depth_2d.f32(ptr addrspace(1) %d, ptr addrspace(2) %s, i32 1, <2 x float> %c, float 5.000000e-01, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)
  %v = extractvalue { float, i8 } %r, 0
  %p = getelementptr inbounds float, ptr addrspace(1) %out, i64 %i
  store float %v, ptr addrspace(1) %p, align 4
  ret void
}

declare { float, i8 } @air.sample_compare_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, float, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"depth2d<float, sample>", !"air.arg_name", !"d"}
!4 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"in"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!7 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
"#;

/// The words of a constexpr sampler addressing every axis to the edge, with the fields Apple's
/// compiler writes for the rest (compare `never`, LOD clamps 0 and `65504`), nearest filters.
const EDGE_WORDS: u64 = 0x007b_ff00_0008_0049;

/// A kernel comparing `depth2d<float, sample>` slot 0 against `0.5` at buffer 0's coordinate
/// through constexpr sampler `words`, storing the `float` result to buffer 1.
fn constexpr_compare_kernel(words: u64) -> String {
    format!(
        r#"@smp = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8

define void @k(ptr addrspace(1) %d, ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %gid) {{
  %i = zext i32 %gid to i64
  %q = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i64 %i
  %c = load <2 x float>, ptr addrspace(1) %q, align 8
  %r = call {{ float, i8 }} @air.sample_compare_depth_2d.f32(ptr addrspace(1) %d, ptr addrspace(2) @smp, i32 1, <2 x float> %c, float 5.000000e-01, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)
  %v = extractvalue {{ float, i8 }} %r, 0
  %p = getelementptr inbounds float, ptr addrspace(1) %out, i64 %i
  store float %v, ptr addrspace(1) %p, align 4
  ret void
}}

declare {{ float, i8 }} @air.sample_compare_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, float, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{{!0}}
!air.sampler_states = !{{!7}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"depth2d<float, sample>", !"air.arg_name", !"d"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"in"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}}
!6 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!7 = !{{!"air.sampler_state", ptr addrspace(2) @smp}}
"#
    )
}

/// A constexpr pixel sampler with a compare function has no Metal answer. Apple's compiler rejects
/// one ("invalid values combination in sampler initialization"), so no AIR from
/// Metal's toolchain carries it, and the compile layer refuses it
/// `NotRepresentable(PixelSamplerCompare)` on every committed profile, whatever the device has. The
/// same comparison through a normalized sampler compiles on `moltenvk-m3` and is not refused for
/// the pixel rule anywhere.
#[test]
fn a_constexpr_pixel_sampler_with_a_compare_function_is_refused() {
    let less = EDGE_WORDS & !(0xf << 16) | 1 << 16;
    let pixel = 1 << 15;
    let (Some(normalized), Some(pixels)) = (
        assemble("normalized_less", &constexpr_compare_kernel(less)),
        assemble("pixel_less", &constexpr_compare_kernel(less | pixel)),
    ) else {
        return;
    };
    ok(&normalized, "k", &m3());
    for (stem, p) in &committed_profiles().expect("profiles") {
        match compile(&pixels, "k", p) {
            Err(Error::Unsupported(u)) => assert_eq!(
                (u.feature, u.needs),
                (Feature::PixelSamplerCompare, Needs::NotRepresentable),
                "{stem}"
            ),
            other => panic!("{stem}: {:?}", other.map(|_| ())),
        }
        if let Err(Error::Unsupported(u)) = compile(&normalized, "k", p) {
            assert_ne!(u.feature, Feature::PixelSamplerCompare, "{stem}");
        }
    }
}

/// Plain Metal reads a constexpr sampler's minimum or maximum reduction as the weighted average on
/// Apple's GPU (an Apple M4 on macOS 27.0.1, whose `supportsSamplerReductionMode` answers no;
/// for linear, nearest and bicubic samplers), so the translation lowers one as the weighted-average
/// sampler: the same module, interface (its immutable sampler's words; the sampler a pipeline state
/// numbers stays the module's), facts, notes and requirements on every committed profile, with no
/// `samplerFilterMinmax`.
#[test]
fn a_constexpr_reduction_compiles_as_the_weighted_average() {
    let linear = EDGE_WORDS | 1 << 9 | 1 << 11;
    let bicubic = EDGE_WORDS | 2 << 9 | 2 << 11;
    for (name, words) in [
        ("nearest", EDGE_WORDS),
        ("linear", linear),
        ("bicubic", bicubic),
    ] {
        let kernel = |reduction: u64| {
            sample_kernel(
                "@smp",
                "zeroinitializer",
                true,
                "0.0",
                "",
                &format!(
                    "@smp = internal addrspace(2) constant [2 x i64] [i64 {}, i64 0], align 8",
                    words | reduction << 58
                ),
                "!air.sampler_states = !{!6}\n!6 = !{!\"air.sampler_state\", ptr addrspace(2) @smp}",
            )
        };
        let Some(average) = assemble(name, &kernel(0)) else {
            return;
        };
        for reduction in [1, 2] {
            let Some(bc) = assemble(name, &kernel(reduction)) else {
                return;
            };
            for (stem, p) in &committed_profiles().expect("profiles") {
                match (compile(&average, "k", p), compile(&bc, "k", p)) {
                    (Ok(a), Ok(b)) => {
                        let at = format!("{name} {reduction} on {stem}");
                        assert_eq!(a.module, b.module, "{at}");
                        // The sampler a pipeline state numbers is the module's state, its
                        // reduction kept: whether Metal's numbering tells two states apart by it
                        // alone is unmeasured, so the words are the module's.
                        let numbered = |w: u64| vec![NumberedSampler::Constexpr([w, 0])];
                        assert_eq!(a.interface.numbered_samplers, numbered(words), "{at}");
                        assert_eq!(
                            b.interface.numbered_samplers,
                            numbered(words | reduction << 58),
                            "{at}"
                        );
                        let mut b_interface = b.interface.clone();
                        b_interface.numbered_samplers = a.interface.numbered_samplers.clone();
                        assert_eq!(a.interface, b_interface, "{at}");
                        assert_eq!(a.facts, b.facts, "{at}");
                        assert_eq!(a.precision, b.precision, "{at}");
                        assert_eq!(a.requirements, b.requirements, "{at}");
                        assert!(
                            !a.requirements
                                .features
                                .contains(&VkFeature::SamplerFilterMinmax),
                            "{name} on {stem}"
                        );
                    }
                    (Err(a), Err(b)) => {
                        assert_eq!(a.to_string(), b.to_string(), "{name} {reduction} on {stem}")
                    }
                    (a, b) => panic!(
                        "{name} {reduction} on {stem}: {:?} {:?}",
                        a.map(|_| ()),
                        b.map(|_| ())
                    ),
                }
            }
        }
    }
}

/// Outside a fragment function an implicit-LOD sample reads the base level: an explicit sample, not
/// tangled, with no derivative mode.
#[test]
fn kernel_implicit_samples_read_the_base_level() {
    let Some(bc) = fixture() else {
        return;
    };
    let a = ok(&bc, "sample_base", &m3());
    assert_eq!(count(&a, OP_IMAGE_SAMPLE_IMPLICIT_LOD), 0);
    assert_eq!(count(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD), 1);
    assert_eq!(uses(&a, "sampler.sample_implicit"), 0);
}

/// A constexpr sampler is baked into the linked layout as the binding's immutable sampler; it takes
/// no sampler slot, fact, or specialization.
#[test]
fn constexpr_samplers_are_immutable_samplers_of_the_layout() {
    let Some(bc) = fixture() else {
        return;
    };
    let p = m3();
    let a = ok(&bc, "sample_constexpr_array", &p);
    let c = combined(&a);
    assert_eq!(c.len(), 1);
    assert!(matches!(c[0].1, SamplerRef::Constexpr(_)));
    assert!(a.facts.iter().all(|f| !matches!(
        f,
        RuntimeFact::SamplerCoordinates { .. } | RuntimeFact::SamplerReduction { .. }
    )));
    assert!(!a
        .interface
        .slots
        .keys()
        .any(|s| s.class == SlotClass::Sampler));
    let l = link(&[&a], &LinkInputs::compute(), &p).expect("link");
    let b = l.layout.sets[0]
        .bindings
        .iter()
        .find(|b| matches!(b.source, BindingSource::CombinedTextureSampler { .. }))
        .expect("combined binding");
    let baked = b.immutable_sampler.expect("immutable sampler");
    assert_eq!(baked.address_mode_u, crate::vk::SamplerAddressMode::REPEAT);
    assert_eq!(baked.min_filter, crate::vk::Filter::LINEAR);
    assert_eq!(uses(&a, "sampler.sample"), 1);
}

/// A constexpr sampler samples a slot of `LoweringInputs::address_maps` with its address modes as
/// the slot's map sends them, so the binding's immutable sampler holds the mapped modes; a slot
/// without a map keeps the sampler's own. Either way the function records `ConstexprSampled` for
/// the slot with the sampler's own modes, which is how the binder knows to put a texture whose map
/// changes them there in the variant. A map to a mode no constexpr sampler can state is a typed
/// refusal.
#[test]
fn a_constexpr_sampler_takes_its_slots_address_map() {
    use crate::metal::sampler::AddressModeMap;
    use crate::metal::SamplerAddressMode as M;
    use crate::pipeline::testing::compile_with;
    let Some(bc) = fixture() else {
        return;
    };
    let p = m3();
    let texture = slot(Stage::Kernel, SlotClass::Texture, 0);
    let under = |map: Option<AddressModeMap>| {
        let mut inputs = crate::lower::select::LowerInputs::new();
        inputs.address_maps.extend(map.map(|map| (texture, map)));
        compile_with(&bc, "sample_constexpr_array", &p, &inputs)
    };
    let baked = |map: Option<AddressModeMap>| {
        let a = under(map).expect("compile");
        link_and_validate(&a, &p).expect("L3");
        assert!(a.facts.contains(&RuntimeFact::ConstexprSampled {
            slot: texture,
            modes: [M::Repeat; 3],
        }));
        let l = link(&[&a], &LinkInputs::compute(), &p).expect("link");
        let b = l.layout.sets[0]
            .bindings
            .iter()
            .find(|b| matches!(b.source, BindingSource::CombinedTextureSampler { .. }))
            .expect("combined binding")
            .immutable_sampler
            .expect("immutable sampler");
        (b.address_mode_u, b.address_mode_v, b.address_mode_w)
    };
    let vk = crate::vk::SamplerAddressMode::REPEAT;
    assert_eq!(baked(None), (vk, vk, vk));
    let edges = AddressModeMap::IDENTITY.with(M::Repeat, M::ClampToEdge);
    let vk = crate::vk::SamplerAddressMode::CLAMP_TO_EDGE;
    assert_eq!(baked(Some(edges)), (vk, vk, vk));
    let mirrored = AddressModeMap::IDENTITY.with(M::Repeat, M::MirrorRepeat);
    let vk = crate::vk::SamplerAddressMode::MIRRORED_REPEAT;
    assert_eq!(baked(Some(mirrored)), (vk, vk, vk));
    // A map that leaves the sampler's repeat alone leaves its modes alone.
    let others = AddressModeMap::IDENTITY.with(M::ClampToZero, M::ClampToEdge);
    let vk = crate::vk::SamplerAddressMode::REPEAT;
    assert_eq!(baked(Some(others)), (vk, vk, vk));
    // MSL has no constexpr mirror-clamp-to-edge.
    let unstated = AddressModeMap::IDENTITY.with(M::Repeat, M::MirrorClampToEdge);
    match under(Some(unstated)) {
        Err(Error::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (Feature::SamplerAddressMode, Needs::NotImplemented)
        ),
        other => panic!("{:?}", other.map(|_| ())),
    }
}

/// One runtime sampler used with and without comparison is two combined bindings, the comparison
/// one on the depth texture; the interfaces record the comparison.
#[test]
fn comparison_splits_the_combined_binding() {
    let Some(bc) = fixture() else {
        return;
    };
    let a = ok(&bc, "sample_compare_depth", &m3());
    let mut c = combined(&a);
    c.sort();
    assert_eq!(
        c,
        vec![
            (0, SamplerRef::Slot(0), true, false),
            (1, SamplerRef::Slot(0), false, false)
        ]
    );
    assert_eq!(count(&a, OP_IMAGE_SAMPLE_DREF_EXPLICIT_LOD), 1);
    match a
        .interface
        .slots
        .get(&slot(Stage::Kernel, SlotClass::Sampler, 0))
    {
        Some(SlotInterface::Sampler(x)) => assert!(x.compare),
        other => panic!("{other:?}"),
    }
    match a
        .interface
        .slots
        .get(&slot(Stage::Kernel, SlotClass::Texture, 0))
    {
        Some(SlotInterface::Texture(x)) => assert!(x.depth_compare),
        other => panic!("{other:?}"),
    }
}

/// Gathers of integer textures, half texels by gradients, and a constexpr pixel sampler.
#[test]
fn gathers_gradients_and_pixel_constexpr_samplers() {
    let Some(bc) = fixture() else {
        return;
    };
    let p = m3();
    let g = ok(&bc, "gather_int", &p);
    assert_eq!(count(&g, OP_IMAGE_GATHER), 1);
    assert_eq!(uses(&g, "sampler.gather"), 1);
    assert!(combined(&g)[0].3, "an integer texture's binding");
    let h = ok(&bc, "sample_grad", &p);
    assert_eq!(uses(&h, "sampler.sample_grad"), 1);
    // The sampled `half` texel is rounded by the module's form: no bare quantize; its NaN is the
    // device's decode.
    assert_half_rounding(&h);
    assert!(h.precision.contains(&PrecisionNote::DeviceNanEncoding));
    // A linear pixel sample of a 2D texture, the device's unnormalized sampler's at the coordinate
    // as given: no division, no `PixelSamplerScaling`.
    let x = ok(&bc, "sample_pixel", &p);
    assert!(!x.precision.contains(&PrecisionNote::PixelSamplerScaling));
    assert_eq!(uses(&x, "sampler.pixel_unnormalized"), 1);
    assert_eq!(count(&x, OP_FDIV), 0);
    assert!(combined_kinds(&x).iter().any(|c| c.unnormalized));
    assert!(x
        .interface
        .read_keys()
        .into_iter()
        .all(|k| !matches!(k, SpecKey::SamplerState { .. })));
}

/// A `half` sample of a texture slot converts its texel by the bound format's class: the slot's
/// `TexelFormat` fact and specialization, a call of the module's conversion routine (whose words
/// the census counts once), and the `HalfTexelConversion` note. A `float` sample takes none of
/// them.
#[test]
fn half_samples_convert_by_the_bound_format() {
    let Some(bc) = fixture() else {
        return;
    };
    let p = m3();
    let t = slot(Stage::Kernel, SlotClass::Texture, 0);
    let h = ok(&bc, "sample_grad", &p);
    assert!(h.facts.contains(&RuntimeFact::TexelFormat { slot: t }));
    assert!(h.interface.reads_key(SpecKey::TexelFormat(t)));
    assert!(h.precision.contains(&PrecisionNote::HalfTexelConversion));
    assert_eq!(count(&h, OP_FUNCTION_CALL), 1);
    assert!(h.strategies.entries.contains_key("texture.half_texel"));
    let f = ok(&bc, "sample_level", &p);
    assert!(f
        .facts
        .iter()
        .all(|x| !matches!(x, RuntimeFact::TexelFormat { .. })));
    assert!(!f.precision.contains(&PrecisionNote::HalfTexelConversion));
    assert!(!f.strategies.entries.contains_key("texture.half_texel"));
}

/// In a fragment function implicit-LOD samples and LOD queries use the fragment's derivatives; a
/// cube samples through its own cube view.
#[test]
fn fragment_sampling_is_implicit_and_queries_lod() {
    let Some(bc) = fixture() else {
        return;
    };
    let a = ok(&bc, "sample_fragment", &m3());
    assert_eq!(uses(&a, "sampler.sample_implicit"), 2);
    assert_eq!(uses(&a, "sampler.query_lod"), 2);
    assert_eq!(count(&a, OP_IMAGE_SAMPLE_IMPLICIT_LOD), 2);
    assert_eq!(count(&a, OP_IMAGE_QUERY_LOD), 2);
    let c = combined(&a);
    // Texture 0 with the runtime sampler and with the constexpr one; the cube with the constexpr.
    assert_eq!(c.len(), 3, "{c:?}");
    assert!(!a
        .facts
        .iter()
        .any(|f| matches!(f, RuntimeFact::ImageViewShape { .. })));
}

/// A kernel sampling `texture2d<float, sample>` slot 0 into `float4 *out` (buffer 0) at an
/// explicit level (`level`) or with a bias, of `value`. `smp` is the sampler operand (`%s` is
/// `[[sampler(0)]]`); `prelude` runs first, and `globals` and `named` add module text.
fn sample_kernel(
    smp: &str,
    offset: &str,
    level: bool,
    value: &str,
    prelude: &str,
    globals: &str,
    named: &str,
) -> String {
    format!(
        r#"{globals}
define void @k(ptr addrspace(1) %t, ptr addrspace(2) %s, ptr addrspace(1) %out) {{
  {prelude}
  %r = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {smp}, <2 x float> zeroinitializer, i1 true, <2 x i32> {offset}, i1 {level}, float {value}, float 0.0, i32 0)
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  store <4 x float> %v, ptr addrspace(1) %out, align 16
  ret void
}}

declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare ptr addrspace(2) @air.get_read_sampler()

!air.kernel = !{{!0}}
{named}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
"#
    )
}

/// The feature and needs of the refusal of kernel `text` on `moltenvk-m3` (its location aside).
fn refusal(name: &str, text: &str) -> Option<(Feature, Needs)> {
    let bc = assemble(name, text)?;
    match compile(&bc, "k", &m3()) {
        Err(Error::Unsupported(u)) => Some((u.feature, u.needs)),
        other => panic!("{name}: {:?}", other.map(|_| ())),
    }
}

/// The operand legality matrix: typed refusals of the forms the sampler family does not lower or
/// the device cannot express.
#[test]
fn unsupported_sampling_forms_are_typed_refusals() {
    let not_impl = |f: Feature| (f, Needs::NotImplemented);
    // The baseline kernel compiles.
    let plain = |offset: &str, level: bool, value: &str| {
        sample_kernel("%s", offset, level, value, "", "", "")
    };
    let Some(bc) = assemble("baseline", &plain("zeroinitializer", true, "0.0")) else {
        return;
    };
    let p = m3();
    ok(&bc, "k", &p);
    // A nonzero bias outside a fragment function is an explicit level, not a refusal.
    if let Some(bc) = assemble("bias", &plain("zeroinitializer", false, "1.0")) {
        ok(&bc, "k", &p);
    }
    // A constant offset is the value Metal applies, its low four bits: `(64, 0)` is no offset, and
    // `(24, 0)` is `(-8, 0)`, beyond a `minTexelOffset` narrowed to -4.
    if let Some(bc) = assemble("offset_64", &plain("<i32 64, i32 0>", true, "0.0")) {
        let a = ok(&bc, "k", &p);
        assert_eq!(
            operand_masks(&a, OP_IMAGE_SAMPLE_EXPLICIT_LOD, 5),
            [MASK_LOD]
        );
    }
    if let Some(bc) = assemble("offset_24", &plain("<i32 24, i32 0>", true, "0.0")) {
        ok(&bc, "k", &p);
        let narrow = m3_with(|l| l.min_texel_offset = -4);
        match compile(&bc, "k", &narrow) {
            Err(Error::Unsupported(u)) => assert_eq!(
                (u.feature, u.needs),
                (
                    Feature::TexelOffset,
                    Needs::AnyOf(vec![vec![Capability::Limit {
                        limit: VkLimit::MinTexelOffset,
                        required: 8,
                        available: 4,
                    }]])
                )
            ),
            other => panic!("{:?}", other.map(|_| ())),
        }
    }
    // The read sampler cannot sample.
    let read = sample_kernel(
        "%rs",
        "zeroinitializer",
        true,
        "0.0",
        "%rs = call ptr addrspace(2) @air.get_read_sampler()",
        "",
        "",
    );
    assert_eq!(
        refusal("read_sampler", &read),
        Some(not_impl(Feature::Lowering(LowerFamily::Sampler)))
    );
    // A bicubic constexpr sampler that mirror-repeats: filter code 2 in both filter fields, address
    // code 3 on every axis, with a constant offset on the mirrored `s` axis, where Metal's
    // footprint is not measured (`lower::sampler::bicubic`).
    let bicubic = sample_kernel(
        "@smp",
        "<i32 1, i32 0>",
        true,
        "0.0",
        "",
        "@smp = internal addrspace(2) constant [2 x i64] [i64 5339, i64 0], align 8",
        "!air.sampler_states = !{!6}\n!6 = !{!\"air.sampler_state\", ptr addrspace(2) @smp}",
    );
    assert_eq!(
        refusal("bicubic", &bicubic),
        Some(not_impl(Feature::SamplerBicubicFilter))
    );
}

/// A kernel reading `texture2d<float, sample>` slot 0 through `[[sampler(0)]]` at the dynamic
/// offset `int2(int(gid & 7) - 4, 1)` into `float4 *out`: a gather of component 0 (`gather`) or
/// a sample at level 0.
fn dynamic_offset_kernel(gather: bool) -> String {
    let (call, decl) = if gather {
        (
            "@air.gather_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> zeroinitializer, i1 true, <2 x i32> %o, i32 0, i32 0)",
            "@air.gather_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)",
        )
    } else {
        (
            "@air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> zeroinitializer, i1 true, <2 x i32> %o, i1 true, float 0.0, float 0.0, i32 0)",
            "@air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)",
        )
    };
    format!(
        r#"
define void @k(ptr addrspace(1) %t, ptr addrspace(2) %s, ptr addrspace(1) %out, i32 %gid) {{
  %m = and i32 %gid, 7
  %x = add nsw i32 %m, -4
  %o = insertelement <2 x i32> <i32 poison, i32 1>, i32 %x, i64 0
  %r = call {{ <4 x float>, i8 }} {call}
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  %i = zext i32 %gid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}}

declare {{ <4 x float>, i8 }} {decl}

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!6 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
"#
    )
}

/// A fragment function returning an implicit-LOD sample of `texture2d<float, sample>` slot 0
/// through `[[sampler(0)]]` at the position, with a dynamic offset from the position's bits.
const DYNAMIC_OFFSET_FRAGMENT: &str = r#"
define <4 x float> @f(<4 x float> %pos, ptr addrspace(1) %t, ptr addrspace(2) %s) {
  %px = extractelement <4 x float> %pos, i64 0
  %xi = bitcast float %px to i32
  %m = and i32 %xi, 7
  %x = add nsw i32 %m, -4
  %o = insertelement <2 x i32> <i32 poison, i32 1>, i32 %x, i64 0
  %uv = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %r = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> %uv, i1 true, <2 x i32> %o, i1 false, float 0.0, float 0.0, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  ret <4 x float> %v
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.fragment = !{!0}
!0 = !{ptr @f, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5, !6}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!6 = !{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
"#;

/// The image-operand mask of every instruction `op`, the mask being operand word `at`.
fn operand_masks(a: &FunctionArtifact, op: u16, at: usize) -> Vec<u32> {
    let w = &a.module.words;
    let mut out = Vec::new();
    let mut i = 5;
    while i < w.len() {
        let n = (w[i] >> 16) as usize;
        if (w[i] & 0xFFFF) as u16 == op {
            out.push(w.get(i + at).copied().unwrap_or(0));
        }
        i += n.max(1);
    }
    out
}

/// The number of `OpBitFieldSExtract` instructions.
fn sign_extracts(a: &FunctionArtifact) -> usize {
    const OP_BIT_FIELD_S_EXTRACT: u32 = 202;
    let w = &a.module.words;
    let mut n = 0;
    let mut i = 5;
    while i < w.len() {
        let len = (w[i] >> 16) as usize;
        if w[i] & 0xFFFF == OP_BIT_FIELD_S_EXTRACT {
            n += 1;
        }
        i += len.max(1);
    }
    n
}

const MASK_LOD: u32 = 0x2;
const MASK_CONST_OFFSET: u32 = 0x8;
const MASK_OFFSET: u32 = 0x10;
const OP_SWITCH: u16 = 251;

/// A dynamic offset on a gather, and on a fragment's implicit-LOD sample, is the `Offset` operand
/// of the value Metal applies, its low four bits sign-extended by one `OpBitFieldSExtract`; only
/// the sample needs `maintenance8`. Without `shaderImageGatherExtended` (the floor) a dynamic gather
/// offset is taken as `ConstOffset` arms: one switch on its dynamic lane's low four bits, 16 gathers
/// each with a `ConstOffset` and none with `Offset`, and the module needs neither feature. A
/// runtime sampler's gather snaps a pixel word's coordinate behind the word and moves nothing by
/// the offset, which the device's sampler places, so it reads the offset once on every device
/// (lavapipe, MoltenVK).
#[test]
fn dynamic_offsets_are_offset_operands_of_their_low_four_bits() {
    let Some(bc) = assemble("gather_dynamic_offset", &dynamic_offset_kernel(true)) else {
        return;
    };
    let lavapipe = profile("lavapipe", DevicePolicy::default()).expect("profile");
    let a = ok(&bc, "k", &lavapipe);
    assert_eq!(operand_masks(&a, OP_IMAGE_GATHER, 6), [MASK_OFFSET]);
    assert_eq!(sign_extracts(&a), 1);
    let p = m3();
    let a = ok(&bc, "k", &p);
    assert_eq!(uses(&a, "sampler.gather"), 1);
    assert_eq!(operand_masks(&a, OP_IMAGE_GATHER, 6), [MASK_OFFSET]);
    assert_eq!(sign_extracts(&a), 1);
    assert!(a
        .requirements
        .features
        .contains(&VkFeature::ShaderImageGatherExtended));
    assert!(!a.requirements.features.contains(&VkFeature::Maintenance8));
    let floor = profile("vk12-floor", DevicePolicy::default()).expect("profile");
    let a = ok(&bc, "k", &floor);
    assert_eq!(count(&a, OP_SWITCH), 1);
    assert_eq!(
        operand_masks(&a, OP_IMAGE_GATHER, 6),
        [MASK_CONST_OFFSET; 16]
    );
    for f in [
        VkFeature::Maintenance8,
        VkFeature::ShaderImageGatherExtended,
    ] {
        assert!(!a.requirements.features.contains(&f), "{f:?}");
    }
    let Some(bc) = assemble("fragment_dynamic_offset", DYNAMIC_OFFSET_FRAGMENT) else {
        return;
    };
    let a = ok(&bc, "f", &p);
    assert_eq!(uses(&a, "sampler.sample_implicit"), 1);
    // The zero bias is dropped: `Offset` is the only operand.
    assert_eq!(
        operand_masks(&a, OP_IMAGE_SAMPLE_IMPLICIT_LOD, 5),
        [MASK_OFFSET]
    );
    assert_eq!(sign_extracts(&a), 1);
    assert!(a.requirements.features.contains(&VkFeature::Maintenance8));
}

/// `moltenvk-m3` with its texel offset limits changed by `edit`.
fn m3_with(edit: impl FnOnce(&mut crate::device::profile::Limits)) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    edit(&mut desc.limits);
    Device::new(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// A dynamic offset may take any value of Metal's range `[-8, 7]`, so the device's range (the
/// gather range for a gather) must contain it; a narrower one is refused naming the limit.
#[test]
fn dynamic_offsets_need_the_device_range_to_contain_metals() {
    let limit = |limit, required, available| {
        (
            Feature::TexelOffset,
            Needs::AnyOf(vec![vec![Capability::Limit {
                limit,
                required,
                available,
            }]]),
        )
    };
    let narrow_sample = m3_with(|l| {
        l.min_texel_offset = -4;
    });
    let narrow_gather = m3_with(|l| {
        l.max_texel_gather_offset = 3;
    });
    for (gather, p, want) in [
        (
            false,
            &narrow_sample,
            Some(limit(VkLimit::MinTexelOffset, 8, 4)),
        ),
        (false, &narrow_gather, None),
        (
            true,
            &narrow_gather,
            Some(limit(VkLimit::MaxTexelGatherOffset, 7, 3)),
        ),
        (true, &narrow_sample, None),
    ] {
        let Some(bc) = assemble("dynamic_offset_range", &dynamic_offset_kernel(gather)) else {
            return;
        };
        match (compile(&bc, "k", p), want) {
            (Ok(a), None) => {
                link_and_validate(&a, p).unwrap_or_else(|e| panic!("gather {gather}: L3: {e}"));
            }
            (Err(Error::Unsupported(u)), Some(want)) => {
                assert_eq!((u.feature, u.needs), want, "gather {gather}")
            }
            (other, want) => panic!("gather {gather}: {:?}, want {want:?}", other.map(|_| ())),
        }
    }
}
