//! Fragment semantics end to end.
//!
//! The owned fixtures `engine/fragment_discard` (MSL 4.0: demote semantics) and
//! `engine/fragment_discard_msl22` (MSL 2.2: terminate semantics) go through the whole
//! per-function pipeline on every committed profile. Each discard site takes the form of the
//! matrix, the module contains exactly that form's instructions, and it passes L3. The owned
//! fixtures `engine/fragment_rog` and `fragment_rog_msl22` cover raster order groups: the body
//! runs as a function called between `Begin/EndInvocationInterlockEXT`, only the entry kills, and
//! the body's loops stay in it. `engine/fragment_fetch` covers the ordered forms of framebuffer
//! fetch: input attachments, and storage images inside the pixel interlock, with the binder's
//! writes. No GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::path::Path;

use crate::bind::{BindState, WriteSource};
use crate::budget::Budget;
use crate::device::requirements::VkFeature;
use crate::device::subgroup::OpBits;
use crate::device::{Device, DeviceDescription};
use crate::device::{DevicePolicy, DeviceProfile};
use crate::error::{BindError, Capability, Error, Feature, LowerFamily, Needs, RuntimeInput};
use crate::library::Library;
use crate::lower::fragment::DiscardForm;
use crate::metal::PixelFormat;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::layout::BindingSource;
use crate::pipeline::lower_inputs::{FramebufferFetchInputs, LoweringInputs};
use crate::pipeline::sym::ResourceKind;
use crate::pipeline::testing::{
    compile, compile_with, engine_dir, interlocked_coherence, link_and_validate, llvm_assemble,
    moltenvk_msl_compile, profile, typed_module, typed_module_with, AssembleError, Scratch,
};
use crate::spirv::ids::FuncId;
use crate::spirv::ir::{ExecMode, Inst, Terminator, TypedModule};
use crate::spirv::testing::{check_words_valid, committed_profiles, is_moltenvk_capture};
use crate::vk;

fn assemble(path: &Path) -> Option<Vec<u8>> {
    match llvm_assemble(path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {}: {why}", path.display()),
    }
}

fn fixture(name: &str) -> Option<Vec<u8>> {
    assemble(&engine_dir().join(format!("{name}.ll")))
}

/// What a module's discard lowering left behind.
#[derive(Debug, Default, PartialEq, Eq)]
struct Kills {
    demotes: usize,
    kills: usize,
    terminates: usize,
}

/// Compile `entry` on `p`, check its discard sites all took `form` and that the module holds
/// exactly that form's instructions, and run L3 on the compiled artifact.
fn check_form(bc: &[u8], entry: &str, stem: &str, p: &DeviceProfile, form: DiscardForm) {
    let (plan, module) =
        typed_module(bc, entry, p).unwrap_or_else(|e| panic!("{entry} on {stem}: {e}"));
    let fp = &plan.families.fragment;
    assert!(!fp.discards.is_empty(), "{entry}: no discard site");
    for (site, f) in &fp.discards {
        assert_eq!(*f, form, "{entry} on {stem}: site {site:?}");
    }
    let mut k = Kills::default();
    for b in module.blocks() {
        k.demotes += b
            .insts
            .iter()
            .filter(|n| matches!(n.inst, Inst::DemoteToHelperInvocation))
            .count();
        match b.term {
            Terminator::Kill => k.kills += 1,
            Terminator::TerminateInvocation => k.terminates += 1,
            _ => {}
        }
    }
    // One kill per site or at the entry's single return, and one per whole-quad exit, which a
    // demoting site's function places too.
    let killed = |n| {
        if fp.terminate_invocation {
            Kills {
                terminates: n,
                ..Kills::default()
            }
        } else {
            Kills {
                kills: n,
                ..Kills::default()
            }
        }
    };
    let exits = fp.quad_exits.len();
    let expected = match form {
        DiscardForm::Demote => Kills {
            demotes: 1,
            ..killed(exits)
        },
        DiscardForm::Terminate | DiscardForm::Predicate => killed(1 + exits),
    };
    assert_eq!(k, expected, "{entry} on {stem} ({form:?})");
    let a = compile(bc, entry, p).unwrap_or_else(|e| panic!("{entry} on {stem}: {e}"));
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{entry} on {stem}: L3: {e}"));
}

fn refusal(bc: &[u8], entry: &str, p: &DeviceProfile) -> (Feature, Needs) {
    match compile(bc, entry, p) {
        Err(Error::Unsupported(u)) => (u.feature, u.needs),
        other => panic!("{entry}: expected a refusal, got {:?}", other.map(|_| ())),
    }
}

/// MSL ≥ 2.3 demotes (or predicates without demote); MSL < 2.3 terminates when nothing tangled
/// follows, and otherwise takes MSL 2.3's form, a loop or a simdgroup operation after the site
/// included, as an M4 runs a discarded MSL 2.2 thread as an MSL 2.3 one. A site before a simdgroup
/// operation is predicated with a whole-quad exit before it where the profile has fragment quad
/// operations; elsewhere it demotes, or is the device gap without demote.
#[test]
fn discard_forms_follow_the_d6_matrix() {
    let (Some(new), Some(old)) = (
        fixture("fragment_discard"),
        fixture("fragment_discard_msl22"),
    ) else {
        return;
    };
    let profiles = committed_profiles().expect("profiles");
    assert!(!profiles.is_empty());
    for (stem, p) in &profiles {
        let helper = if p.fields().fragment.demote_to_helper {
            DiscardForm::Demote
        } else {
            DiscardForm::Predicate
        };
        check_form(&new, "discard_demote", stem, p, helper);
        check_form(&old, "discard_terminate", stem, p, DiscardForm::Terminate);
        check_form(&old, "discard_derivative", stem, p, helper);
        check_form(&old, "discard_loop", stem, p, helper);
        simd_after_discard(&old, "discard_simd", stem, p);
    }
}

/// Whether profile `p` runs quad operations in the fragment stage, which a whole-quad exit
/// needs.
fn fragment_quads(p: &DeviceProfile) -> bool {
    let f = &p.fields().subgroup.fragment;
    f.supported && f.ops.contains(OpBits::QUAD)
}

/// Check `entry`, whose discard a simdgroup operation follows, on profile `p`: predicated with one
/// whole-quad exit where the profile has fragment quad operations, else demoted, else refused
/// needing demote.
///
/// Where the discard form allows it but the profile lacks `shaderQuadControl` and its driver does
/// not run Metal's helpers (`DriverDeviation::MetalFullQuads`), the simdgroup operation is refused
/// needing it: it counts the helpers Metal runs, which only full quads give.
fn simd_after_discard(bc: &[u8], entry: &str, stem: &str, p: &DeviceProfile) {
    let formed = fragment_quads(p) || p.fields().fragment.demote_to_helper;
    let full_quads = p.fields().subgroup.quad_control
        || p.takes_driver_deviation(crate::error::DriverDeviation::MetalFullQuads);
    if formed && !full_quads {
        assert_eq!(
            refusal(bc, entry, p),
            (
                Feature::SubgroupOperations,
                Needs::AnyOf(vec![vec![Capability::Feature(
                    VkFeature::ShaderQuadControl
                )]])
            ),
            "{entry} on {stem}"
        );
    } else if fragment_quads(p) {
        check_form(bc, entry, stem, p, DiscardForm::Predicate);
        let (plan, _) = typed_module(bc, entry, p).expect("plan");
        assert_eq!(
            plan.families.fragment.quad_exits.len(),
            1,
            "{entry} on {stem}"
        );
    } else if p.fields().fragment.demote_to_helper {
        check_form(bc, entry, stem, p, DiscardForm::Demote);
    } else {
        assert_eq!(
            refusal(bc, entry, p),
            (
                Feature::Lowering(LowerFamily::Fragment),
                Needs::AnyOf(vec![vec![Capability::Feature(
                    VkFeature::ShaderDemoteToHelperInvocation
                )]])
            ),
            "{entry} on {stem}"
        );
    }
}

/// The `if !discarded` selections of `m`: conditional branches on the negation of a load of a
/// `Private` variable, as the guard of a side effect a predicated site reaches is written.
fn discard_guards(m: &TypedModule) -> usize {
    use crate::spirv::ir::{BinOp, StorageClass, UnOp, ValueOrigin};
    let defs: std::collections::HashMap<_, _> = m
        .blocks()
        .iter()
        .flat_map(|b| &b.insts)
        .filter_map(|n| n.result.map(|r| (r, &n.inst)))
        .collect();
    let private = |v: crate::spirv::ids::Value| {
        matches!(
            m.values().get(v.index()).map(|d| d.origin),
            Some(ValueOrigin::Global(g))
                if m.globals().get(g.index()).is_some_and(|g| g.class == StorageClass::Private)
        )
    };
    let loaded = |v| matches!(defs.get(&v), Some(Inst::Load { ptr, .. }) if private(*ptr));
    let not_discarded =
        |v| matches!(defs.get(&v), Some(Inst::Unary { op: UnOp::LogicalNot, a }) if loaded(*a));
    // Or the predicate and'ed with "not a helper" (`spirv::helper_branches`), which a guard takes
    // on MoltenVK where its arm calls a routine SPIRV-Cross writes `optnone` code in.
    m.blocks()
        .iter()
        .filter(|b| {
            matches!(&b.term, Terminator::CondBranch { cond, .. }
                if not_discarded(*cond)
                    || matches!(defs.get(cond), Some(Inst::Binary { op: BinOp::LogicalAnd, a, .. })
                        if not_discarded(*a)))
        })
        .count()
}

/// A side effect after a demoted discard needs no guard (a helper invocation has none); after a
/// predicated one it runs only where the invocation was not discarded, under `if !discarded`.
/// `discard_then_store` demotes where the profile has demote, and is predicated with its store
/// guarded where it does not (`lunarg-desktop-baseline-2023`). Every module passes L3.
#[test]
fn predication_guards_the_side_effects_after_a_site() {
    let Some(bc) = fixture("fragment_discard") else {
        return;
    };
    let mut predicated = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let f = p.fields();
        if !f.leaf.fragment_stores_and_atomics {
            continue;
        }
        let what = format!("discard_then_store on {stem}");
        let (form, guards) = if f.fragment.demote_to_helper {
            (DiscardForm::Demote, 0)
        } else {
            predicated += 1;
            (DiscardForm::Predicate, 1)
        };
        check_form(&bc, "discard_then_store", &stem, &p, form);
        let (plan, m) =
            typed_module(&bc, "discard_then_store", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_eq!(plan.families.fragment.guarded.len(), guards, "{what}");
        assert_eq!(discard_guards(&m), guards, "{what}");
        let a = compile(&bc, "discard_then_store", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
    }
    assert!(predicated > 0, "no committed profile predicates");
}

/// The owned fixture `rch_guarded_effects`'s side effects after a predicated discard, each under
/// `if !discarded`: a store, an atomic and the store its old value indexes (two guards; the
/// atomic's result is merged with zero for a discarded invocation), a store in a loop, a texture
/// write, a copy. They are predicated where the profile has no demote
/// (`lunarg-desktop-baseline-2023`), and on MoltenVK on an Apple GPU, whose store-dropping defect
/// defers each discard past the stores: no longer stopped by the side effects, recorded
/// `fragment.discard.deferred` and without `PrecisionNote::DiscardDropsSimdgroupStores`.
/// `rch_guarded_simd`'s simdgroup sum still stops the deferral there (the sites demote with
/// whole-quad exits and record the note); where the profile has no demote it is refused needing
/// demote, as no whole-quad exit can be placed on a profile without fragment quad operations. A
/// device without the defect that demotes (`intel-arl-anv`) demotes each and guards nothing. Every
/// module passes L3.
#[test]
fn a_predicated_discard_guards_each_later_store_atomic_and_image_write() {
    use crate::spirv::ir::ValueOrigin;
    let Some(bc) = fixture("rch_guarded_effects") else {
        return;
    };
    let noted = PrecisionNote::DiscardDropsSimdgroupStores;
    let deferred = "fragment.discard.deferred";
    // (entry, its side effects after the discard)
    let cases = [
        ("rch_guarded_store", 1),
        ("rch_guarded_atomic", 2),
        ("rch_guarded_loop", 1),
        ("rch_guarded_write", 1),
        ("rch_guarded_copy", 1),
        ("rch_guarded_simd", 1),
    ];
    for stem in [
        "moltenvk-m3",
        "lunarg-desktop-baseline-2023",
        "intel-arl-anv",
    ] {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        for (entry, effects) in cases {
            let what = format!("{entry} on {stem}");
            let mvk = stem == "moltenvk-m3";
            let demotes = p.fields().fragment.demote_to_helper;
            let simd = entry == "rch_guarded_simd";
            if simd && !demotes && !fragment_quads(&p) {
                // No whole-quad exit can drop a quad of discarded invocations before the sum,
                // and no demote keeps them helpers.
                assert_eq!(
                    refusal(&bc, entry, &p),
                    (
                        Feature::Lowering(LowerFamily::Fragment),
                        Needs::AnyOf(vec![vec![Capability::Feature(
                            VkFeature::ShaderDemoteToHelperInvocation
                        )]])
                    ),
                    "{what}"
                );
                continue;
            }
            // Predicated where the profile has no demote, or where the defect defers a discard
            // no non-quad subgroup operation follows.
            let predicated = !demotes || (mvk && !simd);
            let (plan, m) = typed_module(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let fp = &plan.families.fragment;
            let form = if predicated {
                DiscardForm::Predicate
            } else {
                DiscardForm::Demote
            };
            assert!(!fp.discards.is_empty(), "{what}");
            assert!(fp.discards.values().all(|f| *f == form), "{what}: {fp:?}");
            let guards = if predicated { effects } else { 0 };
            assert_eq!(fp.guarded.len(), guards, "{what}: {:?}", fp.guarded);
            assert_eq!(discard_guards(&m), guards, "{what}");
            if predicated && entry == "rch_guarded_atomic" {
                // The atomic's old value is merged with a constant for a discarded invocation.
                let zero_merged = m.blocks().iter().flat_map(|b| &b.phis).any(|phi| {
                    phi.incoming.iter().any(|(v, _)| {
                        matches!(
                            m.values().get(v.index()).map(|d| d.origin),
                            Some(ValueOrigin::Const(_))
                        )
                    })
                });
                assert!(zero_merged, "{what}: no merge of the atomic's result");
            }
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(
                a.strategies.entries.contains_key(deferred),
                mvk && predicated,
                "{what}: {:?}",
                a.strategies.entries.keys().collect::<Vec<_>>()
            );
            assert_eq!(
                a.precision.contains(&noted),
                mvk && simd,
                "{what}: the note"
            );
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        }
    }
}

/// A module that does not state its language version cannot say which discard semantics apply, so a
/// discard there is refused.
#[test]
fn a_discard_without_a_language_version_is_refused() {
    let src = engine_dir().join("fragment_discard_msl22.ll");
    let text = std::fs::read_to_string(&src).expect("fixture text");
    let stripped: String = text
        .lines()
        .filter(|l| !l.starts_with("!air.language_version"))
        .map(|l| format!("{l}\n"))
        .collect();
    assert_ne!(stripped.len(), text.len(), "the fixture states its version");
    let scratch = Scratch::new("m2v-fragment-test");
    let dir = scratch.path();
    let path = dir.join("unversioned.ll");
    std::fs::write(&path, stripped).expect("write");
    let bc = assemble(&path);
    let Some(bc) = bc else { return };
    let (_, p) = committed_profiles()
        .expect("profiles")
        .into_iter()
        .find(|(s, _)| s == "moltenvk-m3")
        .expect("moltenvk-m3");
    assert_eq!(
        refusal(&bc, "discard_terminate", &p),
        (
            Feature::Lowering(LowerFamily::Fragment),
            Needs::NotImplemented
        )
    );
}

/// The input attachments `a` declares, by color index, each an `INPUT_ATTACHMENT` descriptor.
fn attachments(a: &crate::library::FunctionArtifact) -> Vec<u32> {
    a.interface
        .descriptors
        .iter()
        .filter_map(|d| match d.slot.kind {
            ResourceKind::InputAttachment(c) => {
                assert_eq!(d.ty, vk::DescriptorType::INPUT_ATTACHMENT, "{d:?}");
                assert_eq!(d.slot.index, c, "{d:?}");
                Some(c)
            }
            _ => None,
        })
        .collect()
}

/// The storage-form attachments `a` declares, by color index, each a `STORAGE_IMAGE` descriptor
/// with its `InterlockedFramebufferFetch` fact.
fn storage_attachments(a: &crate::library::FunctionArtifact) -> Vec<u32> {
    a.interface
        .descriptors
        .iter()
        .filter_map(|d| match d.slot.kind {
            ResourceKind::TileAttachment(c) => {
                assert_eq!(d.ty, vk::DescriptorType::STORAGE_IMAGE, "{d:?}");
                assert_eq!(d.slot.index, c, "{d:?}");
                assert!(
                    a.facts.iter().any(|f| matches!(
                        f,
                        RuntimeFact::InterlockedFramebufferFetch { color, .. } if *color == c
                    )),
                    "{:?}",
                    a.facts
                );
                Some(c)
            }
            _ => None,
        })
        .collect()
}

/// The framebuffer-fetch state of a pipeline whose color attachments are `formats` (by index),
/// read at `read`, `blending` the first (adding the destination: a blend of one and zero would
/// replace it).
fn fetch_state(read: &[u32], formats: &[PixelFormat], blending: bool) -> LoweringInputs {
    let colors: Vec<Option<ColorAttachmentDescriptor>> = formats
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            let mut c = ColorAttachmentDescriptor::new(f);
            if blending && i == 0 {
                c.blending_enabled = true;
                c.destination_rgb_blend_factor = crate::metal::BlendFactor::One;
            }
            Some(c)
        })
        .collect();
    let mut inputs = LoweringInputs::new();
    inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of(
        read.iter().copied(),
        &colors,
        false,
        false,
        1,
    ));
    inputs
}

/// Each color index a function reads is read in an ordered form wherever the device has one. With
/// rasterization-order color attachment access it is one input attachment with a `FramebufferFetch`
/// fact. With only the ordered pixel interlock it is one storage image read (and written) inside
/// the interlock, under its pipeline's state (standalone, the function needs that state); a
/// blended attachment's blend the shader computes, Metal's of a 16-bit float or 8-bit unorm one,
/// the generic one, noted `UnmeasuredFetchBlend`, of another. With neither, it is refused naming
/// both. Every compiled module passes L3.
#[test]
fn framebuffer_fetch_reads_in_an_ordered_form_where_the_device_has_one() {
    let Some(bc) = fixture("fragment_fetch") else {
        return;
    };
    // (entry, colors read, the pipeline's attachment formats): `fetch_blend` reads three colors
    // of distinct classes and writes the first two; the others read and write color 0.
    let entries: [(&str, &[u32], &[PixelFormat]); 3] = [
        (
            "fetch_blend",
            &[0, 1, 2],
            &[
                PixelFormat::RGBA8Unorm,
                PixelFormat::RGBA32Uint,
                PixelFormat::RG16Float,
            ],
        ),
        ("fetch_scalar", &[0], &[PixelFormat::R32Float]),
        ("fetch_ordered", &[0], &[PixelFormat::RGBA16Float]),
    ];
    let mut forms = [0usize; 3];
    for (stem, strict) in committed_profiles().expect("profiles") {
        let frag = strict.fields().fragment;
        for (entry, colors, formats) in entries {
            // A color input in a raster order group needs the interlock, where rasterization-order
            // access orders the read; with neither ordered form the read is refused first, naming
            // both (below).
            if entry == "fetch_ordered"
                && !frag.pixel_interlock
                && frag.rasterization_order_color_access
            {
                assert_eq!(
                    refusal(&bc, entry, &strict),
                    (Feature::RasterOrderGroups, pixel_interlock()),
                    "{stem}"
                );
                continue;
            }
            let clean = fetch_state(colors, formats, false);
            if frag.rasterization_order_color_access {
                forms[0] += 1;
                for inputs in [LoweringInputs::new(), clean] {
                    let a = compile_with(&bc, entry, &strict, &inputs)
                        .unwrap_or_else(|e| panic!("{entry} on {stem}: {e}"));
                    link_and_validate(&a, &strict)
                        .unwrap_or_else(|e| panic!("{entry} on {stem}: L3: {e}"));
                    assert_eq!(attachments(&a), colors, "{entry} on {stem}");
                    for &color in colors {
                        assert!(
                            a.facts.contains(&RuntimeFact::FramebufferFetch { color }),
                            "{entry} on {stem}: {:?}",
                            a.facts
                        );
                    }
                }
            } else if frag.pixel_interlock {
                forms[1] += 1;
                assert!(
                    matches!(
                        compile(&bc, entry, &strict),
                        Err(Error::NeedsInput(RuntimeInput::FramebufferFetch))
                    ),
                    "{entry} on {stem}"
                );
                let a = compile_with(&bc, entry, &strict, &clean)
                    .unwrap_or_else(|e| panic!("{entry} on {stem}: {e}"));
                link_and_validate(&a, &strict)
                    .unwrap_or_else(|e| panic!("{entry} on {stem}: L3: {e}"));
                assert_eq!(storage_attachments(&a), colors, "{entry} on {stem}");
                assert!(attachments(&a).is_empty(), "{entry} on {stem}");
                // Blending the written color 0: the shader computes the blend, Metal's of a 16-bit
                // float or 8-bit unorm one, the generic one, noted, of another (`R32Float`).
                let blended = fetch_state(colors, formats, true);
                let a = compile_with(&bc, entry, &strict, &blended)
                    .unwrap_or_else(|e| panic!("{entry} on {stem}: blended: {e}"));
                link_and_validate(&a, &strict)
                    .unwrap_or_else(|e| panic!("{entry} on {stem}: blended: L3: {e}"));
                assert_eq!(storage_attachments(&a), colors, "{entry} on {stem}");
                assert!(
                    a.facts.iter().any(|f| matches!(
                        f,
                        RuntimeFact::InterlockedFramebufferFetch {
                            color: 0,
                            blend: Some(_),
                            ..
                        }
                    )),
                    "{entry} on {stem}: {:?}",
                    a.facts
                );
                let generic = !matches!(
                    formats[0],
                    PixelFormat::RGBA16Float | PixelFormat::RGBA8Unorm
                );
                assert_eq!(
                    a.precision.contains(&PrecisionNote::UnmeasuredFetchBlend),
                    generic,
                    "{entry} on {stem}: {:?}",
                    a.precision
                );
            } else {
                forms[2] += 1;
                // Each entry writes the color it reads first: the interlock's alternative names
                // what the device lacks of the interlock and the fragment-stage store.
                let mut interlock =
                    vec![Capability::Feature(VkFeature::FragmentShaderPixelInterlock)];
                if !strict.fields().leaf.fragment_stores_and_atomics {
                    interlock.push(Capability::Feature(VkFeature::FragmentStoresAndAtomics));
                }
                assert_eq!(
                    refusal(&bc, entry, &strict),
                    (
                        Feature::FramebufferFetch,
                        Needs::AnyOf(vec![
                            vec![Capability::Feature(
                                VkFeature::RasterizationOrderColorAttachmentAccess
                            )],
                            interlock,
                        ])
                    ),
                    "{entry} on {stem}"
                );
            }
        }
    }
    // The committed profiles cover each kind of device.
    assert!(forms.iter().all(|&n| n > 0), "{forms:?}");
}

/// Whether module `words` holds a constant composite of 256 or more constituents: Metal's sRGB
/// table (`lower::arith::srgb::declare`), which no other initializer of these fixtures reaches.
pub(super) fn holds_srgb_table(words: &[u32]) -> bool {
    const CONSTANT_COMPOSITE: u32 = 44;
    let mut at = 5;
    while let Some(&first) = words.get(at) {
        let count = (first >> 16) as usize;
        if first & 0xffff == CONSTANT_COMPOSITE && count >= 3 + 256 {
            return true;
        }
        at += count.max(1);
    }
    false
}

/// On a device that reads a fetched color through an input attachment (both lavapipes), an
/// `RGBA8Unorm_sRGB` or `BGRA8Unorm_sRGB` color reads the device's own decode through its input
/// attachment: the render pipeline's fragment module is the one an `RGBA8Unorm` attachment's
/// pipeline takes, word for word, holding no Metal sRGB table; the interface records the `float`
/// colors the device decodes (`StageIo::decoded_fetches`), and the pipeline notes
/// `PrecisionNote::SrgbTexelDecode` for the sRGB attachment alone. A fragment compiled alone links
/// into the sRGB pipeline exactly as into the `RGBA8Unorm` one. Every module passes L3.
#[test]
fn an_srgb_input_attachment_reads_the_devices_decode() {
    let (Some(bc), Some(io)) = (fixture("fragment_fetch"), fixture("stage_io")) else {
        return;
    };
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !p.fields().fragment.rasterization_order_color_access {
            continue;
        }
        let plain = compile_with(&bc, "fetch_blend", &p, &LoweringInputs::new())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&plain, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert_eq!(attachments(&plain), [0, 1, 2], "{stem}");
        assert_eq!(
            plain.interface.io.decoded_fetches,
            std::collections::BTreeSet::from([0, 2]),
            "{stem}"
        );
        assert!(!holds_srgb_table(&plain.module.words), "{stem}");
        checked += 1;
    }
    assert!(checked > 0, "no profile reads through input attachments");
    // The whole pipeline on `lavapipe`, and a fragment compiled alone in it.
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/lavapipe.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let dev = Device::new(desc, DevicePolicy::default()).expect("device");
    let function = |bc: &[u8], name: &str| {
        FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        )
    };
    let pipeline = |first: PixelFormat, fragment: StageSource| {
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(
            &io,
            "stage_io_vertex_position",
        )));
        d.fragment = Some(fragment);
        for (i, f) in [first, PixelFormat::RGBA32Uint, PixelFormat::RG16Float]
            .into_iter()
            .enumerate()
        {
            d.color_attachments[i] = Some(ColorAttachmentDescriptor::new(f));
        }
        dev.compile_render_pipeline(&d, &Budget::standard())
    };
    let fragment = || StageSource::Function(function(&bc, "fetch_blend"));
    let linear = pipeline(PixelFormat::RGBA8Unorm, fragment()).expect("RGBA8Unorm");
    let linear_stage = &linear.stages[1];
    assert!(!linear_stage
        .precision_notes
        .contains(&PrecisionNote::SrgbTexelDecode));
    for pf in [PixelFormat::BGRA8UnormSRGB, PixelFormat::RGBA8UnormSRGB] {
        let srgb = pipeline(pf, fragment()).unwrap_or_else(|e| panic!("{pf:?}: {e}"));
        let stage = &srgb.stages[1];
        assert!(!holds_srgb_table(&stage.spirv), "{pf:?}");
        assert_eq!(stage.spirv, linear_stage.spirv, "{pf:?}");
        assert!(
            stage
                .precision_notes
                .contains(&PrecisionNote::SrgbTexelDecode),
            "{pf:?}: {:?}",
            stage.precision_notes
        );
        let alone = dev
            .compile_function(
                &function(&bc, "fetch_blend"),
                &LoweringInputs::new(),
                &Budget::standard(),
            )
            .expect("the fragment alone");
        // The sRGB attachment asks the fragment for nothing the `RGBA8Unorm` one does not.
        assert_eq!(
            pipeline(pf, StageSource::Compiled(alone.clone())).map(|_| ()),
            pipeline(PixelFormat::RGBA8Unorm, StageSource::Compiled(alone)).map(|_| ()),
            "{pf:?}"
        );
    }
}

/// On a device that orders fetch only through the pixel interlock (`moltenvk-m3`), the binder
/// writes a storage-form attachment to its fragment binding as the attachment's own storage view,
/// and refuses a pass of more than one layer, whose layers one 2D view does not reach.
#[test]
fn the_binder_writes_storage_attachments_and_refuses_layered_passes() {
    let (Some(io), Some(fetch)) = (fixture("stage_io"), fixture("fragment_fetch")) else {
        return;
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let dev = Device::new(desc, DevicePolicy::default()).expect("device");
    let function = |bc: &[u8], name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(function(&io, "stage_io_vertex_position"));
    d.fragment = Some(function(&fetch, "fetch_scalar"));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Float));
    let p = dev
        .compile_render_pipeline(&d, &Budget::standard())
        .expect("pipeline");
    let plan = p.plan(&dev, &BindState::default().shape()).expect("plan");
    let storage: Vec<_> = plan
        .descriptors
        .iter()
        .filter(|w| {
            w.source
                == WriteSource::TileAttachment {
                    color_index: 0,
                    format: vk::Format::R32_SFLOAT,
                    view: crate::pipeline::StorageAttachmentView::Attachment,
                    view_alias: None,
                }
        })
        .collect();
    assert_eq!(storage.len(), 1, "{:?}", plan.descriptors);
    let b = p
        .layout
        .sets
        .iter()
        .flat_map(|s| &s.bindings)
        .find(|b| b.binding == storage[0].binding)
        .expect("binding");
    assert_eq!(b.ty, vk::DescriptorType::STORAGE_IMAGE);
    assert_eq!(b.source, BindingSource::TileAttachment(0));
    let layered = BindState {
        render_target_array_length: 2,
        ..Default::default()
    };
    match p.plan(&dev, &layered.shape()) {
        Err(BindError::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (Feature::FramebufferFetch, Needs::NotImplemented)
        ),
        other => panic!("{:?}", other.map(|_| ())),
    }
}

/// A draw into a render pass instance that keeps a color out of its subpass for other draws'
/// interlocked storage form runs the variant whose fragment function writes that color through the
/// same form. `rt6_resolve_white`, writing color 0 of an `RGBA16Float` attachment, planned with
/// colors 0 and 1 interlocked asks for color 0 alone (it writes no color 1); the variant's pipeline
/// writes color 0 as a storage attachment and declares no output for it, passes L3, plans the same
/// variant against itself, and binds the attachment's storage view. The pipeline itself states
/// nothing of the sort. Blended, the variant is refused (the storage form keeps no blend but one
/// and zero), so the runtime keeps such a draw in its own instance.
#[test]
fn a_draw_writes_an_interlocked_color_through_the_storage_form() {
    let Some(bc) = fixture("rt6_resolves") else {
        return;
    };
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            Library::from_air(&bc)
                .expect("library")
                .function(name)
                .expect("function"),
        ))
    };
    let descriptor = |blended: bool| {
        let mut d = RenderPipelineDescriptor::new(function("rt6_resolve_full"));
        d.fragment = Some(function("rt6_resolve_white"));
        let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
        if blended {
            color.blending_enabled = true;
            color.destination_rgb_blend_factor = crate::metal::BlendFactor::One;
        }
        d.color_attachments[0] = Some(color);
        d
    };
    for stem in [
        "moltenvk-m3",
        "amd-780m-radv",
        "intel-uhd770-rpls-anv",
        "nvidia-rtx5080-laptop",
    ] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        let dev = Device::new(desc, DevicePolicy::default()).expect("device");
        let d = descriptor(false);
        let base = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(base.graphics.storage_attachments.is_empty(), "{stem}");
        let state = BindState {
            interlocked_colors: [0, 1].into(),
            ..Default::default()
        };
        let plan = base.plan(&dev, &state.shape()).expect("plan");
        assert_eq!(
            plan.variant.interlocked_colors,
            [0].into(),
            "{stem}: the colors it writes"
        );
        let variant = dev
            .compile_render_variant(&d, &plan.variant, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: variant: {e}"));
        let stored: Vec<(u32, bool)> = variant
            .graphics
            .storage_attachments
            .iter()
            .map(|a| (a.color_index, a.writes))
            .collect();
        assert_eq!(stored, [(0, true)], "{stem}");
        let again = variant
            .plan(&dev, &state.shape())
            .expect("unblended variant plan");
        assert_eq!(
            again.variant, plan.variant,
            "{stem}: unblended: a fixed point"
        );
        assert!(
            again
                .graphics
                .as_ref()
                .is_some_and(|t| !t.other_layer_variant),
            "{stem}: unblended: the variant's draws run on it: {:?}",
            variant
                .dependencies
                .iter()
                .map(|d| &d.fact)
                .collect::<Vec<_>>()
        );
        for m in &variant.stages {
            check_words_valid(&m.spirv, &variant.requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        }
        let again = variant.plan(&dev, &state.shape()).expect("variant plan");
        assert_eq!(again.variant, plan.variant, "{stem}: a fixed point");
        assert!(
            again
                .graphics
                .as_ref()
                .is_some_and(|t| !t.other_layer_variant),
            "{stem}: the variant's draws run on it: {:?} {:?}",
            variant
                .dependencies
                .iter()
                .map(|d| &d.fact)
                .collect::<Vec<_>>(),
            variant.graphics.storage_attachments
        );
        assert!(
            again
                .descriptors
                .iter()
                .any(|w| matches!(w.source, WriteSource::TileAttachment { color_index: 0, .. })),
            "{stem}: {:?}",
            again.descriptors
        );
        // Blended, the form is not built: the variant is refused.
        let blended = descriptor(true);
        let base = dev
            .compile_render_pipeline(&blended, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: blended: {e}"));
        let plan = base.plan(&dev, &state.shape()).expect("blended plan");
        assert!(
            dev.compile_render_variant(&blended, &plan.variant, &Budget::standard())
                .is_err(),
            "{stem}: a blended interlocked color"
        );
    }
}

/// The image reads and writes (`OpImageRead`, `OpImageWrite`) of the SPIR-V module `words`.
fn image_accesses(words: &[u32]) -> (usize, usize) {
    let mut out = (0, 0);
    let mut at = 5;
    while let Some(&word) = words.get(at) {
        match word & 0xffff {
            98 => out.0 += 1,
            99 => out.1 += 1,
            _ => {}
        }
        at += ((word >> 16) as usize).max(1);
    }
    out
}

/// A draw whose render pass instance holds the `half` shadow of an extended-range attachment runs
/// the variant whose fragment function reads and writes the shadow beside the codes.
/// `con_fetch_coverage`, blending a `half4` into its `RGB10A8_2P_XR10` color 0 (premultiplied over)
/// and fetching color 1, planned with colors 0 and 5 shadowed asks for the shadow of color 0 alone
/// (the pipeline has no color 5): the variant binds an `ExtendedShadow(0)` storage image, reads the
/// blend's destination from the shadow in place of the codes (as many image reads as the pipeline
/// unshadowed) and writes both (one write more), passes L3, and plans the same variant against
/// itself, writing `WriteSource::ExtendedShadow`. `rt6_resolve_white`, writing a `BGRA10_XR` color
/// 0 unblended through an ordinary output, takes the storage form under the shadow and writes the
/// codes and the shadow, reading neither. A layered pass's and a multisampled pipeline's shadow are
/// refused, as the interlocked colors' are.
#[test]
fn a_shadowed_extended_range_attachment_reads_and_writes_its_half() {
    use crate::metal::{BlendFactor, ColorWriteMask};
    let (Some(coverage), Some(io), Some(resolves)) = (
        fixture("con_fetch_coverage"),
        fixture("stage_io"),
        fixture("rt6_resolves"),
    ) else {
        return;
    };
    let function = |bc: &[u8], name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        ))
    };
    // The guest's shape: color 0 premultiplied over, color 1 read, color 2 of mask none.
    let blended = || {
        let mut d = RenderPipelineDescriptor::new(function(&io, "stage_io_vertex_position"));
        d.fragment = Some(function(&coverage, "con_fetch_coverage"));
        let mut over = ColorAttachmentDescriptor::new(PixelFormat::RGB10A82PXR10);
        over.blending_enabled = true;
        over.source_rgb_blend_factor = BlendFactor::One;
        over.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        over.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        let mut read = ColorAttachmentDescriptor::new(PixelFormat::RG16Float);
        read.write_mask = ColorWriteMask::NONE;
        let mut unwritten = ColorAttachmentDescriptor::new(PixelFormat::RGB10A82PXR10);
        unwritten.write_mask = ColorWriteMask::NONE;
        d.color_attachments[0] = Some(over);
        d.color_attachments[1] = Some(read);
        d.color_attachments[2] = Some(unwritten);
        d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
        d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
        d
    };
    let plain = || {
        let mut d = RenderPipelineDescriptor::new(function(&resolves, "rt6_resolve_full"));
        d.fragment = Some(function(&resolves, "rt6_resolve_white"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA10XR));
        d
    };
    let fragment_words = |p: &crate::pipeline::RenderPipeline| {
        p.stages
            .iter()
            .find(|m| m.stage == vk::ShaderStage::FRAGMENT)
            .map(|m| m.spirv.to_vec())
            .expect("a fragment stage")
    };
    let shadows = |p: &crate::pipeline::RenderPipeline| -> Vec<(u32, vk::DescriptorType)> {
        p.layout
            .sets
            .iter()
            .flat_map(|s| &s.bindings)
            .filter_map(|b| match b.source {
                BindingSource::ExtendedShadow(c) => Some((c, b.ty)),
                _ => None,
            })
            .collect()
    };
    let refused = |r: Result<crate::bind::BindingPlan, BindError>| match r {
        Err(BindError::Unsupported(u)) => {
            (u.feature, u.needs) == (Feature::FramebufferFetch, Needs::NotImplemented)
        }
        _ => false,
    };
    let mut checked = 0;
    for stem in [
        "moltenvk-m3",
        "amd-780m-radv",
        "intel-uhd770-rpls-anv",
        "nvidia-rtx5080-laptop",
    ] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        let dev = Device::new(desc, DevicePolicy::default()).expect("device");
        if !crate::lower::fragment::interlock_orders(dev.profile()) {
            continue;
        }
        let d = blended();
        let base = match dev.compile_render_pipeline(&d, &Budget::standard()) {
            Ok(base) => base,
            // A device without the raw formats' storage refuses the form whatever the shadow.
            Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => continue,
            Err(e) => panic!("{stem}: {e}"),
        };
        checked += 1;
        assert!(shadows(&base).is_empty(), "{stem}");
        // The storage form's writes are not the function's.
        assert!(!base.graphics.fragment_writes_memory, "{stem}: blended");
        let (reads, writes) = image_accesses(&fragment_words(&base));
        let state = BindState {
            extended_shadows: [0, 5].into(),
            ..Default::default()
        };
        let plan = base.plan(&dev, &state.shape()).expect("plan");
        assert_eq!(plan.variant.extended_shadows, [0].into(), "{stem}");
        let variant = dev
            .compile_render_variant(&d, &plan.variant, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: variant: {e}"));
        assert_eq!(
            shadows(&variant),
            [(0, vk::DescriptorType::STORAGE_IMAGE)],
            "{stem}"
        );
        assert_eq!(
            image_accesses(&fragment_words(&variant)),
            (reads, writes + 1),
            "{stem}: the shadow's read replaces the codes' and both are written"
        );
        for m in &variant.stages {
            check_words_valid(&m.spirv, &variant.requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        }
        let again = variant.plan(&dev, &state.shape()).expect("variant plan");
        assert_eq!(again.variant, plan.variant, "{stem}: a fixed point");
        assert!(
            again
                .descriptors
                .iter()
                .any(|w| w.source == WriteSource::ExtendedShadow { color_index: 0 }),
            "{stem}: {:?}",
            again.descriptors
        );
        // Unshadowed again, the base pipeline's plan.
        let unshadowed = variant
            .plan(&dev, &BindState::default().shape())
            .expect("unshadowed plan");
        assert!(unshadowed.variant.extended_shadows.is_empty(), "{stem}");
        // A layered pass's shadow is refused.
        let layered = BindState {
            extended_shadows: [0].into(),
            render_target_array_length: 2,
            ..Default::default()
        };
        assert!(
            refused(base.plan(&dev, &layered.shape())),
            "{stem}: layered"
        );
        // An unblended write takes the storage form under the shadow.
        let d = plain();
        let base = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: unblended: {e}"));
        assert!(base.graphics.storage_attachments.is_empty(), "{stem}");
        let plan = base.plan(&dev, &state.shape()).expect("unblended plan");
        assert_eq!(plan.variant.extended_shadows, [0].into(), "{stem}");
        let variant = match dev.compile_render_variant(&d, &plan.variant, &Budget::standard()) {
            Ok(variant) => variant,
            // A device without the raw format's storage refuses the form whatever the shadow.
            Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => continue,
            Err(e) => panic!("{stem}: unblended variant: {e}"),
        };
        let stored: Vec<(u32, bool)> = variant
            .graphics
            .storage_attachments
            .iter()
            .map(|a| (a.color_index, a.writes))
            .collect();
        assert_eq!(stored, [(0, true)], "{stem}");
        assert_eq!(
            image_accesses(&fragment_words(&variant)),
            (0, 2),
            "{stem}: the codes and the shadow written, neither read"
        );
        // Nor are the codes' and the shadow's writes.
        assert!(
            !variant.graphics.fragment_writes_memory,
            "{stem}: unblended"
        );
        for m in &variant.stages {
            check_words_valid(&m.spirv, &variant.requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{stem}: unblended: L3: {e}"));
        }
        // A multisampled pipeline's shadow is refused.
        let mut d = plain();
        d.raster_sample_count = 4;
        let base = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: multisampled: {e}"));
        assert!(
            refused(base.plan(&dev, &state.shape())),
            "{stem}: multisampled"
        );
    }
    assert!(checked > 0, "no profile compiled the shadow");
}

/// `fragment_fetch` with `fetch_scalar` discarding a fragment whose color input is negative.
fn fetch_with_discard() -> Option<Vec<u8>> {
    let src = engine_dir().join("fragment_fetch.ll");
    let text = std::fs::read_to_string(&src).expect("fixture text");
    let body = "  %2 = fmul fast float %0, 5.000000e-01\n  ret float %2\n";
    assert!(text.contains(body), "the fixture's fetch_scalar body");
    let discarding = "  %2 = fmul fast float %0, 5.000000e-01\n  %3 = fcmp fast olt float %0, 0.000000e+00\n  br i1 %3, label %4, label %5\n4:\n  tail call void @air.discard_fragment()\n  br label %5\n5:\n  ret float %2\n";
    let edited = text
        .replace(body, discarding)
        .replace(
            "\n; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)\ndefine <4 x float> @fetch_ordered",
            "\ndeclare void @air.discard_fragment() local_unnamed_addr\n\n; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)\ndefine <4 x float> @fetch_ordered",
        )
        .replace(
            "define float @fetch_scalar(float noundef %0) local_unnamed_addr #2",
            "define float @fetch_scalar(float noundef %0) local_unnamed_addr",
        );
    let scratch = Scratch::new("m2v-fetch-discard");
    let dir = scratch.path();
    let path = dir.join("fetch_discard.ll");
    std::fs::write(&path, edited).expect("write");
    assemble(&path)
}

/// The storage form declares early fragment tests where a depth or stencil attachment's tests could
/// drop a fragment after its writes, and where they keep the function's writes Metal's (no discard,
/// no depth, stencil, or coverage export, no other memory write): without a depth or stencil
/// attachment no test can fail, so the module declares none. A function that may discard reads and
/// writes in the storage form without them (a discarded fragment writes nothing). Beside a depth
/// attachment, where its write precedes the depth test, it states `TestedStorageWrites` without
/// early tests, whose draws' plans check their tests, and under `read_only_tests` declares early
/// tests and states it with them. A stage compiled without a depth attachment declares no early
/// tests, so it links into no pipeline with one: `NeedsInput(FramebufferFetch)`.
#[test]
fn the_storage_form_declares_early_tests_only_where_they_keep_metals_writes() {
    let (Some(plain), Some(discarding)) = (fixture("fragment_fetch"), fetch_with_discard()) else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let state = |depth: bool| {
        let mut inputs = LoweringInputs::new();
        inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of(
            [0],
            &[Some(ColorAttachmentDescriptor::new(PixelFormat::R32Float))],
            depth,
            false,
            1,
        ));
        inputs
    };
    let early = |a: &crate::library::FunctionArtifact| {
        a.facts.iter().find_map(|f| match f {
            RuntimeFact::InterlockedFramebufferFetch { early_tests, .. } => Some(*early_tests),
            _ => None,
        })
    };
    let tested = |a: &crate::library::FunctionArtifact| {
        a.facts.iter().find_map(|f| match f {
            RuntimeFact::TestedStorageWrites {
                early_tests,
                early_variant: true,
            } => Some(*early_tests),
            _ => None,
        })
    };
    for (bc, discards) in [(&plain, false), (&discarding, true)] {
        let what = if discards { "discarding" } else { "plain" };
        for (depth, read_only) in [(false, false), (true, false), (true, true)] {
            let mut inputs = state(depth);
            inputs.read_only_tests = read_only;
            let declared = depth && (!discards || read_only);
            let a = compile_with(bc, "fetch_scalar", &p, &inputs)
                .unwrap_or_else(|e| panic!("{what} depth {depth} read-only {read_only}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_eq!(early(&a), Some(declared), "{what} depth {depth}");
            assert_eq!(
                tested(&a),
                (discards && depth).then_some(read_only),
                "{what} depth {depth} read-only {read_only}"
            );
            let (_, m) = typed_module_with(bc, "fetch_scalar", &p, &inputs)
                .unwrap_or_else(|e| panic!("{what} depth {depth}: {e}"));
            let modes = &m.entry().modes;
            assert!(modes.contains(&ExecMode::PixelInterlockOrdered), "{what}");
            assert_eq!(
                modes.contains(&ExecMode::EarlyFragmentTests),
                declared,
                "{what} depth {depth} read-only {read_only}: {modes:?}"
            );
        }
    }
    // Compiled without a depth attachment, into a pipeline with one.
    let Some(io) = fixture("stage_io") else {
        return;
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let dev = Device::new(desc, DevicePolicy::default()).expect("device");
    let function = |bc: &[u8], name: &str| {
        FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        )
    };
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(
        &io,
        "stage_io_vertex_position",
    )));
    let fragment = dev
        .compile_function(
            &function(&plain, "fetch_scalar"),
            &state(false),
            &Budget::standard(),
        )
        .expect("fragment");
    d.fragment = Some(StageSource::Compiled(fragment));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Float));
    d.depth_format = Some(PixelFormat::Depth32Float);
    assert_eq!(
        dev.compile_render_pipeline(&d, &Budget::standard())
            .map(|_| ())
            .unwrap_err(),
        Error::NeedsInput(RuntimeInput::FramebufferFetch)
    );
}

/// A fetching function that writes other memory (`dvp_rog_coherence`'s `fetch_coherent`), beside a
/// depth or stencil attachment, compiles in the storage form without early tests, its stores before
/// the tests, and states `TestedStorageWrites` without the early-tests variant: Metal runs its
/// memory writes for the fragments its tests drop. Without such an attachment it
/// states nothing. The whole pipeline on `moltenvk-m3` (a `BGRA8Unorm_sRGB` color,
/// `Depth32Float_Stencil8`) plans a draw whose tests drop no fragment, and refuses one that does
/// not state its tests, or whose tests a visibility result counts, at the depth attachment. (A draw
/// whose tests may drop fragments but write and count nothing runs the variant whose function runs
/// both tests itself, after its memory writes, as Metal tests after them.)
#[test]
fn a_fetch_that_writes_memory_beside_depth_runs_only_draws_whose_tests_drop_nothing() {
    use crate::bind::FragmentTests;
    use crate::error::{Attachment, Unsupported};
    let (Some(bc), Some(io)) = (fixture("dvp_rog_coherence"), fixture("stage_io")) else {
        return;
    };
    let state = |depth: bool| {
        let mut inputs = fetch_state(&[0], &[PixelFormat::BGRA8UnormSRGB], false);
        let colors = [Some(ColorAttachmentDescriptor::new(
            PixelFormat::BGRA8UnormSRGB,
        ))];
        inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of([0], &colors, depth, false, 1));
        inputs
    };
    let tested = |a: &crate::library::FunctionArtifact| {
        a.facts.iter().find_map(|f| match f {
            RuntimeFact::TestedStorageWrites {
                early_tests,
                early_variant,
            } => Some((*early_tests, *early_variant)),
            _ => None,
        })
    };
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        for depth in [false, true] {
            let a = compile_with(&bc, "fetch_coherent", &p, &state(depth))
                .unwrap_or_else(|e| panic!("{stem} depth {depth}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
            assert_eq!(storage_attachments(&a), [0], "{stem}");
            assert_eq!(tested(&a), depth.then_some((false, false)), "{stem}");
            let (_, m) = typed_module_with(&bc, "fetch_coherent", &p, &state(depth))
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert!(
                !m.entry().modes.contains(&ExecMode::EarlyFragmentTests),
                "{stem} depth {depth}"
            );
        }
        checked += 1;
    }
    assert!(checked > 0, "no interlock profile");
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let dev = Device::new(desc, DevicePolicy::default()).expect("device");
    let function = |bc: &[u8], name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(function(&io, "stage_io_vertex_position"));
    d.fragment = Some(function(&bc, "fetch_coherent"));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8UnormSRGB));
    d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
    d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
    let base = dev
        .compile_render_pipeline(&d, &Budget::standard())
        .expect("the pipeline");
    let shape = |fragment_tests: FragmentTests| {
        BindState {
            fragment_tests,
            ..Default::default()
        }
        .shape()
    };
    let pass = base.plan(&dev, &shape(FragmentTests::Pass)).expect("pass");
    assert!(!pass.variant.read_only_tests);
    assert_eq!(
        base.plan(&dev, &shape(FragmentTests::Any)).map(|_| ()),
        Err(BindError::Unsupported(
            Unsupported::not_implemented(Feature::FramebufferFetch)
                .at_attachment(Attachment::Depth)
        ))
    );
    let drop = base.plan(&dev, &shape(FragmentTests::Drop)).expect("drop");
    assert!(drop.variant.shader_tests && drop.variant.shader_stencil);
    assert!(!drop.variant.read_only_tests);
    let variant = dev
        .compile_render_variant(&d, &drop.variant, &Budget::standard())
        .expect("the in-shader tests' variant");
    for m in &variant.stages {
        check_words_valid(&m.spirv, &variant.requirements, dev.profile())
            .unwrap_or_else(|e| panic!("L3: {e}"));
    }
    let again = variant
        .plan(&dev, &shape(FragmentTests::Drop))
        .expect("the variant's plan");
    assert_eq!(again.variant, drop.variant, "a fixed point");
}

/// The pipeline state of `con_fetch_coverage`, as the guest's path-coverage pipeline has it: a
/// blended `RGBA16Float` color 0 the function writes, the `RG16Float` color 1 it reads and writes
/// under `mask`, an `RGBA16Float` color 2 it never writes (mask none), and a depth and stencil
/// attachment.
fn coverage_colors(
    coverage: PixelFormat,
    mask: crate::metal::ColorWriteMask,
) -> Vec<Option<ColorAttachmentDescriptor>> {
    use crate::metal::{BlendFactor, ColorWriteMask};
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
    color.blending_enabled = true;
    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    let mut covered = ColorAttachmentDescriptor::new(coverage);
    covered.write_mask = mask;
    let mut unwritten = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
    unwritten.write_mask = ColorWriteMask::NONE;
    vec![Some(color), Some(covered), Some(unwritten)]
}

/// In the storage form a write mask short of the fetched attachment's channels keeps the others:
/// the write reads the texel first and writes the kept channels back as it read them, and the fact
/// names its mask's class, so a stage compiled under a mask of one class links into no pipeline of
/// another (`NeedsInput(FramebufferFetch)`). A mask of none of the format's channels writes
/// nothing; one of all of them reads nothing more. A merged `half` write notes the device's NaN
/// decode. A signed normalized format with a kept channel stays refused (its two codes of -1
/// read alike). Every compiled module passes L3, on every committed profile that orders fetch
/// through the interlock; the guest's whole pipeline compiles on `moltenvk-m3`.
#[test]
fn a_masked_fetched_write_keeps_the_channels_its_mask_leaves_out() {
    use crate::metal::ColorWriteMask;
    let Some(bc) = fixture("con_fetch_coverage") else {
        return;
    };
    let entry = "con_fetch_coverage";
    let state = |coverage: PixelFormat, mask: ColorWriteMask| {
        let mut inputs = LoweringInputs::new();
        inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of(
            [1],
            &coverage_colors(coverage, mask),
            true,
            false,
            1,
        ));
        inputs
    };
    let kept = |a: &crate::library::FunctionArtifact| {
        a.facts.iter().find_map(|f| match f {
            RuntimeFact::InterlockedFramebufferFetch { color: 1, kept, .. } => Some(*kept),
            _ => None,
        })
    };
    let count = |m: &TypedModule| {
        let mut reads = 0;
        let mut writes = 0;
        for b in m.blocks() {
            for n in &b.insts {
                match n.inst {
                    Inst::ImageRead { .. } => reads += 1,
                    Inst::ImageWrite { .. } => writes += 1,
                    _ => {}
                }
            }
        }
        (reads, writes)
    };
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        let all = state(PixelFormat::RG16Float, ColorWriteMask::ALL);
        let whole = match compile_with(&bc, entry, &p, &all) {
            Ok(a) => a,
            // A device without the format's storage refuses every mask alike.
            Err(e) => {
                let green = state(PixelFormat::RG16Float, ColorWriteMask::GREEN);
                assert_eq!(compile_with(&bc, entry, &p, &green).map(|_| ()), Err(e));
                continue;
            }
        };
        checked += 1;
        link_and_validate(&whole, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert_eq!(
            kept(&whole),
            Some(vk::ColorComponentFlags::empty()),
            "{stem}"
        );
        let (_, m) = typed_module_with(&bc, entry, &p, &all).expect("module");
        let (reads, writes) = count(&m);
        // The guest's mask: green alone, so red is kept.
        let green = state(PixelFormat::RG16Float, ColorWriteMask::GREEN);
        let a = compile_with(&bc, entry, &p, &green).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert_eq!(kept(&a), Some(vk::ColorComponentFlags::R), "{stem}");
        assert!(
            a.precision.contains(&PrecisionNote::DeviceNanEncoding),
            "{stem}: {:?}",
            a.precision
        );
        let (_, m) = typed_module_with(&bc, entry, &p, &green).expect("module");
        assert_eq!(count(&m), (reads + 1, writes), "{stem}");
        // A mask that covers neither channel: no storage write at all.
        let none = state(PixelFormat::RG16Float, ColorWriteMask::BLUE);
        let a = compile_with(&bc, entry, &p, &none).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert_eq!(
            kept(&a),
            Some(vk::ColorComponentFlags::known_bits()),
            "{stem}"
        );
        let (_, m) = typed_module_with(&bc, entry, &p, &none).expect("module");
        assert_eq!(count(&m), (reads, writes - 1), "{stem}");
        // A signed normalized format with a kept channel.
        let snorm = state(PixelFormat::RG8Snorm, ColorWriteMask::GREEN);
        match compile_with(&bc, entry, &p, &snorm) {
            Err(Error::Unsupported(u)) => assert_eq!(
                (u.feature, u.needs),
                (Feature::FramebufferFetch, Needs::NotImplemented),
                "{stem}"
            ),
            other => panic!("{stem}: {:?}", other.map(|_| ())),
        }
    }
    assert!(
        checked > 0,
        "no interlock profile compiled the storage form"
    );
    // The guest's pipeline, and a stage compiled under its mask linked into another.
    let Some(io) = fixture("stage_io") else {
        return;
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let dev = Device::new(desc, DevicePolicy::default()).expect("device");
    let function = |bc: &[u8], name: &str| {
        FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        )
    };
    let pipeline = |fragment: StageSource, mask: ColorWriteMask| {
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(
            &io,
            "stage_io_vertex_position",
        )));
        d.fragment = Some(fragment);
        for (i, c) in coverage_colors(PixelFormat::RG16Float, mask)
            .into_iter()
            .enumerate()
        {
            d.color_attachments[i] = c;
        }
        d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
        d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
        dev.compile_render_pipeline(&d, &Budget::standard())
    };
    let source = StageSource::Function(function(&bc, entry));
    pipeline(source, ColorWriteMask::GREEN).expect("the guest's pipeline");
    let fragment = dev
        .compile_function(
            &function(&bc, entry),
            &state(PixelFormat::RG16Float, ColorWriteMask::GREEN),
            &Budget::standard(),
        )
        .expect("fragment");
    assert_eq!(
        pipeline(StageSource::Compiled(fragment), ColorWriteMask::ALL)
            .map(|_| ())
            .unwrap_err(),
        Error::NeedsInput(RuntimeInput::FramebufferFetch)
    );
}

/// The write masks that keep some of a fetched attachment's channels share one fragment key and one
/// module in each form. In the storage form (`moltenvk-m3`) the stage compiles under the mask's
/// class: its write still reads the texel first (one image read more than a write of every
/// channel), and the draw's `PushField::StorageKeptChannels` names the channels the pipeline's own
/// mask keeps (`StorageAttachmentUse::kept`, which the plan packs at bits `4c` to `4c + 3`). A
/// stage compiled under one partial mask links into a pipeline of another, and into none of a mask
/// of every channel. In the ordered form (each committed profile with rasterization order access)
/// the stage's inputs hold no mask at all.
#[test]
fn the_partial_write_masks_of_a_fetched_attachment_share_one_module() {
    use crate::bind::FragmentTests;
    use crate::metal::ColorWriteMask;
    use crate::pipeline::artifact::RenderPipeline;
    use crate::pipeline::sym::PushField;
    let (Some(bc), Some(io)) = (fixture("con_fetch_coverage"), fixture("stage_io")) else {
        return;
    };
    let entry = "con_fetch_coverage";
    let device = |stem: &str| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles")
            .join(format!("{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        Device::new(desc, DevicePolicy::default()).expect("device")
    };
    let function = |bc: &[u8], name: &str| {
        FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        )
    };
    let descriptor = |fragment: StageSource, mask: ColorWriteMask| {
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(
            &io,
            "stage_io_vertex_position",
        )));
        d.fragment = Some(fragment);
        for (i, c) in coverage_colors(PixelFormat::RG16Float, mask)
            .into_iter()
            .enumerate()
        {
            d.color_attachments[i] = c;
        }
        d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
        d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
        d
    };
    let source = || StageSource::Function(function(&bc, entry));
    // The storage form: red alone keeps green, green alone keeps red.
    let dev = device("moltenvk-m3");
    let keys = |mask| {
        dev.render_pipeline_key(&descriptor(source(), mask), &Budget::standard())
            .expect("keys")
    };
    let (red, green, all) = (
        keys(ColorWriteMask::RED),
        keys(ColorWriteMask::GREEN),
        keys(ColorWriteMask::ALL),
    );
    assert_eq!(red.fragment, green.fragment);
    assert_eq!(red.fragment_inputs, green.fragment_inputs);
    assert_ne!(red.fragment, all.fragment);
    let made = |source: StageSource, mask| {
        dev.compile_render_pipeline(&descriptor(source, mask), &Budget::standard())
    };
    let fragment_words = |p: &RenderPipeline| {
        p.stages
            .iter()
            .find(|m| m.stage == vk::ShaderStage::FRAGMENT)
            .expect("a fragment stage")
            .spirv
            .clone()
    };
    let shape = BindState {
        fragment_tests: FragmentTests::Pass,
        ..Default::default()
    }
    .shape();
    let kept_word = |p: &RenderPipeline| {
        p.plan(&dev, &shape)
            .expect("plan")
            .graphics
            .expect("a graphics template")
            .storage_kept
    };
    let reads_word = |p: &RenderPipeline| {
        p.interface.push.iter().any(|pl| {
            pl.fields
                .iter()
                .any(|f| f.field == PushField::StorageKeptChannels)
        })
    };
    let (r, g) = (
        made(source(), ColorWriteMask::RED).expect("red"),
        made(source(), ColorWriteMask::GREEN).expect("green"),
    );
    assert_eq!(fragment_words(&r), fragment_words(&g));
    assert!(reads_word(&r) && reads_word(&g));
    let kept = |p: &RenderPipeline| p.graphics.storage_attachments[0].kept;
    assert_eq!(kept(&r), vk::ColorComponentFlags::G);
    assert_eq!(kept(&g), vk::ColorComponentFlags::R);
    // Color 1's four bits.
    assert_eq!(kept_word(&r), 0x2 << 4);
    assert_eq!(kept_word(&g), 0x1 << 4);
    let w = made(source(), ColorWriteMask::ALL).expect("all");
    assert!(!reads_word(&w));
    assert_eq!(kept(&w), vk::ColorComponentFlags::empty());
    assert_eq!(kept_word(&w), 0);
    // The partial class's write reads the texel it merges.
    let p = dev.profile();
    let reads = |inputs: &LoweringInputs| {
        let (_, m) = typed_module_with(&bc, entry, p, inputs).expect("module");
        m.blocks()
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|n| matches!(n.inst, Inst::ImageRead { .. }))
            .count()
    };
    let partial = red.fragment_inputs.as_ref().expect("fragment inputs");
    let whole = all.fragment_inputs.as_ref().expect("fragment inputs");
    assert_eq!(reads(partial), reads(whole) + 1);
    // A stage compiled under red's class links under green, and not under every channel.
    let compiled = dev
        .compile_function(&function(&bc, entry), partial, &Budget::standard())
        .expect("fragment");
    let linked = made(
        StageSource::Compiled(compiled.clone()),
        ColorWriteMask::GREEN,
    )
    .expect("linked under green");
    assert_eq!(kept(&linked), vk::ColorComponentFlags::R);
    assert_eq!(
        made(StageSource::Compiled(compiled), ColorWriteMask::ALL)
            .map(|_| ())
            .unwrap_err(),
        Error::NeedsInput(RuntimeInput::FramebufferFetch)
    );
    // The ordered form: no mask in the inputs.
    let mut ordered = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !p.fields().fragment.rasterization_order_color_access {
            continue;
        }
        let dev = device(&stem);
        let keys = |mask| {
            dev.render_pipeline_key(&descriptor(source(), mask), &Budget::standard())
                .unwrap_or_else(|e| panic!("{stem}: {e}"))
        };
        let (red, green) = (keys(ColorWriteMask::RED), keys(ColorWriteMask::GREEN));
        let inputs = red.fragment_inputs.as_ref().expect("fragment inputs");
        assert_eq!(inputs.framebuffer_fetch, None, "{stem}");
        assert_eq!(red.fragment, green.fragment, "{stem}");
        ordered += 1;
    }
    assert!(
        ordered > 0,
        "no committed profile orders fetch by rasterization order"
    );
}

/// A fetched attachment of `BGRA8Unorm`, `RGBA8Unorm` or `RGBA16Float`, read and written
/// (`fragment_fetch`'s `fetch_ordered`, color 0), is declared without a format wherever the
/// device grants those formats' storage access without one, and its write selects its rounding
/// at the draw, so the fragment stage's module is the same for the three, on every committed
/// profile that orders fetch through the interlock and does not store Metal's color writes
/// itself.
#[test]
fn a_fetched_attachment_s_module_is_the_same_for_each_format_it_reads_alike() {
    use crate::device::requirements::VkProperty;
    use crate::lower::fragment::fetch::interlock_orders;
    use crate::metal::format::{accessed_without_format, StorageUse};
    let (Some(bc), Some(io)) = (fixture("fragment_fetch"), fixture("stage_io")) else {
        return;
    };
    let formats = [
        (PixelFormat::BGRA8Unorm, vk::Format::B8G8R8A8_UNORM),
        (PixelFormat::RGBA8Unorm, vk::Format::R8G8B8A8_UNORM),
        (PixelFormat::RGBA16Float, vk::Format::R16G16B16A16_SFLOAT),
    ];
    let function = |bc: &[u8], name: &str| {
        FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        )
    };
    let mut shared = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let metal = (p.fields().enabled.properties).contains(&VkProperty::MetalColorWrites);
        let access = StorageUse::texture(true, true);
        let untyped = formats
            .iter()
            .all(|&(_, f)| accessed_without_format(f, access, &p));
        if !interlock_orders(&p) || metal || !untyped {
            continue;
        }
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles")
            .join(format!("{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        let dev = Device::new(desc, DevicePolicy::default()).expect("device");
        let mut modules = Vec::new();
        for (pf, _) in formats {
            let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(
                &io,
                "stage_io_vertex_position",
            )));
            d.fragment = Some(StageSource::Function(function(&bc, "fetch_ordered")));
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(pf));
            let made = dev
                .compile_render_pipeline(&d, &Budget::standard())
                .unwrap_or_else(|e| panic!("{stem} {pf:?}: {e}"));
            let fragment = (made.stages.iter())
                .find(|m| m.stage == vk::ShaderStage::FRAGMENT)
                .expect("a fragment stage");
            modules.push(fragment.spirv.clone());
        }
        assert!(
            modules.windows(2).all(|m| m[0] == m[1]),
            "{stem}: {:?} words",
            modules.iter().map(|m| m.len()).collect::<Vec<_>>()
        );
        shared += 1;
    }
    assert!(shared > 0, "no committed profile shares the module");
}

/// In the shape of the pipeline a guest made on ANV: `fetch_ordered` reads and writes color 0, a
/// `BGRA8Unorm` attachment, beside an `RGBA16Float` color 1, no blending, every channel written. On
/// the ANV profile (no `shaderStorageImageReadWithoutFormat`, and SPIR-V names no image format for
/// `B8G8R8A8_UNORM`) the whole pipeline is made, its fetched attachment a written storage
/// attachment viewed in `R8G8B8A8_UNORM` (`StorageAttachmentUse::view_alias`), declared `Rgba8`,
/// which the plan's write binds; the fragment module exchanges red and blue (an `OpVectorShuffle`
/// of lanes 2, 1, 0, 3 on its read and its write) and passes `spirv-val`. So at 4 samples, through
/// sample planes, and under a write mask that keeps green and alpha. A pipeline fetching an
/// `RGBA16Float` color 0 alone is made there too, viewed in its own format, declared `Rgba16f`. On
/// a profile that reads and writes `B8G8R8A8_UNORM` without a format the pipeline keeps its own
/// format, declared without one, as before.
#[test]
fn a_fetched_bgra_attachment_is_viewed_in_its_rgba_twin_where_no_declaration_reaches_it() {
    use crate::metal::ColorWriteMask;
    use crate::pipeline::{StorageAttachmentUse, StorageAttachmentView};
    let (Some(io), Some(fetch)) = (fixture("stage_io"), fixture("fragment_fetch")) else {
        return;
    };
    let device = |stem: &str| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles")
            .join(format!("{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        Device::new(desc, DevicePolicy::default()).expect("device")
    };
    let function = |bc: &[u8], name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        ))
    };
    let descriptor = |colors: &[PixelFormat], samples: u32, mask: ColorWriteMask| {
        let mut d = RenderPipelineDescriptor::new(function(&io, "stage_io_vertex_position"));
        d.fragment = Some(function(&fetch, "fetch_ordered"));
        for (i, &pf) in colors.iter().enumerate() {
            d.color_attachments[i] = Some(ColorAttachmentDescriptor::new(pf));
        }
        if let Some(c) = d.color_attachments[0].as_mut() {
            c.write_mask = mask;
        }
        d.raster_sample_count = samples;
        d
    };
    // The instructions of opcode `op` in `words`, each as its words.
    let instructions = |words: &[u32], op: u32| {
        let mut at = 5;
        let mut found = Vec::new();
        while at < words.len() {
            let count = (words[at] >> 16) as usize;
            if count == 0 || at + count > words.len() {
                break;
            }
            if words[at] & 0xffff == op {
                found.push(words[at..at + count].to_vec());
            }
            at += count;
        }
        found
    };
    // The `OpVectorShuffle`s (79) of four lanes 2, 1, 0, 3: an exchange of red and blue.
    let exchanges = |words: &[u32]| {
        instructions(words, 79)
            .iter()
            .filter(|i| i.len() == 9 && i[5..] == [2, 1, 0, 3])
            .count()
    };
    // Whether `words` declare an `OpTypeImage` (25) of storage (`Sampled` 2) in `format`.
    let declares = |words: &[u32], format: crate::spv::ImageFormat| {
        instructions(words, 25)
            .iter()
            .any(|i| i.len() >= 9 && i[7] == 2 && i[8] == format as u32)
    };
    let consumer = [PixelFormat::BGRA8Unorm, PixelFormat::RGBA16Float];
    let partial = ColorWriteMask::RED.union(ColorWriteMask::BLUE);
    let anv = device("intel-irisxe-anv");
    for (samples, mask) in [
        (1, ColorWriteMask::ALL),
        (4, ColorWriteMask::ALL),
        (1, partial),
    ] {
        let what = format!("{samples}x {mask:?}");
        let p = anv
            .compile_render_pipeline(&descriptor(&consumer, samples, mask), &Budget::standard())
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        let view = match samples {
            1 => StorageAttachmentView::Attachment,
            layers => StorageAttachmentView::SamplePlanes { layers },
        };
        let kept = match mask == partial {
            true => vk::ColorComponentFlags::G | vk::ColorComponentFlags::A,
            false => vk::ColorComponentFlags::empty(),
        };
        let want = StorageAttachmentUse::new(0, vk::Format::B8G8R8A8_UNORM, true, false, view)
            .with_view_alias(vk::Format::R8G8B8A8_UNORM)
            .with_kept(kept);
        assert_eq!(p.graphics.storage_attachments, [want], "{what}");
        assert_eq!(want.view_format(), vk::Format::R8G8B8A8_UNORM);
        let plan = p.plan(&anv, &BindState::default().shape()).expect("plan");
        let bound = plan.descriptors.iter().any(|w| {
            w.source
                == WriteSource::TileAttachment {
                    color_index: 0,
                    format: vk::Format::B8G8R8A8_UNORM,
                    view,
                    view_alias: Some(vk::Format::R8G8B8A8_UNORM),
                }
        });
        assert!(bound, "{what}: {:?}", plan.descriptors);
        let fragment = (p.stages.iter())
            .find(|m| m.stage == vk::ShaderStage::FRAGMENT)
            .expect("a fragment stage");
        let rgba8 = declares(&fragment.spirv, crate::spv::ImageFormat::Rgba8);
        assert!(rgba8, "{what}: no Rgba8 storage image");
        // A read, a write and, under the partial mask, the merge's read of the kept channels.
        let reads = 1 + usize::from(mask == partial);
        assert_eq!(exchanges(&fragment.spirv), reads + 1, "{what}");
        check_words_valid(&fragment.spirv, &p.requirements, anv.profile())
            .unwrap_or_else(|e| panic!("{what}: {e}"));
    }
    // `RGBA16Float` fetched alone: its own format, declared `Rgba16f`.
    let p = anv
        .compile_render_pipeline(
            &descriptor(&[PixelFormat::RGBA16Float], 1, ColorWriteMask::ALL),
            &Budget::standard(),
        )
        .expect("an RGBA16Float fetch");
    let want = StorageAttachmentUse::new(
        0,
        vk::Format::R16G16B16A16_SFLOAT,
        true,
        false,
        StorageAttachmentView::Attachment,
    );
    assert_eq!(p.graphics.storage_attachments, [want]);
    let fragment = (p.stages.iter())
        .find(|m| m.stage == vk::ShaderStage::FRAGMENT)
        .expect("a fragment stage");
    assert!(declares(&fragment.spirv, crate::spv::ImageFormat::Rgba16f));
    assert_eq!(exchanges(&fragment.spirv), 0);
    // A device that reads and writes `B8G8R8A8_UNORM` without a format keeps its own view.
    for stem in ["amd-rdna3-radv", "nvidia-rtx5080-laptop"] {
        let dev = device(stem);
        let p = dev
            .compile_render_pipeline(
                &descriptor(&consumer, 1, ColorWriteMask::ALL),
                &Budget::standard(),
            )
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let want = StorageAttachmentUse::new(
            0,
            vk::Format::B8G8R8A8_UNORM,
            true,
            true,
            StorageAttachmentView::Attachment,
        );
        assert_eq!(p.graphics.storage_attachments, [want], "{stem}");
    }
}

/// In the shapes of two pipelines a guest made: a fetched attachment written under a blend of one
/// and zero, added, for color and alpha (all of `RG16Float`'s channels) beside two
/// `RGB10A8_2P_XR10` attachments of mask none, with early tests (`con_fetch_coverage`); and a
/// fetched attachment written under mask none by a function that discards, beside a depth and
/// stencil attachment, so without early tests (`msf_fetch_discard`). Each takes the storage form on
/// every committed profile that orders fetch through the interlock (a write of mask none stores
/// nothing), passes L3, and the first makes its whole pipeline on `moltenvk-m3`; under a blend that
/// adds the destination too, the function computing Metal's blend of its `half2` output (one and
/// one, added: Metal's `h(s + d)`, measured bit-exact by the narrow sweeps), its
/// fact naming the blend, and passes L3. A stored channel without early tests compiles, stating
/// `TestedStorageWrites`, whose draws' plans check their tests; the write of mask none states
/// nothing of them.
#[test]
fn a_replacing_blend_or_a_write_that_stores_nothing_takes_the_storage_form() {
    use crate::metal::{BlendFactor, ColorWriteMask};
    let (Some(coverage), Some(discard)) = (fixture("con_fetch_coverage"), fixture("msf_fetch_ms"))
    else {
        return;
    };
    // `con_fetch_coverage`'s pipeline: color 1 replaced under `blend`, the others of mask none.
    let replacing = |blend: BlendFactor| {
        let mut unwritten = ColorAttachmentDescriptor::new(PixelFormat::RGB10A82PXR10);
        unwritten.write_mask = ColorWriteMask::NONE;
        let mut replaced = ColorAttachmentDescriptor::new(PixelFormat::RG16Float);
        replaced.blending_enabled = true;
        replaced.destination_rgb_blend_factor = blend;
        replaced.write_mask = ColorWriteMask::RED.union(ColorWriteMask::GREEN);
        vec![Some(unwritten), Some(replaced), Some(unwritten)]
    };
    // `msf_fetch_discard`'s: color 0 under `mask`.
    let discarding = |mask: ColorWriteMask| {
        let mut c = ColorAttachmentDescriptor::new(PixelFormat::RG16Float);
        c.write_mask = mask;
        vec![Some(c)]
    };
    // The pipeline's state for a function that reads color `read`: its color writes and
    // extended-range attachments too, as the render pipeline compile sets them.
    let state = |read: u32, colors: &[Option<ColorAttachmentDescriptor>], p: &DeviceProfile| {
        let mut inputs = LoweringInputs::new();
        inputs.color_writes = crate::pipeline::color_writes::color_writes(
            0..colors.len() as u32,
            colors,
            (false, Default::default()),
            p,
        );
        inputs.extended_range_colors =
            crate::pipeline::color_writes::extended_range_colors(colors, Default::default());
        inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of([read], colors, true, false, 1));
        inputs
    };
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        let one_zero = state(1, &replacing(BlendFactor::Zero), &p);
        let a = match compile_with(&coverage, "con_fetch_coverage", &p, &one_zero) {
            Ok(a) => a,
            // A device without the format's storage refuses the form whatever the blend.
            Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => continue,
            Err(e) => panic!("{stem}: {e}"),
        };
        checked += 1;
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let adds = state(1, &replacing(BlendFactor::One), &p);
        let a = compile_with(&coverage, "con_fetch_coverage", &p, &adds)
            .unwrap_or_else(|e| panic!("{stem}: adds: {e}"));
        let fetched = a.facts.iter().find_map(|f| match f {
            RuntimeFact::InterlockedFramebufferFetch {
                color: 1,
                writes,
                blend,
                ..
            } => Some((*writes, blend.is_some())),
            _ => None,
        });
        assert_eq!(fetched, Some((true, true)), "{stem}: adds: {:?}", a.facts);
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: adds: L3: {e}"));
        let none = state(0, &discarding(ColorWriteMask::NONE), &p);
        let a = compile_with(&discard, "msf_fetch_discard", &p, &none)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert!(
            a.facts.iter().any(|f| matches!(
                f,
                RuntimeFact::InterlockedFramebufferFetch {
                    color: 0,
                    writes: true,
                    early_tests: false,
                    ..
                }
            )),
            "{stem}: {:?}",
            a.facts
        );
        let tested = |a: &crate::library::FunctionArtifact| {
            a.facts.iter().any(|f| {
                matches!(
                    f,
                    RuntimeFact::TestedStorageWrites {
                        early_tests: false,
                        early_variant: true
                    }
                )
            })
        };
        assert!(!tested(&a), "{stem}: {:?}", a.facts);
        let stored = state(0, &discarding(ColorWriteMask::ALL), &p);
        let a = compile_with(&discard, "msf_fetch_discard", &p, &stored)
            .unwrap_or_else(|e| panic!("{stem}: stored: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: stored: L3: {e}"));
        assert!(tested(&a), "{stem}: {:?}", a.facts);
    }
    assert!(
        checked > 0,
        "no interlock profile compiled the storage form"
    );
    // The first pipeline, whole, on `moltenvk-m3`.
    let Some(io) = fixture("stage_io") else {
        return;
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let dev = Device::new(desc, DevicePolicy::default()).expect("device");
    let function = |bc: &[u8], name: &str| {
        FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        )
    };
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(
        &io,
        "stage_io_vertex_position",
    )));
    d.fragment = Some(StageSource::Function(function(
        &coverage,
        "con_fetch_coverage",
    )));
    for (i, c) in replacing(BlendFactor::Zero).into_iter().enumerate() {
        d.color_attachments[i] = c;
    }
    d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
    d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
    dev.compile_render_pipeline(&d, &Budget::standard())
        .expect("the replacing pipeline");
}

/// In the shape of a pipeline a guest made: `con_fetch_coverage` writes a `half4` to color 0, an
/// `RGB10A8_2P_XR10` attachment blended with source factor one and destination factor one minus the
/// source's alpha, added, for color and alpha (premultiplied over), and reads color 1 (`RG16Float`,
/// written under mask none) beside a second `RGB10A8_2P_XR10` attachment of mask none and a depth
/// and stencil attachment. On every committed profile that orders fetch through the interlock the
/// shader blends color 0: it reads the texel, blends the output over its tile value and stores the
/// encode (one image read and one write more than the pipeline unblended), the fact names the
/// blend, and the module passes L3. The whole pipeline is made on `moltenvk-m3`
/// (no vertex descriptor), color 0 a written storage attachment of the raw format, as are its
/// pipelines blending one plus one, one and zero, and source alpha over, and a stage compiled under
/// its state links into no pipeline of another blend there. The `float4` fill
/// (`rt6_fill_rgba32float`) under the premultiplied blend compiles; so do source alpha over and the
/// `half4` fill. The function is built without fast math; the color channels' blend color
/// difference makes the whole pipeline in it and in the same function switched to fast math. The
/// consumer's mask blend (G345: color source factor zero over one minus the source's alpha) is
/// computed in the shader too, for the `half4` output and for the `float4` fill, and makes the
/// whole pipeline on the MoltenVK profile.
/// A fast-math function's source-alpha-over blend (source alpha and one minus it, added, for
/// color and alpha, every channel written) into color 0 of `BGR10_XR_sRGB`,
/// `RGB10A8_2P_XR10_sRGB` or `BGRA10_XR_sRGB`, beside an `R8Unorm` color 1 under the same blend (a
/// consumer's pipeline): on every committed profile that orders fetch through the interlock, the
/// shader blends color 0 by the device's arithmetic, a `half4` output's and a `float4` one's, and
/// the module passes L3.
#[test]
fn a_fast_source_alpha_over_blend_into_an_srgb_extended_range_attachment_is_made() {
    use crate::metal::BlendFactor;
    let Some(fills) = fixture("rt6_samples") else {
        return;
    };
    let colors = |pf: PixelFormat| {
        let over = |pf| {
            let mut c = ColorAttachmentDescriptor::new(pf);
            c.blending_enabled = true;
            c.source_rgb_blend_factor = BlendFactor::SourceAlpha;
            c.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
            c.source_alpha_blend_factor = BlendFactor::SourceAlpha;
            c.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
            c
        };
        vec![Some(over(pf)), Some(over(PixelFormat::R8Unorm))]
    };
    let state = |colors: &[Option<ColorAttachmentDescriptor>], p: &DeviceProfile| {
        let mut inputs = LoweringInputs::new();
        inputs.color_writes = crate::pipeline::color_writes::color_writes(
            0..colors.len() as u32,
            colors,
            (false, Default::default()),
            p,
        );
        inputs.extended_range_colors =
            crate::pipeline::color_writes::extended_range_colors(colors, Default::default());
        inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of([0], colors, true, false, 1));
        inputs
    };
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        for pf in [
            PixelFormat::BGR10XRSRGB,
            PixelFormat::RGB10A82PXR10SRGB,
            PixelFormat::BGRA10XRSRGB,
        ] {
            let inputs = state(&colors(pf), &p);
            for entry in ["rt6_fill_rgba16float", "rt6_fill_rgba32float"] {
                let a = match compile_with(&fills, entry, &p, &inputs) {
                    Ok(a) => a,
                    // A device without the raw formats' storage refuses the form whatever the
                    // blend.
                    Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => continue,
                    Err(e) => panic!("{stem}: {pf:?} {entry}: {e}"),
                };
                link_and_validate(&a, &p)
                    .unwrap_or_else(|e| panic!("{stem}: {pf:?} {entry}: L3: {e}"));
                checked += 1;
            }
        }
    }
    assert!(
        checked > 0,
        "no profile blends an extended-range attachment in the shader"
    );
}

#[test]
fn a_premultiplied_blend_into_an_extended_range_attachment_is_computed_in_the_shader() {
    use crate::metal::format::ExtendedRange;
    use crate::metal::{BlendFactor, BlendOperation, ColorWriteMask};
    use crate::pipeline::lower_inputs::BlendEquation;
    let (Some(coverage), Some(fills)) = (fixture("con_fetch_coverage"), fixture("rt6_samples"))
    else {
        return;
    };
    // The coverage function is built without fast math; the same function under fast math.
    let text = std::fs::read_to_string(engine_dir().join("con_fetch_coverage.ll")).expect("text");
    assert_eq!(text.matches("air.compile.fast_math_disable").count(), 1);
    let fast = text.replace(
        "air.compile.fast_math_disable",
        "air.compile.fast_math_enable",
    );
    let fast_coverage = match metal2vulkan_devtools::llvm_assemble_text(&fast) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the fast coverage: {why}"),
    };
    // The guest's attachments, color 0 blended with source factor `source` for color and
    // destination factor `destination` for color and alpha.
    let colors = |source: BlendFactor, destination: BlendFactor| {
        let mut blended = ColorAttachmentDescriptor::new(PixelFormat::RGB10A82PXR10);
        blended.blending_enabled = true;
        blended.source_rgb_blend_factor = source;
        blended.destination_rgb_blend_factor = destination;
        blended.destination_alpha_blend_factor = destination;
        let mut read = ColorAttachmentDescriptor::new(PixelFormat::RG16Float);
        read.write_mask = ColorWriteMask::NONE;
        let mut unwritten = ColorAttachmentDescriptor::new(PixelFormat::RGB10A82PXR10);
        unwritten.write_mask = ColorWriteMask::NONE;
        vec![Some(blended), Some(read), Some(unwritten)]
    };
    let premul = || colors(BlendFactor::One, BlendFactor::OneMinusSourceAlpha);
    // The color channels' blend color difference, which the shader blends under either math.
    let difference = || {
        let mut c = colors(BlendFactor::BlendColor, BlendFactor::OneMinusBlendColor);
        if let Some(a) = c[0].as_mut() {
            a.rgb_blend_operation = BlendOperation::Subtract;
        }
        c
    };
    // The pipeline's state for a function that reads (or blends) the colors `read`, as the render
    // pipeline compile sets it.
    let state = |read: &[u32], colors: &[Option<ColorAttachmentDescriptor>], p: &DeviceProfile| {
        let mut inputs = LoweringInputs::new();
        inputs.color_writes = crate::pipeline::color_writes::color_writes(
            0..colors.len() as u32,
            colors,
            (false, Default::default()),
            p,
        );
        inputs.extended_range_colors =
            crate::pipeline::color_writes::extended_range_colors(colors, Default::default());
        inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of(
            read.iter().copied(),
            colors,
            true,
            false,
            1,
        ));
        inputs
    };
    let fetched = |a: &crate::library::FunctionArtifact, color: u32| {
        a.facts.iter().find_map(|f| match f {
            RuntimeFact::InterlockedFramebufferFetch {
                color: c,
                writes,
                blend,
                ..
            } if *c == color => Some((*writes, *blend)),
            _ => None,
        })
    };
    let count = |m: &TypedModule| {
        let mut out = (0, 0);
        for b in m.blocks() {
            for n in &b.insts {
                match n.inst {
                    Inst::ImageRead { .. } => out.0 += 1,
                    Inst::ImageWrite { .. } => out.1 += 1,
                    _ => {}
                }
            }
        }
        out
    };
    let equation = BlendEquation {
        source_rgb: BlendFactor::One,
        destination_rgb: BlendFactor::OneMinusSourceAlpha,
        rgb_operation: BlendOperation::Add,
        source_alpha: BlendFactor::One,
        destination_alpha: BlendFactor::OneMinusSourceAlpha,
        alpha_operation: BlendOperation::Add,
    };
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        let a_state = state(&[0, 1], &premul(), &p);
        let a = match compile_with(&coverage, "con_fetch_coverage", &p, &a_state) {
            Ok(a) => a,
            // A device without the raw formats' storage refuses the form whatever the blend.
            Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => continue,
            Err(e) => panic!("{stem}: {e}"),
        };
        checked += 1;
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert_eq!(
            fetched(&a, 0),
            Some((true, Some(equation))),
            "{stem}: {:?}",
            a.facts
        );
        assert_eq!(fetched(&a, 1), Some((true, None)), "{stem}: {:?}", a.facts);
        // Against the same pipeline unblended (color 0 a color attachment, color 1 written
        // nowhere under mask none): color 0's texel is read and written once more.
        let mut unblended = premul();
        if let Some(c) = unblended[0].as_mut() {
            c.blending_enabled = false;
        }
        let plain = state(&[1], &unblended, &p);
        let (_, m) =
            typed_module_with(&coverage, "con_fetch_coverage", &p, &plain).expect("module");
        let (reads, writes) = count(&m);
        let (_, m) =
            typed_module_with(&coverage, "con_fetch_coverage", &p, &a_state).expect("module");
        assert_eq!(count(&m), (reads + 1, writes + 1), "{stem}");
        let over = state(
            &[0, 1],
            &colors(BlendFactor::SourceAlpha, BlendFactor::OneMinusSourceAlpha),
            &p,
        );
        let a = compile_with(&coverage, "con_fetch_coverage", &p, &over)
            .unwrap_or_else(|e| panic!("{stem}: over: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: over: L3: {e}"));
        let differs = state(&[0, 1], &difference(), &p);
        let made = compile_with(&coverage, "con_fetch_coverage", &p, &differs)
            .unwrap_or_else(|e| panic!("{stem}: the blend color difference: {e}"));
        link_and_validate(&made, &p)
            .unwrap_or_else(|e| panic!("{stem}: the blend color difference: L3: {e}"));
        let made = compile_with(&fast_coverage, "con_fetch_coverage", &p, &differs)
            .unwrap_or_else(|e| panic!("{stem}: the blend color difference, fast: {e}"));
        link_and_validate(&made, &p)
            .unwrap_or_else(|e| panic!("{stem}: the blend color difference, fast: L3: {e}"));
        let fill = state(&[0], &premul(), &p);
        let float = compile_with(&fills, "rt6_fill_rgba32float", &p, &fill)
            .unwrap_or_else(|e| panic!("{stem}: float4: {e}"));
        link_and_validate(&float, &p).unwrap_or_else(|e| panic!("{stem}: float4: L3: {e}"));
        let half = compile_with(&fills, "rt6_fill_rgba16float", &p, &fill)
            .unwrap_or_else(|e| panic!("{stem}: half4: {e}"));
        link_and_validate(&half, &p).unwrap_or_else(|e| panic!("{stem}: half4: L3: {e}"));
        let masked = || colors(BlendFactor::Zero, BlendFactor::OneMinusSourceAlpha);
        let mask = state(&[0, 1], &masked(), &p);
        let a = compile_with(&coverage, "con_fetch_coverage", &p, &mask)
            .unwrap_or_else(|e| panic!("{stem}: the mask blend: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: the mask blend: L3: {e}"));
        let mask_equation = BlendEquation {
            source_rgb: BlendFactor::Zero,
            ..equation
        };
        assert_eq!(fetched(&a, 0), Some((true, Some(mask_equation))), "{stem}");
        let fill = state(&[0], &masked(), &p);
        for name in ["rt6_fill_rgba32float", "rt6_fill_rgba16float"] {
            let f = compile_with(&fills, name, &p, &fill)
                .unwrap_or_else(|e| panic!("{stem}: {name} under the mask blend: {e}"));
            link_and_validate(&f, &p)
                .unwrap_or_else(|e| panic!("{stem}: {name} under the mask blend: L3: {e}"));
        }
    }
    assert!(
        checked > 0,
        "no interlock profile compiled the storage form"
    );
    // The whole pipeline on each MoltenVK profile (no vertex descriptor; color 2 of mask none and
    // the depth and stencil attachment in the interlocked pass), and a stage compiled under its
    // state linked into a pipeline of another blend.
    let Some(io) = fixture("stage_io") else {
        return;
    };
    let function = |bc: &[u8], name: &str| {
        FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        )
    };
    {
        let stem = "moltenvk-m3";
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles")
            .join(format!("{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        let dev = Device::new(desc, DevicePolicy::default()).expect("device");
        let pipeline = |fragment: StageSource, colors: Vec<Option<ColorAttachmentDescriptor>>| {
            let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(
                &io,
                "stage_io_vertex_position",
            )));
            d.fragment = Some(fragment);
            for (i, c) in colors.into_iter().enumerate() {
                d.color_attachments[i] = c;
            }
            d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
            d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
            dev.compile_render_pipeline(&d, &Budget::standard())
        };
        let source = StageSource::Function(function(&coverage, "con_fetch_coverage"));
        let pipe = pipeline(source, premul())
            .unwrap_or_else(|e| panic!("{stem}: the guest's pipeline: {e}"));
        let uses: Vec<(u32, vk::Format, bool)> = pipe
            .graphics
            .storage_attachments
            .iter()
            .map(|s| (s.color_index, s.format, s.writes))
            .collect();
        assert_eq!(
            uses,
            [
                (0, ExtendedRange::Planes { srgb: false }.raw_format(), true),
                (1, vk::Format::R16G16_SFLOAT, true),
            ],
            "{stem}"
        );
        let fragment = dev
            .compile_function(
                &function(&coverage, "con_fetch_coverage"),
                &state(&[0, 1], &premul(), dev.profile()),
                &Budget::standard(),
            )
            .expect("fragment");
        let added = colors(BlendFactor::One, BlendFactor::One);
        assert_eq!(
            pipeline(StageSource::Compiled(fragment), added)
                .map(|_| ())
                .unwrap_err(),
            Error::NeedsInput(RuntimeInput::FramebufferFetch),
            "{stem}"
        );
        // Each blend the shader computes makes the whole pipeline, the color channels' blend color
        // difference under either math.
        for (src, dst) in [
            (BlendFactor::One, BlendFactor::One),
            (BlendFactor::One, BlendFactor::Zero),
            (BlendFactor::Zero, BlendFactor::OneMinusSourceAlpha),
            (BlendFactor::SourceAlpha, BlendFactor::OneMinusSourceAlpha),
        ] {
            let source = StageSource::Function(function(&coverage, "con_fetch_coverage"));
            pipeline(source, colors(src, dst))
                .unwrap_or_else(|e| panic!("{stem}: {src:?}, {dst:?}: {e}"));
        }
        let source = StageSource::Function(function(&coverage, "con_fetch_coverage"));
        pipeline(source, difference())
            .unwrap_or_else(|e| panic!("{stem}: the blend color difference: {e}"));
        let source = StageSource::Function(function(&fast_coverage, "con_fetch_coverage"));
        pipeline(source, difference())
            .unwrap_or_else(|e| panic!("{stem}: the blend color difference, fast: {e}"));
    }
}

/// Without fast math, a blend of the second source into an extended-range attachment reads the
/// function's index-1 output in the shader. `stage_io`'s dual-source function, its
/// `air.compile.fast_math_enable` switched to `_disable`, makes the whole pipeline on `moltenvk-m3`
/// into an `RGB10A8_2P_XR10` color 0 blended with the second source's factors in
/// color and alpha: color 0 a written storage attachment, the fact naming the blend and its rounded
/// math, and no dual-source output (`dualSrcBlend` not required); under a write mask of no channel
/// it is made too, the second output stored nowhere. Under fast math the same pipeline is made too;
/// the blend of a function that writes no second output is refused as Metal refuses its pipeline,
/// `InvalidDescriptor(ColorAttachment)`.
/// Into each of the six extended-range formats, unblended (the raw texel through the interface) or
/// blended premultiplied over (the storage form's blend, which reads no second source), with and
/// without fast math, the function's two outputs make the pipeline, the second stored nowhere: no
/// dual-source output.
#[test]
fn a_second_source_blend_without_fast_math_reads_the_second_output() {
    use crate::metal::format::ExtendedRange;
    use crate::metal::BlendFactor;
    use crate::pipeline::lower_inputs::BlendEquation;
    use metal2vulkan_devtools::AssembleError;
    let Some(fast) = fixture("stage_io") else {
        return;
    };
    let text = std::fs::read_to_string(engine_dir().join("stage_io.ll")).expect("text");
    let enable = "air.compile.fast_math_enable";
    assert_eq!(text.matches(enable).count(), 1);
    let text = text.replace(enable, "air.compile.fast_math_disable");
    let safe = match metal2vulkan_devtools::llvm_assemble_text(&text) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the safe stage_io: {why}"),
    };
    let function = |bc: &[u8], name: &str| {
        FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        )
    };
    let equation = BlendEquation {
        source_rgb: BlendFactor::Source1Color,
        destination_rgb: BlendFactor::OneMinusSource1Alpha,
        rgb_operation: crate::metal::BlendOperation::Add,
        source_alpha: BlendFactor::Source1Alpha,
        destination_alpha: BlendFactor::OneMinusSource1Color,
        alpha_operation: crate::metal::BlendOperation::Add,
    };
    let mut blended = ColorAttachmentDescriptor::new(PixelFormat::RGB10A82PXR10);
    blended.blending_enabled = true;
    blended.source_rgb_blend_factor = equation.source_rgb;
    blended.destination_rgb_blend_factor = equation.destination_rgb;
    blended.source_alpha_blend_factor = equation.source_alpha;
    blended.destination_alpha_blend_factor = equation.destination_alpha;
    {
        let stem = "moltenvk-m3";
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles")
            .join(format!("{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        let dev = Device::new(desc, DevicePolicy::default()).expect("device");
        let pipeline = |bc: &[u8], fragment: &str| {
            let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(
                bc,
                "stage_io_vertex_position",
            )));
            d.fragment = Some(StageSource::Function(function(bc, fragment)));
            d.color_attachments[0] = Some(blended);
            dev.compile_render_pipeline(&d, &Budget::standard())
        };
        let pipe = pipeline(&safe, "stage_io_fragment_dual_source")
            .unwrap_or_else(|e| panic!("{stem}: the second source without fast math: {e}"));
        let uses: Vec<(u32, vk::Format, bool)> = pipe
            .graphics
            .storage_attachments
            .iter()
            .map(|s| (s.color_index, s.format, s.writes))
            .collect();
        assert_eq!(
            uses,
            [(0, ExtendedRange::Planes { srgb: false }.raw_format(), true)],
            "{stem}"
        );
        assert!(
            pipe.dependencies.iter().any(|d| matches!(
                d.fact,
                RuntimeFact::InterlockedFramebufferFetch {
                    color: 0,
                    writes: true,
                    blend: Some(b),
                    rounded: true,
                    ..
                } if b == equation
            )),
            "{stem}: {:?}",
            pipe.dependencies
        );
        // No dual-source output: the shader reads the second output itself.
        assert!(
            !pipe
                .requirements
                .features
                .contains(&VkFeature::DualSrcBlend),
            "{stem}"
        );
        // A write of none of the attachment's channels stores nothing, and reads no second
        // output: made, with no dual-source output either.
        let none = {
            let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(
                &safe,
                "stage_io_vertex_position",
            )));
            d.fragment = Some(StageSource::Function(function(
                &safe,
                "stage_io_fragment_dual_source",
            )));
            let mut unwritten = blended;
            unwritten.write_mask = crate::metal::ColorWriteMask::NONE;
            d.color_attachments[0] = Some(unwritten);
            dev.compile_render_pipeline(&d, &Budget::standard())
                .unwrap_or_else(|e| panic!("{stem}: the second source under no channel: {e}"))
        };
        assert!(
            !none
                .requirements
                .features
                .contains(&VkFeature::DualSrcBlend),
            "{stem}: no channel"
        );
        let fast_pipe = pipeline(&fast, "stage_io_fragment_dual_source")
            .unwrap_or_else(|e| panic!("{stem}: the second source under fast math: {e}"));
        assert!(
            !fast_pipe
                .requirements
                .features
                .contains(&VkFeature::DualSrcBlend),
            "{stem}: fast math"
        );
        match pipeline(&safe, "stage_io_fragment_point") {
            Err(Error::InvalidDescriptor(crate::error::DescriptorError::ColorAttachment {
                index: 0,
            })) => {}
            r => panic!("{stem}: no second output: {:?}", r.map(|_| ())),
        }
        // A second output no blend reads, into each extended-range format: stored nowhere.
        for raw in 550..=555u32 {
            let pf = PixelFormat::try_from(raw).expect("format");
            for (bc, math) in [(&safe, "safe"), (&fast, "fast")] {
                for blending in [false, true] {
                    let mut color = ColorAttachmentDescriptor::new(pf);
                    color.blending_enabled = blending;
                    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
                    color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
                    let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(
                        bc,
                        "stage_io_vertex_position",
                    )));
                    d.fragment = Some(StageSource::Function(function(
                        bc,
                        "stage_io_fragment_dual_source",
                    )));
                    d.color_attachments[0] = Some(color);
                    let what = format!("{stem}: {pf:?} {math} blended {blending}");
                    let pipe = dev
                        .compile_render_pipeline(&d, &Budget::standard())
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert!(
                        !pipe
                            .requirements
                            .features
                            .contains(&VkFeature::DualSrcBlend),
                        "{what}"
                    );
                }
            }
        }
    }
}

/// On a device that orders framebuffer fetch through rasterization-order input attachments (every
/// committed profile with that access and no ordering through the pixel interlock alone), the
/// function computes an extended-range attachment's blend over its input attachment. Into each of
/// the six forms, under the premultiplied blend (source one, destination one minus the source's
/// alpha), one plus one, and one and zero, `rt6_samples`' `half4` and `float4` fills make the
/// pipeline exactly where `moltenvk-m3`, which blends in the pixel interlock, makes it, and refuse
/// it by the same gap elsewhere. A made pipeline's color 0 is a color attachment of the raw format
/// with no Vulkan blend and a write of every raw component, read through an input attachment of
/// `uint` texels, and states the blend (`RuntimeFact::ExtendedRangeInputBlend`, keeping no channel)
/// beside the fetch; the function compiled under the pipeline's state passes L3 and its plan bounds
/// its emission. A write mask of the color channels alone keeps alpha: made, the fact naming it,
/// every raw component written. A mask of no channel stores nothing: made, written nowhere and read
/// through no input attachment. A function compiled under one blend links into no pipeline of
/// another (`NeedsInput(FramebufferFetch)`). With and without fast math the second source's blend
/// reads `stage_io`'s index-1 output: made, with no dual-source output. Every made pipeline records
/// the blend-source notes the interlock's records, `xrb_computed`'s whole-vector `float4` among
/// them.
#[test]
fn an_extended_range_blend_through_input_attachments_is_computed_in_the_shader() {
    use crate::metal::format::ExtendedRange;
    use crate::metal::{BlendFactor, ColorWriteMask};
    use crate::pipeline::lower_inputs::BlendEquation;
    use crate::pipeline::sym::SampledType;
    let (Some(fills), Some(fast), Some(computed)) = (
        fixture("rt6_samples"),
        fixture("stage_io"),
        fixture("xrb_computed"),
    ) else {
        return;
    };
    let text = std::fs::read_to_string(engine_dir().join("stage_io.ll")).expect("text");
    let enable = "air.compile.fast_math_enable";
    assert_eq!(text.matches(enable).count(), 1);
    let safe = match metal2vulkan_devtools::llvm_assemble_text(
        &text.replace(enable, "air.compile.fast_math_disable"),
    ) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the safe stage_io: {why}"),
    };
    let function = |bc: &[u8], name: &str| {
        FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        )
    };
    let device = |stem: &str| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles")
            .join(format!("{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        Device::new(desc, DevicePolicy::default()).expect("device")
    };
    let interlock = device("moltenvk-m3");
    // Color 0 of `pf`, blended with source factor one and destination factor `destination`
    // (color and alpha alike), under `mask`.
    let color = |pf: PixelFormat, destination: BlendFactor, mask: ColorWriteMask| {
        let mut c = ColorAttachmentDescriptor::new(pf);
        c.blending_enabled = true;
        c.destination_rgb_blend_factor = destination;
        c.destination_alpha_blend_factor = destination;
        c.write_mask = mask;
        c
    };
    let pipeline = |dev: &Device, (bc, vertex, fragment): (&[u8], &str, &str), c| {
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(bc, vertex)));
        d.fragment = Some(StageSource::Function(function(bc, fragment)));
        d.color_attachments[0] = Some(c);
        dev.compile_render_pipeline(&d, &Budget::standard())
    };
    // The pipeline's state for the fill, as the render pipeline compile sets it.
    let state = |c: ColorAttachmentDescriptor, p: &DeviceProfile| {
        let colors = [Some(c)];
        let mut inputs = LoweringInputs::new();
        inputs.color_writes = crate::pipeline::color_writes::color_writes(
            0..1,
            &colors,
            (false, Default::default()),
            p,
        );
        inputs.extended_range_colors =
            crate::pipeline::color_writes::extended_range_colors(&colors, Default::default());
        inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of([0], &colors, false, false, 1));
        inputs
    };
    // The pipeline on `dev` where `moltenvk-m3` makes it too; `None` where both refuse it by the
    // same gap of Metal's model.
    // The blend-source notes a pipeline's stages record, which follow the blend whatever form
    // computes it.
    let sources = |a: &crate::pipeline::artifact::RenderPipeline| {
        a.stages
            .iter()
            .flat_map(|s| &s.precision_notes)
            .filter(|n| {
                matches!(
                    n,
                    PrecisionNote::ExtendedRangeBlendSource
                        | PrecisionNote::ExtendedRangeFloatBlendSource
                )
            })
            .copied()
            .collect::<Vec<_>>()
    };
    let both = |dev: &Device, fragment: (&[u8], &str, &str), c, what: &str| match (
        pipeline(dev, fragment, c),
        pipeline(&interlock, fragment, c),
    ) {
        (Ok(a), Ok(theirs)) => {
            assert_eq!(sources(&a), sources(&theirs), "{what}: the source notes");
            Some(a)
        }
        (Err(e), Err(theirs)) => {
            assert_eq!(e, theirs, "{what}");
            assert!(
                matches!(
                    e,
                    Error::Unsupported(ref u)
                        if matches!(u.feature, Feature::ExtendedRangeBlend(_))
                ),
                "{what}: {e}"
            );
            None
        }
        (made, theirs) => panic!(
            "{what}: here {:?}, in the interlock {:?}",
            made.map(|_| ()),
            theirs.map(|_| ())
        ),
    };
    let blend_of = |a: &crate::pipeline::artifact::RenderPipeline| {
        a.dependencies
            .iter()
            .filter_map(|d| match d.fact {
                RuntimeFact::ExtendedRangeInputBlend { color, blend, kept } => {
                    Some((color, blend, kept))
                }
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let fetches = |a: &crate::pipeline::artifact::RenderPipeline| {
        a.dependencies
            .iter()
            .any(|d| d.fact == RuntimeFact::FramebufferFetch { color: 0 })
    };
    use vk::ColorComponentFlags as C;
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let f = p.fields().fragment;
        if !f.rasterization_order_color_access || crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        let dev = device(&stem);
        for raw in 550..=555u32 {
            let pf = PixelFormat::try_from(raw).expect("format");
            let form = ExtendedRange::of(pf).expect("an extended-range format");
            let written = match form.raw_components() {
                1 => C::R,
                2 => C::R | C::G,
                _ => C::R | C::G | C::B | C::A,
            };
            for destination in [
                BlendFactor::OneMinusSourceAlpha,
                BlendFactor::One,
                BlendFactor::Zero,
            ] {
                let c = color(pf, destination, ColorWriteMask::ALL);
                let equation = BlendEquation::of(&c).expect("a blend");
                // The computed `float4` (whole-vector arithmetic over loads under fast math): its
                // source notes as the interlock's.
                let fragment = (
                    computed.as_slice(),
                    "xrb_computed_vertex",
                    "xrb_computed_float",
                );
                both(
                    &dev,
                    fragment,
                    c,
                    &format!("{stem}: {pf:?} {destination:?} computed"),
                );
                for fill in ["rt6_fill_rgba16float", "rt6_fill_rgba32float"] {
                    let what = format!("{stem}: {pf:?} {destination:?} {fill}");
                    checked += 1;
                    let fragment = (fills.as_slice(), "rt6_fill_vertex", fill);
                    let Some(a) = both(&dev, fragment, c, &what) else {
                        continue;
                    };
                    let g = &a.graphics;
                    let state0 = g.color_attachments[0].expect("color 0");
                    assert_eq!(state0.format, form.raw_format(), "{what}");
                    assert_eq!(state0.write_mask, written, "{what}");
                    assert!(state0.blend.is_none(), "{what}");
                    let inputs: Vec<(u32, SampledType)> = g
                        .input_attachments
                        .iter()
                        .map(|i| (i.color_index, i.format_class))
                        .collect();
                    assert_eq!(inputs, [(0, SampledType::Uint)], "{what}");
                    assert!(g.storage_attachments.is_empty(), "{what}");
                    assert_eq!(blend_of(&a), [(0, equation, C::empty())], "{what}");
                    assert!(fetches(&a), "{what}");
                    // The function under the pipeline's state: L3, and its plan's bound.
                    let inputs = state(c, &p);
                    let artifact = compile_with(&fills, fill, &p, &inputs)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    link_and_validate(&artifact, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                    let consts = crate::library::FunctionConstantValues::defaults();
                    let plan = crate::pipeline::compile::plan_bitcode(
                        &fills,
                        crate::pipeline::compile::EntryRef::Name(fill),
                        &consts,
                        &p,
                        &inputs,
                        &mut crate::pipeline::testing::meter(),
                    )
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert!(
                        plan.estimate.words >= artifact.module.words.len() as u64,
                        "{what}: estimate {} < emitted {}",
                        plan.estimate.words,
                        artifact.module.words.len()
                    );
                }
            }
            let what = format!("{stem}: {pf:?}");
            let fragment = (fills.as_slice(), "rt6_fill_vertex", "rt6_fill_rgba16float");
            // The color channels alone: alpha kept where the format has it.
            let rgb = ColorWriteMask::RED
                .union(ColorWriteMask::GREEN)
                .union(ColorWriteMask::BLUE);
            let c = color(pf, BlendFactor::OneMinusSourceAlpha, rgb);
            if let Some(a) = both(&dev, fragment, c, &format!("{what}: the color channels")) {
                let kept = match form.channels().contains(ColorWriteMask::ALPHA) {
                    true => C::A,
                    false => C::empty(),
                };
                assert_eq!(
                    blend_of(&a).iter().map(|b| b.2).collect::<Vec<_>>(),
                    [kept],
                    "{what}"
                );
                assert_eq!(
                    a.graphics.color_attachments[0].map(|s| s.write_mask),
                    Some(written),
                    "{what}"
                );
            }
            // No channel: stored nowhere, read through no input attachment.
            let c = color(pf, BlendFactor::OneMinusSourceAlpha, ColorWriteMask::NONE);
            let a = both(&dev, fragment, c, &format!("{what}: no channel"))
                .unwrap_or_else(|| panic!("{what}: no channel refused"));
            assert_eq!(
                blend_of(&a).iter().map(|b| b.2).collect::<Vec<_>>(),
                [C::R | C::G | C::B | C::A],
                "{what}"
            );
            assert_eq!(
                a.graphics.color_attachments[0].map(|s| (s.write_mask, s.blend.is_none())),
                Some((C::empty(), true)),
                "{what}"
            );
            assert!(a.graphics.input_attachments.is_empty(), "{what}");
            assert!(!fetches(&a), "{what}");
        }
        // Compiled under the premultiplied blend, linked into one plus one.
        let premul = color(
            PixelFormat::RGB10A82PXR10,
            BlendFactor::OneMinusSourceAlpha,
            ColorWriteMask::ALL,
        );
        let compiled = dev
            .compile_function(
                &function(&fills, "rt6_fill_rgba16float"),
                &state(premul, dev.profile()),
                &Budget::standard(),
            )
            .expect("fragment");
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(
            &fills,
            "rt6_fill_vertex",
        )));
        d.fragment = Some(StageSource::Compiled(compiled));
        d.color_attachments[0] = Some(color(
            PixelFormat::RGB10A82PXR10,
            BlendFactor::One,
            ColorWriteMask::ALL,
        ));
        assert_eq!(
            dev.compile_render_pipeline(&d, &Budget::standard())
                .map(|_| ())
                .unwrap_err(),
            Error::NeedsInput(RuntimeInput::FramebufferFetch),
            "{stem}"
        );
        // The second source's blend without fast math reads the index-1 output.
        let mut dual = ColorAttachmentDescriptor::new(PixelFormat::RGB10A82PXR10);
        dual.blending_enabled = true;
        dual.source_rgb_blend_factor = BlendFactor::Source1Color;
        dual.destination_rgb_blend_factor = BlendFactor::OneMinusSource1Alpha;
        dual.source_alpha_blend_factor = BlendFactor::Source1Alpha;
        dual.destination_alpha_blend_factor = BlendFactor::OneMinusSource1Color;
        let fragment = (
            safe.as_slice(),
            "stage_io_vertex_position",
            "stage_io_fragment_dual_source",
        );
        let a = pipeline(&dev, fragment, dual)
            .unwrap_or_else(|e| panic!("{stem}: the second source without fast math: {e}"));
        assert_eq!(blend_of(&a).len(), 1, "{stem}");
        assert!(
            !a.requirements.features.contains(&VkFeature::DualSrcBlend),
            "{stem}"
        );
        let fragment = (
            fast.as_slice(),
            "stage_io_vertex_position",
            "stage_io_fragment_dual_source",
        );
        let a = pipeline(&dev, fragment, dual)
            .unwrap_or_else(|e| panic!("{stem}: the second source under fast math: {e}"));
        assert_eq!(blend_of(&a).len(), 1, "{stem}: fast math");
    }
    assert!(checked > 0, "no profile fetches through input attachments");
}

/// In the shape of a pipeline a guest made: a function that reads color 0, an `RGB10A8_2P_XR10`
/// attachment, and writes it back under every channel with blending disabled, beside an `RG16Float`
/// color 1 and a second `RGB10A8_2P_XR10` attachment, both of mask none, and a depth and stencil
/// attachment. On every committed profile that orders fetch through the interlock the storage form
/// takes color 0: the write stores Metal's unblended write of the output (`exf_fetch_blend`'s
/// `float4`, and `exf_fetch_three`'s `float4` and `half4` to two such attachments, the `_sRGB` twin
/// among them), the fact names no blend, and the module passes L3. A mask of some of the channels
/// is refused `NotImplemented`; a blend of the `float4` output is the device's arithmetic. The whole
/// pipeline is made on `moltenvk-m3` (no vertex descriptor), color 0 a written
/// storage attachment of the raw format, and a stage compiled under it links into no pipeline that
/// blends color 0.
#[test]
fn a_fetched_extended_range_attachment_is_written_unblended_in_the_shader() {
    use crate::metal::format::ExtendedRange;
    use crate::metal::{BlendFactor, ColorWriteMask};
    let (Some(blend), Some(three)) = (fixture("exf_fetch_blend"), fixture("exf_fetch_three"))
    else {
        return;
    };
    // The guest's attachments: color 0 of `pf`, unblended under `mask`, beside the two of mask
    // none.
    let colors = |pf: PixelFormat, mask: ColorWriteMask| {
        let mut fetched = ColorAttachmentDescriptor::new(pf);
        fetched.write_mask = mask;
        let mut read = ColorAttachmentDescriptor::new(PixelFormat::RG16Float);
        read.write_mask = ColorWriteMask::NONE;
        let mut unwritten = ColorAttachmentDescriptor::new(PixelFormat::RGB10A82PXR10);
        unwritten.write_mask = ColorWriteMask::NONE;
        vec![Some(fetched), Some(read), Some(unwritten)]
    };
    let guest = || colors(PixelFormat::RGB10A82PXR10, ColorWriteMask::ALL);
    let state = |read: &[u32], colors: &[Option<ColorAttachmentDescriptor>], p: &DeviceProfile| {
        let mut inputs = LoweringInputs::new();
        inputs.color_writes = crate::pipeline::color_writes::color_writes(
            0..colors.len() as u32,
            colors,
            (false, Default::default()),
            p,
        );
        inputs.extended_range_colors =
            crate::pipeline::color_writes::extended_range_colors(colors, Default::default());
        inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of(
            read.iter().copied(),
            colors,
            true,
            false,
            1,
        ));
        inputs
    };
    let ni =
        |r: Result<std::sync::Arc<crate::library::FunctionArtifact>, Error>, f: Feature| match r {
            Err(Error::Unsupported(u)) => (u.feature, u.needs) == (f, Needs::NotImplemented),
            _ => false,
        };
    let fetched = |a: &crate::library::FunctionArtifact, color: u32| {
        a.facts.iter().find_map(|f| match f {
            RuntimeFact::InterlockedFramebufferFetch {
                color: c,
                writes,
                blend,
                ..
            } if *c == color => Some((*writes, *blend)),
            _ => None,
        })
    };
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        let a = match compile_with(&blend, "exf_fetch_blend", &p, &state(&[0], &guest(), &p)) {
            Ok(a) => a,
            // A device without the raw formats' storage refuses the form.
            Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => continue,
            Err(e) => panic!("{stem}: {e}"),
        };
        checked += 1;
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert_eq!(fetched(&a, 0), Some((true, None)), "{stem}: {:?}", a.facts);
        // Two such attachments, a `float4` to the `_sRGB` twin and a `half4`, beside an integer
        // color.
        let mut both = colors(PixelFormat::RGB10A82PXR10SRGB, ColorWriteMask::ALL);
        both[1] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
        both[2] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGB10A82PXR10));
        let a = compile_with(&three, "exf_fetch_three", &p, &state(&[0, 1, 2], &both, &p))
            .unwrap_or_else(|e| panic!("{stem}: three: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: three: L3: {e}"));
        assert_eq!(fetched(&a, 0), Some((true, None)), "{stem}: {:?}", a.facts);
        assert_eq!(fetched(&a, 2), Some((true, None)), "{stem}: {:?}", a.facts);
        // A mask of some of the channels; a `float4` output's blend, its own model.
        let some = colors(PixelFormat::RGB10A82PXR10, ColorWriteMask::RED);
        assert!(
            ni(
                compile_with(&blend, "exf_fetch_blend", &p, &state(&[0], &some, &p)),
                Feature::FramebufferFetch
            ),
            "{stem}: a mask of red"
        );
        let mut over = guest();
        if let Some(c) = over[0].as_mut() {
            c.blending_enabled = true;
            c.source_rgb_blend_factor = BlendFactor::SourceAlpha;
            c.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        }
        let a = compile_with(&blend, "exf_fetch_blend", &p, &state(&[0], &over, &p))
            .unwrap_or_else(|e| panic!("{stem}: over: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: over: L3: {e}"));
    }
    assert!(
        checked > 0,
        "no interlock profile compiled the storage form"
    );
    let Some(io) = fixture("stage_io") else {
        return;
    };
    let function = |bc: &[u8], name: &str| {
        FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        )
    };
    {
        let stem = "moltenvk-m3";
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles")
            .join(format!("{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        let dev = Device::new(desc, DevicePolicy::default()).expect("device");
        let pipeline = |fragment: StageSource, colors: Vec<Option<ColorAttachmentDescriptor>>| {
            let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(
                &io,
                "stage_io_vertex_position",
            )));
            d.fragment = Some(fragment);
            for (i, c) in colors.into_iter().enumerate() {
                d.color_attachments[i] = c;
            }
            d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
            d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
            dev.compile_render_pipeline(&d, &Budget::standard())
        };
        let source = StageSource::Function(function(&blend, "exf_fetch_blend"));
        let pipe = pipeline(source, guest())
            .unwrap_or_else(|e| panic!("{stem}: the guest's pipeline: {e}"));
        let uses: Vec<(u32, vk::Format, bool)> = pipe
            .graphics
            .storage_attachments
            .iter()
            .map(|s| (s.color_index, s.format, s.writes))
            .collect();
        assert_eq!(
            uses,
            [(0, ExtendedRange::Planes { srgb: false }.raw_format(), true)],
            "{stem}"
        );
        let fragment = dev
            .compile_function(
                &function(&blend, "exf_fetch_blend"),
                &state(&[0], &guest(), dev.profile()),
                &Budget::standard(),
            )
            .expect("fragment");
        let mut added = guest();
        if let Some(c) = added[0].as_mut() {
            c.blending_enabled = true;
            c.destination_rgb_blend_factor = BlendFactor::One;
            c.destination_alpha_blend_factor = BlendFactor::One;
        }
        assert_eq!(
            pipeline(StageSource::Compiled(fragment), added)
                .map(|_| ())
                .unwrap_err(),
            Error::NeedsInput(RuntimeInput::FramebufferFetch),
            "{stem}"
        );
    }
}

/// In the shape of a pipeline a guest made: `tsw_fetch_discard` writes a `half4` to color 0, an
/// `RGB10A8_2P_XR10` attachment premultiplied over (which the shader blends), reads color 1
/// (`RG16Float`, written under mask none) beside a second `RGB10A8_2P_XR10` attachment of mask none
/// and a depth and stencil attachment, and discards, without early tests. On every committed
/// profile that orders fetch through the interlock the function compiles, its color 0 write before
/// the tests, and states `TestedStorageWrites` without early tests; under `read_only_tests` it
/// declares early tests and states the fact with them; each module passes L3. `con_fetch_coverage`,
/// which declares early tests, states no such fact. On `moltenvk-m3` the whole pipeline is made; a
/// draw whose tests drop no fragment runs it as compiled, one whose tests drop fragments but write
/// and count nothing asks for the early-tests variant, whose plan states the same key, and any
/// other (or one that states nothing) is refused `NotImplemented(FramebufferFetch)`.
#[test]
fn a_discarding_fetch_beside_depth_stores_before_the_tests_its_draws_check() {
    use crate::bind::FragmentTests;
    use crate::metal::{BlendFactor, ColorWriteMask, CompareFunction, StencilOperation};
    let (Some(tested), Some(early), Some(io)) = (
        fixture("tsw_fetch_discard"),
        fixture("con_fetch_coverage"),
        fixture("stage_io"),
    ) else {
        return;
    };
    let colors = || {
        let mut blended = ColorAttachmentDescriptor::new(PixelFormat::RGB10A82PXR10);
        blended.blending_enabled = true;
        blended.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        blended.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        let mut read = ColorAttachmentDescriptor::new(PixelFormat::RG16Float);
        read.write_mask = ColorWriteMask::NONE;
        let mut unwritten = ColorAttachmentDescriptor::new(PixelFormat::RGB10A82PXR10);
        unwritten.write_mask = ColorWriteMask::NONE;
        vec![Some(blended), Some(read), Some(unwritten)]
    };
    // The pipeline's state, as the render pipeline compile sets it.
    let state = |p: &DeviceProfile, read_only: bool| {
        let colors = colors();
        let mut inputs = LoweringInputs::new();
        inputs.color_writes = crate::pipeline::color_writes::color_writes(
            0..colors.len() as u32,
            &colors,
            (false, Default::default()),
            p,
        );
        inputs.extended_range_colors =
            crate::pipeline::color_writes::extended_range_colors(&colors, Default::default());
        inputs.framebuffer_fetch =
            Some(FramebufferFetchInputs::of([0, 1], &colors, true, false, 1));
        inputs.read_only_tests = read_only;
        inputs
    };
    let facts = |a: &crate::library::FunctionArtifact| {
        let fetched: Vec<(u32, bool, bool)> = a
            .facts
            .iter()
            .filter_map(|f| match *f {
                RuntimeFact::InterlockedFramebufferFetch {
                    color,
                    writes,
                    early_tests,
                    ..
                } => Some((color, writes, early_tests)),
                _ => None,
            })
            .collect();
        let tested = a.facts.iter().find_map(|f| match *f {
            RuntimeFact::TestedStorageWrites {
                early_tests,
                early_variant: true,
            } => Some(early_tests),
            _ => None,
        });
        (fetched, tested)
    };
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        for read_only in [false, true] {
            let a = match compile_with(&tested, "tsw_fetch_discard", &p, &state(&p, read_only)) {
                Ok(a) => a,
                // A device without the raw formats' storage refuses the form whatever the tests.
                Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => continue,
                Err(e) => panic!("{stem} read-only {read_only}: {e}"),
            };
            checked += 1;
            link_and_validate(&a, &p)
                .unwrap_or_else(|e| panic!("{stem} read-only {read_only}: L3: {e}"));
            assert_eq!(
                facts(&a),
                (
                    vec![(0, true, read_only), (1, true, read_only)],
                    Some(read_only)
                ),
                "{stem} read-only {read_only}"
            );
            let (_, m) = typed_module_with(&tested, "tsw_fetch_discard", &p, &state(&p, read_only))
                .expect("module");
            assert_eq!(
                m.entry().modes.contains(&ExecMode::EarlyFragmentTests),
                read_only,
                "{stem} read-only {read_only}"
            );
        }
        if let Ok(a) = compile_with(&early, "con_fetch_coverage", &p, &state(&p, false)) {
            assert_eq!(facts(&a).1, None, "{stem}: declared early tests");
        }
    }
    assert!(
        checked > 0,
        "no interlock profile compiled the storage form"
    );
    // The whole pipeline on `moltenvk-m3`, and its draws' plans.
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let dev = Device::new(desc, DevicePolicy::default()).expect("device");
    let function = |bc: &[u8], name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(function(&io, "stage_io_vertex_position"));
    d.fragment = Some(function(&tested, "tsw_fetch_discard"));
    for (i, c) in colors().into_iter().enumerate() {
        d.color_attachments[i] = c;
    }
    d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
    d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
    let base = dev
        .compile_render_pipeline(&d, &Budget::standard())
        .expect("the guest's pipeline");
    assert!(base.dependencies.iter().any(|x| x.fact
        == RuntimeFact::TestedStorageWrites {
            early_tests: false,
            early_variant: true
        }));
    // The draw's tests: depth `Always`; a stencil `Equal` that keeps the stencil; and a depth
    // `Less` that writes.
    let always = crate::metal::state::DepthStencilDescriptor::default();
    let mut equal = always;
    for face in [&mut equal.front_face_stencil, &mut equal.back_face_stencil] {
        face.stencil_compare_function = CompareFunction::Equal;
    }
    let mut less = always;
    less.depth_compare_function = CompareFunction::Less;
    less.depth_write_enabled = true;
    let mut replaced = equal;
    replaced.front_face_stencil.depth_stencil_pass_operation = StencilOperation::Replace;
    let tests = |ds: &crate::metal::state::DepthStencilDescriptor, counted: bool| {
        FragmentTests::of(ds, true, true, counted)
    };
    assert_eq!(tests(&always, true), FragmentTests::Pass);
    assert_eq!(tests(&less, false), FragmentTests::Depth { writes: true });
    assert_eq!(tests(&less, true), FragmentTests::Any);
    assert_eq!(tests(&equal, false), FragmentTests::Drop);
    assert_eq!(tests(&equal, true), FragmentTests::Any);
    assert_eq!(tests(&replaced, false), FragmentTests::Stencil);
    assert_eq!(tests(&replaced, true), FragmentTests::Any);
    // Without a stencil attachment the stencil drops nothing.
    assert_eq!(
        FragmentTests::of(&equal, true, false, false),
        FragmentTests::Pass
    );
    let shape = |fragment_tests: FragmentTests| {
        BindState {
            fragment_tests,
            ..Default::default()
        }
        .shape()
    };
    let pass = base.plan(&dev, &shape(FragmentTests::Pass)).expect("pass");
    assert!(!pass.variant.read_only_tests);
    let drop = base.plan(&dev, &shape(FragmentTests::Drop)).expect("drop");
    assert!(drop.variant.read_only_tests);
    for refused in [FragmentTests::Any, FragmentTests::default()] {
        assert_eq!(
            base.plan(&dev, &shape(refused)).map(|_| ()),
            Err(BindError::Unsupported(
                crate::error::Unsupported::not_implemented(Feature::FramebufferFetch)
            )),
            "{refused:?}"
        );
    }
    let variant = dev
        .compile_render_variant(&d, &drop.variant, &Budget::standard())
        .expect("the early-tests variant");
    assert!(variant.dependencies.iter().any(|x| x.fact
        == RuntimeFact::TestedStorageWrites {
            early_tests: true,
            early_variant: true
        }));
    for m in &variant.stages {
        check_words_valid(&m.spirv, &variant.requirements, dev.profile())
            .unwrap_or_else(|e| panic!("L3: {e}"));
    }
    let again = variant
        .plan(&dev, &shape(FragmentTests::Drop))
        .expect("variant plan");
    assert_eq!(again.variant, drop.variant, "a fixed point");
    // A depth test that writes nothing keeps the early-tests variant; one that writes runs the
    // variant whose function tests depth itself against the stand-in, at one sample.
    let read_only = base
        .plan(&dev, &shape(FragmentTests::Depth { writes: false }))
        .expect("depth without writes");
    assert!(read_only.variant.read_only_tests && !read_only.variant.shader_tests);
    let depth = base
        .plan(&dev, &shape(FragmentTests::Depth { writes: true }))
        .expect("depth with writes");
    assert!(depth.variant.shader_tests && !depth.variant.read_only_tests);
    let tested = dev
        .compile_render_variant(&d, &depth.variant, &Budget::standard())
        .expect("the in-shader depth test's variant");
    assert!(tested.dependencies.iter().any(|x| x.fact
        == RuntimeFact::ShaderTests {
            formatless: false,
            multisampled: false,
            planes: false,
        }));
    let stand_in = tested.graphics.shader_tests.expect("the stand-in").depth;
    assert_eq!(stand_in.format, crate::vk::Format::R32_SFLOAT);
    assert_eq!(stand_in.samples, 1);
    assert_eq!(stand_in.view, crate::pipeline::StandInView::Image);
    for m in &tested.stages {
        check_words_valid(&m.spirv, &tested.requirements, dev.profile())
            .unwrap_or_else(|e| panic!("L3: {e}"));
    }
    let again = tested
        .plan(&dev, &shape(FragmentTests::Depth { writes: true }))
        .expect("the in-shader depth test's plan");
    assert_eq!(again.variant, depth.variant, "a fixed point");
    assert!(again
        .descriptors
        .iter()
        .any(|w| w.source == crate::bind::WriteSource::DepthStandIn));
}

/// At 4 samples a discarding fetch that stores a color through the interlock beside a
/// `Depth32Float_Stencil8` attachment, without early tests: a draw whose depth test writes runs the
/// variant whose function tests depth itself, and its words validate and its plan is a fixed point.
/// On the M3 Max and the M4, whose per-sample depth the function computes
/// (`VkProperty::CenterFragCoordDepth`), the variant holds the stand-in as sample planes, its
/// vertex function writes the window depth, and its fragment function reads it and records
/// `ComputedSampleDepth`. On RADV, whose `FragCoord.z` is each sample's
/// (`VkProperty::SampleFragCoordDepth`), it holds the stand-in as a multisampled storage image,
/// writes no window depth and records no note. On Intel's ANV, unmeasured, the draw is refused at
/// the depth attachment.
#[test]
fn a_per_sample_depth_test_takes_each_sample_s_depth_where_measured() {
    use crate::bind::FragmentTests;
    use crate::error::{Attachment, Unsupported};
    let (Some(tested), Some(io)) = (fixture("tsw_fetch_discard"), fixture("stage_io")) else {
        return;
    };
    let function = |bc: &[u8], name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        ))
    };
    let device = |stem: &str| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        Device::new(desc, DevicePolicy::default()).expect("device")
    };
    let mut d = RenderPipelineDescriptor::new(function(&io, "stage_io_vertex_position"));
    d.fragment = Some(function(&tested, "tsw_fetch_discard"));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
    // Color 1 is fetched and written, as the guest's color 0 is.
    d.color_attachments[1] = Some(ColorAttachmentDescriptor::new(PixelFormat::RG16Float));
    d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
    d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
    d.raster_sample_count = 4;
    let shape = |fragment_tests: FragmentTests| {
        BindState {
            fragment_tests,
            ..Default::default()
        }
        .shape()
    };
    // (profile, each sample's depth computed, the stand-in as sample planes)
    let measured = [("moltenvk-m3", true, true), ("amd-780m-radv", false, false)];
    for (stem, computed, planes) in measured {
        let dev = device(stem);
        let base = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .expect("the pipeline");
        let early_variant = base
            .dependencies
            .iter()
            .find_map(|x| match x.fact {
                RuntimeFact::TestedStorageWrites {
                    early_tests: false,
                    early_variant,
                } => Some(early_variant),
                _ => None,
            })
            .expect("stores before the tests");
        let plan = base
            .plan(&dev, &shape(FragmentTests::Depth { writes: true }))
            .expect("the plan");
        assert!(plan.variant.shader_tests, "{stem}");
        assert_eq!(plan.variant.window_depth_varying, computed, "{stem}");
        let variant = dev
            .compile_render_variant(&d, &plan.variant, &Budget::standard())
            .expect("the in-shader depth test's variant");
        assert!(
            variant.dependencies.iter().any(|x| x.fact
                == RuntimeFact::ShaderTests {
                    formatless: false,
                    multisampled: true,
                    planes,
                }),
            "{stem}"
        );
        let stand_in = variant.graphics.shader_tests.expect("the stand-in").depth;
        assert_eq!(stand_in.samples, 4, "{stem}");
        let view = match planes {
            true => crate::pipeline::StandInView::SamplePlanes { layers: 4 },
            false => crate::pipeline::StandInView::Image,
        };
        assert_eq!(stand_in.view, view, "{stem}");
        let notes: Vec<PrecisionNote> = variant
            .stages
            .iter()
            .flat_map(|m| m.precision_notes.iter().copied())
            .collect();
        assert_eq!(
            notes.contains(&PrecisionNote::ComputedSampleDepth),
            computed,
            "{stem}"
        );
        for m in &variant.stages {
            check_words_valid(&m.spirv, &variant.requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{stem} L3: {e}"));
        }
        let again = variant
            .plan(&dev, &shape(FragmentTests::Depth { writes: true }))
            .expect("the variant's plan");
        assert_eq!(again.variant, plan.variant, "{stem}: a fixed point");
        assert!(
            again
                .descriptors
                .iter()
                .any(|w| w.source == WriteSource::DepthStandIn),
            "{stem}"
        );
        // A test that writes nothing keeps the early-tests variant where the function has one.
        let read_only = base
            .plan(&dev, &shape(FragmentTests::Depth { writes: false }))
            .expect("the plan without writes");
        assert_eq!(read_only.variant.read_only_tests, early_variant, "{stem}");
        assert_eq!(read_only.variant.shader_tests, !early_variant, "{stem}");
    }
    // Unmeasured on Intel's ANV: a draw whose depth test writes is refused at the depth
    // attachment.
    let anv = device("intel-irisxe-rplp-anv");
    let base = anv
        .compile_render_pipeline(&d, &Budget::standard())
        .expect("ANV's pipeline");
    assert_eq!(
        base.plan(&anv, &shape(FragmentTests::Depth { writes: true }))
            .map(|_| ()),
        Err(BindError::Unsupported(
            Unsupported::not_implemented(Feature::FramebufferFetch)
                .at_attachment(Attachment::Depth)
        ))
    );
}

/// A discarding fetch that stores a color through the interlock beside a `Depth32Float_Stencil8`
/// attachment, without early tests. A draw whose stencil test acts and whose tests write
/// (`FragmentTests::Stencil`) runs the variant whose function runs the stencil test itself before
/// the depth test: at one sample on MoltenVK (M3 Max, M4), RADV and Intel's ANV, and at four on
/// RADV, the variant states `ShaderStencilTest`, holds the stencil's stand-in as an `R8_UINT` image
/// of the pipeline's samples, validates, and plans as a fixed point that binds both stand-ins. A
/// draw whose stencil test drops fragments but whose tests write nothing keeps the early-tests
/// variant. At four samples MoltenVK, whose storage images are single-sample, is refused at the
/// stencil attachment (sample planes are not implemented), and a pass with a stencil attachment
/// alone is refused at it.
#[test]
fn a_stencil_test_that_acts_runs_in_the_shader_before_the_depth_test() {
    use crate::bind::FragmentTests;
    use crate::error::{Attachment, Unsupported};
    let (Some(tested), Some(io)) = (fixture("tsw_fetch_discard"), fixture("stage_io")) else {
        return;
    };
    let function = |bc: &[u8], name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        ))
    };
    let device = |stem: &str| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        Device::new(desc, DevicePolicy::default()).expect("device")
    };
    let descriptor = |samples: u32| {
        let mut d = RenderPipelineDescriptor::new(function(&io, "stage_io_vertex_position"));
        d.fragment = Some(function(&tested, "tsw_fetch_discard"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
        d.color_attachments[1] = Some(ColorAttachmentDescriptor::new(PixelFormat::RG16Float));
        d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
        d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
        d.raster_sample_count = samples;
        d
    };
    let shape = |fragment_tests: FragmentTests| {
        BindState {
            fragment_tests,
            ..Default::default()
        }
        .shape()
    };
    let shipped = [
        ("moltenvk-m3", 1),
        ("amd-780m-radv", 1),
        ("amd-780m-radv", 4),
        ("intel-irisxe-rplp-anv", 1),
    ];
    for (stem, samples) in shipped {
        let at = format!("{stem} at {samples}");
        let dev = device(stem);
        let d = descriptor(samples);
        let base = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .expect("the pipeline");
        let plan = base
            .plan(&dev, &shape(FragmentTests::Stencil))
            .unwrap_or_else(|e| panic!("{at}: {e:?}"));
        assert!(
            plan.variant.shader_tests && plan.variant.shader_stencil,
            "{at}"
        );
        assert!(!plan.variant.read_only_tests, "{at}");
        let variant = dev
            .compile_render_variant(&d, &plan.variant, &Budget::standard())
            .expect("the in-shader tests' variant");
        assert!(
            variant
                .dependencies
                .iter()
                .any(|x| matches!(x.fact, RuntimeFact::ShaderStencilTest { .. })),
            "{at}"
        );
        let tests = variant.graphics.shader_tests.expect("the stand-ins");
        let stencil = tests.stencil.expect("the stencil's stand-in");
        assert_eq!(stencil.format, crate::vk::Format::R8_UINT, "{at}");
        assert_eq!(stencil.samples, samples, "{at}");
        assert_eq!(stencil.view, crate::pipeline::StandInView::Image, "{at}");
        for m in &variant.stages {
            check_words_valid(&m.spirv, &variant.requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{at} L3: {e}"));
        }
        let again = variant
            .plan(&dev, &shape(FragmentTests::Stencil))
            .expect("the variant's plan");
        assert_eq!(again.variant, plan.variant, "{at}: a fixed point");
        for source in [WriteSource::DepthStandIn, WriteSource::StencilStandIn] {
            assert!(
                again.descriptors.iter().any(|w| w.source == source),
                "{at}: {source:?}"
            );
        }
        // A stencil test that drops fragments but writes nothing, beside a depth test that
        // writes nothing, keeps the early-tests variant.
        let drop = base
            .plan(&dev, &shape(FragmentTests::Drop))
            .expect("the plan without writes");
        assert!(
            drop.variant.read_only_tests && !drop.variant.shader_stencil,
            "{at}"
        );
    }
    // MoltenVK at four samples: a multisampled stencil the device's single-sample storage images
    // would hold as sample planes, which are not implemented.
    {
        let stem = "moltenvk-m3";
        let dev = device(stem);
        let base = dev
            .compile_render_pipeline(&descriptor(4), &Budget::standard())
            .expect("the pipeline");
        assert_eq!(
            base.plan(&dev, &shape(FragmentTests::Stencil)).map(|_| ()),
            Err(BindError::Unsupported(
                Unsupported::not_implemented(Feature::FramebufferFetch)
                    .at_attachment(Attachment::Stencil)
            )),
            "{stem}"
        );
    }
    // A pass with a stencil attachment alone.
    let dev = device("amd-780m-radv");
    let mut d = descriptor(1);
    d.depth_format = None;
    d.stencil_format = Some(PixelFormat::Stencil8);
    let base = dev
        .compile_render_pipeline(&d, &Budget::standard())
        .expect("the stencil-only pipeline");
    assert_eq!(
        base.plan(&dev, &shape(FragmentTests::Stencil)).map(|_| ()),
        Err(BindError::Unsupported(
            Unsupported::not_implemented(Feature::FramebufferFetch)
                .at_attachment(Attachment::Stencil)
        ))
    );
}

/// A fetch that writes its color through the interlock and returns its depth
/// (`dsx_fetch_returned_depth`), beside a `Depth32Float` attachment, stores before the tests and
/// has no early-tests variant; a draw whose depth test writes runs the variant whose function tests
/// the depth it returns. At one sample and at four on MoltenVK (M3 Max, M4), RADV and Intel's ANV
/// alike: a returned depth is the same at every sample, so no device needs a per-sample depth
/// source for it, no vertex function writes the window depth, and no note is recorded. The
/// variant's words validate and its plan is a fixed point that binds the stand-in. The compiled
/// pipeline (a draw under `Always`, `FragmentTests::Pass`) and the variant alike record a positive
/// NaN it returns as discarded and kill it once, after the critical section (the lanes: before, the
/// compiled pipeline stored its color).
#[test]
fn a_returned_depth_is_tested_in_the_shader_at_every_sample_count() {
    use crate::bind::FragmentTests;
    let (Some(returned), Some(io)) = (fixture("dsx_fetch_returned_depth"), fixture("stage_io"))
    else {
        return;
    };
    let function = |bc: &[u8], name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        ))
    };
    let device = |stem: &str| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        Device::new(desc, DevicePolicy::default()).expect("device")
    };
    let shape = |fragment_tests: FragmentTests| {
        BindState {
            fragment_tests,
            ..Default::default()
        }
        .shape()
    };
    let profiles = ["moltenvk-m3", "amd-780m-radv", "intel-irisxe-rplp-anv"];
    for samples in [1u32, 4] {
        let mut d = RenderPipelineDescriptor::new(function(&io, "stage_io_vertex_position"));
        d.fragment = Some(function(&returned, "fetch_returned_depth"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Float));
        d.depth_format = Some(PixelFormat::Depth32Float);
        d.raster_sample_count = samples;
        for stem in profiles {
            let at = format!("{stem} at {samples}x");
            let dev = device(stem);
            let base = dev
                .compile_render_pipeline(&d, &Budget::standard())
                .expect("the pipeline");
            assert!(base.graphics.writes_depth, "{at}");
            assert!(
                kills_after_the_interlock(&base.stages),
                "{at}: the compiled pipeline drops a positive NaN after the critical section"
            );
            assert!(
                base.dependencies.iter().any(|x| x.fact
                    == RuntimeFact::TestedStorageWrites {
                        early_tests: false,
                        early_variant: false,
                    }),
                "{at}: stores before the tests, without an early-tests variant"
            );
            let plan = base
                .plan(&dev, &shape(FragmentTests::Depth { writes: true }))
                .unwrap_or_else(|e| panic!("{at}: the plan: {e:?}"));
            assert!(plan.variant.shader_tests, "{at}");
            assert!(!plan.variant.window_depth_varying, "{at}");
            let variant = dev
                .compile_render_variant(&d, &plan.variant, &Budget::standard())
                .expect("the in-shader depth test's variant");
            assert!(
                variant.dependencies.iter().any(|x| matches!(
                    x.fact,
                    RuntimeFact::ShaderTests { multisampled, .. } if multisampled == (samples > 1)
                )),
                "{at}"
            );
            assert!(
                kills_after_the_interlock(&variant.stages),
                "{at}: the variant"
            );
            let notes: Vec<PrecisionNote> = variant
                .stages
                .iter()
                .flat_map(|m| m.precision_notes.iter().copied())
                .collect();
            assert!(!notes.contains(&PrecisionNote::ComputedSampleDepth), "{at}");
            for m in &variant.stages {
                check_words_valid(&m.spirv, &variant.requirements, dev.profile())
                    .unwrap_or_else(|e| panic!("{at} L3: {e}"));
            }
            let again = variant
                .plan(&dev, &shape(FragmentTests::Depth { writes: true }))
                .expect("the variant's plan");
            assert_eq!(again.variant, plan.variant, "{at}: a fixed point");
            assert!(
                again
                    .descriptors
                    .iter()
                    .any(|w| w.source == WriteSource::DepthStandIn),
                "{at}"
            );
        }
    }
}

/// Fetches that write their color through the interlock and return a stencil, a sample mask, or
/// both (`dsx_fetch_returned_stencil`), beside a `Depth32Float_Stencil8` attachment, store before
/// the tests without an early-tests variant. A draw whose stencil test acts runs the variant that
/// runs both tests in the shader, the returned stencil its reference and the returned mask keeping
/// the fragment's samples; a draw whose depth test alone acts, the depth test's variant. At one
/// sample on MoltenVK (M4), RADV, Intel's ANV and NVIDIA (without stencil export, whose compiled
/// pipeline holds a returned stencil in the shader-side stencil's shadow, and refuses a returned
/// sample mask beside it), and at four on RADV: each variant states its tests, holds no shader-side
/// stencil (the stencil output is stored nowhere), validates, and plans as a fixed point that binds
/// its stand-ins.
#[test]
fn a_returned_stencil_and_sample_mask_are_tested_in_the_shader() {
    use crate::bind::FragmentTests;
    let (Some(returned), Some(io)) = (fixture("dsx_fetch_returned_stencil"), fixture("stage_io"))
    else {
        return;
    };
    let function = |bc: &[u8], name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        ))
    };
    let device = |stem: &str| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        Device::new(desc, DevicePolicy::default()).expect("device")
    };
    let shape = |fragment_tests: FragmentTests| {
        BindState {
            fragment_tests,
            ..Default::default()
        }
        .shape()
    };
    let shipped = [
        ("amd-780m-radv", 1),
        ("amd-780m-radv", 4),
        ("intel-irisxe-rplp-anv", 1),
        ("nvidia-rtx5080-laptop", 1),
    ];
    let functions = [
        "fetch_returned_stencil",
        "fetch_returned_mask",
        "fetch_returned_stencil_mask",
    ];
    for (stem, samples) in shipped {
        let dev = device(stem);
        for name in functions {
            let at = format!("{name} on {stem} at {samples}x");
            let mut d = RenderPipelineDescriptor::new(function(&io, "stage_io_vertex_position"));
            d.fragment = Some(function(&returned, name));
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Float));
            d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
            d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
            d.raster_sample_count = samples;
            // Without stencil export the compiled pipeline holds a returned stencil in the
            // shader-side stencil's shadow, which refuses a returned sample mask beside it.
            let shadow_beside_mask = !dev.profile().fields().fragment.stencil_export
                && name == "fetch_returned_stencil_mask";
            let base = match dev.compile_render_pipeline(&d, &Budget::standard()) {
                Err(Error::Unsupported(u)) if shadow_beside_mask => {
                    assert_eq!(u.feature, Feature::StencilExport, "{at}");
                    continue;
                }
                Ok(_) if shadow_beside_mask => panic!("{at}: not refused"),
                compiled => compiled.unwrap_or_else(|e| panic!("{at}: the pipeline: {e:?}")),
            };
            assert!(
                base.dependencies.iter().any(|x| x.fact
                    == RuntimeFact::TestedStorageWrites {
                        early_tests: false,
                        early_variant: false,
                    }),
                "{at}: stores before the tests, without an early-tests variant"
            );
            for (tests, stencil) in [
                (FragmentTests::Stencil, true),
                (FragmentTests::Depth { writes: true }, false),
            ] {
                let at = format!("{at} under {tests:?}");
                let plan = base
                    .plan(&dev, &shape(tests))
                    .unwrap_or_else(|e| panic!("{at}: the plan: {e:?}"));
                assert!(plan.variant.shader_tests, "{at}");
                assert_eq!(plan.variant.shader_stencil, stencil, "{at}");
                let variant = dev
                    .compile_render_variant(&d, &plan.variant, &Budget::standard())
                    .unwrap_or_else(|e| panic!("{at}: the variant: {e:?}"));
                assert_eq!(
                    variant
                        .dependencies
                        .iter()
                        .any(|x| matches!(x.fact, RuntimeFact::ShaderStencilTest { .. })),
                    stencil,
                    "{at}"
                );
                assert!(
                    variant.graphics.shadow_stencil.is_none(),
                    "{at}: no shader-side stencil"
                );
                for m in &variant.stages {
                    check_words_valid(&m.spirv, &variant.requirements, dev.profile())
                        .unwrap_or_else(|e| panic!("{at} L3: {e}"));
                }
                let again = variant
                    .plan(&dev, &shape(tests))
                    .unwrap_or_else(|e| panic!("{at}: the variant's plan: {e:?}"));
                assert_eq!(again.variant, plan.variant, "{at}: a fixed point");
                let sources: &[WriteSource] = match stencil {
                    true => &[WriteSource::DepthStandIn, WriteSource::StencilStandIn],
                    false => &[WriteSource::DepthStandIn],
                };
                for source in sources {
                    assert!(
                        again.descriptors.iter().any(|w| w.source == *source),
                        "{at}: {source:?}"
                    );
                }
            }
        }
    }
}

/// Whether the fragment module of `stages` kills (`OpKill` or `OpTerminateInvocation`) exactly
/// once, after its `OpEndInvocationInterlockEXT`: the interlocked entry's kill of a recorded
/// discard.
fn kills_after_the_interlock(stages: &[crate::pipeline::artifact::StageModule]) -> bool {
    let Some(m) = stages.iter().find(|m| m.stage == vk::ShaderStage::FRAGMENT) else {
        return false;
    };
    let (mut end, mut kills) = (None, Vec::new());
    let mut i = 5;
    while i < m.spirv.len() {
        let op = m.spirv[i] & 0xffff;
        if op == crate::spv::Op::EndInvocationInterlockEXT as u32 {
            end = Some(i);
        }
        if op == crate::spv::Op::Kill as u32 || op == crate::spv::Op::TerminateInvocation as u32 {
            kills.push(i);
        }
        i += ((m.spirv[i] >> 16) as usize).max(1);
    }
    matches!((end, kills.as_slice()), (Some(end), [kill]) if *kill > end)
}

/// Apple's GPU drops every image write of an early-tests fragment shader in a pass of more than one
/// sample, so MoltenVK on it carries `DriverWorkaround::EarlyTestsDropMultisampledImageWrites`,
/// keyed on its driver and vendor, and no other committed profile does. There the per-sample
/// storage form refuses its writes where the module declares early tests, naming the defect. Beside
/// a depth attachment, where the module would declare them only for its own sake, it declares none:
/// it stores before the tests and states `TestedStorageWrites` without the early-tests variant, so
/// the plan runs only its draws whose tests drop no fragment. Without a depth attachment, or at
/// one sample, it compiles as before. A device of the same form without the defect (`intel-arl-anv`: sample
/// planes too) compiles the per-sample form with early tests.
#[test]
fn the_per_sample_form_refuses_early_tests_writes_on_a_driver_that_drops_them() {
    use crate::device::DriverWorkaround;
    let Some(bc) = fixture("fragment_fetch") else {
        return;
    };
    let defect = DriverWorkaround::EarlyTestsDropMultisampledImageWrites;
    for (stem, p) in committed_profiles().expect("profiles") {
        assert_eq!(
            p.fields().workarounds.contains(&defect),
            is_moltenvk_capture(&stem),
            "{stem}"
        );
    }
    let state = |depth: bool, multisampled: bool| {
        let mut inputs = LoweringInputs::new();
        inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of(
            [0],
            &[Some(ColorAttachmentDescriptor::new(PixelFormat::R32Float))],
            depth,
            false,
            1,
        ));
        inputs.multisampled = multisampled;
        inputs
    };
    let early = |a: &crate::library::FunctionArtifact| {
        a.facts.iter().find_map(|f| match f {
            RuntimeFact::InterlockedFramebufferFetch { early_tests, .. } => Some(*early_tests),
            _ => None,
        })
    };
    let mvk = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let anv = profile("intel-arl-anv", DevicePolicy::default()).expect("profile");
    for (depth, multisampled) in [(false, true), (true, false), (false, false)] {
        let a = compile_with(&bc, "fetch_scalar", &mvk, &state(depth, multisampled))
            .unwrap_or_else(|e| panic!("depth {depth} multisampled {multisampled}: {e}"));
        assert_eq!(
            early(&a),
            Some(depth),
            "depth {depth} multisampled {multisampled}"
        );
    }
    let a = compile_with(&bc, "fetch_scalar", &mvk, &state(true, true)).expect("per draw");
    link_and_validate(&a, &mvk).unwrap_or_else(|e| panic!("L3: {e}"));
    assert_eq!(early(&a), Some(false));
    assert!(a.facts.contains(&RuntimeFact::TestedStorageWrites {
        early_tests: false,
        early_variant: false,
    }));
    let (_, m) = typed_module_with(&bc, "fetch_scalar", &mvk, &state(true, true)).expect("module");
    assert!(!m.entry().modes.contains(&ExecMode::EarlyFragmentTests));
    // A function that declares `[[early_fragment_tests]]` keeps them, and the refusal.
    if let Some(declared) = fixture("con_fetch_coverage") {
        let mut inputs = LoweringInputs::new();
        inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of(
            [1],
            &[
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float)),
                Some(ColorAttachmentDescriptor::new(PixelFormat::RG16Float)),
            ],
            true,
            false,
            1,
        ));
        inputs.multisampled = true;
        match compile_with(&declared, "con_fetch_coverage", &mvk, &inputs) {
            Err(Error::Unsupported(u)) => assert_eq!(
                (u.feature, u.needs),
                (Feature::FramebufferFetch, Needs::DriverDefect(defect))
            ),
            other => panic!("{:?}", other.map(|_| ())),
        }
    }
    let a = compile_with(&bc, "fetch_scalar", &anv, &state(true, true)).expect("anv");
    assert_eq!(early(&a), Some(true));
}

/// `moltenvk-m3` with `features` removed from its description.
pub(super) fn moltenvk_without(features: &[VkFeature]) -> DeviceProfile {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    for f in features {
        assert!(desc.features.remove(f), "moltenvk-m3 has {f:?}");
    }
    Device::new(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// The refusal of a raster order group on a device without pixel interlock.
fn pixel_interlock() -> Needs {
    Needs::AnyOf(vec![vec![Capability::Feature(
        VkFeature::FragmentShaderPixelInterlock,
    )]])
}

/// After a site with demote semantics (MSL ≥ 2.3), a simdgroup reduction predicates the site with a
/// whole-quad exit before the reduction where the profile has fragment quad operations (a discarded
/// thread stays a lane while its quad keeps a live thread, and a wholly discarded quad leaves, as
/// in Metal), else keeps the demoted site, else refuses as the device gap needing demote.
#[test]
fn a_later_simdgroup_operation_predicates_with_a_whole_quad_exit() {
    let Some(bc) = fixture("fragment_rog") else {
        return;
    };
    let mut quads = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        quads += usize::from(fragment_quads(&p));
        simd_after_discard(&bc, "discard_then_simd", &stem, &p);
    }
    assert!(
        quads > 0,
        "no committed profile has fragment quad operations"
    );
    // Without demote, the whole-quad exit is the form that compiles (moltenvk has fragment
    // quad operations).
    let p = moltenvk_without(&[VkFeature::ShaderDemoteToHelperInvocation]);
    assert!(fragment_quads(&p));
    simd_after_discard(&bc, "discard_then_simd", "moltenvk-m3 without demote", &p);
}

/// A site a store follows (`hoisted_written_read`) demotes, and where the profile has fragment quad
/// operations a quad whose four invocations are helpers exits before the simdgroup operations after
/// it (one `OpIsHelperInvocationEXT` and a kill per exit), as Metal drops a quad whose every thread
/// is discarded: Vulkan leaves such a quad to the implementation, and lavapipe under
/// `RequireFullQuadsKHR` kept it a lane of a later `simd_sum` (the lane
/// `runtime_lane::discard_hoist`, master's gate at 5991e117).
#[test]
fn a_demoting_site_exits_a_quad_of_helpers_before_a_later_simdgroup_operation() {
    let Some(bc) = fixture("fragment_discard_hoist") else {
        return;
    };
    let mut exited = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let f = p.fields();
        if !f.leaf.fragment_stores_and_atomics
            || !f.fragment.demote_to_helper
            || !f.subgroup.quad_control
        {
            continue;
        }
        let entry = "hoisted_written_read";
        check_form(&bc, entry, &stem, &p, DiscardForm::Demote);
        let (plan, module) =
            typed_module(&bc, entry, &p).unwrap_or_else(|e| panic!("{entry} on {stem}: {e}"));
        let exits = plan.families.fragment.quad_exits.len();
        let helpers = module
            .blocks()
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|n| matches!(n.inst, Inst::IsHelperInvocation))
            .count();
        assert_eq!(helpers, exits, "{entry} on {stem}: a helper test per exit");
        assert_eq!(exits > 0, fragment_quads(&p), "{entry} on {stem}");
        exited += usize::from(exits > 0);
    }
    assert!(exited > 0, "no committed profile placed the exits");
}

/// Where Metal places the sum ahead of each discard of the owned fixture `fragment_discard_hoist`,
/// as Apple M4 and M3 Max draw it, and what canonicalization makes of
/// it: `Moved` where Metal counts only the lanes that survive the discard, `Counted` where it
/// counts the dropped quads too for a measured reason, `Unknown` where no rule covers the shape (a
/// use between, a nested discard, which Metal was measured placing both ways), which alone records
/// `QuadsCountedBeforeDiscard`.
const HOIST_SHAPES: [(&str, Placement); 13] = [
    ("hoisted_known", Placement::Moved),
    ("counted_dependent", Placement::Counted),
    ("hoisted_const", Placement::Moved),
    ("hoisted_device_read", Placement::Moved),
    ("hoisted_written_read", Placement::Moved),
    ("hoisted_sample", Placement::Moved),
    ("hoisted_texture_read", Placement::Moved),
    ("hoisted_loop", Placement::Moved),
    ("hoisted_call", Placement::Moved),
    ("counted_call_use", Placement::Unknown),
    ("hoisted_nested", Placement::Unknown),
    ("counted_store", Placement::Counted),
    ("counted_atomic", Placement::Counted),
];

/// What canonicalization makes of a fixture's sum ahead of a discard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Placement {
    /// Moved past the discard: no discard follows it in the canonical body.
    Moved,
    /// Left ahead of it, as Metal does.
    Counted,
    /// Left ahead of it, Metal's placement unknown.
    Unknown,
}

/// Where the canonical body of `entry` places each of its simdgroup sums of a constant (the
/// count each shape takes ahead of its discard; the later counts of `fragment_discard_hoist` read
/// a buffer value), in layout order.
fn sum_placements(bc: &[u8], entry: &str) -> Vec<Placement> {
    use crate::air::canon::discard_sink::{self, Facts};
    use crate::air::intrinsics::{AirIntrinsic, GroupOp};
    use crate::air::ir::{Callee, InstKind, Value};
    let mut m = crate::budget::Meter::new(&Budget::default());
    let decoded = crate::air::decode(bc, &mut m).unwrap_or_else(|e| panic!("{entry}: {e}"));
    let request = crate::air::SpecRequest {
        call_stack_depth: 1,
        entry: entry.into(),
        ..crate::air::SpecRequest::default()
    };
    let e = crate::air::specialize(decoded, &request, &mut m)
        .unwrap_or_else(|err| panic!("{entry}: {err}"));
    let body = e
        .module
        .func(e.func)
        .and_then(|f| f.body.as_ref())
        .expect("the entry's body");
    let counts: Vec<_> = body
        .blocks
        .iter()
        .flat_map(|(b, _)| {
            body.block_insts(b)
                .filter_map(move |(id, i)| match &i.kind {
                    InstKind::Call(c)
                        if matches!(
                            c.callee,
                            Callee::Air { op: AirIntrinsic::Group(g), .. } if g.op == GroupOp::Sum
                        ) && matches!(c.args.first(), Some(Value::Const(_))) =>
                    {
                        Some((b, id))
                    }
                    _ => None,
                })
        })
        .collect();
    let Some(facts) = Facts::of(body, &mut m).expect("facts") else {
        // No block of the body discards, so every sum in it is placed as Metal places it.
        return counts.into_iter().map(|_| Placement::Moved).collect();
    };
    let placed = counts
        .into_iter()
        .map(
            |(home, op)| match discard_sink::placement(&facts, body, op, home) {
                discard_sink::Placement::NoDiscard => Placement::Moved,
                discard_sink::Placement::Counted => Placement::Counted,
                discard_sink::Placement::Unknown => Placement::Unknown,
                moved @ discard_sink::Placement::Moved(_) => {
                    panic!("{entry}: canonicalization left a movable sum ({moved:?})")
                }
            },
        )
        .collect();
    facts.release(&mut m);
    placed
}

/// Each measured shape of `fragment_discard_hoist` takes Metal's placement, and only a sum whose
/// placement is unknown records `PrecisionNote::QuadsCountedBeforeDiscard` (its lanes may include
/// the quads the discard drops, which Metal's may leave out). The sum ahead of
/// `fragment_discard_counts`' discard moves past it; an operation only after a discard records
/// nothing.
#[test]
fn a_sum_before_a_discard_takes_metals_placement() {
    let (Some(hoist), Some(counts), Some(old)) = (
        fixture("fragment_discard_hoist"),
        fixture("fragment_discard_counts"),
        fixture("fragment_discard_msl22"),
    ) else {
        return;
    };
    let p = profile("amd-780m-radv", DevicePolicy::default()).expect("profile");
    let noted = |bc: &[u8], entry: &str| {
        compile(bc, entry, &p)
            .unwrap_or_else(|e| panic!("{entry}: {e}"))
            .precision
            .contains(&PrecisionNote::QuadsCountedBeforeDiscard)
    };
    for (entry, expected) in HOIST_SHAPES {
        assert_eq!(sum_placements(&hoist, entry), [expected], "{entry}");
        assert_eq!(
            noted(&hoist, entry),
            expected == Placement::Unknown,
            "{entry}: the note"
        );
    }
    // Its count ahead of the discard and the one after it.
    assert_eq!(
        sum_placements(&counts, "discard_counts"),
        [Placement::Moved, Placement::Moved]
    );
    assert!(!noted(&counts, "discard_counts"));
    assert!(!noted(&old, "discard_simd"));
    assert!(!noted(&old, "discard_loop"));
}

/// The kills (`OpKill` and `OpTerminateInvocation`) and returns of function `f`.
fn exits(m: &TypedModule, f: FuncId) -> (usize, usize) {
    let (mut kills, mut returns) = (0, 0);
    for &b in &m.functions()[f.index()].blocks {
        match m.blocks()[b.index()].term {
            Terminator::Kill | Terminator::TerminateInvocation => kills += 1,
            Terminator::Return => returns += 1,
            _ => {}
        }
    }
    (kills, returns)
}

/// The functions `f` calls, in call order.
fn callees(m: &TypedModule, f: FuncId) -> Vec<FuncId> {
    let mut out = Vec::new();
    for &b in &m.functions()[f.index()].blocks {
        for n in &m.blocks()[b.index()].insts {
            if let Inst::Call { func, .. } = &n.inst {
                out.push(*func);
            }
        }
    }
    out
}

/// Check the shape of an interlocked module: the entry is `Begin; call body; End; [if discarded {
/// kill }]; return` under `mode`, and the body holds no interlock instruction and never kills.
/// Every other function is a routine the body reaches (a fused multiply-add on a device without
/// `OpFmaKHR`, say): only the body or another routine calls it, and it neither interlocks nor
/// kills. Returns the body.
fn check_interlocked(m: &TypedModule, mode: ExecMode, what: &str) -> FuncId {
    let entry = m.entry();
    assert!(entry.modes.contains(&mode), "{what}: {:?}", entry.modes);
    let mut order = Vec::new();
    let mut body = None;
    for &b in &m.functions()[entry.func.index()].blocks {
        for n in &m.blocks()[b.index()].insts {
            match &n.inst {
                Inst::BeginInterlock => order.push("begin"),
                Inst::EndInterlock => order.push("end"),
                Inst::Call { func, .. } => {
                    body = Some(*func);
                    order.push("call");
                }
                _ => {}
            }
        }
    }
    assert_eq!(order, ["begin", "call", "end"], "{what}");
    let body = body.expect("the entry calls the body");
    assert_ne!(body, entry.func, "{what}: the body is outlined");
    // Everything the body reaches, and nothing else, besides the entry.
    let mut reached = vec![body];
    let mut i = 0;
    while let Some(&f) = reached.get(i) {
        for c in callees(m, f) {
            assert!(c != entry.func && c != body, "{what}: {f:?} calls {c:?}");
            if !reached.contains(&c) {
                reached.push(c);
            }
        }
        i += 1;
    }
    assert_eq!(
        reached.len() + 1,
        m.functions().len(),
        "{what}: the entry, the body, and the routines the body reaches"
    );
    for &f in &reached {
        for &b in &m.functions()[f.index()].blocks {
            for n in &m.blocks()[b.index()].insts {
                assert!(
                    !matches!(n.inst, Inst::BeginInterlock | Inst::EndInterlock),
                    "{what}: the critical section is the entry's"
                );
            }
        }
        assert_eq!(exits(m, f).0, 0, "{what}: {f:?} under the body never kills");
    }
    body
}

/// A function that names a raster order group on a buffer, a texture, or an unused buffer runs its
/// body inside `PixelInterlockOrderedEXT` where the device has pixel interlock, and passes L3
/// there; elsewhere it is refused.
#[test]
fn raster_order_groups_run_the_body_inside_the_interlock() {
    let Some(bc) = fixture("fragment_rog") else {
        return;
    };
    let entries = [
        "rog_buffer",
        "rog_texture",
        "rog_discard",
        "rog_loop",
        "rog_sample",
    ];
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let f = p.fields();
        for entry in entries {
            if !f.fragment.pixel_interlock {
                match compile(&bc, entry, &p) {
                    Err(Error::Unsupported(_)) => {}
                    other => panic!("{entry} on {stem}: {:?}", other.map(|_| ())),
                }
                continue;
            }
            let what = format!("{entry} on {stem}");
            let m = match typed_module(&bc, entry, &p) {
                Ok((_, m)) => m,
                // A read_write texture without format-less storage reads needs its format, which
                // this test does not supply.
                Err(Error::NeedsInput(_)) if entry == "rog_texture" => continue,
                Err(e) => panic!("{what}: {e}"),
            };
            check_interlocked(&m, ExecMode::PixelInterlockOrdered, &what);
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            compiled += 1;
        }
    }
    assert!(compiled > 0, "no committed profile has pixel interlock");
}

/// A raster order group that holds only a color output the function does not read (`rog_output`)
/// needs no interlock: Vulkan already orders a fragment's color writes and blending per sample in
/// rasterization order, as Metal does. It compiles without an interlock mode or instruction on
/// every committed profile, and on a device with neither interlock, and passes L3. (A group on a
/// color input still needs the interlock:
/// `framebuffer_fetch_reads_in_an_ordered_form_where_the_device_has_one`.)
#[test]
fn a_group_of_color_outputs_alone_needs_no_interlock() {
    let Some(bc) = fixture("fragment_rog") else {
        return;
    };
    let neither = moltenvk_without(&[
        VkFeature::FragmentShaderPixelInterlock,
        VkFeature::FragmentShaderSampleInterlock,
    ]);
    let profiles = committed_profiles()
        .expect("profiles")
        .into_iter()
        .chain([("moltenvk-m3 without interlock".to_string(), neither)]);
    let mut without = 0;
    for (stem, p) in profiles {
        let what = format!("rog_output on {stem}");
        let (_, m) = typed_module(&bc, "rog_output", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        let entry = m.entry();
        assert!(
            !entry.modes.iter().any(|mode| matches!(
                mode,
                ExecMode::PixelInterlockOrdered | ExecMode::SampleInterlockOrdered
            )),
            "{what}: {:?}",
            entry.modes
        );
        for f in m.functions() {
            for &b in &f.blocks {
                for n in &m.blocks()[b.index()].insts {
                    assert!(
                        !matches!(n.inst, Inst::BeginInterlock | Inst::EndInterlock),
                        "{what}: an interlock instruction"
                    );
                }
            }
        }
        let a = compile(&bc, "rog_output", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        if !p.fields().fragment.pixel_interlock {
            without += 1;
        }
    }
    assert!(without > 0, "no profile without pixel interlock");
}

/// With only sample interlock, a function that runs at sample rate is ordered per sample (Metal's
/// own ordering at sample rate), and a pixel-rate one is refused.
#[test]
fn sample_interlock_orders_only_sample_rate_functions() {
    let Some(bc) = fixture("fragment_rog") else {
        return;
    };
    let p = moltenvk_without(&[VkFeature::FragmentShaderPixelInterlock]);
    assert!(p.fields().fragment.sample_interlock);
    let (_, m) = typed_module(&bc, "rog_sample", &p).expect("rog_sample");
    check_interlocked(&m, ExecMode::SampleInterlockOrdered, "rog_sample");
    let a = compile(&bc, "rog_sample", &p).expect("rog_sample");
    link_and_validate(&a, &p).expect("L3");
    assert_eq!(
        refusal(&bc, "rog_buffer", &p),
        (Feature::RasterOrderGroups, pixel_interlock())
    );
}

/// A loop in a raster order group runs inside the interlocked body, which the entry calls once
/// between `Begin` and `End`.
#[test]
fn a_loop_in_a_raster_order_group_stays_in_the_interlocked_body() {
    let Some(bc) = fixture("fragment_rog") else {
        return;
    };
    let mut interlocked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !p.fields().fragment.pixel_interlock {
            continue;
        }
        let (_, m) = typed_module(&bc, "rog_loop", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let body = check_interlocked(&m, ExecMode::PixelInterlockOrdered, &stem);
        assert!(!m.loops().is_empty(), "{stem}: the loop");
        for l in m.loops() {
            let header = &m.blocks()[l.header.index()];
            assert_eq!(header.func, body, "{stem}: the loop is the body's");
        }
        interlocked += 1;
    }
    assert!(interlocked > 0, "no committed profile has pixel interlock");
}

/// Inside the interlocked body a discard never kills. A terminate-semantics site records the
/// discard and returns, a predicated one records it; the entry kills once, after `End`. Demote
/// stays a demote.
#[test]
fn an_interlocked_discard_kills_only_in_the_entry() {
    let (Some(new), Some(old)) = (fixture("fragment_rog"), fixture("fragment_rog_msl22")) else {
        return;
    };
    let mut cases: Vec<(String, DeviceProfile)> = committed_profiles()
        .expect("profiles")
        .into_iter()
        .filter(|(_, p)| p.fields().fragment.pixel_interlock)
        .collect();
    cases.push((
        "moltenvk-m3 without demote".to_owned(),
        moltenvk_without(&[VkFeature::ShaderDemoteToHelperInvocation]),
    ));
    for (stem, p) in &cases {
        let helper = if p.fields().fragment.demote_to_helper {
            DiscardForm::Demote
        } else {
            DiscardForm::Predicate
        };
        let checks: [(&[u8], &str, DiscardForm); 2] = [
            (&old, "rog_terminate", DiscardForm::Terminate),
            (&new, "rog_discard", helper),
        ];
        for (bc, entry, form) in checks {
            let what = format!("{entry} on {stem}");
            let (plan, m) = typed_module(bc, entry, p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let fp = &plan.families.fragment;
            assert!(!fp.discards.is_empty(), "{what}");
            assert!(fp.discards.values().all(|f| *f == form), "{what}: {fp:?}");
            let body = check_interlocked(&m, ExecMode::PixelInterlockOrdered, &what);
            let (kills, _) = exits(&m, m.entry().func);
            let (_, returns) = exits(&m, body);
            match form {
                DiscardForm::Demote => assert_eq!(kills, 0, "{what}"),
                DiscardForm::Predicate => assert_eq!((kills, returns), (1, 1), "{what}"),
                // The discard's own return and the body's.
                DiscardForm::Terminate => assert_eq!((kills, returns), (1, 2), "{what}"),
            }
            let a = compile(bc, entry, p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        }
    }
}

/// One function the interlock coherence test compiles: its fixture, its entry, the
/// framebuffer-fetch colors it reads (with their formats), and whether it needs the Vulkan
/// memory model.
struct Interlocked<'a> {
    bc: &'a [u8],
    entry: &'static str,
    fetch: Option<(&'static [u32], &'static [PixelFormat])>,
    vulkan: bool,
}

/// Every access of memory another invocation may write inside the interlock is coherent at device
/// scope, on all three kinds of that memory: buffers, device addresses, and storage images (a
/// texture's, a storage-form attachment's, an imageblock's, `engine_fragment_imageblock`).
/// Under `GLSL450` the resources are `Coherent` (`rog_buffer`, `rog_texture`, the storage-form
/// fetch); a coherent access (`rog_texture_coherent`, `fetch_coherent`) or a device address, which
/// no decoration covers (`a3_rog_argument_buffer`, `rog_texture_pointer`), selects the Vulkan
/// model, whose accesses carry `MakePointer*`/`MakeTexel*` at `Device` scope, and a volatile
/// texture's `VolatileTexel`. A device without that model refuses what needs it, naming the
/// feature. Every module passes L3, and on `moltenvk-m3` compiles through SPIRV-Cross to Metal.
#[test]
fn every_access_inside_the_interlock_is_coherent() {
    let (Some(rog), Some(dvp), Some(arg), Some(fetch)) = (
        fixture("fragment_rog"),
        fixture("dvp_rog_coherence"),
        fixture("a3_rog_argument_buffer"),
        fixture("fragment_fetch"),
    ) else {
        return;
    };
    const R32: &[PixelFormat] = &[PixelFormat::R32Float];
    const RGBA16: &[PixelFormat] = &[PixelFormat::RGBA16Float];
    let cases = [
        Interlocked {
            bc: &rog,
            entry: "rog_buffer",
            fetch: None,
            vulkan: false,
        },
        Interlocked {
            bc: &rog,
            entry: "rog_texture",
            fetch: None,
            vulkan: false,
        },
        Interlocked {
            bc: &fetch,
            entry: "fetch_scalar",
            fetch: Some((&[0], R32)),
            vulkan: false,
        },
        Interlocked {
            bc: &arg,
            entry: "a3_rog_argument_buffer",
            fetch: None,
            vulkan: true,
        },
        Interlocked {
            bc: &dvp,
            entry: "rog_texture_coherent",
            fetch: None,
            vulkan: true,
        },
        Interlocked {
            bc: &dvp,
            entry: "rog_texture_pointer",
            fetch: None,
            vulkan: true,
        },
        Interlocked {
            bc: &dvp,
            entry: "fetch_coherent",
            fetch: Some((&[0], RGBA16)),
            vulkan: true,
        },
    ];
    // Accesses checked under each model: [GLSL450, Vulkan] x [buffer, device address, image],
    // and the volatile image accesses under the Vulkan model.
    let mut seen = [[0usize; 3]; 2];
    let mut volatile = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !p.fields().fragment.pixel_interlock {
            continue;
        }
        for c in &cases {
            let what = format!("{} on {stem}", c.entry);
            let inputs = match c.fetch {
                Some((read, formats)) => fetch_state(read, formats, false),
                None => LoweringInputs::new(),
            };
            let texture = c.entry.starts_with("rog_texture");
            let a = match compile_with(c.bc, c.entry, &p, &inputs) {
                Ok(a) => a,
                // A read_write texture without format-less storage reads needs its format.
                Err(Error::NeedsInput(_)) if texture => continue,
                Err(e) => panic!("{what}: {e}"),
            };
            let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let n = interlocked_coherence(&words).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(n.vulkan, c.vulkan, "{what}: the memory model");
            // MoltenVK's path (SPIRV-Cross, then Metal) takes the per-access operands.
            if is_moltenvk_capture(&stem) {
                match moltenvk_msl_compile(&words) {
                    Ok(()) => {}
                    Err(AssembleError::Missing(why)) => eprintln!("SKIPPED MSL: {why}"),
                    Err(AssembleError::Rejected(why)) => panic!("{what}: MSL: {why}"),
                }
            }
            let row = &mut seen[usize::from(n.vulkan)];
            row[0] += n.buffer;
            row[1] += n.device_address;
            row[2] += n.storage_image;
            volatile += n.volatile_texel;
        }
    }
    // Every kind is checked under each model that can hold it (`GLSL450` has no coherent device
    // address).
    assert!(
        seen[0][0] > 0 && seen[0][1] == 0 && seen[0][2] > 0,
        "{seen:?}"
    );
    assert!(seen[1].iter().all(|&n| n > 0), "{seen:?}");
    // A texture the device may read back a dropped write from is volatile: under the Vulkan model,
    // which bans the decoration, its accesses carry `VolatileTexel`.
    assert!(volatile > 0, "no volatile texture under the Vulkan model");
    // Without the Vulkan model, a device address inside the interlock is refused.
    let p = moltenvk_without(&[VkFeature::VulkanMemoryModel]);
    assert_eq!(
        refusal(&arg, "a3_rog_argument_buffer", &p),
        (
            Feature::CoherentMemory,
            Needs::AnyOf(vec![vec![Capability::Feature(
                VkFeature::VulkanMemoryModel
            )]])
        )
    );
    let a = compile(&rog, "rog_buffer", &p).expect("rog_buffer");
    let (_, words) = link_and_validate(&a, &p).expect("rog_buffer: L3");
    assert!(!interlocked_coherence(&words).expect("rog_buffer").vulkan);
}

/// A fragment function `f` of a synthetic module: it reads the varying `v` (`float4`, qualified
/// `interp`, AIR's `!"air.center", !"air.perspective"` form) and the `i32` the `body` lines leave
/// in `%r` from `%x`, the varying's first lane's bits, and writes them to color 0.
fn varying_fragment(name: &str, interp: &str, body: &str, decls: &str) -> Option<Vec<u8>> {
    let text = format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <{{ <4 x float> }}> @f(<4 x float> noundef %pos, <4 x float> noundef %v) local_unnamed_addr #0 {{
  %lane = extractelement <4 x float> %v, i64 0
  %x = bitcast float %lane to i32
{body}
  %rf = bitcast i32 %r to float
  %out = insertelement <4 x float> %v, float %rf, i64 1
  %ret = insertvalue <{{ <4 x float> }}> undef, <4 x float> %out, 0
  ret <{{ <4 x float> }}> %ret
}}

{decls}

attributes #0 = {{ convergent mustprogress nounwind willreturn }}
attributes #1 = {{ convergent nounwind willreturn }}

!air.fragment = !{{!0}}
!air.compile_options = !{{!6, !7, !8}}
!air.version = !{{!9}}
!air.language_version = !{{!10}}

!0 = !{{ptr @f, !1, !3}}
!1 = !{{!2}}
!2 = !{{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}}
!3 = !{{!4, !5}}
!4 = !{{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}}
!5 = !{{i32 1, !"air.fragment_input", !"generated(1vDv4_f)", {interp}, !"air.arg_type_name", !"float4", !"air.arg_name", !"v"}}
!6 = !{{!"air.compile.denorms_disable"}}
!7 = !{{!"air.compile.fast_math_enable"}}
!8 = !{{!"air.compile.framebuffer_fetch_disable"}}
!9 = !{{i32 2, i32 8, i32 0}}
!10 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    );
    let scratch = Scratch::new(&format!("m2v-unm-varying-{}", name.replace(' ', "-")));
    let dir = scratch.path();
    let path = dir.join("f.ll");
    std::fs::write(&path, text).expect("write");
    assemble(&path)
}

/// The answer of `f` in the module `bc` on each committed profile, as whether it records `note`:
/// every profile in `noted` compiles it and records the note, and every other profile that
/// compiles it (one may refuse an operation it lacks) does not.
fn notes_on(bc: &[u8], what: &str, note: PrecisionNote, noted: &[&str]) {
    for (stem, p) in committed_profiles().expect("profiles") {
        let want = noted.contains(&stem.as_str());
        match compile(bc, "f", &p) {
            Ok(a) => assert_eq!(a.precision.contains(&note), want, "{what} on {stem}"),
            Err(e) => assert!(!want, "{what} on {stem}: {e}"),
        }
    }
}

/// A fragment function that reads a varying the device interpolates records
/// `PrecisionNote::DeviceInterpolation`, whatever its sampling and perspective, on every committed
/// profile but MoltenVK's, which take `DriverDeviation::MetalInterpolation` (Metal interpolates
/// there): those record none and require `VkProperty::MetalInterpolation` instead. One whose
/// varying is flat, which no interpolation touches, records neither anywhere.
#[test]
fn a_varying_the_device_interpolates_records_the_device_interpolation_note() {
    use crate::device::requirements::VkProperty;
    let note = PrecisionNote::DeviceInterpolation;
    let profiles = committed_profiles().expect("profiles");
    let devices: Vec<&str> = profiles
        .iter()
        .map(|(stem, _)| stem.as_str())
        .filter(|stem| !is_moltenvk_capture(stem))
        .collect();
    assert!(devices.len() > 10, "{devices:?}");
    let body = "  %r = add i32 %x, 1";
    let cases = [
        ("perspective", r#"!"air.center", !"air.perspective""#, true),
        ("linear", r#"!"air.center", !"air.no_perspective""#, true),
        ("centroid", r#"!"air.centroid", !"air.perspective""#, true),
        ("flat", r#"!"air.flat""#, false),
    ];
    for (what, interp, interpolated) in cases {
        let Some(bc) = varying_fragment(what, interp, body, "") else {
            return;
        };
        let noted = if interpolated { &devices[..] } else { &[][..] };
        notes_on(&bc, what, note, noted);
        for (stem, p) in &profiles {
            let Ok(a) = compile(&bc, "f", p) else {
                continue;
            };
            let requires = a
                .requirements
                .properties
                .contains(&VkProperty::MetalInterpolation);
            let by_metal = interpolated && is_moltenvk_capture(stem);
            assert_eq!(requires, by_metal, "{what} on {stem}");
        }
    }
}

/// On a driver that shades a triangle it clips as several
/// (`DriverWorkaround::ClippedTrianglesSplitPixelFragments`: lavapipe), a fragment function that
/// runs a quad or simdgroup operation over the active lanes records
/// `PrecisionNote::QuadsSplitByClipping`; one whose group operation reads a named lane (a
/// shuffle), or that runs none, does not, and no other driver records it.
#[test]
fn an_operation_over_the_active_lanes_records_the_split_quads_note() {
    let note = PrecisionNote::QuadsSplitByClipping;
    let lavapipe = ["gate-lavapipe", "gate-lavapipe-x86_64", "lavapipe"];
    let flat = r#"!"air.flat""#;
    let cases = [
        (
            "quad mask",
            "  %m = tail call i16 @air.quad_active_threads_mask() #1\n  %r = zext i16 %m to i32",
            "declare i16 @air.quad_active_threads_mask() local_unnamed_addr #1",
            &lavapipe[..],
        ),
        (
            "quad sum",
            "  %r = tail call i32 @air.quad_sum.u.i32(i32 %x) #1",
            "declare i32 @air.quad_sum.u.i32(i32) local_unnamed_addr #1",
            &lavapipe[..],
        ),
        (
            "simd ballot",
            "  %c = icmp ugt i32 %x, 7\n  %b = tail call i64 @air.simd_ballot.i64(i1 %c) #1\n  %r = trunc i64 %b to i32",
            "declare i64 @air.simd_ballot.i64(i1) local_unnamed_addr #1",
            &lavapipe[..],
        ),
        (
            "quad shuffle",
            "  %r = tail call i32 @air.quad_shuffle_xor.u.i32(i32 %x, i16 1) #1",
            "declare i32 @air.quad_shuffle_xor.u.i32(i32, i16) local_unnamed_addr #1",
            &[][..],
        ),
        ("no group operation", "  %r = add i32 %x, 1", "", &[][..]),
    ];
    for (what, body, decls, noted) in cases {
        let Some(bc) = varying_fragment(what, flat, body, decls) else {
            return;
        };
        notes_on(&bc, what, note, noted);
    }
}

/// A fragment function blended into an `RGB10A8_2P_XR10` attachment (premultiplied over, and a
/// blend of source alpha into the color channels) records its output's blend note on every
/// interlock profile that makes the storage form, whatever the module's math and however it
/// computes the output: the shader blends it by the device's arithmetic, where Metal's compiler
/// folds the blend into the function in forms of its own. A `half4` output records
/// `PrecisionNote::ExtendedRangeBlendSource` and a `float4` one
/// `PrecisionNote::ExtendedRangeFloatBlendSource`, never the other's: the fills loaded from a
/// buffer, `rt6_resolve_white`'s constant, a computed `half4`, a `float4` computed by whole-vector
/// arithmetic or with a lane moved alone, under fast math, without it (into the `_sRGB` form too)
/// and naming no math mode. Unblended, the constant output records none.
#[test]
fn a_blended_extended_range_output_records_its_blend_note() {
    use crate::metal::BlendFactor;
    let (Some(fills), Some(resolves)) = (fixture("rt6_samples"), fixture("rt6_resolves")) else {
        return;
    };
    let text = std::fs::read_to_string(engine_dir().join("rt6_samples.ll")).expect("text");
    assert_eq!(text.matches("air.compile.fast_math_enable").count(), 1);
    let assemble = |text: &str, what: &str| match metal2vulkan_devtools::llvm_assemble_text(text) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {what}: {why}"),
    };
    let disable = |text: &str| {
        text.replace(
            "air.compile.fast_math_enable",
            "air.compile.fast_math_disable",
        )
    };
    let Some(safe) = assemble(&disable(&text), "the safe fills") else {
        return;
    };
    // The module's options without its math mode.
    let options = "!air.compile_options = !{!106, !107, !108}";
    assert_eq!(text.matches(options).count(), 1);
    assert_eq!(
        text.matches("!107 = !{!\"air.compile.fast_math_enable\"}")
            .count(),
        1
    );
    let unnamed = text.replace(options, "!air.compile_options = !{!106, !108}");
    let Some(unnamed) = assemble(&unnamed, "the fills without a math mode") else {
        return;
    };
    let load = "  ret <4 x half> %19\n";
    assert_eq!(text.matches(load).count(), 1);
    let computed = text.replace(
        load,
        "  %computed = fmul fast <4 x half> %19, <half 0xH3C00, half 0xH3C00, half 0xH3C00, \
         half 0xH3C00>\n  ret <4 x half> %computed\n",
    );
    let Some(safe_computed) = assemble(&disable(&computed), "the safe computed fill") else {
        return;
    };
    let Some(computed) = assemble(&computed, "the computed fill") else {
        return;
    };
    // The `float4` fill computed by whole-vector arithmetic (its load times ones), and computed
    // with its alpha the load's (a lane moved alone).
    let float_load = "  ret <4 x float> %19\n";
    assert_eq!(text.matches(float_load).count(), 1);
    let ones = "<float 1.0, float 1.0, float 1.0, float 1.0>";
    let float_computed = text.replace(
        float_load,
        &format!("  %cf = fmul fast <4 x float> %19, {ones}\n  ret <4 x float> %cf\n"),
    );
    let Some(float_computed) = assemble(&float_computed, "the computed float4 fill") else {
        return;
    };
    let float_mixed = text.replace(
        float_load,
        &format!(
            "  %cf = fmul fast <4 x float> %19, {ones}\n  %mixed = shufflevector <4 x float> %cf, \
             <4 x float> %19, <4 x i32> <i32 0, i32 1, i32 2, i32 7>\n  ret <4 x float> %mixed\n"
        ),
    );
    let Some(float_mixed) = assemble(&float_mixed, "the lane-mixed float4 fill") else {
        return;
    };
    let colors_of = |format: PixelFormat, blended: bool| {
        let mut c = ColorAttachmentDescriptor::new(format);
        c.blending_enabled = blended;
        c.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        c.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        vec![Some(c)]
    };
    let colors = |blended: bool| colors_of(PixelFormat::RGB10A82PXR10, blended);
    let state = |colors: &[Option<ColorAttachmentDescriptor>], p: &DeviceProfile| {
        let mut inputs = LoweringInputs::new();
        inputs.color_writes = crate::pipeline::color_writes::color_writes(
            0..colors.len() as u32,
            colors,
            (false, Default::default()),
            p,
        );
        inputs.extended_range_colors =
            crate::pipeline::color_writes::extended_range_colors(colors, Default::default());
        inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of([0u32], colors, true, false, 1));
        inputs
    };
    let notes = [
        PrecisionNote::ExtendedRangeBlendSource,
        PrecisionNote::ExtendedRangeFloatBlendSource,
    ];
    let (half, float) = (Some(notes[0]), Some(notes[1]));
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::interlock_orders(&p) {
            continue;
        }
        let blended = state(&colors(true), &p);
        // The one of the two notes the function records, if it records one.
        let recorded = |a: &crate::library::FunctionArtifact| {
            let found: Vec<PrecisionNote> = notes
                .into_iter()
                .filter(|n| a.precision.contains(n))
                .collect();
            assert!(found.len() <= 1, "{stem}: {found:?}");
            found.first().copied()
        };
        let noted = |bitcode: &[u8], name: &str, inputs: &LoweringInputs| {
            compile_with(bitcode, name, &p, inputs)
                .map(|a| recorded(&a))
                .map_err(|e| format!("{stem}: {name}: {e}"))
        };
        let float_fill = match compile_with(&fills, "rt6_fill_rgba32float", &p, &blended) {
            Ok(a) => recorded(&a),
            // A device without the raw formats' storage refuses the form whatever the source.
            Err(Error::Unsupported(u)) if u.needs != Needs::NotImplemented => continue,
            Err(e) => panic!("{stem}: {e}"),
        };
        checked += 1;
        assert_eq!(float_fill, float, "{stem}: a loaded float4 under fast math");
        assert_eq!(
            noted(&fills, "rt6_fill_rgba16float", &blended),
            Ok(half),
            "{stem}: a loaded half4 under fast math"
        );
        assert_eq!(
            noted(&resolves, "rt6_resolve_white", &blended),
            Ok(float),
            "{stem}: a constant float4"
        );
        assert_eq!(
            noted(&computed, "rt6_fill_rgba16float", &blended),
            Ok(half),
            "{stem}: a computed half4 under fast math"
        );
        let fill = "rt6_fill_rgba32float";
        assert_eq!(
            noted(&float_computed, fill, &blended),
            Ok(float),
            "{stem}: a computed float4 under fast math"
        );
        let keyed = {
            let mut c = ColorAttachmentDescriptor::new(PixelFormat::RGB10A82PXR10);
            c.blending_enabled = true;
            c.destination_rgb_blend_factor = BlendFactor::SourceAlpha;
            c.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
            state(&[Some(c)], &p)
        };
        assert_eq!(
            noted(&fills, fill, &keyed),
            Ok(float),
            "{stem}: keyed, loaded"
        );
        assert_eq!(
            noted(&float_computed, fill, &keyed),
            Ok(float),
            "{stem}: keyed, computed"
        );
        assert_eq!(
            noted(&float_mixed, fill, &blended),
            Ok(float),
            "{stem}: a float4 with a lane moved alone"
        );
        let unblended = state(&colors(false), &p);
        assert_eq!(
            noted(&resolves, "rt6_resolve_white", &unblended),
            Ok(None),
            "{stem}: a constant float4 unblended"
        );
        let srgb = state(&colors_of(PixelFormat::RGB10A82PXR10SRGB, true), &p);
        for (name, note) in [
            ("rt6_fill_rgba32float", float),
            ("rt6_fill_rgba16float", half),
        ] {
            assert_eq!(
                noted(&safe, name, &blended),
                Ok(note),
                "{stem}: {name} without fast math"
            );
            assert_eq!(
                noted(&unnamed, name, &blended),
                Ok(note),
                "{stem}: {name} of no math mode"
            );
            assert_eq!(
                noted(&safe, name, &srgb),
                Ok(note),
                "{stem}: {name} without fast math into the sRGB form"
            );
        }
        assert_eq!(
            noted(&safe_computed, "rt6_fill_rgba16float", &blended),
            Ok(half),
            "{stem}: a computed half4 without fast math"
        );
    }
    assert!(checked > 0, "no interlock profile made the storage form");
}

/// Apple's compiler drops the texture writes of every live thread of a simdgroup that holds a
/// discarded thread, where the fragment function discards, then writes storage, samples and calls
/// SPIRV-Cross's `optnone` arithmetic helpers, so MoltenVK on an Apple GPU carries
/// `DriverWorkaround::DiscardDropsSimdgroupStores`, and no other committed profile does. There the
/// owned fixture `rch_late_discard`'s function that samples and computes the color it writes in the
/// storage form after its discard predicates it, killing once past the write: the entry kills after
/// `End`, the body never, and its census names the discard `fragment.discard.deferred`. So is one
/// whose buffer store after the discard runs under `if !discarded` (`rch_late_discard_store`), with
/// no `PrecisionNote::DiscardDropsSimdgroupStores`; one that samples nothing
/// (`rch_late_discard_store_unsampled`) and
/// one that writes its color alone (`rch_late_discard_plain`) keep their demote and record
/// nothing. A device without the defect (`intel-arl-anv`: the storage form too) demotes each.
/// Every module passes L3.
#[test]
fn a_discard_waits_past_the_stores_on_a_driver_that_drops_a_simdgroups_stores() {
    use crate::device::DriverWorkaround;
    let Some(bc) = fixture("rch_late_discard") else {
        return;
    };
    let defect = DriverWorkaround::DiscardDropsSimdgroupStores;
    for (stem, p) in committed_profiles().expect("profiles") {
        assert_eq!(
            p.fields().workarounds.contains(&defect),
            is_moltenvk_capture(&stem),
            "{stem}"
        );
    }
    let fetched = || {
        let mut inputs = LoweringInputs::new();
        inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of(
            [0],
            &[Some(ColorAttachmentDescriptor::new(
                PixelFormat::RGBA16Float,
            ))],
            false,
            false,
            1,
        ));
        inputs
    };
    let mvk = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let anv = profile("intel-arl-anv", DevicePolicy::default()).expect("profile");
    let deferred = "fragment.discard.deferred";
    let noted = PrecisionNote::DiscardDropsSimdgroupStores;
    // (entry, reads color 0, its form and whether it records the note on `moltenvk-m3`)
    let cases = [
        ("rch_late_discard", true, DiscardForm::Predicate, false),
        (
            "rch_late_discard_store",
            true,
            DiscardForm::Predicate,
            false,
        ),
        (
            "rch_late_discard_store_unsampled",
            true,
            DiscardForm::Demote,
            false,
        ),
        ("rch_late_discard_plain", false, DiscardForm::Demote, false),
    ];
    for (entry, fetches, form, note) in cases {
        for (stem, p, expected, note) in [
            ("moltenvk-m3", &mvk, form, note),
            ("intel-arl-anv", &anv, DiscardForm::Demote, false),
        ] {
            let what = format!("{entry} on {stem}");
            let inputs = if fetches {
                fetched()
            } else {
                LoweringInputs::new()
            };
            let (plan, m) =
                typed_module_with(&bc, entry, p, &inputs).unwrap_or_else(|e| panic!("{what}: {e}"));
            let fp = &plan.families.fragment;
            assert!(!fp.discards.is_empty(), "{what}");
            assert!(
                fp.discards.values().all(|f| *f == expected),
                "{what}: {fp:?}"
            );
            let demotes = m
                .blocks()
                .iter()
                .flat_map(|b| &b.insts)
                .filter(|n| matches!(n.inst, Inst::DemoteToHelperInvocation))
                .count();
            assert_eq!(
                demotes > 0,
                expected == DiscardForm::Demote,
                "{what}: demotes"
            );
            if expected == DiscardForm::Predicate {
                let body = check_interlocked(&m, ExecMode::PixelInterlockOrdered, &what);
                assert_eq!(exits(&m, m.entry().func).0, 1, "{what}: the entry kills");
                assert_eq!(exits(&m, body).0, 0, "{what}: the body never kills");
            }
            let a = compile_with(&bc, entry, p, &inputs).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(
                a.strategies.entries.contains_key(deferred),
                expected == DiscardForm::Predicate,
                "{what}: {:?}",
                a.strategies.entries.keys().collect::<Vec<_>>()
            );
            assert_eq!(a.precision.contains(&noted), note, "{what}: the note");
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        }
    }
}

/// The selection headers of `m` whose condition or selector asks whether the invocation is a
/// helper, as `spirv::helper_branches` writes it: the condition or'ed with
/// `OpIsHelperInvocationEXT`, and'ed with its negation, or a selector chosen by it.
fn helper_guards(m: &TypedModule) -> usize {
    use crate::spirv::ir::{BinOp, UnOp};
    let defs: std::collections::HashMap<_, _> = m
        .blocks()
        .iter()
        .flat_map(|b| &b.insts)
        .filter_map(|n| n.result.map(|r| (r, &n.inst)))
        .collect();
    let helper = |v| matches!(defs.get(&v), Some(Inst::IsHelperInvocation));
    let not_helper =
        |v| matches!(defs.get(&v), Some(Inst::Unary { op: UnOp::LogicalNot, a }) if helper(*a));
    m.blocks()
        .iter()
        .filter(|b| match &b.term {
            Terminator::CondBranch { cond, .. } => match defs.get(cond) {
                Some(Inst::Binary {
                    op: BinOp::LogicalOr,
                    b: x,
                    ..
                }) => helper(*x),
                Some(Inst::Binary {
                    op: BinOp::LogicalAnd,
                    b: x,
                    ..
                }) => not_helper(*x),
                _ => false,
            },
            Terminator::Switch { sel, .. } => {
                matches!(defs.get(sel), Some(Inst::Select { cond, .. }) if helper(*cond))
            }
            _ => false,
        })
        .count()
}

/// On MoltenVK on an Apple GPU (`DriverWorkaround::DiscardDropsSimdgroupStores`), Apple's compiler
/// loses a simdgroup's storage writes after an `optnone` call only its helper invocations make,
/// with no discard: the quad helpers past a primitive's edges take a branch alone. The owned
/// fixture `rch_helper_branch`'s functions sample, blend `half` values with the high word of a
/// product (`mulhi`, an `OpUMulExtended`, which SPIRV-Cross writes as its `optnone`
/// `spvMulExtended`) in a branch where the coverage they read is not 1, and write a buffer after
/// it. There the branch's helper invocations take its empty arm where nothing they compute
/// differently reaches a derivative: `rch_helper_branch` (no implicit LOD) and
/// `rch_helper_branch_late` (an implicit-LOD sample after the branch at the pixel) are guarded,
/// census `fragment.helper-guard`, and record no note. `rch_helper_branch_gradient`, whose
/// implicit-LOD sample reads a coordinate the branch computes, is not guarded and records no note
/// either: the sample reads the product after it. `rch_helper_branch_loop`, whose product is in a
/// loop as long as the coverage, records `PrecisionNote::HelperBranchDropsSimdgroupStores`. One
/// that samples nothing (`rch_helper_branch_unsampled`) is left as it is. A device without the
/// defect (`intel-arl-anv`) guards nothing. Every module passes L3.
#[test]
fn helper_invocations_skip_an_optnone_branch_on_a_driver_that_drops_a_simdgroups_stores() {
    let Some(bc) = fixture("rch_helper_branch") else {
        return;
    };
    let mvk = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let anv = profile("intel-arl-anv", DevicePolicy::default()).expect("profile");
    let noted = PrecisionNote::HelperBranchDropsSimdgroupStores;
    let census = "fragment.helper-guard";
    // (entry, guarded on `moltenvk-m3`, noted there)
    let cases = [
        ("rch_helper_branch", true, false),
        ("rch_helper_branch_late", true, false),
        ("rch_helper_branch_gradient", false, false),
        ("rch_helper_branch_unsampled", false, false),
        ("rch_helper_branch_loop", false, true),
    ];
    let inputs = LoweringInputs::new();
    for (entry, guarded, note) in cases {
        for (stem, p, guarded, note) in [
            ("moltenvk-m3", &mvk, guarded, note),
            ("intel-arl-anv", &anv, false, false),
        ] {
            let what = format!("{entry} on {stem}");
            let (_, m) =
                typed_module_with(&bc, entry, p, &inputs).unwrap_or_else(|e| panic!("{what}: {e}"));
            let guards = helper_guards(&m);
            let a = compile_with(&bc, entry, p, &inputs).unwrap_or_else(|e| panic!("{what}: {e}"));
            let counted = a.strategies.entries.get(census).map(|u| u.uses as usize);
            let noted_here = a.precision.contains(&noted);
            eprintln!("observed: {what}: guards {guards}, census {counted:?}, note {noted_here}");
            assert_eq!(guards > 0, guarded, "{what}: {guards} guards");
            assert_eq!(counted, guarded.then_some(guards), "{what}: the census");
            assert_eq!(noted_here, note, "{what}: the note");
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        }
    }
}

/// A selection whose condition every invocation of a draw computes alike (a constant block's flag)
/// holds no branch only helper invocations take, so an `optnone` call in its arm is neither guarded
/// nor noted. The owned fixture `rch_uniform_branch` divides a coordinate by a texture's size and
/// takes the high word of a product (`mulhi`, an `OpUMulExtended`, which SPIRV-Cross writes
/// through its `optnone` `spvMulExtended`) in a branch on a constant block's flag, then samples
/// the texture at the coordinate and writes the sample and the product to a buffer. On
/// `moltenvk-m3` the module holds the product, and no helper census entry, no guard, no
/// `HelperBranchDropsSimdgroupStores`, and L3.
#[test]
fn a_branch_every_invocation_takes_alike_leaves_no_helper_alone() {
    use crate::spirv::ir::BinOp;
    let Some(bc) = fixture("rch_uniform_branch") else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let noted = PrecisionNote::HelperBranchDropsSimdgroupStores;
    let what = "rch_uniform_branch on moltenvk-m3";
    let inputs = LoweringInputs::new();
    let (_, m) = typed_module_with(&bc, "rch_uniform_branch", &p, &inputs)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let a = compile_with(&bc, "rch_uniform_branch", &p, &inputs)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let helper: Vec<_> = a
        .strategies
        .entries
        .keys()
        .filter(|k| k.starts_with("fragment.helper"))
        .collect();
    let products = m
        .blocks()
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|n| {
            matches!(
                n.inst,
                Inst::Binary {
                    op: BinOp::UMulExtended,
                    ..
                }
            )
        })
        .count();
    assert_eq!(products, 1, "{what}: the products");
    assert!(helper.is_empty(), "{what}: {helper:?}");
    assert_eq!(helper_guards(&m), 0, "{what}");
    assert!(!a.precision.contains(&noted), "{what}: the note");
    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
}

/// Apple's compiler loses nothing after an `optnone` call only helper invocations make where they
/// make a tangled instruction (an implicit-LOD sample, a derivative, an LOD query, a simdgroup
/// operation) after it, so such a call records no note where every path from it passes a tangled
/// instruction reading what it computes. The owned fixture `rch_helper_tangle` takes the high
/// words of products (`mulhi`, an `OpUMulExtended`, which SPIRV-Cross writes through its `optnone`
/// `spvMulExtended`) where only the quad helpers past a primitive's edges run more than once or at
/// all, beside a tangled instruction, and writes a texture after it. On `moltenvk-m3` no arm is
/// guarded; the `after` functions, whose tangled instruction reads the products (in the arm an
/// implicit-LOD sample at them, `dfdx`, `quad_sum`, the LOD query; a sample after the branch's
/// merge; a sample after a loop only helpers repeat), record no
/// `HelperBranchDropsSimdgroupStores`; the `before` functions and `rch_tangle_unread`, whose
/// tangled instruction reads none of them, record it. Every module passes L3.
#[test]
fn a_tangled_instruction_reading_a_helper_call_keeps_the_note_away() {
    let Some(bc) = fixture("rch_helper_tangle") else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let noted = PrecisionNote::HelperBranchDropsSimdgroupStores;
    // (entry, noted on `moltenvk-m3`)
    let cases = [
        ("rch_tangle_sample_after", false),
        ("rch_tangle_dfdx_after", false),
        ("rch_tangle_quad_after", false),
        ("rch_tangle_lod_after", false),
        ("rch_tangle_merge_after", false),
        ("rch_tangle_loop_after", false),
        ("rch_tangle_quad_before", true),
        ("rch_tangle_sample_before", true),
        ("rch_tangle_unread", true),
    ];
    let inputs = LoweringInputs::new();
    for (entry, note) in cases {
        let what = format!("{entry} on moltenvk-m3");
        let (_, m) =
            typed_module_with(&bc, entry, &p, &inputs).unwrap_or_else(|e| panic!("{what}: {e}"));
        let a = compile_with(&bc, entry, &p, &inputs).unwrap_or_else(|e| panic!("{what}: {e}"));
        let noted_here = a.precision.contains(&noted);
        eprintln!(
            "observed: {what}: guards {}, note {noted_here}",
            helper_guards(&m)
        );
        assert_eq!(helper_guards(&m), 0, "{what}: guards");
        assert_eq!(noted_here, note, "{what}: the note");
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
    }
}

/// The opcodes of every instruction in `words`, and the capabilities it declares.
fn opcodes_and_capabilities(words: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let (mut ops, mut caps) = (Vec::new(), Vec::new());
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let (len, op) = ((w >> 16) as usize, w & 0xFFFF);
        ops.push(op);
        if op == 17 {
            caps.extend(words.get(i + 1).copied());
        }
        if len == 0 {
            break;
        }
        i += len;
    }
    (ops, caps)
}

/// A fragment function's `dfdx`, `dfdy` and `fwidth` are Metal's fine derivatives, each pixel's own
/// quad row's and column's differences (measured on an M3), so they are `OpDPdxFine`, `OpDPdyFine`
/// and `OpFwidthFine` under the `DerivativeControl` capability, never the unqualified `OpDPdx`,
/// `OpDPdy` and `OpFwidth` a driver may compute coarsely, on every committed profile;
/// `fragment_quad_derivatives` makes all three, of `float` and `half` vectors.
#[test]
fn a_fragment_functions_derivatives_are_fine() {
    const UNQUALIFIED: [u32; 3] = [207, 208, 209];
    const FINE: [u32; 3] = [210, 211, 212];
    const DERIVATIVE_CONTROL: u32 = 51;
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/fragment_quad_derivatives.ll");
    let Some(bc) = assemble(&path) else { return };
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("committed profiles") {
        let a = match compile(&bc, "fragment_quad_derivatives", &p) {
            Ok(a) => a,
            Err(Error::Unsupported(_)) => continue,
            Err(e) => panic!("{stem}: {e}"),
        };
        let (ops, caps) = opcodes_and_capabilities(&a.module.words);
        for op in FINE {
            assert!(ops.contains(&op), "{stem}: no opcode {op}");
        }
        assert!(
            !ops.iter().any(|op| UNQUALIFIED.contains(op)),
            "{stem}: an unqualified derivative"
        );
        assert!(caps.contains(&DERIVATIVE_CONTROL), "{stem}: {caps:?}");
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        compiled += 1;
    }
    assert!(compiled >= 10, "{compiled} profiles compiled it");
}

/// A draw that reads a texture its subpass renders through a snapshot (`BindState::
/// snapshot_textures`): owned fixture `tob_renders`'s `tob_rect` and `tob_read`
/// (`t.read(uint2(in.position.xy))`, whose texel reaches only its output), planned with slots 0
/// and 3 asking for snapshots, keeps slot 0 alone (the function has no slot 3) on every profile
/// with the pixel interlock, fragment stores, the feedback loop layout and dynamic color write
/// masks. Its variant binds slot 0's snapshot as a storage image, states `snapshot_reads`, makes
/// the write masks dynamic, requires the feedback loop layout, writes no memory of the
/// function's, passes L3, and plans the same variant
/// against itself, writing `WriteSource::SnapshotTexels` and leaving the phase to the consumer as a
/// fragment hole, zero in the plan's bytes. A pipeline with a depth attachment, a multisampled
/// one, a profile without one of those features, and a draw that asks for no snapshot keep the
/// plain pipeline.
#[test]
fn a_snapshot_read_plans_its_variant_where_the_pipeline_can() {
    use crate::bind::{DrawCall, PushHole};
    use crate::pipeline::sym::{PushField, SlotClass, SlotRef, SnapshotForm, Stage};
    let Some(bc) = fixture("tob_renders") else {
        return;
    };
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            Library::from_air(&bc)
                .expect("library")
                .function(name)
                .expect("function"),
        ))
    };
    let descriptor = |depth: bool, samples: u32| {
        let mut d = RenderPipelineDescriptor::new(function("tob_rect"));
        d.fragment = Some(function("tob_read"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        if depth {
            d.depth_format = Some(PixelFormat::Depth32Float);
        }
        d.raster_sample_count = samples;
        d
    };
    let snapshots = |p: &crate::pipeline::RenderPipeline| -> Vec<(u32, vk::DescriptorType)> {
        p.layout
            .sets
            .iter()
            .flat_map(|s| &s.bindings)
            .filter_map(|b| match b.source {
                BindingSource::SnapshotTexels(slot) => Some((slot, b.ty)),
                _ => None,
            })
            .collect()
    };
    let slot = SlotRef {
        stage: Stage::Fragment,
        class: SlotClass::Texture,
        index: 0,
    };
    let state = BindState {
        snapshot_textures: [0, 3].into(),
        ..Default::default()
    };
    let draw = DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 4,
        instance_count: 1,
        base_instance: 0,
    };
    let mut planned = 0;
    for stem in [
        "intel-irisxe-anv",
        "amd-780m-radv",
        "nvidia-rtx5080-laptop",
        "lavapipe",
        "vk12-floor",
    ] {
        let desc = crate::pipeline::testing::description(stem).expect("description");
        let dev = Device::new(desc, DevicePolicy::default()).expect("device");
        let base = dev
            .compile_render_pipeline(&descriptor(false, 1), &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(snapshots(&base).is_empty(), "{stem}");
        assert!(!base.graphics.snapshot_reads, "{stem}");
        let plain = base
            .plan(&dev, &BindState::default().shape())
            .expect("plain plan");
        assert!(plain.variant.snapshot_textures.is_empty(), "{stem}");
        let plan = base.plan(&dev, &state.shape()).expect("plan");
        // The binder plans the variant where the device enabled what its draws take: the pixel
        // interlock, fragment stores, the feedback loop layout and dynamic write masks.
        let enabled = &dev.profile().fields().enabled.features;
        let takes = dev.profile().fields().fragment.pixel_interlock
            && [
                VkFeature::FragmentStoresAndAtomics,
                VkFeature::AttachmentFeedbackLoopLayout,
                VkFeature::ExtendedDynamicState3ColorWriteMask,
            ]
            .iter()
            .all(|feature| enabled.contains(feature));
        if !takes {
            assert!(plan.variant.snapshot_textures.is_empty(), "{stem}");
            continue;
        }
        planned += 1;
        // No texture bound: the fetched lanes' bits.
        assert_eq!(
            plan.variant.snapshot_textures,
            [(slot, SnapshotForm::Bits)].into(),
            "{stem}"
        );
        assert!(
            matches!(
                plan.instantiate_draw(&state, &draw),
                Err(BindError::NeedsVariant { .. })
            ),
            "{stem}: the base pipeline is the other variant"
        );
        let variant = dev
            .compile_render_variant(&descriptor(false, 1), &plan.variant, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: variant: {e}"));
        assert_eq!(
            snapshots(&variant),
            [(0, vk::DescriptorType::STORAGE_IMAGE)],
            "{stem}"
        );
        assert!(variant.graphics.snapshot_reads, "{stem}");
        assert!(
            variant
                .graphics
                .dynamic_states
                .contains(&vk::DynamicState::COLOR_WRITE_MASK_EXT),
            "{stem}: dynamic write masks"
        );
        assert!(
            variant
                .requirements
                .features
                .contains(&VkFeature::AttachmentFeedbackLoopLayout)
                && variant
                    .requirements
                    .features
                    .contains(&VkFeature::ExtendedDynamicState3ColorWriteMask),
            "{stem}: {:?}",
            variant.requirements.features
        );
        assert!(!variant.graphics.fragment_writes_memory, "{stem}");
        for m in &variant.stages {
            check_words_valid(&m.spirv, &variant.requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        }
        let again = variant.plan(&dev, &state.shape()).expect("variant plan");
        assert_eq!(again.variant, plan.variant, "{stem}: a fixed point");
        assert!(
            again
                .descriptors
                .iter()
                .any(|w| w.source == WriteSource::SnapshotTexels { slot: 0 }),
            "{stem}: {:?}",
            again.descriptors
        );
        let cmds = again
            .instantiate_draw(&state, &draw)
            .unwrap_or_else(|e| panic!("{stem}: draw: {e:?}"));
        let holes: Vec<(vk::ShaderStageFlags, u32)> = cmds
            .push_holes
            .iter()
            .filter_map(|h| match *h {
                PushHole::SnapshotPhase { stages, offset } => Some((stages, offset)),
                _ => None,
            })
            .collect();
        assert_eq!(holes.len(), 1, "{stem}: {:?}", cmds.push_holes);
        let (stages, offset) = holes[0];
        assert_eq!(stages, vk::ShaderStageFlags::FRAGMENT, "{stem}");
        let at = offset as usize;
        let bytes = cmds
            .push_constants
            .iter()
            .find(|w| w.stages == stages)
            .map(|w| (w.offset as usize, &w.bytes))
            .expect("the fragment's push bytes");
        assert_eq!(
            bytes.1.get(at - bytes.0..at - bytes.0 + 4),
            Some(&[0u8; 4][..]),
            "{stem}: phase 0 in the plan's bytes"
        );
        assert!(
            again
                .push
                .fields
                .iter()
                .any(|f| f.field == PushField::SnapshotPhase),
            "{stem}"
        );
        // A bound texture's form: the codes of a normalized or half-float format, the bits of a
        // 32-bit float format's lanes and of an sRGB format's decoded ones; each variant passes L3
        // and binds its snapshot.
        for (format, form) in [
            (PixelFormat::RGBA8Unorm, SnapshotForm::Unorm8),
            (PixelFormat::BGRA8Unorm, SnapshotForm::Unorm8),
            (PixelFormat::R8Unorm, SnapshotForm::Unorm8),
            (PixelFormat::RGBA8Snorm, SnapshotForm::Snorm8),
            (PixelFormat::RG16Unorm, SnapshotForm::Unorm16),
            (PixelFormat::RGBA16Float, SnapshotForm::Half),
            (PixelFormat::RGBA32Float, SnapshotForm::Bits),
            (PixelFormat::RGB10A2Unorm, SnapshotForm::Bits),
        ] {
            let mut bound = state.clone();
            let mut table = crate::bind::state::StageTable::default();
            let facts =
                crate::bind::state::TextureFacts::new(crate::metal::TextureType::Type2D, format);
            table.textures.insert(
                0,
                crate::bind::state::BoundTexture::Texture {
                    facts,
                    resource_id: None,
                },
            );
            bound.stages.insert(Stage::Fragment, table);
            let plan = base.plan(&dev, &bound.shape()).expect("bound plan");
            assert_eq!(
                plan.variant.snapshot_textures,
                [(slot, form)].into(),
                "{stem}: {format:?}"
            );
            let variant = dev
                .compile_render_variant(&descriptor(false, 1), &plan.variant, &Budget::standard())
                .unwrap_or_else(|e| panic!("{stem}: {format:?}: {e}"));
            assert_eq!(
                snapshots(&variant),
                [(0, vk::DescriptorType::STORAGE_IMAGE)],
                "{stem}: {format:?}"
            );
            for m in &variant.stages {
                check_words_valid(&m.spirv, &variant.requirements, dev.profile())
                    .unwrap_or_else(|e| panic!("{stem}: {format:?}: L3: {e}"));
            }
        }
        // No snapshot again: the plain pipeline.
        let unread = variant
            .plan(&dev, &BindState::default().shape())
            .expect("plain plan of the variant");
        assert!(unread.variant.snapshot_textures.is_empty(), "{stem}");
        // A depth attachment, or more than one sample, keeps the plain pipeline.
        for (depth, samples) in [(true, 1), (false, 4)] {
            let other = dev
                .compile_render_pipeline(&descriptor(depth, samples), &Budget::standard())
                .unwrap_or_else(|e| panic!("{stem}: {depth} {samples}: {e}"));
            let plan = other.plan(&dev, &state.shape()).expect("plan");
            assert!(
                plan.variant.snapshot_textures.is_empty(),
                "{stem}: {depth} {samples}"
            );
        }
    }
    assert!(planned >= 4, "the interlocked profiles");
}
