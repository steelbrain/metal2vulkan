//! Fragment semantics end to end.
//!
//! The owned fixtures `engine/fragment_discard` (MSL 4.0: demote semantics) and
//! `engine/fragment_discard_msl22` (MSL 2.2: terminate semantics) go through the whole
//! per-function pipeline on every committed profile. Each discard site takes the form of the
//! matrix, the module contains exactly that form's instructions, and it passes L3. The owned
//! fixtures `engine/fragment_rog` and `fragment_rog_msl22` cover raster order groups: the body
//! runs as a function called between `Begin/EndInvocationInterlockEXT`, only the entry kills, and
//! the body's loops stay in it. `engine/fragment_fetch` covers framebuffer fetch: input
//! attachments, ordered where the device has rasterization-order color attachment access. No GPU
//! is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::path::Path;

use crate::budget::Budget;
use crate::device::requirements::VkFeature;
use crate::device::subgroup::OpBits;
use crate::device::{Device, DeviceDescription};
use crate::device::{DevicePolicy, DeviceProfile};
use crate::error::{Capability, Error, Feature, LowerFamily, Needs};
use crate::library::Library;
use crate::lower::fragment::DiscardForm;
use crate::metal::PixelFormat;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::lower_inputs::{FramebufferFetchInputs, LoweringInputs};
use crate::pipeline::sym::ResourceKind;
use crate::pipeline::testing::{
    compile, compile_with, engine_dir, interlocked_coherence, link_and_validate, llvm_assemble,
    moltenvk_msl_compile, profile, typed_module, AssembleError, Scratch,
};
use crate::spirv::ids::FuncId;
use crate::spirv::ir::{ExecMode, Inst, Terminator, TypedModule};
use crate::spirv::testing::{committed_profiles, is_moltenvk_capture};
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

/// The runtime's floor without `shaderDemoteToHelperInvocation`, a device below that floor: the
/// one that predicates a discard.
const NO_DEMOTE: &str = "runtime-floor without shaderDemoteToHelperInvocation";

fn no_demote() -> DeviceProfile {
    crate::pipeline::testing::profile_without(
        "runtime-floor",
        &[VkFeature::ShaderDemoteToHelperInvocation],
        DevicePolicy::RESIDENT_TABLES,
    )
    .expect(NO_DEMOTE)
}

/// The committed profiles and [`NO_DEMOTE`]'s.
fn discard_profiles() -> Vec<(String, DeviceProfile)> {
    let mut out = committed_profiles().expect("profiles");
    out.push((NO_DEMOTE.to_string(), no_demote()));
    out
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
    let profiles = discard_profiles();
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
/// needing demote. Where the discard has its form and the device runs no subgroup operation in the
/// fragment stage (the runtime's floor), the simdgroup operation is refused needing that stage.
///
/// Where the discard form allows it but the profile lacks `shaderQuadControl`, the simdgroup
/// operation is refused needing it: it counts the helpers Metal runs, which only full quads give.
fn simd_after_discard(bc: &[u8], entry: &str, stem: &str, p: &DeviceProfile) {
    let formed = fragment_quads(p) || p.fields().fragment.demote_to_helper;
    let full_quads = p.fields().subgroup.quad_control;
    if formed && !p.fields().subgroup.fragment.supported {
        let stage = crate::pipeline::sym::Stage::Fragment;
        assert_eq!(
            refusal(bc, entry, p),
            (
                Feature::SubgroupOperations,
                Needs::AnyOf(vec![vec![Capability::Property(
                    crate::device::requirements::VkProperty::SubgroupStage(stage)
                )]])
            ),
            "{entry} on {stem}"
        );
    } else if formed && !full_quads {
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
    use crate::spirv::ir::{StorageClass, UnOp, ValueOrigin};
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
    m.blocks()
        .iter()
        .filter(|b| matches!(&b.term, Terminator::CondBranch { cond, .. } if not_discarded(*cond)))
        .count()
}

/// A side effect after a demoted discard needs no guard (a helper invocation has none); after a
/// predicated one it runs only where the invocation was not discarded, under `if !discarded`.
/// `discard_then_store` demotes where the profile has demote, and is predicated with its store
/// guarded where it does not ([`NO_DEMOTE`]). Every module passes L3.
#[test]
fn predication_guards_the_side_effects_after_a_site() {
    let Some(bc) = fixture("fragment_discard") else {
        return;
    };
    let mut predicated = 0;
    for (stem, p) in discard_profiles() {
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
    assert!(predicated > 0, "no profile predicates");
}

/// The owned fixture `rch_guarded_effects`'s side effects after a predicated discard, each under
/// `if !discarded`: a store, an atomic and the store its old value indexes (two guards; the
/// atomic's result is merged with zero for a discarded invocation), a store in a loop, a texture
/// write, a copy. They are predicated where the profile has no demote
/// ([`NO_DEMOTE`]); `rch_guarded_simd`'s simdgroup sum is refused there needing
/// demote, as no whole-quad exit can be placed on a profile without fragment quad operations. A
/// device that demotes (`moltenvk-m3`, `intel-arl-anv`) demotes each and guards nothing; the
/// simdgroup sum, which counts the helpers Metal runs, is refused needing `shaderQuadControl`
/// where the device lacks it (`moltenvk-m3`), as only full quads give those helpers. Every module
/// passes L3.
#[test]
fn a_predicated_discard_guards_each_later_store_atomic_and_image_write() {
    use crate::spirv::ir::ValueOrigin;
    let Some(bc) = fixture("rch_guarded_effects") else {
        return;
    };
    // (entry, its side effects after the discard)
    let cases = [
        ("rch_guarded_store", 1),
        ("rch_guarded_atomic", 2),
        ("rch_guarded_loop", 1),
        ("rch_guarded_write", 1),
        ("rch_guarded_copy", 1),
        ("rch_guarded_simd", 1),
    ];
    let committed = |stem: &str| profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile");
    for (stem, p) in [
        ("moltenvk-m3", committed("moltenvk-m3")),
        (NO_DEMOTE, no_demote()),
        ("intel-arl-anv", committed("intel-arl-anv")),
    ] {
        for (entry, effects) in cases {
            let what = format!("{entry} on {stem}");
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
            if simd && !p.fields().subgroup.quad_control {
                assert_eq!(
                    refusal(&bc, entry, &p),
                    (
                        Feature::SubgroupOperations,
                        Needs::AnyOf(vec![vec![Capability::Feature(
                            VkFeature::ShaderQuadControl
                        )]])
                    ),
                    "{what}"
                );
                continue;
            }
            // Predicated where the profile has no demote.
            let predicated = !demotes;
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

/// The framebuffer-fetch state of a pipeline whose color attachments are `formats` (by index),
/// read at `read`.
fn fetch_state(read: &[u32], formats: &[PixelFormat]) -> LoweringInputs {
    let colors: Vec<Option<ColorAttachmentDescriptor>> = formats
        .iter()
        .map(|&f| Some(ColorAttachmentDescriptor::new(f)))
        .collect();
    let mut inputs = LoweringInputs::new();
    inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of(read.iter().copied(), &colors));
    inputs
}
/// Each color index a function reads is one input attachment with a `FramebufferFetch` fact,
/// standalone and under its pipeline's fetch state alike: ordered where the device has
/// rasterization-order color attachment access, and elsewhere ordered between draws alone and
/// noted `UnorderedFetchWithinDraw`, a raster order group on it (`fetch_ordered`) taking no
/// interlock there. Every compiled module passes L3.
#[test]
fn framebuffer_fetch_reads_an_input_attachment_ordered_where_the_device_orders_it() {
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
    // [unordered, ordered]
    let mut forms = [0usize; 2];
    for (stem, strict) in committed_profiles().expect("profiles") {
        let frag = strict.fields().fragment;
        let ordered = frag.rasterization_order_color_access;
        for (entry, colors, formats) in entries {
            // A color input in a raster order group needs the interlock where rasterization-order
            // access orders the read; where the read is unordered the group orders it alone.
            if entry == "fetch_ordered" && !frag.pixel_interlock && ordered {
                assert_eq!(
                    refusal(&bc, entry, &strict),
                    (Feature::RasterOrderGroups, pixel_interlock()),
                    "{stem}"
                );
                continue;
            }
            forms[usize::from(ordered)] += 1;
            for inputs in [LoweringInputs::new(), fetch_state(colors, formats)] {
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
                assert_eq!(
                    a.precision
                        .contains(&PrecisionNote::UnorderedFetchWithinDraw),
                    !ordered,
                    "{entry} on {stem}: {:?}",
                    a.precision
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
    let dev = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device");
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

/// In the shape of a pipeline a guest made: a function that reads color 0, an `RGB10A8_2P_XR10`
/// attachment, and writes it back under every channel with blending disabled, beside an `RG16Float`
/// color 1 and a second `RGB10A8_2P_XR10` attachment, both of mask none. Metal's tile holds such an
/// attachment as `half` through the pass, which the codes it holds do not keep, so the fetch is
/// refused `NotImplemented(Format)` at color 0 on every committed profile, in whichever form the
/// device fetches, for each of Apple's extended-range forms at color 0 (`exf_fetch_blend`'s `float4`)
/// and for the `_sRGB` twin among two such attachments (`exf_fetch_three`'s `float4` and `half4`).
/// The whole pipeline is refused so on `moltenvk-m3`.
#[test]
fn a_fetched_extended_range_attachment_is_refused() {
    use crate::metal::ColorWriteMask;
    let (Some(blend), Some(three)) = (fixture("exf_fetch_blend"), fixture("exf_fetch_three"))
    else {
        return;
    };
    // The guest's attachments: color 0 of `pf`, unblended under every channel, beside the two of
    // mask none.
    let colors = |pf: PixelFormat| {
        let fetched = ColorAttachmentDescriptor::new(pf);
        let mut read = ColorAttachmentDescriptor::new(PixelFormat::RG16Float);
        read.write_mask = ColorWriteMask::NONE;
        let mut unwritten = ColorAttachmentDescriptor::new(PixelFormat::RGB10A82PXR10);
        unwritten.write_mask = ColorWriteMask::NONE;
        vec![Some(fetched), Some(read), Some(unwritten)]
    };
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
        inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of(read.iter().copied(), colors));
        inputs
    };
    let refused = |r: Result<std::sync::Arc<crate::library::FunctionArtifact>, Error>,
                   what: &str| match r {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::Format, "{what}: {u:?}");
            assert_eq!(u.needs, Needs::NotImplemented, "{what}: {u:?}");
            let at = u.location.as_ref().and_then(|l| l.attachment());
            assert_eq!(
                at,
                Some(crate::error::Attachment::Color(0)),
                "{what}: {u:?}"
            );
        }
        other => panic!("{what}: {:?}", other.as_ref().map(|a| &a.facts)),
    };
    let xr = [
        PixelFormat::BGR10XR,
        PixelFormat::BGR10XRSRGB,
        PixelFormat::BGRA10XR,
        PixelFormat::BGRA10XRSRGB,
        PixelFormat::RGB10A82PXR10,
        PixelFormat::RGB10A82PXR10SRGB,
    ];
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        for pf in xr {
            let r = compile_with(&blend, "exf_fetch_blend", &p, &state(&[0], &colors(pf), &p));
            refused(r, &format!("{stem}: {pf:?}"));
            checked += 1;
        }
        let mut both = colors(PixelFormat::RGB10A82PXR10SRGB);
        both[1] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
        both[2] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGB10A82PXR10));
        let r = compile_with(&three, "exf_fetch_three", &p, &state(&[0, 1, 2], &both, &p));
        refused(r, &format!("{stem}: three"));
    }
    assert!(checked > 0, "no committed profile");
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
    let stem = "moltenvk-m3";
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let dev = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(
        &io,
        "stage_io_vertex_position",
    )));
    d.fragment = Some(StageSource::Function(function(&blend, "exf_fetch_blend")));
    for (i, c) in colors(PixelFormat::RGB10A82PXR10).into_iter().enumerate() {
        d.color_attachments[i] = c;
    }
    d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
    d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
    match dev.compile_render_pipeline(&d, &Budget::standard()) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::Format, "{stem}: {u:?}");
            assert_eq!(u.needs, Needs::NotImplemented, "{stem}: {u:?}");
        }
        other => panic!("{stem}: the guest's pipeline: {:?}", other.map(|_| ())),
    }
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
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
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
/// `RequireFullQuadsKHR` kept it a lane of a later `simd_sum` (master's gate at 5991e117).
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
    let p = profile("amd-780m-radv", DevicePolicy::RESIDENT_TABLES).expect("profile");
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

/// On every committed profile `Device::from_queries` derives, a fragment function that runs a
/// quad or simdgroup operation (`fragment_discard_msl22`'s `discard_simd`, a `simd_sum`) records
/// `PrecisionNote::QuadsSplitByClipping`, and one that runs none (its `discard_terminate`) does
/// not: no Vulkan query tells a device that shades a clipped triangle as several apart.
#[test]
fn a_fragment_group_operation_records_the_clipping_split_on_every_device() {
    let Some(bc) = fixture("fragment_discard_msl22") else {
        return;
    };
    let note = PrecisionNote::QuadsSplitByClipping;
    let mut noted = 0;
    for (stem, p) in crate::spirv::testing::committed_profiles().expect("committed profiles") {
        let plain = compile(&bc, "discard_terminate", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            !plain.precision.contains(&note),
            "{stem}: discard_terminate"
        );
        // A device without the subgroup operations refuses the sum.
        let Ok(a) = compile(&bc, "discard_simd", &p) else {
            continue;
        };
        assert!(a.precision.contains(&note), "{stem}: discard_simd");
        noted += 1;
    }
    assert!(noted >= 7, "{noted} profiles noted the split");
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
/// Every other function is a routine the body reaches: only the body or another routine calls
/// it, and it neither interlocks nor
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

/// A function that names a raster order group on a buffer or a texture it uses runs its body inside
/// `PixelInterlockOrderedEXT` where the device has pixel interlock, and passes L3 there; elsewhere
/// it is refused. (A group on a parameter the body never uses interlocks nothing:
/// `tests::engine_dead_reads`.)
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
            !entry.modes.iter().any(|mode| mode.is_interlock()),
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

/// A device with the sample interlock and not the pixel interlock refuses a raster order group,
/// one of a function that runs at sample rate included: the translator takes the pixel interlock
/// alone.
#[test]
fn without_the_pixel_interlock_a_sample_rate_group_is_refused() {
    let Some(bc) = fixture("fragment_rog") else {
        return;
    };
    let p = moltenvk_without(&[VkFeature::FragmentShaderPixelInterlock]);
    for entry in ["rog_sample", "rog_buffer"] {
        assert_eq!(
            refusal(&bc, entry, &p),
            (Feature::RasterOrderGroups, pixel_interlock()),
            "{entry}"
        );
    }
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

/// One function the interlock coherence test compiles: its fixture, its entry, and whether it
/// needs the Vulkan memory model.
struct Interlocked<'a> {
    bc: &'a [u8],
    entry: &'static str,
    vulkan: bool,
}

/// Every access of memory another invocation may write inside the interlock is coherent at device
/// scope, on all three kinds of that memory: buffers, device addresses, and a texture's storage
/// images. Under `GLSL450` the resources are `Coherent` (`rog_buffer`, `rog_texture`); a coherent
/// access (`rog_texture_coherent`) or a device address, which no decoration covers
/// (`a3_rog_argument_buffer`, `rog_texture_pointer`), selects the Vulkan
/// model, whose accesses carry `MakePointer*`/`MakeTexel*` at `Device` scope, and a volatile
/// texture's `VolatileTexel`. A device without that model refuses what needs it, naming the
/// feature. Every module passes L3, and on `moltenvk-m3` compiles through SPIRV-Cross to Metal.
#[test]
fn every_access_inside_the_interlock_is_coherent() {
    let (Some(rog), Some(dvp), Some(arg)) = (
        fixture("fragment_rog"),
        fixture("dvp_rog_coherence"),
        fixture("a3_rog_argument_buffer"),
    ) else {
        return;
    };
    let cases = [
        Interlocked {
            bc: &rog,
            entry: "rog_buffer",
            vulkan: false,
        },
        Interlocked {
            bc: &rog,
            entry: "rog_texture",
            vulkan: false,
        },
        Interlocked {
            bc: &arg,
            entry: "a3_rog_argument_buffer",
            vulkan: true,
        },
        Interlocked {
            bc: &dvp,
            entry: "rog_texture_coherent",
            vulkan: true,
        },
        Interlocked {
            bc: &dvp,
            entry: "rog_texture_pointer",
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
            let texture = c.entry.starts_with("rog_texture");
            let a = match compile(c.bc, c.entry, &p) {
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
/// profile. One whose varying is flat, which no interpolation touches, records none anywhere.
#[test]
fn a_varying_the_device_interpolates_records_the_device_interpolation_note() {
    let note = PrecisionNote::DeviceInterpolation;
    let profiles = committed_profiles().expect("profiles");
    let devices: Vec<&str> = profiles.iter().map(|(stem, _)| stem.as_str()).collect();
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
    }
}

/// A fragment function blended into an `RG10Unorm_X12` attachment (premultiplied over, and a
/// blend of source alpha into the color channels) records its output's blend note on every
/// profile that orders fetch through rasterization-order input attachments, whatever the module's
/// math and however it computes the output: the shader blends it over the input attachment by the
/// device's arithmetic, where Metal's compiler
/// folds the blend into the function in forms of its own. A `half4` output records
/// `PrecisionNote::ExtendedRangeBlendSource` and a `float4` one
/// `PrecisionNote::ExtendedRangeFloatBlendSource`, never the other's: the fills loaded from a
/// buffer, `rt6_resolve_white`'s constant, a computed `half4`, a `float4` computed by whole-vector
/// arithmetic or with a lane moved alone, under fast math, without it (into the `_sRGB` form too)
/// and naming no math mode. Unblended, the constant output records none.
#[test]
fn a_blended_padded_output_records_its_blend_note() {
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
    let colors = |blended: bool| colors_of(PixelFormat::RG10UnormX12, blended);
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
        inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of([0u32], colors));
        inputs
    };
    let notes = [
        PrecisionNote::ExtendedRangeBlendSource,
        PrecisionNote::ExtendedRangeFloatBlendSource,
    ];
    let (half, float) = (Some(notes[0]), Some(notes[1]));
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !crate::lower::fragment::inputs_order(&p) {
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
            // A device without the raw format's color attachment refuses the blend whatever the
            // source.
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
            let mut c = ColorAttachmentDescriptor::new(PixelFormat::RG10UnormX12);
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
        let srgb = state(&colors_of(PixelFormat::RG10UnormX12SRGB, true), &p);
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
    assert!(checked > 0, "no profile blended over the input attachment");
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
