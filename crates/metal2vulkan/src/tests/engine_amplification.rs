//! Vertex amplification over the owned fixture `engine/vertex_amplification`:
//!
//! - `[[amplification_id]]` is `ViewIndex` (capability `MultiView`, feature `multiview`) in a
//!   vertex, fragment, and post-tessellation vertex function; a profile that does not enable
//!   `multiview` refuses it as `VertexAmplification` needing the feature.
//! - `[[amplification_count]]` is the `SpecKey::AmplificationCount` specialization constant,
//!   default 1, and the function states `RuntimeFact::AmplificationCount`.
//! - A `[[shared]]` output (`air.shared`) is an ordinary output.
//! - A vertex function that writes `[[render_target_array_index]]` is never drawn amplified, so
//!   its amplification id is 0 and it declares no `ViewIndex`; its module converts to MSL and
//!   compiles as MoltenVK's does (`spirv-cross` and `xcrun metal`, compile only).
//! - On the CPU interpreter, a vertex function run in two views reads each view as its
//!   amplification id and the specialized count.
//! - A render pipeline whose `maxVertexAmplificationCount` is above 1 is a multiview pipeline:
//!   it needs `multiview` and as many of `maxMultiviewViewCount` as it amplifies, up to the
//!   device's, and its render plan states the draw's view mask, base layer, and count, a count
//!   above its maximum included; counts Metal would not draw with are `InvalidVertexAmplification`.
//! - What multiview cannot express (view mappings that offset the viewport or name layers out of
//!   order, more amplifications than the device's views, a `Layer` in either stage) draws one
//!   amplification at a time: the plan's variant is `amplification_draws`, whose stages read the
//!   amplification's id and offsets from push fields, and the draw's commands write each
//!   amplification's before its draw. The variant's modules pass L3, its plan settles on itself, a
//!   profile without `shaderOutputLayer` refuses the layered variant, a tessellation pipeline's
//!   such draws are not implemented, and on the CPU interpreter each draw reads the amplification
//!   its push data names.
//!
//! Every module that compiles passes L3 (`spirv-val` and requirements within the profile) under
//! the stage-only numbering. `llvm-as` assembles the fixture; a missing tool skips visibly. No GPU
//! is used.

use std::collections::BTreeMap;
use std::path::Path;

use crate::bind::{
    BindState, DrawCall, IndirectRef, PushConstantWrite, VertexAmplification, ViewMapping,
    ViewSetup,
};
use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::requirements::{VkFeature as F, VkLimit};
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{BindError, Capability, DescriptorError, Error, Feature, Needs};
use crate::library::{FunctionArtifact, Library};
use crate::pipeline::descriptor::{
    FunctionDescriptor, RenderPipelineDescriptor, StageSource, TessellationDescriptor,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::relocate::relocate;
use crate::pipeline::stage_io::{IoRole, StageBuiltin};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, SpecKey, Stage, VaryingKey};
use crate::pipeline::testing::{
    compile, engine_dir, fixtures_dir, llvm_assemble, moltenvk_msl_compile, profile, typed_module,
    typed_module_with, AssembleError,
};
use crate::pipeline::{AmplificationDraws, RenderPipeline};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::testing::check_words_valid;

use crate::device::SpirvCapability;
use crate::spv::Capability as Cap;

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("vertex_amplification.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => {
            panic!("llvm-as rejected vertex_amplification.ll: {why}")
        }
    }
}

fn prof(stem: &str) -> DeviceProfile {
    profile(stem, DevicePolicy::default()).expect("committed profile")
}

fn description(stem: &str) -> DeviceDescription {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    DeviceDescription::from_khronos_json(&text, None).expect("description")
}

fn device(stem: &str) -> Device {
    Device::from_queries(description(stem), DevicePolicy::default()).expect("device")
}

/// `nvidia-rtx50-linux` with `features` removed from its description (with `Multiview`, the
/// multiview features that need it).
fn nvidia_without(features: &[F]) -> Device {
    let mut desc = description("nvidia-rtx50-linux");
    for f in features {
        assert!(desc.features.remove(f), "nvidia-rtx50-linux has {f:?}");
    }
    Device::from_queries(desc, DevicePolicy::default()).expect("device")
}

/// Link `a` alone, relocate it, and run L3; the specialization constants link declared.
fn validate(a: &FunctionArtifact, p: &DeviceProfile) -> Vec<(SpecKey, u64)> {
    let linked = link(&[a], &LinkInputs::standalone(), p).unwrap_or_else(|e| panic!("link: {e}"));
    let words = relocate(a, &linked.table).expect("relocate");
    check_words_valid(&words, &a.requirements, p).unwrap_or_else(|e| panic!("L3: {e}"));
    linked
        .specialization
        .iter()
        .flat_map(|(_, s)| s.constants.iter().map(|c| (c.key, c.default)))
        .collect()
}

/// Every amplified entry reads `ViewIndex` for its amplification id and the specialized count,
/// and a shared output stays an output. `amp_vertex`'s viewport index (`vid & 1`) differs among a
/// primitive's vertices: it is stored to the selected viewport the synthesized geometry stage
/// reads.
#[test]
fn amplification_builtins_are_the_view_index_and_a_specialized_count() {
    let Some(bc) = fixture() else { return };
    for stem in ["nvidia-rtx50-linux", "lavapipe"] {
        let p = prof(stem);
        for (entry, count) in [
            ("amp_vertex", true),
            ("amp_vertex_store", true),
            ("amp_fragment", true),
            ("amp_fragment_layer", false),
            ("amp_post_tessellation", true),
        ] {
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            assert!(
                a.interface.io.builtins.contains(&StageBuiltin::ViewIndex),
                "{stem} {entry}: {:?}",
                a.interface.io.builtins
            );
            assert!(
                a.requirements.features.contains(&F::Multiview),
                "{stem} {entry}"
            );
            assert!(
                a.requirements
                    .spirv_capabilities
                    .contains(&SpirvCapability::from_raw(Cap::MultiView as u32)),
                "{stem} {entry}"
            );
            assert_eq!(
                a.facts.contains(&RuntimeFact::AmplificationCount),
                count,
                "{stem} {entry}"
            );
            let spec = validate(&a, &p);
            assert_eq!(
                spec.contains(&(SpecKey::AmplificationCount, 1)),
                count,
                "{stem} {entry}: {spec:?}"
            );
        }
        // `uv` is `[[shared]]`: the second varying, beside the amplified color; the third is
        // the viewport index as written, and the fourth, behind the synthesized geometry stage,
        // the selected viewport.
        let a = compile(&bc, "amp_vertex", &p).expect("amp_vertex");
        assert_eq!(
            a.interface.io.outputs.len(),
            4,
            "{stem}: color, uv, viewport index"
        );
        assert!(
            !a.interface
                .io
                .builtins
                .contains(&StageBuiltin::ViewportIndex),
            "{stem}"
        );
        assert!(
            a.interface
                .io
                .outputs
                .iter()
                .any(|o| o.role == IoRole::SelectedViewport),
            "{stem}"
        );
    }
}

/// A profile that does not enable `multiview` cannot declare `ViewIndex`.
#[test]
fn amplification_id_needs_multiview() {
    let Some(bc) = fixture() else { return };
    let dev = nvidia_without(&[
        F::Multiview,
        F::MultiviewGeometryShader,
        F::MultiviewTessellationShader,
    ]);
    let p = dev.profile();
    for entry in ["amp_vertex", "amp_fragment", "amp_post_tessellation"] {
        match compile(&bc, entry, p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::VertexAmplification, "{entry}");
                assert_eq!(
                    u.needs,
                    Needs::AnyOf(vec![vec![Capability::Feature(F::Multiview)]]),
                    "{entry}"
                );
            }
            other => panic!("{entry}: {other:?}"),
        }
    }
}

/// A vertex function that writes its render target slice is never drawn amplified: the binder
/// refuses a pipeline declaring `Layer` at a count above 1. Its `[[amplification_id]]` is therefore
/// 0, not `ViewIndex`: SPIRV-Cross gives a vertex function that reads `ViewIndex` a second
/// `[[render_target_array_index]]` output, which Metal refuses (authored case 013406ac failed
/// MoltenVK pipeline creation so). Without `ViewIndex` it needs neither `MultiView` nor
/// `multiview`. A slice the same at every vertex of a primitive is stored to `Layer`, and its
/// `moltenvk-m3` module compiles as MoltenVK's MSL; one that differs among them goes to the
/// selected slice a synthesized geometry stage reads, which `moltenvk-m3` refuses.
#[test]
fn a_vertex_function_writing_its_slice_reads_amplification_id_zero() {
    let Some(bc) = fixture() else { return };
    let public = match llvm_assemble(&fixtures_dir().join("vertex_amplification_id_layer.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as: {why}"),
    };
    let without = nvidia_without(&[
        F::Multiview,
        F::MultiviewGeometryShader,
        F::MultiviewTessellationShader,
    ]);
    let cases = [
        (&bc, "amp_vertex_layer"),
        (&public, "vertex_amplification_id_layer"),
    ];
    for (bc, entry) in cases {
        for p in [
            prof("nvidia-rtx50-linux"),
            prof("lavapipe"),
            without.profile().clone(),
        ] {
            let a = compile(bc, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
            assert!(
                !a.interface.io.builtins.contains(&StageBuiltin::ViewIndex),
                "{entry}: {:?}",
                a.interface.io.builtins
            );
            // `amp_vertex_layer`'s slice (`slices[vid & 1]`) differs among a primitive's
            // vertices: stored to the selected slice; `vertex_amplification_id_layer`'s does not,
            // and is stored to `Layer`.
            let stored = a
                .interface
                .io
                .uniform_outputs
                .contains(&VaryingKey::render_target_array_index());
            assert_eq!(
                a.interface.io.builtins.contains(&StageBuiltin::Layer),
                stored,
                "{entry}"
            );
            assert_eq!(
                a.interface
                    .io
                    .outputs
                    .iter()
                    .any(|o| o.role == IoRole::SelectedLayer),
                !stored,
                "{entry}"
            );
            assert!(!a.requirements.features.contains(&F::Multiview), "{entry}");
            assert!(
                !a.requirements
                    .spirv_capabilities
                    .contains(&SpirvCapability::from_raw(Cap::MultiView as u32)),
                "{entry}"
            );
            validate(&a, &p);
        }
        let p = prof("moltenvk-m3");
        if entry == "amp_vertex_layer" {
            match compile(bc, entry, &p) {
                Err(Error::Unsupported(u)) => {
                    assert_eq!(u.feature, Feature::LayeredRendering, "{entry}");
                    assert_eq!(
                        u.needs,
                        Needs::AnyOf(vec![vec![Capability::Feature(F::GeometryShader)]]),
                        "{entry}"
                    );
                }
                other => panic!("{entry}: {:?}", other.map(|_| ())),
            }
            continue;
        }
        let a = compile(bc, entry, &p).expect("moltenvk-m3");
        let linked = link(&[&a], &LinkInputs::standalone(), &p).expect("link");
        let words = relocate(&a, &linked.table).expect("relocate");
        match moltenvk_msl_compile(&words) {
            Ok(()) => {}
            Err(AssembleError::Missing(why)) => eprintln!("SKIPPED: {why}"),
            Err(AssembleError::Rejected(why)) => panic!("{entry}: MoltenVK's MSL: {why}"),
        }
    }
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// On the CPU interpreter, three vertices drawn in two views: each invocation stores its
/// amplification id plus one and the count (`(count << 16) | (amp + 1)`) at `vid * 8 + amp`, so
/// each view reads its own index and both read the specialized count.
#[test]
fn each_view_reads_its_amplification_id_and_the_specialized_count() {
    let Some(bc) = fixture() else { return };
    let p = prof("moltenvk-m3");
    let (plan, m) = typed_module(&bc, "amp_vertex_store", &p).expect("amp_vertex_store");
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match f.loc {
            PushLoc::Push { offset } => push[offset as usize..(offset + f.size) as usize].fill(0),
            other => panic!("unexpected push field {:?} at {other:?}", f.field),
        }
    }
    let (vertices, views) = (3u32, 2u32);
    for (count, expected_views) in [(views, views), (1, 1)] {
        let res = Resources {
            buffers: vec![Buffer {
                slot: Some(ResourceSlot {
                    stage: Stage::Vertex,
                    kind: ResourceKind::Buffer,
                    index: 0,
                }),
                address: None,
                bytes: vec![0; 32 * 4],
            }],
            push: push.clone(),
            spec: BTreeMap::from([(SpecKey::AmplificationCount, u64::from(count))]),
        };
        for order in [Order::Forward, Order::Reverse] {
            let cfg = Config {
                workgroups: [vertices, 1, expected_views],
                subgroup: SubgroupSize::W32,
                order,
                ..Config::default()
            };
            let out = run(&m, &cfg, res.clone()).expect("run");
            let got = words(&out.buffers[0]);
            let mut want = vec![0u32; 32];
            for v in 0..vertices {
                for a in 0..expected_views {
                    want[(v * 8 + a) as usize] = (count << 16) | (a + 1);
                }
            }
            assert_eq!(got, want, "count {count}, {order:?}");
        }
    }
}

/// A render pipeline of `vertex` and `fragment` from the fixture with
/// `maxVertexAmplificationCount` `max`.
fn render(bc: &[u8], vertex: &str, fragment: Option<&str>, max: u32) -> RenderPipelineDescriptor {
    let lib = Library::from_air(bc).expect("library");
    let stage = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            lib.function(name).expect("function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(stage(vertex));
    d.fragment = fragment.map(stage);
    d.max_vertex_amplification_count = max;
    d
}

/// The value of the render plan's `AmplificationCount` constant.
fn planned_count(p: &RenderPipeline, plan: &crate::bind::BindingPlan) -> u32 {
    let id = p
        .stages
        .iter()
        .flat_map(|m| m.specialization.constants.iter())
        .find(|c| c.key == SpecKey::AmplificationCount)
        .expect("AmplificationCount constant")
        .id;
    let e = plan
        .specialization
        .entries
        .iter()
        .find(|e| e.constant_id == id)
        .expect("AmplificationCount entry");
    let at = e.offset as usize;
    u32::from_le_bytes(
        plan.specialization.data[at..at + 4]
            .try_into()
            .expect("4 bytes"),
    )
}

/// The bind state of `count` amplifications with `mappings` (viewport and render target offsets)
/// in a pass of `renderTargetArrayLength` 2.
fn amplified(count: u32, mappings: &[(u32, u32)]) -> BindState {
    amplified_in(count, mappings, 2)
}

/// [`amplified`] in a pass of `renderTargetArrayLength` `length`.
fn amplified_in(count: u32, mappings: &[(u32, u32)], length: u32) -> BindState {
    let s = BindState {
        render_target_array_length: length,
        vertex_amplification: VertexAmplification::new(
            count,
            mappings
                .iter()
                .map(|&(viewport, layer)| ViewMapping::new(viewport, layer))
                .collect(),
        ),
        ..Default::default()
    };
    s
}

/// A pipeline of `maxVertexAmplificationCount` 2 is a multiview pipeline: it requires
/// `multiview` and two views, every stage module passes L3 under those requirements, and a render
/// plan states the draw's views and count from the encoder's amplification.
/// A pipeline of `maxVertexAmplificationCount` 1 takes no view mapping: a count of 1 whose
/// mapping offsets the render target slice or the viewport draws into the first layer through
/// viewport 0, without multiview and as one draw, as Metal draws it (Metal probes on the M3 Max and
/// an M4: a slice offset of 1, and a viewport offset of 1 over two viewports, moved nothing under
/// such a pipeline and moved the draw under one of 2).
#[test]
fn a_pipeline_that_does_not_amplify_takes_no_view_mapping() {
    let Some(bc) = fixture() else { return };
    for stem in ["nvidia-rtx50-linux", "lavapipe"] {
        let dev = device(stem);
        let d = render(&bc, "amp_vertex", Some("amp_fragment"), 1);
        let p = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        for mappings in [[(0, 2)], [(1, 0)], [(1, 2)]] {
            let plan = p
                .plan(&dev, &amplified_in(1, &mappings, 4).shape())
                .unwrap_or_else(|e| panic!("{stem} {mappings:?}: {e:?}"));
            assert_eq!(plan.views, ViewSetup::default(), "{stem} {mappings:?}");
            assert_eq!(
                plan.variant.amplification_draws, None,
                "{stem} {mappings:?}"
            );
        }
    }
}

#[test]
fn an_amplified_render_pipeline_plans_its_views() {
    let Some(bc) = fixture() else { return };
    for stem in ["nvidia-rtx50-linux", "lavapipe"] {
        let dev = device(stem);
        let d = render(&bc, "amp_vertex", Some("amp_fragment"), 2);
        let p = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(p.graphics.max_vertex_amplification_count, 2, "{stem}");
        assert!(!p.graphics.declares_layer, "{stem}");
        assert!(p.requirements.features.contains(&F::Multiview), "{stem}");
        assert_eq!(
            p.requirements.limits.get(&VkLimit::MaxMultiviewViewCount),
            Some(&2),
            "{stem}"
        );
        for m in &p.stages {
            check_words_valid(&m.spirv, &p.requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{stem} {:?}: L3: {e}", m.stage));
        }

        // Metal's default: one amplification, no multiview.
        let plan = p.plan(&dev, &BindState::default().shape()).expect("plan");
        assert_eq!(plan.views, ViewSetup::default(), "{stem}");
        assert_eq!(planned_count(&p, &plan), 1, "{stem}");
        // Two amplifications into render target slices 3 and 4 of a pass of 5.
        let plan = p
            .plan(&dev, &amplified_in(2, &[(0, 3), (0, 4)], 5).shape())
            .expect("plan");
        assert_eq!((plan.views.view_mask, plan.views.base_layer), (0b11, 3));
        assert_eq!(planned_count(&p, &plan), 2, "{stem}");
        // Asked to (under a visibility query), the same draws one amplification at a time, and a
        // count of 1 stays a draw into one layer.
        let mut one_at_a_time = amplified_in(2, &[(0, 3), (0, 4)], 5);
        one_at_a_time.vertex_amplification.one_at_a_time = true;
        let plan = p.plan(&dev, &one_at_a_time.shape()).expect("plan");
        assert_eq!(plan.views, ViewSetup::default(), "{stem}");
        assert_eq!(
            plan.variant.amplification_draws,
            Some(AmplificationDraws::new(true, false)),
            "{stem}"
        );
        let mut one = amplified_in(1, &[(0, 2)], 4);
        one.vertex_amplification.one_at_a_time = true;
        let plan = p.plan(&dev, &one.shape()).expect("plan");
        assert_eq!((plan.views.view_mask, plan.views.base_layer), (0, 2));
        assert_eq!(plan.variant.amplification_draws, None, "{stem}");
        // One into slice 2 of a pass of 4, without multiview, and into the first layer where the
        // offset is at or past the pass's length, as Metal renders it (a Metal probe on an M4).
        for (length, base_layer) in [(4, 2), (2, 0), (0, 0)] {
            let plan = p
                .plan(&dev, &amplified_in(1, &[(0, 2)], length).shape())
                .expect("plan");
            assert_eq!(
                (plan.views.view_mask, plan.views.base_layer),
                (0, base_layer),
                "{stem} {length}"
            );
            assert_eq!(planned_count(&p, &plan), 1, "{stem}");
        }

        // A count above the maximum draws that many amplifications, as plain Metal draws it
        // where its validation stops it (a Metal probe): as views where the device has them, and
        // every one into the first slice under `nil` mappings.
        let plan = p
            .plan(&dev, &amplified_in(3, &[(0, 0), (0, 1), (0, 2)], 4).shape())
            .expect("plan");
        assert_eq!((plan.views.view_mask, plan.views.base_layer), (0b111, 0));
        assert_eq!(planned_count(&p, &plan), 3, "{stem}");
        let plan = p.plan(&dev, &amplified(3, &[]).shape()).expect("plan");
        assert_eq!(plan.views, ViewSetup::default(), "{stem}");
        assert_eq!(planned_count(&p, &plan), 3, "{stem}");
        // Counts Metal does not draw this pipeline with.
        for (count, mappings) in [(0, vec![]), (2, vec![(0, 0)])] {
            let err = p
                .plan(&dev, &amplified(count, &mappings).shape())
                .expect_err("invalid amplification");
            assert!(
                matches!(err, BindError::InvalidVertexAmplification { max: 2, .. }),
                "{stem} {count} {mappings:?}: {err:?}"
            );
        }
        // Mappings multiview cannot express draw one amplification at a time: every amplification
        // into one slice (`nil`), a viewport offset, slices out of order, and an offset past the
        // pass's length, whose amplification renders the first layer. The pipeline compiled without
        // the variant does not draw them.
        for (count, mappings, viewports) in [
            (2, vec![], false),
            (2, vec![(0, 0), (1, 1)], true),
            (2, vec![(0, 1), (0, 0)], false),
            (2, vec![(0, 1), (0, 2)], false),
        ] {
            let what = format!("{stem} {count} {mappings:?}");
            let state = amplified(count, &mappings);
            let plan = p
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert_eq!(plan.views, ViewSetup::default(), "{what}");
            assert_eq!(
                plan.variant.amplification_draws,
                Some(AmplificationDraws::new(true, viewports)),
                "{what}"
            );
            assert_eq!(planned_count(&p, &plan), count, "{what}");
            assert_eq!(
                plan.instantiate_draw(&state, &triangle()),
                Err(BindError::NeedsVariant {
                    variant: Box::new(plan.variant.clone())
                }),
                "{what}"
            );
        }
    }
}

/// A direct draw of one triangle.
fn triangle() -> DrawCall {
    DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 0,
    }
}

/// What multiview cannot express draws one amplification at a time: more amplifications than the
/// device's views (`lavapipe` has six), and a render target slice a stage chooses or reads,
/// amplified or into another layer than the first. A profile without `multiview` refuses the
/// pipeline, and a maximum of 0 is invalid.
#[test]
fn amplified_render_pipelines_draw_what_multiview_cannot_express_one_at_a_time() {
    let Some(bc) = fixture() else { return };
    let budget = Budget::standard();
    let refusal = |r: Result<RenderPipeline, Error>| match r {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::VertexAmplification);
            u.needs
        }
        other => panic!("{other:?}"),
    };
    let consecutive = |count: u32| -> Vec<(u32, u32)> { (0..count).map(|a| (0, a)).collect() };
    let lavapipe = device("lavapipe");
    let p = lavapipe
        .compile_render_pipeline(&render(&bc, "amp_vertex", None, 8), &budget)
        .expect("eight amplifications on six views");
    assert_eq!(
        p.requirements.limits.get(&VkLimit::MaxMultiviewViewCount),
        Some(&6)
    );
    let plan = p
        .plan(&lavapipe, &amplified_in(6, &consecutive(6), 8).shape())
        .expect("six");
    assert_eq!(
        (plan.views.view_mask, plan.views.base_layer),
        (0b11_1111, 0)
    );
    assert_eq!(plan.variant, VariantKey::new());
    let plan = p
        .plan(&lavapipe, &amplified_in(8, &consecutive(8), 8).shape())
        .expect("eight");
    assert_eq!(plan.views, ViewSetup::default());
    assert_eq!(
        plan.variant.amplification_draws,
        Some(AmplificationDraws::new(true, false))
    );
    assert_eq!(planned_count(&p, &plan), 8);

    let dev = device("nvidia-rtx50-linux");
    for (vertex, fragment) in [
        ("amp_vertex_layer", None),
        ("amp_vertex", Some("amp_fragment_layer")),
    ] {
        let p = dev
            .compile_render_pipeline(&render(&bc, vertex, fragment, 2), &budget)
            .unwrap_or_else(|e| panic!("{vertex} {fragment:?}: {e}"));
        assert!(p.graphics.declares_layer, "{vertex} {fragment:?}");
        // At a count of 1 into the first layer the layer is ordinary layered rendering;
        // amplified, or into another layer, where Metal adds the offset to the slice chosen and
        // read, each amplification is a draw of its own (a multiview pipeline cannot declare
        // `Layer`).
        let plan = p
            .plan(&dev, &BindState::default().shape())
            .unwrap_or_else(|e| panic!("{vertex} {fragment:?}: {e:?}"));
        assert_eq!(plan.views, ViewSetup::default(), "{vertex} {fragment:?}");
        assert_eq!(
            plan.variant.amplification_draws, None,
            "{vertex} {fragment:?}"
        );
        for state in [amplified(2, &[(0, 0), (0, 1)]), amplified(1, &[(0, 1)])] {
            let plan = p
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{vertex} {fragment:?}: {e:?}"));
            assert_eq!(plan.views, ViewSetup::default(), "{vertex} {fragment:?}");
            assert_eq!(
                plan.variant.amplification_draws,
                Some(AmplificationDraws::new(true, false)),
                "{vertex} {fragment:?}"
            );
        }
    }
    let without = nvidia_without(&[
        F::Multiview,
        F::MultiviewGeometryShader,
        F::MultiviewTessellationShader,
    ]);
    assert_eq!(
        refusal(without.compile_render_pipeline(&render(&bc, "amp_vertex", None, 2), &budget)),
        Needs::AnyOf(vec![vec![Capability::Feature(F::Multiview)]])
    );
    assert!(matches!(
        dev.compile_render_pipeline(&render(&bc, "amp_vertex", None, 0), &budget),
        Err(Error::InvalidDescriptor(
            DescriptorError::VertexAmplificationCount { count: 0 }
        ))
    ));
}

/// The `u32` at push offset `offset` once `writes` are applied in order: the last covering
/// write's.
fn pushed_word<'a>(
    writes: impl IntoIterator<Item = &'a PushConstantWrite>,
    offset: u32,
) -> Option<u32> {
    let mut word = None;
    for w in writes {
        let end = w.offset as usize + w.bytes.len();
        if w.offset <= offset && offset as usize + 4 <= end {
            let at = (offset - w.offset) as usize;
            word = Some(u32::from_le_bytes(
                w.bytes[at..at + 4].try_into().expect("4 bytes"),
            ));
        }
    }
    word
}

/// The push offset of each amplification field the stages of `p` read.
fn amplification_fields(p: &RenderPipeline) -> Vec<(Stage, PushField, u32)> {
    p.interface
        .push
        .iter()
        .flat_map(|pl| pl.fields.iter().map(move |f| (pl.stage, f)))
        .filter(|(_, f)| {
            matches!(
                f.field,
                PushField::AmplificationId
                    | PushField::AmplificationLayerOffset
                    | PushField::AmplificationViewportOffset
            )
        })
        .map(|(stage, f)| match f.loc {
            PushLoc::Push { offset } => (stage, f.field, offset),
            other => panic!("{:?} at {other:?}", f.field),
        })
        .collect()
}

/// A plan whose amplification multiview cannot express states the variant that draws one
/// amplification at a time, which the pipeline compiled without it does not draw. Compiled, the
/// variant reads its amplification id from push data in both stages (no `ViewIndex`), adds the
/// mappings' offsets to the viewport the vertex function selects and writes the slice offset as
/// `Layer` in a pass of two layers (none in a pass of one, where a fragment function that reads
/// the slice reads the offset), passes L3, and states `RuntimeFact::AmplificationDraws`. Its plan
/// for the same state settles on itself, and the draw's commands write each amplification's id
/// and offsets in each stage that reads them, the first's in the draw's own push data.
#[test]
fn a_variant_draws_each_amplification_with_its_push_fields() {
    let Some(bc) = fixture() else { return };
    let budget = Budget::standard();
    for stem in ["nvidia-rtx50-linux", "lavapipe"] {
        let dev = device(stem);
        // Each pipeline, and whether its vertex function selects a viewport itself.
        for (vertex, fragment, selects) in [
            ("amp_vertex", "amp_fragment", true),
            ("amp_vertex", "amp_fragment_layer", true),
            ("amp_vertex_layer", "amp_fragment_layer", false),
        ] {
            let d = render(&bc, vertex, Some(fragment), 4);
            let base = dev
                .compile_render_pipeline(&d, &budget)
                .unwrap_or_else(|e| panic!("{stem} {vertex} {fragment}: {e}"));
            for (count, mappings, length) in [
                (3, vec![(1, 1), (0, 0), (2, 1)], 2),
                (2, vec![(0, 1), (0, 0)], 1),
                (2, vec![], 2),
            ] {
                let what = format!("{stem} {vertex} {fragment} {count} {mappings:?} {length}");
                let state = amplified_in(count, &mappings, length);
                let plan = base
                    .plan(&dev, &state.shape())
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let layered = length > 1;
                let viewports = mappings.iter().any(|&(viewport, _)| viewport != 0);
                let draws = AmplificationDraws::new(layered, viewports);
                assert_eq!(plan.variant.amplification_draws, Some(draws), "{what}");
                let key = plan.variant.clone();
                let keys = dev
                    .render_variant_key(&d, &key, &budget, &crate::pipeline::StageEntries::new())
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                for inputs in [&keys.vertex_inputs, &keys.fragment_inputs] {
                    let inputs = inputs.as_ref().expect("inputs");
                    assert_eq!(inputs.amplification_draws, Some(draws), "{what}");
                }
                let p = dev
                    .compile_render_variant(&d, &key, &budget)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(p.key, keys.pipeline, "{what}");
                for m in &p.stages {
                    check_words_valid(&m.spirv, &p.requirements, dev.profile())
                        .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
                }
                // No stage reads `ViewIndex`.
                assert!(
                    !p.requirements
                        .spirv_capabilities
                        .contains(&SpirvCapability::from_raw(Cap::MultiView as u32)),
                    "{what}"
                );
                assert!(
                    p.dependencies
                        .iter()
                        .any(|d| d.fact == RuntimeFact::AmplificationDraws { layered, viewports }),
                    "{what}"
                );
                assert_eq!(p.graphics.declares_layer, layered, "{what}");
                assert_eq!(
                    p.graphics.writes_viewport_index,
                    selects || viewports,
                    "{what}"
                );

                let plan = p
                    .plan(&dev, &state.shape())
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                assert_eq!(plan.variant, key, "{what}");
                assert_eq!(plan.views, ViewSetup::default(), "{what}");
                let commands = plan
                    .instantiate_draw(&state, &triangle())
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                assert_eq!(commands.amplifications.len(), count as usize, "{what}");
                let fields = amplification_fields(&p);
                for stage in [Stage::Vertex, Stage::Fragment] {
                    assert!(
                        fields
                            .iter()
                            .any(|&(s, f, _)| s == stage && f == PushField::AmplificationId),
                        "{what} {stage:?}: {fields:?}"
                    );
                }
                let mapping = |a: u32| mappings.get(a as usize).copied().unwrap_or((0, 0));
                for a in 0..count {
                    let (viewport, layer) = mapping(a);
                    for &(stage, field, offset) in &fields {
                        let want = match field {
                            PushField::AmplificationId => a,
                            PushField::AmplificationLayerOffset => layer,
                            _ => viewport,
                        };
                        let writes = commands
                            .push_constants
                            .iter()
                            .chain(&commands.amplifications[a as usize]);
                        assert_eq!(
                            pushed_word(writes, offset),
                            Some(want),
                            "{what} amplification {a} {stage:?} {field:?}"
                        );
                        if a == 0 {
                            assert_eq!(
                                pushed_word(&commands.push_constants, offset),
                                Some(want),
                                "{what} {stage:?} {field:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// The variant's layered `Layer` needs `shaderOutputLayer` or `geometryShader`, which
/// `runtime-floor` without `shaderOutputLayer` (below the runtime's floor) both lacks; and a
/// tessellation pipeline is drawn amplified only as views (`lavapipe` has
/// `multiviewTessellationShader`).
#[test]
fn amplification_draws_refuse_what_the_variant_cannot_be() {
    let Some(bc) = fixture() else { return };
    let budget = Budget::standard();
    let dev = below_the_floor();
    let d = render(&bc, "amp_vertex_store", None, 2);
    let p = dev.compile_render_pipeline(&d, &budget).expect("base");
    let state = amplified(2, &[(0, 1), (0, 0)]);
    let plan = p.plan(&dev, &state.shape()).expect("plan");
    match dev.compile_render_variant(&d, &plan.variant, &budget) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::VertexAmplification);
            assert_eq!(u.needs, output_layer_or_geometry());
        }
        other => panic!("{:?}", other.map(|_| ())),
    }

    let dev = device("lavapipe");
    let mut d = render(&bc, "amp_post_tessellation", None, 2);
    d.tessellation = Some(TessellationDescriptor::default());
    let p = dev
        .compile_render_pipeline(&d, &budget)
        .expect("tessellation pipeline");
    let plan = p
        .plan(&dev, &amplified(2, &[(0, 0), (0, 1)]).shape())
        .expect("views");
    assert_eq!(plan.views.view_mask, 0b11);
    match p.plan(&dev, &amplified(2, &[(0, 1), (0, 0)]).shape()) {
        Err(BindError::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::VertexAmplification);
            assert_eq!(u.needs, Needs::NotImplemented);
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
}

/// `runtime-floor` without `shaderOutputLayer`, which the runtime's floor requires: a device below
/// the floor, with neither the output nor `geometryShader`.
fn below_the_floor() -> Device {
    let mut desc = description("runtime-floor");
    assert!(desc.features.remove(&F::ShaderOutputLayer));
    assert!(!desc.features.contains(&F::GeometryShader));
    Device::from_queries(desc, DevicePolicy::default()).expect("device")
}

/// What a variant that stores a layered amplification's slice needs on a device without
/// `shaderOutputLayer`: the feature, or the geometry stage that stores the slice instead.
fn output_layer_or_geometry() -> Needs {
    Needs::AnyOf(vec![
        vec![Capability::Feature(F::ShaderOutputLayer)],
        vec![Capability::Feature(F::GeometryShader)],
    ])
}

/// On a profile without `shaderOutputLayer`, every amplified draw is drawn as Metal or refused
/// typed, never drawn into another slice: as views where multiview draws Metal's answer; one
/// amplification at a time in a pass of one layer or none, whose variant writes no `Layer` and
/// carries no geometry stage (every amplification renders the pass's one layer, as Metal's do:
/// Metal probe case `amp2_len1` on the M3 Max and an M4); and in a pass of more layers, whose
/// variant must store the amplification's slice as `Layer`, through the synthesized geometry
/// stages where the device has `geometryShader` (`lavapipe` without the output), and otherwise
/// (`runtime-floor` without `shaderOutputLayer`) `Unsupported(VertexAmplification)` needing either
/// feature.
#[test]
fn a_profile_without_shader_output_layer_draws_or_refuses_every_amplification() {
    let Some(bc) = fixture() else { return };
    let budget = Budget::standard();
    let mut lavapipe = description("lavapipe");
    assert!(lavapipe.features.remove(&F::ShaderOutputLayer));
    assert!(lavapipe.features.contains(&F::GeometryShader));
    let lavapipe = Device::from_queries(lavapipe, DevicePolicy::default()).expect("device");
    for (stem, dev, geometry) in [
        (
            "runtime-floor without shaderOutputLayer",
            below_the_floor(),
            false,
        ),
        ("lavapipe without shaderOutputLayer", lavapipe, true),
    ] {
        let d = render(&bc, "amp_vertex_store", None, 2);
        let p = dev.compile_render_pipeline(&d, &budget).expect("base");
        let (mut views, mut drawn, mut layered, mut refused) = (0, 0, 0, 0);
        for length in 0..4 {
            for mappings in [
                vec![(0, 0), (0, 1)],
                vec![(0, 1), (0, 2)],
                vec![(0, 1), (0, 0)],
                vec![(0, 0), (0, 2)],
                vec![(0, 0), (0, 0)],
                vec![],
            ] {
                let what = format!("{stem} {mappings:?} in a pass of {length} layers");
                let plan = p
                    .plan(&dev, &amplified_in(2, &mappings, length).shape())
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let Some(draws) = plan.variant.amplification_draws else {
                    assert_ne!(plan.views.view_mask, 0, "{what}");
                    views += 1;
                    continue;
                };
                assert_eq!(draws.layered, length > 1, "{what}");
                match dev.compile_render_variant(&d, &plan.variant, &budget) {
                    Ok(variant) => {
                        let stages = variant
                            .stages
                            .iter()
                            .filter(|m| m.stage == crate::vk::ShaderStage::GEOMETRY)
                            .count();
                        assert_eq!(variant.graphics.declares_layer, draws.layered, "{what}");
                        if draws.layered {
                            assert!(geometry, "{what}: compiled");
                            assert_eq!(stages, 3, "{what}");
                            layered += 1;
                        } else {
                            assert_eq!(stages, 0, "{what}");
                            drawn += 1;
                        }
                        for m in &variant.stages {
                            check_words_valid(&m.spirv, &variant.requirements, dev.profile())
                                .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
                        }
                    }
                    Err(Error::Unsupported(u)) => {
                        assert!(draws.layered && !geometry, "{what}: refused");
                        assert_eq!(u.feature, Feature::VertexAmplification, "{what}");
                        assert_eq!(u.needs, output_layer_or_geometry(), "{what}");
                        refused += 1;
                    }
                    Err(e) => panic!("{what}: {e}"),
                }
            }
        }
        assert!(views > 0 && drawn > 0, "{stem}: {views} {drawn}");
        if geometry {
            assert!(layered > 0 && refused == 0, "{stem}: {layered} {refused}");
        } else {
            assert!(refused > 0 && layered == 0, "{stem}: {layered} {refused}");
        }
    }
}

#[test]
fn each_amplification_draw_reads_its_pushed_id() {
    let Some(bc) = fixture() else { return };
    let p = prof("moltenvk-m3");
    let mut inputs = LoweringInputs::new();
    inputs.amplification_draws = Some(AmplificationDraws::new(true, true));
    let (plan, m) =
        typed_module_with(&bc, "amp_vertex_store", &p, &inputs).expect("amp_vertex_store");
    let (vertices, count) = (3u32, 3u32);
    let mut out = vec![0u8; 32 * 4];
    for a in 0..count {
        let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
        let mut read_id = false;
        for f in &plan.push.fields {
            let PushLoc::Push { offset } = f.loc else {
                panic!("unexpected push field {:?} at {:?}", f.field, f.loc);
            };
            let value: u32 = match f.field {
                PushField::AmplificationId => {
                    read_id = true;
                    a
                }
                PushField::AmplificationLayerOffset => a,
                PushField::RenderTargetArrayLength => 4,
                PushField::AmplificationViewportOffset | PushField::ViewportCount => 1,
                _ => 0,
            };
            if value != 0 {
                push[offset as usize..offset as usize + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        assert!(read_id, "{:?}", plan.push.fields);
        let res = Resources {
            buffers: vec![Buffer {
                slot: Some(ResourceSlot {
                    stage: Stage::Vertex,
                    kind: ResourceKind::Buffer,
                    index: 0,
                }),
                address: None,
                bytes: out.clone(),
            }],
            push,
            spec: BTreeMap::from([(SpecKey::AmplificationCount, u64::from(count))]),
        };
        let cfg = Config {
            workgroups: [vertices, 1, 1],
            subgroup: SubgroupSize::W32,
            order: Order::Forward,
            ..Config::default()
        };
        out = run(&m, &cfg, res).expect("run").buffers[0].clone();
    }
    let mut want = vec![0u32; 32];
    for v in 0..vertices {
        for a in 0..count {
            want[(v * 8 + a) as usize] = (count << 16) | (a + 1);
        }
    }
    assert_eq!(words(&out), want);
}

/// A multiview draw keeps its last instance index within `maxMultiviewInstanceIndex`
/// (`VUID-vkCmdDraw-maxMultiviewInstanceIndex-02688`; `nvidia-rtx50-linux` reports 2^27 - 1), and an
/// indirect one asks for the CPU read of its arguments; a draw without multiview is unaffected.
#[test]
fn multiview_draws_keep_their_instances_within_the_limit() {
    let Some(bc) = fixture() else { return };
    let dev = device("nvidia-rtx50-linux");
    let p = dev
        .compile_render_pipeline(
            &render(&bc, "amp_vertex", Some("amp_fragment"), 2),
            &Budget::standard(),
        )
        .expect("pipeline");
    let draw = |instance_count: u32, base_instance: u32| DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 3,
        instance_count,
        base_instance,
    };
    let indirect = DrawCall::Indirect {
        indirect: IndirectRef::new(7, 0),
        indexed: None,
    };
    let limit = (1u32 << 27) - 1;

    let two = amplified(2, &[(0, 0), (0, 1)]);
    let plan = p.plan(&dev, &two.shape()).expect("plan");
    assert_ne!(plan.views.view_mask, 0);
    plan.instantiate_draw(&two, &draw(1, limit))
        .expect("last instance at the limit");
    plan.instantiate_draw(&two, &draw(0, u32::MAX))
        .expect("no instances");
    match plan.instantiate_draw(&two, &draw(1, 1 << 27)) {
        Err(BindError::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::VertexAmplification);
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![vec![Capability::Limit {
                    limit: VkLimit::MaxMultiviewInstanceIndex,
                    required: 1 << 27,
                    available: u64::from(limit),
                }]])
            );
        }
        other => panic!("{other:?}"),
    }
    assert!(
        plan.instantiate_draw(&two, &indirect)
            .expect("indirect")
            .indirect_needs_arguments
    );

    let one = BindState::default();
    let plan = p.plan(&dev, &one.shape()).expect("plan");
    plan.instantiate_draw(&one, &draw(1, 1 << 27))
        .expect("not multiview");
    assert!(
        !plan
            .instantiate_draw(&one, &indirect)
            .expect("indirect")
            .indirect_needs_arguments
    );
}
