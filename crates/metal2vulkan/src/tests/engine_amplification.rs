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

use crate::device::requirements::VkFeature as F;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{Capability, Error, Feature, Needs};
use crate::library::FunctionArtifact;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::relocate::relocate;
use crate::pipeline::stage_io::StageBuiltin;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, fixtures_dir, llvm_assemble, moltenvk_msl_compile, profile, typed_module,
    typed_module_with, AssembleError,
};
use crate::pipeline::AmplificationDraws;
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
    profile(stem, DevicePolicy::RESIDENT_TABLES).expect("committed profile")
}

fn description(stem: &str) -> DeviceDescription {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    DeviceDescription::from_khronos_json(&text, None).expect("description")
}

/// `nvidia-rtx50-linux` with `features` removed from its description (with `Multiview`, the
/// multiview features that need it).
fn nvidia_without(features: &[F]) -> Device {
    let mut desc = description("nvidia-rtx50-linux");
    for f in features {
        assert!(desc.features.remove(f), "nvidia-rtx50-linux has {f:?}");
    }
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device")
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
/// and a shared output stays an output. `amp_vertex`'s viewport index (`iid & 1`) is the same at
/// every vertex of a primitive: it is stored to `ViewportIndex`.
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
        // the viewport index as written, beside the `ViewportIndex` it stores.
        let a = compile(&bc, "amp_vertex", &p).expect("amp_vertex");
        assert_eq!(
            a.interface.io.outputs.len(),
            3,
            "{stem}: color, uv, viewport index"
        );
        assert!(
            a.interface
                .io
                .builtins
                .contains(&StageBuiltin::ViewportIndex),
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
/// `multiview`. Each fixture's slice is stored to `Layer`, and its `moltenvk-m3` module compiles
/// as MoltenVK's MSL.
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
            assert!(
                a.interface.io.builtins.contains(&StageBuiltin::Layer),
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
