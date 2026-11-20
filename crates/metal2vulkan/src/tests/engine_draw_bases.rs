//! A draw's base vertex and base instance in a vertex function.
//!
//! The owned fixture `engine/rvc_draw_bases` reads `[[base_vertex]]` and `[[base_instance]]`
//! as `uint` (`draw_bases_in_a_vertex_function`) and as `ushort`
//! (`narrow_draw_bases_in_a_vertex_function`) in a vertex function, and `[[base_instance]]` in a
//! post-tessellation vertex function (`base_instance_after_tessellation`). On every committed
//! profile that compiles it, each module passes L3 (`spirv-val` and requirements within the
//! profile), and reads the bases from Vulkan's `BaseVertex` and `BaseInstance` with
//! `shaderDrawParameters` and from `PushField::BaseVertexInstance` without it; an evaluation shader
//! always reads the push field. On the CPU interpreter, each vertex of a draw from a large first
//! vertex and instance reads the draw's bases, the narrow entry their low 16 bits; and a draw of
//! patches writes its base instance into the evaluation stage's push range. No GPU is used.
//!
//! `llvm-as` assembles the fixture; a missing tool skips visibly.

use std::sync::Arc;

use crate::bind::{BindState, DrawCall};
use crate::budget::{Budget, SPIRV_WORDS};
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{Error, Feature};
use crate::library::{FunctionArtifact, FunctionConstantValues, Library};
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::descriptor::{
    FunctionDescriptor, RenderPipelineDescriptor, StageSource, TessellationDescriptor,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::stage_io::StageBuiltin;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    engine_dir, link_and_validate, llvm_assemble, meter, typed_module_with, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::testing::committed_profiles;
use crate::vk;

const WIDE: &str = "draw_bases_in_a_vertex_function";
const NARROW: &str = "narrow_draw_bases_in_a_vertex_function";
const TESSELLATED: &str = "base_instance_after_tessellation";

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("rvc_draw_bases.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected rvc_draw_bases.ll: {why}"),
    }
}

fn compile(bc: &[u8], entry: &str, p: &DeviceProfile) -> Result<Arc<FunctionArtifact>, Error> {
    compile_bitcode(
        bc,
        EntryRef::Name(entry),
        &FunctionConstantValues::defaults(),
        p,
        &LoweringInputs::new(),
        &mut meter(),
        SPIRV_WORDS,
    )
}

fn placed(a: &FunctionArtifact, f: PushField) -> bool {
    a.interface.push.fields.iter().any(|x| x.field == f)
}

/// On every committed profile with `vertexPipelineStoresAndAtomics` (the fixture stores from its
/// vertex functions), both vertex entries pass L3 and state the draw's bases as the binder must
/// supply them: `BaseVertex` and `BaseInstance` (needing `shaderDrawParameters`) on a profile with
/// it, the `BaseVertexInstance` push field on one without it; the runtime fact either way. A
/// profile without the stores refuses the entries for them alone.
#[test]
fn vertex_entries_read_the_builtins_or_the_push_lanes() {
    let Some(bc) = fixture() else { return };
    let (mut builtins, mut pushed) = (0, 0);
    for (stem, p) in committed_profiles().expect("committed profiles") {
        let draw = p.fields().draw_parameters;
        for entry in [WIDE, NARROW] {
            let what = format!("{stem} {entry}");
            let a = match compile(&bc, entry, &p) {
                Ok(a) => a,
                Err(Error::Unsupported(u))
                    if !p.fields().leaf.vertex_stores_and_atomics
                        && u.feature == Feature::StageStoresAndAtomics =>
                {
                    continue
                }
                Err(e) => panic!("{what}: {e}"),
            };
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert!(a.facts.contains(&RuntimeFact::BaseVertexInstance), "{what}");
            assert_eq!(placed(&a, PushField::BaseVertexInstance), !draw, "{what}");
            let io = &a.interface.io;
            assert_eq!(
                io.builtins.contains(&StageBuiltin::BaseVertex),
                draw,
                "{what}"
            );
            assert_eq!(
                io.builtins.contains(&StageBuiltin::BaseInstance),
                draw,
                "{what}"
            );
            assert_eq!(
                a.requirements
                    .features
                    .contains(&VkFeature::ShaderDrawParameters),
                draw,
                "{what}"
            );
            if draw {
                builtins += 1;
            } else {
                pushed += 1;
            }
        }
    }
    assert!(builtins > 0 && pushed > 0, "{builtins} {pushed}");
}

/// The words of `bytes`.
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// On the CPU interpreter: four vertices of two instances drawn from the first vertex and instance
/// `first`, on every committed profile with `vertexPipelineStoresAndAtomics`. Vertex `vid` of
/// instance `iid` writes `vid`, `iid` and the draw's bases (the builtins' on a profile with
/// `shaderDrawParameters`, the push lanes the binder writes on one without) to slot `(iid % 4) *
/// 8 + vid % 8`; the narrow entry's bases are their low 16 bits. The first vertex `0xfffffffa` is
/// Metal's `vertexStart` probe; `0xfffffffd` stands for the push lane of an indexed draw's base
/// vertex of -3 (the interpreter draws no index buffer).
#[test]
fn each_vertex_reads_its_draw_s_bases() {
    let Some(bc) = fixture() else { return };
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("committed profiles") {
        if !p.fields().leaf.vertex_stores_and_atomics {
            continue;
        }
        for (entry, mask) in [(WIDE, u32::MAX), (NARROW, 0xffff)] {
            let (plan, m) = typed_module_with(&bc, entry, &p, &LoweringInputs::new())
                .unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            for first in [
                [0xffff_fffa_u32, 7],
                [0xffff_fffa, 0x1_0007],
                [0xffff_fffd, 2],
            ] {
                let what = format!("{stem} {entry} from {first:x?}");
                let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
                for f in &plan.push.fields {
                    let PushLoc::Push { offset } = f.loc else {
                        panic!("{what}: {:?} spilled to the ArgumentTable", f.field);
                    };
                    let at = offset as usize;
                    match f.field {
                        PushField::SlotBindOffset(_) => {}
                        PushField::BaseVertexInstance => {
                            push[at..at + 4].copy_from_slice(&first[0].to_le_bytes());
                            push[at + 4..at + 8].copy_from_slice(&first[1].to_le_bytes());
                        }
                        other => panic!("{what}: unexpected push field {other:?}"),
                    }
                }
                let res = Resources {
                    buffers: vec![Buffer {
                        slot: Some(ResourceSlot {
                            stage: Stage::Vertex,
                            kind: ResourceKind::Buffer,
                            index: 0,
                        }),
                        address: None,
                        bytes: vec![0xab; 512],
                    }],
                    push,
                    spec: Default::default(),
                };
                let cfg = Config {
                    workgroups: [4, 2, 1],
                    first,
                    ..Config::default()
                };
                let out = run(&m, &cfg, res).unwrap_or_else(|e| panic!("{what}: {e}"));
                let mut want = vec![0xabab_abab_u32; 128];
                for i in 0..2u32 {
                    for v in 0..4u32 {
                        let (vid, iid) = (first[0].wrapping_add(v), first[1] + i);
                        let slot = 4 * ((iid % 4) * 8 + vid % 8) as usize;
                        want[slot..slot + 4].copy_from_slice(&[
                            vid,
                            iid,
                            first[0] & mask,
                            first[1] & mask,
                        ]);
                    }
                }
                assert_eq!(words(&out.buffers[0]), want, "{what}");
                ran += 1;
            }
        }
    }
    assert!(ran > 0, "no profile has vertexPipelineStoresAndAtomics");
}

/// A post-tessellation vertex function's `[[base_instance]]` is the push field's second lane on
/// every committed profile with `tessellationShader` (an evaluation shader has no `BaseInstance`),
/// and its module passes L3.
#[test]
fn the_base_instance_after_tessellation_is_the_push_lane() {
    let Some(bc) = fixture() else { return };
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("committed profiles") {
        if !p.fields().fragment.tessellation_shader {
            continue;
        }
        let a = compile(&bc, TESSELLATED, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(placed(&a, PushField::BaseVertexInstance), "{stem}");
        assert!(a.facts.contains(&RuntimeFact::BaseVertexInstance), "{stem}");
        assert!(
            !a.interface
                .io
                .builtins
                .contains(&StageBuiltin::BaseInstance),
            "{stem}"
        );
        ran += 1;
    }
    assert!(ran > 0, "no profile has tessellationShader");
}

fn device(stem: &str) -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::from_queries(desc, DevicePolicy::default()).expect("device")
}

/// A draw of patches from base instance 7 writes `[patch_start, 7]` into the evaluation stage's
/// `BaseVertexInstance` lanes, so the post-tessellation function reads 7.
#[test]
fn a_draw_of_patches_pushes_its_base_instance() {
    let Some(bc) = fixture() else { return };
    let lib = Library::from_air(&bc).expect("library");
    for stem in ["moltenvk-m3", "lavapipe", "lunarg-desktop-baseline-2023"] {
        let dev = device(stem);
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
            lib.function(TESSELLATED).expect("function"),
        )));
        d.rasterization_enabled = false;
        d.tessellation = Some(TessellationDescriptor::default());
        let p = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        // Without a factor buffer the zero buffer stands in: the patches are culled, and the
        // push lanes are written all the same.
        let state = BindState::default();
        let plan = p.plan(&dev, &state.shape()).expect("plan");
        let draw = DrawCall::Patches {
            control_points: 3,
            patch_start: 2,
            patch_count: 5,
            patch_index: None,
            instance_count: 2,
            base_instance: 7,
            control_point_index: None,
        };
        let cmds = plan.instantiate_draw(&state, &draw).expect("draw");
        let field = p
            .interface
            .push
            .iter()
            .filter(|pl| pl.stage == Stage::Vertex)
            .flat_map(|pl| &pl.fields)
            .find(|f| f.field == PushField::BaseVertexInstance)
            .unwrap_or_else(|| panic!("{stem}: no BaseVertexInstance field"));
        let PushLoc::Push { offset } = field.loc else {
            panic!("{stem}: the field spilled: {:?}", field.loc);
        };
        let write = cmds
            .push_constants
            .iter()
            .find(|w| w.stages == vk::ShaderStage::TESSELLATION_EVALUATION.flags())
            .unwrap_or_else(|| panic!("{stem}: no evaluation push"));
        let at = (offset - write.offset) as usize;
        assert_eq!(words(&write.bytes[at..at + 8]), [2, 7], "{stem}: the lanes");
        assert!(!cmds.indirect_needs_arguments, "{stem}");
    }
}
