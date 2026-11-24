//! A fragment function's `[[stencil]]` output end to end: the direct `FragStencilRefEXT` where the
//! device has stencil export, and elsewhere the output drawn by bits
//! (`RuntimeFact::StencilBitDraws`): no stencil reference, the bit test against
//! `PushField::StencilBit`, the static note `PrecisionNote::StencilOutputOverlap`; and the
//! refusals naming the extension where draws by bits do not give Metal's answer: a function that
//! also writes a color or memory, a pipeline made for indirect command buffers, a mesh pipeline.
//! The public fixture `fragment_depth_stencil` returns a depth and a stencil; the engine fixture
//! `stx_stencil` and `stx_stencil_ms` hold a stencil alone, one beside memory writes, and one beside
//! a color. No GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use crate::budget::Budget;
use crate::device::requirements::VkExtension;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Capability, Error, Feature, Needs};
use crate::library::{FunctionArtifact, Library};
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::{
    FunctionDescriptor, MeshPipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::facts::{Mechanism, RuntimeFact};
use crate::pipeline::stage_io::StageBuiltin;
use crate::pipeline::sym::PushField;
use crate::pipeline::testing::{
    compile, engine_dir, fixtures_dir, link_and_validate, llvm_assemble, AssembleError,
};
use crate::pipeline::{PrecisionNote, RuntimeDependency};
use crate::spirv::testing::committed_profiles;

fn assemble(path: &std::path::Path) -> Option<Vec<u8>> {
    match llvm_assemble(path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {}: {why}", path.display()),
    }
}

const ENTRY: &str = "fragment_depth_stencil";

fn stencil_fixture() -> Option<Vec<u8>> {
    assemble(&fixtures_dir().join("fragment_depth_stencil.ll"))
}

/// The refusal of a stencil output on a device without stencil export.
fn needs_export() -> (Feature, Needs) {
    (
        Feature::StencilExport,
        Needs::AnyOf(vec![vec![Capability::Extension(
            VkExtension::ExtShaderStencilExport,
        )]]),
    )
}

/// Whether `result` is the refusal naming the extension.
fn refused<T>(result: Result<T, Error>, what: &str) {
    match result {
        Err(Error::Unsupported(u)) => assert_eq!((u.feature, u.needs), needs_export(), "{what}"),
        Err(other) => panic!("{what}: {other}"),
        Ok(_) => panic!("{what}: not refused"),
    }
}

/// The stencil output drawn by bits: no stencil reference stored, the fact and the bit's push
/// field stated, and the note recorded.
fn assert_drawn_by_bits(a: &FunctionArtifact, what: &str) {
    assert!(
        !a.interface.io.builtins.contains(&StageBuiltin::StencilRef),
        "{what}: {:?}",
        a.interface.io.builtins
    );
    assert!(
        a.facts.contains(&RuntimeFact::StencilBitDraws),
        "{what}: {:?}",
        a.facts
    );
    let pushed = (a.interface.push.fields.iter()).any(|f| f.field == PushField::StencilBit);
    assert!(pushed, "{what}: {:?}", a.interface.push);
    assert!(
        a.precision.contains(&PrecisionNote::StencilOutputOverlap),
        "{what}: {:?}",
        a.precision
    );
}

/// On every committed profile the output is the direct `FragStencilRefEXT` where the device has
/// stencil export, and is drawn by bits elsewhere, the module passing L3 in both forms.
#[test]
fn the_stencil_output_takes_the_extension_or_draws_by_bits() {
    let Some(bc) = stencil_fixture() else {
        return;
    };
    // [exported, by bits]
    let mut forms = [0usize; 2];
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = compile(&bc, ENTRY, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        if p.fields().fragment.stencil_export {
            forms[0] += 1;
            assert!(
                a.interface.io.builtins.contains(&StageBuiltin::StencilRef),
                "{stem}: {:?}",
                a.interface.io.builtins
            );
            assert!(!a.facts.contains(&RuntimeFact::StencilBitDraws), "{stem}");
            assert!(
                !a.precision.contains(&PrecisionNote::StencilOutputOverlap),
                "{stem}"
            );
        } else {
            forms[1] += 1;
            assert_drawn_by_bits(&a, &stem);
        }
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
    }
    // The committed profiles cover each kind of device.
    assert!(forms.iter().all(|&n| n > 0), "{forms:?}");
}

/// The device of committed profile `stem` under `policy`.
fn device(stem: &str, policy: DevicePolicy) -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::from_queries(desc, policy).expect("device")
}

fn function(bc: &[u8], name: &str) -> StageSource {
    StageSource::Function(FunctionDescriptor::new(
        Library::from_air(bc)
            .expect("library")
            .function(name)
            .expect("function"),
    ))
}

/// A render pipeline of `stage_io_vertex_position` and the fixture's fragment, on a device without
/// stencil export (`nvidia-rtx50-linux`), under the runtime's identity domains and the resident
/// tables, with a combined depth/stencil attachment and without one, at one sample and at four:
/// made, its fragment stage depending on `StencilBitDraws` through a push field and noting the
/// overlap; and refused naming the extension where it is made for indirect command buffers.
#[test]
fn a_render_pipeline_without_stencil_export_draws_the_output_by_bits() {
    let (Some(io), Some(stencil)) = (
        assemble(&engine_dir().join("stage_io.ll")),
        stencil_fixture(),
    ) else {
        return;
    };
    for policy in [DevicePolicy::default(), DevicePolicy::RESIDENT_TABLES] {
        let dev = device("nvidia-rtx50-linux", policy);
        assert!(!dev.profile().fields().fragment.stencil_export);
        for attached in [true, false] {
            for samples in [1, 4] {
                let mut d =
                    RenderPipelineDescriptor::new(function(&io, "stage_io_vertex_position"));
                d.fragment = Some(function(&stencil, ENTRY));
                if attached {
                    d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
                    d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
                }
                d.raster_sample_count = samples;
                let what = format!("{policy:?}, attached {attached}, {samples} samples");
                let pipeline = dev
                    .compile_render_pipeline(&d, &Budget::standard())
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let bits =
                    RuntimeDependency::new(RuntimeFact::StencilBitDraws, Mechanism::PushConstant);
                assert!(pipeline.dependencies.contains(&bits), "{what}");
                let notes = &pipeline.stages[1].precision_notes;
                assert!(
                    notes.contains(&PrecisionNote::StencilOutputOverlap),
                    "{what}"
                );
                d.support_indirect_command_buffers = true;
                refused(
                    dev.compile_render_pipeline(&d, &Budget::standard()),
                    &format!("{what}, made for indirect command buffers"),
                );
            }
        }
    }
}

/// Without stencil export, a function that writes its stencil and a color
/// (`stx_stencil_ms_color`), and one that writes its stencil and memory (`stx_stencil_in_order`),
/// refuse the output naming the extension; one that writes its stencil alone
/// (`stx_stencil_by_pixel`), one beside its depth (`stx_stencil_with_depth`) and one that discards
/// (`stx_stencil_discard`) draw it by bits.
#[test]
fn without_stencil_export_a_function_that_writes_a_color_or_memory_refuses_the_output() {
    let (Some(stx), Some(ms)) = (
        assemble(&engine_dir().join("stx_stencil.ll")),
        assemble(&engine_dir().join("stx_stencil_ms.ll")),
    ) else {
        return;
    };
    let mut seen = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if p.fields().fragment.stencil_export {
            continue;
        }
        seen += 1;
        refused(
            compile(&ms, "stx_stencil_ms_color", &p),
            &format!("{stem}: a color"),
        );
        let memory = compile(&stx, "stx_stencil_in_order", &p);
        if p.fields().fragment.pixel_interlock {
            refused(memory, &format!("{stem}: memory"));
        } else {
            // The function's raster order group is refused first, without the interlock.
            match memory {
                Err(Error::Unsupported(u)) => {
                    assert_eq!(u.feature, Feature::RasterOrderGroups, "{stem}: memory")
                }
                Err(other) => panic!("{stem}: memory: {other}"),
                Ok(_) => panic!("{stem}: memory: not refused"),
            }
        }
        for entry in [
            "stx_stencil_by_pixel",
            "stx_stencil_with_depth",
            "stx_stencil_discard",
        ] {
            let what = format!("{stem}: {entry}");
            let a = compile(&stx, entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_drawn_by_bits(&a, &what);
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        }
    }
    assert!(seen > 0, "no committed profile lacks stencil export");
}

/// A mesh pipeline whose fragment function writes its stencil, on a device with mesh shading and
/// without stencil export, refuses the output naming the extension.
#[test]
fn without_stencil_export_a_mesh_pipeline_refuses_the_output() {
    let (Some(outputs), Some(stx)) = (
        super::engine_mesh::fixture("mesh_outputs"),
        assemble(&engine_dir().join("stx_stencil.ll")),
    ) else {
        return;
    };
    let mut seen = 0;
    for (stem, p) in super::engine_mesh::mesh_profiles() {
        if p.fields().fragment.stencil_export {
            continue;
        }
        seen += 1;
        let dev = device(&stem, DevicePolicy::RESIDENT_TABLES);
        let mut d = MeshPipelineDescriptor::new(function(&outputs, "mesh_outputs_points"));
        d.fragment = Some(function(&stx, "stx_stencil_by_pixel"));
        d.stencil_format = Some(PixelFormat::Stencil8);
        refused(dev.compile_mesh_pipeline(&d, &Budget::standard()), &stem);
    }
    assert!(seen > 0, "no mesh profile lacks stencil export");
}
