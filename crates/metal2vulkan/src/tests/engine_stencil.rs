//! A fragment function's `[[stencil]]` output end to end: the direct `FragStencilRefEXT` where the
//! device has stencil export, and elsewhere a refusal naming the extension, of the function and of
//! its render pipeline. The public fixture `fragment_depth_stencil` returns a depth and a stencil.
//! No GPU is used.
//!
//! `llvm-as` assembles the fixture; a missing tool skips visibly.

use crate::budget::Budget;
use crate::device::requirements::VkExtension;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Capability, Error, Feature, Needs};
use crate::library::Library;
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::{FunctionDescriptor, RenderPipelineDescriptor, StageSource};
use crate::pipeline::stage_io::StageBuiltin;
use crate::pipeline::testing::{
    compile, engine_dir, fixtures_dir, link_and_validate, llvm_assemble, AssembleError,
};
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

/// On every committed profile the output is the direct `FragStencilRefEXT` where the device has
/// stencil export, the module passing L3, and is otherwise refused naming the extension alone.
#[test]
fn the_stencil_output_takes_the_extension_or_a_refusal_naming_it() {
    let Some(bc) = stencil_fixture() else {
        return;
    };
    // [exported, refused]
    let mut forms = [0usize; 2];
    for (stem, p) in committed_profiles().expect("profiles") {
        let result = compile(&bc, ENTRY, &p);
        if p.fields().fragment.stencil_export {
            forms[0] += 1;
            let a = result.unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert!(
                a.interface.io.builtins.contains(&StageBuiltin::StencilRef),
                "{stem}: {:?}",
                a.interface.io.builtins
            );
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        } else {
            forms[1] += 1;
            match result {
                Err(Error::Unsupported(u)) => {
                    assert_eq!((u.feature, u.needs), needs_export(), "{stem}")
                }
                other => panic!("{stem}: {:?}", other.map(|_| ())),
            }
        }
    }
    // The committed profiles cover each kind of device.
    assert!(forms.iter().all(|&n| n > 0), "{forms:?}");
}

/// A render pipeline of `stage_io_vertex_position` and the fixture's fragment, on a device without
/// stencil export (`nvidia-rtx50-linux`), is refused naming the extension, with a combined
/// depth/stencil attachment and without one, at one sample and at four.
#[test]
fn a_render_pipeline_without_stencil_export_is_refused_naming_it() {
    let (Some(io), Some(stencil)) = (
        assemble(&engine_dir().join("stage_io.ll")),
        stencil_fixture(),
    ) else {
        return;
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/nvidia-rtx50-linux.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let dev = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device");
    assert!(!dev.profile().fields().fragment.stencil_export);
    let function = |bc: &[u8], name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        ))
    };
    for attached in [true, false] {
        for samples in [1, 4] {
            let mut d = RenderPipelineDescriptor::new(function(&io, "stage_io_vertex_position"));
            d.fragment = Some(function(&stencil, ENTRY));
            if attached {
                d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
                d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
            }
            d.raster_sample_count = samples;
            let what = format!("attached {attached}, {samples} samples");
            match dev.compile_render_pipeline(&d, &Budget::standard()) {
                Err(Error::Unsupported(u)) => {
                    assert_eq!((u.feature, u.needs), needs_export(), "{what}")
                }
                other => panic!("{what}: {:?}", other.map(|_| ())),
            }
        }
    }
}
