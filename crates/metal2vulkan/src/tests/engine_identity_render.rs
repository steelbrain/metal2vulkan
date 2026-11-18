//! Render pipelines under the identity domains (`PointerDomain::Identity` with
//! `HandleDomain::Identity`), over the owned fixture `examples/runtime_examples` (`colored_vertex`,
//! which reads positions and colours from buffers 0 and 1 by vertex ID, and `colored_fragment`,
//! which returns its flat colour):
//!
//! - On `Device::from_queries`' profile the pipeline compiles: the vertex function reaches each of
//!   its buffer slots through its own device address, pushed (`SlotGuestBase`) in its window, with
//!   no descriptor, no root table and no parameter block; the fragment function pushes nothing;
//!   both modules pass L3.
//! - On `Device::new`'s profile, the same pipeline is refused as `Unsupported{IdentityDomains}`
//!   (not implemented), as before: its push window is not the floor's.
//!
//! `llvm-as` assembles the fixture; a missing tool skips visibly. No GPU is used.

use crate::budget::Budget;
use crate::device::{Device, DevicePolicy, HandleDomain, PointerDomain};
use crate::error::{Error, Feature, Needs};
use crate::library::Library;
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{PushField, SlotClass, SlotRef, Stage};
use crate::pipeline::testing::{description, engine_dir, llvm_assemble, AssembleError};
use crate::spirv::testing::check_words_valid;

/// Profiles with `bufferDeviceAddress` and `shaderInt64`.
const PROFILES: [&str; 2] = ["lavapipe", "moltenvk-m3"];

fn identity() -> DevicePolicy {
    DevicePolicy {
        pointer_domain: PointerDomain::Identity,
        handle_domain: HandleDomain::Identity,
        ..DevicePolicy::default()
    }
}

/// The fixture's library, or `None` (printed) where `llvm-as` is missing.
fn library() -> Option<Library> {
    let ll = engine_dir().join("../examples/runtime_examples.ll");
    match llvm_assemble(&ll) {
        Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(e) => panic!("{}: {e:?}", ll.display()),
    }
}

/// `colored_vertex` and `colored_fragment` into one `BGRA8Unorm` attachment.
fn descriptor(library: &Library) -> RenderPipelineDescriptor {
    let function = |name| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("the function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(function("colored_vertex"));
    d.fragment = Some(function("colored_fragment"));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
    d
}

#[test]
fn from_queries_lowers_a_vertex_function_over_its_pushed_buffer_addresses() {
    let Some(library) = library() else {
        return;
    };
    let d = descriptor(&library);
    for stem in PROFILES {
        let desc = description(stem).unwrap_or_else(|e| panic!("{e}"));
        let device = Device::from_queries(desc, identity()).unwrap_or_else(|e| panic!("{e}"));
        let p = (device.compile_render_pipeline(&d, &Budget::standard()))
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let stages: Vec<Stage> = p.interface.push.iter().map(|s| s.stage).collect();
        assert_eq!(stages, [Stage::Vertex, Stage::Fragment], "{stem}");
        let vertex = &p.interface.push[0];
        let pushed: Vec<(PushField, bool)> = (vertex.fields.iter())
            .map(|f| (f.field, matches!(f.loc, PushLoc::Push { .. })))
            .collect();
        let slot = |index| SlotRef {
            stage: Stage::Vertex,
            class: SlotClass::Buffer,
            index,
        };
        assert_eq!(
            pushed,
            [
                (PushField::SlotGuestBase(slot(0)), true),
                (PushField::SlotGuestBase(slot(1)), true)
            ],
            "{stem}: the vertex function's buffer addresses, pushed"
        );
        assert_eq!(vertex.argument_table_size, 0, "{stem}: no parameter block");
        assert!(
            p.interface.push[1].fields.is_empty(),
            "{stem}: the fragment function pushes nothing"
        );
        assert!(
            !p.interface.needs_pointer_translation && !p.interface.needs_handle_translation,
            "{stem}: no root or handle table"
        );
        assert!(
            p.layout.sets.iter().all(|s| s.bindings.is_empty()),
            "{stem}: no descriptor"
        );
        for stage in &p.stages {
            check_words_valid(&stage.spirv, &p.requirements, device.profile())
                .unwrap_or_else(|e| panic!("{stem}: {:?}: {e}", stage.stage));
        }
    }
}

#[test]
fn device_new_still_refuses_render_functions_under_the_identity_domains() {
    let Some(library) = library() else {
        return;
    };
    let d = descriptor(&library);
    for stem in PROFILES {
        let desc = description(stem).unwrap_or_else(|e| panic!("{e}"));
        let device = Device::new(desc, identity()).unwrap_or_else(|e| panic!("{e}"));
        match device.compile_render_pipeline(&d, &Budget::standard()) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    (u.feature, u.needs),
                    (Feature::IdentityDomains, Needs::NotImplemented),
                    "{stem}"
                );
            }
            other => panic!("{stem}: {:?}", other.map(|_| "compiled")),
        }
    }
}
