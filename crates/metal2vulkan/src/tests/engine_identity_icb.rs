//! Indirect command buffers under the identity domains: GPU-encoded indexed triangle draws.
//!
//! - The owned fixture `engine/icb_merged_indexed.ll`'s kernels, which bind three vertex buffers
//!   and a fragment buffer of each command and draw indexed triangles of 16-bit and of 32-bit
//!   indices, and `engine/icb_indexed_extras.ll`'s `icb_extras_reset`, which resets render
//!   commands, compile and
//!   validate on every profile with device addresses.
//! - A kernel's plain draw (`engine/icb_merged_draws.ll`'s `icb_merged_triangles`) is refused,
//!   `Unsupported{GpuEncodedIndirectCommands, NotImplemented}`.
//! - The vertex function of a state made with `supportIndirectCommandBuffers` pushes the execution
//!   range's address and its command (`PushField::IndirectRange`, `PushField::IndirectSequence`,
//!   `PushField::IndirectBase`)
//!   in its window; the same state made without it pushes neither. Both validate.
//!
//! `llvm-as` assembles the fixtures; without it a test skips visibly.

use crate::budget::Budget;
use crate::device::{Device, DevicePolicy, HandleDomain, PointerDomain};
use crate::error::{Error, Feature, Needs};
use crate::library::Library;
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::PushField;
use crate::pipeline::testing::{
    compile, description, engine_dir, link_and_validate, llvm_assemble, profile, AssembleError,
};
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

/// The bitcode of the engine fixture `name`, or `None` (printed) where `llvm-as` is missing.
fn assemble(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(bitcode) => Some(bitcode),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(e) => panic!("{name}: {e:?}"),
    }
}

/// Kernels that bind each command's buffers, draw indexed triangles of either index size, or
/// reset commands compile and validate under the identity domains on every profile.
#[test]
fn kernels_encode_indexed_triangle_draws_and_resets_under_the_identity_domains() {
    let kernels = [
        ("icb_merged_indexed", "icb_indexed_16"),
        ("icb_merged_indexed", "icb_indexed_32"),
        ("icb_indexed_extras", "icb_extras_reset"),
    ];
    for (fixture, entry) in kernels {
        let Some(bc) = assemble(fixture) else {
            return;
        };
        for stem in PROFILES {
            let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{stem}: {entry}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {entry}: {e}"));
        }
    }
}

/// A kernel's plain draw into an indirect command buffer is not built: the first cut draws
/// indexed triangles alone.
#[test]
fn a_kernels_plain_draw_into_commands_is_refused_under_the_identity_domains() {
    let Some(bc) = assemble("icb_merged_draws") else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, identity()).unwrap_or_else(|e| panic!("{e}"));
        match compile(&bc, "icb_merged_triangles", &p) {
            Err(Error::Unsupported(u)) => assert_eq!(
                (u.feature, u.needs),
                (Feature::GpuEncodedIndirectCommands, Needs::NotImplemented),
                "{stem}"
            ),
            other => panic!("{stem}: {other:?}"),
        }
    }
}

/// The vertex function of a state for indirect command buffers pushes the execution range's
/// address and its command; the same state without the flag pushes neither.
#[test]
fn a_state_for_indirect_command_buffers_pushes_its_execution_range() {
    let Some(bc) = assemble("icb_merged_indexed") else {
        return;
    };
    let library = Library::from_air(&bc).expect("the library");
    let function = |name| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("the function"),
        ))
    };
    for stem in PROFILES {
        let desc = description(stem).unwrap_or_else(|e| panic!("{e}"));
        let device = Device::from_queries(desc, identity()).unwrap_or_else(|e| panic!("{e}"));
        for flagged in [false, true] {
            let mut d = RenderPipelineDescriptor::new(function("icb_indexed_vertex"));
            d.fragment = Some(function("icb_indexed_fragment"));
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
            d.support_indirect_command_buffers = flagged;
            let p = (device.compile_render_pipeline(&d, &Budget::standard()))
                .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
            let vertex = &p.interface.push[0];
            let pushed = |field| {
                (vertex.fields.iter())
                    .any(|f| f.field == field && matches!(f.loc, PushLoc::Push { .. }))
            };
            assert_eq!(
                (
                    pushed(PushField::IndirectRange),
                    pushed(PushField::IndirectSequence),
                    pushed(PushField::IndirectBase)
                ),
                (flagged, flagged, flagged),
                "{stem}: flagged {flagged}"
            );
            for stage in &p.stages {
                check_words_valid(&stage.spirv, &p.requirements, device.profile()).unwrap_or_else(
                    |e| panic!("{stem}: flagged {flagged}: {:?}: {e}", stage.stage),
                );
            }
        }
    }
}
