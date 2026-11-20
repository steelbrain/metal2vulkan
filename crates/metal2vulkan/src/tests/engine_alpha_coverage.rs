//! The device's alpha to coverage, as render pipelines of the owned fixture `engine/a8r_targets`
//! (`a8r_vertex` and `a8r_fill`, a `float4` output to an `RGBA16Float` attachment 0) at one and
//! four samples, alpha to coverage off and on, on committed profiles.
//!
//! Vulkan specifies no algorithm for converting the alpha to a coverage mask, where Metal's
//! coverage is its own (a 2x2 ordered dither at two and four samples, measured on an Apple M3
//! Max), so a pipeline that covers by alpha
//! (`GraphicsState::alpha_to_coverage`) records `PrecisionNote::DeviceAlphaCoverage` on its
//! fragment stage on every profile, and the vertex stage records none. No pipeline requires
//! `VkProperty::MetalAlphaCoverage`. A pipeline without alpha to coverage records no note.

use crate::budget::Budget;
use crate::device::requirements::VkProperty;
use crate::device::{Device, DevicePolicy};
use crate::library::Library;
use crate::metal::PixelFormat;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::descriptor::{ColorAttachmentDescriptor, RenderPipelineDescriptor};
use crate::pipeline::testing::{description, engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::{FunctionDescriptor, StageSource};
use crate::vk;

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("a8r_targets.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected a8r_targets.ll: {why}"),
    }
}

/// On each profile a pipeline that covers by alpha records `DeviceAlphaCoverage` on its fragment
/// stage alone, and requires no `MetalAlphaCoverage`; without alpha to coverage it records no note
/// (the module docs).
#[test]
fn a_pipeline_covering_by_alpha_notes_the_device_s_coverage() {
    let Some(bc) = fixture() else { return };
    let library = Library::from_air(&bc).expect("library");
    let function = |name| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("entry"),
        ))
    };
    for stem in [
        "moltenvk-m3",
        "lavapipe",
        "amd-780m-radv",
        "intel-irisxe-anv",
        "nvidia-rtx5080-laptop",
    ] {
        let dev = Device::from_queries(
            description(stem).expect("committed description"),
            DevicePolicy::default(),
        )
        .expect("device");
        for samples in [1, 4] {
            for coverage in [false, true] {
                let what = format!("{stem} {samples} samples, alpha to coverage {coverage}");
                let mut d = RenderPipelineDescriptor::new(function("a8r_vertex"));
                d.fragment = Some(function("a8r_fill"));
                d.color_attachments[0] =
                    Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
                d.raster_sample_count = samples;
                d.alpha_to_coverage = coverage;
                let pipeline = dev
                    .compile_render_pipeline(&d, &Budget::standard())
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(pipeline.graphics.alpha_to_coverage, coverage, "{what}");
                for stage in &pipeline.stages {
                    let fragment = stage.stage == vk::ShaderStage::FRAGMENT;
                    assert_eq!(
                        stage
                            .precision_notes
                            .contains(&PrecisionNote::DeviceAlphaCoverage),
                        fragment && coverage,
                        "{what}, {:?}: {:?}",
                        stage.stage,
                        stage.precision_notes
                    );
                }
                assert!(
                    !pipeline
                        .requirements
                        .properties
                        .contains(&VkProperty::MetalAlphaCoverage),
                    "{what}: {:?}",
                    pipeline.requirements.properties
                );
            }
        }
    }
}
