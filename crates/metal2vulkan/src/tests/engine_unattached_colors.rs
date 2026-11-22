//! A render pipeline's fragment function keeps no output at a color it writes that the pipeline
//! has no attachment for, whose write Metal discards. No device is opened; nothing runs on a GPU.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use std::path::Path;

use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::library::Library;
use crate::metal::{PixelFormat, VertexFormat};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
    VertexDescriptor,
};
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::{FunctionDescriptor, RenderPipelineDescriptor, StageSource};

fn fixture(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
    }
}

/// Every committed profile as a device, by stem.
fn devices() -> Vec<(String, Device)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dir).expect("profiles") {
        let path = e.expect("entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let text = std::fs::read_to_string(&path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        out.push((
            stem,
            Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device"),
        ));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn device(stem: &str) -> Device {
    devices()
        .into_iter()
        .find(|(s, _)| s == stem)
        .map(|(_, d)| d)
        .expect("committed profile")
}

// ---- Render pipelines --------------------------------------------------------------------------

/// `stage_io_vertex`'s attributes: one interleaved per-vertex buffer at index 0.
fn vertex_descriptor() -> VertexDescriptor {
    let mut d = VertexDescriptor::new();
    d.layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(24)));
    for (index, format, offset) in [
        (0, VertexFormat::Float3, 0),
        (1, VertexFormat::Half2, 12),
        (4, VertexFormat::UShort2, 16),
        (7, VertexFormat::UChar4, 20),
    ] {
        d.attributes
            .insert(index, VertexAttributeDescriptor::new(format, offset, 0));
    }
    d
}

/// `stage_io`'s vertex + fragment pipeline descriptor.
fn render_descriptor(bc: &[u8]) -> RenderPipelineDescriptor {
    let lib = Library::from_air(bc).expect("library");
    let v = lib.function("stage_io_vertex").expect("vertex");
    let f = lib.function("stage_io_fragment").expect("fragment");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(v)));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(f)));
    d.vertex_descriptor = Some(vertex_descriptor());
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    d.color_attachments[2] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    d.depth_format = Some(PixelFormat::Depth32Float);
    d
}

/// `Device::compile_render_pipeline` compiles a `Function` fragment stage with no output at a color
/// it writes that the pipeline has no attachment for, whose write Metal discards: the fixture's
/// fragment, writing colors 0 and 2, states color 0 alone among its color outputs in a pipeline of
/// attachment 0 alone, and both where it has both. Under alpha to coverage its color 0 stays
/// without attachment 0, the coverage reading its alpha.
#[test]
fn a_render_pipeline_writes_no_color_it_has_no_attachment_for() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    for stem in ["lavapipe", "moltenvk-m3", "amd-780m-radv"] {
        let dev = device(stem);
        let outputs = |edit: &dyn Fn(&mut RenderPipelineDescriptor)| {
            let mut d = render_descriptor(&bc);
            edit(&mut d);
            dev.compile_render_pipeline(&d, &Budget::standard())
                .unwrap_or_else(|e| panic!("{stem}: {e}"))
                .graphics
                .color_outputs
        };
        assert_eq!(outputs(&|_| {}), 0b101, "{stem}");
        assert_eq!(outputs(&|d| d.color_attachments[2] = None), 0b1, "{stem}");
        let covered = |d: &mut RenderPipelineDescriptor| {
            d.color_attachments[0] = None;
            d.alpha_to_coverage = true;
        };
        assert_eq!(outputs(&covered), 0b101, "{stem}");
    }
}

// ---- Textures and samplers ---------------------------------------------------------------------
