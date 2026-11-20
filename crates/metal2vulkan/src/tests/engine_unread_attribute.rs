//! A vertex attribute whose value reaches only a varying a function constant gates off is not read
//! by the specialized vertex function: render pipelines of the owned fixture
//! `engine/fca_vertex_unread_attribute` (`fca_vertex_unread_attribute`, whose `color` varying
//! constant 0 gates and which reads attribute 1 only into it, and `fca_fragment_unread_attribute`)
//! on committed profiles.
//!
//! Metal's answer, on an Apple M3 Max (macOS 27.0.1, validation layer off and on): with the
//! constant false the specialized function's attribute 1 is inactive and a vertex descriptor of
//! attribute 0 alone makes the pipeline; with it true attribute 1 is active, that descriptor is
//! refused ("Vertex attribute color(1) is missing from the vertex descriptor"), and one of both
//! attributes makes the pipeline. The translation reads the same: the gated-off varying's field of
//! the returned struct is inserted `undef` (`air::canon`'s `unobserved_code`), so the attribute's
//! read goes and the pipeline's vertex input names attribute 0 alone; gated on, the descriptor
//! without attribute 1 is `InvalidDescriptor(VertexAttribute { attribute: 1 })`.
//!
//! `llvm-as` assembles the fixture; without it the test skips visibly. No GPU is used.

use crate::budget::Budget;
use crate::device::{Device, DevicePolicy};
use crate::error::{DescriptorError, Error};
use crate::library::{FunctionConstantValue, FunctionConstantValues, Library};
use crate::metal::{DataType, PixelFormat, VertexFormat};
use crate::pipeline::artifact::RenderPipeline;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};
use crate::pipeline::testing::{description, engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::{FunctionDescriptor, StageSource};
use crate::spirv::testing::check_words_valid;

fn fixture() -> Option<Vec<u8>> {
    let path = engine_dir().join("fca_vertex_unread_attribute.ll");
    match llvm_assemble(&path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {}: {why}", path.display()),
    }
}

/// The fixture's pipeline on `dev` with constant 0 `has_color` and a vertex descriptor of
/// attributes `0..attributes` (`Float4` each, 16 bytes apart in buffer 0's 32-byte records).
fn compile(
    dev: &Device,
    library: &Library,
    has_color: bool,
    attributes: u32,
) -> Result<RenderPipeline, Error> {
    let constants = FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(DataType::Bool, [u8::from(has_color)]),
    );
    let function = |name| {
        let mut f = FunctionDescriptor::new(library.function(name).expect("entry"));
        f.constants = constants.clone();
        StageSource::Function(f)
    };
    let mut d = RenderPipelineDescriptor::new(function("fca_vertex_unread_attribute"));
    d.fragment = Some(function("fca_fragment_unread_attribute"));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let mut vertex = VertexDescriptor::new();
    for a in 0..attributes {
        vertex.attributes.insert(
            a,
            VertexAttributeDescriptor::new(VertexFormat::Float4, 16 * a, 0),
        );
    }
    vertex
        .layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(32)));
    d.vertex_descriptor = Some(vertex);
    dev.compile_render_pipeline(&d, &Budget::standard())
}

/// The attributes `p`'s vertex input fetches, in fixed function or pulled, and every stage's
/// module through L3.
fn fetched(p: &RenderPipeline, dev: &Device, what: &str) -> usize {
    for stage in &p.stages {
        check_words_valid(&stage.spirv, &p.requirements, dev.profile())
            .unwrap_or_else(|e| panic!("{what}, {:?}: L3: {e}", stage.stage));
    }
    p.vertex_input.attributes.len() + p.vertex_input.pulled.len()
}

/// Gated off, the varying's attribute needs no descriptor entry and the vertex input fetches
/// attribute 0 alone; gated on, a descriptor without it is refused as Metal refuses it, and one
/// with it fetches both (the module docs).
#[test]
fn an_attribute_read_only_into_a_gated_off_varying_needs_no_descriptor_entry() {
    let Some(bc) = fixture() else { return };
    let library = Library::from_air(&bc).expect("library");
    for stem in [
        "moltenvk-m3",
        "lavapipe",
        "amd-780m-radv",
        "intel-irisxe-anv",
        "nvidia-rtx5080-laptop",
        "runtime-floor",
    ] {
        let dev = Device::from_queries(
            description(stem).expect("committed description"),
            DevicePolicy::default(),
        )
        .expect("device");
        let off = compile(&dev, &library, false, 1)
            .unwrap_or_else(|e| panic!("{stem}: gated off, attribute 0 alone: {e}"));
        assert_eq!(fetched(&off, &dev, stem), 1, "{stem}: gated off");
        match compile(&dev, &library, true, 1) {
            Err(Error::InvalidDescriptor(DescriptorError::VertexAttribute { attribute: 1 })) => {}
            other => panic!(
                "{stem}: gated on, attribute 0 alone: {:?}",
                other.map(|_| ())
            ),
        }
        let on = compile(&dev, &library, true, 2)
            .unwrap_or_else(|e| panic!("{stem}: gated on, both attributes: {e}"));
        assert_eq!(fetched(&on, &dev, stem), 2, "{stem}: gated on");
    }
}
