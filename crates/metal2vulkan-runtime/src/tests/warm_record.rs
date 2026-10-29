//! Pipeline records (`warm/record.rs`, `warm/codec.rs`): each kind's descriptor, every field away
//! from its default, reads back from its record's bytes equal; a record carried as bytes is the
//! same record; one library named twice is named once; and bytes of another record format, or
//! damaged, cut or foreign bytes, are no record. CPU only: a record compiles nothing. The
//! functions are those of the owned fixtures `validation/fixtures/public/engine/ob3_*.air`.

use crate::warm::codec::{self, Pipeline, MAGIC, RECORD_FORMAT};
use crate::PipelineRecord;
use metal2vulkan::library::{Function, FunctionConstantValue, FunctionConstantValues, Library};
use metal2vulkan::metal::{self, DataType, PixelFormat};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, FunctionDescriptor, LinkedFunction,
    LinkedFunctions, MeshPipelineDescriptor, RenderPipelineDescriptor,
    StageInputAttributeDescriptor, StageInputDescriptor, StageInputLayoutDescriptor, StageSource,
    TessellationDescriptor, TileRenderPipelineDescriptor, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

/// The library of the committed AIR of fixture `stem`.
fn library(stem: &str) -> Library {
    let air = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine")
        .join(format!("{stem}.air"));
    Library::from_air(&std::fs::read(air).expect("the committed AIR")).expect("a library")
}

/// The function `name` of fixture `stem`.
fn function(stem: &str, name: &str) -> Function {
    library(stem).function(name).expect("the function")
}

/// Constant values by index and by name, of two types.
fn constants() -> FunctionConstantValues {
    FunctionConstantValues::defaults()
        .with_index(
            3,
            FunctionConstantValue::new(DataType::UInt, 7u32.to_le_bytes()),
        )
        .with_name(
            "scale",
            FunctionConstantValue::new(DataType::Float, 1.5f32.to_le_bytes()),
        )
}

/// A stage of `f` with constants and linked functions in every list and a group.
fn stage(f: Function) -> StageSource {
    let linked = function("ob3_texture_array", "ob3_texture_array");
    let mut functions = LinkedFunctions::new()
        .with_function(LinkedFunction::new(linked.clone()).with_constants(constants()))
        .with_private_function(linked.clone());
    functions
        .binary_functions
        .push(LinkedFunction::new(linked.clone()).with_precompiled(true));
    functions
        .groups
        .insert("group".into(), vec![LinkedFunction::new(linked)]);
    let mut descriptor = FunctionDescriptor::new(f);
    descriptor.constants = constants();
    descriptor.linked = functions;
    StageSource::Function(descriptor)
}

/// Two buffers' mutability.
fn mutability() -> BTreeMap<u32, metal::Mutability> {
    BTreeMap::from([
        (0, metal::Mutability::Immutable),
        (5, metal::Mutability::Mutable),
    ])
}

/// A color attachment with every field away from its default.
fn color(format: PixelFormat) -> ColorAttachmentDescriptor {
    let mut c = ColorAttachmentDescriptor::new(format);
    c.blending_enabled = true;
    c.source_rgb_blend_factor = metal::BlendFactor::SourceAlpha;
    c.destination_rgb_blend_factor = metal::BlendFactor::OneMinusSourceAlpha;
    c.rgb_blend_operation = metal::BlendOperation::Max;
    c.source_alpha_blend_factor = metal::BlendFactor::One;
    c.destination_alpha_blend_factor = metal::BlendFactor::SourceAlpha;
    c.alpha_blend_operation = metal::BlendOperation::ReverseSubtract;
    c.write_mask = metal::ColorWriteMask::RED | metal::ColorWriteMask::ALPHA;
    c
}

fn compute() -> Pipeline {
    let mut d =
        ComputePipelineDescriptor::new(stage(function("ob3_texture_array", "ob3_texture_array")));
    let mut input = StageInputDescriptor::new();
    let mut layout = StageInputLayoutDescriptor::new(Some(16));
    layout.step_function = metal::StepFunction::PerInstance;
    layout.step_rate = 2;
    input.layouts.insert(1, layout);
    input.attributes.insert(
        3,
        StageInputAttributeDescriptor::new(metal::AttributeFormat::UChar4, 4, 1),
    );
    input.index_type = metal::IndexType::UInt32;
    input.index_buffer_index = 9;
    d.stage_input = Some(input);
    d.max_total_threads_per_threadgroup = Some(256);
    d.threadgroup_size_is_multiple_of_execution_width = true;
    d.support_indirect_command_buffers = true;
    d.max_call_stack_depth = 3;
    d.required_threads_per_threadgroup = Some([8, 4, 2]);
    d.support_adding_binary_functions = true;
    d.buffers = mutability();
    d.needs_custom_border_color_samplers = true;
    Pipeline::Compute(d, 2)
}

fn render() -> Pipeline {
    let vertex = function("ob3_vertex_input", "ob3_vertex_input");
    let mut d = RenderPipelineDescriptor::new(stage(vertex.clone()));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(vertex)));
    let mut v = VertexDescriptor::new();
    let mut layout = VertexBufferLayoutDescriptor::new(None);
    layout.step_function = metal::VertexStepFunction::PerInstance;
    layout.step_rate = 3;
    v.layouts.insert(2, layout);
    v.attributes.insert(
        1,
        VertexAttributeDescriptor::new(metal::VertexFormat::UChar4, 8, 2),
    );
    d.vertex_descriptor = Some(v);
    d.color_attachments[0] = Some(color(PixelFormat::RGBA8Unorm));
    d.color_attachments[5] = Some(color(PixelFormat::RGBA16Float));
    d.depth_format = Some(PixelFormat::Depth32Float);
    d.stencil_format = Some(PixelFormat::Stencil8);
    d.raster_sample_count = 4;
    d.alpha_to_coverage = true;
    d.alpha_to_one = true;
    d.rasterization_enabled = false;
    d.input_primitive_topology = metal::PrimitiveTopologyClass::Line;
    d.max_vertex_amplification_count = 2;
    let mut t = TessellationDescriptor::default();
    t.max_tessellation_factor = 32;
    t.partition_mode = metal::TessellationPartitionMode::FractionalOdd;
    t.factor_step_function = metal::TessellationFactorStepFunction::PerPatchAndPerInstance;
    t.output_winding_order = metal::Winding::CounterClockwise;
    t.control_point_index_type = metal::TessellationControlPointIndexType::UInt16;
    t.factor_scale_enabled = true;
    d.tessellation = Some(t);
    d.support_indirect_command_buffers = true;
    d.max_vertex_call_stack_depth = 2;
    d.max_fragment_call_stack_depth = 5;
    d.support_adding_vertex_binary_functions = true;
    d.support_adding_fragment_binary_functions = true;
    d.vertex_buffers = mutability();
    d.fragment_buffers = BTreeMap::from([(1, metal::Mutability::Immutable)]);
    d.texture_write_rounding_mode = metal::TextureWriteRoundingMode::NearestEven;
    let p = &mut d.private;
    p.sample_mask = 0x0f;
    p.sample_coverage = 0.5;
    p.sample_coverage_invert = true;
    p.alpha_test_enabled = true;
    p.alpha_test_function = 3;
    p.logic_operation_enabled = true;
    p.logic_operation = 10;
    p.clip_distance_enable_mask = 0x3;
    p.point_smooth_enabled = true;
    p.point_coord_lower_left = true;
    p.point_size_output_vs = true;
    p.two_side_enabled = true;
    p.vertex_depth_compare_clamp_mask = 0x10;
    p.fragment_depth_compare_clamp_mask = 0x8000_0001;
    p.depth_stencil_write_disabled = true;
    p.needs_custom_border_color_samplers = true;
    p.vertex_amplification_mode = 1;
    p.explicit_visibility_group_id = 4;
    Pipeline::Render(d, [1, 3])
}

fn mesh() -> Pipeline {
    let f = function("ob3_vertex_input", "ob3_vertex_input");
    let mut d = MeshPipelineDescriptor::new(stage(f.clone()));
    d.object = Some(StageSource::Function(FunctionDescriptor::new(f.clone())));
    d.fragment = Some(stage(f));
    d.color_attachments[2] = Some(color(PixelFormat::BGRA8Unorm));
    d.depth_format = Some(PixelFormat::Depth32Float);
    d.stencil_format = Some(PixelFormat::Stencil8);
    d.raster_sample_count = 2;
    d.rasterization_enabled = false;
    d.alpha_to_coverage = true;
    d.alpha_to_one = true;
    d.support_indirect_command_buffers = true;
    d.texture_write_rounding_mode = metal::TextureWriteRoundingMode::NearestEven;
    Pipeline::Mesh(d)
}

fn tile() -> Pipeline {
    let mut d = TileRenderPipelineDescriptor::new(stage(function(
        "ob3_texture_array",
        "ob3_texture_array",
    )));
    d.color_attachment_formats[0] = Some(PixelFormat::RGBA8Unorm);
    d.color_attachment_formats[7] = Some(PixelFormat::R32Float);
    d.raster_sample_count = 4;
    d.threadgroup_size_matches_tile_size = true;
    d.max_total_threads_per_threadgroup = Some(64);
    d.max_call_stack_depth = 4;
    d.required_threads_per_threadgroup = Some([16, 16, 1]);
    d.support_adding_binary_functions = true;
    d.buffers = mutability();
    Pipeline::Tile(d, 1)
}

/// Every kind, every field away from its default, reads back from its record's bytes equal to
/// what was recorded; the same record carried as bytes is the same record, and its bytes are its
/// own; and the defaults round-trip too.
#[test]
fn every_kind_reads_back_field_for_field() {
    let defaults = [
        Pipeline::Compute(
            ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
                function("ob3_texture_array", "ob3_texture_array"),
            ))),
            0,
        ),
        Pipeline::Render(
            RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
                function("ob3_vertex_input", "ob3_vertex_input"),
            ))),
            [0, 0],
        ),
    ];
    for pipeline in [compute(), render(), mesh(), tile()]
        .into_iter()
        .chain(defaults)
    {
        let record = PipelineRecord::of(&pipeline).expect("a record");
        let bytes = record.to_bytes();
        assert_eq!(
            codec::decode(&bytes, record.libraries()).expect("it reads back"),
            pipeline
        );
        let carried = PipelineRecord::from_bytes(&bytes).expect("the bytes are a record");
        assert_eq!(
            carried, record,
            "a record carried as bytes is the same record"
        );
        assert_eq!(carried.to_bytes(), bytes);
        assert!(
            carried.libraries().is_empty(),
            "its libraries stay in the directory"
        );
    }
}

/// Equal creations make equal bytes, and a creation that differs in one field makes others.
#[test]
fn a_record_is_its_creation_and_nothing_else() {
    let a = PipelineRecord::of(&render()).unwrap();
    let b = PipelineRecord::of(&render()).unwrap();
    assert_eq!(a, b);
    let Pipeline::Render(mut d, added) = render() else {
        unreachable!()
    };
    d.private.sample_mask = 0x1f;
    assert_ne!(a, PipelineRecord::of(&Pipeline::Render(d, added)).unwrap());
}

/// Functions of one library, in every stage and linked list, name that library once; a function
/// of another library adds it.
#[test]
fn one_library_is_named_once() {
    let record = PipelineRecord::of(&mesh()).unwrap();
    // The mesh, object and fragment functions are of one library, the linked functions of
    // another.
    assert_eq!(record.libraries().len(), 2);
    let one = library("ob3_vertex_input");
    let vertex = one.function("ob3_vertex_input").unwrap();
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        vertex.clone(),
    )));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(vertex)));
    let record = PipelineRecord::of(&Pipeline::Render(d, [0, 0])).unwrap();
    assert_eq!(record.libraries(), std::slice::from_ref(&one));
}

/// `bytes` with their checksum made again: a record whose checksum holds over whatever they say.
fn resealed(mut bytes: Vec<u8>) -> Vec<u8> {
    let body = bytes.len() - 32;
    let checksum: [u8; 32] = Sha256::digest(&bytes[..body]).into();
    bytes[body..].copy_from_slice(&checksum);
    bytes
}

/// A record of another format is dropped as it is read, though its checksum holds; so are
/// damaged, cut, empty and foreign bytes. Bytes whose libraries are not the record's read nothing.
#[test]
fn a_record_of_another_format_or_damaged_is_dropped() {
    let record = PipelineRecord::of(&compute()).unwrap();
    let bytes = record.to_bytes();
    assert_eq!(&bytes[..8], MAGIC);
    assert!(PipelineRecord::from_bytes(&bytes).is_some());
    let mut other = bytes.clone();
    other[8..12].copy_from_slice(&(RECORD_FORMAT + 1).to_le_bytes());
    assert!(
        PipelineRecord::from_bytes(&resealed(other)).is_none(),
        "another format"
    );
    let mut damaged = bytes.clone();
    damaged[20] ^= 1;
    assert!(PipelineRecord::from_bytes(&damaged).is_none(), "damaged");
    assert!(
        PipelineRecord::from_bytes(&bytes[..bytes.len() - 1]).is_none(),
        "cut"
    );
    assert!(PipelineRecord::from_bytes(&[]).is_none(), "empty");
    assert!(PipelineRecord::from_bytes(b"not a record at all, nor near one").is_none());
    let mut magic = bytes.clone();
    magic[0] = b'X';
    assert!(
        PipelineRecord::from_bytes(&resealed(magic)).is_none(),
        "another magic"
    );
    // A body the record's checksum holds over but that reads past or short of its fields.
    let mut longer = bytes[..bytes.len() - 32].to_vec();
    longer.push(0);
    longer.extend_from_slice(&[0; 32]);
    assert!(codec::decode(&resealed(longer), record.libraries()).is_err());
    // Other libraries than the ones it names, or its own in another order.
    assert!(codec::decode(&bytes, &[]).is_err());
    let two = PipelineRecord::of(&mesh()).unwrap();
    let swapped: Vec<Library> = two.libraries().iter().rev().cloned().collect();
    assert_eq!(swapped.len(), 2);
    assert!(codec::decode(&two.to_bytes(), two.libraries()).is_ok());
    assert!(codec::decode(&two.to_bytes(), &swapped).is_err());
}
