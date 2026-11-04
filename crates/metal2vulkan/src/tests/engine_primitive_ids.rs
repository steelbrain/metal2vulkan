//! `[[primitive_id]]` on a device that cannot declare a fragment `PrimitiveId` (neither
//! `geometryShader` nor `tessellationShader`): the fragment function reads a flat varying
//! (`IoRole::PrimitiveId`) that its pipeline's vertex function writes from its vertex index, base
//! vertex and `PushField::VerticesPerPrimitive` (`LoweringInputs::primitive_id_varying`). The
//! owned fixture `engine/primitive_ids` on `lavapipe` with both features withheld, on
//! `vk12-floor`, and on `lavapipe` as it is, where the fragment stage declares the builtin.
//! `llvm-as` assembles the fixture; a missing tool skips visibly.

use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DevicePolicy};
use crate::error::{Error, RuntimeInput};
use crate::library::{CompiledFunction, Library};
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::stage_io::{IoRole, StageBuiltin, StageIo};
use crate::pipeline::sym::PushField;
use crate::pipeline::testing::{description, engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::{LoweringInputs, RenderPipeline, StageModule};
use crate::spirv::testing::check_words_valid;
use crate::vk;

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("primitive_ids.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected primitive_ids.ll: {why}"),
    }
}

/// `lavapipe` without `geometryShader` and `tessellationShader` (and the features that need
/// either), drawing `colors` color attachments at once.
fn withheld(colors: u32) -> Device {
    withholding(colors, &[])
}

/// `withheld(colors)` without `more` either.
fn withholding(colors: u32, more: &[VkFeature]) -> Device {
    let mut desc = description("lavapipe").expect("lavapipe");
    for f in [
        VkFeature::GeometryShader,
        VkFeature::TessellationShader,
        VkFeature::ShaderTessellationAndGeometryPointSize,
        VkFeature::MultiviewGeometryShader,
        VkFeature::MultiviewTessellationShader,
    ]
    .iter()
    .chain(more)
    {
        desc.features.remove(f);
    }
    desc.limits.max_color_attachments = colors;
    desc.limits.max_fragment_output_attachments = colors;
    Device::new(desc, DevicePolicy::default()).expect("device")
}

fn device(stem: &str) -> Device {
    Device::new(description(stem).expect("profile"), DevicePolicy::default()).expect("device")
}

fn function(bc: &[u8], name: &str) -> FunctionDescriptor {
    FunctionDescriptor::new(
        Library::from_air(bc)
            .expect("library")
            .function(name)
            .expect("function"),
    )
}

/// The pipeline of vertex function `vertex` and fragment function `fragment` (none for `None`),
/// with `colors` `RGBA32Uint` attachments.
fn pipeline(
    bc: &[u8],
    vertex: &str,
    fragment: Option<&str>,
    colors: usize,
) -> RenderPipelineDescriptor {
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(bc, vertex)));
    d.fragment = fragment.map(|f| StageSource::Function(function(bc, f)));
    for slot in d.color_attachments.iter_mut().take(colors) {
        *slot = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    }
    d
}

fn compile(dev: &Device, d: &RenderPipelineDescriptor) -> Result<RenderPipeline, Error> {
    dev.compile_render_pipeline(d, &Budget::standard())
}

fn stage(p: &RenderPipeline, s: vk::ShaderStage) -> &StageModule {
    p.stages.iter().find(|m| m.stage == s).expect("stage")
}

fn validate(p: &RenderPipeline, dev: &Device) {
    for m in &p.stages {
        check_words_valid(&m.spirv, &p.requirements, dev.profile())
            .unwrap_or_else(|e| panic!("{:?}: {e}", m.stage));
    }
}

fn reads_varying(io: &StageIo) -> bool {
    io.inputs.iter().any(|v| v.role == IoRole::PrimitiveId)
}

fn writes_varying(io: &StageIo) -> bool {
    io.outputs.iter().any(|v| v.role == IoRole::PrimitiveId)
}

/// Whether the pipeline reads push field `field`.
fn reads_field(p: &RenderPipeline, field: PushField) -> bool {
    p.interface
        .push
        .iter()
        .flat_map(|pl| &pl.fields)
        .any(|f| f.field == field)
}

/// Whether the pipeline's vertex stage reads `PushField::VerticesPerPrimitive`.
fn reads_layout(p: &RenderPipeline) -> bool {
    reads_field(p, PushField::VerticesPerPrimitive)
}

fn standalone(dev: &Device, bc: &[u8], name: &str) -> CompiledFunction {
    dev.compile_function(
        &function(bc, name),
        &LoweringInputs::new(),
        &Budget::standard(),
    )
    .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// A fragment function compiled on its own for a device with neither feature reads
/// `[[primitive_id]]` from a flat `uint` varying, declaring no `PrimitiveId`; with them it reads
/// the builtin. `vk12-floor` has neither: the function no longer needs either feature.
#[test]
fn without_either_feature_the_id_is_a_flat_varying() {
    let Some(bc) = fixture() else { return };
    for dev in [withheld(8), device("vk12-floor")] {
        let f = standalone(&dev, &bc, "pi_ids");
        let io = &f.artifact().interface.io;
        let input = io
            .inputs
            .iter()
            .find(|v| v.role == IoRole::PrimitiveId)
            .expect("the varying");
        assert_eq!(input.ty.components, 1);
        assert!(!io.builtins.contains(&StageBuiltin::PrimitiveId));
        let r = &f.artifact().requirements;
        assert!(!r.features.contains(&VkFeature::GeometryShader), "{r:?}");
        assert!(
            !r.features.contains(&VkFeature::TessellationShader),
            "{r:?}"
        );
    }
    let f = standalone(&device("lavapipe"), &bc, "pi_ids");
    let io = &f.artifact().interface.io;
    assert!(io.builtins.contains(&StageBuiltin::PrimitiveId));
    assert!(!reads_varying(io));
}

/// In a render pipeline the vertex function writes the varying its fragment function reads, read
/// flat at one location, from the draw's base vertex and `PushField::VerticesPerPrimitive`; with
/// the features neither stage carries it. Every module is valid.
#[test]
fn the_vertex_function_writes_the_id_its_fragment_function_reads() {
    let Some(bc) = fixture() else { return };
    let dev = withheld(8);
    for vertex in ["pi_place", "pi_points", "pi_attrib", "pi_counts"] {
        let mut d = pipeline(&bc, vertex, Some("pi_ids"), 1);
        if vertex == "pi_attrib" {
            d.vertex_descriptor = Some(attribute_descriptor());
        }
        let p = compile(&dev, &d).unwrap_or_else(|e| panic!("{vertex}: {e}"));
        assert!(reads_layout(&p), "{vertex}");
        assert!(p.graphics.flat_fragment_inputs, "{vertex}");
        validate(&p, &dev);
        let present = device("lavapipe");
        let q = compile(&present, &d).unwrap_or_else(|e| panic!("{vertex}: {e}"));
        assert!(!reads_layout(&q), "{vertex}");
    }
    // The interfaces, through a vertex function compiled for the pipeline.
    let d = pipeline(&bc, "pi_place", Some("pi_ids"), 1);
    let keys = dev
        .render_pipeline_key(&d, &Budget::standard())
        .expect("keys");
    let inputs = keys.vertex_inputs.expect("vertex inputs");
    assert!(inputs.primitive_id_varying);
    let v = dev
        .compile_function(&function(&bc, "pi_place"), &inputs, &Budget::standard())
        .expect("vertex");
    assert!(writes_varying(&v.artifact().interface.io));
    let f = standalone(&dev, &bc, "pi_ids");
    let input = f
        .artifact()
        .interface
        .io
        .inputs
        .iter()
        .find(|x| x.role == IoRole::PrimitiveId)
        .cloned()
        .expect("input");
    assert_eq!(input.key, crate::pipeline::sym::VaryingKey::primitive_id());
    let mut compiled = d.clone();
    compiled.vertex = StageSource::Compiled(v);
    compiled.fragment = Some(StageSource::Compiled(f));
    let p = compile(&dev, &compiled).expect("compiled stages link");
    validate(&p, &dev);
    // On the device with the features the keys carry no such input.
    let keys = device("lavapipe")
        .render_pipeline_key(&d, &Budget::standard())
        .expect("keys");
    assert!(!keys.vertex_inputs.expect("inputs").primitive_id_varying);
}

/// The vertex function reads the draw's first vertex as `[[base_vertex]]` is read: Vulkan's
/// `BaseVertex` with `shaderDrawParameters`, so an indirect draw stays on the GPU, and lane 0 of
/// `PushField::BaseVertexInstance` without, which the binder writes from the draw (an indirect
/// one's arguments read on the CPU).
#[test]
fn the_first_vertex_is_the_draw_s_base_vertex() {
    let Some(bc) = fixture() else { return };
    let d = pipeline(&bc, "pi_place", Some("pi_ids"), 1);
    for (dev, builtin) in [
        (withheld(8), true),
        (withholding(8, &[VkFeature::ShaderDrawParameters]), false),
    ] {
        let p = compile(&dev, &d).expect("pipeline");
        validate(&p, &dev);
        assert!(reads_layout(&p));
        assert_eq!(reads_field(&p, PushField::BaseVertexInstance), !builtin);
        let inputs = dev
            .render_pipeline_key(&d, &Budget::standard())
            .expect("keys")
            .vertex_inputs
            .expect("vertex inputs");
        let v = dev
            .compile_function(&function(&bc, "pi_place"), &inputs, &Budget::standard())
            .expect("vertex");
        let io = &v.artifact().interface.io;
        assert_eq!(io.builtins.contains(&StageBuiltin::BaseVertex), builtin);
        let r = &v.artifact().requirements;
        assert_eq!(
            r.features.contains(&VkFeature::ShaderDrawParameters),
            builtin,
            "{r:?}"
        );
    }
}

fn attribute_descriptor() -> crate::pipeline::descriptor::VertexDescriptor {
    use crate::pipeline::descriptor::{
        VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
    };
    let mut v = VertexDescriptor::new();
    v.attributes.insert(
        0,
        VertexAttributeDescriptor::new(crate::metal::VertexFormat::Float2, 0, 0),
    );
    v.layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(8)));
    v
}

/// A pipeline whose fragment function reads no primitive ID, or that has none, pays nothing: its
/// vertex function writes no varying and reads no layout.
#[test]
fn a_pipeline_that_reads_no_id_writes_none() {
    let Some(bc) = fixture() else { return };
    let dev = withheld(8);
    let p = compile(&dev, &pipeline(&bc, "pi_place", None, 0)).expect("no fragment");
    assert!(!reads_layout(&p));
    let keys = dev
        .render_pipeline_key(&pipeline(&bc, "pi_place", None, 0), &Budget::standard())
        .expect("keys");
    assert!(!keys.vertex_inputs.expect("inputs").primitive_id_varying);
}

/// Link refuses a vertex function compiled without the input behind a fragment function that
/// reads the varying: it writes none (`NeedsInput(PrimitiveId)`), and the fragment function is
/// flat at a location no stage writes otherwise.
#[test]
fn a_vertex_function_compiled_without_the_input_needs_it() {
    let Some(bc) = fixture() else { return };
    let dev = withheld(8);
    let mut d = pipeline(&bc, "pi_place", Some("pi_ids"), 1);
    d.vertex = StageSource::Compiled(standalone(&dev, &bc, "pi_place"));
    match compile(&dev, &d) {
        Err(Error::NeedsInput(RuntimeInput::PrimitiveId)) => {}
        other => panic!("{:?}", other.map(|_| ())),
    }
}

/// The varying is read flat: the fragment module decorates the input `Flat`, so each fragment
/// reads its primitive's provoking vertex, as Metal's `[[primitive_id]]` names the primitive.
#[test]
fn the_fragment_reads_the_id_flat() {
    let Some(bc) = fixture() else { return };
    let dev = withheld(8);
    let f = standalone(&dev, &bc, "pi_ids");
    let io = &f.artifact().interface.io;
    assert!(reads_varying(io));
    let p = compile(&dev, &pipeline(&bc, "pi_place", Some("pi_ids"), 1)).expect("pipeline");
    let words = &stage(&p, vk::ShaderStage::FRAGMENT).spirv;
    const OP_DECORATE: u32 = 71;
    const DECORATION_FLAT: u32 = 14;
    let mut at = 5;
    let mut flat = 0;
    while at < words.len() {
        let len = (words[at] >> 16) as usize;
        if words[at] & 0xffff == OP_DECORATE && words.get(at + 2) == Some(&DECORATION_FLAT) {
            flat += 1;
        }
        at += len.max(1);
    }
    // The vertex and instance IDs the fixture passes flat, and the primitive ID.
    assert_eq!(flat, 3);
}

/// With five colors on a device of four, each color group's variant reads the ID from the same
/// varying of the same vertex module: the groups' vertex words are identical, and the varying
/// takes one location beside the vertex function's own.
#[test]
fn color_groups_share_the_vertex_function_s_id() {
    let Some(bc) = fixture() else { return };
    let dev = withheld(4);
    let d = pipeline(&bc, "pi_place", Some("pi_five"), 5);
    let groups: Vec<RenderPipeline> = (0..2)
        .map(|g| {
            let mut v = VariantKey::new();
            v.color_group = g;
            dev.compile_render_variant(&d, &v, &Budget::standard())
                .unwrap_or_else(|e| panic!("group {g}: {e}"))
        })
        .collect();
    for p in &groups {
        assert!(reads_layout(p));
        validate(p, &dev);
    }
    assert_eq!(
        stage(&groups[0], vk::ShaderStage::VERTEX).spirv,
        stage(&groups[1], vk::ShaderStage::VERTEX).spirv
    );
}

// ---- Indexed draws: the de-indexed form (VariantKey::primitive_id_indices) -------------------

/// A state binding `pi_place`'s and `pi_counts`' buffers, at device addresses.
fn bound() -> crate::bind::BindState {
    use crate::bind::{BindState, BoundBuffer, StageTable};
    let buffer = |length| BoundBuffer::Buffer {
        length,
        offset: 0,
        gpu_address: 0x1_0000,
        device_address: Some(0x7_0000_0000),
        attribute_stride: None,
    };
    let mut v = StageTable::default();
    for (index, length) in [(0, 1024), (1, 256), (2, 256)] {
        v.buffers.insert(index, buffer(length));
    }
    let mut s = BindState::default();
    s.stages.insert(crate::pipeline::sym::Stage::Vertex, v);
    s
}

/// The `u16` indexed draw of 12 indices from index 2, base vertex 10, two instances from 1.
fn indexed() -> crate::bind::DrawCall {
    crate::bind::DrawCall::Indexed {
        index_type: vk::IndexType::UINT16,
        index_buffer: crate::bind::IndexBufferRef::new(1, 64),
        index_count: 12,
        index_offset: 4,
        instance_count: 2,
        base_vertex: 10,
        base_instance: 1,
    }
}

/// The words of `field`'s place in `commands`' vertex push bytes, under `plan`.
fn pushed(
    plan: &crate::bind::BindingPlan,
    commands: &crate::bind::DrawCommands,
    field: PushField,
    words: usize,
) -> Vec<u32> {
    let place = plan
        .push
        .fields
        .iter()
        .find(|f| f.stage == crate::pipeline::sym::Stage::Vertex && f.field == field)
        .expect("placed");
    let crate::bind::FieldLocation::PushWindow { offset } = place.location else {
        panic!("{field:?} spilled: {place:?}");
    };
    let write = commands
        .push_constants
        .iter()
        .find(|w| w.stages.contains(vk::ShaderStageFlags::VERTEX))
        .expect("vertex push");
    let at = (offset - write.offset) as usize;
    write.bytes[at..at + 4 * words]
        .chunks(4)
        .map(|w| u32::from_le_bytes(w.try_into().expect("a word")))
        .collect()
}

/// An indexed draw of a pipeline whose vertex function writes the ID asks for the de-indexed
/// variant of its index type through the shape (`BindState::primitive_id_indices`). That variant
/// reads its indices at `PushField::CaptureIndexAddress`, writes the ID as `VertexIndex / n` (no
/// base vertex), and draws the indices as `index count` vertices from 0 with layout
/// `[0, index count, base instance, base vertex]`. Every module is valid. A draw without indices
/// asks for the variant without the form; an indexed strip is not implemented.
#[test]
fn an_indexed_draw_runs_the_de_indexed_variant() {
    use crate::bind::{DrawCall, PushHole};
    use crate::error::BindError;
    use crate::error::{Feature, Unsupported};
    use crate::metal::{IndexType, PrimitiveType};
    let Some(bc) = fixture() else { return };
    let dev = withheld(8);
    let d = pipeline(&bc, "pi_place", Some("pi_ids"), 1);
    let base = compile(&dev, &d).expect("base");
    let mut state = bound();
    state.primitive_type = Some(PrimitiveType::Triangle);
    state.primitive_id_indices = Some(IndexType::UInt16);
    let plan = base.plan(&dev, &state.shape()).expect("base plan");
    assert_eq!(plan.variant.primitive_id_indices, Some(IndexType::UInt16));
    let key = plan.variant.clone();
    let p = dev
        .compile_render_variant(&d, &key, &Budget::standard())
        .expect("de-indexed variant");
    validate(&p, &dev);
    assert!(reads_layout(&p));
    assert!(reads_field(&p, PushField::CaptureIndexAddress));
    assert!(reads_field(&p, PushField::CaptureLayout));
    assert!(!reads_field(&p, PushField::BaseVertexInstance));
    let plan = p.plan(&dev, &state.shape()).expect("plan");
    assert_eq!(plan.variant, key);
    let commands = plan.instantiate_draw(&state, &indexed()).expect("draw");
    assert_eq!(
        commands.draw,
        vk::DrawCommand::Draw {
            vertex_count: 12,
            instance_count: 2,
            first_vertex: 0,
            first_instance: 1,
        }
    );
    assert!(commands
        .push_holes
        .iter()
        .any(|h| matches!(h, PushHole::CaptureIndexAddress { .. })));
    assert_eq!(
        pushed(&plan, &commands, PushField::CaptureLayout, 4),
        [0, 12, 1, 10]
    );
    assert_eq!(
        pushed(&plan, &commands, PushField::VerticesPerPrimitive, 1),
        [3]
    );
    let not_implemented = Err(BindError::Unsupported(Unsupported::not_implemented(
        Feature::PrimitiveId,
    )));
    let mut strip = state.clone();
    strip.primitive_type = Some(PrimitiveType::TriangleStrip);
    assert_eq!(
        plan.instantiate_draw(&strip, &indexed()).map(|_| ()),
        not_implemented
    );
    let unindexed = DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 0,
    };
    assert_eq!(
        plan.instantiate_draw(&state, &unindexed).map(|_| ()),
        Err(BindError::NeedsVariant {
            variant: Box::new(VariantKey {
                primitive_id_indices: None,
                ..key.clone()
            }),
        })
    );
    // Without the shape's index type, the base plan refuses the indexed draw.
    let mut plain = state.clone();
    plain.primitive_id_indices = None;
    let plan = base.plan(&dev, &plain.shape()).expect("plan");
    assert!(plan.variant.is_empty());
    assert_eq!(
        plan.instantiate_draw(&plain, &indexed()).map(|_| ()),
        not_implemented
    );
    // A pipeline that writes no ID asks for no such variant.
    let none = compile(&dev, &pipeline(&bc, "pi_place", None, 0)).expect("no fragment");
    let plan = none.plan(&dev, &state.shape()).expect("plan");
    assert_eq!(plan.variant.primitive_id_indices, None);
}

/// A vertex function that writes device memory (`pi_counts`' atomic counter) would run once per
/// index in the de-indexed form: that variant is `NotImplemented(PrimitiveId)`, and its draws
/// without indices still run.
#[test]
fn a_memory_writing_vertex_function_is_refused_de_indexed() {
    use crate::error::{Feature, Needs};
    let Some(bc) = fixture() else { return };
    let dev = withheld(8);
    let d = pipeline(&bc, "pi_counts", Some("pi_ids"), 1);
    compile(&dev, &d).expect("without indices");
    let mut key = VariantKey::new();
    key.primitive_id_indices = Some(crate::metal::IndexType::UInt32);
    match dev.compile_render_variant(&d, &key, &Budget::standard()) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::PrimitiveId);
            assert_eq!(u.needs, Needs::NotImplemented);
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
}

/// A private type's draw in the private list form (`BindState::private_list`) asks for the listed
/// variant through the shape. It reads its list's address (`PushField::CaptureIndexAddress`, a
/// push hole the consumer fills) and no capture layout, and draws the list's positions without
/// indices from the call's first vertex, the list's first index, with its instances; its vertices
/// per primitive are the list's, twice that for Metal's quads listed as two triangles a quad. A
/// vertex function that writes memory, which the de-indexed form refuses, runs in it, on a device
/// with both features too, where nothing reads the primitive ID's varying. Every module is valid.
#[test]
fn a_private_draw_in_the_list_form_runs_the_listed_variant() {
    use crate::bind::{DrawCall, PushHole};
    use crate::metal::PrimitiveType;
    use crate::pipeline::RuntimeFact;
    let Some(bc) = fixture() else { return };
    let listed = |dev: &Device, vertex: &str| {
        let d = pipeline(&bc, vertex, Some("pi_ids"), 1);
        let base = compile(dev, &d).expect("base");
        let mut state = bound();
        state.primitive_type = Some(PrimitiveType::Triangle);
        state.private_list = true;
        let plan = base.plan(dev, &state.shape()).expect("base plan");
        assert!(plan.variant.private_list, "{vertex}: the plan's variant");
        let key = plan.variant.clone();
        let p = dev
            .compile_render_variant(&d, &key, &Budget::standard())
            .unwrap_or_else(|e| panic!("{vertex}: the listed variant: {e}"));
        validate(&p, dev);
        assert!(p
            .dependencies
            .iter()
            .any(|d| d.fact == RuntimeFact::PrivateList));
        assert!(reads_field(&p, PushField::CaptureIndexAddress), "{vertex}");
        assert!(!reads_field(&p, PushField::CaptureLayout), "{vertex}");
        let plan = p.plan(dev, &state.shape()).expect("plan");
        assert_eq!(plan.variant, key);
        let draw = DrawCall::Primitives {
            vertex_start: 1,
            vertex_count: 12,
            instance_count: 2,
            base_instance: 1,
        };
        let commands = plan.instantiate_draw(&state, &draw).expect("draw");
        assert_eq!(
            commands.draw,
            vk::DrawCommand::Draw {
                vertex_count: 12,
                instance_count: 2,
                first_vertex: 1,
                first_instance: 1,
            }
        );
        assert!(commands
            .push_holes
            .iter()
            .any(|h| matches!(h, PushHole::CaptureIndexAddress { .. })));
        (plan, state, draw)
    };
    let (plan, mut state, draw) = listed(&withheld(8), "pi_place");
    let commands = plan.instantiate_draw(&state, &draw).expect("draw");
    assert_eq!(
        pushed(&plan, &commands, PushField::VerticesPerPrimitive, 1),
        [3]
    );
    state.quad_primitive_ids = true;
    let commands = plan.instantiate_draw(&state, &draw).expect("quads");
    assert_eq!(
        pushed(&plan, &commands, PushField::VerticesPerPrimitive, 1),
        [6]
    );
    listed(&withheld(8), "pi_counts");
    listed(&device("lavapipe"), "pi_counts");
}
