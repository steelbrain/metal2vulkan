//! Device tests of a draw's base vertex and base instance in a vertex function on the device the
//! cfg selects, under the validation layer.
//!
//! The committed cases of the owned fixture `rvc_draw_bases` draw rasterization-disabled points
//! whose vertex `vid` of instance `iid` writes `vid`, `iid`, `[[base_vertex]]` and
//! `[[base_instance]]` (as `uint`, or as `ushort` in the narrow entry) to slot
//! `(iid % 4) * 8 + vid % 8` of `out`: four points from `vertexStart` `0xfffffffa` with base
//! instance 7 or `0x10007`, and four of the indices 5 to 8 (16- or 32-bit) under base vertex -3
//! with base instance 2. Metal's base vertex is the draw's `vertexStart` without an index buffer
//! and its `baseVertex` with one, and its base instance the draw's `baseInstance`; the cases'
//! committed observations record its answers on an M3 Max and an M4.
//!
//! Each case's draw runs here directly, indirectly (`MTLDrawPrimitivesIndirectArguments` or
//! `MTLDrawIndexedPrimitivesIndirectArguments` of the same values), and as the one command of an
//! indirect command buffer of draws executed in the encoder, and stores the M3's answer each way.

use super::degrade_notices::on_device_with;
use super::{at_plan, buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use crate::case::{AuthoredCase, IndexType as CaseIndexType, OutputSelection, Primitive};
use crate::store::CorpusStore;
use base64::Engine as _;
use metal2vulkan::metal::{
    IndexType, IndirectCommandType, PrimitiveType, RenderStages, ResourceUsage,
};
use metal2vulkan::pipeline::{RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, Device, IndirectCommandBufferDescriptor, PipelineOptions, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions,
};

/// The fixture the cases draw.
const FIXTURE: &str = "rvc_draw_bases.ll";

/// How a case's draw is issued.
#[derive(Clone, Copy, Debug)]
enum Issued {
    Direct,
    Indirect,
    Command,
}

/// The committed cases of [`FIXTURE`], with the output bytes of each one's current Metal
/// observation in the lane's environment.
fn cases() -> Vec<(AuthoredCase, Vec<u8>)> {
    let source = format!("{}/engine/{FIXTURE}", crate::source::PUBLIC_FIXTURE_DIR);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let store = CorpusStore::new(&root);
    let mut out = Vec::new();
    for shard in 0..64 {
        for case in store.read_case_shard(shard).unwrap() {
            if case.public_source.as_deref() == Some(source.as_str()) {
                let metal = super::metal_output(&store, &case)
                    .unwrap_or_else(|e| panic!("{}: {e}", case.name));
                out.push((case, metal));
            }
        }
    }
    out
}

/// `entry`'s rasterization-disabled pipeline state, usable in an indirect command buffer.
fn state(device: &Device, entry: &str) -> RenderPipelineState {
    let bitcode = llvm_assemble(&engine_dir().join(FIXTURE)).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(entry).unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.rasterization_enabled = false;
    descriptor.support_indirect_command_buffers = true;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A buffer of little-endian `u32`s.
fn words(device: &Device, words: &[u32]) -> Buffer {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    buffer_with(device, &bytes).unwrap()
}

/// The output bytes `case`'s draw stores when `issued`.
fn drawn(device: &Device, case: &AuthoredCase, issued: Issued) -> Vec<u8> {
    let draw = case.draw.as_ref().expect("a vertex case's draw");
    assert_eq!(draw.primitive, Primitive::Point, "{}", case.name);
    let initial = base64::engine::general_purpose::STANDARD
        .decode(
            case.buffers[0]
                .initial_bytes_b64
                .as_deref()
                .expect("the output's initial bytes"),
        )
        .unwrap();
    let out = buffer_with(device, &initial).unwrap();
    let OutputSelection::Buffer { offset, length, .. } = case.output else {
        panic!("{}: a buffer output", case.name)
    };
    let index = draw.index_buffer.as_ref().map(|literal| {
        let bytes = literal.bytes().expect("the indices");
        let index_type = match literal.index_type {
            CaseIndexType::Uint16 => IndexType::UInt16,
            CaseIndexType::Uint32 => IndexType::UInt32,
        };
        (
            buffer_with(device, &bytes).unwrap(),
            index_type,
            u64::from(literal.offset),
        )
    });
    let state = state(device, &case.entry);
    let mut pass = RenderPassDescriptor::new();
    pass.render_target_width = 1;
    pass.render_target_height = 1;
    pass.default_raster_sample_count = 1;
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    let point = PrimitiveType::Point;
    let mut held = Vec::new();
    match issued {
        Issued::Direct | Issued::Indirect => {
            encoder.set_render_pipeline_state(&state).unwrap();
            encoder.set_vertex_buffer(Some(&out), 0, 0).unwrap();
        }
        Issued::Command => {
            encoder
                .use_resource(
                    out.as_resource(),
                    ResourceUsage::WRITE,
                    RenderStages::VERTEX,
                )
                .unwrap();
            if let Some((indices, _, _)) = &index {
                encoder
                    .use_resource(
                        indices.as_resource(),
                        ResourceUsage::READ,
                        RenderStages::VERTEX,
                    )
                    .unwrap();
            }
        }
    }
    match (issued, &index) {
        (Issued::Direct, None) => encoder.draw_primitives(
            point,
            draw.vertex_start,
            draw.vertex_count,
            draw.instance_count,
            draw.base_instance,
        ),
        (Issued::Direct, Some((indices, index_type, offset))) => encoder.draw_indexed_primitives(
            point,
            draw.vertex_count,
            *index_type,
            indices,
            *offset,
            draw.instance_count,
            draw.base_vertex,
            draw.base_instance,
        ),
        // {vertexCount, instanceCount, vertexStart, baseInstance}.
        (Issued::Indirect, None) => {
            let arguments = words(
                device,
                &[
                    draw.vertex_count,
                    draw.instance_count,
                    draw.vertex_start,
                    draw.base_instance,
                ],
            );
            let drawn = encoder.draw_primitives_indirect(point, &arguments, 0);
            held.push(arguments);
            drawn
        }
        // {indexCount, instanceCount, indexStart, baseVertex, baseInstance}.
        (Issued::Indirect, Some((indices, index_type, offset))) => {
            let arguments = words(
                device,
                &[
                    draw.vertex_count,
                    draw.instance_count,
                    0,
                    draw.base_vertex as u32,
                    draw.base_instance,
                ],
            );
            let drawn = encoder.draw_indexed_primitives_indirect(
                point,
                *index_type,
                indices,
                *offset,
                &arguments,
                0,
            );
            held.push(arguments);
            drawn
        }
        (Issued::Command, _) => {
            let mut descriptor = IndirectCommandBufferDescriptor::new();
            descriptor.command_types =
                IndirectCommandType::DRAW.union(IndirectCommandType::DRAW_INDEXED);
            descriptor.max_vertex_buffer_bind_count = 1;
            let icb = device
                .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
                .unwrap();
            let command = icb.indirect_render_command_at(0).unwrap();
            command.set_render_pipeline_state(&state).unwrap();
            command.set_vertex_buffer(Some(&out), 0, 0).unwrap();
            match &index {
                None => command.draw_primitives(
                    point,
                    u64::from(draw.vertex_start),
                    u64::from(draw.vertex_count),
                    u64::from(draw.instance_count),
                    u64::from(draw.base_instance),
                ),
                Some((indices, index_type, offset)) => command.draw_indexed_primitives(
                    point,
                    u64::from(draw.vertex_count),
                    *index_type,
                    indices,
                    *offset,
                    u64::from(draw.instance_count),
                    i64::from(draw.base_vertex),
                    u64::from(draw.base_instance),
                ),
            }
            .unwrap();
            encoder.execute_commands_in_buffer(&icb, 0..1)
        }
    }
    .unwrap_or_else(|e| panic!("{} {issued:?}: {e}", case.name));
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None, "{} {issued:?}", case.name);
    drop(held);
    read_buffer(&out, offset, length).unwrap()
}

/// Each committed case of the fixture, drawn directly, indirectly and from an indirect command
/// buffer, stores the M3's answer: its vertices read the draw's base vertex (the vertex start of
/// a draw without an index buffer, the base vertex of an indexed one) and base instance, whole or
/// as their low 16 bits.
#[test]
fn draws_read_metal_s_base_vertex_and_base_instance() {
    let cases = cases();
    assert_eq!(cases.len(), 4, "the fixture's committed cases");
    on_device_with(at_plan, |device, _| {
        for (case, metal) in &cases {
            for issued in [Issued::Direct, Issued::Indirect, Issued::Command] {
                assert_eq!(
                    drawn(device, case, issued),
                    *metal,
                    "{} {issued:?} on {}",
                    case.name,
                    device.name()
                );
            }
        }
    });
}
