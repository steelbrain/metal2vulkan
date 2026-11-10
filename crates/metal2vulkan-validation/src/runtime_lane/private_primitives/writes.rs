//! Device tests of a vertex function that writes memory under draws of Metal's private primitive
//! types 5 to 8, on the device the cfg selects, under the validation layer. The owned fixture
//! `rt6_private_writes`'s `pw_vertex` counts each of its invocations at its position's slot of
//! buffer 0 (its vertex ID less its base vertex, modulo 64; a second 64 slots for the instances
//! past the first) and stores the vertex ID and base vertex it read at that slot of buffer 1.
//!
//! Metal's answers (probes on the Apple M3 Max, macOS 27.0.1, 25 draws of types 5 to 8 and the
//! public triangle list, two instances each: direct from vertex starts 3, 2^31 - 3, 2^31 + 3 and
//! 2^32 - 4, and of 10 32-bit indices from base vertex 7): the vertex function runs once for each
//! vertex of the list [`metal_list`] states, in each instance, so a vertex the list names `k` times
//! runs `k` times (a fan's first `n - 2` times); `[[vertex_id]]` is the vertex start plus the
//! position, wrapping past `u32::MAX`, or the index plus the base vertex, and `[[base_vertex]]`
//! the vertex start or the base vertex. Vulkan's indexed draw of the list may run a vertex once for
//! all the primitives that share it, so the runtime draws such a function's private draws in the
//! private list form (`VariantKey::private_list`): each list position runs once, its index and
//! Metal's base vertex read at the list's address; a direct draw from a vertex start past
//! `i32::MAX`, which an indexed list's base vertex cannot carry, runs so whatever its functions.
//!
//! Each check draws a private type in one form and asks every slot's count, vertex ID and base
//! vertex to be Metal's, and the draw's pipeline to carry the form (`RuntimeFact::PrivateList`).

use super::super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, Messages, ScratchCache, WAIT,
};
use super::{metal_list, Form, PRIVATE};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use crate::runtime_lane::at_plan;
use crate::runtime_lane::validated_device_with;
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::{IndexType, PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, RuntimeFact, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, Device, DrawPrimitiveType, ExecutedPipeline, LoadAction,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, StoreAction,
    TextureDescriptor,
};

/// The slots of one instance (of the first, and of those past it).
const SLOTS: usize = 64;

/// The value of a slot of buffer 1 no invocation wrote.
const UNSEEN: u32 = 0xeeee_eeee;

/// A draw of a private type.
#[derive(Clone, Debug)]
struct WriteCase {
    primitive: DrawPrimitiveType,
    form: Form,
    count: u32,
    /// A draw without indices: its vertex start.
    start: u32,
    base_vertex: i32,
    /// The index buffer's indices, all below [`SLOTS`] (empty for a draw without indices).
    indices: Vec<u32>,
}

/// Two instances, from base instance 0.
const INSTANCES: u32 = 2;

impl WriteCase {
    /// Each position's index as Metal reads it: a draw without indices its position.
    fn values(&self) -> Vec<u32> {
        match self.form.index_type() {
            None => (0..self.count).collect(),
            Some(_) => self.indices[..self.count as usize].to_vec(),
        }
    }

    /// The base the vertex ID adds to each value, which `[[base_vertex]]` reads.
    fn base(&self) -> u32 {
        match self.form.index_type() {
            None => self.start,
            Some(_) => self.base_vertex as u32,
        }
    }

    /// Metal's counts and seen values, slot by slot, of both instances.
    fn expected(&self) -> (Vec<u32>, Vec<[u32; 2]>) {
        let mut counts = vec![0; 2 * SLOTS];
        let mut seen = vec![[UNSEEN; 2]; 2 * SLOTS];
        let list = metal_list(self.primitive, &self.values(), u32::MAX);
        for instance in 0..INSTANCES as usize {
            for &value in &list {
                let slot = (value as usize % SLOTS) + SLOTS * instance.min(1);
                counts[slot] += 1;
                seen[slot] = [value.wrapping_add(self.base()), self.base()];
            }
        }
        (counts, seen)
    }
}

/// The cases: each type direct from four vertex starts, two of them past `i32::MAX` (the last
/// wrapping), and indirect from one past it; indexed of both index types, direct and indirect.
fn cases() -> Vec<WriteCase> {
    let indices: Vec<u32> = (0..14).map(|i| (i * 5 + 1) % 13).collect();
    let mut cases = Vec::new();
    for primitive in PRIVATE {
        let direct = |form, start| WriteCase {
            primitive,
            form,
            count: 10,
            start,
            base_vertex: 0,
            indices: Vec::new(),
        };
        for start in [3, 0x7fff_fffd, 0x8000_0003, 0xffff_fffc] {
            cases.push(direct(Form::Direct, start));
        }
        cases.push(direct(Form::Indirect, 0x8000_0003));
        for form in [
            Form::Indexed(IndexType::UInt16),
            Form::Indexed(IndexType::UInt32),
            Form::IndexedIndirect(IndexType::UInt32),
        ] {
            cases.push(WriteCase {
                primitive,
                form,
                count: 14,
                start: 0,
                base_vertex: 7,
                indices: indices.clone(),
            });
        }
    }
    cases
}

/// The fixture's pipeline on `device`.
fn pipeline(device: &Device) -> RenderPipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("rt6_private_writes.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = |name| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("pw_vertex"));
    descriptor.fragment = Some(function("pw_fragment"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Draw `case` with `pipeline`: the counts and seen values of both instances, and whether the
/// draw's pipeline carries the private list form.
fn draw(
    device: &Device,
    pipeline: &RenderPipelineState,
    case: &WriteCase,
) -> (Vec<u32>, Vec<[u32; 2]>, bool) {
    let counts = buffer_with(device, &vec![0; 2 * SLOTS * 4]).unwrap();
    let seen = buffer_with(device, &vec![0xee; 2 * SLOTS * 8]).unwrap();
    let index_type = case.form.index_type().unwrap_or(IndexType::UInt32);
    let index_bytes: Vec<u8> = match index_type {
        IndexType::UInt16 => case
            .indices
            .iter()
            .flat_map(|&i| (i as u16).to_le_bytes())
            .collect(),
        _ => case.indices.iter().flat_map(|i| i.to_le_bytes()).collect(),
    };
    let index_buffer = buffer_with(device, &[index_bytes, vec![0; 4]].concat()).unwrap();
    let words: Vec<u32> = match case.form {
        Form::Indirect => vec![case.count, INSTANCES, case.start, 0],
        _ => vec![case.count, INSTANCES, 0, case.base_vertex as u32, 0],
    };
    let arguments: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let indirect = buffer_with(device, &arguments).unwrap();
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 8, 8, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let target = device.new_texture(&target).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target);
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder.set_vertex_buffer(Some(&counts), 0, 0).unwrap();
    encoder.set_vertex_buffer(Some(&seen), 0, 1).unwrap();
    let drawn = match case.form {
        Form::Direct => {
            encoder.draw_primitives(case.primitive, case.start, case.count, INSTANCES, 0)
        }
        Form::Indirect => encoder.draw_primitives_indirect(case.primitive, &indirect, 0),
        Form::Indexed(t) | Form::Written(t) => encoder.draw_indexed_primitives(
            case.primitive,
            case.count,
            t,
            &index_buffer,
            0,
            INSTANCES,
            case.base_vertex,
            0,
        ),
        Form::IndexedIndirect(t) => encoder.draw_indexed_primitives_indirect(
            case.primitive,
            t,
            &index_buffer,
            0,
            &indirect,
            0,
        ),
    };
    drawn.unwrap_or_else(|e| panic!("{case:?}: {e}"));
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{case:?}: {:?}",
        command_buffer.error()
    );
    let listed = command_buffer
        .executions()
        .iter()
        .any(|execution| match execution.pipeline() {
            ExecutedPipeline::Render(pipeline) => pipeline
                .dependencies
                .iter()
                .any(|dependency| dependency.fact == RuntimeFact::PrivateList),
            ExecutedPipeline::Compute(_) => false,
        });
    (words_of(&counts), pairs_of(&seen), listed)
}

/// `buffer`'s words.
fn words_of(buffer: &Buffer) -> Vec<u32> {
    let bytes = read_buffer(buffer, 0, buffer.length()).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// `buffer`'s pairs of words.
fn pairs_of(buffer: &Buffer) -> Vec<[u32; 2]> {
    words_of(buffer).as_chunks::<2>().0.to_vec()
}

/// A vertex function that writes memory runs once for each vertex of Metal's list, in each
/// instance, and reads Metal's vertex ID and base vertex, in every form, from vertex starts past
/// `i32::MAX` too; each draw runs the private list form.
#[test]
fn vertex_functions_that_write_memory_run_once_a_list_vertex() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        let stores = device
            .compile_device()
            .profile()
            .enabled()
            .features
            .contains(&VkFeature::VertexPipelineStoresAndAtomics);
        if !stores {
            eprintln!("{name}: no vertexPipelineStoresAndAtomics, so no vertex writes: skipped");
            return;
        }
        let pipeline = pipeline(&device);
        let cases = cases();
        for case in &cases {
            let (counts, seen, listed) = draw(&device, &pipeline, case);
            let (want_counts, want_seen) = case.expected();
            assert_eq!(
                counts, want_counts,
                "{case:?} on {name}: the invocation counts"
            );
            assert_eq!(
                seen, want_seen,
                "{case:?} on {name}: the vertex IDs and base vertices"
            );
            assert!(
                listed,
                "{case:?} on {name}: not drawn in the private list form"
            );
        }
        eprintln!(
            "{name}: {} private draws of a memory-writing vertex function ran once a list vertex",
            cases.len()
        );
    }
    layer_reported_nothing_at_all(&messages);
}
