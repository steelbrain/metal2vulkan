//! Indirect command buffers' setters (`indirect_command_buffer.rs`) beside a model of a setter as
//! a whole-record edit: the field written by `CommandRecord::write_field` into a copy of the
//! record as it stands, then what the record holds after it (its action and barrier, and the
//! state the command names) added to what the records may hold, every edit. A mix of render and
//! compute commands runs, refused calls and resets among them, records the GPU may have written
//! and one that does not decode; after every call the storage's bytes, what the records may hold
//! (and what a buffer a blit copied the records into holds), and the CPU's writes counted must be
//! the model's.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects; nothing is submitted. The functions are
//! the owned fixtures `validation/fixtures/public/examples/runtime_examples.ll` and
//! `engine/icb_replay_encode_dispatch.ll` (`llvm-as`).

use super::blit_holds::{device, Scratch};
use super::encoder_holds::library;
use crate::error::{internal, invalid};
use crate::write_routes::GpuWriter;
use crate::written_commands::Named;
use crate::{
    Buffer, ComputePipelineState, Device, DrawProblem, Error, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, IndirectCommandBufferProblem, PipelineOptions, Problem,
    Region, RenderPipelineState, ResourceOptions, RuntimeGap, Size,
};
use metal2vulkan::bind::indirect_commands::{
    Action, BufferBinding, BufferStage, CommandRecord, RecordField, RecordView,
    COMMAND_BUFFER_HEADER_BYTES, COMMAND_BYTES,
};
use metal2vulkan::metal::{IndexType, IndirectCommandType, PixelFormat, PrimitiveType};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, CommandWrites, ComputePipelineDescriptor, RenderPipelineDescriptor,
    StageSource,
};
use std::num::NonZeroU64;

/// The commands of each buffer.
const COMMANDS: u32 = 6;

/// The owned fixtures' paths under `validation/fixtures/public/`.
const EXAMPLES: &str = "examples/runtime_examples.ll";
const ENCODES: &str = "engine/icb_replay_encode_dispatch.ll";

/// What a setter's call leaves, as the model has it.
struct Model {
    /// The storage's bytes: the header, then each record.
    bytes: Vec<u8>,
    /// What the state each command names may do, where it names one the CPU set.
    named: Vec<Option<Named>>,
    /// What the records may hold: every edit's record and state, together.
    held: (CommandWrites, Named),
    /// The CPU's writes counted.
    cpu_writes: u64,
}

/// What either may do (`Named`'s union).
fn either(one: Named, other: Named) -> Named {
    Named {
        encodes: one.encodes || other.encodes,
        writes: one.writes.union(other.writes),
    }
}

/// The error a setter answers for a record that does not decode.
fn undecodable() -> Error {
    internal("an indirect command's record does not decode")
}

/// The storage's bytes of `icb`.
fn storage(icb: &IndirectCommandBuffer) -> Vec<u8> {
    let length = COMMAND_BUFFER_HEADER_BYTES + icb.size() as usize * COMMAND_BYTES;
    let contents = icb.storage().contents().unwrap();
    // SAFETY: the storage is mapped and holds the header and every record; nothing writes it
    // meanwhile.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length) }.to_vec()
}

/// The byte range of command `index`'s record in the storage.
fn record_at(index: u32) -> std::ops::Range<usize> {
    let start = COMMAND_BUFFER_HEADER_BYTES + index as usize * COMMAND_BYTES;
    start..start + COMMAND_BYTES
}

impl Model {
    /// The model of `icb` as made.
    fn of(icb: &IndirectCommandBuffer) -> Model {
        Model {
            bytes: storage(icb),
            named: vec![None; icb.size() as usize],
            held: Default::default(),
            cpu_writes: icb.cpu_writes(),
        }
    }

    /// A setter's call that writes `field` into command `index`'s record, setting a state that
    /// may do `set` where `Some`, answered `result`: what the model's edit refuses, the call
    /// must refuse, and the rest it must write.
    fn edit(
        &mut self,
        icb: &IndirectCommandBuffer,
        (result, index, field): (Result<(), Error>, u32, RecordField),
        set: Option<Named>,
        copy: Option<&IndirectCommandBuffer>,
    ) {
        let mut record = self.bytes[record_at(index)].to_vec();
        match CommandRecord::write_field(&mut record, field) {
            Err(_) => assert_eq!(result, Err(undecodable()), "{field:?} of {index}"),
            Ok(_) => {
                assert_eq!(result, Ok(()), "{field:?} of {index}");
                self.bytes[record_at(index)].copy_from_slice(&record);
                self.cpu_writes += 1;
                if set.is_some() {
                    self.named[index as usize] = set;
                }
                let named = self.named[index as usize];
                let mut writes = RecordView::new(&record).unwrap().writes();
                if named.is_some() {
                    writes.operations |= CommandWrites::PIPELINE;
                }
                self.held = (
                    self.held.0.union(writes),
                    either(self.held.1, named.unwrap_or_default()),
                );
            }
        }
        self.check(icb, copy, &format!("{field:?} of {index}"));
    }

    /// A call answered `result` that writes no record: refused with `refused`, or taken (`None`).
    fn unwritten(
        &mut self,
        icb: &IndirectCommandBuffer,
        result: Result<(), Error>,
        refused: Option<Error>,
        copy: Option<&IndirectCommandBuffer>,
    ) {
        let expected = refused.map_or(Ok(()), Err);
        assert_eq!(result, expected);
        self.check(icb, copy, &format!("{expected:?}"));
    }

    /// Commands `range` reset by `reset`, a CPU reset counted as `writes` writes: each record as
    /// `reset` leaves it, the state set on it dropped where `drops`.
    fn reset(
        &mut self,
        icb: &IndirectCommandBuffer,
        result: Result<(), Error>,
        (range, writes): (std::ops::Range<u32>, u64),
        reset: impl Fn(CommandRecord) -> CommandRecord,
        drops: bool,
    ) {
        assert_eq!(result, Ok(()));
        for index in range {
            let at = record_at(index);
            let record = reset(CommandRecord::decode(&self.bytes[at.clone()]).unwrap());
            record.encode_into(&mut self.bytes[at]).unwrap();
            if drops {
                self.named[index as usize] = None;
            }
        }
        self.cpu_writes += writes;
        self.check(icb, None, "reset");
    }

    /// `write` of command `index`'s record, as the GPU writes it, on the buffer and the model
    /// alike: no CPU write.
    fn gpu_write(&mut self, icb: &IndirectCommandBuffer, index: u32, write: impl Fn(&mut [u8])) {
        assert!(icb.gpu_written());
        let at = record_at(index);
        write(&mut self.bytes[at.clone()]);
        let contents = icb.storage().contents().unwrap();
        // SAFETY: the record lies inside the mapped storage; nothing else writes it meanwhile.
        let bytes = unsafe { std::slice::from_raw_parts_mut(contents.as_ptr(), self.bytes.len()) };
        bytes[at.clone()].copy_from_slice(&self.bytes[at]);
    }

    /// The buffer, and `copy`, which a blit copied its records into, against the model.
    fn check(
        &self,
        icb: &IndirectCommandBuffer,
        copy: Option<&IndirectCommandBuffer>,
        after: &str,
    ) {
        assert!(storage(icb) == self.bytes, "the storage after {after}");
        let held = (icb.written().writes(), icb.written().named());
        assert_eq!(held, self.held, "what the records may hold after {after}");
        if let Some(copy) = copy {
            let copied = (copy.written().writes(), copy.written().named());
            assert_eq!(copied, self.held, "what the copy may hold after {after}");
        }
        assert_eq!(
            icb.cpu_writes(),
            self.cpu_writes,
            "the CPU's writes after {after}"
        );
    }
}

/// What `state` may do to indirect command buffers, as a command names it.
fn named_render(state: &RenderPipelineState) -> Named {
    Named {
        encodes: state.can_encode_indirect_commands(),
        writes: state.indirect_command_writes(),
    }
}

/// What `state` may do, as `named_render`'s.
fn named_compute(state: &ComputePipelineState) -> Named {
    Named {
        encodes: state.can_encode_indirect_commands(),
        writes: state.indirect_command_writes(),
    }
}

/// The ID a command's record names for a state of ID `id`.
fn pipeline(id: u64) -> RecordField {
    RecordField::Pipeline(NonZeroU64::new(id).unwrap())
}

/// The field a buffer setter writes: `buffer` (none: nil) at `offset`'s low 32 bits.
fn bound(stage: BufferStage, slot: usize, buffer: Option<&Buffer>, offset: u64) -> RecordField {
    let base = buffer.map_or(0, |buffer| buffer.gpu_address().0);
    RecordField::Buffer {
        stage,
        slot,
        binding: Some(BufferBinding {
            address: base + u64::from(offset as u32),
            attribute_stride: None,
        }),
    }
}

/// A state drawing into `RGBA8Unorm` with the examples' `colored_vertex` and `colored_fragment`,
/// made for indirect command buffers where `icb`.
fn render_state(device: &Device, icb: bool) -> RenderPipelineState {
    let library = library(device, EXAMPLES);
    let vertex = library.new_function_with_name("colored_vertex").unwrap();
    let fragment = library.new_function_with_name("colored_fragment").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.support_indirect_command_buffers = icb;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A state of `kernel` of the fixture at `path`, made for indirect command buffers where `icb`.
fn compute_state(device: &Device, path: &str, kernel: &str, icb: bool) -> ComputePipelineState {
    let function = library(device, path)
        .new_function_with_name(kernel)
        .unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = icb;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

fn problem(call: &'static str, problem: IndirectCommandBufferProblem) -> Error {
    invalid(call, Problem::IndirectCommandBuffer(problem))
}

/// Render commands: each setter's record, what the records may hold and the CPU's writes are
/// the model's, through fields that add, fields that add nothing new, refusals and calls taken
/// mid-command, a command's reset and a range's, a record the GPU may have written and one that
/// does not decode.
#[test]
fn render_setters_write_as_a_whole_record_edit() {
    let scratch = Scratch::new("icb-render-setters");
    let device = device(&scratch.0);
    let other = device_of_another(&scratch);
    let state = render_state(&device, true);
    let unsupported = render_state(&device, false);
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW | IndirectCommandType::DRAW_INDEXED;
    descriptor.max_vertex_buffer_bind_count = 2;
    descriptor.max_fragment_buffer_bind_count = 1;
    let shared = ResourceOptions::shared();
    let icb = device
        .new_indirect_command_buffer(&descriptor, u64::from(COMMANDS), shared)
        .unwrap();
    let copy = device
        .new_indirect_command_buffer(&descriptor, u64::from(COMMANDS), shared)
        .unwrap();
    let (a, b) = (
        device.new_buffer(4096, shared).unwrap(),
        device.new_buffer(4096, shared).unwrap(),
    );
    let indices = device.new_buffer(64, shared).unwrap();
    let foreign = other.new_buffer(64, shared).unwrap();
    let id = state.gpu_resource_id().0;
    let named = Some(named_render(&state));
    let draw = |primitive_type: u32| {
        RecordField::Action(Action::DrawPrimitives {
            primitive_type,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 0,
        })
    };
    let mut model = Model::of(&icb);
    let c: Vec<_> = (0..u64::from(COMMANDS))
        .map(|index| icb.indirect_render_command_at(index).unwrap())
        .collect();

    // A whole command, then the copy's link: what the records hold reaches it from then on.
    model.edit(
        &icb,
        (c[0].set_render_pipeline_state(&state), 0, pipeline(id)),
        named,
        None,
    );
    copy.written().copied_from(icb.written()).mark();
    let copy = Some(&copy);
    let v0 = bound(BufferStage::Vertex, 0, Some(&a), 16);
    model.edit(
        &icb,
        (c[0].set_vertex_buffer(Some(&a), 16, 0), 0, v0),
        None,
        copy,
    );
    let v1 = bound(BufferStage::Vertex, 1, Some(&b), 0);
    model.edit(
        &icb,
        (c[0].set_vertex_buffer(Some(&b), 0, 1), 0, v1),
        None,
        copy,
    );
    let f0 = bound(BufferStage::Fragment, 0, Some(&a), 256);
    model.edit(
        &icb,
        (c[0].set_fragment_buffer(Some(&a), 256, 0), 0, f0),
        None,
        copy,
    );
    let triangles = c[0].draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0);
    model.edit(&icb, (triangles, 0, draw(3)), None, copy);
    // The same command again, adding nothing new; then a draw of another primitive type, and
    // an offset past 32 bits, of which the record keeps the low 32.
    model.edit(
        &icb,
        (c[1].set_render_pipeline_state(&state), 1, pipeline(id)),
        named,
        copy,
    );
    let far = bound(BufferStage::Vertex, 0, Some(&a), (1 << 32) + 32);
    let set = c[1].set_vertex_buffer(Some(&a), (1 << 32) + 32, 0);
    model.edit(&icb, (set, 1, far), None, copy);
    let points = c[1].draw_primitives(PrimitiveType::Point, 0, 3, 1, 0);
    model.edit(&icb, (points, 1, draw(0)), None, copy);
    let indexed = |primitive_type: u32| Action::DrawIndexedPrimitives {
        primitive_type,
        index_count: 3,
        index_type: IndexType::UInt16.as_raw(),
        index_buffer: indices.gpu_address().0 + 2,
        instance_count: 1,
        base_vertex: -1,
        base_instance: 0,
    };
    let set = c[2].draw_indexed_primitives(
        PrimitiveType::Triangle,
        3,
        IndexType::UInt16,
        &indices,
        2,
        1,
        -1,
        0,
    );
    model.edit(&icb, (set, 2, RecordField::Action(indexed(3))), None, copy);

    // Refused and taken mid-command: nothing written, nothing counted.
    let set = c[3].set_render_pipeline_state(&unsupported);
    let refused = problem(
        "set_render_pipeline_state",
        IndirectCommandBufferProblem::PipelineWithoutSupport,
    );
    model.unwritten(&icb, set, Some(refused), copy);
    let set = c[3].set_vertex_buffer(Some(&foreign), 0, 0);
    let foreign_object = invalid("set_vertex_buffer", Problem::ForeignObject);
    model.unwritten(&icb, set, Some(foreign_object), copy);
    model.unwritten(&icb, c[3].set_vertex_buffer(Some(&a), 0, 2), None, copy);
    let set = c[3].draw_indexed_primitives(
        PrimitiveType::Triangle,
        64,
        IndexType::UInt32,
        &indices,
        0,
        1,
        0,
        0,
    );
    let past = DrawProblem::IndexBufferRange {
        offset: 0,
        count: 64,
        length: 64,
    };
    let past = invalid("draw_indexed_primitives", Problem::Draw(past));
    model.unwritten(&icb, set, Some(past), copy);
    let set = c[3].draw_patches(3, 0, 1, None, 0, 1, 0, &a, 0, 0);
    let patches = problem(
        "draw_patches",
        IndirectCommandBufferProblem::CommandType(IndirectCommandType::DRAW_PATCHES),
    );
    model.unwritten(&icb, set, Some(patches), copy);
    let set = c[3].set_depth_bias(1.0, 1.0, 1.0);
    let render_state = problem("set_depth_bias", IndirectCommandBufferProblem::RenderState);
    model.unwritten(&icb, set, Some(render_state), copy);
    // Then written: a nil buffer keeps its offset as the address, and a barrier set and cleared.
    let nil = bound(BufferStage::Fragment, 0, None, 8);
    model.edit(
        &icb,
        (c[3].set_fragment_buffer(None, 8, 0), 3, nil),
        None,
        copy,
    );
    let barrier = RecordField::Barrier(true);
    model.edit(&icb, (c[3].set_barrier(), 3, barrier), None, copy);
    let cleared = RecordField::Barrier(false);
    model.edit(&icb, (c[3].clear_barrier(), 3, cleared), None, copy);

    // A command's reset drops its state; its setters then add what they add again.
    let fresh = |_| CommandRecord::default();
    model.reset(&icb, c[1].reset(), (1..2, 1), fresh, true);
    model.edit(
        &icb,
        (c[1].set_vertex_buffer(Some(&b), 0, 1), 1, v1),
        None,
        copy,
    );
    model.edit(
        &icb,
        (c[1].set_render_pipeline_state(&state), 1, pipeline(id)),
        named,
        copy,
    );
    model.reset(&icb, icb.reset_with_range(0..3), (0..3, 3), fresh, true);
    model.edit(
        &icb,
        (c[0].set_vertex_buffer(Some(&a), 16, 0), 0, v0),
        None,
        copy,
    );
    let triangles = c[0].draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0);
    model.edit(&icb, (triangles, 0, draw(3)), None, copy);

    // Once the GPU may write the records, what one left there joins at the next setter's call:
    // an indexed draw of a triangle strip.
    icb.written_by_gpu(GpuWriter::Reset);
    model.gpu_write(&icb, 4, |record| {
        CommandRecord::write_field(record, RecordField::Action(indexed(4))).unwrap();
    });
    model.edit(
        &icb,
        (c[4].set_vertex_buffer(Some(&a), 16, 0), 4, v0),
        None,
        copy,
    );
    // A record that does not decode refuses every edit of it.
    model.gpu_write(&icb, 5, |record| {
        record[..4].copy_from_slice(&99u32.to_le_bytes())
    });
    model.edit(
        &icb,
        (c[5].set_vertex_buffer(Some(&a), 16, 0), 5, v0),
        None,
        copy,
    );
    let triangles = c[5].draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0);
    model.edit(&icb, (triangles, 5, draw(3)), None, copy);
    let set = c[5].set_render_pipeline_state(&state);
    model.edit(&icb, (set, 5, pipeline(id)), named, copy);
}

/// Compute commands, as `render_setters_write_as_a_whole_record_edit`'s: two states that may do
/// different things, threadgroup lengths in the record and past it, a stage-in region and an
/// imageblock size, a command's reset, which keeps its state, and a range's.
#[test]
fn compute_setters_write_as_a_whole_record_edit() {
    let scratch = Scratch::new("icb-compute-setters");
    let device = device(&scratch.0);
    let other = device_of_another(&scratch);
    let plain = compute_state(&device, EXAMPLES, "scale_and_offset", true);
    let encodes = compute_state(&device, ENCODES, "icb_replay_encode_dispatch", true);
    let unsupported = compute_state(&device, EXAMPLES, "scale_and_offset", false);
    assert_ne!(named_compute(&plain), named_compute(&encodes));
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
    descriptor.max_kernel_buffer_bind_count = 3;
    let shared = ResourceOptions::shared();
    let icb = device
        .new_indirect_command_buffer(&descriptor, u64::from(COMMANDS), shared)
        .unwrap();
    let buffers = [
        device.new_buffer(4096, shared).unwrap(),
        device.new_buffer(4096, shared).unwrap(),
        device.new_buffer(256, shared).unwrap(),
    ];
    let foreign = other.new_buffer(64, shared).unwrap();
    let (plain_id, encodes_id) = (plain.gpu_resource_id().0, encodes.gpu_resource_id().0);
    let kernel =
        |slot: usize, offset: u64| bound(BufferStage::Kernel, slot, Some(&buffers[slot]), offset);
    let dispatch = RecordField::Action(Action::DispatchThreadgroups {
        grid: [4, 1, 1],
        local_size: [64, 1, 1],
    });
    let (groups, threads) = (Size::new(4, 1, 1), Size::new(64, 1, 1));
    let mut model = Model::of(&icb);
    let c: Vec<_> = (0..u64::from(COMMANDS))
        .map(|index| icb.indirect_compute_command_at(index).unwrap())
        .collect();

    let named = Some(named_compute(&plain));
    for index in [0, 2] {
        let set = c[index as usize].set_compute_pipeline_state(&plain);
        model.edit(&icb, (set, index, pipeline(plain_id)), named, None);
        for (slot, buffer) in buffers.iter().enumerate() {
            let set = c[index as usize].set_kernel_buffer(Some(buffer), 16, slot as u32);
            model.edit(&icb, (set, index, kernel(slot, 16)), None, None);
        }
        let set = c[index as usize].concurrent_dispatch_threadgroups(groups, threads);
        model.edit(&icb, (set, index, dispatch), None, None);
    }
    let set = c[1].set_compute_pipeline_state(&encodes);
    let named_encodes = Some(named_compute(&encodes));
    model.edit(&icb, (set, 1, pipeline(encodes_id)), named_encodes, None);
    let length = RecordField::ThreadgroupMemoryLength {
        index: 0,
        length: 64,
    };
    model.edit(
        &icb,
        (c[0].set_threadgroup_memory_length(64, 0), 0, length),
        None,
        None,
    );
    // Index 40 of command 0 is length 2 of command 1; 31 lies past the record.
    let next = RecordField::ThreadgroupMemoryLength {
        index: 2,
        length: 32,
    };
    model.edit(
        &icb,
        (c[0].set_threadgroup_memory_length(32, 40), 1, next),
        None,
        None,
    );
    model.unwritten(&icb, c[0].set_threadgroup_memory_length(16, 31), None, None);
    let region = Region::new_3d(1, 2, 3, 4, 5, 6);
    let stage_in = RecordField::StageInRegion([1, 2, 3, 4, 5, 6]);
    model.edit(
        &icb,
        (c[0].set_stage_in_region(region), 0, stage_in),
        None,
        None,
    );
    let imageblock = RecordField::ImageblockSize([8, 8]);
    model.edit(
        &icb,
        (c[0].set_imageblock_size(8, 8), 0, imageblock),
        None,
        None,
    );
    model.edit(
        &icb,
        (c[0].set_barrier(), 0, RecordField::Barrier(true)),
        None,
        None,
    );
    let stride = c[0].set_kernel_buffer_with_stride(Some(&buffers[1]), 0, 12, 1);
    model.edit(&icb, (stride, 0, kernel(1, 0)), None, None);

    // Refused and taken mid-command.
    let set = c[3].set_compute_pipeline_state(&unsupported);
    let refused = problem(
        "set_compute_pipeline_state",
        IndirectCommandBufferProblem::PipelineWithoutSupport,
    );
    model.unwritten(&icb, set, Some(refused), None);
    let set = c[3].set_kernel_buffer(Some(&foreign), 0, 0);
    let foreign_object = invalid("set_kernel_buffer", Problem::ForeignObject);
    model.unwritten(&icb, set, Some(foreign_object), None);
    model.unwritten(
        &icb,
        c[3].set_kernel_buffer(Some(&buffers[0]), 0, 3),
        None,
        None,
    );
    let set = c[3].set_imageblock_size(70000, 1);
    let wide = Error::NotImplemented(RuntimeGap::IndirectImageblockSize {
        width: 70000,
        height: 1,
    });
    model.unwritten(&icb, set, Some(wide), None);

    // A command's reset keeps its state and the pipeline it runs; a range's too, counting the
    // slots past the records' cleared.
    let reset = |record: CommandRecord| CommandRecord {
        retained_pipeline: record.bindings.pipeline.or(record.retained_pipeline),
        imageblock_size: record.imageblock_size,
        ..CommandRecord::default()
    };
    model.reset(&icb, c[0].reset(), (0..1, 2), reset, false);
    model.edit(
        &icb,
        (
            c[0].set_kernel_buffer(Some(&buffers[0]), 16, 0),
            0,
            kernel(0, 16),
        ),
        None,
        None,
    );
    model.edit(
        &icb,
        (
            c[0].concurrent_dispatch_threadgroups(groups, threads),
            0,
            dispatch,
        ),
        None,
        None,
    );
    model.edit(
        &icb,
        (
            c[0].set_compute_pipeline_state(&encodes),
            0,
            pipeline(encodes_id),
        ),
        named_encodes,
        None,
    );
    let fresh = |_| CommandRecord::default();
    model.reset(&icb, icb.reset_with_range(1..3), (1..3, 3), fresh, true);
    model.edit(
        &icb,
        (
            c[2].concurrent_dispatch_threadgroups(groups, threads),
            2,
            dispatch,
        ),
        None,
        None,
    );

    // What the GPU left joins at the next call, and a record that does not decode refuses it.
    icb.written_by_gpu(GpuWriter::Reset);
    // A draw, of no command type the buffer has: what the GPU left, whatever an execution
    // makes of it.
    model.gpu_write(&icb, 4, |record| {
        let lines = Action::DrawPrimitives {
            primitive_type: 1,
            vertex_start: 0,
            vertex_count: 2,
            instance_count: 1,
            base_instance: 0,
        };
        CommandRecord::write_field(record, RecordField::Action(lines)).unwrap();
    });
    model.edit(
        &icb,
        (
            c[4].set_kernel_buffer(Some(&buffers[0]), 16, 0),
            4,
            kernel(0, 16),
        ),
        None,
        None,
    );
    model.gpu_write(&icb, 5, |record| {
        record[4..8].copy_from_slice(&7u32.to_le_bytes())
    });
    model.edit(
        &icb,
        (
            c[5].set_kernel_buffer(Some(&buffers[0]), 16, 0),
            5,
            kernel(0, 16),
        ),
        None,
        None,
    );
    model.edit(
        &icb,
        (c[5].set_barrier(), 5, RecordField::Barrier(true)),
        None,
        None,
    );
}

/// A second device, whose buffers are another device's.
fn device_of_another(scratch: &Scratch) -> Device {
    let cache = scratch.0.join("other");
    std::fs::create_dir_all(&cache).unwrap();
    device(&cache)
}
