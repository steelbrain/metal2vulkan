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
    Buffer, ComputePipelineState, Device, DrawPrimitiveType, DrawProblem, Error, GpuAddress,
    IndirectBufferBinding, IndirectCommandBuffer, IndirectCommandBufferDescriptor,
    IndirectCommandBufferProblem, IndirectComputeSet, IndirectDraw, IndirectRenderSet,
    PipelineOptions, Problem, Region, RenderPipelineState, ResourceOptions, RuntimeGap, Size,
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
pub(super) const EXAMPLES: &str = "examples/runtime_examples.ll";
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
pub(super) fn render_state(device: &Device, icb: bool) -> RenderPipelineState {
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
pub(super) fn compute_state(
    device: &Device,
    path: &str,
    kernel: &str,
    icb: bool,
) -> ComputePipelineState {
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

/// A blit's copy of commands and its reset of a range count the bytes of the records the GPU
/// moves as each is encoded (`WorkCounts::indirect_record_bytes_gpu_copied`,
/// `indirect_record_bytes_gpu_reset`), whole records; an empty copy counts none.
#[test]
fn a_blit_counts_the_record_bytes_the_gpu_moves() {
    let scratch = Scratch::new("icb-blit-counts");
    let device = device(&scratch.0);
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    let shared = ResourceOptions::shared();
    let icb = device
        .new_indirect_command_buffer(&descriptor, 4, shared)
        .unwrap();
    let copy = device
        .new_indirect_command_buffer(&descriptor, 4, shared)
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let before = device.work_counts();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_indirect_command_buffer(&icb, 0..3, &copy, 1)
        .unwrap();
    blit.copy_indirect_command_buffer(&icb, 2..2, &copy, 0)
        .unwrap();
    blit.reset_commands_in_buffer(&icb, 1..3).unwrap();
    blit.end_encoding().unwrap();
    let after = device.work_counts();
    let record = COMMAND_BYTES as u64;
    assert_eq!(
        (
            after.indirect_record_bytes_gpu_copied - before.indirect_record_bytes_gpu_copied,
            after.indirect_record_bytes_gpu_reset - before.indirect_record_bytes_gpu_reset,
        ),
        (3 * record, 2 * record),
        "the records' bytes copied and reset"
    );
}

/// A blit's reset of commands counts its dispatch's GPU time
/// (`WorkCounts::indirect_record_reset_gpu_nanos`) once its command buffer completed and let go
/// of what it retained, where the device times it (`rate_mapped_pass::PieceTimes::timed_on`:
/// lavapipe does); a command buffer of a copy of commands alone, which dispatches no reset,
/// counts none.
#[test]
fn a_blit_s_reset_counts_its_dispatch_s_gpu_time() {
    let scratch = Scratch::new("icb-reset-time");
    let device = device(&scratch.0);
    let timed = crate::rate_mapped_pass::PieceTimes::timed_on(&device);
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    let shared = ResourceOptions::shared();
    let icb = device
        .new_indirect_command_buffer(&descriptor, 4, shared)
        .unwrap();
    let copy = device
        .new_indirect_command_buffer(&descriptor, 4, shared)
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let nanos = || device.work_counts().indirect_record_reset_gpu_nanos;
    // One command buffer of the blit `encode` encodes, completed.
    let run = |encode: &dyn Fn(&mut crate::BlitCommandEncoder)| {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        encode(&mut blit);
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed().unwrap();
    };
    let before = nanos();
    run(&|blit| {
        blit.copy_indirect_command_buffer(&icb, 0..2, &copy, 0)
            .unwrap()
    });
    let copied = nanos() - before;
    let before = nanos();
    run(&|blit| blit.reset_commands_in_buffer(&icb, 0..4).unwrap());
    // The time is read as the command buffer lets go of what it retained, once it completed.
    let began = std::time::Instant::now();
    while timed && nanos() == before && began.elapsed() < std::time::Duration::from_secs(10) {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let reset = nanos() - before;
    eprintln!("ICB-RESET-TIME timed {timed}: copy alone {copied} ns, reset {reset} ns");
    assert_eq!(
        copied, 0,
        "a command buffer with no reset counts no reset time"
    );
    match timed {
        true => assert!(reset > 0, "the reset's dispatch counted no GPU time"),
        false => assert_eq!(reset, 0, "a device that times no reset counts none"),
    }
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

/// `set_each`, and `set_render_command` and `set_compute_command` with no command's handle, set a
/// command as its setters one call at a time do: the same records and the same CPU writes
/// counted, a render and a compute buffer alike; the first refused stops the rest, answered with
/// its place and the setter's error, those before it made and those after it not taken. The
/// buffer's calls answer the command's own refusals (an index past the size, a compute command of
/// a render buffer) at place 0.
#[test]
fn set_each_sets_as_the_setters_one_at_a_time() {
    let scratch = Scratch::new("icb-set-each");
    let device = device(&scratch.0);
    let state = render_state(&device, true);
    let unsupported = render_state(&device, false);
    let shared = ResourceOptions::shared();
    let (a, b) = (
        device.new_buffer(4096, shared).unwrap(),
        device.new_buffer(4096, shared).unwrap(),
    );
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.max_vertex_buffer_bind_count = 2;
    descriptor.max_fragment_buffer_bind_count = 1;
    let made = || {
        device
            .new_indirect_command_buffer(&descriptor, u64::from(COMMANDS), shared)
            .unwrap()
    };
    let (one_by_one, each, whole) = (made(), made(), made());
    let triangle = DrawPrimitiveType::from_raw(3).unwrap();
    for index in 0..u64::from(COMMANDS) {
        let c = one_by_one.indirect_render_command_at(index).unwrap();
        c.set_render_pipeline_state(&state).unwrap();
        c.set_vertex_buffer(Some(&a), index * 16, 0).unwrap();
        c.set_vertex_buffer(Some(&b), 0, 1).unwrap();
        c.set_fragment_buffer(Some(&a), 256, 0).unwrap();
        c.draw_primitives(triangle, 0, 3, 1, index).unwrap();
        let sets = [
            IndirectRenderSet::SetRenderPipelineState(&state),
            IndirectRenderSet::SetVertexBuffer {
                buffer: Some(&a),
                offset: index * 16,
                index: 0,
            },
            IndirectRenderSet::SetVertexBuffer {
                buffer: Some(&b),
                offset: 0,
                index: 1,
            },
            IndirectRenderSet::SetFragmentBuffer {
                buffer: Some(&a),
                offset: 256,
                index: 0,
            },
            IndirectRenderSet::DrawPrimitives {
                primitive_type: triangle,
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: index,
            },
        ];
        let c = each.indirect_render_command_at(index).unwrap();
        assert_eq!(c.set_each(sets), Ok(()), "command {index}");
        assert_eq!(whole.set_render_command(index, sets), Ok(()));
    }
    assert!(storage(&one_by_one) == storage(&each), "the records");
    assert!(
        storage(&one_by_one) == storage(&whole),
        "the buffer's records"
    );
    assert_eq!(
        one_by_one.cpu_writes(),
        each.cpu_writes(),
        "the writes counted"
    );
    assert_eq!(one_by_one.cpu_writes(), whole.cpu_writes());
    let named = |icb: &IndirectCommandBuffer| {
        (0..u64::from(COMMANDS))
            .map(|index| {
                let c = icb.indirect_render_command_at(index).unwrap();
                c.pipeline_state().unwrap().map(|s| s.gpu_resource_id())
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(named(&one_by_one), named(&each), "the states set");
    assert_eq!(named(&one_by_one), named(&whole));

    // The command's own refusals, at place 0 before any setter, none set.
    let before = (storage(&whole), whole.cpu_writes());
    let past = u64::from(COMMANDS);
    let refused = whole.indirect_render_command_at(past).err().unwrap();
    let draw = [IndirectRenderSet::SetBarrier];
    assert_eq!(whole.set_render_command(past, draw), Err((0, refused)));
    let refused = problem(
        "set_compute_command",
        IndirectCommandBufferProblem::CommandKind,
    );
    assert_eq!(whole.set_compute_command(0, []), Err((0, refused)));
    assert!(
        (storage(&whole), whole.cpu_writes()) == before,
        "nothing set"
    );

    // A refusal mid-way: the buffer before it set, the draw after it not taken.
    let fresh = made();
    let c = fresh.indirect_render_command_at(0).unwrap();
    let before = fresh.cpu_writes();
    let sets = [
        IndirectRenderSet::SetVertexBuffer {
            buffer: Some(&a),
            offset: 32,
            index: 0,
        },
        IndirectRenderSet::SetRenderPipelineState(&unsupported),
        IndirectRenderSet::DrawPrimitives {
            primitive_type: triangle,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 0,
        },
    ];
    let refused = problem(
        "set_render_pipeline_state",
        IndirectCommandBufferProblem::PipelineWithoutSupport,
    );
    assert_eq!(c.set_each(sets), Err((1, refused)));
    let set = c.vertex_buffer(0).unwrap().map(|binding| binding.address);
    assert_eq!(
        set,
        Some(GpuAddress(a.gpu_address().0 + 32)),
        "the buffer before"
    );
    assert_eq!(c.draw().unwrap(), None, "the draw after");
    assert_eq!(
        c.pipeline_state().unwrap().map(|s| s.gpu_resource_id()),
        None
    );
    assert_eq!(fresh.cpu_writes(), before + 1, "the one write made");

    // A compute command, and a refusal first: nothing made.
    let kernel = compute_state(&device, EXAMPLES, "scale_and_offset", true);
    let plain = compute_state(&device, EXAMPLES, "scale_and_offset", false);
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
    descriptor.max_kernel_buffer_bind_count = 2;
    let made = || {
        device
            .new_indirect_command_buffer(&descriptor, 2, shared)
            .unwrap()
    };
    let (one_by_one, each, whole) = (made(), made(), made());
    let (groups, threads) = (Size::new(4, 1, 1), Size::new(64, 1, 1));
    let c = one_by_one.indirect_compute_command_at(1).unwrap();
    c.set_compute_pipeline_state(&kernel).unwrap();
    c.set_kernel_buffer(Some(&a), 48, 0).unwrap();
    c.set_kernel_buffer(Some(&b), 0, 1).unwrap();
    c.concurrent_dispatch_threadgroups(groups, threads).unwrap();
    let c = each.indirect_compute_command_at(1).unwrap();
    let sets = [
        IndirectComputeSet::SetComputePipelineState(&kernel),
        IndirectComputeSet::SetKernelBuffer {
            buffer: Some(&a),
            offset: 48,
            index: 0,
        },
        IndirectComputeSet::SetKernelBuffer {
            buffer: Some(&b),
            offset: 0,
            index: 1,
        },
        IndirectComputeSet::ConcurrentDispatchThreadgroups {
            threadgroups: groups,
            threads_per_threadgroup: threads,
        },
    ];
    assert_eq!(c.set_each(sets), Ok(()));
    assert_eq!(whole.set_compute_command(1, sets), Ok(()));
    assert!(
        storage(&one_by_one) == storage(&each),
        "the compute records"
    );
    assert!(storage(&one_by_one) == storage(&whole));
    assert_eq!(one_by_one.cpu_writes(), each.cpu_writes());
    assert_eq!(one_by_one.cpu_writes(), whole.cpu_writes());
    let refused = whole.indirect_compute_command_at(2).err().unwrap();
    assert_eq!(whole.set_compute_command(2, sets), Err((0, refused)));
    let c = each.indirect_compute_command_at(0).unwrap();
    let before = (storage(&each), each.cpu_writes());
    let sets = [
        IndirectComputeSet::SetComputePipelineState(&plain),
        IndirectComputeSet::SetKernelBuffer {
            buffer: Some(&a),
            offset: 0,
            index: 0,
        },
    ];
    let refused = problem(
        "set_compute_pipeline_state",
        IndirectCommandBufferProblem::PipelineWithoutSupport,
    );
    assert_eq!(c.set_each(sets), Err((0, refused)));
    assert!(
        (storage(&each), each.cpu_writes()) == before,
        "nothing made"
    );
}

/// Setters on two threads at once: each thread's commands read back as it set them, the writes
/// all counted, where the threads set different commands of one buffer (one by its commands'
/// setters, the other by `set_render_command`); and where one thread sets
/// a command's pipeline state over and over while the other sets its buffers, each reads back its
/// own field as it set it after every set, and the command reads back the last state set, its
/// record naming that state's ID and identifier, and the last buffers set: no field one thread
/// wrote is lost to the other's write.
#[test]
fn setters_on_two_threads_keep_each_other_s_fields() {
    const SETS: u64 = 2000;
    let scratch = Scratch::new("icb-two-threads");
    let device = device(&scratch.0);
    let states = [render_state(&device, true), render_state(&device, true)];
    assert_ne!(states[0].gpu_resource_id(), states[1].gpu_resource_id());
    let shared = ResourceOptions::shared();
    let (a, b) = (
        device.new_buffer(1 << 20, shared).unwrap(),
        device.new_buffer(1 << 20, shared).unwrap(),
    );
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.max_vertex_buffer_bind_count = 2;
    descriptor.max_fragment_buffer_bind_count = 1;
    let size = 64;
    let icb = device
        .new_indirect_command_buffer(&descriptor, size, shared)
        .unwrap();
    let triangle = DrawPrimitiveType::from_raw(3).unwrap();
    let before = icb.cpu_writes();
    // Different commands: thread `t` sets the commands `index % 2 == t`, each over and over,
    // the last round's values its own.
    let slot = |t: usize, index: u64, round: u64| {
        if t == 1 {
            let sets = [
                IndirectRenderSet::SetRenderPipelineState(&states[t]),
                IndirectRenderSet::SetVertexBuffer {
                    buffer: Some(&a),
                    offset: (index + round) * 16,
                    index: 0,
                },
                IndirectRenderSet::SetVertexBuffer {
                    buffer: Some(&b),
                    offset: index * 32,
                    index: 1,
                },
                IndirectRenderSet::SetFragmentBuffer {
                    buffer: Some(&b),
                    offset: round * 16,
                    index: 0,
                },
                IndirectRenderSet::DrawPrimitives {
                    primitive_type: triangle,
                    vertex_start: index,
                    vertex_count: 3,
                    instance_count: 1,
                    base_instance: round,
                },
            ];
            return icb.set_render_command(index, sets).unwrap();
        }
        let c = icb.indirect_render_command_at(index).unwrap();
        c.set_render_pipeline_state(&states[t]).unwrap();
        c.set_vertex_buffer(Some(&a), (index + round) * 16, 0)
            .unwrap();
        c.set_vertex_buffer(Some(&b), index * 32, 1).unwrap();
        c.set_fragment_buffer(Some(&b), round * 16, 0).unwrap();
        c.draw_primitives(triangle, index, 3, 1, round).unwrap();
    };
    let rounds = SETS / size;
    std::thread::scope(|scope| {
        for t in 0..2 {
            let slot = &slot;
            scope.spawn(move || {
                for round in 0..rounds {
                    for index in (t as u64..size).step_by(2) {
                        slot(t, index, round);
                    }
                }
            });
        }
    });
    let last = rounds - 1;
    for index in 0..size {
        let c = icb.indirect_render_command_at(index).unwrap();
        let t = (index % 2) as usize;
        let named = c.pipeline_state().unwrap().map(|s| s.gpu_resource_id());
        assert_eq!(named, Some(states[t].gpu_resource_id()), "command {index}");
        let at = |buffer: &Buffer, offset: u64| Some(GpuAddress(buffer.gpu_address().0 + offset));
        let address = |binding: Option<IndirectBufferBinding>| binding.map(|b| b.address);
        assert_eq!(
            address(c.vertex_buffer(0).unwrap()),
            at(&a, (index + last) * 16)
        );
        assert_eq!(address(c.vertex_buffer(1).unwrap()), at(&b, index * 32));
        assert_eq!(address(c.fragment_buffer(0).unwrap()), at(&b, last * 16));
        let drawn = c.draw().unwrap();
        let expected = IndirectDraw::Primitives {
            primitive_type: triangle,
            vertex_start: index,
            vertex_count: 3,
            instance_count: 1,
            base_instance: last,
        };
        assert_eq!(drawn, Some(expected), "command {index}");
    }
    assert_eq!(
        icb.cpu_writes() - before,
        5 * rounds * size,
        "every write counted"
    );

    // One command: its pipeline state on one thread, its buffers on the other, each thread
    // reading back its own field after every set, which the other's writes must leave as set.
    let c = icb.indirect_render_command_at(7).unwrap();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for set in 0..SETS {
                let state = &states[(set % 2) as usize];
                c.set_render_pipeline_state(state).unwrap();
                let id = c.pipeline_id().unwrap();
                assert_eq!(id, Some(state.gpu_resource_id()), "set {set}: the state");
            }
        });
        scope.spawn(|| {
            for set in 0..SETS {
                c.set_vertex_buffer(Some(&b), set * 16, 0).unwrap();
                c.set_fragment_buffer(Some(&a), set * 32, 0).unwrap();
                let address = c.vertex_buffer(0).unwrap().map(|b| b.address);
                let expected = GpuAddress(b.gpu_address().0 + set * 16);
                assert_eq!(address, Some(expected), "set {set}: the vertex buffer");
            }
        });
    });
    let state = &states[((SETS - 1) % 2) as usize];
    let named = c.pipeline_state().unwrap().map(|s| s.gpu_resource_id());
    assert_eq!(named, Some(state.gpu_resource_id()), "the last state set");
    assert_eq!(c.pipeline_id().unwrap(), Some(state.gpu_resource_id()));
    assert_eq!(
        c.pipeline_unique_identifier().unwrap(),
        Some(state.unique_identifier()),
        "the record names the state set"
    );
    let address = c.vertex_buffer(0).unwrap().map(|b| b.address);
    assert_eq!(
        address,
        Some(GpuAddress(b.gpu_address().0 + (SETS - 1) * 16))
    );
    let address = c.fragment_buffer(0).unwrap().map(|b| b.address);
    assert_eq!(
        address,
        Some(GpuAddress(a.gpu_address().0 + (SETS - 1) * 32))
    );
}

/// A setter's write takes the buffer's lock only where it sets a pipeline state the command does
/// not name yet or adds to what the records may hold, on a buffer the GPU may not have written: a
/// buffer slot, a draw or a barrier the CPU's edits added already, a barrier cleared, and the
/// state the command names already go without it, counted apart (`writes_apart`), a call of
/// `set_render_command` counting its writes so once; on a buffer the GPU may write, every write
/// takes it.
#[test]
fn setters_take_the_lock_only_to_set_a_state_or_add_to_the_records() {
    let scratch = Scratch::new("icb-unlocked-writes");
    let device = device(&scratch.0);
    let state = render_state(&device, true);
    let shared = ResourceOptions::shared();
    let buffer = device.new_buffer(4096, shared).unwrap();
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.max_vertex_buffer_bind_count = 1;
    let icb = device
        .new_indirect_command_buffer(&descriptor, u64::from(COMMANDS), shared)
        .unwrap();
    let c: Vec<_> = (0..u64::from(COMMANDS))
        .map(|index| icb.indirect_render_command_at(index).unwrap())
        .collect();
    let (triangle, point) = (PrimitiveType::Triangle, PrimitiveType::Point);
    let mut at = icb.writes_apart();
    // A call's writes, and how many went under the lock and how many without it.
    let mut went = |what: &str, call: Result<(), Error>, locked: u64, unlocked: u64| {
        call.unwrap();
        let now = icb.writes_apart();
        assert_eq!(now, (at.0 + locked, at.1 + unlocked), "{what}");
        at = now;
    };
    went("a state", c[0].set_render_pipeline_state(&state), 1, 0);
    went(
        "a buffer",
        c[0].set_vertex_buffer(Some(&buffer), 0, 0),
        0,
        1,
    );
    went(
        "a first draw",
        c[0].draw_primitives(triangle, 0, 3, 1, 0),
        1,
        0,
    );
    went(
        "the draw again",
        c[1].draw_primitives(triangle, 0, 3, 1, 0),
        0,
        1,
    );
    went(
        "another type",
        c[1].draw_primitives(point, 0, 3, 1, 0),
        1,
        0,
    );
    went("a first barrier", c[1].set_barrier(), 1, 0);
    went("a barrier again", c[2].set_barrier(), 0, 1);
    went("a barrier cleared", c[2].clear_barrier(), 0, 1);
    let other = render_state(&device, true);
    let (one, other) = (&state, &other);
    let states = [
        ("the state, another command", one, 1, 0),
        ("a state named already", one, 0, 1),
        ("another state", other, 1, 0),
        ("that state again", other, 0, 1),
    ];
    for (what, state, locked, unlocked) in states {
        went(
            what,
            c[1].set_render_pipeline_state(state),
            locked,
            unlocked,
        );
    }
    let sets = [
        IndirectRenderSet::SetVertexBuffer {
            buffer: Some(&buffer),
            offset: 64,
            index: 0,
        },
        IndirectRenderSet::DrawPrimitives {
            primitive_type: DrawPrimitiveType::from_raw(3).unwrap(),
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 0,
        },
    ];
    let batch = icb.set_render_command(3, sets).map_err(|(_, error)| error);
    went("a batch", batch, 0, 2);
    icb.written_by_gpu(GpuWriter::Reset);
    went(
        "GPU-written, a draw",
        c[3].draw_primitives(triangle, 0, 3, 1, 0),
        1,
        0,
    );
    went(
        "GPU-written, a buffer",
        c[3].set_vertex_buffer(Some(&buffer), 0, 0),
        1,
        0,
    );
    let named = c[1].set_render_pipeline_state(other);
    went("GPU-written, a state named", named, 1, 0);
    let named = c[1].pipeline_state().unwrap().map(|s| s.gpu_resource_id());
    assert_eq!(named, Some(other.gpu_resource_id()), "the state set last");
}
