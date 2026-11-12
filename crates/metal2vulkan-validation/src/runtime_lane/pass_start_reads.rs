//! A draw's vertex stage reading bytes a fragment stage of its render pass wrote
//! (`runtime/render/pass_start_reads.rs`) on the device the cfg selects, under the validation
//! layer.
//!
//! a probe (an M3 Max, and an M4; plain and validated, 20 runs a case) drew, in one render pass, a
//! draw whose fragments write a buffer and then draws whose vertex function reads the written
//! bytes: the vertex stage read the bytes from before the pass in every run, and plain Metal
//! ignored a memory barrier from the fragment stage to the vertex stage
//! (Metal's validation stops it). Here the owned fixture `rt6_attachmentless`'s `rt6_mark_sample`
//! ORs bit 0 into word 0 of a buffer over the probe pass of `render_state`, whose `rt6_full` then
//! draws at the depth word 0 holds: 0.5 (`0x3f00_0000`) from before the pass, where the written
//! word is `0x3f00_0001`.
//!
//! a probe (the M3 Max, and an M4) found the same of an indirect command buffer's commands, behind
//! their barrier or not, and the other order too: an earlier draw's fragments read what a later
//! draw's vertex function wrote, and their write outlasts the later one's. A command's vertex stage
//! reads the copy taken before the pass here; a vertex stage's write of bytes a fragment stage of
//! its pass read or wrote is announced (`fragments-before-geometry`).

use super::buffer_with;
use super::degrade_notices::{
    assert_made_once, forced, made_render_kinds, shared_in_system_memory,
};
use super::render_state::{
    quiet, validated, validated_checking_synchronization, Pipe, Probe, PASS, WHITE,
};
use super::stage_races::layer_reported_nothing_but_announced_stage_races;
use super::{read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{
    BarrierScope, CompareFunction, IndirectCommandType, PixelFormat, PrimitiveType, RenderStages,
    ResourceUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, Degradation, IndirectCommandBufferDescriptor, LoadAction, PipelineOptions,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions, StoreAction,
};
use std::collections::BTreeSet;
use std::time::Instant;

/// The kinds `pass_start_reads` takes: a vertex-stage read in place of bytes a fragment stage of
/// its pass wrote.
pub(super) const IN_PLACE_VERTEX_READS: [Degradation; 1] = [Degradation::InPlaceVertexReads];

/// The kind `pass_start_reads` takes for a vertex stage's write of bytes a fragment stage of its
/// pass wrote.
pub(super) const FRAGMENTS_BEFORE_GEOMETRY: [Degradation; 1] =
    [Degradation::FragmentsBeforeGeometry];

/// The depth word before the pass: 0.5.
const OLD: u32 = 0x3f00_0000;

/// The depth word the fragments leave: 0.5 with bit 0 set.
const WRITTEN: u32 = 0x3f00_0001;

/// A pipeline of the probe pass's formats (an `RGBA8Unorm` target, a `Depth32Float` depth) from
/// `fixture`'s `vertex` and `fragment` functions; one that rasterizes nothing without a fragment.
/// Made with `supportIndirectCommandBuffers` where `indirect`.
fn pipeline(
    probe: &Probe,
    fixture: &str,
    vertex: &str,
    fragment: Option<&str>,
    indirect: bool,
) -> RenderPipelineState {
    let descriptor = descriptor(probe, fixture, vertex, fragment, indirect);
    probe
        .device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The descriptor of [`pipeline`]'s state.
fn descriptor(
    probe: &Probe,
    fixture: &str,
    vertex: &str,
    fragment: Option<&str>,
    indirect: bool,
) -> RenderPipelineDescriptor {
    let bitcode = llvm_assemble(&engine_dir().join(format!("{fixture}.ll"))).unwrap();
    let library = probe.device.new_library_with_air(&bitcode).unwrap();
    let function = |name: &str| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(function(vertex)));
    match fragment {
        Some(name) => descriptor.fragment = Some(StageSource::Function(function(name))),
        None => descriptor.rasterization_enabled = false,
    }
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.depth_format = Some(PixelFormat::Depth32Float);
    descriptor.support_indirect_command_buffers = indirect;
    descriptor
}

/// The kinds a test's steps force, where the device has not announced them: the makings of the
/// marker's pipeline (made for indirect commands where `indirect`), of the writer's where
/// `writer`, and of the probe's `pipes`; the pass's split where `split` (a memory barrier from
/// the fragment stage to the vertex stage, which Vulkan orders between render pass instances
/// alone); and the `Shared` buffers' system memory.
fn forced_kinds(
    probe: &Probe,
    (indirect, writer): (bool, bool),
    pipes: &[Pipe],
    split: bool,
) -> BTreeSet<Degradation> {
    let device = &probe.device;
    let mut kinds = forced(&[
        (Degradation::RenderPassSplit, split),
        (
            Degradation::SharedInSystemMemory,
            shared_in_system_memory(device),
        ),
    ]);
    let marker = descriptor(
        probe,
        "rt6_attachmentless",
        "rt6_cover",
        Some("rt6_mark_sample"),
        indirect,
    );
    kinds.extend(made_render_kinds(device, &marker));
    if writer {
        let writer = descriptor(
            probe,
            "icb_render_replay_template",
            "icb_render_replay_template",
            None,
            false,
        );
        kinds.extend(made_render_kinds(device, &writer));
    }
    for &pipe in pipes {
        kinds.extend(probe.made(pipe));
    }
    kinds
}

/// The words of a buffer of `length` bytes: 0.5 and a slope of 0 for `rt6_full`, zero after.
fn depth_words(probe: &Probe, length: usize) -> Buffer {
    let mut bytes = vec![0u8; length];
    bytes[..4].copy_from_slice(&OLD.to_le_bytes());
    buffer_with(&probe.device, &bytes).unwrap()
}

/// The marker's draw: every fragment of the pass ORs bit 0 into word 0 of `words`.
fn mark(e: &mut RenderCommandEncoder, marker: &RenderPipelineState, words: &Buffer) {
    e.set_render_pipeline_state(marker).unwrap();
    e.set_fragment_buffer(Some(words), 0, 0).unwrap();
    e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
}

/// The reader's draw: `rt6_full` at the depth `words` holds at `offset`, written under `Always`.
fn read(probe: &Probe, e: &mut RenderCommandEncoder, words: &Buffer, offset: u64) {
    e.set_render_pipeline_state(&probe.pipeline(WHITE.depth(PixelFormat::Depth32Float)))
        .unwrap();
    e.set_depth_stencil_state(&probe.state(CompareFunction::Always, true, None))
        .unwrap();
    e.set_vertex_buffer(Some(words), offset, 0).unwrap();
    e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
}

/// The depths a pass stored, as words.
fn depths(stored: &[u8]) -> Vec<u32> {
    stored
        .chunks(4)
        .map(|t| u32::from_le_bytes([t[0], t[1], t[2], t[3]]))
        .collect()
}

/// What a case encodes between the marker and the last reader.
#[derive(Clone, Copy)]
enum Between {
    Nothing,
    /// A reader at offset 0; the last reads offset 16, in the same copy.
    Read,
    /// A memory barrier over buffers from the fragment stage to the vertex stage.
    Barrier,
}

/// A vertex stage reads the bytes from before its pass's fragment writes, as probe 6 measured:
/// after the marker, after a draw between, at another offset of the same copy, and across a
/// memory barrier from the fragment stage to the vertex stage, which splits the pass; the pass
/// after reads the written bytes. The copy is announced once (`unordered-draw-bytes`).
#[test]
fn a_vertex_stage_reads_the_bytes_from_before_its_pass_s_fragment_writes() {
    let (messages, _cache, probe) = validated();
    let marker = pipeline(
        &probe,
        "rt6_attachmentless",
        "rt6_cover",
        Some("rt6_mark_sample"),
        false,
    );
    let pass = PASS.depth(PixelFormat::Depth32Float, 1.0);
    for (name, between) in [
        ("after_the_marker", Between::Nothing),
        ("at_another_offset_of_the_copy", Between::Read),
        ("across_a_barrier", Between::Barrier),
    ] {
        // Words 0 and 4 hold 0.5: a bind at offset 16 reads the copy from offset 0 there.
        let words = depth_words(&probe, 64);
        let mut bytes = read_buffer(&words, 0, 64).unwrap();
        bytes[16..20].copy_from_slice(&OLD.to_le_bytes());
        let words = buffer_with(&probe.device, &bytes).unwrap();
        let stored = probe.run(pass, |p, e, _| {
            mark(e, &marker, &words);
            let offset = match between {
                Between::Nothing => 0,
                Between::Read => {
                    read(p, e, &words, 0);
                    16
                }
                Between::Barrier => {
                    e.memory_barrier_with_scope(
                        BarrierScope::BUFFERS,
                        RenderStages::FRAGMENT,
                        RenderStages::VERTEX,
                    );
                    0
                }
            };
            read(p, e, &words, offset);
        });
        assert_eq!(
            depths(&stored.depth.unwrap()),
            vec![OLD; 16],
            "{name}: the depth from before the pass"
        );
        let written = read_buffer(&words, 0, 4).unwrap();
        assert_eq!(written, WRITTEN.to_le_bytes(), "{name}: the marker's write");
        // The next pass reads the written word.
        let stored = probe.run(pass, |p, e, _| read(p, e, &words, 0));
        assert_eq!(
            depths(&stored.depth.unwrap()),
            vec![WRITTEN; 16],
            "{name}: the next pass reads the write"
        );
    }
    let reader = [WHITE.depth(PixelFormat::Depth32Float)];
    let first = forced_kinds(&probe, (false, false), &reader, true);
    assert_made_once(&messages, &[Degradation::UnorderedDrawBytes], &first);
    quiet(probe, &messages);
}

/// A later draw's fragments read and write what an earlier draw's vertex stage wrote in the same
/// render pass instance after the write, as Metal's run every draw's vertex functions before any
/// draw's fragments: the writer (the fixture `icb_render_replay_template`, rasterizing nothing)
/// stores `WORD` in word 0, then the marker ORs bit 0 into it, which leaves `WORD | 1`; the
/// validation layer, checking synchronization, reports no hazard between the write and the
/// fragments' access, which records after a memory dependency from the write. Twice: in the
/// second pass both pipelines are made, so the marker's draw records into the writer's render
/// pass instance (in the first, a draw whose pipeline is being made may record into an instance
/// of its own, `render/inline_first.rs`).
#[test]
fn a_fragment_stage_touches_what_an_earlier_vertex_stage_of_its_pass_wrote_after_the_write() {
    const WORD: u32 = 0x100;
    let (messages, _cache, probe) = validated_checking_synchronization();
    let marker = pipeline(
        &probe,
        "rt6_attachmentless",
        "rt6_cover",
        Some("rt6_mark_sample"),
        false,
    );
    let writer = pipeline(
        &probe,
        "icb_render_replay_template",
        "icb_render_replay_template",
        None,
        false,
    );
    // The writer's word: `params[0] * (vid + 1) + params[1]`.
    let params = buffer_with(
        &probe.device,
        &[0u32, WORD]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .unwrap();
    for pass in 0..2 {
        let words = buffer_with(&probe.device, &[0u8; 64]).unwrap();
        probe.run(PASS.depth(PixelFormat::Depth32Float, 1.0), |_, e, _| {
            e.set_render_pipeline_state(&writer).unwrap();
            e.set_vertex_buffer(Some(&params), 0, 0).unwrap();
            e.set_vertex_buffer(Some(&words), 0, 1).unwrap();
            e.draw_primitives(PrimitiveType::Point, 0, 1, 1, 0).unwrap();
            mark(e, &marker, &words);
        });
        assert_eq!(
            read_buffer(&words, 0, 4).unwrap(),
            (WORD | 1).to_le_bytes(),
            "pass {pass}"
        );
    }
    quiet(probe, &messages);
}

/// A buffer a vertex stage of the pass wrote too (the fixture `icb_render_replay_template`,
/// rasterizing nothing, writes words 0 and 1) is read in place, announced once
/// (`in-place-vertex-reads`); the depth is the device's. The layer's report of that race passes
/// as announced (`stage_races.rs`).
#[test]
fn a_vertex_stage_reads_in_place_what_another_stage_of_its_pass_wrote() {
    let (messages, _cache, probe) = validated();
    let marker = pipeline(
        &probe,
        "rt6_attachmentless",
        "rt6_cover",
        Some("rt6_mark_sample"),
        false,
    );
    let writer = pipeline(
        &probe,
        "icb_render_replay_template",
        "icb_render_replay_template",
        None,
        false,
    );
    let words = depth_words(&probe, 64);
    // The writer's words: `params[0] * (vid + 1) + params[1]`, 0.5 and a slope of 0 again.
    let params = buffer_with(
        &probe.device,
        &[0u32, OLD]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .unwrap();
    probe.run(PASS.depth(PixelFormat::Depth32Float, 1.0), |p, e, _| {
        e.set_render_pipeline_state(&writer).unwrap();
        e.set_vertex_buffer(Some(&params), 0, 0).unwrap();
        e.set_vertex_buffer(Some(&words), 0, 1).unwrap();
        e.draw_primitives(PrimitiveType::Point, 0, 1, 1, 0).unwrap();
        mark(e, &marker, &words);
        read(p, e, &words, 0);
    });
    let claim = "the reader's in-place read of the marker's write";
    layer_reported_nothing_but_announced_stage_races(&messages, claim, &[&words]);
    let reader = [WHITE.depth(PixelFormat::Depth32Float)];
    let first = forced_kinds(&probe, (false, true), &reader, false);
    assert_made_once(&messages, &IN_PLACE_VERTEX_READS, &first);
    quiet(probe, &messages);
}

/// A command of an indirect command buffer reads, from its vertex stage, the bytes from before
/// its pass's fragment writes, as Metal's commands read them behind their barrier or not (the M3
/// Max, and an M4): after the encoder's marker before the call, and after a command's, without and
/// with the reader's barrier, which splits the pass. The copy, taken before the pass into the
/// recording that replays the commands, is announced once (`unordered-draw-bytes`).
#[test]
fn a_command_s_vertex_stage_reads_the_bytes_from_before_its_pass_s_fragment_writes() {
    let (messages, _cache, probe) = validated();
    let marker = pipeline(
        &probe,
        "rt6_attachmentless",
        "rt6_cover",
        Some("rt6_mark_sample"),
        true,
    );
    let reader = probe.pipeline(WHITE.depth(PixelFormat::Depth32Float).indirect());
    let state = probe.state(CompareFunction::Always, true, None);
    let pass = PASS.depth(PixelFormat::Depth32Float, 1.0);
    for (name, command_marks, barrier) in [
        ("after_the_encoder_s_marker", false, false),
        ("after_the_encoder_s_marker_behind_the_barrier", false, true),
        ("after_a_command_s_marker", true, false),
        ("after_a_command_s_marker_behind_the_barrier", true, true),
    ] {
        let words = depth_words(&probe, 64);
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW;
        descriptor.max_vertex_buffer_bind_count = 1;
        descriptor.max_fragment_buffer_bind_count = 1;
        let count = 1 + u64::from(command_marks);
        let icb = probe
            .device
            .new_indirect_command_buffer(&descriptor, count, ResourceOptions::shared())
            .unwrap();
        if command_marks {
            let first = icb.indirect_render_command_at(0).unwrap();
            first.set_render_pipeline_state(&marker).unwrap();
            first.set_fragment_buffer(Some(&words), 0, 0).unwrap();
            first
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
        let last = icb.indirect_render_command_at(count - 1).unwrap();
        if barrier {
            last.set_barrier().unwrap();
        }
        last.set_render_pipeline_state(&reader).unwrap();
        last.set_vertex_buffer(Some(&words), 0, 0).unwrap();
        last.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        let stored = probe.run(pass, |_, e, _| {
            if !command_marks {
                mark(e, &marker, &words);
            }
            e.set_depth_stencil_state(&state).unwrap();
            e.use_resource(
                words.as_resource(),
                ResourceUsage::READ | ResourceUsage::WRITE,
                RenderStages::VERTEX.union(RenderStages::FRAGMENT),
            )
            .unwrap();
            e.execute_commands_in_buffer(&icb, 0..count).unwrap();
        });
        assert_eq!(
            depths(&stored.depth.unwrap()),
            vec![OLD; 16],
            "{name}: the depth from before the pass"
        );
        let written = read_buffer(&words, 0, 4).unwrap();
        assert_eq!(written, WRITTEN.to_le_bytes(), "{name}: the marker's write");
    }
    let reader = [WHITE.depth(PixelFormat::Depth32Float).indirect()];
    let first = forced_kinds(&probe, (true, false), &reader, true);
    assert_made_once(&messages, &[Degradation::UnorderedDrawBytes], &first);
    quiet(probe, &messages);
}

/// A vertex stage's write of bytes a fragment stage of its pass read and wrote before it (the
/// marker's, then the fixture `icb_render_replay_template`'s write of words 0 and 1, rasterizing
/// nothing) is announced once (`fragments-before-geometry`). Metal runs the vertex function
/// before the pass's fragments (an earlier fragment read a later vertex function's write, and its
/// own write outlasted it), so the marker ORs bit 0 into the writer's 0.5; the device's order is
/// printed. The layer's report of that race passes as announced (`stage_races.rs`).
#[test]
fn a_vertex_stage_s_write_after_a_fragment_s_access_is_announced() {
    let (messages, _cache, probe) = validated();
    let marker = pipeline(
        &probe,
        "rt6_attachmentless",
        "rt6_cover",
        Some("rt6_mark_sample"),
        false,
    );
    let writer = pipeline(
        &probe,
        "icb_render_replay_template",
        "icb_render_replay_template",
        None,
        false,
    );
    let words = depth_words(&probe, 64);
    // The writer's word 0: `params[0] * (vid + 1) + params[1]`, 0.5.
    let params = buffer_with(
        &probe.device,
        &[0u32, OLD]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .unwrap();
    probe.run(PASS.depth(PixelFormat::Depth32Float, 1.0), |_, e, _| {
        mark(e, &marker, &words);
        e.set_render_pipeline_state(&writer).unwrap();
        e.set_vertex_buffer(Some(&params), 0, 0).unwrap();
        e.set_vertex_buffer(Some(&words), 0, 1).unwrap();
        e.draw_primitives(PrimitiveType::Point, 0, 1, 1, 0).unwrap();
    });
    let claim = "the writer's write after the marker's access";
    layer_reported_nothing_but_announced_stage_races(&messages, claim, &[&words]);
    let word = u32::from_le_bytes(read_buffer(&words, 0, 4).unwrap().try_into().unwrap());
    assert!(
        word == OLD || word == WRITTEN,
        "word 0 {word:#x}, neither the writer's nor the marker's after it"
    );
    let order = match word {
        WRITTEN => "the writer's, then the marker's, as Metal's",
        _ => "the marker's, then the writer's",
    };
    eprintln!("WRITE-ORDER {}: word 0 holds {order}", probe.device.name());
    let first = forced_kinds(&probe, (false, true), &[], false);
    assert_made_once(&messages, &FRAGMENTS_BEFORE_GEOMETRY, &first);
    quiet(probe, &messages);
}

/// The cost of the copy (the coordinator's condition 3): a pass of 256 reader draws of a 16-MiB
/// buffer, after the marker (one copy of the buffer before the pass) and without it (none),
/// each the best of 5 runs from commit to completion, printed.
#[test]
fn the_pass_start_copy_costs_one_copy_a_pass() {
    let (messages, _cache, probe) = validated();
    let marker = pipeline(
        &probe,
        "rt6_attachmentless",
        "rt6_cover",
        Some("rt6_mark_sample"),
        false,
    );
    const LENGTH: usize = 16 << 20;
    let words = depth_words(&probe, LENGTH);
    let reader = probe.pipeline(WHITE.depth(PixelFormat::Depth32Float));
    let state = probe.state(CompareFunction::Always, true, None);
    let color = super::render_state::texture(&probe.device, PixelFormat::RGBA8Unorm, 4);
    let depth = super::render_state::texture(&probe.device, PixelFormat::Depth32Float, 4);
    let time = |marked: bool| {
        (0..5)
            .map(|_| {
                let mut descriptor = RenderPassDescriptor::new();
                let target = &mut descriptor.color_attachments[0];
                target.texture = Some(color.clone());
                target.load_action = LoadAction::Clear;
                target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
                target.store_action = StoreAction::Store;
                descriptor.depth_attachment.texture = Some(depth.clone());
                descriptor.depth_attachment.store_action = StoreAction::Store;
                let command_buffer = probe.queue.command_buffer().unwrap();
                let mut e = command_buffer.render_command_encoder(&descriptor).unwrap();
                if marked {
                    mark(&mut e, &marker, &words);
                }
                e.set_render_pipeline_state(&reader).unwrap();
                e.set_depth_stencil_state(&state).unwrap();
                for _ in 0..256 {
                    e.set_vertex_buffer(Some(&words), 0, 0).unwrap();
                    e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                        .unwrap();
                }
                e.end_encoding().unwrap();
                let start = Instant::now();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None);
                start.elapsed()
            })
            .min()
            .unwrap()
    };
    let (copied, none) = (time(true), time(false));
    eprintln!(
        "PASS-START-COST {}: 256 reader draws of {LENGTH} bytes, best of 5: {:?} with the marker \
         (one copy), {:?} without",
        probe.device.name(),
        copied,
        none
    );
    let _ = messages.take_notices();
    quiet(probe, &messages);
}
