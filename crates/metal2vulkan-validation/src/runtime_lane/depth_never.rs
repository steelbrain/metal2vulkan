//! Draws under a depth compare of `Never` in a pass without a depth attachment (`depth_stencil.rs`'
//! `FailingDraw`) on the device the cfg selects, under the validation layer. the Metal probes's
//! cases run on the runtime with the probes' functions (the owned fixture `cfb_depth_never`): 4x4
//! passes of one `RGBA8Unorm` attachment, cleared and stored, and where a case says, a `Stencil8`
//! attachment cleared to 0x11 and stored, under the reference 0x5a; one fullscreen triangle per
//! draw, whose functions count their invocations (fragments in word 0, vertices in word 1).
//!
//! Every answer is Metal's, which the M3 Max (macOS 26.7) and an M4 (macOS 27.0.1) measured
//! alike, with and without API validation (which stops none of the draws):
//!
//! - The draw writes no color. The fragment function runs once per covered pixel (16), a
//!   discarding one too, and not at all where it declares `[[early_fragment_tests]]`; the vertex
//!   function runs either way (3).
//! - A visibility result counts no sample (`Counting` 0, `Boolean` 0), early tests or not, beside
//!   draws that count.
//! - Over a stencil attachment, a fragment that passes the stencil test takes the face's
//!   depth-failure operation (`IncrementClamp`: 0x12, twice 0x13; `Zero`: 0), one that fails it
//!   the failure operation (`Invert`: 0xee, a comparison of `Never` or of `Less` against 0x11),
//!   one the function discards none (column 0 keeps 0x11), and under early tests the same before
//!   the function, which never runs.
//!
//! Under early tests, a face whose operation depends on the stored value (a comparison other
//! than `Always` and `Never` whose failure and depth-failure operations differ) takes Metal's
//! operation where the function writes no memory (the owned `cfb_dn_red_early`, 0xee as probe 4b's
//! case leaves, whose function Metal never runs); where the function writes memory
//! (`cfb_dn_count_early`), which the runtime would run, it refuses typed,
//! `NotImplemented(DepthNeverWithoutAttachment)` (printed).

use super::render_state::{texels, texture};
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType, StencilOperation};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandQueue, DepthStencilState, Device, Error, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, Library, LoadAction, PipelineOptions, RenderCommandEncoder,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, RuntimeGap, StoreAction,
    VisibilityResultMode,
};

/// The passes' clear color, (0.2, 0.4, 0.6, 1), as `RGBA8Unorm` stores it.
const CLEARED: [u8; 4] = [51, 102, 153, 255];
/// The fragment functions' red.
const RED: [u8; 4] = [255, 0, 0, 255];
/// The stencil attachment's clear and the reference.
const STENCIL: u8 = 0x11;
const REFERENCE: u32 = 0x5a;

/// The lane's device, queue and the fixture's library.
struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

/// One draw: its functions, depth compare, stencil faces, and how many times it is drawn.
#[derive(Clone, Copy)]
struct Draw {
    vertex: &'static str,
    fragment: &'static str,
    compare: CompareFunction,
    stencil: Option<StencilDescriptor>,
    times: u32,
}

/// The probes' default draw: the counting fragment function under `Never`, once.
const NEVER: Draw = Draw {
    vertex: "cfb_dn_full",
    fragment: "cfb_dn_count",
    compare: CompareFunction::Never,
    stencil: None,
    times: 1,
};

impl Draw {
    fn fragment(self, fragment: &'static str) -> Draw {
        Draw { fragment, ..self }
    }

    fn vertex(self, vertex: &'static str) -> Draw {
        Draw { vertex, ..self }
    }

    fn compare(self, compare: CompareFunction) -> Draw {
        Draw { compare, ..self }
    }

    fn stencil(self, face: StencilDescriptor) -> Draw {
        Draw {
            stencil: Some(face),
            ..self
        }
    }

    fn twice(self) -> Draw {
        Draw { times: 2, ..self }
    }
}

/// A stencil face of `compare`, failure operation `Invert`, pass operation `Replace`, and
/// `depth_failure`.
fn face(compare: CompareFunction, depth_failure: StencilOperation) -> StencilDescriptor {
    let mut face = StencilDescriptor::default();
    face.stencil_compare_function = compare;
    face.stencil_failure_operation = StencilOperation::Invert;
    face.depth_failure_operation = depth_failure;
    face.depth_stencil_pass_operation = StencilOperation::Replace;
    face
}

/// What a pass left: the color and stencil texels, the fragment and vertex counts, and the
/// visibility result's word.
#[derive(Debug, PartialEq)]
struct Left {
    color: Vec<u8>,
    stencil: Option<Vec<u8>>,
    fragments: u32,
    vertices: u32,
    visibility: Option<u64>,
}

impl Lane {
    fn new() -> (Messages, ScratchCache, Lane) {
        let messages = Messages::default();
        let cache = ScratchCache::new().unwrap();
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let bitcode = llvm_assemble(&engine_dir().join("cfb_depth_never.ll")).unwrap();
        let lane = Lane {
            queue: device.new_command_queue().unwrap(),
            library: device.new_library_with_air(&bitcode).unwrap(),
            device,
        };
        (messages, cache, lane)
    }

    fn pipeline(&self, draw: Draw, stencil: bool) -> RenderPipelineState {
        self.pipeline_for(draw, stencil, false)
    }

    /// `draw`'s pipeline state, made with `supportIndirectCommandBuffers` where `indirect`.
    fn pipeline_for(&self, draw: Draw, stencil: bool, indirect: bool) -> RenderPipelineState {
        let function = |name| self.library.new_function_with_name(name).unwrap();
        let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
            function(draw.vertex).function_descriptor(),
        ));
        descriptor.fragment = Some(StageSource::Function(
            function(draw.fragment).function_descriptor(),
        ));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        descriptor.stencil_format = stencil.then_some(PixelFormat::Stencil8);
        descriptor.support_indirect_command_buffers = indirect;
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    fn state(&self, draw: Draw) -> DepthStencilState {
        let mut descriptor = DepthStencilDescriptor::default();
        descriptor.depth_compare_function = draw.compare;
        if let Some(face) = draw.stencil {
            descriptor.front_face_stencil = face;
            descriptor.back_face_stencil = face;
        }
        self.device.new_depth_stencil_state(&descriptor).unwrap()
    }

    /// Encode `draw` into `e`, its counters in `counter`.
    fn draw(
        &self,
        e: &mut RenderCommandEncoder,
        draw: Draw,
        counter: &Buffer,
    ) -> Result<(), Error> {
        e.set_render_pipeline_state(&self.pipeline(draw, draw.stencil.is_some()))?;
        e.set_depth_stencil_state(&self.state(draw))?;
        e.set_stencil_reference_value(REFERENCE);
        e.set_fragment_buffer(Some(counter), 0, 0)?;
        e.set_vertex_buffer(Some(counter), 0, 1)?;
        for _ in 0..draw.times {
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)?;
        }
        Ok(())
    }

    /// A pass with a stencil attachment where `stencil`, and a visibility result buffer where
    /// `visibility`, its commands by `body` (given the counter buffer); what it left, or the
    /// first refusal `body` returns (the pass then ends without it).
    fn run(
        &self,
        stencil: bool,
        visibility: bool,
        body: impl FnOnce(&Lane, &mut RenderCommandEncoder, &Buffer) -> Result<(), Error>,
    ) -> Result<Left, Error> {
        let color = texture(&self.device, PixelFormat::RGBA8Unorm, 4);
        let stencil = stencil.then(|| texture(&self.device, PixelFormat::Stencil8, 4));
        let counter = buffer_with(&self.device, &[0; 8]).unwrap();
        let results = buffer_with(&self.device, &[0xab; 8]).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let target = &mut pass.color_attachments[0];
        target.texture = Some(color.clone());
        target.load_action = LoadAction::Clear;
        target.clear_color = ClearColor::new(0.2, 0.4, 0.6, 1.0);
        target.store_action = StoreAction::Store;
        pass.stencil_attachment.texture = stencil.clone();
        pass.stencil_attachment.load_action = LoadAction::Clear;
        pass.stencil_attachment.clear_stencil = u32::from(STENCIL);
        pass.stencil_attachment.store_action = StoreAction::Store;
        pass.visibility_result_buffer = visibility.then(|| results.clone());
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        let encoded = body(self, &mut encoder, &counter);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        encoded?;
        let words = read_buffer(&counter, 0, 8).unwrap();
        let word = |i: usize| u32::from_le_bytes(words[4 * i..4 * i + 4].try_into().unwrap());
        Ok(Left {
            color: texels(&color, 4),
            stencil: stencil.map(|t| texels(&t, 1)),
            fragments: word(0),
            vertices: word(1),
            visibility: visibility.then(|| {
                let bytes = read_buffer(&results, 0, 8).unwrap();
                u64::from_le_bytes(bytes.try_into().unwrap())
            }),
        })
    }

    /// One pass of `draw` alone, with the stencil attachment its draw names, and a visibility
    /// result in `mode` at offset 0 unless `Disabled`.
    fn one(&self, draw: Draw, mode: VisibilityResultMode) -> Result<Left, Error> {
        let visibility = mode != VisibilityResultMode::Disabled;
        self.run(draw.stencil.is_some(), visibility, |lane, e, counter| {
            if visibility {
                e.set_visibility_result_mode(mode, 0)?;
            }
            lane.draw(e, draw, counter)
        })
    }
}

/// The lane dropped, nothing the validation layer reported.
fn quiet(lane: Lane, messages: &Messages) {
    drop(lane);
    layer_reported_nothing_at_all(messages);
}

/// `color` 16 times.
fn each(color: [u8; 4]) -> Vec<u8> {
    color.repeat(16)
}

/// The result `left` holds as `mode`'s contract reads it (`visibility::by_contract`): a `Boolean`
/// word nonzero where a sample passed, the device's own value, against Metal's 1.
fn by_contract(left: &Left, mode: VisibilityResultMode) -> Option<u64> {
    let booleans: &[usize] = match mode {
        VisibilityResultMode::Boolean => &[0],
        _ => &[],
    };
    let unwritten = u64::from_le_bytes([0xab; 8]);
    left.visibility
        .map(|word| super::visibility::by_contract(vec![word], booleans, unwritten)[0])
}

/// Whether the device refused `Counting` results for want of precise occlusion queries
/// (`visibility.rs`' lane), printed.
fn counting_refused(left: &Result<Left, Error>, device: &str) -> bool {
    match left {
        Err(Error::Unsupported(u)) if u.feature == Feature::CountingVisibility => {
            eprintln!("refused: Counting visibility results on {device}: {u}");
            true
        }
        _ => false,
    }
}

/// The fragment function runs (unless it declares early tests) and writes nothing; the vertex
/// function runs (probes 4 and 4b).
#[test]
fn a_never_depth_test_without_an_attachment_writes_nothing() {
    let (messages, _cache, lane) = Lane::new();
    let off = VisibilityResultMode::Disabled;
    let cases = [
        // Probe 4's control: depth `Always` draws.
        ("always", NEVER.compare(CompareFunction::Always), RED, 16, 0),
        ("never", NEVER, CLEARED, 16, 0),
        (
            "never_early",
            NEVER.fragment("cfb_dn_count_early"),
            CLEARED,
            0,
            0,
        ),
        (
            "never_discard",
            NEVER.fragment("cfb_dn_count_discard"),
            CLEARED,
            16,
            0,
        ),
        (
            "never_vertex",
            NEVER.vertex("cfb_dn_count_vertices"),
            CLEARED,
            16,
            3,
        ),
        (
            "never_early_vertex",
            NEVER
                .fragment("cfb_dn_count_early")
                .vertex("cfb_dn_count_vertices"),
            CLEARED,
            0,
            3,
        ),
    ];
    for (name, draw, color, fragments, vertices) in cases {
        let left = lane.one(draw, off).unwrap();
        assert_eq!(left.color, each(color), "{name}");
        assert_eq!(left.fragments, fragments, "{name}");
        assert_eq!(left.vertices, vertices, "{name}");
    }
    quiet(lane, &messages);
}

/// A visibility result counts none of the draw's samples, beside draws that count (probe 4's
/// `never_counting`, `never_boolean`, `always_counting`; probe 4b's `never_early_counting`).
#[test]
fn a_never_depth_test_without_an_attachment_counts_no_sample() {
    let (messages, _cache, lane) = Lane::new();
    let device = lane.device.name().to_owned();
    let (counting, boolean) = (
        VisibilityResultMode::Counting,
        VisibilityResultMode::Boolean,
    );
    let always = NEVER.compare(CompareFunction::Always);
    let early = NEVER.fragment("cfb_dn_count_early");
    for (name, draw, mode, result) in [
        ("never_boolean", NEVER, boolean, 0),
        ("always_boolean", always, boolean, 1),
        ("never_counting", NEVER, counting, 0),
        ("always_counting", always, counting, 16),
        ("never_early_counting", early, counting, 0),
    ] {
        let left = lane.one(draw, mode);
        if counting_refused(&left, &device) {
            continue;
        }
        assert_eq!(by_contract(&left.unwrap(), mode), Some(result), "{name}");
    }
    // One offset over drawn and undrawn samples: the `Never` draw's query pauses around it.
    for (mode, result) in [(boolean, 1), (counting, 32)] {
        let left = lane.run(false, true, |lane, e, counter| {
            e.set_visibility_result_mode(mode, 0)?;
            lane.draw(e, always, counter)?;
            lane.draw(e, NEVER, counter)?;
            lane.draw(e, early, counter)?;
            lane.draw(e, always, counter)
        });
        if counting_refused(&left, &device) {
            continue;
        }
        let left = left.unwrap();
        assert_eq!(
            by_contract(&left, mode),
            Some(result),
            "{mode:?} around never"
        );
        assert_eq!(left.fragments, 48, "{mode:?} around never");
        assert_eq!(left.color, each(RED), "{mode:?} around never");
    }
    quiet(lane, &messages);
}

/// Over a stencil attachment, each fragment takes the operation Metal takes (probes 4 and 4b).
#[test]
fn a_never_depth_test_without_an_attachment_takes_the_stencils_depth_failure_operation() {
    let (messages, _cache, lane) = Lane::new();
    let device = lane.device.name().to_owned();
    use CompareFunction::{Always, Less, Never};
    use StencilOperation::{IncrementClamp, Zero};
    let ops = face(Always, IncrementClamp);
    let early = NEVER.fragment("cfb_dn_count_early");
    let pure_early = NEVER.fragment("cfb_dn_red_early");
    let discarded = |kept: u8, taken: u8| -> Vec<u8> { [kept, taken, taken, taken].repeat(4) };
    let cases = [
        ("never_stencil", NEVER.stencil(ops), vec![0x12; 16], 16),
        (
            "never_stencil_twice",
            NEVER.stencil(ops).twice(),
            vec![0x13; 16],
            32,
        ),
        (
            "always_stencil",
            NEVER.compare(Always).stencil(ops),
            vec![0x5a; 16],
            16,
        ),
        (
            "never_stencil_fails",
            NEVER.stencil(face(Never, IncrementClamp)),
            vec![0xee; 16],
            16,
        ),
        (
            "never_stencil_less",
            NEVER.stencil(face(Less, IncrementClamp)),
            vec![0xee; 16],
            16,
        ),
        (
            "never_stencil_zero",
            NEVER.stencil(face(Always, Zero)),
            vec![0; 16],
            16,
        ),
        (
            "never_stencil_discard",
            NEVER.fragment("cfb_dn_count_discard").stencil(ops),
            discarded(STENCIL, 0x12),
            16,
        ),
        ("never_stencil_early", early.stencil(ops), vec![0x12; 16], 0),
        (
            "never_early_stencil_fails",
            early.stencil(face(Never, IncrementClamp)),
            vec![0xee; 16],
            0,
        ),
        // An operation that depends on the stored value, of a function that writes no memory.
        (
            "never_early_stencil_less",
            pure_early.stencil(face(Less, IncrementClamp)),
            vec![0xee; 16],
            0,
        ),
        (
            "never_early_stencil_always",
            pure_early.stencil(ops),
            vec![0x12; 16],
            0,
        ),
    ];
    for (name, draw, stencil, fragments) in cases {
        let left = lane.one(draw, VisibilityResultMode::Disabled).unwrap();
        let color = match draw.compare {
            Always => RED,
            _ => CLEARED,
        };
        assert_eq!(left.color, each(color), "{name}");
        assert_eq!(left.stencil, Some(stencil), "{name}");
        assert_eq!(left.fragments, fragments, "{name}");
    }
    // Counted, early tests or not: the stencil as above, no sample.
    for (name, draw, stencil) in [
        ("never_stencil_counting", NEVER.stencil(ops), 0x12),
        ("never_early_stencil_counting", early.stencil(ops), 0x12),
        (
            "never_early_stencil_less_counting",
            pure_early.stencil(face(Less, IncrementClamp)),
            0xee,
        ),
    ] {
        let left = lane.one(draw, VisibilityResultMode::Counting);
        if counting_refused(&left, &device) {
            continue;
        }
        let left = left.unwrap();
        assert_eq!(left.stencil, Some(vec![stencil; 16]), "{name}");
        assert_eq!(left.visibility, Some(0), "{name}");
    }
    // Under early tests, an operation that depends on the stored value, of a function that
    // writes memory: refused typed.
    let left = lane.one(
        early.stencil(face(Less, IncrementClamp)),
        VisibilityResultMode::Disabled,
    );
    assert_eq!(
        left,
        Err(Error::NotImplemented(
            RuntimeGap::DepthNeverWithoutAttachment
        )),
        "never_early_stencil_less, a function that writes memory"
    );
    eprintln!(
        "refused: never_early_stencil_less of a function that writes memory on {device}, where \
         Metal leaves 0xee (probe 4b)"
    );
    quiet(lane, &messages);
}

/// An indirect command buffer of `count` draws of the counting functions, each with its
/// counters in `counter`, as `Lane::draw` binds them, the barrier of command `barrier` set.
fn counting_commands(
    lane: &Lane,
    pipeline: &RenderPipelineState,
    (count, barrier): (u64, Option<u64>),
    counter: &Buffer,
) -> Result<IndirectCommandBuffer, Error> {
    use metal2vulkan::metal::IndirectCommandType;
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.max_vertex_buffer_bind_count = 2;
    descriptor.max_fragment_buffer_bind_count = 1;
    let icb =
        lane.device
            .new_indirect_command_buffer(&descriptor, count, ResourceOptions::shared())?;
    for index in 0..count {
        let command = icb.indirect_render_command_at(index)?;
        command.set_render_pipeline_state(pipeline)?;
        command.set_fragment_buffer(Some(counter), 0, 0)?;
        command.set_vertex_buffer(Some(counter), 0, 1)?;
        command.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)?;
        if barrier == Some(index) {
            command.set_barrier()?;
        }
    }
    Ok(icb)
}

/// An indirect command buffer's draw under `Never` counts no sample, as the encoder's does, in
/// an execution beside commands that count and around it (the M3 Max and an M4 alike): its query
/// pauses around the draw in a spare the call lent the execution, and the spares no draw took count
/// nothing. So do commands a barrier splits into runs, each run's draws counted (plain Metal: its
/// validation stops `setBarrier` on a buffer without mesh draws), one execution of the barrier's
/// command and the one after it, and one of the barrier's alone before a `Never` command.
#[test]
fn executed_never_draws_count_no_sample() {
    let (messages, _cache, lane) = Lane::new();
    let device = lane.device.name().to_owned();
    let (counting, boolean) = (
        VisibilityResultMode::Counting,
        VisibilityResultMode::Boolean,
    );
    let always = NEVER.compare(CompareFunction::Always);
    // The commands' functions count; the encoder's state at each call decides the compare.
    let pipeline = lane.pipeline_for(NEVER, false, true);
    for (name, mode, (ranges, barrier), result, fragments, color) in [
        (
            "counting, never between",
            counting,
            (&[(always, 0..1), (NEVER, 1..3), (always, 3..4)][..], None),
            32,
            64,
            RED,
        ),
        (
            "counting, never alone",
            counting,
            (&[(NEVER, 0..2)][..], None),
            0,
            32,
            CLEARED,
        ),
        (
            "boolean, never alone",
            boolean,
            (&[(NEVER, 0..2)][..], None),
            0,
            32,
            CLEARED,
        ),
        (
            "boolean, never then always",
            boolean,
            (&[(NEVER, 0..1), (always, 1..2)][..], None),
            1,
            32,
            RED,
        ),
        (
            "counting, a barrier",
            counting,
            (&[(always, 0..1), (always, 1..3)][..], Some(1)),
            48,
            48,
            RED,
        ),
        (
            "counting, a barrier, then never",
            counting,
            (
                &[(always, 0..1), (always, 1..2), (NEVER, 2..3)][..],
                Some(1),
            ),
            32,
            48,
            RED,
        ),
    ] {
        let left = lane.run(false, true, |lane, e, counter| {
            let icb = counting_commands(lane, &pipeline, (4, barrier), counter)?;
            e.set_visibility_result_mode(mode, 0)?;
            for (draw, range) in ranges {
                e.set_depth_stencil_state(&lane.state(*draw))?;
                e.execute_commands_in_buffer(&icb, range.clone())?;
            }
            Ok(())
        });
        if counting_refused(&left, &device) {
            continue;
        }
        let left = left.unwrap();
        assert_eq!(by_contract(&left, mode), Some(result), "{name} on {device}");
        assert_eq!(left.fragments, fragments, "{name} on {device}");
        assert_eq!(left.color, each(color), "{name} on {device}");
    }
    quiet(lane, &messages);
}
