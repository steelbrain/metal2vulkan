//! Device tests of visibility results (its third part) on the device the cfg selects, under the
//! validation layer.
//!
//! Each pass is one that Metal probes measured (an M4 with macOS 27.0.1, each case with and without
//! Metal's API validation), drawn with the owned fixture `rt6_render_state` into a 4² target: the
//! fullscreen triangle covers every pixel, and the same draw from vertex 3 covers none. Each
//! visibility buffer starts as 0xaa bytes, and its words after the command buffer are compared with
//! the ones Metal wrote: `Boolean` writes 1 or 0 at each offset the encoder set (the runtime the
//! device's nonzero value for 1, compared by [`by_contract`]), `Counting` the
//! samples that passed (16 for the fullscreen triangle, 48 for three instances of it, 64 at 4x), an
//! offset set with no draw after it gets 0, an offset never set keeps its bytes, a later pass
//! rewrites only its own offsets, and an offset taken again in its mode, which Metal's validation
//! stops, adds the draws since to its result. An offset that is not a multiple of 8, which Metal's
//! validation stops, lands at the offset rounded down to 8, where plain Metal writes it. What else
//! Metal's validation stops, the runtime refuses typed; a device without precise occlusion queries
//! refuses `Counting` (`Unsupported`, naming `occlusionQueryPrecise`), printed.

use std::collections::BTreeSet;

use super::degrade_notices::{
    assert_made_once, assert_once, made_once, made_render_kinds, shared_in_system_memory,
};
use super::read_buffer;
use super::render_state::{quiet, texture, validated, Pipe, Probe, PASS, WHITE};
use super::resolves::{self, color_pass};
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{
    CompareFunction, IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, Degradation, Error, IndirectCommandBufferDescriptor,
    Invalid, LoadAction, PipelineOptions, Problem, RenderCommandEncoder, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, StoreAction, Texture, VisibilityProblem,
    VisibilityResultMode, VisibilityResultType,
};

/// A word Metal never wrote: the buffer's starting bytes.
const UNWRITTEN: u64 = 0xaaaa_aaaa_aaaa_aaaa;

/// A visibility buffer of `words` 64-bit words, every byte 0xaa.
fn visibility_buffer(probe: &Probe, words: usize) -> Buffer {
    super::buffer_with(&probe.device, &vec![0xaa; 8 * words]).unwrap()
}

/// The buffer's 64-bit words.
fn words(buffer: &Buffer) -> Vec<u64> {
    read_buffer(buffer, 0, buffer.length())
        .unwrap()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
        .collect()
}

/// The probe's fullscreen draw with `pipe` at depth `z` (`covering`), or the same draw from
/// vertex 3, which covers no pixel.
fn draw(
    probe: &Probe,
    encoder: &mut RenderCommandEncoder,
    held: &mut Vec<Buffer>,
    pipe: Pipe,
    z: f32,
    covering: bool,
) {
    encoder
        .set_render_pipeline_state(&probe.pipeline(pipe))
        .unwrap();
    let depth = probe.depth(z, 0.0);
    encoder.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
    let start = if covering { 0 } else { 3 };
    encoder
        .draw_primitives(PrimitiveType::Triangle, start, 3, 1, 0)
        .unwrap();
    held.push(depth);
}

/// A mode set the probe set.
fn set(encoder: &mut RenderCommandEncoder, mode: VisibilityResultMode, offset: u64) {
    encoder.set_visibility_result_mode(mode, offset).unwrap();
}

/// `words` as their modes' contracts read them, to compare with the words an Apple M3 Max wrote:
/// each word at an index of `booleans`, a `Boolean` result, 1 where it is nonzero (the device's own
/// value where a sample passed, `Degradation::DeviceBooleanVisibility`) and not `unwritten`, the
/// buffer's starting word, so a result never written still differs; every other word as it is.
pub(super) fn by_contract(mut words: Vec<u64>, booleans: &[usize], unwritten: u64) -> Vec<u64> {
    for &index in booleans {
        let word = &mut words[index];
        if *word != 0 && *word != unwritten {
            *word = 1;
        }
    }
    words
}

/// [`by_contract`] of a buffer that started as [`UNWRITTEN`].
fn read(buffer: &Buffer, booleans: &[usize]) -> Vec<u64> {
    by_contract(words(buffer), booleans, UNWRITTEN)
}

/// The words Metal wrote in each pass (the module docs).
#[test]
fn boolean_results_store_metals_words() {
    use VisibilityResultMode::{Boolean, Disabled};
    let (messages, _cache, probe) = validated();
    // Probe 1's visibility_modes, its Boolean offsets: drawn, a draw covering nothing, set with
    // no draw (then Disabled at another offset, which Metal leaves), and two draws.
    let vis = visibility_buffer(&probe, 8);
    probe
        .try_run_with(PASS, Some(&vis), |p, e, h| {
            set(e, Boolean, 0);
            draw(p, e, h, WHITE, 0.5, true);
            set(e, Boolean, 24);
            draw(p, e, h, WHITE, 0.5, false);
            set(e, Boolean, 40);
            set(e, Disabled, 48);
            draw(p, e, h, WHITE, 0.5, true);
            set(e, Boolean, 56);
            draw(p, e, h, WHITE, 0.5, true);
            draw(p, e, h, WHITE, 0.5, true);
        })
        .unwrap();
    let u = UNWRITTEN;
    assert_eq!(read(&vis, &[0, 3, 5, 7]), [1, u, u, 0, u, 0, u, 1]);
    // A second pass over the buffer rewrites only its own offset.
    probe
        .try_run_with(PASS, Some(&vis), |p, e, h| {
            set(e, Boolean, 8);
            draw(p, e, h, WHITE, 0.5, true);
        })
        .unwrap();
    assert_eq!(read(&vis, &[0, 1, 3, 5, 7]), [1, 1, u, 0, u, 0, u, 1]);

    // Probe 5's vis_revisit_boolean_then_none and vis_revisit_none_then_boolean: offset 0 taken
    // again after offset 8 writes 1 where either of its draws passed a sample.
    for (name, first, again, metal) in [
        ("vis_revisit_boolean_then_none", true, false, [1u64, 0]),
        ("vis_revisit_none_then_boolean", false, true, [1, 1]),
    ] {
        let vis = visibility_buffer(&probe, 2);
        probe
            .try_run_with(PASS, Some(&vis), |p, e, h| {
                set(e, Boolean, 0);
                draw(p, e, h, WHITE, 0.5, first);
                set(e, Boolean, 8);
                draw(p, e, h, WHITE, 0.5, !first);
                set(e, Boolean, 0);
                draw(p, e, h, WHITE, 0.5, again);
            })
            .unwrap();
        assert_eq!(read(&vis, &[0, 1]), metal, "{name}");
    }

    // Probe 1's visibility_occluded: a draw behind the cleared depth (0.5 against 0.25 under
    // Less) passes no sample.
    let vis = visibility_buffer(&probe, 2);
    let depth = PixelFormat::Depth32Float;
    probe
        .try_run_with(PASS.depth(depth, 0.25), Some(&vis), |p, e, h| {
            e.set_depth_stencil_state(&p.state(CompareFunction::Less, false, None))
                .unwrap();
            set(e, Boolean, 8);
            draw(p, e, h, WHITE.depth(depth), 0.5, true);
            set(e, Boolean, 0);
            draw(p, e, h, WHITE.depth(depth), 0.125, true);
        })
        .unwrap();
    assert_eq!(read(&vis, &[0, 1]), [1, 0]);

    // Probe 5's vis_indirect_no_instances, as Boolean: an indirect draw of no instances.
    let vis = visibility_buffer(&probe, 1);
    let arguments = super::buffer_with(&probe.device, &args(&[3, 0, 0, 0])).unwrap();
    probe
        .try_run_with(PASS, Some(&vis), |p, e, h| {
            set(e, Boolean, 0);
            e.set_render_pipeline_state(&p.pipeline(WHITE)).unwrap();
            let depth = p.depth(0.5, 0.0);
            e.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
            e.draw_primitives_indirect(PrimitiveType::Triangle, &arguments, 0)
                .unwrap();
            h.push(depth);
        })
        .unwrap();
    assert_eq!(read(&vis, &[0]), [0]);
    quiet(probe, &messages);
}

/// The kind a pass writing a `Boolean` result takes
/// ([`boolean_results_announce_the_device_s_value`]).
pub(super) const BOOLEANS: [Degradation; 1] = [Degradation::DeviceBooleanVisibility];

/// A pass writing a `Boolean` result announces [`BOOLEANS`] once on a device, however many passes
/// write one, naming the pass's first `Boolean` offset; a pass of `Counting` results alone does
/// not. Beside it the device announces what the steps force: the white pipeline's making's kinds
/// and the visibility buffer's system memory.
#[test]
fn boolean_results_announce_the_device_s_value() {
    use VisibilityResultMode::{Boolean, Counting};
    let (messages, _cache, probe) = validated();
    let precise = probe
        .device
        .properties()
        .description
        .has(metal2vulkan::device::VkFeature::OcclusionQueryPrecise);
    let vis = visibility_buffer(&probe, 2);
    if precise {
        probe
            .try_run_with(PASS, Some(&vis), |p, e, h| {
                set(e, Counting, 0);
                draw(p, e, h, WHITE, 0.5, true);
            })
            .unwrap();
        assert!(
            !messages.take_notices().iter().any(|m| m
                .degradation
                .as_ref()
                .is_some_and(|n| n.degradation == BOOLEANS[0])),
            "a pass of Counting results alone"
        );
    }
    for offset in [8, 0] {
        probe
            .try_run_with(PASS, Some(&vis), |p, e, h| {
                set(e, Boolean, offset);
                draw(p, e, h, WHITE, 0.5, true);
            })
            .unwrap();
    }
    let mut first = probe.made(WHITE);
    if shared_in_system_memory(&probe.device) {
        first.insert(Degradation::SharedInSystemMemory);
    }
    let notices = made_once(&messages, &BOOLEANS, &first);
    let notice = notices
        .iter()
        .find_map(|m| {
            m.degradation
                .as_ref()
                .filter(|n| n.degradation == BOOLEANS[0])
        })
        .unwrap();
    assert_eq!(notice.object, "the Boolean visibility result at offset 8");
    quiet(probe, &messages);
}

/// A pass's `Boolean` results are its queries' own words: one `vkCmdCopyQueryPoolResults` of
/// both offsets' queries and no dispatch after it, the word of the draw that passed a sample
/// nonzero and the other's 0.
#[test]
fn boolean_results_are_copied_with_no_dispatch() {
    use VisibilityResultMode::Boolean;
    let (messages, _cache, probe) = validated();
    let vis = visibility_buffer(&probe, 2);
    let before = probe.device.work_counts().commands_recorded;
    probe
        .try_run_with(PASS, Some(&vis), |p, e, h| {
            set(e, Boolean, 0);
            draw(p, e, h, WHITE, 0.5, true);
            set(e, Boolean, 8);
            draw(p, e, h, WHITE, 0.5, false);
        })
        .unwrap();
    let after = probe.device.work_counts().commands_recorded;
    assert_eq!(
        (
            after.cmd_copy_query_pool_results - before.cmd_copy_query_pool_results,
            after.cmd_dispatch - before.cmd_dispatch,
        ),
        (1, 0),
        "query result copies and dispatches"
    );
    let words = words(&vis);
    assert!(words[0] != 0 && words[1] == 0, "{words:?}");
    quiet(probe, &messages);
}

/// A visibility buffer holding `words`.
fn buffer_of(probe: &Probe, words: &[u64]) -> Buffer {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    super::buffer_with(&probe.device, &bytes).unwrap()
}

/// Encode one pass over `color` into `command_buffer`, its visibility results of `kind` into
/// `vis`, `body` encoding into it.
pub(super) fn encode_pass(
    probe: &Probe,
    command_buffer: &CommandBuffer,
    color: &Texture,
    vis: &Buffer,
    kind: VisibilityResultType,
    held: &mut Vec<Buffer>,
    body: impl FnOnce(&Probe, &mut RenderCommandEncoder, &mut Vec<Buffer>),
) {
    let mut descriptor = RenderPassDescriptor::new();
    let target = &mut descriptor.color_attachments[0];
    target.texture = Some(color.clone());
    target.load_action = LoadAction::Clear;
    target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    target.store_action = StoreAction::Store;
    descriptor.visibility_result_buffer = Some(vis.clone());
    descriptor.visibility_result_type = kind;
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    body(probe, &mut encoder, held);
    encoder.end_encoding().unwrap();
}

/// Commit `command_buffer` and wait for it, then drop what its passes held.
pub(super) fn complete(command_buffer: CommandBuffer, held: Vec<Buffer>) {
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(super::WAIT)
        .unwrap();
    assert_eq!(command_buffer.error(), None);
    drop(held);
}

/// Whether the probe's device has precise occlusion queries, which `Counting` needs; printed
/// where it lacks them.
pub(super) fn precise(probe: &Probe) -> bool {
    let precise = probe
        .device
        .properties()
        .description
        .has(metal2vulkan::device::VkFeature::OcclusionQueryPrecise);
    if !precise {
        eprintln!(
            "skipped: Counting visibility results on {}: no precise occlusion queries",
            probe.device.name()
        );
    }
    precise
}

/// `Accumulate`, counted, as an Apple M3 Max answers it (the `visibility-counting-accumulate*`
/// conformance cases): each pass's count added to the word's low 32 bits, a covering draw's 16
/// samples, a draw covering nothing's none, an offset set with no draw keeping its word, the sum
/// wrapping at 32 bits with no carry into the high 32, which it keeps; a second pass in the same
/// command buffer and in the next one adding to the words the first wrote; an offset taken again
/// adding both of its draws. `Reset` over the same words rewrites every offset the pass set, an
/// offset set with no draw to 0.
#[test]
fn accumulated_counts_add_to_the_words() {
    use VisibilityResultMode::Counting;
    use VisibilityResultType::{Accumulate, Reset};
    let (messages, _cache, probe) = validated();
    if !precise(&probe) {
        quiet(probe, &messages);
        return;
    }
    let color = texture(&probe.device, PixelFormat::RGBA8Unorm, 4);
    let (carry, wrap) = (0xffff_fff8u64, u64::MAX - 7);
    let start = [100, 5, 7, carry, wrap];
    let first = |p: &Probe, e: &mut RenderCommandEncoder, h: &mut Vec<Buffer>| {
        set(e, Counting, 0);
        draw(p, e, h, WHITE, 0.5, true);
        set(e, Counting, 8);
        draw(p, e, h, WHITE, 0.5, false);
        set(e, Counting, 16);
        set(e, Counting, 24);
        draw(p, e, h, WHITE, 0.5, true);
        set(e, Counting, 32);
        draw(p, e, h, WHITE, 0.5, true);
    };
    let second = |p: &Probe, e: &mut RenderCommandEncoder, h: &mut Vec<Buffer>| {
        set(e, Counting, 0);
        draw(p, e, h, WHITE, 0.5, true);
        set(e, Counting, 8);
    };
    // The low halves wrap; the high halves stay.
    let after_first = [116, 5, 7, 8, 0xffff_ffff_0000_0008];
    let after_second = [132, 5, 7, 8, 0xffff_ffff_0000_0008];
    // One command buffer a pass.
    let vis = buffer_of(&probe, &start);
    for (body, want, name) in [
        (
            &first as &dyn Fn(&Probe, &mut RenderCommandEncoder, &mut Vec<Buffer>),
            after_first,
            "the first pass",
        ),
        (&second, after_second, "the next command buffer's pass"),
    ] {
        let command_buffer = probe.queue.command_buffer().unwrap();
        let mut held = Vec::new();
        encode_pass(
            &probe,
            &command_buffer,
            &color,
            &vis,
            Accumulate,
            &mut held,
            body,
        );
        complete(command_buffer, held);
        assert_eq!(words(&vis), want, "{name}");
    }
    // Both passes in one command buffer.
    let vis = buffer_of(&probe, &start);
    let command_buffer = probe.queue.command_buffer().unwrap();
    let mut held = Vec::new();
    encode_pass(
        &probe,
        &command_buffer,
        &color,
        &vis,
        Accumulate,
        &mut held,
        first,
    );
    encode_pass(
        &probe,
        &command_buffer,
        &color,
        &vis,
        Accumulate,
        &mut held,
        second,
    );
    complete(command_buffer, held);
    assert_eq!(
        words(&vis),
        after_second,
        "two passes in one command buffer"
    );
    // An offset taken again (a continuation of its query) adds both of its draws.
    let vis = buffer_of(&probe, &[100, 5]);
    let command_buffer = probe.queue.command_buffer().unwrap();
    let mut held = Vec::new();
    encode_pass(
        &probe,
        &command_buffer,
        &color,
        &vis,
        Accumulate,
        &mut held,
        |p, e, h| {
            set(e, Counting, 0);
            draw(p, e, h, WHITE, 0.5, true);
            set(e, Counting, 8);
            draw(p, e, h, WHITE, 0.5, true);
            set(e, Counting, 0);
            draw(p, e, h, WHITE, 0.5, true);
        },
    );
    complete(command_buffer, held);
    assert_eq!(words(&vis), [132, 21], "an offset taken again");
    // `Reset`, the default, over the same words.
    let vis = buffer_of(&probe, &start);
    let command_buffer = probe.queue.command_buffer().unwrap();
    let mut held = Vec::new();
    encode_pass(
        &probe,
        &command_buffer,
        &color,
        &vis,
        Reset,
        &mut held,
        first,
    );
    complete(command_buffer, held);
    assert_eq!(words(&vis), [16, 0, 0, 16, 16], "Reset");
    quiet(probe, &messages);
}

/// `Accumulate`, `Boolean`, as an Apple M3 Max answers it (the `visibility-boolean-accumulate`
/// conformance case): a word becomes nonzero where a sample passed and keeps its value where none
/// did (0, 2 and 3 under each), an offset set with no draw keeps its word, and a later command
/// buffer's pass ORs into the words the first left. `Reset` over the same words writes the
/// results alone, 0 where no sample passed.
#[test]
fn accumulated_booleans_or_into_the_words() {
    use VisibilityResultMode::Boolean;
    use VisibilityResultType::{Accumulate, Reset};
    let (messages, _cache, probe) = validated();
    let color = texture(&probe.device, PixelFormat::RGBA8Unorm, 4);
    let start = [0, 2, 3, 0, 2, 3, 9];
    let first = |p: &Probe, e: &mut RenderCommandEncoder, h: &mut Vec<Buffer>| {
        for (at, covering) in [
            (0, true),
            (8, true),
            (16, true),
            (24, false),
            (32, false),
            (40, false),
        ] {
            set(e, Boolean, at);
            draw(p, e, h, WHITE, 0.5, covering);
        }
        set(e, Boolean, 48);
    };
    let run =
        |vis: &Buffer, kind, body: &dyn Fn(&Probe, &mut RenderCommandEncoder, &mut Vec<Buffer>)| {
            let command_buffer = probe.queue.command_buffer().unwrap();
            let mut held = Vec::new();
            encode_pass(&probe, &command_buffer, &color, vis, kind, &mut held, body);
            complete(command_buffer, held);
            words(vis)
        };
    let vis = buffer_of(&probe, &start);
    let after = run(&vis, Accumulate, &first);
    assert!(
        after[..3].iter().all(|&w| w != 0),
        "where a sample passed: {after:?}"
    );
    assert_eq!(
        after[3..],
        [0, 2, 3, 9],
        "where none passed, and set with no draw"
    );
    let next = run(&vis, Accumulate, &|p, e, h| {
        set(e, Boolean, 24);
        draw(p, e, h, WHITE, 0.5, true);
    });
    assert_ne!(next[3], 0, "the next command buffer's pass: {next:?}");
    assert_eq!(
        (&next[..3], &next[4..]),
        (&after[..3], &after[4..]),
        "the words it did not set"
    );
    // `Reset`: the results alone.
    let vis = buffer_of(&probe, &start);
    let reset = run(&vis, Reset, &first);
    assert!(
        reset[..3].iter().all(|&w| w != 0),
        "Reset where a sample passed: {reset:?}"
    );
    assert_eq!(
        reset[3..],
        [0, 0, 0, 0],
        "Reset where none passed, and set with no draw"
    );
    quiet(probe, &messages);
}

/// Whether `set`, a `Counting` mode set, is the refusal of a device without precise occlusion
/// queries (printed for `device`); any other refusal fails.
fn counting_refused(set: Result<(), Error>, device: &str) -> bool {
    match set {
        Ok(()) => false,
        Err(Error::Unsupported(u)) if u.feature == Feature::CountingVisibility => {
            eprintln!("refused: Counting visibility results on {device}: {u}");
            true
        }
        Err(error) => panic!("Counting on {device}: {error}"),
    }
}

/// The counts Metal wrote in probes 1 and 5 (the module docs), at 1x and 4x.
#[test]
fn counting_results_store_metals_counts() {
    use VisibilityResultMode::{Boolean, Counting, Disabled};
    let (messages, _cache, probe) = validated();
    let device = probe.device.name().to_owned();
    // Probe 1's counts at their offsets: drawn, a draw covering nothing, two draws, three
    // instances, set with no draw; beside a `Boolean` offset, and a draw after `Disabled` that
    // counts nowhere.
    let vis = visibility_buffer(&probe, 8);
    let mut refused = false;
    probe
        .try_run_with(PASS, Some(&vis), |p, e, h| {
            refused = counting_refused(e.set_visibility_result_mode(Counting, 0), &device);
            if refused {
                return;
            }
            draw(p, e, h, WHITE, 0.5, true);
            set(e, Counting, 8);
            draw(p, e, h, WHITE, 0.5, false);
            set(e, Counting, 16);
            draw(p, e, h, WHITE, 0.5, true);
            draw(p, e, h, WHITE, 0.5, true);
            set(e, Counting, 24);
            let depth = p.depth(0.5, 0.0);
            e.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 3, 0)
                .unwrap();
            h.push(depth);
            set(e, Counting, 32);
            set(e, Boolean, 40);
            draw(p, e, h, WHITE, 0.5, true);
            set(e, Disabled, 0);
            draw(p, e, h, WHITE, 0.5, true);
        })
        .unwrap();
    if refused {
        quiet(probe, &messages);
        return;
    }
    let u = UNWRITTEN;
    assert_eq!(read(&vis, &[5]), [16, 0, 32, 48, 0, 1, u, u], "on {device}");

    // Probe 1's visibility_occluded, counted: a draw behind the cleared depth passes no sample.
    let vis = visibility_buffer(&probe, 2);
    let depth = PixelFormat::Depth32Float;
    probe
        .try_run_with(PASS.depth(depth, 0.25), Some(&vis), |p, e, h| {
            e.set_depth_stencil_state(&p.state(CompareFunction::Less, false, None))
                .unwrap();
            set(e, Counting, 8);
            draw(p, e, h, WHITE.depth(depth), 0.5, true);
            set(e, Counting, 0);
            draw(p, e, h, WHITE.depth(depth), 0.125, true);
        })
        .unwrap();
    assert_eq!(words(&vis), [16, 0], "occluded on {device}");

    // Probe 5's vis_indirect_no_instances: an indirect draw of no instances counts 0.
    let vis = visibility_buffer(&probe, 1);
    let arguments = super::buffer_with(&probe.device, &args(&[3, 0, 0, 0])).unwrap();
    probe
        .try_run_with(PASS, Some(&vis), |p, e, h| {
            set(e, Counting, 0);
            e.set_render_pipeline_state(&p.pipeline(WHITE)).unwrap();
            let depth = p.depth(0.5, 0.0);
            e.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
            e.draw_primitives_indirect(PrimitiveType::Triangle, &arguments, 0)
                .unwrap();
            h.push(depth);
        })
        .unwrap();
    assert_eq!(
        words(&vis),
        [0],
        "an indirect draw of no instances on {device}"
    );

    // Probe 1's visibility_misaligned: `Counting` at 4 writes its count at 0, as plain Metal
    // does where its validation stops it. Probe 5d's vis_disabled_misaligned: `Disabled` at 3
    // ends the count at 0.
    let u = UNWRITTEN;
    for (name, disable) in [
        ("visibility_misaligned", false),
        ("vis_disabled_misaligned", true),
    ] {
        let vis = visibility_buffer(&probe, 8);
        probe
            .try_run_with(PASS, Some(&vis), |p, e, h| {
                set(e, Counting, if disable { 0 } else { 4 });
                draw(p, e, h, WHITE, 0.5, true);
                if disable {
                    set(e, Disabled, 3);
                    draw(p, e, h, WHITE, 0.5, true);
                }
            })
            .unwrap();
        assert_eq!(words(&vis), [16, u, u, u, u, u, u, u], "{name} on {device}");
    }
    // Probe 5's vis_revisit_counting and vis_revisit_after_disabled: offset 0 taken again, after
    // offset 8 or after `Disabled`, sums its draws.
    for (name, between, metal) in [
        ("vis_revisit_counting", Counting, [32, 16]),
        ("vis_revisit_after_disabled", Disabled, [32, u]),
    ] {
        let vis = visibility_buffer(&probe, 2);
        probe
            .try_run_with(PASS, Some(&vis), |p, e, h| {
                set(e, Counting, 0);
                draw(p, e, h, WHITE, 0.5, true);
                set(e, between, if between == Disabled { 0 } else { 8 });
                draw(p, e, h, WHITE, 0.5, true);
                set(e, Counting, 0);
                draw(p, e, h, WHITE, 0.5, true);
            })
            .unwrap();
        assert_eq!(words(&vis), metal, "{name} on {device}");
    }
    quiet(probe, &messages);

    // Probe 1's 4x pass: the fullscreen triangle passes every sample of the 4² target.
    let (messages, cache, lane) = resolves::lane();
    let pf = PixelFormat::RGBA8Unorm;
    let pipeline = lane.pipeline((resolves::FULL, resolves::WHITE), &[pf], (None, None), 4);
    let msaa = lane.msaa(pf);
    let vis = super::buffer_with(&lane.device, &[0xaa; 8]).unwrap();
    let mut descriptor = color_pass(&msaa, None, StoreAction::DontCare, (0, 0));
    descriptor.visibility_result_buffer = Some(vis.clone());
    lane.run(&descriptor, |lane, e, h| {
        set(e, Counting, 0);
        lane.draw(e, h, &pipeline, 0.5);
    })
    .unwrap();
    assert_eq!(words(&vis), [64], "4x on {device}");
    eprintln!("Counting visibility results on {device}: Metal's counts");
    drop(cache);
    resolves::quiet(lane, &messages);
}

/// The command buffers [`command_buffers_released_at_completion_leave_nothing_read_later`]
/// releases.
const RELEASES: usize = 128;

/// One pass of two `Boolean` offsets per command buffer, [`RELEASES`] times, each command buffer
/// released as soon as it completed: its occlusion query pools, and its command pool with what the
/// driver made for `vkCmdCopyQueryPoolResults`, are released then, and nothing the device runs
/// later may read them. MoltenVK 1.4.1 reads both from completed handlers of the Metal command
/// buffer that run after the timeline value the runtime waits for, and a process that released them
/// at that value died of `SIGSEGV` there.
#[test]
fn command_buffers_released_at_completion_leave_nothing_read_later() {
    use VisibilityResultMode::Boolean;
    let (messages, _cache, probe) = validated();
    let vis = visibility_buffer(&probe, 2);
    for round in 0..RELEASES {
        let first = round % 2 == 0;
        probe
            .try_run_with(PASS, Some(&vis), |p, e, h| {
                set(e, Boolean, 0);
                draw(p, e, h, WHITE, 0.5, first);
                set(e, Boolean, 8);
                draw(p, e, h, WHITE, 0.5, !first);
            })
            .unwrap();
        let expected = if first { [1, 0] } else { [0, 1] };
        assert_eq!(read(&vis, &[0, 1]), expected, "round {round}");
    }
    quiet(probe, &messages);
}

/// Little-endian words of indirect arguments.
fn args(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Probe 5's two passes over one buffer in one command buffer, each writing offset 0, and its
/// blit later in the command buffer copying the results: the blit sees them.
#[test]
fn results_are_ordered_with_the_command_buffers_later_commands() {
    use VisibilityResultMode::Boolean;
    let (messages, _cache, probe) = validated();
    let vis = visibility_buffer(&probe, 2);
    let copy = visibility_buffer(&probe, 2);
    let color = texture(&probe.device, PixelFormat::RGBA8Unorm, 4);
    let command_buffer = probe.queue.command_buffer().unwrap();
    let mut held = Vec::new();
    for (offset, covering) in [(0, false), (0, true), (8, true)] {
        let mut descriptor = RenderPassDescriptor::new();
        let target = &mut descriptor.color_attachments[0];
        target.texture = Some(color.clone());
        target.load_action = LoadAction::Clear;
        target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        target.store_action = StoreAction::Store;
        descriptor.visibility_result_buffer = Some(vis.clone());
        let mut e = command_buffer.render_command_encoder(&descriptor).unwrap();
        set(&mut e, Boolean, offset);
        draw(&probe, &mut e, &mut held, WHITE, 0.5, covering);
        e.end_encoding().unwrap();
    }
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&vis, 0, &copy, 0, 16)
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(super::WAIT)
        .unwrap();
    assert_eq!(command_buffer.error(), None);
    drop(held);
    assert_eq!(read(&vis, &[0, 1]), [1, 1]);
    assert_eq!(read(&copy, &[0, 1]), [1, 1]);
    quiet(probe, &messages);
}

/// What Metal's validation stops is refused typed, before anything records, and a refused call
/// keeps the mode set before.
#[test]
fn visibility_modes_are_refused_as_metal_refuses_them() {
    use VisibilityResultMode::{Boolean, Counting, Disabled};
    let (messages, _cache, probe) = validated();
    let refused = |problem| {
        Err(Error::Invalid(Invalid::new(
            "set_visibility_result_mode",
            Problem::Visibility(problem),
        )))
    };
    // No visibility buffer (probe 1: stopped even without validation); `Disabled` is taken
    // there (probe 5d).
    probe
        .try_run_with(PASS, None, |_, e, _| {
            assert_eq!(
                e.set_visibility_result_mode(Boolean, 0),
                refused(VisibilityProblem::NoBuffer)
            );
            set(e, Disabled, 0);
        })
        .unwrap();
    let vis = visibility_buffer(&probe, 8);
    probe
        .try_run_with(PASS, Some(&vis), |p, e, h| {
            set(e, Boolean, 0);
            // An offset past the buffer (probe 5), `Boolean` and `Counting`: each refused, and
            // the mode at offset 0 stays set for the draw below.
            assert_eq!(
                e.set_visibility_result_mode(Boolean, 64),
                refused(VisibilityProblem::OffsetOutOfRange {
                    offset: 64,
                    length: 64
                })
            );
            assert_eq!(
                e.set_visibility_result_mode(Counting, 64),
                refused(VisibilityProblem::OffsetOutOfRange {
                    offset: 64,
                    length: 64
                })
            );
            draw(p, e, h, WHITE, 0.5, true);
            // Probe 5d: `Disabled` takes an offset past the buffer, and twice, and its offset is
            // no use of it.
            set(e, Disabled, 64);
            set(e, Disabled, 16);
            set(e, Disabled, 16);
            set(e, Boolean, 16);
            draw(p, e, h, WHITE, 0.5, true);
            // Probe 5: an offset taken before in the other mode (plain Metal's ordered answer
            // is not built), after Disabled too, is refused.
            set(e, Boolean, 56);
            set(e, Disabled, 0);
            assert_eq!(
                e.set_visibility_result_mode(Counting, 0),
                refused(VisibilityProblem::OffsetReused { offset: 0 })
            );
        })
        .unwrap();
    let u = UNWRITTEN;
    assert_eq!(read(&vis, &[0, 2, 7]), [1, u, 1, u, u, u, u, 0]);
    quiet(probe, &messages);
}

/// The probe's fullscreen triangle filled with `rt6_half`, made with
/// `supportIndirectCommandBuffers` so an indirect command can name it.
fn half_for_commands(probe: &Probe) -> RenderPipelineState {
    probe
        .device
        .new_render_pipeline_state(&half_descriptor(probe), &PipelineOptions::default())
        .unwrap()
}

/// The descriptor of [`half_for_commands`]'s state.
fn half_descriptor(probe: &Probe) -> RenderPipelineDescriptor {
    let function = |name| probe.library.new_function_with_name(name).unwrap();
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
        function("rt6_full").function_descriptor(),
    ));
    descriptor.fragment = Some(StageSource::Function(
        function("rt6_half").function_descriptor(),
    ));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.support_indirect_command_buffers = true;
    descriptor
}

/// The kinds an execution test's steps force where the device has not announced them: the
/// commands' half grey state's making, the white one's where `white` draws, and the `Shared`
/// buffers' system memory.
fn execution_kinds(probe: &Probe, white: bool) -> BTreeSet<Degradation> {
    let mut kinds = made_render_kinds(&probe.device, &half_descriptor(probe));
    if white {
        kinds.extend(probe.made(WHITE));
    }
    if shared_in_system_memory(&probe.device) {
        kinds.insert(Degradation::SharedInSystemMemory);
    }
    kinds
}

/// A pass that executes render commands keeps its visibility results, which refused it before: the
/// encoder's draws under a mode store Metal's words as without the execution, the command between
/// them, drawn while no mode is set, counts nowhere, and the pass draws in encode order (the
/// command's half grey over the encoder's white, as a pass that draws the half grey alone stores
/// it).
#[test]
fn a_pass_that_executes_commands_keeps_its_visibility_results() {
    use VisibilityResultMode::{Boolean, Disabled};
    let (messages, _cache, probe) = validated();
    let half = half_for_commands(&probe);
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.max_vertex_buffer_bind_count = 1;
    let icb = probe
        .device
        .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
        .unwrap();
    let depth = probe.depth(0.5, 0.0);
    let command = icb.indirect_render_command_at(0).unwrap();
    command.set_render_pipeline_state(&half).unwrap();
    command.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
    command
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    let alone = probe
        .try_run_with(PASS, None, |_, e, _| {
            e.set_render_pipeline_state(&half).unwrap();
            e.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        })
        .unwrap();
    let vis = visibility_buffer(&probe, 3);
    let stored = probe
        .try_run_with(PASS, Some(&vis), |p, e, h| {
            set(e, Boolean, 0);
            draw(p, e, h, WHITE, 0.5, true);
            set(e, Disabled, 0);
            e.use_resource(
                depth.as_resource(),
                ResourceUsage::READ,
                RenderStages::VERTEX,
            )
            .unwrap();
            e.execute_commands_in_buffer(&icb, 0..1).unwrap();
            set(e, Boolean, 8);
            draw(p, e, h, WHITE, 0.5, false);
        })
        .unwrap();
    assert_eq!(read(&vis, &[0, 1]), [1, 0, UNWRITTEN]);
    assert_eq!(
        stored.color, alone.color,
        "the command draws after the white"
    );
    quiet(probe, &messages);
}

/// An indirect command buffer of `count` commands, each drawing the probe's fullscreen triangle
/// with `pipeline` from `depth`.
fn covering_commands(
    probe: &Probe,
    pipeline: &RenderPipelineState,
    depth: &Buffer,
    count: u64,
) -> metal2vulkan_runtime::IndirectCommandBuffer {
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.max_vertex_buffer_bind_count = 1;
    let icb = probe
        .device
        .new_indirect_command_buffer(&descriptor, count, ResourceOptions::shared())
        .unwrap();
    for index in 0..count {
        let command = icb.indirect_render_command_at(index).unwrap();
        command.set_render_pipeline_state(pipeline).unwrap();
        command.set_vertex_buffer(Some(depth), 0, 0).unwrap();
        command
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }
    icb
}

/// One counted case of [`executions_count_in_the_mode_set_at_them`]: its name, its buffer's words,
/// what it encodes after setting `Counting` at offset 0, and the words Metal wrote.
type CountedCase<'a> = (
    &'a str,
    usize,
    &'a dyn Fn(&Probe, &mut RenderCommandEncoder, &mut Vec<Buffer>),
    &'a [u64],
);

/// An execution's draws count in the visibility result mode set at it, which refused it before, as
/// Metal's did (the M3 Max and the M4 alike; their 8² target's 64 samples per covering draw are the
/// probe's 4² target's 16 here). Each pass is one of their cases: `Boolean` writes 1 for a
/// command's draw, `Counting` sums every command's samples with the encoder's draws before and
/// after the execution, nothing counts while `Disabled`, and an offset set between two executions,
/// or between a draw and an execution, counts the draws after it alone.
#[test]
fn executions_count_in_the_mode_set_at_them() {
    use VisibilityResultMode::{Boolean, Counting, Disabled};
    let (messages, _cache, probe) = validated();
    let device = probe.device.name().to_owned();
    let half = half_for_commands(&probe);
    let depth = probe.depth(0.5, 0.0);
    let icb = covering_commands(&probe, &half, &depth, 2);
    let resident = |e: &mut RenderCommandEncoder| {
        e.use_resource(
            depth.as_resource(),
            ResourceUsage::READ,
            RenderStages::VERTEX,
        )
        .unwrap();
    };
    // bool_icb: one command under `Boolean`.
    let vis = visibility_buffer(&probe, 1);
    probe
        .try_run_with(PASS, Some(&vis), |_, e, _| {
            resident(e);
            set(e, Boolean, 0);
            e.execute_commands_in_buffer(&icb, 0..1).unwrap();
        })
        .unwrap();
    assert_eq!(read(&vis, &[0]), [1], "bool_icb on {device}");
    // bool_none: `Boolean`, then `Disabled`, then the command.
    let vis = visibility_buffer(&probe, 1);
    probe
        .try_run_with(PASS, Some(&vis), |_, e, _| {
            resident(e);
            set(e, Boolean, 0);
            set(e, Disabled, 0);
            e.execute_commands_in_buffer(&icb, 0..1).unwrap();
        })
        .unwrap();
    assert_eq!(read(&vis, &[0]), [0], "bool_none on {device}");
    // The counted cases, where the device counts: whether it did.
    let counted = |cases: &[CountedCase<'_>]| {
        for (name, length, encode, want) in cases {
            let vis = visibility_buffer(&probe, *length);
            let mut refused = false;
            probe
                .try_run_with(PASS, Some(&vis), |p, e, h| {
                    resident(e);
                    refused = counting_refused(e.set_visibility_result_mode(Counting, 0), &device);
                    if !refused {
                        encode(p, e, h);
                    }
                })
                .unwrap();
            if refused {
                return false;
            }
            assert_eq!(words(&vis), *want, "{name} on {device}");
        }
        true
    };
    let u = UNWRITTEN;
    let ran = counted(&[
        // count_icb2: two commands.
        (
            "count_icb2",
            1,
            &|_, e, _| {
                e.execute_commands_in_buffer(&icb, 0..2).unwrap();
            },
            &[32],
        ),
        // count_mixed: a draw, a command, a draw.
        (
            "count_mixed",
            1,
            &|p, e, h| {
                draw(p, e, h, WHITE, 0.5, true);
                e.execute_commands_in_buffer(&icb, 0..1).unwrap();
                draw(p, e, h, WHITE, 0.5, true);
            },
            &[48],
        ),
        // count_disabled_icb: a draw, then `Disabled`, then two commands.
        (
            "count_disabled_icb",
            2,
            &|p, e, h| {
                draw(p, e, h, WHITE, 0.5, true);
                set(e, Disabled, 0);
                e.execute_commands_in_buffer(&icb, 0..2).unwrap();
            },
            &[16, u],
        ),
        // sw_d: a command at offset 0, the next one at 8.
        (
            "sw_d",
            2,
            &|_, e, _| {
                e.execute_commands_in_buffer(&icb, 0..1).unwrap();
                set(e, Counting, 8);
                e.execute_commands_in_buffer(&icb, 1..2).unwrap();
            },
            &[16, 16],
        ),
        // sw_e: a draw at offset 0, two commands at 8.
        (
            "sw_e",
            2,
            &|p, e, h| {
                draw(p, e, h, WHITE, 0.5, true);
                set(e, Counting, 8);
                e.execute_commands_in_buffer(&icb, 0..2).unwrap();
            },
            &[16, 32],
        ),
        // A command at offset 0, then two from the same command at 8: each execution runs its
        // own range, as Metal documents, where Metal's GPU runs other commands, not the same
        // ones from run to run (sw_a drew [32, 32] on the M3 Max and [16, 16] on the M4 at this
        // case's sizes), which the device announces once.
        (
            "sw_a",
            2,
            &|_, e, _| {
                e.execute_commands_in_buffer(&icb, 0..1).unwrap();
                set(e, Counting, 8);
                e.execute_commands_in_buffer(&icb, 0..2).unwrap();
            },
            &[16, 32],
        ),
    ]);
    let notices = messages.take_notices();
    if ran {
        let mut kinds: Vec<Degradation> = execution_kinds(&probe, true).into_iter().collect();
        kinds.extend(REPEATED_STARTS);
        kinds.extend(BOOLEANS);
        assert_once(&notices, &kinds);
        let notice = notices
            .iter()
            .find_map(|m| {
                m.degradation
                    .as_ref()
                    .filter(|n| n.degradation == Degradation::RepeatedExecutionStart)
            })
            .unwrap();
        assert!(
            notice
                .object
                .contains(&format!("{:#x}", icb.gpu_resource_id().0))
                && notice.reason.contains(
                    "from command 0 with length 2 after executing it from there with length 1"
                ),
            "{notice:?}"
        );
    }
    quiet(probe, &messages);
}

/// The kind an execution from a command an earlier one started at with another length takes
/// (`executions_count_in_the_mode_set_at_them`'s `sw_a`, and
/// `executions_of_ranges_read_from_a_buffer_run_their_own_ranges`).
pub(super) const REPEATED_STARTS: [Degradation; 1] = [Degradation::RepeatedExecutionStart];

/// `sw_a` with its ranges read from a buffer (`execute_commands_in_buffer_indirect`): each
/// execution runs its own range, as Metal documents, where Metal's GPU runs other commands, not
/// the same ones from run to run (commands 0..1 then 0..2 so read ran
/// command 1 twice in each of three runs), which the host step announces once.
#[test]
fn executions_of_ranges_read_from_a_buffer_run_their_own_ranges() {
    use VisibilityResultMode::Counting;
    let (messages, _cache, probe) = validated();
    let device = probe.device.name().to_owned();
    let half = half_for_commands(&probe);
    let depth = probe.depth(0.5, 0.0);
    let icb = covering_commands(&probe, &half, &depth, 2);
    // `MTLIndirectCommandBufferExecutionRange`s: commands 0..1 at offset 0, 0..2 at offset 8.
    let range_bytes: Vec<u8> = [0u32, 1, 0, 2]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let ranges = super::buffer_with(&probe.device, &range_bytes).unwrap();
    let vis = visibility_buffer(&probe, 2);
    let mut refused = false;
    probe
        .try_run_with(PASS, Some(&vis), |_, e, _| {
            e.use_resource(
                depth.as_resource(),
                ResourceUsage::READ,
                RenderStages::VERTEX,
            )
            .unwrap();
            refused = counting_refused(e.set_visibility_result_mode(Counting, 0), &device);
            if !refused {
                e.execute_commands_in_buffer_indirect(&icb, &ranges, 0)
                    .unwrap();
                set(e, Counting, 8);
                e.execute_commands_in_buffer_indirect(&icb, &ranges, 8)
                    .unwrap();
            }
        })
        .unwrap();
    if !refused {
        assert_eq!(words(&vis), [16, 32], "on {device}");
        // Beside it, the ranges read from a buffer, which a host step reads
        // (`host-replayed-indirect-commands`), and what the steps force.
        let mut first = execution_kinds(&probe, false);
        first.insert(Degradation::HostReplayedIndirectCommands);
        assert_made_once(&messages, &REPEATED_STARTS, &first);
    }
    quiet(probe, &messages);
}
