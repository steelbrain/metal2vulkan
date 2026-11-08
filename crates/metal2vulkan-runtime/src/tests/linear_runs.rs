//! Runs of passes over a linear attachment rendered apart, merged once (`pass_runs.rs`): one
//! command buffer mixing passes over a render target over a buffer's bytes (each loading and
//! storing it beside a cleared and stored `Depth32Float` depth attachment, blending a rectangle
//! over it) with the encoders and commands that end a run or pass it: a 1x1 attachmentless
//! render encoder (a `-P` stencilprobe's), a pass over another texture, a blit reading the
//! target's texels, an event signal, a pass naming the target's buffer, and a pass whose box
//! reaches past its run's.
//!
//! With the device's rule forced (`DeviceShared::linear_rule_forced`), each pass over the target
//! renders it apart, through the image it keeps, its changed words merged back
//! (`linear_attachment.rs`, `linear_merge.rs`). The sequence runs three times: over the target,
//! where its passes form the runs the steps make, merged once each; over the target again with a
//! blit between every two encoders, which ends every run, so every pass merges after itself; and
//! over a private texture of the same first texels, which merges nothing. The texels each blit
//! read mid-sequence, and those the sequence left, are the same in all three.
//!
//! The frames of a WebGPU frame split at every indexed draw (`checked_frames_merge_once`, the
//! consumer's `many_passes`): a clear, then 706 passes over a target over a `Private` buffer, each
//! after a 1x1 encoder of no attachment whose state rasterizes nothing, beside a depth (and
//! stencil) attachment, each pass's data in the command buffer's upload arena (`setVertexBytes`)
//! or in a no-copy buffer over host pages: one run, merged once, in every shape.
//!
//! Each test waits for the device's in-place check to end before it makes anything
//! (`merge_scratch::settled`), and reads the counts of a window of its own commands only then
//! (`merge_scratch::counts_alone`): the check's own passes over linear images, on its thread, are
//! counted with the test's.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixtures
//! `validation/fixtures/public/engine/tob_renders.ll` and `ob3_index_buffer.ll` (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use super::merge_scratch::{blending, blending_beside, blit, bytes, drawn, first_bytes};
use super::merge_scratch::{counts_alone, settled};
use super::merge_scratch::{staged, target, texels};
use super::merge_scratch::{Host, Over, H, TEXEL};
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandQueue, Device, EventRef, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, Size,
    StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{RenderPipelineDescriptor, StageSource};
use std::sync::atomic::Ordering;

/// One step of the sequence.
#[derive(Clone, Copy, Debug)]
enum Step {
    /// A pass over the target blending rectangle `i` (`drawn`).
    Draw(usize),
    /// A pass over the target blending a rectangle over all of it, of `i`'s color.
    Whole(usize),
    /// A 1x1 render encoder of no attachment, one draw (a `-P` stencilprobe's).
    Lone,
    /// A pass over another texture, binding the target's buffer as a fragment buffer its
    /// function does not read where `true`, else a buffer of its own.
    Other(bool),
    /// A blit copying the target's texels into a buffer, read after the command buffer.
    Read,
    /// An event signal.
    Signal,
}

use Step::{Draw, Lone, Other, Read, Signal, Whole};

/// The sequence, and the runs it makes over the target, each of the passes noted:
/// 1. `Whole(0)` to `Draw(4)`: the 1x1 encoder and the pass over another texture pass it; the
///    blit reading the texels ends it (5 passes);
/// 2. `Whole(5)`, `Draw(6)`: the signal ends it (2);
/// 3. `Whole(7)`, `Draw(8)`: the pass naming the target's buffer ends it (2);
/// 4. `Whole(9)`, `Draw(10)`: the blit ends it (2);
/// 5. `Draw(0)` over its own box, which `Draw(42)`'s reaches past (1);
/// 6. `Draw(42)` over the whole target, `Draw(43)`, and past the 1x1 encoder `Draw(44)`; the
///    blit ends it (3);
/// 7. `Whole(11)`, past the pass over another texture `Draw(12)`, to the command buffer's end
///    (2).
const STEPS: [Step; 26] = [
    Whole(0),
    Draw(1),
    Draw(2),
    Lone,
    Draw(3),
    Other(false),
    Draw(4),
    Read,
    Whole(5),
    Draw(6),
    Signal,
    Whole(7),
    Draw(8),
    Other(true),
    Whole(9),
    Draw(10),
    Read,
    Draw(0),
    Draw(42),
    Draw(43),
    Lone,
    Draw(44),
    Read,
    Whole(11),
    Other(false),
    Draw(12),
];

/// The passes over the target among [`STEPS`], and the runs they make.
const PASSES: u64 = 17;
const RUNS: u64 = 7;

/// The state that draws a rectangle (`tob_rect`) of the color in fragment buffer 0
/// (`tob_solid`) with no attachment.
pub(super) fn attachmentless(device: &Device) -> RenderPipelineState {
    let library = library(device, "engine/tob_renders.ll");
    let function = |name: &str| {
        let function = library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(function("tob_solid"));
    descriptor.color_attachments[0] = None;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// What the sequence runs over: the target, its buffer, the other texture, and the states.
struct Scene<'a> {
    texture: &'a Texture,
    buffer: &'a Buffer,
    other: &'a Texture,
    blending: &'a RenderPipelineState,
    attachmentless: &'a RenderPipelineState,
}

/// A `Private` texture of `format`, `width` x [`H`] texels, with `usage`.
fn private_texture(
    device: &Device,
    format: PixelFormat,
    width: u64,
    usage: TextureUsage,
) -> Texture {
    let mut d = TextureDescriptor::texture_2d(format, width, H, false);
    d.usage = usage;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d).unwrap()
}

/// Encode a pass over `texture` beside `depth`, loading and storing it, that blends the
/// rectangle `corners` of `rgba` with `state`; binding `named` as fragment buffer 1, which the
/// function does not read.
fn pass(
    command_buffer: &CommandBuffer,
    (texture, depth): (&Texture, &Texture),
    state: &RenderPipelineState,
    (corners, rgba): ([f32; 4], [f32; 4]),
    named: Option<&Buffer>,
) {
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(texture.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    descriptor.depth_attachment.texture = Some(depth.clone());
    descriptor.depth_attachment.load_action = LoadAction::Clear;
    descriptor.depth_attachment.store_action = StoreAction::Store;
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    (encoder.set_vertex_bytes(&bytes_of(&corners, f32::to_le_bytes), 0)).unwrap();
    (encoder.set_fragment_bytes(&bytes_of(&rgba, f32::to_le_bytes), 0)).unwrap();
    if let Some(named) = named {
        encoder.set_fragment_buffer(Some(named), 0, 1).unwrap();
    }
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    encoder.end_encoding().unwrap();
}

/// [`STEPS`] over `over` in one command buffer, committed and completed; a blit between every
/// two encoders where `apart`, copying 4 bytes between buffers of its own. The texels each
/// [`Read`] copied, packed, and the device's work counts as it completed.
fn sequence(queue: &CommandQueue, over: &Scene<'_>, apart: bool) -> (Vec<Vec<u8>>, WorkCounts) {
    let device = queue.device();
    let width = over.texture.width();
    let packed = width * TEXEL;
    let depth = private_texture(
        device,
        PixelFormat::Depth32Float,
        width,
        TextureUsage::RENDER_TARGET,
    );
    let unread = device.new_buffer(256, ResourceOptions::private()).unwrap();
    let (from, into) = (staged(device, &[1; 4]), staged(device, &[0; 4]));
    let event = device.new_event().unwrap();
    let reads = (STEPS.iter())
        .filter(|step| matches!(step, Read))
        .map(|_| (device.new_buffer(packed * H, ResourceOptions::shared())).unwrap())
        .collect::<Vec<_>>();
    let mut read = reads.iter();
    let command_buffer = queue.command_buffer().unwrap();
    let whole = |i: usize| ([-1.0, -1.0, 1.0, 1.0], drawn(i).1);
    for (at, step) in STEPS.iter().enumerate() {
        if apart && at > 0 {
            let mut encoder = command_buffer.blit_command_encoder().unwrap();
            (encoder.copy_from_buffer_to_buffer(&from, 0, &into, 0, 4)).unwrap();
            encoder.end_encoding().unwrap();
        }
        let target = (over.texture, &depth);
        match *step {
            Draw(i) => pass(&command_buffer, target, over.blending, drawn(i), None),
            Whole(i) => pass(&command_buffer, target, over.blending, whole(i), None),
            Lone => {
                let mut lone = RenderPassDescriptor::new();
                lone.render_target_width = 1;
                lone.render_target_height = 1;
                lone.default_raster_sample_count = 1;
                let mut encoder = command_buffer.render_command_encoder(&lone).unwrap();
                encoder
                    .set_render_pipeline_state(over.attachmentless)
                    .unwrap();
                let corners = [-1.0f32, -1.0, 1.0, 1.0];
                (encoder.set_vertex_bytes(&bytes_of(&corners, f32::to_le_bytes), 0)).unwrap();
                (encoder.set_fragment_bytes(&bytes_of(&[0.0f32; 4], f32::to_le_bytes), 0)).unwrap();
                (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
                encoder.end_encoding().unwrap();
            }
            Other(names) => {
                let named = match names {
                    true => over.buffer,
                    false => &unread,
                };
                let other = (over.other, &depth);
                pass(
                    &command_buffer,
                    other,
                    over.blending,
                    drawn(at),
                    Some(named),
                );
            }
            Read => {
                let mut encoder = command_buffer.blit_command_encoder().unwrap();
                let origin = Origin::new(0, 0, 0);
                let size = Size::new(width, H, 1);
                let out = read.next().unwrap();
                (encoder.copy_from_texture_to_buffer(
                    over.texture,
                    0,
                    0,
                    origin,
                    size,
                    out,
                    0,
                    packed,
                    packed * H,
                    BlitOption::None,
                ))
                .unwrap();
                encoder.end_encoding().unwrap();
            }
            Signal => {
                (command_buffer.encode_signal_event(EventRef::from(&event), at as u64)).unwrap()
            }
        }
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let reads = (reads.iter())
        .map(|out| super::merge_scratch::read(out, packed * H))
        .collect();
    (reads, device.work_counts())
}

/// The texels of `texture`, of `width` x [`H`], packed: read through a blit.
fn texels_of(queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let (width, packed) = (texture.width(), texture.width() * TEXEL);
    let out = (queue.device())
        .new_buffer(packed * H, ResourceOptions::shared())
        .unwrap();
    blit(queue, |b| {
        (b.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(width, H, 1),
            &out,
            0,
            packed,
            packed * H,
            BlitOption::None,
        ))
        .unwrap()
    });
    super::merge_scratch::read(&out, packed * H)
}

/// [`STEPS`] over a target over `over`'s buffer of `width` texels a row, `row` bytes apart, the
/// rule forced, as runs and with every pass apart, against a private texture's: the same texels
/// read mid-sequence and left at its end, and one merge a run (every pass's where apart), each
/// later pass of a run taking its run's merge over.
fn runs_hold_their_texels(over: Over, (width, row): (u64, u64)) {
    let cache = Scratch::new(&format!("linear-runs-{over:?}-{width}"));
    let host = Host::new(row * H);
    let device = settled(device(&cache.0));
    let queue = device.new_command_queue().unwrap();
    let blending = blending(&device);
    let attachmentless = attachmentless(&device);
    let packed = width * TEXEL;
    let target_usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let other = private_texture(&device, PixelFormat::RGBA8Unorm, width, target_usage);
    // The private texture of the target's first texels.
    let reference = private_texture(&device, PixelFormat::RGBA8Unorm, width, target_usage);
    let source = staged(&device, &first_bytes(width, packed));
    blit(&queue, |b| {
        (b.copy_from_buffer_to_texture(
            &source,
            0,
            packed,
            packed * H,
            Size::new(width, H, 1),
            &reference,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        ))
        .unwrap()
    });
    let unused = device.new_buffer(256, ResourceOptions::private()).unwrap();
    let on_reference = Scene {
        texture: &reference,
        buffer: &unused,
        other: &other,
        blending: &blending,
        attachmentless: &attachmentless,
    };
    let (expected, _) = sequence(&queue, &on_reference, false);
    let left = texels_of(&queue, &reference);
    let forced = &device.shared().linear_rule_forced;
    for apart in [false, true] {
        let Some((buffer, texture)) = target(&queue, over, (width, row), &host) else {
            return;
        };
        // Over host pages, a linear image in place where the device's check passed for render
        // targets there (`in_place.rs`), else a shadow, which `tests::shadow_runs` covers.
        if over == Over::HostPages && !texture.linear_image().unwrap() {
            println!("LINEAR-RUNS {over:?}: the target is a shadow, its runs shadow_runs'");
            return;
        }
        assert!(
            texture.linear_image().unwrap(),
            "the target is a linear image"
        );
        let on_target = Scene {
            texture: &texture,
            buffer: &buffer,
            other: &other,
            blending: &blending,
            attachmentless: &attachmentless,
        };
        let before = counts_alone(&device);
        forced.store(true, Ordering::Relaxed);
        let (reads, after) = sequence(&queue, &on_target, apart);
        forced.store(false, Ordering::Relaxed);
        let delta = |of: fn(&WorkCounts) -> u64| of(&after) - of(&before);
        let through = delta(|c| c.linear_passes_through_copies);
        let recorded = delta(|c| c.linear_merges_recorded);
        let taken_over = delta(|c| c.linear_merges_taken_over);
        println!(
            "LINEAR-RUNS {over:?} {width}x{H}, rows of {row}, apart {apart}: {through} passes \
             through copies, linear merges {recorded} recorded and {taken_over} taken over"
        );
        assert_eq!(
            through, PASSES,
            "every pass over the target renders it apart"
        );
        let runs = match apart {
            true => PASSES,
            false => RUNS,
        };
        assert_eq!(recorded, runs, "one merge a run");
        assert_eq!(
            taken_over,
            PASSES - runs,
            "every later pass of a run takes its merge over"
        );
        for (at, (got, expected)) in reads.iter().zip(&expected).enumerate() {
            let differ = (0..got.len()).filter(|&i| got[i] != expected[i]).count();
            assert_eq!(
                differ, 0,
                "read {at}: the texels differ from the private texture's"
            );
        }
        let got = texels(&bytes(&queue, over, &buffer, row * H), width, row);
        let differ = (0..got.len()).filter(|&i| got[i] != left[i]).count();
        assert_eq!(
            differ, 0,
            "the texels left differ from the private texture's"
        );
        drop((texture, buffer));
    }
}

/// Over a `Shared` buffer, rows packed (1024 bytes).
#[test]
fn runs_over_a_shared_buffer_hold_their_texels() {
    runs_hold_their_texels(Over::Shared, (256, 1024));
}

/// Over a `Private` buffer, packed rows of 1024 bytes.
#[test]
fn runs_over_a_private_buffer_hold_their_texels() {
    runs_hold_their_texels(Over::Private, (256, 1024));
}

/// Over a no-copy buffer's host pages, packed rows of 1200 bytes (an IOSurface's), where the
/// target is a linear image in place.
#[test]
fn runs_over_host_pages_hold_their_texels() {
    runs_hold_their_texels(Over::HostPages, (300, 1200));
}

/// The passes of [`a_rebound_run_over_host_pages_is_encoded_again_whole`].
const REBOUND_PASSES: usize = 6;

/// Passes over a render target in place over a no-copy buffer's host pages (rows of 1200 bytes),
/// the rule forced, behind a wait for a shared event only the host signals: one run, merged once
/// (`pass_runs.rs`). The buffer is rebound over other pages holding the same first bytes before
/// the signal: the rebind encodes every pass again (`command_buffer/reencode.rs`), each leaving
/// the run and merging alone (`pass_runs.rs`'s **Encoded again**), and the new pages hold the
/// texels a private texture of the same first texels holds after the same passes.
#[test]
fn a_rebound_run_over_host_pages_is_encoded_again_whole() {
    let (width, row) = (300, 1200);
    let (host, moved_to) = (Host::new(row * H), Host::new(row * H));
    let cache = Scratch::new("linear-runs-rebound");
    let device = settled(device(&cache.0));
    let queue = device.new_command_queue().unwrap();
    let blending = blending(&device);
    let packed = width * TEXEL;
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let depth_usage = TextureUsage::RENDER_TARGET;
    let depth = private_texture(&device, PixelFormat::Depth32Float, width, depth_usage);
    let reference = private_texture(&device, PixelFormat::RGBA8Unorm, width, usage);
    let source = staged(&device, &first_bytes(width, packed));
    blit(&queue, |b| {
        (b.copy_from_buffer_to_texture(
            &source,
            0,
            packed,
            packed * H,
            Size::new(width, H, 1),
            &reference,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        ))
        .unwrap()
    });
    let command_buffer = queue.command_buffer().unwrap();
    for i in 0..REBOUND_PASSES {
        pass(
            &command_buffer,
            (&reference, &depth),
            &blending,
            drawn(i),
            None,
        );
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let left = texels_of(&queue, &reference);
    let Some((buffer, texture)) = target(&queue, Over::HostPages, (width, row), &host) else {
        return;
    };
    if !texture.linear_image().unwrap() {
        println!("LINEAR-RUNS rebound: the target is a shadow, its runs shadow_runs'");
        return;
    }
    moved_to.write(&first_bytes(width, row));
    let gate = device.new_shared_event().unwrap();
    let forced = &device.shared().linear_rule_forced;
    let before = counts_alone(&device);
    forced.store(true, Ordering::Relaxed);
    let command_buffer = queue.command_buffer().unwrap();
    (command_buffer.encode_wait_for_event(EventRef::Shared(&gate), 1)).unwrap();
    for i in 0..REBOUND_PASSES {
        pass(
            &command_buffer,
            (&texture, &depth),
            &blending,
            drawn(i),
            None,
        );
    }
    command_buffer.commit().unwrap();
    let encoded = device.work_counts();
    assert_eq!(buffer.detach_backing(), Ok(true), "the detach");
    // SAFETY: the pages outlive every buffer over them, which alone uses them while it lives.
    unsafe { buffer.rebind_bytes_no_copy(moved_to.memory(), None, None) }.unwrap();
    let rebound = device.work_counts();
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let after = device.work_counts();
    forced.store(false, Ordering::Relaxed);
    let delta = |of: fn(&WorkCounts) -> u64, (a, b): (&WorkCounts, &WorkCounts)| of(b) - of(a);
    let first_taken = delta(|c| c.linear_merges_taken_over, (&before, &encoded));
    let again = delta(|c| c.encoders_reencoded, (&encoded, &rebound));
    let refused = delta(|c| c.reencodes_refused, (&encoded, &rebound));
    let recorded = delta(|c| c.linear_merges_recorded, (&before, &after));
    println!(
        "LINEAR-RUNS rebound: {:?} {:?}; first encoding's merges taken over {first_taken}; \
         encoders encoded again {again}, refused {refused}; linear merges recorded {recorded}",
        command_buffer.status(),
        command_buffer.error(),
    );
    assert_eq!(command_buffer.error(), None);
    assert_eq!(
        first_taken,
        REBOUND_PASSES as u64 - 1,
        "the passes form one run as first encoded"
    );
    assert_eq!(
        (again, refused),
        (REBOUND_PASSES as u64, 0),
        "the rebind encodes every pass of the run again"
    );
    drop(texture);
    let got = texels(&moved_to.bytes(row * H), width, row);
    let differ = (0..got.len()).filter(|&i| got[i] != left[i]).count();
    assert_eq!(
        differ, 0,
        "the new pages' texels differ from the private texture's"
    );
    drop(buffer);
}

/// The passes of [`passes_reading_host_pages_form_one_run`]: a WebGPU frame split at every
/// indexed draw (the `stencilMask` sample's first: 706).
const CHUNK_PASSES: usize = 706;

/// The bytes between two passes' data in [`passes_reading_host_pages_form_one_run`]'s chunk.
const CHUNK_STRIDE: u64 = 256;

/// Where a pass of [`pass_from`] reads its rectangle and color.
#[derive(Clone, Copy)]
enum Data<'a> {
    /// Its own bytes (`setVertexBytes`, `setFragmentBytes`).
    Bytes(([f32; 4], [f32; 4])),
    /// A buffer's, at an offset and 16 bytes after (as a consumer's `setVertexBytes` and
    /// `setFragmentBytes` land in an upload chunk of its own).
    Chunk(&'a Buffer, u64),
}

/// Encode a pass over `texture` beside `depth`, loading and storing it, that blends with `state`
/// the rectangle and the color `data` gives, inside `scissor` where it has one.
fn pass_from(
    command_buffer: &CommandBuffer,
    (texture, depth): (&Texture, &Texture),
    state: &RenderPipelineState,
    data: Data<'_>,
    scissor: Option<ScissorRect>,
) {
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(texture.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    descriptor.depth_attachment.texture = Some(depth.clone());
    descriptor.depth_attachment.load_action = LoadAction::Clear;
    descriptor.depth_attachment.store_action = StoreAction::Store;
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    match data {
        Data::Bytes((corners, rgba)) => {
            (encoder.set_vertex_bytes(&bytes_of(&corners, f32::to_le_bytes), 0)).unwrap();
            (encoder.set_fragment_bytes(&bytes_of(&rgba, f32::to_le_bytes), 0)).unwrap();
        }
        Data::Chunk(chunk, at) => {
            encoder.set_vertex_buffer(Some(chunk), at, 0).unwrap();
            encoder
                .set_fragment_buffer(Some(chunk), at + 16, 0)
                .unwrap();
        }
    }
    if let Some(rect) = scissor {
        encoder.set_scissor_rect(rect).unwrap();
    }
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    encoder.end_encoding().unwrap();
}

/// [`CHUNK_PASSES`] passes in one command buffer over a target over a `Private` buffer (rows of
/// 2048 bytes), the rule forced, each reading its rectangle and color from a no-copy buffer over
/// host pages, at an offset of its own (a consumer's upload chunk in its guest RAM): one run,
/// merged once (`pass_runs.rs`), every later pass taking its merge over; the texels those a
/// private texture of the same first texels holds after the same passes.
#[test]
fn passes_reading_host_pages_form_one_run() {
    let (width, row) = (512, 2048);
    let packed = width * TEXEL;
    let (host, chunk_pages) = (
        Host::new(row * H),
        Host::new(CHUNK_STRIDE * CHUNK_PASSES as u64),
    );
    let cache = Scratch::new("linear-runs-chunk");
    let device = settled(device(&cache.0));
    let queue = device.new_command_queue().unwrap();
    let blending = blending(&device);
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let depth = private_texture(
        &device,
        PixelFormat::Depth32Float,
        width,
        TextureUsage::RENDER_TARGET,
    );
    let mut data = vec![0u8; (CHUNK_STRIDE * CHUNK_PASSES as u64) as usize];
    for i in 0..CHUNK_PASSES {
        let (corners, rgba) = drawn(i);
        let at = i * CHUNK_STRIDE as usize;
        data[at..at + 16].copy_from_slice(&bytes_of(&corners, f32::to_le_bytes));
        data[at + 16..at + 32].copy_from_slice(&bytes_of(&rgba, f32::to_le_bytes));
    }
    chunk_pages.write(&data);
    // SAFETY: the pages outlive the buffer, which alone uses them while it lives.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            chunk_pages.memory(),
            (CHUNK_STRIDE * CHUNK_PASSES as u64).next_multiple_of(64 << 10),
            ResourceOptions::default(),
            None,
        )
    };
    let chunk = match made {
        Ok(chunk) => chunk,
        Err(crate::Error::Unsupported(gap)) => {
            println!("LINEAR-RUNS chunk: the device imports no host memory: {gap:?}");
            return;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    let encode = |texture: &Texture| {
        let command_buffer = queue.command_buffer().unwrap();
        let started = std::time::Instant::now();
        for i in 0..CHUNK_PASSES {
            let at = (i as u64) * CHUNK_STRIDE;
            let data = Data::Chunk(&chunk, at);
            pass_from(&command_buffer, (texture, &depth), &blending, data, None);
        }
        let encoding = started.elapsed();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
        (encoding, started.elapsed(), gpu)
    };
    let reference = private_texture(&device, PixelFormat::RGBA8Unorm, width, usage);
    let source = staged(&device, &first_bytes(width, packed));
    blit(&queue, |b| {
        (b.copy_from_buffer_to_texture(
            &source,
            0,
            packed,
            packed * H,
            Size::new(width, H, 1),
            &reference,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        ))
        .unwrap()
    });
    let (encoding, whole, gpu) = encode(&reference);
    println!(
        "LINEAR-RUNS chunk on {}: private texture, {CHUNK_PASSES} passes: encoding {encoding:?}, \
         commit to completed {whole:?}, GPU {:.3} ms",
        device.name(),
        gpu * 1e3
    );
    let left = texels_of(&queue, &reference);
    let Some((buffer, texture)) = target(&queue, Over::Private, (width, row), &host) else {
        return;
    };
    assert!(
        texture.linear_image().unwrap(),
        "the target is a linear image"
    );
    let forced = &device.shared().linear_rule_forced;
    let before = counts_alone(&device);
    forced.store(true, Ordering::Relaxed);
    let (encoding, whole, gpu) = encode(&texture);
    let after = device.work_counts();
    forced.store(false, Ordering::Relaxed);
    let delta = |of: fn(&WorkCounts) -> u64| of(&after) - of(&before);
    let through = delta(|c| c.linear_passes_through_copies);
    let recorded = delta(|c| c.linear_merges_recorded);
    let taken_over = delta(|c| c.linear_merges_taken_over);
    let reused = delta(|c| c.linear_merge_scratches_reused);
    println!(
        "LINEAR-RUNS chunk on {}: {CHUNK_PASSES} passes, {through} through copies, linear merges \
         {recorded} recorded and {taken_over} taken over, scratches {reused} reused: encoding \
         {encoding:?}, commit to completed {whole:?}, GPU {:.3} ms",
        device.name(),
        gpu * 1e3
    );
    let got = texels(&bytes(&queue, Over::Private, &buffer, row * H), width, row);
    let differ = (0..got.len()).filter(|&i| got[i] != left[i]).count();
    assert_eq!(differ, 0, "the texels differ from the private texture's");
    assert_eq!(
        through, CHUNK_PASSES as u64,
        "every pass renders the target apart"
    );
    assert!(
        recorded <= 2,
        "the passes merge once a run: {recorded} merges"
    );
    assert_eq!(
        recorded + taken_over,
        CHUNK_PASSES as u64,
        "every later pass takes its run's merge over"
    );
    drop((texture, buffer, chunk));
}

/// The rectangles ([`drawn`]) of [`a_pass_of_a_run_encoded_again_alone_reloads_the_run`]'s
/// passes, whether each reads them from the chunk over host pages the test rebinds, and its
/// scissor's quarter of the target (`[column, row]` of the four), where it has one.
const RELOADED: [(usize, bool, Option<[u64; 2]>); 6] = [
    (0, false, Some([0, 0])),
    (5, true, Some([1, 0])),
    (1, false, None),
    (2, true, None),
    (3, false, None),
    (6, false, None),
];

/// Six passes over a target over a `Private` buffer (rows of 1024 bytes), the rule forced, behind
/// a wait for a shared event only the host signals: the first, scissored to the target's top left
/// quarter, opens a run over it; the second's scissor, the top right quarter, reaches past it and
/// opens one over the whole target (`pass_runs.rs`'s **A run**), which the other four, not
/// scissored, go on with. The second and the fourth read their rectangle and color from a no-copy
/// buffer over host pages, rebound over other pages holding the same bytes before the signal: the
/// rebind encodes those two again alone (`command_buffer/reencode.rs`), each leaving its run,
/// merging after itself and copying the run's box in again and keeping it for the passes after
/// it (`pass_runs.rs`'s **Encoded again**); the texels are those a private texture holds after
/// the same passes. Without the copy in again, the third pass would blend over texels of the
/// bottom half that no copy in wrote.
#[test]
fn a_pass_of_a_run_encoded_again_alone_reloads_the_run() {
    let (width, row) = (256, 1024);
    let packed = width * TEXEL;
    let chunk_bytes = (2 * CHUNK_STRIDE).next_multiple_of(64 << 10);
    let (host, chunk_pages, moved_to) = (
        Host::new(row * H),
        Host::new(chunk_bytes),
        Host::new(chunk_bytes),
    );
    let cache = Scratch::new("linear-runs-reloaded");
    let device = settled(device(&cache.0));
    let queue = device.new_command_queue().unwrap();
    let blending = blending(&device);
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let depth_usage = TextureUsage::RENDER_TARGET;
    let depth = private_texture(&device, PixelFormat::Depth32Float, width, depth_usage);
    // The chunk: the readers' rectangles and colors, one stride apart, in their order.
    let mut data = vec![0u8; chunk_bytes as usize];
    let readers = RELOADED.iter().filter(|(_, reads, _)| *reads);
    for (at, &(i, _, _)) in readers.enumerate() {
        let (corners, rgba) = drawn(i);
        let at = at * CHUNK_STRIDE as usize;
        data[at..at + 16].copy_from_slice(&bytes_of(&corners, f32::to_le_bytes));
        data[at + 16..at + 32].copy_from_slice(&bytes_of(&rgba, f32::to_le_bytes));
    }
    chunk_pages.write(&data);
    moved_to.write(&data);
    // SAFETY: the pages outlive the buffer, which alone uses them while it lives.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            chunk_pages.memory(),
            chunk_bytes,
            ResourceOptions::default(),
            None,
        )
    };
    let chunk = match made {
        Ok(chunk) => chunk,
        Err(crate::Error::Unsupported(gap)) => {
            println!("LINEAR-RUNS reloaded: the device imports no host memory: {gap:?}");
            return;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    // The passes over `texture`: from the chunk where they read it, else their own bytes.
    let encode = |command_buffer: &CommandBuffer, texture: &Texture| {
        let mut read = 0;
        for &(i, reads, quarter) in &RELOADED {
            let data = match reads {
                true => {
                    read += 1;
                    Data::Chunk(&chunk, (read - 1) * CHUNK_STRIDE)
                }
                false => Data::Bytes(drawn(i)),
            };
            let scissor = quarter.map(|[column, row]| ScissorRect {
                x: column * width / 2,
                y: row * H / 2,
                width: width / 2,
                height: H / 2,
            });
            pass_from(command_buffer, (texture, &depth), &blending, data, scissor);
        }
    };
    let reference = private_texture(&device, PixelFormat::RGBA8Unorm, width, usage);
    let source = staged(&device, &first_bytes(width, packed));
    blit(&queue, |b| {
        (b.copy_from_buffer_to_texture(
            &source,
            0,
            packed,
            packed * H,
            Size::new(width, H, 1),
            &reference,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        ))
        .unwrap()
    });
    let command_buffer = queue.command_buffer().unwrap();
    encode(&command_buffer, &reference);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let left = texels_of(&queue, &reference);
    let Some((buffer, texture)) = target(&queue, Over::Private, (width, row), &host) else {
        return;
    };
    assert!(
        texture.linear_image().unwrap(),
        "the target is a linear image"
    );
    let gate = device.new_shared_event().unwrap();
    let forced = &device.shared().linear_rule_forced;
    let before = counts_alone(&device);
    forced.store(true, Ordering::Relaxed);
    let command_buffer = queue.command_buffer().unwrap();
    (command_buffer.encode_wait_for_event(EventRef::Shared(&gate), 1)).unwrap();
    encode(&command_buffer, &texture);
    command_buffer.commit().unwrap();
    let encoded = device.work_counts();
    assert_eq!(chunk.detach_backing(), Ok(true), "the detach");
    // SAFETY: the pages outlive every buffer over them, which alone uses them while it lives.
    unsafe { chunk.rebind_bytes_no_copy(moved_to.memory(), None, None) }.unwrap();
    let rebound = device.work_counts();
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let after = device.work_counts();
    forced.store(false, Ordering::Relaxed);
    let delta = |of: fn(&WorkCounts) -> u64, (a, b): (&WorkCounts, &WorkCounts)| of(b) - of(a);
    let first_taken = delta(|c| c.linear_merges_taken_over, (&before, &encoded));
    let again = delta(|c| c.encoders_reencoded, (&encoded, &rebound));
    let refused = delta(|c| c.reencodes_refused, (&encoded, &rebound));
    let reloaded = delta(|c| c.linear_runs_reloaded, (&encoded, &rebound));
    let recorded = delta(|c| c.linear_merges_recorded, (&before, &after));
    let got = texels(&bytes(&queue, Over::Private, &buffer, row * H), width, row);
    let differ = (0..got.len()).filter(|&i| got[i] != left[i]).count();
    println!(
        "LINEAR-RUNS reloaded on {}: {:?} {:?}; first encoding's merges taken over \
         {first_taken}; encoders encoded again {again}, refused {refused}, runs reloaded \
         {reloaded}; linear merges recorded {recorded}; {differ} of {} texel bytes differ from \
         the private texture's",
        device.name(),
        command_buffer.status(),
        command_buffer.error(),
        got.len(),
    );
    assert_eq!(command_buffer.error(), None);
    assert_eq!(differ, 0, "the texels differ from the private texture's");
    assert_eq!(first_taken, 4, "the passes form two runs as first encoded");
    assert_eq!(
        (again, refused, reloaded),
        (2, 0, 2),
        "the rebind encodes the two readers again, each reloading its run"
    );
    // The first run's, the two readers' own, the merge the fourth had taken over, and the
    // second run's last.
    assert_eq!(recorded, 5, "the merges recorded");
    drop((texture, buffer, chunk));
}

/// The passes of a checked frame after its clear: the measured frame's (the `stencilMask`
/// sample's first, as [`CHUNK_PASSES`]).
const FRAME_PASSES: usize = CHUNK_PASSES;

/// What a checked frame's passes hold beside their target ([`checked_frame`]).
#[derive(Clone, Copy, Debug)]
struct Shape {
    /// A `Depth32Float_Stencil8` attachment, its stencil loaded and stored by each pass after the
    /// clear; else a `Depth32Float` one.
    stencil: bool,
    /// A 1x1 encoder of no attachment before each pass, one draw of a state that rasterizes
    /// nothing (WebKit's check of a draw's indices).
    checks: bool,
    /// Each pass's rectangle and color from a no-copy buffer over host pages, at an offset of its
    /// own (a consumer's upload chunk in its guest RAM); else its own bytes (`setVertexBytes`,
    /// `setFragmentBytes`, which land in the command buffer's upload arena, `upload.rs`).
    chunk: bool,
}

/// The states of a checked frame: the blending one beside each depth format, the check's, and
/// the passes' depth and stencil state (depth written, the test and the stencil `Always`).
struct FrameStates {
    depth: RenderPipelineState,
    stencil: RenderPipelineState,
    check: RenderPipelineState,
    depth_stencil: crate::DepthStencilState,
}

impl FrameStates {
    fn new(device: &Device) -> FrameStates {
        let library = library(device, "engine/ob3_index_buffer.ll");
        let function = library.new_function_with_name("ob3_index_buffer").unwrap();
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        descriptor.rasterization_enabled = false;
        let check = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let combined = PixelFormat::Depth32FloatStencil8;
        let mut descriptor = metal2vulkan::metal::state::DepthStencilDescriptor::default();
        descriptor.depth_write_enabled = true;
        FrameStates {
            depth: blending_beside(device, PixelFormat::Depth32Float, None),
            stencil: blending_beside(device, combined, Some(combined)),
            check,
            depth_stencil: device.new_depth_stencil_state(&descriptor).unwrap(),
        }
    }
}

/// One command buffer over `texture` of `shape`, as a WebGPU frame split at every indexed draw
/// renders: a pass clearing it and its depth (and stencil), then [`FRAME_PASSES`] passes each
/// loading and storing it (and the stencil), clearing and storing the depth, and blending
/// rectangle `i` (`drawn`), from `chunk` at `i` strides where `shape.chunk`; committed and
/// completed. Its GPU time, in seconds.
fn checked_frame(
    queue: &CommandQueue,
    texture: &Texture,
    states: &FrameStates,
    shape: Shape,
    chunk: &Buffer,
) -> f64 {
    let device = queue.device();
    let format = match shape.stencil {
        true => PixelFormat::Depth32FloatStencil8,
        false => PixelFormat::Depth32Float,
    };
    let depth = private_texture(device, format, texture.width(), TextureUsage::RENDER_TARGET);
    let checked = staged(device, &[0; 32]);
    let descriptor = |load: LoadAction| {
        let mut descriptor = RenderPassDescriptor::new();
        let color = &mut descriptor.color_attachments[0];
        color.texture = Some(texture.clone());
        color.load_action = load;
        color.store_action = StoreAction::Store;
        color.clear_color = crate::ClearColor::new(0.0, 0.0, 0.0, 1.0);
        descriptor.depth_attachment.texture = Some(depth.clone());
        descriptor.depth_attachment.load_action = LoadAction::Clear;
        descriptor.depth_attachment.store_action = StoreAction::Store;
        if shape.stencil {
            descriptor.stencil_attachment.texture = Some(depth.clone());
            descriptor.stencil_attachment.load_action = load;
            descriptor.stencil_attachment.store_action = StoreAction::Store;
        }
        descriptor
    };
    let state = match shape.stencil {
        true => &states.stencil,
        false => &states.depth,
    };
    let command_buffer = queue.command_buffer().unwrap();
    let encoder = command_buffer
        .render_command_encoder(&descriptor(LoadAction::Clear))
        .unwrap();
    encoder.end_encoding().unwrap();
    let mut check = RenderPassDescriptor::new();
    check.render_target_width = 1;
    check.render_target_height = 1;
    check.default_raster_sample_count = 1;
    for i in 0..FRAME_PASSES {
        if shape.checks {
            let mut encoder = command_buffer.render_command_encoder(&check).unwrap();
            encoder.set_render_pipeline_state(&states.check).unwrap();
            encoder.set_vertex_buffer(Some(&checked), 0, 0).unwrap();
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).unwrap();
            encoder.end_encoding().unwrap();
        }
        let mut encoder = command_buffer
            .render_command_encoder(&descriptor(LoadAction::Load))
            .unwrap();
        encoder.set_render_pipeline_state(state).unwrap();
        encoder
            .set_depth_stencil_state(&states.depth_stencil)
            .unwrap();
        match shape.chunk {
            true => {
                let at = i as u64 * CHUNK_STRIDE;
                encoder.set_vertex_buffer(Some(chunk), at, 0).unwrap();
                encoder
                    .set_fragment_buffer(Some(chunk), at + 16, 0)
                    .unwrap();
            }
            false => {
                let (corners, rgba) = drawn(i);
                (encoder.set_vertex_bytes(&bytes_of(&corners, f32::to_le_bytes), 0)).unwrap();
                (encoder.set_fragment_bytes(&bytes_of(&rgba, f32::to_le_bytes), 0)).unwrap();
            }
        }
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        encoder.end_encoding().unwrap();
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap()
}

/// Checked frames ([`checked_frame`]) of every [`Shape`] over a target over a `Private` buffer
/// (rows of 2048 bytes), the rule forced, as the consumer's `many_passes` renders its frame: the
/// clear and every pass render the target apart, and the passes after the clear form one run,
/// merged once (`pass_runs.rs`), every later pass taking its merge over, whether a pass's data
/// lies in the command buffer's upload arena or in a no-copy buffer over host pages, beside a
/// stencil it loads or none, a check encoder between every two passes or none; the texels those
/// a private texture holds after the same frame.
#[test]
fn checked_frames_merge_once() {
    let (width, row) = (512, 2048);
    let chunk_bytes = (CHUNK_STRIDE * FRAME_PASSES as u64).next_multiple_of(64 << 10);
    let (host, chunk_pages) = (Host::new(row * H), Host::new(chunk_bytes));
    let cache = Scratch::new("linear-runs-checked");
    let device = device(&cache.0);
    let queue = device.new_command_queue().unwrap();
    let states = FrameStates::new(&device);
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let mut data = vec![0u8; chunk_bytes as usize];
    for i in 0..FRAME_PASSES {
        let (corners, rgba) = drawn(i);
        let at = i * CHUNK_STRIDE as usize;
        data[at..at + 16].copy_from_slice(&bytes_of(&corners, f32::to_le_bytes));
        data[at + 16..at + 32].copy_from_slice(&bytes_of(&rgba, f32::to_le_bytes));
    }
    chunk_pages.write(&data);
    // SAFETY: the pages outlive the buffer, which alone uses them while it lives.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            chunk_pages.memory(),
            chunk_bytes,
            ResourceOptions::default(),
            None,
        )
    };
    let chunk = match made {
        Ok(chunk) => chunk,
        Err(crate::Error::Unsupported(gap)) => {
            println!("LINEAR-RUNS checked: the device imports no host memory: {gap:?}");
            return;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    let reference = private_texture(&device, PixelFormat::RGBA8Unorm, width, usage);
    let passes = FRAME_PASSES as u64;
    let mut wrong = Vec::new();
    for bits in 0..8u8 {
        let shape = Shape {
            stencil: bits & 1 != 0,
            checks: bits & 2 != 0,
            chunk: bits & 4 != 0,
        };
        let private_gpu = checked_frame(&queue, &reference, &states, shape, &chunk);
        let left = texels_of(&queue, &reference);
        let Some((buffer, texture)) = target(&queue, Over::Private, (width, row), &host) else {
            return;
        };
        assert!(
            texture.linear_image().unwrap(),
            "the target is a linear image"
        );
        let forced = &device.shared().linear_rule_forced;
        forced.store(true, Ordering::Relaxed);
        let before = device.work_counts();
        let gpu = checked_frame(&queue, &texture, &states, shape, &chunk);
        let after = device.work_counts();
        forced.store(false, Ordering::Relaxed);
        let delta = |of: fn(&WorkCounts) -> u64| of(&after) - of(&before);
        let through = delta(|c| c.linear_passes_through_copies);
        let recorded = delta(|c| c.linear_merges_recorded);
        let taken_over = delta(|c| c.linear_merges_taken_over);
        let got = texels(&bytes(&queue, Over::Private, &buffer, row * H), width, row);
        let differ = (0..got.len()).filter(|&i| got[i] != left[i]).count();
        println!(
            "LINEAR-RUNS checked on {}: {shape:?}: {through} passes through copies, linear \
             merges {recorded} recorded and {taken_over} taken over; {differ} texel bytes \
             differ; GPU {:.3} ms, {:.3} ms over a private texture",
            device.name(),
            gpu * 1e3,
            private_gpu * 1e3,
        );
        if (through, recorded, taken_over, differ) != (passes + 1, 1, passes - 1, 0) {
            wrong.push((shape, through, recorded, taken_over, differ));
        }
        drop((texture, buffer));
    }
    assert!(
        wrong.is_empty(),
        "every pass renders the target apart, the passes after the clear merge once as one run, \
         and the texels are a private texture's; (shape, passes through copies, merges recorded, \
         taken over, texel bytes differing): {wrong:?}"
    );
    drop(chunk);
}
