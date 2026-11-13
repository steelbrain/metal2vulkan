//! A draw that reads, at its own pixels through a scissor, the texture its own render pass renders
//! (`texture/attached_reads.rs`), over a surface plane of a `Private` buffer the device made: the
//! texture's image aliases the bytes, so the pass renders the truth of the read's shadow in place,
//! apart from the shadow. Its image then holds the truth whole nowhere a command buffer's end
//! could leave standing (`shadow_standing.rs`), so each command buffer's first copy in of the
//! shadow moves the texels its reads take, the scissor's a pixel wider each way
//! (`ShadowUse::loads`), not every texel for a standing (`shadow.rs`'s `writes_truth_apart`).
//!
//! Each pass checks both halves: the bytes the pass leaves, which hold the texels the read took
//! (Metal's answers: the texels from before the pass at a read before any draw, the draw's after
//! a barrier over the render targets, `metal2vulkan-conformance`'s `pass-start-barrier-*` rows),
//! and the texels the copies in moved (`WorkCounts::shadow_texels_copied_in`). A copy cut wrong
//! leaves the read's texels wrong; a copy of every texel leaves the counts wrong. Where the device
//! reads the texture through a snapshot inside the pass (`render/snapshot.rs`: a texture over a
//! buffer's bytes takes it at any area, with a barrier or without), the read copies nothing in,
//! and only the bytes are checked there; a device without the snapshot route (lavapipe, whose
//! passes fetch colors as input attachments) checks the copies.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect`,
//! `tob_solid` and `tob_read` of the owned fixture `validation/fixtures/public/engine/tob_renders.ll`.

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::{
    Buffer, CommandQueue, Device, IoSurfacePlane, LoadAction, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, StoreAction, Texture,
    TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{BarrierScope, PixelFormat, PrimitiveType, RenderStages, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::time::Duration;

/// The plane: `WIDTH` texels a row of 4 bytes, `HEIGHT` rows, rows of `ROW` bytes.
const WIDTH: u64 = 64;
const HEIGHT: u64 = 16;
const ROW: u64 = WIDTH * 4;
const LENGTH: u64 = ROW * HEIGHT;

/// The scissor the read draws through: `[8, 16) x [4, 8)`. Its rectangle a pixel wider each way,
/// `[7, 17) x [3, 9)`, is [`READ_TEXELS`] texels.
const READ_AT: ScissorRect = ScissorRect {
    x: 8,
    y: 4,
    width: 8,
    height: 4,
};
const READ_TEXELS: u64 = 10 * 6;

/// The rows `rows` of the plane, every column, as a scissor.
const fn rows(rows: std::ops::Range<u64>) -> ScissorRect {
    ScissorRect {
        x: 0,
        y: rows.start,
        width: WIDTH,
        height: rows.end - rows.start,
    }
}

/// The pipelines a pass draws with: a solid color over the plane, and the read of the plane's
/// texel under each pixel.
struct Tools {
    queue: CommandQueue,
    solid: RenderPipelineState,
    read: RenderPipelineState,
    back: Buffer,
}

impl Tools {
    fn new(device: &Device) -> Tools {
        let renders = library(device, "engine/tob_renders.ll");
        let vertex = renders.new_function_with_name("tob_rect").unwrap();
        let pipeline = |name: &str| {
            let fragment = renders.new_function_with_name(name).unwrap();
            let mut descriptor =
                RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
            descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
            let options = PipelineOptions::default();
            device
                .new_render_pipeline_state(&descriptor, &options)
                .unwrap()
        };
        Tools {
            queue: device.new_command_queue().unwrap(),
            solid: pipeline("tob_solid"),
            read: pipeline("tob_read"),
            back: device
                .new_buffer(LENGTH, ResourceOptions::shared())
                .unwrap(),
        }
    }

    /// A fill of `buffer`'s `bytes` with `value` in a command buffer of its own.
    fn fill(&self, buffer: &Buffer, bytes: std::ops::Range<u64>, value: u8) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(buffer, bytes, value).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    }

    /// A pass loading and storing `texture`, in a command buffer of its own: where `solid` is
    /// given, a draw of its value (every channel) through its scissor and a barrier over the
    /// render targets; then a draw over the whole texture through [`READ_AT`] that writes the
    /// texture's texel under each pixel, as it reads it. The work counts before and after it.
    fn pass(
        &self,
        texture: &Texture,
        solid: Option<(u8, ScissorRect)>,
    ) -> (WorkCounts, WorkCounts) {
        let device = self.queue.device();
        let before = device.work_counts();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(texture.clone());
        color.load_action = LoadAction::Load;
        color.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
        if let Some((value, scissor)) = solid {
            encoder.set_render_pipeline_state(&self.solid).unwrap();
            encoder.set_scissor_rect(scissor).unwrap();
            encoder.set_vertex_bytes(&corners, 0).unwrap();
            let unorm = f32::from(value) / 255.0;
            let rgba = bytes_of(&[unorm; 4], f32::to_le_bytes);
            encoder.set_fragment_bytes(&rgba, 0).unwrap();
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
            let fragment = RenderStages::FRAGMENT;
            encoder.memory_barrier_with_scope(BarrierScope::RENDER_TARGETS, fragment, fragment);
        }
        encoder.set_render_pipeline_state(&self.read).unwrap();
        encoder.set_vertex_bytes(&corners, 0).unwrap();
        encoder.set_fragment_texture(Some(texture), 0).unwrap();
        encoder.set_scissor_rect(READ_AT).unwrap();
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        (before, device.work_counts())
    }

    /// `buffer`'s bytes, as a blit copy into the read-back buffer moves them.
    fn bytes(&self, buffer: &Buffer) -> Vec<u8> {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        (blit.copy_from_buffer_to_buffer(buffer, 0, &self.back, 0, LENGTH)).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let bytes = self.back.contents().unwrap().as_ptr();
        // SAFETY: the read-back buffer's bytes, which no command writes now.
        unsafe { std::slice::from_raw_parts(bytes, LENGTH as usize) }.to_vec()
    }
}

/// What one pass moved, between its counts `before` and `after`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Read {
    /// `WorkCounts::shadow_texels_copied_in`.
    copied_in: u64,
    /// `WorkCounts::own_pixel_reads`, `attachment_reads_through_copies`,
    /// `attachment_read_splits` and `attachment_read_splits_spared`.
    own_pixel: u64,
    through_copies: u64,
    splits: u64,
    spared: u64,
}

fn read((before, after): (WorkCounts, WorkCounts)) -> (Read, u64) {
    let delta = |count: fn(&WorkCounts) -> u64| count(&after) - count(&before);
    let read = Read {
        copied_in: delta(|c| c.shadow_texels_copied_in),
        own_pixel: delta(|c| c.own_pixel_reads),
        through_copies: delta(|c| c.attachment_reads_through_copies),
        splits: delta(|c| c.attachment_read_splits),
        spared: delta(|c| c.attachment_read_splits_spared),
    };
    (read, delta(|c| c.reading_draws_snapshotted))
}

/// Every byte of row `y` of `bytes`, by `value_of` the row.
fn rows_hold(bytes: &[u8], value_of: impl Fn(u64) -> u8) -> bool {
    (0..HEIGHT).all(|y| {
        let row = &bytes[(y * ROW) as usize..((y + 1) * ROW) as usize];
        row.iter().all(|&byte| byte == value_of(y))
    })
}

/// A read before any draw of its pass takes the texels from before the pass, and the copy in
/// before the pass moves the read's texels alone, command buffer after command buffer, a fill of
/// the bytes between them reaching the next read; a read after a draw and a barrier takes the
/// draw's texels, through a copy after the draw of the read's texels alone where the device
/// splits the pass there. Each reads through a snapshot, with no copy, where the device reads so.
#[test]
fn a_reader_of_its_own_pass_s_target_over_private_bytes_copies_in_what_it_reads() {
    let scratch = Scratch::new("attached-read-standing");
    let device = device(&scratch.0);
    let settled = device.settle_in_place_checks(Duration::from_secs(120));
    let tools = Tools::new(&device);
    let buffer = device
        .new_buffer(LENGTH, ResourceOptions::private())
        .unwrap();
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let plane = IoSurfacePlane::new(0, ROW, true);
    let texture = (buffer.new_texture_with_iosurface_plane(&descriptor, plane)).unwrap();
    assert!(
        texture.shadow().is_none(),
        "the plane's image aliases its bytes (in-place checks settled: {settled})"
    );
    // Row `y` holds the byte 0x20 + y in every channel.
    let first = |y: u64| 0x20 + y as u8;
    for y in 0..HEIGHT {
        tools.fill(&buffer, y * ROW..(y + 1) * ROW, first(y));
    }
    let only_read = Read {
        copied_in: READ_TEXELS,
        own_pixel: 1,
        through_copies: 1,
        splits: 0,
        spared: 0,
    };
    let snapshot = Read {
        copied_in: 0,
        own_pixel: 0,
        through_copies: 0,
        splits: 0,
        spared: 0,
    };
    // With no barrier: the copy in, or a snapshot (the module docs).
    let unordered = |snapshotted: u64| match snapshotted {
        0 => only_read,
        _ => snapshot,
    };
    let (moved, snapshotted) = read(tools.pass(&texture, None));
    assert_eq!(moved, unordered(snapshotted), "the first read's copy in");
    assert!(
        rows_hold(&tools.bytes(&buffer), first),
        "the read wrote the texels from before the pass"
    );
    // Rows 4 to 8, which the read reads, filled with 0x77: the next read takes the fill.
    tools.fill(&buffer, 4 * ROW..8 * ROW, 0x77);
    let filled = |y: u64| match (4..8).contains(&y) {
        true => 0x77,
        false => first(y),
    };
    let (moved, snapshotted) = read(tools.pass(&texture, None));
    assert_eq!(
        moved,
        unordered(snapshotted),
        "the next command buffer's copy in"
    );
    assert!(
        rows_hold(&tools.bytes(&buffer), filled),
        "the read took the fill"
    );
    // A draw of 0x99 over every texel, a barrier, and the read: the draw's texels everywhere.
    let (moved, snapshotted) = read(tools.pass(&texture, Some((0x99, rows(0..HEIGHT)))));
    eprintln!(
        "ATTACHED-READ-STANDING: after a draw and a barrier: {moved:?}, snapshotted {snapshotted}"
    );
    let after_draw = match snapshotted {
        0 => Read {
            splits: 1,
            ..only_read
        },
        _ => snapshot,
    };
    assert_eq!(moved, after_draw, "the read after the draw and the barrier");
    assert!(
        rows_hold(&tools.bytes(&buffer), |_| 0x99),
        "the read took the draw's texels"
    );
}

/// A read after a draw and a barrier inside its instance, where the draw drew none of the texels
/// the read reads (rows 10 to 16, the read's rows 3 to 9 with its helpers'), stays in the
/// instance and reads the copy taken before it, which holds them as a split's copy would
/// (`WorkCounts::attachment_read_splits_spared`); one where the draw drew some of them (rows 6 to
/// 12) splits the pass and reads the draw's texels there. Each reads through a snapshot, with no
/// copy or split, where the device reads so.
#[test]
fn a_reader_after_a_barrier_of_texels_its_instance_never_drew_spares_the_split() {
    let scratch = Scratch::new("attached-read-spared");
    let device = device(&scratch.0);
    let settled = device.settle_in_place_checks(Duration::from_secs(120));
    let tools = Tools::new(&device);
    let buffer = device
        .new_buffer(LENGTH, ResourceOptions::private())
        .unwrap();
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let plane = IoSurfacePlane::new(0, ROW, true);
    let texture = (buffer.new_texture_with_iosurface_plane(&descriptor, plane)).unwrap();
    assert!(
        texture.shadow().is_none(),
        "the plane's image aliases its bytes (in-place checks settled: {settled})"
    );
    // Row `y` holds the byte 0x20 + y in every channel.
    let first = |y: u64| 0x20 + y as u8;
    for y in 0..HEIGHT {
        tools.fill(&buffer, y * ROW..(y + 1) * ROW, first(y));
    }
    let snapshot = Read {
        copied_in: 0,
        own_pixel: 0,
        through_copies: 0,
        splits: 0,
        spared: 0,
    };
    let (moved, snapshotted) = read(tools.pass(&texture, Some((0x99, rows(10..HEIGHT)))));
    eprintln!("ATTACHED-READ-SPARED: apart from the draw: {moved:?}, snapshotted {snapshotted}");
    let apart = match snapshotted {
        0 => Read {
            copied_in: READ_TEXELS,
            own_pixel: 1,
            through_copies: 1,
            splits: 0,
            spared: 1,
        },
        _ => snapshot,
    };
    assert_eq!(
        moved, apart,
        "the read apart from the draw stays in its instance"
    );
    let drawn = |y: u64| match y >= 10 {
        true => 0x99,
        false => first(y),
    };
    assert!(
        rows_hold(&tools.bytes(&buffer), drawn),
        "the read took the texels from before the pass, the draw wrote its rows"
    );
    let (moved, snapshotted) = read(tools.pass(&texture, Some((0x55, rows(6..12)))));
    eprintln!("ATTACHED-READ-SPARED: over the draw: {moved:?}, snapshotted {snapshotted}");
    let over = match snapshotted {
        0 => Read {
            copied_in: READ_TEXELS,
            own_pixel: 1,
            through_copies: 1,
            splits: 1,
            spared: 0,
        },
        _ => snapshot,
    };
    assert_eq!(moved, over, "the read over the draw splits the pass");
    let drawn = |y: u64| match y {
        6..12 => 0x55,
        12.. => 0x99,
        _ => first(y),
    };
    assert!(
        rows_hold(&tools.bytes(&buffer), drawn),
        "the read took the draw's texels over its rows, those from before the pass elsewhere"
    );
}
