//! The consumer's presented surfaces at odd widths (reims-vgpu's
//! `on_vulkan_a_presented_surface_at_an_odd_width_reads_row_for_row`): on a new device whose
//! shader cache holds no in-place record, `BGRA8Unorm` textures of IOSurface planes over no-copy
//! buffers at widths 3662, 1831 and 1001, 16 rows padded to 64 bytes (14656, 7360 and 4032 bytes
//! apart), of `ShaderRead | RenderTarget` usage and then `ShaderRead` alone, each copied out by a
//! blit and read at its own pixels by a draw into a private texture. The first texture's making
//! starts the device's in-place check and waits for none of it (`in_place.rs`): it is a shadow,
//! counted as made before its verdict (`WorkCounts::host_texture_shadows_check_pending`), where a
//! wait for the verdict took 1.8 s on a Radeon 780M in a debug build, past the 2 s the consumer's
//! guest waits for the answer after it. Every texture's copy and draw read the surface row for
//! row. Printed per texture (`FIRST-HOST-TEXTURE`): how long its making took, its path and why it
//! is a shadow, and how long each command buffer took to complete.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect` and
//! `tob_read` of the owned fixture `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{device, over_host, Host, Scratch};
use super::encoder_holds::{bytes_of, library};
use crate::{
    BlitOption, CommandQueue, Device, IoSurfacePlane, LoadAction, Origin, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction, Texture,
    TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The surfaces' rows.
const HEIGHT: u64 = 16;

/// Each surface's width and bytes per row.
const SIZES: [(u64, u64); 3] = [(3662, 14656), (1831, 7360), (1001, 4032)];

/// The longest a command buffer may take here: the consumer's guest waits about 2 s for an
/// answer.
const WAIT: Duration = Duration::from_secs(20);

/// The texel the surface holds at `(x, y)`, `BGRA8Unorm` bytes.
fn pixel(x: u64, y: u64) -> [u8; 4] {
    [
        (x % 251) as u8,
        (y * 13 + x / 251) as u8,
        (x * 7 + y) as u8,
        0xff,
    ]
}

/// The state of `tob_rect` and `tob_read` into a `BGRA8Unorm` color.
fn reader(device: &Device) -> RenderPipelineState {
    let library = library(device, "engine/tob_renders.ll");
    let function = |name: &str| {
        let function = library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(function("tob_read"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Commit `command_buffer`'s work and wait for it: how long it took, `None` past [`WAIT`].
fn completed(command_buffer: &crate::CommandBuffer) -> Option<Duration> {
    let started = Instant::now();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).ok()?;
    assert_eq!(command_buffer.error(), None);
    Some(started.elapsed())
}

/// `texture`'s texels, tight rows, copied out by a blit: and how long its command buffer took.
fn copied(queue: &CommandQueue, texture: &Texture) -> (Vec<u8>, Option<Duration>) {
    let (width, row) = (texture.width(), texture.width() * 4);
    let out = (queue.device())
        .new_buffer(row * HEIGHT, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    (blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(width, HEIGHT, 1),
        &out,
        0,
        row,
        row * HEIGHT,
        BlitOption::None,
    ))
    .unwrap();
    blit.end_encoding().unwrap();
    let took = completed(&command_buffer);
    (super::merge_scratch::read(&out, row * HEIGHT), took)
}

/// The rows of a `width`-wide surface `checked` holds wrong, at most a few.
fn wrong_rows(width: u64, bytes: &[u8]) -> Vec<u64> {
    (0..HEIGHT)
        .filter(|&y| {
            (0..width).any(|x| {
                let at = ((y * width + x) * 4) as usize;
                bytes[at..at + 4] != pixel(x, y)
            })
        })
        .take(4)
        .collect()
}

#[test]
fn odd_width_surfaces_over_host_pages_are_made_and_read() {
    // The pages, declared before the device that uses them, so dropped after it.
    let hosts: Vec<Host> = (0..6).map(|_| Host::new()).collect();
    let cache = Scratch::new("first-host-texture");
    let device = device(&cache.0);
    let queue = device.new_command_queue().unwrap();
    let state = reader(&device);
    let runs = Arc::new(AtomicUsize::new(0));
    let mut wrong = Vec::new();
    let cases = [5u64, 1]
        .into_iter()
        .flat_map(|usage| SIZES.map(|size| (size, usage)));
    for (i, ((width, row), usage)) in cases.enumerate() {
        let host = &hosts[i];
        let Some(buffer) = over_host(&device, host, &runs) else {
            return;
        };
        let contents = buffer.contents().unwrap();
        for y in 0..HEIGHT {
            for x in 0..width {
                let at = (y * row + x * 4) as usize;
                // SAFETY: inside the buffer's first `row * HEIGHT` bytes, which no command uses.
                unsafe { contents.as_ptr().add(at).copy_from(pixel(x, y).as_ptr(), 4) };
            }
        }
        let before = device.work_counts();
        let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, HEIGHT, false);
        d.usage = match usage {
            5 => TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
            _ => TextureUsage::SHADER_READ,
        };
        d.resource_options = buffer.resource_options();
        let started = Instant::now();
        let texture = buffer
            .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, row, false))
            .unwrap();
        let made = started.elapsed();
        let after = device.work_counts();
        let path = match (texture.shadow().is_some(), texture.aliases_buffer()) {
            (true, _) => "shadow",
            (false, true) => "in place",
            (false, false) => "texels",
        };
        let (bytes, copy) = copied(&queue, &texture);
        let rows = wrong_rows(width, &bytes);
        // A draw that reads the surface at its own pixels into a private texture.
        let mut td = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, HEIGHT, false);
        td.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        td.resource_options = ResourceOptions::private();
        let target = device.new_texture(&td).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::DontCare;
        color.store_action = StoreAction::Store;
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&state).unwrap();
        let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
        encoder.set_vertex_bytes(&corners, 0).unwrap();
        encoder.set_fragment_texture(Some(&texture), 0).unwrap();
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        encoder.end_encoding().unwrap();
        let draw = completed(&command_buffer);
        let (drawn, back) = copied(&queue, &target);
        let drawn_rows = wrong_rows(width, &drawn);
        let pending =
            after.host_texture_shadows_check_pending - before.host_texture_shadows_check_pending;
        let refused = (after.host_texture_shadows_refused_pitch
            - before.host_texture_shadows_refused_pitch)
            + (after.host_texture_shadows_refused_rows - before.host_texture_shadows_refused_rows);
        eprintln!(
            "FIRST-HOST-TEXTURE {i} {width}x{HEIGHT} rows of {row} usage {usage}: made in {made:?}, \
             {path} (before its verdict {pending}, pitch or rows refused {refused}); copy \
             {copy:?}, wrong rows {rows:?}; draw {draw:?}, its copy {back:?}, wrong rows \
             {drawn_rows:?}"
        );
        if i == 0 {
            // Its making started the check, and waited for none of it.
            assert_eq!(
                (path, pending),
                ("shadow", 1),
                "the first texture: a shadow made before its verdict"
            );
        }
        if copy.is_none() || draw.is_none() || back.is_none() {
            wrong.push(format!("{i}: a command buffer did not complete"));
        }
        if !rows.is_empty() || !drawn_rows.is_empty() {
            wrong.push(format!("{i}: rows {rows:?}, drawn {drawn_rows:?}"));
        }
    }
    for line in device.in_place_check_report() {
        eprintln!("FIRST-HOST-TEXTURE check: {line}");
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
