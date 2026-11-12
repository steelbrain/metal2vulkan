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
//! **Drawn after its move in place.** A `ShaderRead` surface of tight rows (1024 texels, 4096
//! bytes apart), made so before its verdict, is read by a draw through its shadow; once the check
//! settled, the next command buffer moves it in place where its verdicts pass
//! (`WorkCounts::host_textures_moved_in_place`), and its shadow's image and views go once that
//! command buffer completed (`host_texture_shadows_freed_after_move`). A draw of a later render
//! encoder, which starts from the bind memo the first draw's encoder left its device
//! (`binding::BindMemo`), works out the texture's view again (`bind_views_worked_out`) and reads
//! the texels the CPU wrote into the pages after the move, not through the memo's view, which was
//! the shadow's and went with it. Printed (`MOVED-SURFACE`).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect` and
//! `tob_read` of the owned fixture `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{device, over_host, Host, Scratch};
use super::encoder_holds::{bytes_of, library};
use crate::{
    BlitOption, Buffer, CommandQueue, Device, IoSurfacePlane, LoadAction, Origin, PipelineOptions,
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

/// The surface drawn after its move in place: its width and bytes per row, tight rows.
const MOVED: (u64, u64) = (1024, 4096);

/// The longest a command buffer may take here: the consumer's guest waits about 2 s for an
/// answer.
const WAIT: Duration = Duration::from_secs(20);

/// The longest the device's in-place check may take to settle.
const CHECK_WAIT: Duration = Duration::from_secs(120);

/// The texel the surface holds at `(x, y)`, `BGRA8Unorm` bytes.
fn pixel(x: u64, y: u64) -> [u8; 4] {
    [
        (x % 251) as u8,
        (y * 13 + x / 251) as u8,
        (x * 7 + y) as u8,
        0xff,
    ]
}

/// The texel the CPU writes at `(x, y)` after the move: [`pixel`]'s, each color byte inverted.
fn moved_pixel(x: u64, y: u64) -> [u8; 4] {
    let [b, g, r, a] = pixel(x, y);
    [!b, !g, !r, a]
}

/// Write `texel`'s texels into `buffer`'s first [`HEIGHT`] rows of `width` texels, `row` bytes
/// apart.
fn fill(buffer: &Buffer, (width, row): (u64, u64), texel: fn(u64, u64) -> [u8; 4]) {
    let contents = buffer.contents().unwrap();
    for y in 0..HEIGHT {
        for x in 0..width {
            let at = (y * row + x * 4) as usize;
            // SAFETY: inside the buffer's first `row * HEIGHT` bytes, which no command uses.
            unsafe { contents.as_ptr().add(at).copy_from(texel(x, y).as_ptr(), 4) };
        }
    }
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

/// `texture` read at its own pixels by a draw of `state` into a private texture, copied out by a
/// blit: its texels, tight rows, how long the draw's command buffer took and how long the copy's
/// did.
fn drawn(
    queue: &CommandQueue,
    state: &RenderPipelineState,
    texture: &Texture,
) -> (Vec<u8>, Option<Duration>, Option<Duration>) {
    let mut td =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, texture.width(), HEIGHT, false);
    td.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    td.resource_options = ResourceOptions::private();
    let target = queue.device().new_texture(&td).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::DontCare;
    color.store_action = StoreAction::Store;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
    encoder.set_vertex_bytes(&corners, 0).unwrap();
    encoder.set_fragment_texture(Some(texture), 0).unwrap();
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    encoder.end_encoding().unwrap();
    let draw = completed(&command_buffer);
    let (drawn, back) = copied(queue, &target);
    (drawn, draw, back)
}

/// The rows of a `width`-wide surface `bytes` holds otherwise than `texel`'s texels, at most a
/// few.
fn wrong_rows(width: u64, bytes: &[u8], texel: fn(u64, u64) -> [u8; 4]) -> Vec<u64> {
    (0..HEIGHT)
        .filter(|&y| {
            (0..width).any(|x| {
                let at = ((y * width + x) * 4) as usize;
                bytes[at..at + 4] != texel(x, y)
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
        fill(&buffer, (width, row), pixel);
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
        let rows = wrong_rows(width, &bytes, pixel);
        let (drawn, draw, back) = drawn(&queue, &state, &texture);
        let drawn_rows = wrong_rows(width, &drawn, pixel);
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

#[test]
fn a_surface_moved_in_place_is_drawn_through_its_image_once_its_shadow_went() {
    // The pages, declared before the device that uses them, so dropped after it.
    let host = Host::new();
    let cache = Scratch::new("moved-surface");
    let device = device(&cache.0);
    let queue = device.new_command_queue().unwrap();
    let state = reader(&device);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(buffer) = over_host(&device, &host, &runs) else {
        return;
    };
    let (width, row) = MOVED;
    fill(&buffer, MOVED, pixel);
    let before = device.work_counts();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, HEIGHT, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = buffer.resource_options();
    let texture = buffer
        .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, row, false))
        .unwrap();
    let made = device.work_counts();
    let pending =
        made.host_texture_shadows_check_pending - before.host_texture_shadows_check_pending;
    assert_eq!(
        (texture.shadow().is_some(), pending),
        (true, 1),
        "a shadow made before its verdict"
    );
    // A draw through the shadow: its encoder's bind keeps the shadow's view, in the memo the
    // encoder leaves its device.
    let (early, _, _) = drawn(&queue, &state, &texture);
    let early_rows = wrong_rows(width, &early, pixel);
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "the check settles"
    );
    // The next command buffer, committed while no other is open, moves it in place where its
    // verdicts pass.
    let (_, copy) = copied(&queue, &texture);
    let moved =
        device.work_counts().host_textures_moved_in_place - before.host_textures_moved_in_place;
    // Its shadow's image and views go once that command buffer completed, on the thread that
    // lets it go.
    let started = Instant::now();
    let freed = loop {
        let freed = device.work_counts().host_texture_shadows_freed_after_move
            - before.host_texture_shadows_freed_after_move;
        if freed >= moved || started.elapsed() > WAIT {
            break freed;
        }
        std::thread::sleep(Duration::from_millis(2));
    };
    // Texels the CPU writes now: in place the image is the pages, and a shadow copies them in.
    fill(&buffer, MOVED, moved_pixel);
    let at_draw = device.work_counts();
    let (late, draw, back) = drawn(&queue, &state, &texture);
    let worked_out = device.work_counts().bind_views_worked_out - at_draw.bind_views_worked_out;
    let late_rows = wrong_rows(width, &late, moved_pixel);
    eprintln!(
        "MOVED-SURFACE on {}: {width}x{HEIGHT} rows of {row}: drawn through its shadow, wrong \
         rows {early_rows:?}; after the check, the next commit (completed in {copy:?}) moved it \
         in place {moved}, its shadow freed {freed}; a later encoder's draw worked out {worked_out} \
         views, completed in {draw:?} (its copy {back:?}), wrong rows {late_rows:?}",
        device.name(),
    );
    assert_eq!(early_rows, Vec::<u64>::new(), "the draw through the shadow");
    assert_eq!(freed, moved, "a moved surface's shadow goes");
    if moved == 1 {
        assert!(
            worked_out >= 1,
            "the draw after the move binds its view of the image in place, not the shadow's \
             that went"
        );
    }
    assert!(
        copy.is_some() && draw.is_some() && back.is_some(),
        "every command buffer completed"
    );
    assert_eq!(
        late_rows,
        Vec::<u64>::new(),
        "the draw after the move reads the texels the CPU wrote then"
    );
}
