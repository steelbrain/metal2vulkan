//! What making textures over no-copy buffers costs on the device the cfg selects, the way a
//! consumer's guest browser makes them each frame: report only, in release builds, to weigh the
//! runtime's texture-over-buffer path before and after a change to it.
//!
//! The workload is a guest browser's while it scrolls, as the consumer replays it: each frame, for
//! each of a few IOSurfaces, a new no-copy buffer over the surface's pages and a texture of its
//! plane over that buffer (`new_texture_with_iosurface_plane`, the consumer's
//! `newTextureWithDescriptor:iosurface:plane:`), a render pass that loads and stores the texture
//! and a blit that reads it into a private texture, one command buffer for the frame, waited for;
//! then the textures and buffers are released. The surfaces are `BGRA8Unorm`, of the layouts in
//! [`LAYOUTS`]: the browser's 1307x512 tile with rows padded to 64 bytes (5248, its tight row
//! 5228), a tight-row tile, and other padded rows. Their pages are:
//!
//! - **`anonymous`**: private anonymous memory, whose pointer every measured driver imports;
//! - **`guest-ram`**: a sealed `memfd` mapped shared, as QEMU's guest RAM is, with a dma-buf
//!   source (`HostMemory::with_dma_buf`, `udmabuf`s of the file's pages): RADV refuses the pointer
//!   and imports dma-bufs, ANV and NVIDIA import the pointer. Skipped where `/dev/udmabuf` cannot
//!   be opened.
//!
//! Each case also runs with the buffer kept across frames (`kept`), a new texture over it each
//! frame. Printed per case (`SURFTEX-COST`): the texture's form (aliased or shadowed), the median
//! and 90th percentile of the host's time making the buffer, making the texture and releasing
//! both, per texture; per frame, the encoding and the commit-to-completion wall time and the
//! process's CPU time; and the device's work counts per frame that the path moves (dma-buf images
//! and fallbacks, shadows and those zeroed as made, images started on the host, internal
//! submissions, shadows' copies and the texels their copies in and back move, the merges'
//! scratches reused and zeroed, the merges that fetched their texels, the command buffers begun by
//! what for, the render pass begins and secondary executions recorded, and the passes whose first
//! instance recorded inline).
//!
//! [`surface_texture_cost_parts`] takes the frame apart over anonymous pages: the textures' making
//! around an empty command buffer, the passes that store, the passes that store `DontCare`, the
//! passes that draw and store, the blits, and both, so a shadow's copies in and back show apart
//! from the rest (`SURFTEX-PART`), then over `Private` buffers: a shadow over the pages, which the
//! CPU may write, is merged back, and one over a tracked `Private` buffer's bytes copied back.
//!
//! [`shadow_copy_costs`] times a shadow's copies alone, and the same bytes moved between other
//! memories, to tell the image's tiling from the pages' memory (`SURFTEX-COPY`).
//!
//! [`read_only_padded_costs`] times a compositor's frame over the padded layouts' textures made
//! with `ShaderRead` usage alone, each sampled by a draw: shadows copied in at each use, made each
//! frame or once (`SURFTEX-READS`).
//!
//! All four are ignored by default: measurements, not checks. Run them by name with
//! `--run-ignored all`, in a build with `--cfg m2v_system_alloc`, whose lib test binary keeps the
//! system allocator as a consumer's process does.

use super::dma_buf_imports::{udmabuf_openable, Calls, Memfd, View};
use super::host_memory::{host_page, Mapping, GUEST_PAGE};
use super::no_copy_textures::{blending, floats, staged};
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandQueue, Device, HostMemory, IoSurfacePlane, LoadAction, Origin,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use std::time::Instant;

/// The untimed and the timed frames of each case.
const WARM: usize = 10;
const FRAMES: usize = 100;

/// A surface's plane: its name, width, height and bytes per row (`BGRA8Unorm`, 4 bytes a texel).
const LAYOUTS: [(&str, u64, u64, u64); 5] = [
    // The browser's tile: rows padded to 64 bytes.
    ("1307x512/5248", 1307, 512, 5248),
    // Tight rows, a multiple of 256 bytes.
    ("1024x512/4096", 1024, 512, 4096),
    // Rows padded to 256 bytes.
    ("1307x512/5376", 1307, 512, 5376),
    // A small tile, rows padded to 64 bytes.
    ("300x200/1216", 300, 200, 1216),
    // A full-width strip, rows padded to 64 bytes.
    ("1913x96/7680", 1913, 96, 7680),
];

/// Where a case's surfaces' pages lie.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pages {
    Anonymous,
    GuestRam,
}

/// The process's CPU time so far, every thread's, in microseconds.
fn process_cpu_us() -> f64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a valid timespec for the call to fill.
    let read = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut now) };
    assert_eq!(read, 0, "the process's CPU clock");
    now.tv_sec as f64 * 1e6 + now.tv_nsec as f64 / 1e3
}

/// Microseconds since `at`.
fn since(at: Instant) -> f64 {
    at.elapsed().as_secs_f64() * 1e6
}

/// The median and 90th percentile of `values`.
fn median_p90(mut values: Vec<f64>) -> (f64, f64) {
    values.sort_by(f64::total_cmp);
    let at = |p: usize| values[(values.len() * p / 100).min(values.len() - 1)];
    (at(50), at(90))
}

/// A surface's pages, and how a no-copy buffer is made over them.
enum Surface {
    Anonymous(Mapping),
    GuestRam {
        view: View,
        /// The memfd the view maps (it outlives the view's buffers).
        _memfd: std::rc::Rc<Memfd>,
        calls: Calls,
    },
}

impl Surface {
    fn len(width_bytes: u64, height: u64) -> usize {
        ((width_bytes * height) as usize).next_multiple_of(GUEST_PAGE)
    }

    /// A no-copy buffer of `len` bytes over the surface's first byte, of default options, as the
    /// consumer makes one over a surface's pages for each texture of it.
    fn buffer(&self, device: &Device, len: usize, memfd: Option<&Memfd>) -> Buffer {
        let memory = match self {
            // SAFETY: the mapping outlives every buffer over it (the test drops them first).
            Surface::Anonymous(mapping) => unsafe { HostMemory::new(mapping.pointer()) },
            Surface::GuestRam { view, calls, .. } => {
                let memfd = memfd.expect("the guest RAM's file");
                // SAFETY: the view outlives every buffer over it, and the source makes dma-bufs of
                // its pages.
                unsafe { HostMemory::with_dma_buf(view.pointer(0), memfd.source(view, calls)) }
            }
        };
        // SAFETY: as above.
        unsafe {
            device.new_buffer_with_bytes_no_copy(
                memory,
                len as u64,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap()
    }
}

/// The texture of a surface's plane, `width`x`height` with rows `bpr` bytes apart, over `buffer`.
fn plane_texture(buffer: &Buffer, width: u64, height: u64, bpr: u64) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = buffer.resource_options();
    buffer
        .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, bpr, false))
        .unwrap()
}

/// A texture `width`x`height` with rows `bpr` bytes apart over the `Private` `buffer`'s first
/// byte, as [`plane_texture`]'s is used.
fn private_texture(buffer: &Buffer, width: u64, height: u64, bpr: u64) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    buffer.new_texture(&d, 0, bpr).unwrap()
}

/// Whether `texture`'s texels are its own image over the buffer's memory: aliased, not shadowed.
fn aliased(texture: &Texture) -> bool {
    // SAFETY: the handle is only compared, while the texture lives.
    unsafe { texture.vk_image() }.is_some()
}

/// What a frame encodes over its textures: [`Part::Both`] is the consumer's frame, the others
/// its parts apart ([`surface_texture_cost_parts`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Part {
    /// Nothing: an empty command buffer, committed and waited for, after the textures are made.
    Empty,
    /// A pass loading and storing each texture: a shadow's copy in before it and copy back after.
    PassesStored,
    /// A pass loading each texture and storing it `DontCare`: a shadow's copy in, and no copy back.
    PassesNotStored,
    /// A pass loading each texture, blending a rectangle over its left half, and storing it, as a
    /// browser's pass draws: a shadow's copy in and copy back.
    PassesDrawn,
    /// As [`Part::PassesDrawn`], under a scissor of the left half, as a browser's pass redrawing a
    /// damaged part of a tile draws.
    PassesScissored,
    /// A blit reading each texture into its private texture, which reads a texture over a buffer
    /// as its bytes: no shadow's copy.
    Blits,
    /// The passes that store, then the blits.
    Both,
}

/// A pipeline blending a rectangle over a `BGRA8Unorm` attachment, and its rectangle's corners
/// and color ([`Part::PassesDrawn`]).
struct Drawing {
    pipeline: RenderPipelineState,
    corners: Buffer,
    color: Buffer,
}

impl Drawing {
    fn new(device: &Device) -> Drawing {
        Drawing {
            pipeline: blending(device, PixelFormat::BGRA8Unorm),
            corners: staged(device, &floats([-1.0, -1.0, 0.0, 1.0])),
            color: staged(device, &floats([0.25, 0.5, 0.75, 0.5])),
        }
    }
}

/// One frame's `part` of the work over `textures`: a pass loading and storing each, and a blit
/// reading each into its private texture of `privates`, in one command buffer, committed and
/// waited for; a drawn pass draws through `drawing`. The host's time encoding, and from the commit
/// until the wait returned.
fn frame(
    queue: &CommandQueue,
    textures: &[Texture],
    privates: &[Texture],
    (part, drawing): (Part, Option<&Drawing>),
) -> (f64, f64) {
    let began = Instant::now();
    let command_buffer = queue.command_buffer().unwrap();
    let store_action = match part {
        Part::PassesStored | Part::PassesDrawn | Part::PassesScissored | Part::Both => {
            Some(StoreAction::Store)
        }
        Part::PassesNotStored => Some(StoreAction::DontCare),
        Part::Empty | Part::Blits => None,
    };
    for texture in textures.iter().filter(|_| store_action.is_some()) {
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(texture.clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = store_action.unwrap_or(StoreAction::Store);
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        if matches!(part, Part::PassesDrawn | Part::PassesScissored) {
            let drawing = drawing.expect("a drawn pass's pipeline");
            encoder
                .set_render_pipeline_state(&drawing.pipeline)
                .unwrap();
            if part == Part::PassesScissored {
                let left = ScissorRect::new(0, 0, texture.width() / 2, texture.height());
                encoder.set_scissor_rect(left).unwrap();
            }
            encoder
                .set_vertex_buffer(Some(&drawing.corners), 0, 0)
                .unwrap();
            encoder
                .set_fragment_buffer(Some(&drawing.color), 0, 0)
                .unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
        }
        encoder.end_encoding().unwrap();
    }
    if matches!(part, Part::Blits | Part::Both) {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for (texture, private) in textures.iter().zip(privates) {
            let size = Size::new(texture.width(), texture.height(), 1);
            let origin = Origin::new(0, 0, 0);
            blit.copy_from_texture_to_texture(texture, 0, 0, origin, size, private, 0, 0, origin)
                .unwrap();
        }
        blit.end_encoding().unwrap();
    }
    let encoded = since(began);
    let committed = Instant::now();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    (encoded, since(committed))
}

/// The work counts the texture path moves, `after` less `before`, per frame of `frames`.
fn counts_per_frame(before: &WorkCounts, after: &WorkCounts, frames: usize) -> String {
    let per = |a: u64, b: u64| (a - b) as f64 / frames as f64;
    format!(
        "shadowed textures {:.2} (zeroed {:.2}), dma-buf images {:.2}, dma-buf image fallbacks {:.2}, dma-buf imports \
         {:.2}, images started on the host {:.2}, internal submissions {:.2}, staging submissions \
         {:.2}, shadow copies {:.2} (texels copied in {:.0}, back {:.0}; copies back dispatched \
         {:.2}; keeps and merges dispatched {:.2}, texels kept {:.0}; merge scratches reused \
         {:.2}, zeroed {:.2}; merges fetched {:.2}); command buffers begun {:.2} (encoders' own \
         {:.2}, segments {:.2}, draws' secondaries {:.2}, batches' {:.2}, barriers kept {:.2}, \
         internal {:.2}), render pass begins recorded {:.2}, secondaries executed {:.2}, passes \
         recorded inline {:.2}",
        per(after.shadowed_textures, before.shadowed_textures),
        per(after.shadows_zeroed, before.shadows_zeroed),
        per(after.dma_buf_images, before.dma_buf_images),
        per(
            after.dma_buf_image_fallbacks,
            before.dma_buf_image_fallbacks
        ),
        per(after.dma_buf_imports, before.dma_buf_imports),
        per(
            after.images_started_on_the_host,
            before.images_started_on_the_host
        ),
        per(after.internal_submissions, before.internal_submissions),
        per(after.staging_submissions, before.staging_submissions),
        per(after.shadow_copies, before.shadow_copies),
        per(
            after.shadow_texels_copied_in,
            before.shadow_texels_copied_in
        ),
        per(
            after.shadow_texels_copied_back,
            before.shadow_texels_copied_back
        ),
        per(
            after.shadow_copies_back_dispatched,
            before.shadow_copies_back_dispatched
        ),
        per(
            after.shadow_merge_copies_dispatched,
            before.shadow_merge_copies_dispatched
        ),
        per(after.shadow_texels_kept, before.shadow_texels_kept),
        per(
            after.shadow_merge_scratches_reused,
            before.shadow_merge_scratches_reused
        ),
        per(
            after.shadow_merge_scratches_zeroed,
            before.shadow_merge_scratches_zeroed
        ),
        per(after.shadow_merges_fetched, before.shadow_merges_fetched),
        per(after.command_buffers_begun, before.command_buffers_begun),
        per(
            after.command_buffers_begun_encoders,
            before.command_buffers_begun_encoders
        ),
        per(
            after.command_buffers_begun_segments + after.command_buffers_begun_continuations,
            before.command_buffers_begun_segments + before.command_buffers_begun_continuations
        ),
        per(
            after.command_buffers_begun_secondaries,
            before.command_buffers_begun_secondaries
        ),
        per(
            after.command_buffers_begun_batch_work + after.command_buffers_begun_batch_barriers,
            before.command_buffers_begun_batch_work + before.command_buffers_begun_batch_barriers
        ),
        per(
            after.command_buffers_begun_kept_barriers,
            before.command_buffers_begun_kept_barriers
        ),
        per(
            after.command_buffers_begun_internal,
            before.command_buffers_begun_internal
        ),
        per(
            after.commands_recorded.cmd_begin_render_pass,
            before.commands_recorded.cmd_begin_render_pass
        ),
        per(
            after.commands_recorded.cmd_execute_commands,
            before.commands_recorded.cmd_execute_commands
        ),
        per(after.inline_passes, before.inline_passes),
    )
}

#[test]
#[ignore = "a release-mode measurement: run by name"]
fn surface_texture_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    messages.take_notices();
    let queue = device.new_command_queue().unwrap();
    let page = host_page();
    let lens: Vec<usize> = LAYOUTS
        .iter()
        .map(|&(_, _, h, bpr)| Surface::len(bpr, h))
        .collect();
    let privates: Vec<Texture> = LAYOUTS
        .iter()
        .map(|&(_, w, h, _)| {
            let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, w, h, false);
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = ResourceOptions::private();
            device.new_texture(&d).unwrap()
        })
        .collect();
    for pages in [Pages::Anonymous, Pages::GuestRam] {
        let memfd = match pages {
            Pages::GuestRam if !udmabuf_openable("surface_texture_costs") => continue,
            Pages::GuestRam => Some(std::rc::Rc::new(Memfd::new(
                lens.iter().sum::<usize>() + page,
            ))),
            Pages::Anonymous => None,
        };
        let mut at = 0;
        let surfaces: Vec<Surface> = lens
            .iter()
            .map(|&len| match &memfd {
                None => Surface::Anonymous(Mapping::anonymous(len)),
                Some(memfd) => {
                    let view = memfd.view(&[(at, len)]);
                    at += len;
                    Surface::GuestRam {
                        view,
                        _memfd: std::rc::Rc::clone(memfd),
                        calls: Calls::default(),
                    }
                }
            })
            .collect();
        for kept in [false, true] {
            // Each layout's buffer, kept across frames where `kept`.
            let mut held: Vec<Option<Buffer>> = vec![None; LAYOUTS.len()];
            let mut times = vec![(Vec::new(), Vec::new(), Vec::new()); LAYOUTS.len()];
            let mut forms = vec![false; LAYOUTS.len()];
            let (mut encodes, mut waits, mut cpus) = (Vec::new(), Vec::new(), Vec::new());
            let mut before = device.work_counts();
            for k in 0..WARM + FRAMES {
                if k == WARM {
                    before = device.work_counts();
                }
                let cpu = process_cpu_us();
                let mut textures = Vec::new();
                let mut buffers = Vec::new();
                for (index, &(_, w, h, bpr)) in LAYOUTS.iter().enumerate() {
                    let began = Instant::now();
                    let buffer = match &held[index] {
                        Some(buffer) => buffer.clone(),
                        None => surfaces[index].buffer(&device, lens[index], memfd.as_deref()),
                    };
                    let made_buffer = since(began);
                    if kept {
                        held[index] = Some(buffer.clone());
                    }
                    let began = Instant::now();
                    let texture = plane_texture(&buffer, w, h, bpr);
                    let made_texture = since(began);
                    // Asked once, untimed: handing the image out moves it to `GENERAL` by a
                    // waited submission of its own.
                    if k == 0 {
                        forms[index] = aliased(&texture);
                    }
                    if k >= WARM {
                        times[index].0.push(made_buffer);
                        times[index].1.push(made_texture);
                    }
                    textures.push(texture);
                    buffers.push(buffer);
                }
                let (encoded, waited) = frame(&queue, &textures, &privates, (Part::Both, None));
                for (index, (texture, buffer)) in textures.into_iter().zip(buffers).enumerate() {
                    let began = Instant::now();
                    drop((texture, buffer));
                    if k >= WARM {
                        times[index].2.push(since(began));
                    }
                }
                if k >= WARM {
                    encodes.push(encoded);
                    waits.push(waited);
                    cpus.push(process_cpu_us() - cpu);
                }
            }
            let after = device.work_counts();
            drop(held);
            let mode = if kept { "kept buffer" } else { "new buffer" };
            for (index, (buffers, textures, releases)) in times.into_iter().enumerate() {
                let name = LAYOUTS[index].0;
                let (buffer, buffer90) = median_p90(buffers);
                let (texture, texture90) = median_p90(textures);
                let (release, release90) = median_p90(releases);
                println!(
                    "SURFTEX-COST on {}: {pages:?} pages, {mode}, {name}: {}; per texture: buffer \
                     {buffer:.0} us (p90 {buffer90:.0}), texture {texture:.0} us (p90 \
                     {texture90:.0}), release {release:.0} us (p90 {release90:.0})",
                    device.name(),
                    if forms[index] { "aliased" } else { "shadowed" },
                );
            }
            let (encode, encode90) = median_p90(encodes);
            let (wait, wait90) = median_p90(waits);
            let (cpu, cpu90) = median_p90(cpus);
            println!(
                "SURFTEX-FRAME on {}: {pages:?} pages, {mode}, {} textures a frame: encode \
                 {encode:.0} us (p90 {encode90:.0}), commit to completion {wait:.0} us (p90 \
                 {wait90:.0}), process CPU {cpu:.0} us (p90 {cpu90:.0}); per frame: {}",
                device.name(),
                LAYOUTS.len(),
                counts_per_frame(&before, &after, FRAMES),
            );
        }
        let calls: usize = surfaces
            .iter()
            .map(|surface| match surface {
                Surface::GuestRam { calls, .. } => calls.lock().unwrap().len(),
                Surface::Anonymous(_) => 0,
            })
            .sum();
        println!(
            "SURFTEX-SOURCE on {}: {pages:?} pages: the dma-buf sources were called {calls} times",
            device.name()
        );
        drop(surfaces);
    }
    let notices = super::degrade_notices::counts(&messages.take_notices());
    println!("SURFTEX-NOTICES on {}: {notices:?}", device.name());
}

/// [`surface_texture_costs`]' frame over anonymous pages, a new buffer and texture of each of
/// [`LAYOUTS`] each frame, taken apart ([`Part`]): the textures made and released around an empty
/// command buffer, the passes that store, the passes that store `DontCare`, the passes that draw a
/// rectangle and store (also under a scissor of the left half), the blits, and both;
/// then the same over the packed layouts alone, which no shadow is made beside; then over every
/// layout again, each texture over a `Private` buffer of the layout's bytes made once, a new
/// texture over it each frame. Where a texture is a shadow, the passes that store less the empty
/// frame are its copies in and back and the pass, the passes that do not store less the empty
/// frame its copies in and the pass; the blits read the bytes either way. A shadow over the pages
/// is kept and merged back in place of a copy back, and one over a tracked `Private` buffer's
/// bytes copied back. Printed per set and part (`SURFTEX-PART`): the
/// textures' forms, the medians and 90th percentiles of each one's making (its buffer and
/// texture), of the frame's release, encoding, commit to completion and process CPU time, and the
/// work counts per frame.
#[test]
#[ignore = "a release-mode measurement: run by name"]
fn surface_texture_cost_parts() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    messages.take_notices();
    let queue = device.new_command_queue().unwrap();
    let lens: Vec<usize> = LAYOUTS
        .iter()
        .map(|&(_, _, h, bpr)| Surface::len(bpr, h))
        .collect();
    let privates: Vec<Texture> = LAYOUTS
        .iter()
        .map(|&(_, w, h, _)| {
            let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, w, h, false);
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = ResourceOptions::private();
            device.new_texture(&d).unwrap()
        })
        .collect();
    let surfaces: Vec<Surface> = lens
        .iter()
        .map(|&len| Surface::Anonymous(Mapping::anonymous(len)))
        .collect();
    let private_buffers: Vec<Buffer> = lens
        .iter()
        .map(|&len| {
            device
                .new_buffer(len as u64, ResourceOptions::private())
                .unwrap()
        })
        .collect();
    // Every layout, then the packed ones alone (rows one after another, no texture a shadow).
    let every: Vec<usize> = (0..LAYOUTS.len()).collect();
    let packed: Vec<usize> = every
        .iter()
        .copied()
        .filter(|&index| LAYOUTS[index].3 == LAYOUTS[index].1 * 4)
        .collect();
    let parts = [
        Part::Empty,
        Part::PassesStored,
        Part::PassesNotStored,
        Part::PassesDrawn,
        Part::PassesScissored,
        Part::Blits,
        Part::Both,
    ];
    let drawing = Drawing::new(&device);
    let drawing = Some(&drawing);
    let sets = [
        ("every layout", &every, false),
        ("packed layouts", &packed, false),
        ("every layout over Private buffers", &every, true),
    ];
    for (set, chosen, over_private) in sets {
        let privates: Vec<Texture> = chosen.iter().map(|&i| privates[i].clone()).collect();
        for part in parts {
            let mut forms = Vec::new();
            let mut makes = vec![Vec::new(); chosen.len()];
            let (mut releases, mut encodes, mut waits, mut cpus) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut before = device.work_counts();
            for k in 0..WARM + FRAMES {
                if k == WARM {
                    before = device.work_counts();
                }
                let cpu = process_cpu_us();
                let mut made = Vec::new();
                for (at, &index) in chosen.iter().enumerate() {
                    let (_, w, h, bpr) = LAYOUTS[index];
                    let began = Instant::now();
                    let (buffer, texture) = match over_private {
                        false => {
                            let buffer = surfaces[index].buffer(&device, lens[index], None);
                            let texture = plane_texture(&buffer, w, h, bpr);
                            (buffer, texture)
                        }
                        true => {
                            let buffer = private_buffers[index].clone();
                            let texture = private_texture(&buffer, w, h, bpr);
                            (buffer, texture)
                        }
                    };
                    if k >= WARM {
                        makes[at].push(since(began));
                    }
                    made.push((texture, buffer));
                }
                if k == 0 {
                    forms = made
                        .iter()
                        .map(|(texture, _)| if aliased(texture) { 'a' } else { 's' })
                        .collect();
                }
                let textures: Vec<Texture> =
                    made.iter().map(|(texture, _)| texture.clone()).collect();
                let (encoded, waited) = frame(&queue, &textures, &privates, (part, drawing));
                let began = Instant::now();
                drop((textures, made));
                let released = since(began);
                if k >= WARM {
                    releases.push(released);
                    encodes.push(encoded);
                    waits.push(waited);
                    cpus.push(process_cpu_us() - cpu);
                }
            }
            let after = device.work_counts();
            let forms: String = forms.into_iter().collect();
            let makes: Vec<String> = chosen
                .iter()
                .zip(makes)
                .map(|(&index, times)| {
                    let (median, p90) = median_p90(times);
                    format!("{} {median:.0} (p90 {p90:.0})", LAYOUTS[index].0)
                })
                .collect();
            let (release, release90) = median_p90(releases);
            let (encode, encode90) = median_p90(encodes);
            let (wait, wait90) = median_p90(waits);
            let (cpu, cpu90) = median_p90(cpus);
            println!(
                "SURFTEX-PART on {}: {set}, {part:?}, forms {forms} (a aliased, s \
                 shadowed): \
                 making us {}; release {release:.0} us (p90 {release90:.0}), encode {encode:.0} us \
                 (p90 {encode90:.0}), commit to completion {wait:.0} us (p90 {wait90:.0}), process \
                 CPU {cpu:.0} us (p90 {cpu90:.0}); per frame: {}",
                device.name(),
                makes.join(", "),
                counts_per_frame(&before, &after, FRAMES),
            );
        }
    }
    drop((surfaces, private_buffers));
    let notices = super::degrade_notices::counts(&messages.take_notices());
    println!("SURFTEX-NOTICES on {}: {notices:?}", device.name());
}

/// What a shadow's copies cost apart from the passes around them, on the browser's 1307x512
/// `BGRA8Unorm` tile at rows of 5248 bytes: each copy alone in a command buffer, committed and
/// waited for, between a private texture made as a shadow is (render target and shader read
/// usage) and a no-copy buffer over anonymous pages (a copy back's way, and a copy in's), and, to
/// tell the image's tiling from the pages' memory, between the texture and a private buffer and a
/// `Shared` one, and between a private buffer and the pages. Printed per copy
/// (`SURFTEX-COPY`): the median and 90th percentile of commit to completion.
#[test]
#[ignore = "a release-mode measurement: run by name"]
fn shadow_copy_costs() {
    const BLITS: usize = 50;
    let (w, h, bpr) = (1307, 512, 5248);
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    let queue = device.new_command_queue().unwrap();
    let len = Surface::len(bpr, h);
    let pages = Surface::Anonymous(Mapping::anonymous(len));
    let host = pages.buffer(&device, len, None);
    let private = device
        .new_buffer(len as u64, ResourceOptions::private())
        .unwrap();
    let shared = device
        .new_buffer(len as u64, ResourceOptions::shared())
        .unwrap();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, w, h, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let image = &device.new_texture(&d).unwrap();
    let size = Size::new(w, h, 1);
    let origin = Origin::new(0, 0, 0);
    // Each copy: the texture into a buffer, a buffer into the texture, or one buffer into another.
    let copies: [(&str, Option<&Buffer>, Option<&Buffer>); 6] = [
        ("image to pages (a copy back)", None, Some(&host)),
        ("pages to image (a copy in)", Some(&host), None),
        ("image to private buffer", None, Some(&private)),
        ("image to shared buffer", None, Some(&shared)),
        ("private buffer to pages", Some(&private), Some(&host)),
        ("pages to private buffer", Some(&host), Some(&private)),
    ];
    for (name, from, to) in copies {
        let mut waits = Vec::new();
        for k in 0..WARM + BLITS {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            match (from, to) {
                (None, Some(to)) => blit.copy_from_texture_to_buffer(
                    image,
                    0,
                    0,
                    origin,
                    size,
                    to,
                    0,
                    bpr,
                    bpr * h,
                    BlitOption::None,
                ),
                (Some(from), None) => blit.copy_from_buffer_to_texture(
                    from,
                    0,
                    bpr,
                    bpr * h,
                    size,
                    image,
                    0,
                    0,
                    origin,
                    BlitOption::None,
                ),
                (Some(from), Some(to)) => blit.copy_from_buffer_to_buffer(from, 0, to, 0, bpr * h),
                (None, None) => unreachable!("every copy has a buffer"),
            }
            .unwrap();
            blit.end_encoding().unwrap();
            let committed = Instant::now();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
            if k >= WARM {
                waits.push(since(committed));
            }
        }
        let (wait, wait90) = median_p90(waits);
        println!(
            "SURFTEX-COPY on {}: {name}, {w}x{h} at {bpr} bytes a row: commit to completion \
             {wait:.0} us (p90 {wait90:.0})",
            device.name()
        );
    }
    // The copies, then the buffers, then the pages they lie over go, in that order.
}

/// A compositor's frame over textures that only read: each frame, for each padded layout of
/// [`LAYOUTS`] (the tight one left out), a no-copy buffer over its surface's anonymous pages and a
/// texture of its plane over it with `ShaderRead` usage alone; one pass into a private target of
/// the browser's tile, drawing each texture over the whole target through a nearest sampler; one
/// command buffer, committed and waited for; then the textures and buffers are released. Or, where
/// `reused`, the textures are made once and every frame samples them; each texture a shadow,
/// copied in before the pass; a device each, the two interleaved in rounds so a drift of the
/// host's falls on both. Printed per reuse and round (`SURFTEX-READS`): the textures' forms, the
/// medians and 90th percentiles of a texture's making, the frame's encoding, commit to completion
/// and process CPU time, and the shadow copies a frame.
#[test]
#[ignore = "a release-mode measurement: run by name"]
fn read_only_padded_costs() {
    use super::pitched_surfaces::{reading, sampler};
    use metal2vulkan::metal::SamplerMinMagFilter;
    let chosen: Vec<usize> = (0..LAYOUTS.len())
        .filter(|&i| LAYOUTS[i].3 != LAYOUTS[i].1 * 4)
        .collect();
    let surfaces: Vec<Surface> = LAYOUTS
        .iter()
        .map(|&(_, _, h, bpr)| Surface::Anonymous(Mapping::anonymous(Surface::len(bpr, h))))
        .collect();
    for round in 0..3 {
        for reused in [false, true] {
            let cache = ScratchCache::new().unwrap();
            let messages = Messages::default();
            let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
                options.layers.clear();
            })
            .unwrap();
            messages.take_notices();
            let queue = device.new_command_queue().unwrap();
            let pipeline = reading(&device, PixelFormat::BGRA8Unorm, "tob_sample");
            let nearest = sampler(&device, SamplerMinMagFilter::Nearest);
            let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
            let (_, tw, th, _) = LAYOUTS[0];
            let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, tw, th, false);
            d.usage = TextureUsage::RENDER_TARGET;
            d.resource_options = ResourceOptions::private();
            let target = device.new_texture(&d).unwrap();
            let mut forms = String::new();
            let mut makes = Vec::new();
            let (mut encodes, mut waits, mut cpus) = (Vec::new(), Vec::new(), Vec::new());
            let mut before = device.work_counts();
            let mut made = Vec::new();
            for k in 0..WARM + FRAMES {
                if k == WARM {
                    before = device.work_counts();
                }
                let cpu = process_cpu_us();
                let making = if reused { k == 0 } else { true };
                for &index in chosen.iter().filter(|_| making) {
                    let (_, w, h, bpr) = LAYOUTS[index];
                    let len = Surface::len(bpr, h);
                    let began = Instant::now();
                    let buffer = surfaces[index].buffer(&device, len, None);
                    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, w, h, false);
                    d.usage = TextureUsage::SHADER_READ;
                    d.resource_options = buffer.resource_options();
                    let texture = buffer
                        .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, bpr, false))
                        .unwrap();
                    if k >= WARM || reused {
                        makes.push(since(began));
                    }
                    made.push((texture, buffer));
                }
                if k == 0 {
                    forms = made
                        .iter()
                        .map(|(texture, _)| if aliased(texture) { 'a' } else { 's' })
                        .collect();
                }
                let began = Instant::now();
                let command_buffer = queue.command_buffer().unwrap();
                let mut pass = RenderPassDescriptor::new();
                let attachment = &mut pass.color_attachments[0];
                attachment.texture = Some(target.clone());
                attachment.load_action = LoadAction::Clear;
                attachment.store_action = StoreAction::Store;
                let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
                encoder.set_render_pipeline_state(&pipeline).unwrap();
                encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
                encoder
                    .set_fragment_sampler_state(Some(&nearest), 0)
                    .unwrap();
                for (texture, _) in &made {
                    encoder.set_fragment_texture(Some(texture), 0).unwrap();
                    encoder
                        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                        .unwrap();
                }
                encoder.end_encoding().unwrap();
                let encoded = since(began);
                let committed = Instant::now();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None);
                let waited = since(committed);
                if !reused {
                    made.clear();
                }
                if k >= WARM {
                    encodes.push(encoded);
                    waits.push(waited);
                    cpus.push(process_cpu_us() - cpu);
                }
            }
            drop(made);
            let after = device.work_counts();
            let copies = (after.shadow_copies - before.shadow_copies) as f64 / FRAMES as f64;
            let (make, make90) = median_p90(makes);
            let (encode, encode90) = median_p90(encodes);
            let (wait, wait90) = median_p90(waits);
            let (cpu, cpu90) = median_p90(cpus);
            println!(
                "SURFTEX-READS on {}: round {round}, reused {reused}, forms {forms} \
                 (a aliased, s shadowed): making {make:.0} us a texture (p90 {make90:.0}), encode \
                 {encode:.0} us (p90 {encode90:.0}), commit to completion {wait:.0} us (p90 \
                 {wait90:.0}), process CPU {cpu:.0} us (p90 {cpu90:.0}); shadow copies a frame \
                 {copies:.2}",
                device.name(),
            );
        }
    }
    drop(surfaces);
}
