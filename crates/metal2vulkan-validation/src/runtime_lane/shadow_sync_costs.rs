//! What a compositor's surface costs a command buffer as a shadow over host pages, on the device
//! the cfg selects: report only, in release builds. The surface is a 3840x2400 `BGRA8Unorm`
//! texture of an IOSurface's plane (rows of 15360 bytes) over a `Shared` no-copy buffer of
//! anonymous pages, with `RenderTarget` and `ShaderRead` usage, as a guest's window surface is:
//! a shadow on every device (its usage renders into it). Its memory carries a log of the pages'
//! writes or none (`HostMemory::with_dirty_pages`): the test's own, exact to the page, which keeps
//! writers apart and publishes at once.
//!
//! Each shape runs on a device of its own, one command buffer a frame, committed with
//! `commit_and_submit` (as the consumer commits), in two rounds:
//!
//! - `Write`: a pass loading the surface draws one opaque 512x512 rectangle inside a scissor of
//!   its size, its color alternating frame to frame, and a second pass samples the surface whole
//!   into a `Private` target (one write and one read).
//! - `WriteUnscissored`: the same rectangle drawn with no scissor.
//! - `WriteReadApart`: as `Write`, the read through a second texture of the plane whose usage only
//!   samples it (as another process's texture over the same surface): in place where the
//!   device's check passed, a device copy on a discrete GPU over logged pages.
//! - `Rewrite`: as `Write`, the same color every frame: the GPU writes and no word changes.
//! - `Idle`: the read alone; nothing writes the surface.
//! - `CpuWrite`: the CPU writes 64 rows of the surface before the commit, told to the log; the
//!   read alone.
//! - `Private`: `Write` into a `Private` texture of the surface's size: the passes alone.
//!
//! Printed per round, shape and log (`SHADOW-SYNC`): the medians and 90th percentiles of a frame's
//! `commit_and_submit` (host), its commit to the kernel's end of submission, its GPU time
//! (`GPUEndTime` less `GPUStartTime`), the GPU end to its completed handler, and the process's CPU
//! time; and the work counts per frame.

use super::first_use_writes::CHECK_WAIT;
use super::host_memory::Mapping;
use super::no_copy_textures::{floats, rects, staged};
use super::pitched_surfaces::{reading, sampler};
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, SamplerMinMagFilter, TextureUsage};
use metal2vulkan_runtime::{
    host_time, Buffer, CommandBuffer, Device, DirtyPages, DirtyStamp, HostMemory, IoSurfacePlane,
    LoadAction, RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState,
    ScissorRect, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::{Arc, Mutex};

const WARM: usize = 10;
const FRAMES: usize = 60;
const WIDTH: u64 = 3840;
const HEIGHT: u64 = 2400;
const ROW: u64 = WIDTH * 4;
/// The side of the rectangle a write draws.
const SIDE: u64 = 512;
/// The rows a CPU write writes.
const CPU_ROWS: u64 = 64;
/// The log's page.
const PAGE: u64 = 4096;

/// What a frame does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Write,
    WriteUnscissored,
    WriteReadApart,
    Rewrite,
    Idle,
    CpuWrite,
    Private,
}

const SHAPES: [Shape; 7] = [
    Shape::Write,
    Shape::WriteUnscissored,
    Shape::WriteReadApart,
    Shape::Rewrite,
    Shape::Idle,
    Shape::CpuWrite,
    Shape::Private,
];

/// A log of the writes to a mapping's pages, as a consumer keeps one: each write stamped with the
/// publication it falls in and its writer (0 the consumer's), each stamp the runtime asks for a
/// new publication, published at once.
#[derive(Default)]
struct Log {
    state: Mutex<LogState>,
}

#[derive(Default)]
struct LogState {
    /// The publication writes fall in now.
    now: u64,
    /// Each write's pages, stamp and writer.
    writes: Vec<(Range<u64>, u64, u64)>,
}

impl Log {
    /// Note the CPU's write of the bytes `range`, whole pages.
    fn mark(&self, range: Range<u64>) {
        self.note(range, 0);
    }

    fn note(&self, range: Range<u64>, writer: u64) {
        let mut state = self.state.lock().unwrap();
        let pages = range.start / PAGE * PAGE..range.end.div_ceil(PAGE) * PAGE;
        let now = state.now;
        // Only the latest write of each page and writer counts: the log answers "after".
        state
            .writes
            .retain(|(kept, _, by)| !(*by == writer && *kept == pages));
        state.writes.push((pages, now, writer));
    }
}

impl DirtyPages for Log {
    type Watch = ();

    fn watch(&self, _: Range<u64>) {}

    fn request(&self) -> DirtyStamp {
        let mut state = self.state.lock().unwrap();
        let stamp = state.now;
        state.now += 1;
        DirtyStamp(stamp)
    }

    fn wait(&self, _: DirtyStamp) -> bool {
        true
    }

    fn changed(&self, since: DirtyStamp, out: &mut Vec<Range<u64>>) {
        let state = self.state.lock().unwrap();
        let later = state.writes.iter().filter(|(_, stamp, _)| *stamp > since.0);
        out.extend(later.map(|(pages, _, _)| pages.clone()));
    }

    fn wrote(&self, range: Range<u64>) {
        self.note(range, 0);
    }

    fn wrote_as(&self, range: Range<u64>, writer: NonZeroU64) {
        self.note(range, writer.get());
    }

    fn changed_except(&self, since: DirtyStamp, writer: NonZeroU64, out: &mut Vec<Range<u64>>) {
        let state = self.state.lock().unwrap();
        let later =
            (state.writes.iter()).filter(|(_, stamp, by)| *stamp > since.0 && *by != writer.get());
        out.extend(later.map(|(pages, _, _)| pages.clone()));
    }
}

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

fn median_p90(mut values: Vec<f64>) -> String {
    values.sort_by(f64::total_cmp);
    let at = |p: usize| values[(values.len() * p / 100).min(values.len() - 1)];
    format!("{:.0} (p90 {:.0})", at(50), at(90))
}

/// The surface's buffer: a `Shared` no-copy buffer over `mapping`'s `len` bytes, its memory
/// carrying `log` where given.
fn surface_buffer(
    device: &Device,
    mapping: &Mapping,
    len: usize,
    log: Option<&Arc<Log>>,
) -> Buffer {
    // SAFETY: the mapping outlives every buffer over it (the test drops them first).
    let memory = unsafe { HostMemory::new(mapping.pointer()) };
    let memory = match log {
        Some(log) => memory.with_dirty_pages(Arc::clone(log)),
        None => memory,
    };
    // SAFETY: as above.
    unsafe {
        device.new_buffer_with_bytes_no_copy(memory, len as u64, ResourceOptions::default(), None)
    }
    .unwrap()
}

fn plane_texture(buffer: &Buffer, usage: TextureUsage) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    d.usage = usage;
    d.resource_options = buffer.resource_options();
    buffer
        .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, ROW, false))
        .unwrap()
}

/// What the frames draw and read with.
struct Drawing {
    opaque: RenderPipelineState,
    whole: Buffer,
    /// The rectangle a write draws with no scissor, at the surface's top left.
    corner: Buffer,
    colors: [Buffer; 2],
    read: RenderPipelineState,
    nearest: SamplerState,
}

impl Drawing {
    fn new(device: &Device) -> Drawing {
        let x1 = -1.0 + 2.0 * SIDE as f32 / WIDTH as f32;
        let y1 = 1.0 - 2.0 * SIDE as f32 / HEIGHT as f32;
        Drawing {
            opaque: rects(device, PixelFormat::BGRA8Unorm, false),
            whole: staged(device, &floats([-1.0, -1.0, 1.0, 1.0])),
            corner: staged(device, &floats([-1.0, y1, x1, 1.0])),
            colors: [
                staged(device, &floats([0.25, 0.5, 0.75, 1.0])),
                staged(device, &floats([0.75, 0.5, 0.25, 1.0])),
            ],
            read: reading(device, PixelFormat::BGRA8Unorm, "tob_sample"),
            nearest: sampler(device, SamplerMinMagFilter::Nearest),
        }
    }

    /// A pass loading `texture` that draws the rectangle in color `color`, inside a scissor of
    /// its size at the top left where `scissored`.
    fn write(
        &self,
        command_buffer: &CommandBuffer,
        texture: &Texture,
        color: usize,
        scissored: bool,
    ) {
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(texture.clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.opaque).unwrap();
        let rect = match scissored {
            true => {
                (encoder.set_scissor_rect(ScissorRect::new(0, 0, SIDE, SIDE))).unwrap();
                &self.whole
            }
            false => &self.corner,
        };
        encoder.set_vertex_buffer(Some(rect), 0, 0).unwrap();
        (encoder.set_fragment_buffer(Some(&self.colors[color]), 0, 0)).unwrap();
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        encoder.end_encoding().unwrap();
    }

    /// A pass sampling `texture` whole into `target`.
    fn read(&self, command_buffer: &CommandBuffer, texture: &Texture, target: &Texture) {
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(target.clone());
        attachment.load_action = LoadAction::DontCare;
        attachment.store_action = StoreAction::Store;
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.read).unwrap();
        encoder.set_vertex_buffer(Some(&self.whole), 0, 0).unwrap();
        (encoder.set_fragment_sampler_state(Some(&self.nearest), 0)).unwrap();
        encoder.set_fragment_texture(Some(texture), 0).unwrap();
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        encoder.end_encoding().unwrap();
    }
}

fn per_frame(before: &WorkCounts, after: &WorkCounts) -> String {
    let per = |f: fn(&WorkCounts) -> u64| (f(after) - f(before)) as f64 / FRAMES as f64;
    format!(
        "vkQueueSubmit {:.2} ({:.0} us in the driver), command buffers {:.2}, at commit {:.2}, \
         shadow copies {:.2}, texels in {:.0}, back {:.0}, kept {:.0}, gated {:.2} (whole {:.2}, \
         skipped {:.2}), bytes copied in {:.0} of {:.0}, log waits {:.2} ({:.0} us), merge words \
         examined {:.0}, merges in blocks {:.2}, merges kept in images {:.2} (images made \
         {:.2}), scratches reused {:.2}, zeroed {:.2}, device memory allocated {:.0} bytes, \
         internal submissions {:.2}",
        per(|c| c.queue_submissions),
        per(|c| c.batch_submit_nanos) / 1e3,
        per(|c| c.command_buffers_submitted),
        per(|c| c.batches_submitted_at_commit),
        per(|c| c.shadow_copies),
        per(|c| c.shadow_texels_copied_in),
        per(|c| c.shadow_texels_copied_back),
        per(|c| c.shadow_texels_kept),
        per(|c| c.dirty_copies_in_gated),
        per(|c| c.dirty_copies_in_whole),
        per(|c| c.dirty_copies_in_skipped),
        per(|c| c.dirty_copy_in_bytes_copied),
        per(|c| c.dirty_copy_in_bytes_whole),
        per(|c| c.dirty_waits),
        per(|c| c.dirty_wait_nanos) / 1e3,
        per(|c| c.shadow_merge_words_examined),
        per(|c| c.shadow_merge_dispatches_in_blocks),
        per(|c| c.shadow_merges_kept_in_images),
        per(|c| c.shadow_merge_kept_images_made),
        per(|c| c.shadow_merge_scratches_reused),
        per(|c| c.shadow_merge_scratches_zeroed),
        per(|c| c.device_memory_allocated),
        per(|c| c.internal_submissions),
    )
}

/// The frames of each [`Shape`] over the surface, with and without a log, each on a device of its
/// own, interleaved in two rounds.
#[test]
#[ignore = "a release-mode measurement: run by name"]
fn shadow_sync_costs() {
    let len = ((ROW * HEIGHT) as usize).next_multiple_of(16384);
    for round in 0..2 {
        for shape in SHAPES {
            let logs: &[bool] = match shape {
                Shape::Private => &[false],
                _ => &[false, true],
            };
            for &logged in logs {
                frames(round, shape, logged, len);
            }
        }
    }
}

fn frames(round: usize, shape: Shape, logged: bool, len: usize) {
    let mapping = Mapping::anonymous(len);
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |o| {
        o.layers.clear();
    })
    .unwrap();
    let queue = device.new_command_queue().unwrap();
    let drawing = Drawing::new(&device);
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let target = device.new_texture(&d).unwrap();
    let private = device.new_texture(&d).unwrap();
    let log = Arc::new(Log::default());
    let buffer = surface_buffer(&device, &mapping, len, logged.then_some(&log));
    let surface = match shape {
        Shape::Private => private.clone(),
        _ => plane_texture(
            &buffer,
            TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        ),
    };
    // The sampled texture's form follows the device's in-place check: a texture over host pages
    // starts it.
    let sampled = (shape == Shape::WriteReadApart).then(|| {
        let first = plane_texture(&buffer, TextureUsage::SHADER_READ);
        drop(first);
        device.settle_in_place_checks(CHECK_WAIT);
        plane_texture(&buffer, TextureUsage::SHADER_READ)
    });
    let log_form = match logged {
        true => "logged",
        false => "no log",
    };
    // SAFETY: the handle is only compared, while the texture lives.
    let sampled_form = match sampled.as_ref().map(|t| unsafe { t.vk_image() }.is_some()) {
        None => "",
        Some(true) => ", read in place",
        Some(false) => ", read through a copy",
    };
    let form = match shape {
        Shape::Private => "private".to_owned(),
        _ => format!("shadow, {log_form}{sampled_form}"),
    };
    let (mut commits, mut submits, mut gpus, mut gaps, mut waits, mut cpus) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let mut before = device.work_counts();
    for k in 0..WARM + FRAMES {
        if k == WARM {
            before = device.work_counts();
        }
        let cpu = process_cpu_us();
        let command_buffer = queue.command_buffer().unwrap();
        match shape {
            Shape::Write | Shape::Private => {
                drawing.write(&command_buffer, &surface, k % 2, true);
                drawing.read(&command_buffer, &surface, &target);
            }
            Shape::WriteUnscissored => {
                drawing.write(&command_buffer, &surface, k % 2, false);
                drawing.read(&command_buffer, &surface, &target);
            }
            Shape::WriteReadApart => {
                drawing.write(&command_buffer, &surface, k % 2, true);
                let sampled = sampled.as_ref().unwrap();
                drawing.read(&command_buffer, sampled, &target);
            }
            Shape::Rewrite => {
                drawing.write(&command_buffer, &surface, 0, true);
                drawing.read(&command_buffer, &surface, &target);
            }
            Shape::Idle => drawing.read(&command_buffer, &surface, &target),
            Shape::CpuWrite => {
                let first = (k as u64 * 97) % (HEIGHT - CPU_ROWS);
                let bytes = first * ROW..(first + CPU_ROWS) * ROW;
                mapping.write(
                    bytes.start as usize,
                    &vec![k as u8; (CPU_ROWS * ROW) as usize],
                );
                log.mark(bytes);
                drawing.read(&command_buffer, &surface, &target);
            }
        }
        let handled = Arc::new(Mutex::new(None::<f64>));
        let at = Arc::clone(&handled);
        (command_buffer.add_completed_handler(Box::new(move |_| {
            *at.lock().unwrap() = Some(host_time());
        })))
        .unwrap();
        let committed = host_time();
        command_buffer.commit_and_submit().unwrap();
        let returned = host_time();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let completed = host_time();
        assert_eq!(command_buffer.error(), None);
        // The completed handler may run after the wait returns.
        let handler = loop {
            if let Some(at) = *handled.lock().unwrap() {
                break at;
            }
            std::thread::yield_now();
        };
        if k >= WARM {
            let start = command_buffer.gpu_start_time().unwrap();
            let end = command_buffer.gpu_end_time().unwrap();
            commits.push((returned - committed) * 1e6);
            submits.push((command_buffer.kernel_end_time() - committed) * 1e6);
            gpus.push((end - start) * 1e6);
            gaps.push((handler - end) * 1e6);
            waits.push((completed - committed) * 1e6);
            cpus.push(process_cpu_us() - cpu);
        }
    }
    let after = device.work_counts();
    println!(
        "SHADOW-SYNC on {}: round {round}, {shape:?}, {form}: commit_and_submit {} us, commit to \
         submitted {} us, GPU {} us, GPU end to completed handler {} us, commit to completion {} \
         us, process CPU {} us; per frame: {}",
        device.name(),
        median_p90(commits),
        median_p90(submits),
        median_p90(gpus),
        median_p90(gaps),
        median_p90(waits),
        median_p90(cpus),
        per_frame(&before, &after),
    );
    drop((surface, sampled, buffer, private, target));
    drop(mapping);
}
