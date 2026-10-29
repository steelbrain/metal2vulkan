//! What a guest's frame of render passes into its window's surface costs, on the device the cfg
//! selects: report only, in release builds. The surface is a 1920x1080 `BGRA8Unorm` IOSurface of
//! packed rows (7680 bytes) over anonymous pages, a `Managed` no-copy buffer over them and a
//! texture of its plane, each pass loading and storing it and blending three rectangles (the whole
//! surface, then two quarters). Printed per round and shape (`GUEST-FRAME`): the texture's form,
//! the medians and 90th percentiles of the frame's commit to completion (from the first commit to
//! the last command buffer's completion), its GPU time (each command buffer's `GPUEndTime` less
//! `GPUStartTime`, summed) and the process's CPU time, each busy thread's CPU time per frame (the
//! encoding thread, which records its Vulkan commands, the scheduler, the queue's waiter), and the work counts per frame (the merges' kept images live at the end too).

use super::host_memory::Mapping;
use super::no_copy_textures::{blending, floats, staged};
use super::pitched_surfaces::{reading, sampler};
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, SamplerMinMagFilter, TextureUsage};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CpuCacheMode, Device, HazardTrackingMode, HostMemory, IoSurfacePlane,
    LoadAction, Origin, RenderPassDescriptor, ResourceOptions, Size, StorageMode, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use std::time::Instant;

const WARM: usize = 10;
const FRAMES: usize = 60;
/// The passes of a frame of several.
const PASSES: usize = 12;
const WIDTH: u64 = 1920;
const HEIGHT: u64 = 1080;
const ROW: u64 = WIDTH * 4;

/// What a frame does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    /// One command buffer, one pass into the surface's texture.
    Pass,
    /// One command buffer, [`PASSES`] passes into the surface's texture, an encoder each.
    Passes,
    /// [`PASSES`] command buffers of one pass each into the surface's texture.
    PassesApart,
    /// As [`Shape::Pass`], into a `Private` texture of the surface's size: the pass alone.
    PrivatePass,
    /// As [`Shape::Passes`], into the `Private` texture: the passes alone.
    PrivatePasses,
    /// One command buffer: blits of the surface's bytes into the `Private` texture and of the
    /// texture into a private buffer, [`PASSES`] passes into the texture, blits of the texture
    /// into a second private buffer and of that into the surface's bytes: one shadow's copy in,
    /// keep, changed copy and copy back for every pass of the command buffer.
    SharedShadow,
    /// The consumer's present: a new no-copy buffer and texture over the surface each frame,
    /// sampled by one draw into a `Private` target, beside the buffer kept across frames.
    Presented,
}

const SHAPES: [Shape; 7] = [
    Shape::Pass,
    Shape::PrivatePass,
    Shape::Passes,
    Shape::PassesApart,
    Shape::PrivatePasses,
    Shape::SharedShadow,
    Shape::Presented,
];

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

/// Each thread of the process: its id, its name (Linux keeps 15 bytes of it) and the CPU time it
/// ran, in microseconds.
fn thread_cpu_us() -> Vec<(i32, String, f64)> {
    let mut threads = Vec::new();
    for task in std::fs::read_dir("/proc/self/task").unwrap().flatten() {
        let Some(tid) = task
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<i32>().ok())
        else {
            continue;
        };
        let name = std::fs::read_to_string(task.path().join("comm")).unwrap_or_default();
        // The clock of one thread's CPU time: `MAKE_THREAD_CPUCLOCK(tid, CPUCLOCK_SCHED)`.
        let clock = (!tid << 3) | 6;
        let mut now = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `now` is a valid timespec for the call to fill; a thread gone meanwhile fails
        // the call, and is left out.
        if unsafe { libc::clock_gettime(clock, &mut now) } == 0 {
            let us = now.tv_sec as f64 * 1e6 + now.tv_nsec as f64 / 1e3;
            threads.push((tid, name.trim().to_owned(), us));
        }
    }
    threads
}

/// The CPU time per frame each thread ran from `before` to `after` (of [`thread_cpu_us`]), by
/// name, the calling thread's as `encoding`, the busiest first, those under 1 us a frame left out.
fn threads_per_frame(before: &[(i32, String, f64)], after: &[(i32, String, f64)]) -> String {
    // SAFETY: no precondition.
    let me = unsafe { libc::gettid() };
    let mut by_name: Vec<(String, f64)> = Vec::new();
    for (tid, name, us) in after {
        let start = before
            .iter()
            .find(|(had, _, _)| had == tid)
            .map_or(0.0, |(_, _, us)| *us);
        let name = match *tid == me {
            true => "encoding".to_owned(),
            false => name.clone(),
        };
        let ran = (us - start) / FRAMES as f64;
        match by_name.iter_mut().find(|(known, _)| *known == name) {
            Some((_, total)) => *total += ran,
            None => by_name.push((name, ran)),
        }
    }
    by_name.sort_by(|a, b| b.1.total_cmp(&a.1));
    let busy: Vec<String> = (by_name.iter())
        .filter(|(_, us)| *us >= 1.0)
        .map(|(name, us)| format!("{name} {us:.0}"))
        .collect();
    busy.join(", ")
}

fn median_p90(mut values: Vec<f64>) -> (f64, f64) {
    values.sort_by(f64::total_cmp);
    let at = |p: usize| values[(values.len() * p / 100).min(values.len() - 1)];
    (at(50), at(90))
}

fn managed() -> ResourceOptions {
    ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    )
}

/// A `Managed` no-copy buffer over `mapping`'s first `len` bytes.
fn surface_buffer(device: &Device, mapping: &Mapping, len: usize) -> Buffer {
    // SAFETY: the mapping outlives every buffer over it (the test drops them first).
    let memory = unsafe { HostMemory::new(mapping.pointer()) };
    // SAFETY: as above.
    unsafe { device.new_buffer_with_bytes_no_copy(memory, len as u64, managed(), None) }.unwrap()
}

fn plane_texture(buffer: &Buffer, usage: TextureUsage) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    d.usage = usage;
    d.resource_options = buffer.resource_options();
    buffer
        .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, ROW, false))
        .unwrap()
}

fn form_of(texture: &Texture) -> &'static str {
    // SAFETY: the handle is only compared, while the texture lives.
    match unsafe { texture.vk_image() }.is_some() {
        true => "aliased",
        false => "shadowed",
    }
}

/// What a pass draws.
struct Drawing {
    blend: metal2vulkan_runtime::RenderPipelineState,
    rects: [Buffer; 3],
    color: Buffer,
}

impl Drawing {
    fn pass(&self, command_buffer: &CommandBuffer, texture: &Texture) {
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(texture.clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.blend).unwrap();
        encoder
            .set_fragment_buffer(Some(&self.color), 0, 0)
            .unwrap();
        for rect in &self.rects {
            encoder.set_vertex_buffer(Some(rect), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
        }
        encoder.end_encoding().unwrap();
    }
}

fn per_frame(before: &WorkCounts, after: &WorkCounts) -> String {
    let per = |f: fn(&WorkCounts) -> u64| (f(after) - f(before)) as f64 / FRAMES as f64;
    format!(
        "vkQueueSubmit {:.2}, command buffers {:.2}, internal submissions {:.2}, shadows made \
         {:.2}, zeroed {:.2}, shadow copies {:.2}, texels in {:.0}, back {:.0}, copies back \
         dispatched {:.2}, copies in dispatched {:.2}, merge copies dispatched {:.2}, merge \
         scratches reused {:.2}, zeroed {:.2}, merges kept in images {:.2}, kept images made \
         {:.2}, taken ahead of a holder's completion {:.2}, live at the end {}, owned descriptor \
         pools {:.2}",
        per(|c| c.queue_submissions),
        per(|c| c.command_buffers_submitted),
        per(|c| c.internal_submissions),
        per(|c| c.shadowed_textures),
        per(|c| c.shadows_zeroed),
        per(|c| c.shadow_copies),
        per(|c| c.shadow_texels_copied_in),
        per(|c| c.shadow_texels_copied_back),
        per(|c| c.shadow_copies_back_dispatched),
        per(|c| c.shadow_copies_in_dispatched),
        per(|c| c.shadow_merge_copies_dispatched),
        per(|c| c.shadow_merge_scratches_reused),
        per(|c| c.shadow_merge_scratches_zeroed),
        per(|c| c.shadow_merges_kept_in_images),
        per(|c| c.shadow_merge_kept_images_made),
        per(|c| c.shadow_merge_kept_images_followed),
        after.shadow_merge_kept_images_live,
        per(|c| c.owned_descriptor_pools),
    )
}

/// The guest's frames over its window's surface, each [`Shape`] on a device of its own,
/// interleaved in two rounds.
#[test]
#[ignore = "a release-mode measurement: run by name"]
fn guest_frame_costs() {
    let len = ((ROW * HEIGHT) as usize).next_multiple_of(16384);
    let mapping = Mapping::anonymous(len);
    for round in 0..2 {
        for shape in SHAPES {
            let cache = ScratchCache::new().unwrap();
            let messages = Messages::default();
            let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |o| {
                o.layers.clear();
            })
            .unwrap();
            let queue = device.new_command_queue().unwrap();
            let drawing = Drawing {
                blend: blending(&device, PixelFormat::BGRA8Unorm),
                rects: [
                    staged(&device, &floats([-1.0, -1.0, 1.0, 1.0])),
                    staged(&device, &floats([-1.0, -1.0, 0.0, 0.0])),
                    staged(&device, &floats([0.0, 0.0, 1.0, 1.0])),
                ],
                color: staged(&device, &floats([0.25, 0.5, 0.75, 0.5])),
            };
            let read = reading(&device, PixelFormat::BGRA8Unorm, "tob_sample");
            let nearest = sampler(&device, SamplerMinMagFilter::Nearest);
            let mut d =
                TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
            d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            d.resource_options = ResourceOptions::private();
            let private = device.new_texture(&d).unwrap();
            let size = ROW * HEIGHT;
            let scratches = [
                device.new_buffer(size, ResourceOptions::private()).unwrap(),
                device.new_buffer(size, ResourceOptions::private()).unwrap(),
            ];
            let kept = surface_buffer(&device, &mapping, len);
            let surface = plane_texture(
                &kept,
                TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            );
            let mut form = match shape {
                Shape::PrivatePass | Shape::PrivatePasses | Shape::SharedShadow => "private",
                _ => form_of(&surface),
            };
            let (mut waits, mut gpus, mut cpus) = (Vec::new(), Vec::new(), Vec::new());
            let mut before = device.work_counts();
            let mut threads = thread_cpu_us();
            for k in 0..WARM + FRAMES {
                if k == WARM {
                    before = device.work_counts();
                    threads = thread_cpu_us();
                }
                let cpu = process_cpu_us();
                let mut command_buffers = vec![queue.command_buffer().unwrap()];
                let mut presented = None;
                match shape {
                    Shape::Pass => drawing.pass(&command_buffers[0], &surface),
                    Shape::PrivatePass => drawing.pass(&command_buffers[0], &private),
                    Shape::Passes => {
                        for _ in 0..PASSES {
                            drawing.pass(&command_buffers[0], &surface);
                        }
                    }
                    Shape::PrivatePasses => {
                        for _ in 0..PASSES {
                            drawing.pass(&command_buffers[0], &private);
                        }
                    }
                    Shape::PassesApart => {
                        drawing.pass(&command_buffers[0], &surface);
                        for _ in 1..PASSES {
                            let command_buffer = queue.command_buffer().unwrap();
                            drawing.pass(&command_buffer, &surface);
                            command_buffers.push(command_buffer);
                        }
                    }
                    Shape::SharedShadow => {
                        let command_buffer = &command_buffers[0];
                        let at = Origin::new(0, 0, 0);
                        let extent = Size::new(WIDTH, HEIGHT, 1);
                        let mut blit = command_buffer.blit_command_encoder().unwrap();
                        blit.copy_from_buffer_to_texture(
                            &kept,
                            0,
                            ROW,
                            size,
                            extent,
                            &private,
                            0,
                            0,
                            at,
                            metal2vulkan_runtime::BlitOption::None,
                        )
                        .unwrap();
                        blit.copy_from_texture_to_buffer(
                            &private,
                            0,
                            0,
                            at,
                            extent,
                            &scratches[0],
                            0,
                            ROW,
                            size,
                            metal2vulkan_runtime::BlitOption::None,
                        )
                        .unwrap();
                        blit.end_encoding().unwrap();
                        for _ in 0..PASSES {
                            drawing.pass(command_buffer, &private);
                        }
                        let mut blit = command_buffer.blit_command_encoder().unwrap();
                        blit.copy_from_texture_to_buffer(
                            &private,
                            0,
                            0,
                            at,
                            extent,
                            &scratches[1],
                            0,
                            ROW,
                            size,
                            metal2vulkan_runtime::BlitOption::None,
                        )
                        .unwrap();
                        blit.copy_from_buffer_to_buffer(&scratches[1], 0, &kept, 0, size)
                            .unwrap();
                        blit.end_encoding().unwrap();
                    }
                    Shape::Presented => {
                        let buffer = surface_buffer(&device, &mapping, len);
                        let texture = plane_texture(&buffer, TextureUsage::SHADER_READ);
                        form = form_of(&texture);
                        let mut pass = RenderPassDescriptor::new();
                        let attachment = &mut pass.color_attachments[0];
                        attachment.texture = Some(private.clone());
                        attachment.load_action = LoadAction::Clear;
                        attachment.store_action = StoreAction::Store;
                        let mut encoder = command_buffers[0].render_command_encoder(&pass).unwrap();
                        encoder.set_render_pipeline_state(&read).unwrap();
                        encoder
                            .set_vertex_buffer(Some(&drawing.rects[0]), 0, 0)
                            .unwrap();
                        encoder
                            .set_fragment_sampler_state(Some(&nearest), 0)
                            .unwrap();
                        encoder.set_fragment_texture(Some(&texture), 0).unwrap();
                        encoder
                            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                            .unwrap();
                        encoder.end_encoding().unwrap();
                        presented = Some((texture, buffer));
                    }
                }
                let committed = Instant::now();
                for command_buffer in &command_buffers {
                    command_buffer.commit().unwrap();
                }
                let last = command_buffers.last().unwrap();
                last.wait_until_completed_for(WAIT).unwrap();
                let waited = committed.elapsed().as_secs_f64() * 1e6;
                if k >= WARM {
                    waits.push(waited);
                    let mut gpu = 0.0;
                    for command_buffer in &command_buffers {
                        command_buffer.wait_until_completed_for(WAIT).unwrap();
                        assert_eq!(command_buffer.error(), None);
                        gpu += command_buffer.gpu_end_time().unwrap()
                            - command_buffer.gpu_start_time().unwrap();
                    }
                    gpus.push(gpu * 1e6);
                    cpus.push(process_cpu_us() - cpu);
                }
                drop(presented);
            }
            let after = device.work_counts();
            let threads = threads_per_frame(&threads, &thread_cpu_us());
            let (wait, wait90) = median_p90(waits);
            let (gpu, gpu90) = median_p90(gpus);
            let (cpu, cpu90) = median_p90(cpus);
            println!(
                "GUEST-FRAME on {}: round {round}, {shape:?}: the texture {form}; commit to \
                 completion {wait:.0} us (p90 {wait90:.0}), GPU {gpu:.0} us (p90 {gpu90:.0}), \
                 process CPU {cpu:.0} us (p90 {cpu90:.0}); threads' CPU per frame (us): \
                 {threads}; per frame: {}",
                device.name(),
                per_frame(&before, &after),
            );
            drop((surface, kept, private, scratches));
        }
    }
}
