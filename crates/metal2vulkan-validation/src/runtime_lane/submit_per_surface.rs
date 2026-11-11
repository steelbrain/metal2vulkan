//! What one kernel submission costs against the surfaces a device holds over dma-buf pages, on
//! the device the cfg selects, without the validation layer: report only, in release builds.
//!
//! A consumer's guest surface is a no-copy buffer over its pages, which RADV imports as dma-bufs
//! (`HostMemory::with_dma_buf`), and a texture in place over the buffer
//! (`HostPageTexturePolicy::InPlace`), which on RADV is a `DRM_FORMAT_MOD_LINEAR` image in a
//! dedicated import of a dma-buf of its own pages (`dma_buf.rs` of the runtime). RADV lists every
//! imported `VkDeviceMemory` in every kernel submission (`runtime_lane::submit_per_import`), so
//! each surface may list two: the buffer's import and the image's.
//!
//! For each `n` of [`COUNTS`], `n` surfaces, each [`SIDE`] x [`SIDE`] `BGRA8Unorm` texels over
//! pages of their own (one run of a sealed `memfd`, as QEMU's guest RAM), each texture read once
//! by a blit into a `Private` buffer; then [`COMMITS`] command buffers of one 4-KiB fill, each
//! committed alone and waited for, after [`WARM`] untimed: at once, and again once the surfaces sat
//! idle past the device's first idle window ([`IDLE`]), as a desktop's windows do between frames
//! that touch them. Printed as `SUBMIT-PER-SURFACE` lines, per `n` and residency: the textures in
//! place over a dma-buf of their own (`WorkCounts::dma_buf_textures_in_place`), the device's
//! dma-buf memories live (`dma_buf_memories_live`: the buffers' own imports;
//! `dma_buf_image_memories_live`: the images'), the buffer imports released while idle
//! (`host_imports_released`) and those a texture keeps imported for good
//! (`host_imports_read_apart_texture`), a command buffer's time from its kernelStartTime to its
//! kernelEndTime (50th percentile), and, where the kernel-submission counter
//! (`scripts/i915-execbuffers/counter.c`) is preloaded, the kernel submissions a commit, the
//! dma-bufs each lists and the microseconds each spent in the ioctl.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`; on a
//! host without `/dev/udmabuf` access it says so and is skipped.

use super::commit_costs::percentiles;
use super::dma_buf_imports::{no_copy, udmabuf_openable, Calls, Memfd};
use super::submit_costs::Execbuffers;
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBufferStatus, CommandQueue, Device, Origin, ResourceOptions, Size,
    Texture, TextureDescriptor,
};
use std::time::Duration;

/// The surfaces of each phase.
const COUNTS: [usize; 4] = [0, 32, 64, 128];
/// Each surface's side in texels, and its rows' bytes: 512, a multiple of RADV's 256.
const SIDE: u64 = 128;
const ROW: u64 = SIDE * 4;
/// Command buffers committed untimed first, and measured.
const WARM: usize = 30;
const COMMITS: usize = 300;
/// How long the surfaces sit idle before the second measurement: past the device's first idle
/// window of 2 s and a tick of its residency thread (`host_residency.rs` of the runtime).
const IDLE: Duration = Duration::from_secs(4);

/// A command buffer of `queue` filling `target`'s 4 KiB with `value`, committed and waited for:
/// the seconds from its kernelStartTime to its kernelEndTime.
fn filled(queue: &CommandQueue, target: &Buffer, value: u8) -> f64 {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, 0..4096, value).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    command_buffer.kernel_end_time() - command_buffer.kernel_start_time()
}

/// Each of `textures` read once by a blit into `sink`, in one command buffer, waited for.
fn read_once(queue: &CommandQueue, textures: &[Texture], sink: &Buffer) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for texture in textures {
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::default(),
            Size::new(SIDE, SIDE, 1),
            sink,
            0,
            ROW,
            ROW * SIDE,
            BlitOption::None,
        )
        .unwrap();
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// Commit and time command buffers beside `n` surfaces in `residency`, and print what they took.
fn measure(device: &Device, queue: &CommandQueue, target: &Buffer, n: usize, residency: &str) {
    for value in 0..WARM {
        filled(queue, target, value as u8);
    }
    let execbuffers = Execbuffers::read();
    let times: Vec<f64> = (0..COMMITS)
        .map(|value| filled(queue, target, value as u8) * 1e6)
        .collect();
    let [_, p50, p90] = percentiles(times);
    let kernel = Execbuffers::read()
        .zip(execbuffers)
        .map(|(after, before)| {
            let submissions = after.execbuffers - before.execbuffers;
            let each = |after: u64, before: u64| (after - before) as f64 / submissions as f64;
            format!(
                "; {:.2} kernel submissions a commit, {:.0} objects each ({:.0} dma-bufs), {:.1} \
                 us each in the ioctl",
                submissions as f64 / COMMITS as f64,
                each(after.objects, before.objects),
                each(after.prime, before.prime),
                each(after.nanos, before.nanos) / 1e3,
            )
        })
        .unwrap_or_default();
    let counts = device.work_counts();
    eprintln!(
        "SUBMIT-PER-SURFACE {}: {n} surfaces {residency}: {} textures in place over a dma-buf; \
         dma-buf memories live {} of buffers, {} of images; {} buffer imports released while \
         idle, {} kept for a texture; submit p50 {p50:.1} us, p90 {p90:.1} us{kernel}",
        device.name(),
        counts.dma_buf_textures_in_place,
        counts.dma_buf_memories_live,
        counts.dma_buf_image_memories_live,
        counts.host_imports_released,
        counts.host_imports_read_apart_texture,
    );
}

/// The phase of `n` surfaces: made, read once, measured at once and after their idle window.
fn phase(device: &Device, queue: &CommandQueue, target: &Buffer, n: usize) {
    let bytes = (ROW * SIDE) as usize;
    let ram = Memfd::new(n.max(1) * bytes);
    let calls = Calls::default();
    let source = ram.source(&ram.whole, &calls);
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    let surfaces: Vec<(Buffer, Texture)> = (0..n)
        .map(|i| {
            let buffer = no_copy(device, &ram.whole, i * bytes, bytes, source.clone()).unwrap();
            let texture = buffer.new_texture(&descriptor, 0, ROW).unwrap();
            (buffer, texture)
        })
        .collect();
    let textures: Vec<Texture> = surfaces.iter().map(|(_, t)| t.clone()).collect();
    let sink = device
        .new_buffer(bytes as u64, ResourceOptions::private())
        .unwrap();
    read_once(queue, &textures, &sink);
    measure(device, queue, target, n, "at once");
    std::thread::sleep(IDLE);
    measure(device, queue, target, n, "after their idle window");
    drop(textures);
    drop(surfaces);
}

/// The phases, printed (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn submit_per_surface() {
    if !udmabuf_openable("submit per surface") {
        return;
    }
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record, and takes textures over host pages in
        // place.
        options.record_executions = false;
        options.policy.host_page_textures = HostPageTexturePolicy::InPlace;
    })
    .unwrap();
    assert!(device.settle_submit_checks(WAIT));
    assert!(device.settle_in_place_checks(WAIT));
    eprintln!(
        "SUBMIT-PER-SURFACE {}: the in-place checks: {:?}",
        device.name(),
        device.in_place_check_report()
    );
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    for n in COUNTS {
        phase(&device, &queue, &target, n);
    }
}
