//! A new `Shared` texture that copies on the host waits for no GPU work as it is made, and reads
//! zero (`texture/host_copy.rs`): on a device that enabled `zeroInitializeDeviceMemory` its image
//! starts in memory the driver zeroed, moved out of `VK_IMAGE_LAYOUT_ZERO_INITIALIZED_EXT` with
//! no write of the host's (`WorkCounts::images_started_zeroed`).
//!
//! The host's write of zeros was the CPU's first touch of the image's new memory, which on an RX
//! 7900 XTX (RADV, Mesa 26.2.6) whose GPU had idled into runtime suspend waited for the program's
//! queued GPU work: 2.08 s behind 2 s of fills, and 1.2 ms with the GPU awake. Run as a process's
//! first device work after the GPU idled 5 s (its autosuspend delay there) to see that case.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::blit_holds::{device, Scratch};
use crate::{
    CommandBufferStatus, PixelFormat, Region, ResourceOptions, TextureDescriptor, TextureUsage,
};
use std::time::{Duration, Instant};

/// The side of the texture made while the GPU is busy.
const SIDE: u64 = 1024;

/// How long the busy work runs on the GPU, about.
const BUSY: Duration = Duration::from_millis(1500);

/// The bytes of the buffer the busy work fills.
const FILLED: u64 = 256 << 20;

#[test]
fn a_shared_texture_made_while_the_gpu_is_busy_waits_for_none_of_it_and_reads_zero() {
    let scratch = Scratch::new("zeroed-start");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let target = device
        .new_buffer(FILLED, ResourceOptions::private())
        .unwrap();
    let fill = |count: u32| {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for value in 0..count {
            blit.fill_buffer(&target, 0..FILLED, value as u8).unwrap();
        }
        blit.end_encoding().unwrap();
        command_buffer
    };
    // The fills' pace on this GPU, so the busy work lasts about `BUSY` on any.
    let probe = fill(8);
    let at = Instant::now();
    probe.commit().unwrap();
    probe.wait_until_completed().unwrap();
    let each = at.elapsed() / 8;
    let count = (BUSY.as_secs_f64() / each.as_secs_f64().max(1e-6)).clamp(8.0, 20_000.0) as u32;
    let busy = fill(count);
    let before = device.work_counts();
    busy.commit().unwrap();
    while device.work_counts().queue_submissions == before.queue_submissions {
        assert!(at.elapsed() < Duration::from_secs(30), "never submitted");
        std::thread::yield_now();
    }
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::Rgba8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let made = Instant::now();
    let texture = device.new_texture(&descriptor).unwrap();
    let took = made.elapsed();
    let status = busy.status();
    let after = device.work_counts();
    let shared = device.shared();
    let zeroed = shared.zero_initialize_device_memory && shared.host_copies.is_some();
    println!(
        "ZEROED-START {}: new_texture (Shared, {SIDE} x {SIDE}) took {took:?} with {count} fills \
         queued ({each:?} each), the busy work {status:?} as it returned; started on the host \
         {}, zeroed {}; zeroInitializeDeviceMemory {zeroed}",
        device.name(),
        after.images_started_on_the_host - before.images_started_on_the_host,
        after.images_started_zeroed - before.images_started_zeroed,
    );
    assert_ne!(
        status,
        CommandBufferStatus::Completed,
        "the making waited {took:?} for the GPU's busy work"
    );
    if zeroed {
        assert_eq!(
            after.images_started_zeroed - before.images_started_zeroed,
            1,
            "the image did not start in memory the driver zeroed"
        );
    }
    let row = SIDE * 4;
    let mut read = vec![0xa5u8; (row * SIDE) as usize];
    let whole = Region::new_2d(0, 0, SIDE, SIDE);
    texture.get_bytes(&mut read, row, 0, whole, 0, 0).unwrap();
    assert!(read.iter().all(|&byte| byte == 0), "a texel read nonzero");
    busy.wait_until_completed().unwrap();
}
