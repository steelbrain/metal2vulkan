//! A new `Shared` texture that copies on the host waits for no GPU work as it is made, and reads
//! zero (`texture/host_copy.rs`): on a device that enabled `zeroInitializeDeviceMemory` its image
//! lies in bytes of a block allocated zero-initialized that no resource used, and moves out of
//! `VK_IMAGE_LAYOUT_ZERO_INITIALIZED_EXT` with no write of the host's
//! (`WorkCounts::images_started_zeroed`); and many made in turn take pooled blocks, as without it.
//!
//! The host's write of zeros into new memory waited for the program's queued GPU work on an RX
//! 7900 XTX (RADV, Mesa 26.2.6): 0.2 to 1.1 s behind about 1.5 s of fills. Where the GPU had
//! idled into runtime suspend, the first CPU write into a texture placed in a block made then
//! waited so instead, in the kernel's clear of the block; the blocks host copies place in are
//! made ready ahead, every page read, from the device's first creation call on
//! (`memory/ready.rs`), so the texture's block is one made before the fills, and no such block
//! is made in place (`WorkCounts::host_copy_blocks_made_in_place`). Run as a process's first
//! device work after the GPU idled 5 s (its autosuspend delay there) to see that case.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::blit_holds::{device, Scratch};
use crate::{CommandBufferStatus, Region, ResourceOptions, Texture, TextureDescriptor};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
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
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let made = Instant::now();
    let texture = device.new_texture(&descriptor).unwrap();
    let took = made.elapsed();
    let status = busy.status();
    let after = device.work_counts();
    // A second one, and the CPU's first write into it (one texel), while the busy work may still
    // run: where that write waits, the wait moved from the making to the first copy.
    let written = device.new_texture(&descriptor).unwrap();
    let wrote_at = Instant::now();
    (written.replace_region(Region::new_2d(0, 0, 1, 1), 0, 0, &[1, 2, 3, 4], 4, 0)).unwrap();
    let wrote = wrote_at.elapsed();
    let status_after_write = busy.status();
    let wrote_counts = device.work_counts();
    let shared = device.shared();
    let zeroed = shared.zero_initialize_device_memory && shared.host_copies.is_some();
    println!(
        "ZEROED-START {}: new_texture (Shared, {SIDE} x {SIDE}) took {took:?} with {count} fills \
         queued ({each:?} each), the busy work {status:?} as it returned; started on the host \
         {}, zeroed {}; zeroInitializeDeviceMemory {zeroed}; a second one's first replace_region \
         took {wrote:?}, the busy work {status_after_write:?} after it; host-copy blocks readied \
         {}, waited for {}, made in place {}",
        device.name(),
        after.images_started_on_the_host - before.images_started_on_the_host,
        after.images_started_zeroed - before.images_started_zeroed,
        wrote_counts.host_copy_blocks_readied,
        wrote_counts.host_copy_ready_waits,
        wrote_counts.host_copy_blocks_made_in_place,
    );
    assert_ne!(
        status,
        CommandBufferStatus::Completed,
        "the making waited {took:?} for the GPU's busy work"
    );
    // Where its image copies on the host: elsewhere (a color texture on a device with a staging
    // queue and no host-cached memory, NVIDIA's; `texture/staging_copy.rs`) the staging queue
    // writes its zeros.
    let hosted = after.images_started_on_the_host - before.images_started_on_the_host;
    if zeroed && hosted == 1 {
        assert_eq!(
            after.images_started_zeroed - before.images_started_zeroed,
            1,
            "the image did not start in memory the driver zeroed"
        );
    }
    // Both lie in a block made ready before the fills, its pages read: neither placement made
    // one, whose first touch would wait behind them (`memory/ready.rs`).
    if hosted == 1 {
        assert!(
            wrote_counts.host_copy_blocks_readied >= 1,
            "no block of host copies was made ready ahead"
        );
        assert_eq!(
            wrote_counts.host_copy_blocks_made_in_place, 0,
            "a placement made a block of host copies itself, its pages untouched"
        );
    }
    let row = SIDE * 4;
    let mut read = vec![0xa5u8; (row * SIDE) as usize];
    let whole = Region::new_2d(0, 0, SIDE, SIDE);
    texture.get_bytes(&mut read, row, 0, whole, 0, 0).unwrap();
    assert!(read.iter().all(|&byte| byte == 0), "a texel read nonzero");
    busy.wait_until_completed().unwrap();
}

/// The side of each texture [`many_shared_textures_made_in_turn_take_pooled_blocks`] makes.
const MANY_SIDE: u64 = 128;

/// How many it makes in each round.
const MANY: usize = 1000;

/// `MANY` `Shared` textures made in turn with the GPU idle, kept, then let go and `MANY` made
/// again, twice: the first round's lie past every range placed before, the others' in ranges the
/// first used, the third's in memory the host already wrote on any device. Each making's time,
/// and the blocks they take, are printed (`ZEROED-MAKES`): on a device that starts them zeroed,
/// as many pooled blocks as on one that writes their zeros, and no block of its own each; there
/// every one of the first round that copies on the host starts zeroed. Every texture read reads
/// zero.
#[test]
fn many_shared_textures_made_in_turn_take_pooled_blocks() {
    let scratch = Scratch::new("zeroed-makes");
    let device = device(&scratch.0);
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, MANY_SIDE, MANY_SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let row = MANY_SIDE * 4;
    let whole = Region::new_2d(0, 0, MANY_SIDE, MANY_SIDE);
    let reads_zero = |texture: &Texture| {
        let mut read = vec![0xa5u8; (row * MANY_SIDE) as usize];
        texture.get_bytes(&mut read, row, 0, whole, 0, 0).unwrap();
        read.iter().all(|&byte| byte == 0)
    };
    let shared = device.shared();
    let zeroed = shared.zero_initialize_device_memory && shared.host_copies.is_some();
    let first = device.work_counts();
    for round in ["past every range", "in ranges used", "in ranges used again"] {
        let before = device.work_counts();
        let mut times = Vec::with_capacity(MANY);
        let mut made = Vec::with_capacity(MANY);
        for _ in 0..MANY {
            let at = Instant::now();
            made.push(device.new_texture(&descriptor).unwrap());
            times.push(at.elapsed());
        }
        let after = device.work_counts();
        let total: Duration = times.iter().sum();
        times.sort_unstable();
        println!(
            "ZEROED-MAKES {} {round}: {MANY} new_texture (Shared, {MANY_SIDE} x {MANY_SIDE}) took \
             {:?} each on average, median {:?}, 99th percentile {:?}, longest {:?}; started on \
             the host {}, zeroed {}; pooled blocks made {}, live {}, dedicated live {}; \
             zeroInitializeDeviceMemory {zeroed}",
            device.name(),
            total / MANY as u32,
            times[MANY / 2],
            times[MANY * 99 / 100],
            times[MANY - 1],
            after.images_started_on_the_host - before.images_started_on_the_host,
            after.images_started_zeroed - before.images_started_zeroed,
            after.pooled_memory_blocks_made - before.pooled_memory_blocks_made,
            after.memory_blocks_live,
            after.dedicated_memory_blocks_live,
        );
        assert!(
            after.dedicated_memory_blocks_live <= first.dedicated_memory_blocks_live,
            "the textures took blocks of their own"
        );
        for texture in [&made[0], &made[MANY / 2], &made[MANY - 1]] {
            assert!(reads_zero(texture), "a texel read nonzero ({round})");
        }
        // Every one that started on the host: on a device whose color textures copy on a
        // staging queue, none does (`texture/staging_copy.rs`).
        if zeroed && round == "past every range" {
            assert_eq!(
                after.images_started_zeroed - before.images_started_zeroed,
                after.images_started_on_the_host - before.images_started_on_the_host,
                "a texture past every range placed started on the host but not zeroed"
            );
        }
    }
}
