//! What a device keeps once the program let go of its resources goes once its time passed, while
//! the program makes and releases nothing more: the recyclers' parked images and buffers
//! (`recycle.rs`, a second) and the memory's kept empty blocks (`memory.rs`'s `KEPT_EMPTY_FOR`),
//! but the bounded room its pools keep.
//!
//! The program makes `Private` buffers and textures over several pooled blocks of each kind, uses
//! some of them in three command buffers (one that completes, one whose commit is refused for an
//! abandoned encoder, one whose submission check fails it), lets go of all of them and of the
//! command buffers, then makes and releases nothing. Within [`IDLE`], no image or buffer is still
//! parked, each pool keeps at most one empty pooled block, and the pooled blocks live and the
//! bytes of the device's heaps are back to where they were before the program's resources, but
//! those blocks: freed equal made, within the bound. The device's idle sweep's thread
//! (`idle_sweep.rs`) did it, as no call of the program's came; without it the device kept every
//! block, 32 images and 3 buffers parked (lavapipe).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{Scratch, WAIT};
use crate::{
    Buffer, CommandBufferStatus, CommandQueue, Device, DeviceKind, DeviceOptions, ResourceOptions,
    ShaderCacheOptions, Submission, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::num::NonZeroU64;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Each buffer's bytes: three to a pooled block of 64 MiB, under half a block (no block of its
/// own), and three of them past the bytes the recycler parks (`recycle.rs`'s `BUFFERS`).
const BUFFER: u64 = 20 << 20;
/// Over four pooled blocks.
const BUFFERS: usize = 12;
/// Each texture's side: `RGBA8Unorm`, 4 MiB.
const SIDE: u64 = 1024;
/// Over three pooled blocks of images, past the count the recycler parks (`recycle.rs`'s
/// `IMAGES`).
const TEXTURES: usize = 40;
/// The longest the device may take, idle, to let go: the kept blocks' quiet period
/// (`memory.rs`'s `KEPT_EMPTY_FOR`, 10 s) after the parked images' second, with room for a
/// loaded host.
const IDLE: Duration = Duration::from_secs(40);

/// The device the cfg selects, its shader cache in `cache`.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// A `Private` `RGBA8Unorm` texture of [`SIDE`] a side.
fn texture(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// What the device holds now: its images and buffers parked, its pooled blocks live (made less
/// freed), its empty blocks, the pools that hold a pooled block, and the bytes of the blocks
/// live in its heaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Held {
    images_parked: u64,
    buffers_parked: u64,
    pooled: u64,
    empty: u64,
    pools: u64,
    heap_bytes: u64,
}

fn held(device: &Device) -> Held {
    let counts = device.work_counts();
    let churn = &counts.object_churn;
    let memory = crate::device::lock(&device.shared().memory);
    Held {
        images_parked: churn.images_parked - churn.images_recycled - churn.images_evicted,
        buffers_parked: churn.buffers_parked - churn.buffers_recycled - churn.buffers_evicted,
        pooled: counts.pooled_memory_blocks_made - counts.pooled_memory_blocks_freed,
        empty: counts.empty_memory_blocks_live,
        pools: memory.pooled_by_pool().len() as u64,
        heap_bytes: memory.heaps_live().iter().map(|&(_, live)| live).sum(),
    }
}

/// Whether `now` holds what `start` held, but the bound: nothing parked, at most one empty pooled
/// block a pool, and no more pooled blocks and heap bytes than one block (`block` bytes, the
/// largest pooled block) more for each pool that holds a pooled block.
fn let_go(start: Held, now: Held, block: u64) -> bool {
    now.images_parked == 0
        && now.buffers_parked == 0
        && now.empty <= now.pools
        && now.pooled <= start.pooled + now.pools
        && now.heap_bytes <= start.heap_bytes + now.pools * block
}

/// A blit command buffer of `queue` that fills `buffer` and copies `from` into `to`, ended where
/// `end`, else its encoder dropped unended (abandoned).
fn blit(
    queue: &CommandQueue,
    buffer: &Buffer,
    (from, to): (&Texture, &Texture),
    end: bool,
) -> crate::CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encoder
        .fill_buffer(buffer, 0..buffer.length(), 0x5a)
        .unwrap();
    encoder.copy_texture(from, to).unwrap();
    match end {
        true => encoder.end_encoding().unwrap(),
        false => drop(encoder),
    }
    command_buffer
}

/// Buffers and textures over several blocks, used by a command buffer that completes, one whose
/// commit is refused and one that fails, all let go of, then nothing made or released: the
/// device lets go of everything they held within [`IDLE`], but the bounded room of its pools.
#[test]
fn an_idle_device_lets_go_of_what_its_released_resources_held() {
    let scratch = Scratch::new("idle-sweep");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let block = crate::memory::BLOCK_BYTES;
    let start = held(&device);
    let buffers: Vec<Buffer> = (0..BUFFERS)
        .map(|_| {
            device
                .new_buffer(BUFFER, ResourceOptions::private())
                .unwrap()
        })
        .collect();
    let textures: Vec<Texture> = (0..TEXTURES).map(|_| texture(&device)).collect();
    let made = held(&device);
    assert!(
        made.pooled >= start.pooled + 4,
        "the resources took several pooled blocks: {start:?} then {made:?}"
    );
    // One that completes.
    let completes = blit(&queue, &buffers[0], (&textures[0], &textures[1]), true);
    completes.commit().unwrap();
    // One whose commit is refused: its encoder was abandoned.
    let refused = blit(&queue, &buffers[1], (&textures[2], &textures[3]), false);
    assert!(
        refused.commit().is_err(),
        "a command buffer whose encoder was abandoned is refused at its commit"
    );
    // One that fails: its submission check answers a code.
    let fails = blit(&queue, &buffers[2], (&textures[4], &textures[5]), true);
    fails
        .set_submission_check(Arc::new(|_: &Submission<'_>| NonZeroU64::new(8)))
        .unwrap();
    fails.commit().unwrap();
    for command_buffer in [&completes, &fails] {
        assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    }
    assert_eq!(completes.status(), CommandBufferStatus::Completed);
    assert_eq!(fails.status(), CommandBufferStatus::Error);
    drop((completes, refused, fails));
    drop(buffers);
    drop(textures);
    // Idle: nothing made or released from here on.
    let idle = Instant::now();
    let swept = device.shared().idle_sweep.sweeps();
    let mut now = held(&device);
    while !let_go(start, now, block) && idle.elapsed() < IDLE {
        std::thread::sleep(Duration::from_millis(200));
        now = held(&device);
    }
    assert!(
        let_go(start, now, block),
        "{:.1} s idle, the device still holds what the released resources held: {now:?}, \
         from {start:?} before them ({block} bytes a block)",
        idle.elapsed().as_secs_f64(),
    );
    // The idle sweep's thread let go of them: no call of the program's came.
    assert!(device.shared().idle_sweep.sweeps() > swept);
    eprintln!(
        "let go of in {:.1} s idle: {now:?}, from {start:?}, {made:?} with the resources",
        idle.elapsed().as_secs_f64()
    );
    drop(queue);
}
