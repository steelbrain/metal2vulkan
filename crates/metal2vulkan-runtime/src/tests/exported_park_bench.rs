//! What a guest's frame of textures over a surface cache costs where each texture's image is bound
//! to a dedicated import of its block's dma-buf (`texture.rs`'s `exported_dma_buf_alias`): a
//! `Shared` buffer of a 3840x2160 NV12 picture that declares its padding unobserved, in a block
//! of the runtime's own buffers exportable as a dma-buf too (`memory.rs`'s `Holds::FdDmaBufs`), as
//! a consumer's surface caches lie; each frame makes a texture of each plane (`R8Unorm`
//! 3840x2160, `RG8Unorm` 1920x1080, rows 3840 bytes apart), copies one row of each into a small
//! buffer in one command buffer, commits it, waits for it, and drops both, as a compositor makes
//! and frees two textures from a surface each frame.
//!
//! Two arms, each a round of [`FRAMES`] frames, the rounds alternating: **unparked**, run while
//! the device parks no image (`Device::park_dma_buf_images_at_most(0)`), each texture's image
//! made, its block's dma-buf exported and imported anew, as before such images parked; and
//! **parked**, the device's own bound, each texture taking the image the same plane's texture
//! parked a frame before (`dma_buf/parked.rs`). The first [`WARM`] rounds of each are not counted.
//!
//! Printed (`EXPORTED-PARK-BENCH`), per arm: the median and 90th percentile of `new_texture`'s
//! wall and thread CPU time, the frame's wall time, and per texture the import calls and their
//! time (`WorkCounts::dma_buf_import_calls`, `dma_buf_import_nanos`), the images made
//! (`exported_dma_buf_images`) and taken (`exported_dma_buf_images_taken`).
//!
//! A release benchmark, ignored by default: compiled only under `m2v_gpu`, on Linux. Where the
//! picture lies in no such block, or its textures are shadows, it says so and returns.

use super::host_os::thread_cpu;
use super::host_residency::{device, Scratch};
use crate::memory::BoundMemory;
use crate::{
    BlitOption, Buffer, BufferContents, BufferDescriptor, CommandQueue, Device, Origin,
    ResourceOptions, Size, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::time::{Duration, Instant};

const TEST: &str = "EXPORTED-PARK-BENCH";
/// The bytes from a row's first texel to the next row's, in both planes.
const ROW: u64 = 3840;
const HEIGHT: u64 = 2160;
/// The luma plane's first byte, on a multiple of 256 of the memory (`aligned_byte`).
const LUMA_AT: u64 = 32;
const CHROMA_AT: u64 = LUMA_AT + ROW * HEIGHT;
const LENGTH: u64 = CHROMA_AT + ROW * HEIGHT / 2;
/// Each plane: its format, width, height and first byte.
const PLANES: [(PixelFormat, u64, u64, u64); 2] = [
    (PixelFormat::R8Unorm, 3840, HEIGHT, LUMA_AT),
    (PixelFormat::RG8Unorm, 1920, HEIGHT / 2, CHROMA_AT),
];
const FRAMES: usize = 60;
const ROUNDS: usize = 6;
const WARM: usize = 1;
const WAIT: Duration = Duration::from_secs(60);

/// One arm's samples: per texture, `new_texture`'s wall and thread CPU time; per frame, its wall
/// time; and the counts' moves over its counted frames.
#[derive(Default)]
struct Arm {
    make_wall: Vec<Duration>,
    make_cpu: Vec<Duration>,
    frame: Vec<Duration>,
    textures: u64,
    calls: u64,
    nanos: u64,
    made: u64,
    taken: u64,
}

/// The median and 90th percentile of `samples`, in microseconds.
fn spread(samples: &[Duration]) -> String {
    let mut sorted = samples.to_vec();
    sorted.sort();
    let at = |fraction: f64| {
        let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
        sorted[index].as_secs_f64() * 1e6
    };
    format!("median {:.1} us, p90 {:.1} us", at(0.5), at(0.9))
}

/// One frame over `picture`: a texture of each plane, one row of each copied into `out` in one
/// command buffer, waited for, both dropped. Its samples go to `arm` where `counted`.
fn frame(queue: &CommandQueue, picture: &Buffer, out: &Buffer, arm: &mut Arm, counted: bool) {
    let began = Instant::now();
    let mut textures = Vec::new();
    for (format, width, height, at) in PLANES {
        let mut descriptor = TextureDescriptor::texture_2d(format, width, height, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::shared();
        let (wall, cpu) = (Instant::now(), thread_cpu());
        let texture = picture.new_texture(&descriptor, at, ROW).unwrap();
        if counted {
            arm.make_wall.push(wall.elapsed());
            arm.make_cpu.push(thread_cpu().saturating_sub(cpu));
        }
        textures.push(texture);
    }
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for (index, texture) in textures.iter().enumerate() {
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(texture.width(), 1, 1),
            out,
            index as u64 * ROW,
            ROW,
            ROW,
            BlitOption::None,
        )
        .unwrap();
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    drop(command_buffer);
    drop(textures);
    if counted {
        arm.frame.push(began.elapsed());
    }
}

/// Run `FRAMES` frames into `arm`, the counts' moves added where `counted`.
fn round(
    device: &Device,
    (queue, picture, out): (&CommandQueue, &Buffer, &Buffer),
    arm: &mut Arm,
    counted: bool,
) {
    let before = device.work_counts();
    for _ in 0..FRAMES {
        frame(queue, picture, out, arm, counted);
    }
    let after = device.work_counts();
    if counted {
        let moved = |of: fn(&WorkCounts) -> u64| of(&after) - of(&before);
        arm.textures += (FRAMES * PLANES.len()) as u64;
        arm.calls += moved(|counts| counts.dma_buf_import_calls);
        arm.nanos += moved(|counts| counts.dma_buf_import_nanos);
        arm.made += moved(|counts| counts.exported_dma_buf_images);
        arm.taken += moved(|counts| counts.exported_dma_buf_images_taken);
    }
}

#[test]
#[ignore = "a release benchmark: run with --run-ignored all --no-capture"]
fn exported_park_bench() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let name = device.name().to_owned();
    assert!(device.settle_in_place_checks(WAIT), "{TEST}: the check");
    let picture = (device.new_buffer_with_descriptor(
        BufferDescriptor::new(LENGTH, ResourceOptions::shared(), BufferContents::Zero)
            .with_aligned_byte(LUMA_AT)
            .with_padding_unobserved(),
    ))
    .unwrap();
    if picture.bound_memory() != BoundMemory::DmaBufBlocks {
        println!(
            "{TEST} on {name}: the picture lies in {:?}, no block exportable as a dma-buf too: \
             not run",
            picture.bound_memory()
        );
        return;
    }
    let queue = device.new_command_queue().unwrap();
    let out = (device.new_buffer(ROW * PLANES.len() as u64, ResourceOptions::shared())).unwrap();
    let before = device.work_counts();
    frame(&queue, &picture, &out, &mut Arm::default(), false);
    let probe = device.work_counts();
    if probe.exported_dma_buf_images == before.exported_dma_buf_images {
        println!("{TEST} on {name}: no image over the block's dma-buf was made: not run");
        return;
    }
    let (mut unparked, mut parked) = (Arm::default(), Arm::default());
    let objects = (&queue, &picture, &out);
    for index in 0..ROUNDS {
        let counted = index >= WARM;
        // None parks, and none parked by the last round is left to take.
        device.park_dma_buf_images_at_most(0);
        device.shared().residency.sweep(true);
        round(&device, objects, &mut unparked, counted);
        device.park_dma_buf_images_at_most(crate::host_residency::PARKED_MOST);
        round(&device, objects, &mut parked, counted);
    }
    for (label, arm) in [("unparked", &unparked), ("parked", &parked)] {
        let per = |total: u64| total as f64 / arm.textures as f64;
        println!(
            "{TEST} on {name}: {label}: new_texture wall {}, thread CPU {}; frame of {} textures \
             {}; per texture: import calls {:.2}, import time {:.1} us, images made {:.2}, taken \
             {:.2} ({} textures)",
            spread(&arm.make_wall),
            spread(&arm.make_cpu),
            PLANES.len(),
            spread(&arm.frame),
            per(arm.calls),
            per(arm.nanos) / 1e3,
            per(arm.made),
            per(arm.taken),
            arm.textures
        );
    }
    assert!(
        parked.taken > 0,
        "{TEST}: the parked arm's textures took parked images"
    );
}
