//! The GPU's time to fill a `BGRA8Unorm` texture of [`WIDTH`] by [`HEIGHT`] (15.6 MiB, rows
//! [`PITCH`] bytes apart, tight) from a surface laid out in a guest's RAM, and to copy it back out,
//! by `BlitCommandEncoder::copy_texture_from_host_ram` and `copy_texture_to_host_ram`
//! (`host_ram/texture.rs`), beside `copy_from_host_ram` and `copy_to_host_ram` of the same
//! surface's bytes and a `Private` buffer. The surface's pages: [`BLOCKS`] 2 MiB blocks of pages
//! side by side (as a guest's RAM on huge pages holds a large allocation), the blocks apart over
//! the RAM's four chunks, given as a run a page.
//!
//! Each figure is the median of [`ITERATIONS`] command buffers, each copying the surface alone,
//! waited for before the next, after one of each that imports the chunks: the GPU's time
//! (`gpu_end_time - gpu_start_time`), its rate in ms/MiB, and the copy's regions
//! (`WorkCounts::host_ram_copy_regions`) and dispatches. The texture's texels are checked against
//! the surface's bytes after its fills, and the surface's bytes after the copies out. Prints
//! `HOST-RAM-TEXTURE-BENCH` lines. Ignored: run it in release under `--cfg m2v_gpu`,
//! `cargo nextest run --release -p metal2vulkan-runtime --lib --run-ignored all
//! host_ram_texture_bench --no-capture`.

use super::host_ram_copy_device::{bytes, contents, device, ram_of, write_pattern, PAGE};
use crate::{
    BlitOption, Buffer, CommandBufferStatus, CommandQueue, Device, Origin, ResourceOptions, Size,
    Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::ops::Range;
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);
const WIDTH: u64 = 2560;
const HEIGHT: u64 = 1600;
const TEXEL: u64 = 4;
const PITCH: u64 = WIDTH * TEXEL;
const SURFACE: u64 = PITCH * HEIGHT;
const MIB: f64 = (1 << 20) as f64;
const BLOCK: u64 = 2 << 20;
/// The RAM: four chunks of 64 MiB.
const RAM_BYTES: u64 = 4 * (64 << 20);
/// The 2 MiB blocks of the RAM the surface's bytes lie in, in order: two in each chunk, apart.
const BLOCKS: [u64; 8] = [3, 20, 40, 60, 70, 95, 100, 120];
const ITERATIONS: usize = 10;

/// The surface's bytes in order as runs of the RAM from `start`, a page each.
fn surface_runs(start: u64) -> Vec<Range<u64>> {
    let pages = SURFACE.div_ceil(PAGE);
    (0..pages)
        .map(|page| {
            let at = page * PAGE;
            let host = start + BLOCKS[(at / BLOCK) as usize] * BLOCK + at % BLOCK;
            host..host + PAGE
        })
        .collect()
}

/// The surface's bytes, as the CPU reads them through `runs`.
fn surface_bytes(runs: &[Range<u64>]) -> Vec<u8> {
    let mut out: Vec<u8> = runs.iter().flat_map(|run| bytes(run.clone())).collect();
    out.truncate(SURFACE as usize);
    out
}

/// One command buffer of one blit encoder `encode` fills, committed and waited for: the GPU's
/// time.
fn timed(queue: &CommandQueue, encode: impl FnOnce(&mut crate::BlitCommandEncoder)) -> Duration {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
    Duration::from_secs_f64(gpu.max(0.0))
}

/// Run `copy` once to import the chunks, then [`ITERATIONS`] times timed, and print the median
/// GPU time, its rate, and the regions and dispatches of one copy.
fn measured(
    device: &Device,
    queue: &CommandQueue,
    what: &str,
    copy: impl Fn(&mut crate::BlitCommandEncoder),
) {
    timed(queue, &copy);
    let before: WorkCounts = device.work_counts();
    let mut gpu: Vec<Duration> = (0..ITERATIONS).map(|_| timed(queue, &copy)).collect();
    let after = device.work_counts();
    gpu.sort();
    let ms = gpu[gpu.len() / 2].as_secs_f64() * 1e3;
    let per = |count: u64| count / ITERATIONS as u64;
    eprintln!(
        "HOST-RAM-TEXTURE-BENCH {} {what}: gpu {ms:.3} ms, {:.4} ms/MiB, regions {}, dispatches {}, staged {}",
        device.name(),
        ms / (SURFACE as f64 / MIB),
        per(after.host_ram_copy_regions - before.host_ram_copy_regions),
        per(after.host_ram_copy_dispatches - before.host_ram_copy_dispatches),
        per(after.host_ram_copies_staged + after.host_ram_scatters_staged
            - before.host_ram_copies_staged
            - before.host_ram_scatters_staged),
    );
}

/// The texture's texels as a blit reads them, rows one after another.
fn read_back(device: &Device, queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let out = device
        .new_buffer(SURFACE, ResourceOptions::shared())
        .unwrap();
    timed(queue, |encoder| {
        encoder
            .copy_from_texture_to_buffer(
                texture,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(WIDTH, HEIGHT, 1),
                &out,
                0,
                PITCH,
                SURFACE,
                BlitOption::None,
            )
            .unwrap();
    });
    contents(&out)
}

#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn host_ram_texture_bench() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = ram_of(RAM_BYTES, None);
    let _used = device.use_host_ram(&ram).unwrap();
    let runs = surface_runs(start);
    for run in &runs {
        write_pattern(run.clone(), 41);
    }
    let surface = surface_bytes(&runs);
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let texture = device.new_texture(&descriptor).unwrap();
    let cache: Buffer = device
        .new_buffer(SURFACE, ResourceOptions::private())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let size = Size::new(WIDTH, HEIGHT, 1);
    let origin = Origin::new(0, 0, 0);
    eprintln!(
        "HOST-RAM-TEXTURE-BENCH {}: {WIDTH}x{HEIGHT} BGRA8Unorm, pitch {PITCH}, {:.1} MiB, {} page runs in {} blocks of 2 MiB",
        device.name(),
        SURFACE as f64 / MIB,
        runs.len(),
        BLOCKS.len(),
    );
    measured(&device, &queue, "texture fill", |encoder| {
        (encoder.copy_texture_from_host_ram(&ram, &runs, 0, PITCH, size, &texture, 0, 0, origin))
            .unwrap();
    });
    assert!(
        read_back(&device, &queue, &texture) == surface,
        "the fills' texels"
    );
    measured(&device, &queue, "buffer gather", |encoder| {
        encoder.copy_from_host_ram(&ram, &runs, &cache, 0).unwrap();
    });
    for run in &runs {
        write_pattern(run.clone(), 42);
    }
    measured(&device, &queue, "texture copy out", |encoder| {
        (encoder.copy_texture_to_host_ram(&texture, 0, 0, origin, size, &ram, &runs, 0, PITCH))
            .unwrap();
    });
    assert!(surface_bytes(&runs) == surface, "the copies out's bytes");
    measured(&device, &queue, "buffer scatter", |encoder| {
        encoder.copy_to_host_ram(&cache, 0, &ram, &runs).unwrap();
    });
}
