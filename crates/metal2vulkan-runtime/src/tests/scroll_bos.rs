//! A scroll's memory and commands, for a kernel trace taken beside the run to count the buffer
//! objects the driver makes (amdgpu's `amdgpu_bo_create`), by thread and size, and for the
//! runtime's counts of the same span: what a guest's scroll makes on a device whose pools grow and
//! shrink as tiles come and go.
//!
//! For [`LENGTH`], a command buffer copying [`COPY_BYTES`] between two `Private` buffers is
//! committed about every millisecond and waited for; every [`BUFFER_EVERY`] a `Private` buffer of
//! [`BUFFER_BYTES`] is made, the oldest past the [`BUFFER_WINDOW`] newest released; every
//! [`BURST_EVERY`] a burst of [`BURST`] `Private` `RGBA8Unorm` textures of [`SIDE`] square is made,
//! each cleared by a render pass of its own on a second queue, the oldest past the
//! [`TEXTURE_WINDOW`] newest released. Printed (`SCROLL-BOS`): the span's `CLOCK_MONOTONIC` start
//! and end, the command buffers, the pooled blocks made and freed, those made ahead and freed
//! unused, the `vkAllocateMemory` and `vkFreeMemory` calls, and the five places that allocated
//! most.
//!
//! Only the public API and the runtime's counts, so that the same file runs at earlier trees.
//! Ignored: run it in release under `--cfg m2v_gpu`, one device selected:
//! `cargo nextest run --release -p metal2vulkan-runtime --lib --run-ignored all scroll_bos
//! --no-capture`.

use super::host_residency::{device, Scratch};
use crate::{
    Buffer, ClearColor, CommandBufferStatus, LoadAction, RenderPassDescriptor, ResourceOptions,
    StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
/// How long the scroll runs.
const LENGTH: Duration = Duration::from_secs(15);
/// The bytes each command buffer copies, and the gap between them.
const COPY_BYTES: u64 = 64 * 1024;
const COPY_EVERY: Duration = Duration::from_millis(1);
/// A guest's `Private` buffer, as a scroll's memory log placed them, eight to a pooled block.
const BUFFER_BYTES: u64 = 7_471_104;
const BUFFER_EVERY: Duration = Duration::from_millis(30);
const BUFFER_WINDOW: usize = 64;
/// A tile: `RGBA8Unorm` of this side (4 MiB), [`BURST`] at a time.
const SIDE: u64 = 1024;
const BURST: usize = 16;
const BURST_EVERY: Duration = Duration::from_millis(500);
/// The live tiles: three bursts'.
const TEXTURE_WINDOW: usize = 3 * BURST;

/// `CLOCK_MONOTONIC` now, in seconds: what a kernel trace's `mono` clock reads.
fn monotonic() -> f64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: a valid clock and a timespec to write.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
    now.tv_sec as f64 + now.tv_nsec as f64 * 1e-9
}

#[test]
#[ignore = "a measurement beside a kernel trace: run it alone, in release"]
fn scroll_bos() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let (queue, tiles_queue) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let private = |length| {
        device
            .new_buffer(length, ResourceOptions::private())
            .unwrap()
    };
    let (from, to) = (private(COPY_BYTES), private(COPY_BYTES));
    let mut buffers: Vec<Buffer> = Vec::new();
    let mut textures: Vec<Texture> = Vec::new();
    let start = device.work_counts();
    let began = (Instant::now(), monotonic());
    let (mut next_buffer, mut next_burst, mut submitted) = (began.0, began.0, 0u64);
    while began.0.elapsed() < LENGTH {
        let now = Instant::now();
        if now >= next_burst {
            next_burst += BURST_EVERY;
            for _ in 0..BURST {
                let mut descriptor =
                    TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
                descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
                descriptor.resource_options = ResourceOptions::private();
                let texture = device.new_texture(&descriptor).unwrap();
                let mut pass = RenderPassDescriptor::new();
                let color = &mut pass.color_attachments[0];
                color.texture = Some(texture.clone());
                color.load_action = LoadAction::Clear;
                color.store_action = StoreAction::Store;
                color.clear_color = ClearColor::new(0.0, 0.5, 1.0, 1.0);
                let command_buffer = tiles_queue.command_buffer().unwrap();
                (command_buffer.render_command_encoder(&pass).unwrap())
                    .end_encoding()
                    .unwrap();
                command_buffer.commit().unwrap();
                textures.push(texture);
                if textures.len() > TEXTURE_WINDOW {
                    drop(textures.remove(0));
                }
            }
        }
        if now >= next_buffer {
            next_buffer += BUFFER_EVERY;
            buffers.push(private(BUFFER_BYTES));
            if buffers.len() > BUFFER_WINDOW {
                drop(buffers.remove(0));
            }
        }
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        (encoder.copy_from_buffer_to_buffer(&from, 0, &to, 0, COPY_BYTES)).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        submitted += 1;
        std::thread::sleep(COPY_EVERY);
    }
    let ended = monotonic();
    let counts = device.work_counts();
    let churn = (&counts.object_churn, &start.object_churn);
    let mut sites: Vec<String> = (churn.0.making_sites.iter())
        .filter(|site| site.calls > 0 && site.command == "vkAllocateMemory")
        .map(|site| format!("{}:{} {}", site.file, site.line, site.calls))
        .collect();
    sites.sort_by_key(|site| {
        std::cmp::Reverse(site.rsplit(' ').next().unwrap().parse::<u64>().unwrap())
    });
    sites.truncate(5);
    println!(
        "SCROLL-BOS {}: from {:.4} to {ended:.4}; {submitted} copies; pooled blocks made {}, \
         freed {}, made ahead and freed unused {}; image blocks made ahead {}, in place {}; \
         buffer blocks made ahead {}, in place {}; vkAllocateMemory {}, vkFreeMemory {}; \
         allocated most at [{}]",
        device.name(),
        began.1,
        counts.pooled_memory_blocks_made - start.pooled_memory_blocks_made,
        counts.pooled_memory_blocks_freed - start.pooled_memory_blocks_freed,
        counts.blocks_ahead_unused - start.blocks_ahead_unused,
        counts.image_blocks_made_ahead - start.image_blocks_made_ahead,
        counts.image_blocks_made_in_place - start.image_blocks_made_in_place,
        counts.buffer_blocks_made_ahead - start.buffer_blocks_made_ahead,
        counts.buffer_blocks_made_in_place - start.buffer_blocks_made_in_place,
        churn.0.vk_memory_allocations - churn.1.vk_memory_allocations,
        churn.0.vk_memory_freed - churn.1.vk_memory_freed,
        sites.join(", "),
    );
    drop((textures, buffers, from, to));
}
