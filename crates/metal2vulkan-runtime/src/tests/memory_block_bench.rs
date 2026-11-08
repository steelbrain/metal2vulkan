//! Pooled blocks made and freed while a program makes resources and submits, as a guest's scroll
//! does on a consumer's Radeon 780M: two threads make `Private` buffers of 7,471,104 bytes and
//! 1024 x 1024 `RGBA8Unorm` textures, a few a frame, keep them while the working set grows, then
//! let them all go, twice; meanwhile a compositor commits a frame of GPU work every 16.7 ms on a
//! queue of its own, and a third thread commits a one-word fill on a queue of its own every
//! millisecond and waits for it, so that a submission held by a block's making or freeing shows
//! as a ping that completed late. Printed (`BLOCK-MAKES`): the makings' spread (buffers and
//! textures apart), the pings' and the frames' commit to completion, and the pooled blocks made
//! and freed over the run, and the most live at once. Ignored: run it in release under
//! `--cfg m2v_gpu`, `cargo test --release -p metal2vulkan-runtime --lib memory_block_bench --
//! --ignored --nocapture`.

use crate::{
    Buffer, CommandBufferStatus, Device, DeviceKind, DeviceOptions, ResourceOptions, Texture,
    TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
/// A guest's `Private` buffer, as the scroll's memory log placed them.
const BUFFER_BYTES: u64 = 7_471_104;
/// A texture's side.
const SIDE: u64 = 1024;
/// The objects each maker thread makes in a round, one every [`MAKE_EVERY`].
const OBJECTS: usize = 240;
const MAKE_EVERY: Duration = Duration::from_millis(8);
/// The rounds: grow, hold, let go, rest.
const ROUNDS: usize = 2;
/// The compositor's frame period, and the bytes its frame fills.
const FRAME: Duration = Duration::from_micros(16_667);
const FRAME_BYTES: u64 = 64 << 20;
/// How often the pinger commits.
const PING: Duration = Duration::from_millis(1);

/// Sets its flag as it goes, so every thread of the bench stops however the bench ends.
struct Stops<'a>(&'a AtomicBool);

impl Drop for Stops<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// A milliseconds summary of `values`: median, 99th percentile and slowest, and how many took
/// 5 ms or more.
fn spread(values: &mut [Duration]) -> String {
    if values.is_empty() {
        return "none".to_owned();
    }
    values.sort();
    let ms = |at: usize| values[at.min(values.len() - 1)].as_secs_f64() * 1e3;
    let slow = values.iter().filter(|value| value.as_millis() >= 5).count();
    format!(
        "{} of them: median {:.3} ms, p99 {:.3} ms, slowest {:.3} ms, {slow} of 5 ms or more",
        values.len(),
        ms(values.len() / 2),
        ms(values.len() * 99 / 100),
        ms(values.len() - 1)
    )
}

#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn memory_block_bench() {
    let options = DeviceOptions {
        kinds: vec![
            DeviceKind::Discrete,
            DeviceKind::Integrated,
            DeviceKind::Virtual,
        ],
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    let done = AtomicBool::new(false);
    let live_most = AtomicU64::new(0);
    let (buffer_makes, texture_makes) = (Mutex::new(Vec::new()), Mutex::new(Vec::new()));
    // The pooled blocks made over each round's growth, as the first maker saw them.
    let grown = Mutex::new(Vec::new());
    let before = device.work_counts();
    let (pings, frames) = std::thread::scope(|scope| {
        let stops = Stops(&done);
        let compositor = scope.spawn(|| {
            let queue = device.new_command_queue().unwrap();
            let target = device
                .new_buffer(FRAME_BYTES, ResourceOptions::private())
                .unwrap();
            let mut frames = Vec::new();
            let mut next = Instant::now();
            while !done.load(Ordering::Acquire) {
                let command_buffer = queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&target, 0..FRAME_BYTES, 1).unwrap();
                blit.end_encoding().unwrap();
                let committed = Instant::now();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                frames.push(committed.elapsed());
                next += FRAME;
                std::thread::sleep(next.saturating_duration_since(Instant::now()));
            }
            frames
        });
        let pinger = scope.spawn(|| {
            let queue = device.new_command_queue().unwrap();
            let word = device.new_buffer(4096, ResourceOptions::private()).unwrap();
            let mut pings = Vec::new();
            while !done.load(Ordering::Acquire) {
                let command_buffer = queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&word, 0..4, 2).unwrap();
                blit.end_encoding().unwrap();
                let committed = Instant::now();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                pings.push(committed.elapsed());
                if pings.len() % 16 == 0 {
                    let counts = device.work_counts();
                    live_most.fetch_max(counts.memory_blocks_live, Ordering::Relaxed);
                }
                std::thread::sleep(PING);
            }
            pings
        });
        let makers: Vec<_> = [false, true]
            .into_iter()
            .map(|textures| {
                let (device, buffer_makes, texture_makes, grown) =
                    (&device, &buffer_makes, &texture_makes, &grown);
                scope.spawn(move || {
                    let mut descriptor =
                        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
                    descriptor.usage = TextureUsage::SHADER_READ;
                    descriptor.resource_options = ResourceOptions::private();
                    for _ in 0..ROUNDS {
                        let (mut buffers, mut textures_held): (Vec<Buffer>, Vec<Texture>) =
                            (Vec::new(), Vec::new());
                        let made_before = device.work_counts().pooled_memory_blocks_made;
                        for at in 0..OBJECTS {
                            // Buffers from one thread, buffers and textures in turn from the other.
                            let texture = textures && at % 2 == 1;
                            let started = Instant::now();
                            if texture {
                                textures_held.push(device.new_texture(&descriptor).unwrap());
                                texture_makes.lock().unwrap().push(started.elapsed());
                            } else {
                                let options = ResourceOptions::private();
                                buffers.push(device.new_buffer(BUFFER_BYTES, options).unwrap());
                                buffer_makes.lock().unwrap().push(started.elapsed());
                            }
                            std::thread::sleep(MAKE_EVERY);
                        }
                        if !textures {
                            let made = device.work_counts().pooled_memory_blocks_made;
                            grown.lock().unwrap().push(made - made_before);
                        }
                        std::thread::sleep(Duration::from_secs(1));
                        drop((buffers, textures_held));
                        std::thread::sleep(Duration::from_secs(2));
                    }
                })
            })
            .collect();
        for maker in makers {
            maker.join().unwrap();
        }
        drop(stops);
        (pinger.join().unwrap(), compositor.join().unwrap())
    });
    let after = device.work_counts();
    let (mut pings, mut frames) = (pings, frames);
    eprintln!("BLOCK-MAKES device: {}", device.name());
    eprintln!(
        "BLOCK-MAKES buffer makings: {}",
        spread(&mut buffer_makes.lock().unwrap())
    );
    eprintln!(
        "BLOCK-MAKES texture makings: {}",
        spread(&mut texture_makes.lock().unwrap())
    );
    eprintln!(
        "BLOCK-MAKES pings, commit to completion: {}",
        spread(&mut pings)
    );
    eprintln!(
        "BLOCK-MAKES frames, commit to completion: {}",
        spread(&mut frames)
    );
    let frees = |counts: &crate::WorkCounts| {
        let frees = &counts.memory_frees;
        [
            frees.pooled_emptied,
            frees.kept_stale,
            frees.kept_for_room,
            frees.dedicated,
        ]
    };
    let (freed_after, freed_before) = (frees(&after), frees(&before));
    let freed: Vec<u64> = (freed_after.iter().zip(freed_before))
        .map(|(a, b)| a - b)
        .collect();
    eprintln!(
        "BLOCK-MAKES counts: pooled blocks made {} (over each round's growth {:?}), freed {} \
         (emptied past what is kept {}, stale {}, for room {}, dedicated {}), most live {}, at \
         the end live {} and empty {}",
        after.pooled_memory_blocks_made - before.pooled_memory_blocks_made,
        grown.lock().unwrap(),
        after.pooled_memory_blocks_freed - before.pooled_memory_blocks_freed,
        freed[0],
        freed[1],
        freed[2],
        freed[3],
        live_most.load(Ordering::Relaxed),
        after.memory_blocks_live,
        after.empty_memory_blocks_live,
    );
    let moved = |of: fn(&crate::WorkCounts) -> u64| of(&after) - of(&before);
    eprintln!(
        "BLOCK-MAKES ahead: blocks of buffers made ahead {}, in place {}; of images made ahead {}, \
         in place {}; waited for {}, past the quiet wait {}, unused {}",
        moved(|counts| counts.buffer_blocks_made_ahead),
        moved(|counts| counts.buffer_blocks_made_in_place),
        moved(|counts| counts.image_blocks_made_ahead),
        moved(|counts| counts.image_blocks_made_in_place),
        moved(|counts| counts.blocks_ahead_waited_for),
        moved(|counts| counts.blocks_ahead_past_quiet_wait),
        moved(|counts| counts.blocks_ahead_unused),
    );
}
