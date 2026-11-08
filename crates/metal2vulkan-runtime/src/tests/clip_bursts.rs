//! A clip start's burst of resource makings, on the device the cfg selects: the buffers,
//! textures and decoded pictures a guest's video player has the runtime make as a clip starts,
//! made at once, held while frames go on, released as the clip ends, and made again after a
//! pause, as the next clip makes them. On an RTX 5080 Laptop GPU a guest's 4K HEVC clip's start
//! made 1175 images and 3062 buffers within a second, and its decoder's pictures, while the
//! guest's compositor went on submitting frames; NVIDIA's `vkAllocateMemory` and `vkFreeMemory`
//! hold every submission until they return (`docs/FAST-PATHS.md`).
//!
//! A compositor's thread commits a small fill every [`FRAME_GAP`] throughout, on a queue of its
//! own, and times each command buffer from its commit until it is scheduled (its `vkQueueSubmit`
//! returned). Per clip the test prints (`CLIP-BURSTS` lines) the makings' times by kind, the
//! `VkDeviceMemory` allocated and freed, the pooled blocks made and freed, the frees by cause,
//! and the compositor's command buffers that took [`HELD`] or longer to be scheduled.
//!
//! The mix is the counts' (3062 buffers, 1175 textures) at sizes of a guest's small resources,
//! and 34 decoded 4K NV12 pictures in exportable buffers (`exported_memory.rs`), as many as a
//! guest's decoder pool grew by over a clip.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::blit_holds::{device, Scratch, WAIT};
use crate::error::{Error, RuntimeGap};
use crate::{Buffer, Device, ResourceOptions, Texture, TextureDescriptor, WorkCounts};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// The buffers a clip start makes, `Shared` and `Private` in turn, of these lengths in turn.
const BUFFERS: usize = 3062;
const BUFFER_LENGTHS: [u64; 5] = [256, 4 << 10, 16 << 10, 64 << 10, 256 << 10];

/// The `Private` `BGRA8Unorm` textures a clip start makes, of these sides in turn.
const TEXTURES: usize = 1175;
const TEXTURE_SIDES: [u64; 4] = [32, 128, 256, 512];

/// The decoded pictures, `Shared` exportable buffers of a 3840 x 2160 NV12 picture's bytes.
const PICTURES: usize = 34;
const PICTURE: u64 = 3840 * 2160 * 3 / 2;

/// The clips, each made, held for [`HOLD`] and released, [`PAUSE`] apart: past the allocator's
/// and the recycler's quiet periods (a second each), as one clip's end and the next one's start
/// were apart in the guest.
const CLIPS: usize = 3;
const HOLD: Duration = Duration::from_millis(500);
const PAUSE: Duration = Duration::from_secs(3);

/// The compositor's period, and the time to be scheduled that counts as held.
const FRAME_GAP: Duration = Duration::from_millis(2);
const HELD: Duration = Duration::from_millis(1);

/// One clip's resources.
struct Clip {
    buffers: Vec<Buffer>,
    textures: Vec<Texture>,
    pictures: Vec<Buffer>,
}

/// The makings' times of one kind.
#[derive(Default)]
struct Times(Vec<Duration>);

impl Times {
    fn timed<T>(&mut self, make: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let made = make();
        self.0.push(start.elapsed());
        made
    }

    fn line(&self) -> String {
        let mut times = self.0.clone();
        times.sort();
        let total: Duration = times.iter().sum();
        match times.last() {
            None => "none".into(),
            Some(longest) => format!(
                "{} in {:.1} ms (p50 {:.3} p99 {:.3} max {:.3} ms)",
                times.len(),
                ms(total),
                ms(times[times.len() / 2]),
                ms(times[times.len() * 99 / 100]),
                ms(*longest)
            ),
        }
    }
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

/// Make one clip's resources, timing each making by kind: the pictures `None` where the device
/// exports no buffer's memory.
fn make_clip(device: &Device, exports: bool) -> (Clip, [Times; 3]) {
    let mut times: [Times; 3] = Default::default();
    let buffers = (0..BUFFERS)
        .map(|index| {
            let length = BUFFER_LENGTHS[index % BUFFER_LENGTHS.len()];
            let options = match index % 2 {
                0 => ResourceOptions::shared(),
                _ => ResourceOptions::private(),
            };
            times[0].timed(|| device.new_buffer(length, options).unwrap())
        })
        .collect();
    let textures = (0..TEXTURES)
        .map(|index| {
            let side = TEXTURE_SIDES[index % TEXTURE_SIDES.len()];
            let mut descriptor =
                TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, side, side, false);
            descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
            descriptor.resource_options = ResourceOptions::private();
            times[1].timed(|| device.new_texture(&descriptor).unwrap())
        })
        .collect();
    let pictures = match exports {
        true => (0..PICTURES)
            .map(|_| {
                times[2].timed(|| {
                    (device.new_exportable_buffer(PICTURE, ResourceOptions::shared())).unwrap()
                })
            })
            .collect(),
        false => Vec::new(),
    };
    let clip = Clip {
        buffers,
        textures,
        pictures,
    };
    (clip, times)
}

/// The compositor's command buffers: when each was committed, from the test's start, and how
/// long it took to be scheduled.
type Frames = Mutex<Vec<(Duration, Duration)>>;

/// Commit a small fill on a queue of its own every [`FRAME_GAP`] until `stop`, each command
/// buffer's commit and time to be scheduled into `frames`.
fn composite(device: &Device, start: Instant, stop: &AtomicBool, frames: &Frames) {
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let mut next = Instant::now();
    while !stop.load(Ordering::Relaxed) {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&target, 0..4096, 0x5a).unwrap();
        blit.end_encoding().unwrap();
        let committed = Instant::now();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_scheduled().unwrap();
        let scheduled = committed.elapsed();
        frames.lock().unwrap().push((committed - start, scheduled));
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        next += FRAME_GAP;
        if let Some(wait) = next.checked_duration_since(Instant::now()) {
            std::thread::sleep(wait);
        } else {
            next = Instant::now();
        }
    }
}

/// The compositor's command buffers committed within `phase` that took [`HELD`] or longer to be
/// scheduled: how many of how many, the longest, and their time past [`HELD`] summed.
fn held(frames: &Frames, phase: (Duration, Duration)) -> String {
    let frames = frames.lock().unwrap();
    let within: Vec<Duration> = (frames.iter())
        .filter(|&&(at, _)| at >= phase.0 && at < phase.1)
        .map(|&(_, scheduled)| scheduled)
        .collect();
    let late: Vec<Duration> = within.iter().copied().filter(|&s| s >= HELD).collect();
    let longest = within.iter().max().copied().unwrap_or_default();
    let past: Duration = late.iter().map(|&s| s - HELD).sum();
    format!(
        "{} of {} frames held, longest {:.2} ms, {:.1} ms past {:.0} ms in all",
        late.len(),
        within.len(),
        ms(longest),
        ms(past),
        ms(HELD)
    )
}

/// What a clip's phase changed of the counts that tell memory's churn.
fn churn(before: &WorkCounts, after: &WorkCounts) -> String {
    let (b, a) = (&before.memory_frees, &after.memory_frees);
    format!(
        "vkAllocateMemory {} vkFreeMemory {}; pooled blocks made {} freed {}; frees dedicated {} \
         emptied {} stale {}; live: pooled {} empty {} dedicated {}",
        after.object_churn.vk_memory_allocations - before.object_churn.vk_memory_allocations,
        after.object_churn.vk_memory_freed - before.object_churn.vk_memory_freed,
        after.pooled_memory_blocks_made - before.pooled_memory_blocks_made,
        after.pooled_memory_blocks_freed - before.pooled_memory_blocks_freed,
        a.dedicated - b.dedicated,
        a.pooled_emptied - b.pooled_emptied,
        a.kept_stale - b.kept_stale,
        after.memory_blocks_live,
        after.empty_memory_blocks_live,
        after.dedicated_memory_blocks_live,
    )
}

/// The clips, made, held and released [`PAUSE`] apart while the compositor runs; each clip's
/// burst and release printed with the compositor's held frames.
#[test]
fn a_clip_start_s_makings_beside_a_compositor() {
    let scratch = Scratch::new("clip-bursts");
    let device = device(&scratch.0);
    let name = device.name().to_owned();
    let exports = match device.new_exportable_buffer(PICTURE, ResourceOptions::shared()) {
        Ok(_) => true,
        Err(Error::NotImplemented(RuntimeGap::ExportableMemory)) => {
            eprintln!("CLIP-BURSTS on {name}: no exportable buffers; pictures left out");
            false
        }
        Err(error) => panic!("{name}: {error:?}"),
    };
    // Past the probe's release, and the device's own first makings.
    std::thread::sleep(PAUSE);
    let start = Instant::now();
    let stop = AtomicBool::new(false);
    let frames = Frames::default();
    let made_again = std::thread::scope(|scope| {
        scope.spawn(|| composite(&device, start, &stop, &frames));
        std::thread::sleep(PAUSE);
        let mut made_again = Vec::new();
        for clip in 0..CLIPS {
            let before = device.work_counts();
            let began = start.elapsed();
            let (resources, [buffers, textures, pictures]) = make_clip(&device, exports);
            let made = start.elapsed();
            let after = device.work_counts();
            eprintln!(
                "CLIP-BURSTS on {name}: clip {clip} made in {:.1} ms: buffers {} | textures {} | \
                 pictures {} | {} | {}",
                ms(made - began),
                buffers.line(),
                textures.line(),
                pictures.line(),
                churn(&before, &after),
                held(&frames, (began, made)),
            );
            let allocated = after.object_churn.vk_memory_allocations
                - before.object_churn.vk_memory_allocations;
            if clip > 0 {
                made_again.push(allocated);
            }
            std::thread::sleep(HOLD);
            let before = device.work_counts();
            let began = start.elapsed();
            drop(resources);
            let released = start.elapsed();
            std::thread::sleep(PAUSE);
            let after = device.work_counts();
            eprintln!(
                "CLIP-BURSTS on {name}: clip {clip} released in {:.1} ms, then a pause of {:.0} \
                 ms: {} | release: {} | pause: {}",
                ms(released - began),
                ms(PAUSE),
                churn(&before, &after),
                held(&frames, (began, released)),
                held(&frames, (released, start.elapsed())),
            );
        }
        stop.store(true, Ordering::Relaxed);
        made_again
    });
    eprintln!("CLIP-BURSTS on {name}: allocations of the clips after the first: {made_again:?}");
}
