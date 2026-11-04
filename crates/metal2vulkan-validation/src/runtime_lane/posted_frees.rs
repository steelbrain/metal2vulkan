//! Device tests of the device's free thread (`free_thread.rs` in the runtime): an object whose
//! release frees a block of the runtime's own memory (a buffer or a texture of a block of its own,
//! a host-visible buffer's fresh pages where the device imports them, a heap) returns with the
//! block's `vkFreeMemory` posted to the thread
//! (`WorkCounts::memory_frees_posted`), while a no-copy buffer's import of the caller's pages is
//! freed before its release returns, posted never: Vulkan has "the application must ensure that the
//! imported memory range remains valid and accessible for the lifetime of the imported memory
//! object", and the caller frees its pages once the release returns. An event's release, plain or
//! shared, posts its timeline semaphore's destroy there too
//! (`WorkCounts::semaphore_destroys_posted`). Each release's time is printed (`POSTED-FREES`
//! lines).
//!
//! What the runtime crate's own tests hold the thread for (`tests/posted_frees.rs`): a pinned
//! address and a forced resource ID take a new resource while the old block's free waits, an
//! allocation at `maxMemoryAllocationCount` and a block past its heap's size wait for the posted
//! frees, and the teardown runs them all before it returns.

use super::host_memory::{no_copy, not_importable, on_device, Mapping};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    CpuCacheMode, Device, HazardTrackingMode, HeapDescriptor, HeapType, ResourceOptions,
    StorageMode, TextureDescriptor, WorkCounts,
};
use std::time::{Duration, Instant};

/// A buffer's bytes past half a pooled block (64 MiB) and past the 64 MiB of buffers the
/// runtime's recycler parks: a block of its own, freed as the buffer goes.
const APART: u64 = 96 << 20;

/// Releases of each kind timed.
const ROUNDS: usize = 4;

fn managed() -> ResourceOptions {
    ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    )
}

/// Each release's time and the counts it moved: the blocks freed, those of them posted, the
/// host imports freed, and the semaphores' destroys posted.
struct Released {
    times: Vec<Duration>,
    blocks: u64,
    posted: u64,
    imports: u64,
    semaphores: u64,
}

impl Released {
    fn new() -> Released {
        Released {
            times: Vec::new(),
            blocks: 0,
            posted: 0,
            imports: 0,
            semaphores: 0,
        }
    }

    /// Drop `object`, timed, and count what its release moved.
    fn release<T>(&mut self, device: &Device, object: T) {
        let before = device.work_counts();
        let start = Instant::now();
        drop(object);
        self.times.push(start.elapsed());
        let after = device.work_counts();
        self.add(&before, &after);
    }

    fn add(&mut self, before: &WorkCounts, after: &WorkCounts) {
        self.blocks += after.memory_frees.blocks - before.memory_frees.blocks;
        self.posted += after.memory_frees_posted - before.memory_frees_posted;
        self.imports += after.memory_frees.host_import - before.memory_frees.host_import;
        self.semaphores += after.semaphore_destroys_posted - before.semaphore_destroys_posted;
    }

    fn line(&self, kind: &str) -> String {
        let longest = self.times.iter().max().copied().unwrap_or_default();
        format!(
            "{kind}: {} releases, {} blocks freed, {} posted, {} host imports, {} semaphores' \
             destroys posted; longest release {longest:?}",
            self.times.len(),
            self.blocks,
            self.posted,
            self.imports,
            self.semaphores
        )
    }
}

#[test]
fn a_released_block_is_freed_on_the_free_thread_and_a_host_import_before_the_release_returns() {
    on_device(|device| {
        let name = device.name();
        let mut lines = Vec::new();
        for (kind, options) in [
            ("private buffer", ResourceOptions::private()),
            ("shared buffer", ResourceOptions::shared()),
            ("managed buffer", managed()),
        ] {
            let mut released = Released::new();
            for _ in 0..ROUNDS {
                let buffer = device.new_buffer(APART, options).unwrap();
                released.release(device, buffer);
            }
            assert!(
                released.posted > 0 && released.posted <= released.blocks,
                "{kind}: the block's free was posted ({})",
                released.line(kind)
            );
            lines.push(released.line(kind));
        }
        // 8192x8192 RGBA8: 256 MiB, past what the recycler parks of images.
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 8192, 8192, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::private();
        let mut released = Released::new();
        for _ in 0..ROUNDS {
            let texture = device.new_texture(&descriptor).unwrap();
            released.release(device, texture);
        }
        assert!(
            released.posted > 0 && released.posted <= released.blocks,
            "a texture's block's free was posted ({})",
            released.line("private texture")
        );
        lines.push(released.line("private texture"));
        // A heap's bytes are a buffer's: past what the recycler parks too.
        let heap = HeapDescriptor::new(HeapType::Automatic, APART, ResourceOptions::private());
        let mut released = Released::new();
        for _ in 0..ROUNDS {
            released.release(device, device.new_heap(&heap).unwrap());
        }
        assert!(
            released.posted > 0 && released.posted <= released.blocks,
            "a heap's block's free was posted ({})",
            released.line("private heap")
        );
        lines.push(released.line("private heap"));
        // The caller's pages: freed before the release returns, never posted.
        let len = 4 << 20;
        let mut released = Released::new();
        for _ in 0..ROUNDS {
            let pages = Mapping::anonymous(len);
            // SAFETY: the pages outlive the buffer, which goes before them.
            let buffer = match unsafe {
                no_copy(
                    device,
                    pages.pointer(),
                    len,
                    ResourceOptions::shared(),
                    None,
                )
            } {
                Ok(buffer) => buffer,
                Err(error) if not_importable(&error) => break,
                Err(error) => panic!("a no-copy buffer: {error:?}"),
            };
            released.release(device, buffer);
            drop(pages);
        }
        if !released.times.is_empty() {
            assert_eq!(
                (released.posted, released.imports),
                (0, released.times.len() as u64),
                "each import freed before its release returned, none posted"
            );
            lines.push(released.line("no-copy buffer"));
        }
        // Events: each release posts its semaphore's destroy.
        let mut released = Released::new();
        for _ in 0..ROUNDS {
            released.release(device, device.new_event().unwrap());
            released.release(device, device.new_shared_event().unwrap());
        }
        assert_eq!(
            released.semaphores,
            2 * ROUNDS as u64,
            "each event's semaphore destroy posted ({})",
            released.line("event")
        );
        lines.push(released.line("event"));
        for line in lines {
            eprintln!("POSTED-FREES on {name}: {line}");
        }
    });
}
