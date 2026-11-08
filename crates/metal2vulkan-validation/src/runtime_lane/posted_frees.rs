//! Device tests of the device's free thread (`free_thread.rs` in the runtime): an object whose
//! release frees a block of the runtime's own memory (a buffer or a texture of a block of its own,
//! a host-visible buffer's fresh pages where the device imports them, a heap) returns with the
//! block's `vkFreeMemory` posted to the thread (`WorkCounts::memory_frees_posted`). Such a block is
//! kept for reuse instead while its heap keeps fewer empty bytes than its budget (at most 2 GiB,
//! the runtime's `memory.rs`), so each kind makes one object at a time, each longer than the last,
//! until a release frees its block ([`Released::until_one_frees`]). Meanwhile a no-copy buffer's
//! import of the caller's pages is freed before its release returns, posted never: Vulkan has "the
//! application must ensure that the imported memory range remains valid and accessible for the
//! lifetime of the imported memory object", and the caller frees its pages once the release
//! returns. Each release's time is printed (`POSTED-FREES` lines).
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

/// The most objects of a kind [`Released::until_one_frees`] makes: from 96 MiB, each [`GROWTH`]
/// longer than the last, their bytes pass twice the 2 GiB of empty blocks a heap keeps at most.
const MOST_GROWN: usize = 24;

/// Each object [`Released::until_one_frees`] makes is an eighth longer than the last, so it takes
/// no block kept from the last (a kept block of its own takes a placement no longer than it), and
/// the process holds at most the heap's kept budget and one object at once.
const GROWTH: u64 = 8;

/// The no-copy buffers' releases timed.
const ROUNDS: usize = 4;

fn managed() -> ResourceOptions {
    ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    )
}

/// Each release's time and the counts it moved: the blocks freed, those of them posted, and the
/// host imports freed.
struct Released {
    times: Vec<Duration>,
    blocks: u64,
    posted: u64,
    imports: u64,
}

impl Released {
    fn new() -> Released {
        Released {
            times: Vec::new(),
            blocks: 0,
            posted: 0,
            imports: 0,
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

    /// Make objects through `make`, the first of `bytes` and each next [`GROWTH`] longer, each
    /// released before the next is made, until a release frees a block (at most [`MOST_GROWN`]):
    /// the runtime keeps a block of its own that empties for reuse while its heap keeps fewer
    /// empty bytes than its budget (`memory.rs`'s `KEPT_EMPTY_BYTES`), and such a block takes only
    /// a placement no longer than it, which none of these is.
    fn until_one_frees<T>(&mut self, device: &Device, mut bytes: u64, make: impl Fn(u64) -> T) {
        for _ in 0..MOST_GROWN {
            let object = make(bytes);
            self.release(device, object);
            if self.blocks > 0 {
                return;
            }
            bytes += bytes / GROWTH;
        }
    }

    fn add(&mut self, before: &WorkCounts, after: &WorkCounts) {
        self.blocks += after.memory_frees.blocks - before.memory_frees.blocks;
        self.posted += after.memory_frees_posted - before.memory_frees_posted;
        self.imports += after.memory_frees.host_import - before.memory_frees.host_import;
    }

    fn line(&self, kind: &str) -> String {
        let longest = self.times.iter().max().copied().unwrap_or_default();
        format!(
            "{kind}: {} releases, {} blocks freed, {} posted, {} host imports; longest release \
             {longest:?}",
            self.times.len(),
            self.blocks,
            self.posted,
            self.imports
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
            released.until_one_frees(device, APART, |bytes| {
                device.new_buffer(bytes, options).unwrap()
            });
            assert!(
                released.posted > 0 && released.posted <= released.blocks,
                "{kind}: the block's free was posted ({})",
                released.line(kind)
            );
            lines.push(released.line(kind));
        }
        // RGBA8 from 8192x8192, 256 MiB, past what the recycler parks of images, to Metal's
        // largest side.
        let mut released = Released::new();
        released.until_one_frees(device, 256 << 20, |bytes| {
            let side = ((bytes / 4).isqrt() / 16 * 16).min(16384);
            let mut descriptor =
                TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, side, side, false);
            descriptor.usage = TextureUsage::SHADER_READ;
            descriptor.resource_options = ResourceOptions::private();
            device.new_texture(&descriptor).unwrap()
        });
        assert!(
            released.posted > 0 && released.posted <= released.blocks,
            "a texture's block's free was posted ({})",
            released.line("private texture")
        );
        lines.push(released.line("private texture"));
        // A heap's bytes are a buffer's: past what the recycler parks too.
        let mut released = Released::new();
        released.until_one_frees(device, APART, |bytes| {
            let heap = HeapDescriptor::new(HeapType::Automatic, bytes, ResourceOptions::private());
            device.new_heap(&heap).unwrap()
        });
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
        for line in lines {
            eprintln!("POSTED-FREES on {name}: {line}");
        }
    });
}
