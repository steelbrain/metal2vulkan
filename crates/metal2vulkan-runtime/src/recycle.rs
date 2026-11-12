//! Recycled images and buffers: an image or a buffer the runtime made and placed in memory of its
//! own (`Device::placed_image`, `Device::placed_new_buffer`) is parked as its last holder goes,
//! not destroyed, and the next one of the same making takes it, its `VkImage` or `VkBuffer` and
//! its memory, in place of a new one.
//!
//! - **Why.** MoltenVK 1.4.1 makes each `VkImage`'s Metal texture at its first use, and each
//!   `VkBuffer`'s Metal buffer in its memory's heap at its first use, adds it to its device's
//!   residency set and commits the set (`MVKImagePlane::getMTLTexture`, `MVKBuffer::getMTLBuffer`,
//!   `MVKDevice::makeResident`), and removes it and commits again as the image or buffer is
//!   destroyed (`releaseMTLTexture`, `MVKBuffer::detachMemory`, `removeResidency`), each under one
//!   lock for the whole device. A guest's window server makes and releases textures of the same
//!   shapes every frame, and the runtime its staging and scratch buffers with them: tens of
//!   commits a guest frame, the slow ones taking milliseconds and every other thread that makes or
//!   frees a resource meanwhile waiting on the lock. A parked image keeps its Metal texture, and a
//!   parked buffer its Metal buffer, resident.
//! - **The same making.** An image is taken only for a making of the same plan (format, extent,
//!   levels, layers, samples, usage, flags, view formats: `ImagePlan`), texture type, memory class
//!   and sharing ([`ImageMaking`]); a buffer only for one of the same size, memory class and
//!   pages, the device's or fresh ones it imported ([`BufferMaking`]; every such buffer has the
//!   same usage and the device's sharing): its
//!   `vkCreateImage` or `vkCreateBuffer` parameters and its placement's memory are what a new
//!   one's would be.
//! - **What a new owner sees.** Nothing of the last one's. An image is handed over as a new one
//!   is, to be moved from `UNDEFINED` (valid from any layout:
//!   `VUID-VkImageMemoryBarrier-oldLayout-01197`, "oldLayout must be VK_IMAGE_LAYOUT_UNDEFINED or
//!   the current layout"), and its maker's first move writes what a new texture reads: zero, or
//!   the copies in of a shadow, as for an image just made (`texture/first_move.rs`). A buffer's
//!   maker sets its bytes to zero as it does a new buffer's, on the host or by the device's next
//!   submission (`Buffer::clear_first`), at the same cost, fresh pages it imported among them
//!   (`fresh_pages.rs`), which hold the last owner's bytes; a scratch buffer's bytes hold no value
//!   before the GPU writes them, recycled or new, and an upload arena's chunk none before a region
//!   it hands out is written or zeroed (`upload.rs`). A buffer takes a range of the Metal address
//!   space as a new one does (`address.rs`): its Vulkan device address, which the recycled
//!   `VkBuffer` keeps, is never shown. An image made to start as its memory lies starts as the
//!   parked image's bytes lie, which are memory's as any placement's are.
//! - **When.** As the last holder goes: every command buffer that used the image or buffer
//!   completed (they hold it), so no GPU work reaches it. Not an image of a shareable texture, one
//!   that took its storage form, one presenting a stand-in, or one whose memory is not its
//!   placement alone; not a heap's buffer, nor a buffer of the caller's or over host pages.
//! - **The book.** The recycler keeps the making of each live object it made or handed out, by its
//!   handle, and parks an object as its last holder goes only where the book holds its handle.
//!   A driver gives a destroyed object's handle to a later object (MoltenVK's handle is the
//!   object's address), so every live object the book holds leaves it before it is destroyed:
//!   at its last holder's going (parked, or forgotten where it may not park), and an image swapped
//!   out of its texture as the image swapped out goes (`texture.rs`'s `ImageObjects`: a texture's
//!   first image, which its storage form replaced, `texture/storage_form.rs`). One left in the book
//!   would park the next image the driver makes at its handle, where the book does not note that
//!   one (a `Shared` or `Managed` texture's host-copy image), as the old one's making, and a
//!   texture of that making would take an image of another format or shape: views and copies of
//!   its format over an image of another, which MoltenVK refuses or Metal's texture view creation
//!   aborts on. Counted: `WorkCounts::object_churn.swapped_out_images_forgotten`, the images
//!   swapped out that left the book as they went.
//! - **Which.** A take hands out the latest parked object of its making, found through a map of
//!   each making's parked objects (a hash of the making, not a search of every parked one).
//! - **Bounded.** At most [`IMAGES`]' and [`BUFFERS`]' count and bytes parked, and none parked
//!   longer than their time: one parked longer goes, as it went before (destroyed, its memory
//!   freed), at the next park or take, or within a second after on the device's idle sweep
//!   (`idle_sweep.rs`) where none comes. Past the count or bytes, the one parked earliest goes of
//!   those no take has handed out before; one a take has handed out (its making came again while
//!   it was parked, as a frame's textures of the same shapes do every frame) goes first only
//!   where such hold more than three
//!   quarters of the count or bytes, or no other is parked. So a run of makings that never come
//!   again (a window resized frame by frame) evicts its own, not the objects each frame takes,
//!   and those still leave a quarter of the bounds to makings new to them. The memory a parked
//!   image or buffer holds is the runtime's, never a program object's: no deallocator, range or
//!   use count waits on it. Counted: `WorkCounts::object_churn` (`images_recycled`,
//!   `buffers_parked`, `images_evicted`, ...). A build of `--cfg m2v_no_recycle` parks nothing,
//!   the control that measures what recycling changes.

use super::memory::{Holds, Memory, MemoryClass, Placement};
use crate::image_plan::ImagePlan;
use crate::key_hash::KeyMap;
use crate::release::{self, Released};
use metal2vulkan::metal::TextureType;
use std::collections::BTreeMap;
use std::hash::Hash;
use std::time::{Duration, Instant};

/// How much a recycler parks: at most `most` objects and `most_bytes` of their memory, each for
/// at most `parked_for`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bounds {
    most: usize,
    most_bytes: u64,
    /// The longest an object stays parked: one parked longer goes at the next park or take, or on
    /// the device's idle sweep where none comes (`idle_sweep.rs`).
    pub(crate) parked_for: Duration,
}

/// The most parked: none in a build of `--cfg m2v_no_recycle` (each object then goes as it is
/// parked, as before).
const fn most(count: usize) -> usize {
    if cfg!(m2v_no_recycle) {
        0
    } else {
        count
    }
}

/// The images' bounds: 32 images, 128 MiB, a second.
pub(crate) const IMAGES: Bounds = Bounds {
    most: most(32),
    most_bytes: 128 << 20,
    parked_for: Duration::from_secs(1),
};

/// The buffers' bounds: 64 buffers, 64 MiB, a second.
pub(crate) const BUFFERS: Bounds = Bounds {
    most: most(64),
    most_bytes: 64 << 20,
    parked_for: Duration::from_secs(1),
};

/// What fixes an image's making ([`Device::placed_image`](crate::Device)): its parameters and its
/// placement's memory.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ImageMaking {
    pub(crate) plan: ImagePlan,
    pub(crate) texture_type: TextureType,
    /// The memory class asked.
    pub(crate) class: MemoryClass,
    pub(crate) families: Vec<u32>,
}

/// What fixes a buffer's making (`Device::placed_new_buffer`): its size (padded), the memory
/// class it lies in, whether it lies in fresh pages it imported (`fresh_pages.rs`), its
/// `VkBuffer` made for host allocations, and the kind of block it lies in otherwise
/// (`memory.rs`'s `Holds`), whose `VkBuffer` names the block's handle type where it has one, and
/// which a buffer that declares its padding unobserved takes plain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BufferMaking {
    pub(crate) size: u64,
    pub(crate) class: MemoryClass,
    pub(crate) fresh_pages: bool,
    pub(crate) holds: Holds,
}

/// A parked object.
struct Parked<R, M> {
    making: M,
    raw: R,
    placement: Placement,
    since: Instant,
}

/// A live object's making, and whether a take handed it out (its making came again while it was
/// parked).
struct Live<M> {
    making: M,
    taken: bool,
}

/// Parked objects of one standing, keyed by the order they were parked in, and their memory.
struct Segment<R, M> {
    parked: BTreeMap<u64, Parked<R, M>>,
    bytes: u64,
}

impl<R, M> Segment<R, M> {
    fn new() -> Self {
        Segment {
            parked: BTreeMap::new(),
            bytes: 0,
        }
    }

    fn len(&self) -> usize {
        self.parked.len()
    }

    fn put(&mut self, order: u64, parked: Parked<R, M>) {
        self.bytes += parked.placement.size();
        self.parked.insert(order, parked);
    }

    fn remove(&mut self, order: u64) -> Option<Parked<R, M>> {
        let parked = self.parked.remove(&order)?;
        self.bytes -= parked.placement.size();
        Some(parked)
    }

    /// The one parked earliest, out of the segment, where it was parked at least `parked_for`
    /// before `now` (`None` for no such limit).
    fn pop_first(&mut self, stale: Option<(Instant, Duration)>) -> Option<(u64, Parked<R, M>)> {
        let (_, first) = self.parked.first_key_value()?;
        let kept = stale.is_some_and(|(now, parked_for)| {
            now.saturating_duration_since(first.since) < parked_for
        });
        if kept {
            return None;
        }
        let (order, parked) = self.parked.pop_first()?;
        self.bytes -= parked.placement.size();
        Some((order, parked))
    }
}

/// A device's parked objects of one kind (`R`, made as `M` says), and the makings of its live
/// recyclable ones.
pub(crate) struct Recycler<R, M> {
    bounds: Bounds,
    /// Each live object made or handed out, by its handle.
    live: KeyMap<R, Live<M>>,
    /// Parked objects no take handed out before.
    trial: Segment<R, M>,
    /// Parked objects a take handed out before.
    proven: Segment<R, M>,
    /// The orders of each making's parked objects, the earliest first.
    index: KeyMap<M, Vec<u64>>,
    /// The order the next parked object takes.
    order: u64,
}

/// What a recycler hands back for destruction: objects it no longer keeps.
pub(crate) type Gone<R> = Vec<(R, Placement)>;

impl<R: Copy + Eq + Hash, M: Clone + Eq + Hash> Recycler<R, M> {
    /// A recycler that parks within `bounds`.
    pub(crate) fn new(bounds: Bounds) -> Self {
        Recycler {
            bounds,
            live: KeyMap::default(),
            trial: Segment::new(),
            proven: Segment::new(),
            index: KeyMap::default(),
            order: 0,
        }
    }

    /// A parked object of `making`, the latest parked such, now live; and the objects that stayed
    /// parked past the bounds' time at `now`, to destroy.
    pub(crate) fn take(&mut self, making: &M, now: Instant) -> (Option<(R, Placement)>, Gone<R>) {
        let taken = self.latest(making).map(|parked| {
            self.live.insert(
                parked.raw,
                Live {
                    making: parked.making,
                    taken: true,
                },
            );
            (parked.raw, parked.placement)
        });
        (taken, self.stale(now))
    }

    /// The latest parked object of `making`, out of the recycler.
    fn latest(&mut self, making: &M) -> Option<Parked<R, M>> {
        let orders = self.index.get_mut(making)?;
        let order = orders.pop()?;
        if orders.is_empty() {
            self.index.remove(making);
        }
        self.trial
            .remove(order)
            .or_else(|| self.proven.remove(order))
    }

    /// `raw`, made for `making`, is live.
    pub(crate) fn made(&mut self, raw: R, making: M) {
        self.live.insert(
            raw,
            Live {
                making,
                taken: false,
            },
        );
    }

    /// Whether `raw` is a live object this recycler may park.
    pub(crate) fn holds(&self, raw: R) -> bool {
        self.live.contains_key(&raw)
    }

    /// `raw`, live, goes: parked with `placement` (`Ok`, with the objects that go to keep the
    /// bounds), or, where it is not a live object of this recycler's, handed back (`Err`).
    pub(crate) fn park(
        &mut self,
        raw: R,
        placement: Placement,
        now: Instant,
    ) -> Result<Gone<R>, (R, Placement)> {
        let Some(Live { making, taken }) = self.live.remove(&raw) else {
            return Err((raw, placement));
        };
        let order = self.order;
        self.order += 1;
        match self.index.get_mut(&making) {
            Some(orders) => orders.push(order),
            None => {
                self.index.insert(making.clone(), vec![order]);
            }
        }
        let parked = Parked {
            making,
            raw,
            placement,
            since: now,
        };
        match taken {
            true => self.proven.put(order, parked),
            false => self.trial.put(order, parked),
        }
        let mut gone = self.stale(now);
        while self.trial.len() + self.proven.len() > self.bounds.most
            || self.trial.bytes + self.proven.bytes > self.bounds.most_bytes
        {
            let Some(evicted) = self.evict() else {
                break;
            };
            gone.push(evicted);
        }
        Ok(gone)
    }

    /// `raw`, live, goes without parking (it is not recyclable as it goes): whether the book held
    /// it.
    pub(crate) fn forget(&mut self, raw: R) -> bool {
        self.live.remove(&raw).is_some()
    }

    /// The object that goes to keep the bounds, out of the recycler: the earliest parked trial
    /// one, or the earliest parked proven one where the proven hold more than three quarters of
    /// the bounds' count or bytes, or no trial one is parked.
    fn evict(&mut self) -> Option<(R, Placement)> {
        let over_share = self.proven.len() * 4 > self.bounds.most * 3
            || self.proven.bytes * 4 > self.bounds.most_bytes * 3;
        let segment = match over_share || self.trial.parked.is_empty() {
            true => &mut self.proven,
            false => &mut self.trial,
        };
        let (order, parked) = segment.pop_first(None)?;
        Some(self.unindexed(order, parked))
    }

    /// The objects parked longer than the bounds' time at `now`, out of the recycler, to destroy,
    /// for the device's idle sweep (`idle_sweep.rs`); and when the next still parked goes.
    pub(crate) fn sweep(&mut self, now: Instant) -> (Gone<R>, Option<Instant>) {
        let gone = self.stale(now);
        (gone, self.due())
    }

    /// When the object parked earliest has stayed parked for the bounds' time: `None` where none
    /// is parked.
    pub(crate) fn due(&self) -> Option<Instant> {
        let first = |segment: &Segment<R, M>| {
            let (_, parked) = segment.parked.first_key_value()?;
            Some(parked.since)
        };
        let since = first(&self.trial)
            .into_iter()
            .chain(first(&self.proven))
            .min()?;
        Some(since + self.bounds.parked_for)
    }

    /// The objects parked longer than the bounds' time at `now`, out of the recycler.
    fn stale(&mut self, now: Instant) -> Gone<R> {
        let stale = Some((now, self.bounds.parked_for));
        let mut gone = Gone::new();
        while let Some((order, parked)) = self.trial.pop_first(stale) {
            gone.push(self.unindexed(order, parked));
        }
        while let Some((order, parked)) = self.proven.pop_first(stale) {
            gone.push(self.unindexed(order, parked));
        }
        gone
    }

    /// `parked`, out of a segment at `order`, out of the index too.
    fn unindexed(&mut self, order: u64, parked: Parked<R, M>) -> (R, Placement) {
        if let Some(orders) = self.index.get_mut(&parked.making) {
            orders.retain(|&kept| kept != order);
            if orders.is_empty() {
                self.index.remove(&parked.making);
            }
        }
        (parked.raw, parked.placement)
    }
}

impl<R: Released, M> Recycler<R, M> {
    /// Every parked object destroyed and its memory freed, and none parked from now on: each
    /// live one goes as its last holder lets go of it (`park` hands it back). A lost device whose
    /// recovery carried its objects on, on which nothing is made again (`device/carry.rs`).
    pub(crate) fn close(&mut self, device: &ash::Device, memory: &mut Memory) {
        self.live.clear();
        self.destroy(device, memory);
    }

    /// Every parked object, destroyed and its memory freed, as the device goes.
    pub(crate) fn destroy(&mut self, device: &ash::Device, memory: &mut Memory) {
        self.index.clear();
        for segment in [&mut self.trial, &mut self.proven] {
            segment.bytes = 0;
            let parked = std::iter::from_fn(|| segment.parked.pop_first());
            destroy(device, memory, parked.map(|(_, p)| (p.raw, p.placement)));
        }
    }
}

/// Destroy each object of `gone` and free its memory, as its owner's last holder let it go.
pub(crate) fn destroy<R: Released>(
    device: &ash::Device,
    memory: &mut Memory,
    gone: impl IntoIterator<Item = (R, Placement)>,
) {
    for (raw, placement) in gone {
        // SAFETY: a parked object no command uses (its last holder went), and no view of it
        // lives.
        unsafe { release::object(device, raw) };
        memory.free(device, &placement);
    }
}
