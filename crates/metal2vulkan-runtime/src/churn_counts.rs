//! Counts of the objects whose making and release cost a driver more than the call itself, and of
//! the keeps of shadowed textures' merges by who asked for them (`WorkCounts::object_churn`,
//! `WorkCounts::shadow_keeps`), for the lanes and the guest-frame breakdowns: not part of the API.
//!
//! - **Object churn.** MoltenVK 1.4.1 adds each Metal allocation it makes to its device's residency
//!   set and commits the set, under one lock for the device (`MVKDevice::makeResident`), and
//!   removes and commits again as it lets the allocation go (`removeResidency`): a `VkImage`'s
//!   texture (made at its first use), a `VkBuffer` in device-local memory, a buffer view's texture
//!   and a host-visible or imported `VkDeviceMemory`'s buffer. So each release counted here is,
//!   on MoltenVK, a commit at the release and one at the making before it. The releases of the
//!   whole process, by entry point (`release.rs`), and of a device's images and textures, by
//!   what owned them.
//! - **Object makings.** The other side of the same ledger: the `vkCreateImage`,
//!   `vkCreateBuffer`, `vkCreateImageView`, `vkAllocateMemory`, `vkBindImageMemory` and
//!   `vkBindBufferMemory` calls of the whole process, by entry point and by the place in the
//!   runtime that asked for one (`creation.rs`, their one door). What a texture or buffer
//!   creation made the driver do is then a count of that call's own work, not an inference from a
//!   driver floor timed apart, and a count holds under any load where a time does not. Each is
//!   also counted for the thread that made it, and a pooled block made ahead for the thread of
//!   the call that waited for it, where one did (`ObjectChurn`'s `_this_thread` counts): the
//!   device's thread that makes blocks ahead (`memory/ready.rs`) makes them when the device's
//!   queue uses pause, so the process's counts over a call take whatever such making fell
//!   within it.
//! - **Shadow keeps.** The keeps of merged-back shadows (`shadow_merge.rs`) by the encoder that
//!   planned them (a render pass's attachments, a compute dispatch's bindings, a blit), cut to a
//!   rectangle or of every texel, and the texels they copy.

use crate::step_aside::counted_apart;
use crate::work_counts::{CopySite, CopySites, COPY_SITES};
use std::cell::Cell;
use std::panic::Location;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;

/// The Vulkan releases of the process MoltenVK commits its residency set at, by entry point.
static VK_RELEASES: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The places in the runtime that released those, the first [`COPY_SITES`] met.
static RELEASE_SITES: LazyLock<CopySites> = LazyLock::new(CopySites::default);

/// The Vulkan makings of the process a driver charges for, by entry point ([`VkMaking`]).
static VK_MAKINGS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// The places in the runtime that made those, the first [`COPY_SITES`] met.
static MAKING_SITES: LazyLock<CopySites> = LazyLock::new(CopySites::default);

/// The releases of [`VK_RELEASES`]'s entry points made on a thread of a device's checks
/// (`step_aside.rs`), which it leaves out: the checks' work, never the program's
/// (`WorkCounts::device_checks`).
static CHECK_RELEASES: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The makings of [`VK_MAKINGS`]'s entry points made on a thread of a device's checks, which it
/// leaves out.
static CHECK_MAKINGS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// An entry point [`VK_RELEASES`] counts.
#[derive(Clone, Copy)]
pub(crate) enum VkRelease {
    Image = 0,
    Buffer = 1,
    BufferView = 2,
    Memory = 3,
}

/// An entry point [`VK_MAKINGS`] counts: the makings a driver charges for, whose cost one call of
/// the API carries (`creation.rs`).
#[derive(Clone, Copy)]
pub(crate) enum VkMaking {
    Image = 0,
    Buffer = 1,
    ImageView = 2,
    Memory = 3,
    ImageBinding = 4,
    BufferBinding = 5,
}

/// A release of `what` (`release::object`), called `at`: a check's where a device's check made
/// it ([`CHECK_RELEASES`]).
pub(crate) fn vk_released(what: VkRelease, at: &Location<'static>) {
    if counted_apart() {
        CHECK_RELEASES[what as usize].fetch_add(1, Ordering::Relaxed);
        return;
    }
    VK_RELEASES[what as usize].fetch_add(1, Ordering::Relaxed);
    let command = match what {
        VkRelease::Image => "vkDestroyImage",
        VkRelease::Buffer => "vkDestroyBuffer",
        VkRelease::BufferView => "vkDestroyBufferView",
        VkRelease::Memory => "vkFreeMemory",
    };
    RELEASE_SITES.count(command, at, 0);
}

thread_local! {
    /// Of [`VK_MAKINGS`], those this thread made, and those another thread made for a call of
    /// this thread that waited for them ([`count_on_this_thread`]).
    static THIS_THREAD: [Cell<u64>; 6] = const { [const { Cell::new(0) }; 6] };
}

/// A making of `what` (`creation.rs`), called `at`: a check's where a device's check made it
/// ([`CHECK_MAKINGS`]).
pub(crate) fn vk_made(what: VkMaking, at: &Location<'static>) {
    if counted_apart() {
        CHECK_MAKINGS[what as usize].fetch_add(1, Ordering::Relaxed);
        return;
    }
    VK_MAKINGS[what as usize].fetch_add(1, Ordering::Relaxed);
    THIS_THREAD.with(|counts| {
        let count = &counts[what as usize];
        count.set(count.get() + 1);
    });
    MAKING_SITES.count(what.command(), at, 0);
}

/// This thread's makings so far, by [`VkMaking`] ([`THIS_THREAD`]).
pub(crate) fn this_thread_makings() -> [u64; 6] {
    THIS_THREAD.with(|counts| counts.each_ref().map(Cell::get))
}

/// Count `makings` (by [`VkMaking`]), which another thread made while a call of this one waited
/// for them (a pooled block made ahead, `memory/ready.rs`), as this thread's.
pub(crate) fn count_on_this_thread(makings: [u64; 6]) {
    THIS_THREAD.with(|counts| {
        for (count, made) in counts.iter().zip(makings) {
            count.set(count.get() + made);
        }
    });
}

/// The process's devices' checks' makings of `what` so far.
pub(crate) fn checks_made(what: VkMaking) -> u64 {
    CHECK_MAKINGS[what as usize].load(Ordering::Relaxed)
}

/// The process's devices' checks' releases of `what` so far.
pub(crate) fn checks_released(what: VkRelease) -> u64 {
    CHECK_RELEASES[what as usize].load(Ordering::Relaxed)
}

impl VkMaking {
    /// Its Vulkan entry point.
    fn command(self) -> &'static str {
        match self {
            VkMaking::Image => "vkCreateImage",
            VkMaking::Buffer => "vkCreateBuffer",
            VkMaking::ImageView => "vkCreateImageView",
            VkMaking::Memory => "vkAllocateMemory",
            VkMaking::ImageBinding => "vkBindImageMemory",
            VkMaking::BufferBinding => "vkBindBufferMemory",
        }
    }
}

/// What owned an image or a texture as it went.
#[derive(Clone, Copy)]
pub(crate) enum Owned {
    /// An image of memory of its own (`Owner::Runtime`).
    ImageOwnMemory = 0,
    /// A linear image in a buffer's memory (`Owner::Buffer`).
    ImageInBuffer = 1,
    /// An image in a heap's memory (`Owner::Heap`).
    ImageInHeap = 2,
    /// A heap's shadow or unaliased image, a view's shadow, a sparse image (its own memory).
    ImageOther = 3,
    /// A program's texture whose storage was an image (`Storage::Image`), not a heap's.
    TextureImage = 4,
    /// A heap's texture.
    TextureInHeap = 5,
    /// A texture over a buffer's bytes in an image of its own (`Storage::Shadow`).
    TextureShadow = 6,
    /// A texture over a buffer's bytes as a linear image of them (`Storage::Alias`).
    TextureAlias = 7,
    /// A texture over a buffer's bytes as texels (`Storage::Texels`).
    TextureTexels = 8,
}

/// What a recycler parks (`recycle.rs`).
#[derive(Clone, Copy)]
pub(crate) enum Recycled {
    Image = 0,
    Buffer = 1,
}

/// Who planned a keep ([`ShadowKeeps`]).
#[derive(Clone, Copy)]
pub(crate) enum KeepBy {
    Render = 0,
    Compute = 1,
    Blit = 2,
}

/// A device's counts of the above.
#[derive(Default)]
pub(crate) struct ChurnCounters {
    owned: [AtomicU64; 9],
    /// By [`Recycled`]: the objects a making took from a recycler, parked there, and let go of
    /// past its bounds (`recycle.rs`).
    recycled: [[AtomicU64; 3]; 2],
    /// Images swapped out of their textures that left the image recycler's book as they went.
    swapped_out_forgotten: AtomicU64,
    /// Imports made over host addresses a live import holds some of.
    imports_inside_live: AtomicU64,
    /// No-copy buffers bound inside a live import's memory, with no import of their own.
    buffers_bound_in_import: AtomicU64,
    /// Imports of a no-copy object's own pages whose guest RAM pages a live chunk import holds.
    imports_over_ram_chunk_pages: AtomicU64,
    /// By [`KeepBy`], then cut or whole: the keeps, then their texels.
    keeps: [[AtomicU64; 4]; 3],
}

impl ChurnCounters {
    /// An image or texture `owned` so went.
    pub(crate) fn released(&self, owned: Owned) {
        self.owned[owned as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// An import was made over host addresses a live import holds some of.
    pub(crate) fn import_inside_live(&self) {
        self.imports_inside_live.fetch_add(1, Ordering::Relaxed);
    }

    /// A no-copy buffer was bound inside a live import's memory, with no import of its own.
    pub(crate) fn buffer_bound_in_import(&self) {
        self.buffers_bound_in_import.fetch_add(1, Ordering::Relaxed);
    }

    /// A no-copy object imported its own pages, some of whose guest RAM pages a live chunk
    /// import holds.
    pub(crate) fn import_over_ram_chunk_pages(&self) {
        (self.imports_over_ram_chunk_pages).fetch_add(1, Ordering::Relaxed);
    }

    /// A making took a parked object of `what`.
    pub(crate) fn taken(&self, what: Recycled) {
        self.recycled[what as usize][0].fetch_add(1, Ordering::Relaxed);
    }

    /// An object of `what` went to its recycler.
    pub(crate) fn parked(&self, what: Recycled) {
        self.recycled[what as usize][1].fetch_add(1, Ordering::Relaxed);
    }

    /// `count` parked objects of `what` went, past their recycler's bounds.
    pub(crate) fn evicted(&self, what: Recycled, count: usize) {
        self.recycled[what as usize][2].fetch_add(count as u64, Ordering::Relaxed);
    }

    /// An image swapped out of its texture, which the image recycler's book held as live, left
    /// the book as it went (`recycle.rs`, "The book").
    pub(crate) fn swapped_out_forgotten(&self) {
        self.swapped_out_forgotten.fetch_add(1, Ordering::Relaxed);
    }

    /// A keep planned `by`, cut to a rectangle or not, of `texels`.
    pub(crate) fn kept(&self, by: KeepBy, cut: bool, texels: u64) {
        let row = &self.keeps[by as usize];
        let at = if cut { 0 } else { 2 };
        row[at].fetch_add(1, Ordering::Relaxed);
        row[at + 1].fetch_add(texels, Ordering::Relaxed);
    }

    /// The counts now.
    pub(crate) fn snapshot(&self) -> (ObjectChurn, ShadowKeeps) {
        let owned = |o: Owned| self.owned[o as usize].load(Ordering::Relaxed);
        let vk = |r: VkRelease| VK_RELEASES[r as usize].load(Ordering::Relaxed);
        let made = |m: VkMaking| VK_MAKINGS[m as usize].load(Ordering::Relaxed);
        let this_thread = this_thread_makings();
        let mine = |m: VkMaking| this_thread[m as usize];
        let keep = |by: KeepBy, at: usize| self.keeps[by as usize][at].load(Ordering::Relaxed);
        let recycled =
            |what: Recycled, at: usize| self.recycled[what as usize][at].load(Ordering::Relaxed);
        (
            ObjectChurn {
                vk_images_destroyed: vk(VkRelease::Image),
                vk_buffers_destroyed: vk(VkRelease::Buffer),
                vk_buffer_views_destroyed: vk(VkRelease::BufferView),
                vk_memory_freed: vk(VkRelease::Memory),
                vk_images_made: made(VkMaking::Image),
                vk_buffers_made: made(VkMaking::Buffer),
                vk_image_views_made: made(VkMaking::ImageView),
                vk_memory_allocations: made(VkMaking::Memory),
                vk_image_bindings: made(VkMaking::ImageBinding),
                vk_buffer_bindings: made(VkMaking::BufferBinding),
                vk_images_made_this_thread: mine(VkMaking::Image),
                vk_buffers_made_this_thread: mine(VkMaking::Buffer),
                vk_image_views_made_this_thread: mine(VkMaking::ImageView),
                vk_memory_allocations_this_thread: mine(VkMaking::Memory),
                vk_image_bindings_this_thread: mine(VkMaking::ImageBinding),
                vk_buffer_bindings_this_thread: mine(VkMaking::BufferBinding),
                making_sites: MAKING_SITES.read(),
                making_sites_dropped: MAKING_SITES.dropped(),
                images_own_memory: owned(Owned::ImageOwnMemory),
                images_in_buffers: owned(Owned::ImageInBuffer),
                images_in_heaps: owned(Owned::ImageInHeap),
                images_other: owned(Owned::ImageOther),
                textures_image: owned(Owned::TextureImage),
                textures_in_heaps: owned(Owned::TextureInHeap),
                textures_shadow: owned(Owned::TextureShadow),
                textures_alias: owned(Owned::TextureAlias),
                textures_texels: owned(Owned::TextureTexels),
                images_recycled: recycled(Recycled::Image, 0),
                images_parked: recycled(Recycled::Image, 1),
                images_evicted: recycled(Recycled::Image, 2),
                swapped_out_images_forgotten: self.swapped_out_forgotten.load(Ordering::Relaxed),
                buffers_recycled: recycled(Recycled::Buffer, 0),
                buffers_parked: recycled(Recycled::Buffer, 1),
                buffers_evicted: recycled(Recycled::Buffer, 2),
                imports_inside_live: self.imports_inside_live.load(Ordering::Relaxed),
                buffers_bound_in_import: self.buffers_bound_in_import.load(Ordering::Relaxed),
                imports_over_ram_chunk_pages: self
                    .imports_over_ram_chunk_pages
                    .load(Ordering::Relaxed),
                release_sites: RELEASE_SITES.read(),
                release_sites_dropped: RELEASE_SITES.dropped(),
            },
            ShadowKeeps {
                render_cut: keep(KeepBy::Render, 0),
                render_cut_texels: keep(KeepBy::Render, 1),
                render_whole: keep(KeepBy::Render, 2),
                render_whole_texels: keep(KeepBy::Render, 3),
                compute_cut: keep(KeepBy::Compute, 0),
                compute_cut_texels: keep(KeepBy::Compute, 1),
                compute_whole: keep(KeepBy::Compute, 2),
                compute_whole_texels: keep(KeepBy::Compute, 3),
                blit_cut: keep(KeepBy::Blit, 0),
                blit_cut_texels: keep(KeepBy::Blit, 1),
                blit_whole: keep(KeepBy::Blit, 2),
                blit_whole_texels: keep(KeepBy::Blit, 3),
            },
        )
    }
}

/// Released objects MoltenVK commits its residency set for (the module docs): the process's
/// Vulkan releases by entry point, and a device's images and program textures by what owned them.
/// A field of `WorkCounts`, for the lanes: not part of the API.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ObjectChurn {
    /// `vkDestroyImage` calls of the process.
    pub vk_images_destroyed: u64,
    /// `vkDestroyBuffer` calls of the process.
    pub vk_buffers_destroyed: u64,
    /// `vkDestroyBufferView` calls of the process.
    pub vk_buffer_views_destroyed: u64,
    /// `vkFreeMemory` calls of the process.
    pub vk_memory_freed: u64,
    /// `vkCreateImage` calls of the process (`creation.rs`). An image a making took from the
    /// recycler is not one of these, so this and `images_recycled` are separate tallies and do
    /// not sum to the images a run used: reading 0 here beside 2 recycled is a run that made
    /// none, not a contradiction.
    pub vk_images_made: u64,
    /// `vkCreateBuffer` calls of the process. As with `vk_images_made`, a buffer taken from the
    /// recycler is not one of these; `buffers_recycled` counts those.
    pub vk_buffers_made: u64,
    /// `vkCreateImageView` calls of the process.
    pub vk_image_views_made: u64,
    /// `vkAllocateMemory` calls of the process, the tries a caller that walks the memory types
    /// made included.
    pub vk_memory_allocations: u64,
    /// `vkBindImageMemory` calls of the process.
    pub vk_image_bindings: u64,
    /// `vkBindBufferMemory` calls of the process.
    pub vk_buffer_bindings: u64,
    /// Of `vk_images_made`, those of the thread that read these counts: its own calls', and
    /// those another thread made while a call of it waited for them (a pooled block made ahead
    /// that a creation call found none of and waited for, `memory/ready.rs`), counted on the
    /// thread that took the block. What that thread's calls had the driver make, without what
    /// the device's other threads made meanwhile, such as a block made ahead for a pool that
    /// grew, which no call waited for. Each thread reads its own.
    pub vk_images_made_this_thread: u64,
    /// As `vk_images_made_this_thread`, of `vk_buffers_made` (a block of buffers made ahead
    /// carries one where its buffers take device addresses).
    pub vk_buffers_made_this_thread: u64,
    /// As `vk_images_made_this_thread`, of `vk_image_views_made`.
    pub vk_image_views_made_this_thread: u64,
    /// As `vk_images_made_this_thread`, of `vk_memory_allocations` (a block made ahead is one).
    pub vk_memory_allocations_this_thread: u64,
    /// As `vk_images_made_this_thread`, of `vk_image_bindings`.
    pub vk_image_bindings_this_thread: u64,
    /// As `vk_images_made_this_thread`, of `vk_buffer_bindings`.
    pub vk_buffer_bindings_this_thread: u64,
    /// The device's images that went, of memory of their own.
    pub images_own_memory: u64,
    /// Linear images in buffers' memory.
    pub images_in_buffers: u64,
    /// Images in heaps' memory.
    pub images_in_heaps: u64,
    /// Heaps' shadow and unaliased images, views' shadows, sparse images.
    pub images_other: u64,
    /// The program's textures that went whose storage was an image, not a heap's.
    pub textures_image: u64,
    /// Heaps' textures.
    pub textures_in_heaps: u64,
    /// Textures over buffers' bytes in images of their own (shadows).
    pub textures_shadow: u64,
    /// Textures over buffers' bytes as linear images of them.
    pub textures_alias: u64,
    /// Textures over buffers' bytes as texels.
    pub textures_texels: u64,
    /// Images a making took from the recycler, rather than making one (`recycle.rs`).
    pub images_recycled: u64,
    /// Images that went to the recycler as their textures went, rather than being destroyed.
    pub images_parked: u64,
    /// Parked images destroyed past the recycler's bounds.
    pub images_evicted: u64,
    /// Images swapped out of their textures (a texture's first image, which its storage form
    /// replaced, `texture/storage_form.rs`) that the recycler's book of live images held and that
    /// left it as they were destroyed (`recycle.rs`, "The book"): each a handle that, left there,
    /// would have named the next image the driver made at it as the old one's making.
    pub swapped_out_images_forgotten: u64,
    /// The runtime's buffers a making took from the recycler, rather than making one.
    pub buffers_recycled: u64,
    /// Buffers that went to the recycler as their last holder went, rather than being destroyed.
    pub buffers_parked: u64,
    /// Parked buffers destroyed past the recycler's bounds.
    pub buffers_evicted: u64,
    /// Imports made over host addresses a live import held some of all the same
    /// (`host_aliases.rs`, "One page, one import"): a no-copy buffer's or heap's own import (a
    /// heap's own buffer's among them) over a live import of another no-copy buffer or heap, or
    /// over a live import of a guest RAM's chunk (`host_ram.rs`), and a chunk's import over a
    /// live import of a no-copy buffer or heap made as the chunk's was being made (a chunk such an
    /// import held pages of before is not imported, `WorkCounts::host_ram_chunks_held_back`). A
    /// page imported twice, which
    /// `VK_EXT_external_memory_host` leaves undefined for a host pointer; every other buffer
    /// over pages a live import holds lies inside it (`buffers_bound_in_import`, or a chunk's
    /// `WorkCounts::chunk_bound_buffers_made`) or is a copy
    /// (`WorkCounts::copied_over_live_imports`). Counted by host address, after each import is
    /// made: two made at once over the same pages, each looking before the other's was noted,
    /// count. A listed RAM's run imported alone while its chunk's import is made (two dma-bufs of
    /// the same pages, `WorkCounts::run_imports_live`) does not.
    pub imports_inside_live: u64,
    /// No-copy buffers bound inside the memory of a live import that holds all their pages, at
    /// their offset, with no import of their own (`host_aliases.rs`, "One page, one import"):
    /// over part of another no-copy buffer's or heap's bytes.
    pub buffers_bound_in_import: u64,
    /// Evidence, not a defect count: no-copy buffers' and heaps' own imports of their pages at
    /// addresses other than their guest RAM's (a view of scattered runs, `HostMemory::with_runs`)
    /// where a live import of one of the RAM's chunks held some of the same RAM pages at the
    /// RAM's own addresses as the object's import was made. The same physical pages mapped for
    /// the device twice, at two host addresses; counted at the object's import, not at a chunk's
    /// import made later over the pages of a live view's.
    pub imports_over_ram_chunk_pages: u64,
    /// The places that released the images, buffers, buffer views and memory counted above, each
    /// with its calls (`regions` unused): the first 32 the process met.
    pub release_sites: [CopySite; COPY_SITES],
    /// The releases counted above made at places past those 32, which no slot holds: while this
    /// is 0 the places account for every release counted here, and past it they do not, so a
    /// breakdown read from `release_sites` alone is short by this many. The runtime calls these
    /// entry points from far more than 32 places, so a long run reaches it.
    pub release_sites_dropped: u64,
    /// The places that made the images, buffers, views, memory and bindings counted above, each
    /// with its calls (`regions` unused): the first 32 the process met. What a creation call of
    /// the API made the driver do, and the line of the runtime that asked for it.
    pub making_sites: [CopySite; COPY_SITES],
    /// The makings counted above made at places past those 32, as `release_sites_dropped` is for
    /// the releases: while it is 0 `making_sites` accounts for every making counted here.
    pub making_sites_dropped: u64,
}

/// The keeps of merged-back shadows by who planned them, cut to a rectangle or of every texel
/// (the module docs). A field of `WorkCounts`, for the lanes: not part of the API.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShadowKeeps {
    /// A render pass's keeps cut to a rectangle.
    pub render_cut: u64,
    /// Their texels.
    pub render_cut_texels: u64,
    /// A render pass's keeps of every texel.
    pub render_whole: u64,
    /// Their texels.
    pub render_whole_texels: u64,
    /// A compute dispatch's keeps cut to a rectangle.
    pub compute_cut: u64,
    /// Their texels.
    pub compute_cut_texels: u64,
    /// A compute dispatch's keeps of every texel.
    pub compute_whole: u64,
    /// Their texels.
    pub compute_whole_texels: u64,
    /// A blit's keeps cut to a rectangle.
    pub blit_cut: u64,
    /// Their texels.
    pub blit_cut_texels: u64,
    /// A blit's keeps of every texel.
    pub blit_whole: u64,
    /// Their texels.
    pub blit_whole_texels: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A device check's makings and releases, on its thread, are counted as the checks' alone, so
    /// the process's counts (`ObjectChurn`), which a program's or a case's cost reads, carry none
    /// of them. The test's own thread is marked: each test has a thread of its own.
    #[test]
    fn a_check_s_makings_and_releases_are_its_alone() {
        let process = || {
            let churn = ChurnCounters::default().snapshot().0;
            (churn.vk_memory_allocations, churn.vk_memory_freed)
        };
        let checks = || {
            (
                checks_made(VkMaking::Memory),
                checks_released(VkRelease::Memory),
            )
        };
        let (before, before_checks) = (process(), checks());
        crate::step_aside::mark_a_check_s_thread(crate::step_aside::Counted::Apart);
        vk_made(VkMaking::Memory, Location::caller());
        vk_released(VkRelease::Memory, Location::caller());
        assert_eq!(
            process(),
            before,
            "the process's counts took a check's work"
        );
        assert_eq!(checks(), (before_checks.0 + 1, before_checks.1 + 1));
    }

    /// The in-place check's thread, whose paths carry the program's work too, counts its makings
    /// and releases as the process's (`step_aside::Counted::AsTheDevice`).
    #[test]
    fn a_check_s_makings_through_the_program_s_paths_are_the_process_s() {
        let process = || {
            let churn = ChurnCounters::default().snapshot().0;
            (churn.vk_memory_allocations, churn.vk_memory_freed)
        };
        let checks = || {
            (
                checks_made(VkMaking::Memory),
                checks_released(VkRelease::Memory),
            )
        };
        let (before, before_checks) = (process(), checks());
        crate::step_aside::mark_a_check_s_thread(crate::step_aside::Counted::AsTheDevice);
        vk_made(VkMaking::Memory, Location::caller());
        vk_released(VkRelease::Memory, Location::caller());
        assert_eq!(process(), (before.0 + 1, before.1 + 1));
        assert_eq!(
            checks(),
            before_checks,
            "the checks' counts took the device's work"
        );
    }

    /// A thread's makings are counted for it alone: another thread's, such as a block the ready
    /// thread makes ahead while a call runs, move the process's counts and not the call's
    /// thread's; and a making counted on this thread for a call that waited for it moves this
    /// thread's alone, the process having counted it where it was made.
    #[test]
    fn a_thread_s_makings_are_its_own_and_those_it_waited_for() {
        let snapshot = || ChurnCounters::default().snapshot().0;
        let process = |c: &ObjectChurn| c.vk_memory_allocations;
        let mine = |c: &ObjectChurn| {
            (
                c.vk_memory_allocations_this_thread,
                c.vk_buffers_made_this_thread,
                c.vk_buffer_bindings_this_thread,
            )
        };
        let start = snapshot();
        std::thread::spawn(|| {
            vk_made(VkMaking::Memory, Location::caller());
            vk_made(VkMaking::Buffer, Location::caller());
        })
        .join()
        .unwrap();
        let other = snapshot();
        assert_eq!(process(&other), process(&start) + 1, "the process's count");
        assert_eq!(mine(&other), mine(&start), "another thread's making");

        vk_made(VkMaking::Memory, Location::caller());
        let own = snapshot();
        assert_eq!(process(&own), process(&other) + 1);
        assert_eq!(mine(&own).0, mine(&other).0 + 1, "this thread's making");

        let mut waited = [0; 6];
        waited[VkMaking::Memory as usize] = 1;
        waited[VkMaking::Buffer as usize] = 1;
        waited[VkMaking::BufferBinding as usize] = 1;
        count_on_this_thread(waited);
        let after = snapshot();
        assert_eq!(process(&after), process(&own), "counted where it was made");
        assert_eq!(
            mine(&after),
            (mine(&own).0 + 1, mine(&own).1 + 1, mine(&own).2 + 1),
            "a making a call of this thread waited for"
        );
    }
}
