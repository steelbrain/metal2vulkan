//! Heap resources whose bytes meet, and the moves of heap textures' images from
//! `VK_IMAGE_LAYOUT_UNDEFINED` that Vulkan's memory aliasing asks for: an image over bytes another
//! resource wrote holds undefined contents, and moves from undefined before it is used again
//! (vkspec "Memory Aliasing"). A device that keeps a compression state beside an image's bytes
//! needs the move, as a write through an alias leaves that state the alias's.
//!
//! Each resource placed in a heap is a [`Placement`], linked at placement with the others of its heap
//! whose bytes meet its, both ways and weakly; a release drops its links. A heap with no meeting
//! placements holds no links, and its resources' uses pay one test.
//!
//! A use that writes a placement makes its links stale. A stale texture's next use that writes it
//! moves its whole image from undefined to its home layout first, in a barrier that orders every
//! earlier command before it and every later one after it. Reads move nothing. A buffer is never
//! moved; its writes make its links stale.
//!
//! Inside a command buffer, encode order is the order its commands run in: [`Marks`] keeps what
//! the command buffer did to the placements it touched, and a write of one that a link was
//! written over since moves it in place. A write of one that nothing in the command buffer wrote,
//! nor any of its links, depends on the placement as the command buffer starts: a check at the
//! head of its segment ([`Aliased`]), which the submit thread resolves as it stages the segment,
//! in the order the device's one queue runs segments, a move it finds owed going in a command
//! buffer of its own ahead of the segment's, in its submit info ([`stage`]). A segment that read
//! one of the placement's links before the write ends first, so the move never goes ahead of that
//! read. Then the segment's writes apply, in their order.

use crate::command::DeviceLoss;
use crate::counters::Metric;
use crate::device::{lock, Core};
use crate::error::Error;
use crate::hazard::{Access, HeapId};
use crate::record::{memory_barrier, Reach, Recorder};
use ash::vk;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, DefaultHasher};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Arc, Mutex, PoisonError, Weak};

/// A heap texture's image, which its move from undefined names.
#[derive(Clone, Copy)]
pub(crate) struct Moved {
    pub(crate) image: vk::Image,
    pub(crate) home: vk::ImageLayout,
    pub(crate) aspects: vk::ImageAspectFlags,
}

impl Moved {
    /// The move of the whole image from undefined to its home layout, after every earlier
    /// command's writes and before every later command's reads and writes.
    fn barrier(self) -> vk::ImageMemoryBarrier2<'static> {
        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(self.home)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: self.aspects,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            })
    }
}

/// A link to a placement whose bytes meet another's: the placement, held weakly, and its
/// staleness, which a write of the other sets.
struct Link {
    place: Weak<Placement>,
    stale: Arc<AtomicBool>,
}

/// A resource's bytes in its heap: a heap texture's or a heap buffer's.
pub(crate) struct Placement {
    heap: Arc<HeapId>,
    /// Its first and past-last byte in the heap's block.
    start: u64,
    end: u64,
    /// A texture's image; `None` for a buffer.
    moved: Option<Moved>,
    /// The placements of its heap whose bytes meet its, changed under its heap's lock.
    links: Mutex<Vec<Link>>,
    /// It has links: the one test a placement with none pays, read without the lock.
    linked: AtomicBool,
    /// As the submit thread has staged the segments: a link was written since its own last write,
    /// or it was placed over another's bytes and is not yet written. A texture's alone is read.
    stale: Arc<AtomicBool>,
}

impl Placement {
    /// A resource placed at bytes `[start, end)` of `heap`'s block, an image where `moved`,
    /// linked with the placements of the heap whose bytes meet those. A texture with links is
    /// stale from the start: its bytes may hold another's writes.
    pub(crate) fn new(
        heap: &Arc<HeapId>,
        start: u64,
        end: u64,
        moved: Option<Moved>,
    ) -> Arc<Placement> {
        let place = Arc::new(Placement {
            heap: Arc::clone(heap),
            start,
            end,
            moved,
            links: Mutex::new(Vec::new()),
            linked: AtomicBool::new(false),
            stale: Arc::new(AtomicBool::new(false)),
        });
        let mut places = lock(heap.places());
        places.retain(|other| other.strong_count() > 0);
        // Held until the heap's lock is let go: the last handle of one dropped here would take it.
        let met: Vec<Arc<Placement>> = (places.iter())
            .filter_map(Weak::upgrade)
            .filter(|other| other.start < end && start < other.end)
            .collect();
        let mut links = Vec::with_capacity(met.len());
        for other in &met {
            let mut theirs = lock(&other.links);
            theirs.push(Link {
                place: Arc::downgrade(&place),
                stale: Arc::clone(&place.stale),
            });
            if !other.linked.swap(true, Relaxed) {
                heap.linked(true);
            }
            links.push(Link {
                place: Arc::downgrade(other),
                stale: Arc::clone(&other.stale),
            });
        }
        if !links.is_empty() {
            place.stale.store(moved.is_some(), Relaxed);
            place.linked.store(true, Relaxed);
            heap.linked(true);
            *lock(&place.links) = links;
        }
        places.push(Arc::downgrade(&place));
        drop(places);
        drop(met);
        place
    }

    /// Whether a placement of its heap meets its bytes.
    pub(crate) fn linked(&self) -> bool {
        self.linked.load(Relaxed)
    }

    /// Placed over another's bytes: its first move from the layout it is made in waits for every
    /// earlier command's writes.
    pub(crate) fn over(&self) -> bool {
        self.stale.load(Relaxed)
    }

    fn key(&self) -> usize {
        std::ptr::from_ref(self) as usize
    }
}

impl Drop for Placement {
    fn drop(&mut self) {
        let me: *const Placement = self;
        let met: Vec<Arc<Placement>> = {
            let mut places = lock(self.heap.places());
            places.retain(|other| other.strong_count() > 0);
            let links = self.links.get_mut().unwrap_or_else(PoisonError::into_inner);
            links
                .iter()
                .filter_map(|link| link.place.upgrade())
                .collect()
        };
        if *self.linked.get_mut() {
            self.heap.linked(false);
        }
        for other in &met {
            let mut links = lock(&other.links);
            links.retain(|link| !std::ptr::eq(link.place.as_ptr(), me));
            let linked = !links.is_empty();
            if other.linked.swap(linked, Relaxed) && !linked {
                self.heap.linked(false);
            }
        }
    }
}

/// Of the uses `uses` of resources, the ones of placements with links, with their accesses: the
/// uses a command's marks and moves look at. Empty, and nothing allocated, where none has links.
pub(crate) fn uses<'a>(
    uses: impl Iterator<Item = (Option<&'a Arc<Placement>>, Access)>,
) -> Vec<(&'a Arc<Placement>, Access)> {
    uses.filter_map(|(place, access)| Some((place.filter(|p| p.linked())?, access)))
        .collect()
}

/// What a placement's state is in a command buffer, as its commands so far leave it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    /// Neither it nor a link was written in the command buffer.
    Untouched,
    /// Written, and no link written since.
    Fresh,
    /// A link written since its last write in the command buffer, or with none.
    Stale,
}

/// A placement's mark in a command buffer.
struct Mark {
    /// Held so its address names no other placement while the mark lives.
    _place: Weak<Placement>,
    state: State,
    /// The segment, from 1, whose commands last read it; 0 where none did.
    read: usize,
}

/// What a command buffer's commands did to the linked placements they touched, by placement's
/// address. Its hasher takes no seed, so a command buffer that touches none makes nothing.
#[derive(Default)]
pub(crate) struct Marks {
    marks: HashMap<usize, Mark, BuildHasherDefault<DefaultHasher>>,
}

impl Marks {
    fn mark(&mut self, key: usize, place: impl FnOnce() -> Weak<Placement>) -> &mut Mark {
        self.marks.entry(key).or_insert_with(|| Mark {
            _place: place(),
            state: State::Untouched,
            read: 0,
        })
    }

    fn state(&self, place: &Placement) -> State {
        self.marks
            .get(&place.key())
            .map_or(State::Untouched, |mark| mark.state)
    }

    /// For a command whose uses of linked placements are `uses`, in segment `segment` (from 0):
    /// whether a check it adds must go at the head of a segment of its own, a link of the
    /// placement checked having been read in this one.
    pub(crate) fn splits(&self, uses: &[(&Arc<Placement>, Access)], segment: usize) -> bool {
        let read_here = |link: &Link| {
            (self.marks.get(&(link.place.as_ptr() as usize)))
                .is_some_and(|mark| mark.read == segment + 1)
        };
        (uses.iter())
            .filter(|(place, access)| access.writes() && place.moved.is_some())
            .filter(|(place, _)| self.state(place) == State::Untouched)
            .any(|(place, _)| lock(&place.links).iter().any(read_here))
    }

    /// The command's uses `uses` in segment `segment` (from 0), whose part is `part`: the checks
    /// it adds there and its writes, its marks, and the moves it records in place before it, a
    /// texture's once.
    pub(crate) fn used(
        &mut self,
        uses: &[(&Arc<Placement>, Access)],
        segment: usize,
        part: &mut Aliased,
    ) -> Vec<vk::ImageMemoryBarrier2<'static>> {
        let mut moves: Vec<vk::ImageMemoryBarrier2<'static>> = Vec::new();
        for (place, _) in uses.iter().filter(|(_, access)| access.writes()) {
            let Some(moved) = place.moved else {
                continue;
            };
            match self.state(place) {
                State::Fresh => {}
                State::Stale => {
                    if !moves.iter().any(|m| m.image == moved.image) {
                        moves.push(moved.barrier());
                    }
                }
                State::Untouched => {
                    if !part.checks.iter().any(|p| Arc::ptr_eq(p, place)) {
                        part.checks.push(Arc::clone(place));
                    }
                }
            }
        }
        for (place, access) in uses {
            if access.writes() && self.state(place) != State::Fresh {
                self.mark(place.key(), || Arc::downgrade(place)).state = State::Fresh;
                for link in lock(&place.links).iter() {
                    let key = link.place.as_ptr() as usize;
                    self.mark(key, || link.place.clone()).state = State::Stale;
                }
                part.writes.push(Arc::clone(place));
            }
            if access.reads() {
                self.mark(place.key(), || Arc::downgrade(place)).read = segment + 1;
            }
        }
        moves
    }
}

/// A segment's part: the checks at its head and its writes of linked placements, in order.
#[derive(Default)]
pub(crate) struct Aliased {
    checks: Vec<Arc<Placement>>,
    writes: Vec<Arc<Placement>>,
}

impl Aliased {
    /// For the device tests: its checks.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn checks(&self) -> usize {
        self.checks.len()
    }
}

/// As the submit thread stages a segment whose part is `part`: the moves its checks find owed,
/// recorded into a command buffer of `recorder` that goes ahead of the segment's, each counted
/// (`Metrics::heap_moves`); then its writes, applied in order. `None` where nothing is owed.
///
/// Err: the Vulkan call that failed, which loses the device.
pub(crate) fn stage(
    core: &Core,
    recorder: &mut Option<Recorder>,
    part: &Aliased,
) -> Result<Option<vk::CommandBuffer>, DeviceLoss> {
    let moves: Vec<vk::ImageMemoryBarrier2<'static>> = (part.checks.iter())
        .filter(|place| place.stale.load(Relaxed))
        .filter_map(|place| place.moved.map(Moved::barrier))
        .collect();
    for place in &part.writes {
        place.stale.store(false, Relaxed);
        for link in lock(&place.links).iter() {
            link.stale.store(true, Relaxed);
        }
    }
    if moves.is_empty() {
        return Ok(None);
    }
    let device = &core.device;
    let lost = |error: Error| match error {
        Error::Vulkan { call, result } => DeviceLoss::of(call)(result),
        error => unreachable!("a recorder fails only in a Vulkan call: {error}"),
    };
    let recorder = match recorder {
        Some(recorder) => recorder,
        None => recorder.insert(core.recorders.take(device).map_err(lost)?),
    };
    let command = recorder.next(device).map_err(lost)?;
    let earlier = [memory_barrier(Reach::Commands)];
    let dependency = vk::DependencyInfo::default()
        .memory_barriers(&earlier)
        .image_memory_barriers(&moves);
    // SAFETY: the command buffer is recording, and only the submit thread has it; the command
    // buffer it leads keeps each texture until it retires; a move from undefined names no layout
    // the image must be in.
    unsafe {
        device.cmd_pipeline_barrier2(command, &dependency);
        device.end_command_buffer(command)
    }
    .map_err(DeviceLoss::of("vkEndCommandBuffer"))?;
    core.counters.count(Metric::Barriers);
    for _ in &moves {
        core.counters.count(Metric::HeapMoves);
    }
    Ok(Some(command))
}

/// Records `moves`, heap textures' moves from undefined ([`Marks::used`]), into `command`, the
/// current segment's, in one barrier that orders every earlier command before them and every
/// later one after, each counted (`Metrics::heap_moves`).
///
/// # Safety
///
/// `command` is recording outside a rendering, and keeps each texture until it retires.
pub(crate) unsafe fn record(
    device: &ash::Device,
    command: vk::CommandBuffer,
    moves: &[vk::ImageMemoryBarrier2<'_>],
    count: &mut impl FnMut(Metric),
) {
    let earlier = [memory_barrier(Reach::Commands)];
    let dependency = vk::DependencyInfo::default()
        .memory_barriers(&earlier)
        .image_memory_barriers(moves);
    // SAFETY: the caller's contract; a move from undefined names no layout the image must be in.
    unsafe { device.cmd_pipeline_barrier2(command, &dependency) };
    count(Metric::Barriers);
    for _ in moves {
        count(Metric::HeapMoves);
    }
}
