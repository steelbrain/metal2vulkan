//! Textures over a buffer in an image of their own: where the device's linear image does not hold
//! a texture over a buffer in place ([`Buffer::new_texture`](crate::Buffer::new_texture)), its
//! texels live in an optimal image of the runtime's, and the device copies the bytes shaders
//! reach (a Managed buffer's device copy, where it has one) into it and back, one
//! `vkCmdCopyBufferToImage` or `vkCmdCopyImageToBuffer` region a texture, with no CPU copy. A copy
//! in writes the image's whole level and slice, so its move starts from the undefined layout and
//! discards what the image held first; a copy back's move starts from the image's home layout,
//! which keeps it. Over a Managed buffer's device copy, only Metal's Managed events copy
//! ([`ManagedOvers`]); over any other buffer, the Shared rules, but for a buffer in an exported
//! block, whose leading copy waits for an event that changes its bytes, a write through an address
//! among them only once an address of the block went out ([`ExportedOvers`]):
//!
//! - **In, once a submission.** The upload that leads a submission copies the bytes into the
//!   image of every such texture a segment of the submission uses before any command of the
//!   segment wrote the buffer (the segment leans on the leading copy), all between one pair of
//!   barriers. Metal reads Shared
//!   memory when the command buffer executes; every member of a group was committed before the
//!   group was submitted and none has been seen complete, so a CPU write between two members'
//!   commits races an uncompleted command buffer, and Metal does not define which bytes it sees;
//!   the bytes at the group's execution are one of the allowed outcomes. A guest that waits for a
//!   completion before writing has, by that fact, a command buffer not in a held group.
//! - **In again, after a write of the buffer.** A command that uses the texture after a command
//!   of its segment wrote the buffer copies the bytes in first, as it is encoded. A segment that
//!   leans on the leading copy for a buffer an earlier segment of the held group wrote leads a new
//!   group, the one held going first (`submit`), so its leading copy follows the write.
//! - **Back, after a write of the texture.** A command that wrote the texture leaves its image to
//!   be copied back: before the next command of the segment that reaches the buffer, that uses
//!   another such texture over bytes it holds, or that reaches memory through an address outside
//!   its resources, and at the segment's end, whose end barrier then reaches the host for a
//!   Shared buffer. A segment's end is a submission's edge, so the next submission's leading copy
//!   reads what the image held.
//! - **Across textures over the same bytes.** Two textures over one buffer whose bytes meet alias,
//!   as in Metal: a write through one makes the other's image stale, and the other's next use
//!   takes the first's copy back, then its own copy in. Textures over bytes that do not meet owe
//!   each other nothing.
//!
//! Each copy is counted (`Metrics::over_copies`, and its bytes in `Metrics::over_copy_bytes`
//! and `Metrics::bytes_copied`; a copy over a buffer in an imported block in
//! `Metrics::over_copies_imported` and `Metrics::over_copy_bytes_imported` too). What a command
//! reaches is tracked by the memory it lies in ([`Span`]): a block and the bytes of it, never a
//! buffer's id, so textures over two buffers over the same bytes of a block alias as textures
//! over one buffer do, and a write of a buffer makes stale every such texture whose bytes meet
//! the buffer's, over whichever buffer it lies.
//!
//! A command hands over what it reaches ([`Reached`]): each texture says itself whether it is such
//! a texture, and a buffer it only reads owes nothing while no texture the segment used has an
//! image ahead of its buffer ([`Overs::ahead`]), so a command looks at nothing else. What a
//! segment wrote is kept in sets by block ([`Written`]), so a write costs one lookup however many
//! the segment made.

use crate::copy_out;
use crate::counters::{Counters, Metric};
use crate::device::lock;
use crate::hazard::Access;
use crate::kept::AddressHasher;
use crate::memory::BlockId;
use crate::record::{memory_barrier, Reach};
use crate::texture::{Image, OverCopy, TextureInner, TextureObject};
use ash::vk;
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, Weak};

/// Bytes of memory a resource lies in: `start..end` of block `block`, from the block's first
/// byte. Buffers over the same bytes of a block lie in the same span, whatever their ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Span {
    pub(crate) block: BlockId,
    pub(crate) start: u64,
    pub(crate) end: u64,
}

impl Span {
    /// Whether the two hold a byte in common.
    fn meets(self, other: Span) -> bool {
        self.block == other.block && meets((self.start, self.end), (other.start, other.end))
    }
}

/// The spans a segment's commands wrote, each once, in a set for each block: a write costs one
/// lookup however many the segment made, and a texture's first use looks only through its
/// block's.
#[derive(Default)]
struct Written {
    blocks: HashMap<BlockId, HashSet<(u64, u64)>>,
}

impl Written {
    fn insert(&mut self, span: Span) {
        (self.blocks.entry(span.block).or_default()).insert((span.start, span.end));
    }

    /// Whether a span written meets `span`.
    fn meets(&self, span: Span) -> bool {
        (self.blocks.get(&span.block)).is_some_and(|spans| {
            spans
                .iter()
                .any(|&written| meets(written, (span.start, span.end)))
        })
    }

    fn extend(&mut self, other: &Written) {
        for (block, spans) in &other.blocks {
            self.blocks.entry(*block).or_default().extend(spans);
        }
    }
}

/// A resource a command reaches, as the copies read it: a texture, which says itself whether it
/// lives in an image of its own over a buffer, or a buffer, by the memory it lies in.
#[derive(Clone, Copy)]
pub(crate) enum Reached<'a> {
    /// A texture: one in place over a buffer reaches the buffer.
    Texture(&'a Arc<TextureInner>),
    /// A buffer, by the bytes of memory it lies in
    /// ([`BufferObject::span`](crate::buffer::BufferObject::span)).
    Buffer(Span),
}

/// 1 for a texture over a buffer in an image of its own that the Shared rules copy, 0 for any
/// other or none: what an encoder's count of the ones bound takes for a slot.
pub(crate) fn copied(texture: Option<&Arc<TextureInner>>) -> u32 {
    u32::from(texture.is_some_and(|texture| texture.copied_shared()))
}

/// What a segment does to its textures over a buffer in an image of their own. Once it has
/// ended ([`Overs::end`]), what the submit thread reads of it: the copies the leading copy makes,
/// and the buffers it wrote.
#[derive(Default)]
pub(crate) struct Overs {
    /// The textures it used; once it has ended, those that lean on the leading copy alone.
    used: Vec<Used>,
    /// The blocks the used textures lie in.
    under: HashSet<BlockId, BuildHasherDefault<AddressHasher>>,
    /// The memory the segment's commands wrote other than by a copy back.
    wrote: Written,
    /// The memory copies back wrote, each span with the texture whose copy it was: each pair
    /// once.
    backed: Vec<(Span, u32)>,
    /// A command of the segment wrote through an address outside its resources, or was recorded
    /// while no such texture lived, its writes untracked.
    unbound: bool,
    /// A command of the segment was recorded while no such texture lived, its writes untracked
    /// ([`untracked`](Self::untracked)).
    untracked: bool,
    /// The textures used whose images a command wrote since their bytes last went back
    /// (`Used::buffer_stale`).
    ahead: usize,
}

/// A texture over a buffer in an image of its own that a segment used.
struct Used {
    /// The texture's id, which its views share.
    texture: u32,
    /// The copy between its buffer and its image, with the memory its texels lie in
    /// ([`OverCopy::memory`]).
    copy: OverCopy,
    /// Its first use in the segment came before any write of the buffer in it: the image holds
    /// the bytes of the submission's leading copy.
    leans: bool,
    /// A command wrote memory its texels lie in, or another such texture's image over bytes this
    /// one holds, after the image last took its bytes.
    image_stale: bool,
    /// A command wrote the image after its bytes last went back.
    buffer_stale: bool,
}

/// What a held group's ended segments wrote, as [`Overs`] keeps it.
#[derive(Default)]
pub(crate) struct Leans {
    wrote: Written,
    backed: Vec<(Span, u32)>,
    unbound: bool,
    untracked: bool,
}

impl Leans {
    /// Whether `later`, an ended segment after these in one submission, leans on the leading copy
    /// for memory these wrote (a texture's own copy back aside): where so, it leads a new group.
    pub(crate) fn conflicts(&self, later: &Overs) -> bool {
        (later.used.iter()).any(|used| {
            let memory = used.copy.memory;
            self.unbound
                || self.wrote.meets(memory)
                || (self.backed.iter()).any(|&(span, by)| span.meets(memory) && by != used.texture)
        })
    }

    /// Adds the writes of `later`, an ended segment.
    pub(crate) fn join(&mut self, later: &Overs) {
        self.wrote.extend(&later.wrote);
        self.backed.extend_from_slice(&later.backed);
        self.unbound |= later.unbound;
        self.untracked |= later.untracked;
    }
}

/// The copies a command needs first: each stale buffer back from its image, then each stale image
/// in from its buffer.
#[derive(Default)]
pub(crate) struct Copies {
    pub(crate) back: Vec<OverCopy>,
    pub(crate) into: Vec<OverCopy>,
}

impl Copies {
    pub(crate) fn is_empty(&self) -> bool {
        self.back.is_empty() && self.into.is_empty()
    }
}

impl Overs {
    /// The copies a command that reaches `reached` needs first, in order: each texture over a
    /// buffer in an image of its own taking its image in where a command of the segment wrote
    /// memory its texels lie in since, and each buffer reached (a texture in place over one
    /// reaches it) taking back the images a command wrote since of the textures whose bytes meet
    /// its, every one where `unbound` (the command reaches memory through an address outside its
    /// resources). While no image is ahead of its buffer ([`ahead`](Self::ahead)), a buffer
    /// reached owes nothing.
    pub(crate) fn before<'a>(
        &mut self,
        reached: impl IntoIterator<Item = (Reached<'a>, Access)>,
        unbound: bool,
    ) -> Copies {
        let mut copies = Copies::default();
        for (reached, _) in reached {
            let span = match reached {
                Reached::Texture(texture) => match self.uses(texture, &mut copies) {
                    true => continue,
                    false => match texture.in_place_span() {
                        Some(span) => span,
                        None => continue,
                    },
                },
                Reached::Buffer(span) => span,
            };
            if self.ahead > 0 {
                self.backs(&mut copies.back, |used| {
                    unbound || used.copy.memory.meets(span)
                });
            }
        }
        if unbound && self.ahead > 0 {
            self.backs(&mut copies.back, |_| true);
        }
        copies
    }

    /// A command uses `texture`: where it is a texture over a buffer in an image of its own, first
    /// the copy back of each other such texture whose image over bytes this one holds a command
    /// wrote since they last went back, then its copy in where its image is stale, or, at its
    /// first use in the segment, where a command of the segment wrote memory its texels lie in;
    /// false for any other texture.
    fn uses(&mut self, texture: &Arc<TextureInner>, copies: &mut Copies) -> bool {
        if !texture.copied_shared() {
            return false;
        }
        let id = texture.id();
        let found = self.used.iter().rposition(|used| used.texture == id);
        if self.ahead > 0 {
            let memory = match found {
                Some(at) => self.used[at].copy.memory,
                None => match texture.copied_over() {
                    Some((_, copy)) => copy.memory,
                    None => return false,
                },
            };
            // The other image goes back to the memory, which the copy in reads: this texture's
            // image is stale since that write (`after`), or takes its first copy in after it.
            self.backs(&mut copies.back, |used| {
                used.texture != id && used.copy.memory.meets(memory)
            });
        }
        if let Some(at) = found {
            let used = &mut self.used[at];
            if used.image_stale {
                copies.into.push(used.copy);
                used.image_stale = false;
            }
            return true;
        }
        let Some((_, copy)) = texture.copied_over() else {
            return false;
        };
        let wrote = self.unbound
            || self.wrote.meets(copy.memory)
            || (self.backed.iter()).any(|&(span, _)| span.meets(copy.memory));
        if wrote {
            copies.into.push(copy);
        }
        self.under.insert(copy.memory.block);
        self.used.push(Used {
            texture: id,
            copy,
            leans: !wrote,
            image_stale: false,
            buffer_stale: false,
        });
        true
    }

    /// Each used texture's copy back, into `back`, where its image is ahead of its buffer and
    /// `reaches` holds of it.
    fn backs(&mut self, back: &mut Vec<OverCopy>, reaches: impl Fn(&Used) -> bool) {
        for used in &mut self.used {
            if used.buffer_stale && reaches(used) {
                back.push(used.copy);
                used.buffer_stale = false;
                self.ahead -= 1;
                let pair = (used.copy.memory, used.texture);
                if !self.backed.contains(&pair) {
                    self.backed.push(pair);
                }
            }
        }
    }

    /// After a command that reached `reached`: a texture it wrote is to go back to its buffer,
    /// and every other such texture whose bytes meet its is to take them in again; a buffer it
    /// wrote (a texture in place over one writes it) is to go into every texture whose bytes meet
    /// the buffer's, all where it wrote through an address outside its resources (`unbound`).
    pub(crate) fn after<'a>(
        &mut self,
        reached: impl IntoIterator<Item = (Reached<'a>, Access)>,
        unbound: bool,
    ) {
        for (reached, access) in reached {
            if !access.writes() {
                continue;
            }
            let span = match reached {
                Reached::Texture(texture) => {
                    let id = texture.id();
                    if texture.copied_shared() {
                        if let Some(at) = self.used.iter().rposition(|used| used.texture == id) {
                            let memory = self.used[at].copy.memory;
                            if !self.used[at].buffer_stale {
                                self.used[at].buffer_stale = true;
                                self.ahead += 1;
                            }
                            // The other images over these bytes hold them no longer.
                            for used in &mut self.used {
                                if used.texture != id && used.copy.memory.meets(memory) {
                                    used.image_stale = true;
                                }
                            }
                        }
                        continue;
                    }
                    // A texture of an image of its own writes no buffer.
                    match texture.in_place_span() {
                        Some(span) => span,
                        None => continue,
                    }
                }
                Reached::Buffer(span) => span,
            };
            if self.under.contains(&span.block) {
                for used in &mut self.used {
                    if used.copy.memory.meets(span) {
                        used.image_stale = true;
                    }
                }
            }
            self.wrote.insert(span);
        }
        if unbound {
            self.unbound = true;
            for used in &mut self.used {
                used.image_stale = true;
            }
        }
    }

    /// A command recorded while no such texture lived: its writes, untracked, count as a write
    /// through an address that reaches every block.
    pub(crate) fn untracked(&mut self) {
        self.after([], true);
        self.untracked = true;
    }

    /// Whether a texture the segment used has an image ahead of its buffer: a command wrote it
    /// since its bytes last went back. While none has, a command's reads of buffers owe nothing.
    pub(crate) fn ahead(&self) -> bool {
        self.ahead > 0
    }

    /// At the segment's end: every image a command wrote goes back to its buffer (the copies,
    /// which the caller records), and the textures kept are those that lean on the leading copy.
    pub(crate) fn end(&mut self) -> Vec<OverCopy> {
        let mut back = Vec::new();
        self.backs(&mut back, |_| true);
        self.used.retain(|used| used.leans);
        back
    }

    /// Of an ended segment: the copies of the textures whose images the leading copy fills.
    pub(crate) fn leaning(&self) -> impl Iterator<Item = OverCopy> + '_ {
        self.used.iter().map(|used| used.copy)
    }
}

/// Records `copies` into `command`, a command buffer's, after every earlier command and before
/// every later one: first each buffer's bytes back from its image, then each image's bytes in from
/// its buffer, each set between a pair of barriers; counted, the barriers in
/// `Metrics::other_barriers`. Whether a copy back wrote a Shared buffer, which the host reads, is
/// the answer.
///
/// # Safety
///
/// `command` is recording outside any render pass, and holds each image and buffer until it
/// retires; each image is in its home layout between commands.
pub(crate) unsafe fn record(
    device: &ash::Device,
    command: vk::CommandBuffer,
    copies: &Copies,
    counters: &Counters,
) -> bool {
    // SAFETY: the caller's contract.
    unsafe {
        let cause = Metric::OtherBarriers;
        record_side(device, command, &copies.back, false, cause, counters);
        record_side(device, command, &copies.into, true, cause, counters);
    }
    copies.back.iter().any(|copy| copy.shared)
}

/// Records `copies` into `command`: each buffer's bytes into its image (`into`), or back, between
/// one pair of barriers, counted in `cause`, one of the barrier causes. A copy into an image moves
/// it from the undefined layout, discarding what it held: its one region writes the image's whole
/// level and slice, of its only aspect, the colour one (`texture::over_texel` refuses any other
/// texture over a buffer). A copy back moves it from home, which keeps its texels.
///
/// # Safety
///
/// As [`record`].
pub(crate) unsafe fn record_side(
    device: &ash::Device,
    command: vk::CommandBuffer,
    copies: &[OverCopy],
    into: bool,
    cause: Metric,
    counters: &Counters,
) {
    use vk::AccessFlags2 as A;
    use vk::PipelineStageFlags2 as S;
    if copies.is_empty() {
        return;
    }
    let (layout, access) = match into {
        true => (vk::ImageLayout::TRANSFER_DST_OPTIMAL, A::TRANSFER_WRITE),
        false => (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, A::TRANSFER_READ),
    };
    let image = |copy: &OverCopy, old, new| {
        vk::ImageMemoryBarrier2::default()
            .old_layout(old)
            .new_layout(new)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(copy.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: copy.aspects,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
    };
    let from = |copy: &OverCopy| match into {
        true => vk::ImageLayout::UNDEFINED,
        false => copy.home,
    };
    let away: Vec<_> = (copies.iter())
        .map(|copy| {
            image(copy, from(copy), layout)
                .src_stage_mask(S::ALL_COMMANDS)
                .src_access_mask(A::MEMORY_WRITE)
                .dst_stage_mask(S::COPY)
                .dst_access_mask(access)
        })
        .collect();
    let back: Vec<_> = (copies.iter())
        .map(|copy| {
            image(copy, layout, copy.home)
                .src_stage_mask(S::COPY)
                .src_access_mask(A::TRANSFER_WRITE)
                .dst_stage_mask(S::ALL_COMMANDS)
                .dst_access_mask(A::MEMORY_READ | A::MEMORY_WRITE)
        })
        .collect();
    // Every earlier command runs first, its writes visible to the copies and to every later
    // command, as a blit's first barrier makes them; the copies' writes are visible to every
    // later command.
    let earlier = [memory_barrier(Reach::Commands)];
    let later = [vk::MemoryBarrier2::default()
        .src_stage_mask(S::COPY)
        .src_access_mask(A::TRANSFER_WRITE)
        .dst_stage_mask(S::ALL_COMMANDS)
        .dst_access_mask(A::MEMORY_READ | A::MEMORY_WRITE)];
    // SAFETY: the caller's contract; each region lies in its buffer and image, checked at the
    // texture's creation.
    unsafe {
        device.cmd_pipeline_barrier2(
            command,
            &vk::DependencyInfo::default()
                .memory_barriers(&earlier)
                .image_memory_barriers(&away),
        );
        for copy in copies {
            match into {
                true => device.cmd_copy_buffer_to_image(
                    command,
                    copy.buffer,
                    copy.image,
                    layout,
                    &[copy.region],
                ),
                false => device.cmd_copy_image_to_buffer(
                    command,
                    copy.image,
                    layout,
                    copy.buffer,
                    copy_out::regions(&copy.unit, copy.region).as_slice(),
                ),
            }
            counters.over_copied(copy.bytes, copy.imported, copy.class, copy.kind);
        }
        device.cmd_pipeline_barrier2(
            command,
            &vk::DependencyInfo::default()
                .memory_barriers(&later)
                .image_memory_barriers(&back),
        );
    }
    counters.count(cause);
    counters.count(cause);
}

/// A copy back a reader of a Managed buffer's device copy takes ([`ManagedOvers::backs`]): the
/// texture's id, its copy and its image, which the reader's command buffer or upload keeps.
pub(crate) type Back = (u32, OverCopy, Arc<Image>);

/// A texture over a Managed buffer's device copy in an image of its own, as its buffer keeps it
/// ([`ManagedOvers`]): the copies Metal's Managed events owe it.
struct Entry {
    /// The texture's id.
    texture: u32,
    /// Its image, held while the entry lives.
    image: Arc<Image>,
    copy: OverCopy,
    /// Its bytes in the buffer: `start..end`.
    range: (u64, u64),
    /// An event named its range since its last copy in: its making, or submitted device work that
    /// wrote the buffer or another texture over a range that meets its.
    into: bool,
    /// The commit number of the first capture that copied it in while `into` held: that copy in
    /// runs ahead of every command buffer numbered at or after it (`crate::capture`), which then
    /// takes none, while one numbered before it may run first and still takes its own.
    captured: Option<u64>,
    /// Submitted device work wrote it since its last copy back: the upload that leads a later
    /// submission takes the copy.
    back: bool,
    /// The segments that wrote it and are not yet submitted. The upload that leads a submission
    /// runs before them, so it takes no copy back for their writes; each one's submission makes
    /// the copy back owed to it, unless a later command of its own command buffer took it. One
    /// whose command buffer is never committed leaves the entry until the buffer is released.
    pending: u32,
    /// The texture is released: the entry lives for its copy back alone, and its id may name a
    /// texture made since, which a lookup by id finds instead.
    released: bool,
}

impl Entry {
    /// Whether it is live texture `texture`'s entry.
    fn of(&self, texture: u32) -> bool {
        self.texture == texture && !self.released
    }
}

/// A Managed buffer's textures over its device copy in an image of their own (`Entry`), each owing
/// copies only at Metal's own Managed events, with no tracking of commands:
///
/// - **In** after a didModifyRange over its range: the capture of the range at the next commit
///   (`crate::capture`) makes it, after its copies into the device copy, in the same command
///   buffer. And at its making, or where device work wrote the buffer (a binding whose compiled
///   interface writes it, a blit or fill into it) or another texture over a range that meets its,
///   from the submission after that work's: the upload that leads the first submission that uses
///   the texture makes it, taken as that submission goes, as the device's one `VkQueue` runs
///   submissions in the order they go, so a command buffer of another queue that passes a group
///   held takes it ahead of the group. A capture over its range settles that copy only for the
///   command buffers committed at or after the capture, which run behind its copy in; one
///   committed before it may run first, and takes its own. Both copy from the device copy, never
///   from the consumer's bytes a second time. Apple GPUs may alias Managed memory; the documented
///   contract, not one GPU's answer, is the rule here.
/// - **Back** where device work wrote the texture since the last copy back: before the next
///   reader of the device copy (a binding whose interface reads the buffer, a blit from it, a copy
///   into another texture over a range that meets its, a synchronize, a capture of a range over
///   it), not after each write. A reader in the write's own command buffer, recorded after it,
///   takes it in its own commands. Any other reader runs after the write only where the write was
///   submitted first, Metal running a queue's command buffers in the order they are committed,
///   whichever was encoded first: the upload that leads the reader's submission takes it once the
///   write's segment was submitted, as the upload runs before the submission's commands, and a
///   reader whose group held holds the write ahead of it leads its own group, the write submitted
///   first (`crate::submit`). A capture's is taken by the upload that leads
///   the submission that takes its copy-in, a write in the same group held going first in a
///   submission of its own (`crate::capture`). A texture released while its copy back is owed
///   leaves its image with the buffer until a reader takes it or the buffer is released, which
///   drops it uncopied.
#[derive(Default)]
pub(crate) struct ManagedOvers {
    entries: Mutex<Vec<Entry>>,
    /// The entries' count, read without the lock.
    count: AtomicUsize,
}

/// A texture over a buffer in an exported block in an image of its own, as the submissions keep it
/// ([`ExportedOvers`]).
struct Exported {
    /// The texture, whose image is `image` while it lives.
    texture: Weak<TextureInner>,
    id: u32,
    image: vk::Image,
    memory: Span,
    /// Whether a public call handed out an address of its block's memory
    /// (`crate::memory::Block::address_out`): only then does a write through an address reach it.
    addressed: Arc<AtomicBool>,
    /// An event changed its bytes since its image last took them: the commit number from which
    /// every command buffer runs behind that event (0 for a submitted write).
    owed: Option<u64>,
}

/// The textures over a buffer in an exported block ([`crate::Device::new_exported_block`]) in an
/// image of their own. No host write reaches such a block, which is Private: only the device's
/// commands and the other API, whose writes [`crate::Block::external_writes_done`] announces,
/// change its bytes. So such a texture is copied in once after each event that changes its bytes,
/// by the upload that leads the next submission that leans on the leading copy for it, and a
/// submission that only reads it copies nothing. The events: the texture's making; an acquire or a
/// zero fill of the block, which go ahead of every command buffer committed after them; and a
/// submitted command's write of memory its texels lie in, a write through an address once an
/// address of the block went out (every such texture), or another texture's copy back over those
/// bytes. An address goes out by a public call alone: [`crate::Buffer::gpu_address`] of a buffer
/// whose bytes shaders reach in the block, and a texture buffer's storage element at an index over such bytes. A surface's block in the copy form is
/// the surface's own, so its address goes out only through its buffer
/// ([`crate::Surface::buffer`]). Within a segment the Shared rules stand.
#[derive(Default)]
pub(crate) struct ExportedOvers {
    entries: Vec<Exported>,
}

impl ExportedOvers {
    /// Keeps `texture`, whose copy is `copy` and whose block's flag is `addressed`, owing its
    /// first copy in.
    pub(crate) fn add(
        &mut self,
        texture: &Arc<TextureInner>,
        copy: &OverCopy,
        addressed: Arc<AtomicBool>,
    ) {
        self.entries
            .retain(|entry| entry.texture.strong_count() > 0);
        self.entries.push(Exported {
            texture: Arc::downgrade(texture),
            id: texture.id(),
            image: copy.image,
            memory: copy.memory,
            addressed,
            owed: Some(0),
        });
    }

    /// Work ahead of every command buffer numbered `number` or later changed the bytes `range`
    /// of block `block`: each texture over them owes its copy in.
    pub(crate) fn owe(&mut self, block: BlockId, range: (u64, u64), number: u64) {
        let span = Span {
            block,
            start: range.0,
            end: range.1,
        };
        for entry in &mut self.entries {
            if entry.memory.meets(span) {
                entry.owed = Some(entry.owed.map_or(number, |owed| owed.max(number)));
            }
        }
    }

    /// Whether the submission whose last command buffer is numbered `through` copies `copy` in:
    /// where an event owes it, settled where every such event runs ahead of that command buffer.
    /// A texture over another block's buffer takes its copy, as the Shared rules have it.
    pub(crate) fn takes(&mut self, copy: &OverCopy, through: u64) -> bool {
        let found = (self.entries.iter_mut())
            .find(|entry| entry.image == copy.image && entry.texture.strong_count() > 0);
        let Some(entry) = found else {
            return true;
        };
        let Some(owed) = entry.owed else {
            return false;
        };
        if owed <= through {
            entry.owed = None;
        }
        true
    }

    /// A submission went whose segments wrote `leans`: each texture whose bytes they changed owes
    /// its copy in to every later submission. A write through an address reaches the texture's
    /// bytes only where an address of its block went out before this submission; the flag read
    /// here covers every call made before the commits it holds, and Metal leaves undefined an
    /// address taken after a commit and written for that command buffer to read.
    pub(crate) fn wrote(&mut self, leans: &Leans) {
        self.entries
            .retain(|entry| entry.texture.strong_count() > 0);
        for entry in &mut self.entries {
            let memory = entry.memory;
            let changed = leans.untracked
                || (leans.unbound && entry.addressed.load(Relaxed))
                || leans.wrote.meets(memory)
                || (leans.backed.iter()).any(|&(span, by)| span.meets(memory) && by != entry.id);
            if changed {
                entry.owed = Some(entry.owed.unwrap_or(0));
            }
        }
    }
}

/// Whether two ranges meet.
fn meets(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

/// Whether `range` meets one of `ranges`.
fn meets_any(ranges: &[(u64, u64)], range: (u64, u64)) -> bool {
    ranges.iter().any(|&other| meets(other, range))
}

impl ManagedOvers {
    /// Whether the buffer has an entry.
    pub(crate) fn any(&self) -> bool {
        self.count.load(Relaxed) > 0
    }

    /// Keeps texture `texture`'s entry: its image, its copy and its bytes' `range` in the buffer,
    /// owing its first copy in.
    pub(crate) fn add(&self, texture: u32, image: Arc<Image>, copy: OverCopy, range: (u64, u64)) {
        let mut entries = lock(&self.entries);
        entries.push(Entry {
            texture,
            image,
            copy,
            range,
            into: true,
            captured: None,
            back: false,
            pending: 0,
            released: false,
        });
        self.count.store(entries.len(), Relaxed);
    }

    /// Texture `texture` is released: its entry goes, but for a copy back owed.
    pub(crate) fn release(&self, texture: u32) {
        let mut entries = lock(&self.entries);
        entries.retain_mut(|entry| {
            if entry.of(texture) {
                entry.released = true;
            }
            !entry.released || entry.back || entry.pending > 0
        });
        self.count.store(entries.len(), Relaxed);
    }

    /// A segment not yet submitted wrote texture `texture` for the first time, which its
    /// submission settles ([`ManagedOvers::settle`]).
    pub(crate) fn pend(&self, texture: u32) {
        if let Some(entry) = lock(&self.entries).iter_mut().find(|e| e.of(texture)) {
            entry.pending += 1;
        }
    }

    /// A segment that wrote texture `texture` was submitted: the texture owes its copy back where
    /// `owes`, no later command of its command buffer having taken it, and each other texture
    /// whose range meets its owes its copy in, which takes that copy back first.
    pub(crate) fn settle(&self, texture: u32, owes: bool) {
        let mut entries = lock(&self.entries);
        let Some(range) = (entries.iter_mut())
            .find(|entry| entry.of(texture))
            .map(|entry| {
                entry.pending = entry.pending.saturating_sub(1);
                entry.back |= owes;
                entry.range
            })
        else {
            return;
        };
        for entry in entries.iter_mut() {
            if !entry.of(texture) && meets(entry.range, range) {
                entry.into = true;
                entry.captured = None;
            }
        }
    }

    /// Device work wrote the buffer: each texture owes its copy in.
    pub(crate) fn written(&self) {
        for entry in lock(&self.entries).iter_mut() {
            entry.into = true;
            entry.captured = None;
        }
    }

    /// The upload that leads a submission that reads the device copy over `range` of the buffer
    /// (all of it where `None`): each copy back owed by a submitted write of a texture whose range
    /// meets it, taken, into `backs`. A released texture's entry goes once nothing is owed.
    pub(crate) fn backs(&self, range: Option<(u64, u64)>, backs: &mut Vec<Back>) {
        let mut entries = lock(&self.entries);
        for entry in entries.iter_mut() {
            if entry.back && range.is_none_or(|range| meets(entry.range, range)) {
                backs.push((entry.texture, entry.copy, Arc::clone(&entry.image)));
                entry.back = false;
            }
        }
        entries.retain(|entry| !entry.released || entry.back || entry.pending > 0);
        self.count.store(entries.len(), Relaxed);
    }

    /// A command of the command buffer that wrote `textures` and reads the device copy after
    /// those writes: their copies back, into `backs`, for its own commands to make first. The
    /// copies back owed by other command buffers' writes wait for its submission
    /// ([`backs`](Self::backs)), where the order of the two is known.
    pub(crate) fn own_backs(&self, textures: &[u32], backs: &mut Vec<Back>) {
        let entries = lock(&self.entries);
        for entry in entries
            .iter()
            .filter(|entry| textures.iter().any(|&t| entry.of(t)))
        {
            backs.push((entry.texture, entry.copy, Arc::clone(&entry.image)));
        }
    }

    /// A submission takes the copy-in of a capture of `ranges` of the consumer's bytes into the
    /// device copy (`crate::capture`): the copies back owed by writes submitted before it of
    /// textures whose ranges meet one of `ranges`, or meet a live texture's that does, taken into
    /// `backs`, for the upload that leads the submission to make ahead of the copy-in's copies, so
    /// the consumer's bytes land over them.
    pub(crate) fn backs_before(&self, ranges: &[(u64, u64)], backs: &mut Vec<Back>) {
        let mut entries = lock(&self.entries);
        if !entries.iter().any(|entry| entry.back) {
            return;
        }
        let targets: Vec<(u64, u64)> = (entries.iter())
            .filter(|entry| !entry.released && meets_any(ranges, entry.range))
            .map(|entry| entry.range)
            .collect();
        for entry in entries.iter_mut() {
            if entry.back && (meets_any(ranges, entry.range) || meets_any(&targets, entry.range)) {
                backs.push((entry.texture, entry.copy, Arc::clone(&entry.image)));
                entry.back = false;
            }
        }
        entries.retain(|entry| !entry.released || entry.back || entry.pending > 0);
        self.count.store(entries.len(), Relaxed);
    }

    /// A capture copied `ranges` of the consumer's bytes into the device copy (`crate::capture`),
    /// at the commit numbered `number`: each live texture whose range meets one of them takes its
    /// copy in after those copies, in the same command buffer, its copy and image going to
    /// `into`. A copy in it owed for an event before stays owed to the command buffers numbered
    /// before `number`, which may run ahead of the capture's ([`take_into`](Self::take_into)).
    pub(crate) fn copies_in_after(
        &self,
        ranges: &[(u64, u64)],
        number: u64,
        into: &mut Vec<(OverCopy, Arc<Image>)>,
    ) {
        for entry in lock(&self.entries).iter_mut() {
            if !entry.released && meets_any(ranges, entry.range) {
                if entry.into {
                    entry.captured.get_or_insert(number);
                }
                into.push((entry.copy, Arc::clone(&entry.image)));
            }
        }
    }

    /// The copy in texture `texture` owes, taken, for the upload that leads the submission of a
    /// command buffer numbered `number` that reads it, as that submission goes: after the copies
    /// back owed by submitted writes of textures whose ranges meet its, its own among them, into
    /// `backs`. Taken as a submission goes, not as a command buffer joins a group held, so the
    /// first submission on the device's one `VkQueue` that reads the texture makes it, of
    /// whichever queue. None where it owes none, or where a capture numbered at or before
    /// `number` copies it in ahead of that command buffer, which settles it
    /// ([`copies_in_after`](Self::copies_in_after)).
    pub(crate) fn take_into(
        &self,
        texture: u32,
        number: u64,
        backs: &mut Vec<Back>,
    ) -> Option<OverCopy> {
        let (range, copy) = {
            let mut entries = lock(&self.entries);
            let entry = entries.iter_mut().find(|entry| entry.of(texture))?;
            if !entry.into {
                return None;
            }
            let captured = entry.captured.is_some_and(|at| at <= number);
            entry.into = false;
            entry.captured = None;
            if captured {
                return None;
            }
            (entry.range, entry.copy)
        };
        self.backs(Some(range), backs);
        Some(copy)
    }
}
