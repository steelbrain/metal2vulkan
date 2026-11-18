//! Textures over a buffer in an image of their own: where the device's linear image does not hold
//! a texture over a buffer of no device copy in place
//! ([`Buffer::new_texture`](crate::Buffer::new_texture)), its texels live in an optimal image of
//! the runtime's, and the device copies the buffer's bytes into it and back, one
//! `vkCmdCopyBufferToImage` or `vkCmdCopyImageToBuffer` region a texture, with no CPU copy:
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
//!   be copied back: before the next command of the segment that reaches the buffer, or that
//!   reaches memory through an address outside its resources, and at the segment's end, whose
//!   end barrier then reaches the host for a Shared buffer. A segment's end is a submission's
//!   edge, so the next submission's leading copy reads what the image held.
//!
//! Each copy is counted (`Metrics::over_copies`, and its bytes in `Metrics::bytes_copied`). The
//! tracking is by buffer: a write of any of a buffer's bytes makes every such texture over it
//! stale.

use crate::counters::{Counters, Metric};
use crate::hazard::Access;
use crate::record::{memory_barrier, Reach};
use crate::texture::{OverCopy, Texture};
use ash::vk;

/// What a segment does to its textures over a buffer in an image of their own. Once it has
/// ended ([`Overs::end`]), what the submit thread reads of it: the textures whose images the
/// leading copy fills, and the buffers it wrote.
#[derive(Default)]
pub(crate) struct Overs {
    /// The textures it used; once it has ended, those that lean on the leading copy alone.
    used: Vec<Used>,
    /// The buffers the segment's commands wrote, by id, each with the texture whose copy back
    /// wrote it, `None` for any other write.
    written: Vec<(u32, Option<u32>)>,
    /// A command of the segment wrote through an address outside its resources, or was recorded
    /// while no such texture lived, its writes untracked.
    unbound: bool,
}

/// A texture over a buffer in an image of its own that a segment used.
struct Used {
    texture: Texture,
    /// Its buffer's id.
    buffer: u32,
    /// Its first use in the segment came before any write of the buffer in it: the image holds
    /// the bytes of the submission's leading copy.
    leans: bool,
    /// A command wrote the buffer after the image last took its bytes.
    image_stale: bool,
    /// A command wrote the image after its bytes last went back.
    buffer_stale: bool,
}

/// What a held group's ended segments wrote, as [`Overs`] keeps it.
#[derive(Default)]
pub(crate) struct Leans {
    written: Vec<(u32, Option<u32>)>,
    unbound: bool,
}

impl Leans {
    /// Whether `later`, an ended segment after these in one submission, leans on the leading copy
    /// for a buffer these wrote: where so, it leads a new group.
    pub(crate) fn conflicts(&self, later: &Overs) -> bool {
        (later.used.iter()).any(|used| {
            let id = used.texture.id();
            self.unbound
                || (self.written.iter())
                    .any(|&(written, by)| written == used.buffer && by != Some(id))
        })
    }

    /// Adds the writes of `later`, an ended segment.
    pub(crate) fn join(&mut self, later: &Overs) {
        self.written.extend_from_slice(&later.written);
        self.unbound |= later.unbound;
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
    /// The copies a command that makes `accesses` (by hazard id) needs first, its textures over a
    /// buffer in an image of their own found among `kept` by id: each such texture's image in
    /// where a command of the segment wrote its buffer since, and each buffer the command reaches
    /// back where a command wrote a texture over it since, every one where `unbound` (the command
    /// reaches memory through an address outside its resources).
    pub(crate) fn before(
        &mut self,
        accesses: &[(u32, Access)],
        kept: &[Texture],
        unbound: bool,
    ) -> Copies {
        let mut copies = Copies::default();
        for &(id, _) in accesses {
            if let Some(used) = self.used.iter_mut().find(|used| used.texture.id() == id) {
                if used.image_stale {
                    copies
                        .into
                        .extend(used.texture.copied_over().map(|(_, copy)| copy));
                    used.image_stale = false;
                }
                continue;
            }
            let found = (kept.iter().rev())
                .find(|texture| texture.id() == id)
                .and_then(|texture| Some((texture, texture.copied_over()?)));
            if let Some((texture, (buffer, copy))) = found {
                let buffer = buffer.id();
                let wrote = self.unbound || self.written.iter().any(|&(b, _)| b == buffer);
                if wrote {
                    copies.into.push(copy);
                }
                self.used.push(Used {
                    texture: texture.clone(),
                    buffer,
                    leans: !wrote,
                    image_stale: false,
                    buffer_stale: false,
                });
                continue;
            }
            for used in &mut self.used {
                if used.buffer_stale && (unbound || used.buffer == id) {
                    copies
                        .back
                        .extend(used.texture.copied_over().map(|(_, copy)| copy));
                    used.buffer_stale = false;
                    self.written.push((used.buffer, Some(used.texture.id())));
                }
            }
        }
        if unbound {
            for used in &mut self.used {
                if used.buffer_stale {
                    copies
                        .back
                        .extend(used.texture.copied_over().map(|(_, copy)| copy));
                    used.buffer_stale = false;
                    self.written.push((used.buffer, Some(used.texture.id())));
                }
            }
        }
        copies
    }

    /// After a command that made `accesses`: a texture it wrote is to go back to its buffer, and
    /// a buffer it wrote is to go into every texture over it, all where it wrote through an
    /// address outside its resources (`unbound`).
    pub(crate) fn after(&mut self, accesses: &[(u32, Access)], unbound: bool) {
        for &(id, access) in accesses {
            if !access.writes() {
                continue;
            }
            if let Some(used) = self.used.iter_mut().find(|used| used.texture.id() == id) {
                used.buffer_stale = true;
                continue;
            }
            let mut over = false;
            for used in &mut self.used {
                if used.buffer == id {
                    used.image_stale = true;
                    over = true;
                }
            }
            if over || !self.written.iter().any(|&(b, by)| b == id && by.is_none()) {
                self.written.push((id, None));
            }
        }
        if unbound {
            self.unbound = true;
            for used in &mut self.used {
                used.image_stale = true;
            }
        }
    }

    /// At the segment's end: every image a command wrote goes back to its buffer (the copies,
    /// which the caller records), and the textures kept are those that lean on the leading copy.
    pub(crate) fn end(&mut self) -> Vec<OverCopy> {
        let mut back = Vec::new();
        for used in &mut self.used {
            if used.buffer_stale {
                back.extend(used.texture.copied_over().map(|(_, copy)| copy));
                used.buffer_stale = false;
                self.written.push((used.buffer, Some(used.texture.id())));
            }
        }
        self.used.retain(|used| used.leans);
        back
    }

    /// Of an ended segment: the textures whose images the leading copy fills.
    pub(crate) fn leaning(&self) -> impl Iterator<Item = &Texture> {
        self.used.iter().map(|used| &used.texture)
    }
}

/// Records `copies` into `command`, after every earlier command and before every later one:
/// first each buffer's bytes back from its image, then each image's bytes in from its buffer, each
/// set between a pair of barriers; counted. Whether a copy back wrote a Shared buffer, which the
/// host reads, is the answer.
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
        record_side(device, command, &copies.back, false, counters);
        record_side(device, command, &copies.into, true, counters);
    }
    copies.back.iter().any(|copy| copy.shared)
}

/// Records `copies` into `command`: each buffer's bytes into its image (`into`), or back, between
/// one pair of barriers.
///
/// # Safety
///
/// As [`record`].
pub(crate) unsafe fn record_side(
    device: &ash::Device,
    command: vk::CommandBuffer,
    copies: &[OverCopy],
    into: bool,
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
    let away: Vec<_> = (copies.iter())
        .map(|copy| {
            image(copy, copy.home, layout)
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
                    &[copy.region],
                ),
            }
            counters.count(Metric::OverCopies);
            counters.copied(copy.bytes);
        }
        device.cmd_pipeline_barrier2(
            command,
            &vk::DependencyInfo::default()
                .memory_barriers(&later)
                .image_memory_barriers(&back),
        );
    }
    counters.count(Metric::Barriers);
    counters.count(Metric::Barriers);
}
