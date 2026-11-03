//! What a draw's stages before its fragments read of the bytes the pass's fragment stages wrote,
//! and what they write of the bytes those read or wrote: Metal's tile-based passes run every draw's
//! vertex, object and mesh functions before any draw's fragments.
//!
//! - **Metal's answer.** a probe drew, in one render pass, a draw whose fragments write a buffer,
//!   then draws whose vertex function reads the bytes written: on an M3 Max and an M4, plain and
//!   validated, 20 runs of each of 12 cases (a pixel, a tile, a row of tiles, the whole target; one
//!   writer or four; draws between), the vertex stage read the bytes from before the pass in every
//!   run, and a draw of a later pass read the written bytes. A memory barrier from the fragment
//!   stage to the vertex stage is refused by Metal's validation ("afterStages ... can not contain
//!   MTLRenderStageTile or MTLRenderStageFragment on this device"), and plain Metal ignores it: the
//!   vertex stage reads the bytes from before the pass there too. probes (plain and
//!   validated) found the same of an object or mesh function's read, and of a command of an
//!   indirect command buffer behind its barrier (`setBarrier`): its vertex, object and mesh
//!   functions read the bytes from before the earlier commands' fragments, and the encoder's draws'
//!   before the execution. The order holds the other way too: an earlier draw's fragments read what
//!   a later draw's vertex function wrote, and their write outlasts the later one's, behind the
//!   command's barrier or not.
//! - **The runtime's answer.** Vulkan orders nothing between two draws of a render pass instance,
//!   so a later draw's stages there could read the write, and a split of the pass (a command's
//!   barrier, a memory barrier, a draw of another shape) runs the earlier fragments first. A
//!   buffer bound in a draw's vertex, object or mesh table whose bytes from the bind offset a
//!   fragment stage of the pass wrote, which no other stage of the pass wrote and the stage's
//!   function does not write, is read from a copy of those bytes taken before the pass, as the
//!   indexed indirect draws read theirs (`draw_source.rs`): one copy per buffer and range, which
//!   every later draw of the pass whose bind lies inside it reads, at the bind's offset in it. An
//!   execution of render commands takes its copies before the pass too, into the recording that
//!   replays it (`render/executions.rs`), and starts from what the encoder's draws before the call
//!   noted. The copy is announced once as `Degradation::UnorderedDrawBytes`. The pass's splits
//!   stand as before; the fragment stages' writes stay noted across them, as plain Metal ignores a
//!   barrier after the fragment stage and a command's barrier does not order them.
//! - **Read in place** is every other read of such bytes by those stages: through an argument
//!   buffer or a resource made resident, or of a buffer another stage of the pass wrote or the
//!   stage's function writes. A vertex stage's is announced once as
//!   `Degradation::InPlaceVertexReads`, an object or mesh stage's (an emulated one's where its
//!   work records after the pass's earlier instances) as `Degradation::FragmentsBeforeGeometry`:
//!   the device may order the fragments' write first.
//! - **A write of those stages** to bytes a fragment stage of the pass read or wrote is announced
//!   once as `Degradation::FragmentsBeforeGeometry`: the device may run the earlier fragments
//!   first, where Metal's read the later write and their own write outlasts it. Running a later
//!   draw's stages before an earlier draw's fragments has no Vulkan form short of running the
//!   pass's stages before its fragments as work of their own.

use crate::argument_table::ArgumentTable;
use crate::blit::{reads, writes, TRANSFER};
use crate::buffer::Buffer;
use crate::degradation::Degradation;
use crate::encoder::EncoderCore;
use crate::error::Error;
use crate::resource::ResourceKey;
use crate::tracker::{Barrier, Use, UseRange};
use crate::values::{CpuCacheMode, HazardTrackingMode, ResourceOptions, StorageMode};
use crate::Device;
use ash::vk;
use metal2vulkan::pipeline::{PipelineInterface, SlotClass, SlotInterface, Stage};
use std::ops::Range;

/// The byte ranges one key holds before they are merged into their hull.
const RANGES_PER_KEY: usize = 16;

/// The stages a vertex stage reads in.
const VERTEX_READS: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::VERTEX_SHADER.as_raw()
        | vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT.as_raw(),
);

/// The stages a native object or mesh stage reads and writes in.
const MESH_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::TASK_SHADER_EXT.as_raw()
        | vk::PipelineStageFlags2::MESH_SHADER_EXT.as_raw(),
);

/// Byte ranges of buffers, tracked or not, by key, in its order: the first `held` lists, each
/// with its ranges; those past them are lists left empty with their room
/// ([`clear`](Self::clear)), which the next keys take.
#[derive(Clone, Default)]
struct Written {
    lists: Vec<(ResourceKey, Vec<Range<u64>>)>,
    held: usize,
}

impl Written {
    fn add(&mut self, key: ResourceKey, range: &Range<u64>) {
        let held = self.held;
        let at = match self.lists[..held].binary_search_by_key(&key, |&(at, _)| at) {
            Ok(at) => at,
            Err(at) => {
                match self.lists.get_mut(held) {
                    Some(left) => left.0 = key,
                    None => self.lists.push((key, Vec::new())),
                }
                self.lists[at..=held].rotate_right(1);
                self.held += 1;
                at
            }
        };
        let ranges = &mut self.lists[at].1;
        // A range held already adds nothing: a pass's draws mostly name the same bytes again.
        if ranges.contains(range) {
            return;
        }
        ranges.push(range.clone());
        if ranges.len() > RANGES_PER_KEY {
            let start = ranges.iter().map(|r| r.start).min().unwrap_or(0);
            let end = ranges.iter().map(|r| r.end).max().unwrap_or(0);
            ranges.clear();
            ranges.push(start..end);
        }
    }

    fn overlaps(&self, key: ResourceKey, range: &Range<u64>) -> bool {
        let held = &self.lists[..self.held];
        let at = held.binary_search_by_key(&key, |&(at, _)| at).ok();
        at.is_some_and(|at| {
            held[at]
                .1
                .iter()
                .any(|r| r.start < range.end && range.start < r.end)
        })
    }

    fn is_empty(&self) -> bool {
        self.held == 0
    }

    /// No ranges, its lists' room kept.
    fn clear(&mut self) {
        for (_, ranges) in &mut self.lists[..self.held] {
            ranges.clear();
        }
        self.held = 0;
    }
}

/// The bytes of `used`, where it uses a buffer's.
fn bytes<K>(used: &Use<K>) -> Option<&Range<u64>> {
    match &used.range {
        UseRange::Bytes(bytes) if !bytes.is_empty() => Some(bytes),
        _ => None,
    }
}

/// A copy of a buffer's bytes from `from` to its end, taken before the pass.
struct PassStartCopy {
    /// The buffer's `VkBuffer` and its byte 0 there: which buffer it is.
    source: (vk::Buffer, u64),
    from: u64,
    scratch: Buffer,
}

/// The pass's fragment-stage writes and the copies its stages before the fragments read (the
/// module docs).
#[derive(Default)]
pub(super) struct PassStartReads {
    /// The bytes the pass's fragment stages wrote.
    fragment: Written,
    /// The bytes the pass's fragment stages read.
    fragment_reads: Written,
    /// The bytes the pass's other stages wrote.
    other: Written,
    copies: Vec<PassStartCopy>,
}

/// What a pass's reads note ([`PassStartReads::held`]): its lists' keys and ranges, and its
/// copies' count.
#[cfg(debug_assertions)]
#[derive(PartialEq)]
pub(super) struct HeldReads([Vec<(ResourceKey, Vec<Range<u64>>)>; 3], usize);

/// The tables a draw's stages before its fragments read in place of the encoder's: each with the
/// copies of bytes the pass's fragment stages wrote, `None` where the stage reads the encoder's.
#[derive(Default)]
pub(super) struct PassStartTables {
    vertex: Option<Box<ArgumentTable>>,
    object: Option<Box<ArgumentTable>>,
    mesh: Option<Box<ArgumentTable>>,
}

impl PassStartTables {
    /// The table `stage` reads: the one with the copies, else `encoder`'s.
    pub(super) fn or<'a>(&'a self, stage: Stage, encoder: &'a ArgumentTable) -> &'a ArgumentTable {
        let table = match stage {
            Stage::Vertex => self.vertex.as_deref(),
            Stage::Object => self.object.as_deref(),
            Stage::Mesh => self.mesh.as_deref(),
            _ => None,
        };
        table.unwrap_or(encoder)
    }

    /// Whether every stage reads the encoder's table: no copy replaces a buffer of any.
    pub(super) fn reads_encoders(&self) -> bool {
        self.vertex.is_none() && self.object.is_none() && self.mesh.is_none()
    }
}

impl PassStartReads {
    /// Note the accesses of a draw or tile dispatch of the pass with `uses`.
    pub(super) fn add(&mut self, uses: &[Use<ResourceKey>]) {
        for used in uses {
            let Some(range) = bytes(used) else { continue };
            let stages = used.write.stages;
            if stages.contains(vk::PipelineStageFlags2::FRAGMENT_SHADER) {
                self.fragment.add(used.key, range);
            }
            if !(stages & !vk::PipelineStageFlags2::FRAGMENT_SHADER).is_empty() {
                self.other.add(used.key, range);
            }
            if used
                .read
                .stages
                .contains(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            {
                self.fragment_reads.add(used.key, range);
            }
        }
    }

    /// What it notes, without its copies' buffers (builds with debug assertions:
    /// `render/draw_again.rs`'s check).
    #[cfg(debug_assertions)]
    pub(super) fn held(&self) -> HeldReads {
        let held = |written: &Written| written.lists[..written.held].to_vec();
        HeldReads(
            [&self.fragment, &self.fragment_reads, &self.other].map(held),
            self.copies.len(),
        )
    }

    /// The entries its lists and copies hold room for.
    pub(super) fn room(&self) -> usize {
        let maps = [&self.fragment, &self.fragment_reads, &self.other];
        maps.iter().map(|map| map.lists.len()).sum::<usize>() + self.copies.capacity()
    }

    /// Note nothing, keeping its lists' room for a next pass's
    /// ([`DrawRoom`](super::draw_room::DrawRoom)). Its copies go: their scratch buffers are held
    /// by the command buffer of the pass that took them.
    pub(super) fn clear(&mut self) {
        self.fragment.clear();
        self.fragment_reads.clear();
        self.other.clear();
        self.copies.clear();
    }

    /// Whether a fragment stage of the pass wrote any bytes so far: a later draw's stages before
    /// its fragments read none from a copy, wherever they read.
    pub(super) fn fragment_wrote(&self) -> bool {
        !self.fragment.is_empty()
    }

    /// What the pass noted so far, without its copies: what an execution's recording starts from
    /// (`render/executions.rs`), whose copies are its own.
    pub(super) fn notes(&self) -> PassStartReads {
        PassStartReads {
            fragment: self.fragment.clone(),
            fragment_reads: self.fragment_reads.clone(),
            other: self.other.clone(),
            copies: Vec::new(),
        }
    }

    /// Whether a fragment stage of the pass wrote memory: else every stage of its draws reads the
    /// encoder's tables ([`tables`](Self::tables)).
    pub(super) fn fragments_wrote(&self) -> bool {
        !self.fragment.is_empty()
    }

    /// The tables a draw of `interface`'s pipeline reads in place of the encoder's `vertex`,
    /// `object` and `mesh`, each as `table` makes it.
    pub(super) fn tables(
        &mut self,
        core: &mut EncoderCore,
        [vertex, object, mesh]: [&ArgumentTable; 3],
        interface: &PipelineInterface,
    ) -> Result<PassStartTables, Error> {
        // Each table held apart, as a draw that reads none (most) moves them with nothing in them.
        let table = |table: Option<ArgumentTable>| table.map(Box::new);
        Ok(PassStartTables {
            vertex: table(self.table(core, Stage::Vertex, vertex, interface)?),
            object: table(self.table(core, Stage::Object, object, interface)?),
            mesh: table(self.table(core, Stage::Mesh, mesh, interface)?),
        })
    }

    /// The table of `stage` a draw of `interface`'s pipeline reads in place of `encoder`'s,
    /// `None` where it reads `encoder`'s: each buffer whose bytes from its offset a fragment stage
    /// of the pass wrote, which no other stage of the pass wrote and the stage's function does
    /// not write, replaced by the copy of its bytes taken before the pass (the module docs). A
    /// copy is recorded into `core` before the pass, held until its command buffer completes, and
    /// announced once. Refused as `new_buffer` refuses the copy's buffer.
    fn table(
        &mut self,
        core: &mut EncoderCore,
        stage: Stage,
        encoder: &ArgumentTable,
        interface: &PipelineInterface,
    ) -> Result<Option<ArgumentTable>, Error> {
        if self.fragment.is_empty() || encoder.buffers.is_empty() {
            return Ok(None);
        }
        // The bytes the stage's function writes, from each written slot's offset.
        let mut written = Written::default();
        for (slot, slot_interface) in &interface.slots {
            let SlotInterface::Buffer(buffer) = slot_interface else {
                continue;
            };
            if slot.stage != stage || slot.class != SlotClass::Buffer {
                continue;
            }
            if !buffer.access.write {
                continue;
            }
            if let Some((bound, offset)) = encoder.buffers.get(&slot.index) {
                let used = writes(bound, *offset..bound.length().max(*offset));
                if let Some(range) = bytes(&used) {
                    written.add(used.key, range);
                }
            }
        }
        let mut table: Option<ArgumentTable> = None;
        for (&index, (buffer, offset)) in &encoder.buffers {
            let (offset, length) = (*offset, buffer.length());
            if encoder.function_tables.contains_key(&index) || offset >= length {
                continue;
            }
            let read = reads(buffer, offset..length);
            let Some(range) = bytes(&read) else { continue };
            let key = read.key;
            if !self.fragment.overlaps(key, range)
                || self.other.overlaps(key, range)
                || written.overlaps(key, range)
            {
                continue;
            }
            let (scratch, at) = self.copy(core, stage, buffer, offset)?;
            table
                .get_or_insert_with(|| encoder.clone())
                .buffers
                .insert(index, (scratch, at));
        }
        Ok(table)
    }

    /// The copy of `buffer`'s bytes from `offset` to its end a table of `stage` reads, and the
    /// offset of `offset` in it: one taken before the pass from an offset at or before `offset`,
    /// else a new one, recorded into `core` before the pass between a barrier from everything
    /// before and one to everything after.
    fn copy(
        &mut self,
        core: &mut EncoderCore,
        stage: Stage,
        buffer: &Buffer,
        offset: u64,
    ) -> Result<(Buffer, u64), Error> {
        let source = buffer.at(0);
        if let Some(copy) = self
            .copies
            .iter()
            .find(|copy| copy.source == source && copy.from <= offset)
        {
            return Ok((copy.scratch.clone(), offset - copy.from));
        }
        let size = buffer.length() - offset;
        let device = buffer.device().clone();
        let scratch = device.own_buffer(
            size,
            ResourceOptions::new(
                StorageMode::Private,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Default,
            ),
        )?;
        let (from, src_offset) = buffer.at(offset);
        let (to, dst_offset) = scratch.at(0);
        let region = vk::BufferCopy {
            src_offset,
            dst_offset,
            size,
        };
        let uses = [
            reads(buffer, offset..offset + size),
            writes(&scratch, 0..size),
        ];
        core.barrier(Barrier::everything());
        core.command(TRANSFER, &uses, |raw, command_buffer| {
            // SAFETY: the encoder's command buffer is recording outside a render pass, and only
            // the encoder records it; both buffers are live until it completes (the encoder holds
            // the bound buffer, and the command buffer the copy), and the region lies in both.
            unsafe { raw.cmd_copy_buffer(command_buffer, from, to, &[region]) };
        });
        core.barrier(Barrier::everything());
        core.retain(scratch.clone());
        device.degrade(Degradation::UnorderedDrawBytes, || {
            let stage = match stage {
                Stage::Object => "object",
                Stage::Mesh => "mesh",
                _ => "vertex",
            };
            let object = format!("a draw's {stage} buffer, {size} bytes from offset {offset}");
            let reason = "a fragment stage of the render pass wrote those bytes, which Metal's \
                          vertex, object and mesh stages read from before the pass's fragments \
                          in every run measured; the draws read a copy taken before the pass"
                .to_owned();
            (object, reason)
        });
        self.copies.push(PassStartCopy {
            source,
            from: offset,
            scratch: scratch.clone(),
        });
        Ok((scratch, 0))
    }

    /// Announce once each read of a draw's stages before its fragments among its `uses` of bytes
    /// a fragment stage of the pass wrote, which the draw reads in place, and each write of those
    /// stages to bytes a fragment stage of the pass read or wrote (the module docs). An emulated
    /// object or mesh stage's work, in the compute stage, counts where it records after the
    /// pass's earlier instances (`compute_after`), as the draw's dispatches before its own
    /// instance do (`render.rs`).
    pub(super) fn announce(&self, device: &Device, uses: &[Use<ResourceKey>], compute_after: bool) {
        let in_place = uses.iter().find_map(|used| {
            let range = bytes(used)?;
            let vertex = !(used.read.stages & VERTEX_READS).is_empty();
            (vertex && self.fragment.overlaps(used.key, range)).then(|| range.clone())
        });
        if let Some(range) = in_place {
            device.degrade(Degradation::InPlaceVertexReads, || {
                let object = format!(
                    "a draw's vertex-stage read of bytes {}..{} a fragment stage of its render \
                     pass wrote",
                    range.start, range.end
                );
                let reason = "the read is through an argument buffer or a resident resource, or \
                              of bytes another stage of the pass or the vertex function writes, \
                              where no copy is taken before the pass: the device may order the \
                              fragments' write first, where Metal's vertex stages read the bytes \
                              from before the pass's fragments"
                    .to_owned();
                (object, reason)
            });
        }
        let compute = match compute_after {
            true => vk::PipelineStageFlags2::COMPUTE_SHADER,
            false => vk::PipelineStageFlags2::NONE,
        };
        // The stages whose reads count; their writes count, and the vertex shader's.
        let readers = MESH_STAGES | compute;
        let writers = readers | vk::PipelineStageFlags2::VERTEX_SHADER;
        let ordered = uses.iter().find_map(|used| {
            let range = bytes(used)?;
            let wrote = self.fragment.overlaps(used.key, range);
            let reads = !(used.read.stages & readers).is_empty() && wrote;
            let touched = wrote || self.fragment_reads.overlaps(used.key, range);
            let writes = !(used.write.stages & writers).is_empty() && touched;
            match (reads, writes) {
                (_, true) => Some((range.clone(), "write of", "read or wrote")),
                (true, false) => Some((range.clone(), "in-place read of", "wrote")),
                (false, false) => None,
            }
        });
        if let Some((range, access, fragments)) = ordered {
            device.degrade(Degradation::FragmentsBeforeGeometry, || {
                let object = format!(
                    "a draw's vertex, object or mesh stage's {access} bytes {}..{} a fragment \
                     stage of its render pass {fragments}",
                    range.start, range.end
                );
                let reason = "Metal's GPUs run every draw's vertex, object and mesh functions \
                              before any draw's fragments of the pass, behind a command's \
                              barrier or not, in every run measured; here the device may run the \
                              earlier draws' fragments first, and does across a split of the \
                              pass: the read sees their write, their read misses the write, or \
                              the write outlasts theirs"
                    .to_owned();
                (object, reason)
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Written;
    use crate::resource::ResourceKey;

    /// A buffer's written ranges overlap what they meet and nothing else, by buffer; cleared,
    /// none is noted and the lists keep their room, which the next buffers take.
    #[test]
    fn written_ranges_keep_their_lists_when_cleared() {
        let (a, b) = (ResourceKey::new(), ResourceKey::new());
        let mut written = Written::default();
        assert!(written.is_empty());
        written.add(b, &(0..16));
        written.add(a, &(32..48));
        written.add(b, &(64..80));
        assert!(written.overlaps(b, &(8..9)) && written.overlaps(b, &(70..90)));
        assert!(!written.overlaps(b, &(16..64)) && !written.overlaps(a, &(0..32)));
        assert!(written.overlaps(a, &(40..41)));
        let room: Vec<*const std::ops::Range<u64>> =
            written.lists.iter().map(|(_, r)| r.as_ptr()).collect();
        written.clear();
        assert!(written.is_empty() && !written.overlaps(a, &(40..41)));
        written.add(a, &(0..1));
        written.add(b, &(0..1));
        assert!(written.overlaps(a, &(0..1)) && !written.overlaps(a, &(40..41)));
        let mut taken: Vec<*const std::ops::Range<u64>> =
            written.lists.iter().map(|(_, r)| r.as_ptr()).collect();
        let mut room = room;
        taken.sort();
        room.sort();
        assert_eq!(taken, room, "the lists kept");
    }
}
