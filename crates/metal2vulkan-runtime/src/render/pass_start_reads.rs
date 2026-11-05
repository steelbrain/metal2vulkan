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
//! - **A draw's uses of what its encoder made resident** (`binding::ResidentPrefix`), which begin
//!   its uses, are checked and noted once while what they meet stands: a draw that begins with the
//!   uses the last draw began with, by identity, notes none of them again (a list holds each range
//!   it took, a range it covers adding nothing, until the pass's lists are cleared), and checks
//!   none again where none of them had anything to announce and the bytes the fragment stages
//!   wrote and read are as they were then.

use crate::argument_table::ArgumentTable;
use crate::binding::ResidentPrefix;
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
/// ([`clear`](Self::clear)), which the next keys take. What it answers is whether a range
/// overlaps one it took ([`overlaps`](Self::overlaps)): a range inside one it holds adds nothing,
/// and a merge of a key's ranges into their hull covers each, so each range it took is held, by
/// one that covers it, until it is cleared.
#[derive(Clone, Default)]
struct Written {
    lists: Vec<(ResourceKey, Vec<Range<u64>>)>,
    held: usize,
    /// How often it took a range no range it held covered: what
    /// [`overlaps`](Self::overlaps) answers stands while it does.
    changes: u64,
}

/// Whether one of `ranges` covers `range`.
fn covers(ranges: &[Range<u64>], range: &Range<u64>) -> bool {
    (ranges.iter()).any(|held| held.start <= range.start && range.end <= held.end)
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
        // A range inside one held adds nothing: a pass's draws mostly name the same bytes again,
        // or bytes inside those a resident use named.
        if covers(ranges, range) {
            return;
        }
        ranges.push(range.clone());
        self.changes += 1;
        if ranges.len() > RANGES_PER_KEY {
            let start = ranges.iter().map(|r| r.start).min().unwrap_or(0);
            let end = ranges.iter().map(|r| r.end).max().unwrap_or(0);
            ranges.clear();
            ranges.push(start..end);
        }
    }

    /// Whether it holds `range` of `key`, by a range that covers it: adding it again changes
    /// nothing (builds with debug assertions: [`PassStartReads::add_draw`]'s check).
    #[cfg(debug_assertions)]
    fn holds(&self, key: ResourceKey, range: &Range<u64>) -> bool {
        let held = &self.lists[..self.held];
        let at = held.binary_search_by_key(&key, |&(at, _)| at).ok();
        at.is_some_and(|at| covers(&held[at].1, range))
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

/// The lists a use is noted in, where it uses a buffer's bytes: those of the bytes the
/// fragment stages wrote, the other stages wrote, and the fragment stages read.
fn noted_in<K>(used: &Use<K>) -> [bool; 3] {
    let fragment = vk::PipelineStageFlags2::FRAGMENT_SHADER;
    let writes = used.write.stages;
    [
        writes.contains(fragment),
        !(writes & !fragment).is_empty(),
        used.read.stages.contains(fragment),
    ]
}

/// The first of a draw's uses each kind [`PassStartReads::announce`] announces meets, with its
/// position: a vertex stage's read in place of bytes a fragment stage of the pass wrote, and a
/// read or write of the others the device may order apart from Metal (its access and what the
/// fragment stages did).
type Found = (
    Option<(usize, Range<u64>)>,
    Option<(usize, Range<u64>, &'static str, &'static str)>,
);

/// Bytes of a use's access the device may order apart from Metal, with what it is and what the
/// fragment stages did ([`PassStartReads::announce`]).
type Ordered<'u> = (&'u Range<u64>, &'static str, &'static str);

/// The notice of a draw's vertex-stage read in place of `range`, which a fragment stage of its
/// render pass wrote (`Degradation::InPlaceVertexReads`).
fn in_place_notice(range: &Range<u64>) -> (String, String) {
    let object = format!(
        "a draw's vertex-stage read of bytes {}..{} a fragment stage of its render pass wrote",
        range.start, range.end
    );
    let reason = "the read is through an argument buffer or a resident resource, or of bytes \
                  another stage of the pass or the vertex function writes, where no copy is taken \
                  before the pass: the device may order the fragments' write first, where Metal's \
                  vertex stages read the bytes from before the pass's fragments"
        .to_owned();
    (object, reason)
}

/// The notice of a draw's access the device may order apart from Metal
/// (`Degradation::FragmentsBeforeGeometry`).
fn ordered_notice((range, access, fragments): Ordered<'_>) -> (String, String) {
    let object = format!(
        "a draw's vertex, object or mesh stage's {access} bytes {}..{} a fragment stage of its \
         render pass {fragments}",
        range.start, range.end
    );
    let reason = "Metal's GPUs run every draw's vertex, object and mesh functions before any \
                  draw's fragments of the pass, behind a command's barrier or not, in every run \
                  measured; here the device may run the earlier draws' fragments first, and does \
                  across a split of the pass: the read sees their write, their read misses the \
                  write, or the write outlasts theirs"
        .to_owned();
    (object, reason)
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
    /// The identity and count of the uses of what the encoder made resident the last
    /// [`add_draw`](Self::add_draw) began with: each of those uses' ranges is held until the
    /// lists are cleared ([`Written`]).
    added: Option<(u64, usize)>,
    /// Those the last [`check`](Self::check) began with, where none of them had anything to
    /// announce, with its `compute_after` and the changes then of the bytes the fragment stages
    /// wrote, and read where one of those uses writes in a stage whose writes count: none has
    /// while those stand.
    announced: Option<(u64, usize, bool, (u64, Option<u64>))>,
}

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
    /// [`announce`](Self::announce), then [`add`](Self::add), a draw's `uses`, which begin with
    /// the uses of what its encoder made resident of the identity `prefix`: those through
    /// [`check`](Self::check) and [`add_draw`](Self::add_draw)
    /// (`WorkCounts::resident_uses_against_fragments` counts those each walked).
    pub(super) fn draw(
        &mut self,
        device: &Device,
        (uses, prefix): (&[Use<ResourceKey>], ResidentPrefix),
        compute_after: bool,
    ) {
        let (found, checked) = self.check((uses, prefix), compute_after);
        self.degrade(device, found);
        let walked = [checked, self.add_draw((uses, prefix))];
        let walked = walked.iter().filter(|walked| **walked).count() as u64;
        if walked > 0 {
            let counts = &device.shared().counts;
            counts.resident_uses_against_fragments(walked * prefix.len as u64);
        }
    }

    /// Note the accesses of a draw or tile dispatch of the pass with `uses`.
    pub(super) fn add(&mut self, uses: &[Use<ResourceKey>]) {
        for used in uses {
            let Some(range) = bytes(used) else { continue };
            let [fragment, other, fragment_reads] = noted_in(used);
            if fragment {
                self.fragment.add(used.key, range);
            }
            if other {
                self.other.add(used.key, range);
            }
            if fragment_reads {
                self.fragment_reads.add(used.key, range);
            }
        }
    }

    /// [`add`](Self::add) a draw's `uses`, which begin with the uses of what its encoder made
    /// resident of the identity `prefix`: those not again where the last call of this began with
    /// the same (the module docs). Whether it walked them.
    pub(super) fn add_draw(
        &mut self,
        (uses, prefix): (&[Use<ResourceKey>], ResidentPrefix),
    ) -> bool {
        let now = Some((prefix.id, prefix.len));
        let past = match self.added == now {
            true => prefix.len,
            false => 0,
        };
        #[cfg(debug_assertions)]
        for used in &uses[..past] {
            assert!(
                self.holds(used),
                "a pass noted a draw's uses by their identity less than its uses"
            );
        }
        self.add(&uses[past..]);
        self.added = now;
        past == 0
    }

    /// Whether its lists hold `used`'s range, where they note it: noting it again changes
    /// nothing (builds with debug assertions: [`add_draw`](Self::add_draw)'s check).
    #[cfg(debug_assertions)]
    fn holds(&self, used: &Use<ResourceKey>) -> bool {
        let Some(range) = bytes(used) else {
            return true;
        };
        let lists = [&self.fragment, &self.other, &self.fragment_reads];
        (noted_in(used).into_iter().zip(lists))
            .all(|(noted, list)| !noted || list.holds(used.key, range))
    }

    /// What it notes, without its copies' buffers: its lists' keys and ranges, and its copies'
    /// count (builds with debug assertions: `render/draw_again.rs`'s check, which digests it).
    #[cfg(debug_assertions)]
    pub(super) fn held(&self) -> impl std::fmt::Debug + '_ {
        let lists = [&self.fragment, &self.fragment_reads, &self.other];
        let held = lists.map(|written| &written.lists[..written.held]);
        (held, self.copies.len())
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
        self.added = None;
        self.announced = None;
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
            added: None,
            announced: None,
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
        let in_place = uses.iter().find_map(|used| self.in_place(used));
        if let Some(range) = in_place {
            device.degrade(Degradation::InPlaceVertexReads, || in_place_notice(range));
        }
        let stages = (Self::readers(compute_after), Self::writers(compute_after));
        let ordered = uses.iter().find_map(|used| self.ordered(used, stages));
        if let Some(ordered) = ordered {
            device.degrade(Degradation::FragmentsBeforeGeometry, || {
                ordered_notice(ordered)
            });
        }
    }

    /// Announce what [`check`](Self::check) found, as [`announce`](Self::announce) would.
    fn degrade(&self, device: &Device, (in_place, ordered): Found) {
        if let Some((_, range)) = in_place {
            device.degrade(Degradation::InPlaceVertexReads, || in_place_notice(&range));
        }
        if let Some((_, range, access, fragments)) = ordered {
            let ordered = (&range, access, fragments);
            device.degrade(Degradation::FragmentsBeforeGeometry, || {
                ordered_notice(ordered)
            });
        }
    }

    /// The bytes `used` reads in a stage before the fragments, in place, that a fragment stage
    /// of the pass wrote ([`announce`](Self::announce)). Inlined into each draw's walk of its
    /// uses, as is [`ordered`](Self::ordered).
    #[inline(always)]
    fn in_place<'u>(&self, used: &'u Use<ResourceKey>) -> Option<&'u Range<u64>> {
        let range = bytes(used)?;
        let vertex = !(used.read.stages & VERTEX_READS).is_empty();
        (vertex && self.fragment.overlaps(used.key, range)).then_some(range)
    }

    /// The bytes of `used`'s access of the `stages` ([`readers`](Self::readers),
    /// [`writers`](Self::writers)) that a fragment stage of the pass read or wrote, the device
    /// may order apart from Metal ([`announce`](Self::announce)), with what it is and what the
    /// fragments did.
    #[inline(always)]
    fn ordered<'u>(
        &self,
        used: &'u Use<ResourceKey>,
        (readers, writers): (vk::PipelineStageFlags2, vk::PipelineStageFlags2),
    ) -> Option<Ordered<'u>> {
        let range = bytes(used)?;
        let wrote = self.fragment.overlaps(used.key, range);
        let reads = !(used.read.stages & readers).is_empty() && wrote;
        let touched = wrote || self.fragment_reads.overlaps(used.key, range);
        let writes = !(used.write.stages & writers).is_empty() && touched;
        match (reads, writes) {
            (_, true) => Some((range, "write of", "read or wrote")),
            (true, false) => Some((range, "in-place read of", "wrote")),
            (false, false) => None,
        }
    }

    /// What [`announce`](Self::announce) announces of `uses` ([`Found`], positions from the
    /// first use it walked), and whether it walked those of what the encoder made resident they
    /// begin with, of the identity `prefix`: the uses past them alone where the last call began
    /// with the same and none of them met anything while the bytes the fragment stages wrote and
    /// read stand (the module docs).
    fn check(
        &mut self,
        (uses, prefix): (&[Use<ResourceKey>], ResidentPrefix),
        compute_after: bool,
    ) -> (Found, bool) {
        let changes = (self.fragment.changes, self.fragment_reads.changes);
        let held = |(id, len, after, (wrote, read)): (u64, usize, bool, (u64, Option<u64>))| {
            let same = (prefix.id, prefix.len) == (id, len);
            let stand = wrote == changes.0 && read.is_none_or(|read| read == changes.1);
            same && after == compute_after && stand
        };
        let past = match self.announced.is_some_and(held) {
            true => prefix.len,
            false => 0,
        };
        let (in_place, ordered) = self.found(&uses[past..], compute_after);
        #[cfg(debug_assertions)]
        {
            // What it announces, wherever it lies.
            let answer = |(in_place, ordered): Found| {
                let ordered =
                    ordered.map(|(_, range, access, fragments)| (range, access, fragments));
                (in_place.map(|(_, range)| range), ordered)
            };
            let kept = answer((in_place.clone(), ordered.clone()));
            assert_eq!(
                answer(self.found(uses, compute_after)),
                kept,
                "a pass checked a draw's uses by their identity less than its uses"
            );
        }
        // Where it walked them, whether those uses met nothing, and what that answer rests on.
        if past == 0 {
            let len = prefix.len;
            let met = [
                in_place.as_ref().map(|(at, _)| *at),
                ordered.as_ref().map(|(at, ..)| *at),
            ];
            let clean = met.iter().all(|at| at.is_none_or(|at| at >= len));
            let writes = uses[..len]
                .iter()
                .any(|used| !(used.write.stages & Self::writers(compute_after)).is_empty());
            let read = writes.then_some(changes.1);
            self.announced =
                clean.then_some((prefix.id, prefix.len, compute_after, (changes.0, read)));
        }
        ((in_place, ordered), past == 0)
    }

    /// The first of `uses` each kind [`announce`](Self::announce) announces meets ([`Found`]).
    fn found(&self, uses: &[Use<ResourceKey>], compute_after: bool) -> Found {
        let in_place = (uses.iter().enumerate())
            .find_map(|(at, used)| self.in_place(used).map(|range| (at, range.clone())));
        let stages = (Self::readers(compute_after), Self::writers(compute_after));
        let ordered = uses.iter().enumerate().find_map(|(at, used)| {
            let (range, access, fragments) = self.ordered(used, stages)?;
            Some((at, range.clone(), access, fragments))
        });
        (in_place, ordered)
    }

    /// The stages whose reads of bytes a fragment stage of the pass wrote count: a native object
    /// or mesh stage's, and an emulated one's where its work records after the pass's earlier
    /// instances (`compute_after`).
    fn readers(compute_after: bool) -> vk::PipelineStageFlags2 {
        match compute_after {
            true => MESH_STAGES | vk::PipelineStageFlags2::COMPUTE_SHADER,
            false => MESH_STAGES,
        }
    }

    /// The stages whose writes of bytes a fragment stage of the pass read or wrote count:
    /// [`readers`](Self::readers)' and the vertex shader's.
    fn writers(compute_after: bool) -> vk::PipelineStageFlags2 {
        Self::readers(compute_after) | vk::PipelineStageFlags2::VERTEX_SHADER
    }
}

#[cfg(test)]
mod tests {
    use super::{PassStartReads, Written};
    use crate::binding::ResidentPrefix;
    use crate::resource::ResourceKey;
    use crate::tracker::{Access, Use, UseRange};
    use ash::vk;
    use std::ops::Range;

    /// A use of `range` of `key`'s bytes, read in `stage`, or written (`write`).
    fn used(
        key: ResourceKey,
        range: Range<u64>,
        stage: vk::PipelineStageFlags2,
        write: bool,
    ) -> Use<ResourceKey> {
        let range = UseRange::Bytes(range);
        match write {
            true => Use::writes(
                key,
                true,
                range,
                Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_WRITE),
            ),
            false => Use::reads(
                key,
                true,
                range,
                Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_READ),
            ),
        }
    }

    /// The uses of what an encoder made resident of identity `id`: the first `len` of a draw's.
    fn prefix(id: u64, len: usize) -> ResidentPrefix {
        ResidentPrefix { id, len }
    }

    /// A draw's uses of what its encoder made resident are checked against what the pass's
    /// fragment stages wrote once while none of them meets any and those bytes stand: again after
    /// its own fragments wrote, not after a draw that changed nothing; once another draw's
    /// fragments wrote what they read, again, and while that meets them, at each draw (builds with
    /// debug assertions check each answer against that of the draw's whole uses).
    #[test]
    fn a_draw_checks_its_resident_uses_again_once_the_fragments_wrote_more() {
        let (vertex, fragment) = (
            vk::PipelineStageFlags2::VERTEX_SHADER,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
        );
        let (a, b) = (ResourceKey::new(), ResourceKey::new());
        let mut pass = PassStartReads::default();
        // Its vertex stage reads `a` through the resident set, its fragments write `b`.
        let draw = [
            used(a, 0..16, vertex, false),
            used(b, 0..16, fragment, true),
        ];
        let checked = |pass: &mut PassStartReads| {
            let ((in_place, _), walked) = pass.check((&draw, prefix(1, 1)), false);
            (in_place.map(|(_, range)| range), walked)
        };
        assert_eq!(checked(&mut pass), (None, true));
        assert!(pass.add_draw((&draw, prefix(1, 1))));
        assert_eq!(checked(&mut pass), (None, true), "its fragments wrote");
        assert!(!pass.add_draw((&draw, prefix(1, 1))));
        let unchanged = "nothing it meets changed";
        assert_eq!(checked(&mut pass), (None, false), "{unchanged}");
        // Another draw's fragments write `a`.
        pass.add(&[used(a, 0..16, fragment, true)]);
        for _ in 0..2 {
            assert_eq!(checked(&mut pass), (Some(0..16), true));
        }
    }

    /// A draw's uses of what its encoder made resident are noted once while the pass's lists
    /// hold them: a draw whose own uses past them merge a list's ranges into their hull leaves
    /// them held, by the hull, so later draws note none of them (builds with debug assertions
    /// check each held where it is not noted), until the lists are cleared.
    #[test]
    fn a_draw_notes_its_resident_uses_once_though_a_list_merged_their_ranges() {
        let fragment = vk::PipelineStageFlags2::FRAGMENT_SHADER;
        let key = ResourceKey::new();
        let mut pass = PassStartReads::default();
        // Each draw reads, in its fragment stage, the same bytes of `key` through the resident
        // set, then bytes of its own; the 16th draw's are a 17th range, which merges them.
        let walked: Vec<bool> = (1..=super::RANGES_PER_KEY as u64 + 2)
            .map(|at| {
                let own = 32 * at..32 * at + 16;
                let draw = [
                    used(key, 0..16, fragment, false),
                    used(key, own, fragment, false),
                ];
                pass.add_draw((&draw, prefix(2, 1)))
            })
            .collect();
        let noted: Vec<usize> = (walked.iter().enumerate())
            .filter_map(|(at, walked)| walked.then_some(at))
            .collect();
        assert_eq!(noted, [0], "{walked:?}");
        pass.clear();
        let draw = [used(key, 0..16, fragment, false)];
        assert!(pass.add_draw((&draw, prefix(2, 1))), "cleared");
    }

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
