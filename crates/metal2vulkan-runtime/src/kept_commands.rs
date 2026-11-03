//! Recordings of an indirect command buffer's commands, kept from one execution to the next:
//! a command buffer that executes commands no write reached since an earlier one recorded them,
//! under the same inputs, runs that recording again and records nothing of them.
//!
//! - **By chunk.** The buffer's commands fall into chunks of [`CHUNK`], aligned on their indices.
//! - **From the second execution.** An execution that may keep its commands (each encoder's
//!   rules: `render/executions.rs`, `indirect_execute.rs`) finds what is kept of each chunk its
//!   range covers ([`Kept`]): a kept recording, none but an execution of the chunk before it with
//!   no write reaching it since, or neither (the chunk's first execution since a write reached
//!   it). An execution none of whose chunks ran before so records as one that keeps nothing does,
//!   so a range executed once pays nothing for keeping. Every other one records each chunk its
//!   range covers apart, a render chunk into a
//!   secondary command buffer of the execution's render pass instance, a compute chunk into a
//!   primary of its segment, from a pool, descriptor arena and upload arena of the chunk's own
//!   ([`KeptContext`]), lent to the command buffer while it records the chunk and held by every
//!   command buffer that runs the recording until it completes; a chunk's first execution records
//!   apart from the command buffer's own, and keeps nothing. A chunk whose recording is plain
//!   (each encoder's rules: nothing of it records outside its command buffer, and nothing of it
//!   reaches what the command buffer alone holds) is kept ([`KeptChunk`]): its command buffer,
//!   begun for simultaneous use, the uses its commands noted, the keys they named, weak holds on
//!   what they held, their execution records, and what it was recorded from (the encoder's
//!   inputs, each address its commands named and what it resolved to, the no-copy buffers'
//!   moves).
//! - **Run again.** A later execution of the chunk's commands under equal inputs, where every
//!   address resolves as it did, nothing moved, nothing it named is detached and everything it
//!   held lives, runs the kept command buffer: its command buffer notes the uses again, keeps the
//!   keys, holds what the recording held and the chunk, and adds the execution records, as the
//!   recording's own did, and counts it (`WorkCounts::indirect_chunks_kept`,
//!   `indirect_commands_kept`). Its records are not read.
//! - **Forgotten** at the CPU's every write of one of the chunk's commands (a setter, a reset, a
//!   slot past the record's), under the lock the write takes; for good at the buffer's first
//!   GPU write (`IndirectCommandBuffer::written_by_gpu`), after which no execution reads the
//!   CPU's records at commit; and where its inputs differ, its next recording replacing it. A
//!   recording made from records the CPU wrote while it recorded them is not kept. What a
//!   command buffer recorded from a kept chunk stays its own: a write after its commit has its
//!   host step record the execution again from the records as they stand then, as before.

use crate::command_buffer::{command_pool, hand_out, CommandBuffer};
use crate::descriptor_arena::DescriptorArena;
use crate::device::{lock, Device};
use crate::error::Error;
use crate::execution::Execution;
use crate::residency::Residency;
use crate::resource::ResourceKey;
use crate::spares::CommandPool;
use crate::tracker::Summary;
use crate::upload::Uploads;
use ash::vk;
use metal2vulkan::bind::ReplayFaults;
use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::{Arc, Mutex, Weak};

/// The commands of a chunk: a render chunk's draws record into one secondary command buffer, a
/// compute chunk's dispatches into one primary.
pub(super) const CHUNK: u32 = 256;

/// The chunks `range` falls into, in order: each chunk's index and the commands of `range` it
/// covers.
pub(super) fn chunks(range: Range<u32>) -> impl Iterator<Item = (u32, Range<u32>)> {
    let mut start = range.start;
    std::iter::from_fn(move || {
        if start >= range.end {
            return None;
        }
        let chunk = start / CHUNK;
        let next = (u64::from(chunk) + 1) * u64::from(CHUNK);
        let end = u32::try_from(next).map_or(range.end, |next| next.min(range.end));
        let covered = start..end;
        start = end;
        Some((chunk, covered))
    })
}

/// The chunks of `range`, as [`chunks`] gives them: their indices.
fn chunks_of(range: Range<u32>) -> Range<u32> {
    match range.is_empty() {
        true => 0..0,
        false => range.start / CHUNK..(range.end - 1) / CHUNK + 1,
    }
}

/// What a chunk's recording takes its command buffers, sets and uploads from (the module docs):
/// lent to the command buffer that records the chunk while it does (`State::kept`), held by each
/// command buffer that runs what it recorded until it completes, and handed back to the device's
/// spares as the last hold goes.
pub(super) struct KeptContext {
    device: Device,
    /// Its queue family: its command buffers run on that family's queues alone.
    pub(super) family: u32,
    pool: Mutex<Option<CommandPool>>,
    pub(super) uploads: Arc<Mutex<Uploads>>,
    pub(super) descriptors: Arc<Mutex<DescriptorArena>>,
}

impl KeptContext {
    /// A context for a recording of `command_buffer`'s, of its queue's family: a spare command
    /// pool of the device's, or a new one.
    pub(super) fn new(command_buffer: &CommandBuffer) -> Result<Arc<KeptContext>, Error> {
        let device = command_buffer.device();
        let family = command_buffer.queue_family();
        let pool = command_pool(device, family)?;
        device.shared().counts.kept_context_made();
        Ok(Arc::new(KeptContext {
            device: device.clone(),
            family,
            pool: Mutex::new(Some(pool)),
            uploads: Arc::new(Mutex::new(Uploads::new())),
            descriptors: Arc::new(Mutex::new(DescriptorArena::default())),
        }))
    }

    /// A `VkCommandBuffer` of `level` from its pool (`command_buffer::hand_out`), and how many it
    /// handed out so far, this one included.
    pub(super) fn hand_out(
        &self,
        level: vk::CommandBufferLevel,
    ) -> Result<(vk::CommandBuffer, usize), Error> {
        let mut pool = lock(&self.pool);
        let pool = pool
            .as_mut()
            .ok_or_else(|| crate::error::internal("a kept recording's pool given back"))?;
        let raw = hand_out(&self.device, pool, level)?;
        Ok((raw, pool.handed_out()))
    }

    /// How many command buffers its pool handed out.
    pub(super) fn handed_out(&self) -> usize {
        lock(&self.pool).as_ref().map_or(0, CommandPool::handed_out)
    }
}

/// No command buffer holds it, so nothing of it is pending: its pool goes to the device's spares,
/// to be reset as it is taken again, and its descriptor pools too.
impl Drop for KeptContext {
    fn drop(&mut self) {
        let pool = lock(&self.pool).take();
        if let Some(pool) = pool {
            let shared = self.device.shared();
            shared.spares.give_command_pool(shared, pool, false);
        }
        lock(&self.descriptors).give_back();
    }
}

/// What a kept chunk's recording did to its encoder, which a command buffer that runs it again
/// does to its own (the module docs).
pub(super) struct KeptEffects {
    /// The uses its commands noted, joined, in every stage they ran in.
    pub(super) summary: Summary<ResourceKey>,
    /// The keys its commands named and bound.
    pub(super) keys: BTreeSet<ResourceKey>,
    /// The keys its commands named for themselves, beside those they reached through the
    /// resident set alone (`Taken`).
    pub(super) named: BTreeSet<ResourceKey>,
    /// What it held until its command buffer completed: weakly, so a kept chunk holds nothing a
    /// program let go of.
    held: Vec<Weak<dyn Any + Send + Sync>>,
    /// Its execution records, where the device keeps them.
    pub(super) executions: Vec<Execution>,
}

impl KeptEffects {
    /// The effects of a recording that noted `summary`, kept `keys`, named `named`, held `held`
    /// and recorded `executions`.
    pub(super) fn new(
        summary: Summary<ResourceKey>,
        (keys, named): (BTreeSet<ResourceKey>, BTreeSet<ResourceKey>),
        held: &[Arc<dyn Any + Send + Sync>],
        executions: Vec<Execution>,
    ) -> KeptEffects {
        KeptEffects {
            summary,
            keys,
            named,
            held: held.iter().map(Arc::downgrade).collect(),
            executions,
        }
    }

    /// What it held, each upgraded; `None` where one went.
    fn held(&self) -> Option<Vec<Arc<dyn Any + Send + Sync>>> {
        self.held.iter().map(Weak::upgrade).collect()
    }
}

/// An address a kept chunk's commands named and the buffer it resolved to then (`None`: none).
pub(super) type Resolution = (u64, Option<ResourceKey>);

/// `addresses`, a chunk's commands' (in any order, any repeated), each resolved against
/// `residency` as [`KeptChunk::runs_again`] resolves them again: one address of each buffer an
/// address lies inside of, where the buffer is no heap's (no other buffer holds its addresses, so
/// every address inside it resolves to it while that one does), and every other address, once.
pub(super) fn resolutions(
    addresses: impl IntoIterator<Item = u64>,
    residency: &Residency,
) -> Vec<Resolution> {
    let mut buffers = BTreeMap::new();
    let mut others = BTreeMap::new();
    for address in addresses {
        if others.contains_key(&address) {
            continue;
        }
        let resolved = residency.resolve(address);
        match &resolved {
            Some((buffer, offset)) if buffer.heap().is_none() && *offset < buffer.length() => {
                buffers.entry(buffer.identity().key).or_insert(address);
            }
            Some(_) | None => {
                let key = resolved.map(|(buffer, _)| buffer.identity().key);
                others.insert(address, key);
            }
        }
    }
    let buffers = buffers
        .into_iter()
        .map(|(key, address)| (address, Some(key)));
    buffers.chain(others).collect()
}

/// A chunk's kept recording (the module docs).
pub(super) struct KeptChunk {
    context: Arc<KeptContext>,
    /// Its command buffer, ended: a render chunk's secondary, a compute chunk's primary.
    pub(super) raw: vk::CommandBuffer,
    /// The buffer's commands it records.
    pub(super) covered: Range<u32>,
    /// The inputs its encoder recorded it from, which a later execution's must equal (each
    /// encoder's own).
    key: Box<dyn Any + Send + Sync>,
    /// Each address its commands named, once, and what it resolved to.
    resolved: Vec<Resolution>,
    /// The no-copy buffers' moves when it was recorded (`buffer::host_cells_moved`).
    moved: u64,
    pub(super) effects: KeptEffects,
    /// The resident-set misses and faults its recording found.
    pub(super) misses: u32,
    pub(super) faults: ReplayFaults,
}

impl KeptChunk {
    /// The kept recording `raw`, of `context`, of commands `covered`, under `key`, its addresses
    /// `resolved` so, recorded while the no-copy buffers had moved `moved` times.
    pub(super) fn new(
        (context, raw): (Arc<KeptContext>, vk::CommandBuffer),
        covered: Range<u32>,
        key: Box<dyn Any + Send + Sync>,
        (resolved, moved): (Vec<Resolution>, u64),
        effects: KeptEffects,
        (misses, faults): (u32, ReplayFaults),
    ) -> KeptChunk {
        KeptChunk {
            context,
            raw,
            covered,
            key,
            resolved,
            moved,
            effects,
            misses,
            faults,
        }
    }

    /// What a command buffer of `command_buffer`'s queue family that runs it again under `key`,
    /// resolving against `residency`, holds of it: what its recording held, upgraded; `None`
    /// where it may not run again (the module docs).
    pub(super) fn runs_again<K: PartialEq + 'static>(
        &self,
        command_buffer: &CommandBuffer,
        (key, covered): (&K, &Range<u32>),
        residency: &Residency,
    ) -> Option<Vec<Arc<dyn Any + Send + Sync>>> {
        let same = self.covered == *covered
            && self.context.family == command_buffer.queue_family()
            && self.key.downcast_ref::<K>() == Some(key)
            && self.moved == crate::buffer::host_cells_moved();
        if !same {
            return None;
        }
        let resolves = self.resolved.iter().all(|&(address, key)| {
            residency.resolve(address).map(|(b, _)| b.identity().key) == key
        });
        if !resolves {
            return None;
        }
        let rebinds = &command_buffer.device().shared().rebinds;
        if rebinds.any_detached(self.effects.keys.iter().copied()) {
            return None;
        }
        self.effects.held()
    }
}

/// What an execution that may keep its commands finds of one of its chunks (the module docs).
pub(super) enum Kept {
    /// Its kept recording.
    Recording(Arc<KeptChunk>),
    /// None, but an execution of it ran before, no write reaching it since: its recording now is
    /// kept.
    Again,
    /// Its first execution since a write reached it: it records from its command buffer's own,
    /// kept by none.
    First,
}

/// The kept chunks of a buffer's commands, those whose last recording was not plain, and those
/// an execution ran, by chunk: held under the lock every CPU write of the buffer's records takes
/// (`IndirectCommandBuffer::kept`). A state a chunk, from chunk 0 to the furthest an execution
/// looked up or noted, so a write of commands costs the chunks it covers, however many others hold
/// a state, and nothing while none does.
#[derive(Default)]
pub(super) struct KeptChunks {
    chunks: Vec<ChunkState>,
    /// Whether a chunk took a state since the buffer's records were last forgotten for good.
    any: bool,
}

/// What [`KeptChunks`] holds of one chunk.
#[derive(Default)]
struct ChunkState {
    kept: Option<Arc<KeptChunk>>,
    /// An execution last recorded it otherwise than plainly: an execution of it records as one,
    /// until a write reaches it.
    stepped: bool,
    /// An execution that may keep its commands ran it since a write last reached it.
    seen: bool,
}

impl KeptChunks {
    /// The CPU wrote commands `range`: forget the chunks they lie in. How many chunks' states it
    /// looked at: those `range` covers that hold one, none while no chunk holds a state.
    pub(super) fn wrote(&mut self, range: Range<u32>) -> usize {
        if !self.any {
            return 0;
        }
        let held = self.held(chunks_of(range));
        let looked = held.len();
        for state in &mut self.chunks[held] {
            *state = ChunkState::default();
        }
        looked
    }

    /// The GPU may write the records from now on: forget every chunk.
    pub(super) fn clear(&mut self) {
        self.chunks.clear();
        self.any = false;
    }

    /// What an execution that may keep its commands finds of each chunk of `range`, in order,
    /// each noted as run now.
    pub(super) fn look_up(&mut self, range: Range<u32>) -> Vec<Kept> {
        chunks(range)
            .map(|(chunk, _)| {
                let state = self.state(chunk);
                let first = !state.seen;
                state.seen = true;
                match (&state.kept, first) {
                    (Some(kept), _) => Kept::Recording(Arc::clone(kept)),
                    (None, false) => Kept::Again,
                    (None, true) => Kept::First,
                }
            })
            .collect()
    }

    /// Whether an execution last recorded a chunk of `range` otherwise than plainly.
    pub(super) fn stepped_in(&self, range: Range<u32>) -> bool {
        let held = self.held(chunks_of(range));
        self.chunks[held].iter().any(|state| state.stepped)
    }

    /// Keep `kept`, chunk `chunk`'s recording.
    pub(super) fn keep(&mut self, chunk: u32, kept: Arc<KeptChunk>) {
        let state = self.state(chunk);
        state.stepped = false;
        state.kept = Some(kept);
    }

    /// Note that chunk `chunk`'s recording was not plain.
    pub(super) fn stepped(&mut self, chunk: u32) {
        let state = self.state(chunk);
        state.kept = None;
        state.stepped = true;
    }

    /// Where in the list the states of `chunks` lie that it holds.
    fn held(&self, chunks: Range<u32>) -> Range<usize> {
        let at = |chunk: u32| (chunk as usize).min(self.chunks.len());
        at(chunks.start)..at(chunks.end)
    }

    /// Chunk `chunk`'s state, the list grown to hold it.
    fn state(&mut self, chunk: u32) -> &mut ChunkState {
        let at = chunk as usize;
        if at >= self.chunks.len() {
            self.chunks.resize_with(at + 1, ChunkState::default);
        }
        self.any = true;
        &mut self.chunks[at]
    }
}

#[cfg(test)]
mod tests {
    use super::{chunks, chunks_of, Kept, KeptChunks, CHUNK};

    /// A write looks at the states of the chunks it covers alone, however many others hold one,
    /// and at none while no chunk holds a state; it forgets those, and no other.
    #[test]
    fn a_write_costs_the_chunks_it_covers() {
        let mut kept = KeptChunks::default();
        assert_eq!(kept.wrote(0..CHUNK * 4), 0, "no state held");
        let firsts = kept.look_up(0..CHUNK * 1000);
        assert!(firsts.iter().all(|kept| matches!(kept, Kept::First)));
        kept.stepped(500);
        assert_eq!(kept.wrote(CHUNK * 7 + 5..CHUNK * 7 + 6), 1);
        assert_eq!(kept.wrote(CHUNK * 499..CHUNK * 501), 2);
        assert_eq!(
            kept.wrote(CHUNK * 999..CHUNK * 2000),
            1,
            "past the furthest held"
        );
        assert!(!kept.stepped_in(CHUNK * 500..CHUNK * 501), "forgotten");
        let again = kept.look_up(CHUNK * 6..CHUNK * 9);
        let kinds: Vec<bool> = again.iter().map(|k| matches!(k, Kept::First)).collect();
        assert_eq!(kinds, [false, true, false], "chunk 7 alone written");
        kept.stepped(8);
        assert!(kept.stepped_in(CHUNK * 8..CHUNK * 8 + 1));
        kept.clear();
        assert_eq!(kept.wrote(0..CHUNK * 1000), 0, "forgotten for good");
    }

    #[test]
    fn chunks_cover_a_range_in_aligned_pieces() {
        let pieces: Vec<_> = chunks(10..CHUNK * 2 + 5).collect();
        assert_eq!(
            pieces,
            [
                (0, 10..CHUNK),
                (1, CHUNK..CHUNK * 2),
                (2, CHUNK * 2..CHUNK * 2 + 5)
            ]
        );
        assert_eq!(chunks(5..5).count(), 0);
        let last: Vec<_> = chunks(u32::MAX - 3..u32::MAX).collect();
        assert_eq!(last, [((u32::MAX - 3) / CHUNK, u32::MAX - 3..u32::MAX)]);
        assert_eq!(chunks_of(10..CHUNK * 2 + 5), 0..3);
        assert_eq!(chunks_of(CHUNK..CHUNK * 2), 1..2);
        assert_eq!(chunks_of(7..7), 0..0);
    }
}
