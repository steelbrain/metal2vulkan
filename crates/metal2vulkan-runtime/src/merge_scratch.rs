//! The scratch buffers of a command buffer's merges (`linear_merge.rs`, `shadow_merge.rs`): each
//! merge's own until its encoder ends, then lent to the merges of the command buffer's later
//! encoders, so the scratch a command buffer holds follows the attachments and shadowed textures
//! its encoders merge, not their count.
//!
//! - **Why.** A merge's scratch was its pass's or its encoder's alone, held until the command
//!   buffer completed: a command buffer of hundreds of passes over one linear attachment beside a
//!   depth attachment held hundreds (a WebGPU frame split at every indexed draw: 706 passes over
//!   a 1200x1200 `BGRA8Unorm` surface, 4 to 5.6 GiB on the consumer's RTX 5080 Laptop GPU, whose
//!   `endEncoding` failed out of device memory).
//! - **Lent.** A merge takes the smallest free scratch of its kind of at least the bytes it needs
//!   ([`MergeScratch::take`]); its encoder gives it back as it ends: a linear attachment's merge
//!   once its pass's merge is recorded (`LinearMerge::record_merge`), a shadow's as the encoder
//!   that planned it goes (its [`ScratchLease`], held with the encoder's kept-image leases). A
//!   scratch the pool lends is held by the merges that used it, which their command buffer
//!   retains until it completes; the pool holds none of them (a weak reference each), so the
//!   bytes go as before, once the command buffer completed: a shadow's back to its spares, for a
//!   later command buffer's merges (`shadow_merge::ScratchSpares`), once the last merge that used
//!   it goes.
//! - **Order.** Lending a scratch to a later encoder of the same command buffer needs no wait:
//!   1. A merge's scratch is touched only by commands of the encoder that took it: a linear
//!      attachment's by its pass's fill and keep after the copy in, its reloads at the pass's
//!      barriers and its merge after the pass, all recorded at the pass's end into its encoder's
//!      `VkCommandBuffer`s or the segments it leaves between them, which go between them in
//!      submission order; a shadow's by its keeps and merges, which the encoder records before
//!      its next command that touches the bytes another way, or as it ends (`shadow.rs`).
//!   2. A command buffer's encoders go in order: one is open at a time (a render encoder's end
//!      left to the lowering thread is lowered before the next encoder opens,
//!      `render/deferral.rs`), and its batches go to its queue's one `VkQueue` in order, which the
//!      queue keeps from the command buffer's making to its completion (`queue_share.rs`). A
//!      predicated loop's laying submitted again, and an encoder a rebind encodes again
//!      (`command_buffer/reencode.rs`), repeat or replace whole encoders. So the spans from the
//!      first to the last command of two encoders that took one scratch never interleave in
//!      submission order.
//!   3. Each span writes every byte of the scratch it reads before reading it (the copies into
//!      its parts, and the fill with zero of a linear merge whose rows start or end inside a
//!      word), and each ends with a barrier from every command to every command
//!      (`Barrier::everything`: all commands' memory writes, to all commands' reads and writes),
//!      after its last dispatch. Outside a render pass a pipeline barrier's first
//!      synchronization scope holds every command before it in submission order on its queue,
//!      in earlier command buffers too, and its second every command after it: so the later
//!      span's first write follows the earlier span's reads (an execution dependency) and its
//!      writes (made available and visible). A shadow's scratch is a tracked buffer besides,
//!      whose uses the tracker orders as any buffer's.
//!   4. One scratch serves one merge at a time: taken out of the pool until its encoder gives it
//!      back, so two merges of one pass, or of one encoder, never share one.
//! - **Sizes.** A merge's scratch is sized to what it needs, its box's or its rectangle's rows,
//!   where the pool lends none that holds it. A new one takes at least twice the largest of its
//!   kind the command buffer made (but never more than a scratch of the whole attachment or
//!   texture needs, `whole`), so a run of passes whose boxes grow makes a few, the largest a
//!   whole one, after which each pass takes that one: a command buffer's scratch is at most about
//!   three whole scratches per attachment it merges at once, whatever its passes' count.
//!   Counted: `WorkCounts::linear_merge_scratches_made` and `linear_merge_scratches_reused`, and
//!   for shadows `WorkCounts::shadow_merge_scratches_reused`.

use super::buffer::Buffer;
use super::device::lock;
use super::shadow_merge::ScratchSpares;
use std::sync::{Arc, Mutex, Weak};

/// The kinds of scratch, apart: a linear attachment's merge takes an untracked buffer
/// (`Device::scratch_buffer`), a shadow's a tracked one, whose uses the tracker orders.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ScratchKind {
    Linear,
    Shadow,
}

impl ScratchKind {
    fn at(self) -> usize {
        match self {
            ScratchKind::Linear => 0,
            ScratchKind::Shadow => 1,
        }
    }
}

/// A merge's scratch buffer: held by the merges that used it (their command buffer retains them
/// until it completes), lent through its command buffer's pool ([`MergeScratch`]).
pub(super) struct Scratch {
    buffer: Buffer,
    kind: ScratchKind,
    /// The spares of the shadow it goes back to as the last merge that used it goes; none for a
    /// linear attachment's.
    spares: Option<ScratchSpares>,
}

impl Scratch {
    /// A scratch of `kind` in `buffer`, back to `spares` as it goes, where given.
    pub(super) fn new(
        buffer: Buffer,
        kind: ScratchKind,
        spares: Option<ScratchSpares>,
    ) -> Arc<Scratch> {
        Arc::new(Scratch {
            buffer,
            kind,
            spares,
        })
    }

    /// Its buffer.
    pub(super) fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

impl Drop for Scratch {
    /// A shadow's scratch goes back to its spares: every merge that used it went, so the command
    /// buffers that recorded them completed.
    fn drop(&mut self) {
        if let Some(spares) = &self.spares {
            spares.give(self.buffer.clone());
        }
    }
}

/// A command buffer's pool of the scratch its ended encoders' merges are done with (the module
/// docs). A leaf lock: nothing is made or dropped under it.
#[derive(Default)]
pub(super) struct MergeScratch(Mutex<Pool>);

/// What [`MergeScratch`] holds.
#[derive(Default)]
struct Pool {
    /// The scratch given back and not taken again, each with its kind and bytes: none held, so a
    /// scratch whose merges all went is gone, and its entry with it at the next take.
    free: Vec<(Weak<Scratch>, ScratchKind, u64)>,
    /// The largest scratch of each kind made for the command buffer's merges.
    largest: [u64; 2],
}

impl MergeScratch {
    /// The smallest free scratch of `kind` of at least `length` bytes, taken; none where the pool
    /// lends none.
    pub(super) fn take(&self, kind: ScratchKind, length: u64) -> Option<Arc<Scratch>> {
        let mut pool = lock(&self.0);
        let free = &mut pool.free;
        // Entries of scratch that went are dropped: weak references, so no scratch goes here.
        free.retain(|(scratch, ..)| scratch.strong_count() > 0);
        loop {
            let at = (free.iter().enumerate())
                .filter(|(_, &(_, of, bytes))| of == kind && bytes >= length)
                .min_by_key(|(_, &(_, _, bytes))| bytes)
                .map(|(at, _)| at)?;
            // Gone since the retain only where its last merge went meanwhile: the next.
            if let Some(scratch) = free.swap_remove(at).0.upgrade() {
                return Some(scratch);
            }
        }
    }

    /// The bytes a new scratch of `kind` takes for a merge of `length` bytes whose scratch a
    /// whole attachment's or texture's merge would take as `whole` (the module's sizes): at least
    /// `length`, and twice the largest of its kind the command buffer made but no more than
    /// `whole`; noted as the largest where it is.
    pub(super) fn grown(&self, kind: ScratchKind, length: u64, whole: u64) -> u64 {
        let mut pool = lock(&self.0);
        let largest = &mut pool.largest[kind.at()];
        let grown = length.max(whole.min(largest.saturating_mul(2)));
        *largest = (*largest).max(grown);
        grown
    }

    /// Lend `scratch` to the command buffer's later merges: its encoder ended its use of it.
    pub(super) fn give(&self, scratch: &Arc<Scratch>) {
        let entry = (
            Arc::downgrade(scratch),
            scratch.kind,
            scratch.buffer.length(),
        );
        lock(&self.0).free.push(entry);
    }
}

/// An encoder's hold on the scratch a shadow's merge took from its command buffer's pool: given
/// back as it goes, with the encoder (`EncoderCore`'s leases).
pub(super) struct ScratchLease {
    scratch: Arc<Scratch>,
    pool: Arc<MergeScratch>,
}

impl ScratchLease {
    /// The hold on `scratch`, given back to `pool` as it goes.
    pub(super) fn new(scratch: Arc<Scratch>, pool: Arc<MergeScratch>) -> ScratchLease {
        ScratchLease { scratch, pool }
    }
}

impl Drop for ScratchLease {
    fn drop(&mut self) {
        self.pool.give(&self.scratch);
    }
}
