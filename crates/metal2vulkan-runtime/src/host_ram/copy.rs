//! Copies between a guest RAM and a buffer, by the GPU, from the device's imports of the RAM's
//! chunks (a runtime extension: Metal has none): [`BlitCommandEncoder::copy_from_host_ram`]
//! gathers runs of the RAM into consecutive bytes of a buffer, and
//! [`BlitCommandEncoder::copy_to_host_ram`] scatters them back;
//! [`BlitCommandEncoder::copy_runs_from_host_ram`] and
//! [`BlitCommandEncoder::copy_runs_to_host_ram`] do the same with each run at a byte of the
//! buffer of its own (a buffer that keeps a surface's layout, its rows at a pitch). A VMM whose
//! guest's surface is a scatter of guest pages copies it with no view of the pages and, where the
//! RAM's chunks are imported already (`host_ram.rs`), no import of its own.
//!
//! - **The runs** are host addresses in the RAM's ranges, each starting on a 4-byte word ([`WORD`])
//!   and ending at any byte past it: a surface's whole pages, or each of its rows' texels alone,
//!   whose last bytes the guest's CPU may own past them. Each is at a byte of the buffer ([`Run`]):
//!   one after another from the call's offset ([`one_after_another`]), or each at its own. A run
//!   that starts where the one before it ended, in the RAM and in the buffer, is one run with it
//!   ([`merged`]); a run that starts before (a descending neighbour) stays apart. Hazard tracking
//!   takes the buffer's bytes from the lowest run's first to the highest's last as the copy's; the
//!   bytes between the runs (a row's padding) are neither read nor written. Each run is cut where a
//!   chunk ends ([`pieces`]), and each chunk the pieces touch is taken (`RamOnDevice::for_copy`):
//!   its import where it is made; else, where runs are imported alone (each listed import costing
//!   every submission, `RamOnDevice::runs_alone`), an import of each of the copy's
//!   ranges of it alone (those less than 2 MiB apart one, at most [`MOST_ALONE`]:
//!   [`alone_ranges`]), made on the encoding thread, which faults in and pins those pages and no
//!   more, while the device's thread imports the chunk (the thread's window moved past it, or,
//!   where each listed import costs every submission, the chunk asked for): so no copy waits for a
//!   chunk's import. Those imports are held by the command buffer until it completed, then freed
//!   (`WorkCounts::run_imports_live`). Elsewhere (the thread stopped, failed the chunk or is not
//!   started, the allocation limit's margin, `RuntimeDriverDefect::AliasedImportBlitsUnordered`,
//!   more ranges than [`MOST_ALONE`], a range that is the whole chunk, a range's import that fails)
//!   the copy imports the chunk, or waits for its import under way, which is the chunk's one import
//!   (`Device::import_chunk`), made and counted as an object's is
//!   (`WorkCounts::host_ram_copy_import_waits`, `host_ram_copy_import_wait_nanos`). A copy over
//!   a chunk the device's thread imports takes neither: it takes the CPU's path below, unless it
//!   is encoded again.
//! - **First touch through the CPU** ([`StagedGather`], [`StagedScatter`]). A copy over a chunk not
//!   imported yet, which the device's thread imports (`RamOnDevice::copy_take`), on a device whose
//!   plan is not `Plan::listed`, and over none whose import only the copy would make, is staged
//!   instead of importing its ranges alone or waiting for the chunk's: an import of a copy's
//!   ranges pins every page less than 2 MiB from the next on the encoding thread (a cache of 2,477
//!   runs over five chunks, its runs joined where less than 1 MiB apart, pinned 317 MiB in 61 ms
//!   on an RTX 5080 Laptop GPU, where the CPU copied the runs in 2.8 ms:
//!   `tests/host_ram_first_fill_bench.rs`), a consumer's fills of new surfaces there with three
//!   video clips at once took 1.4 to 2.1 s while chunks were being imported. No import is made or
//!   waited for, and no page pinned for it; each chunk is taken as an object takes it, and asked
//!   of the device's thread, which imports it before its window's next
//!   (`RamOnDevice::take_for_copies`), so the copies after it go through the chunks' imports. A
//!   copy over a chunk that takes no import while it runs is staged always, whatever else it
//!   touches, whether the device's thread tried the chunk or not, so that a copy Metal takes is
//!   never refused for when it came (`CopyTake::CpuOnly`): a chunk the device refuses whatever it
//!   holds (shared memory on a device that imports no pointer to it, the RAM given no source of
//!   dma-bufs: `ChunkSlot::refused`), and one a live import of an object's own pages holds back
//!   (`RamOnDevice::held_back`: an import of their pointer at the RAM's own addresses, or any on
//!   a device whose barriers do not order two imports of the same pages, ANV's among them). A
//!   staged gather whose host bytes some use but a copy's read may write as it is encoded (a
//!   copy into those chunks was recorded, a buffer or heap over host pages lies over them:
//!   `RamOnDevice::written_over`), over any chunk, goes in a part of its own
//!   (`Staging::after_the_gpu`), so that the work before it in its command buffer is submitted,
//!   and completed, before its step reads the pages; and its encoder keeps no log an encoding
//!   again would copy it through the imports from (that copy would start no part of its own). So
//!   a first fill over chunks an earlier download wrote waits for no chunk's import on the
//!   encoding thread either. A gather over bytes a scatter wrote into a chunk that takes no
//!   import is staged too, the scatter's bytes written by the CPU in a host step before the
//!   gather's.
//!   - A gather leaves its work to the host step of its batch (`Deferred::GatherStage`), before
//!     the batch is submitted: the step takes a region of the command buffer's arena as long as
//!     the copy's bytes, host-visible memory of the runtime's own, so the region's memory is made
//!     on neither the encoding nor the committing thread; fills it by the CPU from the pages, each
//!     piece at its byte; and records one `vkCmdCopyBuffer` of the bytes the runs cover from it
//!     into the buffer (at most [`MOST_REGIONS`] regions a command, a byte two runs name taking
//!     the later's, as the CPU wrote it last), at the copy's place among the encoder's commands.
//!     Its uses, noted at the call there, are the chunks' bytes it reads, by their keys, and the
//!     buffer's writes, so hazard tracking orders it as it orders a copy through the imports, the
//!     later writers of those bytes among them. It reads what Metal's copy would: the step runs
//!     once the batch is otherwise ready, after its event waits are met and every batch before it
//!     in its order was submitted, so the CPU's writes before the commit, or before a signal its
//!     batch waits for, are in the pages. Where some use but a copy's read may write the copy's
//!     host bytes (a buffer or heap over host pages lies over them, which it does until every
//!     command buffer that used it completed, or a copy into those chunks was recorded:
//!     `RamOnDevice::written_over`), the GPU may be writing them: the step then waits until every
//!     `VkQueue` completed what was submitted before it, as a host step that reads what the GPU
//!     writes waits (`scheduler/mod.rs`), and reads after; one where that is so as it is encoded
//!     goes in a part of its own (above), since work before it in its own command buffer is not
//!     submitted before its step. A region the device has no memory for fails the command buffer
//!     at the step. A
//!     rebind's encoding again of its encoder (`command_buffer/reencode.rs`) records the copy
//!     through the imports in its place. Counted: `WorkCounts::host_ram_copies_staged`,
//!     `host_ram_staged_fills_gated`, `host_ram_staged_fill_nanos`.
//!   - A scatter leaves its work to two host steps (`Deferred::ScatterStage`,
//!     `Deferred::ScatterDrain`). The host step of the copy's batch, which waits for no work on the
//!     GPU, takes a region of the command buffer's arena as long as the copy's bytes and records
//!     one `vkCmdCopyBuffer` of the bytes the runs cover from the buffer into it (at most
//!     [`MOST_REGIONS`] regions a command), at the copy's place among the encoder's commands: so
//!     the region's memory is made on neither the encoding nor the committing thread. The copy
//!     then goes on in a part of its own, as work whose host step reads what the GPU wrote does
//!     (`encoder.rs`'s segments): that batch's host step waits until every `VkQueue` completed
//!     what was submitted before it, and copies each run's bytes from the region into the pages,
//!     in the runs' order (a host byte two runs name keeps the later's), before the batch is
//!     submitted. So the pages hold the bytes once the command buffer completed, as Metal's
//!     completion says, and every command after the copy, in its command buffer or a later one
//!     that runs after it, runs after the CPU wrote them. Its uses: the buffer's read where the
//!     GPU's copy lies, and the chunks' bytes it writes, by their keys, at the start of the part
//!     after it, where the CPU's copy lies, so hazard tracking orders the later readers and
//!     writers of those bytes after that write, and the RAM's log is told of them as of a copy
//!     through the imports. Until the second step took its gate, the two batches give way to a
//!     later commit of another queue that shares no fence with them, which goes first and reads
//!     the bytes from before the copy, as an x86 Mac's GPU may take it first
//!     (`scheduler/order.rs`, **Held batches give way**); one that waits for an event the copy's
//!     command buffer signals after it runs after it. A region the device has no memory for
//!     fails the command buffer at that host step. A rebind's encoding again of its encoder
//!     (`command_buffer/reencode.rs`) stages it again in its place, from where the buffer lies
//!     then (`HostRamCall::staged_again`), so its parts are the first's; the first's steps go
//!     unrun. Counted: `WorkCounts::host_ram_scatters_staged` (the encoding again's too),
//!     `host_ram_staged_drain_nanos`.
//!
//!   Any other copy encoded again (`replay.rs`) takes the imports' path.
//! - **Lists of runs.** Where each listed import costs every submission and the RAM's chunks are
//!   the device's dma-bufs of the RAM's source (RADV's: `host_ram.rs`'s `Plan::listed`), and the
//!   RAM came with a source of run lists (`HostRam::with_dma_buf_of_runs`), a copy neither imports
//!   a chunk not imported yet nor asks for it (`RamOnDevice::copy_imports`): its ranges of every
//!   such chunk, widened to whole host pages, those that then share or touch a page one
//!   ([`page_runs`]), are taken by address across the chunks and imported as one dma-buf of that
//!   source for each list of at most the source's count of runs and one chunk's bytes
//!   ([`run_lists`]), made on the encoding thread, each run's bytes in the list's `VkBuffer` from
//!   where the runs before it end (`RamChunk::offset_of`), held by the command buffer until it
//!   completed (`WorkCounts::host_ram_run_list_imports`, `run_imports_live`): a tile of 1,800 pages
//!   anywhere in the RAM is one list or two. Each list is kept past the copy for the copies after
//!   it, which take it for every run of theirs one of its runs holds whole and import nothing for
//!   those (`host_ram/kept.rs`: at most 64 MiB and 16 lists, each released once no copy took it for
//!   2 s and no command buffer holds it). So copies pin their own pages while they run or their
//!   lists are kept, and none of the chunks' others, where a chunk's import pins every page of it
//!   and the source's dma-buf allocates the pages of the guest's file the guest never touched, for
//!   good. The chunks already imported take their own imports. Where that cannot be (the allocation
//!   limit's margin, `RuntimeDriverDefect::AliasedImportBlitsUnordered`, a run that is the whole
//!   chunk or passes a list's bytes, a list's import that fails) the copy takes the chunks as
//!   above.
//! - **Runs of words: one dispatch for the imports.** Where every run also ends on a word, and lies
//!   at a word of the buffer's memory ([`in_words`]: a surface's whole pages, or the rows of one of
//!   4-byte texels), the pieces become a table of runs of at most a 4 KiB page each, each the
//!   import it lies in, its first word there, its first word in the buffer and its words
//!   ([`plan`]: whole pages an entry a page), uploaded into the command buffer's arena, and the
//!   runs one dispatch of `bind::pages_copy_module` (a gather or a scatter) over the imports' whole
//!   `VkBuffer`s (the chunks', or the imports the copy took of its own pages: a range's alone, a
//!   list's), an array of up to the device's `PagesCopyModule::imports` of them (30 where it
//!   pushes descriptors and indexes arrays of storage buffers, 1 where it indexes none), the
//!   buffer's bytes and the table: a dispatch for each that many imports the runs touch, never one
//!   region per run, and no other word of the runs' pages read or written. A dispatch writes its
//!   own imports alone into the array where the device enables `descriptorBindingPartiallyBound`
//!   (the array partially bound: an element no invocation reads need not be valid, so a dispatch
//!   over one import of an array of 30 writes 3 descriptors, not 32); elsewhere the elements past
//!   them hold its first import again, so every descriptor of the array is valid.
//!   A copy whose bytes pass the device's storage range, or whose runs of one array pass its
//!   workgroup count, takes more (each window of the buffer's bytes, each part of the runs).
//!   Counted: `WorkCounts::host_ram_copies`, `host_ram_copy_bytes` and
//!   `host_ram_copy_dispatches`. Such a copy's plan is kept ([`KeptPlan`], [`RamCopies::kept`]):
//!   the same copy after it (its direction, runs' form, buffer and runs, the runs compared one by
//!   one) takes its table and dispatches where the buffer's `VkBuffer`, byte and pages, the
//!   device's figures and the copy's imports (by [`RamChunk::serial`], which no other import has)
//!   are the plan's; else it plans anew, the steps it took kept for the full path, none taken
//!   twice. The call finds the plan as it is made ([`RamCopies::kept_for`]): the runs as given,
//!   merged and compared as they are merged ([`merges_to`]), no vector of them made; the call
//!   shares the plan's runs, and its recording takes the plan only while the RAM keeps it.
//!   Every step a copy takes each time is taken either way: `blit_at`, the chunks' takes, the
//!   imports, the binds and the uses. The table is uploaded once a plan: at its first reuse, into
//!   a buffer the plan keeps, which every reuse binds ([`KeptPlan::table_buffer`],
//!   `WorkCounts::host_ram_copy_tables_kept`). A RAM keeps at most [`MOST_PLANS`] on a device, of
//!   at most [`MOST_PLAN_BYTES`] together (`WorkCounts::host_ram_copy_plans_reused`), and of the
//!   device's memory only their tables' buffers, until the command buffers that bound them
//!   complete.
//! - **Other runs: regions.** Where a run ends off a word (a row of 1- or 2-byte texels) or lies at
//!   a byte of the buffer off a word, or the runs write a destination byte twice (a scatter's that
//!   name a host byte twice, a gather's that name a byte of the buffer twice: [`share_a_byte`]),
//!   each piece is a region of `vkCmdCopyBuffer` between its import (the chunk's, or the pages it
//!   covers imported alone, as above: whole pages, imported as now) and the buffer, which takes
//!   byte offsets and sizes: the copy reads or writes the runs' bytes and no other byte of their
//!   pages, a row's padding and the pages around a surface left as they were. The regions of one
//!   import are one command, at most [`MOST_REGIONS`] each. Runs that name a destination byte twice
//!   take a round of commands for each ([`rounds`]), no byte written by two regions of one command,
//!   the later run's bytes kept; and a buffer over host pages the runs' bytes overlap is refused,
//!   no region reading and writing one byte. Counted: `WorkCounts::host_ram_copies`,
//!   `host_ram_copy_bytes` and `host_ram_copy_regions`.
//! - **Ordered as a copy between the buffer and a no-copy buffer over the pages.** Each chunk a
//!   copy uses has a key of its own, its uses' bytes those of the chunk, which hazard tracking
//!   places over the chunk's host bytes (`host_aliases.rs`'s placements), as it places every
//!   no-copy buffer and heap over the bytes it lies over: so the copy's uses (a read of the
//!   chunks' bytes it gathers, a write of those it scatters, and the buffer's write or read) are
//!   ordered, in the encoder and between encoders, command buffers and queues, against every
//!   other use of the same host bytes: the other copies', and those of a no-copy object over the
//!   RAM's own addresses. A chunk's uses name the bytes the runs touch, joined where they are more
//!   than [`MOST_RANGES`] (the gaps between the closest taken too), so a copy of thousands of runs
//!   takes a few uses, and orders more than its bytes need, never less.
//! - **Seen as Metal shows a no-copy buffer's bytes.** The chunks are host-coherent imports of the
//!   pages: the CPU's writes before the commit are seen by the copy (the submission makes them
//!   visible), and the GPU's are in the pages once the command buffer completed (every batch ends
//!   with a barrier to the host). Where the RAM came with a log of the writes to its pages
//!   (`HostRam::with_dirty_pages`), the chunk's key is the log's, so a scatter's writes are
//!   reported to it as the GPU's in place (`dirty_pages.rs`): every copy of those pages (a copied
//!   host buffer's, a shadow's, whose logs are the RAM's where their runs are) copies them in.
//! - **What is refused** ([`Problem::HostRam`]): a run that is empty, starts off a 4-byte word, or
//!   lies outside the RAM's ranges (or, one after another, past `u64`); a RAM the device does not
//!   use; of runs one after another, bytes of the buffer that start off a 4-byte word of its memory
//!   (runs each at a byte of their own are copied by regions there); a copy by regions with a
//!   buffer over host pages its bytes' span overlaps. A device without
//!   `VK_EXT_external_memory_host` uses no RAM that comes with no dma-buf source it imports
//!   (`Unsupported`, naming it); the buffer's bytes past
//!   its length are a range past it ([`Problem::RangeOutOfBounds`]: of runs each at a byte of their
//!   own, the first run's past it). A chunk the device cannot import fails the copy with the
//!   import's error.

use super::{ChunkSpans, CopyImports, CopyTake, HostRam, RamChunk, RamOnDevice};
use crate::blit::{bytes_in, reads_by, writes_by, BlitCommandEncoder, TRANSFER};
use crate::buffer::{Buffer, CallerMemory, HostPages};
use crate::command_buffer::CommandBuffer;
use crate::deferred::Deferred;
use crate::device::lock;
use crate::dirty_pages::PagesLog;
use crate::encoder::{EncoderCore, RecordedSegment};
use crate::error::{internal, invalid, Error, Problem};
use crate::owned_kernel::{OwnedBinds, OwnedKernel, OwnedModule, OwnedSet};
use crate::replay::BlitCall;
use crate::resource::{ObjectId, ResourceKey};
use crate::tracker::{Access, Use, UseRange};
use crate::upload::Upload;
use crate::values::{CpuCacheMode, HazardTrackingMode, ResourceOptions, StorageMode};
use crate::Device;
use ash::vk;
use metal2vulkan::bind::{PagesCopy, PagesCopyModule};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Instant;

/// The unit of a copy's runs: whole 4 KiB pages at 4 KiB-aligned host addresses (a guest's page;
/// every host page and every chunk is whole such pages).
pub(crate) const PAGE: u64 = PagesCopyModule::PAGE_BYTES as u64;

/// The most ranges of one chunk's bytes a copy's uses name; past them the closest are joined.
/// So a copy of thousands of runs keeps the command buffer's writes apart (`dirty_pages.rs`
/// keeps 4,096 writes of a command buffer and 64 ranges of a log apart, past which it takes every
/// copy, or every page, as written).
const MOST_RANGES: usize = 16;

/// The most ranges of a chunk not imported yet a copy imports alone while the device imports the
/// chunk (`RamOnDevice::for_copy`): each import is a fixed cost on the encoding thread (a 2 MiB
/// run's took 0.75 to 0.83 ms on the Radeon 780M and RX 7900 XTX), so a copy whose pages there
/// are more ranges than this imports the chunk, or waits for its import under way.
pub(crate) const MOST_ALONE: usize = 16;

/// The bytes between two of a chunk's ranges a copy imports alone below which they are one
/// import ([`alone_ranges`]): a huge page's.
const JOIN_GAP: u64 = 2 << 20;

/// A RAM's copies on one device: the keys they use its chunks' bytes by, one per chunk, made at
/// the first copy over it, placed over the chunk's host bytes for hazard tracking and, where the
/// RAM came with a log, kept as that log's (the module docs), their places let go of as the RAM's
/// use on the device ends ([`RamCopies::unplace`]); and the plans kept of its copies by
/// dispatches ([`KeptPlan`]), which go with the use, whose chunks never change.
pub(crate) struct RamCopies {
    keys: Box<[OnceLock<ChunkKey>]>,
    plans: Mutex<KeptPlans>,
}

/// One chunk's key, and the log of its pages' writes its uses report to.
struct ChunkKey {
    key: ResourceKey,
    /// Kept while the key lives: the device's table of logs holds it weakly.
    _log: Option<Arc<PagesLog>>,
}

impl RamCopies {
    /// The copies of a RAM of `chunks` chunks, no key made yet and no plan kept.
    pub(crate) fn new(chunks: usize) -> RamCopies {
        RamCopies {
            keys: (0..chunks).map(|_| OnceLock::new()).collect(),
            plans: Mutex::default(),
        }
    }

    /// The plan kept of `call`'s copy, taken now ([`KeptPlans::taken`]): of its direction, its
    /// runs' form, its buffer ([`BufferKey`]) and its runs, compared one by one outside the
    /// lock. Whether the plan holds for the call is for [`BlitCommandEncoder::record_kept`].
    pub(crate) fn kept(&self, call: &HostRamCall) -> Option<Arc<KeptPlan>> {
        let candidates: Vec<Arc<KeptPlan>> = (lock(&self.plans).plans.iter())
            .filter(|plan| plan.may_be(call))
            .cloned()
            .collect();
        let plan = (candidates.into_iter())
            .find(|plan| Arc::ptr_eq(&plan.runs, &call.runs) || plan.runs == call.runs)?;
        lock(&self.plans).taken(&plan);
        Some(plan)
    }

    /// The plan kept of a copy of `copy`'s direction between the RAM and `buffer` of the runs
    /// `given`, taken now ([`KeptPlans::taken`]): one of the same form and buffer
    /// ([`BufferKey`]) whose runs `given`'s merge to ([`merges_to`]), compared outside the lock.
    pub(crate) fn kept_for(
        &self,
        copy: PagesCopy,
        buffer: &Buffer,
        given: Given<'_>,
    ) -> Option<Arc<KeptPlan>> {
        let key = buffer_key(buffer);
        let consecutive = given.consecutive();
        let first = given.runs().next().flatten().map(|(_, run)| run.start);
        let candidates: Vec<Arc<KeptPlan>> = (lock(&self.plans).plans.iter())
            .filter(|plan| {
                plan.copy == copy
                    && plan.consecutive == consecutive
                    && plan.buffer == key
                    && plan.runs.first().map(|(_, run)| run.start) == first
            })
            .cloned()
            .collect();
        let plan = (candidates.into_iter()).find(|plan| merges_to(given, &plan.runs))?;
        lock(&self.plans).taken(&plan);
        Some(plan)
    }

    /// Whether the RAM keeps `plan` still: the very plan, not one like it.
    fn keeps(&self, plan: &Arc<KeptPlan>) -> bool {
        (lock(&self.plans).plans.iter()).any(|kept| Arc::ptr_eq(kept, plan))
    }

    /// Keep `plan` ([`KeptPlans::keep`]).
    fn keep(&self, plan: KeptPlan) {
        lock(&self.plans).keep(plan);
    }

    /// The RAM's use on `device` ended (`HostRamUse`'s drop): the keys lie over no host bytes
    /// from now on, so no later use of those bytes takes theirs. The commands recorded before
    /// keep the uses they took.
    pub(crate) fn unplace(&self, device: &Device) {
        let aliases = &device.shared().aliases;
        for key in self.keys.iter().filter_map(OnceLock::get) {
            aliases.unplace(key.key);
        }
    }
}

/// How a copy takes the chunks it touches ([`BlitCommandEncoder::takes`]): whether one of them is
/// taken through the CPU while the device's thread imports it ([`CopyTake::ThroughCpu`]), only
/// through the CPU ([`CopyTake::CpuOnly`]), or by waiting for its import ([`CopyTake::Waits`]).
#[derive(Clone, Copy, Debug, Default)]
struct Takes {
    through_cpu: bool,
    cpu_only: bool,
    waits: bool,
}

impl Takes {
    /// Whether the copy is staged through the CPU (the module docs' "First touch"): over a chunk
    /// that takes no import while it runs ([`CopyTake::CpuOnly`]: refused, or held back by
    /// objects' own imports of its pages), always; else over a chunk not imported yet that the
    /// device's thread imports ([`CopyTake::ThroughCpu`]) and over none whose import only it
    /// would make ([`CopyTake::Waits`]).
    fn staged(self) -> bool {
        self.cpu_only || (self.through_cpu && !self.waits)
    }

    /// How the copy is staged, where it is ([`Takes::staged`]): a gather some use but a copy's
    /// read may write the host bytes of (`written`: [`RamOnDevice::written_over`], a copy into its
    /// chunks recorded before, its own command buffer's among them) goes in a part of its own
    /// after the GPU completed the work before it ([`Staging::after_the_gpu`]), never through the
    /// imports, whose chunks it would import or wait for on the encoding thread. A scatter's CPU
    /// copy waits for every batch submitted before it to complete, its own command buffer's
    /// before it among them.
    fn staging(self, written: impl FnOnce() -> bool) -> Option<Staging> {
        self.staged().then(|| Staging {
            cpu_only: self.cpu_only,
            after_the_gpu: written(),
        })
    }
}

/// The most plans of copies a RAM keeps on a device ([`KeptPlans`]).
const MOST_PLANS: usize = 8;

/// The most bytes the plans a RAM keeps on a device hold together ([`KeptPlans`]): a plan of
/// more is not kept.
const MOST_PLAN_BYTES: usize = 4 << 20;

/// The device's figures a copy's dispatches are planned for: the bytes of a window of the buffer
/// (its storage range less its offset alignment, in whole pages), that alignment, the most
/// workgroups of a dispatch, and the imports the pages copy's array holds
/// (`PagesCopyModule::imports`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Window {
    bytes: u64,
    align: u64,
    most_groups: u32,
    array: usize,
}

impl Window {
    /// The pages copy of `copy`'s direction on `device` (made at its first use), and the figures
    /// its dispatches are planned for: `figures`' window bytes, alignment and most workgroups,
    /// and its array.
    fn of(
        device: &Device,
        copy: PagesCopy,
        (bytes, align, most_groups): (u64, u64, u32),
    ) -> Result<(OwnedKernel, Window), Error> {
        let kernel = device.owned_kernel(OwnedModule::PagesCopy(copy))?;
        let window = Window {
            bytes,
            align,
            most_groups,
            array: kernel.first_buffers as usize,
        };
        Ok((kernel, window))
    }
}

/// A buffer as a kept plan names it ([`KeptPlan::buffer`]): its identity, its device
/// generation's (the address of that generation's `DeviceShared`) and its length.
type BufferKey = (ObjectId, usize, u64);

/// `buffer`'s [`BufferKey`].
fn buffer_key(buffer: &Buffer) -> BufferKey {
    let generation = std::ptr::from_ref(buffer.generation().shared()).addr();
    (buffer.object_id(), generation, buffer.length())
}

/// A copy's table as its dispatches bind it ([`BlitCommandEncoder::record_dispatches`]): its
/// bytes, uploaded to the command buffer's arena, or the buffer a kept plan holds them in, and
/// their length ([`KeptPlan::table_buffer`]).
enum Table<'a> {
    Upload(&'a [u8]),
    Kept(&'a Buffer, u64),
}

/// What a copy by dispatches planned ([`BlitCommandEncoder::record_host_ram_copy`]), kept for the
/// same copy after it ([`RamCopies::kept`]), which takes it
/// ([`BlitCommandEncoder::record_kept`]) where its buffer's `VkBuffer`, byte and pages, the
/// device's figures and the copy's imports are as they were. The plan holds no import nor the
/// copy's buffer: their serial numbers and identities only (the pages weakly); of the device's
/// memory it keeps its table's buffer alone, once reused ([`KeptPlan::table_buffer`]).
pub(crate) struct KeptPlan {
    /// The copy's direction, its runs' form (one after another, from this byte of the buffer),
    /// its buffer and its runs: the copy that takes the plan is this one.
    copy: PagesCopy,
    consecutive: Option<u64>,
    buffer: BufferKey,
    runs: Arc<[Run]>,
    /// The buffer's `VkBuffer` and the byte of it the copy's bytes start at, as `blit_at` gave
    /// them, and the host pages it lay in (a no-copy buffer's, or its no-copy heap's), held
    /// weakly so their memory goes as they go while no other pages can take their place: a
    /// rebind or a detach lays it in others. A driver may give a new `VkBuffer` an old one's
    /// handle; the plan names no `VkBuffer` (the binds take the buffer's as it is at each copy).
    blit: (vk::Buffer, u64),
    pages: Option<Weak<HostPages>>,
    window: Window,
    /// The buffer's bytes the copy reads or writes, and the bytes it copies.
    span: Range<u64>,
    total: u64,
    /// The bytes of each chunk the copy touches ([`touched_ranges`]), and those joined to at most
    /// [`MOST_RANGES`] a chunk, each with its chunk, for its uses.
    ranges: BTreeMap<usize, Vec<Range<u64>>>,
    touched: Vec<(usize, Range<u64>)>,
    /// The serial numbers of the imports the copy read or wrote through ([`RamChunk::serial`]), in
    /// the order [`RamOnDevice::copy_imports`] gave them, and the chunks each holds bytes of
    /// (`CopyImports::of_chunk`).
    imports: Vec<u64>,
    of_chunk: Vec<(usize, usize)>,
    /// The dispatches' table and the dispatches ([`plan`]).
    table: Vec<u8>,
    dispatches: Vec<Dispatch>,
    /// The table in a buffer of its own on the device, made at the plan's first reuse
    /// ([`KeptPlan::table_buffer`]), which each reuse binds in place of an upload: of the arena's
    /// memory (`upload.rs`), its bytes the table's, written before any command reads them and
    /// never again. Each command buffer that binds it holds it until it completes, so a plan let
    /// go of frees it once those did. It counts in [`MOST_PLAN_BYTES`] as the table does.
    table_kept: OnceLock<Buffer>,
}

impl KeptPlan {
    /// The table's buffer ([`KeptPlan::table_kept`]), made on `device` and written now where the
    /// plan has none (`WorkCounts::host_ram_copy_tables_kept`). The plan is the RAM's on
    /// `device`'s generation (`RamOnDevice`), so the buffer is that generation's: a recovery
    /// carries the RAM's use to a new `RamOnDevice`, whose plans start anew.
    fn table_buffer(&self, device: &Device) -> Result<&Buffer, Error> {
        if let Some(buffer) = self.table_kept.get() {
            return Ok(buffer);
        }
        let options = ResourceOptions::new(
            StorageMode::Shared,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Untracked,
        );
        let (buffer, _) = device.unzeroed_buffer_with(self.table.len() as u64, options)?;
        let mapped = (buffer.mapped_contents())
            .ok_or_else(|| internal("a host RAM copy's kept table in an unmapped buffer"))?;
        // SAFETY: the buffer's mapping holds the table's bytes (as long as it), and no command
        // uses the buffer yet: it is the plan's alone until set below.
        unsafe {
            std::ptr::copy_nonoverlapping(self.table.as_ptr(), mapped.as_ptr(), self.table.len());
        }
        // Another encoder's first reuse of the plan may have set one meanwhile: that one stays.
        if self.table_kept.set(buffer).is_ok() {
            device.shared().counts.host_ram_copy_table_kept();
        }
        (self.table_kept.get()).ok_or_else(|| internal("a host RAM copy's kept table"))
    }

    /// The bytes it holds, about.
    fn bytes(&self) -> usize {
        let range = std::mem::size_of::<Range<u64>>();
        let ranges: usize = self.ranges.values().map(Vec::len).sum();
        std::mem::size_of::<KeptPlan>()
            + std::mem::size_of_val(&*self.runs)
            + (ranges + self.touched.len()) * (range + 8)
            + std::mem::size_of_val(&*self.imports)
            + std::mem::size_of_val(&*self.of_chunk)
            + self.table.len()
            + std::mem::size_of_val(&*self.dispatches)
    }

    /// Whether it may be the plan of `call` by the fields read at once: its direction, form,
    /// buffer, and its runs' count, first and last.
    fn may_be(&self, call: &HostRamCall) -> bool {
        self.copy == call.copy
            && self.consecutive == call.consecutive
            && self.buffer == buffer_key(&call.buffer)
            && self.runs.len() == call.runs.len()
            && self.runs.first() == call.runs.first()
            && self.runs.last() == call.runs.last()
    }

    /// Whether it is the plan of the same copy as `other` (its direction, form, buffer and runs).
    fn same_copy(&self, other: &KeptPlan) -> bool {
        self.copy == other.copy
            && self.consecutive == other.consecutive
            && self.buffer == other.buffer
            && self.runs == other.runs
    }
}

/// The plans a RAM keeps on a device of its copies by dispatches, the one taken last first: at
/// most [`MOST_PLANS`], of at most [`MOST_PLAN_BYTES`] together, the one taken longest ago dropped
/// first.
#[derive(Default)]
struct KeptPlans {
    plans: Vec<Arc<KeptPlan>>,
    bytes: usize,
}

impl KeptPlans {
    /// Keep `plan` first, in place of a plan of the same copy, dropping the plans taken longest
    /// ago past the bounds; not one of more than [`MOST_PLAN_BYTES`].
    fn keep(&mut self, plan: KeptPlan) {
        let bytes = plan.bytes();
        if bytes > MOST_PLAN_BYTES {
            return;
        }
        if let Some(at) = self.plans.iter().position(|kept| kept.same_copy(&plan)) {
            let gone = self.plans.remove(at);
            self.bytes -= gone.bytes();
        }
        self.plans.insert(0, Arc::new(plan));
        self.bytes += bytes;
        while self.plans.len() > MOST_PLANS || self.bytes > MOST_PLAN_BYTES {
            let Some(gone) = self.plans.pop() else {
                break;
            };
            self.bytes -= gone.bytes();
        }
    }

    /// Move `plan` first: taken now.
    fn taken(&mut self, plan: &Arc<KeptPlan>) {
        if let Some(at) = self.plans.iter().position(|kept| Arc::ptr_eq(kept, plan)) {
            let plan = self.plans.remove(at);
            self.plans.insert(0, plan);
        }
    }
}

/// What a copy took of its steps before it found that the plan kept of it did not hold
/// ([`BlitCommandEncoder::record_kept`]), for the full path to go on from, taking none twice: the
/// buffer's `blit_at` (with the byte of the buffer it was for), the chunks' takes, the imports.
#[derive(Default)]
struct Early {
    blit: Option<(u64, (vk::Buffer, u64))>,
    takes: Option<Takes>,
    imports: Option<CopyImports>,
}

/// How a copy went by the plan kept of it ([`BlitCommandEncoder::record_kept`]).
enum Kept {
    /// Recorded by it.
    Recorded,
    /// Not taken; what was taken of the copy's steps.
    Not(Early),
}

impl RamOnDevice {
    /// The key of chunk `slot`'s bytes on `device`: made at the first copy over the chunk, and
    /// placed over its host bytes, its uses reported to the RAM's log where it has one.
    fn chunk_key(&self, device: &Device, slot: usize) -> ResourceKey {
        let made = self.copies.keys[slot].get_or_init(|| {
            let shared = device.shared();
            let span = &self.slots[slot].span;
            let key = ResourceKey::new();
            shared
                .aliases
                .place(key, (span.start, span.end - span.start));
            let log = (self.dirty()).and_then(|dirty| {
                PagesLog::new((span.start, span.end - span.start), Some(dirty), None)
            });
            if let Some(log) = &log {
                shared.dirty.logged_in_place(key, log);
            }
            ChunkKey { key, _log: log }
        });
        made.key
    }

    /// Whether some use but a copy's read may write the host bytes `touched` (each with its
    /// chunk): a copy into a chunk's bytes was recorded ([`RamOnDevice::copy_wrote`]), or a key
    /// lies over them other than the chunks' own keys, which only those copies' reads used
    /// (`HostAliases::others_over`): a buffer over host pages, made or rebound there, or a
    /// heap's own, which lies there until every command buffer that used it let it go. A copy
    /// into a chunk the device refuses whatever it holds (`ChunkSlot::refused`) wrote by the CPU,
    /// in a host step before every later copy's (a staged scatter's drain): the GPU writes none of
    /// its pages.
    fn written_over(&self, device: &Device, touched: &[(usize, Range<u64>)]) -> bool {
        let aliases = &device.shared().aliases;
        (touched.iter()).any(|(slot, bytes)| {
            let wrote = self.copies_wrote(*slot) && !self.slots[*slot].refused;
            let reads_only = |key: ResourceKey| {
                !wrote && (self.copies.keys[*slot].get()).is_some_and(|made| made.key == key)
            };
            wrote || aliases.others_over(bytes.start, bytes.end, reads_only)
        })
    }
}

/// A gather staged through the CPU (the module docs' "First touch"), its work left to the host
/// step of its batch ([`stage`](Self::stage), `Deferred::GatherStage`), before the batch is
/// submitted: it takes a region of its command buffer's arena, copies each piece's host bytes
/// into it, and records the GPU's copy of the bytes the pieces cover from there into the buffer.
/// Held by its segment until its command buffer completes.
pub(crate) struct StagedGather {
    ram: Arc<RamOnDevice>,
    /// The buffer's `VkBuffer` and the byte of it the copy's first byte is at (`blit_at`, whose
    /// command buffer keeps it live until it completes).
    destination: (vk::Buffer, u64),
    /// The copy's bytes: the region's length.
    length: u64,
    /// The bytes of the copy the pieces cover ([`covered`]): the GPU's regions, each byte once.
    covered: Vec<Range<u64>>,
    /// Each piece's first host byte, its length and its byte of the region, in the runs' order:
    /// where two name one byte of the buffer, the later's bytes stay.
    pieces: Vec<(u64, u64, u64)>,
    /// The host bytes the pieces lie in, joined as the copy's uses name them, each with its
    /// chunk.
    touched: Vec<(usize, Range<u64>)>,
    /// The host step found that the GPU may write them ([`StagedGather::waits_for_the_gpu`]).
    gated: AtomicBool,
    /// How [`Takes::staging`] staged it.
    staging: Staging,
}

/// How a copy is staged through the CPU ([`Takes::staging`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Staging {
    /// It is over a chunk that takes no import while it runs ([`CopyTake::CpuOnly`]): an encoding
    /// again of its encoder could not copy it through the imports, so the encoder keeps no log one
    /// encodes it again from (`Deferred::recorded_by_an_encoding_again`).
    cpu_only: bool,
    /// A gather whose host bytes some use but a copy's read may write as it is encoded
    /// ([`RamOnDevice::written_over`]): its host step goes in a part of its own, after the work
    /// before it in its command buffer was submitted, and waits for every `VkQueue` to complete
    /// what was submitted before it (`Deferred::gpu_written`), so it reads the bytes a command
    /// before it in its own command buffer wrote. Its encoder keeps no log either: an encoding
    /// again through the imports would start no part, so its parts would not be the first's.
    after_the_gpu: bool,
}

impl StagedGather {
    /// Whether its host step goes in a part of its own after the work before it completed
    /// ([`Staging::after_the_gpu`]).
    pub(crate) fn after_the_gpu(&self) -> bool {
        self.staging.after_the_gpu
    }

    /// Whether an encoding again of its encoder copies it through the imports in its place: not
    /// where one of its chunks takes no import while it runs ([`Staging::cpu_only`]), nor where it
    /// goes in a part of its own ([`Staging::after_the_gpu`]), which that copy would not start.
    pub(crate) fn encoded_again_through_imports(&self) -> bool {
        !self.staging.cpu_only && !self.staging.after_the_gpu
    }

    /// Whether its host step waits for every batch submitted before it to complete: some use but
    /// a copy's read may write its host bytes ([`RamOnDevice::written_over`]), which the GPU may
    /// be doing. Else no work on the GPU may write them, and the CPU's writes before the commit
    /// are in the pages, so the step reads them at once.
    pub(crate) fn waits_for_the_gpu(&self, device: &Device) -> bool {
        let waits = self.ram.written_over(device, &self.touched);
        if waits {
            self.gated.store(true, Ordering::Relaxed);
        }
        waits
    }

    /// Its host step, on the steps' thread before its batch is submitted: the region of the
    /// arena, each piece's bytes copied into it by the CPU, counted on the device, and the GPU's
    /// copy from it into the buffer, at most [`MOST_REGIONS`] regions a command, after every
    /// command before it (`EncoderCore::segment`'s barrier) and before every later one. Refused,
    /// failing the command buffer, where the device has no memory for the region.
    pub(crate) fn stage(&self, command_buffer: &CommandBuffer) -> Result<RecordedSegment, Error> {
        let device = command_buffer.device().clone();
        let region = (command_buffer.state().uploads).unset(&device, self.length)?;
        let started = Instant::now();
        for &(host, length, at) in &self.pieces {
            // SAFETY: the piece's host bytes lie in the RAM's ranges (`pieces`), which stay
            // mapped and readable while `ram` holds the RAM (`HostRam::new`); the region is
            // memory of the runtime's own, apart from them.
            unsafe { region.copy_in(at, host as *const u8, length) };
        }
        let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        (device.shared().counts).host_ram_staged_filled(nanos, self.gated.load(Ordering::Relaxed));
        let (destination, at) = self.destination;
        let source = region.raw();
        let regions: Vec<vk::BufferCopy> = (self.covered.iter())
            .map(|bytes| vk::BufferCopy {
                src_offset: region.offset() + bytes.start,
                dst_offset: at + bytes.start,
                size: bytes.end - bytes.start,
            })
            .collect();
        let mut core = EncoderCore::segment(command_buffer, true)?;
        core.command(TRANSFER, &[], |functions, raw| {
            for part in regions.chunks(MOST_REGIONS) {
                // SAFETY: the segment's command buffer is recording, outside a render pass; the
                // arena's region and the buffer are live until the command buffer completes (the
                // arena, and `blit_at`); each region lies inside the arena's (as long as the
                // copy's bytes) and inside the buffer's bytes (`bytes_in`), and no byte is written
                // by two (`covered`).
                unsafe { functions.cmd_copy_buffer(raw, source, destination, part) };
            }
        });
        core.end_segment()
    }
}

/// A scatter staged through the CPU (the module docs' "First touch"), its work left to two host
/// steps: the first, of the batch that holds the copy, takes `region` of its command buffer's
/// arena and records the GPU's copy of the bytes the runs cover from the buffer into it
/// ([`stage`](Self::stage), `Deferred::ScatterStage`); the second, of the batch after it, once
/// every batch submitted before it completed, copies each piece's bytes from the region into the
/// pages ([`drain`](Self::drain), `Deferred::ScatterDrain`). Held by its segments until its
/// command buffer completes.
pub(crate) struct StagedScatter {
    /// Kept until the CPU's copy: the pages stay mapped while the RAM lives (`HostRam::new`).
    _ram: Arc<RamOnDevice>,
    /// The buffer's `VkBuffer` and the byte of it the copy's first byte is at (`blit_at`, whose
    /// command buffer keeps it live until it completes).
    source: (vk::Buffer, u64),
    /// The copy's bytes: the region's length.
    length: u64,
    /// The bytes of the copy the pieces cover ([`covered`]): the GPU's regions, each byte once.
    covered: Vec<Range<u64>>,
    /// Each piece's first host byte, its length and its byte of the region, in the runs' order:
    /// where two name one host byte, the later's bytes stay.
    pieces: Vec<(u64, u64, u64)>,
    /// The region, once the first step took it.
    region: OnceLock<Upload>,
}

impl StagedScatter {
    /// The first host step, on the steps' thread before the copy's batch is submitted: the
    /// region of the arena, and the GPU's copy into it, at most [`MOST_REGIONS`] regions a
    /// command, after every command before it (`EncoderCore::segment`'s barrier) and before
    /// every later one. Refused, failing the command buffer, where the device has no memory for
    /// the region.
    pub(crate) fn stage(&self, command_buffer: &CommandBuffer) -> Result<RecordedSegment, Error> {
        let device = command_buffer.device().clone();
        let region = (command_buffer.state().uploads).unset(&device, self.length)?;
        let (source, at) = self.source;
        let target = region.raw();
        let regions: Vec<vk::BufferCopy> = (self.covered.iter())
            .map(|bytes| vk::BufferCopy {
                src_offset: at + bytes.start,
                dst_offset: region.offset() + bytes.start,
                size: bytes.end - bytes.start,
            })
            .collect();
        let mut core = EncoderCore::segment(command_buffer, true)?;
        core.command(TRANSFER, &[], |functions, raw| {
            for part in regions.chunks(MOST_REGIONS) {
                // SAFETY: the segment's command buffer is recording, outside a render pass; the
                // buffer and the arena's region are live until the command buffer completes
                // (`blit_at`, the arena); each region lies inside the buffer's bytes (`bytes_in`)
                // and inside the arena's (as long as the copy's bytes), and no byte is written by
                // two (`covered`).
                unsafe { functions.cmd_copy_buffer(raw, source, target, part) };
            }
        });
        if self.region.set(region).is_err() {
            return Err(internal("a staged scatter's staging taken twice"));
        }
        core.end_segment()
    }

    /// The second host step, once every batch submitted before its own completed, the GPU's
    /// copy into the region among them (its batch ended with the barrier to the host): each
    /// piece's bytes copied from the region into the pages, in the runs' order, counted on
    /// `device`. It records nothing.
    pub(crate) fn drain(&self, device: &Device) -> Result<RecordedSegment, Error> {
        let region = (self.region.get())
            .ok_or_else(|| internal("a staged scatter drained before its staging"))?;
        let started = Instant::now();
        for &(host, length, at) in &self.pieces {
            // SAFETY: the piece's host bytes lie in the RAM's ranges (`pieces`), which stay
            // mapped, readable and writable while `_ram` holds the RAM (`HostRam::new`); the
            // region is memory of the runtime's own, apart from them, and no command writes it
            // any more.
            unsafe { region.copy_out(at, host as *mut u8, length) };
        }
        let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        (device.shared().counts).host_ram_staged_drained(nanos);
        Ok(RecordedSegment::submitted(Vec::new()))
    }
}

/// The host bytes `pieces` (cut where the chunks of `spans` end) lie in, joined as a copy's uses
/// name them ([`chunk_ranges`]), each with its chunk.
fn touched_bytes(pieces: &[Piece], spans: &[Range<u64>]) -> Vec<(usize, Range<u64>)> {
    (chunk_ranges(pieces, MOST_RANGES).into_iter())
        .flat_map(|(slot, ranges)| {
            let start = spans[slot].start;
            (ranges.into_iter()).map(move |range| (slot, start + range.start..start + range.end))
        })
        .collect()
}

/// The bytes of the buffer `pieces` cover (each at its byte of the copy's bytes), sorted, those
/// that meet or touch as one: a staged gather's regions, which write each byte once.
fn covered(pieces: &[Piece]) -> Vec<Range<u64>> {
    let mut bytes = in_buffer(pieces);
    bytes.sort_by_key(|range| range.start);
    let mut out: Vec<Range<u64>> = Vec::with_capacity(bytes.len());
    for range in bytes {
        match out.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => out.push(range),
        }
    }
    out
}

/// The unit a run starts on: a 4-byte word of host address. Its end may be any byte.
pub(crate) const WORD: u64 = 4;

/// The most regions of one `vkCmdCopyBuffer` of a copy by regions (`record_host_ram_copy`): a
/// copy of more takes more commands, so no one command's array grows with the runs (a laid-out
/// 4K surface's rows are 2160 runs a plane).
pub(crate) const MOST_REGIONS: usize = 1024;

/// A run of a copy: the buffer's byte its first byte is copied from or to, and the run's host
/// bytes.
pub(crate) type Run = (u64, Range<u64>);

/// `runs` laid one after another in the buffer from its byte `offset`, as `copy_from_host_ram`
/// and `copy_to_host_ram` take them; `None` where their bytes pass `u64`.
pub(crate) fn one_after_another(offset: u64, runs: &[Range<u64>]) -> Option<Vec<Run>> {
    let mut at = offset;
    (runs.iter())
        .map(|run| {
            let first = at;
            at = at.checked_add(run.end.saturating_sub(run.start))?;
            Some((first, run.clone()))
        })
        .collect()
}

/// `runs`, each starting on a [`WORD`] and holding at least a byte, with each that starts where
/// the one before it ended, in the RAM and in the buffer, joined to it; `None` where a run is
/// empty or starts off a word.
pub(crate) fn merged(runs: &[Run]) -> Option<Vec<Run>> {
    let mut out: Vec<Run> = Vec::with_capacity(runs.len());
    for (at, run) in runs {
        if run.start >= run.end || !run.start.is_multiple_of(WORD) {
            return None;
        }
        match out.last_mut() {
            Some((last_at, last))
                if last.end == run.start
                    && last_at.checked_add(last.end - last.start) == Some(*at) =>
            {
                last.end = run.end
            }
            _ => out.push((*at, run.clone())),
        }
    }
    Some(out)
}

/// Whether every one of `runs` (each starting on a word, [`merged`]) ends on a [`WORD`] too, and
/// lies from a word of the buffer: a copy of such runs is a dispatch of the pages copy per chunk;
/// any other, regions of `vkCmdCopyBuffer`.
pub(crate) fn in_words(runs: &[Run]) -> bool {
    (runs.iter()).all(|(at, run)| at.is_multiple_of(WORD) && run.end.is_multiple_of(WORD))
}

/// Whether two of `ranges` (a copy's destination bytes: a scatter's host bytes, a gather's in the
/// buffer) share a byte: such a copy writes a byte twice, which only a copy by regions orders
/// ([`rounds`]).
pub(crate) fn share_a_byte(ranges: &[Range<u64>]) -> bool {
    let mut sorted: Vec<(u64, u64)> = (ranges.iter())
        .map(|range| (range.start, range.end))
        .collect();
    sorted.sort_unstable();
    sorted.windows(2).any(|pair| pair[1].0 < pair[0].1)
}

/// The bytes of the buffer `pieces` (at their bytes from the copy's first) lie in.
pub(crate) fn in_buffer(pieces: &[Piece]) -> Vec<Range<u64>> {
    (pieces.iter())
        .map(|piece| piece.at..piece.at + piece.length)
        .collect()
}

/// The pieces of a copy by regions as rounds of `vkCmdCopyBuffer`s, from their destination bytes
/// (a scatter's host bytes, a gather's in the buffer): each round the pieces (by index, in order)
/// whose destination bytes no other piece of the round's touches, so that no byte is written by
/// two regions of one command; a later piece over bytes an earlier one of the round touched
/// starts the next round, ordered after it as a command that writes them again is. Only runs
/// that name a destination byte twice take more than one.
pub(crate) fn rounds(destination: &[Range<u64>]) -> Vec<Vec<usize>> {
    let mut rounds: Vec<Vec<usize>> = Vec::new();
    // The round's destination bytes so far, apart, by start.
    let mut taken: BTreeMap<u64, u64> = BTreeMap::new();
    for (index, range) in destination.iter().enumerate() {
        // Of the round's ranges, the one that starts last before this one ends is the only one
        // that may overlap it: those before it end before it starts.
        let before = taken.range(..range.end).next_back();
        let overlaps = before.is_some_and(|(_, &end)| end > range.start);
        match rounds.last_mut() {
            Some(round) if !overlaps => round.push(index),
            _ => {
                rounds.push(vec![index]);
                taken.clear();
            }
        }
        taken.insert(range.start, range.end);
    }
    rounds
}

/// A run's bytes inside one chunk: `length` bytes from byte `offset` of chunk `slot`, at byte
/// `at` of the buffer (from the copy's first, once recorded).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Piece {
    pub(crate) slot: usize,
    pub(crate) offset: u64,
    pub(crate) length: u64,
    pub(crate) at: u64,
}

/// `runs` cut where a chunk of `spans` (the RAM's chunks, by start, apart) ends, in order, each
/// piece at its byte of the buffer, and the bytes they total; `None` where a byte of one lies in
/// no chunk (outside the RAM's ranges) or the total passes `u64`. Each piece's chunk is looked
/// for first where the piece before it lies, as a surface's pages often share one.
pub(crate) fn pieces(runs: &[Run], chunks: &ChunkSpans) -> Option<(Vec<Piece>, u64)> {
    let spans = &chunks.spans;
    let mut out = Vec::with_capacity(runs.len());
    let mut total: u64 = 0;
    let mut last: Option<usize> = None;
    for (run_at, run) in runs {
        let mut at = *run_at;
        let mut start = run.start;
        while start < run.end {
            let slot = match last.filter(|&slot| spans[slot].contains(&start)) {
                Some(slot) => slot,
                None => chunks.of(start)?,
            };
            last = Some(slot);
            let span = &spans[slot];
            let end = run.end.min(span.end);
            let length = end - start;
            out.push(Piece {
                slot,
                offset: start - span.start,
                length,
                at,
            });
            at = at.checked_add(length)?;
            total = total.checked_add(length)?;
            start = end;
        }
    }
    Some((out, total))
}

/// One dispatch of a copy: the runs of the copy's imports (in order: a chunk's, a range of one
/// alone, a list) `array × group` to `array × (group + 1)` (each import `i` of them element
/// `i - array × group` of the dispatch's array, `array` the [`plan`]'s) whose bytes in the buffer
/// lie in window `window`, `entries` of the table from `first_entry`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Dispatch {
    pub(crate) group: usize,
    pub(crate) window: u64,
    pub(crate) first_entry: u32,
    pub(crate) entries: u32,
}

/// The table of `pieces` (each a whole number of words from a word, in their imports) and the
/// dispatches over it, each binding an array of `array` imports: each piece cut into runs of at
/// most a [`PAGE`] from its start and where the buffer's windows of `window_bytes` (a multiple of
/// a page) end, each run an entry of the element of its dispatch's array the import its piece's
/// `slot` names is, its first word in that import, its first word in the buffer counted from its
/// window, and its words; the runs of each `array` imports (`slot / array`) and window together,
/// in the order the pieces give them, at most `most_groups` a dispatch. A piece of whole pages
/// from a page is an entry a page.
pub(crate) fn plan(
    pieces: &[Piece],
    window_bytes: u64,
    most_groups: u32,
    array: usize,
) -> Option<(Vec<u8>, Vec<Dispatch>)> {
    let array = array.max(1);
    // The entries of each array of imports and window, in that order: bucket
    // `group × windows + window`.
    let groups = pieces.iter().map(|piece| piece.slot / array + 1).max();
    let windows = (pieces.iter())
        .map(|piece| (piece.at + piece.length).div_ceil(window_bytes))
        .max();
    let windows = usize::try_from(windows.unwrap_or(0)).ok()?;
    let mut buckets: Vec<Vec<[u32; 4]>> = vec![Vec::new(); groups.unwrap_or(0) * windows];
    for piece in pieces {
        let mut done = 0;
        while done < piece.length {
            let at = piece.at + done;
            let window = at / window_bytes;
            let window_end = (window + 1) * window_bytes;
            let length = (piece.length - done).min(PAGE).min(window_end - at);
            let entry = [
                u32::try_from(piece.slot % array).ok()?,
                u32::try_from((piece.offset + done) / WORD).ok()?,
                u32::try_from((at - window * window_bytes) / WORD).ok()?,
                u32::try_from(length / WORD).ok()?,
            ];
            let bucket = piece.slot / array * windows + usize::try_from(window).ok()?;
            buckets[bucket].push(entry);
            done += length;
        }
    }
    let mut table = Vec::with_capacity(16 * buckets.iter().map(Vec::len).sum::<usize>());
    let mut dispatches = Vec::new();
    let mut entry: u32 = 0;
    for (bucket, entries) in buckets.iter().enumerate() {
        let (group, window) = (bucket / windows.max(1), (bucket % windows.max(1)) as u64);
        for part in entries.chunks(most_groups.max(1) as usize) {
            let count = u32::try_from(part.len()).ok()?;
            dispatches.push(Dispatch {
                group,
                window,
                first_entry: entry,
                entries: count,
            });
            for &[element, scattered, consecutive, words] in part {
                let words = PagesCopyModule::entry(element, scattered, consecutive, words);
                table.extend_from_slice(&words);
            }
            entry = entry.checked_add(count)?;
        }
    }
    Some((table, dispatches))
}

/// The bytes of each chunk `pieces` touch, by chunk: sorted, those that meet or touch as one, and
/// at most `most` (the closest joined, the bytes between them taken too).
pub(crate) fn chunk_ranges(pieces: &[Piece], most: usize) -> BTreeMap<usize, Vec<Range<u64>>> {
    let mut by_chunk: BTreeMap<usize, Vec<Range<u64>>> = BTreeMap::new();
    for piece in pieces {
        let bytes = piece.offset..piece.offset + piece.length;
        by_chunk.entry(piece.slot).or_default().push(bytes);
    }
    (by_chunk.into_iter())
        .map(|(slot, mut ranges)| {
            ranges.sort_unstable_by_key(|range| range.start);
            let mut apart: Vec<Range<u64>> = Vec::with_capacity(ranges.len());
            for range in ranges {
                match apart.last_mut() {
                    Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
                    _ => apart.push(range),
                }
            }
            match apart.len() <= most.max(1) {
                true => (slot, apart),
                false => (slot, joined(&apart, most).into_owned()),
            }
        })
        .collect()
}

/// The bytes of each chunk `pieces` touch (cut where the chunks of `spans` end), by chunk, as
/// [`chunk_ranges`] gives them with no bound (sorted, those that meet or touch as one), and
/// whether two of the pieces share a host byte, as [`share_a_byte`] of their host bytes says: in
/// one pass, and with no sort for pieces of whole pages from a page (a surface's pages), which
/// mark their pages in their chunk's bitmap, read back in order; only the others (rows inside
/// pages) are sorted. The touched chunks' bitmaps lie in one vector. Two pieces share a byte
/// where a piece of whole pages marks a page marked before, where another piece has a byte in a
/// page such a piece marked, or where two of the others overlap.
pub(crate) fn touched_ranges(
    pieces: &[Piece],
    spans: &[Range<u64>],
) -> (BTreeMap<usize, Vec<Range<u64>>>, bool) {
    /// A chunk the pieces touch: its words of the bitmaps, a bit a page, and its pieces.
    struct Met {
        slot: usize,
        words: Range<usize>,
        pieces: usize,
    }
    // Each touched chunk's place in `met`, by chunk.
    let mut place: Vec<usize> = vec![usize::MAX; spans.len()];
    let mut met: Vec<Met> = Vec::new();
    let mut bits: Vec<u64> = Vec::new();
    let mut inside: Vec<(usize, Range<u64>)> = Vec::new();
    let mut twice = false;
    for piece in pieces {
        let at = match place[piece.slot] {
            usize::MAX => {
                let span = &spans[piece.slot];
                let first = bits.len();
                let words = (span.end - span.start).div_ceil(PAGE).div_ceil(64) as usize;
                bits.resize(first + words, 0);
                met.push(Met {
                    slot: piece.slot,
                    words: first..first + words,
                    pieces: 0,
                });
                place[piece.slot] = met.len() - 1;
                met.len() - 1
            }
            at => at,
        };
        met[at].pieces += 1;
        let (first, end) = (piece.offset, piece.offset + piece.length);
        if !first.is_multiple_of(PAGE) || !end.is_multiple_of(PAGE) {
            inside.push((piece.slot, first..end));
            continue;
        }
        let words = met[at].words.clone();
        twice |= mark(&mut bits[words], first / PAGE, end / PAGE);
    }
    inside.sort_unstable_by_key(|(slot, bytes)| (*slot, bytes.start));
    // Another piece's bytes in a marked page, or two of the others that overlap.
    let mut reach: Option<(usize, u64)> = None;
    for (slot, bytes) in &inside {
        let pages = bytes.start / PAGE..bytes.end.div_ceil(PAGE);
        let meets = any_marked(&bits[met[place[*slot]].words.clone()], pages);
        let overlaps = reach.is_some_and(|(last, end)| last == *slot && bytes.start < end);
        twice |= meets || overlaps;
        reach = match reach {
            Some((last, end)) if last == *slot => Some((last, end.max(bytes.end))),
            _ => Some((*slot, bytes.end)),
        };
    }
    // Each chunk's marked pages and other pieces' bytes, in order, those that meet or touch as
    // one: no more ranges than the chunk's pieces.
    met.sort_unstable_by_key(|met| met.slot);
    let mut others = inside.iter().peekable();
    let out = (met.iter()).map(|met| {
        let mut pages = MarkedRuns::new(&bits[met.words.clone()]).peekable();
        let mut ranges: Vec<Range<u64>> = Vec::with_capacity(met.pieces);
        loop {
            let other = others.next_if(|(slot, bytes)| {
                *slot == met.slot && pages.peek().is_none_or(|page| bytes.start < page.start)
            });
            let Some(range) = other
                .map(|(_, bytes)| bytes.clone())
                .or_else(|| pages.next())
            else {
                break;
            };
            match ranges.last_mut() {
                Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
                _ => ranges.push(range),
            }
        }
        (met.slot, ranges)
    });
    (out.collect(), twice)
}

/// Mark the pages `first..end` in `bits`: whether one of them was marked before.
fn mark(bits: &mut [u64], first: u64, end: u64) -> bool {
    let mut met = false;
    let mut page = first;
    while page < end {
        let (word, bit) = ((page / 64) as usize, page % 64);
        let count = (64 - bit).min(end - page);
        let mask = match count {
            64 => u64::MAX,
            _ => ((1u64 << count) - 1) << bit,
        };
        met |= bits[word] & mask != 0;
        bits[word] |= mask;
        page += count;
    }
    met
}

/// Whether one of the pages `pages` is marked in `bits`.
fn any_marked(bits: &[u64], pages: Range<u64>) -> bool {
    let mut pages = pages;
    pages.any(|page| {
        (bits.get((page / 64) as usize)).is_some_and(|word| (word >> (page % 64)) & 1 == 1)
    })
}

/// The bytes of the pages marked in a bitmap (a bit a page, from the first word's lowest), in
/// order, consecutive pages as one (across words too), read as they are asked for.
struct MarkedRuns<'a> {
    bits: &'a [u64],
    /// The word being read, and its bits not read yet.
    word: usize,
    rest: u64,
}

impl<'a> MarkedRuns<'a> {
    fn new(bits: &'a [u64]) -> MarkedRuns<'a> {
        MarkedRuns {
            bits,
            word: 0,
            rest: bits.first().copied().unwrap_or(0),
        }
    }
}

impl Iterator for MarkedRuns<'_> {
    type Item = Range<u64>;

    fn next(&mut self) -> Option<Range<u64>> {
        while self.rest == 0 {
            self.word += 1;
            self.rest = *self.bits.get(self.word)?;
        }
        let bit = u64::from(self.rest.trailing_zeros());
        let first = self.word as u64 * 64 + bit;
        let ones = u64::from((self.rest >> bit).trailing_ones());
        if bit + ones < 64 {
            self.rest &= !((1u64 << (bit + ones)) - 1);
            return Some(first * PAGE..(first + ones) * PAGE);
        }
        // The run reaches the word's end: it goes on through the next words' low ones.
        let mut end = (self.word as u64 + 1) * 64;
        self.rest = 0;
        while let Some(&word) = self.bits.get(self.word + 1) {
            self.word += 1;
            let ones = u64::from(word.trailing_ones());
            end += ones;
            if ones < 64 {
                self.rest = word & !((1u64 << ones) - 1);
                break;
            }
        }
        Some(first * PAGE..end * PAGE)
    }
}

/// One chunk's bytes `ranges` (by start, those that meet or touch as one: [`chunk_ranges`]) as a
/// copy imports them alone where the chunk is not imported yet (`RamOnDevice::for_copy`): those
/// less than [`JOIN_GAP`] apart as one, never joined further.
pub(crate) fn alone_ranges(ranges: &[Range<u64>]) -> Vec<Range<u64>> {
    let mut out: Vec<Range<u64>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match out.last_mut() {
            Some(last) if range.start <= last.end.saturating_add(JOIN_GAP - 1) => {
                last.end = last.end.max(range.end);
            }
            _ => out.push(range.clone()),
        }
    }
    out
}

/// One chunk's bytes `ranges` (offsets in the chunk `span`, by start, apart) as the host pages
/// they lie in, `page` bytes each, for lists of them (`RamOnDevice::run_lists`): each widened to
/// whole pages within the span, those that then share or touch a page one.
pub(crate) fn page_runs(ranges: &[Range<u64>], span: &Range<u64>, page: u64) -> Vec<Range<u64>> {
    let mut out: Vec<Range<u64>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        let start = span.start + range.start / page * page;
        let end = (span.start + range.end.div_ceil(page) * page).min(span.end);
        match out.last_mut() {
            Some(last) if start <= last.end => last.end = last.end.max(end),
            _ => out.push(start..end),
        }
    }
    out
}

/// `runs` (host pages, by start, apart, of every chunk a copy takes lists of) as the lists a copy
/// imports them in, each one dma-buf of the RAM's source of run lists (`RamOnDevice::run_lists`):
/// the runs in order, at most `most_runs` and `most_bytes` bytes a list, each list the indices of
/// its runs. `None` where a run alone passes `most_bytes`.
pub(crate) fn run_lists(
    runs: &[Range<u64>],
    most_runs: usize,
    most_bytes: u64,
) -> Option<Vec<Range<usize>>> {
    let most_runs = most_runs.max(1);
    let mut lists: Vec<Range<usize>> = Vec::new();
    let mut bytes = 0u64;
    for (index, run) in runs.iter().enumerate() {
        let length = run.end - run.start;
        if length > most_bytes {
            return None;
        }
        match lists.last_mut() {
            Some(list) if list.len() < most_runs && bytes.saturating_add(length) <= most_bytes => {
                list.end = index + 1;
                bytes += length;
            }
            _ => {
                lists.push(index..index + 1);
                bytes = length;
            }
        }
    }
    Some(lists)
}

/// The indices of chunk `slot`'s imports in `by_chunk`, (chunk, index of the import) pairs by
/// chunk (`CopyImports::of_chunk`, sorted by chunk alone), in their order there.
fn imports_of(by_chunk: &[(usize, usize)], slot: usize) -> impl Iterator<Item = usize> + '_ {
    let at = by_chunk.partition_point(|&(chunk, _)| chunk < slot);
    (by_chunk[at..].iter())
        .take_while(move |&&(chunk, _)| chunk == slot)
        .map(|&(_, index)| index)
}

/// `out` (by start, apart: those that meet or touch as one, as [`touched_ranges`] gives a
/// chunk's) with the closest joined until at most `most` are left: `out` itself where it holds
/// no more.
fn joined(out: &[Range<u64>], most: usize) -> Cow<'_, [Range<u64>]> {
    let most = most.max(1);
    if out.len() <= most {
        return Cow::Borrowed(out);
    }
    // Keep the `most - 1` widest gaps (of two as wide, the first); join across the others.
    let mut gaps: Vec<(u64, usize)> = (out.windows(2).enumerate())
        .map(|(at, pair)| (pair[1].start - pair[0].end, at))
        .collect();
    let wider = |a: &(u64, usize), b: &(u64, usize)| b.0.cmp(&a.0).then(a.1.cmp(&b.1));
    if most > 1 {
        gaps.select_nth_unstable_by(most - 2, wider);
    }
    let mut kept: Vec<usize> = gaps[..most - 1].iter().map(|&(_, at)| at).collect();
    kept.sort_unstable();
    let mut result = Vec::with_capacity(most);
    let mut start = out[0].start;
    for at in kept {
        result.push(start..out[at].end);
        start = out[at + 1].start;
    }
    result.push(start..out[out.len() - 1].end);
    Cow::Owned(result)
}

/// A copy between a guest RAM and a buffer as its call gave it ([`BlitCall::HostRamCopy`]), which
/// an encoding again runs again (`replay.rs`).
pub(crate) struct HostRamCall {
    copy: PagesCopy,
    /// The device's use of the RAM; `None` where it uses none such.
    ram: Option<Arc<RamOnDevice>>,
    /// The runs, each at its byte of the buffer: merged where each starts on a word and holds a
    /// byte ([`merged`]), else as given (none where one laid after another passes `u64`). Shared
    /// with the plan kept of the same copy, where the call found one ([`HostRamCall::kept`]).
    runs: Arc<[Run]>,
    /// Whether the runs each start on a word and hold a byte (merged).
    checked: bool,
    /// Whether they each end on a word and lie from a word of the buffer besides ([`in_words`]):
    /// copied by the pages copy's dispatches (unless they write a destination byte twice), else
    /// by regions of `vkCmdCopyBuffer`.
    words: bool,
    buffer: Buffer,
    /// For a copy of consecutive bytes of the buffer (`copy_from_host_ram`, `copy_to_host_ram`),
    /// the first: its bytes are checked against the buffer as one range from it, and refused
    /// where it lies off a word of the buffer's memory. `None` for a copy of runs each at a byte
    /// of its own (`copy_runs_from_host_ram`, `copy_runs_to_host_ram`): each run's bytes are
    /// checked against the buffer, and runs off a word of its memory copied by regions.
    consecutive: Option<u64>,
    /// The plan kept of the same copy the call found as it was made ([`RamCopies::kept_for`]):
    /// its runs as the call's, merged, compared one by one as they were merged, no vector made.
    /// Taken at the recording only while the RAM still keeps it ([`RamCopies::keeps`]).
    kept: Option<Arc<KeptPlan>>,
    /// How its first encoding staged it through the CPU, where it did ([`Takes::staging`]): an
    /// encoding again of its encoder stages a scatter so again ([`HostRamCall::staged_again`]).
    staged: OnceLock<Staging>,
}

/// A copy's runs as its call gave them: laid one after another from a byte of the buffer
/// (`copy_from_host_ram`, `copy_to_host_ram`), or each at a byte of its own (`copy_runs_*`).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Given<'a> {
    Laid(u64, &'a [Range<u64>]),
    AtOwn(&'a [Run]),
}

impl<'a> Given<'a> {
    /// The first byte of the buffer of runs laid one after another ([`HostRamCall::consecutive`]).
    fn consecutive(self) -> Option<u64> {
        match self {
            Given::Laid(offset, _) => Some(offset),
            Given::AtOwn(_) => None,
        }
    }

    /// Each run at its byte of the buffer, in order, as [`one_after_another`] lays them; `None`
    /// for a run whose bytes laid there end past `u64`, and each after it.
    fn runs(self) -> impl Iterator<Item = Option<Run>> + 'a {
        let (mut at, laid, own) = match self {
            Given::Laid(offset, runs) => (Some(offset), runs, &[][..]),
            Given::AtOwn(runs) => (None, &[][..], runs),
        };
        // A run whose bytes end past `u64` is `None`, as are those after it.
        let laid = laid.iter().map(move |run| {
            let first = at?;
            at = first.checked_add(run.end.saturating_sub(run.start));
            at.map(|_| (first, run.clone()))
        });
        laid.chain(own.iter().map(|run| Some(run.clone())))
    }

    /// The runs, each at its byte of the buffer, as a vector ([`one_after_another`]).
    fn to_vec(self) -> Option<Vec<Run>> {
        match self {
            Given::Laid(offset, runs) => one_after_another(offset, runs),
            Given::AtOwn(runs) => Some(runs.to_vec()),
        }
    }
}

/// Whether `given`'s runs merged ([`merged`]) are `kept`, in one pass with no vector made: each
/// checked as `merged` checks it, joined as it joins them, and each joined run compared with the
/// next of `kept` once the run after it starts apart. False where `merged` refuses one, or one
/// lies past `u64`.
fn merges_to(given: Given<'_>, kept: &[Run]) -> bool {
    let mut kept = kept.iter();
    let mut open: Option<Run> = None;
    for run in given.runs() {
        let Some((at, run)) = run else {
            return false;
        };
        if run.start >= run.end || !run.start.is_multiple_of(WORD) {
            return false;
        }
        let joins = matches!(&open, Some((last_at, last))
            if last.end == run.start && last_at.checked_add(last.end - last.start) == Some(at));
        if joins {
            if let Some((_, last)) = &mut open {
                last.end = run.end;
            }
            continue;
        }
        if let Some(done) = open.replace((at, run)) {
            if kept.next() != Some(&done) {
                return false;
            }
        }
    }
    match open {
        Some(done) => kept.next() == Some(&done) && kept.next().is_none(),
        None => kept.next().is_none(),
    }
}

impl HostRamCall {
    /// The call that copies as `copy` says between `ram` and `buffer` the runs `runs`, each at its
    /// byte of `buffer`, on `device`; `consecutive` as the field says.
    fn new(
        device: &Device,
        copy: PagesCopy,
        (ram, given): (&HostRam, Given<'_>),
        buffer: &Buffer,
    ) -> HostRamCall {
        let ram = (device.shared().host_ram.used().into_iter())
            .find(|used| std::ptr::eq(Arc::as_ptr(&used.ram), ram));
        let consecutive = given.consecutive();
        // The plan kept of the same copy: its runs, checked and copied by words, are the call's.
        let kept = (ram.as_ref()).and_then(|ram| ram.copies.kept_for(copy, buffer, given));
        if let Some(plan) = kept {
            return HostRamCall {
                copy,
                ram,
                runs: Arc::clone(&plan.runs),
                checked: true,
                words: true,
                buffer: buffer.clone(),
                consecutive,
                kept: Some(plan),
                staged: OnceLock::new(),
            };
        }
        let runs = given.to_vec();
        let (runs, checked): (Arc<[Run]>, bool) = match runs.as_deref().and_then(merged) {
            Some(merged) => (merged.into(), true),
            None => (runs.unwrap_or_default().into(), false),
        };
        let words = checked && in_words(&runs);
        HostRamCall {
            copy,
            ram,
            runs,
            checked,
            words,
            buffer: buffer.clone(),
            consecutive,
            kept: None,
            staged: OnceLock::new(),
        }
    }

    /// How an encoding again of its encoder (`replay.rs`, `command_buffer/reencode.rs`) stages
    /// it: as its first encoding did, where that staged a scatter, whose CPU copy goes in a part
    /// of its own that a copy through the imports would not start, and which a chunk that takes
    /// no import while it runs could not take (`Deferred::left_again_by_an_encoding_again`).
    /// `None` for every other copy, which an encoding again copies through the imports.
    fn staged_again(&self) -> Option<Staging> {
        let staging = *self.staged.get()?;
        (self.copy == PagesCopy::Scatter).then_some(staging)
    }

    /// The call's name, as its refusals give it.
    fn name(&self) -> &'static str {
        match (self.copy, self.consecutive.is_some()) {
            (PagesCopy::Gather, true) => "copy_from_host_ram",
            (PagesCopy::Scatter, true) => "copy_to_host_ram",
            (PagesCopy::Gather, false) => "copy_runs_from_host_ram",
            (PagesCopy::Scatter, false) => "copy_runs_to_host_ram",
        }
    }

    /// The bytes it holds of its own, besides its size: its runs.
    pub(crate) fn extra(&self) -> usize {
        std::mem::size_of_val(&*self.runs)
    }
}

impl BlitCommandEncoder {
    /// Copy `runs` of the guest RAM `ram` into consecutive bytes of `destination` from
    /// `destination_offset`, in the order given, by the GPU, from the device's imports of the RAM's
    /// chunks: a runtime extension, Metal has none (a program copies from a no-copy buffer over the
    /// pages, `copy_from_buffer_to_buffer`). The runs are host addresses inside the RAM's ranges,
    /// each starting on a 4-byte word and ending at any byte past it. Adjacent ascending runs are
    /// one; where every run also ends on a 4-byte word (whole pages, or rows of 4-byte texels), the
    /// runs of up to 30 chunks they touch are one dispatch (the device's
    /// `PagesCopyModule::imports`, one where it indexes no array of storage buffers), and
    /// otherwise each run's bytes in a chunk are one region of a copy; neither reads another byte
    /// of their pages (the module docs of `host_ram/copy.rs`), with no view of the pages and no import of its own. It is ordered and
    /// seen as a copy from a no-copy buffer over the pages into `destination` would be: the CPU's
    /// writes to the pages before the commit are seen, hazard tracking orders it against every
    /// other use of the same host bytes (another copy's, a no-copy buffer's or heap's over the
    /// RAM's own addresses), and `destination` is written as by any blit. An object over another
    /// mapping of the same pages is ordered against it only as the program orders it, as two
    /// no-copy buffers over two mappings of one page are. Where a chunk is not imported yet, the
    /// runs' pages in it are imported for the copy alone, held until its command buffer
    /// completed (`WorkCounts::run_imports_live`): on a device whose held imports add to the cost
    /// of every submission, of a RAM whose chunks are its dma-bufs and that came with a source of
    /// run lists ([`HostRam::with_dma_buf_of_runs`]; RADV's, `Device::use_host_ram`), as one
    /// dma-buf of that source for each list of up to its count of runs
    /// (`WorkCounts::host_ram_run_list_imports`), the chunk neither imported nor asked for, so
    /// the copy makes no page of the guest's RAM resident but its own; on a device whose thread
    /// imports the chunks ahead, through the CPU, importing nothing of its own, while the thread
    /// imports the chunk next (the module docs' "First touch": a gather's pages read by the CPU
    /// before its batch is submitted, a scatter's written by the CPU once its GPU copy
    /// completed, before its command buffer completes); elsewhere, for up to 16 ranges of the
    /// chunk, each range alone, the device's thread asked for the chunk. Where
    /// that cannot be, the chunk is imported first, or its import waited for.
    ///
    /// No runs copy nothing.
    ///
    /// Refused: a device that imports no host memory (`Unsupported`, naming
    /// `VK_EXT_external_memory_host`); a RAM the device does not use ([`Device::use_host_ram`](crate::Device::use_host_ram)), a run that
    /// is empty, starts off a 4-byte word or lies outside the RAM's ranges, bytes of
    /// `destination` that start off a 4-byte word of its memory, or, of a copy by regions (a run
    /// ending off a word), a `destination` over host pages the runs' bytes overlap (`Invalid`,
    /// `Problem::HostRam`);
    /// bytes past `destination`'s length or a buffer of another device (`Invalid`); and a chunk
    /// the device cannot import (the import's error). A caller falls back to its own copy.
    pub fn copy_from_host_ram(
        &mut self,
        ram: &HostRam,
        runs: &[Range<u64>],
        destination: &Buffer,
        destination_offset: u64,
    ) -> Result<(), Error> {
        let device = self.core().device().clone();
        let given = Given::Laid(destination_offset, runs);
        let call = HostRamCall::new(&device, PagesCopy::Gather, (ram, given), destination);
        self.host_ram_copy(&Arc::new(call))
    }

    /// Copy the bytes of `source` from `source_offset` out into `runs` of the guest RAM `ram`, in
    /// the order given, by the GPU, into the device's imports of the RAM's chunks: the scatter
    /// [`copy_from_host_ram`](Self::copy_from_host_ram) gathers, a runtime extension (Metal has
    /// none). Ordered and seen as a copy from `source` into a no-copy buffer over the pages
    /// would be: hazard tracking orders it against every other use of the same host bytes, its
    /// bytes are in the pages once the command buffer completed (over a chunk not imported yet,
    /// copied there by the CPU from the GPU's copy into memory of the command buffer's own, as
    /// [`copy_from_host_ram`](Self::copy_from_host_ram) says), and where the RAM came
    /// with a log of its pages' writes (`HostRam::with_dirty_pages`) they are reported to it as
    /// the GPU's writes in place, so every copy the runtime keeps of those pages takes them. A
    /// run that starts or ends inside a page writes its bytes alone: the rest of the page keeps
    /// what it held. Runs that name a byte twice leave it as the later wrote it (they are copied
    /// by regions, a round for each).
    ///
    /// Refused as [`copy_from_host_ram`](Self::copy_from_host_ram), `source` in place of the
    /// destination.
    pub fn copy_to_host_ram(
        &mut self,
        source: &Buffer,
        source_offset: u64,
        ram: &HostRam,
        runs: &[Range<u64>],
    ) -> Result<(), Error> {
        let device = self.core().device().clone();
        let given = Given::Laid(source_offset, runs);
        let call = HostRamCall::new(&device, PagesCopy::Scatter, (ram, given), source);
        self.host_ram_copy(&Arc::new(call))
    }

    /// Copy each of `runs` of the guest RAM `ram` into `destination` at its own byte: each run a
    /// byte of `destination` and a host run, whose bytes land from that byte on, in the order
    /// given. The form of [`copy_from_host_ram`](Self::copy_from_host_ram) for a buffer that keeps
    /// a surface's layout (row `r` at `base + r × pitch`, the bytes between the rows left as they
    /// were): one call for every row, a runtime extension (Metal has none). The host runs are as
    /// that call takes them (from a 4-byte word to any byte past it), and a run that starts where
    /// the one before it ended, in the RAM and in `destination`, is one with it. Where every run
    /// also ends on a 4-byte word and its byte of `destination` is a 4-byte word of the buffer's
    /// memory, the runs of up to 30 chunks they touch are one dispatch (as
    /// [`copy_from_host_ram`](Self::copy_from_host_ram) says); otherwise, or where two runs
    /// name one byte of `destination` (the later run's bytes are kept), each run's bytes in a
    /// chunk are one region of a copy. Neither reads or writes another byte of the runs' pages
    /// or of `destination`. Ordered and seen as
    /// [`copy_from_host_ram`](Self::copy_from_host_ram) is, hazard tracking taking
    /// `destination`'s bytes from the lowest run's first to the highest's last.
    ///
    /// No runs copy nothing.
    ///
    /// Refused as [`copy_from_host_ram`](Self::copy_from_host_ram) is (`Unsupported` without
    /// `VK_EXT_external_memory_host`; a RAM the device does not use, a run that is empty, starts
    /// off a 4-byte word or lies outside the RAM's ranges: `Invalid`, `Problem::HostRam`; a
    /// buffer of another device; a chunk the device cannot import), and: a run whose bytes pass
    /// `destination`'s length (`Invalid`, `Problem::RangeOutOfBounds`, the first such run's); a
    /// copy by regions with a `destination` over host pages its bytes' span overlaps (`Invalid`,
    /// `Problem::HostRam`). A run at a byte of `destination` off a 4-byte word of its memory is
    /// not refused: the copy takes regions. A caller falls back to its own copy.
    pub fn copy_runs_from_host_ram(
        &mut self,
        ram: &HostRam,
        runs: &[(u64, Range<u64>)],
        destination: &Buffer,
    ) -> Result<(), Error> {
        let device = self.core().device().clone();
        let given = Given::AtOwn(runs);
        let call = HostRamCall::new(&device, PagesCopy::Gather, (ram, given), destination);
        self.host_ram_copy(&Arc::new(call))
    }

    /// Copy the bytes of `source` out into `runs` of the guest RAM `ram`: each run a byte of
    /// `source` and a host run, which takes the bytes from that byte on, in the order given. The
    /// scatter [`copy_runs_from_host_ram`](Self::copy_runs_from_host_ram) gathers, and the form
    /// of [`copy_to_host_ram`](Self::copy_to_host_ram) for a buffer that keeps a surface's
    /// layout: one call for every row, a runtime extension (Metal has none). Copied as
    /// [`copy_runs_from_host_ram`](Self::copy_runs_from_host_ram) copies, but for runs that name
    /// one host byte twice, which are copied by regions and leave it as the later wrote it; seen
    /// as [`copy_to_host_ram`](Self::copy_to_host_ram) is. A run that starts or ends inside a page
    /// writes its bytes alone: the rest of the page keeps what it held.
    ///
    /// Refused as [`copy_runs_from_host_ram`](Self::copy_runs_from_host_ram), `source` in place
    /// of the destination.
    pub fn copy_runs_to_host_ram(
        &mut self,
        source: &Buffer,
        ram: &HostRam,
        runs: &[(u64, Range<u64>)],
    ) -> Result<(), Error> {
        let device = self.core().device().clone();
        let given = Given::AtOwn(runs);
        let call = HostRamCall::new(&device, PagesCopy::Scatter, (ram, given), source);
        self.host_ram_copy(&Arc::new(call))
    }

    /// Run `call`, logged first where the encoder logs (`replay.rs`).
    pub(crate) fn host_ram_copy(&mut self, call: &Arc<HostRamCall>) -> Result<(), Error> {
        self.logged(
            |_| BlitCall::HostRamCopy(Arc::clone(call)),
            |this| this.record_host_ram_copy(call),
        )
    }

    /// Check `call` and record it (the module docs): by the plan kept of the same copy where it
    /// still holds ([`KeptPlan`], [`record_kept`](Self::record_kept)), else planned now, and the
    /// plan kept where the copy goes by dispatches.
    fn record_host_ram_copy(&mut self, call: &HostRamCall) -> Result<(), Error> {
        let call_name = call.name();
        let device = self.core().device().clone();
        self.core()
            .check_device(call_name, call.buffer.generation())?;
        let shared = device.shared();
        let importer = match &shared.host_import {
            Ok(importer) => importer,
            Err(unsupported) => return Err(Error::Unsupported(unsupported.clone())),
        };
        let refused = || invalid(call_name, Problem::HostRam);
        let ram = (call.ram.as_ref())
            .filter(|ram| shared.host_ram.uses(ram))
            .ok_or_else(refused)?;
        if !call.checked {
            return Err(refused());
        }
        if call.runs.is_empty() {
            return Ok(());
        }
        let limits = &device.properties().description.limits;
        let align = limits.min_storage_buffer_offset_alignment.max(4);
        let range = u64::from(limits.max_storage_buffer_range);
        let window_bytes = range.saturating_sub(align) / PAGE * PAGE;
        if window_bytes == 0 {
            return Err(crate::error::internal(
                "a storage range shorter than a page of a host RAM copy",
            ));
        }
        let figures = (window_bytes, align, limits.max_compute_work_group_count[0]);
        // A copy an encoding again stages as its first encoding did takes no plan.
        let again = (self.core().encodes_again())
            .then(|| call.staged_again())
            .flatten();
        // The plan the call named, while the RAM keeps it (gone, the full path plans the call's
        // runs); else one of the same copy kept since.
        let kept = match &call.kept {
            _ if again.is_some() => None,
            Some(plan) => ram.copies.keeps(plan).then(|| Arc::clone(plan)),
            None => ram.copies.kept(call),
        };
        let early = match kept {
            Some(plan) => match self.record_kept(call, ram, importer, figures, &plan)? {
                Kept::Recorded => return Ok(()),
                Kept::Not(early) => early,
            },
            None => Early::default(),
        };
        // The buffer's bytes the copy reads or writes lie in `span`: each run's bytes, checked
        // against the buffer, or, of runs one after another, one range from the first byte,
        // checked once the runs are in the RAM.
        let length = call.buffer.length();
        let mut span = 0..0;
        if call.consecutive.is_none() {
            let (mut low, mut high) = (u64::MAX, 0);
            for (at, run) in call.runs.iter() {
                let bytes = bytes_in(call_name, *at, run.end - run.start, length)?;
                (low, high) = (low.min(bytes.start), high.max(bytes.end));
            }
            span = low..high;
        }
        let chunks = ram.chunks();
        let spans = &chunks.spans;
        let (mut pieces, total) = pieces(&call.runs, &chunks).ok_or_else(refused)?;
        if let Some(offset) = call.consecutive {
            span = bytes_in(call_name, offset, total, length)?;
        }
        // Each piece at its byte of the copy's bytes, from the first.
        for piece in &mut pieces {
            piece.at -= span.start;
        }
        let (raw, at) = match early.blit {
            Some((start, blit)) if start == span.start => blit,
            _ => self.core().blit_at(&call.buffer, span.start)?,
        };
        if call.consecutive.is_some() && !at.is_multiple_of(4) {
            return Err(refused());
        }
        // The bytes of each chunk the pieces touch, sorted, those that meet or touch as one, and
        // whether two pieces share a host byte. A kept plan's steps taken already (`early`) were
        // of these ranges: the same runs over the same chunks.
        let (ranges, host_twice) = touched_ranges(&pieces, spans);
        if call.copy == PagesCopy::Scatter {
            for &slot in ranges.keys() {
                ram.copy_wrote(slot);
            }
        }
        let takes = match early.takes {
            Some(takes) => takes,
            None => self.takes(ram, importer, &ranges),
        };
        let written = || {
            call.copy == PagesCopy::Gather
                && ram.written_over(&device, &touched_bytes(&pieces, spans))
        };
        if let Some(staging) = again.or_else(|| takes.staging(written)) {
            // The first encoding's staging, which an encoding again finds set (its takes are
            // none: [`takes`](Self::takes)).
            let _ = call.staged.set(staging);
            let bytes = (raw, at, span, total);
            return match call.copy {
                PagesCopy::Scatter => {
                    self.record_staged_scatter(call, ram, (&pieces, spans), bytes)
                }
                PagesCopy::Gather => {
                    self.record_staged(call, ram, (&pieces, spans), bytes, staging)
                }
            };
        }
        // The bytes of each chunk the pieces touch, and the imports they are read or written
        // through, held until the command buffer completes: the chunk's, or, where it is not
        // imported yet, lists of the copy's pages, or one of each of those bytes alone while
        // the device's thread imports it (`RamOnDevice::copy_imports`). Each piece is then
        // counted in its import.
        let mut touched: Vec<(usize, Range<u64>)> = Vec::new();
        for (&slot, ranges) in &ranges {
            let joined = joined(ranges, MOST_RANGES);
            touched.extend(joined.iter().map(|bytes| (slot, bytes.clone())));
        }
        let CopyImports {
            imports,
            of_chunk,
            waited,
        } = match early.imports {
            Some(imports) => imports,
            None => ram.copy_imports(&device, importer, &ranges)?,
        };
        if imports.iter().any(|import| import.bytes() > range) {
            return Err(crate::error::internal(
                "an import of host RAM past the device's storage range",
            ));
        }
        if !waited.is_zero() {
            let nanos = u64::try_from(waited.as_nanos()).unwrap_or(u64::MAX);
            shared.counts.host_ram_copy_waited(nanos);
        }
        // The imports of each chunk, by chunk, each chunk's in the order they were given.
        let mut by_chunk = of_chunk.clone();
        by_chunk.sort_by_key(|&(slot, _)| slot);
        let in_imports: Option<Vec<Piece>> = (pieces.iter())
            .map(|piece| {
                let start = spans[piece.slot].start + piece.offset;
                let host = start..start + piece.length;
                let mut found = imports_of(&by_chunk, piece.slot)
                    .filter_map(|index| Some((index, imports[index].offset_of(&host)?)));
                let (index, offset) = found.next()?;
                Some(Piece {
                    slot: index,
                    offset,
                    ..*piece
                })
            })
            .collect();
        let in_imports = in_imports.ok_or_else(|| {
            crate::error::internal("a host RAM copy's piece outside the imports made for it")
        })?;
        // A copy whose runs write a destination byte twice is ordered by rounds of regions
        // alone: a scatter's that name a host byte twice, a gather's that name a byte of the
        // buffer twice (never runs one after another).
        let twice = || match call.copy {
            PagesCopy::Scatter => host_twice,
            PagesCopy::Gather => call.consecutive.is_none() && share_a_byte(&in_buffer(&pieces)),
        };
        if !call.words || !at.is_multiple_of(4) || twice() {
            let pieces = (&pieces[..], &in_imports[..]);
            return self.record_regions(call, ram, pieces, imports, (raw, at, span, total));
        }
        let (kernel, window) = Window::of(&device, call.copy, figures)?;
        let (table, dispatches) = plan(&in_imports, window.bytes, window.most_groups, window.array)
            .ok_or_else(refused)?;
        let serials: Vec<u64> = imports.iter().map(|import| import.serial()).collect();
        let bytes = (raw, at, span.clone(), total);
        let planned = (Table::Upload(&table), &dispatches[..], &touched[..]);
        self.record_dispatches(call, ram, (&kernel, window), imports, bytes, planned)?;
        ram.copies.keep(KeptPlan {
            copy: call.copy,
            consecutive: call.consecutive,
            buffer: buffer_key(&call.buffer),
            runs: Arc::clone(&call.runs),
            blit: (raw, at),
            pages: call.buffer.host_pages().as_ref().map(Arc::downgrade),
            window,
            span,
            total,
            ranges,
            touched,
            imports: serials,
            of_chunk,
            table,
            dispatches,
            table_kept: OnceLock::new(),
        });
        Ok(())
    }

    /// Record `call` by `plan`, kept of the same copy ([`RamCopies::kept`]: its direction,
    /// runs and buffer), where it still holds: the device's figures (`figures`' window bytes,
    /// alignment and most workgroups, and the pages copy's array: [`Window::of`]) and the
    /// buffer's `VkBuffer`, byte and host pages as they were, every chunk taken at once, and the
    /// imports the copy takes the very ones the plan was made over (by [`RamChunk::serial`]).
    /// Each step a
    /// copy takes every time is taken as the full path takes it: the buffer's `blit_at`, a
    /// scatter's [`RamOnDevice::copy_wrote`], the chunks' takes, the imports
    /// ([`RamOnDevice::copy_imports`]), then the table's upload, the binds over the imports and
    /// the buffer, the uses and the counts ([`record_dispatches`](Self::record_dispatches)).
    /// Where the plan does not hold, what was taken of those steps is left to the full path
    /// ([`Early`]), which takes none twice.
    fn record_kept(
        &mut self,
        call: &HostRamCall,
        ram: &RamOnDevice,
        importer: &super::Importer,
        figures: (u64, u64, u32),
        plan: &KeptPlan,
    ) -> Result<Kept, Error> {
        let device = self.core().device().clone();
        let (kernel, window) = Window::of(&device, call.copy, figures)?;
        let mut early = Early::default();
        let blit = self.core().blit_at(&call.buffer, plan.span.start)?;
        early.blit = Some((plan.span.start, blit));
        let pages = call.buffer.host_pages();
        let same_pages = match (&plan.pages, &pages) {
            (None, None) => true,
            (Some(kept), Some(now)) => std::ptr::eq(kept.as_ptr(), Arc::as_ptr(now)),
            _ => false,
        };
        if blit != plan.blit || !same_pages || window != plan.window {
            return Ok(Kept::Not(early));
        }
        if call.copy == PagesCopy::Scatter {
            for &slot in plan.ranges.keys() {
                ram.copy_wrote(slot);
            }
        }
        let takes = self.takes(ram, importer, &plan.ranges);
        early.takes = Some(takes);
        if takes.staged() {
            return Ok(Kept::Not(early));
        }
        let imports = ram.copy_imports(&device, importer, &plan.ranges)?;
        let same = (imports.imports.iter().map(|import| import.serial()))
            .eq(plan.imports.iter().copied())
            && imports.of_chunk == plan.of_chunk;
        if !same {
            early.imports = Some(imports);
            return Ok(Kept::Not(early));
        }
        if !imports.waited.is_zero() {
            let nanos = u64::try_from(imports.waited.as_nanos()).unwrap_or(u64::MAX);
            device.shared().counts.host_ram_copy_waited(nanos);
        }
        let bytes = (blit.0, blit.1, plan.span.clone(), plan.total);
        let table = Table::Kept(plan.table_buffer(&device)?, plan.table.len() as u64);
        let planned = (table, &plan.dispatches[..], &plan.touched[..]);
        self.record_dispatches(
            call,
            ram,
            (&kernel, window),
            imports.imports,
            bytes,
            planned,
        )?;
        device.shared().counts.host_ram_copy_plan_reused();
        Ok(Kept::Recorded)
    }

    /// Record `call`'s dispatches by `kernel` over `imports`, `table` (uploaded to the command
    /// buffer's memory, or a kept plan's buffer: [`Table`]) and `dispatches` its [`plan`], into or
    /// from the buffer's bytes `span`
    /// (`total` of them copied), from byte `at` of its `raw`, in windows of `window`, with the
    /// uses of the chunks' bytes `touched` (each with its chunk) and the buffer's.
    fn record_dispatches(
        &mut self,
        call: &HostRamCall,
        ram: &RamOnDevice,
        (kernel, window): (&OwnedKernel, Window),
        imports: Vec<Arc<RamChunk>>,
        (raw, at, span, total): (vk::Buffer, u64, Range<u64>, u64),
        (table, dispatches, touched): (Table<'_>, &[Dispatch], &[(usize, Range<u64>)]),
    ) -> Result<(), Error> {
        let device = self.core().device().clone();
        let shared = device.shared();
        let (array, align, window_bytes) = (window.array, window.align, window.bytes);
        let (table_raw, table_offset, table_len) = match table {
            Table::Upload(bytes) => {
                let length = bytes.len() as u64;
                let upload = (self.core().command_buffer().state().uploads)
                    .upload(&device, bytes, length)?;
                (upload.raw(), upload.offset(), length)
            }
            Table::Kept(buffer, length) => {
                let (raw, offset) = buffer.at(0);
                self.core().retain(buffer.clone());
                (raw, offset, length)
            }
        };
        let sets: Vec<OwnedSet> = (dispatches.iter())
            .map(|dispatch| {
                // The dispatch's imports, its array's elements, its entries naming no element past
                // them: where the array is partially bound (`OwnedKernel::partial`), all the
                // array's descriptors it writes, as no invocation reads another; else those past
                // them hold its first again, a valid descriptor its entries name none of.
                let group = imports.get(dispatch.group * array..).unwrap_or_default();
                let lead = group.first().ok_or_else(|| {
                    crate::error::internal("a host RAM copy's dispatch over no import")
                })?;
                let written = match kernel.partial {
                    true => group.len().min(array),
                    false => array,
                };
                let mut buffers: Vec<(vk::Buffer, u64, u64)> = (0..written)
                    .map(|element| {
                        let chunk = group.get(element).unwrap_or(lead);
                        (chunk.raw, 0, chunk.bytes())
                    })
                    .collect();
                let first = at + dispatch.window * window_bytes;
                let offset = first - first % align;
                let length =
                    (span.end - span.start - dispatch.window * window_bytes).min(window_bytes);
                // Each dispatch binds the whole window it writes (a gather's, binding 1) and
                // writes only its pieces' words there. The validation layer's synchronization
                // checks count a bound storage range as written whole (its shader-access
                // heuristic) and so report a write-after-write between two dispatches of one
                // window; no word is written twice (`twice` sent such copies to regions above),
                // so the dispatches race on no byte and need no barrier between them.
                buffers.push((raw, offset, first - offset + length));
                buffers.push((table_raw, table_offset, table_len));
                Ok(OwnedSet {
                    images: Vec::new(),
                    storage_images: Vec::new(),
                    buffers,
                })
            })
            .collect::<Result<_, Error>>()?;
        let binds = OwnedBinds::new(&device, kernel, &sets)?;
        let shapes: Vec<([u8; 16], u32)> = (dispatches.iter())
            .map(|dispatch| {
                let first = at + dispatch.window * window_bytes;
                // Below `align`, a multiple of 4: the word of the window's first byte.
                let first_word = ((first % align) / 4) as u32;
                let shape = PagesCopyModule::shape(dispatch.first_entry, first_word);
                (shape, dispatch.entries)
            })
            .collect();
        // The copy's uses: the chunks' bytes it reads or writes, by their keys, and the buffer's.
        let read = Access::new(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_READ,
        );
        let write = Access::new(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
        );
        let mut uses = Vec::with_capacity(touched.len() + 1);
        let mut last: Option<(usize, ResourceKey)> = None;
        for (slot, bytes) in touched {
            let key = match last {
                Some((at, key)) if at == *slot => key,
                _ => ram.chunk_key(&device, *slot),
            };
            last = Some((*slot, key));
            let range = UseRange::Bytes(bytes.clone());
            uses.push(match call.copy {
                PagesCopy::Gather => Use::reads(key, true, range, read),
                PagesCopy::Scatter => Use::writes(key, true, range, write),
            });
        }
        uses.push(match call.copy {
            PagesCopy::Gather => writes_by(&call.buffer, span, write),
            PagesCopy::Scatter => reads_by(&call.buffer, span, read),
        });
        let recorded = dispatches.len();
        self.command_in(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            &uses,
            |functions, raw| {
                let point = vk::PipelineBindPoint::COMPUTE;
                // SAFETY: the encoder's command buffer is recording, outside a render pass; the
                // chunks' buffers, the buffer and the arena's region are live until it completes (held
                // below, by `blit_at` and by the command buffer); every binding lies inside its
                // buffer and within the device's storage range, and each entry's words inside its
                // bindings (`plan`, the windows), so the workgroups stay inside them.
                unsafe {
                    functions.cmd_bind_pipeline(raw, point, kernel.pipeline);
                    for (index, (shape, entries)) in shapes.iter().enumerate() {
                        binds.bind(functions, raw, kernel, index);
                        functions.cmd_push_constants(
                            raw,
                            kernel.layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            shape,
                        );
                        let [x, y, z] = PagesCopyModule::groups(*entries);
                        functions.cmd_dispatch(raw, x, y, z);
                    }
                }
            },
        )?;
        shared.counts.host_ram_copied(total, recorded as u64);
        self.core().retain(imports);
        if let OwnedBinds::Sets(sets) = binds {
            self.core().retain(sets);
        }
        Ok(())
    }

    /// How a copy over the chunks of `ranges` (the bytes of each it touches: [`chunk_ranges`])
    /// takes them, each as [`RamOnDevice::copy_take`] says, for [`Takes::staging`]: none taken in
    /// an encoder encoded again (`replay.rs`, `command_buffer/reencode.rs`), which stages a copy
    /// only as its first encoding did ([`HostRamCall::staged_again`]).
    fn takes(
        &mut self,
        ram: &RamOnDevice,
        importer: &super::Importer,
        ranges: &BTreeMap<usize, Vec<Range<u64>>>,
    ) -> Takes {
        let mut takes = Takes::default();
        if self.core().encodes_again() {
            return takes;
        }
        let device = self.core().device().clone();
        for (&slot, ranges) in ranges {
            match ram.copy_take(&device, importer, slot, ranges) {
                CopyTake::Now => {}
                CopyTake::ThroughCpu => takes.through_cpu = true,
                CopyTake::CpuOnly => takes.cpu_only = true,
                CopyTake::Waits => takes.waits = true,
            }
        }
        takes
    }

    /// Record `call`, a gather staged through the CPU as `staging` says ([`Takes::staging`]),
    /// into the buffer's bytes `span`, from byte `at` of its `raw` (the module docs' "First
    /// touch"): its work left to its batch's host step ([`StagedGather`]), at its place in the
    /// encoder, its uses noted there: the chunks' bytes it reads, by their keys, and the buffer's
    /// writes. One [`Staging::after_the_gpu`] starts a part of its own there, whose host step waits
    /// for the work before it to complete, and its uses are noted again at that part's start,
    /// where its copy lies. Each chunk is taken as an object over it takes it, and asked of the
    /// device's thread where it is not imported ([`RamOnDevice::take_for_copies`]). A buffer detached
    /// since records nothing, as a command over it does.
    fn record_staged(
        &mut self,
        call: &HostRamCall,
        ram: &Arc<RamOnDevice>,
        (pieces, spans): (&[Piece], &[Range<u64>]),
        (raw, at, span, total): (vk::Buffer, u64, Range<u64>, u64),
        staging: Staging,
    ) -> Result<(), Error> {
        let device = self.core().device().clone();
        let read = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let touched = chunk_ranges(pieces, MOST_RANGES);
        let mut uses = Vec::new();
        for (slot, ranges) in &touched {
            let key = ram.chunk_key(&device, *slot);
            for range in ranges {
                uses.push(Use::reads(key, true, UseRange::Bytes(range.clone()), read));
            }
        }
        uses.push(writes_by(&call.buffer, span.clone(), write));
        if !self.core().note_left_uses(TRANSFER, &uses) {
            return Ok(());
        }
        let in_pages: Vec<(u64, u64, u64)> = (pieces.iter())
            .map(|piece| {
                (
                    spans[piece.slot].start + piece.offset,
                    piece.length,
                    piece.at,
                )
            })
            .collect();
        let staged = StagedGather {
            ram: Arc::clone(ram),
            destination: (raw, at),
            length: span.end - span.start,
            covered: covered(pieces),
            pieces: in_pages,
            touched: touched_bytes(pieces, spans),
            // A gather in a part of its own steps after the work before it completed.
            gated: AtomicBool::new(staging.after_the_gpu),
            staging,
        };
        self.core().defer(Deferred::GatherStage(Arc::new(staged)))?;
        if staging.after_the_gpu {
            self.core().note_left_uses(TRANSFER, &uses);
        }
        let slots: Vec<usize> = touched.keys().copied().collect();
        ram.take_for_copies(&slots, true);
        let counts = &device.shared().counts;
        counts.host_ram_copied(total, 0);
        counts.host_ram_copy_staged();
        Ok(())
    }

    /// Record `call`, a scatter staged through the CPU ([`Takes::staging`]), from the
    /// buffer's bytes `span`, at byte `at` of its `raw` (the module docs' "First touch"): its
    /// work left to two host steps ([`StagedScatter`]), the GPU's copy into the staging at its
    /// place in the encoder, its read of the buffer noted there, and the CPU's copy into the
    /// pages at the start of a part of its own after it, its writes of the chunks' bytes noted
    /// there, by their keys. Each chunk is taken as an object over it takes it, and asked of the
    /// device's thread where it is not imported ([`RamOnDevice::take_for_copies`]). A buffer
    /// detached since records nothing, as a command over it does.
    fn record_staged_scatter(
        &mut self,
        call: &HostRamCall,
        ram: &Arc<RamOnDevice>,
        (pieces, spans): (&[Piece], &[Range<u64>]),
        (raw, at, span, total): (vk::Buffer, u64, Range<u64>, u64),
    ) -> Result<(), Error> {
        let device = self.core().device().clone();
        let read = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let reads = [reads_by(&call.buffer, span.clone(), read)];
        if !self.core().note_left_uses(TRANSFER, &reads) {
            return Ok(());
        }
        let in_pages: Vec<(u64, u64, u64)> = (pieces.iter())
            .map(|piece| {
                (
                    spans[piece.slot].start + piece.offset,
                    piece.length,
                    piece.at,
                )
            })
            .collect();
        let staged = Arc::new(StagedScatter {
            _ram: Arc::clone(ram),
            source: (raw, at),
            length: span.end - span.start,
            covered: covered(pieces),
            pieces: in_pages,
            region: OnceLock::new(),
        });
        self.core()
            .defer(Deferred::ScatterStage(Arc::clone(&staged)))?;
        self.core().defer(Deferred::ScatterDrain(staged))?;
        let touched = chunk_ranges(pieces, MOST_RANGES);
        let mut writes = Vec::new();
        for (slot, ranges) in &touched {
            let key = ram.chunk_key(&device, *slot);
            for range in ranges {
                writes.push(Use::writes(
                    key,
                    true,
                    UseRange::Bytes(range.clone()),
                    write,
                ));
            }
        }
        self.core().note_left_uses(TRANSFER, &writes);
        let slots: Vec<usize> = touched.keys().copied().collect();
        ram.take_for_copies(&slots, true);
        let counts = &device.shared().counts;
        counts.host_ram_copied(total, 0);
        counts.host_ram_scatter_staged();
        Ok(())
    }

    /// Record `call`, whose runs end off a word or are a scatter's that write a byte twice, as
    /// regions of `vkCmdCopyBuffer` between `imports` and the buffer's bytes `bytes`, from byte
    /// `at` of its `raw` (the module docs): `pieces` as cut by the RAM's chunks and as they lie in
    /// the imports, one region each, the regions of one import at most [`MOST_REGIONS`] a
    /// command. A scatter whose pieces write a byte twice takes a round of commands for each
    /// ([`rounds`]), each ordered after the one before as a later write of the same bytes is.
    /// Refused (`Problem::HostRam`) where the buffer lies over host pages the pieces' bytes
    /// overlap: no region may read and write one byte.
    fn record_regions(
        &mut self,
        call: &HostRamCall,
        ram: &RamOnDevice,
        (pieces, in_imports): (&[Piece], &[Piece]),
        imports: Vec<Arc<RamChunk>>,
        (raw, at, span, total): (vk::Buffer, u64, Range<u64>, u64),
    ) -> Result<(), Error> {
        let call_name = call.name();
        let device = self.core().device().clone();
        // Each piece's host bytes.
        let host: Vec<Range<u64>> = (pieces.iter())
            .map(|piece| {
                let start = ram.slots[piece.slot].span.start + piece.offset;
                start..start + piece.length
            })
            .collect();
        if call.buffer.caller_memory() == Some(CallerMemory::HostPages) {
            if let Some(contents) = call.buffer.mapped_contents() {
                let start = contents.as_ptr().addr() as u64 + span.start;
                let own = start..start + (span.end - span.start);
                if (host.iter()).any(|range| range.start < own.end && own.start < range.end) {
                    return Err(invalid(call_name, Problem::HostRam));
                }
            }
        }
        let read = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        // A scatter writes the pieces' host bytes; a gather the buffer's, each once where the
        // runs lie one after another.
        let by_round = match call.copy {
            PagesCopy::Gather if call.consecutive.is_some() => vec![(0..pieces.len()).collect()],
            PagesCopy::Gather => rounds(&in_buffer(pieces)),
            PagesCopy::Scatter => rounds(&host),
        };
        let mut regions_recorded = 0u64;
        for round in &by_round {
            // The round's regions, by import, in order.
            let mut by_import: BTreeMap<usize, Vec<vk::BufferCopy>> = BTreeMap::new();
            for &index in round {
                let piece = &in_imports[index];
                let (import_at, buffer_at) = (piece.offset, at + piece.at);
                let (src_offset, dst_offset) = match call.copy {
                    PagesCopy::Gather => (import_at, buffer_at),
                    PagesCopy::Scatter => (buffer_at, import_at),
                };
                let region = vk::BufferCopy {
                    src_offset,
                    dst_offset,
                    size: piece.length,
                };
                by_import.entry(piece.slot).or_default().push(region);
            }
            // The round's uses: the chunks' bytes it reads or writes, by their keys, and the
            // buffer's.
            let round_pieces: Vec<Piece> = round.iter().map(|&index| pieces[index]).collect();
            let mut uses = Vec::new();
            for (slot, ranges) in chunk_ranges(&round_pieces, MOST_RANGES) {
                let key = ram.chunk_key(&device, slot);
                for range in ranges {
                    let range = UseRange::Bytes(range);
                    uses.push(match call.copy {
                        PagesCopy::Gather => Use::reads(key, true, range, read),
                        PagesCopy::Scatter => Use::writes(key, true, range, write),
                    });
                }
            }
            uses.push(match call.copy {
                PagesCopy::Gather => writes_by(&call.buffer, span.clone(), write),
                PagesCopy::Scatter => reads_by(&call.buffer, span.clone(), read),
            });
            regions_recorded += round.len() as u64;
            let imports = &imports;
            self.command_in(TRANSFER, &uses, |functions, command_buffer| {
                for (slot, regions) in &by_import {
                    let chunk = imports[*slot].raw;
                    let (source, destination) = match call.copy {
                        PagesCopy::Gather => (chunk, raw),
                        PagesCopy::Scatter => (raw, chunk),
                    };
                    for part in regions.chunks(MOST_REGIONS) {
                        // SAFETY: the encoder's command buffer is recording, outside a render
                        // pass; the imports and the buffer are live until it completes (held
                        // below and by `blit_at`); each region lies inside its import (`pieces`,
                        // the imports made over them) and inside the buffer's bytes (`bytes_in`),
                        // and no byte is written by two regions of one command (`rounds`) nor
                        // read and written by one (the buffer apart from the pieces' bytes).
                        unsafe {
                            functions.cmd_copy_buffer(command_buffer, source, destination, part)
                        };
                    }
                }
            })?;
        }
        device.shared().counts.host_ram_copied(total, 0);
        device
            .shared()
            .counts
            .host_ram_copy_regions(regions_recorded);
        self.core().retain(imports);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    /// `runs` one after another from the buffer's byte 0.
    fn laid(runs: &[Range<u64>]) -> Vec<Run> {
        one_after_another(0, runs).unwrap()
    }

    /// `runs` one after another from byte 0, merged, as host runs alone.
    fn merged_runs(runs: &[Range<u64>]) -> Option<Vec<Range<u64>>> {
        let merged = merged(&laid(runs))?;
        Some(merged.into_iter().map(|(_, run)| run).collect())
    }

    #[test]
    fn adjacent_ascending_runs_merge_and_descending_ones_stay_apart() {
        let runs = [
            0..PAGE,
            PAGE..3 * PAGE,
            // A descending neighbour: it ends where the run before starts.
            9 * PAGE..10 * PAGE,
            8 * PAGE..9 * PAGE,
            7 * PAGE..8 * PAGE,
            // Ascending, but not adjacent.
            20 * PAGE..21 * PAGE,
        ];
        assert_eq!(
            merged_runs(&runs),
            Some(vec![
                0..3 * PAGE,
                9 * PAGE..10 * PAGE,
                8 * PAGE..9 * PAGE,
                7 * PAGE..8 * PAGE,
                20 * PAGE..21 * PAGE,
            ])
        );
        assert_eq!(merged(&[]), Some(vec![]));
        // Runs each at a byte of its own join only where they touch in the buffer too.
        let rows = [
            (0, 4 * PAGE..4 * PAGE + 100),
            (100, 4 * PAGE + 100..4 * PAGE + 200),
            (300, 4 * PAGE + 200..4 * PAGE + 300),
        ];
        assert_eq!(
            merged(&rows),
            Some(vec![
                (0, 4 * PAGE..4 * PAGE + 200),
                (300, 4 * PAGE + 200..4 * PAGE + 300)
            ])
        );
        assert_eq!(one_after_another(u64::MAX - 4, &one(0..8)), None);
    }

    /// One run, as a list.
    fn one(range: Range<u64>) -> Vec<Range<u64>> {
        std::iter::once(range).collect()
    }

    #[test]
    fn runs_off_a_word_or_empty_are_refused_and_ends_take_any_byte() {
        assert_eq!(merged_runs(&one(PAGE..PAGE)), None, "empty");
        assert_eq!(merged_runs(&one(1..PAGE + 1)), None, "a start off a word");
        assert_eq!(
            merged_runs(&one(PAGE + 2..PAGE + 8)),
            None,
            "a start off a word"
        );
        let backwards = Range {
            start: 2 * PAGE,
            end: PAGE,
        };
        assert_eq!(merged_runs(&one(backwards)), None, "backwards");
        // A start on a word, an end at any byte: an 854-texel row of R8 ends 2 bytes past one.
        assert_eq!(merged_runs(&one(16..16 + 854)), Some(one(16..16 + 854)));
        assert_eq!(merged_runs(&one(0..PAGE + 1)), Some(one(0..PAGE + 1)));
        // Rows that touch are one run; rows apart stay apart.
        let rows = [
            2048 * 3..2048 * 3 + 2046,
            2048 * 4..2048 * 4 + 2044,
            2048 * 4 + 2044..2048 * 5,
        ];
        assert_eq!(
            merged_runs(&rows),
            Some(vec![2048 * 3..2048 * 3 + 2046, 2048 * 4..2048 * 5])
        );
        assert!(in_words(&laid(&[0..PAGE, 3 * PAGE..5 * PAGE])));
        assert!(in_words(&laid(&[16..PAGE - 4, 3 * PAGE..5 * PAGE - 8])));
        assert!(!in_words(&laid(&[0..PAGE, 3 * PAGE..5 * PAGE - 2])));
        assert!(!in_words(&laid(&one(16..PAGE - 1))));
        assert!(in_words(&[]));
        // A run at a byte of the buffer off a word.
        assert!(in_words(&[(8, 16..PAGE), (16380, 3 * PAGE..4 * PAGE)]));
        assert!(!in_words(&[(8, 16..PAGE), (16382, 3 * PAGE..4 * PAGE)]));
        // Runs a scatter writes twice, and runs that only touch.
        assert!(share_a_byte(&[8..20, 0..12]));
        assert!(share_a_byte(&[0..100, 40..44]));
        assert!(!share_a_byte(&[8..20, 0..8, 20..24]));
        assert!(!share_a_byte(&[]));
    }

    /// A scatter's pieces take one round where no byte is named twice, and a new round at each
    /// piece over bytes the round took.
    #[test]
    fn a_round_of_regions_writes_no_byte_twice() {
        let rows: Vec<Range<u64>> = (0..4).map(|y| y * 2048..y * 2048 + 2046).collect();
        assert_eq!(rounds(&rows), vec![vec![0, 1, 2, 3]]);
        // The second names the first's last two bytes; the fourth the third's whole, and the
        // first's (an earlier round's) too.
        let twice = [16..100, 96..200, 300..400, 0..400];
        assert_eq!(rounds(&twice), vec![vec![0], vec![1, 2], vec![3]]);
        // Touching ranges share no byte.
        assert_eq!(rounds(&[0..4, 4..8, 8..12]), vec![vec![0, 1, 2]]);
        assert_eq!(rounds(&[]), Vec::<Vec<usize>>::new());
    }

    #[test]
    fn runs_are_cut_where_a_chunk_ends_and_refused_outside_the_ram() {
        let spans = ChunkSpans::new(&one(0..150 * MIB), super::super::CHUNK_BYTES);
        let runs = [64 * MIB - PAGE..64 * MIB + PAGE, 3 * PAGE..4 * PAGE];
        let (cut, total) = pieces(&laid(&runs), &spans).unwrap();
        assert_eq!(total, 3 * PAGE);
        assert_eq!(
            cut,
            vec![
                Piece {
                    slot: 0,
                    offset: 64 * MIB - PAGE,
                    length: PAGE,
                    at: 0,
                },
                Piece {
                    slot: 1,
                    offset: 0,
                    length: PAGE,
                    at: PAGE,
                },
                Piece {
                    slot: 0,
                    offset: 3 * PAGE,
                    length: PAGE,
                    at: 2 * PAGE,
                },
            ]
        );
        assert_eq!(
            pieces(&laid(&one(150 * MIB..150 * MIB + PAGE)), &spans),
            None
        );
        assert_eq!(
            pieces(&laid(&one(150 * MIB - PAGE..150 * MIB + PAGE)), &spans),
            None
        );
        // Runs each at a byte of their own: each piece at its run's byte, plus its bytes before.
        let rows = [(10 * PAGE, 64 * MIB - 8..64 * MIB + 8), (100, 4..12)];
        let (cut, total) = pieces(&rows, &spans).unwrap();
        assert_eq!(total, 24);
        let at: Vec<(usize, u64, u64, u64)> = (cut.iter())
            .map(|piece| (piece.slot, piece.offset, piece.length, piece.at))
            .collect();
        assert_eq!(
            at,
            vec![
                (0, 64 * MIB - 8, 8, 10 * PAGE),
                (1, 0, 8, 10 * PAGE + 8),
                (0, 4, 8, 100)
            ]
        );
    }

    /// The table's entries as (the import's element of its dispatch's array, the import's word,
    /// the buffer's word in its window, the words).
    fn entries_of(table: &[u8]) -> Vec<[u32; 4]> {
        (table.as_chunks::<4>().0.iter())
            .map(|word| u32::from_le_bytes(*word))
            .collect::<Vec<u32>>()
            .as_chunks::<4>()
            .0
            .to_vec()
    }

    /// The dispatches of a plan as (the group, the window, the first entry, the entries).
    fn shapes_of(dispatches: &[Dispatch]) -> Vec<(usize, u64, u32, u32)> {
        (dispatches.iter())
            .map(|d| (d.group, d.window, d.first_entry, d.entries))
            .collect()
    }

    /// The runs of every import an array holds are one dispatch, each entry naming its import's
    /// element; the imports past an array's take the next array's dispatch; an array of one is a
    /// dispatch per import.
    /// [`touched_ranges`] gives what [`chunk_ranges`] (with no bound) and [`share_a_byte`] of the
    /// pieces' host bytes give, for pieces of whole pages, pieces inside pages (rows that share a
    /// page and rows that overlap) and pieces that name a page twice, over several chunks.
    #[test]
    fn touched_ranges_are_the_chunks_ranges_and_their_overlap_in_one_pass() {
        const CHUNK: u64 = 64 * PAGE;
        let spans: Vec<Range<u64>> = (0..4)
            .map(|chunk| (1 << 30) + chunk * CHUNK..(1 << 30) + (chunk + 1) * CHUNK)
            .collect();
        let expected = |pieces: &[Piece]| {
            let host: Vec<Range<u64>> = (pieces.iter())
                .map(|piece| {
                    let start = spans[piece.slot].start + piece.offset;
                    start..start + piece.length
                })
                .collect();
            (chunk_ranges(pieces, usize::MAX), share_a_byte(&host))
        };
        let piece = |slot: usize, offset: u64, length: u64| Piece {
            slot,
            offset,
            length,
            at: 0,
        };
        // Distinct whole pages, out of order: their ranges, no overlap.
        let pages = [
            piece(2, 5 * PAGE, PAGE),
            piece(0, 0, 2 * PAGE),
            piece(2, 4 * PAGE, PAGE),
        ];
        assert_eq!(touched_ranges(&pages, &spans), expected(&pages));
        assert!(!touched_ranges(&pages, &spans).1);
        // Rows that share a page and no byte; then with one that overlaps another, one inside a
        // whole page, and a page named twice.
        let rows = [
            piece(1, 100, 1000),
            piece(1, 1100, 4000),
            piece(1, 3 * PAGE, PAGE),
        ];
        assert_eq!(touched_ranges(&rows, &spans), expected(&rows));
        assert!(!touched_ranges(&rows, &spans).1);
        for extra in [
            piece(1, 1000, 8),
            piece(1, 3 * PAGE + 8, 8),
            piece(1, 3 * PAGE, PAGE),
        ] {
            let more: Vec<Piece> = rows.iter().copied().chain([extra]).collect();
            assert_eq!(touched_ranges(&more, &spans), expected(&more));
            assert!(touched_ranges(&more, &spans).1);
        }
        // Random mixes of the three.
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move |below: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % below
        };
        for _ in 0..2000 {
            let count = 1 + next(24) as usize;
            let pieces: Vec<Piece> = (0..count)
                .map(|_| {
                    let slot = next(spans.len() as u64) as usize;
                    match next(2) {
                        0 => {
                            let first = next(63);
                            piece(slot, first * PAGE, (1 + next(64 - first - 1)) * PAGE)
                        }
                        _ => {
                            let offset = next(CHUNK / WORD - 1) * WORD;
                            piece(slot, offset, 1 + next((CHUNK - offset).min(3 * PAGE)))
                        }
                    }
                })
                .collect();
            assert_eq!(
                touched_ranges(&pieces, &spans),
                expected(&pieces),
                "{pieces:?}"
            );
        }
    }

    /// [`MarkedRuns`] reads a bitmap's runs as its bits one by one give them: runs inside a
    /// word, ending at its last bit, crossing into the next or through whole words, and the
    /// bitmap's last bit.
    #[test]
    fn marked_runs_are_the_runs_of_the_bits_one_by_one() {
        let by_bit = |bits: &[u64]| {
            let mut runs: Vec<Range<u64>> = Vec::new();
            for page in 0..bits.len() as u64 * 64 {
                if (bits[(page / 64) as usize] >> (page % 64)) & 1 == 0 {
                    continue;
                }
                match runs.last_mut() {
                    Some(last) if last.end == page * PAGE => last.end += PAGE,
                    _ => runs.push(page * PAGE..(page + 1) * PAGE),
                }
            }
            runs
        };
        let read = |bits: &[u64]| MarkedRuns::new(bits).collect::<Vec<_>>();
        for bits in [
            vec![],
            vec![0],
            vec![1 << 63, 1],
            vec![u64::MAX, u64::MAX, 7, 0, 1 << 63],
            vec![0, u64::MAX, u64::MAX],
            vec![0b1011_0110, 0, 0, u64::MAX << 1, !1],
        ] {
            assert_eq!(read(&bits), by_bit(&bits), "{bits:x?}");
        }
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..500 {
            let bits: Vec<u64> = (0..1 + next() % 6)
                .map(|_| match next() % 4 {
                    0 => 0,
                    1 => u64::MAX,
                    2 => next() & next(),
                    _ => u64::MAX << (next() % 64),
                })
                .collect();
            assert_eq!(read(&bits), by_bit(&bits), "{bits:x?}");
        }
    }

    /// [`merges_to`] answers as [`merged`] of the runs, then a compare, answers: for runs laid one
    /// after another and runs each at their own byte, of runs that join, runs apart, an empty run,
    /// a run off a word, runs laid past `u64`; against the runs they merge to, those with a run
    /// more or the last run fewer, those of the runs less their last, and those with a run
    /// changed.
    #[test]
    fn runs_merged_as_they_are_compared_match_the_kept_runs_exactly_as_merged_gives_them() {
        let mut seed = 0x853c_49e6_748f_ea9bu64;
        let mut next = move |below: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % below
        };
        let answers = |given: Given<'_>, kept: &[Run]| {
            let merged = given.to_vec().and_then(|runs| merged(&runs));
            assert_eq!(
                merges_to(given, kept),
                merged.as_deref() == Some(kept),
                "{given:?} against {kept:?}"
            );
        };
        let check = |given: Given<'_>, less_last: Given<'_>| {
            let merge = given.to_vec().and_then(|runs| merged(&runs));
            let kept = merge.clone().unwrap_or_default();
            answers(given, &kept);
            let mut more = kept.clone();
            more.push((u64::MAX - 8, 0..4));
            answers(given, &more);
            if let Some((_, fewer)) = kept.split_last() {
                answers(given, fewer);
            }
            let shorter = less_last.to_vec().and_then(|runs| merged(&runs));
            answers(given, &shorter.unwrap_or_default());
            for at in 0..kept.len() {
                let mut changed = kept.clone();
                changed[at].1.end += WORD;
                answers(given, &changed);
            }
            if merge.is_some() {
                assert!(merges_to(given, &kept), "{given:?} merges to its runs");
            }
        };
        for _ in 0..3000 {
            let count = next(7) as usize;
            let mut laid: Vec<Range<u64>> = Vec::new();
            let mut own: Vec<Run> = Vec::new();
            let mut at = next(4) * WORD;
            for _ in 0..count {
                let start = match (laid.last(), next(4)) {
                    // Where the last ended: joins it.
                    (Some(last), 0 | 1) => last.end,
                    _ => next(64) * WORD,
                };
                let start = match next(16) {
                    0 => start + 2,
                    _ => start,
                };
                let length = match next(16) {
                    0 => 0,
                    _ => 1 + next(3 * WORD),
                };
                laid.push(start..start + length);
                own.push((at, start..start + length));
                at = match next(4) {
                    0 => at + 2 * length,
                    _ => at + length,
                };
            }
            let offset = match next(32) {
                0 => u64::MAX - 2,
                _ => next(4) * WORD,
            };
            let less = laid.len().saturating_sub(1);
            check(
                Given::Laid(offset, &laid),
                Given::Laid(offset, &laid[..less]),
            );
            check(Given::AtOwn(&own), Given::AtOwn(&own[..less]));
        }
        // No runs merge to none.
        assert!(merges_to(Given::AtOwn(&[]), &[]));
        assert!(!merges_to(Given::AtOwn(&[]), &[(0, 0..4)]));
    }

    /// A plan of a copy of `runs` into a buffer named `buffer`, its table `table` bytes.
    fn kept_plan(buffer: ObjectId, runs: &[Run], table: usize) -> KeptPlan {
        KeptPlan {
            copy: PagesCopy::Gather,
            consecutive: None,
            buffer: (buffer, 0, 1 << 20),
            runs: runs.into(),
            blit: (vk::Buffer::null(), 0),
            pages: None,
            window: Window {
                bytes: 1 << 20,
                align: 256,
                most_groups: 65535,
                array: 1,
            },
            span: 0..PAGE,
            total: PAGE,
            ranges: BTreeMap::new(),
            touched: Vec::new(),
            imports: vec![1],
            of_chunk: Vec::new(),
            table: vec![0; table],
            dispatches: Vec::new(),
            table_kept: OnceLock::new(),
        }
    }

    /// A RAM keeps at most [`MOST_PLANS`] plans, of at most [`MOST_PLAN_BYTES`] together, the
    /// one taken longest ago dropped first; a plan of the same copy takes the place of the one
    /// before it, and one past the bytes alone is not kept.
    #[test]
    fn a_ram_keeps_its_last_taken_plans_within_their_bounds() {
        let buffer = ObjectId::new();
        let runs = |n: u64| vec![(0, n * PAGE..(n + 1) * PAGE)];
        let kept = |plans: &KeptPlans| -> Vec<u64> {
            (plans.plans.iter())
                .map(|plan| plan.runs[0].1.start / PAGE)
                .collect()
        };
        let mut plans = KeptPlans::default();
        for n in 0..MOST_PLANS as u64 {
            plans.keep(kept_plan(buffer, &runs(n), 64));
        }
        assert_eq!(
            kept(&plans),
            (0..MOST_PLANS as u64).rev().collect::<Vec<_>>()
        );
        // The oldest taken now: the next plan drops the one taken longest ago, the second.
        let oldest = Arc::clone(plans.plans.last().unwrap());
        plans.taken(&oldest);
        plans.keep(kept_plan(buffer, &runs(100), 64));
        assert_eq!(plans.plans.len(), MOST_PLANS);
        assert_eq!(kept(&plans)[..2], [100, 0]);
        assert!(
            !kept(&plans).contains(&1),
            "the one taken longest ago dropped"
        );
        // The same copy again: in place of its plan, first.
        plans.keep(kept_plan(buffer, &runs(5), 64));
        assert_eq!(plans.plans.len(), MOST_PLANS);
        assert_eq!(kept(&plans)[0], 5);
        assert_eq!(kept(&plans).iter().filter(|&&n| n == 5).count(), 1);
        let held: usize = plans.plans.iter().map(|plan| plan.bytes()).sum();
        assert_eq!(plans.bytes, held, "its count of bytes");
        // Plans of a third of the bytes each: two fit with a small one, a third drops the oldest.
        let mut big = KeptPlans::default();
        for n in 0..3 {
            big.keep(kept_plan(buffer, &runs(n), MOST_PLAN_BYTES / 3));
        }
        assert_eq!(kept(&big), [2, 1]);
        assert!(big.bytes <= MOST_PLAN_BYTES);
        // One past the bytes alone: not kept, and the others stay.
        big.keep(kept_plan(buffer, &runs(9), MOST_PLAN_BYTES + 1));
        assert_eq!(kept(&big), [2, 1]);
        // Another buffer's copy of the same runs is another copy.
        big.keep(kept_plan(ObjectId::new(), &runs(2), 64));
        assert_eq!(kept(&big).iter().filter(|&&n| n == 2).count(), 2);
    }

    #[test]
    fn a_plan_takes_one_dispatch_per_array_of_imports_with_every_page_in_order() {
        let pieces = [
            Piece {
                slot: 1,
                offset: 2 * PAGE,
                length: 2 * PAGE,
                at: 0,
            },
            Piece {
                slot: 0,
                offset: 0,
                length: PAGE,
                at: 2 * PAGE,
            },
            Piece {
                slot: 1,
                offset: 0,
                length: PAGE,
                at: 3 * PAGE,
            },
        ];
        // Both imports in one array: one dispatch, the entries in the pieces' order. Whole pages:
        // an entry a page, of a page's 1024 words.
        let (table, dispatches) = plan(&pieces, 1 << 32, 65535, 2).unwrap();
        assert_eq!(
            entries_of(&table),
            vec![
                [1, 2048, 0, 1024],
                [1, 3072, 1024, 1024],
                [0, 0, 2048, 1024],
                [1, 0, 3072, 1024]
            ]
        );
        assert_eq!(
            dispatches,
            vec![Dispatch {
                group: 0,
                window: 0,
                first_entry: 0,
                entries: 4,
            }]
        );
        // Arrays of one: a dispatch per import, each its element 0.
        let (table, dispatches) = plan(&pieces, 1 << 32, 65535, 1).unwrap();
        assert_eq!(
            entries_of(&table),
            vec![
                [0, 0, 2048, 1024],
                [0, 2048, 0, 1024],
                [0, 3072, 1024, 1024],
                [0, 0, 3072, 1024]
            ]
        );
        assert_eq!(shapes_of(&dispatches), vec![(0, 0, 0, 1), (1, 0, 1, 3)]);
        // A third import past an array of two: the next array's dispatch, its element 0.
        let third = Piece {
            slot: 2,
            offset: 5 * PAGE,
            length: PAGE,
            at: 4 * PAGE,
        };
        let three = [pieces[0], pieces[1], pieces[2], third];
        let (table, dispatches) = plan(&three, 1 << 32, 65535, 2).unwrap();
        assert_eq!(entries_of(&table)[4], [0, 5 * 1024, 4 * 1024, 1024]);
        assert_eq!(shapes_of(&dispatches), vec![(0, 0, 0, 4), (1, 0, 4, 1)]);
        // Windows of two pages, and at most two workgroups a dispatch: by array of imports and
        // window.
        let (_, dispatches) = plan(&pieces, 2 * PAGE, 2, 1).unwrap();
        assert_eq!(
            shapes_of(&dispatches),
            vec![(0, 1, 0, 1), (1, 0, 1, 2), (1, 1, 3, 1)]
        );
        let (_, dispatches) = plan(&pieces, 2 * PAGE, 2, 2).unwrap();
        assert_eq!(shapes_of(&dispatches), vec![(0, 0, 0, 2), (0, 1, 2, 2)]);
    }

    /// Runs of words: each cut into runs of at most a page from its start, and where a window of
    /// the buffer ends.
    #[test]
    fn a_plan_cuts_runs_of_words_at_a_page_and_where_a_window_ends() {
        let pieces = [
            // A row of 2044 bytes 16 bytes into a page.
            Piece {
                slot: 0,
                offset: 16,
                length: 2044,
                at: 0,
            },
            // Two pages and 100 bytes from a word into a page, across the window's end at 2
            // pages of the buffer.
            Piece {
                slot: 0,
                offset: 5 * PAGE + 4,
                length: 2 * PAGE + 100,
                at: 2044,
            },
            // A word, in the second window.
            Piece {
                slot: 0,
                offset: 9 * PAGE,
                length: 4,
                at: 2044 + 2 * PAGE + 100,
            },
        ];
        let (table, dispatches) = plan(&pieces, 2 * PAGE, 65535, 30).unwrap();
        assert_eq!(
            entries_of(&table),
            vec![
                [0, 4, 0, 511],
                // The second piece: a page from its start, what is left of the window, then the
                // rest from the next window's first word.
                [0, 5121, 511, 1024],
                [0, 6145, 1535, 513],
                [0, 6658, 0, 536],
                [0, 9216, 536, 1],
            ]
        );
        assert_eq!(shapes_of(&dispatches), vec![(0, 0, 0, 3), (0, 1, 3, 2)]);
    }

    #[test]
    fn each_chunk_s_imports_are_found_by_chunk_in_the_order_they_were_given() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |below: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % below
        };
        for _ in 0..200 {
            // Imports in order, each holding bytes of one chunk or, as a list does, several.
            let mut of_chunk: Vec<(usize, usize)> = Vec::new();
            let mut by_map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for index in 0..next(24) as usize {
                let first = next(16) as usize;
                for slot in first..first + 1 + next(3) as usize {
                    of_chunk.push((slot, index));
                    by_map.entry(slot).or_default().push(index);
                }
            }
            let mut by_chunk = of_chunk.clone();
            by_chunk.sort_by_key(|&(slot, _)| slot);
            for slot in 0..20 {
                let found: Vec<usize> = imports_of(&by_chunk, slot).collect();
                assert_eq!(found, by_map.get(&slot).cloned().unwrap_or_default());
            }
        }
    }

    #[test]
    fn a_chunks_uses_join_touching_bytes_and_the_closest_past_the_most() {
        let piece = |offset: u64, length: u64| Piece {
            slot: 0,
            offset,
            length,
            at: 0,
        };
        // Descending neighbours touch: one range.
        let down = [
            piece(3 * PAGE, PAGE),
            piece(2 * PAGE, PAGE),
            piece(PAGE, PAGE),
        ];
        assert_eq!(chunk_ranges(&down, 4)[&0], vec![PAGE..4 * PAGE]);
        // Five apart, at most three: the two narrowest gaps join.
        let apart = [
            piece(0, PAGE),
            piece(2 * PAGE, PAGE),
            piece(10 * PAGE, PAGE),
            piece(12 * PAGE, PAGE),
            piece(30 * PAGE, PAGE),
        ];
        assert_eq!(
            chunk_ranges(&apart, 3)[&0],
            vec![0..3 * PAGE, 10 * PAGE..13 * PAGE, 30 * PAGE..31 * PAGE]
        );
    }

    #[test]
    fn a_copy_s_own_imports_join_bytes_less_than_a_huge_page_apart_and_no_more() {
        let piece = |slot: usize, offset: u64| Piece {
            slot,
            offset,
            length: PAGE,
            at: 0,
        };
        // Out of order, two within 2 MiB of each other, one 2 MiB past the second's end, and
        // one in another chunk.
        let pieces = [
            piece(0, 3 * MIB),
            piece(0, 0),
            piece(0, MIB),
            piece(1, 0),
            piece(0, 3 * MIB + PAGE + 2 * MIB),
        ];
        let ranges = chunk_ranges(&pieces, usize::MAX);
        assert_eq!(
            alone_ranges(&ranges[&0]),
            vec![
                0..3 * MIB + PAGE,
                3 * MIB + PAGE + 2 * MIB..3 * MIB + 2 * PAGE + 2 * MIB
            ]
        );
        assert_eq!(alone_ranges(&ranges[&1]), vec![0..PAGE]);
    }

    #[test]
    fn a_copy_s_lists_take_its_pages_alone_by_address_and_split_at_their_most() {
        // A chunk of 16 KiB pages from host address 1 GiB: a row inside one page, two rows that
        // share a page, a run that ends where the next page starts and one that starts there,
        // a run past the chunk's end cut at it, and a page apart.
        const HOST_PAGE: u64 = 4 * PAGE;
        let span = (1 << 30)..(1 << 30) + 8 * HOST_PAGE;
        let ranges = [
            16..16 + 854,
            HOST_PAGE + 8..HOST_PAGE + 100,
            HOST_PAGE + 200..HOST_PAGE + 300,
            2 * HOST_PAGE + 4..3 * HOST_PAGE,
            3 * HOST_PAGE..3 * HOST_PAGE + 4,
            6 * HOST_PAGE + 4..6 * HOST_PAGE + 8,
        ];
        let at = |page: u64| span.start + page * HOST_PAGE;
        let runs = page_runs(&ranges, &span, HOST_PAGE);
        // Pages that share or touch are one run; a page between keeps two apart.
        assert_eq!(runs, vec![at(0)..at(4), at(6)..at(7)]);
        let wide = page_runs(&one(7 * HOST_PAGE..9 * HOST_PAGE), &span, HOST_PAGE);
        assert_eq!(wide, vec![at(7)..at(8)], "cut at the chunk's end");
        // 1,500 runs of a page each, 1,024 a list: two lists, in order.
        let many: Vec<Range<u64>> = (0..1500u64)
            .map(|index| at(2 * index)..at(2 * index + 1))
            .collect();
        let lists = run_lists(&many, 1024, u64::MAX).expect("lists");
        assert_eq!(lists, vec![0..1024, 1024..1500]);
        // At most three pages a list: a run of two starts the next list where it would pass.
        let pages = [at(0)..at(1), at(2)..at(4), at(5)..at(6), at(7)..at(8)];
        let by_bytes = run_lists(&pages, 1024, 3 * HOST_PAGE).expect("lists");
        assert_eq!(by_bytes, vec![0..2, 2..4]);
        assert_eq!(
            run_lists(&pages, 1024, HOST_PAGE),
            None,
            "a run past a list"
        );
        assert_eq!(run_lists(&[], 1024, HOST_PAGE), Some(vec![]));
    }
}
