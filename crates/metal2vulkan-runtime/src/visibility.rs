//! Visibility results (`setVisibilityResultMode:offset:`).
//!
//! Metal writes, at the end of a pass, one 64-bit result at each offset the encoder set a mode at:
//! in `Boolean` mode a nonzero value when any sample of the draws while it was set passed and 0
//! otherwise, in `Counting` mode the samples that passed; an offset set with no draw after it gets
//! 0, an offset the encoder never set keeps its bytes, and a later pass rewrites only its own
//! offsets (Metal's answers on an Apple M3 Max with macOS 27.0.1, which the conformance suite's
//! `visibility` cases record). The nonzero `Boolean` value is the device's: an Apple M3 Max writes
//! 1, Metal on the AMD Radeon Pro 5300M and the Intel UHD Graphics 630 (macOS 26.7.1) the samples
//! that passed, and the runtime the device's occlusion query's result
//! (`Degradation::DeviceBooleanVisibility`).
//!
//! - **Queries.** Each mode set is an occlusion query of the encoder's own pools, begun in the
//!   draws' secondary command buffer when the mode is set and ended when the next one is set, at
//!   `Disabled`, or at `end_encoding`: one query spans every draw of its offset. A `Counting`
//!   query is precise (`VK_QUERY_CONTROL_PRECISE_BIT`): its result is the samples that passed,
//!   Metal's count. A `Boolean` query is imprecise: Vulkan states its result only as zero or not
//!   ("When any samples are passed, boolean queries will return a non-zero result value,
//!   otherwise a result value of zero is returned", `occlusionQueryPrecise`), which is Metal's
//!   `Boolean` contract, so its result is the offset's word as the device writes it.
//!   `end_encoding` resets the pools before the pass (query commands run in submission order, so
//!   the queries begun later see the reset).
//! - **Continuations.** A query begins and ends in one subpass, so where the pass splits into
//!   render pass instances (`render.rs`) while a mode is set, the active query ends in the
//!   instance before and a continuation query of the same offset begins in the one after.
//!   A draw every fragment of which fails Metal's depth test (a compare of `Never` without a
//!   depth attachment) counts no sample, where Vulkan's fragments would pass: the active query
//!   ends before it and a continuation begins after it (a probe). An execution of render commands
//!   ends the active query before it and continues it after it, and its own commands count in a
//!   continuation lent to it (`Lent`), which its recording begins and ends
//!   (`render/executions.rs`).
//! - **Results.** After the pass every query's 64-bit result is copied to its offset
//!   (`vkCmdCopyQueryPoolResults` waiting for each), consecutive ones in one copy, and every
//!   continuation's to a word of an upload region, whose words one dispatch of the crate-owned
//!   `bind::visibility_combine_module` combines into their offsets'
//!   (`bind::VisibilityCombine::plan`; more only where the offsets span more than one storage
//!   binding reaches): summed for a `Counting` offset, ORed for a `Boolean` one, since an
//!   imprecise query's count is any nonzero value and a sum of such counts could wrap to zero. A
//!   pass with no continuation records no dispatch. Where a no-copy buffer's bytes start at no
//!   multiple of 8 of its `VkBuffer`, the results land in an upload region first and are copied on
//!   (`Landing`).
//! - **`Accumulate`** (`MTLVisibilityResultTypeAccumulate`, the pass descriptor's
//!   `visibility_result_type`): no query is copied to its offset; every one, the first included,
//!   is copied to the upload region and the combine takes them into the word as it stands, so a
//!   `Counting` word's low 32 bits gain the pass's count, wrapping there, its high 32 bits kept, a
//!   `Boolean` word ORs the device's results into it, and an offset set with no draw after it
//!   keeps its word: an Apple M3 Max's answers (macOS 27.0.1, conformance cases
//!   `visibility-counting-accumulate*` and `visibility-boolean-accumulate`: 100 and a covering
//!   draw of 16 samples write 116, 0xffff_fff8 and 16 write 8, 0xffff_ffff_ffff_fff8 and 16 write
//!   0xffff_ffff_0000_0008). Where the results land in an upload region, the words are copied into
//!   it before the combine.
//! - **Refusals** (`Invalid`, measured): a pass without a visibility result buffer, an offset
//!   whose 8 bytes run past the buffer, and an offset the encoder took before in the other mode.
//!   `Counting` needs a precise query: `Unsupported`, naming `occlusionQueryPrecise`, where the
//!   device lacks it.
//! - **A driver that counts per color** (`RuntimeDriverDefect::OcclusionCountPerColorOutput`,
//!   ANV): its precise query counts each sample once for every color output the fragment shader
//!   writes. A draw counted by a `Counting` query whose fragment variant writes more than one
//!   color attachment runs, and the device announces `Degradation::VisibilityCountPerColor`: its
//!   result is the samples times those colors, not divided back. `Boolean` results keep their
//!   contract: zero where no sample passed.
//! - **A device that may count after a fragment function's mask** (no
//!   `earlyFragmentMultisampleCoverageAfterSampleCounting`, ANV): a draw under a visibility
//!   result whose fragment function declares `[[early_fragment_tests]]` and writes its
//!   `SampleMask` output (Metal's private sample mask or coverage, which the function applies
//!   itself there, or its own `[[sample_mask]]` return) runs, counted in
//!   `WorkCounts::draws_counted_past_early_masks`, and the device announces
//!   `Degradation::VisibilityPastEarlyMask`: Metal counts every sample the early tests pass, and
//!   the device's query may leave out those the mask drops. Keeping Metal's count would take a
//!   second draw of each counted draw without the mask, as every form that masks the colors
//!   alone (the `SampleMask` output, a discard) is a fragment operation such a device may count
//!   after.
//! - **An offset taken again in its mode** stops Metal's validation ("using the same offset ...
//!   twice in the same command encoder is not allowed"); plain Metal adds the draws since to its
//!   result: `Counting` sums them (16 and 16 write 32), `Boolean` writes 1 when either passed a
//!   sample, `Disabled` between them or not (Metal probe cases `vis_revisit_counting`,
//!   `vis_revisit_boolean_then_none`, `vis_revisit_none_then_boolean` and
//!   `vis_revisit_after_disabled`). The encoder takes it as a continuation of the offset's result.
//!   In the other mode plain Metal answers in order (`Boolean` then `Counting` writes 1 plus the
//!   count, `Counting` then `Boolean` 1: `vis_boolean_then_counting`, `vis_counting_then_boolean`),
//!   which the combine does not build: refused.
//! - **An offset that is not a multiple of 8** stops Metal's validation; plain Metal writes the
//!   result at the offset rounded down to 8 (Metal probe case `visibility_misaligned`: `Counting`
//!   at 4 writes the samples at 0), and the encoder takes that offset (`result_offset`).
//! - **`Disabled`** ends the current query. Its offset is no use of the offset: Metal takes it past
//!   the buffer, twice, before a mode at the same offset, and without a buffer, and plain Metal at
//!   an offset that is not a multiple of 8, which its validation stops (a Metal probe).

use super::degradation::Degradation;
use super::encoder::EncoderCore;
use super::error::{internal, invalid, vulkan, Error, Problem};
use super::exit_guard;
use super::owned_kernel::{OwnedModule, OwnedSets};
use super::release;
use super::tracker::{Access, Barrier, Use, UseRange};
use super::{Buffer, Device};
use ash::vk;
use metal2vulkan::bind::{ContinuedResult, VisibilityCombine};
use metal2vulkan::device::VkFeature;
use metal2vulkan::error::{Capability, Feature, Unsupported};
use std::collections::{BTreeMap, BTreeSet};

/// `MTLVisibilityResultMode`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VisibilityResultMode {
    /// `MTLVisibilityResultModeDisabled`: the draws write no result.
    #[default]
    Disabled,
    /// `MTLVisibilityResultModeBoolean`: nonzero when any sample passed, else 0. The nonzero value
    /// is the device's occlusion query's, where an Apple M3 Max writes 1
    /// (`Degradation::DeviceBooleanVisibility`).
    Boolean,
    /// `MTLVisibilityResultModeCounting`: the samples that passed.
    Counting,
}

/// `MTLVisibilityResultType`: whether a pass's visibility results replace the words at their
/// offsets or are taken into them (the module docs). The pass descriptor's
/// [`visibility_result_type`](crate::RenderPassDescriptor::visibility_result_type).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VisibilityResultType {
    /// `MTLVisibilityResultTypeReset`, Metal's default: each offset the encoder set a mode at
    /// takes the pass's result in place of its word.
    #[default]
    Reset,
    /// `MTLVisibilityResultTypeAccumulate`: each offset the encoder set a mode at keeps its word
    /// and takes the pass's result into it, a `Counting` result added to the word's low 32 bits
    /// (wrapping there, its high 32 bits kept) and a `Boolean` one ORed, so an offset set with no
    /// draw after it keeps its word.
    ///
    /// As an Apple M3 Max answers it (macOS 27.0.1). Metal on the AMD Radeon Pro 5300M and the
    /// Intel UHD Graphics 630 (macOS 26.7.1) answers it as `Reset`.
    Accumulate,
}

impl VisibilityResultType {
    /// Metal's raw value.
    pub fn raw(self) -> i64 {
        match self {
            VisibilityResultType::Reset => 0,
            VisibilityResultType::Accumulate => 1,
        }
    }

    /// The type of Metal's raw value `raw`.
    pub fn from_raw(raw: i64) -> Result<VisibilityResultType, Error> {
        Ok(match raw {
            0 => VisibilityResultType::Reset,
            1 => VisibilityResultType::Accumulate,
            raw => {
                return Err(invalid(
                    "VisibilityResultType::from_raw",
                    Problem::UndefinedValue {
                        ty: "MTLVisibilityResultType",
                        raw: raw as u64,
                    },
                ))
            }
        })
    }
}

impl VisibilityResultMode {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            VisibilityResultMode::Disabled => 0,
            VisibilityResultMode::Boolean => 1,
            VisibilityResultMode::Counting => 2,
        }
    }

    /// The mode of Metal's raw value `raw`.
    pub fn from_raw(raw: u64) -> Result<VisibilityResultMode, Error> {
        Ok(match raw {
            0 => VisibilityResultMode::Disabled,
            1 => VisibilityResultMode::Boolean,
            2 => VisibilityResultMode::Counting,
            raw => {
                return Err(invalid(
                    "VisibilityResultMode::from_raw",
                    Problem::UndefinedValue {
                        ty: "MTLVisibilityResultMode",
                        raw,
                    },
                ))
            }
        })
    }
}

/// What a `set_visibility_result_mode` did wrong (each measured: Metal stops it).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VisibilityProblem {
    /// The pass has no visibility result buffer: Metal stops it with or without API validation.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal's process ends, SIGSEGV,
    /// at `setVisibilityResultMode:offset:` in a pass with no visibility result buffer).
    NoBuffer,
    /// An offset whose 8 bytes run past the buffer's `length`.
    ///
    /// Basis: metal-takes (an Apple M3 Max, macOS 27.0.1: plain Metal takes
    /// `setVisibilityResultMode:Boolean offset:64` on an 8-byte buffer and the pass ends, its
    /// validation alone saying "offset(64) + result-size(8) must be <= `[visibilityResultBuffer
    /// length](8)`").
    OffsetOutOfRange {
        /// The offset given.
        offset: u64,
        /// The visibility result buffer's length.
        length: u64,
    },
    /// An offset the encoder took before in the other mode ("using the same offset ... twice in
    /// the same command encoder is not allowed"), `Disabled` between them or not. In its own mode
    /// the offset continues its result, as plain Metal's (the module docs).
    ///
    /// Basis: metal-documented (Metal asserts "using the same offset ... twice in the same command
    /// encoder is not allowed").
    OffsetReused {
        /// The offset given.
        offset: u64,
    },
}

/// Bytes of one result.
const RESULT_BYTES: u64 = 8;

/// The offset a result of `setVisibilityResultMode:offset:` at `offset` lands at: rounded down to
/// a multiple of 8, as plain Metal writes it (the module docs).
pub(super) fn result_offset(offset: u64) -> u64 {
    offset - offset % RESULT_BYTES
}

/// Queries of an encoder's first pool; each later pool doubles, up to [`MAX_POOL`].
const FIRST_POOL: u32 = 16;
const MAX_POOL: u32 = 1024;

/// One occlusion query pool of an encoder.
struct QueryPool {
    device: Device,
    raw: vk::QueryPool,
    capacity: u32,
    /// Queries taken, from 0.
    used: u32,
}

impl Drop for QueryPool {
    fn drop(&mut self) {
        // Once the process is exiting, the pool is left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // SAFETY: no pending command uses the pool (a command buffer retains it until it
        // completes).
        unsafe { release::object(self.device.shared().vk(), self.raw) };
    }
}

/// The pools of one encoder, held until its command buffer completes.
struct QueryPools(Vec<QueryPool>);

/// One offset an encoder took: its mode and its query.
#[derive(Clone, Copy, Debug)]
struct Taken {
    offset: u64,
    boolean: bool,
    pool: usize,
    query: u32,
}

/// A query that continues the active one in a later render pass instance: its pool and query,
/// taken before the split records ([`Visibility::continuation`]). `pool` is `None` for one of an
/// execution's spares ([`Lent`]), which the lending encoder took as a continuation of its offset
/// already.
#[derive(Clone, Copy, Debug)]
pub(super) struct Continuation {
    pool: Option<usize>,
    raw: vk::QueryPool,
    query: u32,
}

/// A continuation begun: the offset it continues ([`Taken`], by index) and its query.
#[derive(Clone, Copy, Debug)]
struct Continued {
    taken: usize,
    pool: usize,
    query: u32,
}

/// The query an execution of render commands encoded while a query was active counts its
/// commands' draws in (`render/executions.rs`): a continuation of the active offset, which the
/// execution's recording begins and ends in its own secondary command buffer
/// ([`Visibility::borrowing`]). Its pool lives until the command buffer completes.
/// Its `spares` are continuations of the same offset, one for each time a command's draw may end
/// the query and begin another: where it begins another run of the execution, and after a draw that
/// counts no sample. The recording begins and ends each spare it takes, and every other one empty,
/// counting nothing.
#[derive(Clone, Debug)]
pub(super) struct Lent {
    pool: vk::QueryPool,
    query: u32,
    boolean: bool,
    spares: Vec<(vk::QueryPool, u32)>,
}

impl Lent {
    /// Begin the lent query in `draws`, before what counts in it: a draw recorded once its
    /// `VkPipeline` is made (`render/pending.rs`), which lends no spare.
    ///
    /// # Safety
    ///
    /// `draws` is recording inside its render pass instance, with no query active in it; the
    /// pool lives until the command buffer completes, and the lending encoder's `end_encoding`
    /// reset it before the pass.
    pub(super) unsafe fn begin(&self, raw: &ash::Device, draws: vk::CommandBuffer) {
        // SAFETY: the caller's; a precise query's device enables `occlusionQueryPrecise`
        // (checked when its mode was set).
        unsafe { raw.cmd_begin_query(draws, self.pool, self.query, control(self.boolean)) };
    }

    /// End the lent query in `draws`, which began it ([`Lent::begin`]).
    ///
    /// # Safety
    ///
    /// As [`Lent::begin`]'s, after it.
    pub(super) unsafe fn end(&self, raw: &ash::Device, draws: vk::CommandBuffer) {
        // SAFETY: the caller's.
        unsafe { raw.cmd_end_query(draws, self.pool, self.query) };
    }
}

/// The queries an execution takes as it is encoded ([`Visibility::lend`]): the one its
/// commands' draws count in, its spares, and the one that continues the offset after it.
pub(super) struct Loan {
    lent: Continuation,
    spares: Vec<Continuation>,
    next: Continuation,
}

/// The lent query of an encoder replaying an execution's commands ([`Visibility::borrowing`]):
/// its mode, and the spares not yet taken, the next last.
struct Borrowed {
    boolean: bool,
    spares: Vec<(vk::QueryPool, u32)>,
}

/// One render encoder's visibility results.
pub(super) struct Visibility {
    /// The pass's visibility result buffer.
    buffer: Option<Buffer>,
    pools: Vec<QueryPool>,
    taken: Vec<Taken>,
    /// The continuations begun, in order.
    continued: Vec<Continued>,
    /// Each offset taken, with its index in `taken`.
    offsets: BTreeMap<u64, usize>,
    /// The index in `taken` of the offset the active query, or the one last ended, counts for.
    current: usize,
    /// The query begun in the draws' command buffer and not yet ended.
    active: Option<(vk::QueryPool, u32)>,
    /// The active query is an execution's [`Lent`] one, in an encoder replaying its commands: it
    /// has no pools to continue it from (But the lent spares).
    borrowed: Option<Borrowed>,
}

impl Visibility {
    /// No result yet, for a pass with `buffer`.
    pub(super) fn new(buffer: Option<Buffer>) -> Visibility {
        Visibility {
            buffer,
            pools: Vec::new(),
            taken: Vec::new(),
            continued: Vec::new(),
            offsets: BTreeMap::new(),
            current: 0,
            active: None,
            borrowed: None,
        }
    }

    /// The visibility of an encoder replaying an execution's commands into `draws`
    /// (`render/executions.rs`): with `lent`, its query begun in `draws` now, in which the
    /// commands' draws count. A split of the replay's instance, or a draw that counts no sample,
    /// while it is active continues it in the next of the lent spares
    /// ([`Visibility::continuation`]).
    pub(super) fn borrowing(
        buffer: Option<Buffer>,
        lent: Option<Lent>,
        raw: &ash::Device,
        draws: vk::CommandBuffer,
    ) -> Visibility {
        let mut out = Visibility::new(buffer);
        if let Some(lent) = lent {
            // SAFETY: `draws` is recording inside its render pass instance, and only the replay
            // records it; no query is active in it (a new one); the pool lives until the command
            // buffer completes, and the encoder's `end_encoding` reset it before the pass; a
            // precise query's device enables `occlusionQueryPrecise` (checked when its mode was
            // set).
            unsafe { raw.cmd_begin_query(draws, lent.pool, lent.query, control(lent.boolean)) };
            out.active = Some((lent.pool, lent.query));
            let mut spares = lent.spares;
            spares.reverse();
            out.borrowed = Some(Borrowed {
                boolean: lent.boolean,
                spares,
            });
        }
        out
    }

    /// The visibility of an encoder that settles an execution's draws at commit and records
    /// nothing (`render/executions.rs`): [`borrowing`](Visibility::borrowing)'s, active where
    /// `lent`, with no query begun.
    pub(super) fn settling(buffer: Option<Buffer>, lent: Option<&Lent>) -> Visibility {
        let mut out = Visibility::new(buffer);
        if let Some(lent) = lent {
            out.active = Some((lent.pool, lent.query));
            let mut spares = lent.spares.clone();
            spares.reverse();
            out.borrowed = Some(Borrowed {
                boolean: lent.boolean,
                spares,
            });
        }
        out
    }

    /// The queries an execution of render commands encoded while a query is active takes,
    /// before anything of it records: the one its commands count in, `spares` more of the same
    /// offset ([`Lent`]), and the one that continues the offset after it. `None` when no query
    /// is active.
    pub(super) fn lend(&mut self, device: &Device, spares: u64) -> Result<Option<Loan>, Error> {
        if self.active.is_none() {
            return Ok(None);
        }
        let lent = self.next_continuation(device)?;
        let spares = (0..spares)
            .map(|_| self.next_continuation(device))
            .collect::<Result<Vec<_>, _>>()?;
        let next = self.next_continuation(device)?;
        Ok(Some(Loan { lent, spares, next }))
    }

    /// Hand `loan`'s queries (from [`Visibility::lend`]) to an execution whose secondary
    /// command buffer stands between `before` and `after`: end the active query in `before`,
    /// record the execution's query and its spares as continuations of its offset, and begin the
    /// last of the loan in `after` for the same offset. The execution's recording begins and
    /// ends its query and spares ([`Visibility::borrowing`]).
    pub(super) fn lend_to(
        &mut self,
        raw: &ash::Device,
        before: vk::CommandBuffer,
        after: vk::CommandBuffer,
        loan: Loan,
    ) -> Lent {
        self.end(raw, before);
        let boolean = self.taken.get(self.current).is_none_or(|t| t.boolean);
        for taken in std::iter::once(&loan.lent).chain(&loan.spares) {
            if let Some(pool) = taken.pool {
                self.continued.push(Continued {
                    taken: self.current,
                    pool,
                    query: taken.query,
                });
            }
        }
        let out = Lent {
            pool: loan.lent.raw,
            query: loan.lent.query,
            boolean,
            spares: loan.spares.iter().map(|s| (s.raw, s.query)).collect(),
        };
        self.resume(raw, after, loan.next);
        out
    }

    /// End, empty, every spare of the lent query the replay did not take, in `draws`, the last
    /// run's secondary command buffer, after the active query ended: each counts nothing, and
    /// the combine reads a result from it.
    pub(super) fn close_spares(&mut self, raw: &ash::Device, draws: vk::CommandBuffer) {
        let Some(borrowed) = &mut self.borrowed else {
            return;
        };
        let boolean = borrowed.boolean;
        for (pool, query) in borrowed.spares.drain(..) {
            // SAFETY: `draws` is recording inside its render pass instance, and only the replay
            // records it; no query is active in it (ended before); the pool lives until the
            // command buffer completes, and the lending encoder's `end_encoding` reset it before
            // the pass; a precise query's device enables `occlusionQueryPrecise`.
            unsafe {
                raw.cmd_begin_query(draws, pool, query, control(boolean));
                raw.cmd_end_query(draws, pool, query);
            }
        }
    }

    /// Whether `setVisibilityResultMode:offset:` in `call` on `device`, at `offset` (a
    /// [`result_offset`]), is refused as the module docs say, apart from running out of query
    /// memory: [`Visibility::set_mode`]'s checks, which a caller makes before what it records
    /// for the query.
    pub(super) fn check_mode(
        &self,
        call: &'static str,
        device: &Device,
        mode: VisibilityResultMode,
        offset: u64,
    ) -> Result<(), Error> {
        let problem = |problem| invalid(call, Problem::Visibility(problem));
        if mode == VisibilityResultMode::Disabled {
            return Ok(());
        }
        let Some(buffer) = &self.buffer else {
            return Err(problem(VisibilityProblem::NoBuffer));
        };
        let length = buffer.length();
        if offset
            .checked_add(RESULT_BYTES)
            .is_none_or(|end| end > length)
        {
            return Err(problem(VisibilityProblem::OffsetOutOfRange {
                offset,
                length,
            }));
        }
        let boolean = mode == VisibilityResultMode::Boolean;
        if let Some(&index) = self.offsets.get(&offset) {
            if self.taken[index].boolean != boolean {
                return Err(problem(VisibilityProblem::OffsetReused { offset }));
            }
        }
        if mode != VisibilityResultMode::Boolean
            && !device
                .properties()
                .description
                .has(VkFeature::OcclusionQueryPrecise)
        {
            return Err(Error::Unsupported(Unsupported::needs(
                Feature::CountingVisibility,
                Capability::Feature(VkFeature::OcclusionQueryPrecise),
            )));
        }
        Ok(())
    }

    /// `setVisibilityResultMode:offset:` in `call` on `device`, recording into `draws`, the
    /// encoder's secondary command buffer. Refused as the module docs say; a refused call
    /// records nothing and keeps the mode set before.
    pub(super) fn set_mode(
        &mut self,
        call: &'static str,
        device: &Device,
        draws: vk::CommandBuffer,
        mode: VisibilityResultMode,
        offset: u64,
    ) -> Result<(), Error> {
        self.check_mode(call, device, mode, offset)?;
        let raw = device.shared().vk();
        if mode == VisibilityResultMode::Disabled {
            self.end(raw, draws);
            return Ok(());
        }
        let boolean = mode == VisibilityResultMode::Boolean;
        let (pool, query) = self.next_query(device)?;
        // Recording starts here: nothing below can be refused.
        self.end(raw, draws);
        let pool_raw = self.pools[pool].raw;
        // SAFETY: the secondary command buffer is recording inside the pass, and only this
        // encoder records it; no query is active in it (ended above); the pool lives until the
        // command buffer completes, and `end_encoding` resets it before the pass; a precise
        // query's device has `occlusionQueryPrecise` (checked above), which it enables.
        unsafe { raw.cmd_begin_query(draws, pool_raw, query, control(boolean)) };
        self.active = Some((pool_raw, query));
        // An offset taken again (in its mode, `check_mode`) continues its result.
        if let Some(&index) = self.offsets.get(&offset) {
            self.continued.push(Continued {
                taken: index,
                pool,
                query,
            });
            self.current = index;
            return Ok(());
        }
        self.current = self.taken.len();
        self.offsets.insert(offset, self.current);
        self.taken.push(Taken {
            offset,
            boolean,
            pool,
            query,
        });
        Ok(())
    }

    /// Whether a query is begun and not yet ended.
    pub(super) fn active(&self) -> bool {
        self.active.is_some()
    }

    /// Whether the active query counts samples: a `Counting` result's precise query, which the
    /// draws recorded now count in.
    pub(super) fn counting(&self) -> bool {
        self.active.is_some() && !self.boolean()
    }

    /// Whether the current offset's queries are `Boolean` ones: the borrowed execution's offset,
    /// else the one last set (`true` before any is set).
    fn boolean(&self) -> bool {
        match &self.borrowed {
            Some(borrowed) => borrowed.boolean,
            None => self.taken.get(self.current).is_none_or(|t| t.boolean),
        }
    }

    /// The query that will continue the active one in the render pass instance a split opens,
    /// taken now, before anything of the split records: `None` when no query is active. In an
    /// encoder replaying an execution's commands (`borrowing`), whose active query is the
    /// execution's lent one, the next of its spares.
    pub(super) fn continuation(&mut self, device: &Device) -> Result<Option<Continuation>, Error> {
        if self.active.is_none() {
            return Ok(None);
        }
        if let Some(borrowed) = &mut self.borrowed {
            let (raw, query) = borrowed
                .spares
                .pop()
                .ok_or_else(|| internal("an execution's draws past the queries lent to it"))?;
            return Ok(Some(Continuation {
                pool: None,
                raw,
                query,
            }));
        }
        self.next_continuation(device).map(Some)
    }

    /// The next free query (`next_query`), as a continuation.
    fn next_continuation(&mut self, device: &Device) -> Result<Continuation, Error> {
        let (pool, query) = self.next_query(device)?;
        Ok(Continuation {
            pool: Some(pool),
            raw: self.pools[pool].raw,
            query,
        })
    }

    /// End the active query in `from`, the draws' command buffer of the instance before a split,
    /// and begin `next` (from [`Visibility::continuation`]) in `into`, the next instance's, for
    /// the same offset.
    pub(super) fn continue_in(
        &mut self,
        raw: &ash::Device,
        from: vk::CommandBuffer,
        into: vk::CommandBuffer,
        next: Continuation,
    ) {
        self.end(raw, from);
        self.resume(raw, into, next);
    }

    /// Begin `next` (from [`Visibility::continuation`]) in `into`, continuing the offset of the
    /// query last ended: after a split ([`Visibility::continue_in`]), or after a draw that counts
    /// no sample, around which the active query ends (`render.rs`).
    pub(super) fn resume(
        &mut self,
        raw: &ash::Device,
        into: vk::CommandBuffer,
        next: Continuation,
    ) {
        let pool = next.raw;
        let boolean = self.boolean();
        // SAFETY: `into` is recording inside its render pass instance, and only this encoder
        // records it; no query is active in it (a new one, or the one the caller ended the
        // query in); the pool lives until the command buffer completes, and `end_encoding`
        // resets it before the pass (a spare's, the lending encoder's); a precise query's device
        // enables `occlusionQueryPrecise` (checked when its mode was set).
        unsafe { raw.cmd_begin_query(into, pool, next.query, control(boolean)) };
        self.active = Some((pool, next.query));
        // A spare is the lending encoder's continuation already (`lend_to`).
        if let Some(pool) = next.pool {
            self.continued.push(Continued {
                taken: self.current,
                pool,
                query: next.query,
            });
        }
    }

    /// End the active query, if any, in `draws`.
    pub(super) fn end(&mut self, raw: &ash::Device, draws: vk::CommandBuffer) {
        if let Some((pool, query)) = self.active {
            // SAFETY: the query was begun in `draws`, which is still recording inside the pass.
            unsafe { raw.cmd_end_query(draws, pool, query) };
            self.active = None;
        }
    }

    /// The next free query, in a new pool when the last is full.
    fn next_query(&mut self, device: &Device) -> Result<(usize, u32), Error> {
        let count = self.pools.len();
        if let Some(last) = self.pools.last_mut() {
            if last.used < last.capacity {
                last.used += 1;
                return Ok((count - 1, last.used - 1));
            }
        }
        let capacity = self
            .pools
            .last()
            .map_or(FIRST_POOL, |last| (last.capacity * 2).min(MAX_POOL));
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::OCCLUSION)
            .query_count(capacity);
        // SAFETY: the device is live and `info` valid.
        let raw = unsafe { device.shared().vk().create_query_pool(&info, None) }.map_err(vulkan)?;
        self.pools.push(QueryPool {
            device: device.clone(),
            raw,
            capacity,
            used: 1,
        });
        Ok((self.pools.len() - 1, 0))
    }

    /// The results' copies and combine, planned before anything of the pass records: `None`
    /// when the encoder took no offset. Under `kind` `Reset` each offset's first query is copied
    /// to its word and its continuations combined into it; under `Accumulate` every query of an
    /// offset is combined into the word as it stands. The draws' command buffer must have ended
    /// its query ([`Visibility::end`]).
    pub(super) fn plan(
        self,
        core: &mut EncoderCore,
        kind: VisibilityResultType,
    ) -> Result<Option<Results>, Error> {
        let Visibility {
            buffer,
            pools,
            taken,
            continued,
            active,
            ..
        } = self;
        if active.is_some() {
            return Err(internal(
                "a visibility query still active at the pass's end",
            ));
        }
        let Some(buffer) = buffer.filter(|_| !taken.is_empty()) else {
            return Ok(None);
        };
        let accumulate = kind == VisibilityResultType::Accumulate;
        let landing = Landing::new(core, &buffer, &taken, accumulate)?;
        // Consecutive queries of one pool at consecutive offsets copy at once. The copies and the
        // combine address the landing region; the uses, the Metal buffer.
        let mut copies: Vec<ResultCopy> = Vec::new();
        let mut sources = continued;
        if accumulate {
            // Every query is a source the combine takes into the word: the first ones too.
            sources.extend(taken.iter().enumerate().map(|(index, t)| Continued {
                taken: index,
                pool: t.pool,
                query: t.query,
            }));
        } else {
            for t in &taken {
                push_copy(
                    &mut copies,
                    pools[t.pool].raw,
                    t.query,
                    landing.raw,
                    landing.at(t.offset),
                );
            }
        }
        let combine = combine(
            core,
            (&pools, &taken, &sources),
            &landing,
            accumulate,
            &mut copies,
        )?;
        // The offsets whose words the combine reads and writes in place.
        let combined: BTreeSet<usize> = sources.iter().map(|c| c.taken).collect();
        let landed = !landing.onward.is_empty();
        let transfer = |access| Access::new(vk::PipelineStageFlags2::TRANSFER, access);
        let mut uses = Vec::with_capacity(taken.len());
        for (index, t) in taken.iter().enumerate() {
            let (identity, range) = buffer.tracked(t.offset..t.offset + RESULT_BYTES);
            let range = UseRange::Bytes(range);
            let (key, tracked) = (identity.key, identity.tracked);
            // Into the word: its first query's copy (`Reset`) or the onward copy.
            if !accumulate || landed {
                let write = transfer(vk::AccessFlags2::TRANSFER_WRITE);
                uses.push(Use::writes(key, tracked, range.clone(), write));
            }
            // Out of it, into the landing region (`Accumulate`).
            if accumulate && landed {
                let read = transfer(vk::AccessFlags2::TRANSFER_READ);
                uses.push(Use::reads(key, tracked, range.clone(), read));
            }
            // The combine, in place.
            if combined.contains(&index) && !landed {
                let shader = Access::new(
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                );
                uses.push(Use {
                    key,
                    tracked,
                    range,
                    read: shader,
                    write: shader,
                });
            }
        }
        let boolean = taken.iter().find(|t| t.boolean).map(|t| t.offset);
        Ok(Some(Results {
            pools: QueryPools(pools),
            buffer,
            landing,
            copies,
            combine,
            boolean,
            uses,
        }))
    }
}

/// Where a pass's results land, at multiples of 8 as Vulkan copies them: in the buffer's
/// `VkBuffer`, or, where a no-copy buffer's bytes start past such a multiple (`Buffer::base`), in
/// an upload region laid out as the buffer from the first offset taken, whence one more copy, which
/// takes any offset, moves each result into the buffer after the combine. Under `Accumulate` a copy
/// before the combine brings each word into the region.
struct Landing {
    /// The `VkBuffer` the results land in.
    raw: vk::Buffer,
    /// Where the first offset taken lands in `raw`.
    start: u64,
    /// The first offset taken.
    first: u64,
    /// The buffer's `VkBuffer`.
    to: vk::Buffer,
    /// The copies of the results on into the buffer, where `raw` is an upload's.
    onward: Vec<vk::BufferCopy>,
    /// The copies of the words into the upload region before the combine, where `raw` is an
    /// upload's and the results accumulate.
    inward: Vec<vk::BufferCopy>,
}

impl Landing {
    /// Where the results of `taken` (not empty) land, for `buffer`, taken into its words where
    /// they `accumulate`; the upload region is `core`'s command buffer's.
    fn new(
        core: &EncoderCore,
        buffer: &Buffer,
        taken: &[Taken],
        accumulate: bool,
    ) -> Result<Landing, Error> {
        let first = taken.iter().map(|t| t.offset).min().unwrap_or(0);
        let (to, start) = buffer.at(first);
        if start.is_multiple_of(RESULT_BYTES) {
            return Ok(Landing {
                raw: to,
                start,
                first,
                to,
                onward: Vec::new(),
                inward: Vec::new(),
            });
        }
        let end = taken.iter().map(|t| t.offset + RESULT_BYTES).max();
        let len = end.unwrap_or(first) - first;
        let device = core.device();
        let region = core
            .command_buffer()
            .state()
            .uploads
            .upload(device, &[], len)?;
        let onward: Vec<vk::BufferCopy> = taken
            .iter()
            .map(|t| vk::BufferCopy {
                src_offset: region.offset() + (t.offset - first),
                dst_offset: buffer.at(t.offset).1,
                size: RESULT_BYTES,
            })
            .collect();
        let inward = if accumulate {
            onward
                .iter()
                .map(|c| vk::BufferCopy {
                    src_offset: c.dst_offset,
                    dst_offset: c.src_offset,
                    size: RESULT_BYTES,
                })
                .collect()
        } else {
            Vec::new()
        };
        device.degrade(Degradation::LandedVisibilityResults, || {
            let object = format!("visibility results from offset {first}");
            let reason = format!(
                "they start at byte {start} of the buffer's VkBuffer, and Vulkan writes query \
                 results only at multiples of {RESULT_BYTES} bytes"
            );
            (object, reason)
        });
        Ok(Landing {
            raw: region.raw(),
            start: region.offset(),
            first,
            to,
            onward,
            inward,
        })
    }

    /// Where the result at `offset` of the buffer lands.
    fn at(&self, offset: u64) -> u64 {
        self.start + (offset - self.first)
    }
}

/// The control flags of a query of a `Boolean` offset (imprecise) or a `Counting` one (precise).
fn control(boolean: bool) -> vk::QueryControlFlags {
    if boolean {
        vk::QueryControlFlags::empty()
    } else {
        vk::QueryControlFlags::PRECISE
    }
}

/// One copy of query results.
#[derive(Clone, Copy, Debug)]
struct ResultCopy {
    pool: vk::QueryPool,
    first: u32,
    count: u32,
    /// The landing region's buffer, or the continuations' upload region's.
    target: vk::Buffer,
    /// In `target` (`Landing::at` for a result).
    offset: u64,
}

/// Add the copy of `query` of `pool` to `offset` of `target` to `copies`: into the last copy
/// where it continues it.
fn push_copy(
    copies: &mut Vec<ResultCopy>,
    pool: vk::QueryPool,
    query: u32,
    target: vk::Buffer,
    offset: u64,
) {
    if let Some(last) = copies.last_mut() {
        let next = u64::from(last.count) * RESULT_BYTES;
        if last.pool == pool
            && last.first + last.count == query
            && last.target == target
            && last.offset + next == offset
        {
            last.count += 1;
            return;
        }
    }
    copies.push(ResultCopy {
        pool,
        first: query,
        count: 1,
        target,
        offset,
    });
}

/// The copies of the queries `continued` of `pools` (continuations, and where the results
/// `accumulate` every query) of the offsets `taken`, added to `copies`, into an upload region of
/// `core`'s command buffer, one word each, those of one offset consecutive, and the combine
/// dispatches that OR them into their `Boolean` results' landed words and sum them into their
/// `Counting` ones, into the low half alone where they accumulate (as an Apple M3 Max writes it,
/// `bind::visibility_combine_module`): `None` without queries.
fn combine(
    core: &mut EncoderCore,
    (pools, taken, continued): (&[QueryPool], &[Taken], &[Continued]),
    landing: &Landing,
    accumulate: bool,
    copies: &mut Vec<ResultCopy>,
) -> Result<Option<Dispatches>, Error> {
    if continued.is_empty() {
        return Ok(None);
    }
    let device = core.device().clone();
    let words = continued.len() as u64 * RESULT_BYTES;
    let sources = {
        let mut state = core.command_buffer().state();
        state.uploads.upload(&device, &[], words)?
    };
    // An offset taken again continues after others' continuations: each offset's words are
    // gathered (the order within one is no matter, a sum or an OR).
    let mut order: Vec<&Continued> = continued.iter().collect();
    order.sort_by_key(|c| c.taken);
    let mut results: Vec<ContinuedResult> = Vec::new();
    for (word, c) in order.into_iter().enumerate() {
        let at = sources.offset() + word as u64 * RESULT_BYTES;
        push_copy(copies, pools[c.pool].raw, c.query, sources.raw(), at);
        let result = &taken[c.taken];
        let offset = landing.at(result.offset);
        match results.last_mut() {
            Some(last) if last.offset == offset => last.count += 1,
            _ => {
                let first =
                    u32::try_from(word).map_err(|_| internal("too many continuation queries"))?;
                let combined = ContinuedResult::new(offset, first, 1, !result.boolean);
                results.push(match accumulate && !result.boolean {
                    true => combined.in_low_half(),
                    false => combined,
                });
            }
        }
    }
    let limits = &device.properties().description.limits;
    let combines = VisibilityCombine::plan(
        &results,
        limits.min_storage_buffer_offset_alignment,
        limits.max_storage_buffer_range,
        limits.max_compute_work_group_count[0],
    )
    .ok_or_else(|| internal("visibility continuations the combine cannot bind"))?;
    let kernel = device.owned_kernel(OwnedModule::VisibilityCombine)?;
    let mut ranges: Vec<[(vk::Buffer, u64, u64); 3]> = Vec::with_capacity(combines.len());
    {
        let mut state = core.command_buffer().state();
        for c in &combines {
            let bytes = c.table_bytes();
            let table = state.uploads.upload(&device, &bytes, bytes.len() as u64)?;
            ranges.push([
                (table.raw(), table.offset(), bytes.len() as u64),
                (landing.raw, c.dest_range.offset, c.dest_range.range),
                (sources.raw(), sources.offset(), words),
            ]);
        }
    }
    let sets = OwnedSets::write(&device, &kernel, &ranges)?;
    Ok(Some(Dispatches {
        pipeline: kernel.pipeline,
        layout: kernel.layout,
        sets,
        groups: combines.iter().map(|c| c.group_count).collect(),
    }))
}

/// The dispatches of the crate-owned combine after a pass.
struct Dispatches {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    sets: OwnedSets,
    groups: Vec<u32>,
}

/// A pass's visibility results, planned: what records before the pass and after it.
pub(super) struct Results {
    pools: QueryPools,
    buffer: Buffer,
    landing: Landing,
    copies: Vec<ResultCopy>,
    combine: Option<Dispatches>,
    /// The first offset taken in `Boolean` mode, whose word is the device's own nonzero value
    /// where a sample passed (`Degradation::DeviceBooleanVisibility`).
    boolean: Option<u64>,
    uses: Vec<Use<super::resource::ResourceKey>>,
}

impl Results {
    /// Reset the queries the pass begins, in `command_buffer` before the pass.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass.
    pub(super) unsafe fn reset(&self, raw: &ash::Device, command_buffer: vk::CommandBuffer) {
        for pool in &self.pools.0 {
            // SAFETY: the caller's contract; the pool is live until the command buffer completes.
            unsafe { raw.cmd_reset_query_pool(command_buffer, pool.raw, 0, pool.used) };
        }
    }

    /// Record the copies and the combine in `core`'s command buffer after the pass, between
    /// barriers from everything before and to everything after, and hold their objects until it
    /// completes.
    pub(super) fn record(self, core: &mut EncoderCore) {
        let Results {
            pools,
            buffer,
            landing,
            copies,
            combine,
            boolean,
            uses,
        } = self;
        if let Some(offset) = boolean {
            core.device()
                .degrade(Degradation::DeviceBooleanVisibility, || {
                    let object = format!("the Boolean visibility result at offset {offset}");
                    let reason = "a Boolean result is the device's occlusion query result, which \
                                  Vulkan states only as zero or not without \
                                  VK_QUERY_CONTROL_PRECISE_BIT"
                        .to_owned();
                    (object, reason)
                });
        }
        let mut stages = vk::PipelineStageFlags2::TRANSFER;
        if combine.is_some() {
            stages |= vk::PipelineStageFlags2::COMPUTE_SHADER;
        }
        core.device().shared().counts.pool_compiled(
            crate::work_counts::PoolCompiled::QueryResultCopy,
            copies.len() as u64,
        );
        core.barrier(Barrier::everything());
        core.command(stages, &uses, |raw, command_buffer| {
            // SAFETY: the encoder's command buffer is recording outside a render pass, after the
            // pass that ended every query; the pools, the buffer, the sets and the uploads are
            // live until it completes (retained below, and with the command buffer); each copy
            // lies in the landing region or the continuations' upload region at a multiple of 8,
            // and each onward and inward copy in the landing region and in the buffer (checked
            // when the offset was set).
            unsafe {
                // The words the results accumulate into, into the landing region; the combine
                // reads them after the barrier before it.
                if !landing.inward.is_empty() {
                    raw.cmd_copy_buffer(command_buffer, landing.to, landing.raw, &landing.inward);
                }
                for copy in &copies {
                    raw.cmd_copy_query_pool_results(
                        command_buffer,
                        copy.pool,
                        copy.first,
                        copy.count,
                        copy.target,
                        copy.offset,
                        RESULT_BYTES,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    );
                }
                // The continuations combined into their results.
                if let Some(dispatches) = &combine {
                    Barrier::everything().record(raw, command_buffer);
                    raw.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        dispatches.pipeline,
                    );
                    for (set, &groups) in dispatches.sets.sets.iter().zip(&dispatches.groups) {
                        raw.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            dispatches.layout,
                            0,
                            std::slice::from_ref(set),
                            &[],
                        );
                        raw.cmd_dispatch(command_buffer, groups, 1, 1);
                    }
                }
                if !landing.onward.is_empty() {
                    Barrier::everything().record(raw, command_buffer);
                    raw.cmd_copy_buffer(command_buffer, landing.raw, landing.to, &landing.onward);
                }
            }
        });
        core.barrier(Barrier::everything());
        core.retain(pools);
        if let Some(dispatches) = combine {
            core.retain(dispatches.sets);
        }
        core.retain(buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::{VisibilityResultMode, VisibilityResultType};

    /// Metal's raw values round trip, and an undefined one is refused.
    #[test]
    fn modes_take_metals_raw_values() {
        for mode in [
            VisibilityResultMode::Disabled,
            VisibilityResultMode::Boolean,
            VisibilityResultMode::Counting,
        ] {
            assert_eq!(VisibilityResultMode::from_raw(mode.raw()), Ok(mode));
        }
        assert_eq!(
            [0, 1, 2].map(|raw| VisibilityResultMode::from_raw(raw).map(|m| m.raw())),
            [Ok(0), Ok(1), Ok(2)]
        );
        assert!(VisibilityResultMode::from_raw(3).is_err());
    }

    /// `MTLVisibilityResultType`'s raw values round trip, `Reset` is the default, and an undefined
    /// value is refused.
    #[test]
    fn types_take_metals_raw_values() {
        for kind in [
            VisibilityResultType::Reset,
            VisibilityResultType::Accumulate,
        ] {
            assert_eq!(VisibilityResultType::from_raw(kind.raw()), Ok(kind));
        }
        assert_eq!(
            [0, 1].map(|raw| VisibilityResultType::from_raw(raw).map(|t| t.raw())),
            [Ok(0), Ok(1)]
        );
        assert_eq!(VisibilityResultType::default(), VisibilityResultType::Reset);
        for raw in [-1, 2] {
            assert!(VisibilityResultType::from_raw(raw).is_err(), "{raw}");
        }
    }
}
