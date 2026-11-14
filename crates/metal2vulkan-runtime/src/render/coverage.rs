//! What a reader of its own pass's color covered, learned from a pass that ran: the area rule
//! (`render/snapshot.rs`) weighs the snapshot at the share of the draw's scissors least favourable
//! to it, as a draw's recording cannot know what it covers; once a pass showed what a reader
//! covered, its later passes weigh the snapshot at that.
//!
//! - **Why.** A reader through no scissor of a large target weighs the same at recording whether
//!   it covers 3% of the target (the consumer's readers, `runtime_lane::attachment_read_costs::
//!   self_reads`) or all of it (`snapshot_area_costs`' whole-target rows): the rule takes the
//!   snapshot for both, which costs the second about twice the split on the Radeon 780M.
//! - **Measured.** Each reader the rule weighs (a draw that could read through snapshots, of a
//!   private color, every check before the rule passed: `RenderLowering::snapshot_wanted`) records
//!   its drawing commands inside a precise occlusion query, the next of the pool the device's
//!   readers take queries from, which every pass of every command buffer shares, a spare it reset
//!   once all its queries came back where it has one ([`SparePools`]) (Vulkan's
//!   `VK_QUERY_CONTROL_PRECISE_BIT`: "the query result will be the number of samples that passed"),
//!   whichever form it took: its snapshot draws' fragments are its drawing draws' (the snapshot
//!   module's docs), and the pass has one sample (the rule's checks). As the command buffer
//!   completes, each query's count is kept on the reader's first color's texture (the one a view
//!   was made from), by the reader's identity: its pipeline state and its place among that
//!   state's readers of the texture in its pass ([`ReaderKey`]). A texture keeps [`LEARNED`] at
//!   most, the oldest going first, and each later pass's count replaces its reader's.
//! - **Weighed.** A reader whose identity has a count takes the snapshot where the snapshot draw's
//!   `SNAPSHOT_PIXEL_COST` (`render/snapshot.rs`) at each pixel it covered, for each color, costs
//!   no more than the split's copy ([`snapshot_covers`]), the two forms' fixed costs (about a
//!   mebibyte of the copy each, measured) left out on both sides; a reader with none is weighed by
//!   the area rule. Counted: `WorkCounts::reading_draws_by_coverage`, the queries recorded,
//!   `WorkCounts::coverage_queries`, and the pools made, `WorkCounts::coverage_pools_made`.
//! - **What a stale count costs.** A reader that covers another share than its count holds takes
//!   the form the count chose until a pass after the change completes: one pass where each waits
//!   for the one before, more where passes are committed ahead of the GPU. Both forms are exact:
//!   the count picks between them only.
//! - **Not measured** where the device lacks `occlusionQueryPrecise` (an imprecise query states
//!   only zero or not, which is no coverage) or does not reset queries on the host (an adopted
//!   device, `gpu_time.rs`): its readers are weighed by the area rule alone. Nor while a
//!   visibility query of the program's is active, which the rule's checks already refuse the
//!   snapshot for (one occlusion query is active at a time); nor a reader whose pipeline was still
//!   being made (a later step records its draw), that a replay records or that draws views (a
//!   query in a subpass of views takes one of the pool for each), nor past [`MEASURED`] readers of
//!   a pass. An instance recorded inline after a split keeps the query's begin and end
//!   with its draws (`render/inline_splits.rs`), recorded once with them. A query whose commands
//!   never ran (an instance a reroute replaced, `render/reroute.rs`) reads as unavailable and
//!   teaches nothing. A draw that is no such reader records nothing and makes no pool.
//! - **A driver that counts per color** (`RuntimeDriverDefect::OcclusionCountPerColorOutput`,
//!   ANV): its count is the pixels times the colors the fragment function writes, which weighs
//!   the snapshot higher, toward the split the area rule would have weighed too.

use crate::device::Device;
use crate::error::{vulkan, Error};
use crate::exit_guard::{self, Call};
use crate::texture::Texture;
use ash::vk;
use metal2vulkan::device::VkFeature;
use std::collections::VecDeque;
use std::sync::Mutex;

/// The readers one pass measures, at most, and the queries of a pool ([`SparePools`]).
const MEASURED: u32 = 64;

/// The spare pools a device keeps, at most ([`SparePools`]): one that comes back past them is
/// released.
const SPARE_POOLS: usize = 16;

/// The readers' counts a texture keeps, at most.
pub(crate) const LEARNED: usize = 32;

/// A reader's identity across its passes: its pipeline state's (by address, while the program
/// holds it) and its place among that state's readers of the texture in its pass, from 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReaderKey {
    state: usize,
    ordinal: u32,
}

impl ReaderKey {
    /// The identity of the reader at `ordinal` among the readers of a texture in a pass of the
    /// state at `state` (`GraphicsState::address`): for the crate's device tests, which read its
    /// count (`Texture::learned_coverage`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn of(state: usize, ordinal: u32) -> ReaderKey {
        ReaderKey { state, ordinal }
    }
}

/// The counts a texture learned of its readers (`Texture::learned_coverage`), the latest of
/// each, [`LEARNED`] at most.
#[derive(Debug, Default)]
pub(crate) struct Learned(VecDeque<(ReaderKey, u64)>);

impl Learned {
    /// The pixels `key` covered in the latest pass that counted them.
    pub(crate) fn get(&self, key: ReaderKey) -> Option<u64> {
        self.0
            .iter()
            .find(|(k, _)| *k == key)
            .map(|&(_, covered)| covered)
    }

    /// `key` covered `covered` pixels in a pass that completed: its count from now on, the
    /// oldest count going where the texture keeps [`LEARNED`].
    pub(crate) fn learn(&mut self, key: ReaderKey, covered: u64) {
        match self.0.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => entry.1 = covered,
            None => {
                if self.0.len() == LEARNED {
                    self.0.pop_front();
                }
                self.0.push_back((key, covered));
            }
        }
    }
}

/// Whether a reader that covered `covered` pixels over all the colors it reads through
/// snapshots takes the snapshot: its snapshot draw at `SNAPSHOT_PIXEL_COST` (`render/snapshot.rs`)
/// bytes of the split's copy for each costs no more than the split's copy of `copied` bytes.
pub(crate) fn snapshot_covers(covered: u64, copied: u64) -> bool {
    u128::from(covered) * u128::from(super::snapshot::SNAPSHOT_PIXEL_COST) <= u128::from(copied)
}

/// A reader one pass measures: its query, its first color's texture and its identity.
struct Measured {
    query: Query,
    texture: Texture,
    key: ReaderKey,
}

/// One pass's readers: each one's place among its state's readers of its texture, and, where
/// the device measures them, the readers measured with their queries. Nothing (and no
/// allocation) in a pass without readers: a pass's lowering holds a pointer's width of it.
#[derive(Default)]
pub(super) struct Readers(Option<Box<PassReaders>>);

/// What a pass with readers keeps of them ([`Readers`]).
#[derive(Default)]
struct PassReaders {
    ordinals: Vec<((usize, usize), u32)>,
    /// The device measures readers: `None` until the pass's first reader asks.
    measures: Option<bool>,
    measured: Vec<Measured>,
}

impl Readers {
    /// The identity of the draw of the state at `state` reading `texture` (the one a view was
    /// made from) as the next of that state's readers of it in the pass.
    pub(super) fn next_key(&mut self, state: usize, texture: &Texture) -> ReaderKey {
        let readers = self.0.get_or_insert_with(Box::default);
        let pair = (state, texture.root_address());
        let ordinal = match readers.ordinals.iter_mut().find(|(p, _)| *p == pair) {
            Some((_, next)) => {
                *next += 1;
                *next - 1
            }
            None => {
                readers.ordinals.push((pair, 1));
                0
            }
        };
        ReaderKey { state, ordinal }
    }

    /// The identity [`next_key`](Self::next_key) would give the draw of the state at `state`
    /// reading `texture`, taking none: what a kept reader weighs its form by before it records
    /// (`render/draw_again.rs`).
    pub(super) fn peek_key(&self, state: usize, texture: &Texture) -> ReaderKey {
        let pair = (state, texture.root_address());
        let ordinal = (self.0.as_ref())
            .and_then(|readers| readers.ordinals.iter().find(|(p, _)| *p == pair))
            .map_or(0, |&(_, next)| next);
        ReaderKey { state, ordinal }
    }

    /// Take `key`, which [`peek_key`](Self::peek_key) gave for a reader of `texture` that now
    /// records, as [`next_key`](Self::next_key) takes it.
    pub(super) fn take_key(&mut self, key: ReaderKey, texture: &Texture) {
        let taken = self.next_key(key.state, texture);
        debug_assert_eq!(taken, key, "a reader took another identity than it peeked");
    }

    /// Measure the reader `key` of `texture` where `device` measures readers and the pass
    /// measured fewer than [`MEASURED`]: the query its drawing commands record inside, the next of
    /// the device's pool ([`SparePools::take`]). `None` where it is not measured.
    pub(super) fn measure(
        &mut self,
        device: &Device,
        texture: &Texture,
        key: ReaderKey,
    ) -> Result<Option<(vk::QueryPool, u32)>, Error> {
        let readers = self.0.get_or_insert_with(Box::default);
        let measures = *readers.measures.get_or_insert_with(|| {
            let shared = device.shared();
            shared.timing.resets_on_host()
                && device
                    .properties()
                    .description
                    .has(VkFeature::OcclusionQueryPrecise)
        });
        if !measures || readers.measured.len() == MEASURED as usize {
            return Ok(None);
        }
        let query = device.shared().coverage_pools.take(device)?;
        let at = (query.pool, query.index);
        readers.measured.push(Measured {
            query,
            texture: texture.clone(),
            key,
        });
        Ok(Some(at))
    }

    /// What the pass's command buffer holds until it completes, to read its readers' counts
    /// then: `None` where the pass measured none.
    pub(super) fn readback(self) -> Option<Readback> {
        let measured = self.0?.measured;
        (!measured.is_empty()).then_some(Readback { measured })
    }
}

/// A device's occlusion query pools of [`MEASURED`] queries: the one readers take queries from,
/// in turn, whichever pass and command buffer they are of; those whose every query was taken,
/// until the last comes back; and the spares, [`SPARE_POOLS`] at most, whose every query came
/// back, reset on the host as the next pool is taken from them. So a command buffer of many
/// passes that each measure a reader, and command buffers committed ahead of the GPU, take a
/// pool for every [`MEASURED`] readers rather than one for each pass, and make none once spares
/// have come back: a pool made and released for each pass slows the GPU work running as it is
/// released (on the Iris Xe, in `runtime_lane::attachment_read_costs::snapshot_crossover`, four
/// command buffers committed back to back, readers measured from a pool made for each pass took
/// 1.07 to 1.19 times the GPU time of the same forms unmeasured, and from spare pools 1.00 to
/// 1.06), and costs RADV a buffer object's allocation at each (a window server's walk on the
/// XTX made 222 pools in 14 s, one pass each, under 16 spares). Released as the device ends.
#[derive(Default)]
pub(crate) struct SparePools(Mutex<Pools>);

/// The pools of [`SparePools`].
#[derive(Default)]
struct Pools {
    /// The pool readers take queries from: its next query, and its queries taken that have not
    /// come back.
    current: Option<(vk::QueryPool, u32, u32)>,
    /// Pools whose every query was taken, with their queries that have not come back.
    full: Vec<(vk::QueryPool, u32)>,
    /// Pools whose every query came back.
    spares: Vec<vk::QueryPool>,
}

impl SparePools {
    /// The next query of the device's current pool, unavailable (each pool is reset on the host
    /// as it becomes the current one, on a device that resets queries there: `Readers::measure`):
    /// a spare's or a new pool's where the current one has none left.
    fn take(&self, device: &Device) -> Result<Query, Error> {
        let raw = device.shared().vk();
        let mut pools = crate::device::lock(&self.0);
        let current = match pools.current {
            Some(current) if current.1 < MEASURED => current,
            used => {
                if let Some((pool, _, held)) = used {
                    match held {
                        0 => pools.spare(raw, pool),
                        held => pools.full.push((pool, held)),
                    }
                }
                let pool = match pools.spares.pop() {
                    Some(pool) => pool,
                    None => {
                        let info = vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::OCCLUSION)
                            .query_count(MEASURED);
                        // SAFETY: the device is live and `info` valid.
                        let pool = unsafe { raw.create_query_pool(&info, None) }.map_err(vulkan)?;
                        exit_guard::called(Call::CreateQueryPool);
                        device.shared().counts.coverage_pool_made();
                        pool
                    }
                };
                // SAFETY: the device enables `hostQueryReset`; no pending command uses the pool,
                // nor does any other thread: a spare's every query came back once the command
                // buffer that recorded it completed or went unsubmitted, after its result was
                // read ([`Readback`]), and a new one none used.
                unsafe { raw.reset_query_pool(pool, 0, MEASURED) };
                exit_guard::called(Call::ResetQueryPool);
                (pool, 0, 0)
            }
        };
        let (pool, index, held) = current;
        pools.current = Some((pool, index + 1, held + 1));
        Ok(Query {
            device: device.clone(),
            pool,
            index,
        })
    }

    /// Take back a query of `pool`: the pool a spare once its every query came back and it is not
    /// the current one, or released past [`SPARE_POOLS`].
    fn give(&self, device: &ash::Device, pool: vk::QueryPool) {
        let mut pools = crate::device::lock(&self.0);
        if let Some((current, _, held)) = pools.current.as_mut() {
            if *current == pool {
                *held -= 1;
                return;
            }
        }
        let Some(at) = pools.full.iter().position(|&(full, _)| full == pool) else {
            return;
        };
        pools.full[at].1 -= 1;
        if pools.full[at].1 == 0 {
            pools.full.swap_remove(at);
            pools.spare(device, pool);
        }
    }

    /// Release every pool, as the device ends.
    ///
    /// # Safety
    ///
    /// No pending command uses them: the device's queues are idle.
    pub(crate) unsafe fn release(&self, device: &ash::Device) {
        let mut pools = crate::device::lock(&self.0);
        let mut all: Vec<vk::QueryPool> = pools
            .current
            .take()
            .map(|(pool, ..)| pool)
            .into_iter()
            .collect();
        all.extend(pools.full.drain(..).map(|(pool, _)| pool));
        all.append(&mut pools.spares);
        for pool in all {
            // SAFETY: the caller's.
            unsafe { crate::release::object(device, pool) };
        }
    }
}

impl Pools {
    /// Keep `pool`, whose every query came back, as a spare, or release it past [`SPARE_POOLS`].
    fn spare(&mut self, device: &ash::Device, pool: vk::QueryPool) {
        if self.spares.len() < SPARE_POOLS {
            self.spares.push(pool);
            return;
        }
        // SAFETY: no pending command uses the pool: its every query came back.
        unsafe { crate::release::object(device, pool) };
    }
}

/// A query a reader took of its device's current pool ([`SparePools::take`]), given back as it
/// drops, once its result was read where it ran.
struct Query {
    device: Device,
    pool: vk::QueryPool,
    index: u32,
}

impl Drop for Query {
    fn drop(&mut self) {
        // Once the process is exiting, the pool is left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // No pending command uses the query: its pass's command buffer held it until it
        // completed, or never recorded it.
        let shared = self.device.shared();
        shared.coverage_pools.give(shared.vk(), self.pool);
    }
}

/// A pass's measured readers, held by its command buffer until it completes
/// ([`Readers::readback`]): as it drops, each query that ran gives its reader's count to its
/// texture.
pub(super) struct Readback {
    measured: Vec<Measured>,
}

impl Drop for Readback {
    fn drop(&mut self) {
        if self.measured.is_empty() {
            return;
        }
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // Each reader's query, read before it goes back to its pool (`Query`'s drop, after this).
        for measured in &self.measured {
            let Query {
                device,
                pool,
                index,
            } = &measured.query;
            let mut result = [[0u64; 2]; 1];
            // SAFETY: a query of `pool`, recorded by the pass whose command buffer completed or
            // never ran, which no other thread reads or resets while it is taken: a 64-bit value
            // with its availability, so one that never ran reads as unavailable ("If
            // VK_QUERY_RESULT_WAIT_BIT and VK_QUERY_RESULT_PARTIAL_BIT are both not set then no
            // result values are written to pData for queries that are in the unavailable state
            // ... However, availability state is still written").
            let read = unsafe {
                device.shared().vk().get_query_pool_results(
                    *pool,
                    *index,
                    &mut result,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
                )
            };
            exit_guard::called(Call::GetQueryPoolResults);
            let [covered, available] = result[0];
            if matches!(read, Ok(()) | Err(vk::Result::NOT_READY)) && available != 0 {
                measured.texture.learn_coverage(measured.key, covered);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(ordinal: u32) -> ReaderKey {
        ReaderKey { state: 7, ordinal }
    }

    #[test]
    fn a_later_count_replaces_its_readers_and_the_oldest_goes_first() {
        let mut learned = Learned::default();
        learned.learn(key(0), 10);
        learned.learn(key(0), 20);
        assert_eq!(learned.get(key(0)), Some(20));
        for ordinal in 1..LEARNED as u32 {
            learned.learn(key(ordinal), 1);
        }
        assert_eq!(learned.get(key(0)), Some(20));
        learned.learn(key(LEARNED as u32), 1);
        assert_eq!(learned.get(key(0)), None, "the oldest count went");
        assert_eq!(learned.get(key(1)), Some(1));
    }

    #[test]
    fn a_reader_takes_the_snapshot_where_what_it_covered_costs_no_more_than_the_copy() {
        let cost = super::super::snapshot::SNAPSHOT_PIXEL_COST;
        assert!(snapshot_covers(100, 100 * cost));
        assert!(!snapshot_covers(101, 100 * cost));
        // The consumer's reader of 600x400 over 3662x2039 `BGRA8Unorm`: the snapshot.
        assert!(snapshot_covers(600 * 400, 3662 * 2039 * 4));
        // One covering the whole target: the split.
        assert!(!snapshot_covers(3662 * 2039, 3662 * 2039 * 4));
    }
}
