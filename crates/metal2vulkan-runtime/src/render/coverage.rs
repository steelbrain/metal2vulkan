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
//!   its drawing commands inside a precise occlusion query of a pool its pass holds, one of the
//!   device's spares where it has one ([`SparePools`]) (Vulkan's
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

/// The readers one pass measures, at most: its pool's queries.
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
    query: u32,
    texture: Texture,
    key: ReaderKey,
}

/// One pass's readers: each one's place among its state's readers of its texture, and, where
/// the device measures them, the pool its first measured reader took and the readers measured.
/// Nothing (and no allocation) in a pass without readers: a pass's lowering holds a pointer's
/// width of it.
#[derive(Default)]
pub(super) struct Readers(Option<Box<PassReaders>>);

/// What a pass with readers keeps of them ([`Readers`]).
#[derive(Default)]
struct PassReaders {
    ordinals: Vec<((usize, usize), u32)>,
    /// The device measures readers: `None` until the pass's first reader asks.
    measures: Option<bool>,
    pool: Option<Pool>,
    used: u32,
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

    /// Measure the reader `key` of `texture` where `device` measures readers and the pass has a
    /// query left: the query its drawing commands record inside, from the pass's pool, taken at
    /// its first ([`SparePools::take`]). `None` where it is not measured.
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
        if !measures || readers.used == MEASURED {
            return Ok(None);
        }
        let pool = match &readers.pool {
            Some(pool) => pool.raw,
            None => readers.pool.insert(Pool::new(device)?).raw,
        };
        let query = readers.used;
        readers.used += 1;
        readers.measured.push(Measured {
            query,
            texture: texture.clone(),
            key,
        });
        Ok(Some((pool, query)))
    }

    /// What the pass's command buffer holds until it completes, to read its readers' counts
    /// then: `None` where the pass measured none.
    pub(super) fn readback(self) -> Option<Readback> {
        let readers = *self.0?;
        Some(Readback {
            pool: readers.pool?,
            measured: readers.measured,
        })
    }
}

/// A device's occlusion query pools of [`MEASURED`] queries that no pass holds, [`SPARE_POOLS`]
/// at most, which the next pass that measures a reader takes before making one: a pool made
/// and released for each pass slows the GPU work running as it is released. On the Iris Xe, in
/// `runtime_lane::attachment_read_costs::snapshot_crossover` (four command buffers committed back
/// to back), readers measured from a pool made for each pass took 1.07 to 1.19 times the GPU
/// time of the same forms unmeasured, and from spare pools 1.00 to 1.06. Released as the device
/// ends.
#[derive(Default)]
pub(crate) struct SparePools(Mutex<Vec<vk::QueryPool>>);

impl SparePools {
    /// A pool of [`MEASURED`] occlusion queries, each unavailable (reset on the host, on a device
    /// that resets queries there: `Readers::measure`): a spare where the device has one, else a
    /// new one.
    fn take(&self, device: &Device) -> Result<vk::QueryPool, Error> {
        let raw = device.shared().vk();
        let spare = crate::device::lock(&self.0).pop();
        let pool = match spare {
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
        // SAFETY: the device enables `hostQueryReset`; no pending command uses the pool: a spare
        // came back once the command buffer of the pass that held it completed or went
        // unsubmitted ([`Readback`]), and a new one none used; no other thread holds it.
        unsafe { raw.reset_query_pool(pool, 0, MEASURED) };
        exit_guard::called(Call::ResetQueryPool);
        Ok(pool)
    }

    /// Take back `pool`, which no pending command uses: a spare, or released past
    /// [`SPARE_POOLS`].
    fn give(&self, device: &ash::Device, pool: vk::QueryPool) {
        let mut spares = crate::device::lock(&self.0);
        if spares.len() < SPARE_POOLS {
            spares.push(pool);
            return;
        }
        drop(spares);
        // SAFETY: the caller's: no pending command uses the pool.
        unsafe { crate::release::object(device, pool) };
    }

    /// Release every spare, as the device ends.
    ///
    /// # Safety
    ///
    /// No pending command uses them: the device's queues are idle.
    pub(crate) unsafe fn release(&self, device: &ash::Device) {
        for pool in crate::device::lock(&self.0).drain(..) {
            // SAFETY: the caller's.
            unsafe { crate::release::object(device, pool) };
        }
    }
}

/// A pass's occlusion query pool, given back to its device's spares as it drops.
struct Pool {
    device: Device,
    raw: vk::QueryPool,
}

impl Pool {
    /// A pool of [`MEASURED`] occlusion queries, each unavailable, on a device that resets
    /// queries on the host ([`SparePools::take`]).
    fn new(device: &Device) -> Result<Pool, Error> {
        let raw = device.shared().coverage_pools.take(device)?;
        Ok(Pool {
            device: device.clone(),
            raw,
        })
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        // Once the process is exiting, the pool is left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // No pending command uses the pool: its pass's command buffer held it until it
        // completed, or never recorded it.
        let shared = self.device.shared();
        shared.coverage_pools.give(shared.vk(), self.raw);
    }
}

/// A pass's measured readers, held by its command buffer until it completes
/// ([`Readers::readback`]): as it drops, each query that ran gives its reader's count to its
/// texture.
pub(super) struct Readback {
    pool: Pool,
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
        // The measured readers took the pool's first queries, in turn.
        let mut results = [[0u64; 2]; MEASURED as usize];
        let results = &mut results[..self.measured.len()];
        // SAFETY: the pool's queries, each recorded by the pass whose command buffer completed or
        // never ran: 64-bit values, each with its availability, so one that never ran reads as
        // unavailable ("If VK_QUERY_RESULT_WAIT_BIT and VK_QUERY_RESULT_PARTIAL_BIT are both not
        // set then no result values are written to pData for queries that are in the unavailable
        // state ... However, availability state is still written").
        let read = unsafe {
            self.pool.device.shared().vk().get_query_pool_results(
                self.pool.raw,
                0,
                results,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
            )
        };
        exit_guard::called(Call::GetQueryPoolResults);
        if !matches!(read, Ok(()) | Err(vk::Result::NOT_READY)) {
            return;
        }
        for measured in &self.measured {
            let [covered, available] = results[measured.query as usize];
            if available != 0 {
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
