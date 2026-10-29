//! The cost half: what **our** side did to produce a case's transcript.
//!
//! The suite's correctness half records what a case *observed*, field by field, against Metal.
//! This is the other half: for the same case set, the operations our own implementation performed
//! to answer it. The two never mix.
//!
//! # Counters are ours, not Metal's
//!
//! Metal has no counterpart for any of these counts, and a host with no Metal can have no answer
//! for them, so **a count never enters the comparison with Metal and never produces a correctness
//! verdict**. The cost half is a gate of its own, with its own three verdicts
//! ([`CostVerdict`]): `Within`, `Over`, `Unmeasured`.
//!
//! # Why counts and not times
//!
//! A count is deterministic, so it is assertable on any host under any load; a time is not. A
//! regression then reads "this case now does two allocations where it did none", which is a stable
//! gate and a better bug report than a percentage nobody can reproduce. Where a count was measured
//! here *not* to be deterministic, the record says so and names the range it was seen in
//! ([`CostBound::range`]), and that count is reported rather than gated — a bound that fails at
//! random is worse than no bound.
//!
//! # The mechanism is the runtime's own
//!
//! [`metal2vulkan_runtime::WorkCounts`] already counts what the runtime did; nothing here invents a
//! parallel mechanism. It is read twice around each case, and the difference is that case's cost.
//! The read is cheap and always on, because product behaviour may not branch on the environment:
//! the suite never asks the runtime to count *more* for a cost run than for an ordinary one.
//!
//! # Attribution
//!
//! Three regions are kept apart ([`Region`]), because a change that moves work from one to another
//! is not a win:
//!
//! - **translation and pipeline creation**, the one-time cost that already has its own ceilings
//!   (20 seconds and 500 MiB per translation);
//! - **the device API**, what a call does on the calling thread — the gap the plan names;
//! - **submission**, what the commit path, our own other threads and the GPU do afterwards. A call
//!   that returns at once while queueing work for a later commit must not score as a win, so the
//!   work it queued is counted here rather than vanishing.
//!
//! # Where a bound lives
//!
//! Beside the case's transcript, in the case's own recording file ([`crate::recording::Recording`]),
//! so a host with no Metal replays it unchanged. A record names the venue it was measured on and
//! the measurement that justified it; a venue the record does not hold is `Unmeasured`, never a
//! pass.

use metal2vulkan_runtime::WorkCounts;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A field of [`WorkCounts`] the cost half does not read, and why.
///
/// A `*_sites` field is a table of the runtime's own source positions — a file, a line and what
/// was recorded there — rather than a count, so the difference of two readings of it is not a
/// case's cost. What those tables name is already counted elsewhere: `copy_sites` says *where* the
/// copies were recorded, and the copies themselves are `commands_recorded.cmd_copy_*`. The rule is
/// a suffix rather than a list of names because the runtime keeps growing such tables.
const NOT_A_COUNT: [(&str, &str); 1] = [(
    "_sites",
    "a table of the runtime's source positions and what each recorded, not a count; what those \
     places did is counted in the counts beside it",
)];

/// A suffix a field name carries when it holds a high-water mark or an accumulated time rather
/// than a count, with why the cost half does not read it.
///
/// A maximum does not subtract: the difference of two high-water marks is not the work one case
/// did. A time is not a fact a gate can hold, which is the whole reason this half gates on counts.
const NOT_SUBTRACTABLE: [(&str, &str); 3] = [
    ("_most", "a high-water mark, which does not subtract"),
    ("_peak", "a high-water mark, which does not subtract"),
    ("_nanos", "an accumulated time, which no count gate holds"),
];

/// Whether `name` is a field the cost half reads, or why it is not.
pub fn excluded(name: &str) -> Option<&'static str> {
    let head = name.split('.').next().unwrap_or(name);
    for (suffix, why) in NOT_A_COUNT {
        if head.ends_with(suffix) {
            return Some(why);
        }
    }
    let leaf = name.rsplit('.').next().unwrap_or(name);
    for (suffix, why) in NOT_SUBTRACTABLE {
        if leaf.ends_with(suffix) {
            return Some(why);
        }
    }
    None
}

/// Why a count of what is alive, parked or waiting **at the instant the counts were read** is
/// reported and never gated.
///
/// It is not work a case did; it is a reading of process-wide state that happens to be taken at
/// the case's boundary. Measured on this M3 Max: two replays of the whole set on the same binary
/// over the same 350 cases put `empty_memory_blocks_live` over its bound on
/// `xr-two-planes-rgb10a8-2p-xr10-srgb` in one run and on `depth16-texel-reads`,
/// `texture-fills-53` and `sample-fills-d32-2` in the other, and `memory_blocks_live` over it in
/// the second run only. Run alone, the same three texture-buffer cases read
/// `empty_memory_blocks_live` 2 and `memory_blocks_live` 1 where inside the set they read 0 or 1.
/// A count that lands on a different case every run cannot be bounded on any of them.
const INSTANTANEOUS: &str = "a reading of what was alive, parked or waiting at the instant the \
     counts were read, which is process-wide state and not work the case did: measured landing on \
     a different case in each of two replays of the same binary over the same cases";

/// Why a count of what an object's **release** got round to is reported and never gated: the same
/// reason as the `object_churn.` tree, for the counts that live outside it.
///
/// Measured: `sampler_sets_destroyed` reached 1 where the measurement's most was 0 on
/// `xr-two-planes-rgb10a8-2p-xr10-srgb` in two replays of the whole set, and reads 0 when that
/// case runs alone.
const RELEASED: &str = "what a release got round to destroying, which runs on whichever thread \
     drops the object and may land in a later case's window";

/// Why a count of an object **found already made** rather than made is reported and never gated.
///
/// Whether this case makes a pipeline layout or finds one depends on what every case before it in
/// the same process made, so the split between the two is a fact about the selection and not about
/// the case. Its partner stays gated and loses nothing: finding one only ever **lowers** the
/// `_made` count, and the gate is an upper bound.
///
/// Measured, in the one direction that matters. `depth16-texel-reads` run alone makes 2 pipeline
/// layouts and shares none; inside the 350-case set it makes 1 and shares 1. Two and two.
/// `xr-two-planes-rgb10a8-2p-xr10-srgb` run alone makes 8 pipeline variants and takes none back;
/// inside the set it makes 4 and takes 4 back — and `variants_taken_back` is the first count of
/// this family measured to go **up** with more cases around it, from 0 to 4, which is the direction
/// an upper bound cannot survive. `draw_shapes_reused` is the second, and the loudest:
/// `integer-vertex-attributes` reads 0 alone and reached 19 in one replay of the whole set and 11
/// in the next, because a draw takes none of a shape whose derivation announces a notice the device
/// announces at every occurrence (`DeviceOptions::announce_each`, which this suite sets over every
/// kind), so what it reuses depends on the notice state every case before it left behind.
///
/// The `_made` partners stay gated, and a measurement over all 393 cases in both shapes says they
/// can: `addressable_buffers_made`, `pooled_memory_blocks_made`, `arena_pools_made`,
/// `device_memory_allocated`, `creation_pipelines_made`, `variant_pipelines_made`,
/// `shader_modules_made` and `pipeline_layouts_made` read at or above their in-set reading in every
/// case that ran alone, so company only lowers them.
///
/// That is a fact about those eight counts and not a law about `_made` counts, and the contrast the
/// measurement now takes is what tells one from the other. Two counts an earlier version of this
/// doc claimed alongside them — `command_buffers_begun` and `command_buffers_submitted` — move
/// **both** ways: alone above their in-set reading in 104 cases and below it in 6, because
/// `xr-clears-bgr10xr` and its siblings begin a command buffer in 64 segments inside the set and in
/// 1 alone. A count like that is bounded from neither shape, which is why [`CostRecord::composed`]
/// reports it instead of either shape gating it.
const FOUND: &str = "an object found already made rather than made, which depends on what every \
     case before it in the same process made. Its `_made` partner stays gated, and loses nothing: \
     finding one only lowers that count, and the gate is an upper bound";

/// Counts of work the runtime **avoided**, where a reading below the measurement's is the
/// regression and a reading above it is a win.
///
/// Every other count here is gated from above: a case that does more work than the measurement saw
/// is [`CostVerdict::Over`]. A count of work avoided inverts that. `creation_pipelines_drawn`
/// counts the predicted `VkPipeline`s a command actually took, so a change that makes the
/// prediction miss **lowers** it and raises `variant_pipelines_made` — and an upper bound passes
/// both, because one went down and the other has room. The per-case recordings say how much room:
/// over the 1103 committed recordings, 508 of the 745 pipelines made at a state's creation were
/// never taken by any command, and 83 cases hold a deterministic `creation_pipelines_drawn` above
/// zero that nothing today holds them to.
///
/// A name here is gated from below at the bound's `most`, and only where the bound is gated at
/// all: a count the measurement saw move between runs ([`CostBound::range`]), or that reads
/// differently with the case alone ([`CostBound::composed`]), or that [`racy`] names, stays
/// reported in both directions. For a gated count `most` is every run's reading, so the floor and
/// the ceiling are the same number and the gate says a case kept avoiding exactly what it avoided.
///
/// **Each name is here on its own measurement**, as in every other table in this file, and never
/// by analogy to another. `pipeline_layouts_shared`, `shader_modules_shared`, `draw_shapes_reused`
/// and `variants_taken_back` all count work avoided too, and none of them is here: the
/// measurements quoted under [`FOUND`] show each of them moving with what ran before the case, so
/// a floor on one would fail the moment a case lands beside it.
const SPARED: &[(&str, &str)] = &[(
    "creation_pipelines_drawn",
    "the predicted pipelines a command took, so a prediction that stops being taken lowers it \
     while `variant_pipelines_made` rises into the room an upper bound leaves",
)];

/// The name the gate's **derived** row carries, which is not a field of
/// [`metal2vulkan_runtime::WorkCounts`] and never comes out of [`Counts`].
///
/// A breach of the pipelines-per-state ratio is reported as an [`Overrun`] like any other, because
/// that is what it is — this case made more pipelines than its bound allows — and a reader should
/// not have to learn a second shape to read it. It is named apart so nobody greps the runtime for
/// it: the two counts it is derived from are `creation_pipelines_made` and
/// `variant_pipelines_made`, and the denominator is the suite's own ([`crate::states`]).
pub const PER_STATE: &str = "pipelines_per_pipeline_state";

/// The `VkPipeline`s our leg made over the Metal pipeline states the case created: **the ratio**,
/// kept as the two counts it is the quotient of so the gate never compares two floats.
///
/// # Why a ratio is gated where two counts already are
///
/// `variant_pipelines_made` is gated from above and `creation_pipelines_drawn` from below
/// (the `SPARED` table), and **both pass a change that makes one more pipeline per state while the
/// case
/// grows a state**: the absolute counts rise for two reasons at once and an absolute bound cannot
/// tell them apart. A re-bless then writes the higher number down, which is how the doubled
/// predicted pipeline on `integer-vertex-attributes` came to sit in a committed recording as the
/// blessed ceiling — 64 states, 64 predicted, 128 made — for three passes.
///
/// The ratio separates the two. A case that grows a state grows both halves and reads the same
/// ratio; a change that makes more pipelines per state raises it. Measured on this M3 Max over all
/// 1,103 cases, 8 runs of the whole selection, in the 639 cases that create a state and carry a
/// cost record: 1,755 states, 2,665 `VkPipeline`s, **1.519 each**, the worst row
/// `swizzled-storage-rgba8unorm` at 11.25. The committed bounds' own census of the creation-time
/// prediction, over the 585 records that make one at all, is 1,439 made and 815 taken by a command
/// (`the_creation_prediction`, beside this module).
///
/// # Why it needs no timing window
///
/// It is a count, like everything else this half gates, so it is judgeable on a host running nine
/// other builds. The same cost read as time is 3.62x worst per case, under the times half's
/// per-case margin of 10x, and spread across a call the per-call gate reads at 31x while that
/// call's own cases run faster than Metal. Neither half of the times instrument can see it. This
/// is the form that can.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelinesPerState {
    /// The `VkPipeline`s the measurement saw made: `creation_pipelines_made` plus
    /// `variant_pipelines_made`.
    pub pipelines: u64,
    /// The Metal pipeline states the case created in the same run ([`crate::states`]). Never zero
    /// in a record: a case that creates none has no ratio and carries no bound.
    pub states: u64,
}

impl PipelinesPerState {
    /// The ratio this measurement read, where it read one.
    pub fn ratio(&self) -> Option<f64> {
        (self.states > 0).then(|| self.pipelines as f64 / self.states as f64)
    }

    /// The most pipelines this bound allows a run that created `states` states.
    ///
    /// Floored, and that is exact rather than lenient: for an integer count, `made > p * s / b` and
    /// `made > floor(p * s / b)` are the same condition, so the number printed in a failure is the
    /// number the gate judged by. In `u128`, because a case has reached 128 pipelines and a
    /// cross-multiplication of two such counts has no business being the thing that overflows.
    pub fn allows(&self, states: u64) -> u64 {
        if self.states == 0 {
            return u64::MAX;
        }
        let allowed = u128::from(self.pipelines) * u128::from(states) / u128::from(self.states);
        u64::try_from(allowed).unwrap_or(u64::MAX)
    }

    /// Whether `made` pipelines over `states` states is more per state than this bound.
    pub fn exceeded_by(&self, made: u64, states: u64) -> bool {
        self.states > 0 && made > self.allows(states)
    }
}

impl std::fmt::Display for PipelinesPerState {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.ratio() {
            Some(ratio) => write!(
                out,
                "{} pipeline(s) over {} state(s), {ratio:.3} each",
                self.pipelines, self.states
            ),
            None => write!(out, "{} pipeline(s) over no state", self.pipelines),
        }
    }
}

/// The two counts the ratio's numerator adds: every `VkPipeline` our leg made for a case.
///
/// Named here rather than summed at each reader, because a third counter of made pipelines would
/// otherwise have to be remembered in several places. `pipelines_linked` is deliberately not one of
/// them: the runtime's own doc says a linked render variant is "each counted in
/// `variant_pipelines_made` too", so adding it would count those pipelines twice.
pub const PIPELINES_MADE: [&str; 2] = ["creation_pipelines_made", "variant_pipelines_made"];

/// The `VkPipeline`s `counts` says were made.
pub fn pipelines_made(counts: &Counts) -> u64 {
    PIPELINES_MADE.iter().map(|count| counts.get(count)).sum()
}

/// Every count gated from below, with its reason (this module's own `SPARED` table), so a test
/// can hold the table in
/// both directions.
pub fn spared_table() -> impl Iterator<Item = (&'static str, &'static str)> {
    SPARED.iter().copied()
}

/// Whether `count` measures work the runtime avoided, and why it is gated from below (the
/// `SPARED` table).
pub fn spared(count: &str) -> Option<&'static str> {
    SPARED
        .iter()
        .find(|(name, _)| *name == count)
        .map(|(_, why)| *why)
}

/// Why `completed_waits` is reported and never gated.
///
/// It looked like the one deterministic breach — exactly one over its bound, in the same three
/// cases, in two replays of the whole set. The contrast run says otherwise: run **alone**, those
/// cases wait 25, 225 and 5 times; inside the 350-case set they wait 30, 230 and 10, and inside
/// the 126-case set the bounds were taken from, 29, 229 and 9. So the case's own waits are the
/// smaller number and the rest are waits of whatever ran before it, landing in this case's window
/// because the completion waiter is a thread of ours. How many it borrows moves with the
/// selection, which is why a bound on it is a bound on the selection.
const BORROWED: &str = "waits a case is charged for that belong to whatever ran before it: the \
     completion waiter is a thread of ours, so a wait lands in whichever case's window it reaches. \
     Measured: 25, 225 and 5 waits when three cases run alone against 30, 230 and 10 inside the \
     whole set";

/// Why `pipeline_pool_busy` is reported and never gated, which the runtime's own doc comment says
/// before any measurement does: it is "not a count of events, but the pool's state then".
///
/// It is the `*_live` family wearing a name the suffix does not catch. Subtracting two readings of
/// how many pieces the device's pipeline pool holds gives neither a count of work nor a difference
/// in state: the pool's threads are ours, and whether one is still running when the counts are
/// read is a question about this instant. Measured: it reached 1 where the measurement's most was
/// 0 on `xr-blend-bgra10xr-one-zero` in one whole-set replay and read 0 in the next.
const POOL_STATE: &str = "a reading of how many pieces the device's pipeline pool holds at the \
     instant the counts are read, which the runtime documents as the pool's state and not a count \
     of events";

/// Why `encoder_primaries_begun_ahead` is reported and never gated.
///
/// The runtime documents it as a `VkCommandBuffer` begun "by the thread that completed the command
/// buffer that gave it back where no thread waited for that one", so whether this case's first
/// encoder finds a primary already begun depends on whether a completion thread got round to
/// beginning it, for a pool an earlier case gave back. It is also a *split* of
/// `encoder_command_buffers_begun`, which stays gated, so no work escapes the gate by it.
///
/// Measured: `chroma-pair-copies-gbgr422` and `chroma-pair-texel-samples-gbgr422` read 2 and 1
/// inside the 393-case set where the measurement's most was 0, and **0** when each runs alone --
/// the opposite direction to a pooled making, and the reason a bound measured alone would fail
/// them for ever.
///
/// `primaries_begun_ahead` is the same work counted a step earlier, and the runtime's doc gives it
/// a second race of its own on top: it counts a primary "taken by an encoder or not yet", "once its
/// pool is among the spares again". So it moves when a pool comes back, which is a thread of ours
/// finishing something an earlier case submitted. Measured over the 541-case set: it reached 1
/// where the measurement's most was 0 in **seven** cases of one whole-set replay —
/// `border-swizzles-bgra8unorm`, `pixel-downscales-1536x1080`, `two-plane-row-blits`,
/// `texture-fills-30`, `texture-fills-134`, `texture-fills-236` and `shared-event-rising-value` —
/// and in **none** of the replay before it.
const AHEAD: &str = "whether this case's first encoder found a primary command buffer already \
     begun, which the thread that completed an earlier command buffer does ahead when nothing \
     waits for it. Its total, encoder_command_buffers_begun, stays gated, so nothing is lost";

/// Why `completed_where_seen` is reported and never gated: it is the **complement of a count the
/// rule already names**, so it cannot be deterministic while that one is not.
///
/// The runtime documents it as the command buffers whose transition to `Completed` "ran on the queue
/// waiter that saw their work complete, **with no handler thread woken**". The other outcome of that
/// same transition is a handler thread being woken, which `wakes_handler_threads` counts and the
/// rule reports. A count that is one side of a coin whose other side is a thread-arrival count is
/// the same question asked the other way round.
///
/// Measured: it reached 2 where the measurement's most was 1 on `event-held-command-buffer` in one
/// whole-set replay and stayed within bound in the replay before it.
///
/// `scheduled_where_submitted` is the same shape one transition earlier — "ran on the scheduler's
/// thread as their last batch went, with no handler thread woken" — and is **deliberately left
/// gated**, because nothing has yet been seen to move it and a row here would be a claim with no
/// measurement behind it. It is the next false alarm this half is likely to raise; when it raises
/// one, it belongs here with that reading, and neither it nor `completed_where_seen` has a gated
/// total covering it, so reporting them is a real loss of reach rather than a free move.
const COMPLEMENT: &str = "which of our own threads ran a command buffer's transition: this counts \
     the ones no handler thread was woken for, and waking one is itself a thread-arrival count the \
     rule reports";

/// Why `pooled_memory_blocks_freed` is reported and never gated: the runtime frees such a block
/// when it has been "kept empty **past their quiet period**", so whether one is freed inside a
/// case's window is a question about elapsed time.
///
/// Its partner `pooled_memory_blocks_made` stays gated, at the reading of the lone run
/// (the `MONOTONE` table), so what a case *allocates* is still held; only what the pool got round to
/// handing back is not.
///
/// Measured: it reached 1 where the measurement's most was 0 on `sample-positions-custom-2x` in two
/// whole-set replays, and both measurement shapes read 0 — a release that lands in the replay's
/// timing and not in the measurement's is exactly what a quiet period does.
const QUIET: &str = "a pooled memory block freed because it had been kept empty past its quiet \
     period, which is a question about elapsed time rather than about the case's work";

/// Why `finales_folded` is reported and never gated: it is one half of a split the **submission
/// path** decides as it runs, not a count of what the case asked for.
///
/// A batch's end is either folded into its last command buffer or submitted in one of its own, and
/// the runtime's doc for the other half, `finales_apart`, lists what forces it apart: an event
/// signal that waits for another `VkQueue`'s earlier one, a submission left a segment or a sparse
/// update, a texture that moves between layouts, or a last command buffer that "does not go to the
/// queue as recorded". Several of those are about what the queue did with work submitted **before
/// this case**.
///
/// Measured, in both ways it moves. Across the case set it reads 3 with the case alone and 4 in
/// company in 76 cases. And it moves between replays of the *identical* selection, which
/// composition alone cannot explain: it reached 2 where the measurement's most was 0 on
/// `tile-order-even-threadgroups` in one whole-set replay and stayed within bound in the other of
/// the same pair.
///
/// `finales_apart` is its partner and stays **gated**, for the same reason
/// `scheduled_where_submitted` does: nothing has been seen to move it, and a row here with no
/// measurement behind it is a claim this half does not make. So work moving from folded to apart
/// still fires the gate, and work moving the other way fires nothing — the asymmetry is worth
/// knowing rather than papering over.
const FOLDED: &str =
    "whether a batch's end was folded into its last command buffer or sent in one \
     of its own, which the submission path decides from what the queue did with work submitted \
     before this case";

/// Why a count of an object made because no spare was free is reported and never gated.
const POOLED: &str = "an object made because no spare was free, where a spare comes back on \
     another thread: which case has to make one depends on when that thread returned the one \
     before it, not on the case's own work. Measured: `fences_made` reached 1 where the \
     measurement's most was 0 on three whole-set replays in a row, in a different case each time";

/// A count whose value says **which thread arrived first**, with why it is reported and never
/// gated, whatever a measurement happened to see.
///
/// The cost half gates on counts because a count is deterministic. These are not, structurally,
/// and no number of measurements makes them so: holding them to a bound gives a gate that fires at
/// random on honest work. Every entry's reason is the runtime's own words about itself, and
/// every one of them was then measured here.
///
/// Measured on this M3 Max over seven measurements of the whole case set (3 to 10 rounds of 126
/// cases, about 6,300 counts a round): 24, 27, 35, 38, 47, 49 and 68 readings moved, and every one
/// of them was one of these families. Replays of the whole set then went over a bound six times,
/// again every time one of them, which is what sent the rule here instead of into a longer
/// measurement.
///
/// `fences_made` is the one that had to be read twice. Each measurement saw it move by a single
/// reading in a single case, which looked like too little to be a family — until three replays in
/// a row went over its bound in a **different case each time**. A measurement's range cannot cover
/// that, because the case the race lands in is not fixed, and the mechanism is the same as the
/// others': a fence is made where no spare one is free, and a spare comes back on the completion
/// thread.
///
/// `completed_waits` is the one that had to be read a third time, and it is the sharpest warning
/// against trusting a measurement over an argument: it held still over every round of every
/// measurement here, then came in at **exactly one over** its bound in the same three cases in two
/// replays of a larger case set — which reads like a regression and is not one. See [`BORROWED`].
/// `slot_returns` beside it stays gated: it counts work items, one a command buffer, and has held
/// still alone and in both case sets.
const RACY: [(&str, &str); 32] = [
    ("fences_made", POOLED),
    // What is alive, parked or waiting at the instant the counts were read.
    ("memory_blocks_live", INSTANTANEOUS),
    ("empty_memory_blocks_live", INSTANTANEOUS),
    ("dedicated_memory_blocks_live", INSTANTANEOUS),
    ("host_imports_live", INSTANTANEOUS),
    ("dma_buf_memories_live", INSTANTANEOUS),
    ("dma_buf_image_memories_live", INSTANTANEOUS),
    ("shadow_merge_kept_images_live", INSTANTANEOUS),
    ("freed_memory_parked", INSTANTANEOUS),
    ("freed_memory_waiting", INSTANTANEOUS),
    ("host_import_releases_pending", INSTANTANEOUS),
    ("host_imports_resident", INSTANTANEOUS),
    // A release the owning thread got round to, outside the `object_churn.` tree.
    ("sampler_sets_destroyed", RELEASED),
    // Found already made rather than made. Named one by one rather than by a suffix: `_shared` is
    // also carried by `set_zero_writes_if_shapes_shared`, which is a per-draw count of the case's
    // own work, and `_reused` by counts of what a case's own encoder settled once.
    ("pipeline_layouts_shared", FOUND),
    ("shader_modules_shared", FOUND),
    // An ask for a stage's function answered by the device's memo, a binary archive or the shader
    // cache on disk. The memo and the cache are the device's and the run's, so whether this case's
    // *first* ask for a function finds it already translated depends on whether a case before it
    // in the same run asked under the same compile key -- and in company that turns a translation
    // into a sharing, which moves this count **up**. Its partner `functions_translated` stays
    // gated for the same reason the other `_made` partners do: company only ever lowers it.
    ("functions_shared", FOUND),
    ("sampler_sets_shared", FOUND),
    ("variants_taken_back", FOUND),
    ("draw_shapes_reused", FOUND),
    ("completed_waits", BORROWED),
    ("pipeline_pool_busy", POOL_STATE),
    ("encoder_primaries_begun_ahead", AHEAD),
    ("primaries_begun_ahead", AHEAD),
    ("completed_where_seen", COMPLEMENT),
    ("pooled_memory_blocks_freed", QUIET),
    ("finales_folded", FOLDED),
    ("command_pools_made", POOLED),
    ("spare_command_pools_", POOLED),
    ("live_tables_uploaded", POOLED),
    (
        "scheduler_",
        "how many times one of our own background threads woke, slept or made a pass, which is \
         when that thread ran and not what the case asked for",
    ),
    (
        "wakes_",
        "whether a transition had to wake a thread or found one already running, which is a \
         question about timing rather than about the case's work",
    ),
    (
        "object_churn.",
        "the release path, which runs on whichever thread drops the object. A release may land in \
         a later case's window, and whether a making takes a parked object or makes a new one \
         depends on whether the release before it has landed",
    ),
];

/// The counts whose larger reading is the **no-sharing extreme**, so the gate keeps them at that
/// extreme instead of reporting them.
///
/// A count here is one the two shapes read differently for one reason only: a case with nothing
/// before it makes the object itself where a case in company finds it already made. Its reading
/// alone is therefore its ceiling, no composition can exceed it, and a bound taken there does not
/// go stale when a case lands beside this one. Gating it is the one place the contrast gives back
/// what it takes away.
///
/// **Each name is here on its own measurement and never by analogy to another.** Over all 393
/// cases in both shapes, every one of these read at or above its in-set reading in every case that
/// moved it — `below` counts the cases where the lone run read *lower*, which is the direction that
/// would make the bound wrong:
///
/// | count | cases | below | the shapes differ | largest alone |
/// |---|---:|---:|---:|---:|
/// | `addressable_buffers_made` | 385 | 0 | 384 | 2 |
/// | `pooled_memory_blocks_made` | 385 | 0 | 384 | 5 |
/// | `arena_pools_made` | 215 | 0 | 207 | 30 |
/// | `creation_pipelines_made` | 181 | 0 | 109 | 64 |
/// | `variant_pipelines_made` | 196 | 0 | 114 | 128 |
/// | `shader_modules_made` | 199 | 0 | 160 | 34 |
/// | `pipeline_layouts_made` | 199 | 0 | 174 | 4 |
///
/// **`device_memory_allocated` is monotone over the same 393 cases and is deliberately not here.**
/// It passed the direction test — 385 cases, 0 below — and fails the other test a bound has to
/// pass: it counts *bytes*, so the two shapes differ by 268,435,456 where the largest of these
/// object counts differs by 8. A bound of 256 MiB on a case that allocates nothing in company is a
/// number no regression can reach, and a bound that cannot fail is worse than no bound. It stays
/// reported, with both readings, where a reader can see it.
///
/// **A count of bytes can never take a row here**, which is the general form of the paragraph
/// above and is held by `tests::no_monotone_bound_is_a_count_of_bytes` rather than left to whoever
/// reads it. The gap between the two shapes scales with the allocation and the bound is the larger
/// shape's, so for bytes the bound lands orders of magnitude above anything a regression reaches
/// while looking exactly like a bound that bites.
///
/// A ninth name earns a row here on data: a measurement over the whole case set in both shapes
/// showing no case where the lone run reads lower, and a gap small enough that the bound still
/// bites. Never by resembling one of these.
///
/// **And a count of this shape needs no row to be handled correctly meanwhile**, which is worth
/// saying because the shape keeps arriving. The runtime builds some compute kernels of its own
/// lazily, inside whichever Metal call first needs one (`owned_kernel.rs`), so a case run alone
/// builds one and reads non-zero where the same case run after another that needed it reads zero.
/// That is this family exactly. The contrast already answers it: a count the two shapes read
/// differently lands in [`CostRecord::composed`] and is reported rather than gated, with both
/// readings where a reader can see them, and every committed record now carries a contrast
/// (`tests::the_contrast` holds that). A `MONOTONE` row is the **opt-in** that keeps such a count
/// *gated*, at the
/// lone run's ceiling, and it is the thing that needs the measurement. So the order is: let the
/// contrast report it, then earn the row. `owned_kernels_made` is the next candidate on that path
/// and `owned_kernel_spirv_bytes` can never be one, by the bytes rule above.
const MONOTONE: [(&str, &str); 7] = [
    ("addressable_buffers_made", "2 alone against 0 in company"),
    ("pooled_memory_blocks_made", "4 alone against 0 in company"),
    ("arena_pools_made", "30 alone against 22 in company"),
    ("creation_pipelines_made", "4 alone against 0 in company"),
    ("variant_pipelines_made", "6 alone against 0 in company"),
    ("shader_modules_made", "7 alone against 0 in company"),
    ("pipeline_layouts_made", "4 alone against 0 in company"),
];

/// Where `name`'s reading with the case alone is its ceiling, and the worst gap the measurement saw.
///
/// The names and the measurement that earned each are in this module's own `MONOTONE` table.
///
/// The claim is about a direction, so the code holds it to one: a measurement that finds the lone
/// run reading *lower* than the in-set run for such a count reports it like any other count the
/// selection decides, because the fact its row rests on did not hold there.
pub fn monotone(name: &str) -> Option<&'static str> {
    MONOTONE
        .iter()
        .find(|(count, _)| *count == name)
        .map(|(_, worst)| *worst)
}

/// Counts that are **not a property of the case at all**, read off what the counter *is* in the
/// runtime's own source rather than off whether a reading moved.
///
/// This is the third class, beside gated and merely reported, and it is the one a measurement
/// cannot discover. A count here is shared state, a snapshot of it, or a split the machine's
/// progress decides: no bound on it is a bound on the case, so the number of runs it survives says
/// nothing. Deriving the class from "it moved between runs" is how a gate that can only loosen gets
/// built — a count moves when the machine is busy and holds still when it is quiet, and a rule
/// written from the quiet afternoon fires on the busy one. So each row cites its definition, and a
/// measurement that corroborates it is noted as corroboration and never as the reason.
///
/// Two streams found this class the hard way on the same afternoon. One ran the whole set twice,
/// same binary, same tree, same venue, nothing changed between runs, and got **four cases over
/// bounds both times with only one case in common** — and found that only because its first log was
/// truncated and it reran. The other read the counters behind its own movements and found every one
/// of them process-shared. Neither symptom could be read off a single run, which is the property
/// that makes this class worth naming rather than discovering again.
const SHARED: [(&str, &str); 9] = [
    // `work_counts.rs`: "every batch, by how far its queue's work ran behind its submission".
    // Four buckets of one histogram, and which bucket a batch lands in is decided by how fast the
    // GPU drained what was submitted before it, earlier cases included. The sum is the case's own
    // batch count; the split is the machine's. `batches_with_none_pending` was the most widely
    // gated count in the suite, at 440 cases, and `batches_with_one_pending` was bounded at 0 in
    // all but one — so a batch that found one earlier batch still pending moved a count from the
    // first bucket into a bound of zero in the second, in whichever cases the GPU happened to be
    // behind. That is the shape of four-over-bounds-in-a-different-four-each-run.
    ("batches_with_none_pending", QUEUE_DEPTH),
    ("batches_with_one_pending", QUEUE_DEPTH),
    ("batches_with_two_or_three_pending", QUEUE_DEPTH),
    ("batches_with_four_or_more_pending", QUEUE_DEPTH),
    // Groups "that stayed held past the pass that found no batch ready, while two or more earlier
    // submissions of their `VkQueue` were not seen complete", and of those the ones that went
    // "because a thread began to wait". Both are about what the host had seen finish, which is the
    // same question as the histogram above. **No measurement corroborates these two**: they read 0
    // in every case of the set today, so they are gated at zero and would fire the first time the
    // queue ran behind. They are here on their definitions alone, which is the point of the class.
    ("groups_kept_pending", QUEUE_DEPTH),
    ("groups_sent_for_waits", QUEUE_DEPTH),
    // "found a submission holding its place in the submissions' order and waited until that
    // submission was made", on a device that cannot write heap sets while a command buffer binding
    // them is pending. Whether a registration finds a submission holding the order is a race with
    // that submission. Reads 0 on this venue, and is here on its definition.
    ("registrations_waited", QUEUE_DEPTH),
    // The two halves of a transition's thread that an earlier pass left gated because nothing had
    // been seen to move them. Definition-first says otherwise, and consistency matters more than
    // the reach: `scheduled_where_submitted` is documented as running "on the scheduler's thread
    // as their last batch went, with no handler thread woken", the complement of
    // `wakes_handler_threads`, exactly as `completed_where_seen` is; and `finales_apart` is forced
    // apart partly by "whose last command buffer does not go to the queue as recorded", which is
    // submission-path state. Neither has been seen to move. Both belong here anyway, because the
    // reason a count is reported is what the counter counts.
    ("scheduled_where_submitted", COMPLEMENT),
    ("finales_apart", FOLDED),
];

/// Why a count of how far a queue's work ran behind is reported and never gated.
const QUEUE_DEPTH: &str = "how far this `VkQueue`'s work ran behind when the case submitted, or \
     what the host had seen complete by then: the machine's progress on work submitted before this \
     case, not anything the case asked for";

/// Where `name` is not a property of the case at all.
///
/// The names and the definition that put each one here are in this module's own `SHARED` table,
/// named in prose rather than linked because this function is public and that table is not.
pub fn not_the_cases_own(name: &str) -> Option<&'static str> {
    SHARED
        .iter()
        .find(|(count, _)| *count == name)
        .map(|(_, why)| *why)
}

/// Why `name` is reported and never gated, where it is.
pub fn racy(name: &str) -> Option<&'static str> {
    RACY.iter()
        .find(|(pattern, _)| name.starts_with(pattern))
        .map(|(_, why)| *why)
        .or_else(|| not_the_cases_own(name))
}

/// Which region of the work a count belongs to.
///
/// The regions are kept apart in every report, because moving work between them is not a win.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Region {
    /// Making a shader module, a pipeline layout or a pipeline: the one-time cost, which already
    /// has its own ceilings.
    Translation,
    /// What a call does on the calling thread: the region the plan names as the gap.
    DeviceApi,
    /// What the commit path, our own other threads and the GPU do: the commands recorded into
    /// `VkCommandBuffer`s, the submissions, the scheduler's passes and the waits. Work a call
    /// queues for later is counted here, so a call that returns at once cannot score as a win.
    Submission,
}

impl Region {
    /// Every region, in the order a report prints them.
    pub const ALL: [Region; 3] = [Region::Translation, Region::DeviceApi, Region::Submission];

    /// The words a report prints.
    pub fn word(self) -> &'static str {
        match self {
            Region::Translation => "translation",
            Region::DeviceApi => "device-api",
            Region::Submission => "submission",
        }
    }
}

/// The name prefixes that put a count in [`Region::Translation`]: making a module, a layout or a
/// pipeline.
const TRANSLATION: [&str; 19] = [
    // The AIR decode and specialization a pipeline creation does for its stages, and the question
    // it asks about a linked function's uses. Translation work by definition -- an entry's
    // specialization depends on the functions and the constants, never on the pipeline state -- so
    // without these rows all three score as the device API's residual and read as a call costing
    // nothing. They are named here before the counters land, which costs nothing: `region_of`
    // matches a prefix, and a prefix nothing is named with classifies nothing.
    //
    // They exist because the render-pipeline row's cause turned out to be exactly this work being
    // redone: every creation decoded and specialized its stages' AIR again, and
    // `stage_entries_specialized` fell from 64 to 2 on `integer-vertex-attributes` and from 45 to 1
    // on `vertex-format-constant-step`. A count that large landing in the wrong region would have
    // hidden the fix as well as the bug.
    "stage_entries_",
    "linked_use_questions_asked",
    // The descriptor key a creation builds to look its compiles, its linked-use answers and its
    // reflection up under: a digest of every field of the descriptor, each stage by its compile
    // key, which the creation builds once and passes to all of them.
    "descriptor_keys_built",
    // The lowering itself: a stage's function translated to SPIR-V, against an ask the memo, an
    // archive or the disk answered. It sits between the two rows around it and is the region's
    // whole point -- `stage_entries_` is the AIR decode *before* lowering and `shader_modules_` is
    // the object made from the SPIR-V *after* it, so a state that translated again and came out
    // byte-identical moved neither, which is how a compute sweep going from 36 translations to 1
    // moved no bound in this file.
    "functions_",
    "shader_modules_",
    "pipeline_layouts_",
    "variant_pipelines_",
    "pipelines_pooled",
    "pipeline_pool_busy",
    "creation_pipelines_",
    "graphics_pipelines_looked_up",
    "pipeline_libraries_",
    "fragment_shader_libraries_",
    "pipelines_linked",
    "dispatch_pipelines_pooled",
    "tile_pipelines_pooled",
    "variants_taken_back",
    "flights_followed",
    "residual_form_makings",
];

/// The name prefixes that put a count in [`Region::Submission`]: the commit path, our own other
/// threads and the GPU.
const SUBMISSION: [&str; 48] = [
    "commands_recorded.",
    "command_buffers_",
    "command_pools_made",
    "spare_command_pools_",
    "encoder_command_buffers_begun",
    "encoder_primaries_begun_ahead",
    "primaries_begun_ahead",
    "queue_submissions",
    "submit_infos",
    "internal_submissions",
    "staging_submissions",
    "grouped_submissions",
    "batches_",
    "group_ends_",
    "groups_",
    "registrations_waited",
    "scheduler_",
    "wakes_",
    "slot_returns",
    "slot_waits",
    "completed_waits",
    "event_waits_met",
    "shared_event_waits_held",
    "timeline_waits",
    // The sites one wait for a completion is made of, which the runtime counts one counter a
    // site. Each is a wait, so each belongs here beside `completed_waits` and `timeline_waits`
    // rather than in the device API's residual: `fence_waits` is the explicit alternative to
    // `timeline_waits` (which object the queue's waiter waited for), `completion_arms_` is a
    // queue waiter arming for its queue's front batch, `completion_reads_ran_ahead` is the
    // second wait a read past the wait's value costs, and the gate and trail rows are the two
    // halves of `waitUntilCompleted`'s own wait. Named one by one and not by a `_waits` suffix,
    // because the suffix would also catch a count of what a call asked for.
    "fence_waits",
    "completion_arms",
    "completion_reads_ran_ahead",
    "gate_waits",
    "gate_wakes_found_shut",
    "trail_waits",
    "missed_completion_wakes",
    "completed_where_seen",
    "scheduled_where_submitted",
    "timestamp_slots_",
    "calibration",
    "queue_drains",
    "commits_handed_off",
    "render_log_",
    "replay_logs_kept",
    "encoders_reencoded",
    "segments_recorded_again",
    "replay_tables_pointed",
    "reencodes_refused",
    "finales_",
    "barrier_buffers_kept",
    "stamped_buffers_",
    "bare_starts",
    "prefixed_starts",
];

/// Which region `name` belongs to.
///
/// The longest matching prefix wins, and a name no rule names is the device API's: that is the
/// residual on purpose, because the device API is where a count lands unless it is demonstrably
/// one of the other two, and a count quietly attributed to translation or to submission would read
/// as a call costing nothing.
pub fn region_of(name: &str) -> Region {
    let best = |rules: &[&str]| -> usize {
        rules
            .iter()
            .filter(|rule| name.starts_with(**rule))
            .map(|rule| rule.len())
            .max()
            .unwrap_or(0)
    };
    let translation = best(&TRANSLATION);
    let submission = best(&SUBMISSION);
    if translation == 0 && submission == 0 {
        return Region::DeviceApi;
    }
    if translation >= submission {
        Region::Translation
    } else {
        Region::Submission
    }
}

/// The counts one run moved, by name: the cost half's reading of [`WorkCounts`].
///
/// The names are the runtime's own field names, nested ones joined with a dot
/// (`commands_recorded.cmd_draw`), so a failure names the counter a reader can grep for.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Counts(BTreeMap<String, u64>);

impl Counts {
    /// The counts `counts` holds, flattened.
    ///
    /// Read out of the derived `Debug`, which is the one complete view of the struct a sibling
    /// crate has: [`WorkCounts`] carries several hundred fields of its own plus its nested count
    /// structs, and a hand-written list of them would be stale the day a runtime change added one
    /// -- as a count written here is, which is why this says no number. A field whose
    /// value is not a count fails here by name rather than being dropped, so a new field of
    /// another shape is a loud failure in this crate's own tests instead of a silent hole in the
    /// gate.
    pub fn of(counts: &WorkCounts) -> Result<Self, String> {
        Self::read(&format!("{counts:#?}"))
    }

    /// [`Counts::of`]'s parse, over the `Debug` text itself, so a test can feed it a shape.
    pub fn read(debug: &str) -> Result<Self, String> {
        let mut read: BTreeMap<String, u64> = BTreeMap::new();
        let mut path: Vec<String> = Vec::new();
        let mut skipping: Option<usize> = None;
        for (index, line) in debug.lines().enumerate() {
            let at = index + 1;
            let body = line.trim();
            if body.is_empty() {
                continue;
            }
            let indent = line.len() - line.trim_start().len();
            if indent % 4 != 0 {
                return Err(format!("line {at}: {body}: an indent of {indent} spaces"));
            }
            let depth = indent / 4;
            if let Some(from) = skipping {
                if depth == from && (body == "]," || body == "}," || body == "]" || body == "}") {
                    skipping = None;
                }
                continue;
            }
            if body == "}," || body == "]," || body == "}" || body == "]" {
                path.pop();
                continue;
            }
            if body.ends_with('{') || body.ends_with('[') {
                let name = match body.split_once(':') {
                    Some((name, _)) => name.trim().to_owned(),
                    // The root (`WorkCounts {`), or an element of an array.
                    None => String::new(),
                };
                if excluded(&name).is_some() {
                    skipping = Some(depth);
                } else {
                    path.push(name);
                }
                continue;
            }
            let (name, value) = body
                .split_once(':')
                .ok_or_else(|| format!("line {at}: {body}: neither a field nor a brace"))?;
            let value = value.trim().trim_end_matches(',');
            let count: u64 = value.parse().map_err(|_| {
                format!(
                    "line {at}: {}: {value} is not a count. A field of WorkCounts that is not a \
                     count must be named in cost::NOT_A_COUNT with the reason, so the cost gate \
                     does not quietly drop it.",
                    name.trim()
                )
            })?;
            let mut full = String::new();
            for part in path.iter().skip(1) {
                full.push_str(part);
                full.push('.');
            }
            full.push_str(name.trim());
            if excluded(&full).is_none() {
                read.insert(full, count);
            }
        }
        if !path.is_empty() {
            return Err(format!("{} unclosed brace(s)", path.len()));
        }
        Ok(Self(read))
    }

    /// The work done between `before` and this reading: what one case cost.
    ///
    /// A count that went down (a count of live objects, where the case released more than it made)
    /// reads as zero: the cost half holds what a case *did*, and a bound on a negative difference
    /// would hold nothing.
    pub fn since(&self, before: &Counts) -> Counts {
        let mut moved = BTreeMap::new();
        for (name, after) in &self.0 {
            let was = before.0.get(name).copied().unwrap_or(0);
            let did = after.saturating_sub(was);
            if did > 0 {
                moved.insert(name.clone(), did);
            }
        }
        Counts(moved)
    }

    /// The counts and their values, by name.
    pub fn iter(&self) -> impl Iterator<Item = (&str, u64)> {
        self.0.iter().map(|(name, count)| (name.as_str(), *count))
    }

    /// What `name` reached, zero where the run never moved it.
    pub fn get(&self, name: &str) -> u64 {
        self.0.get(name).copied().unwrap_or(0)
    }

    /// How many counts the run moved.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the run moved none.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The counts of `region`, with their values.
    pub fn in_region(&self, region: Region) -> Vec<(&str, u64)> {
        self.iter()
            .filter(|(name, _)| region_of(name) == region)
            .collect()
    }

    /// The largest of two readings, count by count: what a measurement of several runs holds.
    pub fn largest(&self, other: &Counts) -> Counts {
        let mut most = self.0.clone();
        for (name, count) in &other.0 {
            let held = most.entry(name.clone()).or_insert(0);
            *held = (*held).max(*count);
        }
        Counts(most)
    }

    /// The smallest of two readings, count by count: the other end of a measurement's range.
    pub fn smallest(&self, other: &Counts) -> Counts {
        let mut least: BTreeMap<String, u64> = BTreeMap::new();
        for name in self.0.keys().chain(other.0.keys()) {
            let mine = self.get(name);
            let theirs = other.get(name);
            least.insert(name.clone(), mine.min(theirs));
        }
        Counts(least)
    }
}

/// One count a case is held to, and the measurement's own reading of it.
///
/// A view of one entry of a [`CostRecord`], built from it rather than stored: the record keeps the
/// counts as two maps of name to number, one line a count, because the case set's bounds are about
/// 6,300 counts and what a reader diffs after a re-measurement should be one line per count that
/// moved rather than five.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CostBound {
    /// The runtime's name for the count.
    pub count: String,
    /// Which region of the work it belongs to, as [`region_of`] says. Read from the name rather
    /// than stored, so a file can never disagree with the code's own classification.
    pub region: Region,
    /// The most the measurement saw: the bound. A run over it is [`CostVerdict::Over`].
    pub most: u64,
    /// The least the measurement saw, where the measurement's runs disagreed.
    ///
    /// A count that moved between runs of the same case on the same venue is **not** deterministic,
    /// so it is reported rather than gated: a bound that fails at random is worse than no bound.
    /// This is the evidence for not gating it, and it names the range it was seen in.
    pub range: Option<u64>,
    /// What the same count read when the case ran **alone**, where the two shapes of the
    /// measurement disagreed.
    ///
    /// A count that reads one number inside the case set and another with nothing before it is not
    /// a property of the case at all: it is a property of the selection, and the next case to land
    /// beside this one changes it. Such a count is reported rather than gated, and this is the
    /// reading that says so.
    pub composed: Option<u64>,
    /// Why this count is reported and never gated whatever the measurement saw ([`racy`]).
    pub racy: Option<&'static str>,
}

impl CostBound {
    /// Whether this count is gated: the measurement found it deterministic between runs **and**
    /// between the two shapes it was run in, and it is not one of the counts that say which thread
    /// arrived first.
    pub fn gated(&self) -> bool {
        self.range.is_none() && self.composed.is_none() && self.racy.is_none()
    }
}

/// What one venue's runs of a case cost, and the measurement behind it.
///
/// It lives in the case's recording, beside the Metal answers, so a host with no Metal reads it
/// with the rest of the file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostRecord {
    /// The venue the counts were measured on ([`crate::case::Venue::label`]). A run on any other
    /// venue is [`CostVerdict::Unmeasured`]: our own cost is the implementation's *and* the
    /// driver's, and one venue's counts are no bound for another's.
    pub venue: String,
    /// The runs behind the bounds.
    pub runs: u32,
    /// The measurement that justified the bounds, in words: what was run, where, and how the
    /// bounds were taken from it. The suite's rule is that a bound names its measurement, and this
    /// is where a reader finds it.
    pub measured: String,
    /// The measurement of the same cases **alone**, in words, where one was taken: the contrast
    /// that decided which counts the gate holds. `None` means no contrast has been taken for this
    /// venue, and then nothing in the record is claimed to be independent of the selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contrasted: Option<String>,
    /// The degrade-notice codes the measurement's runs announced, sorted, but the device's native
    /// answers ([`crate::replay::degrading`]).
    ///
    /// **This is what keeps a device's gap from reading as a cost win.** A count that is lower
    /// because the device did less of the case — it lacks the exact form, so the runtime announced
    /// a notice and did something cheaper — is not work we saved. A run whose announced set is not
    /// this one did a different amount of work, so it is [`CostVerdict::Unmeasured`] against these
    /// bounds rather than compared with them: the same rule the correctness half applies when it
    /// keeps a device's honest gap out of the pass and the failure counts. A native note is not in
    /// it, here or in the run's set: the native form is the only one built, so it marks no
    /// different amount of work.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub degraded: Vec<String>,
    /// The most each count the runs moved reached. A count absent from here is bounded at zero, so
    /// a case that starts doing something it did not do before is `Over` rather than unnoticed.
    pub bounds: BTreeMap<String, u64>,
    /// The least each count reached whose readings **disagreed** between the runs: the other end
    /// of its range, and the record of the fact that the count is not deterministic here. A count
    /// named here is reported rather than gated.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub moving: BTreeMap<String, u64>,
    /// What each count read when the case ran **alone** on a device nothing else had touched,
    /// where that disagreed with its reading inside the case set. A count named here is reported
    /// rather than gated.
    ///
    /// This is the contrast run, taken by the measurement rather than by hand. It is the one test
    /// that tells a count of the case's own work from a count of the selection's, and it is the
    /// reason a bound stops going stale when a case lands beside this one: a reading a landing can
    /// change is exactly a reading that depends on what ran before it, which is what this
    /// disagreement is.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub composed: BTreeMap<String, u64>,
    /// What each count read with the case **alone**, where that reading is the count's ceiling and
    /// so its bound: the lone run made the object itself where company finds it already made, and
    /// nothing in the `MONOTONE` table's direction can exceed that.
    ///
    /// Such a count is **gated**, unlike one in `composed`, so this map is not a list of
    /// exemptions — it is where the bound came from. It has to be written down rather than left
    /// implicit in `bounds`, because a later measurement of the gate's own shape reads a *lower*
    /// number for exactly these counts: without this map, re-measuring the whole set alone would
    /// drop every ceiling bound and quietly weaken the gate on the one family the contrast gives
    /// back.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub ceilings: BTreeMap<String, u64>,
    /// The `VkPipeline`s the measurement made over the Metal pipeline states the case created:
    /// the one bound in this record that is a **ratio** and not a count
    /// ([`PipelinesPerState`]).
    ///
    /// It is gated, and it is the only bound here taken in one shape alone. Every count is bounded
    /// at whichever shape is its ceiling, but the lone run makes *more* pipelines per state than
    /// the set does — it finds nothing already made — so its ratio is looser than the gate's own
    /// run will ever read, and keeping the larger of the two would give up the gate entirely.
    /// [`judge`] refuses any selection but the whole set, so the whole set is the only shape this
    /// is ever judged in, and it is measured there.
    ///
    /// **A re-bless of this field means one thing: that the tree now makes this many pipelines per
    /// Metal pipeline state in this case, and that whoever blessed it looked at why.** A case that
    /// grew a state needs no re-bless, because the ratio did not move; a number that moved is a
    /// number of pipelines per state that moved, and the only honest reasons to write a larger one
    /// down are that the case asks for work it did not ask for before, or that a device cannot do
    /// what it did before — and the second shows up as a changed `degraded` set, which makes the
    /// record `Unmeasured` rather than comparable.
    ///
    /// A record taken before this field existed holds none, and gates no ratio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_state: Option<PipelinesPerState>,
    /// Why the venue cannot measure this case at all, where it cannot.
    ///
    /// A run the venue stops -- it lacks a feature the case needs, or the case could not be
    /// replayed -- did a part of the case, so its counts bound nothing. Before this, the writing
    /// command printed that reason and dropped it, which left the recording byte-identical to one
    /// nobody had measured: a refusal and an unmeasured gap read the same from the files, and a
    /// silent absence is what the owner's rule forbids. Written here, a refusal is durable, and
    /// the count of them is readable without a device.
    ///
    /// A record carrying this holds no bounds and gates nothing. [`judge`] reads it as
    /// [`CostVerdict::Unmeasured`] **before** it reads any count, because an absent bound
    /// otherwise means zero and would gate every count the case moves at nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unmeasurable: Option<String>,
}

impl CostRecord {
    /// The bound for `count`, where the record holds one.
    pub fn bound(&self, count: &str) -> Option<CostBound> {
        let most = *self.bounds.get(count)?;
        Some(CostBound {
            count: count.to_owned(),
            region: region_of(count),
            most,
            range: self.moving.get(count).copied(),
            composed: self.composed.get(count).copied(),
            racy: racy(count),
        })
    }

    /// Every bound the record holds, in name order.
    pub fn held(&self) -> Vec<CostBound> {
        self.bounds
            .keys()
            .filter_map(|count| self.bound(count))
            .collect()
    }

    /// This record, a fresh measurement of the gate's own shape, carrying everything `held`'s
    /// contrast established.
    ///
    /// **Every piece of evidence a bound rests on is written in the record rather than derivable
    /// from it**, and this is the method that depends on that. Re-measuring the whole case set says
    /// nothing new about what each case does alone, so what the lone run established has to survive
    /// the re-measure — and it can only survive if it was written down. Three kinds of evidence are
    /// carried:
    ///
    /// - `moving`, so a count the new runs happened not to see move is not re-gated at zero;
    /// - `ceilings`, the sharp one: those counts are **gated**, and the gate's own shape reads a
    ///   *lower* number for every one of them by construction, so dropping them replaces a real
    ///   ceiling with whatever this composition happened to share. Measured, by re-measuring in
    ///   this order before `ceilings` existed: 389 such bounds vanished across 149 cases, almost
    ///   all `addressable_buffers_made` and `pooled_memory_blocks_made`, with nothing in any output
    ///   saying so;
    /// - `composed`, so a count the selection decides is not re-gated on this composition's
    ///   reading.
    ///
    /// A bound only ever rises here from a *named* piece of the contrast, never because it was
    /// higher last time. Keeping `max(new, held)` for every count would ratchet bounds upward for
    /// ever and hide a genuine reduction, and a gate that can only loosen eventually passes
    /// everything.
    ///
    /// It is idempotent, and `a_second_measurement_of_the_same_shape_changes_nothing` holds it to
    /// that: applying it to a record that already carries a contrast changes no byte.
    pub fn carrying_the_contrast_of(mut self, held: &CostRecord) -> CostRecord {
        self.contrasted = held.contrasted.clone();
        // The ratio is this measurement's own: a fresh whole-set run is the shape the ratio is
        // taken in, so there is nothing of the contrast to carry. It is named rather than left to
        // the reader of a field list, because the ceiling defect this method exists for was
        // exactly a field nobody carried.
        self.per_state = self.per_state.or(held.per_state);
        for (count, least) in &held.moving {
            if racy(count).is_some() {
                continue;
            }
            let most = held.bounds.get(count).copied().unwrap_or(0);
            let bound = self.bounds.entry(count.clone()).or_insert(most);
            *bound = (*bound).max(most);
            let seen = self.moving.entry(count.clone()).or_insert(*least);
            *seen = (*seen).min(*least);
        }
        for (count, apart) in &held.ceilings {
            if racy(count).is_some() {
                continue;
            }
            self.ceilings.insert(count.clone(), *apart);
            let bound = self.bounds.entry(count.clone()).or_insert(*apart);
            *bound = (*bound).max(*apart);
        }
        for (count, apart) in &held.composed {
            if racy(count).is_some() {
                continue;
            }
            let most = held.bounds.get(count).copied().unwrap_or(0);
            let bound = self.bounds.entry(count.clone()).or_insert(most);
            *bound = (*bound).max(most).max(*apart);
            self.composed.entry(count.clone()).or_insert(*apart);
        }
        self
    }

    /// How many of its bounds are gated rather than only reported.
    pub fn gated(&self) -> usize {
        self.held().iter().filter(|bound| bound.gated()).count()
    }
}

/// What a re-measurement would change about the gate, as against what the record holds.
///
/// The cost half's load-bearing fields are the ones the gate reads: a gated bound, and the ceiling
/// a gated bound rests on. Everything else in a record is measured noise by design — a reported
/// count's bound, a range that widened, a count the selection decides — and it moves between runs
/// on any busy machine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Drift {
    pub count: String,
    pub region: Region,
    pub what: Drifted,
    /// What the record holds, where it holds anything.
    pub was: Option<u64>,
    /// What the fresh measurement read, where it read anything.
    pub now: Option<u64>,
    /// The ratio each side holds, where this drift is about one ([`Drifted::PerState`]). A ratio
    /// is two numbers and `was`/`now` are one each, so it needs its own pair rather than a
    /// quotient nobody can read back.
    pub was_ratio: Option<PipelinesPerState>,
    pub now_ratio: Option<PipelinesPerState>,
}

/// Which load-bearing fact a [`Drift`] is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drifted {
    /// A gated bound changed value. The gate's own number moved, which is how a regression gets
    /// blessed into the bounds by a re-measure nobody read.
    GatedBound,
    /// A ceiling changed value, or went missing. The second is the defect this comparison exists
    /// for: 389 ceilings vanished across 149 cases on one re-measure, with nothing in any output
    /// saying so, because a ceiling was a raised number rather than a recorded fact.
    Ceiling,
    /// A count the record gates is no longer gated. Reach the gate had and now does not.
    GateLost,
    /// A count the record does not gate now is. Reach the gate gains, which is not a failure but
    /// is not silent either.
    GateGained,
    /// The pipelines-per-state ratio moved, or went missing. A gated number like a bound, and the
    /// one a re-bless must be read for: the counts under it move whenever a case grows, and the
    /// ratio moves only when the tree makes more pipelines for the same work.
    PerState,
}

impl Drifted {
    /// Whether this is a reason to fail a check: the gate's own numbers, and the facts under them.
    ///
    /// Reach moving either way is reported and not fatal, because reclassifying a count is
    /// deliberate work and the person doing it is reading the output. A number moving under a gate
    /// that still looks the same is the thing nobody reads.
    pub fn is_load_bearing(&self) -> bool {
        matches!(
            self,
            Drifted::GatedBound | Drifted::Ceiling | Drifted::PerState
        )
    }

    /// The words a report prints.
    pub fn word(&self) -> &'static str {
        match self {
            Drifted::GatedBound => "a gated bound moved",
            Drifted::Ceiling => "a ceiling moved or went missing",
            Drifted::GateLost => "no longer gated",
            Drifted::GateGained => "now gated",
            Drifted::PerState => "the pipelines-per-state ratio moved or went missing",
        }
    }
}

impl std::fmt::Display for Drift {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let say = |held: Option<u64>| match held {
            Some(number) => number.to_string(),
            None => "none".to_owned(),
        };
        if self.what == Drifted::PerState {
            let ratio = |held: Option<PipelinesPerState>| match held {
                Some(held) => held.to_string(),
                None => "none".to_owned(),
            };
            return write!(
                out,
                "{} ({}): {}, {} in the record and {} in this measurement",
                self.count,
                self.region.word(),
                self.what.word(),
                ratio(self.was_ratio),
                ratio(self.now_ratio)
            );
        }
        write!(
            out,
            "{} ({}): {}, {} in the record and {} in this measurement",
            self.count,
            self.region.word(),
            self.what.word(),
            say(self.was),
            say(self.now)
        )
    }
}

/// What `fresh` would change about `held`'s gate, in name order.
///
/// This is the tool the ceiling defect asked for. Re-measuring a case writes a whole record, and
/// most of a record legitimately differs between two runs of the same binary over the same cases:
/// a reported count's bound, a range the union widened, a count the selection decides. So a diff
/// of the files says nothing, and the one time it said something — 389 ceilings gone across 149
/// cases — the signal sat under a hundred files of noise and was read as something else first.
///
/// So the comparison is of the gate's own facts alone: a gated bound's value, a ceiling's value and
/// its presence, and whether each count is gated at all. Everything else is skipped, named here
/// rather than filtered silently.
///
/// What makes this readable is that the noise it skips is now only the reported classes. While a
/// queue-depth histogram was still gated, a re-measure moved gated numbers in whichever cases the
/// GPU happened to be behind, and no comparison of gated facts could have separated that from a
/// real change. Reporting that class first is what leaves a gated-bound or ceiling movement meaning
/// something.
pub fn drift(held: &CostRecord, fresh: &CostRecord) -> Vec<Drift> {
    let mut drifted = Vec::new();
    let names: std::collections::BTreeSet<&str> = held
        .bounds
        .keys()
        .chain(fresh.bounds.keys())
        .chain(held.ceilings.keys())
        .chain(fresh.ceilings.keys())
        .map(String::as_str)
        .collect();
    for count in names {
        let (before, after) = (held.bound(count), fresh.bound(count));
        let one = |what: Drifted, was: Option<u64>, now: Option<u64>| Drift {
            count: count.to_owned(),
            region: region_of(count),
            what,
            was,
            now,
            was_ratio: None,
            now_ratio: None,
        };
        // A ceiling is compared by presence as well as by value: losing one is the defect, and it
        // leaves a bound that still looks like a bound.
        let (ceiling, fresh_ceiling) = (
            held.ceilings.get(count).copied(),
            fresh.ceilings.get(count).copied(),
        );
        if ceiling != fresh_ceiling {
            drifted.push(one(Drifted::Ceiling, ceiling, fresh_ceiling));
        }
        match (&before, &after) {
            (Some(before), Some(after)) => match (before.gated(), after.gated()) {
                (true, true) if before.most != after.most => {
                    drifted.push(one(
                        Drifted::GatedBound,
                        Some(before.most),
                        Some(after.most),
                    ));
                }
                (true, false) => {
                    drifted.push(one(Drifted::GateLost, Some(before.most), Some(after.most)))
                }
                (false, true) => drifted.push(one(
                    Drifted::GateGained,
                    Some(before.most),
                    Some(after.most),
                )),
                _ => {}
            },
            // A bound the record holds and the fresh measurement does not: the count is gated at
            // zero now, which is reach lost on anything that was gated.
            (Some(before), None) if before.gated() => {
                drifted.push(one(Drifted::GateLost, Some(before.most), None));
            }
            (None, Some(after)) if after.gated() => {
                drifted.push(one(Drifted::GateGained, None, Some(after.most)));
            }
            _ => {}
        }
    }
    // The ratio, which no count name reaches: it is derived from two of them and from a count of
    // the suite's own, so the loop above cannot see it moving.
    if held.per_state != fresh.per_state {
        drifted.push(Drift {
            count: PER_STATE.to_owned(),
            region: Region::Translation,
            what: Drifted::PerState,
            was: held.per_state.map(|held| held.pipelines),
            now: fresh.per_state.map(|held| held.pipelines),
            was_ratio: held.per_state,
            now_ratio: fresh.per_state,
        });
    }
    drifted
}

/// How much reach a comparison gives up and how much it takes on, counted apart.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Reach {
    /// Counts the record gates that this measurement does not. The gate stops gating them, which
    /// shows up as its coverage falling.
    pub given_up: usize,
    /// Counts this measurement gates that the record does not. This is the direction worth
    /// watching: the bound has no evidence but the measurement that just created it, and it reads
    /// as the gate's coverage *rising*.
    pub taken_on: usize,
}

/// Count the two reach directions in a comparison apart from each other.
///
/// They are not equally safe, so one number for both hides the one that matters. Over the
/// recordings' history 16 of 11,392 classifications flipped and every one gave a gate up; the
/// same count the other way would be the gate holding bounds nothing measured, and a number
/// looked at once cannot catch that sign changing.
pub fn reach(drifted: &[Drift]) -> Reach {
    let count = |what: Drifted| drifted.iter().filter(|one| one.what == what).count();
    Reach {
        given_up: count(Drifted::GateLost),
        taken_on: count(Drifted::GateGained),
    }
}

/// One count's drift across every case a comparison looked at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Moved {
    /// The count's dotted name.
    pub count: String,
    /// How many cases it differed in.
    pub cases: usize,
    /// How many of those are numbers the gate reads.
    pub gated: usize,
}

/// Which counts a whole comparison moved, and in how many cases each, most cases first.
///
/// A re-measure after a landing reports its differences case by case, and that is the wrong shape
/// for the first question anyone asks of it: whether the drift belongs to the thing that changed
/// or is spread across the set. Case by case, 318 differences in 51 cases say nothing either way.
/// By count, `images_started_on_the_host` in 40 cases and `host_texel_copies` in 38 name a
/// host-copy path, and a reader can see at once that the submission counts moved with them rather
/// than on their own.
///
/// The grouping is the counts themselves, which are structural. Grouping by case name would read
/// families out of name prefixes, and a count's own name already says which path moved.
pub fn counts_that_moved(per_case: &[Vec<Drift>]) -> Vec<Moved> {
    let mut all: std::collections::BTreeMap<&str, (usize, usize)> =
        std::collections::BTreeMap::new();
    for case in per_case {
        // A count is one case's drift however many lines it took, so the cases are counted once
        // each: a count that moved twice in one case has not moved in two cases.
        let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for one in case {
            if !seen.insert(one.count.as_str()) {
                continue;
            }
            let row = all.entry(one.count.as_str()).or_insert((0, 0));
            row.0 += 1;
            if one.what.is_load_bearing() {
                row.1 += 1;
            }
        }
    }
    let mut moved: Vec<Moved> = all
        .into_iter()
        .map(|(count, (cases, gated))| Moved {
            count: count.to_owned(),
            cases,
            gated,
        })
        .collect();
    // Most cases first, and by name within a tie so the order is the same on every run.
    moved.sort_by(|a, b| b.cases.cmp(&a.cases).then_with(|| a.count.cmp(&b.count)));
    moved
}

/// A count a run moved further than its bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Overrun {
    pub count: String,
    pub region: Region,
    pub bound: u64,
    pub found: u64,
}

impl std::fmt::Display for Overrun {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            out,
            "{} ({}) reached {} where the measurement's most was {}",
            self.count,
            self.region.word(),
            self.found,
            self.bound
        )
    }
}

/// A gated count of work avoided that read below its bound (the `SPARED` table): this case stopped
/// avoiding work it avoided every run of the measurement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Underrun {
    pub count: String,
    pub region: Region,
    /// The reading every run of the measurement saw, which is the floor as well as the ceiling.
    pub bound: u64,
    pub found: u64,
    /// Why the count is gated from below (the `SPARED` table).
    pub why: &'static str,
}

impl std::fmt::Display for Underrun {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            out,
            "{} ({}) read {} where every run of the measurement saw {}: {}",
            self.count,
            self.region.word(),
            self.found,
            self.bound,
            self.why
        )
    }
}

/// What the cost half says of one case on one venue. Never a correctness verdict, and never mixed
/// with one: Metal has no counterpart for any of these counts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CostVerdict {
    /// Every gated count stayed at or under its bound.
    Within {
        /// The gated counts the run was held to.
        held: usize,
    },
    /// A gated count went over its bound: this case now does more work than the measurement saw.
    Over { over: Vec<Overrun> },
    /// A gated count of work avoided (the `SPARED` table) read below its bound: this case stopped
    /// avoiding
    /// work it avoided every run of the measurement. Held apart from [`CostVerdict::Over`] because
    /// the two say opposite things about the same case, and a report that printed one word for both
    /// would read as "more work" where the finding is "a saving gone".
    Under { under: Vec<Underrun> },
    /// No bound stands for this run, and why. Counted, never a pass: "untimed" and "uncounted" are
    /// tracked the way "unrecorded" is.
    Unmeasured { why: String },
}

impl CostVerdict {
    /// The word a report prints.
    pub fn word(&self) -> &'static str {
        match self {
            CostVerdict::Within { .. } => "within",
            CostVerdict::Over { .. } => "over",
            CostVerdict::Under { .. } => "under",
            CostVerdict::Unmeasured { .. } => "unmeasured",
        }
    }

    /// Whether the run is within its bounds. An unmeasured run is not over them, so it does not
    /// fail the gate; it is counted as the gap it is.
    pub fn is_within(&self) -> bool {
        !matches!(self, CostVerdict::Over { .. } | CostVerdict::Under { .. })
    }
}

impl std::fmt::Display for CostVerdict {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CostVerdict::Within { held } => write!(out, "within {held} bound(s)"),
            CostVerdict::Over { over } => {
                write!(out, "over {} bound(s)", over.len())?;
                for one in over {
                    write!(out, "\n  {one}")?;
                }
                Ok(())
            }
            CostVerdict::Under { under } => {
                let each: Vec<String> = under.iter().map(ToString::to_string).collect();
                write!(out, "under: {}", each.join("; "))
            }
            CostVerdict::Unmeasured { why } => write!(out, "unmeasured: {why}"),
        }
    }
}

/// Hold `observed` to the bounds `records` holds for `venue`.
///
/// `whole_set` says whether the run was of the whole case set, which is the measurement's own
/// shape: a case's counts depend on what the process did before it, because the runtime pools
/// command pools, memory blocks and descriptor sets across cases. A selection of a few cases does
/// *more* fresh work than the same cases inside a whole-set run, so its counts are not the ones
/// the bounds were measured over, and holding them to those bounds would fail honestly-identical
/// work. Such a run is `Unmeasured`. The contrast ([`CostRecord::composed`]) does not lift that:
/// the two shapes it compares are the two extremes of composition, and agreeing at both ends is no
/// evidence about a selection in between.
pub fn judge(
    records: &[CostRecord],
    venue: &str,
    whole_set: bool,
    observed: &Counts,
    states: u64,
    stopped: Option<&str>,
    degraded: &[String],
) -> CostVerdict {
    if let Some(why) = stopped {
        return CostVerdict::Unmeasured {
            why: format!("the run stopped before the case finished ({why}), so its counts are of a part of the case"),
        };
    }
    if !whole_set {
        return CostVerdict::Unmeasured {
            why: "the selection is not the whole case set, and a case's counts depend on what ran \
                  before it in the same process"
                .to_owned(),
        };
    }
    let Some(record) = records.iter().find(|record| record.venue == venue) else {
        let held: Vec<&str> = records.iter().map(|record| record.venue.as_str()).collect();
        return CostVerdict::Unmeasured {
            why: if held.is_empty() {
                format!("no cost record was taken for {venue}, nor for any venue")
            } else {
                format!("no cost record was taken for {venue}; the case holds {held:?}")
            },
        };
    };
    // A refusal the venue wrote down: it cannot measure this case at all, so the record holds no
    // bound. This has to precede every count, because an absent bound means zero and would gate
    // each count the case moves at nothing.
    if let Some(why) = &record.unmeasurable {
        return CostVerdict::Unmeasured {
            why: format!("the venue recorded that it cannot measure this case: {why}"),
        };
    }
    // A device that could not do the case exactly did less of it, and its lower counts are its
    // gap and not our saving. Comparing such a run with a measurement of a different gap would
    // read the gap as a win, so the two are not comparable at all. The device's native answers
    // count on neither side, the record's included, since one taken before the rule left them out
    // may still hold them.
    let degraded = crate::replay::degrading(degraded.iter().cloned());
    let measured = crate::replay::degrading(record.degraded.iter().cloned());
    if measured != degraded {
        return CostVerdict::Unmeasured {
            why: format!(
                "this run announced {degraded:?} and the measurement announced {measured:?}, so \
                 the two did not do the same amount of the case: a count lower because the \
                 device lacks the exact form is the device's gap, not a cost we saved",
            ),
        };
    }
    let mut over = Vec::new();
    let mut under = Vec::new();
    for (count, found) in observed.iter() {
        let (bound, gated) = match record.bound(count) {
            Some(bound) => (bound.most, bound.gated()),
            // A count the measurement never saw move is bounded at zero, and it is gated unless
            // it is one of the thread-race counts: a case that starts doing something it did not
            // do before is exactly what this half is for.
            None => (0, racy(count).is_none()),
        };
        // A count of work avoided is gated from BELOW ONLY: more of it is a win, and a ceiling on
        // it reads that win as "cost more". Measured on the whole set: `integer-vertex-attributes`
        // takes all 64 of its predictions where the record's reading was 0, and the ceiling was its
        // only breach. Its `creation_pipelines_made` keeps its own ceiling, and no command can take
        // a prediction that was not made, so the ceiling here held nothing the other one did not.
        if gated && found > bound && spared(count).is_none() {
            over.push(Overrun {
                count: count.to_owned(),
                region: region_of(count),
                bound,
                found,
            });
        }
    }
    // The ratio, which is the one bound here that is not a count ([`PipelinesPerState`]). It is
    // judged from the record's own field rather than from `observed`, because a run that made no
    // pipeline at all moved neither count and must still be held: a case that stopped predicting
    // reads zero made at creation and more variants, and only the ratio sees the total.
    if let Some(per_state) = record.per_state {
        let made = pipelines_made(observed);
        if per_state.exceeded_by(made, states) {
            over.push(Overrun {
                count: PER_STATE.to_owned(),
                region: Region::Translation,
                bound: per_state.allows(states),
                found: made,
            });
        }
    }
    // A count of work avoided is gated from below as well ([`SPARED`]), read from the record's own
    // names rather than from the run's: a count the run never moved is absent from `observed`, and
    // a saving that went to nothing is exactly the reading this gate is for. A count the
    // measurement never saw is bounded at zero, and zero is no floor.
    for bound in record.held() {
        let Some(why) = spared(&bound.count) else {
            continue;
        };
        let found = observed.get(&bound.count);
        if bound.gated() && found < bound.most {
            under.push(Underrun {
                count: bound.count.clone(),
                region: bound.region,
                bound: bound.most,
                found,
                why,
            });
        }
    }
    match (over.is_empty(), under.is_empty()) {
        (true, true) => CostVerdict::Within {
            held: record.gated(),
        },
        // An overrun and an underrun together is still `Over`: more work done is the louder
        // finding, and the underruns ride in the report's own listing of the case.
        (false, _) => CostVerdict::Over { over },
        (true, false) => CostVerdict::Under { under },
    }
}

/// Which shape a cost measurement was taken in.
///
/// Neither bounds the other, and that is the whole reason both are taken. A pooled making is
/// largest with the case **alone**, because a case with nothing before it makes what a case in
/// company finds already made, so a bound taken in company goes stale the moment a case lands
/// beside this one and makes the shared thing first. A count one of our own background threads
/// moves is often largest **in company**, because the work it does ahead and the waits it borrows
/// come from whatever ran before, so a bound taken alone would fail such a count for ever.
///
/// So the gate holds only the counts the two shapes agree on, and each shape is measured by its
/// own command into the same record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// The whole selection in the case set's own order: the shape the gate itself runs in.
    InSet,
    /// Each case on a device nothing else has touched.
    Alone,
}

/// What one case's cost run came to: the counts it moved, and how long it took.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaseCost {
    pub case: String,
    /// The counts of the run with the largest readings, where several runs were taken.
    pub counts: Counts,
    /// The smallest reading of each count, where several runs were taken: the other end of the
    /// range. Equal to `counts` where every run agreed.
    pub least: Counts,
    /// The runs taken.
    pub runs: u32,
    /// The **most** Metal pipeline states any run of the case created ([`crate::states`]): the
    /// denominator of the pipelines-per-state ratio.
    ///
    /// The most rather than the least, to match `counts`: both halves of the ratio come from the
    /// run that did the most, so the quotient is of one shape of the measurement and not of two.
    /// A case that creates none reads zero and carries no ratio bound.
    pub states: u64,
    /// Why the run stopped before the case finished, where it did. Its counts are of a part of the
    /// case, so nothing is recorded or gated from them.
    pub stopped: Option<String>,
    /// The degrade-notice codes the runs announced, sorted, but the device's native answers
    /// ([`crate::replay::degrading`]): what the device could not do exactly, and so how much of
    /// the case the counts are of.
    pub degraded: Vec<String>,
    /// How long the longest run took, in nanoseconds. **Advisory, and it stays advisory.**
    ///
    /// *Gate on counters, report on times.* A count is deterministic, so it is assertable on any
    /// host under any load; a time is not, and that asymmetry is the whole reason the counts carry
    /// this gate. So no time is stored in a [`CostRecord`], [`judge`] reads none, and
    /// [`CostVerdict`] has no variant for one. This number is printed so a reader can see which
    /// cases the counts are large for, and that is all it is for.
    ///
    /// A case's recorded timings are a separate half with its own storage on the recording, taken
    /// by its own leg. Nothing there feeds this gate either. A proposal to make a time fail a run
    /// is a change to the round's rules and not to this field.
    pub nanos: u64,
}

impl CaseCost {
    /// The ratio this measurement justifies, where the case created a state at all.
    ///
    /// A case that creates none has no denominator and carries no bound: nothing is gated at a
    /// ratio of zero states, because every such case would read as infinitely over the moment it
    /// made one pipeline.
    pub fn per_state(&self) -> Option<PipelinesPerState> {
        (self.states > 0).then(|| PipelinesPerState {
            pipelines: pipelines_made(&self.counts),
            states: self.states,
        })
    }

    /// The record this measurement justifies, for `venue`, with `measured` the words that say what
    /// was run.
    ///
    /// A count whose runs disagreed carries its range and is reported rather than gated.
    pub fn record(&self, venue: &str, measured: String) -> CostRecord {
        let bounds = self
            .counts
            .iter()
            .map(|(count, most)| (count.to_owned(), most))
            .collect();
        // The stored list **is** the printed one ([`CaseCost::moved`]), not a second computation of
        // the same rule. Two paths over one rule drift silently and only one of them is ever
        // checked: breaking the ceiling exception in `apart_from` alone, where the printed and
        // stored forms of *that* rule were separate, left every test in this crate passing. A
        // command's output and a record's value are worth collapsing wherever they can be.
        let moving = self
            .moved()
            .into_iter()
            .map(|(count, least, _)| (count.to_owned(), least))
            .collect();
        CostRecord {
            venue: venue.to_owned(),
            runs: self.runs,
            measured,
            contrasted: None,
            degraded: self.degraded.clone(),
            bounds,
            moving,
            composed: BTreeMap::new(),
            ceilings: BTreeMap::new(),
            per_state: self.per_state(),
            unmeasurable: None,
        }
    }

    /// Fold this measurement of the cases **alone** into `held`, the record the gate's own shape
    /// wrote: every count the two shapes read differently becomes one the selection decides.
    ///
    /// The two shapes are two commands and not two legs of one, because one process cannot measure
    /// both cleanly. Measured: `bc1-rgba` run on its own reads `commands_recorded.cmd_dispatch` 4
    /// and `command_buffers_deferred` 6 in three invocations out of three, and **0** for both when
    /// the same leg runs after a whole-set leg in the same process — the deferred replay is a
    /// thread of ours, and whether it lands inside a case's window is not the same question after
    /// 393 cases have run. So each shape is measured in a process of its own and the record
    /// accumulates, which is also what lets either shape be re-measured by itself.
    pub fn fold_alone_into(&self, held: &CostRecord, measured: String) -> CostRecord {
        let mut folded = held.clone();
        folded.contrasted = Some(measured);
        // `folded.per_state` stays the whole-set measurement's. A case alone makes more pipelines
        // per state than the same case in company, because it finds nothing already made, so this
        // leg's ratio is looser than anything the gate will ever read and writing it down would
        // give the gate up. Measured: `xr-two-planes-rgb10a8-2p-xr10-srgb` makes 8 variants alone
        // and 4 in the set over the same states.
        // The counts the selection decides are the ones the command prints ([`apart_from`]), the
        // same list and not a second computation of it. The rule it applies — which counts the two
        // shapes disagree on, less the ones the race rule already reports and the ones whose
        // ceiling the lone run is — lives there alone.
        for (count, _, apart) in self.apart_from(held) {
            folded.composed.insert(count.to_owned(), apart);
        }
        // And the counts this run read *above* the gate's own shape for the one reason that is a
        // ceiling rather than a disagreement. They stay gated, at this reading, and it is written
        // down so a later measurement of the gate's own shape cannot drop it.
        for (count, apart) in self.counts.iter() {
            let inside = held.bounds.get(count).copied().unwrap_or(0);
            if monotone(count).is_some() && apart > inside {
                folded.ceilings.insert(count.to_owned(), apart);
            }
        }
        // The bound stays the largest either shape saw, so the record's own number is honest about
        // what was observed. For a gated count the two shapes agree, so only the bounds of counts
        // that are reported anyway widen. A count the race rule names keeps the gate's own shape's
        // bound, because the lone run says nothing about a reading that is a thread's arrival.
        for (count, _) in self.counts.iter() {
            if racy(count).is_some() {
                continue;
            }
            let apart = self.counts.get(count);
            let bound = folded.bounds.entry(count.to_owned()).or_insert(apart);
            *bound = (*bound).max(apart);
        }
        folded
    }

    /// The counts this measurement reads differently from `held`: the ones the selection decides.
    ///
    /// Each is `(count, the other shape's reading, this one's)`. Counts the [`racy`] rule already
    /// names are left out, because the rule ungates them whatever either shape read, and so is a
    /// count reading higher alone whose ceiling that is ([`monotone`]): the gate keeps it at the
    /// larger reading rather than giving it up.
    pub fn apart_from<'a>(&'a self, held: &'a CostRecord) -> Vec<(&'a str, u64, u64)> {
        let names: std::collections::BTreeSet<&str> = self
            .counts
            .iter()
            .map(|(count, _)| count)
            .chain(held.bounds.keys().map(String::as_str))
            .collect();
        names
            .into_iter()
            .filter_map(|count| {
                let (inside, apart) = (
                    held.bounds.get(count).copied().unwrap_or(0),
                    self.counts.get(count),
                );
                let its_ceiling = monotone(count).is_some() && apart > inside;
                (inside != apart && !its_ceiling && racy(count).is_none())
                    .then_some((count, inside, apart))
            })
            .collect()
    }

    /// The counts whose readings disagreed between runs: the ones no bound gates.
    pub fn moved(&self) -> Vec<(&str, u64, u64)> {
        self.counts
            .iter()
            .filter_map(|(count, most)| {
                let least = self.least.get(count);
                (least < most && racy(count).is_none()).then_some((count, least, most))
            })
            .collect()
    }

    /// The counts whose readings disagreed for a reason the [`racy`] rule already names: which
    /// thread arrived first. Reported apart, because they explain themselves.
    pub fn moved_by_a_race(&self) -> Vec<(&str, u64, u64)> {
        self.counts
            .iter()
            .filter_map(|(count, most)| {
                let least = self.least.get(count);
                (least < most && racy(count).is_some()).then_some((count, least, most))
            })
            .collect()
    }
}

/// How many counts a reading moved in each region, for the cost report's own summary.
///
/// How *many*, not their sum. The counts are of different things — command buffers, images,
/// allocations, texels copied, bytes of device memory — so adding them gives a number with no
/// unit: one case's reading here summed to 1,073,750,616 in the device-API region, almost all of
/// it one texel count. The individual counts are what the gate holds; this says where a case's
/// work falls.
pub fn by_region(counts: &Counts) -> BTreeMap<Region, usize> {
    let mut many = BTreeMap::new();
    for (name, _) in counts.iter() {
        *many.entry(region_of(name)).or_default() += 1;
    }
    many
}

/// Every cost record carries the **contrast** its gated bounds rest on, held over the committed
/// recordings.
///
/// # The defect this is for, found by chasing two cases
///
/// Two cases read `over` on four real breaches in a whole-set replay —
/// `render-state-render-barriers` and `render-state-heap-residency`, over
/// `creation_pipelines_made`, `variant_pipelines_made`, `pipeline_layouts_made` and
/// `shader_modules_made` — and a `--write` pass wrote the same numbers back. Neither a regression
/// nor a case that grew: **their records have no contrast at all**, so the one protection those
/// four counts have was never applied to them.
///
/// The four are all in [`MONOTONE`], whose whole purpose is that they be gated at the **lone run's**
/// reading, which is their ceiling, so company can only lower them. That reading comes from
/// `cost --alone --write` and is kept in [`CostRecord::ceilings`]. With no contrast there are no
/// ceilings, and the bound is the single in-set reading the measurement happened to take.
///
/// For these two that reading was taken in a run where the variant **took back a released
/// variant's objects**. The runtime's own docs say what that does: a variant made again whose
/// released twin's objects were kept makes no `VkPipeline`, no layout and no module
/// (`pipeline/kept.rs`), and `variants_taken_back` counts it. So:
///
/// | case | made (creation/variant/layout/module) | taken back | binds | draws |
/// |---|---|---|---|---|
/// | `render-state-heap-residency` | 0 / 0 / 0 / 0 | 1 | 1 | 1 |
/// | `render-state-render-barriers` | 1 / 1 / 1 / 2 | 1 | 2 | 2 |
/// | `render-state-boxes` | 2 / 2 / 2 / 3 | 0 | 2 | 6 |
/// | `render-state-facing` | 1 / 2 / 1 / 2 | 0 | 3 | 10 |
/// | `render-state-depth-clip` | 0 / 1 / 0 / 0 | 2 | 4 | 5 |
///
/// `render-state-heap-residency` bound and drew a pipeline it made **nothing** for and shared
/// nothing either: it took one back, so all four bounds are zero, and a run where the release had
/// not landed makes them itself and reads four counts over bounds of nought. The siblings whose
/// measurement caught no take-back carry the full counts, and the two that read `over` are the two
/// cases in the suite that share the identical shader reference
/// (`tob_renders.ll`, `tob_rect+tob_solid+tob_write+tob_read`), which is what makes a take-back
/// between them possible at all.
///
/// And the count that decides it is one this half **refuses to gate**: [`FOUND`] names
/// `variants_taken_back` as the first of its family measured to go *up* with more cases around it,
/// 0 alone against 4 in the set. So whether those four `_made` counts read 0 or 1 turns on whether
/// a release on one of our own threads had landed — the composition class, exactly.
///
/// # What the gate is
///
/// A record with no contrast gates its `MONOTONE` counts on whichever composition the measurement
/// met, and nothing says so. **All 1,045 records that hold bounds carry one**, so this is a gate at
/// zero rather than a ledger: a record written without a contrast fails here.
///
/// What the contrast did to the two cases that sent this stream looking:
/// `render-state-heap-residency`'s four bounds went from 0 / 0 / 0 / 0 to **1 / 1 / 1 / 2**, each
/// now a recorded ceiling, and `render-state-render-barriers`'s from 1 / 1 / 1 / 2 to
/// **2 / 2 / 2 / 4**. Those are the readings the breaches had been showing all along. Both now also
/// report `creation_pipelines_drawn` rather than gating it — 0 in the set against 1 alone for the
/// first, 1 against 2 for the second — which is a floor honestly given up by the contrast instead
/// of one resting on a single composition.
///
/// **Those four numbers were measured twice, by two legs that did not know about each other**, and
/// the agreement is worth more than either reading. This stream took the eleven records' missing
/// contrast with `cost --alone` and got 1 / 1 / 1 / 2 and 2 / 2 / 2 / 4; the whole-set cost landing
/// measured all 1,103 cases in both shapes independently and wrote the same four numbers for both
/// cases. A composition-decided bound reproduced across two separate measurements is the strongest
/// form the diagnosis below could have, so the records are the landing's and the diagnosis is kept
/// here.
#[cfg(test)]
mod the_contrast {
    /// Every cost record that holds bounds holds the contrast they rest on.
    ///
    /// A gate at zero and not a ledger, because the bill is paid: the way to pay it is
    /// `cost --alone --runs 8 --write <the cases>`, and it takes a selection rather than the whole
    /// set, because the alone shape is per-case by construction -- each case on a fresh device
    /// nothing else has touched -- and folds into the whole-set record the gate's own shape wrote.
    #[test]
    fn every_cost_record_carries_its_contrast() {
        let mut without: Vec<String> = Vec::new();
        let mut held = 0;
        for case in crate::cases::all() {
            let Ok(recording) = crate::recording::read(case.name()) else {
                continue;
            };
            for record in &recording.costs {
                // A record of a refusal holds no bounds, so it has nothing for a contrast to say
                // anything about.
                if record.unmeasurable.is_some() || record.bounds.is_empty() {
                    continue;
                }
                if record.contrasted.is_some() {
                    held += 1;
                } else {
                    without.push(format!("{} on {}", case.name(), record.venue));
                }
            }
        }
        println!(
            "{held} cost record(s) carry the contrast their gated bounds rest on, {} do not",
            without.len()
        );
        assert!(
            held > 600,
            "only {held} record(s) were found to carry a contrast, so this census is reading the \
             wrong thing"
        );
        assert!(
            without.is_empty(),
            "{} cost record(s) hold bounds with no contrast behind them. Such a record gates its \
             MONOTONE counts -- creation_pipelines_made, \
             variant_pipelines_made, pipeline_layouts_made, shader_modules_made and the rest -- on \
             whichever composition its one measurement met, which is how two cases came to be \
             bounded at the reading where their variant took back a released twin's objects and \
             made nothing. Take the contrast with `cost --alone --runs 8 --write <the cases>`:\n  {}",
            without.len(),
            without.join("\n  ")
        );
    }
}

/// The submission counters' own arithmetic, held over the **committed recordings**.
///
/// `queue_submissions` counts the `vkQueueSubmit` and `vkQueueBindSparse` calls the runtime made,
/// and its documentation puts **internal and staging submissions among them**: each is a
/// submission of its own to the driver. `submit_infos` counts the `VkSubmitInfo`s those calls
/// held. So on a record that makes an internal or staging submission at all, the calls must
/// exceed the infos by exactly those submissions, and that is arithmetic rather than a
/// measurement.
///
/// **This exists because a bless got it wrong and nothing could see it.** One whole-set `--write`
/// pass recorded `queue_submissions` equal to `submit_infos` on all 20 records that make such a
/// submission -- 30/30 with 5 internal on the fourteen `compressed-moves-*`, 36/36 with 12 on the
/// six `pvrtc-cpu-order-*` -- which is the bound understating the calls by exactly the internal
/// submissions. An understated bound makes the gate *tighter* than the tree, so those 20 cases
/// read `Over` on every run from that bless until the next one, and the only thing that found it
/// was a person reading a diff. The measurement before that bless and the measurement after it
/// both hold this identity (137 = 81 + 56 at the host-shadow landing, 35 = 30 + 5 and 48 = 36 + 12
/// now), and no commit between them changed a line of the counting.
///
/// It is deliberately **not** stated as `queue_submissions >= submit_infos` over every record: a
/// batch's new resources ride as a second `VkSubmitInfo` inside the batch's own call
/// (`texture/first_move.rs`), so 391 committed records hold one more info than calls and are right
/// to. The identity is claimed only where an internal or staging submission exists, which is where
/// the counters' own definitions decide it.
#[cfg(test)]
mod the_submissions {
    /// The fewest records that must exercise the identity, so the census cannot rot into passing
    /// by matching nothing -- the same floor [`super::the_ratio`]'s own census carries.
    ///
    /// Twenty, measured: the fourteen `compressed-moves-*` and the six `pvrtc-cpu-order-*`, the
    /// families whose block textures are decoded on the host and whose shadows each cost an
    /// internal submission.
    const EXERCISE_IT: usize = 20;

    /// **The calls exceed the infos by exactly the internal and staging submissions.**
    #[test]
    fn a_submission_of_its_own_is_counted_among_the_queue_submissions() {
        let mut exercised = 0;
        let mut broken: Vec<String> = Vec::new();
        for case in crate::cases::all() {
            let Ok(recording) = crate::recording::read(case.name()) else {
                continue;
            };
            for record in &recording.costs {
                if record.unmeasurable.is_some() || record.bounds.is_empty() {
                    continue;
                }
                let held = |count: &str| record.bounds.get(count).copied().unwrap_or(0);
                let (internal, staging) =
                    (held("internal_submissions"), held("staging_submissions"));
                if internal + staging == 0 {
                    continue;
                }
                // A count the runs disagreed on is reported rather than gated, and its bound is
                // then one end of a range: the arithmetic says nothing about it.
                let moved = [
                    "queue_submissions",
                    "submit_infos",
                    "internal_submissions",
                    "staging_submissions",
                ]
                .iter()
                .any(|count| record.moving.contains_key(*count));
                if moved {
                    continue;
                }
                exercised += 1;
                let (calls, infos) = (held("queue_submissions"), held("submit_infos"));
                if calls != infos + internal + staging {
                    broken.push(format!(
                        "{} on {}: queue_submissions {calls}, submit_infos {infos}, \
                         internal_submissions {internal}, staging_submissions {staging} \
                         (the calls exceed the infos by {}, where {} submission(s) of their own \
                         were made)",
                        case.name(),
                        record.venue,
                        calls as i64 - infos as i64,
                        internal + staging,
                    ));
                }
            }
        }
        println!("{exercised} cost record(s) make a submission of their own and hold the identity");
        assert!(
            exercised >= EXERCISE_IT,
            "only {exercised} record(s) exercise the identity, where {EXERCISE_IT} did, so this \
             census is reading the wrong thing or a `--write` pass dropped the counts"
        );
        assert!(
            broken.is_empty(),
            "{} cost record(s) break the submission counters' own arithmetic. An internal or \
             staging submission is a `vkQueueSubmit` call of its own, so it is counted in \
             `queue_submissions` as well as in its own counter. A record where it is not has an \
             understated bound, which makes the gate TIGHTER than the tree: every run then reads \
             `Over` on a count nothing regressed. Re-measure the gate's own shape with \
             `conformance cost --runs 8 --write`:\n  {}",
            broken.len(),
            broken.join("\n  ")
        );
    }
}

/// The ratio's own census, held over the **committed recordings** rather than over a run.
///
/// [`PipelinesPerState`] is the one bound a reader has to look at rather than diff, and for most of
/// this suite's life it was a field no record held: the records were written by a binary that
/// predated it, `per_state` is `skip_serializing_if = "Option::is_none"`, and a record without one
/// gates no ratio. **0 of 1,103 recordings carried it, the gate was green, and it held nothing** —
/// which is the shape a gate rots into and the shape nothing in the tree could see. These three
/// numbers are what makes it visible, and they need no device: every figure is read off the
/// committed files, so they hold in CI and on a loaded host.
///
/// The ratio is written only by a whole-set `cost --runs 8 --write`, because [`judge`] refuses any
/// selection but the whole set and a lone run's ratio is looser than the gate will ever read.
#[cfg(test)]
mod the_ratio {
    /// The fewest cost records that must carry a ratio.
    ///
    /// A floor, measured: 643 of the 1,117 recordings hold one (8 runs of the whole set on an Apple
    /// M3 Max). The other 474 carry none for a reason the record itself states — 464 cases create
    /// no Metal pipeline state, so the ratio has no denominator, and 10 hold a recorded refusal
    /// instead ([`super::CostRecord::unmeasurable`]).
    ///
    /// It falls when a `--write` pass drops the field, which is exactly how it came to be zero.
    const CARRY_A_RATIO: usize = 643;

    /// The most `VkPipeline`s the committed ratios may hold per Metal pipeline state over the whole
    /// suite, as its two counts: 2,687 over 1,762, 1.525 each.
    ///
    /// The per-case bound is tighter than this everywhere, so this is not a second gate on the same
    /// thing. It is the one number that sees a **re-bless**: a pass that raises many cases' ratios
    /// by a little leaves every per-case bound satisfied by its own new reading, and only the total
    /// says the suite now makes more pipelines for the same work.
    ///
    /// **Re-blessed from 2,665 over 1,755 by the whole-set measurement that earned it**, which is
    /// what this constant asks for in its own panic: the four cases the round's packets added that
    /// create a pipeline state carry the whole of the rise, and the rate it works out to moved
    /// 1.519 to 1.525 while no per-case worst row did — `swizzled-storage-rgba8unorm` is still
    /// 45 over 4.
    const PIPELINES: u64 = 2_687;
    /// The states [`PIPELINES`] was measured over.
    const STATES: u64 = 1_762;

    /// The most records that may make a creation pipeline and carry **no** ratio.
    ///
    /// Two, and the reason is not that they were missed. `binary-archive-adders` and
    /// `binary-archive-serializing` move `creation_pipelines_made` from inside
    /// `addComputePipelineFunctionsWithDescriptor:` and the archive's step replay, where **Metal
    /// creates no pipeline state at all** — so the denominator is honestly zero and the suite's own
    /// door ([`crate::states::making_a_state`]) is right not to count one. Their numerator is
    /// therefore bounded by `creation_pipelines_made` alone and by no ratio, which is the open
    /// shape here: a `VkPipeline` we make for no Metal pipeline state is the worst row this project
    /// has and the ratio cannot express it.
    ///
    /// A ceiling and not permission: a third row means either another call of ours makes pipelines
    /// outside a state creation, or a case that does create a state lost its ratio.
    const NO_DENOMINATOR: usize = 2;

    /// One recording's ratio, or its absence.
    struct Row {
        case: String,
        venue: String,
        pipelines: u64,
        states: u64,
    }

    /// Every committed cost record that holds bounds, with the ratio it carries.
    fn census() -> (Vec<Row>, Vec<Row>) {
        let mut held = Vec::new();
        let mut without = Vec::new();
        for case in crate::cases::all() {
            let Ok(recording) = crate::recording::read(case.name()) else {
                continue;
            };
            for record in &recording.costs {
                if record.unmeasurable.is_some() {
                    continue;
                }
                match record.per_state {
                    Some(ratio) => held.push(Row {
                        case: case.name().to_owned(),
                        venue: record.venue.clone(),
                        pipelines: ratio.pipelines,
                        states: ratio.states,
                    }),
                    // Only a record that made a creation pipeline is owed one: every other case
                    // either creates no state or made nothing to divide.
                    None => {
                        let made = record
                            .bounds
                            .get("creation_pipelines_made")
                            .copied()
                            .unwrap_or(0);
                        if made > 0 {
                            without.push(Row {
                                case: case.name().to_owned(),
                                venue: record.venue.clone(),
                                pipelines: made,
                                states: 0,
                            });
                        }
                    }
                }
            }
        }
        (held, without)
    }

    /// **The ratio is a bound the recordings carry**, and this is the number that says how many.
    #[test]
    fn the_recordings_carry_the_ratio_they_are_gated_by() {
        let (held, _) = census();
        let pipelines: u64 = held.iter().map(|row| row.pipelines).sum();
        let states: u64 = held.iter().map(|row| row.states).sum();
        println!(
            "{} cost record(s) carry a pipelines-per-state ratio: {pipelines} pipeline(s) over \
             {states} state(s)",
            held.len()
        );
        assert!(
            held.len() >= CARRY_A_RATIO,
            "{} cost record(s) carry a ratio, where {CARRY_A_RATIO} did. A record without one \
             gates no ratio at all, and that is how this field came to be absent from every one of \
             the 1,103 recordings while the gate read green. Re-measure the shape the gate runs in: \
             `conformance cost --runs 8 --write`",
            held.len()
        );
        // A record may not carry a ratio over no state: the quotient would have no denominator and
        // `allows` answers `u64::MAX`, which is a bound that holds nothing.
        let empty: Vec<&str> = held
            .iter()
            .filter(|row| row.states == 0)
            .map(|row| row.case.as_str())
            .collect();
        assert!(
            empty.is_empty(),
            "{} record(s) carry a ratio over no state, which bounds nothing: {empty:?}",
            empty.len()
        );
    }

    /// **The suite-wide ratio must not rise**, which is the one reading a per-case re-bless cannot
    /// hide.
    #[test]
    fn the_suite_makes_no_more_pipelines_per_state_than_it_does_today() {
        let (held, _) = census();
        let pipelines: u64 = held.iter().map(|row| row.pipelines).sum();
        let states: u64 = held.iter().map(|row| row.states).sum();
        // Cross-multiplied, never two floats: pipelines/states > PIPELINES/STATES.
        let over =
            u128::from(pipelines) * u128::from(STATES) > u128::from(PIPELINES) * u128::from(states);
        if over {
            let mut worst: Vec<&Row> = held.iter().filter(|row| row.states > 0).collect();
            worst.sort_by(|one, two| {
                (two.pipelines * one.states).cmp(&(one.pipelines * two.states))
            });
            let named: Vec<String> = worst
                .iter()
                .take(12)
                .map(|row| {
                    format!(
                        "{} on {}: {} pipeline(s) over {} state(s)",
                        row.case, row.venue, row.pipelines, row.states
                    )
                })
                .collect();
            panic!(
                "the committed ratios now hold {pipelines} pipeline(s) over {states} state(s), \
                 where the ceiling is {PIPELINES} over {STATES}. Every per-case bound can be \
                 satisfied by its own re-blessed reading while this rises, which is the whole \
                 reason it is counted. Either make fewer pipelines per state, or raise the ceiling \
                 in the same commit as the whole-set measurement that justifies it. The worst \
                 rows:\n  {}",
                named.join("\n  ")
            );
        }
        println!("{pipelines} pipeline(s) over {states} state(s) against the ceiling {PIPELINES} over {STATES}");
    }

    /// **A record that makes a `VkPipeline` for no Metal pipeline state is bounded by no ratio**,
    /// and this counts them rather than letting the class grow unseen.
    #[test]
    fn no_more_records_make_pipelines_outside_a_state_creation() {
        let (_, without) = census();
        println!(
            "{} cost record(s) make a creation pipeline and carry no ratio",
            without.len()
        );
        if without.len() > NO_DENOMINATOR {
            let named: Vec<String> = without
                .iter()
                .map(|row| {
                    format!(
                        "{} on {}: {} made at a creation, no state",
                        row.case, row.venue, row.pipelines
                    )
                })
                .collect();
            panic!(
                "{} cost record(s) make a creation pipeline while the case created no Metal \
                 pipeline state, where {NO_DENOMINATOR} did. Such a record has no denominator, so \
                 the ratio gates it at nothing and only `creation_pipelines_made` holds it. Either \
                 the new row makes pipelines outside a state creation -- which is a cost with no \
                 ratio to express it -- or a case that does create a state lost its ratio to a \
                 `--write` pass:\n  {}",
                without.len(),
                named.join("\n  ")
            );
        }
    }
}

/// The creation-time prediction's ledger, held over the **committed recordings** rather than over a
/// run.
///
/// # What it holds and why it can be held at all
///
/// A pipeline state's creation makes a `VkPipeline` for the key it predicts the first command will
/// use ([`PipelinesPerState`] is the ratio above it). `creation_pipelines_made` counts those and
/// `creation_pipelines_drawn` the ones a command then took; the difference is a compile nothing
/// used yet.
///
/// **The difference is not a defect, and no rule here says it must be zero.** A creation cannot
/// know whether a draw or a dispatch will come, and paying the compile there is the whole point,
/// because Metal pays it at library creation. What is a defect is the difference *widening*, and
/// nothing held that. So this is a ceiling on the quantity.
///
/// It needs no device and no quiet host: every number comes from the bounds the cases' own
/// recordings already carry, so it holds in CI and on a machine running nine other builds.
///
/// # The census these numbers are, read off the committed recordings
///
/// 589 recordings carry a cost record that makes a creation pipeline at all. Over them: **1,446
/// made, 815 taken by a command, 631 taken by none.** 158 cases take every one they make, 411 take
/// none of theirs, and 16 take some and not others — the `xr-blend-planes-*` family at 3 made and 1
/// taken, `topology-class-plain-function` and `frame-length-chain` at 4 and 2.
///
/// **The ceiling rose from 508 to 624 because the coverage did, and the rate improved sharply while
/// it did.** Every one of the 1,103 recordings carries a cost record now, where 399 carried none,
/// and the cases that landing added are largely the many-pipeline sweeps that take every pipeline
/// they make: `vertex-format-constant-step` and `vertex-format-float4` at 45 made and 45 taken,
/// `blend-sweep-write-masks` at 32 and 32, `integer-vertex-attributes` at 64 and 64. So the taken
/// share went from 242 of 750 (32%) to 815 of 1,439 (57%) as the number of rows grew by 225. A
/// ceiling that rises with coverage is the honest reading of a bigger census; what it must not do
/// is rise while the rows stay the same, and that is what this fails on.
///
/// # And the total is two predictions, which is the thing it cannot say
///
/// `creation_pipelines_made` is one count over the render prediction (`pipeline/predicted.rs`) and
/// the compute one (`pipeline/dispatch_making.rs`), so 631 says nothing about either. Split by the
/// commands each record's own counts say our leg recorded ([`Speculated`], and no new counter was
/// needed for it), the two are nothing like each other: **the render prediction is taken 754 times
/// in 799 over the 154 rows that draw and do not dispatch, and the compute prediction 0 times in
/// 461 over the 277 rows that dispatch and do not draw.** 461 of the 631 is one prediction with no
/// hit in the whole committed census, and 455 of those 461 are rows that make no render pass at
/// all. [`WASTED_BY_CLASS`] holds the four classes apart and [`DRAWS_TAKEN`] holds the render
/// prediction's reach, which a ceiling on waste does not.
#[cfg(test)]
mod the_creation_prediction {
    /// The most `VkPipeline`s the committed recordings may hold made at a creation and taken by no
    /// command: the waste where it stands today, over the whole suite.
    ///
    /// A ceiling and not a target. Lowering it is a fix and raising it is a change somebody has to
    /// mean: the first thing a reader of a raise should ask is whether the cases that grew were
    /// cases that never draw — legitimate — or cases that mispredict a draw they do make, which
    /// is the 121-pipeline bug this family has had once already.
    ///
    /// **Re-blessed from 624 by the whole-set measurement that earned it, and the question above
    /// is answered rather than assumed.** Diffed row by row against the recordings the pass
    /// replaced: all 585 rows that existed before read the same made, the same taken and the same
    /// waste, and `taken` did not move from 815. The whole of the rise is four rows that did not
    /// exist, every one of them a `read-write-tier-*` case of this round — `read-write-tier-held`
    /// at 1 made and 0 taken, and the three `read-write-tier-round-trip-*` at 2 and 0. They
    /// dispatch rather than draw, so no command can take a predicted render pipeline, which is
    /// the legitimate half of the question and the shape `known-slow-calls.tsv` already names as
    /// the largest single share of this count.
    const WASTED: u64 = 631;

    /// **The same ceiling, per class of speculation**, which is the statement [`WASTED`] cannot
    /// make: it is one number over two predictions, so it holds neither.
    ///
    /// The classes are [`Speculated`]'s, and here is what the committed recordings read today,
    /// every figure from the files alone:
    ///
    /// | class | rows | made | taken | by none | taken |
    /// |---|---|---|---|---|---|
    /// | draws and no dispatch | 154 | 799 | 754 | 45 | 94.4% |
    /// | dispatches and no draw | 277 | 461 | 0 | 461 | 0.0% |
    /// | both kinds | 84 | 107 | 61 | 46 | 57.0% |
    /// | no command at all | 74 | 79 | 0 | 79 | 0.0% |
    /// | all | 589 | 1,446 | 815 | 631 | 56.4% |
    ///
    /// **So the two predictions are nothing like each other, and the total hides it.** The render
    /// prediction is taken 754 times in 799 on the cases that draw, and its 45 misses are spread
    /// one or two to a row over 42 rows led by `sample-positions-*` and the `a2c-*` sweeps — the
    /// shape this module's docs already name as unpredictable, since `GraphicsKey` carries
    /// `sample_positions` and the pass. The compute prediction is taken **0** times in 461 over
    /// 277 rows. 461 of the 631 is one prediction, and 455 of those 461 are rows that make no
    /// render pass at all.
    ///
    /// # Why the total cannot hold this
    ///
    /// A ceiling on the sum passes a trade between the halves. Were the compute prediction's 461
    /// to go to nothing while the render prediction lost half its accuracy — 799 made, about 400
    /// taken — the census would read about 586 against the ceiling of 631 and this family would
    /// report everything well. The per-class ceilings below fail it, and name the class that
    /// widened.
    ///
    /// Raising one of these is the same act as raising [`WASTED`] and wants the same answer: were
    /// the rows that grew rows that run no command of that kind — legitimate — or rows that
    /// mispredict a command they do run.
    const WASTED_BY_CLASS: [(Speculated, u64); 4] = [
        (Speculated::Draws, 45),
        (Speculated::Dispatches, 461),
        (Speculated::Both, 46),
        (Speculated::NoCommand, 79),
    ];

    /// **The floor under the render prediction's reach**: the predicted `VkPipeline`s a draw took,
    /// over the rows that draw and do not dispatch.
    ///
    /// A ceiling on waste is not this statement. A case family that stops drawing, or a landing
    /// that removes one, lowers the waste and the reach together, and only the ceiling notices
    /// nothing. This is the half that says the prediction is still *right*, and it is the reason
    /// the render prediction must not be made lazy: at 754 of 799 it is taken 94.4% of the time,
    /// so a lazy render pipeline would move a creation's compile into the first draw's command
    /// buffer 94 times in 100 — which is the cold making inside a command buffer's execution that
    /// `pipeline/predicted.rs` exists to have removed.
    const DRAWS_TAKEN: u64 = 754;

    /// The cases whose prediction the existing pair of bounds already holds **exactly**, and the
    /// reach that must not be lost.
    ///
    /// Where `creation_pipelines_made` is gated at `n` from above and `creation_pipelines_drawn` is
    /// gated at `n` from below ([`super::SPARED`]), the two together say the case takes every
    /// pipeline its creations make — a stricter statement than either bound, held by nothing that
    /// says so. 111 of the 158 cases that take every one are in that position.
    ///
    /// The other 47 are **not**, and they are families rather than a list of names: 19
    /// `rotated-planes-*` rows, 8 `visibility-*` rows, and then ones and twos across
    /// `blend-sweep-*`, `depthstencil-*` and `blend-interaction-mask-blend`. The
    /// `rotated-planes-*` ones are the clean compute examples, where the prediction is right
    /// because the case follows Apple's own threadgroup guide, and their
    /// `creation_pipelines_drawn` is reported rather than gated because the two measurement shapes
    /// read it differently: the record carries bound 1 with the lone run's 1 in `composed`, so the
    /// case in company read 0 made and 0 taken where alone it read 1 and 1. **The equality
    /// survives that and neither count does**, which is the same thing the ratio above exploits,
    /// so holding those wants a recorded relation rather than a recorded count. The `visibility-*`
    /// rows give up `creation_pipelines_made` as well, so for them even the ceiling is the
    /// selection's. The four `render-state-*` rows reached this class by being **fixed**: taking
    /// their missing contrast (`the_contrast`, beside this module) raised their `_made` bounds to
    /// the lone
    /// run's ceiling and moved `creation_pipelines_drawn` into `composed`, which is a floor
    /// honestly given up rather than a floor lost. Blessing a relation needs a whole-set and an
    /// alone measurement on a device, so the gap is recorded here and not guessed at.
    const EXACTLY_HELD: usize = 111;

    /// **Which speculation a record's creation pipelines are**, and so which prediction its waste
    /// belongs to: the one thing the two counts are blind to, because both are totals over the
    /// render prediction (`pipeline/predicted.rs`) and the compute one
    /// (`pipeline/dispatch_making.rs`) together.
    ///
    /// It is read off the commands the record's own counts say our leg recorded, and from nothing
    /// else — no case name, no family, no list. A `VkPipeline` made at a creation can only ever be
    /// taken by a command of its own kind, so a record that recorded draws and no dispatch can
    /// only hold the render prediction's pipelines, and one that recorded dispatches and no draw
    /// only the compute one's. That is a structural fact about the record, available in every
    /// committed recording today, which is why this needed no new counter.
    ///
    /// **The signal is `commands_recorded.cmd_draw*` and `cmd_dispatch*` and not
    /// `draw_states_recorded`**, which looks like the same question and is not: 22 of these rows
    /// record a `vkCmdDraw` while reading `draw_states_recorded` 0 — every `sample-fills-*` and
    /// `buffer-depth-samples-*` row — so classifying by it would file a drawing case under
    /// [`Speculated::NoCommand`] and hide exactly the rows this family already knows mispredict.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Speculated {
        /// Draw commands and no dispatch: the render prediction's, all of it.
        Draws,
        /// Dispatch commands and no draw: the compute prediction's, all of it.
        Dispatches,
        /// Both kinds, so the record cannot say which prediction a pipeline of it belongs to.
        /// Attributing these wants the counter split named in this module's docs.
        Both,
        /// Neither kind reached the driver, so no command could have taken anything the creation
        /// made. The one class where the waste is not a mispredict at all.
        NoCommand,
    }

    impl Speculated {
        /// What `bounds`, a record's, say the record's creation pipelines are.
        fn of(bounds: &std::collections::BTreeMap<String, u64>) -> Speculated {
            let any = |prefix: &str| {
                bounds
                    .iter()
                    .any(|(count, most)| count.starts_with(prefix) && *most > 0)
            };
            match (
                any("commands_recorded.cmd_draw"),
                any("commands_recorded.cmd_dispatch"),
            ) {
                (true, false) => Speculated::Draws,
                (false, true) => Speculated::Dispatches,
                (true, true) => Speculated::Both,
                (false, false) => Speculated::NoCommand,
            }
        }

        /// The class in words, for a failure and for the census the tests print.
        fn label(self) -> &'static str {
            match self {
                Speculated::Draws => "draws and no dispatch",
                Speculated::Dispatches => "dispatches and no draw",
                Speculated::Both => "both kinds",
                Speculated::NoCommand => "no command at all",
            }
        }
    }

    /// One recording's reading of the prediction.
    struct Row {
        case: String,
        made: u64,
        drawn: u64,
        /// Which prediction the row's pipelines are ([`Speculated`]).
        class: Speculated,
        /// Whether both counts are gated, so the pair already says the case takes every one.
        held: bool,
    }

    /// Every committed cost record that makes a creation pipeline, read off the recordings.
    fn census() -> Vec<Row> {
        let mut rows = Vec::new();
        for case in crate::cases::all() {
            let Ok(recording) = crate::recording::read(case.name()) else {
                continue;
            };
            for record in &recording.costs {
                let made = record.bounds.get("creation_pipelines_made").copied();
                let Some(made) = made.filter(|made| *made > 0) else {
                    continue;
                };
                let drawn = record
                    .bounds
                    .get("creation_pipelines_drawn")
                    .copied()
                    .unwrap_or(0);
                let gated = |count: &str| record.bound(count).is_some_and(|bound| bound.gated());
                rows.push(Row {
                    case: case.name().to_owned(),
                    made,
                    drawn,
                    class: Speculated::of(&record.bounds),
                    held: made == drawn
                        && gated("creation_pipelines_made")
                        && gated("creation_pipelines_drawn"),
                });
            }
        }
        rows
    }

    /// What one class of speculation reads over the census.
    struct Class {
        rows: usize,
        made: u64,
        drawn: u64,
    }

    impl Class {
        /// The rows of `class` in `rows`, added up.
        fn of(rows: &[Row], class: Speculated) -> Class {
            let mine = rows.iter().filter(|row| row.class == class);
            let mut read = Class {
                rows: 0,
                made: 0,
                drawn: 0,
            };
            for row in mine {
                read.rows += 1;
                read.made += row.made;
                read.drawn += row.drawn;
            }
            read
        }

        /// The pipelines of this class no command took.
        fn wasted(&self) -> u64 {
            self.made - self.drawn
        }
    }

    /// **The suite-wide ceiling on predicted pipelines no command took.**
    ///
    /// The one gate in this family that needs nothing but the committed files. A change that makes
    /// the prediction miss more often widens the difference and fails here, naming the rows that
    /// waste most so the reader starts from the cause rather than from the total.
    #[test]
    fn the_creation_prediction_wastes_no_more_than_it_does_today() {
        let rows = census();
        let made: u64 = rows.iter().map(|row| row.made).sum();
        let drawn: u64 = rows.iter().map(|row| row.drawn).sum();
        let wasted = made - drawn;
        println!(
            "{} recording(s) make a creation pipeline: {made} made, {drawn} taken by a command, \
             {wasted} by none",
            rows.len()
        );
        if wasted > WASTED {
            let mut worst: Vec<&Row> = rows.iter().filter(|row| row.made > row.drawn).collect();
            worst.sort_by_key(|row| std::cmp::Reverse(row.made - row.drawn));
            let named: Vec<String> = worst
                .iter()
                .take(12)
                .map(|row| format!("{} made {} took {}", row.case, row.made, row.drawn))
                .collect();
            panic!(
                "the committed recordings now hold {wasted} predicted pipeline(s) no command \
                 took, where the ceiling is {WASTED}. A creation cannot know whether a command \
                 will come, so a difference is not a defect -- but a WIDENING one is, and this is \
                 the number that says so. Either fix the prediction, or, if the cases that grew \
                 are cases that never draw at all, raise the ceiling in the same commit as the \
                 measurement that justifies it. The rows that waste most:\n  {}",
                named.join("\n  ")
            );
        }
        // A floor under the ceiling, because a gate that reads the wrong thing passes for ever: a
        // tree where this census found nothing would pass this test while holding nothing.
        assert!(
            rows.len() > 300,
            "only {} recording(s) were found to make a creation pipeline at all, so this census \
             is reading the wrong thing",
            rows.len()
        );
    }

    /// **The same ceiling held per class of speculation** ([`WASTED_BY_CLASS`]), which the total
    /// cannot do: it is one number over the render prediction and the compute one, so a trade
    /// between them passes it.
    #[test]
    fn neither_prediction_wastes_more_than_it_does_today() {
        let rows = census();
        let mut over = Vec::new();
        for (class, ceiling) in WASTED_BY_CLASS {
            let read = Class::of(&rows, class);
            let taken = match read.made {
                0 => 0.0,
                made => read.drawn as f64 / made as f64 * 100.0,
            };
            println!(
                "{:24} {:4} row(s), {:5} made, {:5} taken, {:5} by none against the ceiling \
                 {ceiling} ({taken:5.1}% taken)",
                class.label(),
                read.rows,
                read.made,
                read.drawn,
                read.wasted(),
            );
            if read.wasted() > ceiling {
                let mut worst: Vec<&Row> = rows
                    .iter()
                    .filter(|row| row.class == class && row.made > row.drawn)
                    .collect();
                worst.sort_by_key(|row| std::cmp::Reverse(row.made - row.drawn));
                let named: Vec<String> = worst
                    .iter()
                    .take(8)
                    .map(|row| format!("{} made {} took {}", row.case, row.made, row.drawn))
                    .collect();
                over.push(format!(
                    "{}: {} predicted pipeline(s) no command took, where the ceiling is \
                     {ceiling}. Worst rows:\n    {}",
                    class.label(),
                    read.wasted(),
                    named.join("\n    ")
                ));
            }
        }
        assert!(
            over.is_empty(),
            "a class of speculation now wastes more than it did, which the suite-wide ceiling \
             cannot see: it is one number over two predictions, so it passes a change that breaks \
             one while the other is fixed. Fix the named prediction, or raise that class's \
             ceiling in the same commit as the measurement that justifies it:\n  {}",
            over.join("\n  ")
        );
        // The classes must still partition the census, or a ceiling above holds a class that no
        // longer has any rows in it.
        let classed: usize = WASTED_BY_CLASS
            .iter()
            .map(|(class, _)| Class::of(&rows, *class).rows)
            .sum();
        assert_eq!(
            classed,
            rows.len(),
            "the classes do not partition the census, so a ceiling holds rows nobody counted"
        );
        for (class, _) in WASTED_BY_CLASS {
            assert!(
                Class::of(&rows, class).rows > 0,
                "no row of the census is {}, so that class's ceiling holds nothing",
                class.label()
            );
        }
    }

    /// **The render prediction is still taken as often as it is** ([`DRAWS_TAKEN`]), which is the
    /// half a ceiling on waste does not hold.
    #[test]
    fn the_prediction_a_drawing_case_makes_is_still_taken_as_often() {
        let rows = census();
        let read = Class::of(&rows, Speculated::Draws);
        println!(
            "the {} row(s) that draw and do not dispatch take {} of the {} pipelines their \
             creations make, against the floor {DRAWS_TAKEN}",
            read.rows, read.drawn, read.made,
        );
        assert!(
            read.drawn >= DRAWS_TAKEN,
            "the rows that draw and do not dispatch now take {} of the {} pipelines their \
             creations make, where {DRAWS_TAKEN} were taken. A ceiling on waste cannot say this: \
             a family that stops drawing lowers the waste and the reach together. The render \
             prediction being taken 94% of the time is why it is made at a creation at all, so a \
             drop here is a cold compile moving back inside a first draw's command buffer",
            read.drawn,
            read.made,
        );
    }

    /// The classifier reads the commands and nothing else, over rows this test writes itself.
    ///
    /// The census above reads committed files, so a classifier that answered one class for
    /// everything would still make its ceilings add up. This is the input that says it does not.
    #[test]
    fn a_record_is_classed_by_the_commands_it_recorded() {
        let bounds = |counts: &[(&str, u64)]| {
            counts
                .iter()
                .map(|(name, most)| ((*name).to_owned(), *most))
                .collect::<std::collections::BTreeMap<String, u64>>()
        };
        let draw = ("commands_recorded.cmd_draw_indexed_indirect", 1);
        let dispatch = ("commands_recorded.cmd_dispatch_indirect", 1);
        let other = ("commands_recorded.cmd_copy_buffer", 7);
        assert_eq!(
            Speculated::of(&bounds(&[draw, other])),
            Speculated::Draws,
            "an indirect indexed draw is a draw"
        );
        assert_eq!(
            Speculated::of(&bounds(&[dispatch, other])),
            Speculated::Dispatches,
            "an indirect dispatch is a dispatch"
        );
        assert_eq!(Speculated::of(&bounds(&[draw, dispatch])), Speculated::Both);
        assert_eq!(
            Speculated::of(&bounds(&[other])),
            Speculated::NoCommand,
            "a copy is neither"
        );
        assert_eq!(Speculated::of(&bounds(&[])), Speculated::NoCommand);
        // A count present at zero is a count that did not happen: `bounds` holds the most a count
        // reached, and a recording that carries the name at 0 recorded no such command.
        assert_eq!(
            Speculated::of(&bounds(&[("commands_recorded.cmd_draw", 0), dispatch])),
            Speculated::Dispatches,
            "a draw count of zero is not a draw"
        );
        // `draw_states_recorded` is not the signal, and must not become one: 22 census rows read
        // it at 0 while recording a draw.
        assert_eq!(
            Speculated::of(&bounds(&[("draw_states_recorded", 4), dispatch])),
            Speculated::Dispatches,
            "a recorded draw state is not a draw command"
        );
    }

    /// **The reach over the cases whose prediction is already held exactly must not be lost.**
    ///
    /// For those cases the ceiling on `creation_pipelines_made` and the floor on
    /// `creation_pipelines_drawn` together say the case takes every pipeline its creations make.
    /// Nothing says that in so many words, so nothing notices it going away: a re-bless that raises
    /// the made bound, or that moves the drawn count into `moving` or `composed`, gives the
    /// statement up with no output saying so. This counts it.
    #[test]
    fn the_cases_that_take_every_predicted_pipeline_are_still_held_to_it() {
        let rows = census();
        let exact: Vec<&Row> = rows
            .iter()
            .filter(|row| row.made == row.drawn && row.made > 0)
            .collect();
        let held: Vec<&&Row> = exact.iter().filter(|row| row.held).collect();
        println!(
            "{} case(s) take every predicted pipeline they make, {} of them held to it by a gated \
             ceiling and a gated floor together",
            exact.len(),
            held.len()
        );
        if held.len() < EXACTLY_HELD {
            let lost: Vec<&str> = exact
                .iter()
                .filter(|row| !row.held)
                .map(|row| row.case.as_str())
                .take(12)
                .collect();
            panic!(
                "{} case(s) are held to taking every predicted pipeline they make, where {} were. \
                 A case loses that statement when its made bound rises or its drawn count stops \
                 being gated, and neither says so in any output. Cases that take every one and \
                 are NOT held to it: {lost:?}",
                held.len(),
                EXACTLY_HELD
            );
        }
    }
}
