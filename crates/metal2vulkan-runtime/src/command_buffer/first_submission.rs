//! A command buffer's first submission: Metal resolves what a command buffer's commands reach as it
//! submits it, not as it was committed, so a command buffer committed before a rebind (`rebind.rs`)
//! or a release and submitted after it behaves as one committed after it (the consumer's
//! measurement on the M3 Max, its rebind-reach tables 1 to 3).
//!
//! - **Until then, held weakly.** From the commit to the first submission the command buffer
//!   holds the pages its commands took of buffers over host pages (`rebind::Taken`) and the live
//!   buffers' table its resident set was made of (`Live`) weakly: a rebind or a release meanwhile
//!   lets the pages go, their deallocator run, where no submitted command buffer holds them
//!   (the question, answered without revoking anything). The pages an event group's commands took
//!   (its batches between event waits and signals, `State::event_group`) are held from that
//!   group's first batch's submission until the command buffer completes, so submitted work keeps
//!   the pages it took, as Metal's does (table 3: a submitted command buffer never sees the new
//!   pages); a detach waits for, or refuses on, that group only until its work completed
//!   (`rebind::SubmittedHolds`), as Metal holds a submission's uses. A root table recorded
//!   before then, by a rebind's encoding again (`reencode.rs`) or a host step, holds the table
//!   (`State::hold_live`) until the first submission, which lets go of it where the list changed
//!   meanwhile: at once, as Metal lets a released buffer go, but on a handler thread
//!   (`Device::let_go_elsewhere`), as its last hold of a released buffer's pages runs that
//!   buffer's deallocator, which may not run on the scheduler's thread under its locks. The
//!   deallocator's wait from the release until then, past Metal's, is announced
//!   (`host-deallocator-delayed`).
//! - **Resolved again where the list changed.** Where the live buffers' list changed since the
//!   commit (a buffer made or gone, a rebind), the submission lists again what the commit listed
//!   (`LiveBuffers::reach`, a rebind having ended its use), makes its resident set again of the
//!   list as it stands and of what its encoders made resident as it lies now, and points every
//!   root table at it: a buffer its encoders made resident reaches the pages it lies in now
//!   (case 3, `useResource:` and `useHeap:`, and after two rebinds the second's), and one it
//!   reaches only unnamed reads 0 once a rebind ended its use (cases 4 and 4x). A root table's
//!   own range whose Metal address the remade set holds anew goes: the buffer lies elsewhere.
//!   (A set of any count; before, past the search's 2^16 entries, each root table took back its own
//!   set.)
//! - **What it named, encoded again; else typed.** A command that named a buffer over host pages
//!   was resolved as it was encoded (a binding's descriptor or root-table static, a blit's
//!   operand, an attachment, a vertex, index or indirect buffer, a texture over the pages, an
//!   indirect command's resolved buffer), where Metal reaches the pages the buffer lies in at the
//!   submission (cases 1, 2 and 5). The rebind encodes the compute, blit and render encoders that
//!   named it again before this submission, which waits for them (`reencode.rs`). Where an encoder
//!   that could not be encoded again named a buffer that lies in other pages by then, each of
//!   Metal's submissions holding such a command (`checks::metal_submissions`, the split) runs
//!   nothing and the command buffer completes with `CommandBufferError::EncodedUseRebound`, every
//!   other submission running, its signals firing and its handlers running: the gap until the
//!   encoded uses resolve at submission, announced once
//!   (`Degradation::EncodedUseRebound`). Work recorded on the GPU alone (`gpu_execute.rs`,
//!   `render/executions/gpu.rs`) resolves its commands' records through the resident set its
//!   preprocess's table names: where the list changed or a buffer of it moved since the commit,
//!   the submission points every such table at the set as it lies now, with its entries'
//!   extents (`State::point_replay_tables`, `ResidentSet::replay_bytes`), so a rebind or a
//!   release meanwhile takes effect in its commands, which are recorded once. What its commands
//!   named is encoded or recorded again as an encoder's commands are (`reencode.rs`).
//! - **Each submission at its own batch**. The first submission finds which of Metal's submissions
//!   of the first event group are rebound or hold a detached buffer, and each later group's first
//!   batch those of its group, against where the buffers lie then, as Metal decides each
//!   submission as it submits it (a detach between two submissions fails the later one, code 8,
//!   the earlier one having run: the M3 Max under macOS 27.0.1). A rebind before a later group's
//!   submission encodes its encoders that named the buffer again, as before the first
//!   (`reencode.rs`): Metal's later submission runs over the new pages. The table its first
//!   submission fixes serves every later batch: a later group reaches a buffer detached or
//!   rebound since then through an address (made resident, or unnamed) through the pages it took,
//!   where Metal's reads the new pages or zeros. Each fails only as its own batch is submitted,
//!   after that batch's submission checks (`checks.rs`). One its
//!   check failed already keeps the check's code (8 for the consumer's detach) and announces
//!   nothing, and so does one whose every encoder the commit's failure failed (`commit_failed`'s);
//!   the submissions before it run as Metal runs them, and the command buffer fails with the first
//!   failure in submission order. Any other rebound submission runs nothing and is announced,
//!   though the command buffer failed already and keeps that first code (move14: the consumer's
//!   review; Metal runs it over the new pages).

use super::checks::metal_submissions;
use super::rebound_use::{ReboundAt, ReboundUse, Recording};
use super::{write_fields, CommandBufferShared, Resident, State};
use crate::error::{internal, CommandBufferError, Error};
use crate::live_buffers::LiveTable;
use crate::rebind::{SubmittedHolds, Submitting};
use crate::residency::Residency;
use crate::upload::Upload;
use crate::used_resources::UsedResources;
use crate::values::CommandBufferStatus;
use crate::Device;
use metal2vulkan::bind::{ResidentRange, ResidentSet, RootTable};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::{Arc, Weak};

/// A root table a command buffer's commands read (`State::root_table`): its upload and its
/// device address.
#[derive(Clone)]
pub(crate) struct NotedTable {
    pub(crate) upload: Upload,
    pub(crate) address: u64,
}

/// A replay preprocess's resident root table (`render/executions/gpu.rs`, `gpu_execute.rs`): its
/// upload, where it lies there, and its device address. The command buffer's first submission
/// points it at the resident set as it lies then, where that changed since the commit
/// (`State::point_replay_tables`).
pub(crate) struct ReplayTable {
    pub(crate) upload: Upload,
    pub(crate) at: u64,
    pub(crate) address: u64,
}

/// The live buffers' table a command buffer's resident set was made of (the module docs).
pub(super) enum Live {
    /// Weakly, with the generation of the list it was made of: from the commit to the first
    /// submission.
    Waiting {
        table: Weak<LiveTable>,
        generation: u64,
    },
    /// From the first submission (or a host step before it) until the command buffer completes.
    Held(Arc<LiveTable>),
}

impl Live {
    /// `table`, as a command buffer holds it: weakly before its first submission (`submitted`
    /// false), unless no other holds it (a table made as the list changed meanwhile, which the
    /// device keeps not), whose last hold would let pages go here.
    pub(super) fn of(table: Arc<LiveTable>, submitted: bool) -> Live {
        if submitted || Arc::strong_count(&table) == 1 {
            return Live::Held(table);
        }
        Live::Waiting {
            generation: table.generation(),
            table: Arc::downgrade(&table),
        }
    }

    /// The table's set, where it is held.
    pub(super) fn set(&self) -> Option<&ResidentSet> {
        match self {
            Live::Held(table) => Some(table.set()),
            Live::Waiting { .. } => None,
        }
    }
}

/// What a commit listed among the live buffers (`LiveBuffers::reach`), and the generation of the
/// list after it.
pub(super) struct Listing {
    used: UsedResources,
    generation: u64,
}

impl Listing {
    /// `used`, listed on `device` just now.
    pub(super) fn new(device: &Device, used: UsedResources) -> Listing {
        Listing {
            generation: device.shared().live_buffers.generation(),
            used,
        }
    }
}

/// The Metal addresses a resident set's ranges hold, to find whether a root table's own range
/// lies in one of them: each range's start, by its order, with the furthest end of every range
/// starting there or before (ranges nest: a heap's holds its buffers').
struct Covered(Vec<(u64, u64)>);

impl Covered {
    fn of(ranges: impl IntoIterator<Item = ResidentRange>) -> Covered {
        let mut starts: Vec<(u64, u64)> = ranges
            .into_iter()
            .map(|range| (range.guest, range.guest.saturating_add(range.length.max(1))))
            .collect();
        starts.sort_unstable();
        let mut furthest = 0;
        for (_, end) in &mut starts {
            furthest = furthest.max(*end);
            *end = furthest;
        }
        Covered(starts)
    }

    /// Whether Metal address `guest` lies in one of the ranges.
    fn holds(&self, guest: u64) -> bool {
        let below = self.0.partition_point(|&(start, _)| start <= guest);
        below.checked_sub(1).is_some_and(|at| guest < self.0[at].1)
    }
}

impl State {
    /// As the command buffer commits (its segments recorded): what its commands took is held
    /// weakly until its first submission, which lists `listing` again where the list changed.
    pub(super) fn wait_for_submission(&mut self, listing: Listing) {
        for record in &mut self.ended {
            record.taken.wait();
        }
        let segments = self
            .recorded
            .iter_mut()
            .flat_map(|part| part.segments.iter_mut())
            .filter_map(|segment| segment.recorded.as_mut());
        for done in segments {
            done.taken.wait();
        }
        self.listing = Some(listing);
    }

    /// Hold the live buffers' table its resident set was made of from now on: made again where
    /// the list changed since (the module docs). A root table a host step uploads before the
    /// first submission is pointed at the set as it stands then.
    /// A table held already, by a host step's root table or an encoder encoded again at a rebind
    /// before the first submission (`reencode.rs`), is made again where the list changed since
    /// it was made: the first submission resolves the set as the list stands then.
    pub(super) fn hold_live(&mut self, device: &Device) -> Result<(), Error> {
        let now = device.shared().live_buffers.generation();
        match &self.resident {
            Some(Resident {
                live: Live::Waiting { table, generation },
                ..
            }) => {
                // The list unchanged, the device keeps the table: the upgrade is not its last
                // hold. Changed, the table is not upgraded at all: an upgrade let go of here,
                // under the scheduler's lock and this state's, could be the last hold of a gone
                // buffer's pages, whose release runs the program's code.
                let kept = (*generation == now).then(|| table.upgrade()).flatten();
                if let Some(table) = kept {
                    if let Some(resident) = &mut self.resident {
                        resident.live = Live::Held(table);
                    }
                    return Ok(());
                }
            }
            Some(Resident {
                live: Live::Held(table),
                ..
            }) if table.generation() != now => {}
            _ => return Ok(()),
        }
        let fresh = self.fresh_ranges();
        self.remake_resident(device, &fresh)
    }

    /// Its work on the GPU alone, at its first submission: each preprocess's resident root table
    /// (`replay_tables`), recorded with the resident set as it lay then, pointed at the set as
    /// the live buffers and what its encoders made resident lie now, where that changed
    /// (`moved`), with its extents (`ResidentSet::replay_bytes`), uploaded once for all: a
    /// rebind or a release since takes effect in its commands, as Metal resolves them at the
    /// submission, and nothing is recorded again. The live buffers' table is held until it
    /// completes either way, as their ranges lie in its pages.
    fn point_replay_tables(&mut self, device: &Device, moved: bool) -> Result<(), Error> {
        let live = device.shared().live_buffers.table(device)?;
        if moved {
            let ranges = live.ranges().chain(self.fresh_ranges());
            let (bytes, count) = ResidentSet::replay_bytes(ranges);
            let len = bytes.len() as u64;
            let upload =
                (self.uploads).upload_aligned(device, &bytes, len, RootTable::ENTRY_ALIGNMENT)?;
            let set = (upload.device_address())
                .ok_or_else(|| internal("a replay's resident set without a device address"))?;
            for table in &self.replay_tables {
                let fields = RootTable::resident_fields(table.address, set, count)
                    .ok_or_else(|| internal("a replay's resident table off its set's alignment"))?;
                let at = table.at + u64::from(RootTable::RESIDENT_FIELDS_OFFSET);
                table.upload.write_at(at, &fields);
            }
            let pointed = self.replay_tables.len() as u64;
            device.shared().counts.replay_tables_pointed(pointed);
            self.retained.push(Box::new(upload));
        }
        self.replay_tables.clear();
        self.retained.push(Box::new(live));
        Ok(())
    }

    /// The ranges of what its encoders made resident, as it lies now.
    fn fresh_ranges(&self) -> Vec<ResidentRange> {
        self.residency
            .as_ref()
            .map(Residency::resident_ranges)
            .unwrap_or_default()
    }

    /// Its resident set made again of the live buffers as they stand now and `fresh`, what its
    /// encoders made resident as it lies now, and every noted root table pointed at it. A table
    /// it held goes at once, on a handler thread.
    fn remake_resident(&mut self, device: &Device, fresh: &[ResidentRange]) -> Result<(), Error> {
        let live = device.shared().live_buffers.table(device)?;
        let set = live.set();
        // A root table's own range whose Metal address the remade set holds goes: the buffer
        // there lies as the set says now.
        let covered = Covered::of(live.ranges().chain(fresh.iter().copied()));
        let mut beyond: Vec<ResidentRange> = self
            .root_ranges
            .iter()
            .filter(|range| !covered.holds(range.guest))
            .copied()
            .collect();
        beyond.extend(fresh.iter().copied());
        beyond.retain(|range| !set.holds(range));
        let (merged, address, count) = match beyond.is_empty() {
            true => (None, live.address(), set.len()),
            false => {
                let merged = set.with(beyond);
                let (address, count) = self.upload_set(device, &merged)?;
                (Some(merged), Some(address), count)
            }
        };
        for noted in &self.root_tables {
            write_fields(
                &noted.upload,
                noted.address,
                address.unwrap_or(noted.address),
                count,
            )?;
        }
        let replaced = self.resident.replace(Resident {
            live: Live::Held(live),
            merged,
            address,
            count,
        });
        // A table held already, made of the list before it changed, may hold a gone buffer's
        // pages last, whose release runs the program's code (a deallocator), which may not run
        // here: on the scheduler's thread, under its lock and this state's. It goes at once all
        // the same, as Metal lets go of a buffer no submitted command buffer holds, on a handler
        // thread.
        if let Some(Resident {
            live: Live::Held(table),
            ..
        }) = replaced
        {
            device.let_go_elsewhere(Box::new(table));
        }
        Ok(())
    }

    /// As its first batch is submitted (the module docs), on the thread that submits it, under
    /// the scheduler's lock and its own, after that batch's checks: Metal's submissions of its first
    /// event group ([`State::event_group`]) found rebound, or holding a detached buffer, go to
    /// `failing`, each to fail as its own batch is submitted (`State::failing_at`). What the
    /// group's commands took is judged and held under one lock of the device's buffers over host
    /// pages (`Submitting`), so a detach comes either before (and fails the submission) or after
    /// (and waits for the group's work); a later group's is judged and held as its own first
    /// batch is submitted (`State::failing_at`). Nothing here lets pages go: the weak holds become
    /// strong, and what it held strongly it retains.
    /// `Ok(false)`, changing nothing, while a rebind since its commit has encoders of it, which
    /// `buffer`'s rebinds mark, to encode again (`reencode.rs`): read under the lock its rebinds
    /// mark them under, so a rebind comes either before this submission (which waits for them)
    /// or after it (which keeps the old pages for this group). From here it waits for no
    /// rebind's replay; its encoders' logs stay until its release
    /// (`CommandBufferShared::complete`).
    pub(super) fn first_submission(
        &mut self,
        device: &Device,
        buffer: &CommandBufferShared,
    ) -> Result<bool, Error> {
        let shared = device.shared();
        let mut verdicts = BTreeMap::new();
        let mut holds = SubmittedHolds::new(device);
        let group = self.event_group(0);
        let parts = self.group_parts(&group);
        let encoders = self.first_parts(parts.clone());
        let moved = {
            let mut submitting = shared.rebinds.submitting();
            if buffer.reencode_pending() {
                return Ok(false);
            }
            self.submitted = true;
            self.judged_to = group.end;
            if group.end == self.batches.all().len() {
                submitting.submitted_now(&shared.rebinds, buffer);
            }
            self.judge_group(
                &mut submitting,
                &encoders,
                parts.clone(),
                &mut verdicts,
                &mut holds,
            )
        };
        if !holds.is_empty() {
            self.submitted_holds.push((group.end - 1, holds));
        }
        // The list changed since the commit: what it listed is listed again, a rebind having
        // ended its use, and its resident set made of the list as it stands.
        let changed = self
            .listing
            .as_ref()
            .is_some_and(|listing| listing.generation != shared.live_buffers.generation());
        if changed || moved {
            if let Some(listing) = &self.listing {
                shared.live_buffers.reach(&listing.used);
            }
        }
        self.hold_live(device)?;
        // The tables host steps before now resolved against go at once, off this thread, under
        // whose locks the last hold of a released buffer's pages may not go
        // (`State::keep_stepped_table`).
        for earlier in self.stepped_tables.drain(..) {
            device.let_go_elsewhere(Box::new(earlier));
        }
        // The resident tables of its work on the GPU alone, pointed at the set as it lies now
        // where the list changed or a buffer of it moved since the commit.
        if !self.replay_tables.is_empty() {
            self.point_replay_tables(device, changed || moved)?;
        }
        // Found here, against the pages and the table fixed for every later batch; each fails
        // at its own batch, after its check.
        self.failing = self.failing_submissions(&verdicts);
        // Their names let go; the listing released without the lock.
        self.judged(&encoders, parts);
        let State {
            retained, listing, ..
        } = self;
        // An empty listing (a command buffer that used no no-copy buffer and no heap) holds
        // nothing to let go of.
        if let Some(Listing { used, .. }) = listing.take() {
            if used != UsedResources::default() {
                retained.push(Box::new(used));
            }
        }
        self.root_tables.clear();
        self.root_ranges.clear();
        Ok(true)
    }

    /// Metal's submissions (the split) that hold an encoder `verdicts` fails, by their encoders,
    /// each with the failure of its encoders: `DetachedBacking` where one holds a detached buffer,
    /// else `EncodedUseRebound`.
    fn failing_submissions(
        &self,
        verdicts: &BTreeMap<usize, CommandBufferError>,
    ) -> Vec<(Range<usize>, CommandBufferError)> {
        if verdicts.is_empty() {
            return Vec::new();
        }
        let encoders = self.ended.iter().map(|record| {
            (
                record.kind,
                record.event_waits_before,
                record.event_signals_before,
            )
        });
        metal_submissions(encoders)
            .into_iter()
            .filter_map(|(_, encoders)| {
                let mut failure = None;
                for encoder in encoders.clone() {
                    if let Some(error) = verdicts.get(&encoder) {
                        judge_one(&mut failure, *error);
                    }
                }
                failure.map(|failure| (encoders, failure))
            })
            .collect()
    }

    /// As batch `index` is submitted, after the first submission, where it begins a later event
    /// group ([`State::event_group`]): that group's commands are judged and held as the first
    /// submission judged the first group's, as Metal decides each submission as it submits it,
    /// those found rebound or holding a detached buffer going to `failing`. `false`, changing
    /// nothing, while a rebind since the last group's submission has encoders of this one, which
    /// `buffer`'s rebinds mark, to encode again (`reencode.rs`): read under the lock its rebinds
    /// mark them under, so a rebind comes either before this group's submission (which waits for
    /// them) or after it. From the last group on it waits for no rebind's replay.
    pub(super) fn later_group(
        &mut self,
        device: &Device,
        buffer: &CommandBufferShared,
        index: usize,
    ) -> bool {
        let group = self.event_group(index);
        if index == 0 || group.start != index {
            return true;
        }
        let shared = device.shared();
        let parts = self.group_parts(&group);
        let encoders = self.first_parts(parts.clone());
        let mut verdicts = BTreeMap::new();
        let mut holds = SubmittedHolds::new(device);
        {
            let mut submitting = shared.rebinds.submitting();
            if buffer.reencode_pending() {
                return false;
            }
            self.judged_to = group.end;
            if group.end == self.batches.all().len() {
                submitting.submitted_now(&shared.rebinds, buffer);
            }
            self.judge_group(
                &mut submitting,
                &encoders,
                parts.clone(),
                &mut verdicts,
                &mut holds,
            );
        }
        self.judged(&encoders, parts);
        let failing = self.failing_submissions(&verdicts);
        self.failing.extend(failing);
        if !holds.is_empty() {
            self.submitted_holds.push((group.end - 1, holds));
        }
        true
    }

    /// Whether it completed or failed: no rebind encodes it again.
    pub(super) fn finished(&self) -> bool {
        matches!(
            self.status,
            CommandBufferStatus::Completed | CommandBufferStatus::Error
        )
    }

    /// The first part of the event groups not judged yet ([`State::later_group`]): the parts from
    /// it on are what a rebind may encode again (`reencode.rs`).
    pub(super) fn unjudged_parts(&self) -> usize {
        match self.batches.all().len() {
            len if self.judged_to >= len => self.recorded.len(),
            _ => self.batch_range(self.judged_to).start,
        }
    }

    /// As batch `index` is submitted, after its checks ran: the segments a host step recorded
    /// for it since the first submission are judged and held as the first submission judged the
    /// rest (`State::first_submission`); then each of Metal's submissions found failing that
    /// begins in the batch, or that holds such a segment, runs nothing from the batch on
    /// (`State::fail_part`), unless its check failed it already, which keeps the check's code,
    /// or the commit's failure failed every encoder of it (`State::failed_at_commit`); and the
    /// command buffer fails with the first such submission's failure unless it failed already,
    /// keeping its first code. Whether one of them failed rebound here (`EncodedUseRebound`),
    /// for the caller to announce without the lock: a rebound submission runs nothing where
    /// Metal's runs, whatever the command buffer failed with before.
    pub(super) fn failing_at(&mut self, device: &Device, index: usize) -> bool {
        let range = self.batch_range(index);
        let group = self.event_group(index);
        let mut later = BTreeMap::new();
        let mut holds = SubmittedHolds::new(device);
        let mut found_later = None;
        {
            let mut submitting = device.shared().rebinds.submitting();
            let State {
                recorded, retained, ..
            } = self;
            for part in &mut recorded[range.clone()] {
                let segments = part.segments.iter_mut().filter_map(|s| s.recorded.as_mut());
                for done in segments {
                    let found = submitting.submit(&mut done.taken, retained, &mut holds);
                    judge(&mut later, part.encoder, found.failure());
                    done.taken.judged();
                    // Recorded after the first submission: by a host step.
                    if found.failure() == Some(CommandBufferError::EncodedUseRebound) {
                        found_later.get_or_insert((part.encoder, Recording::HostStep));
                    }
                }
            }
        }
        if let Some((encoder, recording)) = found_later {
            if let Some(kind) = self.ended.get(encoder).map(|record| record.kind) {
                self.rebound_use.get_or_insert(ReboundUse {
                    encoder,
                    kind,
                    at: ReboundAt::Segment(recording),
                });
            }
        }
        if !holds.is_empty() {
            self.submitted_holds.push((group.end - 1, holds));
        }
        if self.failing.is_empty() && later.is_empty() {
            return false;
        }
        let here: BTreeSet<usize> = self.recorded[range.clone()]
            .iter()
            .map(|part| part.encoder)
            .collect();
        let mut due = Vec::new();
        self.failing.retain(|(encoders, failure)| {
            let begins = encoders.clone().any(|encoder| here.contains(&encoder));
            if begins {
                due.push((encoders.clone(), *failure));
            }
            !begins
        });
        due.extend(self.failing_submissions(&later));
        due.retain(|(encoders, _)| !self.check_failed(encoders) && !self.commit_failed(encoders));
        if due.is_empty() {
            return false;
        }
        due.sort_by_key(|(encoders, _)| encoders.start);
        let State {
            recorded, retained, ..
        } = self;
        let failing = |encoder: &usize| due.iter().any(|(encoders, _)| encoders.contains(encoder));
        for part in recorded[range.start..]
            .iter_mut()
            .filter(|part| failing(&part.encoder))
        {
            State::fail_part(part, retained);
        }
        self.unrunnable
            .extend(due.iter().flat_map(|(encoders, _)| encoders.clone()));
        if self.failure.is_none() {
            self.failure = Some(due[0].1);
        }
        due.iter()
            .any(|(_, failure)| *failure == CommandBufferError::EncodedUseRebound)
    }

    /// The batches of the event group batch `index` lies in: the run of batches around it that
    /// no event (or binary semaphore) wait or signal splits. Metal holds what a group's
    /// submissions use from the group's submission until its work completed (a buffer's detach
    /// waits for that work, and one with its later work behind a wait detaches at once, measured
    /// on the M3 Max), and the runtime judges and holds a group's commands as its first batch is
    /// submitted ([`State::first_submission`], [`State::failing_at`]), letting go as its last
    /// batch's work completed (`CommandBufferShared::holds_until`).
    pub(super) fn event_group(&self, index: usize) -> Range<usize> {
        let batches = self.batches.all();
        let splits = |at: usize| {
            at == 0
                || at >= batches.len()
                || !batches[at].waits.is_empty()
                || !batches[at].binary_waits.is_empty()
                || !batches[at - 1].signals.is_empty()
                || !batches[at - 1].binary_signals.is_empty()
        };
        let start = (0..=index).rev().find(|&at| splits(at)).unwrap_or(0);
        let end = (index + 1..=batches.len())
            .find(|&at| splits(at))
            .unwrap_or(batches.len());
        start..end
    }

    /// The parts of the ended encoders the batches of `group` hold.
    fn group_parts(&self, group: &Range<usize>) -> Range<usize> {
        let start = self.batch_range(group.start).start;
        let end = self.batch_range(group.end - 1).end;
        start..end
    }

    /// The ended encoders whose first part lies among `parts`, in order.
    fn first_parts(&self, parts: Range<usize>) -> Vec<usize> {
        parts
            .filter(|&at| at == 0 || self.recorded[at - 1].encoder != self.recorded[at].encoder)
            .map(|at| self.recorded[at].encoder)
            .collect()
    }

    /// Judge and hold what `encoders` took, and the segments of `parts` recorded so far, against
    /// where their buffers lie now (`Submitting::submit`): their failures go to `verdicts`, their
    /// holds to `holds`, their pages to what the command buffer retains. Whether a buffer one took
    /// lies elsewhere now.
    fn judge_group(
        &mut self,
        submitting: &mut Submitting<'_>,
        encoders: &[usize],
        parts: Range<usize>,
        verdicts: &mut BTreeMap<usize, CommandBufferError>,
        holds: &mut SubmittedHolds,
    ) -> bool {
        let mut moved = false;
        let State {
            ended,
            recorded,
            retained,
            rebound_use,
            ..
        } = self;
        for &at in encoders {
            let Some(record) = ended.get_mut(at) else {
                continue;
            };
            let found = submitting.submit(&mut record.taken, retained, holds);
            moved |= found.moved;
            judge(verdicts, at, found.failure());
            if found.failure() == Some(CommandBufferError::EncodedUseRebound) {
                let at_commands = match (&record.replay, &record.refused) {
                    (None, _) => ReboundAt::Unlogged(record.unlogged),
                    (Some(_), Some(refusal)) => ReboundAt::Refused(refusal.clone()),
                    (Some(_), None) => ReboundAt::NotEncodedAgain,
                };
                rebound_use.get_or_insert(ReboundUse {
                    encoder: at,
                    kind: record.kind,
                    at: at_commands,
                });
            }
        }
        for part in &mut recorded[parts] {
            let kind = ended.get(part.encoder).map(|record| record.kind);
            for segment in &mut part.segments {
                let recording = match (segment.gpu_alone, segment.stepped) {
                    (true, _) => Recording::GpuAlone,
                    (false, true) => Recording::HostStep,
                    (false, false) => Recording::Commit,
                };
                let Some(done) = segment.recorded.as_mut() else {
                    continue;
                };
                let found = submitting.submit(&mut done.taken, retained, holds);
                moved |= found.moved;
                judge(verdicts, part.encoder, found.failure());
                if let (Some(CommandBufferError::EncodedUseRebound), Some(kind)) =
                    (found.failure(), kind)
                {
                    rebound_use.get_or_insert(ReboundUse {
                        encoder: part.encoder,
                        kind,
                        at: ReboundAt::Segment(recording),
                    });
                }
            }
        }
        moved
    }

    /// Let go of the names of what `encoders` and the segments of `parts` took, once judged.
    fn judged(&mut self, encoders: &[usize], parts: Range<usize>) {
        for &at in encoders {
            if let Some(record) = self.ended.get_mut(at) {
                record.taken.judged();
            }
        }
        let segments = self.recorded[parts]
            .iter_mut()
            .flat_map(|part| part.segments.iter_mut())
            .filter_map(|segment| segment.recorded.as_mut());
        for done in segments {
            done.taken.judged();
        }
    }

    /// Whether the commit's failure failed every encoder of Metal's submission of `encoders`.
    fn commit_failed(&self, encoders: &Range<usize>) -> bool {
        self.failed_at_commit
            .as_ref()
            .is_some_and(|failed| encoders.clone().all(|encoder| failed.fails(encoder)))
    }
}

/// Note `failure`, an encoder's at `at`, among `verdicts`: a detached buffer's over a rebound one.
fn judge(
    verdicts: &mut BTreeMap<usize, CommandBufferError>,
    at: usize,
    failure: Option<CommandBufferError>,
) {
    if let Some(failure) = failure {
        let mut held = verdicts.remove(&at);
        judge_one(&mut held, failure);
        if let Some(held) = held {
            verdicts.insert(at, held);
        }
    }
}

/// `failure` joined to `held`: `DetachedBacking` wins over `EncodedUseRebound`.
fn judge_one(held: &mut Option<CommandBufferError>, failure: CommandBufferError) {
    let detached = |error: &CommandBufferError| *error == CommandBufferError::DetachedBacking;
    if held
        .as_ref()
        .is_none_or(|held| !detached(held) && detached(&failure))
    {
        *held = Some(failure);
    }
}

impl State {
    /// Refuse (internal) to submit batch `range`'s parts where one of an encoder that recorded
    /// nothing for a command (`Taken::dead`, of its own or of a segment of it), or of one whose
    /// submission failed rebound or detached (`unrunnable`), still holds work: the structural
    /// hold, that a command buffer holding a detached or rebound use never submits that use's
    /// Vulkan work. A build with debug assertions stops there.
    pub(super) fn runs_nothing_unrunnable(&self, range: Range<usize>) -> Result<(), Error> {
        let held = self.recorded[range].iter().any(|part| {
            let dead = self
                .ended
                .get(part.encoder)
                .is_some_and(|record| record.taken.dead());
            (dead || self.unrunnable.contains(&part.encoder)) && !part.raws.is_empty()
        });
        debug_assert!(!held, "a detached or rebound use's work submitted");
        match held {
            true => Err(crate::error::internal(
                "a detached or rebound use's work submitted",
            )),
            false => Ok(()),
        }
    }
}
