//! A committed command buffer's encoders encoded again at a rebind, before the submission of their
//! event group (the logs are `replay.rs`'s).
//!
//! - **Marked under the rebind's lock.** Its commit notes the command buffer, with the keys of
//!   the buffers over host pages its replayable encoders named, while its encoders' holds still
//!   refuse a rebind of them (`Rebinds::wait_for_replay`), until its last event group is
//!   submitted or it completes. A rebind of one of them marks it (`reencode`) under the device's
//!   lock of those buffers, the lock each group's submission reads the mark under
//!   (`State::first_submission`, `State::later_group`): that submission came before the rebind,
//!   and keeps the old pages (Metal's table 3), or it waits, held at its place in the
//!   scheduler's order, until the mark is gone. A later group's encoders are encoded again as
//!   the first's are: Metal's later submission of a command buffer whose earlier one ran reaches
//!   the new pages (the M3 Max under macOS 27.0.1).
//! - **Encoded again in the rebind's call.** The rebind then encodes again, on its own thread
//!   (the program's call, under no lock of the runtime's), each encoder of the command buffer's
//!   groups not yet submitted a command of which named a buffer that lies elsewhere now than it
//!   took it, from its log, into
//!   new `VkCommandBuffer`s of the command buffer's pool, and puts what it recorded in the
//!   encoder's place: its recordings, uses, executions, keys and what its commands took (held
//!   weakly, as the commit holds them). The first recording is never submitted. No GPU work is
//!   waited for, and no pipeline's making: a dispatch whose pipelines are being made leaves a
//!   segment of its own, as the first encoding left one (`pending_dispatch.rs`), and so does a
//!   render pass of a draw's or a tile dispatch's being made (`render/pending_tile.rs`), which
//!   the submission or the host step records once they are made, so no Metal compile runs or is
//!   waited for on the program's thread inside the rebind's call.
//!   Rebinds on two threads take turns (`reencoding`); a rebind during another's
//!   turn marks the command buffer again, and its own turn encodes what moved since. A host step
//!   of the command buffer, which records into its pool on the scheduler's thread, takes the
//!   same turns: Vulkan synchronizes a pool's recordings externally.
//! - **Work recorded on the GPU alone, recorded again.** Work its commit recorded on the GPU
//!   alone, for good (a GPU-written execution's preprocess and its dispatches or draws,
//!   `gpu_execute.rs`, `render/executions/gpu.rs`), reaches what its commands' records name
//!   through the resident set its preprocess's table names, which the first submission points
//!   at the set as it lies then (`first_submission.rs`): no rebind or release records it again
//!   for that. What its commands named (the pass's attachments, say) was resolved as it was
//!   recorded: an encoder that left such work keeps its log as one that left none does, and its
//!   encoding again records that work at once, on the GPU alone, in segments of the same parts
//!   (`EncoderCore::finish_replay`), its own commands (a render pass's shadowed textures'
//!   copies and its other steps among them) recorded again around them. Its commit notes the
//!   keys that work named too (`State::keep_replays`): a rebind of one of them before the first
//!   submission (which alone points those tables) records the work of the encoders it does not
//!   encode again in its place, as the commit did, over where the buffers lie now: its recording, what its commands took and what
//!   its host step needs. Work a host step recorded again since (with the barriers between its
//!   commands, `gpu_execute::Rerecord`, or as a route outgrew it, `DeferredPass::outgrown`)
//!   keeps that step's recording.
//! - **What cannot be encoded again stays as before.** An encoder whose log is gone (a compute
//!   encoder of a command buffer that used command predication whose unit laid a step before it
//!   opened, or one of its own that the unit's later encoders read, or a loop, or left work out;
//!   one failed at the commit) or whose second
//!   encoding is refused or records otherwise (`EncoderCore::finish_replay`), its work among
//!   them where it no longer runs on the GPU alone, keeps its first recording, which the first
//!   submission finds rebound: `CommandBufferError::EncodedUseRebound`, as before. So does work
//!   on the GPU alone the rebind no longer records there.

use super::rebound_use::Unlogged;
use super::{lock, CommandBuffer, CommandBufferShared, State};
use crate::deferred::{Deferred, StepInputs};
use crate::encoder::{RecordedSegment, Replayed, ReplayedPart, Segment};
use crate::pipeline::budget::Urgency;
use crate::replay::Replay;
use crate::residency::Residency;
use crate::resource::ResourceKey;
use crate::used_resources::EncoderKind;
use std::collections::BTreeSet;
use std::sync::atomic::Ordering;
use std::sync::Arc;

impl CommandBufferShared {
    /// A rebind since its commit, of a buffer one of its replayable encoders named, has them
    /// to encode again ([`reencode`](Self::reencode)) before the submission of their event group:
    /// marked under
    /// the device's lock of its buffers over host pages (`Rebinds::replace`).
    pub(crate) fn mark_reencode(&self) {
        self.reencode.fetch_add(1, Ordering::AcqRel);
    }

    /// Whether the submission of its next event group waits for its encoders encoded again (the
    /// module docs).
    pub(crate) fn reencode_pending(&self) -> bool {
        self.reencode.load(Ordering::Acquire) > 0
    }

    /// Encode again each encoder of its event groups not yet submitted that a command of named a
    /// buffer over host pages that lies elsewhere now than the command took it, and record again
    /// the work its commit recorded on the GPU alone before the first submission (the module
    /// docs), on the rebind's thread, with no lock of the runtime's held; then the submission of
    /// its next group may go, and the scheduler looks again. Nothing here fails the rebind: an
    /// encoder it cannot encode again, or work it cannot record on the GPU alone again, keeps its
    /// first recording, which its group's submission fails. A command buffer that completed
    /// encodes nothing again.
    pub(crate) fn reencode(self: &Arc<Self>) {
        let turn = lock(&self.reencoding);
        let marked = self.reencode.load(Ordering::Acquire);
        if marked == 0 {
            return;
        }
        let device = &self.device;
        // The encoders of its event groups not submitted yet: those before were judged, and run
        // over the pages they took.
        // So are those whose pass took a rasterization-rate map since reset; one that kept no
        // log draws by the map as it took it, announced (`rate_mapped_pass.rs`).
        let mut unencoded = Vec::new();
        let moved: Vec<(usize, Arc<Replay>)> = {
            let state = lock(&self.state);
            match state.finished() {
                true => Vec::new(),
                false => {
                    let open: BTreeSet<usize> = (state.recorded[state.unjudged_parts()..].iter())
                        .map(|part| part.encoder)
                        .collect();
                    let submitting = device.shared().rebinds.submitting();
                    (state.ended.iter().enumerate())
                        .filter(|(at, _)| open.contains(at))
                        .filter_map(|(at, record)| {
                            let reset = record.taken.rate_map_reset();
                            let Some(replay) = record.replay.as_ref() else {
                                if reset {
                                    unencoded.push(record.unlogged);
                                }
                                return None;
                            };
                            let moved = submitting.rebound(&record.taken) || reset;
                            moved.then(|| (at, Arc::clone(replay)))
                        })
                        .collect()
                }
            }
        };
        for unlogged in unencoded {
            crate::rate_mapped_pass::degrade_unencoded(device, unlogged);
        }
        let handle = CommandBuffer(Arc::clone(self));
        // The encoders encoded again, whose segments the encoding again recorded.
        let mut spliced = Vec::new();
        for (at, replay) in moved {
            // A compute encoder of a command buffer that used command predication, which
            // encodes again exactly, does so in its unit begun anew
            // (`Predication::replaying`).
            let residency = {
                let mut state = lock(&self.state);
                state.replaying_predication = match state.predicated {
                    true => state.predication.replaying(at),
                    false => None,
                };
                state.residency_before(at, replay.kind())
            };
            let run = replay.run(&handle, residency);
            // Its steps' gates listed as the commit lists the first encoding's.
            let run = {
                let mut state = lock(&self.state);
                let state = &mut *state;
                match (run, state.replaying_predication.take()) {
                    (Ok(replayed), Some(mut predication)) => predication
                        .commit(&mut state.uploads, device)
                        .map(|()| (replayed, Some(predication))),
                    (run, _) => run.map(|replayed| (replayed, None)),
                }
            };
            let counts = &device.shared().counts;
            let ((replayed, residency), predication) = match run {
                Ok(replayed) => replayed,
                Err(error) => {
                    counts.reencode_refused();
                    let reset = {
                        let mut state = lock(&self.state);
                        state.ended[at].refused = Some(error.to_string());
                        state.ended[at].taken.rate_map_reset()
                    };
                    if reset {
                        crate::rate_mapped_pass::degrade_unencoded(device, None);
                    }
                    continue;
                }
            };
            // What a refused splice held goes after the lock.
            let spliced_now = {
                let mut state = lock(&self.state);
                let spliced = state.splice_replay(at, replayed, residency);
                if let (Ok(()), Some(predication)) = (&spliced, predication) {
                    state.predication.take_checks(predication);
                }
                spliced
            };
            match spliced_now {
                Ok(()) => {
                    counts.encoder_reencoded();
                    spliced.push(at);
                }
                Err(unspliced) => {
                    counts.reencode_refused();
                    let refusal = "its place holds other parts or work than its encoding again's";
                    let reset = {
                        let mut state = lock(&self.state);
                        state.ended[at].refused = Some(refusal.to_owned());
                        state.ended[at].taken.rate_map_reset()
                    };
                    drop(unspliced);
                    if reset {
                        crate::rate_mapped_pass::degrade_unencoded(device, None);
                    }
                }
            }
        }
        // Each making a spliced segment now waits for wakes the scheduler as it
        // finishes, with no lock held, registered before the scheduler looks at the batches again
        // (the wake at the end of this turn): a making that finished between that look and a
        // later registration would leave the batch held with nothing to wake the scheduler. Its
        // batch waits for them now, so they go before the makings nothing waits for, at the rank
        // the commit asks for the first encoding's (`CommandBuffer::commit`).
        let pending: Vec<Deferred> = {
            let state = lock(&self.state);
            (state.recorded.iter())
                .filter(|part| spliced.contains(&part.encoder))
                .flat_map(|part| part.segments.iter())
                .filter(|segment| segment.steps_after_pipelines())
                .map(|segment| segment.work.clone())
                .collect()
        };
        if !pending.is_empty() {
            let waker = device.shared().scheduler.waker();
            for work in &pending {
                work.when_made(&waker);
                work.hasten(Urgency::Committed);
            }
        }
        // Its work recorded on the GPU alone in encoders not encoded again that named a buffer
        // over host pages that lies elsewhere now, before its first submission (which alone
        // points their resident tables), each segment by its part and its place there, which
        // hold through this turn: only the first submission places segments, and
        // it waits for the turn; a host step, which records one, takes the turn too. What it
        // reaches through its resident set its first submission points it at as it lies then.
        let alone: Vec<((usize, usize), Deferred)> = {
            let state = lock(&self.state);
            match state.submitted {
                true => Vec::new(),
                false => {
                    let submitting = device.shared().rebinds.submitting();
                    let moved = |segment: &Segment| {
                        segment.gpu_alone
                            && (segment.recorded.as_ref())
                                .is_some_and(|done| submitting.rebound(&done.taken))
                    };
                    (state.recorded.iter().enumerate())
                        .filter(|(_, recorded)| !spliced.contains(&recorded.encoder))
                        .flat_map(|(part, recorded)| {
                            let segments = recorded.segments.iter().enumerate();
                            segments
                                .filter(|(_, segment)| moved(segment))
                                .map(move |(at, segment)| ((part, at), segment.work.clone()))
                        })
                        .collect()
                }
            }
        };
        for (place, work) in alone {
            // Its notices were announced as it committed (`replay::is_replaying`), and its
            // commands take what they use without holding it.
            let again = crate::replay::replaying(|| work.record_on_gpu(&handle));
            let counts = &device.shared().counts;
            let Ok(Some((recorded, inputs))) = again else {
                counts.reencode_refused();
                continue;
            };
            let put = lock(&self.state).splice_segment(place, recorded, inputs);
            match put {
                Ok(()) => counts.segment_recorded_again(),
                Err(unspliced) => {
                    counts.reencode_refused();
                    drop(unspliced);
                }
            }
        }
        drop(handle);
        // Its next group's submission may go, unless a rebind marked it meanwhile: that rebind's
        // turn comes next and encodes what moved since.
        let _ = self
            .reencode
            .compare_exchange(marked, 0, Ordering::AcqRel, Ordering::Acquire);
        drop(turn);
        device.shared().scheduler.wake();
    }
}

impl State {
    /// The residency an ended encoder `at` of `kind` started with, as the command buffer's
    /// encoders ended before it leave it (`CommandBuffer::inherited_residency`): none for a blit
    /// encoder, which makes nothing resident.
    pub(super) fn residency_before(&self, at: usize, kind: EncoderKind) -> Residency {
        if kind == EncoderKind::Blit {
            return Residency::default();
        }
        let mut joined = Residency::default();
        for record in self.ended.iter().take(at) {
            if let Some(residency) = &record.residency {
                joined.join_encoder(residency, record.kind == EncoderKind::Compute);
            }
        }
        // The textures made in its heaps since, as the commit's sync made them resident.
        drop(joined.sync_heaps());
        joined.shadows(kind == EncoderKind::Render)
    }

    /// Put `replayed`, ended encoder `at` encoded again, and what it made resident in its place,
    /// before the submission of its event group: each of its parts' recordings, uses and segments
    /// (recorded on the GPU alone), its executions (the later encoders' and segments' places
    /// among them moved by the difference), its keys, and what its commands and its segments'
    /// took (held weakly until that submission). What its first recording retained stays retained,
    /// and so does what its first segments recorded and what their host steps needed; its
    /// `VkCommandBuffer`s go with the pool unsubmitted. Refused, changing nothing and giving
    /// `replayed` back, where its place holds other parts than the encoding again's, or a segment
    /// but one whose pipelines are being made (which goes, the encoding again having left one of
    /// its own in its place) or work its commit recorded on the GPU alone, or its group was
    /// submitted, or the command buffer completed.
    pub(super) fn splice_replay(
        &mut self,
        at: usize,
        replayed: Replayed,
        residency: Option<Residency>,
    ) -> Result<(), Box<(Replayed, Option<Residency>)>> {
        let parts: Vec<usize> = (self.recorded.iter().enumerate())
            .filter(|(_, part)| part.encoder == at)
            .map(|(index, _)| index)
            .collect();
        let replaces = |part: &usize| {
            let recorded = &self.recorded[*part];
            recorded.sparse.is_empty()
                && (recorded.segments.iter())
                    .all(|segment| segment.keeps_pipelines() || segment.gpu_alone)
        };
        let unjudged = self.unjudged_parts();
        let fits = !self.finished()
            && !parts.is_empty()
            && parts.iter().all(|&part| part >= unjudged)
            && parts.len() == replayed.parts.len()
            && parts.iter().all(replaces)
            && !self.unrunnable.contains(&at)
            && self
                .ended
                .get(at)
                .is_some_and(|record| record.replay.is_some());
        if !fits {
            return Err(Box::new((replayed, residency)));
        }
        let Replayed {
            parts: mut replayed_parts,
            held,
            retained,
            executions,
            keys,
            mut taken,
            heaps,
        } = replayed;
        // Held weakly until its group's submission, as the commit held what the first recording
        // and its segments took.
        taken.wait();
        let segments = (replayed_parts.iter_mut())
            .flat_map(|part| part.segments.iter_mut())
            .filter_map(|segment| segment.recorded.as_mut());
        for done in segments {
            done.taken.wait();
        }
        let first = self.ended[at].executions.clone();
        for (&part, replayed) in parts.iter().zip(replayed_parts) {
            let ReplayedPart {
                raws,
                summary,
                segments,
            } = replayed;
            let recorded = &mut self.recorded[part];
            recorded.raws = raws;
            recorded.summary = summary;
            // Its first segments go: the encoding again left a segment of its own in each
            // pending dispatch's place, and its work on the GPU alone in segments of its own.
            // What the commit recorded of one stays retained, never submitted, as the first
            // recording does.
            for segment in recorded.segments.drain(..) {
                if let Some(mut before) = segment.recorded {
                    self.retained.extend(before.retained.into_entry());
                    before.taken.retain_into(&mut self.retained);
                }
                self.retained.push(Box::new((segment.work, segment.inputs)));
            }
            for mut segment in segments {
                segment.executions_at += first.start;
                recorded.segments.push(segment);
            }
        }
        // Its executions in its place; an encoder that met a detached buffer at its first
        // encoding may have recorded fewer.
        let count = executions.len();
        self.executions.splice(first.clone(), executions);
        if count != first.len() {
            let moved = |place: usize| place - first.end + first.start + count;
            for later in self.ended[at + 1..].iter_mut() {
                if later.replay.is_some() {
                    later.executions = moved(later.executions.start)..moved(later.executions.end);
                }
            }
            let last = parts.last().copied().unwrap_or_default();
            let segments = self.recorded[last + 1..]
                .iter_mut()
                .flat_map(|later| later.segments.iter_mut());
            for segment in segments {
                segment.executions_at = moved(segment.executions_at);
            }
        }
        let record = &mut self.ended[at];
        record.executions = first.start..first.start + count;
        record.keys = keys;
        record.held = held;
        record.taken = taken;
        if let Some(residency) = residency {
            if let Some(first) = record.residency.replace(residency) {
                self.retained.push(Box::new(first));
            }
        }
        self.retained.extend(retained.into_entry());
        self.retained.push(Box::new(heaps));
        Ok(())
    }

    /// Put `recorded`, segment `at` of part `part` recorded again on the GPU alone, in its place
    /// before the first submission, with `inputs`, what its host step needs now: what its
    /// commands took held weakly until the first submission, as the commit holds them. What the first
    /// recording retained, its executions and its host step's inputs stay retained; its
    /// `VkCommandBuffer`s go with the pool unsubmitted, and what its commands took, held weakly,
    /// goes. Refused, changing nothing and giving `recorded` and `inputs` back, where the place
    /// holds no such segment, or the command buffer was submitted.
    pub(super) fn splice_segment(
        &mut self,
        (part, at): (usize, usize),
        mut recorded: RecordedSegment,
        inputs: StepInputs,
    ) -> Result<(), Box<(RecordedSegment, StepInputs)>> {
        let fits = !self.submitted
            && (self.recorded.get(part))
                .and_then(|recorded| recorded.segments.get(at))
                .is_some_and(|segment| segment.gpu_alone && segment.recorded.is_some());
        if !fits {
            return Err(Box::new((recorded, inputs)));
        }
        recorded.taken.wait();
        let segment = &mut self.recorded[part].segments[at];
        let first = segment.recorded.replace(recorded);
        let before = segment.inputs.replace(inputs);
        if let Some(first) = first {
            let RecordedSegment {
                retained,
                executions,
                ..
            } = first;
            self.retained.extend(retained.into_entry());
            self.retained.push(Box::new(executions));
        }
        if let Some(before) = before {
            self.retained.push(Box::new(before));
        }
        Ok(())
    }

    /// As it commits: the logs of encoders no rebind encodes again go, for the caller to let go
    /// of without this lock: where an encoder used command predication, those of the compute
    /// encoders their unit of predication reaches past, or that laid a loop or left work out
    /// (`Predication::alone`: a command of theirs encoded again after the commit would run
    /// outside its block, or leave a command of their unit's reading a gate nothing writes); and those of
    /// encoders the commit's failure fails (they run nothing). The keys of buffers over host pages
    /// the encoders whose logs stay named, and the work its commit recorded on the GPU alone,
    /// sorted, each once, and the identities of the rasterization-rate maps its encoders' passes
    /// are over, logged or not (a reset encodes the one again, or announces the other:
    /// `Rebinds::wait_for_replay`).
    pub(super) fn keep_replays(&mut self) -> KeptReplays {
        let mut gone = Vec::new();
        let mut keys = Vec::new();
        let mut maps = Vec::new();
        for (at, record) in self.ended.iter_mut().enumerate() {
            let fails = self
                .failed_at_commit
                .as_ref()
                .is_some_and(|failed| failed.fails(at));
            maps.extend(record.taken.rate_map_id());
            let predicated = self.predicated
                && record.kind == EncoderKind::Compute
                && !self.predication.alone(at);
            if predicated || fails {
                if let Some(log) = record.replay.take() {
                    gone.push(log);
                    record.unlogged = Some(match predicated {
                        true => Unlogged::Predicated,
                        false => Unlogged::Fails,
                    });
                }
                continue;
            }
            if record.replay.is_some() {
                keys.extend(record.taken.named_pins());
            }
        }
        let alone = (self.recorded.iter())
            .flat_map(|part| &part.segments)
            .filter(|segment| segment.gpu_alone)
            .filter_map(|segment| segment.recorded.as_ref());
        for done in alone {
            keys.extend(done.taken.named_pins());
        }
        keys.sort_unstable();
        keys.dedup();
        maps.sort_unstable();
        maps.dedup();
        KeptReplays {
            gone,
            keys: keys.into_boxed_slice(),
            maps: maps.into_boxed_slice(),
        }
    }

    /// As it commits, once its replays are kept ([`State::keep_replays`]): whether an encoder
    /// whose pass took a rasterization-rate map since reset, before the commit, is to be encoded
    /// again before its first submission, its log kept (`rate_mapped_pass.rs`); and why each that
    /// kept no log did not, which draws by the map as it took it, for the caller to announce
    /// without this lock (`rate-map-reset-unencoded`).
    pub(super) fn rate_maps_reset(&self) -> (bool, Vec<Option<Unlogged>>) {
        let mut again = false;
        let mut unencoded = Vec::new();
        for record in &self.ended {
            if !record.taken.rate_map_reset() {
                continue;
            }
            match record.replay {
                Some(_) => again = true,
                None => unencoded.push(record.unlogged),
            }
        }
        (again, unencoded)
    }
}

/// What [`State::keep_replays`] keeps and lets go of.
pub(super) struct KeptReplays {
    /// The logs no rebind encodes again, for the caller to let go of without the lock.
    pub(super) gone: Vec<Arc<Replay>>,
    /// The keys of the buffers over host pages the kept logs' encoders named.
    pub(super) keys: Box<[ResourceKey]>,
    /// The identities of the rasterization-rate maps its encoders' passes are over.
    pub(super) maps: Box<[usize]>,
}
