//! A command buffer's groups of encoders recorded into one `VkCommandBuffer`
//! (`encoder_fold.rs`): the group its tail holds, which encoder joins it, its logs, and the
//! failures and left out work that split a group.

use super::{FailedEncoders, FailedSignals, Failure, State};
use crate::encoder::Overwrite;
use crate::encoder_fold::{Fold, FoldEnd, FoldOrder, Joining, Stashed};
use crate::replay::Replay;
use crate::resource::ResourceKey;
use crate::tracker::Summary;
use crate::used_resources::EncoderKind;
use crate::work_counts::Counters;
use std::collections::BTreeSet;
use std::ops::Range;

impl State {
    /// The group the encoder opening now joins, asking `joining`, where its
    /// command buffer's tail holds one it may join (`encoder_fold.rs`'s **Where**): taken, with
    /// the tail, which the encoder records into, and, where it is the group's second encoder, the
    /// group's order built from its first part's summary (`Fold::order`). `None` where it joins
    /// none, the group left in place: the encoder's open closes it once it opened
    /// ([`close_fold`](Self::close_fold)). On a device that `splits_at_aliases`, a render
    /// encoder joins no group whose earlier encoders wrote bytes: its pass records no barrier, nor
    /// a new `VkCommandBuffer`, before a use of an image over them (`alias_split.rs`).
    pub(crate) fn join_fold(
        &mut self,
        joining: Joining,
        (trail, splits_at_aliases): (bool, bool),
    ) -> Option<Box<Fold>> {
        let fold = self.fold.as_ref()?;
        let joins = fold.admits(joining)
            && self.tail == Some(fold.tail)
            && self.batches.all().len() == fold.batches
            && !self.batches.signals_pending()
            && !self.predicated
            && self.protection.options() == 0
            && !trail;
        if !joins {
            return None;
        }
        // Its second encoder orders after its first part, as that part's end left it.
        let order = match fold.members.len() {
            1 => Some(FoldOrder::after(&self.recorded.get(fold.leader)?.summary)),
            _ => None,
        };
        let render = matches!(joining, Joining::Render { .. });
        let wrote_bytes = || (order.as_ref().unwrap_or(&fold.order).byte_writes().next()).is_some();
        if splits_at_aliases && render && wrote_bytes() {
            return None;
        }
        self.tail = None;
        let mut fold = self.fold.take()?;
        if let Some(order) = order {
            fold.order = order;
        }
        Some(fold)
    }

    /// An encoder that asked a group ended (`end`), its last `VkCommandBuffer` the tail: it goes on in the
    /// group it joined where the tail is still that group's and it leaves nothing that needs a
    /// boundary, else the group closes and it begins one of its own where it may lead one (its
    /// one part leaves nothing that needs a boundary, the command buffer used no command
    /// predication, and its protection options are 0). Its log, which it did not keep, goes with
    /// the group it ends in; the logs that come back are the caller's to give back without this
    /// lock ([`close_fold`](Self::close_fold)).
    pub(crate) fn fold_ended(&mut self, end: FoldEnd, counts: &Counters) -> Vec<Replay> {
        let FoldEnd {
            joining,
            fold,
            in_tail,
            encoder,
            plain,
            summary,
            log,
        } = end;
        let may_lead = plain && !self.predicated && self.protection.options() == 0;
        let leads = joining.filter(|_| may_lead);
        let mut log = log.map(|(log, executions)| Stashed {
            encoder,
            log,
            executions,
        });
        let mut unkept = Vec::new();
        if let Some(mut fold) = fold {
            fold.members.end = encoder + 1;
            if let Some(record) = self.ended.get_mut(encoder) {
                record.in_tail = in_tail;
            }
            fold.stashed.extend(log.take());
            let goes_on = self.tail == Some(fold.tail);
            if let (Some(joining), Some(summary), true) = (leads, &summary, goes_on) {
                fold.went_on(joining, summary);
                fold.batches = self.batches.all().len();
                self.fold = Some(fold);
                return unkept;
            }
            self.fold = Some(fold);
            unkept = self.close_fold(counts);
        }
        match (leads, self.tail) {
            (Some(joining), Some(tail)) => {
                // The tail is its last `VkCommandBuffer`.
                if let Some(record) = self.ended.get_mut(encoder) {
                    record.in_tail = true;
                }
                let (leader, batches) = (self.recorded.len() - 1, self.batches.all().len());
                // Its stash takes the room the command buffer's lists keep (`record_storage.rs`).
                let mut stashed = self.stash_room.take().unwrap_or_default();
                stashed.extend(log);
                self.fold = Some(Box::new(Fold::begun(
                    tail,
                    (leader, encoder),
                    (batches, joining),
                    stashed,
                )));
            }
            _ => unkept.extend(log.map(|stashed| stashed.log)),
        }
        unkept
    }

    /// Join `worked`, the uses and fence waits of an encoder whose commands went into the
    /// `VkCommandBuffer` of the group it joined, to the summary of that group's first part,
    /// `leader`, which the batch's submission plans the group's barrier and waits by; and
    /// `overwrites`, its whole writes that stand as that part's would (`Fold::overwrites_join`),
    /// to that part's, which the submission judges them by.
    pub(crate) fn join_leader(
        &mut self,
        leader: usize,
        worked: Summary<ResourceKey>,
        overwrites: Vec<Overwrite>,
    ) {
        if let Some(part) = self.recorded.get_mut(leader) {
            part.summary.join(worked);
            part.overwrites.extend(overwrites);
        }
    }

    /// Close the group the command buffer's tail holds, where it holds one: no encoder joins it
    /// any more. Where it holds more than one encoder, each notes the group's encoders, which a
    /// failure and a rebind take together (`split_failure`, `reencode.rs`), and the logs its
    /// members did not keep are kept where one member kept its own and the command buffer used no
    /// command predication, so a rebind can encode the group again; else they are held until the
    /// commit, whose failure may split the group ([`split_failure`](Self::split_failure)). A lone
    /// encoder's comes back, for the caller to give back to the device's spares without this lock
    /// (`replay.rs`). A group of more gives the room of its stash back, emptied, for the next
    /// group's (`record_storage.rs`).
    pub(crate) fn close_fold(&mut self, counts: &Counters) -> Vec<Replay> {
        let Some(fold) = self.fold.take() else {
            return Vec::new();
        };
        let Fold {
            members,
            mut stashed,
            ..
        } = *fold;
        if members.len() < 2 {
            return stashed.into_iter().map(|stashed| stashed.log).collect();
        }
        for record in &mut self.ended[members.clone()] {
            record.group = Some(members.clone());
        }
        let logged = (self.ended[members.clone()].iter()).any(|record| record.replay.is_some());
        if !logged || self.predicated {
            self.held_logs.append(&mut stashed);
            self.stash_room = Some(stashed);
            return Vec::new();
        }
        let mut unkept = Vec::new();
        for Stashed {
            encoder,
            log,
            executions,
        } in stashed.drain(..)
        {
            let bytes = log.bytes();
            match self.keep_replay(encoder, Some(log), executions) {
                Some(log) => unkept.push(log),
                None => counts.replay_log_kept(bytes),
            }
        }
        self.stash_room = Some(stashed);
        unkept
    }

    /// At the commit, `failure` and the work a command-predication record left out
    /// (`State::left_out_from`) as the groups of encoders recorded into one `VkCommandBuffer`
    /// take them (`encoder_fold.rs`'s **Failed**): a group they fail or leave out in part is
    /// split, each encoder of it that runs to be encoded again from its log, kept now, into
    /// `VkCommandBuffer`s of its own before the first submission (`reencode.rs`), and the group's
    /// shared one never submitted; where one of them cannot be encoded again
    /// ([`encodes_again`](Self::encodes_again)), the failure fails that whole group instead (a
    /// failure that fails those encoders alone, with no error, where the commit has none). A
    /// command buffer with a submission check (`checks.rs`) splits so each group whose shared
    /// `VkCommandBuffer` holds commands of more than one of Metal's submissions
    /// ([`in_tail`](Self::in_tail)), those commands' encoders encoded again apart, so a check
    /// fails one submission's encoders alone; one of them that cannot be encoded again exactly
    /// (no log, or work of a segment's, whose encoding again records it otherwise) leaves the group
    /// whole, and a check that fails one of those submissions fails each of them
    /// (`State::fail_submission`). With the failure, the encoders that run of each group split,
    /// the units to encode again. The logs held for the commit that no split takes stay held,
    /// for the command buffer's release to give back to the device's spares
    /// ([`unsplit_logs`](Self::unsplit_logs)).
    pub(super) fn split_failure(
        &mut self,
        failure: Option<Failure>,
    ) -> (Option<Failure>, Vec<Vec<usize>>) {
        let mut units = Vec::new();
        let left_out = self.left_out_from;
        let checked = self.check.is_some();
        if failure.is_none() && left_out.is_none() && !checked {
            return (failure, units);
        }
        let groups: BTreeSet<(usize, usize)> = (self.ended.iter())
            .filter_map(|record| record.group.as_ref())
            .map(|group| (group.start, group.end))
            .collect();
        // Each ended encoder's submission among Metal's, where a check is set.
        let mut submission = Vec::new();
        if checked && !groups.is_empty() {
            for (at, (_, encoders)) in self.metal_submissions().into_iter().enumerate() {
                submission.extend(encoders.map(|_| at));
            }
        }
        let mut whole = BTreeSet::new();
        for (start, end) in groups {
            let runs = |encoder: &usize| {
                Failure::runs(failure.as_ref(), *encoder)
                    && left_out.is_none_or(|first| *encoder < first)
            };
            let running: Vec<usize> = (start..end).filter(runs).collect();
            if running.is_empty() {
                continue;
            }
            if running.len() < end - start {
                match running.iter().all(|&encoder| self.encodes_again(encoder)) {
                    true => units.push(self.split_apart(start..end, running)),
                    false => whole.extend(running),
                }
                continue;
            }
            // Nothing fails it: where a check may fail one of the submissions whose commands its
            // shared `VkCommandBuffer` holds and not another, those commands' encoders are
            // encoded again apart, where each is so exactly, its work no segment's.
            let tail = self.in_tail(start..end);
            let first = tail.first().and_then(|&encoder| submission.get(encoder));
            let last = tail.last().and_then(|&encoder| submission.get(encoder));
            let exact = |encoder: &usize| {
                self.encodes_again(*encoder)
                    && (self.recorded.iter())
                        .filter(|part| part.encoder == *encoder)
                        .all(|part| part.segments.is_empty())
            };
            if checked && first != last && tail.iter().all(exact) {
                units.push(self.split_apart(start..end, tail));
            }
        }
        if whole.is_empty() {
            return (failure, units);
        }
        let ended = self.ended.len();
        let failure = match failure {
            Some(mut failure) => {
                failure.encoders = failure.encoders.with(&whole, ended);
                failure
            }
            None => Failure {
                encoders: FailedEncoders::Set(whole),
                signals: FailedSignals::Every,
                error: None,
            },
        };
        (Some(failure), units)
    }

    /// `encoders` of `group`, a group recorded into one `VkCommandBuffer`, to be encoded again
    /// apart, their logs kept: the group's encoders share no `VkCommandBuffer` any more, the
    /// others' commands lying in `VkCommandBuffer`s of their own. The unit to encode again.
    fn split_apart(&mut self, group: Range<usize>, encoders: Vec<usize>) -> Vec<usize> {
        self.keep_held_logs(&encoders);
        for record in &mut self.ended[group] {
            record.group = None;
        }
        encoders
    }

    /// The encoders of `group`, a group recorded into one `VkCommandBuffer`, whose commands lie
    /// there, some or all (`EncoderRecord::in_tail`): its first, and each later one a command of
    /// which recorded there.
    pub(super) fn in_tail(&self, group: Range<usize>) -> Vec<usize> {
        group
            .filter(|&encoder| self.ended.get(encoder).is_some_and(|record| record.in_tail))
            .collect()
    }

    /// Whether ended encoder `encoder`, of a group recorded into one `VkCommandBuffer`, may be
    /// encoded again apart: it does as its command buffer's command predication holds it
    /// ([`encodes_again_apart`](Self::encodes_again_apart)), and its log is kept, or held for the
    /// commit.
    pub(super) fn encodes_again(&self, encoder: usize) -> bool {
        self.encodes_again_apart(encoder)
            && (self.ended[encoder].replay.is_some()
                || self.held_logs.iter().any(|held| held.encoder == encoder))
    }

    /// Keep the logs held for the commit of `encoders`, each its own, for an encoding again.
    pub(super) fn keep_held_logs(&mut self, encoders: &[usize]) {
        for &encoder in encoders {
            let Some(at) = (self.held_logs.iter()).position(|held| held.encoder == encoder) else {
                continue;
            };
            let Stashed {
                log, executions, ..
            } = self.held_logs.swap_remove(at);
            if let Some(log) = self.keep_replay(encoder, Some(log), executions.clone()) {
                self.held_logs.push(Stashed {
                    encoder,
                    log,
                    executions,
                });
            }
        }
    }

    /// Whether ended encoder `encoder`, of a group recorded into one `VkCommandBuffer`, encodes
    /// again apart as the command buffer's command predication holds it: every one where it used
    /// none; a compute encoder whose unit nothing reached into or out of (`Predication::alone`);
    /// a blit encoder, which belongs to no unit (`predication.rs`).
    pub(super) fn encodes_again_apart(&self, encoder: usize) -> bool {
        !self.predicated
            || (self.ended.get(encoder)).is_some_and(|record| record.kind != EncoderKind::Compute)
            || self.predication.alone(encoder)
    }

    /// The logs held for the commit that no split took (`split_failure`): for the command
    /// buffer's release to give back to the device's spares without this lock, on its queue's
    /// handler thread, as it gives back the logs its encoders kept (`replay.rs`).
    pub(super) fn unsplit_logs(&mut self) -> Vec<Replay> {
        self.held_logs.drain(..).map(|held| held.log).collect()
    }

    /// `encoders`, a group's encoders the commit split that run (`split_failure`), whose
    /// encoding again failed: they run nothing after all, as the failure's encoders do, the
    /// shared `VkCommandBuffer` among them never submitted (`encoder_fold.rs`'s **Failed**).
    pub(super) fn fail_unsplit(&mut self, encoders: &[usize]) {
        let failed: BTreeSet<usize> = encoders.iter().copied().collect();
        let State {
            recorded,
            retained,
            ended,
            ..
        } = self;
        for part in recorded
            .iter_mut()
            .filter(|part| failed.contains(&part.encoder))
        {
            State::fail_part(part, ended, retained);
        }
        let ended = self.ended.len();
        self.failed_at_commit = Some(match self.failed_at_commit.take() {
            Some(before) => before.with(&failed, ended),
            None => FailedEncoders::Set(failed),
        });
    }
}
