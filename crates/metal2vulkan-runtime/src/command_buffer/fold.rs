//! A command buffer's groups of compute encoders recorded into one `VkCommandBuffer`
//! (`encoder_fold.rs`): the group its tail holds, which compute encoder joins it, its logs, and
//! the failures that split a group.

use super::{FailedEncoders, Failure, State};
use crate::encoder_fold::{Fold, FoldEnd, FoldOrder, Joining, Stashed};
use crate::replay::Replay;
use crate::resource::ResourceKey;
use crate::tracker::Summary;
use crate::work_counts::Counters;
use std::collections::BTreeSet;

impl State {
    /// The group the compute encoder opening now joins, asking `joining`, where its command
    /// buffer's tail holds one it may join (`encoder_fold.rs`'s **Where**): taken, with the tail,
    /// which the encoder records into. `None` where it joins none, the group left in place: the
    /// encoder's open closes it once it opened ([`close_fold`](Self::close_fold)).
    pub(crate) fn join_fold(&mut self, joining: Joining, trail: bool) -> Option<Box<Fold>> {
        let fold = self.fold.as_ref()?;
        let joins = fold.joining == joining
            && self.tail == Some(fold.tail)
            && self.batches.all().len() == fold.batches
            && !self.batches.signals_pending()
            && !self.predicated
            && self.protection.options() == 0
            && !trail;
        if !joins {
            return None;
        }
        self.tail = None;
        self.fold.take()
    }

    /// A compute encoder ended (`end`), its last `VkCommandBuffer` the tail: it goes on in the
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
            encoder,
            summary,
            log,
        } = end;
        let may_lead = !self.predicated && self.protection.options() == 0;
        let leads = joining.zip(summary).filter(|_| may_lead);
        let mut log = log.map(|(log, executions)| Stashed {
            encoder,
            log,
            executions,
        });
        let mut unkept = Vec::new();
        if let Some(mut fold) = fold {
            fold.members.end = encoder + 1;
            fold.stashed.extend(log.take());
            let goes_on = self.tail == Some(fold.tail);
            if let (Some((_, summary)), true) = (&leads, goes_on) {
                fold.order.join(summary);
                fold.batches = self.batches.all().len();
                self.fold = Some(fold);
                return unkept;
            }
            self.fold = Some(fold);
            unkept = self.close_fold(counts);
        }
        match (leads, self.tail) {
            (Some((joining, summary)), Some(tail)) => {
                self.fold = Some(Box::new(Fold {
                    tail,
                    leader: self.recorded.len() - 1,
                    members: encoder..encoder + 1,
                    batches: self.batches.all().len(),
                    joining,
                    order: FoldOrder::after(&summary),
                    stashed: log.into_iter().collect(),
                }));
            }
            _ => unkept.extend(log.map(|stashed| stashed.log)),
        }
        unkept
    }

    /// Join `worked`, the uses and fence waits of a compute encoder whose commands went into the
    /// `VkCommandBuffer` of the group it joined, to the summary of that group's first part,
    /// `leader`, which the batch's submission plans the group's barrier and waits by.
    pub(crate) fn join_leader(&mut self, leader: usize, worked: Summary<ResourceKey>) {
        if let Some(part) = self.recorded.get_mut(leader) {
            part.summary.join(worked);
        }
    }

    /// Close the group the command buffer's tail holds, where it holds one: no encoder joins it
    /// any more. Where it holds more than one encoder, each notes the group's encoders, which a
    /// failure and a rebind take together (`split_failure`, `reencode.rs`), and the logs its
    /// members did not keep are kept where one member kept its own and the command buffer used no
    /// command predication, so a rebind can encode the group again; else they are held until the
    /// commit, whose failure may split the group ([`split_failure`](Self::split_failure)). A lone
    /// encoder's comes back, for the caller to give back to the device's spares without this lock
    /// (`replay.rs`).
    pub(crate) fn close_fold(&mut self, counts: &Counters) -> Vec<Replay> {
        let Some(fold) = self.fold.take() else {
            return Vec::new();
        };
        let Fold {
            members, stashed, ..
        } = *fold;
        if members.len() < 2 {
            return stashed.into_iter().map(|stashed| stashed.log).collect();
        }
        for record in &mut self.ended[members.clone()] {
            record.group = Some(members.clone());
        }
        let logged = (self.ended[members.clone()].iter()).any(|record| record.replay.is_some());
        if !logged || self.predicated {
            self.held_logs.extend(stashed);
            return Vec::new();
        }
        let mut unkept = Vec::new();
        for Stashed {
            encoder,
            log,
            executions,
        } in stashed
        {
            let bytes = log.bytes();
            match self.keep_replay(encoder, Some(log), executions) {
                Some(log) => unkept.push(log),
                None => counts.replay_log_kept(bytes),
            }
        }
        unkept
    }

    /// At the commit, `failure` as the groups of compute encoders recorded into one
    /// `VkCommandBuffer` take it (`encoder_fold.rs`'s **Failed**): a group it fails in part is
    /// split, each encoder of it that runs to be encoded again from its log, kept now, into
    /// `VkCommandBuffer`s of its own before the first submission (`reencode.rs`), and the group's
    /// shared one never submitted; where one of them cannot be encoded again (its unit of command
    /// predication laid a loop or left work out, `Predication::alone`), the failure fails that
    /// whole group instead. With
    /// `failure`, the encoders that run of each group split, the units to encode again. The logs
    /// held for the commit that no split takes come back, for the caller to give back to the
    /// device's spares without this lock.
    pub(super) fn split_failure(
        &mut self,
        failure: Option<Failure>,
    ) -> (Option<Failure>, Vec<Vec<usize>>, Vec<Replay>) {
        let mut units = Vec::new();
        let Some(mut failure) = failure else {
            let held = self.held_logs.drain(..).map(|held| held.log).collect();
            return (None, units, held);
        };
        let groups: BTreeSet<(usize, usize)> = (self.ended.iter())
            .filter_map(|record| record.group.as_ref())
            .map(|group| (group.start, group.end))
            .collect();
        let mut whole = BTreeSet::new();
        for (start, end) in groups {
            let fails = |encoder: &usize| failure.encoders.fails(*encoder);
            let running: Vec<usize> = (start..end).filter(|encoder| !fails(encoder)).collect();
            if running.is_empty() || running.len() == end - start {
                continue;
            }
            let logged = running.iter().all(|&encoder| {
                (!self.predicated || self.predication.alone(encoder))
                    && (self.ended[encoder].replay.is_some()
                        || self.held_logs.iter().any(|held| held.encoder == encoder))
            });
            if !logged {
                whole.extend(start..end);
                continue;
            }
            for &encoder in &running {
                let Some(at) = (self.held_logs.iter()).position(|held| held.encoder == encoder)
                else {
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
            // Encoded again apart, its encoders share no `VkCommandBuffer` any more.
            for record in &mut self.ended[start..end] {
                record.group = None;
            }
            units.push(running);
        }
        if !whole.is_empty() {
            failure.encoders = failure.encoders.with(&whole, self.ended.len());
        }
        let held = self.held_logs.drain(..).map(|held| held.log).collect();
        (Some(failure), units, held)
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
