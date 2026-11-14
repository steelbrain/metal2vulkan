//! A command buffer's groups of compute encoders recorded into one `VkCommandBuffer`
//! (`encoder_fold.rs`): the group its tail holds, which compute encoder joins it, its logs, and
//! the failures that fail a whole group.

use super::{FailedEncoders, Failure, State};
use crate::encoder_fold::{Fold, FoldEnd, FoldOrder, Joining, Stashed};
use crate::replay::Replay;
use crate::resource::ResourceKey;
use crate::tracker::Summary;
use crate::work_counts::Counters;
use std::collections::BTreeSet;
use std::ops::Range;

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
    /// failure and a rebind take together (`widen_failure`, `reencode.rs`), and the logs its
    /// members did not keep are kept where one member kept its own and the command buffer used no
    /// command predication, so the group can be encoded again; else they come back, for the
    /// caller to give back to the device's spares without this lock (`replay.rs`).
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
            return stashed.into_iter().map(|stashed| stashed.log).collect();
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

    /// `failure`, its failed encoders widened to every encoder of each group one of them lies in
    /// (`encoder_fold.rs`'s **Failed**): a group's encoders record into one `VkCommandBuffer`, which
    /// runs all of them or none. Unchanged where it fails no group in part.
    pub(super) fn widen_failure(&self, mut failure: Failure) -> Failure {
        let group = |encoder: usize| -> Option<Range<usize>> {
            self.ended
                .get(encoder)
                .and_then(|record| record.group.clone())
        };
        failure.encoders = match failure.encoders {
            FailedEncoders::From(first) => match group(first) {
                Some(group) => FailedEncoders::From(group.start),
                None => FailedEncoders::From(first),
            },
            FailedEncoders::Set(set) => {
                let widened: BTreeSet<usize> = (set.iter())
                    .flat_map(|&encoder| group(encoder).unwrap_or(encoder..encoder + 1))
                    .collect();
                FailedEncoders::Set(widened)
            }
        };
        failure
    }
}
