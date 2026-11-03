//! The scheduler's order, as a pure structure: committed batches in one first-in-first-out line per
//! logical queue, and the choice of the batch to submit next.
//!
//! A logical queue runs its command buffers in commit order, and a command buffer its batches in
//! encoded order, so only the head of each line may be submitted. Among the heads that are ready
//! (each of their waits satisfied by work already submitted, or by the host), the one committed
//! first goes next. A head that is not ready holds back its own line, and the heads of other lines
//! committed after it that it conflicts with:
//!
//! - **Tracked order across queues is commit order.** Metal tracks a resource across queues in
//!   the order it schedules the command buffers, which is their commit order unless one waits
//!   for an event (its tracking across queues). The runtime's tracker plans each batch as it is
//!   submitted, so a batch held for a reason of the runtime's own (its host step, its submission
//!   checks) must not let a later commit's batch that conflicts with it (a tracked resource's range
//!   that one of them writes, or a fence) go before it: such a head is [`behind`](Order::behind)
//!   it, and waits.
//! - **Which batches keep their place.** Every batch of a line up to the first that waits for
//!   an event value not yet reached by a submitted signal or the host, or for a binary
//!   semaphore whose signal is not yet submitted. That one and those after it wait for the
//!   program's signal, as Metal schedules them after it, so a later commit may go first: Metal
//!   follows the event.
//! - **No deadlock.** A head waits behind batches committed before it that keep their place
//!   only; the first of those waits for no batch but its line's earlier ones, or for the runtime
//!   (a host step's gate or pipelines, its checks), which never waits for a later commit.
//!
//! Since a batch is submitted only once every wait it has is satisfied by submitted work, the
//! batches on a `VkQueue` never wait for a signal not yet submitted, and no `VkQueue` stalls on
//! one: many logical queues on one `VkQueue` cannot deadlock through the order they wait for each
//! other in.
//!
//! **Found as asked**. Whether a head waits behind an earlier batch is answered for the heads that
//! would otherwise go ([`is_behind`](Order::is_behind)), not for every head on every pass:
//! [`next`](Order::next) asks it of the ready heads in commit order until one goes, and the
//! scheduler of the heads whose host step or checks would otherwise start. Each line's batches that
//! keep their place and each footprint are taken once a pass ([`Behind`]), as far as an asking head
//! needs them. A pass over many lines whose heads wait for events, or go in commit order, so takes
//! no footprint at all: it was every head against every earlier batch of every other line, each
//! pass (16385 command buffers in flight over 261 queues took about 55 s to commit and did not
//! drain within 60 s).
//!
//! **Kept between passes.** A program that commits one command buffer at a time empties its
//! queue's line at each submission and makes it again at its next commit: an emptied line's
//! storage is kept for the next line made ([`SPARE_LINES`] of them, each small), as is the list
//! [`next`](Order::next) orders the heads in.

use std::collections::{BTreeMap, VecDeque};

/// The most emptied lines an order keeps for lines made later, and the most batches one of them
/// may have room for.
const SPARE_LINES: usize = 4;
const SPARE_ROOM: usize = 64;

/// The committed batches not yet submitted, by logical queue.
#[derive(Debug)]
pub(crate) struct Order<B> {
    /// Each logical queue's line, by the queue's identity, while it holds a batch.
    lines: Vec<(u64, VecDeque<(u64, B)>)>,
    /// The commit sequence number the next batch takes.
    next: u64,
    /// Emptied lines, kept for lines made later (the module docs).
    spare: Vec<VecDeque<(u64, B)>>,
    /// The heads [`next`](Order::next) orders, by commit sequence number and line.
    heads: Vec<(u64, usize)>,
}

impl<B> Default for Order<B> {
    fn default() -> Order<B> {
        Order {
            lines: Vec::new(),
            next: 0,
            spare: Vec::new(),
            heads: Vec::new(),
        }
    }
}

impl<B> Order<B> {
    /// A new line's storage: an emptied one where one is kept.
    fn new_line(&mut self) -> VecDeque<(u64, B)> {
        self.spare.pop().unwrap_or_default()
    }

    /// Line `at`, empty, goes: its storage is kept for a later line while few are, and it has
    /// little room.
    fn remove_line(&mut self, at: usize) {
        let (_, line) = self.lines.swap_remove(at);
        if self.spare.len() < SPARE_LINES && line.capacity() <= SPARE_ROOM {
            self.spare.push(line);
        }
    }

    /// Whether two lines or more hold batches: with one, no head waits behind another line's
    /// batch ([`is_behind`](Self::is_behind)).
    pub(crate) fn several_lines(&self) -> bool {
        self.lines.len() > 1
    }

    /// Append the batches of one command buffer committed now on logical queue `queue`.
    pub(crate) fn push(&mut self, queue: u64, batches: impl IntoIterator<Item = B>) {
        let at = match self.lines.iter().position(|(line, _)| *line == queue) {
            Some(at) => at,
            None => {
                let line = self.new_line();
                self.lines.push((queue, line));
                self.lines.len() - 1
            }
        };
        for batch in batches {
            self.lines[at].1.push_back((self.next, batch));
            self.next += 1;
        }
    }

    /// Whether the line head of commit sequence number `sequence` waits behind a batch committed
    /// before it on another line: one that keeps its place there (`keeps` holds for it and for
    /// every batch before it on its line), whose footprint (`footprint`) and the head's `ordered`
    /// says run in commit order. `memo` keeps, for one pass over an unchanged order, each answer,
    /// each line's batches found to keep their place, and each footprint taken. False for a
    /// sequence number no head has.
    pub(crate) fn is_behind<F>(
        &self,
        sequence: u64,
        memo: &mut Behind<F>,
        (keeps, footprint, ordered): (
            &mut impl FnMut(&B) -> bool,
            &mut impl FnMut(&B) -> F,
            &impl Fn(&F, &F) -> bool,
        ),
    ) -> bool {
        if let Some(&answer) = memo.answers.get(&sequence) {
            return answer;
        }
        let answer = self.find_behind(sequence, memo, (keeps, footprint, ordered));
        memo.answers.insert(sequence, answer);
        answer
    }

    /// [`is_behind`](Self::is_behind)'s answer, found.
    fn find_behind<F>(
        &self,
        sequence: u64,
        memo: &mut Behind<F>,
        (keeps, footprint, ordered): (
            &mut impl FnMut(&B) -> bool,
            &mut impl FnMut(&B) -> F,
            &impl Fn(&F, &F) -> bool,
        ),
    ) -> bool {
        if self.lines.len() < 2 {
            return false;
        }
        let Some((at, head)) = self.lines.iter().enumerate().find_map(|(at, (_, line))| {
            line.front()
                .filter(|(first, _)| *first == sequence)
                .map(|(_, head)| (at, head))
        }) else {
            return false;
        };
        if memo.kept.len() < self.lines.len() {
            memo.kept.resize(self.lines.len(), Kept::default());
        }
        for (other, (_, earlier)) in self.lines.iter().enumerate() {
            if other == at || earlier.front().is_none_or(|(first, _)| *first > sequence) {
                continue;
            }
            for (index, (first, batch)) in earlier.iter().enumerate() {
                if *first > sequence {
                    break;
                }
                // Whether it and every batch before it on its line keep their place, each asked
                // once a pass.
                let kept = &mut memo.kept[other];
                if index >= kept.scanned {
                    if kept.stopped || !keeps(batch) {
                        kept.stopped = true;
                        break;
                    }
                    kept.scanned = index + 1;
                }
                memo.footprints
                    .entry(*first)
                    .or_insert_with(|| footprint(batch));
                memo.footprints
                    .entry(sequence)
                    .or_insert_with(|| footprint(head));
                if ordered(&memo.footprints[first], &memo.footprints[&sequence]) {
                    return true;
                }
            }
        }
        false
    }

    /// The commit sequence numbers of the line heads `select` holds for.
    pub(crate) fn heads_where(&self, mut select: impl FnMut(&B) -> bool) -> Vec<u64> {
        self.lines
            .iter()
            .filter_map(|(_, line)| line.front())
            .filter(|(_, head)| select(head))
            .map(|(sequence, _)| *sequence)
            .collect()
    }

    /// Put `batch`, which [`next`](Self::next) took from `queue`'s line with commit sequence
    /// number `sequence`, back at the line's head: the order is as it was before it was taken.
    pub(crate) fn put_back(&mut self, queue: u64, sequence: u64, batch: B) {
        match self.lines.iter_mut().find(|(line, _)| *line == queue) {
            Some((_, line)) => line.push_front((sequence, batch)),
            None => {
                let mut line = self.new_line();
                line.push_back((sequence, batch));
                self.lines.push((queue, line));
            }
        }
    }

    /// Take the batch to submit next: of the line heads that `ready` prepares and that are not
    /// `behind` (asked of those, in commit order, until one goes), the one committed first, with
    /// its line's queue, its commit sequence number (for [`put_back`](Self::put_back)) and what
    /// `ready` made of it. `None` when no head is ready.
    ///
    /// `ready` takes each head it looks at mutably, so a caller may note on a head it holds back
    /// why it did (`scheduler/mod.rs`'s [`HeldFor`](crate::scheduler::HeldFor)); it is asked of
    /// each head in commit order until one is both ready and not `behind`.
    pub(crate) fn next<T>(
        &mut self,
        mut behind: impl FnMut(&Self, u64) -> bool,
        mut ready: impl FnMut(&mut B) -> Option<T>,
    ) -> Option<(u64, u64, B, T)> {
        let Order { lines, heads, .. } = self;
        heads.clear();
        heads.extend(
            (lines.iter().enumerate())
                .filter_map(|(at, (_, line))| line.front().map(|(sequence, _)| (*sequence, at))),
        );
        heads.sort_unstable();
        let mut found = None;
        for index in 0..self.heads.len() {
            let (sequence, at) = self.heads[index];
            let Some((_, head)) = self.lines[at].1.front_mut() else {
                continue;
            };
            let Some(prepared) = ready(head) else {
                continue;
            };
            if behind(self, sequence) {
                continue;
            }
            found = Some((at, prepared));
            break;
        }
        let (at, prepared) = found?;
        let queue = self.lines[at].0;
        let (sequence, batch) = self.lines[at].1.pop_front()?;
        if self.lines[at].1.is_empty() {
            self.remove_line(at);
        }
        Some((queue, sequence, batch, prepared))
    }

    /// Take the line head committed first of all, where `ready` prepares it: with its line's
    /// queue, its commit sequence number (for [`put_back`](Self::put_back)) and what `ready` made
    /// of it. `None` when no batch is held, or when `ready` holds that one head back.
    ///
    /// It passes no head over, so a caller that takes batches with this alone submits them in
    /// commit order and leaves no batch that keeps its place behind one committed after it:
    /// [`is_behind`](Self::is_behind) has nothing to answer for it. That is what
    /// [`next`](Self::next) needs it for, which takes the first *ready* head and so may go
    /// before a head that is not.
    pub(crate) fn take_first<T>(
        &mut self,
        ready: impl FnOnce(&B) -> Option<T>,
    ) -> Option<(u64, u64, B, T)> {
        let (_, at) = (self.lines.iter().enumerate())
            .filter_map(|(at, (_, line))| line.front().map(|(sequence, _)| (*sequence, at)))
            .min()?;
        let prepared = ready(&self.lines[at].1.front()?.1)?;
        let queue = self.lines[at].0;
        let (sequence, batch) = self.lines[at].1.pop_front()?;
        if self.lines[at].1.is_empty() {
            self.remove_line(at);
        }
        Some((queue, sequence, batch, prepared))
    }

    /// The line head committed first of all, the one [`take_first`](Self::take_first) would take,
    /// to update in place.
    pub(crate) fn first_mut(&mut self) -> Option<&mut B> {
        let (_, at) = (self.lines.iter().enumerate())
            .filter_map(|(at, (_, line))| line.front().map(|(sequence, _)| (*sequence, at)))
            .min()?;
        self.lines[at].1.front_mut().map(|(_, head)| head)
    }

    /// The line head of commit sequence number `sequence`, to update in place; `None` where no
    /// head has it ([`Behind`] holds the numbers one pass asked about).
    pub(crate) fn head_mut_at(&mut self, sequence: u64) -> Option<&mut B> {
        (self.lines.iter_mut()).find_map(|(_, line)| {
            line.front_mut()
                .filter(|(first, _)| *first == sequence)
                .map(|(_, head)| head)
        })
    }

    /// Each line's head not `behind`, to update in place.
    pub(crate) fn heads_mut<'a>(
        &'a mut self,
        behind: &'a [u64],
    ) -> impl Iterator<Item = &'a mut B> + 'a {
        self.lines.iter_mut().filter_map(move |(_, line)| {
            line.front_mut()
                .filter(|(sequence, _)| !behind.contains(sequence))
                .map(|(_, head)| head)
        })
    }

    /// Each line's head not `behind`, then the batches after it, at most `count` of them, up to
    /// the first committed after a batch another line holds, to update in place: `visit` takes
    /// each, told whether it is the head, and answers whether to go on along its line. No batch
    /// it reaches waits behind another line's ([`is_behind`](Self::is_behind)): every batch
    /// committed before it is its line's.
    pub(crate) fn leading_mut(
        &mut self,
        behind: &[u64],
        count: usize,
        mut visit: impl FnMut(&mut B, bool) -> bool,
    ) {
        let firsts: Vec<u64> = (self.lines.iter())
            .map(|(_, line)| line.front().map_or(u64::MAX, |(sequence, _)| *sequence))
            .collect();
        for (at, (_, line)) in self.lines.iter_mut().enumerate() {
            if line
                .front()
                .is_none_or(|(sequence, _)| behind.contains(sequence))
            {
                continue;
            }
            let before = (firsts.iter().enumerate())
                .filter(|&(other, _)| other != at)
                .map(|(_, &first)| first)
                .min()
                .unwrap_or(u64::MAX);
            for (position, (sequence, batch)) in line.iter_mut().take(1 + count).enumerate() {
                if position > 0 && *sequence > before {
                    break;
                }
                if !visit(batch, position == 0) {
                    break;
                }
            }
        }
    }

    /// Drop the batches at the head of `queue`'s line while `drop` holds (the rest of a command
    /// buffer that failed), and give them back.
    pub(crate) fn drop_head(&mut self, queue: u64, mut drop: impl FnMut(&B) -> bool) -> Vec<B> {
        let Some(at) = self.lines.iter().position(|(line, _)| *line == queue) else {
            return Vec::new();
        };
        let mut dropped = Vec::new();
        while self.lines[at].1.front().is_some_and(|(_, head)| drop(head)) {
            if let Some((_, batch)) = self.lines[at].1.pop_front() {
                dropped.push(batch);
            }
        }
        if self.lines[at].1.is_empty() {
            self.remove_line(at);
        }
        dropped
    }

    /// How many batches are held.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lines.iter().map(|(_, line)| line.len()).sum()
    }

    /// Whether no batch is held.
    pub(crate) fn is_empty(&self) -> bool {
        self.lines.iter().all(|(_, line)| line.is_empty())
    }
}

/// What one pass over an unchanged [`Order`] found of which heads wait behind an earlier batch
/// ([`Order::is_behind`]): each answer, each line's batches found to keep their place, and each
/// footprint taken, by commit sequence number. Made new for each pass.
pub(crate) struct Behind<F> {
    answers: BTreeMap<u64, bool>,
    kept: Vec<Kept>,
    footprints: BTreeMap<u64, F>,
}

impl<F> Behind<F> {
    /// The commit sequence numbers this pass asked about and found waiting behind a batch
    /// committed before them ([`Order::is_behind`]), for the caller to count the hold of.
    pub(crate) fn behind(&self) -> impl Iterator<Item = u64> + '_ {
        (self.answers.iter()).filter_map(|(&sequence, &behind)| behind.then_some(sequence))
    }
}

impl<F> Default for Behind<F> {
    fn default() -> Behind<F> {
        Behind {
            answers: BTreeMap::new(),
            kept: Vec::new(),
            footprints: BTreeMap::new(),
        }
    }
}

/// How far one line's batches were found to keep their place: the first `scanned` do, and, where
/// `stopped`, the next does not.
#[derive(Clone, Copy, Default)]
struct Kept {
    scanned: usize,
    stopped: bool,
}
