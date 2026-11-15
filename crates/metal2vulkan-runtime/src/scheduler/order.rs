//! The scheduler's order, as a pure structure: committed batches in one first-in-first-out line per
//! logical queue, and the choice of the batch to submit next.
//!
//! A logical queue runs its command buffers in commit order, and a command buffer its batches in
//! encoded order, so only the head of each line may be submitted. Among the heads that are ready
//! (each of their waits satisfied by work already submitted, or by the host), the one committed
//! first goes next, of those the caller puts first where one is ready (below). A head that is not
//! ready holds back its own line, and the heads of other lines committed after it that it
//! conflicts with:
//!
//! - **Tracked order across queues is the order the batches go.** Metal tracks a resource across
//!   queues in the order the GPU takes the command buffers, which need not be their commit order:
//!   a command buffer committed after another on another queue, writing or reading one tracked
//!   buffer the first writes, ran before it now and then on a Radeon Pro 5300M and a UHD 630
//!   (macOS 26.7.1), and in commit order on an M3 Max (macOS 27.0.1)
//!   (`metal2vulkan-conformance`'s `probes/cross_queue_surface_order.m`, its `nocopy-one`
//!   control). The runtime's tracker plans each batch as it is submitted, so the batches it
//!   submits run in submission order wherever they conflict, and the order picks which goes
//!   first: a batch held for a reason of the runtime's own that keeps its place (its submission
//!   checks; below) must not let a later commit's batch that conflicts with it (a tracked
//!   resource's range that one of them writes, or a fence) go before it: such a head is
//!   [`behind`](Order::is_behind) it, and waits.
//! - **Held batches give way.** A batch held unsubmitted for its host step before the step read
//!   anything the GPU writes ([`Place::GivesWay`]: its pipelines being made, or a step that reads
//!   nothing of the GPU's) does not keep its place: a later commit of another queue that
//!   conflicts with it goes, as an x86 Mac's GPU may take it first, and the held batch is
//!   submitted after it, so the tracker orders the two that way; the batches after it on its line
//!   give way with it. A head still waits behind such a batch where the two share a fence (one
//!   updates a fence the other waits for or updates, or a batch whose uses are not known yet and
//!   one that names a fence): a batch is never moved after a fence update it waits for. A step
//!   whose gate was taken still gives way, but to a later write of what it reads on the host (its
//!   executions' records and ranges): what it reads stands from its gate on, and that write going
//!   first would land after the step read; the caller's `ordered` says so for it. A step whose
//!   host reads are not known keeps its place from its gate on. (A cold compute pipeline's making
//!   of 3 s on an Iris Xe held one guest process's command buffer, and with it every other
//!   process's frames over a surface they shared: the guest's display froze for 3.1 s. A step
//!   waiting at its gate for an 89 ms pass before it in its command buffer held every queue's
//!   commits on an RTX 3090, a compositor's included.)
//! - **Which batches keep their place.** Every batch of a line up to the first that waits for
//!   an event value not yet reached by a submitted signal or the host, or for a binary
//!   semaphore whose signal is not yet submitted. That one and those after it wait for the
//!   program's signal, as Metal schedules them after it, so a later commit may go first: Metal
//!   follows the event.
//! - **No deadlock.** A head waits behind batches committed before it that keep their place, or
//!   that give way but share a fence with it, only; the first of those waits for no batch but its
//!   line's earlier ones, or for the runtime (a host step's gate or pipelines, its checks), which
//!   never waits for a later commit.
//! - **Heads put first.** The caller may put heads first ([`next`](Order::next)'s `first`): a
//!   ready one goes before the others, still behind any batch it conflicts with that keeps its
//!   place, as every head. The scheduler puts first an unshared queue's head on a `VkQueue` whose
//!   other queues' work is held back for it (`unshared_hold.rs`): a pass that finds that work let
//!   go and the unshared queue's command buffer ready together, as the pass of a scheduler thread
//!   the host ran late does, would otherwise send the whole of that work ahead of it.
//!
//! Since a batch is submitted only once every wait it has is satisfied by submitted work, the
//! batches on a `VkQueue` never wait for a signal not yet submitted, and no `VkQueue` stalls on
//! one: many logical queues on one `VkQueue` cannot deadlock through the order they wait for each
//! other in.
//!
//! **Found as asked**. Whether a head waits behind an earlier batch is answered for the heads that
//! would otherwise go ([`is_behind`](Order::is_behind)), not for every head on every pass:
//! [`next`](Order::next) asks it of the ready heads in its order until one goes, and the
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
    /// The heads [`next`](Order::next) orders: those not put first after the others, each by
    /// commit sequence number, with its line.
    heads: Vec<(bool, u64, usize)>,
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
    /// before it on another line: one that keeps its place there or gives way (`keeps` answers
    /// [`Place::Keeps`] or [`Place::GivesWay`] for it and for every batch before it on its line,
    /// one after a batch that gives way giving way too), whose footprint (`footprint`) and the
    /// head's `ordered` says run in commit order, given whether the earlier gives way. `memo`
    /// keeps, for one pass over an unchanged order, each answer, each line's batches found to keep
    /// their place, and each footprint taken. False for a sequence number no head has.
    pub(crate) fn is_behind<F>(
        &self,
        sequence: u64,
        memo: &mut Behind<F>,
        (keeps, footprint, ordered): (
            &mut impl FnMut(&B) -> Place,
            &mut impl FnMut(&B) -> F,
            &impl Fn(&F, &F, bool) -> bool,
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
            &mut impl FnMut(&B) -> Place,
            &mut impl FnMut(&B) -> F,
            &impl Fn(&F, &F, bool) -> bool,
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
                // Whether it and every batch before it on its line keep their place or give way,
                // each asked once a pass.
                let kept = &mut memo.kept[other];
                if index >= kept.scanned {
                    if kept.stopped {
                        break;
                    }
                    match keeps(batch) {
                        Place::Stops => {
                            kept.stopped = true;
                            break;
                        }
                        Place::GivesWay => {
                            kept.gives_way.get_or_insert(index);
                        }
                        Place::Keeps => {}
                    }
                    kept.scanned = index + 1;
                }
                let gives_way = kept.gives_way.is_some_and(|from| index >= from);
                memo.footprints
                    .entry(*first)
                    .or_insert_with(|| footprint(batch));
                memo.footprints
                    .entry(sequence)
                    .or_insert_with(|| footprint(head));
                let (earlier, later) = (&memo.footprints[first], &memo.footprints[&sequence]);
                if ordered(earlier, later, gives_way) {
                    return true;
                }
            }
        }
        false
    }

    /// Each line's head, with its commit sequence number.
    pub(crate) fn heads(&self) -> impl Iterator<Item = (u64, &B)> + '_ {
        (self.lines.iter())
            .filter_map(|(_, line)| line.front())
            .map(|(sequence, head)| (*sequence, head))
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
    /// `behind` (asked of those, those `first` puts first in commit order, then the others in
    /// commit order, until one goes), the first asked, with its line's queue, its commit sequence
    /// number (for [`put_back`](Self::put_back)) and what `ready` made of it. `None` when no head
    /// is ready.
    ///
    /// `ready` takes each head it looks at mutably, so a caller may note on a head it holds back
    /// why it did (`scheduler/mod.rs`'s [`HeldFor`](crate::scheduler::HeldFor)); it is asked of
    /// each head in that order until one is both ready and not `behind`.
    pub(crate) fn next<T>(
        &mut self,
        first: impl Fn(&B) -> bool,
        mut behind: impl FnMut(&Self, u64) -> bool,
        mut ready: impl FnMut(&mut B) -> Option<T>,
    ) -> Option<(u64, u64, B, T)> {
        let Order { lines, heads, .. } = self;
        heads.clear();
        heads.extend((lines.iter().enumerate()).filter_map(|(at, (_, line))| {
            (line.front()).map(|(sequence, head)| (!first(head), *sequence, at))
        }));
        heads.sort_unstable();
        let mut found = None;
        for index in 0..self.heads.len() {
            let (_, sequence, at) = self.heads[index];
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

    /// The line head of commit sequence number `sequence`, to update in place; `None` where no
    /// head has it ([`Behind`] holds the numbers one pass asked about).
    pub(crate) fn head_mut_at(&mut self, sequence: u64) -> Option<&mut B> {
        (self.lines.iter_mut()).find_map(|(_, line)| {
            line.front_mut()
                .filter(|(first, _)| *first == sequence)
                .map(|(_, head)| head)
        })
    }

    /// The head of logical queue `queue`'s line, to update in place; `None` where the line holds
    /// no batch.
    pub(crate) fn head_mut_of(&mut self, queue: u64) -> Option<&mut B> {
        (self.lines.iter_mut())
            .find(|(line, _)| *line == queue)
            .and_then(|(_, line)| line.front_mut())
            .map(|(_, head)| head)
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

    /// The head committed first among those `take` holds of, taken from its line, with its
    /// logical queue; `None` where `take` holds of no head (a lost device's batches, failed in
    /// commit order whatever they wait for, `scheduler/mod.rs`'s `fail_lost`).
    pub(crate) fn pop_first_where(&mut self, mut take: impl FnMut(&B) -> bool) -> Option<(u64, B)> {
        let (_, at) = (self.lines.iter().enumerate())
            .filter_map(|(at, (_, line))| {
                let (sequence, head) = line.front()?;
                take(head).then_some((*sequence, at))
            })
            .min()?;
        let queue = self.lines[at].0;
        let (_, batch) = self.lines[at].1.pop_front()?;
        if self.lines[at].1.is_empty() {
            self.remove_line(at);
        }
        Some((queue, batch))
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

/// How far one line's batches were found to keep their place: the first `scanned` do or give way,
/// those from `gives_way` on giving way, and, where `stopped`, the next does neither.
#[derive(Clone, Copy, Default)]
struct Kept {
    scanned: usize,
    gives_way: Option<usize>,
    stopped: bool,
}

/// Where a batch stands in the order across queues ([`Order::is_behind`], the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Place {
    /// It keeps its place: a later commit of another line that conflicts with it waits.
    Keeps,
    /// It is held unsubmitted for its host step before the step read anything the GPU writes, or
    /// at or past its gate where what the step reads on the host is known: a later commit of
    /// another line goes ahead of it, but where the caller's `ordered` says the two still run in
    /// commit order given that it gives way (a fence both name; a write of what the step reads),
    /// and so do the batches after it on its line (**Held batches give way**).
    GivesWay,
    /// It waits for the program (an event value or a binary semaphore's signal): it and the
    /// batches after it on its line keep no place.
    Stops,
}
