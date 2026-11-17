//! The queues' serial contexts as a pure structure: one first-in-first-out line of items per
//! logical queue, each line run by at most one thread at a time, and the lines waiting for a thread
//! in the order they became ready.
//!
//! A line is ready when it holds an item, its first item's gate is open, and no thread runs it. A
//! thread takes the first ready line's first item and runs it; until it is done with it, that
//! line's later items wait, while every other line can be taken by another thread. A line that is
//! done and still holds items is ready again, behind the lines that became ready meanwhile.
//!
//! **Gates.** An item may be pushed behind a gate ([`Lines::push_behind`]), which holds it, and its
//! line's later items, until every closing of the gate is opened again ([`Lines::close_new`],
//! [`Lines::close`], [`Lines::open`]): it keeps its place in its line meanwhile.
//!
//! **Places.** Each item pushed takes the next place, whatever its line, so places rise along a
//! line. A line keeps the places of its items that may block a thread that runs them (the
//! `may_block` of [`Lines::new`]), and whether what a thread runs of it may: whether an item
//! leads its line ([`Lines::leads`]) is a look at the line's first item and first such place,
//! however many items the line holds.

use std::collections::{HashMap, VecDeque};

/// The items not yet run, by logical queue.
#[derive(Debug)]
pub(crate) struct Lines<I> {
    /// Each queue's line while it holds an item or a thread runs it, by the queue's identity.
    lines: HashMap<u64, Line<I>>,
    /// The ready lines, in the order they became ready.
    ready: VecDeque<u64>,
    /// The gates not yet open, by token: how many closings each still holds.
    closed: HashMap<u64, u32>,
    /// The places taken so far: the last item pushed took this one, the first 1.
    placed: u64,
    /// Whether an item may block a thread that runs it ([`Lines::new`]).
    may_block: fn(&I) -> bool,
}

#[derive(Debug)]
struct Line<I> {
    items: VecDeque<Entry<I>>,
    /// The places of the items it holds that may block, in order.
    blocking: VecDeque<u64>,
    /// A thread runs the line's first item.
    running: bool,
    /// What that thread runs may block: false where no thread runs the line.
    running_blocks: bool,
}

impl<I> Line<I> {
    fn new(running: bool, running_blocks: bool) -> Line<I> {
        Line {
            items: VecDeque::new(),
            blocking: VecDeque::new(),
            running,
            running_blocks,
        }
    }
}

#[derive(Debug)]
struct Entry<I> {
    item: I,
    /// The gate it waits behind, if any ([`Lines::push_behind`]).
    gate: Option<u64>,
    /// Its place ([`Lines::push`]).
    place: u64,
}

impl<I> Lines<I> {
    /// Lines in which `may_block` tells whether an item may block a thread that runs it: what
    /// [`Lines::leads`] looks past.
    pub(crate) fn new(may_block: fn(&I) -> bool) -> Lines<I> {
        Lines {
            lines: HashMap::new(),
            ready: VecDeque::new(),
            closed: HashMap::new(),
            placed: 0,
            may_block,
        }
    }

    /// Append `item` to `queue`'s line: its place.
    pub(crate) fn push(&mut self, queue: u64, item: I) -> u64 {
        self.push_entry(queue, item, None)
    }

    /// Close the new gate `token` once ([`Lines::push_behind`]).
    pub(crate) fn close_new(&mut self, token: u64) {
        self.closed.insert(token, 1);
    }

    /// Append `item` to `queue`'s line behind the gate `token` ([`Lines::close_new`]): it runs once
    /// every closing of the gate is opened ([`Lines::open`]). Its place.
    pub(crate) fn push_behind(&mut self, queue: u64, item: I, token: u64) -> u64 {
        self.push_entry(queue, item, Some(token))
    }

    fn push_entry(&mut self, queue: u64, item: I, gate: Option<u64>) -> u64 {
        self.placed += 1;
        let place = self.placed;
        let blocks = (self.may_block)(&item);
        let entry = Entry { item, gate, place };
        let open = self.open_entry(&entry);
        let line = (self.lines.entry(queue)).or_insert_with(|| Line::new(false, false));
        if line.items.is_empty() && !line.running && open {
            self.ready.push_back(queue);
        }
        if blocks {
            line.blocking.push_back(place);
        }
        line.items.push_back(entry);
        place
    }

    /// Whether `entry` waits behind no gate that is still closed.
    fn open_entry(&self, entry: &Entry<I>) -> bool {
        entry
            .gate
            .is_none_or(|token| !self.closed.contains_key(&token))
    }

    /// Close the gate `token` once more, where it is not open yet: whether it was not.
    pub(crate) fn close(&mut self, token: u64) -> bool {
        match self.closed.get_mut(&token) {
            Some(closings) => {
                *closings += 1;
                true
            }
            None => false,
        }
    }

    /// Open one closing of the gate `token`, whose item is in `queue`'s line: once none is left,
    /// the line is ready where that item is its first and no thread runs it.
    pub(crate) fn open(&mut self, queue: u64, token: u64) {
        let Some(closings) = self.closed.get_mut(&token) else {
            return;
        };
        *closings -= 1;
        if *closings > 0 {
            return;
        }
        self.closed.remove(&token);
        let Some(line) = self.lines.get(&queue) else {
            return;
        };
        let first = line
            .items
            .front()
            .is_some_and(|entry| entry.gate == Some(token));
        if first && !line.running {
            self.ready.push_back(queue);
        }
    }

    /// Take the first item of the first ready line: the line is running until
    /// [`Lines::done`]. `None` when no line is ready.
    pub(crate) fn take(&mut self) -> Option<(u64, I)> {
        let queue = self.ready.pop_front()?;
        let line = self.lines.get_mut(&queue)?;
        line.running = true;
        let entry = line.items.pop_front()?;
        line.running_blocks = line.blocking.front() == Some(&entry.place);
        if line.running_blocks {
            line.blocking.pop_front();
        }
        Some((queue, entry.item))
    }

    /// The thread running `queue`'s line is done with the item it took: the line is ready again
    /// if it holds more and the first of them is open, and is forgotten if it holds none.
    pub(crate) fn done(&mut self, queue: u64) {
        let Some(line) = self.lines.get_mut(&queue) else {
            return;
        };
        line.running = false;
        line.running_blocks = false;
        let Some(first) = line.items.front() else {
            self.lines.remove(&queue);
            return;
        };
        let open = first
            .gate
            .is_none_or(|token| !self.closed.contains_key(&token));
        if open {
            self.ready.push_back(queue);
        }
    }

    /// Take `queue`'s line for the calling thread, where no thread runs it and `plain` holds for
    /// each item it holds, none behind a closed gate: those items, in order, which the caller
    /// runs before anything it runs in the line itself (which may block where `blocks`); the
    /// line is running until [`Lines::done`]. `None` otherwise, changing nothing.
    pub(crate) fn take_plain(
        &mut self,
        queue: u64,
        plain: impl Fn(&I) -> bool,
        blocks: bool,
    ) -> Option<VecDeque<I>> {
        if let Some(line) = self.lines.get(&queue) {
            let runs = |entry: &Entry<I>| plain(&entry.item) && self.open_entry(entry);
            if line.running || !line.items.iter().all(runs) {
                return None;
            }
        }
        let (items, blocking) = match self.lines.remove(&queue) {
            Some(line) => {
                let items: VecDeque<I> = line.items.into_iter().map(|entry| entry.item).collect();
                (items, !line.blocking.is_empty())
            }
            None => (VecDeque::new(), false),
        };
        if !items.is_empty() {
            // A line that holds an item and runs on no thread is ready, once.
            self.ready.retain(|ready| *ready != queue);
        }
        self.lines
            .insert(queue, Line::new(true, blocking || blocks));
        Some(items)
    }

    /// Take `queue`'s line for the calling thread where it holds no item and no thread runs it,
    /// for what may block where `blocks`: the line is running until [`Lines::done`]. False
    /// otherwise, changing nothing.
    pub(crate) fn take_idle(&mut self, queue: u64, blocks: bool) -> bool {
        if self.lines.contains_key(&queue) {
            return false;
        }
        self.lines.insert(queue, Line::new(true, blocks));
        true
    }

    /// Whether the item placed at `at` in `queue`'s line ([`Lines::push`]), which the caller knows
    /// is not done, leads the line: where it waits there, no item before it may block, nor what a
    /// thread runs of the line; where it no longer waits there, a thread runs it. One look at the
    /// line's first item and first place that may block, however many items the line holds.
    pub(crate) fn leads(&self, queue: u64, at: u64) -> bool {
        let Some(line) = self.lines.get(&queue) else {
            return false;
        };
        let waits = line.items.front().is_some_and(|first| first.place <= at);
        match waits {
            true => {
                let before = line.blocking.front().is_some_and(|&place| place < at);
                !line.running_blocks && !before
            }
            false => line.running,
        }
    }

    /// How many lines are ready.
    pub(crate) fn ready(&self) -> usize {
        self.ready.len()
    }

    /// Whether a line is ready.
    pub(crate) fn is_ready(&self) -> bool {
        !self.ready.is_empty()
    }

    /// Whether no line holds an item or is running.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}
