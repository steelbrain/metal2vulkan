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
}

#[derive(Debug)]
struct Line<I> {
    items: VecDeque<Entry<I>>,
    /// A thread runs the line's first item.
    running: bool,
}

#[derive(Debug)]
struct Entry<I> {
    item: I,
    /// The gate it waits behind, if any ([`Lines::push_behind`]).
    gate: Option<u64>,
}

impl<I> Default for Lines<I> {
    fn default() -> Lines<I> {
        Lines {
            lines: HashMap::new(),
            ready: VecDeque::new(),
            closed: HashMap::new(),
        }
    }
}

impl<I> Lines<I> {
    /// Append `item` to `queue`'s line.
    pub(crate) fn push(&mut self, queue: u64, item: I) {
        self.push_entry(queue, Entry { item, gate: None });
    }

    /// Close the new gate `token` once ([`Lines::push_behind`]).
    pub(crate) fn close_new(&mut self, token: u64) {
        self.closed.insert(token, 1);
    }

    /// Append `item` to `queue`'s line behind the gate `token` ([`Lines::close_new`]): it runs once
    /// every closing of the gate is opened ([`Lines::open`]).
    pub(crate) fn push_behind(&mut self, queue: u64, item: I, token: u64) {
        let gate = Some(token);
        self.push_entry(queue, Entry { item, gate });
    }

    fn push_entry(&mut self, queue: u64, entry: Entry<I>) {
        let open = self.open_entry(&entry);
        let line = self.lines.entry(queue).or_insert_with(|| Line {
            items: VecDeque::new(),
            running: false,
        });
        if line.items.is_empty() && !line.running && open {
            self.ready.push_back(queue);
        }
        line.items.push_back(entry);
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
        Some((queue, entry.item))
    }

    /// The thread running `queue`'s line is done with the item it took: the line is ready again
    /// if it holds more and the first of them is open, and is forgotten if it holds none.
    pub(crate) fn done(&mut self, queue: u64) {
        let Some(line) = self.lines.get_mut(&queue) else {
            return;
        };
        line.running = false;
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
    /// runs before anything it runs in the line itself; the line is running until
    /// [`Lines::done`]. `None` otherwise, changing nothing.
    pub(crate) fn take_plain(
        &mut self,
        queue: u64,
        plain: impl Fn(&I) -> bool,
    ) -> Option<VecDeque<I>> {
        if let Some(line) = self.lines.get(&queue) {
            let runs = |entry: &Entry<I>| plain(&entry.item) && self.open_entry(entry);
            if line.running || !line.items.iter().all(runs) {
                return None;
            }
        }
        let items: VecDeque<I> = (self.lines.remove(&queue)).map_or_else(VecDeque::new, |line| {
            line.items.into_iter().map(|entry| entry.item).collect()
        });
        if !items.is_empty() {
            // A line that holds an item and runs on no thread is ready, once.
            self.ready.retain(|ready| *ready != queue);
        }
        self.lines.insert(
            queue,
            Line {
                items: VecDeque::new(),
                running: true,
            },
        );
        Some(items)
    }

    /// Take `queue`'s line for the calling thread where it holds no item and no thread runs it:
    /// the line is running until [`Lines::done`]. False otherwise, changing nothing.
    pub(crate) fn take_idle(&mut self, queue: u64) -> bool {
        if self.lines.contains_key(&queue) {
            return false;
        }
        self.lines.insert(
            queue,
            Line {
                items: VecDeque::new(),
                running: true,
            },
        );
        true
    }

    /// Whether the item `is` picks out leads `queue`'s line: it is the first item and no thread
    /// runs the line; or, where the caller knows it was pushed (`posted`) and is not done, a thread
    /// runs the line and no item left in it is it, so the thread runs it.
    pub(crate) fn leads(&self, queue: u64, is: impl Fn(&I) -> bool, posted: bool) -> bool {
        let Some(line) = self.lines.get(&queue) else {
            return false;
        };
        match line.running {
            true => posted && !line.items.iter().any(|entry| is(&entry.item)),
            false => line.items.front().is_some_and(|entry| is(&entry.item)),
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
