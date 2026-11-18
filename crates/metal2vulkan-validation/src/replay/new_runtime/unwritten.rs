//! The Shared buffers a dispatch read before anything the stream records put bytes in them: a
//! buffer of the Shared class, made without bytes, that a compute encoder binds or uses, whose
//! command buffer commits before any recorded write, upload or GPU pass reached its bytes since its
//! making. Such a buffer plays with whatever its memory held, which is where a recording that
//! missed the program's writes shows. Counted on the report's `unwritten` line, never refused: a
//! dispatch that writes such a buffer before it reads it is counted too.
//!
//! What reaches a buffer's bytes: an `m2v.write` into it, into its block over them or into another
//! buffer of the block over them; a call of a blit or render encoder that names it or another
//! buffer of the block over them (a copy, a fill, an upload from the guest's RAM, an attachment);
//! and the commit of a dispatch that binds it, which may write it. A texture over a buffer is not
//! followed.

use super::super::stream::Call;
use super::objects_named;
use std::collections::{HashMap, HashSet};
use std::ops::Range;

/// How many of the buffers counted the report names.
const NAMED: usize = 8;

#[derive(Default)]
pub(in crate::replay) struct Unwritten {
    /// Each live buffer made in a Shared block: the block and its byte range there.
    placed: HashMap<u64, (u64, Range<u64>)>,
    /// The live Shared buffers nothing has reached since their making.
    waiting: HashSet<u64>,
    /// Those of them that lie in a block, by the block.
    in_block: HashMap<u64, Vec<u64>>,
    /// The waiting buffers each open compute encoder bound or used since its last dispatch.
    bound: HashMap<u64, Vec<u64>>,
    /// The waiting buffers a dispatch of each uncommitted command buffer bound.
    read: HashMap<u64, Vec<u64>>,
    /// The buffers a committed dispatch read while waiting, and the first of them with the
    /// command buffer that read each.
    count: u64,
    first: Vec<(u64, u64)>,
}

impl Unwritten {
    /// Shared buffer `id` made without bytes; `placed` is its block and its range there, where it
    /// lies in one.
    pub(in crate::replay) fn made(&mut self, id: u64, placed: Option<(u64, Range<u64>)>) {
        self.waiting.insert(id);
        if let Some((block, range)) = placed {
            self.in_block.entry(block).or_default().push(id);
            self.placed.insert(id, (block, range));
        }
    }

    /// Bytes `range` of buffer or block `id` written.
    pub(in crate::replay) fn written(&mut self, id: u64, range: Range<u64>) {
        self.waiting.remove(&id);
        match self.placed.get(&id) {
            Some((block, at)) => {
                let block = *block;
                let range = at.start + range.start..at.start + range.end;
                self.reach(block, range);
            }
            None => self.reach(id, range),
        }
    }

    /// Buffer or block `id` named by a GPU pass, which may write any of its bytes.
    fn named(&mut self, id: u64) {
        self.waiting.remove(&id);
        match self.placed.get(&id) {
            Some((block, at)) => {
                let (block, range) = (*block, at.clone());
                self.reach(block, range);
            }
            None => self.reach(id, 0..u64::MAX),
        }
    }

    /// Bytes `range` of block `block` reached: every waiting buffer of the block over them is.
    fn reach(&mut self, block: u64, range: Range<u64>) {
        let Some(ids) = self.in_block.get_mut(&block) else {
            return;
        };
        let (waiting, placed) = (&mut self.waiting, &self.placed);
        ids.retain(|id| {
            let over = (placed.get(id))
                .is_some_and(|(_, at)| at.start < range.end && range.start < at.end);
            if over {
                waiting.remove(id);
            }
            waiting.contains(id)
        });
    }

    /// Call `call` of an encoder of command buffer `command_buffer`, a compute encoder where
    /// `compute`.
    pub(in crate::replay) fn encoded(&mut self, call: &Call, command_buffer: u64, compute: bool) {
        if self.waiting.is_empty() {
            return;
        }
        if compute && call.selector.starts_with("dispatch") {
            if let Some(bound) = self.bound.remove(&call.receiver) {
                self.read.entry(command_buffer).or_default().extend(bound);
            }
            return;
        }
        let mut named = Vec::new();
        objects_named(&call.args, &mut named);
        for id in named {
            match compute {
                true if self.waiting.contains(&id) => {
                    self.bound.entry(call.receiver).or_default().push(id)
                }
                true => {}
                false => self.named(id),
            }
        }
    }

    /// Command buffer `command_buffer` committed: the buffers its dispatches read still waiting
    /// are counted, and every one of them is reached, as the dispatches may write them.
    pub(in crate::replay) fn committed(&mut self, command_buffer: u64) {
        for id in self.read.remove(&command_buffer).unwrap_or_default() {
            if self.waiting.contains(&id) {
                self.count += 1;
                if self.first.len() < NAMED {
                    self.first.push((id, command_buffer));
                }
            }
            self.named(id);
        }
    }

    /// Object `id` released.
    pub(in crate::replay) fn released(&mut self, id: u64) {
        self.waiting.remove(&id);
        if let Some((block, _)) = self.placed.remove(&id) {
            if let Some(ids) = self.in_block.get_mut(&block) {
                ids.retain(|other| *other != id);
            }
        }
        self.in_block.remove(&id);
        self.bound.remove(&id);
        self.read.remove(&id);
    }

    /// The report's line.
    pub(in crate::replay) fn line(&self) -> String {
        let first: Vec<String> = (self.first.iter())
            .map(|(buffer, command_buffer)| format!("#{buffer} in #{command_buffer}"))
            .collect();
        let first = match first.is_empty() {
            true => String::new(),
            false => format!("; first {}", first.join(", ")),
        };
        format!(
            "unwritten {}: Shared buffers a dispatch read before any recorded write, upload or GPU \
             pass reached their bytes since their making{first}",
            self.count
        )
    }
}
