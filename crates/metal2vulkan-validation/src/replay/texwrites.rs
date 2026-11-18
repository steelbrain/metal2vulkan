//! CPU writes into Managed textures that a committed command buffer may still be using: for each
//! call on a texture that carries bytes (`replaceRegion:...`), into a texture the backend says is
//! Managed, whether a command buffer that used the texture was committed and the stream records no
//! completion of it (`Report::texture_writes`).
//!
//! The play waits for each command buffer at its commit; the stream records a completion only where
//! the program's CPU waited. So a user is "not seen complete" in the recording's order with only
//! those waits: a wait until it or a later command buffer of its queue completed (a queue
//! completes in order), or `waitUntilSignaledValue:timeoutMS:` reaching a value that it or a later
//! command buffer of its queue signals. Each such write is counted by the presents between the
//! user's commit and the write.
//!
//! A command buffer uses a texture where one of its encoders names it or a view of it, or where an
//! encoder of it was made with a descriptor (an object an `MTL…Descriptor` class call made, or one
//! made on such an object) that names it.

use super::player::TextureWrites;
use super::stream::Call;
use super::value::Value;
use std::collections::{HashMap, HashSet};

/// How many of the writes counted the report names.
const NAMED: usize = 8;
/// The signals kept per event: an older one's completion is the likeliest already recorded.
const SIGNALS: usize = 64;

/// The latest committed user of a texture on one queue.
#[derive(Clone, Copy)]
struct User {
    command_buffer: u64,
    number: u64,
    frame: u64,
}

#[derive(Default)]
pub(super) struct TexWrites {
    /// The live textures, views included, and each view's texture.
    textures: HashSet<u64>,
    base: HashMap<u64, u64>,
    /// Each open encoder's command buffer.
    owner: HashMap<u64, u64>,
    /// Each descriptor object's root, and the textures each root names.
    described: HashMap<u64, u64>,
    holds: HashMap<u64, HashSet<u64>>,
    /// The textures each uncommitted command buffer used.
    uses: HashMap<u64, HashSet<u64>>,
    /// Each texture's latest committed user on each queue.
    users: HashMap<u64, HashMap<u64, User>>,
    /// Each live committed command buffer's queue and number there.
    commits: HashMap<u64, (u64, u64)>,
    /// Each queue's commits so far, and the number through which its completions are recorded.
    numbers: HashMap<u64, u64>,
    done: HashMap<u64, u64>,
    /// Each event's signals by committed command buffers: value, queue, number.
    signals: HashMap<u64, Vec<(u64, u64, u64)>>,
    frame: u64,
    written: HashSet<u64>,
    written_pending: HashSet<u64>,
    counts: TextureWrites,
}

/// The objects `values` name, lists and maps included.
fn named(values: &[Value], out: &mut Vec<u64>) {
    for value in values {
        match value {
            Value::Object(id) => out.push(*id),
            Value::List(items) => named(items, out),
            Value::Map(fields) => {
                for (_, item) in fields {
                    named(std::slice::from_ref(item), out);
                }
            }
            _ => {}
        }
    }
}

impl TexWrites {
    fn base_of(&self, id: u64) -> u64 {
        self.base.get(&id).copied().unwrap_or(id)
    }

    /// Call `call`, its blobs resolved, played at `position`; `managed` tells whether a texture
    /// is Managed, where the backend knows.
    pub(super) fn saw(
        &mut self,
        call: &Call,
        position: u64,
        managed: impl Fn(u64) -> Option<bool>,
    ) {
        let (selector, receiver, result) = (&*call.selector, call.receiver, call.result);
        let mut names = Vec::new();
        named(&call.args, &mut names);
        if receiver == 0 && result != 0 && selector.starts_with("MTL") {
            if selector.contains("Descriptor") {
                self.described.insert(result, result);
            }
            return;
        }
        if let Some(&root) = self.described.get(&receiver) {
            if result != 0 {
                self.described.insert(result, root);
            }
            let held: Vec<u64> = (names.iter())
                .filter(|id| self.textures.contains(id))
                .map(|&id| self.base_of(id))
                .collect();
            if !held.is_empty() {
                self.holds.entry(root).or_default().extend(held);
            }
            return;
        }
        if result != 0 && selector.starts_with("newTexture") {
            if self.textures.contains(&receiver) {
                self.base.insert(result, self.base_of(receiver));
            }
            self.textures.insert(result);
            return;
        }
        if result != 0 && selector.contains("CommandEncoder") {
            let command_buffer = self.owner.get(&receiver).copied().unwrap_or(receiver);
            self.owner.insert(result, command_buffer);
            let held: Vec<u64> = (names.iter())
                .filter_map(|id| self.described.get(id))
                .filter_map(|root| self.holds.get(root))
                .flatten()
                .copied()
                .collect();
            self.uses.entry(command_buffer).or_default().extend(held);
            return;
        }
        if let Some(&command_buffer) = self.owner.get(&receiver) {
            let used: Vec<u64> = (names.iter())
                .filter(|id| self.textures.contains(id))
                .map(|&id| self.base_of(id))
                .collect();
            if !used.is_empty() {
                self.uses.entry(command_buffer).or_default().extend(used);
            }
            return;
        }
        if !self.textures.contains(&receiver) {
            return;
        }
        let bytes: u64 = (call.args.iter())
            .filter_map(|arg| match arg {
                Value::Bytes(data) => Some(data.len() as u64),
                _ => None,
            })
            .sum();
        if !call.args.iter().any(|arg| matches!(arg, Value::Bytes(_))) {
            return;
        }
        let texture = self.base_of(receiver);
        match managed(texture) {
            None => self.counts.unknown += 1,
            Some(false) => {}
            Some(true) => self.wrote(texture, bytes, position),
        }
    }

    /// `bytes` written into Managed texture `texture` at `position`.
    fn wrote(&mut self, texture: u64, bytes: u64, position: u64) {
        self.counts.writes += 1;
        self.counts.bytes += bytes;
        self.written.insert(texture);
        let done = &self.done;
        let open = (self.users.get(&texture).into_iter())
            .flatten()
            .filter(|(queue, user)| user.number > done.get(queue).copied().unwrap_or(0))
            .map(|(_, user)| *user)
            .max_by_key(|user| (user.frame, user.number));
        let Some(user) = open else {
            return;
        };
        let presents = self.frame - user.frame;
        self.counts.pending += 1;
        self.counts.pending_bytes += bytes;
        self.counts.by_presents[presents.min(3) as usize] += 1;
        self.written_pending.insert(texture);
        if self.counts.first.len() < NAMED {
            (self.counts.first).push((position, texture, user.command_buffer, presents));
        }
    }

    /// Command buffer `command_buffer` of `queue` committed, with the event signals it encoded.
    pub(super) fn committed(&mut self, command_buffer: u64, queue: u64, signals: &[(u64, u64)]) {
        let number = self.numbers.entry(queue).or_default();
        *number += 1;
        let number = *number;
        self.commits.insert(command_buffer, (queue, number));
        let user = User {
            command_buffer,
            number,
            frame: self.frame,
        };
        for texture in self.uses.remove(&command_buffer).unwrap_or_default() {
            self.users.entry(texture).or_default().insert(queue, user);
        }
        for &(event, value) in signals {
            let kept = self.signals.entry(event).or_default();
            if kept.len() == SIGNALS {
                kept.remove(0);
            }
            kept.push((value, queue, number));
        }
    }

    /// The CPU waited for command buffer `command_buffer` to complete.
    pub(super) fn waited(&mut self, command_buffer: u64) {
        if let Some(&(queue, number)) = self.commits.get(&command_buffer) {
            let done = self.done.entry(queue).or_default();
            *done = (*done).max(number);
        }
    }

    /// The CPU waited for event `event` to reach `value`.
    pub(super) fn event_waited(&mut self, event: u64, value: u64) {
        let Some(kept) = self.signals.get_mut(&event) else {
            return;
        };
        for &(signaled, queue, number) in kept.iter() {
            if signaled <= value {
                let done = self.done.entry(queue).or_default();
                *done = (*done).max(number);
            }
        }
        kept.retain(|&(signaled, ..)| signaled > value);
    }

    /// A frame presented.
    pub(super) fn presented(&mut self) {
        self.frame += 1;
    }

    /// Object `id` released.
    pub(super) fn released(&mut self, id: u64) {
        self.textures.remove(&id);
        self.base.remove(&id);
        self.owner.remove(&id);
        self.described.remove(&id);
        self.holds.remove(&id);
        self.uses.remove(&id);
        self.users.remove(&id);
        self.commits.remove(&id);
    }

    /// What was counted.
    pub(super) fn counts(&self) -> TextureWrites {
        let mut counts = self.counts.clone();
        counts.textures = self.written.len() as u64;
        counts.pending_textures = self.written_pending.len() as u64;
        counts
    }
}
