//! The pre-pass a consumer that holds a whole command buffer runs before it encodes it, read
//! ahead: each compute dispatch's state, grid, threadgroup size and threadgroup memory lengths,
//! due at the stream position where such a consumer could first prepare it, once its command
//! buffer and its state are both made. The player hands each position's dispatches to the
//! backend once that position's call has played (`Backend::prepare`), in stream order. A
//! dispatch the same as an earlier one of its state (the same grid, size and lengths) is not due
//! again: what it needs is prepared once.

use super::stream::Call;
use super::value::Value;
use std::collections::{HashMap, HashSet};

/// A compute dispatch to prepare: its compute pipeline state's id, its grid as its call took it,
/// its threadgroup size, and the encoder's threadgroup memory lengths as (index, length) pairs,
/// each index once, in index order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ahead {
    pub state: u64,
    pub grid: AheadGrid,
    pub size: [u32; 3],
    pub lengths: Vec<(u32, u32)>,
}

/// A direct dispatch's grid, as its call took it. The player plays no indirect dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AheadGrid {
    /// `dispatchThreads:threadsPerThreadgroup:`'s threads per grid.
    Threads([u32; 3]),
    /// `dispatchThreadgroups:threadsPerThreadgroup:`'s threadgroups per grid.
    Threadgroups([u32; 3]),
}

/// The read-ahead's state while it reads the stream, and the dispatches due by position.
#[derive(Default)]
pub(super) struct Prepass {
    /// Each live command buffer's id to the position of the call that made it: a split or a part
    /// of the host's own goes on with its receiver's.
    buffers: HashMap<u64, u64>,
    /// Each live compute pipeline state's id to the position of the call that made it.
    states: HashMap<u64, u64>,
    /// Each live compute encoder's command buffer (its making's position), state and threadgroup
    /// memory lengths.
    encoders: HashMap<u64, (u64, u64, Vec<(u32, u32)>)>,
    /// The dispatches due so far, with their state's making's position.
    seen: HashSet<(u64, Ahead)>,
    /// The dispatches due at each position, in stream order.
    due: HashMap<u64, Vec<Ahead>>,
}

/// A size argument's three extents.
fn size(value: Option<&Value>) -> Option<[u32; 3]> {
    let fields = value?.as_map()?;
    let get = |key: &str| {
        (fields.iter())
            .find(|(k, _)| &**k == key)
            .and_then(|(_, v)| v.as_uint())
            .and_then(|v| u32::try_from(v).ok())
    };
    Some([get("width")?, get("height")?, get("depth")?])
}

impl Prepass {
    /// Reads `call`, at stream position `position`.
    pub(super) fn see(&mut self, position: u64, call: &Call) {
        let s = &*call.selector;
        if s == "release" {
            self.buffers.remove(&call.receiver);
            self.states.remove(&call.receiver);
            self.encoders.remove(&call.receiver);
            return;
        }
        if call.result != 0 {
            if s.starts_with("commandBuffer") {
                self.buffers.insert(call.result, position);
            } else if s == "m2v.split" || s == "m2v.part" {
                let buffer = self.buffers.get(&call.receiver).copied();
                self.buffers.insert(call.result, buffer.unwrap_or(position));
            } else if s.starts_with("newComputePipelineState") {
                self.states.insert(call.result, position);
            } else if s.starts_with("computeCommandEncoder") {
                if let Some(&buffer) = self.buffers.get(&call.receiver) {
                    self.encoders.insert(call.result, (buffer, 0, Vec::new()));
                }
            }
            return;
        }
        let Some(encoder) = self.encoders.get_mut(&call.receiver) else {
            return;
        };
        match s {
            "setComputePipelineState:" => {
                encoder.1 = call.args.first().and_then(Value::as_object).unwrap_or(0);
            }
            "setThreadgroupMemoryLength:atIndex:" => {
                let get = |at: usize| {
                    (call.args.get(at))
                        .and_then(Value::as_uint)
                        .and_then(|v| u32::try_from(v).ok())
                };
                if let (Some(length), Some(index)) = (get(0), get(1)) {
                    let lengths = &mut encoder.2;
                    match lengths.binary_search_by_key(&index, |&(at, _)| at) {
                        Ok(at) => lengths[at].1 = length,
                        Err(at) => lengths.insert(at, (index, length)),
                    }
                }
            }
            "dispatchThreads:threadsPerThreadgroup:"
            | "dispatchThreadgroups:threadsPerThreadgroup:" => {
                let (Some(grid), Some(size)) = (size(call.args.first()), size(call.args.get(1)))
                else {
                    return;
                };
                let (buffer, state, lengths) = (encoder.0, encoder.1, encoder.2.clone());
                let Some(&made) = self.states.get(&state) else {
                    return;
                };
                let grid = match s {
                    "dispatchThreads:threadsPerThreadgroup:" => AheadGrid::Threads(grid),
                    _ => AheadGrid::Threadgroups(grid),
                };
                let ahead = Ahead {
                    state,
                    grid,
                    size,
                    lengths,
                };
                if self.seen.insert((made, ahead.clone())) {
                    self.due.entry(buffer.max(made)).or_default().push(ahead);
                }
            }
            _ => {}
        }
    }

    /// The dispatches due once the call at `position` has played.
    pub(super) fn due(&mut self, position: u64) -> Option<Vec<Ahead>> {
        self.due.remove(&position)
    }
}
