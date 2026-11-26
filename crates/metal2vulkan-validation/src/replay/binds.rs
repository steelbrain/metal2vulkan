//! The guest's buffer binds (`docs/REPLAY-STREAM.md`, section 10): the slots the stream's
//! set-buffer calls on compute and render encoders set, each counted, and split into buffer
//! changes (another buffer than the slot held, the first set of a slot of its encoder among them)
//! and offset-only changes (the buffer the slot held, at another offset). A set of the buffer and
//! offset the slot holds is neither. Counts the stream fixes, the same on every play of it.

use super::args::range;
use super::stream::Call;
use super::value::Value;
use std::collections::HashMap;

/// The counts, in the order [`Binds::bound`] gives them: on render encoders, beside the draws,
/// the slots set, the buffer changes and the offset-only changes; then the same on compute
/// encoders, beside the dispatches.
pub(super) const BINDS: [&str; 6] = [
    "render_buffer_binds",
    "render_buffer_changes",
    "render_offset_changes",
    "compute_buffer_binds",
    "compute_buffer_changes",
    "compute_offset_changes",
];

/// A slot: its encoder, its stage (compute, vertex or fragment) and its index.
type Slot = (u64, Stage, u64);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Stage {
    Compute,
    Vertex,
    Fragment,
}

/// What a set-buffer call sets.
enum Set<'c> {
    /// A buffer (none where `Nil`) at an offset, from a slot on.
    Buffers(&'c [Value], Vec<u64>, u64),
    /// An offset alone, at a slot.
    Offset(u64, u64),
}

/// What each slot of the live encoders holds.
#[derive(Default)]
pub(super) struct Binds {
    /// Each slot set: its buffer (0 for none) and offset.
    held: HashMap<Slot, (u64, u64)>,
}

impl Binds {
    /// The counts `call`, a call on encoder `call.receiver`, adds, in the order of [`BINDS`]: none
    /// where it sets no buffer, or its arguments are not a set-buffer call's.
    pub(super) fn bound(&mut self, call: &Call) -> [u64; 6] {
        let mut counts = [0; 6];
        let Some((stage, set)) = set_of(call) else {
            return counts;
        };
        let at = match stage {
            Stage::Compute => 3,
            Stage::Vertex | Stage::Fragment => 0,
        };
        let mut set_slot = |index: u64, buffer: Option<u64>, offset: u64| {
            let slot = (call.receiver, stage, index);
            let held = self.held.get(&slot).copied();
            let buffer = buffer.or(held.map(|(buffer, _)| buffer)).unwrap_or(0);
            counts[at] += 1;
            match held {
                Some(held) if held == (buffer, offset) => {}
                Some((same, _)) if same == buffer => counts[at + 2] += 1,
                _ => counts[at + 1] += 1,
            }
            self.held.insert(slot, (buffer, offset));
        };
        match set {
            Set::Buffers(buffers, offsets, first) => {
                for ((index, buffer), offset) in (first..).zip(buffers).zip(offsets) {
                    set_slot(index, Some(buffer.as_object().unwrap_or(0)), offset);
                }
            }
            Set::Offset(offset, index) => set_slot(index, None, offset),
        }
        counts
    }

    /// `encoder` ended, or was released: its slots go.
    pub(super) fn ended(&mut self, encoder: u64) {
        self.held.retain(|&(of, ..), _| of != encoder);
    }
}

/// The stage and what `call` sets, where it is a set-buffer call with the arguments of one.
fn set_of(call: &Call) -> Option<(Stage, Set<'_>)> {
    let args = &call.args;
    let uint = |at: usize| args.get(at).and_then(Value::as_uint);
    let one = |stage| {
        let buffer = std::slice::from_ref(args.first()?);
        Some((stage, Set::Buffers(buffer, vec![uint(1)?], uint(2)?)))
    };
    let offset = |stage| Some((stage, Set::Offset(uint(0)?, uint(1)?)));
    let many = |stage| {
        let buffers = args.first()?.as_list()?;
        let offsets = args.get(1)?.as_uints()?;
        let first = range(call, args.get(2)?).ok()?.start;
        (buffers.len() == offsets.len()).then_some((stage, Set::Buffers(buffers, offsets, first)))
    };
    match &*call.selector {
        "setBuffer:offset:atIndex:" => one(Stage::Compute),
        "setVertexBuffer:offset:atIndex:" => one(Stage::Vertex),
        "setFragmentBuffer:offset:atIndex:" => one(Stage::Fragment),
        "setBufferOffset:atIndex:" => offset(Stage::Compute),
        "setVertexBufferOffset:atIndex:" => offset(Stage::Vertex),
        "setFragmentBufferOffset:atIndex:" => offset(Stage::Fragment),
        "setBuffers:offsets:withRange:" => many(Stage::Compute),
        "setVertexBuffers:offsets:withRange:" => many(Stage::Vertex),
        "setFragmentBuffers:offsets:withRange:" => many(Stage::Fragment),
        _ => None,
    }
}
