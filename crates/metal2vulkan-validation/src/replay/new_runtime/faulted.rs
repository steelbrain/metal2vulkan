//! A lost device's fault addresses placed among the stream's buffers and heaps, for the report's
//! `loss` line (`docs/REPLAY-STREAM.md` section 10). An access's range is its address rounded
//! down to its precision, `precision` bytes long. Each live buffer or heap whose memory meets it
//! is named with the address's offset in it, then each of the last [`FREED_KEPT`] released ones
//! that met it, newest first, with the frame and call of its release; where none meets it, the
//! line says so. A texture is placed only through the heap or block it lies in, and the play's
//! own buffers are not placed.

use metal2vulkan_runtime as rt;
use std::collections::VecDeque;

/// The released buffers and heaps kept for placing, oldest dropped first.
pub(super) const FREED_KEPT: usize = 1 << 16;

/// The places named for one address, of the live ones and of the released ones each.
const SHOWN: usize = 8;

/// A buffer's or heap's memory: its object, whether it is a heap (or a block of the stream),
/// and its GPU address range in the play.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Span {
    pub(super) object: u64,
    pub(super) heap: bool,
    pub(super) start: u64,
    pub(super) len: u64,
}

impl Span {
    fn meets(&self, low: u64, high: u64) -> bool {
        self.start < high && low < self.start.saturating_add(self.len)
    }

    /// `#OBJECT buffer +OFFSET`, the address's offset from the start (`-` where it lies below).
    fn name(&self, address: u64) -> String {
        let kind = if self.heap { "heap" } else { "buffer" };
        match address.checked_sub(self.start) {
            Some(offset) => format!("#{} {kind} +{offset:#x}", self.object),
            None => format!("#{} {kind} -{:#x}", self.object, self.start - address),
        }
    }
}

/// The released buffers and heaps, oldest first, each with the frame and call of its release.
#[derive(Default)]
pub(super) struct Freed(VecDeque<(Span, (u64, u64))>);

impl Freed {
    /// `span` was released at `at` (frame, call).
    pub(super) fn released(&mut self, span: Span, at: (u64, u64)) {
        if self.0.len() == FREED_KEPT {
            self.0.pop_front();
        }
        self.0.push_back((span, at));
    }
}

/// Where `address` lies: ` in PLACE live, ...` and ` in PLACE released at frame F call C, ...`,
/// at most [`SHOWN`] of each with the count of the rest, or ` outside every buffer and heap of
/// the stream`.
pub(super) fn place(address: &rt::FaultAddress, live: &[Span], freed: &Freed) -> String {
    let low = address.address & !address.precision.saturating_sub(1);
    let high = low.saturating_add(address.precision.max(1));
    let mut places = Vec::new();
    // Live ones by address, then object: the table is a map.
    let mut live: Vec<&Span> = live.iter().filter(|span| span.meets(low, high)).collect();
    live.sort_unstable_by_key(|span| (span.start, span.object));
    let freed = (freed.0.iter().rev()).filter(|(span, _)| span.meets(low, high));
    let live: Vec<String> = (live.iter())
        .map(|span| format!("{} live", span.name(address.address)))
        .collect();
    let freed: Vec<String> = freed
        .map(|(span, (frame, call))| {
            let name = span.name(address.address);
            format!("{name} released at frame {frame} call {call}")
        })
        .collect();
    for named in [live, freed] {
        let rest = named.len().saturating_sub(SHOWN);
        places.extend(named.into_iter().take(SHOWN));
        if rest > 0 {
            places.push(format!("{rest} more"));
        }
    }
    match places.is_empty() {
        true => " outside every buffer and heap of the stream".to_owned(),
        false => format!(" in {}", places.join(", ")),
    }
}
