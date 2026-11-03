//! The resident set and root tables.
//!
//! The resident set of a draw or dispatch is every bound buffer with a device address and every
//! `Buffer`/`Heap` used resource with both addresses, coalesced (`resident_set`): sorted by
//! guest address, an allocation inside another dropped (the outermost wins: a heap over its
//! sub-buffers), a partially overlapping one trimmed of its overlapping head (host and guest
//! advancing together), and empty ones dropped, of any count (At most 2¹⁶ entries before, else
//! `ResidentSetTooLarge`). Indirect command buffers' records (from `RECORDS_ADDRESS_BASE`) are
//! kept whole however they overlap, one a start: a kernel finds an ID's records only where they
//! start (`indirect_commands::records_address`). Each stage that reads a root table gets its own (`stage_root_table`):
//! its `Psb` entries are that stage's slots; the resident set is shared.

use crate::bind::instantiate::{padded, RootTable};
use crate::bind::state::{BindState, BoundBuffer, ResourceClass};
use crate::indirect_commands::RECORDS_ADDRESS_BASE;

/// One root-table entry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct RootEntry {
    /// The allocation's device address.
    pub host: u64,
    /// The allocation's Metal address.
    pub guest: u64,
    /// The allocation's size, padded to 16 bytes.
    pub size: u64,
    /// The Metal bind offset (parameter entries only; 0 in the resident set).
    pub bind_offset: u64,
}

/// The resident set of `all` allocations: sorted by guest address; an allocation inside another one
/// is dropped (the outermost wins: a heap over its sub-buffers); a partially overlapping one loses
/// its overlapping head (its host and guest addresses advance together, which is exact when both
/// ranges view one mapping); empty allocations are dropped. Records of indirect command buffers
/// (from `RECORDS_ADDRESS_BASE`, where no Metal allocation lies) are kept whole, one for each
/// start (the largest): an ID's records may run past later IDs' starts, and a kernel searches
/// them only at their start, so each start must stay an entry's.
pub(crate) fn resident_set(mut all: Vec<RootEntry>) -> Vec<RootEntry> {
    all.retain(|e| e.size > 0);
    all.sort_by(|a, b| a.guest.cmp(&b.guest).then(b.size.cmp(&a.size)));
    let end = |e: &RootEntry| u128::from(e.guest) + u128::from(e.size);
    let mut out: Vec<RootEntry> = Vec::with_capacity(all.len());
    for mut e in all {
        if let Some(last) = out.last() {
            if e.guest >= RECORDS_ADDRESS_BASE {
                if e.guest == last.guest {
                    continue;
                }
                e.bind_offset = 0;
                out.push(e);
                continue;
            }
            let last_end = end(last);
            if end(&e) <= last_end {
                continue;
            }
            if u128::from(e.guest) < last_end {
                // `last_end - e.guest < e.size`, so the cut fits in `u64`.
                let cut = (last_end - u128::from(e.guest)) as u64;
                e.guest += cut;
                e.host = e.host.wrapping_add(cut);
                e.size -= cut;
            }
        }
        e.bind_offset = 0;
        out.push(e);
    }
    out
}

/// The coalesced resident set of `state`: every bound buffer with a device address (every stage)
/// and every `Buffer`/`Heap` used resource with both addresses, at their padded sizes.
pub(crate) fn resident_entries(state: &BindState) -> Vec<RootEntry> {
    coalesced(resident_allocations(state))
}

/// The coalesced resident set a root table of `state` holds: [`resident_entries`], or, where the
/// consumer points its root tables at a set holding the used resources and the listed ranges
/// (`BindState::resident_apart`), the bound buffers' alone.
pub(crate) fn table_entries(state: &BindState) -> Vec<RootEntry> {
    match state.resident_apart {
        true => coalesced(bound_allocations(state)),
        false => resident_entries(state),
    }
}

/// The resident set of `allocations`, each `(host, guest, length)`, at their padded sizes.
fn coalesced(allocations: impl Iterator<Item = (u64, u64, u64)>) -> Vec<RootEntry> {
    let all = allocations
        .map(|(host, guest, length)| RootEntry {
            host,
            guest,
            size: padded(length),
            bind_offset: 0,
        })
        .collect();
    resident_set(all)
}

/// Every allocation of `state`'s resident set before coalescing, as `(host, guest, length)` at
/// its own length (before the padding): the bound buffers, the used buffers and heaps, and the
/// ranges listed apart (`BindState::resident_ranges`).
fn resident_allocations(state: &BindState) -> impl Iterator<Item = (u64, u64, u64)> + '_ {
    let used = state.used_resources.iter().filter_map(|u| {
        match (u.class, u.gpu_address, u.device_address) {
            (ResourceClass::Buffer | ResourceClass::Heap, Some(guest), Some(host)) => {
                Some((host, guest, u.length))
            }
            _ => None,
        }
    });
    let listed = state
        .resident_ranges
        .iter()
        .map(|range| (range.host, range.guest, range.length));
    bound_allocations(state).chain(used).chain(listed)
}

/// The bound buffers of `state` with a device address (every stage), as `(host, guest, length)`
/// at their own lengths.
fn bound_allocations(state: &BindState) -> impl Iterator<Item = (u64, u64, u64)> + '_ {
    state.stages.values().flat_map(|stage_table| {
        stage_table.buffers.values().filter_map(|b| match *b {
            BoundBuffer::Buffer {
                length,
                gpu_address,
                device_address: Some(host),
                ..
            } => Some((host, gpu_address, length)),
            _ => None,
        })
    })
}

/// The unpadded extent of each entry of `resident` (`state`'s resident set,
/// [`resident_entries`]): how many bytes from the entry's guest address the allocations of `state`
/// hold without a gap at their own lengths (before the padding), at most the entry's size. Every
/// address an entry holds below its extent is one an allocation holds, as a host resolve finds it
/// (`BufferBinding::resolve_buffer`); the converse fails only for an address past a gap inside one
/// padded window, which only allocations whose guest addresses are not 16-aligned leave. Sorting,
/// then one forward pass per entry over the allocations that start inside its window (the windows
/// are disjoint but for records, which hold few allocations), bounds the work by the allocations'
/// count times its logarithm.
pub(crate) fn resident_extents(state: &BindState, resident: &[RootEntry]) -> Vec<u64> {
    let allocations = resident_allocations(state).map(|(_, guest, length)| (guest, length));
    extents_of(allocations, resident)
}

/// The unpadded extents a replay preprocess reads right after its resident set
/// (`memory::roots::ResidentView`): a `u64` per entry of `resident`, `state`'s resident set
/// ([`resident_extents`]), then one zero, which a search that found no entry reads.
pub(crate) fn replay_extents(state: &BindState, resident: &[RootEntry]) -> Vec<u8> {
    extents_bytes(resident_extents(state, resident))
}

/// `extents`, little-endian, then one zero.
pub(crate) fn extents_bytes(extents: impl IntoIterator<Item = u64>) -> Vec<u8> {
    let mut bytes: Vec<u8> = extents.into_iter().flat_map(u64::to_le_bytes).collect();
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes
}

/// [`resident_extents`] of the entries `resident` over the allocations `allocations`, each its
/// `(guest, length)` at its own length.
pub(crate) fn extents_of(
    allocations: impl Iterator<Item = (u64, u64)>,
    resident: &[RootEntry],
) -> Vec<u64> {
    let mut ranges: Vec<(u128, u128)> = allocations
        .filter(|&(_, length)| length > 0)
        .map(|(guest, length)| (u128::from(guest), u128::from(guest) + u128::from(length)))
        .collect();
    ranges.sort_unstable();
    // The furthest end of the allocations up to each one.
    let mut furthest = Vec::with_capacity(ranges.len());
    let mut reach = 0u128;
    for &(_, end) in &ranges {
        reach = reach.max(end);
        furthest.push(reach);
    }
    resident
        .iter()
        .map(|e| {
            let start = u128::from(e.guest);
            let limit = start + u128::from(e.size);
            // The allocations starting at or below the entry: any that reaches past its start
            // holds it.
            let below = ranges.partition_point(|&(guest, _)| guest <= start);
            let mut cover = match below {
                0 => start,
                n => furthest[n - 1].max(start),
            };
            for &(guest, end) in &ranges[below..] {
                if guest > cover || cover >= limit {
                    break;
                }
                cover = cover.max(end);
            }
            // `cover - start` is at most `size`, a `u64`.
            (cover.min(limit) - start) as u64
        })
        .collect()
}

/// The root table of one stage: its `Psb` entries (`(n, entry)` for `[[buffer(n)]]`) and the
/// coalesced `resident` set from root `first`.
pub(crate) fn stage_root_table(
    statics: &[(u32, RootEntry)],
    first: u32,
    resident: &[RootEntry],
) -> RootTable {
    RootTable::build(statics, u64::from(first), resident)
}

#[cfg(test)]
mod tests;
