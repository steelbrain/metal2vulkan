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
use std::cmp::Ordering;

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
pub(crate) fn resident_set(all: Vec<RootEntry>) -> Vec<RootEntry> {
    coalesced_of_sorted(&sorted(all))
}

/// `all` as a resident set takes its allocations: the empty ones dropped, the rest sorted by guest
/// address, the larger first at one address, else in their order (a stable sort).
pub(crate) fn sorted(mut all: Vec<RootEntry>) -> Vec<RootEntry> {
    all.retain(|e| e.size > 0);
    all.sort_by(order);
    all
}

/// The order [`sorted`] takes allocations in.
fn order(a: &RootEntry, b: &RootEntry) -> Ordering {
    a.guest.cmp(&b.guest).then(b.size.cmp(&a.size))
}

/// The resident set of `sorted`, allocations in [`sorted`]'s order.
pub(crate) fn coalesced_of_sorted(sorted: &[RootEntry]) -> Vec<RootEntry> {
    let mut out: Vec<RootEntry> = Vec::with_capacity(sorted.len());
    for &e in sorted {
        coalesce(&mut out, e);
    }
    out
}

/// Add `e`, an allocation at or after every one `out` was made of in [`sorted`]'s order, to `out`,
/// the resident set of those: dropped where it lies inside the last entry, or a record's at the
/// last entry's start; trimmed of its head where it overlaps the last entry; else whole. Whether
/// it went in whole.
fn coalesce(out: &mut Vec<RootEntry>, mut e: RootEntry) -> bool {
    let end = |e: &RootEntry| u128::from(e.guest) + u128::from(e.size);
    e.bind_offset = 0;
    let Some(last) = out.last() else {
        out.push(e);
        return true;
    };
    if e.guest >= RECORDS_ADDRESS_BASE {
        if e.guest == last.guest {
            return false;
        }
        out.push(e);
        return true;
    }
    let last_end = end(last);
    if end(&e) <= last_end {
        return false;
    }
    if u128::from(e.guest) >= last_end {
        out.push(e);
        return true;
    }
    // `last_end - e.guest < e.size`, so the cut fits in `u64`.
    let cut = (last_end - u128::from(e.guest)) as u64;
    e.guest += cut;
    e.host = e.host.wrapping_add(cut);
    e.size -= cut;
    out.push(e);
    false
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
