//! `residency.rs`: the index a residency looks a Metal address up in finds what a look at every
//! buffer made resident, in the order made resident, finds, without a device.

use super::super::residency::AddressIndex;

/// What a look at each of `ranges` (`(start, length, token)`, in token order) finds of `address`:
/// the first that holds it, else the first whose end it is.
fn scanned(ranges: &[(u64, u64, u32)], address: u64) -> Option<u32> {
    let offset = |&(start, length, token): &(u64, u64, u32)| {
        let offset = address.checked_sub(start)?;
        (offset <= length).then_some((offset, length, token))
    };
    let inside = (ranges.iter().filter_map(offset)).find(|&(offset, length, _)| offset < length);
    inside
        .or_else(|| ranges.iter().filter_map(offset).next())
        .map(|(_, _, token)| token)
}

/// Every address at, beside and between the ranges' starts and ends.
fn probes(ranges: &[(u64, u64, u32)]) -> Vec<u64> {
    let mut probes: Vec<u64> = ranges
        .iter()
        .flat_map(|&(start, length, _)| [start, start.saturating_add(length)])
        .flat_map(|at| [at.saturating_sub(1), at, at.saturating_add(1)])
        .collect();
    probes.extend([0, 1, u64::MAX]);
    probes.sort_unstable();
    probes.dedup();
    probes
}

/// The index finds what the look finds: in disjoint buffers, adjacent ones (an address at one's
/// end and the next's start), nested ones (a heap over buffers placed in it, made resident before
/// them and after), overlapping ones, empty ones, and many pseudo-random sets of them.
#[test]
fn the_index_finds_the_first_made_resident_holding_an_address() {
    let sets: Vec<Vec<(u64, u64, u32)>> = vec![
        vec![],
        vec![(0x1000, 0x100, 0), (0x2000, 0x40, 1), (0x3000, 0x10, 2)],
        vec![(0x1000, 0x100, 0), (0x1100, 0x100, 1)],
        // A heap's own buffer, then buffers placed in it; and buffers, then their heap.
        vec![
            (0x10000, 0x1000, 0),
            (0x10000, 0x100, 1),
            (0x10800, 0x100, 2),
        ],
        vec![
            (0x10000, 0x100, 0),
            (0x10800, 0x100, 1),
            (0x10000, 0x1000, 2),
        ],
        vec![(0x1000, 0x200, 0), (0x1100, 0x200, 1), (0x1180, 0x10, 2)],
        vec![(0x1000, 0, 0), (0x1000, 0x10, 1), (0x2000, 0, 2)],
        vec![(u64::MAX - 0x10, 0x10, 0), (u64::MAX - 0x20, 0x30, 1)],
    ];
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move |bound: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % bound
    };
    let random: Vec<Vec<(u64, u64, u32)>> = (0..200)
        .map(|_| {
            let count = next(24) as u32;
            (0..count)
                .map(|token| (0x1000 + 0x10 * next(64), 0x10 * next(8), token))
                .collect()
        })
        .collect();
    for ranges in sets.iter().chain(&random) {
        let index = AddressIndex::of(ranges.iter().copied());
        for address in probes(ranges) {
            assert_eq!(
                index.first_holding(address),
                scanned(ranges, address),
                "{address:#x} in {ranges:x?}"
            );
        }
    }
}
