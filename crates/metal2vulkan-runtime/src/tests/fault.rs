//! A lost device's fault report keeps the driver's addresses as reported, the instruction pointers
//! apart from the accesses, each list in the driver's order.

use crate::fault::{split, FaultAddress, RangeKind, Spans, FREED};
use crate::BlockId;
use ash::vk::{self, DeviceFaultAddressInfoEXT, DeviceFaultAddressTypeEXT as Kind, Handle};

#[test]
fn instruction_pointers_stay_apart_from_accesses_as_reported() {
    let at = |kind, address, precision| FaultAddress {
        kind,
        address,
        precision,
    };
    // An RTX 3090's report of a read through a freed buffer's address (job m3-max-62203), and a
    // write after it.
    let reported = [
        at(Kind::INSTRUCTION_POINTER_FAULT, 0xdefdfeb950, 0x10),
        at(Kind::READ_INVALID, 0xdef63a3000, 0x1000),
        at(Kind::WRITE_INVALID, 0xdef63a5123, 0x1000),
    ];
    let infos = reported.map(|a| DeviceFaultAddressInfoEXT {
        address_type: a.kind,
        reported_address: a.address,
        address_precision: a.precision,
    });
    let (accesses, instructions) = split(&infos);
    assert_eq!(
        accesses,
        reported[1..],
        "the accesses, in order, as reported"
    );
    assert_eq!(
        instructions,
        reported[..1],
        "the instruction pointer, as reported"
    );
}

/// Two opaque handles' surface allocations, blocks of no buffer and address 0, are not noted: an
/// address in their first bytes lies in no range, and a later range is placed as noted.
#[test]
fn a_block_of_no_buffer_is_placed_nowhere() {
    let spans = Spans::default();
    spans.made(
        vk::Buffer::null(),
        0,
        0x180_0000,
        RangeKind::Export(BlockId(1)),
    );
    spans.made(
        vk::Buffer::null(),
        0,
        0x260_0000,
        RangeKind::Export(BlockId(2)),
    );
    spans.made(
        vk::Buffer::from_raw(7),
        0x4000_0000,
        0x10_0000,
        RangeKind::Import,
    );
    let place = spans.place(
        &FaultAddress {
            kind: Kind::READ_INVALID,
            address: 0x10_0000,
            precision: 0x1000,
        },
        0,
    );
    assert_eq!(
        (place.holders.len(), place.below),
        (0, None),
        "no range at address 0"
    );
    assert_eq!(
        place.above.map(|range| (range.kind, range.address)),
        Some((RangeKind::Import, 0x4000_0000)),
        "the import above"
    );
    spans.freed(vk::Buffer::null(), 0, None);
    assert!(spans.freed_kept().is_empty(), "nothing of no buffer freed");
}

/// The device's book of its ranges keeps the last [`FREED`] freed, the oldest dropped first, and
/// places an address among the ranges that meet its bytes, the live ones first, then the freed
/// ones newest first, with the nearest range wholly below and wholly above them.
#[test]
fn an_address_is_placed_among_the_live_and_the_last_freed_ranges() {
    const MIB: u64 = 1 << 20;
    const BASE: u64 = 0x1000_0000;
    let spans = Spans::default();
    let buffer = vk::Buffer::from_raw;
    // Three past the bound, each of 1 MiB at its own address, freed with 2 in flight.
    let made = FREED as u64 + 3;
    for n in 0..made {
        spans.made(buffer(n + 1), BASE + n * MIB, MIB, RangeKind::Import);
        spans.freed(buffer(n + 1), 2, None);
    }
    let kept = spans.freed_kept();
    assert_eq!(kept.len(), FREED, "the freed ranges kept, at the bound");
    assert_eq!(kept[0].address, BASE + 3 * MIB, "the oldest three dropped");
    // A live range over the newest freed one's bytes, and one far above.
    let newest = BASE + (made - 1) * MIB;
    spans.made(buffer(made + 1), newest, 2 * MIB, RangeKind::RingBlock);
    spans.made(
        buffer(made + 2),
        0x9000_0000,
        MIB,
        RangeKind::Export(BlockId(9)),
    );
    spans.freed(buffer(made + 100), 0, None);
    let at = |address, precision| FaultAddress {
        kind: Kind::READ_INVALID,
        address,
        precision,
    };
    let place = spans.place(&at(newest + 0x1234, 0x1000), 0);
    let holders: Vec<_> = (place.holders.iter())
        .map(|range| (range.kind, range.address, range.freed.map(|f| f.in_flight)))
        .collect();
    assert_eq!(
        holders,
        [
            (RangeKind::RingBlock, newest, None),
            (RangeKind::Import, newest, Some(2)),
        ],
        "the live range, then the freed one"
    );
    let below = place
        .below
        .map(|range| (range.address, range.freed.is_some()));
    assert_eq!(below, Some((newest - MIB, true)), "the freed one below");
    let above = place.above.map(|range| (range.kind, range.freed));
    assert_eq!(
        above,
        Some((RangeKind::Export(BlockId(9)), None)),
        "the live one above, by its block's number"
    );
    let line = place.to_string();
    assert!(
        line.starts_with(&format!(
            "{:#x}: in parameter ring block [{newest:#x}, +0x200000) made ",
            newest + 0x1234
        )),
        "{line}"
    );
    assert!(
        line.contains(", freed 0.000 s before with 2 command buffers in flight;"),
        "{line}"
    );
    assert!(
        line.contains("; nearest above: export block 9 [0x90000000, +0x100000) made "),
        "{line}"
    );
    let outside = spans.place(&at(0x100, 0), 0);
    assert_eq!(
        (outside.precision, outside.holders.len(), outside.below),
        (1, 0, None),
        "no range meets it, and none lies below"
    );
    assert_eq!(
        outside.above.map(|range| range.address),
        Some(BASE + 3 * MIB),
        "the oldest kept lies above"
    );
    assert!(
        (outside.to_string())
            .starts_with("0x100: in none of the runtime's ranges; nearest above: import"),
        "{outside}"
    );
}
