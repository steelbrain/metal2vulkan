//! A lost device's fault report keeps the driver's addresses as reported, the instruction pointers
//! apart from the accesses, each list in the driver's order.

use crate::fault::{split, FaultAddress};
use ash::vk::{DeviceFaultAddressInfoEXT, DeviceFaultAddressTypeEXT as Kind};

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
