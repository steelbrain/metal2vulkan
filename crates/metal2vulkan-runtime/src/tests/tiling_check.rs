//! The tiling-cost check (`tiling_check.rs`), the parts that need no device: its verdict on fixed
//! medians, on both sides of each of its bounds, its medians, and the record that keeps its
//! answer.
//!
//! - **The verdict.** Linear costly only where the linear frame passes the optimal one by more
//!   than `COSTLY_EXCESS` and more than `COSTLY_RATIO` times: either bound alone answers cheap.
//! - **The record.** Its medians read back for its identity alone, whole and unchanged, and
//!   never a record of no frames.

use crate::in_place::DriverIdentity;
use crate::tiling_check::{
    decode, encode, median, record_name, verdict_of, Medians, COSTLY_EXCESS, COSTLY_RATIO,
};
use crate::TilingVerdict;

/// `micros` microseconds, in nanoseconds.
fn us(micros: u64) -> u64 {
    micros * 1_000
}

#[test]
fn the_verdict_is_costly_only_past_both_bounds() {
    let excess = u64::try_from(COSTLY_EXCESS.as_nanos()).unwrap();
    // A 1 ms optimal frame: the ratio bound (2 ms) is past the excess bound (2 ms is 1 ms more):
    // just past both, costly; at the ratio, not.
    let optimal = us(1_000);
    let at_ratio = (optimal as f64 * COSTLY_RATIO) as u64;
    assert_eq!(
        verdict_of(at_ratio + 1, optimal),
        TilingVerdict::LinearCostly
    );
    assert_eq!(verdict_of(at_ratio, optimal), TilingVerdict::LinearCheap);
    // A 10 ms optimal frame: 0.5 ms more passes the excess bound and not the ratio.
    let optimal = us(10_000);
    assert_eq!(
        verdict_of(optimal + excess + 1, optimal),
        TilingVerdict::LinearCheap
    );
    assert_eq!(
        verdict_of((optimal as f64 * COSTLY_RATIO) as u64 + 1, optimal),
        TilingVerdict::LinearCostly
    );
    // A 0.1 ms optimal frame: four times it passes the ratio and not the excess bound; past
    // the excess bound, costly.
    let optimal = us(100);
    assert_eq!(verdict_of(us(400), optimal), TilingVerdict::LinearCheap);
    assert_eq!(
        verdict_of(optimal + excess, optimal),
        TilingVerdict::LinearCheap
    );
    assert_eq!(
        verdict_of(optimal + excess + 1, optimal),
        TilingVerdict::LinearCostly
    );
    // A linear frame faster than the optimal one is cheap.
    assert_eq!(verdict_of(us(500), us(4_000)), TilingVerdict::LinearCheap);
}

#[test]
fn the_median_is_the_upper_middle_one() {
    assert_eq!(median(&mut [5, 1, 9, 3, 7]), 5);
    assert_eq!(median(&mut [4, 1, 3, 2]), 3);
    assert_eq!(median(&mut []), 0);
}

fn identity() -> DriverIdentity {
    DriverIdentity {
        vendor_id: 0x8086,
        device_id: 0xa7a0,
        driver_version: 0x0068_2006,
        driver_id: 6,
        driver_name: "Intel open-source Mesa driver".to_owned(),
        driver_info: "Mesa 26.2.6".to_owned(),
        device_uuid: [3; 16],
    }
}

#[test]
fn the_record_reads_back_its_medians_for_its_identity_alone() {
    let identity = identity();
    let medians = Medians {
        linear: us(4_120),
        optimal: us(1_010),
        download: us(2_200),
        upload: us(2_050),
    };
    let bytes = encode(&identity, &medians);
    assert_eq!(decode(&identity, &bytes), Some(medians));
    assert_eq!(
        decode(&identity, &bytes).map(|m| m.cost().verdict),
        Some(TilingVerdict::LinearCostly)
    );
    let mut changed = bytes.clone();
    changed[60] ^= 1;
    assert_eq!(decode(&identity, &changed), None, "a byte changed");
    assert_eq!(decode(&identity, &bytes[1..]), None, "cut short");
    let mut other = identity.clone();
    other.driver_info.push('x');
    assert_eq!(decode(&other, &bytes), None, "another identity");
    assert_ne!(record_name(&identity), record_name(&other));
    let none = Medians {
        linear: 0,
        ..medians
    };
    assert_eq!(
        decode(&identity, &encode(&identity, &none)),
        None,
        "no frames"
    );
    // Its name is not the other checks' of the same identity.
    assert_ne!(
        record_name(&identity),
        crate::queue_share::record_name(&identity)
    );
    assert_ne!(
        record_name(&identity),
        crate::in_place::record_name(&identity)
    );
}
