//! First-use waits by site, without a device: each wait adds to its own site's count, time and
//! longest, and to the total's.

use crate::counters::{Counters, FirstUse};

/// A wait at each site adds to that site alone and to the total; the longest is each one's own.
#[test]
fn each_first_use_wait_adds_to_its_site_and_the_total() {
    let counters = Counters::new();
    counters.first_use_wait(FirstUse::Variant, 3_000);
    counters.first_use_wait(FirstUse::Variant, 7_000);
    counters.first_use_wait(FirstUse::Lines, 11_000);
    let m = counters.metrics();
    assert_eq!(
        (
            m.first_use_variant_waits,
            m.first_use_variant_wait_nanos,
            m.first_use_variant_wait_max_nanos
        ),
        (2, 10_000, 7_000)
    );
    assert_eq!(
        (
            m.first_use_line_waits,
            m.first_use_line_wait_nanos,
            m.first_use_line_wait_max_nanos
        ),
        (1, 11_000, 11_000)
    );
    assert_eq!(
        (
            m.first_use_waits,
            m.first_use_wait_nanos,
            m.first_use_wait_max_nanos
        ),
        (3, 21_000, 11_000)
    );
}
