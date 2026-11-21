//! First-use waits by site, without a device: each wait adds to its own site's count, time and
//! longest, and to the total's. The counters keep the last variant makes to end, past their bound
//! the oldest leaving first.

use crate::counters::{Counters, FirstUse, VARIANT_MAKES};
use crate::{DispatchVariant, VariantMake, VariantStart};

/// A wait at each site adds to that site alone and to the total; the longest is each one's own.
#[test]
fn each_first_use_wait_adds_to_its_site_and_the_total() {
    let counters = Counters::new();
    counters.first_use_wait(FirstUse::Variant, 3_000);
    counters.first_use_wait(FirstUse::Variant, 7_000);
    counters.first_use_wait(FirstUse::Lines, 11_000);
    counters.first_use_wait(FirstUse::Points, 5_000);
    counters.first_use_wait(FirstUse::Raster, 13_000);
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
            m.first_use_point_waits,
            m.first_use_point_wait_nanos,
            m.first_use_point_wait_max_nanos
        ),
        (1, 5_000, 5_000)
    );
    assert_eq!(
        (
            m.first_use_raster_waits,
            m.first_use_raster_wait_nanos,
            m.first_use_raster_wait_max_nanos
        ),
        (1, 13_000, 13_000)
    );
    assert_eq!(
        (
            m.first_use_waits,
            m.first_use_wait_nanos,
            m.first_use_wait_max_nanos
        ),
        (5, 39_000, 13_000)
    );
}

/// The counters keep the last 256 variant makes to end, numbered from 0 in that order: filled
/// past the bound, the oldest leave first, and a read answers the kept makes from its number on.
#[test]
fn the_counters_keep_the_last_variant_makes_past_their_bound() {
    let counters = Counters::new();
    let total = VARIANT_MAKES as u64 + 44;
    for content in 0..total {
        counters.variant_made(|number| VariantMake {
            number,
            variant: DispatchVariant::of_content(content),
            start: VariantStart::Ahead,
            began: 2 * content,
            ended: 2 * content + 1,
            made: true,
            waits: 0,
            first_wait: 0,
        });
    }
    let kept = counters.variant_makes(0);
    let numbers: Vec<u64> = kept.iter().map(|make| make.number).collect();
    assert_eq!(numbers, (44..total).collect::<Vec<u64>>());
    assert!(
        (kept.iter()).all(|make| make.variant == DispatchVariant::of_content(make.number)),
        "each make keeps its own key"
    );
    let last: Vec<u64> = (counters.variant_makes(total - 3).iter())
        .map(|make| make.number)
        .collect();
    assert_eq!(last, [total - 3, total - 2, total - 1]);
    assert!(counters.variant_makes(total).is_empty());
}
