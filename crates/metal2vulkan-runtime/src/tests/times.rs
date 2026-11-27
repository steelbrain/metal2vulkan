//! A command buffer's times (`times.rs`) without a device: the durations of the GPU's ticks across
//! the wrap of their valid bits; timestamps placed on the host clock by a calibration either side
//! of its reading, and by the completion seen where the calibration would place them after it;
//! and the host clock they are placed on rising.

use crate::times::{Calibration, Ticks};

/// A calibration that read the GPU's tick `tick` at host nanosecond `host`.
fn calibration(tick: u64, host: u64) -> Calibration {
    Calibration { tick, host }
}

#[test]
fn durations_count_ticks_of_the_period_across_a_wrap() {
    let ticks = Ticks::new(52.083_332, 36);
    assert_eq!(ticks.span(1_000, 1_000), 0);
    assert_eq!(ticks.span(1_000, 1_096), 5_000);
    // From 10 ticks before the 36 bits wrap to 5 after: 15 ticks.
    let before = (1u64 << 36) - 10;
    assert_eq!(ticks.span(before, 5), 781);
    // An end before the start is no duration.
    assert_eq!(ticks.span(1_096, 1_000), 0);
    let full = Ticks::new(1.0, 64);
    assert_eq!(full.span(u64::MAX - 1, 3), 5);
}

#[test]
fn a_calibration_places_timestamps_on_the_hosts_clock_either_side_of_it() {
    let ticks = Ticks::new(2.0, 64);
    let pair = calibration(1_000, 5_000_000);
    assert_eq!(pair.at(&ticks, 1_000), 5_000_000);
    assert_eq!(pair.at(&ticks, 1_500), 5_001_000);
    assert_eq!(pair.at(&ticks, 500), 4_999_000);
    // Across the bits' wrap, the nearer direction: 8 bits, 6 ticks after 250 is 0 after the wrap.
    let narrow = Ticks::new(1.0, 8);
    let pair = calibration(250, 1_000);
    assert_eq!(pair.at(&narrow, 0), 1_006);
    assert_eq!(pair.at(&narrow, 244), 994);
}

/// A command buffer's two timestamps that the calibration places before the completion seen are
/// the calibration's placement.
#[test]
fn a_span_is_placed_by_the_calibration() {
    let ticks = Ticks::new(1.0, 64);
    let mut reading = calibration(100, 10_000);
    assert_eq!(reading.place(&ticks, (150, 400), 99_999), (10_050, 10_300));
}

/// A calibration 65 s off, as a Radeon 780M's after two MODE2 resets (the device's timestamps run
/// 65.15 s past the domain the calibration read), would place the times far past the completion
/// seen: they are placed by the completion, the end at it and the start their exact span before.
#[test]
fn a_calibration_65_s_ahead_places_times_by_the_completion() {
    let ticks = Ticks::new(1.0, 64);
    // The work ran 1_000 ns from 2_000_100 to 2_001_100 (ticks equal host nanoseconds by the
    // sound calibration), and its completion was seen at 2_002_000.
    let (start, end, seen) = (2_000_100, 2_001_100, 2_002_000);
    let ahead = 65_152_099_677;
    let mut off = calibration(1_000_000, 1_000_000);
    assert_eq!(
        off.place(&ticks, (start + ahead, end + ahead), seen),
        (2_001_000, 2_002_000)
    );
}

#[test]
fn the_host_time_rises() {
    let first = crate::host_time();
    let second = crate::host_time();
    assert!(first > 0 && second >= first, "{first} then {second}");
}
