//! A command buffer's times (`times.rs`) without a device: the durations of the GPU's ticks across
//! the wrap of their valid bits, and the host clock they are placed on rising.

use crate::times::Ticks;

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
fn the_host_time_rises() {
    let first = crate::host_time();
    let second = crate::host_time();
    assert!(first > 0 && second >= first, "{first} then {second}");
}
