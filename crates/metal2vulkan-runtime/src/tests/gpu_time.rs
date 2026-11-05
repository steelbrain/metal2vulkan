//! A command buffer's GPU times (`gpu_time.rs`): timestamps placed on the host's clock by a
//! calibration pair or by the completion, across a wrap of the timestamps' bits; a pair that
//! places them outside the command buffer's life by more than its deviation and the clocks'
//! drift allow set aside for the completion; the pair, the
//! reading of the least deviation; the host time domain a calibration reads; the extension's
//! preferred form; and the device's timestamp pools, their slots held by the command buffers in
//! flight, made as those need them up to the most.

use crate::gpu_time::{
    host_domain, host_time, least_deviation, Extension, GpuClock, Life, Pair, Placement, Pool,
    Pools, Reading, DRIFT_PPM, PLAIN, PREFIXED, SLOTS,
};
use ash::vk;
use std::collections::BTreeSet;
use std::ffi::CString;

/// An uncalibrated clock of `valid_bits` bits, `period` nanoseconds a tick.
fn clock(valid_bits: u32, period: f32) -> GpuClock {
    GpuClock::new(valid_bits, period, None).expect("a clock with timestamps")
}

#[test]
fn a_family_without_timestamp_bits_has_no_clock() {
    assert!(GpuClock::new(0, 1.0, None).is_none());
    assert_eq!(clock(36, 1.0).valid_bits, 36);
    // More bits than a timestamp holds are its 64.
    assert_eq!(clock(70, 1.0).valid_bits, 64);
}

#[test]
fn durations_count_ticks_of_the_period_across_a_wrap() {
    let clock = clock(36, 52.083_332);
    assert_eq!(clock.duration(1_000, 1_000), 0);
    assert_eq!(clock.duration(1_000, 1_096), 5_000);
    // From 10 ticks before the 36 bits wrap to 5 after: 15 ticks.
    let before = (1u64 << 36) - 10;
    assert_eq!(clock.duration(before, 5), 781);
    // An end before the start is no duration.
    assert_eq!(clock.duration(1_096, 1_000), 0);
    let full = self::clock(64, 1.0);
    assert_eq!(full.duration(u64::MAX - 1, 3), 5);
}

#[test]
fn a_pair_places_timestamps_on_the_hosts_clock_either_side_of_it() {
    let clock = clock(64, 2.0);
    let pair = Pair {
        ticks: 1_000,
        host: 5_000_000,
    };
    assert_eq!(clock.place(pair, 1_000), 5_000_000);
    assert_eq!(clock.place(pair, 1_500), 5_001_000);
    assert_eq!(clock.place(pair, 500), 4_999_000);
    // Across the bits' wrap, the nearer direction: 8 bits, 6 ticks after 250 is 0 after the wrap.
    let narrow = self::clock(8, 1.0);
    let pair = Pair {
        ticks: 250,
        host: 1_000,
    };
    assert_eq!(narrow.place(pair, 0), 1_006);
    assert_eq!(narrow.place(pair, 244), 994);
}

/// A calibration pair is read again while each reading halves the deviation the one before it
/// stated, and while the smallest so far states more than `SETTLED` (20 us), `READINGS` (4) at
/// most, and the reading of the smallest deviation is kept: the Iris Xe's first reading after an
/// idle GPU (165 us) is followed by one of 7.6 us, kept, and one more that halves nothing.
/// Readings a preempted thread widened (about 70 us each in turn on an RX 7900 XTX under a full
/// parallel suite) go on though they halve nothing, until one settles, or keep the least of four.
/// A failed reading ends them, the least so far kept; a failed first is none.
#[test]
fn a_pair_is_the_reading_of_the_least_deviation() {
    let least = |deviations: &[Option<u64>]| {
        let mut readings = deviations.iter().enumerate().map(|(at, deviation)| {
            deviation.map(|deviation| Reading {
                pair: Pair {
                    ticks: at as u64,
                    host: 1_000 * at as u64,
                },
                deviation,
            })
        });
        least_deviation(|| readings.next().flatten())
            .map(|(reading, calls)| (reading.pair.ticks, reading.deviation, calls))
    };
    assert_eq!(
        least(&[Some(165_000), Some(7_600), Some(7_600)]),
        Some((1, 7_600, 3))
    );
    assert_eq!(least(&[Some(7_600), Some(7_500)]), Some((1, 7_500, 2)));
    assert_eq!(least(&[Some(7_500), Some(7_600)]), Some((0, 7_500, 2)));
    assert_eq!(least(&[Some(30_000), Some(20_000)]), Some((1, 20_000, 2)));
    assert_eq!(
        least(&[Some(100), Some(40), Some(15), Some(6), Some(1)]),
        Some((3, 6, 4))
    );
    // Wide readings that halve nothing: read again until one settles, four at most.
    assert_eq!(
        least(&[Some(70_100), Some(69_500), Some(6_000), Some(5_000)]),
        Some((3, 5_000, 4))
    );
    assert_eq!(
        least(&[
            Some(70_100),
            Some(69_500),
            Some(80_000),
            Some(65_000),
            Some(1)
        ]),
        Some((3, 65_000, 4))
    );
    assert_eq!(
        least(&[Some(90_000), Some(40_000), Some(60_000), Some(50_000)]),
        Some((1, 40_000, 4))
    );
    assert_eq!(least(&[Some(100_000), Some(60_000)]), Some((1, 60_000, 3)));
    assert_eq!(least(&[Some(100), None, Some(1)]), Some((0, 100, 2)));
    assert_eq!(least(&[None, Some(1)]), None);
}

/// A command buffer's life from `kernel_start` to `completed`.
fn life(kernel_start: u64, completed: u64) -> Life {
    Life {
        kernel_start,
        completed,
    }
}

#[test]
fn a_span_is_placed_by_the_pair_or_ends_at_the_completion() {
    let clock = clock(64, 1.0);
    let reading = Reading {
        pair: Pair {
            ticks: 100,
            host: 10_000,
        },
        deviation: 0,
    };
    assert_eq!(
        clock.span(Some(reading), 150, 400, life(10_000, 99_999)),
        ((10_050, 10_300), Placement::Calibrated)
    );
    // Uncalibrated: the end is the completion seen, the start its exact duration before.
    assert_eq!(
        clock.span(None, 150, 400, life(0, 20_000)),
        ((19_750, 20_000), Placement::Uncalibrated)
    );
    assert_eq!(
        clock.span(None, 150, 400, life(0, 100)),
        ((0, 100), Placement::Uncalibrated)
    );
}

/// A pair whose placement leaves the command buffer's life (before its kernel start, or after
/// its completion seen) by more than its stated deviation and the drift since it
/// ([`DRIFT_PPM`] of the time from the pair to that bound) disagrees with the device's
/// timestamps: the times are placed by the completion, as without a pair. Within the allowance
/// they stay the pair's. A pair 65 s off, as a Radeon 780M's after two MODE2 resets: every time
/// set aside, however the drift is counted.
#[test]
fn a_pair_that_places_times_outside_the_life_is_set_aside() {
    let clock = clock(64, 1.0);
    let at = |ticks: u64, host: u64, deviation: u64| Reading {
        pair: Pair { ticks, host },
        deviation,
    };
    // The work ran 1_000 ns from 2_000_100 to 2_001_100 (ticks equal host nanoseconds by the
    // sound pair), inside a life from 2_000_000 to 2_002_000, 1 ms after the pair.
    let sound = at(1_000_000, 1_000_000, 50);
    let (start, end, inside) = (2_000_100, 2_001_100, life(2_000_000, 2_002_000));
    let by_completion = ((2_001_000, 2_002_000), Placement::Uncalibrated);
    assert_eq!(
        clock.span(Some(sound), start, end, inside),
        ((2_000_100, 2_001_100), Placement::Calibrated)
    );
    // The drift since the pair to the bound: 500 ppm of 1 ms, 500 ns, beside the 50 ns stated.
    let allowance = 50 + 1_000_000 * DRIFT_PPM / 1_000_000;
    assert_eq!(allowance, 550);
    // Early by the allowance and no more: the pair's.
    let early = at(1_000_000 + 100 + allowance, 1_000_000, 50);
    assert_eq!(
        clock.span(Some(early), start, end, inside),
        (
            (2_000_000 - allowance, 2_001_000 - allowance),
            Placement::Calibrated
        )
    );
    // Early by one more: set aside, by its completion.
    let earlier = at(1_000_000 + 100 + allowance + 1, 1_000_000, 50);
    assert_eq!(
        clock.span(Some(earlier), start, end, inside),
        (
            by_completion.0,
            Placement::Disagreed {
                by: allowance + 1,
                deviation: 50
            }
        )
    );
    // Late past the completion by the allowance at the end (50 ns, and the drift to 2_002_000,
    // 1.002 ms after the pair: 501 ns), and by one more.
    let allowance = 50 + 1_002_000 * DRIFT_PPM / 1_000_000;
    assert_eq!(allowance, 551);
    let late = at(1_000_000 - 900 - allowance, 1_000_000, 50);
    assert_eq!(
        clock.span(Some(late), start, end, inside),
        ((2_001_551, 2_002_551), Placement::Calibrated)
    );
    let later = at(1_000_000 - 900 - allowance - 1, 1_000_000, 50);
    assert_eq!(
        clock.span(Some(later), start, end, inside),
        (
            by_completion.0,
            Placement::Disagreed {
                by: allowance + 1,
                deviation: 50
            }
        )
    );
    // 65 s ahead: the device's timestamps run 65.15 s past the domain the pair read.
    let ahead = 65_152_099_677;
    let off = at(1_000_000, 1_000_000, 3_578);
    let (times, placement) = clock.span(Some(off), start + ahead, end + ahead, inside);
    assert_eq!(times, by_completion.0);
    assert!(
        matches!(placement, Placement::Disagreed { by, deviation: 3_578 } if by > 65_000_000_000),
        "{placement:?}"
    );
    // 65 s behind, the start before the kernel start: set aside too.
    let behind = at(1_000_000 + ahead, 1_000_000, 3_578);
    assert!(matches!(
        clock.span(Some(behind), start, end, inside).1,
        Placement::Disagreed { .. }
    ));
}

#[cfg(unix)]
#[test]
fn calibration_reads_the_devices_domain_and_a_monotonic_host_one() {
    use vk::TimeDomainKHR as D;
    // `TimeDomainKHR` has no `Debug` without ash's `debug` feature: compared, not printed.
    let chosen = |offered: &[D]| host_domain(offered);
    assert!(chosen(&[D::DEVICE, D::CLOCK_MONOTONIC]) == Some(D::CLOCK_MONOTONIC));
    // The raw clock (the host's own, off macOS) before the slewed one.
    assert!(
        chosen(&[D::CLOCK_MONOTONIC, D::DEVICE, D::CLOCK_MONOTONIC_RAW])
            == Some(D::CLOCK_MONOTONIC_RAW)
    );
    assert!(chosen(&[D::CLOCK_MONOTONIC_RAW]).is_none());
    assert!(chosen(&[D::DEVICE, D::QUERY_PERFORMANCE_COUNTER]).is_none());
    assert!(chosen(&[]).is_none());
}

#[test]
fn the_khr_form_is_preferred_where_both_are_listed() {
    let both: BTreeSet<CString> = [
        vk::EXT_CALIBRATED_TIMESTAMPS_NAME,
        vk::KHR_CALIBRATED_TIMESTAMPS_NAME,
    ]
    .into_iter()
    .map(|name| name.to_owned())
    .collect();
    let preferred = Extension::among(&both).expect("listed");
    assert_eq!(preferred.name(), vk::KHR_CALIBRATED_TIMESTAMPS_NAME);
    let ext: BTreeSet<CString> = [vk::EXT_CALIBRATED_TIMESTAMPS_NAME.to_owned()].into();
    assert_eq!(
        Extension::among(&ext).expect("listed").name(),
        vk::EXT_CALIBRATED_TIMESTAMPS_NAME
    );
    assert!(Extension::among(&BTreeSet::new()).is_none());
}

#[test]
fn the_host_time_rises() {
    let first = host_time();
    let second = host_time();
    assert!(first > 0.0 && second >= first, "{first} then {second}");
}

#[test]
fn an_uncalibrated_clock_takes_no_pair() {
    assert_eq!(clock(64, 1.0).pair(), None);
}

/// The pool hands out each slot once, its queries apart, slot 0 first; none once every slot is
/// held; and a slot dropped is given back, so the slots held are the command buffers in flight,
/// however many were committed. A slot of three queries has the first encoder's start after its
/// end; one of two has none.
#[test]
fn a_slot_is_held_until_dropped_and_then_given_back() {
    let plain = Pool::new(vk::QueryPool::null(), SLOTS, PLAIN);
    let slot = plain.take().unwrap();
    assert_eq!((slot.first(), slot.prefix()), (0, None));
    let slot = plain.take().unwrap();
    assert_eq!((slot.first(), slot.prefix()), (2, None));
    let pool = Pool::new(vk::QueryPool::null(), SLOTS, PREFIXED);
    let held: Vec<_> = (0..SLOTS).map(|_| pool.take().unwrap()).collect();
    let firsts: Vec<(u32, Option<u32>)> = held
        .iter()
        .map(|slot| (slot.first(), slot.prefix()))
        .collect();
    assert_eq!(
        firsts,
        (0..SLOTS)
            .map(|index| (3 * index, Some(3 * index + 2)))
            .collect::<Vec<_>>()
    );
    assert!(pool.take().is_none());
    let mut held = held;
    let given = held.swap_remove(7);
    let first = given.first();
    drop(given);
    let again = pool.take().unwrap();
    assert_eq!(again.first(), first);
    assert!(pool.take().is_none());
    drop(held);
    drop(again);
    let all: Vec<_> = (0..SLOTS).map(|_| pool.take().unwrap()).collect();
    assert!(pool.take().is_none());
    drop(all);
    // Commits one in flight at a time never run out.
    for _ in 0..3 * SLOTS {
        assert!(pool.take().is_some());
    }
}

/// A device's pools: the first made at the first slot taken, another only when every slot of
/// those made is held, with no most where the device makes every one asked; a freed slot of any
/// pool serves the next commit, with no pool made; the pool that gave the last slot is asked
/// first.
#[test]
fn pools_are_made_as_the_command_buffers_in_flight_need_them() {
    use ash::vk::Handle;
    const MADE: u64 = 5;
    let made = std::cell::Cell::new(0u64);
    // Each pool of three queries a slot.
    let make = |count: u32| {
        assert_eq!(count, PREFIXED * SLOTS);
        made.set(made.get() + 1);
        Some(vk::QueryPool::from_raw(made.get()))
    };
    let mut pools = Pools::new(None, PREFIXED);
    assert_eq!(pools.slots(), 0);
    let mut held = Vec::new();
    for pool in 1..=MADE {
        for _ in 0..SLOTS {
            let slot = pools.take(make).unwrap();
            assert_eq!(slot.queries().as_raw(), pool);
            held.push(slot);
        }
    }
    assert_eq!(pools.slots(), MADE as usize * SLOTS as usize);
    // A slot of the first pool given back serves the next commit; no pool more is made, and
    // past it, with every slot held again, one is.
    let given = held.swap_remove(5);
    let (queries, first) = (given.queries(), given.first());
    drop(given);
    let again = pools.take(make).unwrap();
    assert_eq!((again.queries(), again.first()), (queries, first));
    assert_eq!(made.get(), MADE);
    held.push(again);
    let sixth = pools.take(make).unwrap();
    assert_eq!((made.get(), sixth.queries().as_raw()), (MADE + 1, MADE + 1));
    held.push(sixth);
    // The pool that gave the last slot is asked first: with a slot of the second pool given
    // back, the next commit's slot is still the sixth's, which has slots free.
    drop(held.swap_remove(SLOTS as usize + 3));
    let next = pools.take(make).unwrap();
    assert_eq!(next.queries().as_raw(), MADE + 1);
    assert_eq!(
        pools
            .queries()
            .map(|queries| queries.as_raw())
            .collect::<Vec<_>>(),
        (1..=MADE + 1).collect::<Vec<_>>()
    );
}

/// A pool the device refuses ends the making: the commit gets no slot, and no later commit tries
/// again while every slot of the pools made is held. A device of scarce pools makes one, of two
/// queries a slot: 4096 timestamps, as a Metal counter sample buffer holds.
#[test]
fn a_refused_pool_ends_the_making_and_scarce_pools_are_one() {
    use ash::vk::Handle;
    let mut pools = Pools::new(None, PREFIXED);
    let held: Vec<_> = (0..SLOTS)
        .map(|_| pools.take(|_| Some(vk::QueryPool::from_raw(1))).unwrap())
        .collect();
    assert!(pools.take(|_| None).is_none());
    let mut asked = false;
    assert!(pools
        .take(|_| {
            asked = true;
            Some(vk::QueryPool::from_raw(2))
        })
        .is_none());
    assert!(!asked);
    assert_eq!(pools.slots(), SLOTS as usize);
    drop(held);
    assert!(pools.take(|_| None).is_some());

    let mut scarce = Pools::new(Some(1), PLAIN);
    let held: Vec<_> = (0..SLOTS)
        .map(|_| {
            scarce
                .take(|count| {
                    assert_eq!(count, 4096);
                    Some(vk::QueryPool::from_raw(1))
                })
                .unwrap()
        })
        .collect();
    let mut asked = false;
    assert!(scarce
        .take(|_| {
            asked = true;
            Some(vk::QueryPool::from_raw(2))
        })
        .is_none());
    assert!(!asked);
    assert_eq!(scarce.slots(), SLOTS as usize);
    drop(held);
}
