//! The plan that takes a driver's stall in one signalling submission of every 1024 to a queue
//! while the scheduler is idle (`device/submit_stalls.rs`), against a model of the driver.

use crate::device::submit_stalls::{
    Caller, Period, BEHIND_NANOS, IDLE_NANOS, MARGIN, MOST_EMPTY, PERIOD, STALL_NANOS,
};

/// The stall the RTX 5080's driver took, and its other calls' time, in nanoseconds.
const STALL: u64 = 2_300_000;
const CALL: u64 = 2_000;

/// A driver whose every [`PERIOD`]th signalling call to the queue stalls, its first at call
/// `first`, and every `early`th period `by` calls early; or none, where `stalls` is false. Its
/// stalls take `stall`, and the calls of `costs` (each a call's number and its time) take long
/// elsewhere.
struct Driver {
    calls: u64,
    first: u64,
    stalls: bool,
    stall: u64,
    costs: Vec<(u64, u64)>,
    /// The call that last stalled, and the periods so far.
    last: Option<u64>,
    periods: u64,
    early: Option<(u64, u64)>,
}

impl Driver {
    fn new(first: u64, stalls: bool) -> Driver {
        Driver {
            calls: 0,
            first,
            stalls,
            stall: STALL,
            costs: Vec::new(),
            last: None,
            periods: 0,
            early: None,
        }
    }

    /// One signalling call: its time, and whether it stalled.
    fn call(&mut self) -> (u64, bool) {
        self.calls += 1;
        let period = match self.early {
            Some((every, by)) if (self.periods + 1).is_multiple_of(every) => u64::from(PERIOD) - by,
            _ => u64::from(PERIOD),
        };
        let stalls = self.stalls
            && match self.last {
                None => self.calls == self.first,
                Some(last) => self.calls - last >= period,
            };
        if !stalls {
            let cost = self.costs.iter().find(|(call, _)| *call == self.calls);
            return (cost.map_or(CALL, |(_, took)| *took), false);
        }
        if self.last.is_some() {
            self.periods += 1;
        }
        self.last = Some(self.calls);
        (self.stall, true)
    }
}

/// What a run of frames did: the stalls on work, those taken idle, and the empty calls made, as
/// the plan counts them; and the driver's stalls that fell on work.
#[derive(Debug, Default, PartialEq, Eq)]
struct Run {
    on_work: u32,
    idle: u32,
    empty: u32,
    driver_on_work: u32,
}

/// `frames` frames of `per_frame` submissions 15 us apart, each frame followed by an idle of 8 ms
/// in which the scheduler takes the stall due where `take`, as `IdleStalls::take` does.
fn frames(driver: &mut Driver, frames: u32, per_frame: u32, take: bool) -> Run {
    let (mut period, mut run, mut now) = (Period::default(), Run::default(), 0);
    for _ in 0..frames {
        for _ in 0..per_frame {
            let (took, stalled) = driver.call();
            now += took + 15_000;
            if period.called(Caller::Work, took, now) {
                run.on_work += 1;
            }
            run.driver_on_work += u32::from(stalled);
        }
        now += 8_000_000;
        if take && period.may_be_due() {
            for _ in 0..period.due_idle() {
                let (took, _) = driver.call();
                now += took;
                run.empty += 1;
                if period.called(Caller::Empty, took, now) {
                    run.idle += 1;
                    break;
                }
            }
        }
    }
    run
}

/// 200 frames of 25 command buffers (5000 submissions): without the plan, one in 1024 falls on
/// work; with it, the first alone, which teaches the plan the queue's place, and every later one
/// is taken idle, by no more empty calls a stall than a burst and the margin.
#[test]
fn every_stall_after_the_first_is_taken_idle() {
    let without = frames(&mut Driver::new(28, true), 200, 25, false);
    assert_eq!(without.on_work, 5, "{without:?}");
    let with = frames(&mut Driver::new(28, true), 200, 25, true);
    assert_eq!(with.on_work, 1, "{with:?}");
    assert!(with.idle >= 4, "{with:?}");
    assert!(with.empty <= with.idle * (25 + MARGIN), "{with:?}");
    // Whatever the driver's place at the first stall.
    for first in [1, 25, 26, 500, 1023, 1024] {
        let with = frames(&mut Driver::new(first, true), 200, 25, true);
        assert_eq!(with.on_work, 1, "first {first}: {with:?}");
        assert!(with.idle >= 4, "first {first}: {with:?}");
    }
}

/// A stall that comes 24 calls before the 1024th, as 5 of 58 did on the RTX 5080, is taken
/// idle too: the margin covers it.
#[test]
fn a_stall_that_comes_early_is_taken_idle() {
    let mut driver = Driver::new(28, true);
    driver.early = Some((3, 24));
    let with = frames(&mut driver, 400, 25, true);
    assert!(driver.periods >= 9, "{}", driver.periods);
    assert_eq!(with.on_work, 1, "{with:?}");
    assert_eq!(u64::from(with.idle), driver.periods, "{with:?}");
}

/// As on the RTX 5080 (615.71.09) in the device test: stalls of 1.97 ms, the device's 62nd
/// submission 10 ms and its 162nd 1.3 ms before the first stall, and a call of 1.26 ms well
/// before a later one. The long calls before the first stall give way to it; the one after it is
/// no stall; every stall after the first is taken idle, each by no more empty calls than a burst
/// and the margin.
#[test]
fn long_calls_elsewhere_and_short_stalls_spend_no_empty_call() {
    let mut driver = Driver::new(997, true);
    driver.stall = 1_970_000;
    driver.costs = vec![(62, 10_000_000), (162, 1_300_000), (5013, 1_260_000)];
    let with = frames(&mut driver, 200, 25, true);
    assert_eq!(driver.periods, 4, "{with:?}");
    assert_eq!(with.driver_on_work, 1, "{with:?}");
    assert_eq!(with.idle, 4, "{with:?}");
    assert!(with.empty <= with.idle * (25 + MARGIN), "{with:?}");
}

/// Once a stall came where the count put it, a long call elsewhere is no stall and leaves the
/// place: the next stall is still due where the count puts it.
#[test]
fn a_long_call_away_from_a_confirmed_place_leaves_it() {
    let mut period = Period::default();
    let mut now = 0;
    let mut call = |period: &mut Period, took: u64| {
        now += took + 15_000;
        period.called(Caller::Work, took, now)
    };
    assert!(call(&mut period, STALL));
    for _ in 0..PERIOD - 1 {
        assert!(!call(&mut period, CALL));
    }
    // Where the count put it: the place is confirmed.
    assert!(call(&mut period, STALL));
    for _ in 0..500 {
        assert!(!call(&mut period, CALL));
    }
    assert!(
        !call(&mut period, 10_000_000),
        "a long call away from the place"
    );
    for _ in 0..PERIOD - 502 {
        assert!(!call(&mut period, CALL));
    }
    assert!(period.may_be_due());
    assert!(
        call(&mut period, STALL_NANOS),
        "the stall where the count put it"
    );
}

/// A driver that never stalls gets no empty call: no place is learnt.
#[test]
fn no_stall_seen_takes_none() {
    let run = frames(&mut Driver::new(28, false), 200, 25, true);
    assert_eq!(run, Run::default());
}

/// A stall due that did not come forgets the place: no empty call until the next stall.
#[test]
fn a_stall_due_that_does_not_come_forgets_the_place() {
    let mut period = Period::default();
    assert!(period.called(Caller::Work, STALL, 10 * STALL));
    for at in 0..PERIOD - 1 {
        assert!(!period.called(Caller::Work, CALL, 20 * STALL + u64::from(at)));
    }
    assert!(period.may_be_due());
    assert!(!period.called(Caller::Work, CALL, 30 * STALL));
    assert!(!period.may_be_due());
    assert_eq!(period.due_idle(), 0);
}

/// A use of the queue the runtime cannot count forgets the place.
#[test]
fn an_uncounted_use_forgets_the_place() {
    let mut period = Period::default();
    period.called(Caller::Work, STALL, STALL);
    for at in 0..PERIOD - 10 {
        period.called(Caller::Work, CALL, STALL + 15_000 * u64::from(at));
    }
    assert!(period.may_be_due());
    period.forget();
    assert!(!period.may_be_due());
    assert_eq!(period.due_idle(), 0);
}

/// A stall further off than the next burst (the largest of the last, plus the margin) waits for
/// a later idle; one within it is due, the calls left up to and including its own; never more
/// than `MOST_EMPTY`.
#[test]
fn a_stall_is_taken_within_the_next_burst() {
    let burst = |period: &mut Period, calls: u32, now: &mut u64| {
        *now += 2 * IDLE_NANOS;
        for _ in 0..calls {
            *now += 15_000;
            period.called(Caller::Work, CALL, *now);
        }
    };
    let (mut period, mut now) = (Period::default(), 0);
    period.called(Caller::Work, STALL, STALL);
    // 38 bursts of 25 after the stall (the first with the stall's call): 950 calls since, 74
    // left.
    for _ in 0..38 {
        burst(&mut period, 25, &mut now);
    }
    assert_eq!(period.due_idle(), 0, "74 left, past 26 + {MARGIN}");
    burst(&mut period, 25, &mut now);
    assert_eq!(
        period.due_idle(),
        49,
        "49 left, within the next burst and the margin"
    );
    // Bursts larger than the most: due within `MOST_EMPTY` calls alone.
    let (mut period, mut now) = (Period::default(), 0);
    period.called(Caller::Work, STALL, STALL);
    burst(&mut period, 700, &mut now);
    assert_eq!(period.due_idle(), 0, "324 left");
    burst(&mut period, 200, &mut now);
    assert_eq!(period.due_idle(), 124, "124 left, within {MOST_EMPTY}");
}

/// Calls 15 us apart are one burst, and one an idle apart begins the next.
#[test]
fn an_idle_splits_bursts() {
    let (mut period, mut now) = (Period::default(), 0);
    period.called(Caller::Work, STALL, STALL);
    for _ in 0..1000 {
        now += 15_000;
        period.called(Caller::Work, CALL, now);
    }
    // One burst of 1001 calls: 24 left, within the most.
    assert_eq!(period.due_idle(), 24);
    let (mut period, mut now) = (Period::default(), 0);
    period.called(Caller::Work, STALL, STALL);
    for at in 0..924 {
        now += match at % 10 {
            0 => 2 * IDLE_NANOS,
            _ => 15_000,
        };
        period.called(Caller::Work, CALL, now);
    }
    // Bursts of 10 (the first with the stall's, 11): 100 left, past 11 + the margin.
    assert_eq!(period.due_idle(), 0);
}

/// A call under `STALL_NANOS` is no stall, whatever its length.
#[test]
fn a_long_call_short_of_a_stall_is_none() {
    let mut period = Period::default();
    assert!(!period.called(Caller::Work, STALL_NANOS - 1, STALL_NANOS));
    assert!(!period.may_be_due());
    assert!(period.called(Caller::Work, STALL_NANOS, 3 * STALL_NANOS));
}

/// The call that takes the queue's lock next waited behind a stall taken idle where an empty
/// call stalled before it and it waited `BEHIND_NANOS` or more; the one after it did not, nor one
/// after a stall that fell on work.
#[test]
fn a_wait_behind_a_stall_taken_idle_is_counted_once() {
    let mut period = Period::default();
    period.called(Caller::Empty, STALL, STALL);
    assert_eq!(period.entered(BEHIND_NANOS - 1), None);
    period.called(Caller::Empty, STALL, 2 * STALL);
    assert_eq!(period.entered(STALL / 2), Some(STALL / 2));
    assert_eq!(period.entered(STALL / 2), None);
    period.called(Caller::Work, STALL, 3 * STALL);
    assert_eq!(period.entered(STALL / 2), None);
}
