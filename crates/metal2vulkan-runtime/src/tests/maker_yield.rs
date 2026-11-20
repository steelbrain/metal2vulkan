//! The block-maker thread's wait between its memory calls (`crate::submit_calls`), with a submit
//! call counted in standing in for one inside the driver. GPU-free: a submit call inside the
//! driver as the maker's call returned has returned before the maker's next call starts, the
//! maker goes on with what it made at once, and a submit call that entered after its call
//! returned does not hold its next one; on a quiet queue the maker goes straight on; the wait
//! before a free is not part of the free; and a thread that makes no memory ahead of need waits
//! for nothing. Under a device cfg, on every device of the cfg's kind: the block-maker thread's
//! unit made while a submit call is inside the driver joins the pool at once, its next unit's
//! allocation waits that submit call out, a take's own allocation on the consumer's thread waits
//! for nothing, and an idle allocation's free on a quiet queue goes straight on, every gauge at
//! zero after.

use crate::counters::Counters;
use crate::device::lock;
use crate::memory::{make_ahead_here, maker_call};
use crate::submit_calls::SubmitCalls;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// How long a stand-in submit call stays inside the driver once the maker waits for it.
const HOLD: Duration = Duration::from_millis(50);

/// Waits until `done`, for at most 10 seconds.
fn until(what: &str, done: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(1));
    }
}

/// Runs `body` on a thread that makes memory ahead of need for `calls`, as the block-maker
/// thread does, and waits for it to end, for at most 10 seconds.
fn on_maker(calls: &Arc<SubmitCalls>, body: impl FnOnce() + Send + 'static) {
    let calls = Arc::clone(calls);
    let (ended, end) = mpsc::channel();
    thread::spawn(move || {
        make_ahead_here(Some(calls));
        body();
        let _ = ended.send(());
    });
    end.recv_timeout(Duration::from_secs(10))
        .expect("the maker's calls end");
}

/// The block-maker thread's counts in `counters`: its memory calls, the waits before them and
/// whether those waits took any time.
fn counts(counters: &Counters) -> (u64, u64, bool) {
    let metrics = counters.metrics();
    let waited = metrics.maker_yield_nanos > 0;
    (metrics.maker_memory_calls, metrics.maker_yields, waited)
}

/// A submit call inside the driver as the maker's call returned has returned before the maker's
/// next call starts, while the maker goes on with what the first call made at once; a submit call
/// that entered after the first call returned does not hold the next one, and the wait lasts
/// until the call it waits for returns.
#[test]
fn the_makers_next_call_starts_once_each_submit_inside_the_driver_as_its_call_returned_returned() {
    let counters = Counters::new();
    let calls = Arc::new(SubmitCalls::new(counters.clone()));
    let log: Arc<Mutex<Vec<&str>>> = Arc::default();
    let first = calls.enter();
    let maker = {
        let (calls, log) = (Arc::clone(&calls), Arc::clone(&log));
        thread::spawn(move || {
            make_ahead_here(Some(calls));
            maker_call(|| lock(&log).push("the maker's call"));
            lock(&log).push("the maker hands on what it made");
            maker_call(|| lock(&log).push("the maker's next call"));
        })
    };
    until("the maker to wait or go on", || {
        calls.waiting() == 1 || lock(&log).len() == 3
    });
    let second = calls.enter();
    thread::sleep(HOLD);
    lock(&log).push("the first submit call returned");
    drop(first);
    // The second is still inside the driver.
    until("the maker's next call", || lock(&log).len() == 4);
    maker.join().expect("the maker");
    assert_eq!(
        *lock(&log),
        [
            "the maker's call",
            "the maker hands on what it made",
            "the first submit call returned",
            "the maker's next call",
        ]
    );
    drop(second);
    let metrics = counters.metrics();
    assert_eq!((metrics.maker_memory_calls, metrics.maker_yields), (2, 1));
    assert!(
        metrics.maker_yield_nanos >= HOLD.as_nanos() as u64,
        "the wait lasted until the call returned: {} ns",
        metrics.maker_yield_nanos
    );
}

/// With no submit call left inside the driver that its previous call met, each of the maker's
/// calls goes straight on: a submit call that entered and returned before holds nothing, and no
/// wait is counted.
#[test]
fn on_a_quiet_queue_the_maker_goes_straight_on() {
    let counters = Counters::new();
    let calls = Arc::new(SubmitCalls::new(counters.clone()));
    drop(calls.enter());
    on_maker(&calls, || {
        maker_call(|| ());
        maker_call(|| ());
    });
    assert_eq!(counts(&counters), (2, 0, false));
}

/// A free is marked as one for its own length: the maker's wait before it, for a submit call its
/// previous free met, is not part of it.
#[test]
fn the_wait_before_a_free_is_not_part_of_the_free() {
    let counters = Counters::new();
    let calls = Arc::new(SubmitCalls::new(counters.clone()));
    let inside = calls.enter();
    let marked: Arc<Mutex<Vec<bool>>> = Arc::default();
    let maker = {
        let (calls, marked) = (Arc::clone(&calls), Arc::clone(&marked));
        thread::spawn(move || {
            make_ahead_here(Some(Arc::clone(&calls)));
            calls.free(|| lock(&marked).push(calls.freeing()));
            calls.free(|| lock(&marked).push(calls.freeing()));
        })
    };
    until("the maker to wait before its next free", || {
        calls.waiting() == 1
    });
    assert!(
        !calls.freeing(),
        "the wait before the free is not marked as one"
    );
    drop(inside);
    maker.join().expect("the maker");
    assert_eq!(*lock(&marked), [true, true], "each free is marked as one");
    assert!(!calls.freeing());
    assert_eq!(counts(&counters), (2, 1, true));
}

/// A thread that makes no memory ahead of need, a consumer's take among them, waits for nothing
/// around its calls, a submit call inside the driver or not, and counts none of them.
#[test]
fn a_thread_that_makes_nothing_ahead_waits_for_nothing() {
    let counters = Counters::new();
    let calls = Arc::new(SubmitCalls::new(counters.clone()));
    let inside = calls.enter();
    let (ended, end) = mpsc::channel();
    thread::spawn(move || {
        maker_call(|| ());
        maker_call(|| ());
        let _ = ended.send(());
    });
    end.recv_timeout(Duration::from_secs(10))
        .expect("the calls go on with a submit call inside the driver");
    drop(inside);
    assert_eq!(counts(&counters), (0, 0, false));
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use super::HOLD;
    use crate::tests::command::on_device::{each_device, within};
    use crate::MemoryClass;
    use std::thread;

    /// With a submit call counted in as inside the driver, the block-maker thread's next Shared
    /// unit, made as a take uses up the class's last, joins the pool at once, and the unit after
    /// it waits that submit call out before its allocation starts; a take that allocates on the
    /// consumer's thread goes straight on; with none inside, the free of the idle allocation a
    /// take gives back waits for nothing.
    #[test]
    fn every_devices_block_maker_waits_out_a_submit_call_inside_the_driver() {
        each_device("maker_yield", |name, _, device| {
            let core = device.core();
            let counters = device.counters();
            within("both classes stocked", || {
                core.pool.stocked(MemoryClass::Shared) && core.pool.stocked(MemoryClass::Private)
            });
            let made = counters.metrics();
            // Each class's first unit in the device's making, and the zero source, the ring's
            // first block and the staging region on the block-maker thread.
            assert_eq!(
                (made.maker_memory_calls, made.maker_yields),
                (5, 0),
                "{name}: the memory calls made ahead of need"
            );
            let len = MemoryClass::Shared.block_len();
            let inside = core.submit_calls.enter();
            let long = (device.take_block(MemoryClass::Private, 2 * len))
                .unwrap_or_else(|error| panic!("{name}: a long take: {error}"));
            let taken = counters.metrics();
            assert_eq!(
                (taken.consumer_allocations, taken.maker_memory_calls),
                (made.consumer_allocations + 1, made.maker_memory_calls),
                "{name}: the long take allocated on the consumer's thread, waiting for nothing"
            );
            // The unit made while the submit call is inside the driver joins the pool at once.
            let first = (device.take_block(MemoryClass::Shared, len))
                .unwrap_or_else(|error| panic!("{name}: the Shared unit: {error}"));
            within("the next Shared unit", || {
                core.pool.stocked(MemoryClass::Shared)
            });
            let next = counters.metrics();
            assert_eq!(
                (next.maker_memory_calls, next.maker_yields),
                (made.maker_memory_calls + 1, 0),
                "{name}: the next unit's allocation started at once"
            );
            // The unit after it waits for the submit call its allocation met.
            let second = (device.take_block(MemoryClass::Shared, len))
                .unwrap_or_else(|error| panic!("{name}: the next Shared unit: {error}"));
            within("the block-maker thread to wait", || {
                core.submit_calls.waiting() == 1
            });
            thread::sleep(HOLD);
            assert!(
                core.submit_calls.waiting() == 1 && !core.pool.stocked(MemoryClass::Shared),
                "{name}: the third unit's allocation waits for the submit call"
            );
            drop(inside);
            within("the third Shared unit", || {
                core.pool.stocked(MemoryClass::Shared)
            });
            let yielded = counters.metrics();
            assert_eq!(
                (yielded.maker_memory_calls, yielded.maker_yields),
                (next.maker_memory_calls + 1, 1),
                "{name}: the third unit's allocation waited"
            );
            assert!(
                yielded.maker_yield_nanos >= HOLD.as_nanos() as u64,
                "{name}: the wait lasted until the submit call returned: {} ns",
                yielded.maker_yield_nanos
            );
            device.release_block(first).expect("a release");
            within("the idle unit's free", || core.pool.frees().len() == 1);
            let freed = counters.metrics();
            assert_eq!(
                (freed.maker_memory_calls, freed.maker_yields),
                (yielded.maker_memory_calls + 1, 1),
                "{name}: the free on a quiet queue went straight on"
            );
            println!(
                "maker_yield: {name}: calls {}, waits {} of {} ns",
                freed.maker_memory_calls, freed.maker_yields, freed.maker_yield_nanos
            );
            device.release_block(second).expect("a release");
            device.release_block(long).expect("a release");
        });
    }
}
