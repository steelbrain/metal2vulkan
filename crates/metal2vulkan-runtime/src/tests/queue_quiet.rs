//! The wait for a burst of a device's queue uses to end (`device/queue_quiet.rs`), which a block
//! made ahead runs before its `vkAllocateMemory` (`memory/ready.rs`): none on a device idle for
//! the wait's bound, the end of a burst that begins after the wait does, its bound where a use
//! never ends, and none once asked.
//!
//! Times here are bounds a loaded host still meets: each burst lies well inside its wait's bound,
//! and what each row holds is how the wait ended.

use crate::device::queue_quiet::{QueueQuiet, Waited, SETTLE};
use crate::gpu_time;
use std::time::{Duration, Instant};

/// A bound longer than any pause a loaded host puts between two steps here.
const LONG: Duration = Duration::from_secs(5);

#[test]
fn a_device_with_no_use_waits_for_none() {
    let quiet = QueueQuiet::default();
    let at = Instant::now();
    assert_eq!(quiet.wait_after_burst(LONG, || false), Waited::Idle);
    assert!(at.elapsed() < LONG / 2);
}

#[test]
fn a_wait_ends_once_a_burst_that_began_after_it_ended() {
    let quiet = QueueQuiet::default();
    // A use just ended: the device is not idle, and that burst is not the one waited for.
    quiet.began();
    quiet.ended();
    std::thread::scope(|scope| {
        // The host's clock as the burst's use ends, read before it ends on the clock the wait
        // reads, so that no pause between the two can put it past the end the wait saw.
        let ended = scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(20));
            quiet.began();
            std::thread::sleep(Duration::from_millis(5));
            let ending = gpu_time::host_nanos();
            quiet.ended();
            ending
        });
        let waited = quiet.wait_after_burst(LONG, || false);
        let returned = gpu_time::host_nanos();
        let ended = ended.join().unwrap();
        assert_eq!(waited, Waited::AfterBurst);
        assert!(
            returned >= ended + SETTLE.as_nanos() as u64,
            "the wait ended before the burst settled"
        );
    });
}

#[test]
fn a_use_that_never_ends_runs_the_wait_to_its_bound() {
    let quiet = QueueQuiet::default();
    quiet.began();
    let most = Duration::from_millis(30);
    let at = Instant::now();
    assert_eq!(quiet.wait_after_burst(most, || false), Waited::Bound);
    assert!(at.elapsed() >= most);
}

#[test]
fn a_wait_asked_to_stop_stops() {
    let quiet = QueueQuiet::default();
    quiet.began();
    let at = Instant::now();
    assert_eq!(quiet.wait_after_burst(LONG, || true), Waited::Asked);
    assert!(at.elapsed() < LONG / 2);
}
