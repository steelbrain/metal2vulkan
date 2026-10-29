//! `exit_guard.rs`: sections end before the hook returns, none starts after it closed, the hook
//! never waits for the exiting thread's own sections, its wait is bounded, only its first run
//! waits for a CPU device's work, and it acts only in the process it was registered in. A child's
//! exit while the driver runs the runtime's work ends at once with its status where the hook learns
//! it (Linux). Each test uses a guard of its own; only the fork tests use the process's, which
//! nothing here closes.

use crate::exit_guard::{self, hook, Busy, Call, Guard, InProcess, InProcessDevices};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Barrier, Mutex, Weak};
use std::time::{Duration, Instant};

/// A guard of the test's own, for the life of the test process.
fn guard() -> &'static Guard {
    Box::leak(Box::new(Guard::new()))
}

/// No device whose queues run in this process.
fn none_in_process() -> InProcessDevices {
    InProcessDevices::new()
}

#[test]
fn sections_start_until_the_guard_closes_and_never_after() {
    let guard = guard();
    let before = exit_guard::own();
    {
        let first = guard.enter();
        assert!(first.is_some());
        let nested = guard.enter();
        assert!(nested.is_some());
        assert_eq!(exit_guard::own(), before + 2);
    }
    assert_eq!(exit_guard::own(), before);
    assert!(!guard.is_closed());
    assert!(guard.close(0, Duration::ZERO, Duration::ZERO));
    assert!(guard.is_closed());
    assert!(guard.enter().is_none());
    assert_eq!(exit_guard::own(), before);
    // On every thread.
    assert!(std::thread::spawn(move || guard.enter().is_none())
        .join()
        .unwrap());
}

/// The hook waits for a section in progress on another thread (a submission, a poll, a device's
/// teardown) and returns once it ended. That section runs to its end, the sections inside it
/// included (a drop's releases), while no section starts afresh after the close.
#[test]
fn the_hook_waits_for_the_sections_in_progress_on_other_threads() {
    let guard = guard();
    let (entered, in_section) = (Arc::new(Barrier::new(2)), Arc::new(Barrier::new(2)));
    let worker = {
        let (entered, in_section) = (Arc::clone(&entered), Arc::clone(&in_section));
        std::thread::spawn(move || {
            let busy = guard.enter().expect("open");
            entered.wait();
            // The guard closes meanwhile.
            in_section.wait();
            std::thread::sleep(Duration::from_millis(200));
            let nested_after_close = guard.enter().is_some();
            drop(busy);
            let fresh_after_close = guard.enter().is_some();
            (nested_after_close, fresh_after_close)
        })
    };
    entered.wait();
    let start = Instant::now();
    let closer = std::thread::spawn(move || {
        guard.close(0, Duration::from_secs(10), Duration::from_secs(10))
    });
    while !guard.is_closed() {
        std::thread::yield_now();
    }
    in_section.wait();
    assert!(
        closer.join().unwrap(),
        "the hook gave up on a section that ended"
    );
    let waited = start.elapsed();
    assert!(
        waited >= Duration::from_millis(200),
        "the hook returned after {waited:?}, before the section ended"
    );
    assert!(waited < Duration::from_secs(5), "{waited:?}");
    let (nested, fresh) = worker.join().unwrap();
    assert!(nested, "a section inside one in progress did not start");
    assert!(!fresh, "a section started afresh after the close");
}

/// A section that never ends (a driver or a message sink that blocks) holds the hook back at most
/// its bound, and the hook says so.
#[test]
fn the_hook_s_wait_is_bounded() {
    let guard = guard();
    let (held, release) = (mpsc::channel(), mpsc::channel::<()>());
    let (entered, stop) = (held.0, release.1);
    let worker = std::thread::spawn(move || {
        let _busy: Busy<'static> = guard.enter().expect("open");
        entered.send(()).unwrap();
        let _ = stop.recv();
    });
    held.1.recv().unwrap();
    let bound = Duration::from_millis(300);
    let start = Instant::now();
    assert!(!guard.close(0, bound, bound), "a stuck section ended");
    let waited = start.elapsed();
    assert!(
        waited >= bound && waited < bound + Duration::from_secs(2),
        "the hook waited {waited:?} for a bound of {bound:?}"
    );
    release.0.send(()).unwrap();
    worker.join().unwrap();
}

/// A section around a driver's making of a `VkPipeline` holds the hook past its bound until it
/// ends, within the making's own bound (`MAKING_BOUND` where the hook waits, on macOS): a making
/// outlasts the bound, and the program need not wait for it. Where the making's bound is not
/// longer (the hook that ends the process at once, on Linux), it holds the hook no longer than the
/// bound. One run only on a thread in a section takes such a section (`driver_making`).
#[test]
fn the_hook_waits_past_its_bound_for_a_making_in_progress() {
    let making = Duration::from_secs(10);
    let guard = guard();
    let (held, release) = (mpsc::channel(), mpsc::channel::<()>());
    let (entered, stop) = (held.0, release.1);
    let worker = std::thread::spawn(move || {
        let _busy: Busy<'static> = guard.enter_making().expect("open");
        entered.send(()).unwrap();
        let _ = stop.recv();
        std::thread::sleep(Duration::from_millis(300));
    });
    held.1.recv().unwrap();
    release.0.send(()).unwrap();
    let bound = Duration::from_millis(50);
    let start = Instant::now();
    assert!(
        guard.close(0, bound, making),
        "the making's section did not end"
    );
    let waited = start.elapsed();
    assert!(
        waited >= Duration::from_millis(250) && waited < making,
        "the hook waited {waited:?}"
    );
    assert!(guard.enter_making().is_none());
    worker.join().unwrap();

    // With the making's bound no longer than the bound.
    let guard = self::guard();
    let (held, release) = (mpsc::channel(), mpsc::channel::<()>());
    let (entered, stop) = (held.0, release.1);
    let worker = std::thread::spawn(move || {
        let _busy: Busy<'static> = guard.enter_making().expect("open");
        entered.send(()).unwrap();
        let _ = stop.recv();
    });
    held.1.recv().unwrap();
    let start = Instant::now();
    assert!(!guard.close(0, bound, bound), "a stuck making ended");
    let waited = start.elapsed();
    assert!(
        waited >= bound && waited < Duration::from_millis(250),
        "the hook waited {waited:?} for a bound of {bound:?}"
    );
    release.0.send(()).unwrap();
    worker.join().unwrap();

    // `driver_making` takes such a section of the process's guard only on a thread in a section.
    let before = exit_guard::own();
    assert_eq!(exit_guard::driver_making(exit_guard::own), before);
    let busy = exit_guard::enter().expect("open");
    assert_eq!(exit_guard::driver_making(exit_guard::own), before + 2);
    drop(busy);
}

/// A making that ends as the hook looks, past its bound, is seen ended: whether it ends between
/// the hook's read of the sections and its read of the makings, or the hook reads both between
/// the section's end and its making's. The hook gave up on such a making once (`close` returned
/// false as the section ended, on an RX 7900 XTX under the runtime lane's stress), and then left
/// the work of the process's own devices undrained. The steps run between the two counts
/// (`exit_guard::BETWEEN_COUNTS`), on this thread: no timing.
#[test]
fn the_hook_sees_a_making_that_ends_as_it_looks_ended() {
    // Between the hook's two reads: one section in progress, then no making.
    let guard = guard();
    let making = guard.enter_making().expect("open");
    exit_guard::BETWEEN_COUNTS
        .with(|step| *step.borrow_mut() = Some(Box::new(move || drop(making))));
    assert!(
        guard.close(0, Duration::ZERO, Duration::ZERO),
        "the hook gave up on a making that ended between its reads"
    );

    // Both reads between the section's end and its making's.
    let guard = self::guard();
    let making = guard.enter_making().expect("open");
    let seen = std::rc::Rc::new(std::cell::Cell::new(None));
    let looked = std::rc::Rc::clone(&seen);
    exit_guard::BETWEEN_COUNTS.with(|step| {
        *step.borrow_mut() = Some(Box::new(move || {
            looked.set(Some(guard.close(0, Duration::ZERO, Duration::ZERO)));
        }))
    });
    drop(making);
    assert_eq!(
        seen.get(),
        Some(true),
        "the hook gave up on a making whose section had ended"
    );
}

/// An exit from inside a section (a message sink that calls `exit`) does not wait for the exiting
/// thread's own section, which can never end first; it still waits for the others.
#[test]
fn the_hook_does_not_wait_for_the_exiting_thread_s_own_sections() {
    let guard = guard();
    let own = guard.enter().expect("open");
    let start = Instant::now();
    assert!(guard.close(
        exit_guard::own(),
        Duration::from_secs(10),
        Duration::from_secs(10)
    ));
    assert!(start.elapsed() < Duration::from_secs(1));
    drop(own);

    let guard = self::guard();
    let own = guard.enter().expect("open");
    let (other, done) = (mpsc::channel(), mpsc::channel::<()>());
    let worker = std::thread::spawn(move || {
        let _busy = guard.enter().expect("open");
        other.0.send(()).unwrap();
        let _ = done.1.recv();
    });
    other.1.recv().unwrap();
    // One section is this thread's, the other a stuck one's: the hook waits for that one only.
    assert!(!guard.close(
        exit_guard::own(),
        Duration::from_millis(100),
        Duration::from_millis(100)
    ));
    done.0.send(()).unwrap();
    worker.join().unwrap();
    drop(own);
}

/// Many threads entering and leaving sections while the guard closes: once the hook returned, no
/// thread is in a section, and none enters one again.
#[test]
fn no_section_runs_once_the_hook_returned() {
    let guard = guard();
    let inside = Arc::new(AtomicUsize::new(0));
    let after = Arc::new(AtomicBool::new(false));
    let late = Arc::new(AtomicUsize::new(0));
    let workers: Vec<_> = (0..8)
        .map(|_| {
            let (inside, after, late) =
                (Arc::clone(&inside), Arc::clone(&after), Arc::clone(&late));
            std::thread::spawn(move || {
                for _ in 0..200_000 {
                    let Some(busy) = guard.enter() else {
                        return;
                    };
                    inside.fetch_add(1, Ordering::SeqCst);
                    if after.load(Ordering::SeqCst) {
                        late.fetch_add(1, Ordering::SeqCst);
                    }
                    inside.fetch_sub(1, Ordering::SeqCst);
                    drop(busy);
                }
            })
        })
        .collect();
    std::thread::sleep(Duration::from_millis(5));
    assert!(guard.close(0, Duration::from_secs(10), Duration::from_secs(10)));
    after.store(true, Ordering::SeqCst);
    assert_eq!(
        inside.load(Ordering::SeqCst),
        0,
        "a section ran past the hook"
    );
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(
        late.load(Ordering::SeqCst),
        0,
        "a section started after the hook"
    );
}

/// The hook acts only in the process it was registered in.
#[test]
fn the_hook_does_nothing_in_another_process() {
    let guard = guard();
    let busy = guard.enter().expect("open");
    let start = Instant::now();
    let devices = none_in_process();
    assert_eq!(
        hook(
            guard,
            &devices,
            41,
            42,
            0,
            Duration::from_secs(10),
            Duration::from_secs(10)
        ),
        None
    );
    assert!(start.elapsed() < Duration::from_secs(1));
    assert!(!guard.is_closed());
    drop(busy);
    assert_eq!(
        hook(
            guard,
            &devices,
            42,
            42,
            0,
            Duration::from_secs(10),
            Duration::from_secs(10)
        ),
        Some(true)
    );
    assert!(guard.is_closed());
}

/// A device whose queues run in this process, as the hook sees it: its work completes when the
/// test says so, and it records when the hook began to wait for it.
struct Running {
    done: Mutex<Option<mpsc::Receiver<()>>>,
    asked: Mutex<Option<Instant>>,
}

impl Running {
    fn new(done: mpsc::Receiver<()>) -> Arc<Running> {
        Arc::new(Running {
            done: Mutex::new(Some(done)),
            asked: Mutex::new(None),
        })
    }

    fn asked(&self) -> Option<Instant> {
        *self.asked.lock().unwrap()
    }
}

impl InProcess for Running {
    fn drain(&self, deadline: Option<Instant>) -> bool {
        self.asked.lock().unwrap().get_or_insert_with(Instant::now);
        let done = self.done.lock().unwrap().take();
        let left = deadline.map_or(Duration::MAX, |deadline| {
            deadline.saturating_duration_since(Instant::now())
        });
        done.is_some_and(|done| done.recv_timeout(left).is_ok())
    }
}

/// Add `device` to `devices`, as a device whose queues run in this process.
fn add(devices: &InProcessDevices, device: &Arc<Running>) {
    let weak: Weak<dyn InProcess> = Arc::<Running>::downgrade(device);
    devices.add(weak);
}

/// A bound past what the host's clock holds is no deadline: the hook waits until the work
/// completes.
#[test]
fn a_bound_past_the_clock_waits_until_the_work_completes() {
    let guard = guard();
    let devices = none_in_process();
    let (finish, done) = mpsc::channel();
    let running = Running::new(done);
    add(&devices, &running);
    let finishing = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        finish.send(()).unwrap();
    });
    let pid = std::process::id();
    let hooked = hook(guard, &devices, pid, pid, 0, Duration::MAX, Duration::MAX);
    assert_eq!(hooked, Some(true));
    finishing.join().unwrap();
}

/// The interleaving of an exit while a CPU device runs submitted work (the in-flight dispatch
/// a host step's exit leaves on lavapipe), made deterministic: a section is in progress as the
/// hook starts, and the device's work completes only after it ended. The hook waits for the
/// section first, then for the work, and returns once the work completed, in the bound.
#[test]
fn the_hook_waits_for_a_cpu_device_s_work_once_the_sections_ended() {
    let guard = guard();
    let devices = none_in_process();
    let (finish, done) = mpsc::channel();
    let running = Running::new(done);
    add(&devices, &running);
    let (entered, end_section) = (mpsc::channel(), mpsc::channel::<()>());
    let section = std::thread::spawn(move || {
        let busy = guard.enter().expect("open");
        entered.0.send(()).unwrap();
        let _ = end_section.1.recv();
        // Before the count falls: the hook cannot see the section end earlier.
        let ending = Instant::now();
        drop(busy);
        ending
    });
    entered.1.recv().unwrap();
    let pid = std::process::id();
    let start = Instant::now();
    let hooked = std::thread::scope(|scope| {
        let hooking = scope.spawn(|| {
            hook(
                guard,
                &devices,
                pid,
                pid,
                0,
                Duration::from_secs(10),
                Duration::from_secs(10),
            )
        });
        std::thread::sleep(Duration::from_millis(100));
        assert!(
            running.asked().is_none(),
            "the hook drained before the section ended"
        );
        end_section.0.send(()).unwrap();
        let ended = section.join().unwrap();
        while running.asked().is_none() {
            std::thread::yield_now();
        }
        assert!(
            running.asked().unwrap() >= ended,
            "drained before the section ended"
        );
        std::thread::sleep(Duration::from_millis(100));
        assert!(
            !hooking.is_finished(),
            "the hook returned before the work completed"
        );
        finish.send(()).unwrap();
        hooking.join().unwrap()
    });
    assert_eq!(hooked, Some(true));
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "{:?}",
        start.elapsed()
    );
}

/// Work that never completes holds the hook back at most its bound, sections and work together,
/// and the hook says so; a device gone, or made in another process (a forked child's parent's),
/// is not waited for.
#[test]
fn the_hook_s_wait_for_a_cpu_device_s_work_shares_its_bound() {
    let pid = std::process::id();
    let guard = guard();
    let devices = none_in_process();
    let (_never, done) = mpsc::channel();
    let stuck = Running::new(done);
    add(&devices, &stuck);
    let bound = Duration::from_millis(300);
    let start = Instant::now();
    assert_eq!(
        hook(guard, &devices, pid, pid, 0, bound, bound),
        Some(false)
    );
    let waited = start.elapsed();
    assert!(
        waited >= bound && waited < bound + Duration::from_secs(2),
        "the hook waited {waited:?} for a bound of {bound:?}"
    );

    let guard = self::guard();
    let devices = none_in_process();
    let (_never, done) = mpsc::channel();
    let gone = Running::new(done);
    add(&devices, &gone);
    drop(gone);
    let start = Instant::now();
    assert_eq!(
        hook(
            guard,
            &devices,
            pid,
            pid,
            0,
            Duration::from_secs(10),
            Duration::from_secs(10)
        ),
        Some(true)
    );
    assert!(start.elapsed() < Duration::from_secs(1));

    // A device of another process: as a child forked after the parent made it sees it.
    let devices = none_in_process();
    let (_never, done) = mpsc::channel();
    let parents = Running::new(done);
    add(&devices, &parents);
    assert!(devices.take(pid + 1).is_empty());
    assert!(parents.asked().is_none());
}

/// A device whose queues run in this process, which counts the hook's waits for its work.
struct Counted(AtomicUsize);

impl InProcess for Counted {
    fn drain(&self, _deadline: Option<Instant>) -> bool {
        self.0.fetch_add(1, Ordering::SeqCst);
        true
    }
}

/// A [`Counted`] device of this process, added to `devices`.
fn counted(devices: &InProcessDevices) -> Arc<Counted> {
    let device = Arc::new(Counted(AtomicUsize::new(0)));
    let weak: Weak<dyn InProcess> = Arc::<Counted>::downgrade(&device);
    devices.add(weak);
    device
}

/// The hook runs once per registration, and each run after the first follows the destructors
/// registered after its own registration: the first run alone waits for a CPU device's work (a
/// Vulkan call), and a later one finds no device.
#[test]
fn only_the_hook_s_first_run_waits_for_a_cpu_device_s_work() {
    let pid = std::process::id();
    let guard = guard();
    let devices = none_in_process();
    let device = counted(&devices);
    for _ in 0..3 {
        assert_eq!(
            hook(
                guard,
                &devices,
                pid,
                pid,
                0,
                Duration::from_secs(10),
                Duration::from_secs(10)
            ),
            Some(true)
        );
    }
    assert_eq!(device.0.load(Ordering::SeqCst), 1);
}

/// A first run whose wait for the sections outlasts its bound waits for no CPU device's work and
/// still takes the devices: a later run, once the section ended, finds none.
#[test]
fn a_hook_run_whose_sections_outlast_its_bound_still_takes_the_devices() {
    let pid = std::process::id();
    let guard = guard();
    let devices = none_in_process();
    let device = counted(&devices);
    let (entered, end_section) = (mpsc::channel(), mpsc::channel::<()>());
    let section = std::thread::spawn(move || {
        let busy = guard.enter().expect("open");
        entered.0.send(()).unwrap();
        let _ = end_section.1.recv();
        drop(busy);
    });
    entered.1.recv().unwrap();
    assert_eq!(
        hook(
            guard,
            &devices,
            pid,
            pid,
            0,
            Duration::from_millis(100),
            Duration::from_millis(100)
        ),
        Some(false)
    );
    end_section.0.send(()).unwrap();
    section.join().unwrap();
    assert_eq!(
        hook(
            guard,
            &devices,
            pid,
            pid,
            0,
            Duration::from_secs(10),
            Duration::from_secs(10)
        ),
        Some(true)
    );
    assert_eq!(device.0.load(Ordering::SeqCst), 0);
}

/// A child forked (without exec) from a process whose hook is registered, while another of the
/// parent's threads is in a section of the process's guard, exits at once: the hook it inherits
/// checks the process, and the section's thread is not the child's, so the hook neither waits for
/// it nor ends the child (with the exit's status, 3 here) for it.
///
/// The child runs the guard's exit handler alone and ends with `_exit`, not `exit`: the handlers
/// a child inherits are every library's the parent loaded, not only the hook and the C library's.
/// A Vulkan driver's may wait for threads only the parent has (Mesa's thread queues join their
/// threads at exit), so a child calling `exit` while another test's lavapipe device lives in this
/// process never ends.
#[cfg(unix)]
#[test]
fn a_forked_child_s_exit_does_not_wait_for_its_parent_s_sections() {
    exit_guard::register();
    let (entered, done) = (mpsc::channel(), mpsc::channel::<()>());
    let worker = std::thread::spawn(move || {
        let _busy = exit_guard::enter().expect("the process is not exiting");
        entered.0.send(()).unwrap();
        let _ = done.1.recv();
    });
    entered.1.recv().unwrap();
    let start = Instant::now();
    // SAFETY: the child runs only the guard's exit handler (a process id and atomics) and
    // `_exit`, which runs no handler.
    let child = unsafe { libc::fork() };
    if child == 0 {
        exit_guard::run_on_exit(3);
        // SAFETY: as above.
        unsafe { libc::_exit(0) };
    }
    assert!(child > 0, "fork");
    let mut status = 0;
    // SAFETY: `child` is this process's child.
    let reaped = unsafe { libc::waitpid(child, &mut status, 0) };
    let took = start.elapsed();
    done.0.send(()).unwrap();
    worker.join().unwrap();
    assert_eq!(reaped, child);
    assert!(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0);
    assert!(
        took < Duration::from_secs(1),
        "the child's exit took {took:?}: it waited for its parent's section"
    );
}

/// Each entry point has a bit of its own.
#[test]
fn every_entry_point_has_a_bit_of_its_own() {
    let mut bits = 0_u128;
    for &call in Call::ALL {
        let bit = call.bit();
        assert_eq!(bit.count_ones(), 1, "{call:?} has no bit of the word");
        assert_eq!(bits & bit, 0, "{call:?} shares a bit");
        bits |= bit;
    }
}

/// The exit status of a child that registers its hook (its first device), then a destructor (a
/// driver's or a layer's static, made at its first use), then runs `sections`, and exits. The
/// destructor ends the child with 42 when it sees the guard closed (the hook ran first), and 43
/// when it ran first.
#[cfg(unix)]
fn late_destructor_exit(sections: fn()) -> libc::c_int {
    late_destructor_exit_after(|| {}, sections)
}

/// [`late_destructor_exit`], the child running `before` between its hook's registration and the
/// destructor's.
#[cfg(unix)]
fn late_destructor_exit_after(before: fn(), sections: fn()) -> libc::c_int {
    extern "C" fn late_destructor() {
        let code = if exit_guard::closed() { 42 } else { 43 };
        // SAFETY: ends the child at once, from its exit handlers.
        unsafe { libc::_exit(code) };
    }
    // SAFETY: the child registers its hook and the destructor, runs sections (atomics and the
    // C library's `atexit`), and exits.
    let child = unsafe { libc::fork() };
    if child == 0 {
        exit_guard::register();
        before();
        // SAFETY: `late_destructor` is a function with no arguments that never unwinds.
        unsafe { libc::atexit(late_destructor) };
        sections();
        // SAFETY: as above.
        unsafe { libc::exit(0) };
    }
    assert!(child > 0, "fork");
    let mut status = 0;
    // SAFETY: `child` is this process's child.
    let reaped = unsafe { libc::waitpid(child, &mut status, 0) };
    assert_eq!(reaped, child);
    assert!(libc::WIFEXITED(status), "the child did not exit: {status}");
    libc::WEXITSTATUS(status)
}

/// With no section after the destructor's registration, the destructor runs first: the case the
/// registrations below close.
#[cfg(unix)]
#[test]
fn a_destructor_registered_after_the_hook_runs_before_it() {
    assert_eq!(late_destructor_exit(|| {}), 43);
}

/// A section that ends after the first call of an entry point registers the hook again, so a
/// destructor registered before it (by that call) runs after the hook. Its count is not a power
/// of two, so that trigger is the first call's.
#[cfg(unix)]
#[test]
fn a_section_after_an_entry_point_s_first_call_registers_the_hook_again() {
    assert_eq!(
        late_destructor_exit(|| {
            let call = Call::ALL
                .iter()
                .copied()
                .find(|&call| !exit_guard::was_called(call))
                .expect("an entry point no section called");
            // Sections before the destructor, until the next one's count is no power of two.
            while (exit_guard::sections() + 1).is_power_of_two() {
                drop(exit_guard::enter());
            }
            let busy = exit_guard::enter();
            exit_guard::called(call);
            drop(busy);
        }),
        42,
        "the destructor ran before the hook"
    );
}

/// A section whose count reaches a power of two registers the hook again, for statics made at a
/// later call of an entry point already called.
#[cfg(unix)]
#[test]
fn a_section_at_a_power_of_two_registers_the_hook_again() {
    assert_eq!(
        late_destructor_exit(|| loop {
            drop(exit_guard::enter());
            if exit_guard::sections().is_power_of_two() {
                break;
            }
        }),
        42,
        "the destructor ran before the hook"
    );
}

/// An instance made registers the hook again after it, so an exit handler a layer registered at
/// its `vkCreateInstance` (the validation layer's, which forgets every device it knows) runs after
/// the hook: the child's destructor stands for it.
#[cfg(unix)]
#[test]
fn an_instance_made_registers_the_hook_again() {
    assert_eq!(
        late_destructor_exit(exit_guard::instance_made),
        42,
        "the layer's handler ran before the hook"
    );
}

/// The exit of a child that registers an end (which ends it with 42), its hook (its first
/// device), a destructor (a layer's static first made by a call on the program's thread, which
/// registers no hook), and its hook again (a section after an entry point's first call), with a
/// device whose queues run in it: the destructor runs between the hook's two runs, and the
/// device's wait ends the child with 43 when it comes after the destructor. Seen before the hook's
/// later runs found no device: lavapipe processes under the validation layer ended with `SIGABRT`
/// inside the layer's `vkWaitSemaphores`, in the run of the hook's first registration.
#[cfg(unix)]
#[test]
fn no_wait_for_a_cpu_device_s_work_follows_a_destructor_between_the_hook_s_runs() {
    static DESTROYED: AtomicBool = AtomicBool::new(false);
    extern "C" fn end() {
        // SAFETY: ends the child at once, from its exit handlers.
        unsafe { libc::_exit(42) };
    }
    extern "C" fn destructor() {
        DESTROYED.store(true, Ordering::SeqCst);
    }
    /// A device whose wait ends the child with 43 once the destructor ran.
    struct Checked;
    impl InProcess for Checked {
        fn drain(&self, _deadline: Option<Instant>) -> bool {
            if DESTROYED.load(Ordering::SeqCst) {
                // SAFETY: ends the child at once, from its exit handlers.
                unsafe { libc::_exit(43) };
            }
            true
        }
    }
    // SAFETY: the child registers exit handlers and a device (atomics, a lock and the C
    // library's `atexit`), and exits.
    let child = unsafe { libc::fork() };
    if child == 0 {
        // SAFETY: `end` is a function with no arguments that never unwinds.
        unsafe { libc::atexit(end) };
        let device: Arc<dyn InProcess> = Arc::new(Checked);
        exit_guard::runs_in_process(Arc::downgrade(&device));
        exit_guard::register();
        // SAFETY: `destructor` is a function with no arguments that never unwinds.
        unsafe { libc::atexit(destructor) };
        exit_guard::register();
        // SAFETY: as above; `device` stays live, since `exit` drops nothing of this frame.
        unsafe { libc::exit(0) };
    }
    assert!(child > 0, "fork");
    let mut status = 0;
    // SAFETY: `child` is this process's child.
    let reaped = unsafe { libc::waitpid(child, &mut status, 0) };
    assert_eq!(reaped, child);
    assert!(libc::WIFEXITED(status), "the child did not exit: {status}");
    assert_eq!(
        libc::WEXITSTATUS(status),
        42,
        "a wait for the device's work followed the destructor"
    );
}

/// A child's exit, as its parent saw it: its status, how long it took from the child's call of
/// `exit` to its end, and what it wrote on standard error.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
struct ChildExit {
    status: libc::c_int,
    took: Duration,
    stderr: String,
}

/// The clock every process reads alike (`CLOCK_MONOTONIC`), in nanoseconds.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn monotonic_ns() -> u64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: writes the clock's time into `now`, which lives for the call.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
    now.tv_sec as u64 * 1_000_000_000 + now.tv_nsec as u64
}

/// Fork a child that runs `child` (its exit handlers, its devices, what is in progress as it
/// exits), then calls the C library's `exit` with `status`; its standard error goes to a pipe.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn child_exit(child: fn(), status: libc::c_int) -> ChildExit {
    use std::io::Read;
    use std::os::fd::FromRawFd;
    let (mut stderr, mut called) = ([0; 2], [0; 2]);
    // SAFETY: each makes a pipe into an array of two descriptors that lives for the call.
    unsafe {
        assert_eq!(libc::pipe(stderr.as_mut_ptr()), 0);
        assert_eq!(libc::pipe(called.as_mut_ptr()), 0);
    }
    // SAFETY: the child moves its standard error to the pipe, runs `child` (atomics, locks, a
    // thread, the C library's `atexit`), writes its clock, and exits.
    let pid = unsafe { libc::fork() };
    if pid == 0 {
        // SAFETY: the pipes' descriptors are this process's.
        unsafe {
            libc::dup2(stderr[1], 2);
            libc::close(stderr[0]);
            libc::close(called[0]);
        }
        child();
        let now = monotonic_ns().to_ne_bytes();
        // SAFETY: writes `now`, live for the call, to the pipe; then exits.
        unsafe {
            libc::write(called[1], now.as_ptr().cast(), now.len());
            libc::exit(status);
        }
    }
    assert!(pid > 0, "fork");
    // SAFETY: the write ends are this process's, and the child has its own.
    unsafe {
        libc::close(stderr[1]);
        libc::close(called[1]);
    }
    let mut raw = 0;
    // SAFETY: `pid` is this process's child.
    let reaped = unsafe { libc::waitpid(pid, &mut raw, 0) };
    let ended = monotonic_ns();
    assert_eq!(reaped, pid);
    let read = |fd| {
        // SAFETY: `fd` is the read end of a pipe this process owns, read once and closed.
        let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        bytes
    };
    let stderr = String::from_utf8_lossy(&read(stderr[0])).into_owned();
    assert!(
        libc::WIFEXITED(raw),
        "the child did not exit: {raw}; its standard error: {stderr}"
    );
    let called: [u8; 8] = read(called[0])
        .try_into()
        .expect("the child's clock at its exit");
    ChildExit {
        status: libc::WEXITSTATUS(raw),
        took: Duration::from_nanos(ended.saturating_sub(u64::from_ne_bytes(called))),
        stderr,
    }
}

/// Where the hook learns the exit's status (Linux with the GNU C library), an exit while the
/// driver runs the runtime's work (a CPU device's work in flight, a section in progress on another
/// thread) ends the process at once with that status, says so on standard error in one line, and
/// runs no handler registered before the hook: Metal's process ends within 2 ms of its `exit`.
/// Before, the hook waited 2 s, returned, and the handler ran.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn a_busy_exit_ends_at_once_with_its_status() {
    /// An exit handler the program registered before the runtime's (a terminal's restore): it
    /// ends the child with 43.
    extern "C" fn registered_before() {
        // SAFETY: ends the child at once, from its exit handlers.
        unsafe { libc::_exit(43) };
    }
    /// A CPU device whose submitted work never completes.
    struct Stuck;
    impl InProcess for Stuck {
        fn drain(&self, deadline: Option<Instant>) -> bool {
            let deadline = deadline.expect("the hook's bound is a deadline");
            std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
            false
        }
    }
    fn work_in_flight() {
        // SAFETY: a function with no arguments that never unwinds.
        unsafe { libc::atexit(registered_before) };
        let device: Arc<dyn InProcess> = Arc::new(Stuck);
        exit_guard::runs_in_process(Arc::downgrade(&device));
        // Live through the exit, as a device the program holds.
        std::mem::forget(device);
        exit_guard::register();
    }
    fn section_in_progress() {
        // SAFETY: as above.
        unsafe { libc::atexit(registered_before) };
        exit_guard::register();
        let (entered, held) = mpsc::channel();
        std::thread::spawn(move || {
            let _busy = exit_guard::enter().expect("the child is not exiting");
            entered.send(()).unwrap();
            loop {
                std::thread::park();
            }
        });
        held.recv().unwrap();
    }
    let children: [(&str, fn()); 2] = [
        ("a CPU device's work", work_in_flight),
        ("a section", section_in_progress),
    ];
    for (busy, child) in children {
        for status in [0, 7] {
            let exit = child_exit(child, status);
            assert_eq!(
                exit.status, status,
                "{busy} in progress: a handler registered before the hook ran, or the status \
                 changed; its standard error: {}",
                exit.stderr
            );
            assert!(
                exit.took < Duration::from_millis(100),
                "{busy} in progress: the exit took {:?}",
                exit.took
            );
            assert_eq!(
                exit.stderr,
                exit_guard::AT_ONCE_NOTICE,
                "{busy} in progress"
            );
        }
    }
}

/// The runtime's own threads may reach the device only inside a section or while they run the
/// program's code; the program's threads always may.
#[test]
fn runtime_threads_reach_the_device_only_inside_sections() {
    assert!(exit_guard::may_call(), "the program's thread");
    let answers = exit_guard::spawn(std::thread::Builder::new(), || {
        let outside = exit_guard::may_call();
        let busy = exit_guard::enter().expect("the process is not exiting");
        let inside = exit_guard::may_call();
        drop(busy);
        let mut in_program = false;
        exit_guard::program(|| in_program = exit_guard::may_call());
        let after_program = exit_guard::may_call();
        let mut after_panic = true;
        exit_guard::program(|| panic!("the program's code panics"));
        after_panic &= !exit_guard::may_call();
        (outside, inside, in_program, after_program, after_panic)
    })
    .unwrap()
    .join()
    .unwrap();
    assert_eq!(answers, (false, true, true, false, true));
}

/// A runtime thread starts at the default quality of service, whatever the class of the thread
/// that started it, as a plain thread does: `std::thread` makes both with `pthread_create` and no
/// class in their attributes, and such a thread starts at `QOS_CLASS_DEFAULT` (`<sys/qos.h>`). So
/// none runs below the default class for having been started from a program's thread at the
/// background or utility class, with nothing done for it (`exit_guard::spawn`), and none takes a
/// higher class from its creator.
#[cfg(target_os = "macos")]
#[test]
fn a_runtime_thread_starts_at_the_default_quality_of_service_whoever_started_it() {
    use libc::qos_class_t as Q;
    fn class() -> u32 {
        let mut class = Q::QOS_CLASS_UNSPECIFIED;
        let mut relative = 0;
        // SAFETY: the calling thread's own handle, and two out-parameters that live for the call.
        let read = unsafe {
            libc::pthread_get_qos_class_np(libc::pthread_self(), &mut class, &mut relative)
        };
        assert_eq!(read, 0, "pthread_get_qos_class_np");
        class as u32
    }
    let default = Q::QOS_CLASS_DEFAULT as u32;
    for start in [
        Q::QOS_CLASS_BACKGROUND,
        Q::QOS_CLASS_UTILITY,
        Q::QOS_CLASS_DEFAULT,
        Q::QOS_CLASS_USER_INITIATED,
        Q::QOS_CLASS_USER_INTERACTIVE,
    ] {
        // A plain thread's class and a runtime thread's, each started from a thread at `start`.
        let (creator, plain, runtime) = std::thread::spawn(move || {
            // SAFETY: sets the calling thread's own class; no pointer.
            assert_eq!(unsafe { libc::pthread_set_qos_class_self_np(start, 0) }, 0);
            let plain = std::thread::spawn(class).join().unwrap();
            let runtime = exit_guard::spawn(std::thread::Builder::new(), class)
                .unwrap()
                .join()
                .unwrap();
            (class(), plain, runtime)
        })
        .join()
        .unwrap();
        assert_eq!(creator, start as u32, "the creator's class");
        assert_eq!(
            (plain, runtime),
            (default, default),
            "a plain thread's and a runtime thread's from {start:?}"
        );
    }
}

/// The threads that take a command buffer from its commit to its handlers run at the
/// user-interactive quality of service from their start, whatever the class of the thread that
/// started them, as read from inside each: the scheduler's, each `VkQueue`'s waiter (which makes
/// the completion turns), and the handler pool's first thread and a later one, which also run their
/// items there. Another of the runtime's threads, the reaper, runs as [`exit_guard::spawn`]
/// starts a thread, at the default class.
#[cfg(target_os = "macos")]
#[test]
fn the_threads_that_take_command_buffers_to_their_handlers_run_at_the_user_interactive_class() {
    use crate::command_buffer::completion::Completion;
    use crate::contexts::{Contexts, Run, THREADS};
    use crate::pipeline::reaper::Reaper;
    use crate::scheduler::Scheduler;
    use libc::qos_class_t as Q;
    use std::thread::ThreadId;

    // The calling thread's requested class (`<sys/qos.h>`), which `libc` does not declare.
    unsafe extern "C" {
        safe fn qos_class_self() -> u32;
    }
    /// A pool's item.
    struct Item(Box<dyn FnOnce() + Send>);
    impl Run for Item {
        fn run(self) {
            (self.0)();
        }
    }
    /// Sends the class of the thread that drops it: the reaper's, which lets go of it.
    struct Reaped(mpsc::Sender<u32>);
    impl Drop for Reaped {
        fn drop(&mut self) {
            let _ = self.0.send(qos_class_self());
        }
    }
    const WAIT: Duration = Duration::from_secs(20);
    let interactive = Q::QOS_CLASS_USER_INTERACTIVE as u32;

    for start in [
        Q::QOS_CLASS_BACKGROUND,
        Q::QOS_CLASS_UTILITY,
        Q::QOS_CLASS_DEFAULT,
        Q::QOS_CLASS_USER_INITIATED,
        Q::QOS_CLASS_USER_INTERACTIVE,
    ] {
        // On a thread at `start`, standing for the program's that makes a device.
        let (started, items, reaper, plain) = std::thread::spawn(move || {
            // SAFETY: sets the calling thread's own class; no pointer.
            assert_eq!(unsafe { libc::pthread_set_qos_class_self_np(start, 0) }, 0);
            let (note, notes) = mpsc::channel();
            exit_guard::ON_START.with(|on_start| {
                *on_start.borrow_mut() = Some(Arc::new(move || {
                    let thread = std::thread::current();
                    let name = thread.name().unwrap_or_default().to_owned();
                    let _ = note.send((name, qos_class_self()));
                }));
            });
            let scheduler = Scheduler::start(1, false).unwrap();
            let completion = Completion::start(2).unwrap();
            // The pool's first thread runs an item that blocks; another queue's item then starts
            // a thread of its own.
            let contexts: Contexts<Item> = Contexts::start(THREADS).unwrap();
            let (ran, runs) = mpsc::channel::<(ThreadId, u32)>();
            let (release, released) = mpsc::channel::<()>();
            let blocking = ran.clone();
            contexts.post(
                1,
                Item(Box::new(move || {
                    blocking
                        .send((std::thread::current().id(), qos_class_self()))
                        .unwrap();
                    released.recv_timeout(WAIT).unwrap();
                })),
            );
            let first = runs.recv_timeout(WAIT).unwrap();
            contexts.post(
                2,
                Item(Box::new(move || {
                    ran.send((std::thread::current().id(), qos_class_self()))
                        .unwrap();
                })),
            );
            let later = runs.recv_timeout(WAIT).unwrap();
            release.send(()).unwrap();
            contexts.stop();
            scheduler.stop();
            completion.stop();
            exit_guard::ON_START.with(|on_start| on_start.borrow_mut().take());
            let started: Vec<(String, u32)> = notes.try_iter().collect();

            let reaper = Reaper::default();
            let (dropped, drops) = mpsc::channel();
            reaper.start().unwrap().reap(Box::new(Reaped(dropped)));
            let reaped = drops.recv_timeout(WAIT).unwrap();
            reaper.stop();
            let plain = exit_guard::spawn(std::thread::Builder::new(), || qos_class_self())
                .unwrap()
                .join()
                .unwrap();
            (started, [first, later], reaped, plain)
        })
        .join()
        .unwrap();

        let mut names: Vec<&str> = started.iter().map(|(name, _)| name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "metal2vulkan-handlers",
                "metal2vulkan-handlers",
                "metal2vulkan-queue-wait-0",
                "metal2vulkan-queue-wait-1",
                "metal2vulkan-scheduler",
            ],
            "from {start:?}"
        );
        for (name, class) in &started {
            assert_eq!(*class, interactive, "{name} from {start:?}");
        }
        let [(first, first_class), (later, later_class)] = items;
        assert_ne!(first, later, "the later item ran on a thread of its own");
        assert_eq!(
            (first_class, later_class),
            (interactive, interactive),
            "the pool's items from {start:?}"
        );
        assert_eq!(
            (reaper, plain),
            (Q::QOS_CLASS_DEFAULT as u32, Q::QOS_CLASS_DEFAULT as u32),
            "the reaper and a runtime thread from {start:?}"
        );
    }
}

/// On Linux the threads that take a command buffer from its commit to its handlers run with the
/// utilization floor at its top from their start, as read from inside each (`sched_getattr(2)`):
/// the scheduler's, each `VkQueue`'s waiter (which makes the completion turns), and the handler
/// pool's first thread and a later one, which also run their items there. Each keeps the policy and
/// nice value of the thread that started it, and has `SCHED_FLAG_RESET_ON_FORK`: a thread one of
/// them starts as the runtime starts its own ([`exit_guard::spawn`]) begins with no floor and
/// without the setting, at the same policy and nice value. From threads at `SCHED_OTHER`,
/// `SCHED_BATCH` and `SCHED_IDLE`, each where the test's thread may enter it (a session a daemon
/// such as ananicy-cpp put in `SCHED_IDLE` may not leave it), `SCHED_IDLE` always. Where the kernel
/// refuses the floor (built without the clamps), every thread reads none, and the kernel refuses
/// one the test asks for itself.
#[cfg(target_os = "linux")]
#[test]
fn the_threads_that_take_command_buffers_to_their_handlers_run_with_the_utilization_floor() {
    use crate::command_buffer::completion::Completion;
    use crate::contexts::{Contexts, Run, THREADS};
    use crate::exit_guard::SchedAttr;
    use crate::scheduler::Scheduler;

    // `<linux/sched.h>`.
    const SCHED_OTHER: u32 = 0;
    const SCHED_BATCH: u32 = 3;
    const SCHED_IDLE: u32 = 5;
    const RESET_ON_FORK: u64 = 0x01;
    const KEEP_POLICY: u64 = 0x08;
    const KEEP_PARAMS: u64 = 0x10;
    const UTIL_CLAMP_MIN: u64 = 0x20;
    const WAIT: Duration = Duration::from_secs(20);

    /// A pool's item.
    struct Item(Box<dyn FnOnce() + Send>);
    impl Run for Item {
        fn run(self) {
            (self.0)();
        }
    }
    /// What a thread reads of itself.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Read {
        policy: u32,
        nice: i32,
        floor: u32,
        reset: bool,
    }
    fn read() -> Read {
        let attr = SchedAttr::own().expect("sched_getattr");
        Read {
            policy: attr.sched_policy,
            nice: attr.sched_nice,
            floor: attr.sched_util_min,
            reset: attr.sched_flags & RESET_ON_FORK != 0,
        }
    }
    /// This thread's reading, and that of a thread it starts as the runtime starts its own.
    fn with_child() -> (Read, Read) {
        let child = exit_guard::spawn(std::thread::Builder::new(), read)
            .unwrap()
            .join()
            .unwrap();
        (read(), child)
    }

    let mut entered = Vec::new();
    for policy in [SCHED_OTHER, SCHED_BATCH, SCHED_IDLE] {
        // On a thread at `policy`, standing for the program's that makes a device.
        let run = std::thread::spawn(move || {
            let param = libc::sched_param { sched_priority: 0 };
            // SAFETY: sets the calling thread's own policy from `param`, which lives for the call.
            if unsafe { libc::sched_setscheduler(0, policy as i32, &param) } != 0 {
                return None;
            }
            let creator = read();
            assert_eq!(creator.policy, policy, "the creator's policy");
            let (note, notes) = mpsc::channel();
            exit_guard::ON_START.with(|on_start| {
                *on_start.borrow_mut() = Some(Arc::new(move || {
                    let thread = std::thread::current();
                    let name = thread.name().unwrap_or_default().to_owned();
                    let _ = note.send((name, with_child()));
                }));
            });
            let scheduler = Scheduler::start(1, false).unwrap();
            let completion = Completion::start(2).unwrap();
            // The pool's first thread runs an item that blocks; another queue's item then starts
            // a thread of its own.
            let contexts: Contexts<Item> = Contexts::start(THREADS).unwrap();
            let (ran, runs) = mpsc::channel();
            let (release, released) = mpsc::channel::<()>();
            let blocking = ran.clone();
            contexts.post(
                1,
                Item(Box::new(move || {
                    let reading = with_child();
                    blocking
                        .send((std::thread::current().id(), reading))
                        .unwrap();
                    released.recv_timeout(WAIT).unwrap();
                })),
            );
            let first = runs.recv_timeout(WAIT).unwrap();
            contexts.post(
                2,
                Item(Box::new(move || {
                    let reading = with_child();
                    ran.send((std::thread::current().id(), reading)).unwrap();
                })),
            );
            let later = runs.recv_timeout(WAIT).unwrap();
            release.send(()).unwrap();
            contexts.stop();
            scheduler.stop();
            completion.stop();
            exit_guard::ON_START.with(|on_start| on_start.borrow_mut().take());
            let started: Vec<(String, (Read, Read))> = notes.try_iter().collect();
            // Where the threads read no floor: whether the kernel takes one asked for here.
            let mut ask = SchedAttr::own().expect("sched_getattr");
            ask.sched_flags = KEEP_POLICY | KEEP_PARAMS | UTIL_CLAMP_MIN;
            ask.sched_util_min = 1024;
            let refused = started.iter().all(|(_, (own, _))| own.floor == 0) && !ask.set_own();
            Some((creator, started, [first, later], refused))
        })
        .join()
        .unwrap();
        let Some((creator, started, items, refused)) = run else {
            eprintln!("policy {policy}: this thread may not enter it");
            continue;
        };
        entered.push(policy);

        let mut names: Vec<&str> = started.iter().map(|(name, _)| name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "metal2vulkan-handlers",
                "metal2vulkan-handlers",
                "metal2vulkan-queue-wait-0",
                "metal2vulkan-queue-wait-1",
                "metal2vulkan-scheduler",
            ],
            "from policy {policy}"
        );
        let [(first, first_read), (later, later_read)] = items;
        assert_ne!(first, later, "the later item ran on a thread of its own");
        let (floor, reset) = if refused { (0, false) } else { (1024, true) };
        let readings = started
            .iter()
            .map(|(name, reading)| (name.as_str(), *reading))
            .chain([
                ("the first item", first_read),
                ("the later item", later_read),
            ]);
        for (name, (own, child)) in readings {
            let kept = Read {
                floor,
                reset,
                ..creator
            };
            assert_eq!(own, kept, "{name} from policy {policy}: {creator:?}");
            let started = Read {
                floor: 0,
                reset: false,
                nice: if reset {
                    creator.nice.max(0)
                } else {
                    creator.nice
                },
                ..creator
            };
            assert_eq!(
                child, started,
                "a thread {name} started, from policy {policy}"
            );
        }
        eprintln!("policy {policy}: floor {floor}, {creator:?}");
    }
    assert!(entered.contains(&SCHED_IDLE), "entered {entered:?}");
}

/// In builds with debug assertions, a runtime thread that reaches the device outside a section
/// ends the process.
#[cfg(all(unix, debug_assertions))]
#[test]
fn a_runtime_thread_reaching_the_device_outside_a_section_aborts() {
    // SAFETY: the child starts one thread, which checks and aborts, and otherwise exits.
    let child = unsafe { libc::fork() };
    if child == 0 {
        let checked = exit_guard::spawn(std::thread::Builder::new(), exit_guard::check)
            .map(|thread| thread.join());
        let code = if checked.is_ok() { 43 } else { 44 };
        // SAFETY: ends the child at once.
        unsafe { libc::_exit(code) };
    }
    assert!(child > 0, "fork");
    let mut status = 0;
    // SAFETY: `child` is this process's child.
    let reaped = unsafe { libc::waitpid(child, &mut status, 0) };
    assert_eq!(reaped, child);
    assert!(
        libc::WIFSIGNALED(status) && libc::WTERMSIG(status) == libc::SIGABRT,
        "the child was not aborted: {status}"
    );
}

/// In builds with debug assertions, a release of the program's (a deallocator) on a thread command
/// buffers' completion depends on ends the process: its wait for such a command buffer would
/// never return. It runs on any other thread, and on that one inside the program's own code (a
/// submission handler's drops).
#[cfg(all(unix, debug_assertions))]
#[test]
fn a_release_of_the_program_s_on_a_thread_completion_depends_on_aborts() {
    let mut ran = 0;
    exit_guard::program_release(|| ran += 1);
    let in_program = exit_guard::spawn(std::thread::Builder::new(), || {
        exit_guard::serves_completion();
        let mut ran = false;
        exit_guard::program(|| exit_guard::program_release(|| ran = true));
        ran
    })
    .unwrap()
    .join()
    .unwrap();
    assert_eq!((ran, in_program), (1, true));
    // SAFETY: the child starts one thread, which releases and aborts, and otherwise exits.
    let child = unsafe { libc::fork() };
    if child == 0 {
        let released = exit_guard::spawn(std::thread::Builder::new(), || {
            exit_guard::serves_completion();
            exit_guard::program_release(|| {});
        })
        .map(|thread| thread.join());
        let code = if released.is_ok() { 43 } else { 44 };
        // SAFETY: ends the child at once.
        unsafe { libc::_exit(code) };
    }
    assert!(child > 0, "fork");
    let mut status = 0;
    // SAFETY: `child` is this process's child.
    let reaped = unsafe { libc::waitpid(child, &mut status, 0) };
    assert_eq!(reaped, child);
    assert!(
        libc::WIFSIGNALED(status) && libc::WTERMSIG(status) == libc::SIGABRT,
        "the child was not aborted: {status}"
    );
}

/// The runtime's sources, without comments, by path relative to `src`; its tests left out.
fn runtime_sources() -> Vec<(String, String)> {
    fn walk(dir: &std::path::Path, root: &std::path::Path, out: &mut Vec<(String, String)>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let relative = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .into_owned();
            if path.is_dir() {
                if relative != "tests" {
                    walk(&path, root, out);
                }
            } else if relative.ends_with(".rs") {
                let text = std::fs::read_to_string(&path).unwrap();
                let code: Vec<&str> = text
                    .lines()
                    .map(|line| line.find("//").map_or(line, |at| &line[..at]))
                    .collect();
                out.push((relative, code.join("\n")));
            }
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut sources = Vec::new();
    walk(&root, &root, &mut sources);
    assert!(sources.len() > 50, "the runtime's sources were not found");
    sources
}

/// The places in the runtime's sources, outside `allowed`, where `needle` is followed (after
/// spaces and line breaks) by one of `then`.
fn uses(allowed: &[&str], needle: &str, then: &[&str]) -> Vec<String> {
    let mut found = Vec::new();
    for (path, code) in runtime_sources() {
        if allowed.contains(&path.as_str()) {
            continue;
        }
        let compact: String = code.split_whitespace().collect();
        let mut rest = compact.as_str();
        while let Some(at) = rest.find(needle) {
            let after = &rest[at + needle.len()..];
            if then.is_empty() || then.iter().any(|next| after.starts_with(next)) {
                found.push(format!("{path}: {}", &rest[at..(at + 60).min(rest.len())]));
            }
            rest = after;
        }
    }
    found
}

/// Every release of a Vulkan object goes through `release.rs`, which runs it inside a section
/// and marks its entry point: no other file calls a `vkDestroy*` or `vkFree*` entry point.
#[test]
fn every_release_goes_through_the_release_door() {
    let found: Vec<String> = [".destroy_", ".free_"]
        .iter()
        .flat_map(|needle| uses(&["release.rs"], needle, &[]))
        .collect();
    assert!(
        found.is_empty(),
        "Vulkan releases outside release.rs: {found:#?}"
    );
}

/// Every image, render pass and framebuffer the runtime makes goes through `creation.rs`, whose
/// entry points are in the exit guard's table and among a host step's: no other file calls
/// `vkCreateImage`, `vkCreateRenderPass`, `vkCreateRenderPass2` or `vkCreateFramebuffer`, but the
/// device's priming, on the thread that makes the device.
#[test]
fn every_image_render_pass_and_framebuffer_goes_through_the_creation_door() {
    let found: Vec<String> = [
        ".create_image(",
        ".create_render_pass",
        ".create_framebuffer(",
    ]
    .iter()
    .flat_map(|needle| uses(&["creation.rs", "device/prime.rs"], needle, &[]))
    .collect();
    assert!(
        found.is_empty(),
        "images, render passes or framebuffers made outside creation.rs: {found:#?}"
    );
}

/// Every buffer, image view, memory allocation and binding the runtime makes goes through
/// `creation.rs` too, where each counts itself at its caller's place (`churn_counts.rs`): a
/// making outside the door would be driver work no call of the API could be held to. The device's
/// priming makes its own, on the thread that makes the device.
#[test]
fn every_buffer_view_allocation_and_binding_goes_through_the_creation_door() {
    let found: Vec<String> = [
        ".create_buffer(",
        ".create_image_view(",
        ".allocate_memory(",
        ".bind_image_memory(",
        ".bind_buffer_memory(",
    ]
    .iter()
    .flat_map(|needle| uses(&["creation.rs", "device/prime.rs"], needle, &[]))
    .collect();
    assert!(
        found.is_empty(),
        "buffers, views, allocations or bindings made outside creation.rs: {found:#?}"
    );
}

/// Every instance the runtime creates goes through `device/create.rs`'s door (`new_instance`),
/// which registers the exit hook again after it: a layer registers an exit handler at each
/// `vkCreateInstance`. The runtime's sources call `vkCreateInstance` once, in that door, and the
/// door registers after the call.
#[test]
fn every_instance_goes_through_the_instance_door() {
    let found = uses(&[], ".create_instance(", &[]);
    assert!(
        found.len() == 1 && found[0].starts_with("device/create.rs: "),
        "vkCreateInstance called outside device/create.rs's door: {found:#?}"
    );
    let (_, code) = runtime_sources()
        .into_iter()
        .find(|(path, _)| path == "device/create.rs")
        .unwrap();
    let compact: String = code.split_whitespace().collect();
    let door = &compact[compact.find("fnnew_instance(").expect("the door")..];
    let door = &door[..door[1..].find("fn").map_or(door.len(), |at| at + 1)];
    let call = door.find(".create_instance(").expect("the door's call");
    let registers = door
        .find("exit_guard::instance_made()")
        .expect("the door's registration");
    assert!(
        call < registers,
        "the door registers before its call: {door}"
    );
}

/// Every command a `VkCommandBuffer` takes that the runtime names, through `ash`'s tables or a
/// function pointer it loads, goes through the devices' slots (`recorder`), which route it
/// (`recorder::Command::ALL`): no file outside `recorder/` names one that is not routed.
#[test]
fn every_command_the_runtime_records_is_routed() {
    /// `name` lowercase, without `vk` and underscores, nor an `ext` or `khr` suffix: an
    /// extension's command and the core one it became compare equal.
    fn plain(name: &str) -> String {
        let lower: String = name.chars().filter(|c| *c != '_').collect::<String>();
        let lower = lower.to_lowercase();
        let lower = lower.strip_prefix("vk").unwrap_or(&lower).to_owned();
        let lower = lower.strip_prefix("pfnvk").unwrap_or(&lower).to_owned();
        ["ext", "khr"]
            .iter()
            .find_map(|suffix| lower.strip_suffix(suffix))
            .unwrap_or(&lower)
            .to_owned()
    }
    let routed: Vec<String> = crate::recorder::Command::ALL
        .iter()
        .map(|command| plain(command.name().to_str().unwrap()))
        .collect();
    let mut unrouted = Vec::new();
    for (path, code) in runtime_sources() {
        if path.starts_with("recorder") {
            continue;
        }
        let mut rest = code.as_str();
        while let Some(at) = rest.find(['.', 'P']) {
            let after = &rest[at + 1..];
            let word: String = after
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            let named = match &rest[at..=at] {
                "." if word.starts_with("cmd_")
                    || word == "begin_command_buffer"
                    || word == "end_command_buffer" =>
                {
                    Some(word.clone())
                }
                "P" if word.starts_with("FN_vkCmd") => Some(format!("P{word}")),
                _ => None,
            };
            if let Some(named) = named {
                if !routed.contains(&plain(&named)) {
                    unrouted.push(format!("{path}: {named}"));
                }
            }
            rest = after;
        }
    }
    assert!(
        unrouted.is_empty(),
        "commands named outside recorder/ that it does not route: {unrouted:#?}"
    );
}

/// Every thread the runtime starts is started by `exit_guard::spawn`, so the check knows it, and
/// every call of the program's code goes through `exit_guard::program`, which catches its panics.
#[test]
fn threads_and_the_program_s_code_go_through_the_guard() {
    let spawns = uses(&["exit_guard.rs"], ".spawn(", &[]);
    assert!(spawns.is_empty(), "threads started elsewhere: {spawns:#?}");
    let spawns = uses(&["exit_guard.rs"], "thread::spawn(", &[]);
    assert!(spawns.is_empty(), "threads started elsewhere: {spawns:#?}");
    let calls = uses(&["exit_guard.rs"], "catch_unwind(", &[]);
    assert!(
        calls.is_empty(),
        "the program's code called elsewhere: {calls:#?}"
    );
}

/// The device's functions are kept only in `device/functions.rs`, whose door
/// (`Functions::get`, behind `DeviceShared::vk`) checks the caller and
/// hands a section the exit guard's table alone, and in the parts a device is assembled from.
#[test]
fn the_device_s_functions_are_reached_through_checked_accessors() {
    let kept = uses(
        &["device/functions.rs", "device/mod.rs"],
        ":ash::Device,",
        &[],
    );
    assert!(
        kept.is_empty(),
        "the device's functions kept elsewhere: {kept:#?}"
    );
}

/// The modules that may keep a copy of the device's functions by value, each with why the copy
/// cannot be called after the device's teardown. Any other module that keeps one (a field, a
/// tuple field, a type argument or a by-value parameter) fails
/// `only_the_listed_modules_keep_the_device_s_functions`; so does a listed module that keeps none
/// any more.
const KEEPERS: [(&str, &str); 23] = [
    (
        "binding.rs",
        "the push command (vkCmdPushDescriptorSet), a call's parameter (`push_set`) and the \
         device's lookup of it as it is made (`push_descriptor_command`)",
    ),
    (
        "device/adopt.rs",
        "the consumer's loader entry, read while the device is adopted",
    ),
    (
        "device/compiles.rs",
        "the driver's pipeline makings, found by the device's handle until the runtime's last \
         hold of the device lets go in its teardown (`unregister`); a making for a device \
         without a record fails before any driver",
    ),
    ("device/functions.rs", "the device's functions themselves"),
    (
        "device/mod.rs",
        "the device's shared state, which the teardown ends",
    ),
    (
        "device/prime.rs",
        "a priming call's parameter, while the device is made",
    ),
    (
        "device/relay.rs",
        "the queues' flights, taken back from the instance's relay first in the teardown",
    ),
    (
        "device/submit_stalls.rs",
        "the empty submissions that take a driver's stalls idle, made by the scheduler's thread \
         inside a section; the teardown waits for the device to idle, the thread having stopped, \
         before it releases their semaphores",
    ),
    (
        "draw_source.rs",
        "a draw's recording (vkCmd*), held by what holds its command buffer",
    ),
    (
        "dynamic_state.rs",
        "the device's extended dynamic state commands (DeviceShared)",
    ),
    (
        "freed_memory.rs",
        "the watch of the program's marks and the release of freed memory, posted to the handler \
         threads; the teardown closes the parking first and waits for those threads, and one that \
         starts after finds the parking closed and calls nothing",
    ),
    (
        "generated_commands.rs",
        "the device's generated-commands loader (DeviceShared)",
    ),
    ("gpu_time.rs", "the device's timing (DeviceShared)"),
    (
        "host_memory.rs",
        "the device's host-memory importer (DeviceShared)",
    ),
    (
        "index_bounds.rs",
        "a draw's recording (vkCmd*), held by what holds its command buffer",
    ),
    (
        "index_rewrite.rs",
        "a draw's recording (vkCmd*), held by what holds its command buffer",
    ),
    (
        "pipeline/creation_log.rs",
        "MoltenVK's statistics call, inside a call that borrows the device",
    ),
    (
        "recorder/mod.rs",
        "the device's mesh draw commands, routed through its slot (DeviceShared)",
    ),
    (
        "recorder/thunks.rs",
        "a slot's driver entry points, called by its thunks and cleared at the slot's release",
    ),
    (
        "render.rs",
        "a draw's recording (vkCmd*), held by what holds its command buffer",
    ),
    (
        "render/inline_splits.rs",
        "an inline instance's draws' recordings, kept until its group ends, held by its encoder",
    ),
    (
        "render/pending.rs",
        "a pending draw's recording, held by its command buffer",
    ),
    (
        "scheduler/mod.rs",
        "the scheduler's nudge, signalled and waited for only while its thread waits on the \
         GPU, a thread the teardown stops first",
    ),
];

/// The device itself: a holder of it keeps the device alive, so it is no copy of its functions.
const THE_DEVICE: [&str; 3] = ["Device", "DeviceShared", "InstanceShared"];

/// The runtime's sources as `uses` reads them (no comments, no whitespace), `use` declarations
/// removed: they name types without keeping them.
fn compact_sources() -> Vec<(String, String)> {
    runtime_sources()
        .into_iter()
        .map(|(path, code)| {
            let compact: String = code.split_whitespace().collect();
            (path, without_uses(&compact))
        })
        .collect()
}

/// `code` without its `use` declarations: `use` where an item starts (after `;`, `{`, `}`, `]`,
/// a `pub` or `pub(..)`, or at the start), through its `;`.
fn without_uses(code: &str) -> String {
    let b = code.as_bytes();
    let mut out = String::with_capacity(code.len());
    let mut i = 0;
    let mut kept = 0;
    while i < b.len() {
        let item_start = i == 0
            || matches!(b[i - 1], b';' | b'{' | b'}' | b']' | b')')
            || (i >= 3 && b[i - 3..i] == *b"pub");
        if item_start && b[i..].starts_with(b"use") {
            if let Some(end) = code[i..].find(';') {
                // A path list only (`a::b::{C,D as E}`), not a field or an expression.
                let path = &code[i + 3..i + end];
                let listed = path.bytes().all(|c| {
                    c.is_ascii_alphanumeric()
                        || matches!(c, b'_' | b':' | b'{' | b'}' | b',' | b'*')
                }) && !path.replace("::", "").contains(':');
                if listed {
                    out.push_str(&code[kept..i]);
                    i += end + 1;
                    kept = i;
                    continue;
                }
            }
        }
        i += 1;
    }
    out.push_str(&code[kept..]);
    out
}

/// The start and end of each path (`a::b::C`) in `code`.
fn paths(code: &str) -> Vec<(usize, usize)> {
    let b = code.as_bytes();
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let first = |c: u8| c.is_ascii_alphabetic() || c == b'_';
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if !first(b[i]) {
            i += 1;
            continue;
        }
        let start = i;
        loop {
            while i < b.len() && ident(b[i]) {
                i += 1;
            }
            if i + 2 < b.len() && b[i] == b':' && b[i + 1] == b':' && first(b[i + 2]) {
                i += 2;
            } else {
                break;
            }
        }
        out.push((start, i));
    }
    out
}

/// Whether `path` names a function table: a Vulkan function pointer, ash's device functions or a
/// device extension's, or one of the runtime's types in `own` (bare, or behind modules only).
fn is_table(path: &str, own: &[String]) -> bool {
    let segments: Vec<&str> = path.split("::").collect();
    let last = segments.last().copied().unwrap_or_default();
    if last.starts_with("PFN_") {
        return true;
    }
    if segments.first() == Some(&"ash")
        && segments.get(1) != Some(&"vk")
        && ["Device", "DeviceFn", "InstanceFn"].contains(&last)
    {
        return true;
    }
    own.iter().any(|name| name == last)
        && segments[..segments.len() - 1]
            .iter()
            .all(|segment| segment.starts_with(|c: char| c.is_ascii_lowercase()))
}

/// Each place in `code` that keeps a function table by value: a field's or a by-value
/// parameter's type, a type argument, a tuple's member, or a function-pointer type.
fn stored(code: &str, own: &[String]) -> Vec<String> {
    let b = code.as_bytes();
    let around = |start: usize, end: usize| {
        let mut from = start.saturating_sub(30);
        while !code.is_char_boundary(from) {
            from -= 1;
        }
        let mut to = (end + 5).min(code.len());
        while !code.is_char_boundary(to) {
            to += 1;
        }
        code[from..to].to_string()
    };
    let mut found = Vec::new();
    for (start, end) in paths(code) {
        if !is_table(&code[start..end], own) {
            continue;
        }
        let before = if start > 0 { b[start - 1] } else { 0 };
        let after = b.get(end).copied().unwrap_or(0);
        if after == b'{' {
            // A value's construction, not a place that keeps one.
            continue;
        }
        let typed = before == b':' && (start < 2 || b[start - 2] != b':');
        if typed
            || before == b'<'
            || before == b'('
            || (before == b',' && (after == b')' || after == b'>'))
        {
            found.push(around(start, end));
        }
    }
    let pointer = "extern\"system\"fn(";
    let mut from = 0;
    while let Some(at) = code[from..].find(pointer) {
        found.push(around(from + at, from + at + pointer.len()));
        from += at + pointer.len();
    }
    found
}

/// The structs, enums and type aliases whose definitions keep a function table, by name, and
/// whether each is visible outside its module (`pub`, `pub(..)`).
fn wrappers(sources: &[(String, String)], own: &[String]) -> Vec<(String, bool)> {
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut found = Vec::new();
    for (_, code) in sources {
        let b = code.as_bytes();
        for keyword in ["struct", "enum", "type"] {
            let mut from = 0;
            while let Some(at) = code[from..].find(keyword) {
                let k = from + at;
                from = k + keyword.len();
                let visible = (k >= 3 && b[k - 3..k] == *b"pub")
                    || (k >= 1
                        && b[k - 1] == b')'
                        && code[..k].rfind("pub(").is_some_and(|p| {
                            code[p + 4..k - 1]
                                .bytes()
                                .all(|c| c.is_ascii_lowercase() || c == b':')
                        }));
                if !visible && k > 0 && ident(b[k - 1]) {
                    continue;
                }
                let name_start = from;
                if !b.get(name_start).is_some_and(u8::is_ascii_uppercase) {
                    continue;
                }
                let mut i = name_start;
                while i < b.len() && ident(b[i]) {
                    i += 1;
                }
                let name = code[name_start..i].to_string();
                // Skip generics, then take the body: a type alias's right side, or a struct's
                // or enum's braces or parentheses.
                let mut depth = 0i32;
                while i < b.len() {
                    match b[i] {
                        b'<' => depth += 1,
                        b'>' => depth -= 1,
                        _ if depth == 0 => break,
                        _ => {}
                    }
                    i += 1;
                }
                let body = if keyword == "type" {
                    if b.get(i) != Some(&b'=') {
                        continue;
                    }
                    let end = code[i..].find(';').map_or(code.len(), |e| i + e);
                    &code[i + 1..end]
                } else {
                    if !matches!(b.get(i), Some(b'{' | b'(')) {
                        continue;
                    }
                    let open = i;
                    let mut depth = 0i32;
                    while i < b.len() {
                        match b[i] {
                            b'{' | b'(' => depth += 1,
                            b'}' | b')' => {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            _ => {}
                        }
                        i += 1;
                    }
                    &code[open..(i + 1).min(code.len())]
                };
                let keeps = if keyword == "type" {
                    paths(body).iter().any(|&(s, e)| is_table(&body[s..e], own))
                        || body.contains("extern\"system\"fn(")
                } else {
                    !stored(body, own).is_empty()
                };
                if keeps {
                    found.push((name, visible));
                }
            }
        }
    }
    found
}

/// Only the listed modules keep a copy of the device's functions (`KEEPERS`). A copy is any by
/// value: a Vulkan function pointer (`PFN_*`, an `extern "system" fn` type), ash's device
/// functions or a device extension's (`ash::Device`, `ash::*::Device`, `*::DeviceFn`), the
/// runtime's `Functions` and generated-commands `Commands`, the visible types of the runtime's
/// that keep one of those, and the visible types that keep such a type. Elsewhere a copy could be
/// called after the device's teardown wherever its holder outlived the device: the SIGSEGV, a
/// shader module's own `ash::Device`. An object that releases Vulkan objects holds the device
/// (`Device`, reaching its functions through `DeviceShared::vk`), or is the device's and destroyed
/// by it (`pipeline/content_keyed.rs`).
#[test]
fn only_the_listed_modules_keep_the_device_s_functions() {
    let sources = compact_sources();
    let mut own: Vec<String> = vec!["Functions".into(), "Commands".into()];
    // Two levels: the types that keep a table, and the visible types that keep one of those.
    // Deeper holders (an encoder, a command buffer) hold the device themselves.
    for _ in 0..2 {
        let more: Vec<String> = wrappers(&sources, &own)
            .into_iter()
            .filter(|(name, visible)| {
                *visible && !own.contains(name) && !THE_DEVICE.contains(&name.as_str())
            })
            .map(|(name, _)| name)
            .collect();
        if more.is_empty() {
            break;
        }
        for name in more {
            if !own.contains(&name) {
                own.push(name);
            }
        }
    }
    let mut outside = Vec::new();
    let mut keeping = std::collections::BTreeSet::new();
    for (path, code) in &sources {
        let found = stored(code, &own);
        if found.is_empty() {
            continue;
        }
        keeping.insert(path.clone());
        if !KEEPERS.iter().any(|(file, _)| file == path) {
            outside.extend(found.into_iter().map(|at| format!("{path}: {at}")));
        }
    }
    assert!(
        outside.is_empty(),
        "copies of the device's functions outside the listed modules (types {own:?}): {outside:#?}"
    );
    let stale: Vec<&str> = KEEPERS
        .iter()
        .map(|(file, _)| *file)
        .filter(|file| !keeping.contains(*file))
        .collect();
    assert!(
        stale.is_empty(),
        "listed modules that keep no copy any more: {stale:?}"
    );
    // The scan sees the copies it must: the device's own functions and an extension's.
    assert!(
        own.iter().any(|name| name == "ExtendedDynamicState"),
        "{own:?}"
    );
    assert!(keeping.contains("host_memory.rs"));
    // Each kind of copy is seen wherever it is written, a holder of the device is not, and a
    // `use` names a type without keeping it.
    let scan = |code: &str| {
        stored(
            &without_uses(&code.split_whitespace().collect::<String>()),
            &own,
        )
    };
    for copy in [
        "struct A { destroy: vk::PFN_vkDestroyShaderModule }",
        "struct A { loader: ash::khr::swapchain::DeviceFn }",
        "struct A(ash::Device);",
        "struct A { table: Arc<Functions> }",
        "struct A { dynamic: Option<ExtendedDynamicState> }",
        "fn a(raw: extern \"system\" fn()) {}",
    ] {
        assert!(!scan(copy).is_empty(), "not seen: {copy}");
    }
    for no_copy in [
        "struct A { device: Device, raw: vk::ShaderModule }",
        "use ash::vk::PFN_vkDestroyShaderModule; struct A { shared: Arc<DeviceShared> }",
    ] {
        assert!(scan(no_copy).is_empty(), "seen: {no_copy}");
    }
}

/// The exit guard's table names each entry point once, by its Vulkan name.
#[test]
fn the_table_names_each_entry_point_once() {
    let names: Vec<&std::ffi::CStr> = Call::ALL.iter().map(|call| call.name()).collect();
    for (at, name) in names.iter().enumerate() {
        assert!(name.to_bytes().starts_with(b"vk"), "{name:?}");
        assert!(!names[..at].contains(name), "{name:?} twice");
    }
}

/// Priming's check misses nothing but the entry points it cannot make (and, on a queue family
/// that writes no timestamps, the timestamp commands, on a device without device addresses, a
/// buffer's address, on one without generated commands, the release of their objects, on one
/// without the push command, a host step's push, on one that calibrates no timestamps, a
/// completion turn's calibration reading, on one whose instance reports no messages, a pipeline's
/// debug name, and on one without a render encoder's command of an extension or of Vulkan 1.3,
/// that command), and names an entry point of the table it did not make.
#[test]
fn the_priming_check_names_what_was_not_made() {
    use crate::device::prime::{
        missing, Has, CALIBRATION_CALLS, COLOR_WRITE_MASK_CALLS, DEVICE_ADDRESS_CALLS,
        EXTENDED_DYNAMIC_STATE_CALLS, GENERATED_COMMANDS_CALLS, HOST_QUERY_RESET_CALLS,
        NAMING_CALLS, PRIMITIVE_RESTART_CALLS, PUSH_DESCRIPTOR_CALLS, SIZED_INDEX_BIND_CALLS,
        SPARSE_CALLS, TIMESTAMP_CALLS, UNPRIMED, YCBCR_CONVERSION_CALLS,
    };
    let has = |timestamps, device_addresses| Has {
        timestamps,
        host_query_reset: true,
        device_addresses,
        generated_commands: true,
        sparse_binding: true,
        push_descriptor: true,
        calibration: true,
        object_names: true,
        extended_dynamic_state: true,
        primitive_restart: true,
        color_write_masks: true,
        sized_index_bind: true,
        ycbcr_conversion: true,
    };
    let all = Call::ALL.iter().fold(0, |bits, call| bits | call.bit());
    assert!(missing(all, has(true, true)).is_empty());
    let made = Call::ALL
        .iter()
        .filter(|call| !UNPRIMED.contains(call))
        .fold(0, |bits, call| bits | call.bit());
    assert!(missing(made, has(true, true)).is_empty());
    assert_eq!(
        missing(0, has(true, true)).len(),
        Call::ALL.len() - UNPRIMED.len()
    );
    for &call in Call::ALL {
        if !UNPRIMED.contains(&call) {
            assert_eq!(missing(made & !call.bit(), has(true, true)), [call]);
        }
    }
    // Without timestamps, their commands are neither made nor missed, nor is their reset on the
    // host.
    let untimed = (TIMESTAMP_CALLS.iter().chain(&HOST_QUERY_RESET_CALLS))
        .fold(made, |bits, call| bits & !call.bit());
    assert!(missing(untimed, has(false, true)).is_empty());
    let mut timed = TIMESTAMP_CALLS.to_vec();
    timed.extend(HOST_QUERY_RESET_CALLS);
    timed.sort_by_key(|call| Call::ALL.iter().position(|each| each == call));
    assert_eq!(missing(untimed, has(true, true)), timed);
    // Without `hostQueryReset`, nor is the reset on the host.
    let unreset = HOST_QUERY_RESET_CALLS
        .iter()
        .fold(made, |bits, call| bits & !call.bit());
    let without = Has {
        host_query_reset: false,
        ..has(true, true)
    };
    assert!(missing(unreset, without).is_empty());
    assert_eq!(missing(unreset, has(true, true)), HOST_QUERY_RESET_CALLS);
    // Without device addresses, nor is a buffer's address.
    let unaddressed = DEVICE_ADDRESS_CALLS
        .iter()
        .fold(made, |bits, call| bits & !call.bit());
    assert!(missing(unaddressed, has(true, false)).is_empty());
    assert_eq!(missing(unaddressed, has(true, true)), DEVICE_ADDRESS_CALLS);
    // Without generated commands, nor is the release of their objects.
    let ungenerated = GENERATED_COMMANDS_CALLS
        .iter()
        .fold(made, |bits, call| bits & !call.bit());
    let without = Has {
        generated_commands: false,
        ..has(true, true)
    };
    assert!(missing(ungenerated, without).is_empty());
    assert_eq!(
        missing(ungenerated, has(true, true)),
        GENERATED_COMMANDS_CALLS
    );
    // Without sparse binding, nor is a batch's bind.
    let unbound = SPARSE_CALLS
        .iter()
        .fold(made, |bits, call| bits & !call.bit());
    let without = Has {
        sparse_binding: false,
        ..has(true, true)
    };
    assert!(missing(unbound, without).is_empty());
    assert_eq!(missing(unbound, has(true, true)), SPARSE_CALLS);
    // Without the push command, nor is a host step's push.
    let unpushed = PUSH_DESCRIPTOR_CALLS
        .iter()
        .fold(made, |bits, call| bits & !call.bit());
    let without = Has {
        push_descriptor: false,
        ..has(true, true)
    };
    assert!(missing(unpushed, without).is_empty());
    assert_eq!(missing(unpushed, has(true, true)), PUSH_DESCRIPTOR_CALLS);
    // Without calibrated timestamps, nor is a completion turn's calibration reading.
    let uncalibrated = CALIBRATION_CALLS
        .iter()
        .fold(made, |bits, call| bits & !call.bit());
    let without = Has {
        calibration: false,
        ..has(true, true)
    };
    assert!(missing(uncalibrated, without).is_empty());
    assert_eq!(missing(uncalibrated, has(true, true)), CALIBRATION_CALLS);
    // Without messages reported, nor is a pipeline's debug name.
    let unnamed = NAMING_CALLS
        .iter()
        .fold(made, |bits, call| bits & !call.bit());
    let without = Has {
        object_names: false,
        ..has(true, true)
    };
    assert!(missing(unnamed, without).is_empty());
    assert_eq!(missing(unnamed, has(true, true)), NAMING_CALLS);
    // Without a render encoder's command of an extension or of Vulkan 1.3, nor is that command.
    let groups: [(&[Call], fn(Has) -> Has); 5] = [
        (&EXTENDED_DYNAMIC_STATE_CALLS, |has| Has {
            extended_dynamic_state: false,
            ..has
        }),
        (&PRIMITIVE_RESTART_CALLS, |has| Has {
            primitive_restart: false,
            ..has
        }),
        (&COLOR_WRITE_MASK_CALLS, |has| Has {
            color_write_masks: false,
            ..has
        }),
        (&SIZED_INDEX_BIND_CALLS, |has| Has {
            sized_index_bind: false,
            ..has
        }),
        // Without a YCbCr format the device converts, nor is a conversion's release.
        (&YCBCR_CONVERSION_CALLS, |has| Has {
            ycbcr_conversion: false,
            ..has
        }),
    ];
    for (calls, without) in groups {
        let unmade = calls.iter().fold(made, |bits, call| bits & !call.bit());
        assert!(missing(unmade, without(has(true, true))).is_empty());
        assert_eq!(missing(unmade, has(true, true)), calls);
    }
}

/// A thread is in a section, so that the device's functions are the table's alone, while its
/// innermost frame is one of the runtime's sections: not while it runs the program's code a
/// section called (a message sink), and again inside a section that code starts.
#[test]
fn the_table_applies_inside_the_runtime_s_sections_only() {
    use std::cell::Cell;
    let guard = guard();
    assert!(!exit_guard::in_section());
    let outer = guard.enter();
    assert!(exit_guard::in_section());
    let seen = Cell::new((true, false, true));
    exit_guard::program(|| {
        let program = exit_guard::in_section();
        let inner = guard.enter();
        let nested = exit_guard::in_section();
        drop(inner);
        seen.set((program, nested, exit_guard::in_section()));
    });
    assert_eq!(seen.get(), (false, true, false));
    assert!(exit_guard::in_section());
    drop(outer);
    assert!(!exit_guard::in_section());
}

/// A device registers the hook when it can have made statics since the hook's last registration:
/// the process's first (a forked child's included), one of an identity new since the last load
/// of a shared object, and the first after such a load. Made again on what stays loaded, devices
/// register nothing, however many, and the identities kept stay as many as the distinct ones.
#[test]
fn devices_register_the_hook_only_when_they_can_have_made_statics() {
    use crate::exit_guard::{Admitted, Devices};
    let mut devices = Devices::new();
    assert_eq!(
        devices.admit(false, Some(7), vec![1]),
        Some(Admitted::Load),
        "the process's first"
    );
    assert_eq!(devices.admit(true, Some(7), vec![1]), None);
    assert_eq!(
        devices.admit(true, Some(7), vec![2]),
        Some(Admitted::Identity),
        "a new identity"
    );
    for _ in 0..1000 {
        assert_eq!(devices.admit(true, Some(7), vec![1]), None);
        assert_eq!(devices.admit(true, Some(7), vec![2]), None);
    }
    assert_eq!(devices.kept(), 2);
    assert_eq!(
        devices.admit(true, Some(8), vec![1]),
        Some(Admitted::Load),
        "the first after a load"
    );
    assert_eq!(devices.kept(), 1);
    assert_eq!(devices.admit(true, Some(8), vec![1]), None);
    assert_eq!(
        devices.admit(true, Some(8), vec![2]),
        Some(Admitted::Identity),
        "known before the load, new since"
    );
    assert_eq!(
        devices.admit(false, Some(8), vec![1]),
        Some(Admitted::Load),
        "a forked child's first"
    );
    // Where the host keeps no count, every device registers as at a load, and one identity is
    // kept.
    for _ in 0..10 {
        assert_eq!(devices.admit(true, None, vec![1]), Some(Admitted::Load));
    }
    assert_eq!(devices.kept(), 1);
}

/// A section calling the table's first entry point.
#[cfg(unix)]
fn call_the_first_entry_point() {
    let busy = exit_guard::enter();
    exit_guard::called(Call::ALL[0]);
    drop(busy);
}

/// The call before the layer's load, then empty sections until the next section's count is no
/// power of two. Those sections run before the destructor's registration: one ending at a power
/// of two registers the hook again, and after it that would pass the tests below for that
/// trigger's reason, not the first call's.
#[cfg(unix)]
fn call_the_first_entry_point_then_skip_powers_of_two() {
    call_the_first_entry_point();
    while (exit_guard::sections() + 1).is_power_of_two() {
        drop(exit_guard::enter());
    }
}

/// The gap this closes: a layer loaded again registers a destructor at an entry point's first call
/// since (after the hook's registration at the device), and a section calling that entry point,
/// called before the load, registers nothing, so the destructor runs before the hook.
#[cfg(unix)]
#[test]
fn a_call_made_before_a_load_registers_nothing_again_unforgotten() {
    assert_eq!(
        late_destructor_exit_after(
            call_the_first_entry_point_then_skip_powers_of_two,
            call_the_first_entry_point
        ),
        43
    );
}

/// Once the device made after the load forgot the entry points called
/// (`exit_guard::forget_calls`), the section calling one again registers the hook again as it
/// ends, so the layer's destructor registered before runs after the hook.
#[cfg(unix)]
#[test]
fn a_section_after_an_entry_point_s_first_call_since_a_load_registers_the_hook_again() {
    assert_eq!(
        late_destructor_exit_after(call_the_first_entry_point_then_skip_powers_of_two, || {
            exit_guard::forget_calls();
            call_the_first_entry_point();
        }),
        42,
        "the destructor ran before the hook"
    );
}

/// The process's count of loaded shared objects never falls, and rises when one is loaded (where
/// a library not loaded yet can be found here).
#[cfg(any(target_os = "linux", target_vendor = "apple"))]
#[test]
fn the_count_of_loaded_objects_rises_with_a_load() {
    use std::ffi::CString;
    let before = crate::loaded::count().expect("the host keeps a count");
    let candidates: &[&str] = if cfg!(target_vendor = "apple") {
        &[
            "/usr/lib/libz.1.dylib",
            "/usr/lib/libbz2.1.0.dylib",
            "/usr/lib/libxml2.2.dylib",
        ]
    } else {
        &[
            "libz.so.1",
            "libbz2.so.1.0",
            "libexpat.so.1",
            "libxml2.so.2",
        ]
    };
    let fresh = candidates.iter().find_map(|name| {
        let name = CString::new(*name).unwrap();
        // SAFETY: a library name; `RTLD_NOLOAD` loads nothing, so only one not loaded yet is
        // loaded below.
        let loaded = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_NOLOAD) };
        if !loaded.is_null() {
            // SAFETY: the handle `dlopen` just answered.
            unsafe { libc::dlclose(loaded) };
            return None;
        }
        // SAFETY: as above; a system library, kept loaded.
        let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        (!handle.is_null()).then_some(name)
    });
    let after = crate::loaded::count().unwrap();
    match fresh {
        Some(name) => assert!(after > before, "{name:?} loaded: {before} then {after}"),
        None => {
            eprintln!("no library left to load among {candidates:?}");
            assert!(after >= before);
        }
    }
}

/// The public query (`process_exiting`) reads the process's guard: open while the process runs. The
/// exits lane reads it closed in an exit handler that runs after the hook (`runtime_lane::exits`).
#[test]
fn the_process_is_not_exiting_while_its_tests_run() {
    assert!(!crate::process_exiting());
    assert_eq!(crate::process_exiting(), exit_guard::closed());
}
