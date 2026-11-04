//! `scheduler/inbox.rs`: a thread about to block passes once where the inbox holds commits and no
//! passer is counted, and never where one is, so two threads never both pass; what is left to such
//! a pass it reads back for the scheduler's thread instead of passing again; and a commit posts
//! and returns at once, whether or not a pass is under way,
//! and its commit goes next, in commit order; one that finds no pass under way returns at once
//! while the state's lock is held, for its caller to wake the thread under the lock of its sleep,
//! which no pass takes; the scheduler's pass looks again for a commit
//! left to it until a pass answers it has work of its own first; commits from many threads racing
//! passes, ones that stop looking again, and short holds are all passed, each thread's in its
//! order, with no pass made for them at the end; a nudge finding a pass under way returns at once
//! and is passed for once that pass ends; and wakers racing passes and commits have every change
//! they made read by a pass; and a pass leaves the inbox room for twice the commits it took in.

use crate::device::lock;
use crate::scheduler::inbox::{Inbox, ROOM};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// The longest a step of these tests may take: a commit that waits for the pass under way
/// instead of returning fails the test after it, rather than hanging it.
const WAIT: Duration = Duration::from_secs(30);

/// The state: the commits in the order the passes took them in.
type Log = Inbox<Vec<u32>, u32>;

/// A pass that only takes the commits in, and looks again.
fn take_in(log: &mut Vec<u32>, commits: &mut Vec<u32>) -> bool {
    log.append(commits);
    true
}

/// Post `commit` from another thread while this one makes a pass: it is left to this thread's
/// pass, which looks again for it, so the committer wakes nothing.
pub(super) fn commit_elsewhere<S: Send, C: Send>(inbox: &Inbox<S, C>, commit: C) {
    thread::scope(|scope| {
        let committer = scope.spawn(|| inbox.post(commit));
        assert!(
            !committer.join().unwrap(),
            "a commit that found a pass under way woke the thread"
        );
    });
}

/// A pass by the scheduler's thread that stays under way, the lock held, until the test sends on
/// the channel it answers, taking in what it finds and looking again; its thread answers how many
/// passes it made.
fn held_pass(inbox: &Arc<Log>) -> (thread::JoinHandle<u32>, mpsc::Sender<()>) {
    let (entered, in_pass) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let passer = {
        let inbox = Arc::clone(inbox);
        thread::spawn(move || {
            let mut passes = 0;
            inbox.pass(|log, commits| {
                take_in(log, commits);
                passes += 1;
                if passes == 1 {
                    entered.send(()).unwrap();
                    released.recv_timeout(WAIT).unwrap();
                }
                true
            });
            passes
        })
    };
    in_pass.recv_timeout(WAIT).unwrap();
    (passer, release)
}

#[test]
fn a_commit_finding_a_pass_under_way_returns_at_once_and_goes_next_in_commit_order() {
    let inbox = Arc::new(Log::new(Vec::new()));
    assert!(
        inbox.post(0),
        "no pass under way: its caller wakes the thread"
    );
    let (passer, release) = held_pass(&inbox);
    let (done, returned) = mpsc::channel();
    let committer = {
        let inbox = Arc::clone(&inbox);
        thread::spawn(move || {
            for commit in 1..=3 {
                done.send(inbox.post(commit)).unwrap();
            }
        })
    };
    for commit in 1..=3 {
        let wakes = returned.recv_timeout(WAIT).unwrap_or_else(|_| {
            panic!("commit {commit} waited for the pass under way instead of returning")
        });
        assert!(!wakes, "commit {commit} was left to the pass under way");
    }
    committer.join().unwrap();
    assert!(inbox.holds_commits());
    release.send(()).unwrap();
    assert_eq!(passer.join().unwrap(), 2, "it looked again for them");
    // The passer took them in once its pass ended, in commit order.
    assert_eq!(*lock(inbox.state()), [0, 1, 2, 3]);
    assert!(!inbox.holds_commits());
}

/// A commit that finds no pass under way makes none either, and takes no lock of the state's: it
/// returns at once while that lock is held, as a pass counted after its look may hold it across a
/// `vkQueueSubmit` that waits for a held `VkQueue`, for its caller to wake the thread; the
/// thread's next pass takes it in.
#[test]
fn a_commit_finding_no_pass_returns_while_the_lock_is_held() {
    let inbox = Arc::new(Log::new(Vec::new()));
    let held = lock(inbox.state());
    let (done, returned) = mpsc::channel();
    let committer = {
        let inbox = Arc::clone(&inbox);
        thread::spawn(move || done.send(inbox.post(7)).unwrap())
    };
    let wakes = returned.recv_timeout(WAIT);
    drop(held);
    committer.join().unwrap();
    assert_eq!(
        wakes,
        Ok(true),
        "the commit waited for the lock, or found a pass under way"
    );
    assert!(lock(inbox.state()).is_empty(), "and made no pass");
    assert!(inbox.holds_commits());
    inbox.pass(|log, commits| {
        log.append(commits);
        false
    });
    assert_eq!(*lock(inbox.state()), [7]);
    assert!(!inbox.holds_commits());
}

/// A new inbox holds [`ROOM`] commits without growing: so a frame's commits, which wait at most
/// that many, push theirs with no allocation on their own thread, in `commit`, however far the
/// thread making passes falls behind them before its first pass took in as many.
#[test]
fn a_new_inbox_holds_room_commits_without_growing() {
    let inbox = Log::new(Vec::new());
    let room = inbox.room();
    assert!(room >= ROOM, "a new inbox has room for {room}");
    for commit in 0..ROOM as u32 {
        inbox.post(commit);
    }
    assert_eq!(inbox.room(), room, "{ROOM} commits grew the inbox");
}

/// A pass leaves the inbox room for twice the commits it took in, made on the passing thread: so
/// the commits after it grow the inbox on their own thread, an allocation in `commit`, only where
/// more than twice as many wait. Grown by the pushes alone, it had room for 8 after 5 (Rust
/// 1.98.1's growth).
#[test]
fn a_pass_leaves_the_inbox_room_for_twice_the_commits_it_took_in() {
    let inbox = Log::new(Vec::new());
    for commit in 0..5 {
        inbox.post(commit);
    }
    inbox.pass(|log, commits| {
        log.append(commits);
        false
    });
    assert_eq!(*lock(inbox.state()), [0, 1, 2, 3, 4]);
    assert!(
        inbox.room() >= 10,
        "a pass that took in 5 left the inbox room for {}",
        inbox.room()
    );
}

#[test]
fn the_scheduler_s_pass_looks_again_until_a_pass_answers_it_has_work_first() {
    for again in [true, false] {
        let inbox = Arc::new(Log::new(Vec::new()));
        let (entered, in_pass) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let scheduler = {
            let inbox = Arc::clone(&inbox);
            thread::spawn(move || {
                let mut passes = 0;
                inbox.pass(|log, commits| {
                    take_in(log, commits);
                    passes += 1;
                    // The first pass stays under way until the test lets it end.
                    if passes == 1 {
                        entered.send(()).unwrap();
                        released.recv_timeout(WAIT).unwrap();
                    }
                    again
                });
                passes
            })
        };
        in_pass.recv_timeout(WAIT).unwrap();
        commit_elsewhere(&inbox, 1);
        release.send(()).unwrap();
        let passes = scheduler.join().unwrap();
        if again {
            // It looked again and took the commit in.
            assert_eq!(passes, 2);
            assert_eq!(*lock(inbox.state()), [1]);
            assert!(!inbox.holds_commits());
        } else {
            // It stopped: the commit waits for its own look.
            assert_eq!(passes, 1);
            assert!(lock(inbox.state()).is_empty());
            assert!(inbox.holds_commits());
            inbox.pass(take_in);
            assert_eq!(*lock(inbox.state()), [1]);
            assert!(!inbox.holds_commits());
        }
    }
}

/// The state of [`Model`]'s passes: the commits in the order they took them in, each waker's
/// value as the last pass read it, and the stop.
#[derive(Default)]
struct Seen {
    log: Vec<u32>,
    values: Vec<u64>,
    stop: bool,
}

/// The scheduler's thread and its wakes over an inbox, as `scheduler/mod.rs` makes them: it sleeps
/// under a lock of its own, which no pass takes, and a commit or a nudge that finds no pass under
/// way wakes it through that lock alone.
struct Model {
    inbox: Inbox<Seen, u32>,
    /// Whether the thread sleeps, under the lock of its sleep.
    asleep: Mutex<bool>,
    woken: Condvar,
}

impl Model {
    fn new(wakers: usize) -> Model {
        Model {
            inbox: Inbox::new(Seen {
                values: vec![0; wakers],
                ..Seen::default()
            }),
            asleep: Mutex::new(false),
            woken: Condvar::new(),
        }
    }

    /// `Scheduler::commit`'s: post, and wake the thread where no pass was under way. Whether it
    /// was left to a pass under way.
    fn commit(&self, commit: u32) -> bool {
        if self.inbox.post(commit) {
            self.wake_thread();
            return false;
        }
        true
    }

    /// `Scheduler::wake`'s: whether it was left to a pass under way.
    fn wake(&self) -> bool {
        if !self.inbox.nudge() {
            self.wake_thread();
            return false;
        }
        true
    }

    /// `SchedulerShared::wake_thread`'s: under the lock of the sleep alone, notify the thread
    /// where it sleeps, marking it awake.
    fn wake_thread(&self) {
        let mut asleep = lock(&self.asleep);
        if *asleep {
            *asleep = false;
            self.woken.notify_one();
        }
    }

    /// The thread's loop: it waits for a commit or a nudge left to it, with no timed look, then
    /// passes, its passes answering `again` of their count whether to look again (false: it has
    /// work of its own first, done at once, after which it looks at the inbox again). It reads the
    /// inbox again under the lock of its sleep before it sleeps, and lets go of the state's lock
    /// then. Once stopped, it ends when it has nothing to pass for.
    fn run(&self, mut pass: impl FnMut(&mut Seen, &mut Vec<u32>), again: impl Fn(u64) -> bool) {
        let asked = || self.inbox.holds_commits() || self.inbox.nudged();
        let mut passes = 0;
        loop {
            let mut seen = lock(self.inbox.state());
            while !asked() && !seen.stop {
                let mut asleep = lock(&self.asleep);
                if asked() {
                    continue;
                }
                *asleep = true;
                drop(seen);
                while *asleep {
                    asleep = self.woken.wait(asleep).unwrap();
                }
                drop(asleep);
                seen = lock(self.inbox.state());
            }
            if !asked() {
                return;
            }
            drop(seen);
            self.inbox.pass(|seen, commits| {
                pass(seen, commits);
                passes += 1;
                again(passes)
            });
        }
    }

    fn stop(&self) {
        lock(self.inbox.state()).stop = true;
        self.wake_thread();
    }
}

/// A pass that holds the lock a while, so that commits and wakes find it under way.
fn slow(seen: &mut Seen, commits: &mut Vec<u32>) {
    for _ in 0..200 {
        std::hint::spin_loop();
    }
    seen.log.append(commits);
}

/// Whether `done` holds of the model's state within [`WAIT`], read every millisecond.
fn soon(model: &Model, done: impl Fn(&Seen) -> bool) -> bool {
    let until = Instant::now() + WAIT;
    while !done(&lock(model.inbox.state())) {
        if Instant::now() > until {
            return false;
        }
        thread::sleep(Duration::from_millis(1));
    }
    true
}

/// Commits from several threads race the scheduler thread's passes, which answer in turn that
/// they have work first, and short holds: each is passed once, each thread's in its order, by a
/// pass under way that looked again for it or by the thread its committer woke, with no pass
/// made for them at the end.
#[test]
fn commits_racing_passes_and_short_holds_are_all_passed_each_thread_s_in_its_order() {
    const THREADS: u32 = 4;
    const COMMITS: u32 = 2000;
    let model = Model::new(0);
    let all = (THREADS * COMMITS) as usize;
    let (left, passed_all) = thread::scope(|scope| {
        // The scheduler's thread: every other pass answers it has work of its own first.
        let scheduler = scope.spawn(|| model.run(slow, |pass| pass % 2 == 0));
        let committers: Vec<_> = (0..THREADS)
            .map(|index| {
                let model = &model;
                scope.spawn(move || {
                    (0..COMMITS)
                        .filter(|commit| model.commit(index * COMMITS + commit))
                        .count()
                })
            })
            .collect();
        // A decode count: short holds that make no pass.
        let marker = scope.spawn(|| {
            for _ in 0..COMMITS {
                drop(lock(model.inbox.state()));
                thread::yield_now();
            }
        });
        let left: usize = (committers.into_iter())
            .map(|committer| committer.join().unwrap())
            .sum();
        marker.join().unwrap();
        // The thread passes for nothing on its own: a commit waits for a look or its wake.
        let passed_all = soon(&model, |seen| seen.log.len() == all);
        model.stop();
        scheduler.join().unwrap();
        (left, passed_all)
    });
    let seen = lock(model.inbox.state());
    assert!(
        passed_all,
        "{} of {all} passed ({left} left to a pass under way)",
        seen.log.len()
    );
    eprintln!("INBOX-RACE commits: {left} of {all} left to a pass under way, the rest woke it");
    assert!(!model.inbox.holds_commits());
    assert_eq!(seen.log.len(), all, "each once");
    for index in 0..THREADS {
        let own: Vec<u32> = (seen.log.iter().copied())
            .filter(|commit| commit / COMMITS == index)
            .collect();
        let ordered: Vec<u32> = (index * COMMITS..(index + 1) * COMMITS).collect();
        assert_eq!(own, ordered, "thread {index}'s commits in its order");
    }
}

/// A nudge (an event's value rose, a batch completed) that finds a pass under way returns at once,
/// left to it, and the passer passes again once done, with no commit; one that finds no passer
/// answers so, for its caller to wake the thread, and the next pass takes it in.
#[test]
fn a_nudge_finding_a_pass_under_way_is_passed_for_once_it_ends() {
    let inbox = Arc::new(Log::new(Vec::new()));
    assert!(inbox.post(0));
    let (passer, release) = held_pass(&inbox);
    assert!(inbox.nudge(), "the nudge was left to the pass under way");
    release.send(()).unwrap();
    assert_eq!(
        passer.join().unwrap(),
        2,
        "the passer passed again for the nudge"
    );
    assert!(!inbox.nudged());
    assert_eq!(*lock(inbox.state()), [0]);

    assert!(!inbox.nudge(), "no passer: the caller wakes the thread");
    assert!(inbox.nudged());
    inbox.pass(|log, commits| {
        take_in(log, commits);
        false
    });
    assert!(!inbox.nudged(), "the next pass took it in");
}

/// Wakers each raise a value of their own and wake for a pass (an event's value rising, a batch
/// completing), racing the scheduler thread's passes, committing threads' passes and their wakes:
/// once they are done, a pass read every waker's last value, made for its wake, with no pass made
/// for it at the end. A nudge taken in by a pass is read with what its waker changed before it.
#[test]
fn wakers_racing_passes_and_commits_have_every_change_read_by_a_pass() {
    const WAKERS: usize = 3;
    const RAISES: u64 = 3000;
    const THREADS: u32 = 2;
    const COMMITS: u32 = 2000;
    let model = Model::new(WAKERS);
    let values: Vec<AtomicU64> = (0..WAKERS).map(|_| AtomicU64::new(0)).collect();
    let read = |seen: &mut Seen, commits: &mut Vec<u32>| {
        for _ in 0..100 {
            std::hint::spin_loop();
        }
        seen.log.append(commits);
        // Raised with no ordering of their own: only the wake orders a raise before a pass.
        for (seen, value) in seen.values.iter_mut().zip(&values) {
            *seen = value.load(Ordering::Relaxed);
        }
    };
    let (read_all, left) = thread::scope(|scope| {
        // Every third pass answers it has work of its own first.
        let scheduler = scope.spawn(|| model.run(read, |pass| pass % 3 != 0));
        let committers: Vec<_> = (0..THREADS)
            .map(|index| {
                let model = &model;
                scope.spawn(move || {
                    for commit in 0..COMMITS {
                        model.commit(index * COMMITS + commit);
                    }
                })
            })
            .collect();
        let wakers: Vec<_> = (0..WAKERS)
            .map(|waker| {
                let (model, value) = (&model, &values[waker]);
                scope.spawn(move || {
                    let mut left = 0;
                    for raise in 1..=RAISES {
                        value.store(raise, Ordering::Relaxed);
                        left += u64::from(model.wake());
                        if raise % 8 == 0 {
                            thread::yield_now();
                        }
                    }
                    left
                })
            })
            .collect();
        for thread in committers {
            thread.join().unwrap();
        }
        let left: u64 = (wakers.into_iter())
            .map(|thread| thread.join().unwrap())
            .sum();
        // The last raises' wakes were passed for; the thread passes on its own for nothing.
        let read_all = soon(&model, |seen| seen.values == [RAISES; WAKERS]);
        model.stop();
        scheduler.join().unwrap();
        (read_all, left)
    });
    let seen = lock(model.inbox.state());
    let wakes = RAISES * WAKERS as u64;
    eprintln!("INBOX-RACE wakes: {left} of {wakes} left to a pass under way");
    assert!(
        read_all,
        "a wake was lost: the passes read {:?} of {RAISES}",
        seen.values
    );
    assert_eq!(seen.log.len(), (THREADS * COMMITS) as usize, "each once");
    assert!(!model.inbox.holds_commits() && !model.inbox.nudged());
}

/// A thread about to block passes where the inbox holds a commit and no passer is counted: it
/// takes the commits in itself, in commit order, and makes one pass alone.
#[test]
fn a_thread_about_to_block_passes_once_for_the_commits_the_inbox_holds() {
    let inbox = Log::new(Vec::new());
    assert!(
        !inbox.pass_once(|_, _| panic!("a pass with no commit in the inbox")),
        "it passed for an empty inbox"
    );
    for commit in 0..3 {
        inbox.post(commit);
    }
    let mut passes = 0;
    assert!(inbox.pass_once(|log, commits| {
        take_in(log, commits);
        passes += 1;
    }));
    assert_eq!(passes, 1, "it made {passes} passes");
    assert_eq!(*lock(inbox.state()), [0, 1, 2]);
    assert!(!inbox.holds_commits());
    assert!(
        !inbox.passed_over(),
        "it read work left over after a pass that took everything in"
    );
}

/// A thread about to block never passes while a passer is counted: the scheduler's pass takes the
/// commits in, and the waiting thread sleeps as it does today.
#[test]
fn a_thread_about_to_block_never_passes_while_a_passer_is_counted() {
    let inbox = Arc::new(Log::new(Vec::new()));
    let (passer, release) = held_pass(&inbox);
    commit_elsewhere(&inbox, 1);
    assert!(inbox.holds_commits());
    {
        let inbox = Arc::clone(&inbox);
        let waiter =
            thread::spawn(move || inbox.pass_once(|_, _| panic!("two threads passed at once")));
        assert!(
            !waiter.join().unwrap(),
            "a thread about to block passed while the scheduler was passing"
        );
    }
    release.send(()).unwrap();
    assert_eq!(passer.join().unwrap(), 2);
    assert_eq!(*lock(inbox.state()), [1]);
}

/// A commit left to a pass a thread about to block made is read back by it once it counted out
/// ([`Inbox::passed_over`]), for the scheduler's thread: it is going to sleep, so it never passes
/// again, and the commit would otherwise wait for nothing.
#[test]
fn a_commit_left_to_a_blocking_thread_s_pass_is_read_back_for_the_scheduler() {
    let inbox = Arc::new(Log::new(Vec::new()));
    inbox.post(0);
    assert!(inbox.pass_once(|log, commits| {
        take_in(log, commits);
        // Posted while this pass is counted: its committer wakes nothing and leaves it here.
        commit_elsewhere(&inbox, 1);
    }));
    assert_eq!(*lock(inbox.state()), [0]);
    assert!(
        inbox.passed_over(),
        "the commit left to the pass was read by nobody"
    );
    // The scheduler's thread, woken for it, takes it in.
    inbox.pass(|log, commits| {
        take_in(log, commits);
        false
    });
    assert_eq!(*lock(inbox.state()), [0, 1]);
    assert!(!inbox.passed_over());
}

/// A nudge left to such a pass is read back the same way: a waker that found the pass counted
/// woke nothing, and the pass takes none afterwards.
#[test]
fn a_nudge_left_to_a_blocking_thread_s_pass_is_read_back_for_the_scheduler() {
    let inbox = Arc::new(Log::new(Vec::new()));
    inbox.post(0);
    assert!(inbox.pass_once(|log, commits| {
        take_in(log, commits);
        assert!(inbox.nudge(), "the waker found no passer counted");
    }));
    assert!(inbox.nudged(), "the nudge was taken in by the pass");
    assert!(inbox.passed_over());
}
