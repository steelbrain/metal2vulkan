//! The queues' serial contexts (`contexts/`): the lines as a structure, and the pool of handler
//! threads with the waits it counts, on plain items without a device.

use crate::command_buffer::own::Context;
use crate::contexts::lines::Lines;
use crate::contexts::{block, Contexts, Run, THREADS};
use crate::error::DeadlockCause;
use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The longest any step here may take.
const WAIT: Duration = Duration::from_secs(20);

#[test]
fn a_line_runs_one_item_at_a_time_in_order_and_others_go_on() {
    let mut lines = Lines::default();
    lines.push(1, "a1");
    lines.push(1, "a2");
    lines.push(2, "b1");
    assert_eq!(lines.ready(), 2);
    assert_eq!(lines.take(), Some((1, "a1")));
    // Line 1 runs: its next item waits, line 2 does not.
    assert_eq!(lines.take(), Some((2, "b1")));
    assert_eq!(lines.take(), None);
    lines.push(2, "b2");
    lines.done(1);
    // Line 1 is ready again, behind nothing; line 2 still runs.
    assert_eq!(lines.take(), Some((1, "a2")));
    lines.done(2);
    assert_eq!(lines.take(), Some((2, "b2")));
    lines.done(1);
    lines.done(2);
    assert!(lines.is_empty());
    assert!(!lines.is_ready());
}

/// A line is taken whole for the calling thread where no thread runs it and each item it holds
/// is plain: its items come with it, it is no longer ready, and items pushed meanwhile wait
/// for its end.
#[test]
fn a_line_is_taken_whole_where_it_runs_nowhere_and_holds_only_plain_items() {
    let plain = |item: &&str| !item.ends_with('!');
    let mut lines = Lines::default();
    // An idle line: taken, with nothing before.
    assert_eq!(lines.take_plain(1, plain), Some(VecDeque::new()));
    lines.push(1, "a1");
    assert!(!lines.is_ready());
    assert_eq!(lines.take_plain(1, plain), None);
    lines.done(1);
    // A line holding plain items, on no thread: taken with them, in order.
    lines.push(1, "a2");
    assert_eq!(lines.ready(), 1);
    assert_eq!(
        lines.take_plain(1, plain),
        Some(VecDeque::from(["a1", "a2"]))
    );
    assert!(!lines.is_ready());
    lines.done(1);
    assert!(lines.is_empty());
    // A line holding an item that is not plain stays as it is.
    lines.push(2, "b1!");
    lines.push(2, "b2");
    assert_eq!(lines.take_plain(2, plain), None);
    assert_eq!(lines.take(), Some((2, "b1!")));
    // So does a line a thread runs.
    assert_eq!(lines.take_plain(2, plain), None);
    lines.done(2);
    assert_eq!(lines.take(), Some((2, "b2")));
    lines.done(2);
    assert!(lines.is_empty());
}

/// A line is taken for the calling thread only where it holds no item and no thread runs it:
/// items pushed meanwhile wait for its end.
#[test]
fn an_idle_line_alone_is_taken_for_the_calling_thread() {
    let mut lines = Lines::default();
    assert!(lines.take_idle(1));
    lines.push(1, "a1");
    assert!(!lines.is_ready());
    assert!(!lines.take_idle(1));
    lines.done(1);
    // A line holding an item, plain or not, stays as it is.
    assert_eq!(lines.ready(), 1);
    assert!(!lines.take_idle(1));
    assert_eq!(lines.take(), Some((1, "a1")));
    // So does a line a thread runs.
    assert!(!lines.take_idle(1));
    lines.done(1);
    assert!(lines.is_empty());
}

#[test]
fn ready_lines_are_taken_in_the_order_they_became_ready() {
    let mut lines = Lines::default();
    for queue in [3, 1, 2] {
        lines.push(queue, queue);
    }
    let taken: Vec<u64> = std::iter::from_fn(|| lines.take().map(|(queue, _)| queue)).collect();
    assert_eq!(taken, [3, 1, 2]);
}

/// An item behind a gate keeps its place in its line and holds the line's later items until every
/// closing of the gate is opened, whenever the item was pushed; other lines go on meanwhile, and
/// no caller takes its line whole.
#[test]
fn an_item_behind_a_gate_holds_its_line_until_every_closing_is_opened() {
    let plain = |_: &&str| true;
    let mut lines = Lines::default();
    lines.push(1, "a1");
    lines.close_new(7);
    assert!(lines.close(7), "closed again while closed");
    lines.push_behind(1, "a2", 7);
    lines.push(1, "a3");
    lines.push(2, "b1");
    assert_eq!(lines.take(), Some((1, "a1")));
    lines.done(1);
    // a2 leads line 1 behind its gate: only line 2 is ready.
    assert_eq!(lines.take(), Some((2, "b1")));
    lines.done(2);
    assert_eq!(lines.take(), None);
    assert_eq!(lines.take_plain(1, plain), None);
    lines.open(1, 7);
    assert_eq!(lines.take(), None, "one closing is left");
    lines.open(1, 7);
    assert!(!lines.close(7), "open");
    assert_eq!(lines.take(), Some((1, "a2")));
    lines.done(1);
    assert_eq!(lines.take(), Some((1, "a3")));
    lines.done(1);
    assert!(lines.is_empty());
    // A gate opened before its item is pushed: the item goes as any.
    lines.close_new(8);
    lines.close(8);
    lines.open(3, 8);
    lines.push_behind(3, "c1", 8);
    assert_eq!(lines.take(), None);
    lines.open(3, 8);
    assert_eq!(lines.take(), Some((3, "c1")));
    lines.done(3);
    lines.close_new(9);
    lines.open(3, 9);
    lines.push_behind(3, "c2", 9);
    assert_eq!(lines.take(), Some((3, "c2")));
    lines.done(3);
    assert!(lines.is_empty());
}

/// An item leads its line where it is the first and no thread runs the line, or where it was
/// pushed, a thread runs the line and no item left is it (the thread runs it); one not pushed yet
/// leads no running line.
#[test]
fn an_item_leads_its_line_with_nothing_before_it() {
    let mut lines = Lines::default();
    let leads =
        |lines: &Lines<&str>, name: &str, posted| lines.leads(1, |item| *item == name, posted);
    assert!(!leads(&lines, "a2", true), "no line");
    lines.push(1, "a1");
    assert!(leads(&lines, "a1", true));
    assert_eq!(lines.take(), Some((1, "a1")));
    // a1 runs: it leads; a2, not pushed yet, does not.
    assert!(leads(&lines, "a1", true));
    assert!(!leads(&lines, "a2", false));
    lines.push(1, "a2");
    assert!(!leads(&lines, "a2", true), "a2 waits behind a1");
    lines.done(1);
    assert!(leads(&lines, "a2", true));
    assert_eq!(lines.take(), Some((1, "a2")));
    assert!(leads(&lines, "a2", true));
    lines.done(1);
    assert!(lines.is_empty());
}

/// A plain item.
struct Job(Box<dyn FnOnce() + Send>);

impl Run for Job {
    fn run(self) {
        (self.0)();
    }
}

fn job(f: impl FnOnce() + Send + 'static) -> Job {
    Job(Box::new(f))
}

/// An item that says whether it is plain, and whether it runs where its completion was seen.
struct Said {
    run: Box<dyn FnOnce() + Send>,
    plain: bool,
    seen: bool,
}

impl Run for Said {
    fn run(self) {
        (self.run)();
    }

    fn plain(&self) -> bool {
        self.plain
    }

    fn runs_where_seen(&self) -> bool {
        self.plain || self.seen
    }
}

fn said(plain: bool, f: impl FnOnce() + Send + 'static) -> Said {
    Said {
        run: Box::new(f),
        plain,
        seen: false,
    }
}

/// An item that is not plain but runs where its completion was seen (a transition whose only code
/// of the program's is its completion-seen handlers).
fn seen(f: impl FnOnce() + Send + 'static) -> Said {
    Said {
        run: Box::new(f),
        plain: false,
        seen: true,
    }
}

/// A plain item runs on the calling thread, after the plain items its context holds, and the
/// items posted to its context meanwhile run after it, on the pool's thread. Where the context
/// runs on a thread, or holds an item that is not plain, it is given back. The pool's one thread
/// is held on another context meanwhile, so what is posted waits.
#[test]
fn a_plain_item_runs_here_after_its_context_s_plain_items_and_before_later_ones() {
    let contexts: Contexts<Said> = Contexts::start(1).unwrap();
    let here = std::thread::current().id();
    let log = Arc::new(Mutex::new(Vec::new()));
    let note = move |log: &Arc<Mutex<Vec<(&'static str, bool)>>>, name: &'static str| {
        let log = Arc::clone(log);
        move || {
            let on_here = std::thread::current().id() == here;
            log.lock().unwrap().push((name, on_here));
        }
    };
    let (release, released) = mpsc::channel::<()>();
    let (started, starts) = mpsc::channel::<()>();
    contexts.post(
        9,
        said(false, move || {
            started.send(()).unwrap();
            released.recv_timeout(WAIT).unwrap();
        }),
    );
    starts.recv_timeout(WAIT).unwrap();
    contexts.post(1, said(true, note(&log, "earlier")));
    let (done, finished) = mpsc::channel::<()>();
    let later = {
        let (contexts, log) = (contexts.clone(), Arc::clone(&log));
        move || {
            let posted = note(&log, "later");
            contexts.post(
                1,
                said(true, move || {
                    posted();
                    done.send(()).unwrap();
                }),
            );
            note(&log, "here")();
        }
    };
    assert!(contexts.run_here(1, said(true, later), Said::plain).is_ok());
    assert_eq!(
        *log.lock().unwrap(),
        [("earlier", true), ("here", true)],
        "the context's earlier item, then the item, on this thread"
    );
    // A context holding an item that is not plain, and one a thread runs: given back.
    contexts.post(2, said(false, || {}));
    assert!(contexts
        .run_here(2, said(true, || {}), Said::plain)
        .is_err());
    assert!(contexts
        .run_here(9, said(true, || {}), Said::plain)
        .is_err());
    assert!(contexts
        .run_here(3, said(false, || {}), Said::plain)
        .is_err());
    release.send(()).unwrap();
    finished.recv_timeout(WAIT).unwrap();
    assert_eq!(
        *log.lock().unwrap(),
        [("earlier", true), ("here", true), ("later", false)],
        "the item posted meanwhile ran after it, on the pool's thread"
    );
    contexts.stop();
}

/// An item whose only code of the program's is its completion-seen handlers runs on the calling
/// thread where that thread may run them (`Run::runs_where_seen`), after its context's items that
/// may run there too; where it may run only plain items (a thread blocking in a wait), it is
/// given back, and so is any item behind an earlier one of its context that runs elsewhere.
#[test]
fn an_item_that_runs_where_seen_runs_here_only_where_the_thread_may_run_it() {
    let contexts: Contexts<Said> = Contexts::start(1).unwrap();
    let here = std::thread::current().id();
    let log = Arc::new(Mutex::new(Vec::new()));
    let note = move |log: &Arc<Mutex<Vec<(&'static str, bool)>>>, name: &'static str| {
        let log = Arc::clone(log);
        move || {
            let on_here = std::thread::current().id() == here;
            log.lock().unwrap().push((name, on_here));
        }
    };
    // Plain items only: given back, changing nothing.
    assert!(contexts
        .run_here(1, seen(note(&log, "refused")), Said::plain)
        .is_err());
    // Where the thread may run completion-seen handlers: after the context's earlier one of the
    // same kind, on this thread.
    let (release, released) = mpsc::channel::<()>();
    let (started, starts) = mpsc::channel::<()>();
    contexts.post(
        9,
        said(false, move || {
            started.send(()).unwrap();
            released.recv_timeout(WAIT).unwrap();
        }),
    );
    starts.recv_timeout(WAIT).unwrap();
    contexts.post(1, seen(note(&log, "earlier")));
    assert!(contexts
        .run_here(1, seen(note(&log, "here")), Said::runs_where_seen)
        .is_ok());
    assert_eq!(
        *log.lock().unwrap(),
        [("earlier", true), ("here", true)],
        "the context's earlier item, then the item, on this thread"
    );
    // Behind an item that runs only on the pool's threads: given back.
    contexts.post(2, said(false, || {}));
    assert!(contexts
        .run_here(2, seen(|| {}), Said::runs_where_seen)
        .is_err());
    release.send(()).unwrap();
    contexts.stop();
}

/// A plain item runs alone on the calling thread where its context runs on no thread and holds
/// nothing, and the items posted to its context meanwhile run after it, on the pool's thread.
/// Where the context holds an item, plain or not, or runs on a thread, or the item is not plain,
/// it is given back. The pool's one thread is held on another context meanwhile, so what is
/// posted waits.
#[test]
fn a_plain_item_runs_alone_only_where_its_context_is_idle() {
    let contexts: Contexts<Said> = Contexts::start(1).unwrap();
    let here = std::thread::current().id();
    let log = Arc::new(Mutex::new(Vec::new()));
    let note = move |log: &Arc<Mutex<Vec<(&'static str, bool)>>>, name: &'static str| {
        let log = Arc::clone(log);
        move || {
            let on_here = std::thread::current().id() == here;
            log.lock().unwrap().push((name, on_here));
        }
    };
    let (release, released) = mpsc::channel::<()>();
    let (started, starts) = mpsc::channel::<()>();
    contexts.post(
        9,
        said(false, move || {
            started.send(()).unwrap();
            released.recv_timeout(WAIT).unwrap();
        }),
    );
    starts.recv_timeout(WAIT).unwrap();
    let (done, finished) = mpsc::channel::<()>();
    let later = {
        let (contexts, log) = (contexts.clone(), Arc::clone(&log));
        move || {
            let posted = note(&log, "later");
            contexts.post(
                1,
                said(true, move || {
                    posted();
                    done.send(()).unwrap();
                }),
            );
            note(&log, "here")();
        }
    };
    assert!(contexts.run_alone(1, said(true, later)).is_ok());
    assert_eq!(
        *log.lock().unwrap(),
        [("here", true)],
        "the item, on this thread"
    );
    // A context holding the item posted meanwhile, one holding a plain item, one a thread runs,
    // and an item that is not plain: given back.
    assert!(contexts.run_alone(1, said(true, || {})).is_err());
    contexts.post(2, said(true, || {}));
    assert!(contexts.run_alone(2, said(true, || {})).is_err());
    assert!(contexts.run_alone(9, said(true, || {})).is_err());
    assert!(contexts.run_alone(3, said(false, || {})).is_err());
    release.send(()).unwrap();
    finished.recv_timeout(WAIT).unwrap();
    assert_eq!(
        *log.lock().unwrap(),
        [("here", true), ("later", false)],
        "the item posted meanwhile ran after it, on the pool's thread"
    );
    contexts.stop();
}

fn context(device: usize, queue: u64) -> Context {
    Context { device, queue }
}

/// While one of a queue's items blocks, the queue's next item waits and another queue's runs,
/// on another thread (Metal's measured order).
///
/// The items after the blocking one are posted once it runs. The pool starts the items of
/// different queues in no order, so another queue's item posted beside it may run before it
/// starts: under load one did, and this test failed, reading `["c"]`. The blocking item starts
/// late here (20 ms), as a thread the host schedules late does, so the test holds by its waits.
#[test]
fn a_blocked_item_holds_back_its_queue_and_no_other() {
    let contexts: Contexts<Job> = Contexts::start(THREADS).unwrap();
    let log = Arc::new(Mutex::new(Vec::new()));
    let (release, released) = mpsc::channel::<()>();
    let (started, starts) = mpsc::channel::<()>();
    let (sender, done) = mpsc::channel();
    {
        let log = Arc::clone(&log);
        contexts.post(
            1,
            job(move || {
                std::thread::sleep(Duration::from_millis(20));
                log.lock().unwrap().push("a starts");
                started.send(()).unwrap();
                released.recv_timeout(WAIT).unwrap();
                log.lock().unwrap().push("a ends");
            }),
        );
    }
    starts.recv_timeout(WAIT).unwrap();
    for (queue, name) in [(1, "b"), (2, "c")] {
        let (log, sender) = (Arc::clone(&log), sender.clone());
        contexts.post(
            queue,
            job(move || {
                log.lock().unwrap().push(name);
                sender.send(name).unwrap();
            }),
        );
    }
    assert_eq!(done.recv_timeout(WAIT).unwrap(), "c");
    assert_eq!(*log.lock().unwrap(), ["a starts", "c"]);
    release.send(()).unwrap();
    assert_eq!(done.recv_timeout(WAIT).unwrap(), "b");
    assert_eq!(*log.lock().unwrap(), ["a starts", "c", "a ends", "b"]);
    contexts.stop();
}

/// Each of `count` queues' items counts a wait for another context and holds it until told: the
/// waits each item's `block` answered (`Ok`, or the refusal's cause), in the order answered.
fn blocked_waits(
    contexts: &Contexts<Job>,
    device: usize,
    count: u64,
) -> Vec<Result<(), DeadlockCause>> {
    let (release, released) = mpsc::channel::<()>();
    let released = Arc::new(Mutex::new(released));
    let (sender, answers) = mpsc::channel();
    for queue in 0..count {
        let (sender, released) = (sender.clone(), Arc::clone(&released));
        contexts.post(
            queue,
            job(move || {
                let blocked = block(context(device, queue), context(device, count));
                sender
                    .send(blocked.as_ref().map(|_| ()).map_err(|&cause| cause))
                    .unwrap();
                if blocked.is_ok() {
                    released.lock().unwrap().recv_timeout(WAIT).unwrap();
                }
            }),
        );
    }
    let answered: Vec<Result<(), DeadlockCause>> = (0..count)
        .map(|_| answers.recv_timeout(WAIT).unwrap())
        .collect();
    for _ in answered.iter().filter(|answer| answer.is_ok()) {
        release.send(()).unwrap();
    }
    answered
}

/// Metal ran 511 handlers blocked at once, and hung at the 512th: the pool counts 511 waits, and
/// refuses the 512th.
#[test]
fn as_many_waits_as_metal_ran_at_once_are_all_counted() {
    let contexts: Contexts<Job> = Contexts::start(THREADS).unwrap();
    let answered = blocked_waits(&contexts, 0x100, 512);
    assert_eq!(answered.iter().filter(|answer| answer.is_ok()).count(), 511);
    let refused: Vec<_> = answered.iter().filter(|answer| answer.is_err()).collect();
    assert_eq!(refused, [&Err(DeadlockCause::HandlerThreadCap)]);
    contexts.stop();
}

/// A pool of four threads counts three blocked waits; the fourth would leave it no free thread,
/// and is refused for that cause.
#[test]
fn the_wait_that_would_leave_no_free_thread_is_refused() {
    let contexts: Contexts<Job> = Contexts::start(4).unwrap();
    let answered = blocked_waits(&contexts, 0x200, 4);
    assert_eq!(answered.iter().filter(|answer| answer.is_ok()).count(), 3);
    let refused: Vec<_> = answered.iter().filter(|answer| answer.is_err()).collect();
    assert_eq!(refused, [&Err(DeadlockCause::HandlerThreadCap)]);
    // Every wait let go: the count is back to none.
    let deadline = Instant::now() + WAIT;
    while contexts.census().1 != 0 {
        assert!(Instant::now() < deadline, "blocked waits left counted");
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(contexts.census().0 <= 4);
    contexts.stop();
}

/// Queue 1's item waits for queue 2; queue 2's item's wait for queue 1 would close the cycle.
#[test]
fn a_wait_closing_a_cycle_is_refused() {
    let contexts: Contexts<Job> = Contexts::start(THREADS).unwrap();
    let device = 0x300;
    let (release, released) = mpsc::channel::<()>();
    let (sender, answers) = mpsc::channel();
    {
        let sender = sender.clone();
        contexts.post(
            1,
            job(move || {
                let blocked = block(context(device, 1), context(device, 2));
                sender
                    .send(("1 waits for 2", blocked.as_ref().err().copied()))
                    .unwrap();
                released.recv_timeout(WAIT).unwrap();
            }),
        );
    }
    assert_eq!(answers.recv_timeout(WAIT).unwrap(), ("1 waits for 2", None));
    contexts.post(
        2,
        job(move || {
            let blocked = block(context(device, 2), context(device, 1));
            sender
                .send(("2 waits for 1", blocked.as_ref().err().copied()))
                .unwrap();
        }),
    );
    assert_eq!(
        answers.recv_timeout(WAIT).unwrap(),
        ("2 waits for 1", Some(DeadlockCause::Cycle))
    );
    release.send(()).unwrap();
    contexts.stop();
}

/// A thread that is not one of the pool's has no wait to count: refused, as no free handler
/// thread is left for it.
#[test]
fn a_wait_off_the_handler_threads_is_not_counted() {
    assert_eq!(
        block(context(0x400, 1), context(0x400, 2)).err(),
        Some(DeadlockCause::HandlerThreadCap)
    );
}

/// After a burst, the threads beyond the first end once idle; a stop from one of the threads
/// does not wait for itself.
#[test]
fn idle_threads_end_and_a_thread_s_own_stop_returns() {
    let contexts: Arc<Contexts<Job>> = Arc::new(Contexts::start(THREADS).unwrap());
    let answered = blocked_waits(&contexts, 0x500, 16);
    assert!(answered.iter().all(Result::is_ok));
    let deadline = Instant::now() + WAIT;
    while contexts.census().0 != 1 {
        assert!(Instant::now() < deadline, "idle threads stayed");
        std::thread::sleep(Duration::from_millis(50));
    }
    let (sender, stopped) = mpsc::channel();
    let inner = Arc::clone(&contexts);
    contexts.post(
        1,
        job(move || {
            inner.stop();
            sender.send(()).unwrap();
        }),
    );
    stopped.recv_timeout(WAIT).unwrap();
    let deadline = Instant::now() + WAIT;
    while contexts.census().0 != 0 {
        assert!(Instant::now() < deadline, "the last thread stayed");
        std::thread::sleep(Duration::from_millis(10));
    }
}
