//! The process's limit of pipeline makings' words in flight (`pipeline/compile_memory.rs`): its
//! holds, waits and lent turns, on threads of their own.

use crate::pipeline::budget::{self, Budget, Urgency};
use crate::pipeline::compile_memory::{Limit, COUNTED_WORDS};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

/// A limit of four counted makings' least words.
const MOST: usize = 4 * COUNTED_WORDS;

/// Wait until `done` holds, at most a minute (a guard against a hang).
fn until(done: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while !done() {
        assert!(Instant::now() < deadline, "timed out");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Hold `words` of `limit` at `rank` on a thread of its own, note `name` in `order` once held,
/// and keep them until `release` is sent.
fn holder(
    limit: &Arc<Limit>,
    order: &Arc<Mutex<Vec<&'static str>>>,
    (name, words, rank): (&'static str, usize, (Urgency, u64)),
) -> (std::thread::JoinHandle<()>, mpsc::Sender<()>) {
    let (limit, order) = (Arc::clone(limit), Arc::clone(order));
    let (release, released) = mpsc::channel::<()>();
    let thread = std::thread::spawn(move || {
        let held = limit.hold(words, rank);
        order.lock().unwrap().push(name);
        released.recv().unwrap();
        drop(held);
    });
    (thread, release)
}

/// A making below the counted words holds nothing, however much the counted ones in flight
/// hold; a counted one that fits goes at once, without a wait.
#[test]
fn a_making_below_the_counted_words_is_not_held() {
    let limit = Limit::new(MOST);
    let big = limit.hold(MOST * 2, (Urgency::Program, 0)).unwrap();
    assert!(big.1.is_none(), "alone, a larger making goes at once");
    assert!(limit
        .hold(COUNTED_WORDS - 1, (Urgency::Program, 1))
        .is_none());
    assert_eq!(limit.census(), (MOST * 2, 0));
    drop(big);
    let (a, waited_a) = limit.hold(COUNTED_WORDS, (Urgency::Ahead, 2)).unwrap();
    let (b, waited_b) = limit.hold(COUNTED_WORDS, (Urgency::Ahead, 3)).unwrap();
    assert!(waited_a.is_none() && waited_b.is_none());
    assert_eq!(limit.census(), (2 * COUNTED_WORDS, 0));
    drop((a, b));
    assert_eq!(limit.census(), (0, 0));
}

/// Two makings that do not fit together never hold their words at once: the second waits
/// until the first gives its back, and says how long it waited.
#[test]
fn two_large_makings_never_hold_their_words_at_once() {
    let limit = Arc::new(Limit::new(MOST));
    let order = Arc::new(Mutex::new(Vec::new()));
    let (first, release_first) = holder(&limit, &order, ("first", MOST, (Urgency::Program, 0)));
    until(|| limit.census() == (MOST, 0));
    let (second, release_second) = holder(&limit, &order, ("second", MOST, (Urgency::Program, 1)));
    until(|| limit.census() == (MOST, 1));
    assert_eq!(*order.lock().unwrap(), ["first"]);
    release_first.send(()).unwrap();
    first.join().unwrap();
    until(|| order.lock().unwrap().len() == 2);
    assert_eq!(limit.census(), (MOST, 0));
    release_second.send(()).unwrap();
    second.join().unwrap();
    assert_eq!(limit.census(), (0, 0));
}

/// A smaller making that would fit beside the one in flight does not pass a larger one waiting
/// before it, and the waiters go in rank order, the most urgent first, whatever order they
/// asked in.
#[test]
fn no_counted_making_passes_one_waiting_before_it() {
    let limit = Arc::new(Limit::new(MOST));
    let order = Arc::new(Mutex::new(Vec::new()));
    let (first, release_first) = holder(
        &limit,
        &order,
        ("in flight", 2 * COUNTED_WORDS, (Urgency::Ahead, 0)),
    );
    until(|| limit.census() == (2 * COUNTED_WORDS, 0));
    let (large, release_large) = holder(&limit, &order, ("large", MOST, (Urgency::Ahead, 1)));
    until(|| limit.census().1 == 1);
    let (small, release_small) = holder(
        &limit,
        &order,
        ("small", COUNTED_WORDS, (Urgency::Ahead, 2)),
    );
    until(|| limit.census().1 == 2);
    let (urgent, release_urgent) =
        holder(&limit, &order, ("committed", MOST, (Urgency::Committed, 3)));
    until(|| limit.census().1 == 3);
    assert_eq!(*order.lock().unwrap(), ["in flight"]);
    release_first.send(()).unwrap();
    first.join().unwrap();
    until(|| order.lock().unwrap().len() == 2);
    assert_eq!(*order.lock().unwrap(), ["in flight", "committed"]);
    release_urgent.send(()).unwrap();
    urgent.join().unwrap();
    until(|| order.lock().unwrap().len() == 3);
    assert_eq!(order.lock().unwrap()[2], "large");
    release_large.send(()).unwrap();
    large.join().unwrap();
    until(|| order.lock().unwrap().len() == 4);
    release_small.send(()).unwrap();
    small.join().unwrap();
    assert_eq!(
        *order.lock().unwrap(),
        ["in flight", "committed", "large", "small"]
    );
    assert_eq!(limit.census(), (0, 0));
}

/// A making that waits for words gives its turn back for the wait: a making asked for
/// meanwhile takes it, though the budget has that one turn; the waiter takes one again once
/// its words are held.
#[test]
fn a_making_waiting_for_words_lends_its_turn() {
    let budget = Arc::new(Budget::new(1, 1));
    let limit = Arc::new(Limit::new(MOST));
    let order = Arc::new(Mutex::new(Vec::new()));
    let (first, release_first) = holder(&limit, &order, ("first", MOST, (Urgency::Program, 0)));
    until(|| limit.census() == (MOST, 0));
    let (held, waiting) = mpsc::channel();
    let waiter = {
        let (budget, limit) = (Arc::clone(&budget), Arc::clone(&limit));
        std::thread::spawn(move || {
            let _turn = budget::waited_turn(&budget, budget::ticket());
            let words = limit.hold(MOST, (Urgency::Program, 1));
            assert!(budget::holds_turn(), "its turn is taken again");
            held.send(words.is_some_and(|(_, waited)| waited.is_some()))
                .unwrap();
        })
    };
    until(|| limit.census() == (MOST, 1));
    until(|| budget.census() == (0, 0));
    let other = {
        let budget = Arc::clone(&budget);
        std::thread::spawn(move || drop(budget::waited_turn(&budget, budget::ticket())))
    };
    other.join().unwrap();
    release_first.send(()).unwrap();
    first.join().unwrap();
    assert!(waiting.recv().unwrap(), "the waiter says it waited");
    waiter.join().unwrap();
    until(|| budget.census() == (0, 0));
    assert_eq!(limit.census(), (0, 0));
}

/// The memory log's lines end with glibc's heap where the process's allocator is glibc's:
/// `mallinfo2`'s figures and the trims the process's large makings made; elsewhere with nothing
/// (`compile_memory::heap`).
#[test]
fn the_memory_log_states_glibcs_heap_where_it_is_the_allocator() {
    let heap = crate::pipeline::compile_memory::heap();
    if cfg!(all(target_os = "linux", target_env = "gnu")) {
        let heap = heap.expect("glibc's heap");
        assert!(heap.starts_with("glibc heap: "), "{heap}");
        assert!(heap.contains(" bytes in use, "), "{heap}");
        assert!(heap.ends_with(" trims"), "{heap}");
    } else {
        assert_eq!(heap, None);
    }
}
