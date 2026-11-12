//! The process's limit on its large pipeline makings in flight, by the SPIR-V words they hand the
//! driver's compiler: two large makings never compile at once.
//!
//! - **What counts.** A program's pipeline making whose stages hold [`COUNTED_WORDS`] SPIR-V
//!   words or more: a compute pipeline's kernel (`objects.rs`), a render pipeline's stages and a
//!   graphics pipeline library's own (`graphics.rs`). Every other making goes as the budget's
//!   turns let it (`budget.rs`), the runtime's own pipelines among them, each far below.
//! - **The limit.** The counted makings in flight hold at most [`WORDS_IN_FLIGHT`] words at once,
//!   a single larger one alone. A driver's compiler takes memory as it compiles, more the larger
//!   the module: RADV (Mesa 26.2.6, ACO) on the Radeon 780M, one making at a time in a process of
//!   its own, peaked at 3.6 GiB resident for a kernel of 1.9 M words, 4.2 GiB for 3.8 M, 8.5 GiB
//!   for 2.3 M and 10.3 GiB for 4.5 M (two forms of two kernels), where a guest's own memory
//!   shares the host's.
//! - **Waiting** ([`hold`]). A counted making that does not fit waits for the counted makings in
//!   flight, holding no turn of the budget's: it lends its thread's for the wait, as a wait for a
//!   twin does (`budget::without_turn`), and takes one again after it. The waiters go in the
//!   order of their turns' ranks, as they were when they asked (`budget::held_rank`: the most
//!   urgent first, then the earlier ticket); no counted making passes one waiting before it, so a
//!   large making waits only for those in flight, never for a stream of later ones.
//! - **Only around the driver call.** A making holds its words from just before its
//!   `vkCreate*Pipelines` call to just after it, or it may wait for its turn again holding them.
//!   No making waits for words holding a turn, so the two never wait on each other, and every
//!   holder's driver call ends.
//! - **Nothing seen is decided here.** Which pipelines are made, their keys and everything the
//!   program sees stay the callers'; the limit only delays a driver call.
//! - **Counted.** A making that waited counts in `WorkCounts::makings_held_for_compile_memory`,
//!   and its wait in `WorkCounts::compile_memory_held_nanos`.

use super::budget::{self, Rank, Urgency};
use crate::device::lock;
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The fewest SPIR-V words a making's stages hold for it to count: 65,536.
pub(crate) const COUNTED_WORDS: usize = 1 << 16;

/// The most words of counted makings in flight at once, a single larger making alone:
/// 4,194,304, about one large kernel's (the module docs).
pub(crate) const WORDS_IN_FLIGHT: usize = 1 << 22;

/// The process's limit, which every device's makings share.
static PROCESS: OnceLock<Limit> = OnceLock::new();

/// A limit of words in flight (the module docs): the process's, or a test's own.
pub(crate) struct Limit {
    state: Mutex<State>,
    freed: Condvar,
    most: usize,
}

struct State {
    /// The words the counted makings in flight hold.
    held: usize,
    /// Those waiting, each by an id of its own, with its rank and words.
    waiting: Vec<(u64, Rank, usize)>,
    /// The id of the next waiter.
    next: u64,
}

/// The words a making holds, given back when dropped.
pub(crate) struct Words<'a> {
    limit: &'a Limit,
    words: usize,
}

impl Drop for Words<'_> {
    fn drop(&mut self) {
        lock(&self.limit.state).held -= self.words;
        self.limit.freed.notify_all();
    }
}

impl Limit {
    /// A limit of `most` words in flight.
    pub(crate) fn new(most: usize) -> Limit {
        Limit {
            state: Mutex::new(State {
                held: 0,
                waiting: Vec::new(),
                next: 0,
            }),
            freed: Condvar::new(),
            most,
        }
    }

    /// Hold `words` for a making whose stages hold that many, ranked `rank` where it waits: `None`
    /// where it does not count ([`COUNTED_WORDS`]); else its words, and how long it waited for
    /// them where it did. The wait lends this thread's turn (the module docs).
    pub(crate) fn hold(&self, words: usize, rank: Rank) -> Option<(Words<'_>, Option<Duration>)> {
        if words < COUNTED_WORDS {
            return None;
        }
        let mut state = lock(&self.state);
        if state.waiting.is_empty() && self.fits(state.held, words) {
            state.held += words;
            return Some((self.words(words), None));
        }
        let id = state.next;
        state.next += 1;
        state.waiting.push((id, rank, words));
        drop(state);
        let asked = Instant::now();
        budget::without_turn(|| self.wait(id));
        Some((self.words(words), Some(asked.elapsed())))
    }

    /// Wait as waiter `id` until it is first among the waiters and fits, then hold its words.
    fn wait(&self, id: u64) {
        let mut state = lock(&self.state);
        loop {
            let first = (state.waiting.iter().enumerate())
                .min_by_key(|(_, (at, rank, _))| (*rank, *at))
                .map(|(i, &(at, _, words))| (i, at, words));
            if let Some((i, at, words)) = first {
                if at == id && self.fits(state.held, words) {
                    state.waiting.remove(i);
                    state.held += words;
                    drop(state);
                    // The next waiter may fit too.
                    self.freed.notify_all();
                    return;
                }
            }
            state = self
                .freed
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// Whether a making of `words` may go beside makings holding `held`.
    fn fits(&self, held: usize, words: usize) -> bool {
        held == 0 || held + words <= self.most
    }

    fn words(&self, words: usize) -> Words<'_> {
        Words { limit: self, words }
    }

    /// The words held now and the waiters (the tests).
    #[cfg(test)]
    fn census(&self) -> (usize, usize) {
        let state = lock(&self.state);
        (state.held, state.waiting.len())
    }
}

/// [`Limit::hold`] of the process's limit, ranked by this thread's turn where it holds one, else as
/// a program's making asked now.
pub(crate) fn hold(words: usize) -> Option<(Words<'static>, Option<Duration>)> {
    if words < COUNTED_WORDS {
        return None;
    }
    let rank = budget::held_rank().unwrap_or_else(|| (Urgency::Program, budget::ticket()));
    PROCESS
        .get_or_init(|| Limit::new(WORDS_IN_FLIGHT))
        .hold(words, rank)
}

#[cfg(test)]
mod tests {
    use super::{Limit, COUNTED_WORDS};
    use crate::pipeline::budget::{self, Budget, Urgency};
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
        let (second, release_second) =
            holder(&limit, &order, ("second", MOST, (Urgency::Program, 1)));
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
}
