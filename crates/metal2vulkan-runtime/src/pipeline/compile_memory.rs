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
    pub(crate) fn census(&self) -> (usize, usize) {
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
