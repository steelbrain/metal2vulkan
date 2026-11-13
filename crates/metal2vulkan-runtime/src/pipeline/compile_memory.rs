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
//! - **The free heap handed back** ([`LargeMaking`]). What the compiler took it frees when the
//!   making returns, and glibc's `malloc` keeps what a thread frees in that thread's arena, so a
//!   process whose threads each made a large pipeline holds each one's peak as free heap until it
//!   ends: on an RX 7900 XTX (RADV, Mesa 26.2.6), one kernel of 200,000 fused multiply-adds with 600
//!   values live at once made as 9 `VkPipeline`s took the process's resident set from 105 MiB to
//!   7.3 GiB, of which `malloc_trim` handed back all but 327 MiB (one run; the `VkPipeline`s and the
//!   driver's caches held about 170 MiB of it). So where a large making's return leaves glibc more
//!   than [`FREE_FLOOR`] bytes free (`mallinfo2`'s `fordblks`, every arena's), the making's thread
//!   calls `malloc_trim(0)` before it gives its words back, counted in `WorkCounts::heap_trims`.
//!   It trims glibc's own arenas, process-wide: the host's free heap goes back with ours. The
//!   trade: the next large making faults its working set in again, about 1 s per 3.5 GiB of it,
//!   where it would have reused the free heap; the floor spares makings that leave less. Only
//!   where the process's allocator is glibc's (Linux, the GNU environment): on any other nothing is
//!   called, and a host whose `malloc` is not glibc's sees no change.
//! - **Stated by the memory log** ([`heap`]). Each line of `METAL2VULKAN_LOG_MEMORY`'s log
//!   (`memory_log.rs`) ends with glibc's heap as `mallinfo2` states it then and the trims the
//!   process made so far, so a resident set that grows and stays reads as free heap, heap in use,
//!   or neither (memory `malloc` did not map).

use super::budget::{self, Rank, Urgency};
use crate::device::lock;
use crate::work_counts::Counters;
use std::sync::atomic::{AtomicU64, Ordering};
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

/// The most free heap a large making's return leaves glibc before it is handed back: 256 MiB
/// (the module docs).
pub(crate) const FREE_FLOOR: usize = 256 << 20;

/// A large making's words of the process's limit, held around its driver call: when dropped, it
/// hands glibc's free heap back where it holds more than [`FREE_FLOOR`] bytes (the module docs),
/// then gives the words back.
pub(crate) struct LargeMaking<'a> {
    pub(crate) _words: Words<'static>,
    pub(crate) counts: &'a Counters,
}

impl Drop for LargeMaking<'_> {
    fn drop(&mut self) {
        if return_free_heap() {
            self.counts.heap_trimmed();
        }
    }
}

/// The `malloc_trim` calls the process's large makings made ([`return_free_heap`]), every
/// device's: the memory log's count ([`heap`]). `WorkCounts::heap_trims` counts a device's.
static TRIMS: AtomicU64 = AtomicU64::new(0);

/// Hand glibc's free heap back to the system where it holds more than [`FREE_FLOOR`] bytes free,
/// every arena's: whether it called `malloc_trim` (the module docs).
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn return_free_heap() -> bool {
    // SAFETY: no precondition.
    let free = unsafe { libc::mallinfo2() }.fordblks;
    if free <= FREE_FLOOR {
        return false;
    }
    // SAFETY: no precondition; it returns free pages only.
    unsafe { libc::malloc_trim(0) };
    TRIMS.fetch_add(1, Ordering::Relaxed);
    true
}

/// glibc's heap now, every arena's, as `mallinfo2` states it, and the trims the process's large
/// makings made: the end of a memory log line (the module docs). `None` where the process's
/// allocator is not glibc's. `mallinfo2` takes each arena's lock and walks its free chunks, which
/// only the log, a default-off diagnostic, asks for.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub(crate) fn heap() -> Option<String> {
    // SAFETY: no precondition.
    let info = unsafe { libc::mallinfo2() };
    Some(format!(
        "glibc heap: {} bytes in use, {} free, of {} its arenas hold, {} mapped apart; {} trims",
        info.uordblks,
        info.fordblks,
        info.arena,
        info.hblkhd,
        TRIMS.load(Ordering::Relaxed)
    ))
}

/// Nothing where the process's allocator is not glibc's (the module docs).
#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
pub(crate) fn heap() -> Option<String> {
    None
}

/// Nothing where the process's allocator is not glibc's (the module docs).
#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn return_free_heap() -> bool {
    false
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
