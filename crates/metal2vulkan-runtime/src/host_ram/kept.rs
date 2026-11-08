//! The lists of a copy's runs a RAM keeps past the copy (`RamOnDevice::run_lists`): each import
//! of a list (`HostRam::with_dma_buf_of_runs`) is kept once its copy made it, so that a later copy
//! of runs it holds (a device cache filled and downloaded again, a tile copied both ways) takes it
//! again instead of a list of its own, as a copy over an imported chunk imports nothing.
//!
//! - **Bounded.** The lists kept hold at most [`KEPT_BYTES`] and are at most [`KEPT_LISTS`]; the
//!   least recently used go first past either ([`Kept::keep`]). Each kept list is one more import
//!   every submission of the device lists, as each live chunk is, and keeps its pages pinned
//!   (neither swapped out nor moved). Its copy made those pages resident as it imported them, and
//!   they stay in the RAM's file once it is freed, so keeping it makes no more of the RAM
//!   resident.
//! - **Released when idle.** A list no copy took for `host_residency::IDLE` and no command buffer
//!   holds goes ([`Kept::idle`], every `host_residency::TICK` by the device's thread for the RAM,
//!   `listed_chunks`); all go as the device's use of the RAM ends ([`Kept::all`]).
//! - **Taken whole or not at all.** A copy's run is taken from a kept list only where one of the
//!   list's runs holds it whole ([`Kept::find`]); the copy's other runs take lists of their own.

use std::ops::Range;
use std::time::{Duration, Instant};

/// The most bytes of lists a RAM keeps past their copies on one device: 64 MiB, the pages of
/// about nine tiles of 1,800 guest pages.
pub(crate) const KEPT_BYTES: u64 = 64 << 20;

/// The most lists a RAM keeps past their copies on one device: as many imports as an 8 GiB RAM's
/// 512 MiB chunks, each listed in every submission.
pub(crate) const KEPT_LISTS: usize = 16;

/// The lists kept, `T` each one's import, and their bytes together.
pub(crate) struct Kept<T> {
    lists: Vec<KeptList<T>>,
    bytes: u64,
}

/// One list kept: its runs (host pages, by start, apart), its import, its bytes and when a copy
/// last took it.
struct KeptList<T> {
    runs: Box<[Range<u64>]>,
    import: T,
    bytes: u64,
    used: Instant,
}

impl<T> Default for Kept<T> {
    fn default() -> Kept<T> {
        Kept {
            lists: Vec::new(),
            bytes: 0,
        }
    }
}

/// Whether one of `runs` (by start, apart) holds `run` whole.
fn holds(runs: &[Range<u64>], run: &Range<u64>) -> bool {
    let after = runs.partition_point(|held| held.start <= run.start);
    after
        .checked_sub(1)
        .is_some_and(|at| run.end <= runs[at].end)
}

impl<T> Kept<T> {
    /// For each of `runs` (by start), the index of a kept list that holds it whole, where one
    /// does: the list that held the run before it is asked first, so a copy of a kept list's runs
    /// again asks one list a run. Each list found was taken at `now`.
    pub(crate) fn find(&mut self, runs: &[Range<u64>], now: Instant) -> Vec<Option<usize>> {
        let mut last: Option<usize> = None;
        let mut found = Vec::with_capacity(runs.len());
        for run in runs {
            let at = (last.into_iter())
                .chain(0..self.lists.len())
                .find(|&index| holds(&self.lists[index].runs, run));
            if let Some(index) = at {
                self.lists[index].used = now;
                last = Some(index);
            }
            found.push(at);
        }
        found
    }

    /// The import of list `index` ([`Kept::find`]'s).
    pub(crate) fn import(&self, index: usize) -> &T {
        &self.lists[index].import
    }

    /// Keep `import` of the list of `runs` (`bytes` of pages), made at `now`, the least recently
    /// used lists let go of until those kept hold at most `most_bytes` and are at most
    /// `most_lists`: what goes, for the caller to drop with no lock held (`import` itself where
    /// it alone passes `most_bytes`).
    pub(crate) fn keep(
        &mut self,
        (runs, import, bytes): (Box<[Range<u64>]>, T, u64),
        now: Instant,
        (most_bytes, most_lists): (u64, usize),
    ) -> Vec<T> {
        if bytes > most_bytes || most_lists == 0 {
            return vec![import];
        }
        self.lists.push(KeptList {
            runs,
            import,
            bytes,
            used: now,
        });
        self.bytes += bytes;
        let mut gone = Vec::new();
        while self.bytes > most_bytes || self.lists.len() > most_lists {
            let Some(oldest) = (0..self.lists.len()).min_by_key(|&index| self.lists[index].used)
            else {
                break;
            };
            let list = self.lists.swap_remove(oldest);
            self.bytes -= list.bytes;
            gone.push(list.import);
        }
        gone
    }

    /// The lists no copy took for `window` by `now` and that `held` says nothing else holds, let
    /// go of, for the caller to drop with no lock held.
    pub(crate) fn idle(
        &mut self,
        now: Instant,
        window: Duration,
        held: impl Fn(&T) -> bool,
    ) -> Vec<T> {
        let mut gone = Vec::new();
        let mut index = 0;
        while index < self.lists.len() {
            let list = &self.lists[index];
            if now.duration_since(list.used) < window || held(&list.import) {
                index += 1;
                continue;
            }
            let list = self.lists.swap_remove(index);
            self.bytes -= list.bytes;
            gone.push(list.import);
        }
        gone
    }

    /// Every list, let go of: the device's use of the RAM ended.
    pub(crate) fn all(&mut self) -> Vec<T> {
        self.bytes = 0;
        self.lists.drain(..).map(|list| list.import).collect()
    }

    /// The bytes of the lists kept.
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: u64 = 4096;

    fn pages(first: u64, count: u64) -> Range<u64> {
        first * PAGE..(first + count) * PAGE
    }

    #[test]
    fn a_run_is_taken_from_a_kept_list_only_where_one_of_its_runs_holds_it_whole() {
        let now = Instant::now();
        let mut kept: Kept<u32> = Kept::default();
        let one: Box<[Range<u64>]> = Box::new([pages(0, 4), pages(10, 2)]);
        let two: Box<[Range<u64>]> = Box::new([pages(20, 1), pages(30, 3)]);
        let most = (KEPT_BYTES, KEPT_LISTS);
        assert!(kept.keep((one, 1, 6 * PAGE), now, most).is_empty());
        assert!(kept.keep((two, 2, 4 * PAGE), now, most).is_empty());
        let runs = [
            pages(1, 2),  // inside the first list's first run
            pages(10, 2), // its second, whole
            pages(11, 2), // past its end: no list
            pages(30, 3), // the second list's
            pages(40, 1), // none
        ];
        let found = kept.find(&runs, now);
        assert_eq!(found, vec![Some(0), Some(0), None, Some(1), None]);
        assert_eq!((*kept.import(0), *kept.import(1)), (1, 2));
    }

    #[test]
    fn kept_lists_past_their_bytes_or_count_let_the_least_recently_used_go_and_idle_ones_go() {
        let start = Instant::now();
        let later = start + Duration::from_secs(1);
        let mut kept: Kept<u32> = Kept::default();
        let list = |first: u64| -> Box<[Range<u64>]> { Box::new([pages(first, 1)]) };
        let most = (3 * PAGE, 16);
        assert!(kept.keep((list(0), 1, PAGE), start, most).is_empty());
        assert!(kept.keep((list(10), 2, PAGE), start, most).is_empty());
        // The first taken again: the second is the least recently used.
        assert_eq!(kept.find(&[pages(0, 1)], later), vec![Some(0)]);
        assert_eq!(kept.keep((list(20), 3, 2 * PAGE), later, most), vec![2]);
        assert_eq!(kept.bytes(), 3 * PAGE);
        // A list past the bytes alone is not kept.
        assert_eq!(kept.keep((list(30), 4, 4 * PAGE), later, most), vec![4]);
        // Past the count: the least recently used goes, whatever the bytes.
        let mut counted: Kept<u32> = Kept::default();
        assert!(counted
            .keep((list(0), 5, PAGE), start, (KEPT_BYTES, 2))
            .is_empty());
        assert!(counted
            .keep((list(1), 6, PAGE), later, (KEPT_BYTES, 2))
            .is_empty());
        assert_eq!(
            counted.keep((list(2), 7, PAGE), later, (KEPT_BYTES, 2)),
            vec![5]
        );
        // Idle for the window: those nothing else holds go.
        let window = Duration::from_secs(2);
        let idle = start + Duration::from_millis(2500);
        assert!(
            kept.idle(idle, window, |_| false).is_empty(),
            "taken at 1 s"
        );
        let gone = kept.idle(later + window, window, |&import| import == 3);
        assert_eq!(gone, vec![1], "one held");
        assert_eq!(kept.all(), vec![3]);
        assert_eq!(kept.bytes(), 0);
    }
}
