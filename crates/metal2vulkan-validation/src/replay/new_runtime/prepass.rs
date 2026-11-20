//! The consumer's pre-pass the player stands in for: threads of the player's own that prepare the
//! compute dispatches the read-ahead found due (`Backend::prepare`) with
//! `ComputePipelineState::prepare_dispatch`, each thread taking the next in stream order while
//! the play goes on. A dispatch that needs what one of them is making waits for that make alone.

use super::rt;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;

/// A dispatch to prepare: its state, its grid, its threadgroup size and the encoder's threadgroup
/// memory lengths.
pub(super) type Item = (
    rt::ComputePipelineState,
    rt::Dispatch,
    [u32; 3],
    Vec<(u32, u32)>,
);

/// The pre-pass's threads and their queue.
pub(super) struct Pool {
    queue: Option<Sender<Item>>,
    /// Set at the drop: what is still queued is left.
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl Pool {
    /// A pre-pass `width` threads wide.
    pub(super) fn new(width: usize) -> Pool {
        let (queue, items) = channel::<Item>();
        let items = Arc::new(Mutex::new(items));
        let stop = Arc::new(AtomicBool::new(false));
        let threads = (0..width)
            .filter_map(|at| {
                let (items, stop) = (Arc::clone(&items), Arc::clone(&stop));
                let run = move || loop {
                    let next = (items.lock().unwrap_or_else(PoisonError::into_inner)).recv();
                    let Ok((state, grid, size, lengths)) = next else {
                        return;
                    };
                    if stop.load(Relaxed) {
                        return;
                    }
                    state.prepare_dispatch(grid, size, &lengths);
                };
                let named = std::thread::Builder::new().name(format!("m2v-prepass-{at}"));
                named.spawn(run).ok()
            })
            .collect();
        Pool {
            queue: Some(queue),
            stop,
            threads,
        }
    }

    /// Queues `item` for the next free thread.
    pub(super) fn give(&self, item: Item) {
        if let Some(queue) = &self.queue {
            let _ = queue.send(item);
        }
    }

    /// The pre-pass's width: its threads.
    pub(super) fn width(&self) -> usize {
        self.threads.len()
    }
}

impl Drop for Pool {
    /// Leaves what is queued and waits for each thread's call in flight, so no state outlives the
    /// play's device on a pre-pass thread.
    fn drop(&mut self) {
        self.stop.store(true, Relaxed);
        self.queue = None;
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}
