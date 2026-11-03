//! A command buffer's batches: its encoders, split where it waits for events or for the caller's
//! binary semaphores, and before the part of an encoder that executes commands the GPU may write,
//! which runs once everything before it completed.
//!
//! A batch is its waits, then its encoders, then its signals, and the scheduler submits each batch
//! once its waits are satisfied. A wait after an encoder or a signal starts a new batch, since
//! everything encoded before it must be able to run, and to signal, while it waits (measured on
//! Metal). An encoder after a signal starts one too, so the signal happens once the encoders before
//! it completed, not the ones after. Within one batch, two waits for one event are the wait for the
//! larger value, and two signals of one event the signal of the larger value, which is what Metal's
//! maximum makes of them. A binary semaphore is waited for or signaled at most once per batch: each
//! wait consumes one signal, so a second wait or signal of the same semaphore starts a new batch,
//! in order. Every command buffer has at least one batch, so even one with nothing encoded
//! completes in its queue's order.

/// One batch: waits, a run of encoders, and signals, with events of type `E` and binary
/// semaphores of type `S`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Batch<E, S> {
    /// The events waited for, each once, with the value.
    pub(crate) waits: Vec<(E, u64)>,
    /// The binary semaphores waited for, each once.
    pub(crate) binary_waits: Vec<S>,
    /// The number of the command buffer's encoders before this batch's end.
    pub(crate) end: usize,
    /// The events signaled, each once, with the value.
    pub(crate) signals: Vec<(E, u64)>,
    /// The binary semaphores signaled, each once.
    pub(crate) binary_signals: Vec<S>,
}

impl<E, S> Batch<E, S> {
    fn new(end: usize) -> Batch<E, S> {
        Batch {
            waits: Vec::new(),
            binary_waits: Vec::new(),
            end,
            signals: Vec::new(),
            binary_signals: Vec::new(),
        }
    }

    /// Whether the batch signals anything.
    fn signals_any(&self) -> bool {
        !self.signals.is_empty() || !self.binary_signals.is_empty()
    }
}

/// The batches of a command buffer being encoded: the last one is the current one.
#[derive(Debug)]
pub(crate) struct Batches<E, S> {
    batches: Vec<Batch<E, S>>,
    /// The encoders of the current batch.
    encoders: usize,
    /// The resource state encoders the current batch begins with.
    binds: usize,
    /// The event waits encoded so far (`encodeWaitForEvent:value:`), each call counted.
    event_waits: usize,
    /// The event signals encoded so far (`encodeSignalEvent:value:`), each call counted.
    event_signals: usize,
}

impl<E, S> Default for Batches<E, S> {
    fn default() -> Batches<E, S> {
        Batches {
            batches: vec![Batch::new(0)],
            encoders: 0,
            binds: 0,
            event_waits: 0,
            event_signals: 0,
        }
    }
}

impl<E, S> Batches<E, S> {
    /// An encoder ended.
    pub(crate) fn encoder(&mut self) {
        if self.current().signals_any() {
            self.split();
        }
        self.encoders += 1;
        self.current().end += 1;
    }

    /// The next encoder, the part of one that executes commands the GPU may write, starts a
    /// batch of its own: the scheduler submits it once everything before it completed and its
    /// host step recorded the commands.
    pub(crate) fn host_step(&mut self) {
        self.split();
    }

    /// The next encoder is a resource state encoder: its updates bind before its batch's work, so
    /// it starts a batch unless the current one holds only such encoders and no signal.
    pub(crate) fn sparse(&mut self) {
        if self.encoders > self.binds || self.current().signals_any() {
            self.split();
        }
        self.binds += 1;
    }

    /// The batches, in order: at least one.
    pub(crate) fn all(&self) -> &[Batch<E, S>] {
        &self.batches
    }

    /// Whether the current batch signals, so the next encoder starts a batch of its own
    /// ([`encoder`](Self::encoder)).
    pub(crate) fn signals_pending(&self) -> bool {
        self.batches.last().is_some_and(Batch::signals_any)
    }

    /// The event waits and the event signals encoded so far, each call counted, binary
    /// semaphores' none: an encoder that ends now has them all before it
    /// (`CommandBuffer::ended_encoders`).
    pub(crate) fn events(&self) -> (usize, usize) {
        (self.event_waits, self.event_signals)
    }

    /// Signal no event (`CommandBuffer::commit_failed` without signals): every batch keeps its
    /// waits, encoders and binary semaphore signals.
    pub(crate) fn drop_signals(&mut self) {
        for batch in &mut self.batches {
            batch.signals.clear();
        }
    }

    /// Signal no event after the first `parts` encoders (or parts of them): a command buffer
    /// Metal fails from an encoder on signals what was encoded before it. The batches that end
    /// past them keep their waits, encoders and binary semaphore signals.
    pub(crate) fn drop_signals_after(&mut self, parts: usize) {
        for batch in self.batches.iter_mut().filter(|batch| batch.end > parts) {
            batch.signals.clear();
        }
    }

    fn current(&mut self) -> &mut Batch<E, S> {
        let last = self.batches.len() - 1;
        &mut self.batches[last]
    }

    fn split(&mut self) {
        let end = self.current().end;
        self.batches.push(Batch::new(end));
        self.encoders = 0;
        self.binds = 0;
    }

    /// Start a new batch where a wait cannot join the current one: it has encoders or signals.
    fn before_wait(&mut self) {
        if self.encoders > 0 || self.current().signals_any() {
            self.split();
        }
    }
}

impl<E: PartialEq, S: PartialEq> Batches<E, S> {
    /// `encodeWaitForEvent:value:`.
    pub(crate) fn wait(&mut self, event: E, value: u64) {
        self.event_waits += 1;
        self.before_wait();
        add_max(&mut self.current().waits, event, value);
    }

    /// The runtime's own wait for `event` to reach `value`, which no `encodeWaitForEvent:value:`
    /// made: it splits the batches as one does, and is not counted among the event waits
    /// ([`events`](Self::events)).
    pub(crate) fn hold(&mut self, event: E, value: u64) {
        self.before_wait();
        add_max(&mut self.current().waits, event, value);
    }

    /// The runtime's own signal of `event` to `value`, which no `encodeSignalEvent:value:` made:
    /// it ends the batch as one does, and is not counted among the event signals
    /// ([`events`](Self::events)).
    pub(crate) fn hold_signal(&mut self, event: E, value: u64) {
        add_max(&mut self.current().signals, event, value);
    }

    /// `encodeSignalEvent:value:`.
    pub(crate) fn signal(&mut self, event: E, value: u64) {
        self.event_signals += 1;
        add_max(&mut self.current().signals, event, value);
    }

    /// A wait for binary semaphore `semaphore` (`encode_wait_for_vk_semaphore`).
    pub(crate) fn binary_wait(&mut self, semaphore: S) {
        self.before_wait();
        if self.current().binary_waits.contains(&semaphore) {
            self.split();
        }
        self.current().binary_waits.push(semaphore);
    }

    /// A signal of binary semaphore `semaphore` (`encode_signal_vk_semaphore`).
    pub(crate) fn binary_signal(&mut self, semaphore: S) {
        if self.current().binary_signals.contains(&semaphore) {
            self.split();
        }
        self.current().binary_signals.push(semaphore);
    }
}

/// Add `event` at `value` to `list`, which holds each event once at its largest value.
fn add_max<E: PartialEq>(list: &mut Vec<(E, u64)>, event: E, value: u64) {
    match list.iter_mut().find(|(known, _)| *known == event) {
        Some((_, known)) => *known = (*known).max(value),
        None => list.push((event, value)),
    }
}
