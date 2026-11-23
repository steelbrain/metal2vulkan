//! The pipeline state makes of a play, written as they happen, for a driver that ends the process
//! inside a make: with a state log (`NewRuntime::with_states`), each make writes a `M2V-STATE begin` line
//! before its first driver call, a `M2V-STATE modules` line once its shader modules are made and
//! before its `VkPipeline` is, and a `M2V-STATE end` line after, each written and flushed to the
//! standard output and to the log as it happens. So the last `begin` with no `end` names the make
//! a crash ended, and its `modules` line, or none, says which of the two driver calls ended it.
//!
//! A states-only play's makes can go to a pool of the player's own threads (`with_states`),
//! which make them at once through the device's one pipeline cache, as a consumer's compile
//! threads do; the runtime's reaper reads the cache between them as it does for any consumer.
//! Without the pool each state is made on the play's thread, one after another. A list of render
//! contents' keys (the hex of the translator's key of each descriptor, the name of its record in
//! the cache directory) makes those states alone and passes over every other state, each named on
//! a `M2V-STATE pass` line. With the digests on, a render state's `M2V-STATE modules` line ends in
//! its content digest (`RenderPipelineState::content_digest`), which two builds share where they
//! translate its content to the same pipeline.

use super::rt;
use metal2vulkan::cache::Hash256;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering::SeqCst};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Instant;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A make the pool runs: its state's line, and the make, given the pool's sampler numbering and
/// the make's step marks. The state it made is dropped at once: the device's map of contents
/// keeps its content and `VkPipeline`s, as it keeps those of a state a program holds.
type Job = (
    String,
    Box<dyn FnOnce(&Numbering, &Step) -> Result<(), rt::Error> + Send>,
);

/// Where the lines go: the standard output and the state log, each line flushed as it is written;
/// nowhere without a log, a pool or the digests. With `digests`, a render state's modules line
/// carries its content digest.
struct Log {
    on: bool,
    digests: bool,
    file: Mutex<Option<File>>,
}

impl Log {
    fn line(&self, line: &str) {
        if !self.on {
            return;
        }
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{line}");
        let _ = out.flush();
        if let Some(file) = lock(&self.file).as_mut() {
            let _ = writeln!(file, "{line}");
            let _ = file.flush();
        }
    }
}

/// The makes begun, and how those that ended ended: made, refused (the runtime's or the
/// translator's typed answer) or failed (a Vulkan call's error).
#[derive(Default)]
struct Tally {
    begun: AtomicU64,
    made: AtomicU64,
    refused: AtomicU64,
    failed: AtomicU64,
    passed: AtomicU64,
}

/// One make's marks: its sequence number in the play and its thread's name.
pub(super) struct Step<'a> {
    log: &'a Log,
    seq: u64,
    thread: String,
}

impl Step<'_> {
    /// The make's shader modules are made, and its `VkPipeline` is made next: with the digests
    /// on, the line of a render state, whose `digest` is given, ends in `digest HEX`.
    pub(super) fn modules(&self, digest: Option<Hash256>) {
        let (seq, thread) = (self.seq, &self.thread);
        let line = format!("M2V-STATE modules {seq} {thread}");
        match digest.filter(|_| self.log.digests) {
            Some(digest) => {
                (self.log).line(&format!("{line} digest {}", crate::replay::hex(&digest)))
            }
            None => self.log.line(&line),
        }
    }
}

/// The constexpr samplers' indices of the pool's makes: one index for equal words, each new one
/// the highest of the sampler class not taken yet. A states-only play makes no sampler state, so
/// no other object holds one.
pub(super) struct Numbering(Mutex<(HashMap<[u64; 2], u32>, u32)>);

impl Numbering {
    fn new() -> Numbering {
        let end = rt::IndexClass::Sampler.range().end;
        Numbering(Mutex::new((HashMap::new(), end)))
    }

    /// The index of each of `samplers`, 0 for a texture read's.
    pub(super) fn ids(&self, samplers: &[rt::NumberedSampler]) -> Result<Vec<u32>, rt::Error> {
        let mut held = lock(&self.0);
        let (taken, next) = &mut *held;
        let mut ids = Vec::with_capacity(samplers.len());
        for sampler in samplers {
            let rt::NumberedSampler::Constexpr(words) = sampler else {
                ids.push(0);
                continue;
            };
            let index = match taken.get(words) {
                Some(&index) => index,
                None if *next > rt::IndexClass::Sampler.range().start => {
                    *next -= 1;
                    taken.insert(*words, *next);
                    *next
                }
                None => {
                    return Err(rt::Error::Refused(rt::Refusal::IndicesNotGranted {
                        class: rt::IndexClass::Sampler,
                        index: rt::IndexClass::Sampler.range().end,
                    }))
                }
            };
            ids.push(index);
        }
        Ok(ids)
    }
}

/// The lines, the tally and the order of a play's makes, shared by the play's thread and the
/// pool's.
struct Shared {
    log: Log,
    tally: Tally,
}

impl Shared {
    /// `make` of the state `what`, between its begin and its end lines, on `thread`.
    fn run<T>(
        &self,
        what: &str,
        thread: String,
        make: impl FnOnce(&Step) -> Result<T, rt::Error>,
    ) -> Result<T, rt::Error> {
        let seq = self.tally.begun.fetch_add(1, SeqCst);
        self.log
            .line(&format!("M2V-STATE begin {seq} {thread} {what}"));
        let began = Instant::now();
        let step = Step {
            log: &self.log,
            seq,
            thread,
        };
        let made = make(&step);
        let ms = began.elapsed().as_secs_f64() * 1e3;
        let (count, ended) = match &made {
            Ok(_) => (&self.tally.made, format!("made in {ms:.1} ms")),
            Err(error @ rt::Error::Vulkan { .. }) => {
                (&self.tally.failed, format!("failed {error}"))
            }
            Err(error) => (&self.tally.refused, format!("refused {error}")),
        };
        count.fetch_add(1, SeqCst);
        let thread = &step.thread;
        self.log
            .line(&format!("M2V-STATE end {seq} {thread} {ended}"));
        made
    }
}

/// The pool's threads and their queue; its drop waits for every make queued, then ends the
/// threads.
struct Pool {
    queue: Option<Sender<Job>>,
    threads: Vec<JoinHandle<()>>,
}

impl Pool {
    fn new(width: usize, shared: &Arc<Shared>) -> Pool {
        let (queue, jobs) = channel::<Job>();
        let jobs = Arc::new(Mutex::new(jobs));
        let numbering = Arc::new(Numbering::new());
        let threads = (0..width)
            .filter_map(|at| {
                let (jobs, shared) = (Arc::clone(&jobs), Arc::clone(shared));
                let numbering = Arc::clone(&numbering);
                let thread = format!("m2v-state-{at}");
                let run = {
                    let thread = thread.clone();
                    move || loop {
                        let next = lock(&jobs).recv();
                        let Ok((what, make)) = next else {
                            return;
                        };
                        let _ = shared.run(&what, thread.clone(), |step| make(&numbering, step));
                    }
                };
                std::thread::Builder::new().name(thread).spawn(run).ok()
            })
            .collect();
        Pool {
            queue: Some(queue),
            threads,
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.queue = None;
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

/// A play's state makes: their lines, their pool where it has one, and the keys it makes alone.
pub(super) struct States {
    shared: Arc<Shared>,
    pool: Mutex<Option<Pool>>,
    /// The pool's width, 0 for none.
    width: usize,
    keys: Option<HashSet<String>>,
}

impl Default for States {
    /// No lines, no pool and no key list: each state made on the play's thread, as a play
    /// without these makes them.
    fn default() -> States {
        States::new(None, 0, None, false)
    }
}

impl States {
    /// The makes' lines written to the standard output and to `log`, where a log is given, a pool
    /// is or `digests` is set; the makes run by `width` threads of the player's own, or on the
    /// play's thread with 0; with `keys`, only the render states of those contents made; with
    /// `digests`, each render state's modules line ending in its content digest.
    pub(super) fn new(
        log: Option<File>,
        width: usize,
        keys: Option<HashSet<String>>,
        digests: bool,
    ) -> States {
        let shared = Arc::new(Shared {
            log: Log {
                on: log.is_some() || width > 0 || digests,
                digests,
                file: Mutex::new(log),
            },
            tally: Tally::default(),
        });
        let pool = (width > 0).then(|| Pool::new(width, &shared));
        States {
            shared,
            pool: Mutex::new(pool),
            width,
            keys,
        }
    }

    /// Whether the state `what`, a render state of content `key` or a compute state (`None`), is
    /// made: every state without a key list, a render state of a listed key with one. One that
    /// is not is named on a `pass` line.
    pub(super) fn selects(&self, key: Option<&str>, what: &str) -> bool {
        let selected = match (&self.keys, key) {
            (None, _) => true,
            (Some(keys), Some(key)) => keys.contains(key),
            (Some(_), None) => false,
        };
        if !selected {
            self.shared.tally.passed.fetch_add(1, SeqCst);
            self.shared.log.line(&format!("M2V-STATE pass {what}"));
        }
        selected
    }

    /// Whether the makes go to the pool.
    pub(super) fn pooled(&self) -> bool {
        self.width > 0
    }

    /// Queues `make` of the state `what` for the pool's next free thread.
    pub(super) fn give(
        &self,
        what: String,
        make: impl FnOnce(&Numbering, &Step) -> Result<(), rt::Error> + Send + 'static,
    ) {
        if let Some(queue) = lock(&self.pool)
            .as_ref()
            .and_then(|pool| pool.queue.as_ref())
        {
            let _ = queue.send((what, Box::new(make)));
        }
    }

    /// A handle that makes a state on the play's thread between its lines ([`Inline::run`]).
    pub(super) fn inline(&self) -> Inline {
        Inline(Arc::clone(&self.shared))
    }

    /// Ends the pool once it made every state queued.
    pub(super) fn finish(&self) {
        drop(lock(&self.pool).take());
    }

    /// The report's `states` line where the makes were written or pooled: the makes begun and
    /// how they ended, the states passed over, and the threads that made them. It ends the pool
    /// first ([`finish`](Self::finish)).
    pub(super) fn line(&self) -> Option<String> {
        self.finish();
        if !self.shared.log.on {
            return None;
        }
        let t = &self.shared.tally;
        let threads = match self.width {
            0 => "the play's thread".to_owned(),
            width => format!("{width} threads"),
        };
        Some(format!(
            "states begun {} made {} refused {} failed {} passed over {} on {threads}",
            t.begun.load(SeqCst),
            t.made.load(SeqCst),
            t.refused.load(SeqCst),
            t.failed.load(SeqCst),
            t.passed.load(SeqCst),
        ))
    }
}

/// The play's thread's handle on the makes' lines.
pub(super) struct Inline(Arc<Shared>);

impl Inline {
    /// `make` of the state `what` on the play's thread, between its begin and end lines.
    pub(super) fn run<T>(
        &self,
        what: &str,
        make: impl FnOnce(&Step) -> Result<T, rt::Error>,
    ) -> Result<T, rt::Error> {
        self.0.run(what, "play".to_owned(), make)
    }
}

/// The line of the render state that `call` (object `object`, at stream position `at`) makes of
/// `descriptor`, and its content's key: the hex of the translator's key of the descriptor, which
/// names its record in the cache directory, then its colour attachments' pixel formats by index
/// (0 for none), its depth and stencil formats and its raster sample count.
pub(super) fn render_line(
    device: &rt::Device,
    descriptor: &rt::RenderPipelineDescriptor,
    object: u64,
    at: u64,
) -> (String, String) {
    let key = device
        .translator()
        .queried_render_descriptor_key(descriptor);
    let key = crate::replay::hex(key.digest());
    let colors: Vec<u32> = (descriptor.color_attachments.iter())
        .map(|color| {
            color
                .as_ref()
                .map_or(0, |color| u32::from(color.pixel_format))
        })
        .collect();
    let format = |format: Option<rt::PixelFormat>| format.map_or(0, u32::from);
    let line = format!(
        "render #{object} call {at} key {key} colors {colors:?} depth {} stencil {} samples {} \
         fragment {}",
        format(descriptor.depth_format),
        format(descriptor.stencil_format),
        descriptor.raster_sample_count,
        descriptor.fragment.is_some(),
    );
    (key, line)
}
