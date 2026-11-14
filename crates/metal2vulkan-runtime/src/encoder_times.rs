//! The encoder times: a default-off diagnostic, `METAL2VULKAN_ENCODER_TIMES`
//! (`metal2vulkan::env_vars`), that splits a device's GPU time by encoder, and gives each
//! encoder's to its kind and to the pipeline states it bound.
//!
//! - **Two timestamps an encoder**, each `vkCmdWriteTimestamp` at the bottom of the pipe, in the
//!   encoder's own recording; no batch, group or submission changes. The start is the first
//!   command of the `VkCommandBuffer` the encoder opens in (`EncoderCore::open`): its own, after
//!   its command buffer's start (`gpu_time.rs`) and the barrier from everything an encoder past the
//!   planned ones begins with, or the shared one of the compute encoders before it where it records
//!   on in theirs (`encoder_fold.rs`), at its place there. A render encoder's pass then takes a
//!   `VkCommandBuffer` of its own (`EncoderCore::hold_pass`), so the commands recorded before the
//!   pass go after the start. The end is the last command of the `VkCommandBuffer` the encoder ends
//!   in (`EncoderCore::end_with_replay`): after its render pass instance's end and its copies back,
//!   in a new one where its last went to a segment. A timestamp at the bottom of the pipe is
//!   written once every command before it completed, so an encoder's time runs from the moment
//!   the work before it completed (or the GPU reached its start, where it waited for nothing) to
//!   the moment its own completed: a queue's encoders share its busy time out with no time counted
//!   twice. Where dispatches of two encoders overlap (no barrier between them), the overlap counts
//!   for the one that ends first; within an encoder, its dispatches and draws share its time and
//!   are not timed apart, so a workload of one dispatch an encoder gives each kernel's time.
//! - **Where it runs.** On a device whose queue family writes timestamps, that resets query pools
//!   on the host (every device the runtime made, `gpu_time.rs`), and whose pools are not Metal
//!   counter sample buffers nor its timestamps waits for every queue's (MoltenVK): elsewhere the
//!   variable does nothing.
//! - **Pools** ([`EncoderTimes`]): the device's own, apart from the command buffers' GPU times',
//!   of slots of two queries ([`PLAIN`], `gpu_time::Pools`), 2048 encoders a pool, made as its
//!   encoders in flight need them until the device refuses one; an encoder that finds no slot free
//!   is not timed. A slot is reset on the host as it is taken, and given back once its command
//!   buffer's completion read it (or with the command buffer, where it never ran).
//! - **What an encoder ran nothing of reads nothing.** A query the GPU did not write reads
//!   unavailable, and its encoder counts as untimed: one encoded again (`command_buffer/reencode.rs`:
//!   its first recording is never submitted), one whose parts a command-predication record left out
//!   or its command buffer's failure ran nothing of, and one that ended inside a predicated loop's
//!   laying, which its check may submit again, so it writes no end. An encoder whose parts went in
//!   batches apart (a host step between them) counts the time between them too.
//! - **What it is given to** ([`Row`]): its kind and the label hashes of the pipeline states it
//!   bound, each once, in the order it first did, at most [`KEPT`]: the encoder trail's
//!   (`encoder_trail::label_hash`, the first 8 bytes of the SHA-256 of a state's label and its
//!   functions' names; no name and no shader text). An encoder of several states is one row of
//!   them all: its time is not split among them. Its dispatches and draws are the program's calls.
//! - **The lines** (id name [`ID_NAME`], through the device's message sink at `Info`, else on
//!   stderr). Once [`WINDOW`] passed since the first completion after the last summary, the next
//!   completion prints one, and the device's teardown the last: a line of the command buffers,
//!   encoders and GPU time since, then the [`TOP`] rows of the most GPU time, one line each: its
//!   encoders, dispatches and draws, total time and share, and the time per encoder at the median,
//!   the 90th percentile (each within 1/32 of the truth, [`Histogram`]) and the most. At most
//!   [`ROWS`] rows are kept a summary; the rest join one. Where the value is of the form `<N>ms`,
//!   each command buffer whose GPU time (its own timestamps', else its encoders') is `N` ms or more
//!   prints one line, [`LINES`] at most a device: each encoder's place, kind, states, dispatches and
//!   draws, offset from the command buffer's start and time.
//! - **Off, it costs nothing**: the variable is read once; an encoder that opens on a device with
//!   it off takes no slot and writes no timestamp, and a completion reads one flag.

use crate::device::lock;
use crate::encoder_trail::label_hash;
use crate::exit_guard::{self, Call};
use crate::gpu_time::{GpuClock, Pools, Slot, PLAIN};
use crate::pipeline::{ComputePipelineState, RenderPipelineState};
use crate::release;
use crate::used_resources::EncoderKind;
use ash::vk;
use std::collections::{BTreeMap, HashMap};
use std::ffi::OsStr;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The id name of the lines.
pub(crate) const ID_NAME: &str = "metal2vulkan-encoder-times";

/// How long after the first completion since the last summary the next completion prints one.
pub(crate) const WINDOW: Duration = Duration::from_secs(5);

/// The rows a summary prints.
const TOP: usize = 20;

/// The most rows a summary keeps apart; past them, the rest join one.
const ROWS: usize = 4096;

/// The most pipeline states an encoder names.
const KEPT: usize = 16;

/// The most lines a device prints of command buffers.
const LINES: u32 = 1024;

/// A threshold of no command buffer line.
const NO_LINES: u64 = u64::MAX;

/// What the process asked (the module docs): read once. `None` where off; else the threshold, in
/// nanoseconds, of a command buffer's line, [`NO_LINES`] for none.
fn asked() -> Option<u64> {
    static ASKED: OnceLock<Option<u64>> = OnceLock::new();
    *ASKED.get_or_init(|| metal2vulkan::env_vars::encoder_times().map(|value| threshold(&value)))
}

/// The threshold `value` sets, in nanoseconds: `N` milliseconds where it is `<N>ms`, else
/// [`NO_LINES`].
fn threshold(value: &OsStr) -> u64 {
    let read = value.to_str().and_then(|text| text.strip_suffix("ms"));
    let millis = read.and_then(|millis| millis.trim().parse::<f64>().ok());
    match millis {
        Some(millis) if millis.is_finite() && millis >= 0.0 => (millis * 1e6) as u64,
        _ => NO_LINES,
    }
}

/// A device's encoder times (the module docs).
pub(crate) struct EncoderTimes {
    /// The device can time its encoders: a device test's check.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    able: bool,
    on: AtomicBool,
    /// The threshold, in nanoseconds, of a command buffer's line; [`NO_LINES`] for none.
    over: AtomicU64,
    pools: Mutex<Pools>,
    window: Mutex<Window>,
    /// The command buffer lines printed.
    lines: AtomicU32,
}

impl EncoderTimes {
    /// The encoder times of a device that can time its encoders where `able` (the module docs):
    /// on where the process asked too.
    pub(crate) fn new(able: bool) -> EncoderTimes {
        let asked = asked().filter(|_| able);
        EncoderTimes {
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            able,
            on: AtomicBool::new(asked.is_some()),
            over: AtomicU64::new(asked.unwrap_or(NO_LINES)),
            pools: Mutex::new(Pools::new(None, PLAIN)),
            window: Mutex::new(Window::default()),
            lines: AtomicU32::new(0),
        }
    }

    /// Whether it times the device's encoders.
    pub(crate) fn on(&self) -> bool {
        self.on.load(Ordering::Relaxed)
    }

    /// A device test's: it times the device's encoders from now on, whatever the process's
    /// environment, with a line for each command buffer of `over` or more where given.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn time_encoders(&self, over: Option<Duration>) {
        assert!(self.able, "the device cannot time its encoders");
        let over = over.map_or(NO_LINES, |over| over.as_nanos() as u64);
        self.over.store(over, Ordering::Relaxed);
        self.on.store(true, Ordering::Relaxed);
    }

    /// The slots of the pools it made: a device test's view.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn slots(&self) -> usize {
        lock(&self.pools).slots()
    }

    /// Where it is on, an encoder opening in `raw`, just begun or recording on: a slot, reset on
    /// the host, and its start written into `raw` (the module docs). `None` where it is off, or
    /// every slot is held and no pool more can be made.
    ///
    /// # Safety
    ///
    /// `device` is the live device of these times, which enables `hostQueryReset`; `raw` is
    /// recording, outside any render pass instance, and only the caller records it now.
    pub(crate) unsafe fn open(&self, device: &ash::Device, raw: vk::CommandBuffer) -> Option<Open> {
        if !self.on() {
            return None;
        }
        let slot = lock(&self.pools).take(|count| {
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(count);
            // SAFETY: the caller's; `info` is valid.
            unsafe { device.create_query_pool(&info, None) }.ok()
        })?;
        // SAFETY: the caller's: the device enables `hostQueryReset`; the slot's queries are free
        // (its last holding was read or never ran, `gpu_time::Slot`), and only its holder resets
        // them. `raw` is recording outside a render pass instance, the queries reset before.
        unsafe {
            device.reset_query_pool(slot.queries(), slot.first(), PLAIN);
            exit_guard::called(Call::ResetQueryPool);
            let bottom = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
            device.cmd_write_timestamp(raw, bottom, slot.queries(), slot.first());
            exit_guard::called(Call::CmdWriteTimestamp);
        }
        Some(Open {
            slot,
            opened_in: raw,
            states: Vec::new(),
            more: 0,
            work: 0,
            ended: false,
        })
    }

    /// Release its pools, at the device's teardown, inside its section.
    ///
    /// # Safety
    ///
    /// `device` is the live device of these times, and none of its command buffers is pending.
    pub(crate) unsafe fn release(&self, device: &ash::Device) {
        for queries in lock(&self.pools).queries() {
            // SAFETY: the caller's.
            unsafe { release::object(device, queries) };
        }
    }

    /// The encoders `timed` of the command buffer of object ID `command_buffer`, its work
    /// completed, whose own timestamps read `span` where they were read: their times read through
    /// `device`, by `clock`, and tallied; their slots given back. The lines due: the command
    /// buffer's, where it is over the threshold, and the summary, where its window passed.
    ///
    /// # Safety
    ///
    /// `device` is the live device of these times; the work that writes the slots' queries
    /// completed (the completion turn's, inside its section).
    pub(crate) unsafe fn completed(
        &self,
        device: &ash::Device,
        clock: &GpuClock,
        (command_buffer, span): (u64, Option<(u64, u64)>),
        timed: Vec<Timed>,
    ) -> Vec<String> {
        let read: Vec<Read> = (timed.into_iter())
            .map(|timed| {
                // SAFETY: the caller's.
                let ticks = unsafe { timed.read(device) };
                Read {
                    encoder: timed.encoder,
                    kind: timed.kind,
                    states: timed.states(),
                    work: timed.work,
                    ticks,
                }
            })
            .collect();
        let mut lines = Vec::new();
        let over = self.over.load(Ordering::Relaxed);
        if over != NO_LINES {
            let gpu = command_buffer_time(clock, span, &read);
            if gpu >= over && self.lines.fetch_add(1, Ordering::Relaxed) < LINES {
                lines.push(command_buffer_line(clock, command_buffer, span, &read));
            }
        }
        let mut window = lock(&self.window);
        window.tally(clock, &read);
        if window.due() {
            lines.extend(window.summary());
            *window = Window::default();
        }
        lines
    }

    /// The summary of what it tallied since its last, where any: the device's teardown's, once
    /// no completion tallies more.
    pub(crate) fn last(&self) -> Vec<String> {
        let mut window = lock(&self.window);
        match window.started.is_some() {
            true => {
                let lines = window.summary();
                *window = Window::default();
                lines
            }
            false => Vec::new(),
        }
    }
}

/// What an open encoder notes for its times (the module docs): its slot, where it wrote its start,
/// and the states it bound with the dispatches and draws it called.
pub(crate) struct Open {
    slot: Slot,
    /// The `VkCommandBuffer` its start was written in.
    opened_in: vk::CommandBuffer,
    /// The label hash of each state it bound, once, in the order it first did, at most [`KEPT`].
    states: Vec<String>,
    /// The states it bound past [`KEPT`].
    more: usize,
    /// Its dispatches and draws.
    work: u64,
    /// Its end was written.
    ended: bool,
}

impl Open {
    /// The `VkCommandBuffer` its start was written in: the first of its own.
    pub(crate) fn opened_in(&self) -> vk::CommandBuffer {
        self.opened_in
    }

    /// It bound `state`, a compute pipeline state.
    pub(crate) fn compute(&mut self, state: &ComputePipelineState) {
        self.bound(label_hash(state.label(), &[state.function_name()]));
    }

    /// It bound `state`, a render pipeline state.
    pub(crate) fn render(&mut self, state: &RenderPipelineState) {
        let names = state.function_names();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        self.bound(label_hash(state.label(), &names));
    }

    /// It bound a state of label hash `hash`.
    fn bound(&mut self, hash: String) {
        if self.states.contains(&hash) {
            return;
        }
        match self.states.len() < KEPT {
            true => self.states.push(hash),
            false => self.more += 1,
        }
    }

    /// The program called a dispatch or a draw of it.
    pub(crate) fn worked(&mut self) {
        self.work += 1;
    }

    /// Write its end into `raw`, the `VkCommandBuffer` it ends in, after its last command.
    ///
    /// # Safety
    ///
    /// `device` is its device; `raw` is recording, outside any render pass instance, after every
    /// command of the encoder, and only the caller records it now; it is submitted once at most.
    pub(crate) unsafe fn end(&mut self, device: &ash::Device, raw: vk::CommandBuffer) {
        let bottom = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
        // SAFETY: the caller's; the end's query was reset on the host with the start's.
        unsafe {
            device.cmd_write_timestamp(raw, bottom, self.slot.queries(), self.slot.first() + 1)
        };
        exit_guard::called(Call::CmdWriteTimestamp);
        self.ended = true;
    }

    /// It ended as encoder `encoder` of its command buffer, of `kind`: what its command buffer
    /// holds until its completion reads it.
    pub(crate) fn ended(self, kind: EncoderKind, encoder: usize) -> Timed {
        Timed {
            encoder,
            kind,
            slot: self.slot,
            ended: self.ended,
            states: self.states,
            more: self.more,
            work: self.work,
        }
    }
}

/// An ended encoder's times, which its command buffer holds until its completion reads them.
#[derive(Debug)]
pub(crate) struct Timed {
    /// Its place among its command buffer's ended encoders.
    encoder: usize,
    kind: EncoderKind,
    slot: Slot,
    /// Its end was written.
    ended: bool,
    states: Vec<String>,
    more: usize,
    work: u64,
}

impl Timed {
    /// Its states, as a row names them.
    fn states(&self) -> String {
        let mut text = self.states.join(" ");
        if self.more > 0 {
            let _ = write!(text, " and {} more", self.more);
        }
        text
    }

    /// Its start and end ticks, where the GPU wrote both.
    ///
    /// # Safety
    ///
    /// `device` is its device; the work that writes its queries completed, or never ran.
    unsafe fn read(&self, device: &ash::Device) -> Option<(u64, u64)> {
        if !self.ended {
            return None;
        }
        let mut result = [[0u64; 2]; 2];
        // SAFETY: the caller's; the queries were reset as the slot was taken; the result is
        // 64-bit with availability, two words a query.
        let read = unsafe {
            device.get_query_pool_results(
                self.slot.queries(),
                self.slot.first(),
                &mut result,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
            )
        };
        exit_guard::called(Call::GetQueryPoolResults);
        match (read, result) {
            (Ok(()) | Err(vk::Result::NOT_READY), [[start, ready], [end, done]])
                if ready != 0 && done != 0 =>
            {
                Some((start, end))
            }
            _ => None,
        }
    }
}

/// An encoder's times as read.
struct Read {
    encoder: usize,
    kind: EncoderKind,
    states: String,
    work: u64,
    /// Its start and end ticks, where read.
    ticks: Option<(u64, u64)>,
}

/// The GPU time, in nanoseconds, of a command buffer whose timestamps read `span` where read, and
/// whose encoders read `read`: its span's, else from its encoders' first start to their last end.
fn command_buffer_time(clock: &GpuClock, span: Option<(u64, u64)>, read: &[Read]) -> u64 {
    if let Some((start, end)) = span {
        return clock.duration(start, end);
    }
    let ticks: Vec<(u64, u64)> = read.iter().filter_map(|read| read.ticks).collect();
    let Some(&(first, _)) = ticks.first() else {
        return 0;
    };
    // Each end after the first start by its duration from it: the ticks may wrap.
    (ticks.iter())
        .map(|&(_, end)| clock.duration(first, end))
        .max()
        .unwrap_or(0)
}

/// The line of the command buffer of object ID `command_buffer`, whose timestamps read `span`
/// where read, and whose encoders read `read` (the module docs).
fn command_buffer_line(
    clock: &GpuClock,
    command_buffer: u64,
    span: Option<(u64, u64)>,
    read: &[Read],
) -> String {
    let origin = span.map(|(start, _)| start).or_else(|| {
        read.iter()
            .find_map(|read| read.ticks.map(|(start, _)| start))
    });
    let timed: u64 = (read.iter())
        .filter_map(|read| read.ticks)
        .map(|(start, end)| clock.duration(start, end))
        .sum();
    let mut text = format!(
        "metal2vulkan: encoder times: command buffer {command_buffer}: GPU time {} over {} \
         encoder(s), {} of them in its encoders:",
        ms(command_buffer_time(clock, span, read)),
        read.len(),
        ms(timed),
    );
    for (at, encoder) in read.iter().enumerate() {
        let sep = if at == 0 { " " } else { "; " };
        let _ = write!(
            text,
            "{sep}{} {:?} [{}] {} dispatch(es)/draw(s)",
            encoder.encoder, encoder.kind, encoder.states, encoder.work
        );
        match (encoder.ticks, origin) {
            (Some((start, end)), Some(origin)) => {
                let _ = write!(
                    text,
                    " at +{} for {}",
                    ms(clock.duration(origin, start)),
                    ms(clock.duration(start, end))
                );
            }
            _ => text.push_str(" untimed"),
        }
    }
    text
}

/// `ns` nanoseconds, in milliseconds as a line prints them.
fn ms(ns: u64) -> String {
    format!("{:.3} ms", ns as f64 / 1e6)
}

/// What a summary tallies (the module docs).
#[derive(Default)]
struct Window {
    /// The first completion it tallied.
    started: Option<Instant>,
    command_buffers: u64,
    /// The encoders it timed, and those it could not.
    timed: u64,
    untimed: u64,
    /// Their GPU time, in nanoseconds.
    total: u64,
    /// By kind and states, at most [`ROWS`].
    rows: HashMap<(EncoderKind, String), Row>,
    /// The rows past [`ROWS`], and how many.
    others: Row,
    other_rows: u64,
}

impl Window {
    /// Tally the encoders `read` of one command buffer, by `clock`.
    fn tally(&mut self, clock: &GpuClock, read: &[Read]) {
        self.started.get_or_insert_with(Instant::now);
        self.command_buffers += 1;
        for encoder in read {
            let Some((start, end)) = encoder.ticks else {
                self.untimed += 1;
                continue;
            };
            let ns = clock.duration(start, end);
            self.timed += 1;
            self.total = self.total.saturating_add(ns);
            let key = (encoder.kind, encoder.states.clone());
            let row = match (self.rows.len() < ROWS, self.rows.contains_key(&key)) {
                (false, false) => {
                    self.other_rows += 1;
                    &mut self.others
                }
                _ => self.rows.entry(key).or_default(),
            };
            row.add(ns, encoder.work);
        }
    }

    /// Whether its summary is due: [`WINDOW`] passed since its first completion.
    fn due(&self) -> bool {
        self.started
            .is_some_and(|started| started.elapsed() >= WINDOW)
    }

    /// Its rows, the most GPU time first.
    fn ranked(&self) -> Vec<(&(EncoderKind, String), &Row)> {
        let mut rows: Vec<_> = self.rows.iter().collect();
        rows.sort_by(|(a_key, a), (b_key, b)| {
            b.total.cmp(&a.total).then_with(|| a_key.1.cmp(&b_key.1))
        });
        rows
    }

    /// Its summary's lines (the module docs).
    fn summary(&self) -> Vec<String> {
        let seconds = self
            .started
            .map_or(0.0, |started| started.elapsed().as_secs_f64());
        let rows = self.ranked();
        let shown = rows.len().min(TOP);
        let mut lines = vec![format!(
            "metal2vulkan: encoder times: over {seconds:.1} s: {} command buffer(s), {} encoder(s) \
             timed and {} not, {} of GPU time in them; the {shown} of the most:",
            self.command_buffers,
            self.timed,
            self.untimed,
            ms(self.total)
        )];
        let share = |ns: u64| match self.total {
            0 => 0.0,
            total => ns as f64 * 100.0 / total as f64,
        };
        for (place, ((kind, states), row)) in rows.iter().take(TOP).enumerate() {
            lines.push(format!(
                "metal2vulkan: encoder times: #{} {kind:?} [{states}]: {} encoder(s), {} \
                 dispatch(es)/draw(s), {} ({:.1}%), per encoder p50 {} p90 {} max {}",
                place + 1,
                row.encoders,
                row.work,
                ms(row.total),
                share(row.total),
                ms(row.histogram.percentile(0.5)),
                ms(row.histogram.percentile(0.9)),
                ms(row.most),
            ));
        }
        let mut rest = rows
            .iter()
            .skip(TOP)
            .fold(Row::default(), |mut rest, (_, row)| {
                rest.join(row);
                rest
            });
        rest.join(&self.others);
        if rest.encoders > 0 {
            let count = rows.len().saturating_sub(TOP) as u64 + self.other_rows;
            lines.push(format!(
                "metal2vulkan: encoder times: the rest, {count} row(s): {} encoder(s), {} \
                 dispatch(es)/draw(s), {} ({:.1}%)",
                rest.encoders,
                rest.work,
                ms(rest.total),
                share(rest.total)
            ));
        }
        lines
    }
}

/// One row of a summary: the encoders of one kind and states (the module docs).
#[derive(Default)]
struct Row {
    encoders: u64,
    work: u64,
    total: u64,
    most: u64,
    histogram: Histogram,
}

impl Row {
    /// An encoder of `ns` nanoseconds and `work` dispatches and draws.
    fn add(&mut self, ns: u64, work: u64) {
        self.encoders += 1;
        self.work += work;
        self.total = self.total.saturating_add(ns);
        self.most = self.most.max(ns);
        self.histogram.add(ns);
    }

    /// `other`'s encoders joined.
    fn join(&mut self, other: &Row) {
        self.encoders += other.encoders;
        self.work += other.work;
        self.total = self.total.saturating_add(other.total);
        self.most = self.most.max(other.most);
        for (&bucket, &count) in &other.histogram.buckets {
            *self.histogram.buckets.entry(bucket).or_default() += count;
        }
    }
}

/// The buckets of an octave: a time's bucket holds the times of its top 5 bits, so the middle of
/// a bucket stands within 1/32 of any time in it.
const SUB: u64 = 16;

/// Encoders' times by bucket ([`bucket`]), each counted: at most 976 buckets.
#[derive(Default)]
struct Histogram {
    buckets: BTreeMap<u16, u64>,
}

impl Histogram {
    fn add(&mut self, ns: u64) {
        *self.buckets.entry(bucket(ns)).or_default() += 1;
    }

    /// The time at `share` (0 to 1) of the counted ones, in order: the middle of its bucket.
    fn percentile(&self, share: f64) -> u64 {
        let count: u64 = self.buckets.values().sum();
        let rank = ((share * count as f64).ceil() as u64).clamp(1, count.max(1));
        let mut seen = 0;
        for (&bucket, &counted) in &self.buckets {
            seen += counted;
            if seen >= rank {
                return middle(bucket);
            }
        }
        0
    }
}

/// The bucket of `ns`: itself below [`SUB`], else [`SUB`] buckets an octave by the bits below its
/// top one.
fn bucket(ns: u64) -> u16 {
    if ns < SUB {
        return ns as u16;
    }
    let octave = u64::from(63 - ns.leading_zeros());
    let shift = octave - 4;
    let mantissa = (ns >> shift) - SUB;
    ((octave - 3) * SUB + mantissa) as u16
}

/// The middle of `bucket`'s times.
fn middle(bucket: u16) -> u64 {
    let bucket = u64::from(bucket);
    if bucket < SUB {
        return bucket;
    }
    let shift = bucket / SUB - 1;
    let low = (bucket % SUB + SUB) << shift;
    low + (1 << shift) / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bucket's middle stands within 1/32 of every time in it, and buckets rise with times.
    #[test]
    fn a_bucket_holds_its_times_within_a_thirty_second() {
        let mut last = 0;
        for ns in (0..1u64 << 12).chain([1 << 20, 123_456_789, u64::MAX / 3, u64::MAX]) {
            let at = bucket(ns);
            assert!(at >= last, "{ns}");
            last = at;
            let mid = middle(at);
            assert!(mid.abs_diff(ns) <= ns / 32, "{ns}: {mid}");
        }
        assert!(bucket(u64::MAX) < 976);
        let mut histogram = Histogram::default();
        for ns in 1..=100u64 {
            histogram.add(ns * 1000);
        }
        let p50 = histogram.percentile(0.5);
        let p90 = histogram.percentile(0.9);
        assert!(p50.abs_diff(50_000) <= 50_000 / 32, "{p50}");
        assert!(p90.abs_diff(90_000) <= 90_000 / 32, "{p90}");
    }

    /// The threshold of a command buffer's line: `<N>ms` alone.
    #[test]
    fn a_value_of_milliseconds_sets_the_command_buffer_lines() {
        assert_eq!(threshold(OsStr::new("20ms")), 20_000_000);
        assert_eq!(threshold(OsStr::new("0.5ms")), 500_000);
        assert_eq!(threshold(OsStr::new("0ms")), 0);
        assert_eq!(threshold(OsStr::new("1")), NO_LINES);
        assert_eq!(threshold(OsStr::new("on")), NO_LINES);
        assert_eq!(threshold(OsStr::new("-3ms")), NO_LINES);
    }

    /// A summary ranks its rows by GPU time and names each by its kind and states, an encoder of
    /// several states one row of them all; one that read nothing counts as untimed; the command
    /// buffer's line gives each encoder's offset and time.
    #[test]
    fn a_summary_ranks_rows_and_a_line_places_encoders() {
        let clock = GpuClock::new(64, 1.0, None).unwrap();
        let read = |encoder, kind, states: &str, work, ticks| Read {
            encoder,
            kind,
            states: states.to_owned(),
            work,
            ticks,
        };
        let encoders = [
            read(0, EncoderKind::Compute, "aa", 1, Some((100, 300))),
            read(1, EncoderKind::Compute, "bb", 1, Some((300, 5_300))),
            read(2, EncoderKind::Render, "aa cc", 7, Some((5_300, 6_300))),
            read(3, EncoderKind::Blit, "", 0, None),
            read(4, EncoderKind::Compute, "aa", 1, Some((6_300, 6_700))),
        ];
        let mut window = Window::default();
        window.tally(&clock, &encoders);
        assert_eq!((window.timed, window.untimed, window.total), (4, 1, 6_600));
        let lines = window.summary();
        assert_eq!(lines.len(), 4, "{lines:?}");
        assert!(
            lines[0].contains("1 command buffer(s), 4 encoder(s) timed and 1 not"),
            "{}",
            lines[0]
        );
        assert!(
            lines[1].contains("#1 Compute [bb]: 1 encoder(s), 1 dispatch(es)/draw(s), 0.005 ms"),
            "{}",
            lines[1]
        );
        assert!(
            lines[2].contains("#2 Render [aa cc]: 1 encoder(s), 7"),
            "{}",
            lines[2]
        );
        assert!(
            lines[3].contains("#3 Compute [aa]: 2 encoder(s), 2 dispatch(es)/draw(s)"),
            "{}",
            lines[3]
        );
        let line = command_buffer_line(&clock, 9, Some((50, 6_850)), &encoders);
        assert!(line.starts_with("metal2vulkan: encoder times: command buffer 9: GPU time 0.007 ms over 5 encoder(s)"), "{line}");
        assert!(
            line.contains("; 1 Compute [bb] 1 dispatch(es)/draw(s) at +0.000 ms for 0.005 ms;"),
            "{line}"
        );
        assert!(
            line.contains("; 3 Blit [] 0 dispatch(es)/draw(s) untimed;"),
            "{line}"
        );
    }
}
