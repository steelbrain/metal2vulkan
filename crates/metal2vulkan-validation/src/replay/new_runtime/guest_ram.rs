//! The host's copies between buffers and the guest's RAM (`m2v.copyToGuestRam`,
//! `m2v.copyFromGuestRam`, `docs/REPLAY-STREAM.md` sections 7 and 10), played in the host's form
//! on the new runtime, the one-copy form: one buffer copy a call, in a command buffer of the
//! play's own on the queue of the call's command buffer, through a Shared scratch of the play's
//! own that the CPU fills or empties. A copy back is one copy of the whole words its runs span
//! into the scratch, committed just after its command buffer; a wait for that command buffer
//! waits for the copy too, as the guest sees its RAM written when its command buffer completes,
//! and the CPU then copies each run out of the scratch into memory of the play's own that nothing
//! reads. A copy in is the mirror: its pieces gathered on the CPU into the scratch laid out as the
//! bytes they span, and one copy into the buffer, committed just before its command buffer. This
//! work is the play's stand-in for the host's, counted apart from the stream's (`guest_ram_*`).

use super::super::args::{arg, refuse};
use super::places::Place;
use super::{failed, granule, NewRuntime};
use crate::replay::player::PlayError;
use crate::replay::stream::Call;
use metal2vulkan_runtime as rt;
use std::collections::{HashMap, VecDeque};
use std::ops::Range;

/// A copy back owed after its command buffer's commit: the bytes of `from` it copies, and each
/// run's offset from their start and length.
struct Download {
    from: rt::Buffer,
    span: Range<u64>,
    runs: Vec<(u64, u64)>,
}

/// A copy in owed before its command buffer's commit: its scratch, gathered, and the scratch's
/// place, its target, its copies (offset in the scratch, offset in the target, length), and the
/// pieces gathered.
struct Fill {
    scratch: rt::Buffer,
    place: Place,
    target: rt::Buffer,
    copies: Vec<(u64, u64, u64)>,
    pieces: u64,
}

/// A copy back committed whose runs the CPU has not copied out: its commit, its scratch and the
/// scratch's place, and its runs.
struct Downloading {
    commit: u64,
    scratch: rt::Buffer,
    place: Place,
    runs: Vec<(u64, u64)>,
}

/// What the play's stand-in did (`counted`).
#[derive(Default)]
struct Counts {
    downloads: u64,
    download_bytes: u64,
    runs_out: u64,
    run_bytes_out: u64,
    fills: u64,
    pieces_in: u64,
    fill_copies: u64,
    fill_bytes: u64,
    gpu_nanos: u64,
}

/// The copies owed, in flight, and made.
#[derive(Default)]
pub(super) struct GuestRam {
    /// The copies in and the copies back each command buffer not yet committed owes, in the
    /// stream's order.
    fills: HashMap<u64, Vec<Fill>>,
    downloads: HashMap<u64, Vec<Download>>,
    /// The copies back committed whose runs the CPU has not copied out, by the stream's command
    /// buffer they follow.
    downloading: HashMap<u64, Vec<Downloading>>,
    /// The command buffers of the play's own not yet seen complete, for their GPU time.
    timing: VecDeque<rt::CommandBuffer>,
    /// The memory of the play's own each run is copied into, which nothing reads.
    sink: Vec<u8>,
    counts: Counts,
}

impl GuestRam {
    /// The commits of the copies back that follow command buffer `command_buffer` and are not yet
    /// copied out.
    pub(super) fn downloads_of(&self, command_buffer: u64) -> Vec<u64> {
        let downloads = self.downloading.get(&command_buffer).into_iter().flatten();
        downloads.map(|download| download.commit).collect()
    }

    /// Adds the GPU time of the command buffers of the play's own that completed since the last
    /// call.
    fn timed(&mut self) {
        while let Some(front) = self.timing.front() {
            match front.status() {
                rt::CommandBufferStatus::Completed => {
                    let (start, end) = (front.gpu_start_time(), front.gpu_end_time());
                    self.counts.gpu_nanos += end.saturating_sub(start);
                }
                rt::CommandBufferStatus::Error => {}
                _ => return,
            }
            self.timing.pop_front();
        }
    }

    /// The executed counters.
    pub(super) fn counted(&mut self) -> [(&'static str, u64); 9] {
        self.timed();
        let c = &self.counts;
        [
            ("guest_ram_downloads", c.downloads),
            ("guest_ram_download_bytes", c.download_bytes),
            ("guest_ram_runs_out", c.runs_out),
            ("guest_ram_run_bytes_out", c.run_bytes_out),
            ("guest_ram_fills", c.fills),
            ("guest_ram_pieces_in", c.pieces_in),
            ("guest_ram_fill_copies", c.fill_copies),
            ("guest_ram_fill_bytes", c.fill_bytes),
            ("guest_ram_gpu_nanos", c.gpu_nanos),
        ]
    }

    /// The report's line.
    pub(super) fn line(&self) -> String {
        let c = &self.counts;
        format!(
            "guest RAM: copy-back played in the one-copy form: {} copies back ({} bytes), {} runs \
             ({} bytes) copied out by the CPU after their copy completed; {} copies in, {} pieces \
             gathered by the CPU, {} buffer copies ({} bytes): one a copy, and one for each run \
             of touching words into a buffer with no mapping; {} ns of GPU time",
            c.downloads,
            c.download_bytes,
            c.runs_out,
            c.run_bytes_out,
            c.fills,
            c.pieces_in,
            c.fill_copies,
            c.fill_bytes,
            c.gpu_nanos
        )
    }
}

impl NewRuntime {
    /// A copy back to the guest's RAM (`m2v.copyToGuestRam`) in blit encoder `call.receiver`: a
    /// surface's texels into the bytes first, in the encoder (`surfaces`), then one copy owed
    /// after the command buffer's commit (`guest_ram_downloads`) of the whole words the runs
    /// span within the buffer.
    pub(super) fn copy_back(&mut self, call: &Call) -> Result<(), PlayError> {
        let id = call.receiver;
        let object = Self::id(call, 0)?;
        let from = self.buffer_of(call, object)?;
        let runs = crate::replay::player::placed_runs(arg(call, 1)?);
        let runs = runs.ok_or_else(|| refuse(call, "argument 1 is no list of placed runs"))?;
        let words = from.length() / 4 * 4;
        let mut read: Vec<Range<u64>> = Vec::new();
        let mut kept = Vec::new();
        for (at, length) in runs {
            let end = at.checked_add(length);
            let end = end.ok_or_else(|| refuse(call, "a run past 2^64 bytes"))?;
            if end > from.length() {
                return Err(refuse(call, "a run past its buffer"));
            }
            let (start, stop) = (at & !3, end.next_multiple_of(4).min(words));
            if start < stop {
                read.push(start..stop);
                kept.push((at, length));
            }
        }
        let start = read.iter().map(|span| span.start).min();
        let stop = read.iter().map(|span| span.end).max();
        let (Some(start), Some(stop)) = (start, stop) else {
            return Ok(());
        };
        self.surface_copying_out(call, id, (object, &from), &read)?;
        let command_buffer = self.owner(call)?;
        let runs = kept.into_iter().map(|(at, length)| (at - start, length));
        let download = Download {
            from,
            span: start..stop,
            runs: runs.collect(),
        };
        let owed = self.guest_ram.downloads.entry(command_buffer).or_default();
        owed.push(download);
        Ok(())
    }

    /// A copy from the guest's RAM into a buffer (`m2v.copyFromGuestRam`, `…At`) in blit encoder
    /// `call.receiver`: the pieces gathered by the CPU into a Shared scratch of the play's own,
    /// laid out as the target's bytes they span, and one copy into the target owed before the
    /// command buffer's commit; then a surface's texels under those bytes, in the encoder
    /// (`surfaces`). A buffer copy takes whole 4-byte words at 4-byte offsets: the bytes of those
    /// words no piece carries (a piece's edges, the gaps between pieces) come from the target's
    /// mapping where it has one, as they stand. Into a target with none, a piece with such edges
    /// is skipped and counted (`skipped_guest_ram_writes`), and the copy is one for each run of
    /// touching words, so no gap is written.
    pub(super) fn guest_ram_copy(&mut self, call: &Call) -> Result<(), PlayError> {
        let id = call.receiver;
        let upload = crate::replay::player::upload(call)?;
        let upload = upload.ok_or_else(|| refuse(call, "no guest-RAM copy"))?;
        let target = self.buffer_of(call, upload.target)?;
        let mapped = target.contents();
        // Each piece's whole-word span in the target.
        let mut pieces = Vec::new();
        for &(at, entry) in &upload.pieces {
            let bytes = crate::replay::player::piece(&call.args, entry);
            let end = at + bytes.len() as u64;
            let (start, stop) = (at & !3, end.next_multiple_of(4));
            if stop > target.length() {
                return Err(refuse(call, "a piece past its buffer"));
            }
            if (start, stop) != (at, end) && mapped.is_none() {
                self.skips.guest_ram_writes += 1;
                continue;
            }
            if start < stop {
                pieces.push((at, bytes, start..stop));
            }
        }
        if pieces.is_empty() {
            return Ok(());
        }
        let written: Vec<Range<u64>> = pieces.iter().map(|(_, _, span)| span.clone()).collect();
        let mut spans = written.clone();
        spans.sort_by_key(|span| span.start);
        let mut runs: Vec<Range<u64>> = Vec::new();
        for span in spans {
            match runs.last_mut() {
                Some(last) if mapped.is_some() || span.start <= last.end => {
                    last.end = last.end.max(span.end);
                }
                _ => runs.push(span),
            }
        }
        let len: u64 = runs.iter().map(|run| run.end - run.start).sum();
        let completed = self.completed();
        // The play's own scratch: no recorded slack.
        let place = (self.places).take(rt::MemoryClass::Shared, len, granule(None), completed);
        let place = place.map_err(failed(call))?;
        let scratch = (self.device).new_buffer(place.block, place.range.start, len, None);
        let scratch = scratch.map_err(failed(call))?;
        // Each run's offset in the scratch, and its bytes as they stand where the target is
        // mapped.
        let mut copies = Vec::with_capacity(runs.len());
        let mut into = 0;
        for run in &runs {
            let size = run.end - run.start;
            if let Some(mapped) = mapped {
                // SAFETY: the run lies in the target's mapped bytes (checked above).
                let standing = unsafe {
                    std::slice::from_raw_parts(
                        mapped.as_ptr().add(run.start as usize),
                        size as usize,
                    )
                };
                super::write_into(&scratch, into, standing)?;
            }
            copies.push((into, run.start, size));
            into += size;
        }
        for (at, bytes, _) in &pieces {
            let copy = copies
                .iter()
                .find(|&&(_, to, size)| to <= *at && at - to < size);
            let Some(&(from, to, _)) = copy else {
                unreachable!("a piece outside every run");
            };
            super::write_into(&scratch, from + (at - to), bytes)?;
        }
        let command_buffer = self.owner(call)?;
        let owed = self.guest_ram.fills.entry(command_buffer).or_default();
        owed.push(Fill {
            scratch,
            place,
            target: target.clone(),
            copies,
            pieces: pieces.len() as u64,
        });
        self.surface_copied_in(call, id, (upload.target, &target), &written)
    }

    /// The command buffer of blit encoder `call.receiver`.
    fn owner(&self, call: &Call) -> Result<u64, PlayError> {
        let owner = self.owners.get(&call.receiver).copied();
        owner.ok_or_else(|| refuse(call, "a blit encoder of no command buffer"))
    }

    /// Before command buffer `call.receiver` commits: each copy in it owes, in a command buffer of
    /// the play's own committed now, ahead of it on its queue.
    pub(super) fn guest_ram_fill(&mut self, call: &Call) -> Result<(), PlayError> {
        let Some(fills) = self.guest_ram.fills.remove(&call.receiver) else {
            return Ok(());
        };
        let queue = self.queue_for(call)?;
        for fill in fills {
            let command_buffer = queue.command_buffer().map_err(failed(call))?;
            let mut blit = command_buffer
                .blit_command_encoder()
                .map_err(failed(call))?;
            for &(from, to, size) in &fill.copies {
                let copied =
                    blit.copy_from_buffer_to_buffer(&fill.scratch, from, &fill.target, to, size);
                copied.map_err(failed(call))?;
            }
            blit.end_encoding();
            self.guest_ram.timing.push_back(command_buffer.clone());
            self.commit(&queue, command_buffer, 0)
                .map_err(failed(call))?;
            self.places.give(fill.place, self.commits, Vec::new());
            let counts = &mut self.guest_ram.counts;
            counts.fills += 1;
            counts.pieces_in += fill.pieces;
            counts.fill_copies += fill.copies.len() as u64;
            counts.fill_bytes += fill.copies.iter().map(|&(_, _, size)| size).sum::<u64>();
        }
        Ok(())
    }

    /// After command buffer `call.receiver` commits: each copy back it owes, one copy into a
    /// Shared scratch of the play's own in a command buffer of the play's own committed now,
    /// behind it on its queue.
    pub(super) fn guest_ram_download(&mut self, call: &Call) -> Result<(), PlayError> {
        let Some(downloads) = self.guest_ram.downloads.remove(&call.receiver) else {
            return Ok(());
        };
        let queue = self.queue_for(call)?;
        for download in downloads {
            let len = download.span.end - download.span.start;
            let completed = self.completed();
            // The play's own scratch: no recorded slack.
            let place = (self.places).take(rt::MemoryClass::Shared, len, granule(None), completed);
            let place = place.map_err(failed(call))?;
            let scratch = (self.device).new_buffer(place.block, place.range.start, len, None);
            let scratch = scratch.map_err(failed(call))?;
            let command_buffer = queue.command_buffer().map_err(failed(call))?;
            let mut blit = command_buffer
                .blit_command_encoder()
                .map_err(failed(call))?;
            let start = download.span.start;
            let copied = blit.copy_from_buffer_to_buffer(&download.from, start, &scratch, 0, len);
            copied.map_err(failed(call))?;
            blit.end_encoding();
            self.guest_ram.timing.push_back(command_buffer.clone());
            self.commit(&queue, command_buffer, 0)
                .map_err(failed(call))?;
            let guest_ram = &mut self.guest_ram;
            guest_ram.counts.downloads += 1;
            guest_ram.counts.download_bytes += len;
            let downloading = guest_ram.downloading.entry(call.receiver).or_default();
            downloading.push(Downloading {
                commit: self.commits,
                scratch,
                place,
                runs: download.runs,
            });
        }
        Ok(())
    }

    /// Command buffer `command_buffer` released uncommitted: it runs nothing, so its copies are
    /// not made, and their scratch places are given back.
    pub(super) fn guest_ram_released(&mut self, command_buffer: u64) {
        self.guest_ram.downloads.remove(&command_buffer);
        let fills = self
            .guest_ram
            .fills
            .remove(&command_buffer)
            .unwrap_or_default();
        for fill in fills {
            self.places.give(fill.place, self.commits, Vec::new());
        }
    }

    /// After a wait for command buffer `command_buffer` and its copies back (`downloads_of`):
    /// each run copied out of their scratch by the CPU, and the scratch's place given back.
    pub(super) fn guest_ram_downloaded(&mut self, command_buffer: u64) {
        let Some(done) = self.guest_ram.downloading.remove(&command_buffer) else {
            return;
        };
        let guest_ram = &mut self.guest_ram;
        for download in done {
            if let Some(contents) = download.scratch.contents() {
                let len = download.scratch.length();
                for &(at, run) in &download.runs {
                    let end = (at + run).min(len);
                    let Some(size) = end.checked_sub(at).filter(|&size| size > 0) else {
                        continue;
                    };
                    let size = size as usize;
                    if guest_ram.sink.len() < size {
                        guest_ram.sink.resize(size, 0);
                    }
                    // SAFETY: `at..end` lies in the scratch's mapped bytes, and the copy into
                    // them has completed.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(contents.as_ptr().add(at as usize), size)
                    };
                    guest_ram.sink[..size].copy_from_slice(bytes);
                    guest_ram.counts.runs_out += 1;
                    guest_ram.counts.run_bytes_out += size as u64;
                }
            }
            self.places
                .give(download.place, download.commit, Vec::new());
        }
    }
}
