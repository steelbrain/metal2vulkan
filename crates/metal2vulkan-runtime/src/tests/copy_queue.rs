//! A queue made for copies (`Device::new_command_queue_for_copies`, `queue.rs`) and a blit
//! encoder's runs of buffer copies (`copy_run.rs`).
//!
//! - **The copy queue.** Where the device gives one (a copy engine of two queues or more: NVIDIA's;
//!   RADV's under `RADV_PERFTEST=transfer_queue`), a command buffer of another queue fills a
//!   `Private` buffer behind 1 GiB of fills, and a command buffer of the copy queue committed right
//!   after it, with no wait between, copies 64 rows of it, side by side in rising offsets, into a
//!   no-copy buffer over host pages: the copies read what the fill wrote (the tracker's wait across
//!   the two `VkQueue`s), land where they were asked and nowhere else, run on the copy engine's
//!   `VkQueue` (`WorkCounts::copy_queue_batches`) as one `vkCmdCopyBuffer`
//!   (`WorkCounts::buffer_copies_merged`), and their command buffer's completion sync file signals.
//!   Its command buffers refuse a compute encoder and every blit command but buffer copies, fills
//!   and fences (`Problem::CopyQueue`), and the validation layer reports nothing. Where the device
//!   gives none (lavapipe's, RADV's by default, MoltenVK's), the answer is `None`, printed
//!   (`COPY-QUEUE-NONE`), and the device has no copy engine's `VkQueue` to give.
//! - **Runs of copies, on any queue.** 64 such row copies of one blit encoder are one
//!   `vkCmdCopyBuffer` (63 merged); copies that meet (a copy of bytes a copy before wrote, two
//!   copies into one range) or go between other buffers take a barrier or a command of their own
//!   and merge nothing; every byte lands as the copies in their order leave it.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`).

use super::blit_holds::{busy, over_host, Host, Scratch, WAIT};
use super::layer_claim::Watched;
use crate::{
    CommandBufferStatus, CopyQueueProblem, Error, Invalid, Problem, ResourceOptions, ResourceRef,
};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;

/// The rows each copy leg copies.
const ROWS: u64 = 64;

/// The bytes of a row.
const ROW: u64 = 256;

/// Where the rows lie in the source: rows of a picture `SOURCE_PITCH` bytes apart, from byte
/// `SOURCE_AT` of each.
const SOURCE_PITCH: u64 = 1024;
const SOURCE_AT: u64 = 128;

/// Where they go in the destination: `DESTINATION_PITCH` bytes apart, from byte
/// `DESTINATION_AT` of each.
const DESTINATION_PITCH: u64 = 512;
const DESTINATION_AT: u64 = 64;

/// The byte the fill writes into the source.
const FILLED: u8 = 0x5a;

/// The word the destination's host pages hold before the copies.
const BEFORE: u32 = 0xeeee_eeee;

/// The row copies: each row's source and destination offsets, both rising.
fn rows() -> impl Iterator<Item = (u64, u64)> {
    (0..ROWS).map(|row| {
        (
            row * SOURCE_PITCH + SOURCE_AT,
            row * DESTINATION_PITCH + DESTINATION_AT,
        )
    })
}

/// Whether byte `at` of the destination is one a row copy wrote.
fn copied(at: u64) -> bool {
    let row = at / DESTINATION_PITCH;
    let within = at % DESTINATION_PITCH;
    row < ROWS && (DESTINATION_AT..DESTINATION_AT + ROW).contains(&within)
}

/// The invalid call's problem.
fn problem(result: Result<impl std::fmt::Debug, Error>) -> Problem {
    match result {
        Err(Error::Invalid(Invalid { problem, .. })) => problem,
        other => panic!("not an invalid call: {other:?}"),
    }
}

/// The copy queue's leg (the module docs).
#[test]
fn copies_on_the_copy_engine_follow_another_queue_s_writes() {
    const LEG: &str = "COPY-QUEUE";
    let host = Host::new();
    host.fill(BEFORE);
    let watched = Watched::default();
    let scratch = Scratch::new("copy-queue");
    let device = super::pending_rebind::device(&watched, &scratch.0);
    let name = device.name().to_owned();
    let Some(copies) = device.new_command_queue_for_copies().unwrap() else {
        // `None` for want of a copy engine's queue (lavapipe's one queue, RADV's families by
        // default), or of host query resets.
        let shared = device.shared();
        let slot = shared.copy_slot();
        let resets = shared.timing.resets_on_host();
        eprintln!(
            "COPY-QUEUE-NONE on {name}: the device gives no queue made for copies (copy engine's \
             VkQueue {slot:?}, host query resets {resets})"
        );
        assert!(
            slot.is_none() || !resets,
            "{LEG} on {name}: a copy engine refused"
        );
        watched.layer_reported_nothing(LEG);
        return;
    };
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(destination) = over_host(&device, &host, &runs) else {
        return;
    };
    let queue = device.new_command_queue().unwrap();
    let source = device
        .new_buffer(ROWS * SOURCE_PITCH, ResourceOptions::private())
        .unwrap();
    let delay = device
        .new_buffer(super::blit_holds::FILLED, ResourceOptions::private())
        .unwrap();
    // Refusals, which leave the command buffer as it was.
    let refused = copies.command_buffer().unwrap();
    assert_eq!(
        problem(refused.compute_command_encoder()),
        Problem::CopyQueue(CopyQueueProblem::Encoder)
    );
    let mut blit = refused.blit_command_encoder().unwrap();
    assert_eq!(
        problem(blit.synchronize_resource(ResourceRef::Buffer(&source))),
        Problem::CopyQueue(CopyQueueProblem::Command)
    );
    blit.end_encoding().unwrap();
    drop(refused);
    // The source filled on another queue, behind 1 GiB of fills; the copies committed right after
    // it, with no wait.
    let before = device.work_counts();
    let filling = busy(&queue, &delay);
    let fill = queue.command_buffer().unwrap();
    let mut blit = fill.blit_command_encoder().unwrap();
    blit.fill_buffer(&source, 0..ROWS * SOURCE_PITCH, FILLED)
        .unwrap();
    blit.end_encoding().unwrap();
    fill.commit().unwrap();
    let copying = copies.command_buffer().unwrap();
    let mut blit = copying.blit_command_encoder().unwrap();
    for (from, to) in rows() {
        blit.copy_from_buffer_to_buffer(&source, from, &destination, to, ROW)
            .unwrap();
    }
    blit.end_encoding().unwrap();
    copying.add_completion_sync_file().unwrap();
    let committed = Instant::now();
    copying.commit().unwrap();
    let file = match copying.export_completion_sync_file() {
        Ok(file) => Some(file),
        Err(Error::Invalid(Invalid {
            problem: Problem::NotExportable,
            ..
        })) => {
            eprintln!("{LEG} on {name}: the device exports no sync file");
            None
        }
        Err(error) => panic!("{LEG} on {name}: export_completion_sync_file: {error:?}"),
    };
    assert!(
        copying.wait_until_completed_for(WAIT).is_ok(),
        "{LEG}: the copies"
    );
    assert_eq!(copying.status(), CommandBufferStatus::Completed);
    let waited = committed.elapsed();
    assert!(
        filling.wait_until_completed_for(WAIT).is_ok(),
        "{LEG}: the fills"
    );
    assert!(
        fill.wait_until_completed_for(WAIT).is_ok(),
        "{LEG}: the fill"
    );
    let after = device.work_counts();
    let signalled = match &file {
        Some(Some(file)) => super::sync_file::signaled(file),
        // `None`: it ended before the export looked, a sync file already signalled.
        Some(None) | None => true,
    };
    assert!(signalled, "{LEG} on {name}: the sync file is not signalled");
    // Every row the fill's bytes, every other byte as the CPU left it: the command buffers that
    // used the pages completed.
    let bytes: &[u8] = bytemuck_bytes(host.words());
    let wrong: Vec<u64> = (0..bytes.len() as u64)
        .filter(|&at| {
            let want = match copied(at) {
                true => FILLED,
                false => BEFORE.to_ne_bytes()[(at % 4) as usize],
            };
            bytes[at as usize] != want
        })
        .take(8)
        .collect();
    assert!(
        wrong.is_empty(),
        "{LEG} on {name}: bytes {wrong:?} of the destination are wrong"
    );
    let batches = after.copy_queue_batches - before.copy_queue_batches;
    let merged = after.buffer_copies_merged - before.buffer_copies_merged;
    eprintln!(
        "{LEG} on {name}: {batches} batch(es) on the copy engine's VkQueue, {merged} of {ROWS} \
         copies merged, completed {:.2} ms after the commit",
        waited.as_secs_f64() * 1e3
    );
    assert!(
        batches >= 1,
        "{LEG} on {name}: no batch ran on the copy queue's VkQueue"
    );
    assert_eq!(merged, ROWS - 1, "{LEG} on {name}: the row copies merged");
    drop(destination);
    watched.layer_reported_nothing(LEG);
}

/// The runs' leg, on any queue (the module docs).
#[test]
fn row_copies_between_two_buffers_are_one_copy_command() {
    const LEG: &str = "COPY-RUNS";
    let watched = Watched::default();
    let scratch = Scratch::new("copy-runs");
    let device = super::pending_rebind::device(&watched, &scratch.0);
    let name = device.name().to_owned();
    let queue = device.new_command_queue().unwrap();
    let length = ROWS * SOURCE_PITCH;
    let shared = ResourceOptions::default();
    let [source, destination, third] = [(); 3].map(|_| device.new_buffer(length, shared).unwrap());
    // The source's bytes: each byte its offset's low byte.
    write(&source, |at| at as u8);
    write(&destination, |_| 0xee);
    write(&third, |_| 0x11);
    let merged = |run: &dyn Fn(&mut crate::BlitCommandEncoder)| -> u64 {
        let before = device.work_counts().buffer_copies_merged;
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        run(&mut blit);
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        assert!(
            command_buffer.wait_until_completed_for(WAIT).is_ok(),
            "{LEG}"
        );
        device.work_counts().buffer_copies_merged - before
    };
    // The rows in rising offsets: one command.
    let rows_merged = merged(&|blit| {
        for (from, to) in rows() {
            blit.copy_from_buffer_to_buffer(&source, from, &destination, to, ROW)
                .unwrap();
        }
    });
    assert_eq!(rows_merged, ROWS - 1, "{LEG} on {name}: the rows");
    let held = read(&destination);
    for at in 0..length {
        let want = match copied(at) {
            true => {
                let (row, within) = (at / DESTINATION_PITCH, at % DESTINATION_PITCH);
                (row * SOURCE_PITCH + SOURCE_AT + within - DESTINATION_AT) as u8
            }
            false => 0xee,
        };
        assert_eq!(held[at as usize], want, "{LEG} on {name}: byte {at}");
    }
    // A copy of what the copy before wrote, two copies into one range, and a copy between other
    // buffers between two of the pair's: none merges, and the later writes win.
    let none = merged(&|blit| {
        blit.copy_from_buffer_to_buffer(&source, 0, &destination, 0, 64)
            .unwrap();
        blit.copy_from_buffer_to_buffer(&destination, 0, &third, 0, 64)
            .unwrap();
        blit.copy_from_buffer_to_buffer(&source, 100, &third, 0, 64)
            .unwrap();
        blit.copy_from_buffer_to_buffer(&source, 200, &third, 0, 64)
            .unwrap();
        blit.copy_from_buffer_to_buffer(&third, 0, &destination, 1000, 32)
            .unwrap();
        blit.copy_from_buffer_to_buffer(&source, 300, &third, 500, 64)
            .unwrap();
    });
    assert_eq!(none, 0, "{LEG} on {name}: copies that meet merged");
    let third_held = read(&third);
    let destination_held = read(&destination);
    for at in 0..64u64 {
        assert_eq!(
            third_held[at as usize],
            (200 + at) as u8,
            "{LEG}: the later write"
        );
    }
    for at in 0..32u64 {
        assert_eq!(
            destination_held[1000 + at as usize],
            (200 + at) as u8,
            "{LEG}: in order"
        );
    }
    for at in 0..64u64 {
        assert_eq!(
            third_held[500 + at as usize],
            (300 + at) as u8,
            "{LEG}: the last"
        );
    }
    eprintln!("{LEG} on {name}: {rows_merged} of {ROWS} row copies merged; {none} of the rest");
    watched.layer_reported_nothing(LEG);
}

/// The bytes of `words`.
fn bytemuck_bytes(words: &[u32]) -> &[u8] {
    // SAFETY: the words' own bytes, as many as they hold, which outlive the slice.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), std::mem::size_of_val(words)) }
}

/// Set each byte of `buffer`, a `Shared` one no command uses now, to `byte(at)`.
fn write(buffer: &crate::Buffer, byte: impl Fn(u64) -> u8) {
    let contents = buffer.contents().unwrap();
    for at in 0..buffer.length() {
        // SAFETY: within the buffer's mapping, which no command uses now.
        unsafe { contents.as_ptr().add(at as usize).write(byte(at)) };
    }
}

/// The bytes of `buffer`, a `Shared` one no command uses now.
fn read(buffer: &crate::Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's mapping, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}
