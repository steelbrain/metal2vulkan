//! What a submission costs with a guest's buffers over one shared dma-buf (a guest RAM region's)
//! live, by how their source answers requests longer than the device's sparse granule whose pages
//! lie in the region at page offsets off it (`DmaBufRequest`): a source that keeps each piece's
//! offset at a multiple of the request's `alignment` (the granule there: RADV's 64 KiB on Mesa
//! 26.2.4) answers each such request with a dma-buf of its own, as every source had to before
//! `one_chunk_alignment`; one that keeps a lone piece's at a multiple of `one_chunk_alignment`
//! answers each with one piece of the region. RADV lists every imported memory in every kernel
//! submission, so the first lists one import per buffer and the second one for the region.
//!
//! The arms, interleaved over [`ROUNDS`] rounds: **none** (no buffer over the region), **own** and
//! **piece**, each with [`BUFFERS`] buffers of [`BYTES`] live, a page past each other in the
//! region's file, their Metal addresses handed out so that the residency keeps their imports
//! (`host_residency.rs`). Each commits [`COMMITS`] command buffers of one dispatch, each waited
//! for. Printed (`ONE-PIECE`), per arm: the dma-buf memories live and the imports made a round
//! (`WorkCounts::dma_buf_memories_live`, `dma_buf_imports`, `dma_buf_pieces_joined`), the time
//! the imports and joins took (`dma_buf_import_nanos`, `dma_buf_join_nanos`), and the submission
//! calls' median and 90th percentile (`WorkCounts::batch_submit_nanos` over
//! `queue_submissions`).
//!
//! Ignored: run it in release under `--cfg m2v_gpu` where `/dev/udmabuf` opens,
//! `cargo test --release -p metal2vulkan-runtime --lib one_piece_bench -- --ignored --nocapture`.

use super::blit_holds::{Scratch, WAIT};
use super::compute_parked::Kernel;
use super::dma_buf_shared_files::no_copy;
use super::host_residency::{host_page, udmabuf, udmabuf_openable, Ram};
use crate::{Buffer, DmaBufChunk, DmaBufRequest, DmaBufSource, WorkCounts};
use std::os::fd::OwnedFd;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The buffers over the region an arm keeps live.
const BUFFERS: usize = 128;
/// The bytes of each, past RADV's 64 KiB granule.
const BYTES: usize = 256 << 10;
/// The command buffers an arm commits a round.
const COMMITS: usize = 200;
/// The rounds.
const ROUNDS: usize = 3;
/// How long the buffers' releases may take to settle.
const SETTLE: Duration = Duration::from_secs(10);

/// How an arm's source answers.
#[derive(Clone, Copy)]
enum Answer {
    /// A piece only where its offset is a multiple of the request's `alignment`, else a dma-buf
    /// of the request's own pages.
    Own,
    /// A piece wherever its offset is a multiple of the request's `one_chunk_alignment`.
    Piece,
}

/// A source over `ram`'s file, of which `region` is a dma-buf from its first byte (the view maps
/// the file from its first byte), answering as `answer` says.
fn source(ram: &Ram, region: &Arc<OwnedFd>, answer: Answer) -> DmaBufSource {
    let base = ram.pointer(0).as_ptr() as u64;
    let region = Arc::clone(region);
    let file = ram.file().try_clone().unwrap();
    Arc::new(move |request: DmaBufRequest| {
        let at = (request.start.as_ptr() as u64)
            .checked_sub(base)
            .ok_or_else(|| std::io::Error::other("pages before the view"))?;
        let alignment = match answer {
            Answer::Own => request.alignment,
            Answer::Piece => request.one_chunk_alignment,
        };
        if at.is_multiple_of(alignment.max(1)) {
            return Ok(vec![DmaBufChunk {
                file: Arc::clone(&region),
                offset: at,
                length: request.length,
            }]);
        }
        Ok(vec![DmaBufChunk {
            file: Arc::new(udmabuf(&file, at, request.length)?),
            offset: 0,
            length: request.length,
        }])
    })
}

/// The `at` quantile of `values`, sorted.
fn quantile(values: &[f64], at: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted
        .get(((sorted.len() as f64 * at) as usize).min(sorted.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0.0)
}

/// Wait until the device's live dma-buf memories fall to `count`, at most [`SETTLE`].
fn settled_to(kernel: &Kernel, count: u64) -> WorkCounts {
    let start = Instant::now();
    loop {
        let counts = kernel.device.work_counts();
        if counts.dma_buf_memories_live <= count || start.elapsed() > SETTLE {
            return counts;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// What an arm measured: its submission calls' nanoseconds, and its counts each round.
#[derive(Default)]
struct Arm {
    calls: Vec<f64>,
    live: Vec<u64>,
    imports: u64,
    joined: u64,
    import_nanos: u64,
    join_nanos: u64,
}

#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn one_piece_bench() {
    if !udmabuf_openable("one_piece_bench") {
        return;
    }
    let scratch = Scratch::new("one-piece");
    let kernel = Kernel::new(&scratch);
    kernel.warm();
    let device = &kernel.device;
    println!("ONE-PIECE on {}", device.name());
    let page = host_page();
    let stride = BYTES + page;
    let len = BUFFERS * stride + page;
    let ram = Ram::new(len, 0, len);
    let region = Arc::new(udmabuf(ram.file(), 0, len as u64).unwrap());
    let queue = device.new_command_queue().unwrap();
    let output = kernel.output(1);
    let mut arms = [Arm::default(), Arm::default(), Arm::default()];
    let base = device.work_counts().dma_buf_memories_live;
    let answers = [None, Some(Answer::Own), Some(Answer::Piece)];
    for _ in 0..ROUNDS {
        for (answer, arm) in answers.into_iter().zip(&mut arms) {
            let before = device.work_counts();
            let buffers: Vec<Buffer> = match answer {
                None => Vec::new(),
                Some(answer) => {
                    let source = source(&ram, &region, answer);
                    (0..BUFFERS)
                        .map(|index| {
                            let at = page + index * stride;
                            let buffer =
                                no_copy(device, &ram, (at, BYTES), Arc::clone(&source)).unwrap();
                            // Handed out, its import is kept for good, as a guest's bound
                            // buffers' are.
                            let _ = buffer.gpu_address();
                            buffer
                        })
                        .collect()
                }
            };
            let made = device.work_counts();
            arm.live.push(made.dma_buf_memories_live);
            arm.imports += made.dma_buf_imports - before.dma_buf_imports;
            arm.joined += made.dma_buf_pieces_joined - before.dma_buf_pieces_joined;
            arm.import_nanos += made.dma_buf_import_nanos - before.dma_buf_import_nanos;
            arm.join_nanos += made.dma_buf_join_nanos - before.dma_buf_join_nanos;
            for _ in 0..COMMITS {
                let command_buffer = queue.command_buffer().unwrap();
                kernel.dispatch(&command_buffer, (&output, 0));
                let committed = device.work_counts();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                let done = device.work_counts();
                let calls = done.queue_submissions - committed.queue_submissions;
                if calls > 0 {
                    let nanos = done.batch_submit_nanos - committed.batch_submit_nanos;
                    arm.calls.push(nanos as f64 / calls as f64);
                }
            }
            drop(buffers);
            settled_to(&kernel, base);
        }
    }
    for (name, arm) in ["none", "own", "piece"].into_iter().zip(&arms) {
        let rounds = ROUNDS as f64;
        println!(
            "ONE-PIECE {name}: {BUFFERS} buffers of {BYTES} bytes a page apart; dma-buf memories \
             live {:?}; {:.1} imports and {:.1} joins a round, taking {:.2} ms and {:.3} ms; \
             submission call median {:.1} us, p90 {:.1} us over {} calls",
            arm.live,
            arm.imports as f64 / rounds,
            arm.joined as f64 / rounds,
            arm.import_nanos as f64 / rounds / 1e6,
            arm.join_nanos as f64 / rounds / 1e6,
            quantile(&arm.calls, 0.5) / 1e3,
            quantile(&arm.calls, 0.9) / 1e3,
            arm.calls.len(),
        );
    }
}
