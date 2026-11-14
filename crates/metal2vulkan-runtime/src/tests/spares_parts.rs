//! What the device's spare command pools make and hold under command buffers of one-dispatch
//! compute encoders that each wait for a fence the one before updated, so none records on in the
//! one before's `VkCommandBuffer` (`encoder_fold.rs`) and the scheduler records a command buffer
//! of barriers before each, many in flight, of a count of encoders that varies from one to the
//! next (`spares.rs`): a guest's command buffers split into parts. Once each count ran with as
//! many in flight, the spares hold what they take: the command buffers made stay flat, and they
//! hold about what is in use at once; once the load falls, they shed down to it.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`). The kernel is `scale_and_offset` of the owned fixture
//! `validation/fixtures/public/examples/runtime_examples.ll`.

use crate::spares::{IDLE_FOR, POOL_HOLDS, RECENT_FOR};
use crate::tests::encoder_holds::library;
use crate::tests::freed_memory::device;
use crate::{
    Buffer, CommandBuffer, CommandQueue, ComputePipelineState, Device, PipelineOptions,
    ResourceOptions, Size,
};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// The compute encoders of the command buffers, in turn: 13 on average.
const PARTS: [usize; 8] = [2, 12, 5, 30, 9, 24, 3, 18];

/// The command buffers in flight.
const IN_FLIGHT: usize = 16;

/// The command buffers a round commits: eight of each count.
const ROUND: usize = 64;

/// The rounds that warm the spares, and those measured after them.
const WARM: usize = 8;
const MEASURED: usize = 3;

/// The longest a command buffer is waited for.
const WAIT: Duration = Duration::from_secs(30);

/// A device's queue, with the kernel and its buffers.
struct Load {
    device: Device,
    queue: CommandQueue,
    state: ComputePipelineState,
    buffers: [Buffer; 3],
}

impl Load {
    fn new(device: Device) -> Load {
        let queue = device.new_command_queue().unwrap();
        let examples = library(&device, "examples/runtime_examples.ll");
        let kernel = examples.new_function_with_name("scale_and_offset").unwrap();
        let state = device
            .new_compute_pipeline_state_with_function(&kernel, &PipelineOptions::default())
            .unwrap();
        let private = ResourceOptions::private();
        let buffers = [
            device.new_buffer(256, private).unwrap(),
            device.new_buffer(256, private).unwrap(),
            device.new_buffer(16, ResourceOptions::shared()).unwrap(),
        ];
        Load {
            device,
            queue,
            state,
            buffers,
        }
    }

    /// A command buffer of `encoders` one-dispatch compute encoders, each after the first waiting
    /// for a fence of its own the one before updated, and updating it.
    fn encoded(&self, encoders: usize) -> CommandBuffer {
        let command_buffer = self.queue.command_buffer().unwrap();
        let fence = self.device.new_fence();
        for at in 0..encoders {
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            if at > 0 {
                encoder.wait_for_fence(&fence).unwrap();
            }
            encoder.set_compute_pipeline_state(&self.state).unwrap();
            for (index, buffer) in self.buffers.iter().enumerate() {
                encoder.set_buffer(Some(buffer), 0, index as u32).unwrap();
            }
            (encoder.dispatch_threads(Size::new(64, 1, 1), Size::new(64, 1, 1))).unwrap();
            encoder.update_fence(&fence).unwrap();
            encoder.end_encoding().unwrap();
        }
        command_buffer
    }

    /// `count` command buffers of `encoders(index)` encoders each, at most `in_flight` of them
    /// committed whose pools are not yet given back, then every one's given back, and the frees
    /// their give-backs posted. A completed handler runs once its command buffer's pools went back
    /// to the spares (`command_buffer/mod.rs`), so each is counted by its handler: a command
    /// buffer's completion seen by a waiter may come before its give-back.
    fn run(&self, count: usize, in_flight: usize, encoders: impl Fn(usize) -> usize) {
        let (given, back) = mpsc::channel();
        let mut out = 0;
        for index in 0..count {
            let command_buffer = self.encoded(encoders(index));
            let given = given.clone();
            (command_buffer.add_completed_handler(Box::new(move |_: &CommandBuffer| {
                let _ = given.send(());
            })))
            .unwrap();
            command_buffer.commit().unwrap();
            out += 1;
            while out > in_flight {
                back.recv_timeout(WAIT).unwrap();
                out -= 1;
            }
        }
        for _ in 0..out {
            back.recv_timeout(WAIT).unwrap();
        }
        self.device.shared().frees.drain();
    }

    /// A round of [`ROUND`] command buffers of [`PARTS`] encoders in turn, [`IN_FLIGHT`] in
    /// flight.
    fn round(&self) {
        self.run(ROUND, IN_FLIGHT, |index| PARTS[index % PARTS.len()]);
    }

    /// The command buffers and command pools the device made so far, the command buffers the
    /// spares hold now, and those they freed so far (past recent uses, past the bound, shed).
    fn counts(&self) -> (u64, u64, usize, [u64; 3]) {
        let counts = self.device.work_counts();
        let spares = &self.device.shared().spares;
        let churn = spares.churn();
        let freed = [churn.past_recent_uses, churn.past_held, churn.shed];
        let held = spares.command_buffers_held();
        (
            counts.command_buffers_made,
            counts.command_pools_made,
            held,
            freed,
        )
    }
}

/// The command buffers holding their pools at once: [`IN_FLIGHT`] committed, and the one encoding.
const ALIVE: usize = IN_FLIGHT + 1;

/// At most the primaries the load has in use at once, of [`ALIVE`] command buffers of [`PARTS`]
/// encoders in turn, the largest counts together: each encoder's own and one of barriers before
/// it, and the batch's own and its first's.
fn in_use() -> usize {
    let each = |encoders: usize| 2 * encoders + 2;
    let mut largest = PARTS;
    largest.sort_unstable_by(|a, b| b.cmp(a));
    let whole = ALIVE / PARTS.len() * PARTS.iter().map(|&n| each(n)).sum::<usize>();
    whole
        + largest[..ALIVE % PARTS.len()]
            .iter()
            .map(|&n| each(n))
            .sum::<usize>()
}

/// Once each count ran with as many in flight, the spares hold what the load takes: the command
/// buffers made stay flat, and the spares free none. The most the load takes at once follows the
/// threads' timing, so a later peak may still make a few: fewer than one a command buffer of a
/// round ([`ROUND`]) over the measured rounds here, where none were made after the fourth round
/// in three release runs and after the sixth in a debug one, and sections that took no spare
/// made about 60 a round (lavapipe). And the spares hold at most what the load has in use at once, with what the pools
/// out may hold unused as a command buffer is made: a command buffer takes a pool only once the
/// pools it took handed out every one of the level they held, so only its last may hold some
/// unused, at most [`POOL_HOLDS`]; and each pool holds a primary begun ahead and a secondary
/// besides. Before, the barriers' command buffers, recorded in a section of the exit guard, took
/// no spare, and a command buffer made what it lacked in its first pool until it had handed out
/// 16: the pools grew past what was in use at once to the bound, and each give-back shed the
/// oldest whole, which the next takers made again (the module docs of `spares.rs`).
#[test]
fn fenced_command_buffers_in_flight_stay_flat_once_warm() {
    let scratch = std::env::temp_dir().join(format!("m2v-spares-parts-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let load = Load::new(device(&scratch));
    let mut rounds = Vec::new();
    for _ in 0..WARM + MEASURED {
        load.round();
        rounds.push(load.counts());
    }
    let (warm, _, _, freed_warm) = rounds[WARM - 1];
    let (made, pools, after, freed) = *rounds.last().unwrap();
    let bound = in_use() + ALIVE * POOL_HOLDS + 2 * pools as usize;
    println!(
        "SPARES-PARTS {}: rounds of {ROUND} command buffers of {PARTS:?} fenced encoders, \
         {IN_FLIGHT} in flight: command buffers made, command pools made, command buffers the \
         spares hold and those they freed after each {rounds:?}; at most {} in use at once, the \
         bound {bound}",
        load.device.name(),
        in_use()
    );
    assert!(
        made - warm < ROUND as u64,
        "{} command buffers made once warm",
        made - warm
    );
    assert!(after <= bound, "the spares hold {after}, past {bound}");
    assert_eq!(
        freed, freed_warm,
        "the spares freed command buffers under a steady load"
    );
    drop(load);
    let _ = std::fs::remove_dir_all(&scratch);
}

/// Once the load falls to one command buffer of one encoder at a time, the spares shed what it no
/// longer takes: each pool it leaves idle for [`IDLE_FOR`] sheds its own at a later give-back,
/// and those it takes keep what their last uses, and their uses in their last two spans of
/// [`RECENT_FOR`], handed out.
#[test]
fn fenced_command_buffers_in_flight_shed_down_once_the_load_falls() {
    let scratch =
        std::env::temp_dir().join(format!("m2v-spares-parts-drop-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let load = Load::new(device(&scratch));
    for _ in 0..WARM {
        load.round();
    }
    let (_, _, warm, _) = load.counts();
    let start = Instant::now();
    let mut trace = Vec::new();
    while start.elapsed() < IDLE_FOR + 2 * RECENT_FOR {
        let second = Instant::now();
        while second.elapsed() < Duration::from_secs(1) {
            load.run(1, 0, |_| 1);
            std::thread::sleep(Duration::from_millis(5));
        }
        trace.push(load.counts().2);
    }
    let after = *trace.last().unwrap();
    let left: Vec<_> = (load.device.shared().spares.command_pools_held())
        .into_iter()
        .map(|(_, held)| held)
        .collect();
    println!(
        "SPARES-PARTS-DROP {}: the spares held {warm} after {WARM} rounds, then each second of \
         one command buffer of one encoder every 5 ms {trace:?}, in pools holding {left:?}",
        load.device.name(),
    );
    assert!(warm > 4 * after, "the spares hold {after}, from {warm}");
    drop(load);
    let _ = std::fs::remove_dir_all(&scratch);
}
