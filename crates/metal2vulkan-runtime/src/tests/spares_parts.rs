//! What the device's spare command pools make and hold under command buffers of one-dispatch
//! compute encoders that each wait for a fence the one before updated, so none records on in the
//! one before's `VkCommandBuffer` (`encoder_fold.rs`) and the scheduler records a command buffer
//! of barriers before each, many in flight, of a count of encoders that varies from one to the
//! next (`spares.rs`): a guest's command buffers split into parts. The device makes command
//! buffers only as far as the most the load had in use at once needs, with what the pools out
//! of the spares may hold unused: once warm, more are made only for a later, larger peak, and the
//! spares free none and hold about what is in use at once; once the load falls, they shed down to
//! it.
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
use std::collections::BTreeMap;
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
    /// their give-backs posted; and the most they held at once ([`Peak`]). A completed handler
    /// runs once its command buffer's pools went back to the spares (`command_buffer/mod.rs`), so
    /// each is counted by its handler: a command buffer's completion seen by a waiter may come
    /// before its give-back. As each command buffer is encoded, those committed whose handler had
    /// not run as it began, with it, hold at least what is in use until the next begins: those
    /// before it only complete meanwhile.
    fn run(&self, count: usize, in_flight: usize, encoders: impl Fn(usize) -> usize) -> Peak {
        let (given, back) = mpsc::channel();
        let mut out = BTreeMap::new();
        let mut peak = Peak::default();
        for index in 0..count {
            while let Ok(done) = back.try_recv() {
                out.remove(&done);
            }
            let encoders = encoders(index);
            let before = out.values().copied().map(primaries).sum::<usize>();
            peak = peak.max(Peak {
                alive: out.len() + 1,
                primaries: before + primaries(encoders),
            });
            let command_buffer = self.encoded(encoders);
            let given = given.clone();
            (command_buffer.add_completed_handler(Box::new(move |_: &CommandBuffer| {
                let _ = given.send(index);
            })))
            .unwrap();
            command_buffer.commit().unwrap();
            out.insert(index, encoders);
            while out.len() > in_flight {
                out.remove(&back.recv_timeout(WAIT).unwrap());
            }
        }
        while !out.is_empty() {
            out.remove(&back.recv_timeout(WAIT).unwrap());
        }
        self.device.shared().frees.drain();
        peak
    }

    /// A round of [`ROUND`] command buffers of [`PARTS`] encoders in turn, [`IN_FLIGHT`] in
    /// flight, and the most they held at once.
    fn round(&self) -> Peak {
        self.run(ROUND, IN_FLIGHT, |index| PARTS[index % PARTS.len()])
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

/// The most a run's command buffers held at once ([`Load::run`]).
#[derive(Clone, Copy, Debug, Default)]
struct Peak {
    /// The command buffers holding their pools: those committed whose completed handler had not
    /// run, and the one encoding.
    alive: usize,
    /// The primaries those have in use, at most ([`primaries`]).
    primaries: usize,
}

impl Peak {
    /// The larger of each.
    fn max(self, other: Peak) -> Peak {
        Peak {
            alive: self.alive.max(other.alive),
            primaries: self.primaries.max(other.primaries),
        }
    }
}

/// At most the primaries a command buffer of `encoders` of the load's encoders has in use: each
/// encoder's own and one of barriers before it, and the batch's own and its first's.
fn primaries(encoders: usize) -> usize {
    2 * encoders + 2
}

/// At most the primaries the load has in use at once, of [`ALIVE`] command buffers of [`PARTS`]
/// encoders in turn, the largest counts together ([`primaries`]).
fn in_use() -> usize {
    let mut largest = PARTS;
    largest.sort_unstable_by(|a, b| b.cmp(a));
    let whole = ALIVE / PARTS.len() * PARTS.iter().copied().map(primaries).sum::<usize>();
    whole
        + largest[..ALIVE % PARTS.len()]
            .iter()
            .copied()
            .map(primaries)
            .sum::<usize>()
}

/// After every round, the command buffers the device holds (made, less those the spares freed) are
/// at most the primaries the load had in use at once so far, by the most its command buffers
/// held at once that the test saw ([`Peak`]), with what the pools out of the spares may hold
/// unused as a command buffer is made: the device makes one only where no spare holds one of
/// its level, and a command buffer takes a pool only once the pools it took handed out every one
/// of the level they held, so only its last may hold some unused, at most [`POOL_HOLDS`]; and
/// each pool holds a primary begun ahead and a secondary besides. The most the load has in use at
/// once follows the threads' timing, as a busier device keeps more in flight: once warm, more are
/// made only as a later peak needs them, where the bound grows with it. Once warm, the spares free
/// none. And the spares hold at most what the load can have in use at once ([`in_use`]), with what
/// the pools out may hold unused. Before, the barriers' command buffers, recorded in a section of
/// the exit guard, took no spare, and a command buffer made what it lacked in its first pool until
/// it had handed out 16: the pools grew past what was in use at once to the bound, and each
/// give-back shed the oldest whole, which the next takers made again (the module docs of
/// `spares.rs`).
#[test]
fn fenced_command_buffers_in_flight_stay_flat_once_warm() {
    let scratch = std::env::temp_dir().join(format!("m2v-spares-parts-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let load = Load::new(device(&scratch));
    let mut rounds = Vec::new();
    let mut peak = Peak::default();
    for _ in 0..WARM + MEASURED {
        peak = peak.max(load.round());
        let (made, pools, held, freed) = load.counts();
        rounds.push((made, pools, held, freed, peak.alive, peak.primaries));
    }
    let (warm, _, _, freed_warm, _, _) = rounds[WARM - 1];
    let (made, pools, after, freed, _, _) = *rounds.last().unwrap();
    let unused = |pools: u64| ALIVE * POOL_HOLDS + 2 * pools as usize;
    let bound = in_use() + unused(pools);
    println!(
        "SPARES-PARTS {}: rounds of {ROUND} command buffers of {PARTS:?} fenced encoders, \
         {IN_FLIGHT} in flight: command buffers made, command pools made, command buffers the \
         spares hold and those they freed after each, and the most command buffers alive at once \
         and the primaries they had in use so far {rounds:?}; {} made once warm; at most {} in \
         use at once, the bound {bound}",
        load.device.name(),
        made - warm,
        in_use()
    );
    for (round, &(made, pools, _, freed, _, needed)) in rounds.iter().enumerate() {
        let holds = made - freed.iter().sum::<u64>();
        let most = needed + unused(pools);
        assert!(
            holds <= most as u64,
            "after round {round} the device holds {holds} command buffers, past the {needed} \
             primaries the load had in use at once and what the pools may hold unused ({most})"
        );
    }
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
