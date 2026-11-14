//! An indirect command buffer across its device's recovery, whose carry takes the buffer to the
//! generation it makes (`indirect_command_buffer/carry.rs`):
//!
//! - the commands the CPU encoded before it read back and run after it, as Metal keeps a
//!   buffer's records across its GPU's restart, where the records lie in the runtime's own pages,
//!   as they do on a device that imports host pointers; and a setter's write without the lock,
//!   held by a hook (`hold_next_set`) inside it while the recovery runs to the end, lands there;
//! - on a device that imports none, whose buffers' records lie in pooled memory, which the carry
//!   moves to new memory and resets: a set held once its batch is made, while the recovery runs to
//!   the end, writes the records the recovered buffer holds once let go, as a set made after the
//!   carry does, under the lock or without it; a set of the state the command named before the
//!   carry takes the lock; and one held inside its write without the lock, once it read the
//!   records' bytes, is waited for by the carry (`indirect_command_buffer/writers.rs`), so it
//!   lands before the carry; a getter of the slots past the records held once it read their
//!   address reads what they held before the carry.
//!
//! The tests print whether the carry kept the records' pages (`Kept`) or moved them (`Lost`).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).
//! The function is the owned fixture `examples/runtime_examples.ll` (`llvm-as`).

use super::device_recovery::{device, pooled_device, Scratch};
use super::device_recovery_carry::BYTES;
use super::device_recovery_live_table::{
    dispatched, dispatches, encode_command, execute, indirect_state, WORDS,
};
use crate::indirect_command_buffer::{hold_next_set, on_carry_waits, HoldAt};
use crate::{
    Buffer, ComputePipelineState, Device, Error, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, ResourceOptions, Size,
};
use metal2vulkan::metal::IndirectCommandType;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// How long a test waits for the carry to reach the write it holds.
const WAIT: Duration = Duration::from_secs(60);

/// What a recovery's carry did with a buffer's records, from their address before it: kept
/// their pages, or moved them to new memory.
fn carried(icb: &IndirectCommandBuffer, before: usize) -> &'static str {
    match icb.records_address() == before {
        true => "Kept",
        false => "Lost",
    }
}

/// A racing setter's thread, held by its hook until let go.
struct Racer {
    thread: std::thread::JoinHandle<Result<(), Error>>,
    go: mpsc::Sender<()>,
}

impl Racer {
    /// `set` of `icb` on a thread of its own, held at `at` by its hook, once it is held there.
    fn held(
        icb: &IndirectCommandBuffer,
        at: HoldAt,
        set: impl FnOnce(&IndirectCommandBuffer) -> Result<(), Error> + Send + 'static,
    ) -> Racer {
        let (held_tx, held) = mpsc::channel();
        let (go, go_rx) = mpsc::channel::<()>();
        let thread = {
            let icb = icb.clone();
            std::thread::spawn(move || {
                hold_next_set(at, move || {
                    held_tx.send(()).unwrap();
                    go_rx.recv().unwrap();
                });
                set(&icb)
            })
        };
        held.recv().unwrap();
        Racer { thread, go }
    }

    /// Let it go: its answer.
    fn go(self) -> Result<(), Error> {
        self.go.send(()).unwrap();
        self.thread.join().unwrap()
    }
}

/// `set` of `icb` held once its batch is made while `device` recovers to the end, then let go:
/// its answer, and what the recovery's carry did with the records ([`carried`]).
fn raced(
    device: &Device,
    icb: &IndirectCommandBuffer,
    set: impl FnOnce(&IndirectCommandBuffer) -> Result<(), Error> + Send + 'static,
) -> (Result<(), Error>, &'static str) {
    let before = icb.records_address();
    let racer = Racer::held(icb, HoldAt::Set, set);
    device.recover_for_test().unwrap();
    let carry = carried(icb, before);
    (racer.go(), carry)
}

/// Three `Private` buffers of `device`: a dispatch's input, output and scale.
fn buffers(device: &Device) -> [Buffer; 3] {
    let private = || {
        device
            .new_buffer(BYTES, ResourceOptions::private())
            .unwrap()
    };
    [private(), private(), private()]
}

/// The writes of `icb` since `before`, `writes_apart`'s: those under the lock and those without.
fn writes_since(icb: &IndirectCommandBuffer, before: (u64, u64)) -> (u64, u64) {
    let (locked, unlocked) = icb.writes_apart();
    (locked - before.0, unlocked - before.1)
}

/// `icb`'s command 0, with `state` and each of `buffers` but at slot `unset`, run: what
/// `execute` reads back.
fn run_but(
    device: &Device,
    icb: &IndirectCommandBuffer,
    state: &ComputePipelineState,
    buffers: &[Buffer; 3],
    unset: Option<u32>,
) -> Vec<f32> {
    let command = icb.indirect_compute_command_at(0).unwrap();
    command.set_compute_pipeline_state(state).unwrap();
    for (slot, buffer) in (0..).zip(buffers) {
        if Some(slot) != unset {
            command.set_kernel_buffer(Some(buffer), 0, slot).unwrap();
        }
    }
    let queue = device.new_command_queue().unwrap();
    let [input, output, scale] = buffers;
    execute(device, &queue, icb, [input, output, scale])
}

/// The first dispatch set on a buffer of pooled memory, which adds to what its records may hold
/// and so is written under the buffer's lock, held once its batch is made while a recovery
/// carries the buffer: let go, it takes the lock after the carry and writes the records there.
/// The recovered command answers the dispatch, and runs it once its state and buffers are set.
/// Without the records read under the lock, the write landed in the lost generation's memory the
/// batch found.
#[test]
fn a_locked_set_held_across_a_recovery_lands_on_the_recovered_records() {
    let scratch = Scratch::new();
    let device = pooled_device(&scratch.0);
    let compute = indirect_state(&device);
    let icb = dispatches(&device);
    let before = icb.writes_apart();
    let (answer, carry) = raced(&device, &icb, |icb| {
        icb.indirect_compute_command_at(0)?
            .concurrent_dispatch_threads(Size::new(WORDS, 1, 1), Size::new(32, 1, 1))
    });
    answer.unwrap();
    eprintln!("ICB-CARRY-RACE locked set, pooled: {carry}");
    // The racing write went under the lock: the path under test ran.
    assert_eq!(writes_since(&icb, before), (1, 0));
    let command = icb.indirect_compute_command_at(0).unwrap();
    assert_eq!(
        command.command_type().unwrap(),
        IndirectCommandType::CONCURRENT_DISPATCH_THREADS,
        "the dispatch set across the recovery ({carry})"
    );
    let buffers = buffers(&device);
    assert_eq!(
        run_but(&device, &icb, &compute, &buffers, None),
        dispatched()
    );
}

/// A buffer slot's set on a buffer of pooled memory, written without the lock, held once its
/// batch is made while a recovery carries the buffer: let go, it reads the records' bytes after
/// the carry and writes the records there. The recovered command answers the buffer, and runs
/// with it once the rest is set. Without the bytes read in the write, the batch's bytes were the
/// lost generation's.
#[test]
fn an_unlocked_set_held_across_a_recovery_lands_on_the_recovered_records() {
    let scratch = Scratch::new();
    let device = pooled_device(&scratch.0);
    let compute = indirect_state(&device);
    let icb = dispatches(&device);
    let buffers = buffers(&device);
    let output = buffers[1].clone();
    let address = output.gpu_address();
    let before = icb.writes_apart();
    let (answer, carry) = raced(&device, &icb, move |icb| {
        icb.indirect_compute_command_at(0)?
            .set_kernel_buffer(Some(&output), 0, 1)
    });
    answer.unwrap();
    eprintln!("ICB-CARRY-RACE unlocked set, pooled: {carry}");
    // The racing write went without the lock: the path under test ran.
    assert_eq!(writes_since(&icb, before), (0, 1));
    let command = icb.indirect_compute_command_at(0).unwrap();
    let slot = command.kernel_buffer(1).unwrap();
    assert_eq!(
        slot.map(|binding| binding.address),
        Some(address),
        "the buffer set across the recovery ({carry})"
    );
    command
        .concurrent_dispatch_threads(Size::new(WORDS, 1, 1), Size::new(32, 1, 1))
        .unwrap();
    assert_eq!(
        run_but(&device, &icb, &compute, &buffers, Some(1)),
        dispatched()
    );
}

/// A set of the state the command names, on a buffer of pooled memory, held once its batch is
/// made while a recovery carries the buffer (which resets the command and forgets the commands'
/// keys): let go, it finds the command naming no state since the carry, and sets the state under
/// the lock with every check, rather than writing its ID again without it. The command runs the
/// state.
#[test]
fn a_set_of_the_named_state_across_a_recovery_takes_the_lock() {
    let scratch = Scratch::new();
    let device = pooled_device(&scratch.0);
    let compute = indirect_state(&device);
    let icb = dispatches(&device);
    let buffers = buffers(&device);
    let [input, output, scale] = &buffers;
    encode_command(&icb, &compute, [input, output, scale]);
    // Set again before the recovery: the command names the state, so it goes without the lock.
    let before = icb.writes_apart();
    let command = icb.indirect_compute_command_at(0).unwrap();
    command.set_compute_pipeline_state(&compute).unwrap();
    assert_eq!(writes_since(&icb, before), (0, 1));
    let before = icb.writes_apart();
    let state = compute.clone();
    let (answer, carry) = raced(&device, &icb, move |icb| {
        icb.indirect_compute_command_at(0)?
            .set_compute_pipeline_state(&state)
    });
    answer.unwrap();
    eprintln!("ICB-CARRY-RACE named state, pooled: {carry}");
    assert_eq!(
        writes_since(&icb, before),
        (1, 0),
        "the set after the carry took the lock ({carry})"
    );
    let command = icb.indirect_compute_command_at(0).unwrap();
    command
        .concurrent_dispatch_threads(Size::new(WORDS, 1, 1), Size::new(32, 1, 1))
        .unwrap();
    for (slot, buffer) in (0..).zip(&buffers) {
        command.set_kernel_buffer(Some(buffer), 0, slot).unwrap();
    }
    let queue = device.new_command_queue().unwrap();
    assert_eq!(
        execute(&device, &queue, &icb, [input, output, scale]),
        dispatched()
    );
}

/// A buffer slot's set on a buffer of pooled memory, held inside its write without the lock once
/// it read the records' bytes: a recovery on another thread reaches the buffer's carry, which
/// waits for the write (the carry's hook says so, the recovery not ended), and ends once the
/// write is let go. The write landed before the carry, which moved the records to new memory and
/// reset the command there, so it holds no buffer. Without the wait, the recovery ended while the
/// write held the lost generation's bytes.
#[test]
fn a_recovery_s_carry_waits_for_a_write_without_the_lock() {
    let scratch = Scratch::new();
    let device = pooled_device(&scratch.0);
    let compute = indirect_state(&device);
    let icb = dispatches(&device);
    let buffers = buffers(&device);
    let output = buffers[1].clone();
    let records = icb.records_address();
    let before = icb.writes_apart();
    let racer = Racer::held(&icb, HoldAt::Write, move |icb| {
        icb.indirect_compute_command_at(0)?
            .set_kernel_buffer(Some(&output), 0, 1)
    });
    let (waits_tx, waits) = mpsc::channel();
    let recovery = {
        let device = device.clone();
        std::thread::spawn(move || {
            on_carry_waits(move || waits_tx.send(()).unwrap());
            device.recover_for_test()
        })
    };
    let began = Instant::now();
    loop {
        match waits.recv_timeout(Duration::from_millis(10)) {
            Ok(()) => break,
            Err(_) => {
                assert!(
                    !recovery.is_finished(),
                    "the recovery ended while a write without the lock held the records' bytes"
                );
                assert!(began.elapsed() < WAIT, "the carry never reached the write");
            }
        }
    }
    // The carry waits for the write held: the recovery cannot end before it is let go.
    assert!(!recovery.is_finished());
    racer.go().unwrap();
    recovery.join().unwrap().unwrap();
    let carry = carried(&icb, records);
    eprintln!("ICB-CARRY-RACE held write, pooled: {carry}");
    assert_eq!(carry, "Lost", "pooled records moved by the carry");
    assert_eq!(writes_since(&icb, before), (0, 1));
    let slot = icb
        .indirect_compute_command_at(0)
        .unwrap()
        .kernel_buffer(1)
        .unwrap()
        .map(|binding| binding.address);
    assert_eq!(slot, None, "the command reset by the carry");
    let [input, output, scale] = &buffers;
    encode_command(&icb, &compute, [input, output, scale]);
    let queue = device.new_command_queue().unwrap();
    assert_eq!(
        execute(&device, &queue, &icb, [input, output, scale]),
        dispatched()
    );
}

/// A buffer slot's set on a buffer whose records lie in the runtime's own pages, held inside its
/// write without the lock once it read the records' bytes while a recovery runs to the end: the
/// carry keeps the pages and waits for nothing, and the write, let go, lands in the records the
/// recovered buffer holds. The command answers the buffer and runs with it.
#[test]
fn a_write_held_across_a_recovery_lands_on_the_records_it_keeps() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let compute = indirect_state(&device);
    let icb = dispatches(&device);
    let buffers = buffers(&device);
    let output = buffers[1].clone();
    let address = output.gpu_address();
    let records = icb.records_address();
    let before = icb.writes_apart();
    let racer = Racer::held(&icb, HoldAt::Write, move |icb| {
        icb.indirect_compute_command_at(0)?
            .set_kernel_buffer(Some(&output), 0, 1)
    });
    device.recover_for_test().unwrap();
    let carry = carried(&icb, records);
    eprintln!("ICB-CARRY-RACE held write, own pages: {carry}");
    racer.go().unwrap();
    assert_eq!(
        carry, "Kept",
        "records in the runtime's own pages kept by the carry"
    );
    assert_eq!(writes_since(&icb, before), (0, 1));
    let command = icb.indirect_compute_command_at(0).unwrap();
    let slot = command.kernel_buffer(1).unwrap();
    assert_eq!(slot.map(|binding| binding.address), Some(address));
    command
        .concurrent_dispatch_threads(Size::new(WORDS, 1, 1), Size::new(32, 1, 1))
        .unwrap();
    assert_eq!(
        run_but(&device, &icb, &compute, &buffers, Some(1)),
        dispatched()
    );
}

/// A getter of a threadgroup length past the record's, which the slots past the records hold, on
/// a buffer of pooled memory, held once it read their bytes' address while a recovery carries the
/// buffer to the end (the slots moved to new memory, where they read zero): let go, it reads the
/// length set before the carry from the lost generation's bytes, which its hold kept. Without the
/// hold, the read reached memory the lost generation let go of at its rest.
#[test]
fn a_getter_held_across_a_recovery_reads_the_slots_it_found() {
    let scratch = Scratch::new();
    let device = pooled_device(&scratch.0);
    let icb = dispatches(&device);
    let command = icb.indirect_compute_command_at(0).unwrap();
    command.set_threadgroup_memory_length(64, 31).unwrap();
    let records = icb.records_address();
    let (held_tx, held) = mpsc::channel();
    let (go, go_rx) = mpsc::channel::<()>();
    let reader = {
        let icb = icb.clone();
        std::thread::spawn(move || {
            hold_next_set(HoldAt::Read, move || {
                held_tx.send(()).unwrap();
                go_rx.recv().unwrap();
            });
            icb.indirect_compute_command_at(0)?
                .threadgroup_memory_length(31)
        })
    };
    held.recv().unwrap();
    device.recover_for_test().unwrap();
    go.send(()).unwrap();
    assert_eq!(reader.join().unwrap().unwrap(), Some(64));
    // Read again, the carried slots, moved to new memory: zero.
    let carry = carried(&icb, records);
    let now = command.threadgroup_memory_length(31).unwrap();
    eprintln!("ICB-CARRY-RACE held read, pooled: {carry}, the slot after the carry: {now:?}");
    assert_eq!((carry, now), ("Lost", Some(0)));
}

/// The render commands of [`a_recovery_keeps_the_commands_the_cpu_encoded`]'s buffer of draws.
const DRAWS: u64 = 64;

/// The commands the CPU encoded before a recovery, read back and run after it as they were, as
/// Metal keeps a buffer's records across its GPU's restart: a compute command encoded whole
/// (state, buffers, dispatch) runs, and each of a buffer of draws' commands reads back its two
/// vertex and two fragment buffers, nil at offsets of their own, which the setters wrote without
/// the lock. The records lie in the runtime's own pages where the device imports host pointers,
/// which the carry keeps (`Kept`); in pooled memory it moved them to new memory and reset every
/// command (`Lost`), where Metal's are kept.
#[test]
fn a_recovery_keeps_the_commands_the_cpu_encoded() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let compute = indirect_state(&device);
    let icb = dispatches(&device);
    let private = || {
        device
            .new_buffer(BYTES, ResourceOptions::private())
            .unwrap()
    };
    let (input, output, scale) = (private(), private(), private());
    encode_command(&icb, &compute, [&input, &output, &scale]);
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.max_vertex_buffer_bind_count = 2;
    descriptor.max_fragment_buffer_bind_count = 2;
    let draws = device
        .new_indirect_command_buffer(&descriptor, DRAWS, ResourceOptions::shared())
        .unwrap();
    // Command `i`'s vertex buffers 0 and 1 and fragment buffers 0 and 1: nil, at offsets of its
    // own (a nil buffer's address is its offset).
    let at = |i: u64| {
        let first = (i + 1) * 64;
        [first, first + 16, first + 32, first + 48]
    };
    for i in 0..DRAWS {
        let command = draws.indirect_render_command_at(i).unwrap();
        let [v0, v1, f0, f1] = at(i);
        command.set_vertex_buffer(None, v0, 0).unwrap();
        command.set_vertex_buffer(None, v1, 1).unwrap();
        command.set_fragment_buffer(None, f0, 0).unwrap();
        command.set_fragment_buffer(None, f1, 1).unwrap();
    }
    let records = [icb.records_address(), draws.records_address()];
    device.recover_for_test().unwrap();
    let kept = [icb.records_address(), draws.records_address()] == records;
    let carry = if kept { "Kept" } else { "Lost" };
    eprintln!("ICB-CARRY-RACE encoded before the recovery: {carry}");
    let read = |i: u64| {
        let command = draws.indirect_render_command_at(i).unwrap();
        let address = |binding: Result<Option<crate::IndirectBufferBinding>, Error>| {
            binding.unwrap().map(|binding| binding.address.0)
        };
        [
            address(command.vertex_buffer(0)),
            address(command.vertex_buffer(1)),
            address(command.fragment_buffer(0)),
            address(command.fragment_buffer(1)),
        ]
    };
    for i in 0..DRAWS {
        assert_eq!(read(i), at(i).map(Some), "render command {i} ({carry})");
    }
    let command = icb.indirect_compute_command_at(0).unwrap();
    assert_eq!(
        command.command_type().unwrap(),
        IndirectCommandType::CONCURRENT_DISPATCH_THREADS,
        "the dispatch encoded before the recovery ({carry})"
    );
    let queue = device.new_command_queue().unwrap();
    assert_eq!(
        execute(&device, &queue, &icb, [&input, &output, &scale]),
        dispatched(),
        "the command encoded before the recovery ran ({carry})"
    );
}
