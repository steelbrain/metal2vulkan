//! An indirect command buffer across its device's recovery, whose carry takes the buffer to the
//! generation it makes (`indirect_command_buffer/carry.rs`):
//!
//! - the commands the CPU encoded before it read back and run after it, as Metal keeps a
//!   buffer's records across its GPU's restart, where the records lie in the runtime's own pages,
//!   as they do on a device that imports host pointers;
//! - a set held by a hook (`hold_next_set`) once its batch is made, while the recovery runs to the
//!   end, writes the records the recovered buffer holds once let go, as a set made after the
//!   carry does, where the carry moved them to new memory (a buffer of pooled memory: its records
//!   reset there) as where it kept them.
//!
//! The tests print whether the carry kept the records' pages (`Kept`) or moved them (`Lost`).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).
//! The function is the owned fixture `examples/runtime_examples.ll` (`llvm-as`).

use super::device_recovery::{device, Scratch};
use super::device_recovery_carry::BYTES;
use super::device_recovery_live_table::{
    dispatched, dispatches, encode_command, execute, indirect_state, WORDS,
};
use crate::indirect_command_buffer::hold_next_set;
use crate::{
    Device, Error, IndirectCommandBuffer, IndirectCommandBufferDescriptor, ResourceOptions, Size,
};
use metal2vulkan::metal::IndirectCommandType;
use std::sync::mpsc;

/// `set` of `icb` on a thread of its own, held by its first field's hook while `device` recovers,
/// then let go: its answer, and whether the recovery's carry moved the records to new memory.
fn raced(
    device: &Device,
    icb: &IndirectCommandBuffer,
    set: impl FnOnce(&IndirectCommandBuffer) -> Result<(), Error> + Send + 'static,
) -> (Result<(), Error>, bool) {
    let (held_tx, held) = mpsc::channel();
    let (go, go_rx) = mpsc::channel::<()>();
    let before = icb.records_address();
    let racer = {
        let icb = icb.clone();
        std::thread::spawn(move || {
            hold_next_set(move || {
                held_tx.send(()).unwrap();
                go_rx.recv().unwrap();
            });
            set(&icb)
        })
    };
    held.recv().unwrap();
    device.recover_for_test().unwrap();
    let moved = icb.records_address() != before;
    go.send(()).unwrap();
    (racer.join().unwrap(), moved)
}

/// The first dispatch set on a buffer, which adds to what its records may hold and so is written
/// under the buffer's lock, held once its batch is made while a recovery carries the buffer:
/// let go, it takes the lock after the carry and writes the records there. The recovered
/// command answers the dispatch, and runs it once its state and buffers are set. Without the
/// records read under the lock, the write landed in the lost generation's memory the batch found.
#[test]
fn a_locked_set_held_across_a_recovery_lands_on_the_recovered_records() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let compute = indirect_state(&device);
    let icb = dispatches(&device);
    let (locked, unlocked) = icb.writes_apart();
    let (answer, moved) = raced(&device, &icb, |icb| {
        icb.indirect_compute_command_at(0)?
            .concurrent_dispatch_threads(Size::new(WORDS, 1, 1), Size::new(32, 1, 1))
    });
    answer.unwrap();
    eprintln!("ICB-CARRY-RACE locked set: the carry moved the records: {moved}");
    // The racing write went under the lock: the path under test ran.
    let (locked_now, unlocked_now) = icb.writes_apart();
    assert_eq!((locked_now - locked, unlocked_now - unlocked), (1, 0));
    let command = icb.indirect_compute_command_at(0).unwrap();
    assert_eq!(
        command.command_type().unwrap(),
        IndirectCommandType::CONCURRENT_DISPATCH_THREADS,
        "the dispatch set across the recovery (records moved: {moved})"
    );
    let queue = device.new_command_queue().unwrap();
    let private = || {
        device
            .new_buffer(BYTES, ResourceOptions::private())
            .unwrap()
    };
    let (input, output, scale) = (private(), private(), private());
    command.set_compute_pipeline_state(&compute).unwrap();
    for (slot, buffer) in [&input, &output, &scale].into_iter().enumerate() {
        command
            .set_kernel_buffer(Some(buffer), 0, slot as u32)
            .unwrap();
    }
    assert_eq!(
        execute(&device, &queue, &icb, [&input, &output, &scale]),
        dispatched()
    );
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
