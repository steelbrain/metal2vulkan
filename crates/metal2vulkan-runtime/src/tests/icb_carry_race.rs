//! An indirect command buffer's setter racing its device's recovery, whose carry takes the
//! buffer to the generation it makes (`indirect_command_buffer/carry.rs`): a set held by a hook
//! (`hold_next_set`) once its batch is made, while the recovery runs to the end, writes the
//! records the recovered buffer holds once let go, as a set made after the carry does, where
//! the carry moved them to new memory (a buffer of pooled memory: its records reset there) as
//! where it kept them.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).
//! The function is the owned fixture `examples/runtime_examples.ll` (`llvm-as`).

use super::device_recovery::{device, Scratch};
use super::device_recovery_carry::BYTES;
use super::device_recovery_live_table::{dispatched, dispatches, execute, indirect_state, WORDS};
use crate::indirect_command_buffer::hold_next_set;
use crate::{Device, Error, IndirectCommandBuffer, ResourceOptions, Size};
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
