//! What a device's recovery carries to its next generation (`device/carry.rs`): the program's
//! events, `Private` buffers and buffers in fresh pages go on as the same objects, a pooled `Shared`
//! buffer stays refused, and the lost generation goes once nothing records on it and the last
//! object left on it went, however many recoveries the program's objects outlive.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).

use super::device_recovery::{
    device, heard_device, outcome, recovery_notices, Heard, Scratch, WAIT,
};
use crate::fresh_pages::FRESH_PAGES_PAST;
use crate::{
    Buffer, CommandBufferStatus, CommandQueue, Device, Error, EventRef, ResourceOptions,
    RuntimeGap, SharedEvent, SharedEventListener, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// The bytes the tests write and read of each buffer.
const BYTES: u64 = 4096;

/// The bytes of each of [`REUSED_BUFFERS`] buffers the program lets go of before a loss: under half
/// a pooled block, so that each is placed in a pooled block whatever a driver pads it by, and more
/// than a block in all, so that one is placed in a block apart from the test's other buffers,
/// which the lost generation keeps for reuse as it empties, or with the buffer parked in it.
const REUSED: u64 = crate::memory::BLOCK_BYTES * 3 / 8;

/// How many buffers of [`REUSED`] bytes the program lets go of before a loss.
const REUSED_BUFFERS: usize = 3;

/// The recoveries one process's objects outlive in [`seven_recoveries_end_with_one_live_vk_device`]:
/// one less than the live `VkDevice`s an NVIDIA driver lets a process hold.
const RECOVERIES: u64 = 7;

/// `buffer`'s first [`BYTES`] bytes set to `byte` by `queue`'s GPU work, completed.
#[track_caller]
fn fill(queue: &CommandQueue, buffer: &Buffer, byte: u8) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..BYTES, byte).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
}

/// `source`'s first [`BYTES`] bytes as `queue`'s GPU work reads them: copied into a new `Shared`
/// buffer of `device`'s serving generation, completed. Refused as the copy's encoding refuses
/// `source`.
#[track_caller]
fn read(device: &Device, queue: &CommandQueue, source: &Buffer) -> Result<Vec<u8>, Error> {
    let target = device.new_buffer(BYTES, ResourceOptions::shared())?;
    let command_buffer = queue.command_buffer()?;
    let mut blit = command_buffer.blit_command_encoder()?;
    blit.copy_from_buffer_to_buffer(source, 0, &target, 0, BYTES)?;
    blit.end_encoding()?;
    command_buffer.commit()?;
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    let contents = target.contents().unwrap();
    // SAFETY: a `Shared` buffer of `BYTES` bytes, which no work in flight writes.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), BYTES as usize) };
    Ok(bytes.to_vec())
}

/// `event` signalled to `value` by `queue`'s GPU work, completed.
#[track_caller]
fn signal(queue: &CommandQueue, event: &SharedEvent, value: u64) {
    let command_buffer = queue.command_buffer().unwrap();
    (command_buffer.encode_signal_event(EventRef::from(event), value)).unwrap();
    command_buffer.commit().unwrap();
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    assert!(event.wait_until_signaled_value(value, WAIT));
}

/// The device's live `VkDevice`s once they are `live`, waited for at most [`WAIT`]: a generation
/// goes as its last holder lets go of it, which a runtime thread may be a moment after the test's.
fn live_vk_devices(device: &Device, live: u64) -> u64 {
    let began = Instant::now();
    loop {
        let now = device.work_counts().vk_devices_live;
        if now == live || began.elapsed() > WAIT {
            return now;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// A recovery carries a `Private` buffer (its address kept, its bytes zero), a buffer in fresh
/// pages (its `contents` pointer and bytes kept, the host's writes through that pointer after
/// the recovery read by the GPU), and a shared event (its value, a listener's notification, a
/// GPU signal after it), and leaves a pooled `Shared` buffer refused, as its notice counts. The
/// pooled buffer keeps the lost `VkDevice`, but nothing it kept for reuse: the memory of buffers
/// the program let go of before the loss (their `VkBuffer`s parked, or a block kept empty) is
/// freed, and no empty block is left. The lost `VkDevice` goes with the pooled buffer.
#[test]
fn a_recovery_carries_events_private_and_fresh_page_buffers() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    // The device's checks answered first, so that where its blocks allocated exportable passed,
    // its own buffers lie in them (`Holds::FdBuffers`), the `Private` one below among them.
    let _settled = device.settle_in_place_checks(WAIT);
    let queue = device.new_command_queue().unwrap();
    let private = device
        .new_buffer(BYTES, ResourceOptions::private())
        .unwrap();
    fill(&queue, &private, 7);
    assert_eq!(
        read(&device, &queue, &private).unwrap(),
        vec![7; BYTES as usize]
    );
    let address = private.gpu_address();
    // Which of the device's own kinds of block it lies in, both carried: a plain one, or one
    // allocated exportable and never exported, once the device's check of those passed.
    eprintln!(
        "CARRY-PRIVATE-BLOCK: {}: the `Private` buffer lies in a block allocated exportable: {}",
        device.name(),
        private.bound_memory() == crate::memory::BoundMemory::Blocks
    );
    // A buffer of the program's in fresh pages, wherever the device takes them.
    let fresh = {
        let made_on = private.device().clone();
        let fresh = made_on
            .fresh_pages_check_buffer(2 * FRESH_PAGES_PAST)
            .unwrap();
        made_on.visible(fresh)
    };
    assert!(
        fresh.in_fresh_pages(),
        "{} took no fresh pages: their carry is not exercised",
        device.name()
    );
    let pointer = fresh.contents().unwrap();
    // SAFETY: the buffer's first `BYTES` bytes, which no work in flight uses.
    unsafe { std::ptr::write_bytes(pointer.as_ptr(), 3, BYTES as usize) };
    let pooled = device.new_buffer(BYTES, ResourceOptions::shared()).unwrap();
    let event = device.new_shared_event().unwrap();
    event.set_signaled_value(5);
    let listener = SharedEventListener::new();
    let (sent, notified) = mpsc::channel();
    let notification = move |_: &SharedEvent, value: u64| {
        let _ = sent.send(value);
    };
    event.notify_listener(&listener, 9, Box::new(notification));
    // Memory the lost generation keeps for reuse: buffers the program let go of.
    let reused: Vec<Buffer> = (0..REUSED_BUFFERS)
        .map(|_| {
            device
                .new_buffer(REUSED, ResourceOptions::private())
                .unwrap()
        })
        .collect();
    drop(reused);
    let before = pooled.device().work_counts_now();
    let from = heard.lock().unwrap().len();
    device.recover_for_test().unwrap();
    let said = recovery_notices(&heard, from, "the device");
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(
        said[0].contains(
            "it carried 1 events, 1 buffers with their contents and 1 `Private` buffers (4096 \
             bytes) whose contents were lost (they read zero); left refused \
             (`MadeBeforeRecovery`): 1 pooled `Shared` or `Managed` buffers"
        ),
        "{said:?}"
    );
    // The `Private` buffer: the same buffer on the next generation, its bytes zero.
    assert!(private.device().generation_serves());
    assert_eq!(private.gpu_address(), address);
    assert_eq!(
        read(&device, &queue, &private).unwrap(),
        vec![0; BYTES as usize]
    );
    fill(&queue, &private, 9);
    assert_eq!(
        read(&device, &queue, &private).unwrap(),
        vec![9; BYTES as usize]
    );
    // The buffer in fresh pages: the same pointer, the same bytes, and the host's writes through
    // the pointer taken before the recovery reach the GPU after it.
    assert!(fresh.device().generation_serves() && fresh.in_fresh_pages());
    assert_eq!(fresh.contents(), Some(pointer));
    assert_eq!(
        read(&device, &queue, &fresh).unwrap(),
        vec![3; BYTES as usize]
    );
    // SAFETY: as above.
    unsafe { std::ptr::write_bytes(pointer.as_ptr(), 4, BYTES as usize) };
    assert_eq!(
        read(&device, &queue, &fresh).unwrap(),
        vec![4; BYTES as usize]
    );
    // The event: the value the host saw, its listener's notification, the GPU's signals after.
    assert_eq!(event.signaled_value(), 5);
    signal(&queue, &event, 9);
    assert_eq!(notified.recv_timeout(WAIT), Ok(9));
    // The pooled buffer stays the lost generation's, refused where it meets the next.
    assert!(!pooled.device().generation_serves());
    let refused = read(&device, &queue, &pooled);
    assert!(
        matches!(
            refused,
            Err(Error::NotImplemented(RuntimeGap::MadeBeforeRecovery))
        ),
        "{refused:?}"
    );
    // It holds the lost `VkDevice`, which keeps no memory for work.
    assert_eq!(live_vk_devices(&device, 2), 2);
    let after = pooled.device().work_counts_now();
    let blocks =
        |counts: &crate::WorkCounts| counts.memory_blocks_live + counts.empty_memory_blocks_live;
    eprintln!(
        "CARRY-LOST-MEMORY: {}: the lost generation held {} blocks before the loss, {} at rest, \
         {} freed as it came to rest",
        device.name(),
        blocks(&before),
        blocks(&after),
        after.memory_frees.teardown - before.memory_frees.teardown
    );
    assert_eq!(after.empty_memory_blocks_live, 0, "{after:?}");
    assert!(
        blocks(&after) < blocks(&before),
        "the lost generation holds {} blocks, {} before the loss",
        blocks(&after),
        blocks(&before)
    );
    assert!(after.memory_frees.teardown > before.memory_frees.teardown);
    drop(pooled);
    assert_eq!(live_vk_devices(&device, 1), 1);
}

/// Seven recoveries, each with the program's carried buffer and event in use between them, end
/// with one live `VkDevice`: no carried object holds a lost generation, so each goes once its
/// command buffers went. The buffer and the event serve on the last generation.
#[test]
fn seven_recoveries_end_with_one_live_vk_device() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let private = device
        .new_buffer(BYTES, ResourceOptions::private())
        .unwrap();
    let event = device.new_shared_event().unwrap();
    let began = Instant::now();
    for round in 1..=RECOVERIES {
        device.recover_for_test().unwrap();
        let byte = u8::try_from(round).unwrap();
        fill(&queue, &private, byte);
        assert_eq!(
            read(&device, &queue, &private).unwrap(),
            vec![byte; BYTES as usize]
        );
        signal(&queue, &event, round);
    }
    let counts = device.work_counts();
    assert_eq!(counts.device_recoveries, RECOVERIES);
    assert_eq!(counts.vk_devices_made, RECOVERIES + 1);
    assert_eq!(live_vk_devices(&device, 1), 1);
    eprintln!(
        "CARRY-RECOVERIES: {}: {RECOVERIES} recoveries in {:?}, {} VkDevices live at the end",
        device.name(),
        began.elapsed(),
        device.work_counts().vk_devices_live
    );
}

/// A call outside command buffers' work that reaches a carried buffer's Vulkan objects (here a
/// texture over its bytes) holds the parts it read (`Buffer::held_parts`): parked there while the
/// recovery carries the buffer and the lost generation comes to rest, it keeps the buffer's
/// parts there, which go as it ends. The buffer serves on the next generation meanwhile.
#[test]
fn a_call_under_way_keeps_the_parts_it_read_past_the_lost_generation_s_rest() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let buffer = device
        .new_buffer(BYTES, ResourceOptions::private())
        .unwrap();
    let made_on = buffer.device().clone();
    let (parked, at_hook) = mpsc::channel();
    let (go, wait) = mpsc::channel::<()>();
    let call = {
        let buffer = buffer.clone();
        std::thread::spawn(move || {
            let mut descriptor =
                TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
            descriptor.usage = TextureUsage::SHADER_READ;
            descriptor.resource_options = buffer.resource_options();
            let row = buffer
                .device()
                .minimum_linear_texture_alignment_for_pixel_format(PixelFormat::RGBA8Unorm)
                .unwrap()
                .max(16);
            // This thread's call parks inside its hold of the buffer's parts.
            crate::buffer::HELD_HOOK.with(|hook| {
                *hook.borrow_mut() = Some(Box::new(move || {
                    let _ = parked.send(());
                    let _ = wait.recv();
                }));
            });
            buffer
                .new_texture(&descriptor, 0, row)
                .map(|texture| texture.device().generation_serves())
        })
    };
    at_hook.recv_timeout(WAIT).unwrap();
    device.recover_for_test().unwrap();
    // The lost generation is at rest, none of its command buffers left, but the call holds the
    // buffer's parts there.
    assert!(made_on.shared().carry.at_rest());
    assert!(!buffer.first_parts_freed());
    // The buffer serves on the next generation meanwhile.
    fill(&queue, &buffer, 6);
    assert_eq!(
        read(&device, &queue, &buffer).unwrap(),
        vec![6; BYTES as usize]
    );
    go.send(()).unwrap();
    // The call completed on the generation it began on: its texture is that one's, where the
    // lost device made one.
    let made = call.join().unwrap();
    assert!(!matches!(made, Ok(true)), "{made:?}");
    assert!(buffer.first_parts_freed());
}
