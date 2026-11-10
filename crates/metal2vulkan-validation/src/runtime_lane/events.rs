//! Device tests of the runtime's events, shared events and scheduler on the device the cfg selects,
//! under the validation layer.
//!
//! The expected answers are Metal's, as measured on the M3 and the M4: event values only rise, a
//! command buffer waiting for an event holds back only its own queue, work before a wait runs and
//! signals while the wait is pending, and tracked buffers and textures across queues follow the
//! order the command buffers are scheduled in. Logical queues share the device's `VkQueue`s (one on
//! most devices), so a wait committed before its signal deadlocks unless the scheduler holds it
//! back. A device whose last command buffer was waited for is gone when the program's drop of its
//! last handle returns.

use super::{
    at_plan, layer_reported_nothing_at_all, open_device, open_devices, pipeline_state, public_case,
    run_with, validated_device_with, validated_plan_options, Messages, ScratchCache,
};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::DeviceOptions;
use metal2vulkan_runtime::{
    host_time, BlitCommandEncoder, BlitOption, Buffer, CommandBuffer, CommandBufferStatus,
    CommandQueue, CommitFailure, CpuCacheMode, Device, Error, EventRef, HazardTrackingMode,
    Invalid, Origin, Problem, ResourceOptions, RuntimeGap, SharedEvent, SharedEventListener, Size,
    StorageMode, Texture, TextureDescriptor,
};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The longest a command buffer or an event of these tests may take.
const WAIT: Duration = Duration::from_secs(60);

/// Large enough that work two queues ran at once would overlap on a GPU.
const BIG: u64 = 8 << 20;

/// The width and height of the texture tests' RGBA8 texture: `BIG` bytes.
const TEXTURE_WIDTH: u64 = 2048;
const TEXTURE_HEIGHT: u64 = BIG / (TEXTURE_WIDTH * 4);

/// A committed public case that runs on every venue: its pipeline gives the driver's cache data.
const CASE: &str = "threadgroup-word";

fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

fn options(storage: StorageMode, hazard: HazardTrackingMode) -> ResourceOptions {
    ResourceOptions::new(storage, CpuCacheMode::DefaultCache, hazard)
}

/// Run `test` on a validated device, then check that once everything is dropped (the device
/// last) the layer reported nothing.
fn on_device(test: impl FnOnce(&Device)) {
    on_device_with(|_| {}, test)
}

/// [`on_device`], `adjust` applied to the options of each device it makes (`at_plan` gives them
/// their queue plan).
fn on_device_with(adjust: fn(&mut DeviceOptions), test: impl FnOnce(&Device)) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
        test(&device);
    }
    layer_reported_nothing_at_all(&messages);
}

/// Wait for `command_buffer` and check it completed without error.
fn completed(command_buffer: &CommandBuffer) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The bytes of a host-visible buffer.
fn bytes_of(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().expect("a host-visible buffer");
    // SAFETY: the buffer holds `length` bytes at `contents`, alive with the buffer, and no
    // command buffer of these tests uses them while they are read.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// Fill a host-visible buffer with `byte` from the host.
fn fill_on_host(buffer: &Buffer, byte: u8) {
    let contents = buffer.contents().expect("a host-visible buffer");
    // SAFETY: as in `bytes_of`; nothing on the device uses the buffer meanwhile.
    unsafe { std::ptr::write_bytes(contents.as_ptr(), byte, buffer.length() as usize) };
}

/// A command buffer of `queue` with one blit encoder that `encode` fills.
fn blit(queue: &CommandQueue, encode: impl FnOnce(&mut BlitCommandEncoder)) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer
}

/// A private buffer of `BIG` bytes, all `byte`, filled and waited for on `queue`.
fn private_filled(
    device: &Device,
    queue: &CommandQueue,
    hazard: HazardTrackingMode,
    byte: u8,
) -> Buffer {
    let buffer = device
        .new_buffer(BIG, options(StorageMode::Private, hazard))
        .unwrap();
    let fill = blit(queue, |encoder| {
        encoder.fill_buffer(&buffer, 0..BIG, byte).unwrap()
    });
    fill.commit().unwrap();
    completed(&fill);
    buffer
}

/// A tracked private RGBA8 texture of `BIG` bytes, every byte `byte`: copied from a filled buffer
/// and waited for on `queue`.
fn private_texture_filled(device: &Device, queue: &CommandQueue, byte: u8) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(
        PixelFormat::RGBA8Unorm,
        TEXTURE_WIDTH,
        TEXTURE_HEIGHT,
        false,
    );
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = options(StorageMode::Private, HazardTrackingMode::Tracked);
    let texture = device.new_texture(&descriptor).unwrap();
    let source = private_filled(device, queue, HazardTrackingMode::Tracked, byte);
    let fill = blit(queue, |encoder| copy_into(encoder, &source, &texture));
    fill.commit().unwrap();
    completed(&fill);
    texture
}

/// Encode a copy of the whole of `source` into `texture` (rows tightly packed).
fn copy_into(encoder: &mut BlitCommandEncoder, source: &Buffer, texture: &Texture) {
    encoder
        .copy_from_buffer_to_texture(
            source,
            0,
            TEXTURE_WIDTH * 4,
            0,
            Size::new(TEXTURE_WIDTH, TEXTURE_HEIGHT, 1),
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
}

/// Encode a copy of the whole of `texture` into `destination` (rows tightly packed).
fn copy_out(encoder: &mut BlitCommandEncoder, texture: &Texture, destination: &Buffer) {
    encoder
        .copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(TEXTURE_WIDTH, TEXTURE_HEIGHT, 1),
            destination,
            0,
            TEXTURE_WIDTH * 4,
            0,
            BlitOption::None,
        )
        .unwrap();
}

/// A wait committed on one queue before the signal another queue commits later completes: the
/// scheduler submits the signal first, so the `VkQueue` they share never holds the wait ahead of
/// it.
#[test]
fn a_wait_committed_before_its_signal_never_deadlocks() {
    on_device_with(at_plan, |device| {
        let (first, second) = (
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        );
        let event = device.new_event().unwrap();
        let shared = device.new_buffer(BIG, ResourceOptions::shared()).unwrap();
        let waiting = first.command_buffer().unwrap();
        waiting
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut encoder = waiting.blit_command_encoder().unwrap();
        encoder.fill_buffer(&shared, 0..BIG / 2, 1).unwrap();
        encoder.end_encoding().unwrap();
        waiting.commit().unwrap();
        assert_eq!(waiting.status(), CommandBufferStatus::Committed);
        let signaling = second.command_buffer().unwrap();
        let mut encoder = signaling.blit_command_encoder().unwrap();
        encoder.fill_buffer(&shared, BIG / 2..BIG, 2).unwrap();
        encoder.end_encoding().unwrap();
        signaling
            .encode_signal_event(EventRef::from(&event), 1)
            .unwrap();
        signaling.commit().unwrap();
        for command_buffer in [&signaling, &waiting] {
            completed(command_buffer);
        }
        let bytes = bytes_of(&shared);
        assert!(bytes[..BIG as usize / 2].iter().all(|&byte| byte == 1));
        assert!(bytes[BIG as usize / 2..].iter().all(|&byte| byte == 2));
    });
}

/// The consumer's paging: three queues, blit-up, exec and blit-down, ordered only by events over
/// untracked buffers, each round committed downstream first.
#[test]
fn three_queues_ordered_by_events_as_paging_does() {
    on_device_with(at_plan, |device| {
        let queue = || device.new_command_queue().unwrap();
        let (up_queue, exec_queue, down_queue) = (queue(), queue(), queue());
        let (up, exec) = (device.new_event().unwrap(), device.new_event().unwrap());
        let down = device.new_shared_event().unwrap();
        let untracked = |storage| {
            device
                .new_buffer(BIG, options(storage, HazardTrackingMode::Untracked))
                .unwrap()
        };
        let staging_in = untracked(StorageMode::Shared);
        let (a, b) = (
            untracked(StorageMode::Private),
            untracked(StorageMode::Private),
        );
        let staging_out = untracked(StorageMode::Shared);
        for round in 1..=4u64 {
            // The previous round finished: the host may write the staging buffer again.
            assert!(down.wait_until_signaled_value(round - 1, WAIT));
            fill_on_host(&staging_in, round as u8);
            let down_buffer = down_queue.command_buffer().unwrap();
            down_buffer
                .encode_wait_for_event(EventRef::from(&exec), round)
                .unwrap();
            let mut encoder = down_buffer.blit_command_encoder().unwrap();
            encoder
                .copy_from_buffer_to_buffer(&b, 0, &staging_out, 0, BIG)
                .unwrap();
            encoder.end_encoding().unwrap();
            down_buffer
                .encode_signal_event(EventRef::from(&down), round)
                .unwrap();
            down_buffer.commit().unwrap();

            let exec_buffer = exec_queue.command_buffer().unwrap();
            exec_buffer
                .encode_wait_for_event(EventRef::from(&up), round)
                .unwrap();
            let mut encoder = exec_buffer.blit_command_encoder().unwrap();
            encoder
                .copy_from_buffer_to_buffer(&a, 0, &b, 0, BIG)
                .unwrap();
            encoder.fill_buffer(&b, 0..16, 0xee).unwrap();
            encoder.end_encoding().unwrap();
            exec_buffer
                .encode_signal_event(EventRef::from(&exec), round)
                .unwrap();
            exec_buffer.commit().unwrap();
            // Downstream waits: neither is scheduled before blit-up commits.
            assert_eq!(down_buffer.status(), CommandBufferStatus::Committed);
            assert_eq!(exec_buffer.status(), CommandBufferStatus::Committed);

            let up_buffer = up_queue.command_buffer().unwrap();
            let mut encoder = up_buffer.blit_command_encoder().unwrap();
            encoder
                .copy_from_buffer_to_buffer(&staging_in, 0, &a, 0, BIG)
                .unwrap();
            encoder.end_encoding().unwrap();
            up_buffer
                .encode_signal_event(EventRef::from(&up), round)
                .unwrap();
            up_buffer.commit().unwrap();

            assert!(down.wait_until_signaled_value(round, WAIT));
            for command_buffer in [&up_buffer, &exec_buffer, &down_buffer] {
                completed(command_buffer);
            }
            let bytes = bytes_of(&staging_out);
            assert!(bytes[..16].iter().all(|&byte| byte == 0xee));
            assert!(
                bytes[16..].iter().all(|&byte| byte == round as u8),
                "round {round}"
            );
        }
    });
}

/// A tracked buffer written on one queue and read on another, committed back to back in both
/// orders: the reader sees what the order the runtime schedules them in (the commit order, as on
/// the M4) gives it, and the two never overlap (synchronization validation).
#[test]
fn a_tracked_buffer_across_queues_follows_the_commit_order() {
    on_device_with(at_plan, |device| {
        let (writer_queue, reader_queue) = (
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        );
        for writer_first in [true, false] {
            let tracked = private_filled(device, &writer_queue, HazardTrackingMode::Tracked, 0x11);
            let read = device.new_buffer(BIG, ResourceOptions::shared()).unwrap();
            let writer = blit(&writer_queue, |encoder| {
                encoder.fill_buffer(&tracked, 0..BIG, 0x22).unwrap()
            });
            let reader = blit(&reader_queue, |encoder| {
                encoder
                    .copy_from_buffer_to_buffer(&tracked, 0, &read, 0, BIG)
                    .unwrap()
            });
            if writer_first {
                writer.commit().unwrap();
                reader.commit().unwrap();
            } else {
                reader.commit().unwrap();
                writer.commit().unwrap();
            }
            completed(&writer);
            completed(&reader);
            let expected = if writer_first { 0x22 } else { 0x11 };
            assert!(
                bytes_of(&read).iter().all(|&byte| byte == expected),
                "writer first: {writer_first}"
            );
        }
    });
}

/// An event orders a tracked buffer's reader against the commit order: the reader, committed
/// first, waits for the writer's signal, is scheduled after it, and reads what it wrote.
#[test]
fn an_event_orders_a_tracked_buffer_against_the_commit_order() {
    on_device_with(at_plan, |device| {
        let (writer_queue, reader_queue) = (
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        );
        let event = device.new_event().unwrap();
        let tracked = private_filled(device, &writer_queue, HazardTrackingMode::Tracked, 0x11);
        let read = device.new_buffer(BIG, ResourceOptions::shared()).unwrap();
        let reader = reader_queue.command_buffer().unwrap();
        reader
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut encoder = reader.blit_command_encoder().unwrap();
        encoder
            .copy_from_buffer_to_buffer(&tracked, 0, &read, 0, BIG)
            .unwrap();
        encoder.end_encoding().unwrap();
        reader.commit().unwrap();
        let writer = blit(&writer_queue, |encoder| {
            encoder.fill_buffer(&tracked, 0..BIG, 0x22).unwrap()
        });
        writer
            .encode_signal_event(EventRef::from(&event), 1)
            .unwrap();
        assert_eq!(reader.status(), CommandBufferStatus::Committed);
        writer.commit().unwrap();
        completed(&reader);
        completed(&writer);
        assert!(bytes_of(&read).iter().all(|&byte| byte == 0x22));
    });
}

/// A tracked texture written on one queue and read on another, committed back to back in both
/// orders: as with a buffer, the reader sees what the commit order gives it, and the two never
/// overlap (synchronization validation).
#[test]
fn a_tracked_texture_across_queues_follows_the_commit_order() {
    on_device_with(at_plan, |device| {
        let (writer_queue, reader_queue) = (
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        );
        let source = private_filled(device, &writer_queue, HazardTrackingMode::Tracked, 0x22);
        for writer_first in [true, false] {
            let tracked = private_texture_filled(device, &writer_queue, 0x11);
            let read = device.new_buffer(BIG, ResourceOptions::shared()).unwrap();
            let writer = blit(&writer_queue, |encoder| {
                copy_into(encoder, &source, &tracked)
            });
            let reader = blit(&reader_queue, |encoder| copy_out(encoder, &tracked, &read));
            if writer_first {
                writer.commit().unwrap();
                reader.commit().unwrap();
            } else {
                reader.commit().unwrap();
                writer.commit().unwrap();
            }
            completed(&writer);
            completed(&reader);
            let expected = if writer_first { 0x22 } else { 0x11 };
            assert!(
                bytes_of(&read).iter().all(|&byte| byte == expected),
                "writer first: {writer_first}"
            );
        }
    });
}

/// An event orders a tracked texture's reader against the commit order: the reader, committed
/// first, waits for the writer's signal, is scheduled after it, and reads what it wrote.
#[test]
fn an_event_orders_a_tracked_texture_against_the_commit_order() {
    on_device_with(at_plan, |device| {
        let (writer_queue, reader_queue) = (
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        );
        let event = device.new_event().unwrap();
        let source = private_filled(device, &writer_queue, HazardTrackingMode::Tracked, 0x22);
        let tracked = private_texture_filled(device, &writer_queue, 0x11);
        let read = device.new_buffer(BIG, ResourceOptions::shared()).unwrap();
        let reader = reader_queue.command_buffer().unwrap();
        reader
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut encoder = reader.blit_command_encoder().unwrap();
        copy_out(&mut encoder, &tracked, &read);
        encoder.end_encoding().unwrap();
        reader.commit().unwrap();
        let writer = blit(&writer_queue, |encoder| {
            copy_into(encoder, &source, &tracked)
        });
        writer
            .encode_signal_event(EventRef::from(&event), 1)
            .unwrap();
        assert_eq!(reader.status(), CommandBufferStatus::Committed);
        writer.commit().unwrap();
        completed(&reader);
        completed(&writer);
        assert!(bytes_of(&read).iter().all(|&byte| byte == 0x22));
    });
}

/// Two queues signal one event with 7 and 4, in both submission orders: the event ends at 7, as
/// Metal's did whichever signal executed last (measured on the M3 and the M4).
#[test]
fn signals_of_one_event_from_two_queues_end_at_the_highest() {
    on_device_with(at_plan, |device| {
        let (first, second) = (
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        );
        let scratch = device.new_buffer(BIG, ResourceOptions::private()).unwrap();
        for (a, b) in [(7, 4), (4, 7)] {
            let event = device.new_shared_event().unwrap();
            let signal = |queue: &CommandQueue, value| {
                let command_buffer = blit(queue, |encoder| {
                    encoder.fill_buffer(&scratch, 0..BIG, value as u8).unwrap()
                });
                command_buffer
                    .encode_signal_event(EventRef::from(&event), value)
                    .unwrap();
                command_buffer.commit().unwrap();
                command_buffer
            };
            let (one, two) = (signal(&first, a), signal(&second, b));
            completed(&one);
            completed(&two);
            assert_eq!(event.signaled_value(), 7, "submitted {a} then {b}");
        }
    });
}

/// A shared event's value only rises, from the host and the GPU alike; host waits answer within
/// their timeout (measured on Metal).
#[test]
fn a_shared_events_value_only_rises() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let event = device.new_shared_event().unwrap();
        event.set_label("rises");
        assert_eq!(event.label().as_deref(), Some("rises"));
        assert_eq!(event.signaled_value(), 0);
        event.set_signaled_value(5);
        event.set_signaled_value(3);
        event.set_signaled_value(5);
        assert_eq!(event.signaled_value(), 5);
        for value in [3, 5] {
            let command_buffer = queue.command_buffer().unwrap();
            command_buffer
                .encode_signal_event(EventRef::from(&event), value)
                .unwrap();
            command_buffer.commit().unwrap();
            completed(&command_buffer);
            assert_eq!(event.signaled_value(), 5);
        }
        assert!(event.wait_until_signaled_value(3, Duration::from_millis(10)));
        assert!(event.wait_until_signaled_value(5, Duration::from_millis(10)));
        assert!(!event.wait_until_signaled_value(6, Duration::from_millis(10)));
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_signal_event(EventRef::from(&event), u64::MAX)
            .unwrap();
        command_buffer.commit().unwrap();
        assert!(event.wait_until_signaled_value(u64::MAX, WAIT));
        completed(&command_buffer);
    });
}

/// A command buffer waiting for an event stays committed and holds back the later command
/// buffers of its queue, not those of another queue; its scheduled handler runs once the host
/// signals, before its completed handler (as measured on Metal).
#[test]
fn a_held_command_buffer_holds_back_only_its_own_queue() {
    on_device_with(at_plan, |device| {
        let (held_queue, other_queue) = (
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        );
        let event = device.new_shared_event().unwrap();
        let scratch = device.new_buffer(BIG, ResourceOptions::private()).unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let held = held_queue.command_buffer().unwrap();
        held.encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut encoder = held.blit_command_encoder().unwrap();
        encoder.fill_buffer(&scratch, 0..BIG, 1).unwrap();
        encoder.end_encoding().unwrap();
        for (name, scheduled) in [("scheduled", true), ("completed", false)] {
            let calls = Arc::clone(&calls);
            let handler = Box::new(move |_: &CommandBuffer| {
                calls.lock().unwrap().push(name);
            });
            if scheduled {
                held.add_scheduled_handler(handler).unwrap();
            } else {
                held.add_completed_handler(handler).unwrap();
            }
        }
        held.commit().unwrap();
        assert_eq!(
            held.add_scheduled_handler(Box::new(|_| {})),
            Err(invalid(
                "add_scheduled_handler",
                Problem::HandlerAfterCommit
            ))
        );
        let behind = blit(&held_queue, |encoder| {
            encoder.fill_buffer(&scratch, 0..BIG, 2).unwrap()
        });
        behind.commit().unwrap();
        let other = blit(&other_queue, |encoder| {
            encoder.fill_buffer(&scratch, 0..16, 3).unwrap()
        });
        other.commit().unwrap();
        completed(&other);
        assert_eq!(held.status(), CommandBufferStatus::Committed);
        assert_eq!(behind.status(), CommandBufferStatus::Committed);
        assert!(calls.lock().unwrap().is_empty());
        event.set_signaled_value(1);
        held.wait_until_scheduled().unwrap();
        completed(&held);
        completed(&behind);
        assert_eq!(*calls.lock().unwrap(), ["scheduled", "completed"]);
    });
}

/// The work before a wait runs and signals while the wait is pending, and the command buffer
/// stays committed until its last batch goes (as measured on Metal).
#[test]
fn a_signal_before_a_wait_lands_while_the_wait_is_pending() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let shared = || device.new_shared_event().unwrap();
        let (before, gate, after) = (shared(), shared(), shared());
        let scratch = device.new_buffer(BIG, ResourceOptions::private()).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        encoder.fill_buffer(&scratch, 0..BIG, 1).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer
            .encode_signal_event(EventRef::from(&before), 1)
            .unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&gate), 1)
            .unwrap();
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        encoder.fill_buffer(&scratch, 0..BIG, 2).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer
            .encode_signal_event(EventRef::from(&after), 1)
            .unwrap();
        command_buffer.commit().unwrap();
        assert!(before.wait_until_signaled_value(1, WAIT));
        assert_eq!(after.signaled_value(), 0);
        assert_eq!(command_buffer.status(), CommandBufferStatus::Committed);
        gate.set_signaled_value(1);
        completed(&command_buffer);
        assert_eq!(after.signaled_value(), 1);
    });
}

/// A listener's notification for a value already reached comes at once, and one for a later
/// value once a command buffer signals it, with the value asked for (as measured on Metal).
#[test]
fn listeners_hear_reached_and_later_values() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let event = device.new_shared_event().unwrap();
        let listener = SharedEventListener::new();
        let (send, receive) = mpsc::channel();
        event.set_signaled_value(3);
        for (name, at) in [("passed", 2), ("later", 5)] {
            let send = send.clone();
            event.notify_listener(
                &listener,
                at,
                Box::new(move |event: &SharedEvent, value| {
                    let _ = send.send((name, value, event.signaled_value()));
                }),
            );
        }
        assert_eq!(receive.recv_timeout(WAIT), Ok(("passed", 2, 3)));
        assert!(receive.try_recv().is_err());
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_signal_event(EventRef::from(&event), 6)
            .unwrap();
        command_buffer.commit().unwrap();
        assert_eq!(receive.recv_timeout(WAIT), Ok(("later", 5, 6)));
        completed(&command_buffer);
    });
}

/// An in-process handle opens the same event: values set through either show through both, and
/// a GPU wait on one is met by a signal of the other. Another device's handle is refused.
#[test]
fn shared_event_handles_open_the_same_event() {
    on_device_with(at_plan, |device| {
        let queue = device.new_command_queue().unwrap();
        let event = device.new_shared_event().unwrap();
        event.set_label("paging");
        event.set_signaled_value(4);
        let handle = event.new_shared_event_handle();
        assert_eq!(handle.label().as_deref(), Some("paging"));
        let opened = device.new_shared_event_with_handle(&handle).unwrap();
        assert_eq!(opened.signaled_value(), 4);
        opened.set_signaled_value(9);
        assert_eq!(event.signaled_value(), 9);
        let waiting = queue.command_buffer().unwrap();
        waiting
            .encode_wait_for_event(EventRef::from(&opened), 10)
            .unwrap();
        waiting.commit().unwrap();
        let signaling = device
            .new_command_queue()
            .unwrap()
            .command_buffer()
            .unwrap();
        signaling
            .encode_signal_event(EventRef::from(&event), 10)
            .unwrap();
        signaling.commit().unwrap();
        completed(&waiting);
        completed(&signaling);

        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        {
            let other =
                validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
            assert_eq!(
                other.new_shared_event_with_handle(&handle).err(),
                Some(Error::NotImplemented(RuntimeGap::SharedEventAcrossDevices))
            );
            // Nor does a command buffer take another device's event.
            let command_buffer = other.new_command_queue().unwrap().command_buffer().unwrap();
            assert_eq!(
                command_buffer.encode_signal_event(EventRef::from(&event), 11),
                Err(invalid("encode_signal_event", Problem::ForeignObject))
            );
        }
        layer_reported_nothing_at_all(&messages);
    });
}

/// Two calls with equal options make one device, as Metal's default device made twice is one
/// object and the first of all its devices that object (the probe's `dev_identity`, an M3 Max on
/// macOS 26.7 and an M4 on macOS 27.0.1, by pointer and `registryID`): the default device made
/// twice and the first of every device (`open_device`, `open_devices`) share their Vulkan device,
/// an event's handle opens on the second call's device, and its command buffers wait for and signal
/// the first's event. Once every handle drops, the device tears down, and a later call with the
/// same options makes a device that works. (A device made with other options is another device:
/// `shared_event_handles_open_the_same_event`.)
#[test]
fn equal_options_make_one_device() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let options = validated_plan_options(&messages, cache.path());
    {
        let first = open_device(TEST_DEVICE, options.clone()).unwrap();
        let second = open_device(TEST_DEVICE, options.clone()).unwrap();
        let all = open_devices(TEST_DEVICE, options.clone()).unwrap();
        // SAFETY: the handles are only read, while the devices live.
        let vulkan = |device: &Device| unsafe { device.raw() }.device.handle();
        assert_eq!(vulkan(&first), vulkan(&second));
        assert_eq!(vulkan(&first), vulkan(&all[0]));
        assert_eq!(first.registry_id(), second.registry_id());
        let event = first.new_shared_event().unwrap();
        let opened = second
            .new_shared_event_with_handle(&event.new_shared_event_handle())
            .unwrap();
        let waiting = second
            .new_command_queue()
            .unwrap()
            .command_buffer()
            .unwrap();
        waiting
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        waiting.commit().unwrap();
        let signaling = first.new_command_queue().unwrap().command_buffer().unwrap();
        signaling
            .encode_signal_event(EventRef::from(&opened), 1)
            .unwrap();
        signaling.commit().unwrap();
        completed(&waiting);
        completed(&signaling);
        assert_eq!(event.signaled_value(), 1);
    }
    {
        let again = open_device(TEST_DEVICE, options.clone()).unwrap();
        let event = again.new_shared_event().unwrap();
        let command_buffer = again.new_command_queue().unwrap().command_buffer().unwrap();
        command_buffer
            .encode_signal_event(EventRef::from(&event), 2)
            .unwrap();
        command_buffer.commit().unwrap();
        completed(&command_buffer);
        assert_eq!(event.signaled_value(), 2);
    }
    layer_reported_nothing_at_all(&messages);
}

/// Event waits and signals are refused where Metal stops them: with an encoder open (Metal
/// asserts) and after commit; a wait for 0, and a command buffer with nothing encoded, complete.
#[test]
fn event_commands_are_refused_where_metal_stops_them() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let event = device.new_event().unwrap();
        let scratch = device.new_buffer(64, ResourceOptions::private()).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        encoder.fill_buffer(&scratch, 0..64, 1).unwrap();
        assert_eq!(
            command_buffer.encode_signal_event(EventRef::from(&event), 1),
            Err(invalid("encode_signal_event", Problem::EncoderOpen))
        );
        assert_eq!(
            command_buffer.encode_wait_for_event(EventRef::from(&event), 1),
            Err(invalid("encode_wait_for_event", Problem::EncoderOpen))
        );
        encoder.end_encoding().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 0)
            .unwrap();
        command_buffer.commit().unwrap();
        assert_eq!(
            command_buffer.encode_signal_event(EventRef::from(&event), 1),
            Err(invalid("encode_signal_event", Problem::AlreadyCommitted))
        );
        completed(&command_buffer);
        let empty = queue.command_buffer().unwrap();
        empty.commit().unwrap();
        completed(&empty);
    });
}

/// The consumer's signal-when-scheduled, composed: a scheduled handler that sets a shared event,
/// which another queue's command buffer waits for.
#[test]
fn a_scheduled_handler_signals_a_shared_event() {
    on_device_with(at_plan, |device| {
        let (first, second) = (
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        );
        let gate = device.new_shared_event().unwrap();
        let scratch = device.new_buffer(BIG, ResourceOptions::private()).unwrap();
        let waiting = second.command_buffer().unwrap();
        waiting
            .encode_wait_for_event(EventRef::from(&gate), 1)
            .unwrap();
        let mut encoder = waiting.blit_command_encoder().unwrap();
        encoder.fill_buffer(&scratch, 0..BIG, 2).unwrap();
        encoder.end_encoding().unwrap();
        waiting.commit().unwrap();
        let scheduled = blit(&first, |encoder| {
            encoder.fill_buffer(&scratch, 0..BIG, 1).unwrap()
        });
        let signal = gate.clone();
        scheduled
            .add_scheduled_handler(Box::new(move |_: &CommandBuffer| {
                signal.set_signaled_value(1);
            }))
            .unwrap();
        scheduled.commit().unwrap();
        completed(&scheduled);
        completed(&waiting);
        assert_eq!(gate.signaled_value(), 1);
    });
}

/// A command buffer's scheduled signal (`encode_signal_event_scheduled`) sets its value before the
/// status reads `Scheduled`, before every scheduled handler runs (one added before it too) and
/// before `wait_until_scheduled` returns, as Metal's does (an M3 Max, macOS 27.0.1), and releases
/// another queue's wait for the value. A value below the event's changes nothing, and two signals
/// end at the higher. One held for an event the host has not set is not scheduled, and its signal
/// waits with it. Encoded with an encoder open it is refused (Metal asserts); after the commit it
/// does nothing, as Metal's, until the command buffer completed, from which it is refused (Metal's
/// process ends).
#[test]
fn a_scheduled_signal_sets_its_value_as_the_command_buffer_is_scheduled() {
    on_device_with(at_plan, |device| {
        let (first, second) = (
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        );
        // Before the status and every scheduled handler.
        let event = device.new_shared_event().unwrap();
        let command_buffer = first.command_buffer().unwrap();
        let (send, receive) = mpsc::channel();
        for name in ["added before", "added after"] {
            if name == "added after" {
                command_buffer
                    .encode_signal_event_scheduled(&event, 5)
                    .unwrap();
            }
            let (send, event) = (send.clone(), event.clone());
            command_buffer
                .add_scheduled_handler(Box::new(move |buffer: &CommandBuffer| {
                    let _ = send.send((name, event.signaled_value(), buffer.status()));
                }))
                .unwrap();
        }
        command_buffer.commit().unwrap();
        command_buffer.wait_until_scheduled().unwrap();
        assert_eq!(
            event.signaled_value(),
            5,
            "at wait_until_scheduled's return"
        );
        let scheduled = CommandBufferStatus::Scheduled;
        assert_eq!(
            receive.recv_timeout(WAIT),
            Ok(("added before", 5, scheduled))
        );
        assert_eq!(
            receive.recv_timeout(WAIT),
            Ok(("added after", 5, scheduled))
        );
        completed(&command_buffer);

        // Releasing a wait on another queue.
        let gate = device.new_shared_event().unwrap();
        let scratch = device.new_buffer(BIG, ResourceOptions::private()).unwrap();
        let waiting = second.command_buffer().unwrap();
        waiting
            .encode_wait_for_event(EventRef::from(&gate), 2)
            .unwrap();
        let mut encoder = waiting.blit_command_encoder().unwrap();
        encoder.fill_buffer(&scratch, 0..BIG, 2).unwrap();
        encoder.end_encoding().unwrap();
        waiting.commit().unwrap();
        let signalling = blit(&first, |encoder| {
            encoder.fill_buffer(&scratch, 0..BIG, 1).unwrap()
        });
        signalling.encode_signal_event_scheduled(&gate, 2).unwrap();
        signalling.commit().unwrap();
        completed(&signalling);
        completed(&waiting);
        assert_eq!(gate.signaled_value(), 2);

        // Values only rise.
        let floor = device.new_shared_event().unwrap();
        floor.set_signaled_value(10);
        let rising = device.new_shared_event().unwrap();
        let command_buffer = first.command_buffer().unwrap();
        command_buffer
            .encode_signal_event_scheduled(&floor, 4)
            .unwrap();
        for value in [6, 3] {
            command_buffer
                .encode_signal_event_scheduled(&rising, value)
                .unwrap();
        }
        command_buffer.commit().unwrap();
        completed(&command_buffer);
        assert_eq!((floor.signaled_value(), rising.signaled_value()), (10, 6));

        // Held for an event: not scheduled, and nor is its signal.
        let held_for = device.new_shared_event().unwrap();
        let later = device.new_shared_event().unwrap();
        let held = first.command_buffer().unwrap();
        held.encode_wait_for_event(EventRef::from(&held_for), 1)
            .unwrap();
        held.encode_signal_event_scheduled(&later, 3).unwrap();
        held.commit().unwrap();
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(held.status(), CommandBufferStatus::Committed);
        assert_eq!(later.signaled_value(), 0);
        held_for.set_signaled_value(1);
        completed(&held);
        assert_eq!(later.signaled_value(), 3);

        // Refused with an encoder open; after the commit, nothing. The command buffer waits for
        // an event the host signals only after the call, so it cannot complete first (a call
        // after its completion is refused, below).
        let release = device.new_shared_event().unwrap();
        let command_buffer = first.command_buffer().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&release), 1)
            .unwrap();
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        encoder.fill_buffer(&scratch, 0..64, 1).unwrap();
        assert_eq!(
            command_buffer.encode_signal_event_scheduled(&later, 9),
            Err(invalid(
                "encode_signal_event_scheduled",
                Problem::EncoderOpen
            ))
        );
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        assert_eq!(
            command_buffer.encode_signal_event_scheduled(&later, 9),
            Ok(())
        );
        release.set_signaled_value(1);
        completed(&command_buffer);
        assert_eq!(later.signaled_value(), 3);
        // Once it completed, refused: Metal's process ends there.
        assert_eq!(
            command_buffer.encode_signal_event_scheduled(&later, 9),
            Err(invalid(
                "encode_signal_event_scheduled",
                Problem::ScheduledSignalAfterCompletion
            ))
        );
        assert_eq!(later.signaled_value(), 3);
        // As once it failed (status `Error`): Metal's process ends there too.
        let failed = first.command_buffer().unwrap();
        failed.commit_failed(CommitFailure::new(8)).unwrap();
        failed.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(failed.status(), CommandBufferStatus::Error);
        assert_eq!(
            failed.encode_signal_event_scheduled(&later, 9),
            Err(invalid(
                "encode_signal_event_scheduled",
                Problem::ScheduledSignalAfterCompletion
            ))
        );
        assert_eq!(later.signaled_value(), 3);
    });
}

/// A program that waits for its last command buffer and then drops its last handles destroys the
/// device on its own thread, before the drop returns. The device's instance holds the round's
/// message list (its debug messenger's sink) until the instance is destroyed, the teardown's last
/// step, so once the drop returned the test is the list's only holder, whatever the driver (the
/// instance the runtime keeps for the drivers is bare and holds no list).
/// Each making wrote its pipeline cache record as it made its pipeline (The device's one driver
/// cache was saved a last time at its teardown), so the device's records are on disk before its
/// drop; a driver whose data ends inside its own header (lavapipe on Mesa 26.2) keeps none, and its
/// makings count that reason instead, as `runtime_lane::cache` does.
///
/// Each round ends with an empty command buffer, after everything else of the round's device
/// went: the program's last drops (the command buffer, its queue, the device) then only release
/// references, so they race the runtime's threads to the device's last one. Even rounds complete
/// it in a completion turn; odd rounds give it a scheduled handler that outlasts its
/// execution, so the scheduler's thread completes it. A runtime thread that kept a reference past
/// the waiter's return would destroy the device itself, perhaps while the process exits (which
/// RADV ended with `SIGSEGV`).
///
/// Printed: `WAIT-AFTER-HANDLER`, how long after the last command buffer's completed handler ran
/// (or after the wait was asked, if later) its wait returned, each round and as the rounds' median
/// and longest. Metal's wait for a command buffer's completion returns after its completed handlers
/// ended, 10 us after at the median and 23 us at p99 on the M3 Max.
#[test]
fn a_device_dropped_after_a_wait_is_gone_when_the_drop_returns() {
    let case = public_case(CASE).unwrap();
    let mut after_handler = Vec::new();
    for round in 0..16 {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        let first = {
            let device =
                validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
            {
                let pipeline = pipeline_state(&device, &case).unwrap();
                run_with(&device, &case, &pipeline).unwrap();
            }
            let first = device.shader_cache_statistics();
            let queue = device.new_command_queue().unwrap();
            let last = queue.command_buffer().unwrap();
            if round % 2 == 1 {
                last.add_scheduled_handler(Box::new(|_: &CommandBuffer| {
                    std::thread::sleep(Duration::from_millis(20));
                }))
                .unwrap();
            }
            let handled = Arc::new(Mutex::new(None));
            let noted = Arc::clone(&handled);
            last.add_completed_handler(Box::new(move |_: &CommandBuffer| {
                *noted.lock().unwrap() = Some(host_time());
            }))
            .unwrap();
            last.commit().unwrap();
            let asked = host_time();
            last.wait_until_completed_for(WAIT).unwrap();
            let returned = host_time();
            completed(&last);
            let handled = handled.lock().unwrap().expect("the completed handler ran");
            let us = (returned - handled.max(asked)) * 1e6;
            eprintln!("WAIT-AFTER-HANDLER round {round}: {us:.1} us");
            after_handler.push(us);
            first
        };
        assert_eq!(
            messages.holders(),
            1,
            "round {round}: the first device was not destroyed when its drop returned"
        );
        if first.driver_data_truncated == 0 {
            assert!(
                first.driver_saves > 0,
                "round {round}: the first device's makings wrote no record: {first:?}"
            );
        } else {
            // The driver's data ends inside its own header: nothing was kept.
            assert_eq!(first.driver_saves, 0, "round {round}: {first:?}");
        }
        let next = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        drop(next);
        layer_reported_nothing_at_all(&messages);
    }
    after_handler.sort_by(f64::total_cmp);
    eprintln!(
        "WAIT-AFTER-HANDLER rounds {}: median {:.1} us, longest {:.1} us",
        after_handler.len(),
        after_handler[after_handler.len() / 2],
        after_handler[after_handler.len() - 1]
    );
}
