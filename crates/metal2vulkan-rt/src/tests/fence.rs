//! Fences, under a device cfg on every device of the cfg's kind: an encoder waiting on a fence
//! after another command buffer's encoder updated it sees that command buffer's writes, and the
//! fence calls of a compute and a blit encoder record no device work (the same submissions,
//! barriers, bytes copied and passes as the same work without them). Across queues: a fence any
//! queue's encoders update and wait on is taken; a wait on another queue goes after an update the
//! submit thread holds in a group, a wait with nothing for the GPU completes after it, and so does
//! a wait held at an event while an update committed after it is held so, a render encoder's
//! wait among them; a wait does not wait for an update held at an event wait or behind an
//! uncommitted enqueue, as on an M3 Max, nor on a fence never updated.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::blit::on_device::settled;
    use crate::tests::command::on_device::{behind_busy, each_device, finished, within, word};
    use crate::tests::draw::on_device::{pass, target};
    use crate::{
        Buffer, BufferLayout, CommandBuffer, CommandBufferStatus, CommandQueue, Device,
        DispatchType, Fence, MemoryClass, Metrics, PixelFormat, SharedEvent, TextureDescriptor,
        TexturePlace, TextureType, TextureUsage,
    };
    use std::thread;
    use std::time::Duration;

    /// `command_buffer` committed and complete with no error.
    fn run(name: &str, command_buffer: &CommandBuffer) {
        command_buffer.commit().expect("a commit");
        finished(name, command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// The submissions, barriers, bytes copied and passes from `before` to `after`.
    fn delta(before: &Metrics, after: &Metrics) -> [u64; 4] {
        [
            after.submissions - before.submissions,
            after.barriers - before.barriers,
            after.bytes_copied - before.bytes_copied,
            after.passes - before.passes,
        ]
    }

    #[test]
    fn every_device_orders_a_wait_after_an_update_and_records_nothing_for_either() {
        each_device("fences", |name, _, device| {
            let block = (device.take_block(MemoryClass::Shared, 4096)).expect("a Shared block");
            let filled = (device.new_buffer(block.id, 0, 64, None)).expect("a buffer");
            let out = (device.new_buffer(block.id, 256, 64, None)).expect("a buffer");
            let mut d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 4, 4);
            d.usage = TextureUsage::RENDER_TARGET;
            let texture = device.new_texture_for_views(&d).expect("a texture");
            let queue = device.new_command_queue(4).expect("a queue");
            let layout = BufferLayout {
                offset: 0,
                bytes_per_row: 16,
                bytes_per_image: 0,
            };
            let counters = device.counters();
            // The first command buffer fills `filled` and updates the fence from a blit and a
            // compute encoder; the second waits on it from both, then copies `filled` through the
            // texture into `out`. Both are committed before either completes.
            let pair = |value: u8, fence: Option<&Fence>| {
                let first = queue.command_buffer().expect("a command buffer");
                let mut blit = first.blit_command_encoder().expect("an encoder");
                blit.fill_buffer(&filled, 0..64, value).expect("a fill");
                if let Some(fence) = fence {
                    blit.update_fence(fence).expect("an update");
                }
                blit.end_encoding();
                let mut compute =
                    (first.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
                if let Some(fence) = fence {
                    compute.update_fence(fence).expect("an update");
                }
                compute.end_encoding();
                let second = queue.command_buffer().expect("a command buffer");
                let mut compute =
                    (second.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
                if let Some(fence) = fence {
                    compute.wait_for_fence(fence).expect("a wait");
                }
                compute.end_encoding();
                let mut blit = second.blit_command_encoder().expect("an encoder");
                if let Some(fence) = fence {
                    blit.wait_for_fence(fence).expect("a wait");
                }
                let size = [4, 4, 1];
                (blit.copy_from_buffer(&filled, layout, size, &texture, TexturePlace::default()))
                    .expect("a copy");
                (blit.copy_from_texture_to_buffer(
                    &texture,
                    TexturePlace::default(),
                    size,
                    &out,
                    layout,
                ))
                .expect("a copy");
                blit.end_encoding();
                first.commit().expect("a commit");
                run(name, &second);
                finished(name, &first);
                assert_eq!(first.error(), None, "{name}");
            };
            // Once to move the texture home, outside the counts.
            pair(1, None);
            let fence = device.new_fence();
            settled(&device);
            let before = counters.metrics();
            pair(0x5a, Some(&fence));
            let with = delta(&before, &counters.metrics());
            let contents = out.contents().expect("a Shared buffer");
            // SAFETY: the command buffer that wrote the buffer completed, and its writes reach
            // the host.
            let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), 64) };
            assert!(
                bytes.iter().all(|&b| b == 0x5a),
                "{name}: the wait sees the fill"
            );
            settled(&device);
            let before = counters.metrics();
            pair(0x5b, None);
            let without = delta(&before, &counters.metrics());
            assert_eq!(
                with, without,
                "{name}: submissions, barriers, bytes copied and passes with the fence and without"
            );
            other_queue(name, &device, &queue);
        });
    }

    /// A fence `queue` used first is taken by another queue's encoders, a compute encoder's wait
    /// and a blit encoder's update, and so is one another queue used first.
    fn other_queue(name: &str, device: &Device, queue: &CommandQueue) {
        let fence = device.new_fence();
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut blit = command_buffer.blit_command_encoder().expect("an encoder");
        blit.update_fence(&fence).expect("the first queue's update");
        blit.end_encoding();
        run(name, &command_buffer);
        let second = device.new_command_queue(1).expect("a queue");
        let command_buffer = second.command_buffer().expect("a command buffer");
        let mut compute =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        compute
            .wait_for_fence(&fence)
            .expect("a compute encoder of a second queue waits");
        compute.end_encoding();
        let mut blit = command_buffer.blit_command_encoder().expect("an encoder");
        blit.update_fence(&fence)
            .expect("a blit encoder of a second queue updates");
        let own = device.new_fence();
        blit.update_fence(&own)
            .expect("a fence the second queue used first");
        blit.end_encoding();
        run(name, &command_buffer);
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut blit = command_buffer.blit_command_encoder().expect("an encoder");
        blit.wait_for_fence(&own)
            .expect("the first queue waits on the second's fence");
        blit.end_encoding();
        run(name, &command_buffer);
    }

    /// Two buffers in one Shared block: what queue A writes, filled with 0 from the host, and
    /// where queue B copies it, filled with 0x11.
    fn written_and_copy(device: &Device) -> (Buffer, Buffer) {
        let block = (device.take_block(MemoryClass::Shared, 4096)).expect("a Shared block");
        let written = (device.new_buffer(block.id, 0, 64, None)).expect("a buffer");
        let copy = (device.new_buffer(block.id, 256, 64, None)).expect("a buffer");
        for (buffer, value) in [(&written, 0), (&copy, 0x11)] {
            let contents = buffer.contents().expect("a Shared buffer");
            // SAFETY: no command buffer uses the buffer yet, and it is 64 bytes long.
            unsafe { contents.as_ptr().write_bytes(value, 64) };
        }
        (written, copy)
    }

    /// On `queue`: fills `written` with 0x5a and updates `fence`, committed.
    fn update(queue: &CommandQueue, written: &Buffer, fence: &Fence) -> CommandBuffer {
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut blit = command_buffer.blit_command_encoder().expect("an encoder");
        blit.fill_buffer(written, 0..64, 0x5a).expect("a fill");
        blit.update_fence(fence).expect("an update");
        blit.end_encoding();
        command_buffer.commit().expect("a commit");
        command_buffer
    }

    /// On `queue`: waits for `event`'s value 1 where one is given, then on `fence`, then copies
    /// `written` into `copy`, not yet committed.
    fn wait(
        queue: &CommandQueue,
        written: &Buffer,
        copy: &Buffer,
        fence: &Fence,
        event: Option<&SharedEvent>,
    ) -> CommandBuffer {
        let command_buffer = queue.command_buffer().expect("a command buffer");
        if let Some(event) = event {
            (command_buffer.encode_wait_for_event(event.as_event(), 1)).expect("an event wait");
        }
        let mut blit = command_buffer.blit_command_encoder().expect("an encoder");
        blit.wait_for_fence(fence).expect("a wait");
        (blit.copy_from_buffer_to_buffer(written, 0, copy, 0, 64)).expect("a copy");
        blit.end_encoding();
        command_buffer
    }

    /// Waits until `command_buffer` joined a submission or a group: it is no longer committed.
    fn staged(name: &str, command_buffer: &CommandBuffer) {
        within(name, || {
            command_buffer.status() != CommandBufferStatus::Committed
        });
    }

    /// An update on queue A held in a group behind as many busy submissions of A as it keeps in
    /// flight, which wait for the host, and a wait on queue B, which has nothing in flight:
    /// `body` commits both and returns them, the update first. Once the busy work is let go, the
    /// wait completes, the update having completed before it.
    fn behind_held_update(
        name: &str,
        device: &Device,
        body: impl FnOnce(&CommandQueue, &CommandQueue) -> (CommandBuffer, CommandBuffer),
    ) {
        let a = device.new_command_queue(8).expect("a queue");
        let b = device.new_command_queue(8).expect("a queue");
        behind_busy(name, device, &a, |busy, _, _| {
            let (updater, waiter) = body(&a, &b);
            busy.open();
            finished(name, &waiter);
            assert_eq!(
                updater.status(),
                CommandBufferStatus::Completed,
                "{name}: the update completed before the wait"
            );
            finished(name, &updater);
        });
    }

    /// An update on queue A that the submit thread holds in a group, behind A's submissions in
    /// flight, which Metal would have submitted at its commit: a wait on queue B committed after
    /// it, staged at once as B has nothing in flight, reads its write. Without the held group
    /// going first, the wait's submission goes ahead of it and reads the words before the fill.
    #[test]
    fn every_device_orders_a_wait_on_another_queue_after_an_update_held_in_a_group() {
        each_device("fence_group", |name, _, device| {
            let (written, copy) = written_and_copy(&device);
            let fence = device.new_fence();
            behind_held_update(name, &device, |a, b| {
                let updater = update(a, &written, &fence);
                staged(name, &updater);
                let waiter = wait(b, &written, &copy, &fence, None);
                waiter.commit().expect("a commit");
                staged(name, &waiter);
                (updater, waiter)
            });
            assert_eq!(word(&copy), 0x5a5a_5a5a, "{name}: the wait reads the fill");
        });
    }

    /// A wait on queue B held at a shared event's value, committed before an update on queue A
    /// that then joins a group A holds, and released after: Metal evaluates a wait at its kernel
    /// submission, which comes once the event's value is reached, so it waits for the update
    /// submitted by then. The held group goes first and the wait reads the update's write.
    #[test]
    fn every_device_orders_a_wait_released_from_an_event_after_an_update_held_in_a_group_since() {
        each_device("fence_group_later", |name, _, device| {
            let (written, copy) = written_and_copy(&device);
            let fence = device.new_fence();
            let event = device.new_shared_event().expect("a shared event");
            behind_held_update(name, &device, |a, b| {
                let waiter = wait(b, &written, &copy, &fence, Some(&event));
                waiter.commit().expect("a commit");
                let updater = update(a, &written, &fence);
                staged(name, &updater);
                event.set_signaled_value(1);
                staged(name, &waiter);
                (updater, waiter)
            });
            assert_eq!(word(&copy), 0x5a5a_5a5a, "{name}: the wait reads the fill");
        });
    }

    /// A wait on queue B from a render encoder, whose command buffer then copies what queue A's
    /// update wrote in a blit encoder that waits on nothing: the render encoder's wait orders the
    /// command buffer after the update A holds in a group.
    #[test]
    fn every_device_orders_a_render_encoders_wait_on_another_queue_after_an_update_held_in_a_group()
    {
        each_device("fence_group_render", |name, _, device| {
            let (written, copy) = written_and_copy(&device);
            let fence = device.new_fence();
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            behind_held_update(name, &device, |a, b| {
                let updater = update(a, &written, &fence);
                staged(name, &updater);
                let waiter = b.command_buffer().expect("a command buffer");
                let mut render =
                    (waiter.render_command_encoder(&pass(&color, None))).expect("an encoder");
                render.wait_for_fence(&fence).expect("a wait");
                render.end_encoding();
                let mut blit = waiter.blit_command_encoder().expect("an encoder");
                (blit.copy_from_buffer_to_buffer(&written, 0, &copy, 0, 64)).expect("a copy");
                blit.end_encoding();
                waiter.commit().expect("a commit");
                staged(name, &waiter);
                (updater, waiter)
            });
            assert_eq!(word(&copy), 0x5a5a_5a5a, "{name}: the copy reads the fill");
        });
    }

    /// A wait on a fence no encoder updated waits for nothing, while queue A holds in a group an
    /// update of another fence: on queue B, an encoder that only waits on it completes at once,
    /// and a command buffer with work after the wait goes in one submission of its own, A's group
    /// staying held.
    #[test]
    fn every_device_runs_a_wait_on_a_fence_never_updated_at_once() {
        each_device("fence_never_updated", |name, _, device| {
            let (written, copy) = written_and_copy(&device);
            let updated = device.new_fence();
            let never = device.new_fence();
            let counters = device.counters();
            let a = device.new_command_queue(8).expect("a queue");
            let b = device.new_command_queue(8).expect("a queue");
            behind_busy(name, &device, &a, |busy, _, _| {
                let updater = update(&a, &written, &updated);
                staged(name, &updater);
                let empty = b.command_buffer().expect("a command buffer");
                let mut compute =
                    (empty.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
                compute.wait_for_fence(&never).expect("a wait");
                compute.end_encoding();
                run(name, &empty);
                let before = counters.metrics().submissions;
                let waiter = wait(&b, &written, &copy, &never, None);
                waiter.commit().expect("a commit");
                staged(name, &waiter);
                thread::sleep(Duration::from_millis(50));
                assert_eq!(
                    counters.metrics().submissions - before,
                    1,
                    "{name}: the wait's submission alone, the group A holds held"
                );
                busy.open();
                finished(name, &waiter);
                finished(name, &updater);
            });
        });
    }

    /// A wait on queue B with nothing for the GPU (an encoder that only waits) after an update on
    /// queue A held in a group: it completes no sooner than the update, as Metal's does (job
    /// m3-max-52194: its completed handler 0.11 to 0.16 ms after the update's GPU end on an M3
    /// Max). It takes no submission, so without the fence it would complete at once.
    #[test]
    fn every_device_completes_an_empty_wait_on_another_queue_after_the_update() {
        each_device("fence_empty_wait", |name, _, device| {
            let (written, _) = written_and_copy(&device);
            let fence = device.new_fence();
            behind_held_update(name, &device, |a, b| {
                let updater = update(a, &written, &fence);
                staged(name, &updater);
                let waiter = b.command_buffer().expect("a command buffer");
                let mut compute =
                    (waiter.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
                compute.wait_for_fence(&fence).expect("a wait");
                compute.end_encoding();
                waiter.commit().expect("a commit");
                thread::sleep(Duration::from_millis(50));
                assert_ne!(
                    waiter.status(),
                    CommandBufferStatus::Completed,
                    "{name}: the wait, before the update ran"
                );
                (updater, waiter)
            });
        });
    }

    /// Metal's rule, measured on an M3 Max under macOS 27.0.1 (probe
    /// `cross_queue_fence_held_update.m` in the conformance crate, jobs m3-max-52186 and
    /// m3-max-52188): a wait binds to the updates already submitted, and an update in a command
    /// buffer Metal holds is not waited for. So a wait on queue B committed after an update on
    /// queue A held at a shared event's wait, or behind a command buffer of A enqueued and not
    /// committed, completes while the update is held, without its write, and the update runs
    /// once let go. Holding the wait behind it instead would end in a deadlock Metal does not
    /// have: with A waiting for an event that a command buffer B commits after the wait
    /// signals, every one of the three completes, the wait first.
    #[test]
    fn every_device_runs_a_wait_on_another_queue_before_an_update_held_at_an_event_or_an_enqueue() {
        each_device("fence_held", |name, _, device| {
            let a = device.new_command_queue(8).expect("a queue");
            let b = device.new_command_queue(8).expect("a queue");
            let enqueued = a.command_buffer().expect("a command buffer");
            for held in ["an event", "an enqueue", "an event B signals"] {
                let (written, copy) = written_and_copy(&device);
                let fence = device.new_fence();
                let event = device.new_shared_event().expect("a shared event");
                let updater = a.command_buffer().expect("a command buffer");
                match held {
                    "an enqueue" => enqueued.enqueue().expect("an enqueue"),
                    _ => {
                        (updater.encode_wait_for_event(event.as_event(), 1)).expect("an event wait")
                    }
                }
                let mut blit = updater.blit_command_encoder().expect("an encoder");
                blit.fill_buffer(&written, 0..64, 0x5a).expect("a fill");
                blit.update_fence(&fence).expect("an update");
                blit.end_encoding();
                updater.commit().expect("a commit");
                let waiter = wait(&b, &written, &copy, &fence, None);
                match held {
                    "an event B signals" => {
                        waiter.commit().expect("a commit");
                        let signal = b.command_buffer().expect("a command buffer");
                        (signal.encode_signal_event(event.as_event(), 1)).expect("a signal");
                        run(name, &signal);
                        finished(name, &waiter);
                        assert_eq!(waiter.error(), None, "{name}");
                    }
                    _ => {
                        run(name, &waiter);
                        assert_eq!(
                            updater.status(),
                            CommandBufferStatus::Committed,
                            "{name}: held at {held}, the update waits"
                        );
                    }
                }
                assert_eq!(word(&copy), 0, "{name}: held at {held}, the wait ran first");
                match held {
                    "an event" => event.set_signaled_value(1),
                    "an enqueue" => enqueued.commit().expect("a commit"),
                    _ => {}
                }
                finished(name, &updater);
                assert_eq!(
                    (updater.error(), word(&written)),
                    (None, 0x5a5a_5a5a),
                    "{name}: held at {held}, the update ran once let go"
                );
            }
            finished(name, &enqueued);
        });
    }
}
