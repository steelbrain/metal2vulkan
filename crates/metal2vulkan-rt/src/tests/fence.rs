//! Fences, under a device cfg on every device of the cfg's kind: an encoder waiting on a fence
//! after another command buffer's encoder updated it sees that command buffer's writes, and the
//! fence calls of a compute and a blit encoder record no device work (the same submissions,
//! barriers, bytes copied and passes as the same work without them). A fence keeps the first queue
//! that uses it: a use on another queue is `Gap::FenceOtherQueue`.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::blit::on_device::settled;
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        BufferLayout, CommandBuffer, CommandQueue, Device, DispatchType, Error, Fence, Gap,
        MemoryClass, Metrics, PixelFormat, TextureDescriptor, TexturePlace, TextureType,
        TextureUsage,
    };

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
            let block = (device.take_block(MemoryClass::Shared, 4096))
                .expect("a Shared block")
                .id;
            let filled = (device.new_buffer(block, 0, 64, false)).expect("a buffer");
            let out = (device.new_buffer(block, 256, 64, false)).expect("a buffer");
            let mut d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 4, 4);
            d.usage = TextureUsage::RENDER_TARGET;
            let texture = device.new_texture(&d, None).expect("a texture");
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

    /// A fence `queue` used first refuses another queue's encoders; one another queue used first
    /// is that queue's.
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
        assert!(
            matches!(
                compute.wait_for_fence(&fence),
                Err(Error::Unsupported(Gap::FenceOtherQueue))
            ),
            "{name}: a compute encoder of a second queue"
        );
        compute.end_encoding();
        let mut blit = command_buffer.blit_command_encoder().expect("an encoder");
        assert!(
            matches!(
                blit.update_fence(&fence),
                Err(Error::Unsupported(Gap::FenceOtherQueue))
            ),
            "{name}: a blit encoder of a second queue"
        );
        let own = device.new_fence();
        blit.update_fence(&own)
            .expect("a fence the second queue used first");
        blit.end_encoding();
        run(name, &command_buffer);
    }
}
