//! Buffers of over half a GiB, the lengths a replayed frame makes before a GPU fault, under a
//! device cfg on every device of the cfg's kind, in each class: taken as a block of its own
//! length and made zeroed, its first, middle and last pages read through its device address by
//! the very next commit and then written and read again; the same with other work in flight when
//! it is made, with a blit into it, with two of them alive at once, and as the frame's three
//! (two Shared, one Managed) alive at once. Every read sees the bytes its buffer holds, and every
//! gauge reads zero after teardown.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished, Arena};
    use crate::tests::compute::on_device::{add, step, WORDS};
    use crate::{Buffer, CommandQueue, ComputePipelineState, Device, DispatchType, MemoryClass};

    /// The lengths of the three buffers a replayed frame makes one after another before a GPU
    /// fault: 548 MiB, and 547 MiB and 1,312 bytes twice.
    const LEN: u64 = 574_619_648;
    const LEN_PINNED: u64 = 573_723_936;

    /// The offsets of the first, middle and last pages of [`WORDS`] words of a buffer of `len`.
    fn offsets(len: u64) -> [u64; 3] {
        [0, (len / 2) & !0xfff, (len - 4 * WORDS as u64) & !0xf]
    }

    fn words(buffer: &Buffer) -> Vec<u32> {
        let contents = buffer.contents().expect("a Shared buffer").cast::<u32>();
        // SAFETY: the buffer is alive and holds `WORDS` words, and its work completed.
        (0..WORDS)
            .map(|index| unsafe { contents.as_ptr().add(index).read_volatile() })
            .collect()
    }

    struct Kernels {
        sum: ComputePipelineState,
        twice: ComputePipelineState,
    }

    impl Kernels {
        fn new(device: &Device) -> Kernels {
            Kernels {
                sum: add(device),
                twice: step(device),
            }
        }

        /// The words at each offset of `big`, read by one command buffer's dispatches through
        /// its address, `o = a + 0` into a Shared buffer of the arena's.
        fn read(
            &self,
            name: &str,
            queue: &CommandQueue,
            arena: &mut Arena,
            big: &Buffer,
        ) -> Vec<Vec<u32>> {
            let zeros = vec![0u8; 4 * WORDS];
            let outs: Vec<Buffer> = offsets(big.length())
                .iter()
                .map(|_| arena.buffer())
                .collect();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder
                .set_compute_pipeline_state(&self.sum)
                .expect("a state");
            for (offset, out) in offsets(big.length()).into_iter().zip(&outs) {
                encoder
                    .set_buffer(Some(big), offset, 0)
                    .expect("the big buffer");
                encoder.set_bytes(&zeros, 1).expect("zeros");
                encoder.set_buffer(Some(out), 0, 2).expect("an output");
                encoder
                    .dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1])
                    .expect("a read");
            }
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}: the reads");
            outs.iter().map(words).collect()
        }

        /// `x = x * 2 + 1` over the words at each offset of `big`, in one command buffer.
        fn write(&self, name: &str, queue: &CommandQueue, big: &Buffer) {
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder
                .set_compute_pipeline_state(&self.twice)
                .expect("a state");
            for offset in offsets() {
                encoder
                    .set_buffer(Some(big), offset, 0)
                    .expect("the big buffer");
                encoder
                    .dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1])
                    .expect("a write");
            }
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}: the writes");
        }

        /// Reads `big`'s pages as `before`, steps them, and reads them as `before * 2 + 1`, on
        /// the device and, where `mapped`, through its mapping.
        fn check(
            &self,
            name: &str,
            queue: &CommandQueue,
            arena: &mut Arena,
            big: &Buffer,
            before: &dyn Fn(u32) -> u32,
            mapped: bool,
        ) {
            let pages = |word: &dyn Fn(u32) -> u32| -> Vec<Vec<u32>> {
                let page: Vec<u32> = (0..WORDS as u32).map(word).collect();
                vec![page; 3]
            };
            assert_eq!(
                self.read(name, queue, arena, big),
                pages(before),
                "{name}: read"
            );
            self.write(name, queue, big);
            let after = |index| before(index) * 2 + 1;
            assert_eq!(
                self.read(name, queue, arena, big),
                pages(&after),
                "{name}: written"
            );
            if let Some(contents) = big.contents().filter(|_| mapped) {
                for offset in offsets(big.length()) {
                    let at = contents.as_ptr();
                    // SAFETY: the page lies in the buffer, and its work completed.
                    let page: Vec<u32> = (0..WORDS)
                        .map(|index| unsafe {
                            at.add(offset as usize + 4 * index)
                                .cast::<u32>()
                                .read_unaligned()
                        })
                        .collect();
                    assert_eq!(page, pages(&after)[0], "{name}: the mapping at {offset}");
                }
            }
        }
    }

    fn classes(name: &str) -> [(String, MemoryClass); 2] {
        [
            (format!("{name} Shared"), MemoryClass::Shared),
            (format!("{name} Private"), MemoryClass::Private),
        ]
    }

    /// A buffer of [`LEN`] bytes in a block of its own, made zeroed: the very next commit reads
    /// zero at its first, middle and last pages, and the writes after it read back.
    #[test]
    fn every_device_reads_and_writes_the_far_pages_of_a_buffer_of_over_half_a_gib() {
        each_device("large_buffer", |name, _, device| {
            let kernels = Kernels::new(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let mut arena = Arena::new(&device);
            for (name, class) in classes(name) {
                let block = (device.take_block(class, LEN))
                    .unwrap_or_else(|error| panic!("{name}: the block: {error}"));
                let big = (device.new_buffer(block.id, 0, LEN, Some(0)))
                    .unwrap_or_else(|error| panic!("{name}: the buffer: {error}"));
                kernels.check(&name, &queue, &mut arena, &big, &|_| 0, true);
                drop((big, block));
            }
        });
    }

    /// The same, made while a fill of another 64 MiB buffer is in flight on another queue.
    #[test]
    fn every_device_reads_a_buffer_of_over_half_a_gib_made_while_other_work_is_in_flight() {
        each_device("large_buffer_busy", |name, _, device| {
            let kernels = Kernels::new(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let other_queue = device.new_command_queue(4).expect("a queue");
            let mut arena = Arena::new(&device);
            for (name, class) in classes(name) {
                let other_block =
                    (device.take_block(MemoryClass::Private, 64 << 20)).expect("the other block");
                let other = (device.new_buffer(other_block.id, 0, 64 << 20, None))
                    .expect("the other buffer");
                let busy = other_queue.command_buffer().expect("a command buffer");
                let mut blit = busy.blit_command_encoder().expect("a blit encoder");
                for round in 0..8u8 {
                    blit.fill_buffer(&other, 0..64 << 20, round)
                        .expect("a fill");
                }
                blit.end_encoding();
                busy.commit().expect("a commit");
                let block = (device.take_block(class, LEN))
                    .unwrap_or_else(|error| panic!("{name}: the block: {error}"));
                let big = (device.new_buffer(block.id, 0, LEN, Some(0)))
                    .unwrap_or_else(|error| panic!("{name}: the buffer: {error}"));
                kernels.check(&name, &queue, &mut arena, &big, &|_| 0, true);
                finished(&name, &busy);
                drop((big, block, busy, other, other_block));
            }
        });
    }

    /// A blit copies words into its first, middle and last pages, which then read back.
    #[test]
    fn every_device_reads_a_blit_into_a_buffer_of_over_half_a_gib() {
        each_device("large_buffer_blit", |name, _, device| {
            let kernels = Kernels::new(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let mut arena = Arena::new(&device);
            let source = arena.buffer();
            let contents = source.contents().expect("a Shared buffer").cast::<u32>();
            for index in 0..WORDS {
                // SAFETY: the buffer holds `WORDS` words, and no GPU work uses it.
                unsafe {
                    contents
                        .as_ptr()
                        .add(index)
                        .write_volatile(1000 + index as u32)
                };
            }
            for (name, class) in classes(name) {
                let block = (device.take_block(class, LEN))
                    .unwrap_or_else(|error| panic!("{name}: the block: {error}"));
                let big = (device.new_buffer(block.id, 0, LEN, Some(0)))
                    .unwrap_or_else(|error| panic!("{name}: the buffer: {error}"));
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut blit = command_buffer
                    .blit_command_encoder()
                    .expect("a blit encoder");
                for offset in offsets(LEN) {
                    blit.copy_from_buffer_to_buffer(&source, 0, &big, offset, 4 * WORDS as u64)
                        .expect("a copy");
                }
                blit.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(&name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}: the copies");
                kernels.check(&name, &queue, &mut arena, &big, &|index| 1000 + index, true);
                drop((big, block));
            }
        });
    }

    /// Two such buffers alive at once, each in a block of its own: each reads and writes its own
    /// pages.
    #[test]
    fn every_device_reads_two_buffers_of_over_half_a_gib_alive_at_once() {
        each_device("large_buffer_two", |name, _, device| {
            let kernels = Kernels::new(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let mut arena = Arena::new(&device);
            for (name, class) in classes(name) {
                let made: Vec<_> = (0..2)
                    .map(|_| {
                        let block = (device.take_block(class, LEN))
                            .unwrap_or_else(|error| panic!("{name}: a block: {error}"));
                        let big = (device.new_buffer(block.id, 0, LEN, Some(0)))
                            .unwrap_or_else(|error| panic!("{name}: a buffer: {error}"));
                        (big, block)
                    })
                    .collect();
                for (index, (big, _)) in made.iter().enumerate() {
                    let name = format!("{name} {index}");
                    kernels.check(&name, &queue, &mut arena, big, &|_| 0, true);
                }
                drop(made);
            }
        });
    }

    /// The replayed frame's three: a Shared buffer of [`LEN`] bytes and one of [`LEN_PINNED`],
    /// both zeroed, and a Managed one of [`LEN_PINNED`] whose consumer bytes at its pages hold
    /// words the next commit copies, all alive at once and each in a block of its own: each reads
    /// and writes its own pages, the Managed one through its device copy where it has one.
    #[test]
    fn every_device_reads_the_three_buffers_of_a_replayed_frame_alive_at_once() {
        each_device("large_buffer_frame", |name, _, device| {
            let kernels = Kernels::new(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let mut arena = Arena::new(&device);
            let shared = |len: u64| {
                let block = (device.take_block(MemoryClass::Shared, len))
                    .unwrap_or_else(|error| panic!("{name}: a block: {error}"));
                let big = (device.new_buffer(block.id, 0, len, Some(0)))
                    .unwrap_or_else(|error| panic!("{name}: a buffer: {error}"));
                (big, block)
            };
            let (first, first_block) = shared(LEN);
            let (second, second_block) = shared(LEN_PINNED);
            let managed_block = (device.take_block(MemoryClass::Shared, LEN_PINNED))
                .unwrap_or_else(|error| panic!("{name}: the Managed block: {error}"));
            let mapping = managed_block.contents.expect("the Managed block's mapping");
            for offset in offsets(LEN_PINNED) {
                for index in 0..WORDS {
                    // SAFETY: the word lies in the block, which no GPU work uses yet.
                    unsafe {
                        (mapping
                            .as_ptr()
                            .add(offset as usize + 4 * index)
                            .cast::<u32>())
                        .write_unaligned(5000 + index as u32)
                    };
                }
            }
            let managed = (device.new_managed_buffer(managed_block.id, 0, LEN_PINNED, LEN_PINNED))
                .unwrap_or_else(|error| panic!("{name}: the Managed buffer: {error}"));
            kernels.check(
                &format!("{name} Shared"),
                &queue,
                &mut arena,
                &first,
                &|_| 0,
                true,
            );
            kernels.check(
                &format!("{name} pinned"),
                &queue,
                &mut arena,
                &second,
                &|_| 0,
                true,
            );
            let unified = device.has_unified_memory();
            let words = |index| 5000 + index;
            kernels.check(
                &format!("{name} Managed"),
                &queue,
                &mut arena,
                &managed,
                &words,
                unified,
            );
            drop((
                first,
                first_block,
                second,
                second_block,
                managed,
                managed_block,
            ));
        });
    }

    /// A buffer of [`LEN`] bytes, written and read by a command buffer that a 512 MiB fill of
    /// another buffer keeps in flight, with the buffer and its block given up right after the
    /// commit, as a program releases a buffer its last command buffer still uses: the reads see
    /// the writes, and the block's memory is freed only once the command buffer retired.
    #[test]
    fn every_device_keeps_a_buffer_of_over_half_a_gib_released_in_flight_until_its_work_retires() {
        each_device("large_buffer_released", |name, _, device| {
            let kernels = Kernels::new(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let mut arena = Arena::new(&device);
            let counters = device.counters();
            for (name, class) in classes(name) {
                let other_block =
                    (device.take_block(MemoryClass::Private, 64 << 20)).expect("the other block");
                let other = (device.new_buffer(other_block.id, 0, 64 << 20, None))
                    .expect("the other buffer");
                let block = (device.take_block(class, LEN))
                    .unwrap_or_else(|error| panic!("{name}: the block: {error}"));
                let big = (device.new_buffer(block.id, 0, LEN, Some(0)))
                    .unwrap_or_else(|error| panic!("{name}: the buffer: {error}"));
                let outs: Vec<Buffer> = offsets(LEN).iter().map(|_| arena.buffer()).collect();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut blit = command_buffer
                    .blit_command_encoder()
                    .expect("a blit encoder");
                for round in 0..8u8 {
                    blit.fill_buffer(&other, 0..64 << 20, round)
                        .expect("a fill");
                }
                blit.end_encoding();
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder
                    .set_compute_pipeline_state(&kernels.twice)
                    .expect("a state");
                for offset in offsets(LEN) {
                    encoder
                        .set_buffer(Some(&big), offset, 0)
                        .expect("the big buffer");
                    encoder
                        .dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1])
                        .expect("a write");
                }
                encoder
                    .set_compute_pipeline_state(&kernels.sum)
                    .expect("a state");
                let zeros = vec![0u8; 4 * WORDS];
                for (offset, out) in offsets(LEN).into_iter().zip(&outs) {
                    encoder
                        .set_buffer(Some(&big), offset, 0)
                        .expect("the big buffer");
                    encoder.set_bytes(&zeros, 1).expect("zeros");
                    encoder.set_buffer(Some(out), 0, 2).expect("an output");
                    encoder
                        .dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1])
                        .expect("a read");
                }
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                let freed = counters.metrics().allocations_freed;
                drop((big, block));
                finished(&name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}: the command buffer");
                assert_eq!(
                    counters.metrics().allocations_freed,
                    freed,
                    "{name}: an allocation freed before the command buffer retired"
                );
                for out in &outs {
                    assert_eq!(words(out), vec![1; WORDS], "{name}: read after the release");
                }
                drop((command_buffer, other, other_block));
            }
        });
    }
}
