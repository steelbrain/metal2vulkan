//! Blocks held by command buffers, and a take longer than the pool's own allocations, under a
//! device cfg on every device of the cfg's kind:
//!
//! - a buffer of 548 MiB in each class, taken as a block of its own length and made zeroed: the
//!   very next commit reads zero at its first, middle and last pages through its device address,
//!   and the writes after it read back;
//! - such a buffer and its block given up right after the commit of a command buffer that writes
//!   and reads it: the command buffer completes and reads its writes;
//! - a block given up while an encoded, uncommitted command buffer holds a buffer in it is not
//!   taken again until that command buffer retires, and goes back to the pool once it is dropped
//!   uncommitted;
//!
//! and every gauge reads zero after teardown.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished, Arena};
    use crate::tests::compute::on_device::{add, step, WORDS};
    use crate::{Buffer, CommandQueue, ComputePipelineState, Device, DispatchType, MemoryClass};
    use std::thread;
    use std::time::{Duration, Instant};

    /// The length of the first of three buffers a replayed frame makes, 548 MiB: a take longer
    /// than any allocation the pool makes on its own.
    const LEN: u64 = 574_619_648;

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
            for offset in offsets(big.length()) {
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
        /// the device and, for a Shared buffer, through its mapping.
        fn check(
            &self,
            name: &str,
            queue: &CommandQueue,
            arena: &mut Arena,
            big: &Buffer,
            before: &dyn Fn(u32) -> u32,
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
            if let Some(contents) = big.contents() {
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
                kernels.check(&name, &queue, &mut arena, &big, &|_| 0);
                drop((big, block));
            }
        });
    }

    /// A buffer of [`LEN`] bytes, written and read by a command buffer that a 512 MiB fill of
    /// another buffer keeps in flight, with the buffer and its block given up right after the
    /// commit, as a program releases a buffer its last command buffer still uses: the command
    /// buffer completes without error and its reads see its writes, so the block's memory lived
    /// until the work retired.
    #[test]
    fn every_device_keeps_a_buffer_of_over_half_a_gib_released_in_flight_until_its_work_retires() {
        each_device("large_buffer_released", |name, _, device| {
            let kernels = Kernels::new(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let mut arena = Arena::new(&device);
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
                drop((big, block));
                finished(&name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}: the command buffer");
                for out in &outs {
                    assert_eq!(words(out), vec![1; WORDS], "{name}: read after the release");
                }
                drop((command_buffer, other, other_block));
            }
        });
    }

    /// A Shared block of 4 MiB whose buffer a dispatch encoded in an uncommitted command buffer
    /// writes, the buffer and the block given up before the commit: a take of the same class and
    /// length then gets another block, whose bytes the dispatch leaves as they were once the
    /// command buffer is committed and completes, while the first block's range reads the
    /// dispatch's write through no handle. A command buffer dropped uncommitted gives such a
    /// block back to the pool, which holds it only once the command buffer is gone.
    #[test]
    fn every_device_takes_no_block_an_encoded_uncommitted_command_buffer_holds() {
        const SMALL: u64 = 4 << 20;
        each_device("block_held_by_encoding", |name, _, device| {
            let kernels = Kernels::new(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let take = |len: u64| {
                (device.take_block(MemoryClass::Shared, len))
                    .unwrap_or_else(|error| panic!("{name}: a block: {error}"))
            };
            let encode = |buffer: &Buffer| {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder
                    .set_compute_pipeline_state(&kernels.twice)
                    .expect("a state");
                encoder.set_buffer(Some(buffer), 0, 0).expect("the buffer");
                encoder
                    .dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1])
                    .expect("a write");
                encoder.end_encoding();
                command_buffer
            };
            let first = take(SMALL);
            let first_address = first.gpu_address;
            let buffer = (device.new_buffer(first.id, 0, SMALL, Some(0))).expect("a buffer");
            let command_buffer = encode(&buffer);
            drop((buffer, first));
            let second = take(SMALL);
            assert!(
                second.gpu_address + SMALL <= first_address
                    || first_address + SMALL <= second.gpu_address,
                "{name}: a take got the range an encoded command buffer holds"
            );
            let contents = second.contents.expect("a Shared block").cast::<u32>();
            for index in 0..WORDS {
                // SAFETY: the word lies in the block, which no GPU work uses.
                unsafe {
                    contents
                        .as_ptr()
                        .add(index)
                        .write_volatile(0xabcd_0000 + index as u32)
                };
            }
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}: the write");
            // SAFETY: the block is alive and holds `WORDS` words; the work completed.
            let after: Vec<u32> = (0..WORDS)
                .map(|index| unsafe { contents.as_ptr().add(index).read_volatile() })
                .collect();
            let expected: Vec<u32> = (0..WORDS as u32).map(|index| 0xabcd_0000 + index).collect();
            assert_eq!(after, expected, "{name}: the other block's bytes");
            drop((command_buffer, second));

            // Dropped uncommitted, the command buffer gives the block back.
            let len = SMALL + (64 << 10);
            let third = take(len);
            let third_id = third.id;
            let buffer = (device.new_buffer(third.id, 0, len, Some(0))).expect("a buffer");
            let command_buffer = encode(&buffer);
            drop((buffer, third));
            assert!(
                !device.core().pool.holds(third_id),
                "{name}: the pool holds a block an encoded command buffer holds"
            );
            drop(command_buffer);
            let deadline = Instant::now() + Duration::from_secs(120);
            while !device.core().pool.holds(third_id) {
                assert!(Instant::now() < deadline, "{name}: the block given back");
                thread::sleep(Duration::from_millis(1));
            }
        });
    }
}
