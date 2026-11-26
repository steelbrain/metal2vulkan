//! What a kernel's reads and writes outside its buffer reach when the buffer lies between two
//! neighbours in one Shared block, under a device cfg on every device of the cfg's kind. The
//! owned fixture `examples/pool_reach.ll` (its source `pool_reach.metal`) holds `pool_reach`:
//! each thread reads the word `at[i]` words from the bound place of slot 0 into `seen[i]` and
//! then writes its mark, `0xa5a5_0000 | i`, there.
//!
//! Five buffers of 256 bytes are consecutive in one block ([`Arena`]): the neighbour before, the
//! target, the neighbour after, then `seen` and `at`. Every place the indices reach outside the
//! target lies past the slack the runtime README's contract states for a bound buffer (a start
//! off 16 bytes and a length off 4; the target has neither), so on every device each such read
//! is zero and each such write lands nowhere:
//!
//! - bound at offset 0: words 0 and 63 are the target's own (read, then marked); words -1, -4 and
//!   -64 lie in the neighbour before, 64, 68 and 127 in the neighbour after, and -2^20 and 2^20
//!   far outside the block;
//! - bound at offset 16: words 0 and 59 are the target's 4th and last (read, then marked); word -5
//!   is the neighbour before's last and word 60 the neighbour after's first.
//!
//! Each neighbour keeps its fill and each word of the target the indices do not reach keeps its
//! own. A device whose translator binds a kernel's buffers
//! (`DeviceProfile::binds_kernel_buffers`) runs both dispatches in that form, its pipeline's
//! robust buffer access 2 containing them (`Metrics::bound_dispatches`); any other runs them at
//! the target's checked device address (`Metrics::address_dispatches`). The test prints which.
//!
//! `llvm-as` assembles the fixture; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, Arena};
    use crate::tests::draw::on_device::{library, run};
    use crate::tests::no_constexpr_samplers;
    use crate::{Buffer, ComputePipelineDescriptor, DispatchType, FunctionDescriptor, StageSource};
    use metal2vulkan_devtools::fixtures_dir;

    /// Words in each 256-byte buffer.
    const WORDS: usize = 64;
    /// The neighbours' fills, and the target's words (`TARGET | index`).
    const BEFORE: u32 = 0x1111_1111;
    const AFTER: u32 = 0x3333_3333;
    const TARGET: u32 = 0x2222_0000;
    /// A thread's mark.
    const MARK: u32 = 0xa5a5_0000;

    fn write(buffer: &Buffer, word: impl Fn(usize) -> u32) {
        let contents = buffer.contents().expect("a Shared buffer").cast::<u32>();
        for index in 0..WORDS {
            // SAFETY: the buffer is alive, 256 bytes long, and no GPU work uses it.
            unsafe { contents.as_ptr().add(index).write_volatile(word(index)) };
        }
    }

    fn read(buffer: &Buffer) -> Vec<u32> {
        let contents = buffer.contents().expect("a Shared buffer").cast::<u32>();
        // SAFETY: the buffer is alive and 256 bytes long, and its work completed.
        (0..WORDS)
            .map(|index| unsafe { contents.as_ptr().add(index).read_volatile() })
            .collect()
    }

    #[test]
    fn every_device_keeps_a_kernel_s_reaches_outside_its_buffer_from_its_neighbours() {
        let Some(library) = library(&fixtures_dir().join("examples/pool_reach.ll")) else {
            return;
        };
        let function = library.function("pool_reach").expect("pool_reach");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        // Each dispatch: the target's bind offset, then each thread's word index and the target
        // word it reaches (None outside the target).
        let dispatches: [(u64, &[(i32, Option<usize>)]); 2] = [
            (
                0,
                &[
                    (0, Some(0)),
                    (63, Some(63)),
                    (-1, None),
                    (-4, None),
                    (-64, None),
                    (64, None),
                    (68, None),
                    (127, None),
                    (-(1 << 20), None),
                    (1 << 20, None),
                ],
            ),
            (16, &[(0, Some(4)), (59, Some(63)), (-5, None), (60, None)]),
        ];
        each_device("pool_reach", |name, _, device| {
            let state = (device.new_compute_pipeline_state(&descriptor, no_constexpr_samplers))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let bound = device.translator().profile().binds_kernel_buffers();
            let queue = device.new_command_queue(4).expect("a queue");
            let mut arena = Arena::new(&device);
            let (before, target, after, seen, at) = (
                arena.buffer(),
                arena.buffer(),
                arena.buffer(),
                arena.buffer(),
                arena.buffer(),
            );
            for &(offset, reaches) in &dispatches {
                let what = format!("{name}: bound at {offset}");
                write(&before, |_| BEFORE);
                write(&after, |_| AFTER);
                write(&target, |index| TARGET | index as u32);
                write(&seen, |_| u32::MAX);
                write(&at, |index| {
                    reaches.get(index).map_or(0, |&(word, _)| word as u32)
                });
                let metrics = device.counters().metrics();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                for (slot, (buffer, at_offset)) in [(&target, offset), (&seen, 0), (&at, 0)]
                    .into_iter()
                    .enumerate()
                {
                    encoder
                        .set_buffer(Some(buffer), at_offset, slot as u32)
                        .expect("a buffer");
                }
                let threads = reaches.len() as u32;
                encoder
                    .dispatch_threads([threads, 1, 1], [threads, 1, 1])
                    .unwrap_or_else(|error| panic!("{what}: {error}"));
                encoder.end_encoding();
                run(&what, &command_buffer);
                let after_run = device.counters().metrics();
                let grown = (
                    after_run.bound_dispatches - metrics.bound_dispatches,
                    after_run.address_dispatches - metrics.address_dispatches,
                );
                assert_eq!(
                    grown,
                    if bound { (1, 0) } else { (0, 1) },
                    "{what}: dispatches by form"
                );
                let mut want_target: Vec<u32> = (0..WORDS).map(|i| TARGET | i as u32).collect();
                let mut want_seen = vec![u32::MAX; WORDS];
                for (thread, &(_, reached)) in reaches.iter().enumerate() {
                    want_seen[thread] = reached.map_or(0, |index| TARGET | index as u32);
                    if let Some(index) = reached {
                        want_target[index] = MARK | thread as u32;
                    }
                }
                assert_eq!(read(&seen), want_seen, "{what}: the words read");
                assert_eq!(read(&target), want_target, "{what}: the target");
                assert_eq!(
                    read(&before),
                    [BEFORE; WORDS],
                    "{what}: the neighbour before"
                );
                assert_eq!(read(&after), [AFTER; WORDS], "{what}: the neighbour after");
            }
            let form = match bound {
                true => "bound, robust buffer access 2",
                false => "checked device address",
            };
            println!(
                "{name}: reads outside a pooled buffer read zero and writes there land nowhere \
                 ({form}), at bind offsets 0 and 16"
            );
        });
    }
}
