//! The device's pools of sets of set 0 (`crate::bound`), under a device cfg on every device of the
//! cfg's kind, through the owned kernel `scale_and_offset` (`examples/runtime_examples.ll`):
//! `output[i] = input[i] * scale + i`, its three buffers bound. Each dispatch binds another output
//! than the one before and one of three scales in turn, so its descriptors begin no set written
//! before and it writes a set of its own (`Metrics::buffer_sets_written`).
//!
//! On a device whose translator binds a kernel's buffers (`DeviceProfile::binds_kernel_buffers`),
//! command buffers share the device's pools:
//!
//! - 200 command buffers of one dispatch each, held at once, take one set each from one pool;
//! - past capacity: one command buffer of `(FREE_POOLS + 2) * POOL_SETS` dispatches holds that many
//!   pools; once it retires, the device keeps `FREE_POOLS` free besides the one it hands out from
//!   and destroys the one past them; a command buffer of `(FREE_POOLS + 1) * POOL_SETS` then takes
//!   those and makes none; and none is left once the device drops;
//! - two threads encoding a command buffer each at once take runs from the same pools, under the
//!   validation layer, whose thread-safety checks report two threads at one pool or one set.
//!
//! Each command buffer's outputs are checked. A device that keeps its kernels' buffers at their
//! addresses writes no set and holds no pool. `llvm-as` assembles the fixture; without it the
//! tests skip visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::bound::{FREE_POOLS, POOL_SETS, RUN_SETS};
    use crate::tests::command::on_device::{each_device, within, Arena};
    use crate::tests::draw::on_device::{examples, run};
    use crate::tests::real_size::on_device::kernel;
    use crate::{Buffer, CommandBuffer, CommandQueue, ComputePipelineState, Device, DispatchType};
    use std::sync::Barrier;
    use std::thread;

    /// Floats in each 256-byte buffer: the threads of a dispatch.
    const FLOATS: usize = 64;

    fn write(buffer: &Buffer, value: impl Fn(usize) -> f32) {
        let contents = buffer.contents().expect("a Shared buffer").cast::<f32>();
        for index in 0..FLOATS {
            // SAFETY: the buffer is alive, 256 bytes long, and no GPU work uses it.
            unsafe { contents.as_ptr().add(index).write_volatile(value(index)) };
        }
    }

    fn read(buffer: &Buffer) -> Vec<f32> {
        let contents = buffer.contents().expect("a Shared buffer").cast::<f32>();
        // SAFETY: the buffer is alive and 256 bytes long, and its work completed.
        (0..FLOATS)
            .map(|index| unsafe { contents.as_ptr().add(index).read_volatile() })
            .collect()
    }

    /// One encoder's buffers: the input, two outputs and three scales, each scale's word its own.
    struct Work {
        input: Buffer,
        outputs: [Buffer; 2],
        scales: [(Buffer, f32); 3],
    }

    impl Work {
        fn new(arena: &mut Arena, scales: [f32; 3]) -> Work {
            let input = arena.buffer();
            write(&input, |index| index as f32 * 0.5);
            let outputs = [arena.buffer(), arena.buffer()];
            let scales = scales.map(|scale| {
                let buffer = arena.buffer();
                write(&buffer, |_| scale);
                (buffer, scale)
            });
            Work {
                input,
                outputs,
                scales,
            }
        }

        /// A command buffer of `queue` with `dispatches` dispatches, the `k`th into output `k % 2`
        /// at scale `k % 3`, each output first cleared.
        fn encode(
            &self,
            name: &str,
            queue: &CommandQueue,
            state: &ComputePipelineState,
            dispatches: usize,
        ) -> CommandBuffer {
            for output in &self.outputs {
                write(output, |_| -1.0);
            }
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(state).expect("a state");
            encoder
                .set_buffer(Some(&self.input), 0, 0)
                .expect("the input");
            for k in 0..dispatches {
                encoder
                    .set_buffer(Some(&self.outputs[k % 2]), 0, 1)
                    .expect("an output");
                encoder
                    .set_buffer(Some(&self.scales[k % 3].0), 0, 2)
                    .expect("a scale");
                encoder
                    .dispatch_threads([FLOATS as u32, 1, 1], [FLOATS as u32, 1, 1])
                    .unwrap_or_else(|error| panic!("{name}: dispatch {k}: {error}"));
            }
            encoder.end_encoding();
            command_buffer
        }

        /// Each output as the last of `dispatches` that wrote it left it.
        fn check(&self, what: &str, dispatches: usize) {
            for (output, buffer) in self.outputs.iter().enumerate() {
                let last = (0..dispatches).rev().find(|k| k % 2 == output);
                let want: Vec<f32> = (0..FLOATS)
                    .map(|index| match last {
                        Some(k) => index as f32 * 0.5 * self.scales[k % 3].1 + index as f32,
                        None => -1.0,
                    })
                    .collect();
                assert_eq!(read(buffer), want, "{what}: output {output}");
            }
        }
    }

    fn state(name: &str, device: &Device) -> Option<ComputePipelineState> {
        let library = examples()?;
        Some(kernel(name, device, &library, "scale_and_offset"))
    }

    #[test]
    fn every_device_shares_its_set_pools_and_keeps_at_most_seventeen() {
        each_device("set_pools", |name, _, device| {
            let Some(state) = state(name, &device) else {
                return;
            };
            let bound = device.translator().profile().binds_kernel_buffers();
            let counters = device.counters();
            let pools = || counters.gauges().buffer_set_pools;
            let written = || counters.metrics().buffer_sets_written;
            let queue = device.new_command_queue(256).expect("a queue");
            let mut arena = Arena::new(&device);
            let work = Work::new(&mut arena, [2.0, 3.0, 5.0]);
            let full = POOL_SETS as usize;
            // Held at once, one set each.
            let start = written();
            let held: Vec<_> = (0..200)
                .map(|_| work.encode(name, &queue, &state, 1))
                .collect();
            let (sets, made) = (written() - start, pools());
            assert_eq!(
                (sets, made),
                if bound { (200, 1) } else { (0, 0) },
                "{name}: 200 command buffers of one dispatch: sets written and pools"
            );
            drop(held);
            assert_eq!(pools(), made, "{name}: the pool kept as they drop");
            // Past capacity: one command buffer holds FREE_POOLS + 2 pools.
            let past = (FREE_POOLS + 2) * full;
            let start = written();
            let command_buffer = work.encode(name, &queue, &state, past);
            let held = pools();
            assert_eq!(
                (written() - start, held),
                if bound {
                    (past as u64, FREE_POOLS as u64 + 2)
                } else {
                    (0, 0)
                },
                "{name}: {past} dispatches in one command buffer: sets written and pools"
            );
            run(name, &command_buffer);
            drop(command_buffer);
            let kept = if bound { FREE_POOLS as u64 + 1 } else { 0 };
            within("the pools' return", || pools() == kept);
            work.check(&format!("{name}: {past} dispatches"), past);
            // The kept pools serve a command buffer as many sets long, which makes none.
            let again = (FREE_POOLS + 1) * full;
            let command_buffer = work.encode(name, &queue, &state, again);
            assert_eq!(
                pools(),
                kept,
                "{name}: {again} dispatches from the kept pools"
            );
            run(name, &command_buffer);
            drop(command_buffer);
            within("the pools' second return", || pools() == kept);
            work.check(&format!("{name}: {again} dispatches"), again);
            drop((state, work, arena, queue, device));
            assert_eq!(pools(), 0, "{name}: the pools at the device's drop");
            println!(
                "{name}: 200 command buffers of a set each from {made} pool(s); {past} sets in one \
                 held {held} pools, {kept} kept as it retired, and {again} more made none; 0 at drop"
            );
        });
    }

    #[test]
    fn every_device_takes_sets_from_two_threads_at_once() {
        each_device("set_pools_threads", |name, _, device| {
            let Some(state) = state(name, &device) else {
                return;
            };
            let bound = device.translator().profile().binds_kernel_buffers();
            let counters = device.counters();
            let mut arena = Arena::new(&device);
            let works = [
                Work::new(&mut arena, [2.0, 3.0, 5.0]),
                Work::new(&mut arena, [7.0, 11.0, 13.0]),
            ];
            let queues = [
                device.new_command_queue(4).expect("a queue"),
                device.new_command_queue(4).expect("a queue"),
            ];
            const DISPATCHES: usize = 600;
            let start = counters.metrics().buffer_sets_written;
            let together = Barrier::new(2);
            let command_buffers: Vec<CommandBuffer> = thread::scope(|scope| {
                let encoding: Vec<_> = (works.iter().zip(&queues))
                    .map(|(work, queue)| {
                        let (state, together) = (&state, &together);
                        scope.spawn(move || {
                            together.wait();
                            work.encode(name, queue, state, DISPATCHES)
                        })
                    })
                    .collect();
                (encoding.into_iter())
                    .map(|thread| thread.join().expect("an encoding thread"))
                    .collect()
            });
            let sets = counters.metrics().buffer_sets_written - start;
            let pools = counters.gauges().buffer_set_pools;
            // Each holds at most RUN_SETS - 1 sets unused.
            let most = (2 * (DISPATCHES + RUN_SETS as usize - 1)).div_ceil(POOL_SETS as usize);
            match bound {
                true => {
                    assert_eq!(sets, 2 * DISPATCHES as u64, "{name}: the sets written");
                    assert!(
                        (1..=most as u64).contains(&pools),
                        "{name}: {pools} pools for {sets} sets, past {most}"
                    );
                }
                false => assert_eq!((sets, pools), (0, 0), "{name}: sets and pools"),
            }
            for (at, command_buffer) in command_buffers.iter().enumerate() {
                run(&format!("{name}: thread {at}"), command_buffer);
            }
            for (at, work) in works.iter().enumerate() {
                work.check(&format!("{name}: thread {at}"), DISPATCHES);
            }
            println!(
                "{name}: two threads encoding {DISPATCHES} dispatches each at once wrote {sets} \
                 sets from {pools} pool(s)"
            );
        });
    }
}
