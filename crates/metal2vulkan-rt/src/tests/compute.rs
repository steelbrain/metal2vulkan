//! Compute encoders, under a device cfg on every device of the cfg's kind: dispatches over bound
//! buffers and `set_bytes`, parameters past the push window
//! through the ring, indirect dispatch, barriers only at hazards in a serial encoder and only on
//! request in a concurrent one, and Metal's refusals.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::hazard::Access;
    use crate::pipeline::{Constant, Field, FieldValue, Interface, Place};
    use crate::tests::command::on_device::{each_device, finished, Arena};
    use crate::tests::kernels;
    use crate::{
        Buffer, CommandBuffer, ComputePipelineState, Device, DispatchType, Error, Gap, Limit,
        Refusal,
    };

    /// Words in each test buffer: 256 bytes.
    const WORDS: usize = 64;

    fn region() -> Field {
        Field {
            place: Place::Push(0),
            value: FieldValue::DispatchRegion,
        }
    }

    fn address(slot: u32, place: Place) -> Field {
        Field {
            place,
            value: FieldValue::BufferAddress(slot),
        }
    }

    fn interface(buffers: Vec<(u32, Access)>, fields: Vec<Field>, push_len: u32) -> Interface {
        Interface {
            buffers,
            reads_unbound: false,
            writes_unbound: false,
            push_len,
            block_len: 0,
            block_address: 0,
            fields,
            constants: vec![
                (0, Constant::Size(0)),
                (1, Constant::Size(1)),
                (2, Constant::Size(2)),
            ],
            threadgroups: Vec::new(),
            max_total_threads: 1024,
        }
    }

    /// `o = a + b` over slots 0, 1 and 2.
    fn add(device: &Device) -> ComputePipelineState {
        let fields = vec![
            region(),
            address(0, Place::Push(32)),
            address(1, Place::Push(40)),
            address(2, Place::Push(48)),
        ];
        let buffers = vec![(0, Access::Read), (1, Access::Read), (2, Access::Write)];
        ComputePipelineState::for_test(device, &kernels::ADD, interface(buffers, fields, 56))
            .expect("the add kernel")
    }

    /// `x = x * 2 + 1` over slot 0.
    fn step(device: &Device) -> ComputePipelineState {
        let fields = vec![region(), address(0, Place::Push(32))];
        let buffers = vec![(0, Access::ReadWrite)];
        ComputePipelineState::for_test(device, &kernels::STEP, interface(buffers, fields, 40))
            .expect("the step kernel")
    }

    fn write_words(buffer: &Buffer, word: impl Fn(u32) -> u32) {
        let contents = buffer.contents().expect("a Shared buffer").cast::<u32>();
        for index in 0..WORDS {
            // SAFETY: the buffer is alive, 256 bytes long, and no GPU work uses it.
            unsafe {
                contents
                    .as_ptr()
                    .add(index)
                    .write_volatile(word(index as u32))
            };
        }
    }

    fn words(buffer: &Buffer) -> Vec<u32> {
        let contents = buffer.contents().expect("a Shared buffer").cast::<u32>();
        // SAFETY: the buffer is alive and 256 bytes long, and its work completed.
        (0..WORDS)
            .map(|index| unsafe { contents.as_ptr().add(index).read_volatile() })
            .collect()
    }

    fn run(name: &str, command_buffer: &CommandBuffer) {
        command_buffer.commit().expect("a commit");
        finished(name, command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// A dispatch reads bound buffers and `set_bytes` data by address and writes a bound buffer.
    #[test]
    fn every_device_dispatches_over_bound_buffers_and_set_bytes() {
        each_device("compute_bound", |name, _, device| {
            let mut arena = Arena::new(&device);
            let (a, o) = (arena.buffer(), arena.buffer());
            write_words(&a, |index| index);
            let state = add(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            encoder.set_buffer(Some(&a), 0, 0).expect("a buffer");
            let bytes: Vec<u8> = (0..WORDS as u32)
                .flat_map(|index| (100 + index).to_le_bytes())
                .collect();
            encoder.set_bytes(&bytes, 1).expect("bytes");
            encoder.set_buffer(Some(&o), 0, 2).expect("a buffer");
            encoder
                .dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1])
                .expect("a dispatch");
            encoder.end_encoding();
            run(name, &command_buffer);
            let expected: Vec<u32> = (0..WORDS as u32).map(|index| 2 * index + 100).collect();
            assert_eq!(words(&o), expected, "{name}");
        });
    }

    /// A serial encoder places a barrier at each hazard and none between independent dispatches;
    /// a concurrent encoder places one only on request. The values show the order held.
    #[test]
    fn every_device_places_barriers_at_hazards_only() {
        each_device("compute_barriers", |name, _, device| {
            let mut arena = Arena::new(&device);
            let state = step(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let count = 8;

            // Dependent: every dispatch reads and writes one buffer.
            let x = arena.buffer();
            write_words(&x, |index| index);
            let before = device.counters().metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            encoder.set_buffer(Some(&x), 0, 0).expect("a buffer");
            for _ in 0..count {
                encoder
                    .dispatch_threadgroups([4, 1, 1], [16, 1, 1])
                    .expect("a dispatch");
            }
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = device.counters().metrics();
            // A barrier before each dispatch but the first, and the segment's end.
            assert_eq!(after.barriers - before.barriers, count, "{name}: dependent");
            let chained = |start: u32| (0..count).fold(start, |x, _| x * 2 + 1);
            let expected: Vec<u32> = (0..WORDS as u32).map(chained).collect();
            assert_eq!(words(&x), expected, "{name}: dependent");

            // Independent: each dispatch has a buffer of its own.
            let buffers: Vec<Buffer> = (0..count).map(|_| arena.buffer()).collect();
            for buffer in &buffers {
                write_words(buffer, |index| index);
            }
            let before = device.counters().metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            for buffer in &buffers {
                encoder.set_buffer(Some(buffer), 0, 0).expect("a buffer");
                encoder
                    .dispatch_threadgroups([4, 1, 1], [16, 1, 1])
                    .expect("a dispatch");
            }
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = device.counters().metrics();
            assert_eq!(after.barriers - before.barriers, 1, "{name}: independent");
            let expected: Vec<u32> = (0..WORDS as u32).map(|x| x * 2 + 1).collect();
            for buffer in &buffers {
                assert_eq!(words(buffer), expected, "{name}: independent");
            }

            // Concurrent: barriers where asked for, and none at the hazards.
            write_words(&x, |index| index);
            let before = device.counters().metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Concurrent)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            encoder.set_buffer(Some(&x), 0, 0).expect("a buffer");
            for index in 0..count {
                if index > 0 {
                    encoder.memory_barrier();
                }
                encoder
                    .dispatch_threadgroups([4, 1, 1], [16, 1, 1])
                    .expect("a dispatch");
            }
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = device.counters().metrics();
            assert_eq!(
                after.barriers - before.barriers,
                count,
                "{name}: concurrent"
            );
            let expected: Vec<u32> = (0..WORDS as u32).map(chained).collect();
            assert_eq!(words(&x), expected, "{name}: concurrent");
        });
    }

    /// Eleven buffer addresses fill the push window after the region and the block's address;
    /// five more go in a block in the parameter ring.
    #[test]
    fn every_device_reads_parameters_past_the_push_window_from_the_ring() {
        each_device("compute_ring", |name, _, device| {
            let mut fields = vec![region()];
            for slot in 0..11 {
                fields.push(address(slot, Place::Push(40 + 8 * slot)));
            }
            for slot in 11..16 {
                fields.push(address(slot, Place::Block(8 * (slot - 11))));
            }
            let mut buffers: Vec<(u32, Access)> =
                (0..15).map(|slot| (slot, Access::Read)).collect();
            buffers.push((15, Access::Write));
            let mut wide = interface(buffers, fields, 128);
            wide.block_len = 48;
            wide.block_address = 32;
            let state = ComputePipelineState::for_test(&device, &kernels::WIDE, wide)
                .expect("the wide kernel");
            let mut arena = Arena::new(&device);
            let inputs: Vec<Buffer> = (0..15).map(|_| arena.buffer()).collect();
            for (k, buffer) in inputs.iter().enumerate() {
                write_words(buffer, |index| index * (k as u32 + 1));
            }
            let out = arena.buffer();
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            for (slot, buffer) in inputs.iter().chain([&out]).enumerate() {
                encoder
                    .set_buffer(Some(buffer), 0, slot as u32)
                    .expect("a buffer");
            }
            encoder
                .dispatch_threads([WORDS as u32, 1, 1], [32, 1, 1])
                .expect("a dispatch");
            encoder.end_encoding();
            run(name, &command_buffer);
            // The sum of index * k for k in 1..=15.
            let expected: Vec<u32> = (0..WORDS as u32).map(|index| index * 120).collect();
            assert_eq!(words(&out), expected, "{name}");
        });
    }

    /// An indirect dispatch reads its threadgroup counts from a buffer, and its kernel takes the
    /// grid from them.
    #[test]
    fn every_device_dispatches_indirectly() {
        each_device("compute_indirect", |name, _, device| {
            let mut arena = Arena::new(&device);
            let (a, b, o, arguments) = (
                arena.buffer(),
                arena.buffer(),
                arena.buffer(),
                arena.buffer(),
            );
            write_words(&a, |index| index);
            write_words(&b, |index| 1000 * index);
            write_words(&arguments, |index| match index {
                4 => 4,
                5 | 6 => 1,
                _ => 0,
            });
            let state = add(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            for (slot, buffer) in [&a, &b, &o].into_iter().enumerate() {
                encoder
                    .set_buffer(Some(buffer), 0, slot as u32)
                    .expect("a buffer");
            }
            encoder
                .dispatch_threadgroups_with_indirect_buffer(&arguments, 16, [16, 1, 1])
                .expect("an indirect dispatch");
            encoder.end_encoding();
            run(name, &command_buffer);
            let expected: Vec<u32> = (0..WORDS as u32).map(|index| 1001 * index).collect();
            assert_eq!(words(&o), expected, "{name}");
        });
    }

    /// What Metal refuses in an encoder, and the limits the device sets.
    #[test]
    fn every_device_refuses_what_metal_refuses_in_an_encoder() {
        each_device("compute_refusals", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let state = add(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            let refused = |result: Result<(), Error>| match result {
                Err(Error::Refused(refusal)) => refusal,
                other => panic!("{name}: {other:?}"),
            };
            assert_eq!(
                refused(encoder.dispatch_threadgroups([1, 1, 1], [1, 1, 1])),
                Refusal::NoPipeline
            );
            assert_eq!(
                refused(encoder.set_buffer(Some(&buffer), 0, 31)),
                Refusal::BufferIndex { index: 31 }
            );
            assert_eq!(
                refused(encoder.set_buffer_offset(4, 3)),
                Refusal::NoBufferAtIndex { index: 3 }
            );
            encoder.set_compute_pipeline_state(&state).expect("a state");
            assert_eq!(
                refused(encoder.dispatch_threadgroups([1, 1, 1], [0, 1, 1])),
                Refusal::ThreadgroupSize {
                    size: [0, 1, 1],
                    max: 1024
                }
            );
            assert_eq!(
                refused(encoder.dispatch_threadgroups([1, 1, 1], [64, 32, 1])),
                Refusal::ThreadgroupSize {
                    size: [64, 32, 1],
                    max: 1024
                }
            );
            // An axis whose count the device bounds below `u32::MAX`: a group count past it.
            let raw = device.raw();
            // SAFETY: the physical device is the instance's.
            let limits = unsafe {
                raw.instance
                    .get_physical_device_properties(raw.physical_device)
                    .limits
            };
            if let Some(axis) =
                (0..3).find(|&axis| limits.max_compute_work_group_count[axis] < u32::MAX)
            {
                let mut count = [1; 3];
                count[axis] = limits.max_compute_work_group_count[axis] + 1;
                assert!(matches!(
                    encoder.dispatch_threadgroups(count, [1, 1, 1]),
                    Err(Error::Limit(Limit::GroupCount { .. }))
                ));
            }
            // Partial threadgroups are not built.
            assert!(matches!(
                encoder.dispatch_threads([60, 1, 1], [16, 1, 1]),
                Err(Error::Unsupported(Gap::PartialThreadgroups { .. }))
            ));
            assert_eq!(
                refused(encoder.dispatch_threadgroups_with_indirect_buffer(&buffer, 2, [1, 1, 1])),
                Refusal::IndirectOffset {
                    offset: 2,
                    length: 256
                }
            );
            assert_eq!(
                refused(encoder.dispatch_threadgroups_with_indirect_buffer(
                    &buffer,
                    248,
                    [1, 1, 1]
                )),
                Refusal::IndirectOffset {
                    offset: 248,
                    length: 256
                }
            );
            // While the encoder is open, the command buffer takes no other call.
            assert!(matches!(
                command_buffer.compute_command_encoder(DispatchType::Serial),
                Err(Error::Refused(Refusal::EncoderOpen))
            ));
            assert_eq!(refused(command_buffer.commit()), Refusal::EncoderOpen);
            // A grid with no threadgroups records nothing.
            encoder
                .dispatch_threadgroups([0, 4, 1], [16, 1, 1])
                .expect("an empty dispatch");
            drop(encoder);
            run(name, &command_buffer);
        });
    }
}
