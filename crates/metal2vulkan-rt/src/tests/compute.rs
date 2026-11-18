//! Compute encoders, under a device cfg on every device of the cfg's kind: dispatches over bound
//! buffers and `set_bytes`, parameters past the push window
//! through the ring, indirect dispatch, barriers only at hazards in a serial encoder and only on
//! request in a concurrent one, end barriers that reach the host only after Shared writes and
//! signals, a dispatch of a kernel without effects that records nothing, and Metal's refusals.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::hazard::Access;
    use crate::pipeline::{Constant, Field, FieldValue, Interface, Memory, OwnMemory, Place};
    use crate::tests::command::on_device::{each_device, finished, shared_event, Arena, PATIENCE};
    use crate::tests::kernels;
    use crate::{
        Buffer, CommandBuffer, ComputeCommandEncoder, ComputePipelineState, Device, DispatchType,
        Error, Gap, MemoryClass, Refusal,
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
            value: FieldValue::BufferAddress { slot, mask: 0 },
        }
    }

    fn interface(buffers: Vec<(u32, Access)>, fields: Vec<Field>, push_len: u32) -> Interface {
        Interface {
            buffers,
            textures: Vec::new(),
            numbered: Vec::new(),
            constexprs: Vec::new(),
            reads_unbound: false,
            writes_unbound: false,
            has_effects: true,
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
            memory: Memory {
                own: OwnMemory::Static(0),
                vote: 0,
            },
            multiples: Vec::new(),
            required_subgroup_size: None,
            stage_flags: ash::vk::PipelineShaderStageCreateFlags::empty(),
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

    /// The binder pushes a bound buffer's address rounded down to a multiple of the alignment the
    /// kernel takes it to have (`FieldValue::BufferAddress`'s mask, `RuntimeFact::AddressAlignment`
    /// in a translated kernel), where Metal's API validation refuses the bind offset. At each
    /// alignment, 2, 4, 8 and 16, a buffer bound 1, 3, 4, 8, 12 or 15 bytes past a multiple of 16
    /// (each offset that is not a multiple of the alignment and rounds down to a word: 15 binds)
    /// is stepped from its address rounded down to the alignment, and no other word changes. An
    /// 8- or 16-byte alignment's bind 4, 8 or 12 bytes past a multiple of 16 is a word apart from
    /// its rounded address, so a bind not rounded steps other words on every device.
    #[test]
    fn every_device_rounds_a_bound_address_down_to_its_alignment() {
        each_device("compute_rounded_address", |name, _, device| {
            let mut arena = Arena::new(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let mut runs = 0;
            for align in [2u64, 4, 8, 16] {
                let fields = vec![
                    region(),
                    Field {
                        place: Place::Push(32),
                        value: FieldValue::BufferAddress {
                            slot: 0,
                            mask: align - 1,
                        },
                    },
                ];
                let buffers = vec![(0, Access::ReadWrite)];
                let state = ComputePipelineState::for_test(
                    &device,
                    &kernels::STEP,
                    interface(buffers, fields, 40),
                )
                .expect("the step kernel");
                for at in [1u64, 3, 4, 8, 12, 15] {
                    let rounded = (32 + at) & !(align - 1);
                    if at % align == 0 || rounded % 4 != 0 {
                        continue;
                    }
                    let x = arena.buffer();
                    assert_eq!(x.gpu_address() % 16, 0, "{name}: the buffer's address");
                    write_words(&x, |index| index);
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    let mut encoder = command_buffer
                        .compute_command_encoder(DispatchType::Serial)
                        .expect("an encoder");
                    encoder.set_compute_pipeline_state(&state).expect("a state");
                    encoder.set_buffer(Some(&x), 32 + at, 0).expect("a buffer");
                    encoder
                        .dispatch_threadgroups([1, 1, 1], [16, 1, 1])
                        .expect("a dispatch");
                    encoder.end_encoding();
                    run(name, &command_buffer);
                    let stepped = rounded as u32 / 4..rounded as u32 / 4 + 16;
                    let expected: Vec<u32> = (0..WORDS as u32)
                        .map(|index| match stepped.contains(&index) {
                            true => index * 2 + 1,
                            false => index,
                        })
                        .collect();
                    assert_eq!(
                        words(&x),
                        expected,
                        "{name}: alignment {align}, bound at 32 + {at}"
                    );
                    runs += 1;
                }
            }
            assert_eq!(runs, 15, "{name}: the binds");
            println!("{name}: {runs} binds off their alignment, each run at the rounded address");
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

    /// A segment's end barrier reaches the host only where the segment wrote Shared memory or
    /// signals an event, and the host reads the bytes once the command buffer completes or it sees
    /// the event. A segment that wrote only Private memory leaves the host out, and a later command
    /// buffer still reads its writes.
    #[test]
    fn every_device_reaches_the_host_only_after_shared_writes_and_signals() {
        each_device("compute_host_reach", |name, _, device| {
            let mut arena = Arena::new(&device);
            let counters = device.counters();
            let (add, step) = (add(&device), step(&device));
            let queue = device.new_command_queue(4).expect("a queue");
            let block = device
                .take_block(MemoryClass::Private, 1)
                .expect("a Private block");
            let private = device
                .new_buffer(block.id, 0, 256, true)
                .expect("a Private buffer");
            let bytes: Vec<u8> = (0..WORDS as u32)
                .flat_map(|index| (100 + index).to_le_bytes())
                .collect();
            let encode_step = |command_buffer: &CommandBuffer, buffer: &Buffer| {
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&step).expect("a state");
                encoder.set_buffer(Some(buffer), 0, 0).expect("a buffer");
                encoder
                    .dispatch_threadgroups([4, 1, 1], [16, 1, 1])
                    .expect("a dispatch");
                encoder.end_encoding();
            };
            // `o = private + 100 + index`, into Shared memory.
            let encode_add = |command_buffer: &CommandBuffer, o: &Buffer| {
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&add).expect("a state");
                encoder.set_buffer(Some(&private), 0, 0).expect("a buffer");
                encoder.set_bytes(&bytes, 1).expect("bytes");
                encoder.set_buffer(Some(o), 0, 2).expect("a buffer");
                encoder
                    .dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1])
                    .expect("a dispatch");
                encoder.end_encoding();
            };

            // Private only: the end barrier leaves the host out.
            let before = counters.host_reaches();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            encode_step(&command_buffer, &private);
            run(name, &command_buffer);
            assert_eq!(counters.host_reaches(), before, "{name}: Private only");

            // A later command buffer adds the Private words into Shared memory: the host reads
            // them once it completes.
            let o = arena.buffer();
            let before = counters.host_reaches();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            encode_add(&command_buffer, &o);
            run(name, &command_buffer);
            assert_eq!(counters.host_reaches(), before + 1, "{name}: Shared write");
            let expected: Vec<u32> = (0..WORDS as u32).map(|index| 101 + index).collect();
            assert_eq!(words(&o), expected, "{name}: after Private only");

            // A Private write before a signal reaches the host; the add after it reads 3.
            let signaled = shared_event(&device);
            let o = arena.buffer();
            let before = counters.host_reaches();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            encode_step(&command_buffer, &private);
            command_buffer
                .encode_signal_event(signaled.as_event(), 1)
                .expect("a signal");
            encode_add(&command_buffer, &o);
            run(name, &command_buffer);
            assert!(signaled.wait_until_signaled_value(1, PATIENCE), "{name}");
            assert_eq!(counters.host_reaches(), before + 2, "{name}: signal");
            let expected: Vec<u32> = (0..WORDS as u32).map(|index| 103 + index).collect();
            assert_eq!(words(&o), expected, "{name}: after a signal");

            // A Shared write before a signal: the host reads it once it sees the event, while the
            // command buffer waits on the host behind it.
            let x = arena.buffer();
            write_words(&x, |index| index);
            let (signaled, release) = (shared_event(&device), shared_event(&device));
            let before = counters.host_reaches();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            encode_step(&command_buffer, &x);
            command_buffer
                .encode_signal_event(signaled.as_event(), 1)
                .expect("a signal");
            command_buffer
                .encode_wait_for_event(release.as_event(), 1)
                .expect("a wait");
            command_buffer.commit().expect("a commit");
            assert!(signaled.wait_until_signaled_value(1, PATIENCE), "{name}");
            let expected: Vec<u32> = (0..WORDS as u32).map(|index| 2 * index + 1).collect();
            assert_eq!(words(&x), expected, "{name}: at the signal");
            release.set_signaled_value(1);
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            assert_eq!(
                counters.host_reaches(),
                before + 1,
                "{name}: Shared write and signal"
            );
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

    /// A dispatch of a kernel that writes no memory (`Interface::has_effects` false, from the
    /// translator's `PipelineInterface::writes_memory`) records nothing: given the step kernel's
    /// words under such an interface, a direct and an indirect dispatch leave its buffer as it was,
    /// a third with no buffer bound owes no bind, and each is counted.
    #[test]
    fn every_device_records_nothing_for_a_dispatch_of_a_kernel_without_effects() {
        each_device("compute_without_effects", |name, _, device| {
            let mut arena = Arena::new(&device);
            let (x, arguments) = (arena.buffer(), arena.buffer());
            write_words(&x, |index| index);
            write_words(&arguments, |index| match index {
                0..=2 => 1,
                _ => 0,
            });
            let fields = vec![region(), address(0, Place::Push(32))];
            let mut quiet = interface(vec![(0, Access::ReadWrite)], fields, 40);
            quiet.has_effects = false;
            let state = ComputePipelineState::for_test(&device, &kernels::STEP, quiet)
                .expect("the step kernel");
            let before = device.counters().metrics().dispatches_without_effects;
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            encoder
                .dispatch_threadgroups([1, 1, 1], [16, 1, 1])
                .expect("an unbound dispatch");
            encoder.set_buffer(Some(&x), 0, 0).expect("a buffer");
            encoder
                .dispatch_threadgroups([4, 1, 1], [16, 1, 1])
                .expect("a dispatch");
            encoder
                .dispatch_threadgroups_with_indirect_buffer(&arguments, 0, [16, 1, 1])
                .expect("an indirect dispatch");
            encoder.end_encoding();
            run(name, &command_buffer);
            let expected: Vec<u32> = (0..WORDS as u32).collect();
            assert_eq!(words(&x), expected, "{name}: the words");
            assert_eq!(
                device.counters().metrics().dispatches_without_effects - before,
                3,
                "{name}: the dispatches counted"
            );
        });
    }

    /// An indirect dispatch after a write that no barrier since has ordered before indirect
    /// argument reads gets a barrier that does: a dispatch writes the arguments, a direct
    /// dispatch that reads them follows (its barrier reaches shader accesses alone), and the
    /// indirect dispatch, which conflicts with neither, reads them.
    #[test]
    fn every_device_orders_an_indirect_dispatch_after_the_write_of_its_arguments() {
        each_device("compute_indirect_after_write", |name, _, device| {
            let mut arena = Arena::new(&device);
            let (a, b, arguments, c, o) = (
                arena.buffer(),
                arena.buffer(),
                arena.buffer(),
                arena.buffer(),
                arena.buffer(),
            );
            write_words(&a, |index| if index == 4 { 3 } else { index * 10 });
            write_words(&b, |index| match index {
                4..=6 => 1,
                _ => 0,
            });
            for buffer in [&arguments, &c, &o] {
                write_words(buffer, |_| 0);
            }
            let state = add(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let before = device.counters().metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            let bind = |encoder: &mut ComputeCommandEncoder, buffers: [&Buffer; 3]| {
                for (slot, buffer) in buffers.into_iter().enumerate() {
                    encoder
                        .set_buffer(Some(buffer), 0, slot as u32)
                        .expect("a buffer");
                }
            };
            // The arguments, a + b: (4, 1, 1) threadgroups at words 4 to 6.
            bind(&mut encoder, [&a, &b, &arguments]);
            encoder
                .dispatch_threadgroups([4, 1, 1], [16, 1, 1])
                .expect("a dispatch");
            // A direct dispatch that reads them, after a barrier for shader accesses alone.
            bind(&mut encoder, [&arguments, &b, &c]);
            encoder
                .dispatch_threadgroups([4, 1, 1], [16, 1, 1])
                .expect("a dispatch");
            // The indirect dispatch: a barrier that reaches its argument read.
            bind(&mut encoder, [&a, &b, &o]);
            encoder
                .dispatch_threadgroups_with_indirect_buffer(&arguments, 16, [16, 1, 1])
                .expect("an indirect dispatch");
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = device.counters().metrics();
            // The direct dispatch's, the indirect dispatch's, and the segment's end.
            assert_eq!(after.barriers - before.barriers, 3, "{name}");
            let sum = |index: u32| {
                if index == 4 {
                    4
                } else {
                    index * 10 + u32::from((4..=6).contains(&index))
                }
            };
            let expected: Vec<u32> = (0..WORDS as u32).map(sum).collect();
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
            // A kernel without a partial-threadgroups variant runs no partial threadgroup.
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
