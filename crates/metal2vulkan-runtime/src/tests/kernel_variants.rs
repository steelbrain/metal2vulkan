//! A kernel's variants, its `VkPipeline`s by workgroup size, threadgroup memory lengths,
//! imageblock size and tiles, under a device cfg on every device of the cfg's kind:
//!
//! - threadgroup memory lengths within one multiple of 16 bytes run one variant;
//! - dispatched at 70 keys, each in a command buffer of its own that retires before the next, a
//!   kernel keeps 64: the first 8 it made and the 56 dispatched last;
//! - one command buffer dispatches 70 keys: it holds every variant past the first 8, each
//!   dispatch runs, and once it retires the kernel's next make evicts the idle ones past 64;
//! - a loop of dispatches and retirements keeps the live variants flat between iterations 100
//!   and 200, and none is live once the device is gone;
//! - a variant evicted and dispatched again is made again, through the device's pipeline cache,
//!   on an instance without the validation layer, and both makes' times are printed.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished, on_each_device, within, Arena};
    use crate::tests::compute::on_device::{step_with_length, words, WORDS};
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::tests::validation::validation_tool;
    use crate::{
        Buffer, CommandBuffer, CommandQueue, ComputeCommandEncoder, ComputePipelineState, Counters,
        Device, Dispatch, DispatchType, Instance,
    };

    /// The keys the walks take: 35 threadgroup sizes, one row wide, at threadgroup memory lengths
    /// of 0 and 16 bytes.
    const KEYS: usize = 70;

    /// Key `k`'s threadgroup size and threadgroup memory length: 70 and past it a third length.
    fn key(k: usize) -> ([u32; 3], u32) {
        ([(k % 35) as u32 + 1, 1, 1], 16 * (k / 35) as u32)
    }

    /// Whether the pipeline of key `k`'s dispatch of `state` is made.
    fn made(state: &ComputePipelineState, k: usize) -> bool {
        let (size, length) = key(k);
        let groups = Dispatch::Threadgroups([1, 1, 1]);
        let answer = state.dispatch_variant(groups, size, &[(0, length)], [1, 1]);
        answer.expect("a key").1
    }

    /// An encoder of `command_buffer` with `state` set and `buffer` at slot 0.
    fn encoder(
        command_buffer: &CommandBuffer,
        state: &ComputePipelineState,
        buffer: &Buffer,
    ) -> ComputeCommandEncoder {
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        encoder.set_compute_pipeline_state(state).expect("a state");
        encoder.set_buffer(Some(buffer), 0, 0).expect("a buffer");
        encoder
    }

    /// One threadgroup of key `k`: each of its threads doubles and increments its word.
    fn dispatch(name: &str, encoder: &mut ComputeCommandEncoder, k: usize) {
        let (size, length) = key(k);
        encoder
            .set_threadgroup_memory_length(length, 0)
            .expect("a length");
        (encoder.dispatch_threadgroups([1, 1, 1], size))
            .unwrap_or_else(|error| panic!("{name}: key {k}: {error}"));
    }

    /// The words of a zeroed buffer once the dispatches of `keys` ran over it.
    fn expected(keys: impl IntoIterator<Item = usize>) -> Vec<u32> {
        let mut words = vec![0_u32; WORDS];
        for k in keys {
            let width = key(k).0[0] as usize;
            for word in &mut words[..width] {
                *word = word.wrapping_mul(2).wrapping_add(1);
            }
        }
        words
    }

    /// Commits `command_buffer`, which completes with no error, and waits for its retirement,
    /// past which it holds no variant (its queue's command buffer gauge falls after its holds go).
    fn retired(name: &str, counters: &Counters, command_buffer: CommandBuffer) {
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
        drop(command_buffer);
        within("the command buffer's retirement", || {
            counters.gauges().command_buffers == 0
        });
    }

    /// Dispatches `state` at each of `keys`, each in a command buffer of its own that retires
    /// before the next, over `buffer`; the live variants after each.
    fn walk(
        name: &str,
        (device, queue): (&Device, &CommandQueue),
        (state, buffer): (&ComputePipelineState, &Buffer),
        keys: impl IntoIterator<Item = usize>,
    ) -> Vec<u64> {
        let counters = device.counters();
        let mut live = Vec::new();
        for k in keys {
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = encoder(&command_buffer, state, buffer);
            dispatch(name, &mut encoder, k);
            encoder.end_encoding();
            retired(name, &counters, command_buffer);
            live.push(counters.gauges().kernel_variants);
        }
        live
    }

    /// Lengths rounded up to one multiple of 16 bytes run one variant, as Metal takes a length
    /// only in multiples of 16: 17 and 32 one, 16 and 33 others.
    #[test]
    fn every_device_runs_lengths_of_one_multiple_of_16_as_one_variant() {
        each_device("kernel_variant_lengths", |name, _, device| {
            let state = step_with_length(&device);
            let groups = Dispatch::Threadgroups([1, 1, 1]);
            let variant = |length: u32| {
                let answer = state.dispatch_variant(groups, [4, 1, 1], &[(0, length)], [1, 1]);
                answer.expect("a key")
            };
            assert_eq!(variant(17).0, variant(32).0, "{name}");
            assert_ne!(variant(16).0, variant(32).0, "{name}");
            assert_ne!(variant(33).0, variant(32).0, "{name}");
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = encoder(&command_buffer, &state, &buffer);
            encoder
                .set_threadgroup_memory_length(17, 0)
                .expect("a length");
            (encoder.dispatch_threadgroups([1, 1, 1], [4, 1, 1]))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            retired(name, &device.counters(), command_buffer);
            assert!(variant(32).1, "{name}: 32 bytes run 17's variant");
            assert_eq!(device.counters().gauges().kernel_variants, 1, "{name}");
            println!("kernel_variant_lengths: {name}: 17 and 32 bytes run one variant");
        });
    }

    /// A kernel dispatched at 70 keys, each in a command buffer of its own that retires before the
    /// next: it keeps 64, the first 8 it made and the 56 dispatched last, each key past 64
    /// evicting the least recently dispatched one past the 8. The state's variants go with it.
    #[test]
    fn every_device_keeps_64_variants_of_a_kernel_dispatched_at_70_keys() {
        each_device("kernel_variants_kept", |name, _, device| {
            let counters = device.counters();
            let state = step_with_length(&device);
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = device.new_command_queue(4).expect("a queue");
            let live = walk(name, (&device, &queue), (&state, &buffer), 0..KEYS);
            let kept: Vec<usize> = (0..KEYS).filter(|&k| made(&state, k)).collect();
            let first_and_last: Vec<usize> = (0..8).chain(14..KEYS).collect();
            assert_eq!(
                kept, first_and_last,
                "{name}: the keys whose variant is kept"
            );
            let bound: Vec<u64> = (1..=KEYS as u64).map(|n| n.min(64)).collect();
            assert_eq!(live, bound, "{name}: the live variants after each key");
            assert_eq!(
                words(&buffer),
                expected(0..KEYS),
                "{name}: the dispatches' words"
            );
            drop(state);
            assert_eq!(counters.gauges().kernel_variants, 0, "{name}");
            drop(device);
            println!(
                "M2V-MATRIX gauge kernel_variants_kept/kernel_variants@drop {}",
                counters.gauges().kernel_variants
            );
        });
    }

    /// One command buffer dispatches a kernel at 70 keys: it holds each variant past the 8 cells,
    /// so none is evicted and every dispatch runs. Once it retires, the kernel's next make evicts
    /// the idle variants past 64.
    #[test]
    fn every_device_runs_70_variants_of_a_kernel_in_one_command_buffer() {
        each_device("kernel_variants_held", |name, _, device| {
            let counters = device.counters();
            let live = || counters.gauges().kernel_variants;
            let state = step_with_length(&device);
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = encoder(&command_buffer, &state, &buffer);
            for k in 0..KEYS {
                dispatch(name, &mut encoder, k);
            }
            assert_eq!(live(), KEYS as u64, "{name}: work holds every variant");
            encoder.end_encoding();
            retired(name, &counters, command_buffer);
            assert_eq!(
                words(&buffer),
                expected(0..KEYS),
                "{name}: the dispatches' words"
            );
            assert_eq!(live(), KEYS as u64, "{name}: nothing made since");
            let after = walk(name, (&device, &queue), (&state, &buffer), [KEYS]);
            assert_eq!(after, [64], "{name}: the next make keeps 64");
            drop(state);
            assert_eq!(live(), 0, "{name}");
            drop(device);
            println!(
                "M2V-MATRIX gauge kernel_variants_held/kernel_variants@drop {}",
                counters.gauges().kernel_variants
            );
        });
    }

    /// A loop that dispatches, retires and dispatches again: 200 iterations, each one command
    /// buffer that dispatches four of 70 keys, each iteration one key further, so past the first
    /// laps each iteration makes a key its kernel evicted. The live variants read the same after
    /// iterations 100 and 200, 64, and zero once the device is gone.
    #[test]
    fn every_device_s_kernel_variants_hold_flat_across_a_dispatch_and_retire_loop() {
        each_device("kernel_variants_loop", |name, _, device| {
            let counters = device.counters();
            let state = step_with_length(&device);
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = device.new_command_queue(4).expect("a queue");
            let mut read = Vec::new();
            for iteration in 1..=200 {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = encoder(&command_buffer, &state, &buffer);
                for k in 0..4 {
                    dispatch(name, &mut encoder, (iteration + k) % KEYS);
                }
                encoder.end_encoding();
                retired(name, &counters, command_buffer);
                if iteration % 100 == 0 {
                    read.push(counters.gauges().kernel_variants);
                }
            }
            println!(
                "kernel_variants_loop: {name}: kernel variants after iterations 100 and 200: \
                 {read:?}"
            );
            assert_eq!(read, [64, 64], "{name}");
            drop(state);
            drop(device);
            println!(
                "M2V-MATRIX gauge kernel_variants_loop/kernel_variants@drop {}",
                counters.gauges().kernel_variants
            );
        });
    }

    /// A kernel's variant made, evicted past the 64 its kernel keeps, and dispatched again: made
    /// again, through the device's pipeline cache. Both makes' times are printed, from their
    /// `VariantMake`s. On an instance without the validation layer, whose checks of each create
    /// would be the time read.
    #[test]
    fn every_device_makes_an_evicted_variant_again() {
        let instance = Instance::new(&[]).expect("an instance at the floor");
        for (name, physical) in devices_of_the_cfgs_kind(&instance) {
            let tool = validation_tool(&instance, &name, physical.raw());
            println!("kernel_variant_remade: {name}: validation tool {tool:?}");
        }
        on_each_device("kernel_variant_remade", &instance, |name, _, device| {
            let counters = device.counters();
            let state = step_with_length(&device);
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = device.new_command_queue(4).expect("a queue");
            let (size, length) = key(8);
            let groups = Dispatch::Threadgroups([1, 1, 1]);
            let answer = state.dispatch_variant(groups, size, &[(0, length)], [1, 1]);
            let variant = answer.expect("a key").0;
            let from = (counters.variant_makes(0).last()).map_or(0, |make| make.number + 1);
            walk(name, (&device, &queue), (&state, &buffer), 0..65);
            assert!(!made(&state, 8), "{name}: key 8 is evicted by the 65th");
            walk(name, (&device, &queue), (&state, &buffer), [8]);
            let makes: Vec<_> = (counters.variant_makes(from).into_iter())
                .filter(|make| make.variant == variant)
                .collect();
            assert_eq!(makes.len(), 2, "{name}: key 8 made twice: {makes:?}");
            assert!(makes.iter().all(|make| make.made), "{name}: {makes:?}");
            let micros = |at: usize| (makes[at].ended - makes[at].began) as f64 / 1000.0;
            println!(
                "kernel_variant_remade: {name}: key 8 made in {:.1} us, evicted, and made again in \
                 {:.1} us",
                micros(0),
                micros(1)
            );
        });
    }
}
