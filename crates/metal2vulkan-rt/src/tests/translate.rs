//! Translated kernels, under a device cfg on every device of the cfg's kind. A kernel that does not
//! observe its threadgroup partition runs every direct dispatch in tiles of 256 threads on the one
//! `VkPipeline` made with its state, and an indirect dispatch at its own threadgroup size; a
//! precise one answers the same words in a tile as at the app's shape. A kernel that does observe it runs whole grids at
//! their own size, a `dispatchThreads` grid that is not a whole number of threadgroups on its
//! partial-threadgroups variant, and a threadgroup count past the device's limit split. Each
//! answer is checked against the program's, and each `VkPipeline` made after creation is a
//! first-use wait the counters record. Vector accesses through pointers cast from a `uint`
//! buffer land at their own addresses, 4, 8 or 12 bytes past a multiple of 16, as Metal places them.
//! An access through a buffer argument at 2 GiB and past lands there, on every device that can
//! allocate the buffer.
//!
//! `llvm-as` assembles the kernels; without it a test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished, within};
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{
        host_time, Block, Buffer, ComputePipelineDescriptor, ComputePipelineState, Device,
        DeviceOptions, DispatchType, Error, FunctionDescriptor, Gauges, Instance, Library, Limit,
        MemoryClass, PrecisionNote, Refusal, StageSource, MAX_BUFFER_LENGTH,
    };
    use metal2vulkan::pipeline::{RuntimeFact, StageEntries};
    use metal2vulkan::Budget;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};
    use std::path::{Path, PathBuf};
    use std::time::SystemTime;

    /// `float` patterns none of which is a NaN or a denormal (the fixture compiles with fast math).
    const FLOATS: [f32; 6] = [0.0, -0.0, 0.5, 2.0, 1.0, -3.0];
    /// The same values' `half` bits.
    const HALVES: [u16; 6] = [0x0000, 0x8000, 0x3800, 0x4000, 0x3c00, 0xc200];
    /// Threads in a threadgroup, and threadgroups, of the narrow dispatches.
    const GROUP: u32 = 4;
    const GROUPS: u32 = 3;
    /// Threads of the wide grid: a 256-wide tile and a partial one.
    const WIDE: u32 = 300;
    /// The bind offsets of the three buffers, so a dropped offset shows.
    const OFFSETS: [u64; 3] = [32, 16, 4];

    /// The kernel `name` of the engine fixture `name.ll`, or None where `llvm-as` is missing.
    fn descriptor(name: &str) -> Option<ComputePipelineDescriptor> {
        descriptor_of(name, name)
    }

    /// The kernel `entry` of the engine fixture `fixture.ll`, or None where `llvm-as` is missing.
    fn descriptor_of(fixture: &str, entry: &str) -> Option<ComputePipelineDescriptor> {
        let bitcode = match llvm_assemble(&engine_dir().join(format!("{fixture}.ll"))) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return None;
            }
            Err(error) => panic!("{fixture}: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function = library.function(entry).expect("the kernel");
        Some(ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        )))
    }

    /// The state of `descriptor` on `device`, printing what the translator gave where it is
    /// refused: the plain module, and for a `Function` kernel its partial-threadgroups variant.
    fn state(
        name: &str,
        device: &Device,
        descriptor: &ComputePipelineDescriptor,
    ) -> ComputePipelineState {
        device
            .new_compute_pipeline_state(descriptor, crate::tests::no_constexpr_samplers)
            .unwrap_or_else(|error| {
                let translator = &device.core().translator;
                let budget = Budget::standard();
                let mut translated = vec![(
                    "plain",
                    translator.compile_compute_pipeline(descriptor, &budget),
                )];
                if let StageSource::Function(function) = &descriptor.kernel {
                    let mut inputs = descriptor.lowering_inputs();
                    inputs.partial_threadgroups = true;
                    let variant = translator
                        .compile_function(function, &inputs, &budget)
                        .and_then(|compiled| {
                            let mut variant = descriptor.clone();
                            variant.kernel = StageSource::Compiled(compiled);
                            translator.compile_compute_pipeline(&variant, &budget)
                        });
                    translated.push(("partial-threadgroups variant", variant));
                }
                for (which, pipeline) in translated {
                    match pipeline {
                        Ok(pipeline) => eprintln!(
                            "{name}: {which}: interface {:?}\nconstants {:?}\ndependencies {:?}\nsets {:?}\nflags {:?} subgroup {:?} blob {}",
                            pipeline.interface,
                            pipeline.stage.specialization.constants,
                            pipeline.dependencies,
                            pipeline.layout.sets,
                            pipeline.stage.stage_flags,
                            pipeline.stage.required_subgroup_size,
                            pipeline.stage.constant_blob.is_some(),
                        ),
                        Err(refused) => eprintln!("{name}: {which}: refused: {refused}"),
                    }
                }
                panic!("{name}: {error}")
            })
    }

    /// The facts the runtime reads of `descriptor`'s kernel, one that observes its threadgroup
    /// partition, on `device`, lowered as the state lowers it: of its plain module, under the
    /// inputs the kernel reads, and of its emulated-SIMD-group variant, the same inputs with
    /// emulated SIMD groups.
    fn facts_read(
        name: &str,
        device: &Device,
        descriptor: &ComputePipelineDescriptor,
    ) -> [Vec<RuntimeFact>; 2] {
        let StageSource::Function(function) = &descriptor.kernel else {
            unreachable!("a fixture's kernel is a function");
        };
        let translator = &device.core().translator;
        let budget = Budget::standard();
        let entries = StageEntries::new();
        let plain = translator
            .kernel_inputs_read(function, &descriptor.lowering_inputs(), &budget, &entries)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut emulated = plain.clone();
        emulated.emulated_simdgroups = true;
        [plain, emulated].map(|inputs| {
            let compiled = translator
                .compile_function_from_entries(function, &inputs, &budget, &entries)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut lowered = descriptor.clone();
            lowered.kernel = StageSource::Compiled(compiled);
            translator
                .compile_compute_pipeline(&lowered, &budget)
                .unwrap_or_else(|error| panic!("{name}: {error}"))
                .dependencies
                .iter()
                .map(|dependency| dependency.fact)
                .collect()
        })
    }

    /// Whether the reaper made `state`'s emulated-SIMD-group module.
    fn emulated_made(state: &ComputePipelineState) -> bool {
        use crate::pipeline::Lowered;

        state
            .inner
            .compiled
            .emulated
            .as_ref()
            .and_then(|emulated| emulated.lowered.get())
            .is_some_and(|lowered| matches!(lowered, Lowered::Made(..)))
    }

    /// Zeroed Shared buffers in one block, each at a 256-byte boundary.
    struct Buffers {
        device: Device,
        block: Block,
        next: u64,
    }

    impl Buffers {
        fn new(device: &Device, len: u64) -> Buffers {
            let block = device
                .take_block(MemoryClass::Shared, len)
                .expect("a Shared block");
            Buffers {
                device: device.clone(),
                block,
                next: 0,
            }
        }

        fn buffer(&mut self, len: u64) -> Buffer {
            let buffer = self
                .device
                .new_buffer(self.block.id, self.next, len, Some(0))
                .expect("a buffer");
            self.next += len.next_multiple_of(256);
            buffer
        }
    }

    /// Writes `bytes` at `offset` of a Shared buffer.
    fn write(buffer: &Buffer, offset: u64, bytes: &[u8]) {
        let contents = buffer.contents().expect("a Shared buffer");
        assert!(offset as usize + bytes.len() <= buffer.length() as usize);
        // SAFETY: the range lies in the buffer, which is alive, and no GPU work uses it.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                contents.as_ptr().add(offset as usize),
                bytes.len(),
            )
        };
    }

    /// The words of a Shared buffer from `offset`, `count` of them.
    fn words(buffer: &Buffer, offset: u64, count: usize) -> Vec<u32> {
        let contents = buffer.contents().expect("a Shared buffer");
        assert!(offset as usize + 4 * count <= buffer.length() as usize);
        (0..count)
            .map(|index| {
                // SAFETY: the word lies in the buffer, which is alive, and its work completed.
                unsafe {
                    contents
                        .as_ptr()
                        .add(offset as usize + 4 * index)
                        .cast::<u32>()
                        .read_unaligned()
                }
            })
            .collect()
    }

    fn le_bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    /// Thread `i`'s input lanes: indices into [`FLOATS`] and [`HALVES`].
    fn lanes(i: usize) -> ([usize; 4], [usize; 4]) {
        (
            std::array::from_fn(|j| (i + j) % FLOATS.len()),
            std::array::from_fn(|j| (3 * i + j) % HALVES.len()),
        )
    }

    /// What `a1_bool_from_float` writes for thread `i`: each lane's `x != 0`, the `float` lanes in
    /// bits 0-3 and 4-7, the `half` lanes in bits 8-11 and 12-15.
    fn want(i: usize) -> u32 {
        let (f, h) = lanes(i);
        let mut r = 0u32;
        for (j, (&fi, &hi)) in f.iter().zip(&h).enumerate() {
            let fb = u32::from(FLOATS[fi] != 0.0);
            let hb = u32::from(HALVES[hi] & 0x7fff != 0);
            r |= fb << j | fb << (4 + j) | hb << (8 + j) | hb << (12 + j);
        }
        r
    }

    /// `a1_bool_from_float`, which does not observe its threadgroup partition, translated: its
    /// state made the `VkPipeline` of its tiles, which a wide `dispatchThreads` grid and a narrow
    /// threadgroup dispatch both run without a wait; an indirect dispatch runs the app's own size,
    /// made by a first-use wait. Every answer is the program's, over buffers bound at offsets.
    #[test]
    fn every_device_tiles_a_kernel_that_does_not_observe_its_threadgroups() {
        let Some(descriptor) = descriptor("a1_bool_from_float") else {
            return;
        };
        each_device("compute_translated", |name, _, device| {
            let mut bits = Vec::new();
            let mut halves = Vec::new();
            for i in 0..WIDE as usize {
                let (f, h) = lanes(i);
                for (&fi, &hi) in f.iter().zip(&h) {
                    bits.extend_from_slice(&FLOATS[fi].to_bits().to_le_bytes());
                    halves.extend_from_slice(&HALVES[hi].to_le_bytes());
                }
            }
            let mut buffers = Buffers::new(&device, 64 * 1024);
            let a = buffers.buffer(OFFSETS[0] + bits.len() as u64);
            let b = buffers.buffer(OFFSETS[1] + halves.len() as u64);
            let outs: Vec<Buffer> = (0..3)
                .map(|_| buffers.buffer(OFFSETS[2] + 4 * u64::from(WIDE) + 4))
                .collect();
            let arguments = buffers.buffer(16);
            write(&a, OFFSETS[0], &bits);
            write(&b, OFFSETS[1], &halves);
            write(&arguments, 0, &le_bytes(&[GROUPS, 1, 1]));
            let made = device.counters().metrics();
            let state = state(name, &device, &descriptor);
            let created = device.counters().metrics();
            assert_eq!(
                created.pipelines_created - made.pipelines_created,
                1,
                "{name}: the tiles' pipeline, made with the state"
            );
            assert_eq!(created.first_use_waits, made.first_use_waits, "{name}");
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            encoder
                .set_buffer(Some(&a), OFFSETS[0], 0)
                .expect("a buffer");
            encoder
                .set_buffer(Some(&b), OFFSETS[1], 1)
                .expect("a buffer");
            encoder
                .set_buffer(Some(&outs[0]), OFFSETS[2], 2)
                .expect("a buffer");
            encoder
                .dispatch_threads([WIDE, 1, 1], [GROUP, 1, 1])
                .expect("a wide dispatch");
            encoder
                .set_buffer(Some(&outs[1]), OFFSETS[2], 2)
                .expect("a buffer");
            encoder
                .dispatch_threadgroups([GROUPS, 1, 1], [GROUP, 1, 1])
                .expect("a narrow dispatch");
            encoder
                .set_buffer(Some(&outs[2]), OFFSETS[2], 2)
                .expect("a buffer");
            encoder
                .dispatch_threadgroups_with_indirect_buffer(&arguments, 0, [GROUP, 1, 1])
                .expect("an indirect dispatch");
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            // The word before the offset and the ones past the last thread stay 0.
            for (out, threads) in outs.iter().zip([WIDE, GROUP * GROUPS, GROUP * GROUPS]) {
                let mut expected = vec![0];
                expected.extend((0..threads as usize).map(want));
                expected.resize(WIDE as usize + 2, 0);
                assert_eq!(
                    words(out, OFFSETS[2] - 4, WIDE as usize + 2),
                    expected,
                    "{name}: {threads} threads"
                );
            }
            let after = device.counters().metrics();
            let waits = after.first_use_waits - created.first_use_waits;
            let nanos = after.first_use_wait_nanos - created.first_use_wait_nanos;
            assert_eq!(waits, 1, "{name}: first-use waits");
            assert!(nanos > 0, "{name}: first-use wait time");
            assert!(
                after.first_use_wait_max_nanos >= nanos / 2,
                "{name}: longest wait"
            );
            assert_eq!(
                after.pipelines_created - created.pipelines_created,
                1,
                "{name}: pipelines"
            );
            println!(
                "{name}: tiled kernel checked over {WIDE} and {} threads and an indirect dispatch; first-use waits {waits}, {nanos} ns; precision notes {}",
                GROUP * GROUPS,
                state.precision_notes().len()
            );
        });
    }

    /// Metal's answer of `icb_replay_grid_template` over `grid` in threadgroups of `size`, the
    /// last along an axis running only the threads left, with `params` `[base, slots]`.
    fn grid_template(grid: [u32; 3], size: [u32; 3], params: [u32; 2]) -> Vec<u32> {
        let mut out = vec![0u32; params[1] as usize];
        let groups: [u32; 3] = std::array::from_fn(|axis| grid[axis].div_ceil(size[axis]));
        for z in 0..grid[2] {
            for y in 0..grid[1] {
                for x in 0..grid[0] {
                    let tid = [x, y, z];
                    let group: [u32; 3] = std::array::from_fn(|axis| tid[axis] / size[axis]);
                    let lid: [u32; 3] = std::array::from_fn(|axis| tid[axis] % size[axis]);
                    let threads: [u32; 3] = std::array::from_fn(|axis| {
                        size[axis].min(grid[axis] - group[axis] * size[axis])
                    });
                    let lin = lid[0] + lid[1] * threads[0] + lid[2] * threads[0] * threads[1];
                    let n = threads[0] * threads[1] * threads[2];
                    let neighbour = (lin + 1) % n;
                    let terms = [
                        params[0],
                        3 * threads[0],
                        5 * threads[1],
                        7 * size[0],
                        11 * size[1],
                        13 * group[0],
                        17 * group[1],
                        19 * groups[0],
                        23 * groups[1],
                        29 * grid[0],
                        31 * grid[1],
                        37 * lid[0],
                        41 * lid[1],
                        43 * neighbour,
                    ];
                    let v = terms.iter().fold(0u32, |sum, &term| sum.wrapping_add(term));
                    let index = x + y * grid[0] + z * grid[0] * grid[1];
                    let slot = &mut out[(index % params[1]) as usize];
                    *slot = slot.wrapping_add(v);
                }
            }
        }
        out
    }

    /// `icb_replay_grid_template`, which observes its threadgroup partition (threadgroup memory,
    /// a barrier, and every builtin of the partition), translated twice at creation: a whole
    /// `dispatchThreads` grid runs its plain module at the call's size, a ragged one its
    /// partial-threadgroups variant, and a threadgroup count past the device's limit on an axis
    /// runs split, each piece reading Metal's whole grid. Threadgroups deeper than 64, 1 by 1 by
    /// 320 and 2 by 3 by 100, run as one row of their threads on every device, whole, ragged and
    /// indirect. Each answer is Metal's, and each new size or module a first-use wait. Prints the
    /// creation's time and one lowering's.
    #[test]
    fn every_device_runs_an_observing_kernel_at_its_own_threadgroup_size() {
        let Some(descriptor) = descriptor("icb_replay_grid_template") else {
            return;
        };
        each_device("compute_observer", |name, _, device| {
            let raw = device.raw();
            // SAFETY: the physical device is the instance's.
            let limits = unsafe {
                raw.instance
                    .get_physical_device_properties(raw.physical_device)
                    .limits
            };
            // A count one past the device's on the second axis, where a test can hold its grid.
            let past = limits.max_compute_work_group_count[1]
                .checked_add(3)
                .filter(|&count| count <= 1 << 20);
            // How each run dispatches: a `dispatchThreads` grid, or threadgroup counts, direct or
            // from a buffer.
            #[derive(Clone, Copy, PartialEq)]
            enum How {
                Threads,
                Groups,
                Indirect,
            }
            let runs: Vec<([u32; 3], [u32; 3], How)> = [
                ([10, 3, 1], [4, 2, 1], How::Threads),
                ([8, 4, 1], [4, 2, 1], How::Threads),
                ([3, 2, 1], [4, 2, 1], How::Groups),
                ([1, 1, 2], [1, 1, 320], How::Groups),
                ([1, 1, 500], [1, 1, 320], How::Threads),
                ([1, 1, 2], [1, 1, 320], How::Indirect),
                ([4, 3, 200], [2, 3, 100], How::Threads),
            ]
            .into_iter()
            .chain(past.map(|count| ([1, count, 1], [1, 1, 1], How::Groups)))
            .collect();
            let mut buffers = Buffers::new(&device, 8 << 20);
            let arguments = buffers.buffer(16);
            write(&arguments, 0, &le_bytes(&[1, 1, 2]));
            let outs: Vec<(Buffer, Vec<u32>)> = runs
                .iter()
                .map(|&(grid, size, how)| {
                    // A threadgroup dispatch's grid is its counts times its size.
                    let grid = match how {
                        How::Threads => grid,
                        How::Groups | How::Indirect => {
                            std::array::from_fn(|axis| grid[axis] * size[axis])
                        }
                    };
                    let slots = grid.iter().product::<u32>();
                    let want = grid_template(grid, size, [1000, slots]);
                    (buffers.buffer(4 * u64::from(slots)), want)
                })
                .collect();
            let made = device.counters().metrics();
            let began = host_time();
            let state = state(name, &device, &descriptor);
            let creation = host_time() - began;
            let began = host_time();
            device
                .core()
                .translator
                .compile_function(
                    match &descriptor.kernel {
                        StageSource::Function(function) => function,
                        StageSource::Compiled(_) => unreachable!("a Function descriptor"),
                    },
                    &descriptor.lowering_inputs(),
                    &Budget::standard(),
                )
                .expect("the plain lowering");
            let lowering = host_time() - began;
            let created = device.counters().metrics();
            assert_eq!(
                created.pipelines_created, made.pipelines_created,
                "{name}: an observing kernel's pipelines wait for their sizes"
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            for (&(grid, size, how), (out, want)) in runs.iter().zip(&outs) {
                let n = size.iter().product::<u32>();
                encoder
                    .set_threadgroup_memory_length((4 * n).next_multiple_of(16), 0)
                    .expect("a length");
                encoder
                    .set_bytes(&le_bytes(&[1000, want.len() as u32]), 0)
                    .expect("the parameters");
                encoder.set_buffer(Some(out), 0, 1).expect("a buffer");
                match how {
                    How::Threads => encoder.dispatch_threads(grid, size),
                    How::Groups => encoder.dispatch_threadgroups(grid, size),
                    How::Indirect => {
                        encoder.dispatch_threadgroups_with_indirect_buffer(&arguments, 0, size)
                    }
                }
                .unwrap_or_else(|error| panic!("{name}: {grid:?} by {size:?}: {error}"));
            }
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            for (&(grid, size, _), (out, want)) in runs.iter().zip(&outs) {
                let got = words(out, 0, want.len());
                let first = got.iter().zip(want).position(|(got, want)| got != want);
                assert_eq!(
                    first, None,
                    "{name}: {grid:?} by {size:?}: the first word that differs"
                );
            }
            let after = device.counters().metrics();
            // The ragged grid's variant, the whole grid's size, the one-thread size; the deep
            // size, its ragged grid's variant (its indirect dispatch runs the whole one) and the
            // second deep size.
            let sizes = 5 + u64::from(past.is_some());
            assert_eq!(
                after.first_use_waits - created.first_use_waits,
                sizes,
                "{name}: first-use waits"
            );
            println!(
                "{name}: observing kernel checked over {} runs{}; creation {creation} ns, one lowering {lowering} ns; precision notes {}",
                runs.len(),
                match past {
                    Some(count) => format!(", {count} threadgroups split"),
                    None => String::from(", no split (the device's count is past a test's grid)"),
                },
                state.precision_notes().len()
            );
        });
    }

    /// Float patterns from a fixed generator: every sign, exponent and mantissa.
    fn patterns(count: usize, seed: u32) -> Vec<u32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                state
            })
            .collect()
    }

    /// `a1_sign_and_division_precise`, which does not observe its threadgroup partition and
    /// carries no fast-math note, answers the same words in a tile as at the app's own shape (an
    /// indirect dispatch's size), a NaN's bits aside: those are the device's own
    /// (`NativeNanEncoding`), and a device may take either operand's payload per pipeline. Its
    /// fast-math twin's state carries `FastMathRewrites`, the note of the rounding a device may
    /// choose per pipeline.
    #[test]
    fn every_device_answers_a_precise_kernel_alike_in_a_tile_and_at_the_apps_shape() {
        let (Some(precise), Some(fast)) = (
            descriptor("a1_sign_and_division_precise"),
            descriptor("a1_sign_and_division"),
        ) else {
            return;
        };
        const THREADS: u32 = 4096;
        const SHAPE: u32 = 32;
        each_device("compute_precise", |name, _, device| {
            let fast = state(name, &device, &fast);
            assert!(
                fast.precision_notes()
                    .contains(&PrecisionNote::FastMathRewrites),
                "{name}: the fast-math twin's notes {:?}",
                fast.precision_notes()
            );
            let state = state(name, &device, &precise);
            assert!(
                !state
                    .precision_notes()
                    .contains(&PrecisionNote::FastMathRewrites),
                "{name}: the precise kernel's notes {:?}",
                state.precision_notes()
            );
            let threads = THREADS as usize;
            let mut buffers = Buffers::new(&device, 1 << 20);
            let inputs = [
                le_bytes(&patterns(4 * threads, 1)),
                le_bytes(&patterns(2 * threads, 2)),
                le_bytes(&patterns(2 * threads, 3)),
            ];
            let bound: Vec<Buffer> = inputs
                .iter()
                .map(|bytes| {
                    let buffer = buffers.buffer(bytes.len() as u64);
                    write(&buffer, 0, bytes);
                    buffer
                })
                .collect();
            let outs = [buffers.buffer(16 * 4096), buffers.buffer(16 * 4096)];
            let arguments = buffers.buffer(16);
            write(&arguments, 0, &le_bytes(&[THREADS / SHAPE, 1, 1]));
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            for (index, buffer) in bound.iter().enumerate() {
                encoder
                    .set_buffer(Some(buffer), 0, index as u32)
                    .expect("a buffer");
            }
            encoder.set_buffer(Some(&outs[0]), 0, 3).expect("a buffer");
            encoder
                .dispatch_threads([THREADS, 1, 1], [SHAPE, 1, 1])
                .expect("a tiled dispatch");
            encoder.set_buffer(Some(&outs[1]), 0, 3).expect("a buffer");
            encoder
                .dispatch_threadgroups_with_indirect_buffer(&arguments, 0, [SHAPE, 1, 1])
                .expect("a dispatch at the app's shape");
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let (tiled, shaped) = (
                words(&outs[0], 0, 4 * threads),
                words(&outs[1], 0, 4 * threads),
            );
            // Words 0 to 2 are `float`s, word 3 two `half`s; two NaNs are alike where the state
            // notes NaN bits as the device's own.
            let nan_bits = state
                .precision_notes()
                .contains(&PrecisionNote::NativeNanEncoding);
            let alike = |word: usize, a: u32, b: u32| {
                let float_nans = || f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan();
                let half_nan = |bits: u32| bits & 0x7c00 == 0x7c00 && bits & 0x3ff != 0;
                let halves_alike = || {
                    [0, 16].iter().all(|&at| {
                        let (a, b) = ((a >> at) & 0xffff, (b >> at) & 0xffff);
                        a == b || (half_nan(a) && half_nan(b))
                    })
                };
                let nans = match word % 4 {
                    3 => halves_alike(),
                    _ => float_nans(),
                };
                a == b || (nan_bits && nans)
            };
            let first = (0..tiled.len()).find(|&word| !alike(word, tiled[word], shaped[word]));
            if let Some(word) = first {
                let thread = word / 4;
                let pair = &inputs[1][8 * thread..8 * thread + 8];
                let (a, b) = (
                    f32::from_le_bytes(pair[..4].try_into().expect("4 bytes")),
                    f32::from_le_bytes(pair[4..].try_into().expect("4 bytes")),
                );
                panic!(
                    "{name}: word {word} (thread {thread}, word {} of its 4) answers {:#010x} in a tile and {:#010x} at {SHAPE} by 1; the division's operands {a:e} ({:#010x}) / {b:e} ({:#010x}); the state's notes {:?}",
                    word % 4,
                    tiled[word],
                    shaped[word],
                    a.to_bits(),
                    b.to_bits(),
                    state.precision_notes()
                );
            }
            println!(
                "{name}: precise kernel alike over {THREADS} threads in a tile and at {SHAPE} by 1; fast-math twin notes {:?}",
                fast.precision_notes()
            );
        });
    }

    /// Lane `lane`'s input of the SIMD-group products: two of magnitude 2 and six of 0.5 at most
    /// in a SIMD group, so every product and prefix product is exact in any order.
    fn simd_input(lane: u32) -> f32 {
        let sign = if lane % 2 == 0 { 1.0 } else { -1.0 };
        match lane {
            0..8 => 2.0 * sign,
            8..14 => 0.5 * sign,
            _ if lane % 3 == 0 => -1.0,
            _ => 1.0,
        }
    }

    /// `simd_float_products` of `gsm2_simd_products`: SIMD-group products and prefix products
    /// over each thread's input, in threadgroups of 64 threads (two whole SIMD groups), of 32, of
    /// 80 (a partial SIMD group of 16 last) and of one. Where the device lets compute pin a
    /// subgroup of 32 (compute in `requiredSubgroupSizeStages`, 32 between its sizes), the
    /// translator's tier for it is Direct: a pinned subgroup of 32 per SIMD group, its local size
    /// flattened to whole SIMD groups, which runs the 64 and the 32. A device of 64-lane
    /// subgroups that cannot pin runs each SIMD group as half of one, and only whole subgroups:
    /// the 64. Whatever the module runs only whole groups of, a threadgroup of another size runs
    /// the emulated-SIMD-group module the reaper lowered. Every answer is Metal's: the products
    /// over each SIMD group's threads, exact in any order for these inputs. A device that cannot
    /// pin 32 may take a tier that is refused.
    #[test]
    fn every_device_runs_simd_groups_at_metals_width() {
        use crate::pipeline::Lowered;
        use crate::{Error, Gap};
        use ash::vk;

        let Some(descriptor) = descriptor_of("gsm2_simd_products", "simd_float_products") else {
            return;
        };
        const GROUPS: u32 = 3;
        each_device("compute_simd", |name, _, device| {
            let raw = device.raw();
            let mut vulkan13 = vk::PhysicalDeviceVulkan13Properties::default();
            let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut vulkan13);
            // SAFETY: the physical device is the instance's, a Vulkan 1.3 one.
            unsafe {
                raw.instance
                    .get_physical_device_properties2(raw.physical_device, &mut properties)
            };
            let pinnable = vulkan13
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
                && (vulkan13.min_subgroup_size..=vulkan13.max_subgroup_size).contains(&32);
            let state = match device
                .new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers)
            {
                Ok(state) => state,
                Err(Error::Unsupported(Gap::KernelInput)) if !pinnable => {
                    println!(
                        "{name}: SIMD groups not pinnable at 32; the tier's kernel is refused"
                    );
                    return;
                }
                Err(error) => panic!("{name}: {error}"),
            };
            let interface = &state.inner.compiled.whole.interface;
            if pinnable {
                assert_eq!(
                    (interface.required_subgroup_size, &interface.multiples[..]),
                    (Some(32), &[32][..]),
                    "{name}: the Direct tier, a pinned subgroup of 32 per SIMD group"
                );
            }
            let queue = device.new_command_queue(4).expect("a queue");
            for size in [64, 32, 80, 1] {
                let threads = GROUPS * size;
                let input: Vec<u8> = (0..threads)
                    .flat_map(|t| simd_input(t % 32).to_le_bytes())
                    .collect();
                let mut buffers = Buffers::new(&device, 1 << 16);
                let a = buffers.buffer(u64::from(threads) * 4);
                let out = buffers.buffer(u64::from(threads) * 16);
                write(&a, 0, &input);
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                encoder.set_buffer(Some(&a), 0, 0).expect("a buffer");
                encoder.set_buffer(Some(&out), 0, 1).expect("a buffer");
                encoder
                    .dispatch_threadgroups([GROUPS, 1, 1], [size, 1, 1])
                    .unwrap_or_else(|error| panic!("{name}: threadgroups of {size}: {error}"));
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
                let mut want = Vec::new();
                for t in 0..threads {
                    let first = t - t % size;
                    let base = t - (t - first) % 32;
                    let group = (base..(first + size).min(base + 32)).map(|u| simd_input(u % 32));
                    let product: f32 = group.clone().product();
                    let inclusive: f32 = group.clone().take((t - base + 1) as usize).product();
                    let exclusive: f32 = group.clone().take((t - base) as usize).product();
                    let negated: f32 = group.map(|x| -x).product();
                    want.extend([product, inclusive, exclusive, negated].map(f32::to_bits));
                }
                assert_eq!(
                    words(&out, 0, want.len()),
                    want,
                    "{name}: the products in threadgroups of {size}"
                );
            }
            let emulated = state
                .inner
                .compiled
                .emulated
                .as_ref()
                .and_then(|emulated| emulated.lowered.get());
            if !interface.multiples.is_empty() {
                assert!(
                    matches!(emulated, Some(Lowered::Made(..))),
                    "{name}: the sizes that are not whole groups of {:?} ran the emulated module",
                    interface.multiples
                );
            }
            println!(
                "{name}: SIMD-group products checked in threadgroups of 64, 32, 80 and 1; pinned \
                 at {:?}, whole groups of {:?}, the others {}",
                interface.required_subgroup_size,
                interface.multiples,
                match emulated {
                    Some(Lowered::Made(..)) => "emulated",
                    _ => "on the same module",
                }
            );
        });
    }

    /// `simd_shape` of `pad_simd_shape`: each thread's simdgroup index, lane, the threads of its
    /// simdgroup (a simdgroup sum of 1) and its index in the threadgroup, in threadgroups of 80,
    /// 1, 32, 64 and 40 threads. Metal's answer, measured on an M3 Max (macOS 27.0.1) at those
    /// sizes: thread `i` is lane `i % 32` of simdgroup `i / 32`, whose threads are
    /// `min(32, n - 32 (i / 32))`, so 80 falls 32, 32 and 16. Every device answers Metal's, on
    /// whichever module runs the size: the whole one, or the emulated one where the size is not
    /// whole groups of what the whole one runs.
    #[test]
    fn every_device_lays_out_simdgroups_as_metal_does() {
        use crate::pipeline::Lowered;

        let Some(descriptor) = descriptor_of("pad_simd_shape", "simd_shape") else {
            return;
        };
        const GROUPS: u32 = 2;
        each_device("compute_simd_shape", |name, _, device| {
            let state = state(name, &device, &descriptor);
            let queue = device.new_command_queue(4).expect("a queue");
            let mut ran = Vec::new();
            for size in [80, 1, 32, 64, 40] {
                let threads = GROUPS * size;
                let mut buffers = Buffers::new(&device, 1 << 16);
                let out = buffers.buffer(u64::from(threads) * 16);
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                encoder.set_buffer(Some(&out), 0, 0).expect("a buffer");
                encoder
                    .dispatch_threadgroups([GROUPS, 1, 1], [size, 1, 1])
                    .unwrap_or_else(|error| panic!("{name}: threadgroups of {size}: {error}"));
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
                let want: Vec<u32> = (0..threads)
                    .flat_map(|t| {
                        let i = t % size;
                        [i / 32, i % 32, 32.min(size - 32 * (i / 32)), i]
                    })
                    .collect();
                assert_eq!(
                    words(&out, 0, want.len()),
                    want,
                    "{name}: the simdgroups of a threadgroup of {size}"
                );
                let emulated = state
                    .inner
                    .compiled
                    .emulated
                    .as_ref()
                    .and_then(|emulated| emulated.lowered.get())
                    .is_some_and(|lowered| matches!(lowered, Lowered::Made(..)))
                    && state
                        .inner
                        .compiled
                        .whole
                        .interface
                        .multiples
                        .iter()
                        .any(|&multiple| size % multiple != 0);
                ran.push(format!(
                    "{size} {}",
                    if emulated { "emulated" } else { "whole" }
                ));
            }
            println!("{name}: Metal's simdgroups at {}", ran.join(", "));
        });
    }

    /// `unit_axes_walk`: a loop of `(y & 3) + 1` iterations at row `y`, each a SIMD-group sum over
    /// the threads still in it, in one threadgroup of 8 x 8 threads and one of 8 x 10 (a partial
    /// SIMD group of 16 last). Where its SIMD groups are emulated its walk is predicated, and the
    /// translator states the unit axes that would make it less divergent (`RuntimeFact::UnitAxes`
    /// with no axes): the module is the general one, and every device creates the state and runs
    /// both sizes, on the module whose walk states it where its tier emulates SIMD groups, and on
    /// the emulated module at 8 x 10 where it runs only whole ones. Every answer is Metal's (M3 Max,
    /// macOS 27.0.1): thread `t` adds, at each iteration `i` of its own, `i + 1` times the threads
    /// of its SIMD group whose own trip count exceeds `i`.
    #[test]
    fn every_device_runs_a_kernel_whose_walk_unit_axes_could_relieve() {
        let Some(descriptor) = descriptor("unit_axes_walk") else {
            return;
        };
        each_device("compute_unit_axes", |name, _, device| {
            let state = state(name, &device, &descriptor);
            let queue = device.new_command_queue(4).expect("a queue");
            for rows in [8, 10] {
                let size = 8 * rows;
                let mut buffers = Buffers::new(&device, 1 << 16);
                let out = buffers.buffer(u64::from(size) * 4);
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                encoder.set_buffer(Some(&out), 0, 0).expect("a buffer");
                encoder
                    .dispatch_threadgroups([1, 1, 1], [8, rows, 1])
                    .unwrap_or_else(|error| panic!("{name}: a threadgroup of 8 x {rows}: {error}"));
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
                let trip = |u: u32| ((u / 8) & 3) + 1;
                let want: Vec<u32> = (0..size)
                    .map(|t| {
                        let base = t - t % 32;
                        let simdgroup = base..(base + 32).min(size);
                        (0..trip(t))
                            .map(|i| {
                                let still = simdgroup.clone().filter(|&u| trip(u) > i).count();
                                (i + 1) * still as u32
                            })
                            .sum()
                    })
                    .collect();
                assert_eq!(
                    words(&out, 0, want.len()),
                    want,
                    "{name}: the sums in a threadgroup of 8 x {rows}"
                );
            }
            // Which modules the runtime read the relief of: the plain one, and the emulated
            // variant.
            let relieved = |facts: &[RuntimeFact]| {
                facts.iter().any(|fact| match *fact {
                    RuntimeFact::UnitAxes { axes, relief } => axes == 0 && relief != 0,
                    _ => false,
                })
            };
            let [plain, emulated] = facts_read(name, &device, &descriptor);
            let (plain, emulated) = (relieved(&plain), relieved(&emulated));
            let made = emulated_made(&state);
            assert!(
                plain || (emulated && made),
                "{name}: a module that states unit axes with no axes ran (plain {plain}, emulated \
                 {emulated}, emulated made {made})"
            );
            println!(
                "{name}: unit axes with no axes stated by the {} module, which ran; sums checked \
                 in threadgroups of 8 x 8 and 8 x 10",
                if plain { "plain" } else { "emulated" }
            );
        });
    }

    /// `row_lockstep_tree`: a tree reduction over threadgroup memory indexed by the thread's
    /// position along x, a workgroup barrier per step and none between a step's load and its
    /// store, in four threadgroups each of 16 x 2, 32 x 1 and 12 x 2 threads. In a threadgroup of
    /// two rows within one SIMD group the rows share x, and only the SIMD group's lockstep orders
    /// one row's load of a word before the other row's store to it: the translator keeps that
    /// order with barriers that order only pairs meeting across rows, and states it
    /// (`RuntimeFact::RowLockstep`). Every device creates the state and runs the three shapes;
    /// the plain module states the fact exactly where its local size is flattened (the Direct and
    /// Cluster32 tiers, which state `LocalSizeConstraint`), and where the emulated module states
    /// it and a shape ran on it, the reaper made that module. Every answer is Metal's (M3 Max,
    /// macOS 27.0.1): each thread writes the sum of the x below min(16, width), 120 or 66.
    #[test]
    fn every_device_runs_a_kernel_whose_lockstep_orders_rows() {
        let Some(descriptor) = descriptor("row_lockstep_tree") else {
            return;
        };
        const GROUPS: u32 = 4;
        each_device("compute_row_lockstep", |name, _, device| {
            let state = state(name, &device, &descriptor);
            let multiples = state.inner.compiled.whole.interface.multiples.clone();
            let queue = device.new_command_queue(4).expect("a queue");
            let mut on_emulated = Vec::new();
            for (width, rows) in [(16, 2), (32, 1), (12, 2)] {
                let threads = width * rows * GROUPS;
                let mut buffers = Buffers::new(&device, 1 << 16);
                let out = buffers.buffer(u64::from(threads) * 4);
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                encoder.set_buffer(Some(&out), 0, 0).expect("a buffer");
                encoder
                    .dispatch_threadgroups([GROUPS, 1, 1], [width, rows, 1])
                    .unwrap_or_else(|error| panic!("{name}: {width} x {rows}: {error}"));
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
                let want = vec![(0..width.min(16)).sum::<u32>(); threads as usize];
                assert_eq!(
                    words(&out, 0, want.len()),
                    want,
                    "{name}: threadgroups of {width} x {rows}"
                );
                if multiples
                    .iter()
                    .any(|&multiple| (width * rows) % multiple.max(1) != 0)
                {
                    on_emulated.push(format!("{width} x {rows}"));
                }
            }
            let [plain_facts, emulated] = facts_read(name, &device, &descriptor);
            let flat = (plain_facts.iter())
                .any(|fact| matches!(fact, RuntimeFact::LocalSizeConstraint { .. }));
            let (plain, emulated) = (
                plain_facts.contains(&RuntimeFact::RowLockstep),
                emulated.contains(&RuntimeFact::RowLockstep),
            );
            assert_eq!(
                plain, flat,
                "{name}: RowLockstep where the plain module's local size is flattened: \
                 {plain_facts:?}"
            );
            if emulated && !on_emulated.is_empty() {
                assert!(
                    emulated_made(&state),
                    "{name}: the emulated module, which states RowLockstep, ran {on_emulated:?}"
                );
            }
            println!(
                "{name}: sums checked in 16 x 2, 32 x 1 and 12 x 2; plain module flattened \
                 {flat} ({plain_facts:?}); RowLockstep stated by the plain module {plain}, by the \
                 emulated module {emulated}, which ran {on_emulated:?}"
            );
        });
    }

    /// A state made after the device was quiesced has its emulated-SIMD-group variant marked not
    /// lowered at once, as the reaper has ended: no dispatch waits for it.
    #[test]
    fn every_device_marks_the_emulated_variant_of_a_state_made_after_quiesce() {
        use crate::pipeline::Lowered;

        let Some(descriptor) = descriptor_of("gsm2_simd_products", "simd_float_products") else {
            return;
        };
        each_device("compute_simd_quiesced", |name, _, device| {
            assert!(device.quiesce(std::time::Duration::from_secs(10)), "{name}");
            let Ok(state) =
                device.new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers)
            else {
                println!("{name}: the kernel is refused; nothing to mark");
                return;
            };
            let Some(emulated) = &state.inner.compiled.emulated else {
                println!("{name}: no emulated variant");
                return;
            };
            assert!(
                matches!(emulated.lowered.get(), Some(Lowered::Quiesced)),
                "{name}: marked at once"
            );
            println!("{name}: the emulated variant is marked not lowered");
        });
    }

    /// Every file the store keeps under `dir`, its count (`state`) aside: its path under it, its
    /// length and its time on disk, in order.
    fn files(dir: &Path) -> Vec<(String, u64, SystemTime)> {
        fn under(root: &Path, dir: &Path, all: &mut Vec<(String, u64, SystemTime)>) {
            for entry in std::fs::read_dir(dir).expect("the directory").flatten() {
                let meta = entry.metadata().expect("its metadata");
                if meta.is_dir() {
                    under(root, &entry.path(), all);
                } else {
                    let path = entry.path();
                    let file = path.strip_prefix(root).expect("under the root");
                    let file = file.to_string_lossy().replace('\\', "/");
                    all.push((file, meta.len(), meta.modified().expect("its time")));
                }
            }
        }
        let mut all = Vec::new();
        under(dir, dir, &mut all);
        all.retain(|(file, ..)| file != "state");
        all.sort();
        all
    }

    /// The pipeline cache blobs among `files`.
    fn blobs(files: &[(String, u64, SystemTime)]) -> Vec<&(String, u64, SystemTime)> {
        (files.iter())
            .filter(|(file, ..)| file.starts_with("pipelines-"))
            .collect()
    }

    /// A device on `dir`.
    fn on_directory(name: &str, physical: &crate::PhysicalDevice, dir: &Path) -> Device {
        let options = DeviceOptions {
            pipeline_cache_dir: Some(dir.to_owned()),
            ..DeviceOptions::default()
        };
        Device::new(physical, options).unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// A device on a cache directory a device before it used lowers nothing at creation: the
    /// tiling kernel's lowering and the observing kernel's two are read from the directory, and
    /// the tile's `VkPipeline` is made through a pipeline cache started from the device's blob.
    /// The first device's writes (`Metrics::cache_bytes_written`) are at least the files' bytes,
    /// the blob among them (written once creation was quiet, and again at the drop where the
    /// cache grew since); the second writes nothing and leaves every file as it was.
    #[test]
    fn every_device_takes_the_kernels_a_device_before_it_lowered_from_the_cache_directory() {
        let (Some(tiled), Some(observing)) = (
            descriptor("a1_bool_from_float"),
            descriptor("icb_replay_grid_template"),
        ) else {
            return;
        };
        let instance = Instance::new(&[]).expect("an instance at the floor");
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let dir = std::env::temp_dir().join(format!("m2v-rt-cache-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let mut first = Vec::new();
            for round in 0..2 {
                let device = on_directory(name, physical, &dir);
                let counters = device.counters();
                let states = [
                    state(name, &device, &tiled),
                    state(name, &device, &observing),
                ];
                let metrics = counters.metrics();
                let counted = (metrics.translations, metrics.translations_loaded);
                // One lowering for the tiling kernel, two for the observing one.
                let want = [(3, 0), (0, 3)][round];
                assert_eq!(counted, want, "{name}: device {round} on the directory");
                drop(states);
                drop(device);
                // The teardown ends once the reaper wrote what it was given.
                within("the teardown", || counters.gauges() == Gauges::default());
                let written = counters.metrics().cache_bytes_written;
                let files = files(&dir);
                if round == 0 {
                    let bytes: u64 = files.iter().map(|&(_, len, _)| len).sum();
                    assert!(
                        written >= bytes,
                        "{name}: {written} written, {bytes} in files"
                    );
                    assert_eq!(blobs(&files).len(), 1, "{name}: the device's blob");
                    first = files;
                } else {
                    assert_eq!(written, 0, "{name}: the second device writes nothing");
                    assert_eq!(files, first, "{name}: every file as it was");
                }
            }
            let _ = std::fs::remove_dir_all(&dir);
            println!("{name}: 3 lowerings and the blob written, nothing by a second device");
        }
    }

    /// The second process's part of
    /// `every_device_shares_the_cache_directory_with_a_second_process`: on the device and
    /// directory it names, the first process's kernel, then a kernel of its own.
    const SECOND: &str = "tests::translate::on_device::second_process_on_a_cache_directory";

    /// A second process on a cache directory: its device takes the kernel the first process's
    /// device left there and writes none of its lowerings again; the kernel of its own adds its
    /// lowering, and its `VkPipeline`'s data grows the blob, which the second process replaces.
    #[test]
    fn every_device_shares_the_cache_directory_with_a_second_process() {
        let Some(tiled) = descriptor("a1_bool_from_float") else {
            return;
        };
        let instance = Instance::new(&[]).expect("an instance at the floor");
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let dir = std::env::temp_dir().join(format!("m2v-rt-cache2-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let device = on_directory(name, physical, &dir);
            let counters = device.counters();
            drop(state(name, &device, &tiled));
            drop(device);
            within("the teardown", || counters.gauges() == Gauges::default());
            let before = files(&dir);
            let test = std::env::current_exe().expect("the test binary");
            let status = std::process::Command::new(test)
                .args(["--exact", SECOND, "--ignored", "--nocapture"])
                .env("M2V_TEST_CACHE_DIR", &dir)
                .env("M2V_TEST_DEVICE", name)
                .status()
                .expect("the second process");
            assert!(status.success(), "{name}: the second process");
            let after = files(&dir);
            let _ = std::fs::remove_dir_all(&dir);
            let (blob, blob_after) = (blobs(&before), blobs(&after));
            assert_eq!((blob.len(), blob_after.len()), (1, 1), "{name}: one blob");
            assert!(blob_after[0].1 > blob[0].1, "{name}: the blob grew");
            for file in before.iter().filter(|file| !blob.contains(file)) {
                assert!(
                    after.contains(file),
                    "{name}: {} written again or gone",
                    file.0
                );
            }
            let added: Vec<&String> = (after.iter())
                .filter(|file| !before.contains(file) && !blob_after.contains(file))
                .map(|(file, ..)| file)
                .collect();
            assert!(!added.is_empty(), "{name}: its own kernel's lowering");
            assert!(
                added.iter().all(|file| file.contains("translations/")),
                "{name}: added {added:?}"
            );
            println!("{name}: the second process added {added:?}, grew the blob, rewrote nothing");
        }
    }

    /// Run only by `every_device_shares_the_cache_directory_with_a_second_process`, in a process
    /// of its own: on the device `M2V_TEST_DEVICE` names, a device on the directory
    /// `M2V_TEST_CACHE_DIR` names makes the first process's kernel, lowering nothing and writing
    /// nothing, then a kernel of its own.
    #[test]
    #[ignore = "the second process of every_device_shares_the_cache_directory_with_a_second_process"]
    fn second_process_on_a_cache_directory() {
        let dir = PathBuf::from(std::env::var_os("M2V_TEST_CACHE_DIR").expect("the directory"));
        let wanted = std::env::var("M2V_TEST_DEVICE").expect("the device");
        let (Some(tiled), Some(own)) = (
            descriptor("a1_bool_from_float"),
            descriptor("a1_sign_and_division"),
        ) else {
            panic!("the kernels");
        };
        let instance = Instance::new(&[]).expect("an instance at the floor");
        let devices = devices_of_the_cfgs_kind(&instance);
        let (name, physical) = (devices.iter())
            .find(|(name, _)| *name == wanted)
            .expect("the first process's device");
        let device = on_directory(name, physical, &dir);
        let counters = device.counters();
        let first = state(name, &device, &tiled);
        let metrics = counters.metrics();
        assert_eq!(
            metrics.translations, 0,
            "{name}: the first process's kernel"
        );
        let own = state(name, &device, &own);
        drop((first, own));
        drop(device);
        within("the teardown", || counters.gauges() == Gauges::default());
        assert_eq!(
            counters.metrics().pipelines_created,
            2,
            "{name}: two pipelines made"
        );
    }

    /// The length of the blob of `device`'s pipeline cache, 0 where there is none.
    fn blob_len(device: &Device) -> u64 {
        let blob = device.core().caches.blob().expect("a cache directory");
        std::fs::metadata(blob).map_or(0, |meta| meta.len())
    }

    /// Makes the state of `descriptor` on `device` and waits for the reaper's look at the pipeline
    /// cache once creation is quiet after it.
    fn state_and_look(
        name: &str,
        device: &Device,
        descriptor: &ComputePipelineDescriptor,
    ) -> ComputePipelineState {
        let caches = &device.core().caches;
        let looks = caches.looks_and_writes().0;
        let made = state(name, device, descriptor);
        within("the reaper's look", || caches.looks_and_writes().0 > looks);
        made
    }

    /// The pipeline cache filled past its bound: at the look after a creation takes it past, the
    /// reaper replaces it with a fresh one, which the gauge falls to, keeps the old one while it is
    /// held, and destroys it at the first look after it is let go; creations go on through the
    /// fresh cache, and the next write replaces the blob with the fresh cache's data alone.
    #[test]
    fn every_device_replaces_its_pipeline_cache_past_its_bound() {
        let (Some(first), Some(second), Some(third)) = (
            descriptor("a1_bool_from_float"),
            descriptor("a1_sign_and_division"),
            descriptor("icb_replay_grid_template"),
        ) else {
            return;
        };
        let instance = Instance::new(&[]).expect("an instance at the floor");
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let dir = std::env::temp_dir().join(format!("m2v-rt-bound-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let device = on_directory(name, physical, &dir);
            let caches = &device.core().caches;
            let counters = device.counters();
            let a = state_and_look(name, &device, &first);
            let full = counters.gauges().pipeline_cache_bytes;
            assert!(full > 0, "{name}: the cache's size read");
            let first_blob = blob_len(&device);
            assert_eq!(
                first_blob,
                24 + full,
                "{name}: the blob written at the look"
            );
            // Held as a creation holds it, across the next look.
            let held = caches.hold();
            caches.set_bound(full - 1);
            let b = state_and_look(name, &device, &second);
            let fresh = counters.gauges().pipeline_cache_bytes;
            assert!(
                fresh < full,
                "{name}: the gauge falls from {full} to {fresh}"
            );
            assert_eq!(caches.retired(), 1, "{name}: the old cache kept while held");
            assert_eq!(
                blob_len(&device),
                first_blob,
                "{name}: no write at the replacement"
            );
            drop(held);
            caches.set_bound(crate::cache::PIPELINE_CACHE_BOUND);
            let c = state_and_look(name, &device, &third);
            assert_eq!(
                caches.retired(),
                0,
                "{name}: the old cache destroyed once let go"
            );
            let now = counters.gauges().pipeline_cache_bytes;
            assert!(now > fresh, "{name}: the fresh cache grew to {now}");
            assert_eq!(
                blob_len(&device),
                24 + now,
                "{name}: the blob holds the fresh cache"
            );
            drop((a, b, c));
            drop(device);
            within("the teardown", || counters.gauges() == Gauges::default());
            let _ = std::fs::remove_dir_all(&dir);
            println!(
                "{name}: {full} bytes past a bound of {}, replaced by {fresh}, grown to {now}",
                full - 1
            );
        }
    }

    /// A blob the device cannot take, cut short or of another device's header, is removed when
    /// the reaper makes the cache, which starts empty: creations go on, and the drop writes the
    /// blob a next device takes. That device's look finds the cache no larger, so neither it nor
    /// the drop writes.
    #[test]
    fn every_device_removes_a_blob_it_cannot_take() {
        let Some(tiled) = descriptor("a1_bool_from_float") else {
            return;
        };
        let instance = Instance::new(&[]).expect("an instance at the floor");
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let other = {
                let mut header = vec![0; 32];
                header[..4].copy_from_slice(&32u32.to_le_bytes());
                header[4..8].copy_from_slice(&1u32.to_le_bytes());
                header.extend([7; 64]);
                [&crate::cache::frame(&header)[..], &header].concat()
            };
            for (what, bytes) in [
                ("cut short", other[..other.len() - 1].to_vec()),
                ("another device's", other),
            ] {
                let dir =
                    std::env::temp_dir().join(format!("m2v-rt-refuse-{}", std::process::id()));
                let _ = std::fs::remove_dir_all(&dir);
                let device = on_directory(name, physical, &dir);
                within("the pipeline cache", || device.core().caches.is_made());
                let blob = device.core().caches.blob().expect("a blob").to_owned();
                drop(device);
                assert!(!blob.exists(), "{name}: no creation, no blob");
                std::fs::write(&blob, &bytes).expect("a blob");
                let device = on_directory(name, physical, &dir);
                within("the pipeline cache", || device.core().caches.is_made());
                assert!(!blob.exists(), "{name}: a blob {what} removed");
                drop(state(name, &device, &tiled));
                drop(device);
                let written = std::fs::read(&blob).expect("the blob the drop wrote");
                assert_ne!(written, bytes, "{name}: replaced");
                let device = on_directory(name, physical, &dir);
                let counters = device.counters();
                let made = state_and_look(name, &device, &tiled);
                let caches = &device.core().caches;
                let (looks, writes) = caches.looks_and_writes();
                assert!(
                    looks >= 1 && writes == 0,
                    "{name}: {looks} looks, {writes} writes"
                );
                drop(made);
                drop(device);
                within("the teardown", || counters.gauges() == Gauges::default());
                let blob_written = counters.metrics().cache_bytes_written;
                assert_eq!(blob_written, 0, "{name}: the next device writes nothing");
                assert_eq!(std::fs::read(&blob).expect("the blob"), written, "{name}");
                let _ = std::fs::remove_dir_all(&dir);
                println!("{name}: a blob {what} removed, rewritten, then taken");
            }
        }
    }

    /// `vector_word_phases`'s kernels (`metal2vulkan_devtools::vector_word_phases`): with
    /// `vwp_words`'s two `uint` buffers bound at each word phase, 0, 4, 8 and 12, so its `uint4`
    /// and `float4` accesses through cast pointers land that far past a multiple of 16 whatever
    /// the AIR's `align 16`, and `vwp_vec16`'s two `float4` buffers at multiples of 16, all four
    /// buffers hold Metal's words afterwards, byte for byte from their first byte: the
    /// conformance crate's `vector-word-phases-*` cases hold the answer an Apple M3 Max recorded
    /// equal to the same words.
    #[test]
    fn every_device_places_each_vector_access_at_its_own_address() {
        use metal2vulkan_devtools::vector_word_phases::{
            answer, inputs, vec16_offsets, FIXTURE, OFFSETS, THREADS, VEC16, WORDS,
        };
        let (Some(phased), Some(aligned)) =
            (descriptor_of(FIXTURE, WORDS), descriptor_of(FIXTURE, VEC16))
        else {
            return;
        };
        each_device("compute_vector_words", |name, _, device| {
            let before = inputs();
            let (phased, aligned) = (
                state(name, &device, &phased),
                state(name, &device, &aligned),
            );
            let mut buffers = Buffers::new(&device, 64 * 1024);
            let held: Vec<(Buffer, Vec<u8>)> = [before.src, before.dst, before.a, before.b]
                .into_iter()
                .map(|bytes| (buffers.buffer(bytes.len() as u64), bytes))
                .collect();
            let queue = device.new_command_queue(4).expect("a queue");
            for offset in OFFSETS {
                for (buffer, bytes) in &held {
                    write(buffer, 0, bytes);
                }
                let (a_at, b_at) = vec16_offsets(offset);
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                for (state, binds) in [
                    (&phased, [(0, offset), (1, offset)]),
                    (&aligned, [(2, a_at), (3, b_at)]),
                ] {
                    encoder.set_compute_pipeline_state(state).expect("a state");
                    for (index, (buffer, at)) in binds.into_iter().enumerate() {
                        encoder
                            .set_buffer(Some(&held[buffer].0), at, index as u32)
                            .expect("a buffer");
                    }
                    encoder
                        .dispatch_threadgroups([1, 1, 1], [THREADS, 1, 1])
                        .expect("a dispatch");
                }
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
                let want = answer(offset);
                for ((buffer, bytes), (which, want)) in held.iter().zip([
                    ("src", &want.src),
                    ("dst", &want.dst),
                    ("a", &want.a),
                    ("b", &want.b),
                ]) {
                    let got = words(buffer, 0, bytes.len() / 4);
                    let want: Vec<u32> = want
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|w| u32::from_le_bytes(*w))
                        .collect();
                    let first = got.iter().zip(&want).position(|(x, y)| x != y);
                    assert_eq!(
                        first, None,
                        "{name}: offset {offset}: {which} differs from Metal's words"
                    );
                }
            }
            println!("{name}: Metal's words at bind offsets {OFFSETS:?}");
        });
    }

    /// `vwp_vec16` takes its `float4 *` arguments' addresses to be multiples of 16 and `vwp_words`
    /// its `uint *` arguments' multiples of 4 (`RuntimeFact::AddressAlignment`). A dispatch with
    /// either argument bound off that alignment (`vwp_words`'s 1 and 3 bytes past each word phase,
    /// `vwp_vec16`'s 4 and 12 bytes past its multiples of 16) is refused, as Metal's API
    /// validation refuses the bind (`Refusal::BufferOffsetAlignment`, naming the index, the offset
    /// and the alignment), and records nothing: the command buffer completes without error and
    /// every buffer holds its bytes as they were.
    #[test]
    fn an_argument_bound_off_its_alignment_is_refused() {
        use metal2vulkan_devtools::vector_word_phases::{
            inputs, vec16_offsets, FIXTURE, OFFSETS, THREADS, VEC16, WORDS,
        };
        let (Some(phased), Some(aligned)) =
            (descriptor_of(FIXTURE, WORDS), descriptor_of(FIXTURE, VEC16))
        else {
            return;
        };
        each_device("compute_misaligned_binds", |name, _, device| {
            let before = inputs();
            let (phased, aligned) = (
                state(name, &device, &phased),
                state(name, &device, &aligned),
            );
            let mut buffers = Buffers::new(&device, 64 * 1024);
            let held: Vec<(Buffer, Vec<u8>)> = [before.src, before.dst, before.a, before.b]
                .into_iter()
                .map(|bytes| (buffers.buffer(bytes.len() as u64), bytes))
                .collect();
            assert!(
                held.iter()
                    .all(|(buffer, _)| buffer.gpu_address() % 16 == 0),
                "{name}: the buffers' addresses"
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let mut refusals = 0;
            for offset in OFFSETS {
                for (buffer, bytes) in &held {
                    write(buffer, 0, bytes);
                }
                let (a_at, b_at) = vec16_offsets(offset);
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                // Each case binds one argument off its alignment and the other on it.
                for (state, binds, off, alignment) in [
                    (&phased, [(0, offset + 1), (1, offset)], 0, 4),
                    (&phased, [(0, offset), (1, offset + 3)], 1, 4),
                    (&aligned, [(2, a_at + 4), (3, b_at)], 0, 16),
                    (&aligned, [(2, a_at), (3, b_at + 12)], 1, 16),
                ] {
                    encoder.set_compute_pipeline_state(state).expect("a state");
                    for (index, (buffer, at)) in binds.into_iter().enumerate() {
                        encoder
                            .set_buffer(Some(&held[buffer].0), at, index as u32)
                            .expect("a buffer");
                    }
                    let at = binds[off as usize].1;
                    assert!(
                        matches!(
                            encoder.dispatch_threadgroups([1, 1, 1], [THREADS, 1, 1]),
                            Err(Error::Refused(Refusal::BufferOffsetAlignment {
                                index,
                                offset: refused,
                                alignment: refused_alignment,
                            })) if index == off && refused == at && refused_alignment == alignment
                        ),
                        "{name}: offset {offset}: index {off} bound at {at}"
                    );
                    refusals += 1;
                }
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}: offset {offset}");
                for ((buffer, bytes), which) in held.iter().zip(["src", "dst", "a", "b"]) {
                    let got = words(buffer, 0, bytes.len() / 4);
                    let want: Vec<u32> = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|w| u32::from_le_bytes(*w))
                        .collect();
                    assert_eq!(got, want, "{name}: offset {offset}: {which} changed");
                }
            }
            assert_eq!(refusals, 4 * OFFSETS.len(), "{name}: the binds");
            println!("{name}: {refusals} dispatches with an argument off its alignment refused");
        });
    }

    /// `ob1_storage_buffer` loads `src[idx[t]]` into `out[t]` and stores `0xA0000000 | t` to
    /// `out[idx[8 + t]]`, with `src` and `out` one Shared buffer of [`MAX_BUFFER_LENGTH`] (2 GiB)
    /// bound at its start and the indices near it and up to its last word. Each access lands at
    /// its own word. A device whose largest block or memory is smaller skips visibly. It runs alone
    /// (`.config/nextest.toml`): its 2 GiB is host memory on a device whose Shared memory type is
    /// the host's.
    #[test]
    fn an_access_up_to_2_gib_lands_at_its_own_word() {
        let Some(descriptor) = descriptor("ob1_storage_buffer") else {
            return;
        };
        const LEN: u64 = MAX_BUFFER_LENGTH;
        // The buffer's last word.
        let last = (LEN / 4) as u32 - 1;
        let loads = [
            64,
            last - 4096,
            last - 1000,
            last - 4,
            last - 3,
            last - 2,
            last - 1,
            last,
        ];
        let stores = [
            128,
            last - 4097,
            last - 2000,
            last - 11,
            last - 10,
            last - 9,
            last - 8,
            last - 5,
        ];
        each_device("compute_to_2_gib", |name, _, device| {
            let block = match device.take_block(MemoryClass::Shared, LEN) {
                Ok(block) => block,
                Err(
                    error @ (Error::Limit(Limit::BlockLen { .. })
                    | Error::Vulkan {
                        result:
                            ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                            | ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY,
                        ..
                    }),
                ) => {
                    eprintln!("SKIPPED: {name}: no Shared block of {LEN} bytes: {error}");
                    return;
                }
                Err(error) => panic!("{name}: {error}"),
            };
            let kernel = state(name, &device, &descriptor);
            // A taken block reads zero, so the buffer needs no second zeroing.
            let big = device
                .new_buffer(block.id, 0, LEN, None)
                .expect("the large buffer");
            let mut buffers = Buffers::new(&device, 64 * 1024);
            let idx = buffers.buffer(64);
            write(&idx, 0, &le_bytes(&[loads, stores].concat()));
            for (t, &word) in loads.iter().enumerate() {
                write(
                    &big,
                    4 * u64::from(word),
                    &(0x5000_0000 | t as u32).to_le_bytes(),
                );
            }
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder
                .set_compute_pipeline_state(&kernel)
                .expect("a state");
            encoder.set_buffer(Some(&idx), 0, 0).expect("a buffer");
            encoder.set_buffer(Some(&big), 0, 1).expect("a buffer");
            encoder.set_buffer(Some(&big), 0, 2).expect("a buffer");
            encoder
                .dispatch_threadgroups([1, 1, 1], [8, 1, 1])
                .expect("a dispatch");
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let loaded: Vec<u32> = (0..8).map(|t| 0x5000_0000 | t).collect();
            assert_eq!(words(&big, 0, 8), loaded, "{name}: the loads at {loads:x?}");
            for (t, &word) in stores.iter().enumerate() {
                assert_eq!(
                    words(&big, 4 * u64::from(word), 1),
                    [0xA000_0000 | t as u32],
                    "{name}: the store to word {word:#x}"
                );
            }
            println!("{name}: loads and stores up to 2 GiB landed at their own words");
        });
    }
}
