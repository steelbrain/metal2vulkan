//! Translated kernels, under a device cfg on every device of the cfg's kind. A kernel that does not
//! observe its threadgroup partition runs every direct dispatch in tiles, the 256-wide one made
//! with its state, and an indirect dispatch at its own threadgroup size; a precise one answers the
//! same words in a tile as at the app's shape. A kernel that does observe it runs whole grids at
//! their own size, a `dispatchThreads` grid that is not a whole number of threadgroups on its
//! partial-threadgroups variant, and a threadgroup count past the device's limit split. Each
//! answer is checked against the program's, and each `VkPipeline` made after creation is a
//! first-use wait the counters record.
//!
//! `llvm-as` assembles the kernels; without it a test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        host_time, BlockId, Buffer, ComputePipelineDescriptor, ComputePipelineState, Device,
        DispatchType, FunctionDescriptor, Library, MemoryClass, PrecisionNote, StageSource,
    };
    use metal2vulkan::Budget;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

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
        let bitcode = match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return None;
            }
            Err(error) => panic!("{name}: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function = library.function(name).expect("the kernel");
        Some(ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        )))
    }

    /// The state of `descriptor` on `device`, printing what the translator gave where it is
    /// refused.
    fn state(
        name: &str,
        device: &Device,
        descriptor: &ComputePipelineDescriptor,
    ) -> ComputePipelineState {
        device
            .new_compute_pipeline_state(descriptor)
            .unwrap_or_else(|error| {
                let translated = device
                    .core()
                    .translator
                    .compile_compute_pipeline(descriptor, &Budget::standard());
                if let Ok(pipeline) = translated {
                    eprintln!(
                        "{name}: interface {:?}\nconstants {:?}\ndependencies {:?}\nsets {:?}\nflags {:?} subgroup {:?} blob {}",
                        pipeline.interface,
                        pipeline.stage.specialization.constants,
                        pipeline.dependencies,
                        pipeline.layout.sets,
                        pipeline.stage.stage_flags,
                        pipeline.stage.required_subgroup_size,
                        pipeline.stage.constant_blob.is_some(),
                    );
                }
                panic!("{name}: {error}")
            })
    }

    /// Zeroed Shared buffers in one block, each at a 256-byte boundary.
    struct Buffers {
        device: Device,
        block: BlockId,
        next: u64,
    }

    impl Buffers {
        fn new(device: &Device, len: u64) -> Buffers {
            let block = device
                .take_block(MemoryClass::Shared, len)
                .expect("a Shared block");
            Buffers {
                device: device.clone(),
                block: block.id,
                next: 0,
            }
        }

        fn buffer(&mut self, len: u64) -> Buffer {
            let buffer = self
                .device
                .new_buffer(self.block, self.next, len, true)
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
    /// state made the 256-wide tile, which a wide `dispatchThreads` grid runs without a wait; a
    /// narrow threadgroup dispatch runs a narrower tile and an indirect dispatch the app's own
    /// size, each made by a first-use wait. Every answer is the program's, over buffers bound at
    /// offsets.
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
                "{name}: the 256-wide tile, made with the state"
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
            assert_eq!(waits, 2, "{name}: first-use waits");
            assert!(nanos > 0, "{name}: first-use wait time");
            assert!(
                after.first_use_wait_max_nanos >= nanos / 2,
                "{name}: longest wait"
            );
            assert_eq!(
                after.pipelines_created - created.pipelines_created,
                2,
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
    /// runs split, each piece reading Metal's whole grid. Each answer is Metal's, and each new
    /// size or module a first-use wait. Prints the creation's time and one lowering's.
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
            let runs: Vec<([u32; 3], [u32; 3], bool)> = [
                ([10, 3, 1], [4, 2, 1], true),
                ([8, 4, 1], [4, 2, 1], true),
                ([3, 2, 1], [4, 2, 1], false),
            ]
            .into_iter()
            .chain(past.map(|count| ([1, count, 1], [1, 1, 1], false)))
            .collect();
            let mut buffers = Buffers::new(&device, 8 << 20);
            let outs: Vec<(Buffer, Vec<u32>)> = runs
                .iter()
                .map(|&(grid, size, threads)| {
                    // A threadgroup dispatch's grid is its counts times its size.
                    let grid = match threads {
                        true => grid,
                        false => std::array::from_fn(|axis| grid[axis] * size[axis]),
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
            for (&(grid, size, threads), (out, want)) in runs.iter().zip(&outs) {
                let n = size.iter().product::<u32>();
                encoder
                    .set_threadgroup_memory_length((4 * n).next_multiple_of(16), 0)
                    .expect("a length");
                encoder
                    .set_bytes(&le_bytes(&[1000, want.len() as u32]), 0)
                    .expect("the parameters");
                encoder.set_buffer(Some(out), 0, 1).expect("a buffer");
                match threads {
                    true => encoder.dispatch_threads(grid, size),
                    false => encoder.dispatch_threadgroups(grid, size),
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
            // The ragged grid's variant, the whole grid's size, the one-thread size.
            let sizes = 2 + u64::from(past.is_some());
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
    /// indirect dispatch's size); its fast-math twin's state carries `FastMathRewrites`, the note
    /// of the rounding a device may choose per pipeline.
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
            let first = tiled.iter().zip(&shaped).position(|(a, b)| a != b);
            assert_eq!(
                first, None,
                "{name}: the first word the shapes answer apart"
            );
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
    /// over each thread's input. Where the device lets compute pin a subgroup of 32 (compute in
    /// `requiredSubgroupSizeStages`, 32 between its sizes), the translator's tier for it is
    /// Direct: a pinned subgroup of 32 per SIMD group, its local size flattened to whole SIMD
    /// groups. The state runs whole SIMD groups with Metal's answers and refuses a threadgroup of
    /// a partial one. A device that cannot pin 32 takes another tier, which may be refused.
    #[test]
    fn every_device_runs_simd_groups_at_metals_width() {
        use crate::{Error, Gap};
        use ash::vk;

        let bitcode = match llvm_assemble(&engine_dir().join("gsm2_simd_products.ll")) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("gsm2_simd_products: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function = library.function("simd_float_products").expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        const THREADS: u32 = 128;
        const GROUP: u32 = 64;
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
            let state = match device.new_compute_pipeline_state(&descriptor) {
                Ok(state) => state,
                Err(Error::Unsupported(Gap::KernelInput)) if !pinnable => {
                    println!(
                        "{name}: SIMD groups not pinnable at 32; the tier's kernel is refused"
                    );
                    return;
                }
                Err(error) => panic!("{name}: {error}"),
            };
            let interface = &state.inner.whole.interface;
            if pinnable {
                assert_eq!(
                    (interface.required_subgroup_size, &interface.multiples[..]),
                    (Some(32), &[32][..]),
                    "{name}: the Direct tier, a pinned subgroup of 32 per SIMD group"
                );
            }
            let input: Vec<u8> = (0..THREADS)
                .flat_map(|t| simd_input(t % 32).to_le_bytes())
                .collect();
            let block = device
                .take_block(crate::MemoryClass::Shared, 1)
                .expect("a block");
            let a = device
                .new_buffer(block.id, 0, u64::from(THREADS) * 4, true)
                .expect("a buffer");
            let out = device
                .new_buffer(block.id, 4096, u64::from(THREADS) * 16, true)
                .expect("a buffer");
            write(&a, 0, &input);
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            encoder.set_buffer(Some(&a), 0, 0).expect("a buffer");
            encoder.set_buffer(Some(&out), 0, 1).expect("a buffer");
            if pinnable {
                assert!(matches!(
                    encoder.dispatch_threads([THREADS, 1, 1], [16, 1, 1]),
                    Err(Error::Unsupported(Gap::PartialSimdgroups { width: 32, .. }))
                ));
            }
            encoder
                .dispatch_threads([THREADS, 1, 1], [GROUP, 1, 1])
                .expect("a dispatch of whole SIMD groups");
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let mut want = Vec::new();
            for t in 0..THREADS {
                let lanes = (t - t % 32..t - t % 32 + 32).map(|lane| simd_input(lane % 32));
                let product: f32 = lanes.clone().product();
                let inclusive: f32 = lanes.clone().take((t % 32 + 1) as usize).product();
                let exclusive: f32 = lanes.take((t % 32) as usize).product();
                want.extend([product, inclusive, exclusive, product].map(f32::to_bits));
            }
            assert_eq!(words(&out, 0, want.len()), want, "{name}: the products");
            println!(
                "{name}: SIMD-group products checked over {THREADS} threads; {}",
                match interface.required_subgroup_size {
                    Some(size) => format!("Direct, pinned at {size}"),
                    None => String::from("not pinned"),
                }
            );
        });
    }
}
