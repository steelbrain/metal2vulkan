//! Translated kernels, under a device cfg on every device of the cfg's kind. A kernel that does not
//! observe its threadgroup partition runs every direct dispatch in tiles of 256 threads on the one
//! `VkPipeline` its finish_pipeline makes, and an indirect dispatch at its own threadgroup size; a
//! precise one answers the same words in a tile as at the app's shape. A kernel that does observe it runs whole grids at
//! their own size, a `dispatchThreads` grid that is not a whole number of threadgroups on its
//! partial-threadgroups variant, and a threadgroup count past the device's limit split; one whose
//! only observations are lockstep barriers that run where each SIMD group lies within one row runs
//! grids in rows on its one-row tiles, while one whose lockstep barriers all order pairs across
//! rows, and one that reads another row's word after a threadgroup barrier, keep their own size
//! there. Each
//! answer is checked against the program's, and each `VkPipeline` made after creation is a
//! first-use wait the counters record. Vector accesses through pointers cast from a `uint`
//! buffer land at their own addresses, 4, 8 or 12 bytes past a multiple of 16, as Metal places them.
//! An access through a buffer argument at 2 GiB and past lands there, on every device that can
//! allocate the buffer. A kernel that reads the stage-in region stores the region each dispatch
//! was encoded under, and is refused where the encoder set none; a kernel's `[[stage_in]]`
//! attributes are fetched from the element each step function names in that region, in a ragged
//! grid's partial threadgroup too, whose invocations past the grid write nothing. A read of a texture the kernel's function constants removed takes zero,
//! and a read through a buffer argument bound to nothing reads the nil buffer's zeros.
//! An observer runs its whole grids, threadgroup dispatches and indirect ones on its plain module,
//! and its ragged grids on its partial-threadgroups variant, whose vote word may take its
//! threadgroup memory past the device's limit. Threadgroup memory, a declared array's and a threadgroup argument's, reads zero until
//! the program writes it, in a dispatch's later threadgroups, in a second dispatch and on the
//! emulated-SIMD-group module alike. Every module is lowered with the state, the variants
//! included, so no dispatch makes a translation; an observing kernel's state made at its
//! descriptor's `requiredThreadsPerThreadgroup` makes the pipelines its grids run at that size. A
//! dispatch that `prepare_dispatch` prepared with its own arguments finds its pipeline made by the
//! call, and what the runtime records and submits is the same with the calls and without.
//!
//! `llvm-as` assembles the kernels; without it a test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished, within};
    use crate::tests::{devices_of_the_cfgs_kind, keeps_a_blob};
    use crate::StageInKind;
    use crate::{
        host_time, Block, Buffer, ComputePipelineDescriptor, ComputePipelineState, Device,
        DeviceOptions, DispatchType, Error, FunctionDescriptor, Gap, Gauges, Library, Limit,
        MemoryClass, PartialThreadgroupsSite, PrecisionNote, Refusal, Region, StageSource,
        MAX_BUFFER_LENGTH,
    };
    use metal2vulkan::metal::{AttributeFormat, IndexType, StepFunction};
    use metal2vulkan::pipeline::{
        KernelVariant, RuntimeFact, StageEntries, StageInputAttributeDescriptor,
        StageInputDescriptor, StageInputLayoutDescriptor,
    };
    use metal2vulkan::Budget;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, llvm_assemble_text, AssembleError};
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

    /// The state of `descriptor` on `device`, its main pipeline then finished, as a consumer
    /// finishes it once it has answered; printing what the translator gave where it is refused:
    /// the plain module, and for a `Function` kernel its partial-threadgroups variant.
    fn state(
        name: &str,
        device: &Device,
        descriptor: &ComputePipelineDescriptor,
    ) -> ComputePipelineState {
        let state = unfinished(name, device, descriptor);
        (state.finish_pipeline()).unwrap_or_else(|error| panic!("{name}: finished: {error}"));
        state
    }

    /// The state of `descriptor` on `device`, its main pipeline not made yet, printing what the
    /// translator gave where it is refused, as [`state`] does.
    fn unfinished(
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
                    let inputs = (descriptor.lowering_inputs())
                        .kernel_variant(KernelVariant::PartialThreadgroups);
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

    /// Whether `state`'s make made its emulated-SIMD-group module.
    fn emulated_made(state: &ComputePipelineState) -> bool {
        use crate::pipeline::Lowered;

        (state.inner.compiled.emulated.as_ref())
            .is_some_and(|lowered| matches!(lowered, Lowered::Made(..)))
    }

    /// How many modules `state`'s make made: its whole one, and each variant it lowered beside
    /// it.
    fn modules_made(state: &ComputePipelineState) -> usize {
        use crate::pipeline::{Form, Lowered, Partial};

        let partial = match &state.inner.compiled.form {
            Form::Sized {
                partial: Partial::Beside(lowered),
            } => matches!(**lowered, Lowered::Made(..)),
            _ => false,
        };
        1 + usize::from(partial) + usize::from(emulated_made(state))
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
                "{name}: the tiles' pipeline, made by finish_pipeline"
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
    /// a barrier, and every builtin of the partition), translated at creation, plainly and as its
    /// partial-threadgroups variant: a whole `dispatchThreads` grid, threadgroup counts and an
    /// indirect dispatch run its plain module at the call's size, none of them counted in
    /// `Metrics::partial_variant_whole_grids`, and a ragged grid the variant, each counted in
    /// `Metrics::partial_variant_ragged_grids`, with no translation at any dispatch, and a
    /// threadgroup count past the device's limit on an axis
    /// runs split, each piece reading Metal's whole grid. Threadgroups deeper than 64, 1 by 1 by
    /// 320 and 2 by 3 by 100, run as one row of their threads on every device, whole, ragged and
    /// indirect. Each answer is Metal's, and each new size or module's pipeline a first-use wait.
    /// Prints the creation's time and one lowering's.
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
            assert_eq!(
                created.translations - made.translations,
                2,
                "{name}: the plain module and the partial-threadgroups variant at creation"
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
            assert_eq!(
                after.translations, created.translations,
                "{name}: no translation at a dispatch, the first ragged grid's included"
            );
            // One pipeline for each module and size the runs take: a ragged grid the variant; a
            // threadgroup of whole SIMD groups of the device's tier the plain module, direct or
            // indirect; any other the emulated variant.
            let simd = &state.inner.compiled.whole.interface.multiples;
            let whole_simd = |size: [u32; 3]| {
                let threads = size.iter().product::<u32>();
                simd.iter()
                    .all(|&width| threads.is_multiple_of(width.max(1)))
            };
            let mut pipelines = std::collections::HashSet::new();
            let mut ragged_grids = 0;
            for &(grid, size, how) in &runs {
                let ragged = how == How::Threads
                    && (0..3).any(|axis| !grid[axis].is_multiple_of(size[axis]));
                let module = match (ragged, whole_simd(size)) {
                    (true, _) => {
                        ragged_grids += 1;
                        "variant"
                    }
                    (false, false) => "emulated",
                    (false, true) => "plain",
                };
                pipelines.insert((module, size));
            }
            assert_eq!(
                after.first_use_waits - created.first_use_waits,
                pipelines.len() as u64,
                "{name}: first-use waits, one for each of {pipelines:?}"
            );
            assert_eq!(
                after.partial_variant_whole_grids - created.partial_variant_whole_grids,
                0,
                "{name}: whole threadgroups, direct and indirect, on the partial-threadgroups variant"
            );
            assert_eq!(
                after.partial_variant_ragged_grids - created.partial_variant_ragged_grids,
                ragged_grids,
                "{name}: ragged grids on the partial-threadgroups variant"
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

    /// How a prepared case dispatches.
    #[derive(Clone, Copy, Debug)]
    enum How {
        Threads,
        Groups,
        Indirect,
    }

    /// Records `how` over `grid` (threads, or threadgroup counts) at `size` of
    /// `icb_replay_grid_template` on `device`, its threadgroup memory `length` at index 0, waits
    /// for it, and checks its answer against Metal's. The dispatch's own answer.
    fn grid_template_run(
        name: &str,
        device: &Device,
        state: &ComputePipelineState,
        (grid, size, how): ([u32; 3], [u32; 3], How),
        length: u32,
    ) -> Result<(), Error> {
        let threads = match how {
            How::Threads => grid,
            How::Groups | How::Indirect => std::array::from_fn(|axis| grid[axis] * size[axis]),
        };
        let slots = threads.iter().product::<u32>().max(1);
        let want = grid_template(threads, size, [1000, slots]);
        let mut buffers = Buffers::new(device, 1 << 16);
        let out = buffers.buffer(4 * u64::from(slots));
        let arguments = buffers.buffer(16);
        write(&arguments, 0, &le_bytes(&grid));
        let queue = device.new_command_queue(4).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = command_buffer
            .compute_command_encoder(DispatchType::Serial)
            .expect("an encoder");
        encoder.set_compute_pipeline_state(state).expect("a state");
        encoder
            .set_threadgroup_memory_length(length, 0)
            .expect("a length");
        encoder
            .set_bytes(&le_bytes(&[1000, slots]), 0)
            .expect("the parameters");
        encoder.set_buffer(Some(&out), 0, 1).expect("a buffer");
        let dispatched = match how {
            How::Threads => encoder.dispatch_threads(grid, size),
            How::Groups => encoder.dispatch_threadgroups(grid, size),
            How::Indirect => {
                encoder.dispatch_threadgroups_with_indirect_buffer(&arguments, 0, size)
            }
        };
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
        if dispatched.is_ok() && slots > 1 {
            assert_eq!(
                words(&out, 0, want.len()),
                want,
                "{name}: {how:?} {grid:?} by {size:?}"
            );
        }
        dispatched
    }

    /// `ComputePipelineState::prepare_dispatch` makes what a dispatch of `icb_replay_grid_template`,
    /// an observing kernel, then finds made: for a whole `dispatchThreads` grid, a ragged one (its
    /// partial-threadgroups variant, lowered with the state), a `dispatchThreadgroups` grid and an
    /// indirect dispatch, each prepared with the dispatch's own arguments, the call makes exactly
    /// one pipeline and lowers nothing, and the dispatch adds no first-use wait and makes nothing;
    /// an indirect dispatch of the whole grid's size finds the whole grid's made, since both run
    /// the plain module. A `dispatchThreadgroups` grid whose threads saturate on an axis, and a
    /// `dispatchThreads` grid of exactly `u32::MAX` threads on a size that does not divide it,
    /// prepared alone, make what a small grid of the same call and size then runs with no wait:
    /// the plain module and the partial-threadgroups variant. A second call with the same
    /// arguments makes nothing. A call for what the dispatch records nothing for (a grid with no
    /// threads) or refuses (a threadgroup past Metal's, threadgroup memory past the device's)
    /// makes nothing, and the dispatch answers as it does unprepared. Each answer is Metal's.
    #[test]
    fn every_device_finds_made_what_prepare_dispatch_made() {
        use crate::Dispatch;

        let Some(descriptor) = descriptor("icb_replay_grid_template") else {
            return;
        };
        each_device("compute_prepare", |name, _, device| {
            let raw = device.raw();
            // SAFETY: the physical device is the instance's.
            let shared = unsafe {
                raw.instance
                    .get_physical_device_properties(raw.physical_device)
                    .limits
                    .max_compute_shared_memory_size
            };
            let state = state(name, &device, &descriptor);
            let counters = device.counters();
            let length = |size: [u32; 3]| (4 * size.iter().product::<u32>()).next_multiple_of(16);
            // What is prepared, at what size; the dispatch that then runs; the pipelines the call
            // makes.
            let cases = [
                (
                    Dispatch::Threads([8, 4, 1]),
                    [4, 2, 1],
                    ([8, 4, 1], How::Threads),
                    1,
                ),
                (
                    Dispatch::Threads([10, 3, 1]),
                    [4, 2, 1],
                    ([10, 3, 1], How::Threads),
                    1,
                ),
                (Dispatch::Indirect, [4, 2, 1], ([2, 2, 1], How::Indirect), 0),
                (
                    Dispatch::Threadgroups([3, 2, 1]),
                    [2, 2, 1],
                    ([3, 2, 1], How::Groups),
                    1,
                ),
                (Dispatch::Indirect, [1, 1, 4], ([1, 1, 2], How::Indirect), 1),
                (
                    Dispatch::Threadgroups([u32::MAX, 1, 1]),
                    [2, 1, 1],
                    ([3, 1, 1], How::Groups),
                    1,
                ),
                (
                    Dispatch::Threads([u32::MAX, 1, 1]),
                    [2, 1, 1],
                    ([3, 1, 1], How::Threads),
                    1,
                ),
            ];
            for (prepared, size, (grid, how), made) in cases {
                let lengths = [(0, length(size))];
                let before = counters.metrics();
                state
                    .prepare_dispatch(prepared, size, &lengths, [1, 1])
                    .expect("made ahead");
                let ahead = counters.metrics();
                assert_eq!(
                    (
                        ahead.made_ahead_variants - before.made_ahead_variants,
                        ahead.pipelines_created - before.pipelines_created,
                        ahead.translations - before.translations,
                    ),
                    (made, made, 0),
                    "{name}: {prepared:?} by {size:?}: what the call made"
                );
                state
                    .prepare_dispatch(prepared, size, &lengths, [1, 1])
                    .expect("made ahead");
                let again = counters.metrics();
                assert_eq!(
                    (again.made_ahead_variants, again.pipelines_created),
                    (ahead.made_ahead_variants, ahead.pipelines_created),
                    "{name}: {prepared:?} by {size:?}: a second call makes nothing"
                );
                grid_template_run(name, &device, &state, (grid, size, how), length(size))
                    .unwrap_or_else(|error| panic!("{name}: {how:?} {grid:?}: {error}"));
                let after = counters.metrics();
                assert_eq!(
                    (
                        after.first_use_waits - again.first_use_waits,
                        after.pipelines_created - again.pipelines_created,
                        after.translations - again.translations,
                    ),
                    (0, 0, 0),
                    "{name}: {how:?} {grid:?} by {size:?}: the dispatch found it made"
                );
            }
            // What the dispatch records nothing for, or refuses: the call makes nothing.
            let refusals = [
                (
                    ([0, 4, 1], [4, 4, 1], How::Threads),
                    length([4, 4, 1]),
                    true,
                ),
                (([8, 1, 1], [2048, 1, 1], How::Threads), 16, false),
                (([8, 1, 1], [8, 1, 1], How::Threads), shared + 16, false),
            ];
            for ((grid, size, how), length, runs) in refusals {
                let before = counters.metrics();
                state
                    .prepare_dispatch(Dispatch::Threads(grid), size, &[(0, length)], [1, 1])
                    .expect("made ahead");
                let ahead = counters.metrics();
                assert_eq!(
                    (
                        ahead.made_ahead_variants - before.made_ahead_variants,
                        ahead.pipelines_created - before.pipelines_created,
                    ),
                    (0, 0),
                    "{name}: {grid:?} by {size:?} at {length}: the call made nothing"
                );
                let answer = grid_template_run(name, &device, &state, (grid, size, how), length);
                match (runs, answer) {
                    (true, Ok(())) => {}
                    (false, Err(Error::Refused(Refusal::ThreadgroupSize { .. }))) => {}
                    (false, Err(Error::Limit(Limit::ThreadgroupMemory { .. }))) => {}
                    (_, answer) => panic!("{name}: {grid:?} by {size:?} at {length}: {answer:?}"),
                }
                let after = counters.metrics();
                assert_eq!(
                    after.pipelines_created - ahead.pipelines_created,
                    0,
                    "{name}: {grid:?} by {size:?} at {length}: the dispatch made nothing"
                );
            }
            println!(
                "{name}: prepared dispatches found made: {} kinds, 3 that make nothing",
                cases.len()
            );
        });
    }

    /// Each make of a kernel's `VkPipeline` is kept in the device's counters with its key and who
    /// started it: a tiling kernel's tiles by its state's `finish_pipeline`, a size
    /// `prepare_dispatch` was given by that call, and a size only a dispatch gives by the
    /// dispatch, a first-use wait counted on the make. Each key is the one `dispatch_variant`
    /// answers for the dispatch, and the first-use waits the dispatch added are the waits its make
    /// counts.
    #[test]
    fn every_device_keeps_each_variant_make_with_its_key_and_who_started_it() {
        use crate::{Counters, Dispatch, VariantMake, VariantStart};

        let (Some(tiling), Some(observing)) = (
            descriptor("a1_bool_from_float"),
            descriptor("icb_replay_grid_template"),
        ) else {
            return;
        };
        let next = |counters: &Counters| {
            (counters.variant_makes(0).last()).map_or(0, |make| make.number + 1)
        };
        let one = |name: &str, makes: Vec<VariantMake>, what: &str| -> VariantMake {
            assert_eq!(makes.len(), 1, "{name}: {what}: {makes:?}");
            let make = makes[0];
            assert!(
                make.made && make.began <= make.ended,
                "{name}: {what}: {make:?}"
            );
            make
        };
        each_device("variant_makes", |name, _, device| {
            let counters = device.counters();
            let from = next(&counters);
            let tiles = state(name, &device, &tiling);
            let make = one(name, counters.variant_makes(from), "the state's tiles");
            let key =
                (tiles.dispatch_variant(Dispatch::Threads([300, 1, 1]), [4, 1, 1], &[], [1, 1]))
                    .map(|(variant, _)| variant);
            assert_eq!(
                (Some(make.variant), make.start, make.waits, make.first_wait),
                (key, VariantStart::Finish, 0, 0),
                "{name}: the state's tiles"
            );

            let state = state(name, &device, &observing);
            let length = |size: [u32; 3]| (4 * size.iter().product::<u32>()).next_multiple_of(16);
            let (grid, size) = ([8, 4, 1], [4, 2, 1]);
            let lengths = [(0, length(size))];
            let from = next(&counters);
            state
                .prepare_dispatch(Dispatch::Threads(grid), size, &lengths, [1, 1])
                .expect("made ahead");
            let make = one(name, counters.variant_makes(from), "prepare_dispatch");
            let key = (state.dispatch_variant(Dispatch::Threads(grid), size, &lengths, [1, 1]))
                .map(|(variant, _)| variant);
            assert_eq!(
                (Some(make.variant), make.start, make.waits),
                (key, VariantStart::Ahead, 0),
                "{name}: prepare_dispatch"
            );

            let (groups, size) = ([3, 2, 1], [2, 2, 1]);
            let lengths = [(0, length(size))];
            let (from, before) = (next(&counters), counters.metrics());
            grid_template_run(
                name,
                &device,
                &state,
                (groups, size, How::Groups),
                length(size),
            )
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let after = counters.metrics();
            let make = one(name, counters.variant_makes(from), "the first use");
            let key =
                (state.dispatch_variant(Dispatch::Threadgroups(groups), size, &lengths, [1, 1]))
                    .map(|(variant, _)| variant);
            assert_eq!(
                (Some(make.variant), make.start, make.waits),
                (key, VariantStart::FirstUse, 1),
                "{name}: the first use"
            );
            assert_eq!(
                after.first_use_variant_waits - before.first_use_variant_waits,
                1,
                "{name}: the first use's wait"
            );
            assert!(
                make.first_wait > 0 && make.first_wait <= make.began,
                "{name}: the wait began before the make: {make:?}"
            );
            println!("{name}: variant makes kept: the state's, prepare_dispatch's, a first use's");
        });
    }

    /// A state answers before its main pipeline is made: its creation makes no `VkPipeline` and
    /// answers what Metal's state answers, the tiles' key answered not made; every answer but
    /// that one bit is the same after the finish as before it. Its
    /// `finish_pipeline` makes that pipeline after the answer, counted as finished and kept as a
    /// make the call started, a second call makes nothing, and a dispatch then waits for nothing.
    /// A state of another content that is never finished is dispatched all the same: the dispatch
    /// makes its main pipeline, a first-use wait on an unfinished pipeline, and a
    /// `finish_pipeline` after it makes nothing.
    #[test]
    fn every_device_answers_a_state_before_its_main_pipeline_is_made() {
        use crate::{Dispatch, Metrics, VariantStart};

        let Some(descriptor) = descriptor("a1_bool_from_float") else {
            return;
        };
        // Another content of the same kernel: a required size is part of a content's key.
        let mut required = descriptor.clone();
        required.required_threads_per_threadgroup = Some([GROUP, 1, 1]);
        each_device("compute_answered_early", |name, _, device| {
            let counters = device.counters();
            let mut buffers = Buffers::new(&device, 64 * 1024);
            let slots: Vec<Buffer> = (0..3).map(|_| buffers.buffer(8 * 1024)).collect();
            let queue = device.new_command_queue(4).expect("a queue");
            let dispatch = |state: &ComputePipelineState| {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(state).expect("a state");
                for (index, slot) in (0..).zip(&slots) {
                    encoder.set_buffer(Some(slot), 0, index).expect("a buffer");
                }
                encoder
                    .dispatch_threads([WIDE, 1, 1], [GROUP, 1, 1])
                    .expect("a dispatch");
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
            };
            // The pipelines made, finished, failed to finish and waited for unfinished.
            let moved = |before: &Metrics, after: &Metrics| {
                (
                    after.pipelines_created - before.pipelines_created,
                    after.finished_pipelines - before.finished_pipelines,
                    after.failed_finishes - before.failed_finishes,
                    after.first_use_unfinished_waits - before.first_use_unfinished_waits,
                )
            };
            let tiles = |state: &ComputePipelineState| {
                state.dispatch_variant(Dispatch::Threads([WIDE, 1, 1]), [GROUP, 1, 1], &[], [1, 1])
            };
            // Every answer of the state, `dispatch_variant`'s key among them but not whether its
            // pipeline is made.
            let answers = |state: &ComputePipelineState| {
                (
                    state.max_total_threads_per_threadgroup(),
                    state.static_threadgroup_memory_length(),
                    state.thread_execution_width(),
                    state
                        .imageblock_memory_length_for_dimensions([8, 4, 3])
                        .ok(),
                    state.required_threads_per_threadgroup(),
                    state.support_indirect_command_buffers(),
                    state.texture_accesses(),
                    state.buffer_accesses(),
                    state.precision_notes(),
                    tiles(state).map(|(key, _)| key),
                )
            };

            let before = counters.metrics();
            let state = unfinished(name, &device, &descriptor);
            let answered = counters.metrics();
            assert_eq!(
                moved(&before, &answered),
                (0, 0, 0, 0),
                "{name}: no pipeline at the answer"
            );
            let early = answers(&state);
            assert_eq!(state.thread_execution_width(), 32, "{name}: Metal's answer");
            assert!(state.max_total_threads_per_threadgroup() > 0, "{name}");
            let (key, made) = tiles(&state).expect("the tiles' key");
            assert!(
                !made,
                "{name}: the tiles' pipeline is not made at the answer"
            );
            state.finish_pipeline().expect("finished");
            assert_eq!(
                answers(&state),
                early,
                "{name}: every answer is the same before the finish and after it"
            );
            let finished_at = counters.metrics();
            assert_eq!(
                moved(&answered, &finished_at),
                (1, 1, 0, 0),
                "{name}: the main pipeline, made after the answer"
            );
            assert!(
                finished_at.finished_pipeline_nanos > answered.finished_pipeline_nanos,
                "{name}: the finish's time"
            );
            let make = *(counters.variant_makes(0).last()).expect("a make kept");
            assert_eq!(
                (make.variant, make.start, make.made, make.waits),
                (key, VariantStart::Finish, true, 0),
                "{name}: the finish's make"
            );
            assert_eq!(tiles(&state), Some((key, true)), "{name}: made");
            state.finish_pipeline().expect("finished again");
            dispatch(&state);
            assert_eq!(
                moved(&finished_at, &counters.metrics()),
                (0, 0, 0, 0),
                "{name}: a second finish and a dispatch make nothing"
            );
            assert_eq!(
                counters.metrics().first_use_waits,
                finished_at.first_use_waits,
                "{name}: the dispatch waited for nothing"
            );

            let before = counters.metrics();
            let never = unfinished(name, &device, &required);
            dispatch(&never);
            let dispatched = counters.metrics();
            assert_eq!(
                moved(&before, &dispatched),
                (1, 0, 0, 1),
                "{name}: the dispatch made the unfinished pipeline and waited for it"
            );
            assert!(
                dispatched.first_use_unfinished_wait_nanos > before.first_use_unfinished_wait_nanos,
                "{name}: the wait's time"
            );
            never
                .finish_pipeline()
                .expect("finished after the dispatch");
            assert_eq!(
                moved(&dispatched, &counters.metrics()),
                (0, 0, 0, 0),
                "{name}: a finish after the dispatch makes nothing"
            );
            println!(
                "{name}: answered with no pipeline made; finished after the answer in {} ns; \
                 an unfinished state's dispatch made its pipeline in {} ns",
                finished_at.finished_pipeline_nanos - answered.finished_pipeline_nanos,
                dispatched.first_use_unfinished_wait_nanos - before.first_use_unfinished_wait_nanos,
            );
        });
    }

    /// A state's main pipeline is made once, whichever of its `finish_pipeline` and a
    /// `prepare_dispatch` call for its key comes first: a preparation after the finish makes no
    /// second `VkPipeline`, a finish after a preparation that made it makes nothing, and the two
    /// called at once on threads of their own make one between them, the later returning at once
    /// while the earlier makes it. So a consumer that reads `dispatch_variant`'s `made` false and
    /// prepares costs no second driver create.
    #[test]
    fn every_device_makes_a_state_s_main_pipeline_once_across_its_finish_and_a_preparation() {
        use crate::{Dispatch, Metrics};
        use std::sync::Barrier;
        use std::thread;

        let Some(descriptor) = descriptor("a1_bool_from_float") else {
            return;
        };
        // Contents of one kernel, a required size being part of a content's key, so that each
        // order starts from a state whose main pipeline no other state made.
        let sized = |size: u32| {
            let mut sized = descriptor.clone();
            sized.required_threads_per_threadgroup = Some([size, 1, 1]);
            sized
        };
        let (prepared_first, at_once) = (sized(GROUP), sized(GROUP / 2));
        each_device("compute_finish_and_prepare", |name, _, device| {
            let counters = device.counters();
            // The pipelines made, and the makes the finish and the preparation counted.
            let moved = |before: &Metrics, after: &Metrics| {
                (
                    after.pipelines_created - before.pipelines_created,
                    after.finished_pipelines - before.finished_pipelines,
                    after.made_ahead_variants - before.made_ahead_variants,
                )
            };
            // A preparation of a direct dispatch, which runs the tiles, the main pipeline.
            let prepare = |state: &ComputePipelineState, size: u32| {
                (state.prepare_dispatch(
                    Dispatch::Threads([WIDE, 1, 1]),
                    [size, 1, 1],
                    &[],
                    [1, 1],
                ))
                .expect("prepared");
            };

            let state = unfinished(name, &device, &descriptor);
            state.finish_pipeline().expect("finished");
            let finished_at = counters.metrics();
            prepare(&state, GROUP);
            assert_eq!(
                moved(&finished_at, &counters.metrics()),
                (0, 0, 0),
                "{name}: a preparation after the finish makes nothing"
            );

            let state = unfinished(name, &device, &prepared_first);
            let before = counters.metrics();
            prepare(&state, GROUP);
            let prepared = counters.metrics();
            assert_eq!(
                moved(&before, &prepared),
                (1, 0, 1),
                "{name}: the preparation made the main pipeline"
            );
            state.finish_pipeline().expect("finished");
            assert_eq!(
                moved(&prepared, &counters.metrics()),
                (0, 0, 0),
                "{name}: a finish after the preparation makes nothing"
            );

            let state = unfinished(name, &device, &at_once);
            let before = counters.metrics();
            let start = Barrier::new(2);
            thread::scope(|scope| {
                let finishing = scope.spawn(|| {
                    start.wait();
                    state.finish_pipeline()
                });
                start.wait();
                prepare(&state, GROUP / 2);
                (finishing.join())
                    .expect("the finishing thread")
                    .expect("finished");
            });
            let (created, finished, ahead) = moved(&before, &counters.metrics());
            assert_eq!(
                (created, finished + ahead),
                (1, 1),
                "{name}: one make between a finish and a preparation at once"
            );
            println!(
                "{name}: one main pipeline each: finish then prepare, prepare then finish, and \
                 both at once (made by the {})",
                if finished == 1 {
                    "finish"
                } else {
                    "preparation"
                }
            );
        });
    }

    /// A state's make counts the time of its `vkCreateComputePipelines` calls in
    /// `Metrics::pipeline_creation_nanos` and of its calls into the translator in
    /// `translation_nanos`: each above 0, and the two, made on the make's own thread, within the
    /// make's time.
    #[test]
    fn every_device_times_a_make_s_pipeline_creation_apart_from_its_translation() {
        let Some(descriptor) = descriptor("a1_bool_from_float") else {
            return;
        };
        each_device("compute_make_times", |name, _, device| {
            crate::tests::blit::on_device::settled(&device);
            let counters = device.counters();
            let before = counters.metrics();
            let began = host_time();
            let made = state(name, &device, &descriptor);
            let wall = host_time() - began;
            let after = counters.metrics();
            let creation = after.pipeline_creation_nanos - before.pipeline_creation_nanos;
            let translation = after.translation_nanos - before.translation_nanos;
            assert!(
                after.pipelines_created > before.pipelines_created,
                "{name}: the make created a pipeline"
            );
            assert!(
                creation > 0 && translation > 0,
                "{name}: creation {creation} ns, translation {translation} ns"
            );
            assert!(
                creation + translation < wall,
                "{name}: creation {creation} ns and translation {translation} ns within {wall} ns"
            );
            println!(
                "{name}: the make took {wall} ns: creation {creation} ns, translation {translation} ns"
            );
            drop((made, device));
        });
    }

    /// `ComputePipelineState::dispatch_variant`'s keys name exactly the `VkPipeline`s the
    /// dispatches run: over a table of dispatches of `icb_replay_grid_template`, an observing
    /// kernel with a threadgroup memory argument, two keys are equal exactly when the entries run
    /// one module at one size and threadgroup memory length. Grids that differ only in their
    /// extent or their call, and an indirect dispatch, share a key on the plain module, the ragged
    /// ones on the partial-threadgroups variant another, and a different threadgroup size or
    /// length gives another. `prepare_dispatch` on
    /// each entry in turn makes one pipeline for each key not met before and none for one met. A
    /// second state of the same content answers the same keys and finds their pipelines made; a
    /// state of another content (the same function with a required threadgroup size) answers
    /// another key and makes its own pipeline. What the dispatch refuses or records nothing for
    /// has no key.
    #[test]
    fn every_device_keys_a_dispatch_by_the_pipeline_it_runs() {
        use crate::Dispatch;
        use std::collections::HashSet;

        let Some(descriptor) = descriptor("icb_replay_grid_template") else {
            return;
        };
        each_device("compute_dispatch_variant", |name, _, device| {
            let raw = device.raw();
            // SAFETY: the physical device is the instance's.
            let shared = unsafe {
                raw.instance
                    .get_physical_device_properties(raw.physical_device)
                    .limits
                    .max_compute_shared_memory_size
            };
            let length = |size: [u32; 3]| (4 * size.iter().product::<u32>()).next_multiple_of(16);
            let (size, small) = ([4, 2, 1], [2, 2, 1]);
            // Each entry: the dispatch, its size, its threadgroup memory length, and the pipeline
            // it runs.
            let table = [
                (Dispatch::Threads([8, 4, 1]), size, length(size), "whole"),
                (Dispatch::Threads([16, 8, 1]), size, length(size), "whole"),
                (
                    Dispatch::Threadgroups([3, 2, 1]),
                    size,
                    length(size),
                    "whole",
                ),
                (Dispatch::Indirect, size, length(size), "whole"),
                (Dispatch::Threads([10, 3, 1]), size, length(size), "partial"),
                (Dispatch::Threads([9, 5, 1]), size, length(size), "partial"),
                (
                    Dispatch::Threadgroups([3, 2, 1]),
                    small,
                    length(small),
                    "whole, small",
                ),
                (
                    Dispatch::Threads([8, 4, 1]),
                    size,
                    2 * length(size),
                    "whole, longer",
                ),
            ];
            let first = state(name, &device, &descriptor);
            type Entry<'a> = (Dispatch, [u32; 3], u32, &'a str);
            let key = |state: &ComputePipelineState, (dispatch, size, length, pipeline): Entry| {
                (state.dispatch_variant(dispatch, size, &[(0, length)], [1, 1]))
                    .map(|(variant, _)| variant)
                    .unwrap_or_else(|| panic!("{name}: {pipeline}: {dispatch:?} by {size:?}"))
            };
            let keys: Vec<_> = table.iter().map(|&entry| key(&first, entry)).collect();
            for (a, key_a) in table.iter().zip(&keys) {
                for (b, key_b) in table.iter().zip(&keys) {
                    assert_eq!(key_a == key_b, a.3 == b.3, "{name}: {a:?} and {b:?}");
                }
            }
            let counters = device.counters();
            let mut met = HashSet::new();
            for (&(dispatch, size, length, pipeline), variant) in table.iter().zip(&keys) {
                let before = counters.metrics();
                first
                    .prepare_dispatch(dispatch, size, &[(0, length)], [1, 1])
                    .expect("made ahead");
                let after = counters.metrics();
                let new = u64::from(met.insert(*variant));
                assert_eq!(
                    (
                        after.made_ahead_variants - before.made_ahead_variants,
                        after.pipelines_created - before.pipelines_created,
                    ),
                    (new, new),
                    "{name}: {pipeline}: {dispatch:?} by {size:?}: a make for a key not met"
                );
            }
            let pipelines: HashSet<&str> = table.iter().map(|entry| entry.3).collect();
            assert_eq!(
                met.len(),
                pipelines.len(),
                "{name}: one key for each pipeline"
            );
            let again = state(name, &device, &descriptor);
            let before = counters.metrics();
            for (&entry, made) in table.iter().zip(&keys) {
                assert_eq!(
                    key(&again, entry),
                    *made,
                    "{name}: {entry:?}: the content's key"
                );
                let (dispatch, size, length, _) = entry;
                again
                    .prepare_dispatch(dispatch, size, &[(0, length)], [1, 1])
                    .expect("made ahead");
            }
            assert_eq!(
                counters.metrics().made_ahead_variants - before.made_ahead_variants,
                0,
                "{name}: the content's pipelines, made"
            );
            let mut required = descriptor.clone();
            required.required_threads_per_threadgroup = Some(size);
            let other = state(name, &device, &required);
            let other_key = key(&other, table[0]);
            assert!(!keys.contains(&other_key), "{name}: another content's key");
            let before = counters.metrics();
            let (dispatch, at, bytes, _) = table[0];
            other
                .prepare_dispatch(dispatch, at, &[(0, bytes)], [1, 1])
                .expect("made ahead");
            assert_eq!(
                counters.metrics().made_ahead_variants - before.made_ahead_variants,
                1,
                "{name}: another content's pipeline"
            );
            let refused = [
                (Dispatch::Threads([0, 4, 1]), [4, 4, 1], length([4, 4, 1])),
                (Dispatch::Threads([8, 1, 1]), [2048, 1, 1], 16),
                (Dispatch::Threads([8, 1, 1]), [8, 1, 1], shared + 16),
            ];
            for (dispatch, size, length) in refused {
                assert_eq!(
                    first.dispatch_variant(dispatch, size, &[(0, length)], [1, 1]),
                    None,
                    "{name}: {dispatch:?} by {size:?} at {length}: no key"
                );
            }
            println!(
                "{name}: {} dispatches, {} keys, one make each",
                table.len(),
                met.len()
            );
        });
    }

    /// `dispatch_variant` answers with each key whether its pipeline is made: false for each of
    /// ten threadgroup sizes of `icb_replay_grid_template` until `prepare_dispatch` makes it, true
    /// after, the same key both times, past the kernel's eight variant cells too; and true for each
    /// once all are made, for a second state of the content as well.
    #[test]
    fn every_device_answers_whether_a_dispatch_s_pipeline_is_made() {
        use crate::Dispatch;

        let Some(descriptor) = descriptor("icb_replay_grid_template") else {
            return;
        };
        each_device("compute_variant_made", |name, _, device| {
            let state = state(name, &device, &descriptor);
            let dispatch = Dispatch::Threadgroups([2, 1, 1]);
            let lengths = |x: u32| [(0, (4 * x).next_multiple_of(16))];
            let mut keys = Vec::new();
            for x in 1..=10 {
                let (size, lengths) = ([x, 1, 1], lengths(x));
                let (key, made) = (state.dispatch_variant(dispatch, size, &lengths, [1, 1]))
                    .unwrap_or_else(|| panic!("{name}: {size:?}: no key"));
                assert!(!made, "{name}: {size:?}: made before its make");
                state
                    .prepare_dispatch(dispatch, size, &lengths, [1, 1])
                    .expect("made ahead");
                let answer = state.dispatch_variant(dispatch, size, &lengths, [1, 1]);
                assert_eq!(
                    answer,
                    Some((key, true)),
                    "{name}: {size:?}: after its make"
                );
                keys.push(key);
            }
            let again = self::state(name, &device, &descriptor);
            for (x, key) in (1..=10).zip(keys) {
                let (size, lengths) = ([x, 1, 1], lengths(x));
                for state in [&state, &again] {
                    let answer = state.dispatch_variant(dispatch, size, &lengths, [1, 1]);
                    assert_eq!(answer, Some((key, true)), "{name}: {size:?}: all made");
                }
            }
            println!("{name}: made answers: ten sizes, before and after their makes");
        });
    }

    /// `simd_float_products`, a kernel that runs only whole SIMD groups, has its emulated-SIMD-group
    /// variant lowered with its state; `prepare_dispatch` in threadgroups of 80, which are not
    /// whole SIMD groups, makes that variant's pipeline and lowers nothing: the dispatch then adds
    /// no first-use wait and makes nothing. On a device whose module runs any threadgroup, the
    /// state has no such variant and the call makes the size's pipeline. A device that cannot pin
    /// SIMD groups at 32 has no state of the kernel, and the test says so.
    #[test]
    fn every_device_prepares_the_emulated_simdgroup_variant() {
        use crate::Dispatch;

        let Some(descriptor) = descriptor_of("gsm2_simd_products", "simd_float_products") else {
            return;
        };
        each_device("compute_prepare_simd", |name, _, device| {
            let state = match device
                .new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers)
            {
                Ok(state) => state,
                Err(Error::Unsupported(Gap::KernelInput { .. })) => {
                    println!(
                        "{name}: SIMD groups not pinnable at 32; the tier's kernel is refused"
                    );
                    return;
                }
                Err(error) => panic!("{name}: {error}"),
            };
            let emulates = !state.inner.compiled.whole.interface.multiples.is_empty();
            assert_eq!(
                emulated_made(&state),
                emulates,
                "{name}: the emulated module, made with the state"
            );
            let counters = device.counters();
            let (groups, size) = ([3, 1, 1], [80, 1, 1]);
            let before = counters.metrics();
            state
                .prepare_dispatch(Dispatch::Threadgroups(groups), size, &[], [1, 1])
                .expect("made ahead");
            let ahead = counters.metrics();
            assert_eq!(
                (
                    ahead.made_ahead_variants - before.made_ahead_variants,
                    ahead.translations - before.translations,
                ),
                (1, 0),
                "{name}: what the call made"
            );
            let threads = groups[0] * size[0];
            let mut buffers = Buffers::new(&device, 1 << 16);
            let a = buffers.buffer(u64::from(threads) * 4);
            let out = buffers.buffer(u64::from(threads) * 16);
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            encoder.set_buffer(Some(&a), 0, 0).expect("a buffer");
            encoder.set_buffer(Some(&out), 0, 1).expect("a buffer");
            encoder
                .dispatch_threadgroups(groups, size)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let after = counters.metrics();
            assert_eq!(
                (
                    after.first_use_waits - ahead.first_use_waits,
                    after.pipelines_created - ahead.pipelines_created,
                ),
                (0, 0),
                "{name}: the dispatch found it made"
            );
            println!("{name}: threadgroups of 80 prepared, emulated {emulates}");
        });
    }

    /// One run of `icb_replay_grid_template`'s dispatches on `device`, each prepared first where
    /// `prepare`: the counts of what the runtime recorded and submitted.
    fn recorded_counts(
        name: &str,
        device: &Device,
        descriptor: &ComputePipelineDescriptor,
        prepare: bool,
    ) -> Vec<(&'static str, u64)> {
        let state = state(name, device, descriptor);
        let counters = device.counters();
        let before = counters.metrics();
        let runs = [
            ([10, 3, 1], [4, 2, 1], How::Threads),
            ([8, 4, 1], [4, 2, 1], How::Threads),
            ([3, 2, 1], [4, 2, 1], How::Groups),
            ([1, 1, 2], [1, 1, 320], How::Groups),
            ([1, 1, 2], [1, 1, 8], How::Indirect),
        ];
        for (grid, size, how) in runs {
            let length = (4 * size.iter().product::<u32>()).next_multiple_of(16);
            if prepare {
                let dispatch = match how {
                    How::Threads => crate::Dispatch::Threads(grid),
                    How::Groups => crate::Dispatch::Threadgroups(grid),
                    How::Indirect => crate::Dispatch::Indirect,
                };
                state
                    .prepare_dispatch(dispatch, size, &[(0, length)], [1, 1])
                    .expect("made ahead");
            }
            grid_template_run(name, device, &state, (grid, size, how), length)
                .unwrap_or_else(|error| panic!("{name}: {how:?} {grid:?}: {error}"));
        }
        let after = counters.metrics();
        vec![
            ("submissions", after.submissions - before.submissions),
            ("submit_infos", after.submit_infos - before.submit_infos),
            ("passes", after.passes - before.passes),
            ("barriers", after.barriers - before.barriers),
            ("bytes_copied", after.bytes_copied - before.bytes_copied),
            (
                "pipelines_created",
                after.pipelines_created - before.pipelines_created,
            ),
            ("translations", after.translations - before.translations),
            ("ring_writes", after.ring_writes - before.ring_writes),
            ("pushes", after.pushes - before.pushes),
        ]
    }

    /// What the runtime records of a run of dispatches is the same with each prepared first and
    /// without: on two devices of one physical device, the same dispatches of
    /// `icb_replay_grid_template` (whole, ragged, threadgroup counts, deep and indirect) submit,
    /// pass, barrier, copy, push and write their ring alike, make the same pipelines and
    /// translations, and answer Metal's words.
    #[test]
    fn every_device_records_the_same_with_and_without_prepare_dispatch() {
        let Some(descriptor) = descriptor("icb_replay_grid_template") else {
            return;
        };
        each_device("compute_prepare_recorded", |name, instance, device| {
            let devices = devices_of_the_cfgs_kind(instance);
            let (_, physical) = (devices.iter())
                .find(|(other, _)| other == name)
                .expect("the device's own physical device");
            let plain = recorded_counts(name, &device, &descriptor, false);
            let second = Device::new(physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let prepared = recorded_counts(name, &second, &descriptor, true);
            drop(second);
            assert_eq!(plain, prepared, "{name}: what was recorded and submitted");
            println!("{name}: recorded alike with and without prepare_dispatch: {plain:?}");
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
    /// the emulated-SIMD-group module the state's make lowered, so no dispatch makes a translation.
    /// Every answer is Metal's: the products over each SIMD group's threads, exact in any order
    /// for these inputs. A device that cannot pin 32 may take a tier that is refused.
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
                Err(Error::Unsupported(Gap::KernelInput { .. })) if !pinnable => {
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
            let made = device.counters().metrics().translations;
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
                let emulated = || device.counters().metrics().emulated_simdgroup_dispatches;
                let before = emulated();
                encoder
                    .dispatch_threadgroups([GROUPS, 1, 1], [size, 1, 1])
                    .unwrap_or_else(|error| panic!("{name}: threadgroups of {size}: {error}"));
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
                let partial = (interface.multiples.iter()).any(|&width| size % width.max(1) != 0);
                assert_eq!(
                    emulated() - before,
                    u64::from(partial),
                    "{name}: threadgroups of {size}, whole groups of {:?}: emulated dispatches",
                    interface.multiples
                );
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
            assert_eq!(
                device.counters().metrics().translations,
                made,
                "{name}: no translation at a dispatch"
            );
            let emulated = state.inner.compiled.emulated.as_ref();
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

    /// `quad_shuffle_rotate` of `quad_shuffle`, a quad-only kernel: each thread reads the next
    /// lane of its quad, `(t & !3) | ((t + 1) & 3)` of thread `t`, Metal's answer for every whole
    /// quad, in threadgroups of 64, 32, 40, 12 and 4 threads. Where the device's narrowest compute
    /// subgroup is below 32 lanes and at least 4, compute can take a size
    /// (`requiredSubgroupSizeStages`) and runs full subgroups, the kernel takes its quads from the
    /// subgroup at the size the device gives: no pinned size, `ALLOW_VARYING_SUBGROUP_SIZE`, and
    /// whole groups of the largest size; elsewhere it keeps the tier its class takes. A
    /// threadgroup that is not whole groups of the module's runs the emulated-SIMD-group module.
    /// Prints the dispatches each module ran.
    #[test]
    fn every_device_runs_a_quad_only_kernel_at_the_size_it_gives() {
        use ash::vk;

        let Some(descriptor) = descriptor_of("quad_shuffle", "quad_shuffle_rotate") else {
            return;
        };
        const GROUPS: u32 = 2;
        each_device("compute_quads", |name, _, device| {
            let (min, max, quad_free) = quad_sizes(&device);
            let state = device
                .new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let interface = &state.inner.compiled.whole.interface;
            let varying = interface
                .stage_flags
                .contains(vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE);
            assert_eq!(varying, quad_free, "{name}: subgroups {min}..{max}");
            if quad_free {
                assert_eq!(
                    (interface.required_subgroup_size, &interface.multiples[..]),
                    (None, &[max][..]),
                    "{name}: the quads at the size the device gives"
                );
                assert!(
                    interface
                        .stage_flags
                        .contains(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS),
                    "{name}"
                );
            }
            let queue = device.new_command_queue(4).expect("a queue");
            let made = device.counters().metrics().translations;
            let (mut whole, mut emulated) = (0, 0);
            for size in [64, 32, 40, 12, 4] {
                let threads = GROUPS * size;
                let mut buffers = Buffers::new(&device, 1 << 16);
                let out = buffers.buffer(u64::from(threads) * 4);
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                encoder.set_buffer(Some(&out), 0, 0).expect("a buffer");
                let count = || device.counters().metrics().emulated_simdgroup_dispatches;
                let before = count();
                encoder
                    .dispatch_threadgroups([GROUPS, 1, 1], [size, 1, 1])
                    .unwrap_or_else(|error| panic!("{name}: threadgroups of {size}: {error}"));
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
                let partial = (interface.multiples.iter()).any(|&width| size % width.max(1) != 0);
                assert_eq!(
                    count() - before,
                    u64::from(partial),
                    "{name}: threadgroups of {size}, whole groups of {:?}: emulated dispatches",
                    interface.multiples
                );
                if partial {
                    emulated += 1;
                } else {
                    whole += 1;
                }
                let want: Vec<u32> = (0..size).map(|t| (t & !3) | ((t + 1) & 3)).collect();
                assert_eq!(
                    words(&out, 0, want.len()),
                    want,
                    "{name}: the quads in threadgroups of {size}"
                );
            }
            assert_eq!(
                device.counters().metrics().translations,
                made,
                "{name}: no translation at a dispatch"
            );
            println!(
                "{name}: quad-only kernel, subgroups {min}..{max}: {} ({:?} pinned, whole groups of \
                 {:?}); dispatches on that module {whole}, on the emulated module {emulated}",
                if varying {
                    "quads at the size the device gives"
                } else {
                    "the tier its class takes"
                },
                interface.required_subgroup_size,
                interface.multiples,
            );
        });
    }

    /// The compute subgroup sizes of `device`, `minSubgroupSize` and `maxSubgroupSize`, and
    /// whether a quad-only kernel takes its quads from the subgroup at the size the device gives
    /// there: compute can take a size (`requiredSubgroupSizeStages`, `subgroupSizeControl`), runs
    /// full subgroups, and its narrowest size is at least 4 and below 32.
    fn quad_sizes(device: &Device) -> (u32, u32, bool) {
        use ash::vk;

        let raw = device.raw();
        let mut vulkan13 = vk::PhysicalDeviceVulkan13Properties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut vulkan13);
        // SAFETY: the physical device is the instance's, a Vulkan 1.3 one.
        unsafe {
            raw.instance
                .get_physical_device_properties2(raw.physical_device, &mut properties)
        };
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut features13);
        // SAFETY: as above.
        unsafe {
            raw.instance
                .get_physical_device_features2(raw.physical_device, &mut features)
        };
        let (min, max) = (vulkan13.min_subgroup_size, vulkan13.max_subgroup_size);
        let quad_free = vulkan13
            .required_subgroup_size_stages
            .contains(vk::ShaderStageFlags::COMPUTE)
            && features13.subgroup_size_control == vk::TRUE
            && features13.compute_full_subgroups == vk::TRUE
            && (4..32).contains(&min);
        (min, max, quad_free)
    }

    /// `quad_edge`, a quad-only kernel: thread `g` of the grid (`thread_position_in_grid`) takes
    /// `g` from the next thread of its quad (`quad_shuffle`, to quad lane `(t + 1) & 3` of its
    /// `thread_index_in_threadgroup` `t`) and its quad's ballot of `g % 3 != 0` (`quad_ballot`, bit
    /// `i` quad lane `i`), and writes the first above the second's 8 bits at `out[g]`.
    const QUAD_EDGE: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @quad_edge(ptr addrspace(1) noundef "air-buffer-no-alias" %out, i32 noundef %g, i32 noundef %t) local_unnamed_addr #0 {
  %t16 = trunc i32 %t to i16
  %next = add i16 %t16, 1
  %lane = and i16 %next, 3
  %v = tail call i32 @air.quad_shuffle.u.i32(i32 %g, i16 %lane) #1
  %r = urem i32 %g, 3
  %p = icmp ne i32 %r, 0
  %b = tail call i16 @air.quad_ballot(i1 %p) #1
  %bits = zext i16 %b to i32
  %high = shl i32 %v, 8
  %word = or i32 %high, %bits
  %i = zext i32 %g to i64
  %at = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %word, ptr addrspace(1) %at, align 4
  ret void
}

declare i32 @air.quad_shuffle.u.i32(i32, i16) local_unnamed_addr #0
declare i16 @air.quad_ballot(i1) local_unnamed_addr #0

attributes #0 = { convergent mustprogress nounwind willreturn }
attributes #1 = { convergent nounwind willreturn }

!air.kernel = !{!0}
!0 = !{ptr @quad_edge, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"g"}
!5 = !{i32 2, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

    /// [`QUAD_EDGE`] over `dispatchThreads` grids that are not whole threadgroups: in threadgroups
    /// of 64 threads (or of the variant's largest SIMD-group multiple, where that is larger), two
    /// whole threadgroups and one of 22, whose last quad has two threads, and one threadgroup of
    /// 61, whose last quad has one. A ragged grid runs the kernel's partial-threadgroups variant
    /// (`Metrics::partial_variant_ragged_grids`, and no emulated-SIMD-group dispatch), which takes
    /// its quads from the subgroup at the size the device gives where the plain module does (no
    /// pinned size, `ALLOW_VARYING_SUBGROUP_SIZE`, whole groups of `maxSubgroupSize`): its
    /// invocations past the grid return before the body. Every thread's word is Metal's: its
    /// quad's ballot sets the bits of the quad's threads in the grid whose `g % 3 != 0`, and the
    /// shuffle reads its quad's next thread where that thread is in the grid (a read of a thread
    /// past the grid is not promised, MSL 4.1 section 6.10.2, so the word's upper bits are not
    /// checked there), checked against the CPU's answer; every word past the grid keeps its fill,
    /// so no invocation past the grid writes. Prints whether the variant ran on quads at the size
    /// the device gives, and the ragged grids it ran.
    #[test]
    fn every_device_runs_a_quad_only_kernels_ragged_grid_on_its_partial_variant() {
        use crate::pipeline::Which;
        use ash::vk;

        let bitcode = match llvm_assemble_text(QUAD_EDGE) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("quad_edge: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function = library.function("quad_edge").expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        const FILL: u32 = 0xEEEE_EEEE;
        const PAST: u32 = 8;
        each_device("compute_quad_edge", |name, _, device| {
            let (min, max, quad_free) = quad_sizes(&device);
            let state = state(name, &device, &descriptor);
            let compiled = &state.inner.compiled;
            let partial = (compiled.variant_kernel(Which::Partial, |_| {
                panic!("{name}: the partial-threadgroups variant is refused")
            }))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let interface = &partial.interface;
            let varying = interface
                .stage_flags
                .contains(vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE);
            assert_eq!(varying, quad_free, "{name}: subgroups {min}..{max}");
            if quad_free {
                assert_eq!(
                    (interface.required_subgroup_size, &interface.multiples[..]),
                    (None, &[max][..]),
                    "{name}: the variant's quads at the size the device gives"
                );
                assert!(
                    interface
                        .stage_flags
                        .contains(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS),
                    "{name}"
                );
            }
            let size = interface.multiples.iter().copied().fold(64, u32::max);
            let queue = device.new_command_queue(4).expect("a queue");
            let made = device.counters().metrics().translations;
            let mut ragged = 0;
            for threads in [2 * size + 22, 61] {
                assert!(!threads.is_multiple_of(size), "{name}: {threads} in {size}");
                let mut buffers = Buffers::new(&device, 1 << 16);
                let out = buffers.buffer(u64::from(threads + PAST) * 4);
                write(&out, 0, &le_bytes(&vec![FILL; (threads + PAST) as usize]));
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                encoder.set_buffer(Some(&out), 0, 0).expect("a buffer");
                let before = device.counters().metrics();
                encoder
                    .dispatch_threads([threads, 1, 1], [size, 1, 1])
                    .unwrap_or_else(|error| panic!("{name}: {threads} threads: {error}"));
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
                let after = device.counters().metrics();
                assert_eq!(
                    (
                        after.partial_variant_ragged_grids - before.partial_variant_ragged_grids,
                        after.emulated_simdgroup_dispatches - before.emulated_simdgroup_dispatches,
                    ),
                    (1, 0),
                    "{name}: {threads} threads in {size}: on the partial-threadgroups variant"
                );
                ragged += 1;
                let got = words(&out, 0, (threads + PAST) as usize);
                for (g, &word) in (0..threads).zip(&got) {
                    let quad = g & !3;
                    let ballot = (0..4)
                        .filter(|&lane| quad + lane < threads && (quad + lane) % 3 != 0)
                        .fold(0, |bits, lane| bits | (1 << lane));
                    assert_eq!(
                        word & 0xFF,
                        ballot,
                        "{name}: {threads} threads in {size}: thread {g}'s quad ballot"
                    );
                    let next = quad | ((g + 1) & 3);
                    if next < threads {
                        assert_eq!(
                            word >> 8,
                            next,
                            "{name}: {threads} threads in {size}: thread {g}'s quad shuffle"
                        );
                    }
                }
                assert_eq!(
                    &got[threads as usize..],
                    &[FILL; PAST as usize][..],
                    "{name}: {threads} threads: the words past the grid"
                );
            }
            assert_eq!(
                device.counters().metrics().translations,
                made,
                "{name}: no translation at a dispatch"
            );
            println!(
                "{name}: quad-only kernel's ragged grids, subgroups {min}..{max}: the \
                 partial-threadgroups variant ran {}, {:?} pinned, whole groups of {:?}; ragged \
                 grids on it {ragged}",
                if varying {
                    "on quads at the size the device gives"
                } else {
                    "at the tier its class takes"
                },
                interface.required_subgroup_size,
                interface.multiples,
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
                let emulated = (state.inner.compiled.emulated.as_ref())
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
    /// SIMD group of 16 last). Where its SIMD groups are emulated its walk is predicated: every
    /// device creates the state and runs both sizes, on the module whose walk is predicated where
    /// its tier emulates SIMD groups, and on the emulated module at 8 x 10 where it runs only
    /// whole ones. Every answer is Metal's (M3 Max, macOS 27.0.1): thread `t` adds, at each
    /// iteration `i` of its own, `i + 1` times the threads of its SIMD group whose own trip count
    /// exceeds `i`.
    #[test]
    fn every_device_runs_a_kernel_whose_walk_diverges_by_row() {
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
            println!(
                "{name}: sums checked in threadgroups of 8 x 8 and 8 x 10 (emulated module made: \
                 {})",
                emulated_made(&state)
            );
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

    /// A device on a cache directory a device before it used lowers nothing at creation, and asks
    /// the translator nothing: the tiling kernel's lowering and the observing kernel's plain one
    /// and partial-threadgroups variant are read from the directory, each once, so no stage entry
    /// is decoded, and the tile's `VkPipeline` is made through a pipeline cache started from the
    /// device's blob.
    /// The first device's writes (`Metrics::cache_bytes_written`) are at least the files' bytes,
    /// the blob among them where the device keeps one (written once creation was quiet, and again
    /// at the drop where the cache grew since); the second writes nothing and leaves every file as
    /// it was.
    #[test]
    fn every_device_takes_the_kernels_a_device_before_it_lowered_from_the_cache_directory() {
        let (Some(tiled), Some(observing)) = (
            descriptor("a1_bool_from_float"),
            descriptor("icb_replay_grid_template"),
        ) else {
            return;
        };
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let dir = std::env::temp_dir().join(format!("m2v-rt-cache-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let mut first = Vec::new();
            for round in 0..2 {
                let device = on_directory(name, physical, &dir);
                let kept = keeps_a_blob(name, &device);
                let counters = device.counters();
                let states = [
                    state(name, &device, &tiled),
                    state(name, &device, &observing),
                ];
                let metrics = counters.metrics();
                let counted = (metrics.translations, metrics.translations_loaded);
                // The tiling kernel's lowering, the observing kernel's plain one and its variant.
                let want = [(3, 0), (0, 3)][round];
                assert_eq!(counted, want, "{name}: device {round} on the directory");
                if round == 1 {
                    let decoded = device.core().entries.held();
                    assert_eq!(
                        decoded,
                        (0, 0, 0),
                        "{name}: the second device decodes nothing"
                    );
                }
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
                    let want = usize::from(kept);
                    assert_eq!(blobs(&files).len(), want, "{name}: the device's blob");
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

    /// `scan_static` ([`scan_kernel`]) over declared threadgroup memory of the device's
    /// `maxComputeSharedMemorySize`, whose partial-threadgroups variant the translator refuses for
    /// its vote word: a device on a cache directory keeps the refusal with its reason (the
    /// directory holds the vote word's marker), and a device on the directory after it lowers
    /// nothing, reads the refusal from the directory, and still names the vote word as the site
    /// of its ragged grid's refusal, with the kernel's static bytes and the limit.
    #[test]
    fn every_device_names_the_vote_word_site_from_the_cache_directory() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let dir = std::env::temp_dir().join(format!("m2v-rt-vote-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            for round in 0..2 {
                let device = on_directory(name, physical, &dir);
                let limit = device.core().limits.max_shared_memory;
                let bitcode = match metal2vulkan_devtools::llvm_assemble_text(&scan_kernel(Some(
                    limit / 4,
                ))) {
                    Ok(bitcode) => bitcode,
                    Err(AssembleError::Missing(why)) => {
                        eprintln!("SKIPPED: {why}");
                        return;
                    }
                    Err(error) => panic!("scan_static: {error:?}"),
                };
                let library = Library::from_air(&bitcode).expect("the library");
                let function = library.function("scan_static").expect("the kernel");
                let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                    FunctionDescriptor::new(function),
                ));
                let counters = device.counters();
                let state = state(name, &device, &descriptor);
                let metrics = counters.metrics();
                if round == 1 {
                    assert_eq!(
                        metrics.translations, 0,
                        "{name}: the second device lowers nothing"
                    );
                    assert!(
                        metrics.translations_loaded > 0,
                        "{name}: the second device reads the directory"
                    );
                }
                {
                    let queue = device.new_command_queue(1).expect("a queue");
                    let mut buffers = Buffers::new(&device, 1 << 16);
                    let out = buffers.buffer(4 * 1024);
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    let mut encoder = command_buffer
                        .compute_command_encoder(DispatchType::Serial)
                        .expect("an encoder");
                    encoder.set_compute_pipeline_state(&state).expect("a state");
                    encoder.set_buffer(Some(&out), 0, 0).expect("a buffer");
                    match encoder.dispatch_threads([25, 1, 1], [1024, 1, 1]) {
                        Err(Error::Unsupported(Gap::PartialThreadgroups {
                            site: PartialThreadgroupsSite::VoteWord { static_length, max },
                            ..
                        })) if static_length == limit && max == limit => {}
                        other => panic!("{name}: device {round} on the directory: {other:?}"),
                    }
                    encoder.end_encoding();
                    command_buffer.commit().expect("a commit");
                    finished(name, &command_buffer);
                }
                drop(state);
                drop(device);
                // The teardown ends once the reaper wrote what it was given.
                within("the teardown", || counters.gauges() == Gauges::default());
                if round == 0 {
                    let marker = crate::cache::REFUSED_VOTE;
                    let kept = (files(&dir).iter()).any(|(file, len, _)| {
                        *len == marker.len() as u64
                            && std::fs::read(dir.join(file)).is_ok_and(|bytes| bytes == marker)
                    });
                    assert!(kept, "{name}: the directory keeps the vote word's refusal");
                }
            }
            let _ = std::fs::remove_dir_all(&dir);
            println!(
                "{name}: the vote word named as the site by a first and a second device on the \
                 directory, the second lowering nothing"
            );
        }
    }

    /// What a state of a kernel module answers, and what a dispatch checks of it, beside the
    /// dispatch's own module: for [`every_device_s_observers_answer_the_same_on_their_variant`].
    #[derive(Debug, PartialEq)]
    struct ModuleAnswers {
        /// Metal's `maxTotalThreadsPerThreadgroup`, `threadExecutionWidth` and
        /// `staticThreadgroupMemoryLength`, and its own maximum.
        metal: metal2vulkan::pipeline::ComputePipelineMetalInfo,
        imageblock: Option<bool>,
        notes: Vec<PrecisionNote>,
        numbered: Vec<metal2vulkan::pipeline::NumberedSampler>,
        constexprs: Vec<(usize, metal2vulkan::pipeline::CombinedSampler)>,
        buffers: Vec<(u32, crate::hazard::Access)>,
        textures: Vec<(u32, crate::hazard::Access)>,
        storage_reads: Vec<u32>,
        formatless: Vec<u32>,
        unbound: [bool; 3],
        has_effects: bool,
        stage_in_region: bool,
        private_arena: Option<u32>,
        threadgroups: Vec<u32>,
        threads: [u32; 2],
        multiples: Vec<u32>,
        required_subgroup_size: Option<u32>,
        alignments: Vec<RuntimeFact>,
    }

    impl ModuleAnswers {
        /// The answers of `pipeline`, whose lowering declares an imageblock as `imageblock`
        /// says, on a device whose push window is `window` bytes; Err: what the module reads
        /// that the encoder does not supply.
        fn of(
            pipeline: &metal2vulkan::pipeline::ComputePipeline,
            imageblock: Option<bool>,
            window: u32,
        ) -> Result<ModuleAnswers, crate::KernelInputKind> {
            let interface = crate::pipeline::interface(pipeline, None, window)?;
            let mut notes = pipeline.stage.precision_notes.clone();
            notes.sort_unstable();
            notes.dedup();
            let mut alignments: Vec<RuntimeFact> = (pipeline.dependencies.iter())
                .map(|dependency| dependency.fact)
                .filter(|fact| matches!(fact, RuntimeFact::AddressAlignment { .. }))
                .collect();
            alignments.sort_unstable();
            Ok(ModuleAnswers {
                metal: pipeline.metal_info,
                imageblock,
                notes,
                numbered: interface.numbered,
                constexprs: interface.constexprs,
                buffers: interface.buffers,
                textures: interface.textures,
                storage_reads: interface.storage_reads,
                formatless: interface.formatless,
                unbound: [
                    interface.reads_unbound,
                    interface.writes_unbound,
                    interface.unbound_storage,
                ],
                has_effects: interface.has_effects,
                stage_in_region: interface.stage_in_region,
                private_arena: interface.private_arena,
                threadgroups: interface.threadgroups,
                threads: [
                    interface.max_total_threads,
                    interface.metal_max_total_threads,
                ],
                multiples: interface.multiples,
                required_subgroup_size: interface.required_subgroup_size,
                alignments,
            })
        }
    }

    /// Every authored kernel of the public fixtures (`fixtures/public` and the directories under
    /// it: the engine's, which the conformance suite's cases run, and the harness's among them)
    /// that observes its threadgroup partition on the device, lowered plainly and as its
    /// partial-threadgroups variant by the device's translator. Where the translator gives both,
    /// the variant answers what the plain module does ([`ModuleAnswers`]): Metal's
    /// `maxTotalThreadsPerThreadgroup`, `threadExecutionWidth` and `staticThreadgroupMemoryLength`
    /// and its own maximum, the imageblock answer, the precision notes, the samplers' numbering
    /// and constexpr uses, the buffer and texture accesses, whether the kernel writes memory,
    /// reads the stage-in region or takes a private arena, its storage reads, formatless slots and
    /// unbound accesses, its threadgroup arguments, SIMD-group multiples and subgroup size, and the
    /// buffer alignments it takes. So a ragged grid, which runs the variant, finds what the
    /// state's plain module answers.
    ///
    /// One exception: on a device without maximal reconvergence, the variant may carry
    /// [`PrecisionNote::DeviceReconvergence`] where the plain module does not. The variant's
    /// invocations past the grid return before the body, and SPIR-V states reconvergence over all
    /// invocations, returned ones included (SPIR-V 1.6 section 2.2.5, "Dynamic Instance":
    /// "Invocations may only resume executing the same dynamic instances when all invocations
    /// reach the same static instruction"). Only `MaximallyReconvergesKHR` lets an invocation
    /// that returned escape reconvergence, so without it the variant's tangled instructions
    /// reconverge as the device does, even where the plain module's are reached together. With
    /// maximal reconvergence the notes are equal too. Prints how many kernels it compared, how
    /// many took the exception, and each one it could not compare, with why.
    #[test]
    fn every_device_s_observers_answer_the_same_on_their_variant() {
        use metal2vulkan::library::FunctionStage;
        use metal2vulkan::pipeline::{ComputePipeline, LoweringInputs};

        // Every `.ll` under the public fixtures that declares a kernel, in path order.
        let mut files = Vec::new();
        let mut dirs = vec![metal2vulkan_devtools::fixtures_dir()];
        while let Some(dir) = dirs.pop() {
            for entry in std::fs::read_dir(&dir).expect("the fixtures") {
                let path = entry.expect("a fixture").path();
                if path.is_dir() {
                    dirs.push(path);
                } else if path.extension().is_some_and(|extension| extension == "ll") {
                    files.push(path);
                }
            }
        }
        files.sort();
        let mut libraries = Vec::new();
        let mut unread = Vec::new();
        for path in files {
            let text = std::fs::read_to_string(&path).expect("a fixture's text");
            if !text.contains("!air.kernel") {
                continue;
            }
            let file = path.strip_prefix(metal2vulkan_devtools::fixtures_dir());
            let file = file.unwrap_or(&path).display().to_string();
            match llvm_assemble(&path).map(|bitcode| Library::from_air(&bitcode)) {
                Ok(Ok(library)) => libraries.push((file, library)),
                Ok(Err(error)) => unread.push(format!("{file}: {error}")),
                Err(AssembleError::Missing(why)) => {
                    eprintln!("SKIPPED: {why}");
                    return;
                }
                Err(error) => unread.push(format!("{file}: {error:?}")),
            }
        }
        each_device("compute_variant_answers", |name, _, device| {
            let translator = device.translator();
            let budget = Budget::standard();
            let entries = StageEntries::new();
            let maximal = translator.profile().subgroups().maximal_reconvergence;
            let (mut compared, mut skipped, mut alone) = (0, unread.clone(), Vec::new());
            let mut noted = 0;
            for (file, library) in &libraries {
                for info in library.functions() {
                    if info.stage != FunctionStage::Kernel {
                        continue;
                    }
                    let what = format!("{file}: {}", info.name);
                    let function = library.function(info.name).expect("a listed function");
                    let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                        FunctionDescriptor::new(function),
                    ));
                    let StageSource::Function(function) = &descriptor.kernel else {
                        unreachable!("a Function descriptor");
                    };
                    let linked = |inputs: &LoweringInputs| -> Result<(ComputePipeline, Option<bool>), String> {
                        let lowered = translator
                            .compile_function_from_entries(function, inputs, &budget, &entries)
                            .map_err(|error| error.to_string())?;
                        let imageblock =
                            crate::pipeline::declares_imageblock(&descriptor, &lowered, &budget);
                        let mut linked = descriptor.clone();
                        linked.kernel = StageSource::Compiled(lowered);
                        let pipeline = (translator.compile_compute_pipeline(&linked, &budget))
                            .map_err(|error| error.to_string())?;
                        Ok((pipeline, imageblock))
                    };
                    let modules: Result<Option<_>, String> = (|| {
                        let plain = translator
                            .kernel_inputs_read(
                                function,
                                &descriptor.lowering_inputs(),
                                &budget,
                                &entries,
                            )
                            .map_err(|error| format!("inputs: {error}"))?;
                        let partial = plain.kernel_variant(KernelVariant::PartialThreadgroups);
                        let observes = translator
                            .kernel_observes_partition(function, &partial, &budget, &entries)
                            .map_err(|error| format!("partition: {error}"))?;
                        if !observes {
                            return Ok(None);
                        }
                        Ok(Some((linked(&plain), linked(&partial))))
                    })();
                    let (plain, variant) = match modules {
                        Ok(Some((Ok(plain), Ok(variant)))) => (plain, variant),
                        Ok(None) => continue,
                        Ok(Some((Err(plain), Ok(_)))) => {
                            alone.push(format!("{what}: plain: {plain}"));
                            continue;
                        }
                        Ok(Some((Ok(_), Err(variant)))) => {
                            skipped.push(format!("{what}: variant: {variant}"));
                            continue;
                        }
                        Ok(Some((Err(plain), Err(_)))) => {
                            skipped.push(format!("{what}: plain: {plain}"));
                            continue;
                        }
                        Err(why) => {
                            skipped.push(format!("{what}: {why}"));
                            continue;
                        }
                    };
                    let of = |(pipeline, imageblock): &(ComputePipeline, Option<bool>)| {
                        ModuleAnswers::of(pipeline, *imageblock, translator.profile().push_window())
                    };
                    match (of(&plain), of(&variant)) {
                        (Ok(plain), Ok(mut variant)) => {
                            // The one exception (the doc comment's): without maximal
                            // reconvergence the variant may add the device's reconvergence.
                            let note = PrecisionNote::DeviceReconvergence;
                            if !maximal
                                && !plain.notes.contains(&note)
                                && variant.notes.contains(&note)
                            {
                                variant.notes.retain(|n| *n != note);
                                noted += 1;
                            }
                            assert_eq!(variant, plain, "{name}: {what}: the variant's answers");
                            compared += 1;
                        }
                        (plain, variant) => skipped.push(format!(
                            "{what}: unsupplied: plain {:?}, variant {:?}",
                            plain.err(),
                            variant.err()
                        )),
                    }
                }
            }
            for why in &skipped {
                println!("{name}: not compared: {why}");
            }
            for why in &alone {
                println!("{name}: the variant alone lowers: {why}");
            }
            println!(
                "{name}: {compared} observing kernels answer the same on their variant as on \
                 their plain module, {noted} of them but for the variant's DeviceReconvergence \
                 (maximal reconvergence {maximal}); {} whose plain module alone the translator \
                 refuses; {} not compared",
                alone.len(),
                skipped.len()
            );
            assert!(compared > 0, "{name}: no observing kernel compared");
        });
    }

    /// Each module a kernel's state lowers is the translator's public lowering of its variant on
    /// the device's translator (`Device::translator`), under `LoweringInputs::kernel_variant` of
    /// the inputs `kernel_inputs_read` gives: for the observing kernel, whose plain module checks
    /// its one store at a computed index against its buffer (`PushField::SlotBounds` of buffer 1)
    /// and leaves its two loads at constant offsets unchecked (none of buffer 0), and for a kernel
    /// of SIMD-group products. Each lowering a first device kept in its cache directory is found
    /// under the key of one variant's inputs, its artifact the public compile's byte for byte, and
    /// as many are found as the states made modules.
    #[test]
    fn every_device_s_kernel_modules_are_the_translator_s_lowerings_of_their_variants() {
        use metal2vulkan::pipeline::{PushField, SlotClass, SlotRef};

        let (Some(observing), Some(products)) = (
            descriptor("icb_replay_grid_template"),
            descriptor_of("gsm2_simd_products", "simd_float_products"),
        ) else {
            return;
        };
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let dir = std::env::temp_dir().join(format!("m2v-rt-variants-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let first = on_directory(name, physical, &dir);
            let counters = first.counters();
            let states = [
                state(name, &first, &observing),
                state(name, &first, &products),
            ];
            let made: usize = states.iter().map(modules_made).sum();
            drop(states);
            drop(first);
            // The teardown ends once the reaper wrote what it was given.
            within("the teardown", || counters.gauges() == Gauges::default());
            let second = on_directory(name, physical, &dir);
            let translator = second.translator();
            let budget = Budget::standard();
            let entries = StageEntries::new();
            let mut found = 0;
            for descriptor in [&observing, &products] {
                let StageSource::Function(function) = &descriptor.kernel else {
                    unreachable!("a fixture's kernel is a function");
                };
                let plain = translator
                    .kernel_inputs_read(function, &descriptor.lowering_inputs(), &budget, &entries)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                for variant in KernelVariant::ALL {
                    let inputs = plain.kernel_variant(variant);
                    let key = translator.function_compile_key(function, &inputs);
                    let Some(kept) = crate::pipeline::loaded(second.core(), key.digest()) else {
                        continue;
                    };
                    let public = translator
                        .compile_function(function, &inputs, &budget)
                        .unwrap_or_else(|error| panic!("{name}: {variant:?}: {error}"));
                    assert!(
                        kept.to_bytes() == public.to_bytes(),
                        "{name}: the state's {variant:?} module is the public lowering"
                    );
                    found += 1;
                }
            }
            assert!(
                found >= made,
                "{name}: {found} lowerings found of the {made} modules the states made"
            );
            let StageSource::Function(function) = &observing.kernel else {
                unreachable!("a fixture's kernel is a function");
            };
            let mut plain = observing.clone();
            plain.kernel = StageSource::Compiled(
                translator
                    .compile_function(function, &observing.lowering_inputs(), &budget)
                    .unwrap_or_else(|error| panic!("{name}: {error}")),
            );
            let pipeline = translator
                .compile_compute_pipeline(&plain, &budget)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let checked: Vec<u32> = (pipeline.interface.push.iter())
                .flat_map(|placement| &placement.fields)
                .filter_map(|placed| match placed.field {
                    PushField::SlotBounds(SlotRef {
                        class: SlotClass::Buffer,
                        index,
                        ..
                    }) => Some(index),
                    _ => None,
                })
                .collect();
            assert_eq!(
                checked,
                [1],
                "{name}: the buffers whose accesses are checked"
            );
            drop(second);
            let _ = std::fs::remove_dir_all(&dir);
            println!("{name}: {found} modules, each a variant's public lowering");
        }
    }

    /// The second process's part of
    /// `every_device_shares_the_cache_directory_with_a_second_process`: on the device and
    /// directory it names, the first process's kernel, then a kernel of its own.
    const SECOND: &str = "tests::translate::on_device::second_process_on_a_cache_directory";

    /// A second process on a cache directory: its device takes the kernel the first process's
    /// device left there and writes none of its lowerings again; the kernel of its own adds its
    /// lowering, and its `VkPipeline`'s data grows the blob, which the second process replaces,
    /// where the device keeps one.
    #[test]
    fn every_device_shares_the_cache_directory_with_a_second_process() {
        let Some(tiled) = descriptor("a1_bool_from_float") else {
            return;
        };
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let dir = std::env::temp_dir().join(format!("m2v-rt-cache2-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let device = on_directory(name, physical, &dir);
            let kept = usize::from(keeps_a_blob(name, &device));
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
            assert_eq!(
                (blob.len(), blob_after.len()),
                (kept, kept),
                "{name}: its blobs"
            );
            if let (Some(blob), Some(blob_after)) = (blob.first(), blob_after.first()) {
                assert!(blob_after.1 > blob.1, "{name}: the blob grew");
            }
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
        let instance = crate::tests::validation::instance(&[]);
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

    /// Makes the state of `descriptor`, a kernel whose state makes its `VkPipeline`, on `device`
    /// and waits for the reaper's look at the pipeline cache once creation is quiet after it.
    fn state_and_look(
        name: &str,
        device: &Device,
        descriptor: &ComputePipelineDescriptor,
    ) -> ComputePipelineState {
        let caches = &device.core().caches;
        let looks = caches.looks_and_writes().0;
        let pipelines = device.counters().metrics().pipelines_created;
        let made = state(name, device, descriptor);
        assert!(
            device.counters().metrics().pipelines_created > pipelines,
            "{name}: the state made its VkPipeline"
        );
        within("the reaper's look", || caches.looks_and_writes().0 > looks);
        made
    }

    /// The pipeline cache filled past its bound: at the look after a creation takes it past, the
    /// reaper replaces it with a fresh one, which the gauge falls to, keeps the old one while it is
    /// held, and destroys it at the first look after it is let go; creations go on through the
    /// fresh cache, and the next write replaces the blob with the fresh cache's data alone. The
    /// replacement counts once in `Metrics::pipeline_caches_replaced`. Then a creation grows the
    /// cache while another holds creation open, so no look comes before the teardown, and with
    /// the bound lowered past the cache the teardown's write is skipped, counted once in
    /// `Metrics::pipeline_cache_writes_skipped`, and the blob keeps the fresh cache's data.
    #[test]
    fn every_device_replaces_its_pipeline_cache_past_its_bound() {
        let (Some(first), Some(second), Some(third), Some(fourth)) = (
            descriptor("a1_bool_from_float"),
            descriptor("a1_sign_and_division"),
            descriptor("a1_sign_and_division_precise"),
            descriptor("ob1_storage_buffer"),
        ) else {
            return;
        };
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let dir = std::env::temp_dir().join(format!("m2v-rt-bound-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let device = on_directory(name, physical, &dir);
            // A cache whose data holds no header holds nothing to bound.
            if !keeps_a_blob(name, &device) {
                drop(device);
                let _ = std::fs::remove_dir_all(&dir);
                continue;
            }
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
            let replaced = counters.metrics().pipeline_caches_replaced;
            assert_eq!(replaced, 1, "{name}: one replacement counted");
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
            // Creation held open keeps the reaper from looking until the teardown stops it.
            let core = std::sync::Arc::clone(device.core());
            let blob = caches.blob().expect("a cache directory").to_path_buf();
            let open = crate::cache::creation(&core);
            let pipelines = counters.metrics().pipelines_created;
            let d = state(name, &device, &fourth);
            assert!(
                counters.metrics().pipelines_created > pipelines,
                "{name}: the fourth state made its VkPipeline"
            );
            caches.set_bound(1);
            drop((a, b, c, d));
            drop(device);
            drop(open);
            assert_eq!(
                counters.metrics().pipeline_cache_writes_skipped,
                0,
                "{name}"
            );
            drop(core);
            let skipped = counters.metrics().pipeline_cache_writes_skipped;
            assert_eq!(skipped, 1, "{name}: the teardown's write skipped");
            let kept = std::fs::metadata(&blob).map_or(0, |meta| meta.len());
            assert_eq!(kept, 24 + now, "{name}: the blob keeps the fresh cache");
            within("the teardown", || counters.gauges() == Gauges::default());
            let _ = std::fs::remove_dir_all(&dir);
            println!(
                "{name}: {full} bytes past a bound of {}, replaced by {fresh}, grown to {now}; \
                 replacements {replaced}, writes skipped at the teardown {skipped}",
                full - 1
            );
        }
    }

    /// A state's lowerings go to the store on the reaper: with the reaper's writes held, the
    /// creation returns and nothing is written. Once the queue is full, the next state's lowerings
    /// are dropped and counted, never waited for. Let go, the reaper writes what was queued: a
    /// device after it reads the first kernel's lowering and lowers the second again.
    #[test]
    fn every_device_writes_its_lowerings_off_the_creating_thread_and_drops_them_past_its_queue() {
        let (Some(first), Some(second)) = (
            descriptor("a1_sign_and_division"),
            descriptor("a1_sign_and_division_precise"),
        ) else {
            return;
        };
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let dir = std::env::temp_dir().join(format!("m2v-rt-pending-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let device = on_directory(name, physical, &dir);
            let caches = &device.core().caches;
            let counters = device.counters();
            caches.stall_writes(true);
            let a = state(name, &device, &first);
            let metrics = counters.metrics();
            assert!(metrics.translations > 0, "{name}: the first kernel lowered");
            assert_eq!(
                metrics.cache_bytes_written, 0,
                "{name}: the creation returned with nothing written"
            );
            // Full: the bound is what waits now.
            let waiting = caches.set_pending_bound(0);
            caches.set_pending_bound(waiting);
            let b = state(name, &device, &second);
            assert!(
                counters.metrics().cache_writes_dropped > 0,
                "{name}: the second kernel's lowering dropped and counted"
            );
            caches.set_pending_bound(crate::cache::PENDING_BYTES);
            caches.stall_writes(false);
            device.core().reaper.poke();
            within("the writes", || counters.metrics().cache_bytes_written > 0);
            drop((a, b));
            drop(device);
            within("the teardown", || counters.gauges() == Gauges::default());
            let next = on_directory(name, physical, &dir);
            let counters = next.counters();
            let a = state(name, &next, &first);
            assert!(
                counters.metrics().translations_loaded > 0,
                "{name}: the first kernel's lowering was written"
            );
            let lowered = counters.metrics().translations;
            let b = state(name, &next, &second);
            assert!(
                counters.metrics().translations > lowered,
                "{name}: the second kernel's, dropped, is lowered again"
            );
            drop((a, b));
            drop(next);
            within("the teardown", || counters.gauges() == Gauges::default());
            let _ = std::fs::remove_dir_all(&dir);
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
        let instance = crate::tests::validation::instance(&[]);
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
                if !keeps_a_blob(name, &device) {
                    drop(device);
                    let _ = std::fs::remove_dir_all(&dir);
                    break;
                }
                let blob = device.core().caches.blob().expect("a blob").to_owned();
                drop(device);
                assert!(!blob.exists(), "{name}: no creation, no blob");
                std::fs::write(&blob, &bytes).expect("a blob");
                let device = on_directory(name, physical, &dir);
                within("the pipeline cache", || device.core().caches.is_seeded());
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

    /// A device keeps a blob only where its driver's data of an empty cache begins with a header
    /// (`cache::usable_header`): then it names one and its drop writes it, and a device after it
    /// on the directory takes it. Else (lavapipe's 20 bytes) it names none, and neither it nor a
    /// device after it reads or writes one; its creations go through its cache all the same.
    #[test]
    fn every_device_keeps_a_blob_only_where_its_cache_data_begins_with_a_header() {
        let Some(tiled) = descriptor("a1_bool_from_float") else {
            return;
        };
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let dir = std::env::temp_dir().join(format!("m2v-rt-header-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            for round in 0..2 {
                let device = on_directory(name, physical, &dir);
                let caches = &device.core().caches;
                within("the seed", || caches.is_seeded());
                // SAFETY: the caches are the device's.
                let data = unsafe { caches.data(&device.core().device) }.expect("the cache's data");
                let header = crate::cache::usable_header(&data);
                assert_eq!(
                    caches.blob().is_some(),
                    header.is_some(),
                    "{name}: device {round}, {} bytes of data",
                    data.len()
                );
                let counters = device.counters();
                drop(state(name, &device, &tiled));
                assert_eq!(
                    counters.metrics().pipelines_created,
                    1,
                    "{name}: device {round}"
                );
                drop(device);
                within("the teardown", || counters.gauges() == Gauges::default());
                let kept = blobs(&files(&dir)).len();
                assert_eq!(
                    kept,
                    usize::from(header.is_some()),
                    "{name}: device {round}'s blobs"
                );
                println!(
                    "{name}: device {round}, {} bytes of an empty cache's data, {kept} blob",
                    data.len()
                );
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// A creation while the seed reads and takes in the blob goes through the empty cache and
    /// never waits for it; it counts as a creation before the seed, and once it has ended the
    /// empty cache, retired at the swap, is destroyed, with no creation holding the lock creations
    /// take during a look.
    #[test]
    fn every_device_creates_through_the_empty_cache_while_its_seed_takes_the_blob() {
        let (Some(first), Some(second)) = (
            descriptor("a1_bool_from_float"),
            descriptor("a1_sign_and_division_precise"),
        ) else {
            return;
        };
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let dir = std::env::temp_dir().join(format!("m2v-rt-seed-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let device = on_directory(name, physical, &dir);
            // With no blob there is no seed to take.
            if !keeps_a_blob(name, &device) {
                drop(device);
                let _ = std::fs::remove_dir_all(&dir);
                continue;
            }
            let blob = device.core().caches.blob().expect("a blob").to_owned();
            drop(state(name, &device, &first));
            drop(device);
            assert!(blob.exists(), "{name}: the drop wrote the blob");
            crate::cache::Caches::hold_next_seed();
            let device = on_directory(name, physical, &dir);
            let caches = &device.core().caches;
            within("the seed", || caches.seed_hold.waiting());
            let counters = device.counters();
            let before = counters.metrics().creations_before_seed;
            let (made, during) = std::thread::scope(|scope| {
                let (ended, end) = std::sync::mpsc::channel();
                let (device, second) = (&device, &second);
                let creating = scope.spawn(move || {
                    let made = state(name, device, second);
                    let _ = ended.send(());
                    made
                });
                // Let go either way, so a creation that waits for the seed ends.
                let during = end.recv_timeout(std::time::Duration::from_secs(30)).is_ok();
                caches.seed_hold.release();
                (creating.join().expect("the creation"), during)
            });
            assert!(during, "{name}: the creation waited for the seed");
            assert!(
                counters.metrics().creations_before_seed > before,
                "{name}: the creation counted before the seed"
            );
            within("the seed", || caches.is_seeded());
            within("the empty cache's destruction", || caches.retired() == 0);
            assert_eq!(caches.overlaps(), 0, "{name}: a creation during a look");
            drop(made);
            drop(device);
            within("the teardown", || counters.gauges() == Gauges::default());
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// The reaper's look at the pipeline cache holds the lock creations take exclusive around its
    /// reads of the cache's size and data, and takes it by a try. A creation that
    /// begins once the reaper chose to look, before its try, makes the look skip, counted
    /// (`Metrics::pipeline_cache_looks_skipped`), and the reaper looks again once creation is
    /// quiet again. A creation that begins while a look holds the lock waits for that look, and
    /// its wait is counted (`Metrics::pipeline_cache_lock_wait_max_nanos`). No creation holds the
    /// lock while a look does.
    #[test]
    fn every_device_reads_its_pipeline_cache_only_while_no_creation_runs() {
        /// How long the look holds the lock while a creation waits.
        const HELD: std::time::Duration = std::time::Duration::from_millis(100);
        let (Some(first), Some(second), Some(third)) = (
            descriptor("a1_bool_from_float"),
            descriptor("a1_sign_and_division"),
            descriptor("a1_sign_and_division_precise"),
        ) else {
            return;
        };
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let dir = std::env::temp_dir().join(format!("m2v-rt-alone-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let device = on_directory(name, physical, &dir);
            let caches = &device.core().caches;
            let counters = device.counters();
            let a = state_and_look(name, &device, &first);
            // A creation that begins once the reaper chose to look, before it tries the lock.
            let looks = caches.looks_and_writes().0;
            let skipped = counters.metrics().pipeline_cache_looks_skipped;
            caches.look_hold.hold();
            let b = state(name, &device, &second);
            within("the look", || caches.look_hold.waiting());
            let creation = crate::cache::creation(device.core());
            caches.look_hold.release();
            within("the look's try", || {
                counters.metrics().pipeline_cache_looks_skipped > skipped
                    || caches.looks_and_writes().0 > looks
            });
            assert_eq!(
                caches.looks_and_writes().0,
                looks,
                "{name}: the reaper looked while a creation ran"
            );
            assert_eq!(
                counters.metrics().pipeline_cache_looks_skipped,
                skipped + 1,
                "{name}: the look skipped once"
            );
            // The look after it, once creation is quiet again, holds the lock while a creation
            // begins.
            caches.alone_hold.hold();
            drop(creation);
            within("the look holding the lock", || caches.alone_hold.waiting());
            assert_eq!(caches.looks_and_writes().0, looks, "{name}: still looking");
            let (c, during) = std::thread::scope(|scope| {
                let (ended, end) = std::sync::mpsc::channel();
                let (device, third) = (&device, &third);
                let creating = scope.spawn(move || {
                    let made = state(name, device, third);
                    let _ = ended.send(());
                    made
                });
                within("the creation's start", || {
                    caches.creating() > 0 || creating.is_finished()
                });
                std::thread::sleep(HELD);
                let during = end.try_recv().is_ok();
                caches.alone_hold.release();
                (creating.join().expect("the creation"), during)
            });
            assert!(
                !during,
                "{name}: the creation ran while the look held the lock"
            );
            within("the look's end", || caches.looks_and_writes().0 > looks);
            let metrics = counters.metrics();
            let held = HELD.as_nanos() as u64;
            assert!(
                metrics.pipeline_cache_lock_wait_max_nanos >= held / 2,
                "{name}: the creation's longest wait {} ns, the look held the lock {held} ns",
                metrics.pipeline_cache_lock_wait_max_nanos
            );
            assert!(
                metrics.pipeline_cache_lock_wait_nanos
                    >= metrics.pipeline_cache_lock_wait_max_nanos
            );
            assert!(
                metrics.pipeline_cache_lock_waits >= 1,
                "{name}: the creation's wait counted"
            );
            let overlaps = caches.overlaps();
            assert_eq!(overlaps, 0, "{name}: a creation during a look");
            drop((a, b, c));
            drop(device);
            within("the teardown", || counters.gauges() == Gauges::default());
            let _ = std::fs::remove_dir_all(&dir);
            println!(
                "{name}: {} look skipped, {} creations waited for a look, the longest {} ns, \
                 {overlaps} overlaps",
                metrics.pipeline_cache_looks_skipped - skipped,
                metrics.pipeline_cache_lock_waits,
                metrics.pipeline_cache_lock_wait_max_nanos,
            );
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

    /// A kernel whose one texture `t`, which the boolean constant 0 gates, it reads at (0, 0) on
    /// its one path, storing the texel and `t`'s width into words 0 to 4 of `out`.
    const ABSENT_READ: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@has_t.MTL_FC_INIT_0_b = linkonce_odr hidden local_unnamed_addr addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@__metal_implicit_fc_pred_0 = internal addrspace(2) global i8 0, align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_, ptr null }]

define void @absent_read_kernel(ptr addrspace(1) %t, ptr addrspace(1) %out) {
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  store <4 x float> %v, ptr addrspace(1) %out, align 4
  %w = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 0)
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 4
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
declare i8 @air.normalize_function_constant_predicate.i8(i8)

define internal void @_GLOBAL__sub_I_() section "air.static_init" {
  %1 = load i8, ptr addrspace(2) @has_t.MTL_FC_INIT_0_b, align 1
  %2 = tail call i8 @air.normalize_function_constant_predicate.i8(i8 %1)
  store i8 %2, ptr addrspace(2) @__metal_implicit_fc_pred_0, align 1
  ret void
}

!air.kernel = !{!0}
!air.function_constants = !{!10}
!air.version = !{!20}
!air.language_version = !{!21}

!0 = !{ptr @absent_read_kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.function_constant", !5, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{ptr addrspace(2) @__metal_implicit_fc_pred_0, !"bool", !"has_t"}
!10 = !{ptr addrspace(2) @has_t.MTL_FC_INIT_0_b, !"bool", !"has_t", i32 0}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

    /// A kernel of 64 threads whose thread `t` stores word `t` of buffer argument `in` (index 0)
    /// at word `t` of `out` (index 1), and word `16320 + t`, one of `in`'s 64 KiB's last 64, at
    /// word `64 + t`.
    const NIL_READ: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @nil_read_kernel(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %t) {
  %p = getelementptr inbounds i32, ptr addrspace(1) %in, i32 %t
  %x = load i32, ptr addrspace(1) %p, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %x, ptr addrspace(1) %o, align 4
  %t2 = add i32 %t, 16320
  %q = getelementptr inbounds i32, ptr addrspace(1) %in, i32 %t2
  %y = load i32, ptr addrspace(1) %q, align 4
  %t3 = add i32 %t, 64
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t3
  store i32 %y, ptr addrspace(1) %o2, align 4
  ret void
}

!air.kernel = !{!0}
!air.version = !{!20}
!air.language_version = !{!21}

!0 = !{ptr @nil_read_kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

    /// A read through a buffer argument bound to nothing reads the nil buffer, as Metal's nil
    /// buffer reads zeros: [`NIL_READ`]'s 128 words of `out`, which held 0xFFFF_FFFF, land as
    /// zeros, its first 64 words and its last 64 alike, and the command buffer completes with no
    /// error.
    #[test]
    fn every_device_reads_zeros_through_a_buffer_argument_bound_to_nothing() {
        let bitcode = match metal2vulkan_devtools::llvm_assemble_text(NIL_READ) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("nil_read_kernel: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function =
            FunctionDescriptor::new(library.function("nil_read_kernel").expect("the kernel"));
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(function));
        each_device("compute_nil_read", |name, _, device| {
            let state = state(name, &device, &descriptor);
            let mut buffers = Buffers::new(&device, 4096);
            let out = buffers.buffer(128 * 4);
            write(&out, 0, &le_bytes(&[u32::MAX; 128]));
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            encoder.set_buffer(Some(&out), 0, 1).expect("a buffer");
            encoder
                .dispatch_threadgroups([1, 1, 1], [64, 1, 1])
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            assert_eq!(
                words(&out, 0, 128),
                [0; 128],
                "{name}: the nil buffer's words"
            );
            println!("{name}: a buffer argument bound to nothing read zeros");
        });
    }

    /// `engine/kernel_stage_in`'s `stage_in_region_short`, which reads the stage-in region
    /// (`[[grid_origin]]` and `[[grid_size]]` as `ushort2`s, with no `[[stage_in]]`), stores at
    /// each thread `(origin.xy, size.xy)` of the region the encoder set before its dispatch: two
    /// dispatches under two regions store each its own, and nothing past the grid. A dispatch of
    /// it before any region is set is `Gap::NoStageInRegion`.
    #[test]
    fn every_device_pushes_the_stage_in_region_a_kernel_reads() {
        let Some(descriptor) = descriptor_of("kernel_stage_in", "stage_in_region_short") else {
            return;
        };
        const THREADS: u32 = 8;
        let regions = [
            Region {
                origin: [3, 5, 7],
                size: [11, 13, 1],
            },
            Region {
                origin: [17, 0, 2],
                size: [1, 19, 4],
            },
        ];
        each_device("compute_stage_in_region", |name, _, device| {
            let state = state(name, &device, &descriptor);
            let mut buffers = Buffers::new(&device, 4096);
            // An element, 8 bytes, a thread, and a spare one past the grid.
            let outs: Vec<Buffer> = (0..2)
                .map(|_| buffers.buffer(8 * u64::from(THREADS + 1)))
                .collect();
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            encoder.set_buffer(Some(&outs[0]), 0, 0).expect("a buffer");
            match encoder.dispatch_threads([THREADS, 1, 1], [4, 1, 1]) {
                Err(Error::Unsupported(Gap::NoStageInRegion)) => {}
                other => panic!("{name}: a dispatch before any region: {other:?}"),
            }
            for (out, &region) in outs.iter().zip(&regions) {
                encoder.set_buffer(Some(out), 0, 0).expect("a buffer");
                encoder.set_stage_in_region(region);
                encoder
                    .dispatch_threads([THREADS, 1, 1], [4, 1, 1])
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            }
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            for (out, region) in outs.iter().zip(&regions) {
                let pair = |a: u32, b: u32| a | b << 16;
                let element = [
                    pair(region.origin[0], region.origin[1]),
                    pair(region.size[0], region.size[1]),
                ];
                let mut expected: Vec<u32> = (0..THREADS).flat_map(|_| element).collect();
                expected.extend([0, 0]);
                assert_eq!(
                    words(out, 0, 2 * (THREADS as usize + 1)),
                    expected,
                    "{name}: {region:?}"
                );
            }
            println!("{name}: each dispatch read the stage-in region set before it");
        });
    }

    /// The words of element `e` of `every_device_fetches_stage_in_by_each_step_function`'s source
    /// buffer, under its four attributes: `float4` at 0, `uint4` at 16, `int` at 32 and `half2` at
    /// 36, 40 bytes an element. None is a NaN or a denormal.
    fn stage_in_element(e: u32) -> [u32; 10] {
        let f = [e as f32, e as f32 + 0.5, 1.0, -2.0].map(f32::to_bits);
        let u = [e, 2 * e, e + 100, 0xdead_0000 | e];
        let i = (-(e as i32) - 1) as u32;
        let h = (0x3c00 + e) | (0x4000 + e) << 16;
        [f[0], f[1], f[2], f[3], u[0], u[1], u[2], u[3], i, h]
    }

    /// `plain` (`engine/kernel_stage_in`'s `stage_in_fetch`) under a stage-input descriptor of its
    /// four attributes, `Float4` at 0, `UInt4` at 16, `Int` at 32 and `Half2` at 36 of the buffer
    /// at index 1, whose layout has `stride` (`None` for `MTLBufferLayoutStrideDynamic`), `step` and
    /// `rate`; 16-bit indices at index 2.
    fn stage_in_fetch(
        plain: &ComputePipelineDescriptor,
        stride: Option<u32>,
        step: StepFunction,
        rate: u32,
    ) -> ComputePipelineDescriptor {
        let mut input = StageInputDescriptor::new();
        let mut layout = StageInputLayoutDescriptor::new(stride);
        layout.step_function = step;
        layout.step_rate = rate;
        input.layouts.insert(1, layout);
        input.index_type = IndexType::UInt16;
        input.index_buffer_index = 2;
        let attributes = [
            (AttributeFormat::Float4, 0),
            (AttributeFormat::UInt4, 16),
            (AttributeFormat::Int, 32),
            (AttributeFormat::Half2, 36),
        ];
        for (index, (format, offset)) in (0..).zip(attributes) {
            let attribute = StageInputAttributeDescriptor::new(format, offset, 1);
            input.attributes.insert(index, attribute);
        }
        let mut descriptor = plain.clone();
        descriptor.stage_input = Some(input);
        descriptor
    }

    /// `engine/kernel_stage_in`'s `stage_in_fetch` under [`stage_in_fetch`]'s descriptor, its
    /// layout of a 40-byte element and the buffer at index 1 bound at an offset of 16, once per
    /// step function a kernel may use: each thread `x`
    /// of a grid of 8 reports the bits of element `origin + min(x / rate, size - 1)` of a
    /// thread-position step, of element `index[min(x, size - 1)] + origin` of an indexed one (16-bit
    /// indices at index 2), and of element 0 of a constant step, so threads past the region read
    /// its last element. Before the runtime fetched stage inputs the state was
    /// `Gap::KernelInput`.
    #[test]
    fn every_device_fetches_stage_in_by_each_step_function() {
        let Some(plain) = descriptor_of("kernel_stage_in", "stage_in_fetch") else {
            return;
        };
        const THREADS: u32 = 8;
        const ELEMENTS: u32 = 16;
        const OFFSET: u64 = 16;
        const INDICES: [u16; 8] = [5, 3, 0, 7, 2, 6, 1, 4];
        // Each step, its rate, the region's origin and size on x, and the element each thread
        // reads.
        let element = |x: u32, step: StepFunction, rate: u32, origin: u32, size: u32| match step {
            StepFunction::Constant => 0,
            StepFunction::ThreadPositionInGridXIndexed => {
                u32::from(INDICES[x.min(size - 1) as usize]) + origin
            }
            _ => origin + (x / rate).min(size - 1),
        };
        let steps = [
            (StepFunction::ThreadPositionInGridX, 1, 1, 5),
            (StepFunction::ThreadPositionInGridX, 2, 0, 3),
            (StepFunction::ThreadPositionInGridXIndexed, 1, 2, 6),
            (StepFunction::Constant, 0, 3, 8),
        ];
        each_device("compute_stage_in_steps", |name, _, device| {
            let mut buffers = Buffers::new(&device, 8192);
            let source = buffers.buffer(OFFSET + 40 * u64::from(ELEMENTS));
            let bytes: Vec<u8> = (0..ELEMENTS)
                .flat_map(stage_in_element)
                .flat_map(u32::to_le_bytes)
                .collect();
            write(&source, OFFSET, &bytes);
            let indices = buffers.buffer(2 * INDICES.len() as u64);
            write(&indices, 0, &INDICES.map(u16::to_le_bytes).concat());
            let queue = device.new_command_queue(4).expect("a queue");
            for (step, rate, origin, size) in steps {
                let descriptor = stage_in_fetch(&plain, Some(40), step, rate);
                let label = format!("{name}: {step:?} at rate {rate}");
                let state = state(&label, &device, &descriptor);
                // Three `uint4`s a thread, and a spare thread's past the grid.
                let out = buffers.buffer(48 * u64::from(THREADS + 1));
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                encoder.set_buffer(Some(&out), 0, 0).expect("a buffer");
                encoder
                    .set_buffer(Some(&source), OFFSET, 1)
                    .expect("a buffer");
                encoder.set_buffer(Some(&indices), 0, 2).expect("a buffer");
                encoder.set_stage_in_region(Region {
                    origin: [origin, 0, 0],
                    size: [size, 1, 1],
                });
                encoder
                    .dispatch_threads([THREADS, 1, 1], [4, 1, 1])
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(&label, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{label}");
                // Per thread: the `float4`'s bits, the `uint4`, then the `int`'s bits, each `half`'s
                // and the thread's index.
                let mut expected: Vec<u32> = (0..THREADS)
                    .flat_map(|x| {
                        let w = stage_in_element(element(x, step, rate, origin, size));
                        let [front @ .., i, h] = w;
                        front.into_iter().chain([i, h & 0xffff, h >> 16, x])
                    })
                    .collect();
                expected.extend([0; 12]);
                assert_eq!(
                    words(&out, 0, 12 * (THREADS as usize + 1)),
                    expected,
                    "{label}: the attributes each thread fetched"
                );
                println!("{label}: each thread fetched the element its step names");
            }
        });
    }

    /// The forms of a kernel's `[[stage_in]]` the runtime does not build are each refused as
    /// `Gap::StageIn` of its own kind: a thread-position layout of `MTLBufferLayoutStrideDynamic`
    /// (`DynamicStride`) and an indexed step at a step rate of 2 (`IndexedStepRate`) at the state's
    /// creation, and `setStageInRegionWithIndirectBuffer:indirectBufferOffset:`
    /// (`IndirectRegion`) at its call. A constant step of a dynamic layout reads no stride, and an
    /// attribute the kernel does not read (a fifth, at buffer 3, of a dynamic layout) is fetched by
    /// nothing: both states are made.
    #[test]
    fn every_device_refuses_each_unbuilt_stage_in_form_by_its_kind() {
        let Some(plain) = descriptor_of("kernel_stage_in", "stage_in_fetch") else {
            return;
        };
        let refused = |made: Result<ComputePipelineState, Error>, kind: StageInKind| matches!(made, Err(Error::Unsupported(Gap::StageIn { kind: found })) if found == kind);
        let dynamic = stage_in_fetch(&plain, None, StepFunction::ThreadPositionInGridX, 1);
        let indexed = stage_in_fetch(
            &plain,
            Some(40),
            StepFunction::ThreadPositionInGridXIndexed,
            2,
        );
        let constant = stage_in_fetch(&plain, None, StepFunction::Constant, 0);
        let mut unread = stage_in_fetch(&plain, Some(40), StepFunction::ThreadPositionInGridX, 1);
        if let Some(input) = &mut unread.stage_input {
            let mut layout = StageInputLayoutDescriptor::new(None);
            layout.step_function = StepFunction::ThreadPositionInGridX;
            input.layouts.insert(3, layout);
            let attribute = StageInputAttributeDescriptor::new(AttributeFormat::Float, 0, 3);
            input.attributes.insert(4, attribute);
        }
        each_device("compute_stage_in_unbuilt", |name, _, device| {
            let made = |descriptor: &ComputePipelineDescriptor| {
                device.new_compute_pipeline_state(descriptor, crate::tests::no_constexpr_samplers)
            };
            for (descriptor, kind) in [
                (&dynamic, StageInKind::DynamicStride),
                (&indexed, StageInKind::IndexedStepRate),
            ] {
                let answer = made(descriptor);
                assert!(
                    refused(answer, kind),
                    "{name}: not Gap::StageIn of {kind:?}"
                );
            }
            for (descriptor, what) in [
                (&constant, "a constant step of a dynamic layout"),
                (&unread, "an unread attribute of a dynamic layout"),
            ] {
                made(descriptor).unwrap_or_else(|error| panic!("{name}: {what}: {error}"));
            }
            let mut buffers = Buffers::new(&device, 4096);
            let region = buffers.buffer(24);
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            let indirect = encoder.set_stage_in_region_with_indirect_buffer(&region, 0);
            assert!(
                matches!(
                    indirect,
                    Err(Error::Unsupported(Gap::StageIn {
                        kind: StageInKind::IndirectRegion
                    }))
                ),
                "{name}: the indirect region: {indirect:?}"
            );
            encoder.end_encoding();
            println!("{name}: each unbuilt stage-in form was refused by its kind");
        });
    }

    /// A `dispatchThreads` grid of 25 threads in threadgroups of 1,024, one partial threadgroup of
    /// 25, of each kernel that fetches a `[[stage_in]]` attribute by thread position, under a
    /// stage-in region at origin 7 and 1,024 elements long, over a source buffer that ends at the
    /// last thread's element, 7 + 24: `engine/kernel_stage_in`'s `stage_in_fetch` under
    /// [`stage_in_fetch`]'s descriptor, which does not observe its threadgroup partition and so
    /// runs its grid in tiles, and `engine/kernel_stage_in_groups`' `stage_in_neighbour` (one
    /// `uint` at index 1, 4 bytes an element), which observes it and holds a threadgroup barrier,
    /// so its partial threadgroup walks the body predicated and adds one to
    /// `Metrics::partial_variant_ragged_grids`. Each of the 25 threads reports what Metal's does
    /// (`stage_in_neighbour`: its threadgroup's next thread's word, 25 threads, its position and
    /// index), and every word past the grid keeps its fill, so no invocation past the threadgroup
    /// writes. A fetch past the grid would read past the buffer, which the fetch's check reads as
    /// zero and no output shows: the translator's interpreter test of the same layout traps on it.
    /// Before the partial-threadgroups variant took a stage-input region, each dispatch was
    /// `Gap::PartialThreadgroups` at the translator's site.
    #[test]
    fn every_device_runs_a_stage_in_kernels_partial_threadgroup() {
        let (Some(plain), Some(walked)) = (
            descriptor_of("kernel_stage_in", "stage_in_fetch"),
            descriptor_of("kernel_stage_in_groups", "stage_in_neighbour"),
        ) else {
            return;
        };
        const THREADS: u32 = 25;
        const ORIGIN: u32 = 7;
        const FILL: u32 = 0xEEEE_EEEE;
        let tiled = stage_in_fetch(&plain, Some(40), StepFunction::ThreadPositionInGridX, 1);
        let mut walked = walked;
        let mut input = StageInputDescriptor::new();
        let mut layout = StageInputLayoutDescriptor::new(Some(4));
        layout.step_function = StepFunction::ThreadPositionInGridX;
        input.layouts.insert(1, layout);
        let attribute = StageInputAttributeDescriptor::new(AttributeFormat::UInt, 0, 1);
        input.attributes.insert(0, attribute);
        walked.stage_input = Some(input);
        let word = |e: u32| 0x5150_0000 + 3 * e;
        // Elements 0 to 7 + 24, the last thread's.
        let elements = 0..ORIGIN + THREADS;
        let fetched: Vec<u32> = elements.clone().flat_map(stage_in_element).collect();
        let fetched = le_bytes(&fetched);
        let neighbours = le_bytes(&elements.map(word).collect::<Vec<u32>>());
        // Thread `x`'s words: `stage_in_fetch`'s twelve, `stage_in_neighbour`'s four.
        let fetch_words = |x: u32| -> Vec<u32> {
            let [front @ .., i, h] = stage_in_element(ORIGIN + x);
            front
                .into_iter()
                .chain([i, h & 0xffff, h >> 16, x])
                .collect()
        };
        let neighbour_words = |x: u32| vec![word(ORIGIN + (x + 1) % THREADS), THREADS, x, x];
        type Words<'a> = &'a dyn Fn(u32) -> Vec<u32>;
        // Each kernel, its descriptor, its source's bytes, its words a thread, and thread `x`'s.
        let kernels: [(&str, &ComputePipelineDescriptor, &[u8], usize, Words<'_>); 2] = [
            ("stage_in_fetch", &tiled, &fetched, 12, &fetch_words),
            (
                "stage_in_neighbour",
                &walked,
                &neighbours,
                4,
                &neighbour_words,
            ),
        ];
        each_device("compute_stage_in_partial_threadgroup", |name, _, device| {
            let queue = device.new_command_queue(4).expect("a queue");
            let mut buffers = Buffers::new(&device, 1 << 17);
            let mut failed = Vec::new();
            for (kernel, descriptor, bytes, per, expected) in kernels {
                let label = format!("{name}: {kernel}");
                let state = state(&label, &device, descriptor);
                let source = buffers.buffer(bytes.len() as u64);
                write(&source, 0, bytes);
                // A thread's words for every invocation of the threadgroup, and one past it.
                let count = per * 1025;
                let out = buffers.buffer(4 * count as u64);
                write(&out, 0, &le_bytes(&vec![FILL; count]));
                let before = device.counters().metrics();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                encoder.set_buffer(Some(&out), 0, 0).expect("a buffer");
                encoder.set_buffer(Some(&source), 0, 1).expect("a buffer");
                encoder.set_stage_in_region(Region {
                    origin: [ORIGIN, 0, 0],
                    size: [1024, 1, 1],
                });
                let dispatched = encoder.dispatch_threads([THREADS, 1, 1], [1024, 1, 1]);
                encoder.end_encoding();
                if let Err(error) = dispatched {
                    println!("{label}: the dispatch: {error:?}");
                    failed.push(label);
                    continue;
                }
                command_buffer.commit().expect("a commit");
                finished(&label, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{label}");
                let mut want: Vec<u32> = (0..THREADS).flat_map(expected).collect();
                want.resize(count, FILL);
                assert_eq!(words(&out, 0, count), want, "{label}: the words written");
                let ragged = device.counters().metrics().partial_variant_ragged_grids
                    - before.partial_variant_ragged_grids;
                assert_eq!(
                    ragged,
                    u64::from(kernel == "stage_in_neighbour"),
                    "{label}: ragged grids on the partial-threadgroups variant"
                );
                println!("{label}: 25 threads of a threadgroup of 1,024 answered Metal's words");
            }
            assert!(failed.is_empty(), "refused: {failed:?}");
        });
    }

    /// The kernel `entry` of the harness fixture (`fixtures/public/harness/workloads.ll`, our
    /// own), or None where `llvm-as` is missing.
    fn harness_kernel(entry: &str) -> Option<ComputePipelineDescriptor> {
        let fixture = metal2vulkan_devtools::fixtures_dir().join("harness/workloads.ll");
        let bitcode = match llvm_assemble(&fixture) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return None;
            }
            Err(error) => panic!("{entry}: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function = library.function(entry).expect("the kernel");
        Some(ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        )))
    }

    /// The harness fixture's `row_cells`: each thread inside the grid's width and height reads
    /// `v = source[y * stride + x]` and writes `v`, `2 v` and `v + 1` to the words `3 cell` to
    /// `3 cell + 2` of `destination`, `cell = y * width + x`. The translator gives the kernel no
    /// lockstep order (the identity domains), so it does not observe its threadgroup partition:
    /// the state lowers its partial-threadgroups variant, which runs its grid in tiles, and makes
    /// its tiles' `VkPipeline` with it, one pipeline and no first-use wait. Every grid runs on that
    /// pipeline, no other made and no wait: 32 x 2, 64 x 8 and 512 x 1 over 256 x 24 threads, 64 x
    /// 4 over 256 x 23 and 48 x 1 over 200 x 23, the last three ragged, and 10 x 6 threadgroups of
    /// 16 x 4, whose SIMD groups span two rows. Every answer is the program's.
    #[test]
    fn every_device_runs_every_grid_of_the_cells_kernel_in_its_tiles() {
        let Some(descriptor) = harness_kernel("row_cells") else {
            return;
        };
        each_device("compute_row_cells", |name, _, device| {
            let made = device.counters().metrics();
            let state = state(name, &device, &descriptor);
            let created = device.counters().metrics();
            assert!(
                matches!(state.inner.compiled.form, crate::pipeline::Form::Tiled),
                "{name}: the kernel runs its grids in tiles"
            );
            assert_eq!(
                created.pipelines_created - made.pipelines_created,
                1,
                "{name}: the tiles' pipeline, made by finish_pipeline"
            );
            assert_eq!(
                created.first_use_waits, made.first_use_waits,
                "{name}: first-use waits"
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let mut buffers = Buffers::new(&device, 1 << 20);
            let cells: u32 = 256 * 24;
            let src = buffers.buffer(4 * u64::from(cells));
            let value = |cell: u32| (7 * cell + 3) as f32;
            let values: Vec<u32> = (0..cells).map(|cell| value(cell).to_bits()).collect();
            write(&src, 0, &le_bytes(&values));
            // Each dispatch: Metal's grid, its threadgroups, whether it is `dispatchThreads`.
            let run = |grid: [u32; 3], size: [u32; 3], threads: bool| {
                let width = match threads {
                    true => grid[0],
                    false => grid[0] * size[0],
                };
                let height = match threads {
                    true => grid[1],
                    false => grid[1] * size[1],
                };
                let mut buffers = Buffers::new(&device, 1 << 20);
                let dst = buffers.buffer(12 * u64::from(cells));
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                encoder.set_buffer(Some(&src), 0, 0).expect("a buffer");
                encoder.set_buffer(Some(&dst), 0, 1).expect("a buffer");
                for (index, word) in [(2, width), (3, height), (4, width)] {
                    (encoder.set_bytes(&le_bytes(&[word]), index)).expect("a uniform");
                }
                let dispatched = match threads {
                    true => encoder.dispatch_threads(grid, size),
                    false => encoder.dispatch_threadgroups(grid, size),
                };
                dispatched.unwrap_or_else(|error| panic!("{name}: {grid:?} in {size:?}: {error}"));
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
                let want: Vec<u32> = (0..width * height)
                    .flat_map(|cell| {
                        let v = value(cell);
                        [v, 2.0 * v, v + 1.0].map(f32::to_bits)
                    })
                    .collect();
                assert_eq!(
                    words(&dst, 0, want.len()),
                    want,
                    "{name}: {grid:?} in {size:?}"
                );
            };
            for size in [[32, 2, 1], [64, 8, 1], [512, 1, 1]] {
                run([256, 24, 1], size, true);
            }
            run([256, 23, 1], [64, 4, 1], true);
            run([200, 23, 1], [48, 1, 1], true);
            run([10, 6, 1], [16, 4, 1], false);
            let after = device.counters().metrics();
            assert_eq!(
                after.pipelines_created, created.pipelines_created,
                "{name}: every grid runs on the tiles' pipeline"
            );
            assert_eq!(
                after.first_use_waits, created.first_use_waits,
                "{name}: first-use waits"
            );
            println!(
                "{name}: row cells checked in 32 x 2, 64 x 8, 512 x 1, 64 x 4, 48 x 1 and 16 x 4 on \
                 the tiles' pipeline, made by finish_pipeline"
            );
        });
    }

    /// The harness fixture's `row_barrier` holds no threadgroup memory and observes its
    /// threadgroup partition through `threadgroup_barrier(mem_flags::mem_device)`: each thread
    /// stores `((y << 16) | x) + 1` to its cell, then, past the barrier, copies the word of its
    /// column in row `y ^ 1` where that row is in the grid. The state lowers both modules and makes
    /// no pipeline, and its grids run at their own size: 512 x 4 in 32 x 2 on the plain module,
    /// 512 x 3 in 32 x 2 and 96 x 5 in 32 x 4 on the partial-threadgroups variant, each its own
    /// pipeline made by a first-use wait, with no translation; no grid counts in
    /// `Metrics::partial_variant_whole_grids`. A state of the descriptor with
    /// `requiredThreadsPerThreadgroup` 32 x 2 makes, with the state, the pipelines a whole and a
    /// ragged grid run at that size, so its 512 x 4 and 512 x 3 grids make nothing and wait for
    /// nothing. Every read sees the other row's word, Metal's answer: the probe
    /// `metal2vulkan-conformance/probes/row_barrier_reads.m` ran the same source over the same
    /// grids on an Apple M3 Max under macOS 27.0.1, and no word of `seen` differed from it. A
    /// state whose descriptor declares `maxTotalThreadsPerThreadgroup` 64 and no required size
    /// makes, with the state, the pipelines a whole and a ragged grid run at 64 x 1, so its first
    /// 512 x 1 and 500 x 1 grids in 64 x 1 make nothing and wait for nothing; one row reads no
    /// other row's word.
    #[test]
    fn every_device_keeps_a_kernel_that_reads_across_rows_after_a_barrier_at_its_own_size() {
        let Some(descriptor) = harness_kernel("row_barrier") else {
            return;
        };
        const FILL: u32 = 0xEEEE_EEEE;
        each_device("compute_row_barrier", |name, _, device| {
            let made = device.counters().metrics();
            let state = state(name, &device, &descriptor);
            let created = device.counters().metrics();
            assert_eq!(
                (
                    created.pipelines_created - made.pipelines_created,
                    created.translations - made.translations,
                ),
                (0, 2),
                "{name}: no pipeline made with the state, and both modules lowered"
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let run = |state: &ComputePipelineState, grid: [u32; 3], size: [u32; 3]| {
                let count = grid[0] * grid[1];
                let mut buffers = Buffers::new(&device, 1 << 20);
                let cells = buffers.buffer(4 * u64::from(count));
                let seen = buffers.buffer(4 * u64::from(count));
                write(&cells, 0, &le_bytes(&vec![0; count as usize]));
                write(&seen, 0, &le_bytes(&vec![FILL; count as usize]));
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(state).expect("a state");
                encoder.set_buffer(Some(&cells), 0, 0).expect("a buffer");
                encoder.set_buffer(Some(&seen), 0, 1).expect("a buffer");
                (encoder.dispatch_threads(grid, size))
                    .unwrap_or_else(|error| panic!("{name}: {grid:?} in {size:?}: {error}"));
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
                let want: Vec<u32> = (0..grid[1])
                    .flat_map(|y| {
                        (0..grid[0]).map(move |x| match y ^ 1 {
                            other if other < grid[1] => ((other << 16) | x) + 1,
                            _ => FILL,
                        })
                    })
                    .collect();
                assert_eq!(
                    words(&seen, 0, want.len()),
                    want,
                    "{name}: {grid:?} in {size:?}"
                );
            };
            // Each grid at a size its state has not run: one pipeline, made by a first-use wait,
            // and no translation.
            let mut before = created;
            for (grid, size, what) in [
                ([512, 4, 1], [32, 2, 1], "the plain module at 32 x 2"),
                ([512, 3, 1], [32, 2, 1], "the partial variant at 32 x 2"),
                ([96, 5, 1], [32, 4, 1], "the partial variant at 32 x 4"),
            ] {
                run(&state, grid, size);
                let after = device.counters().metrics();
                assert_eq!(
                    (
                        after.pipelines_created - before.pipelines_created,
                        after.first_use_waits - before.first_use_waits,
                        after.translations - before.translations,
                        after.partial_variant_whole_grids - before.partial_variant_whole_grids,
                    ),
                    (1, 1, 0, 0),
                    "{name}: pipelines, first-use waits, translations and whole grids of {what}"
                );
                before = after;
            }
            // At the descriptor's required size, the make makes what both grids run.
            let mut required = descriptor.clone();
            required.required_threads_per_threadgroup = Some([32, 2, 1]);
            let sized = self::state(name, &device, &required);
            let made = device.counters().metrics();
            assert_eq!(
                (
                    made.pipelines_created - before.pipelines_created,
                    made.translations - before.translations,
                ),
                (2, 2),
                "{name}: the required size's state: its whole and ragged grids' pipelines"
            );
            run(&sized, [512, 4, 1], [32, 2, 1]);
            run(&sized, [512, 3, 1], [32, 2, 1]);
            let after = device.counters().metrics();
            assert_eq!(
                (
                    after.pipelines_created - made.pipelines_created,
                    after.first_use_waits - made.first_use_waits,
                    after.translations - made.translations,
                ),
                (0, 0, 0),
                "{name}: the required size's grids made nothing and waited for nothing"
            );
            // At a declared bound and no required size, the make makes what a whole and a ragged
            // grid run at the bound's row.
            let mut bounded = descriptor.clone();
            bounded.max_total_threads_per_threadgroup = Some(64);
            let before = device.counters().metrics();
            let bound = self::state(name, &device, &bounded);
            let made = device.counters().metrics();
            assert_eq!(bound.max_total_threads_per_threadgroup(), 64, "{name}");
            assert_eq!(
                made.pipelines_created - before.pipelines_created,
                2,
                "{name}: the declared bound's state: its whole and ragged grids' pipelines"
            );
            run(&bound, [512, 1, 1], [64, 1, 1]);
            run(&bound, [500, 1, 1], [64, 1, 1]);
            let after = device.counters().metrics();
            assert_eq!(
                (
                    after.pipelines_created - made.pipelines_created,
                    after.first_use_waits - made.first_use_waits,
                ),
                (0, 0),
                "{name}: the declared bound's grids made nothing and waited for nothing"
            );
            println!(
                "{name}: row barrier reads checked in 512 x 4 and 512 x 3 in 32 x 2 and 96 x 5 in \
                 32 x 4, at their own size; declared bound 64: whole and ragged grids' pipelines \
                 made by finish_pipeline"
            );
        });
    }

    /// `compute_thread_limit_64` declares `[[max_total_threads_per_threadgroup(64)]]` and observes
    /// its threadgroups, with no required size: its state makes the pipelines a whole and a
    /// ragged grid run at 64 x 1 x 1, the plain module's and the partial-threadgroups variant's,
    /// so its first 128 x 1 and 100 x 1 grids in 64 x 1 make nothing and wait for nothing, the
    /// whole one on the plain module (uncounted in `Metrics::partial_variant_whole_grids`), and
    /// each grid runs each of its threads once in threadgroups of 64. The same kernel without the attribute, bounded only by the device,
    /// makes none with its state.
    #[test]
    fn every_device_makes_a_declared_bound_s_grids_with_the_state() {
        let (Some(bounded), Some(plain)) = (
            descriptor_of("compute_thread_limit", "compute_thread_limit_64"),
            descriptor_of("compute_thread_limit", "compute_thread_limit_plain"),
        ) else {
            return;
        };
        each_device("compute_declared_bound", |name, _, device| {
            let counters = device.counters();
            let before = counters.metrics();
            let unbounded = state(name, &device, &plain);
            let made = counters.metrics();
            assert_eq!(
                made.pipelines_created - before.pipelines_created,
                0,
                "{name}: the device's own limit ({}) is no declared bound",
                unbounded.max_total_threads_per_threadgroup()
            );
            let state = state(name, &device, &bounded);
            let created = counters.metrics();
            assert_eq!(state.max_total_threads_per_threadgroup(), 64, "{name}");
            assert_eq!(
                created.pipelines_created - made.pipelines_created,
                2,
                "{name}: the whole and ragged grids' pipelines, made by finish_pipeline"
            );
            let queue = device.new_command_queue(4).expect("a queue");
            for threads in [128, 100] {
                let mut buffers = Buffers::new(&device, 4096);
                let out = buffers.buffer(16);
                write(&out, 0, &le_bytes(&[0; 4]));
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                encoder.set_buffer(Some(&out), 0, 0).expect("a buffer");
                (encoder.dispatch_threads([threads, 1, 1], [64, 1, 1]))
                    .unwrap_or_else(|error| panic!("{name}: {threads} threads: {error}"));
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
                assert_eq!(
                    words(&out, 0, 4),
                    [threads, 64, 1, 1],
                    "{name}: {threads} threads in 64"
                );
            }
            let after = counters.metrics();
            assert_eq!(
                (
                    after.pipelines_created - created.pipelines_created,
                    after.first_use_waits - created.first_use_waits,
                    after.partial_variant_whole_grids - created.partial_variant_whole_grids,
                ),
                (0, 0, 0),
                "{name}: the declared bound's grids made nothing and waited for nothing, the \
                 whole one on the plain module"
            );
            println!(
                "{name}: declared bound 64: 128 x 1 and 100 x 1 in 64 x 1 made by finish_pipeline, \
                 the whole one on the plain module and the ragged one on the partial variant"
            );
        });
    }

    /// A kernel's read of a texture its function constants removed takes zero and marks no
    /// status, so the state is made and the dispatch runs: the read's texel and the texture's
    /// width land as zeros over words that held 0xFFFF_FFFF, and the command buffer completes
    /// with no error.
    #[test]
    fn every_device_dispatches_a_kernel_whose_read_of_a_removed_texture_takes_zero() {
        let bitcode = match metal2vulkan_devtools::llvm_assemble_text(ABSENT_READ) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("absent_read_kernel: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let mut function =
            FunctionDescriptor::new(library.function("absent_read_kernel").expect("the kernel"));
        function.constants = metal2vulkan::library::FunctionConstantValues::defaults().with_index(
            0,
            metal2vulkan::library::FunctionConstantValue::new(
                metal2vulkan::metal::DataType::Bool,
                [0],
            ),
        );
        function.specialized = true;
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(function));
        each_device("compute_absent_read", |name, _, device| {
            let state = state(name, &device, &descriptor);
            let mut buffers = Buffers::new(&device, 4096);
            let out = buffers.buffer(32);
            write(&out, 0, &le_bytes(&[u32::MAX; 8]));
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            encoder.set_buffer(Some(&out), 0, 0).expect("a buffer");
            encoder
                .dispatch_threadgroups([1, 1, 1], [1, 1, 1])
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let want = [0, 0, 0, 0, 0, u32::MAX, u32::MAX, u32::MAX];
            assert_eq!(
                words(&out, 0, 8),
                want,
                "{name}: the texel, the width, and the rest"
            );
            println!("{name}: the removed texture's read and width took zero");
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

    /// A kernel of 256 threads whose every threadgroup access is a whole `float` word, over one
    /// `[[threadgroup(0)]]` region of `params[0]` words: each thread stores each of its words'
    /// indices, and after a barrier writes, for each of its words `k`, the word `params[0] - 1 - k`
    /// to `out[k]`, and after a barrier on device memory, for each of its `float4`s `j`, the
    /// `float4`'s sum to `out[params[0] + j]`. The kernel leaves no pair to lockstep, so no device
    /// runs it on emulated SIMD groups, whose arena keeps composed words: it masks `params[0]` to
    /// 18 bits (every device's limit is fewer words), which bounds each loop's counter, and each
    /// loop indexes one pointer by its counter (the sums' is `out + params[0]`), so two threads of
    /// one SIMD group never meet within a loop; and the barrier orders the two loops' stores to
    /// `out`, which two such threads could make to one word (a loop runs once in a thread past
    /// the region).
    const FLOAT_ARENA: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @float_arena(ptr addrspace(3) %a, ptr addrspace(1) %out, ptr addrspace(2) %params, i32 %t) {
entry:
  %given = load i32, ptr addrspace(2) %params, align 4
  %w = and i32 %given, 262143
  %last = sub i32 %w, 1
  %w4 = lshr i32 %w, 2
  br label %fill

fill:
  %i = phi i32 [ %t, %entry ], [ %i2, %fill ]
  %f = uitofp i32 %i to float
  %pa = getelementptr inbounds float, ptr addrspace(3) %a, i32 %i
  store float %f, ptr addrspace(3) %pa, align 4
  %i2 = add i32 %i, 256
  %more = icmp ult i32 %i2, %w
  br i1 %more, label %fill, label %sync

sync:
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %back

back:
  %k = phi i32 [ %t, %sync ], [ %k2, %back ]
  %r = sub i32 %last, %k
  %q = getelementptr inbounds float, ptr addrspace(3) %a, i32 %r
  %x = load float, ptr addrspace(3) %q, align 4
  %xo = fptoui float %x to i32
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %k
  store i32 %xo, ptr addrspace(1) %o, align 4
  %k2 = add i32 %k, 256
  %more2 = icmp ult i32 %k2, %w
  br i1 %more2, label %back, label %apart

apart:
  tail call void @air.wg.barrier(i32 1, i32 1)
  %sums = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %w
  br label %vectors

vectors:
  %j = phi i32 [ %t, %apart ], [ %j2, %vectors ]
  %j4 = shl i32 %j, 2
  %q4 = getelementptr inbounds float, ptr addrspace(3) %a, i32 %j4
  %y = load <4 x float>, ptr addrspace(3) %q4, align 16
  %y0 = extractelement <4 x float> %y, i32 0
  %y1 = extractelement <4 x float> %y, i32 1
  %y2 = extractelement <4 x float> %y, i32 2
  %y3 = extractelement <4 x float> %y, i32 3
  %ya = fadd float %y0, %y1
  %yb = fadd float %y2, %y3
  %ys = fadd float %ya, %yb
  %yo = fptoui float %ys to i32
  %o4 = getelementptr inbounds i32, ptr addrspace(1) %sums, i32 %j
  store i32 %yo, ptr addrspace(1) %o4, align 4
  %j2 = add i32 %j, 256
  %more3 = icmp ult i32 %j2, %w4
  br i1 %more3, label %vectors, label %done

done:
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!air.version = !{!7}
!air.language_version = !{!8}
!0 = !{ptr @float_arena, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"params"}
!6 = !{i32 3, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!7 = !{i32 2, i32 8, i32 0}
!8 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

    /// [`FLOAT_ARENA`] with its region filling the device's `maxComputeSharedMemorySize` to the
    /// byte. The translator holds the arena in `float` words (its accesses are the `.floats`
    /// ones), and the array is the limit's words exactly, so the dispatch runs where Metal runs
    /// it, and every answer is the program's.
    #[test]
    fn every_device_runs_a_float_arena_that_fills_the_threadgroup_memory_limit() {
        let bitcode = match metal2vulkan_devtools::llvm_assemble_text(FLOAT_ARENA) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("float_arena: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function = library.function("float_arena").expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        each_device("compute_float_arena_at_the_limit", |name, _, device| {
            let limit = device.core().limits.max_shared_memory;
            let n = limit / 4;
            let StageSource::Function(function) = &descriptor.kernel else {
                unreachable!("a Function descriptor");
            };
            let lowered = device
                .core()
                .translator
                .compile_function(function, &descriptor.lowering_inputs(), &Budget::standard())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let census = &lowered.strategies().entries;
            let lockstep: Vec<_> = (census.keys())
                .filter(|strategy| strategy.starts_with("barrier.lockstep"))
                .collect();
            assert!(lockstep.is_empty(), "{name}: {lockstep:?}");
            for strategy in [
                "access.threadgroup.store.floats",
                "access.threadgroup.load.floats",
            ] {
                assert!(
                    census.contains_key(strategy),
                    "{name}: {strategy} in {:?}",
                    census.keys().collect::<Vec<_>>()
                );
            }
            let state = state(name, &device, &descriptor);
            let mut buffers = Buffers::new(&device, 1 << 20);
            let out = buffers.buffer(5 * u64::from(limit) / 4);
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            encoder
                .set_threadgroup_memory_length(limit, 0)
                .expect("a length");
            encoder.set_buffer(Some(&out), 0, 0).expect("a buffer");
            encoder
                .set_bytes(&le_bytes(&[n]), 1)
                .expect("the parameters");
            encoder
                .dispatch_threadgroups([1, 1, 1], [256, 1, 1])
                .unwrap_or_else(|error| panic!("{name}: a region of {limit} bytes: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let want: Vec<u32> = (0..n)
                .map(|k| n - 1 - k)
                .chain((0..n / 4).map(|j| 16 * j + 6))
                .collect();
            let got = words(&out, 0, want.len());
            let first = got.iter().zip(&want).position(|(got, want)| got != want);
            assert_eq!(first, None, "{name}: the first word that differs");
            println!(
                "{name}: a float arena of {limit} bytes, the device's limit, answered all {} words",
                want.len()
            );
        });
    }

    /// A kernel of 64 threads that observes its threadgroups through two barriers in a loop of four
    /// iterations over 64 words of threadgroup memory: each iteration, thread `t` stores its value
    /// at word `t`, and past a barrier takes three times word `(t + 1) % 64` plus the iteration's
    /// index as its next value, before another barrier; it writes its last to `out[g]`, `g` its
    /// position in the grid. Its partial-threadgroups variant's predicated walk votes in the loop,
    /// so the variant takes a 4-byte vote word of threadgroup memory the plain module does not.
    /// `$NAME` and its region are filled by [`vote_kernel`].
    const VOTE_WORD: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"
$GLOBAL
define void @$NAME($PARAMSptr addrspace(1) %out, i32 %t, i32 %g) {
entry:
  %pa = getelementptr inbounds i32, ptr addrspace(3) $BASE, i32 %t
  %t1 = add i32 %t, 1
  %nb = and i32 %t1, 63
  %pn = getelementptr inbounds i32, ptr addrspace(3) $BASE, i32 %nb
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i2, %loop ]
  %acc = phi i32 [ %t, %entry ], [ %acc2, %loop ]
  store i32 %acc, ptr addrspace(3) %pa, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %x = load i32, ptr addrspace(3) %pn, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %x3 = mul i32 %x, 3
  %acc2 = add i32 %x3, %i
  %i2 = add i32 %i, 1
  %more = icmp ult i32 %i2, 4
  br i1 %more, label %loop, label %done

done:
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %g
  store i32 %acc2, ptr addrspace(1) %o, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!air.version = !{!7}
!air.language_version = !{!8}
!0 = !{ptr @$NAME, !1, !2}
!1 = !{}
!2 = !{$REGION!4, !5, !6}
$REGION_META!4 = !{i32 $OUT, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 $T, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!6 = !{i32 $G, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"g"}
!7 = !{i32 2, i32 8, i32 0}
!8 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

    /// [`VOTE_WORD`] as `vote_static`, over a declared array of `words` words, or with `None` as
    /// `vote_length`, over its `[[threadgroup(0)]]` region, whose length the encoder sets.
    fn vote_kernel(words: Option<u32>) -> String {
        let region = r#"!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"a"}
"#;
        let (name, global, params, base, meta) = match words {
            Some(words) => (
                "vote_static",
                format!("@tg = internal addrspace(3) global [{words} x i32] undef, align 4\n"),
                "",
                "@tg",
                "",
            ),
            None => (
                "vote_length",
                String::new(),
                "ptr addrspace(3) %a, ",
                "%a",
                region,
            ),
        };
        let first = u32::from(words.is_none());
        VOTE_WORD
            .replace("$NAME", name)
            .replace("$GLOBAL", &global)
            .replace("$PARAMS", params)
            .replace("$BASE", base)
            .replace("$REGION_META", meta)
            .replace("$REGION", if words.is_some() { "" } else { "!3, " })
            .replace("$OUT", &first.to_string())
            .replace("$T", &(first + 1).to_string())
            .replace("$G", &(first + 2).to_string())
    }

    /// The word [`VOTE_WORD`] writes for each thread of a threadgroup, by its index.
    fn vote_answers() -> Vec<u32> {
        let mut values: Vec<u32> = (0..64).collect();
        for i in 0..4 {
            values = (0..64)
                .map(|t| values[(t + 1) % 64].wrapping_mul(3).wrapping_add(i))
                .collect();
        }
        values
    }

    /// Two observers whose threadgroup memory is within the device's `maxComputeSharedMemorySize`
    /// on their plain module and past it by the 4-byte vote word on their partial-threadgroups
    /// variant ([`VOTE_WORD`]): `vote_static`, whose declared array of the limit's bytes less 12
    /// Metal reports as the limit to the byte (`staticThreadgroupMemoryLength` rounds up to 16), and
    /// `vote_length`, whose region the encoder sets at the limit. Each one's whole grid,
    /// threadgroup dispatch and indirect dispatch run on its plain module, as no dispatch of the
    /// variant can, uncounted in `Metrics::partial_variant_whole_grids`, and answer the program's
    /// words; its ragged grid, which only the variant runs, is refused
    /// (`Limit::ThreadgroupMemory`). `vote_static`'s state at `requiredThreadsPerThreadgroup` 64
    /// makes the plain module's pipeline with the state, which its three dispatches run, making
    /// nothing and waiting for nothing.
    #[test]
    fn every_device_runs_an_observer_on_its_plain_module_where_the_vote_word_passes_the_limit() {
        use crate::pipeline::Which;

        each_device("compute_vote_word_at_the_limit", |name, _, device| {
            let limit = device.core().limits.max_shared_memory;
            let answers = vote_answers();
            let want: Vec<u32> = (0..256).map(|g| answers[g % 64]).collect();
            for (declared, length) in [(Some(limit / 4 - 3), None), (None, Some(limit))] {
                let bitcode =
                    match metal2vulkan_devtools::llvm_assemble_text(&vote_kernel(declared)) {
                        Ok(bitcode) => bitcode,
                        Err(AssembleError::Missing(why)) => {
                            eprintln!("SKIPPED: {why}");
                            return;
                        }
                        Err(error) => panic!("vote_word: {error:?}"),
                    };
                let kernel = match declared {
                    Some(_) => "vote_static",
                    None => "vote_length",
                };
                let library = Library::from_air(&bitcode).expect("the library");
                let function = library.function(kernel).expect("the kernel");
                let mut descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                    FunctionDescriptor::new(function),
                ));
                if declared.is_some() {
                    descriptor.required_threads_per_threadgroup = Some([64, 1, 1]);
                }
                let before = device.counters().metrics();
                let state = state(name, &device, &descriptor);
                let made = device.counters().metrics();
                let compiled = &state.inner.compiled;
                let partial = (compiled.variant_kernel(Which::Partial, |_| {
                    panic!("{name}: {kernel}: the partial-threadgroups variant is refused")
                }))
                .unwrap_or_else(|error| panic!("{name}: {kernel}: {error}"));
                assert_eq!(
                    (
                        compiled.whole.interface.memory.vote,
                        partial.interface.memory.vote,
                    ),
                    (0, 4),
                    "{name}: {kernel}: the vote word, the variant's alone"
                );
                assert_eq!(
                    state.static_threadgroup_memory_length(),
                    if declared.is_some() { limit } else { 0 },
                    "{name}: {kernel}"
                );
                if declared.is_some() {
                    assert_eq!(
                        made.pipelines_created - before.pipelines_created,
                        1,
                        "{name}: {kernel}: the plain module's pipeline, made by finish_pipeline"
                    );
                }
                let queue = device.new_command_queue(4).expect("a queue");
                let mut buffers = Buffers::new(&device, 1 << 16);
                let arguments = buffers.buffer(16);
                write(&arguments, 0, &le_bytes(&[4, 1, 1]));
                for how in [How::Threads, How::Groups, How::Indirect] {
                    let out = buffers.buffer(4 * 256);
                    write(&out, 0, &le_bytes(&[0xEEEE_EEEE; 256]));
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    let mut encoder = command_buffer
                        .compute_command_encoder(DispatchType::Serial)
                        .expect("an encoder");
                    encoder.set_compute_pipeline_state(&state).expect("a state");
                    if let Some(length) = length {
                        (encoder.set_threadgroup_memory_length(length, 0)).expect("a length");
                    }
                    encoder.set_buffer(Some(&out), 0, 0).expect("a buffer");
                    match how {
                        How::Threads => encoder.dispatch_threads([256, 1, 1], [64, 1, 1]),
                        How::Groups => encoder.dispatch_threadgroups([4, 1, 1], [64, 1, 1]),
                        How::Indirect => encoder.dispatch_threadgroups_with_indirect_buffer(
                            &arguments,
                            0,
                            [64, 1, 1],
                        ),
                    }
                    .unwrap_or_else(|error| panic!("{name}: {kernel}: {how:?}: {error}"));
                    encoder.end_encoding();
                    command_buffer.commit().expect("a commit");
                    finished(name, &command_buffer);
                    assert_eq!(command_buffer.error(), None, "{name}: {kernel}: {how:?}");
                    assert_eq!(words(&out, 0, 256), want, "{name}: {kernel}: {how:?}");
                }
                let after = device.counters().metrics();
                assert_eq!(
                    after.partial_variant_whole_grids - made.partial_variant_whole_grids,
                    0,
                    "{name}: {kernel}: whole threadgroups, direct and indirect, on the plain module"
                );
                if declared.is_some() {
                    assert_eq!(
                        (
                            after.pipelines_created - made.pipelines_created,
                            after.first_use_waits - made.first_use_waits,
                        ),
                        (0, 0),
                        "{name}: {kernel}: the dispatches made nothing and waited for nothing"
                    );
                }
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                if let Some(length) = length {
                    (encoder.set_threadgroup_memory_length(length, 0)).expect("a length");
                }
                let out = buffers.buffer(4 * 256);
                encoder.set_buffer(Some(&out), 0, 0).expect("a buffer");
                match encoder.dispatch_threads([200, 1, 1], [64, 1, 1]) {
                    Err(Error::Limit(Limit::ThreadgroupMemory {
                        length,
                        max,
                        static_length,
                    })) => {
                        assert_eq!(
                            (length, max, static_length),
                            (
                                u64::from(limit) + 4,
                                limit,
                                state.static_threadgroup_memory_length()
                            ),
                            "{name}: {kernel}"
                        )
                    }
                    answer => panic!("{name}: {kernel}: a ragged grid: {answer:?}"),
                }
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                println!(
                    "{name}: {kernel}: at the limit of {limit} bytes, a whole grid, threadgroups \
                     and an indirect dispatch ran on the plain module; a ragged grid refused"
                );
            }
        });
    }

    /// An inclusive scan of a threadgroup's threads over threadgroup memory at `$BASE`: thread `t`
    /// stores `t + 1` at word `t`, then for each stride `s` of 1, 2, 4 and on below
    /// `threads_per_threadgroup`, past a barrier adds word `t - s` where `t >= s`, and past another
    /// barrier stores its sum at word `t`; it writes its last sum to `out[g]`, `g` its position in
    /// the grid. The loop holds the barriers, so the partial-threadgroups variant's predicated walk
    /// votes in it, and its trip count is the threadgroup's threads, which a partial threadgroup
    /// has fewer of. `$NAME`, its memory and its region are filled by [`scan_kernel`].
    const SCAN: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"
$GLOBAL
define void @$NAME($PARAMSptr addrspace(1) %out, i32 %t, i32 %g, <3 x i32> %size) {
entry:
  %n = extractelement <3 x i32> %size, i64 0
  %pt = getelementptr inbounds i32, ptr addrspace(3) $BASE, i32 %t
  %v0 = add i32 %t, 1
  store i32 %v0, ptr addrspace(3) %pt, align 4
  br label %head

head:
  %s = phi i32 [ 1, %entry ], [ %s2, %body ]
  %acc = phi i32 [ %v0, %entry ], [ %acc2, %body ]
  %more = icmp ult i32 %s, %n
  br i1 %more, label %body, label %done

body:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %has = icmp uge i32 %t, %s
  %back = sub i32 %t, %s
  %at = select i1 %has, i32 %back, i32 %t
  %pa = getelementptr inbounds i32, ptr addrspace(3) $BASE, i32 %at
  %x = load i32, ptr addrspace(3) %pa, align 4
  %add = select i1 %has, i32 %x, i32 0
  %acc2 = add i32 %acc, %add
  tail call void @air.wg.barrier(i32 2, i32 1)
  store i32 %acc2, ptr addrspace(3) %pt, align 4
  %s2 = shl i32 %s, 1
  br label %head

done:
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %g
  store i32 %acc, ptr addrspace(1) %o, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!air.version = !{!7}
!air.language_version = !{!8}
!0 = !{ptr @$NAME, !1, !2}
!1 = !{}
!2 = !{$REGION!3, !4, !5, !6}
$REGION_META!3 = !{i32 $OUT, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 $T, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!5 = !{i32 $G, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"g"}
!6 = !{i32 $S, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"size"}
!7 = !{i32 2, i32 8, i32 0}
!8 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

    /// [`SCAN`] as `scan_static`, over a declared array of `words` words, which a guest sizes from
    /// the limit it is told when it builds the kernel, or with `None` as `scan_length`, over its
    /// `[[threadgroup(0)]]` region alone, whose length the encoder sets.
    fn scan_kernel(words: Option<u32>) -> String {
        let region = r#"!9 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"a"}
"#;
        let (name, global, params, base, meta) = match words {
            Some(words) => (
                "scan_static",
                format!("@tg = internal addrspace(3) global [{words} x i32] undef, align 4\n"),
                "",
                "@tg",
                "",
            ),
            None => (
                "scan_length",
                String::new(),
                "ptr addrspace(3) %a, ",
                "%a",
                region,
            ),
        };
        let first = u32::from(words.is_none());
        SCAN.replace("$NAME", name)
            .replace("$GLOBAL", &global)
            .replace("$PARAMS", params)
            .replace("$BASE", base)
            .replace("$REGION_META", meta)
            .replace("$REGION", if words.is_some() { "" } else { "!9, " })
            .replace("$OUT", &first.to_string())
            .replace("$T", &(first + 1).to_string())
            .replace("$G", &(first + 2).to_string())
            .replace("$S", &(first + 3).to_string())
    }

    /// [`SCAN`] at 1,024 threads, Metal's largest threadgroup, in its two shapes ([`scan_kernel`]):
    /// `scan_static`'s declared array and `scan_length`'s region the encoder sets, each of the
    /// bytes a guest is told for Metal's `maxThreadgroupMemoryLength`: Metal's own 32 KiB (or the
    /// device's `maxComputeSharedMemorySize` where smaller), and `room`, the device's answer
    /// (`Device::max_threadgroup_memory_length`: its limit less the predicated walk's 4-byte vote
    /// word, rounded down to 16 and at most 32 KiB, which the test checks). Each state's
    /// ragged grid of 25 threads, one partial threadgroup of 25, and its whole grid of 1,024
    /// threads: at `room` every one runs and answers the program's inclusive sums,
    /// `(t + 1)(t + 2) / 2` for thread `t` of its threadgroup, and each ragged grid that runs adds
    /// one to `Metrics::partial_variant_ragged_grids`, which a whole grid or a refused one leaves
    /// alone, so the count is above zero at the end. At 32 KiB, where the bytes and the
    /// word pass the device's limit, the ragged grids are refused, `scan_static`'s as
    /// `Gap::PartialThreadgroups` at the vote word (the translator refuses the variant), naming
    /// its static bytes and the limit, and `scan_length`'s as `Limit::ThreadgroupMemory` of the
    /// bytes and the word, naming no static bytes; the whole grids run on the plain module and
    /// answer the sums. Each outcome is printed; the test fails on any other.
    #[test]
    fn every_device_runs_a_1024_thread_observers_ragged_grid_in_the_limit_less_its_vote_word() {
        use crate::pipeline::{Which, VOTE_BYTES};

        each_device("compute_scan_at_the_told_limit", |name, _, device| {
            let limit = device.core().limits.max_shared_memory;
            let metal = limit.min(32 * 1024);
            let room = device.max_threadgroup_memory_length();
            assert_eq!(
                room,
                ((limit - VOTE_BYTES) / 16 * 16).min(32 * 1024),
                "{name}: the answer, the limit less the vote word, rounded down to 16, at most 32 KiB"
            );
            let mut told = vec![metal];
            if room != metal {
                told.push(room);
            }
            let queue = device.new_command_queue(4).expect("a queue");
            let mut buffers = Buffers::new(&device, 1 << 16);
            let mut failed = Vec::new();
            let mut ragged = 0;
            for bytes in told {
                for declared in [Some(bytes / 4), None] {
                    let bitcode =
                        match metal2vulkan_devtools::llvm_assemble_text(&scan_kernel(declared)) {
                            Ok(bitcode) => bitcode,
                            Err(AssembleError::Missing(why)) => {
                                eprintln!("SKIPPED: {why}");
                                return;
                            }
                            Err(error) => panic!("scan: {error:?}"),
                        };
                    let kernel = match declared {
                        Some(_) => "scan_static",
                        None => "scan_length",
                    };
                    let library = Library::from_air(&bitcode).expect("the library");
                    let function = library.function(kernel).expect("the kernel");
                    let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
                        FunctionDescriptor::new(function),
                    ));
                    let state = state(name, &device, &descriptor);
                    let compiled = &state.inner.compiled;
                    // The variant's vote word, or why the translator refused the variant.
                    let mut refused = None;
                    let vote = (compiled.variant_kernel(Which::Partial, |why| {
                        refused = Some(why);
                        Error::Quiesced
                    }))
                    .map(|partial| partial.interface.memory.vote)
                    .map_err(|_| refused);
                    let past = bytes + 4 > limit;
                    for grid in [25, 1024] {
                        let want: Vec<u32> = (0..grid).map(|t| (t + 1) * (t + 2) / 2).collect();
                        let out = buffers.buffer(4 * 1024);
                        write(&out, 0, &le_bytes(&[0xEEEE_EEEE; 1024]));
                        let before = device.counters().metrics();
                        let command_buffer = queue.command_buffer().expect("a command buffer");
                        let mut encoder = command_buffer
                            .compute_command_encoder(DispatchType::Serial)
                            .expect("an encoder");
                        encoder.set_compute_pipeline_state(&state).expect("a state");
                        if declared.is_none() {
                            encoder
                                .set_threadgroup_memory_length(bytes, 0)
                                .expect("a length");
                        }
                        encoder.set_buffer(Some(&out), 0, 0).expect("a buffer");
                        let dispatched = encoder.dispatch_threads([grid, 1, 1], [1024, 1, 1]);
                        encoder.end_encoding();
                        command_buffer.commit().expect("a commit");
                        finished(name, &command_buffer);
                        let counted = device.counters().metrics().partial_variant_ragged_grids
                            - before.partial_variant_ragged_grids;
                        ragged += counted;
                        let sums = dispatched.is_ok()
                            && command_buffer.error().is_none()
                            && words(&out, 0, grid as usize) == want;
                        let expected = match (past && grid == 25, declared) {
                            (false, _) => sums,
                            (true, Some(_)) => matches!(
                                dispatched,
                                Err(Error::Unsupported(Gap::PartialThreadgroups {
                                    threads_per_grid: [25, 1, 1],
                                    threads_per_threadgroup: [1024, 1, 1],
                                    site: PartialThreadgroupsSite::VoteWord { static_length, max },
                                })) if static_length == bytes && max == limit
                            ),
                            (true, None) => matches!(
                                dispatched,
                                Err(Error::Limit(Limit::ThreadgroupMemory {
                                    length,
                                    max,
                                    static_length: 0,
                                })) if length == u64::from(bytes) + 4 && max == limit
                            ),
                        };
                        // Each ragged grid that runs is counted once; a whole or refused one is not.
                        let expected =
                            expected && counted == u64::from(grid == 25 && dispatched.is_ok());
                        let outcome = match &dispatched {
                            Err(error) => format!("refused: {error:?}"),
                            Ok(()) if command_buffer.error().is_some() => {
                                format!("failed: {:?}", command_buffer.error())
                            }
                            Ok(()) if sums => "ran, the program's sums".to_owned(),
                            Ok(()) => format!("ran, other words: {:?}", words(&out, 0, 8)),
                        };
                        println!(
                            "{name}: {kernel}, {bytes} of {limit} bytes (vote word {vote:?}), \
                             [{grid}, 1, 1] in [1024, 1, 1]: {outcome}; ragged grids counted \
                             {counted}"
                        );
                        if !expected {
                            failed.push(format!(
                                "{kernel}, {bytes} bytes, grid {grid}: {outcome}, {counted} counted"
                            ));
                        }
                    }
                }
            }
            assert!(failed.is_empty(), "{name}: {failed:?}");
            assert!(
                ragged > 0,
                "{name}: ragged grids on the partial-threadgroups variant, counted"
            );
        });
    }

    /// A kernel of one barrier, in no loop, over its `[[threadgroup(0)]]` region: thread `t` stores
    /// `t + 1` at word `t`, and past the barrier writes word `(t + 1) % n` to `out[g]`, `n` its
    /// threadgroup's threads and `g` its position in the grid. It observes its threadgroup
    /// partition, and its partial-threadgroups variant's predicated walk takes no vote word.
    const NEIGHBOUR_LENGTH: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @neighbour_length(ptr addrspace(3) %a, ptr addrspace(1) %out, i32 %t, i32 %g, <3 x i32> %size) {
entry:
  %n = extractelement <3 x i32> %size, i64 0
  %pt = getelementptr inbounds i32, ptr addrspace(3) %a, i32 %t
  %v = add i32 %t, 1
  store i32 %v, ptr addrspace(3) %pt, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %nb = urem i32 %v, %n
  %pn = getelementptr inbounds i32, ptr addrspace(3) %a, i32 %nb
  %x = load i32, ptr addrspace(3) %pn, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %g
  store i32 %x, ptr addrspace(1) %o, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!air.version = !{!8}
!air.language_version = !{!9}
!0 = !{ptr @neighbour_length, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"g"}
!7 = !{i32 4, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"size"}
!8 = !{i32 2, i32 8, i32 0}
!9 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

    /// [`NEIGHBOUR_LENGTH`] at 1,024 threads with its region set at Metal's 32 KiB (the device's
    /// `maxComputeSharedMemorySize` where smaller), past `Device::max_threadgroup_memory_length`
    /// on a device of 32,768 bytes: the answer changes only what a guest is told, and a kernel
    /// whose variant takes no vote word runs its ragged grid of 25 threads and its whole grid of
    /// 1,024 within the device's own limit, each answering `(t + 1) % n + 1` for thread `t` of a
    /// threadgroup of `n`.
    #[test]
    fn every_device_runs_a_kernel_without_a_vote_word_at_a_length_past_its_answer() {
        use crate::pipeline::Which;

        let bitcode = match metal2vulkan_devtools::llvm_assemble_text(NEIGHBOUR_LENGTH) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("neighbour_length: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function = library.function("neighbour_length").expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        each_device("compute_length_past_the_answer", |name, _, device| {
            let limit = device.core().limits.max_shared_memory;
            let length = limit.min(32 * 1024);
            let answer = device.max_threadgroup_memory_length();
            let state = state(name, &device, &descriptor);
            let compiled = &state.inner.compiled;
            let partial = (compiled.variant_kernel(Which::Partial, |why| {
                panic!("{name}: the partial-threadgroups variant is refused: {why:?}")
            }))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(partial.interface.memory.vote, 0, "{name}: no vote word");
            let queue = device.new_command_queue(4).expect("a queue");
            let mut buffers = Buffers::new(&device, 1 << 16);
            for grid in [25, 1024] {
                let want: Vec<u32> = (0..grid).map(|t| (t + 1) % grid + 1).collect();
                let out = buffers.buffer(4 * 1024);
                write(&out, 0, &le_bytes(&[0xEEEE_EEEE; 1024]));
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                (encoder.set_threadgroup_memory_length(length, 0)).expect("a length");
                encoder.set_buffer(Some(&out), 0, 0).expect("a buffer");
                (encoder.dispatch_threads([grid, 1, 1], [1024, 1, 1])).unwrap_or_else(|error| {
                    panic!("{name}: a length of {length} past the answer {answer}: {error}")
                });
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
                let got = words(&out, 0, grid as usize);
                let first = got.iter().zip(&want).position(|(got, want)| got != want);
                assert_eq!(
                    first, None,
                    "{name}: grid {grid}: the first word that differs"
                );
            }
            println!(
                "{name}: a length of {length} of {limit} bytes, past the answer {answer}, ran \
                 [25, 1, 1] and [1024, 1, 1] in [1024, 1, 1] and answered every word"
            );
        });
    }

    /// A kernel that reads, through the argument buffer at buffer 0 (one `texture2d<float, read>`
    /// member, its 8-byte resource ID), the texture's width, height and texel `(0, 0)`'s red bits
    /// into words 0 to 2 of `out` (buffer 1).
    const ARGUMENT_TEXTURE: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.T = type { ptr addrspace(1) }
%struct.R = type { %struct.T }

define void @argument_texture(ptr addrspace(2) %r, ptr addrspace(1) noundef "air-buffer-no-alias" %out) {
entry:
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %r, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r0 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { <4 x float>, i8 } %r0, 0
  %red = extractelement <4 x float> %v, i64 0
  %rb = bitcast float %red to i32
  %w = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 0)
  %h = tail call i32 @air.get_height_texture_2d(ptr addrspace(1) %t, i32 0)
  store i32 %w, ptr addrspace(1) %out, align 4
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %h, ptr addrspace(1) %o1, align 4
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 2
  store i32 %rb, ptr addrspace(1) %o2, align 4
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
declare i32 @air.get_height_texture_2d(ptr addrspace(1), i32)

!air.kernel = !{!0}
!air.version = !{!7}
!air.language_version = !{!8}
!0 = !{ptr @argument_texture, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"r"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, read>", !"t", !"air.indirect_argument", !6}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!6 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!7 = !{i32 2, i32 8, i32 0}
!8 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

    /// An argument buffer's texture ID is the program's word, so the shader masks its index to
    /// the set-1 image arrays: an ID whose index is 65,536 past texture 5's reads texture 5, as
    /// its own ID does, and never an element outside the arrays.
    #[test]
    fn every_device_reads_an_argument_buffer_texture_at_its_index_masked_to_the_arrays() {
        use crate::{
            IndexClass, PixelFormat, Region, Resource, ResourceUsage, TextureDescriptor,
            TextureType, TextureUsage,
        };
        let bitcode = match metal2vulkan_devtools::llvm_assemble_text(ARGUMENT_TEXTURE) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("argument_texture: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function = library.function("argument_texture").expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        each_device("compute_argument_texture_masked", |name, _, device| {
            let _grant = device
                .grant_indices(IndexClass::Texture, 1, 8)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (width, height) = (7u32, 3u32);
            let mut d = TextureDescriptor::new(
                TextureType::Type2D,
                PixelFormat::RGBA32Float,
                width,
                height,
            );
            d.usage = TextureUsage::SHADER_READ;
            let texture = device
                .new_texture(&d, Some(5))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let texel = [0.75f32, 0.0, 0.0, 1.0];
            let bytes: Vec<u8> = (0..width * height)
                .flat_map(|_| texel.iter().flat_map(|c| c.to_le_bytes()))
                .collect();
            let row = u64::from(width) * 16;
            texture
                .replace_region(
                    Region {
                        origin: [0, 0, 0],
                        size: [width, height, 1],
                    },
                    0,
                    0,
                    crate::HostBytes::from(&bytes[..]),
                    row,
                    row * u64::from(height),
                )
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let id = texture.gpu_resource_id();
            assert_eq!(id.index, 5, "{name}");
            let state = state(name, &device, &descriptor);
            let mut buffers = Buffers::new(&device, 1 << 16);
            let queue = device.new_command_queue(4).expect("a queue");
            for index in [id.index, id.index + 0x1_0000] {
                let argument = buffers.buffer(8);
                write(
                    &argument,
                    0,
                    &(u64::from(index) | u64::from(id.word) << 32).to_le_bytes(),
                );
                let out = buffers.buffer(16);
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                encoder.set_buffer(Some(&argument), 0, 0).expect("a buffer");
                encoder.set_buffer(Some(&out), 0, 1).expect("a buffer");
                encoder
                    .use_resource(Resource::Texture(&texture), ResourceUsage::READ)
                    .expect("a resident texture");
                encoder
                    .dispatch_threadgroups([1, 1, 1], [1, 1, 1])
                    .expect("a dispatch");
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
                assert_eq!(
                    words(&out, 0, 3),
                    [width, height, 0.75f32.to_bits()],
                    "{name}: an ID of index {index:#x}"
                );
            }
            println!("{name}: index 5 and index 0x10005 both read texture 5");
        });
    }

    /// `tgz_reads`: every thread reads its words of a declared threadgroup array and of a
    /// `[[threadgroup(0)]]` region before any thread of its threadgroup stores a nonzero pattern
    /// there, then reads its neighbour's words back (`metal2vulkan_devtools::threadgroup_start`).
    /// Two dispatches of 2048 threadgroups each, in threadgroups of 256 threads, of 250 and of 30,
    /// the last two not whole SIMD groups. Every device reads zero before the stores, in later
    /// threadgroups of a dispatch and in the second dispatch alike, as Metal does (the
    /// translator's null initializers on `Device::from_queries`' profile), and every pattern back;
    /// a size that is not whole groups of what the plain module runs runs the emulated-SIMD-group
    /// module.
    #[test]
    fn every_device_reads_zero_from_threadgroup_memory_until_the_program_writes_it() {
        use metal2vulkan_devtools::threadgroup_start::{
            answer, DISPATCHES, FIXTURE, GROUPS, REGION_BYTES, SIZES,
        };

        let Some(descriptor) = descriptor(FIXTURE) else {
            return;
        };
        each_device("compute_threadgroup_start", |name, _, device| {
            let state = state(name, &device, &descriptor);
            let multiples = state.inner.compiled.whole.interface.multiples.clone();
            let queue = device.new_command_queue(4).expect("a queue");
            let mut ran = Vec::new();
            for size in SIZES {
                let mut buffers = Buffers::new(&device, 1 << 12);
                let outs: Vec<Buffer> = (0..DISPATCHES).map(|_| buffers.buffer(16)).collect();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                encoder
                    .set_threadgroup_memory_length(REGION_BYTES, 0)
                    .expect("a length");
                for out in &outs {
                    encoder.set_buffer(Some(out), 0, 0).expect("a buffer");
                    encoder
                        .dispatch_threadgroups([GROUPS, 1, 1], [size, 1, 1])
                        .unwrap_or_else(|error| panic!("{name}: threadgroups of {size}: {error}"));
                }
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
                let want = answer(GROUPS, size);
                for (dispatch, out) in outs.iter().enumerate() {
                    assert_eq!(
                        words(out, 0, 4),
                        want,
                        "{name}: dispatch {dispatch} of threadgroups of {size} (words: decl read \
                         nonzero, arg read nonzero, the patterns' sum, the SIMD groups' shape)"
                    );
                }
                let emulated = multiples.iter().any(|&multiple| size % multiple != 0);
                if emulated {
                    assert!(
                        emulated_made(&state),
                        "{name}: threadgroups of {size}, not whole groups of {multiples:?}, ran \
                         the emulated module"
                    );
                }
                ran.push(format!(
                    "{size} {}",
                    if emulated { "emulated" } else { "plain" }
                ));
            }
            println!(
                "{name}: threadgroup memory read zero before the program's stores in two \
                 dispatches of {GROUPS} threadgroups of {}; pinned at {:?}, whole groups of \
                 {multiples:?}",
                ran.join(", "),
                state.inner.compiled.whole.interface.required_subgroup_size,
            );
        });
    }

    /// A refused state's error names its function by its own name in its library: a compute
    /// state of the examples' vertex function `colored_vertex`, which Metal makes no compute state
    /// of, is `Error::Translation` naming it, and its text prints the name.
    #[test]
    fn every_device_names_the_function_of_a_refused_compute_state() {
        let Some(examples) = crate::tests::draw::on_device::examples() else {
            return;
        };
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(examples.function("colored_vertex").expect("the function")),
        ));
        each_device("refused compute names", |name, _, device| {
            let got = device
                .new_compute_pipeline_state(&descriptor, |samplers| Ok(vec![0; samplers.len()]));
            match got {
                Err(Error::Translation { error, functions }) => {
                    assert_eq!(*functions, ["colored_vertex"], "{name}: {error}");
                    let text = Error::Translation { error, functions }.to_string();
                    assert!(
                        text.starts_with("the translator made no pipeline of colored_vertex: "),
                        "{name}: {text}"
                    );
                    println!("{name}: {text}");
                }
                Err(other) => panic!("{name}: {other}"),
                Ok(_) => panic!("{name}: a compute state of a vertex function was made"),
            }
        });
    }
}
