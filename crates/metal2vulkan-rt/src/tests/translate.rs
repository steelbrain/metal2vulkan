//! Translated kernels, under a device cfg on every device of the cfg's kind: a kernel the
//! translator compiles under the identity domains runs with its buffers bound at offsets, its
//! answer checked against the program's, and its `VkPipeline` made by the first dispatch alone, a
//! first-use wait the counters record.
//!
//! `llvm-as` assembles the kernel; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished, Arena};
    use crate::{
        Buffer, ComputePipelineDescriptor, DispatchType, FunctionDescriptor, Library, StageSource,
    };
    use metal2vulkan::Budget;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

    /// `float` patterns none of which is a NaN or a denormal (the fixture compiles with fast math).
    const FLOATS: [f32; 6] = [0.0, -0.0, 0.5, 2.0, 1.0, -3.0];
    /// The same values' `half` bits.
    const HALVES: [u16; 6] = [0x0000, 0x8000, 0x3800, 0x4000, 0x3c00, 0xc200];
    /// Threads in a threadgroup, and threadgroups.
    const GROUP: u32 = 4;
    const GROUPS: u32 = 3;
    /// The bind offsets of the three buffers, so a dropped offset shows.
    const OFFSETS: [u64; 3] = [32, 16, 4];

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

    /// `a1_bool_from_float`, translated, computes the program's answer over buffers bound at
    /// offsets, by `dispatch_threadgroups` and by `dispatch_threads` of one threadgroup size; the
    /// first dispatch made its `VkPipeline`, one first-use wait, and the second reused it.
    #[test]
    fn every_device_runs_a_translated_kernel() {
        let bitcode = match llvm_assemble(&engine_dir().join("a1_bool_from_float.ll")) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("a1_bool_from_float: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function = library.function("a1_bool_from_float").expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        each_device("compute_translated", |name, _, device| {
            let state = match device.new_compute_pipeline_state(&descriptor) {
                Ok(state) => state,
                Err(error) => {
                    // What the translator gave, for the refusal's reason.
                    let translated = device
                        .core()
                        .translator
                        .compile_compute_pipeline(&descriptor, &Budget::standard());
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
                    panic!("{name}: {error}");
                }
            };
            let threads = (GROUP * GROUPS) as usize;
            let mut bits = Vec::new();
            let mut halves = Vec::new();
            for i in 0..threads {
                let (f, h) = lanes(i);
                for (&fi, &hi) in f.iter().zip(&h) {
                    bits.extend_from_slice(&FLOATS[fi].to_bits().to_le_bytes());
                    halves.extend_from_slice(&HALVES[hi].to_le_bytes());
                }
            }
            let mut arena = Arena::new(&device);
            let (a, b) = (arena.buffer(), arena.buffer());
            let outs = [arena.buffer(), arena.buffer()];
            write(&a, OFFSETS[0], &bits);
            write(&b, OFFSETS[1], &halves);
            let before = device.counters().metrics();
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
                .dispatch_threadgroups([GROUPS, 1, 1], [GROUP, 1, 1])
                .expect("a dispatch");
            encoder
                .set_buffer(Some(&outs[1]), OFFSETS[2], 2)
                .expect("a buffer");
            encoder
                .dispatch_threads([GROUP * GROUPS, 1, 1], [GROUP, 1, 1])
                .expect("a dispatch");
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let mut expected = vec![0];
            expected.extend((0..threads).map(want));
            // The word before the offset and the one past the last thread stay 0.
            expected.push(0);
            for out in &outs {
                assert_eq!(
                    words(out, OFFSETS[2] - 4, threads + 2),
                    expected,
                    "{name}: out"
                );
            }
            let after = device.counters().metrics();
            let waits = after.first_use_waits - before.first_use_waits;
            let nanos = after.first_use_wait_nanos - before.first_use_wait_nanos;
            assert_eq!(waits, 1, "{name}: first-use waits");
            assert!(nanos > 0, "{name}: first-use wait time");
            assert!(
                after.first_use_wait_max_nanos >= nanos,
                "{name}: longest wait"
            );
            assert_eq!(
                after.pipelines_created - before.pipelines_created,
                1,
                "{name}: pipelines"
            );
            println!(
                "{name}: translated kernel checked over {threads} threads; first-use waits {waits}, {nanos} ns; precision notes {}",
                state.precision_notes().len()
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
            let interface = &state.inner.interface;
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
