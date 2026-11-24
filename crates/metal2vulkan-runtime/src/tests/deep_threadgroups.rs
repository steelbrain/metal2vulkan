//! A threadgroup deeper than the floor's `maxComputeWorkGroupSize` z, under a device cfg on every
//! device of the cfg's kind: `simd_sum_k` (the owned fixture `engine/simd_ops.ll`, each thread's
//! `simd_sum(t) + simd_sum(float(t & 7)) * 100000` at its threadgroup index `t`) dispatched as one
//! threadgroup of `[1, 1, 96]`, which a kernel whose workgroup is its threadgroup runs as one row
//! (`SpecKey::DeepThreadgroup`) and a flattened one as its flat row, so the dispatch is within
//! every device's limits. Each simdgroup is 32 consecutive threads, as Metal's are: thread `t`
//! answers `1024 (t / 32) + 496 + 11200000`.
//!
//! Also every thread of every workgroup writes its own word, deep or not: `grid_words` (an owned
//! kernel with no SIMD-group function, so its workgroup is its threadgroup, and which reads its
//! threadgroup's position and size, so it runs at its own size) stores `i + 1` at its index `i` in
//! the grid, over threadgroups of `[1, 1, 96]` (one row), `[8, 8, 1]` and `[4, 4, 4]`, several
//! of each. The module each create hands the driver holds no specialization constant: the
//! workgroup's size is its entry point's `LocalSize` literals (`Create::specialize`), so no driver
//! can size a workgroup from a constant's default.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        ComputePipelineDescriptor, DispatchType, FunctionDescriptor, Library, MemoryClass,
        StageSource,
    };
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, llvm_assemble_text, AssembleError};

    /// The threadgroup's depth: past `DEEP_THREADGROUP`, the floor's z.
    const DEPTH: u32 = 96;

    /// What thread `t` writes.
    fn want(t: u32) -> u32 {
        1024 * (t / 32) + 496 + 112 * 100_000
    }

    /// One threadgroup of `[1, 1, 96]` answers Metal's simdgroups of 32 consecutive threads on
    /// every device.
    #[test]
    fn every_device_runs_a_threadgroup_deeper_than_the_floor() {
        let path = engine_dir().join("simd_ops.ll");
        let bitcode = match llvm_assemble(&path) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("simd_ops: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function = library.function("simd_sum_k").expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        let out_len = u64::from(DEPTH * 4);
        each_device("deep_threadgroups", |name, _, device| {
            let state = device
                .new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let block = device
                .take_block(MemoryClass::Shared, out_len)
                .expect("a Shared block");
            let output = device
                .new_buffer(block.id, 0, out_len, Some(0))
                .expect("the output");
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            encoder.set_buffer(Some(&output), 0, 0).expect("a buffer");
            encoder
                .dispatch_threads([1, 1, DEPTH], [1, 1, DEPTH])
                .unwrap_or_else(|error| panic!("{name}: the dispatch: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let contents = output.contents().expect("a Shared buffer");
            let mut wrong = Vec::new();
            for t in 0..DEPTH {
                // SAFETY: the word lies in the buffer, which is alive, and its work completed.
                let got = unsafe {
                    (contents.as_ptr().cast::<u32>())
                        .add(t as usize)
                        .read_unaligned()
                };
                if got != want(t) {
                    wrong.push((t, got, want(t)));
                }
            }
            assert!(wrong.is_empty(), "{name}: words differ: {wrong:?}");
            println!("{name}: one threadgroup of [1, 1, {DEPTH}], Metal's simdgroup sums");
        });
    }

    /// The owned synthetic kernel: thread `group * size + pos` of a grid of `grid` threads stores
    /// its index in the grid plus 1 at that index, `x` fastest.
    const GRID_WORDS: &str = "; Owned synthetic kernel.
source_filename = \"grid_words.metal\"

define void @grid_words(ptr addrspace(1) %out, <3 x i32> %group, <3 x i32> %size, <3 x i32> %pos, <3 x i32> %grid) {
entry:
  %g = mul <3 x i32> %group, %size
  %q = add <3 x i32> %g, %pos
  %x = extractelement <3 x i32> %q, i64 0
  %y = extractelement <3 x i32> %q, i64 1
  %z = extractelement <3 x i32> %q, i64 2
  %gx = extractelement <3 x i32> %grid, i64 0
  %gy = extractelement <3 x i32> %grid, i64 1
  %zy = mul i32 %z, %gy
  %row = add i32 %zy, %y
  %rx = mul i32 %row, %gx
  %i = add i32 %rx, %x
  %v = add i32 %i, 1
  %w = zext i32 %i to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %w
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @grid_words, !1, !2}
!1 = !{}
!2 = !{!10, !11, !12, !13, !14}
!10 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}
!11 = !{i32 1, !\"air.threadgroup_position_in_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"group\"}
!12 = !{i32 2, !\"air.threads_per_threadgroup\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"size\"}
!13 = !{i32 3, !\"air.thread_position_in_threadgroup\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"pos\"}
!14 = !{i32 4, !\"air.threads_per_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"grid\"}
";

    /// Every thread of every workgroup writes its own word, whether its threadgroup runs as one
    /// row (`[1, 1, 96]`) or as its own workgroup (`[8, 8, 1]`, `[4, 4, 4]`), on every device.
    #[test]
    fn every_device_runs_every_thread_of_each_workgroup_deep_or_not() {
        let bitcode = match llvm_assemble_text(GRID_WORDS) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("grid_words: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function = library.function("grid_words").expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        // Each threadgroup, and its grid of two or more of them on each axis it spans.
        let cases: [([u32; 3], [u32; 3]); 3] = [
            ([1, 1, DEPTH], [2, 1, 2 * DEPTH]),
            ([8, 8, 1], [16, 16, 1]),
            ([4, 4, 4], [8, 8, 8]),
        ];
        each_device("deep_threadgroups grid_words", |name, _, device| {
            let state = device
                .new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let queue = device.new_command_queue(4).expect("a queue");
            for (threadgroup, grid) in cases {
                let words = grid.iter().product::<u32>();
                let len = u64::from(words * 4);
                let block = (device.take_block(MemoryClass::Shared, len)).expect("a Shared block");
                let output = (device.new_buffer(block.id, 0, len, Some(0))).expect("the output");
                let contents = output.contents().expect("a Shared buffer");
                // SAFETY: the buffer is alive and nothing else reaches its bytes.
                unsafe { std::ptr::write_bytes(contents.as_ptr(), 0xff, len as usize) };
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer
                    .compute_command_encoder(DispatchType::Serial)
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                encoder.set_buffer(Some(&output), 0, 0).expect("a buffer");
                encoder
                    .dispatch_threads(grid, threadgroup)
                    .unwrap_or_else(|error| panic!("{name}: the dispatch: {error}"));
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
                let wrong: Vec<u32> = (0..words)
                    .filter(|&i| {
                        // SAFETY: the word lies in the buffer, which is alive, and its work
                        // completed.
                        let got = unsafe {
                            (contents.as_ptr().cast::<u32>())
                                .add(i as usize)
                                .read_unaligned()
                        };
                        got != i + 1
                    })
                    .collect();
                assert!(
                    wrong.is_empty(),
                    "{name}: {grid:?} in threadgroups of {threadgroup:?}: {} of {words} words \
                     wrong, the first {:?}",
                    wrong.len(),
                    &wrong[..wrong.len().min(8)]
                );
                println!("{name}: {grid:?} in threadgroups of {threadgroup:?}: {words} words");
                drop(output);
                (device.release_block(block)).unwrap_or_else(|e| panic!("{name}: {e}"));
            }
        });
    }

    /// The words each create of `grid_words` makes its shader module of hold no specialization
    /// constant (`OpSpecConstant*`), and its workgroup's size is its `LocalSize` mode's literals:
    /// the threadgroup's own, or one row of its threads for the deep one.
    #[test]
    fn a_kernel_reaches_the_driver_with_its_workgroup_size_as_literals() {
        let bitcode = match llvm_assemble_text(GRID_WORDS) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("grid_words: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function = library.function("grid_words").expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        let cases: [([u32; 3], [u32; 3]); 3] = [
            ([1, 1, DEPTH], [DEPTH, 1, 1]),
            ([8, 8, 1], [8, 8, 1]),
            ([4, 4, 4], [4, 4, 4]),
        ];
        each_device("deep_threadgroups literal sizes", |name, _, device| {
            let state = device
                .new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let whole = &state.inner.compiled.whole;
            for (threadgroup, workgroup) in cases {
                let variant = crate::pipeline::Variant {
                    size: threadgroup,
                    lengths: crate::encoder::Lengths::zeros(0),
                    tiles: false,
                };
                let words = (whole.driver_words(&variant))
                    .unwrap_or_else(|error| panic!("{name}: {threadgroup:?}: {error}"));
                let mut at = 5;
                let mut sizes = Vec::new();
                while at < words.len() {
                    let (count, opcode) = ((words[at] >> 16) as usize, words[at] & 0xFFFF);
                    // OpSpecConstantTrue (48) through OpSpecConstantOp (52).
                    assert!(
                        !(48..=52).contains(&opcode),
                        "{name}: {threadgroup:?}: a specialization constant, opcode {opcode}"
                    );
                    // OpExecutionMode (16) LocalSize (17).
                    if opcode == 16 && words[at + 2] == 17 {
                        sizes.push(words[at + 3..at + 6].to_vec());
                    }
                    at += count;
                }
                assert_eq!(sizes, [workgroup.to_vec()], "{name}: {threadgroup:?}");
            }
        });
    }
}
