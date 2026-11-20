//! A threadgroup deeper than the floor's `maxComputeWorkGroupSize` z, under a device cfg on every
//! device of the cfg's kind: `simd_sum_k` (the owned fixture `engine/simd_ops.ll`, each thread's
//! `simd_sum(t) + simd_sum(float(t & 7)) * 100000` at its threadgroup index `t`) dispatched as one
//! threadgroup of `[1, 1, 96]`, which a kernel whose workgroup is its threadgroup runs as one row
//! (`SpecKey::DeepThreadgroup`) and a flattened one as its flat row, so the dispatch is within
//! every device's limits. Each simdgroup is 32 consecutive threads, as Metal's are: thread `t`
//! answers `1024 (t / 32) + 496 + 11200000`.
//!
//! `llvm-as` assembles the fixture; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        ComputePipelineDescriptor, DispatchType, FunctionDescriptor, Library, MemoryClass,
        StageSource,
    };
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

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
}
