//! Products by a constant zero of either sign, under a device cfg on every device of the cfg's
//! kind: a kernel the translator compiles multiplies one value by `+0.0` and by `-0.0`, and the
//! value splat by `(+0.0, -0.0, 2.0, -0.0)`. On every device the module multiplies by `+0.0` where
//! a factor has `-0.0` and negates, so no product it holds has a `-0.0` factor beside a `+0.0` one,
//! which a driver may compile as one multiply (ANV's `brw` backend, Mesa 26.2.6, stored the first
//! of two such products for both), and every product answers IEEE's bits: a zero of the sign of
//! the factors' product.
//!
//! `llvm-as` assembles the kernel; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::tests::spirv_instructions;
    use crate::{
        ComputePipelineDescriptor, DispatchType, FunctionDescriptor, Library, MemoryClass,
        StageSource,
    };
    use metal2vulkan::Budget;
    use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};
    use std::collections::BTreeSet;

    /// The inputs, one a thread: no NaN, no infinity (whose product by a zero is a NaN, its sign
    /// the device's) and no denormal (which a device may flush before the product by 2).
    const INPUTS: [f32; 16] = [
        -2.5,
        2.5,
        -0.0,
        0.0,
        1.0,
        -1.0,
        3.0,
        -7.0,
        1e-30,
        -1e-30,
        1e-20,
        -1e-20,
        f32::MAX,
        -f32::MAX,
        0.5,
        -0.5,
    ];

    /// The words each thread writes: `x * +0.0`, `x * -0.0`, then the four lanes of `x` splat
    /// times [`FACTOR`].
    const WORDS: usize = 6;

    /// The vector factor.
    const FACTOR: [f32; 4] = [0.0, -0.0, 2.0, -0.0];

    /// The owned synthetic kernel: thread `gid` reads `in[gid]` and writes its six products from
    /// `out[6 * gid]`.
    const KERNEL: &str = r#"; Owned synthetic kernel.
source_filename = "signed_zeros.metal"

define void @signed_zeros(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %gid) {
entry:
  %w = zext i32 %gid to i64
  %xp = getelementptr inbounds float, ptr addrspace(1) %in, i64 %w
  %x = load float, ptr addrspace(1) %xp, align 4
  %p = fmul float %x, 0.000000e+00
  %n = fmul float %x, -0.000000e+00
  %v0 = insertelement <4 x float> poison, float %x, i64 0
  %v = shufflevector <4 x float> %v0, <4 x float> poison, <4 x i32> zeroinitializer
  %q = fmul <4 x float> %v, <float 0.000000e+00, float -0.000000e+00, float 2.000000e+00, float -0.000000e+00>
  %q0 = extractelement <4 x float> %q, i64 0
  %q1 = extractelement <4 x float> %q, i64 1
  %q2 = extractelement <4 x float> %q, i64 2
  %q3 = extractelement <4 x float> %q, i64 3
  %o = mul i64 %w, 6
  %o0 = getelementptr inbounds float, ptr addrspace(1) %out, i64 %o
  store float %p, ptr addrspace(1) %o0, align 4
  %o1 = getelementptr inbounds float, ptr addrspace(1) %o0, i64 1
  store float %n, ptr addrspace(1) %o1, align 4
  %o2 = getelementptr inbounds float, ptr addrspace(1) %o0, i64 2
  store float %q0, ptr addrspace(1) %o2, align 4
  %o3 = getelementptr inbounds float, ptr addrspace(1) %o0, i64 3
  store float %q1, ptr addrspace(1) %o3, align 4
  %o4 = getelementptr inbounds float, ptr addrspace(1) %o0, i64 4
  store float %q2, ptr addrspace(1) %o4, align 4
  %o5 = getelementptr inbounds float, ptr addrspace(1) %o0, i64 5
  store float %q3, ptr addrspace(1) %o5, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @signed_zeros, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
"#;

    /// The six words of input `x`, IEEE's products.
    fn want(x: f32) -> [u32; WORDS] {
        let mut w = [(x * 0.0).to_bits(), (x * -0.0).to_bits(), 0, 0, 0, 0];
        for (lane, factor) in FACTOR.into_iter().enumerate() {
            w[2 + lane] = (x * factor).to_bits();
        }
        w
    }

    /// The products of `words` (`OpFMul`, `OpVectorTimesScalar`) with a constant factor that has
    /// a `-0.0` component, and its `OpFNegate`s.
    fn negative_zero_factors(words: &[u32]) -> (usize, usize) {
        const TYPE_FLOAT: u32 = 22;
        const CONSTANT: u32 = 43;
        const CONSTANT_COMPOSITE: u32 = 44;
        const F_NEGATE: u32 = 127;
        const F_MUL: u32 = 133;
        const VECTOR_TIMES_SCALAR: u32 = 142;
        let (mut floats, mut negative) = (BTreeSet::new(), BTreeSet::new());
        let (mut products, mut negations) = (0, 0);
        for inst in spirv_instructions(words) {
            match inst[0] & 0xffff {
                TYPE_FLOAT if inst[2] == 32 => {
                    floats.insert(inst[1]);
                }
                CONSTANT if floats.contains(&inst[1]) && inst[3] == 0x8000_0000 => {
                    negative.insert(inst[2]);
                }
                CONSTANT_COMPOSITE if inst[3..].iter().any(|c| negative.contains(c)) => {
                    negative.insert(inst[2]);
                }
                F_MUL | VECTOR_TIMES_SCALAR if inst[3..5].iter().any(|v| negative.contains(v)) => {
                    products += 1;
                }
                F_NEGATE => negations += 1,
                _ => {}
            }
        }
        (products, negations)
    }

    /// The kernel, translated for every device, holds no product by a `-0.0` factor, and answers
    /// IEEE's products bit for bit.
    #[test]
    fn every_device_answers_products_by_zeros_of_either_sign() {
        let bitcode = match llvm_assemble_text(KERNEL) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("signed_zeros: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function = library.function("signed_zeros").expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        let threads = INPUTS.len() as u32;
        let out_len = (INPUTS.len() * WORDS * 4) as u64;
        each_device("signed_zeros", |name, _, device| {
            let pipeline = (device.core().translator)
                .compile_compute_pipeline(&descriptor, &Budget::standard())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (products, negations) = negative_zero_factors(&pipeline.stage.spirv);
            assert_eq!(products, 0, "{name}: products by a -0.0 factor");
            assert!(negations > 0, "{name}: no negated product");
            let state = device
                .new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let block = device
                .take_block(MemoryClass::Shared, 64 + out_len)
                .expect("a Shared block");
            let input = device
                .new_buffer(block.id, 0, 64, Some(0))
                .expect("the input");
            let output = device
                .new_buffer(block.id, 64, out_len, Some(0))
                .expect("the output");
            let bytes: Vec<u8> = INPUTS.iter().flat_map(|x| x.to_le_bytes()).collect();
            let contents = input.contents().expect("a Shared buffer");
            // SAFETY: the 64 bytes lie in the buffer, which is alive, and no GPU work uses it.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len())
            };
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer
                .compute_command_encoder(DispatchType::Serial)
                .expect("an encoder");
            encoder.set_compute_pipeline_state(&state).expect("a state");
            encoder.set_buffer(Some(&input), 0, 0).expect("a buffer");
            encoder.set_buffer(Some(&output), 0, 1).expect("a buffer");
            encoder
                .dispatch_threads([threads, 1, 1], [16, 1, 1])
                .expect("a dispatch");
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let contents = output.contents().expect("a Shared buffer");
            let mut wrong = Vec::new();
            for (gid, x) in INPUTS.into_iter().enumerate() {
                for (k, w) in want(x).into_iter().enumerate() {
                    // SAFETY: the word lies in the buffer, which is alive, and its work completed.
                    let got = unsafe {
                        (contents.as_ptr().cast::<u32>())
                            .add(gid * WORDS + k)
                            .read_unaligned()
                    };
                    if got != w {
                        wrong.push((x, k, got, w));
                    }
                }
            }
            assert!(wrong.is_empty(), "{name}: words differ: {wrong:x?}");
            println!(
                "{name}: {} products by zeros of either sign, IEEE's bits",
                INPUTS.len() * WORDS
            );
        });
    }
}
