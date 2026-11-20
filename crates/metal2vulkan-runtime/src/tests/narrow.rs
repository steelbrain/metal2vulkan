//! A narrow integer's mask over a signed clamp, under a device cfg on every device of the cfg's
//! kind: a kernel the translator compiles adds and subtracts two values sign-extended from 16 and
//! from 8 bits, clamps each result signed to exactly its width's range, truncates it and
//! zero-extends it. On every device the module clears the bits above each width with
//! `OpBitwiseAnd` by the width's mask, never `OpBitFieldUExtract(x, 0, N)`, which Intel's Windows
//! driver 101.2145 loses its device on (a driver's compiler dropped the mask after the clamp on
//! NVIDIA 615.71.09 on Ampere GA10x, which answered sign-extended words; 620.70.06 does not), and
//! every word is zero-extended.
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

    /// The edge values, as words: each lane reads its low 16 or 8 bits.
    const EDGES: [u32; 16] = [
        0x0000_0000,
        0x0000_0001,
        0x0000_7ffe,
        0x0000_7fff,
        0x0000_8000,
        0x0000_8001,
        0x0000_ffff,
        0x0000_fffe,
        0x0000_007f,
        0x0000_0080,
        0x0000_00ff,
        0x0000_0081,
        0x0000_4000,
        0x0000_c000,
        0x0000_0040,
        0x0000_ffc0,
    ];

    /// The threads: every pair of edges.
    const THREADS: u32 = 256;

    /// The words each thread writes.
    const WORDS: u32 = 4;

    /// The owned synthetic kernel: thread `gid` reads edges `gid & 15` and `gid >> 4`, and writes
    /// their 16-bit sum clamped by `llvm.smin` then `llvm.smax`, their 16-bit difference clamped
    /// by `llvm.smax` then `llvm.smin`, and their 8-bit sum and difference clamped by
    /// `air.clamp.s.i32`, each truncated to its width and zero-extended.
    const KERNEL: &str = r#"; Owned synthetic kernel.
source_filename = "narrow_clamps.metal"

declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.smax.i32(i32, i32)
declare i32 @air.clamp.s.i32(i32, i32, i32)

define void @narrow_clamps(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %gid) {
entry:
  %xi = and i32 %gid, 15
  %yi = lshr i32 %gid, 4
  %xw = zext i32 %xi to i64
  %yw = zext i32 %yi to i64
  %xp = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %xw
  %yp = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %yw
  %x = load i32, ptr addrspace(1) %xp, align 4
  %y = load i32, ptr addrspace(1) %yp, align 4
  %x16 = trunc i32 %x to i16
  %y16 = trunc i32 %y to i16
  %a16 = sext i16 %x16 to i32
  %b16 = sext i16 %y16 to i32
  %s16 = add nsw i32 %a16, %b16
  %s16a = call i32 @llvm.smin.i32(i32 %s16, i32 32767)
  %s16b = call i32 @llvm.smax.i32(i32 %s16a, i32 -32768)
  %s16t = trunc i32 %s16b to i16
  %s16z = zext i16 %s16t to i32
  %d16 = sub nsw i32 %a16, %b16
  %d16a = call i32 @llvm.smax.i32(i32 %d16, i32 -32768)
  %d16b = call i32 @llvm.smin.i32(i32 %d16a, i32 32767)
  %d16t = trunc i32 %d16b to i16
  %d16z = zext i16 %d16t to i32
  %x8 = trunc i32 %x to i8
  %y8 = trunc i32 %y to i8
  %a8 = sext i8 %x8 to i32
  %b8 = sext i8 %y8 to i32
  %s8 = add nsw i32 %a8, %b8
  %s8c = call i32 @air.clamp.s.i32(i32 %s8, i32 -128, i32 127)
  %s8t = trunc i32 %s8c to i8
  %s8z = zext i8 %s8t to i32
  %d8 = sub nsw i32 %a8, %b8
  %d8c = call i32 @air.clamp.s.i32(i32 %d8, i32 -128, i32 127)
  %d8t = trunc i32 %d8c to i8
  %d8z = zext i8 %d8t to i32
  %o = mul i32 %gid, 4
  %o0 = zext i32 %o to i64
  %p0 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %o0
  store i32 %s16z, ptr addrspace(1) %p0, align 4
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 1
  store i32 %d16z, ptr addrspace(1) %p1, align 4
  %p2 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 2
  store i32 %s8z, ptr addrspace(1) %p2, align 4
  %p3 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 3
  store i32 %d8z, ptr addrspace(1) %p3, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @narrow_clamps, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
"#;

    /// `x op y` of the low `bits` bits of two words, sign-extended, clamped to `bits` bits'
    /// range, its low `bits` bits zero-extended.
    fn clamped(x: u32, y: u32, bits: u32, sub: bool) -> u32 {
        let low = |w: u32| i64::from((w << (32 - bits)) as i32 >> (32 - bits));
        let (a, b) = (low(x), low(y));
        let half = 1i64 << (bits - 1);
        let v = if sub { a - b } else { a + b }.clamp(-half, half - 1);
        (v as u32) & ((1u32 << bits) - 1)
    }

    /// The four words of thread `gid`.
    fn want(gid: u32) -> [u32; 4] {
        let (x, y) = (EDGES[(gid & 15) as usize], EDGES[(gid >> 4) as usize]);
        [
            clamped(x, y, 16, false),
            clamped(x, y, 16, true),
            clamped(x, y, 8, false),
            clamped(x, y, 8, true),
        ]
    }

    /// The extracts of `words` (`OpBitFieldUExtract`), and its `OpBitwiseAnd`s by a constant
    /// 16-bit or 8-bit mask.
    fn masks(words: &[u32]) -> (usize, usize) {
        const CONSTANT: u32 = 43;
        const BITWISE_AND: u32 = 199;
        const BIT_FIELD_U_EXTRACT: u32 = 203;
        let mut narrow = std::collections::BTreeSet::new();
        let (mut extracts, mut ands) = (0, 0);
        for inst in spirv_instructions(words) {
            match inst[0] & 0xffff {
                CONSTANT if inst.len() == 4 && [0xffff, 0xff].contains(&inst[3]) => {
                    narrow.insert(inst[2]);
                }
                BIT_FIELD_U_EXTRACT => extracts += 1,
                BITWISE_AND if inst[3..5].iter().any(|v| narrow.contains(v)) => ands += 1,
                _ => {}
            }
        }
        (extracts, ands)
    }

    /// The kernel, translated, clears its narrow results' high bits by masks and answers every word
    /// zero-extended on every device.
    #[test]
    fn every_device_zero_extends_a_narrow_clamped_sum() {
        let bitcode = match llvm_assemble_text(KERNEL) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("narrow_clamps: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function = library.function("narrow_clamps").expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        let out_len = u64::from(THREADS * WORDS * 4);
        each_device("narrow_clamps", |name, _, device| {
            let pipeline = (device.core().translator)
                .compile_compute_pipeline(&descriptor, &Budget::standard())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (extracts, ands) = masks(&pipeline.stage.spirv);
            assert!(ands >= 4, "{name}: {ands} masks by a narrow width's bits");
            assert_eq!(extracts, 0, "{name}: {extracts} extracts");
            let state = device
                .new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let block = device
                .take_block(MemoryClass::Shared, 2 * out_len)
                .expect("a Shared block");
            let input = device
                .new_buffer(block.id, 0, 64, Some(0))
                .expect("the input");
            let output = device
                .new_buffer(block.id, out_len, out_len, Some(0))
                .expect("the output");
            let edges: Vec<u8> = EDGES.iter().flat_map(|w| w.to_le_bytes()).collect();
            let contents = input.contents().expect("a Shared buffer");
            // SAFETY: the 64 bytes lie in the buffer, which is alive, and no GPU work uses it.
            unsafe {
                std::ptr::copy_nonoverlapping(edges.as_ptr(), contents.as_ptr(), edges.len())
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
                .dispatch_threads([THREADS, 1, 1], [64, 1, 1])
                .expect("a dispatch");
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let contents = output.contents().expect("a Shared buffer");
            let mut wrong = 0;
            let mut first = None;
            for gid in 0..THREADS {
                for (k, w) in want(gid).into_iter().enumerate() {
                    let at = (gid * WORDS) as usize + k;
                    // SAFETY: the word lies in the buffer, which is alive, and its work completed.
                    let got = unsafe { contents.as_ptr().cast::<u32>().add(at).read_unaligned() };
                    if got != w {
                        wrong += 1;
                        first.get_or_insert((gid, k, got, w));
                    }
                }
            }
            assert_eq!(wrong, 0, "{name}: words differ, the first {first:x?}");
            println!(
                "{name}: narrow clamps zero-extended over {THREADS} threads, {} words",
                THREADS * WORDS
            );
        });
    }
}
