//! A fragment function whose group operations read other lanes (`floor`'s `QUAD_CONTROL`), under a
//! device cfg on every device of the cfg's kind: where device creation enabled
//! `VK_KHR_shader_quad_control`, the translator's profile has `shaderQuadControl`, the state is
//! made and the draw answers Metal's values; elsewhere creation refuses the state, naming
//! `SubgroupOperations`.
//!
//! `fragment_quad_bitwise` (engine `gsm2_quad_ops`) writes its quad's AND, OR and XOR of a word of
//! each pixel's position, drawn by a triangle whose edge cuts the diagonal quads, so each of those
//! quads runs one helper. Metal runs the helpers, and they take part (`lower::simd`'s
//! `full_quads`), so every drawn pixel reads all four of its quad's words.
//!
//! `quad_sum_after_a_return` returns early from the top two rows, then sums a sample of its own
//! texel over a loop whose trip count it loads, the sample at implicit LOD and the running sum
//! carried through a `half`, and writes its quad's sum of that: the shape of a fragment whose
//! creation once killed a driver's compiler (an early return before a loop of implicit-LOD
//! samples and `float`-`half` round trips, under `MaximallyReconvergesKHR`), with a quad
//! reduction after the loop. Every device of the cfg's kind with quad control makes it and draws
//! each pixel's sums.
//!
//! `quad_votes` writes its quad's `quad_any` of "the pixel is right of the diagonal" and
//! `quad_all` of "the pixel is on or left of it", drawn by `fragment_quad_bitwise`'s triangle: in a
//! diagonal quad only the helper is right of the diagonal, so `quad_any` holds and `quad_all` does
//! not there, and the other way round in a quad the triangle covers whole.
//!
//! The `QUAD_READERS` functions write the other quad operations, which read which of their quad's
//! lanes are active or read another lane's value: `quad_ballot`, `quad_active_threads_mask`,
//! `quad_is_first`, `quad_broadcast_first`, the prefix sums, and the shuffles and broadcast by
//! lane. Each is drawn by that triangle and by the one across the same edge, which covers one pixel
//! of each diagonal quad and runs its other three, quad lane 0 among them, as helpers. Each runs
//! its operations in a block every invocation executes, where each quad is whole, but
//! `quad_masks_divergent`, which runs `quad_masks`' under a branch on the pixel's `x` being odd,
//! so quad lanes 1 and 3 take part, helpers included.
//!
//! `fragment_quad_bitwise` drawn over the whole of a 7 by 7 target reads the helpers of the quads
//! the target cuts, past its last column and row.
//!
//! `llvm-as` assembles the functions; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{
        descriptor, examples, floats, library, pass, read, run, target, vertices, Buffers,
        CONSTEXPR, SIDE,
    };
    use crate::tests::spirv_instructions;
    use crate::{
        Device, Error, FunctionDescriptor, IndexClass, Library, NumberedSampler, PixelFormat,
        PrimitiveType, Region, RenderPipelineDescriptor, RenderPipelineState, StageSource,
        StorageMode, TextureDescriptor, TextureType, TextureUsage,
    };
    use metal2vulkan::device::VkFeature;
    use metal2vulkan::error::Feature;
    use metal2vulkan::Budget;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble_text, AssembleError};

    /// SPIR-V's `OpEntryPoint`, `OpLoopMerge`, `OpImageSampleImplicitLod`, `OpReturn`,
    /// `OpReturnValue`, `OpGroupNonUniformBallot`, `OpGroupNonUniformQuadAllKHR` and
    /// `OpGroupNonUniformQuadAnyKHR`.
    const OP_ENTRY_POINT: u32 = 15;
    const OP_LOOP_MERGE: u32 = 246;
    const OP_SAMPLE_IMPLICIT_LOD: u32 = 87;
    const OP_RETURN: u32 = 253;
    const OP_RETURN_VALUE: u32 = 254;
    const OP_BALLOT: u32 = 339;
    const OP_QUAD_ALL: u32 = 5110;
    const OP_QUAD_ANY: u32 = 5111;

    /// The loop's trip count, the first word of the fragment's buffer 0.
    const TRIPS: u32 = 3;

    /// `quad_sum_after_a_return`: pixel `(x, y)` of the top two rows returns `(-1, -1, -1, 1)`;
    /// every other pixel adds its texel's red channel times `i + 1` for each trip `i` of `TRIPS`,
    /// the sum carried through a `half` on each trip, and returns `(quad_sum(sum), sum, 0, 1)`.
    const QUAD_SUM_AFTER_A_RETURN: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601020489, i64 0], align 8

define <4 x float> @quad_sum_after_a_return(<4 x float> %pos, ptr addrspace(1) %t, ptr addrspace(2) %np) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %p = call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %y = extractelement <2 x i32> %p, i64 1
  %top = icmp ult i32 %y, 2
  br i1 %top, label %early, label %pre
early:
  ret <4 x float> <float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 1.000000e+00>
pre:
  %n = load i32, ptr addrspace(2) %np, align 4
  %uv = fmul <2 x float> %xy, <float 1.250000e-01, float 1.250000e-01>
  br label %loop
loop:
  %i = phi i32 [ 0, %pre ], [ %in, %loop ]
  %acc = phi float [ 0.000000e+00, %pre ], [ %back, %loop ]
  %s = call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x half>, i8 } %s, 0
  %h = extractelement <4 x half> %v, i64 0
  %r = fpext half %h to float
  %in = add i32 %i, 1
  %k = uitofp i32 %in to float
  %rk = fmul float %r, %k
  %sum = fadd float %acc, %rk
  %narrow = fptrunc float %sum to half
  %back = fpext half %narrow to float
  %more = icmp ult i32 %in, %n
  br i1 %more, label %loop, label %done
done:
  %q = call float @air.quad_sum.f32(float %back)
  %o0 = insertelement <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, float %q, i64 0
  %o = insertelement <4 x float> %o0, float %back, i64 1
  ret <4 x float> %o
}

declare <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float>)
declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare float @air.quad_sum.f32(float)

!air.fragment = !{!0}
!air.sampler_states = !{!7}
!0 = !{ptr @quad_sum_after_a_return, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5, !6}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"t"}
!6 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"n"}
!7 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;

    /// `quad_votes`: pixel `(x, y)` returns `(quad_any(x > y), quad_all(x <= y), 0, 1)`.
    const QUAD_VOTES: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <4 x i32> @quad_votes(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %p = call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %p, i64 0
  %y = extractelement <2 x i32> %p, i64 1
  %right = icmp ugt i32 %x, %y
  %left = icmp ule i32 %x, %y
  %any = call i1 @air.quad_any(i1 %right)
  %all = call i1 @air.quad_all(i1 %left)
  %a = zext i1 %any to i32
  %b = zext i1 %all to i32
  %o0 = insertelement <4 x i32> <i32 0, i32 0, i32 0, i32 1>, i32 %a, i64 0
  %o = insertelement <4 x i32> %o0, i32 %b, i64 1
  ret <4 x i32> %o
}

declare <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float>)
declare i1 @air.quad_any(i1)
declare i1 @air.quad_all(i1)

!air.fragment = !{!0}
!0 = !{ptr @quad_votes, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"uint4"}
!3 = !{!4}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
"#;

    /// The functions `MASKS_AND_SCANS` and `SHUFFLES` name. Pixel `(x, y)` takes the word
    /// `(x * 7 + y * 13) ^ 0x5a` (`word`), the predicate `x > y`, and its quad lane
    /// `x & 1 | (y & 1) << 1`.
    const QUAD_READERS: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <4 x i32> @quad_masks(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %p = call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %p, i64 0
  %y = extractelement <2 x i32> %p, i64 1
  %right = icmp ugt i32 %x, %y
  %x7 = mul i32 %x, 7
  %y13 = mul i32 %y, 13
  %s = add i32 %x7, %y13
  %w = xor i32 %s, 90
  %b = call i16 @air.quad_ballot(i1 %right)
  %m = call i16 @air.quad_active_threads_mask()
  %f = call i1 @air.quad_is_first()
  %bf = call i32 @air.quad_broadcast_first.u.i32(i32 %w)
  %b4 = zext i16 %b to i32
  %m4 = zext i16 %m to i32
  %f4 = zext i1 %f to i32
  %o0 = insertelement <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i32 %b4, i64 0
  %o1 = insertelement <4 x i32> %o0, i32 %m4, i64 1
  %o2 = insertelement <4 x i32> %o1, i32 %f4, i64 2
  %o = insertelement <4 x i32> %o2, i32 %bf, i64 3
  ret <4 x i32> %o
}

define <4 x i32> @quad_scans(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %p = call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %p, i64 0
  %y = extractelement <2 x i32> %p, i64 1
  %x7 = mul i32 %x, 7
  %y13 = mul i32 %y, 13
  %s = add i32 %x7, %y13
  %w = xor i32 %s, 90
  %ex = call i32 @air.quad_prefix_exclusive_sum.u.i32(i32 %w)
  %in = call i32 @air.quad_prefix_inclusive_sum.u.i32(i32 %w)
  %o0 = insertelement <4 x i32> <i32 0, i32 0, i32 0, i32 1>, i32 %ex, i64 0
  %o = insertelement <4 x i32> %o0, i32 %in, i64 1
  ret <4 x i32> %o
}

define <4 x i32> @quad_reads(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %p = call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %p, i64 0
  %y = extractelement <2 x i32> %p, i64 1
  %x7 = mul i32 %x, 7
  %y13 = mul i32 %y, 13
  %s = add i32 %x7, %y13
  %w = xor i32 %s, 90
  %xl = and i32 %x, 1
  %yl = and i32 %y, 1
  %yl2 = shl i32 %yl, 1
  %q = or i32 %xl, %yl2
  %far = xor i32 %q, 3
  %far16 = trunc i32 %far to i16
  %sh = call i32 @air.quad_shuffle.u.i32(i32 %w, i16 %far16)
  %bc = call i32 @air.quad_broadcast.u.i32(i32 %w, i16 1)
  %o0 = insertelement <4 x i32> <i32 0, i32 0, i32 0, i32 1>, i32 %sh, i64 0
  %o = insertelement <4 x i32> %o0, i32 %bc, i64 1
  ret <4 x i32> %o
}

define <4 x i32> @quad_moves(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %p = call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %p, i64 0
  %y = extractelement <2 x i32> %p, i64 1
  %x7 = mul i32 %x, 7
  %y13 = mul i32 %y, 13
  %s = add i32 %x7, %y13
  %w = xor i32 %s, 90
  %xr = call i32 @air.quad_shuffle_xor.u.i32(i32 %w, i16 1)
  %dn = call i32 @air.quad_shuffle_down.u.i32(i32 %w, i16 1)
  %up = call i32 @air.quad_shuffle_up.u.i32(i32 %w, i16 1)
  %rd = call i32 @air.quad_shuffle_rotate_down.u.i32(i32 %w, i16 1)
  %o0 = insertelement <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i32 %xr, i64 0
  %o1 = insertelement <4 x i32> %o0, i32 %dn, i64 1
  %o2 = insertelement <4 x i32> %o1, i32 %up, i64 2
  %o = insertelement <4 x i32> %o2, i32 %rd, i64 3
  ret <4 x i32> %o
}

define <4 x i32> @quad_masks_divergent(<4 x float> %pos) {
entry:
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %p = call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %x = extractelement <2 x i32> %p, i64 0
  %y = extractelement <2 x i32> %p, i64 1
  %odd = and i32 %x, 1
  %in = icmp ne i32 %odd, 0
  br i1 %in, label %arm, label %out
arm:
  %right = icmp ugt i32 %x, %y
  %x7 = mul i32 %x, 7
  %y13 = mul i32 %y, 13
  %s = add i32 %x7, %y13
  %w = xor i32 %s, 90
  %b = call i16 @air.quad_ballot(i1 %right)
  %m = call i16 @air.quad_active_threads_mask()
  %f = call i1 @air.quad_is_first()
  %bf = call i32 @air.quad_broadcast_first.u.i32(i32 %w)
  %b4 = zext i16 %b to i32
  %m4 = zext i16 %m to i32
  %f4 = zext i1 %f to i32
  %o0 = insertelement <4 x i32> <i32 0, i32 0, i32 0, i32 0>, i32 %b4, i64 0
  %o1 = insertelement <4 x i32> %o0, i32 %m4, i64 1
  %o2 = insertelement <4 x i32> %o1, i32 %f4, i64 2
  %o = insertelement <4 x i32> %o2, i32 %bf, i64 3
  br label %out
out:
  %r = phi <4 x i32> [ <i32 7, i32 7, i32 7, i32 7>, %entry ], [ %o, %arm ]
  ret <4 x i32> %r
}

declare <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float>)
declare i16 @air.quad_ballot(i1)
declare i16 @air.quad_active_threads_mask()
declare i1 @air.quad_is_first()
declare i32 @air.quad_broadcast_first.u.i32(i32)
declare i32 @air.quad_prefix_exclusive_sum.u.i32(i32)
declare i32 @air.quad_prefix_inclusive_sum.u.i32(i32)
declare i32 @air.quad_shuffle.u.i32(i32, i16)
declare i32 @air.quad_broadcast.u.i32(i32, i16)
declare i32 @air.quad_shuffle_xor.u.i32(i32, i16)
declare i32 @air.quad_shuffle_down.u.i32(i32, i16)
declare i32 @air.quad_shuffle_up.u.i32(i32, i16)
declare i32 @air.quad_shuffle_rotate_down.u.i32(i32, i16)

!air.fragment = !{!0, !5, !6, !7, !8}
!0 = !{ptr @quad_masks, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"uint4"}
!3 = !{!4}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!5 = !{ptr @quad_scans, !1, !3}
!6 = !{ptr @quad_reads, !1, !3}
!7 = !{ptr @quad_moves, !1, !3}
!8 = !{ptr @quad_masks_divergent, !1, !3}
"#;

    /// The functions of `QUAD_READERS` that count which of their quad's lanes are active or read
    /// the first active one, or sum over the active lanes below, and the operation each word of
    /// their target holds, in order (a word past them is a constant).
    const MASKS_AND_SCANS: [(&str, &[&str]); 3] = [
        (
            "quad_masks",
            &[
                "quad_ballot",
                "quad_active_threads_mask",
                "quad_is_first",
                "quad_broadcast_first",
            ],
        ),
        (
            "quad_masks_divergent",
            &[
                "quad_ballot",
                "quad_active_threads_mask",
                "quad_is_first",
                "quad_broadcast_first",
            ],
        ),
        (
            "quad_scans",
            &["quad_prefix_exclusive_sum", "quad_prefix_inclusive_sum"],
        ),
    ];

    /// The functions of `QUAD_READERS` that read another lane of their quad by its index, and the
    /// operation each word of their target holds.
    const SHUFFLES: [(&str, &[&str]); 2] = [
        ("quad_reads", &["quad_shuffle", "quad_broadcast"]),
        (
            "quad_moves",
            &[
                "quad_shuffle_xor",
                "quad_shuffle_down",
                "quad_shuffle_up",
                "quad_shuffle_rotate_down",
            ],
        ),
    ];
    /// The word `fragment_quad_bitwise` takes of pixel `[x, y]`.
    fn word([x, y]: [u32; 2]) -> u32 {
        (x * 7 + y * 13) ^ 0x5a
    }

    /// The clip-space triangle (y up) that covers the pixels `[x, y]` with `x <= y`: its edge
    /// `x + y = 0.125` passes no pixel centre and cuts each diagonal quad at its top right pixel,
    /// quad lane 1, which runs as a helper.
    const ON_OR_LEFT: [f32; 6] = [-1.0, -1.0, 1.125, -1.0, -1.0, 1.125];

    /// The triangle on the other side of the same edge, which covers the pixels with `x > y`: of
    /// each diagonal quad it covers the top right pixel alone, and quad lanes 0, 2 and 3 run as
    /// helpers.
    const RIGHT: [f32; 6] = [1.25, -1.125, 1.25, 1.25, -1.125, 1.25];

    /// Draw `state` into an `RGBA32Uint` target by the clip-space `triangle` (`ON_OR_LEFT` or
    /// `RIGHT`), and read the target back.
    fn diagonal(
        name: &str,
        device: &Device,
        state: &RenderPipelineState,
        triangle: &[f32; 6],
    ) -> Vec<u8> {
        let mut buffers = Buffers::new(device);
        let positions = buffers.of(device, &floats(triangle));
        let colors = buffers.of(device, &floats(&[0.0; 12]));
        let color = target(device, PixelFormat::RGBA32Uint, TextureUsage::RENDER_TARGET);
        let queue = device.new_command_queue(4).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let ok =
            |result: Result<(), Error>| result.unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        ok(encoder.set_render_pipeline_state(state));
        ok(encoder.set_vertex_buffer(Some(&positions), 0, 0));
        ok(encoder.set_vertex_buffer(Some(&colors), 0, 1));
        ok(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
        encoder.end_encoding();
        run(name, &command_buffer);
        read(&color, 16)
    }

    /// The opcode of each instruction of `d`'s fragment stage as `device`'s translator makes it.
    fn fragment_ops(name: &str, device: &Device, d: &RenderPipelineDescriptor) -> Vec<u32> {
        let pipeline = (device.core().translator)
            .compile_render_pipeline(d, &Budget::standard())
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let words = (pipeline.stages.iter())
            .map(|stage| &stage.spirv)
            .find(|words| {
                spirv_instructions(words)
                    .iter()
                    .any(|i| i[0] & 0xffff == OP_ENTRY_POINT && i.get(1) == Some(&4))
            })
            .expect("a fragment stage");
        (spirv_instructions(words).iter())
            .map(|i| i[0] & 0xffff)
            .collect()
    }

    /// The first `n` words of pixel `[x, y]` of a target `diagonal` read.
    fn words(texels: &[u8], [x, y]: [u32; 2], n: usize) -> Vec<u32> {
        let at = ((y * SIDE + x) * 16) as usize;
        (texels[at..at + 4 * n].as_chunks::<4>().0.iter())
            .map(|w| u32::from_le_bytes(*w))
            .collect()
    }

    /// Pixel `p`'s quad lane, in Metal's order: top left 0, top right 1, bottom left 2, bottom
    /// right 3.
    fn quad_lane([x, y]: [u32; 2]) -> u32 {
        (x & 1) | ((y & 1) << 1)
    }

    /// The pixel at quad lane `j` of pixel `p`'s quad.
    fn mate([x, y]: [u32; 2], j: u32) -> [u32; 2] {
        [(x & !1) | (j & 1), (y & !1) | (j >> 1)]
    }

    /// What a `QUAD_READERS` function writes at pixel `p` when the lanes of `active` (bit `j` quad
    /// lane `j`) take part: Metal's answer where all four do, and where only the covered ones do,
    /// the answer of a device that leaves the helpers out, a read of a lane that takes no part
    /// reading the calling lane as `lower::simd` reads it. Quad lane `j` holds the word of its
    /// pixel and the predicate "x > y" of it.
    fn quad_answer(function: &str, p: [u32; 2], active: u32) -> [u32; 4] {
        let q = quad_lane(p);
        let on = |j: &u32| (active >> j) & 1 == 1;
        let w = |j: u32| word(mate(p, j));
        let read = |j: u32| if on(&j) { w(j) } else { w(q) };
        let right = |j: u32| {
            let [x, y] = mate(p, j);
            u32::from(x > y)
        };
        let first = active.trailing_zeros();
        match function {
            // Its pixels of even `x` take no part and write 7s; those of odd `x`, quad lanes 1 and
            // 3, run `quad_masks` among themselves.
            "quad_masks_divergent" if p[0] & 1 == 0 => [7; 4],
            "quad_masks_divergent" => quad_answer("quad_masks", p, active & 0b1010),
            "quad_masks" => [
                (0..4).filter(on).map(|j| right(j) << j).sum(),
                active,
                u32::from(q == first),
                w(first),
            ],
            "quad_scans" => [
                (0..q).filter(on).map(w).sum(),
                (0..=q).filter(on).map(w).sum(),
                0,
                1,
            ],
            "quad_reads" => [read(q ^ 3), read(1), 0, 1],
            _ => [
                read(q ^ 1),
                if q < 3 { read(q + 1) } else { w(q) },
                if q > 0 { read(q - 1) } else { w(q) },
                read((q + 1) & 3),
            ],
        }
    }

    /// How many covered pixels a word was read at, how many of those were wrong, and how many of
    /// those held the answer with the quad's helpers left out.
    #[derive(Clone, Copy, Default)]
    struct Tally {
        read: u32,
        wrong: u32,
        left_out: u32,
    }

    /// Make each of `functions` (from `QUAD_READERS`) on every device with quad control, check its
    /// fragment stage's opcodes with `stage`, draw it by `ON_OR_LEFT` and by `RIGHT`, and check
    /// each covered pixel's words against Metal's (`quad_answer`), printing each operation's count
    /// of wrong pixels per triangle. The helpers of the quads a triangle cuts are active lanes of
    /// their quad: MSL 4.1 section 6.10.2 has "An active thread is a thread that is executing" and
    /// "Helper threads may also be active and inactive. For example, if a helper thread finishes
    /// executing, it becomes an inactive helper thread"; section 6.10.3 has "A quad-group function
    /// is a SIMD-group function ... with an execution width of 4. The active and inactive thread
    /// terminology is the same", and `quad_active_threads_mask` "sets the bits that represent
    /// active threads to 1". These functions neither return nor discard before their calls, so
    /// every helper is executing at each.
    fn quad_readers(test: &str, functions: &[(&str, &[&str])], stage: impl Fn(&str, &str, &[u32])) {
        let readers = match llvm_assemble_text(QUAD_READERS) {
            Ok(bitcode) => Library::from_air(&bitcode).expect("the library"),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("{test}: {error:?}"),
        };
        let Some(examples) = examples() else {
            return;
        };
        each_device(test, |name, _, device| {
            if !(device.core().translator.profile().enabled().features)
                .contains(&VkFeature::ShaderQuadControl)
            {
                println!("{test}: {name}: no quad control; not made");
                return;
            }
            let mut wrong = Vec::new();
            for &(function, ops) in functions {
                let mut d = descriptor(&examples, "colored_fragment", PixelFormat::RGBA32Uint);
                d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
                    readers.function(function).expect("the function"),
                )));
                stage(name, function, &fragment_ops(name, &device, &d));
                let state = (device
                    .new_render_pipeline_state(&d, crate::tests::no_constexpr_samplers))
                .unwrap_or_else(|error| panic!("{name}: {function}: {error}"));
                for (side, triangle) in [("x <= y", &ON_OR_LEFT), ("x > y", &RIGHT)] {
                    let covers = |[x, y]: [u32; 2]| (x > y) == (side == "x > y");
                    let texels = diagonal(name, &device, &state, triangle);
                    let mut tallies = vec![Tally::default(); ops.len()];
                    for p in (0..SIDE).flat_map(|y| (0..SIDE).map(move |x| [x, y])) {
                        if !covers(p) {
                            continue;
                        }
                        let covered = (0..4)
                            .filter(|&j| covers(mate(p, j)))
                            .fold(0, |m, j| m | (1 << j));
                        let (metal, left_out) = (
                            quad_answer(function, p, 15),
                            quad_answer(function, p, covered),
                        );
                        let got = words(&texels, p, 4);
                        for (i, t) in tallies.iter_mut().enumerate() {
                            t.read += 1;
                            if got[i] != metal[i] {
                                t.wrong += 1;
                                t.left_out += u32::from(got[i] == left_out[i]);
                            }
                        }
                    }
                    for (op, t) in ops.iter().zip(tallies) {
                        println!(
                            "{test}: {name}: {op} drawn over {side}: {} of {} pixels wrong, {} \
                             of them the answer with the helpers left out",
                            t.wrong, t.read, t.left_out
                        );
                        if t.wrong > 0 {
                            wrong.push(format!("{op} over {side}"));
                        }
                    }
                }
            }
            assert!(wrong.is_empty(), "{name}: wrong: {wrong:?}");
        });
    }

    /// Where Vulkan offers `shaderQuadControl`, a fragment function's quad operations read their
    /// quad's helpers, as Metal's do; elsewhere the state is refused naming the feature.
    #[test]
    fn every_device_runs_quads_whole_where_it_offers_quad_control() {
        let (Some(examples), Some(quads)) =
            (examples(), library(&engine_dir().join("gsm2_quad_ops.ll")))
        else {
            return;
        };
        let mut d = descriptor(&examples, "colored_fragment", PixelFormat::RGBA32Uint);
        d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
            quads
                .function("fragment_quad_bitwise")
                .expect("the function"),
        )));
        each_device("quad_control", |name, _, device| {
            // The translator's profile has the feature only where device creation enabled its
            // extension.
            let listed = (device.extensions().iter())
                .any(|enabled| enabled.as_c_str() == ash::khr::shader_quad_control::NAME);
            let core = device.core();
            let enabled = &core.translator.profile().enabled().features;
            let on = enabled.contains(&VkFeature::ShaderQuadControl);
            assert!(
                listed || !on,
                "{name}: shaderQuadControl without its extension"
            );
            let state =
                match device.new_render_pipeline_state(&d, crate::tests::no_constexpr_samplers) {
                    Ok(state) => state,
                    Err(Error::Translation {
                        error: metal2vulkan::error::Error::Unsupported(u),
                        ..
                    }) if !on => {
                        assert_eq!(u.feature, Feature::SubgroupOperations, "{name}: {u:?}");
                        println!("quad_control: {name}: no quad control; the state is refused");
                        return;
                    }
                    Err(error) => panic!("{name}: {error}"),
                };
            assert!(on, "{name}: a state made without quad control");
            let texels = diagonal(name, &device, &state, &ON_OR_LEFT);
            let mut drawn = 0;
            for y in 0..SIDE {
                for x in (0..SIDE).filter(|&x| x <= y) {
                    let quad = [[0, 0], [1, 0], [0, 1], [1, 1]]
                        .map(|[dx, dy]| word([(x & !1) + dx, (y & !1) + dy]));
                    let want = [
                        quad.iter().fold(!0, |all, w| all & (w | 0x100)),
                        quad.iter().fold(0, |any, w| any | w),
                        quad.iter().fold(0, |odd, w| odd ^ w),
                    ];
                    assert_eq!(words(&texels, [x, y], 3), want, "{name}: pixel [{x}, {y}]");
                    drawn += 1;
                }
            }
            println!(
                "quad_control: {name}: shaderQuadControl enabled; {drawn} pixels read their \
                 quads whole, helpers included"
            );
        });
    }

    /// `quad_votes`' state is made on every device with quad control, its fragment stage holding
    /// one `OpGroupNonUniformQuadAllKHR`, one `OpGroupNonUniformQuadAnyKHR` and no ballot, and
    /// each pixel the triangle covers reads its quad's votes with the helper's predicate counted:
    /// a diagonal quad's `quad_any(x > y)` holds and its `quad_all(x <= y)` does not, as Metal
    /// answers.
    #[test]
    fn every_device_votes_over_its_quads_helpers() {
        let fragment = match llvm_assemble_text(QUAD_VOTES) {
            Ok(bitcode) => Library::from_air(&bitcode).expect("the library"),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("quad_votes: {error:?}"),
        };
        let Some(examples) = examples() else {
            return;
        };
        let mut d = descriptor(&examples, "colored_fragment", PixelFormat::RGBA32Uint);
        d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
            fragment.function("quad_votes").expect("the function"),
        )));
        each_device("quad_votes", |name, _, device| {
            if !(device.core().translator.profile().enabled().features)
                .contains(&VkFeature::ShaderQuadControl)
            {
                println!("quad_votes: {name}: no quad control; not made");
                return;
            }
            // The votes are the quad vote instructions, which Vulkan runs over the quad's helpers
            // in every entry, and no ballot.
            let ops = fragment_ops(name, &device, &d);
            let count = |op: u32| ops.iter().filter(|&&o| o == op).count();
            assert_eq!(
                [count(OP_QUAD_ALL), count(OP_QUAD_ANY), count(OP_BALLOT)],
                [1, 1, 0],
                "{name}: the fragment's quad votes"
            );
            let state = (device.new_render_pipeline_state(&d, crate::tests::no_constexpr_samplers))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let texels = diagonal(name, &device, &state, &ON_OR_LEFT);
            let mut drawn = 0;
            let mut wrong = Vec::new();
            for y in 0..SIDE {
                for x in (0..SIDE).filter(|&x| x <= y) {
                    let helper = x >> 1 == y >> 1;
                    let want = vec![u32::from(helper), u32::from(!helper), 0, 1];
                    let got = words(&texels, [x, y], 4);
                    if got != want {
                        wrong.push(([x, y], got, want));
                    }
                    drawn += 1;
                }
            }
            assert!(
                wrong.is_empty(),
                "{name}: {} of {drawn} pixels differ ([x, y], got, want): {wrong:?}",
                wrong.len()
            );
            println!(
                "quad_votes: {name}: {drawn} pixels read their quads' votes, helpers included"
            );
        });
    }

    /// `quad_sum_after_a_return`'s state is made on every device with quad control, its fragment
    /// stage holding the loop of implicit-LOD samples and the quad votes of its reduction, and its
    /// draw writes each pixel's sums: in a row past the top two, at texel `(x, y)` (red `x / 16`),
    /// `sum` is `x / 16 * (1 + 2 + 3)`, and the quad of `x` in `{2k, 2k + 1}` sums `3k + 0.75`.
    #[test]
    fn every_device_makes_a_quad_sum_after_a_return_before_a_sampling_loop() {
        let examples = examples();
        let fragment = match llvm_assemble_text(QUAD_SUM_AFTER_A_RETURN) {
            Ok(bitcode) => Library::from_air(&bitcode).expect("the library"),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("quad_sum_after_a_return: {error:?}"),
        };
        let Some(examples) = examples else {
            return;
        };
        let mut d = descriptor(&examples, "colored_fragment", PixelFormat::RGBA32Float);
        d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
            fragment
                .function("quad_sum_after_a_return")
                .expect("the function"),
        )));
        each_device("quad_sum_after_a_return", |name, _, device| {
            let core = device.core();
            if !(core.translator.profile().enabled().features)
                .contains(&VkFeature::ShaderQuadControl)
            {
                println!("quad_sum_after_a_return: {name}: no quad control; not made");
                return;
            }
            let ops = fragment_ops(name, &device, &d);
            let count = |op: u32| ops.iter().filter(|&&o| o == op).count();
            assert!(count(OP_LOOP_MERGE) > 0, "{name}: no loop");
            assert!(
                count(OP_SAMPLE_IMPLICIT_LOD) > 0,
                "{name}: no implicit-LOD sample"
            );
            assert_eq!(count(OP_QUAD_ANY), 4, "{name}: the reduction's quad votes");
            let returns = count(OP_RETURN) + count(OP_RETURN_VALUE);
            for (class, first) in [(IndexClass::Texture, 4096), (IndexClass::Sampler, 1)] {
                (device.grant_indices(class, first, 4))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            }
            let state = (device.new_render_pipeline_state(&d, |samplers| {
                assert!(
                    matches!(samplers, [NumberedSampler::Constexpr(_)]),
                    "{name}: {samplers:?}"
                );
                Ok(vec![CONSTEXPR])
            }))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut sd =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA16Float, SIDE, SIDE);
            sd.usage = TextureUsage::SHADER_READ;
            let source = (device.new_texture(&sd, Some(4096)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            // Each texel's red channel is x / 16, which a half holds exactly; the others 0.
            let half = |k: u32| -> [u8; 2] {
                match k {
                    0 => [0, 0],
                    _ => {
                        let bits = (k as f32 / 16.0).to_bits();
                        let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
                        (((exponent as u16) << 10) | ((bits >> 13) & 0x3ff) as u16).to_le_bytes()
                    }
                }
            };
            let bytes: Vec<u8> = (0..SIDE)
                .flat_map(|_| 0..SIDE)
                .flat_map(|x| [x, 0, 0, 0].into_iter().flat_map(half))
                .collect();
            let whole = Region {
                origin: [0; 3],
                size: [SIDE, SIDE, 1],
            };
            let row = u64::from(SIDE * 8);
            (source.replace_region(
                whole,
                0,
                0,
                crate::HostBytes::from(&bytes[..]),
                row,
                row * u64::from(SIDE),
            ))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let trips = buffers.of(&device, &TRIPS.to_le_bytes());
            let color = target(
                &device,
                PixelFormat::RGBA32Float,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            draw(encoder.set_render_pipeline_state(&state));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            draw(encoder.set_fragment_texture(Some(&source), 0));
            draw(encoder.set_fragment_buffer(Some(&trips), 0, 0));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let texels = read(&color, 16);
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let want = if y < 2 {
                        [-1.0, -1.0, -1.0, 1.0]
                    } else {
                        [3.0 * (x / 2) as f32 + 0.75, x as f32 * 0.375, 0.0, 1.0]
                    };
                    let at = ((y * SIDE + x) * 16) as usize;
                    let got: Vec<f32> = (texels[at..at + 16].as_chunks::<4>().0.iter())
                        .map(|w| f32::from_le_bytes(*w))
                        .collect();
                    assert_eq!(got, want, "{name}: pixel [{x}, {y}]");
                }
            }
            println!(
                "quad_sum_after_a_return: {name}: made and drawn; the fragment stage holds \
                 {} loop(s), {} implicit-LOD sample(s) and {returns} return(s)",
                count(OP_LOOP_MERGE),
                count(OP_SAMPLE_IMPLICIT_LOD)
            );
        });
    }

    /// A fragment's `quad_ballot`, `quad_active_threads_mask`, `quad_is_first`,
    /// `quad_broadcast_first` and quad prefix sums count the helpers of a quad the triangle cuts as
    /// the active lanes they are (`quad_readers`), on every device with quad control, and their
    /// fragment stages hold no ballot: they read their quad through quad instructions, which
    /// Vulkan runs over the quad's helpers in every entry (section 9.31). Where every invocation
    /// runs their block, each quad is whole and they take no quad vote; under a branch on the
    /// pixel (`quad_masks_divergent`, whose pixels of odd `x` run them), they vote which lanes take
    /// part.
    #[test]
    fn every_device_s_quad_masks_and_scans_count_their_quads_helpers() {
        quad_readers(
            "quad_masks_and_scans",
            &MASKS_AND_SCANS,
            |name, function, ops| {
                assert!(
                    !ops.contains(&OP_BALLOT),
                    "{name}: {function}'s fragment stage holds a ballot"
                );
                let votes = ops.iter().filter(|&&o| o == OP_QUAD_ANY).count();
                let divergent = function.ends_with("_divergent");
                assert_eq!(votes > 0, divergent, "{name}: {function}'s quad votes");
                println!("quad_masks_and_scans: {name}: {function} holds {votes} quad vote(s)");
            },
        );
    }

    /// A fragment's quad shuffles and broadcast by lane read the helper of a quad the triangle
    /// cuts (`quad_readers`), on every device with quad control, and their fragment stages hold no
    /// ballot: they read their quad through quad instructions, which Vulkan runs over the quad's
    /// helpers in every entry (section 9.31).
    #[test]
    fn every_device_s_quad_shuffles_read_their_quads_helpers() {
        quad_readers("quad_shuffles", &SHUFFLES, |name, function, ops| {
            assert!(
                !ops.contains(&OP_BALLOT),
                "{name}: {function}'s fragment stage holds a ballot"
            );
            assert!(
                !ops.contains(&OP_QUAD_ANY),
                "{name}: {function}'s whole quads take a vote"
            );
        });
    }

    /// The side of the target `every_device_reads_the_helpers_of_a_quad_its_target_cuts` draws:
    /// odd, so the quads of its last column and row hang past it, their pixels there run as
    /// helpers.
    const CUT: u32 = 7;

    /// A fragment's quad reductions in a block every invocation runs read the helpers of the quads
    /// its target cuts, as Metal runs and counts them: `fragment_quad_bitwise` drawn over the whole
    /// of a `CUT` by `CUT` target, on every device with quad control, writes at every pixel its
    /// quad's AND, OR and XOR of the four pixels' words, those past the target included.
    #[test]
    fn every_device_reads_the_helpers_of_a_quad_its_target_cuts() {
        let (Some(examples), Some(quads)) =
            (examples(), library(&engine_dir().join("gsm2_quad_ops.ll")))
        else {
            return;
        };
        let mut d = descriptor(&examples, "colored_fragment", PixelFormat::RGBA32Uint);
        d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
            quads
                .function("fragment_quad_bitwise")
                .expect("the function"),
        )));
        each_device("quad_target_cut", |name, _, device| {
            if !(device.core().translator.profile().enabled().features)
                .contains(&VkFeature::ShaderQuadControl)
            {
                println!("quad_target_cut: {name}: no quad control; not made");
                return;
            }
            let state = (device.new_render_pipeline_state(&d, crate::tests::no_constexpr_samplers))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut td =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA32Uint, CUT, CUT);
            td.usage = TextureUsage::RENDER_TARGET;
            td.storage_mode = StorageMode::Shared;
            let color = (device.new_texture_for_views(&td))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut buffers = Buffers::new(&device);
            let positions = buffers.of(&device, &floats(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]));
            let colors = buffers.of(&device, &floats(&[0.0; 12]));
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let ok = |result: Result<(), Error>| {
                result.unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            ok(encoder.set_render_pipeline_state(&state));
            ok(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            ok(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            ok(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let row = CUT as usize * 16;
            let bytes = row * CUT as usize;
            let mut texels = vec![0; bytes];
            let region = Region {
                origin: [0; 3],
                size: [CUT, CUT, 1],
            };
            (color.get_bytes(&mut texels, row as u64, bytes as u64, region, 0, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut past = 0;
            for y in 0..CUT {
                for x in 0..CUT {
                    let quad = [[0, 0], [1, 0], [0, 1], [1, 1]]
                        .map(|[dx, dy]| [(x & !1) + dx, (y & !1) + dy]);
                    let w = quad.map(word);
                    let want = [
                        w.iter().fold(!0, |all, w| all & (w | 0x100)),
                        w.iter().fold(0, |any, w| any | w),
                        w.iter().fold(0, |odd, w| odd ^ w),
                    ];
                    let at = (y as usize * CUT as usize + x as usize) * 16;
                    let got: Vec<u32> = (texels[at..at + 12].as_chunks::<4>().0.iter())
                        .map(|w| u32::from_le_bytes(*w))
                        .collect();
                    assert_eq!(got, want, "{name}: pixel [{x}, {y}]");
                    past += usize::from(quad.iter().any(|&[qx, qy]| qx >= CUT || qy >= CUT));
                }
            }
            println!(
                "quad_target_cut: {name}: {} pixels read their quads whole, {past} of them with \
                 helpers past the target",
                CUT * CUT
            );
        });
    }
}
