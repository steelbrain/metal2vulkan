; Owned, written by hand in the shape a GL fragment's AIR takes where it emulates a cube texture as
; a 2D array: a `switch` on a value only the run knows picks a texture unit (5 by default, 7 for
; case 5, 8 for case 29), a PHI of constants, and the texture of that unit, a PHI of three handles.
; A bit test of the unit (`trunc (lshr 416, zext unit)`) then picks between sampling the texture as
; a 2D array and sampling it as a cube. Bits 5, 7 and 8 of 416 are set, so every unit takes the 2D
; array's way: the cube sample, of a texture the entry declares `texture2d_array`, never runs.
;
; Each of the three textures is 2x2 with one slice. Texture k's texel (x, y) carries the four
; channels 100(k+1) + 10(2y + x) + c for c in 1..4, and the kernel folds them as
; r + 10g + 100b + 1000a. A `nearest`, `clamp_to_edge`, normalized sample at (0.25, 0.75) is texel
; (0, 1) exactly. Thread t reads which[t]: 0 or 7 (the default) answers 137641 (texture 0), 5
; answers 248741 (texture 1) and 29 answers 359841 (texture 2). Every term is an integer below
; 2^24.
source_filename = "kernel_texture_unit_bit_test"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx14.0.0"

define void @kernel_texture_unit_bit_test(ptr addrspace(1) %t5, ptr addrspace(1) %t7, ptr addrspace(1) %t8, ptr addrspace(2) %samp, ptr addrspace(1) "air-buffer-no-alias" %which, ptr addrspace(1) "air-buffer-no-alias" %out, i32 %tid) #0 {
entry:
  %t = zext i32 %tid to i64
  %wp = getelementptr inbounds i32, ptr addrspace(1) %which, i64 %t
  %w = load i32, ptr addrspace(1) %wp, align 4
  switch i32 %w, label %pick [
    i32 5, label %five
    i32 29, label %nine
  ]

five:
  br label %pick

nine:
  br label %pick

pick:
  %unit = phi i32 [ 5, %entry ], [ 7, %five ], [ 8, %nine ]
  %tex = phi ptr addrspace(1) [ %t5, %entry ], [ %t7, %five ], [ %t8, %nine ]
  %wide = zext i32 %unit to i64
  %bits = lshr i64 416, %wide
  %seamed = trunc i64 %bits to i1
  br i1 %seamed, label %array, label %cube

array:
  %a = call { <4 x float>, i8 } @air.sample_texture_2d_array.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %samp, <2 x float> <float 2.500000e-01, float 7.500000e-01>, i32 0, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0) #1
  br label %done

cube:
  %c = call { <4 x float>, i8 } @air.sample_texture_cube.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %samp, <3 x float> <float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0) #1
  br label %done

done:
  %r = phi { <4 x float>, i8 } [ %a, %array ], [ %c, %cube ]
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <4 x float> %v, i64 0
  %y = extractelement <4 x float> %v, i64 1
  %y10 = fmul float %y, 1.000000e+01
  %s1 = fadd float %x, %y10
  %z = extractelement <4 x float> %v, i64 2
  %z100 = fmul float %z, 1.000000e+02
  %s2 = fadd float %s1, %z100
  %q = extractelement <4 x float> %v, i64 3
  %q1000 = fmul float %q, 1.000000e+03
  %s3 = fadd float %s2, %q1000
  %op = getelementptr inbounds float, ptr addrspace(1) %out, i64 %t
  store float %s3, ptr addrspace(1) %op, align 4
  ret void
}

declare { <4 x float>, i8 } @air.sample_texture_2d_array.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i32, i1, <2 x i32>, i1, float, float, i32) #1

declare { <4 x float>, i8 } @air.sample_texture_cube.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, float, float, i32) #1

attributes #0 = { convergent nounwind }
attributes #1 = { convergent nounwind willreturn memory(argmem: read) }

!air.kernel = !{!0}
!air.version = !{!10}
!air.language_version = !{!11}

!0 = !{ptr @kernel_texture_unit_bit_test, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7, !8, !9}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d_array<float, sample>", !"air.arg_name", !"t5"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d_array<float, sample>", !"air.arg_name", !"t7"}
!5 = !{i32 2, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d_array<float, sample>", !"air.arg_name", !"t8"}
!6 = !{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"samp"}
!7 = !{i32 4, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"which"}
!8 = !{i32 5, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!9 = !{i32 6, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!10 = !{i32 2, i32 8, i32 0}
!11 = !{!"Metal", i32 3, i32 0, i32 0}
