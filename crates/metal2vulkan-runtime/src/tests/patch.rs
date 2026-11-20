//! Patch draws: a render pipeline state whose vertex function is post-tessellation, and its
//! draws.
//!
//! Under a device cfg, on every device of the cfg's kind, of the owned fixture
//! `engine/tessellation.ll`'s `tess_triangle_points` (a triangle patch of three control points a
//! vertex descriptor fetches, and a per-patch tint), with a fragment function that writes the
//! colour it is given:
//! - two patches drawn, fractional odd up to 64, with their factors 8 bytes into the factor
//!   buffer past bytes of zero: the first covers the top half of clip space in its tint, and the
//!   second, whose first edge factor is 0, is culled. The pass is one submission, its barrier and
//!   its pass, and makes no pipeline and copies no byte of the runtime's own.
//! - the same patches of a post-tessellation function whose fields pass its push window (16 more
//!   buffer arguments, written here): its parameter block survives the control stage's placement
//!   of the vertex stage, and every buffer's address and bounds reach the function from it.
//! - the same patches behind a fragment function whose fields fill the push range (4 to 24 buffer
//!   arguments, written here): the state is made, as Metal makes it, with the control stage's
//!   range at 0 below the post-tessellation function's window, and the fragment function's fields
//!   past the range in its block.
//! - what Metal refuses is refused: a patch draw with a plain state, a plain draw with a patch
//!   state, a patch draw with no factor buffer and factors at an odd byte. What the runtime does
//!   not build is a gap: a patch index buffer, a per-instance factor step and an enabled factor
//!   scale; a control-point index type is the translator's refusal.
//! - the user-space instructions a patch draw retires against a plain draw's, printed per draw
//!   where the kernel offers the counter (Linux), with the build profile and the validation
//!   layer the device tests load: a patch draw that changes nothing records what a plain draw
//!   records.
//!
//! `llvm-as` assembles the fixtures; without it a device test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::blit::on_device::settled;
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{
        descriptor, examples, floats, library, pass, read, run, state, target, texels, vertices,
        Buffers, BLACK, BLUE, GREEN, SIDE,
    };
    use crate::{
        Buffer, ColorAttachmentDescriptor, Device, DrawKind, Error, FunctionDescriptor, Gap,
        Library, PixelFormat, PrimitiveType, Refusal, RenderPipelineDescriptor, RenderPipelineKind,
        StageSource, TextureUsage,
    };
    use metal2vulkan::error::Feature;
    use metal2vulkan::metal::{
        TessellationControlPointIndexType, TessellationFactorStepFunction,
        TessellationPartitionMode, VertexFormat, VertexStepFunction, Winding,
    };
    use metal2vulkan::pipeline::{
        TessellationDescriptor, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
        VertexDescriptor,
    };
    use metal2vulkan_devtools::{engine_dir, llvm_assemble_text, AssembleError};

    /// A fragment function that writes its `user(locn0)` colour.
    const FRAGMENT: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <4 x float> @tess_fragment(<4 x float> %0) local_unnamed_addr #0 {
  ret <4 x float> %0
}

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.fragment = !{!9}
!air.compile_options = !{!14, !15}
!air.version = !{!16}
!air.language_version = !{!17}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @tess_fragment, !10, !12}
!10 = !{!11}
!11 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!12 = !{!13}
!13 = !{i32 0, !"air.fragment_input", !"user(locn0)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!14 = !{!"air.compile.denorms_disable"}
!15 = !{!"air.compile.fast_math_enable"}
!16 = !{i32 2, i32 8, i32 0}
!17 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

    /// The buffer arguments of [`wide_vertex`]'s function.
    const WIDE: u32 = 16;

    /// [`wide_fragment`]'s text but its buffer arguments, their reads and their metadata.
    const WIDE_FRAGMENT_TEMPLATE: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <4 x float> @tess_wide_fragment(<4 x float> %0ARGS) local_unnamed_addr #0 {
READS  ret <4 x float> SUM
}

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.fragment = !{!9}
!air.compile_options = !{!14, !15}
!air.version = !{!16}
!air.language_version = !{!17}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @tess_wide_fragment, !10, !12}
!10 = !{!11}
!11 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!12 = !{!13BUFFERS}
!13 = !{i32 0, !"air.fragment_input", !"user(locn0)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!14 = !{!"air.compile.denorms_disable"}
!15 = !{!"air.compile.fast_math_enable"}
!16 = !{i32 2, i32 8, i32 0}
!17 = !{!"Metal", i32 4, i32 0, i32 0}
BUFFER_META"#;

    /// The buffer arguments of each of [`wide_fragment`]'s functions: the narrowest pushes
    /// every field; behind a post-tessellation function whose window started at 0, 19 would end
    /// the pair's fields at 252, where the factors' 8-byte address no longer fits above them; the
    /// widest spills.
    const FRAGMENT_WIDTHS: [u32; 4] = [4, 12, 19, 24];

    /// A fragment function that writes its `user(locn0)` colour plus `width` `float4` buffer
    /// arguments at indices 0 and on, each read at element 0, a read checked against the
    /// buffer's bounds.
    fn wide_fragment(width: u32) -> String {
        let args: String = (0..width)
            .map(|k| {
                format!(
                    r#", ptr addrspace(1) noundef readonly align 16 captures(none) "air-buffer-no-alias" %f{k}"#
                )
            })
            .collect();
        let mut reads = String::new();
        let mut sum = "%0".to_owned();
        for k in 0..width {
            reads += &format!(
                "  %v{k} = load <4 x float>, ptr addrspace(1) %f{k}, align 16\n  \
                 %s{k} = fadd fast <4 x float> {sum}, %v{k}\n"
            );
            sum = format!("%s{k}");
        }
        let buffers: String = (0..width).map(|k| format!(", !{}", 30 + k)).collect();
        let meta: String = (0..width)
            .map(|k| {
                format!(
                    "!{} = !{{i32 {}, !\"air.buffer\", !\"air.location_index\", i32 {k}, i32 1, \
                     !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 16, \
                     !\"air.arg_type_align_size\", i32 16, !\"air.arg_type_name\", !\"float4\", \
                     !\"air.arg_name\", !\"f{k}\"}}\n",
                    30 + k,
                    1 + k
                )
            })
            .collect();
        WIDE_FRAGMENT_TEMPLATE
            .replace("ARGS", &args)
            .replace("READS", &reads)
            .replace("SUM", &sum)
            .replace("BUFFERS", &buffers)
            .replace("BUFFER_META", &meta)
    }

    /// [`wide_vertex`]'s text but its buffer arguments, their reads and their metadata.
    const WIDE_TEMPLATE: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <{ <4 x float>, <4 x float> }> @tess_wide_points(ptr %0, <4 x float> %1, <3 x float> noundef %2, i32 noundef %3ARGS) local_unnamed_addr #0 {
  %5 = tail call { <4 x float>, <2 x half>, <4 x i8> } @_Z12ControlPoint.MTL_CONTROL_POINT_FN(i32 0, ptr %0) #4
  %6 = extractvalue { <4 x float>, <2 x half>, <4 x i8> } %5, 0
  %7 = shufflevector <3 x float> %2, <3 x float> undef, <4 x i32> zeroinitializer
  %8 = fmul fast <4 x float> %6, %7
  %9 = tail call { <4 x float>, <2 x half>, <4 x i8> } @_Z12ControlPoint.MTL_CONTROL_POINT_FN(i32 1, ptr %0) #4
  %10 = extractvalue { <4 x float>, <2 x half>, <4 x i8> } %9, 0
  %11 = shufflevector <3 x float> %2, <3 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %12 = fmul fast <4 x float> %10, %11
  %13 = fadd fast <4 x float> %12, %8
  %14 = tail call { <4 x float>, <2 x half>, <4 x i8> } @_Z12ControlPoint.MTL_CONTROL_POINT_FN(i32 2, ptr %0) #4
  %15 = extractvalue { <4 x float>, <2 x half>, <4 x i8> } %14, 0
  %16 = shufflevector <3 x float> %2, <3 x float> undef, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %17 = fmul fast <4 x float> %15, %16
  %18 = fadd fast <4 x float> %13, %17
  %19 = tail call fast float @air.convert.f.f32.u.i32(i32 %3) #3
  %20 = insertelement <4 x float> poison, float %19, i64 0
  %21 = shufflevector <4 x float> %20, <4 x float> poison, <4 x i32> zeroinitializer
  %22 = fadd fast <4 x float> %21, %1
  %i = zext i32 %3 to i64
READS  %23 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %18, 0
  %24 = insertvalue <{ <4 x float>, <4 x float> }> %23, <4 x float> SUM, 1
  ret <{ <4 x float>, <4 x float> }> %24
}

declare { <4 x float>, <2 x half>, <4 x i8> } @_Z12ControlPoint.MTL_CONTROL_POINT_FN(i32, ptr) local_unnamed_addr #2 section "air.externally_defined"

declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

attributes #0 = { convergent mustprogress nofree nounwind willreturn memory(read) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent mustprogress nofree nounwind willreturn memory(read) }
attributes #3 = { nounwind willreturn memory(none) }
attributes #4 = { convergent nounwind willreturn memory(read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.vertex = !{!9}
!air.compile_options = !{!23, !24}
!air.version = !{!25}
!air.language_version = !{!26}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @tess_wide_points, !10, !13, !22}
!10 = !{!11, !12}
!11 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!12 = !{!"air.vertex_output", !"user(locn0)", !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!13 = !{!14, !19, !20, !21BUFFERS}
!14 = !{i32 0, !"air.patch_control_point_input", !15, !16, !17, !18}
!15 = !{!"air.patch_control_point_function", ptr @_Z12ControlPoint.MTL_CONTROL_POINT_FN}
!16 = !{!"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!17 = !{!"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"half2", !"air.arg_name", !"uv", !"air.arg_unused"}
!18 = !{!"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"uchar4", !"air.arg_name", !"color", !"air.arg_unused"}
!19 = !{i32 1, !"air.patch_input", !"air.location_index", i32 5, i32 1, !"air.arg_type_name", !"float4", !"air.arg_name", !"tint"}
!20 = !{i32 2, !"air.position_in_patch", !"air.arg_type_name", !"float3", !"air.arg_name", !"bc"}
!21 = !{i32 3, !"air.instance_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"instance"}
!22 = !{!"air.patch", !"triangle", !"air.patch_control_point", i32 3}
!23 = !{!"air.compile.denorms_disable"}
!24 = !{!"air.compile.fast_math_enable"}
!25 = !{i32 2, i32 8, i32 0}
!26 = !{!"Metal", i32 4, i32 0, i32 0}
BUFFER_META"#;

    /// A post-tessellation function whose fields pass the vertex function's push window:
    /// `tess_triangle_points` without the control points' `half2` (its control points and its
    /// patch's tint), plus [`WIDE`] `float4` buffer arguments at indices 2 and on, each read at
    /// the instance's element, a read checked against the buffer's bounds, and added into the
    /// colour.
    fn wide_vertex() -> String {
        let args: String = (0..WIDE)
            .map(|k| {
                format!(
                    r#", ptr addrspace(1) noundef readonly align 16 captures(none) "air-buffer-no-alias" %b{k}"#
                )
            })
            .collect();
        let mut reads = String::new();
        let mut sum = "%22".to_owned();
        for k in 0..WIDE {
            reads += &format!(
                "  %p{k} = getelementptr inbounds <4 x float>, ptr addrspace(1) %b{k}, i64 %i\n  \
                 %v{k} = load <4 x float>, ptr addrspace(1) %p{k}, align 16\n  \
                 %s{k} = fadd fast <4 x float> {sum}, %v{k}\n"
            );
            sum = format!("%s{k}");
        }
        let buffers: String = (0..WIDE).map(|k| format!(", !{}", 30 + k)).collect();
        let meta: String = (0..WIDE)
            .map(|k| {
                format!(
                    "!{} = !{{i32 {}, !\"air.buffer\", !\"air.location_index\", i32 {}, i32 1, \
                     !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 16, \
                     !\"air.arg_type_align_size\", i32 16, !\"air.arg_type_name\", !\"float4\", \
                     !\"air.arg_name\", !\"b{k}\"}}\n",
                    30 + k,
                    4 + k,
                    2 + k
                )
            })
            .collect();
        WIDE_TEMPLATE
            .replace("ARGS", &args)
            .replace("READS", &reads)
            .replace("SUM", &sum)
            .replace("BUFFERS", &buffers)
            .replace("BUFFER_META", &meta)
    }

    /// The half-precision factors of one triangle patch: three edges, then the inside.
    fn factors(edges: [u16; 3], inside: u16) -> Vec<u8> {
        [edges[0], edges[1], edges[2], inside]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect()
    }

    /// 3, 5, 7 and 4 in half precision.
    const THREE: u16 = 0x4200;
    const FIVE: u16 = 0x4500;
    const SEVEN: u16 = 0x4700;
    const FOUR: u16 = 0x4400;

    /// The fixture's post-tessellation function and the fragment function, or None where `llvm-as`
    /// is missing.
    fn libraries() -> Option<(Library, Library)> {
        let vertex = library(&engine_dir().join("tessellation.ll"))?;
        let fragment = match llvm_assemble_text(FRAGMENT) {
            Ok(bitcode) => Library::from_air(&bitcode).expect("the fragment library"),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return None;
            }
            Err(error) => panic!("the fragment: {error:?}"),
        };
        Some((vertex, fragment))
    }

    /// `tess_triangle_points` with the fragment function into one RGBA8 attachment: control
    /// points of a position, a `half2` and a `uchar4` 24 bytes apart in buffer 0, a tint per patch
    /// in buffer 1; fractional odd up to 64, counter-clockwise, factors per patch.
    fn patches(libraries: &(Library, Library)) -> RenderPipelineDescriptor {
        let (vertex, fragment) = libraries;
        let function = |library: &Library, name| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(function(vertex, "tess_triangle_points"));
        d.fragment = Some(function(fragment, "tess_fragment"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let mut v = VertexDescriptor::new();
        for (a, format, offset, buffer) in [
            (0, VertexFormat::Float4, 0, 0),
            (1, VertexFormat::Half2, 16, 0),
            (2, VertexFormat::UChar4, 20, 0),
            (5, VertexFormat::Float4, 0, 1),
        ] {
            v.attributes
                .insert(a, VertexAttributeDescriptor::new(format, offset, buffer));
        }
        let mut points = VertexBufferLayoutDescriptor::new(Some(24));
        points.step_function = VertexStepFunction::PerPatchControlPoint;
        let mut tints = VertexBufferLayoutDescriptor::new(Some(16));
        tints.step_function = VertexStepFunction::PerPatch;
        v.layouts.insert(0, points);
        v.layouts.insert(1, tints);
        d.vertex_descriptor = Some(v);
        let mut t = TessellationDescriptor::default();
        // Metal's default step is constant: every patch would read the first patch's factors.
        t.factor_step_function = TessellationFactorStepFunction::PerPatch;
        t.partition_mode = TessellationPartitionMode::FractionalOdd;
        t.max_tessellation_factor = 64;
        t.output_winding_order = Winding::CounterClockwise;
        d.tessellation = Some(t);
        d
    }

    /// The tessellation state of `d`, a state of [`patches`].
    fn tessellation(d: &mut RenderPipelineDescriptor) -> &mut TessellationDescriptor {
        d.tessellation.as_mut().expect("a tessellation state")
    }

    /// The two patches' control points and tints, and their factor buffer: the top half of clip
    /// space (y up) in green, factors 3, 5, 7 and 4, then the whole of it in blue with a first
    /// edge factor of 0; the factors 8 bytes past bytes of zero.
    fn patch_buffers(device: &Device, buffers: &mut Buffers) -> (Buffer, Buffer, Buffer) {
        let top = [[-1.0, 0.0], [3.0, 0.0], [-1.0, 4.0]];
        let whole = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]];
        let points: Vec<u8> = [top, whole]
            .iter()
            .flatten()
            .flat_map(|[x, y]| {
                let mut point = floats(&[*x, *y, 0.0, 1.0]);
                // The control point's `half2` (added to the position's x) and its unread colour.
                point.extend([0; 4]);
                point.extend([0xff; 4]);
                point
            })
            .collect();
        let tints = floats(&[GREEN, BLUE].concat());
        let factor_bytes = [
            vec![0; 8],
            factors([THREE, FIVE, SEVEN], FOUR),
            factors([0, FIVE, SEVEN], FOUR),
        ]
        .concat();
        (
            buffers.of(device, &points),
            buffers.of(device, &tints),
            buffers.of(device, &factor_bytes),
        )
    }

    /// Two patches draw where their control points put them, from factors at an offset; the
    /// second, whose edge factor is 0, is culled. One submission, one barrier, one pass, no
    /// pipeline made and no byte copied.
    #[test]
    fn every_device_draws_patches_from_their_factors() {
        let Some(libraries) = libraries() else {
            return;
        };
        let d = patches(&libraries);
        each_device("draw_patches", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (points, tints, factors) = patch_buffers(&device, &mut buffers);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            // The upload that moves the new target to its home layout, in a submission apart.
            let first = queue.command_buffer().expect("a command buffer");
            (first.record_for_test(&[], |_, _| {})).expect("a command buffer's commands");
            run(name, &first);
            let counters = device.counters();
            settled(&device);
            let before = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            draw(encoder.set_render_pipeline_state(&state));
            draw(encoder.set_vertex_buffer(Some(&points), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&tints), 0, 1));
            draw(encoder.set_tessellation_factor_buffer(Some(&factors), 8, 0));
            draw(encoder.draw_patches(3, 0, 2, None, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = counters.metrics();
            assert_eq!(
                (
                    after.submissions - before.submissions,
                    after.barriers - before.barriers,
                    after.passes - before.passes,
                    after.pipelines_created - before.pipelines_created,
                    after.bytes_copied - before.bytes_copied,
                ),
                (1, 1, 1, 0, 0),
                "{name}: submissions, barriers, passes, pipelines made, bytes copied"
            );
            let want = texels(|_, y| if y < 4 { GREEN } else { BLACK });
            assert_eq!(read(&color, 4), want, "{name}: the texels");
            println!("{name}: two patches drawn, the second culled; 1 barrier, no pipeline made");
        });
    }

    /// A post-tessellation function whose fields pass its push window ([`wide_vertex`]) spills
    /// the rest into its parameter block, which each draw writes into the ring and the function
    /// reads: the pipeline's control stage, whose fields the translator states as a second
    /// placement of the vertex stage, leaves the vertex function's block, and the block holds
    /// fields of the summed buffers. Buffer `k` holds `2^(k % 4) / 255` in channel `k / 4`, so the
    /// patch's colour is 15 in every channel only where every buffer's address and bounds reach
    /// the function where it reads them; one read through a wrong or empty field drops or repeats
    /// a bit. The second patch, whose first edge factor is 0, is culled.
    #[test]
    fn every_device_draws_patches_whose_fields_spill_into_their_block() {
        use crate::pipeline::FieldValue;
        use crate::render::{DrawPlace, VERTEX};
        let Some(libraries) = libraries() else {
            return;
        };
        let wide = match llvm_assemble_text(&wide_vertex()) {
            Ok(bitcode) => Library::from_air(&bitcode).expect("the wide library"),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("the wide function: {error:?}"),
        };
        let mut d = patches(&libraries);
        d.vertex = StageSource::Function(FunctionDescriptor::new(
            wide.function("tess_wide_points").expect("the function"),
        ));
        each_device("draw_patches_spilled", |name, _, device| {
            let state = state(name, &device, &d);
            let draws = &state.inner.compiled.draws;
            let spilled: Vec<FieldValue> = (draws.fields.iter())
                .filter(|(place, ..)| matches!(place, DrawPlace::Block(VERTEX, _)))
                .map(|&(_, _, value)| value)
                .collect();
            assert!(
                spilled.iter().any(|value| matches!(
                    value,
                    FieldValue::BufferAddress { slot, .. } | FieldValue::BufferBounds(slot)
                        if (2..2 + WIDE).contains(slot)
                )),
                "{name}: the vertex function's block holds a summed buffer's field"
            );
            assert!(
                draws.blocks[VERTEX].is_some(),
                "{name}: the vertex function's block, past the control stage's placement"
            );
            let mut buffers = Buffers::new(&device);
            let (points, _, factors) = patch_buffers(&device, &mut buffers);
            let tints = buffers.of(&device, &[0; 32]);
            let summed: Vec<Buffer> = (0..WIDE)
                .map(|k| {
                    let mut value = [0.0; 4];
                    value[(k / 4) as usize] = f32::from(1u8 << (k % 4)) / 255.0;
                    buffers.of(&device, &floats(&value))
                })
                .collect();
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            draw(encoder.set_render_pipeline_state(&state));
            draw(encoder.set_vertex_buffer(Some(&points), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&tints), 0, 1));
            for (k, buffer) in summed.iter().enumerate() {
                draw(encoder.set_vertex_buffer(Some(buffer), 0, 2 + k as u32));
            }
            draw(encoder.set_tessellation_factor_buffer(Some(&factors), 8, 0));
            draw(encoder.draw_patches(3, 0, 2, None, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let black = [0, 0, 0, 255];
            let want: Vec<u8> = (0..SIDE * SIDE)
                .flat_map(|at| if at / SIDE < 4 { [15; 4] } else { black })
                .collect();
            assert_eq!(read(&color, 4), want, "{name}: the texels");
            println!(
                "{name}: a patch of {} fields in its block drawn from every buffer",
                spilled.len()
            );
        });
    }

    /// A patch state whose fragment function's fields fill the floor's push range behind the
    /// post-tessellation function's ([`wide_fragment`]) is made, as Metal makes it, and draws its
    /// patches: the control stage pushes the factors' address at 0, below the post-tessellation
    /// function's window, and the fragment function's fields past the range go to its block. A
    /// pair's fields reach past 248 bytes, where no range of the factors' address is left above
    /// them, and the widest fragment function's spill. Every state is made before any draw.
    /// Buffer `k` holds `2^(k / 4) / 255` in channel `k % 4` and the tints are 0, so the first
    /// patch's colour holds every bit of the buffers a function reads only where each buffer's
    /// address and bounds reach it; the second patch is culled.
    #[test]
    fn every_device_draws_patches_whose_fragment_fills_the_push_range() {
        use crate::pipeline::FieldValue;
        use crate::render::DrawPlace;
        let Some(libraries) = libraries() else {
            return;
        };
        let mut fragments = Vec::new();
        for width in FRAGMENT_WIDTHS {
            match llvm_assemble_text(&wide_fragment(width)) {
                Ok(bitcode) => fragments.push((
                    width,
                    Library::from_air(&bitcode).expect("the wide fragment library"),
                )),
                Err(AssembleError::Missing(why)) => {
                    eprintln!("SKIPPED: {why}");
                    return;
                }
                Err(error) => panic!("the wide fragment {width}: {error:?}"),
            }
        }
        each_device("draw_patches_full_range", |name, _, device| {
            let states: Vec<_> = (fragments.iter())
                .map(|(width, library)| {
                    let mut d = patches(&libraries);
                    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
                        library
                            .function("tess_wide_fragment")
                            .expect("the function"),
                    )));
                    (
                        *width,
                        state(&format!("{name}: {width} buffers"), &device, &d),
                    )
                })
                .collect();
            let (mut widest, mut spilled) = (0, false);
            for (width, state) in &states {
                let draws = &state.inner.compiled.draws;
                assert!(
                    (draws.fields.iter()).any(|(place, _, value)| matches!(
                        (place, value),
                        (DrawPlace::Push(0), FieldValue::TessellationFactors)
                    )),
                    "{name}: {width} buffers: the factors' address at 0"
                );
                widest = widest.max(draws.push_len);
                spilled |= draws.blocks[crate::render::FRAGMENT].is_some();
                let mut buffers = Buffers::new(&device);
                let (points, _, factors) = patch_buffers(&device, &mut buffers);
                let tints = buffers.of(&device, &[0; 32]);
                let summed: Vec<Buffer> = (0..*width)
                    .map(|k| {
                        let mut value = [0.0; 4];
                        value[(k % 4) as usize] = f32::from(1u8 << (k / 4)) / 255.0;
                        buffers.of(&device, &floats(&value))
                    })
                    .collect();
                let color = target(
                    &device,
                    PixelFormat::RGBA8Unorm,
                    TextureUsage::RENDER_TARGET,
                );
                let queue = device.new_command_queue(4).expect("a queue");
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let draw =
                    |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
                draw(encoder.set_render_pipeline_state(state));
                draw(encoder.set_vertex_buffer(Some(&points), 0, 0));
                draw(encoder.set_vertex_buffer(Some(&tints), 0, 1));
                for (k, buffer) in summed.iter().enumerate() {
                    draw(encoder.set_fragment_buffer(Some(buffer), 0, k as u32));
                }
                draw(encoder.set_tessellation_factor_buffer(Some(&factors), 8, 0));
                draw(encoder.draw_patches(3, 0, 2, None, 1, 0));
                encoder.end_encoding();
                run(name, &command_buffer);
                let mut sums = [0u8; 4];
                for k in 0..*width {
                    sums[(k % 4) as usize] += 1 << (k / 4);
                }
                let black = [0, 0, 0, 255];
                let want: Vec<u8> = (0..SIDE * SIDE)
                    .flat_map(|at| if at / SIDE < 4 { sums } else { black })
                    .collect();
                assert_eq!(read(&color, 4), want, "{name}: {width} buffers: the texels");
                println!(
                    "{name}: {width} fragment buffers, a push range of {} bytes, the fragment \
                     function's block {:?}",
                    draws.push_len,
                    draws.blocks[crate::render::FRAGMENT]
                );
            }
            assert!(
                widest > 248,
                "{name}: the widest pair's fields end at {widest}, not past 248"
            );
            assert!(spilled, "{name}: no fragment function's fields spill");
        });
    }

    /// What Metal refuses is refused, and what the runtime does not build is a gap.
    #[test]
    fn every_device_refuses_the_patch_draws_it_does_not_build() {
        let Some(libraries) = libraries() else {
            return;
        };
        let Some(examples) = examples() else {
            return;
        };
        let d = patches(&libraries);
        let plain = descriptor(&examples, "colored_fragment", PixelFormat::RGBA8Unorm);
        let mut per_instance = d.clone();
        let mut scaled = d.clone();
        let mut indexed = d.clone();
        tessellation(&mut per_instance).factor_step_function =
            TessellationFactorStepFunction::PerInstance;
        tessellation(&mut scaled).factor_scale_enabled = true;
        tessellation(&mut indexed).control_point_index_type =
            TessellationControlPointIndexType::UInt16;
        each_device("draw_patches_refused", |name, _, device| {
            for (case, d) in [("a per-instance step", &per_instance), ("a scale", &scaled)] {
                let got = device.new_render_pipeline_state(d, crate::tests::no_constexpr_samplers);
                assert!(
                    matches!(
                        got,
                        Err(Error::Unsupported(Gap::RenderPipeline {
                            kind: RenderPipelineKind::Tessellation
                        }))
                    ),
                    "{name}: {case}: {:?}",
                    got.err()
                );
            }
            let got =
                device.new_render_pipeline_state(&indexed, crate::tests::no_constexpr_samplers);
            assert!(
                matches!(
                    &got,
                    Err(Error::Translation(metal2vulkan::error::Error::Unsupported(u)))
                        if u.feature == Feature::Tessellation
                ),
                "{name}: a control-point index type: {:?}",
                got.err()
            );
            let patch_state = state(name, &device, &d);
            let plain_state = state(name, &device, &plain);
            let mut buffers = Buffers::new(&device);
            let (points, tints, factors) = patch_buffers(&device, &mut buffers);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let same = |got: Result<(), Error>, want: Result<(), Error>, case: &str| {
                assert_eq!(format!("{got:?}"), format!("{want:?}"), "{name}: {case}");
            };
            let refused = |refusal| Err::<(), _>(Error::Refused(refusal));
            encoder
                .set_render_pipeline_state(&patch_state)
                .expect("a state");
            encoder
                .set_vertex_buffer(Some(&points), 0, 0)
                .expect("a buffer");
            encoder
                .set_vertex_buffer(Some(&tints), 0, 1)
                .expect("a buffer");
            same(
                encoder.draw_patches(3, 0, 2, None, 1, 0),
                refused(Refusal::NoTessellationFactorBuffer),
                "no factor buffer",
            );
            same(
                encoder.set_tessellation_factor_buffer(Some(&factors), 9, 0),
                refused(Refusal::TessellationFactorOffset { offset: 9 }),
                "factors at an odd byte",
            );
            encoder
                .set_tessellation_factor_buffer(Some(&factors), 8, 0)
                .expect("the factors");
            same(
                encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0),
                refused(Refusal::PatchDraw),
                "a plain draw of a patch state",
            );
            same(
                encoder.draw_patches(3, 0, 2, Some((&factors, 0)), 1, 0),
                Err(Error::Unsupported(Gap::Draw {
                    kind: DrawKind::PatchIndexBuffer,
                })),
                "a patch index buffer",
            );
            encoder
                .set_render_pipeline_state(&plain_state)
                .expect("a state");
            same(
                encoder.draw_patches(3, 0, 2, None, 1, 0),
                refused(Refusal::PatchDraw),
                "a patch draw of a plain state",
            );
            encoder.end_encoding();
            run(name, &command_buffer);
            println!("{name}: each patch draw refused by kind");
        });
    }

    /// The user-space instructions a draw retires on the encoding thread, per draw of 512 in one
    /// encoder that change nothing between them: a plain draw of two buffers' triangle, and a
    /// patch draw of two buffers' patches and their factors. Printed per draw with the build
    /// profile, under the validation layer the device tests load, where the kernel offers the
    /// counter (Linux). A patch draw records what a plain draw records, so it is held within a
    /// quarter of one.
    #[test]
    fn every_device_counts_a_patch_draw_s_instructions() {
        use metal2vulkan_devtools::instructions::InstructionCounter;
        const DRAWS: u32 = 512;
        let Some(libraries) = libraries() else {
            return;
        };
        let Some(examples) = examples() else {
            return;
        };
        let d = patches(&libraries);
        let plain = descriptor(&examples, "colored_fragment", PixelFormat::RGBA8Unorm);
        each_device("draw_patches_instructions", |name, _, device| {
            let Ok(mut counter) = InstructionCounter::new() else {
                println!("{name}: no instruction counter here");
                return;
            };
            let patch_state = state(name, &device, &d);
            let plain_state = state(name, &device, &plain);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let (points, tints, factors) = patch_buffers(&device, &mut buffers);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let mut per_draw = Vec::new();
            for patched in [false, true] {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let one = |encoder: &mut crate::RenderCommandEncoder| match patched {
                    false => encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0),
                    true => encoder.draw_patches(3, 0, 2, None, 1, 0),
                };
                if patched {
                    draw(encoder.set_render_pipeline_state(&patch_state));
                    draw(encoder.set_vertex_buffer(Some(&points), 0, 0));
                    draw(encoder.set_vertex_buffer(Some(&tints), 0, 1));
                    draw(encoder.set_tessellation_factor_buffer(Some(&factors), 8, 0));
                } else {
                    draw(encoder.set_render_pipeline_state(&plain_state));
                    draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                    draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                }
                // One draw first, so the counted ones record what a draw records alone.
                draw(one(&mut encoder));
                let ((), counted) = (counter.measure(|| {
                    for _ in 0..DRAWS {
                        draw(one(&mut encoder));
                    }
                }))
                .expect("a count");
                encoder.end_encoding();
                run(name, &command_buffer);
                per_draw.push(counted / u64::from(DRAWS));
            }
            let profile = if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            };
            println!(
                "{name}: user-space instructions a draw ({profile} build, validation layer \
                 loaded): {} a plain draw, {} a patch draw",
                per_draw[0], per_draw[1]
            );
            assert!(
                per_draw[1] * 4 <= per_draw[0] * 5,
                "{name}: a patch draw records what a plain draw records ({} against {})",
                per_draw[1],
                per_draw[0]
            );
        });
    }
}
