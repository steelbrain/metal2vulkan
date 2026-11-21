//! Metal's pull-model interpolation where some of the quad's invocations may have left: under a
//! device cfg, on every device of the cfg's kind, `colored_vertex` (the owned fixture
//! `examples/runtime_examples.ll`) with fragment functions written here that read its colour, the
//! same at each of a triangle's vertices, as an `interpolant` at sample 0 or at an offset of
//! (0.25, 0.25):
//! - past the function's first block (in a branch on the pixel's column), at a sample or at an
//!   offset, and after a `discard_fragment` in the first block: the state is refused on every
//!   device, `Unsupported(DivergentInterpolation)`, a typed gap of its own;
//! - the same reads in the first block: the state is made and draws the colour, which is every
//!   read's value.
//!
//! `llvm-as` assembles the functions; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{
        descriptor, examples, pass, read, run, state, target, texels, vertices, Buffers, BLACK,
        GREEN,
    };
    use crate::{Error, Library, PixelFormat, PrimitiveType, StageSource, TextureUsage};
    use metal2vulkan::error::{Feature, Needs};
    use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};

    /// The fragment functions' module but their bodies (`BODIES`).
    const TEMPLATE: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

BODIES
declare void @air.discard_fragment() local_unnamed_addr #1
declare <4 x float> @air.interpolate_sample_perspective.v4f32(ptr addrspace(1) readonly captures(none), i32) local_unnamed_addr #2
declare <4 x float> @air.interpolate_offset_perspective.v4f32(ptr addrspace(1) readonly captures(none), <2 x float>) local_unnamed_addr #2

attributes #0 = { mustprogress nounwind willreturn }
attributes #1 = { mustprogress nounwind willreturn }
attributes #2 = { mustprogress nofree nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.fragment = !{!9, !14, !15, !16, !17}
!air.compile_options = !{!21, !22}
!air.version = !{!18}
!air.language_version = !{!19}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @first_block, !10, !11}
!10 = !{!20}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!13 = !{i32 1, !"air.fragment_input", !"generated(5colorDv4_f)", !"air.interpolation_function", !"air.perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!14 = !{ptr @sample_in_branch, !10, !11}
!15 = !{ptr @offset_in_branch, !10, !11}
!16 = !{ptr @after_discard, !10, !11}
!17 = !{ptr @discard_after_reads, !10, !11}
!18 = !{i32 2, i32 8, i32 0}
!19 = !{!"Metal", i32 4, i32 0, i32 0}
!20 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!21 = !{!"air.compile.denorms_disable"}
!22 = !{!"air.compile.fast_math_enable"}
"#;

    /// The functions: each reads the colour at sample 0 and at the offset, and returns their
    /// mean. `first_block` reads both in its first block; `sample_in_branch` and
    /// `offset_in_branch` read one of them in a branch taken right of column 4, and the other in
    /// the first block; `after_discard` discards in its first block, then reads both there;
    /// `discard_after_reads` reads both, then discards where both channels' green is below a half
    /// (never, for a green colour).
    const BODIES: &str = r#"define <4 x float> @first_block(<4 x float> %0, ptr addrspace(1) readonly captures(none) %1) local_unnamed_addr #0 {
  %s = tail call <4 x float> @air.interpolate_sample_perspective.v4f32(ptr addrspace(1) readonly captures(none) %1, i32 0)
  %o = tail call <4 x float> @air.interpolate_offset_perspective.v4f32(ptr addrspace(1) readonly captures(none) %1, <2 x float> <float 2.500000e-01, float 2.500000e-01>)
  %t = fadd <4 x float> %s, %o
  %m = fmul <4 x float> %t, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  ret <4 x float> %m
}

define <4 x float> @sample_in_branch(<4 x float> %0, ptr addrspace(1) readonly captures(none) %1) local_unnamed_addr #0 {
  %o = tail call <4 x float> @air.interpolate_offset_perspective.v4f32(ptr addrspace(1) readonly captures(none) %1, <2 x float> <float 2.500000e-01, float 2.500000e-01>)
  %x = extractelement <4 x float> %0, i64 0
  %right = fcmp ogt float %x, 4.000000e+00
  br i1 %right, label %read, label %done

read:
  %s = tail call <4 x float> @air.interpolate_sample_perspective.v4f32(ptr addrspace(1) readonly captures(none) %1, i32 0)
  br label %done

done:
  %v = phi <4 x float> [ %s, %read ], [ %o, %2 ]
  ret <4 x float> %v
}

define <4 x float> @offset_in_branch(<4 x float> %0, ptr addrspace(1) readonly captures(none) %1) local_unnamed_addr #0 {
  %s = tail call <4 x float> @air.interpolate_sample_perspective.v4f32(ptr addrspace(1) readonly captures(none) %1, i32 0)
  %x = extractelement <4 x float> %0, i64 0
  %right = fcmp ogt float %x, 4.000000e+00
  br i1 %right, label %read, label %done

read:
  %o = tail call <4 x float> @air.interpolate_offset_perspective.v4f32(ptr addrspace(1) readonly captures(none) %1, <2 x float> <float 2.500000e-01, float 2.500000e-01>)
  br label %done

done:
  %v = phi <4 x float> [ %o, %read ], [ %s, %2 ]
  ret <4 x float> %v
}

define <4 x float> @after_discard(<4 x float> %0, ptr addrspace(1) readonly captures(none) %1) local_unnamed_addr #0 {
  tail call void @air.discard_fragment()
  %s = tail call <4 x float> @air.interpolate_sample_perspective.v4f32(ptr addrspace(1) readonly captures(none) %1, i32 0)
  %o = tail call <4 x float> @air.interpolate_offset_perspective.v4f32(ptr addrspace(1) readonly captures(none) %1, <2 x float> <float 2.500000e-01, float 2.500000e-01>)
  %t = fadd <4 x float> %s, %o
  ret <4 x float> %t
}

define <4 x float> @discard_after_reads(<4 x float> %0, ptr addrspace(1) readonly captures(none) %1) local_unnamed_addr #0 {
  %s = tail call <4 x float> @air.interpolate_sample_perspective.v4f32(ptr addrspace(1) readonly captures(none) %1, i32 0)
  %o = tail call <4 x float> @air.interpolate_offset_perspective.v4f32(ptr addrspace(1) readonly captures(none) %1, <2 x float> <float 2.500000e-01, float 2.500000e-01>)
  %t = fadd <4 x float> %s, %o
  %m = fmul <4 x float> %t, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %g = extractelement <4 x float> %m, i64 1
  %dim = fcmp olt float %g, 5.000000e-01
  br i1 %dim, label %drop, label %keep

drop:
  tail call void @air.discard_fragment()
  br label %keep

keep:
  ret <4 x float> %m
}
"#;

    /// The library of the fragment functions, or None where `llvm-as` is missing.
    fn fragments() -> Option<Library> {
        match llvm_assemble_text(&TEMPLATE.replace("BODIES", BODIES)) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the fragment library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("the fragments: {error:?}"),
        }
    }

    /// A read at a sample or an offset past the first block, or after a discard in it, is one
    /// typed gap on every device; the same reads in the first block, and before a discard, draw
    /// the colour.
    #[test]
    fn every_device_refuses_interpolation_where_the_quad_may_have_left() {
        let (Some(examples), Some(fragments)) = (examples(), fragments()) else {
            return;
        };
        let with = |name: &str| {
            let mut d = descriptor(&examples, "colored_fragment", PixelFormat::RGBA8Unorm);
            d.fragment = Some(StageSource::Function(crate::FunctionDescriptor::new(
                fragments.function(name).expect("the function"),
            )));
            d
        };
        each_device("divergent_interpolation", |name, _, device| {
            for function in ["sample_in_branch", "offset_in_branch", "after_discard"] {
                match device
                    .new_render_pipeline_state(&with(function), crate::tests::no_constexpr_samplers)
                {
                    Err(Error::Translation {
                        error: metal2vulkan::error::Error::Unsupported(u),
                        ..
                    }) if u.feature == Feature::DivergentInterpolation
                        && u.needs == Needs::NotImplemented => {}
                    Err(error) => panic!("{name}: {function}: {error}"),
                    Ok(_) => panic!("{name}: {function} was made"),
                }
            }
            for function in ["first_block", "discard_after_reads"] {
                let state = state(&format!("{name}: {function}"), &device, &with(function));
                let mut buffers = Buffers::new(&device);
                let (positions, colors) = vertices(&device, &mut buffers);
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
                draw(encoder.set_render_pipeline_state(&state));
                draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                encoder.end_encoding();
                run(name, &command_buffer);
                let want = texels(|_, y| match y {
                    0..4 => GREEN,
                    _ => BLACK,
                });
                assert_eq!(read(&color, 4), want, "{name}: {function}: the texels");
            }
            println!(
                "{name}: three reads where the quad may have left refused, two in the first block drawn"
            );
        });
    }
}
