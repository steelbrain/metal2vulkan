//! A kernel's read of a texture of a format held for copies alone
//! (`metal2vulkan::metal::format::copies_only`): the dispatch that binds it at a `float` slot is
//! refused `NotImplemented(Format)`, the format's own refusal (the binder's `bound_sampled_type`),
//! as a render pass naming such a texture is. While the binder took the binding for a mismatch,
//! the dispatch was made, its command buffer completed, and every texel read the neutral
//! texture's zeros, where Metal reads the texture's values (reims-vgpu's measurement on an Apple
//! M3 Max under macOS 27.0.1, `read_texels_float` over 16x16 textures that `replaceRegion:`
//! filled with `0x5a`). The texture's copies still move its bytes.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use crate::{Error, PipelineOptions, Region, ResourceOptions, Size, TextureDescriptor};
use metal2vulkan::error::{BindError, Feature, Needs};
use metal2vulkan::metal::format::{copies_only, footprint, Footprint};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// A kernel that reads every texel of texture 0 into buffer 0, `float4`s in rows of the grid.
const READ: &str = r#"; Owned synthetic kernel.
source_filename = "copies_only_reads.metal"

define void @read_texels(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)

!air.kernel = !{!0}
!air.version = !{!30}
!air.language_version = !{!31}
!0 = !{ptr @read_texels, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!6 = !{i32 3, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"size"}
!30 = !{i32 2, i32 8, i32 0}
!31 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The side of each texture.
const SIDE: u64 = 16;

/// The byte each texture's texels hold.
const FILL: u8 = 0x5a;

/// Whether `error` is the format's own refusal, `NotImplemented(Format)`.
fn format_refused(error: &Error) -> bool {
    matches!(
        error,
        Error::Bind(BindError::Unsupported(u))
            if (u.feature, &u.needs) == (Feature::Format, &Needs::NotImplemented)
    )
}

/// Each format held for copies alone, a 16x16 `ShaderRead` texture of it filled with [`FILL`] by
/// a host copy: a dispatch of [`READ`] that binds it is refused `NotImplemented(Format)`, and its
/// bytes read back as they were written.
#[test]
fn a_kernel_s_read_of_a_texture_held_for_copies_alone_is_refused() {
    let scratch = Scratch::new("copies-only-reads");
    let device = device(&scratch.0);
    let air = metal2vulkan_devtools::llvm_assemble_text(READ).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name("read_texels")
        .unwrap();
    let state = device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap();
    let out = device
        .new_buffer(16 * SIDE * SIDE, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let held: Vec<PixelFormat> = (PixelFormat::ALL.iter().copied())
        .filter(|&pf| copies_only(pf))
        .collect();
    assert_eq!(held.len(), 20, "{held:?}");
    for pf in held {
        let bytes = match footprint(pf) {
            Some(Footprint::Tiled { bytes, .. } | Footprint::OnePlane { bytes, .. }) => {
                u64::from(bytes)
            }
            other => panic!("{pf:?}: {other:?}"),
        };
        let mut descriptor = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        let texture = device.new_texture(&descriptor).unwrap();
        let row = bytes * SIDE;
        let written = vec![FILL; (row * SIDE) as usize];
        let region = Region::new_2d(0, 0, SIDE, SIDE);
        texture.replace_region(region, 0, 0, &written, row).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&state).unwrap();
        encoder.set_texture(Some(&texture), 0).unwrap();
        encoder.set_buffer(Some(&out), 0, 0).unwrap();
        match encoder.dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(8, 8, 1)) {
            Err(error) if format_refused(&error) => {}
            other => panic!("{pf:?}: the read's dispatch answered {other:?}"),
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let mut read = vec![0; written.len()];
        texture.get_bytes(&mut read, row, region, 0).unwrap();
        assert_eq!(read, written, "{pf:?}: the texture's bytes");
    }
}
