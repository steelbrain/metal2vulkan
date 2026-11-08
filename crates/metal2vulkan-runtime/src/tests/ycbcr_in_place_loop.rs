//! A two-plane YCbCr texture whose planes are read in place, sampled in a loop.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The surface is a `Shared` buffer
//! standing for a CoreVideo `420v` surface of 40 x 24 (the layout of `ycbcr_refresh.rs`).
//!
//! On a device that reads the planes in place (an integrated GPU, a CPU device), the sample's
//! variant reads the texture's words from its push field, a `uvec4[2]` loaded at the function's
//! entry; a sample in a loop reads them in the loop. On MoltenVK the module materializes that
//! load (`LoadsForwardedIntoLoops`), and the copy of the array must be one Metal's compiler takes:
//! the whole array's `OpCopyObject` became `spvUnsafeArray<uint4, 2> x = push.words;`, which it
//! refused, failing the pipeline (`VK_ERROR_INITIALIZATION_FAILED`, the guest's Core Animation
//! fragment that samples a video frame in a loop). The loop's texels are the loop-free sample's.

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    Buffer, ComputePipelineState, Device, IoSurfacePlane, IoSurfaceYCbCr, PipelineOptions,
    ResourceOptions, SamplerState, Size, Texture, TextureDescriptor,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, TextureUsage};

const W: u64 = 40;
const H: u64 = 24;
const ROW: u64 = 64;
const CHROMA_AT: u64 = ROW * H;
/// The chroma plane's pairs across and rows.
const CW: u64 = W / 2;
const CH: u64 = H / 2;
/// The surface's bytes: both planes, with room past the chroma plane's last row.
const LENGTH: u64 = CHROMA_AT + ROW * CH;
/// `'420v'`.
const VIDEO: u32 = u32::from_be_bytes(*b"420v");
/// The loop's iterations.
const ITERATIONS: u32 = 3;

/// A kernel that samples texture 0 through sampler 0 at each texel's centre `n[0]` times in a
/// loop (`looped`), or once (`once`), storing the sample into buffer 1, `float4`s in rows of the
/// grid.
fn kernel(name: &str, looped: bool) -> String {
    let body = match looped {
        true => {
            r#"  %count = load i32, ptr addrspace(1) %n, align 4
  %any = icmp ne i32 %count, 0
  br i1 %any, label %loop, label %done
loop:
  %k = phi i32 [ 0, %entry ], [ %k1, %loop ]
  %r = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %val = extractvalue { <4 x float>, i8 } %r, 0
  store <4 x float> %val, ptr addrspace(1) %p, align 16
  %k1 = add i32 %k, 1
  %more = icmp ult i32 %k1, %count
  br i1 %more, label %loop, label %done
done:
  ret void"#
        }
        false => {
            r#"  %r = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %val = extractvalue { <4 x float>, i8 } %r, 0
  store <4 x float> %val, ptr addrspace(1) %p, align 16
  ret void"#
        }
    };
    format!(
        r#"; Owned synthetic kernel.
source_filename = "ycbcr_in_place_loop.metal"

define void @{name}(ptr addrspace(1) %t, ptr addrspace(2) %s, ptr addrspace(1) %n, ptr addrspace(1) noalias %out, <2 x i32> %gid, <2 x i32> %size) {{
entry:
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %h = extractelement <2 x i32> %size, i64 1
  %fx = uitofp i32 %x to float
  %fy = uitofp i32 %y to float
  %fw = uitofp i32 %w to float
  %fh = uitofp i32 %h to float
  %cx = fadd float %fx, 5.000000e-01
  %cy = fadd float %fy, 5.000000e-01
  %u = fdiv float %cx, %fw
  %v = fdiv float %cy, %fh
  %uv0 = insertelement <2 x float> undef, float %u, i64 0
  %uv = insertelement <2 x float> %uv0, float %v, i64 1
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
{body}
}}

declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{{!0}}
!air.version = !{{!30}}
!air.language_version = !{{!31}}
!0 = !{{ptr @{name}, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6, !7, !8}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"n"}}
!6 = !{{i32 3, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!7 = !{{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}}
!8 = !{{i32 5, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"size"}}
!30 = !{{i32 2, i32 8, i32 0}}
!31 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The state of kernel `name` (`kernel`) on `device`.
fn state(device: &Device, name: &str, looped: bool) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(&kernel(name, looped)).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name(name)
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

/// Bytes of both planes from `seed` (a linear congruential sequence), every other byte `0xa5`.
fn surface_bytes(seed: u32) -> Vec<u8> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 24) as u8
    };
    let mut bytes = vec![0xa5; LENGTH as usize];
    for y in 0..H {
        for x in 0..W {
            bytes[(y * ROW + x) as usize] = next();
        }
    }
    for y in 0..CH {
        for x in 0..2 * CW {
            bytes[(CHROMA_AT + y * ROW + x) as usize] = next();
        }
    }
    bytes
}

/// A `YCBCR8_420_2P` texture of `surface`'s plane 0.
fn texture(surface: &Buffer) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::YCBCR84202P, W, H, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let facts = IoSurfaceYCbCr::new(VIDEO, 0, [W, H], CHROMA_AT, ROW);
    surface
        .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, ROW, true).with_ycbcr(facts))
        .unwrap()
}

/// Every texel `state` samples of `texture` through `sampler`, in one command buffer of its own,
/// as the bytes of its `float4`s.
fn sample(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
    sampler: &SamplerState,
) -> Vec<u8> {
    let length = 16 * W * H;
    let out = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let count = device
        .new_buffer_with_bytes(&ITERATIONS.to_le_bytes(), ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_sampler_state(Some(sampler), 0).unwrap();
    encoder.set_buffer(Some(&count), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(W, H, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer
        .commit()
        .expect("the commit makes the sample's pipeline");
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses now (the command buffer completed).
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) }.to_vec()
}

/// A sample in a loop of a texture whose planes are read in place makes its pipeline and reads
/// the loop-free sample's texels. Where the device reads the planes in place
/// (`WorkCounts::ycbcr_textures_in_place`), both dispatches bind the planes
/// (`ycbcr_planes_bound`); elsewhere both sample the texture's image.
#[test]
fn a_sample_in_a_loop_of_planes_read_in_place_makes_its_pipeline() {
    let scratch = Scratch::new("ycbcr-in-place-loop");
    let device = device(&scratch.0);
    let looped = state(&device, "sample_loop", true);
    let once = state(&device, "sample_once", false);
    let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
    let bytes = surface_bytes(11);
    let surface = device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap();
    let before = device.work_counts();
    let texture = texture(&surface);
    let in_place = device.work_counts().ycbcr_textures_in_place - before.ycbcr_textures_in_place;
    let want = sample(&device, &once, &texture, &sampler);
    let got = sample(&device, &looped, &texture, &sampler);
    assert_eq!(got, want, "the loop's texels are the loop-free sample's");
    assert_ne!(got, vec![0; got.len()], "the samples read the texture");
    let bound = device.work_counts().ycbcr_planes_bound - before.ycbcr_planes_bound;
    let reads = device
        .compile_device()
        .profile()
        .reads_ycbcr_planes_in_place();
    eprintln!("planes in place: {reads}; textures in place {in_place}, planes bound {bound}");
    match reads {
        true => {
            assert_eq!(in_place, 1, "the texture reads its planes in place");
            assert!(bound >= 2, "both dispatches bind the planes ({bound})");
        }
        false => assert_eq!((in_place, bound), (0, 0)),
    }
}

/// A kernel that reads every texel of texture 0 into buffer 0, `float4`s in rows of the grid: a
/// `read()`, which the in-place form does not read, so it binds the texture's converted image.
const READ: &str = r#"; Owned synthetic kernel.
source_filename = "ycbcr_in_place_read.metal"

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

/// The pipeline state of `READ`.
pub(super) fn reader(device: &Device) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(READ).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name("read_texels")
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

/// Every texel of `texture` read (`READ`) in one command buffer of its own.
fn read(device: &Device, texture: &Texture) {
    read_with(device, &reader(device), texture, [W, H]);
}

/// The texels of `texture` in the `grid` from its origin read by `state`, [`reader`]'s, in one
/// command buffer of its own.
pub(super) fn read_with(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
    [w, h]: [u64; 2],
) {
    let out = device
        .new_buffer(16 * w * h, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder
        .dispatch_threads(Size::new(w, h, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// A texture whose planes every function so far read in place holds no conversion: its
/// samples took no conversion's scratch buffer (`WorkCounts::ycbcr_conversion_scratch_buffers`),
/// however many dispatches sampled it. Its first `read()`, which reads the converted image, makes
/// the conversion, once. On a device that keeps the image alone, the texture's first sample
/// makes it.
#[test]
fn a_texture_read_only_in_place_makes_no_conversion() {
    let scratch = Scratch::new("ycbcr-in-place-conversion");
    let device = device(&scratch.0);
    let once = state(&device, "sample_once", false);
    let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
    let surface = device
        .new_buffer_with_bytes(&surface_bytes(13), ResourceOptions::shared())
        .unwrap();
    let reads = device
        .compile_device()
        .profile()
        .reads_ycbcr_planes_in_place();
    let scratches = || device.work_counts().ycbcr_conversion_scratch_buffers;
    let before = scratches();
    let texture = texture(&surface);
    assert_eq!(
        scratches() - before,
        0,
        "the texture is made with no conversion"
    );
    sample(&device, &once, &texture, &sampler);
    sample(&device, &once, &texture, &sampler);
    let sampled = scratches() - before;
    read(&device, &texture);
    read(&device, &texture);
    let all = scratches() - before;
    eprintln!(
        "planes in place: {reads}; conversions after the samples {sampled}, after the reads {all}"
    );
    match reads {
        true => assert_eq!((sampled, all), (0, 1)),
        false => assert_eq!((sampled, all), (1, 1)),
    }
}
