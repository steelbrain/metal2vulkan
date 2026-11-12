//! A heap texture sampled through the constexpr sampler each lane chooses among three (the
//! translator's constexpr sampler array, `ResourceKind::SamplerArray`) answers, lane by lane, what
//! a kernel sampling it through that lane's sampler alone answers, and what Metal's address modes
//! define: an `R32Float` texture four texels wide holding `10, 11, 12, 13`, sampled nearest at
//! level 0 and `(1.375, 0.5)`, reads texel 3 clamped to the edge, texel 1 repeating, and texel 2
//! mirrored.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernels are LLVM text written here
//! (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    Buffer, CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, Region,
    ResourceOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};

/// The constexpr samplers' words: nearest, normalized coordinates, no mip filter, every axis
/// clamped to the edge, repeating, and mirrored, with the fields Apple's compiler writes for the
/// rest.
const SAMPLERS: [u64; 3] = [
    0x007b_ff00_0008_0000 | 0o111,
    0x007b_ff00_0008_0000 | 0o222,
    0x007b_ff00_0008_0000 | 0o333,
];

/// The texels of the texture's one row.
const TEXELS: [f32; 4] = [10.0, 11.0, 12.0, 13.0];

/// What each sampler reads at `(1.375, 0.5)`: texel 3 clamped to the edge, texel 1 repeating
/// (`0.375`), texel 2 mirrored (`0.625`).
const READ: [f32; 3] = [13.0, 11.0, 12.0];

/// The lanes of a dispatch: each sampler twice.
const LANES: usize = 6;

/// A kernel `k` sampling the `texture2d<float>` its argument buffer (buffer 0) holds at level 0
/// and `(1.375, 0.5)` through `pick[gid % pick.len()]` (indices of [`SAMPLERS`], each a constexpr
/// sampler of the module), written to buffer 1's lane `gid`.
fn kernel(pick: &[usize]) -> String {
    let n = pick.len();
    let mut choose = format!("  %sel = urem i32 %gid, {n}\n");
    let mut chosen = format!("@s{}", pick[n - 1]);
    for (k, s) in pick[..n - 1].iter().enumerate().rev() {
        choose += &format!(
            "  %c{k} = icmp eq i32 %sel, {k}\n  %p{k} = select i1 %c{k}, ptr addrspace(2) @s{s}, ptr addrspace(2) {chosen}\n"
        );
        chosen = format!("%p{k}");
    }
    let globals: String = (SAMPLERS.iter().enumerate())
        .map(|(i, w)| {
            format!("@s{i} = internal addrspace(2) constant [2 x i64] [i64 {w}, i64 0], align 8\n")
        })
        .collect();
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.M = type {{ ptr addrspace(1) }}
{globals}
define void @k(ptr addrspace(2) %m, ptr addrspace(1) %out, i32 %gid) {{
  %pt = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
{choose}  %r = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {chosen}, <2 x float> <float 1.375, float 0.5>, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  %o = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %gid
  store <4 x float> %v, ptr addrspace(1) %o, align 16
  ret void
}}
declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
!air.kernel = !{{!0}}
!air.version = !{{!11}}
!air.language_version = !{{!12}}
!air.sampler_states = !{{!20, !21, !22}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !6, !8}}
!3 = !{{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"M", !"air.arg_name", !"m"}}
!4 = !{{i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"t", !"air.indirect_argument", !5}}
!5 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!6 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!8 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!11 = !{{i32 2, i32 8, i32 0}}
!12 = !{{!"Metal", i32 4, i32 0, i32 0}}
!20 = !{{!"air.sampler_state", ptr addrspace(2) @s0}}
!21 = !{{!"air.sampler_state", ptr addrspace(2) @s1}}
!22 = !{{!"air.sampler_state", ptr addrspace(2) @s2}}
"#
    )
}

/// The compute state of [`kernel`]`(pick)` on `device`.
fn state(device: &Device, pick: &[usize]) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(&kernel(pick)).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name("k")
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A shared buffer holding `bytes`.
fn holding(device: &Device, bytes: &[u8]) -> Buffer {
    device
        .new_buffer_with_bytes(bytes, ResourceOptions::shared())
        .unwrap()
}

/// The `R32Float` texture of one row of [`TEXELS`].
fn texture(device: &Device) -> Texture {
    let width = TEXELS.len() as u64;
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::R32Float, width, 1, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes: Vec<u8> = TEXELS.iter().flat_map(|t| t.to_le_bytes()).collect();
    let row = bytes.len() as u64;
    texture
        .replace_region(Region::new_2d(0, 0, width, 1), 0, 0, &bytes, row, 0)
        .unwrap();
    texture
}

/// What [`kernel`]`(pick)` reads of `texture`, named by its resource ID in an argument buffer, in
/// each of [`LANES`] lanes.
fn run(device: &Device, pick: &[usize], texture: &Texture) -> Vec<[f32; 4]> {
    let state = state(device, pick);
    let argument = holding(device, &texture.gpu_resource_id().0.to_le_bytes());
    let out = holding(device, &[0; 16 * LANES]);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_buffer(Some(&argument), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    let threads = Size::new(LANES as u64, 1, 1);
    encoder.dispatch_threads(threads, threads).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's bytes, which the completed command buffer wrote.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), 16 * LANES) };
    let words: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&word| f32::from_le_bytes(word))
        .collect();
    words.as_chunks::<4>().0.to_vec()
}

#[test]
fn a_lane_sampling_through_its_chosen_constexpr_sampler_reads_what_that_sampler_reads() {
    let scratch = Scratch::new("heap-sampler-array");
    let device = device(&scratch.0);
    let texture = texture(&device);
    let chosen = run(&device, &[0, 1, 2], &texture);
    for (sampler, read) in READ.iter().enumerate() {
        let alone = run(&device, &[sampler], &texture);
        for lane in (sampler..LANES).step_by(SAMPLERS.len()) {
            assert_eq!(
                chosen[lane], alone[lane],
                "lane {lane}: through sampler {sampler} chosen and alone"
            );
            assert_eq!(
                chosen[lane],
                [*read, 0.0, 0.0, 1.0],
                "lane {lane}: sampler {sampler}'s texel"
            );
        }
    }
}
