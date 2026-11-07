//! A texture read's `{T, i8}` result in the shapes Apple's GL code generator writes for a read it
//! swizzles: the texel taken out, rearranged, written back into the read's own result with
//! `insertvalue`, and taken out again. Run over a texture buffer and compared with the values the
//! IR computes:
//!
//! - **swizzle**: texel `i` read, its channels `zyxw`, written back and taken again;
//! - **combined**: texels `i`, `i + 1` and `i + 2` read, the second's and third's `x` placed in
//!   the first's `y` and `z`, written back into the first read and taken again.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The modules are LLVM text written here
//! (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    ComputePipelineState, Device, PipelineOptions, Region, ResourceOptions, Size, Texture,
    TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The texels of the texture buffer.
const TEXELS: u64 = 16;

/// Texel `i`'s channels.
fn texel(i: u64) -> [f32; 4] {
    let x = i as f32;
    [x, 10.0 + x, 20.0 + x, 30.0 + x]
}

/// A kernel `@k` over `texture_buffer<float, read>` 0 and `float4 *out` (buffer 0) whose thread
/// `gid` runs `body` (reading through `@read`, the read sampler `%s`) and stores `%v` to
/// `out[gid]`.
fn kernel(body: &str) -> String {
    format!(
        r#"; Owned synthetic kernel.
define void @k(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, i32 %gid) {{
entry:
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
{body}
  %j = zext i32 %gid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}}

declare ptr addrspace(2) @air.get_read_sampler()
declare {{ <4 x float>, i8 }} @air.read_texture_buffer_1d.v4f32(ptr addrspace(1), ptr addrspace(2), i32, i32)

!air.kernel = !{{!0}}
!air.version = !{{!30}}
!air.language_version = !{{!31}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture_buffer<float, read>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!30 = !{{i32 2, i32 8, i32 0}}
!31 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The read of texel `at` as `name`.
fn read(name: &str, at: &str) -> String {
    format!(
        "  %{name} = tail call {{ <4 x float>, i8 }} @air.read_texture_buffer_1d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, i32 {at}, i32 1)\n"
    )
}

/// The swizzle shape (`zyxw`).
fn swizzle() -> String {
    format!(
        "{}  %a0 = extractvalue {{ <4 x float>, i8 }} %a, 0\n\
         \x20 %abc = shufflevector <4 x float> %a0, <4 x float> undef, <4 x i32> <i32 2, i32 1, i32 0, i32 3>\n\
         \x20 %w = insertvalue {{ <4 x float>, i8 }} %a, <4 x float> %abc, 0\n\
         \x20 %v = extractvalue {{ <4 x float>, i8 }} %w, 0",
        read("a", "%gid")
    )
}

/// The combined shape: texels `gid`, `gid + 1` and `gid + 2`.
fn combined() -> String {
    format!(
        "  %g1 = add i32 %gid, 1\n  %g2 = add i32 %gid, 2\n{}{}{}\
         \x20 %a0 = extractvalue {{ <4 x float>, i8 }} %a, 0\n\
         \x20 %b0 = extractvalue {{ <4 x float>, i8 }} %b, 0\n\
         \x20 %c0 = extractvalue {{ <4 x float>, i8 }} %c, 0\n\
         \x20 %bx = extractelement <4 x float> %b0, i64 0\n\
         \x20 %ab = insertelement <4 x float> %a0, float %bx, i64 1\n\
         \x20 %cx = extractelement <4 x float> %c0, i64 0\n\
         \x20 %abc = insertelement <4 x float> %ab, float %cx, i64 2\n\
         \x20 %w = insertvalue {{ <4 x float>, i8 }} %a, <4 x float> %abc, 0\n\
         \x20 %v = extractvalue {{ <4 x float>, i8 }} %w, 0",
        read("a", "%gid"),
        read("b", "%g1"),
        read("c", "%g2")
    )
}

/// The state of `@k` with `body`.
fn state(device: &Device, body: &str) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(&kernel(body)).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name("k")
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

/// The texture buffer of [`TEXELS`] texels, texel `i` [`texel`]`(i)`.
fn texels(device: &Device) -> Texture {
    let descriptor = TextureDescriptor::texture_buffer(
        PixelFormat::RGBA32Float,
        TEXELS,
        ResourceOptions::default(),
        TextureUsage::SHADER_READ,
    );
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes: Vec<u8> = (0..TEXELS)
        .flat_map(texel)
        .flat_map(f32::to_le_bytes)
        .collect();
    texture
        .replace_region(Region::new_1d(0, TEXELS), 0, 0, &bytes, 16 * TEXELS, 0)
        .unwrap();
    texture
}

/// `state` run over `threads` threads reading `texture`: each thread's `float4`.
fn run(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
    threads: u64,
) -> Vec<[f32; 4]> {
    let out = device
        .new_buffer(16 * threads, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder
        .dispatch_threads(Size::new(threads, 1, 1), Size::new(threads, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses now.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), 16 * threads as usize) };
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|t| {
            [0, 1, 2, 3]
                .map(|c| f32::from_le_bytes([t[4 * c], t[4 * c + 1], t[4 * c + 2], t[4 * c + 3]]))
        })
        .collect()
}

#[test]
fn a_read_written_back_into_its_own_result_reads_its_texels() {
    let scratch = Scratch::new("texture-aggregates");
    let device = device(&scratch.0);
    let texture = texels(&device);
    let swizzled: Vec<[f32; 4]> = (0..TEXELS)
        .map(|i| {
            let [x, y, z, w] = texel(i);
            [z, y, x, w]
        })
        .collect();
    assert_eq!(
        run(&device, &state(&device, &swizzle()), &texture, TEXELS),
        swizzled
    );
    let threads = TEXELS - 2;
    let combined_texels: Vec<[f32; 4]> = (0..threads)
        .map(|i| {
            let [x, _, _, w] = texel(i);
            [x, texel(i + 1)[0], texel(i + 2)[0], w]
        })
        .collect();
    assert_eq!(
        run(&device, &state(&device, &combined()), &texture, threads),
        combined_texels
    );
}
