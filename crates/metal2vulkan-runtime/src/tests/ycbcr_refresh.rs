//! A two-plane YCbCr texture's conversion through its refresh (`ycbcr.rs`): the planes' words
//! compared with the copy the last conversion read, the conversion dispatched through indirect
//! arguments that run no workgroup where no word changed.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The surface is a `Shared` buffer
//! standing for a CoreVideo `420v` surface of 40 x 24, laid out as the validation lane's
//! (`runtime_lane/ycbcr_textures.rs`): the luma plane at byte 0, rows 64 bytes apart, the
//! interleaved chroma plane after it, rows 64 bytes apart. The oracle is a texture made over the
//! same bytes afterwards, whose first use converts every texel.

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    Buffer, ComputePipelineState, Device, IoSurfacePlane, IoSurfaceYCbCr, PipelineOptions,
    ResourceOptions, Size, Texture, TextureDescriptor,
};
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
/// The words of the refresh's state the test reads: the indirect arguments, the completed word
/// the refresh read, and the completed word.
const STATE_WORDS: u64 = 5;

/// A kernel that reads every texel of texture 0 into buffer 0, `float4`s in rows of the grid.
const READ: &str = r#"; Owned synthetic kernel.
source_filename = "ycbcr_refresh.metal"

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

/// The read kernel's state on `device`.
fn read_state(device: &Device) -> ComputePipelineState {
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

/// `buffer`'s bytes `range`, through a shared buffer a blit copies them into.
fn bytes_of(device: &Device, buffer: &Buffer, at: u64, length: u64) -> Vec<u8> {
    let back = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(buffer, at, &back, 0, length)
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let contents = back.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) }.to_vec()
}

/// Every texel `state` reads of `texture`, in one command buffer of its own; then the refresh's
/// state words of the conversion it made, and the conversion's workgroups across.
fn read(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
) -> (Vec<u8>, Vec<u32>, u32) {
    let length = 16 * W * H;
    let out = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder
        .dispatch_threads(Size::new(W, H, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let texels = bytes_of(device, &out, 0, length);
    let copies = texture.shadow().expect("a converted texture's shadow");
    let Some(conversion) = copies.planes_conversion() else {
        panic!("a converted texture's source");
    };
    let (staging, at) = conversion
        .refresh_state()
        .expect("the conversion's refresh");
    let words = bytes_of(device, staging, at, 4 * STATE_WORDS)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    (texels, words, conversion.groups_across())
}

/// Write `bytes` over the surface's, as the CPU writes an IOSurface's planes unseen.
fn write(surface: &Buffer, bytes: &[u8]) {
    let contents = surface.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses now (each read waited for its command
    // buffer).
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
}

/// A texture's first use converts every texel (its copy completed no conversion); a use whose
/// planes are as the last conversion read them dispatches the conversion with no workgroup, and
/// reads the same texels; and after each change the CPU makes to the planes' bytes (a luma byte,
/// a chroma byte at a chroma row's end, the last chroma byte, many bytes) the conversion runs whole
/// and reads what a texture made over the same bytes afterwards reads, word for word. Each
/// conversion goes through the refresh (`WorkCounts::ycbcr_refreshes`), and leaves its completed
/// word set. A texture made again over the same surface converts at its first use.
#[test]
fn a_ycbcr_texture_converts_only_where_its_planes_changed() {
    let scratch = Scratch::new("ycbcr-refresh");
    let device = device(&scratch.0);
    let state = read_state(&device);
    let mut bytes = surface_bytes(7);
    let surface = device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap();
    let texture = texture(&surface);
    let before = device.work_counts();
    let (first, words, groups) = read(&device, &state, &texture);
    assert_eq!(words, [groups, 3, 1, 0, 1], "the first use converts");
    let (again, words, _) = read(&device, &state, &texture);
    assert_eq!(words, [0, 3, 1, 1, 1], "unchanged planes convert nothing");
    assert_eq!(again, first, "the texels of unchanged planes");
    let changes: [&[usize]; 4] = [
        &[(5 * ROW + 7) as usize],
        &[(CHROMA_AT + 3 * ROW + 2 * CW - 1) as usize],
        &[(CHROMA_AT + (CH - 1) * ROW + 2 * CW - 1) as usize],
        &[
            0,
            9,
            (11 * ROW + 33) as usize,
            (CHROMA_AT + 6 * ROW + 20) as usize,
        ],
    ];
    for (n, changed) in changes.into_iter().enumerate() {
        for &at in changed {
            bytes[at] ^= 0x5a;
        }
        write(&surface, &bytes);
        let (texels, words, _) = read(&device, &state, &texture);
        assert_eq!(words, [groups, 3, 1, 1, 1], "change {n} converts");
        let oracle = self::texture(&surface);
        let (want, _, _) = read(&device, &state, &oracle);
        assert_eq!(texels, want, "change {n}: the texels of a whole conversion");
        assert_ne!(texels, first, "change {n} changed texels");
        let (unchanged, words, _) = read(&device, &state, &texture);
        assert_eq!(words, [0, 3, 1, 1, 1], "change {n}, then unchanged");
        assert_eq!(unchanged, want);
    }
    let made_again = self::texture(&surface);
    let (texels, words, _) = read(&device, &state, &made_again);
    assert_eq!(words, [groups, 3, 1, 0, 1], "a texture made again converts");
    let (want, _, _) = read(&device, &state, &texture);
    assert_eq!(texels, want);
    let after = device.work_counts();
    // Two first reads, then three per change (the texture's, the oracle's first, the texture's
    // again), and the texture made again and the texture once more.
    assert_eq!(
        (
            after.ycbcr_refreshes - before.ycbcr_refreshes,
            after.ycbcr_whole_conversions - before.ycbcr_whole_conversions,
        ),
        (2 + 3 * 4 + 2, 0),
        "conversions through the refresh, and whole"
    );
}
