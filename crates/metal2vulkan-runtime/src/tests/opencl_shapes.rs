//! Kernels of OpenCL mode (`ComputePipelineDescriptor::open_cl_mode_enabled`) in the shapes Apple's
//! OpenCL compiler writes for OpenCL C, each kernel our own and its module the text that compiler
//! made of it (Metal's OpenCL mode, `newLibraryWithSource` with `sourceLanguage` 1: the probe
//! `crates/metal2vulkan-conformance/probes/opencl_operand_encodings.m`, M3 Max, macOS 27.0.1, job
//! m3-max-76861), its attributes and type-based alias nodes left out. Each runs with the three
//! hidden buffers bound as Metal's OpenCL dispatcher binds them (the dispatch's global offset and
//! dimension count by `setBytes` at index 0, the second and third as buffers at 1 and 2, the
//! kernel's own buffers from 3), and each answer is computed here:
//!
//! - **write**: `write_imagef` on a write-only image, `air.write_texture_2d` of access 2 through
//!   `_Z12write_imagef14ocl_image2d_woDv2_iDv4_f`, through an identity view of a 4 by 2
//!   texture filled with `0xee`: work-item `i` writes `((17, 81, 145, 209) + i) / 255` to texel
//!   `(i & 3, i >> 2)`, and the texture's bytes read back are the written ones. Over the two
//!   identity views the probe `opencl_swizzled_images.m` measured (job m3-max-16336): an
//!   `RGBA8Unorm` view of an `RGBA8Unorm` texture made with the identity swizzle, whose writes the
//!   M3 Max stores as written, and an `RGBA8Unorm` view of an `RGBA8Uint` texture, the texture
//!   Apple's OpenCL binds for a `CL_RGBA`, `CL_UNORM_INT8` image.
//!
//! Under a device cfg, on every device of the cfg's kind. The modules are LLVM text written here;
//! `llvm-as` assembles them, and without it the tests skip visibly.

use crate::tests::command::on_device::{each_device, finished};
use crate::tests::opencl_mode::{kernel_list, library, state, words_buffer};
use crate::{
    Buffer, ComputePipelineState, Device, DispatchType, IndexClass, PixelFormat, Region,
    StorageMode, Texture, TextureDescriptor, TextureSwizzle, TextureSwizzleChannels, TextureType,
    TextureUsage,
};

/// The images' size, and the work-items that use them: work-item `i` takes texel
/// `(i & 3, i >> 2)`.
const W: u32 = 4;
const H: u32 = 2;
const N: u32 = W * H;

/// The node of argument `n`, a texture at `index` of the MSL type `ty`, its access `access`
/// (`sample` or `write`).
fn image(n: u32, index: u32, access: &str, ty: &str, name: &str) -> String {
    format!(
        "!{{i32 {n}, !\"air.texture\", !\"air.location_index\", i32 {index}, i32 1, !\"air.{access}\", !\"air.arg_type_name\", !\"{ty}\", !\"air.arg_name\", !\"{name}\"}}"
    )
}

/// What a dispatch binds besides the hidden buffers.
struct Bound<'a> {
    /// The textures from index 0.
    textures: &'a [&'a Texture],
    /// The kernel's own buffers, from index 3.
    buffers: &'a [&'a Buffer],
    /// The length of each `__local` argument, from threadgroup index 0.
    local: &'a [u32],
}

/// `state` dispatched on `device` (named `name`) over `threads` work-items in work-groups of
/// `group`, one dimension, the hidden buffers bound as Metal's OpenCL dispatcher binds them (the
/// first's bytes at index 0: no offset and one dimension) and `bound`'s textures, buffers and
/// `__local` lengths at their indices; committed and finished without an error.
fn run(
    name: &str,
    device: &Device,
    state: &ComputePipelineState,
    (threads, group): (u32, u32),
    bound: &Bound<'_>,
) {
    let first: Vec<u8> = [0u32, 0, 0, 1]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let hidden = [
        words_buffer(device, &[0; 16]),
        words_buffer(device, &[0; 16]),
    ];
    let queue = device.new_command_queue(1).expect("a queue");
    let command_buffer = queue.command_buffer().expect("a command buffer");
    let mut encoder =
        (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
    encoder
        .set_compute_pipeline_state(state)
        .expect("the state");
    encoder
        .set_bytes(&first, 0)
        .expect("the first hidden buffer");
    for (index, b) in (1u32..).zip(&hidden) {
        (encoder.set_buffer(Some(b), 0, index)).expect("a hidden buffer");
    }
    for (index, b) in (3u32..).zip(bound.buffers) {
        (encoder.set_buffer(Some(b), 0, index))
            .unwrap_or_else(|error| panic!("{name}: buffer {index}: {error}"));
    }
    for (index, t) in (0u32..).zip(bound.textures) {
        (encoder.set_texture(Some(t), index))
            .unwrap_or_else(|error| panic!("{name}: texture {index}: {error}"));
    }
    for (index, &length) in (0u32..).zip(bound.local) {
        (encoder.set_threadgroup_memory_length(length, index))
            .unwrap_or_else(|error| panic!("{name}: __local {index}: {error}"));
    }
    (encoder.dispatch_threads([threads, 1, 1], [group, 1, 1]))
        .unwrap_or_else(|error| panic!("{name}: the dispatch: {error}"));
    encoder.end_encoding();
    command_buffer.commit().expect("a commit");
    finished(name, &command_buffer);
    assert_eq!(command_buffer.error(), None, "{name}");
}

/// The images' whole region.
fn region() -> Region {
    Region {
        origin: [0; 3],
        size: [W, H, 1],
    }
}

/// A Shared 4 by 2 texture of `format` at texture index 1, usable as a view of another format,
/// holding `bytes` (four a texel).
fn texture(name: &str, device: &Device, format: PixelFormat, bytes: &[u8]) -> Texture {
    let mut descriptor = TextureDescriptor::new(TextureType::Type2D, format, W, H);
    descriptor.usage =
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::PIXEL_FORMAT_VIEW;
    descriptor.storage_mode = StorageMode::Shared;
    let texture = (device.new_texture(&descriptor, Some(1)))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    let row = u64::from(W) * 4;
    (texture.replace_region(
        region(),
        0,
        0,
        crate::HostBytes::from(bytes),
        row,
        row * u64::from(H),
    ))
    .unwrap_or_else(|error| panic!("{name}: {error}"));
    texture
}

/// The bytes of the 4 by 2 texture `texture`, four a texel.
fn texels(name: &str, texture: &Texture) -> Vec<u8> {
    let row = u64::from(W) * 4;
    let mut out = vec![0u8; (row * u64::from(H)) as usize];
    (texture.get_bytes(&mut out, row, row * u64::from(H), region(), 0, 0))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    out
}

/// `__kernel void ocl_write(__write_only image2d_t img)`: work-item `i` writes
/// `((17, 81, 145, 209) + i) / 255` to texel `(i & 3, i >> 2)`.
fn write() -> String {
    let body = "define void @__entry_ocl_write(ptr addrspace(1) %0) #0 {\n\
entry:\n\
  tail call fastcc void @ocl_write(ptr addrspace(1) %0)\n\
  ret void\n}\n\
define internal fastcc void @ocl_write(ptr addrspace(1) %img) #1 {\n\
entry:\n\
  %call = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %conv = trunc i64 %call to i32\n\
  %0 = tail call float @air.convert.f.f32.s.i32(i32 %conv)\n\
  %splat.splatinsert = insertelement <4 x float> poison, float %0, i64 0\n\
  %splat.splat = shufflevector <4 x float> %splat.splatinsert, <4 x float> poison, <4 x i32> zeroinitializer\n\
  %add = fadd <4 x float> %splat.splat, <float 1.700000e+01, float 8.100000e+01, float 1.450000e+02, float 2.090000e+02>\n\
  %div = fdiv <4 x float> %add, <float 2.550000e+02, float 2.550000e+02, float 2.550000e+02, float 2.550000e+02>, !fpmath !80\n\
  %and = and i32 %conv, 3\n\
  %vecinit = insertelement <2 x i32> undef, i32 %and, i64 0\n\
  %shr = ashr i32 %conv, 2\n\
  %vecinit2 = insertelement <2 x i32> %vecinit, i32 %shr, i64 1\n\
  tail call fastcc void @_Z12write_imagef14ocl_image2d_woDv2_iDv4_f(ptr addrspace(1) %img, <2 x i32> %vecinit2, <4 x float> %div)\n\
  ret void\n}\n\
define internal fastcc void @_Z12write_imagef14ocl_image2d_woDv2_iDv4_f(ptr addrspace(1) %0, <2 x i32> %1, <4 x float> %2) #2 {\n\
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %0, <2 x i32> %1, <4 x float> %2, i32 0, i32 2)\n\
  ret void\n}\n\
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32) #0\n\
declare float @air.convert.f.f32.s.i32(i32) #3\n\
!80 = !{float 2.500000e+00}\n";
    let args = [image(0, 0, "write", "texture2d<unknown, write>", "img")];
    format!("{body}{}", kernel_list("__entry_ocl_write", &args))
}

#[test]
fn opencl_mode_writes_through_an_identity_view_store_the_written_texels() {
    let Some(write) = library(&write()) else {
        return;
    };
    use TextureSwizzle::{Alpha, Blue, Green, Red};
    let identity = TextureSwizzleChannels {
        red: Red,
        green: Green,
        blue: Blue,
        alpha: Alpha,
    };
    // Byte `c` of texel `i` after the write: `0x11 + 0x40 c + i`.
    let want: Vec<u8> = (0..N as usize)
        .flat_map(|i| (0..4).map(move |c| (0x11 + 0x40 * c + i) as u8))
        .collect();
    each_device("opencl-shapes-write", |name, _, device| {
        let _grant = (device.grant_indices(IndexClass::Texture, 1, 2))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let write = state(&device, &write, "__entry_ocl_write");
        for (view, format) in [
            ("RGBA8Unorm of RGBA8Unorm", PixelFormat::RGBA8Unorm),
            ("RGBA8Unorm of RGBA8Uint", PixelFormat::RGBA8Uint),
        ] {
            let at = format!("{name}: {view}");
            let parent = texture(&at, &device, format, &[0xee; (4 * N) as usize]);
            let image = (parent.new_texture_view_with_swizzle(
                PixelFormat::RGBA8Unorm,
                TextureType::Type2D,
                0..1,
                0..1,
                identity,
                Some(2),
            ))
            .unwrap_or_else(|error| panic!("{at}: the view: {error}"));
            let bound = Bound {
                textures: &[&image],
                buffers: &[],
                local: &[],
            };
            run(&at, &device, &write, (N, N), &bound);
            assert_eq!(texels(&at, &parent), want, "{at}");
        }
        println!("{name}: writes through each identity view store the written texels");
    });
}
