//! Images in compute pipelines of OpenCL mode (`ComputePipelineDescriptor::open_cl_mode_enabled`),
//! each kernel in the shape Apple's OpenCL compiler writes OpenCL C's image functions, run with the
//! textures, samplers and hidden buffers bound as Apple's OpenCL binds them, and compared with what
//! an M3 Max (macOS 27.0.1) and a Radeon Pro 5300M (macOS 26.7.1) compute for kernels of the same
//! shapes over the same data:
//!
//! - **formats**: `get_image_channel_order` and `get_image_channel_data_type` answer the words of
//!   the third hidden buffer (`__image_order_data_type_table`) of the image's texture index `t`:
//!   `2t + 1` for the order, `2t` for the data type. With the buffer's word `i` holding
//!   `i * 0x10000 + 0x1234`: the order of the 1D image at texture 1 is `0x31234`, the data type of
//!   the 2D array image at 2 `0x41234`, and the 2D image at 0 answers `0x11234` and `0x1234`.
//! - **lines**: a 1D image bound as a 2D texture of height 1 and a 1D array image as a 2D array
//!   texture of height 1, as Apple's OpenCL binds them, read with and without a sampler and
//!   queried: each read answers the 1D image's texel (row 0, of the layer an array read names;
//!   a linear filter at a border-clamped edge blends in no row of the border), and the width and
//!   layer count are the image's.
//! - **wlines**: writes to the same two bindings land in row 0, of the layer named.
//! - **real lines**: the same reads over a real 1D texture and a real 1D array texture, which
//!   Apple's OpenCL never binds, answer the 1D images' values, layer 1 of the array: the Radeon
//!   Pro 5300M's answer (an Intel UHD 630 reads layer 0 of the array, an M3 Max texel 0 at every
//!   coordinate).
//!
//! Under a device cfg, on every device of the cfg's kind. The modules are LLVM text written here;
//! `llvm-as` assembles them, and without it the tests skip visibly.

use crate::tests::command::on_device::{each_device, finished};
use crate::tests::opencl_mode::{
    buffer, floats, floats_buffer, kernel_list, library, state, words, words_buffer,
};
use crate::{
    Buffer, ComputePipelineState, Device, DispatchType, IndexClass, PixelFormat, Region,
    SamplerAddressMode, SamplerDescriptor, SamplerMinMagFilter, SamplerState, StorageMode, Texture,
    TextureDescriptor, TextureType, TextureUsage,
};

/// The 1D images' width, in texels, and the threads that read or write them.
const WIDTH: u32 = 16;

/// The node of argument `n`, a texture at `index` of the MSL type `ty` (`texture1d<unknown,
/// read>`), its access `access` (`sample` or `write`).
pub(super) fn image(n: u32, index: u32, access: &str, ty: &str, name: &str) -> String {
    format!(
        "!{{i32 {n}, !\"air.texture\", !\"air.location_index\", i32 {index}, i32 1, !\"air.{access}\", !\"air.arg_type_name\", !\"{ty}\", !\"air.arg_name\", !\"{name}\"}}"
    )
}

/// The node of argument `n`, a sampler at `index`.
pub(super) fn sampler_node(n: u32, index: u32, name: &str) -> String {
    format!(
        "!{{i32 {n}, !\"air.sampler\", !\"air.location_index\", i32 {index}, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"{name}\"}}"
    )
}

/// The node of argument `n`, a read-write `float4` buffer at `index`.
pub(super) fn float4s(n: u32, index: u32, name: &str) -> String {
    format!(
        "!{{i32 {n}, !\"air.buffer\", !\"air.location_index\", i32 {index}, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 16, !\"air.arg_type_align_size\", i32 16, !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"{name}\"}}"
    )
}

/// `__kernel void formats(__read_only image2d_t a, __read_only image1d_t b, __read_only
/// image2d_array_t c, __global int *o)`: `o[0..4]` take the order of `b`, the data type of `c`, and
/// the order and the data type of `a`.
fn formats() -> String {
    let body = "define void @__entry_formats(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #0 {\n\
entry:\n\
  tail call fastcc void @formats(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3)\n\
  ret void\n}\n\
define internal fastcc void @formats(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %o) #1 {\n\
entry:\n\
  %ob = tail call fastcc i32 @_Z23get_image_channel_order14ocl_image1d_ro(ptr addrspace(1) %b)\n\
  store i32 %ob, ptr addrspace(1) %o, align 4\n\
  %dc = tail call fastcc i32 @_Z27get_image_channel_data_type20ocl_image2d_array_ro(ptr addrspace(1) %c)\n\
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 1\n\
  store i32 %dc, ptr addrspace(1) %o1, align 4\n\
  %oa = tail call fastcc i32 @_Z23get_image_channel_order14ocl_image2d_ro(ptr addrspace(1) %a)\n\
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 2\n\
  store i32 %oa, ptr addrspace(1) %o2, align 4\n\
  %da = tail call fastcc i32 @_Z27get_image_channel_data_type14ocl_image2d_ro(ptr addrspace(1) %a)\n\
  %o3 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 3\n\
  store i32 %da, ptr addrspace(1) %o3, align 4\n\
  ret void\n}\n\
define internal fastcc i32 @_Z23get_image_channel_order14ocl_image1d_ro(ptr addrspace(1) %0) #2 {\n\
  %2 = tail call i32 @air.get_channel_order_texture_1d(ptr addrspace(1) %0) #0\n\
  ret i32 %2\n}\n\
define internal fastcc i32 @_Z27get_image_channel_data_type20ocl_image2d_array_ro(ptr addrspace(1) %0) #2 {\n\
  %2 = tail call i32 @air.get_channel_data_type_texture_2d_array(ptr addrspace(1) %0) #0\n\
  ret i32 %2\n}\n\
define internal fastcc i32 @_Z23get_image_channel_order14ocl_image2d_ro(ptr addrspace(1) %0) #2 {\n\
  %2 = tail call i32 @air.get_channel_order_texture_2d(ptr addrspace(1) %0) #0\n\
  ret i32 %2\n}\n\
define internal fastcc i32 @_Z27get_image_channel_data_type14ocl_image2d_ro(ptr addrspace(1) %0) #2 {\n\
  %2 = tail call i32 @air.get_channel_data_type_texture_2d(ptr addrspace(1) %0) #0\n\
  ret i32 %2\n}\n\
declare i32 @air.get_channel_order_texture_1d(ptr addrspace(1)) #0\n\
declare i32 @air.get_channel_data_type_texture_2d_array(ptr addrspace(1)) #0\n\
declare i32 @air.get_channel_order_texture_2d(ptr addrspace(1)) #0\n\
declare i32 @air.get_channel_data_type_texture_2d(ptr addrspace(1)) #0\n";
    let args = [
        image(0, 0, "sample", "texture2d<unknown, read>", "a"),
        image(1, 1, "sample", "texture1d<unknown, read>", "b"),
        image(2, 2, "sample", "texture2d_array<unknown, read>", "c"),
        buffer(3, 0, "read_write", 1, "int", "o"),
    ];
    format!("{body}{}", kernel_list("__entry_formats", &args))
}

/// `__kernel void lines(__read_only image1d_t a, __read_only image1d_array_t b, sampler_t s,
/// sampler_t l, __global float4 *o)`: at `x = get_global_id(0)`, `o[6x..6x + 6]` take `a` read at
/// `x`, sampled through `s` at `x` and through `l` at `(x + 0.5) / 16`, layer 1 of `b` read at `x`
/// and sampled through `l` at `(x + 0.5) / 16`, and `(width(a), width(b), array_size(b), 0)`.
fn lines() -> String {
    let body = "define void @__entry_lines(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2, ptr addrspace(2) %3, ptr addrspace(1) %4) #0 {\n\
entry:\n\
  tail call fastcc void @lines(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(2) %2, ptr addrspace(2) %3, ptr addrspace(1) %4)\n\
  ret void\n}\n\
define internal fastcc void @lines(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(2) %s, ptr addrspace(2) %l, ptr addrspace(1) %o) #1 {\n\
entry:\n\
  %g = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %x = trunc i64 %g to i32\n\
  %base = mul i64 %g, 6\n\
  %fx = sitofp i32 %x to float\n\
  %h = fadd float %fx, 5.000000e-01\n\
  %nx = fmul float %h, 6.250000e-02\n\
  %rs = tail call ptr addrspace(2) @air.get_read_sampler()\n\
  %r0 = tail call { <4 x float>, i8 } @air.read_texture_1d.v4f32(ptr addrspace(1) %a, ptr addrspace(2) %rs, i32 %x, i32 0, i32 1)\n\
  %t0 = extractvalue { <4 x float>, i8 } %r0, 0\n\
  %p0 = getelementptr inbounds <4 x float>, ptr addrspace(1) %o, i64 %base\n\
  store <4 x float> %t0, ptr addrspace(1) %p0, align 16\n\
  %r1 = tail call { <4 x float>, i8 } @air.sample_texture_1d.v4f32(ptr addrspace(1) %a, ptr addrspace(2) %s, float %fx, i1 false, i32 0, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)\n\
  %t1 = extractvalue { <4 x float>, i8 } %r1, 0\n\
  %i1 = add i64 %base, 1\n\
  %p1 = getelementptr inbounds <4 x float>, ptr addrspace(1) %o, i64 %i1\n\
  store <4 x float> %t1, ptr addrspace(1) %p1, align 16\n\
  %r2 = tail call { <4 x float>, i8 } @air.sample_texture_1d.v4f32(ptr addrspace(1) %a, ptr addrspace(2) %l, float %nx, i1 false, i32 0, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)\n\
  %t2 = extractvalue { <4 x float>, i8 } %r2, 0\n\
  %i2 = add i64 %base, 2\n\
  %p2 = getelementptr inbounds <4 x float>, ptr addrspace(1) %o, i64 %i2\n\
  store <4 x float> %t2, ptr addrspace(1) %p2, align 16\n\
  %r3 = tail call { <4 x float>, i8 } @air.read_texture_1d_array.v4f32(ptr addrspace(1) %b, ptr addrspace(2) %rs, i32 %x, i32 1, i32 0, i32 1)\n\
  %t3 = extractvalue { <4 x float>, i8 } %r3, 0\n\
  %i3 = add i64 %base, 3\n\
  %p3 = getelementptr inbounds <4 x float>, ptr addrspace(1) %o, i64 %i3\n\
  store <4 x float> %t3, ptr addrspace(1) %p3, align 16\n\
  %r4 = tail call { <4 x float>, i8 } @air.sample_texture_1d_array.v4f32(ptr addrspace(1) %b, ptr addrspace(2) %l, float %nx, i32 1, i1 false, i32 0, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)\n\
  %t4 = extractvalue { <4 x float>, i8 } %r4, 0\n\
  %i4 = add i64 %base, 4\n\
  %p4 = getelementptr inbounds <4 x float>, ptr addrspace(1) %o, i64 %i4\n\
  store <4 x float> %t4, ptr addrspace(1) %p4, align 16\n\
  %wa = tail call i32 @air.get_width_texture_1d(ptr addrspace(1) %a, i32 0)\n\
  %wb = tail call i32 @air.get_width_texture_1d_array(ptr addrspace(1) %b, i32 0)\n\
  %nb = tail call i32 @air.get_array_size_texture_1d_array(ptr addrspace(1) %b)\n\
  %fwa = sitofp i32 %wa to float\n\
  %fwb = sitofp i32 %wb to float\n\
  %fnb = uitofp i32 %nb to float\n\
  %v0 = insertelement <4 x float> <float poison, float poison, float poison, float 0.000000e+00>, float %fwa, i64 0\n\
  %v1 = insertelement <4 x float> %v0, float %fwb, i64 1\n\
  %v2 = insertelement <4 x float> %v1, float %fnb, i64 2\n\
  %i5 = add i64 %base, 5\n\
  %p5 = getelementptr inbounds <4 x float>, ptr addrspace(1) %o, i64 %i5\n\
  store <4 x float> %v2, ptr addrspace(1) %p5, align 16\n\
  ret void\n}\n\
declare ptr addrspace(2) @air.get_read_sampler() #0\n\
declare { <4 x float>, i8 } @air.read_texture_1d.v4f32(ptr addrspace(1), ptr addrspace(2), i32, i32, i32) #0\n\
declare { <4 x float>, i8 } @air.read_texture_1d_array.v4f32(ptr addrspace(1), ptr addrspace(2), i32, i32, i32, i32) #0\n\
declare { <4 x float>, i8 } @air.sample_texture_1d.v4f32(ptr addrspace(1), ptr addrspace(2), float, i1, i32, i1, float, float, i32) #0\n\
declare { <4 x float>, i8 } @air.sample_texture_1d_array.v4f32(ptr addrspace(1), ptr addrspace(2), float, i32, i1, i32, i1, float, float, i32) #0\n\
declare i32 @air.get_width_texture_1d(ptr addrspace(1), i32) #0\n\
declare i32 @air.get_width_texture_1d_array(ptr addrspace(1), i32) #0\n\
declare i32 @air.get_array_size_texture_1d_array(ptr addrspace(1)) #0\n";
    let args = [
        image(0, 0, "sample", "texture1d<unknown, read>", "a"),
        image(1, 1, "sample", "texture1d_array<unknown, read>", "b"),
        sampler_node(2, 0, "s"),
        sampler_node(3, 1, "l"),
        float4s(4, 0, "o"),
    ];
    format!("{body}{}", kernel_list("__entry_lines", &args))
}

/// `__kernel void wlines(__write_only image1d_t a, __write_only image1d_array_t b)`: at
/// `x = get_global_id(0)`, `a` takes `(x, 1, 2, 3)` at `x` and layer 1 of `b` `(x, 4, 5, 6)`.
fn wlines() -> String {
    let body = "define void @__entry_wlines(ptr addrspace(1) %0, ptr addrspace(1) %1) #0 {\n\
entry:\n\
  tail call fastcc void @wlines(ptr addrspace(1) %0, ptr addrspace(1) %1)\n\
  ret void\n}\n\
define internal fastcc void @wlines(ptr addrspace(1) %a, ptr addrspace(1) %b) #1 {\n\
entry:\n\
  %g = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %x = trunc i64 %g to i32\n\
  %fx = sitofp i32 %x to float\n\
  %ta = insertelement <4 x float> <float poison, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, float %fx, i64 0\n\
  tail call void @air.write_texture_1d.v4f32(ptr addrspace(1) %a, i32 %x, <4 x float> %ta, i32 0, i32 2)\n\
  %tb = insertelement <4 x float> <float poison, float 4.000000e+00, float 5.000000e+00, float 6.000000e+00>, float %fx, i64 0\n\
  tail call void @air.write_texture_1d_array.v4f32(ptr addrspace(1) %b, i32 %x, i32 1, <4 x float> %tb, i32 0, i32 2)\n\
  ret void\n}\n\
declare void @air.write_texture_1d.v4f32(ptr addrspace(1), i32, <4 x float>, i32, i32) #0\n\
declare void @air.write_texture_1d_array.v4f32(ptr addrspace(1), i32, i32, <4 x float>, i32, i32) #0\n";
    let args = [
        image(0, 0, "write", "texture1d<unknown, write>", "a"),
        image(1, 1, "write", "texture1d_array<unknown, write>", "b"),
    ];
    format!("{body}{}", kernel_list("__entry_wlines", &args))
}

/// A Shared `RGBA32Float` texture of `texture_type` at texture index `index`, `width` by
/// `height`, of `layers` layers, each of `texels` where given.
fn texture(
    device: &Device,
    texture_type: TextureType,
    (width, height, layers): (u32, u32, u32),
    texels: Option<&[f32]>,
    index: u32,
) -> Texture {
    let mut descriptor =
        TextureDescriptor::new(texture_type, PixelFormat::RGBA32Float, width, height);
    descriptor.array_length = layers;
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    descriptor.storage_mode = StorageMode::Shared;
    let texture = (device.new_texture(&descriptor, Some(index)))
        .unwrap_or_else(|error| panic!("{descriptor:?}: {error}"));
    let image = (width * height * 4) as usize;
    if let Some(texels) = texels {
        for (layer, texels) in (0u32..).zip(texels.chunks(image)) {
            let bytes: Vec<u8> = texels.iter().flat_map(|f| f.to_le_bytes()).collect();
            let row = u64::from(16 * width);
            let region = Region {
                origin: [0; 3],
                size: [width, height, 1],
            };
            let bytes = crate::HostBytes::from(&bytes[..]);
            (texture.replace_region(region, 0, layer, bytes, row, row * u64::from(height)))
                .unwrap_or_else(|error| panic!("layer {layer}: {error}"));
        }
    }
    texture
}

/// Row 0 of layer `layer` of `texture`, `WIDTH` texels wide, as floats.
fn row(texture: &Texture, layer: u32) -> Vec<f32> {
    let mut bytes = vec![0u8; (16 * WIDTH) as usize];
    let region = Region {
        origin: [0; 3],
        size: [WIDTH, 1, 1],
    };
    let row = u64::from(16 * WIDTH);
    (texture.get_bytes(&mut bytes, row, row, region, 0, layer)).expect("the row");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect()
}

/// A sampler state at sampler index `index` of `filter`, `address` and `normalized` coordinates.
pub(super) fn sampler(
    device: &Device,
    filter: SamplerMinMagFilter,
    address: SamplerAddressMode,
    normalized: bool,
    index: u32,
) -> SamplerState {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.min_filter = filter;
    descriptor.mag_filter = filter;
    descriptor.s_address_mode = address;
    descriptor.t_address_mode = address;
    descriptor.r_address_mode = address;
    descriptor.normalized_coordinates = normalized;
    (device.new_sampler_state(&descriptor, index)).expect("a sampler state")
}

/// What `state` binds besides its hidden buffers.
pub(super) struct Bound<'a> {
    pub(super) textures: &'a [&'a Texture],
    pub(super) samplers: &'a [&'a SamplerState],
    /// The kernel's own buffer at 3 (its index 0), if any.
    pub(super) out: Option<&'a Buffer>,
}

/// `state` dispatched on `device` (named `name`) over `threads` threads, one dimension, its hidden
/// buffers bound as Apple's OpenCL binds them (the first's bytes at index 0 no offset and one
/// dimension, the third holding word `i` as `i * 0x10000 + 0x1234`), and `bound`'s textures,
/// samplers and buffer at their indices; committed and finished without an error.
pub(super) fn run(
    name: &str,
    device: &Device,
    state: &ComputePipelineState,
    threads: u32,
    bound: &Bound<'_>,
) {
    let table: Vec<u32> = (0..64).map(|i| i * 0x10000 + 0x1234).collect();
    let first: Vec<u8> = [0u32, 0, 0, 1]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let hidden = [words_buffer(device, &[0; 16]), words_buffer(device, &table)];
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
    if let Some(out) = bound.out {
        encoder
            .set_buffer(Some(out), 0, 3)
            .expect("the kernel's buffer");
    }
    for (index, t) in (0u32..).zip(bound.textures) {
        (encoder.set_texture(Some(t), index))
            .unwrap_or_else(|error| panic!("{name}: texture {index}: {error}"));
    }
    for (index, s) in (0u32..).zip(bound.samplers) {
        (encoder.set_sampler_state(Some(s), index))
            .unwrap_or_else(|error| panic!("{name}: sampler {index}: {error}"));
    }
    (encoder.dispatch_threads([threads, 1, 1], [threads, 1, 1]))
        .unwrap_or_else(|error| panic!("{name}: the dispatch: {error}"));
    encoder.end_encoding();
    command_buffer.commit().expect("a commit");
    finished(name, &command_buffer);
    assert_eq!(command_buffer.error(), None, "{name}");
}

/// The 1D image's texels: lane `i` of the row is `10 + i`.
fn line() -> Vec<f32> {
    (0..WIDTH * 4).map(|i| 10.0 + i as f32).collect()
}

/// The 1D array image's texels: lane `i` of its two layers is `1000 + i`.
fn layers() -> Vec<f32> {
    (0..2 * WIDTH * 4).map(|i| 1000.0 + i as f32).collect()
}

#[test]
fn opencl_mode_image_formats_read_the_third_hidden_buffer() {
    let Some(formats) = library(&formats()) else {
        return;
    };
    each_device("opencl-images-formats", |name, _, device| {
        let _grant = (device.grant_indices(IndexClass::Texture, 1, 3))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let formats = state(&device, &formats, "__entry_formats");
        let a = texture(&device, TextureType::Type2D, (8, 4, 1), None, 1);
        let b = texture(&device, TextureType::Type2D, (WIDTH, 1, 1), None, 2);
        let c = texture(&device, TextureType::Type2DArray, (8, 4, 3), None, 3);
        let o = words_buffer(&device, &[0xeeee_eeee; 4]);
        let bound = Bound {
            textures: &[&a, &b, &c],
            samplers: &[],
            out: Some(&o),
        };
        run(name, &device, &formats, 1, &bound);
        assert_eq!(words(&o), [0x31234, 0x41234, 0x11234, 0x1234], "{name}");
        println!("{name}: each image's order and data type are the third hidden buffer's words");
    });
}

/// [`lines`] run on `device` (named `name`) over `a`, its 1D image, and `b`, its 1D array image
/// (each holding [`line`] and [`layers`]), with a nearest sampler of unnormalized coordinates and a
/// linear one of normalized coordinates clamped to zero: every read, sample and size query answers
/// the 1D images' values.
fn lines_read(name: &str, device: &Device, lines: &ComputePipelineState, a: &Texture, b: &Texture) {
    let (line, layers) = (line(), layers());
    let _grant = (device.grant_indices(IndexClass::Sampler, 1, 2))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    let s = sampler(
        device,
        SamplerMinMagFilter::Nearest,
        SamplerAddressMode::ClampToEdge,
        false,
        1,
    );
    let l = sampler(
        device,
        SamplerMinMagFilter::Linear,
        SamplerAddressMode::ClampToZero,
        true,
        2,
    );
    let o = floats_buffer(device, &vec![-1.0; (WIDTH * 6 * 4) as usize]);
    let bound = Bound {
        textures: &[a, b],
        samplers: &[&s, &l],
        out: Some(&o),
    };
    run(name, device, lines, WIDTH, &bound);
    let got = floats(&o);
    for x in 0..WIDTH as usize {
        let at = |column: usize| &got[(6 * x + column) * 4..(6 * x + column + 1) * 4];
        let texel = &line[4 * x..4 * x + 4];
        let layer1 = &layers[(WIDTH as usize + x) * 4..(WIDTH as usize + x + 1) * 4];
        assert_eq!(at(0), texel, "{name}: read of a at {x}");
        assert_eq!(at(1), texel, "{name}: sample of a through s at {x}");
        assert_eq!(at(2), texel, "{name}: sample of a through l at {x}");
        assert_eq!(at(3), layer1, "{name}: read of b's layer 1 at {x}");
        assert_eq!(
            at(4),
            layer1,
            "{name}: sample of b's layer 1 through l at {x}"
        );
        assert_eq!(at(5), [16.0, 16.0, 2.0, 0.0], "{name}: sizes at {x}");
    }
}

#[test]
fn opencl_mode_1d_images_read_their_2d_bindings_row_0() {
    let Some(lines) = library(&lines()) else {
        return;
    };
    each_device("opencl-images-lines", |name, _, device| {
        let _grant = (device.grant_indices(IndexClass::Texture, 1, 2))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let lines = state(&device, &lines, "__entry_lines");
        let (line, layers) = (line(), layers());
        let a = texture(&device, TextureType::Type2D, (WIDTH, 1, 1), Some(&line), 1);
        let b = texture(
            &device,
            TextureType::Type2DArray,
            (WIDTH, 1, 2),
            Some(&layers),
            2,
        );
        lines_read(name, &device, &lines, &a, &b);
        println!("{name}: 1D images read row 0 of their 2D bindings");
    });
}

/// A 1D image bound as a real 1D texture and a 1D array image as a real 1D array texture, which
/// Apple's OpenCL never binds, read as the 1D images they are: the Radeon Pro 5300M's answer
/// (macOS 26.7.1), every read, sample and size query the 1D image's value, layer 1 of the array.
#[test]
fn opencl_mode_1d_images_read_real_1d_textures() {
    let Some(lines) = library(&lines()) else {
        return;
    };
    each_device("opencl-images-real-lines", |name, _, device| {
        let _grant = (device.grant_indices(IndexClass::Texture, 1, 2))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let lines = state(&device, &lines, "__entry_lines");
        let (line, layers) = (line(), layers());
        let a = texture(&device, TextureType::Type1D, (WIDTH, 1, 1), Some(&line), 1);
        let b = texture(
            &device,
            TextureType::Type1DArray,
            (WIDTH, 1, 2),
            Some(&layers),
            2,
        );
        lines_read(name, &device, &lines, &a, &b);
        println!("{name}: 1D images read real 1D textures");
    });
}

#[test]
fn opencl_mode_1d_images_write_their_2d_bindings_row_0() {
    let Some(wlines) = library(&wlines()) else {
        return;
    };
    each_device("opencl-images-wlines", |name, _, device| {
        let _grant = (device.grant_indices(IndexClass::Texture, 1, 2))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let wlines = state(&device, &wlines, "__entry_wlines");
        let zeros = vec![0.0; (2 * WIDTH * 4) as usize];
        let one_layer = &zeros[..(WIDTH * 4) as usize];
        let a = texture(
            &device,
            TextureType::Type2D,
            (WIDTH, 1, 1),
            Some(one_layer),
            1,
        );
        let b = texture(
            &device,
            TextureType::Type2DArray,
            (WIDTH, 1, 2),
            Some(&zeros),
            2,
        );
        let bound = Bound {
            textures: &[&a, &b],
            samplers: &[],
            out: None,
        };
        run(name, &device, &wlines, WIDTH, &bound);
        let texels = |lanes: [f32; 3]| -> Vec<f32> {
            (0..WIDTH)
                .flat_map(|x| [x as f32, lanes[0], lanes[1], lanes[2]])
                .collect()
        };
        assert_eq!(row(&a, 0), texels([1.0, 2.0, 3.0]), "{name}");
        assert_eq!(row(&b, 0), vec![0.0; (WIDTH * 4) as usize], "{name}");
        assert_eq!(row(&b, 1), texels([4.0, 5.0, 6.0]), "{name}");
        println!("{name}: 1D images write row 0 of their 2D bindings");
    });
}
