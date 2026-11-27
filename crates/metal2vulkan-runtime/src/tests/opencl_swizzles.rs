//! Images of OpenCL mode (`ComputePipelineDescriptor::open_cl_mode_enabled`) over swizzled texture
//! views, as Apple's OpenCL binds an image of most channel orders, compared with what an M3 Max
//! (macOS 27.0.1) answers for kernels of the same shapes over the same views (our probe
//! `crates/metal2vulkan-conformance/probes/opencl_swizzled_images.m`, job m3-max-16336).
//!
//! Each view is 4 by 2, of an `RGBA8Unorm` or `R8Unorm` texture, made with the swizzle of one of
//! OpenCL's channel orders (`BGRA`, `ARGB`, `A`, `INTENSITY`, `LUMINANCE`; the identity as a
//! control). Texel `i` (`x + 4y`) of a filled texture holds bytes `(0x10 + i, 0x50 + i, 0x90 + i,
//! 0xd0 + i)` (`R8Unorm`: `0x10 + i`); a write stores `(0x11 + i, 0x51 + i, 0x91 + i, 0xd1 + i) /
//! 255` into a texture filled with `0xee`. The M3 Max answers, on every view, for plain Metal and
//! for OpenCL mode alike:
//!
//! - **query**: a write-only image the kernel only queries answers the view's width and height
//!   and, for its channel order and data type, the third hidden buffer's words, which no swizzle
//!   reaches (`4, 2, 0x11234, 0x1234`);
//! - **read**: a read-only image's read, without a sampler and through one (nearest, at the
//!   texel's centre), answers the texel swizzled: lane `l` the channel the swizzle's `l` names (a
//!   format's missing green and blue 0, its missing alpha 1), `Zero` 0 and `One` 1.
//!
//! A write-only image's write is stored as written, channel `c` from lane `c`: the runtime's rule
//! (its README's known gaps of textures, "A swizzle reaches a texture's sampled element alone";
//! docs/API.md's accepted differences). The M3 Max stores channel `c` from the first lane whose
//! source is `c`, and lane `c` where no lane names it, the swizzle undone (job m3-max-16336): on
//! `rgba8/BGRA` texel 0 it stores `[0x91, 0x51, 0x11, 0xd1]` where the runtime stores `[0x11, 0x51,
//! 0x91, 0xd1]`, and on `rgba8/ARGB`, `rgba8/A` and `r8/A` it differs too. A plain-Metal kernel's
//! write through an `RGBA8Unorm` view of `(B, G, R, A)` is stored the same way there (probe
//! `swizzled_view_stores.m`, job m3-max-76829), and Metal's validation refuses such a write: the
//! view reports no `ShaderWrite` usage.
//!
//! Not run here: a `read_write` image of a swizzled view (Metal's `read_write` texture, read then
//! written, in OpenCL mode; Apple's OpenCL compiler takes no `__read_write` image), which the M3
//! Max reads through the swizzle and writes with the swizzle undone. The runtime refuses its
//! dispatch as `Gap::SwizzledReadWrite`: "a read_write texture slot 0 bound to a swizzled texture
//! is not built".
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
    TextureDescriptor, TextureSwizzle, TextureSwizzleChannels, TextureType, TextureUsage,
};

/// The views' size, and the threads that use them: thread `i` takes texel `(i & 3, i >> 2)`.
const W: u32 = 4;
const H: u32 = 2;
const N: u32 = W * H;

/// The node of argument `n`, a texture at `index` of the MSL type `ty` (`texture2d<unknown,
/// read>`), its access `access` (`sample` or `write`).
fn image(n: u32, index: u32, access: &str, ty: &str, name: &str) -> String {
    format!(
        "!{{i32 {n}, !\"air.texture\", !\"air.location_index\", i32 {index}, i32 1, !\"air.{access}\", !\"air.arg_type_name\", !\"{ty}\", !\"air.arg_name\", !\"{name}\"}}"
    )
}

/// The node of argument `n`, a sampler at `index`.
fn sampler_node(n: u32, index: u32, name: &str) -> String {
    format!(
        "!{{i32 {n}, !\"air.sampler\", !\"air.location_index\", i32 {index}, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"{name}\"}}"
    )
}

/// The node of argument `n`, a read-write `float4` buffer at `index`.
fn float4s(n: u32, index: u32, name: &str) -> String {
    format!(
        "!{{i32 {n}, !\"air.buffer\", !\"air.location_index\", i32 {index}, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 16, !\"air.arg_type_align_size\", i32 16, !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"{name}\"}}"
    )
}

/// A sampler state at sampler index `index` of `filter`, `address` and `normalized` coordinates.
fn sampler(
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
struct Bound<'a> {
    textures: &'a [&'a Texture],
    samplers: &'a [&'a SamplerState],
    /// The kernel's own buffer at 3 (its index 0), if any.
    out: Option<&'a Buffer>,
}

/// `state` dispatched on `device` (named `name`) over `threads` threads, one dimension, its hidden
/// buffers bound as Apple's OpenCL binds them (the first's bytes at index 0 no offset and one
/// dimension, the third holding word `i` as `i * 0x10000 + 0x1234`), and `bound`'s textures,
/// samplers and buffer at their indices; committed and finished without an error.
fn run(name: &str, device: &Device, state: &ComputePipelineState, threads: u32, bound: &Bound<'_>) {
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

/// The thread's texel: `%g` its global id, `%i` that as `i32`, `%p` the texel's `<2 x i32>`.
const TEXEL: &str = "  %g = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %i = trunc i64 %g to i32\n\
  %x = and i32 %i, 3\n\
  %y = lshr i32 %i, 2\n\
  %p0 = insertelement <2 x i32> poison, i32 %x, i64 0\n\
  %p = insertelement <2 x i32> %p0, i32 %y, i64 1\n";

/// The value thread `i` writes, `%v`: `(0x11 + i, 0x51 + i, 0x91 + i, 0xd1 + i) / 255`.
const VALUE: &str = "  %fi = uitofp i32 %i to float\n\
  %v0 = insertelement <4 x float> poison, float %fi, i64 0\n\
  %vs = shufflevector <4 x float> %v0, <4 x float> poison, <4 x i32> zeroinitializer\n\
  %va = fadd <4 x float> %vs, <float 1.700000e+01, float 8.100000e+01, float 1.450000e+02, float 2.090000e+02>\n\
  %v = fdiv <4 x float> %va, <float 2.550000e+02, float 2.550000e+02, float 2.550000e+02, float 2.550000e+02>\n";

/// `__kernel void sw_query(__write_only image2d_t img, __global int *o)`: `o[0..4]` take the
/// image's width, height, channel order and channel data type.
fn query() -> String {
    let body = "define void @__entry_sw_query(ptr addrspace(1) %0, ptr addrspace(1) %1) #0 {\n\
entry:\n\
  tail call fastcc void @sw_query(ptr addrspace(1) %0, ptr addrspace(1) %1)\n\
  ret void\n}\n\
define internal fastcc void @sw_query(ptr addrspace(1) %img, ptr addrspace(1) %o) #1 {\n\
entry:\n\
  %w = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) %img, i32 0)\n\
  store i32 %w, ptr addrspace(1) %o, align 4\n\
  %h = tail call i32 @air.get_height_texture_2d(ptr addrspace(1) %img, i32 0)\n\
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 1\n\
  store i32 %h, ptr addrspace(1) %o1, align 4\n\
  %c = tail call fastcc i32 @_Z23get_image_channel_order14ocl_image2d_wo(ptr addrspace(1) %img)\n\
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 2\n\
  store i32 %c, ptr addrspace(1) %o2, align 4\n\
  %d = tail call fastcc i32 @_Z27get_image_channel_data_type14ocl_image2d_wo(ptr addrspace(1) %img)\n\
  %o3 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 3\n\
  store i32 %d, ptr addrspace(1) %o3, align 4\n\
  ret void\n}\n\
define internal fastcc i32 @_Z23get_image_channel_order14ocl_image2d_wo(ptr addrspace(1) %0) #2 {\n\
  %2 = tail call i32 @air.get_channel_order_texture_2d(ptr addrspace(1) %0) #0\n\
  ret i32 %2\n}\n\
define internal fastcc i32 @_Z27get_image_channel_data_type14ocl_image2d_wo(ptr addrspace(1) %0) #2 {\n\
  %2 = tail call i32 @air.get_channel_data_type_texture_2d(ptr addrspace(1) %0) #0\n\
  ret i32 %2\n}\n\
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32) #0\n\
declare i32 @air.get_height_texture_2d(ptr addrspace(1), i32) #0\n\
declare i32 @air.get_channel_order_texture_2d(ptr addrspace(1)) #0\n\
declare i32 @air.get_channel_data_type_texture_2d(ptr addrspace(1)) #0\n";
    let args = [
        image(0, 0, "write", "texture2d<unknown, write>", "img"),
        buffer(1, 0, "read_write", 1, "int", "o"),
    ];
    format!("{body}{}", kernel_list("__entry_sw_query", &args))
}

/// `__kernel void sw_write(__write_only image2d_t img)`: thread `i` writes its value to its texel.
fn write() -> String {
    let body = format!(
        "define void @__entry_sw_write(ptr addrspace(1) %0) #0 {{\n\
entry:\n\
  tail call fastcc void @sw_write(ptr addrspace(1) %0)\n\
  ret void\n}}\n\
define internal fastcc void @sw_write(ptr addrspace(1) %img) #1 {{\n\
entry:\n\
{TEXEL}{VALUE}\
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %img, <2 x i32> %p, <4 x float> %v, i32 0, i32 2)\n\
  ret void\n}}\n\
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32) #0\n"
    );
    let args = [image(0, 0, "write", "texture2d<unknown, write>", "img")];
    format!("{body}{}", kernel_list("__entry_sw_write", &args))
}

/// `__kernel void sw_read(__read_only image2d_t img, sampler_t s, __global float4 *o)`: thread `i`
/// stores its texel read without a sampler to `o[2i]`, and sampled through `s` at its centre to
/// `o[2i + 1]`.
fn read() -> String {
    let body = format!(
        "define void @__entry_sw_read(ptr addrspace(1) %0, ptr addrspace(2) %1, ptr addrspace(1) %2) #0 {{\n\
entry:\n\
  tail call fastcc void @sw_read(ptr addrspace(1) %0, ptr addrspace(2) %1, ptr addrspace(1) %2)\n\
  ret void\n}}\n\
define internal fastcc void @sw_read(ptr addrspace(1) %img, ptr addrspace(2) %s, ptr addrspace(1) %o) #1 {{\n\
entry:\n\
{TEXEL}\
  %rs = tail call ptr addrspace(2) @air.get_read_sampler()\n\
  %r0 = tail call {{ <4 x float>, i8 }} @air.read_texture_2d.v4f32(ptr addrspace(1) %img, ptr addrspace(2) %rs, <2 x i32> %p, <2 x i32> zeroinitializer, i32 0, i32 1)\n\
  %t0 = extractvalue {{ <4 x float>, i8 }} %r0, 0\n\
  %b0 = mul i64 %g, 2\n\
  %q0 = getelementptr inbounds <4 x float>, ptr addrspace(1) %o, i64 %b0\n\
  store <4 x float> %t0, ptr addrspace(1) %q0, align 16\n\
  %fp = uitofp <2 x i32> %p to <2 x float>\n\
  %c = fadd <2 x float> %fp, <float 5.000000e-01, float 5.000000e-01>\n\
  %r1 = tail call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %img, ptr addrspace(2) %s, <2 x float> %c, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)\n\
  %t1 = extractvalue {{ <4 x float>, i8 }} %r1, 0\n\
  %b1 = add i64 %b0, 1\n\
  %q1 = getelementptr inbounds <4 x float>, ptr addrspace(1) %o, i64 %b1\n\
  store <4 x float> %t1, ptr addrspace(1) %q1, align 16\n\
  ret void\n}}\n\
declare ptr addrspace(2) @air.get_read_sampler() #0\n\
declare {{ <4 x float>, i8 }} @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) #0\n\
declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32) #0\n"
    );
    let args = [
        image(0, 0, "sample", "texture2d<unknown, read>", "img"),
        sampler_node(1, 0, "s"),
        float4s(2, 0, "o"),
    ];
    format!("{body}{}", kernel_list("__entry_sw_read", &args))
}

/// A view the probe measured: its name, its texture's format and channel count, and its swizzle.
struct View {
    name: &'static str,
    format: PixelFormat,
    channels: usize,
    swizzle: [TextureSwizzle; 4],
}

/// The probe's views: each OpenCL channel order's swizzle over `RGBA8Unorm`, the one-channel
/// orders' over `R8Unorm` too, and the identity.
fn views() -> Vec<View> {
    use TextureSwizzle::{Alpha as A, Blue as B, Green as G, One as I, Red as R, Zero as O};
    let rgba = |name, swizzle| View {
        name,
        format: PixelFormat::RGBA8Unorm,
        channels: 4,
        swizzle,
    };
    let r8 = |name, swizzle| View {
        name,
        format: PixelFormat::R8Unorm,
        channels: 1,
        swizzle,
    };
    vec![
        rgba("rgba8/identity", [R, G, B, A]),
        rgba("rgba8/BGRA", [B, G, R, A]),
        rgba("rgba8/ARGB", [G, B, A, R]),
        rgba("rgba8/A", [O, O, O, R]),
        rgba("rgba8/INTENSITY", [R, R, R, R]),
        rgba("rgba8/LUMINANCE", [R, R, R, I]),
        r8("r8/A", [O, O, O, R]),
        r8("r8/INTENSITY", [R, R, R, R]),
        r8("r8/LUMINANCE", [R, R, R, I]),
    ]
}

/// Byte `c` of texel `i` of a filled texture.
fn filled(i: usize, c: usize) -> u8 {
    (0x10 + 0x40 * c + i) as u8
}

/// Lane `l` of the value thread `i` writes, as a byte.
fn written(i: usize, l: usize) -> u8 {
    (0x11 + 0x40 * l + i) as u8
}

/// The channel a swizzle source names, if any.
fn channel(s: TextureSwizzle) -> Option<usize> {
    match s {
        TextureSwizzle::Red => Some(0),
        TextureSwizzle::Green => Some(1),
        TextureSwizzle::Blue => Some(2),
        TextureSwizzle::Alpha => Some(3),
        _ => None,
    }
}

/// The bytes a read of texel `i` of a filled texture through `v` answers: the read rule.
fn read_rule(v: &View, i: usize) -> [u8; 4] {
    std::array::from_fn(|l| match (v.swizzle[l], channel(v.swizzle[l])) {
        (TextureSwizzle::One, _) => 0xff,
        (_, Some(c)) if c < v.channels => filled(i, c),
        (_, Some(3)) => 0xff,
        _ => 0,
    })
}

/// A Shared texture of `v`'s format, 4 by 2, at texture index 1, every byte `fill` or the filled
/// texels, and its view through `v`'s swizzle at texture index 2.
fn swizzled(device: &Device, v: &View, fill: Option<u8>) -> (Texture, Texture) {
    let mut descriptor = TextureDescriptor::new(TextureType::Type2D, v.format, W, H);
    descriptor.usage =
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::PIXEL_FORMAT_VIEW;
    descriptor.storage_mode = StorageMode::Shared;
    let texture = (device.new_texture(&descriptor, Some(1)))
        .unwrap_or_else(|error| panic!("{}: {error}", v.name));
    let bytes: Vec<u8> = (0..N as usize)
        .flat_map(|i| (0..v.channels).map(move |c| fill.unwrap_or(filled(i, c))))
        .collect();
    let row = u64::from(W) * v.channels as u64;
    (texture.replace_region(
        whole(),
        0,
        0,
        crate::HostBytes::from(&bytes[..]),
        row,
        row * u64::from(H),
    ))
    .unwrap_or_else(|error| panic!("{}: {error}", v.name));
    let [red, green, blue, alpha] = v.swizzle;
    let swizzle = TextureSwizzleChannels {
        red,
        green,
        blue,
        alpha,
    };
    let view = (texture.new_texture_view_with_swizzle(
        v.format,
        TextureType::Type2D,
        0..1,
        0..1,
        swizzle,
        Some(2),
    ))
    .unwrap_or_else(|error| panic!("{}: the view: {error}", v.name));
    (texture, view)
}

/// The views' whole region.
fn whole() -> Region {
    Region {
        origin: [0; 3],
        size: [W, H, 1],
    }
}

/// The bytes of `texture`, of `v`'s format.
fn bytes(texture: &Texture, v: &View) -> Vec<u8> {
    let row = u64::from(W) * v.channels as u64;
    let mut out = vec![0u8; (row * u64::from(H)) as usize];
    (texture.get_bytes(&mut out, row, row * u64::from(H), whole(), 0, 0))
        .unwrap_or_else(|error| panic!("{}: {error}", v.name));
    out
}

/// The bytes of the `float4` at `at` of `got`: each lane times 255, rounded.
fn lanes(got: &[f32], at: usize) -> [u8; 4] {
    std::array::from_fn(|l| (got[4 * at + l] * 255.0).round() as u8)
}

#[test]
fn opencl_mode_queries_of_a_swizzled_write_only_image_answer_the_view() {
    let Some(query) = library(&query()) else {
        return;
    };
    each_device("opencl-swizzles-query", |name, _, device| {
        let _grant = (device.grant_indices(IndexClass::Texture, 1, 2))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let query = state(&device, &query, "__entry_sw_query");
        for v in views() {
            let (_texture, view) = swizzled(&device, &v, None);
            let o = words_buffer(&device, &[0xcdcd_cdcd; 4]);
            let bound = Bound {
                textures: &[&view],
                samplers: &[],
                out: Some(&o),
            };
            run(name, &device, &query, 1, &bound);
            assert_eq!(words(&o), [4, 2, 0x11234, 0x1234], "{name}: {}", v.name);
        }
        println!("{name}: queries of each swizzled view answer the view");
    });
}

#[test]
fn opencl_mode_writes_through_a_swizzled_image_store_the_value_as_written() {
    let Some(write) = library(&write()) else {
        return;
    };
    each_device("opencl-swizzles-write", |name, _, device| {
        let _grant = (device.grant_indices(IndexClass::Texture, 1, 2))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let write = state(&device, &write, "__entry_sw_write");
        for v in views() {
            let (texture, view) = swizzled(&device, &v, Some(0xee));
            let bound = Bound {
                textures: &[&view],
                samplers: &[],
                out: None,
            };
            run(name, &device, &write, N, &bound);
            let want: Vec<u8> = (0..N as usize)
                .flat_map(|i| (0..v.channels).map(move |c| written(i, c)))
                .collect();
            assert_eq!(bytes(&texture, &v), want, "{name}: {}", v.name);
        }
        println!("{name}: writes through each swizzled view store the value as written");
    });
}

#[test]
fn opencl_mode_reads_of_a_swizzled_image_answer_the_swizzled_texel() {
    let Some(read) = library(&read()) else {
        return;
    };
    each_device("opencl-swizzles-read", |name, _, device| {
        let _textures = (device.grant_indices(IndexClass::Texture, 1, 2))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let _samplers = (device.grant_indices(IndexClass::Sampler, 1, 1))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let read = state(&device, &read, "__entry_sw_read");
        let s = sampler(
            &device,
            SamplerMinMagFilter::Nearest,
            SamplerAddressMode::ClampToEdge,
            false,
            1,
        );
        for v in views() {
            let (_texture, view) = swizzled(&device, &v, None);
            let o = floats_buffer(&device, &vec![-1.0; (2 * N * 4) as usize]);
            let bound = Bound {
                textures: &[&view],
                samplers: &[&s],
                out: Some(&o),
            };
            run(name, &device, &read, N, &bound);
            let got = floats(&o);
            for i in 0..N as usize {
                let want = read_rule(&v, i);
                let at = |k| lanes(&got, k);
                assert_eq!(at(2 * i), want, "{name}: {} read of texel {i}", v.name);
                assert_eq!(
                    at(2 * i + 1),
                    want,
                    "{name}: {} sample of texel {i}",
                    v.name
                );
            }
        }
        println!("{name}: reads of each swizzled view answer the swizzled texel");
    });
}
