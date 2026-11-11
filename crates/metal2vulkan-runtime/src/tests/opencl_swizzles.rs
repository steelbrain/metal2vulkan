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
//! - **write**: a write-only image's write stores channel `c` from the first lane whose source is
//!   `c`, and lane `c` where no lane names it: the swizzle undone;
//! - **read**: a read-only image's read, without a sampler and through one (nearest, at the
//!   texel's centre), answers the texel swizzled: lane `l` the channel the swizzle's `l` names (a
//!   format's missing green and blue 0, its missing alpha 1), `Zero` 0 and `One` 1;
//! - **read-write**: Apple's OpenCL compiler takes no `__read_write` image (OpenCL C 2.0's), so
//!   the shape is plain Metal's `read_write` texture, read then written, in OpenCL mode: the read
//!   and the write rules above, as the M3 Max answers a `read_write` texture of a swizzled view.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The modules are LLVM text written here
//! (`llvm-as`).

use super::blit_holds::{device, Scratch};
use super::opencl_images::{float4s, image, run, sampler, sampler_node, Bound};
use super::opencl_mode::{buffer, floats, floats_buffer, kernel_list, state, words, words_buffer};
use crate::{Device, Region, ResourceOptions, Texture, TextureDescriptor, TextureSwizzleChannels};
use metal2vulkan::metal::{
    PixelFormat, SamplerAddressMode, SamplerMinMagFilter, TextureSwizzle, TextureType, TextureUsage,
};

/// The views' size, and the threads that use them: thread `i` takes texel `(i & 3, i >> 2)`.
const W: u64 = 4;
const H: u64 = 2;
const N: u64 = W * H;

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

/// A `read_write` image (Metal's; OpenCL C 1.2 has none) in OpenCL mode: thread `i` stores its
/// texel to `o[i]`, then writes its value there.
fn read_write() -> String {
    let body = format!(
        "define void @__entry_sw_rw(ptr addrspace(1) %0, ptr addrspace(1) %1) #0 {{\n\
entry:\n\
  tail call fastcc void @sw_rw(ptr addrspace(1) %0, ptr addrspace(1) %1)\n\
  ret void\n}}\n\
define internal fastcc void @sw_rw(ptr addrspace(1) %img, ptr addrspace(1) %o) #1 {{\n\
entry:\n\
{TEXEL}\
  %rs = tail call ptr addrspace(2) @air.get_read_sampler()\n\
  %r0 = tail call {{ <4 x float>, i8 }} @air.read_texture_2d.v4f32(ptr addrspace(1) %img, ptr addrspace(2) %rs, <2 x i32> %p, <2 x i32> zeroinitializer, i32 0, i32 3)\n\
  %t0 = extractvalue {{ <4 x float>, i8 }} %r0, 0\n\
  %q0 = getelementptr inbounds <4 x float>, ptr addrspace(1) %o, i64 %g\n\
  store <4 x float> %t0, ptr addrspace(1) %q0, align 16\n\
{VALUE}\
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %img, <2 x i32> %p, <4 x float> %v, i32 0, i32 3)\n\
  ret void\n}}\n\
declare ptr addrspace(2) @air.get_read_sampler() #0\n\
declare {{ <4 x float>, i8 }} @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) #0\n\
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32) #0\n"
    );
    let args = [
        image(0, 0, "read_write", "texture2d<unknown, read_write>", "img"),
        float4s(1, 0, "o"),
    ];
    format!("{body}{}", kernel_list("__entry_sw_rw", &args))
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

/// The bytes texel `i` holds after its write through `v`: the write rule.
fn write_rule(v: &View, i: usize) -> Vec<u8> {
    (0..v.channels)
        .map(|c| {
            let lane = (0..4)
                .find(|&l| channel(v.swizzle[l]) == Some(c))
                .unwrap_or(c);
            written(i, lane)
        })
        .collect()
}

/// A shared texture of `v`'s format, 4 by 2, every byte `fill` or the filled texels, and its view
/// through `v`'s swizzle.
fn swizzled(device: &Device, v: &View, fill: Option<u8>) -> (Texture, Texture) {
    let mut descriptor = TextureDescriptor::texture_2d(v.format, W, H, false);
    descriptor.usage =
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::PIXEL_FORMAT_VIEW;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes: Vec<u8> = (0..N as usize)
        .flat_map(|i| (0..v.channels).map(move |c| fill.unwrap_or(filled(i, c))))
        .collect();
    let row = W * v.channels as u64;
    texture
        .replace_region(Region::new_2d(0, 0, W, H), 0, 0, &bytes, row, 0)
        .unwrap();
    let [r, g, b, a] = v.swizzle;
    let view = texture
        .new_texture_view_with_swizzle(
            v.format,
            TextureType::Type2D,
            0..1,
            0..1,
            TextureSwizzleChannels::new(r, g, b, a),
        )
        .unwrap();
    (texture, view)
}

/// The bytes of `texture`, of `v`'s format.
fn bytes(texture: &Texture, v: &View) -> Vec<u8> {
    let row = W * v.channels as u64;
    let mut out = vec![0u8; (row * H) as usize];
    texture
        .get_bytes(&mut out, row, 0, Region::new_2d(0, 0, W, H), 0, 0)
        .unwrap();
    out
}

/// The bytes of the `float4` at `at` of `got`: each lane times 255, rounded.
fn lanes(got: &[f32], at: usize) -> [u8; 4] {
    std::array::from_fn(|l| (got[4 * at + l] * 255.0).round() as u8)
}

#[test]
fn opencl_mode_queries_of_a_swizzled_write_only_image_answer_the_view() {
    let scratch = Scratch::new("opencl-swizzles-query");
    let device = device(&scratch.0);
    let query = state(&device, &query(), "__entry_sw_query");
    for v in views() {
        let (_texture, view) = swizzled(&device, &v, None);
        let o = words_buffer(&device, &[0xcdcd_cdcd; 4]);
        run(
            &device,
            &query,
            1,
            &Bound {
                textures: &[&view],
                samplers: &[],
                out: Some(&o),
            },
        );
        assert_eq!(words(&o), [4, 2, 0x11234, 0x1234], "{}", v.name);
    }
}

#[test]
fn opencl_mode_writes_through_a_swizzled_image_undo_the_swizzle() {
    let scratch = Scratch::new("opencl-swizzles-write");
    let device = device(&scratch.0);
    let write = state(&device, &write(), "__entry_sw_write");
    for v in views() {
        let (texture, view) = swizzled(&device, &v, Some(0xee));
        run(
            &device,
            &write,
            N,
            &Bound {
                textures: &[&view],
                samplers: &[],
                out: None,
            },
        );
        let want: Vec<u8> = (0..N as usize).flat_map(|i| write_rule(&v, i)).collect();
        assert_eq!(bytes(&texture, &v), want, "{}", v.name);
    }
}

#[test]
fn opencl_mode_reads_of_a_swizzled_image_answer_the_swizzled_texel() {
    let scratch = Scratch::new("opencl-swizzles-read");
    let device = device(&scratch.0);
    let read = state(&device, &read(), "__entry_sw_read");
    let s = sampler(
        &device,
        SamplerMinMagFilter::Nearest,
        SamplerAddressMode::ClampToEdge,
        false,
    );
    for v in views() {
        let (_texture, view) = swizzled(&device, &v, None);
        let o = floats_buffer(&device, &vec![-1.0; (2 * N * 4) as usize]);
        run(
            &device,
            &read,
            N,
            &Bound {
                textures: &[&view],
                samplers: &[&s],
                out: Some(&o),
            },
        );
        let got = floats(&o);
        for i in 0..N as usize {
            let want = read_rule(&v, i);
            assert_eq!(lanes(&got, 2 * i), want, "{} read of texel {i}", v.name);
            assert_eq!(
                lanes(&got, 2 * i + 1),
                want,
                "{} sample of texel {i}",
                v.name
            );
        }
    }
}

#[test]
fn opencl_mode_read_write_images_of_a_swizzled_view_read_and_write_as_metal() {
    let scratch = Scratch::new("opencl-swizzles-rw");
    let device = device(&scratch.0);
    let rw = state(&device, &read_write(), "__entry_sw_rw");
    for v in views() {
        let (texture, view) = swizzled(&device, &v, None);
        let o = floats_buffer(&device, &vec![-1.0; (N * 4) as usize]);
        run(
            &device,
            &rw,
            N,
            &Bound {
                textures: &[&view],
                samplers: &[],
                out: Some(&o),
            },
        );
        let got = floats(&o);
        for i in 0..N as usize {
            assert_eq!(
                lanes(&got, i),
                read_rule(&v, i),
                "{} read of texel {i}",
                v.name
            );
        }
        let want: Vec<u8> = (0..N as usize).flat_map(|i| write_rule(&v, i)).collect();
        assert_eq!(bytes(&texture, &v), want, "{} write", v.name);
    }
}
