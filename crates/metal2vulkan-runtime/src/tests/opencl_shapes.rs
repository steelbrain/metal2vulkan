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
//! - **atomics**: OpenCL 1.2's `atomic_add` and `atomic_cmpxchg` on `__local` and `__global`
//!   memory, `air.atomic.{local,global}.{add.s,cmpxchg.weak}.i32` in the AIR 2.9 form, order 0,
//!   scope 1 on `__local` and 2 on `__global`, `mem_flags` 0, volatile; the exchange through
//!   its expected value's `alloca`, the old value read back from it. Over 16 work-groups of 128
//!   work-items, more than one wave on every device: each work-group's work-items add their
//!   global ids into one `__local` word and count themselves into another through a
//!   compare-exchange loop, then do the same into two `__global` words, each storing the old
//!   value its exchange succeeded on as its ticket. Each work-group's sum and count, the total
//!   sum and count, and the tickets (each work-group's `__local` ones every value below 128 once,
//!   the `__global` ones every value below 2048 once) are what no lost update can give.
//! - **sampler**: two `sampler_t` constants whose OpenCL bits differ from Metal's sampler words,
//!   which Apple's compiler writes as `air.sampler_states` globals of Metal's two words:
//!   `CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST`
//!   (`0x7bff0000088049`), read with integer coordinates (`air.convert` to `float`, then
//!   `air.sample_texture_2d`, access 0), and `CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT |
//!   CLK_FILTER_LINEAR` (`0x7bff0000080a92`), read with `float` ones; each work-item reads a 4 by 2
//!   `RGBA8Unorm` image through both. The first at `(i - 2, 3 (i & 1) - 1)`, a texel coordinate
//!   on each side of the image, answers the texel clamped to the edge; the second at
//!   `(((i & 3) + 1) / 4, ((i >> 2) + 0.5) / 2)`, half-way between texel `(i & 3, i >> 2)` and the
//!   next one across, which wraps to the row's first, answers their mean. Every byte of the image
//!   is even, so each mean is a whole byte.
//!
//! Under a device cfg, on every device of the cfg's kind. The modules are LLVM text written here;
//! `llvm-as` assembles them, and without it the tests skip visibly.

use crate::tests::command::on_device::{each_device, finished};
use crate::tests::opencl_mode::{
    buffer, floats, floats_buffer, kernel_list, library, state, words, words_buffer,
};
use crate::{
    Buffer, ComputePipelineDescriptor, ComputePipelineState, Device, DispatchType,
    FunctionDescriptor, IndexClass, Library, NumberedSampler, PixelFormat, Region, StageSource,
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

/// The atomics' work-groups and the work-items of each.
const GROUPS: u32 = 16;
const GROUP: u32 = 128;

/// `__kernel void ocl_atomics(__global int *g, __global int *per, __global int *tick, __local int
/// *l)`: `l[0]` and `l[1]` cleared behind a barrier; `atomic_add(&l[0], i)`; `atomic_cmpxchg`
/// on `l[1]` from 0 until it succeeds, its old value then `tick[2i]`; the same into `g[0]` and
/// `g[1]`, the old value `tick[2i + 1]`; behind a barrier, work-item 0 of work-group `w` stores
/// `l[0]` and `l[1]` to `per[2w]` and `per[2w + 1]`.
fn atomics() -> String {
    let body = "define void @__entry_ocl_atomics(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(3) %3) #0 {\n\
entry:\n\
  tail call fastcc void @ocl_atomics(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(3) %3)\n\
  ret void\n}\n\
define internal fastcc void @ocl_atomics(ptr addrspace(1) %g, ptr addrspace(1) %per, ptr addrspace(1) %tick, ptr addrspace(3) %l) #1 {\n\
entry:\n\
  %call = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %conv = trunc i64 %call to i32\n\
  %call1 = tail call fastcc i64 @_Z12get_local_idj(i32 0)\n\
  %conv2 = trunc i64 %call1 to i32\n\
  %call3 = tail call fastcc i64 @_Z12get_group_idj(i32 0)\n\
  %conv4 = trunc i64 %call3 to i32\n\
  %cmp = icmp eq i32 %conv2, 0\n\
  br i1 %cmp, label %if.then, label %if.end\n\
if.then:\n\
  %arrayidx6 = getelementptr inbounds i32, ptr addrspace(3) %l, i64 1\n\
  store i32 0, ptr addrspace(3) %l, align 4\n\
  store i32 0, ptr addrspace(3) %arrayidx6, align 4\n\
  br label %if.end\n\
if.end:\n\
  tail call fastcc void @_Z7barrierj(i32 1)\n\
  %call8 = tail call fastcc i32 @_Z10atomic_addPU7CLlocalVii(ptr addrspace(3) %l, i32 %conv)\n\
  %arrayidx9 = getelementptr inbounds i32, ptr addrspace(3) %l, i64 1\n\
  br label %while.cond\n\
while.cond:\n\
  %old.0 = phi i32 [ 0, %if.end ], [ %call10, %while.cond ]\n\
  %add = add nsw i32 %old.0, 1\n\
  %call10 = tail call fastcc i32 @_Z14atomic_cmpxchgPU7CLlocalViii(ptr addrspace(3) %arrayidx9, i32 %old.0, i32 %add)\n\
  %cmp11.not = icmp eq i32 %call10, %old.0\n\
  br i1 %cmp11.not, label %while.end, label %while.cond\n\
while.end:\n\
  %mul = shl nsw i32 %conv, 1\n\
  %idxprom = sext i32 %mul to i64\n\
  %arrayidx13 = getelementptr inbounds i32, ptr addrspace(1) %tick, i64 %idxprom\n\
  store i32 %old.0, ptr addrspace(1) %arrayidx13, align 4\n\
  %call15 = tail call fastcc i32 @_Z10atomic_addPU8CLglobalVii(ptr addrspace(1) %g, i32 %conv)\n\
  %arrayidx17 = getelementptr inbounds i32, ptr addrspace(1) %g, i64 1\n\
  br label %while.cond16\n\
while.cond16:\n\
  %old.1 = phi i32 [ 0, %while.end ], [ %call19, %while.cond16 ]\n\
  %add18 = add nsw i32 %old.1, 1\n\
  %call19 = tail call fastcc i32 @_Z14atomic_cmpxchgPU8CLglobalViii(ptr addrspace(1) %arrayidx17, i32 %old.1, i32 %add18)\n\
  %cmp20.not = icmp eq i32 %call19, %old.1\n\
  br i1 %cmp20.not, label %while.end23, label %while.cond16\n\
while.end23:\n\
  %add25 = or i32 %mul, 1\n\
  %idxprom26 = sext i32 %add25 to i64\n\
  %arrayidx27 = getelementptr inbounds i32, ptr addrspace(1) %tick, i64 %idxprom26\n\
  store i32 %old.1, ptr addrspace(1) %arrayidx27, align 4\n\
  tail call fastcc void @_Z7barrierj(i32 1)\n\
  br i1 %cmp, label %if.then30, label %if.end40\n\
if.then30:\n\
  %mul32 = shl nsw i32 %conv4, 1\n\
  %add37 = or i32 %mul32, 1\n\
  %idxprom38 = sext i32 %add37 to i64\n\
  %arrayidx39 = getelementptr inbounds i32, ptr addrspace(1) %per, i64 %idxprom38\n\
  %idxprom33 = sext i32 %mul32 to i64\n\
  %arrayidx34 = getelementptr inbounds i32, ptr addrspace(1) %per, i64 %idxprom33\n\
  %0 = load i32, ptr addrspace(3) %l, align 4\n\
  store i32 %0, ptr addrspace(1) %arrayidx34, align 4\n\
  %1 = load i32, ptr addrspace(3) %arrayidx9, align 4\n\
  store i32 %1, ptr addrspace(1) %arrayidx39, align 4\n\
  br label %if.end40\n\
if.end40:\n\
  ret void\n}\n\
define internal fastcc i32 @_Z14atomic_cmpxchgPU8CLglobalViii(ptr addrspace(1) %0, i32 %1, i32 %2) #2 {\n\
  %4 = alloca i32, align 4\n\
  store i32 %1, ptr %4, align 4\n\
  %5 = call i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1) %0, ptr nonnull %4, i32 %2, i32 0, i32 0, i32 2, i32 0, i1 true)\n\
  %6 = load i32, ptr %4, align 4\n\
  ret i32 %6\n}\n\
define internal fastcc i32 @_Z10atomic_addPU8CLglobalVii(ptr addrspace(1) %0, i32 %1) #2 {\n\
  %3 = call i32 @air.atomic.global.add.s.i32(ptr addrspace(1) %0, i32 %1, i32 0, i32 2, i32 0, i1 true)\n\
  ret i32 %3\n}\n\
define internal fastcc i32 @_Z14atomic_cmpxchgPU7CLlocalViii(ptr addrspace(3) %0, i32 %1, i32 %2) #2 {\n\
  %4 = alloca i32, align 4\n\
  store i32 %1, ptr %4, align 4\n\
  %5 = call i32 @air.atomic.local.cmpxchg.weak.i32(ptr addrspace(3) %0, ptr nonnull %4, i32 %2, i32 0, i32 0, i32 1, i32 0, i1 true)\n\
  %6 = load i32, ptr %4, align 4\n\
  ret i32 %6\n}\n\
define internal fastcc i32 @_Z10atomic_addPU7CLlocalVii(ptr addrspace(3) %0, i32 %1) #2 {\n\
  %3 = call i32 @air.atomic.local.add.s.i32(ptr addrspace(3) %0, i32 %1, i32 0, i32 1, i32 0, i1 true)\n\
  ret i32 %3\n}\n\
declare i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1), ptr, i32, i32, i32, i32, i32, i1) #0\n\
declare i32 @air.atomic.global.add.s.i32(ptr addrspace(1), i32, i32, i32, i32, i1) #0\n\
declare i32 @air.atomic.local.cmpxchg.weak.i32(ptr addrspace(3), ptr, i32, i32, i32, i32, i32, i1) #0\n\
declare i32 @air.atomic.local.add.s.i32(ptr addrspace(3), i32, i32, i32, i32, i1) #0\n";
    let args = [
        buffer(0, 0, "read_write", 1, "int", "g"),
        buffer(1, 1, "read_write", 1, "int", "per"),
        buffer(2, 2, "read_write", 1, "int", "tick"),
        buffer(3, 0, "read_write", 3, "int", "l"),
    ];
    format!("{body}{}", kernel_list("__entry_ocl_atomics", &args))
}

#[test]
fn opencl_mode_atomics_on_local_and_global_memory_lose_no_update() {
    let Some(atomics) = library(&atomics()) else {
        return;
    };
    let threads = GROUPS * GROUP;
    let sum = |ids: std::ops::Range<u32>| ids.fold(0u32, u32::wrapping_add);
    each_device("opencl-shapes-atomics", |name, _, device| {
        let atomics = state(&device, &atomics, "__entry_ocl_atomics");
        let g = words_buffer(&device, &[0; 2]);
        let per = words_buffer(&device, &vec![0xeeee_eeee; 2 * GROUPS as usize]);
        let tick = words_buffer(&device, &vec![0xeeee_eeee; 2 * threads as usize]);
        let bound = Bound {
            textures: &[],
            buffers: &[&g, &per, &tick],
            local: &[16],
        };
        run(name, &device, &atomics, (threads, GROUP), &bound);
        assert_eq!(
            words(&g),
            [sum(0..threads), threads],
            "{name}: the __global sum and count"
        );
        let (per, tick) = (words(&per), words(&tick));
        for w in 0..GROUPS {
            let ids = w * GROUP..(w + 1) * GROUP;
            let at = 2 * w as usize;
            assert_eq!(
                per[at..at + 2],
                [sum(ids.clone()), GROUP],
                "{name}: work-group {w}'s __local sum and count"
            );
            let mut local: Vec<u32> = ids.map(|i| tick[2 * i as usize]).collect();
            local.sort_unstable();
            assert!(
                local.iter().copied().eq(0..GROUP),
                "{name}: work-group {w}'s __local tickets {local:?}"
            );
        }
        let mut global: Vec<u32> = (0..threads as usize).map(|i| tick[2 * i + 1]).collect();
        global.sort_unstable();
        let missing: Vec<u32> = (0..threads)
            .filter(|t| global.binary_search(t).is_err())
            .collect();
        assert!(
            global.iter().copied().eq(0..threads),
            "{name}: the __global tickets miss {} values, the first {:?}",
            missing.len(),
            &missing[..missing.len().min(8)]
        );
        println!("{name}: {GROUPS} work-groups of {GROUP} lose no __local or __global update");
    });
}

/// `__kernel void ocl_sampler(__read_only image2d_t img, __global float4 *o)`, of the constant
/// samplers `s_edge` (clamp to edge, nearest, unnormalized) and `s_repeat` (repeat, linear,
/// normalized): work-item `i` stores `read_imagef(img, s_edge, (int2)(i - 2, 3 * (i & 1) - 1))`
/// to `o[2i]` and `read_imagef(img, s_repeat, (float2)(((i & 3) + 1) / 4.0, ((i >> 2) + 0.5) /
/// 2))` to `o[2i + 1]`.
fn sampler() -> String {
    let body = "@__air_sampler_state.1 = internal addrspace(2) constant [2 x i64] [i64 34901797601020562, i64 0], align 8\n\
@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601050697, i64 0], align 8\n\
define void @__entry_ocl_sampler(ptr addrspace(1) %0, ptr addrspace(1) %1) #0 {\n\
entry:\n\
  tail call fastcc void @ocl_sampler(ptr addrspace(1) %0, ptr addrspace(1) %1)\n\
  ret void\n}\n\
define internal fastcc void @ocl_sampler(ptr addrspace(1) %img, ptr addrspace(1) %o) #1 {\n\
entry:\n\
  %call = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %conv = trunc i64 %call to i32\n\
  %sub = add nsw i32 %conv, -2\n\
  %vecinit = insertelement <2 x i32> undef, i32 %sub, i64 0\n\
  %and = and i32 %conv, 1\n\
  %mul = mul nuw nsw i32 %and, 3\n\
  %sub1 = add nsw i32 %mul, -1\n\
  %vecinit2 = insertelement <2 x i32> %vecinit, i32 %sub1, i64 1\n\
  %call3 = tail call fastcc <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_i(ptr addrspace(1) %img, ptr addrspace(2) @__air_sampler_state, <2 x i32> %vecinit2)\n\
  %mul4 = shl nsw i32 %conv, 1\n\
  %idxprom = sext i32 %mul4 to i64\n\
  %arrayidx = getelementptr inbounds <4 x float>, ptr addrspace(1) %o, i64 %idxprom\n\
  store <4 x float> %call3, ptr addrspace(1) %arrayidx, align 16\n\
  %and6 = and i32 %conv, 3\n\
  %0 = tail call float @air.convert.f.f32.s.i32(i32 %and6)\n\
  %add = fadd float %0, 1.000000e+00\n\
  %div = fmul float %add, 2.500000e-01\n\
  %vecinit7 = insertelement <2 x float> undef, float %div, i64 0\n\
  %shr = ashr i32 %conv, 2\n\
  %1 = tail call float @air.convert.f.f32.s.i32(i32 %shr)\n\
  %add8 = fadd float %1, 5.000000e-01\n\
  %div9 = fmul float %add8, 5.000000e-01\n\
  %vecinit10 = insertelement <2 x float> %vecinit7, float %div9, i64 1\n\
  %call11 = tail call fastcc <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(ptr addrspace(1) %img, ptr addrspace(2) @__air_sampler_state.1, <2 x float> %vecinit10)\n\
  %add13 = or i32 %mul4, 1\n\
  %idxprom14 = sext i32 %add13 to i64\n\
  %arrayidx15 = getelementptr inbounds <4 x float>, ptr addrspace(1) %o, i64 %idxprom14\n\
  store <4 x float> %call11, ptr addrspace(1) %arrayidx15, align 16\n\
  ret void\n}\n\
define internal fastcc <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(ptr addrspace(1) %0, ptr addrspace(2) %1, <2 x float> %2) #2 {\n\
  %4 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %0, ptr addrspace(2) %1, <2 x float> %2, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)\n\
  %5 = extractvalue { <4 x float>, i8 } %4, 0\n\
  ret <4 x float> %5\n}\n\
define internal fastcc <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_i(ptr addrspace(1) %0, ptr addrspace(2) %1, <2 x i32> %2) #2 {\n\
  %4 = tail call <2 x float> @air.convert.f.v2f32.s.v2i32(<2 x i32> %2)\n\
  %5 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %0, ptr addrspace(2) %1, <2 x float> %4, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)\n\
  %6 = extractvalue { <4 x float>, i8 } %5, 0\n\
  ret <4 x float> %6\n}\n\
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32) #0\n\
declare float @air.convert.f.f32.s.i32(i32) #3\n\
declare <2 x float> @air.convert.f.v2f32.s.v2i32(<2 x i32>) #3\n\
!air.sampler_states = !{!70, !71}\n\
!70 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state.1}\n\
!71 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n";
    let args = [
        image(0, 0, "sample", "texture2d<unknown, read>", "img"),
        buffer(1, 0, "read_write", 1, "float4", "o"),
    ];
    format!("{body}{}", kernel_list("__entry_ocl_sampler", &args))
}

/// The state of kernel `entry` of `library`, a pipeline of OpenCL mode whose two constant
/// samplers take sampler indices 1 and 2.
fn sampled_state(
    name: &str,
    device: &Device,
    library: &Library,
    entry: &str,
) -> ComputePipelineState {
    let function = library.function(entry).expect("the kernel");
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(function)));
    descriptor.open_cl_mode_enabled = true;
    (device.new_compute_pipeline_state(&descriptor, |samplers| {
        assert!(
            matches!(
                samplers,
                [NumberedSampler::Constexpr(_), NumberedSampler::Constexpr(_)]
            ),
            "{name}: {samplers:?}"
        );
        Ok(vec![1, 2])
    }))
    .unwrap_or_else(|error| panic!("{name}: {entry}: {error}"))
}

/// Byte `c` of texel `(x, y)` of the sampled image: even, as every byte of it is.
fn sampled(x: u32, y: u32, c: usize) -> u8 {
    [0x10, 0x30, 0x50, 0x90][c] + (0x20 * x + 0x08 * y) as u8
}

#[test]
fn opencl_mode_constant_samplers_read_through_metals_words() {
    let Some(sampler) = library(&sampler()) else {
        return;
    };
    let texel = |x: u32, y: u32| -> [u8; 4] { std::array::from_fn(|c| sampled(x, y, c)) };
    let mean = |a: [u8; 4], b: [u8; 4]| -> [u8; 4] {
        std::array::from_fn(|c| ((u16::from(a[c]) + u16::from(b[c])) / 2) as u8)
    };
    let image: Vec<u8> = (0..N).flat_map(|i| texel(i % W, i / W)).collect();
    each_device("opencl-shapes-sampler", |name, _, device| {
        let _textures = (device.grant_indices(IndexClass::Texture, 1, 1))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let _samplers = (device.grant_indices(IndexClass::Sampler, 1, 2))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let sampler = sampled_state(name, &device, &sampler, "__entry_ocl_sampler");
        let img = texture(name, &device, PixelFormat::RGBA8Unorm, &image);
        let o = floats_buffer(&device, &vec![-1.0; (2 * N * 4) as usize]);
        let bound = Bound {
            textures: &[&img],
            buffers: &[&o],
            local: &[],
        };
        run(name, &device, &sampler, (N, N), &bound);
        let got = floats(&o);
        let bytes = |k: usize| -> [u8; 4] {
            std::array::from_fn(|c| (got[4 * k + c] * 255.0).round() as u8)
        };
        for i in 0..N {
            let edge = texel(
                (i as i32 - 2).clamp(0, W as i32 - 1) as u32,
                (3 * (i & 1) as i32 - 1).clamp(0, H as i32 - 1) as u32,
            );
            let (x, y) = (i % W, i / W);
            let repeat = mean(texel(x, y), texel((x + 1) % W, y));
            let k = 2 * i as usize;
            assert_eq!(
                bytes(k),
                edge,
                "{name}: work-item {i}'s clamped, nearest read"
            );
            assert_eq!(
                bytes(k + 1),
                repeat,
                "{name}: work-item {i}'s repeated, linear read"
            );
        }
        println!("{name}: both constant samplers read as their Metal words say");
    });
}
