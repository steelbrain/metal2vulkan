//! OpenCL C's vector data functions and `native_powr` in compute pipelines of OpenCL mode
//! (`ComputePipelineDescriptor::open_cl_mode_enabled`), each kernel in the shape Apple's OpenCL
//! compiler writes (`__entry_<name>` calling the kernel; `vl3_odd` calls the intrinsics through
//! the internal wrappers the compiler writes, the others call them directly), run with the hidden
//! buffers bound as Metal's OpenCL dispatcher binds them, and compared with what an M3 Max (macOS
//! 27.0.1), a Radeon Pro 5300M and a UHD Graphics 630 (macOS 26.7.1) compute for kernels of the
//! same shapes over the same data:
//!
//! - **vl3_odd**: `vstore3(vload3(i, g + 1), i, o)` over 10 threads, `g[j] = 100 + j`: `o[j]` is
//!   `101 + j`, three `uint`s a thread from one past the buffer's start (no 16-byte alignment).
//! - **vlodd**: `vstore3(vload3(i, g + 1), i, o + 1)` of `uchar`s and `vstore2(vload2(i, s + 1), i,
//!   t + 1)` of `short`s over 4 threads, `g[j] = 100 + j`, `s[j] = 1000 + j`: bytes 1 to 12 of `o`
//!   are 101 to 112 and shorts 1 to 8 of `t` are 1001 to 1008, the rest untouched.
//! - **vha3**: `vload_half3(i, g)` and `vloada_half3(i, g)` of halves `g[j] = 0x5000 + j` over 4
//!   threads read halves `3i` to `3i + 2` and `4i` to `4i + 2`; `vstore_half3(_, i, h)` writes
//!   halves 0 to 11 and `vstorea_half3_rte(_, i, h + 32)` three of each four from half 32, on the
//!   M3 Max and the UHD Graphics 630. The Radeon Pro 5300M's `vstorea_half3` also writes the
//!   fourth half of each four (the half after the three `vloada_half3` read); what no Mac
//!   guarantees is no part of the store, and the crate leaves it.
//! - **vsh_round**: `vstore_half` and its `_rte`, `_rtz`, `_rtp` and `_rtn` forms of 32 floats:
//!   the Radeon Pro 5300M and the UHD Graphics 630 store every form as `vstore_half_rte`, word for
//!   word, and the M3 Max rounds as each names; the crate stores every form as the device converts
//!   (to nearest even, checked on the values whose `half` is normal; toward zero too where the
//!   module records `NativeHalfConversion`, as on NVIDIA's RTX 2080).
//! - **vsh_round** of NaNs: each form stores a `half` NaN, its bits the device's conversion's,
//!   and the pipeline announces `native-nan-encoding` (but where the device's narrowing is
//!   measured to move a NaN as Metal's).
//! - **npowr**: `native_powr` of `float4`s: each device answers within a few ulps of the others
//!   (`powr(100, 0.5)` is `0x1.400002p+3` on the M3 Max, `0x1.4p+3` on the Radeon Pro 5300M); the
//!   crate's answers are within `2^-12` of the M3 Max's, relatively (the device's own `powr`).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The modules are LLVM text written here
//! (`llvm-as`).

use super::blit_holds::{device, options, Scratch};
use super::opencl_mode::{
    floats, floats_buffer, kernel_list, line, run, state, whole, words, words_buffer,
};
use crate::{Buffer, Degradation, DegradeNotice, Device, MessageSink, ResourceOptions};
use metal2vulkan::error::DriverDeviation;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The node of argument `n`, a buffer at `index` of `size`-byte elements of `ty`, `access` one of
/// `read`, `read_write`.
fn data(n: u32, index: u32, access: &str, (ty, size): (&str, u32), name: &str) -> String {
    format!(
        "!{{i32 {n}, !\"air.buffer\", !\"air.location_index\", i32 {index}, i32 1, !\"air.{access}\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 {size}, !\"air.arg_type_align_size\", i32 {size}, !\"air.arg_type_name\", !\"{ty}\", !\"air.arg_name\", !\"{name}\"}}"
    )
}

/// `__kernel void vl3_odd(__global const uint *g, __global uint *o)`.
fn vl3_odd() -> String {
    let body = "define void @__entry_vl3_odd(ptr addrspace(1) %0, ptr addrspace(1) %1) #0 {\n\
entry:\n\
  tail call fastcc void @vl3_odd(ptr addrspace(1) %0, ptr addrspace(1) %1)\n\
  ret void\n}\n\
define internal fastcc void @vl3_odd(ptr addrspace(1) %g, ptr addrspace(1) %o) #1 {\n\
entry:\n\
  %i = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %g1 = getelementptr inbounds i32, ptr addrspace(1) %g, i64 1\n\
  %v = tail call fastcc <3 x i32> @_Z6vload3mPU8CLglobalKj(i64 %i, ptr addrspace(1) %g1)\n\
  tail call fastcc void @_Z7vstore3Dv3_jmPU8CLglobalj(<3 x i32> %v, i64 %i, ptr addrspace(1) %o)\n\
  ret void\n}\n\
define internal fastcc void @_Z7vstore3Dv3_jmPU8CLglobalj(<3 x i32> %0, i64 %1, ptr addrspace(1) %2) #2 {\n\
  tail call void @air.vstore.global.v3i32(<3 x i32> %0, i64 %1, ptr addrspace(1) %2) #6\n\
  ret void\n}\n\
define internal fastcc <3 x i32> @_Z6vload3mPU8CLglobalKj(i64 %0, ptr addrspace(1) %1) #2 {\n\
  %3 = tail call <3 x i32> @air.vload.global.v3i32(i64 %0, ptr addrspace(1) %1) #7\n\
  ret <3 x i32> %3\n}\n\
declare void @air.vstore.global.v3i32(<3 x i32>, i64, ptr addrspace(1)) #6\n\
declare <3 x i32> @air.vload.global.v3i32(i64, ptr addrspace(1)) #7\n\
attributes #6 = { nounwind willreturn memory(argmem: write) }\n\
attributes #7 = { nounwind willreturn memory(argmem: read) }\n";
    let args = [
        data(0, 0, "read", ("uint", 4), "g"),
        data(1, 1, "read_write", ("uint", 4), "o"),
    ];
    format!("{body}{}", kernel_list("__entry_vl3_odd", &args))
}

/// `__kernel void vlodd(__global const uchar *g, __global const short *s, __global uchar *o,
/// __global short *t)`.
fn vlodd() -> String {
    let body = "define void @__entry_vlodd(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #0 {\n\
entry:\n\
  tail call fastcc void @vlodd(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3)\n\
  ret void\n}\n\
define internal fastcc void @vlodd(ptr addrspace(1) %g, ptr addrspace(1) %s, ptr addrspace(1) %o, ptr addrspace(1) %t) #1 {\n\
entry:\n\
  %i = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %g1 = getelementptr inbounds i8, ptr addrspace(1) %g, i64 1\n\
  %o1 = getelementptr inbounds i8, ptr addrspace(1) %o, i64 1\n\
  %b = tail call <3 x i8> @air.vload.global.v3i8(i64 %i, ptr addrspace(1) %g1) #7\n\
  tail call void @air.vstore.global.v3i8(<3 x i8> %b, i64 %i, ptr addrspace(1) %o1) #6\n\
  %s1 = getelementptr inbounds i16, ptr addrspace(1) %s, i64 1\n\
  %t1 = getelementptr inbounds i16, ptr addrspace(1) %t, i64 1\n\
  %h = tail call <2 x i16> @air.vload.global.v2i16(i64 %i, ptr addrspace(1) %s1) #7\n\
  tail call void @air.vstore.global.v2i16(<2 x i16> %h, i64 %i, ptr addrspace(1) %t1) #6\n\
  ret void\n}\n\
declare <3 x i8> @air.vload.global.v3i8(i64, ptr addrspace(1)) #7\n\
declare void @air.vstore.global.v3i8(<3 x i8>, i64, ptr addrspace(1)) #6\n\
declare <2 x i16> @air.vload.global.v2i16(i64, ptr addrspace(1)) #7\n\
declare void @air.vstore.global.v2i16(<2 x i16>, i64, ptr addrspace(1)) #6\n\
attributes #6 = { nounwind willreturn memory(argmem: write) }\n\
attributes #7 = { nounwind willreturn memory(argmem: read) }\n";
    let args = [
        data(0, 0, "read", ("uchar", 1), "g"),
        data(1, 1, "read", ("short", 2), "s"),
        data(2, 2, "read_write", ("uchar", 1), "o"),
        data(3, 3, "read_write", ("short", 2), "t"),
    ];
    format!("{body}{}", kernel_list("__entry_vlodd", &args))
}

/// `__kernel void vha3(__global const half *g, __global float *o, __global half *h)`: `o[6i..]`
/// take `vload_half3(i, g)` and `vloada_half3(i, g)`, stored to `h` by `vstore_half3` and to
/// `h + 32` by `vstorea_half3_rte`.
fn vha3() -> String {
    let mut body = String::from(
        "define void @__entry_vha3(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) #0 {\n\
entry:\n\
  tail call fastcc void @vha3(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2)\n\
  ret void\n}\n\
define internal fastcc void @vha3(ptr addrspace(1) %g, ptr addrspace(1) %o, ptr addrspace(1) %h) #1 {\n\
entry:\n\
  %i = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %a = tail call <3 x float> @air.vload_half.global.v3f32(i64 %i, ptr addrspace(1) %g) #7\n\
  %b = tail call <3 x float> @air.vloada_half.global.v3f32(i64 %i, ptr addrspace(1) %g) #7\n\
  %i6 = mul i64 %i, 6\n",
    );
    for (k, (v, lane)) in [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1), ("b", 2)]
        .into_iter()
        .enumerate()
    {
        body.push_str(&format!(
            "  %x{k} = extractelement <3 x float> %{v}, i32 {lane}\n\
  %j{k} = add i64 %i6, {k}\n\
  %p{k} = getelementptr inbounds float, ptr addrspace(1) %o, i64 %j{k}\n\
  store float %x{k}, ptr addrspace(1) %p{k}, align 4\n"
        ));
    }
    body.push_str(
        "  tail call void @air.vstore_half.global.rte.v3f32(<3 x float> %a, i64 %i, ptr addrspace(1) %h) #6\n\
  %h32 = getelementptr inbounds half, ptr addrspace(1) %h, i64 32\n\
  tail call void @air.vstorea_half.global.rte.v3f32(<3 x float> %b, i64 %i, ptr addrspace(1) %h32) #6\n\
  ret void\n}\n\
declare <3 x float> @air.vload_half.global.v3f32(i64, ptr addrspace(1)) #7\n\
declare <3 x float> @air.vloada_half.global.v3f32(i64, ptr addrspace(1)) #7\n\
declare void @air.vstore_half.global.rte.v3f32(<3 x float>, i64, ptr addrspace(1)) #6\n\
declare void @air.vstorea_half.global.rte.v3f32(<3 x float>, i64, ptr addrspace(1)) #6\n\
attributes #6 = { nounwind willreturn memory(argmem: write) }\n\
attributes #7 = { nounwind willreturn memory(argmem: read) }\n",
    );
    let args = [
        data(0, 0, "read", ("half", 2), "g"),
        data(1, 1, "read_write", ("float", 4), "o"),
        data(2, 2, "read_write", ("half", 2), "h"),
    ];
    format!("{body}{}", kernel_list("__entry_vha3", &args))
}

/// `__kernel void vsh_round(__global const float *f, __global half *h)`: `f[i]` stored at half
/// `i` by `vstore_half`, and at `32 + i`, `64 + i`, `96 + i` and `128 + i` by its `_rte`,
/// `_rtz`, `_rtp` and `_rtn` forms.
fn vsh_round() -> String {
    let mut body = String::from(
        "define void @__entry_vsh_round(ptr addrspace(1) %0, ptr addrspace(1) %1) #0 {\n\
entry:\n\
  tail call fastcc void @vsh_round(ptr addrspace(1) %0, ptr addrspace(1) %1)\n\
  ret void\n}\n\
define internal fastcc void @vsh_round(ptr addrspace(1) %f, ptr addrspace(1) %h) #1 {\n\
entry:\n\
  %i = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %pf = getelementptr inbounds float, ptr addrspace(1) %f, i64 %i\n\
  %x = load float, ptr addrspace(1) %pf, align 4\n",
    );
    let mut decls = String::new();
    for (k, rounding) in ["rte", "rte", "rtz", "rtp", "rtn"].into_iter().enumerate() {
        let name = format!("@air.vstore_half.global.{rounding}.f32");
        body.push_str(&format!(
            "  %h{k} = getelementptr inbounds half, ptr addrspace(1) %h, i64 {}\n\
  tail call void {name}(float %x, i64 %i, ptr addrspace(1) %h{k}) #6\n",
            32 * k
        ));
        let decl = format!("declare void {name}(float, i64, ptr addrspace(1)) #6\n");
        if !decls.contains(&decl) {
            decls.push_str(&decl);
        }
    }
    body.push_str("  ret void\n}\n");
    body.push_str(&decls);
    body.push_str("attributes #6 = { nounwind willreturn memory(argmem: write) }\n");
    let args = [
        data(0, 0, "read", ("float", 4), "f"),
        data(1, 1, "read_write", ("half", 2), "h"),
    ];
    format!("{body}{}", kernel_list("__entry_vsh_round", &args))
}

/// `__kernel void npowr(__global const float4 *a, __global const float4 *b, __global float4 *o)`:
/// `o[i] = native_powr(a[i], b[i])`.
fn npowr() -> String {
    let body = "define void @__entry_npowr(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) #0 {\n\
entry:\n\
  tail call fastcc void @npowr(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2)\n\
  ret void\n}\n\
define internal fastcc void @npowr(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %o) #1 {\n\
entry:\n\
  %i = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %pa = getelementptr inbounds <4 x float>, ptr addrspace(1) %a, i64 %i\n\
  %va = load <4 x float>, ptr addrspace(1) %pa, align 16\n\
  %pb = getelementptr inbounds <4 x float>, ptr addrspace(1) %b, i64 %i\n\
  %vb = load <4 x float>, ptr addrspace(1) %pb, align 16\n\
  %r = tail call fastcc <4 x float> @_Z11native_powrDv4_fS_(<4 x float> %va, <4 x float> %vb)\n\
  %po = getelementptr inbounds <4 x float>, ptr addrspace(1) %o, i64 %i\n\
  store <4 x float> %r, ptr addrspace(1) %po, align 16\n\
  ret void\n}\n\
define internal fastcc <4 x float> @_Z11native_powrDv4_fS_(<4 x float> %0, <4 x float> %1) #2 {\n\
  %3 = tail call <4 x float> @air.native_powr.v4f32(<4 x float> %0, <4 x float> %1) #3\n\
  ret <4 x float> %3\n}\n\
declare <4 x float> @air.native_powr.v4f32(<4 x float>, <4 x float>) #3\n";
    let args = [
        data(0, 0, "read", ("float4", 16), "a"),
        data(1, 1, "read", ("float4", 16), "b"),
        data(2, 2, "read_write", ("float4", 16), "o"),
    ];
    format!("{body}{}", kernel_list("__entry_npowr", &args))
}

/// A shared buffer of `bytes`.
fn bytes_buffer(device: &Device, bytes: &[u8]) -> Buffer {
    device
        .new_buffer_with_bytes(bytes, ResourceOptions::shared())
        .unwrap()
}

/// A shared buffer of the 16-bit words `halves`.
fn halves_buffer(device: &Device, halves: &[u16]) -> Buffer {
    let bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
    bytes_buffer(device, &bytes)
}

/// The bytes of `buffer`, which no command uses now.
fn bytes(buffer: &Buffer) -> Vec<u8> {
    words(buffer)
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect()
}

/// The 16-bit words of `buffer`, which no command uses now.
fn halves(buffer: &Buffer) -> Vec<u16> {
    bytes(buffer)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|h| u16::from_le_bytes(*h))
        .collect()
}

/// No dispatch offset, one dimension.
const ONE: ([u32; 3], u32) = ([0; 3], 1);

#[test]
fn opencl_mode_vload3_reads_three_elements_past_an_unaligned_pointer() {
    let scratch = Scratch::new("opencl-data-vload3");
    let device = device(&scratch.0);
    let s = state(&device, &vl3_odd(), "__entry_vl3_odd");
    let g = words_buffer(&device, &(0..64).map(|j| 100 + j).collect::<Vec<_>>());
    let o = words_buffer(&device, &[0xeeee_eeee; 30]);
    run(&device, &s, line(10, 1), ONE, &[whole(&g), whole(&o)], None);
    assert_eq!(words(&o), (101..131).collect::<Vec<u32>>());
}

#[test]
fn opencl_mode_vector_data_moves_bytes_and_shorts_at_odd_elements() {
    let scratch = Scratch::new("opencl-data-narrow");
    let device = device(&scratch.0);
    let s = state(&device, &vlodd(), "__entry_vlodd");
    let g = bytes_buffer(&device, &(100..164).collect::<Vec<u8>>());
    let sh = halves_buffer(&device, &(1000..1064).collect::<Vec<u16>>());
    let o = bytes_buffer(&device, &[0xee; 16]);
    let t = halves_buffer(&device, &[0xeeee; 16]);
    let args = [whole(&g), whole(&sh), whole(&o), whole(&t)];
    run(&device, &s, line(4, 1), ONE, &args, None);
    let mut want = vec![0xee; 16];
    want[1..13].copy_from_slice(&(101..113).collect::<Vec<u8>>());
    assert_eq!(bytes(&o), want);
    let mut want = vec![0xeeee; 16];
    want[1..9].copy_from_slice(&(1001..1009).collect::<Vec<u16>>());
    assert_eq!(halves(&t), want);
}

#[test]
fn opencl_mode_half_vector_data_steps_three_or_four_halves() {
    let scratch = Scratch::new("opencl-data-half3");
    let device = device(&scratch.0);
    let s = state(&device, &vha3(), "__entry_vha3");
    let g = halves_buffer(&device, &(0..64).map(|j| 0x5000 + j).collect::<Vec<u16>>());
    let o = floats_buffer(&device, &[0.0; 24]);
    let h = halves_buffer(&device, &[0xeeee; 64]);
    run(
        &device,
        &s,
        line(4, 1),
        ONE,
        &[whole(&g), whole(&o), whole(&h)],
        None,
    );
    // Each half `0x5000 + j` is `32 + j / 32`.
    let read: Vec<f32> = floats(&o).iter().map(|x| (x - 32.0) * 32.0).collect();
    #[rustfmt::skip]
    let want: [f32; 24] = [
        0.0, 1.0, 2.0, 0.0, 1.0, 2.0,
        3.0, 4.0, 5.0, 4.0, 5.0, 6.0,
        6.0, 7.0, 8.0, 8.0, 9.0, 10.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
    ];
    assert_eq!(read, want);
    let mut want = vec![0xeeee; 64];
    for (j, w) in (0..).zip(want.iter_mut().take(12)) {
        *w = 0x5000 + j;
    }
    for i in 0..4u16 {
        for lane in 0..3 {
            want[usize::from(32 + 4 * i + lane)] = 0x5000 + 4 * i + lane;
        }
    }
    assert_eq!(halves(&h), want);
}

/// The 32 floats of `vsh_round`, with the half `vstore_half_rte` stores for each on the three
/// Macs where that half is a normal number or an infinity (`None` for the rest: a NaN, a zero, a
/// subnormal, whose encoding the device's conversion decides).
#[rustfmt::skip]
const ROUNDED: [(f32, Option<u16>); 32] = [
    // 1 + 2^-11 (a tie), 1 + 3 * 2^-12, and its negation.
    (1.0 + 1.0 / 2048.0, Some(0x3c00)), (1.0 + 3.0 / 4096.0, Some(0x3c01)),
    (-(1.0 + 3.0 / 4096.0), Some(0xbc01)),
    // Past the largest half: 65520 is the tie with the next power of two.
    (65520.0, Some(0x7c00)), (70000.0, Some(0x7c00)), (-70000.0, Some(0xfc00)),
    (1e-8, None), (-1e-8, None), (3e-5, None), (-3e-5, None), (f32::NAN, None),
    (f32::INFINITY, Some(0x7c00)), (f32::NEG_INFINITY, Some(0xfc00)), (-0.0, None),
    (0.1, Some(0x2e66)), (1.0 / 3.0, Some(0x3555)), (2049.0, Some(0x6800)), (2051.0, Some(0x6802)),
    (-2049.0, Some(0xe800)), (65504.0, Some(0x7bff)), (65519.0, Some(0x7bff)),
    // 2^-24, 1.5 * 2^-25, 2^-25, -1.5 * 2^-25, 6.1e-5: zeros and subnormals.
    (1.0 / 16_777_216.0, None), (1.5 / 33_554_432.0, None), (1.0 / 33_554_432.0, None),
    (-1.5 / 33_554_432.0, None), (6.1e-5, None),
    // 2^-14 - 2^-25 (the tie between the largest subnormal and the least normal) and
    // 2^-14 - 2^-26.
    (1.0 / 16_384.0 - 1.0 / 33_554_432.0, Some(0x0400)),
    (1.0 / 16_384.0 - 1.0 / 67_108_864.0, Some(0x0400)),
    (1e10, Some(0x7c00)), (-1e10, Some(0xfc00)), (0.5, Some(0x3800)),
    (std::f32::consts::PI, Some(0x4248)),
];

/// `x`'s `half` rounded toward zero (finite `x`): 65504 from there up, a subnormal truncated.
fn half_toward_zero(x: f32) -> u16 {
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    let magnitude = if a.is_infinite() {
        0x7c00
    } else if a >= 65504.0 {
        0x7bff
    } else if a >= 1.0 / 16_384.0 {
        let b = a.to_bits();
        u16::try_from((((b >> 23) - 112) << 10) | ((b >> 13) & 0x3ff)).unwrap()
    } else {
        // A power-of-two scaling, exact; the cast truncates.
        (a * 16_777_216.0) as u16
    };
    sign | magnitude
}

#[test]
fn opencl_mode_vstore_half_stores_every_rounding_as_the_device_converts() {
    let scratch = Scratch::new("opencl-data-vstore-half");
    let heard = Heard::default();
    let device = noting_device(&scratch.0, &heard, Degradation::NativeHalfConversion);
    let s = state(&device, &vsh_round(), "__entry_vsh_round");
    let f = floats_buffer(&device, &ROUNDED.map(|(x, _)| x));
    let h = halves_buffer(&device, &[0xeeee; 160]);
    run(&device, &s, line(32, 1), ONE, &[whole(&f), whole(&h)], None);
    let h = halves(&h);
    let (plain, forms) = h.split_at(32);
    // Every rounding the names spell stores what `vstore_half` stores, as the x86 Macs do.
    for (k, form) in forms.chunks(32).enumerate() {
        assert_eq!(form, plain, "form {k} of rte, rtz, rtp, rtn");
    }
    // That is the device's conversion: to nearest, ties to even, where the half is normal. Where
    // the module records `NativeHalfConversion`, Vulkan leaves the conversion's rounding to the
    // device (no `RoundingModeRTE`), and the note's bound lets it round toward zero (NVIDIA's RTX
    // 3090 and RTX 2080) and flush a 16-bit denormal to the zero of its sign.
    let devices_rounding = !heard.lock().unwrap().is_empty();
    for (i, (&(x, rte), &got)) in ROUNDED.iter().zip(plain).enumerate() {
        if let Some(rte) = rte {
            let rtz = half_toward_zero(x);
            let flushed = (rtz & 0x7fff) < 0x0400 && got == (rtz & 0x8000);
            assert!(
                got == rte || (devices_rounding && (got == rtz || flushed)),
                "{i}: {x:e}: {got:#06x}, to nearest {rte:#06x}, toward zero {rtz:#06x}"
            );
        }
    }
}

/// What the device's sink heard of one kind of notice.
type Heard = Arc<Mutex<Vec<DegradeNotice>>>;

/// The device the cfg selects, its shader cache in `cache`, announcing `kind` at every occurrence
/// into `heard`, so a note an earlier pipeline announced does not hide this one's.
fn noting_device(cache: &Path, heard: &Heard, kind: Degradation) -> Device {
    let heard = Arc::clone(heard);
    let mut options = options(cache);
    options.messages = Some(MessageSink::new(move |message| {
        if let Some(notice) = &message.degradation {
            if notice.degradation == kind {
                heard.lock().unwrap().push(notice.clone());
            }
        }
    }));
    options.announce_each = [kind].into_iter().collect();
    Device::system_default_with(&options).unwrap()
}

/// A NaN `vstore_half` narrows (each rounding's form) is the device's conversion's NaN, whose bits
/// are its own (NVIDIA's `0x7fff` where Metal stores `0x7e00`): the compute pipeline of OpenCL
/// mode announces `native-nan-encoding` for it, but where the device's narrowing is measured to
/// move a NaN as Metal's (`DriverDeviation::MetalNanMoves`).
#[test]
fn opencl_mode_vstore_half_of_a_nan_announces_the_devices_nan_encoding() {
    let scratch = Scratch::new("opencl-data-vstore-half-nan");
    let heard = Heard::default();
    let device = noting_device(&scratch.0, &heard, Degradation::NativeNanEncoding);
    let s = state(&device, &vsh_round(), "__entry_vsh_round");
    let nans = [0x7fc0_0000u32, 0x7fff_ffff, 0xffc0_0000, 0x7f80_0001];
    let inputs: Vec<f32> = (0..32)
        .map(|i| nans.get(i).map_or(1.0, |&b| f32::from_bits(b)))
        .collect();
    let f = floats_buffer(&device, &inputs);
    let h = halves_buffer(&device, &[0xeeee; 160]);
    run(&device, &s, line(32, 1), ONE, &[whole(&f), whole(&h)], None);
    for (k, form) in halves(&h).chunks(32).enumerate() {
        let (nan, one) = form.split_at(nans.len());
        for (i, &got) in nan.iter().enumerate() {
            assert!(
                got & 0x7c00 == 0x7c00 && got & 0x3ff != 0,
                "form {k}, NaN {i}: {got:#06x}"
            );
        }
        assert!(one.iter().all(|&g| g == 0x3c00), "form {k}: {one:x?}");
    }
    let moves = device
        .compile_device()
        .profile()
        .takes_driver_deviation(DriverDeviation::MetalNanMoves);
    let heard = heard.lock().unwrap();
    assert!(
        moves
            || heard
                .iter()
                .any(|n| n.degradation == Degradation::NativeNanEncoding),
        "{heard:?}"
    );
}

/// What a line of the pipeline creation log starts with.
const LOG_LINE: &str = "metal2vulkan: pipeline creation: compute pipeline `__entry_vsh_round`: ";

/// With `METAL2VULKAN_LOG_PIPELINES` set, the compute `VkPipeline` a dispatch of `vsh_round`
/// makes has its line in the pipeline creation log (`pipeline/creation_log.rs`), naming the
/// kernel, its module, its specialization, its precision notes (`NativeNanEncoding` among them,
/// but where the device's narrowing is measured to move a NaN as Metal's) and what it was made
/// from (the function, its bitcode's SHA-256 and its constant values); with
/// `METAL2VULKAN_DUMP_PIPELINES` naming a directory, its module is written there by the hash the
/// line names it by. A child process of this test binary ([`compute_log_child`]) dispatches it
/// with the log on.
#[test]
fn the_pipeline_creation_log_names_a_compute_pipeline_and_its_notes() {
    let scratch = Scratch::new("opencl-data-compute-log");
    let dumped = scratch.0.join("dump");
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "tests::opencl_data::compute_log_child",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .current_dir(&scratch.0)
        .env("METAL2VULKAN_LOG_PIPELINES", "1")
        .env("METAL2VULKAN_DUMP_PIPELINES", &dumped)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "the child failed: {stderr}");
    let logged = stderr
        .lines()
        .find(|l| l.starts_with(LOG_LINE))
        .unwrap_or_else(|| panic!("no compute pipeline's line: {stderr}"));
    assert!(logged.contains(" words), specialization ["), "{logged}");
    assert!(
        logged.contains("; descriptor: `__entry_vsh_round` air ")
            && logged.contains(" constants {"),
        "{logged}"
    );
    assert!(!logged.contains("SPIR-V dump into"), "{logged}");
    let hash = logged
        .split_once(", spirv ")
        .and_then(|(_, rest)| rest.split_once(' '))
        .map(|(hash, _)| hash)
        .unwrap_or_else(|| panic!("no module hash: {logged}"));
    let module = std::fs::read(dumped.join(format!("{hash}.spv"))).unwrap();
    assert_eq!(
        module.get(..4),
        Some(&0x0723_0203u32.to_le_bytes()[..]),
        "{logged}"
    );
    let moves = device(&scratch.0)
        .compile_device()
        .profile()
        .takes_driver_deviation(DriverDeviation::MetalNanMoves);
    assert!(moves || logged.contains("NativeNanEncoding"), "{logged}");
}

/// The child of [`the_pipeline_creation_log_names_a_compute_pipeline_and_its_notes`], run as its
/// own process: one dispatch of `vsh_round`.
#[test]
#[ignore = "a child process of the_pipeline_creation_log_names_a_compute_pipeline_and_its_notes"]
fn compute_log_child() {
    let scratch = Scratch::new("opencl-data-compute-log-child");
    let device = device(&scratch.0);
    let s = state(&device, &vsh_round(), "__entry_vsh_round");
    let f = floats_buffer(&device, &[1.0; 32]);
    let h = halves_buffer(&device, &[0xeeee; 160]);
    run(&device, &s, line(32, 1), ONE, &[whole(&f), whole(&h)], None);
}

#[test]
fn opencl_mode_native_powr_answers_the_devices_powr() {
    let scratch = Scratch::new("opencl-data-native-powr");
    let device = device(&scratch.0);
    let s = state(&device, &npowr(), "__entry_npowr");
    #[rustfmt::skip]
    let (a, b, m3) = (
        [2.0, 0.5, 10.0, 3.0, 1.5, 100.0, 0.25, 7.0, 1e-3, 1e3, 4.0, 9.0],
        [3.0, 2.0, -1.0, 0.5, 7.25, 0.5, -2.0, 1.5, 2.0, -1.0, 0.5, 0.5],
        [
            8.0, 0.25, f32::from_bits(0x3dcc_cccd), f32::from_bits(0x3fdd_b3d7),
            f32::from_bits(0x4197_4504), f32::from_bits(0x4120_0001), 16.0,
            f32::from_bits(0x4194_297e), f32::from_bits(0x3586_37be), f32::from_bits(0x3a83_126f),
            2.0, 3.0,
        ],
    );
    let (ab, bb) = (floats_buffer(&device, &a), floats_buffer(&device, &b));
    let o = floats_buffer(&device, &[0.0; 12]);
    run(
        &device,
        &s,
        line(3, 1),
        ONE,
        &[whole(&ab), whole(&bb), whole(&o)],
        None,
    );
    for (i, (got, want)) in floats(&o).into_iter().zip(m3).enumerate() {
        let error = ((got - want) / want).abs();
        assert!(
            error <= 2f32.powi(-12),
            "powr({}, {}): {got:e}, the M3 Max {want:e}",
            a[i],
            b[i]
        );
    }
}
