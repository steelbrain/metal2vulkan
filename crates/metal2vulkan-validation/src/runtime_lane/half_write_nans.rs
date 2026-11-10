//! A `half4` written to a float storage texture on the device the cfg selects, under the validation
//! layer: the bits each format stores.
//!
//! What plain Metal does (the M3 Max under macOS 26.7, measured by the consumer, and the M4
//! under macOS 27.0.1, alike: `Shared` and `Private`, `write` and `read_write`):
//! a 16-bit float texel stores the half's own bits, a signalling NaN's included (`0x7d49` stays
//! `0x7d49`); a 32-bit float texel stores the half's exact widening, but `0x7fc00000` for every
//! NaN; an `RG11B10Float` texel stores [`METAL_RG11B10`], each NaN channel the format's
//! canonical NaN, as a `float` write of `0x7fc00000` does.
//!
//! The kernels are this lane's own AIR: `write` (a `texture2d<half, write>`), `read_write` (a
//! `texture2d<half, read_write>`) and `held` (an argument buffer's `texture2d<half, write>`,
//! a heap element) each write the `half4` buffer 0 holds at lane `gid` (its bits, [`texel`]) at
//! texel `(gid, 0)` of a [`N`] by 1 `Private` texture whose every byte was [`FILL`]. Each
//! format's stored words are printed with every lane that differs from Metal's (`in -> stored,
//! Metal's`). The write stores the half as the device's conversion widens it from the buffer's
//! bits. Where the module records `native-nan-encoding` (a device whose moves of a NaN are not
//! measured to be Metal's), that widening makes a NaN the device's NaN, which the store keeps
//! (NVIDIA's 615.71.09 on the RTX 5080 Laptop GPU: `0x7fff` for every half NaN in a 16-bit float
//! channel), so a lane may differ there as a NaN stored as a NaN. Otherwise a NaN reaches the
//! store with its payload in the significand's top bits: a 16-bit float channel stores Metal's on a
//! device with `VkProperty::MetalHalfTexelWrites`, a 32-bit one the widened `float`'s bits,
//! Metal's `0x7fc00000` on a device with `VkProperty::MetalNanMoves`, and a packed small float one
//! Metal's on a device with `VkProperty::MetalSmallFloatNans`; elsewhere the NaN is the device's,
//! and a pipeline whose stores differ announces `device-nan-encoding`. On a device with those
//! properties a texture slot's write into such a texture announces neither `device-nan-encoding`
//! nor `native-nan-encoding` (a heap element's, which may be any format, announces the first).

use super::degrade_notices::{counts, on_device_with};
use super::{at_plan, buffer_with, layer_reported_nothing, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::device::VkProperty;
use metal2vulkan::metal::{PixelFormat, ResourceUsage, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    BlitOption, CommandBufferStatus, ComputePipelineState, Degradation, Device, Origin,
    PipelineOptions, ResourceOptions, Size, TextureDescriptor,
};

/// The halves written: quiet NaNs, signalling NaNs, negative NaNs, then infinities, zeros,
/// denormals and normals.
const V: [u16; 25] = [
    0x7e00, 0x7e01, 0x7f30, 0x7fff, 0x7e55, // quiet
    0x7c01, 0x7d49, 0x7d30, 0x7dff, 0x7c80, // signalling
    0xfe00, 0xfe01, 0xffff, 0xfc01, 0xfd49, 0xfdff, // negative
    0x7c00, 0xfc00, 0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x3c00, 0x7bff, // controls
];

/// The texture's width: one texel per value.
const N: u64 = V.len() as u64;

/// The byte every texel holds before the kernel runs.
const FILL: u8 = 0xab;

/// Texel `x`'s four halves: each value at every channel position across the texels.
fn texel(x: usize) -> [u16; 4] {
    std::array::from_fn(|c| V[(x + c) % V.len()])
}

/// Metal's `RG11B10Float` words of the texels [`texel`] writes, as our probe stored them
/// (`write` and `read_write`, `Shared` and `Private`, and the `float` write of each half
/// widened by the shader, identical).
const METAL_RG11B10: [u32; 25] = [
    0xfc3f07e0, 0xfc3f07e0, 0xfc3f07e0, 0xfc3f07e0, 0xfc3f07e0, 0xfc3f07e0, 0xfc3f07e0, 0xfc3f07e0,
    0xfc3f07e0, 0xfc3f07e0, 0xfc3f07e0, 0xfc3f07e0, 0xfc3f07e0, 0xfc3f07e0, 0xf83f07e0, 0x003e07e0,
    0x000007c0, 0x00000000, 0x00000000, 0x00000000, 0x07c00000, 0x7801f800, 0xf7de003f, 0xfc3dfbc0,
    0xfc3f07bf,
];

/// Synthetic AIR written for this lane (the module docs): `access` `write` or `read_write`
/// binds the texture at slot 0, `held` in the argument buffer at buffer 2.
fn kernels(access: &str) -> String {
    if access == "held" {
        return HELD.to_owned();
    }
    format!(
        r#"; Owned synthetic kernel.
source_filename = "half_write_nans.metal"

define void @k(ptr addrspace(1) %v, ptr addrspace(1) %t, i32 %gid) {{
entry:
  %i = zext i32 %gid to i64
  %p = getelementptr <4 x i16>, ptr addrspace(1) %v, i64 %i
  %w = load <4 x i16>, ptr addrspace(1) %p, align 8
  %h = bitcast <4 x i16> %w to <4 x half>
  %at = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  tail call void @air.write_texture_2d.v4f16(ptr addrspace(1) %t, <2 x i32> %at, <4 x half> %h, i32 0, i32 2)
  ret void
}}

declare void @air.write_texture_2d.v4f16(ptr addrspace(1), <2 x i32>, <4 x half>, i32, i32)

!air.kernel = !{{!0}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ushort4", !"air.arg_name", !"v"}}
!4 = !{{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.{access}", !"air.arg_type_name", !"texture2d<half, {access}>", !"air.arg_name", !"t"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The `held` kernel: the texture an argument buffer holds.
const HELD: &str = r#"; Owned synthetic kernel.
source_filename = "half_write_nans.metal"

define void @k(ptr addrspace(1) %v, ptr addrspace(2) %held, i32 %gid) {
entry:
  %i = zext i32 %gid to i64
  %p = getelementptr <4 x i16>, ptr addrspace(1) %v, i64 %i
  %w = load <4 x i16>, ptr addrspace(1) %p, align 8
  %h = bitcast <4 x i16> %w to <4 x half>
  %at = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  %t = load ptr addrspace(1), ptr addrspace(2) %held, align 8
  tail call void @air.write_texture_2d.v4f16(ptr addrspace(1) %t, <2 x i32> %at, <4 x half> %h, i32 0, i32 2)
  ret void
}

declare void @air.write_texture_2d.v4f16(ptr addrspace(1), <2 x i32>, <4 x half>, i32, i32)

!air.kernel = !{!0}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ushort4", !"air.arg_name", !"v"}
!4 = !{i32 1, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !6, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Held", !"air.arg_name", !"held"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!6 = !{i32 0, i32 8, i32 0, !"texture2d<half, write>", !"t", !"air.indirect_argument", !7}
!7 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"t"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The kernel of `access` on `device`.
fn kernel(device: &Device, access: &str) -> ComputePipelineState {
    let library = device
        .new_library_with_air(&llvm_assemble_text(&kernels(access)).unwrap())
        .unwrap();
    let function = library.new_function_with_name("k").unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// A format the lane writes: its name, its texel's bytes, its channels and their bytes (a packed
/// format is one 4-byte word).
struct Format {
    format: PixelFormat,
    name: &'static str,
    bytes: u64,
    words: usize,
    word_bytes: usize,
}

const FORMATS: [Format; 7] = [
    Format {
        format: PixelFormat::R16Float,
        name: "r16f",
        bytes: 2,
        words: 1,
        word_bytes: 2,
    },
    Format {
        format: PixelFormat::RG16Float,
        name: "rg16f",
        bytes: 4,
        words: 2,
        word_bytes: 2,
    },
    Format {
        format: PixelFormat::RGBA16Float,
        name: "rgba16f",
        bytes: 8,
        words: 4,
        word_bytes: 2,
    },
    Format {
        format: PixelFormat::R32Float,
        name: "r32f",
        bytes: 4,
        words: 1,
        word_bytes: 4,
    },
    Format {
        format: PixelFormat::RG32Float,
        name: "rg32f",
        bytes: 8,
        words: 2,
        word_bytes: 4,
    },
    Format {
        format: PixelFormat::RGBA32Float,
        name: "rgba32f",
        bytes: 16,
        words: 4,
        word_bytes: 4,
    },
    Format {
        format: PixelFormat::RG11B10Float,
        name: "rg11b10f",
        bytes: 4,
        words: 1,
        word_bytes: 4,
    },
];

/// The exact `f32` bits of half `h`.
fn widened(h: u16) -> u32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x3ff);
    match exp {
        0x1f => sign | 0x7f80_0000 | mant << 13,
        0 => sign | (mant as f32 * 2f32.powi(-24)).to_bits(),
        e => sign | (e + 112) << 23 | mant << 13,
    }
}

/// Metal's stored word for channel `c` of texel `x` of `f` (the module docs).
fn metal(f: &Format, x: usize, c: usize) -> u32 {
    let h = texel(x)[c];
    let nan = h & 0x7c00 == 0x7c00 && h & 0x3ff != 0;
    match (f.format, f.word_bytes) {
        (PixelFormat::RG11B10Float, _) => METAL_RG11B10[x],
        (_, 2) => u32::from(h),
        _ if nan => 0x7fc0_0000,
        _ => widened(h),
    }
}

/// Whether word `w` of `f` is a NaN: a 16- or 32-bit float channel's (a packed format's word is
/// none).
fn nan_word(f: &Format, w: u32) -> bool {
    match (f.format, f.word_bytes) {
        (PixelFormat::RG11B10Float, _) => false,
        (_, 2) => w & 0x7c00 == 0x7c00 && w & 0x3ff != 0,
        _ => f32::from_bits(w).is_nan(),
    }
}

/// The words texel by texel that `state` (of `access`) stored into a `Private` texture of `f`.
fn stored(
    device: &Device,
    (state, access): (&ComputePipelineState, &str),
    f: &Format,
) -> Vec<Vec<u32>> {
    let mut d = TextureDescriptor::texture_2d(f.format, N, 1, false);
    d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    d.resource_options = ResourceOptions::private();
    let texture = device.new_texture(&d).unwrap();
    let row = N * f.bytes;
    let fill = buffer_with(device, &vec![FILL; row as usize]).unwrap();
    let out = buffer_with(device, &vec![0; row as usize]).unwrap();
    let bits: Vec<u8> = (0..V.len())
        .flat_map(texel)
        .flat_map(u16::to_le_bytes)
        .collect();
    let input = buffer_with(device, &bits).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &fill,
        0,
        row,
        row,
        Size::new(N, 1, 1),
        &texture,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    let held = buffer_with(device, &texture.gpu_resource_id().0.to_le_bytes()).unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&input), 0, 0).unwrap();
    if access == "held" {
        encoder.set_buffer(Some(&held), 0, 2).unwrap();
        encoder
            .use_resource(texture.as_resource(), ResourceUsage::WRITE)
            .unwrap();
    } else {
        encoder.set_texture(Some(&texture), 0).unwrap();
    }
    encoder
        .dispatch_threads(Size::new(N, 1, 1), Size::new(N, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(N, 1, 1),
        &out,
        0,
        row,
        row,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let bytes = read_buffer(&out, 0, row).unwrap();
    bytes
        .chunks_exact(f.bytes as usize)
        .map(|t| {
            t.chunks_exact(f.word_bytes)
                .take(f.words)
                .map(|w| {
                    let mut b = [0u8; 4];
                    b[..w.len()].copy_from_slice(w);
                    u32::from_le_bytes(b)
                })
                .collect()
        })
        .collect()
}

/// Each format's stored words through `write`, `read_write` and `held`, printed with the lanes
/// that differ from Metal's (the module docs): none in a 16-bit float channel on a device with
/// `VkProperty::MetalHalfTexelWrites` whose pipeline records no `native-nan-encoding`, nor,
/// through a texture slot, announced or differing in a 32-bit float channel on a device with
/// `VkProperty::MetalNanMoves` or a packed small float one on a device with
/// `VkProperty::MetalSmallFloatNans`; a pipeline whose stores differ announces
/// `device-nan-encoding` (at its making, or at the dispatch that binds such a texture to a slot of
/// no declared format), or `native-nan-encoding` where each lane that differs is a NaN stored as
/// a NaN.
#[test]
fn half_writes_store_their_nans_as_metal_or_announce() {
    let mut report = Vec::new();
    // Every case on one device. A device announces a kind once unless its options ask for each
    // occurrence: with `device-nan-encoding` announced at each dispatch that binds a variant
    // taking it, each case reads its own notices as on a device of its own, and its validation
    // messages fail it by name.
    on_device_with(
        |options| {
            options.announce_each.insert(Degradation::DeviceNanEncoding);
            options.announce_each.insert(Degradation::NativeNanEncoding);
            at_plan(options)
        },
        |device, messages| {
            for access in ["write", "read_write", "held"] {
                for f in &FORMATS {
                    {
                        let state = kernel(device, access);
                        let texels = stored(device, (&state, access), f);
                        let announced = counts(&messages.take_notices());
                        let properties = &device.compile_device().profile().enabled().properties;
                        let keeps = properties.contains(&VkProperty::MetalHalfTexelWrites);
                        // The NaN a texture slot's write stores, Metal's on the device's measured
                        // deviation: a 32-bit float channel's where the widening is Metal's, a
                        // packed small float one's where its store is.
                        let measured = access != "held"
                            && match f.format {
                                PixelFormat::RG11B10Float => {
                                    properties.contains(&VkProperty::MetalSmallFloatNans)
                                }
                                _ => {
                                    f.word_bytes == 4
                                        && properties.contains(&VkProperty::MetalNanMoves)
                                }
                            };
                        let mut differ = Vec::new();
                        // Whether every lane that differs is a NaN stored as a NaN.
                        let mut nans_only = true;
                        for (x, words) in texels.iter().enumerate() {
                            for (c, &got) in words.iter().enumerate() {
                                let want = metal(f, x, c);
                                if want != got {
                                    let h = texel(x)[c];
                                    nans_only &= nan_word(f, want) && nan_word(f, got);
                                    differ.push(format!("{h:04x}->{got:x} (Metal {want:x})"));
                                }
                            }
                        }
                        differ.sort();
                        differ.dedup();
                        let line = texels
                            .iter()
                            .map(|w| {
                                w.iter()
                                    .map(|v| format!("{v:x}"))
                                    .collect::<Vec<_>>()
                                    .join(",")
                            })
                            .collect::<Vec<_>>()
                            .join(" ");
                        eprintln!("half write {access} {}: {line}", f.name);
                        eprintln!(
                            "half write {access} {}: {} lanes differ from Metal's: {differ:?}",
                            f.name,
                            differ.len()
                        );
                        let noted = announced.contains_key(&Degradation::DeviceNanEncoding);
                        // The half read from its bits widens by the device's conversion, a NaN to
                        // the device's NaN, where the module records `native-nan-encoding`: the
                        // store then holds that NaN.
                        let widened_own = announced.contains_key(&Degradation::NativeNanEncoding);
                        assert!(
                            differ.is_empty() || noted || (widened_own && nans_only),
                            "{access} {}: {} lanes differ from Metal's unannounced",
                            f.name,
                            differ.len()
                        );
                        if measured {
                            assert!(
                                differ.is_empty() && !noted && !widened_own,
                                "{access} {}: {} lanes differ from Metal's, device-nan-encoding \
                                 {noted}, native-nan-encoding {widened_own}",
                                f.name,
                                differ.len()
                            );
                        }
                        // A 16-bit float channel stores the half's bits where the device keeps
                        // them and its widening is Metal's.
                        if keeps && f.word_bytes == 2 && !widened_own {
                            assert!(
                                differ.is_empty(),
                                "{access} {}: {} lanes differ from Metal's",
                                f.name,
                                differ.len()
                            );
                        }
                        report.push(format!(
                            "{access} {}: {} differ, device-nan-encoding {}",
                            f.name,
                            differ.len(),
                            if noted { "announced" } else { "not announced" }
                        ));
                    }
                    layer_reported_nothing(messages, &format!("{access} {}", f.name));
                }
            }
        },
    );
    eprintln!("half write NaNs: {report:#?}");
}
