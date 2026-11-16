//! Reads of an argument buffer's `read_write` textures on the device the cfg selects, made without
//! `shaderStorageImageReadWithoutFormat`, under the validation layer. At the runtime's Vulkan 1.2
//! instance `VK_KHR_format_feature_flags2` then enables the SPIR-V capability
//! `StorageImageReadWithoutFormat`, and a read of the heap's storage class, declared without a
//! format, goes by its view format's own `STORAGE_READ_WITHOUT_FORMAT` bit, as on Intel's ANV
//! before Gfx12.5, which lacks the feature (`DeviceProfile::storage_reads_by_format_bits`).
//!
//! Each of Metal's float read-write tier formats reads the texels uploaded to it, and each other
//! format reads them where its view has the bit and zero where it lacks it (Metal leaves that read
//! undefined: docs/API.md's accepted differences); a write through the same handle stores its
//! texel either way.

use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::{heap_classes, heap_element, HeapElement, TextureFacts};
use metal2vulkan::device::{FormatUsage, VkExtension, VkFeature};
use metal2vulkan::metal::{PixelFormat, ResourceUsage, TextureType, TextureUsage};
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, HandleClass, ImageAccess, ImageDim, ImageHandleClass, SampledType,
    StageSource,
};
use metal2vulkan::vk;
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    CommandBufferStatus, Device, PipelineOptions, Region, ResourceOptions, Size, TextureDescriptor,
};

/// Kernel `k` reads texel `(gid, 0)` of the `texture2d<float, read_write>` argument buffer 0 holds
/// into buffer 1's `float4` at `gid`, then writes (1, 0, 1, 1) there through the same handle.
const KERNEL: &str = r#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.T = type { ptr addrspace(1) }
define void @k(ptr addrspace(2) %r, ptr addrspace(1) %dst, i32 %gid) {
  %pt = getelementptr inbounds %struct.T, ptr addrspace(2) %r, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %dst, i32 %gid
  %c = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  %smp = call ptr addrspace(2) @air.get_read_sampler()
  %x = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x i32> %c, <2 x i32> zeroinitializer, i32 0, i32 3)
  %xv = extractvalue { <4 x float>, i8 } %x, 0
  store <4 x float> %xv, ptr addrspace(1) %p, align 16
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %c, <4 x float> <float 1.0, float 0.0, float 1.0, float 1.0>, i32 0, i32 3)
  ret void
}
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
!air.kernel = !{!0}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !7, !8}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"T", !"air.arg_name", !"r"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, read_write>", !"t", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"t"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"dst"}
!8 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The texels of each texture, one row.
const WIDTH: u64 = 4;

/// A format the lane reads and writes: whether it is one of Metal's read-write tier formats, the
/// bytes of the texel uploaded to every texel, what a read of it answers, and the bytes the
/// kernel's write of (1, 0, 1, 1) stores.
struct Case {
    format: PixelFormat,
    tier: bool,
    texel: Vec<u8>,
    read: [f32; 4],
    written: Vec<u8>,
}

/// `words`' little-endian bytes.
fn le<const N: usize, T: Copy>(words: [T; N], bytes: fn(T) -> Vec<u8>) -> Vec<u8> {
    words.into_iter().flat_map(bytes).collect()
}

fn f32s<const N: usize>(v: [f32; N]) -> Vec<u8> {
    le(v, |x| x.to_le_bytes().to_vec())
}

fn u16s<const N: usize>(v: [u16; N]) -> Vec<u8> {
    le(v, |x| x.to_le_bytes().to_vec())
}

fn u32s<const N: usize>(v: [u32; N]) -> Vec<u8> {
    le(v, |x| x.to_le_bytes().to_vec())
}

/// The float read-write tier formats and the non-tier formats the Iris Xe cannot read without a
/// format (`BGRA8Unorm`, `RGB10A2Unorm`, `RG11B10Float`), each texel exact in its format. A
/// one-channel format reads (r, 0, 0, 1) and `RG11B10Float` alpha 1. Half bits: 0x3800 is 0.5,
/// 0x3C00 1, 0x4000 2, 0xC000 -2. `RG11B10Float`'s red and green: 5-bit exponent (bias 15), 6-bit
/// mantissa; blue's: 5-bit exponent, 5-bit mantissa.
fn cases() -> Vec<Case> {
    use PixelFormat as P;
    let ten = |r: u32, g: u32, b: u32, a: u32| r | g << 10 | b << 20 | a << 30;
    let eleven = |r: u32, g: u32, b: u32| r | g << 11 | b << 22;
    vec![
        Case {
            format: P::R32Float,
            tier: true,
            texel: f32s([0.75]),
            read: [0.75, 0.0, 0.0, 1.0],
            written: f32s([1.0]),
        },
        Case {
            format: P::RGBA32Float,
            tier: true,
            texel: f32s([0.25, -2.0, 3.5, 0.5]),
            read: [0.25, -2.0, 3.5, 0.5],
            written: f32s([1.0, 0.0, 1.0, 1.0]),
        },
        Case {
            format: P::RGBA16Float,
            tier: true,
            texel: u16s([0x3800, 0xC000, 0x4000, 0x3C00]),
            read: [0.5, -2.0, 2.0, 1.0],
            written: u16s([0x3C00, 0, 0x3C00, 0x3C00]),
        },
        Case {
            format: P::R16Float,
            tier: true,
            texel: u16s([0x3800]),
            read: [0.5, 0.0, 0.0, 1.0],
            written: u16s([0x3C00]),
        },
        Case {
            format: P::RGBA8Unorm,
            tier: true,
            texel: vec![0, 255, 0, 255],
            read: [0.0, 1.0, 0.0, 1.0],
            written: vec![255, 0, 255, 255],
        },
        Case {
            format: P::R8Unorm,
            tier: true,
            texel: vec![255],
            read: [1.0, 0.0, 0.0, 1.0],
            written: vec![255],
        },
        Case {
            format: P::BGRA8Unorm,
            tier: false,
            texel: vec![255, 0, 0, 255],
            read: [0.0, 0.0, 1.0, 1.0],
            written: vec![255, 0, 255, 255],
        },
        Case {
            format: P::RGB10A2Unorm,
            tier: false,
            texel: u32s([ten(0, 1023, 0, 0)]),
            read: [0.0, 1.0, 0.0, 0.0],
            written: u32s([ten(1023, 0, 1023, 3)]),
        },
        Case {
            format: P::RG11B10Float,
            tier: false,
            texel: u32s([eleven(16 << 6, 14 << 6, 15 << 5)]),
            read: [2.0, 0.5, 1.0, 1.0],
            written: u32s([eleven(15 << 6, 0, 15 << 5)]),
        },
    ]
}

/// The heap's float 2D storage class on `device`, which holds each case's texture.
fn storage_class(device: &Device) -> HandleClass {
    let profile = device.compile_device().profile();
    let float_2d = |c: &ImageHandleClass| {
        c.dim == ImageDim::D2
            && !c.arrayed
            && !c.multisampled
            && !c.depth_compare
            && c.sampled_type == SampledType::Float
            && c.access == ImageAccess::Storage
    };
    heap_classes(profile)
        .into_iter()
        .find(|c| matches!(c, HandleClass::Image(c) if float_2d(c)))
        .expect("the heap's float 2D storage class")
}

/// Whether `device` reads the storage view the heap holds a `read_write` texture of `format`
/// through without a format: its format's `STORAGE_READ_WITHOUT_FORMAT` bit.
fn view_readable(device: &Device, format: PixelFormat) -> bool {
    let profile = device.compile_device().profile();
    let mut facts = TextureFacts::new(TextureType::Type2D, format);
    facts.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let class = storage_class(device);
    let element = heap_element(profile, &class, &facts, vk::ImageLayout::GENERAL);
    let Ok(HeapElement::View(view)) = element else {
        panic!("{format:?}: the storage class holds no view: {element:?}");
    };
    profile.supports_format(view.format, FormatUsage::StorageImageReadWithoutFormat)
}

/// On the device made without `shaderStorageImageReadWithoutFormat`, a kernel reading and writing
/// an argument buffer's `read_write` texture compiles, by its formats' own bits. Each float tier
/// format's texture reads its uploaded texels and stores the kernel's write; each other format's
/// reads them where its view has the bit (lavapipe: every one), zero where it lacks it (the Iris
/// Xe: all three), and stores the write either way. Printed (`HEAP-STORAGE-READ`).
#[test]
fn heap_storage_reads_go_by_the_formats_bits_without_the_feature() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let feature = VkFeature::ShaderStorageImageReadWithoutFormat;
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.withheld_features.insert(feature);
    })
    .unwrap();
    let name = device.name().to_owned();
    let enabled = device.compile_device().profile().enabled();
    assert!(!enabled.features.contains(&feature), "{name}: {feature:?}");
    if !enabled
        .extensions
        .contains(&VkExtension::KhrFormatFeatureFlags2)
    {
        println!("HEAP-STORAGE-READ {name}: no VK_KHR_format_feature_flags2, not checked");
        return;
    }
    let library = device
        .new_library_with_air(&llvm_assemble_text(KERNEL).unwrap())
        .unwrap();
    let function = library.new_function_with_name("k").unwrap();
    let state = device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap_or_else(|e| panic!("{name}: the kernel: {e}"));
    let queue = device.new_command_queue().unwrap();
    for case in cases() {
        let pf = case.format;
        let readable = view_readable(&device, pf);
        assert!(
            readable || !case.tier,
            "{name}: tier format {pf:?} unreadable"
        );
        let mut d = TextureDescriptor::texture_2d(pf, WIDTH, 1, false);
        d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        d.resource_options = ResourceOptions::shared();
        let texture = device
            .new_texture(&d)
            .unwrap_or_else(|e| panic!("{name}: a {pf:?} texture: {e}"));
        let size = case.texel.len() as u64;
        let row = case.texel.repeat(WIDTH as usize);
        let region = Region::new_2d(0, 0, WIDTH, 1);
        texture
            .replace_region(region, 0, 0, &row, size * WIDTH, 0)
            .unwrap();
        let ids = texture.gpu_resource_id().0.to_le_bytes();
        let arguments = buffer_with(&device, &ids).unwrap();
        let out = buffer_with(&device, &vec![0xee; 16 * WIDTH as usize]).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&state).unwrap();
        encoder.set_buffer(Some(&arguments), 0, 0).unwrap();
        encoder.set_buffer(Some(&out), 0, 1).unwrap();
        let usage = ResourceUsage::READ | ResourceUsage::WRITE;
        encoder.use_resource(texture.as_resource(), usage).unwrap();
        encoder
            .dispatch_threads(Size::new(WIDTH, 1, 1), Size::new(WIDTH, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let bytes = read_buffer(&out, 0, 16 * WIDTH).unwrap();
        let got: Vec<[f32; 4]> = bytes
            .chunks(16)
            .map(|t| {
                std::array::from_fn(|i| f32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap()))
            })
            .collect();
        let want = match readable {
            true => case.read,
            false => [0.0; 4],
        };
        assert!(
            got.iter().all(|&v| v == want),
            "{name}: {pf:?} (view readable {readable}) read {got:?}, not {want:?}"
        );
        let mut stored = vec![0u8; (size * WIDTH) as usize];
        texture
            .get_bytes(&mut stored, size * WIDTH, 0, region, 0, 0)
            .unwrap();
        assert_eq!(
            stored,
            case.written.repeat(WIDTH as usize),
            "{name}: {pf:?}: the write"
        );
        println!(
            "HEAP-STORAGE-READ {name}: {pf:?} (tier {}, view readable {readable}): read {want:?}, write stored",
            case.tier
        );
    }
    drop(device);
    layer_reported_nothing_at_all(&messages);
}
