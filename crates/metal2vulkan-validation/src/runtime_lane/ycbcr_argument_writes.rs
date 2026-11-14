//! Device tests of a kernel's `write()` to a two-plane YCbCr texture through an argument buffer
//! (`bind::heap`, `memory::handles::FLAG_YCBCR_WRITES`), on the device the cfg selects, under the
//! validation layer. Metal stores the same bytes through an argument buffer's texture as through a
//! slot (an Apple M3 Max under macOS 27.0.1: `YCBCR8_444_2P`, `YCBCR10_444_2P` and the `_sRGB`
//! twins byte for byte the slot's, `RGB8_444_2P` and `RGB10_444_2P` as their slot's, and nothing
//! of `YCBCR8_420_2P`, the command buffer completed), so each format's argument-buffer write is
//! checked against its slot write over identical surfaces ([`super::ycbcr_srgb_twins`]'s layout),
//! whose codes the slot lanes check against Metal's. The writes' alpha is 0, so a write that did
//! not mark the texels it stores leaves the planes as they were.
//!
//! A command whose argument buffer names no such texture specializes the kernel as before the
//! writes were built (`SpecKey::HeapYCbCrWrites` 0): it makes no new pipeline.

use super::buffer_with;
use super::degrade_notices::on_device;
use super::read_buffer;
use super::ycbcr_srgb_twins::{
    complete, library, surface_bytes, twin_texture, COLORS, H, LENGTH, W,
};
use crate::air_library::AirLibrary;
use metal2vulkan::error::{Error as CompileError, Feature};
use metal2vulkan::metal::{PixelFormat, ResourceUsage, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    Buffer, CommandQueue, ComputePipelineState, Device, Error, Library, PipelineOptions, Size,
    Texture, TextureDescriptor,
};

/// Synthetic AIR written for this lane: kernel `write_held` writes buffer 0's `float4` `gid.x` to
/// texel `gid` of the `texture2d<float, write>` the argument buffer at buffer 1 holds.
const HELD: &str = r#"; Owned synthetic shader.
%struct.Held = type { ptr addrspace(1) }

define void @write_held(ptr addrspace(2) noalias %held, ptr addrspace(1) noalias %colors, <2 x i32> %gid) {
entry:
  %pt = getelementptr inbounds %struct.Held, ptr addrspace(2) %held, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %x = extractelement <2 x i32> %gid, i64 0
  %j = zext i32 %x to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %colors, i64 %j
  %v = load <4 x float>, ptr addrspace(1) %p, align 16
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %gid, <4 x float> %v, i32 0, i32 2)
  ret void
}

declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)

!air.kernel = !{!0}
!air.version = !{!97}
!air.language_version = !{!98}
!0 = !{ptr @write_held, !1, !2}
!1 = !{}
!2 = !{!3, !6, !7}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Held", !"air.arg_name", !"held"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, write>", !"t", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"t"}
!6 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"colors"}
!7 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!97 = !{i32 2, i32 8, i32 0}
!98 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The formats, each over a surface of its OSType, and whether Metal stores its writes.
const FORMATS: [(PixelFormat, &str, bool); 7] = [
    (PixelFormat::YCBCR84202P, "420v", false),
    (PixelFormat::YCBCR84442P, "444v", true),
    (PixelFormat::YCBCR104442P, "x444", true),
    (PixelFormat::YCBCR84442PSRGB, "444v", true),
    (PixelFormat::YCBCR104442PSRGB, "x444", true),
    (PixelFormat::RGB84442P, "444v", true),
    (PixelFormat::RGB104442P, "x444", true),
];

/// The library of [`HELD`].
fn held_library(device: &Device) -> Library {
    device
        .new_library_with_air(&llvm_assemble_text(HELD).unwrap())
        .unwrap()
}

/// The compute pipeline state of kernel `name` of `library`; `None` where the device refuses it
/// for a heap it lacks (`ArgumentBufferTextures`).
fn state(device: &Device, library: &Library, name: &str) -> Option<ComputePipelineState> {
    let function = library.new_function_with_name(name).unwrap();
    let made = device.new_compute_pipeline_state(
        &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
        &PipelineOptions::default(),
    );
    match made {
        Ok(state) => Some(state),
        Err(Error::Unsupported(u)) | Err(Error::Compile(CompileError::Unsupported(u)))
            if u.feature == Feature::ArgumentBufferTextures =>
        {
            println!("YCBCR-HELD {}: {name} refused: {u}", device.name());
            None
        }
        Err(e) => panic!("{name}: {e}"),
    }
}

/// [`COLORS`] as `float4`s, alpha 0.
fn colors(device: &Device) -> Buffer {
    let bytes: Vec<u8> = COLORS
        .iter()
        .flat_map(|&[r, g, b]| [r, g, b, 0.0])
        .flat_map(f32::to_le_bytes)
        .collect();
    buffer_with(device, &bytes).unwrap()
}

/// Run `state` on `device`'s `queue` over the whole of `texture`, with `values` at buffer 0: the texture at slot 0, or,
/// with `held`, its resource ID in a buffer at buffer 1, the texture resident for writing.
fn write(
    (device, queue): (&Device, &CommandQueue),
    state: &ComputePipelineState,
    (texture, held): (&Texture, bool),
    values: &Buffer,
) {
    let ids = buffer_with(device, &texture.gpu_resource_id().0.to_le_bytes()).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(values), 0, 0).unwrap();
    match held {
        true => {
            encoder.set_buffer(Some(&ids), 0, 1).unwrap();
            encoder
                .use_resource(texture.as_resource(), ResourceUsage::WRITE)
                .unwrap();
        }
        false => encoder.set_texture(Some(texture), 0).unwrap(),
    }
    encoder
        .dispatch_threads(Size::new(W, H, 1), Size::new(W, H, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    complete(&command_buffer);
}

/// A kernel's writes of the eight colors through an argument buffer's texture store the bytes its
/// writes through a slot store, over identical random surfaces, keeping every other byte: of each
/// 4:4:4 format whose writes Metal stores, codes in both planes; of `YCBCR8_420_2P`, whose writes
/// Metal drops, nothing.
#[test]
fn argument_buffer_writes_to_ycbcr_textures_store_the_slot_s_bytes() {
    on_device(|device, _| {
        let Some(held) = state(device, &held_library(device), "write_held") else {
            return;
        };
        let slot = state(device, &library(device), "write_columns").unwrap();
        let queue = device.new_command_queue().unwrap();
        let values = colors(device);
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        for (seed, (pf, ostype, stored)) in (0x5300..).zip(FORMATS) {
            let bytes = surface_bytes(seed);
            let after: Vec<Vec<u8>> = [(&slot, false), (&held, true)]
                .into_iter()
                .map(|(state, through)| {
                    let surface = buffer_with(device, &bytes).unwrap();
                    let texture = twin_texture(&surface, pf, ostype, usage).unwrap();
                    write((device, &queue), state, (&texture, through), &values);
                    read_buffer(&surface, 0, LENGTH).unwrap()
                })
                .collect();
            let apart = (after[0].iter().zip(&after[1])).filter(|(a, b)| a != b);
            assert_eq!(
                apart.count(),
                0,
                "{pf:?} over {ostype}: the bytes the argument buffer's writes stored differ from \
                 the slot's"
            );
            assert_eq!(
                after[0] != bytes,
                stored,
                "{pf:?} over {ostype}: the slot's writes stored codes"
            );
            println!("YCBCR-HELD {pf:?} over {ostype}: the slot's bytes, stored {stored}");
        }
    });
}

/// A command whose argument buffer holds no texture whose writes go to a write shadow makes no new
/// pipeline of the kernel: after an `RGBA8Unorm` member's, an `RGBA16Float` member's and a
/// `YCBCR8_420_2P` member's (whose writes Metal drops) dispatches make none; a `YCBCR8_444_2P`
/// member's makes its variant once, which a second such dispatch takes.
#[test]
fn argument_buffer_writes_without_a_ycbcr_member_make_no_new_pipeline() {
    on_device(|device, _| {
        let Some(held) = state(device, &held_library(device), "write_held") else {
            return;
        };
        let queue = device.new_command_queue().unwrap();
        let values = colors(device);
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let plain = |pf| {
            let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
            d.usage = usage;
            device.new_texture(&d).unwrap()
        };
        let surface = buffer_with(device, &surface_bytes(0x5310)).unwrap();
        let dropped = twin_texture(&surface, PixelFormat::YCBCR84202P, "420v", usage).unwrap();
        let other = buffer_with(device, &surface_bytes(0x5311)).unwrap();
        let shadowed = twin_texture(&other, PixelFormat::YCBCR84442P, "444v", usage).unwrap();
        let made = || device.work_counts().variant_pipelines_made;
        write(
            (device, &queue),
            &held,
            (&plain(PixelFormat::RGBA8Unorm), true),
            &values,
        );
        let first = made();
        write(
            (device, &queue),
            &held,
            (&plain(PixelFormat::RGBA16Float), true),
            &values,
        );
        write((device, &queue), &held, (&dropped, true), &values);
        assert_eq!(
            made(),
            first,
            "a command naming no shadowed texture made a pipeline"
        );
        write((device, &queue), &held, (&shadowed, true), &values);
        let shadowing = made();
        assert!(shadowing > first, "the shadowed texture's variant was made");
        write((device, &queue), &held, (&shadowed, true), &values);
        assert_eq!(
            made(),
            shadowing,
            "a second shadowed command made a pipeline"
        );
        println!(
            "YCBCR-HELD pipelines: {first} after the plain and dropped members, {shadowing} after \
             the shadowed one"
        );
    });
}
