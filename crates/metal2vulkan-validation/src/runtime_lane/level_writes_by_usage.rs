//! A kernel's writes at a level from a buffer into a texture of every usage on the device the cfg
//! selects, under the validation layer.
//!
//! What plain Metal does (the M3 Max under macOS 26.7, plain and with the
//! validation layer; private and shared storage, the texture's own memory and a placement heap
//! alike): a write stores at every level, through every write form (`write` at level 0, at a
//! level from a buffer, at the constant level 1, and an explicit imageblock's slice write at a
//! level from a buffer), whatever the texture's usage: `Unknown`, `ShaderRead`, `RenderTarget`,
//! Apple's private `BlockWritesOnly` (`0x10000`), and their unions, CoreAnimation's blur textures'
//! `ShaderRead | RenderTarget | BlockWritesOnly` (`0x10005`) among them. Metal's validation takes
//! a write to a texture of `ShaderWrite` or `BlockWritesOnly` and stops one to any other.
//!
//! The kernels are this lane's own AIR: `slice_at_level` fills its 16 by 16 explicit imageblock
//! with [`VALUE`] and writes the whole block, as a slice, at its threadgroup's 16 by 16 square of
//! the level buffer 0 names; `write_at_level` writes [`VALUE`] at its texel of that level. Each
//! runs once a level into a 64 by 64 `RGBA16Float` `Private` texture of three levels whose every
//! byte was `0x11`: every texel of every level then holds [`VALUE`], as Metal's do. A texture of
//! `BlockWritesOnly` usage takes the storage usage in its image, so none takes a shadow; any
//! other takes one where its image lacks the usage (`undeclared-usage-shadow`). Where the binder
//! bound such a texture's level views as neutrals every write was dropped, and the texels kept
//! their `0x11` bytes.

use super::degrade_notices::{counts, on_device_with};
use super::{at_plan, buffer_with, layer_reported_nothing, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    BlitOption, CommandBufferStatus, ComputePipelineState, Degradation, Device, Origin,
    PipelineOptions, ResourceOptions, Size, TextureDescriptor,
};

/// The texture's side at level 0, and its levels (64, 32 and 16 texels a side: each a whole
/// number of 16 by 16 blocks).
const SIDE: u64 = 64;
const LEVELS: u64 = 3;

/// The texel the kernels write, `half4(0.25, 0.5, 0.75, 1.0)`, as `RGBA16Float` bytes.
const VALUE: [u8; 8] = [0x00, 0x34, 0x00, 0x38, 0x00, 0x3a, 0x00, 0x3c];

/// The byte every texel holds before the kernels run.
const FILL: u8 = 0x11;

/// Synthetic AIR written for this lane (the module docs).
const KERNELS: &str = r#"; Owned synthetic kernels.
source_filename = "level_writes_by_usage.metal"

%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

define void @slice_at_level(ptr addrspace(1) %t, ptr addrspace(2) %level, %"struct.metal::_imageblock_base" %img, <2 x i16> %lid, <2 x i32> %tg) {
entry:
  %cell = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %lid, i32 0, i16 0)
  store <4 x half> <half 0xH3400, half 0xH3800, half 0xH3A00, half 0xH3C00>, ptr addrspace(4) %cell, align 8
  tail call void @air.wg.barrier(i32 8, i32 1)
  %first = icmp eq <2 x i16> %lid, zeroinitializer
  %all = tail call i1 @air.all.v2i1(<2 x i1> %first)
  br i1 %all, label %write, label %done

write:
  %origin = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> zeroinitializer, i32 0, i16 0)
  %l16 = load i16, ptr addrspace(2) %level, align 2
  %l = zext i16 %l16 to i32
  %at = shl <2 x i32> %tg, <i32 4, i32 4>
  tail call void @air.write_imageblock_slice_to_texture_2d.v4f16(ptr addrspace(1) %t, ptr addrspace(4) %origin, i1 true, <2 x i16> zeroinitializer, <2 x i16> <i16 16, i16 16>, <2 x i32> %at, i32 %l, i1 false, i32 2)
  br label %done

done:
  ret void
}

define void @write_at_level(ptr addrspace(1) %t, ptr addrspace(2) %level, <2 x i32> %gid) {
entry:
  %l16 = load i16, ptr addrspace(2) %level, align 2
  %l = zext i16 %l16 to i32
  tail call void @air.write_texture_2d.v4f16(ptr addrspace(1) %t, <2 x i32> %gid, <4 x half> <half 0xH3400, half 0xH3800, half 0xH3A00, half 0xH3C00>, i32 %l, i32 2)
  ret void
}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.wg.barrier(i32, i32)
declare i1 @air.all.v2i1(<2 x i1>)
declare void @air.write_imageblock_slice_to_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i32>, i32, i1, i32)
declare void @air.write_texture_2d.v4f16(ptr addrspace(1), <2 x i32>, <4 x half>, i32, i32)

!air.kernel = !{!0, !10}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @slice_at_level, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !7, !8}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 2, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"ushort", !"air.arg_name", !"level"}
!5 = !{i32 2, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 8, !"air.struct_type_info", !6, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"imageblock<Cell, layout_explicit>", !"air.arg_name", !"img"}
!6 = !{i32 0, i32 8, i32 0, !"half4", !"data"}
!7 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"lid"}
!8 = !{i32 4, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"tg"}
!10 = !{ptr @write_at_level, !1, !11}
!11 = !{!3, !4, !12}
!12 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// Kernel `name` of [`KERNELS`] on `device`.
pub(super) fn kernel(device: &Device, name: &str) -> ComputePipelineState {
    let library = device
        .new_library_with_air(&llvm_assemble_text(KERNELS).unwrap())
        .unwrap();
    let function = library.new_function_with_name(name).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// A level's side.
fn side(level: u64) -> u64 {
    SIDE >> level
}

/// Each level's bytes after `state` ran once a level (a slice kernel with a 16 by 16 imageblock,
/// in 16 by 16 threadgroups; a plain one in 8 by 8 ones) into a [`SIDE`]-square `RGBA16Float`
/// `Private` texture of [`LEVELS`] levels and `usage` whose every byte was [`FILL`], all in one
/// encoder, each level read back by a blit.
fn written(
    device: &Device,
    state: &ComputePipelineState,
    slice: bool,
    usage: TextureUsage,
) -> Vec<Vec<u8>> {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, SIDE, SIDE, true);
    d.mipmap_level_count = LEVELS;
    d.usage = usage;
    d.resource_options = ResourceOptions::private();
    let texture = device.new_texture(&d).unwrap();
    let bytes = |level| side(level) * side(level) * 8;
    let fill = buffer_with(device, &vec![FILL; bytes(0) as usize]).unwrap();
    let outs: Vec<_> = (0..LEVELS)
        .map(|level| buffer_with(device, &vec![0; bytes(level) as usize]).unwrap())
        .collect();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for level in 0..LEVELS {
        let s = side(level);
        blit.copy_from_buffer_to_texture(
            &fill,
            0,
            s * 8,
            bytes(level),
            Size::new(s, s, 1),
            &texture,
            0,
            level,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
    }
    blit.end_encoding().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(&texture), 0).unwrap();
    for level in 0..LEVELS {
        let s = side(level);
        encoder.set_bytes(&(level as u16).to_le_bytes(), 0).unwrap();
        if slice {
            encoder.set_imageblock_size(16, 16).unwrap();
            encoder
                .dispatch_threadgroups(Size::new(s / 16, s / 16, 1), Size::new(16, 16, 1))
                .unwrap();
        } else {
            encoder
                .dispatch_threadgroups(Size::new(s / 8, s / 8, 1), Size::new(8, 8, 1))
                .unwrap();
        }
    }
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for (level, out) in outs.iter().enumerate() {
        let level = level as u64;
        let s = side(level);
        blit.copy_from_texture_to_buffer(
            &texture,
            0,
            level,
            Origin::new(0, 0, 0),
            Size::new(s, s, 1),
            out,
            0,
            s * 8,
            bytes(level),
            BlitOption::None,
        )
        .unwrap();
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    outs.iter()
        .enumerate()
        .map(|(level, out)| read_buffer(out, 0, bytes(level as u64)).unwrap())
        .collect()
}

/// Every texel of every level holds [`VALUE`] after either kernel ran into a texture of each
/// usage our probe measured, as Metal's do (the module docs); a texture of `BlockWritesOnly`
/// usage takes no shadow. The form each usage took is printed.
#[test]
fn writes_at_every_level_store_whatever_the_usage() {
    let blur =
        TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET | TextureUsage::BLOCK_WRITES_ONLY;
    let usages = [
        blur,
        TextureUsage::SHADER_READ | TextureUsage::BLOCK_WRITES_ONLY,
        TextureUsage::BLOCK_WRITES_ONLY,
        TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
        TextureUsage::SHADER_READ,
        TextureUsage::RENDER_TARGET,
        TextureUsage::UNKNOWN,
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
    ];
    let mut forms = Vec::new();
    // Every case on one device. A device announces a kind once unless its options ask for each
    // occurrence: with `undeclared-usage-shadow` announced for each texture that takes a shadow,
    // each case reads its own texture's notice as on a device of its own, and its validation
    // messages fail it by name.
    on_device_with(
        |options| {
            options
                .announce_each
                .insert(Degradation::UndeclaredUsageShadow);
            at_plan(options)
        },
        |device, messages| {
            for (name, slice) in [("slice_at_level", true), ("write_at_level", false)] {
                for usage in usages {
                    let what = format!("{name}, usage {:#x}", usage.as_raw());
                    {
                        let state = kernel(device, name);
                        messages.take_notices();
                        let levels = written(device, &state, slice, usage);
                        for (level, bytes) in levels.iter().enumerate() {
                            let wrong = bytes
                                .as_chunks::<8>()
                                .0
                                .iter()
                                .filter(|texel| **texel != VALUE)
                                .count();
                            assert_eq!(
                        wrong,
                        0,
                        "{what}: level {level}: {wrong} of {} texels not written (first {:02x?})",
                        bytes.len() / 8,
                        &bytes[..8]
                    );
                        }
                        let shadowed = counts(&messages.take_notices())
                            .contains_key(&Degradation::UndeclaredUsageShadow);
                        assert!(
                            !(shadowed && usage.intersects(TextureUsage::BLOCK_WRITES_ONLY)),
                            "{what}: a BlockWritesOnly texture took a shadow"
                        );
                        forms.push((
                            what.clone(),
                            if shadowed { "shadowed" } else { "own image" },
                        ));
                    }
                    layer_reported_nothing(messages, &what);
                }
            }
        },
    );
    eprintln!("level writes by usage: {forms:#?}");
}
