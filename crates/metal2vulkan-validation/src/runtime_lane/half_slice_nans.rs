//! A kernel's `half4` NaNs written to a 16-bit float texture through CoreAnimation's blur write
//! on the device the cfg selects, under the validation layer: the bits each write stores, the bits
//! a read gives back, and the notices either announces.
//!
//! What plain Metal does (the M3 Max under macOS 26.7, measured by the consumer, and the M4 under
//! macOS 27.0.1): a `half` texel written to a 16-bit float channel stores the
//! half's own bits, a signalling NaN's and a negative NaN's included.
//!
//! The kernels are this lane's own AIR, the consumer's
//! `imageblock_slice_writes_at_every_usage_are_metals` cases with NaN payloads. `slice` is the
//! blur's write (an explicit imageblock's slice written at a level from a buffer,
//! `write_imageblock_slice_to_texture_2d.v4f16`): each thread of a 16 by 16 threadgroup stores the
//! `half4` of buffer 1 at its index in the block into its imageblock cell, and the first writes
//! the whole block at its threadgroup's square of the level; `slice_l0` the same at level 0, a
//! constant; `write` writes the same `half4` at its texel of the level (`write_texture_2d.v4f16`).
//! `slice` and `write` run at both levels, `slice_l0` at level 0, of a 32 by 32 `Private` texture
//! of two levels whose every byte was [`FILL`], of each 16-bit float format, and of each usage
//! the consumer's test sweeps: the blur textures' (`ShaderRead | RenderTarget |
//! BlockWritesOnly`, `0x10005`), `ShaderRead | BlockWritesOnly` (`0x10001`), and either with
//! `ShaderWrite` (`0x10007`, `0x7`). A blit reads every level back: every channel must hold the
//! half written (a level not written its fill), but for a NaN the device stores as another NaN
//! where the dispatch announces `device-nan-encoding` (the device's write of a NaN, its native
//! answer) or the kernel's stage records `native-nan-encoding` (its read of the buffer's half is
//! the device's widening, a NaN to the device's NaN: NVIDIA's 615.71.09 on the RTX 5080 Laptop
//! GPU stores `0x7fff` for every one); the making may announce only the notes its kernels' stages
//! record (the device's own answers, [`recorded`]), and the dispatch nothing else. `read` then
//! reads level 0 back through a `texture2d<half, access::read>` and stores the half's bits: each must be the half stored, or the read announces
//! `device-nan-encoding` (a read's NaN is the device's decode). The host's buffers are
//! write-combined, which a device without unified memory places in device-local memory.
//!
//! `RGBX16Float` (the consumer's M3 run at ee1727a55: 13 texels' NaNs `0x7e00`, announced) stores
//! the written green's half in its fourth channel, Metal's X field, and reads
//! alpha 1.

use super::degrade_notices::{counts, on_device_with};
use super::{at_plan, buffer_with_options, layer_reported_nothing, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::cache::VariantKey;
use metal2vulkan::library::Library;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, Stage, StageSource};
use metal2vulkan::Budget;
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBufferStatus, ComputePipelineState, CpuCacheMode, Degradation,
    Device, HazardTrackingMode, Origin, PipelineOptions, ResourceOptions, Size, StorageMode,
    Texture, TextureDescriptor,
};
use std::collections::BTreeSet;

/// The halves written: quiet NaNs, signalling NaNs, negative NaNs of both kinds, then
/// infinities, zeros, denormals and normals.
const V: [u16; 25] = [
    0x7e00, 0x7e01, 0x7f30, 0x7fff, 0x7e55, // quiet
    0x7c01, 0x7d49, 0x7d30, 0x7dff, 0x7c80, // signalling
    0xfe00, 0xfe01, 0xffff, 0xfc01, 0xfd49, 0xfdff, // negative
    0x7c00, 0xfc00, 0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x3c00, 0x7bff, // controls
];

/// A block's side: the imageblock's, the threadgroup's, and the texture's at level 1.
const BLOCK: u64 = 16;

/// The texture's side at level 0, and its levels.
const SIDE: u64 = 32;
const LEVELS: u64 = 2;

/// The byte every texel holds before the kernels run.
const FILL: u8 = 0x11;

/// The four halves of the texel at index `i` of a block: each value at every channel position.
fn texel(i: usize) -> [u16; 4] {
    std::array::from_fn(|c| V[(i + c) % V.len()])
}

/// Synthetic AIR written for this lane (the module docs).
const KERNELS: &str = r#"; Owned synthetic kernels.
source_filename = "half_slice_nans.metal"

%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

define void @slice(ptr addrspace(1) %t, ptr addrspace(2) %level, ptr addrspace(1) %v, %"struct.metal::_imageblock_base" %img, <2 x i16> %lid, <2 x i32> %tg) {
entry:
  %x = extractelement <2 x i16> %lid, i64 0
  %y = extractelement <2 x i16> %lid, i64 1
  %x64 = zext i16 %x to i64
  %y64 = zext i16 %y to i64
  %row = mul i64 %y64, 16
  %i = add i64 %row, %x64
  %p = getelementptr <4 x i16>, ptr addrspace(1) %v, i64 %i
  %w = load <4 x i16>, ptr addrspace(1) %p, align 8
  %h = bitcast <4 x i16> %w to <4 x half>
  %cell = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %lid, i32 0, i16 0)
  store <4 x half> %h, ptr addrspace(4) %cell, align 8
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

define void @slice_l0(ptr addrspace(1) %t, ptr addrspace(2) %level, ptr addrspace(1) %v, %"struct.metal::_imageblock_base" %img, <2 x i16> %lid, <2 x i32> %tg) {
entry:
  %x = extractelement <2 x i16> %lid, i64 0
  %y = extractelement <2 x i16> %lid, i64 1
  %x64 = zext i16 %x to i64
  %y64 = zext i16 %y to i64
  %row = mul i64 %y64, 16
  %i = add i64 %row, %x64
  %p = getelementptr <4 x i16>, ptr addrspace(1) %v, i64 %i
  %w = load <4 x i16>, ptr addrspace(1) %p, align 8
  %h = bitcast <4 x i16> %w to <4 x half>
  %cell = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %lid, i32 0, i16 0)
  store <4 x half> %h, ptr addrspace(4) %cell, align 8
  tail call void @air.wg.barrier(i32 8, i32 1)
  %first = icmp eq <2 x i16> %lid, zeroinitializer
  %all = tail call i1 @air.all.v2i1(<2 x i1> %first)
  br i1 %all, label %write, label %done

write:
  %origin = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> zeroinitializer, i32 0, i16 0)
  %at = shl <2 x i32> %tg, <i32 4, i32 4>
  tail call void @air.write_imageblock_slice_to_texture_2d.v4f16(ptr addrspace(1) %t, ptr addrspace(4) %origin, i1 true, <2 x i16> zeroinitializer, <2 x i16> <i16 16, i16 16>, <2 x i32> %at, i32 0, i1 false, i32 2)
  br label %done

done:
  ret void
}

define void @write(ptr addrspace(1) %t, ptr addrspace(2) %level, ptr addrspace(1) %v, <2 x i32> %gid) {
entry:
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %bx = and i32 %x, 15
  %by = and i32 %y, 15
  %row = mul i32 %by, 16
  %i = add i32 %row, %bx
  %j = zext i32 %i to i64
  %p = getelementptr <4 x i16>, ptr addrspace(1) %v, i64 %j
  %w = load <4 x i16>, ptr addrspace(1) %p, align 8
  %h = bitcast <4 x i16> %w to <4 x half>
  %l16 = load i16, ptr addrspace(2) %level, align 2
  %l = zext i16 %l16 to i32
  tail call void @air.write_texture_2d.v4f16(ptr addrspace(1) %t, <2 x i32> %gid, <4 x half> %h, i32 %l, i32 2)
  ret void
}

define void @read(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, <2 x i32> %gid) {
entry:
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { <4 x half>, i8 } @air.read_texture_2d.v4f16(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  %h = extractvalue { <4 x half>, i8 } %r, 0
  %w = bitcast <4 x half> %h to <4 x i16>
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %row = mul i32 %y, 32
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr <4 x i16>, ptr addrspace(1) %out, i64 %j
  store <4 x i16> %w, ptr addrspace(1) %p, align 8
  ret void
}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.wg.barrier(i32, i32)
declare i1 @air.all.v2i1(<2 x i1>)
declare void @air.write_imageblock_slice_to_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i32>, i32, i1, i32)
declare void @air.write_texture_2d.v4f16(ptr addrspace(1), <2 x i32>, <4 x half>, i32, i32)
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x half>, i8 } @air.read_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)

!air.kernel = !{!0, !18, !10, !13}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @slice, !1, !2}
!1 = !{}
!2 = !{!3, !4, !9, !5, !7, !8}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 2, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"ushort", !"air.arg_name", !"level"}
!9 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ushort4", !"air.arg_name", !"v"}
!5 = !{i32 3, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 8, !"air.struct_type_info", !6, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"imageblock<Cell, layout_explicit>", !"air.arg_name", !"img"}
!6 = !{i32 0, i32 8, i32 0, !"half4", !"data"}
!7 = !{i32 4, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"lid"}
!8 = !{i32 5, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"tg"}
!18 = !{ptr @slice_l0, !1, !2}
!10 = !{ptr @write, !1, !11}
!11 = !{!3, !4, !9, !12}
!12 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!13 = !{ptr @read, !1, !14}
!14 = !{!15, !16, !17}
!15 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<half, read>", !"air.arg_name", !"t"}
!16 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ushort4", !"air.arg_name", !"out"}
!17 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The kinds making the pipeline states of `names` may announce on `device`: the precision notes
/// their stages record that no specialization removes, as the device's compile layer compiles them
/// under the empty variant key.
fn recorded(device: &Device, names: &[&str]) -> BTreeSet<Degradation> {
    let compile = device.compile_device();
    let budget = Budget::standard();
    let inputs = VariantKey::new().lowering_inputs(Stage::Kernel);
    let library = Library::from_air(&llvm_assemble_text(KERNELS).unwrap()).unwrap();
    let mut kinds = BTreeSet::new();
    for name in names {
        let function = FunctionDescriptor::new(library.function(name).unwrap());
        let compiled = compile
            .compile_function(&function, &inputs, &budget)
            .unwrap();
        let descriptor = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
        let stage = compile
            .compile_compute_pipeline(&descriptor, &budget)
            .unwrap()
            .stage;
        kinds.extend(
            stage
                .precision_notes
                .iter()
                .filter(|&&note| !stage.specialized_notes.iter().any(|s| s.note == note))
                .map(|&note| Degradation::from_precision_note(note)),
        );
    }
    kinds
}

/// Whether `h` is a NaN's half.
fn nan(h: u16) -> bool {
    h & 0x7c00 == 0x7c00 && h & 0x3ff != 0
}

/// Kernel `name` of [`KERNELS`] on `device`.
fn kernel(device: &Device, name: &str) -> ComputePipelineState {
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

/// A `Shared` buffer of `bytes` in the write-combined CPU cache mode, which a device without
/// unified memory places in device-local memory (no `shared-in-system-memory`, which is not what
/// the lane measures).
fn buffer_with(device: &Device, bytes: &[u8]) -> Buffer {
    let options = ResourceOptions::new(
        StorageMode::Shared,
        CpuCacheMode::WriteCombined,
        HazardTrackingMode::Default,
    );
    buffer_with_options(device, bytes, options).unwrap()
}

/// A level's side.
fn side(level: u64) -> u64 {
    SIDE >> level
}

/// The halves of a texture's texel, `channels` of them, as a blit reads them.
fn halves(texel: &[u8], channels: usize) -> Vec<u16> {
    texel
        .as_chunks::<2>()
        .0
        .iter()
        .take(channels)
        .map(|h| u16::from_le_bytes([h[0], h[1]]))
        .collect()
}

/// A texture of `format` (`channels` 16-bit channels) and `usage` that `state` (of kernel
/// `kind`) wrote at each level (level 0 alone for the constant level's), every byte [`FILL`]
/// before; and each level's bytes as a blit read them back.
fn written(
    device: &Device,
    (state, kind): (&ComputePipelineState, Kind),
    (format, channels): (PixelFormat, u64),
    usage: TextureUsage,
) -> (Texture, Vec<Vec<u8>>) {
    let texel_bytes = 2 * channels;
    let mut d = TextureDescriptor::texture_2d(format, SIDE, SIDE, true);
    d.mipmap_level_count = LEVELS;
    d.usage = usage;
    d.resource_options = ResourceOptions::private();
    let texture = device.new_texture(&d).unwrap();
    let bytes = |level| side(level) * side(level) * texel_bytes;
    let fill = buffer_with(device, &vec![FILL; bytes(0) as usize]);
    let outs: Vec<_> = (0..LEVELS)
        .map(|level| buffer_with(device, &vec![0; bytes(level) as usize]))
        .collect();
    let values: Vec<u8> = (0..(BLOCK * BLOCK) as usize)
        .flat_map(texel)
        .flat_map(u16::to_le_bytes)
        .collect();
    let values = buffer_with(device, &values);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for level in 0..LEVELS {
        let s = side(level);
        blit.copy_from_buffer_to_texture(
            &fill,
            0,
            s * texel_bytes,
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
    encoder.set_buffer(Some(&values), 0, 1).unwrap();
    for level in 0..kind.levels() {
        let blocks = side(level) / BLOCK;
        encoder.set_bytes(&(level as u16).to_le_bytes(), 0).unwrap();
        if kind != Kind::Write {
            encoder.set_imageblock_size(16, 16).unwrap();
        }
        encoder
            .dispatch_threadgroups(Size::new(blocks, blocks, 1), Size::new(BLOCK, BLOCK, 1))
            .unwrap();
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
            s * texel_bytes,
            bytes(level),
            BlitOption::None,
        )
        .unwrap();
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let levels: Vec<Vec<u8>> = outs
        .iter()
        .enumerate()
        .map(|(level, out)| read_buffer(out, 0, bytes(level as u64)).unwrap())
        .collect();
    // The outputs read again after a later command buffer of the queue completed: bytes that
    // changed reached the host after the wait returned; bytes that differ from the halves stored
    // and did not change were never written (seen once on the Radeon 780M under a full suite's
    // load: the outputs held their zeros).
    let later = queue.command_buffer().unwrap();
    later.commit().unwrap();
    later.wait_until_completed_for(WAIT).unwrap();
    for (level, (out, read)) in outs.iter().zip(&levels).enumerate() {
        let again = read_buffer(out, 0, bytes(level as u64)).unwrap();
        assert!(
            again == *read,
            "level {level}: its bytes reached the host after the wait returned ({} of {} \
             changed)",
            again.iter().zip(read).filter(|(a, r)| a != r).count(),
            read.len()
        );
    }
    (texture, levels)
}

/// The halves `state` (the read kernel) read from level 0 of `texture`, texel by texel, four
/// each.
fn read_back(device: &Device, state: &ComputePipelineState, texture: &Texture) -> Vec<[u16; 4]> {
    let out = buffer_with(device, &vec![0; (SIDE * SIDE * 8) as usize]);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder
        .dispatch_threadgroups(
            Size::new(SIDE / BLOCK, SIDE / BLOCK, 1),
            Size::new(BLOCK, BLOCK, 1),
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    read_buffer(&out, 0, SIDE * SIDE * 8)
        .unwrap()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|t| std::array::from_fn(|c| u16::from_le_bytes([t[2 * c], t[2 * c + 1]])))
        .collect()
}

/// A kernel of the lane: the consumer's two slice writes and the plain write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// `slice`: the slice written at the level buffer 0 names.
    Slice,
    /// `slice_l0`: the slice written at level 0, a constant.
    SliceL0,
    /// `write`: the texel written at the level buffer 0 names.
    Write,
}

impl Kind {
    const ALL: [Kind; 3] = [Kind::Slice, Kind::SliceL0, Kind::Write];

    fn name(self) -> &'static str {
        match self {
            Kind::Slice => "slice",
            Kind::SliceL0 => "slice_l0",
            Kind::Write => "write",
        }
    }

    /// The levels the kernel writes, from level 0.
    fn levels(self) -> u64 {
        match self {
            Kind::SliceL0 => 1,
            _ => LEVELS,
        }
    }
}

/// The half written at texel `(x, y)` of any level, channel `c`.
fn want(x: u64, y: u64, c: usize) -> u16 {
    texel(((y % BLOCK) * BLOCK + x % BLOCK) as usize)[c]
}

/// The half an RGBX format (`rgbx`) stores at texel `(x, y)`, channel `c`: the written green's in
/// its fourth channel, Metal's X field; any other format's the half written.
fn stored(x: u64, y: u64, c: usize, rgbx: bool) -> u16 {
    match (rgbx, c) {
        (true, 3) => want(x, y, 1),
        _ => want(x, y, c),
    }
}

/// The half Metal's `half` read gives at texel `(x, y)`, channel `c`: an RGBX format's alpha 1.
fn read_half(x: u64, y: u64, c: usize, rgbx: bool) -> u16 {
    match (rgbx, c) {
        (true, 3) => 0x3c00,
        _ => want(x, y, c),
    }
}

/// Every channel of every texel of both levels holds the half the blur's slice write and the
/// plain write stored, a NaN's sign and payload included, with no notice at the making or the
/// dispatch but the notes its stages record, on every 16-bit float format and both usages, or a
/// NaN where the dispatch or the kernel's stage announces the device's (the module docs). Level 0 read back
/// through `access::read` gives the halves stored, or the read announces `device-nan-encoding`.
#[test]
fn half_nans_written_through_the_blur_keep_their_bits() {
    // The consumer's (`imageblock_slice_writes_at_every_usage_are_metals`): the blur's, without
    // its render target, and either with `ShaderWrite`.
    let usages =
        [0x10005u32, 0x10001, 0x10007, 0x7].map(|raw| TextureUsage::try_from(raw).unwrap());
    let formats = [
        (PixelFormat::RGBA16Float, 4),
        (PixelFormat::RG16Float, 2),
        (PixelFormat::R16Float, 1),
        (PixelFormat::RGBX16Float, 4),
    ];
    let mut report = Vec::new();
    let mut failures = Vec::new();
    // Every case on one device. A device announces a kind once unless its options ask for each
    // occurrence: with `device-nan-encoding` announced at each dispatch that binds a variant
    // taking it, each case reads its own making's, dispatch's and read's notices as on a device
    // of its own, and its validation messages fail it by name.
    on_device_with(
        |options| {
            options.announce_each.insert(Degradation::DeviceNanEncoding);
            at_plan(options)
        },
        |device, messages| {
            let names = Kind::ALL.map(Kind::name);
            // Whether kernel `name`'s read of the buffer's half is the device's widening, a NaN
            // to the device's NaN, where its stage records `native-nan-encoding`.
            let widens_own =
                |name: &str| recorded(device, &[name]).contains(&Degradation::NativeNanEncoding);
            let recorded = recorded(device, &[names.as_slice(), &["read"]].concat());
            for kind in Kind::ALL {
                let name = kind.name();
                let widened_own = widens_own(name);
                for (format, channels) in formats {
                    let rgbx = format == PixelFormat::RGBX16Float;
                    for usage in usages {
                        let what = format!("{name} {format:?} usage {:#x}", usage.as_raw());
                        {
                            let state = kernel(device, name);
                            let reader = kernel(device, "read");
                            let made = counts(&messages.take_notices());
                            let (texture, levels) =
                                written(device, (&state, kind), (format, channels), usage);
                            let dispatched = counts(&messages.take_notices());
                            let mut differ = Vec::new();
                            // Whether every stored half that differs is a NaN stored as a NaN.
                            let mut nans_only = true;
                            for (level, bytes) in levels.iter().enumerate() {
                                let s = side(level as u64);
                                for (i, t) in bytes.chunks_exact(2 * channels as usize).enumerate()
                                {
                                    let (x, y) = (i as u64 % s, i as u64 / s);
                                    for (c, got) in
                                        halves(t, channels as usize).into_iter().enumerate()
                                    {
                                        // A level the kernel does not write keeps its fill.
                                        let w = match level as u64 >= kind.levels() {
                                            true => u16::from_le_bytes([FILL, FILL]),
                                            false => stored(x, y, c, rgbx),
                                        };
                                        if got != w {
                                            nans_only &= nan(w) && nan(got);
                                            differ.push(format!("{w:04x}->{got:04x}"));
                                        }
                                    }
                                }
                            }
                            differ.sort();
                            differ.dedup();
                            let read = read_back(device, &reader, &texture);
                            let read_notices = counts(&messages.take_notices());
                            let mut read_differ = Vec::new();
                            for (i, got) in read.iter().enumerate() {
                                let (x, y) = (i as u64 % SIDE, i as u64 / SIDE);
                                for (c, &g) in got.iter().enumerate().take(channels as usize) {
                                    let w = read_half(x, y, c, rgbx);
                                    if g != w {
                                        read_differ.push(format!("{w:04x}->{g:04x}"));
                                    }
                                }
                            }
                            read_differ.sort();
                            read_differ.dedup();
                            eprintln!(
                        "HALF-SLICE-NANS {what}: making {made:?}, dispatch {dispatched:?}; {} \
                         stored halves differ: {differ:?}; read back: {} differ, notices \
                         {read_notices:?}: {read_differ:?}",
                        differ.len(),
                        read_differ.len()
                    );
                            let made_unrecorded = made.keys().any(|k| !recorded.contains(k));
                            let dispatch_kinds: Vec<_> = dispatched.keys().copied().collect();
                            let dispatch_announced =
                                dispatch_kinds == [Degradation::DeviceNanEncoding];
                            let stored_unexplained = !differ.is_empty()
                                && !(nans_only && (dispatch_announced || widened_own));
                            if stored_unexplained
                                || made_unrecorded
                                || !(dispatch_kinds.is_empty() || dispatch_announced)
                            {
                                failures.push(format!(
                                    "{what}: {} stored halves differ, making announced {made:?}, \
                             dispatch {dispatched:?}",
                                    differ.len()
                                ));
                            }
                            let read_kinds: Vec<_> = read_notices.keys().copied().collect();
                            let read_announced = read_kinds == [Degradation::DeviceNanEncoding];
                            if !(read_kinds.is_empty() || read_announced)
                                || (!read_differ.is_empty() && !read_announced)
                            {
                                failures.push(format!(
                            "{what}: read back {} halves differ, announced {read_notices:?}",
                            read_differ.len()
                        ));
                            }
                            report.push(format!(
                                "{what}: stored {} differ, read {} differ{}",
                                differ.len(),
                                read_differ.len(),
                                if read_announced { " (announced)" } else { "" }
                            ));
                        }
                        layer_reported_nothing(messages, &what);
                    }
                }
            }
        },
    );
    eprintln!("half slice NaNs: {report:#?}");
    assert!(failures.is_empty(), "{failures:#?}");
}
