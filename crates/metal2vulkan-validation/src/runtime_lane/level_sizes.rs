//! Size queries at a level read from a buffer, on the device the cfg selects: the width and
//! height at levels inside and past the texture's level count, as Metal answers them
//! (`lower::texture::emit::level_extent`).

use super::host_memory::run;
use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with,
    Messages, ScratchCache,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ComputePipelineState, Device, PipelineOptions, ResourceOptions, Size, Texture,
    TextureDescriptor,
};

/// The levels each query is asked at: inside a 3-level texture's count, past it, and past 255,
/// where Metal takes the level's low 8 bits.
const LEVELS: [u32; 13] = [
    0,
    1,
    2,
    3,
    5,
    7,
    31,
    32,
    255,
    256,
    257,
    0x8000_0001,
    u32::MAX,
];

/// Metal's extent at level `l` of a texture whose first level's extent is `e`:
/// `max(1, e >> (l & 255))`, a shift of 32 or more answering 1. Measured on an Apple M3 Max under
/// macOS 27.0.1 (2D, 3D, cube, array and depth textures, sampled and storage, levels 0 to 1025 and
/// up to 2^32 - 1).
fn metal_extent(e: u64, l: u32) -> u32 {
    let e = u32::try_from(e).unwrap();
    e.checked_shr(l & 255).unwrap_or(0).max(1)
}

/// The pipeline state of `entry` in the public engine fixture `fixture`.
fn state(device: &Device, fixture: &str, entry: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join(fixture)).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(entry).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap_or_else(|error| panic!("{entry} on {}: {error}", device.name()))
}

/// One thread of `state` with `textures` at textures 0 and up, `level` at buffer 0 and `bytes`
/// bytes of `out` at buffer 1: the words of `out`.
fn dispatch(
    device: &Device,
    state: &ComputePipelineState,
    textures: &[&Texture],
    level: u32,
    bytes: u64,
) -> Vec<u32> {
    let queue = device.new_command_queue().unwrap();
    let level = buffer_with(device, &level.to_le_bytes()).unwrap();
    let out = buffer_with(device, &vec![0xab; bytes as usize]).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    for (index, texture) in (0..).zip(textures) {
        encoder.set_texture(Some(texture), index).unwrap();
    }
    encoder.set_buffer(Some(&level), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    let one = Size::new(1, 1, 1);
    encoder.dispatch_threads(one, one).unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
    (read_buffer(&out, 0, bytes)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter())
    .map(|w| u32::from_le_bytes(*w))
    .collect()
}

/// `a4_query_dynamic_level` (a sampled 2D texture's width and height at the level `buffer(0)`
/// holds, and its level count) over a 37 x 5 texture of 3 levels, and `txt_level_queries` (a
/// `write` 2D texture's level count, width and height at the level, and a `write` cube's level
/// count) over a 100 x 40 texture of 3 levels and an 8 x 8 cube of 2: at every level of
/// [`LEVELS`], Metal's answer.
#[test]
fn size_queries_at_a_level_answer_metal_s_extent() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = &validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
    let texture = |descriptor: &mut TextureDescriptor, usage| {
        descriptor.mipmap_level_count = descriptor.mipmap_level_count.min(3);
        descriptor.usage = usage;
        descriptor.resource_options = ResourceOptions::private();
        device.new_texture(descriptor).unwrap()
    };
    let sampled = texture(
        &mut TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 37, 5, true),
        TextureUsage::SHADER_READ,
    );
    let written = texture(
        &mut TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 100, 40, true),
        TextureUsage::SHADER_WRITE,
    );
    let mut cube = TextureDescriptor::texture_cube(PixelFormat::RGBA16Float, 8, true);
    cube.mipmap_level_count = 2;
    let cube = texture(&mut cube, TextureUsage::SHADER_WRITE);
    let queries = state(
        device,
        "a4_query_dynamic_level.ll",
        "a4_query_dynamic_level",
    );
    let levels = state(device, "txt_level_queries.ll", "txt_level_queries");
    for l in LEVELS {
        let got = dispatch(device, &queries, &[&sampled], l, 12);
        let want = [metal_extent(37, l), metal_extent(5, l), 3];
        eprintln!("LEVELSIZES sampled 37x5/3 level {l}: {got:?}, Metal's {want:?}");
        assert_eq!(got, want, "sampled, level {l}");
        let got = dispatch(device, &levels, &[&written, &cube], l, 16);
        let want = [3, metal_extent(100, l), metal_extent(40, l), 2];
        eprintln!("LEVELSIZES storage 100x40/3 level {l}: {got:?}, Metal's {want:?}");
        assert_eq!(got, want, "storage, level {l}");
    }
    layer_reported_nothing_at_all(&messages);
}
