//! Multisample colour textures in a heap over the device's memory (`texture/heap.rs`,
//! `Device::device_heap_layout`): where their samples lie in the heap's bytes beside Metal's
//! tiles, and the notices of those whose bytes are not Metal's.

use super::heap_textures::{sample_round_trip, SampleAspect};
use super::host_memory::{gpu_fill, gpu_read, run};
use super::{buffer_with, layer_reported_nothing_at_all, validated_device, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::DriverFact;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, CpuCacheMode, Degradation, Error, HazardTrackingMode, HeapDescriptor, HeapType,
    ResourceOptions, StorageMode, TextureDescriptor,
};

/// A multisample colour texture of `ShaderRead | RenderTarget` usage with
/// `allowGPUOptimizedContents` off, which Metal keeps uncompressed, in a `Private` placement heap,
/// for each of `R8Uint`, `R16Uint`, `R32Uint`, `RGBA8Unorm`, `RG32Uint`, `RGBA16Uint` and
/// `RGBA32Uint`, of 37 x 5, 33 x 33, 64 x 64 and 100 x 70 pixels, 2 and 4 samples: each sample
/// uploaded by a blit naming it and found through a buffer over the heap, and bytes written there
/// read back by a blit (`heap_textures::sample_round_trip`, Metal's tiles of `samples · B`-byte
/// elements, sample `s` at `s · B`: on the M3 Max under macOS 27.0.1 every sample of these shapes
/// and of 8 x 8, 16 x 16 and 300 x 200 lies there, drawn per sample or blitted). On a driver
/// whose multisample colour images lie so
/// (`DriverFact::MultisampleColorImagesLieAsMetalHeapTextures`: MoltenVK 1.4.2 on the M3 Max,
/// measured by this lane) every sample lies there and the device counts and announces none for
/// `heap-texture-untiled-layout`; elsewhere it counts every texture but a shadow of its samples
/// in Metal's tiles (`shadowed-texture`, announced: the NVIDIA RTX 5080 Laptop's,
/// `texture/heap.rs`), and one it counts for none lies in Metal's tiles each way. Then one
/// Metal keeps compressed (`RGBA8Unorm`, 64 x 64, 4 samples, the property on), whose
/// `heap-texture-lossless-layout` notice says its image lies in the heap's bytes, not in memory of
/// its own. `HEAPTEX device-samples` lines print the samples off Metal's tiles.
#[test]
fn device_heap_multisample_colour_textures_lie_where_announced() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let untiled = Degradation::HeapTextureUntiledLayout;
    let device = &validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let as_metal = (device.properties().driver_facts)
        .contains(&DriverFact::MultisampleColorImagesLieAsMetalHeapTextures);
    let queue = device.new_command_queue().unwrap();
    let options = ResourceOptions::new(
        StorageMode::Private,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Tracked,
    );
    let formats = [
        (PixelFormat::R8Uint, 1),
        (PixelFormat::R16Uint, 2),
        (PixelFormat::R32Uint, 4),
        (PixelFormat::RGBA8Unorm, 4),
        (PixelFormat::RG32Uint, 8),
        (PixelFormat::RGBA16Uint, 8),
        (PixelFormat::RGBA32Uint, 16),
    ];
    let descriptor = |format, [w, h]: [u64; 2], samples, allow| {
        let mut descriptor = TextureDescriptor::texture_2d(format, w, h, false);
        descriptor.texture_type = TextureType::Type2DMultisample;
        descriptor.sample_count = samples;
        descriptor.resource_options = options;
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        descriptor.allow_gpu_optimized_contents = allow;
        descriptor
    };
    let (mut textures, mut counted, mut off_tiles) = (0, 0, 0);
    for (format, unit) in formats {
        for side in [[37, 5], [33, 33], [64, 64], [100, 70]] {
            for samples in [2, 4] {
                let [w, h] = side;
                let what = format!("{format:?} {w}x{h} x{samples}");
                let descriptor = descriptor(format, side, samples, false);
                let answer = match device.heap_texture_size_and_align(&descriptor) {
                    Err(Error::Unsupported(u)) if u.feature == Feature::TextureSampleCount => {
                        eprintln!("HEAPTEX device-samples {what}: no such sample count");
                        continue;
                    }
                    answer => answer.unwrap(),
                };
                let heap = device
                    .new_heap(&HeapDescriptor::new(
                        HeapType::Placement,
                        answer.size,
                        options,
                    ))
                    .unwrap();
                let before = device.degraded_objects(untiled).unwrap();
                let texture = heap.new_texture_at_offset(&descriptor, 0).unwrap();
                let announced = device.degraded_objects(untiled).unwrap() > before;
                let over = heap.new_buffer_at_offset(answer.size, options, 0).unwrap();
                gpu_fill(&queue, &over, 0..answer.size, 0xee);
                let read = || gpu_read(&queue, &over, 0..answer.size);
                let write = |bytes: &[u8]| {
                    let source = buffer_with(device, bytes).unwrap();
                    let command_buffer = queue.command_buffer().unwrap();
                    let mut copy = command_buffer.blit_command_encoder().unwrap();
                    copy.copy_from_buffer_to_buffer(&source, 0, &over, 0, answer.size)
                        .unwrap();
                    copy.end_encoding().unwrap();
                    run(&command_buffer);
                };
                let aspects: [SampleAspect; 1] = [(unit, BlitOption::None, 0, 0)];
                let Some((off, read_off)) =
                    sample_round_trip(&queue, &texture, (side, samples), &aspects, &read, &write)
                else {
                    eprintln!("HEAPTEX device-samples {what}: its blits refused");
                    continue;
                };
                eprintln!(
                    "HEAPTEX device-samples {what}: size {} align {}; {off} of {} samples off \
                     Metal's tiles, {read_off} read otherwise from them; announced {announced}",
                    answer.size,
                    answer.align,
                    w * h * samples
                );
                textures += 1;
                counted += u64::from(announced);
                off_tiles += u64::from(off + read_off > 0);
                if as_metal {
                    assert!(!announced, "{what}: counted untiled");
                }
                if !announced {
                    assert_eq!((off, read_off), (0, 0), "{what}: samples off Metal's tiles");
                }
            }
        }
    }
    eprintln!(
        "HEAPTEX device-samples: {textures} textures, {counted} counted untiled, {off_tiles} with \
         samples off Metal's tiles (fact {as_metal})"
    );
    let kinds: Vec<Degradation> = (messages.take_notices().iter())
        .filter_map(|message| message.degradation.as_ref())
        .map(|notice| notice.degradation)
        .collect();
    let untiled_notices = kinds.iter().filter(|&&kind| kind == untiled).count();
    assert_eq!(untiled_notices, usize::from(counted > 0), "untiled notices");
    // The textures not counted on a driver without the fact are shadows of Metal's tiles.
    if !as_metal && counted < textures {
        assert!(
            kinds.contains(&Degradation::ShadowedTexture),
            "shadows of Metal's tiles: {kinds:?}"
        );
    }
    let lossless = Degradation::HeapTextureLosslessLayout;
    let compressed = descriptor(PixelFormat::RGBA8Unorm, [64, 64], 4, true);
    let answer = device.heap_texture_size_and_align(&compressed).unwrap();
    let heap = device
        .new_heap(&HeapDescriptor::new(
            HeapType::Placement,
            answer.size,
            options,
        ))
        .unwrap();
    let before = device.degraded_objects(lossless).unwrap();
    let texture = heap.new_texture_at_offset(&compressed, 0).unwrap();
    assert_eq!(device.degraded_objects(lossless).unwrap(), before + 1);
    let notices = messages.take_notices();
    let reasons: Vec<&str> = (notices.iter())
        .filter_map(|message| message.degradation.as_ref())
        .filter(|notice| notice.degradation == lossless)
        .map(|notice| notice.reason.as_str())
        .collect();
    eprintln!("HEAPTEX device-samples compressed: {reasons:?}");
    assert_eq!(reasons.len(), 1, "{reasons:?}");
    assert!(
        reasons[0].contains("its image lies in the heap's bytes")
            && !reasons[0].contains("memory of its own"),
        "{}",
        reasons[0]
    );
    drop((texture, heap));
    layer_reported_nothing_at_all(&messages);
}
