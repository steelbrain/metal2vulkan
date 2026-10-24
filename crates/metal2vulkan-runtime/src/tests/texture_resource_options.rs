//! A texture's `resourceOptions` (`Texture::resource_options`): the storage and cache modes its
//! descriptor stated, and the hazard tracking mode the texture **resolved** rather than the one
//! the descriptor asked for.
//!
//! Metal reports no resource's options as `MTLHazardTrackingModeDefault`: a texture made outside a
//! heap reads `0x200` shared and `0x220` private, measured on an Apple M3 Max under macOS 27.0.1
//! (the `resource-own-getters`, `resource-labels` and `resource-aliasing` conformance cases, whose
//! sixteen `resource_options` sites recorded Metal's `0x200`/`0x220` against our `0x0`/`0x20`).
//! `Texture::hazard_tracking_mode` resolved it correctly all along, so the two getters disagreed:
//! the word is the mode, and every shape here holds the two to one answer.
//!
//! A heap texture's and a buffer texture's words are held to their own
//! `hazard_tracking_mode`, which is a relation between two getters of ours rather than a claim
//! about Metal's answer for those shapes, which nothing here has measured.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Nothing runs on the GPU.

use crate::{
    CpuCacheMode, Device, DeviceKind, DeviceOptions, HazardTrackingMode, HeapDescriptor, HeapType,
    ResourceOptions, StorageMode, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU.
fn device() -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// A 64 by 64 `RGBA8Unorm` texture descriptor of four levels, read by a shader, of `options`.
fn descriptor(options: ResourceOptions) -> TextureDescriptor {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 64, 64, true);
    d.mipmap_level_count = 4;
    d.usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
    d.resource_options = options;
    d
}

/// `texture`'s options word, its resolved hazard tracking mode in it and never `Default`.
fn word_holds_the_resolved_mode(texture: &Texture, what: &str) -> u64 {
    let options = texture.resource_options();
    let mode = texture.hazard_tracking_mode();
    assert_ne!(mode, HazardTrackingMode::Default, "{what}: resolved mode");
    assert_eq!(
        options.hazard_tracking_mode(),
        Ok(mode),
        "{what}: {options:?} against {mode:?}"
    );
    assert_eq!(
        texture.storage_mode(),
        options.storage_mode().unwrap(),
        "{what}: storage mode"
    );
    assert_eq!(
        texture.cpu_cache_mode(),
        options.cpu_cache_mode().unwrap(),
        "{what}: cache mode"
    );
    options.raw()
}

/// A texture outside a heap reports the mode it resolved, which for a descriptor's `Default` is
/// `Tracked`: Metal's `0x200` shared and `0x220` private. A descriptor that states a mode reports
/// that one, and a view reports its texture's.
#[test]
fn a_device_textures_options_report_the_mode_it_resolved() {
    let device = device();
    let options =
        |storage, hazard| ResourceOptions::new(storage, CpuCacheMode::DefaultCache, hazard);
    for (storage, shift) in [(StorageMode::Shared, 0x0), (StorageMode::Private, 0x20)] {
        for (hazard, field) in [
            (HazardTrackingMode::Default, 0x200),
            (HazardTrackingMode::Tracked, 0x200),
            (HazardTrackingMode::Untracked, 0x100),
        ] {
            let what = format!("{storage:?} {hazard:?}");
            let texture = device
                .new_texture(&descriptor(options(storage, hazard)))
                .unwrap();
            assert_eq!(
                word_holds_the_resolved_mode(&texture, &what),
                shift | field,
                "{what}"
            );
            let view = texture.new_texture_view(PixelFormat::RGBA8Uint).unwrap();
            assert_eq!(
                word_holds_the_resolved_mode(&view, &format!("{what} view")),
                shift | field,
                "{what} view"
            );
        }
    }
}

/// A heap texture's and a buffer texture's words carry their own resolved mode too: a texture
/// placed in a heap of no stated mode is untracked, as `Identity::new` resolves it, and a texture
/// over a buffer takes the buffer's.
#[test]
fn a_placed_textures_options_report_its_own_resolved_mode() {
    let device = device();
    let private = ResourceOptions::private();
    let heap = device
        .new_heap(&HeapDescriptor::new(HeapType::Automatic, 4 << 20, private))
        .unwrap();
    let mut d = descriptor(private);
    d.usage = TextureUsage::SHADER_READ;
    let placed = heap.new_texture(&d).unwrap();
    assert_eq!(
        word_holds_the_resolved_mode(&placed, "a heap texture"),
        0x120,
        "a heap texture takes the heap's untracked mode"
    );

    let buffer = device
        .new_buffer(64 * 64 * 4, ResourceOptions::shared())
        .unwrap();
    let mut flat = descriptor(ResourceOptions::shared());
    flat.mipmap_level_count = 1;
    let over = buffer.new_texture(&flat, 0, 64 * 4).unwrap();
    assert_eq!(
        word_holds_the_resolved_mode(&over, "a texture over a buffer"),
        0x200,
        "a texture over a shared buffer outside a heap is tracked"
    );
}
