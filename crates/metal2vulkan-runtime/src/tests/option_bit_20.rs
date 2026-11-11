//! `MTLResourceOptions` bit 20 ([`ResourceOptions::BIT_20`]), as Metal answers it on an M3 Max and
//! a Radeon Pro 5300M (`crates/metal2vulkan-conformance/probes/resource_option_bit20.m`): a buffer
//! of a length, of bytes or over host pages, of each storage mode, a heap and a heap's buffer are
//! made with it and report their options without it; a texture descriptor with it is refused
//! (`Invalid`, `Problem::UndefinedValue`), where Metal ends the process.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Host, Scratch, BYTES};
use crate::{
    Error, HeapDescriptor, HeapType, Invalid, Problem, ResourceOptions, StorageMode,
    TextureDescriptor,
};
use metal2vulkan::metal::PixelFormat;

/// `options` with bit 20.
fn with_bit_20(options: ResourceOptions) -> ResourceOptions {
    ResourceOptions::from_raw(options.raw() | ResourceOptions::BIT_20).unwrap()
}

/// The options of a storage mode, the other modes at their defaults.
fn of(storage: StorageMode) -> ResourceOptions {
    ResourceOptions::shared().with_storage_mode(storage)
}

#[test]
fn bit_20_is_taken_by_buffers_and_heaps_and_refused_by_textures() {
    let scratch = Scratch::new("option-bit-20");
    let device = device(&scratch.0);
    // Reported as Metal reports them: the storage mode, the hazard tracking mode resolved to
    // tracked (0x200), and no bit 20.
    for storage in [
        StorageMode::Shared,
        StorageMode::Managed,
        StorageMode::Private,
    ] {
        let asked = with_bit_20(of(storage));
        let reported = 0x200 | storage.raw() << 4;
        let buffer = device.new_buffer(4096, asked).unwrap();
        assert_eq!(buffer.resource_options().raw(), reported, "{storage:?}");
        if storage == StorageMode::Private {
            continue;
        }
        let buffer = device.new_buffer_with_bytes(&[7; 64], asked).unwrap();
        assert_eq!(buffer.resource_options().raw(), reported, "{storage:?}");
        let host = Host::new();
        // SAFETY: the pages outlive the buffer, which alone uses them.
        let made = unsafe {
            device.new_buffer_with_bytes_no_copy(host.memory(), BYTES as u64, asked, None)
        };
        match made {
            Ok(buffer) => assert_eq!(buffer.resource_options().raw(), reported, "{storage:?}"),
            Err(Error::Unsupported(gap)) => {
                eprintln!("OPTION-BIT-20: the device imports no host memory: {gap:?}");
            }
            Err(error) => panic!("no-copy {storage:?}: {error:?}"),
        }
    }
    for storage in [StorageMode::Shared, StorageMode::Private] {
        let descriptor = HeapDescriptor {
            heap_type: HeapType::Automatic,
            size: 1 << 20,
            resource_options: with_bit_20(of(storage)),
            ..HeapDescriptor::default()
        };
        let heap = device.new_heap(&descriptor).unwrap();
        assert_eq!(
            heap.resource_options().without_bit_20(),
            heap.resource_options()
        );
        let buffer = heap.new_buffer(4096, with_bit_20(of(storage))).unwrap();
        assert_eq!(
            buffer.resource_options().raw() & ResourceOptions::BIT_20,
            0,
            "{storage:?}"
        );
    }
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 16, 16, false);
    descriptor.resource_options = with_bit_20(ResourceOptions::shared());
    let raw = descriptor.resource_options.raw();
    assert_eq!(
        device.new_texture(&descriptor).err(),
        Some(Error::Invalid(Invalid::new(
            "new_texture",
            Problem::UndefinedValue {
                ty: "MTLResourceOptions",
                raw,
            },
        )))
    );
}
