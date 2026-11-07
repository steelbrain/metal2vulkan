//! The lifetimes a new image's waiting first move depends on, checked on the image itself, so a
//! lapse fails here on every device rather than as undefined behaviour one driver happens to crash
//! on (a barrier on a destroyed image: RADV's SIGSEGV in our probe; lavapipe under the
//! validation layer passed it):
//!
//! - a move of a `VkImage` a detach let go of is dropped, never recorded;
//! - a host call's copy of an image over host pages makes the image's waiting move first, which
//!   no batch makes until one names the image;
//! - an image in a buffer's memory holds the memory, so it outlives a batch that took its move
//!   and holds only the image; and never the buffer, which goes at the program's release, as Metal
//!   frees it.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Where the device makes no image of the
//! kind a case needs (no host memory import, or another route for the texture), the case says
//! so and checks nothing.

use super::first_move::Fresh;
use super::{Image, Owner, Storage, Texture};
use crate::buffer::MemoryHold;
use crate::device::lock;
use crate::{
    Device, DeviceKind, DeviceOptions, HeapDescriptor, HeapType, HostMemory, Region,
    ResourceOptions, ShaderCacheOptions, TextureDescriptor,
};
use metal2vulkan::metal::PixelFormat;
use std::alloc::Layout;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;

/// The host memory's bytes: a whole page of any host.
const BYTES: usize = 64 << 10;

/// The textures' side, in texels: 16 x 16 `RGBA8Unorm`, which a heap over host pages keeps
/// untiled, bound to the pages, where Metal compresses it (`heap_tiled.rs`).
const SIDE: u64 = 16;

/// A row of the textures' texels, in bytes.
const ROW: u64 = SIDE * 4;

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(case: &str) -> Scratch {
        let path = std::env::temp_dir().join(format!("m2v-{case}-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Page-aligned host memory of [`BYTES`], freed when dropped: declared before the device that
/// uses it, so dropped after it.
struct Host(NonNull<u8>);

impl Host {
    fn layout() -> Layout {
        Layout::from_size_align(BYTES, BYTES).unwrap()
    }

    fn new() -> Host {
        // SAFETY: the layout's size is not zero.
        let pointer = unsafe { std::alloc::alloc_zeroed(Host::layout()) };
        Host(NonNull::new(pointer).unwrap())
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `Host::new`.
        unsafe { std::alloc::dealloc(self.0.as_ptr(), Host::layout()) };
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache
/// in `cache`.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// A 16 x 16 `RGBA8Unorm` `Shared` texture's descriptor.
fn descriptor() -> TextureDescriptor {
    TextureDescriptor {
        resource_options: ResourceOptions::shared(),
        ..TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false)
    }
}

/// Texels naming their byte.
fn texels() -> Vec<u8> {
    (0..ROW * SIDE).map(|byte| (byte % 251) as u8).collect()
}

/// The image `texture`'s storage holds, as an alias of a buffer's bytes or as its own.
fn image_of(texture: &Texture) -> Option<&Arc<Image>> {
    match texture.storage() {
        Storage::Alias { image, .. } | Storage::Image(image) => Some(image),
        Storage::Texels(_) | Storage::Shadow { .. } => None,
    }
}

/// Whether `image`'s move waits, and is one only a batch naming the image makes (an image over
/// host pages).
fn waits_for_its_name(image: &Image) -> bool {
    lock(&image.first_move)
        .as_ref()
        .is_some_and(|first| first.rebindable())
}

/// A texture over a no-copy buffer that only the host wrote keeps its image's move waiting (no
/// command named it); a detach lets that `VkImage` go with the pages. The move is then dropped
/// as the next submission takes it, and none is recorded on the destroyed image.
#[test]
fn a_waiting_move_of_an_image_a_detach_let_go_of_is_dropped() {
    let host = Host::new();
    let scratch = Scratch::new("first-move-detach");
    let device = device(&scratch.0);
    // SAFETY: the memory outlives the device (dropped after it), and nothing else uses it.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(host.0),
            BYTES as u64,
            ResourceOptions::shared(),
            None,
        )
    };
    let Ok(buffer) = made else {
        eprintln!(
            "first move lifetimes: no buffer over host memory on {}",
            device.name()
        );
        return;
    };
    let texture = buffer.new_texture(&descriptor(), 0, ROW).unwrap();
    let image = match image_of(&texture) {
        Some(image) if waits_for_its_name(image) => Arc::clone(image),
        _ => {
            eprintln!(
                "first move lifetimes: the texture over host pages on {} is no image bound to \
                 them",
                device.name()
            );
            return;
        }
    };
    texture
        .replace_region(Region::new_2d(0, 0, SIDE, SIDE), 0, 0, &texels(), ROW, 0)
        .unwrap();
    assert!(waits_for_its_name(&image), "the host's write named nothing");
    let own = lock(&image.first_move).as_ref().unwrap().image;
    assert_eq!(buffer.detach_backing(), Ok(true));
    assert!(!image.holds(own), "the detach let its VkImage go");
    assert!(
        image.take().is_empty(),
        "a move of the VkImage the detach let go of was taken"
    );
}

/// A host call's copy into a texture in a heap over host pages, before any command named it,
/// makes the image's waiting move first, on its own submission: no batch would until one names
/// the image, and the copy is recorded in `GENERAL`. The bytes read back are the bytes written.
#[test]
fn a_host_copy_makes_the_waiting_move_of_an_image_over_host_pages() {
    let host = Host::new();
    let scratch = Scratch::new("first-move-host-copy");
    let device = device(&scratch.0);
    let heap_descriptor =
        HeapDescriptor::new(HeapType::Automatic, BYTES as u64, ResourceOptions::shared());
    // SAFETY: the memory outlives the device (dropped after it), and nothing else uses it.
    let made = unsafe {
        device.new_heap_with_bytes_no_copy(HostMemory::new(host.0), &heap_descriptor, None, None)
    };
    let Ok(heap) = made else {
        eprintln!(
            "first move lifetimes: no heap over host memory on {}",
            device.name()
        );
        return;
    };
    let texture = heap.new_texture(&descriptor()).unwrap();
    let image = match image_of(&texture) {
        Some(image) if waits_for_its_name(image) => Arc::clone(image),
        _ => {
            eprintln!(
                "first move lifetimes: the heap texture over host pages on {} is no image bound \
                 to them",
                device.name()
            );
            return;
        }
    };
    let written = texels();
    texture
        .replace_region(Region::new_2d(0, 0, SIDE, SIDE), 0, 0, &written, ROW, 0)
        .unwrap();
    assert!(
        lock(&image.first_move).is_none(),
        "the copy left the image's move waiting"
    );
    let mut read = vec![0; written.len()];
    texture
        .get_bytes(&mut read, ROW, 0, Region::new_2d(0, 0, SIDE, SIDE), 0, 0)
        .unwrap();
    assert_eq!(read, written);
}

/// An image in a buffer's memory holds the memory, not the buffer: with the texture and the buffer
/// dropped, the buffer goes, as Metal frees it at their release, and the image
/// (which a batch that took its move holds alone) keeps the buffer's memory until it goes.
#[test]
fn an_image_in_a_buffer_s_memory_holds_the_memory_not_the_buffer() {
    let scratch = Scratch::new("first-move-buffer");
    let device = device(&scratch.0);
    let buffer = device
        .new_buffer(ROW * SIDE, ResourceOptions::shared())
        .unwrap();
    let texture = buffer.new_texture(&descriptor(), 0, ROW).unwrap();
    let Some(image) = image_of(&texture).map(Arc::clone) else {
        eprintln!(
            "first move lifetimes: the texture over a buffer on {} is no image in its memory",
            device.name()
        );
        return;
    };
    let pages = match &image.owner {
        Owner::Buffer { memory } => match &*lock(memory) {
            Some(MemoryHold::Fixed { _pages }) => Arc::downgrade(_pages),
            _ => panic!("the image holds no memory of the runtime's"),
        },
        _ => panic!("the image is no buffer's"),
    };
    let weak = buffer.downgrade();
    drop(texture);
    drop(buffer);
    assert!(!weak.lives(), "the image kept the buffer");
    assert!(
        pages.strong_count() > 0,
        "the memory went while its image lives"
    );
    drop(image);
    assert_eq!(
        pages.strong_count(),
        0,
        "the image kept the memory after it went"
    );
}
