//! Resources past a full device-local heap (`memory.rs`'s `overflow_candidates`). Metal makes
//! `Private` and `Managed` resources on a discrete GPU past its memory rather than refusing them (a
//! 4 GiB Radeon Pro 5300M made, used and read back 12 GiB of `Managed` buffers, six 1 GiB `Private`
//! buffers and a 17 GB `Private` heap), so the runtime places such a resource in system memory
//! where no device-local memory type it takes has room.
//!
//! - **Filled.** On a device without unified memory whose `Private` buffers may overflow,
//!   `Private` buffers of [`FILL`] bytes are made until one overflows
//!   (`WorkCounts::device_memory_overflows` moves) or the device-local heap's bytes and [`MARGIN`]
//!   more are made: a driver that pages device-local memory out to make room
//!   (`VK_EXT_pageable_device_local_memory`) may take them all, and nothing overflows there.
//! - **Past it.** `Private` buffers, a `Private` heap with a buffer and a texture in it, `Private`
//!   textures and `Managed` buffers are made, each written by the GPU (a fill, or a copy of a
//!   pattern) and read back, and a `Managed` buffer's host writes reach the GPU: every byte is
//!   the one written. Where the device-local heap was full, something past it overflowed, and
//!   `device-memory-overflow` is announced once.
//! - **Room again.** Once every resource is gone, a `Private` buffer lies in device-local memory.
//! - **Unified memory.** On a device with unified memory nothing is filled: the same resources are
//!   made and read back, and nothing overflows or is announced.
//!
//! The test prints the device's memory types and heaps and what each resource past the fill lies
//! in (`VRAM-OVERFLOW`).
//!
//! A GPU hazard (it allocates up to the device's memory): compiled on a GPU only under
//! `m2v_gpu_hazards`, and under `m2v_cpu_icd`, where lavapipe's memory is unified.

use super::blit_holds::WAIT;
use crate::device::lock;
use crate::memory::{candidates, overflow_candidates, MemoryClass};
use crate::{
    BlitCommandEncoder, BlitOption, Buffer, CommandBufferStatus, CommandQueue, CpuCacheMode,
    Degradation, Device, DeviceKind, DeviceOptions, HazardTrackingMode, HeapDescriptor, HeapType,
    MessageSink, Origin, ResourceOptions, Size, StorageMode, Texture, TextureDescriptor,
};
use metal2vulkan::metal::PixelFormat;
use std::sync::{Arc, Mutex};

/// The bytes of each `Private` buffer that fills the device-local heap.
const FILL: u64 = 256 << 20;
/// The bytes made past the device-local heap's size before the fill stops without an overflow.
const MARGIN: u64 = 2 << 30;
/// The bytes of each buffer past the fill.
const PAST: u64 = 64 << 20;
/// The side of each texture past the fill, `RGBA8Unorm`: 4 MiB.
const SIDE: u64 = 1024;
const TEXTURE_BYTES: u64 = SIDE * SIDE * 4;

/// The notices the device announced, by kind.
type Heard = Arc<Mutex<Vec<Degradation>>>;

fn device() -> (Device, Heard) {
    let heard: Heard = Arc::default();
    let hearing = Arc::clone(&heard);
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        messages: Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                hearing.lock().unwrap().push(notice.degradation);
            }
        })),
        ..DeviceOptions::default()
    };
    (Device::system_default_with(&options).unwrap(), heard)
}

fn overflows(device: &Device) -> u64 {
    device.work_counts().device_memory_overflows
}

fn announced(heard: &Heard) -> usize {
    let notices = heard.lock().unwrap();
    notices
        .iter()
        .filter(|&&kind| kind == Degradation::DeviceMemoryOverflow)
        .count()
}

fn private() -> ResourceOptions {
    ResourceOptions::private()
}

fn managed() -> ResourceOptions {
    ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    )
}

fn texture_descriptor() -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.resource_options = private();
    descriptor
}

/// Commit one command buffer of `queue` with a blit encoder for each of `encodes`, in order, and
/// wait for it.
fn blits(queue: &CommandQueue, encodes: &mut [&mut dyn FnMut(&mut BlitCommandEncoder)]) {
    let command_buffer = queue.command_buffer().unwrap();
    for encode in encodes {
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        encode(&mut encoder);
        encoder.end_encoding().unwrap();
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The first `length` bytes of the host-visible `buffer`.
fn host_bytes(buffer: &Buffer, length: u64) -> &[u8] {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's host bytes, at least `length` long, which no command writes now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) }
}

/// Whether the first `length` bytes of `buffer` are all `value`.
fn all(buffer: &Buffer, length: u64, value: u8) -> bool {
    host_bytes(buffer, length).iter().all(|&byte| byte == value)
}

/// The GPU fills `buffer` with `value`, then copies it into `out`, which then holds `value`.
fn fill_and_read(queue: &CommandQueue, buffer: &Buffer, out: &Buffer, value: u8) {
    blits(
        queue,
        &mut [
            &mut |encoder| encoder.fill_buffer(buffer, 0..PAST, value).unwrap(),
            &mut |encoder| {
                (encoder.copy_from_buffer_to_buffer(buffer, 0, out, 0, PAST)).unwrap();
            },
        ],
    );
    assert!(
        all(out, PAST, value),
        "a buffer past the fill reads {value}"
    );
}

/// The GPU copies `pattern`'s bytes, all `value`, into `texture`, then the texture into `out`,
/// which then holds `value`.
fn copy_and_read(queue: &CommandQueue, texture: &Texture, pattern: &Buffer, out: &Buffer) {
    let value = host_bytes(pattern, 1)[0];
    let size = Size::new(SIDE, SIDE, 1);
    blits(
        queue,
        &mut [
            &mut |encoder| {
                let origin = Origin::new(0, 0, 0);
                (encoder.copy_from_buffer_to_texture(
                    pattern,
                    0,
                    SIDE * 4,
                    TEXTURE_BYTES,
                    size,
                    texture,
                    0,
                    0,
                    origin,
                    BlitOption::None,
                ))
                .unwrap();
            },
            &mut |encoder| {
                let origin = Origin::new(0, 0, 0);
                (encoder.copy_from_texture_to_buffer(
                    texture,
                    0,
                    0,
                    origin,
                    size,
                    out,
                    0,
                    SIDE * 4,
                    TEXTURE_BYTES,
                    BlitOption::None,
                ))
                .unwrap();
            },
        ],
    );
    assert!(
        all(out, TEXTURE_BYTES, value),
        "a texture past the fill reads {value}"
    );
}

/// A `Managed` buffer's host writes reach the GPU (`did_modify_range`, read by a copy into
/// `out`), and the GPU's fill reaches its host copy (`synchronize_resource`).
fn managed_round_trip(queue: &CommandQueue, buffer: &Buffer, out: &Buffer, value: u8) {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's host copy, `PAST` bytes long, which no command uses now.
    unsafe { std::ptr::write_bytes(contents.as_ptr(), value, PAST as usize) };
    buffer.did_modify_range(0..PAST).unwrap();
    blits(
        queue,
        &mut [&mut |encoder| {
            (encoder.copy_from_buffer_to_buffer(buffer, 0, out, 0, PAST)).unwrap();
        }],
    );
    assert!(
        all(out, PAST, value),
        "a managed buffer's GPU copy reads {value}"
    );
    let filled = value.wrapping_add(1);
    blits(
        queue,
        &mut [
            &mut |encoder| encoder.fill_buffer(buffer, 0..PAST, filled).unwrap(),
            &mut |encoder| encoder.synchronize_resource(buffer.as_resource()).unwrap(),
        ],
    );
    assert!(
        all(buffer, PAST, filled),
        "a managed buffer's host copy reads {filled}"
    );
}

#[test]
fn resources_past_device_local_memory_are_made_used_and_read_back() {
    let (device, heard) = device();
    let shared = device.shared();
    let view = lock(&shared.memory).type_view(shared.vk()).unwrap();
    let unified = device.properties().has_unified_memory;
    let buffer_types = overflow_candidates(&view.types, view.buffers, MemoryClass::Private);
    let image_types = overflow_candidates(&view.types, view.images, MemoryClass::Private);
    let own = candidates(&view.types, view.buffers, MemoryClass::Private, unified);
    let heap = own
        .first()
        .map(|&best| view.types[best as usize].heap_index as usize);
    let heap_bytes = heap.map_or(0, |heap| view.heap_sizes[heap]);
    let pageable = device.enables_extension(ash::ext::pageable_device_local_memory::NAME);
    let types: Vec<String> = (view.types.iter())
        .map(|ty| format!("{:#x}@{}", ty.property_flags.as_raw(), ty.heap_index))
        .collect();
    eprintln!(
        "VRAM-OVERFLOW on {}: unified {unified}, pageable {pageable}, types {types:?}, heaps {:?}, \
         buffer types {:#b}, image types {:#b}, Private buffers overflow into {buffer_types:?}, \
         images into {image_types:?}",
        device.name(),
        view.heap_sizes,
        view.buffers,
        view.images,
    );
    let start = overflows(&device);
    let mut filled = Vec::new();
    if !unified && !buffer_types.is_empty() {
        while overflows(&device) == start && (filled.len() as u64) * FILL < heap_bytes + MARGIN {
            filled.push(device.new_buffer(FILL, private()).unwrap());
        }
    }
    let full = overflows(&device) > start;
    eprintln!(
        "VRAM-OVERFLOW: filled {} buffers of {FILL} bytes against a heap of {heap_bytes}; \
         the device-local heap was full: {full}",
        filled.len(),
    );
    let past = overflows(&device);
    let queue = device.new_command_queue().unwrap();
    let out = device.new_buffer(PAST, ResourceOptions::shared()).unwrap();
    let pattern = device
        .new_buffer(TEXTURE_BYTES, ResourceOptions::shared())
        .unwrap();
    let mut value = 0x30;
    let buffers: Vec<Buffer> = (0..4)
        .map(|_| device.new_buffer(PAST, private()).unwrap())
        .collect();
    for buffer in &buffers {
        value += 1;
        fill_and_read(&queue, buffer, &out, value);
    }
    let private_heap = (device.new_heap(&HeapDescriptor::new(
        HeapType::Automatic,
        2 * PAST,
        private(),
    )))
    .unwrap();
    let in_heap = private_heap.new_buffer(PAST, private()).unwrap();
    value += 1;
    fill_and_read(&queue, &in_heap, &out, value);
    let heap_texture = private_heap.new_texture(&texture_descriptor()).unwrap();
    value += 1;
    // SAFETY: the pattern's host bytes, which no command uses now.
    unsafe {
        std::ptr::write_bytes(
            pattern.contents().unwrap().as_ptr(),
            value,
            TEXTURE_BYTES as usize,
        )
    };
    copy_and_read(&queue, &heap_texture, &pattern, &out);
    let textures: Vec<Texture> = (0..4)
        .map(|_| device.new_texture(&texture_descriptor()).unwrap())
        .collect();
    for texture in &textures {
        value += 1;
        // SAFETY: as above.
        unsafe {
            std::ptr::write_bytes(
                pattern.contents().unwrap().as_ptr(),
                value,
                TEXTURE_BYTES as usize,
            )
        };
        copy_and_read(&queue, texture, &pattern, &out);
    }
    let managed: Vec<Buffer> = (0..2)
        .map(|_| device.new_buffer(PAST, managed()).unwrap())
        .collect();
    for buffer in &managed {
        value += 2;
        managed_round_trip(&queue, buffer, &out, value);
    }
    let lies = |overflowed: bool| match overflowed {
        true => "system",
        false => "device-local",
    };
    let buffer_lies: Vec<&str> = (buffers.iter().chain(&managed))
        .map(|buffer| lies(buffer.own_pages_overflowed()))
        .collect();
    let texture_lies: Vec<&str> = (textures.iter())
        .map(|texture| lies(texture.overflowed()))
        .collect();
    let past_overflows = overflows(&device) - past;
    eprintln!(
        "VRAM-OVERFLOW: past the fill, Private then Managed buffers in {buffer_lies:?}, textures \
         in {texture_lies:?}; {past_overflows} placements overflowed, {} announced",
        announced(&heard),
    );
    if full {
        assert!(past_overflows > 0, "something past a full heap overflowed");
        assert_eq!(announced(&heard), 1, "announced once");
    }
    if unified || buffer_types.is_empty() {
        assert_eq!(overflows(&device), start, "nothing overflowed");
        assert_eq!(announced(&heard), 0, "nothing announced");
    }
    drop((
        buffers,
        managed,
        textures,
        heap_texture,
        in_heap,
        private_heap,
    ));
    drop(filled);
    let again = device.new_buffer(PAST, private()).unwrap();
    value += 1;
    fill_and_read(&queue, &again, &out, value);
    assert!(
        !again.own_pages_overflowed(),
        "a buffer made once the fill is gone lies in device-local memory"
    );
}
