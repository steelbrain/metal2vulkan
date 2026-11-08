//! A program's texture released while a runtime thread holds it for a moment gives back what it
//! held before the program's next making takes it (`resource_ids.rs`, **Passing holds**): a
//! batch holds the slots' writes (`heap_sets::hold`, as the in-place check's submissions do on
//! lavapipe), so the slot of the texture made meanwhile is written as the hold ends, on the
//! batch's thread, which takes the texture to write it. The tests pause that thread there
//! (`DeviceHeaps::pause_holding`), holding the texture, while the program releases it and makes
//! the next resource:
//!
//! - a texture: without the wait for passing holds it took ID 2, the released one's ID still held
//!   by the batch's handle, where Metal's next texture takes 1
//!   (`tests::in_place_check::texture_ids_keep_metals_order_while_the_in_place_check_runs` failed
//!   so on lavapipe, in two runs of the runtime's suite, its textures made while the check's
//!   submissions held the writes);
//! - a buffer pinned where the buffer the texture was made over lay, a buffer in the slot of an
//!   address pool that buffer lay in, and a texture in an automatic heap the texture filled:
//!   without the wait, and without the texture's going inside the hold, each was refused (a
//!   `PlacementConflict::Held` pin and slot, an `OutOfMemory(Heap)` heap), the bytes still held by
//!   the texture the batch's handle kept, where Metal's release had freed them.
//!
//! A hold that lets go of a texture's last handle runs none of the program's code inside the hold
//! (`exit_guard::program_release`): a texture over a no-copy buffer, the buffer released, whose
//! last handle a passing hold lets go of, gives its ID to the texture made next inside the hold,
//! and its buffer's deallocator runs once the hold ended, not inside it.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, over_host, shadow_device, Host, Scratch};
use crate::device::lock;
use crate::heap_sets;
use crate::{
    AddressPlacement, Buffer, BufferContents, BufferDescriptor, Device, Error, GpuAddress,
    HeapDescriptor, HeapType, MemoryKind, ResourceOptions, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long the batch's thread holds the released texture at most: until the program's next
/// making answered, where that making does not wait for it, else this long.
const HOLDING: Duration = Duration::from_millis(500);

/// The name of the batch's thread, which alone pauses (the in-place check's own batches end
/// their holds on the device meanwhile).
const BATCH: &str = "passing-holds-batch";

/// Where the pinned buffers lie: the first byte of a 32-KiB granule of the Metal space.
const PIN: GpuAddress = GpuAddress(1 << 40);

/// A private 4x4 `RGBA8Unorm` texture of the program's, at a plain ID.
fn texture(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// The row pitch of a 4x4 `RGBA8Unorm` texture over a buffer of `device`'s, and the buffer's
/// length for it.
fn rows(device: &Device) -> (u64, u64) {
    let alignment = device
        .minimum_linear_texture_alignment_for_pixel_format(PixelFormat::RGBA8Unorm)
        .unwrap();
    let row = alignment.max(16);
    (row, 4 * row)
}

/// A 4x4 `RGBA8Unorm` texture over `buffer`'s bytes, at a plain ID.
fn over(buffer: &Buffer) -> Texture {
    let (row, _) = rows(buffer.device());
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = buffer.resource_options();
    buffer.new_texture(&descriptor, 0, row).unwrap()
}

/// Make what `make` makes while a batch holds the slots' writes, so the batch's thread writes the
/// slot of the texture among it as it ends the hold, and pauses there holding that texture; run
/// `meanwhile` with it, which releases the program's handles and makes the next resource, then let
/// the thread go: what `meanwhile` answered. A making that waits for passing holds answers once
/// the pause ended, at most [`HOLDING`] later.
fn while_a_batch_holds<T, R>(
    device: &Device,
    make: impl FnOnce() -> T,
    meanwhile: impl FnOnce(T) -> R,
) -> R {
    let hold = heap_sets::hold(device);
    let made = make();
    let (entered, paused) = mpsc::channel();
    let (go, resume) = mpsc::channel::<()>();
    let resume = Mutex::new(resume);
    lock(&device.shared().heaps).pause_holding = Some(Arc::new(move || {
        if std::thread::current().name() != Some(BATCH) {
            return;
        }
        entered.send(()).ok();
        let _ = lock(&resume).recv_timeout(HOLDING);
    }));
    // The batch's thread ends the hold, writes the slot and pauses, holding the texture.
    let batch = std::thread::Builder::new()
        .name(BATCH.to_owned())
        .spawn(move || drop(hold))
        .unwrap();
    paused.recv().unwrap();
    let answer = meanwhile(made);
    go.send(()).ok();
    batch.join().unwrap();
    lock(&device.shared().heaps).pause_holding = None;
    answer
}

#[test]
fn a_texture_released_while_a_batch_writes_its_slot_gives_its_id_back_first() {
    let scratch = Scratch::new("passing-holds");
    let device = device(&scratch.0);
    let id = while_a_batch_holds(
        &device,
        || {
            let first = texture(&device);
            assert_eq!(first.gpu_resource_id().0, 1, "the device's first texture");
            first
        },
        |first| {
            // The program releases its texture meanwhile and makes the next.
            drop(first);
            texture(&device).gpu_resource_id().0
        },
    );
    assert_eq!(
        id, 1,
        "the next texture takes the ID the released one gave back, as Metal's next texture does"
    );
}

#[test]
fn a_buffer_pinned_where_a_released_buffer_lay_is_made_while_a_batch_holds_a_texture_over_it() {
    let scratch = Scratch::new("passing-holds-pin");
    let device = device(&scratch.0);
    let (_, length) = rows(&device);
    let pinned = || {
        let contents = BufferContents::Zero;
        let descriptor = BufferDescriptor::new(length, ResourceOptions::shared(), contents);
        device.new_buffer_with_descriptor(descriptor.with_placement(AddressPlacement::At(PIN)))
    };
    let again = while_a_batch_holds(
        &device,
        || {
            let buffer = pinned().unwrap();
            let texture = over(&buffer);
            (buffer, texture)
        },
        |(buffer, texture)| {
            drop((texture, buffer));
            pinned()
        },
    );
    let again = again.unwrap_or_else(|error| {
        panic!("the pin where the released buffer lay, as Metal makes it: {error:?}")
    });
    assert_eq!(again.gpu_address(), PIN);
}

#[test]
fn a_pool_slot_a_released_buffer_held_is_made_while_a_batch_holds_a_texture_over_it() {
    let scratch = Scratch::new("passing-holds-slot");
    let device = device(&scratch.0);
    let (_, length) = rows(&device);
    let pool = device
        .reserve_address_pool(AddressPlacement::Any, 0x20000)
        .unwrap();
    let slot = || {
        let contents = BufferContents::Zero;
        let descriptor = BufferDescriptor::new(length, ResourceOptions::shared(), contents);
        device.new_buffer_with_descriptor(descriptor.with_pool_slot(&pool, 0))
    };
    let again = while_a_batch_holds(
        &device,
        || {
            let buffer = slot().unwrap();
            let texture = over(&buffer);
            (buffer, texture)
        },
        |(buffer, texture)| {
            drop((texture, buffer));
            slot()
        },
    );
    let again = again.unwrap_or_else(|error| {
        panic!("the slot the released buffer held, as Metal makes it: {error:?}")
    });
    assert_eq!(again.gpu_address(), pool.gpu_address());
}

#[test]
fn an_automatic_heap_s_range_a_released_texture_held_is_taken_while_a_batch_holds_it() {
    let scratch = Scratch::new("passing-holds-heap");
    let device = device(&scratch.0);
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 128, 128, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let size = device
        .heap_texture_size_and_align(&descriptor)
        .unwrap()
        .size;
    let heap_descriptor =
        HeapDescriptor::new(HeapType::Automatic, size, ResourceOptions::private());
    let heap = device.new_heap(&heap_descriptor).unwrap();
    let again = while_a_batch_holds(
        &device,
        || {
            let texture = heap.new_texture(&descriptor).unwrap();
            assert!(
                heap.max_available_size(1).unwrap() < size,
                "the texture fills the heap: no second fits beside it"
            );
            texture
        },
        |texture| {
            drop(texture);
            heap.new_texture(&descriptor)
        },
    );
    match again {
        Ok(_) => {}
        Err(Error::OutOfMemory(MemoryKind::Heap)) => {
            panic!("the heap's range the released texture held is still held")
        }
        Err(error) => panic!("the texture again: {error:?}"),
    }
}

#[test]
fn a_texture_a_passing_hold_lets_go_of_last_runs_its_buffer_s_deallocator_once_the_hold_ended() {
    // The pages, declared before the device that uses them, so dropped after it.
    let host = Host::new();
    let scratch = Scratch::new("passing-holds-after");
    // Every texture over host pages a shadow: no move in place takes the texture meanwhile.
    let device = shadow_device(&scratch.0);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(buffer) = over_host(&device, &host, &runs) else {
        return;
    };
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 64, 64, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = buffer.resource_options();
    let over = buffer.new_texture(&descriptor, 0, 256).unwrap();
    // The texture holds the buffer, whose deallocator runs as the texture goes.
    drop(buffer);
    let id = over.gpu_resource_id().0;
    let hold = device.passing_hold();
    // The hold's handle goes inside it: the texture's last.
    drop(over);
    let inside = (
        texture(&device).gpu_resource_id().0,
        runs.load(Ordering::SeqCst),
    );
    drop(hold);
    assert_eq!(
        inside,
        (id, 0),
        "inside the hold: the texture made next takes the released texture's ID, and its \
         buffer's deallocator has not run"
    );
    assert_eq!(
        runs.load(Ordering::SeqCst),
        1,
        "the deallocator ran once the hold ended"
    );
}
