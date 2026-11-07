//! A program's texture released while a runtime thread holds it for a moment gives its ID back
//! before the program's next texture takes one (`resource_ids.rs`, **Passing holds**): a batch
//! holds the slots' writes (`heap_sets::hold`, as the in-place check's submissions do on
//! lavapipe), so the slot of the texture made meanwhile is written as the hold ends, on the
//! batch's thread, which takes the texture to write it. The test pauses that thread there
//! (`DeviceHeaps::pause_holding`), holding the texture, while the program releases it and makes
//! the next: without the wait for passing holds that texture took ID 2, the released one's ID
//! still held by the batch's handle, where Metal's next texture takes 1
//! (`tests::in_place_check::texture_ids_keep_metals_order_while_the_in_place_check_runs` failed
//! so on lavapipe, in two runs of the runtime's suite, its textures made while the check's
//! submissions held the writes).
//!
//! A hold that lets go of a texture's last handle gives its ID back inside the hold and runs none
//! of the program's code there (`PassingHold::let_go`): a texture over a no-copy buffer, the
//! buffer released, whose last handle a passing hold lets go of, gives its ID to the texture made
//! next inside the hold, and its buffer's deallocator runs once the hold ended, not inside it.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, over_host, shadow_device, Host, Scratch};
use crate::device::lock;
use crate::heap_sets;
use crate::{Device, ResourceOptions, Texture, TextureDescriptor};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long the batch's thread holds the released texture at most: until the program's next
/// texture is made, where that texture's ID does not wait for it, else this long.
const HOLDING: Duration = Duration::from_millis(500);

/// A private 4x4 `RGBA8Unorm` texture of the program's, at a plain ID.
fn texture(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

#[test]
fn a_texture_released_while_a_batch_writes_its_slot_gives_its_id_back_first() {
    let scratch = Scratch::new("passing-holds");
    let device = device(&scratch.0);
    // A batch holds the slots' writes: the slot of the texture made now is written as it ends.
    let hold = heap_sets::hold(&device);
    let first = texture(&device);
    assert_eq!(first.gpu_resource_id().0, 1, "the device's first texture");
    let (entered, paused) = mpsc::channel();
    let (go, resume) = mpsc::channel::<()>();
    let resume = Mutex::new(resume);
    lock(&device.shared().heaps).pause_holding = Some(Arc::new(move || {
        entered.send(()).ok();
        let _ = lock(&resume).recv_timeout(HOLDING);
    }));
    // The batch's thread ends the hold, writes the slot and pauses, holding the texture.
    let batch = std::thread::spawn(move || drop(hold));
    paused.recv().unwrap();
    // The program releases its texture meanwhile and makes the next.
    drop(first);
    let next = texture(&device);
    let id = next.gpu_resource_id().0;
    go.send(()).ok();
    batch.join().unwrap();
    lock(&device.shared().heaps).pause_holding = None;
    assert_eq!(
        id, 1,
        "the next texture takes the ID the released one gave back, as Metal's next texture does"
    );
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
    hold.let_go([over]);
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
