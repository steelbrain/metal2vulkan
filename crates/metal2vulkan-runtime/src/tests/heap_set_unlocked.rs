//! A new heap set's layout and pool are made with no lock held (`heap_sets.rs`,
//! `SetEntry::unplaced`): a driver may clear every element of the pool as it makes it (lavapipe
//! takes milliseconds over arrays of 65536 elements), and a texture made on another thread
//! meanwhile takes the `heaps` lock to place itself. The test pauses the making thread once it made the set's layout and pool
//! (`DeviceHeaps::pause_making`), makes a texture meanwhile, which is made at once where it waited
//! for the pause to end, then lets the making go: the set, placed after, holds the texture, which
//! the dispatch that made the set reads through its resource ID.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `handles.ll`'s
//! `argument_buffer_read` (a sampled `texture2d<float>` read through an argument buffer).

use super::blit_holds::{device, Scratch, WAIT};
use crate::device::lock;
use crate::{
    Buffer, CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, Region,
    ResourceOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The texture's width, in texels, and the dispatch's threads.
const SIDE: u64 = 8;

/// How long the making thread pauses at most: until the texture was made, where its making does
/// not wait for the pause, else this long.
const HOLDING: Duration = Duration::from_secs(4);

/// The compute state of `entry` of the owned engine fixture `stem` on `device`.
fn state(device: &Device, stem: &str, entry: &str) -> ComputePipelineState {
    let ll = metal2vulkan_devtools::engine_dir().join(format!("{stem}.ll"));
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name(entry)
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A shared buffer of `length` zero bytes.
fn zeroed(device: &Device, length: u64) -> Buffer {
    let buffer = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let contents = buffer.contents().unwrap();
    // SAFETY: the new buffer's bytes, which no command uses yet.
    unsafe { std::ptr::write_bytes(contents.as_ptr(), 0, length as usize) };
    buffer
}

/// Texel `x` of the texture.
fn texel(x: u64) -> [f32; 4] {
    [x as f32 + 1.0, 2.0 * x as f32 + 1.0, 0.25, 1.0]
}

#[test]
fn a_texture_made_while_a_new_set_s_pool_is_made_is_made_at_once_and_held_by_the_set() {
    let scratch = Scratch::new("heap-set-unlocked");
    let device = device(&scratch.0);
    let read = state(&device, "handles", "argument_buffer_read");
    let argument = zeroed(&device, 8);
    let out = zeroed(&device, 16 * SIDE);
    let (entered, paused) = mpsc::channel();
    let (go, resume) = mpsc::channel::<()>();
    let resume = Mutex::new(resume);
    // The test makes one set, whichever thread its command's binding runs on.
    lock(&device.shared().heaps).pause_making = Some(Arc::new(move || {
        entered.send(()).ok();
        let _ = lock(&resume).recv_timeout(HOLDING);
    }));
    // The making thread encodes the first command of the layout, which makes its set, and pauses
    // once it made the set's layout and pool; it commits once the pause ended.
    let making = {
        let (device, argument, out) = (device.clone(), argument.clone(), out.clone());
        std::thread::spawn(move || {
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&read).unwrap();
            encoder.set_buffer(Some(&argument), 0, 0).unwrap();
            encoder.set_buffer(Some(&out), 0, 1).unwrap();
            encoder
                .dispatch_threads(Size::new(SIDE, 1, 1), Size::new(SIDE, 1, 1))
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            assert_eq!(command_buffer.error(), None);
        })
    };
    paused
        .recv_timeout(WAIT)
        .expect("the dispatch made the layout's set, pausing once its layout and pool were made");
    // Made while the making thread holds the new set's layout and pool, unplaced.
    let began = Instant::now();
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, SIDE, 1, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture: Texture = device.new_texture(&descriptor).unwrap();
    let took = began.elapsed();
    let texels: Vec<u8> = (0..SIDE)
        .flat_map(texel)
        .flat_map(f32::to_le_bytes)
        .collect();
    let whole = Region::new_2d(0, 0, SIDE, 1);
    texture
        .replace_region(whole, 0, 0, &texels, 16 * SIDE, 0)
        .unwrap();
    let id = texture.gpu_resource_id().0.to_le_bytes();
    let contents = argument.contents().unwrap();
    // SAFETY: the argument buffer's 8 bytes, which no committed command reads yet.
    unsafe { std::ptr::copy_nonoverlapping(id.as_ptr(), contents.as_ptr(), id.len()) };
    go.send(()).ok();
    making.join().unwrap();
    lock(&device.shared().heaps).pause_making = None;
    assert!(
        took < HOLDING / 2,
        "the texture was made while the new set's pool was made, in {took:?}, where it waited for \
         the making"
    );
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's bytes, which the completed command buffer wrote.
    let read = unsafe { std::slice::from_raw_parts(contents.as_ptr(), texels.len()) };
    assert_eq!(
        read, texels,
        "the set, placed after the texture was made, holds it at its ID"
    );
}
