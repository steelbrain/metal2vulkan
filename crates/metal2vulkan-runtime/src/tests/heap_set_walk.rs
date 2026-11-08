//! A new heap set walks the live textures for its arrays of a class no live set holds with no lock
//! held, in parts each inside a passing hold of its own (`heap_sets.rs`, `Walk`): a driver may take
//! milliseconds over thousands of textures (the views made at their first use, the descriptor
//! writes), while a texture going on another thread meanwhile takes the `heaps` lock to clear its
//! slot, and a texture made there waits for every passing hold and takes the lock to place itself.
//! The test makes two textures, then pauses the making thread once it walked them
//! (`DeviceHeaps::pause_walking`). Meanwhile it releases the first, whose element the walk took,
//! and makes a texture, which takes the first's resource ID back: both are done at once where they
//! waited for the walk. Then it lets the walk go: the set, placed after, holds the texture made
//! meanwhile at that ID, not the walk's view of the released one, which the dispatch that made the
//! set reads through the ID; and it holds the second texture as the walk wrote it, which a second
//! dispatch reads.
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

/// The textures' width, in texels, and the dispatch's threads.
const SIDE: u64 = 8;

/// How long the making thread pauses at most: until the texture was made, where its making does
/// not wait for the walk, else this long.
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

/// A new texture the kernel reads.
fn made(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, SIDE, 1, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    device.new_texture(&descriptor).unwrap()
}

/// Fill `texture` with texels that start at `from`: their bytes.
fn filled(texture: &Texture, from: f32) -> Vec<u8> {
    let texels: Vec<u8> = (0..SIDE)
        .flat_map(|x| [x as f32 + from, 2.0 * x as f32 + from, 0.25, 1.0])
        .flat_map(f32::to_le_bytes)
        .collect();
    let whole = Region::new_2d(0, 0, SIDE, 1);
    texture
        .replace_region(whole, 0, 0, &texels, 16 * SIDE, 0)
        .unwrap();
    texels
}

/// Name `texture` in `argument`, which the kernel reads its resource ID from.
fn name(argument: &Buffer, texture: &Texture) {
    let id = texture.gpu_resource_id().0.to_le_bytes();
    let contents = argument.contents().unwrap();
    // SAFETY: the argument buffer's 8 bytes, which no committed command reads yet.
    unsafe { std::ptr::copy_nonoverlapping(id.as_ptr(), contents.as_ptr(), id.len()) };
}

/// Run `read` once on `device`: it reads the texture `argument` names into `out`.
fn dispatch(device: &Device, read: &ComputePipelineState, argument: &Buffer, out: &Buffer) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(read).unwrap();
    encoder.set_buffer(Some(argument), 0, 0).unwrap();
    encoder.set_buffer(Some(out), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(SIDE, 1, 1), Size::new(SIDE, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
}

/// The bytes `out` holds, as many as `like`'s.
fn read_back(out: &Buffer, like: &[u8]) -> Vec<u8> {
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's bytes, which the completed command buffer wrote.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), like.len()) }.to_vec()
}

#[test]
fn a_texture_made_while_a_new_set_walks_the_live_textures_is_made_at_once_and_held_by_the_set() {
    let scratch = Scratch::new("heap-set-walk");
    let device = device(&scratch.0);
    let read = state(&device, "handles", "argument_buffer_read");
    let argument = zeroed(&device, 8);
    let out = zeroed(&device, 16 * SIDE);
    let released = made(&device);
    filled(&released, 1.0);
    let kept = made(&device);
    let kept_texels = filled(&kept, 100.0);
    let released_id = released.gpu_resource_id().0;
    let (entered, paused) = mpsc::channel();
    let (go, resume) = mpsc::channel::<()>();
    let resume = Mutex::new(resume);
    // The test makes one set, whose walk of the two textures is one part: it pauses once.
    lock(&device.shared().heaps).pause_walking = Some(Arc::new(move || {
        entered.send(()).ok();
        let _ = lock(&resume).recv_timeout(HOLDING);
    }));
    // The making thread encodes the first command of the layout, which makes its set, and pauses
    // once its walk took the two textures; it commits once the pause ended.
    let making = {
        let (device, read, argument, out) =
            (device.clone(), read.clone(), argument.clone(), out.clone());
        std::thread::spawn(move || dispatch(&device, &read, &argument, &out))
    };
    paused
        .recv_timeout(WAIT)
        .expect("the dispatch made the layout's set, pausing once its walk took the textures");
    // Released and made while the making thread holds the walk's elements, unplaced.
    let began = Instant::now();
    drop(released);
    let meanwhile = made(&device);
    let took = began.elapsed();
    let texels = filled(&meanwhile, 50.0);
    name(&argument, &meanwhile);
    go.send(()).ok();
    making.join().unwrap();
    lock(&device.shared().heaps).pause_walking = None;
    assert!(
        took < HOLDING / 2,
        "a texture went and one was made while the new set walked the live textures, in \
         {took:?}, where they waited for the walk"
    );
    assert_eq!(
        meanwhile.gpu_resource_id().0,
        released_id,
        "the texture made meanwhile took the released one's resource ID back"
    );
    assert_eq!(
        read_back(&out, &texels),
        texels,
        "the set, placed after the texture was made, holds it at the released one's ID"
    );
    name(&argument, &kept);
    dispatch(&device, &read, &argument, &out);
    assert_eq!(
        read_back(&out, &kept_texels),
        kept_texels,
        "the set holds the texture the walk took as it wrote it"
    );
}
