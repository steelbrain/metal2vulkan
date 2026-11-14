//! A compute encoder's binds take the views, sampler descriptions and host pages its device's
//! compute encoders' binds before it kept (`binding::BindMemo`, kept with the encoder's bind state:
//! `compute/live.rs`), and work each out again once what it is a function of moved, with the
//! answer a fresh bind gives:
//!
//! - **The views epoch** (`DeviceShared::views_epoch`, which an image letting its views go moves,
//!   `Image::views_go`): moved here as such an image moves it, between two encoders of one command
//!   buffer, the next bind works the texture's view out again (`WorkCounts::bind_views_worked_out`)
//!   and nothing else.
//! - **The sampler state**: a dispatch of a linear `SamplerState` after a nearest one, on a 2 by 1
//!   `R32Float` texture of 0 and 1 at u = 0.5, works its description out
//!   (`bind_samplers_worked_out`) and reads 0.5; the nearest one again, kept, works none out and
//!   reads one texel.
//! - **A no-copy buffer's move**: the coordinates in a no-copy buffer, rebound to other pages
//!   between two command buffers, are read again where they lie (`bind_spaces_read`), and the
//!   dispatch reads the new pages' coordinates.
//!
//! Every dispatch binds one texture, one sampler state and, in the last, one buffer over host
//! pages: a bind that keeps nothing counts one of each, a steady one none.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `fpv_swz_sample` of the
//! owned fixture `validation/fixtures/public/engine/fpv_swizzled_reads.ll` (`llvm-as`):
//! `o[i] = t.sample(s, p[i])`, its texture and sampler at 0, `o` at buffer 0 and `p` at 1.

use super::blit_holds::{device, over_host, Host, Scratch, WAIT};
use super::encoder_holds::library;
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, ComputePipelineState, Device, PipelineOptions,
    Region, ResourceOptions, SamplerState, Size, Texture, TextureDescriptor,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, SamplerMinMagFilter, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// The bytes apart the dispatches' outputs are written.
const APART: u64 = 256;

/// What one dispatch binds besides its output.
struct Bound<'a> {
    state: &'a ComputePipelineState,
    texture: &'a Texture,
    sampler: &'a SamplerState,
    coords: &'a Buffer,
}

/// The counts a bind that keeps nothing moves: the views, the sampler descriptions and the reads
/// of where a buffer over host pages lies its binds worked out.
fn counts(device: &Device) -> [u64; 3] {
    let counts = device.work_counts();
    [
        counts.bind_views_worked_out,
        counts.bind_samplers_worked_out,
        counts.bind_spaces_read,
    ]
}

/// The state of `fpv_swz_sample`.
fn sampling(device: &Device) -> ComputePipelineState {
    let function = library(device, "engine/fpv_swizzled_reads.ll")
        .new_function_with_name("fpv_swz_sample")
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A 2 by 1 `R32Float` texture of 0 and 1.
fn texels(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::R32Float, 2, 1, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes: Vec<u8> = [0.0f32, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    (texture.replace_region(Region::new_2d(0, 0, 2, 1), 0, 0, &bytes, 8, 0)).unwrap();
    texture
}

/// A sampler state of `filter` for magnification and minification, clamped to the edge.
fn sampler(device: &Device, filter: SamplerMinMagFilter) -> SamplerState {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.min_filter = filter;
    descriptor.mag_filter = filter;
    device.new_sampler_state(&descriptor).unwrap()
}

/// The words of the coordinates `(u, 0.5)`.
fn coords_words(u: f32) -> [u32; 2] {
    [u.to_bits(), 0.5f32.to_bits()]
}

/// A `Shared` buffer of the coordinates `(u, 0.5)`.
fn coords(device: &Device, u: f32) -> Buffer {
    let bytes: Vec<u8> = coords_words(u)
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    let contents = buffer.contents().unwrap();
    // SAFETY: the new buffer's bytes, which no command uses yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    buffer
}

/// A `Shared` buffer of `dispatches` outputs, [`APART`] bytes apart.
fn outputs(device: &Device, dispatches: u64) -> Buffer {
    (device.new_buffer(dispatches * APART, ResourceOptions::shared())).unwrap()
}

/// The red of the `at`th output of `out`.
fn red(out: &Buffer, at: u64) -> f32 {
    let contents = out.contents().unwrap();
    // SAFETY: inside the buffer, which no command uses any more.
    let bytes =
        unsafe { std::slice::from_raw_parts(contents.as_ptr().add((at * APART) as usize), 4) };
    f32::from_le_bytes(bytes.try_into().unwrap())
}

/// Encode, in a compute encoder of its own in `command_buffer`, one thread of `bound`'s dispatch
/// writing `out`'s `at`th output: the counts its encoding moved.
fn dispatch(
    command_buffer: &CommandBuffer,
    bound: &Bound<'_>,
    (out, at): (&Buffer, u64),
) -> [u64; 3] {
    let device = command_buffer.device();
    let before = counts(device);
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(bound.state).unwrap();
    encoder.set_texture(Some(bound.texture), 0).unwrap();
    encoder.set_sampler_state(Some(bound.sampler), 0).unwrap();
    encoder.set_buffer(Some(out), at * APART, 0).unwrap();
    encoder.set_buffer(Some(bound.coords), 0, 1).unwrap();
    let one = Size::new(1, 1, 1);
    encoder.dispatch_threads(one, one).unwrap();
    encoder.end_encoding().unwrap();
    let after = counts(device);
    [0, 1, 2].map(|at| after[at] - before[at])
}

/// Commit `command_buffer` and wait for it.
fn completed(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// After the views epoch moved, the next dispatch works the texture's view out again, and its
/// sampler description and output stay.
#[test]
fn a_moved_views_epoch_works_the_view_out_again() {
    let scratch = Scratch::new("bind-memo-views");
    let device = device(&scratch.0);
    let (state, texture) = (sampling(&device), texels(&device));
    let nearest = sampler(&device, SamplerMinMagFilter::Nearest);
    let coords = coords(&device, 0.75);
    let bound = Bound {
        state: &state,
        texture: &texture,
        sampler: &nearest,
        coords: &coords,
    };
    let out = outputs(&device, 3);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    assert_eq!(
        dispatch(&command_buffer, &bound, (&out, 0)),
        [1, 1, 0],
        "the first"
    );
    assert_eq!(
        dispatch(&command_buffer, &bound, (&out, 1)),
        [0, 0, 0],
        "a steady one"
    );
    // As an image that lets its views go moves it (`Image::views_go`).
    device.shared().views_epoch.fetch_add(1, Ordering::AcqRel);
    let moved = dispatch(&command_buffer, &bound, (&out, 2));
    assert_eq!(moved, [1, 0, 0], "after the views epoch moved");
    completed(&command_buffer);
    for at in 0..3 {
        assert_eq!(red(&out, at), 1.0, "output {at}: texel 1 at u = 0.75");
    }
}

/// A dispatch of another sampler state works its description out and samples through it; the
/// first state again, kept, works none out.
#[test]
fn another_sampler_state_works_its_description_out() {
    let scratch = Scratch::new("bind-memo-samplers");
    let device = device(&scratch.0);
    let (state, texture) = (sampling(&device), texels(&device));
    let nearest = sampler(&device, SamplerMinMagFilter::Nearest);
    let linear = sampler(&device, SamplerMinMagFilter::Linear);
    let coords = coords(&device, 0.5);
    let bound = |sampler| Bound {
        state: &state,
        texture: &texture,
        sampler,
        coords: &coords,
    };
    let out = outputs(&device, 3);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let first = dispatch(&command_buffer, &bound(&nearest), (&out, 0));
    assert_eq!(first, [1, 1, 0], "the first, nearest");
    let swapped = dispatch(&command_buffer, &bound(&linear), (&out, 1));
    assert_eq!(swapped, [0, 1, 0], "linear after nearest");
    let back = dispatch(&command_buffer, &bound(&nearest), (&out, 2));
    assert_eq!(back, [0, 0, 0], "nearest again");
    completed(&command_buffer);
    let (nearest, linear, again) = (red(&out, 0), red(&out, 1), red(&out, 2));
    assert!(
        nearest == 0.0 || nearest == 1.0,
        "nearest at u = 0.5 reads one texel: {nearest}"
    );
    assert_eq!(linear, 0.5, "linear at u = 0.5: halfway between 0 and 1");
    assert_eq!(again, nearest, "nearest again");
}

/// The coordinates in a no-copy buffer rebound between two command buffers are read where they
/// lie now: the next dispatch reads where the buffer lies again and samples at the new pages'
/// coordinates.
#[test]
fn a_rebound_no_copy_buffer_is_read_where_it_lies_now() {
    // Declared before the device, so dropped after it.
    let (old, new) = (Host::new(), Host::new());
    let scratch = Scratch::new("bind-memo-spaces");
    let device = device(&scratch.0);
    let (state, texture) = (sampling(&device), texels(&device));
    let nearest = sampler(&device, SamplerMinMagFilter::Nearest);
    old.set(0, &coords_words(0.75));
    new.set(0, &coords_words(0.25));
    let runs = Arc::new(AtomicUsize::new(0));
    let coords = over_host(&device, &old, &runs).expect("the device imports host memory");
    let bound = Bound {
        state: &state,
        texture: &texture,
        sampler: &nearest,
        coords: &coords,
    };
    let out = outputs(&device, 3);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    assert_eq!(
        dispatch(&command_buffer, &bound, (&out, 0)),
        [1, 1, 1],
        "the first"
    );
    assert_eq!(
        dispatch(&command_buffer, &bound, (&out, 1)),
        [0, 0, 0],
        "a steady one"
    );
    completed(&command_buffer);
    // SAFETY: `new` is dropped after the device, and only the buffer uses it while it lives.
    unsafe { coords.rebind_bytes_no_copy(new.memory(), None, None) }.unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let moved = dispatch(&command_buffer, &bound, (&out, 2));
    assert_eq!(moved, [0, 0, 1], "after the buffer moved");
    completed(&command_buffer);
    assert_eq!(red(&out, 0), 1.0, "texel 1 at u = 0.75, the old pages'");
    assert_eq!(red(&out, 1), 1.0, "texel 1 at u = 0.75, the old pages'");
    assert_eq!(red(&out, 2), 0.0, "texel 0 at u = 0.25, the new pages'");
}
