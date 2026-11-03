//! A command buffer's merges lend their scratch to its later encoders' merges
//! (`merge_scratch.rs`): one command buffer of hundreds of passes over a render target over a
//! buffer's bytes, each loading and storing it beside a cleared and stored `Depth32Float` depth
//! attachment and blending one rectangle over it, as a WebGPU frame split at every indexed draw
//! renders. The targets: over a `Shared` buffer, rows packed; over a `Private` buffer, packed rows
//! of a multiple of 256 bytes (a `-T` stencilprobe's, rounded up to 256); over a `Shared` buffer
//! of packed rows of no multiple of 256 bytes (an IOSurface's, `4 * width`); and over a no-copy
//! buffer's host pages, rows likewise.
//!
//! With the device's rule forced (`DeviceShared::linear_rule_forced`), each pass renders a target
//! that is its own linear image apart, through the image it keeps, its changed words merged back
//! (`linear_attachment.rs`, `linear_merge.rs`), as NVIDIA's devices render it beside a depth
//! attachment: the merges make one scratch for the whole command buffer, every later one taking
//! it, and the runtime's device memory grows by no scratch a pass. A target that is a shadow
//! (host pages on every device, and rows the device's linear images do not lay out) is merged
//! back at each encoder's end (`shadow_merge.rs`): its merges after the first take the scratch an
//! earlier encoder was done with. Either way the texels equal those the same passes leave in a
//! private texture of the same first bytes.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::{
    BlitOption, Buffer, ClearColor, CommandQueue, Device, Error, HostMemory, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{BlendFactor, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;

/// The targets' height and texel bytes (`RGBA8Unorm`).
const H: u64 = 96;
const TEXEL: u64 = 4;

/// The passes of the command buffer.
const PASSES: usize = 240;

/// The state that blends a half-transparent rectangle (`tob_rect`, corners from vertex buffer 0)
/// of the color in fragment buffer 0 (`tob_solid`) over an `RGBA8Unorm` color, beside a
/// `Depth32Float` depth attachment it does not test.
fn blending(device: &Device) -> RenderPipelineState {
    let library = library(device, "engine/tob_renders.ll");
    let function = |name: &str| {
        let function = library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(function("tob_solid"));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    color.blending_enabled = true;
    color.source_rgb_blend_factor = BlendFactor::SourceAlpha;
    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    color.source_alpha_blend_factor = BlendFactor::One;
    color.destination_alpha_blend_factor = BlendFactor::Zero;
    descriptor.color_attachments[0] = Some(color);
    descriptor.depth_format = Some(PixelFormat::Depth32Float);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// What a target's bytes hold first, rows `row` bytes apart: a counting pattern no pass draws.
/// Every target's rows are packed, with no bytes between them.
fn first_bytes(row: u64) -> Vec<u8> {
    (0..row * H).map(|i| (i * 7 % 251) as u8).collect()
}

/// A `Shared` buffer holding `bytes`.
fn staged(device: &Device, bytes: &[u8]) -> Buffer {
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    write(&buffer, bytes);
    buffer
}

/// Write `bytes` at the start of `buffer`, host-visible, which no command uses now.
fn write(buffer: &Buffer, bytes: &[u8]) {
    let contents = buffer.contents().unwrap();
    // SAFETY: inside the buffer's length, which no command uses now.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
}

/// The first `length` bytes of `buffer`, host-visible, which no command uses now.
fn read(buffer: &Buffer, length: u64) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: inside the buffer's length, which no command writes now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) }.to_vec()
}

/// Run one blit encoder's commands, `encode` encoding them, in a command buffer that completes.
fn blit(queue: &CommandQueue, encode: impl FnOnce(&mut crate::BlitCommandEncoder)) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// Page-aligned host memory of `length` bytes (whole pages of any host), freed when dropped:
/// declared before the device's buffer over it, so dropped after it.
struct Host(NonNull<u8>, Layout);

impl Host {
    fn new(length: u64) -> Host {
        let layout =
            Layout::from_size_align(length.next_multiple_of(64 << 10) as usize, 64 << 10).unwrap();
        // SAFETY: the layout's size is not zero.
        let pointer = unsafe { std::alloc::alloc_zeroed(layout) };
        Host(NonNull::new(pointer).unwrap(), layout)
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `new`.
        unsafe { std::alloc::dealloc(self.0.as_ptr(), self.1) };
    }
}

/// Where a target's bytes lie.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Over {
    /// A `Shared` buffer.
    Shared,
    /// A `Private` buffer: written and read through blits.
    Private,
    /// A no-copy buffer over host pages.
    HostPages,
}

/// A target of `width` texels a row, `row` bytes apart, over a buffer of `over`'s holding
/// [`first_bytes`]; `None`, printed, where the device imports no host memory.
fn target(
    queue: &CommandQueue,
    over: Over,
    (width, row): (u64, u64),
    host: &Host,
) -> Option<(Buffer, Texture)> {
    let device = queue.device();
    let first = first_bytes(row);
    let length = row * H;
    let buffer = match over {
        Over::Shared => device
            .new_buffer(length, ResourceOptions::shared())
            .unwrap(),
        Over::Private => {
            let buffer = device
                .new_buffer(length, ResourceOptions::private())
                .unwrap();
            let source = staged(device, &first);
            blit(queue, |b| {
                (b.copy_from_buffer_to_buffer(&source, 0, &buffer, 0, length)).unwrap()
            });
            buffer
        }
        Over::HostPages => {
            // SAFETY: the memory outlives the buffer, and only its commands use it meanwhile.
            let made = unsafe {
                device.new_buffer_with_bytes_no_copy(
                    HostMemory::new(host.0),
                    host.1.size() as u64,
                    ResourceOptions::default(),
                    None,
                )
            };
            match made {
                Ok(buffer) => buffer,
                Err(Error::Unsupported(gap)) => {
                    eprintln!("MERGE-SCRATCH: the device imports no host memory: {gap:?}");
                    return None;
                }
                Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
            }
        }
    };
    if over != Over::Private {
        write(&buffer, &first);
    }
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, width, H, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = buffer.resource_options();
    let texture = buffer.new_texture(&descriptor, 0, row).unwrap();
    Some((buffer, texture))
}

/// The bytes of `buffer` (`length` of them), read through a blit where it is `Private`.
fn bytes(queue: &CommandQueue, over: Over, buffer: &Buffer, length: u64) -> Vec<u8> {
    if over != Over::Private {
        return read(buffer, length);
    }
    let out = (queue.device())
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    blit(queue, |b| {
        (b.copy_from_buffer_to_buffer(buffer, 0, &out, 0, length)).unwrap()
    });
    read(&out, length)
}

/// Pass `i`'s rectangle, in normalized device coordinates (`[x0, y0, x1, y1]`), and its color:
/// rectangles over a grid of places, each overlapping its neighbours, half transparent.
fn drawn(i: usize) -> ([f32; 4], [f32; 4]) {
    let (x, y) = ((i % 12) as f32, ((i / 12) % 6) as f32);
    let x0 = -1.0 + 0.125 * x;
    let y0 = -1.0 + 0.25 * y;
    let corners = [x0, y0, (x0 + 0.5).min(1.0), (y0 + 0.75).min(1.0)];
    let color = [
        (i % 5) as f32 / 4.0,
        (i % 7) as f32 / 6.0,
        (i % 3) as f32 / 2.0,
        0.5,
    ];
    (corners, color)
}

/// One command buffer of [`PASSES`] passes over `texture`, each loading and storing it beside a
/// `Depth32Float` depth attachment it clears and stores, blending its rectangle ([`drawn`]);
/// committed and completed. The device's work counts after the passes were encoded, before the
/// commit.
fn passes(queue: &CommandQueue, state: &RenderPipelineState, texture: &Texture) -> WorkCounts {
    let device = queue.device();
    let (width, height) = (texture.width(), texture.height());
    let mut d = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, width, height, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    let depth = device.new_texture(&d).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    for i in 0..PASSES {
        let mut descriptor = RenderPassDescriptor::new();
        let color = &mut descriptor.color_attachments[0];
        color.texture = Some(texture.clone());
        color.load_action = LoadAction::Load;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        descriptor.depth_attachment.texture = Some(depth.clone());
        descriptor.depth_attachment.load_action = LoadAction::Clear;
        descriptor.depth_attachment.store_action = StoreAction::Store;
        let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
        encoder.set_render_pipeline_state(state).unwrap();
        let (corners, rgba) = drawn(i);
        encoder
            .set_vertex_bytes(&bytes_of(&corners, f32::to_le_bytes), 0)
            .unwrap();
        encoder
            .set_fragment_bytes(&bytes_of(&rgba, f32::to_le_bytes), 0)
            .unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    let encoded = device.work_counts();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    encoded
}

/// The texels [`passes`] leave in a private texture of `width` texels a row holding
/// [`first_bytes`]'s texels first, packed.
fn reference(queue: &CommandQueue, state: &RenderPipelineState, width: u64) -> Vec<u8> {
    let device = queue.device();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, width, H, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let texture = device.new_texture(&d).unwrap();
    let packed = width * TEXEL;
    let source = staged(device, &first_bytes(packed));
    let size = Size::new(width, H, 1);
    blit(queue, |b| {
        let origin = Origin::new(0, 0, 0);
        b.copy_from_buffer_to_texture(
            &source,
            0,
            packed,
            packed * H,
            size,
            &texture,
            0,
            0,
            origin,
            BlitOption::None,
        )
        .unwrap()
    });
    passes(queue, state, &texture);
    let out = device
        .new_buffer(packed * H, ResourceOptions::shared())
        .unwrap();
    blit(queue, |b| {
        let origin = Origin::new(0, 0, 0);
        b.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            origin,
            size,
            &out,
            0,
            packed,
            packed * H,
            BlitOption::None,
        )
        .unwrap()
    });
    read(&out, packed * H)
}

/// [`passes`] over a target over `over`'s buffer of `width` texels a row, `row` bytes apart, the
/// rule forced: its texels equal the private texture's, and its merges (where it is a linear
/// image, or a shadow merged back) take one scratch, lent to every later one, the device's memory
/// growing by less than an eighth of what a scratch a pass would take.
fn one_scratch(over: Over, (width, row): (u64, u64)) {
    let cache = Scratch::new(&format!("merge-scratch-{over:?}-{width}"));
    let host = Host::new(row * H);
    let device = device(&cache.0);
    let queue = device.new_command_queue().unwrap();
    let state = blending(&device);
    let expected = reference(&queue, &state, width);
    let Some((buffer, texture)) = target(&queue, over, (width, row), &host) else {
        return;
    };
    let linear = texture.linear_image().unwrap();
    let shadow = texture.shadow().is_some();
    // A shadow merged back at each encoder's end: one over bytes another writer may write (all
    // but a tracked `Private` buffer's, whose shadow is copied back).
    let merged = texture.shadow().is_some_and(|copies| copies.merges_back());
    println!(
        "MERGE-SCRATCH {over:?} {width}x{H}, rows of {row}: linear image {linear}, shadow {shadow} \
         (merged back {merged})"
    );
    let forced = &device.shared().linear_rule_forced;
    forced.store(true, Ordering::Relaxed);
    let before = device.work_counts();
    let encoded = passes(&queue, &state, &texture);
    let after = device.work_counts();
    forced.store(false, Ordering::Relaxed);
    let delta = |of: fn(&WorkCounts) -> u64| of(&after) - of(&before);
    let apart = delta(|c| c.linear_passes_through_copies);
    let made = delta(|c| c.linear_merge_scratches_made);
    let reused = delta(|c| c.linear_merge_scratches_reused);
    let shadow_reused = delta(|c| c.shadow_merge_scratches_reused);
    let grown = encoded.device_memory_allocated - before.device_memory_allocated;
    // A merge of every texel's rows: its kept and changed parts.
    let scratch = 2 * width * TEXEL * H;
    println!(
        "MERGE-SCRATCH {over:?}: {apart} passes apart, linear scratches {made} made and {reused} \
         reused, shadow scratches {shadow_reused} reused; device memory grew {grown} bytes while \
         encoding ({} merges' scratch)",
        grown / scratch
    );
    match (linear, merged) {
        (true, _) => {
            assert_eq!(apart, PASSES as u64, "every pass renders the texture apart");
            assert_eq!(made, 1, "the command buffer's merges make one scratch");
            assert_eq!(reused, PASSES as u64 - 1, "every later merge takes it");
        }
        (false, true) => assert!(
            shadow_reused >= PASSES as u64 - 1,
            "every shadow merge after the first takes a scratch: {shadow_reused}"
        ),
        (false, false) => println!("MERGE-SCRATCH {over:?}: no merge"),
    }
    assert!(
        grown < scratch * PASSES as u64 / 8,
        "encoding the passes grew the device's memory by {grown} bytes, {} merges' scratch",
        grown / scratch
    );
    let got = bytes(&queue, over, &buffer, row * H);
    let differ = (0..got.len()).filter(|&i| got[i] != expected[i]).count();
    assert_eq!(differ, 0, "the texels differ from the private texture's");
    drop((texture, buffer));
}

/// Over a `Shared` buffer, rows packed (1024 bytes).
#[test]
fn many_passes_over_a_shared_buffer_take_one_scratch() {
    one_scratch(Over::Shared, (256, 1024));
}

/// Over a `Private` buffer, packed rows of 1024 bytes, a multiple of 256 (a `-T` stencilprobe's
/// rows rounded up to 256 bytes, with no bytes between them).
#[test]
fn many_passes_over_a_private_buffer_take_one_scratch() {
    one_scratch(Over::Private, (256, 1024));
}

/// Over a `Shared` buffer of packed rows of 300 texels, 1200 bytes, of no multiple of 256 (an
/// IOSurface's).
#[test]
fn many_passes_over_packed_rows_take_one_scratch() {
    one_scratch(Over::Shared, (300, 1200));
}

/// Over a no-copy buffer's host pages, packed rows of 1200 bytes: a shadow on every device.
#[test]
fn many_passes_over_host_pages_take_one_scratch() {
    one_scratch(Over::HostPages, (300, 1200));
}
