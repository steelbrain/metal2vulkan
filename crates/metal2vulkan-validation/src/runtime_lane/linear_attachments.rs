//! Device tests of render passes over linear images on the device the cfg selects, under the
//! validation layer. A render target over a `Shared` buffer of the runtime's memory is its own
//! linear image where the device renders one there ([`surface`]), as on every device that renders
//! one only by the rule below; one over a no-copy buffer's host pages is an image in place where
//! the device's check passed for render targets over such pages, else a shadow
//! (`shadowed-texture`), which the detach, rebind and host-page tests take ([`host_surface`]). On a
//! device that renders a linear image only by `VK_NV_linear_color_attachment` (NVIDIA's, whose
//! linear tiling has the extension's bit and not the color attachment feature: [`by_the_rule`]),
//! the runtime enables the extension and renders it in place, but only where every color attachment
//! of the render pass instance is linear, as the extension rules. A pass beside a private
//! attachment, and a multisampled pass resolving into it, render it through an image of the
//! runtime's, copied in and back, and announce `linear-attachment-copies` ([`LINEAR_COPIES`]) once
//! a device. Elsewhere nothing is announced.
//!
//! - [`a_cleared_render_target_over_a_shared_buffer_renders_through_copies`]: a 256x64
//!   `BGRA8Unorm` render target over a `Shared` buffer, aliased; a pass of it alone that loads it
//!   and draws nothing changes no texel and records nothing, counted neither way; one that clears
//!   it renders it through copies and announces `linear-attachment-clear-copies`, by the rule
//!   (counted so); the buffer holds what each pass stored.
//! - [`a_render_target_over_host_pages_is_a_shadow`]: a 256x64 `BGRA8Unorm` render target over a
//!   no-copy buffer is a shadow, never an image bound to the pages; a pass of it alone that loads
//!   it and one that clears it each leave the pages holding what it stored, and neither announces
//!   a linear attachment's copies.
//! - [`a_linear_attachment_beside_a_private_one_renders_through_copies`]: a pass of that texture
//!   (loaded and stored) beside a private one (cleared), then one clearing the texture and
//!   loading the private one: the buffer keeps its bytes, then holds the clear, the private texture
//!   the other way round; announced exactly where the texture is aliased by the rule.
//! - [`a_resolve_into_a_linear_texture_resolves_through_a_scratch`]: a 4-sample private
//!   attachment cleared and resolved into the texture: the buffer holds the clear; announced
//!   exactly where the texture is aliased by the rule.
//!
//! Where the rule lets a pass render the texture in place, the pass still chooses as it ends
//! (`linear_attachment::choose`): one that loads it renders it in place; one that clears it renders
//! it through copies, copied back, announcing `linear-attachment-clear-copies` ([`CHOSEN_COPIES`]),
//! only where no command of it may read the texture's bytes (no barrier inside it, no draw reaching
//! memory it does not name, no command naming the bytes), else in place. A pass rendered apart
//! (beside a private color, or a depth or stencil attachment) copies the texture in over its draws'
//! box and merges the words it changed back into the bytes (`linear_merge.rs`), merging and copying
//! in again before the draws after a memory barrier inside it. Each pass's pages are compared with
//! the same pass over a private texture of the same bytes: the in-place and the merged forms equal
//! it, and so each other, where nothing else writes the bytes while the pass runs (a race Metal
//! leaves unordered); a pixel a draw rewrites with its old value is written in place and skipped by
//! the merge, the same bytes.
//!
//! - [`blends_over_most_of_a_linear_texture_render_in_place`]: three blended draws, the left half,
//!   the right half and the whole, unscissored: in place.
//! - [`scissored_blends_over_a_corner_render_in_place`]: one blended full-viewport draw through a
//!   64x32 scissor: in place, and the bytes past the scissor keep theirs.
//! - [`a_blend_over_most_of_it_inside_a_box_renders_in_place`]: two blended draws through a
//!   192x48 scissor: in place, and the bytes past the box keep theirs.
//! - [`a_blend_with_no_depth_or_stencil_renders_the_linear_texture_in_place`],
//!   [`a_blend_beside_a_depth_attachment_renders_the_linear_texture_apart`] and
//!   [`a_blend_beside_a_stencil_attachment_renders_the_linear_texture_apart`]: one blended
//!   full-viewport draw, beside nothing, a `Depth32Float` depth attachment or a `Stencil8` stencil
//!   one: beside either, rendered apart and announced exactly where the texture is aliased by the
//!   rule, else in place, unannounced.
//! - [`a_colorless_clip_zeroes_past_the_blends_box`] and
//!   [`a_colorless_clip_beside_a_stencil_zeroes_past_the_blends_box`]: two blended rounded panels
//!   (`zwd_panel`) through the left half's scissor, then a clip of a state of no color
//!   (`zwd_clip`) through the right quarter's, which Metal zeroes the color under: in place, and
//!   apart beside a stencil, merged over a box that holds the zeroing.
//! - [`a_cpu_write_at_an_undrawn_pixel_survives_a_loaded_pass`] and
//!   [`a_cpu_write_at_an_undrawn_pixel_survives_a_pass_apart`]: the CPU writes the surface's last
//!   64 rows, in no tile a draw reaches, while a long loaded pass runs (after a shared event the
//!   command buffer signals before it): Metal keeps them, and so do the pass in place and the
//!   merge.
//! - [`a_pass_sampling_its_surface_after_a_barrier_reads_its_draws`],
//!   [`a_cleared_pass_sampling_its_surface_after_a_barrier_renders_in_place`] and
//!   [`a_pass_apart_sampling_its_surface_after_a_barrier_reads_its_draws`] (loaded and cleared):
//!   a panel, a render-target barrier, then a draw sampling the surface's own texture at its
//!   pixel: the buffer holds the private texture's pass. A cleared one renders in place (a command
//!   of it may read the bytes); one apart is split at the barrier, merged and copied in again.
//!   Where the device renders the texture's own image as any other, the pass alone records its
//!   instances inline (`render/inline_first.rs`, `render/inline_splits.rs`), counted.
//! - [`a_detach_while_a_pass_over_host_pages_is_open_fails_its_command_buffer`] and
//!   [`a_detach_while_a_pass_in_a_no_copy_heap_is_open_fails_its_command_buffer`]: the bytes
//!   detached after the pass's first draw: the command buffer fails `DetachedBacking`, the pages
//!   keep their bytes, and the validation layer reports nothing, where the pass records inline
//!   too (`render/inline_first.rs`; a heap's texture of its own image does, unless the rule binds
//!   it).
//! - [`a_rebind_while_a_pass_over_host_pages_is_open_is_refused`]: refused `EncodedUses`, before
//!   any draw.
//!
//! A command buffer of [`MANY`] passes over one such target, each loading and storing it beside a
//! cleared and stored `Depth32Float` depth attachment and blending one rectangle over it (a WebGPU
//! frame split at every indexed draw, `stencilMask`'s), holds its texels as the same passes leave
//! a private texture, the bytes between its rows untouched; where the device renders it apart
//! (`linear_merge.rs`) or it is a shadow merged back (`shadow_merge.rs`), its merges take one
//! scratch and lend it to every later merge of the command buffer (`merge_scratch.rs`), so the
//! device's memory grows by no scratch a pass while they are encoded:
//! [`many_passes_over_a_private_buffer_take_one_scratch`] (rows of a multiple of 256 bytes),
//! [`many_passes_over_packed_rows_take_one_scratch`] (a `Shared` buffer of an IOSurface's packed
//! rows, of no multiple of 256), [`many_passes_over_host_pages_take_one_scratch`] (such rows over
//! host pages) and [`many_passes_over_host_pages_of_whole_rows_take_one_scratch`] (host pages,
//! rows of a multiple of 256).
//!
//! Each prints the texture's form (`LINEAR-ATTACHMENT`).

use super::degrade_notices::on_device_with;
use super::degrade_notices::{counts, on_device};
use super::first_use_writes::CHECK_WAIT;
use super::host_memory::{counting, no_copy, not_importable, Mapping, Seen, GUEST_PAGE};
use super::in_place_rows;
use super::no_copy_heaps::{no_copy_heap, shared, HEAP};
use super::no_copy_textures::{blending, floats, rects_beside, staged};
use super::rebinds::rebind;
use super::{at_plan, Messages, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{
    BarrierScope, BlendFactor, DataType, PixelFormat, PrimitiveType, RenderStages, TextureType,
    TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk::{Format, FormatFeatureFlags2};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBufferError, CommandQueue, Degradation, Device, Error,
    EventRef, HeapType, Invalid, LoadAction, Origin, PipelineOptions, Problem, RebindProblem,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

/// The kind a pass that renders a linear image apart takes.
pub(super) const LINEAR_COPIES: [Degradation; 1] = [Degradation::LinearAttachmentCopies];

/// The kind a pass that could render a linear image in place takes where it renders it through
/// copies: it clears it, and no command of it may read the texture's bytes.
pub(super) const CHOSEN_COPIES: [Degradation; 1] = [Degradation::LinearAttachmentClearCopies];

/// The texture's size, and its texel's bytes.
const W: u64 = 256;
const H: u64 = 64;
const TEXEL: u64 = 4;

/// A clear color whose `BGRA8Unorm` bytes are exact: `(51, 102, 153, 255) / 255`.
const CLEAR: [f64; 4] = [0.2, 0.4, 0.6, 1.0];
/// Its bytes, in `BGRA8Unorm`'s order.
const CLEAR_BYTES: [u8; 4] = [153, 102, 51, 255];

/// Whether `device` renders a linear `B8G8R8A8_UNORM` image only by
/// `VK_NV_linear_color_attachment`'s rule: where the device reports `VkFormatProperties3` (Vulkan
/// 1.3, or `VK_KHR_format_feature_flags2` below it, which the runtime enables with the extension;
/// the runtime's instance being at Vulkan 1.2), its linear tiling has the extension's bit and not
/// the color attachment feature.
fn by_the_rule(device: &Device) -> bool {
    let description = &device.properties().description;
    let linear = description
        .formats
        .formats
        .get(&Format::B8G8R8A8_UNORM)
        .map(|p| p.linear_tiling)
        .unwrap_or_default();
    description.has_format_feature_flags2()
        && linear.contains(FormatFeatureFlags2::LINEAR_COLOR_ATTACHMENT_NV)
        && !linear.contains(FormatFeatureFlags2::COLOR_ATTACHMENT)
}

/// The bytes of a `Shared` buffer, which the CPU reads and writes through its contents.
struct Contents(Buffer);

impl Contents {
    /// Bytes `range` of the buffer.
    fn read(&self, range: std::ops::Range<usize>) -> Vec<u8> {
        assert!(range.end as u64 <= self.0.length());
        let contents = self.0.contents().unwrap();
        // SAFETY: inside the buffer's length; the tests read once the commands writing it
        // completed.
        unsafe { std::slice::from_raw_parts(contents.as_ptr().add(range.start), range.len()) }
            .to_vec()
    }

    /// Write `bytes` at byte `at` of the buffer.
    fn write(&self, at: usize, bytes: &[u8]) {
        assert!((at + bytes.len()) as u64 <= self.0.length());
        let contents = self.0.contents().unwrap();
        // SAFETY: inside the buffer's length; no command reads or writes these bytes meanwhile
        // (a pass running meanwhile writes others).
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr().add(at), bytes.len())
        };
    }
}

/// A `Shared` buffer of the runtime's memory whose bytes hold `counting` from `start`, and a
/// `BGRA8Unorm` render target over it, its rows packed: the texture's own linear image where the
/// device makes one there, and always on a device that renders one only by
/// `VK_NV_linear_color_attachment`'s rule ([`by_the_rule`]), so each pass over it there takes the
/// rule's in place or apart way, counted (`WorkCounts::linear_passes_in_place`, `_through_copies`).
fn surface(device: &Device, start: usize) -> (Contents, Buffer, Texture) {
    surface_of(device, start, W, H)
}

/// [`surface`] of `width` and `height`.
fn surface_of(
    device: &Device,
    start: usize,
    width: u64,
    height: u64,
) -> (Contents, Buffer, Texture) {
    let n = (width * height * TEXEL) as usize;
    let buffer = device
        .new_buffer(n as u64, ResourceOptions::shared())
        .unwrap();
    let contents = Contents(buffer.clone());
    contents.write(0, &counting(n, start));
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = buffer.resource_options();
    let texture = buffer.new_texture(&d, 0, width * TEXEL).unwrap();
    assert!(
        !by_the_rule(device) || aliased(&texture),
        "a render target over a Shared buffer is no linear image on a device that renders one by \
         the rule"
    );
    (contents, buffer, texture)
}

/// A no-copy buffer over a new mapping of the texture's bytes, which hold `counting` from
/// `start`, and a `BGRA8Unorm` render target over it, in place where the device's check passed
/// for it ([`renders_in_place`]), else a shadow; `None`, printed, where the device imports no host
/// memory.
fn host_surface(device: &Device, start: usize) -> Option<(Mapping, Buffer, Texture)> {
    host_surface_of(device, start, W, H)
}

/// [`host_surface`] of `width` and `height`.
fn host_surface_of(
    device: &Device,
    start: usize,
    width: u64,
    height: u64,
) -> Option<(Mapping, Buffer, Texture)> {
    let n = (width * height * TEXEL) as usize;
    let len = n.next_multiple_of(GUEST_PAGE);
    let mapping = Mapping::anonymous(len);
    mapping.write(0, &counting(n, start));
    // SAFETY: each test drops the mapping after every buffer over it.
    let made = unsafe {
        no_copy(
            device,
            mapping.pointer(),
            len,
            ResourceOptions::default(),
            None,
        )
    };
    let buffer = match made {
        Ok(buffer) => buffer,
        Err(error) if not_importable(&error) => {
            eprintln!("LINEAR-ATTACHMENT: the device imports no host memory: {error:?}");
            return None;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = buffer.resource_options();
    let texture = buffer.new_texture(&d, 0, width * TEXEL).unwrap();
    Some((mapping, buffer, texture))
}

/// Whether `device`'s check passed for render targets over pointer pages, once it answered: a
/// texture over host pages starts it, and one made after it answered takes its verdict.
fn renders_in_place(device: &Device) -> bool {
    if let Some((mapping, buffer, texture)) = host_surface_of(device, 0, 16, 16) {
        drop((texture, buffer));
        drop(mapping);
    }
    let settled = device.settle_in_place_checks(CHECK_WAIT);
    println!(
        "LINEAR-ATTACHMENT on {}: the in-place check settled {settled}: {:?}",
        device.name(),
        device.in_place_check_report()
    );
    // The surface's rows lie one after another from the start of its pages.
    let on_lines = in_place_rows::on_lines(device, 0, W * TEXEL, W * TEXEL, H);
    in_place_rows::passed(device, "Pointer renders", on_lines)
}

/// Whether `texture`'s texels are its own image's: aliased, not a shadow.
fn aliased(texture: &Texture) -> bool {
    // SAFETY: the handle is only compared, while the texture lives.
    unsafe { texture.vk_image() }.is_some()
}

/// A private `BGRA8Unorm` texture of the surface's size and `samples`.
fn private(device: &Device, samples: u64) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, W, H, false);
    if samples > 1 {
        d.texture_type = TextureType::Type2DMultisample;
        d.sample_count = samples;
    }
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d).unwrap()
}

/// `texture`'s texels blitted out into a new `Shared` buffer, in a command buffer that completes.
fn blit_out(queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let (bpr, len) = (W * TEXEL, W * H * TEXEL);
    let out = queue
        .device()
        .new_buffer(len, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(W, H, 1),
        &out,
        0,
        bpr,
        len,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let contents = out.contents().unwrap();
    // SAFETY: a `Shared` buffer of `len` bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), len as usize) }.to_vec()
}

/// One render pass, of no draws, over `attachments` (each its texture, load action and resolve
/// texture), storing each (resolving where it has a resolve texture), clearing to [`CLEAR`], in a
/// command buffer that completes.
fn pass(queue: &CommandQueue, attachments: &[(&Texture, LoadAction, Option<&Texture>)]) {
    let mut descriptor = RenderPassDescriptor::new();
    for (index, &(texture, load, resolve)) in attachments.iter().enumerate() {
        let attachment = &mut descriptor.color_attachments[index];
        attachment.texture = Some(texture.clone());
        attachment.load_action = load;
        attachment.store_action = match resolve {
            Some(_) => StoreAction::MultisampleResolve,
            None => StoreAction::Store,
        };
        attachment.resolve_texture = resolve.cloned();
        attachment.clear_color = ClearColor::new(CLEAR[0], CLEAR[1], CLEAR[2], CLEAR[3]);
    }
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .render_command_encoder(&descriptor)
        .unwrap()
        .end_encoding()
        .unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// The texture's bytes cleared to [`CLEAR`].
fn cleared() -> Vec<u8> {
    CLEAR_BYTES
        .iter()
        .copied()
        .cycle()
        .take((W * H * TEXEL) as usize)
        .collect()
}

/// The [`LINEAR_COPIES`] notices among those delivered so far (removed), counted.
fn copies_announced(messages: &Messages) -> usize {
    announced(messages, Degradation::LinearAttachmentCopies)
}

/// The notices of `kind` among those delivered so far (all removed), counted.
fn announced(messages: &Messages, kind: Degradation) -> usize {
    counts(&messages.take_notices())
        .get(&kind)
        .copied()
        .unwrap_or(0)
}

/// The NDC corners of the whole viewport and its halves, as the rectangle's corners.
const WHOLE: [f32; 4] = [-1.0, -1.0, 1.0, 1.0];
const LEFT: [f32; 4] = [-1.0, -1.0, 0.0, 1.0];
const RIGHT: [f32; 4] = [0.0, -1.0, 1.0, 1.0];

/// One render pass over `texture`, which it loads and stores, of `pipeline`'s blended draws of a
/// half-transparent color, one over each of `corners`, through `scissor` where given, in a
/// command buffer that completes; beside a private depth or stencil attachment of `beside`'s
/// format where given (`Depth32Float` a depth one, any other a stencil one), cleared and not
/// stored.
fn blend_pass(
    queue: &CommandQueue,
    pipeline: &RenderPipelineState,
    texture: &Texture,
    scissor: Option<ScissorRect>,
    corners: &[[f32; 4]],
    beside: Option<PixelFormat>,
) {
    let device = queue.device();
    let color = staged(device, &floats([0.9, 0.3, 0.1, 0.5]));
    let corners: Vec<Buffer> = corners
        .iter()
        .map(|c| staged(device, &floats(*c)))
        .collect();
    let mut descriptor = RenderPassDescriptor::new();
    let attachment = &mut descriptor.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = LoadAction::Load;
    attachment.store_action = StoreAction::Store;
    if let Some(pf) = beside {
        let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::private();
        let zeta = device.new_texture(&d).unwrap();
        match pf {
            PixelFormat::Depth32Float => {
                descriptor.depth_attachment.texture = Some(zeta);
                descriptor.depth_attachment.load_action = LoadAction::Clear;
                descriptor.depth_attachment.store_action = StoreAction::DontCare;
            }
            _ => {
                descriptor.stencil_attachment.texture = Some(zeta);
                descriptor.stencil_attachment.load_action = LoadAction::Clear;
                descriptor.stencil_attachment.store_action = StoreAction::DontCare;
            }
        }
    }
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    if let Some(scissor) = scissor {
        encoder.set_scissor_rect(scissor).unwrap();
    }
    encoder.set_fragment_buffer(Some(&color), 0, 0).unwrap();
    for corner in &corners {
        encoder.set_vertex_buffer(Some(corner), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// [`blend_pass`]'s bytes over a private texture holding `bytes` first.
fn reference(
    queue: &CommandQueue,
    pipeline: &RenderPipelineState,
    bytes: &[u8],
    scissor: Option<ScissorRect>,
    corners: &[[f32; 4]],
    beside: Option<PixelFormat>,
) -> Vec<u8> {
    let device = queue.device();
    let texture = private(device, 1);
    let source = staged(device, bytes);
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        W * TEXEL,
        W * H * TEXEL,
        Size::new(W, H, 1),
        &texture,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    blend_pass(queue, pipeline, &texture, scissor, corners, beside);
    blit_out(queue, &texture)
}

/// Where `pages` differ from `expected`, the private texture's bytes of a `W` x `H` `BGRA8Unorm`
/// pass through `scissor`: how many bytes and rows, and the first [`FIRST_DIFFERING`] bytes, each
/// with its row, column and channel, the private pass's value and the pages', and where it lies
/// against the scissor (inside, on its edge, or outside): stale or partial rows read as a missing
/// barrier, values a little off as the blend.
fn differing(pages: &[u8], expected: &[u8], scissor: Option<ScissorRect>) -> String {
    let at = |i: usize| {
        let texel = i as u64 / TEXEL;
        (texel % W, texel / W, i as u64 % TEXEL)
    };
    let place = |x: u64, y: u64| match scissor {
        None => "with no scissor",
        Some(s) => {
            let inside = (s.x..s.x + s.width).contains(&x) && (s.y..s.y + s.height).contains(&y);
            let edge = x == s.x || x + 1 == s.x + s.width || y == s.y || y + 1 == s.y + s.height;
            match (inside, edge) {
                (true, true) => "on the scissor's edge",
                (true, false) => "inside the scissor",
                (false, _) => "outside the scissor",
            }
        }
    };
    let differ: Vec<usize> = (0..pages.len())
        .filter(|&i| pages[i] != expected[i])
        .collect();
    let rows: std::collections::BTreeSet<u64> = differ.iter().map(|&i| at(i).1).collect();
    let first: Vec<String> = (differ.iter().take(FIRST_DIFFERING))
        .map(|&i| {
            let (x, y, channel) = at(i);
            format!(
                "row {y} column {x} {} private {:#04x} pages {:#04x} {}",
                ["B", "G", "R", "A"][channel as usize],
                expected[i],
                pages[i],
                place(x, y)
            )
        })
        .collect();
    format!(
        "{} bytes differ in {} rows (from {:?} to {:?}); the first: {}",
        differ.len(),
        rows.len(),
        rows.first(),
        rows.last(),
        first.join("; ")
    )
}

/// The differing bytes [`differing`] lists.
const FIRST_DIFFERING: usize = 32;

/// [`blend_pass`] over a render target over a `Shared` buffer (counting from `start`), twice, each
/// time over a fresh one: its bytes hold what the same pass leaves in a private texture of the same
/// bytes, and the bytes past `scissor` keep theirs. Where the texture is aliased by the rule both
/// render in place, announcing nothing; elsewhere no pass counts a way.
/// Its devices take their queue plan (`at_plan`): the work of the tests that call it
/// changes placements under the lane's queue limits.
fn blends_over_the_pages(start: usize, scissor: Option<ScissorRect>, corners: &[[f32; 4]]) {
    on_device_with(at_plan, |device, messages| {
        messages.take_notices();
        let before = device.work_counts();
        let mut rule = false;
        for _ in 0..2 {
            let (contents, buffer, texture) = surface(device, start);
            rule = by_the_rule(device) && aliased(&texture);
            println!(
                "LINEAR-ATTACHMENT blended over on {}: scissor {scissor:?}, {} draws, by the rule \
                 {rule}",
                device.name(),
                corners.len()
            );
            let queue = device.new_command_queue().unwrap();
            let pipeline = blending(device, PixelFormat::BGRA8Unorm);
            let n = (W * H * TEXEL) as usize;
            let bytes = counting(n, start);
            let expected = reference(&queue, &pipeline, &bytes, scissor, corners, None);
            blend_pass(&queue, &pipeline, &texture, scissor, corners, None);
            let pages = contents.read(0..n);
            assert!(
                pages == expected,
                "the pages differ from the private texture's pass: {}",
                differing(&pages, &expected, scissor)
            );
            if let Some(s) = scissor {
                let inside = |i: usize| {
                    let (x, y) = ((i as u64 / TEXEL) % W, (i as u64 / TEXEL) / W);
                    (s.x..s.x + s.width).contains(&x) && (s.y..s.y + s.height).contains(&y)
                };
                let changed = (0..n)
                    .filter(|&i| !inside(i) && pages[i] != bytes[i])
                    .count();
                assert_eq!(changed, 0, "bytes past the scissor changed");
            }
            drop((texture, buffer));
            drop(contents);
        }
        assert_eq!(
            ways(device, &before),
            match rule {
                true => (2, 0),
                false => (0, 0),
            },
            "the passes' ways, in place and through copies"
        );
        let notices = counts(&messages.take_notices());
        assert_eq!(notices.get(&Degradation::LinearAttachmentClearCopies), None);
        assert_eq!(notices.get(&Degradation::LinearAttachmentCopies), None);
    });
}

/// The passes over a bound linear attachment since `before`, in place and through copies.
fn ways(device: &Device, before: &WorkCounts) -> (u64, u64) {
    let after = device.work_counts();
    (
        after.linear_passes_in_place - before.linear_passes_in_place,
        after.linear_passes_through_copies - before.linear_passes_through_copies,
    )
}

/// Three blended draws over the whole texture, unscissored, twice its pixels: in place.
#[test]
fn blends_over_most_of_a_linear_texture_render_in_place() {
    blends_over_the_pages(11, None, &[LEFT, RIGHT, WHOLE]);
}

/// One blended full-viewport draw through a 64x32 scissor, its box once: in place.
#[test]
fn scissored_blends_over_a_corner_render_in_place() {
    blends_over_the_pages(13, Some(ScissorRect::new(0, 0, 64, 32)), &[WHOLE]);
}

/// Two blended draws through a 192x48 scissor, their box twice: in place.
#[test]
fn a_blend_over_most_of_it_inside_a_box_renders_in_place() {
    blends_over_the_pages(17, Some(ScissorRect::new(32, 8, 192, 48)), &[WHOLE, WHOLE]);
}

/// One blended full-viewport draw over a render target over a `Shared` buffer (counting from
/// `start`), which the pass loads and stores, beside a private depth or stencil attachment of
/// `beside`'s format where given (cleared, not stored, not tested): its bytes hold what the same
/// pass leaves in a private texture of the same bytes. Where the texture is aliased by the rule, a
/// pass with a depth or stencil attachment renders it apart, announcing `linear-attachment-copies`
/// with a reason naming the attachment, as NVIDIA's hardware renders a linear color only with
/// neither (an RTX 3090 lost the device on such a pass in place); the pass
/// with neither renders it in place, announcing nothing. Each case runs in its own test, so a
/// device lost in one leaves the others theirs.
fn blend_beside(start: usize, beside: Option<PixelFormat>) {
    on_device(|device, messages| {
        messages.take_notices();
        let (contents, buffer, texture) = surface(device, start);
        let rule = by_the_rule(device) && aliased(&texture);
        println!(
            "LINEAR-ATTACHMENT blended beside {beside:?} on {}: aliased by the rule {rule}",
            device.name()
        );
        let queue = device.new_command_queue().unwrap();
        let pipeline = rects_beside(device, PixelFormat::BGRA8Unorm, true, beside);
        let n = (W * H * TEXEL) as usize;
        let bytes = counting(n, start);
        let expected = reference(&queue, &pipeline, &bytes, None, &[WHOLE], beside);
        messages.take_notices();
        let before = device.work_counts();
        blend_pass(&queue, &pipeline, &texture, None, &[WHOLE], beside);
        assert_eq!(
            ways(device, &before),
            match (rule, beside.is_some()) {
                (true, false) => (1, 0),
                (true, true) => (0, 1),
                (false, _) => (0, 0),
            },
            "the pass's way, in place and through copies"
        );
        let pages = contents.read(0..n);
        let differ = (0..n).filter(|&i| pages[i] != expected[i]).count();
        println!("LINEAR-ATTACHMENT blended beside {beside:?}: {differ} bytes differ");
        assert_eq!(
            differ, 0,
            "the pages differ from the private texture's pass"
        );
        let notices = messages.take_notices();
        let reasons: Vec<&str> = notices
            .iter()
            .filter_map(|m| m.degradation.as_ref())
            .filter(|d| d.degradation == Degradation::LinearAttachmentCopies)
            .map(|d| d.reason.as_str())
            .collect();
        for reason in &reasons {
            println!("LINEAR-ATTACHMENT blended beside {beside:?}: announced {reason}");
        }
        let apart = rule && beside.is_some();
        assert_eq!(reasons.len(), usize::from(apart), "{notices:?}");
        let named = match beside {
            Some(PixelFormat::Depth32Float) => "a depth attachment",
            Some(_) => "a stencil attachment",
            None => "",
        };
        assert!(
            reasons.iter().all(|r| r.ends_with(named)),
            "the reason names {named}: {reasons:?}"
        );
        drop((texture, buffer));
        drop(contents);
    });
}

/// [`blend_beside`] with no depth or stencil attachment: in place.
#[test]
fn a_blend_with_no_depth_or_stencil_renders_the_linear_texture_in_place() {
    blend_beside(17, None);
}

/// [`blend_beside`] with a `Depth32Float` depth attachment: apart.
#[test]
fn a_blend_beside_a_depth_attachment_renders_the_linear_texture_apart() {
    blend_beside(19, Some(PixelFormat::Depth32Float));
}

/// [`blend_beside`] with a `Stencil8` stencil attachment: apart.
#[test]
fn a_blend_beside_a_stencil_attachment_renders_the_linear_texture_apart() {
    blend_beside(23, Some(PixelFormat::Stencil8));
}

/// The owned fixture `zwd_shapes`' library on `device`.
fn zwd_library(device: &Device) -> metal2vulkan_runtime::Library {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/zwd_shapes.ll");
    device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap()
}

/// A source-over blend of premultiplied colors into a `BGRA8Unorm` color.
fn over() -> ColorAttachmentDescriptor {
    let mut over = ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm);
    over.blending_enabled = true;
    over.source_rgb_blend_factor = BlendFactor::One;
    over.source_alpha_blend_factor = BlendFactor::One;
    over.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    over.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    over
}

/// `zwd_quad` with `fragment` of `library`, its function constant 0 `variant`, into `color`, with
/// a `Stencil8` stencil format where `stencil`.
fn zwd_state(
    device: &Device,
    library: &metal2vulkan_runtime::Library,
    (fragment, variant): (&str, u32),
    color: Option<ColorAttachmentDescriptor>,
    stencil: bool,
) -> RenderPipelineState {
    let constants = FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(DataType::UInt, variant.to_le_bytes().to_vec()),
    );
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name_constant_values(name, &constants)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut d = RenderPipelineDescriptor::new(function("zwd_quad"));
    d.fragment = Some(function(fragment));
    d.color_attachments[0] = color;
    if stencil {
        d.stencil_format = Some(PixelFormat::Stencil8);
    }
    device
        .new_render_pipeline_state(&d, &PipelineOptions::default())
        .unwrap()
}

/// The panel and the colorless clip of [`a_colorless_clip_zeroes_past_the_blends_box`]:
/// `zwd_panel` blended source-over into a `BGRA8Unorm` color, and `zwd_clip` into no color (their
/// function constant 3); for a pass with a stencil attachment where `stencil`.
fn panel_and_clip(device: &Device, stencil: bool) -> (RenderPipelineState, RenderPipelineState) {
    let library = zwd_library(device);
    (
        zwd_state(device, &library, ("zwd_panel", 3), Some(over()), stencil),
        zwd_state(device, &library, ("zwd_clip", 3), None, stencil),
    )
}

/// `ZwdShape`: a box `rect` (pixels) with the premultiplied `color` and outline `ring`, corners
/// of radius 13 + 3, in a target of `size`.
fn zwd_shape_in(rect: [f32; 4], color: [f32; 4], ring: [f32; 4], size: [u64; 2]) -> Vec<u8> {
    rect.iter()
        .chain(&color)
        .chain(&ring)
        .chain(&[13.0, size[0] as f32, size[1] as f32, 0.0])
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

/// [`zwd_shape_in`] of no outline in the texture.
fn zwd_shape(rect: [f32; 4], color: [f32; 4]) -> Vec<u8> {
    zwd_shape_in(rect, color, [0.0; 4], [W, H])
}

/// A private `Stencil8` attachment of `size` beside `descriptor`'s colors, cleared and not
/// stored.
fn beside_stencil(device: &Device, descriptor: &mut RenderPassDescriptor, size: [u64; 2]) {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::Stencil8, size[0], size[1], false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    let stencil = &mut descriptor.stencil_attachment;
    stencil.texture = Some(device.new_texture(&d).unwrap());
    stencil.load_action = LoadAction::Clear;
    stencil.store_action = StoreAction::DontCare;
}

/// [`a_colorless_clip_zeroes_past_the_blends_box`]'s pass over `texture`, loaded and stored,
/// beside a stencil attachment where `stencil`: two blended panels over the left half through
/// its scissor, then the colorless clip over the right quarter through its scissor.
fn zeroing_pass(
    queue: &CommandQueue,
    (panel, clip): &(RenderPipelineState, RenderPipelineState),
    texture: &Texture,
    stencil: bool,
) {
    let mut descriptor = RenderPassDescriptor::new();
    let attachment = &mut descriptor.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = LoadAction::Load;
    attachment.store_action = StoreAction::Store;
    if stencil {
        beside_stencil(queue.device(), &mut descriptor, [W, H]);
    }
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    let (w, h) = (W as f32, H as f32);
    let left = ScissorRect::new(0, 0, W / 2, H);
    let right = ScissorRect::new(3 * W / 4, 0, W / 4, H);
    let draws = [
        (
            panel,
            left,
            [4.0, 4.0, w / 2.0 - 4.0, h - 4.0],
            [0.5, 0.25, 0.125, 0.5],
        ),
        (
            panel,
            left,
            [24.0, 8.0, w / 2.0 - 24.0, h - 8.0],
            [0.25, 0.5, 0.25, 0.25],
        ),
        (
            clip,
            right,
            [3.0 * w / 4.0 + 4.0, 4.0, w - 4.0, h - 4.0],
            [0.0; 4],
        ),
    ];
    for (state, scissor, rect, color) in draws {
        let bytes = zwd_shape(rect, color);
        encoder.set_render_pipeline_state(state).unwrap();
        encoder.set_scissor_rect(scissor).unwrap();
        encoder.set_vertex_bytes(&bytes, 0).unwrap();
        encoder.set_fragment_bytes(&bytes, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// A private texture holding `bytes`, the surface's size, filled in a command buffer that
/// completes.
fn private_holding(queue: &CommandQueue, bytes: &[u8], size: [u64; 2]) -> Texture {
    let device = queue.device();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, size[0], size[1], false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let texture = device.new_texture(&d).unwrap();
    let source = staged(device, bytes);
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        size[0] * TEXEL,
        size[0] * size[1] * TEXEL,
        Size::new(size[0], size[1], 1),
        &texture,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    texture
}

/// `texture`'s texels of `size`, blitted out into a new `Shared` buffer, in a command buffer that
/// completes.
fn read_out(queue: &CommandQueue, texture: &Texture, size: [u64; 2]) -> Vec<u8> {
    let (bpr, len) = (size[0] * TEXEL, size[0] * size[1] * TEXEL);
    let out = queue
        .device()
        .new_buffer(len, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(size[0], size[1], 1),
        &out,
        0,
        bpr,
        len,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let contents = out.contents().unwrap();
    // SAFETY: a `Shared` buffer of `len` bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), len as usize) }.to_vec()
}

/// Two blended panels through the left half's scissor, their box twice, and a clip of a state of
/// no color through the right quarter's, which zeroes the color inside its rounded rectangle,
/// twice, each over fresh pages, beside a stencil attachment where `stencil`: each pass's pages
/// hold what the same pass leaves in a private texture of the same bytes. Where the texture is
/// aliased by the rule, the passes render in place, or, beside the stencil, apart over a box that
/// holds both, merged back.
/// Its devices take their queue plan (`at_plan`): the work of the tests that call it
/// changes placements under the lane's queue limits.
fn colorless_clip(start: usize, stencil: bool) {
    on_device_with(at_plan, |device, _| {
        let before = device.work_counts();
        let states = panel_and_clip(device, stencil);
        let mut rule = false;
        for _ in 0..2 {
            let (contents, buffer, texture) = surface(device, start);
            rule = by_the_rule(device) && aliased(&texture);
            println!(
                "LINEAR-ATTACHMENT a colorless clip past the blends on {}: stencil {stencil}, by \
                 the rule {rule}",
                device.name()
            );
            let queue = device.new_command_queue().unwrap();
            let n = (W * H * TEXEL) as usize;
            let bytes = counting(n, start);
            let reference = private_holding(&queue, &bytes, [W, H]);
            zeroing_pass(&queue, &states, &reference, stencil);
            let expected = read_out(&queue, &reference, [W, H]);
            // The clip's middle texel, zeroed.
            let middle = ((H / 2 * W + 7 * W / 8) * TEXEL) as usize;
            assert_eq!(
                &expected[middle..middle + 4],
                [0; 4],
                "the clip zeroes the color"
            );
            zeroing_pass(&queue, &states, &texture, stencil);
            let pages = contents.read(0..n);
            let differ = (0..n / 4)
                .filter(|&t| pages[t * 4..t * 4 + 4] != expected[t * 4..t * 4 + 4])
                .count();
            assert_eq!(differ, 0, "texels differ from the private texture's pass");
            drop((texture, buffer));
            drop(contents);
        }
        assert_eq!(
            ways(device, &before),
            match (rule, stencil) {
                (true, false) => (2, 0),
                (true, true) => (0, 2),
                (false, _) => (0, 0),
            },
            "the passes' ways, in place and through copies"
        );
    });
}

/// [`colorless_clip`] alone: in place.
#[test]
fn a_colorless_clip_zeroes_past_the_blends_box() {
    colorless_clip(23, false);
}

/// [`colorless_clip`] beside a stencil attachment: apart, merged over the box.
#[test]
fn a_colorless_clip_beside_a_stencil_zeroes_past_the_blends_box() {
    colorless_clip(29, true);
}

/// The reading lanes' surface: 512x256.
const WIDE: [u64; 2] = [512, 256];

/// The rows at the bottom of [`WIDE`] no draw of the CPU-write lanes reaches, and the byte the
/// CPU writes there while the pass runs. Enough rows that no tile a draw reaches holds them:
/// Metal stores each tile a draw reaches whole, its rows no draw reaches too, so a CPU write to
/// such a row while the pass runs is kept or lost by when it lands, on Metal as anywhere; a tile
/// no draw reaches it never stores. Measured on an M3 Max, macOS 27.0.1, through Metal alone
/// with this lane's pass at 6000 panels: with the last 8 rows left out, a write late in the pass
/// kept 5163, 11282 and 13322 of 16384 bytes; with the last 64, every byte at each of 11 delays
/// across the pass, and of 7 beside the stencil attachment.
const UNDRAWN: u64 = 64;
const MARK: u8 = 0xab;

/// The blended panels of the CPU-write lanes' pass: enough that the pass runs while the CPU
/// writes.
const LAYERS: usize = 600;

/// A loaded pass over a [`WIDE`] render target over a `Shared` buffer (counting from `start`),
/// beside a stencil attachment where `stencil`, of [`LAYERS`] blended rounded panels through a
/// scissor that leaves out the last [`UNDRAWN`] rows, in a command buffer that signals a shared
/// event before it: once the event is signalled the CPU writes [`MARK`] over those rows, which the
/// pass never writes, while the pass runs (the more likely the longer it runs). Metal keeps what
/// the CPU wrote there, as it stores no tile no draw reaches; so does the pass in place and apart,
/// merged (a copy back of the area would put the rows' old bytes back where the write landed after
/// its copy in). Every other byte holds what the same pass leaves in a private texture of the same
/// bytes.
fn cpu_write(start: usize, stencil: bool) {
    on_device(|device, _| {
        let (contents, buffer, texture) = surface_of(device, start, WIDE[0], WIDE[1]);
        let rule = by_the_rule(device) && aliased(&texture);
        println!(
            "LINEAR-ATTACHMENT a CPU write during the pass on {}: stencil {stencil}, by the rule \
             {rule}",
            device.name()
        );
        let library = zwd_library(device);
        let panel = zwd_state(device, &library, ("zwd_panel", 3), Some(over()), stencil);
        let queue = device.new_command_queue().unwrap();
        let [w, h] = WIDE;
        let drawn = h - UNDRAWN;
        let shape = zwd_shape_in(
            [4.0, 4.0, w as f32 - 4.0, drawn as f32 - 4.0],
            [0.01, 0.02, 0.03, 0.04],
            [0.0; 4],
            WIDE,
        );
        let encode = |texture: &Texture, command_buffer: &metal2vulkan_runtime::CommandBuffer| {
            let mut descriptor = RenderPassDescriptor::new();
            let attachment = &mut descriptor.color_attachments[0];
            attachment.texture = Some(texture.clone());
            attachment.load_action = LoadAction::Load;
            attachment.store_action = StoreAction::Store;
            if stencil {
                beside_stencil(device, &mut descriptor, WIDE);
            }
            let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
            encoder.set_render_pipeline_state(&panel).unwrap();
            encoder
                .set_scissor_rect(ScissorRect::new(0, 0, w, drawn))
                .unwrap();
            encoder.set_vertex_bytes(&shape, 0).unwrap();
            encoder.set_fragment_bytes(&shape, 0).unwrap();
            for _ in 0..LAYERS {
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
            }
            encoder.end_encoding().unwrap();
        };
        let n = (w * h * TEXEL) as usize;
        let bytes = counting(n, start);
        let reference = private_holding(&queue, &bytes, WIDE);
        let command_buffer = queue.command_buffer().unwrap();
        encode(&reference, &command_buffer);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let mut expected = read_out(&queue, &reference, WIDE);
        let rows = (drawn * w * TEXEL) as usize..n;
        expected[rows.clone()].fill(MARK);
        let event = device.new_shared_event().unwrap();
        let before = device.work_counts();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_signal_event(EventRef::from(&event), 1)
            .unwrap();
        encode(&texture, &command_buffer);
        command_buffer.commit().unwrap();
        let signalled = event.wait_until_signaled_value(1, WAIT);
        std::thread::sleep(Duration::from_millis(1));
        contents.write(rows.start, &vec![MARK; rows.len()]);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let pages = contents.read(0..n);
        let kept = pages[rows.clone()].iter().filter(|&&b| b == MARK).count();
        println!(
            "LINEAR-ATTACHMENT a CPU write during the pass: signalled {signalled}, {kept} of {} \
             bytes kept",
            rows.len()
        );
        assert!(signalled, "the event was not signalled");
        assert_eq!(
            ways(device, &before),
            match (rule, stencil) {
                (true, false) => (1, 0),
                (true, true) => (0, 1),
                (false, _) => (0, 0),
            },
            "the pass's way, in place and through copies"
        );
        assert_eq!(kept, rows.len(), "the pass wrote rows no draw reaches");
        let differ = (0..n).filter(|&i| pages[i] != expected[i]).count();
        assert_eq!(
            differ, 0,
            "the pages differ from the private texture's pass"
        );
        drop((texture, buffer));
        drop(contents);
    });
}

/// [`cpu_write`] alone: in place where the texture is aliased by the rule.
#[test]
fn a_cpu_write_at_an_undrawn_pixel_survives_a_loaded_pass() {
    cpu_write(31, false);
}

/// [`cpu_write`] beside a stencil attachment: apart where the texture is aliased by the rule,
/// merged back.
#[test]
fn a_cpu_write_at_an_undrawn_pixel_survives_a_pass_apart() {
    cpu_write(37, true);
}

/// A pass over `texture` of [`WIDE`], loading or clearing it (`load`), beside a stencil
/// attachment where `stencil`: a blended rounded panel (`zwd_panel`), a barrier over the render
/// targets from the fragment stage to the fragment stage, then a full-target draw
/// (`zwd_mask_nearest`) that samples `texture` itself at each pixel's own texel and blends its
/// red channel times a color over it, in a command buffer that completes.
fn sampling_pass(
    queue: &CommandQueue,
    (panel, reader): &(RenderPipelineState, RenderPipelineState),
    texture: &Texture,
    load: LoadAction,
    stencil: bool,
) {
    let device = queue.device();
    let mut descriptor = RenderPassDescriptor::new();
    let attachment = &mut descriptor.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = load;
    attachment.store_action = StoreAction::Store;
    attachment.clear_color = ClearColor::new(CLEAR[0], CLEAR[1], CLEAR[2], CLEAR[3]);
    if stencil {
        beside_stencil(device, &mut descriptor, WIDE);
    }
    let [w, h] = WIDE.map(|v| v as f32);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    let drawn = zwd_shape_in(
        [16.0, 8.0, w - 16.0, h - 8.0],
        [0.5, 0.25, 0.125, 0.5],
        [0.0; 4],
        WIDE,
    );
    encoder.set_render_pipeline_state(panel).unwrap();
    encoder.set_vertex_bytes(&drawn, 0).unwrap();
    encoder.set_fragment_bytes(&drawn, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.memory_barrier_with_scope(
        BarrierScope::RENDER_TARGETS,
        RenderStages::FRAGMENT,
        RenderStages::FRAGMENT,
    );
    let read = zwd_shape_in(
        [0.0, 0.0, w, h],
        [0.25, 0.375, 0.5, 0.5],
        [1.0, 0.0, 0.0, 0.0],
        WIDE,
    );
    encoder.set_render_pipeline_state(reader).unwrap();
    encoder.set_fragment_texture(Some(texture), 0).unwrap();
    encoder.set_vertex_bytes(&read, 0).unwrap();
    encoder.set_fragment_bytes(&read, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// [`sampling_pass`] over a [`WIDE`] render target over a `Shared` buffer (counting from `start`),
/// and over a private texture of the same bytes: its bytes hold the private texture's texels, so
/// the draw after the barrier read the panel the pass drew. Where the texture is aliased by the
/// rule the pass renders in place, loaded or cleared (a command of it may read the texture's bytes,
/// so no clear is copied), announcing no `linear-attachment-clear-copies`; beside a stencil
/// attachment apart, split at the barrier: the stand-in merged back (or, cleared, copied back) and
/// copied in again before the reading draw. Where the device renders the texture's own image as any
/// other, the pass alone records both its instances inline, the reading draw's after the split
/// (`WorkCounts::inline_split_instances`).
fn sampling(start: usize, load: LoadAction, stencil: bool) {
    on_device(|device, messages| {
        messages.take_notices();
        let (contents, buffer, texture) = surface_of(device, start, WIDE[0], WIDE[1]);
        let rule = by_the_rule(device) && aliased(&texture);
        let inline = aliased(&texture) && !by_the_rule(device) && !stencil;
        println!(
            "LINEAR-ATTACHMENT sampled after a barrier on {}: {load:?}, stencil {stencil}, by the \
             rule {rule}, inline {inline}",
            device.name()
        );
        let library = zwd_library(device);
        let states = (
            zwd_state(device, &library, ("zwd_panel", 3), Some(over()), stencil),
            zwd_state(
                device,
                &library,
                ("zwd_mask_nearest", 0),
                Some(over()),
                stencil,
            ),
        );
        let queue = device.new_command_queue().unwrap();
        let n = (WIDE[0] * WIDE[1] * TEXEL) as usize;
        let bytes = counting(n, start);
        let reference = private_holding(&queue, &bytes, WIDE);
        sampling_pass(&queue, &states, &reference, load, stencil);
        let expected = read_out(&queue, &reference, WIDE);
        let before = device.work_counts();
        sampling_pass(&queue, &states, &texture, load, stencil);
        let inlined = inlined(device, &before);
        println!("LINEAR-ATTACHMENT sampled after a barrier: inline counts {inlined:?}");
        assert_eq!(
            inlined,
            match inline {
                true => [1, 1, 1],
                false => [0, 0, 0],
            },
            "passes inline, of them split, and instances after a split inline"
        );
        let pages = contents.read(0..n);
        let differ = (0..n / 4)
            .filter(|&t| pages[t * 4..t * 4 + 4] != expected[t * 4..t * 4 + 4])
            .count();
        println!("LINEAR-ATTACHMENT sampled after a barrier: {differ} texels differ");
        assert_eq!(differ, 0, "texels differ from the private texture's pass");
        assert_eq!(
            ways(device, &before),
            match (rule, stencil) {
                (true, false) => (1, 0),
                (true, true) => (0, 1),
                (false, _) => (0, 0),
            },
            "the pass's way, in place and through copies"
        );
        let notices = counts(&messages.take_notices());
        assert_eq!(notices.get(&Degradation::LinearAttachmentClearCopies), None);
        drop((texture, buffer));
        drop(contents);
    });
}

/// The passes since `before` whose first instance recorded inline, those of them split, and
/// their instances after a split that recorded inline too.
fn inlined(device: &Device, before: &WorkCounts) -> [u64; 3] {
    let after = device.work_counts();
    [
        after.inline_passes - before.inline_passes,
        after.inline_passes_split - before.inline_passes_split,
        after.inline_split_instances - before.inline_split_instances,
    ]
}

/// [`sampling`] of a loaded pass: in place.
#[test]
fn a_pass_sampling_its_surface_after_a_barrier_reads_its_draws() {
    sampling(41, LoadAction::Load, false);
}

/// [`sampling`] of a cleared pass: in place, the clear not copied.
#[test]
fn a_cleared_pass_sampling_its_surface_after_a_barrier_renders_in_place() {
    sampling(43, LoadAction::Clear, false);
}

/// [`sampling`] of a loaded pass beside a stencil attachment: apart, merged and copied in again
/// at the barrier.
#[test]
fn a_pass_apart_sampling_its_surface_after_a_barrier_reads_its_draws() {
    sampling(47, LoadAction::Load, true);
}

/// [`sampling`] of a cleared pass beside a stencil attachment: apart, copied back at the barrier.
#[test]
fn a_cleared_pass_apart_sampling_its_surface_after_a_barrier_reads_its_draws() {
    sampling(53, LoadAction::Clear, true);
}

/// A loaded pass over `texture`, whose bytes over host pages `detach` detaches while the pass is
/// open, after its first draw (the panel of [`sampling_pass`] over `size`), then the barrier and
/// the draw sampling the texture: the command buffer fails `DetachedBacking` and runs nothing.
/// The passes whose first instance recorded inline (`WorkCounts::inline_passes`).
fn detached_in_an_open_pass(
    device: &Device,
    texture: &Texture,
    size: [u64; 2],
    detach: impl FnOnce() -> Result<bool, Error>,
) -> u64 {
    let library = zwd_library(device);
    let panel = zwd_state(device, &library, ("zwd_panel", 3), Some(over()), false);
    let reader = zwd_state(
        device,
        &library,
        ("zwd_mask_nearest", 0),
        Some(over()),
        false,
    );
    let queue = device.new_command_queue().unwrap();
    let before = device.work_counts();
    let mut descriptor = RenderPassDescriptor::new();
    let attachment = &mut descriptor.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = LoadAction::Load;
    attachment.store_action = StoreAction::Store;
    let [w, h] = size.map(|v| v as f32);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    let drawn = zwd_shape_in(
        [16.0, 8.0, w - 16.0, h - 8.0],
        [0.5, 0.25, 0.125, 0.5],
        [0.0; 4],
        size,
    );
    encoder.set_render_pipeline_state(&panel).unwrap();
    encoder.set_vertex_bytes(&drawn, 0).unwrap();
    encoder.set_fragment_bytes(&drawn, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    assert_eq!(detach(), Ok(true), "the detach");
    encoder.memory_barrier_with_scope(
        BarrierScope::RENDER_TARGETS,
        RenderStages::FRAGMENT,
        RenderStages::FRAGMENT,
    );
    let read = zwd_shape_in(
        [0.0, 0.0, w, h],
        [0.25, 0.375, 0.5, 0.5],
        [1.0, 0.0, 0.0, 0.0],
        size,
    );
    encoder.set_render_pipeline_state(&reader).unwrap();
    encoder.set_fragment_texture(Some(texture), 0).unwrap();
    encoder.set_vertex_bytes(&read, 0).unwrap();
    encoder.set_fragment_bytes(&read, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.error(),
        Some(CommandBufferError::DetachedBacking)
    );
    device.work_counts().inline_passes - before.inline_passes
}

/// [`detached_in_an_open_pass`] over a [`WIDE`] render target over a no-copy buffer's host pages
/// (counting from 59): the pages keep their bytes, and the validation layer reports nothing. The
/// pass holds the pages from its open (`render.rs`), so the images over them the detach replaces
/// stay until the command buffer goes (`detach.rs`). Where the texture is its own image outside
/// `VK_NV_linear_color_attachment`'s rule ([`by_the_rule`]), the pass records inline.
#[test]
fn a_detach_while_a_pass_over_host_pages_is_open_fails_its_command_buffer() {
    on_device(|device, messages| {
        messages.take_notices();
        let Some((mapping, buffer, texture)) = host_surface_of(device, 59, WIDE[0], WIDE[1]) else {
            return;
        };
        let n = (WIDE[0] * WIDE[1] * TEXEL) as usize;
        let inline = detached_in_an_open_pass(device, &texture, WIDE, || buffer.detach_backing());
        let own = aliased(&texture) && !by_the_rule(device);
        println!(
            "LINEAR-ATTACHMENT detached in an open pass on {}: aliased {}, by the rule {}, inline \
             {inline}",
            device.name(),
            aliased(&texture),
            by_the_rule(device)
        );
        if own {
            assert_eq!(inline, 1, "passes inline");
        }
        assert!(
            mapping.read(0..n) == counting(n, 59),
            "the pages changed: the command buffer ran"
        );
        drop((texture, buffer));
        drop(mapping);
    });
}

/// [`detached_in_an_open_pass`] over a render target in a no-copy heap, detached while the pass is
/// open: where the texture is an image of its own (no `shadowed-texture` announced) that the device
/// renders without `VK_NV_linear_color_attachment`'s rule ([`by_the_rule`]; a pass of an image the
/// rule binds records into a secondary), the pass's first instance records inline into a command
/// buffer it holds, naming the image's view from its first draw; the detach leaves that image and
/// view until the command buffer goes, so the command buffer stays valid to its end, and the
/// validation layer reports nothing.
#[test]
fn a_detach_while_a_pass_in_a_no_copy_heap_is_open_fails_its_command_buffer() {
    on_device(|device, messages| {
        messages.take_notices();
        let mapping = Mapping::anonymous(HEAP);
        // SAFETY: the mapping outlives the heap, dropped after it.
        let Some(heap) =
            (unsafe { no_copy_heap(device, mapping.pointer(), HeapType::Placement, None) })
        else {
            return;
        };
        let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, W, H, false);
        d.resource_options = shared();
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let texture = heap.new_texture_at_offset(&d, 0).unwrap();
        let shadowed = counts(&messages.take_notices()).contains_key(&Degradation::ShadowedTexture);
        let rule = by_the_rule(device);
        let inline = detached_in_an_open_pass(device, &texture, [W, H], || heap.detach_backing());
        println!(
            "LINEAR-ATTACHMENT detached in an open pass in a no-copy heap on {}: shadowed \
             {shadowed}, by the rule {rule}, inline {inline}",
            device.name()
        );
        if !shadowed && !rule {
            assert_eq!(inline, 1, "passes inline");
        }
        drop((texture, heap));
        drop(mapping);
    });
}

/// A rebind of a no-copy buffer under a pass's render target while the pass is open, before any
/// draw, is refused `EncodedUses`: the render encoder uses its attachments' bytes from its open.
#[test]
fn a_rebind_while_a_pass_over_host_pages_is_open_is_refused() {
    on_device(|device, _| {
        let Some((mapping, buffer, texture)) = host_surface(device, 61) else {
            return;
        };
        let queue = device.new_command_queue().unwrap();
        let mut descriptor = RenderPassDescriptor::new();
        let attachment = &mut descriptor.color_attachments[0];
        attachment.texture = Some(texture.clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
        let command_buffer = queue.command_buffer().unwrap();
        let encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
        let seen = Arc::new(Seen::default());
        let other = Mapping::anonymous(buffer.length() as usize);
        assert_eq!(
            rebind(&buffer, other, &seen),
            Err(Error::Invalid(Invalid::new(
                "rebind_bytes_no_copy",
                Problem::Rebind(RebindProblem::EncodedUses)
            )))
        );
        assert_eq!(
            seen.runs.load(Ordering::SeqCst),
            0,
            "a refused rebind's deallocator ran"
        );
        encoder.end_encoding().unwrap();
        drop((command_buffer, texture, buffer));
        drop(mapping);
    });
}

/// A render target over a `Shared` buffer is aliased, and on a device that renders it by the rule
/// a pass of it alone that loads it and draws nothing changes no texel, so it records nothing
/// (`render/untouched.rs`) and is counted neither way, announcing nothing; one that clears it
/// renders it through copies, announcing `linear-attachment-clear-copies` once, counted so. The
/// buffer holds what each stored.
#[test]
fn a_cleared_render_target_over_a_shared_buffer_renders_through_copies() {
    on_device(|device, messages| {
        messages.take_notices();
        let (contents, buffer, texture) = surface(device, 3);
        let rule = by_the_rule(device);
        let form = aliased(&texture);
        println!(
            "LINEAR-ATTACHMENT over a Shared buffer on {}: by the rule {rule}, aliased {form}",
            device.name()
        );
        let queue = device.new_command_queue().unwrap();
        let n = (W * H * TEXEL) as usize;
        let before = device.work_counts();
        pass(&queue, &[(&texture, LoadAction::Load, None)]);
        assert_eq!(
            ways(device, &before),
            (0, 0),
            "a pass that loads the texture and draws nothing, in place and through copies"
        );
        assert!(
            contents.read(0..n) == counting(n, 3),
            "a load and store changed the buffer"
        );
        assert_eq!(
            announced(messages, Degradation::LinearAttachmentClearCopies),
            0,
            "a pass that loads the texture"
        );
        let before = device.work_counts();
        pass(&queue, &[(&texture, LoadAction::Clear, None)]);
        assert_eq!(
            ways(device, &before),
            (0, u64::from(rule)),
            "a pass that clears the texture, in place and through copies"
        );
        assert!(
            contents.read(0..n) == cleared(),
            "a clear did not reach the buffer"
        );
        let notices = counts(&messages.take_notices());
        let count = |kind| notices.get(&kind).copied().unwrap_or(0);
        assert_eq!(
            count(Degradation::LinearAttachmentCopies),
            0,
            "a pass of the texture alone"
        );
        assert_eq!(
            count(Degradation::LinearAttachmentClearCopies),
            usize::from(rule),
            "a pass that clears the texture, once a device that renders it by the rule"
        );
        drop((texture, buffer));
        drop(contents);
    });
}

/// A render target over host pages is an image bound to them exactly where the device's check
/// passed for render targets over pointer pages, else a shadow: a pass of it alone that loads it
/// and one that clears it each leave the pages holding what it stored; a shadow's passes announce
/// no linear attachment's copies, and an image in place announces the clear's copies once where
/// the device renders it by the rule, as one over a `Shared` buffer does.
#[test]
fn a_render_target_over_host_pages_is_in_place_where_the_check_passed() {
    on_device(|device, messages| {
        let in_place = renders_in_place(device);
        messages.take_notices();
        let Some((mapping, buffer, texture)) = host_surface(device, 3) else {
            return;
        };
        let rule = by_the_rule(device);
        println!(
            "LINEAR-ATTACHMENT on {}: by the rule {rule}, aliased {}, the check passed for render \
             targets {in_place}",
            device.name(),
            aliased(&texture)
        );
        assert_eq!(
            aliased(&texture),
            in_place,
            "an image bound to the pages exactly where the check passed for them"
        );
        let queue = device.new_command_queue().unwrap();
        let n = (W * H * TEXEL) as usize;
        pass(&queue, &[(&texture, LoadAction::Load, None)]);
        assert!(
            mapping.read(0..n) == counting(n, 3),
            "a load and store changed the pages"
        );
        pass(&queue, &[(&texture, LoadAction::Clear, None)]);
        assert!(
            mapping.read(0..n) == cleared(),
            "a clear did not reach the pages"
        );
        let notices = counts(&messages.take_notices());
        let count = |kind| notices.get(&kind).copied().unwrap_or(0);
        assert_eq!(
            (
                count(Degradation::LinearAttachmentCopies),
                count(Degradation::LinearAttachmentClearCopies)
            ),
            (0, usize::from(in_place && rule)),
            "a pass of the texture alone, and one that clears it"
        );
        drop((texture, buffer));
        drop(mapping);
    });
}

/// A pass of the texture beside a private attachment renders it through copies, announced once
/// where it is aliased by the rule: what it loads and stores, and clears, reaches the buffer, and
/// the private texture keeps its own.
#[test]
fn a_linear_attachment_beside_a_private_one_renders_through_copies() {
    on_device(|device, messages| {
        messages.take_notices();
        let (contents, buffer, texture) = surface(device, 5);
        let apart = by_the_rule(device) && aliased(&texture);
        println!(
            "LINEAR-ATTACHMENT beside a private one on {}: apart {apart}",
            device.name()
        );
        let queue = device.new_command_queue().unwrap();
        let other = private(device, 1);
        let n = (W * H * TEXEL) as usize;
        let before = device.work_counts();
        pass(
            &queue,
            &[
                (&texture, LoadAction::Load, None),
                (&other, LoadAction::Clear, None),
            ],
        );
        assert!(
            contents.read(0..n) == counting(n, 5),
            "a load and store changed the pages"
        );
        assert!(
            blit_out(&queue, &other) == cleared(),
            "the private texture's clear"
        );
        pass(
            &queue,
            &[
                (&texture, LoadAction::Clear, None),
                (&other, LoadAction::Load, None),
            ],
        );
        assert!(
            contents.read(0..n) == cleared(),
            "a clear did not reach the pages"
        );
        assert!(
            blit_out(&queue, &other) == cleared(),
            "the private texture's load"
        );
        assert_eq!(
            ways(device, &before),
            (0, 2 * u64::from(apart)),
            "the passes' ways, in place and through copies"
        );
        assert_eq!(
            copies_announced(messages),
            usize::from(apart),
            "once a device"
        );
        drop((texture, buffer, other));
        drop(contents);
    });
}

/// A multisampled attachment cleared and resolved into the texture: the buffer holds the clear,
/// announced once where the texture is aliased by the rule.
#[test]
fn a_resolve_into_a_linear_texture_resolves_through_a_scratch() {
    on_device(|device, messages| {
        messages.take_notices();
        let (contents, buffer, texture) = surface(device, 7);
        let apart = by_the_rule(device) && aliased(&texture);
        println!(
            "LINEAR-ATTACHMENT resolved into on {}: apart {apart}",
            device.name()
        );
        let queue = device.new_command_queue().unwrap();
        let samples = private(device, 4);
        let n = (W * H * TEXEL) as usize;
        pass(&queue, &[(&samples, LoadAction::Clear, Some(&texture))]);
        assert!(
            contents.read(0..n) == cleared(),
            "the resolve did not reach the pages"
        );
        assert_eq!(
            copies_announced(messages),
            usize::from(apart),
            "once a device"
        );
        drop((texture, buffer, samples));
        drop(contents);
    });
}

/// The passes of [`many_passes`]: a WebGPU frame that splits its pass at every indexed draw
/// (the `stencilMask` sample's first: 706), and the height of their targets.
const MANY: usize = 706;
const MANY_H: u64 = 240;

/// The byte between a [`many_target`]'s rows, which no pass writes.
const BETWEEN: u8 = 0xa5;

/// Where a [`many_target`]'s bytes lie.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ManyOver {
    /// A `Private` buffer, written and read through blits.
    Private,
    /// A `Shared` buffer.
    Shared,
    /// A no-copy buffer over host pages.
    HostPages,
}

/// The bytes a [`many_target`] of `width` texels a row, `row` bytes apart, holds first: a counting
/// pattern in the texels, [`BETWEEN`] between the rows.
fn many_first(width: u64, row: u64) -> Vec<u8> {
    let pattern = counting((row * MANY_H) as usize, 29);
    (0..(row * MANY_H) as usize)
        .map(|i| match (i as u64 % row) < width * TEXEL {
            true => pattern[i],
            false => BETWEEN,
        })
        .collect()
}

/// The texels of `bytes`, rows `row` bytes apart, packed.
fn many_texels(bytes: &[u8], width: u64, row: u64) -> Vec<u8> {
    (bytes.chunks(row as usize))
        .flat_map(|r| &r[..(width * TEXEL) as usize])
        .copied()
        .collect()
}

/// A blit encoder's commands, `encode` encoding them, in a command buffer that completes.
fn blit_with(
    queue: &CommandQueue,
    encode: impl FnOnce(&mut metal2vulkan_runtime::BlitCommandEncoder),
) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// A `BGRA8Unorm` render target of `width` x [`MANY_H`] texels, rows `row` bytes apart, over a
/// buffer of `over`'s holding [`many_first`] (the buffer, its mapping where it is over host
/// pages); `None`, printed, where the device imports no host memory.
fn many_target(
    queue: &CommandQueue,
    over: ManyOver,
    (width, row): (u64, u64),
) -> Option<(Buffer, Option<Mapping>, Texture)> {
    let device = queue.device();
    let first = many_first(width, row);
    let length = row * MANY_H;
    let (buffer, mapping) = match over {
        ManyOver::Private => {
            let buffer = device
                .new_buffer(length, ResourceOptions::private())
                .unwrap();
            let source = staged(device, &first);
            blit_with(queue, |b| {
                (b.copy_from_buffer_to_buffer(&source, 0, &buffer, 0, length)).unwrap()
            });
            (buffer, None)
        }
        ManyOver::Shared => {
            let buffer = staged(device, &first);
            (buffer, None)
        }
        ManyOver::HostPages => {
            let len = (length as usize).next_multiple_of(GUEST_PAGE);
            let mapping = Mapping::anonymous(len);
            mapping.write(0, &first);
            // SAFETY: the test drops the mapping after every buffer over it.
            let made = unsafe {
                no_copy(
                    device,
                    mapping.pointer(),
                    len,
                    ResourceOptions::default(),
                    None,
                )
            };
            match made {
                Ok(buffer) => (buffer, Some(mapping)),
                Err(error) if not_importable(&error) => {
                    eprintln!("LINEAR-ATTACHMENT: the device imports no host memory: {error:?}");
                    return None;
                }
                Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
            }
        }
    };
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, MANY_H, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = buffer.resource_options();
    let texture = buffer.new_texture(&d, 0, row).unwrap();
    Some((buffer, mapping, texture))
}

/// The first `length` bytes of `buffer`, of `over`'s, read through a blit where it is `Private`.
fn many_bytes(queue: &CommandQueue, over: ManyOver, buffer: &Buffer, length: u64) -> Vec<u8> {
    let device = queue.device();
    let shared = match over {
        ManyOver::Private => {
            let out = device
                .new_buffer(length, ResourceOptions::shared())
                .unwrap();
            blit_with(queue, |b| {
                (b.copy_from_buffer_to_buffer(buffer, 0, &out, 0, length)).unwrap()
            });
            out
        }
        ManyOver::Shared | ManyOver::HostPages => buffer.clone(),
    };
    Contents(shared).read(0..length as usize)
}

/// [`MANY`] passes in one command buffer over `texture`, each loading and storing it beside a
/// `Depth32Float` depth attachment it clears and stores, blending one half-transparent rectangle
/// of its own color over a place of its own (overlapping its neighbours'); committed and
/// completed. The device's work counts once they were encoded, before the commit.
fn many_passes(
    queue: &CommandQueue,
    pipeline: &RenderPipelineState,
    texture: &Texture,
) -> WorkCounts {
    let device = queue.device();
    let mut d =
        TextureDescriptor::texture_2d(PixelFormat::Depth32Float, texture.width(), MANY_H, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    let depth = device.new_texture(&d).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    for i in 0..MANY {
        let mut descriptor = RenderPassDescriptor::new();
        let color = &mut descriptor.color_attachments[0];
        color.texture = Some(texture.clone());
        color.load_action = LoadAction::Load;
        color.store_action = StoreAction::Store;
        descriptor.depth_attachment.texture = Some(depth.clone());
        descriptor.depth_attachment.load_action = LoadAction::Clear;
        descriptor.depth_attachment.store_action = StoreAction::Store;
        let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
        encoder.set_render_pipeline_state(pipeline).unwrap();
        let (x, y) = ((i % 12) as f32, ((i / 12) % 6) as f32);
        let (x0, y0) = (-1.0 + 0.125 * x, -1.0 + 0.25 * y);
        let corners = [x0, y0, (x0 + 0.5).min(1.0), (y0 + 0.75).min(1.0)];
        let rgba = [
            (i % 5) as f32 / 4.0,
            (i % 7) as f32 / 6.0,
            (i % 3) as f32 / 2.0,
            0.5,
        ];
        encoder.set_vertex_bytes(&floats(corners), 0).unwrap();
        encoder.set_fragment_bytes(&floats(rgba), 0).unwrap();
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

/// The texels [`many_passes`] leave in a private texture of `width` texels a row holding
/// [`many_first`]'s texels first, packed.
fn many_reference(queue: &CommandQueue, pipeline: &RenderPipelineState, width: u64) -> Vec<u8> {
    let device = queue.device();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, MANY_H, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let texture = device.new_texture(&d).unwrap();
    let packed = width * TEXEL;
    let first = many_first(width, packed);
    let source = staged(device, &first);
    let size = Size::new(width, MANY_H, 1);
    blit_with(queue, |b| {
        let origin = Origin::new(0, 0, 0);
        let (bpr, bpi) = (packed, packed * MANY_H);
        (b.copy_from_buffer_to_texture(
            &source,
            0,
            bpr,
            bpi,
            size,
            &texture,
            0,
            0,
            origin,
            BlitOption::None,
        ))
        .unwrap()
    });
    many_passes(queue, pipeline, &texture);
    let out = device
        .new_buffer(packed * MANY_H, ResourceOptions::shared())
        .unwrap();
    blit_with(queue, |b| {
        let origin = Origin::new(0, 0, 0);
        let (bpr, bpi) = (packed, packed * MANY_H);
        (b.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            origin,
            size,
            &out,
            0,
            bpr,
            bpi,
            BlitOption::None,
        ))
        .unwrap()
    });
    Contents(out).read(0..(packed * MANY_H) as usize)
}

/// [`many_passes`] over a target over `over`'s buffer of `width` texels a row, `row` bytes apart
/// (`merge_scratch.rs`): its texels equal those the same passes leave in a private texture, and
/// the bytes between its rows keep theirs. Where the device renders it apart by the rule
/// (`linear_merge.rs`), the passes form a run or two merged once each (`pass_runs.rs`), each
/// merge taking one scratch; where it is a shadow merged back (`shadow_merge.rs`), the passes
/// form one run after the first, which merges in place: kept before the first and the second,
/// and merged after the first and the last (`pass_runs.rs`).
/// Either way the device's memory grows by less than an eighth of a scratch a pass while they are
/// encoded.
fn many_passes_over(over: ManyOver, (width, row): (u64, u64)) {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let pipeline = rects_beside(
            device,
            PixelFormat::BGRA8Unorm,
            true,
            Some(PixelFormat::Depth32Float),
        );
        let expected = many_reference(&queue, &pipeline, width);
        let Some((buffer, mapping, texture)) = many_target(&queue, over, (width, row)) else {
            return;
        };
        let apart = by_the_rule(device) && aliased(&texture);
        // A shadow merged back: one over bytes another writer may write, all but a tracked
        // `Private` buffer's, whose shadow is copied back.
        let shadow = !aliased(&texture) && over != ManyOver::Private;
        let before = device.work_counts();
        let encoded = many_passes(&queue, &pipeline, &texture);
        let after = device.work_counts();
        let delta = |of: fn(&WorkCounts) -> u64| of(&after) - of(&before);
        let through = delta(|c| c.linear_passes_through_copies);
        let made = delta(|c| c.linear_merge_scratches_made);
        let reused = delta(|c| c.linear_merge_scratches_reused);
        let recorded = delta(|c| c.linear_merges_recorded);
        let taken_over = delta(|c| c.linear_merges_taken_over);
        let shadow_reused = delta(|c| c.shadow_merge_scratches_reused);
        let kept = delta(|c| c.shadow_texels_kept);
        let grown = encoded.device_memory_allocated - before.device_memory_allocated;
        // A merge of every texel's rows: its kept and changed parts.
        let scratch = 2 * width * TEXEL * MANY_H;
        println!(
            "LINEAR-ATTACHMENT many passes over {over:?} {width}x{MANY_H}, rows of {row}, on {}: \
             apart by the rule {apart}, shadow {shadow}; {through} passes through copies, linear \
             merges {recorded} recorded and {taken_over} taken over, linear scratches {made} made \
             and {reused} reused, shadow scratches {shadow_reused} reused, shadow texels kept \
             {kept}; device memory grew {grown} bytes while encoding",
            device.name()
        );
        // One run (`pass_runs.rs`), or two where the second pass's box reaches past the
        // first's (then one over the whole target), merged once each.
        if apart {
            assert_eq!(through, MANY as u64, "every pass renders the texture apart");
            assert!(
                recorded <= 2,
                "the passes merge once a run: {recorded} merges"
            );
            assert_eq!(
                recorded + taken_over,
                MANY as u64,
                "every later pass takes its run's"
            );
            assert_eq!(
                made + reused,
                recorded,
                "each run's merge takes one scratch"
            );
        }
        // One run of passes (`pass_runs.rs`): kept twice, every texel, before the first pass,
        // which merges in place, and before the second, which opens the run.
        if shadow {
            assert_eq!(
                kept,
                2 * width * MANY_H,
                "the passes keep twice, as one run"
            );
        }
        assert!(
            grown < scratch * MANY as u64 / 8,
            "encoding the passes grew the device's memory by {grown} bytes"
        );
        let pages = many_bytes(&queue, over, &buffer, row * MANY_H);
        let between = (0..pages.len())
            .filter(|&i| (i as u64 % row) >= width * TEXEL && pages[i] != BETWEEN)
            .count();
        assert_eq!(between, 0, "bytes between the rows changed");
        let got = many_texels(&pages, width, row);
        let differ = (0..got.len()).filter(|&i| got[i] != expected[i]).count();
        assert_eq!(differ, 0, "the texels differ from the private texture's");
        drop((texture, buffer));
        drop(mapping);
    });
}

/// Over a `Private` buffer, packed rows of 320 texels: 1280 bytes, a multiple of 256 (rows
/// rounded up to 256 bytes with none between them).
#[test]
fn many_passes_over_a_private_buffer_take_one_scratch() {
    many_passes_over(ManyOver::Private, (320, 1280));
}

/// Over a `Shared` buffer, packed rows of 300 texels: 1200 bytes, of no multiple of 256 (an
/// IOSurface's).
#[test]
fn many_passes_over_packed_rows_take_one_scratch() {
    many_passes_over(ManyOver::Shared, (300, 1200));
}

/// Over a no-copy buffer's host pages, packed rows of 300 texels (an IOSurface's over guest
/// pages).
#[test]
fn many_passes_over_host_pages_take_one_scratch() {
    many_passes_over(ManyOver::HostPages, (300, 1200));
}

/// Over a no-copy buffer's host pages, rows of 320 texels: 1280 bytes, a multiple of 256.
#[test]
fn many_passes_over_host_pages_of_whole_rows_take_one_scratch() {
    many_passes_over(ManyOver::HostPages, (320, 1280));
}
