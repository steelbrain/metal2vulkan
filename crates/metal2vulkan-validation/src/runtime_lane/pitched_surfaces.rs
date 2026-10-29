//! Device tests of textures over no-copy buffers at a guest browser's padded row pitches, on the
//! device the cfg selects, under the validation layer. Each texture is made as the consumer makes
//! a surface's (`new_texture_with_iosurface_plane`, render target and shader read usage) over a
//! no-copy buffer of private anonymous pages, at one of [`LAYOUTS`]: the browser's 1307x512
//! `BGRA8Unorm` tile with rows of 5248 bytes (its texels 5228), and the same tile and others with
//! rows padded to 16, 256 and 512 bytes, a small tile, a full-width strip, and an `RGBA16Float`
//! tile, so the rows' padding takes each power of two from 16 to 512 bytes; and tiles whose rows
//! lie one after another, of a height that is no multiple of 4, with the pages' later bytes past
//! their last row (a surface's next plane).
//!
//! Each texture is a shadow (no image over the pages: `vk_image` answers none) on the lane's
//! devices, which take the exact answers (`HostPageTexturePolicy::Shadow`); the tests named
//! `_in_place` run on a device of `HostPageTexturePolicy::InPlace`, where a texture that only reads
//! is an image aliasing the pages if the device's check passed and its pitch is taken. Every step
//! reads and writes Metal's bytes at Metal's places, and no write reaches the bytes between the
//! rows or past the last. The steps:
//!
//! - **reads**: a blit out, `get_bytes`, a fragment function's `read`, and its `sample` through a
//!   nearest sampler give the caller's texels; a linear sampler gives what it gives over a private
//!   texture of the same texels, byte for byte; the caller's later writes are what the next read
//!   reads;
//! - **writes**: a blit of a buffer into the texture, `replace_region` of a box at an odd origin,
//!   and a blit of a private texture's box into it land in the pages;
//! - **renders**: a pass that loads, blends a rectangle and stores, and one that clears, blends and
//!   stores, leave in the pages what they leave in a private texture of the same texels; one that
//!   loads, blends and stores `DontCare` leaves the pages as they were, as Apple's GPU leaves the
//!   attachment's earlier texels;
//! - **blits out**: a box at an odd origin copied into a private texture holds the pages' texels.
//!
//! [`a_fragment_function_s_writes_leave_the_bytes_between_rows`] makes the textures with shader
//! read and write usage instead, and writes every texel through a fragment function's `write`: the
//! pages' texels are a private texture's after the same writes, and no other byte changes.
//!
//! [`a_padded_texture_s_shadow_names_what_may_write_its_padding`] checks the shadow's reason over
//! an ordinary buffer's padded rows: a texture that only reads names its layout transitions, a
//! render target a render pass's load and store operations too.
//!
//! [`a_texture_that_only_reads_keeps_every_byte_of_its_pages`] makes a texture that only reads
//! (`ShaderRead` usage) over each layout and reads it first, before anything else touches the
//! pages. Then it samples, blits out, reads the caller's later bytes (the bytes between the rows
//! rewritten too), blits a buffer's box and a private texture's box in and replaces a box: after
//! each, every byte of the pages is checked. [`a_padded_texture_that_only_reads_is_a_shadow`]
//! counts its copies (`WorkCounts::shadow_copies`: some for its uses), and
//! [`a_padded_texture_that_renders_or_writes_is_a_shadow`] keeps a render target's, a written
//! texture's and an `Unknown` usage's shadow.
//! [`a_texture_that_only_reads_keeps_every_byte_of_its_pages_in_place`] takes the same steps
//! where such a texture is in place.
//!
//! [`a_texture_over_read_only_pages_reads_them`] makes a texture that only reads over pages the
//! process maps read-only: its first use reads their texels and writes none of their bytes;
//! [`a_texture_over_read_only_pages_reads_them_in_place`] likewise where it is in place.
//!
//! [`a_loaded_color_beside_a_cleared_depth_copies_inside_its_scissor`] draws under a scissor
//! beside a depth attachment the pass clears and stores, of the color's size and larger, whose
//! clear then reaches past the render area: the color's copies stay inside the scissor, and its
//! bytes are a private texture's.
//!
//! [`a_shadowed_color_beside_a_memoryless_scratch_blends_over_part_of_it`] blends over a shadowed
//! texture beside a memoryless scratch loaded and stored `DontCare`, over a render area smaller
//! than both: the bytes are a private texture's inside the render area and unchanged past it.
//!
//! [`a_pass_s_copies_in_share_their_barriers`] reads several shadowed textures in one pass that
//! loads another: their copies in share one barrier before and one after, and the pass records
//! as many pipeline barriers whatever the number of textures it reads.
//!
//! [`passes_on_one_shadow_count_once_a_command_buffer`] counts the passes on shadowed attachments
//! and the textures each command buffer takes them on (`WorkCounts::shadow_attachment_passes`,
//! `shadow_attachment_textures`).
//!
//! Printed per layout (`PITCHED-SURFACE`): the texture's form, the shadows the making counted
//! (`WorkCounts::shadowed_textures`), and the shadow's reason. Printed once
//! (`SHADOW-COPY-BACK`, [`a_pass_copies_in_and_back_only_the_texels_its_scissors_reach`]): whether
//! a shadow's copy back ran as a dispatch (`WorkCounts::shadow_copies_back_dispatched`).

use super::draw_work::pushes;
use super::first_use_writes::CHECK_WAIT;
use super::host_memory::{counting, no_copy, not_importable, Mapping, GUEST_PAGE};
use super::memoryless::memoryless_options;
use super::no_copy_textures::{blending, blending_beside_scratch, floats, rects_beside, staged};
use super::{
    layer_reported_nothing, layer_reported_nothing_at_all, validated_device, validated_device_with,
    Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::{DriverFact, HostPageTexturePolicy};
use metal2vulkan::metal::{
    BarrierScope, PixelFormat, PrimitiveType, RenderStages, SamplerMinMagFilter, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk::DriverId;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandQueue, Degradation, Device, IoSurfacePlane, LoadAction,
    Origin, PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    SamplerState, ScissorRect, Size, StoreAction, Texture, TextureDescriptor, Viewport, WorkCounts,
};

/// A surface's plane: its name, pixel format and texel bytes, width, height and bytes per row.
#[derive(Clone, Copy, Debug)]
struct Layout {
    name: &'static str,
    pf: PixelFormat,
    texel: u64,
    w: u64,
    h: u64,
    bpr: u64,
}

impl Layout {
    const fn bgra8(name: &'static str, w: u64, h: u64, bpr: u64) -> Layout {
        Layout {
            name,
            pf: PixelFormat::BGRA8Unorm,
            texel: 4,
            w,
            h,
            bpr,
        }
    }

    /// The bytes of a row's texels.
    fn row(&self) -> u64 {
        self.w * self.texel
    }

    /// The bytes the texels span, the last row's padding with them.
    fn len(&self) -> usize {
        (self.bpr * self.h) as usize
    }

    /// Whether the rows lie with bytes between them that are not texels.
    fn padded(&self) -> bool {
        self.bpr != self.row()
    }
}

/// The layouts, each its own test.
const LAYOUTS: [Layout; 11] = [
    // The browser's tile: rows padded to 64 bytes (5248 = 41 x 128).
    Layout::bgra8("1307x512/5248", 1307, 512, 5248),
    // Rows padded to 16 bytes (5232 = 327 x 16).
    Layout::bgra8("1307x512/5232", 1307, 512, 5232),
    // Rows padded to 256 bytes (5376 = 21 x 256).
    Layout::bgra8("1307x512/5376", 1307, 512, 5376),
    // Rows padded past 256, to 512 bytes (5632 = 11 x 512).
    Layout::bgra8("1307x512/5632", 1307, 512, 5632),
    // A small tile, rows padded to 64 bytes (1216 = 19 x 64).
    Layout::bgra8("300x200/1216", 300, 200, 1216),
    // A full-width strip, rows padded to 256 bytes (7680 = 15 x 512; 7652 of texels).
    Layout::bgra8("1913x96/7680", 1913, 96, 7680),
    // A narrow tile, rows padded to 32 bytes (1056 = 33 x 32).
    Layout::bgra8("257x64/1056", 257, 64, 1056),
    // An `RGBA16Float` tile, rows padded to 64 bytes (5312 = 83 x 64).
    Layout {
        name: "657x128/5312 RGBA16Float",
        pf: PixelFormat::RGBA16Float,
        texel: 8,
        w: 657,
        h: 128,
        bpr: 5312,
    },
    // Rows one after another (a surface's rows are multiples of 16 bytes), 510 of them: a
    // driver's image of 512 reaches past the texels.
    Layout::bgra8("1308x510/5232", 1308, 510, 5232),
    // Rows one after another, of a whole number of cache lines (1200 = 18.75 x 64).
    Layout::bgra8("300x200/1200", 300, 200, 1200),
    // An `RGBA16Float` tile, rows one after another, 126 of them.
    Layout {
        name: "658x126/5264 RGBA16Float",
        pf: PixelFormat::RGBA16Float,
        texel: 8,
        w: 658,
        h: 126,
        bpr: 5264,
    },
];

/// Texels of `layout` for `n` bytes, varied by `k`, each a value every read, sample and blend
/// keeps: `RGBA16Float` halves in [2^-7, 1), other formats any bits.
fn texels_of(layout: &Layout, n: usize, k: usize) -> Vec<u8> {
    if layout.texel == 8 {
        (0..n / 2)
            .flat_map(|i| (0x2000u16 + ((i * 97 + k * 13) % 0x1c00) as u16).to_le_bytes())
            .collect()
    } else {
        counting(n, k)
    }
}

/// The texels of `bytes`, rows `layout.bpr` apart: what a texture over them reads and writes, row
/// after row.
fn rows(layout: &Layout, bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks(layout.bpr as usize)
        .take(layout.h as usize)
        .flat_map(|r| &r[..layout.row() as usize])
        .copied()
        .collect()
}

/// `texels` (rows of `layout.row()` bytes) laid into `bytes` at rows `layout.bpr` apart, the bytes
/// between the rows left.
fn lay(layout: &Layout, bytes: &mut [u8], texels: &[u8]) {
    let row = layout.row() as usize;
    for (y, texels) in texels.chunks(row).enumerate() {
        let at = y * layout.bpr as usize;
        bytes[at..at + row].copy_from_slice(texels);
    }
}

/// The texels (`texel` bytes each) of `a` and `b` that differ, and the first.
fn differing(a: &[u8], b: &[u8], texel: u64) -> (usize, Option<usize>) {
    let mut count = 0;
    let mut first = None;
    for (i, (x, y)) in a
        .chunks_exact(texel as usize)
        .zip(b.chunks_exact(texel as usize))
        .enumerate()
    {
        if x != y {
            count += 1;
            first.get_or_insert(i);
        }
    }
    (count, first)
}

/// Where `now` differs from `expected` outside `layout`'s texels, in words: how many bytes, the
/// rows they lie in, and the first, with its place in its row and the bytes it holds.
fn padding_changes(layout: &Layout, now: &[u8], expected: &[u8]) -> String {
    let (bpr, row) = (layout.bpr as usize, layout.row() as usize);
    let changed: Vec<usize> = (0..now.len().min(expected.len()))
        .filter(|&i| now[i] != expected[i] && (i % bpr >= row || i / bpr >= layout.h as usize))
        .collect();
    let Some(&first) = changed.first() else {
        return "none outside the texels".to_owned();
    };
    let rows: std::collections::BTreeSet<usize> = changed.iter().map(|i| i / bpr).collect();
    let end = (first + 8).min(now.len());
    format!(
        "{} bytes outside the texels in {} rows (rows {:?}..={:?}), the first at row {} byte {} \
         (a row's texels end at byte {row}), now {:02x?}, was {:02x?}",
        changed.len(),
        rows.len(),
        rows.first(),
        rows.last(),
        first / bpr,
        first % bpr,
        &now[first..end],
        &expected[first..end]
    )
}

/// The bytes of a `Shared` buffer, which no command uses now.
fn bytes_of(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's `length` bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// Commit `command_buffer` and wait for it, which must complete without an error.
fn run(command_buffer: &metal2vulkan_runtime::CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// `texture`'s texels, blitted out into a new `Shared` buffer at tight rows.
fn blit_out(queue: &CommandQueue, layout: &Layout, texture: &Texture) -> Vec<u8> {
    let (row, len) = (layout.row(), layout.row() * layout.h);
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
        Size::new(layout.w, layout.h, 1),
        &out,
        0,
        row,
        len,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    bytes_of(&out)
}

/// `texels` (tight rows) blitted into `texture` from a `Shared` buffer.
fn blit_in(queue: &CommandQueue, layout: &Layout, texture: &Texture, texels: &[u8]) {
    let source = staged(queue.device(), texels);
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        layout.row(),
        texels.len() as u64,
        Size::new(layout.w, layout.h, 1),
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
}

/// A blit of `size` texels from `source` at `from` into `destination` at `to`.
fn blit_box(
    queue: &CommandQueue,
    (source, from): (&Texture, Origin),
    (destination, to): (&Texture, Origin),
    size: Size,
) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_texture(source, 0, 0, from, size, destination, 0, 0, to)
        .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
}

/// A private texture of `layout`'s format and size holding `texels` (tight rows).
fn private_with(queue: &CommandQueue, layout: &Layout, texels: &[u8]) -> Texture {
    let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let texture = queue.device().new_texture(&d).unwrap();
    blit_in(queue, layout, &texture, texels);
    texture
}

/// The owned fixture's `tob_rect` over the whole target with `fragment` (`tob_read` or
/// `tob_sample`), into one attachment of `pf`.
pub(super) fn reading(device: &Device, pf: PixelFormat, fragment: &str) -> RenderPipelineState {
    let library = device
        .new_library_with_air(&llvm_assemble(&engine_dir().join("tob_renders.ll")).unwrap())
        .unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut d = RenderPipelineDescriptor::new(function("tob_rect"));
    d.fragment = Some(function(fragment));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(pf));
    device
        .new_render_pipeline_state(&d, &PipelineOptions::default())
        .unwrap()
}

/// What a draw of `pipeline` over a whole target of `layout`'s format and size reads from
/// `texture` (through `sampler`, where given): the target's texels, tight rows.
fn drawn(
    queue: &CommandQueue,
    layout: &Layout,
    pipeline: &RenderPipelineState,
    texture: &Texture,
    sampler: Option<&SamplerState>,
) -> Vec<u8> {
    let (command_buffer, target) = draw_reading(queue, layout, pipeline, texture, sampler);
    run(&command_buffer);
    blit_out(queue, layout, &target)
}

/// [`drawn`]'s command buffer, encoded and not committed, and its target.
fn draw_reading(
    queue: &CommandQueue,
    layout: &Layout,
    pipeline: &RenderPipelineState,
    texture: &Texture,
    sampler: Option<&SamplerState>,
) -> (metal2vulkan_runtime::CommandBuffer, Texture) {
    let device = queue.device();
    let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    let target = device.new_texture(&d).unwrap();
    let corners = staged(device, &floats([-1.0, -1.0, 1.0, 1.0]));
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
    encoder.set_fragment_texture(Some(texture), 0).unwrap();
    if let Some(sampler) = sampler {
        encoder
            .set_fragment_sampler_state(Some(sampler), 0)
            .unwrap();
    }
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    (command_buffer, target)
}

/// A pass over `texture` that loads it (clears it to (0.1, 0.2, 0.3, 0.4) where `clear`), blends
/// `color` over the rectangle `corners` and stores (by `store`).
fn blend_pass(
    queue: &CommandQueue,
    pipeline: &RenderPipelineState,
    (texture, store): (&Texture, StoreAction),
    clear: bool,
    (corners, color): ([f32; 4], [f32; 4]),
) {
    let device = queue.device();
    let corners = staged(device, &floats(corners));
    let color = staged(device, &floats(color));
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = if clear {
        LoadAction::Clear
    } else {
        LoadAction::Load
    };
    attachment.store_action = store;
    attachment.clear_color = ClearColor::new(0.1, 0.2, 0.3, 0.4);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
    encoder.set_fragment_buffer(Some(&color), 0, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
}

/// A sampler of `filter`, clamped to the edge.
pub(super) fn sampler(device: &Device, filter: SamplerMinMagFilter) -> SamplerState {
    let mut d = SamplerDescriptor::new();
    d.min_filter = filter;
    d.mag_filter = filter;
    device.new_sampler_state(&d).unwrap()
}

/// A texture of `layout` over a no-copy buffer of new anonymous pages, made as a surface's with
/// `usage`: the pages hold a pattern of their own, the texels `texels_of(.., 1)` laid at the rows;
/// with the buffer, the mapping and the pages as written. `None` where the device imports no host
/// memory. It is a shadow, whatever its usage: counted so, and `vk_image` answers no image; on a
/// device that takes such textures in place ([`in_place_form`]) one that only reads may be in
/// place instead, counted so, and `vk_image` answers its image.
fn surface(device: &Device, layout: &Layout, usage: TextureUsage) -> Option<Surface> {
    let len = layout.len().next_multiple_of(GUEST_PAGE);
    let mapping = Mapping::anonymous(len);
    // The pages: texels a blend keeps finite, and the bytes between the rows a pattern of
    // their own, which no write through the texture reaches.
    let mut pages = counting(len, 7);
    lay(
        layout,
        &mut pages,
        &texels_of(layout, (layout.row() * layout.h) as usize, 1),
    );
    mapping.write(0, &pages);
    // SAFETY: the mapping outlives the buffer and the texture (dropped first, by `Surface`).
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
            eprintln!("PITCHED-SURFACE: the device imports no host memory: {error:?}");
            return None;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
    d.usage = usage;
    d.resource_options = buffer.resource_options();
    let before = device.work_counts();
    let texture = buffer
        .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, layout.bpr, false))
        .unwrap();
    let what = format!("{} ({}, {usage:?})", layout.name, layout.pf.name());
    let (shadowed, aliased) = in_place_form(device, &before, &texture, usage, &what);
    Some(Surface {
        texture,
        buffer,
        mapping,
        pages,
        len,
        what,
        shadowed,
        aliased,
    })
}

/// The form a texture over host pages of `usage` took, its making counted from `before`: the
/// shadows counted, and whether it is in place. A shadow, `vk_image` answering none, on a device
/// of `HostPageTexturePolicy::Shadow` and for any usage that renders or writes; else one of
/// `ShaderRead` usage alone may be in place, counted so (`host_textures_in_place`), and `vk_image`
/// answers its image.
fn in_place_form(
    device: &Device,
    before: &WorkCounts,
    texture: &Texture,
    usage: TextureUsage,
    what: &str,
) -> (u64, bool) {
    let counts = device.work_counts();
    let shadowed = counts.shadowed_textures - before.shadowed_textures;
    let in_place = counts.host_textures_in_place - before.host_textures_in_place;
    let policy = device.compile_device().policy().host_page_textures;
    let may_alias = policy == HostPageTexturePolicy::InPlace && usage == TextureUsage::SHADER_READ;
    // SAFETY: the handle is only compared, while the texture lives.
    let aliased = unsafe { texture.vk_image() }.is_some();
    assert!(
        !aliased || may_alias,
        "{what}: a texture over host pages answered an image"
    );
    let expected = match aliased {
        true => (0, 1),
        false => (1, 0),
    };
    assert_eq!(
        (shadowed, in_place),
        expected,
        "{what}: its form not counted"
    );
    (shadowed, aliased)
}

/// A texture over a no-copy buffer's pages ([`surface`]).
struct Surface {
    texture: Texture,
    buffer: Buffer,
    mapping: Mapping,
    /// The pages as written before the texture was made.
    pages: Vec<u8>,
    /// The pages' bytes.
    len: usize,
    /// The layout, format and usage, in words.
    what: String,
    /// The shadows its making counted.
    shadowed: u64,
    /// Whether it is in place: an image aliasing the pages.
    aliased: bool,
}

impl Surface {
    /// Check that the pages hold `expected`: its texels (by `layout`), then every byte.
    fn check(&self, layout: &Layout, step: &str, expected: &[u8]) {
        let what = &self.what;
        let now = self.mapping.read(0..self.len);
        let (count, first) = differing(&rows(layout, &now), &rows(layout, expected), layout.texel);
        assert!(
            count == 0,
            "{what}, {step}: {count} texels in the pages differ from Metal's, the first at texel \
             {first:?}"
        );
        assert!(
            now == expected,
            "{what}, {step}: the bytes between the rows (or past the last) changed: {}",
            padding_changes(layout, &now, expected)
        );
    }

    /// Print the shadows counted and the shadow's reason (`PITCHED-SURFACE`), and drop the
    /// texture, then the buffer, then the pages.
    fn report(self, device: &Device, messages: &Messages) {
        let reason = messages
            .take_notices()
            .into_iter()
            .filter_map(|message| message.degradation)
            .find(|notice| notice.degradation == Degradation::ShadowedTexture)
            .map(|notice| format!(": {}", notice.reason))
            .unwrap_or_default();
        eprintln!(
            "PITCHED-SURFACE on {}: {}: {} (shadows {}){reason}",
            device.name(),
            self.what,
            if self.aliased { "in place" } else { "shadowed" },
            self.shadowed,
        );
        let Surface {
            texture,
            buffer,
            mapping,
            ..
        } = self;
        drop((texture, buffer));
        drop(mapping);
    }
}

/// The module docs' steps over a texture of `layout` over a no-copy buffer: each read gives the
/// pages' texels (or the private texture's answer), each write lands in the pages, and the bytes
/// between the rows stay the caller's.
fn reads_writes_and_renders_as_metal(layout: Layout) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let texel = layout.texel;
        let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let Some(surface) = surface(&device, &layout, usage) else {
            return;
        };
        let (texture, what) = (&surface.texture, &surface.what);
        // The pages as Metal leaves them, after each step.
        let mut expected = surface.pages.clone();
        let check = |step: &str, expected: &[u8]| surface.check(&layout, step, expected);
        let read = |step: &str, ours: &[u8], theirs: &[u8]| {
            let (count, first) = differing(ours, theirs, texel);
            assert!(
                count == 0,
                "{what}, {step}: {count} texels differ, the first at texel {first:?}"
            );
        };

        // Reads of the caller's bytes as they were before the texture was made.
        let texels = rows(&layout, &expected);
        read("a blit out", &blit_out(&queue, &layout, texture), &texels);
        let mut got = vec![0u8; texels.len()];
        let whole = Region::new_2d(0, 0, layout.w, layout.h);
        texture
            .get_bytes(&mut got, layout.row(), 0, whole, 0, 0)
            .unwrap();
        read("get_bytes", &got, &texels);
        let reads = reading(&device, layout.pf, "tob_read");
        read(
            "a fragment function's read",
            &drawn(&queue, &layout, &reads, texture, None),
            &texels,
        );
        let samples = reading(&device, layout.pf, "tob_sample");
        let nearest = sampler(&device, SamplerMinMagFilter::Nearest);
        let linear = sampler(&device, SamplerMinMagFilter::Linear);
        read(
            "a nearest sample",
            &drawn(&queue, &layout, &samples, texture, Some(&nearest)),
            &texels,
        );
        let private = private_with(&queue, &layout, &texels);
        read(
            "a linear sample, against a private texture's",
            &drawn(&queue, &layout, &samples, texture, Some(&linear)),
            &drawn(&queue, &layout, &samples, &private, Some(&linear)),
        );
        check("the reads", &expected);

        // The caller's later bytes, read by a sample and a blit.
        let later = texels_of(&layout, texels.len(), 2);
        lay(&layout, &mut expected, &later);
        surface.mapping.write(0, &expected);
        read(
            "a nearest sample of the caller's later bytes",
            &drawn(&queue, &layout, &samples, texture, Some(&nearest)),
            &later,
        );
        read(
            "a blit out of the caller's later bytes",
            &blit_out(&queue, &layout, texture),
            &later,
        );

        // Writes: a blit of a buffer, `replace_region` of a box, a blit of a private texture's box.
        let written = texels_of(&layout, later.len(), 3);
        blit_in(&queue, &layout, texture, &written);
        lay(&layout, &mut expected, &written);
        check("a blit's write", &expected);
        let (bx, by, bw, bh) = (3, 5, layout.w / 2 + 1, layout.h / 3 + 1);
        let boxed = texels_of(&layout, (bw * bh * texel) as usize, 4);
        texture
            .replace_region(Region::new_2d(bx, by, bw, bh), 0, 0, &boxed, bw * texel, 0)
            .unwrap();
        for (y, row) in boxed.chunks((bw * texel) as usize).enumerate() {
            let at = ((by + y as u64) * layout.bpr + bx * texel) as usize;
            expected[at..at + row.len()].copy_from_slice(row);
        }
        check("replace_region", &expected);
        let source = private_with(&queue, &layout, &texels_of(&layout, later.len(), 5));
        let (fx, fy, tx, ty) = (7, 1, layout.w / 3, 2);
        let size = Size::new(layout.w / 2, layout.h / 2, 1);
        blit_box(
            &queue,
            (&source, Origin::new(fx, fy, 0)),
            (texture, Origin::new(tx, ty, 0)),
            size,
        );
        let from = blit_out(&queue, &layout, &source);
        for y in 0..size.height {
            let start = ((fy + y) * layout.row() + fx * texel) as usize;
            let row = &from[start..start + (size.width * texel) as usize];
            let at = ((ty + y) * layout.bpr + tx * texel) as usize;
            expected[at..at + row.len()].copy_from_slice(row);
        }
        check("a blit of a private texture's box", &expected);

        // Renders that store, against a private texture of the same texels.
        let blend = blending(&device, layout.pf);
        let reference = private_with(&queue, &layout, &rows(&layout, &expected));
        let left = ([-1.0, -1.0, 0.0, 1.0], [0.2, 0.4, 0.6, 0.5]);
        let right = ([0.0, -1.0, 1.0, 1.0], [0.9, 0.6, 0.1, 0.25]);
        for (step, clear, rect) in [
            ("a pass that loads, blends and stores", false, left),
            ("a pass that clears, blends and stores", true, right),
        ] {
            blend_pass(
                &queue,
                &blend,
                (&reference, StoreAction::Store),
                clear,
                rect,
            );
            blend_pass(&queue, &blend, (texture, StoreAction::Store), clear, rect);
            lay(
                &layout,
                &mut expected,
                &blit_out(&queue, &layout, &reference),
            );
            check(step, &expected);
        }
        // A pass that does not store leaves the pages as they were.
        blend_pass(
            &queue,
            &blend,
            (texture, StoreAction::DontCare),
            false,
            left,
        );
        check("a pass that loads, blends and stores DontCare", &expected);

        // A blit of the texture's box out into a private texture.
        let (fx, fy) = (layout.w / 4 + 1, 3);
        let size = Size::new(layout.w / 2, layout.h / 2, 1);
        let out = private_with(&queue, &layout, &vec![0; later.len()]);
        blit_box(
            &queue,
            (texture, Origin::new(fx, fy, 0)),
            (&out, Origin::new(0, 0, 0)),
            size,
        );
        let copied = blit_out(&queue, &layout, &out);
        let pages = rows(&layout, &expected);
        for y in 0..size.height {
            let at = (y * layout.row()) as usize;
            let from = ((fy + y) * layout.row() + fx * texel) as usize;
            let width = (size.width * texel) as usize;
            assert!(
                copied[at..at + width] == pages[from..from + width],
                "{what}: a blit of the texture's box out: row {y} differs from the pages'"
            );
        }
        check("the last blit", &expected);
        surface.report(&device, &messages);
    }
    layer_reported_nothing_at_all(&messages);
}

/// A texture of `layout` over a no-copy buffer whose first use reads it (a fragment function's
/// `read`), before any command reads or writes the pages another way: its shadow's copy in reads
/// the pages' texels. A lapse fails here: the read differs from the texels, or the pages change.
fn first_use_reads_the_bytes(device: &Device, messages: &Messages, layout: Layout) {
    {
        let queue = device.new_command_queue().unwrap();
        let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let Some(surface) = surface(device, &layout, usage) else {
            return;
        };
        let texels = rows(&layout, &surface.pages);
        let reads = reading(device, layout.pf, "tob_read");
        let read = drawn(&queue, &layout, &reads, &surface.texture, None);
        let (count, first) = differing(&read, &texels, layout.texel);
        assert!(
            count == 0,
            "{}, its first use: {count} texels read differ from the pages', the first at texel \
             {first:?}",
            surface.what
        );
        surface.check(&layout, "its first use", &surface.pages);
        surface.report(device, messages);
    }
    layer_reported_nothing(messages, layout.name);
}

/// [`first_use_reads_the_bytes`] over each of [`LAYOUTS`], on one device.
#[test]
fn a_texture_s_first_use_reads_the_bytes_it_was_made_over() {
    on_shared_device(|device, messages| {
        for layout in LAYOUTS {
            first_use_reads_the_bytes(device, messages, layout);
        }
    });
}

/// Packed rows (multiples of 256 bytes): a presented 1080p surface's, and a tile's.
const PACKED: [Layout; 2] = [
    Layout::bgra8("1920x1080/7680", 1920, 1080, 7680),
    Layout::bgra8("1024x512/4096", 1024, 512, 4096),
];

/// A texture that only reads, of `layout`, over a no-copy buffer whose pages the process maps
/// read-only by its first use: made read-only after the buffer was made (Linux refuses read-only
/// pages at the import, `a_read_only_page_is_refused_on_linux_and_read_elsewhere`, but not later),
/// or, where `from_the_start`, before it (a guest's pages the host maps read-only, which MoltenVK
/// imports in place). It is a shadow, whose first use copies the pages' texels in and stores
/// nothing into them, or in place ([`in_place_form`]), whose first move the device's check showed
/// keeping every byte: a store into the read-only pages would end the process (`SIGBUS` on macOS,
/// `SIGSEGV` on Linux). The read gives the pages' texels and every byte of the pages stays.
/// Printed (`FIRST-USE-READ-ONLY`). `None` where the device makes no buffer over the pages.
fn first_use_over_read_only_pages(
    device: &Device,
    messages: &Messages,
    layout: Layout,
    from_the_start: bool,
) -> Option<()> {
    let queue = device.new_command_queue().unwrap();
    let reads = reading(device, layout.pf, "tob_read");
    let len = layout.len().next_multiple_of(GUEST_PAGE);
    let mapping = Mapping::anonymous(len);
    let mut pages = counting(len, 7);
    lay(
        &layout,
        &mut pages,
        &texels_of(&layout, (layout.row() * layout.h) as usize, 1),
    );
    mapping.write(0, &pages);
    if from_the_start {
        mapping.protect_read_only();
    }
    // SAFETY: the mapping outlives the buffer and the texture (dropped first), and nothing writes
    // the buffer.
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
            eprintln!("FIRST-USE-READ-ONLY: no buffer over the pages: {error:?}");
            return None;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    if !from_the_start {
        mapping.protect_read_only();
    }
    let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = buffer.resource_options();
    let before = device.work_counts();
    let texture = buffer
        .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, layout.bpr, false))
        .unwrap();
    let what = format!(
        "{} ({}), read-only {}",
        layout.name,
        layout.pf.name(),
        if from_the_start {
            "from the start"
        } else {
            "after the buffer"
        }
    );
    let (_, aliased) = in_place_form(device, &before, &texture, d.usage, &what);
    let read = drawn(&queue, &layout, &reads, &texture, None);
    let (count, first) = differing(&read, &rows(&layout, &pages), layout.texel);
    eprintln!(
        "FIRST-USE-READ-ONLY on {}: {what}: {}, {count} texels read differ",
        device.name(),
        if aliased { "in place" } else { "shadowed" },
    );
    assert!(
        count == 0,
        "{what}: {count} texels read differ from the pages', the first at texel {first:?}"
    );
    let now = mapping.read(0..len);
    assert!(
        now == pages,
        "{what}: the read-only pages changed: {}",
        padding_changes(&layout, &now, &pages)
    );
    layer_reported_nothing(messages, &what);
    drop((texture, buffer));
    drop(mapping);
    Some(())
}

/// [`first_use_over_read_only_pages`] over [`PACKED`], with the pages made read-only after the
/// buffer, and off Linux also before it, on one device. On Linux only lavapipe and NVIDIA's driver
/// keep an import running once its pages lose write permission (`HostMemory::new`'s Linux
/// contract: RADV failed the command buffers submitted after it), so only they take the case.
#[test]
fn a_texture_over_read_only_pages_reads_them() {
    on_shared_device(read_only_pages_read);
}

/// [`a_texture_over_read_only_pages_reads_them`] on a device that takes a texture that only reads
/// in place where its check passed.
#[test]
fn a_texture_over_read_only_pages_reads_them_in_place() {
    on_in_place_device(read_only_pages_read);
}

/// [`first_use_over_read_only_pages`] over [`PACKED`] on `device`, as
/// [`a_texture_over_read_only_pages_reads_them`] says.
fn read_only_pages_read(device: &Device, messages: &Messages) {
    {
        let driver = device.properties().description.driver_id;
        let keeps_running = matches!(
            driver,
            Some(DriverId::MESA_LLVMPIPE | DriverId::NVIDIA_PROPRIETARY)
        );
        if cfg!(target_os = "linux") && !keeps_running {
            eprintln!(
                "FIRST-USE-READ-ONLY on {}: {driver:?} is held to the writable pages of \
                 HostMemory::new's Linux contract",
                device.name()
            );
            return;
        }
        for layout in PACKED {
            for from_the_start in [false, true] {
                if from_the_start && cfg!(target_os = "linux") {
                    continue;
                }
                if first_use_over_read_only_pages(device, messages, layout, from_the_start)
                    .is_none()
                {
                    return;
                }
            }
        }
    }
}

/// A texture over a buffer's padded rows (the browser's tile, [`LAYOUTS`]' first) is a shadow on
/// every device, its buffer an ordinary one's; and its `shadowed-texture` notice names what Vulkan
/// lets write the padding: for a texture that only reads, its image's layout transitions (the
/// first move from the layout it is made in), never a render pass, which no image of its takes;
/// for a render target, a render pass's load and store operations too. Each usage on a device of its own, which announces the kind once.
#[test]
fn a_padded_texture_s_shadow_names_what_may_write_its_padding() {
    let layout = LAYOUTS[0];
    assert!(layout.padded(), "{}", layout.name);
    for (usage, renders) in [
        (TextureUsage::SHADER_READ, false),
        (
            TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            true,
        ),
    ] {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        {
            let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
            // What the device announced as it was made.
            messages.take_notices();
            let buffer = device
                .new_buffer(layout.bpr * layout.h, ResourceOptions::default())
                .unwrap();
            let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
            d.usage = usage;
            d.resource_options = buffer.resource_options();
            let texture = buffer.new_texture(&d, 0, layout.bpr).unwrap();
            // SAFETY: the handle is only compared, while the texture lives.
            let aliased = unsafe { texture.vk_image() }.is_some();
            assert!(!aliased, "{} ({usage:?}): aliased", layout.name);
            let reason = messages
                .take_notices()
                .into_iter()
                .filter_map(|message| message.degradation)
                .find(|notice| notice.degradation == Degradation::ShadowedTexture)
                .map(|notice| notice.reason)
                .unwrap_or_else(|| {
                    panic!("{} ({usage:?}): no shadowed-texture notice", layout.name)
                });
            eprintln!(
                "PITCHED-SURFACE on {}: {} ({usage:?}), an ordinary buffer's: shadowed: {reason}",
                device.name(),
                layout.name
            );
            let rows = format!(
                "its rows lie {} bytes apart, {} bytes of texels each",
                layout.bpr,
                layout.row()
            );
            assert!(reason.contains(&rows), "{usage:?}: {reason}");
            assert!(
                reason.contains("its layout transitions read and write"),
                "{usage:?}: {reason}"
            );
            assert_eq!(
                reason.contains("a render pass's load and store operations write back"),
                renders,
                "{usage:?}: {reason}"
            );
            drop((texture, buffer));
        }
        layer_reported_nothing_at_all(&messages);
    }
}

/// A pass over a shadowed texture (the browser's padded tile over an ordinary buffer, a shadow on
/// every device) that loads it and draws nothing leaves its texels as they were, as Metal's leaves
/// the tiles no draw reaches: it takes no copy in and no copy back (`WorkCounts::shadow_copies`),
/// whether it stores the attachment or stores it `DontCare`, and the buffer keeps every byte. A
/// pass that draws a rectangle copies in before it, keeps the image after the copy in, and merges
/// back what it drew in place of a copy back, as the CPU may write the `Shared` buffer's bytes:
/// three copies.
#[test]
fn a_pass_that_draws_nothing_copies_no_shadow() {
    let layout = LAYOUTS[0];
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let len = layout.bpr * layout.h;
        let buffer = device.new_buffer(len, ResourceOptions::default()).unwrap();
        let bytes = texels_of(&layout, len as usize, 3);
        let contents = buffer.contents().unwrap();
        // SAFETY: a new `Shared` buffer of `len` bytes, which no command uses yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = buffer.resource_options();
        let texture = buffer.new_texture(&d, 0, layout.bpr).unwrap();
        // SAFETY: the handle is only compared, while the texture lives.
        assert!(unsafe { texture.vk_image() }.is_none(), "a shadow");
        let copies = || device.work_counts().shadow_copies;
        for store in [StoreAction::Store, StoreAction::DontCare] {
            let before = copies();
            let mut pass = RenderPassDescriptor::new();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(texture.clone());
            attachment.load_action = LoadAction::Load;
            attachment.store_action = store;
            let command_buffer = queue.command_buffer().unwrap();
            let encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
            assert_eq!(copies() - before, 0, "{store:?}: a pass drawing nothing");
            assert!(
                bytes_of(&buffer) == bytes,
                "{store:?}: the buffer's bytes changed"
            );
        }
        let before = copies();
        let blend = blending(&device, layout.pf);
        let rect = ([-1.0, -1.0, 0.0, 1.0], [0.25, 0.5, 0.75, 0.5]);
        blend_pass(&queue, &blend, (&texture, StoreAction::Store), false, rect);
        assert_eq!(
            copies() - before,
            3,
            "a pass drawing a rectangle: in, kept and merged back"
        );
        drop((texture, buffer));
    }
    layer_reported_nothing_at_all(&messages);
}

/// A pass that loads a shadowed texture (the browser's padded tile over an ordinary buffer, a
/// shadow on every device) and draws under a scissor copies in and back only the texels inside it
/// (`WorkCounts::shadow_texels_copied_in`, `shadow_texels_copied_back`): a fragment outside the
/// scissor has no coverage, and the pass's load and store write back what they read. The buffer
/// holds the blend inside the scissor, as a private texture of the same texels blended alike
/// does, and every other byte as it was; a later draw reading every texel of the texture reads
/// the buffer's, none the image kept from before. Without a scissor, both copies move every
/// texel. The CPU may write the `Shared` buffer's bytes, so its texture's copy back is a merge,
/// which moves no texel back as a copy and dispatches no copy back; its keep moves the texels
/// inside the scissor alone (`WorkCounts::shadow_texels_kept`). The merge fetches the changed
/// texels from the shadow, copying none (`WorkCounts::shadow_merges_fetched`): through its
/// unsigned-integer view on a device that takes the copy back's dispatch
/// (`DriverFact::ShadowCopyBackDispatchFaster`), else through its own view where the texture is
/// `BGRA8Unorm` or `RGBA8Unorm`, whose bytes the merge rebuilds from their unsigned normalized
/// values; another format's merge copies them. On a device that also takes a copy in with its
/// keep as one dispatch
/// (`DriverFact::ShadowCopyInDispatchFaster`), the copy in and the keep, of the same texels, are
/// that dispatch (`WorkCounts::shadow_copies_in_dispatched`, one of
/// `WorkCounts::shadow_merge_copies_dispatched`, printed as `SHADOW-COPY-BACK`); on one that takes
/// the copy back's alone, the keep copies the texels into an image the merge fetches them from
/// beside the shadow (`WorkCounts::shadow_merges_kept_in_images`), which makes no scratch.
/// Neither merge zeroes its scratch (`WorkCounts::shadow_merge_scratches_zeroed`). Where the
/// device pushes set 0, the keep's and the merge's dispatches push their descriptors and make no
/// descriptor pool; elsewhere each makes one (`WorkCounts::owned_descriptor_pools`).
#[test]
fn a_pass_copies_in_and_back_only_the_texels_its_scissors_reach() {
    let layout = LAYOUTS[0];
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let len = layout.bpr * layout.h;
        let buffer = device.new_buffer(len, ResourceOptions::default()).unwrap();
        let bytes = texels_of(&layout, len as usize, 5);
        let contents = buffer.contents().unwrap();
        // SAFETY: a new `Shared` buffer of `len` bytes, which no command uses yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = buffer.resource_options();
        let texture = buffer.new_texture(&d, 0, layout.bpr).unwrap();
        // SAFETY: the handle is only compared, while the texture lives.
        assert!(unsafe { texture.vk_image() }.is_none(), "a shadow");
        let private = private_with(&queue, &layout, &rows(&layout, &bytes));
        let blend = blending(&device, layout.pf);
        let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
        let color = staged(&device, &floats([0.25, 0.5, 0.75, 0.5]));
        let scissor = ScissorRect::new(100, 40, 300, 200);
        let draw = |target: &Texture, scissor: Option<ScissorRect>| {
            let mut pass = RenderPassDescriptor::new();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Load;
            attachment.store_action = StoreAction::Store;
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&blend).unwrap();
            if let Some(scissor) = scissor {
                encoder.set_scissor_rect(scissor).unwrap();
            }
            encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
            encoder.set_fragment_buffer(Some(&color), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
        };
        // Each pass's merge fetches the shadow's texels (`WorkCounts::shadow_merges_fetched`):
        // through the shadow's unsigned-integer view where the device runs a copy back faster as
        // a dispatch (`DriverFact::ShadowCopyBackDispatchFaster`), else through its own view where
        // the texture's format is one whose bytes the merge rebuilds from its unsigned normalized
        // values (`BGRA8Unorm`, `RGBA8Unorm`); another format's merge copies them.
        let facts = &device.properties().driver_facts;
        let dispatches = facts.contains(&DriverFact::ShadowCopyBackDispatchFaster);
        let copies_in = facts.contains(&DriverFact::ShadowCopyInDispatchFaster);
        let unorm = matches!(layout.pf, PixelFormat::BGRA8Unorm | PixelFormat::RGBA8Unorm);
        let moved = || {
            let counts = device.work_counts();
            (
                counts.shadow_texels_copied_in,
                counts.shadow_texels_copied_back,
                counts.shadow_copies_back_dispatched,
                counts.shadow_merge_copies_dispatched,
                counts.shadow_texels_kept,
                counts.shadow_merges_fetched,
                counts.owned_descriptor_pools,
                counts.shadow_copies_in_dispatched,
                counts.shadow_merges_kept_in_images,
            )
        };
        let since = |before: (u64, u64, u64, u64, u64, u64, u64, u64, u64)| {
            let now = moved();
            (
                now.0 - before.0,
                now.1 - before.1,
                now.2 - before.2,
                now.3 - before.3,
                now.4 - before.4,
                now.5 - before.5,
                now.6 - before.6,
                now.7 - before.7,
                now.8 - before.8,
            )
        };
        // Merged back: no texel goes back as a copy, and no copy back is dispatched. Where the
        // device takes the copy back's dispatch, the merge fetches through the unsigned-integer
        // view, its keep then the copy in's dispatch where the device takes that too
        // (`DriverFact::ShadowCopyInDispatchFaster`), else a copy into a kept image; elsewhere a
        // fetch through the shadow's own view keeps the texels in the merge's scratch.
        let (dispatched, back) = (0, |_: u64| 0);
        let (merged, fetched, kept_in_image) = match (dispatches, copies_in) {
            (true, true) => (1, 1, 0),
            (true, false) => (0, 1, 1),
            (false, _) => (0, u64::from(unorm), 0),
        };
        // The merge's kernel, and the keep where it is a dispatch, each make a descriptor pool
        // where the device does not push set 0.
        let pools = if pushes(&device) { 0 } else { 1 + merged };
        // The keep's dispatch takes the copy in of the same texels with it, where the device takes
        // the two as one.
        let copied_in = merged;
        let zeroed = device.work_counts().shadow_merge_scratches_zeroed;
        let before = moved();
        draw(&texture, Some(scissor));
        let scissored = since(before);
        println!(
            "SHADOW-COPY-BACK on {}: a scissored pass dispatched {} copies back and {} of its \
             keep and merge, kept {} texels, fetched {} merges, dispatched {} copies in with \
             their keeps and kept {} merges' texels in images (the device {} the dispatch)",
            device.name(),
            scissored.2,
            scissored.3,
            scissored.4,
            scissored.5,
            scissored.7,
            scissored.8,
            if dispatches { "takes" } else { "does not take" },
        );
        assert_eq!(
            scissored,
            (
                300 * 200,
                back(300 * 200),
                dispatched,
                merged,
                300 * 200,
                fetched,
                pools,
                copied_in,
                kept_in_image
            ),
            "a scissored pass"
        );
        draw(&private, Some(scissor));
        let mut expected = bytes.clone();
        lay(&layout, &mut expected, &blit_out(&queue, &layout, &private));
        let now = bytes_of(&buffer);
        let (count, first) = differing(
            &rows(&layout, &now),
            &rows(&layout, &expected),
            layout.texel,
        );
        assert!(
            count == 0,
            "{count} texels differ from the private texture's, the first at texel {first:?}"
        );
        assert!(now == expected, "the bytes between the rows changed");
        // Every texel, read in a later encoder: the buffer's, though the scissored pass left the
        // image's texels outside its scissor as they were before.
        let read = drawn(
            &queue,
            &layout,
            &reading(&device, layout.pf, "tob_read"),
            &texture,
            None,
        );
        let (count, first) = differing(&read, &rows(&layout, &expected), layout.texel);
        assert!(
            count == 0,
            "{count} texels read differ from the buffer's, the first at texel {first:?}"
        );
        let before = moved();
        draw(&texture, None);
        let every = layout.w * layout.h;
        assert_eq!(
            since(before),
            (
                every,
                back(every),
                dispatched,
                merged,
                every,
                fetched,
                pools,
                copied_in,
                kept_in_image
            ),
            "an unscissored pass"
        );
        // Both merges go in place over whole words of each row, so their copies write every byte
        // their kernels read, and neither zeroes its scratch.
        assert_eq!(
            device.work_counts().shadow_merge_scratches_zeroed,
            zeroed,
            "merge scratches zeroed"
        );
        drop((texture, buffer, private));
    }
    layer_reported_nothing_at_all(&messages);
}

/// A pass that loads a shadowed texture (the browser's padded tile over an ordinary buffer) over
/// a render area smaller than it, of whole 32-pixel tiles on neither axis, beside a memoryless
/// `RGBA16Float` scratch of its size loaded and stored `DontCare` (the consumer's guest's shadowed
/// passes' shape): the buffer holds the blend inside the render area, as a private texture of the
/// same texels drawn alike does, and every other byte as it was.
#[test]
fn a_shadowed_color_beside_a_memoryless_scratch_blends_over_part_of_it() {
    let layout = LAYOUTS[0];
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let len = layout.bpr * layout.h;
        let buffer = device.new_buffer(len, ResourceOptions::default()).unwrap();
        let bytes = texels_of(&layout, len as usize, 9);
        let contents = buffer.contents().unwrap();
        // SAFETY: a new `Shared` buffer of `len` bytes, which no command uses yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = buffer.resource_options();
        let texture = buffer.new_texture(&d, 0, layout.bpr).unwrap();
        // SAFETY: the handle is only compared, while the texture lives.
        assert!(unsafe { texture.vk_image() }.is_none(), "a shadow");
        let private = private_with(&queue, &layout, &rows(&layout, &bytes));
        let mut d =
            TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, layout.w, layout.h, false);
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = memoryless_options();
        let scratch = device.new_texture(&d).unwrap();
        let blend = blending_beside_scratch(&device, layout.pf, PixelFormat::RGBA16Float);
        let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
        let color = staged(&device, &floats([0.25, 0.5, 0.75, 0.5]));
        let area = [layout.w - 37, layout.h - 21];
        assert!(area.iter().all(|side| side % 32 != 0), "tiles cut");
        let draw = |target: &Texture| {
            let mut pass = RenderPassDescriptor::new();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Load;
            attachment.store_action = StoreAction::Store;
            let attachment = &mut pass.color_attachments[1];
            attachment.texture = Some(scratch.clone());
            attachment.load_action = LoadAction::DontCare;
            attachment.store_action = StoreAction::DontCare;
            [pass.render_target_width, pass.render_target_height] = area;
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&blend).unwrap();
            encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
            encoder.set_fragment_buffer(Some(&color), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
        };
        draw(&texture);
        draw(&private);
        let mut expected = bytes.clone();
        lay(&layout, &mut expected, &blit_out(&queue, &layout, &private));
        let now = bytes_of(&buffer);
        let (count, first) = differing(
            &rows(&layout, &now),
            &rows(&layout, &expected),
            layout.texel,
        );
        assert!(
            count == 0,
            "{count} texels differ from the private texture's, the first at texel {first:?}"
        );
        assert!(now == expected, "the bytes between the rows changed");
        drop((texture, buffer, private, scratch));
    }
    layer_reported_nothing_at_all(&messages);
}

/// A pass's copies in share their barriers (`encoder.rs`'s `ShadowRun`): a pass that loads a
/// shadowed texture over a `Shared` buffer and draws `n` rectangles, each under its own scissor
/// reading its own shadowed texture over the same buffer, copies the `n` textures in, and the
/// loaded texture in and kept (it merges back), before it. The copies in of different textures
/// share one barrier before them and one after; the keep, which reads the shadow its texture's
/// copy in wrote, takes a barrier of its own. So the pass records as many pipeline barriers
/// (`WorkCounts::commands_recorded`) whatever `n`, where a barrier before and one after each copy
/// grew them by two a texture (counted after a first pass of the shape). The buffer holds each
/// texture's texels inside its rectangle, as the draws read them, and every other byte as it
/// was. Printed: `SHADOW-RUN`.
#[test]
fn a_pass_s_copies_in_share_their_barriers() {
    let layout = LAYOUTS[4];
    assert!(layout.padded(), "{}", layout.name);
    let most = 4;
    let stride = (layout.len() as u64).next_multiple_of(4096);
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let len = stride * (most + 1);
        let buffer = device.new_buffer(len, ResourceOptions::default()).unwrap();
        let bytes = texels_of(&layout, len as usize, 3);
        let contents = buffer.contents().unwrap();
        // SAFETY: a new `Shared` buffer of `len` bytes, which no command uses yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        // The textures each rectangle reads at the buffer's start, the loaded one after them.
        let texture_at = |at: u64| {
            let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
            d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            d.resource_options = buffer.resource_options();
            let texture = buffer.new_texture(&d, at, layout.bpr).unwrap();
            // SAFETY: the handle is only compared, while the texture lives.
            assert!(unsafe { texture.vk_image() }.is_none(), "a shadow");
            texture
        };
        let read: Vec<Texture> = (0..most).map(|k| texture_at(k * stride)).collect();
        let loaded = texture_at(most * stride);
        let reads = reading(&device, layout.pf, "tob_read");
        let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
        // Rectangle `k` of `n`: a band of the texture's rows.
        let band = |k: u64, n: u64| {
            let rows = layout.h / n;
            ScissorRect::new(0, k * rows, layout.w, rows)
        };
        let pass = |n: u64| {
            let mut pass = RenderPassDescriptor::new();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(loaded.clone());
            attachment.load_action = LoadAction::Load;
            attachment.store_action = StoreAction::Store;
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&reads).unwrap();
            encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
            for (k, texture) in read.iter().take(n as usize).enumerate() {
                encoder.set_scissor_rect(band(k as u64, n)).unwrap();
                encoder.set_fragment_texture(Some(texture), 0).unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
            }
            encoder.end_encoding().unwrap();
            let before = device.work_counts().commands_recorded.cmd_pipeline_barrier;
            run(&command_buffer);
            device.work_counts().commands_recorded.cmd_pipeline_barrier - before
        };
        // A first pass makes what every pass of the shape needs (the textures' first moves, the
        // pipelines' sets), then the counted passes.
        let mut expected = bytes.clone();
        let mut barriers = Vec::new();
        for n in [most, 1, most] {
            barriers.push(pass(n));
            // Each band of the loaded texture's rows holds its texture's texels there.
            for k in 0..n {
                let rect = band(k, n);
                for y in rect.y..rect.y + rect.height {
                    let row = (y * layout.bpr) as usize;
                    let at = (most * stride) as usize + row;
                    let from = (k * stride) as usize + row;
                    let width = layout.row() as usize;
                    let texels = bytes[from..from + width].to_vec();
                    expected[at..at + width].copy_from_slice(&texels);
                }
            }
            let now = bytes_of(&buffer);
            let (count, first) = differing(&now, &expected, 1);
            assert!(
                count == 0,
                "after a pass of {n} rectangles {count} bytes differ, the first at {first:?}"
            );
        }
        println!(
            "SHADOW-RUN on {}: a pass reading 1 shadowed texture recorded {} pipeline barriers, \
             one reading {most} recorded {}",
            device.name(),
            barriers[1],
            barriers[2],
        );
        assert_eq!(
            barriers[1], barriers[2],
            "barriers of a pass of 1 and of {most} copies in"
        );
        drop((read, loaded, buffer));
    }
    layer_reported_nothing_at_all(&messages);
}

/// A pass that loads a shadowed texture and draws under a scissor beside a depth attachment it
/// clears and stores, which reaches every tile (`clear_reach::PassReach::Every`), `past` pixels
/// wider and taller than the color: where that is some, the depth's clear reaches past the render
/// area to whole tiles and is made before the pass (`clear_reach::ReachClears`,
/// `WorkCounts::attachment_clears_recorded`). Neither clear writes the color, so the draws alone
/// write it, inside the scissor, and its copy in and keep move the texels inside the scissor alone
/// (`WorkCounts::shadow_texels_copied_in`, `shadow_texels_kept`,
/// `shadow_attachment_loads_within`). The buffer holds the blend inside the scissor, as a private
/// texture of the same texels drawn alike does, and every other byte as it was.
fn a_loaded_color_beside_a_cleared_depth_copies_inside_its_scissor(past: u64) {
    let layout = LAYOUTS[0];
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let len = layout.bpr * layout.h;
        let buffer = device.new_buffer(len, ResourceOptions::default()).unwrap();
        let bytes = texels_of(&layout, len as usize, 7);
        let contents = buffer.contents().unwrap();
        // SAFETY: a new `Shared` buffer of `len` bytes, which no command uses yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = buffer.resource_options();
        let texture = buffer.new_texture(&d, 0, layout.bpr).unwrap();
        // SAFETY: the handle is only compared, while the texture lives.
        assert!(unsafe { texture.vk_image() }.is_none(), "a shadow");
        let private = private_with(&queue, &layout, &rows(&layout, &bytes));
        // The render area is the color's, 1307 pixels wide, which the depth's clear reaches past
        // to 1312, a whole number of 32-pixel tiles, where the depth is wider.
        let mut z = TextureDescriptor::texture_2d(
            PixelFormat::Depth32Float,
            layout.w + past,
            layout.h + past,
            false,
        );
        z.usage = TextureUsage::RENDER_TARGET;
        z.resource_options = ResourceOptions::private();
        let depth = device.new_texture(&z).unwrap();
        let blend = rects_beside(&device, layout.pf, true, Some(PixelFormat::Depth32Float));
        let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
        let color = staged(&device, &floats([0.25, 0.5, 0.75, 0.5]));
        let scissor = ScissorRect::new(100, 40, 300, 200);
        let draw = |target: &Texture| {
            let mut pass = RenderPassDescriptor::new();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Load;
            attachment.store_action = StoreAction::Store;
            pass.depth_attachment.texture = Some(depth.clone());
            pass.depth_attachment.load_action = LoadAction::Clear;
            pass.depth_attachment.store_action = StoreAction::Store;
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&blend).unwrap();
            encoder.set_scissor_rect(scissor).unwrap();
            encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
            encoder.set_fragment_buffer(Some(&color), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
        };
        let moved = || {
            let counts = device.work_counts();
            [
                counts.attachment_clears_recorded,
                counts.shadow_texels_copied_in,
                counts.shadow_texels_kept,
                counts.shadow_attachment_loads_within,
                counts.shadow_attachment_loads_whole,
            ]
        };
        let before = moved();
        draw(&texture);
        let after = moved();
        let [clears, copied_in, kept, within, whole] =
            [0, 1, 2, 3, 4].map(|i| after[i] - before[i]);
        println!(
            "CLEARED-DEPTH-BOUND on {}: a scissored pass beside a cleared depth {past} pixels \
             past the color recorded {clears} attachment clears, copied in {copied_in} texels \
             and kept {kept} ({within} colors loaded within the scissor, {whole} whole)",
            device.name(),
        );
        assert_eq!(
            clears > 0,
            past > 0,
            "the depth's clear past the render area"
        );
        assert_eq!(
            (copied_in, kept, within, whole),
            (300 * 200, 300 * 200, 1, 0),
            "the color's copies inside the scissor"
        );
        draw(&private);
        let mut expected = bytes.clone();
        lay(&layout, &mut expected, &blit_out(&queue, &layout, &private));
        let now = bytes_of(&buffer);
        let (count, first) = differing(
            &rows(&layout, &now),
            &rows(&layout, &expected),
            layout.texel,
        );
        assert!(
            count == 0,
            "{count} texels differ from the private texture's, the first at texel {first:?}"
        );
        assert!(now == expected, "the bytes between the rows changed");
        drop((texture, buffer, private, depth));
    }
    layer_reported_nothing_at_all(&messages);
}

/// [`a_loaded_color_beside_a_cleared_depth_copies_inside_its_scissor`] beside a depth of the
/// color's size.
#[test]
fn a_loaded_color_beside_a_cleared_depth_of_its_size_copies_inside_its_scissor() {
    a_loaded_color_beside_a_cleared_depth_copies_inside_its_scissor(0);
}

/// [`a_loaded_color_beside_a_cleared_depth_copies_inside_its_scissor`] beside a larger depth,
/// whose clear reaches past the render area.
#[test]
fn a_clear_past_the_render_area_leaves_a_loaded_color_s_copies_inside_its_scissor() {
    a_loaded_color_beside_a_cleared_depth_copies_inside_its_scissor(64);
}

/// One draw of [`merged_writes_match_metal`]'s: its scissor (none: the whole target), its
/// viewport (none: the whole target), its color, and a barrier over textures before it, which
/// splits the pass there.
#[derive(Clone, Copy)]
struct Stroke {
    scissor: Option<ScissorRect>,
    viewport: Option<[f64; 4]>,
    color: usize,
    split: bool,
}

/// [`Stroke`] under `scissor` alone.
fn stroke(scissor: (u64, u64, u64, u64), color: usize) -> Stroke {
    let (x, y, w, h) = scissor;
    Stroke {
        scissor: Some(ScissorRect::new(x, y, w, h)),
        viewport: None,
        color,
        split: false,
    }
}

/// Passes that load a shadowed texture over a buffer's bytes and draw under scissors that change
/// from draw to draw, in instances a barrier splits, across encoders and command buffers, with a
/// viewport and no scissor, under scissors at the right and bottom edges, and through a second
/// texture over the same bytes between them, leave the bytes as a private texture of the same
/// texels drawn alike holds them: each merge's keep and merge are cut to the rectangle the
/// encoder's commands write in (`shadow_merge.rs`), and no write outside it is lost. Over a
/// `Shared` buffer and over a no-copy buffer's host pages, of `layout`.
fn merged_writes_match_metal(layout: Layout) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let blend = blending(&device, layout.pf);
        let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
        let colors = [
            staged(&device, &floats([0.25, 0.5, 0.75, 0.5])),
            staged(&device, &floats([0.9, 0.1, 0.3, 0.75])),
        ];
        let (w, h) = (layout.w, layout.h);
        // Each command buffer's passes, each pass's strokes, the second texture's pass marked.
        // Every scissor lies inside the target, as Metal's validation requires, at fractions of
        // its size, so each stroke draws over every layout.
        let passes: Vec<Vec<(bool, Vec<Stroke>)>> = vec![
            vec![
                (
                    false,
                    vec![
                        stroke((w / 13, h / 12, w / 4, h / 3), 0),
                        stroke((w / 2, h / 2, w / 4, h / 4), 1),
                    ],
                ),
                (
                    false,
                    vec![
                        stroke((w / 26, h / 6, w / 5, h / 5), 1),
                        Stroke {
                            split: true,
                            ..stroke((w * 5 / 8, h * 5 / 8, w / 4, h / 5), 0)
                        },
                        Stroke {
                            split: true,
                            ..stroke((5, h * 5 / 8, w / 9, h / 3), 1)
                        },
                    ],
                ),
            ],
            vec![
                (true, vec![stroke((w / 3, h / 10, w / 4, h / 2), 0)]),
                (
                    false,
                    vec![Stroke {
                        scissor: None,
                        viewport: Some([
                            w as f64 / 4.0,
                            h as f64 / 5.0,
                            w as f64 / 3.0,
                            h as f64 / 4.0,
                        ]),
                        color: 1,
                        split: false,
                    }],
                ),
                (
                    false,
                    vec![
                        stroke((w - 13, h - 7, 13, 7), 0),
                        stroke((w - 1, 0, 1, h), 1),
                    ],
                ),
                (false, vec![stroke((0, 0, w, h), 0)]),
            ],
        ];
        let encode = |command_buffer: &metal2vulkan_runtime::CommandBuffer,
                      target: &Texture,
                      strokes: &[Stroke]| {
            let mut pass = RenderPassDescriptor::new();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Load;
            attachment.store_action = StoreAction::Store;
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&blend).unwrap();
            encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
            for s in strokes {
                if s.split {
                    encoder.memory_barrier_with_scope(
                        BarrierScope::TEXTURES,
                        RenderStages::FRAGMENT,
                        RenderStages::FRAGMENT,
                    );
                }
                let scissor = s.scissor.unwrap_or(ScissorRect::new(0, 0, w, h));
                assert!(
                    scissor.width > 0
                        && scissor.height > 0
                        && scissor.x + scissor.width <= w
                        && scissor.y + scissor.height <= h,
                    "{}: the scissor {scissor:?} lies outside the {w}x{h} target",
                    layout.name
                );
                encoder.set_scissor_rect(scissor).unwrap();
                let [x, y, vw, vh] = s.viewport.unwrap_or([0.0, 0.0, w as f64, h as f64]);
                encoder
                    .set_viewport(Viewport::new(x, y, vw, vh, 0.0, 1.0))
                    .unwrap();
                encoder
                    .set_fragment_buffer(Some(&colors[s.color]), 0, 0)
                    .unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
            }
            encoder.end_encoding().unwrap();
        };
        // The passes through `first` (and `second`, a texture over the same bytes, where marked),
        // command buffer by command buffer, waited for at the end.
        let draw_all = |first: &Texture, second: &Texture| {
            let mut committed = Vec::new();
            for buffer_passes in &passes {
                let command_buffer = queue.command_buffer().unwrap();
                for (twin, strokes) in buffer_passes {
                    encode(&command_buffer, if *twin { second } else { first }, strokes);
                }
                command_buffer.commit().unwrap();
                committed.push(command_buffer);
            }
            for command_buffer in &committed {
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None);
            }
        };
        let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let texture_over = |buffer: &Buffer| {
            let mut d = TextureDescriptor::texture_2d(layout.pf, w, h, false);
            d.usage = usage;
            d.resource_options = buffer.resource_options();
            buffer.new_texture(&d, 0, layout.bpr).unwrap()
        };
        // Over a `Shared` buffer.
        let len = layout.bpr * h;
        let buffer = device.new_buffer(len, ResourceOptions::default()).unwrap();
        let bytes = texels_of(&layout, len as usize, 3);
        let contents = buffer.contents().unwrap();
        // SAFETY: a new `Shared` buffer of `len` bytes, which no command uses yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        let (first, second) = (texture_over(&buffer), texture_over(&buffer));
        // SAFETY: the handle is only compared, while the texture lives.
        assert!(unsafe { first.vk_image() }.is_none(), "a shadow");
        let private = private_with(&queue, &layout, &rows(&layout, &bytes));
        draw_all(&first, &second);
        draw_all(&private, &private);
        let mut expected = bytes.clone();
        lay(&layout, &mut expected, &blit_out(&queue, &layout, &private));
        let now = bytes_of(&buffer);
        let (count, at) = differing(
            &rows(&layout, &now),
            &rows(&layout, &expected),
            layout.texel,
        );
        assert!(
            count == 0,
            "{}, a Shared buffer: {count} texels differ from the private texture's, the first at \
             texel {at:?}",
            layout.name
        );
        assert!(
            now == expected,
            "{}, a Shared buffer: the bytes between the rows changed",
            layout.name
        );
        drop((first, second, buffer, private));
        // Over a no-copy buffer's host pages.
        if let Some(surface) = surface(&device, &layout, usage) {
            let second = surface
                .buffer
                .new_texture(
                    &{
                        let mut d = TextureDescriptor::texture_2d(layout.pf, w, h, false);
                        d.usage = usage;
                        d.resource_options = surface.buffer.resource_options();
                        d
                    },
                    0,
                    layout.bpr,
                )
                .unwrap();
            let private = private_with(&queue, &layout, &rows(&layout, &surface.pages));
            draw_all(&surface.texture, &second);
            draw_all(&private, &private);
            let mut expected = surface.pages.clone();
            lay(&layout, &mut expected, &blit_out(&queue, &layout, &private));
            surface.check(&layout, "passes cut to their scissors", &expected);
            drop((second, private));
            surface.report(&device, &messages);
        }
    }
    layer_reported_nothing_at_all(&messages);
}

#[test]
fn merged_writes_match_metal_over_the_browser_tile() {
    merged_writes_match_metal(LAYOUTS[0]);
}

#[test]
fn merged_writes_match_metal_over_a_half_float_tile() {
    merged_writes_match_metal(LAYOUTS[7]);
}

#[test]
fn merged_writes_match_metal_over_a_narrow_tile() {
    merged_writes_match_metal(LAYOUTS[6]);
}

/// A texture of `layout` over a no-copy buffer, with shader read and write usage, written at every
/// texel by a fragment function's `write` (`tob_write`): the pages' texels are what a private
/// texture of the same texels holds after the same writes, and no other byte of the pages changes.
fn writes_as_metal(layout: Layout) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let Some(surface) = surface(&device, &layout, usage) else {
            return;
        };
        let mut expected = surface.pages.clone();
        let texels = rows(&layout, &expected);
        let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
        d.usage = usage;
        d.resource_options = ResourceOptions::private();
        let reference = device.new_texture(&d).unwrap();
        blit_in(&queue, &layout, &reference, &texels);
        let writes = reading(&device, layout.pf, "tob_write");
        drawn(&queue, &layout, &writes, &reference, None);
        drawn(&queue, &layout, &writes, &surface.texture, None);
        lay(
            &layout,
            &mut expected,
            &blit_out(&queue, &layout, &reference),
        );
        surface.check(&layout, "a fragment function's writes", &expected);
        surface.report(&device, &messages);
    }
    layer_reported_nothing_at_all(&messages);
}

/// One test a layout, each its own process: `$name` runs [`LAYOUTS`]`[$index]`.
macro_rules! per_layout {
    ($($name:ident => $index:expr;)*) => {
        $(
            #[test]
            fn $name() {
                reads_writes_and_renders_as_metal(LAYOUTS[$index]);
            }
        )*
    };
}

per_layout! {
    a_browser_tile_at_5248_byte_rows_reads_writes_and_renders_as_metal => 0;
    a_tile_at_16_byte_padded_rows_reads_writes_and_renders_as_metal => 1;
    a_tile_at_256_byte_padded_rows_reads_writes_and_renders_as_metal => 2;
    a_tile_at_512_byte_padded_rows_reads_writes_and_renders_as_metal => 3;
    a_small_tile_reads_writes_and_renders_as_metal => 4;
    a_full_width_strip_reads_writes_and_renders_as_metal => 5;
    a_narrow_tile_reads_writes_and_renders_as_metal => 6;
    a_half_float_tile_reads_writes_and_renders_as_metal => 7;
    a_tile_of_510_packed_rows_reads_writes_and_renders_as_metal => 8;
    a_small_tile_of_packed_rows_reads_writes_and_renders_as_metal => 9;
    a_half_float_tile_of_126_packed_rows_reads_writes_and_renders_as_metal => 10;
}

/// A fragment function's writes over [`LAYOUTS`]: padded rows (a browser's tile, a narrow tile, a
/// half-float tile) and packed ones of a height that is no multiple of 4, and of one that is.
#[test]
fn a_fragment_function_s_writes_leave_the_bytes_between_rows() {
    for index in [0, 6, 7, 8, 9, 10] {
        writes_as_metal(LAYOUTS[index]);
    }
}

/// Passes on one shadowed texture count once a pass, and once a command buffer and texture
/// (`WorkCounts::shadow_attachment_passes`, `shadow_attachment_textures`): a command buffer of
/// three passes into one texture over host pages and one into another between them, then a command
/// buffer of one pass into the first, count five passes on three textures (printed:
/// `SHADOW-ATTACHMENTS`).
#[test]
fn passes_on_one_shadow_count_once_a_command_buffer() {
    let layout = PACKED[1];
    on_shared_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let Some(first) = surface(device, &layout, usage) else {
            return;
        };
        let Some(second) = surface(device, &layout, usage) else {
            return;
        };
        let pipeline = blending(device, layout.pf);
        let corners = staged(device, &floats([-1.0, -1.0, 0.0, 0.0]));
        let color = staged(device, &floats([0.25, 0.5, 0.75, 0.5]));
        let pass = |command_buffer: &metal2vulkan_runtime::CommandBuffer, texture: &Texture| {
            let mut pass = RenderPassDescriptor::new();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(texture.clone());
            attachment.load_action = LoadAction::Load;
            attachment.store_action = StoreAction::Store;
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&pipeline).unwrap();
            encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
            encoder.set_fragment_buffer(Some(&color), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
        };
        let before = device.work_counts();
        let command_buffer = queue.command_buffer().unwrap();
        for texture in [
            &first.texture,
            &second.texture,
            &first.texture,
            &first.texture,
        ] {
            pass(&command_buffer, texture);
        }
        run(&command_buffer);
        let command_buffer = queue.command_buffer().unwrap();
        pass(&command_buffer, &first.texture);
        run(&command_buffer);
        let after = device.work_counts();
        let passes = after.shadow_attachment_passes - before.shadow_attachment_passes;
        let textures = after.shadow_attachment_textures - before.shadow_attachment_textures;
        eprintln!(
            "SHADOW-ATTACHMENTS on {}: {passes} passes on shadowed attachments, {textures} \
             textures a command buffer",
            device.name()
        );
        assert_eq!((passes, textures), (5, 3));
        first.report(device, messages);
        second.report(device, messages);
    });
}

/// Run `test` on one lane device, which announces `shadowed-texture` for each texture so that each
/// case prints its own reason; then fail on any validation message, leaks included. Each case
/// takes its own validation messages first, to fail by name.
fn on_shared_device(test: impl FnOnce(&Device, &Messages)) {
    on_device_of(HostPageTexturePolicy::Shadow, test);
}

/// [`on_shared_device`] on a device of `HostPageTexturePolicy::InPlace`, once its check answered
/// (printed, `IN-PLACE-CHECK`).
fn on_in_place_device(test: impl FnOnce(&Device, &Messages)) {
    on_device_of(HostPageTexturePolicy::InPlace, |device, messages| {
        assert!(device.settle_in_place_checks(CHECK_WAIT));
        for line in device.in_place_check_report() {
            eprintln!("IN-PLACE-CHECK on {}: {line}", device.name());
        }
        test(device, messages);
    });
}

/// [`on_shared_device`] on a device of `policy`.
fn on_device_of(policy: HostPageTexturePolicy, test: impl FnOnce(&Device, &Messages)) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.announce_each.insert(Degradation::ShadowedTexture);
            options.policy.host_page_textures = policy;
        })
        .unwrap();
        test(&device, &messages);
    }
    layer_reported_nothing_at_all(&messages);
}

/// `boxed` (rows of `bw` texels) laid into `expected` at texel (`bx`, `by`) of `layout`'s rows.
fn lay_box(layout: &Layout, expected: &mut [u8], (bx, by, bw): (u64, u64, u64), boxed: &[u8]) {
    for (y, row) in boxed.chunks((bw * layout.texel) as usize).enumerate() {
        let at = ((by + y as u64) * layout.bpr + bx * layout.texel) as usize;
        expected[at..at + row.len()].copy_from_slice(row);
    }
}

/// A texture of `layout` that only reads (`ShaderRead` usage), over a no-copy buffer: its first use
/// (a fragment function's `read`) comes before any command reads or writes the pages another way,
/// and must read the pages' texels and leave every byte of the pages, the bytes between the rows
/// among them, as it was. Then its reads (a nearest and a linear sample, `get_bytes`, a blit out),
/// the caller's later bytes (texels and the bytes between the rows alike) read by a sample and a
/// blit, and its writes that Metal makes of any texture (a buffer's box blitted in at an odd
/// origin, `replace_region` of a box, a private texture's box blitted in): each reads Metal's
/// texels, and after each every byte of the pages is the program's. `None` where the device
/// imports no host memory.
fn reads_and_blits_as_metal(device: &Device, messages: &Messages, layout: Layout) -> Option<()> {
    {
        let queue = device.new_command_queue().unwrap();
        let texel = layout.texel;
        let surface = surface(device, &layout, TextureUsage::SHADER_READ)?;
        let (texture, what) = (&surface.texture, &surface.what);
        let mut expected = surface.pages.clone();
        let check = |step: &str, expected: &[u8]| surface.check(&layout, step, expected);
        let read = |step: &str, ours: &[u8], theirs: &[u8]| {
            let (count, first) = differing(ours, theirs, texel);
            assert!(
                count == 0,
                "{what}, {step}: {count} texels differ, the first at texel {first:?}"
            );
        };

        // The first use, through the image, before anything else touches the pages.
        let texels = rows(&layout, &expected);
        let reads = reading(device, layout.pf, "tob_read");
        read(
            "its first use, a fragment function's read",
            &drawn(&queue, &layout, &reads, texture, None),
            &texels,
        );
        check("its first use", &expected);

        // Its other reads.
        let samples = reading(device, layout.pf, "tob_sample");
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        read(
            "a nearest sample",
            &drawn(&queue, &layout, &samples, texture, Some(&nearest)),
            &texels,
        );
        let private = private_with(&queue, &layout, &texels);
        read(
            "a linear sample, against a private texture's",
            &drawn(&queue, &layout, &samples, texture, Some(&linear)),
            &drawn(&queue, &layout, &samples, &private, Some(&linear)),
        );
        let mut got = vec![0u8; texels.len()];
        let whole = Region::new_2d(0, 0, layout.w, layout.h);
        texture
            .get_bytes(&mut got, layout.row(), 0, whole, 0, 0)
            .unwrap();
        read("get_bytes", &got, &texels);
        read("a blit out", &blit_out(&queue, &layout, texture), &texels);
        check("the reads", &expected);

        // The caller's later bytes: new texels, and new bytes between the rows, which no later
        // command may move.
        let mut later_pages = counting(surface.len, 9);
        let later = texels_of(&layout, texels.len(), 2);
        lay(&layout, &mut later_pages, &later);
        expected.copy_from_slice(&later_pages);
        surface.mapping.write(0, &expected);
        read(
            "a nearest sample of the caller's later bytes",
            &drawn(&queue, &layout, &samples, texture, Some(&nearest)),
            &later,
        );
        read(
            "a fragment function's read of the caller's later bytes",
            &drawn(&queue, &layout, &reads, texture, None),
            &later,
        );
        check("the reads of the caller's later bytes", &expected);

        // Writes Metal makes of any texture: a buffer's box blitted in at an odd origin.
        let (bx, by, bw, bh) = (3, 5, layout.w / 2 + 1, layout.h / 3 + 1);
        let boxed = texels_of(&layout, (bw * bh * texel) as usize, 3);
        let source = staged(device, &boxed);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            bw * texel,
            boxed.len() as u64,
            Size::new(bw, bh, 1),
            texture,
            0,
            0,
            Origin::new(bx, by, 0),
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        lay_box(&layout, &mut expected, (bx, by, bw), &boxed);
        check("a buffer's box blitted in", &expected);
        // `replace_region` of a box.
        let (rx, ry, rw, rh) = (layout.w / 3, 1, layout.w / 4 + 1, layout.h / 2);
        let replaced = texels_of(&layout, (rw * rh * texel) as usize, 4);
        texture
            .replace_region(
                Region::new_2d(rx, ry, rw, rh),
                0,
                0,
                &replaced,
                rw * texel,
                0,
            )
            .unwrap();
        lay_box(&layout, &mut expected, (rx, ry, rw), &replaced);
        check("replace_region", &expected);
        // A private texture's box blitted in.
        let other = private_with(&queue, &layout, &texels_of(&layout, texels.len(), 5));
        let (fx, fy, tx, ty) = (7, 1, layout.w / 3, 2);
        let size = Size::new(layout.w / 2, layout.h / 2, 1);
        blit_box(
            &queue,
            (&other, Origin::new(fx, fy, 0)),
            (texture, Origin::new(tx, ty, 0)),
            size,
        );
        let from = blit_out(&queue, &layout, &other);
        for y in 0..size.height {
            let start = ((fy + y) * layout.row() + fx * texel) as usize;
            let row = &from[start..start + (size.width * texel) as usize];
            let at = ((ty + y) * layout.bpr + tx * texel) as usize;
            expected[at..at + row.len()].copy_from_slice(row);
        }
        check("a private texture's box blitted in", &expected);
        // And the writes read back through the image.
        read(
            "a fragment function's read after the writes",
            &drawn(&queue, &layout, &reads, texture, None),
            &rows(&layout, &expected),
        );
        check("the last read", &expected);
        surface.report(device, messages);
    }
    layer_reported_nothing(messages, layout.name);
    Some(())
}

/// [`reads_and_blits_as_metal`] over each of [`LAYOUTS`]: a texture that only reads keeps every
/// byte of its pages through its first use, its reads, the caller's writes and Metal's blits into
/// it. Every layout on one device.
#[test]
fn a_texture_that_only_reads_keeps_every_byte_of_its_pages() {
    on_shared_device(every_layout_reads_and_blits_as_metal);
}

/// [`a_texture_that_only_reads_keeps_every_byte_of_its_pages`] on a device that takes such a
/// texture in place where its check passed and the device takes its pitch.
#[test]
fn a_texture_that_only_reads_keeps_every_byte_of_its_pages_in_place() {
    on_in_place_device(every_layout_reads_and_blits_as_metal);
}

/// [`reads_and_blits_as_metal`] over each of [`LAYOUTS`] on `device`.
fn every_layout_reads_and_blits_as_metal(device: &Device, messages: &Messages) {
    for layout in LAYOUTS {
        if reads_and_blits_as_metal(device, messages, layout).is_none() {
            return;
        }
    }
}

/// A texture that only reads over padded rows (each padded layout of [`LAYOUTS`] over a no-copy
/// buffer) is a shadow: its uses (three sampling passes and a blit out) copy it in
/// (`WorkCounts::shadow_copies`), and read the pages' texels. Every layout on one device.
#[test]
fn a_padded_texture_that_only_reads_is_a_shadow() {
    on_shared_device(|device, messages| {
        for layout in LAYOUTS.into_iter().filter(Layout::padded) {
            let queue = device.new_command_queue().unwrap();
            let usage = TextureUsage::SHADER_READ;
            let Some(surface) = surface(device, &layout, usage) else {
                return;
            };
            let what = surface.what.clone();
            let texels = rows(&layout, &surface.pages);
            let samples = reading(device, layout.pf, "tob_sample");
            let nearest = sampler(device, SamplerMinMagFilter::Nearest);
            let copies = || device.work_counts().shadow_copies;
            let before = copies();
            for _ in 0..3 {
                let got = drawn(&queue, &layout, &samples, &surface.texture, Some(&nearest));
                assert!(
                    got == texels,
                    "{what}: a sample differs from the pages' texels"
                );
            }
            assert!(
                blit_out(&queue, &layout, &surface.texture) == texels,
                "{what}"
            );
            let copied = copies() - before;
            eprintln!(
                "PITCHED-READS on {}: {what}: shadowed, {copied} shadow copies over three samples \
                 and a blit out",
                device.name(),
            );
            assert!(copied > 0, "{what}: a shadow its uses copied nothing into");
            surface.check(&layout, "its uses", &surface.pages);
            surface.report(device, messages);
            layer_reported_nothing(messages, &what);
        }
    });
}

/// A padded texture that renders (`RenderTarget`), that a shader writes (`ShaderWrite`), or of
/// `Unknown` usage (which Metal reports as both) is a shadow: over a no-copy buffer's pages and
/// over an ordinary buffer alike (the browser's tile, [`LAYOUTS`]' first). Every usage
/// on one device.
#[test]
fn a_padded_texture_that_renders_or_writes_is_a_shadow() {
    let layout = LAYOUTS[0];
    assert!(layout.padded(), "{}", layout.name);
    on_shared_device(|device, messages| {
        for usage in [
            TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
            TextureUsage::UNKNOWN,
        ] {
            // Over a no-copy buffer's pages (`surface` checks that it is a shadow).
            if let Some(surface) = surface(device, &layout, usage) {
                surface.report(device, messages);
            }
            // Over an ordinary buffer.
            let buffer = device
                .new_buffer(layout.bpr * layout.h, ResourceOptions::default())
                .unwrap();
            let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
            d.usage = usage;
            d.resource_options = buffer.resource_options();
            let texture = buffer.new_texture(&d, 0, layout.bpr).unwrap();
            // SAFETY: the handle is only compared, while the texture lives.
            let aliased = unsafe { texture.vk_image() }.is_some();
            assert!(
                !aliased,
                "{} ({usage:?}), an ordinary buffer's: aliased",
                layout.name
            );
            messages.take_notices();
            drop((texture, buffer));
            layer_reported_nothing(messages, &format!("{} ({usage:?})", layout.name));
        }
    });
}

/// A texture that only reads over an ordinary `Shared` buffer's padded rows (each padded layout
/// of [`LAYOUTS`]) is a shadow on every device, and its first use, a fragment function's read,
/// reads the buffer's texels and leaves every byte of the buffer, the bytes between the rows among
/// them, as it was; so does a nearest sample. Printed (`PITCHED-READS`): each layout's shadow
/// copies and reason. Every layout on one device.
#[test]
fn a_padded_texture_that_only_reads_over_an_ordinary_buffer_keeps_its_bytes() {
    on_shared_device(|device, messages| {
        for layout in LAYOUTS.into_iter().filter(Layout::padded) {
            let queue = device.new_command_queue().unwrap();
            let len = layout.bpr * layout.h;
            let buffer = device.new_buffer(len, ResourceOptions::default()).unwrap();
            let mut bytes = counting(len as usize, 11);
            lay(
                &layout,
                &mut bytes,
                &texels_of(&layout, (layout.row() * layout.h) as usize, 6),
            );
            let contents = buffer.contents().unwrap();
            // SAFETY: a new `Shared` buffer of `len` bytes, which no command uses yet.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len())
            };
            let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = buffer.resource_options();
            let texture = buffer.new_texture(&d, 0, layout.bpr).unwrap();
            // SAFETY: the handle is only compared, while the texture lives.
            let aliased = unsafe { texture.vk_image() }.is_some();
            let what = format!("{} (ShaderRead), an ordinary buffer's", layout.name);
            assert!(!aliased, "{what}: aliased");
            let texels = rows(&layout, &bytes);
            let copies = || device.work_counts().shadow_copies;
            let before = copies();
            let reads = reading(device, layout.pf, "tob_read");
            let read = drawn(&queue, &layout, &reads, &texture, None);
            let (count, first) = differing(&read, &texels, layout.texel);
            assert!(
                count == 0,
                "{what}, its first use: {count} texels read differ from the buffer's, the first \
                 at texel {first:?}"
            );
            let now = bytes_of(&buffer);
            assert!(
                now == bytes,
                "{what}, its first use: the bytes between the rows (or past the last) changed: {}",
                padding_changes(&layout, &now, &bytes)
            );
            let samples = reading(device, layout.pf, "tob_sample");
            let nearest = sampler(device, SamplerMinMagFilter::Nearest);
            let got = drawn(&queue, &layout, &samples, &texture, Some(&nearest));
            assert!(
                got == texels,
                "{what}: a sample differs from the buffer's texels"
            );
            let copied = copies() - before;
            assert!(
                bytes_of(&buffer) == bytes,
                "{what}: the buffer's bytes changed"
            );
            let reason = messages
                .take_notices()
                .into_iter()
                .filter_map(|message| message.degradation)
                .find(|notice| notice.degradation == Degradation::ShadowedTexture)
                .map(|notice| format!(": {}", notice.reason))
                .unwrap_or_default();
            eprintln!(
                "PITCHED-READS on {}: {what}: shadowed, {copied} shadow copies over a read and a \
                 sample{reason}",
                device.name(),
            );
            drop((texture, buffer));
            layer_reported_nothing(messages, &what);
        }
    });
}
