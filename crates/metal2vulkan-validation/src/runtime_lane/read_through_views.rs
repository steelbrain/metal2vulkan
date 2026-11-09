//! Device tests of a view in other channels of its image's bits, on the device the cfg selects,
//! under the validation layer: an `A8Unorm` view of an `R8Unorm` texture (a glyph atlas's, made
//! with `ShaderRead | ShaderWrite | RenderTarget | PixelFormatView` as a guest's are, and with
//! `ShaderRead` alone), which its bindings take as the texture's image through a swizzle
//! (`texture/view_shadow.rs`, `metal::format::swizzle_over`) where the image takes no Vulkan view
//! of `A8Unorm`, and a pass renders through a shadow.
//!
//! - [`a_view_reads_its_image_through_a_swizzle`]: the view, sampled at every texel's center
//!   through a nearest and a linear sampler and under a clamp-to-border sampler past its edge
//!   (opaque black, transparent black, opaque white), draws what a plain `A8Unorm` texture of the
//!   same bytes draws, and each texel is `(0, 0, 0, code)`; where the device reads through the
//!   swizzle, each view counts in `WorkCounts::views_through_swizzle`, no encoder plans a shadow
//!   copy and no view shadow is made or announced.
//! - [`a_write_through_the_view_stores_the_alpha_lane`]: a kernel's `write` through the view
//!   stores the first lane whose source is alpha in the texture's bytes, as Metal does (probe
//!   `a8write` on an M3 Max under macOS 27.0.1): the alpha lane through the identity, the red lane
//!   through `(Alpha, Red, One, Zero)`; no view shadow where the device reads through the swizzle.
//! - [`a_pass_renders_into_the_view_through_its_shadow`]: a pass that renders into the view stores
//!   the fragment's alpha in the texture's bytes inside its draw and keeps them outside, as Metal
//!   does (probe `a8write`), through one view shadow; `shadowed-texture-view` is announced naming
//!   the two formats' classes, not a missing `PixelFormatView`. Its copies between `R8_UNORM` and
//!   `A8_UNORM` go through a buffer, which the validation layer holds them to
//!   (`VUID-vkCmdCopyImage-srcImage-01548`).
//! - [`mipmaps_through_a_view_are_the_formats`]: `generate_mipmaps` of the view writes the levels a
//!   plain `A8Unorm` texture of the same bytes takes, each the device's filter of the level above
//!   (`device-filtered-mipmaps`): a texel the box of four texels above it averages to, or, off a
//!   code, one of the two codes around that average, as Vulkan's conversion to a normalized code
//!   lets the device take either ([`box_filtered`]). Which of the two may differ between the
//!   formats (RADV on the RX 550 rounds `R8_UNORM`'s ties up and `A8_UNORM`'s down).

use super::degrade_notices::on_device;
use super::Messages;
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::FormatUsage;
use metal2vulkan::metal::format::{swizzle_over, vk_format};
use metal2vulkan::metal::{
    PixelFormat, PrimitiveType, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter,
    TextureSwizzleChannels, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk::Format;
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandQueue, Degradation, DegradeNotice, Device, LoadAction, Origin,
    PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size,
    StoreAction, Texture, TextureDescriptor,
};
use std::path::Path;
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// The atlas's side, in texels: 16 rows of 16 codes, every byte once.
const SIDE: u64 = 16;

/// The target's side: the atlas, and 8 pixels past its right and bottom edges.
const TARGET: u64 = SIDE + 8;

/// Every byte, row by row.
fn codes() -> Vec<u8> {
    (0..=255u8).collect()
}

/// A `Shared` 2D texture of `pf`, `SIDE` square, of `levels` levels and `usage`, holding
/// [`codes`] at level 0.
fn atlas(device: &Device, pf: PixelFormat, usage: TextureUsage, levels: bool) -> Texture {
    let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, levels);
    d.usage = usage;
    d.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&d).unwrap();
    texture
        .replace_region(Region::new_2d(0, 0, SIDE, SIDE), 0, 0, &codes(), SIDE, 0)
        .unwrap();
    texture
}

/// The `tob_rect` and `tob_sample` state into an unblended `RGBA8Unorm` color.
pub(super) fn sampling_state(device: &Device) -> RenderPipelineState {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/tob_renders.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
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
    d.fragment = Some(function("tob_sample"));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&d, &PipelineOptions::default())
        .unwrap()
}

/// How a draw samples the atlas.
#[derive(Clone, Copy, Debug)]
pub(super) enum Sampled {
    /// Nearest, at each texel's center.
    Nearest,
    /// Linear, at each texel's center.
    Linear,
    /// Nearest, clamped to this border color past the atlas's edges.
    Border(SamplerBorderColor),
}

/// A pass into a new private `RGBA8Unorm` target of [`TARGET`] square, cleared, drawing the
/// whole target with `state` sampling `texture` as `sampled` says (`tob_sample` reads the pixel's
/// center over the texture's size: the target's pixel `p` at the atlas's texel `p`'s center, and
/// past its right and bottom edges beyond them), and the target's bytes after it, read back.
pub(super) fn draw(
    queue: &CommandQueue,
    state: &RenderPipelineState,
    texture: &Texture,
    sampled: Sampled,
) -> Vec<u8> {
    let device = queue.device();
    let mut s = SamplerDescriptor::new();
    match sampled {
        Sampled::Nearest => {}
        Sampled::Linear => {
            s.min_filter = SamplerMinMagFilter::Linear;
            s.mag_filter = SamplerMinMagFilter::Linear;
        }
        Sampled::Border(color) => {
            s.s_address_mode = SamplerAddressMode::ClampToBorderColor;
            s.t_address_mode = SamplerAddressMode::ClampToBorderColor;
            s.border_color = color;
        }
    }
    let sampler = device.new_sampler_state(&s).unwrap();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, TARGET, TARGET, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let target = device.new_texture(&d).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    let rect: Vec<u8> = [-1.0f32, -1.0, 1.0, 1.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let back = device
        .new_buffer(TARGET * TARGET * 4, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_bytes(&rect, 0).unwrap();
    encoder.set_fragment_texture(Some(texture), 0).unwrap();
    encoder
        .set_fragment_sampler_state(Some(&sampler), 0)
        .unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(TARGET, TARGET, 1),
        &back,
        0,
        TARGET * 4,
        TARGET * TARGET * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    bytes_of(&back)
}

/// The bytes of a `Shared` buffer no command uses now.
fn bytes_of(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's `length` bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// Whether `device`'s bindings take an `A8Unorm` view of an `R8Unorm` image as the image itself
/// through a swizzle: `A8Unorm` samples as `A8_UNORM`, which the image takes no view of, and
/// `swizzle_over` answers for the pair. A device that samples `A8Unorm` as `R8_UNORM` views the
/// image itself.
fn through_a_swizzle(device: &Device) -> bool {
    let profile = device.compile_device().profile();
    let a8 = vk_format(
        PixelFormat::A8Unorm,
        &[FormatUsage::SampledImage].into(),
        profile,
    );
    let through = swizzle_over(
        PixelFormat::A8Unorm,
        TextureSwizzleChannels::DEFAULT,
        PixelFormat::R8Unorm,
        profile,
    );
    through.is_some() && a8.is_ok_and(|m| m.format == Format::A8_UNORM)
}

/// The `shadowed-texture-view` notices among those delivered so far (all removed).
fn shadows_announced(messages: &Messages) -> Vec<DegradeNotice> {
    messages
        .take_notices()
        .into_iter()
        .filter_map(|m| m.degradation)
        .filter(|n| n.degradation == Degradation::ShadowedTextureView)
        .collect()
}

/// The guest's usage of a glyph atlas.
fn guest_usage() -> TextureUsage {
    TextureUsage::SHADER_READ
        | TextureUsage::SHADER_WRITE
        | TextureUsage::RENDER_TARGET
        | TextureUsage::PIXEL_FORMAT_VIEW
}

/// The bytes of `texture`'s first level, read back by a blit.
fn atlas_bytes(queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let back = queue
        .device()
        .new_buffer(SIDE * SIDE, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        SIDE,
        SIDE * SIDE,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    bytes_of(&back)
}

/// A function of the `name` fixture under `fixtures/public/engine`.
fn function(device: &Device, name: &str, fixture: &str) -> StageSource {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/public/engine")
        .join(fixture);
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    StageSource::Function(
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor(),
    )
}

/// The draws of [`draw`], each sampling, through the A8 view of `base` and through `plain`: equal,
/// and every texel inside the atlas `(0, 0, 0, code)` where nearest; the shadow copies the view's
/// draws planned, summed.
fn compare(queue: &CommandQueue, base: &Texture, plain: &Texture) -> u64 {
    let device = queue.device();
    let state = sampling_state(device);
    let view = base.new_texture_view(PixelFormat::A8Unorm).unwrap();
    let mut copies = 0;
    for sampled in [
        Sampled::Nearest,
        Sampled::Linear,
        Sampled::Border(SamplerBorderColor::OpaqueBlack),
        Sampled::Border(SamplerBorderColor::TransparentBlack),
        Sampled::Border(SamplerBorderColor::OpaqueWhite),
    ] {
        let before = device.work_counts().shadow_copies;
        let through = draw(queue, &state, &view, sampled);
        copies += device.work_counts().shadow_copies - before;
        let expected = draw(queue, &state, plain, sampled);
        let differing = through
            .chunks(4)
            .zip(expected.chunks(4))
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "READ-THROUGH on {}: {sampled:?}: {differing} texels differ from a plain A8Unorm \
             texture's",
            device.name()
        );
        assert_eq!(differing, 0, "{sampled:?}: the view draws otherwise");
        if let Sampled::Nearest = sampled {
            for (t, code) in codes().into_iter().enumerate() {
                let (x, y) = (t as u64 % SIDE, t as u64 / SIDE);
                let at = ((y * TARGET + x) * 4) as usize;
                assert_eq!(through[at..at + 4], [0, 0, 0, code], "texel {t}");
            }
        }
    }
    copies
}

/// The module docs' first test.
#[test]
fn a_view_reads_its_image_through_a_swizzle() {
    on_device(|device, messages| {
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        let through = through_a_swizzle(device);
        let plain = atlas(
            device,
            PixelFormat::A8Unorm,
            TextureUsage::SHADER_READ,
            false,
        );
        for usage in [TextureUsage::SHADER_READ, guest_usage()] {
            let base = atlas(device, PixelFormat::R8Unorm, usage, false);
            let before = device.work_counts();
            let copies = compare(&queue, &base, &plain);
            let after = device.work_counts();
            let views = after.views_through_swizzle - before.views_through_swizzle;
            let shadows = after.view_shadows - before.view_shadows;
            let announced = shadows_announced(messages).len();
            println!(
                "READ-THROUGH on {}: usage {usage:?}, through a swizzle {through}: {views} views \
                 through it, {shadows} view shadows, {copies} shadow copies planned, \
                 shadowed-texture-view announced {announced} times",
                device.name()
            );
            if through {
                assert_eq!(views, 1, "the view not taken through the swizzle");
                assert_eq!(shadows, 0, "a view shadow made");
                assert_eq!(copies, 0, "the view's draws planned shadow copies");
                assert_eq!(announced, 0, "shadowed-texture-view announced");
            }
        }
    });
}

/// The module docs' second test.
#[test]
fn a_write_through_the_view_stores_the_alpha_lane() {
    use metal2vulkan::metal::TextureSwizzle as T;
    use metal2vulkan::metal::TextureType;
    use metal2vulkan::pipeline::ComputePipelineDescriptor;
    on_device(|device, messages| {
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        let through = through_a_swizzle(device);
        let state = device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(function(
                    device,
                    "rrp_sw_write_f",
                    "rrp_swizzled_storage.ll",
                )),
                &PipelineOptions::default(),
            )
            .unwrap();
        // Lanes of codes 51, 102, 153 and 204.
        let lanes: Vec<u8> = [0.2f32, 0.4, 0.6, 0.8]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let swizzled = TextureSwizzleChannels::new(T::Alpha, T::Red, T::One, T::Zero);
        for (swizzle, stored) in [(TextureSwizzleChannels::DEFAULT, 204u8), (swizzled, 51)] {
            let base = atlas(device, PixelFormat::R8Unorm, guest_usage(), false);
            let view = base
                .new_texture_view_with_swizzle(
                    PixelFormat::A8Unorm,
                    TextureType::Type2D,
                    0..1,
                    0..1,
                    swizzle,
                )
                .unwrap();
            let before = device.work_counts().view_shadows;
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&state).unwrap();
            encoder.set_texture(Some(&view), 0).unwrap();
            encoder.set_bytes(&lanes, 1).unwrap();
            encoder
                .dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(SIDE, SIDE, 1))
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
            let shadows = device.work_counts().view_shadows - before;
            let bytes = atlas_bytes(&queue, &base);
            let wrong = bytes.iter().filter(|&&b| b != stored).count();
            println!(
                "READ-THROUGH on {}: a write through the view of swizzle {swizzle:?}: {wrong} of \
                 {} bytes are not {stored}, {shadows} view shadows",
                device.name(),
                bytes.len()
            );
            assert_eq!(wrong, 0, "{swizzle:?}: the write stored another lane");
            if through {
                assert_eq!(shadows, 0, "a view shadow made");
            }
        }
        messages.take_notices();
    });
}

/// The module docs' third test.
#[test]
fn a_pass_renders_into_the_view_through_its_shadow() {
    on_device(|device, messages| {
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        let through = through_a_swizzle(device);
        let mut d = RenderPipelineDescriptor::new(function(device, "tob_rect", "tob_renders.ll"));
        d.fragment = Some(function(device, "tob_solid", "tob_renders.ll"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::A8Unorm));
        let state = match device.new_render_pipeline_state(&d, &PipelineOptions::default()) {
            Ok(state) => state,
            Err(error) => {
                println!(
                    "READ-THROUGH on {}: SKIPPED, no A8Unorm render pipeline: {error}",
                    device.name()
                );
                return;
            }
        };
        let base = atlas(device, PixelFormat::R8Unorm, guest_usage(), false);
        let view = base.new_texture_view(PixelFormat::A8Unorm).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(view.clone());
        color.load_action = LoadAction::Load;
        color.store_action = StoreAction::Store;
        // The left half of the atlas, in a colour whose alpha is code 204.
        let rect: Vec<u8> = [-1.0f32, -1.0, 0.0, 1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let solid: Vec<u8> = [0.2f32, 0.4, 0.6, 0.8]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let before = device.work_counts().view_shadows;
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder.set_vertex_bytes(&rect, 0).unwrap();
        encoder.set_fragment_bytes(&solid, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let shadows = device.work_counts().view_shadows - before;
        let bytes = atlas_bytes(&queue, &base);
        let wrong = bytes
            .iter()
            .enumerate()
            .filter(|&(t, &b)| {
                b != if (t as u64 % SIDE) < SIDE / 2 {
                    204
                } else {
                    t as u8
                }
            })
            .count();
        let announced = shadows_announced(messages);
        println!(
            "READ-THROUGH on {}: a pass through the view: {wrong} of {} bytes wrong, {shadows} \
             view shadows, announced {:?}",
            device.name(),
            bytes.len(),
            announced
                .iter()
                .map(|n| format!("{}: {}", n.object, n.reason))
                .collect::<Vec<_>>()
        );
        assert_eq!(wrong, 0, "the pass stored other bytes");
        if through {
            assert_eq!(shadows, 1, "the pass made no view shadow, or more than one");
            assert_eq!(
                announced.len(),
                1,
                "shadowed-texture-view not announced once"
            );
            assert!(
                !announced[0].reason.contains("no PixelFormatView usage"),
                "the notice names a usage the texture has"
            );
            assert!(
                announced[0].reason.contains("compatibility classes"),
                "the notice does not name the formats' classes"
            );
        }
    });
}

/// `SIDE`-square levels below the first of `texture`, read back by a blit: each level's codes.
fn levels(queue: &CommandQueue, texture: &Texture) -> Vec<Vec<u8>> {
    let device = queue.device();
    let count = texture.mipmap_level_count();
    let mut out = Vec::new();
    for level in 1..count {
        let side = (SIDE >> level).max(1);
        let back = device
            .new_buffer(side * side, ResourceOptions::shared())
            .unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            level,
            Origin::new(0, 0, 0),
            Size::new(side, side, 1),
            &back,
            0,
            side,
            side * side,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        out.push(bytes_of(&back));
    }
    out
}

/// Whether `level`, a level of one-byte codes, is the box filter of `above`, the level twice its
/// side: each texel is the average of the four texels above it where that is a code, else one of
/// the two codes around it.
fn box_filtered(above: &[u8], level: &[u8]) -> bool {
    let side = (above.len() as f64).sqrt() as usize;
    let half = side / 2;
    level.len() == half * half
        && (0..half * half).all(|i| {
            let (x, y) = (2 * (i % half), 2 * (i / half));
            let sum: u32 = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
                .iter()
                .map(|&(x, y)| u32::from(above[y * side + x]))
                .sum();
            // `4 * code` within four of the sum: below one code of the average.
            (4 * u32::from(level[i])).abs_diff(sum) < 4
        })
}

/// `generate_mipmaps` of `texture` in a command buffer that completes.
fn mipmaps(queue: &CommandQueue, texture: &Texture) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.generate_mipmaps(texture).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// The module docs' fourth test.
#[test]
fn mipmaps_through_a_view_are_the_formats() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let base = atlas(
            device,
            PixelFormat::R8Unorm,
            TextureUsage::SHADER_READ,
            true,
        );
        let plain = atlas(
            device,
            PixelFormat::A8Unorm,
            TextureUsage::SHADER_READ,
            true,
        );
        let view = base.new_texture_view(PixelFormat::A8Unorm).unwrap();
        mipmaps(&queue, &view);
        mipmaps(&queue, &plain);
        let (through, expected) = (levels(&queue, &base), levels(&queue, &plain));
        assert_eq!(through.len(), expected.len(), "the view's levels");
        for (what, chain) in [
            ("through the view", &through),
            ("of the plain texture", &expected),
        ] {
            let mut above = codes();
            for (level, got) in (1..).zip(chain) {
                assert!(
                    box_filtered(&above, got),
                    "mipmaps {what}: level {level}: {got:?} of {above:?}"
                );
                above.clone_from(got);
            }
        }
        let apart: usize = (through.iter().zip(&expected))
            .map(|(a, b)| a.iter().zip(b).filter(|(a, b)| a != b).count())
            .sum();
        println!(
            "READ-THROUGH on {}: mipmaps through the view: {} of {} levels equal the plain \
             texture's, {apart} texels apart",
            device.name(),
            through
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a == b)
                .count(),
            expected.len()
        );
        messages.take_notices();
    });
}
