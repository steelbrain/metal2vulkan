//! A pass over a bound linear attachment (`linear_attachment.rs`) whose first instance records
//! inline, in place (`render/inline_first.rs`), and whose later instance renders the color in
//! another format through a stand-in: with the device's rule forced
//! (`DeviceShared::linear_rule_forced`), one pass of one color, an `R32Float` render target over a
//! `Shared` buffer's bytes that is its own linear image, loaded and stored, of three draws: an
//! `R32Float` state over most of it, recorded inline into the target itself; an `RGBA8Unorm` state
//! blended over its left half, which the image takes no view of (no `PixelFormatView`), through a
//! stand-in of that format (`attachment_stand_in::stand_in_as`) in an instance of its own, in a
//! secondary; and the `R32Float` state over its right half, after it. The pass's end renders the
//! bound color apart from that instance on (`linear_attachment::settle`): the image its texture
//! keeps takes the target's texels right after the inline instance, and the words the later
//! instances changed are merged back (`linear_merge.rs`). The bytes equal those the same draws
//! leave in a private texture of the same first texels, made with `PixelFormatView`, whose second
//! draw renders through a view. On a device whose draws read the colors as input attachments
//! (`framebuffer_fetch::through_inputs`: lavapipe's rasterization-order access), every color is
//! taken as the pass ends, and the bound color put apart there is refused
//! (`NotImplemented(Render)`), as it was before the first instance recorded inline: the first
//! instance still records inline.
//!
//! Waits for the device's in-place check to end before it makes anything
//! (`merge_scratch::settled`), and reads the counts of its own pass only then
//! (`merge_scratch::counts_alone`). Every draw waits for its pipeline
//! (`blit_holds::exact_waits_device`), so the first one records inline wherever the layout allows.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{exact_waits_device, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use super::merge_scratch::{blit, counts_alone, first_bytes, read, settled, staged, H, TEXEL};
use crate::{
    BlitOption, CommandQueue, Device, Error, LoadAction, Origin, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, RuntimeGap, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{BlendFactor, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::sync::atomic::Ordering;

/// The target's width: rows of 256 bytes.
const WIDTH: u64 = 64;

/// The pass's draws: by the `R32Float` state (`false`) or the blended `RGBA8Unorm` one (`true`),
/// over the rectangle (normalized device coordinates, `[x0, y0, x1, y1]`), of the color.
const DRAWS: [(bool, [f32; 4], [f32; 4]); 3] = [
    (false, [-1.0, -1.0, 0.5, 0.5], [0.25, 0.0, 0.0, 1.0]),
    (true, [-1.0, -1.0, 0.0, 1.0], [0.2, 0.4, 0.6, 0.8]),
    (false, [0.0, -1.0, 1.0, 0.0], [0.75, 0.0, 0.0, 1.0]),
];

/// The rectangle state (`tob_rect`, corners from vertex buffer 0) of the color in fragment buffer
/// 0 (`tob_solid`) into a color of `format`, blending it over the color by its alpha where
/// `blends`.
fn state(device: &Device, format: PixelFormat, blends: bool) -> RenderPipelineState {
    let library = library(device, "engine/tob_renders.ll");
    let function = |name: &str| {
        let function = library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(function("tob_solid"));
    let mut color = ColorAttachmentDescriptor::new(format);
    if blends {
        color.blending_enabled = true;
        color.source_rgb_blend_factor = BlendFactor::SourceAlpha;
        color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        color.source_alpha_blend_factor = BlendFactor::One;
        color.destination_alpha_blend_factor = BlendFactor::Zero;
    }
    descriptor.color_attachments[0] = Some(color);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// One pass over `texture`, loading and storing it, of [`DRAWS`] by `states` (`R32Float`,
/// `RGBA8Unorm`), committed and completed. The device's work counts as it completed; or the
/// refusal of its end, uncommitted.
fn pass(
    queue: &CommandQueue,
    texture: &Texture,
    states: (&RenderPipelineState, &RenderPipelineState),
) -> Result<WorkCounts, Error> {
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(texture.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    for (bytes, corners, rgba) in DRAWS {
        let state = match bytes {
            false => states.0,
            true => states.1,
        };
        encoder.set_render_pipeline_state(state).unwrap();
        (encoder.set_vertex_bytes(&bytes_of(&corners, f32::to_le_bytes), 0)).unwrap();
        (encoder.set_fragment_bytes(&bytes_of(&rgba, f32::to_le_bytes), 0)).unwrap();
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    }
    encoder.end_encoding()?;
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    Ok(queue.device().work_counts())
}

/// The texels of `texture`, [`WIDTH`] x [`H`], packed: read through a blit.
fn texels_of(queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let packed = WIDTH * TEXEL;
    let out = (queue.device())
        .new_buffer(packed * H, ResourceOptions::shared())
        .unwrap();
    blit(queue, |b| {
        (b.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(WIDTH, H, 1),
            &out,
            0,
            packed,
            packed * H,
            BlitOption::None,
        ))
        .unwrap()
    });
    read(&out, packed * H)
}

/// The module docs' pass, over a private texture and then over the target with the rule forced:
/// the target's first instance recorded inline, and the pass rendered apart from its later
/// instances on, the target's bytes the private texture's texels; or, where the draws read the
/// colors as input attachments, refused as it ends.
#[test]
fn a_bound_color_recorded_inline_goes_apart_after_its_first_instance() {
    let cache = Scratch::new("bound-inline");
    let device = settled(exact_waits_device(&cache.0));
    let queue = device.new_command_queue().unwrap();
    let states = (
        state(&device, PixelFormat::R32Float, false),
        state(&device, PixelFormat::RGBA8Unorm, true),
    );
    let states = (&states.0, &states.1);
    let packed = WIDTH * TEXEL;
    let first = first_bytes(WIDTH, packed);
    // The private texture of the target's first texels, which takes views of other formats.
    let mut d = TextureDescriptor::texture_2d(PixelFormat::R32Float, WIDTH, H, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::PIXEL_FORMAT_VIEW;
    d.resource_options = ResourceOptions::private();
    let reference = device.new_texture(&d).unwrap();
    let source = staged(&device, &first);
    blit(&queue, |b| {
        (b.copy_from_buffer_to_texture(
            &source,
            0,
            packed,
            packed * H,
            Size::new(WIDTH, H, 1),
            &reference,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        ))
        .unwrap()
    });
    pass(&queue, &reference, states).unwrap();
    let expected = texels_of(&queue, &reference);
    // The target: its own linear image over a `Shared` buffer of the same first texels, which
    // takes no view of another format.
    let buffer = staged(&device, &first);
    let mut d = TextureDescriptor::texture_2d(PixelFormat::R32Float, WIDTH, H, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = buffer.resource_options();
    let texture = buffer.new_texture(&d, 0, packed).unwrap();
    if !texture.linear_image().unwrap() {
        println!(
            "BOUND-INLINE on {}: the target is no linear image",
            device.name()
        );
        return;
    }
    let inputs = crate::framebuffer_fetch::through_inputs(device.compile_device().profile());
    let forced = &device.shared().linear_rule_forced;
    let before = counts_alone(&device);
    forced.store(true, Ordering::Relaxed);
    let ended = pass(&queue, &texture, states);
    forced.store(false, Ordering::Relaxed);
    let after = device.work_counts();
    let delta = |of: fn(&WorkCounts) -> u64| of(&after) - of(&before);
    let inline = delta(|c| c.inline_passes);
    let refused = delta(|c| c.passes_not_inline_attachments);
    assert_eq!(
        (inline, refused),
        (1, 0),
        "the pass's first instance recorded inline"
    );
    if inputs {
        println!(
            "BOUND-INLINE on {}: colors read as input attachments; passes inline {inline}, the \
             end {ended:?}",
            device.name()
        );
        assert_eq!(
            ended.err(),
            Some(Error::NotImplemented(RuntimeGap::Render)),
            "the bound color put apart as the pass ends, every color taken"
        );
        return;
    }
    ended.unwrap();
    let through = delta(|c| c.linear_passes_through_copies);
    let got = read(&buffer, packed * H);
    let differ = (0..got.len() / 4)
        .filter(|&t| got[t * 4..t * 4 + 4] != expected[t * 4..t * 4 + 4])
        .count();
    println!(
        "BOUND-INLINE on {}: passes inline {inline}, refused for their attachments {refused}, \
         through copies {through}; {differ} texels differ",
        device.name()
    );
    assert_eq!(
        through, 1,
        "the pass rendered the bound color apart from its later instances"
    );
    assert_eq!(differ, 0, "the texels differ from the private texture's");
    drop((texture, buffer));
}
