//! Device tests of separate multisampled depth and stencil textures in one pass
//! (`runtime::depth_stencil_scratch`) on the device the cfg selects, under the validation layer.
//! Such a pass renders into a combined multisampled scratch image, and each aspect it loads or
//! stores is copied bit for bit, every sample, through a multisampled colour image of the aspect's
//! bits (maintenance8's depth-stencil/colour copies), or, for the stencil where those copies are
//! missing or wrong, by bit-plane stencil draws (`runtime::stencil_copy`). The single-sampled
//! form's test is `render_state`'s `separate_load_dontcare`.
//!
//! Each sample's depth and stencil are read back sample by sample
//! (`candidate::runtime_samples::encode`), never resolved, and compared with what Metal's rules
//! give: a loaded aspect keeps every sample's value, a stored one holds what the pass left in each
//! sample, an aspect stored `DontCare` keeps the texture's texels (Metal kept them in every case
//! measured), and a cleared one holds its clear value.

use super::degrade_notices::counts;
use super::render_state::stencil;
use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, validated_device, validated_device_with,
    Messages, ScratchCache, WAIT,
};
use crate::candidate::runtime_samples::{encode, library, shape, SampleFill};
use crate::case::TextureFormat;
use crate::executor_contract::multisample::SampleRead;
use crate::gpu::TEST_DEVICE;
use crate::literal::TextureLayout;
use metal2vulkan::device::{VkExtension, VkFeature};
use metal2vulkan::error::{Capability, Feature, Needs, RuntimeDriverDefect, Unsupported};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    CompareFunction, PixelFormat, PrimitiveType, StencilOperation, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk::DriverId;
use metal2vulkan_runtime::{
    CommandBuffer, CommandBufferStatus, Degradation, Device, DriverFact, Error, LoadAction,
    PipelineOptions, RenderPassDescriptor, ResourceOptions, StoreAction, Texture,
    TextureDescriptor,
};

/// The textures' extent and samples: the fragment-depth probe's.
const EXTENT: [u32; 2] = [2, 2];
const SAMPLES: u32 = 4;

/// The extents the separate textures' passes run at: 2x2, and the shapes whose colour copies lost
/// the RTX 3090 (4x1, 4x2, 1x4), which the draws now take there.
const EXTENTS: [[u32; 2]; 4] = [[2, 2], [4, 1], [4, 2], [1, 4]];

/// Each pixel's samples' depths as the fill writes them.
const FILLED: [f32; 4] = [0.2, 0.4, 0.6, 0.8];

/// A 2D multisample texture of `pf` and `extent` the passes render and the readback reads.
fn multisampled(device: &Device, pf: PixelFormat, extent: [u32; 2]) -> Result<Texture, Error> {
    multisampled_for(
        device,
        pf,
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        extent,
    )
}

/// A private 2D multisample texture of `pf`, `usage` and `extent`.
fn multisampled_for(
    device: &Device,
    pf: PixelFormat,
    usage: TextureUsage,
    extent: [u32; 2],
) -> Result<Texture, Error> {
    let mut d = TextureDescriptor::texture_2d(pf, extent[0].into(), extent[1].into(), false);
    d.texture_type = TextureType::Type2DMultisample;
    d.sample_count = SAMPLES.into();
    d.usage = usage;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d)
}

/// The bytes of every pixel's samples of `extent` holding `values`, sample by sample.
fn every_pixel<T: Copy>(values: [T; 4], extent: [u32; 2], bytes: impl Fn(T) -> Vec<u8>) -> Vec<u8> {
    (0..extent[0] * extent[1])
        .flat_map(|_| values.into_iter().flat_map(&bytes))
        .collect()
}

fn depths(values: [f32; 4], extent: [u32; 2]) -> Vec<u8> {
    every_pixel(values, extent, |v| v.to_le_bytes().to_vec())
}

fn stencils(values: [u8; 4], extent: [u32; 2]) -> Vec<u8> {
    every_pixel(values, extent, |v| vec![v])
}

/// A pass's actions on both textures, depth first: its loads, its stores, the depth and stencil
/// it clears to, and the depth of each sample of every pixel it draws the fullscreen triangle at,
/// if it draws.
struct Actions {
    loads: [LoadAction; 2],
    stores: [StoreAction; 2],
    clears: (f64, u32),
    draw: Option<[f32; 4]>,
}

/// Encode a pass of `textures` with `actions` into `command_buffer`. Its draw is at sample rate,
/// under depth `Less` (written) and a stencil increment where the depth passes. The pass's
/// refusal is returned.
fn pass(
    device: &Device,
    command_buffer: &CommandBuffer,
    (depth, stencil_texture): (&Texture, &Texture),
    actions: &Actions,
) -> Result<(), Error> {
    let extent = [depth.width() as u32, depth.height() as u32];
    let mut descriptor = RenderPassDescriptor::new();
    let attachment = &mut descriptor.depth_attachment;
    attachment.texture = Some(depth.clone());
    attachment.load_action = actions.loads[0];
    attachment.store_action = actions.stores[0];
    attachment.clear_depth = actions.clears.0;
    let attachment = &mut descriptor.stencil_attachment;
    attachment.texture = Some(stencil_texture.clone());
    attachment.load_action = actions.loads[1];
    attachment.store_action = actions.stores[1];
    attachment.clear_stencil = actions.clears.1;
    let mut encoder = command_buffer.render_command_encoder(&descriptor)?;
    if let Some(z) = actions.draw {
        let library = library(device).unwrap();
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut pipeline = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
        pipeline.fragment = Some(function("rt6_fill_depth32float"));
        pipeline.depth_format = Some(PixelFormat::Depth32Float);
        pipeline.stencil_format = Some(PixelFormat::Stencil8);
        pipeline.raster_sample_count = SAMPLES;
        let pipeline = device
            .new_render_pipeline_state(&pipeline, &PipelineOptions::default())
            .unwrap();
        let mut state = DepthStencilDescriptor::default();
        state.depth_compare_function = CompareFunction::Less;
        state.depth_write_enabled = true;
        let face = stencil(
            CompareFunction::Always,
            StencilOperation::IncrementClamp,
            0xff,
            0xff,
        );
        state.front_face_stencil = face;
        state.back_face_stencil = face;
        let state = device.new_depth_stencil_state(&state).unwrap();
        let values = buffer_with(device, &depths(z, extent)).unwrap();
        let shape = shape(device, extent[0], SAMPLES).unwrap();
        encoder.set_render_pipeline_state(&pipeline).unwrap();
        encoder.set_depth_stencil_state(&state).unwrap();
        encoder.set_fragment_buffer(Some(&values), 0, 0).unwrap();
        encoder.set_fragment_buffer(Some(&shape), 0, 1).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    Ok(())
}

/// Commit `command_buffer` after encoding the readback of every sample of `depth` and of
/// `stencil_texture`, and give their bytes once it completed.
fn read_back(
    device: &Device,
    command_buffer: CommandBuffer,
    depth: &Texture,
    stencil_texture: &Texture,
) -> (Vec<u8>, Vec<u8>) {
    let extent = [depth.width() as u32, depth.height() as u32];
    let read =
        |texture, read| encode(device, &command_buffer, texture, read, extent, SAMPLES).unwrap();
    let depth = read(depth, SampleRead::Depth);
    let stencil_texture = read(stencil_texture, SampleRead::Stencil);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    (depth.bytes().unwrap(), stencil_texture.bytes().unwrap())
}

/// Separate 4x `Depth32Float` and `Stencil8` textures through three passes at each of
/// [`EXTENTS`], each sample read back: a pass that loads both (each sample's depth filled, the stencil cleared to 0x40) and
/// draws at 0.5 stores the new depth and an incremented stencil in exactly the samples whose
/// depth was farther; a second that loads both and draws at 0.1 stores its depth everywhere,
/// and its stencil store `DontCare` leaves the texture as it was; a third clears both and stores
/// the clears. The passes' degrade is announced once.
///
/// On lavapipe and MoltenVK, whose drivers copy multisampled depth/stencil images to and from
/// colour ones wrong (`driver_defects.rs`), and on a device without maintenance8, the aspects are
/// copied by draws instead (on NVIDIA's Ampere GA10x, which loses the device copying the stencil
/// aspect, the stencil alone), where the depth's draw is exact (the driver fact
/// `FragmentDepthStoresUnchanged`, lavapipe from the release measured) or measured to change some
/// values (MoltenVK, NVIDIA's: none of these depths, announced once as `drawn-depth-copy`), and the
/// first pass is refused, typed, elsewhere (lavapipe's earlier releases included); every other
/// device runs the three through colour images.
#[test]
fn separate_multisampled_depth_and_stencil_keep_every_sample() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        // Every aspect's copies wrong (the depth's then drawn too), or the stencil's alone, whose
        // depth the colour copies move.
        let defect = RuntimeDriverDefect::MultisampledDepthStencilColourCopies;
        let stencil_only = RuntimeDriverDefect::MultisampledStencilColourCopies;
        let defects = &device.properties().driver_defects;
        let defective = defects.contains(&defect);
        let description = &device.properties().description;
        // The defects where they were measured: the stencil's alone on NVIDIA's RTX 3090
        // (0x2204, Ampere GA10x, whose other devices the table refuses unmeasured), none on the
        // RTX 5080s (0x2c02, 0x2c19).
        let measured = match description.driver_id {
            Some(DriverId::MESA_LLVMPIPE | DriverId::MOLTENVK) => Some((true, false)),
            Some(DriverId::NVIDIA_PROPRIETARY) => match description.device_id {
                0x2204 => Some((false, true)),
                0x2c02 | 0x2c19 => Some((false, false)),
                _ => None,
            },
            _ => Some((false, false)),
        };
        if let Some(measured) = measured {
            let found = (defective, defects.contains(&stencil_only));
            assert_eq!(found, measured, "{name}: the driver's defects");
        }
        // Where the depth's draw is exact: the driver's measured fact (`driver_facts.rs`).
        let draws_exact = device
            .properties()
            .driver_facts
            .contains(&DriverFact::FragmentDepthStoresUnchanged);
        // Where the driver's fragment depth is not measured to keep every value (measured to
        // change some, MoltenVK's and NVIDIA's, or not measured: the runtime's
        // `depth_copy::drawn`): the draw runs as the announced `drawn-depth-copy`, and every depth
        // here (0.1 to 1) lands exactly.
        let draws_change = !draws_exact;
        let mut ran = false;
        let mut drawn = false;
        for extent in EXTENTS {
            let [w, h] = extent;
            match three_passes(&device, extent) {
                Outcome::TexturesRefused(u) => {
                    eprintln!(
                        "separate multisampled depth and stencil on {name} at {w}x{h}: textures \
                         refused, {u:?}"
                    );
                }
                Outcome::Ran => {
                    assert!(
                        !defective || draws_exact || draws_change,
                        "{name} ran what its driver copies wrong"
                    );
                    let by_draws =
                        defective || !device.enabled().features.contains(&VkFeature::Maintenance8);
                    drawn |= by_draws && draws_change;
                    eprintln!(
                        "separate multisampled depth and stencil on {name} at {w}x{h}: every \
                         sample loaded, stored, kept and cleared as Metal's rules give"
                    );
                    ran = true;
                }
                Outcome::Refused(u) => {
                    assert!(
                        !draws_exact && !draws_change,
                        "{name} refused what its draws copy: {u:?}"
                    );
                    assert_eq!(
                        u.feature,
                        Feature::SeparateMultisampledDepthStencil,
                        "{u:?}"
                    );
                    let maintenance8 =
                        Needs::AnyOf(vec![vec![Capability::Feature(VkFeature::Maintenance8)]]);
                    if defective {
                        assert_eq!(u.needs, Needs::RuntimeDriverDefect(defect), "{u:?}");
                    } else {
                        assert_eq!(u.needs, maintenance8, "{u:?}");
                    }
                    eprintln!(
                        "separate multisampled depth and stencil on {name} at {w}x{h}: refused, \
                         {u:?}"
                    );
                }
            }
        }
        if ran {
            let counted = counts(&messages.take_notices());
            assert_eq!(
                counted.get(&Degradation::SeparateDepthStencil),
                Some(&1),
                "the passes' degrade, once: {counted:?}"
            );
            let expected = drawn.then_some(&1);
            assert_eq!(
                counted.get(&Degradation::DrawnDepthCopy),
                expected,
                "the depth's drawn copies, once where they change values: {counted:?}"
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// How [`three_passes`] ended.
enum Outcome {
    /// Every pass ran, and every sample was checked.
    Ran,
    /// The device makes no such textures.
    TexturesRefused(Unsupported),
    /// The device refused the first pass.
    Refused(Unsupported),
}

/// The three passes of [`separate_multisampled_depth_and_stencil_keep_every_sample`] at
/// `extent`, each sample of both textures checked after each.
fn three_passes(device: &Device, extent: [u32; 2]) -> Outcome {
    let depth = multisampled(device, PixelFormat::Depth32Float, extent);
    let stencil_texture = multisampled(device, PixelFormat::Stencil8, extent);
    let (depth, stencil_texture) = match (depth, stencil_texture) {
        (Ok(depth), Ok(stencil_texture)) => (depth, stencil_texture),
        (Err(Error::Unsupported(u)), _) | (_, Err(Error::Unsupported(u))) => {
            return Outcome::TexturesRefused(u)
        }
        (Err(error), _) | (_, Err(error)) => panic!("new_texture: {error}"),
    };
    let textures = (&depth, &stencil_texture);
    let queue = device.new_command_queue().unwrap();

    // The fill: each sample's depth, and the stencil cleared by a pass of its own.
    let command_buffer = queue.command_buffer().unwrap();
    let layout = TextureLayout {
        width: extent[0],
        height: extent[1],
        depth: 1,
        array_layers: 1,
        sample_count: SAMPLES,
    };
    SampleFill::new(
        device,
        &depth,
        TextureFormat::Depth32Float,
        layout,
        &depths(FILLED, extent),
    )
    .unwrap()
    .encode(&command_buffer)
    .unwrap();
    let mut clear = RenderPassDescriptor::new();
    clear.stencil_attachment.texture = Some(stencil_texture.clone());
    clear.stencil_attachment.load_action = LoadAction::Clear;
    clear.stencil_attachment.clear_stencil = 0x40;
    clear.stencil_attachment.store_action = StoreAction::Store;
    let encoder = command_buffer.render_command_encoder(&clear).unwrap();
    encoder.end_encoding().unwrap();
    // Both loaded and stored, drawn at 0.5.
    let loaded = Actions {
        loads: [LoadAction::Load; 2],
        stores: [StoreAction::Store; 2],
        clears: (0.0, 0),
        draw: Some([0.5; 4]),
    };
    match pass(device, &command_buffer, textures, &loaded) {
        Ok(()) => {}
        Err(Error::Unsupported(u)) => return Outcome::Refused(u),
        Err(error) => panic!("render_command_encoder: {error}"),
    }
    let (depths_read, stencils_read) = read_back(device, command_buffer, &depth, &stencil_texture);
    let [w, h] = extent;
    assert_eq!(
        depths_read,
        depths([0.2, 0.4, 0.5, 0.5], extent),
        "{w}x{h}: drawn at 0.5"
    );
    assert_eq!(
        stencils_read,
        stencils([0x40, 0x40, 0x41, 0x41], extent),
        "{w}x{h}: drawn at 0.5"
    );

    // Both loaded, drawn at 0.1; the stencil stored `DontCare`.
    let command_buffer = queue.command_buffer().unwrap();
    let kept = Actions {
        loads: [LoadAction::Load; 2],
        stores: [StoreAction::Store, StoreAction::DontCare],
        clears: (0.0, 0),
        draw: Some([0.1; 4]),
    };
    pass(device, &command_buffer, textures, &kept).unwrap();
    let (depths_read, stencils_read) = read_back(device, command_buffer, &depth, &stencil_texture);
    assert_eq!(
        depths_read,
        depths([0.1; 4], extent),
        "{w}x{h}: drawn at 0.1"
    );
    assert_eq!(
        stencils_read,
        stencils([0x40, 0x40, 0x41, 0x41], extent),
        "{w}x{h}: a DontCare stencil store keeps the texture's samples"
    );

    // Both cleared and stored.
    let command_buffer = queue.command_buffer().unwrap();
    let cleared = Actions {
        loads: [LoadAction::Clear; 2],
        stores: [StoreAction::Store; 2],
        clears: (1.0, 0x10),
        draw: None,
    };
    pass(device, &command_buffer, textures, &cleared).unwrap();
    let (depths_read, stencils_read) = read_back(device, command_buffer, &depth, &stencil_texture);
    assert_eq!(depths_read, depths([1.0; 4], extent), "{w}x{h}: cleared");
    assert_eq!(
        stencils_read,
        stencils([0x10; 4], extent),
        "{w}x{h}: cleared"
    );
    Outcome::Ran
}

/// Separate 4x `Depth32Float` and `Stencil8` textures through two passes that copy only the
/// stencil, at each of [`EXTENTS`]: the depth cleared in the pass and stored `DontCare`, the stencil loaded and stored,
/// each pass drawing every pixel's samples at depths 0.2, 0.4, 0.6 and 0.8 and incrementing the
/// stencil where the depth passes `Less`. Over the stencil cleared to 0x40, a pass clearing the
/// depth to 0.5 leaves samples 0x41, 0x41, 0x40, 0x40, and a second clearing it to 0.7 leaves
/// 0x42, 0x42, 0x41, 0x40: each sample loaded and stored as Metal's rules give, distinct across
/// the samples. Where maintenance8's depth-stencil/colour copies are missing or wrong (lavapipe,
/// MoltenVK, NVIDIA's Ampere GA10x), the stencil is copied by the plane draws
/// (`runtime/stencil_copy.rs`), so every device runs both passes unless it lacks what the draws
/// need, which it names. The passes' degrade is announced once.
#[test]
fn a_separate_multisampled_stencil_keeps_every_sample() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        let mut ran_any = false;
        for extent in EXTENTS {
            if !stencil_passes(&device, &name, extent) {
                continue;
            }
            ran_any = true;
        }
        if ran_any {
            // Render targets that shaders do not read: where the colour copies cannot move the
            // stencil, the runtime makes their images sampled where that costs nothing
            // (`depth_stencil_scratch::sampled_candidate`), so the pass runs; lavapipe lays them
            // out alike.
            let unread =
                |pf| multisampled_for(&device, pf, TextureUsage::RENDER_TARGET, EXTENT).unwrap();
            let (depth, stencil_texture) = (
                unread(PixelFormat::Depth32Float),
                unread(PixelFormat::Stencil8),
            );
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let actions = Actions {
                loads: [LoadAction::Clear, LoadAction::Load],
                stores: [StoreAction::DontCare, StoreAction::Store],
                clears: (0.5, 0),
                draw: Some(FILLED),
            };
            match pass(
                &device,
                &command_buffer,
                (&depth, &stencil_texture),
                &actions,
            ) {
                Ok(()) => {
                    command_buffer.commit().unwrap();
                    command_buffer.wait_until_completed_for(WAIT).unwrap();
                    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                    eprintln!(
                        "a separate multisampled stencil on {name}: unread render targets run"
                    );
                }
                Err(Error::Unsupported(u)) => {
                    let lavapipe =
                        device.properties().description.driver_id == Some(DriverId::MESA_LLVMPIPE);
                    assert!(!lavapipe, "{name} refused unread render targets: {u:?}");
                    eprintln!(
                        "a separate multisampled stencil on {name}: unread render targets \
                         refused, {u:?}"
                    );
                }
                Err(error) => panic!("render_command_encoder: {error}"),
            }
            let counted = counts(&messages.take_notices());
            assert_eq!(
                counted.get(&Degradation::SeparateDepthStencil),
                Some(&1),
                "the passes' degrade, once: {counted:?}"
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// The two passes of [`a_separate_multisampled_stencil_keeps_every_sample`] at `extent` on
/// `device` (named `name`), each sample checked: whether both ran (`false` where the device makes
/// no such textures or refuses the pass, which is printed).
fn stencil_passes(device: &Device, name: &str, extent: [u32; 2]) -> bool {
    let [w, h] = extent;
    let depth = multisampled(device, PixelFormat::Depth32Float, extent);
    let stencil_texture = multisampled(device, PixelFormat::Stencil8, extent);
    let (depth, stencil_texture) = match (depth, stencil_texture) {
        (Ok(depth), Ok(stencil_texture)) => (depth, stencil_texture),
        (Err(Error::Unsupported(u)), _) | (_, Err(Error::Unsupported(u))) => {
            eprintln!(
                "a separate multisampled stencil on {name} at {w}x{h}: textures refused, {u:?}"
            );
            return false;
        }
        (Err(error), _) | (_, Err(error)) => panic!("new_texture: {error}"),
    };
    let textures = (&depth, &stencil_texture);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut clear = RenderPassDescriptor::new();
    clear.stencil_attachment.texture = Some(stencil_texture.clone());
    clear.stencil_attachment.load_action = LoadAction::Clear;
    clear.stencil_attachment.clear_stencil = 0x40;
    clear.stencil_attachment.store_action = StoreAction::Store;
    let encoder = command_buffer.render_command_encoder(&clear).unwrap();
    encoder.end_encoding().unwrap();
    // The clear runs with the first pass.
    let mut first = Some(command_buffer);
    for (cleared_depth, expected) in [
        (0.5, [0x41, 0x41, 0x40, 0x40]),
        (0.7, [0x42, 0x42, 0x41, 0x40]),
    ] {
        let command_buffer = match first.take() {
            Some(command_buffer) => command_buffer,
            None => queue.command_buffer().unwrap(),
        };
        let actions = Actions {
            loads: [LoadAction::Clear, LoadAction::Load],
            stores: [StoreAction::DontCare, StoreAction::Store],
            clears: (cleared_depth, 0),
            draw: Some(FILLED),
        };
        match pass(device, &command_buffer, textures, &actions) {
            Ok(()) => {}
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    u.feature,
                    Feature::SeparateMultisampledDepthStencil,
                    "{name}: {u:?}"
                );
                assert!(
                    !matches!(u.needs, Needs::RuntimeDriverDefect(_)),
                    "{name} refused the stencil's copies for the colour copies: {u:?}"
                );
                eprintln!("a separate multisampled stencil on {name} at {w}x{h}: refused, {u:?}");
                return false;
            }
            Err(error) => panic!("render_command_encoder: {error}"),
        }
        let (_, stencils_read) = read_back(device, command_buffer, &depth, &stencil_texture);
        assert_eq!(
            stencils_read,
            stencils(expected, extent),
            "{name} at {w}x{h}: the depth cleared to {cleared_depth}"
        );
    }
    eprintln!(
        "a separate multisampled stencil on {name} at {w}x{h}: every sample loaded and stored as \
         Metal's rules give"
    );
    true
}

/// The `float` bit patterns the fragment-depth probe stores, one per sample of the 2x2, 4-sample
/// texture, by pixel then sample: first the values a depth texture holds in `[0, 1]` (signed zero,
/// the smallest and largest denormals, the smallest normal, the largest below 1, 1 and 0.2), then
/// the rest (2, -1, a negative denormal, both infinities, a quiet NaN with and without a payload,
/// a signalling NaN).
const PROBED: [u32; 16] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x007f_ffff,
    0x0080_0000,
    0x3f7f_ffff,
    0x3f80_0000,
    0x3e4c_cccd,
    0x4000_0000,
    0xbf80_0000,
    0x8000_0001,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0x7fc0_0001,
    0x7f80_0001,
];

/// How many of [`PROBED`] lie in `[0, 1]`, first.
const PROBED_IN_RANGE: usize = 8;

/// The depth the fill's fragment function hands `device`'s driver for the bits `bits`: Metal's
/// stored answer, as the translator gives it (`stage_io::clamp_depth`). `None` for a positive
/// NaN, which fails Metal's depth test and is killed on every device. Where the device enables
/// `VK_EXT_depth_range_unrestricted`, under which Vulkan stores the depth unclamped, a value with
/// the sign bit set is +0 and one past 1 up to +inf is 1; any other value, and every value
/// elsewhere, is `bits`.
fn handed(device: &Device, bits: u32) -> Option<u32> {
    if bits > 0x7f80_0000 && bits < 0x8000_0000 {
        return None;
    }
    let clamps = device
        .enabled()
        .extensions
        .contains(&VkExtension::ExtDepthRangeUnrestricted);
    Some(match bits {
        _ if !clamps => bits,
        0x8000_0000.. => 0,
        0x3f80_0001.. => 0x3f80_0000,
        _ => bits,
    })
}

/// Store `bits` into a fresh 4x `Depth32Float` texture by a sample-rate draw that writes each as
/// its sample's fragment depth (`runtime_samples`' fill), and read every sample back.
fn stored_depths(device: &Device, bits: &[u32]) -> Vec<u32> {
    let texture = multisampled(device, PixelFormat::Depth32Float, EXTENT).unwrap();
    let layout = TextureLayout {
        width: EXTENT[0],
        height: EXTENT[1],
        depth: 1,
        array_layers: 1,
        sample_count: SAMPLES,
    };
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    SampleFill::new(
        device,
        &texture,
        TextureFormat::Depth32Float,
        layout,
        &bytes,
    )
    .unwrap()
    .encode(&command_buffer)
    .unwrap();
    let read = encode(
        device,
        &command_buffer,
        &texture,
        SampleRead::Depth,
        EXTENT,
        SAMPLES,
    )
    .unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    read.bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

/// The fragment-depth probe: a copy of a multisampled depth by draws that write each sample's
/// fetched depth as its fragment depth is exact where the device stores every value a depth texture
/// can hold unchanged: each value in `[0, 1]` it is given (a clear can leave any of them), and, for
/// the rest, the value it stored the first time when given it again (what a draw left is what a
/// copy reads). [`PROBED`] is stored once and what was stored is stored again, each sample read
/// back; the outcome is printed for each value.
///
/// The fill is a Metal fragment function, which hands the driver Metal's stored depth
/// ([`handed`]): a positive NaN writes nothing, and on a device with
/// `VK_EXT_depth_range_unrestricted` a value outside `[0, 1]` or with the sign bit set arrives
/// clamped. There the probe measures the driver on the values a fragment write leaves; the raw
/// ones were measured before the clamp (`driver_facts.rs`).
#[test]
fn fragment_depth_stores_depth_values_unchanged() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        let once = stored_depths(&device, &PROBED);
        let twice = stored_depths(&device, &once);
        let mut exact = true;
        for (index, ((&given, &first), &second)) in PROBED.iter().zip(&once).zip(&twice).enumerate()
        {
            let Some(handed) = handed(&device, given) else {
                eprintln!("fragment depth on {name}: {given:#010x} written by no fragment");
                continue;
            };
            let kept = if index < PROBED_IN_RANGE {
                first == handed && second == first
            } else {
                second == first
            };
            exact &= kept;
            eprintln!(
                "fragment depth on {name}: {given:#010x} handed as {handed:#010x}, stored as \
                 {first:#010x}, then {second:#010x}{}",
                if kept { "" } else { " (changed)" }
            );
        }
        let description = &device.properties().description;
        eprintln!(
            "fragment depth on {name} ({:?}, driverVersion {:#x}): {}",
            description.driver_id,
            description.driver_version,
            if exact {
                "stores every probed value unchanged"
            } else {
                "changes a probed value"
            }
        );
        // The driver fact the runtime copies a multisampled depth by such draws on
        // (`driver_facts.rs`): the probe holds wherever the device's driver claims it. Where it
        // was probed and not listed, it does not hold: MoltenVK 1.4.1 on an M4 stores -0 as +0 and
        // clamps to [0, 1]; the RTX 3090 (615.71.09) flushes denormals to 0. Elsewhere (another
        // driver, or a release before the one measured) the runtime does not rely on it, and the
        // probe's answer is printed above.
        let claimed = device
            .properties()
            .driver_facts
            .contains(&DriverFact::FragmentDepthStoresUnchanged);
        let measured = match description.driver_id {
            _ if claimed => Some(true),
            Some(DriverId::MOLTENVK) => Some(false),
            Some(DriverId::NVIDIA_PROPRIETARY) if description.device_id == 0x2204 => Some(false),
            _ => None,
        };
        if let Some(measured) = measured {
            assert_eq!(
                exact, measured,
                "{name}: the probe against the driver's facts"
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// The 16-bit depth tests' extent: its 4 samples a pixel hold each of the 65536 codes once.
const CODES_EXTENT: [u32; 2] = [128, 128];

/// Every 16-bit code's depth, the code over 65535 as a `float`, one a sample, by pixel then
/// sample.
fn every_code() -> Vec<f32> {
    (0..=u16::MAX)
        .map(|code| f32::from(code) / 65535.0)
        .collect()
}

/// A 4x `Depth16Unorm` texture of [`CODES_EXTENT`] for render and shader reads.
fn depth16(device: &Device) -> Result<Texture, Error> {
    multisampled_for(
        device,
        PixelFormat::Depth16Unorm,
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        CODES_EXTENT,
    )
}

/// Encode into `command_buffer` a pass of `texture`'s depth alone that writes `depths`, one a
/// sample by pixel then sample, as each sample's fragment depth (the fill's fragment function,
/// under compare `Always` with depth writes on).
fn draw_depths(device: &Device, command_buffer: &CommandBuffer, texture: &Texture, depths: &[f32]) {
    let library = library(device).unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut pipeline = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
    pipeline.fragment = Some(function("rt6_fill_depth32float"));
    pipeline.depth_format = Some(texture.pixel_format());
    pipeline.raster_sample_count = SAMPLES;
    let pipeline = device
        .new_render_pipeline_state(&pipeline, &PipelineOptions::default())
        .unwrap();
    let mut state = DepthStencilDescriptor::default();
    state.depth_compare_function = CompareFunction::Always;
    state.depth_write_enabled = true;
    let state = device.new_depth_stencil_state(&state).unwrap();
    let bytes: Vec<u8> = depths.iter().flat_map(|d| d.to_le_bytes()).collect();
    let values = buffer_with(device, &bytes).unwrap();
    let shape = shape(device, texture.width() as u32, SAMPLES).unwrap();
    let mut pass = RenderPassDescriptor::new();
    pass.depth_attachment.texture = Some(texture.clone());
    pass.depth_attachment.load_action = LoadAction::DontCare;
    pass.depth_attachment.store_action = StoreAction::Store;
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&pipeline).unwrap();
    encoder.set_depth_stencil_state(&state).unwrap();
    encoder.set_fragment_buffer(Some(&values), 0, 0).unwrap();
    encoder.set_fragment_buffer(Some(&shape), 0, 1).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
}

/// The bits of the `float` each sample of `texture`, a 4x depth texture of [`CODES_EXTENT`],
/// reads as after the work in `command_buffer`, which this commits and waits for.
fn read_depths(device: &Device, command_buffer: CommandBuffer, texture: &Texture) -> Vec<u32> {
    let read = encode(
        device,
        &command_buffer,
        texture,
        SampleRead::Depth,
        CODES_EXTENT,
        SAMPLES,
    )
    .unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    words(&read.bytes().unwrap())
}

/// `bytes` as little-endian words.
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

/// Store `depths` into a fresh texture of [`depth16`] by a draw, and read every sample back as
/// the bits of the `float` it reads as; the device's refusal of the texture.
fn stored_codes(device: &Device, depths: &[f32]) -> Result<Vec<u32>, Error> {
    let texture = depth16(device)?;
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    draw_depths(device, &command_buffer, &texture, depths);
    Ok(read_depths(device, command_buffer, &texture))
}

/// The 16-bit depth probe: a copy of a multisampled `Depth16Unorm` depth by draws that write each
/// sample's fetched depth as its fragment depth keeps every code where the driver stores each
/// code's fetched `float` as that code. Every code's depth (the code over 65535) is stored once by
/// a draw and every sample read back, as the `float` it reads as, and what was read is stored again
/// and read back. The draw keeps every code where the first reads are 65536 different values, so
/// every code was stored and the reads tell the codes apart, and the second reads are the first's,
/// so storing a code's read gives the code back. Printed: how many first reads are not the code
/// over 65535 (Vulkan does not state the read's rounding), and the first codes changed. Asserted
/// where the driver claims `DriverFact::FragmentDepthKeepsDepth16Codes`.
#[test]
fn fragment_depth_keeps_every_16_bit_code() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        probe_16_bit_codes(&device);
        messages.take_notices();
    }
    layer_reported_nothing_at_all(&messages);
}

/// [`fragment_depth_keeps_every_16_bit_code`] on `device`.
fn probe_16_bit_codes(device: &Device) {
    let name = device.name().to_owned();
    let given = every_code();
    let once = match stored_codes(device, &given) {
        Ok(once) => once,
        Err(Error::Unsupported(u)) => {
            eprintln!("16-bit fragment depth on {name}: the texture refused, {u:?}");
            return;
        }
        Err(error) => panic!("16-bit fragment depth on {name}: {error}"),
    };
    let read: Vec<f32> = once.iter().map(|&bits| f32::from_bits(bits)).collect();
    let twice = stored_codes(device, &read).unwrap();
    let distinct = once.iter().collect::<std::collections::BTreeSet<_>>().len();
    let off = once
        .iter()
        .zip(&given)
        .filter(|(&bits, depth)| bits != depth.to_bits())
        .count();
    let changed: Vec<_> = once
        .iter()
        .zip(&twice)
        .enumerate()
        .filter(|(_, (first, second))| first != second)
        .collect();
    let exact = distinct == given.len() && changed.is_empty();
    for (code, (first, second)) in changed.iter().take(8) {
        eprintln!(
            "16-bit fragment depth on {name}: code {code} read {first:#010x}, stored again \
             {second:#010x}"
        );
    }
    let description = &device.properties().description;
    eprintln!(
        "16-bit fragment depth on {name} ({:?}, driverVersion {:#x}): {distinct} different \
         reads of {} codes, {off} not the code over 65535, {} changed when stored again: {}",
        description.driver_id,
        description.driver_version,
        given.len(),
        changed.len(),
        if exact {
            "keeps every code"
        } else {
            "changes a code"
        }
    );
    let claimed = device
        .properties()
        .driver_facts
        .contains(&DriverFact::FragmentDepthKeepsDepth16Codes);
    if claimed {
        assert!(exact, "{name}: the probe against the driver's facts");
    }
}

/// Separate 4x `Depth16Unorm` and `Stencil8` textures of [`CODES_EXTENT`] in one pass that loads
/// and stores the depth and clears and stores the stencil, drawing nothing: every sample keeps
/// its code (each of the 65536 once, stored by a draw before the pass, read back before and
/// after it) and the stencil holds its clear, as Metal keeps a loaded and stored aspect. The
/// depth moves into the combined scratch image and back by maintenance8's colour copies, or,
/// where those are missing or wrong, by the fragment-depth draws where the driver keeps a 16-bit
/// depth's codes (`DriverFact::FragmentDepthKeepsDepth16Codes`), and is refused typed
/// (`SeparateMultisampledDepthStencil`) elsewhere. A device that refuses the textures, or the
/// pass for a format it lacks (`Feature::Format`: lavapipe and ANV make no `D16_UNORM_S8_UINT`
/// attachment for the scratch image), is printed.
#[test]
fn a_separate_multisampled_16_bit_depth_keeps_every_code() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        separate_16_bit_codes(&device);
        messages.take_notices();
    }
    layer_reported_nothing_at_all(&messages);
}

/// [`a_separate_multisampled_16_bit_depth_keeps_every_code`] on `device`.
fn separate_16_bit_codes(device: &Device) {
    let name = device.name().to_owned();
    let stencil_texture = multisampled_for(
        device,
        PixelFormat::Stencil8,
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        CODES_EXTENT,
    );
    let (depth, stencil_texture) = match (depth16(device), stencil_texture) {
        (Ok(depth), Ok(stencil_texture)) => (depth, stencil_texture),
        (Err(Error::Unsupported(u)), _) | (_, Err(Error::Unsupported(u))) => {
            eprintln!("separate 16-bit depth on {name}: textures refused, {u:?}");
            return;
        }
        (Err(error), _) | (_, Err(error)) => panic!("new_texture: {error}"),
    };
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    draw_depths(device, &command_buffer, &depth, &every_code());
    let before = read_depths(device, command_buffer, &depth);
    let command_buffer = queue.command_buffer().unwrap();
    let actions = Actions {
        loads: [LoadAction::Load, LoadAction::Clear],
        stores: [StoreAction::Store; 2],
        clears: (0.0, 0x40),
        draw: None,
    };
    let defect = RuntimeDriverDefect::MultisampledDepthStencilColourCopies;
    let by_draws = device.properties().driver_defects.contains(&defect)
        || !device.enabled().features.contains(&VkFeature::Maintenance8);
    let keeps = device
        .properties()
        .driver_facts
        .contains(&DriverFact::FragmentDepthKeepsDepth16Codes);
    match pass(
        device,
        &command_buffer,
        (&depth, &stencil_texture),
        &actions,
    ) {
        Ok(()) => {
            assert!(
                !by_draws || keeps,
                "{name} copied a depth its draws do not keep"
            );
            let (after, stencils_read) =
                read_back(device, command_buffer, &depth, &stencil_texture);
            let after = words(&after);
            let differ = before.iter().zip(&after).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "{name}: samples whose code the pass changed");
            assert_eq!(
                stencils_read,
                stencils([0x40; 4], CODES_EXTENT),
                "{name}: the stencil's clear"
            );
            let distinct = before
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            eprintln!(
                "separate 16-bit depth on {name}: every sample kept ({distinct} different codes), \
                 {}",
                if by_draws {
                    "by draws"
                } else {
                    "by colour copies"
                }
            );
        }
        // A format the device lacks (lavapipe and ANV make no `D16_UNORM_S8_UINT` attachment)
        // is its gap; past it, the draws run where they keep the codes.
        Err(Error::Unsupported(u)) if u.feature == Feature::Format => {
            eprintln!("separate 16-bit depth on {name}: refused, {u:?}");
        }
        Err(Error::Unsupported(u)) => {
            eprintln!("separate 16-bit depth on {name}: refused, {u:?}");
            assert!(by_draws && !keeps, "{name} refused what it copies: {u:?}");
            assert_eq!(
                u.feature,
                Feature::SeparateMultisampledDepthStencil,
                "{name}: {u:?}"
            );
        }
        Err(error) => panic!("render_command_encoder: {error}"),
    }
}
