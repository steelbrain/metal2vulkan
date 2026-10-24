//! Blit copies between multisampled depth or stencil textures and colour textures of their byte
//! size (`runtime::sample_copy`) on the device the cfg selects, under the validation layer, against
//! what a probe measured on an Apple M3 Max (macOS 26.7) and an M4 (macOS 27.0.1), plain Metal:
//! each sample's bytes land in the same sample of the destination.
//!
//! Each colour texture is filled sample by sample (`candidate::runtime_samples`), copied into the
//! depth or stencil texture and back into a fresh colour texture, and both are read back sample
//! by sample; a 32-bit float depth texture is also filled by a draw and copied out alone. Depths
//! lie in `[0, 1]`, so no device's clamping decides. Where the device lacks maintenance8 or its
//! driver copies such images wrong (lavapipe, MoltenVK, NVIDIA's Ampere GA10x:
//! `RuntimeDriverDefect::MultisampledDepthStencilColourCopies`), the copies go by sample-rate
//! draws. A copy into a 32-bit float depth there is exact where the driver's fragment depth
//! stores every value (`DriverFact::FragmentDepthStoresUnchanged`), and elsewhere (measured to
//! change some values, MoltenVK's and NVIDIA's, or not measured) runs announced once
//! ([`DRAWN_DEPTH`]), exact for these depths in `[0, 1]`; for a 16-bit depth the copy is refused,
//! typed, naming `Feature::Format`.

use super::degrade_notices::shared_in_system_memory;
use super::degrade_notices::signs_depth_tests;
use super::degrade_notices::{assert_made_once, clamps_depth_uploads, forced};
use super::degrade_notices::{loses_depth_denormals, made_compute_kinds, made_render_kinds};
use super::{validated_device, Messages, ScratchCache, WAIT};
use crate::candidate::runtime_samples::SampleFill;
use crate::candidate::runtime_samples::{encode, fill_descriptor, read_function, read_view};
use crate::case::TextureFormat;
use crate::executor_contract::multisample::{ReadbackView, SampleRead};
use crate::gpu::TEST_DEVICE;
use crate::literal::TextureLayout;
use metal2vulkan::device::{DriverFact, VkFeature};
use metal2vulkan::error::{Capability, Feature, Needs, RuntimeDriverDefect, Unsupported};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    BlitCommandEncoder, CommandBufferStatus, Degradation, Device, Error, LoadAction, Origin,
    RenderPassDescriptor, ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
};
use std::collections::BTreeSet;

/// The kind a copy into a 32-bit float depth by the fragment-depth draw takes where the
/// driver's fragment depth changes some values
/// ([`multisampled_depth_and_stencil_copy_every_sample_with_colour`]).
pub(super) const DRAWN_DEPTH: [Degradation; 1] = [Degradation::DrawnDepthCopy];

/// The textures' extent and samples.
const EXTENT: [u32; 2] = [4, 2];
const SAMPLES: u32 = 4;

/// Every sample of the extent.
const TEXELS: usize = (EXTENT[0] * EXTENT[1] * SAMPLES) as usize;

/// A private 2D multisample texture of `pf` the fills render and the readbacks read.
fn multisampled(device: &Device, pf: PixelFormat) -> Result<Texture, Error> {
    let mut d = TextureDescriptor::texture_2d(pf, EXTENT[0].into(), EXTENT[1].into(), false);
    d.texture_type = TextureType::Type2DMultisample;
    d.sample_count = SAMPLES.into();
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d)
}

/// `n` bytes a sample, every sample distinct and seeded: depth values in `[0, 1]` for 4-byte
/// floats, any code otherwise.
fn values(n: usize, float_depth: bool, seed: u32) -> Vec<u8> {
    let mut x = seed;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x
    };
    (0..TEXELS)
        .flat_map(|i| {
            if float_depth {
                // 0, 1 and steps of 2^-24 between: each exact in a 32-bit float depth.
                let v = match i {
                    0 => 0.0f32,
                    1 => 1.0,
                    _ => (next() >> 8) as f32 / 16_777_216.0,
                };
                v.to_le_bytes().to_vec()
            } else {
                next().to_le_bytes()[..n].to_vec()
            }
        })
        .collect()
}

/// One pair the test copies between: a colour format the fill writes, its readback, the
/// depth/stencil format, whether the copy names its stencil view, and how the depth/stencil
/// texture is read.
struct Pair {
    colour: TextureFormat,
    view: ReadbackView,
    depth_stencil: PixelFormat,
    stencil_view: bool,
    read: SampleRead,
}

const PAIRS: [Pair; 6] = [
    Pair {
        colour: TextureFormat::R32Float,
        view: ReadbackView::R32Uint,
        depth_stencil: PixelFormat::Depth32Float,
        stencil_view: false,
        read: SampleRead::Depth,
    },
    Pair {
        colour: TextureFormat::R32Uint,
        view: ReadbackView::R32Uint,
        depth_stencil: PixelFormat::Depth32Float,
        stencil_view: false,
        read: SampleRead::Depth,
    },
    // Not in maintenance8's table: through a colour image of the depth's bits.
    Pair {
        colour: TextureFormat::Rgba8Unorm,
        view: ReadbackView::R32Uint,
        depth_stencil: PixelFormat::Depth32Float,
        stencil_view: false,
        read: SampleRead::Depth,
    },
    Pair {
        colour: TextureFormat::R16Uint,
        view: ReadbackView::R16Uint,
        depth_stencil: PixelFormat::Depth16Unorm,
        stencil_view: false,
        read: SampleRead::Depth,
    },
    Pair {
        colour: TextureFormat::R8Unorm,
        view: ReadbackView::R8Uint,
        depth_stencil: PixelFormat::Stencil8,
        stencil_view: false,
        read: SampleRead::Stencil,
    },
    Pair {
        colour: TextureFormat::R8Unorm,
        view: ReadbackView::R8Uint,
        depth_stencil: PixelFormat::Depth32FloatStencil8,
        stencil_view: true,
        read: SampleRead::Stencil,
    },
];

/// How one pair ended.
enum Outcome {
    Ran,
    Refused(Unsupported),
}

/// The copy forms the pairs take in turn: by region, by slices, and whole.
fn copy(
    blit: &mut BlitCommandEncoder,
    form: usize,
    from: &Texture,
    to: &Texture,
) -> Result<(), Error> {
    let at = Origin::new(0, 0, 0);
    let size = Size::new(EXTENT[0].into(), EXTENT[1].into(), 1);
    match form % 3 {
        0 => blit.copy_from_texture_to_texture(from, 0, 0, at, size, to, 0, 0, at),
        1 => blit.copy_texture_slices(from, 0, 0, to, 0, 0, 1, 1),
        _ => blit.copy_texture(from, to),
    }
}

/// Copy `pair`'s colour texture, filled with seeded samples, into its depth/stencil texture and
/// back into a fresh colour texture by copy form `form`, and check every sample of both.
fn run_pair(device: &Device, pair: &Pair, form: usize) -> Outcome {
    let colour_pf = crate::candidate::engine::resources::pixel_format(pair.colour);
    let texel = pair.colour.bytes_per_pixel();
    let float_depth = pair.depth_stencil == PixelFormat::Depth32Float;
    let filled = values(texel, float_depth, 0x9e37_79b9 ^ form as u32);
    let colour = multisampled(device, colour_pf).unwrap();
    let back = multisampled(device, colour_pf).unwrap();
    let depth_stencil = multisampled(device, pair.depth_stencil).unwrap();
    let side = if pair.stencil_view {
        depth_stencil
            .new_texture_view(PixelFormat::X32Stencil8)
            .unwrap()
    } else {
        depth_stencil.clone()
    };
    let layout = TextureLayout {
        width: EXTENT[0],
        height: EXTENT[1],
        depth: 1,
        array_layers: 1,
        sample_count: SAMPLES,
    };
    let fill = SampleFill::new(device, &colour, pair.colour, layout, &filled).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    fill.encode(&command_buffer).unwrap();
    if pair.stencil_view {
        // The combined texture's depth, cleared to 0.5, which the stencil's copies keep.
        let mut pass = RenderPassDescriptor::new();
        pass.depth_attachment.texture = Some(depth_stencil.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.store_action = StoreAction::Store;
        pass.depth_attachment.clear_depth = 0.5;
        pass.stencil_attachment.texture = Some(depth_stencil.clone());
        pass.stencil_attachment.load_action = LoadAction::Clear;
        pass.stencil_attachment.store_action = StoreAction::Store;
        let encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let copied =
        copy(&mut blit, form, &colour, &side).and_then(|()| copy(&mut blit, form, &side, &back));
    blit.end_encoding().unwrap();
    if let Err(error) = copied {
        return match error {
            Error::Unsupported(u) => Outcome::Refused(u),
            other => panic!("{pair:?}: {other}", pair = pair.depth_stencil),
        };
    }
    let read_side = encode(device, &command_buffer, &side, pair.read, EXTENT, SAMPLES).unwrap();
    let read_back = encode(
        device,
        &command_buffer,
        &back,
        SampleRead::Color(pair.view),
        EXTENT,
        SAMPLES,
    )
    .unwrap();
    let kept = pair.stencil_view.then(|| {
        encode(
            device,
            &command_buffer,
            &depth_stencil,
            SampleRead::Depth,
            EXTENT,
            SAMPLES,
        )
        .unwrap()
    });
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let name = format!(
        "{:?} with {:?}, form {form}",
        pair.colour, pair.depth_stencil
    );
    assert_eq!(read_back.bytes().unwrap(), filled, "{name}: the copy back");
    let side_bytes = read_side.bytes().unwrap();
    match pair.depth_stencil {
        // A 16-bit depth reads as its normalized value: each code exactly.
        PixelFormat::Depth16Unorm => {
            let codes: Vec<u8> = side_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|w| {
                    let v = f32::from_le_bytes([w[0], w[1], w[2], w[3]]);
                    ((v * 65535.0).round() as u16).to_le_bytes()
                })
                .collect();
            assert_eq!(codes, filled, "{name}: the depth's codes");
        }
        _ => assert_eq!(side_bytes, filled, "{name}: the depth/stencil's samples"),
    }
    if let Some(kept) = kept {
        let half = 0.5f32.to_le_bytes();
        assert!(
            kept.bytes()
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .all(|w| *w == half),
            "{name}: the combined texture's depth kept"
        );
    }
    Outcome::Ran
}

/// Fill a multisampled `Depth32Float` texture by a draw with seeded depths, copy it into a fresh
/// texture of `pair`'s colour format by copy form `form`, and check every sample.
fn run_out(device: &Device, pair: &Pair, form: usize) {
    let colour_pf = crate::candidate::engine::resources::pixel_format(pair.colour);
    let filled = values(4, true, 0x2545_f491 ^ form as u32);
    let depth = multisampled(device, PixelFormat::Depth32Float).unwrap();
    let back = multisampled(device, colour_pf).unwrap();
    let layout = TextureLayout {
        width: EXTENT[0],
        height: EXTENT[1],
        depth: 1,
        array_layers: 1,
        sample_count: SAMPLES,
    };
    let fill =
        SampleFill::new(device, &depth, TextureFormat::Depth32Float, layout, &filled).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    fill.encode(&command_buffer).unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let name = format!("{:?} out of a drawn Depth32Float, form {form}", pair.colour);
    copy(&mut blit, form, &depth, &back).unwrap_or_else(|e| panic!("{name}: {e}"));
    blit.end_encoding().unwrap();
    let read_back = encode(
        device,
        &command_buffer,
        &back,
        SampleRead::Color(pair.view),
        EXTENT,
        SAMPLES,
    )
    .unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(read_back.bytes().unwrap(), filled, "{name}");
}

/// Every pair, by each copy form in turn, both ways: each sample's bytes kept in the same
/// sample, by maintenance8's copies or by draws; or, where neither copies the pair exactly, the
/// copy refused, typed. A 32-bit float depth's copy out runs on every device.
#[test]
fn multisampled_depth_and_stencil_copy_every_sample_with_colour() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let defects = &device.properties().driver_defects;
        let maintenance8 = device.enabled().features.contains(&VkFeature::Maintenance8);
        let exact_depth = device
            .properties()
            .driver_facts
            .contains(&DriverFact::FragmentDepthStoresUnchanged);
        let exact_codes = device
            .properties()
            .driver_facts
            .contains(&DriverFact::FragmentDepthKeepsDepth16Codes);
        messages.take_notices();
        // Whether a copy into a 32-bit float depth ran by a fragment-depth draw that is not exact.
        let mut degraded = false;
        // Whether a copy into a 32-bit float depth ran, a readback viewed a colour texture in
        // another format than its own, and the kinds the fills' and readbacks' state makings
        // announce.
        let mut float_depth = false;
        let mut views_apart = false;
        let mut made = BTreeSet::new();
        let render = |format| {
            let descriptor = fill_descriptor(&device, format, SAMPLES).unwrap();
            made_render_kinds(&device, &descriptor)
        };
        let compute = |format, read| {
            made_compute_kinds(&device, read_function(&device, format, read).unwrap())
        };
        for (form, pair) in PAIRS.iter().enumerate() {
            let colour_pf = crate::candidate::engine::resources::pixel_format(pair.colour);
            made.extend(render(pair.colour));
            // The driver's defect for the pair's aspect: every aspect's, or the stencil's alone.
            let stencil = matches!(pair.read, SampleRead::Stencil);
            let every = RuntimeDriverDefect::MultisampledDepthStencilColourCopies;
            let stencil_only = RuntimeDriverDefect::MultisampledStencilColourCopies;
            let defect = if defects.contains(&every) {
                Some(every)
            } else {
                (stencil && defects.contains(&stencil_only)).then_some(stencil_only)
            };
            let copies = maintenance8 && defect.is_none();
            // The draws copy a stencil (the plane draws are exact), a 32-bit float depth where
            // the driver's fragment depth stores it unchanged, and a 16-bit normalized one where
            // the driver keeps its codes, which the draw crosses as codes rather than as floats
            // (`depth_copy::copies_exactly`). A 32-bit float depth's copy in elsewhere runs
            // announced.
            let drawn = if matches!(pair.read, SampleRead::Stencil) {
                true
            } else if pair.depth_stencil == PixelFormat::Depth16Unorm {
                exact_codes
            } else {
                exact_depth
            };
            let near = !copies && !drawn && pair.depth_stencil == PixelFormat::Depth32Float;
            match run_pair(&device, pair, form) {
                Outcome::Ran => {
                    assert!(
                        copies || drawn || near,
                        "{name} ran what neither its copies nor its draws copy"
                    );
                    degraded |= near;
                    float_depth |= pair.depth_stencil == PixelFormat::Depth32Float;
                    views_apart |= read_view(colour_pf).unwrap() != colour_pf;
                    made.extend(compute(pair.depth_stencil, pair.read));
                    made.extend(compute(colour_pf, SampleRead::Color(pair.view)));
                    if pair.stencil_view {
                        made.extend(compute(pair.depth_stencil, SampleRead::Depth));
                    }
                    eprintln!(
                        "{name}: {:?} with {:?} copied every sample both ways",
                        pair.colour, pair.depth_stencil
                    );
                }
                Outcome::Refused(u) => {
                    assert!(!copies && !drawn, "{name} refused what it copies: {u:?}");
                    assert_eq!(u.feature, Feature::Format, "{u:?}");
                    if !maintenance8 {
                        assert_eq!(
                            u.needs,
                            Needs::AnyOf(vec![vec![Capability::Feature(VkFeature::Maintenance8)]]),
                            "{u:?}"
                        );
                    } else {
                        let defect = defect.expect("refused for a defect");
                        assert_eq!(u.needs, Needs::RuntimeDriverDefect(defect), "{u:?}");
                    }
                    eprintln!(
                        "{name}: {:?} with {:?} refused, {u:?}",
                        pair.colour, pair.depth_stencil
                    );
                }
            }
            if pair.depth_stencil == PixelFormat::Depth32Float {
                made.extend(render(TextureFormat::Depth32Float));
                views_apart |= read_view(colour_pf).unwrap() != colour_pf;
                made.extend(compute(colour_pf, SampleRead::Color(pair.view)));
                run_out(&device, pair, form);
                eprintln!(
                    "{name}: {:?} out of a drawn Depth32Float, every sample",
                    pair.colour
                );
            }
        }
        // Each fill loads its texture `DontCare` and draws where the host cannot bound
        // (`dont-care-load-tiles`), from `Shared` buffers; a readback views a colour texture
        // made without `PixelFormatView` usage in another format apart; and a copy into a float
        // depth announces what its device may store.
        let mut first = forced(&[
            (Degradation::DontCareLoadTiles, true),
            (Degradation::ShadowedTextureView, views_apart),
            (
                Degradation::DepthDenormals,
                float_depth && loses_depth_denormals(&device, false),
            ),
            (
                Degradation::RestrictedDepthUpload,
                float_depth && clamps_depth_uploads(&device),
            ),
            (
                Degradation::SignedDepthTest,
                float_depth && signs_depth_tests(&device),
            ),
            (
                Degradation::SharedInSystemMemory,
                shared_in_system_memory(&device),
            ),
            (Degradation::DrawnDepthCopy, degraded),
        ]);
        first.extend(made);
        let announced = assert_made_once(&messages, &[], &first);
        for kind in DRAWN_DEPTH {
            assert_eq!(
                announced.contains_key(&kind),
                degraded,
                "{kind:?} on {name} exactly where a depth copy in ran by an inexact draw: \
                 {announced:?}"
            );
        }
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}
