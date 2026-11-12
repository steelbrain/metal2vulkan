//! A sampler state's `lodBias` on the device the cfg selects, under the validation layer, as the
//! Mac the device presents samples it (`DeviceOptions::mac_class`):
//!
//! - an Apple silicon Mac samples nothing differently for it (an Apple M3 Max on macOS 27.0.1,
//!   Apple9: every sample kind, biases from -1000 to 1000, NaN and the infinities);
//! - a Mac with an Intel or AMD GPU adds it to every sample's level, explicit, gradient and
//!   implicit alike, and clamps the sum into the texture's levels (an AMD Radeon Pro 5300M and an
//!   Intel UHD 630, the consumer's probe): the device adds what its samplers take of it
//!   (`metal::sampler::device_lod_bias`), a bias they cannot take as it is announced once
//!   ([`CAPPED_LOD_BIAS`]).
//!
//! Either way Metal makes another state for another bias, one state for `0` and `-0`, and one for
//! two NaNs of one encoding.
//!
//! The functions are `mip_level_sampling`'s (`validation/fixtures/public/engine`), over an
//! `r32Float` texture of five levels whose level `L` holds `L` in every texel, so a nearest
//! sample, or a linear mip filter's at a whole level, states the level it read. Its kernel
//! `mip_bias_kernel` samples the runtime sampler at slot 0 at each thread's coordinate with its
//! own shader bias, a kernel's default level 0 shifted by it: the device's explicit level. Its
//! fragment function `mip_implicit_fragment`, drawn by `tob_renders`'s `tob_rect` over a target of
//! one row of 2x2 quads, samples quad `q` at an implicit level of exactly `q`.

use super::degrade_notices::{assert_once, counts};
use super::mip_levels::{texture, LEVELS};
use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with,
    Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::sampler::{device_lod_bias, DeviceLodBias};
use metal2vulkan::metal::{
    PixelFormat, PrimitiveType, SamplerMinMagFilter, SamplerMipFilter, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, CommandBufferStatus, ComputePipelineState, Degradation, Device, LoadAction,
    MacClass, PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    SamplerState, Size, StoreAction, Texture, TextureDescriptor,
};
use std::collections::BTreeMap;

/// The kind a state of a bias the device's samplers cannot take as it is takes, on a device
/// presenting a Mac with an Intel or AMD GPU
/// ([`a_sampler_s_lod_bias_moves_the_level_on_an_intel_or_amd_mac`]).
pub(super) const CAPPED_LOD_BIAS: [Degradation; 1] = [Degradation::CappedLodBias];

/// The shader biases each thread samples at, one thread each: whole levels, levels between them,
/// below the first and past the last.
const SHADER_BIASES: [f32; 9] = [0.0, 0.5, 1.0, 1.25, 2.0, 3.0, 4.0, 6.0, -1.0];

/// The sampler biases, each a state of its own beside the unbiased one: small ones of both signs,
/// Apple's stated range's ends and past them, values past every device's `maxSamplerLodBias`, a
/// NaN and an infinity.
const SAMPLER_BIASES: [f32; 10] = [
    1.0,
    -1.0,
    2.5,
    15.999,
    -16.0,
    17.0,
    -17.0,
    100.0,
    f32::NAN,
    f32::INFINITY,
];

/// The quads along the fragment function's target: quad `q` samples at level `q`, the last one
/// past the texture's last level.
const QUADS: u64 = LEVELS + 1;

/// The kernel on `device`.
fn pipeline(device: &Device) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("mip_level_sampling.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name("mip_bias_kernel").unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// `tob_renders`'s `tob_rect` and `mip_level_sampling`'s `mip_implicit_fragment` on `device`, into
/// an `RGBA32Float` color attachment.
fn implicit_pipeline(device: &Device) -> RenderPipelineState {
    let function = |fixture: &str, name: &str| {
        let bitcode = llvm_assemble(&engine_dir().join(fixture)).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_renders.ll", "tob_rect"));
    descriptor.fragment = Some(function("mip_level_sampling.ll", "mip_implicit_fragment"));
    descriptor.color_attachments[0] =
        Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A sampler state of nearest filters, mip filter `mip` and LOD bias `bias`.
fn sampler(device: &Device, mip: SamplerMipFilter, bias: f32) -> SamplerState {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.min_filter = SamplerMinMagFilter::Nearest;
    descriptor.mag_filter = SamplerMinMagFilter::Nearest;
    descriptor.mip_filter = mip;
    descriptor.lod_bias = bias;
    device.new_sampler_state(&descriptor).unwrap()
}

/// What each thread reads of `texture` through `sampler` at its shader bias.
fn run(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
    sampler: &SamplerState,
) -> Vec<f32> {
    let params: Vec<u8> = SHADER_BIASES
        .iter()
        .flat_map(|&b| [0.3f32, 0.6, b, 0.0])
        .flat_map(f32::to_le_bytes)
        .collect();
    let params = buffer_with(device, &params).unwrap();
    let length = (SHADER_BIASES.len() * 16) as u64;
    let output = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_sampler_state(Some(sampler), 0).unwrap();
    encoder.set_buffer(Some(&params), 0, 0).unwrap();
    encoder.set_buffer(Some(&output), 0, 1).unwrap();
    let threads = SHADER_BIASES.len() as u64;
    encoder
        .dispatch_threads(Size::new(threads, 1, 1), Size::new(threads, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    // The red channel of each thread's `float4`.
    read_buffer(&output, 0, length)
        .unwrap()
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(c[..4].try_into().unwrap()))
        .collect()
}

/// What each quad of the fragment function's target reads of `texture` through `sampler` at its
/// implicit level, every pixel of the quad alike.
fn drawn(
    device: &Device,
    pipeline: &RenderPipelineState,
    texture: &Texture,
    sampler: &SamplerState,
) -> Vec<f32> {
    let (w, h) = (2 * QUADS, 2);
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, w, h, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&target).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(-1.0, 0.0, 0.0, 0.0);
    let corners: Vec<u8> = [-1.0f32, -1.0, 1.0, 1.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let corners = buffer_with(device, &corners).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
    encoder.set_fragment_texture(Some(texture), 0).unwrap();
    encoder
        .set_fragment_sampler_state(Some(sampler), 0)
        .unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let mut bytes = vec![0u8; (16 * w * h) as usize];
    target
        .get_bytes(&mut bytes, 16 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
        .unwrap();
    let red: Vec<f32> = bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(c[..4].try_into().unwrap()))
        .collect();
    (0..QUADS as usize)
        .map(|q| {
            let quad = [red[2 * q], red[2 * q + 1], red[w as usize + 2 * q]];
            let last = red[w as usize + 2 * q + 1];
            assert!(
                quad.iter().all(|r| r.to_bits() == last.to_bits()),
                "quad {q}: {quad:?} {last}"
            );
            last
        })
        .collect()
}

/// A lane device presenting a Mac of `class`.
fn presenting(messages: &Messages, cache: &ScratchCache, class: MacClass) -> Device {
    validated_device_with(TEST_DEVICE, messages, cache.path(), |options| {
        at_plan(options);
        options.mac_class = Some(class);
    })
    .unwrap()
}

/// On a device presenting an Apple silicon Mac, every sample through a state with a LOD bias
/// reads what the unbiased state reads, bit for bit: the kernel's under a linear and a nearest mip
/// filter, the level the shader bias alone selects, which the unbiased state reads at each whole
/// level; and the fragment function's implicit level `q` under a nearest one. A state of another
/// bias is another state, `-0` is `0`'s, and a NaN bias's state is the next NaN's. Nothing is
/// announced for any bias. Prints the samples checked.
#[test]
fn a_sampler_s_lod_bias_reads_the_unbiased_level_on_an_apple_silicon_mac() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = presenting(&messages, &cache, MacClass::AppleSilicon);
        let name = device.name().to_owned();
        let levels = texture(&device, TextureType::Type2D, |l, _, _, _| l as f32);
        let state = pipeline(&device);
        let render = implicit_pipeline(&device);
        let mut checked = 0;
        for mip in [SamplerMipFilter::Linear, SamplerMipFilter::Nearest] {
            let unbiased = sampler(&device, mip, 0.0);
            let want = run(&device, &state, &levels, &unbiased);
            // The kernel reads levels: each whole shader bias's level, the texture's levels
            // bounding it.
            for (&b, &w) in SHADER_BIASES.iter().zip(&want) {
                if b.fract() == 0.0 {
                    let level = b.clamp(0.0, (LEVELS - 1) as f32);
                    assert_eq!(w, level, "{name}: {mip:?}, shader bias {b}");
                }
            }
            let implicit = (mip == SamplerMipFilter::Nearest)
                .then(|| drawn(&device, &render, &levels, &unbiased));
            if let Some(implicit) = &implicit {
                for (q, &got) in implicit.iter().enumerate() {
                    let level = (q as f32).min((LEVELS - 1) as f32);
                    assert_eq!(got, level, "{name}: implicit level {q}");
                }
            }
            for bias in SAMPLER_BIASES {
                messages.take_notices();
                let biased = sampler(&device, mip, bias);
                assert_ne!(
                    biased.gpu_resource_id(),
                    unbiased.gpu_resource_id(),
                    "{name}: bias {bias} is another state"
                );
                assert_eq!(
                    counts(&messages.take_notices()),
                    BTreeMap::new(),
                    "{name}: bias {bias}"
                );
                let got = run(&device, &state, &levels, &biased);
                for ((&b, g), w) in SHADER_BIASES.iter().zip(&got).zip(&want) {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "{name}: {mip:?}, sampler bias {bias}, shader bias {b}: {g} against \
                         the unbiased {w}"
                    );
                    checked += 1;
                }
                if let Some(implicit) = &implicit {
                    let got = drawn(&device, &render, &levels, &biased);
                    for (q, (g, w)) in got.iter().zip(implicit).enumerate() {
                        assert_eq!(
                            g.to_bits(),
                            w.to_bits(),
                            "{name}: sampler bias {bias}, implicit level {q}: {g} against the \
                             unbiased {w}"
                        );
                        checked += 1;
                    }
                }
            }
            let zero = sampler(&device, mip, -0.0);
            assert_eq!(zero.gpu_resource_id(), unbiased.gpu_resource_id());
            let (nan, again) = (
                sampler(&device, mip, f32::NAN),
                sampler(&device, mip, f32::NAN),
            );
            assert_eq!(nan.gpu_resource_id(), again.gpu_resource_id());
        }
        println!("LOD-BIAS {name}: {checked} biased samples read the unbiased level");
    }
    layer_reported_nothing_at_all(&messages);
}

/// The sampler biases of [`a_sampler_s_lod_bias_moves_the_level_on_an_intel_or_amd_mac`]: whole
/// ones of both signs, so each whole shader bias's sum is a whole level, ones past every device's
/// `maxSamplerLodBias`, an infinity and a NaN.
const INTEL_OR_AMD_BIASES: [f32; 8] =
    [1.0, -1.0, 2.0, -3.0, 100.0, -100.0, f32::INFINITY, f32::NAN];

/// On a device presenting a Mac with an Intel or AMD GPU, a sample through a state with a LOD
/// bias reads the level its own level and the bias select, clamped into the texture's levels: the
/// kernel's at its shader bias plus the device's bias, under a linear and a nearest mip filter,
/// and the fragment function's at its implicit level `q` plus it, under a nearest one. The
/// device's bias is what its samplers take of the state's (`device_lod_bias`): the bias, its
/// limit past it, none for a NaN and none where the device withholds `samplerMipLodBias`, the
/// state announced `capped-lod-bias` once where it is not the bias itself. Checked where the sum
/// is a whole level; on lavapipe the biased levels differ from the unbiased ones. Prints the
/// samples checked and how many read another level than the unbiased state.
#[test]
fn a_sampler_s_lod_bias_moves_the_level_on_an_intel_or_amd_mac() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = presenting(&messages, &cache, MacClass::IntelOrAmd);
        let name = device.name().to_owned();
        let profile = device.compile_device().profile();
        let levels = texture(&device, TextureType::Type2D, |l, _, _, _| l as f32);
        let state = pipeline(&device);
        let render = implicit_pipeline(&device);
        let last = (LEVELS - 1) as f32;
        let (mut checked, mut moved) = (0, 0);
        let capped = INTEL_OR_AMD_BIASES
            .iter()
            .any(|&b| !matches!(device_lod_bias(b, profile), DeviceLodBias::Exact(_)));
        for mip in [SamplerMipFilter::Linear, SamplerMipFilter::Nearest] {
            let unbiased = sampler(&device, mip, 0.0);
            let want = run(&device, &state, &levels, &unbiased);
            messages.take_notices();
            let states: Vec<(f32, SamplerState)> = INTEL_OR_AMD_BIASES
                .iter()
                .map(|&bias| (bias, sampler(&device, mip, bias)))
                .collect();
            // Announced once per device: at the first mip filter's states.
            let notices = messages.take_notices();
            if capped && mip == SamplerMipFilter::Linear {
                assert_eq!(assert_once(&notices, &CAPPED_LOD_BIAS).len(), 1, "{name}");
            } else {
                assert_eq!(counts(&notices), BTreeMap::new(), "{name}: {mip:?}");
            }
            for (bias, biased) in &states {
                assert_ne!(
                    biased.gpu_resource_id(),
                    unbiased.gpu_resource_id(),
                    "{name}: bias {bias} is another state"
                );
                let added = device_lod_bias(*bias, profile).bias();
                let got = run(&device, &state, &levels, biased);
                for ((&b, &g), &w) in SHADER_BIASES.iter().zip(&got).zip(&want) {
                    let sum = b + added;
                    if sum.fract() != 0.0 {
                        continue;
                    }
                    let level = sum.clamp(0.0, last);
                    assert_eq!(
                        g, level,
                        "{name}: {mip:?}, sampler bias {bias} (the device's {added}), shader \
                         bias {b}: {g} against level {level}"
                    );
                    checked += 1;
                    moved += usize::from(g.to_bits() != w.to_bits());
                }
                if mip != SamplerMipFilter::Nearest {
                    continue;
                }
                for (q, &g) in drawn(&device, &render, &levels, biased).iter().enumerate() {
                    let sum = q as f32 + added;
                    if sum.fract() != 0.0 {
                        continue;
                    }
                    let level = sum.clamp(0.0, last);
                    let unbiased_level = (q as f32).min(last);
                    assert_eq!(
                        g, level,
                        "{name}: sampler bias {bias} (the device's {added}), implicit level {q}: \
                         {g} against level {level}"
                    );
                    checked += 1;
                    moved += usize::from(g != unbiased_level);
                }
            }
            let zero = sampler(&device, mip, -0.0);
            assert_eq!(zero.gpu_resource_id(), unbiased.gpu_resource_id());
            let (nan, again) = (
                sampler(&device, mip, f32::NAN),
                sampler(&device, mip, f32::NAN),
            );
            assert_eq!(nan.gpu_resource_id(), again.gpu_resource_id());
        }
        let withheld = matches!(device_lod_bias(1.0, profile), DeviceLodBias::Withheld);
        assert!(checked > 0, "{name}");
        // Where the device takes a bias, the biased states read other levels.
        assert_eq!(
            moved > 0,
            !withheld,
            "{name}: {moved} moved, withheld {withheld}"
        );
        println!(
            "LOD-BIAS {name}: {checked} biased samples read the biased level, {moved} of them \
             another than the unbiased state's"
        );
    }
    layer_reported_nothing_at_all(&messages);
}
