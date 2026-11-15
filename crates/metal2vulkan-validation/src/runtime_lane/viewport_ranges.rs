//! Device tests of viewports past Vulkan's: depth ranges past `[0, 1]` and negative widths
//! on the device the cfg selects, under the validation layer.
//!
//! Each case is one a Metal probe measured (an M3 Max with macOS 26.7 and an M4 with macOS 27.0.1
//! alike), drawn with the same functions (the owned fixture `rfb_viewports`) into the same pass: a
//! 4² `RGBA32Float` target cleared to -9 and a `Depth32Float` depth cleared to 1, written under
//! `Always`. The fullscreen triangle at depth `z + slope * x` is drawn through one viewport of the
//! full height, in a depth clip mode, with a depth bias or a cull mode where the case names one.
//! The fragment function stores the position it reads and its facing, or writes a depth and stores
//! the position's depth. What it stored and the depth stored are compared with Metal's, within
//! `TOLERANCE`.
//!
//! Metal's answers: the fragment reads the depth `n + (f - n) z` unclamped, a constant depth bias
//! included. In clip mode a primitive clips at the near and far planes and each depth stored,
//! interpolated or written, is clamped to `[0, 1]`. In clamp mode nothing clips and each depth
//! is clamped to `[0, 1]`, then to the range. A negative width mirrors the draw across the
//! viewport's origin, which flips its facing.
//!
//! Where the device takes a depth range natively (`VK_EXT_depth_range_unrestricted`,
//! `depthClamp`, `depthClipEnable` and `depthClampControl`), every case past `[0, 1]` draws.
//! Elsewhere the vertex stage compensates it where the device has `shaderClipDistance` and
//! `depthClamp` and its driver draws the compensated depths exactly (`depth_ranges_refused`):
//! every case draws but a clamp mode one whose one clamp lies outside `[0, 1]` or holds one depth,
//! refused at the draw, `NotImplemented(ViewportDepthRange)`, as its fragment function reads the
//! position's depth (a function that reads none draws, the draw's Vulkan range collapsed to the
//! one depth: `one_depth_clamps_store_their_depth`). Without the features `set_viewport` refuses
//! the range `NotImplemented`, and on MoltenVK, whose sloped compensated depths are about 2e-4 off
//! Metal's, `Unsupported` with its driver defect. A negative width draws on every device, the
//! vertex stage mirroring it. The two clamp mode cases whose depths cross 1 within the primitive
//! are the open near-equivalent: Metal's rule there is unidentified, so they print the device's
//! depths beside Metal's and assert that they lie in the clamp.
//!
//! A written depth's special values (Metal probes, an M3 Max and an M4 alike) are compared by their
//! bits: through viewports of `[0, 1]` and `[0.25, 0.5]` in both clip modes, Metal stores +0 for
//! any value with the sign bit set, 1 for one past 1 up to +inf, a positive denormal unchanged, and
//! then the range's clamp in clamp mode. A positive NaN fails the depth test: neither depth nor
//! color is written. A negative NaN stores 0 even past a clamp above it, which the fragment
//! function's variant stores where the draw's Vulkan clamp can hold 0; elsewhere the draw announces
//! `nan-depth` and the row prints. A positive denormal may store +0 where the device's depth
//! storage flushes it (RADV, NVIDIA and ANV; x86 lavapipe through a depth clamp), the
//! near-equivalent.

use super::degrade_notices::{assert_made_once, forced, made_compute_kinds};
use super::degrade_notices::{made_render_kinds, resolve_kind, shared_in_system_memory};
use super::render_state::{quiet, texels, texture, Probe};
use super::{validated_device, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::candidate::runtime_case::buffer_with;
use crate::candidate::runtime_samples::encode as samples_of;
use crate::candidate::runtime_samples::read_function;
use crate::executor_contract::multisample::SampleRead;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::{DriverFact, VkExtension, VkFeature};
use metal2vulkan::error::{Feature, RuntimeDriverDefect, Unsupported};
use metal2vulkan::metal::state::StencilDescriptor;
use metal2vulkan::metal::{
    CompareFunction, CullMode, DepthClipMode, PixelFormat, PrimitiveType, StencilOperation,
    TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    Buffer, ClearColor, Degradation, Device, Error, LoadAction, PipelineOptions,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions, RuntimeGap,
    StoreAction, Texture, TextureDescriptor, Viewport,
};
use std::collections::BTreeSet;

/// How far a depth the device reads or stores may lie from Metal's: the rounding of an
/// interpolated depth, under 2^-19.
const TOLERANCE: f32 = 2e-6;

/// Whether `device` takes a viewport depth range past `[0, 1]` natively: it enabled
/// `VK_EXT_depth_range_unrestricted`, `depthClamp`, `depthClipEnable` and `depthClampControl`
/// (the runtime's `render_state::native_depth_range`).
pub(super) fn native_depth_range(device: &Device) -> bool {
    let enabled = device.enabled();
    enabled
        .extensions
        .contains(&VkExtension::ExtDepthRangeUnrestricted)
        && [
            VkFeature::DepthClamp,
            VkFeature::DepthClipEnable,
            VkFeature::DepthClampControl,
        ]
        .iter()
        .all(|feature| enabled.features.contains(feature))
}

/// How `set_viewport` refuses a depth range past `[0, 1]` on `device` (the runtime's
/// `render_state::depth_ranges_refused`), `None` where it takes one: natively, or by the vertex
/// stage's compensation, where it enabled `shaderClipDistance` and `depthClamp` and its driver
/// draws the compensated depths exactly. MoltenVK's do not (Metal's rasterizer, a probe:
/// `RuntimeDriverDefect::NormalizedDepthPastUnitRange`).
pub(super) fn depth_ranges_refused(device: &Device) -> Option<Error> {
    let features = &device.enabled().features;
    if native_depth_range(device) {
        None
    } else if !(features.contains(&VkFeature::ShaderClipDistance)
        && features.contains(&VkFeature::DepthClamp))
    {
        Some(Error::NotImplemented(RuntimeGap::ViewportDepthRange))
    } else {
        let defect = RuntimeDriverDefect::NormalizedDepthPastUnitRange;
        let defective = device.properties().driver_defects.contains(&defect);
        defective.then(|| {
            Error::Unsupported(Unsupported::runtime_driver_defect(
                Feature::ViewportDepthRange,
                defect,
            ))
        })
    }
}

/// Whether the vertex stage compensates clamp mode's one clamp of the range `[znear, zfar]`
/// (Metal's: to `[0, 1]`, then to the range, and at least 0): only a clamp inside `[0, 1]` of
/// two depths, which a Vulkan viewport's range can be. A one clamp of one depth collapses the
/// draw's Vulkan range to it instead, which a fragment function that reads its position's
/// depth, as both of this test's do, cannot take (`one_depth_clamps_store_their_depth`).
fn compensates_clamp(znear: f64, zfar: f64) -> bool {
    let (lo, hi) = (znear.min(zfar), znear.max(zfar));
    let [low, high] = if lo > 1.0 {
        [lo, lo]
    } else if hi < 0.0 {
        [0.0, 0.0]
    } else {
        [lo.max(0.0), hi.min(1.0)]
    };
    low < high
}

/// The fragment function of a case: `rfb_position`, which stores the position it reads and its
/// facing (Metal's facing is `front`), or `rfb_depth`, which writes this depth and stores the
/// position's depth in red.
#[derive(Clone, Copy, Debug)]
enum Fragment {
    Position { front: bool },
    Depth(f32),
}

/// One probe case and Metal's answer, the same in every row: the position's depth each column's
/// fragment read (`None` where nothing drew), and each column's stored depth.
struct Case {
    name: &'static str,
    /// The viewport's origin x, width, near and far depths; its origin y is 0 and its height 4.
    viewport: [f64; 4],
    /// The triangle's depth `[z, slope]`.
    depth: [f32; 2],
    mode: DepthClipMode,
    fragment: Fragment,
    /// The depth bias `[depth_bias, slope_scale, clamp]` set, if any.
    bias: Option<[f32; 3]>,
    cull: CullMode,
    seen: [Option<f32>; 4],
    stored: [f32; 4],
    /// The open near-equivalent (the module docs).
    near: bool,
}

/// The fixture's library on `device`.
fn library(device: &Device) -> metal2vulkan_runtime::Library {
    let ll = metal2vulkan_devtools::engine_dir().join("rfb_viewports.ll");
    let bitcode = llvm_assemble(&ll).unwrap();
    device.new_library_with_air(&bitcode).unwrap()
}

/// The probe's pipeline of `fragment` on `probe`'s device.
fn pipeline(probe: &Probe, fragment: &str) -> RenderPipelineState {
    pipeline_at(probe, fragment, 1)
}

/// The probe's pipeline of `fragment` on `probe`'s device, rasterizing `samples` samples.
fn pipeline_at(probe: &Probe, fragment: &str, samples: u32) -> RenderPipelineState {
    let descriptor = descriptor_at(probe, fragment, samples);
    probe
        .device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The descriptor of [`pipeline_at`]'s pipeline.
fn descriptor_at(probe: &Probe, fragment: &str, samples: u32) -> RenderPipelineDescriptor {
    let function = |name| {
        probe
            .library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(function("rfb_full")));
    descriptor.fragment = Some(StageSource::Function(function(fragment)));
    descriptor.color_attachments[0] =
        Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
    descriptor.depth_format = Some(PixelFormat::Depth32Float);
    descriptor.raster_sample_count = samples;
    descriptor
}

/// The kinds the makings of the probe's two pipelines at `samples` announce, and the `Shared`
/// buffers of its draws: what a test of [`Pipelines`] announces besides its own kinds.
fn made_at(probe: &Probe, samples: u32) -> BTreeSet<Degradation> {
    let mut first = forced(&[(
        Degradation::SharedInSystemMemory,
        shared_in_system_memory(&probe.device),
    )]);
    for fragment in ["rfb_position", "rfb_depth"] {
        let descriptor = descriptor_at(probe, fragment, samples);
        first.extend(made_render_kinds(&probe.device, &descriptor));
    }
    first
}

/// The probe's two pipelines: `rfb_position`'s and `rfb_depth`'s.
struct Pipelines {
    position: RenderPipelineState,
    depth: RenderPipelineState,
}

/// `case`'s state and draw on `encoder`, whose buffers `held` keeps until the command buffer
/// completed; the runtime's refusal of the state, if it refuses it.
fn encode(
    probe: &Probe,
    pipelines: &Pipelines,
    encoder: &mut RenderCommandEncoder,
    case: &Case,
    held: &mut Vec<Buffer>,
) -> Result<(), Error> {
    let [x, width, znear, zfar] = case.viewport;
    encoder.set_viewport(Viewport::new(x, 0.0, width, 4.0, znear, zfar))?;
    encoder.set_depth_clip_mode(case.mode)?;
    encoder.set_cull_mode(case.cull);
    if let Some([depth_bias, slope_scale, clamp]) = case.bias {
        encoder.set_depth_bias(depth_bias, slope_scale, clamp)?;
    }
    encoder.set_depth_stencil_state(&probe.state(CompareFunction::Always, true, None))?;
    let depth = probe.depth(case.depth[0], case.depth[1]);
    encoder.set_vertex_buffer(Some(&depth), 0, 0)?;
    held.push(depth);
    match case.fragment {
        Fragment::Position { .. } => encoder.set_render_pipeline_state(&pipelines.position)?,
        Fragment::Depth(written) => {
            encoder.set_render_pipeline_state(&pipelines.depth)?;
            let written = buffer_with(&probe.device, &written.to_le_bytes()).unwrap();
            encoder.set_fragment_buffer(Some(&written), 0, 0)?;
            held.push(written);
        }
    }
    encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
}

/// The `f32`s of `bytes`, little-endian.
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// `case` drawn in the probe's pass: the target's texels (four `f32`s each) and the depths
/// stored, row by row, or the runtime's refusal of its state.
fn draw(probe: &Probe, pipelines: &Pipelines, case: &Case) -> Result<(Vec<f32>, Vec<f32>), Error> {
    let color = texture(&probe.device, PixelFormat::RGBA32Float, 4);
    let depth = texture(&probe.device, PixelFormat::Depth32Float, 4);
    let mut descriptor = RenderPassDescriptor::new();
    let target = &mut descriptor.color_attachments[0];
    target.texture = Some(color.clone());
    target.load_action = LoadAction::Clear;
    target.clear_color = ClearColor::new(-9.0, -9.0, -9.0, -9.0);
    target.store_action = StoreAction::Store;
    descriptor.depth_attachment.texture = Some(depth.clone());
    descriptor.depth_attachment.load_action = LoadAction::Clear;
    descriptor.depth_attachment.clear_depth = 1.0;
    descriptor.depth_attachment.store_action = StoreAction::Store;
    let command_buffer = probe.queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    let mut held = Vec::new();
    let encoded = encode(probe, pipelines, &mut encoder, case, &mut held);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None, "{}", case.name);
    drop(held);
    encoded?;
    Ok((floats(&texels(&color, 16)), floats(&texels(&depth, 4))))
}

/// A `Private` 4² `pf` render target of `samples` samples that shaders read.
fn multisampled(device: &Device, pf: PixelFormat, samples: u64) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(pf, 4, 4, false);
    descriptor.texture_type = TextureType::Type2DMultisample;
    descriptor.sample_count = samples;
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// `case` drawn as [`draw`] draws it, into targets of `samples` samples: the color target's
/// texels resolved (four `f32`s each) and each sample's stored depth's bits, row by row, a
/// pixel's samples together, or the runtime's refusal of its state.
fn draw_samples(
    probe: &Probe,
    pipelines: &Pipelines,
    case: &Case,
    samples: u64,
) -> Result<(Vec<f32>, Vec<u32>), Error> {
    let color = multisampled(&probe.device, PixelFormat::RGBA32Float, samples);
    let resolved = texture(&probe.device, PixelFormat::RGBA32Float, 4);
    let depth = multisampled(&probe.device, PixelFormat::Depth32Float, samples);
    let mut descriptor = RenderPassDescriptor::new();
    let target = &mut descriptor.color_attachments[0];
    target.texture = Some(color.clone());
    target.resolve_texture = Some(resolved.clone());
    target.load_action = LoadAction::Clear;
    target.clear_color = ClearColor::new(-9.0, -9.0, -9.0, -9.0);
    target.store_action = StoreAction::MultisampleResolve;
    descriptor.depth_attachment.texture = Some(depth.clone());
    descriptor.depth_attachment.load_action = LoadAction::Clear;
    descriptor.depth_attachment.clear_depth = 1.0;
    descriptor.depth_attachment.store_action = StoreAction::Store;
    let command_buffer = probe.queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    let mut held = Vec::new();
    let encoded = encode(probe, pipelines, &mut encoder, case, &mut held);
    encoder.end_encoding().unwrap();
    let read = samples_of(
        &probe.device,
        &command_buffer,
        &depth,
        SampleRead::Depth,
        [4, 4],
        samples as u32,
    );
    let read = read.unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None, "{}", case.name);
    drop(held);
    encoded?;
    let words = read.bytes().unwrap();
    let words = words.as_chunks::<4>().0.iter();
    let depths = words.map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    Ok((floats(&texels(&resolved, 16)), depths.collect()))
}

/// What `case` stores in Metal: each texel's four `f32`s and each depth, row by row.
fn metal(case: &Case) -> (Vec<f32>, Vec<f32>) {
    let mut color = Vec::new();
    let mut depth = Vec::new();
    for row in 0..4 {
        for (column, seen) in case.seen.iter().enumerate() {
            let texel = match (*seen, case.fragment) {
                (None, _) => [-9.0; 4],
                (Some(z), Fragment::Position { front }) => [
                    column as f32 + 0.5,
                    row as f32 + 0.5,
                    z,
                    if front { 1.0 } else { -1.0 },
                ],
                (Some(z), Fragment::Depth(_)) => [z, 0.0, 0.0, 1.0],
            };
            color.extend(texel);
            depth.push(case.stored[column]);
        }
    }
    (color, depth)
}

/// Whether each of `a` lies within `TOLERANCE` of `b`'s.
fn near(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| (a - b).abs() <= TOLERANCE)
}

/// Each probe case draws as Metal drew it where the device takes its viewport, and is refused
/// typed where it does not (the module docs). Every case runs, and the ones that differ are
/// listed together at the end.
#[test]
fn viewports_past_vulkans_draw_as_metal() {
    use DepthClipMode::{Clamp, Clip};
    use Fragment::{Depth, Position};
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let probe = Probe {
        queue: device.new_command_queue().unwrap(),
        library: library(&device),
        device,
    };
    let pipelines = Pipelines {
        position: pipeline(&probe, "rfb_position"),
        depth: pipeline(&probe, "rfb_depth"),
    };
    let native = native_depth_range(&probe.device);
    let refused = depth_ranges_refused(&probe.device);
    println!(
        "{}: native depth ranges {native}, depth ranges refused {refused:?}",
        probe.device.name()
    );
    let mut differ = Vec::new();
    let cases = [
        Case {
            name: "vp_pos_clip_0_1_z075",
            viewport: [0.0, 4.0, 0.0, 1.0],
            depth: [0.75, 0.0],
            mode: Clip,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.75); 4],
            stored: [0.75; 4],
            near: false,
        },
        Case {
            name: "vp_pos_clip_05_2_z075",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.75, 0.0],
            mode: Clip,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [Some(1.625); 4],
            stored: [1.0; 4],
            near: false,
        },
        Case {
            name: "vp_pos_clip_05_2_z025",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.25, 0.0],
            mode: Clip,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.875); 4],
            stored: [0.875; 4],
            near: false,
        },
        Case {
            name: "vp_pos_clip_m1_1_z025",
            viewport: [0.0, 4.0, -1.0, 1.0],
            depth: [0.25, 0.0],
            mode: Clip,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [Some(-0.5); 4],
            stored: [0.0; 4],
            near: false,
        },
        Case {
            name: "vp_pos_clip_2_m1_z05",
            viewport: [0.0, 4.0, 2.0, -1.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.5); 4],
            stored: [0.5; 4],
            near: false,
        },
        Case {
            name: "vp_pos_clip_15_3_z05",
            viewport: [0.0, 4.0, 1.5, 3.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [Some(2.25); 4],
            stored: [1.0; 4],
            near: false,
        },
        Case {
            name: "vp_pos_clip_05_2_tilt",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.5, 0.25],
            mode: Clip,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.96875), Some(1.15625), Some(1.34375), Some(1.53125)],
            stored: [0.96875, 1.0, 1.0, 1.0],
            near: false,
        },
        Case {
            name: "vp_pos_clip_05_2_zm05",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [-0.5, 0.0],
            mode: Clip,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [None; 4],
            stored: [1.0; 4],
            near: false,
        },
        Case {
            name: "vp_pos_clip_05_2_z15",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [1.5, 0.0],
            mode: Clip,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [None; 4],
            stored: [1.0; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clip_0_1_d15",
            viewport: [0.0, 4.0, 0.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Depth(1.5),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.5); 4],
            stored: [1.0; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clip_0_1_dm05",
            viewport: [0.0, 4.0, 0.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Depth(-0.5),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.5); 4],
            stored: [0.0; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clip_0_1_d03",
            viewport: [0.0, 4.0, 0.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Depth(0.3),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.5); 4],
            stored: [0.3; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clip_0_1_d075",
            viewport: [0.0, 4.0, 0.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Depth(0.75),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.5); 4],
            stored: [0.75; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clip_05_2_d15",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Depth(1.5),
            bias: None,
            cull: CullMode::None,
            seen: [Some(1.25); 4],
            stored: [1.0; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clip_05_2_dm05",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Depth(-0.5),
            bias: None,
            cull: CullMode::None,
            seen: [Some(1.25); 4],
            stored: [0.0; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clip_05_2_d03",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Depth(0.3),
            bias: None,
            cull: CullMode::None,
            seen: [Some(1.25); 4],
            stored: [0.3; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clip_05_2_d075",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Depth(0.75),
            bias: None,
            cull: CullMode::None,
            seen: [Some(1.25); 4],
            stored: [0.75; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clip_025_05_d15",
            viewport: [0.0, 4.0, 0.25, 0.5],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Depth(1.5),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.375); 4],
            stored: [1.0; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clip_025_05_dm05",
            viewport: [0.0, 4.0, 0.25, 0.5],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Depth(-0.5),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.375); 4],
            stored: [0.0; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clip_025_05_d03",
            viewport: [0.0, 4.0, 0.25, 0.5],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Depth(0.3),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.375); 4],
            stored: [0.3; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clip_025_05_d075",
            viewport: [0.0, 4.0, 0.25, 0.5],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Depth(0.75),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.375); 4],
            stored: [0.75; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clip_m1_1_d15",
            viewport: [0.0, 4.0, -1.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Depth(1.5),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.0); 4],
            stored: [1.0; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clip_m1_1_dm05",
            viewport: [0.0, 4.0, -1.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Depth(-0.5),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.0); 4],
            stored: [0.0; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clip_m1_1_d03",
            viewport: [0.0, 4.0, -1.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Depth(0.3),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.0); 4],
            stored: [0.3; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clip_m1_1_d075",
            viewport: [0.0, 4.0, -1.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Depth(0.75),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.0); 4],
            stored: [0.75; 4],
            near: false,
        },
        Case {
            name: "vp_bias_clip_05_2_z0",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.0, 0.0],
            mode: Clip,
            fragment: Position { front: false },
            bias: Some([1048576.0, 0.0, 0.0]),
            cull: CullMode::None,
            seen: [Some(0.5625); 4],
            stored: [0.5625; 4],
            near: false,
        },
        Case {
            name: "vp_bias_clip_05_2_z025",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.25, 0.0],
            mode: Clip,
            fragment: Position { front: false },
            bias: Some([1048576.0, 0.0, 0.0]),
            cull: CullMode::None,
            seen: [Some(0.9375); 4],
            stored: [0.9375; 4],
            near: false,
        },
        Case {
            name: "vp_bias_clip_05_2_z025_neg",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.25, 0.0],
            mode: Clip,
            fragment: Position { front: false },
            bias: Some([-1048576.0, 0.0, 0.0]),
            cull: CullMode::None,
            seen: [Some(0.8125); 4],
            stored: [0.8125; 4],
            near: false,
        },
        Case {
            name: "vp_bias_clip_0_1_z05",
            viewport: [0.0, 4.0, 0.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Position { front: false },
            bias: Some([1048576.0, 0.0, 0.0]),
            cull: CullMode::None,
            seen: [Some(0.5625); 4],
            stored: [0.5625; 4],
            near: false,
        },
        Case {
            name: "vp_bias_clip_05_2_slope",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.25, 0.125],
            mode: Clip,
            fragment: Position { front: false },
            bias: Some([0.0, 1.0, 0.0]),
            cull: CullMode::None,
            seen: [
                Some(0.828125),
                Some(0.921875),
                Some(1.015625),
                Some(1.109375),
            ],
            stored: [0.828125, 0.921875, 1.0, 1.0],
            near: false,
        },
        Case {
            name: "vp_pos_clamp_0_1_z075",
            viewport: [0.0, 4.0, 0.0, 1.0],
            depth: [0.75, 0.0],
            mode: Clamp,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.75); 4],
            stored: [0.75; 4],
            near: false,
        },
        Case {
            name: "vp_pos_clamp_05_2_z075",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.75, 0.0],
            mode: Clamp,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [Some(1.625); 4],
            stored: [1.0; 4],
            near: false,
        },
        Case {
            name: "vp_pos_clamp_05_2_z025",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.25, 0.0],
            mode: Clamp,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.875); 4],
            stored: [0.875; 4],
            near: false,
        },
        Case {
            name: "vp_pos_clamp_m1_1_z025",
            viewport: [0.0, 4.0, -1.0, 1.0],
            depth: [0.25, 0.0],
            mode: Clamp,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [Some(-0.5); 4],
            stored: [0.0; 4],
            near: false,
        },
        Case {
            name: "vp_pos_clamp_2_m1_z05",
            viewport: [0.0, 4.0, 2.0, -1.0],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.5); 4],
            stored: [0.5; 4],
            near: false,
        },
        Case {
            name: "vp_pos_clamp_15_3_z05",
            viewport: [0.0, 4.0, 1.5, 3.0],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [Some(2.25); 4],
            stored: [1.5; 4],
            near: false,
        },
        Case {
            name: "vp_pos_clamp_05_2_tilt",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.5, 0.25],
            mode: Clamp,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.96875), Some(1.15625), Some(1.34375), Some(1.53125)],
            stored: [0.8875, 0.91249996, 0.9375, 0.9625],
            near: true,
        },
        Case {
            name: "vp_pos_clamp_05_2_zm05",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [-0.5, 0.0],
            mode: Clamp,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [Some(-0.25); 4],
            stored: [0.5; 4],
            near: false,
        },
        Case {
            name: "vp_pos_clamp_05_2_z15",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [1.5, 0.0],
            mode: Clamp,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [Some(2.75); 4],
            stored: [1.0; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clamp_0_1_d15",
            viewport: [0.0, 4.0, 0.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Depth(1.5),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.5); 4],
            stored: [1.0; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clamp_0_1_dm05",
            viewport: [0.0, 4.0, 0.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Depth(-0.5),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.5); 4],
            stored: [0.0; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clamp_0_1_d03",
            viewport: [0.0, 4.0, 0.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Depth(0.3),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.5); 4],
            stored: [0.3; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clamp_0_1_d075",
            viewport: [0.0, 4.0, 0.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Depth(0.75),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.5); 4],
            stored: [0.75; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clamp_05_2_d15",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Depth(1.5),
            bias: None,
            cull: CullMode::None,
            seen: [Some(1.25); 4],
            stored: [1.0; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clamp_05_2_dm05",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Depth(-0.5),
            bias: None,
            cull: CullMode::None,
            seen: [Some(1.25); 4],
            stored: [0.5; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clamp_05_2_d03",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Depth(0.3),
            bias: None,
            cull: CullMode::None,
            seen: [Some(1.25); 4],
            stored: [0.5; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clamp_05_2_d075",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Depth(0.75),
            bias: None,
            cull: CullMode::None,
            seen: [Some(1.25); 4],
            stored: [0.75; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clamp_025_05_d15",
            viewport: [0.0, 4.0, 0.25, 0.5],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Depth(1.5),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.375); 4],
            stored: [0.5; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clamp_025_05_dm05",
            viewport: [0.0, 4.0, 0.25, 0.5],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Depth(-0.5),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.375); 4],
            stored: [0.25; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clamp_025_05_d03",
            viewport: [0.0, 4.0, 0.25, 0.5],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Depth(0.3),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.375); 4],
            stored: [0.3; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clamp_025_05_d075",
            viewport: [0.0, 4.0, 0.25, 0.5],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Depth(0.75),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.375); 4],
            stored: [0.5; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clamp_m1_1_d15",
            viewport: [0.0, 4.0, -1.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Depth(1.5),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.0); 4],
            stored: [1.0; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clamp_m1_1_dm05",
            viewport: [0.0, 4.0, -1.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Depth(-0.5),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.0); 4],
            stored: [0.0; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clamp_m1_1_d03",
            viewport: [0.0, 4.0, -1.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Depth(0.3),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.0); 4],
            stored: [0.3; 4],
            near: false,
        },
        Case {
            name: "vp_fd_clamp_m1_1_d075",
            viewport: [0.0, 4.0, -1.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Depth(0.75),
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.0); 4],
            stored: [0.75; 4],
            near: false,
        },
        Case {
            name: "vp_bias_clamp_05_2_z0",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.0, 0.0],
            mode: Clamp,
            fragment: Position { front: false },
            bias: Some([1048576.0, 0.0, 0.0]),
            cull: CullMode::None,
            seen: [Some(0.5625); 4],
            stored: [0.5625; 4],
            near: false,
        },
        Case {
            name: "vp_bias_clamp_05_2_z025",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.25, 0.0],
            mode: Clamp,
            fragment: Position { front: false },
            bias: Some([1048576.0, 0.0, 0.0]),
            cull: CullMode::None,
            seen: [Some(0.9375); 4],
            stored: [0.9375; 4],
            near: false,
        },
        Case {
            name: "vp_bias_clamp_05_2_z025_neg",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.25, 0.0],
            mode: Clamp,
            fragment: Position { front: false },
            bias: Some([-1048576.0, 0.0, 0.0]),
            cull: CullMode::None,
            seen: [Some(0.8125); 4],
            stored: [0.8125; 4],
            near: false,
        },
        Case {
            name: "vp_bias_clamp_0_1_z05",
            viewport: [0.0, 4.0, 0.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clamp,
            fragment: Position { front: false },
            bias: Some([1048576.0, 0.0, 0.0]),
            cull: CullMode::None,
            seen: [Some(0.5625); 4],
            stored: [0.5625; 4],
            near: false,
        },
        Case {
            name: "vp_bias_clamp_05_2_slope",
            viewport: [0.0, 4.0, 0.5, 2.0],
            depth: [0.25, 0.125],
            mode: Clamp,
            fragment: Position { front: false },
            bias: Some([0.0, 1.0, 0.0]),
            cull: CullMode::None,
            seen: [
                Some(0.828125),
                Some(0.921875),
                Some(1.015625),
                Some(1.109375),
            ],
            stored: [0.78125, 0.84375, 0.90625, 0.96875],
            near: true,
        },
        Case {
            name: "negw_full",
            viewport: [4.0, -4.0, 0.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Position { front: true },
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.5); 4],
            stored: [0.5; 4],
            near: false,
        },
        Case {
            name: "negw_half",
            viewport: [2.0, -2.0, 0.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Position { front: true },
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.5), Some(0.5), None, None],
            stored: [0.5, 0.5, 1.0, 1.0],
            near: false,
        },
        Case {
            name: "negw_frac",
            viewport: [3.5, -3.0, 0.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Position { front: true },
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.5), Some(0.5), Some(0.5), None],
            stored: [0.5, 0.5, 0.5, 1.0],
            near: false,
        },
        Case {
            name: "negw_full_cull_back",
            viewport: [4.0, -4.0, 0.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Position { front: true },
            bias: None,
            cull: CullMode::Back,
            seen: [Some(0.5); 4],
            stored: [0.5; 4],
            near: false,
        },
        Case {
            name: "negw_full_cull_front",
            viewport: [4.0, -4.0, 0.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::Front,
            seen: [None; 4],
            stored: [1.0; 4],
            near: false,
        },
        Case {
            name: "posw_full",
            viewport: [0.0, 4.0, 0.0, 1.0],
            depth: [0.5, 0.0],
            mode: Clip,
            fragment: Position { front: false },
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.5); 4],
            stored: [0.5; 4],
            near: false,
        },
        Case {
            name: "negw_tilt",
            viewport: [4.0, -4.0, 0.0, 1.0],
            depth: [0.5, 0.25],
            mode: Clip,
            fragment: Position { front: true },
            bias: None,
            cull: CullMode::None,
            seen: [Some(0.6875), Some(0.5625), Some(0.4375), Some(0.3125)],
            stored: [0.6875, 0.5625, 0.4375, 0.3125],
            near: false,
        },
        Case {
            name: "negw_range_05_2",
            viewport: [4.0, -4.0, 0.5, 2.0],
            depth: [0.75, 0.0],
            mode: Clip,
            fragment: Position { front: true },
            bias: None,
            cull: CullMode::None,
            seen: [Some(1.625); 4],
            stored: [1.0; 4],
            near: false,
        },
    ];
    for case in &cases {
        let [_, _, znear, zfar] = case.viewport;
        let outside = |z: f64| !(0.0..=1.0).contains(&z);
        let compensated = !native && (outside(znear) || outside(zfar));
        let clamps = matches!(case.mode, Clamp);
        let refusal = match compensated {
            false => None,
            true => refused.clone().or_else(|| {
                (clamps && !compensates_clamp(znear, zfar))
                    .then_some(Error::NotImplemented(RuntimeGap::ViewportDepthRange))
            }),
        };
        let drawn = draw(&probe, &pipelines, case);
        let (color, depth) = match (refusal, drawn) {
            (Some(error), drawn) => {
                if drawn.as_ref().err() != Some(&error) {
                    differ.push(format!("{}: {drawn:?}, not refused {error:?}", case.name));
                }
                continue;
            }
            (None, Err(error)) => {
                differ.push(format!("{}: refused {error:?}", case.name));
                continue;
            }
            (None, Ok(drawn)) => drawn,
        };
        let (metal_color, metal_depth) = metal(case);
        println!(
            "{}: read {:?} stored {:?}; Metal read {:?} stored {:?}",
            case.name,
            color.chunks(4).take(4).collect::<Vec<_>>(),
            &depth[..4],
            case.seen,
            case.stored
        );
        if case.near {
            // Metal's rule is unidentified (the module docs): the read depths are Metal's, and
            // the stored ones lie in the clamp, `[0.5, 1]` for these cases' range of 0.5 to 2.
            let [low, high] = [znear.min(zfar).max(0.0), znear.max(zfar).min(1.0)];
            let clamped = depth
                .iter()
                .all(|&z| (low as f32..=high as f32).contains(&z));
            if !near(&color, &metal_color) || !clamped {
                differ.push(format!(
                    "{}: read {color:?} stored {depth:?}, outside the near-equivalent",
                    case.name
                ));
            }
            continue;
        }
        if !near(&color, &metal_color) || !near(&depth, &metal_depth) {
            differ.push(format!(
                "{}: read {color:?} stored {depth:?}, Metal read {metal_color:?} stored \
                 {metal_depth:?}",
                case.name
            ));
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
    quiet(probe, &messages);
}

/// One written depth of probe 3 and the bits Metal stored for it through each of
/// `SPECIAL_VIEWPORTS`, or `None` where the fragment failed the depth test, which writes neither
/// depth nor color (probe 3b).
struct Special {
    name: &'static str,
    written: u32,
    stored: [Option<u32>; 4],
}

/// Probe 3's written depths and what Metal stored for each through `SPECIAL_VIEWPORTS`.
fn specials() -> [Special; 15] {
    let (zero, quarter, half, one) = (0, 0x3e80_0000, 0x3f00_0000, 0x3f80_0000);
    let unwritten = [None; 4];
    [
        Special {
            name: "qnan",
            written: 0x7fc0_0000,
            stored: unwritten,
        },
        Special {
            name: "nan_payload",
            written: 0x7fc0_1234,
            stored: unwritten,
        },
        Special {
            name: "snan",
            written: 0x7fa0_0001,
            stored: unwritten,
        },
        Special {
            name: "neg_qnan",
            written: 0xffc0_0000,
            stored: [Some(zero); 4],
        },
        Special {
            name: "pos_zero",
            written: 0,
            stored: [Some(zero), Some(zero), Some(zero), Some(quarter)],
        },
        Special {
            name: "neg_zero",
            written: 0x8000_0000,
            stored: [Some(zero), Some(zero), Some(zero), Some(quarter)],
        },
        Special {
            name: "pos_inf",
            written: 0x7f80_0000,
            stored: [Some(one), Some(one), Some(one), Some(half)],
        },
        Special {
            name: "neg_inf",
            written: 0xff80_0000,
            stored: [Some(zero), Some(zero), Some(zero), Some(quarter)],
        },
        Special {
            name: "denorm_min",
            written: 1,
            stored: [Some(1), Some(1), Some(1), Some(quarter)],
        },
        Special {
            name: "denorm_max",
            written: 0x007f_ffff,
            stored: [
                Some(0x007f_ffff),
                Some(0x007f_ffff),
                Some(0x007f_ffff),
                Some(quarter),
            ],
        },
        Special {
            name: "neg_denorm",
            written: 0x8000_0001,
            stored: [Some(zero), Some(zero), Some(zero), Some(quarter)],
        },
        Special {
            name: "one_up",
            written: 0x3f80_0001,
            stored: [Some(one), Some(one), Some(one), Some(half)],
        },
        Special {
            name: "one_and_half",
            written: 0x3fc0_0000,
            stored: [Some(one), Some(one), Some(one), Some(half)],
        },
        Special {
            name: "neg_half",
            written: 0xbf00_0000,
            stored: [Some(zero), Some(zero), Some(zero), Some(quarter)],
        },
        Special {
            name: "third",
            written: 0x3eaa_aaab,
            stored: [Some(0x3eaa_aaab); 4],
        },
    ]
}

/// Probe 3's viewports: their near and far depths, and their clip mode.
const SPECIAL_VIEWPORTS: [(&str, [f64; 2], DepthClipMode); 4] = [
    ("clip_0_1", [0.0, 1.0], DepthClipMode::Clip),
    ("clip_025_05", [0.25, 0.5], DepthClipMode::Clip),
    ("clamp_0_1", [0.0, 1.0], DepthClipMode::Clamp),
    ("clamp_025_05", [0.25, 0.5], DepthClipMode::Clamp),
];

/// Whether probe 3's `written` through viewport `viewport` (an index of `SPECIAL_VIEWPORTS`) is a
/// negative NaN through the clamp above 0, which the fragment function clamps itself where the
/// draw's Vulkan clamp can hold 0, and otherwise announces `nan-depth`.
fn negative_nan_clamped(written: u32, viewport: usize) -> bool {
    f32::from_bits(written).is_nan() && written & 0x8000_0000 != 0 && viewport == 3
}

/// The kinds `viewport_ranges` takes: a draw whose fragment function writes its depth, in depth
/// clamp mode through a range above 0, that keeps the device's clamp of that depth.
pub(super) const NAN_DEPTH: [Degradation; 1] = [Degradation::NanDepth];

/// The kind `biased_depths_past_unit_range_store_metals_clamp` takes: a draw of triangles in depth
/// clip mode under a bias that can take its depth past `[0, 1]`, on a device that stores it
/// unclamped and cannot clamp it.
pub(super) const UNCLAMPED_BIAS: [Degradation; 1] = [Degradation::BiasedDepthPastUnitRange];

/// Whether a draw on `device` whose fragment function writes its depth, in depth clamp mode
/// through a range above 0 inside `[0, 1]`, clamps it itself (the runtime's
/// `render_state::written_depth_clamp`): natively, or through the vertex stage's compensation
/// where the device takes depth ranges past `[0, 1]` by it.
fn clamps_written_depth(device: &Device) -> bool {
    depth_ranges_refused(device).is_none()
}

/// Each depth probe 3 wrote stores Metal's bits through each of its viewports, or nothing where
/// Metal's depth test failed, except the open row, which prints (the module docs).
#[test]
fn written_depths_store_metals_bits() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let probe = Probe {
        queue: device.new_command_queue().unwrap(),
        library: library(&device),
        device,
    };
    let pipelines = Pipelines {
        position: pipeline(&probe, "rfb_position"),
        depth: pipeline(&probe, "rfb_depth"),
    };
    let specials = specials();
    let mut differ = Vec::new();
    for special in &specials {
        for (v, (viewport, [znear, zfar], mode)) in SPECIAL_VIEWPORTS.iter().enumerate() {
            // Nothing is written where the depth test failed: the clear values stay.
            let seen = special.stored[v].map(|_| (znear + (zfar - znear) * 0.5) as f32);
            let stored = special.stored[v].unwrap_or(1.0f32.to_bits());
            let case = Case {
                name: special.name,
                viewport: [0.0, 4.0, *znear, *zfar],
                depth: [0.5, 0.0],
                mode: *mode,
                fragment: Fragment::Depth(f32::from_bits(special.written)),
                bias: None,
                cull: CullMode::None,
                seen: [seen; 4],
                stored: [f32::from_bits(stored); 4],
                near: false,
            };
            let (color, depth) = match draw(&probe, &pipelines, &case) {
                Ok(drawn) => drawn,
                Err(error) => {
                    differ.push(format!("{} {viewport}: refused {error:?}", special.name));
                    continue;
                }
            };
            let bits: Vec<u32> = depth.iter().map(|d| d.to_bits()).collect();
            let answer = special.stored[v].map_or("none (the depth test fails)".to_owned(), |b| {
                format!("{b:#010x}")
            });
            println!(
                "fd3 {} {viewport}: wrote {:#010x}, stored {:#010x}; Metal stored {answer}",
                special.name, special.written, bits[0]
            );
            if negative_nan_clamped(special.written, v) && !clamps_written_depth(&probe.device) {
                continue;
            }
            let (metal_color, _) = metal(&case);
            // A device's depth storage may flush a positive denormal to +0 (RADV, NVIDIA and
            // ANV do, x86 lavapipe through a depth clamp; the near-equivalent): Metal keeps it.
            let flushed = |b: u32| stored != 0 && stored < 0x0080_0000 && b == 0;
            if !near(&color, &metal_color) || bits.iter().any(|&b| b != stored && !flushed(b)) {
                differ.push(format!(
                    "{} {viewport}: read {:?} stored {bits:#010x?}, Metal stored {stored:#010x}",
                    special.name,
                    &color[..4]
                ));
            }
        }
    }
    let expected: &[Degradation] = match clamps_written_depth(&probe.device) {
        true => &[],
        false => &NAN_DEPTH,
    };
    let announced = assert_made_once(&messages, expected, &made_at(&probe, 1));
    if clamps_written_depth(&probe.device) {
        assert!(
            !announced.contains_key(&Degradation::NanDepth),
            "nan-depth announced where the draw clamps its depth itself"
        );
    }
    assert!(differ.is_empty(), "{differ:#?}");
    quiet(probe, &messages);
}

/// Each depth probe 3 wrote, written in a pass of 4 samples, stores Metal's bits in every sample
/// through each of its viewports, or nothing where Metal's depth test failed: Metal's answers at
/// 4 samples are its answers at 1 (an M4 under macOS 27.0.1: every sample of a 4x `Depth32Float`
/// target alike). The color target is resolved and compared as [`written_depths_store_metals_bits`]
/// compares it; its rules hold sample by sample.
#[test]
fn written_depths_store_metals_bits_in_every_sample() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let probe = Probe {
        queue: device.new_command_queue().unwrap(),
        library: library(&device),
        device,
    };
    let pipelines = Pipelines {
        position: pipeline_at(&probe, "rfb_position", 4),
        depth: pipeline_at(&probe, "rfb_depth", 4),
    };
    let mut differ = Vec::new();
    for special in &specials() {
        for (v, (viewport, [znear, zfar], mode)) in SPECIAL_VIEWPORTS.iter().enumerate() {
            let seen = special.stored[v].map(|_| (znear + (zfar - znear) * 0.5) as f32);
            let stored = special.stored[v].unwrap_or(1.0f32.to_bits());
            let case = Case {
                name: special.name,
                viewport: [0.0, 4.0, *znear, *zfar],
                depth: [0.5, 0.0],
                mode: *mode,
                fragment: Fragment::Depth(f32::from_bits(special.written)),
                bias: None,
                cull: CullMode::None,
                seen: [seen; 4],
                stored: [f32::from_bits(stored); 4],
                near: false,
            };
            let (color, bits) = match draw_samples(&probe, &pipelines, &case, 4) {
                Ok(drawn) => drawn,
                Err(error) => {
                    differ.push(format!("{} {viewport}: refused {error:?}", special.name));
                    continue;
                }
            };
            let answer = special.stored[v].map_or("none (the depth test fails)".to_owned(), |b| {
                format!("{b:#010x}")
            });
            let mut distinct: Vec<u32> = Vec::new();
            for &b in &bits {
                if !distinct.contains(&b) {
                    distinct.push(b);
                }
            }
            println!(
                "fd4 {} {viewport}: wrote {:#010x}, stored {distinct:08x?} in {} samples; Metal \
                 stored {answer}",
                special.name,
                special.written,
                bits.len()
            );
            if negative_nan_clamped(special.written, v) && !clamps_written_depth(&probe.device) {
                continue;
            }
            let (metal_color, _) = metal(&case);
            // As at 1 sample: a device's depth storage may flush a positive denormal to +0.
            let flushed = |b: u32| stored != 0 && stored < 0x0080_0000 && b == 0;
            let every = bits.len() == 16 * 4;
            if !every
                || !near(&color, &metal_color)
                || bits.iter().any(|&b| b != stored && !flushed(b))
            {
                differ.push(format!(
                    "{} {viewport}: read {:?} stored {distinct:08x?}, Metal stored {stored:#010x}",
                    special.name,
                    &color[..4]
                ));
            }
        }
    }
    let expected: &[Degradation] = match clamps_written_depth(&probe.device) {
        true => &[],
        false => &NAN_DEPTH,
    };
    // The passes resolve their 4x color, stored `MultisampleResolve`, which a driver not
    // measured to store `DONT_CARE` as Metal's may store otherwise (`dont-care-store`); the
    // depths are read by the samples' kernel.
    let mut first = made_at(&probe, 4);
    let dont_care_metals = probe
        .device
        .properties()
        .driver_facts
        .contains(&DriverFact::DontCareStoreIsMetals);
    first.extend(resolve_kind(&probe.device, 4));
    first.extend(forced(&[(Degradation::DontCareStore, !dont_care_metals)]));
    let read = read_function(&probe.device, PixelFormat::Depth32Float, SampleRead::Depth).unwrap();
    first.extend(made_compute_kinds(&probe.device, read));
    let announced = assert_made_once(&messages, expected, &first);
    if clamps_written_depth(&probe.device) {
        assert!(
            !announced.contains_key(&Degradation::NanDepth),
            "nan-depth announced where the draw clamps its depth itself"
        );
    }
    assert!(differ.is_empty(), "{differ:#?}");
    quiet(probe, &messages);
}

/// A positive NaN depth the fragment function writes fails Metal's depth test, `Always`
/// included, and runs the stencil's depth-failure operation (Metal probes, an M3 Max and an M4:
/// neither depth nor color written); the runtime discards the fragment, which runs no stencil
/// operation: the stencil keeps its clear where Metal's `Invert` stores 0xff, and nothing else is
/// written on either. Announced once as `nan-depth`, naming the depth-failure operation. A depth of
/// 0.5 passes, stores 0.5, keeps the stencil and draws, as Metal does.
#[test]
fn a_positive_nan_depth_runs_no_stencil_depth_failure_operation() {
    use super::render_state::{Pass, PASS, WHITE};
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let probe = Probe {
        queue: device.new_command_queue().unwrap(),
        library: library(&device),
        device,
    };
    let name = probe.device.name().to_owned();
    let pipe = WHITE
        .vertex("rfb_full")
        .fragment("rfb_depth")
        .depth(PixelFormat::Depth32Float)
        .stencil();
    let mut face = StencilDescriptor::default();
    face.depth_failure_operation = StencilOperation::Invert;
    let pass: Pass = PASS.depth(PixelFormat::Depth32Float, 1.0).stencil(0);
    let run = |written: f32| {
        probe.run(pass, |p, e, h| {
            e.set_depth_stencil_state(&p.state(CompareFunction::Always, true, Some(face)))
                .unwrap();
            e.set_render_pipeline_state(&p.pipeline(pipe)).unwrap();
            let depth = p.depth(0.5, 0.0);
            e.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
            let written = buffer_with(&p.device, &written.to_le_bytes()).unwrap();
            e.set_fragment_buffer(Some(&written), 0, 0).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            h.extend([depth, written]);
        })
    };
    let passed = run(0.5);
    let depths = floats(passed.depth.as_deref().unwrap());
    assert_eq!(depths, vec![0.5; 16], "0.5 on {name}: the depth");
    assert_eq!(
        passed.stencil,
        Some(vec![0; 16]),
        "0.5 on {name}: the stencil"
    );
    assert!(
        passed.color.iter().any(|&b| b != 0),
        "0.5 on {name}: the draw"
    );
    let failed = run(f32::from_bits(0x7fc0_0000));
    let depths = floats(failed.depth.as_deref().unwrap());
    assert_eq!(depths, vec![1.0; 16], "NaN on {name}: the depth kept");
    assert_eq!(failed.color, vec![0; 64], "NaN on {name}: nothing drawn");
    // Metal's `Invert` stores 0xff; the discard keeps the clear (the near-equivalent).
    assert_eq!(
        failed.stencil,
        Some(vec![0; 16]),
        "NaN on {name}: the stencil, the near-equivalent's"
    );
    // The pass's separate depth and stencil textures render through a combined scratch; the
    // draws read `Shared` buffers.
    let mut first = forced(&[
        (Degradation::SeparateDepthStencil, true),
        (
            Degradation::SharedInSystemMemory,
            shared_in_system_memory(&probe.device),
        ),
    ]);
    first.extend(probe.made(pipe));
    let pending = messages.take_notices();
    messages.restore_notices(pending.clone());
    assert_made_once(&messages, &NAN_DEPTH, &first);
    let notices = pending;
    let reason = notices
        .iter()
        .filter_map(|message| message.degradation.as_ref())
        .find(|notice| notice.degradation == Degradation::NanDepth)
        .map(|notice| notice.reason.clone())
        .unwrap();
    assert!(
        reason.contains("depth-failure operation"),
        "the notice on {name}: {reason}"
    );
    quiet(probe, &messages);
}

/// A saturated or offset depth bias through a depth range past `[0, 1]` stores Metal's depths (a
/// Metal probe on an M4 with macOS 27.0.1, plain and validated): `rfb_full` without a fragment
/// function, its depth `0.5 + 0.5 x` (`z` 0.125, 0.375, 0.625 and 0.875 at the columns), the
/// probe's ramp. In clip mode a saturated depth is 1 or 0 and an offset adds to the depth before
/// the clamp to `[0, 1]`; in clamp mode a saturated depth is the range's larger depth, past 1 too,
/// or its smaller one but at least 0. A compensated saturated depth past the one clamp is refused
/// typed where the device holds no Vulkan range past 1 (Taken under
/// `VK_EXT_depth_range_unrestricted`). Offsets through a range with a negative depth or in clamp
/// mode are [`offsets_through_clamped_vertices_store_metals_depths`]', whose quads stay inside
/// Metal's planes. Not run where the device takes no such range (`depth_ranges_refused`).
#[test]
fn saturated_and_offset_biases_store_metals_depths() {
    use DepthClipMode::{Clamp, Clip};
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    if let Some(refused) = depth_ranges_refused(&device) {
        println!("{}: depth ranges refused, {refused:?}", device.name());
        return;
    }
    let probe = Probe {
        queue: device.new_command_queue().unwrap(),
        library: library(&device),
        device,
    };
    let native = native_depth_range(&probe.device);
    let function = |name| {
        probe
            .library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(function("rfb_full")));
    descriptor.depth_format = Some(PixelFormat::Depth32Float);
    let pipeline = probe
        .device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    let gap = Err(Error::NotImplemented(RuntimeGap::NonFiniteDepthBias));
    // Past the one clamp: natively, or through a Vulkan range past 1.
    let unrestricted = probe
        .device
        .enabled()
        .extensions
        .contains(&VkExtension::ExtDepthRangeUnrestricted);
    let past_clamp = match native || unrestricted {
        true => Ok([1.5; 4]),
        false => gap.clone(),
    };
    type BiasCase = (
        &'static str,
        [f64; 2],
        DepthClipMode,
        [f32; 3],
        Result<[f32; 4], Error>,
    );
    let cases: [BiasCase; 12] = [
        (
            "clip_05_15_inf",
            [0.5, 1.5],
            Clip,
            [inf, 0.0, 0.0],
            Ok([1.0; 4]),
        ),
        (
            "clip_05_15_neginf",
            [0.5, 1.5],
            Clip,
            [-inf, 0.0, 0.0],
            Ok([0.0; 4]),
        ),
        (
            "clip_05_15_nan",
            [0.5, 1.5],
            Clip,
            [nan, 0.0, 0.0],
            Ok([1.0; 4]),
        ),
        (
            "clip_05_15_inf_clamp_quarter",
            [0.5, 1.5],
            Clip,
            [inf, 0.0, 0.25],
            Ok([0.875, 1.0, 1.0, 1.0]),
        ),
        (
            "clip_05_15_neginf_clamp_negquarter",
            [0.5, 1.5],
            Clip,
            [-inf, 0.0, -0.25],
            Ok([0.375, 0.625, 0.875, 1.0]),
        ),
        (
            "clip_m05_15_inf",
            [-0.5, 1.5],
            Clip,
            [inf, 0.0, 0.0],
            Ok([1.0; 4]),
        ),
        (
            "clip_m05_15_neginf",
            [-0.5, 1.5],
            Clip,
            [-inf, 0.0, 0.0],
            Ok([0.0; 4]),
        ),
        (
            "clamp_m05_05_inf",
            [-0.5, 0.5],
            Clamp,
            [inf, 0.0, 0.0],
            Ok([0.5; 4]),
        ),
        (
            "clamp_m05_15_neginf",
            [-0.5, 1.5],
            Clamp,
            [-inf, 0.0, 0.0],
            Ok([0.0; 4]),
        ),
        (
            "clamp_05_15_neginf",
            [0.5, 1.5],
            Clamp,
            [-inf, 0.0, 0.0],
            Ok([0.5; 4]),
        ),
        (
            "clamp_05_15_nan",
            [0.5, 1.5],
            Clamp,
            [nan, 0.0, 0.0],
            past_clamp.clone(),
        ),
        (
            "clamp_m05_15_inf",
            [-0.5, 1.5],
            Clamp,
            [inf, 0.0, 0.0],
            past_clamp,
        ),
    ];
    let mut differ = Vec::new();
    for (name, [znear, zfar], mode, [constant, slope, clamp], expected) in cases {
        let depth = texture(&probe.device, PixelFormat::Depth32Float, 4);
        let mut pass = RenderPassDescriptor::new();
        pass.depth_attachment.texture = Some(depth.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.clear_depth = 0.5;
        pass.depth_attachment.store_action = StoreAction::Store;
        let command_buffer = probe.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        let ramp = probe.depth(0.5, 0.5);
        let drawn = (|| {
            encoder.set_render_pipeline_state(&pipeline)?;
            encoder.set_depth_stencil_state(&probe.state(CompareFunction::Always, true, None))?;
            encoder.set_viewport(Viewport::new(0.0, 0.0, 4.0, 4.0, znear, zfar))?;
            encoder.set_depth_clip_mode(mode)?;
            encoder.set_depth_bias(constant, slope, clamp)?;
            encoder.set_vertex_buffer(Some(&ramp), 0, 0)?;
            encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        })();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{name}");
        let stored = floats(&texels(&depth, 4));
        let rows_alike = stored.chunks(4).all(|row| row == &stored[..4]);
        let got = drawn.map(|()| [0, 1, 2, 3].map(|column| stored[column]));
        let matches = match (&got, &expected) {
            (Ok(got), Ok(metal)) => near(got, metal),
            (got, metal) => got == metal,
        };
        if !matches || !rows_alike {
            differ.push(format!("{name}: {got:?}, Metal {expected:?}"));
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
    quiet(probe, &messages);
}

/// In depth clamp mode a viewport depth range whose one clamp is one depth stores that depth alone
/// (the probe's 1.5 of 1.5 to 3; a Metal probe on the M3 Max, plain and validated, each case below:
/// 0 for a range below 0): `rfb_full` without a fragment function, flat at 0.5 and tilted from 0.25
/// to 0.75 across the columns, unbiased and under `set_depth_bias(1e9, 0, 0.75)`. Natively every
/// case stores it; elsewhere the draw's Vulkan range collapses to it, a depth past `[0, 1]` only
/// under `VK_EXT_depth_range_unrestricted` (lavapipe), refused `NotImplemented(ViewportDepthRange)`
/// otherwise. A fragment function that reads its position's depth (`rfb_position`) would read the
/// one depth there: refused but natively. A bias's offset moves Metal's depth off the one depth
/// where the range's clamp to itself is not one depth (1.75 of 1.5 to 3 and of 1 to 2): stored
/// where the vertex stage clamps each vertex's depth and the device holds a Vulkan range past 1,
/// for a function that does not read its position's depth; refused elsewhere. Not run where the
/// device takes no such range (`depth_ranges_refused`).
#[test]
fn one_depth_clamps_store_their_depth() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    if let Some(refused) = depth_ranges_refused(&device) {
        println!("{}: depth ranges refused, {refused:?}", device.name());
        return;
    }
    let probe = Probe {
        queue: device.new_command_queue().unwrap(),
        library: library(&device),
        device,
    };
    let native = native_depth_range(&probe.device);
    let unrestricted = probe
        .device
        .enabled()
        .extensions
        .contains(&VkExtension::ExtDepthRangeUnrestricted);
    let function = |name| {
        probe
            .library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(function("rfb_full")));
    descriptor.depth_format = Some(PixelFormat::Depth32Float);
    let bare = probe
        .device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let reading = pipeline(&probe, "rfb_position");
    let refused = Err(Error::NotImplemented(RuntimeGap::ViewportDepthRange));
    let takes = probe
        .device
        .enabled()
        .features
        .contains(&VkFeature::ShaderClipDistance);
    let mut differ = Vec::new();
    // Each range, its one depth, and whether a bias moves Metal's depth off it.
    for ([znear, zfar], depth, moved) in [
        ([1.5, 3.0], 1.5, true),
        ([3.0, 1.5], 1.5, true),
        ([1.0, 2.0], 1.0, true),
        ([-1.0, 0.0], 0.0, false),
        ([0.0, -1.0], 0.0, false),
        ([-2.0, -1.0], 0.0, false),
        ([-1.0, -2.0], 0.0, false),
        ([2.0, 2.0], 2.0, false),
    ] {
        let held = native || unrestricted || depth <= 1.0;
        let cases = [
            ([0.5f32, 0.0], false, false),
            ([0.5, 0.25], false, false),
            ([0.5, 0.0], true, false),
            ([0.5, 0.25], false, true),
        ];
        for (ramp, reads, biased) in cases {
            let name = format!("{znear} {zfar} ramp {ramp:?} reads {reads} biased {biased}");
            // Moved: each vertex's depth clamped to 1, the offset of 0.75 added, within the range.
            let widened = takes && unrestricted && !reads;
            let drawn = match biased && moved {
                true => widened,
                false => held && (native || !reads),
            };
            let stored = match biased && moved {
                true => 1.75f32.clamp(f64::min(znear, zfar) as f32, f64::max(znear, zfar) as f32),
                false => depth as f32,
            };
            let expected = match drawn {
                true => Ok([stored; 4]),
                false => refused.clone(),
            };
            let target = texture(&probe.device, PixelFormat::Depth32Float, 4);
            let color = texture(&probe.device, PixelFormat::RGBA32Float, 4);
            let mut pass = RenderPassDescriptor::new();
            pass.depth_attachment.texture = Some(target.clone());
            pass.depth_attachment.load_action = LoadAction::Clear;
            pass.depth_attachment.clear_depth = 0.125;
            pass.depth_attachment.store_action = StoreAction::Store;
            if reads {
                let attachment = &mut pass.color_attachments[0];
                attachment.texture = Some(color.clone());
                attachment.load_action = LoadAction::Clear;
                attachment.clear_color = ClearColor::new(-9.0, -9.0, -9.0, -9.0);
                attachment.store_action = StoreAction::Store;
            }
            let command_buffer = probe.queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            let buffer = probe.depth(ramp[0], ramp[1]);
            let encoded = (|| {
                encoder.set_render_pipeline_state(if reads { &reading } else { &bare })?;
                encoder.set_depth_stencil_state(&probe.state(
                    CompareFunction::Always,
                    true,
                    None,
                ))?;
                encoder.set_viewport(Viewport::new(0.0, 0.0, 4.0, 4.0, znear, zfar))?;
                encoder.set_depth_clip_mode(DepthClipMode::Clamp)?;
                if biased {
                    encoder.set_depth_bias(1e9, 0.0, 0.75)?;
                }
                encoder.set_vertex_buffer(Some(&buffer), 0, 0)?;
                encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            })();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None, "{name}");
            let stored = floats(&texels(&target, 4));
            let rows_alike = stored.chunks(4).all(|row| row == &stored[..4]);
            let got = encoded.map(|()| [0, 1, 2, 3].map(|column| stored[column]));
            let matches = match (&got, &expected) {
                (Ok(got), Ok(metal)) => near(got, metal),
                (got, metal) => got == metal,
            };
            if !matches || !rows_alike {
                differ.push(format!("{name}: {got:?}, Metal {expected:?}"));
            }
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
    quiet(probe, &messages);
}

/// The kinds `vertex_clamps_store_metals_depths` takes: a draw whose vertex stage clamps each
/// vertex's depth (`clipped-depth-pieces`, for the primitives that cross Metal's planes or guard
/// band), a draw whose fragment function reads its position's depth, which keeps a clamp of each
/// sample (`sample-depth-clamp`), and bias factors in depth clamp mode through a range past 1
/// (`depth-bias-past-one`).
pub(super) const VERTEX_CLAMPS: [Degradation; 3] = [
    Degradation::ClippedDepthPieces,
    Degradation::SampleDepthClamp,
    Degradation::DepthBiasPastOne,
];

/// One quad of a Metal probe (the M3 Max and an M4 alike, plain and validated): `rfb_span` over `y`
/// -1 to 1 from `x0` to `x1`, its depth running from `z0` at `x0` to `z1` at `x1`, through a 64x1
/// viewport of the depth range `[n, f]`.
struct Span {
    name: &'static str,
    range: [f64; 2],
    x: [f32; 2],
    z: [f32; 2],
}

/// The depth stored at each of the 64 pixels of `span`, in depth clamp mode where `clamps`: each
/// vertex's window depth `n + (f - n) z` clamped before the interpolation where `per_vertex`
/// (Metal's: at 0 in clip mode, to `[0, 1]` in clamp mode; within 1e-7 of each probe row below),
/// interpolated at the pixel's center, a bias's `offset` added (Within 1e-5 of every row of the
/// Metal probes on an M4 and the M3 Max), and each sample clamped (to `[0, 1]` in clip mode, to
/// `[max(min(n, f), 0), max(n, f)]` in clamp mode, its top at least its bottom).
fn span_depths(span: &Span, clamps: bool, per_vertex: bool, offset: f64) -> Vec<f32> {
    let [n, f] = span.range;
    let [x0, x1] = span.x.map(f64::from);
    let vertex = |z: f32| {
        let d = n + (f - n) * f64::from(z);
        match (per_vertex, clamps) {
            (false, _) => d,
            (true, false) => d.max(0.0),
            (true, true) => d.clamp(0.0, 1.0),
        }
    };
    let [d0, d1] = span.z.map(vertex);
    let [low, high] = match clamps {
        false => [0.0, 1.0],
        true => {
            let low = n.min(f).max(0.0);
            [low, n.max(f).max(low)]
        }
    };
    (0..64)
        .map(|i| {
            let x = (f64::from(i) + 0.5) / 32.0 - 1.0;
            let u = (x - x0) / (x1 - x0);
            (d0 + (d1 - d0) * u + offset).clamp(low, high) as f32
        })
        .collect()
}

/// `span` drawn by `pipeline` (into an `RGBA32Float` color too where `reads`) through its range
/// in `mode`, under the depth bias `[depth_bias, slope_scale, clamp]` where set, into a 64x1
/// `Depth32Float` cleared to 0.125 under `Always`: the 64 depths stored, or the runtime's
/// refusal.
fn draw_span(
    probe: &Probe,
    pipeline: &RenderPipelineState,
    span: &Span,
    mode: DepthClipMode,
    reads: bool,
    bias: Option<[f32; 3]>,
) -> Result<Vec<f32>, Error> {
    let target = |pf| {
        let mut descriptor = TextureDescriptor::texture_2d(pf, 64, 1, false);
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::shared();
        probe.device.new_texture(&descriptor).unwrap()
    };
    let depth = target(PixelFormat::Depth32Float);
    let color = target(PixelFormat::RGBA32Float);
    let mut pass = RenderPassDescriptor::new();
    pass.depth_attachment.texture = Some(depth.clone());
    pass.depth_attachment.load_action = LoadAction::Clear;
    pass.depth_attachment.clear_depth = 0.125;
    pass.depth_attachment.store_action = StoreAction::Store;
    if reads {
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(color.clone());
        attachment.load_action = LoadAction::Clear;
        attachment.clear_color = ClearColor::new(-9.0, -9.0, -9.0, -9.0);
        attachment.store_action = StoreAction::Store;
    }
    let bytes: Vec<u8> = [span.x[0], span.x[1], span.z[0], span.z[1]]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let buffer = buffer_with(&probe.device, &bytes).unwrap();
    let [znear, zfar] = span.range;
    let command_buffer = probe.queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    let encoded = (|| {
        encoder.set_render_pipeline_state(pipeline)?;
        encoder.set_depth_stencil_state(&probe.state(CompareFunction::Always, true, None))?;
        encoder.set_viewport(Viewport::new(0.0, 0.0, 64.0, 1.0, znear, zfar))?;
        encoder.set_depth_clip_mode(mode)?;
        if let Some([depth_bias, slope_scale, clamp]) = bias {
            encoder.set_depth_bias(depth_bias, slope_scale, clamp)?;
        }
        encoder.set_vertex_buffer(Some(&buffer), 0, 0)?;
        encoder.draw_primitives(PrimitiveType::TriangleStrip, 0, 4, 1, 0)
    })();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None, "{}", span.name);
    encoded?;
    Ok(floats(&texels(&depth, 4)))
}

/// Metal clamps each vertex's depth before the interpolation (Metal probes on the M3 Max and an
/// M4): a probe's quads inside Metal's guard band and planes, through -0.5 to 1.5 in both clip
/// modes and 0.5 to 2 in clamp mode (clip mode's, which clamps no vertex there, alike), `rfb_span`
/// without a fragment function, store Metal's depths, where the device takes such ranges (natively
/// too: the draw takes the vertex stage's compensation). A draw whose fragment function reads its
/// position's depth (`rfb_position`) keeps a clamp of each sample, and bias factors in clamp mode
/// through 0.5 to 2 hold a depth past 1 at 1 where the device holds no Vulkan range past 1: each
/// announced once, with the vertex stage's clamp. Not run where the device takes no such range
/// (`depth_ranges_refused`).
#[test]
fn vertex_clamps_store_metals_depths() {
    use DepthClipMode::{Clamp, Clip};
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    if let Some(refused) = depth_ranges_refused(&device) {
        println!("{}: depth ranges refused, {refused:?}", device.name());
        return;
    }
    let probe = Probe {
        queue: device.new_command_queue().unwrap(),
        library: library(&device),
        device,
    };
    let function = |name| {
        probe
            .library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(function("rfb_span")));
    descriptor.depth_format = Some(PixelFormat::Depth32Float);
    let bare = probe
        .device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    // Each making's kinds, and the `Shared` buffers of the draws.
    let mut first = forced(&[(
        Degradation::SharedInSystemMemory,
        shared_in_system_memory(&probe.device),
    )]);
    first.extend(made_render_kinds(&probe.device, &descriptor));
    descriptor.fragment = Some(StageSource::Function(function("rfb_position")));
    descriptor.color_attachments[0] =
        Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
    let reading = probe
        .device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    first.extend(made_render_kinds(&probe.device, &descriptor));
    // The vertex stage clamps where the device has the clip distances of Metal's planes (every
    // device measured that takes these ranges).
    let takes = probe
        .device
        .enabled()
        .features
        .contains(&VkFeature::ShaderClipDistance);
    let wide = [-0.5, 1.5];
    let spans = [
        ("guard_right_1.0", wide, [-1.0, 1.0], [0.1, 0.3]),
        ("guard_left_1.0", wide, [-1.0, 1.0], [0.3, 0.1]),
        ("guard_right_1.25", wide, [-1.0, 1.25], [0.1, 0.325]),
        ("guard_left_1.25", wide, [-1.25, 1.0], [0.325, 0.1]),
        ("guard_right_1.5", wide, [-1.0, 1.5], [0.1, 0.35]),
        ("guard_left_1.5", wide, [-1.5, 1.0], [0.35, 0.1]),
        ("split_05_2_inside", [0.5, 2.0], [-1.0, 1.0], [0.1, 0.6]),
    ]
    .map(|(name, range, x, z)| Span { name, range, x, z });
    let mut differ = Vec::new();
    for span in &spans {
        for mode in [Clip, Clamp] {
            let metal = span_depths(span, mode == Clamp, takes, 0.0);
            let drawn = draw_span(&probe, &bare, span, mode, false, None);
            println!("{} {mode:?}: {drawn:?}", span.name);
            if !drawn.as_ref().is_ok_and(|stored| near(stored, &metal)) {
                differ.push(format!(
                    "{} {mode:?}: {drawn:?}, Metal {metal:?}",
                    span.name
                ));
            }
        }
    }
    // A ramp from 0.1 to 0.6 through -0.5 to 1.5 in clip mode, read by its fragment function:
    // `clamp(-0.3 + u, 0, 1)`, where Metal stores `0.7 u`.
    let ramp = Span {
        name: "reading",
        range: wide,
        x: [-1.0, 1.0],
        z: [0.1, 0.6],
    };
    let kept = span_depths(&ramp, false, false, 0.0);
    let drawn = draw_span(&probe, &reading, &ramp, Clip, true, None);
    if !drawn.as_ref().is_ok_and(|stored| near(stored, &kept)) {
        differ.push(format!("reading: {drawn:?}, each sample clamped {kept:?}"));
    }
    let biased = Span {
        name: "biased",
        range: [0.5, 2.0],
        ..ramp
    };
    if let Err(error) = draw_span(&probe, &bare, &biased, Clamp, false, Some([1.0, 1.0, 0.0])) {
        differ.push(format!("biased: {error:?}"));
    }
    assert!(differ.is_empty(), "{differ:#?}");
    let unrestricted = probe
        .device
        .enabled()
        .extensions
        .contains(&VkExtension::ExtDepthRangeUnrestricted);
    let expected: Vec<Degradation> = VERTEX_CLAMPS
        .into_iter()
        .filter(|kind| match kind {
            Degradation::ClippedDepthPieces => takes,
            Degradation::DepthBiasPastOne => !(takes && unrestricted),
            _ => true,
        })
        .collect();
    assert_made_once(&messages, &expected, &first);
    quiet(probe, &messages);
}

/// A bias's offset through a range whose vertices Metal clamps stores Metal's depths (Metal probes
/// on an M4 and the M3 Max, plain and validated; every row of both within 1e-5 of [`span_depths`]'
/// model, the offset added after the interpolation): `rfb_span` without a fragment function over
/// the whole target, its `z` from 0 to 1 (probe 15's ramp) or 0.1 to 0.6 (probe 20's). An offset
/// (`set_depth_bias(±inf, 0, ±0.25)`) through -0.5 to 1.5 in clip mode where the vertex stage
/// clamps each vertex's depth at 0, and in clamp mode, past 1, where the device also holds a Vulkan
/// range past 1 (`VK_EXT_depth_range_unrestricted`); through 0.25 to 0.75 in clamp mode everywhere.
/// Bias factors of an offset of 0.75 (`set_depth_bias(1e9, 0, 0.75)`) through 0.5 to 1.5 and 1.5 to
/// 3 in clamp mode store Metal's depths past 1 where the vertex stage clamps and the device holds
/// such a range; elsewhere 0.5 to 1.5 holds them at 1 (`depth-bias-past-one`) and 1.5 to 3 is
/// refused `NotImplemented(ViewportDepthRange)`. An offset where the runtime does not build it:
/// `NotImplemented(NonFiniteDepthBias)`. The vertex stage's clamp announces `clipped-depth-pieces`,
/// and a bias in clamp mode through a range inside `[0, 1]` `sample-depth-clamp` (a primitive past
/// Metal's planes, which Metal adds the bias to after its clamp of each vertex's depth to `[0, 1]`,
/// stores otherwise). Not run where the device takes no such range (`depth_ranges_refused`).
#[test]
fn offsets_through_clamped_vertices_store_metals_depths() {
    use DepthClipMode::{Clamp, Clip};
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    if let Some(refused) = depth_ranges_refused(&device) {
        println!("{}: depth ranges refused, {refused:?}", device.name());
        return;
    }
    let probe = Probe {
        queue: device.new_command_queue().unwrap(),
        library: library(&device),
        device,
    };
    let name = probe.device.name().to_owned();
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
        probe
            .library
            .new_function_with_name("rfb_span")
            .unwrap()
            .function_descriptor(),
    ));
    descriptor.depth_format = Some(PixelFormat::Depth32Float);
    let bare = probe
        .device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let mut first = forced(&[(
        Degradation::SharedInSystemMemory,
        shared_in_system_memory(&probe.device),
    )]);
    first.extend(made_render_kinds(&probe.device, &descriptor));
    let takes = probe
        .device
        .enabled()
        .features
        .contains(&VkFeature::ShaderClipDistance);
    let unrestricted = probe
        .device
        .enabled()
        .extensions
        .contains(&VkExtension::ExtDepthRangeUnrestricted);
    let widens = takes && unrestricted;
    let inf = f32::INFINITY;
    let offset_gap = Err(Error::NotImplemented(RuntimeGap::NonFiniteDepthBias));
    let range_gap = Err(Error::NotImplemented(RuntimeGap::ViewportDepthRange));
    let ramp = [0.0, 1.0];
    let probe20 = [0.1, 0.6];
    // (name, range, mode, z, bias, the offset Metal adds, the runtime builds it, its depths
    // past 1 held at 1 where the device takes no Vulkan range of Metal's clamp of each sample)
    type Case = (
        &'static str,
        [f64; 2],
        DepthClipMode,
        [f32; 2],
        [f32; 3],
        f64,
        Result<(), Error>,
        bool,
    );
    let built = |yes: bool, gap: &Result<(), Error>| match yes {
        true => Ok(()),
        false => gap.clone(),
    };
    let cases: [Case; 8] = [
        (
            "clip_m05_15_offset",
            [-0.5, 1.5],
            Clip,
            ramp,
            [inf, 0.0, 0.25],
            0.25,
            built(takes, &offset_gap),
            false,
        ),
        (
            "clip_m05_15_negoffset",
            [-0.5, 1.5],
            Clip,
            ramp,
            [-inf, 0.0, -0.25],
            -0.25,
            built(takes, &offset_gap),
            false,
        ),
        (
            "clamp_m05_15_offset",
            [-0.5, 1.5],
            Clamp,
            ramp,
            [inf, 0.0, 0.25],
            0.25,
            built(widens, &offset_gap),
            false,
        ),
        (
            "clamp_025_075_offset",
            [0.25, 0.75],
            Clamp,
            ramp,
            [inf, 0.0, 0.25],
            0.25,
            Ok(()),
            false,
        ),
        (
            "clamp_025_075_negoffset",
            [0.25, 0.75],
            Clamp,
            ramp,
            [-inf, 0.0, -0.25],
            -0.25,
            Ok(()),
            false,
        ),
        (
            "clamp_05_15_factors",
            [0.5, 1.5],
            Clamp,
            ramp,
            [1e9, 0.0, 0.75],
            0.75,
            Ok(()),
            true,
        ),
        (
            "clamp_15_3_up",
            [1.5, 3.0],
            Clamp,
            probe20,
            [1e9, 0.0, 0.75],
            0.75,
            built(widens, &range_gap),
            false,
        ),
        (
            "clamp_15_3_down",
            [1.5, 3.0],
            Clamp,
            probe20,
            [-1e9, 0.0, -0.75],
            -0.75,
            built(widens, &range_gap),
            false,
        ),
    ];
    let mut differ = Vec::new();
    for (case, range, mode, z, bias, offset, built, held) in cases {
        let span = Span {
            name: case,
            range,
            x: [-1.0, 1.0],
            z,
        };
        let metal = span_depths(&span, mode == Clamp, true, offset);
        let drawn = draw_span(&probe, &bare, &span, mode, false, Some(bias));
        println!("{case} {mode:?} on {name}: {drawn:?}");
        let fine = match (&drawn, built) {
            (Ok(stored), Ok(())) if held && !widens => {
                // The one clamp holds at most 1: Metal's depths, each held at 1.
                let capped: Vec<f32> = metal.iter().map(|d| d.min(1.0)).collect();
                near(stored, &capped)
            }
            (Ok(stored), Ok(())) => near(stored, &metal),
            (Err(error), Err(gap)) => *error == gap,
            _ => false,
        };
        if !fine {
            differ.push(format!("{case} {mode:?}: {drawn:?}, Metal {metal:?}"));
        }
    }
    assert!(differ.is_empty(), "{differ:#?} on {name}");
    // The offsets through 0.25 to 0.75 announce `sample-depth-clamp` on every device: Metal
    // adds them to each vertex's depth clamped to `[0, 1]`, which a primitive past its planes
    // would show.
    let expected: &[Degradation] = match (takes, unrestricted) {
        (true, true) => &VERTEX_CLAMPS[..2],
        (true, false) => &VERTEX_CLAMPS,
        (false, _) => &VERTEX_CLAMPS[1..],
    };
    assert_made_once(&messages, expected, &first);
    quiet(probe, &messages);
}

/// A depth a bias takes past `[0, 1]` in clip mode stores Metal's clamp. The fullscreen triangle
/// at depth `z + 0.05 x` (0.025 per pixel) through a viewport of `[0, 1]`, under a slope-scaled
/// bias of 4 (0.1 at that slope), reads `z + 0.1` and more along its columns: from 0.9 its
/// columns 2 and 3 read 1.0125 and 1.0375, which Metal stores as 1 (in clip mode a depth is clamped
/// to `[0, 1]`); under a bias of -4 from 0.1 its columns 0 and 1 read -0.0375 and -0.0125, stored
/// as +0. Each other column stores the depth its fragment read. A device that enabled
/// `VK_EXT_depth_range_unrestricted`, and so stores such a depth unclamped, without the native
/// range (`native_depth_range`) announces `biased-depth-past-unit-range` once instead, and its
/// clamped columns print.
#[test]
fn biased_depths_past_unit_range_store_metals_clamp() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let unclamped = device
        .enabled()
        .extensions
        .contains(&VkExtension::ExtDepthRangeUnrestricted)
        && !native_depth_range(&device);
    let probe = Probe {
        queue: device.new_command_queue().unwrap(),
        library: library(&device),
        device,
    };
    let pipelines = Pipelines {
        position: pipeline(&probe, "rfb_position"),
        depth: pipeline(&probe, "rfb_depth"),
    };
    let name = probe.device.name().to_owned();
    let case = |name, z, slope_scale, seen: [f32; 4]| Case {
        name,
        viewport: [0.0, 4.0, 0.0, 1.0],
        depth: [z, 0.05],
        mode: DepthClipMode::Clip,
        // Metal's facing of the fullscreen triangle through a viewport of positive width, as
        // `vp_pos_clip_0_1_z075`'s.
        fragment: Fragment::Position { front: false },
        bias: Some([0.0, slope_scale, 0.0]),
        cull: CullMode::None,
        seen: seen.map(Some),
        stored: seen.map(|z| z.clamp(0.0, 1.0)),
        near: false,
    };
    let cases = [
        case("biased past 1", 0.9, 4.0, [0.9625, 0.9875, 1.0125, 1.0375]),
        case(
            "biased below 0",
            0.1,
            -4.0,
            [-0.0375, -0.0125, 0.0125, 0.0375],
        ),
    ];
    let mut differ = Vec::new();
    for case in &cases {
        let (color, depth) = draw(&probe, &pipelines, case).unwrap();
        let (metal_color, metal_depth) = metal(case);
        // The clamped columns store the bound itself, +0 below.
        let bounds = depth
            .iter()
            .zip(&metal_depth)
            .all(|(d, m)| !(*m == 0.0 || *m == 1.0) || d.to_bits() == m.to_bits());
        let stored = near(&depth, &metal_depth) && bounds;
        println!(
            "biased depth on {name}, {}: read {:?}, stored {:?}{}",
            case.name,
            &color
                .as_chunks::<4>()
                .0
                .iter()
                .map(|t| t[2])
                .collect::<Vec<_>>()[..4],
            &depth[..4],
            if stored { "" } else { " (unclamped)" }
        );
        if !near(&color, &metal_color) || (!stored && !unclamped) {
            differ.push((case.name, color, depth));
        }
    }
    assert!(differ.is_empty(), "on {name}: {differ:?}");
    if unclamped {
        assert_made_once(&messages, &UNCLAMPED_BIAS, &made_at(&probe, 1));
    } else {
        let notices = messages.take_notices();
        let kind = Some(Degradation::BiasedDepthPastUnitRange);
        assert!(
            !notices
                .iter()
                .any(|n| Degradation::from_code(&n.id_name) == kind),
            "on {name}: {notices:?}"
        );
    }
    quiet(probe, &messages);
}
