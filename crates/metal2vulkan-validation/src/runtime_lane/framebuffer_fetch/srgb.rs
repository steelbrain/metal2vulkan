//! An 8-bit sRGB color attachment fetched (the sRGB fetch form) on the device the cfg selects,
//! under the validation layer: Metal's answers measured by our probes
//! (an Apple M4, macOS 27.0.1), the same passes.
//!
//! A `SIDE`² private `BGRA8Unorm_sRGB` or `RGBA8Unorm_sRGB` render target, single-sampled or of
//! `MSAA` samples, holds in its texel or sample `k` (`k = i · samples + s` for sample `s` of texel
//! `i = y · SIDE + x`) the memory bytes `k`, `37 k + k / 256`, `101 k + 7 (k / 256)`, `k / 16`
//! (each mod 256): each byte takes every value. A pass loads and stores it and draws
//! `fetch_ordered` (`fragment_fetch`: `float4` read in a raster order group, half of it returned)
//! over the fullscreen triangle at depth 0.5 `N` times, at the target's samples, no blending,
//! every channel written. Metal stores in each color byte the byte [`HALVED`] gives (the stored
//! code decoded, halved and encoded; red, green and blue alike), and in alpha `(a + 1) / 2`, once
//! per draw and on every sample: each draw reads the code the last one stored (the composition of
//! the one-draw map matched every case measured, both formats, both sample counts). The cases,
//! each a pass over a fresh target:
//!
//! - no depth or stencil attachment, 1, 2 and 3 draws;
//! - a `Depth32Float_Stencil8` attachment of the target's samples loaded at 1.0 (cleared by a
//!   pass before), depth `Always` without writes, 3 draws;
//! - the same attachment with its left half (`x < SIDE / 2`) at 0.25 (that pass drew there, depth
//!   `Always` with writes, no color writes), depth `Less` with writes, 1 and 2 draws: the left
//!   half keeps its bytes, the right half is halved once (the second draw fails `Less` against
//!   the 0.5 the first wrote).
//!
//! A single-sampled target is filled by a blit and read back by one. A multisampled one is filled
//! sample by sample by `rt6_samples`' fill (each code drawn as the float that encodes it exactly,
//! as `resolve_rounding.rs` fills its samples) and read back sample by sample through its
//! `RGBA8Uint` view (`candidate::runtime_samples`); the fill is checked to store the bytes drawn.
//!
//! In the pixel interlock (RADV, ANV, NVIDIA, MoltenVK) the compile layer reads and writes the
//! attachment through an `R32_UINT` view, decoding and encoding by the device's conversion
//! (`lower::fragment::fetch`, `device-pack-conversion`): the pass renders the target into a scratch image made mutable to
//! that view (`runtime::attachment_stand_in`), or at 4x, on a device without multisampled storage
//! images, into sample planes made so (`runtime::sample_planes`). Through input attachments
//! (lavapipe) the function reads the device's own decode of the code, which on llvmpipe is not
//! Metal's, and the pipeline records `srgb-texel-decode` there alone (asserted).
//! Every byte is Metal's in either form, but where the pipeline records `device-color-conversion`
//! (the device's encode of the written color, every device but MoltenVK's) or
//! `device-pack-conversion` (the interlock's decode and encode): there a channel within
//! one code of Metal's holds, counted and printed (`device_rounded`), and a case of more than one
//! draw without a failing depth test is held to Metal's one-draw map over the bytes the device
//! stored after one draw fewer (each draw reads what the last one stored, so the device's codes
//! compound: llvmpipe's two draws land two codes from Metal's at the dark end). The form each
//! device took is printed.
//!
//! A driver that drops a multisampled pass's image writes under early fragment tests
//! (`DriverWorkaround::EarlyTestsDropMultisampledImageWrites`: MoltenVK on Apple's GPU) has the
//! per-sample form beside a depth and stencil attachment declare no early tests and store before
//! the tests, so each draw's tests decide: there the multisampled `Always` case, whose tests drop
//! no fragment, is drawn, Metal's bytes; the `Less` cases are refused at the draw, typed,
//! `NotImplemented(FramebufferFetch)` at `Attachment::Depth`, and each refusal is asserted and
//! printed; the cases without the attachment are drawn. Where the fragment function computes each
//! sample's depth (`VkProperty::CenterFragCoordDepth`, MoltenVK on an M4), the `Less` cases run the
//! depth test in the function against the depth's stand-in and are drawn, Metal's bytes.

use super::{device_rounded, form, library, reads_planes, through_inputs};
use crate::candidate::runtime_samples::{encode_through, shape};
use crate::executor_contract::multisample::{ReadbackView, SampleRead};
use crate::runtime_lane::degrade_notices::made_render_kinds;
use crate::runtime_lane::resolves::{lane, quiet, Lane, FULL, WHITE};
use crate::runtime_lane::{buffer_with, WAIT};
use metal2vulkan::device::{DriverWorkaround, VkProperty};
use metal2vulkan::error::{Attachment, BindError, Feature, Needs};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    ColorWriteMask, CompareFunction, PixelFormat, PrimitiveType, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, Degradation, Error, Library, LoadAction, Origin, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, Size, StoreAction,
    Texture, TextureDescriptor,
};
use std::collections::BTreeMap;

/// The target's side, as the probes'.
const SIDE: u64 = 64;

/// The multisampled targets' samples, as the probe's.
const MSAA: u32 = 4;

/// The depth and stencil attachment's format.
const DS: PixelFormat = PixelFormat::Depth32FloatStencil8;

/// The fullscreen triangle's depth in the measured pass, and the left half's in the setup pass.
const Z: f32 = 0.5;
const LEFT_Z: f32 = 0.25;

/// What Metal stores in a red, green or blue byte of an 8-bit sRGB texel holding `c` when a draw
/// returns half of what it fetched (`HALVED[c]`): our probe's `COLOR_HALF_F`, which our probe
/// measured the same for `RGBA8Unorm_sRGB` and at 4 samples.
const HALVED: [u8; 256] = [
    0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 6, 6, 6, 7, 8, //
    9, 10, 10, 11, 12, 13, 13, 14, 15, 15, 16, 17, 18, 18, 19, 20, //
    21, 21, 22, 23, 23, 24, 25, 26, 27, 27, 28, 29, 30, 30, 31, 32, //
    33, 33, 34, 35, 36, 36, 37, 38, 38, 39, 40, 41, 42, 42, 43, 44, //
    44, 45, 46, 47, 48, 48, 49, 50, 50, 51, 52, 53, 53, 54, 55, 56, //
    57, 57, 58, 59, 59, 60, 61, 62, 62, 63, 64, 65, 65, 66, 67, 68, //
    68, 69, 70, 71, 71, 72, 73, 74, 74, 75, 76, 77, 77, 78, 79, 80, //
    80, 81, 82, 83, 83, 84, 85, 86, 86, 87, 88, 89, 89, 90, 91, 92, //
    92, 93, 94, 95, 95, 96, 97, 98, 98, 99, 100, 101, 101, 102, 103, 104, //
    104, 105, 106, 107, 107, 108, 109, 110, 110, 111, 112, 113, 113, 114, 115, 116, //
    116, 117, 118, 119, 119, 120, 121, 122, 122, 123, 124, 125, 125, 126, 127, 128, //
    128, 129, 130, 131, 131, 132, 133, 134, 134, 135, 136, 137, 137, 138, 139, 140, //
    140, 141, 142, 143, 143, 144, 145, 146, 146, 147, 148, 149, 149, 150, 151, 152, //
    152, 153, 154, 155, 155, 156, 157, 158, 158, 159, 160, 161, 161, 162, 163, 164, //
    164, 165, 166, 167, 167, 168, 169, 170, 170, 171, 172, 173, 173, 174, 175, 176, //
    176, 177, 178, 179, 179, 180, 181, 182, 182, 183, 184, 185, 185, 186, 187, 188, //
];

/// The depth and stencil attachment of a case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Depth {
    /// None.
    None,
    /// Loaded at 1.0; depth `Always`, no writes.
    Always,
    /// Loaded at 0.25 left and 1.0 right; depth `Less`, writes on.
    Less,
}

/// The cases each target takes: its depth attachment, and the draws.
const CASES: [(Depth, usize); 6] = [
    (Depth::None, 1),
    (Depth::None, 2),
    (Depth::None, 3),
    (Depth::Always, 3),
    (Depth::Less, 1),
    (Depth::Less, 2),
];

/// A target: its format and samples.
#[derive(Clone, Copy, Debug)]
struct Target {
    pf: PixelFormat,
    samples: u32,
}

impl Target {
    /// The bytes of each texel or sample, before the pass (the module docs).
    fn initial(self) -> Vec<u8> {
        (0..(SIDE * SIDE) as usize * self.samples as usize)
            .flat_map(|k| [k, 37 * k + k / 256, 101 * k + 7 * (k / 256), k / 16].map(|b| b as u8))
            .collect()
    }

    /// Metal's bytes after `draws` draws of the case `depth` (the module docs).
    fn expected(self, depth: Depth, draws: usize) -> Vec<u8> {
        self.expected_over(self.initial(), depth, draws)
    }

    /// Metal's bytes after `draws` draws of the case `depth` over a target holding `bytes`.
    fn expected_over(self, mut bytes: Vec<u8>, depth: Depth, draws: usize) -> Vec<u8> {
        for (k, texel) in bytes.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let x = (k / self.samples as usize) as u64 % SIDE;
            let halvings = match depth {
                Depth::Less if x < SIDE / 2 => 0,
                Depth::Less => 1,
                _ => draws,
            };
            for _ in 0..halvings {
                for byte in &mut texel[..3] {
                    *byte = HALVED[usize::from(*byte)];
                }
                texel[3] = texel[3].div_ceil(2);
            }
        }
        bytes
    }

    /// A private `SIDE`² texture of `pf` and `usage`, of the target's samples.
    fn texture(self, lane: &Lane, pf: PixelFormat, usage: TextureUsage) -> Texture {
        let mut descriptor = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
        if self.samples > 1 {
            descriptor.texture_type = TextureType::Type2DMultisample;
            descriptor.sample_count = self.samples.into();
        }
        descriptor.usage = usage;
        descriptor.resource_options = ResourceOptions::private();
        lane.device.new_texture(&descriptor).unwrap()
    }

    /// The color target, holding [`Target::initial`].
    fn filled(self, lane: &Lane) -> Texture {
        let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        if self.samples > 1 {
            let texture = self.texture(lane, self.pf, usage | TextureUsage::PIXEL_FORMAT_VIEW);
            self.fill_samples(lane, &texture);
            return texture;
        }
        let texture = self.texture(lane, self.pf, usage);
        let bytes = self.initial();
        let source = buffer_with(&lane.device, &bytes).unwrap();
        let command_buffer = lane.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            4 * SIDE,
            bytes.len() as u64,
            Size::new(SIDE, SIDE, 1),
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

    /// Draw [`Target::initial`] into the multisampled `texture`, sample by sample: each code the
    /// float that encodes it exactly (an sRGB color channel's decode, alpha `a / 255`), each
    /// float's channel the byte it lands on (a `BGRA` format's red its third byte).
    fn fill_samples(self, lane: &Lane, texture: &Texture) {
        let decode = |code: u8| {
            let c = f64::from(code) / 255.0;
            let linear = if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            };
            linear as f32
        };
        let bgra = self.pf == PixelFormat::BGRA8UnormSRGB;
        let floats: Vec<u8> = self
            .initial()
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|b| {
                let (red, blue) = if bgra { (b[2], b[0]) } else { (b[0], b[2]) };
                [
                    decode(red),
                    decode(b[1]),
                    decode(blue),
                    f32::from(b[3]) / 255.0,
                ]
            })
            .flat_map(f32::to_le_bytes)
            .collect();
        let samples = library(lane, "rt6_samples");
        let function = |name| {
            samples
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor()
        };
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function("rt6_fill_vertex")));
        descriptor.fragment = Some(StageSource::Function(function("rt6_fill_rgba32float")));
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(self.pf));
        descriptor.raster_sample_count = self.samples;
        let pipeline = lane
            .device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let values = buffer_with(&lane.device, &floats).unwrap();
        let shape = shape(&lane.device, SIDE as u32, self.samples).unwrap();
        let mut fill = RenderPassDescriptor::new();
        let target = &mut fill.color_attachments[0];
        target.texture = Some(texture.clone());
        target.load_action = LoadAction::DontCare;
        target.store_action = StoreAction::Store;
        lane.run(&fill, |_, encoder, _| {
            encoder.set_render_pipeline_state(&pipeline).unwrap();
            encoder.set_fragment_buffer(Some(&values), 0, 0).unwrap();
            encoder.set_fragment_buffer(Some(&shape), 0, 1).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        })
        .unwrap();
    }

    /// Every texel's bytes (every sample's, in order, of a multisampled target) of `texture`.
    fn read(self, lane: &Lane, texture: &Texture) -> Vec<u8> {
        if self.samples == 1 {
            return lane.read(texture, (0, 0), 4, BlitOption::None);
        }
        let view = texture.new_texture_view(PixelFormat::RGBA8Uint).unwrap();
        let command_buffer = lane.queue.command_buffer().unwrap();
        let readback = encode_through(
            &lane.device,
            &command_buffer,
            &view,
            "rt6_read_bytes4",
            SampleRead::Color(ReadbackView::R32Uint),
            [SIDE as u32, SIDE as u32],
            self.samples,
        )
        .unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        readback.bytes().unwrap()
    }

    /// A pipeline of `fragment` (a library's function) over the fullscreen triangle, drawing the
    /// target at its samples, with the depth and stencil attachment's format where `depth`,
    /// writing the color's `mask`, or its refusal.
    fn pipeline(
        self,
        lane: &Lane,
        (library, fragment): (&Library, &str),
        depth: bool,
        mask: ColorWriteMask,
    ) -> Result<RenderPipelineState, Error> {
        lane.device.new_render_pipeline_state(
            &self.descriptor(lane, (library, fragment), depth, mask),
            &PipelineOptions::default(),
        )
    }

    /// The descriptor of [`Target::pipeline`].
    fn descriptor(
        self,
        lane: &Lane,
        (library, fragment): (&Library, &str),
        depth: bool,
        mask: ColorWriteMask,
    ) -> RenderPipelineDescriptor {
        let function = |library: &Library, name| {
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor()
        };
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function(&lane.library, FULL)));
        descriptor.fragment = Some(StageSource::Function(function(library, fragment)));
        let mut color = ColorAttachmentDescriptor::new(self.pf);
        color.write_mask = mask;
        descriptor.color_attachments[0] = Some(color);
        if depth {
            descriptor.depth_format = Some(DS);
            descriptor.stencil_format = Some(DS);
        }
        descriptor.raster_sample_count = self.samples;
        descriptor
    }

    /// Run the case `depth` of `draws` draws of `fetch` over a fresh target: its bytes after, or
    /// the first draw's refusal (the pass then draws nothing more).
    fn case(
        self,
        lane: &Lane,
        fetch: &RenderPipelineState,
        depth: Depth,
        draws: usize,
    ) -> Result<Vec<u8>, Error> {
        let color = self.filled(lane);
        let attachment =
            (depth != Depth::None).then(|| self.texture(lane, DS, TextureUsage::RENDER_TARGET));
        if let Some(attachment) = &attachment {
            let setup = pass(&color, Some(attachment), true);
            let left = (depth == Depth::Less).then(|| {
                let white = self
                    .pipeline(lane, (&lane.library, WHITE), true, ColorWriteMask::NONE)
                    .unwrap();
                let state = depth_state(CompareFunction::Always, true);
                let state = lane.device.new_depth_stencil_state(&state).unwrap();
                (white, state)
            });
            lane.run(&setup, |lane, encoder, held| {
                if let Some((white, state)) = &left {
                    let z = buffer_with(&lane.device, &LEFT_Z.to_le_bytes()).unwrap();
                    encoder.set_render_pipeline_state(white).unwrap();
                    encoder.set_depth_stencil_state(state).unwrap();
                    encoder.set_vertex_buffer(Some(&z), 0, 0).unwrap();
                    encoder
                        .set_scissor_rect(ScissorRect::new(0, 0, SIDE / 2, SIDE))
                        .unwrap();
                    encoder
                        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                        .unwrap();
                    held.push(z);
                }
            })
            .unwrap();
        }
        let measured = pass(&color, attachment.as_ref(), false);
        let state = match depth {
            Depth::None => None,
            Depth::Always => Some(depth_state(CompareFunction::Always, false)),
            Depth::Less => Some(depth_state(CompareFunction::Less, true)),
        }
        .map(|state| lane.device.new_depth_stencil_state(&state).unwrap());
        let mut drawn = Ok(());
        lane.run(&measured, |lane, encoder, held| {
            let z = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
            encoder.set_render_pipeline_state(fetch).unwrap();
            if let Some(state) = &state {
                encoder.set_depth_stencil_state(state).unwrap();
            }
            encoder.set_vertex_buffer(Some(&z), 0, 0).unwrap();
            for _ in 0..draws {
                drawn = encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0);
                if drawn.is_err() {
                    break;
                }
            }
            held.push(z);
        })
        .unwrap();
        drawn.map(|()| self.read(lane, &color))
    }
}

/// A pass over `color`, loaded and stored, and over `depth` (both aspects), cleared to 1.0 and 0
/// where `clear`, else loaded, and stored.
fn pass(color: &Texture, depth: Option<&Texture>, clear: bool) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    let target = &mut descriptor.color_attachments[0];
    target.texture = Some(color.clone());
    target.load_action = LoadAction::Load;
    target.store_action = StoreAction::Store;
    if let Some(depth) = depth {
        let load = match clear {
            true => LoadAction::Clear,
            false => LoadAction::Load,
        };
        let attachment = &mut descriptor.depth_attachment;
        attachment.texture = Some(depth.clone());
        attachment.load_action = load;
        attachment.clear_depth = 1.0;
        attachment.store_action = StoreAction::Store;
        let attachment = &mut descriptor.stencil_attachment;
        attachment.texture = Some(depth.clone());
        attachment.load_action = load;
        attachment.clear_stencil = 0;
        attachment.store_action = StoreAction::Store;
    }
    descriptor
}

/// A depth state of `compare`, writing where `write`.
fn depth_state(compare: CompareFunction, write: bool) -> DepthStencilDescriptor {
    let mut state = DepthStencilDescriptor::default();
    state.depth_compare_function = compare;
    state.depth_write_enabled = write;
    state
}

/// The texels or samples of `read` that differ from `want`, by index: each read's and Metal's
/// bytes.
fn differing(read: &[u8], want: &[u8]) -> Vec<(usize, [u8; 4], [u8; 4])> {
    let texel = |bytes: &[u8]| -> [u8; 4] { bytes.try_into().unwrap() };
    read.as_chunks::<4>()
        .0
        .iter()
        .zip(want.as_chunks::<4>().0.iter())
        .enumerate()
        .filter(|(_, (r, w))| r != w)
        .map(|(k, (r, w))| (k, texel(r), texel(w)))
        .collect()
}

/// Each case's bytes are Metal's, on every device, for both formats single-sampled and at `MSAA`
/// samples, but within one code where the pipeline records `device-color-conversion` (the module
/// docs).
#[test]
fn an_srgb_target_is_fetched_as_metal_fetches_it() {
    let (messages, _cache, lane) = lane();
    let fetch = library(&lane, "fragment_fetch");
    let device = lane.device.name();
    let inputs = through_inputs(&lane);
    for pf in [PixelFormat::BGRA8UnormSRGB, PixelFormat::RGBA8UnormSRGB] {
        for samples in [1, MSAA] {
            let target = Target { pf, samples };
            if samples > 1 {
                let filled = target.filled(&lane);
                assert_eq!(
                    target.read(&lane, &filled),
                    target.initial(),
                    "{pf:?} at {samples}x on {device}: the fill stored other bytes than drawn"
                );
            }
            let all = ColorWriteMask::ALL;
            let plain = target
                .pipeline(&lane, (&fetch, "fetch_ordered"), false, all)
                .unwrap();
            let descriptor = target.descriptor(&lane, (&fetch, "fetch_ordered"), false, all);
            let made = made_render_kinds(&lane.device, &descriptor);
            let converts = made.contains(&Degradation::DeviceColorConversion)
                || made.contains(&Degradation::DevicePackConversion);
            assert_eq!(
                made.contains(&Degradation::SrgbTexelDecode),
                inputs,
                "{pf:?} at {samples}x on {device}: the device's decode of the input attachment"
            );
            // What the cases without the attachment stored, by their draws: where the device
            // converts, a later draw is Metal's map over the device's own bytes (the module docs).
            let mut stored: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
            // With the depth and stencil attachment, where the driver drops the per-sample
            // form's writes under early tests, only a draw whose tests drop no fragment runs (the
            // module docs).
            let defect = DriverWorkaround::EarlyTestsDropMultisampledImageWrites;
            let defective = samples > 1
                && !inputs
                && lane
                    .device
                    .compile_device()
                    .profile()
                    .workarounds()
                    .contains(&defect);
            let computed = lane
                .device
                .compile_device()
                .profile()
                .enabled()
                .properties
                .contains(&VkProperty::CenterFragCoordDepth);
            let tested = target
                .pipeline(&lane, (&fetch, "fetch_ordered"), true, all)
                .unwrap_or_else(|error| {
                    panic!(
                        "{pf:?} at {samples}x with {DS:?} on {device} (driver defect \
                         {defective}): {error}"
                    )
                });
            for (depth, draws) in CASES {
                let state = match depth {
                    Depth::None => &plain,
                    _ => &tested,
                };
                let what = format!("{pf:?} at {samples}x, {depth:?} x{draws} on {device}");
                // Unless the function tests the depth itself, each sample's computed.
                let refused = defective && depth == Depth::Less && !computed;
                let read = match (target.case(&lane, state, depth, draws), refused) {
                    (Ok(read), false) => read,
                    (Err(Error::Unsupported(u) | Error::Bind(BindError::Unsupported(u))), true) => {
                        assert_eq!(
                            (
                                u.feature,
                                u.needs,
                                u.location.as_ref().and_then(|at| at.attachment())
                            ),
                            (
                                Feature::FramebufferFetch,
                                Needs::NotImplemented,
                                Some(Attachment::Depth)
                            ),
                            "{what}"
                        );
                        eprintln!(
                            "sRGB framebuffer fetch {what} (driver defect {defect:?}): refused at \
                             the draw, at the depth attachment, typed"
                        );
                        continue;
                    }
                    (other, refused) => panic!(
                        "{what} (refusal expected {refused}): {other:?}",
                        other = other.map(|_| "the bytes")
                    ),
                };
                let before = match depth {
                    Depth::None | Depth::Always if converts => stored.get(&(draws - 1)),
                    _ => None,
                };
                let want = match before {
                    Some(bytes) => target.expected_over(bytes.clone(), depth, 1),
                    None => target.expected(depth, draws),
                };
                if depth == Depth::None {
                    stored.insert(draws, read.clone());
                }
                let (wrong, rounded) = device_rounded(differing(&read, &want), converts);
                let total = want.len() / 4;
                let shown = &wrong[..wrong.len().min(8)];
                assert!(
                    wrong.is_empty(),
                    "{what}: {} of {total} texels or samples differ from Metal's, first (index, \
                     read, Metal's) {shown:?}",
                    wrong.len()
                );
                let planes = samples > 1 && reads_planes(&lane);
                let bytes = match rounded {
                    0 => "Metal's bytes".to_owned(),
                    n => format!("{n} channels one code from Metal's, the device's conversion"),
                };
                eprintln!(
                    "sRGB framebuffer fetch {what} through {}{}: {bytes}",
                    form(&lane),
                    if planes { " (sample planes)" } else { "" },
                );
            }
        }
    }
    quiet(lane, &messages);
}
