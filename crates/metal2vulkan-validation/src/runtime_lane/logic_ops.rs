//! The private logic operation of OpenGL on Metal (`PrivateRenderState::logic_operation_enabled`
//! and `logic_operation`), on the device the cfg selects, under the validation layer, over the
//! owned fixture `a8r_targets` (a fullscreen triangle and a fragment writing buffer 0's color).
//!
//! Metal's rule, measured with our own probes `spi`, `spi2` and `logic_operation` on an Apple M3
//! Max under macOS 27.0.1 (targets cleared to one color, a fullscreen triangle of another, every
//! operation, blended by one and one and not): the low four bits of `logicOperation` choose the
//! operation in Direct3D's order, applied to the fragment's code and the stored one of each
//! normalized and integer channel, blending ignored; a `RGBA16Float` attachment ignores the
//! operation and blends. Each draw here is checked against that rule, the codes computed as Metal
//! converts the colors, and a float attachment's against the same draw without the operation.
//! Copy and no-op draw on every device; the others through Vulkan's `logicOp` where the device has
//! it, and through the fragment function where it does not and orders framebuffer fetch (MoltenVK
//! through the pixel interlock; lavapipe, its `logicOp` withheld, through rasterization-order input
//! attachments), each run here both ways where the device has the feature; on a device that
//! orders fetch in neither form every other operation is refused naming the three features.
//! Metal operates on `RGB10A2Unorm`'s codes widened to 16 bits and narrows the result back (the
//! `logic_operation` probe), as the fragment function does; Vulkan's `logicOp` operates on the
//! 10-bit codes themselves, a known gap (`docs/FAST-PATHS.md`) whose texels the lane counts apart.
//!
//! Where the device converts a color its own way, the operation takes the device's code: a drawn
//! channel's where the pipeline's fragment stage records `PrecisionNote::DeviceColorConversion`
//! (the function's own operation records it for every normalized attachment: its codes are the
//! device's `float` product rounded to nearest even), and a cleared channel's where the device
//! announces `device-color-conversion` for the clears (Vulkan lets a device store either code
//! nearest a normalized value). There each channel holds the operation of either code nearest the
//! drawn value and either code nearest the cleared one. Measured on the Radeon 780M (RADV
//! PHOENIX), whose pipelines into `RGB10A2Unorm` record the note: the drawn green 562.65 stores 562
//! (Metal 563) with and without the operation, and the cleared blue 613.8 stores 613 (Metal 614).

use super::{
    buffer_with, layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::VkFeature;
use metal2vulkan::error::{Capability, Feature, Needs, PrivateRenderField};
use metal2vulkan::metal::{BlendFactor, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, PrecisionNote, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan::vk;
use metal2vulkan::Budget;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, CommandQueue, Degradation, Device, Error, Library, LoadAction, PipelineOptions,
    Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions, StoreAction,
    TextureDescriptor,
};

/// The target's side.
const SIDE: u64 = 4;

/// The color each target is cleared to (the probes' own).
const CLEARED: [f64; 4] = [0.2, 0.4, 0.6, 0.8];

/// The color drawn: no channel's code is a tie in any format here.
const DRAWN: [f32; 4] = [0.25, 0.55, 0.75, 0.65];

/// The color drawn into the signed format, two channels negative.
const DRAWN_SIGNED: [f32; 4] = [-0.25, 0.55, -0.75, 0.65];

/// A normalized format's channels (bits, signed) in memory order with the color channel each
/// holds, and the texel's bytes.
type Layout = (&'static [(u32, bool, usize)], usize);

/// The formats drawn: each normalized one's [`Layout`]; `None` for the float format.
fn layout(pf: PixelFormat) -> Option<Layout> {
    Some(match pf {
        PixelFormat::RGBA8Unorm => (
            &[(8, false, 0), (8, false, 1), (8, false, 2), (8, false, 3)],
            4,
        ),
        PixelFormat::BGRA8Unorm => (
            &[(8, false, 2), (8, false, 1), (8, false, 0), (8, false, 3)],
            4,
        ),
        PixelFormat::RGBA8Snorm => (&[(8, true, 0), (8, true, 1), (8, true, 2), (8, true, 3)], 4),
        PixelFormat::RGB10A2Unorm => (
            &[
                (10, false, 0),
                (10, false, 1),
                (10, false, 2),
                (2, false, 3),
            ],
            4,
        ),
        PixelFormat::R16Unorm => (&[(16, false, 0)], 2),
        PixelFormat::RGBA16Snorm => (
            &[(16, true, 0), (16, true, 1), (16, true, 2), (16, true, 3)],
            8,
        ),
        _ => return None,
    })
}

/// The formats the lane draws.
const FORMATS: [PixelFormat; 7] = [
    PixelFormat::RGBA8Unorm,
    PixelFormat::BGRA8Unorm,
    PixelFormat::RGBA8Snorm,
    PixelFormat::RGB10A2Unorm,
    PixelFormat::R16Unorm,
    PixelFormat::RGBA16Snorm,
    PixelFormat::RGBA16Float,
];

/// Whether `pf` holds signed channels, drawn [`DRAWN_SIGNED`].
fn signed(pf: PixelFormat) -> bool {
    matches!(pf, PixelFormat::RGBA8Snorm | PixelFormat::RGBA16Snorm)
}

/// A texel's bytes in `pf`.
fn texel_bytes(pf: PixelFormat) -> usize {
    match pf {
        PixelFormat::RGBA16Float | PixelFormat::RGBA16Snorm => 8,
        PixelFormat::R16Unorm => 2,
        _ => 4,
    }
}

/// Metal's code of `x` in a channel of `bits` (two's complement where `signed`): the product with
/// the largest code rounded to nearest.
fn code(x: f64, bits: u32, signed: bool) -> u32 {
    let mask = (1u64 << bits) as u32 - 1;
    match signed {
        true => {
            let max = f64::from((1u32 << (bits - 1)) - 1);
            ((x.clamp(-1.0, 1.0) * max).round() as i32) as u32 & mask
        }
        false => (x.clamp(0.0, 1.0) * f64::from(mask)).round() as u32,
    }
}

/// Metal's operation `raw` (its low four bits) of source code `s` and stored code `d` in a channel
/// of `bits`.
fn operate(raw: u64, s: u32, d: u32, bits: u32) -> u32 {
    let r = match raw & 15 {
        0 => 0,
        1 => !0,
        2 => s,
        3 => !s,
        4 => d,
        5 => !d,
        6 => s & d,
        7 => !(s & d),
        8 => s | d,
        9 => !(s | d),
        10 => s ^ d,
        11 => !(s ^ d),
        12 => s & !d,
        13 => !s & d,
        14 => s | !d,
        _ => !s | d,
    };
    r & ((1u64 << bits) as u32 - 1)
}

/// The codes a channel of `bits` (two's complement where `signed`) may hold for `x`: Metal's
/// ([`code`]), or, where the device converts it its own way (`either`), either code nearest the
/// scaled value.
fn codes(x: f64, bits: u32, signed: bool, either: bool) -> Vec<u32> {
    if !either {
        return vec![code(x, bits, signed)];
    }
    let mask = (1u64 << bits) as u32 - 1;
    let scaled = match signed {
        true => x.clamp(-1.0, 1.0) * f64::from((1u32 << (bits - 1)) - 1),
        false => x.clamp(0.0, 1.0) * f64::from(mask),
    };
    let mut both: Vec<u32> = [scaled.floor(), scaled.ceil()]
        .iter()
        .map(|&c| (c as i32) as u32 & mask)
        .collect();
    both.dedup();
    both
}

/// How an operation takes a channel's codes: Metal's, which widens a 10-bit format's codes to 16
/// bits, operates there and narrows the result back (probe `logic_operation`), or per channel, on
/// each channel's own codes, as Vulkan's `logicOp` does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    Metal,
    PerChannel,
}

/// The codes a channel of `bits` (two's complement where `signed`) may hold under operation `raw`
/// of the drawn value `x` over the cleared value `cleared`, by `rule`: each conversion Metal's, or,
/// where the device converts the drawn value (`drawn_either`, the result too where Metal narrows
/// one) or the clear (`cleared_either`) its own way, either code nearest.
fn results(
    raw: u64,
    (x, cleared): (f64, f64),
    (bits, signed): (u32, bool),
    rule: Rule,
    (drawn_either, cleared_either): (bool, bool),
) -> Vec<u32> {
    let stored = codes(cleared, bits, signed, cleared_either);
    let mut out = Vec::new();
    if rule == Rule::Metal && matches!(bits, 10 | 2) {
        let max = f64::from((1u32 << bits) - 1);
        for s in codes(x, 16, false, drawn_either) {
            for &d in &stored {
                let d16 = (f64::from(d) * 65535.0 / max).round() as u32;
                let r = f64::from(operate(raw, s, d16, 16)) * max / 65535.0;
                match drawn_either {
                    true => out.extend([r.floor() as u32, r.ceil() as u32]),
                    false => out.push(r.round() as u32),
                }
            }
        }
        return out;
    }
    for s in codes(x, bits, signed, drawn_either) {
        out.extend(stored.iter().map(|&d| operate(raw, s, d, bits)));
    }
    out
}

/// Whether `got` is a texel `rule` allows in normalized format `pf` under operation `raw`: each
/// channel's code one of [`results`]'s.
fn allowed(
    pf: PixelFormat,
    (raw, rule): (u64, Rule),
    drawn: [f32; 4],
    got: &[u8],
    either: (bool, bool),
) -> bool {
    let (channels, bytes) = layout(pf).expect("a normalized format");
    let mut word = 0u64;
    for (i, &b) in got.iter().take(bytes).enumerate() {
        word |= u64::from(b) << (8 * i);
    }
    let mut shift = 0;
    channels.iter().all(|&(bits, signed, channel)| {
        let held = ((word >> shift) & ((1u64 << bits) - 1)) as u32;
        shift += bits;
        let values = (f64::from(drawn[channel]), CLEARED[channel]);
        results(raw, values, (bits, signed), rule, either).contains(&held)
    })
}

/// The bytes `rule` stores in a texel of normalized format `pf` under operation `raw`, each
/// conversion Metal's.
fn expected(pf: PixelFormat, (raw, rule): (u64, Rule), drawn: [f32; 4]) -> Vec<u8> {
    let (channels, bytes) = layout(pf).expect("a normalized format");
    let mut word = 0u64;
    let mut shift = 0;
    for &(bits, signed, channel) in channels {
        let values = (f64::from(drawn[channel]), CLEARED[channel]);
        let code = results(raw, values, (bits, signed), rule, (false, false))[0];
        word |= u64::from(code) << shift;
        shift += bits;
    }
    word.to_le_bytes()[..bytes].to_vec()
}

struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

impl Lane {
    /// The lane's device, its `logicOp` withheld where `withheld`.
    fn new(messages: &Messages, cache: &ScratchCache, withheld: bool) -> Lane {
        let device = validated_device_with(TEST_DEVICE, messages, cache.path(), |o| {
            if withheld {
                o.withheld_features.insert(VkFeature::LogicOp);
            }
        })
        .unwrap();
        let bitcode = llvm_assemble(&engine_dir().join("a8r_targets.ll")).unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            library: device.new_library_with_air(&bitcode).unwrap(),
            device,
        }
    }

    fn function(&self, name: &str) -> StageSource {
        StageSource::Function(
            self.library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    }

    /// The fullscreen fill into `pf`, blended by one and one where `blended`, under the logic
    /// operation `raw` where one is given.
    fn pipeline(
        &self,
        pf: PixelFormat,
        blended: bool,
        raw: Option<u64>,
    ) -> Result<RenderPipelineState, Error> {
        self.device.new_render_pipeline_state(
            &self.descriptor(pf, blended, raw),
            &PipelineOptions::default(),
        )
    }

    /// Whether [`Lane::pipeline`]'s fragment stage records `PrecisionNote::DeviceColorConversion`,
    /// as the device's compile layer compiles it: the device converts the drawn color its own way.
    fn converts(&self, pf: PixelFormat, blended: bool, raw: u64) -> bool {
        let compiled = self
            .device
            .compile_device()
            .compile_render_pipeline(
                &self.descriptor(pf, blended, Some(raw)),
                &Budget::standard(),
            )
            .unwrap();
        compiled.stages.iter().any(|stage| {
            stage.stage == vk::ShaderStage::FRAGMENT
                && stage
                    .precision_notes
                    .contains(&PrecisionNote::DeviceColorConversion)
        })
    }

    /// [`Lane::pipeline`]'s descriptor.
    fn descriptor(
        &self,
        pf: PixelFormat,
        blended: bool,
        raw: Option<u64>,
    ) -> RenderPipelineDescriptor {
        let mut d = RenderPipelineDescriptor::new(self.function("a8r_vertex"));
        d.fragment = Some(self.function("a8r_fill"));
        let mut a = ColorAttachmentDescriptor::new(pf);
        if blended {
            a.blending_enabled = true;
            a.source_rgb_blend_factor = BlendFactor::One;
            a.destination_rgb_blend_factor = BlendFactor::One;
            a.source_alpha_blend_factor = BlendFactor::One;
            a.destination_alpha_blend_factor = BlendFactor::One;
        }
        d.color_attachments[0] = Some(a);
        if let Some(raw) = raw {
            d.private.logic_operation_enabled = true;
            d.private.logic_operation = raw;
        }
        d
    }

    /// One pass into a `pf` target cleared to [`CLEARED`], drawing `color` through `pipeline`
    /// where one is given: the first texel's bytes.
    fn draw(
        &self,
        pf: PixelFormat,
        pipeline: Option<&RenderPipelineState>,
        color: [f32; 4],
    ) -> Vec<u8> {
        self.draw_over(pf, pipeline, color, (CLEARED, false))
    }

    /// [`Lane::draw`] into a target cleared to `cleared`, beside a cleared `Depth32Float_Stencil8`
    /// depth and stencil attachment where `depth`.
    fn draw_over(
        &self,
        pf: PixelFormat,
        pipeline: Option<&RenderPipelineState>,
        color: [f32; 4],
        (cleared, depth): ([f64; 4], bool),
    ) -> Vec<u8> {
        let texel = texel_bytes(pf);
        let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::shared();
        let target = self.device.new_texture(&d).unwrap();
        let bytes: Vec<u8> = color.iter().flat_map(|v| v.to_le_bytes()).collect();
        let color = buffer_with(&self.device, &bytes).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let a = &mut pass.color_attachments[0];
        a.texture = Some(target.clone());
        a.load_action = LoadAction::Clear;
        a.clear_color = ClearColor::new(cleared[0], cleared[1], cleared[2], cleared[3]);
        a.store_action = StoreAction::Store;
        if depth {
            let mut d =
                TextureDescriptor::texture_2d(PixelFormat::Depth32FloatStencil8, SIDE, SIDE, false);
            d.usage = TextureUsage::RENDER_TARGET;
            d.resource_options = ResourceOptions::private();
            let texture = self.device.new_texture(&d).unwrap();
            pass.depth_attachment.texture = Some(texture.clone());
            pass.depth_attachment.load_action = LoadAction::Clear;
            pass.depth_attachment.store_action = StoreAction::DontCare;
            pass.stencil_attachment.texture = Some(texture);
            pass.stencil_attachment.load_action = LoadAction::Clear;
            pass.stencil_attachment.store_action = StoreAction::DontCare;
        }
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut e = command_buffer.render_command_encoder(&pass).unwrap();
        if let Some(pipeline) = pipeline {
            e.set_render_pipeline_state(pipeline).unwrap();
            e.set_fragment_buffer(Some(&color), 0, 0).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
        e.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let mut out = vec![0u8; texel];
        target
            .get_bytes(
                &mut out,
                SIDE * texel as u64,
                0,
                Region::new_2d(0, 0, 1, 1),
                0,
                0,
            )
            .unwrap();
        out
    }
}

/// Whether the lane's device orders framebuffer fetch, through the pixel interlock or
/// rasterization-order input attachments: without `logicOp`, the fragment function computes the
/// operation there.
fn orders_fetch(device: &Device) -> bool {
    let features = &device.enabled().features;
    features.contains(&VkFeature::FragmentShaderPixelInterlock)
        || features.contains(&VkFeature::RasterizationOrderColorAttachmentAccess)
}

/// The refusal of an operation other than copy and no-op on a device without `logicOp` that
/// orders fetch in neither form.
fn unordered() -> metal2vulkan::error::Unsupported {
    metal2vulkan::error::Unsupported::needs_any_of(
        Feature::PrivateRenderState(PrivateRenderField::LogicOperation),
        vec![
            vec![Capability::Feature(VkFeature::LogicOp)],
            vec![Capability::Feature(VkFeature::FragmentShaderPixelInterlock)],
            vec![Capability::Feature(
                VkFeature::RasterizationOrderColorAttachmentAccess,
            )],
        ],
    )
}

/// Every operation (raw 0 to 16, 16 taken as its low four bits) into each format of [`FORMATS`],
/// blended by one and one and not, on the device and, where it has `logicOp`, on the device with
/// the feature withheld: a normalized target stores Metal's operation of the codes, a float one
/// the draw without the operation (the module docs). Without `logicOp`, copy and no-op draw so;
/// every other operation is the fragment function's where the device orders fetch, and refused
/// naming the three features elsewhere. Vulkan's `logicOp` operates on `RGB10A2Unorm`'s own
/// codes, where Metal widens them to 16 bits: such a draw may store exactly the per-channel
/// rule's texel instead (a known gap), counted apart.
/// Where the device converts a drawn color (the pipeline's note) or the clears (their
/// announcement, taken from a pass of each normalized format's clear alone, before any pipeline
/// is made) its own way, that channel's code is either one nearest its value (the module docs).
#[test]
fn the_logic_operation_stores_metal_s_codes() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let lane = Lane::new(&messages, &cache, false);
    let has = lane.device.enabled().features.contains(&VkFeature::LogicOp);
    let mut summaries = vec![run(&lane, &messages)];
    drop(lane);
    if has {
        let withheld = Lane::new(&messages, &cache, true);
        summaries.push(run(&withheld, &messages));
        drop(withheld);
    }
    eprintln!("{}", summaries.join("\n"));
    layer_reported_nothing_at_all(&messages);
}

/// [`the_logic_operation_stores_metal_s_codes`] on `lane`'s device: its summary line.
fn run(lane: &Lane, messages: &Messages) -> String {
    let name = lane.device.name().to_owned();
    let has = lane.device.enabled().features.contains(&VkFeature::LogicOp);
    let shader = !has && orders_fetch(&lane.device);
    messages.take_notices();
    for pf in FORMATS.into_iter().filter(|&pf| layout(pf).is_some()) {
        lane.draw(pf, None, DRAWN);
    }
    messages.take_notices();
    let cleared_either = messages
        .taken_kinds()
        .contains(&Degradation::DeviceColorConversion);
    let (mut wrong, mut checked, mut refused) = (Vec::new(), 0usize, 0usize);
    // The draws whose texel is not Metal's, each where the device converts its own way, and those
    // of Vulkan's operation on a 10-bit format's own codes.
    let (mut theirs, mut noted, mut per_channel) = (0usize, 0usize, 0usize);
    for pf in FORMATS {
        let drawn = match signed(pf) {
            true => DRAWN_SIGNED,
            false => DRAWN,
        };
        let float = layout(pf).is_none();
        for blended in [false, true] {
            // A float target's draw without the operation (a normalized one's blend is dropped,
            // and lavapipe blends no snorm format).
            let without = float.then(|| {
                let plain = lane.pipeline(pf, blended, None).unwrap();
                lane.draw(pf, Some(&plain), drawn)
            });
            for raw in 0..=16u64 {
                let blendless = matches!(raw & 15, 2 | 4);
                let not_implemented = |u: &metal2vulkan::error::Unsupported| {
                    assert_eq!(
                        (&u.feature, &u.needs),
                        (
                            &Feature::PrivateRenderState(PrivateRenderField::LogicOperation),
                            &Needs::NotImplemented
                        ),
                        "{name}: {pf:?} {raw}"
                    );
                };
                let pipeline = match lane.pipeline(pf, blended, Some(raw)) {
                    Ok(p) => p,
                    // A float attachment that blends under an operation Vulkan's turns off.
                    Err(Error::Unsupported(u)) if has && float && blended && !blendless => {
                        not_implemented(&u);
                        refused += 1;
                        continue;
                    }
                    Err(Error::Unsupported(u)) if !has && !shader && !blendless => {
                        assert_eq!(u, unordered(), "{name}: {pf:?} {raw}");
                        refused += 1;
                        continue;
                    }
                    Err(e) => panic!("{name}: {pf:?} blended {blended} operation {raw}: {e}"),
                };
                let got = lane.draw(pf, Some(&pipeline), drawn);
                let want = match &without {
                    Some(without) => without.clone(),
                    None => expected(pf, (raw, Rule::Metal), drawn),
                };
                checked += 1;
                if got == want {
                    continue;
                }
                // Vulkan's operation on a 10-bit format's own codes, where Metal's widens them
                // (a known gap, `docs/FAST-PATHS.md`): exactly the per-channel rule's texel.
                let rule = match has {
                    true => Rule::PerChannel,
                    false => Rule::Metal,
                };
                if !float && has && got == expected(pf, (raw, rule), drawn) {
                    per_channel += 1;
                    continue;
                }
                let drawn_either = !float && lane.converts(pf, blended, raw);
                noted += usize::from(drawn_either);
                let either = (drawn_either, cleared_either);
                if !float && allowed(pf, (raw, rule), drawn, &got, either) {
                    theirs += 1;
                    continue;
                }
                wrong.push(format!(
                    "{pf:?} blended {blended} operation {raw}: {got:02x?}, Metal {want:02x?} \
                     (drawn color noted converted {drawn_either}, clears announced \
                     {cleared_either})"
                ));
            }
        }
    }
    let form = match (has, shader) {
        (true, _) => "Vulkan's logicOp",
        (false, true) => "the fragment function's operation",
        (false, false) => "no operation but copy and no-op",
    };
    assert!(
        wrong.is_empty(),
        "{name} ({form}): {} of {checked} draws differ from Metal's:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    format!(
        "{name} ({form}): {checked} logic-operated draws store Metal's texels or the device's \
         conversion's ({theirs} the device's: {noted} of them through a pipeline noted \
         DeviceColorConversion, clears announced {cleared_either}; {per_channel} Vulkan's \
         operation on a 10-bit format's own codes), {refused} refused"
    )
}

/// The consumer's shape: an unblended `RGBA8Unorm` attachment written whole beside a
/// `Depth32Float_Stencil8` depth and stencil attachment, xor (raw 10) of the fragment's (1, 1, 0,
/// 0) over the stored (64, 128, 191, 255), which Metal stores (191, 127, 191, 255) (the probe on
/// an Apple M3 Max under macOS 27.0.1): so on the device, and where it has `logicOp`, with the
/// feature withheld; refused naming the features only where the device orders fetch in neither
/// form.
#[test]
fn the_logic_operation_beside_a_depth_and_stencil_attachment_is_metal_s() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let stored = [64.0 / 255.0, 128.0 / 255.0, 191.0 / 255.0, 1.0];
    let mut seen = Vec::new();
    for withheld in [false, true] {
        // Withholding a feature the device lacks changes nothing.
        if withheld && !seen.first().is_some_and(|(_, had): &(String, bool)| *had) {
            break;
        }
        let lane = Lane::new(&messages, &cache, withheld);
        let name = lane.device.name().to_owned();
        let has = lane.device.enabled().features.contains(&VkFeature::LogicOp);
        let mut d = lane.descriptor(PixelFormat::RGBA8Unorm, false, Some(10));
        d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
        d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
        match lane
            .device
            .new_render_pipeline_state(&d, &PipelineOptions::default())
        {
            Ok(pipeline) => {
                let got = lane.draw_over(
                    PixelFormat::RGBA8Unorm,
                    Some(&pipeline),
                    [1.0, 1.0, 0.0, 0.0],
                    (stored, true),
                );
                assert_eq!(got, [191, 127, 191, 255], "{name} (logicOp {has})");
            }
            Err(Error::Unsupported(u)) if !has && !orders_fetch(&lane.device) => {
                assert_eq!(u, unordered(), "{name}");
            }
            Err(e) => panic!("{name} (logicOp {has}): {e}"),
        }
        seen.push((name, has));
        drop(lane);
    }
    eprintln!("the consumer's xor stores Metal's texel on {seen:?} (name, logicOp)");
    layer_reported_nothing_at_all(&messages);
}
