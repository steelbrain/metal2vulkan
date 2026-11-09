//! The private `fragmentDepthCompareClampMask` of OpenGL on Metal
//! (`PrivateRenderState::fragment_depth_compare_clamp_mask`), on the device the cfg selects, under
//! the validation layer, over the owned fixture `pdc_compare_clamp`.
//!
//! Metal's rule, measured with our own probes `spi` to `spi3` on an Apple M3 Max under macOS
//! 27.0.1 (a 4 x 4 `Depth32Float` texture cleared to one depth, a fullscreen triangle comparing it
//! at the centre): bit `s` clamps the reference of a fragment function's comparison through
//! `[[sampler(s)]]` to `[0, 1]`, a NaN taken as 0, whichever texture slot it compares, for
//! `sample_compare` and `gather_compare` alike; bit 31 clamps one through any constexpr sampler;
//! the comparison is then `reference OP texel`, false for a NaN reference left unclamped. The
//! rule matched all 432 probed `sample_compare` rows (depths 0, 0.5 and 1, four functions, masks 0
//! to 3, nine references among them NaN and the infinities), and the slot, constexpr and gather
//! rows. Each draw here is checked against it. The comparisons are the device's, which announce
//! `restricted-depth-range` only on a device whose depth images take no depth outside `[0, 1]`
//! (`degrade_notices::clamps_depth_uploads`: without `VK_EXT_depth_range_unrestricted`).

use super::degrade_notices::{clamps_depth_uploads, counts};
use super::{
    buffer_with, layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, CommandQueue, Degradation, Device, Library, LoadAction, PipelineOptions, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState, StoreAction, Texture,
    TextureDescriptor,
};

/// The textures' side.
const SIDE: u64 = 4;

/// The depths the compared texture is cleared to.
const DEPTHS: [f32; 3] = [0.0, 0.5, 1.0];

/// The references compared: inside `[0, 1]`, past it, NaN and the infinities.
const REFERENCES: [f32; 8] = [
    f32::NAN,
    -0.5,
    1.5,
    0.0,
    0.25,
    1.0,
    f32::NEG_INFINITY,
    f32::INFINITY,
];

/// Bit 31: a constexpr sampler's.
const CONSTEXPR: u32 = 1 << 31;

/// Metal's comparison of `reference` with `texel` under `function`, the reference clamped where
/// `clamped` (the module docs).
fn compares(function: CompareFunction, reference: f32, texel: f32, clamped: bool) -> bool {
    let r = match (clamped, reference.is_nan()) {
        (true, true) => 0.0,
        (true, false) => reference.clamp(0.0, 1.0),
        (false, _) => reference,
    };
    match function {
        CompareFunction::Less => r < texel,
        CompareFunction::Equal => r == texel,
        CompareFunction::GreaterEqual => r >= texel,
        _ => unreachable!("a function the lane does not draw"),
    }
}

struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

impl Lane {
    fn new(messages: &Messages, cache: &ScratchCache) -> Lane {
        let device = validated_device_with(TEST_DEVICE, messages, cache.path(), |_| {}).unwrap();
        let bitcode = llvm_assemble(&engine_dir().join("pdc_compare_clamp.ll")).unwrap();
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

    fn pipeline(&self, fragment: &str, mask: u32) -> RenderPipelineState {
        let mut d = RenderPipelineDescriptor::new(self.function("pdc_vertex"));
        d.fragment = Some(self.function(fragment));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        d.private.fragment_depth_compare_clamp_mask = mask;
        self.device
            .new_render_pipeline_state(&d, &PipelineOptions::default())
            .unwrap_or_else(|e| panic!("{fragment} mask {mask:#x}: {e}"))
    }

    /// A `Depth32Float` texture cleared to `depth`.
    fn depth(&self, depth: f32) -> Texture {
        let mut d = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, SIDE, SIDE, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        let texture = self.device.new_texture(&d).unwrap();
        let mut pass = RenderPassDescriptor::new();
        pass.depth_attachment.texture = Some(texture.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.store_action = StoreAction::Store;
        pass.depth_attachment.clear_depth = depth.into();
        let command_buffer = self.queue.command_buffer().unwrap();
        command_buffer
            .render_command_encoder(&pass)
            .unwrap()
            .end_encoding()
            .unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        texture
    }

    fn sampler(&self, function: CompareFunction) -> SamplerState {
        let mut d = SamplerDescriptor::new();
        d.compare_function = function;
        self.device.new_sampler_state(&d).unwrap()
    }

    /// One draw through `pipeline` comparing `depth` (at texture slot `texture`, through
    /// `sampler` at slot 2 where one is given) against the references `x` (`sample_compare`) and
    /// `y` (`gather_compare`): the red and green bytes of the target's first texel.
    fn draw(
        &self,
        pipeline: &RenderPipelineState,
        (depth, texture): (&Texture, u32),
        sampler: Option<&SamplerState>,
        (x, y): (f32, f32),
    ) -> [u8; 2] {
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::shared();
        let target = self.device.new_texture(&d).unwrap();
        let refs: Vec<u8> = [x, y, 0.0, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let refs = buffer_with(&self.device, &refs).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let a = &mut pass.color_attachments[0];
        a.texture = Some(target.clone());
        a.load_action = LoadAction::Clear;
        a.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        a.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut e = command_buffer.render_command_encoder(&pass).unwrap();
        e.set_render_pipeline_state(pipeline).unwrap();
        e.set_fragment_buffer(Some(&refs), 0, 0).unwrap();
        e.set_fragment_texture(Some(depth), texture).unwrap();
        if let Some(s) = sampler {
            e.set_fragment_sampler_state(Some(s), 2).unwrap();
        }
        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        e.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let mut out = [0u8; 4];
        target
            .get_bytes(&mut out, SIDE * 4, 0, Region::new_2d(0, 0, 1, 1), 0, 0)
            .unwrap();
        [out[0], out[1]]
    }
}

/// Every reference of [`REFERENCES`] compared with every depth of [`DEPTHS`], through
/// `[[sampler(2)]]` under `Less`, `Equal` and `GreaterEqual`, and through a constexpr `Less`
/// sampler, under masks that clamp the sampler's comparisons and masks that clamp others: each
/// `sample_compare` and `gather_compare` answers as Metal's rule says (the module docs), and
/// `restricted-depth-range` is announced exactly where the device's depth images take no value
/// outside `[0, 1]`.
#[test]
fn the_depth_compare_clamp_mask_clamps_metal_s_references() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let lane = Lane::new(&messages, &cache);
    let name = lane.device.name().to_owned();
    let slot = 1u32 << 2;
    let (mut wrong, mut checked) = (Vec::new(), 0usize);
    let functions = [
        CompareFunction::Less,
        CompareFunction::Equal,
        CompareFunction::GreaterEqual,
    ];
    let samplers = functions.map(|f| (f, lane.sampler(f)));
    // (fragment, the texture's slot, the sampler's bit, masks).
    let cases = [
        ("pdc_slot", 1, slot, [0, slot, 1 << 1, u32::MAX]),
        (
            "pdc_constexpr",
            0,
            CONSTEXPR,
            [0, CONSTEXPR, 0x7fff_ffff, u32::MAX],
        ),
    ];
    for (fragment, texture, bit, masks) in cases {
        for mask in masks {
            let pipeline = lane.pipeline(fragment, mask);
            let clamped = mask & bit != 0;
            for depth in DEPTHS {
                let held = lane.depth(depth);
                let runs: Vec<(CompareFunction, Option<&SamplerState>)> = match bit {
                    CONSTEXPR => vec![(CompareFunction::Less, None)],
                    _ => samplers.iter().map(|(f, s)| (*f, Some(s))).collect(),
                };
                for (function, sampler) in runs {
                    for (i, &x) in REFERENCES.iter().enumerate() {
                        let y = REFERENCES[(i + 3) % REFERENCES.len()];
                        let got = lane.draw(&pipeline, (&held, texture), sampler, (x, y));
                        let byte = |r| match compares(function, r, depth, clamped) {
                            true => 0xff,
                            false => 0,
                        };
                        let want = [byte(x), byte(y)];
                        checked += 1;
                        if got != want {
                            wrong.push(format!(
                                "{fragment} mask {mask:#x} {function:?} depth {depth} \
                                 references ({x}, {y}): {got:02x?}, Metal {want:02x?}"
                            ));
                        }
                    }
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{name}: {} of {checked} comparisons differ from Metal's:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    eprintln!("{name}: {checked} masked depth comparisons answer as Metal's");
    let restricted =
        counts(&messages.take_notices()).contains_key(&Degradation::RestrictedDepthRange);
    assert_eq!(
        restricted,
        clamps_depth_uploads(&lane.device),
        "{name}: restricted-depth-range announced"
    );
    drop(lane);
    layer_reported_nothing_at_all(&messages);
}
