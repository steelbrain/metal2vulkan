//! Layered and amplified draws in a rate-mapped pass on the device the cfg selects, under the
//! validation layer. The owned fixture `cfb_rate_layers` draws a fullscreen triangle whose
//! fragments write their position, 1, and the slice their vertex function chose, into an
//! `RGBA32Float` 2D array of two 64-pixel slices, cleared to -1, in a pass of both slices over a
//! map of a 64-pixel screen whose layer 0 is at quality 0.5 and layer 1 the identity (their
//! physical sizes 33 by 64 and 64 by 64: the two layers share one vertical axis), or whose layers
//! are both at 0.5 (33 by 33), as Metal probes drew them on the M3 Max and an M4, alike with and
//! without API validation:
//!
//! - Metal rasterizes each slice by the map's layer of that slice: slice 0 shades its 33 by 64
//!   physical pixels and slice 1 its 64 by 64, whether a vertex function chooses the slice, two
//!   amplifications are mapped to slices 0 and 1 in either order, an amplification is mapped past
//!   the pass (to slice 0), or a count of 1 is mapped to slice 1 under a pipeline of maximum
//!   count 2; under one of maximum count 1 the mapping moves nothing. Each shaded pixel's
//!   position is its physical centre.
//! - The runtime draws each slice by its layer's regions where it knows the slice: every draw
//!   over the map whose layers are alike, and over the other each amplification (drawn one at a
//!   time) and each draw its view mapping moves. A pipeline whose vertex function chooses its
//!   slice over the map whose layers differ is refused,
//!   `NotImplemented(RateMapRegions(LayeredPipeline))`, the slices unknown to the encoder.

use super::rate_maps::descriptor;
use super::{
    at_plan, layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{
    PixelFormat, PrimitiveTopologyClass, PrimitiveType, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, CommandBufferStatus, Device, Error, Library, LoadAction, PipelineOptions,
    RasterizationRateMap, Region, RegionRefusal, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, RuntimeGap, StoreAction, Texture, TextureDescriptor,
    VertexAmplificationViewMapping,
};

/// The screen's side, and the target's.
const SIDE: u64 = 64;

/// What a slice holds after a draw: the pixels shaded, the greatest `(x, y)` among them, and the
/// slice values their fragments wrote. `None` for a slice no fragment reached.
type Slice = Option<(usize, [u64; 2], Vec<u32>)>;

/// The functions of `cfb_rate_layers` on one device.
struct Fixture {
    device: Device,
    library: Library,
}

/// One draw, as the probes drew it: its vertex and fragment functions, the pipeline's
/// `maxVertexAmplificationCount`, its instances, and the amplification count and the slice
/// offsets of its view mappings (none set where `None`).
struct Draw {
    vertex: &'static str,
    fragment: &'static str,
    max: u32,
    instances: u32,
    amplification: Option<&'static [u32]>,
}

/// The vertex function that chooses its slice by instance, two instances.
const LAYERED: Draw = Draw {
    vertex: "cfb_rl_layer",
    fragment: "cfb_rl_position_layer",
    max: 1,
    instances: 2,
    amplification: None,
};

/// The amplified vertex function, its slices `offsets`, under a pipeline of maximum count 2.
const fn amplified(offsets: &'static [u32]) -> Draw {
    Draw {
        vertex: "cfb_rl_amp",
        fragment: "cfb_rl_position",
        max: 2,
        instances: 1,
        amplification: Some(offsets),
    }
}

impl Fixture {
    fn new(device: &Device) -> Fixture {
        let bitcode = llvm_assemble(&engine_dir().join("cfb_rate_layers.ll")).unwrap();
        Fixture {
            library: device.new_library_with_air(&bitcode).unwrap(),
            device: device.clone(),
        }
    }

    /// The map of the screen whose layers are at `qualities` on both axes.
    fn map(&self, qualities: &[f32]) -> RasterizationRateMap {
        let layers: Vec<(f32, f32)> = qualities.iter().map(|&q| (q, q)).collect();
        self.device
            .new_rasterization_rate_map(&descriptor(SIDE as u32, SIDE as u32, &layers))
            .unwrap()
    }

    /// The state of `draw`'s functions over one `RGBA32Float` attachment, or why the device
    /// makes none.
    fn state(&self, draw: &Draw) -> Result<RenderPipelineState, Error> {
        let function = |name: &str| {
            StageSource::Function(
                self.library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function(draw.vertex));
        descriptor.fragment = Some(function(draw.fragment));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
        descriptor.input_primitive_topology = PrimitiveTopologyClass::Triangle;
        descriptor.max_vertex_amplification_count = draw.max;
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
    }

    /// Run `draw` in a pass of both slices of a new target over `map`, and answer each slice, or
    /// the encoder's or the draw's refusal.
    fn run(&self, map: &RasterizationRateMap, draw: &Draw) -> Result<[Slice; 2], Error> {
        let state = self.state(draw)?;
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, SIDE, SIDE, false);
        descriptor.texture_type = TextureType::Type2DArray;
        descriptor.array_length = 2;
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::shared();
        let target = self.device.new_texture(&descriptor).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(target.clone());
        attachment.load_action = LoadAction::Clear;
        attachment.clear_color = ClearColor::new(-1.0, -1.0, -1.0, -1.0);
        attachment.store_action = StoreAction::Store;
        pass.render_target_array_length = 2;
        pass.rasterization_rate_map = Some(map.clone());
        let queue = self.device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        encoder.set_render_pipeline_state(&state)?;
        if let Some(offsets) = draw.amplification {
            let mappings: Vec<VertexAmplificationViewMapping> = offsets
                .iter()
                .map(|&slice| VertexAmplificationViewMapping::new(0, slice))
                .collect();
            encoder.set_vertex_amplification_count(offsets.len() as u64, Some(&mappings))?;
        }
        let drawn = encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, draw.instances, 0);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        drawn?;
        Ok([slice(&target, 0), slice(&target, 1)])
    }
}

/// What slice `index` of `target` holds (`Slice`); every shaded pixel's position is its centre.
fn slice(target: &Texture, index: u64) -> Slice {
    let mut bytes = vec![0u8; (SIDE * SIDE * 16) as usize];
    target
        .get_bytes(
            &mut bytes,
            SIDE * 16,
            0,
            Region::new_2d(0, 0, SIDE, SIDE),
            0,
            index,
        )
        .unwrap();
    let texel = |x: u64, y: u64| -> [f32; 4] {
        let at = ((y * SIDE + x) * 16) as usize;
        std::array::from_fn(|k| {
            f32::from_le_bytes(bytes[at + 4 * k..at + 4 * k + 4].try_into().unwrap())
        })
    };
    let mut count = 0;
    let mut most = [0, 0];
    let mut slices = Vec::new();
    for y in 0..SIDE {
        for x in 0..SIDE {
            let [px, py, shaded, chosen] = texel(x, y);
            if shaded != 1.0 {
                continue;
            }
            assert_eq!(
                [px, py],
                [x as f32 + 0.5, y as f32 + 0.5],
                "slice {index}'s pixel ({x}, {y})"
            );
            count += 1;
            most = [most[0].max(x), most[1].max(y)];
            if !slices.contains(&(chosen as u32)) {
                slices.push(chosen as u32);
            }
        }
    }
    (count > 0).then_some((count, most, slices))
}

/// Slice 0 by layer 0 of the map whose layers differ: 33 by 64 physical pixels.
fn half(chosen: u32) -> Slice {
    Some((33 * 64, [32, 63], vec![chosen]))
}

/// Probes 6 and 6b's draws: each slice shaded by its own layer, and the refusal (the module docs).
#[test]
fn each_slice_of_a_rate_mapped_pass_is_shaded_by_its_layer() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        let f = Fixture::new(&device);
        let differ = f.map(&[0.5, 1.0]);
        let alike = f.map(&[0.5, 0.5]);
        let whole = |chosen: u32| -> Slice { Some((64 * 64, [63, 63], vec![chosen])) };
        let quarter = |chosen: u32| -> Slice { Some((33 * 33, [32, 32], vec![chosen])) };
        let plain = Draw {
            vertex: "cfb_rl_full",
            fragment: "cfb_rl_position",
            max: 1,
            instances: 1,
            amplification: None,
        };
        let cases: [(&str, &RasterizationRateMap, Draw, [Slice; 2]); 9] = [
            ("flat_differ", &differ, plain, [half(0), None]),
            ("layer_same", &alike, LAYERED, [quarter(0), quarter(1)]),
            (
                "amp_differ",
                &differ,
                amplified(&[0, 1]),
                [half(0), whole(0)],
            ),
            (
                "amp_differ_swapped",
                &differ,
                amplified(&[1, 0]),
                [half(0), whole(0)],
            ),
            (
                "amp_same",
                &alike,
                amplified(&[0, 1]),
                [quarter(0), quarter(0)],
            ),
            (
                "map_amp_past",
                &differ,
                amplified(&[1, 2]),
                [half(0), whole(0)],
            ),
            (
                "map_one_to_1_max2",
                &differ,
                amplified(&[1]),
                [None, whole(0)],
            ),
            (
                "mapsame_one_to_1_max2",
                &alike,
                amplified(&[1]),
                [None, quarter(0)],
            ),
            (
                "amp_one_to_1",
                &differ,
                Draw {
                    max: 1,
                    ..amplified(&[1])
                },
                [half(0), None],
            ),
        ];
        for (case, map, draw, metal) in cases {
            match f.run(map, &draw) {
                Ok(slices) => assert_eq!(slices, metal, "{case} on {name}"),
                Err(Error::Unsupported(why)) => {
                    eprintln!("refused: {case} on {name}: {why:?}");
                }
                Err(error) => panic!("{case} on {name}: {error}"),
            }
        }
        // A vertex function that chooses its slice, over the map whose layers differ: Metal
        // shades slice 0 by layer 0 and slice 1 by layer 1, which the encoder cannot plan.
        let layered =
            Error::NotImplemented(RuntimeGap::RateMapRegions(RegionRefusal::LayeredPipeline));
        assert_eq!(
            f.run(&differ, &LAYERED),
            Err(layered),
            "layer_differ on {name}"
        );
    }
    layer_reported_nothing_at_all(&messages);
}
