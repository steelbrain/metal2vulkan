//! Which vertex's render target slice and viewport a primitive takes, on the device the cfg
//! selects, under the validation layer. The owned fixture `rt6_first_vertex` (and
//! `rt6_first_vertex_point`, the same with a point size) stores at every vertex its own slice
//! (`indices[vid] & 0xff`) and viewport (`indices[vid] >> 8`), so the vertices of one primitive
//! may store different ones, and its fragment function returns `0x10000 | viewport << 8 | slice`
//! as it reads them. Each draw renders into an 8 by 8 `R32Uint`
//! array of 4 slices through a pass of 4 layers cleared to [`FILL`], with 4 viewports, one per 4 by
//! 4 quadrant. For every slice and quadrant, the words other than [`FILL`] the draw left there are
//! compared with the ones Metal left for the same draws (a Metal probe of the fixture's functions
//! on the M3 Max under macOS 27.0.1), not how many texels hold each.
//!
//! Metal takes a primitive's slice and viewport from its first vertex, whatever its other vertices
//! store, in triangle lists in every order, triangle strips, lines, line strips and points; an index
//! at or past the count takes the first slice or viewport there, and the fragment function reads the
//! first vertex's index as it wrote it. Vulkan takes `Layer` and `ViewportIndex` only where every
//! vertex of a primitive stores the same. A device whose profile takes
//! `DriverDeviation::MetalFirstVertexIndices` (MoltenVK, which hands them to Metal) stores the
//! indices from the vertex function; every other device stores them through the synthesized
//! geometry stages (three, one per primitive class), and every draw asserts how many geometry stages
//! the pipeline it executed carries. A device with neither refuses the pipelines typed,
//! `LayeredRendering` needing `geometryShader`. The draws run again with `shaderOutputLayer` and
//! `shaderOutputViewportIndex` withheld.

use super::{buffer_with, read_buffer, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::VkFeature;
use metal2vulkan::error::{Capability, DriverDeviation, Feature, Needs};
use metal2vulkan::metal::{
    PixelFormat, PrimitiveTopologyClass, PrimitiveType, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk;
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, ClearColor, CommandBuffer, CommandBufferStatus, CommandQueue, Device, Error,
    ExecutedPipeline, LoadAction, Origin, PipelineOptions, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, Size, StoreAction, Texture, TextureDescriptor, Viewport,
};

/// The target's width and height.
const SIDE: u64 = 8;

/// A quadrant's width and height: one viewport each.
const HALF: u64 = SIDE / 2;

/// The target's slices, and the pass's layers.
const SLICES: u64 = 4;

/// The word the pass clears every texel to.
const FILL: u32 = 0xee;

/// How many geometry stages a pipeline that stores the indices through synthesized geometry
/// stages carries: the point, line and triangle classes'.
const CLASSES: usize = 3;

/// A full-screen triangle.
const TRIANGLE: &[[f32; 4]] = &[
    [-1.0, -1.0, 0.5, 1.0],
    [3.0, -1.0, 0.5, 1.0],
    [-1.0, 3.0, 0.5, 1.0],
];

/// A full-screen quad as a triangle strip.
const QUAD: &[[f32; 4]] = &[
    [-1.0, -1.0, 0.5, 1.0],
    [1.0, -1.0, 0.5, 1.0],
    [-1.0, 1.0, 0.5, 1.0],
    [1.0, 1.0, 0.5, 1.0],
];

/// A line across the viewport, a little above its middle.
const LINE: &[[f32; 4]] = &[[-1.0, 0.1, 0.5, 1.0], [1.0, 0.1, 0.5, 1.0]];

/// The same line in two segments.
const LINE_STRIP: &[[f32; 4]] = &[
    [-1.0, 0.1, 0.5, 1.0],
    [0.0, 0.1, 0.5, 1.0],
    [1.0, 0.1, 0.5, 1.0],
];

/// Two points of 3 pixels.
const POINTS: &[[f32; 4]] = &[[-0.5, -0.5, 0.5, 1.0], [0.5, 0.5, 0.5, 1.0]];

/// One draw of the probe: its vertices' positions and indices (`viewport << 8 | slice`), and the
/// words Metal left, as `s<slice>q<quadrant>=<word>,...` for every slice and quadrant holding
/// one, in slice then quadrant order.
struct Draw {
    name: &'static str,
    primitive: PrimitiveType,
    positions: &'static [[f32; 4]],
    indices: &'static [u32],
    metal: &'static [&'static str],
}

/// The probe's draws, and the words Metal left (the M3 Max under macOS 27.0.1): every primitive in
/// the slice and quadrant its first vertex names (the first slice or viewport where that is at or
/// past the count), its fragments reading the first vertex's indices as written. A strip's second
/// triangle and second segment take their first vertex, the strip's second.
const DRAWS: &[Draw] = &[
    Draw {
        name: "triangle 0x102 0x201 0x300",
        primitive: PrimitiveType::Triangle,
        positions: TRIANGLE,
        indices: &[0x102, 0x201, 0x300],
        metal: &["s2q1=0x10102"],
    },
    Draw {
        name: "triangle 0x201 0x300 0x102",
        primitive: PrimitiveType::Triangle,
        positions: TRIANGLE,
        indices: &[0x201, 0x300, 0x102],
        metal: &["s1q2=0x10201"],
    },
    Draw {
        name: "triangle 0x300 0x102 0x201",
        primitive: PrimitiveType::Triangle,
        positions: TRIANGLE,
        indices: &[0x300, 0x102, 0x201],
        metal: &["s0q3=0x10300"],
    },
    Draw {
        name: "triangle 0x101 0x707 0x707",
        primitive: PrimitiveType::Triangle,
        positions: TRIANGLE,
        indices: &[0x101, 0x707, 0x707],
        metal: &["s1q1=0x10101"],
    },
    Draw {
        name: "triangle 0x707 0x101 0x101",
        primitive: PrimitiveType::Triangle,
        positions: TRIANGLE,
        indices: &[0x707, 0x101, 0x101],
        metal: &["s0q0=0x10707"],
    },
    Draw {
        name: "triangle 0x003 0x200 0x200",
        primitive: PrimitiveType::Triangle,
        positions: TRIANGLE,
        indices: &[0x003, 0x200, 0x200],
        metal: &["s3q0=0x10003"],
    },
    Draw {
        name: "strip 0x000 0x101 0x202 0x303",
        primitive: PrimitiveType::TriangleStrip,
        positions: QUAD,
        indices: &[0x000, 0x101, 0x202, 0x303],
        metal: &["s0q0=0x10000", "s1q1=0x10101"],
    },
    Draw {
        name: "strip 0x303 0x202 0x101 0x000",
        primitive: PrimitiveType::TriangleStrip,
        positions: QUAD,
        indices: &[0x303, 0x202, 0x101, 0x000],
        metal: &["s2q2=0x10202", "s3q3=0x10303"],
    },
    Draw {
        name: "line 0x101 0x202",
        primitive: PrimitiveType::Line,
        positions: LINE,
        indices: &[0x101, 0x202],
        metal: &["s1q1=0x10101"],
    },
    Draw {
        name: "line 0x202 0x101",
        primitive: PrimitiveType::Line,
        positions: LINE,
        indices: &[0x202, 0x101],
        metal: &["s2q2=0x10202"],
    },
    Draw {
        name: "line strip 0x000 0x101 0x202",
        primitive: PrimitiveType::LineStrip,
        positions: LINE_STRIP,
        indices: &[0x000, 0x101, 0x202],
        metal: &["s0q0=0x10000", "s1q1=0x10101"],
    },
    Draw {
        name: "points 0x103 0x302",
        primitive: PrimitiveType::Point,
        positions: POINTS,
        indices: &[0x103, 0x302],
        metal: &["s2q3=0x10302", "s3q1=0x10103"],
    },
];

/// The device, its queue, and the fixture's pipeline for each primitive class.
struct Lane {
    device: Device,
    queue: CommandQueue,
    /// The point, line and triangle classes' pipelines.
    pipelines: [RenderPipelineState; 3],
    /// The geometry stages each draw's pipeline carries.
    stages: usize,
}

impl Lane {
    /// The lane on `device`, or the device's refusal of the fixture's pipelines.
    fn new(device: Device) -> Result<Lane, Error> {
        let ll = metal2vulkan_devtools::engine_dir().join("rt6_first_vertex.ll");
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
        let pipeline = |class: PrimitiveTopologyClass| {
            // Metal refuses a point size outside the point class.
            let vertex = match class {
                PrimitiveTopologyClass::Point => "rt6_first_vertex_point",
                _ => "rt6_first_vertex",
            };
            let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
            descriptor.fragment = Some(function("rt6_first_vertex_read"));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
            descriptor.input_primitive_topology = class;
            device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        };
        let takes = device
            .compile_device()
            .profile()
            .takes_driver_deviation(DriverDeviation::MetalFirstVertexIndices)
            && [
                VkFeature::ShaderOutputLayer,
                VkFeature::ShaderOutputViewportIndex,
            ]
            .iter()
            .all(|f| {
                device
                    .compile_device()
                    .profile()
                    .enabled()
                    .features
                    .contains(f)
            });
        Ok(Lane {
            pipelines: [
                pipeline(PrimitiveTopologyClass::Point)?,
                pipeline(PrimitiveTopologyClass::Line)?,
                pipeline(PrimitiveTopologyClass::Triangle)?,
            ],
            stages: if takes { 0 } else { CLASSES },
            queue: device.new_command_queue().unwrap(),
            device,
        })
    }

    /// Commit `command_buffer` and wait for it to complete.
    fn complete(command_buffer: &CommandBuffer) {
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }

    /// `draw` into a fresh target: the words it left, in the form of [`Draw::metal`].
    fn run(&self, draw: &Draw) -> Vec<String> {
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::R32Uint, SIDE, SIDE, false);
        descriptor.texture_type = TextureType::Type2DArray;
        descriptor.array_length = SLICES;
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::private();
        let target = self.device.new_texture(&descriptor).unwrap();
        let mut pass = RenderPassDescriptor::new();
        pass.render_target_array_length = SLICES;
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(f64::from(FILL), 0.0, 0.0, 1.0);
        color.store_action = StoreAction::Store;
        let pipeline = match draw.primitive {
            PrimitiveType::Point => &self.pipelines[0],
            PrimitiveType::Line | PrimitiveType::LineStrip => &self.pipelines[1],
            _ => &self.pipelines[2],
        };
        let positions: Vec<u8> = draw
            .positions
            .iter()
            .flatten()
            .flat_map(|c| c.to_le_bytes())
            .collect();
        let indices: Vec<u8> = draw.indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        let positions = buffer_with(&self.device, &positions).unwrap();
        let indices = buffer_with(&self.device, &indices).unwrap();
        let viewports: Vec<Viewport> = (0..4u64)
            .map(|q| {
                Viewport::new(
                    ((q & 1) * HALF) as f64,
                    ((q >> 1) * HALF) as f64,
                    HALF as f64,
                    HALF as f64,
                    0.0,
                    1.0,
                )
            })
            .collect();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(pipeline).unwrap();
        encoder.set_viewports(&viewports).unwrap();
        encoder.set_vertex_buffer(Some(&positions), 0, 0).unwrap();
        encoder.set_vertex_buffer(Some(&indices), 0, 1).unwrap();
        encoder
            .draw_primitives(draw.primitive, 0, draw.indices.len() as u32, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        Lane::complete(&command_buffer);
        let executions = command_buffer.executions();
        let ExecutedPipeline::Render(executed) = executions[0].pipeline() else {
            panic!("{}: the draw executed no render pipeline", draw.name);
        };
        let stages = executed
            .stages
            .iter()
            .filter(|s| s.stage == vk::ShaderStage::GEOMETRY)
            .count();
        assert_eq!(
            stages,
            self.stages,
            "{}: the geometry stages of the pipeline the draw executed on {}",
            draw.name,
            self.device.name()
        );
        drop((positions, indices));
        self.words(&target)
    }

    /// The words other than [`FILL`] each slice and quadrant of `target` holds, in the form of
    /// [`Draw::metal`].
    fn words(&self, target: &Texture) -> Vec<String> {
        let bytes = SIDE * SIDE * 4;
        let out = buffer_with(&self.device, &vec![0u8; (bytes * SLICES) as usize]).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for slice in 0..SLICES {
            blit.copy_from_texture_to_buffer(
                target,
                slice,
                0,
                Origin::new(0, 0, 0),
                Size::new(SIDE, SIDE, 1),
                &out,
                slice * bytes,
                SIDE * 4,
                bytes,
                BlitOption::None,
            )
            .unwrap();
        }
        blit.end_encoding().unwrap();
        Lane::complete(&command_buffer);
        let words: Vec<u32> = read_buffer(&out, 0, bytes * SLICES)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let mut cells = Vec::new();
        for slice in 0..SLICES {
            for quadrant in 0..4u64 {
                let mut held: Vec<u32> = Vec::new();
                for y in (quadrant >> 1) * HALF..(quadrant >> 1) * HALF + HALF {
                    for x in (quadrant & 1) * HALF..(quadrant & 1) * HALF + HALF {
                        let word = words[(slice * SIDE * SIDE + y * SIDE + x) as usize];
                        if word != FILL && !held.contains(&word) {
                            held.push(word);
                        }
                    }
                }
                if !held.is_empty() {
                    held.sort_unstable();
                    let held: Vec<String> = held.iter().map(|w| format!("{w:#x}")).collect();
                    cells.push(format!("s{slice}q{quadrant}={}", held.join(",")));
                }
            }
        }
        cells
    }
}

/// Every draw on the test device without `withheld` leaves Metal's words in every slice and
/// quadrant; a device that can store neither index from its first vertex refuses the pipelines,
/// needing `geometryShader`.
fn first_vertex(withheld: &[VkFeature]) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.withheld_features.extend(withheld.iter().copied())
    })
    .unwrap();
    let name = device.name().to_owned();
    let enabled = |f: VkFeature| {
        device
            .compile_device()
            .profile()
            .enabled()
            .features
            .contains(&f)
    };
    if !enabled(VkFeature::MultiViewport) {
        eprintln!("SKIPPED: {name} without {withheld:?} draws through one viewport");
        return;
    }
    let geometry = enabled(VkFeature::GeometryShader);
    let lane = match Lane::new(device) {
        Ok(lane) => lane,
        Err(Error::Unsupported(u)) if !geometry => {
            assert_eq!(u.feature, Feature::LayeredRendering, "{name}");
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![vec![Capability::Feature(VkFeature::GeometryShader)]]),
                "{name}"
            );
            eprintln!("refused: the first vertex's indices on {name} without {withheld:?}: {u:?}");
            return;
        }
        Err(other) => panic!("the first vertex's indices on {name}: {other}"),
    };
    for draw in DRAWS {
        assert_eq!(
            lane.run(draw),
            draw.metal,
            "{} on {name} without {withheld:?}",
            draw.name
        );
    }
    eprintln!(
        "the first vertex's indices on {name} without {withheld:?}: {} draws, {} geometry stages: \
         Metal's words",
        DRAWS.len(),
        lane.stages
    );
}

/// Each draw leaves Metal's words: the slice and viewport of every primitive its first vertex's.
#[test]
fn a_primitive_takes_its_first_vertex_s_slice_and_viewport_as_metal() {
    first_vertex(&[]);
}

/// The same draws on the device without the outputs, through the synthesized geometry stages
/// where the device has them.
#[test]
fn without_the_outputs_a_primitive_takes_its_first_vertex_s_slice_and_viewport_as_metal() {
    first_vertex(&[
        VkFeature::ShaderOutputLayer,
        VkFeature::ShaderOutputViewportIndex,
    ]);
}
