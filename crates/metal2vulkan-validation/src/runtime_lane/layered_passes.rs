//! Device tests of layered render passes on the device the cfg selects, under the validation layer.
//! A pass whose `renderTargetArrayLength` is 2 or more renders that many layers of every attachment
//! from its slice (a cube's face), loading, clearing and storing those alone: the owned fixture
//! `rt6_layered` draws the fullscreen triangle at depth 0.5 once per instance, instance `i` into
//! layer `layers[i]` of the pass, and its fragment function returns `(i + 1) << 8 | the layer it
//! reads`, counted from the pass's first, as the vertex function wrote it (a pass from slice 2
//! drawing layer 0 stores 0x200 there, the second instance's tag over layer 0). Every slice of an
//! `R32Uint` target holds 0xEE first; the pass clears its layers to 0x11 or loads them. Each slice
//! is compared with the texels Metal stored for the same functions and passes (a Metal probe on an
//! M4 with macOS 27.0.1, with and without API validation, every texel of a slice alike). A
//! `Depth32Float` array beside the color one, each slice cleared to 0.75 by a pass of its own
//! first, is cleared to 1.0 and tested `Less` with writes.
//!
//! An index at or past the length renders the pass's first layer, and a fragment function reads the
//! index as written in a pass of any length, a pass of length 1 included (a Metal probe on an M4),
//! and a pass of length 0, Metal's default, too: there plain Metal draws the slice-choosing
//! function as at a length of 1, where its debug layer stops the draw (a Metal probe on an M4;
//! another probe, an M3 Max and an M4). What Metal's validation stops is refused typed: an
//! attachment without the pass's layers (`RenderPassProblem::ArrayLengthPastSlices`).
//!
//! The passes run again with `shaderOutputLayer` withheld, where the synthesized geometry stages
//! store `Layer` from the slice the vertex function selects, and every draw asserts how many
//! geometry stages the pipeline it executed carries (three in a pass of two layers or more, none
//! in a pass of one layer or none). `rt6_layered_weights` reads the primitive ID and the
//! barycentric weights beside the slice, on the device as made and with `fragmentShaderBarycentric`
//! withheld too, where one set of stages serves all three (a Metal probe on an M3 Max with macOS
//! 27.0.1 for its texels).

use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device,
    validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{
    BarrierScope, CompareFunction, PixelFormat, PrimitiveType, RenderStages, StencilOperation,
    TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk;
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    Attachment, BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus, CommandQueue,
    CpuCacheMode, DepthStencilState, Device, Error, ExecutedPipeline, HazardTrackingMode, Invalid,
    LoadAction, MultisampleDepthResolveFilter, MultisampleStencilResolveFilter, Origin,
    PipelineOptions, Problem, RenderPassDescriptor, RenderPassProblem, RenderPipelineState,
    ResolveTarget, ResourceOptions, Size, StorageMode, StoreAction, Texture, TextureDescriptor,
    VertexAmplificationViewMapping,
};

/// The targets' width and height.
const SIDE: u64 = 4;

/// The word every color slice holds before a pass.
const FILLED: u32 = 0xee;

/// The first color attachment's clear, and the second's.
const CLEARED: [u32; 2] = [0x11, 0x22];

/// The depth every slice of the depth array holds before a pass, and the pass's clear.
const DEPTHS: [f32; 2] = [0.75, 1.0];

/// What a probe case's color attachment is.
#[derive(Clone, Copy)]
enum Kind {
    /// A 2D texture.
    Plain,
    /// A 2D array of this many slices.
    Array(u64),
    /// A cube, its six faces.
    Cube,
}

impl Kind {
    /// Its slices.
    fn slices(self) -> u64 {
        match self {
            Kind::Plain => 1,
            Kind::Array(slices) => slices,
            Kind::Cube => 6,
        }
    }
}

/// One pass, as a Metal probe on an M4 drew it.
struct Case {
    name: &'static str,
    kind: Kind,
    /// The color attachment's slice (a cube's face).
    slice: u64,
    length: u64,
    load: LoadAction,
    /// The layer of the pass, counted from its first, each instance renders into.
    layers: &'static [u32],
    /// A depth array beside the color one.
    depth: bool,
    /// A second color attachment, a 2D array of this many slices.
    second: Option<u64>,
}

/// Probe 12c's case of a 2D array of 4 slices, from slice 0, of length 2, cleared, drawing into
/// slices 0 and 1.
const PLAIN: Case = Case {
    name: "",
    kind: Kind::Array(4),
    slice: 0,
    length: 2,
    load: LoadAction::Clear,
    layers: &[0, 1],
    depth: false,
    second: None,
};

/// What a pass stored: each slice's word of the color attachment, of the second one, and each
/// slice's depth (every texel of a slice alike, checked).
#[derive(Debug, PartialEq)]
struct Stored {
    color: Vec<u32>,
    second: Vec<u32>,
    depth: Vec<f32>,
}

/// How many geometry stages a pipeline that stores the slice, or reads the weights, through
/// synthesized geometry stages carries: the point, line and triangle classes'.
const CLASSES: usize = 3;

/// The fixture's fragment function a lane draws with.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fragment {
    /// `rt6_layered_fragment`: the tag and the slice.
    Slice,
    /// `rt6_layered_weights`: the tag and the slice, the primitive ID, and whether every
    /// barycentric weight lies in [0, 1].
    Weights,
}

/// The device, its queue, and the fixture's pipelines.
struct Lane {
    device: Device,
    queue: CommandQueue,
    /// The fixture's pipeline drawing one `R32Uint` color attachment, and with a second one or a
    /// `Depth32Float` attachment.
    plain: RenderPipelineState,
    two_colors: RenderPipelineState,
    with_depth: RenderPipelineState,
    less: DepthStencilState,
    /// The device stores the slice through the synthesized geometry stages (it has no
    /// `shaderOutputLayer`): a pass of two layers or more runs them.
    slice_stages: bool,
    /// The fragment function reads the weights as varyings the synthesized geometry stages write
    /// (the device has no `fragmentShaderBarycentric`): every draw runs them.
    weight_stages: bool,
}

/// The lane on the test device without `withheld`, drawing with `fragment`; `Err` with the
/// device's name and its refusal of the fixture's pipeline.
fn lane_without(
    messages: &Messages,
    cache: &ScratchCache,
    withheld: &[VkFeature],
    fragment: Fragment,
) -> Result<Lane, (String, Box<Error>)> {
    let device = validated_device_with(TEST_DEVICE, messages, cache.path(), |options| {
        options.withheld_features.extend(withheld.iter().copied())
    })
    .unwrap();
    let name = device.name().to_owned();
    Lane::new(device, fragment).map_err(|e| (name, Box::new(e)))
}

impl Lane {
    /// The lane on `device` drawing with `fragment`; `Err` with the device's refusal of the
    /// fixture's pipeline.
    fn new(device: Device, fragment: Fragment) -> Result<Lane, Error> {
        let ll = metal2vulkan_devtools::engine_dir().join("rt6_layered.ll");
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
        let pipeline = |second: bool, depth: bool| {
            let mut descriptor = RenderPipelineDescriptor::new(function("rt6_layered_vertex"));
            descriptor.fragment = Some(function(match fragment {
                Fragment::Slice => "rt6_layered_fragment",
                Fragment::Weights => "rt6_layered_weights",
            }));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
            if second {
                descriptor.color_attachments[1] =
                    Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
            }
            if depth {
                descriptor.depth_format = Some(PixelFormat::Depth32Float);
            }
            device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        };
        let mut less = DepthStencilDescriptor::default();
        less.depth_compare_function = CompareFunction::Less;
        less.depth_write_enabled = true;
        let enabled = |feature| {
            device
                .compile_device()
                .profile()
                .enabled()
                .features
                .contains(&feature)
        };
        let slice_stages = !enabled(VkFeature::ShaderOutputLayer);
        let weight_stages =
            fragment == Fragment::Weights && !enabled(VkFeature::FragmentShaderBarycentric);
        Ok(Lane {
            slice_stages,
            weight_stages,
            plain: pipeline(false, false)?,
            two_colors: pipeline(true, false)?,
            with_depth: pipeline(false, true)?,
            less: device.new_depth_stencil_state(&less).unwrap(),
            queue: device.new_command_queue().unwrap(),
            device,
        })
    }

    /// How many geometry stages the pipeline a draw in a pass of `length` layers executes
    /// carries: a pass of one layer or none runs the variant that stores no slice
    /// (`VariantKey::dropped_vertex_layer`).
    fn geometry_stages(&self, length: u64) -> usize {
        match self.weight_stages || (self.slice_stages && length > 1) {
            true => CLASSES,
            false => 0,
        }
    }

    /// A private render target of `pf` and `kind`.
    fn texture(&self, pf: PixelFormat, kind: Kind) -> Texture {
        let mut descriptor = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
        match kind {
            Kind::Plain => {}
            Kind::Array(slices) => {
                descriptor.texture_type = TextureType::Type2DArray;
                descriptor.array_length = slices;
            }
            Kind::Cube => descriptor.texture_type = TextureType::Cube,
        }
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::private();
        self.device.new_texture(&descriptor).unwrap()
    }

    /// Commit `command_buffer` and wait for it to complete.
    fn complete(command_buffer: &CommandBuffer) {
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }

    /// Fill every slice of the color texture `texture` with [`FILLED`].
    fn fill(&self, texture: &Texture, slices: u64) {
        let words: Vec<u8> = (0..SIDE * SIDE)
            .flat_map(|_| FILLED.to_le_bytes())
            .collect();
        let source = buffer_with(&self.device, &words).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for slice in 0..slices {
            blit.copy_from_buffer_to_texture(
                &source,
                0,
                SIDE * 4,
                SIDE * SIDE * 4,
                Size::new(SIDE, SIDE, 1),
                texture,
                slice,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
            .unwrap();
        }
        blit.end_encoding().unwrap();
        Lane::complete(&command_buffer);
    }

    /// Clear every slice of the depth texture `texture` to the first of [`DEPTHS`], a pass each.
    fn fill_depth(&self, texture: &Texture, slices: u64) {
        let command_buffer = self.queue.command_buffer().unwrap();
        for slice in 0..slices {
            let mut pass = RenderPassDescriptor::new();
            let depth = &mut pass.depth_attachment;
            depth.texture = Some(texture.clone());
            depth.slice = slice;
            depth.load_action = LoadAction::Clear;
            depth.clear_depth = f64::from(DEPTHS[0]);
            depth.store_action = StoreAction::Store;
            let encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.end_encoding().unwrap();
        }
        Lane::complete(&command_buffer);
    }

    /// Each slice's first word of `texture`, read by a blit, every texel of a slice alike.
    fn slices(&self, texture: &Texture, slices: u64) -> Vec<u32> {
        let bytes = SIDE * SIDE * 4;
        let out = buffer_with(&self.device, &vec![0u8; (bytes * slices) as usize]).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for slice in 0..slices {
            blit.copy_from_texture_to_buffer(
                texture,
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
        let words: Vec<u32> = read_buffer(&out, 0, bytes * slices)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        words
            .as_chunks::<{ (SIDE * SIDE) as usize }>()
            .0
            .iter()
            .map(|slice| {
                assert!(slice.iter().all(|&w| w == slice[0]), "{slice:x?}");
                slice[0]
            })
            .collect()
    }

    /// The pass of `case`, encoded and run; the encoder's or the draw's refusal.
    fn try_run(&self, case: &Case) -> Result<Stored, Error> {
        let slices = case.kind.slices();
        let color = self.texture(PixelFormat::R32Uint, case.kind);
        self.fill(&color, slices);
        let second = case.second.map(|slices| {
            (
                self.texture(PixelFormat::R32Uint, Kind::Array(slices)),
                slices,
            )
        });
        if let Some((second, slices)) = &second {
            self.fill(second, *slices);
        }
        let depth = case
            .depth
            .then(|| self.texture(PixelFormat::Depth32Float, case.kind));
        if let Some(depth) = &depth {
            self.fill_depth(depth, slices);
        }
        let mut pass = RenderPassDescriptor::new();
        pass.render_target_array_length = case.length;
        let target = &mut pass.color_attachments[0];
        target.texture = Some(color.clone());
        target.slice = case.slice;
        target.load_action = case.load;
        target.clear_color = ClearColor::new(f64::from(CLEARED[0]), 0.0, 0.0, 1.0);
        target.store_action = StoreAction::Store;
        if let Some((second, _)) = &second {
            let target = &mut pass.color_attachments[1];
            target.texture = Some(second.clone());
            target.load_action = LoadAction::Clear;
            target.clear_color = ClearColor::new(f64::from(CLEARED[1]), 0.0, 0.0, 1.0);
            target.store_action = StoreAction::Store;
        }
        if let Some(depth) = &depth {
            let target = &mut pass.depth_attachment;
            target.texture = Some(depth.clone());
            target.slice = case.slice;
            target.load_action = match case.load {
                LoadAction::Clear => LoadAction::Clear,
                _ => LoadAction::Load,
            };
            target.clear_depth = f64::from(DEPTHS[1]);
            target.store_action = StoreAction::Store;
        }
        let pipeline = match (&second, &depth) {
            (Some(_), _) => &self.two_colors,
            (None, Some(_)) => &self.with_depth,
            (None, None) => &self.plain,
        };
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        let mut held: Vec<Buffer> = Vec::new();
        if !case.layers.is_empty() {
            encoder.set_render_pipeline_state(pipeline)?;
            if depth.is_some() {
                encoder.set_depth_stencil_state(&self.less)?;
            }
            let words: Vec<u8> = case.layers.iter().flat_map(|l| l.to_le_bytes()).collect();
            let layers = buffer_with(&self.device, &words).unwrap();
            encoder.set_vertex_buffer(Some(&layers), 0, 0)?;
            encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, case.layers.len() as u32, 0)?;
            held.push(layers);
        }
        encoder.end_encoding()?;
        Lane::complete(&command_buffer);
        drop(held);
        if !case.layers.is_empty() {
            let executions = command_buffer.executions();
            let ExecutedPipeline::Render(executed) = executions[0].pipeline() else {
                panic!("{}: the draw executed no render pipeline", case.name);
            };
            let stages = executed
                .stages
                .iter()
                .filter(|s| s.stage == vk::ShaderStage::GEOMETRY)
                .count();
            assert_eq!(
                stages,
                self.geometry_stages(case.length),
                "{}: the geometry stages of the pipeline the draw executed on {}",
                case.name,
                self.device.name()
            );
        }
        Ok(Stored {
            color: self.slices(&color, slices),
            second: second
                .map(|(second, slices)| self.slices(&second, slices))
                .unwrap_or_default(),
            depth: depth
                .map(|depth| {
                    self.slices(&depth, slices)
                        .into_iter()
                        .map(f32::from_bits)
                        .collect()
                })
                .unwrap_or_default(),
        })
    }
}

/// Each pass of probe 12c stores Metal's texels in every slice of every attachment: the layers
/// from the attachment's slice, each drawn into by the instances that name it and cleared or
/// loaded where none does, and nothing past them. A device that refuses the fixture's pipeline
/// (no `shaderOutputLayer`) prints its typed refusal.
#[test]
fn a_layered_pass_renders_each_layer_as_metal() {
    layered_passes(&[]);
}

/// [`a_layered_pass_renders_each_layer_as_metal`] with `shaderOutputLayer` withheld: the vertex
/// function stores the slice to a varying and the synthesized geometry stages store `Layer` from
/// it, in every pass of two layers or more (asserted from the pipeline each draw executed). A
/// device without `geometryShader` refuses the fixture's pipeline (printed).
#[test]
fn through_a_geometry_stage_a_layered_pass_renders_each_layer_as_metal() {
    layered_passes(&[VkFeature::ShaderOutputLayer]);
}

/// [`a_layered_pass_renders_each_layer_as_metal`]'s passes on the test device without `withheld`.
fn layered_passes(withheld: &[VkFeature]) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let lane = match lane_without(&messages, &cache, withheld, Fragment::Slice) {
            Ok(lane) => lane,
            Err((name, error)) => match *error {
                Error::Unsupported(u) => {
                    eprintln!("refused: layered passes on {name} without {withheld:?}: {u:?}");
                    return;
                }
                other => panic!("layered passes on {name}: {other}"),
            },
        };
        let name = lane.device.name().to_owned();
        // `rt6_layered_vertex` writes `[[render_target_array_index]]` and no viewport index.
        assert!(lane.plain.writes_render_target_array_index());
        assert!(!lane.plain.writes_viewport_array_index());
        const E: u32 = FILLED;
        const C: u32 = CLEARED[0];
        let colors = |color: &[u32]| Stored {
            color: color.to_vec(),
            second: Vec::new(),
            depth: Vec::new(),
        };
        // Metal's texels (a Metal probe on an M4; the M3 Max with macOS 27.0.1 stores the same).
        let cases = [
            (
                Case {
                    name: "len2_slice0",
                    ..PLAIN
                },
                colors(&[0x100, 0x201, E, E]),
            ),
            (
                Case {
                    name: "len2_no_draw",
                    layers: &[],
                    ..PLAIN
                },
                colors(&[C, C, E, E]),
            ),
            (
                Case {
                    name: "len2_load_one",
                    load: LoadAction::Load,
                    layers: &[1],
                    ..PLAIN
                },
                colors(&[E, 0x101, E, E]),
            ),
            (
                Case {
                    name: "len2_slice1",
                    slice: 1,
                    ..PLAIN
                },
                colors(&[E, 0x100, 0x201, E]),
            ),
            (
                Case {
                    name: "len2_slice2_reversed",
                    slice: 2,
                    layers: &[1, 0],
                    ..PLAIN
                },
                colors(&[E, E, 0x200, 0x101]),
            ),
            (
                Case {
                    name: "len4_all",
                    length: 4,
                    layers: &[3, 2, 1, 0],
                    ..PLAIN
                },
                colors(&[0x400, 0x301, 0x202, 0x103]),
            ),
            (
                Case {
                    name: "len6_cube",
                    kind: Kind::Cube,
                    length: 6,
                    layers: &[5, 0, 3],
                    ..PLAIN
                },
                colors(&[0x200, C, C, 0x303, C, 0x105]),
            ),
            (
                Case {
                    name: "len2_cube_face4",
                    kind: Kind::Cube,
                    slice: 4,
                    layers: &[1, 0],
                    ..PLAIN
                },
                colors(&[E, E, E, E, 0x200, 0x101]),
            ),
            (
                Case {
                    name: "len2_depth",
                    layers: &[1, 0, 1],
                    depth: true,
                    ..PLAIN
                },
                Stored {
                    color: vec![0x200, 0x101, E, E],
                    second: Vec::new(),
                    depth: vec![0.5, 0.5, DEPTHS[0], DEPTHS[0]],
                },
            ),
            (
                Case {
                    name: "len2_two_colors",
                    second: Some(2),
                    ..PLAIN
                },
                Stored {
                    color: vec![0x100, 0x201, E, E],
                    second: vec![CLEARED[1]; 2],
                    depth: Vec::new(),
                },
            ),
        ];
        for (case, metal) in &cases {
            assert_eq!(
                lane.try_run(case).unwrap(),
                *metal,
                "{} on {name}: Metal's texels",
                case.name
            );
        }
        eprintln!(
            "layered passes on {name}: {} cases, {} geometry stages in a pass of 2: Metal's texels",
            cases.len(),
            lane.geometry_stages(2)
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// An index at or past the pass's length renders the pass's first layer, and the fragment function
/// reads the index as written, in a pass of length 1 too: each pass of probe 12b stores Metal's
/// texels in every slice. So does a pass of length 0, Metal's default, which plain Metal draws as a
/// pass of length 1 (Metal probe cases `len0_*` on an M4; its debug layer stops these draws). The
/// probe's pass into plane 1 of a 3D texture is not drawn (the runtime refuses a 3D attachment),
/// and its depth pass is drawn as one draw of index 5: the probe drew index 4 into the depth alone
/// first (depth 0.5 in the first layer, 1.0 in the second, as here) and then index 5 behind it,
/// which its depth test dropped.
#[test]
fn an_index_past_the_length_renders_the_first_layer_as_metal() {
    index_past_the_length(&[]);
}

/// [`an_index_past_the_length_renders_the_first_layer_as_metal`] with `shaderOutputLayer`
/// withheld: through the synthesized geometry stages in a pass of two layers or more, and through
/// the variant that stores no slice, with no stage, in a pass of one layer or none.
#[test]
fn through_a_geometry_stage_an_index_past_the_length_renders_the_first_layer_as_metal() {
    index_past_the_length(&[VkFeature::ShaderOutputLayer]);
}

/// [`an_index_past_the_length_renders_the_first_layer_as_metal`]'s passes on the test device
/// without `withheld`.
fn index_past_the_length(withheld: &[VkFeature]) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let lane = match lane_without(&messages, &cache, withheld, Fragment::Slice) {
            Ok(lane) => lane,
            Err((name, error)) => match *error {
                Error::Unsupported(u) => {
                    eprintln!("refused: layered passes on {name} without {withheld:?}: {u:?}");
                    return;
                }
                other => panic!("layered passes on {name}: {other}"),
            },
        };
        let name = lane.device.name().to_owned();
        const E: u32 = FILLED;
        const C: u32 = CLEARED[0];
        let colors = |color: &[u32]| Stored {
            color: color.to_vec(),
            second: Vec::new(),
            depth: Vec::new(),
        };
        // Metal's texels (a Metal probe on an M4; the M3 Max with macOS 27.0.1 stores the same).
        let cases = [
            (
                Case {
                    name: "len2_index2_alone",
                    layers: &[2],
                    ..PLAIN
                },
                colors(&[0x102, C, E, E]),
            ),
            (
                Case {
                    name: "len2_index7_alone",
                    layers: &[7],
                    ..PLAIN
                },
                colors(&[0x107, C, E, E]),
            ),
            (
                Case {
                    name: "len2_index_max",
                    layers: &[u32::MAX],
                    ..PLAIN
                },
                colors(&[u32::MAX, C, E, E]),
            ),
            (
                Case {
                    name: "len2_slice1_index2",
                    slice: 1,
                    layers: &[2],
                    ..PLAIN
                },
                colors(&[E, 0x102, C, E]),
            ),
            (
                Case {
                    name: "len2_slice1_index5",
                    slice: 1,
                    layers: &[5],
                    ..PLAIN
                },
                colors(&[E, 0x105, C, E]),
            ),
            (
                Case {
                    name: "len1_slice2_index1",
                    slice: 2,
                    length: 1,
                    layers: &[1],
                    ..PLAIN
                },
                colors(&[E, E, 0x101, E]),
            ),
            (
                Case {
                    name: "len1_slice2_index0",
                    slice: 2,
                    length: 1,
                    layers: &[0],
                    ..PLAIN
                },
                colors(&[E, E, 0x100, E]),
            ),
            (
                Case {
                    name: "len3_slice1_index3_then_1",
                    slice: 1,
                    length: 3,
                    layers: &[3, 1],
                    ..PLAIN
                },
                colors(&[E, 0x103, 0x201, C]),
            ),
            (
                Case {
                    name: "len6_cube_index6",
                    kind: Kind::Cube,
                    length: 6,
                    layers: &[6],
                    ..PLAIN
                },
                colors(&[0x106, C, C, C, C, C]),
            ),
            (
                Case {
                    name: "len2_cube_face4_index3",
                    kind: Kind::Cube,
                    slice: 4,
                    layers: &[3],
                    ..PLAIN
                },
                colors(&[E, E, E, E, 0x103, C]),
            ),
            (
                Case {
                    name: "len2_depth_index5",
                    layers: &[5],
                    depth: true,
                    ..PLAIN
                },
                Stored {
                    color: vec![0x105, C, E, E],
                    second: Vec::new(),
                    depth: vec![0.5, DEPTHS[1], DEPTHS[0], DEPTHS[0]],
                },
            ),
            (
                Case {
                    name: "len0_index1",
                    length: 0,
                    layers: &[1],
                    ..PLAIN
                },
                colors(&[0x101, E, E, E]),
            ),
            (
                Case {
                    name: "len0_index_max",
                    length: 0,
                    layers: &[u32::MAX],
                    ..PLAIN
                },
                colors(&[u32::MAX, E, E, E]),
            ),
            (
                Case {
                    name: "len0_index_3_then_0",
                    length: 0,
                    layers: &[3, 0],
                    ..PLAIN
                },
                colors(&[0x200, E, E, E]),
            ),
            (
                Case {
                    name: "len0_2d_texture_index1",
                    kind: Kind::Plain,
                    length: 0,
                    layers: &[1],
                    ..PLAIN
                },
                colors(&[0x101]),
            ),
            (
                Case {
                    name: "len0_slice2_index1",
                    slice: 2,
                    length: 0,
                    layers: &[1],
                    ..PLAIN
                },
                colors(&[E, E, 0x101, E]),
            ),
        ];
        for (case, metal) in &cases {
            assert_eq!(
                lane.try_run(case).unwrap(),
                *metal,
                "{} on {name}: Metal's texels",
                case.name
            );
        }
        eprintln!(
            "an index past the length on {name}: {} cases, {} geometry stages in a pass of 2: \
             Metal's texels",
            cases.len(),
            lane.geometry_stages(2)
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// `rt6_layered_weights` reads the slice and the tag, the primitive ID and the barycentric weights
/// together. Each pass stores Metal's texels (a Metal probe of the fixture's functions on an M3 Max
/// with macOS 27.0.1, with and without API validation where it draws): the primitive ID restarts
/// at 0 in each instance, and every weight of the fullscreen triangle at a pixel's center lies in
/// [0, 1] (bit 23).
#[test]
fn the_slice_the_weights_and_the_primitive_id_read_as_metal() {
    weights_passes(&[]);
}

/// [`the_slice_the_weights_and_the_primitive_id_read_as_metal`] with `shaderOutputLayer` and
/// `fragmentShaderBarycentric` withheld: one set of synthesized geometry stages stores the slice
/// and writes the weights and the primitive ID, in every pass (asserted from the pipeline each draw
/// executed), the pass of one layer included, whose variant stores no slice. A device without
/// `geometryShader` refuses the fixture's pipeline (printed).
#[test]
fn through_one_geometry_stage_the_slice_the_weights_and_the_primitive_id_read_as_metal() {
    weights_passes(&[
        VkFeature::ShaderOutputLayer,
        VkFeature::FragmentShaderBarycentric,
    ]);
}

/// [`the_slice_the_weights_and_the_primitive_id_read_as_metal`]'s passes on the test device
/// without `withheld`.
fn weights_passes(withheld: &[VkFeature]) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let lane = match lane_without(&messages, &cache, withheld, Fragment::Weights) {
            Ok(lane) => lane,
            Err((name, error)) => match *error {
                Error::Unsupported(u) => {
                    eprintln!(
                        "refused: the weights in layered passes on {name} without {withheld:?}: {u:?}"
                    );
                    return;
                }
                other => panic!("the weights in layered passes on {name}: {other}"),
            },
        };
        let name = lane.device.name().to_owned();
        const E: u32 = FILLED;
        const C: u32 = CLEARED[0];
        /// Every weight in [0, 1], and the primitive ID 0.
        const W: u32 = 1 << 23;
        let colors = |color: &[u32]| Stored {
            color: color.to_vec(),
            second: Vec::new(),
            depth: Vec::new(),
        };
        // Metal's texels (a Metal probe on an M3 Max with macOS 27.0.1).
        let cases = [
            (
                Case {
                    name: "len2_slice0",
                    ..PLAIN
                },
                colors(&[W | 0x100, W | 0x201, E, E]),
            ),
            (
                Case {
                    name: "len4_all",
                    length: 4,
                    layers: &[3, 2, 1, 0],
                    ..PLAIN
                },
                colors(&[W | 0x400, W | 0x301, W | 0x202, W | 0x103]),
            ),
            (
                Case {
                    name: "len4_same_layer",
                    length: 4,
                    layers: &[2, 2, 2],
                    ..PLAIN
                },
                colors(&[C, C, W | 0x302, C]),
            ),
            (
                Case {
                    name: "len6_cube",
                    kind: Kind::Cube,
                    length: 6,
                    layers: &[5, 0, 3],
                    ..PLAIN
                },
                colors(&[W | 0x200, C, C, W | 0x303, C, W | 0x105]),
            ),
            (
                Case {
                    name: "len2_index7_alone",
                    layers: &[7],
                    ..PLAIN
                },
                colors(&[W | 0x107, C, E, E]),
            ),
            (
                Case {
                    name: "len1_slice2_index1",
                    slice: 2,
                    length: 1,
                    layers: &[1],
                    ..PLAIN
                },
                colors(&[E, E, W | 0x101, E]),
            ),
            (
                Case {
                    name: "len1_two_instances",
                    length: 1,
                    layers: &[0, 3],
                    ..PLAIN
                },
                colors(&[W | 0x203, E, E, E]),
            ),
            (
                Case {
                    name: "len0_index_3_then_0",
                    length: 0,
                    layers: &[3, 0],
                    ..PLAIN
                },
                colors(&[W | 0x200, E, E, E]),
            ),
        ];
        for (case, metal) in &cases {
            assert_eq!(
                lane.try_run(case).unwrap(),
                *metal,
                "{} on {name}: Metal's texels",
                case.name
            );
        }
        eprintln!(
            "the weights in layered passes on {name}: {} cases, {} geometry stages in a pass of 1: \
             Metal's texels",
            cases.len(),
            lane.geometry_stages(1)
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// What Metal's validation stops in a layered pass is refused typed before anything runs (Metal
/// probes on an M4): an attachment without the pass's slices from its own, a second color
/// attachment's and a 2D texture's among them ("slice + renderTargetArrayLength is 3, but the
/// texture only has 2 slices").
#[test]
fn a_layered_pass_is_refused_as_metal_refuses_it() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = match Lane::new(device, Fragment::Slice) {
            Ok(lane) => lane,
            Err(Error::Unsupported(u)) => {
                eprintln!("refused: layered passes on {name}: {u:?}");
                return;
            }
            Err(other) => panic!("layered passes on {name}: {other}"),
        };
        let past = |attachment, end, slices| {
            Error::Invalid(Invalid::new(
                "render_command_encoder",
                Problem::RenderPass(RenderPassProblem::ArrayLengthPastSlices {
                    attachment,
                    end,
                    slices,
                }),
            ))
        };
        let refusals = [
            (
                Case {
                    name: "len3_two_colors_short",
                    length: 3,
                    layers: &[2],
                    second: Some(2),
                    ..PLAIN
                },
                past(Attachment::Color(1), 3, 2),
            ),
            (
                Case {
                    name: "len2_2d_texture",
                    kind: Kind::Plain,
                    ..PLAIN
                },
                past(Attachment::Color(0), 2, 1),
            ),
            (
                Case {
                    name: "len2_slice3_past",
                    slice: 3,
                    layers: &[0],
                    ..PLAIN
                },
                past(Attachment::Color(0), 5, 4),
            ),
        ];
        for (case, refusal) in &refusals {
            assert_eq!(
                lane.try_run(case).unwrap_err(),
                *refusal,
                "{} on {name}",
                case.name
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// A resolve lane texel (RGBA8): white where a draw covered the pixel, the clear (zero) where none
/// did, and what a resolve texture holds before its pass (0x808080ff, as in a Metal probe on the M3
/// Max and an M4).
const WHITE: u32 = 0xffff_ffff;
const CLEAR: u32 = 0;
const FILL: u32 = 0xff80_8080;

/// The steps a layered pass takes of every layer: resolves, separate depth and stencil textures,
/// stand-ins. The fixture `rt6_layered`'s vertex function draws the fullscreen triangle at depth
/// 0.5 into the layers its buffer names, one per instance; `rt6_resolves`' fragment function
/// writes white.
struct Steps {
    device: Device,
    queue: CommandQueue,
    /// White into an RGBA8 attachment at 4 samples.
    white: RenderPipelineState,
    /// The layered fragment function's word into an `R32Uint` attachment beside `Depth32Float`
    /// and `Stencil8` ones.
    separate: RenderPipelineState,
    /// `Less` with writes, the stencil replaced by the reference.
    replace: DepthStencilState,
    /// White into an RGBA8 attachment beside separate `Depth32Float` and `Stencil8` ones, and
    /// beside a `Depth32Float_Stencil8` one, each at 4 samples.
    separate_samples: RenderPipelineState,
    combined_samples: RenderPipelineState,
    /// The amplified fixture's vertex function, with white, up to 2 amplifications at 4 samples.
    amplified: Option<RenderPipelineState>,
}

/// One layered resolve pass of the lane: a 4x RGBA8 2D multisample array of `slices` (memoryless
/// where `memoryless`), rendered from `slice` in a pass of `length`, cleared to zero and stored by
/// `store`; each draw of `draws` names its layer (a memory barrier over buffers between two where
/// `barrier`); resolved into `resolve` (a 2D array of its slices, or a 2D texture for 1) at
/// `resolve_slice`, with render target usage where `rendered`, else only shader reads; `late`
/// sets that target with `set_color_resolve_target` over the descriptor's resolve texture of 2
/// slices. `amplified` draws once, an amplification count of 1 mapped to that layer.
struct ResolveCase {
    name: &'static str,
    slices: u64,
    slice: u64,
    length: u64,
    store: StoreAction,
    draws: &'static [u32],
    barrier: bool,
    memoryless: bool,
    resolve: u64,
    resolve_slice: u64,
    rendered: bool,
    late: bool,
    amplified: Option<u32>,
}

const RESOLVE: ResolveCase = ResolveCase {
    name: "",
    slices: 2,
    slice: 0,
    length: 2,
    store: StoreAction::MultisampleResolve,
    draws: &[],
    barrier: false,
    memoryless: false,
    resolve: 2,
    resolve_slice: 0,
    rendered: true,
    late: false,
    amplified: None,
};

impl Steps {
    /// The lane on `device`; `Err` with the device's refusal of the fixtures' pipelines.
    fn new(device: Device) -> Result<Steps, Error> {
        let library = |name: &str| {
            let ll = metal2vulkan_devtools::engine_dir().join(name);
            device
                .new_library_with_air(&llvm_assemble(&ll).unwrap())
                .unwrap()
        };
        let layered = library("rt6_layered.ll");
        let resolves = library("rt6_resolves.ll");
        let amplified_library = library("rt6_amplified.ll");
        let function = |library: &metal2vulkan_runtime::Library, name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut white = RenderPipelineDescriptor::new(function(&layered, "rt6_layered_vertex"));
        white.fragment = Some(function(&resolves, "rt6_resolve_white"));
        white.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        white.raster_sample_count = 4;
        let mut separate = RenderPipelineDescriptor::new(function(&layered, "rt6_layered_vertex"));
        separate.fragment = Some(function(&layered, "rt6_layered_fragment"));
        separate.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
        separate.depth_format = Some(PixelFormat::Depth32Float);
        separate.stencil_format = Some(PixelFormat::Stencil8);
        let mut amplified =
            RenderPipelineDescriptor::new(function(&amplified_library, "rt6_amplified_vertex"));
        amplified.fragment = Some(function(&resolves, "rt6_resolve_white"));
        amplified.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        amplified.raster_sample_count = 4;
        amplified.max_vertex_amplification_count = 2;
        let samples = |depth: PixelFormat, stencil: PixelFormat| {
            let mut d = RenderPipelineDescriptor::new(function(&layered, "rt6_layered_vertex"));
            d.fragment = Some(function(&resolves, "rt6_resolve_white"));
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            d.depth_format = Some(depth);
            d.stencil_format = Some(stencil);
            d.raster_sample_count = 4;
            d
        };
        let separate_samples = samples(PixelFormat::Depth32Float, PixelFormat::Stencil8);
        let combined = PixelFormat::Depth32FloatStencil8;
        let combined_samples = samples(combined, combined);
        let mut stencil = StencilDescriptor::default();
        stencil.stencil_compare_function = CompareFunction::Always;
        stencil.depth_stencil_pass_operation = StencilOperation::Replace;
        let mut replace = DepthStencilDescriptor::default();
        replace.depth_compare_function = CompareFunction::Less;
        replace.depth_write_enabled = true;
        replace.front_face_stencil = stencil;
        replace.back_face_stencil = stencil;
        let options = PipelineOptions::default();
        Ok(Steps {
            white: device.new_render_pipeline_state(&white, &options)?,
            separate: device.new_render_pipeline_state(&separate, &options)?,
            replace: device.new_depth_stencil_state(&replace).unwrap(),
            separate_samples: device.new_render_pipeline_state(&separate_samples, &options)?,
            combined_samples: device.new_render_pipeline_state(&combined_samples, &options)?,
            amplified: match device.new_render_pipeline_state(&amplified, &options) {
                Ok(pipeline) => Some(pipeline),
                Err(Error::Unsupported(u)) => {
                    eprintln!("refused: amplification on {}: {u:?}", device.name());
                    None
                }
                Err(other) => return Err(other),
            },
            queue: device.new_command_queue().unwrap(),
            device,
        })
    }

    /// A private 4² texture of `pf`, `ty`, `slices` and `samples`, of `usage` (memoryless where
    /// `memoryless`); the device's refusal of a memoryless one.
    fn texture(
        &self,
        pf: PixelFormat,
        ty: TextureType,
        (slices, samples): (u64, u64),
        usage: TextureUsage,
        memoryless: bool,
    ) -> Result<Texture, Error> {
        let mut descriptor = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
        descriptor.texture_type = ty;
        descriptor.array_length = slices;
        descriptor.sample_count = samples;
        descriptor.usage = usage;
        descriptor.resource_options = if memoryless {
            ResourceOptions::new(
                StorageMode::Memoryless,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Default,
            )
        } else {
            ResourceOptions::private()
        };
        self.device.new_texture(&descriptor)
    }

    /// Each slice's word of `texture` (`bytes` per texel), every texel of a slice alike, checked.
    fn words(&self, texture: &Texture, slices: u64, bytes: u64) -> Vec<u32> {
        self.words_through(texture, slices, bytes, BlitOption::None)
    }

    /// [`Steps::words`] read through `option`.
    fn words_through(
        &self,
        texture: &Texture,
        slices: u64,
        bytes: u64,
        option: BlitOption,
    ) -> Vec<u32> {
        let length = SIDE * SIDE * bytes;
        let out = buffer_with(&self.device, &vec![0u8; (length * slices) as usize]).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for slice in 0..slices {
            blit.copy_from_texture_to_buffer(
                texture,
                slice,
                0,
                Origin::new(0, 0, 0),
                Size::new(SIDE, SIDE, 1),
                &out,
                slice * length,
                SIDE * bytes,
                length,
                option,
            )
            .unwrap();
        }
        blit.end_encoding().unwrap();
        Lane::complete(&command_buffer);
        let read = read_buffer(&out, 0, length * slices).unwrap();
        read.chunks_exact(length as usize)
            .map(|slice| {
                let texels: Vec<u32> = slice
                    .chunks_exact(bytes as usize)
                    .map(|t| {
                        let mut word = [0u8; 4];
                        word[..t.len()].copy_from_slice(t);
                        u32::from_le_bytes(word)
                    })
                    .collect();
                assert!(texels.iter().all(|&w| w == texels[0]), "{texels:x?}");
                texels[0]
            })
            .collect()
    }

    /// Fill every slice of the RGBA8 `texture` with [`FILL`].
    fn fill(&self, texture: &Texture, slices: u64) {
        let words: Vec<u8> = (0..SIDE * SIDE).flat_map(|_| FILL.to_le_bytes()).collect();
        let source = buffer_with(&self.device, &words).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for slice in 0..slices {
            blit.copy_from_buffer_to_texture(
                &source,
                0,
                SIDE * 4,
                SIDE * SIDE * 4,
                Size::new(SIDE, SIDE, 1),
                texture,
                slice,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
            .unwrap();
        }
        blit.end_encoding().unwrap();
        Lane::complete(&command_buffer);
    }

    /// A layered pass of length 2 over separate `Depth32Float` and `Stencil8` 2D arrays of
    /// `slices` (each slice cleared to 0.75 and 0x11 by a pass of its own first) beside an
    /// `R32Uint` one, from `slice`, loading or clearing (to 1.0 and 0x22) depth and stencil and
    /// storing both: the fixture's triangle at depth 0.5 into each layer of `layers`, `Less` with
    /// writes, the stencil replaced by 0x33. Each slice's depth bits and stencil; the encoder's or
    /// the draw's refusal.
    fn separate(
        &self,
        slices: u64,
        slice: u64,
        load: LoadAction,
        layers: &[u32],
    ) -> Result<(Vec<u32>, Vec<u32>), Error> {
        let array = |pf| {
            self.texture(
                pf,
                TextureType::Type2DArray,
                (slices, 1),
                TextureUsage::RENDER_TARGET,
                false,
            )
            .unwrap()
        };
        let color = array(PixelFormat::R32Uint);
        let depth = array(PixelFormat::Depth32Float);
        let stencil = array(PixelFormat::Stencil8);
        let command_buffer = self.queue.command_buffer().unwrap();
        for each in 0..slices {
            let mut pass = RenderPassDescriptor::new();
            pass.depth_attachment.texture = Some(depth.clone());
            pass.depth_attachment.slice = each;
            pass.depth_attachment.load_action = LoadAction::Clear;
            pass.depth_attachment.clear_depth = 0.75;
            pass.depth_attachment.store_action = StoreAction::Store;
            pass.stencil_attachment.texture = Some(stencil.clone());
            pass.stencil_attachment.slice = each;
            pass.stencil_attachment.load_action = LoadAction::Clear;
            pass.stencil_attachment.clear_stencil = 0x11;
            pass.stencil_attachment.store_action = StoreAction::Store;
            command_buffer
                .render_command_encoder(&pass)
                .unwrap()
                .end_encoding()
                .unwrap();
        }
        Lane::complete(&command_buffer);
        let mut pass = RenderPassDescriptor::new();
        pass.render_target_array_length = 2;
        let target = &mut pass.color_attachments[0];
        target.texture = Some(color.clone());
        target.slice = slice;
        target.load_action = LoadAction::Clear;
        target.store_action = StoreAction::Store;
        let target = &mut pass.depth_attachment;
        target.texture = Some(depth.clone());
        target.slice = slice;
        target.load_action = load;
        target.clear_depth = 1.0;
        target.store_action = StoreAction::Store;
        let target = &mut pass.stencil_attachment;
        target.texture = Some(stencil.clone());
        target.slice = slice;
        target.load_action = load;
        target.clear_stencil = 0x22;
        target.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        encoder.set_render_pipeline_state(&self.separate)?;
        encoder.set_depth_stencil_state(&self.replace)?;
        encoder.set_stencil_reference_value(0x33);
        let words: Vec<u8> = layers.iter().flat_map(|l| l.to_le_bytes()).collect();
        let buffer = buffer_with(&self.device, &words).unwrap();
        encoder.set_vertex_buffer(Some(&buffer), 0, 0)?;
        encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, layers.len() as u32, 0)?;
        encoder.end_encoding()?;
        Lane::complete(&command_buffer);
        drop(buffer);
        Ok((
            self.words(&depth, slices, 4),
            self.words(&stencil, slices, 1),
        ))
    }

    /// Each slice of the multisampled `Depth32Float` and `Stencil8` arrays `depth` and `stencil`
    /// of `slices`, read by a pass of that slice alone resolving sample 0 of each into a 2D
    /// texture: its depth bits and stencil (every sample alike where the lane draws).
    fn sample_zero(
        &self,
        depth: &Texture,
        stencil: &Texture,
        slices: u64,
    ) -> Result<(Vec<u32>, Vec<u32>), Error> {
        let (mut depths, mut stencils) = (Vec::new(), Vec::new());
        for slice in 0..slices {
            for (texture, pf) in [
                (depth, PixelFormat::Depth32Float),
                (stencil, PixelFormat::Stencil8),
            ] {
                let resolve = self.texture(
                    pf,
                    TextureType::Type2D,
                    (1, 1),
                    TextureUsage::RENDER_TARGET,
                    false,
                )?;
                let mut pass = RenderPassDescriptor::new();
                if pf == PixelFormat::Depth32Float {
                    let target = &mut pass.depth_attachment;
                    target.texture = Some(texture.clone());
                    target.slice = slice;
                    target.load_action = LoadAction::Load;
                    target.store_action = StoreAction::MultisampleResolve;
                    target.resolve_texture = Some(resolve.clone());
                } else {
                    let target = &mut pass.stencil_attachment;
                    target.texture = Some(texture.clone());
                    target.slice = slice;
                    target.load_action = LoadAction::Load;
                    target.store_action = StoreAction::MultisampleResolve;
                    target.resolve_texture = Some(resolve.clone());
                }
                let command_buffer = self.queue.command_buffer().unwrap();
                command_buffer
                    .render_command_encoder(&pass)?
                    .end_encoding()?;
                Lane::complete(&command_buffer);
                let bytes = if pf == PixelFormat::Depth32Float {
                    4
                } else {
                    1
                };
                let word = self.words(&resolve, 1, bytes)[0];
                match pf {
                    PixelFormat::Depth32Float => depths.push(word),
                    _ => stencils.push(word),
                }
            }
        }
        Ok((depths, stencils))
    }

    /// [`Steps::separate`] at 4 samples: separate multisampled `Depth32Float` and `Stencil8` 2D
    /// arrays of `slices` (each slice cleared to 0.75 and 0x11 by a pass of its own first) beside
    /// a multisampled RGBA8 one, rendered in a pass of `length` layers from `slice`, loading or
    /// clearing (to 1.0 and 0x22) and storing both: the fixture's triangle into each layer of
    /// `layers` (no draw for none), `Less` with writes, the stencil replaced by 0x33. Each slice's
    /// depth bits and stencil, read by [`Steps::sample_zero`]; the encoder's or the draw's
    /// refusal.
    fn separate_multisampled(
        &self,
        (slices, slice, length): (u64, u64, u64),
        load: LoadAction,
        layers: &[u32],
    ) -> Result<(Vec<u32>, Vec<u32>), Error> {
        let array = |pf| {
            self.texture(
                pf,
                TextureType::Type2DMultisampleArray,
                (slices, 4),
                TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
                false,
            )
        };
        let color = array(PixelFormat::RGBA8Unorm)?;
        let depth = array(PixelFormat::Depth32Float)?;
        let stencil = array(PixelFormat::Stencil8)?;
        let command_buffer = self.queue.command_buffer().unwrap();
        for each in 0..slices {
            let mut pass = RenderPassDescriptor::new();
            pass.depth_attachment.texture = Some(depth.clone());
            pass.depth_attachment.slice = each;
            pass.depth_attachment.load_action = LoadAction::Clear;
            pass.depth_attachment.clear_depth = 0.75;
            pass.depth_attachment.store_action = StoreAction::Store;
            pass.stencil_attachment.texture = Some(stencil.clone());
            pass.stencil_attachment.slice = each;
            pass.stencil_attachment.load_action = LoadAction::Clear;
            pass.stencil_attachment.clear_stencil = 0x11;
            pass.stencil_attachment.store_action = StoreAction::Store;
            command_buffer
                .render_command_encoder(&pass)?
                .end_encoding()?;
        }
        Lane::complete(&command_buffer);
        let mut pass = RenderPassDescriptor::new();
        pass.render_target_array_length = length;
        let target = &mut pass.color_attachments[0];
        target.texture = Some(color);
        target.slice = slice;
        target.load_action = LoadAction::Clear;
        target.store_action = StoreAction::DontCare;
        let target = &mut pass.depth_attachment;
        target.texture = Some(depth.clone());
        target.slice = slice;
        target.load_action = load;
        target.clear_depth = 1.0;
        target.store_action = StoreAction::Store;
        let target = &mut pass.stencil_attachment;
        target.texture = Some(stencil.clone());
        target.slice = slice;
        target.load_action = load;
        target.clear_stencil = 0x22;
        target.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        encoder.set_render_pipeline_state(&self.separate_samples)?;
        encoder.set_depth_stencil_state(&self.replace)?;
        encoder.set_stencil_reference_value(0x33);
        // No layers: a pass of no draws, which loads and stores every layer.
        let mut held = None;
        if !layers.is_empty() {
            let words: Vec<u8> = layers.iter().flat_map(|l| l.to_le_bytes()).collect();
            let buffer = buffer_with(&self.device, &words).unwrap();
            encoder.set_vertex_buffer(Some(&buffer), 0, 0)?;
            encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, layers.len() as u32, 0)?;
            held = Some(buffer);
        }
        encoder.end_encoding()?;
        Lane::complete(&command_buffer);
        drop(held);
        self.sample_zero(&depth, &stencil, slices)
    }

    /// A pass of `length` layers (2, or 1 for a pass of one layer) from `slice` over a
    /// multisampled `Depth32Float_Stencil8` 2D array of 3 slices (and an RGBA8 one), cleared to
    /// 1.0 and 0, the fixture's triangle drawn into layer 0 with the stencil replaced by 0x33 and,
    /// in a layered pass, into layer 1 by 0x55, both aspects resolved by `filters` into a 2D array
    /// of 3 slices at `resolve_slice` (each slice cleared to 0.75 and 7 first). Each resolve
    /// slice's depth bits and stencil; the encoder's or the draw's refusal.
    fn resolved_depth_stencil(
        &self,
        (depth_filter, stencil_filter): (
            MultisampleDepthResolveFilter,
            MultisampleStencilResolveFilter,
        ),
        (slice, length): (u64, u64),
        resolve_slice: u64,
    ) -> Result<(Vec<u32>, Vec<u32>), Error> {
        let combined = PixelFormat::Depth32FloatStencil8;
        let read = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let ty = TextureType::Type2DMultisampleArray;
        let color = self.texture(PixelFormat::RGBA8Unorm, ty, (3, 4), read, false)?;
        let samples = self.texture(combined, ty, (3, 4), read, false)?;
        let resolve = self.texture(
            combined,
            TextureType::Type2DArray,
            (3, 1),
            TextureUsage::RENDER_TARGET,
            false,
        )?;
        let command_buffer = self.queue.command_buffer().unwrap();
        for each in 0..3 {
            let mut pass = RenderPassDescriptor::new();
            pass.depth_attachment.texture = Some(resolve.clone());
            pass.depth_attachment.slice = each;
            pass.depth_attachment.load_action = LoadAction::Clear;
            pass.depth_attachment.clear_depth = 0.75;
            pass.depth_attachment.store_action = StoreAction::Store;
            pass.stencil_attachment.texture = Some(resolve.clone());
            pass.stencil_attachment.slice = each;
            pass.stencil_attachment.load_action = LoadAction::Clear;
            pass.stencil_attachment.clear_stencil = 7;
            pass.stencil_attachment.store_action = StoreAction::Store;
            command_buffer
                .render_command_encoder(&pass)?
                .end_encoding()?;
        }
        Lane::complete(&command_buffer);
        let mut pass = RenderPassDescriptor::new();
        pass.render_target_array_length = length;
        let target = &mut pass.color_attachments[0];
        target.texture = Some(color);
        target.slice = slice;
        target.load_action = LoadAction::Clear;
        target.store_action = StoreAction::DontCare;
        let target = &mut pass.depth_attachment;
        target.texture = Some(samples.clone());
        target.slice = slice;
        target.load_action = LoadAction::Clear;
        target.clear_depth = 1.0;
        target.store_action = StoreAction::MultisampleResolve;
        target.resolve_texture = Some(resolve.clone());
        target.resolve_slice = resolve_slice;
        target.depth_resolve_filter = depth_filter;
        let target = &mut pass.stencil_attachment;
        target.texture = Some(samples);
        target.slice = slice;
        target.load_action = LoadAction::Clear;
        target.clear_stencil = 0;
        target.store_action = StoreAction::MultisampleResolve;
        target.resolve_texture = Some(resolve.clone());
        target.resolve_slice = resolve_slice;
        target.stencil_resolve_filter = stencil_filter;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        encoder.set_render_pipeline_state(&self.combined_samples)?;
        encoder.set_depth_stencil_state(&self.replace)?;
        let mut held = Vec::new();
        let draws: &[(u32, u32)] = match length > 1 {
            true => &[(0, 0x33), (1, 0x55)],
            false => &[(0, 0x33)],
        };
        for &(layer, reference) in draws {
            encoder.set_stencil_reference_value(reference);
            let buffer = buffer_with(&self.device, &layer.to_le_bytes()).unwrap();
            encoder.set_vertex_buffer(Some(&buffer), 0, 0)?;
            encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)?;
            held.push(buffer);
        }
        encoder.end_encoding()?;
        Lane::complete(&command_buffer);
        drop(held);
        Ok((
            self.words_through(&resolve, 3, 4, BlitOption::DepthFromDepthStencil),
            self.words_through(&resolve, 3, 1, BlitOption::StencilFromDepthStencil),
        ))
    }

    /// The pass of `case`, run: each resolve slice's texel, and the descriptor's resolve
    /// texture's where the case sets a late target; the encoder's or a draw's refusal.
    fn resolve(&self, case: &ResolveCase) -> Result<(Vec<u32>, Vec<u32>), Error> {
        let rgba8 = PixelFormat::RGBA8Unorm;
        let msaa = self.texture(
            rgba8,
            TextureType::Type2DMultisampleArray,
            (case.slices, 4),
            TextureUsage::RENDER_TARGET,
            case.memoryless,
        )?;
        let (ty, slices) = match case.resolve {
            1 => (TextureType::Type2D, 1),
            slices => (TextureType::Type2DArray, slices),
        };
        let usage = if case.rendered {
            TextureUsage::RENDER_TARGET
        } else {
            TextureUsage::SHADER_READ
        };
        let resolve = self.texture(rgba8, ty, (slices, 1), usage, false).unwrap();
        self.fill(&resolve, slices);
        let decoy = self
            .texture(
                rgba8,
                TextureType::Type2DArray,
                (2, 1),
                TextureUsage::RENDER_TARGET,
                false,
            )
            .unwrap();
        self.fill(&decoy, 2);
        let mut pass = RenderPassDescriptor::new();
        pass.render_target_array_length = case.length;
        let target = &mut pass.color_attachments[0];
        target.texture = Some(msaa.clone());
        target.slice = case.slice;
        target.load_action = LoadAction::Clear;
        target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        target.store_action = case.store;
        if case.late {
            target.resolve_texture = Some(decoy.clone());
        } else {
            target.resolve_texture = Some(resolve.clone());
            target.resolve_slice = case.resolve_slice;
        }
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        let mut held: Vec<Buffer> = Vec::new();
        match case.amplified {
            Some(layer) => {
                let pipeline = self.amplified.as_ref().expect("an amplified pipeline");
                encoder.set_render_pipeline_state(pipeline)?;
                let mapping = [VertexAmplificationViewMapping::new(0, layer)];
                encoder.set_vertex_amplification_count(1, Some(&mapping))?;
                encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)?;
            }
            None => {
                encoder.set_render_pipeline_state(&self.white)?;
                for (index, &layer) in case.draws.iter().enumerate() {
                    if index > 0 && case.barrier {
                        encoder.memory_barrier_with_scope(
                            BarrierScope::BUFFERS,
                            RenderStages::VERTEX,
                            RenderStages::VERTEX,
                        );
                    }
                    let layers = buffer_with(&self.device, &layer.to_le_bytes()).unwrap();
                    encoder.set_vertex_buffer(Some(&layers), 0, 0)?;
                    encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)?;
                    held.push(layers);
                }
            }
        }
        if case.late {
            let mut late = ResolveTarget::new(resolve.clone());
            late.slice = case.resolve_slice;
            encoder.set_color_resolve_target(Some(late), 0)?;
        }
        encoder.end_encoding()?;
        Lane::complete(&command_buffer);
        drop(held);
        let decoyed = if case.late {
            self.words(&decoy, 2, 4)
        } else {
            Vec::new()
        };
        Ok((self.words(&resolve, slices, 4), decoyed))
    }
}

/// A layered pass resolves each of its layers as Metal does (a Metal probe on an M3 Max with macOS
/// 26.7 and an M4 with 27.0.1, alike, with and without API validation): layer `i`, counted from the
/// attachment's slice, into slice `resolveSlice + i` of the resolve texture, none past its slices
/// (a 2D resolve texture takes layer 0 alone), a layer no draw reached resolving the clear. Through
/// the resolve texture's own slices where they hold the pass's layers, else through a scratch image
/// whose fitting layers are copied (a texture without render target usage too); a late target
/// likewise, the descriptor's texture keeping its texels; and where the pass's last instance
/// renders one layer (an amplification moved there), through one more instance over every layer.
#[test]
fn a_layered_pass_resolves_each_layer_as_metal() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let steps = match Steps::new(device) {
            Ok(steps) => steps,
            Err(Error::Unsupported(u)) => {
                eprintln!("refused: layered resolves on {name}: {u:?}");
                return;
            }
            Err(other) => panic!("layered resolves on {name}: {other}"),
        };
        let cases = [
            (
                ResolveCase {
                    name: "len2_into2_at0",
                    draws: &[1],
                    ..RESOLVE
                },
                vec![CLEAR, WHITE],
            ),
            (
                ResolveCase {
                    name: "len2_into3_at1",
                    draws: &[0],
                    resolve: 3,
                    resolve_slice: 1,
                    ..RESOLVE
                },
                vec![FILL, WHITE, CLEAR],
            ),
            (
                ResolveCase {
                    name: "len2_into3_at0",
                    draws: &[1],
                    resolve: 3,
                    ..RESOLVE
                },
                vec![CLEAR, WHITE, FILL],
            ),
            (
                ResolveCase {
                    name: "len2_into2_at1",
                    draws: &[0],
                    resolve_slice: 1,
                    ..RESOLVE
                },
                vec![FILL, WHITE],
            ),
            (
                ResolveCase {
                    name: "len2_into_2d",
                    draws: &[1],
                    resolve: 1,
                    ..RESOLVE
                },
                vec![CLEAR],
            ),
            (
                ResolveCase {
                    name: "len2_into_2d_drawn",
                    draws: &[0],
                    resolve: 1,
                    ..RESOLVE
                },
                vec![WHITE],
            ),
            (
                ResolveCase {
                    name: "len2_from1_of3_into3_at0",
                    slices: 3,
                    slice: 1,
                    draws: &[1],
                    resolve: 3,
                    ..RESOLVE
                },
                vec![CLEAR, WHITE, FILL],
            ),
            (
                ResolveCase {
                    name: "len3_one_drawn",
                    slices: 3,
                    length: 3,
                    draws: &[0],
                    resolve: 3,
                    ..RESOLVE
                },
                vec![WHITE, CLEAR, CLEAR],
            ),
            (
                ResolveCase {
                    name: "len2_store_and_resolve",
                    store: StoreAction::StoreAndMultisampleResolve,
                    draws: &[0, 1],
                    ..RESOLVE
                },
                vec![WHITE, WHITE],
            ),
            (
                ResolveCase {
                    name: "len2_unrendered_into2_at0",
                    draws: &[1],
                    rendered: false,
                    ..RESOLVE
                },
                vec![CLEAR, WHITE],
            ),
            (
                ResolveCase {
                    name: "len2_unrendered_into3_at1",
                    draws: &[1],
                    resolve: 3,
                    resolve_slice: 1,
                    rendered: false,
                    ..RESOLVE
                },
                vec![FILL, CLEAR, WHITE],
            ),
            (
                ResolveCase {
                    name: "len2_late_into3_at1",
                    draws: &[0],
                    resolve: 3,
                    resolve_slice: 1,
                    late: true,
                    ..RESOLVE
                },
                vec![FILL, WHITE, CLEAR],
            ),
            (
                ResolveCase {
                    name: "len2_amplified_into_layer1",
                    amplified: Some(1),
                    ..RESOLVE
                },
                vec![CLEAR, WHITE],
            ),
            // A memoryless attachment of a pass a barrier splits renders into a stand-in of the
            // pass's layers.
            (
                ResolveCase {
                    name: "len2_memoryless_barrier",
                    draws: &[0, 1],
                    barrier: true,
                    memoryless: true,
                    ..RESOLVE
                },
                vec![WHITE, WHITE],
            ),
            (
                ResolveCase {
                    name: "len2_memoryless_barrier_same",
                    draws: &[0, 0],
                    barrier: true,
                    memoryless: true,
                    ..RESOLVE
                },
                vec![WHITE, CLEAR],
            ),
        ];
        for (case, metal) in &cases {
            if case.amplified.is_some() && steps.amplified.is_none() {
                eprintln!("skipped: {} on {name}: no amplified pipeline", case.name);
                continue;
            }
            match steps.resolve(case) {
                Ok((resolved, decoy)) => {
                    eprintln!(
                        "layered resolve {} on {name}: {resolved:08x?} (Metal's {metal:08x?})",
                        case.name
                    );
                    assert_eq!(&resolved, metal, "{} on {name}", case.name);
                    if case.late {
                        assert_eq!(decoy, vec![FILL, FILL], "{} on {name}", case.name);
                    }
                }
                Err(Error::Unsupported(u)) if case.memoryless => {
                    eprintln!("refused: {} on {name}: {u:?}", case.name);
                }
                Err(other) => panic!("{} on {name}: {other}", case.name),
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// One separate depth and stencil case: its name, the arrays' slices, the attachments' slice,
/// both aspects' load action, the layers drawn, and Metal's depth bits and stencils per slice.
type SeparateCase = (
    &'static str,
    u64,
    u64,
    LoadAction,
    &'static [u32],
    [&'static [u32]; 2],
);

/// A layered pass over separate depth and stencil textures tests and stores each layer of each
/// as Metal does (a probe, an M3 Max and an M4): it renders through a scratch image of the pass's
/// layers, each aspect loaded and stored over every layer, from the attachments' slice.
#[test]
fn separate_depth_and_stencil_render_each_layer_as_metal() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let steps = match Steps::new(device) {
            Ok(steps) => steps,
            Err(Error::Unsupported(u)) => {
                eprintln!("refused: layered depth and stencil on {name}: {u:?}");
                return;
            }
            Err(other) => panic!("layered depth and stencil on {name}: {other}"),
        };
        // The depth bits of 0.75 loaded, 0.5 drawn and 1.0 cleared.
        const LOADED: u32 = 0x3f40_0000;
        const DRAWN: u32 = 0x3f00_0000;
        const CLEARED: u32 = 0x3f80_0000;
        let cases: [SeparateCase; 4] = [
            (
                "sep_layered_load",
                2,
                0,
                LoadAction::Load,
                &[1],
                [&[LOADED, DRAWN], &[0x11, 0x33]],
            ),
            (
                "sep_layered_clear",
                2,
                0,
                LoadAction::Clear,
                &[0],
                [&[DRAWN, CLEARED], &[0x33, 0x22]],
            ),
            (
                "sep_layered_from1",
                3,
                1,
                LoadAction::Load,
                &[0],
                [&[LOADED, DRAWN, LOADED], &[0x11, 0x33, 0x11]],
            ),
            (
                "sep_layered_both",
                2,
                0,
                LoadAction::Clear,
                &[0, 1],
                [&[DRAWN, DRAWN], &[0x33, 0x33]],
            ),
        ];
        for (case, slices, slice, load, layers, [depth, stencil]) in cases {
            let (depths, stencils) = steps
                .separate(slices, slice, load, layers)
                .unwrap_or_else(|e| panic!("{case} on {name}: {e}"));
            eprintln!("layered depth and stencil {case} on {name}: {depths:08x?} {stencils:02x?}");
            assert_eq!(depths, depth, "{case} on {name}");
            assert_eq!(stencils, stencil, "{case} on {name}");
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// A layered pass over separate multisampled depth and stencil textures tests and stores each
/// layer of each, every sample, as the single-sampled ones do (a probe): where the device copies
/// multisampled aspects by draws (without maintenance8, or with the driver defect
/// `MultisampledDepthStencilColourCopies`) those run once per layer. Each slice is read back by a
/// pass of its own resolving sample 0 (every sample alike here). A device that refuses the
/// textures or the copies prints its typed refusal.
#[test]
fn separate_multisampled_depth_and_stencil_render_each_layer_as_metal() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let differ = {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let steps = match Steps::new(device) {
            Ok(steps) => steps,
            Err(Error::Unsupported(u)) => {
                eprintln!("refused: layered depth and stencil on {name}: {u:?}");
                return;
            }
            Err(other) => panic!("layered depth and stencil on {name}: {other}"),
        };
        const LOADED: u32 = 0x3f40_0000;
        const DRAWN: u32 = 0x3f00_0000;
        const CLEARED: u32 = 0x3f80_0000;
        // Each case's pass length beside it: one, a pass of one layer from the slice.
        let cases: [(u64, SeparateCase); 9] = [
            (
                2,
                (
                    "sep_samples_layered_load",
                    2,
                    0,
                    LoadAction::Load,
                    &[1],
                    [&[LOADED, DRAWN], &[0x11, 0x33]],
                ),
            ),
            (
                2,
                (
                    "sep_samples_layered_load0",
                    2,
                    0,
                    LoadAction::Load,
                    &[0],
                    [&[DRAWN, LOADED], &[0x33, 0x11]],
                ),
            ),
            (
                2,
                (
                    "sep_samples_layered_clear",
                    2,
                    0,
                    LoadAction::Clear,
                    &[0],
                    [&[DRAWN, CLEARED], &[0x33, 0x22]],
                ),
            ),
            (
                2,
                (
                    "sep_samples_layered_clear1",
                    2,
                    0,
                    LoadAction::Clear,
                    &[1],
                    [&[CLEARED, DRAWN], &[0x22, 0x33]],
                ),
            ),
            (
                2,
                (
                    "sep_samples_layered_from1",
                    3,
                    1,
                    LoadAction::Load,
                    &[0],
                    [&[LOADED, DRAWN, LOADED], &[0x11, 0x33, 0x11]],
                ),
            ),
            (
                2,
                (
                    "sep_samples_layered_both",
                    2,
                    0,
                    LoadAction::Clear,
                    &[0, 1],
                    [&[DRAWN, DRAWN], &[0x33, 0x33]],
                ),
            ),
            (
                2,
                (
                    "sep_samples_layered_undrawn",
                    2,
                    0,
                    LoadAction::Load,
                    &[],
                    [&[LOADED, LOADED], &[0x11, 0x11]],
                ),
            ),
            (
                1,
                (
                    "sep_samples_slice1_load",
                    2,
                    1,
                    LoadAction::Load,
                    &[0],
                    [&[LOADED, DRAWN], &[0x11, 0x33]],
                ),
            ),
            (
                1,
                (
                    "sep_samples_slice1_clear",
                    2,
                    1,
                    LoadAction::Clear,
                    &[0],
                    [&[LOADED, DRAWN], &[0x11, 0x33]],
                ),
            ),
        ];
        let mut differ = Vec::new();
        for (length, (case, slices, slice, load, layers, [depth, stencil])) in cases {
            let shape = (slices, slice, length);
            let (depths, stencils) = match steps.separate_multisampled(shape, load, layers) {
                Ok(read) => read,
                Err(error @ (Error::Unsupported(_) | Error::NotImplemented(_))) => {
                    eprintln!("refused: {case} on {name}: {error:?}");
                    continue;
                }
                Err(error) => panic!("{case} on {name}: {error}"),
            };
            eprintln!("layered samples {case} on {name}: {depths:08x?} {stencils:02x?}");
            if depths != depth || stencils != stencil {
                differ.push(format!(
                    "{case} on {name}: {depths:08x?} {stencils:02x?}, Metal's {depth:08x?} \
                     {stencil:02x?}"
                ));
            }
        }
        differ
    };
    layer_reported_nothing_at_all(&messages);
    assert!(differ.is_empty(), "{differ:#?}");
}

/// A pass of one layer resolves its depth and stencil from its attachment's slice into the resolve
/// texture's `resolveSlice` alone (a Metal probe on an M4: the resolve writes only the resolve
/// texture's `resolveLevel` and `resolveSlice`): from slice 1 of a 4-sample `Depth32Float_Stencil8`
/// array into slices 2 and 0 of a 3-slice one, and from slice 0 into slice 2, under each filter
/// pair of `a_layered_pass_resolves_its_depth_and_stencil_from_its_first_layer`. On lavapipe, whose
/// render pass resolves a depth/stencil attachment from its image's first slice into the resolve
/// image's first (`RuntimeDriverDefect::DepthStencilResolveFirstSubresource`), the runtime resolves
/// a copy of the attachment's slice into a scratch image copied into the resolve slice. A device
/// whose resolve modes lack a filter, where no draw takes it, prints its typed refusal.
#[test]
fn a_depth_and_stencil_resolve_reads_and_writes_their_own_slices() {
    use MultisampleDepthResolveFilter as D;
    use MultisampleStencilResolveFilter as S;
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let steps = match Steps::new(device) {
            Ok(steps) => steps,
            Err(Error::Unsupported(u)) => {
                eprintln!("refused: depth and stencil resolves on {name}: {u:?}");
                return;
            }
            Err(other) => panic!("depth and stencil resolves on {name}: {other}"),
        };
        const KEPT: u32 = 0x3f40_0000;
        const DRAWN: u32 = 0x3f00_0000;
        for filters in [
            (D::Sample0, S::Sample0),
            (D::Min, S::DepthResolvedSample),
            (D::Max, S::DepthResolvedSample),
        ] {
            for (slice, resolve_slice) in [(1u64, 2u64), (1, 0), (0, 2)] {
                let case = format!("{filters:?} from slice {slice} into slice {resolve_slice}");
                let resolved = steps.resolved_depth_stencil(filters, (slice, 1), resolve_slice);
                let (depths, stencils) = match resolved {
                    Ok(read) => read,
                    Err(error @ (Error::Unsupported(_) | Error::NotImplemented(_))) => {
                        eprintln!("refused: {case} on {name}: {error:?}");
                        continue;
                    }
                    Err(error) => panic!("{case} on {name}: {error}"),
                };
                eprintln!("resolve {case} on {name}: {depths:08x?} {stencils:02x?}");
                let at = |kept: u32, resolved: u32| -> Vec<u32> {
                    (0..3)
                        .map(|s| if s == resolve_slice { resolved } else { kept })
                        .collect()
                };
                assert_eq!(depths, at(KEPT, DRAWN), "{case} on {name}");
                assert_eq!(stencils, at(7, 0x33), "{case} on {name}");
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// A layered pass resolves its depth and stencil from its first layer alone, into slice
/// `resolveSlice`, the later layers not resolved: a probe (an M3 Max and an M4 alike, plain and
/// validated) drew layer 0 at 0.25 and layer 1 at 0.75 into a 4-sample `Depth32Float_Stencil8`
/// array and found every filter's resolve of layer 0 alone in slice `resolveSlice` of a 3-slice
/// array, the other slices keeping their texels. Here both layers are drawn at 0.5, layer 0's
/// stencil 0x33 and layer 1's 0x55, under `Sample0`, `Min` and `Max` with the stencil's `Sample0`
/// or `DepthResolvedSample` (drawn after the pass from the samples), at resolve slices 1 and 0. A
/// device whose resolve modes lack a filter, where no draw takes it, prints its typed refusal.
#[test]
fn a_layered_pass_resolves_its_depth_and_stencil_from_its_first_layer() {
    use MultisampleDepthResolveFilter as D;
    use MultisampleStencilResolveFilter as S;
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let steps = match Steps::new(device) {
            Ok(steps) => steps,
            Err(Error::Unsupported(u)) => {
                eprintln!("refused: layered depth and stencil on {name}: {u:?}");
                return;
            }
            Err(other) => panic!("layered depth and stencil on {name}: {other}"),
        };
        const KEPT: u32 = 0x3f40_0000;
        const DRAWN: u32 = 0x3f00_0000;
        for filters in [
            (D::Sample0, S::Sample0),
            (D::Min, S::DepthResolvedSample),
            (D::Max, S::DepthResolvedSample),
        ] {
            for resolve_slice in [1u64, 0] {
                let case = format!("{filters:?} at slice {resolve_slice}");
                let resolved = steps.resolved_depth_stencil(filters, (0, 2), resolve_slice);
                let (depths, stencils) = match resolved {
                    Ok(read) => read,
                    Err(error @ (Error::Unsupported(_) | Error::NotImplemented(_))) => {
                        eprintln!("refused: {case} on {name}: {error:?}");
                        continue;
                    }
                    Err(error) => panic!("{case} on {name}: {error}"),
                };
                eprintln!("layered resolve {case} on {name}: {depths:08x?} {stencils:02x?}");
                let at = |kept: u32, resolved: u32| -> Vec<u32> {
                    (0..3)
                        .map(|s| if s == resolve_slice { resolved } else { kept })
                        .collect()
                };
                assert_eq!(depths, at(KEPT, DRAWN), "{case} on {name}");
                assert_eq!(stencils, at(7, 0x33), "{case} on {name}");
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
}
