//! Device tests of vertex amplification on the device the cfg selects, under the validation layer.
//! The owned fixture `rt6_amplified` draws the fullscreen triangle at depth 0.5 once per instance
//! and amplification, and its fragment function returns `(i + 1) << 8 | a` for instance `i` and
//! amplification `a`. An `R32Uint` 2D array of 4 slices holds 0xEE in every slice first; a layered
//! pass from a slice clears its layers to 0x11 or loads them, and each draw sets (or not) the
//! encoder's amplification count and view mappings first. Each slice is compared with the texels
//! Metal stored for the same functions and passes (a Metal probe on an M4 with macOS 27.0.1;
//! another Metal probe on the M3 Max with macOS 26.7 and an M4 with macOS 27.0.1, alike; each with
//! and without API validation): amplification `a` renders the layer of the pass its render target
//! offset (plus the slice the vertex function chooses) names, below the pass's
//! `renderTargetArrayLength`, else the first; the viewport its viewport offset (plus the one the
//! function selects) names, below the encoder's viewport count, else the first, through that
//! viewport's scissor rectangle; a fragment function reads the sums as they are; a count of 1
//! applies the first mapping too, under a pipeline of a larger maximum count (one of maximum count
//! 1 takes none: Metal probes); and Metal renders every primitive of one amplification before the
//! next's. A `Depth32Float` array beside the color one, each slice cleared to 0.75 by a pass of its
//! own first, is cleared to 1.0 and tested `Less` with writes. A visibility result counts the
//! samples of every amplification.
//!
//! What multiview draws as Metal does is drawn as views; the rest one amplification at a time, and
//! so is every amplified draw while a visibility result mode is set. What Metal's validation stops
//! and plain Metal's answer does not settle is refused typed (`Invalid`): a count above a
//! pipeline's `maxVertexAmplificationCount` of 1, or above 8, at the draw
//! (`DrawProblem::AmplificationCount`), and fewer mappings than the count at the setter
//! (`DrawProblem::ViewMappings`); a count of 0 draws nothing and one above a larger maximum draws
//! that many amplifications, as plain Metal draws them. A device that draws one amplification at a
//! time announces it once ([`AMPLIFICATION`]); one that draws views announces nothing, but
//! `render-pass-split` where a pass's views leave out layers it clears or its draws render
//! different layers (`Case::split`). Each case's notices are checked as it runs (the sites).

use super::degrade_notices::{
    assert_made_once, assert_once, forced, made_render_kinds, shared_in_system_memory,
};
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device,
    validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::VkFeature;
use metal2vulkan::error::{Capability, Feature, Needs};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    CompareFunction, IndirectCommandType, PixelFormat, PrimitiveTopologyClass, PrimitiveType,
    TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, ClearColor, CommandBuffer, CommandBufferStatus, CommandQueue, Degradation,
    DepthStencilState, Device, DrawProblem, Error, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, Invalid, Library, LoadAction, Origin, PipelineOptions,
    Problem, RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, Size,
    StoreAction, Texture, TextureDescriptor, VertexAmplificationViewMapping, Viewport,
    VisibilityResultMode,
};
use std::collections::BTreeSet;
use std::path::Path;

/// The kind a draw of one amplification at a time takes, announced once per device.
pub(super) const AMPLIFICATION: [Degradation; 1] = [Degradation::AmplificationDraws];

/// The targets' width and height.
const SIDE: u64 = 4;

/// The slices of every target.
const SLICES: u64 = 4;

/// The word every color slice holds before a pass.
const FILLED: u32 = 0xee;

/// The color attachment's clear.
const CLEARED: u32 = 0x11;

/// The depth every slice of the depth array holds before a pass, and the pass's clear.
const DEPTHS: [f32; 2] = [0.75, 1.0];

/// One draw: the amplification count set before it (`None`: none set), its view mappings
/// (`(viewport offset, render target offset)` each; `None` for `nil`), the
/// `setVertexAmplificationMode:value:` after them (`None`: none), and its instances.
#[derive(Clone, Copy)]
struct Draw {
    count: Option<u64>,
    mappings: Option<&'static [(u32, u32)]>,
    mode: Option<(u64, u64)>,
    instances: u32,
}

impl Draw {
    /// The count the draw takes: the amplification mode's (the value's set bits in mode 1, the
    /// value in another mode) where one is set after the count, else the count set.
    fn amplification(&self) -> Option<u64> {
        match self.mode {
            Some((1, value)) => Some(u64::from(value.count_ones())),
            Some((_, value)) => Some(value),
            None => self.count,
        }
    }
}

/// Probe 13's default draw: two amplifications into the pass's first two layers.
const AMP2: Draw = Draw {
    count: Some(2),
    mappings: Some(&[(0, 0), (0, 1)]),
    mode: None,
    instances: 1,
};

/// Two amplifications with view mappings `mappings`.
const fn amp2(mappings: &'static [(u32, u32)]) -> Draw {
    Draw {
        mappings: Some(mappings),
        ..AMP2
    }
}

/// The viewports (and scissor rectangles) the encoder sets.
#[derive(Clone, Copy, PartialEq)]
enum Viewports {
    /// Metal's default: one viewport over the target.
    One,
    /// Two: the left half and the right half.
    Halves,
    /// Four: a column of one pixel each.
    Columns,
    /// The four columns, each with the scissor rectangle of its top two rows.
    ScissoredColumns,
}

/// Where the encoder sets a visibility result mode.
#[derive(Clone, Copy, PartialEq)]
enum Visibility {
    Off,
    /// Before the draws, at offset 0.
    Before(VisibilityResultMode),
    /// After the draws, at offset 8.
    After(VisibilityResultMode),
}

/// One pass, as the Metal probes on the M3 Max and an M4 drew it.
#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    length: u64,
    /// The attachments' slice.
    slice: u64,
    load: LoadAction,
    /// The pipeline's `maxVertexAmplificationCount`.
    max: u32,
    draws: &'static [Draw],
    /// A depth array beside the color one.
    depth: bool,
    vertex: &'static str,
    fragment: &'static str,
    /// The slice-choosing vertex function's layer per instance.
    layers: &'static [u32],
    viewports: Viewports,
    visibility: Visibility,
    /// Each draw is the one render command of an indirect command buffer, executed where the
    /// encoder would draw (its pipeline made with `supportIndirectCommandBuffers`).
    indirect: bool,
    /// The pass runs as more than one Vulkan render pass instance (`render-pass-split`): a draw
    /// drawn as views renders the layers of its instance alone (Vulkan multiview's view mask, or
    /// one layer a mapping moves a count of 1 to), so a pass that clears layers its first draw's
    /// views leave out clears them in an instance of its own, and a later draw of other views
    /// starts another.
    split: bool,
}

/// Probe 13's default case: a pass of length 2 from slice 0, cleared, of one [`AMP2`] draw.
const PLAIN: Case = Case {
    name: "",
    length: 2,
    slice: 0,
    load: LoadAction::Clear,
    max: 2,
    draws: &[AMP2],
    depth: false,
    vertex: "rt6_amplified_vertex",
    fragment: "rt6_amplified_fragment",
    layers: &[],
    viewports: Viewports::One,
    visibility: Visibility::Off,
    indirect: false,
    split: false,
};

/// What a pass stored: each color slice's texels, row by row; each depth slice's depth (every
/// texel alike; checked); and the visibility result at the mode's offset, where one was set.
#[derive(Debug, PartialEq)]
struct Stored {
    color: Vec<Vec<u32>>,
    depth: Vec<f32>,
    visibility: Option<u64>,
}

/// A slice whose rows are each `row`.
fn rows(row: [u32; 4]) -> Vec<u32> {
    row.repeat(SIDE as usize)
}

/// Every color slice alike in every texel, `color`, without depth or visibility.
fn colors(color: [u32; 4]) -> Stored {
    Stored {
        color: color.iter().map(|&word| rows([word; 4])).collect(),
        depth: Vec::new(),
        visibility: None,
    }
}

/// Each color slice's left half and right half alike, `(left, right)`.
fn halves(slices: [(u32, u32); 4]) -> Stored {
    Stored {
        color: slices.iter().map(|&(l, r)| rows([l, l, r, r])).collect(),
        ..colors([0; 4])
    }
}

/// Each color slice's columns, every row alike.
fn columns(slices: [[u32; 4]; 4]) -> Stored {
    Stored {
        color: slices.iter().map(|&row| rows(row)).collect(),
        ..colors([0; 4])
    }
}

/// A slice whose top two rows are `row` and whose others hold the pass's clear.
fn top(row: [u32; 4]) -> Vec<u32> {
    [row, row, [CLEARED; 4], [CLEARED; 4]].concat()
}

impl Case {
    /// Whether a draw of the case renders views of its own (a `ViewSetup` other than the
    /// default, `bind::ViewSetup`): an amplification count above 1 (multiview), or a count of 1
    /// whose first mapping moves it to a layer of the pass past its first, on a pipeline that
    /// takes mappings (a maximum count above 1). An indirect command that does records into a run
    /// of its own and what follows the execution into another (`render/executions.rs`), so its
    /// pass splits wherever the encoder's own draw would not.
    fn renders_views(&self) -> bool {
        self.draws.iter().any(|d| match d.amplification() {
            Some(n) if n > 1 => true,
            Some(1) => {
                self.max > 1
                    && d.mappings
                        .and_then(|m| m.first())
                        .is_some_and(|&(_, slice)| slice != 0 && u64::from(slice) < self.length)
            }
            _ => false,
        })
    }

    /// Whether a draw of the case runs one amplification at a time, where `drawn` says the case's
    /// amplified draws do, and announces `amplification-draws`: a count above 1 (a count of 1 is
    /// one draw either way).
    fn amplifies_singly(&self, drawn: Drawn) -> bool {
        drawn == Drawn::OneAtATime
            && self
                .draws
                .iter()
                .any(|d| d.amplification().unwrap_or(1) > 1)
    }
}

/// The device, its queue, the fixture's library, and the depth test.
struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
    less: DepthStencilState,
}

impl Lane {
    /// The lane on `device`; `Err` with the device's refusal of an amplified pipeline.
    fn new(device: Device) -> Result<Lane, Error> {
        let ll =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/rt6_amplified.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let mut less = DepthStencilDescriptor::default();
        less.depth_compare_function = CompareFunction::Less;
        less.depth_write_enabled = true;
        let lane = Lane {
            less: device.new_depth_stencil_state(&less).unwrap(),
            queue: device.new_command_queue().unwrap(),
            library,
            device,
        };
        lane.pipeline(&PLAIN)?;
        Ok(lane)
    }

    /// The pipeline `case` draws.
    fn pipeline(&self, case: &Case) -> Result<RenderPipelineState, Error> {
        self.device
            .new_render_pipeline_state(&self.descriptor(case), &PipelineOptions::default())
    }

    /// The kinds the making of `case`'s pipeline announces ([`made_render_kinds`]).
    fn made(&self, case: &Case) -> BTreeSet<Degradation> {
        made_render_kinds(&self.device, &self.descriptor(case))
    }

    /// The kinds the run of `case`, its draws `drawn`, announces where the device has not
    /// announced them: its pipeline's making's ([`Lane::made`]), its draws' of one amplification
    /// at a time ([`Case::amplifies_singly`]), its pass's split ([`Case::split`], or an indirect
    /// command's views, [`Case::renders_views`]), and those of its `Shared` buffers
    /// ([`shared_in_system_memory`]).
    fn forced(&self, case: &Case, drawn: Drawn) -> BTreeSet<Degradation> {
        let mut kinds = forced(&[
            (
                Degradation::AmplificationDraws,
                case.amplifies_singly(drawn),
            ),
            (
                Degradation::RenderPassSplit,
                case.split || (case.indirect && case.renders_views()),
            ),
            (
                Degradation::SharedInSystemMemory,
                shared_in_system_memory(&self.device),
            ),
            (
                Degradation::DeviceBooleanVisibility,
                matches!(
                    case.visibility,
                    Visibility::Before(VisibilityResultMode::Boolean)
                        | Visibility::After(VisibilityResultMode::Boolean)
                ),
            ),
        ]);
        kinds.extend(self.made(case));
        kinds
    }

    /// The descriptor of the pipeline `case` draws.
    fn descriptor(&self, case: &Case) -> RenderPipelineDescriptor {
        let function = |name: &str| {
            StageSource::Function(
                self.library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function(case.vertex));
        descriptor.fragment = Some(function(case.fragment));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
        if case.depth {
            descriptor.depth_format = Some(PixelFormat::Depth32Float);
        }
        descriptor.input_primitive_topology = PrimitiveTopologyClass::Triangle;
        descriptor.max_vertex_amplification_count = case.max;
        descriptor.support_indirect_command_buffers = case.indirect;
        descriptor
    }

    /// A buffer of one render command of `pipeline`, drawing the triangle's `instances`, its
    /// buffers inherited.
    fn indirect(
        &self,
        pipeline: &RenderPipelineState,
        instances: u32,
    ) -> Result<IndirectCommandBuffer, Error> {
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW;
        descriptor.inherit_buffers = true;
        let icb =
            self.device
                .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())?;
        let command = icb.indirect_render_command_at(0)?;
        command.set_render_pipeline_state(pipeline)?;
        command.draw_primitives(PrimitiveType::Triangle, 0, 3, u64::from(instances), 0)?;
        Ok(icb)
    }

    /// A private 2D array render target of `pf`.
    fn texture(&self, pf: PixelFormat) -> Texture {
        let mut descriptor = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
        descriptor.texture_type = TextureType::Type2DArray;
        descriptor.array_length = SLICES;
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
    fn fill(&self, texture: &Texture) {
        let words: Vec<u8> = (0..SIDE * SIDE)
            .flat_map(|_| FILLED.to_le_bytes())
            .collect();
        let source = buffer_with(&self.device, &words).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for slice in 0..SLICES {
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
    fn fill_depth(&self, texture: &Texture) {
        let command_buffer = self.queue.command_buffer().unwrap();
        for slice in 0..SLICES {
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

    /// Each slice's words of `texture`, row by row, read by a blit.
    fn slices(&self, texture: &Texture) -> Vec<Vec<u32>> {
        let bytes = SIDE * SIDE * 4;
        let out = buffer_with(&self.device, &vec![0u8; (bytes * SLICES) as usize]).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for slice in 0..SLICES {
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
        let words: Vec<u32> = read_buffer(&out, 0, bytes * SLICES)
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
            .map(|row| row.to_vec())
            .collect()
    }

    /// The pass of `case`, encoded and run; the encoder's or the draw's refusal.
    fn try_run(&self, case: &Case) -> Result<Stored, Error> {
        let pipeline = self.pipeline(case)?;
        let color = self.texture(PixelFormat::R32Uint);
        self.fill(&color);
        let depth = case.depth.then(|| self.texture(PixelFormat::Depth32Float));
        if let Some(depth) = &depth {
            self.fill_depth(depth);
        }
        let mut pass = RenderPassDescriptor::new();
        pass.render_target_array_length = case.length;
        let target = &mut pass.color_attachments[0];
        target.texture = Some(color.clone());
        target.slice = case.slice;
        target.load_action = case.load;
        target.clear_color = ClearColor::new(f64::from(CLEARED), 0.0, 0.0, 1.0);
        target.store_action = StoreAction::Store;
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
        // Metal's probe wrote 0xDEAD where the result goes first.
        let unwritten = [0xdead_u64.to_le_bytes(), 0xdead_u64.to_le_bytes()].concat();
        let results = buffer_with(&self.device, &unwritten).unwrap();
        if case.visibility != Visibility::Off {
            pass.visibility_result_buffer = Some(results.clone());
        }
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        encoder.set_render_pipeline_state(&pipeline)?;
        if depth.is_some() {
            encoder.set_depth_stencil_state(&self.less)?;
        }
        let side = SIDE as f64;
        match case.viewports {
            Viewports::One => {}
            Viewports::Halves => {
                let half = side / 2.0;
                encoder.set_viewports(&[
                    Viewport::new(0.0, 0.0, half, side, 0.0, 1.0),
                    Viewport::new(half, 0.0, half, side, 0.0, 1.0),
                ])?;
            }
            Viewports::Columns | Viewports::ScissoredColumns => {
                let columns: Vec<Viewport> = (0..SIDE)
                    .map(|x| Viewport::new(x as f64, 0.0, 1.0, side, 0.0, 1.0))
                    .collect();
                encoder.set_viewports(&columns)?;
            }
        }
        if case.viewports == Viewports::ScissoredColumns {
            let rects: Vec<ScissorRect> = (0..SIDE).map(|x| ScissorRect::new(x, 0, 1, 2)).collect();
            encoder.set_scissor_rects(&rects)?;
        }
        let words: Vec<u8> = case.layers.iter().flat_map(|l| l.to_le_bytes()).collect();
        let layers = buffer_with(&self.device, &words).unwrap();
        if !case.layers.is_empty() {
            encoder.set_vertex_buffer(Some(&layers), 0, 0)?;
        }
        if let Visibility::Before(mode) = case.visibility {
            encoder.set_visibility_result_mode(mode, 0)?;
        }
        let mut executed = Vec::new();
        for draw in case.draws {
            if let Some(count) = draw.count {
                let mappings: Option<Vec<VertexAmplificationViewMapping>> =
                    draw.mappings.map(|mappings| {
                        mappings
                            .iter()
                            .map(|&(viewport, slice)| {
                                VertexAmplificationViewMapping::new(viewport, slice)
                            })
                            .collect()
                    });
                encoder.set_vertex_amplification_count(count, mappings.as_deref())?;
            }
            if let Some((mode, value)) = draw.mode {
                encoder.set_vertex_amplification_mode(mode, value)?;
            }
            if case.indirect {
                let icb = self.indirect(&pipeline, draw.instances)?;
                encoder.execute_commands_in_buffer(&icb, 0..1)?;
                executed.push(icb);
            } else {
                encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, draw.instances, 0)?;
            }
        }
        if let Visibility::After(mode) = case.visibility {
            encoder.set_visibility_result_mode(mode, 8)?;
        }
        encoder.end_encoding()?;
        command_buffer.commit()?;
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        drop(executed);
        let visibility = match case.visibility {
            Visibility::Off => None,
            Visibility::Before(mode) => Some((0, mode)),
            Visibility::After(mode) => Some((8, mode)),
        }
        .map(|(offset, mode)| {
            let bytes = read_buffer(&results, offset, 8).unwrap();
            let word = u64::from_le_bytes(bytes.try_into().unwrap());
            // A `Boolean` word by its contract: the device's nonzero value where a sample passed.
            let booleans: &[usize] = match mode {
                VisibilityResultMode::Boolean => &[0],
                _ => &[],
            };
            super::visibility::by_contract(vec![word], booleans, 0xdead)[0]
        });
        drop(layers);
        Ok(Stored {
            color: self.slices(&color),
            depth: depth
                .map(|depth| {
                    self.slices(&depth)
                        .into_iter()
                        .map(|slice| {
                            assert!(slice.iter().all(|&w| w == slice[0]), "{slice:x?}");
                            f32::from_bits(slice[0])
                        })
                        .collect()
                })
                .unwrap_or_default(),
            visibility,
        })
    }
}

/// The lane on the test device, or `None` where it refuses an amplified pipeline (printed).
fn lane(device: Device) -> Option<Lane> {
    let name = device.name().to_owned();
    match Lane::new(device) {
        Ok(lane) => Some(lane),
        Err(Error::Unsupported(u)) => {
            eprintln!("refused: amplified pipelines on {name}: {u:?}");
            None
        }
        Err(other) => panic!("amplified pipelines on {name}: {other}"),
    }
}

/// How a test's cases draw: every one as views, or some one amplification at a time.
#[derive(Clone, Copy, PartialEq)]
enum Drawn {
    AsViews,
    OneAtATime,
}

impl Drawn {
    /// Run each of `cases` on the test device under the validation layer and compare what it
    /// stored with Metal's; a case the device refuses `Unsupported` is printed. The validation
    /// layer stays silent, and each case that runs announces exactly the kinds it forces that
    /// the device has not announced before ([`Lane::forced`]: [`AMPLIFICATION`] where it draws
    /// one amplification at a time, `render-pass-split` where its pass splits), and a refused
    /// one nothing.
    fn compare(self, cases: &[(Case, Stored)]) {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        {
            let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
            let name = device.name().to_owned();
            let Some(lane) = lane(device) else {
                return;
            };
            // The plain pipeline the lane makes first.
            assert_made_once(&messages, &[], &lane.made(&PLAIN));
            for (case, metal) in cases {
                match lane.try_run(case) {
                    Ok(stored) => {
                        assert_eq!(stored, *metal, "{} on {name}", case.name);
                        assert_made_once(&messages, &[], &lane.forced(case, self));
                    }
                    Err(Error::Unsupported(u)) => {
                        eprintln!("refused: {} on {name}: {u:?}", case.name);
                        assert_once(&messages.take_notices(), &[]);
                    }
                    Err(e) => panic!("{} on {name}: {e}", case.name),
                }
            }
        }
        layer_reported_nothing_at_all(&messages);
    }
}

const E: u32 = FILLED;
const C: u32 = CLEARED;

/// Each pass of probe 13 that multiview draws stores Metal's texels in every slice: the layers
/// the amplifications' offsets name drawn, the pass's other layers cleared or loaded, and nothing
/// past them; a count set back to 1 draws into the first layer, and a pipeline never amplified
/// draws amplification 0 there. Two viewports leave the views in viewport 0. A device that
/// refuses amplified pipelines (no `multiview`) prints its typed refusal.
#[test]
fn an_amplified_draw_renders_each_view_as_metal() {
    Drawn::AsViews.compare(&as_views());
}

/// Each draw of [`an_amplified_draw_renders_each_view_as_metal`]'s passes as the render command
/// of an indirect command buffer executed in its place stores the same texels, as Metal's did
/// (an amplified command rendered each view as the encoder's draw): the execution's commands that
/// render views of their own run in render pass instances of those views, runs of the execution
/// (`render/executions.rs`).
#[test]
fn amplified_indirect_render_commands_render_each_view_as_metal() {
    let cases: Vec<(Case, Stored)> = as_views()
        .into_iter()
        .map(|(case, stored)| {
            (
                Case {
                    indirect: true,
                    ..case
                },
                stored,
            )
        })
        .collect();
    Drawn::AsViews.compare(&cases);
}

/// The passes of probe 13 that multiview draws, and the texels Metal stored for each.
fn as_views() -> Vec<(Case, Stored)> {
    vec![
        (
            Case {
                name: "amp2_len2",
                ..PLAIN
            },
            colors([0x100, 0x101, E, E]),
        ),
        (
            Case {
                name: "amp2_len4_rto12",
                split: true,
                length: 4,
                draws: const { &[amp2(&[(0, 1), (0, 2)])] },
                ..PLAIN
            },
            colors([C, 0x100, 0x101, C]),
        ),
        (
            Case {
                name: "amp2_len2_slice1",
                slice: 1,
                ..PLAIN
            },
            colors([E, 0x100, 0x101, E]),
        ),
        (
            Case {
                name: "amp2_len3_two_instances",
                split: true,
                length: 3,
                draws: &[Draw {
                    instances: 2,
                    ..AMP2
                }],
                ..PLAIN
            },
            colors([0x200, 0x201, C, E]),
        ),
        (
            Case {
                name: "amp2_len4_load_rto23",
                length: 4,
                load: LoadAction::Load,
                draws: const { &[amp2(&[(0, 2), (0, 3)])] },
                ..PLAIN
            },
            colors([E, E, 0x100, 0x101]),
        ),
        (
            Case {
                name: "amp2_depth",
                depth: true,
                ..PLAIN
            },
            Stored {
                depth: vec![0.5, 0.5, DEPTHS[0], DEPTHS[0]],
                ..colors([0x100, 0x101, E, E])
            },
        ),
        (
            Case {
                name: "amp2_then_1",
                split: true,
                length: 4,
                draws: const {
                    &[
                        amp2(&[(0, 2), (0, 3)]),
                        Draw {
                            count: Some(1),
                            mappings: None,
                            mode: None,
                            instances: 2,
                        },
                    ]
                },
                ..PLAIN
            },
            colors([0x200, C, 0x100, 0x101]),
        ),
        (
            Case {
                name: "never_set",
                length: 4,
                draws: &[Draw {
                    count: None,
                    mappings: None,
                    mode: None,
                    instances: 1,
                }],
                ..PLAIN
            },
            colors([0x100, C, C, C]),
        ),
        (
            Case {
                name: "amp1_rto1",
                split: true,
                length: 4,
                draws: &[Draw {
                    count: Some(1),
                    mappings: Some(&[(0, 1)]),
                    mode: None,
                    instances: 1,
                }],
                ..PLAIN
            },
            colors([C, 0x100, C, C]),
        ),
        // A pipeline of maximum count 1 takes no mapping: neither its slice offset (a Metal probe)
        // nor its viewport offset (probe 6c).
        (
            Case {
                name: "amp1_rto1_pipeline1",
                length: 4,
                max: 1,
                draws: &[Draw {
                    count: Some(1),
                    mappings: Some(&[(0, 1)]),
                    mode: None,
                    instances: 1,
                }],
                ..PLAIN
            },
            colors([0x100, C, C, C]),
        ),
        (
            Case {
                name: "vp_amp1_pipeline1",
                max: 1,
                draws: &[Draw {
                    count: Some(1),
                    mappings: Some(&[(2, 0)]),
                    mode: None,
                    instances: 1,
                }],
                viewports: Viewports::Columns,
                ..PLAIN
            },
            columns([[0x100, C, C, C], [C; 4], [E; 4], [E; 4]]),
        ),
        (
            Case {
                name: "amp2_vpo_zero",
                viewports: Viewports::Halves,
                ..PLAIN
            },
            halves([(0x100, C), (0x101, C), (E, E), (E, E)]),
        ),
        // `setVertexAmplificationMode:value:` after the count sets its count, the value's set
        // bits in mode 1 and the value in another mode, and keeps the mappings (the consumer's
        // M3 Max measurements): as `amp2_len4_rto12`, and as a count of 1 into the first mapping.
        (
            Case {
                name: "mode1_two_bits_keeps_mappings",
                split: true,
                length: 4,
                draws: &[Draw {
                    count: Some(2),
                    mappings: Some(&[(0, 1), (0, 2)]),
                    mode: Some((1, 0b1010_0000)),
                    instances: 1,
                }],
                ..PLAIN
            },
            colors([C, 0x100, 0x101, C]),
        ),
        (
            Case {
                name: "mode5_value1_after_amp2",
                draws: &[Draw {
                    mode: Some((5, 1)),
                    ..AMP2
                }],
                ..PLAIN
            },
            colors([0x100, C, E, E]),
        ),
    ]
}

/// A count above a pipeline's `maxVertexAmplificationCount` of 2 or more, which Metal's validation
/// stops at the draw, draws that many amplifications, as plain Metal drew them (a Metal probe
/// `amp_3_on_2`, `amp_4_on_2`, `amp_4_on_3`, an M4 with macOS 27.0.1, the probe's functions): each
/// into the layer its mapping names, as views.
#[test]
fn a_count_above_the_pipeline_s_maximum_draws_as_metal() {
    const THREE: Draw = Draw {
        count: Some(3),
        mappings: Some(&[(0, 0), (0, 1), (0, 2)]),
        mode: None,
        instances: 1,
    };
    const FOUR: Draw = Draw {
        count: Some(4),
        mappings: Some(&[(0, 0), (0, 1), (0, 2), (0, 3)]),
        mode: None,
        instances: 1,
    };
    Drawn::AsViews.compare(&[
        (
            Case {
                name: "amp_3_on_2",
                split: true,
                length: 4,
                draws: &[THREE],
                ..PLAIN
            },
            colors([0x100, 0x101, 0x102, C]),
        ),
        (
            Case {
                name: "amp_4_on_2",
                length: 4,
                draws: &[FOUR],
                ..PLAIN
            },
            colors([0x100, 0x101, 0x102, 0x103]),
        ),
        (
            Case {
                name: "amp_4_on_3",
                length: 4,
                max: 3,
                draws: &[FOUR],
                ..PLAIN
            },
            colors([0x100, 0x101, 0x102, 0x103]),
        ),
    ]);
}

/// What multiview does not draw as Metal does draws one amplification at a time and stores Metal's
/// texels (a Metal probe on the M3 Max and an M4): every amplification into one layer (`nil`
/// mappings, the same offset twice, offsets past a pass of length 0, 1 or 2), the last drawn
/// winning; layers out of order or apart; viewport offsets, folded onto the first viewport and its
/// scissor rectangle at or past the viewport count; and a pipeline that reads or chooses its slice
/// amplified or moved, whose fragment function reads the slice and the viewport plus the offsets,
/// unfolded.
#[test]
fn what_multiview_does_not_express_draws_one_amplification_at_a_time() {
    let layered = Case {
        length: 4,
        draws: const { &[amp2(&[(0, 0), (0, 2)])] },
        vertex: "rt6_amplified_layered_vertex",
        fragment: "rt6_amplified_layered_fragment",
        ..PLAIN
    };
    Drawn::OneAtATime.compare(&[
        (
            Case {
                name: "amp2_nil",
                draws: &[Draw {
                    mappings: None,
                    ..AMP2
                }],
                ..PLAIN
            },
            colors([0x101, C, E, E]),
        ),
        (
            Case {
                name: "amp2_reversed",
                draws: const { &[amp2(&[(0, 1), (0, 0)])] },
                ..PLAIN
            },
            colors([0x101, 0x100, E, E]),
        ),
        // A count above the pipeline's maximum with `nil` mappings: every amplification into the
        // first layer, the last drawn winning (a Metal probe `amp_3_on_2_nil`).
        (
            Case {
                name: "amp_3_on_2_nil",
                length: 4,
                draws: &[Draw {
                    count: Some(3),
                    mappings: None,
                    mode: None,
                    instances: 1,
                }],
                ..PLAIN
            },
            colors([0x102, C, C, C]),
        ),
        // The same count set by `setVertexAmplificationMode:value:` in mode 0, no mappings set.
        (
            Case {
                name: "mode0_3_on_2_nil",
                length: 4,
                draws: &[Draw {
                    count: None,
                    mappings: None,
                    mode: Some((0, 3)),
                    instances: 1,
                }],
                ..PLAIN
            },
            colors([0x102, C, C, C]),
        ),
        (
            Case {
                name: "amp2_gap",
                length: 4,
                draws: const { &[amp2(&[(0, 0), (0, 2)])] },
                ..PLAIN
            },
            colors([0x100, C, 0x101, C]),
        ),
        (
            Case {
                name: "amp2_same",
                length: 4,
                draws: const { &[amp2(&[(0, 1), (0, 1)])] },
                ..PLAIN
            },
            colors([C, 0x101, C, C]),
        ),
        (
            Case {
                name: "amp2_past",
                draws: const { &[amp2(&[(0, 1), (0, 2)])] },
                ..PLAIN
            },
            colors([0x101, 0x100, E, E]),
        ),
        (
            Case {
                name: "amp2_past_far",
                draws: const { &[amp2(&[(0, 6), (0, 7)])] },
                ..PLAIN
            },
            colors([0x101, C, E, E]),
        ),
        (
            Case {
                name: "amp2_len1",
                length: 1,
                slice: 1,
                ..PLAIN
            },
            colors([E, 0x101, E, E]),
        ),
        (
            Case {
                name: "amp2_len0",
                length: 0,
                slice: 1,
                ..PLAIN
            },
            colors([E, 0x101, E, E]),
        ),
        (
            Case {
                name: "amp2_vpo",
                draws: const { &[amp2(&[(0, 0), (1, 1)])] },
                viewports: Viewports::Halves,
                ..PLAIN
            },
            halves([(0x100, C), (C, 0x101), (E, E), (E, E)]),
        ),
        (
            Case {
                name: "amp2_slice_read",
                length: 4,
                draws: const { &[amp2(&[(0, 1), (0, 2)])] },
                fragment: "rt6_amplified_slice_fragment",
                ..PLAIN
            },
            colors([C, 0x101, 0x112, C]),
        ),
        (
            Case {
                name: "amp1_slice_read",
                length: 4,
                draws: &[Draw {
                    count: Some(1),
                    mappings: Some(&[(0, 2)]),
                    mode: None,
                    instances: 1,
                }],
                fragment: "rt6_amplified_slice_fragment",
                ..PLAIN
            },
            colors([C, C, 0x102, C]),
        ),
        (
            Case {
                name: "amp2_vertex_layer",
                layers: &[1],
                ..layered
            },
            colors([C, 0x101, C, 0x113]),
        ),
        (
            Case {
                name: "amp2_vertex_layer_past",
                layers: &[3],
                ..layered
            },
            colors([0x115, C, C, 0x103]),
        ),
        (
            Case {
                name: "vp_stereo",
                draws: const { &[amp2(&[(0, 0), (1, 0)])] },
                viewports: Viewports::Halves,
                ..PLAIN
            },
            halves([(0x100, 0x101), (C, C), (E, E), (E, E)]),
        ),
        (
            Case {
                name: "vp_stereo_columns",
                draws: const { &[amp2(&[(1, 0), (3, 0)])] },
                viewports: Viewports::Columns,
                ..PLAIN
            },
            columns([[C, 0x100, C, 0x101], [C; 4], [E; 4], [E; 4]]),
        ),
        (
            Case {
                name: "vp_past_halves",
                draws: const { &[amp2(&[(0, 0), (2, 1)])] },
                viewports: Viewports::Halves,
                ..PLAIN
            },
            halves([(0x100, C), (0x101, C), (E, E), (E, E)]),
        ),
        (
            Case {
                name: "vp_past_far",
                draws: const { &[amp2(&[(3, 0), (17, 1)])] },
                viewports: Viewports::Columns,
                ..PLAIN
            },
            columns([[C, C, C, 0x100], [0x101, C, C, C], [E; 4], [E; 4]]),
        ),
        (
            Case {
                name: "vp_past_same_layer",
                draws: const { &[amp2(&[(1, 0), (2, 0)])] },
                viewports: Viewports::Halves,
                ..PLAIN
            },
            halves([(0x101, 0x100), (C, C), (E, E), (E, E)]),
        ),
        (
            Case {
                name: "vp_default",
                draws: const { &[amp2(&[(0, 0), (1, 1)])] },
                ..PLAIN
            },
            colors([0x100, 0x101, E, E]),
        ),
        (
            Case {
                name: "vp_amp1",
                draws: &[Draw {
                    count: Some(1),
                    mappings: Some(&[(2, 0)]),
                    mode: None,
                    instances: 1,
                }],
                viewports: Viewports::Columns,
                ..PLAIN
            },
            columns([[C, C, 0x100, C], [C; 4], [E; 4], [E; 4]]),
        ),
        (
            Case {
                name: "vp_scissor",
                draws: const { &[amp2(&[(0, 0), (2, 1)])] },
                viewports: Viewports::ScissoredColumns,
                ..PLAIN
            },
            Stored {
                color: vec![
                    top([0x100, C, C, C]),
                    top([C, C, 0x101, C]),
                    rows([E; 4]),
                    rows([E; 4]),
                ],
                ..colors([0; 4])
            },
        ),
        (
            Case {
                name: "vp_scissor_past",
                draws: const { &[amp2(&[(1, 0), (5, 1)])] },
                viewports: Viewports::ScissoredColumns,
                ..PLAIN
            },
            Stored {
                color: vec![
                    top([C, 0x100, C, C]),
                    top([0x101, C, C, C]),
                    rows([E; 4]),
                    rows([E; 4]),
                ],
                ..colors([0; 4])
            },
        ),
    ]);
}

/// A visibility result over amplified draws counts the samples of every amplification, as one query
/// over draws of one amplification each does (a Metal probe on the M3 Max and an M4: 16 for one
/// amplification of the 16-pixel target, 32 for two into two layers or into one, and with a depth
/// test; 1 for `Boolean`, the device's nonzero value read by its contract). A mode set after a draw
/// amplified as views leaves its texels as Metal's.
#[test]
fn a_visibility_result_counts_every_amplification() {
    use VisibilityResultMode::{Boolean, Counting};
    let counted = |stored: Stored, samples| Stored {
        visibility: Some(samples),
        ..stored
    };
    Drawn::OneAtATime.compare(&[
        (
            Case {
                name: "vis_count_single",
                draws: &[Draw {
                    count: Some(1),
                    mappings: Some(&[(0, 0)]),
                    mode: None,
                    instances: 1,
                }],
                visibility: Visibility::Before(Counting),
                ..PLAIN
            },
            counted(colors([0x100, C, E, E]), 16),
        ),
        (
            Case {
                name: "vis_count_views",
                visibility: Visibility::Before(Counting),
                ..PLAIN
            },
            counted(colors([0x100, 0x101, E, E]), 32),
        ),
        (
            Case {
                name: "vis_count_nil",
                draws: &[Draw {
                    mappings: None,
                    ..AMP2
                }],
                visibility: Visibility::Before(Counting),
                ..PLAIN
            },
            counted(colors([0x101, C, E, E]), 32),
        ),
        (
            Case {
                name: "vis_count_views_depth",
                depth: true,
                visibility: Visibility::Before(Counting),
                ..PLAIN
            },
            counted(
                Stored {
                    depth: vec![0.5, 0.5, DEPTHS[0], DEPTHS[0]],
                    ..colors([0x100, 0x101, E, E])
                },
                32,
            ),
        ),
        (
            Case {
                name: "vis_bool_views",
                visibility: Visibility::Before(Boolean),
                ..PLAIN
            },
            counted(colors([0x100, 0x101, E, E]), 1),
        ),
    ]);
    // A mode set after the views: Metal's texels, and the query runs (its count is not
    // compared: Metal's for a query over no draw is not measured).
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let Some(lane) = lane(device) else {
            return;
        };
        let after = Case {
            name: "amp2_visibility_after",
            visibility: Visibility::After(Boolean),
            ..PLAIN
        };
        let stored = lane
            .try_run(&after)
            .unwrap_or_else(|e| panic!("{} on {name}: {e}", after.name));
        assert_eq!(stored.color, colors([0x100, 0x101, E, E]).color, "{name}");
    }
    layer_reported_nothing_at_all(&messages);
}

/// What Metal's validation stops and plain Metal's answer does not settle is `Invalid` (a Metal
/// probe on an M4: "Vertex Amplification Count (3) must be between (inclusive) 1 and the maximum
/// vertex amplification count specified in the pipeline state (2)"; the M3 Max alike): at the draw
/// a count above a pipeline's maximum of 1 (plain Metal stored garbage, probe case
/// `amp2_pipeline1`, a probe `amp_3_on_1`) or above Metal's 8 (unmeasured), and at the setter too
/// few mappings. A count of 0 draws nothing, as plain Metal draws it, and one above a larger
/// maximum draws (`a_count_above_the_pipeline_s_maximum_draws_as_metal`). Each refusal records
/// nothing, and the validation layer stays silent.
#[test]
fn an_amplified_draw_is_refused_as_typed() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let Some(lane) = lane(device) else {
            return;
        };
        let count = |count, max| {
            Error::Invalid(Invalid::new(
                "draw_primitives",
                Problem::Draw(DrawProblem::AmplificationCount { count, max }),
            ))
        };
        let invalid = [
            (
                Case {
                    name: "amp2_pipeline1",
                    max: 1,
                    ..PLAIN
                },
                count(2, 1),
            ),
            (
                Case {
                    name: "amp3_pipeline1",
                    length: 4,
                    max: 1,
                    draws: &[Draw {
                        count: Some(3),
                        mappings: Some(&[(0, 0), (0, 1), (0, 2)]),
                        mode: None,
                        instances: 1,
                    }],
                    ..PLAIN
                },
                count(3, 1),
            ),
            (
                Case {
                    name: "amp9_pipeline2",
                    length: 4,
                    draws: &[Draw {
                        count: Some(9),
                        mappings: None,
                        mode: None,
                        instances: 1,
                    }],
                    ..PLAIN
                },
                count(9, 2),
            ),
            (
                Case {
                    name: "amp2_one_mapping",
                    draws: const { &[amp2(&[(0, 0)])] },
                    ..PLAIN
                },
                Error::Invalid(Invalid::new(
                    "set_vertex_amplification_count",
                    Problem::Draw(DrawProblem::ViewMappings { count: 2, given: 1 }),
                )),
            ),
        ];
        for (case, refusal) in &invalid {
            assert_eq!(
                lane.try_run(case).unwrap_err(),
                *refusal,
                "{} on {name}",
                case.name
            );
        }
        // A count of 0, which Metal's validation stops too, draws nothing, as plain Metal draws
        // it (probe 13 `amp0`: both layers keep their clear).
        let amp0 = Case {
            name: "amp0",
            draws: &[Draw {
                count: Some(0),
                mappings: None,
                mode: None,
                instances: 1,
            }],
            ..PLAIN
        };
        let stored = lane
            .try_run(&amp0)
            .unwrap_or_else(|e| panic!("amp0 on {name}: {e}"));
        assert_eq!(stored, colors([C, C, E, E]), "amp0 on {name}");
    }
    layer_reported_nothing_at_all(&messages);
}

/// On a device without `multiViewport` (withheld here, with `shaderOutputViewportIndex`) a draw has
/// one viewport, so an amplification's viewport offset selects viewport 0 whatever it is, as Metal
/// draws an offset at or past the viewports set, and the variant stores no `ViewportIndex`: under
/// Metal's one default viewport the offsets 0 and 1 (`vp_default`) store Metal's texels. The same
/// draw while the encoder holds two viewports (`amp2_vpo`), where Metal selects the second, is
/// refused, `Unsupported(LayeredRendering)` needing `multiViewport`. The device announces
/// [`AMPLIFICATION`] once, and the validation layer stays silent.
#[test]
fn with_one_viewport_an_amplification_s_viewport_offset_draws_through_it() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.withheld_features.extend([
                VkFeature::MultiViewport,
                VkFeature::ShaderOutputViewportIndex,
            ]);
        })
        .unwrap();
        let name = device.name().to_owned();
        let Some(lane) = lane(device) else {
            return;
        };
        let vp_default = Case {
            name: "vp_default",
            draws: const { &[amp2(&[(0, 0), (1, 1)])] },
            ..PLAIN
        };
        let stored = lane
            .try_run(&vp_default)
            .unwrap_or_else(|e| panic!("vp_default on {name}: {e}"));
        assert_eq!(stored, colors([0x100, 0x101, E, E]), "vp_default on {name}");
        let amp2_vpo = Case {
            name: "amp2_vpo",
            draws: const { &[amp2(&[(0, 0), (1, 1)])] },
            viewports: Viewports::Halves,
            ..PLAIN
        };
        assert_eq!(
            lane.try_run(&amp2_vpo).unwrap_err(),
            Error::Unsupported(metal2vulkan::error::Unsupported::needs(
                Feature::LayeredRendering,
                Capability::Feature(VkFeature::MultiViewport),
            )),
            "amp2_vpo on {name}"
        );
        let mut first = forced(&[(
            Degradation::SharedInSystemMemory,
            shared_in_system_memory(&lane.device),
        )]);
        first.extend(lane.made(&PLAIN));
        assert_made_once(&messages, &AMPLIFICATION, &first);
    }
    layer_reported_nothing_at_all(&messages);
}

/// A device without `shaderOutputLayer` (withheld here: no committed real GPU lacks it) draws an
/// amplified pass as Metal or refuses it typed, never into another slice: what multiview draws
/// (`amp2_len2`) and one amplification at a time into a pass of one layer (`amp2_len1`, whose
/// variant writes no `Layer`) store Metal's texels; one amplification at a time into a pass of
/// more layers (`amp2_reversed`, `amp2_gap`), whose variant must store each amplification's
/// slice as `Layer`, stores Metal's texels through the synthesized geometry stages on a device
/// with `geometryShader`, and is refused `Unsupported(VertexAmplification)` needing either
/// feature on one without. The device announces [`AMPLIFICATION`] once, and the validation layer
/// stays silent.
#[test]
fn a_device_without_shader_output_layer_draws_or_refuses_typed() {
    let feature = VkFeature::ShaderOutputLayer;
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.withheld_features.insert(feature);
        })
        .unwrap();
        assert!(!device.enabled().features.contains(&feature));
        let name = device.name().to_owned();
        let Some(lane) = lane(device) else {
            return;
        };
        let drawn = [
            (
                Case {
                    name: "amp2_len2",
                    ..PLAIN
                },
                colors([0x100, 0x101, E, E]),
            ),
            (
                Case {
                    name: "amp2_len1",
                    length: 1,
                    slice: 1,
                    ..PLAIN
                },
                colors([E, 0x101, E, E]),
            ),
        ];
        for (case, metal) in &drawn {
            let stored = lane
                .try_run(case)
                .unwrap_or_else(|e| panic!("{} on {name}: {e}", case.name));
            assert_eq!(stored, *metal, "{} on {name}", case.name);
        }
        // Metal's texels (`what_multiview_does_not_express_draws_one_amplification_at_a_time`).
        let layered = [
            (
                Case {
                    name: "amp2_reversed",
                    draws: const { &[amp2(&[(0, 1), (0, 0)])] },
                    ..PLAIN
                },
                colors([0x101, 0x100, E, E]),
            ),
            (
                Case {
                    name: "amp2_gap",
                    length: 4,
                    draws: const { &[amp2(&[(0, 0), (0, 2)])] },
                    ..PLAIN
                },
                colors([0x100, C, 0x101, C]),
            ),
        ];
        let geometry = lane
            .device
            .enabled()
            .features
            .contains(&VkFeature::GeometryShader);
        for (case, metal) in &layered {
            match (lane.try_run(case), geometry) {
                (Ok(stored), true) => assert_eq!(stored, *metal, "{} on {name}", case.name),
                (Err(Error::Unsupported(u)), false) => {
                    assert_eq!(u.feature, Feature::VertexAmplification, "{}", case.name);
                    assert_eq!(
                        u.needs,
                        Needs::AnyOf(vec![
                            vec![Capability::Feature(feature)],
                            vec![Capability::Feature(VkFeature::GeometryShader)],
                        ]),
                        "{} on {name}",
                        case.name
                    );
                }
                (other, _) => panic!("{} on {name}: {:?}", case.name, other.map(|_| ())),
            }
        }
        // The plain pipeline's making, the pass of one layer's draws of one amplification at a
        // time, and the `Shared` buffers; the refused passes announce nothing.
        let mut first = forced(&[(
            Degradation::SharedInSystemMemory,
            shared_in_system_memory(&lane.device),
        )]);
        first.extend(lane.made(&PLAIN));
        assert_made_once(&messages, &AMPLIFICATION, &first);
    }
    layer_reported_nothing_at_all(&messages);
}
