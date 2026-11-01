//! A draw's shape: what it derives from the encoder's state alone, not from what its tables bind
//! (its settings, [`ShapeInputs::settings`]; its draw state, [`ShapeInputs::draw_state`]; and its
//! `VkPipeline`), and the memo that keeps it from one draw of the encoder to the next
//! ([`ShapeMemo`]).
//!
//! - **Inputs.** The derivation reads [`ShapeInputs`] alone: the device, the pass's layout, the
//!   encoder's state and vertex amplification, whether a visibility query counts the draws, the
//!   pass's rasterization-rate map, and the routing the encoder kept for its drawing state (a
//!   cache of a function of the state and the pass's colors, which changes no piece). Beside them
//!   each piece takes the draw's own: its drawing state and assembly; for the draw state its
//!   variant's compiled state and what the draw's failing depth and shader-side tests make of it;
//!   for the pipeline its graphics key (`DrawKey`) and variant.
//! - **Reuse.** A draw takes a piece an earlier draw of the encoder derived where every input is
//!   the same: the drawing state and variant by identity (each held), the assembly and whether a
//!   query counts by value, and the layout, the state and the amplification by version
//!   (`versioned.rs`: any mutable access moves it); its pipeline where each field of its key is,
//!   by value or by what it is made of ([`PipelineInputs`]). The device and the rate map do not
//!   change in an encoder, whose memo this is; what the pieces read of the device is fixed at its
//!   creation (its profile, its depth ranges, its depth bias command, its region limits). Only a
//!   piece derived without a refusal is kept, and each is taken at the step of the draw that
//!   derives it, so a refused draw is refused by the same step as before.
//! - **Notices.** A derivation announces its notices once per device (`Device::degrade`): a
//!   reused piece's were announced as it was derived. A piece whose derivation met a notice the
//!   device announces at every occurrence (`DeviceOptions::announce_each`), or refuses at every
//!   occurrence while it lowers the call (`DeviceOptions::lowering_refuses`), is not kept, so
//!   each draw derives it anew and meets it again (`degradation::EachWatch`, which watches the
//!   derivation itself, not a list of the kinds it may announce). A device that announces at
//!   every occurrence only kinds a draw's shape does not meet reuses shapes as one that announces
//!   each kind once.
//! - **What is not kept.** The pass's hidden depth and stencil a draw gives it, made for that
//!   draw before its settings are looked up (`RenderCommandEncoder::needed_hidden`): the settings
//!   take only whether it gives them. Nor a pipeline still being made, nor a residual form a draw
//!   takes meanwhile (`render/residual.rs`), nor a mesh draw's, of its own specialization.
//! - **Checked.** A debug build derives reused settings and draw states again from the same
//!   inputs and asserts them equal; a reused pipeline's key is compared whole.
//!   `WorkCounts::draw_shapes_reused` counts the draws that reused all three.

use super::draw_problem;
use super::line_width::LineRoute;
use super::provoking;
use super::{Assembly, DrawSettings, DrawState, Routed, RoutedColors};
use crate::amplification::Amplification;
use crate::color_reformat::{self, Routing};
use crate::degradation::Degradation;
use crate::depth_stencil::FailingDraw;
use crate::error::{internal, Error, RuntimeGap};
use crate::pipeline::{FixedState, GraphicsState as DrawingState};
use crate::pipeline::{RawPipeline, RenderVariant};
use crate::rate_mapped_pass::RateMappedPass;
use crate::render_pass::{PassLayout, HIDDEN_DEPTH_FORMAT};
use crate::render_state::{self, DepthBiasRecord, EncoderState, ViewportList};
use crate::render_state::{Compensation, DepthBiasEffect, WrittenDepthClamp};
use crate::render_target::RenderPassObject;
use crate::render_values::Viewport;
use crate::Device;
use metal2vulkan::bind::BindingPlan;
use metal2vulkan::metal::{PixelFormat, PrimitiveTopologyClass, PrimitiveType};
use metal2vulkan::pipeline::{GraphicsState, Stage};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// The encoder's part of what a draw's shape is derived from (the module docs):
/// `RenderCommandEncoder::shape_inputs`.
pub(super) struct ShapeInputs<'a> {
    pub(super) device: &'a Device,
    pub(super) layout: &'a PassLayout,
    pub(super) state: &'a EncoderState,
    pub(super) amplification: &'a Amplification,
    /// Whether the pass's draws draw as a visibility query counts them
    /// (`RenderCommandEncoder::counts_visibility`).
    pub(super) counts_visibility: bool,
    pub(super) rate_map: Option<&'a RateMappedPass>,
    /// The routing the encoder kept as its drawing state was set: a cache of a function of the
    /// state and the pass's colors alone ([`routing`](Self::routing)).
    pub(super) routed: Option<&'a Routed>,
    /// The versions of `layout`, `state` and `amplification` (`versioned.rs`).
    pub(super) versions: [u64; 3],
}

/// Whether a draw of `pipeline` writes its colors: a fragment function, rasterizing.
pub(super) fn writes_colors(pipeline: &DrawingState) -> bool {
    pipeline.descriptor().stage(Stage::Fragment).is_some()
        && !pipeline.base().graphics.rasterizer_discard
}

impl ShapeInputs<'_> {
    /// How a draw of `pipeline` stores its colors into the pass's (`color_reformat::Routing`):
    /// each state color into the pass color at its place in Metal's per-sample record, for a
    /// state that writes colors (Metal probes); each color at its own index for one that writes
    /// none. `Err(index)`, refused `AttachmentFormat(Color(index))`, where plain Metal stores part
    /// of a color, or more than one, into an attachment (`color_reformat::routing`), `index` the
    /// lowest color that moves. The one the encoder kept as `pipeline` was set
    /// (`RenderCommandEncoder::set_render_pipeline_state`) where the pass's colors
    /// are still the ones it was found for, else found again.
    pub(super) fn routing(&self, pipeline: &DrawingState) -> Result<Arc<Routing>, u32> {
        let pass = self.layout.pass_formats();
        match self.routed {
            Some(routed) if routed.state.same(pipeline) && routed.pass == pass => {
                routed.routing.clone()
            }
            _ => pipeline.routed(&pass, || Self::route(pipeline, &pass)),
        }
    }

    /// [`routing`](Self::routing), found for a pass of the colors `pass`.
    pub(super) fn route(
        pipeline: &DrawingState,
        pass: &[Option<PixelFormat>],
    ) -> Result<Routing, u32> {
        let state = pipeline.descriptor().color_formats();
        if !writes_colors(pipeline) {
            return Ok(Routing::by_index(&state, pass));
        }
        match color_reformat::routing(&state, pass) {
            Some(routing) => Ok(routing),
            None => match color_reformat::misplaced(&state, pass) {
                Some(index) => Err(index),
                None => Ok(Routing::by_index(&state, pass)),
            },
        }
    }

    /// The pass colors a draw of `pipeline`, routed by `routing`, zeroes and the colors of its own
    /// it drops: those the encoder kept as `pipeline` was set where the pass's colors are still
    /// the ones they were found for (a function of the state and those colors alone, as `routing`
    /// is), else found again.
    pub(super) fn routed_colors(
        &self,
        pipeline: &DrawingState,
        routing: &Routing,
    ) -> Arc<RoutedColors> {
        match self.routed {
            Some(Routed {
                state,
                pass,
                colors: Some(colors),
                ..
            }) if state.same(pipeline) && *pass == self.layout.pass_formats() => Arc::clone(colors),
            _ => Arc::new(self.colors_by(pipeline, routing)),
        }
    }

    /// [`zeroed_colors`](Self::zeroed_colors) and [`dropped_colors`](Self::dropped_colors) of a
    /// draw of `pipeline` routed by `routing`.
    pub(super) fn colors_by(&self, pipeline: &DrawingState, routing: &Routing) -> RoutedColors {
        let zeroed = self.zeroed_colors(pipeline, routing);
        let dropped = self.dropped_colors(pipeline, routing, zeroed.is_empty());
        RoutedColors { zeroed, dropped }
    }

    /// The pass's colors at no place of `pipeline`'s record (`routing`), the pass's format of
    /// each by the location the draw references it at, where plain Metal zeroes them at every
    /// sample a draw covers (`color_reformat::Uncovered::Zeroed`; Metal probes, probes): a draw
    /// runs the variant whose fragment function writes the zero (`BindShape::zeroed_colors`). Empty
    /// where Metal keeps them (a state color blended reading the destination, or writing some of
    /// its channels alone), and for a mesh render pipeline's state and a state that writes no color
    /// (what Metal stores there is unmeasured), which are refused such a pass.
    pub(super) fn zeroed_colors(
        &self,
        pipeline: &DrawingState,
        routing: &Routing,
    ) -> BTreeMap<u32, PixelFormat> {
        let descriptor = pipeline.descriptor();
        let colors = descriptor.color_attachments();
        let zeroed = color_reformat::uncovered(colors.iter().flatten());
        if descriptor.vertex().is_none()
            || !writes_colors(pipeline)
            || zeroed != color_reformat::Uncovered::Zeroed
        {
            return BTreeMap::new();
        }
        routing
            .uncovered
            .iter()
            .filter_map(|(&location, &index)| {
                let color = self.layout.colors.get(index as usize)?.as_ref()?;
                Some((location, color.target.texture.pixel_format()))
            })
            .collect()
    }

    /// The locations where `pipeline`'s fragment function's writes plain Metal drops (a Metal
    /// probe): a draw runs the variant whose fragment function keeps no output there
    /// (`BindShape::dropped_colors`), so its pipeline writes no location without an attachment,
    /// and none into a pass color it keeps. They are the state's colors at no place of the pass's
    /// record (`routing`); a pass color the draw keeps (`none_zeroed`: its `zeroed_colors` names
    /// none), at whose location the state has no color; and in a routed draw each location its
    /// subpass leaves out where the state has no color, which the function may write all the same.
    /// Empty for a state without a fragment function, which writes no color.
    fn dropped_colors(
        &self,
        pipeline: &DrawingState,
        routing: &Routing,
        none_zeroed: bool,
    ) -> BTreeSet<u32> {
        let descriptor = pipeline.descriptor();
        if descriptor.vertex().is_some_and(|d| d.fragment.is_none()) {
            return BTreeSet::new();
        }
        let state = descriptor.color_formats();
        let lacks = |location: u32| {
            state
                .get(location as usize)
                .copied()
                .flatten()
                .is_none_or(|pf| pf == PixelFormat::Invalid)
        };
        let mut dropped = routing.dropped.clone();
        if writes_colors(pipeline) && none_zeroed {
            dropped.extend(routing.uncovered.keys().copied().filter(|&l| lacks(l)));
        }
        let left_out = (0u32..).zip(&routing.locations);
        dropped.extend(
            left_out
                .filter(|&(l, index)| index.is_none() && lacks(l))
                .map(|(l, _)| l),
        );
        dropped
    }

    /// What a draw of `pipeline` as `assembly` settles from the encoder's state before its bind
    /// state ([`DrawSettings`]), as a draw settles it and as an execution's precompile at commit
    /// does (`render/executions.rs`), `hidden` where the draw gives the pass its hidden depth and
    /// stencil (`RenderCommandEncoder::needed_hidden`). Where the draw keeps Vulkan's clipping of
    /// points, or a depth the fragment function writes past Metal's clamp, that is announced
    /// once where `announce`: the precompile, which draws nothing, announces nothing. Refused
    /// as `call`'s draw is.
    pub(super) fn settings(
        &self,
        call: &'static str,
        (pipeline, assembly): (&DrawingState, Assembly),
        hidden: bool,
        announce: bool,
    ) -> Result<DrawSettings, Error> {
        let meshes = pipeline.descriptor().vertex().is_none();
        // A draw of another class than the pipeline's input primitive topology class, which
        // Metal's validation stops, draws its own primitives, as plain Metal draws them (a probe).
        // Points rasterize nothing unless the vertex function writes a point size or the class is
        // the point class, which sizes them 1 (probes 1 and 2, the vertex function run for every
        // point either way): the draw runs its vertex function alone, through empty scissors.
        let class = pipeline
            .descriptor()
            .vertex()
            .map_or(PrimitiveTopologyClass::Unspecified, |d| {
                d.input_primitive_topology
            });
        let unsized_points = matches!(assembly, Assembly::Primitives(PrimitiveType::Point))
            && class != PrimitiveTopologyClass::Point
            && !pipeline.base().graphics.writes_point_size;
        let (topology, primitive_restart) = match assembly {
            Assembly::Primitives(primitive_type) => {
                metal2vulkan::metal::state::input_assembly(primitive_type)
            }
            // A private type draws its program's list, without restart (`render/restart.rs`).
            Assembly::Private(program) => match program.vertices() {
                2 => (metal2vulkan::vk::PrimitiveTopology::LINE_LIST, false),
                _ => (metal2vulkan::vk::PrimitiveTopology::TRIANGLE_LIST, false),
            },
            // An emulated tessellation pipeline draws the tessellator's triangles as a list,
            // which Vulkan biases as Metal biases them.
            Assembly::Patches => match pipeline.base().graphics.emulated_tessellation {
                Some(_) => (metal2vulkan::vk::PrimitiveTopology::TRIANGLE_LIST, false),
                None => (metal2vulkan::vk::PrimitiveTopology::PATCH_LIST, false),
            },
            // A mesh pipeline has no input assembly. Its key takes a triangle list, which
            // enables the depth bias Metal applies to triangles: Vulkan biases a mesh function's
            // triangles and none of its lines or points, as Metal does. An emulated mesh
            // function's synthesized vertex stage draws a list of the mesh's topology, which
            // Vulkan biases alike.
            Assembly::Meshes => match &pipeline.base().graphics.mesh_records {
                Some(records) => (records.topology(), false),
                None => (metal2vulkan::vk::PrimitiveTopology::TRIANGLE_LIST, false),
            },
        };
        // The pass's depth and stencil: its attachment's, or the hidden ones a draw whose state
        // uses depth or the stencil the pass lacks gives it from that draw on, as plain Metal
        // tests such a draw (its validation stops it; `render_state.rs`).
        let ds = self.layout.depth_stencil.as_ref();
        let (depth, _) = self.layout.aspects();
        // A compare of `Never` without Metal's depth attachment fails every fragment (probes), the
        // hidden depth's too: the draw runs as `FailingDraw` says.
        let metal_depth = depth && !ds.is_some_and(|ds| ds.hidden);
        let failing_state = self.state.failing_depth(metal_depth).cloned();
        let max = pipeline.base().graphics.max_vertex_amplification_count;
        let mut amplification = self
            .amplification
            .bind_state(max)
            .map_err(|problem| draw_problem(call, problem))?;
        // A query counts an amplified draw drawn one amplification at a time, as Metal counts
        // every amplification's samples: a multiview instance's would take one per view. Over a
        // rate map whose layers differ each amplification records its own slice's regions
        // (`rate_mapped_pass.rs`).
        amplification.one_at_a_time = self.counts_visibility
            || self
                .rate_map
                .as_ref()
                .is_some_and(|pass| !pass.layers_alike());
        // The pass's tile threadgroup memory, which the fragment function's threadgroup
        // arguments read, as a tile dispatch's do (`render/tile_memory.rs`): its tiles, and the
        // arguments the encoder set.
        let tiles = match pipeline.reads_tile_memory() {
            true => {
                let tile = self.layout.tile_size()?;
                Some((tile, self.layout.tile_area(tile)))
            }
            false => None,
        };
        // The viewports a Vulkan viewport cannot be, compensated by the vertex function's
        // variant; a pipeline that rasterizes nothing draws through no viewport.
        let base = &pipeline.base().graphics;
        let device = self.device;
        let native = render_state::native_depth_range(device);
        // A saturating or offsetting depth bias folds into a compensated depth, or is refused
        // there (`EncoderState::compensation`).
        let bias = self
            .state
            .draw_bias(topology, device.compile_device().profile())?;
        // A clamp mode one clamp of one depth collapses the Vulkan range to it where the
        // fragment function reads no depth of its position (`render_state::OneDepth`).
        let one_depth = render_state::OneDepth::of(device, base);
        // Metal clamps each vertex's depth before the interpolation: the vertex stage's
        // compensation does too where it can, natively as well (`render_state::VertexClamp`).
        let vertex_clamp = render_state::VertexClamp::of(
            device, self.state, base, topology, meshes, bias, announce,
        );
        let selects = base.writes_viewport_index;
        let compensation = match base.rasterizer_discard {
            true => None,
            false => match self
                .state
                .compensation(selects, native, bias, one_depth, vertex_clamp)
            {
                // Natively, viewports whose clamped depths need different compensations keep
                // the device's ranges, each sample clamped.
                Err(Error::NotImplemented(RuntimeGap::ViewportDepthRange))
                    if native && vertex_clamp.takes =>
                {
                    let unclamped = render_state::VertexClamp::default();
                    let kept = self
                        .state
                        .compensation(selects, native, bias, one_depth, unclamped)?;
                    render_state::VertexClamp::kept(
                        device,
                        self.state,
                        selects,
                        "the draw's viewports have different depth ranges, and the vertex stage \
                         holds one compensation",
                        announce,
                    );
                    kept
                }
                compensation => compensation?,
            },
        };
        render_state::VertexClamp::taken(device, self.state, selects, compensation, announce);
        // A fragment function that writes its depth, in depth clamp mode through a range above
        // 0, clamps the depth to the range itself under a Vulkan clamp that holds 0, as Metal
        // stores a negative NaN as 0 there (`render_state::written_depth_clamp`).
        let written_clamp = match base.writes_depth && !base.rasterizer_discard {
            true => render_state::written_depth_clamp(
                device,
                self.state,
                base.writes_viewport_index,
                topology,
                base.clip_distances,
                meshes,
                announce,
            )?,
            false => None,
        };
        // A positive NaN it writes runs no stencil depth-failure operation, where Metal runs one:
        // announced once (`render_state::announce_nan_depth_failure`).
        if announce && base.writes_depth && !base.rasterizer_discard {
            render_state::announce_nan_depth_failure(device, self.state);
        }
        // Points that rasterize, which Metal sizes none of: Metal's are undefined, and the
        // runtime's none is one of its answers. Announced at the draw, which for an execution's
        // command is where the runtime records it (its commit, or its host step).
        if announce && unsized_points && !base.rasterizer_discard {
            device.degrade(Degradation::UnsizedPoints, || {
                (
                    format!(
                        "{call}: a draw of points under a pipeline state whose vertex function \
                         writes no point size, of the {class:?} input primitive topology class"
                    ),
                    "Metal sizes no point outside the point class without a point size, and \
                     what it draws there is undefined"
                        .to_owned(),
                )
            });
        }
        let compensation = written_clamp.and_then(|w| w.compensation).or(compensation);
        // A draw of points takes Metal's guard band where it can, else keeps Vulkan's clipping
        // of points, announced once (`render_state::guard_points`). Unsized points rasterize
        // nothing, and a mesh pipeline's points are unmeasured on Metal.
        let points = topology == metal2vulkan::vk::PrimitiveTopology::POINT_LIST
            && !meshes
            && !base.rasterizer_discard
            && !unsized_points;
        let compensation = match points {
            false => compensation,
            true => render_state::guard_points(
                device,
                self.state,
                base.writes_viewport_index,
                compensation,
                base.clip_distances,
                self.rate_map.is_some(),
                announce,
            ),
        };
        // The pass's colors the pipeline lacks, zeroed where the draw covers them, and the
        // pipeline's colors the pass lacks, whose writes are dropped.
        // Each by its place in Metal's per-sample record (`routing`), the draw's subpass
        // referencing each pass color at the location the state's color there writes.
        let routing = self
            .routing(pipeline)
            .map_err(|_| internal("a draw of a state its pass refuses"))?;
        let routed_colors = self.routed_colors(pipeline, &routing);
        // A compensated depth, which a vertex clamp takes natively too, replaces the native
        // range (`clamp_draw_depth`).
        let compensated = compensation.is_some_and(|c| c.depth.is_some());
        let native = native && !compensated;
        // How it draws its lines at the encoder's width, and a band's push: Vulkan viewport 0's
        // half extent, its height negative as the Y-flipped viewport's is, and the half width
        // (`render/line_width.rs`). A band takes its flat inputs from its first vertex.
        let line = self.line_route(call, (pipeline, assembly), announce)?;
        let line_band = match line {
            LineRoute::Band { half } => {
                let viewports =
                    EncoderState::vk_viewports(&self.state.used_viewports(false)[..1], None, None);
                let v = viewports[0];
                Some([v.width / 2.0, v.height / 2.0, half, 0.0])
            }
            _ => None,
        };
        let provoking_last =
            !line.band() && provoking::natively_last(device, self.state, (pipeline, assembly));
        Ok(DrawSettings {
            unsized_points,
            topology,
            primitive_restart,
            hidden,
            failing_state,
            amplification,
            tiles,
            native,
            compensation,
            written_clamp,
            bias,
            routing,
            routed_colors,
            provoking_last,
            line,
            line_band,
        })
    }

    /// The state a draw of `graphics`, its variant's, takes statically before its depth range
    /// ([`clamp_draw_depth`](Self::clamp_draw_depth)): the encoder's; a failing draw's depth and
    /// stencil, masking the colors; and no stencil test where the function is the test (`shadow`, a
    /// shader-side stencil).
    pub(super) fn draw_fixed(
        &self,
        graphics: &GraphicsState,
        failing: Option<&FailingDraw>,
        (shadow, shader_tests): (bool, bool),
    ) -> Result<FixedState, Error> {
        let profile = self.device.compile_device().profile();
        let selects = graphics.writes_viewport_index;
        // An amplification's viewport offset selects among the encoder's viewports too; on a
        // device without `multiViewport` its variant stores no `ViewportIndex` (a draw there has
        // one viewport), exact only while the encoder holds one.
        if self.amplification.offsets_viewports() && self.state.viewport_count(true) > 1 {
            EncoderState::needs_viewports(profile)?;
        }
        let mut fixed = self
            .state
            .fixed(self.state.viewport_count(selects), profile)?;
        if let Some(FailingDraw::Runs(failing) | FailingDraw::Fails(failing)) = failing {
            fixed.depth_stencil = metal2vulkan::metal::state::depth_stencil_state(failing);
            fixed.masks_color = true;
        }
        // The function is the stencil test: the pass's own is off, and never writes the stencil.
        if shadow {
            fixed.depth_stencil.stencil_test_enable = false;
        }
        // The function is the depth test: the pass's own passes every fragment and never writes
        // the depth, and the stencil test is off (the binder refuses a draw whose stencil test
        // could drop or change anything).
        if shader_tests {
            let ds = &mut fixed.depth_stencil;
            ds.depth_test_enable = true;
            ds.depth_compare_op = metal2vulkan::vk::CompareOp::ALWAYS;
            ds.depth_write_enable = false;
            ds.stencil_test_enable = false;
        }
        Ok(fixed)
    }

    /// `fixed`'s depth range and clamp, for a draw of `graphics` under its settings' `native`,
    /// `written_clamp`, `compensation` and `bias` ([`DrawSettings`]): a depth range past
    /// `[0, 1]`, which the device takes natively, clamped, clipped or not (a clamp-mode saturated
    /// depth where Metal stores it), elsewhere the vertex stage's compensation, whose clip
    /// distances clip and whose Vulkan range `depthClamp` clamps to; and for a fragment function
    /// that clamps its depth to the range itself, a Vulkan clamp that holds 0 and the range.
    pub(super) fn clamp_draw_depth(
        &self,
        fixed: &mut FixedState,
        graphics: &GraphicsState,
        (native, written_clamp, compensation, bias): (
            bool,
            Option<WrittenDepthClamp>,
            Option<Compensation>,
            Option<DepthBiasEffect>,
        ),
    ) -> Result<(), Error> {
        fixed.depth_range = match (native, written_clamp.and_then(|w| w.native)) {
            (true, Some(range)) => Some(range),
            (true, None) => self
                .state
                .depth_range(graphics.writes_viewport_index, bias)?,
            (false, _) => None,
        };
        let depth = compensation.and_then(|c| c.depth);
        fixed.depth_clamp |= fixed.depth_range.is_some() || depth.is_some();
        Ok(())
    }

    /// The state a draw of `graphics`, its variant's, records with under `settings`
    /// ([`DrawState`]: statically, or as dynamic state), failing as `failing` says, its fragment
    /// function the stencil test (`shadow`, a shader-side stencil) or the depth test (`tests`);
    /// with the viewports in Metal's terms and the compensated
    /// depth range a rate-mapped pass draws each region through (`rate_mapped_pass.rs`). A 16-bit
    /// unorm depth's notice and an unclamped bias's are announced.
    pub(super) fn draw_state(
        &self,
        graphics: &GraphicsState,
        settings: &DrawSettings,
        (failing, shadow, tests): (Option<&FailingDraw>, bool, bool),
    ) -> Result<ShapedState, Error> {
        let device = self.device;
        let profile = device.compile_device().profile();
        let selects = graphics.writes_viewport_index;
        let topology = settings.topology;
        let (native, compensation) = (settings.native, settings.compensation);
        let bias_effect = settings.bias;
        let mut fixed = self.draw_fixed(graphics, failing, (shadow, tests))?;
        fixed.provoking_last = settings.provoking_last;
        settings.line.fix(&mut fixed);
        // The depth the draw renders against: the pass's attachment's, or the hidden one's,
        // which Metal biases as the 32-bit float depth it is.
        let depth_format = self
            .layout
            .depth_stencil
            .as_ref()
            .and_then(|ds| ds.depth_format)
            .or(settings.hidden.then_some(HIDDEN_DEPTH_FORMAT));
        let bias = self.state.vk_depth_bias(topology, profile)?;
        let depth_bias = DepthBiasRecord::of(&bias);
        // A 16-bit unorm depth, biased or not, stores Metal's codes where the driver is measured
        // to (`render_state::announce_unorm_depth`).
        render_state::announce_unorm_depth(device, depth_format, &fixed.depth_stencil, &bias);
        // In clip mode a natively ranged device clamps a biased depth to `[0, 1]`
        // (`EncoderState::depth_range`); one that stores it unclamped without that announces it.
        if !graphics.rasterizer_discard {
            let compensated = compensation.is_some_and(|c| c.depth.is_some());
            render_state::announce_unclamped_bias(device, self.state, bias_effect, compensated);
        }
        // A compensated depth's Vulkan range replaces the viewports' ranges, a saturating or
        // offsetting bias folded into the compensation (`EncoderState::compensation`).
        let depth = compensation.and_then(|c| c.depth);
        let viewports = match depth {
            Some(_) => self.state.used_viewports(selects).iter().copied().collect(),
            None => self
                .state
                .draw_viewports(selects, topology, profile, native)?,
        };
        let clamp = (native, settings.written_clamp, compensation, bias_effect);
        self.clamp_draw_depth(&mut fixed, graphics, clamp)?;
        let state = DrawState {
            viewports: EncoderState::vk_viewports(
                &viewports,
                depth,
                compensation.and_then(|c| c.guard),
            ),
            scissors: self.state.vk_scissors(selects),
            fixed,
            stencil_reference: self.state.stencil_reference(),
            depth_bias,
            blend_color: self.state.blend_color(),
        };
        Ok(ShapedState {
            state,
            viewports,
            depth,
        })
    }
}

/// A draw's state as it records it ([`DrawState`]), and what a rate-mapped pass draws each of its
/// regions through: the draw's viewports in Metal's terms, and its compensated depth range
/// ([`ShapeInputs::draw_state`]).
#[derive(Clone, PartialEq)]
pub(super) struct ShapedState {
    pub(super) state: DrawState,
    pub(super) viewports: ViewportList<Viewport>,
    pub(super) depth: Option<[f32; 2]>,
}

/// The most entries a memo keeps: a pass's draws take a few drawing states and assemblies in turn
/// (a layer's and its shadow's; strips and indexed lists), each its own entry.
const ENTRIES: usize = 4;

/// The pieces of the encoder's draws' shapes, each with what it was derived from (the module
/// docs): an entry for each of the latest few settings' inputs, held in place, so a draw that
/// keeps one allocates nothing for it and a draw that finds one moves none.
#[derive(Default)]
pub(super) struct ShapeMemo {
    /// `None` where none was kept.
    entries: [Option<Entry>; ENTRIES],
    /// The count of draws that found or kept an entry: an entry's id, and its last use.
    ticks: u64,
}

/// The pieces derived under one draw's settings' inputs.
struct Entry {
    /// Its own, which the draw that found or kept it holds ([`ShapeToken`]).
    id: u64,
    /// The tick of the last draw that found or kept it: the least recent goes first.
    used: u64,
    /// The drawing state, held: compared by identity.
    pipeline: DrawingState,
    assembly: Assembly,
    /// [`ShapeInputs::versions`] and [`ShapeInputs::counts_visibility`].
    versions: [u64; 3],
    counts_visibility: bool,
    settings: DrawSettings,
    /// The draw state of a draw of the variant, held.
    state: Option<(Arc<RenderVariant>, ShapedState)>,
    found: Option<FoundPipeline>,
}

/// A draw's hold on the memo entry its settings came from or went into, where its draw state and
/// pipeline are found and kept: its place, and its id.
#[derive(Clone, Copy)]
pub(super) struct ShapeToken {
    at: usize,
    id: u64,
}

/// What a draw finds its `VkPipeline` by (its graphics key, `DrawKey`, and its variant), each
/// field by value or by what it is made of: its variant and plan (the plan's specialization),
/// its instance's shape (by version) and compatible pass, its vertex strides (the pipeline's own,
/// none where it takes them as dynamic state), its fixed state before the pipeline's dynamic
/// states are left out, and its keyed topology. The pass's sample positions are the layout's,
/// whose version the entry holds.
pub(super) struct PipelineInputs<'k> {
    pub(super) variant: &'k Arc<RenderVariant>,
    pub(super) plan: &'k Arc<BindingPlan>,
    pub(super) shape: u64,
    pub(super) compatible: &'k Arc<RenderPassObject>,
    pub(super) strides: &'k [(u32, u32)],
    pub(super) fixed: FixedState,
    pub(super) topology: (metal2vulkan::vk::PrimitiveTopology, bool),
}

/// A `VkPipeline` a draw found made, with what it found it by ([`PipelineInputs`], held).
struct FoundPipeline {
    variant: Arc<RenderVariant>,
    plan: Arc<BindingPlan>,
    shape: u64,
    compatible: Arc<RenderPassObject>,
    /// Inline: a draw's strides are few, none where the pipeline takes them as dynamic state.
    strides: ViewportList<(u32, u32)>,
    fixed: FixedState,
    topology: (metal2vulkan::vk::PrimitiveTopology, bool),
    made: Arc<RawPipeline>,
}

impl FoundPipeline {
    /// Whether it was found by `inputs`.
    fn by(&self, inputs: &PipelineInputs<'_>) -> bool {
        Arc::ptr_eq(&self.variant, inputs.variant)
            && Arc::ptr_eq(&self.plan, inputs.plan)
            && self.shape == inputs.shape
            && Arc::ptr_eq(&self.compatible, inputs.compatible)
            && *self.strides == *inputs.strides
            && self.fixed == inputs.fixed
            && self.topology == inputs.topology
    }
}

impl ShapeMemo {
    /// The settings an earlier draw of `pipeline` as `assembly` derived under the encoder's
    /// `versions` and query count, with the token of their entry, which is used now.
    pub(super) fn settings(
        &mut self,
        (pipeline, assembly): (&DrawingState, Assembly),
        (versions, counts_visibility): ([u64; 3], bool),
    ) -> Option<(DrawSettings, ShapeToken)> {
        self.ticks += 1;
        let (at, entry) = (0..).zip(&mut self.entries).find_map(|(at, entry)| {
            let entry = entry.as_mut().filter(|entry| {
                entry.pipeline.same(pipeline)
                    && entry.assembly == assembly
                    && entry.versions == versions
                    && entry.counts_visibility == counts_visibility
            })?;
            Some((at, entry))
        })?;
        entry.used = self.ticks;
        let token = ShapeToken { at, id: entry.id };
        Some((entry.settings.clone(), token))
    }

    /// Keep `settings`, derived for a draw of `pipeline` as `assembly` under the encoder's
    /// `versions` and query count, in a new entry, in place of the least recently used of
    /// [`ENTRIES`]: its token.
    pub(super) fn keep_settings(
        &mut self,
        (pipeline, assembly): (&DrawingState, Assembly),
        (versions, counts_visibility): ([u64; 3], bool),
        settings: &DrawSettings,
    ) -> ShapeToken {
        self.ticks += 1;
        let at = (0..ENTRIES)
            .min_by_key(|&at| self.entries[at].as_ref().map_or(0, |entry| entry.used))
            .unwrap_or_default();
        self.entries[at] = Some(Entry {
            id: self.ticks,
            used: self.ticks,
            pipeline: pipeline.clone(),
            assembly,
            versions,
            counts_visibility,
            settings: settings.clone(),
            state: None,
            found: None,
        });
        ShapeToken { at, id: self.ticks }
    }

    fn entry(&self, token: ShapeToken) -> Option<&Entry> {
        (self.entries[token.at].as_ref()).filter(|entry| entry.id == token.id)
    }

    fn entry_mut(&mut self, token: ShapeToken) -> Option<&mut Entry> {
        (self.entries[token.at].as_mut()).filter(|entry| entry.id == token.id)
    }

    /// The draw state a draw of `variant` derived under `token`'s settings.
    pub(super) fn state(
        &self,
        token: ShapeToken,
        variant: &Arc<RenderVariant>,
    ) -> Option<ShapedState> {
        let (kept, state) = self.entry(token)?.state.as_ref()?;
        Arc::ptr_eq(kept, variant).then(|| state.clone())
    }

    /// Keep `state`, a draw of `variant`'s derived under `token`'s settings.
    pub(super) fn keep_state(
        &mut self,
        token: ShapeToken,
        variant: &Arc<RenderVariant>,
        state: &ShapedState,
    ) {
        if let Some(entry) = self.entry_mut(token) {
            entry.state = Some((Arc::clone(variant), state.clone()));
        }
    }

    /// The `VkPipeline` a draw found made by `inputs` under `token`'s settings and its variant's
    /// draw state.
    pub(super) fn pipeline(
        &self,
        token: ShapeToken,
        inputs: &PipelineInputs<'_>,
    ) -> Option<Arc<RawPipeline>> {
        let found = self.entry(token)?.found.as_ref()?;
        found.by(inputs).then(|| Arc::clone(&found.made))
    }

    /// Keep `made`, the `VkPipeline` a draw found made by `inputs` under `token`'s settings.
    pub(super) fn keep_pipeline(
        &mut self,
        token: ShapeToken,
        inputs: &PipelineInputs<'_>,
        made: &Arc<RawPipeline>,
    ) {
        if let Some(entry) = self.entry_mut(token) {
            entry.found = Some(FoundPipeline {
                variant: Arc::clone(inputs.variant),
                plan: Arc::clone(inputs.plan),
                shape: inputs.shape,
                compatible: Arc::clone(inputs.compatible),
                strides: inputs.strides.iter().copied().collect(),
                fixed: inputs.fixed,
                topology: inputs.topology,
                made: Arc::clone(made),
            });
        }
    }
}
