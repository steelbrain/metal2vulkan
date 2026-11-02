//! What a render encoder checks as each call is made, where the device lowers its calls after they
//! return (`RenderLoweringPolicy::Worker`, `render/deferral.rs`): every refusal the call would
//! return at once, made by the lowering's own checks over the encoder's Metal state as the calls
//! so far set it, so the call returns it.
//!
//! - **What is checked.** The argument tables' setters (`render/table_facts.rs`), the encoder's
//!   state setters (`EncoderState`'s, the same methods), the vertex amplification count, a
//!   pipeline state set (`ShapeInputs::check_pipeline`, `drawn_as`), the objects every call names
//!   (`encoder::check_device`), and a draw of primitives: its arguments (`check_arguments`), its
//!   kind (`drawn_by`) and its settings (`ShapeInputs::settings`, which announces nothing here).
//! - **Lowered at the call.** A call whose checks read what the lowering decides as it runs, or
//!   that sets state these checks do not hold, is lowered inside the call, the calls logged before
//!   it first: its refusal returns from the call itself, as `RenderLoweringPolicy::AtCall`
//!   returns it ([`Checked::Now`]). From some of them on (`sticky`) the encoder lowers every call
//!   at its call: a pipeline state whose draws fetch the pass's colors, read its tile memory, use
//!   an imageblock, draw patches or meshes, or render a color in another format than the pass's
//!   attachment (each changes the pass's attachments as the lowering decides); a tile dispatch,
//!   an execution of indirect render commands, a memory barrier, the restart and other Apple
//!   calls; and a draw that gives the pass its hidden depth and stencil. Others are lowered at the
//!   call alone: store actions, resolve targets, a visibility result mode, threadgroup memory
//!   lengths, fences, patch and mesh draws; a draw while a vertex buffer bound runs past 4 GiB of
//!   its offset (`render/vertex_reach.rs`) or a visible-function table is bound
//!   (`ArgumentTable::check_function_tables`); and any call while a buffer of the device is
//!   detached (`detach.rs`), whose skips judge the detachment as of the call.
//! - **The end.** The encoder's end checks the depth/stencil attachment's store actions and its
//!   stencil's store action options as they stand then; where it may, it is lowered inside
//!   `end_encoding`, which returns the refusal (`CallChecks::ends_at_call`), and is lowered after
//!   the call otherwise (`render/deferral.rs`).
//! - **Found only as lowered.** What the lowering finds of its own (a runtime gap of the draw's
//!   variant, its instance or its resources; out of memory; a Vulkan error) fails the command
//!   buffer, naming the call (`CommandBufferError::CallRefused`), the end's as `endEncoding`.

use super::draw_shape::{LayoutView, ShapeInputs};
use super::replay::{DrawIndexed, Table};
use super::table_facts::{check_samplers, check_textures, TableFacts};
use super::{check_arguments, drawn_as, drawn_by, Assembly, RenderCall, RenderLowering};
use crate::amplification::Amplification;
use crate::device::Device;
use crate::encoder::check_device;
use crate::error::Error;
use crate::fetch_reach::REACHED;
use crate::pipeline::{GraphicsState as DrawingState, RenderPipelineState, RenderStateKind};
use crate::rate_mapped_pass::RateMappedPass;
use crate::render_pass::{AspectTarget, PassLayout};
use crate::render_state::{depth_ranges_refused, EncoderState};
use crate::render_values::{StoreAction, StoreActionOptions};
use crate::replay::CallLog;
use crate::versioned::Versioned;
use metal2vulkan::metal::PrimitiveType;

/// What the program's encoder does with a call its checks passed (the module docs).
pub(super) enum Checked {
    /// Log it: it is lowered later.
    Logged,
    /// Lower it inside the call, the calls logged before it first; and, `sticky`, every later call
    /// of the encoder at its call.
    Now { sticky: bool },
}

/// A call lowered at the call alone, and one from which the encoder lowers every call there.
const NOW: Result<Checked, Error> = Ok(Checked::Now { sticky: false });
const FROM_NOW: Result<Checked, Error> = Ok(Checked::Now { sticky: true });

/// The checks of a render encoder's calls, with the encoder's Metal state as its calls set it
/// (the module docs).
pub(crate) struct CallChecks {
    /// The vertex, fragment, object, mesh and tile tables, by [`Table`].
    tables: [TableFacts; 5],
    pipeline: Option<RenderPipelineState>,
    state: Versioned<EncoderState>,
    amplification: Versioned<Amplification>,
    layout: PassLayout,
    rate_map: Option<RateMappedPass>,
    /// The pipeline state, assembly and state versions of the last draw whose settings passed:
    /// a draw of the same inputs passes them too.
    settled: Option<(DrawingState, Assembly, [u64; 2])>,
    /// A call set a store action, its options or a resolve target, which these checks do not
    /// hold: the encoder's end checks them, so it ends at its call ([`Self::ends_at_call`]).
    stores_set: bool,
}

impl CallChecks {
    /// The checks of `lowering`'s encoder, its pass just opened: its state as the lowering's.
    pub(super) fn new(lowering: &RenderLowering) -> CallChecks {
        CallChecks {
            tables: Default::default(),
            pipeline: None,
            state: Versioned::new((*lowering.state).clone()),
            amplification: Versioned::new((*lowering.amplification).clone()),
            layout: (*lowering.layout).clone(),
            rate_map: lowering.rate_map.clone(),
            settled: None,
            stores_set: false,
        }
    }

    /// These checks made the checks of `lowering`'s encoder, as [`new`](Self::new) makes them, in
    /// their own lists: with no allocation where an earlier encoder's left room enough.
    pub(super) fn reset(&mut self, lowering: &RenderLowering) {
        for table in &mut self.tables {
            table.clear();
        }
        self.pipeline = None;
        (*self.state).clone_from(&lowering.state);
        (*self.amplification).clone_from(&lowering.amplification);
        self.layout.clone_from(&lowering.layout);
        self.rate_map.clone_from(&lowering.rate_map);
        self.settled = None;
        self.stores_set = false;
    }

    /// Whether the encoder's end is to be lowered inside `end_encoding`, which then returns its
    /// refusal: where the end makes one of Metal's checks of its own, which the call returns. The
    /// end checks the depth/stencil attachment's store actions as they stand then (one still
    /// `Unknown` settles to `DontCare` and its resolve is made again, `render/late_resolve.rs`)
    /// and its stencil's store action options against them (`sample_positions::check_options`):
    /// so where the attachment has a store action `Unknown` or options, or a call set any store
    /// action, its options or a resolve target. Every other end is lowered after the call, its
    /// refusals the lowering's (`CommandBufferError::CallRefused`).
    pub(super) fn ends_at_call(&self) -> bool {
        let checked = |aspect: Option<AspectTarget>| {
            aspect.is_some_and(|aspect| {
                aspect.store == StoreAction::Unknown || aspect.options != StoreActionOptions::NONE
            })
        };
        self.stores_set
            || (self.layout.depth_stencil.as_ref())
                .is_some_and(|ds| checked(ds.depth) || checked(ds.stencil))
    }

    /// Let go of every object the checks hold, their lists' room kept: their encoder ended, and
    /// none of what its calls named outlives it here.
    pub(super) fn release(&mut self) {
        for table in &mut self.tables {
            table.clear();
        }
        self.pipeline = None;
        let extent = self.layout.extent;
        *self.state = EncoderState::metal_defaults(extent);
        self.layout.colors.clear();
        self.layout.depth_stencil = None;
        self.layout.visibility = None;
        self.rate_map = None;
        self.settled = None;
    }

    /// Check `call` of an encoder of `device`, whose slices and held objects lie in `log` (its
    /// indexed draw's arguments in `indexed`), and take what it sets: a refusal, or what the
    /// encoder does with it.
    pub(super) fn check(
        &mut self,
        device: &Device,
        call: &RenderCall,
        (log, indexed): (&CallLog<RenderCall>, &[DrawIndexed]),
    ) -> Result<Checked, Error> {
        use RenderCall as C;
        if device.shared().rebinds.any_detached_now() {
            return NOW;
        }
        match call {
            C::PipelineState(held) => self.pipeline_state(device, log.render_state(*held)),
            C::Buffer(table, buffer, offset, index) => {
                let buffer = buffer.map(|held| log.buffer(held));
                let call = staged(table, "buffer");
                self.tables[table_index(table)]
                    .set_buffer(device, call, (buffer, *offset), *index)
                    .map(logged)
            }
            C::Buffers(table, buffers, offsets, first) => {
                let (buffers, offsets) = (log.buffer_slice(*buffers), log.words_of(*offsets));
                let call = staged(table, "buffers");
                self.tables[table_index(table)]
                    .set_buffers(device, call, (buffers, offsets, None), *first)
                    .map(logged)
            }
            C::BufferOffset(table, offset, index) => {
                let call = staged(table, "buffer_offset");
                (self.tables[table_index(table)].set_buffer_offset(call, *offset, *index))
                    .map(logged)
            }
            C::Bytes(table, bytes, index) => {
                let call = staged(table, "bytes");
                let length = log.bytes_of(*bytes).len();
                (self.tables[table_index(table)].set_bytes(call, length, *index)).map(logged)
            }
            C::Texture(table, texture, index) => {
                let texture = texture.map(|held| log.texture(held));
                let call = staged(table, "texture");
                check_textures(device, call, [texture].into_iter(), *index).map(logged)
            }
            C::Textures(table, textures, first) => {
                let textures = log.texture_slice(*textures).iter().map(Option::as_ref);
                check_textures(device, staged(table, "textures"), textures, *first).map(logged)
            }
            C::Sampler(table, sampler, index) => {
                let sampler = sampler.map(|held| log.sampler(held));
                let call = staged(table, "sampler_state");
                check_samplers(device, call, [sampler].into_iter(), None, *index).map(logged)
            }
            C::Samplers(table, samplers) => {
                let clamps = (samplers.clamps.as_ref()).map(|(min, max)| (min.len(), max.len()));
                let call = staged(table, "sampler_states");
                let each = samplers.samplers.iter().map(Option::as_ref);
                check_samplers(device, call, each, clamps, samplers.first).map(logged)
            }
            C::FunctionTable(table @ (Table::Vertex | Table::Fragment | Table::Tile), t, index) => {
                let call = staged(table, "visible_function_table");
                let tables = [t.as_ref()].into_iter();
                (self.tables[table_index(table)].set_function_tables(device, call, tables, *index))
                    .map(logged)
            }
            C::FunctionTables(
                table @ (Table::Vertex | Table::Fragment | Table::Tile),
                t,
                first,
            ) => {
                let call = staged(table, "visible_function_tables");
                let tables = t.iter().map(Option::as_ref);
                (self.tables[table_index(table)].set_function_tables(device, call, tables, *first))
                    .map(logged)
            }
            C::FunctionTable(..) | C::FunctionTables(..) => NOW,
            C::VertexBufferWithStride(buffer, offset, _, index) => {
                let buffer = buffer.map(|held| log.buffer(held));
                let call = "set_vertex_buffer_with_stride";
                (self.tables[0].set_buffer(device, call, (buffer, *offset), *index)).map(logged)
            }
            C::VertexBuffersWithStrides(spans) => {
                let (buffers, offsets, strides, first) = **spans;
                let buffers = log.buffer_slice(buffers);
                let (offsets, strides) = (log.words_of(offsets), log.words_of(strides).len());
                let call = "set_vertex_buffers_with_strides";
                let set = (buffers, offsets, Some(strides));
                self.tables[0]
                    .set_buffers(device, call, set, first)
                    .map(logged)
            }
            C::VertexBufferOffsetWithStride(offset, _, index) => {
                let call = "set_vertex_buffer_offset_with_stride";
                (self.tables[0].set_buffer_offset(call, *offset, *index)).map(logged)
            }
            C::VertexBytesWithStride(bytes, _, index) => {
                let length = log.bytes_of(*bytes).len();
                let call = "set_vertex_bytes_with_stride";
                (self.tables[0].set_bytes(call, length, *index)).map(logged)
            }
            C::Viewport(viewport) => {
                let refused = depth_ranges_refused(device);
                let viewports = std::slice::from_ref(&**viewport);
                (self.state.set_viewports("set_viewport", viewports, refused)).map(logged)
            }
            C::Viewports(viewports) => {
                let refused = depth_ranges_refused(device);
                (self
                    .state
                    .set_viewports("set_viewports", viewports, refused))
                .map(logged)
            }
            // Checked alone: no later check reads the rectangles.
            C::ScissorRect(rect) => {
                let rects = std::slice::from_ref(rect);
                self.state
                    .scissor_rects("set_scissor_rect", rects)
                    .map(|_| Checked::Logged)
            }
            C::ScissorRects(rects) => self
                .state
                .scissor_rects("set_scissor_rects", rects)
                .map(|_| Checked::Logged),
            C::CullMode(mode) => {
                self.state.set_cull_mode(*mode);
                Ok(Checked::Logged)
            }
            C::FrontFacingWinding(winding) => {
                self.state.set_winding(*winding);
                Ok(Checked::Logged)
            }
            C::DepthStencilState(held) => {
                let state = held.map(|held| log.depth_stencil(held));
                if let Some(state) = state {
                    check_device(device, "set_depth_stencil_state", state.device())?;
                }
                self.state.set_depth_stencil(state);
                Ok(Checked::Logged)
            }
            C::StencilReferenceValue(value) => {
                self.state.set_stencil_references(*value, *value);
                Ok(Checked::Logged)
            }
            C::StencilReferenceValues(front, back) => {
                self.state.set_stencil_references(*front, *back);
                Ok(Checked::Logged)
            }
            C::DepthBias(bias, slope, clamp) => {
                self.state.set_depth_bias(*bias, *slope, *clamp).map(logged)
            }
            C::DepthClipMode(mode) => {
                let profile = device.compile_device().profile();
                self.state.set_clip_mode(*mode, profile).map(logged)
            }
            C::TriangleFillMode(mode) => {
                let profile = device.compile_device().profile();
                self.state.set_fill_mode(*mode, profile).map(logged)
            }
            C::BlendColor(color) => {
                self.state.set_blend_color(*color);
                Ok(Checked::Logged)
            }
            C::TessellationFactorBuffer(buffer, ..) => {
                if let Some(buffer) = buffer {
                    check_device(device, "set_tessellation_factor_buffer", buffer.device())?;
                }
                Ok(Checked::Logged)
            }
            C::TessellationFactorScale(_) => Ok(Checked::Logged),
            C::VertexAmplificationCount(count, mappings) => {
                let call = "set_vertex_amplification_count";
                let set = Amplification::new(*count, mappings.as_deref())
                    .map_err(|problem| super::draw_problem(call, problem))?;
                *self.amplification = set;
                Ok(Checked::Logged)
            }
            C::DrawPrimitives(primitive, _, vertices, instances, _) => {
                let empty = *vertices == 0 || *instances == 0;
                self.draw(device, "draw_primitives", Assembly::of(*primitive), empty)
            }
            C::DrawIndexedPrimitives(at) => {
                let draw = indexed
                    .get(*at as usize)
                    .ok_or_else(|| crate::error::internal("a logged draw without its arguments"))?;
                let call = "draw_indexed_primitives";
                check_device(device, call, log.buffer(draw.index_buffer).device())?;
                let empty = draw.index_count == 0 || draw.instance_count == 0;
                self.draw(device, call, Assembly::of(draw.primitive_type), empty)
            }
            C::DrawPrimitivesIndirect(primitive, buffer, offset) => {
                let call = "draw_primitives_indirect";
                check_device(device, call, buffer.device())?;
                check_arguments(call, buffer.length(), *offset, 16)?;
                self.draw(device, call, Assembly::of(*primitive), false)
            }
            C::DrawIndexedPrimitivesIndirect(draw) => {
                let call = "draw_indexed_primitives_indirect";
                check_device(device, call, draw.index_buffer.device())?;
                check_device(device, call, draw.indirect_buffer.device())?;
                let indirect = &draw.indirect_buffer;
                check_arguments(call, indirect.length(), draw.indirect_buffer_offset, 20)?;
                self.draw(device, call, Assembly::of(draw.primitive_type), false)
            }
            C::UseResource(resource, ..) => {
                check_device(device, "use_resource", resource.as_ref().device())?;
                Ok(Checked::Logged)
            }
            C::UseResources(resources, ..) => {
                for resource in resources.iter() {
                    check_device(device, "use_resources", resource.as_ref().device())?;
                }
                Ok(Checked::Logged)
            }
            C::UseHeap(heap, _) => {
                check_device(device, "use_heap", heap.device())?;
                Ok(Checked::Logged)
            }
            C::UseHeaps(heaps, _) => {
                for heap in heaps.iter() {
                    check_device(device, "use_heaps", heap.device())?;
                }
                Ok(Checked::Logged)
            }
            // The pass's attachments as the lowering keeps them, which the end checks too.
            C::ColorStoreAction(..)
            | C::DepthStoreAction(_)
            | C::StencilStoreAction(_)
            | C::ColorStoreActionOptions(..)
            | C::DepthStoreActionOptions(_)
            | C::StencilStoreActionOptions(_)
            | C::ColorResolveTarget(..)
            | C::DepthResolveTarget(_)
            | C::StencilResolveTarget(_) => {
                self.stores_set = true;
                NOW
            }
            // The encoder's queries, threadgroup lengths and fences, which these checks do not
            // hold; patch and mesh draws, whose pipeline states lower every call at its call.
            C::VisibilityResultMode(..)
            | C::ThreadgroupMemoryLength(..)
            | C::ObjectThreadgroupMemoryLength(..)
            | C::UpdateFence(..)
            | C::WaitForFence(..)
            | C::DrawPatches(_)
            | C::DrawPatchesIndirect(_)
            | C::DrawIndexedPatches(_)
            | C::DrawIndexedPatchesIndirect(_)
            | C::MeshThreadgroups(_)
            | C::MeshThreads(_)
            | C::MeshThreadgroupsIndirect(_) => NOW,
            // What changes the pass's instances or state past these checks.
            C::Execute(..)
            | C::ExecuteIndirect(..)
            | C::ExecuteChecked(_)
            | C::ExecuteIndirectChecked(_)
            | C::ThreadsPerTile(_)
            | C::ThreadsPerTileInRegion(_)
            | C::BarrierWithScope(..)
            | C::BarrierWithResources(..)
            | C::PrimitiveRestartEnabled(..)
            | C::AlphaTestReferenceValue(_)
            | C::LineWidth(_)
            | C::PointSize(_)
            | C::ProvokingVertexMode(_)
            | C::ViewportTransformEnabled(_)
            | C::VertexAmplificationMode(..)
            | C::DepthCleared
            | C::StencilCleared => FROM_NOW,
        }
    }

    /// The encoder's state as a draw's settings read it, on `device`.
    fn shape_inputs<'a>(&'a self, device: &'a Device) -> ShapeInputs<'a> {
        ShapeInputs {
            device,
            layout: LayoutView::of(&self.layout),
            state: &self.state,
            amplification: &self.amplification,
            // Which reads no refusal: `ShapeInputs::settings` reads it after its last.
            counts_visibility: false,
            rate_map: self.rate_map.as_ref(),
            executes: false,
            routed: None,
            versions: [0, self.state.version(), self.amplification.version()],
        }
    }

    /// `set_render_pipeline_state` of `state`: refused as the lowering's `set_state` refuses it,
    /// where the lowering decides nothing of the pass for it (the module docs).
    fn pipeline_state(
        &mut self,
        device: &Device,
        state: &RenderPipelineState,
    ) -> Result<Checked, Error> {
        const CALL: &str = "set_render_pipeline_state";
        check_device(device, CALL, state.device())?;
        let RenderStateKind::Graphics(graphics) = state.kind() else {
            return FROM_NOW;
        };
        let base = &graphics.base().graphics;
        let decides = !base.input_attachments.is_empty()
            || !base.storage_attachments.is_empty()
            || base.fragment_imageblock.is_some()
            || graphics.reads_tile_memory()
            || graphics.descriptor().vertex().is_none()
            || crate::tessellation::patches(base).is_some()
            || self.reformats(device, graphics);
        if decides {
            return FROM_NOW;
        }
        let drawn = drawn_as(self.layout.samples, state)?;
        (self.shape_inputs(device)).check_pipeline(CALL, &drawn, &self.layout)?;
        self.pipeline = Some(drawn);
        Ok(Checked::Logged)
    }

    /// Whether a draw of `pipeline` renders a pass color in another format than its attachment's
    /// (`ShapeInputs::viewing`, which reads the attachment as the lowering keeps it).
    fn reformats(&self, device: &Device, pipeline: &DrawingState) -> bool {
        let Ok(routing) = self.shape_inputs(device).routing(pipeline) else {
            return false;
        };
        let wanted = pipeline.descriptor().color_formats();
        routing.placed.iter().any(|(&index, &other)| {
            let pass = self
                .layout
                .colors
                .get(other as usize)
                .and_then(Option::as_ref);
            let wanted = wanted.get(index as usize).copied().flatten();
            pass.is_some_and(|pass| Some(pass.target.texture.pixel_format()) != wanted)
        })
    }

    /// A draw of primitives in `call` of `assembly`, `empty` where it draws nothing: refused as
    /// the lowering's `draw_resident` refuses it up to its settings (the module docs).
    fn draw(
        &mut self,
        device: &Device,
        call: &'static str,
        assembly: Option<Assembly>,
        empty: bool,
    ) -> Result<Checked, Error> {
        // Primitive type 9 draws nothing, checked as a draw of a public type is
        // (`RenderLowering::draw_nine`).
        let Some(assembly) = assembly else {
            let stand_in = Assembly::Primitives(PrimitiveType::Point);
            drawn_by(call, self.pipeline.as_ref(), stand_in, empty)?;
            return Ok(Checked::Logged);
        };
        let Some(pipeline) = drawn_by(call, self.pipeline.as_ref(), assembly, empty)? else {
            return Ok(Checked::Logged);
        };
        // Plain Metal draws nothing whatever the rest of the encoder's state.
        if empty || self.amplification.draws_nothing() {
            return Ok(Checked::Logged);
        }
        let [vertex, fragment, ..] = &self.tables;
        if vertex.holds_past(REACHED) || vertex.holds_tables() || fragment.holds_tables() {
            return NOW;
        }
        let (depth, stencil) = self.layout.aspects();
        if self.state.needs_hidden(depth, stencil) {
            return FROM_NOW;
        }
        let versions = [self.state.version(), self.amplification.version()];
        let settled = self
            .settled
            .as_ref()
            .is_some_and(|(kept, kept_assembly, kept_versions)| {
                kept.same(&pipeline) && *kept_assembly == assembly && *kept_versions == versions
            });
        // Its count changes how it draws its lines alone, which refuses nothing
        // (`render/line_width.rs`).
        if !settled {
            (self.shape_inputs(device)).settings(
                call,
                (&pipeline, assembly, true),
                false,
                false,
            )?;
            self.settled = Some((pipeline, assembly, versions));
        }
        Ok(Checked::Logged)
    }
}

/// A setter's passing: its call is logged.
fn logged((): ()) -> Checked {
    Checked::Logged
}

/// The place of `table` among [`CallChecks`]'s tables.
fn table_index(table: &Table) -> usize {
    match table {
        Table::Vertex => 0,
        Table::Fragment => 1,
        Table::Object => 2,
        Table::Mesh => 3,
        Table::Tile => 4,
    }
}

/// The call name of `table`'s stage's setter of `what` (`set_vertex_buffer`), as the lowering's
/// setters name their refusals.
fn staged(table: &Table, what: &'static str) -> &'static str {
    macro_rules! names {
        ($($what:literal),+) => {
            match (table, what) {
                $(
                    (Table::Vertex, $what) => concat!("set_vertex_", $what),
                    (Table::Fragment, $what) => concat!("set_fragment_", $what),
                    (Table::Object, $what) => concat!("set_object_", $what),
                    (Table::Mesh, $what) => concat!("set_mesh_", $what),
                    (Table::Tile, $what) => concat!("set_tile_", $what),
                )+
                _ => what,
            }
        };
    }
    names!(
        "buffer",
        "buffers",
        "buffer_offset",
        "bytes",
        "texture",
        "textures",
        "sampler_state",
        "sampler_states",
        "visible_function_table",
        "visible_function_tables"
    )
}
