//! A render pass over a rasterization-rate map (`MTLRenderPassDescriptor.rasterizationRateMap`):
//! what Metal takes, and each draw drawn region by region.
//!
//! - **What Metal's API validation stops**, measured by our probes on an M3 Max (macOS 26.7) and an
//!   M4 (macOS 27.0.1), identically, is refused as the encoder is made, `Invalid` with a
//!   [`RenderPassProblem`]: a map of another device; a `renderTargetArrayLength` past the
//!   layer count of a map whose layers differ ("renderTargetArrayLength (2) must be <=
//!   rasterizationRateMap.layerCount (1)"); and, without attachments, a render target width or
//!   height past the map's screen ("renderTargetWidth (128) must be <=
//!   rasterizationRateMap.screenSize.width (64)"). Plain Metal runs two of the forms its
//!   validation stops, as the runtime does: past the one layer of a map, it rasterizes every slice
//!   by that layer (probe 16's `map1_array`, a vertex function choosing slices 0 and 1, and
//!   `map1_array_flat`, a draw of slice 0 alone), as a map of layers alike draws here; and
//!   attachments narrower or shorter than the map's largest physical size ("maximum physical
//!   rendering width (33) must be <= minimum attachment width (32)") take the physical pixels that
//!   lie in them (`att_32`: 32 by 32, each at its centre; `att_small`: 16 by 16), the render area
//!   cutting each region's scissor.
//! - **The pass's screen area.** Metal reads a rate-mapped pass's viewports and scissor
//!   rectangles in screen pixels. With attachments the area is the map's screen, whatever their
//!   size: the default viewport and scissor cover it (a rectangle over the default viewport's left
//!   half shaded the same 16 physical columns into attachments of 33, 48, 64 and 128 pixels), and
//!   a scissor past it stops ("render pass width(64)" with attachments of 33 to 128 pixels).
//!   Without attachments it is the render target size, which may be smaller than the screen: a
//!   pass of 33 by 33 over a 64-pixel screen shades its first 17 physical columns and rows. The
//!   encoder's state is made over that area ([`RateMappedPass::area`]).
//! - **What a draw shades.** A physical pixel is shaded where its tile's line sends its centre into
//!   the primitive, inside the screen-space viewport, and never past the map's physical size,
//!   whatever the attachments' size. The draw is recorded once per region of the compile layer's
//!   plan (`RasterizationRateMap::regions`) of the layer below, each with its composed viewport and
//!   its scissor of physical pixels, cut to the pixels whose centres lie inside its viewport, its
//!   left and top edges in and its right and bottom edges out (Metal's implicit scissor), and to
//!   the render area ([`RateMappedPass::draws`]). A region the cut leaves no pixel is not recorded:
//!   it would shade nothing, and a plan of several regions has no vertex function that writes
//!   memory. Where the cut leaves no region a pixel, the first alone is recorded, through its empty
//!   scissor, so the vertex function runs as Metal's does. A composed viewport's edge falls
//!   between pixel centres, and a driver need not clip a primitive there: RADV on an AMD 780M
//!   (Mesa 26.2.4) shaded the column past such an edge where the scissor held the pixels the
//!   viewport only overlaps. The lines are this crate's tiles', so a pass over a map that is not
//!   the identity is a near-equivalent within the bound; the map's creation announced it
//!   (`rasterization-rate-map`).
//! - **Regions several at a time.** Where the device enables `multiDraw` (`VK_EXT_multi_draw`),
//!   `multiViewport`, `shaderOutputViewportIndex` and `shaderDrawParameters`, a direct draw of one
//!   amplification whose pipeline is a vertex and a fragment function, its vertex function
//!   choosing no slice or viewport and taking no position compensation or band
//!   (`RenderCommandEncoder::draws_region_batches`), records its regions `k` at a time, `k` the
//!   lesser of the device's `maxViewports` and one below its `maxMultiDrawCount` ("drawCount
//!   must be less than" it, `VUID-vkCmdDrawMultiEXT-drawCount-04934`): each batch sets `k`
//!   viewports and scissors, its regions' (past its regions repeating its first viewport through
//!   empty scissors), and records one `vkCmdDrawMultiEXT` or
//!   `vkCmdDrawMultiIndexedEXT` of as many draws as it has regions, each the draw's vertices,
//!   indices and instances, packed (`VUID-vkCmdDrawMultiEXT-drawCount-09628`). Its variant's
//!   vertex function stores `ViewportIndex` as `DrawIndex` (`VariantKey::region_viewports`): "the
//!   zero-based index of the draw" of such a command, dynamically uniform (the specification's
//!   `DrawIndex`), so every vertex of a primitive stores the same index, and that index selects
//!   "the viewport transform, scissor rectangle, and exclusive scissor rectangle" (its
//!   `ViewportIndex`), which its `VkPipeline` takes `k` of. Each region's draw so rasterizes
//!   through the viewport and scissor a draw of it alone takes. The regions' scissors are
//!   disjoint, so a pixel's fragments all come from its region's draw, whose primitives keep the
//!   specification's primitive order (instances, then vertices or indices), whatever order the
//!   specification leaves between a command's draws: the pixels are the region by region
//!   route's. A fragment
//!   function that reads `[[viewport_array_index]]` reads 0 under it, as it does drawn region by
//!   region. Neither the route nor `k` follows the map: a draw of the pass routes so over any
//!   map, the identity and a map of no layers among them, so a reset (below) never changes the
//!   pipeline its pass's encoding again draws with. Every other draw, and every draw on another
//!   device, records region by region.
//! - **Layers**. Metal rasterizes each primitive by the map's layer of the slice it renders (probes
//!   on the M3 Max and an M4, alike with and without API validation: over a 2-slice target and a
//!   map whose layer 0 is at quality 0.5 and layer 1 the identity, a vertex function's slice,
//!   amplifications mapped to slices 0 and 1 in either order, one mapped past the pass to slice 0,
//!   and a count of 1 moved to slice 1 each shaded a slice by that slice's layer). Over a map whose
//!   layers are alike every draw takes layer 0's plan, layered or amplified, views included. Over
//!   one whose layers differ a draw that renders one slice takes that slice's layer's plan: a draw
//!   that renders the first, one a count of 1 moves to its slice, and each amplification of an
//!   amplified draw, which the encoder then draws one amplification at a time so that each records
//!   its own regions; a pipeline whose vertex function chooses its slice is refused
//!   (`LayeredPipeline`), its slices unknown to the encoder.
//! - **The scissor.** Metal applies a scissor in screen pixels (probe 16: a 16-pixel scissor kept
//!   8 physical columns, one at (8, 8, 16, 16) the physical pixels 4 to 11), and its validation
//!   warns that a viewport or scissor that is not full screen "may not behave as expected on this
//!   platform". A scissor that holds the viewport's pixels clips nothing the viewport does not, so
//!   the plan takes none; one that cuts into the viewport over a map that is not the identity is
//!   refused, as the compile layer refuses it ([`RegionRefusal::Scissor`]).
//! - **A map of no layers, and a mutable map not yet reset**. Metal makes a map of no layers
//!   where its API validation is off, and a pass over one made while no map of layers has been
//!   released runs its load, clear, and store actions and its draws write nothing (an M4, each in
//!   its own process three times: a full-screen triangle over a cleared target of 8 and of 64
//!   pixels, with and without a scissor, a map of 256 pixels over a target of 8, beside a live map
//!   of one layer, and in a sequence of passes); a pass over a mutable map not yet reset alike
//!   (the M3 Max under macOS 27.0.1). Metal runs each draw's vertex function there (the M3 Max:
//!   a vertex function counting its runs ran 3 times for a triangle, over a map made of no
//!   layers, immutable or mutable, and over a mutable one not yet reset), so the runtime records
//!   each draw through an empty scissor: its vertex function runs, and no fragment. A map of no
//!   layers made after a map of layers was released, or reset from layers, draws on Metal, an M4
//!   and an M3 Max alike: a 64-pixel one over a 64-pixel target its physical rows 0 to 15 at full
//!   width, at the positions of a pass with no map, whatever map was released. The runtime draws
//!   nothing there and announces it: a draw over such a map announces
//!   `zero-layer-map-after-release`.
//! - **A reset** (`RasterizationRateMap::reset_using_descriptor`). Metal's pass reads the map as
//!   the GPU runs it (our probe `rm`, the M3 Max under macOS 27.0.1, each case in a process of its
//!   own: a reset from the identity to half quality after the encoder was made, after its draw,
//!   after `endEncoding`, after `commit` and after `commit` while the command buffer waited for an
//!   event the CPU signalled after the reset each shaded the half-quality map's pixels; two passes
//!   of one waiting command buffer, the first ended before a reset and both before a second, shaded
//!   the second's; a reset after the command buffer completed changed nothing). The runtime plans
//!   each draw by the map as its encoder took it as it opened, which the encoder notes
//!   (`EncoderCore::take_rate_map`), and keeps the pass's log (`render/replay.rs`). A reset puts
//!   the map's new state in place under the lock a submission reads its mark under, and marks each
//!   committed command buffer whose last event group is not yet submitted and whose pass took the
//!   map; their passes that took it are encoded again from their logs, in the reset's call, before
//!   that submission (`command_buffer/reencode.rs`), and a command buffer not yet committed has
//!   them encoded again at its commit. A group submitted before the reset draws by the map as its
//!   passes took it: Metal's GPU, already reading, may do either. An encoder that kept no log (an
//!   encoder opened while a buffer of the device was detached, one that left work for a host
//!   step), or whose encoding again is refused (the pass
//!   checked against the map as reset), draws by the map as it took it, announced
//!   (`rate-map-reset-unencoded`).
//! - **A reset to another screen** (our probe `rs`, the M3 Max under macOS 27.0.1: 101 passes
//!   over a mutable map of the identity of a square screen reset to the identity of another, each
//!   case in a process of its own, repeats alike). A pass whose map was reset after its encoder
//!   took it, whether after the encoder opened, after it ended or after the commit of a waiting
//!   command buffer, keeps the screen it took: its default viewport, scissor and attachment-less
//!   size are that screen's, and it shades the physical pixels to [`reset_bound`] on each axis:
//!   to the 32-pixel tiles of the new screen where the new screen spans one tile fewer than the
//!   screen taken (64 pixels reset to 32 or to 24 shades 32, 128 reset to 80 shades 96), else to
//!   the screen taken (16 reset to 8 shades 16, 128 reset to 8, to 64 or to 100 shades 128, 8
//!   reset to 16 shades 8), within the attachments. A reset before the encoder opened is the new
//!   map's alone. The runtime's encoding again keeps the screen the pass took
//!   (`RenderLog::screen`) and draws so over a map reset to the identity; over a map reset to
//!   another, which no probe measured, it draws by the map as reset within the screen taken,
//!   inside the near-equivalent the map's compile announces (`rasterization-rate-map`).
//! - **A mutable map's pass past the screen** (the M3 Max under macOS 27.0.1, a quad far past the
//!   clip volume): Metal rasterizes a pass over a `Mutable` map past the screen's edge along the
//!   last tiles' lines, to a bound not found (a 64-pixel screen at half quality shaded 49 physical
//!   columns of its 33, one at 0.75 shaded 64 of 54), where a pass over a map of any other
//!   mutability stops at the physical size. An identity map's pass stopped at its physical size
//!   in every case measured, mutable or not. The runtime's pass stops at the physical size, inside
//!   the near-equivalent the map's compile announces (`rasterization-rate-map`).
//! - **A pass of no draw call over a mutable map** (the M3 Max under macOS 27.0.1, our probe
//!   `rs`, each case in a process of its own): Metal's GPU hangs at it, failing its command
//!   buffer with `GpuHang`, over a `Mutable` map of layers or of none, reset or never, with only a
//!   pipeline state set; and runs it, its clear stored, over a map of any other mutability, and
//!   over a mutable one once a command buffer with a draw call over a mutable map (that map or
//!   another; a draw of no vertices too) was committed and ran before. A drawn pass earlier in
//!   its own command buffer does not keep it from hanging, nor does one over an immutable map or
//!   none. The runtime fails the command buffer from such a pass on as Metal's GPU hang at a draw
//!   of primitive type 9 (`command_buffer/hang.rs`), where no command buffer of the device with a
//!   draw call over a mutable map was committed before (`DeviceShared::mutable_rate_map_called`,
//!   raised at that commit).
//! - **Pieces.** Drawn region by region, a pass's executed commands may hold more GPU work than a
//!   driver lets one kernel submission run: amdgpu ends a job that runs past its timeout (2 s by
//!   default) with a reset of the GPU, and the work is lost. So where an execution's commands
//!   record inline, at commit or at the host step (`render/executions.rs`; a rate-mapped pass's
//!   executions always do), each command's region draws count their work as they record
//!   ([`draw_work`]: each region draw's vertices or indices times its instances, and
//!   [`REGION_DRAW_WORK`] for the draw itself), across the pass's executions. Once it reaches
//!   [`PIECE_WORK`], the next command that draws begins the execution's next run, in a render pass
//!   instance of its own after a barrier from everything before, as a command's barrier begins
//!   one, and that run begins a *piece*: the pass's recording ends its `VkCommandBuffer` before it
//!   and goes on in a new one (`EncoderCore::cut_piece`), and its batch's submission submits each
//!   piece but the last as a `VkSubmitInfo` of its own that signals its `VkQueue`'s timeline of
//!   pieces (`command_buffer/submit.rs`). Vulkan defines no kernel submission; Mesa's common queue
//!   merges a submission that signals nothing into the next one's (`command_buffer/group.rs`) and
//!   keeps one that signals apart, so there each piece runs as a kernel submission of its own. The
//!   pixels are the uncut pass's: each run loads what the one before stored, and the commands keep
//!   their order. Each cut costs a store and a load of the pass's attachments.
//!
//!   A command's draw that alone would hold more than [`PIECE_WORK`] is drawn as consecutive
//!   ranges of its indices, each a draw of its own that a piece may begin before
//!   ([`index_range`]: as many indices as fit [`PIECE_WORK`] at the most region draws a draw of
//!   the execution's commands made so far, or at the most a plan holds before any), where the
//!   ranges draw what Metal's one draw does: an indexed draw of points, lines or triangles (a
//!   list, which Metal never restarts, cut at a primitive's end), of one instance (a cut of
//!   several would put an instance's later primitives before the next instance's earlier ones),
//!   whose pipeline reads no `[[primitive_id]]` (each range counts its primitives from 0) and
//!   writes no memory. Each range's vertices, instance and base vertex are the draw's
//!   (`WorkCounts::rate_map_index_ranges`). Any other draw rides whole in the piece it begins in,
//!   however much work it holds: a strip, an instanced draw, a draw without indices, a mesh or
//!   patch draw, one of such a pipeline; and the encoder's own draws are not cut at all
//!   (`docs/FAST-PATHS.md`, known gaps). So a piece of draws cut so holds less than twice
//!   [`PIECE_WORK`].
//!
//!   The bound is a forecast, not a measure: twice [`PIECE_WORK`] is about 56 ms of an Iris Xe's
//!   GPU at a trivial vertex function, so a piece whose vertex function costs 30 times as much
//!   still runs within 2 s there. Where the device writes timestamps each piece of a pass recorded
//!   in several is timed (`WorkCounts::rate_map_piece_gpu_nanos_most`), also on a device asked to
//!   answer command buffers' GPU times from the host (`DeviceOptions::gpu_times_from_host`), but
//!   not where its driver's timestamps wait for every queue's
//!   (`DriverFact::TimestampsWaitForEveryQueue`), and the pieces are counted
//!   (`WorkCounts::rate_map_pieces`).
//! - **Fan-out.** Drawn region by region, a draw runs its vertex function once per region of the
//!   plan, which may hold thousands. Where a host step records an execution's commands
//!   (`render/executions.rs`), under `RateMapRegionPolicy::FanOut` (the default) on a device that
//!   enables `geometryShader`, `shaderClipDistance` and `vertexPipelineStoresAndAtomics`, a draw
//!   of triangles filled, of one amplification, without a compensation or a band, whose pipeline
//!   chooses no slice or viewport, over a map whose layers are alike and whose plan is a grid of
//!   at most 128 columns and rows (`bind::fan_out::table`), is drawn once instead: its variant's
//!   geometry stage (`VariantKey::region_fan_out`) emits a copy of each triangle for each region
//!   it touches, through one viewport over the render area, each copy's window coordinates its
//!   region viewport's and four clip distances keeping it to its region's pixels. Before the
//!   pass, the step records the commands' draws once more as a probe whose geometry stage emits
//!   nothing: for each draw it flags a triangle that crosses `w = 0` or touches more regions than
//!   the fan-out's copies (each region widened by an eighth of a pixel), and counts the copies of
//!   the others. The probe is submitted on its own and waited for (`WorkCounts::rate_map_probes`,
//!   its GPU time `WorkCounts::rate_map_probe_gpu_nanos`); then a flagged draw is drawn region by
//!   region (`WorkCounts::rate_map_flagged_draws`) and the others fanned out
//!   (`WorkCounts::rate_map_fanned_draws`), each in its command's place, so a pixel's fragments
//!   keep their order: a pixel lies in one region. The probe also marks a draw a triangle of
//!   which touches more regions than few copies (`bind::fan_out::FEW_COPIES`); a draw it marks
//!   none of is drawn through the variant of that many copies (`RegionFanOut::DrawFew`,
//!   `WorkCounts::rate_map_fanned_few_draws`), whose geometry stage declares the output of fewer
//!   vertices. A fanned-out draw's work is its vertices, its
//!   triangles' geometry stage and three vertices per copy ([`fan_out_work`]), and it is never
//!   cut into ranges. The probe stops at a command's barrier, and after a draw whose pipeline
//!   writes memory, as it runs before every draw of the pass: the later commands are drawn region
//!   by region. A pass whose attachments are memoryless or that uses a shadowed texture is not
//!   probed, nor an execution whose commands record as several runs. `commit`'s settle of the
//!   pass (`RenderExecution::settle`) begins the probe's and the fanned-out draws' variants and
//!   `VkPipeline`s on the device's pool beside the draws' own, so the step takes them made, or
//!   waits for them (`WorkCounts::rate_map_fan_out_pipelines_ready`,
//!   `rate_map_fan_out_pipelines_waited`), and makes none itself for the states the settle
//!   foresaw, the regions variant a flagged draw takes included. The probe and the fanned-out
//!   draws each upload the region table to the command buffer as they first bind it
//!   (`WorkCounts::rate_map_fan_tables_uploaded`); the settle, which records nothing, uploads
//!   none. A copy's window coordinates are computed in the geometry stage, so a pixel centre
//!   within a rounding error of a triangle's edge may be covered by one route and not by the
//!   other. The encoder's own draws,
//!   and the commands `commit` records, are drawn region by region (`docs/FAST-PATHS.md`, known
//!   gaps).
//! - **What the plan refuses** is `NotImplemented(RuntimeGap::RateMapRegions)`, naming the
//!   compile layer's reason ([`RegionRefusal`]): a viewport of no extent or a flipped one, a
//!   scissor as above, a plan past its bound or past the device's viewport limits, a pipeline whose
//!   vertex function chooses its slice over a map whose layers differ, one that selects its
//!   viewport, and a vertex function that writes memory in a plan of several regions, which would
//!   write once per region.

use super::degradation::Degradation;
use super::device::functions::Vk;
use super::error::{internal, invalid, Error, Problem, RuntimeGap};
use super::exit_guard::{self, Call};
use super::gpu_time::{FromHost, Slot};
use super::rate_map::{MapState, RasterizationRateMap};
use super::render_pass::{PassLayout, RenderPassDescriptor, RenderPassProblem};
use super::render_values::{ScissorRect, Viewport};
use super::Device;
use ash::vk;
use metal2vulkan::bind::{RegionError, RegionLimits};
use metal2vulkan::metal::Mutability;
use metal2vulkan::pipeline::GraphicsState;
use metal2vulkan::vk::DrawCommand;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Why a draw in a rate-mapped pass is not drawn region by region: the compile layer's
/// [`RegionError`], by kind (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionRefusal {
    /// The viewport has no width or height, where Metal's answer in a rate-mapped pass is not
    /// measured.
    EmptyViewport,
    /// The viewport is flipped (a negative width or height), which the plan does not compose.
    NegativeViewport,
    /// A scissor that cuts into the viewport, over a map that is not the identity.
    Scissor,
    /// The plan would hold more regions than its bound.
    TooManyRegions,
    /// A region's viewport is past the device's `maxViewportDimensions` or
    /// `viewportBoundsRange`, or its extent is not a normal `f32`.
    ViewportLimits,
    /// The pipeline may reach a layer other than the one planned: its vertex function chooses its
    /// slice, over a map whose layers differ.
    LayeredPipeline,
    /// The pipeline's vertex function selects each primitive's viewport, where the plan sets one
    /// viewport per region.
    ViewportArray,
    /// The pipeline's vertex function writes memory, which a plan of several regions would run
    /// once per region.
    VertexSideEffects,
}

/// A render pass's rasterization-rate map, checked, with the pass's screen area (the module
/// docs).
#[derive(Clone)]
pub(super) struct RateMappedPass {
    map: RasterizationRateMap,
    /// The map as the encoder took it, which its draws are planned by.
    state: Arc<MapState>,
    /// The map's screen size as the pass first took it, which an encoding again for a reset
    /// keeps (the module docs).
    screen: [u32; 2],
    area: [u32; 2],
    /// Every layer of the map is the first's: each slice is rasterized alike.
    alike: bool,
}

/// What `descriptor` asks of its map before its attachments are read: the map is this device's,
/// and the pass's array length is at most its layer count where its layers differ (the module
/// docs).
pub(super) fn check_descriptor(
    call: &'static str,
    device: &Device,
    descriptor: &RenderPassDescriptor,
) -> Result<(), Error> {
    let Some(map) = &descriptor.rasterization_rate_map else {
        return Ok(());
    };
    if !map.root_device().same(device) {
        return Err(map.root_device().foreign(call, device));
    }
    let state = map.state();
    let layers = state.layer_count();
    // Past the layers of a map whose layers are alike every slice takes layer 0, as plain Metal
    // rasterizes past one layer (the module docs).
    if descriptor.render_target_array_length > u64::from(layers) && !state.layers_alike() {
        return Err(invalid(
            call,
            Problem::RenderPass(RenderPassProblem::RateMapArrayLength {
                length: descriptor.render_target_array_length,
                layers,
            }),
        ));
    }
    Ok(())
}

impl RateMappedPass {
    /// The map of `descriptor`, whose pass renders as `layout`, checked against its render target
    /// size where it has no attachments (the module docs), its screen `screen` where the pass
    /// took the map before (an encoding again for a reset keeps it), else the map's; `None` for a
    /// pass without one.
    pub(super) fn of(
        call: &'static str,
        descriptor: &RenderPassDescriptor,
        layout: &PassLayout,
        screen: Option<[u32; 2]>,
    ) -> Result<Option<RateMappedPass>, Error> {
        let Some(map) = &descriptor.rasterization_rate_map else {
            return Ok(None);
        };
        let state = map.state();
        let screen = screen.unwrap_or_else(|| state.screen_size());
        let attached =
            layout.colors.iter().flatten().next().is_some() || layout.depth_stencil.is_some();
        let area = if !attached {
            // Without attachments the render target size is the area (`render_pass::area`).
            if layout.extent[0] > screen[0] || layout.extent[1] > screen[1] {
                return Err(invalid(
                    call,
                    Problem::RenderPass(RenderPassProblem::RateMapTargetSize {
                        target: layout.extent,
                        screen,
                    }),
                ));
            }
            layout.extent
        } else {
            // Attachments smaller than the physical size take the physical pixels in them: the
            // render area cuts each region's scissor (the module docs).
            screen
        };
        Ok(Some(RateMappedPass {
            map: map.clone(),
            alike: state.layers_alike(),
            state,
            screen,
            area,
        }))
    }

    /// The map and the map as the pass took it, which its encoder notes (`EncoderCore::take_rate_map`).
    pub(super) fn taken(&self) -> (RasterizationRateMap, Arc<MapState>) {
        (self.map.clone(), Arc::clone(&self.state))
    }

    /// Whether every layer of the map is the first's, so a draw rasterizes every slice alike
    /// whichever slices it renders (the module docs).
    pub(super) fn layers_alike(&self) -> bool {
        self.alike
    }

    /// Whether the map is `Mutable`, over which Metal's GPU hangs at a pass of no draw call until
    /// a command buffer with a draw call over such a map was committed (the module docs).
    pub(super) fn mutable(&self) -> bool {
        self.map.mutability() == Mutability::Mutable
    }

    /// The pass's screen area: the extent its encoder's viewports and scissor rectangles are read
    /// in and default to.
    pub(super) fn area(&self) -> [u32; 2] {
        self.area
    }

    /// The `(viewport, scissor)` each recording of a draw of `graphics` takes, one per region of
    /// the plan for the encoder's first `viewport` and `scissor`, within `limits`, its scissor
    /// cut to the pixels whose centres lie inside its viewport and to the render area `extent`:
    /// the regions the cut leaves a pixel, else the first alone (the module docs). `slice` is the one slice of the pass the recording renders (a draw of
    /// one amplification, or one its view mapping moves), whose map layer is the plan's, or
    /// `None` where its primitives choose their slices (a vertex function that writes it, or a
    /// multiview draw's views), which only a map of layers alike plans.
    pub(super) fn draws(
        &self,
        slice: Option<u32>,
        viewport: &Viewport,
        scissor: &ScissorRect,
        graphics: &GraphicsState,
        limits: &RegionLimits,
        extent: [u32; 2],
    ) -> Result<Vec<(vk::Viewport, vk::Rect2D)>, Error> {
        let guest = [
            viewport.origin_x,
            viewport.origin_y,
            viewport.width,
            viewport.height,
        ];
        // A scissor that holds every pixel the viewport overlaps clips nothing more.
        let pixels = |origin: f64, size: f64| [origin.floor(), (origin + size).ceil()];
        let [vx, vy] = [
            pixels(viewport.origin_x, viewport.width),
            pixels(viewport.origin_y, viewport.height),
        ];
        let holds = |from: u64, size: u64, [lo, hi]: [f64; 2]| {
            from as f64 <= lo.max(0.0) && (from + size) as f64 >= hi
        };
        let clip = (!(holds(scissor.x, scissor.width, vx) && holds(scissor.y, scissor.height, vy)))
            .then(|| {
                // Inside the area, which fits a `u32` (checked at the setter).
                [scissor.x, scissor.y, scissor.width, scissor.height].map(|v| v as u32)
            });
        // A map not yet reset, and a map of no layers, rasterize nothing: the draw runs its vertex
        // function alone, through an empty scissor, as Metal's does. One made, or reset, after a
        // map of layers is one Metal draws from that map's memory: announced.
        let compiled = match self.state.compiled() {
            Some(map) if map.layer_count() > 0 => map,
            compiled => {
                if self.state.made_after_release() {
                    let [width, height] = self.state.screen_size();
                    self.map
                        .generation()
                        .degrade(Degradation::ZeroLayerMapAfterRelease, || {
                            (
                                format!(
                                    "a render pass over a rasterization rate map of no layers, \
                                     {width} by {height} screen pixels, made after a map of \
                                     layers was released, or reset from layers"
                                ),
                                "Metal draws such a pass (an Apple M4 and an Apple M3 Max drew \
                                 a 64-pixel map's rows 0 to 15 at the positions of a pass with \
                                 no map, whatever map was released); the runtime draws \
                                 nothing"
                                    .to_owned(),
                            )
                        });
                }
                debug_assert!(compiled.is_none_or(|map| map.layer_count() == 0));
                let viewport = vk::Viewport {
                    x: 0.0,
                    y: 1.0,
                    width: 1.0,
                    height: -1.0,
                    min_depth: viewport.znear as f32,
                    max_depth: viewport.zfar as f32,
                };
                return Ok(vec![(viewport, vk::Rect2D::default())]);
            }
        };
        // Metal rasterizes each primitive by the map's layer of the slice it renders: the plan of
        // that slice's layer bounds a draw of one slice, and the first layer's any draw over a map
        // of layers alike; a draw whose primitives choose over layers that differ is refused as
        // layered.
        let (layer, bounded) = match (slice, self.alike) {
            (_, true) => (0, true),
            (Some(slice), false) => (slice, true),
            (None, false) => (0, false),
        };
        let plan = compiled
            .regions(layer, guest, clip, limits)
            .and_then(|plan| {
                match bounded {
                    true => plan.admits_planned_layers(graphics),
                    false => plan.admits(graphics),
                }
                .map(|()| plan)
            })
            .map_err(refusal)?;
        // Over the identity reset to another screen, Metal's pass keeps the screen it took and
        // shades physical pixels to `reset_bound`, along the identity (the module docs); the
        // plan's refusals hold alike.
        let now = compiled.screen_size();
        if self.screen != now && compiled.is_identity() {
            let bound = [0, 1].map(|a| reset_bound(self.screen[a], now[a]));
            let [x0, y0, x1, y1] = match clip {
                Some([x, y, w, h]) => [x, y, x.saturating_add(w), y.saturating_add(h)],
                None => [0, 0, u32::MAX, u32::MAX],
            };
            let (sx, ex) = (x0.min(bound[0]), x1.min(bound[0]));
            let (sy, ey) = (y0.min(bound[1]), y1.min(bound[1]));
            let region = (
                guest.map(|v| v as f32),
                [sx, sy],
                [ex.saturating_sub(sx), ey.saturating_sub(sy)],
            );
            return Ok(vec![on_device(region, viewport, extent)]);
        }
        let all = plan.regions().iter().map(|region| {
            let r = (
                region.viewport(),
                region.scissor_offset(),
                region.scissor_extent(),
            );
            on_device(r, viewport, extent)
        });
        // A region the cut leaves no pixel shades nothing, and its vertex function may repeat
        // nothing (a plan of several regions refuses one that writes memory): it is not drawn.
        // Where no region holds a pixel the first is drawn alone, through its empty scissor, so
        // the vertex function still runs.
        let first = all.clone().take(1);
        let drawn: Vec<_> = all
            .filter(|(_, scissor)| scissor.extent.width > 0 && scissor.extent.height > 0)
            .collect();
        Ok(match drawn.is_empty() {
            true => first.collect(),
            false => drawn,
        })
    }
}

/// The device's viewport and scissor of a region of a rate-mapped pass, `([x, y, width, height],
/// scissor offset, scissor extent)` in physical pixels, its viewport y down: the scissor cut to the
/// pixels whose centres lie inside the viewport and to the render area `extent` (the module docs),
/// the depth range the guest's `viewport`'s.
fn on_device(
    ([x, y, width, height], [sx, sy], [sw, sh]): ([f32; 4], [u32; 2], [u32; 2]),
    viewport: &Viewport,
    extent: [u32; 2],
) -> (vk::Viewport, vk::Rect2D) {
    let area = extent.map(f64::from);
    // The pixels whose centres lie in `[origin, origin + length)`: from `ceil(origin - 0.5)` up
    // to `ceil(origin + length - 0.5)`.
    let cut = |from: u32, size: u32, origin: f32, length: f32, limit: f64| {
        let centre = |edge: f64| (edge - 0.5).ceil().clamp(0.0, limit) as u32;
        let lo = centre(f64::from(origin));
        let hi = centre(f64::from(origin) + f64::from(length));
        let first = from.max(lo);
        let end = from.saturating_add(size).min(hi).max(first);
        (first, end - first)
    };
    let (ox, ex) = cut(sx, sw, x, width, area[0]);
    let (oy, ey) = cut(sy, sh, y, height, area[1]);
    (
        vk::Viewport {
            x,
            y: y + height,
            width,
            height: -height,
            min_depth: viewport.znear as f32,
            max_depth: viewport.zfar as f32,
        },
        vk::Rect2D {
            // Inside the render area: each fits an `i32`.
            offset: vk::Offset2D {
                x: ox as i32,
                y: oy as i32,
            },
            extent: vk::Extent2D {
                width: ex,
                height: ey,
            },
        },
    )
}

/// How far along one axis a pass over the identity shades, its map reset after the pass took it
/// from a screen of `taken` pixels to one of `now` (the module docs, our probe `rs`): to the
/// 32-pixel tiles of `now` where it spans one tile fewer than `taken`, else to `taken`.
pub(super) fn reset_bound(taken: u32, now: u32) -> u32 {
    let tiles = |side: u32| side.div_ceil(32);
    match tiles(taken).checked_sub(tiles(now)) {
        Some(1) => (tiles(now) * 32).min(taken),
        _ => taken,
    }
}

/// The work a piece of a rate-mapped pass's recorded commands holds before the next command
/// begins another (the module docs, **Pieces**), in vertex invocations: about 28 ms of an Intel
/// Iris Xe's GPU (ANV) at the 0.42 ns a vertex invocation of a trivial vertex function cost it.
const PIECE_WORK: u64 = 1 << 26;

/// What one region's draw costs beyond its vertices, in vertex invocations (the module docs,
/// **Pieces**): about 240 ns of an Iris Xe's GPU (ANV) per region drawn 16 at a time, the
/// viewports' and scissors' setting among it.
const REGION_DRAW_WORK: u64 = 512;

/// The work a piece holds before the next command begins another ([`PIECE_WORK`]; a device
/// test's own where it set one, [`set_piece_work`]).
pub(super) fn piece_work() -> u64 {
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    if let Some(work) = std::num::NonZeroU64::new(TEST_PIECE_WORK.load(Ordering::Relaxed)) {
        return work.get();
    }
    PIECE_WORK
}

/// The work of `draw` recorded through `scissors`, a recording's (the module docs, **Pieces**):
/// one region draw for each scissor that holds a pixel (at least one), each of the draw's
/// vertices or indices times its instances, an indirect or mesh draw's counted as none.
pub(super) fn draw_work(scissors: &[vk::Rect2D], draw: &DrawCommand) -> u64 {
    let regions = region_draws(scissors);
    regions.saturating_mul(vertices(draw).saturating_add(REGION_DRAW_WORK))
}

/// The work a fanned-out `draw` holds past [`draw_work`]'s of its one recording (the module docs,
/// **Fan-out**): its geometry stage's invocations, at most one a vertex, and the three vertices
/// of each of the `copies` the probe counted.
pub(super) fn fan_out_work(draw: &DrawCommand, copies: u32) -> u64 {
    vertices(draw).saturating_add(3 * u64::from(copies))
}

/// The vertices or indices of `draw` times its instances, an indirect or mesh draw's counted as
/// none.
fn vertices(draw: &DrawCommand) -> u64 {
    match *draw {
        DrawCommand::Draw {
            vertex_count,
            instance_count,
            ..
        } => u64::from(vertex_count) * u64::from(instance_count),
        DrawCommand::DrawIndexed {
            index_count,
            instance_count,
            ..
        } => u64::from(index_count) * u64::from(instance_count),
        DrawCommand::DrawIndirect { .. }
        | DrawCommand::DrawIndexedIndirect { .. }
        | DrawCommand::DrawMeshTasks { .. }
        | DrawCommand::DrawMeshTasksIndirect { .. }
        | DrawCommand::DrawMeshTasksIndirectCount { .. } => 0,
    }
}

/// The region draws a recording through `scissors` makes (the module docs, **Pieces**): one for
/// each scissor that holds a pixel, at least one.
pub(super) fn region_draws(scissors: &[vk::Rect2D]) -> u64 {
    (scissors.iter())
        .filter(|scissor| scissor.extent.width > 0 && scissor.extent.height > 0)
        .count()
        .max(1) as u64
}

/// The most indices of one range of a draw cut into ranges (the module docs, **Pieces**) whose
/// recordings make `regions` region draws an index, of primitives of `per_primitive` vertices:
/// what fits the piece work, whole primitives of an even count, so a 16-bit range starts 4-byte
/// aligned as its draw does, at least two.
pub(super) fn index_range(regions: u64, per_primitive: u32) -> u32 {
    let step = 2 * u64::from(per_primitive);
    let fits = (piece_work() / regions.max(1)).saturating_sub(REGION_DRAW_WORK);
    let most = fits.min(u64::from(u32::MAX));
    (most / step * step).max(step) as u32
}

/// The GPU times of a pass's pieces (the module docs, **Pieces**): a timestamp slot of the
/// device's for each (`gpu_time::Slot`), its start written before the piece's first command and
/// its end after its last; read, counted (`WorkCounts::rate_map_piece_gpu_nanos`) and given back
/// as the command buffer lets go of what it retained, once its work completed or it failed. Or
/// the GPU time of a host step's probe (the module docs, **Fan-out**), one piece, counted
/// (`WorkCounts::rate_map_probe_gpu_nanos`) and given back once the step waited for it.
pub(super) struct PieceTimes {
    device: Device,
    slots: Vec<Slot>,
    probe: bool,
}

impl PieceTimes {
    /// Slots for the `pieces` pieces of a pass recorded on `device`, reset on the host where the
    /// device resets them there; `None` where it has fewer than two, the device writes no
    /// timestamps or its driver's timestamps wait for every queue's (a device asked to answer
    /// command buffers' GPU times from the host still times pieces), or a slot cannot be had.
    pub(super) fn take(device: &Device, pieces: usize) -> Option<PieceTimes> {
        if pieces < 2 {
            return None;
        }
        PieceTimes::slots(device, pieces, false)
    }

    /// A slot for a host step's probe recorded on `device` (the module docs, **Fan-out**), its one
    /// piece; `None` as for [`take`](Self::take).
    pub(super) fn probe(device: &Device) -> Option<PieceTimes> {
        PieceTimes::slots(device, 1, true)
    }

    /// `pieces` slots of `device`'s, a probe's where `probe` says.
    fn slots(device: &Device, pieces: usize, probe: bool) -> Option<PieceTimes> {
        let shared = device.shared();
        // Timestamps that wait for every queue's would serialize the device's queues.
        if shared.timing.host_clock_reason() == Some(FromHost::Driver) {
            return None;
        }
        let raw = shared.vk();
        let slots = (0..pieces)
            // SAFETY: the timing is the live device's.
            .map(|_| unsafe { shared.timing.slot(raw, &shared.counts) })
            .collect::<Option<Vec<Slot>>>()?;
        Some(PieceTimes {
            device: device.clone(),
            slots,
            probe,
        })
    }

    /// Record piece `piece`'s start into `cb`: its queries reset first where the device does not
    /// reset them on the host.
    ///
    /// # Safety
    ///
    /// `cb` is a command buffer of the device's recording outside a render pass, and these times
    /// are held until it completes.
    pub(super) unsafe fn begin(&self, raw: &Vk, cb: vk::CommandBuffer, piece: usize) {
        let Some(slot) = self.slots.get(piece) else {
            return;
        };
        // SAFETY: the caller's; the slot's two queries are its own, which no other work uses.
        unsafe {
            if !self.device.shared().timing.resets_on_host() {
                raw.cmd_reset_query_pool(cb, slot.queries(), slot.first(), 2);
                exit_guard::called(Call::CmdResetQueryPool);
            }
            let top = vk::PipelineStageFlags::TOP_OF_PIPE;
            raw.cmd_write_timestamp(cb, top, slot.queries(), slot.first());
        }
        exit_guard::called(Call::CmdWriteTimestamp);
    }

    /// Record piece `piece`'s end into `cb`.
    ///
    /// # Safety
    ///
    /// As [`begin`](Self::begin)'s, after the piece's last command.
    pub(super) unsafe fn end(&self, raw: &Vk, cb: vk::CommandBuffer, piece: usize) {
        let Some(slot) = self.slots.get(piece) else {
            return;
        };
        let bottom = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
        // SAFETY: the caller's.
        unsafe { raw.cmd_write_timestamp(cb, bottom, slot.queries(), slot.first() + 1) };
        exit_guard::called(Call::CmdWriteTimestamp);
    }
}

impl Drop for PieceTimes {
    fn drop(&mut self) {
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let shared = self.device.shared();
        let Some(clock) = shared.timing.clock() else {
            return;
        };
        let raw = shared.vk();
        for slot in &self.slots {
            // Each query's value, then its availability: a piece that never ran reads none.
            let mut result = [[0u64; 2]; 2];
            // SAFETY: the slot's queries, reset before the pass was recorded, written by work that
            // completed or never ran; the result is 64-bit with availability, two words a query.
            let read = unsafe {
                raw.get_query_pool_results(
                    slot.queries(),
                    slot.first(),
                    &mut result,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
                )
            };
            exit_guard::called(Call::GetQueryPoolResults);
            let [[start, started], [end, ended]] = result;
            if matches!(read, Ok(()) | Err(vk::Result::NOT_READY)) && started != 0 && ended != 0 {
                let nanos = clock.duration(start, end);
                match self.probe {
                    true => shared.counts.rate_map_probe_time(nanos),
                    false => shared.counts.rate_map_piece_time(nanos),
                }
            }
        }
    }
}

/// A device test's piece work in place of [`PIECE_WORK`] (0: none), so a small pass is recorded
/// in pieces.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
static TEST_PIECE_WORK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Set the piece work a device test's passes are recorded in pieces by (`None`: [`PIECE_WORK`]).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn set_piece_work(work: Option<u64>) {
    TEST_PIECE_WORK.store(work.unwrap_or(0), Ordering::Relaxed);
}

/// The runtime's refusal of the compile layer's `error` (the module docs).
fn refusal(error: RegionError) -> Error {
    let kind = match error {
        RegionError::EmptyViewport { .. } => RegionRefusal::EmptyViewport,
        RegionError::NegativeViewport { .. } => RegionRefusal::NegativeViewport,
        RegionError::Scissor { .. } => RegionRefusal::Scissor,
        RegionError::TooManyRegions { .. } => RegionRefusal::TooManyRegions,
        RegionError::ViewportDimensions { .. }
        | RegionError::ViewportBounds { .. }
        | RegionError::RegionExtent { .. } => RegionRefusal::ViewportLimits,
        RegionError::LayeredPipeline => RegionRefusal::LayeredPipeline,
        RegionError::ViewportArray => RegionRefusal::ViewportArray,
        RegionError::VertexSideEffects { .. } => RegionRefusal::VertexSideEffects,
        // A slice is below the pass's length, which is at most the map's layer count (checked
        // as the encoder is made), and the setters refuse a viewport that is not finite.
        other => return internal(&format!("a rate-mapped draw's plan: {other}")),
    };
    Error::NotImplemented(RuntimeGap::RateMapRegions(kind))
}

/// Announce, at the commit of its command buffer, that an encoder whose pass took a
/// rasterization-rate map since reset, and that kept no log (`unlogged` says why, where known),
/// draws by the map as it took it (the module docs).
pub(super) fn degrade_unencoded(
    device: &Device,
    unlogged: Option<crate::command_buffer::rebound_use::Unlogged>,
) {
    device.degrade(Degradation::RateMapResetUnencoded, || {
        let why = match unlogged {
            Some(why) => format!(", not encoded again as {why}"),
            None => String::new(),
        };
        (
            format!(
                "a render pass over a rasterization rate map reset after the pass took it{why}"
            ),
            "the pass draws by the map as it took it, where Metal's GPU draws by the map as reset"
                .to_owned(),
        )
    });
}
