//! The render encoder's state (step 6's second part): Metal's defaults, what each setter takes and
//! refuses, and the state in Vulkan's terms that a draw takes, statically in its pipeline's
//! graphics key or as dynamic state.
//!
//! Metal's answers were measured on an M4 (macOS 27.0.1, by our Metal probes, each case without and
//! with API validation). "Stops" is the validation layer's assertion, typed `Invalid` here.
//!
//! - **Viewports** (`set_viewport`, `set_viewports`). Metal's defaults cover the render area. A
//!   count outside 1 to 16 stops, and so does a NaN or infinite field (measured for the origin; the
//!   other fields by the same rule). Plain Metal draws nothing through a viewport whose origin x is
//!   NaN (Metal probe case `viewport_nan`), as the encoder does, holding it as a viewport of no
//!   width; any other field NaN or infinite, unmeasured, is refused. Metal takes a viewport past
//!   the render area, a negative height (a flip), and a depth range past `[0, 1]`. A viewport is
//!   Vulkan's with its height negated (`y + height`, `-height`), so framebuffer rows, `FragCoord`
//!   and derivatives follow Metal's y-up clip space; Metal's own negative height is then a positive
//!   Vulkan one. A negative width, which mirrors the draw across the viewport's origin and flips
//!   its facing (a probe), is the Vulkan viewport over the same pixels with the vertex stage
//!   negating the position's x (`EncoderState::compensation`).
//! - **Depth ranges past `[0, 1]`** (a probe, an M3 Max and an M4 alike, with and without API
//!   validation). The fragment reads the depth `n + (f - n) z` unclamped, a constant bias included.
//!   In clip mode a primitive clips at the near and far planes and each depth stored, interpolated
//!   or written, is clamped to `[0, 1]`; in clamp mode nothing clips and each depth is clamped to
//!   `[0, 1]`, then to the range. A device with `VK_EXT_depth_range_unrestricted`, `depthClamp`,
//!   `depthClipEnable` and `depthClampControl` (`native_depth_range`) takes the range itself: the
//!   draw's pipeline clamps the depth to `[0, 1]` or to the range's one clamp, and clips in clip
//!   mode only (`EncoderState::depth_range`). A device with `shaderClipDistance` and `depthClamp`
//!   takes it by the vertex stage's compensation (`EncoderState::compensation`): the position's
//!   depth rewritten into a Vulkan range inside `[0, 1]`, clip distances for Metal's planes; in
//!   clamp mode a range whose one clamp is one depth collapses the Vulkan range to it instead.
//!   Elsewhere `set_viewport` refuses the range (`depth_ranges_refused`): `NotImplemented` without
//!   the features, and `Unsupported` on MoltenVK, whose rasterizer (Metal's on Apple's GPUs) draws
//!   the compensated depths about 2e-4 off where Metal's own viewport draws them exactly (a probe,
//!   `RuntimeDriverDefect::NormalizedDepthPastUnitRange`). A negative width is mirrored there
//!   still. Metal clamps each vertex's depth before the interpolation (Metal probes on the M3 Max
//!   and an M4): through a range with a negative depth in clip mode, or past `[0, 1]` in clamp
//!   mode, a draw of lines or triangles takes that clamp in the vertex stage's compensation,
//!   natively too (`VertexClamp`), announcing the primitives that cross Metal's planes or guard
//!   band and the draws that cannot take it.
//! - **The implicit scissor.** Metal clips a draw to the pixels its viewport overlaps: a point
//!   wider than its viewport is cut there, fractional edges included, and a viewport of no width
//!   or height draws nothing (measured at an integer origin). Each Vulkan scissor is therefore
//!   Metal's rectangle clipped to those pixels, which a triangle's own coverage never leaves; a
//!   viewport of no width or height is a one-pixel Vulkan viewport behind an empty scissor.
//! - **Scissor rectangles** (`set_scissor_rect`, `set_scissor_rects`). Metal's default is the
//!   render area. A count outside 1 to 16 stops; an empty rectangle inside the area draws nothing.
//!   A rectangle past the render area (`x + width` past its width, `y + height` past its height)
//!   stops Metal's validation, and plain Metal clips it to the area (Metal probe cases
//!   `scissor_past_width` and `scissor_past_height`: only the columns or rows inside are drawn;
//!   `scissor_origin_outside`: nothing), as the encoder does. At a draw, a scissor count other than
//!   1 and the viewport count stops; plain Metal draws a pipeline that selects no viewport through
//!   the first rectangle (Metal probe case `scissors_two`), and each viewport of one that selects
//!   through the rectangle of its index, one rectangle serving every viewport, and a viewport past
//!   two or more rectangles through none (probes `vs_*`, an M4 with macOS 27.0.1: a rectangle past
//!   the viewports unused, the third of three viewports under two rectangles and the fourth of four
//!   under three drew nothing), as the encoder draws them.
//! - **Viewport arrays.** A draw whose vertex function selects a viewport
//!   (`GraphicsState::writes_viewport_index`) uses as many as the encoder holds, with one scissor
//!   each (a single scissor repeated). More than one needs `multiViewport`: on a device without
//!   it the vertex function stores no index (a draw's one viewport is viewport 0 whatever the
//!   index, as Metal's past the count), and such a draw while the encoder holds more than one
//!   viewport is refused, `Unsupported(LayeredRendering)` needing the feature. Any other draw uses
//!   the first viewport and scissor only, as Metal's does. Measured
//!   (a Metal probe on an M4 with macOS 27.0.1): each primitive draws through the viewport of its
//!   index and that viewport's scissor rectangle, or the one rectangle set, up to sixteen
//!   viewports. An index at or past the encoder's viewport count draws through the first viewport
//!   and its scissor rectangle, and the fragment function reads the index as written (probes 7b and
//!   7c, an M4 and an M3 Max with macOS 27.0.1, with and without API validation), where Vulkan
//!   leaves the viewport undefined: the draw gives the vertex function the count
//!   (`BindState::viewport_count`), and it selects viewport 0 there.
//! - **Depth/stencil state** (`set_depth_stencil_state`): Metal's default state is the default
//!   descriptor's, and nil sets it again. In a pass without a depth attachment, a draw under a
//!   state that writes depth, or compares other than `Always` and `Never`, stops, and so does one
//!   under a state that uses the stencil in a pass without a stencil attachment
//!   (`depth_stencil.rs`). Plain Metal tests such a draw against a hidden depth of `float`, 1.0 at
//!   the pass's start, and a hidden stencil of 0, kept per sample through the pass: the encoder
//!   gives the pass a hidden depth/stencil attachment from the draw on
//!   (`EncoderState::needs_hidden`, `render.rs`). Under `Never` without writes Metal draws nothing
//!   and runs the fragment function, where Vulkan would draw everything (`depth_stencil.rs`'
//!   `FailingDraw`).
//! - **Stencil references** saturate at 255: Metal stores 255 for a reference of 0x1a5 replaced
//!   into an 8-bit stencil, where Vulkan's masks would keep 0xa5.
//! - **Culling and winding** follow Metal's names: the default winding is clockwise, Vulkan's
//!   `CLOCKWISE` under the flipped viewport, and a culled face is the Vulkan face of its name.
//! - **Depth bias** biases triangles only (`metal::state::depth_bias`); Metal's constant and
//!   slope factors land on Vulkan's to the ulp (`Depth32Float` and `Depth16Unorm`). A clamp other
//!   than 0 needs `depthBiasClamp` (`Unsupported` at the draw). A NaN or infinite factor, which
//!   Vulkan leaves undefined, takes Metal's measured answer through finite state
//!   ([`depth_bias_effect`]): a depth saturated at the top or bottom of Metal's range is every
//!   viewport's depth range collapsed there, and an infinite bias a clamp bounds is every
//!   viewport's range moved by the clamp, where that range stays in `[0, 1]`. What depends on each
//!   triangle's slope, and a moved range past `[0, 1]` or in depth clamp mode, is
//!   `NotImplemented(NonFiniteDepthBias)`, but where the compensation or a native depth range takes
//!   it in clip mode through a range of no negative depth, and where the compensation adds it after
//!   the vertex stage's clamp of each vertex's depth or, in depth clamp mode, folds it through a
//!   range inside `[0, 1]`.
//! - **Fill and clip modes.** `Lines` needs `fillModeNonSolid` and `Clamp` needs `depthClamp`
//!   (`Unsupported` at the setter); both are pipeline state.
//! - **Blend color**: Metal's default is zero; Vulkan clamps the constants as Metal does for a
//!   normalized attachment.

use super::degradation::Degradation;
use super::depth_stencil::{DepthStencilState, DepthUse};
use super::error::{internal, invalid, Error, Problem, RuntimeGap};
use super::pipeline::{DepthRange, FixedState};
use super::render::LineWidth;
use super::render_values::{ScissorRect, Viewport};
use super::Device;
use ash::vk;
use metal2vulkan::bind::{CompensatedViewport, RegionLimits};
use metal2vulkan::device::{DeviceProfile, VkExtension, VkFeature};
use metal2vulkan::error::{Capability, Feature, RuntimeDriverDefect, Unsupported};
use metal2vulkan::metal::{
    CullMode, DepthClipMode, PixelFormat, StencilOperation, TriangleFillMode, Winding,
};
use metal2vulkan::pipeline::{GraphicsState, ViewportCompensation};

/// Metal's most viewports and scissor rectangles.
pub(super) const MAX_VIEWPORTS: usize = 16;

/// A draw's viewports or scissor rectangles, which every draw makes, or an encoder's, which every
/// render pass starts with: the one most use held in place, so such a draw or pass allocates none
/// for them, or a list of the others.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum ViewportList<T> {
    One(T),
    Many(Vec<T>),
}

impl<T> FromIterator<T> for ViewportList<T> {
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> ViewportList<T> {
        let mut items = items.into_iter();
        let Some(first) = items.next() else {
            return ViewportList::Many(Vec::new());
        };
        let Some(second) = items.next() else {
            return ViewportList::One(first);
        };
        let mut many = Vec::with_capacity(2 + items.size_hint().0);
        many.push(first);
        many.push(second);
        many.extend(items);
        ViewportList::Many(many)
    }
}

impl<T> std::ops::Deref for ViewportList<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            ViewportList::One(item) => std::slice::from_ref(item),
            ViewportList::Many(items) => items,
        }
    }
}

impl<T> std::ops::DerefMut for ViewportList<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            ViewportList::One(item) => std::slice::from_mut(item),
            ViewportList::Many(items) => items,
        }
    }
}

/// What a viewport, scissor rectangle or depth/stencil setter did wrong.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderStateProblem {
    /// A count of viewports outside 1 to 16, or of no scissor rectangle (more than 16 scissor
    /// rectangles are taken, as plain Metal takes them).
    ///
    /// Basis: unknown (plain Metal's answer for a viewport count outside 1 to 16, and for no
    /// scissor rectangle, is unmeasured; the more than 16 scissor rectangles it takes are not
    /// refused).
    Count {
        /// The count given.
        count: usize,
    },
    /// A viewport field that is NaN or infinite, but for a NaN origin x, which draws nothing, as
    /// plain Metal's.
    ///
    /// Basis: unknown (plain Metal's answer for a viewport field that is NaN or infinite is
    /// measured only for a NaN origin x, which draws nothing and is not refused).
    ViewportNotFinite,
}

/// The render encoder's state, in Metal's terms.
#[derive(Clone)]
pub(super) struct EncoderState {
    /// The render area's width and height.
    extent: [u32; 2],
    viewports: ViewportList<Viewport>,
    scissors: ViewportList<ScissorRect>,
    cull_mode: CullMode,
    winding: Winding,
    /// `None` is Metal's default state, the default descriptor's.
    depth_stencil: Option<DepthStencilState>,
    /// The front and back stencil references, saturated at 255.
    stencil_reference: [u32; 2],
    /// What Metal's depth bias, slope scale and clamp do to a triangle's depth.
    depth_bias: DepthBiasEffect,
    blend_color: [f32; 4],
    /// `setAlphaTestReferenceValue:` (SPI); `None` until set, which Metal's test compares as 1.
    alpha_test_reference: Option<f32>,
    fill_mode: TriangleFillMode,
    clip_mode: DepthClipMode,
    /// Apple's `setProvokingVertexMode:` at 1: a primitive's flat inputs read its last vertex
    /// (`render/provoking.rs`). Metal's default, and every new encoder's, is the first.
    provoking_last: bool,
    /// The mode is past 1, which Metal draws a third way, the runtime as mode 1 but for patches
    /// (`render/provoking.rs`).
    provoking_third: bool,
    /// Apple's `setLineWidth:`, as Metal quantizes it (`render/line_width.rs`). Metal's default,
    /// and every new encoder's, is 1.
    line_width: LineWidth,
}

impl EncoderState {
    /// Metal's defaults over a render area of `extent`.
    pub(super) fn metal_defaults(extent: [u32; 2]) -> EncoderState {
        let [width, height] = extent;
        EncoderState {
            extent,
            viewports: ViewportList::One(Viewport::new(
                0.0,
                0.0,
                f64::from(width),
                f64::from(height),
                0.0,
                1.0,
            )),
            scissors: ViewportList::One(ScissorRect::new(
                0,
                0,
                u64::from(width),
                u64::from(height),
            )),
            cull_mode: CullMode::None,
            winding: Winding::Clockwise,
            depth_stencil: None,
            stencil_reference: [0, 0],
            depth_bias: DepthBiasEffect::Factors {
                constant: 0.0,
                slope: 0.0,
                clamp: 0.0,
            },
            blend_color: [0.0; 4],
            alpha_test_reference: None,
            fill_mode: TriangleFillMode::Fill,
            clip_mode: DepthClipMode::Clip,
            provoking_last: false,
            provoking_third: false,
            line_width: LineWidth::ONE,
        }
    }

    /// `setViewports:count:` (one viewport for `setViewport:`), checked as the module says: a
    /// depth range past `[0, 1]` refused as `refused` says (`depth_ranges_refused`), `None` on a
    /// device that takes it natively or by the vertex stage's compensation.
    pub(super) fn set_viewports(
        &mut self,
        call: &'static str,
        viewports: &[Viewport],
        refused: Option<Error>,
    ) -> Result<(), Error> {
        check_count(call, viewports.len())?;
        // A NaN origin x draws nothing, as plain Metal's: a viewport of no width.
        let viewports: ViewportList<Viewport> = viewports
            .iter()
            .map(|v| match v.origin_x.is_nan() {
                true => Viewport {
                    origin_x: 0.0,
                    width: 0.0,
                    ..*v
                },
                false => *v,
            })
            .collect();
        for v in viewports.iter() {
            let fields = [v.origin_x, v.origin_y, v.width, v.height, v.znear, v.zfar];
            if fields.iter().any(|f| !f.is_finite()) {
                return Err(invalid(
                    call,
                    Problem::RenderState(RenderStateProblem::ViewportNotFinite),
                ));
            }
        }
        if let Some(error) = refused.filter(|_| viewports.iter().any(outside_depth)) {
            return Err(error);
        }
        self.viewports = viewports;
        Ok(())
    }

    /// `setScissorRects:count:` (one rectangle for `setScissorRect:`), checked as the module
    /// says.
    pub(super) fn set_scissors(
        &mut self,
        call: &'static str,
        scissors: &[ScissorRect],
    ) -> Result<(), Error> {
        // More than 16 rectangles stop Metal's validation, and plain Metal draws (Metal probe case
        // `scissors_17` on an M4): a draw goes through the rectangles its viewports name
        // (`EncoderState::vk_scissors`). None, unmeasured, is refused.
        if scissors.is_empty() {
            return Err(invalid(
                call,
                Problem::RenderState(RenderStateProblem::Count { count: 0 }),
            ));
        }
        // A rectangle past the render area is clipped to it, as plain Metal clips it.
        let [width, height] = self.extent.map(u64::from);
        let clip = |origin: u64, size: u64, end: u64| {
            let start = origin.min(end);
            (start, origin.saturating_add(size).min(end) - start)
        };
        self.scissors = scissors
            .iter()
            .map(|s| {
                let (x, width) = clip(s.x, s.width, width);
                let (y, height) = clip(s.y, s.height, height);
                ScissorRect::new(x, y, width, height)
            })
            .collect();
        Ok(())
    }

    pub(super) fn set_cull_mode(&mut self, mode: CullMode) {
        self.cull_mode = mode;
    }

    /// `setProvokingVertexMode:`: the last vertex where `last`, else the first
    /// (`render/provoking.rs`).
    pub(super) fn set_provoking_last(&mut self, last: bool) {
        self.provoking_last = last;
    }

    /// Whether a primitive's flat inputs read its last vertex (`setProvokingVertexMode:` 1).
    pub(super) fn provoking_last(&self) -> bool {
        self.provoking_last
    }

    /// `setProvokingVertexMode:`: a mode past 1 where `third` (`render/provoking.rs`).
    pub(super) fn set_provoking_third(&mut self, third: bool) {
        self.provoking_third = third;
    }

    /// Whether the mode is past 1, which Metal draws a third way.
    pub(super) fn provoking_third(&self) -> bool {
        self.provoking_third
    }

    /// `setLineWidth:`, as Metal quantizes it (`render/line_width.rs`).
    pub(super) fn set_line_width(&mut self, width: LineWidth) {
        self.line_width = width;
    }

    /// The width the encoder's lines are drawn at.
    pub(super) fn line_width(&self) -> LineWidth {
        self.line_width
    }

    /// `setTriangleFillMode:`'s mode.
    pub(super) fn fill_mode(&self) -> TriangleFillMode {
        self.fill_mode
    }

    pub(super) fn set_winding(&mut self, winding: Winding) {
        self.winding = winding;
    }

    pub(super) fn set_depth_stencil(&mut self, state: Option<&DepthStencilState>) {
        self.depth_stencil = state.cloned();
    }

    /// The bound depth/stencil state uses the stencil on either face
    /// (`DepthStencilState::uses_stencil`; Metal's default state does not).
    pub(super) fn uses_stencil(&self) -> bool {
        self.depth_stencil
            .as_ref()
            .is_some_and(DepthStencilState::uses_stencil)
    }

    /// The bound depth/stencil state writes the depth (Metal's default state does not).
    pub(super) fn depth_writes(&self) -> bool {
        self.depth_stencil
            .as_ref()
            .is_some_and(|s| s.descriptor().depth_write_enabled)
    }

    /// The front and back references, saturated at 255 (the module docs).
    pub(super) fn set_stencil_references(&mut self, front: u32, back: u32) {
        self.stencil_reference = [front.min(0xff), back.min(0xff)];
    }

    /// `setDepthBias:slopeScale:clamp:`, by Metal's rule ([`depth_bias_effect`]): factors whose
    /// effect depends on each triangle's slope are `NotImplemented(NonFiniteDepthBias)`.
    pub(super) fn set_depth_bias(
        &mut self,
        depth_bias: f32,
        slope_scale: f32,
        clamp: f32,
    ) -> Result<(), Error> {
        self.depth_bias = depth_bias_effect(depth_bias, slope_scale, clamp)
            .ok_or(Error::NotImplemented(RuntimeGap::NonFiniteDepthBias))?;
        Ok(())
    }

    pub(super) fn set_blend_color(&mut self, color: [f32; 4]) {
        self.blend_color = color;
    }

    /// `setAlphaTestReferenceValue:` (SPI), which a pipeline with the private alpha test compares
    /// its fragments' alpha against (`BindState::alpha_test_reference`).
    pub(super) fn set_alpha_test_reference(&mut self, value: f32) {
        self.alpha_test_reference = Some(value);
    }

    /// `setTriangleFillMode:`: `Lines` needs `fillModeNonSolid`.
    pub(super) fn set_fill_mode(
        &mut self,
        mode: TriangleFillMode,
        profile: &DeviceProfile,
    ) -> Result<(), Error> {
        metal2vulkan::metal::state::polygon_mode(mode, profile).map_err(device_error)?;
        self.fill_mode = mode;
        Ok(())
    }

    /// `setDepthClipMode:`: `Clamp` needs `depthClamp`.
    pub(super) fn set_clip_mode(
        &mut self,
        mode: DepthClipMode,
        profile: &DeviceProfile,
    ) -> Result<(), Error> {
        metal2vulkan::metal::state::depth_clamp_enable(mode, profile).map_err(device_error)?;
        self.clip_mode = mode;
        Ok(())
    }

    /// Whether a draw under the state, in a pass with (or without) a depth and a stencil
    /// attachment, needs the pass's hidden depth and stencil (the module docs): where Metal's
    /// validation stops it, a state that writes depth or compares other than `Always` and
    /// `Never` without a depth, or uses the stencil without a stencil, plain Metal tests it
    /// against them. A scissor count other than 1 and the viewport count, which Metal's
    /// validation stops too, draws as plain Metal draws it ([`EncoderState::vk_scissors`]).
    pub(super) fn needs_hidden(&self, depth: bool, stencil: bool) -> bool {
        self.depth_stencil.as_ref().is_some_and(|state| {
            (!depth && state.depth_without_attachment() == DepthUse::Stops)
                || (!stencil && state.uses_stencil())
        })
    }

    /// The viewports a draw of `topology` uses ([`EncoderState::used_viewports`]), their depth
    /// ranges as the depth bias leaves them ([`depth_bias_effect`]): collapsed at the top or
    /// bottom of Metal's range for a saturated bias (in depth clamp mode the range's larger
    /// depth, or its smaller one but at least 0, `saturated`), moved by an offset. Refused
    /// (`NotImplemented(NonFiniteDepthBias)`) for an offset that moves a range past `[0, 1]`,
    /// where Metal clamps each depth to `[0, 1]`, or in depth clamp mode, where it clamps each
    /// to the range after the move (which the compensation folds, `EncoderState::compensation`), or
    /// through a range with a negative depth, where it clamps each vertex's depth at 0 before the
    /// move. On a device that takes a range past `[0, 1]` natively (`native`), a clip-mode offset
    /// through a range of no negative depth moves it past `[0, 1]` too, as the native clamp to `[0,
    /// 1]` after the move is Metal's (`depth_range`).
    pub(super) fn draw_viewports(
        &self,
        selects: bool,
        topology: metal2vulkan::vk::PrimitiveTopology,
        profile: &DeviceProfile,
        native: bool,
    ) -> Result<ViewportList<Viewport>, Error> {
        let used = self.used_viewports(selects);
        let clamps = self.clip_mode == DepthClipMode::Clamp;
        // In clamp mode a range past `[0, 1]` whose one clamp holds one depth stores that depth
        // alone, which its Vulkan range collapses to where the device takes no range natively
        // (`EncoderState::compensation` decides the draw may).
        let unbiased = |v: &Viewport| match clamp_range(v) {
            [low, high] if clamps && !native && low == high && outside_depth(v) => {
                [f64::from(low); 2]
            }
            _ => [v.znear, v.zfar],
        };
        let Some(bias) = self.draw_bias(topology, profile)? else {
            return Ok(used
                .iter()
                .map(|v| {
                    let [znear, zfar] = unbiased(v);
                    Viewport { znear, zfar, ..*v }
                })
                .collect());
        };
        used.iter()
            .map(|v| {
                let depth = match bias {
                    DepthBiasEffect::Factors { .. } => unbiased(v),
                    DepthBiasEffect::Saturated { high } if clamps => [saturated(v, high); 2],
                    DepthBiasEffect::Saturated { high } => [if high { 1.0 } else { 0.0 }; 2],
                    DepthBiasEffect::Offset(offset) => {
                        let moved = [v.znear, v.zfar].map(|z| z + f64::from(offset));
                        let past = moved.iter().any(|z| !(0.0..=1.0).contains(z));
                        let negative = v.znear.min(v.zfar) < 0.0;
                        if clamps || negative || (past && !native) {
                            return Err(Error::NotImplemented(RuntimeGap::NonFiniteDepthBias));
                        }
                        moved
                    }
                };
                let [znear, zfar] = depth;
                Ok(Viewport { znear, zfar, ..*v })
            })
            .collect()
    }

    /// The depth clip and clamp of a draw through the viewports set (all of them where the
    /// pipeline `selects` one, else the first) where one has a depth range past `[0, 1]`, which
    /// only a device that takes it natively holds (`native_depth_range`); `None` for ranges
    /// inside `[0, 1]`, Vulkan's own. The clamp follows the ranges set, not the ones a saturated
    /// depth bias moves (`EncoderState::draw_viewports`), but for the saturated depth below.
    /// Metal's answers (a probe, an M3 Max and an M4 alike, plain and validated): the fragment
    /// reads the depth `n + (f - n) z` unclamped; in clip mode a primitive clips at the near and
    /// far planes and each depth, interpolated or written, is clamped to `[0, 1]`; in clamp mode
    /// nothing clips and each depth is clamped to `[0, 1]`, then to the range (a range of 1.5 to 3
    /// stored 1.5), which is the one clamp of `clamp_range`. Refused
    /// (`NotImplemented(ViewportDepthRange)`): clamp mode through viewports of different such
    /// clamps, as Vulkan clamps every viewport's depth to one range.
    ///
    /// A saturated depth `bias` ([`EncoderState::draw_bias`]) in clamp mode clamps to the depth
    /// Metal stores (a Metal probe on an M4 with macOS 27.0.1): the range's larger depth, past 1
    /// too (1.5 of a range of -0.5 to 1.5), or its smaller one but at least 0, which the draw's
    /// viewports collapse to (`draw_viewports`).
    pub(super) fn depth_range(
        &self,
        selects: bool,
        bias: Option<DepthBiasEffect>,
    ) -> Result<Option<DepthRange>, Error> {
        let viewports = self.used_viewports(selects);
        // Or a range an offset moves past `[0, 1]` (`draw_viewports`).
        let moved_out = |v: &Viewport| match bias {
            Some(DepthBiasEffect::Offset(offset)) => {
                let offset = f64::from(offset);
                outside_depth(&Viewport {
                    znear: v.znear + offset,
                    zfar: v.zfar + offset,
                    ..*v
                })
            }
            _ => false,
        };
        if !viewports.iter().any(|v| outside_depth(v) || moved_out(v)) {
            // In clip mode Metal clamps a depth a bias takes past `[0, 1]`, which a device that
            // takes depth ranges natively stores unclamped: the Vulkan clamp `[0, 1]`, clipped.
            let clip = self.clip_mode == DepthClipMode::Clip;
            return Ok((clip && moves_depth(bias)).then_some(DepthRange {
                clip: true,
                clamp: [0.0f32, 1.0].map(f32::to_bits),
            }));
        }
        let clamps = self.clip_mode == DepthClipMode::Clamp;
        let clamp_of = |v: &Viewport| match bias {
            Some(DepthBiasEffect::Saturated { high }) => [saturated(v, high) as f32; 2],
            _ => clamp_range(v),
        };
        let clamp = if clamps {
            let mut ranges = viewports.iter().map(clamp_of);
            let first = ranges.next().unwrap_or([0.0, 1.0]);
            if ranges.any(|range| range != first) {
                return Err(Error::NotImplemented(RuntimeGap::ViewportDepthRange));
            }
            first
        } else {
            [0.0, 1.0]
        };
        Ok(Some(DepthRange {
            clip: !clamps,
            clamp: clamp.map(f32::to_bits),
        }))
    }

    /// How a draw of `topology` through the viewports it uses compensates them in the vertex
    /// stage, where a Vulkan viewport cannot be Metal's: a negative width, mirrored, and, on a
    /// device that does not take a depth range past `[0, 1]` natively (`native`), such a range.
    /// `None` where every viewport is Vulkan's (the common case: no variant, no push field).
    ///
    /// The position becomes `(sx x, y, a w + b z, w)`. A mirror is `sx` -1 with the Vulkan viewport
    /// at `x + width` of width `-width` (`vk_viewports`), which flips the facing as Metal's does. A
    /// depth range in clip mode is a Vulkan range of `[0, 1]` with `a` = `n`, `b` = `f - n`, so the
    /// window depth is Metal's `n + (f - n) z / w`, clip distances `z` and `w - z` clipping at
    /// Metal's planes, and `depthClamp` (`FixedState::depth_clamp`) both replacing Vulkan's own
    /// clipping and clamping each depth to `[0, 1]`. In clamp mode the Vulkan range is the one
    /// clamp `[L, H]` (`clamp_range`), `a` = `(n - L) / (H - L)` and `b` = `(f - n) / (H - L)`, and
    /// nothing clips. A one clamp of one depth `D` (every one outside `[0, 1]` is: 1.5 of a range
    /// of 1.5 to 3, 0 of -1 to 0 or of -2 to -1) stores `D` alone, whatever the window depth (Metal
    /// clamps each vertex's depth to `[0, 1]` and each sample to the range; Metal probes on the M3
    /// Max and an M4), so its Vulkan range collapses to `[D, D]` with no compensation of the depth
    /// (`draw_viewports`): where `one_depth` says the fragment function reads no depth of its
    /// position, which would read `D` where Metal's reads `n + (f - n) z`, and the device takes a
    /// Vulkan range of `D` (`VK_EXT_depth_range_unrestricted` past `[0, 1]`); and, natively too,
    /// where the draw's bias moves no sample off `D` (`OneDepth::bias_holds`). Refused
    /// (`NotImplemented(ViewportDepthRange)`): such a one clamp elsewhere, a saturated depth off it
    /// `NotImplemented(NonFiniteDepthBias)`; and used viewports whose depths need different
    /// compensations, as the vertex stage holds one. Used viewports that differ in their mirror
    /// alone take one compensation of `sx` 1 under `ViewportCompensation::mirrors`, bit `i` of
    /// `CompensatedViewport::mirrors` set where viewport `i` is mirrored, which the vertex stage
    /// negates `x` by for the viewport it selects.
    ///
    /// The draw's depth `bias` ([`EncoderState::draw_bias`]), where it saturates or offsets, is
    /// folded into `a` and `b`, as Metal biases (Metal probes on an M4 with macOS 27.0.1 and the M3
    /// Max): each sample is Metal's depth of its vertices clamped (at 0 in clip mode, to `[0, 1]`
    /// in clamp mode), plus the bias, clamped to `[0, 1]` in clip mode and in clamp mode to Metal's
    /// clamp of each sample, `[max(min(n, f), 0), max(n, f)]` (`sample_clamp`). In clip mode a
    /// saturated depth is 1 or 0 (`a` 1 or 0, `b` 0), and an offset `o` adds to the depth before
    /// the clamp to `[0, 1]` (`a` = `n + o`) through a range of no negative depth, where Metal's
    /// unbiased depth is the compensation's, and through one with a negative depth after the vertex
    /// stage's clamp at 0. In clamp mode a saturated depth is the range's larger depth, or its
    /// smaller one but at least 0 (`a` = `(d - L) / (H - L)`, `b` 0), where the Vulkan range `[L,
    /// H]` holds it: the one clamp, or Metal's clamp of each sample under
    /// `VK_EXT_depth_range_unrestricted` (`one_depth`) on a device that does not take the ranges
    /// natively; and an offset adds to the depth through every range (`a` = `(n + o - L) / (H -
    /// L)`): inside `[0, 1]` through the one clamp, past it after the vertex stage's clamp to `[0,
    /// 1]` through Metal's clamp of each sample, where the device holds it (its top at most 1, or
    /// under the extension). Refused (`NotImplemented(NonFiniteDepthBias)`): an offset in clip mode
    /// through a range with a negative depth whose vertices the stage does not clamp; one in clamp
    /// mode through a range past `[0, 1]` whose vertices the stage does not clamp or whose clamp of
    /// each sample the device does not hold; and a saturated depth past the Vulkan range.
    ///
    /// Where `vertex_clamp` takes it ([`VertexClamp`]), a viewport through which Metal clamps each
    /// vertex's depth before the interpolation ([`clamps_vertices`]), under no depth bias, one of
    /// factors or an offset, takes the depth's compensation under
    /// `ViewportCompensation::vertex_clamp`, natively too: the vertex stage clamps the window depth
    /// at 0 in clip mode (`z / w` to `[o, +inf]` of an offset `o`, else 0), and to `[0, 1]` in
    /// clamp mode (`z / w` to `[(o - L) / (H - L), (1 + o - L) / (H - L)]` of the Vulkan range `[L,
    /// H]`), so that the offset adds after the clamp. Bias factors through such a viewport in clamp
    /// mode take Metal's clamp of each sample as the Vulkan range where the device holds it, as an
    /// offset does. A device that takes the ranges natively takes the compensation where it does
    /// for every used viewport; the caller decides the rest (`VertexClamp::kept`).
    pub(super) fn compensation(
        &self,
        selects: bool,
        native: bool,
        bias: Option<DepthBiasEffect>,
        one_depth: OneDepth,
        vertex_clamp: VertexClamp,
    ) -> Result<Option<Compensation>, Error> {
        let clamps = self.clip_mode == DepthClipMode::Clamp;
        let refused = || Err(Error::NotImplemented(RuntimeGap::NonFiniteDepthBias));
        // Whether the bias moves the samples, and an offset folded into the compensation.
        let offsets = offsets(bias);
        let offset = match bias {
            Some(DepthBiasEffect::Offset(offset)) => f64::from(offset),
            _ => 0.0,
        };
        let one = |v: &Viewport| -> Result<Option<Compensation>, Error> {
            let sx = if v.width < 0.0 { -1.0 } else { 1.0 };
            let mirror = (sx < 0.0).then(|| Compensation {
                viewport: CompensatedViewport::new(
                    ViewportCompensation::new(false),
                    [-1.0, 0.0, 1.0, 0.0],
                ),
                depth: None,
                guard: None,
            });
            // Metal clamps each vertex's depth before the interpolation: the vertex stage does
            // too, natively as well, where the draw lets it and no saturated bias was folded in
            // (an offset adds after the clamp).
            let saturates = matches!(bias, Some(DepthBiasEffect::Saturated { .. }));
            let clamped = vertex_clamp.takes && !saturates && clamps_vertices(v, clamps, offsets);
            // In clamp mode Metal clamps each biased sample to the range itself, past 1 too:
            // under the vertex clamp the Vulkan range is that clamp where the device holds it.
            let sample = sample_clamp(v);
            let widened =
                clamps && offsets && clamped && (one_depth.unrestricted || sample[1] <= 1.0);
            // A saturated depth past the one clamp is that clamp's top, where the device holds
            // a Vulkan range past 1 (natively the range's own clamp holds it).
            let saturated_past = match bias {
                Some(DepthBiasEffect::Saturated { high: up }) if clamps && !native => {
                    let [low, high] = clamp_range(v).map(f64::from);
                    !(low..=high).contains(&saturated(v, up)) && one_depth.unrestricted
                }
                _ => false,
            };
            // A one clamp of one depth, which every one outside `[0, 1]` is: the Vulkan range
            // collapses to it (`draw_viewports`) where nothing reads or moves another.
            let one_clamp = clamp_range(v);
            if clamps
                && outside_depth(v)
                && one_clamp[0] == one_clamp[1]
                && !widened
                && !saturated_past
            {
                return match native {
                    true => OneDepth::bias_holds(v, bias).map(|()| mirror),
                    false => one_depth.collapses(v, one_clamp[0], bias).map(|()| mirror),
                };
            }
            // An offset in clamp mode folds into a compensation through every range, which
            // keeps the clamp to the range that moving the range would move.
            let folds = clamps && matches!(bias, Some(DepthBiasEffect::Offset(_)));
            if !clamped && !folds && (native || !outside_depth(v)) {
                return Ok(mirror);
            }
            let (n, f) = (v.znear, v.zfar);
            let (values, depth) = if clamps {
                // The Vulkan range: Metal's clamp of each biased sample where it widens the one
                // clamp or holds a saturated depth past it, else the one clamp.
                let [low, high] = match widened || saturated_past {
                    true => sample,
                    false => one_clamp.map(f64::from),
                };
                let span = high - low;
                let values = match bias {
                    Some(DepthBiasEffect::Saturated { high: up }) => {
                        let depth = saturated(v, up);
                        if !(low..=high).contains(&depth) {
                            return refused();
                        }
                        [(depth - low) / span, 0.0, 0.0]
                    }
                    // An offset adds to the depth of the vertices clamped to `[0, 1]`: inside
                    // `[0, 1]` no vertex needs the clamp, outside it the vertex stage's, through
                    // Metal's clamp of each sample.
                    Some(DepthBiasEffect::Offset(_)) if outside_depth(v) && !widened => {
                        return refused();
                    }
                    // One depth (`n` is `f` inside `[0, 1]`): that depth alone.
                    Some(DepthBiasEffect::Offset(_)) if span == 0.0 => [0.0, 0.0, 0.0],
                    Some(DepthBiasEffect::Offset(_)) => {
                        [(n + offset - low) / span, (f - n) / span, 0.0]
                    }
                    Some(DepthBiasEffect::Factors { .. }) | None => {
                        [(n - low) / span, (f - n) / span, 0.0]
                    }
                };
                (values, [low, high])
            } else {
                let values = match bias {
                    Some(DepthBiasEffect::Saturated { high: up }) => {
                        [if up { 1.0 } else { 0.0 }, 0.0, 1.0]
                    }
                    // Through a range with a negative depth, after the vertex stage's clamp at 0.
                    Some(DepthBiasEffect::Offset(_)) if n.min(f) >= 0.0 || clamped => {
                        [n + offset, f - n, 1.0]
                    }
                    Some(DepthBiasEffect::Offset(_)) => return refused(),
                    Some(DepthBiasEffect::Factors { .. }) | None => [n, f - n, 1.0],
                };
                (values, [0.0, 1.0])
            };
            let [a, b, clip] = values;
            let viewport = CompensatedViewport::new(
                ViewportCompensation::new(true).with_vertex_clamp(clamped),
                [sx, a, b, clip].map(|x| x as f32),
            );
            // The window depth `L + (H - L) z / w` clamped to `[0, 1]` in clamp mode, at 0 in
            // clip mode, whose Vulkan range is `[0, 1]`; an offset's after the clamp.
            let viewport = match (clamped, clamps) {
                (false, _) => viewport,
                (true, true) => {
                    let [low, high] = depth;
                    let span = high - low;
                    viewport.with_vertex_clamp(
                        [offset - low, 1.0 + offset - low].map(|t| (t / span) as f32),
                    )
                }
                (true, false) => viewport.with_vertex_clamp([offset as f32, f32::INFINITY]),
            };
            Ok(Some(Compensation {
                viewport,
                depth: Some(depth.map(|z| z as f32)),
                guard: None,
            }))
        };
        // Each viewport's, in order (the first refusal is the answer), compared as they come.
        let used = self.used_viewports(selects);
        let mut each = used.iter().map(&one);
        let first = each.next().transpose()?.flatten();
        let mut same = true;
        for compensation in each {
            same &= compensation? == first;
        }
        if same {
            return Ok(first);
        }
        // Viewports that differ in their mirror alone: one compensation of `sx` 1, the vertex
        // stage negating `x` by the selected viewport's bit. Different depths stay refused.
        let unmirrored = |c: Option<Compensation>| {
            let c = c.unwrap_or_else(Compensation::identity);
            let [_, a, b, clip] = c.viewport.values;
            let viewport = CompensatedViewport::new(c.viewport.compensation, [1.0, a, b, clip])
                .with_vertex_clamp(c.viewport.vertex_clamp);
            Compensation { viewport, ..c }
        };
        let base = unmirrored(first);
        for viewport in used {
            if unmirrored(one(viewport)?) != base {
                return Err(Error::NotImplemented(RuntimeGap::ViewportDepthRange));
            }
        }
        Ok(Some(base.mirrored_by_index(used)))
    }

    /// The depth bias a draw of `topology` takes ([`depth_bias_effect`]): `None` for a primitive
    /// Metal does not bias (points and lines).
    pub(super) fn draw_bias(
        &self,
        topology: metal2vulkan::vk::PrimitiveTopology,
        profile: &DeviceProfile,
    ) -> Result<Option<DepthBiasEffect>, Error> {
        let biased = metal2vulkan::metal::state::depth_bias(topology, 0.0, 0.0, 0.0, profile)
            .map_err(device_error)?
            .enable;
        Ok(biased.then_some(self.depth_bias))
    }

    /// The first scissor rectangle, which a draw of a pipeline that selects no viewport uses.
    pub(super) fn first_scissor(&self) -> &ScissorRect {
        &self.scissors[0]
    }

    /// The powers of two `[sx, sy]` a draw of points widens the Vulkan viewports it uses by,
    /// about their centers ([`EncoderState::vk_viewports`]), its vertex stage scaling the
    /// position by their inverses and culling at Metal's guard band ([`guard_points`]): on each
    /// axis the least power of two from 2 whose widened viewports hold Metal's 1.5 and every
    /// pixel each viewport overlaps with one more on either side, so that no device clips a
    /// point's pixels inside them and the implicit scissor ([`EncoderState::vk_scissors`]) cuts
    /// them where Metal does. `None` where a widened viewport would leave the device's
    /// `maxViewportDimensions` or `viewportBoundsRange`, or its origin would not be a `float`
    /// exactly: only an exact one keeps the viewport's center, so every pixel center, where the
    /// device computes it.
    pub(super) fn point_guard(&self, selects: bool, limits: &RegionLimits) -> Option<[f64; 2]> {
        let vk = Self::vk_viewports(self.used_viewports(selects), None, None);
        let scale = |extent: f32| {
            let need = 1.0 + 2.0 / f64::from(extent.abs());
            let mut s = 2.0f64;
            while s < need && s.is_finite() {
                s *= 2.0;
            }
            s
        };
        let sx = vk.iter().map(|v| scale(v.width)).fold(2.0, f64::max);
        let sy = vk.iter().map(|v| scale(v.height)).fold(2.0, f64::max);
        let fits = vk.iter().all(|v| {
            let [x, y, width, height] = widened(v, [sx, sy]);
            let exact = |c: f64| c.is_finite() && f64::from(c as f32) == c;
            let [low, high] = limits.viewport_bounds_range.map(f64::from);
            let within = |origin: f64, extent: f64, max: u32| {
                let (lo, hi) = (origin.min(origin + extent), origin.max(origin + extent));
                extent.abs() <= f64::from(max) && lo >= low && hi <= high
            };
            let [max_width, max_height] = limits.max_viewport_dimensions;
            exact(x) && exact(y) && within(x, width, max_width) && within(y, height, max_height)
        });
        fits.then_some([sx, sy])
    }

    /// The viewports a draw uses: every one for a pipeline that selects a viewport per
    /// primitive, else the first.
    pub(super) fn used_viewports(&self, selects: bool) -> &[Viewport] {
        if selects {
            &self.viewports
        } else {
            &self.viewports[..1]
        }
    }

    /// A draw's `viewports` ([`EncoderState::draw_viewports`]) in Vulkan's terms (the module
    /// docs). A viewport of no width or height, which Vulkan's cannot be, takes one pixel: its
    /// implicit scissor is empty ([`EncoderState::vk_scissors`]), so nothing rasterizes through
    /// it. A negative width, which the vertex stage mirrors (`EncoderState::compensation`), is
    /// the Vulkan viewport over the same pixels, from `x + width`; a compensated depth takes the
    /// compensation's Vulkan range (`depth`). A draw of points under Metal's guard band widens
    /// each by its `guard` about its center ([`EncoderState::point_guard`]).
    pub(super) fn vk_viewports(
        viewports: &[Viewport],
        depth: Option<[f32; 2]>,
        guard: Option<[f64; 2]>,
    ) -> ViewportList<vk::Viewport> {
        viewports
            .iter()
            .map(|v| {
                let (x, width) = if v.width < 0.0 {
                    (v.origin_x + v.width, -v.width)
                } else if v.width == 0.0 {
                    (v.origin_x, 1.0)
                } else {
                    (v.origin_x, v.width)
                };
                let height = if v.height == 0.0 { 1.0 } else { v.height };
                let [min_depth, max_depth] = depth.unwrap_or([v.znear as f32, v.zfar as f32]);
                let viewport = vk::Viewport {
                    x: x as f32,
                    y: (v.origin_y + height) as f32,
                    width: width as f32,
                    height: (-height) as f32,
                    min_depth,
                    max_depth,
                };
                match guard {
                    None => viewport,
                    Some(scale) => {
                        let [x, y, width, height] = widened(&viewport, scale).map(|c| c as f32);
                        vk::Viewport {
                            x,
                            y,
                            width,
                            height,
                            ..viewport
                        }
                    }
                }
            })
            .collect()
    }

    /// The draw's scissors in Vulkan's terms: one per viewport it uses, the Metal rectangle of
    /// its index, the one rectangle where the encoder holds one, and an empty one past two or
    /// more (the module docs), each clipped to the pixels its viewport overlaps, Metal's implicit
    /// scissor ([`viewport_pixels`]).
    pub(super) fn vk_scissors(&self, selects: bool) -> ViewportList<vk::Rect2D> {
        self.each_vk_scissor(selects).collect()
    }

    /// [`vk_scissors`](Self::vk_scissors) into `rects`, of [`MAX_VIEWPORTS`] (the most a draw
    /// uses), without allocating.
    pub(super) fn vk_scissors_in<'a>(
        &self,
        selects: bool,
        rects: &'a mut [vk::Rect2D; MAX_VIEWPORTS],
    ) -> &'a [vk::Rect2D] {
        let count = rects
            .iter_mut()
            .zip(self.each_vk_scissor(selects))
            .map(|(rect, scissor)| *rect = scissor)
            .count();
        &rects[..count]
    }

    /// Each of [`vk_scissors`](Self::vk_scissors), in order.
    fn each_vk_scissor(&self, selects: bool) -> impl Iterator<Item = vk::Rect2D> + '_ {
        let extent = self.extent.map(i64::from);
        let none = ScissorRect::new(0, 0, 0, 0);
        self.used_viewports(selects)
            .iter()
            .enumerate()
            .map(move |(i, viewport)| {
                let s = match self.scissors.len() {
                    1 => &self.scissors[0],
                    _ => self.scissors.get(i).unwrap_or(&none),
                };
                // Inside the render area (checked at the setter).
                let scissor = [
                    [s.x as i64, (s.x + s.width) as i64],
                    [s.y as i64, (s.y + s.height) as i64],
                ];
                let pixels = viewport_pixels(viewport, extent);
                let [x, y] = [0, 1].map(|axis| {
                    let lo = scissor[axis][0].max(pixels[axis][0]);
                    let hi = scissor[axis][1].min(pixels[axis][1]);
                    [lo, hi.max(lo)]
                });
                vk::Rect2D {
                    // Each bound lies in the render area: it fits an `i32` and a `u32`.
                    offset: vk::Offset2D {
                        x: x[0] as i32,
                        y: y[0] as i32,
                    },
                    extent: vk::Extent2D {
                        width: (x[1] - x[0]) as u32,
                        height: (y[1] - y[0]) as u32,
                    },
                }
            })
    }

    /// The draw's viewport count. More than one needs `multiViewport`, which
    /// [`EncoderState::fixed`] checks.
    pub(super) fn viewport_count(&self, selects: bool) -> u32 {
        self.used_viewports(selects).len() as u32
    }

    /// The encoder's depth/stencil state's descriptor: Metal's default one (no depth or stencil
    /// test) where it set none. A shader-side stencil's draws take their stencil state from it
    /// (`BindState::depth_stencil`).
    pub(super) fn depth_stencil_descriptor(
        &self,
    ) -> metal2vulkan::metal::state::DepthStencilDescriptor {
        self.depth_stencil
            .as_ref()
            .map(|state| *state.descriptor())
            .unwrap_or_default()
    }

    /// The encoder's depth/stencil state where Metal's depth test fails every fragment of a
    /// draw, a compare of `Never` in a pass without a depth attachment (`depth` false); `None`
    /// for every other draw.
    pub(super) fn failing_depth(&self, depth: bool) -> Option<&DepthStencilState> {
        self.depth_stencil
            .as_ref()
            .filter(|state| !depth && state.depth_without_attachment() == DepthUse::FailsAll)
    }

    /// The Vulkan depth/stencil state of the encoder's depth/stencil state.
    pub(super) fn vk_depth_stencil(&self) -> metal2vulkan::metal::state::DepthStencilState {
        match &self.depth_stencil {
            Some(state) => state.vulkan(),
            None => metal2vulkan::metal::state::depth_stencil_state(
                &metal2vulkan::metal::state::DepthStencilDescriptor::default(),
            ),
        }
    }

    pub(super) fn stencil_reference(&self) -> [u32; 2] {
        self.stencil_reference
    }

    pub(super) fn blend_color(&self) -> [f32; 4] {
        self.blend_color
    }

    /// The alpha test reference; `None` where the encoder never set it.
    pub(super) fn alpha_test_reference(&self) -> Option<f32> {
        self.alpha_test_reference
    }

    /// Whether the depth clip mode is `Clamp`: whether a fragment function that runs the draw's
    /// depth test itself clamps its depth to the range (`BindState::depth_clip_mode_clamp`;
    /// otherwise to `[0, 1]`).
    pub(super) fn depth_clip_mode_clamp(&self) -> bool {
        self.clip_mode == DepthClipMode::Clamp
    }

    /// The depth range a fragment function that runs a draw's depth test itself computes each
    /// sample's depth by and, in depth clamp mode, clamps it to (`BindState::depth_range`): the
    /// draw's own Vulkan range, its viewports' as the depth bias moves or collapses them and as a
    /// clamp of one depth collapses them ([`EncoderState::draw_viewports`]), so the function tests
    /// and stores the depth the device's own draw would. Refused
    /// (`NotImplemented(ViewportDepthRange)`) where a compensated depth replaces the range
    /// (`compensated`) or the draw's viewports have different ranges, the function reading one,
    /// and in depth clamp mode where the draw's range is not the encoder's, which Metal clamps
    /// to and the function would not (one range pushed).
    ///
    /// A function that returns its depth (`returned`) tests that, clamped in clamp mode to the
    /// viewport's own range as Metal clamps it, and reads no window depth: a compensation of ranges
    /// inside `[0, 1]` (the written depth's clamp, `written_depth_clamp`, on a device that takes no
    /// range natively) leaves that range its own (Refused there before, ANV measured).
    pub(super) fn shader_tests_range(
        &self,
        selects: bool,
        topology: metal2vulkan::vk::PrimitiveTopology,
        profile: &DeviceProfile,
        (native, compensated): (bool, bool),
        returned: bool,
    ) -> Result<[f32; 2], Error> {
        let refused = || Err(Error::NotImplemented(RuntimeGap::ViewportDepthRange));
        let inside = !self.used_viewports(selects).iter().any(outside_depth);
        if compensated && !(returned && inside) {
            return refused();
        }
        let viewports = self.draw_viewports(selects, topology, profile, native)?;
        let Some((first, rest)) = viewports.split_first() else {
            return Err(internal("a draw through no viewport"));
        };
        if rest
            .iter()
            .any(|v| (v.znear, v.zfar) != (first.znear, first.zfar))
        {
            return refused();
        }
        let moved = viewports
            .iter()
            .zip(self.used_viewports(selects))
            .any(|(drawn, set)| (drawn.znear, drawn.zfar) != (set.znear, set.zfar));
        if self.clip_mode == DepthClipMode::Clamp && moved {
            return refused();
        }
        Ok([first.znear as f32, first.zfar as f32])
    }

    /// The draw's depth bias for `topology` (`metal::state::depth_bias`): a clamp needs
    /// `depthBiasClamp`, and the constant counts the device's unit. A saturated bias or an offset
    /// biases nothing here: the draw's viewports take it ([`EncoderState::draw_viewports`]).
    pub(super) fn vk_depth_bias(
        &self,
        topology: metal2vulkan::vk::PrimitiveTopology,
        profile: &DeviceProfile,
    ) -> Result<metal2vulkan::metal::state::DepthBias, Error> {
        let [bias, slope, clamp] = match self.depth_bias {
            DepthBiasEffect::Factors {
                constant,
                slope,
                clamp,
            } => [constant, slope, clamp],
            DepthBiasEffect::Saturated { .. } | DepthBiasEffect::Offset(_) => [0.0; 3],
        };
        metal2vulkan::metal::state::depth_bias(topology, bias, slope, clamp, profile)
            .map_err(device_error)
    }

    /// `Unsupported(LayeredRendering)` needing `multiViewport` on a device without it, for a
    /// draw that would select among more than one viewport.
    pub(super) fn needs_viewports(profile: &DeviceProfile) -> Result<(), Error> {
        match profile
            .enabled()
            .features
            .contains(&VkFeature::MultiViewport)
        {
            true => Ok(()),
            false => Err(Error::Unsupported(Unsupported::needs(
                Feature::LayeredRendering,
                Capability::Feature(VkFeature::MultiViewport),
            ))),
        }
    }

    /// The Vulkan front face of the encoder's front-facing winding.
    pub(super) fn vk_front_face(&self) -> metal2vulkan::vk::FrontFace {
        match self.winding {
            Winding::Clockwise => metal2vulkan::vk::FrontFace::CLOCKWISE,
            Winding::CounterClockwise => metal2vulkan::vk::FrontFace::COUNTER_CLOCKWISE,
        }
    }

    /// The state a pipeline takes statically, for `viewports` viewports.
    pub(super) fn fixed(
        &self,
        viewports: u32,
        profile: &DeviceProfile,
    ) -> Result<FixedState, Error> {
        // A draw that selects among the encoder's viewports on a device without `multiViewport`:
        // its vertex function stores no `ViewportIndex` there (one viewport is all a draw has),
        // so the draw is exact only while the encoder holds one.
        if viewports > 1 {
            EncoderState::needs_viewports(profile)?;
        }
        Ok(FixedState {
            cull_mode: match self.cull_mode {
                CullMode::None => metal2vulkan::vk::CullModeFlags::NONE,
                CullMode::Front => metal2vulkan::vk::CullModeFlags::FRONT,
                CullMode::Back => metal2vulkan::vk::CullModeFlags::BACK,
            },
            front_face: self.vk_front_face(),
            depth_stencil: self.vk_depth_stencil(),
            polygon_mode: metal2vulkan::metal::state::polygon_mode(self.fill_mode, profile)
                .map_err(device_error)?,
            depth_clamp: metal2vulkan::metal::state::depth_clamp_enable(self.clip_mode, profile)
                .map_err(device_error)?,
            viewports,
            depth_range: None,
            masks_color: false,
            provoking_last: false,
            line_width: 1.0_f32.to_bits(),
        })
    }
}

/// What a device takes of viewport depth ranges past `[0, 1]` ([`native_depth_range`],
/// [`unrestricted`]): facts of the profile its compile device was made with, which stays the
/// device's for its life, so read once at its making (`DeviceShared::depth_ranges`) rather than
/// looked up in the profile's sets at every draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DepthRanges {
    /// It enabled `VK_EXT_depth_range_unrestricted`, `depthClamp`, `VK_EXT_depth_clip_enable`'s
    /// `depthClipEnable` and `VK_EXT_depth_clamp_control`'s `depthClampControl`.
    pub(super) native: bool,
    /// It enabled `VK_EXT_depth_range_unrestricted`.
    pub(super) unrestricted: bool,
}

impl DepthRanges {
    /// The facts of a device of `profile`.
    pub(super) fn of(profile: &DeviceProfile) -> DepthRanges {
        let enabled = profile.enabled();
        let unrestricted = enabled
            .extensions
            .contains(&VkExtension::ExtDepthRangeUnrestricted);
        let native = unrestricted
            && [
                VkFeature::DepthClamp,
                VkFeature::DepthClipEnable,
                VkFeature::DepthClampControl,
            ]
            .iter()
            .all(|feature| enabled.features.contains(feature));
        DepthRanges {
            native,
            unrestricted,
        }
    }
}

/// Whether `device` takes a viewport depth range past `[0, 1]` natively: it enabled
/// `VK_EXT_depth_range_unrestricted`, `depthClamp`, `VK_EXT_depth_clip_enable`'s
/// `depthClipEnable` and `VK_EXT_depth_clamp_control`'s `depthClampControl`
/// (`EncoderState::depth_range`).
pub(super) fn native_depth_range(device: &Device) -> bool {
    device.shared().depth_ranges.native
}

/// Why `device` does not take a viewport depth range past `[0, 1]`, `None` where it does:
/// natively (`native_depth_range`), or by the vertex stage's compensation, which needs
/// `shaderClipDistance` for Metal's planes and `depthClamp` to replace Vulkan's own clipping
/// (`EncoderState::compensation`), on a driver that rasterizes the compensated depths exactly.
/// Refused: `NotImplemented(ViewportDepthRange)` without the features, and
/// `Unsupported(ViewportDepthRange)` with `RuntimeDriverDefect::NormalizedDepthPastUnitRange`
/// where the driver draws normalized depths past `[0, 1]` inexactly (MoltenVK,
/// `driver_defects.rs`).
pub(super) fn depth_ranges_refused(device: &Device) -> Option<Error> {
    if native_depth_range(device) {
        return None;
    }
    let features = &device.compile_device().profile().enabled().features;
    if !(features.contains(&VkFeature::ShaderClipDistance)
        && features.contains(&VkFeature::DepthClamp))
    {
        return Some(Error::NotImplemented(RuntimeGap::ViewportDepthRange));
    }
    let defect = RuntimeDriverDefect::NormalizedDepthPastUnitRange;
    device
        .properties()
        .driver_defects
        .contains(&defect)
        .then(|| {
            Error::Unsupported(Unsupported::runtime_driver_defect(
                Feature::ViewportDepthRange,
                defect,
            ))
        })
}

/// Whether `viewport`'s depth range reaches past `[0, 1]`.
fn outside_depth(viewport: &Viewport) -> bool {
    let outside = |z: f64| !(0.0..=1.0).contains(&z);
    outside(viewport.znear) || outside(viewport.zfar)
}

/// Whether Metal's clamp of each vertex's depth before the interpolation differs from a clamp of
/// each sample through `viewport`, in depth clamp mode where `clamps` (Metal probes on the M3 Max
/// and an M4 alike): in clip mode, which clamps each vertex's depth at 0 and each sample to `[0,
/// 1]`, through a range with a negative depth; in clamp mode, which clamps each vertex's depth to
/// `[0, 1]` and each sample to the range, through a range past `[0, 1]` whose one clamp holds two
/// depths (one of one depth stores it alone, `OneDepth`). Inside `[0, 1]` the pieces a primitive is
/// clipped into at the planes clamp nothing (but a piece past them, which stores the clamp either
/// way). Under a bias that `offsets` the samples, also a clamp mode range above 1 of two depths:
/// its one clamp is its smaller depth, but Metal adds the offset to each vertex's depth clamped to
/// `[0, 1]` and clamps the sum to the range (a Metal probe on the M3 Max: 1.75 of 1.5 to 3 under an
/// offset of 0.75).
fn clamps_vertices(viewport: &Viewport, clamps: bool, offsets: bool) -> bool {
    match clamps {
        false => viewport.znear.min(viewport.zfar) < 0.0,
        true => {
            outside_depth(viewport) && {
                let [low, high] = clamp_range(viewport);
                let [sample_low, sample_high] = sample_clamp(viewport);
                low < high || (offsets && sample_low < sample_high)
            }
        }
    }
}

/// Whether a draw's `bias` moves its samples off their unbiased depths before Metal's clamp of
/// each sample: factors of a constant or a slope, or an offset.
fn offsets(bias: Option<DepthBiasEffect>) -> bool {
    match bias {
        Some(DepthBiasEffect::Factors {
            constant, slope, ..
        }) => constant != 0.0 || slope != 0.0,
        Some(DepthBiasEffect::Offset(_)) => true,
        Some(DepthBiasEffect::Saturated { .. }) | None => false,
    }
}

/// Metal's clamp of each biased sample in depth clamp mode through `viewport`: `[max(min(n, f), 0),
/// max(n, f)]`, its top at least its bottom, past 1 too (Metal probes on an M4 and the M3 Max; a
/// range below 0 stores 0).
fn sample_clamp(viewport: &Viewport) -> [f64; 2] {
    let low = viewport.znear.min(viewport.zfar).max(0.0);
    [low, viewport.znear.max(viewport.zfar).max(low)]
}

/// Whether a draw's vertex stage clamps each vertex's depth as Metal does
/// ([`EncoderState::compensation`]). Metal clamps the depth of each vertex of a primitive, at 0 in
/// depth clip mode and to `[0, 1]` in depth clamp mode, before it interpolates (Metal probes on the
/// M3 Max and an M4 alike, plain and validated), which differs from a clamp of each sample through
/// the viewports [`clamps_vertices`] names, for a draw of lines or triangles whose fragment
/// function writes no depth, under no depth bias, one of factors or an offset (a saturated depth is
/// one depth). The vertex stage's compensation takes the clamp
/// (`ViewportCompensation::vertex_clamp`), on a device that takes the ranges natively too, but for
/// a fragment function that reads its position's depth (Metal's unclamped `n + (f - n) z`, which
/// the clamp would change), a mesh pipeline, which has no vertex function, on a device without
/// `shaderClipDistance`, or for a vertex function whose clip distances leave no room for the
/// compensation's two: those keep the clamp of each sample, announced once (`sample-depth-clamp`).
/// Where it takes the clamp, a primitive that crosses Metal's planes or its guard band stores the
/// interpolation of its own clamped vertices, where Metal clamps the vertices of the pieces it
/// clips the primitive into first (a Metal probe on the M3 Max and an M4): announced once
/// (`clipped-depth-pieces`, [`VertexClamp::taken`]). In depth clamp mode under a bias that moves
/// the depth, Metal adds the bias to each vertex's depth clamped to `[0, 1]` through a range inside
/// `[0, 1]` too (a Metal probe on the M3 Max: 1.75 of a vertex depth past 1 under an offset of
/// 0.75), where the device adds it before its clamp of each sample, which a primitive past Metal's
/// near or far plane shows: announced once there too (`sample-depth-clamp`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct VertexClamp {
    /// The vertex stage clamps each vertex's depth through the viewports that need it.
    pub(super) takes: bool,
}
impl VertexClamp {
    /// The clamp a draw of `topology` and `graphics` (a mesh pipeline's where `meshes`) under
    /// `bias` through the viewports it uses takes on `device`; announced where `announce` (none for
    /// an execution's precompile at commit, which draws nothing): `sample-depth-clamp` where Metal
    /// clamps each vertex's depth and the vertex stage cannot (a mesh pipeline only where the
    /// device takes the ranges natively: elsewhere its compensation is refused), and
    /// `depth-bias-past-one` for bias factors in depth clamp mode through a range past 1, whose
    /// depths past 1 the Vulkan clamp of at most 1 holds at 1 where Metal clamps them to the
    /// range's larger depth (Metal probes on an M4 and the M3 Max): where the vertex stage does not
    /// clamp or the device holds no Vulkan range past 1 (`VK_EXT_depth_range_unrestricted`; the
    /// compensation's range is then Metal's clamp of each sample).
    pub(super) fn of(
        device: &Device,
        state: &EncoderState,
        graphics: &GraphicsState,
        topology: metal2vulkan::vk::PrimitiveTopology,
        meshes: bool,
        bias: Option<DepthBiasEffect>,
        announce: bool,
    ) -> VertexClamp {
        let clamp = VertexClamp::decided(device, state, graphics, topology, meshes, bias, announce);
        let clamps = state.clip_mode == DepthClipMode::Clamp;
        let drawn = !graphics.writes_depth && !graphics.rasterizer_discard;
        let factors = matches!(bias, Some(DepthBiasEffect::Factors { .. })) && offsets(bias);
        let widens = clamp.takes && unrestricted(device);
        if announce && drawn && clamps && factors && !widens {
            let used = state.used_viewports(graphics.writes_viewport_index);
            let past = used.iter().find(|v| {
                let [low, high] = clamp_range(v);
                v.znear.max(v.zfar) > 1.0 && low < high
            });
            if let Some(v) = past {
                device.degrade(Degradation::DepthBiasPastOne, || {
                    (
                        format!(
                            "a draw of triangles in depth clamp mode through a viewport depth \
                             range of {} to {}, under a depth bias of factors",
                            v.znear, v.zfar
                        ),
                        "the Vulkan clamp of its depths holds at most 1".to_owned(),
                    )
                });
            }
        }
        clamp
    }

    /// [`VertexClamp::of`] but its `depth-bias-past-one`: the clamp, `sample-depth-clamp`
    /// announced where it is kept.
    fn decided(
        device: &Device,
        state: &EncoderState,
        graphics: &GraphicsState,
        topology: metal2vulkan::vk::PrimitiveTopology,
        meshes: bool,
        bias: Option<DepthBiasEffect>,
        announce: bool,
    ) -> VertexClamp {
        let none = VertexClamp::default();
        let clamps = state.clip_mode == DepthClipMode::Clamp;
        let used = state.used_viewports(graphics.writes_viewport_index);
        let drawn = !graphics.writes_depth && !graphics.rasterizer_discard;
        // In clamp mode Metal adds a bias to each vertex's depth clamped to `[0, 1]` through a
        // range inside it too, which the device adds before its clamp of each sample: a
        // primitive past Metal's near or far plane stores otherwise, announced once.
        let inside = used.iter().find(|v| !outside_depth(v) && v.znear != v.zfar);
        if let Some(v) = inside.filter(|_| announce && drawn && clamps && offsets(bias)) {
            device.degrade(Degradation::SampleDepthClamp, || {
                (
                    described(state, v),
                    "Metal adds the depth bias to each vertex's depth clamped to [0, 1], where                      the device adds it before its clamp of each sample, so a primitive past                      Metal's near or far plane stores otherwise"
                        .to_owned(),
                )
            });
        }
        if !used
            .iter()
            .any(|v| clamps_vertices(v, clamps, offsets(bias)))
        {
            return none;
        }
        let sloped = topology != metal2vulkan::vk::PrimitiveTopology::POINT_LIST
            && !matches!(bias, Some(DepthBiasEffect::Saturated { .. }));
        if !drawn || !sloped {
            return none;
        }
        let profile = device.compile_device().profile();
        let limits = profile.limits();
        let most = limits
            .max_clip_distances
            .min(limits.max_combined_clip_and_cull_distances);
        let clip_distance = profile
            .enabled()
            .features
            .contains(&VkFeature::ShaderClipDistance);
        let kept = if graphics.reads_position_depth {
            Some(
                "its fragment function reads its position's depth, which Metal gives unclamped \
                 and the vertex stage's clamp would change",
            )
        } else if meshes {
            // Without the native range the draw's compensation is refused for a mesh pipeline.
            native_depth_range(device)
                .then_some("a mesh pipeline has no vertex function to clamp the depth")
        } else if !clip_distance {
            Some("the device has no shaderClipDistance for Metal's planes")
        } else if graphics.clip_distances + 2 > most {
            Some(
                "the device's clip distances cannot hold the compensation's two beside the \
                 vertex function's own",
            )
        } else {
            return VertexClamp { takes: true };
        };
        if let Some(reason) = kept {
            VertexClamp::kept(
                device,
                state,
                graphics.writes_viewport_index,
                reason,
                announce,
            );
        }
        none
    }

    /// Announce once (`sample-depth-clamp`) that a draw through the viewports `state` uses
    /// where it `selects` one keeps a clamp of each sample where Metal clamps each vertex's
    /// depth, for `reason`, where `announce`.
    pub(super) fn kept(
        device: &Device,
        state: &EncoderState,
        selects: bool,
        reason: &str,
        announce: bool,
    ) {
        let clamps = state.clip_mode == DepthClipMode::Clamp;
        let used = state.used_viewports(selects);
        let viewport = used.iter().find(|v| clamps_vertices(v, clamps, false));
        if let Some(viewport) = viewport.filter(|_| announce) {
            device.degrade(Degradation::SampleDepthClamp, || {
                (described(state, viewport), reason.to_owned())
            });
        }
    }

    /// Announce once (`clipped-depth-pieces`) that the draw of `compensation`, which clamps each
    /// vertex's depth, through the viewports `state` uses where it `selects` one, stores a
    /// primitive that crosses Metal's planes or guard band otherwise than Metal, where
    /// `announce`.
    pub(super) fn taken(
        device: &Device,
        state: &EncoderState,
        selects: bool,
        compensation: Option<Compensation>,
        announce: bool,
    ) {
        let taken = compensation.is_some_and(|c| c.viewport.compensation.vertex_clamp);
        let clamps = state.clip_mode == DepthClipMode::Clamp;
        let used = state.used_viewports(selects);
        let viewport = used.iter().find(|v| clamps_vertices(v, clamps, true));
        if let Some(viewport) = viewport.filter(|_| announce && taken) {
            device.degrade(Degradation::ClippedDepthPieces, || {
                (
                    described(state, viewport),
                    "the vertex stage clamps each vertex's depth as Metal does, before the \
                     device clips the primitive"
                        .to_owned(),
                )
            });
        }
    }
}

/// A draw of lines or triangles through `viewport` in `state`'s clip mode, for a notice.
fn described(state: &EncoderState, viewport: &Viewport) -> String {
    let mode = match state.clip_mode == DepthClipMode::Clamp {
        true => "clamp",
        false => "clip",
    };
    format!(
        "a draw of lines or triangles in depth {mode} mode through a viewport depth range of {} \
         to {}",
        viewport.znear, viewport.zfar
    )
}

/// A draw's viewport compensation (`EncoderState::compensation`): what the vertex function's
/// variant applies, and the Vulkan depth range every viewport takes under a depth's
/// compensation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Compensation {
    pub(super) viewport: CompensatedViewport,
    pub(super) depth: Option<[f32; 2]>,
    /// The powers of two a draw of points widens its Vulkan viewports by
    /// ([`EncoderState::point_guard`]), whose inverses the vertex stage scales its position by.
    pub(super) guard: Option<[f64; 2]>,
}

impl Compensation {
    /// The compensation of nothing: the position as the function writes it.
    fn identity() -> Compensation {
        Compensation {
            viewport: CompensatedViewport::new(
                ViewportCompensation::new(false),
                [1.0, 0.0, 1.0, 0.0],
            ),
            depth: None,
            guard: None,
        }
    }

    /// This compensation, of `sx` 1, through `used` viewports some of which are mirrored: the
    /// variant that negates `x` where the viewport the function selects is
    /// (`ViewportCompensation::mirrors`), bit `i` of its mirrors set where viewport `i` has a
    /// negative width, as Metal mirrors each viewport by its own width (a Metal probe, an M3 Max
    /// and an M4 alike).
    fn mirrored_by_index(self, used: &[Viewport]) -> Compensation {
        let mirrors = used
            .iter()
            .enumerate()
            .filter(|(_, v)| v.width < 0.0)
            .fold(0u32, |bits, (i, _)| bits | 1 << i);
        let viewport = CompensatedViewport::new(
            self.viewport.compensation.with_mirrors(true),
            self.viewport.values,
        )
        .with_guard(self.viewport.guard)
        .with_mirrors(mirrors)
        .with_vertex_clamp(self.viewport.vertex_clamp);
        Compensation { viewport, ..self }
    }
}

/// Whether `device` enabled `VK_EXT_depth_range_unrestricted`: a Vulkan viewport depth range past
/// `[0, 1]`.
fn unrestricted(device: &Device) -> bool {
    device.shared().depth_ranges.unrestricted
}

/// What a draw's device and pipeline let a depth clamp mode viewport whose one clamp holds one
/// depth collapse its Vulkan range to that depth ([`EncoderState::compensation`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct OneDepth {
    /// The device enabled `VK_EXT_depth_range_unrestricted`: a Vulkan range past `[0, 1]`.
    pub(super) unrestricted: bool,
    /// The fragment function may read its position's depth
    /// (`GraphicsState::reads_position_depth`).
    pub(super) reads: bool,
}

impl OneDepth {
    /// A draw of `graphics` on `device`.
    pub(super) fn of(device: &Device, graphics: &GraphicsState) -> OneDepth {
        OneDepth {
            unrestricted: unrestricted(device),
            reads: graphics.reads_position_depth,
        }
    }

    /// Whether a draw's bias factors keep Metal's samples at `viewport`'s one clamp of one depth in
    /// depth clamp mode: Metal clamps each vertex's depth to `[0, 1]` and each sample, the bias's
    /// offset added, to the range itself but at least 0 (a Metal probe on the M3 Max: 1.75 of a
    /// range of 1.5 to 3 under an offset of 0.75), so only factors of no offset (constant and slope
    /// 0) keep it, or a range whose clamp to itself is one depth (its larger depth at most its
    /// smaller one but 0: 0 of -1 to 0, 2 of 2 to 2). Refused `NotImplemented(ViewportDepthRange)`
    /// elsewhere, natively too, whose Vulkan clamp is the one depth (`EncoderState::depth_range`).
    /// A saturated or offset bias is not factors: the caller decides those.
    fn bias_holds(viewport: &Viewport, bias: Option<DepthBiasEffect>) -> Result<(), Error> {
        let (lo, hi) = (
            viewport.znear.min(viewport.zfar),
            viewport.znear.max(viewport.zfar),
        );
        match bias {
            Some(DepthBiasEffect::Factors {
                constant, slope, ..
            }) if hi > lo.max(0.0) && (constant != 0.0 || slope != 0.0) => {
                Err(Error::NotImplemented(RuntimeGap::ViewportDepthRange))
            }
            _ => Ok(()),
        }
    }

    /// Whether `viewport`'s Vulkan range may collapse to `depth`, its one clamp of one depth in
    /// depth clamp mode, under the draw's `bias`: Metal stores `depth` alone (Metal probes, one on
    /// the M3 Max) where the bias's factors keep it (`bias_holds`) or a saturated bias is `depth`,
    /// which a fragment function that reads its position's depth would read in place of Metal's
    /// `n + (f - n) z`, and which the device holds past `[0, 1]` only under
    /// `VK_EXT_depth_range_unrestricted`. Refused `NotImplemented(ViewportDepthRange)`, a saturated
    /// depth off it or an offset `NotImplemented(NonFiniteDepthBias)`.
    fn collapses(
        self,
        viewport: &Viewport,
        depth: f32,
        bias: Option<DepthBiasEffect>,
    ) -> Result<(), Error> {
        let range = Err(Error::NotImplemented(RuntimeGap::ViewportDepthRange));
        if self.reads || (!(0.0..=1.0).contains(&depth) && !self.unrestricted) {
            return range;
        }
        OneDepth::bias_holds(viewport, bias)?;
        match bias {
            Some(DepthBiasEffect::Saturated { high })
                if saturated(viewport, high) as f32 != depth =>
            {
                Err(Error::NotImplemented(RuntimeGap::NonFiniteDepthBias))
            }
            Some(DepthBiasEffect::Offset(_)) => {
                Err(Error::NotImplemented(RuntimeGap::NonFiniteDepthBias))
            }
            _ => Ok(()),
        }
    }
}

/// `viewport`, a Vulkan viewport, widened about its center by `[sx, sy]`: its `[x, y, width,
/// height]` exactly, each origin moved by `(s - 1)` half-extents and each extent scaled by `s`.
fn widened(viewport: &vk::Viewport, [sx, sy]: [f64; 2]) -> [f64; 4] {
    let axis = |origin: f32, extent: f32, s: f64| {
        let (origin, extent) = (f64::from(origin), f64::from(extent));
        [origin - (s - 1.0) * extent / 2.0, s * extent]
    };
    let [x, width] = axis(viewport.x, viewport.width, sx);
    let [y, height] = axis(viewport.y, viewport.height, sy);
    [x, y, width, height]
}

/// A draw of points under Metal's guard band (probes, an M3 Max and an M4 alike, plain and
/// validated, 2,810 points each): Metal keeps a point while its `|x|` and `|y|` are at most `1.5 w`
/// and `0 <= z <= w`, and draws the pixels whose centers its square covers inside the pixels its
/// viewport overlaps ([`viewport_pixels`]), where Vulkan culls a point whose center leaves the clip
/// volume (`|x| > w`) and clips a wide one at the viewport's edges by the device's own rule.
/// `compensation` (or none) with the band: the vertex stage scales the position by `[1 / sx, 1 /
/// sy]` into Vulkan viewports widened by `[sx, sy]` ([`EncoderState::point_guard`]) and culls at
/// `1.5 w` by four clip distances. Where the band cannot be taken (in a rate-mapped pass, whose
/// regions the runtime does not widen; on a device without `shaderClipDistance`, or whose clip
/// distances cannot hold the band's four beside the vertex function's `clip_distances` and the
/// depth's two; or through viewports `EncoderState::point_guard` cannot widen) the draw keeps
/// Vulkan's clipping of points and `compensation`, announced once (`point-guard-band`) where
/// `announce` (An execution's precompile at commit, which draws nothing, announces nothing).
pub(super) fn guard_points(
    device: &Device,
    state: &EncoderState,
    selects: bool,
    compensation: Option<Compensation>,
    clip_distances: u32,
    rate_mapped: bool,
    announce: bool,
) -> Option<Compensation> {
    let profile = device.compile_device().profile();
    let limits = profile.limits();
    let depth = compensation.is_some_and(|c| c.depth.is_some());
    let needed = clip_distances + 4 + if depth { 2 } else { 0 };
    let refused = if rate_mapped {
        Some("the render pass is rate-mapped, and the runtime widens no region's viewport")
    } else if !profile
        .enabled()
        .features
        .contains(&VkFeature::ShaderClipDistance)
    {
        Some("the device lacks shaderClipDistance, which the band's four planes need")
    } else if needed
        > limits
            .max_clip_distances
            .min(limits.max_combined_clip_and_cull_distances)
    {
        Some("the device's clip distances cannot hold the band's four beside the draw's own")
    } else {
        None
    };
    let scale = match refused {
        None => state.point_guard(selects, &device.shared().region_limits),
        Some(_) => None,
    };
    let Some(scale) = scale else {
        let reason = refused.unwrap_or(
            "a Vulkan viewport widened to hold the band would leave the device's viewport limits \
             or not have a float origin exactly",
        );
        if announce {
            device.degrade(Degradation::PointGuardBand, || {
                ("a draw of points".to_owned(), reason.to_owned())
            });
        }
        return compensation;
    };
    let c = compensation.unwrap_or_else(Compensation::identity);
    let viewport = CompensatedViewport::new(
        c.viewport.compensation.with_point_guard(true),
        c.viewport.values,
    )
    .with_guard(scale.map(|s| (1.0 / s) as f32))
    .with_mirrors(c.viewport.mirrors);
    Some(Compensation {
        viewport,
        guard: Some(scale),
        ..c
    })
}

/// How a draw whose fragment function writes its depth stores it as Metal does in depth clamp
/// mode through a range above 0 ([`written_depth_clamp`]): the range the function clamps the
/// depth to itself (`PushField::DepthClampRange`), and the draw's Vulkan clamp that holds 0 and
/// the range, natively (`native`) or through the vertex stage's compensation into a Vulkan range
/// of `[0, 1]` (`compensation`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct WrittenDepthClamp {
    pub(super) range: [f32; 2],
    pub(super) native: Option<DepthRange>,
    pub(super) compensation: Option<Compensation>,
}

/// Whether a depth bias of `bias` can move a triangle's depth past `[0, 1]`: Vulkan's bias of a
/// constant or slope factor other than 0. A saturated bias stores 1 or 0, and an offset one the
/// draw's viewports take (`EncoderState::depth_range`).
pub(super) fn moves_depth(bias: Option<DepthBiasEffect>) -> bool {
    matches!(
        bias,
        Some(DepthBiasEffect::Factors { constant, slope, .. }) if constant != 0.0 || slope != 0.0
    )
}

/// Announce, once (`biased-depth-past-unit-range`), a draw of triangles in depth clip mode under
/// a depth bias that can move their depth past `[0, 1]` ([`moves_depth`]), on a device that
/// enabled `VK_EXT_depth_range_unrestricted` but does not take depth ranges natively
/// ([`native_depth_range`]): such a device stores the biased depth unclamped (RADV and NVIDIA),
/// where Metal clamps it to `[0, 1]`, and without `depthClampControl` and `depthClipEnable` no
/// Vulkan clamp of `[0, 1]` that keeps the clipping exists. A draw whose vertex stage compensates
/// its depth clamps it under its own Vulkan range (`compensated`).
pub(super) fn announce_unclamped_bias(
    device: &Device,
    state: &EncoderState,
    bias: Option<DepthBiasEffect>,
    compensated: bool,
) {
    let clip = state.clip_mode == DepthClipMode::Clip;
    if unrestricted(device)
        && clip
        && moves_depth(bias)
        && !compensated
        && !native_depth_range(device)
    {
        device.degrade(Degradation::BiasedDepthPastUnitRange, || {
            (
                "a draw of triangles in depth clip mode under a depth bias".to_owned(),
                "the device stores a depth the bias takes past [0, 1] unclamped, and has no \
                 `VK_EXT_depth_clamp_control` and `VK_EXT_depth_clip_enable` to clamp it to \
                 [0, 1] while it clips"
                    .to_owned(),
            )
        });
    }
}

/// Announce, once (`nan-depth`), a draw whose fragment function writes its depth under a
/// depth/stencil state whose stencil has a depth-failure operation that changes it (an operation
/// other than `Keep` under a write mask that leaves some of the 8 bits): Metal fails a positive
/// NaN depth, under `Always` too, and runs that operation (Metal probes, an M3 Max and an M4:
/// neither depth nor color written, no sample counted); the compile layer discards such a fragment
/// (SEMANTICS 1143), which runs no stencil operation. A near-equivalent (the orchestrator's ruling
/// of 2026-10-07): the exact path would test every depth-writing pipeline's stencil in the shader
/// (`shadow_stencil.rs`).
pub(super) fn announce_nan_depth_failure(device: &Device, state: &EncoderState) {
    let Some(ds) = state.depth_stencil.as_ref() else {
        return;
    };
    let descriptor = ds.descriptor();
    let fails = [
        &descriptor.front_face_stencil,
        &descriptor.back_face_stencil,
    ]
    .into_iter()
    .any(|face| {
        face.depth_failure_operation != StencilOperation::Keep && face.write_mask & 0xff != 0
    });
    if fails {
        device.degrade(Degradation::NanDepth, || {
            (
                "a draw whose fragment function writes its depth, under a stencil depth-failure \
                 operation"
                    .to_owned(),
                "a positive NaN depth fails Metal's depth test, which runs the stencil's \
                 depth-failure operation, and the fragment is discarded instead, which runs no \
                 stencil operation"
                    .to_owned(),
            )
        });
    }
}

/// A draw of `topology` whose fragment function writes its depth, in depth clamp mode through
/// viewports whose one clamp (`clamp_range`) lies above 0: Metal stores a negative NaN depth as
/// 0 there, past the clamp, and every other depth clamped to `[0, 1]` and then to the range
/// (a Metal probe, an M3 Max and an M4 alike: `0xffc00000` stored 0 through `[0.25, 0.5]`), where
/// every device measured stored the range's low end (RADV, ANV, NVIDIA, lavapipe and MoltenVK,
/// `runtime_lane::viewport_ranges`). The function's variant clamps the depth itself
/// (`LoweringInputs::depth_clamp_range`) under a Vulkan clamp that holds 0: the user-defined range
/// `[0, max(1, high)]` on a device that takes depth ranges natively (`native_depth_range`), else
/// the vertex stage's compensation into a Vulkan range of `[0, 1]` (`a` = `n`, `b` = `f - n`,
/// clipping nothing, as `EncoderState::compensation` does in clip mode), which reads Metal's depth
/// in the fragment. `None` for any other draw, and, announced once (`nan-depth`), where neither
/// holds: viewports of different clamps or depth ranges, a mesh pipeline without the native range,
/// a device whose compensation is not exact or missing (`depth_ranges_refused`), a depth bias that
/// saturates or offsets the range, or clip distances that cannot hold the compensation's two beside
/// the vertex function's `clip_distances`. Announced where `announce` (An execution's precompile at
/// commit, which draws nothing, announces nothing).
pub(super) fn written_depth_clamp(
    device: &Device,
    state: &EncoderState,
    selects: bool,
    topology: metal2vulkan::vk::PrimitiveTopology,
    clip_distances: u32,
    meshes: bool,
    announce: bool,
) -> Result<Option<WrittenDepthClamp>, Error> {
    if state.clip_mode != DepthClipMode::Clamp {
        return Ok(None);
    }
    let used = state.used_viewports(selects);
    let ranges: Vec<[f32; 2]> = used.iter().map(clamp_range).collect();
    if ranges.iter().all(|range| range[0] <= 0.0) {
        return Ok(None);
    }
    let kept = |reason: &str| {
        if announce {
            device.degrade(Degradation::NanDepth, || {
                (
                    "a draw whose fragment function writes its depth, in depth clamp mode \
                     through a range above 0"
                        .to_owned(),
                    reason.to_owned(),
                )
            });
        }
        Ok(None)
    };
    let range = ranges[0];
    if ranges.iter().any(|other| *other != range) {
        return kept("the draw's viewports clamp to different ranges, and the function reads one");
    }
    if native_depth_range(device) {
        let clamp = [0.0, range[1].max(1.0)];
        return Ok(Some(WrittenDepthClamp {
            range,
            native: Some(DepthRange {
                clip: false,
                clamp: clamp.map(f32::to_bits),
            }),
            compensation: None,
        }));
    }
    if meshes {
        return kept("a mesh pipeline has no vertex function to compensate the depth");
    }
    if depth_ranges_refused(device).is_some() {
        return kept(
            "the device has no `VK_EXT_depth_clamp_control`, and its vertex stage cannot \
             compensate the depth exactly",
        );
    }
    // A one clamp past 1 is one depth, which the draw's Vulkan range collapses to
    // (`EncoderState::compensation`): the compensation's range of `[0, 1]` cannot hold it.
    if range[1] > 1.0 {
        return kept(
            "the device has no `VK_EXT_depth_clamp_control`, and the one clamp lies past 1, \
             outside the compensation's Vulkan range of [0, 1]",
        );
    }
    let profile = device.compile_device().profile();
    let biased = metal2vulkan::metal::state::depth_bias(topology, 0.0, 0.0, 0.0, profile)
        .map_err(device_error)?
        .enable;
    if biased && !matches!(state.depth_bias, DepthBiasEffect::Factors { .. }) {
        return kept(
            "a depth bias that saturates or offsets the range, which the compensation's Vulkan \
             range replaces",
        );
    }
    let limits = profile.limits();
    let most = limits
        .max_clip_distances
        .min(limits.max_combined_clip_and_cull_distances);
    if clip_distances + 2 > most {
        return kept(
            "the device's clip distances cannot hold the compensation's two beside the vertex \
             function's own",
        );
    }
    // One depth's compensation; mirrors alike take its `sx`, mixed ones the selected viewport's
    // bit.
    let values = |v: &Viewport, sx: f64| [sx, v.znear, v.zfar - v.znear, 0.0].map(|x| x as f32);
    let first = values(&used[0], 1.0);
    if used.iter().any(|v| values(v, 1.0) != first) {
        return kept(
            "the draw's viewports have different depth ranges, and the vertex stage holds one \
             compensation",
        );
    }
    let of = |values| Compensation {
        viewport: CompensatedViewport::new(ViewportCompensation::new(true), values),
        depth: Some([0.0, 1.0]),
        guard: None,
    };
    let mirrored = used.iter().filter(|v| v.width < 0.0).count();
    let compensation = match mirrored {
        0 => of(first),
        n if n == used.len() => of(values(&used[0], -1.0)),
        _ => of(first).mirrored_by_index(used),
    };
    Ok(Some(WrittenDepthClamp {
        range,
        native: None,
        compensation: Some(compensation),
    }))
}

/// The depth Metal stores in depth clamp mode under `viewport`'s range for a bias saturated at the
/// top (`high`) or the bottom: the range's larger depth, or its smaller one but at least 0 (a Metal
/// probe on an M4: 1.5 and 0 of a range of -0.5 to 1.5, 1.5 and 0.5 of 0.5 to 1.5).
fn saturated(viewport: &Viewport, high: bool) -> f64 {
    match high {
        true => viewport.znear.max(viewport.zfar),
        false => viewport.znear.min(viewport.zfar).max(0.0),
    }
}

/// The one clamp of Metal's two in depth clamp mode under `viewport`'s range: to `[0, 1]`, then to
/// the range `[lo, hi]` (a probe). A range that meets `[0, 1]` clamps to their intersection; one
/// above 1 to its smaller depth; one below 0 to 0, as Metal stores no depth below 0 (a Metal probe
/// on the M3 Max stored 0 through -2 to -1, -1 to -2 and -1 to -1, biased or not, where 484 clamped
/// to the range's larger depth).
fn clamp_range(viewport: &Viewport) -> [f32; 2] {
    let lo = viewport.znear.min(viewport.zfar);
    let hi = viewport.znear.max(viewport.zfar);
    let range = if lo > 1.0 {
        [lo, lo]
    } else if hi < 0.0 {
        [0.0, 0.0]
    } else {
        [lo.max(0.0), hi.min(1.0)]
    };
    range.map(|z| z as f32)
}

/// Announce once what a draw of `depth_stencil` stores in a `Depth16Unorm` depth
/// (`depth_format`) under the Vulkan bias `bias`: the device biases and converts the depth, so a
/// constant bias, or a slope alone, which brings Metal's constant unit, stores Metal's codes only
/// where the profile takes `DriverDeviation::MetalDepth16BiasedStores` (`unorm-depth-bias`), and
/// any depth it tests or writes only where the profile takes `DriverDeviation::MetalDepth16Stores`
/// (`unorm-depth-rounding`). Metal biases by `c ± 1` units of 1/65535 as `float`, adds the bias to
/// the depth rounded toward zero, and stores the sum rounded to nearest, a tie up (probes); the
/// device biases by `c` of its own unit (`metal::state::depth_bias`).
pub(super) fn announce_unorm_depth(
    device: &Device,
    depth_format: Option<PixelFormat>,
    depth_stencil: &metal2vulkan::metal::state::DepthStencilState,
    bias: &metal2vulkan::metal::state::DepthBias,
) {
    if depth_format != Some(PixelFormat::Depth16Unorm) {
        return;
    }
    use metal2vulkan::error::DriverDeviation::{MetalDepth16BiasedStores, MetalDepth16Stores};
    let profile = device.compile_device().profile();
    let stores = profile.takes_driver_deviation(MetalDepth16Stores);
    if depth_stencil.depth_test_enable && !stores {
        device.degrade(Degradation::UnormDepthRounding, || {
            let object = "a draw testing or writing a Depth16Unorm depth".to_owned();
            let reason = "the device converts the depth to its 16-bit code its own way, where \
                          Metal rounds the depth scaled by 65535 to nearest, a tie up: a depth \
                          at or just above a code's midpoint can store and compare as the code \
                          below Metal's"
                .to_owned();
            (object, reason)
        });
    }
    if !bias.enable
        || (bias.constant_factor == 0.0 && bias.slope_factor == 0.0)
        || profile.takes_driver_deviation(MetalDepth16BiasedStores)
    {
        return;
    }
    device.degrade(Degradation::UnormDepthBias, || {
        let object = "a draw biasing a Depth16Unorm depth by a constant, or by a slope that \
                      brings Metal's constant unit"
            .to_owned();
        let reason = "the device biases by the constant c in its own unit, up to twice 2^-16 \
                      (none for a slope alone), and converts the sum in its own rounding, where \
                      Metal biases by c + 1 units of 1/65535 for c of 0 or more (a slope alone \
                      brings one) and c - 1 below, adding them rounding toward zero: a biased \
                      depth can store another code than Metal's"
            .to_owned();
        (object, reason)
    });
}

/// What a depth bias command sets: the factors' bits. Two records set the same state where
/// theirs are equal.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct DepthBiasSetting {
    factors: [u32; 3],
}

/// How a draw records its depth bias ([`metal2vulkan::metal::state::DepthBias`]):
/// `vkCmdSetDepthBias` of its factors.
#[derive(Clone, Copy, PartialEq)]
pub(super) struct DepthBiasRecord {
    /// Constant factor, clamp, slope factor.
    factors: [f32; 3],
}

impl DepthBiasRecord {
    /// How `bias` records.
    pub(super) fn of(bias: &metal2vulkan::metal::state::DepthBias) -> DepthBiasRecord {
        DepthBiasRecord {
            factors: [bias.constant_factor, bias.clamp, bias.slope_factor],
        }
    }

    /// What recording it sets: its factors, bit for bit.
    pub(super) fn setting(&self) -> DepthBiasSetting {
        DepthBiasSetting {
            factors: self.factors.map(f32::to_bits),
        }
    }

    /// Record it into `command_buffer`.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, of `raw`'s device, under a pipeline whose depth bias is
    /// dynamic state.
    pub(super) unsafe fn record(&self, raw: &ash::Device, command_buffer: vk::CommandBuffer) {
        let [constant, clamp, slope] = self.factors;
        // SAFETY: the caller's.
        unsafe { raw.cmd_set_depth_bias(command_buffer, constant, clamp, slope) }
    }
}

/// What Metal's depth bias does to a triangle's depth ([`depth_bias_effect`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum DepthBiasEffect {
    /// Vulkan's bias of these finite factors: Metal's, the clamp 0 where Metal's was NaN or
    /// infinite, which clamps nothing there.
    Factors {
        constant: f32,
        slope: f32,
        clamp: f32,
    },
    /// Every fragment's depth at the top (`high`) or the bottom of Metal's range: 1 or 0, or in
    /// depth clamp mode the viewport's larger or smaller depth.
    Saturated { high: bool },
    /// Every fragment's depth its own plus the offset: an infinite or NaN bias a clamp bounds.
    Offset(f32),
}

/// Metal's depth bias of `constant`, `slope` and `clamp` factors (`setDepthBias:slopeScale:clamp:`)
/// as measured (Metal probes on an Apple M3 Max on macOS 26.7 and an Apple M4 on 27.0.1, each case
/// plain and validated, alike):
/// - a NaN constant factor biases as +inf does, and a NaN or infinite clamp clamps nothing;
/// - a triangle of depth slope `m` is biased `constant * r + slope * m` in IEEE arithmetic, so an
///   infinite slope factor on a flat triangle (`m` 0), or opposite infinities, give NaN;
/// - a clamp above 0 bounds the bias as `fmin` and one below 0 as `fmax`, which take the clamp
///   where the bias is NaN: the depth is then the triangle's plus the clamp;
/// - an unbounded infinite bias saturates the depth at 1 or 0 (in depth clamp mode at the
///   viewport's larger or smaller depth, either order of its range), and an unbounded NaN bias at
///   1, which every depth test compares as 1.
///
/// `None` where the effect differs between flat and sloped triangles, which no static Vulkan
/// state tells apart.
pub(super) fn depth_bias_effect(constant: f32, slope: f32, clamp: f32) -> Option<DepthBiasEffect> {
    let clamp = if clamp.is_finite() { clamp } else { 0.0 };
    let constant = if constant.is_nan() {
        f32::INFINITY
    } else {
        constant
    };
    if constant.is_finite() && slope.is_finite() {
        return Some(DepthBiasEffect::Factors {
            constant,
            slope,
            clamp,
        });
    }
    // The bias of a flat triangle and of a sloped one (`m` 1 stands for any `m` above 0, as
    // only the bias's sign and infinity matter): each is +inf, -inf or NaN.
    let effect = |bias: f32| {
        if bias.is_nan() {
            if clamp == 0.0 {
                DepthBiasEffect::Saturated { high: true }
            } else {
                DepthBiasEffect::Offset(clamp)
            }
        } else if (clamp > 0.0 && bias > 0.0) || (clamp < 0.0 && bias < 0.0) {
            DepthBiasEffect::Offset(clamp)
        } else {
            DepthBiasEffect::Saturated { high: bias > 0.0 }
        }
    };
    let flat = effect(constant + slope * 0.0);
    let sloped = effect(constant + slope);
    (flat == sloped).then_some(flat)
}

/// The pixels of `viewport` inside a render area of `extent`, per axis `[first, end)`: every pixel
/// its rectangle overlaps, and none for a viewport of no width or height. Metal clips to them
/// (Metal probes on an M4): a 4-pixel point is cut to the pixels its viewport overlaps, fractional
/// edges included ((4.5, 0, 1, 8) keeps columns 4 and 5), and a viewport of no width or height (at
/// an integer origin) draws nothing. A triangle's own coverage never leaves them.
fn viewport_pixels(viewport: &Viewport, extent: [i64; 2]) -> [[i64; 2]; 2] {
    let empty = viewport.width == 0.0 || viewport.height == 0.0;
    let axis = |origin: f64, size: f64, limit: i64| {
        let (lo, hi) = if size < 0.0 {
            (origin + size, origin)
        } else {
            (origin, origin + size)
        };
        let first = lo.floor().clamp(0.0, limit as f64) as i64;
        let end = hi.ceil().clamp(0.0, limit as f64) as i64;
        if empty {
            [first, first]
        } else {
            [first, end.max(first)]
        }
    };
    [
        axis(viewport.origin_x, viewport.width, extent[0]),
        axis(viewport.origin_y, viewport.height, extent[1]),
    ]
}

/// A count of viewports outside 1 to 16 stops (measured).
fn check_count(call: &'static str, count: usize) -> Result<(), Error> {
    if !(1..=MAX_VIEWPORTS).contains(&count) {
        return Err(invalid(
            call,
            Problem::RenderState(RenderStateProblem::Count { count }),
        ));
    }
    Ok(())
}

/// A compile-layer refusal of a state the device lacks, as the runtime's error.
fn device_error(error: metal2vulkan::error::Error) -> Error {
    match error {
        metal2vulkan::error::Error::Unsupported(unsupported) => Error::Unsupported(unsupported),
        other => internal(&format!("a render state check refused {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A draw's one viewport or scissor rectangle is held in place, more in a list in order, and
    /// none is an empty list; a refusal among them is the list's.
    #[test]
    fn viewport_lists_hold_one_in_place() {
        let none: ViewportList<u32> = std::iter::empty().collect();
        assert!(none.is_empty());
        let one: ViewportList<u32> = [7].into_iter().collect();
        assert!(matches!(one, ViewportList::One(7)));
        let mut three: ViewportList<u32> = [1, 2, 3].into_iter().collect();
        three[2] = 4;
        assert_eq!(&*three, &[1, 2, 4]);
        let refused: Result<ViewportList<u32>, ()> = [Ok(1), Err(()), Ok(3)].into_iter().collect();
        assert!(refused.is_err());
    }

    /// Metal's defaults cover the render area, and a viewport is Vulkan's with its height
    /// negated; Metal's own negative height is a positive Vulkan one.
    #[test]
    fn viewports_flip_into_vulkan() {
        let mut state = EncoderState::metal_defaults([8, 4]);
        let fields = |v: &[vk::Viewport]| -> Vec<[f32; 6]> {
            v.iter()
                .map(|v| [v.x, v.y, v.width, v.height, v.min_depth, v.max_depth])
                .collect()
        };
        assert_eq!(
            fields(&EncoderState::vk_viewports(
                state.used_viewports(false),
                None,
                None
            )),
            vec![[0.0, 4.0, 8.0, -4.0, 0.0, 1.0]]
        );
        state.viewports = [
            Viewport::new(0.0, 4.0, 4.0, -4.0, 0.0, 1.0),
            Viewport::new(2.0, 0.0, 2.0, 2.0, 0.0, 1.0),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            fields(&EncoderState::vk_viewports(
                state.used_viewports(false),
                None,
                None
            )),
            vec![[0.0, 0.0, 4.0, 4.0, 0.0, 1.0]]
        );
        assert_eq!(state.used_viewports(true).len(), 2);
        assert_eq!(state.vk_scissors(true).len(), 2, "one scissor each");
    }

    /// The scissor rules (probe 4): inside the render area, an empty one included; past it,
    /// clipped to it, as plain Metal draws it; 1 to 16 of them.
    #[test]
    fn scissors_stay_inside_the_render_area() {
        let mut state = EncoderState::metal_defaults([4, 4]);
        let set = |state: &mut EncoderState, s: &[ScissorRect]| state.set_scissors("t", s);
        for (given, kept) in [
            (ScissorRect::new(1, 1, 2, 2), ScissorRect::new(1, 1, 2, 2)),
            (ScissorRect::new(0, 0, 0, 0), ScissorRect::new(0, 0, 0, 0)),
            (ScissorRect::new(2, 0, 4, 4), ScissorRect::new(2, 0, 2, 4)),
            (ScissorRect::new(0, 3, 4, 2), ScissorRect::new(0, 3, 4, 1)),
            (ScissorRect::new(5, 5, 0, 0), ScissorRect::new(4, 4, 0, 0)),
            (
                ScissorRect::new(u64::MAX, 0, 1, 1),
                ScissorRect::new(4, 0, 0, 1),
            ),
        ] {
            assert_eq!(set(&mut state, &[given]), Ok(()), "{given:?}");
            assert_eq!(*state.scissors, [kept], "{given:?}");
        }
        assert_eq!(
            set(&mut state, &[]),
            Err(invalid(
                "t",
                Problem::RenderState(RenderStateProblem::Count { count: 0 })
            ))
        );
        // More than 16, as plain Metal takes them (Metal probe case `scissors_17`).
        assert_eq!(set(&mut state, &[ScissorRect::new(0, 0, 1, 1); 17]), Ok(()));
    }

    /// A viewport's pixels are those it overlaps (probes 4, 4b and 4c: a 1-pixel viewport at x 4
    /// cuts a 4-pixel point to column 4, one at x 4.5 to columns 4 and 5; the fractional viewport
    /// (0.5, 0.25, 3.5, 2.75) draws columns 0 to 3 and rows 0 to 2), and none without width or
    /// height.
    #[test]
    fn viewports_clip_to_their_pixels() {
        let pixels = |x, y, w, h| viewport_pixels(&Viewport::new(x, y, w, h, 0.0, 1.0), [8, 8]);
        assert_eq!(pixels(4.0, 0.0, 1.0, 8.0), [[4, 5], [0, 8]]);
        assert_eq!(pixels(0.5, 0.25, 3.5, 2.75), [[0, 4], [0, 3]]);
        assert_eq!(pixels(0.0, 8.0, 8.0, -8.0), [[0, 8], [0, 8]], "a flip");
        assert_eq!(
            pixels(-2.0, -2.0, 4.0, 4.0),
            [[0, 2], [0, 2]],
            "past the area"
        );
        assert_eq!(pixels(2.0, 2.0, 8.0, 8.0), [[2, 8], [2, 8]]);
        assert_eq!(pixels(4.0, 0.0, 0.0, 8.0), [[4, 4], [0, 0]], "no width");
        assert_eq!(pixels(0.0, 4.0, 8.0, 0.0), [[0, 0], [4, 4]], "no height");
        assert_eq!(pixels(4.5, 0.0, 1.0, 8.0)[0], [4, 6], "probe 4c");
        assert_eq!(pixels(3.25, 2.75, 1.5, 2.5), [[3, 5], [2, 6]], "probe 4c");
        let mut state = EncoderState::metal_defaults([8, 8]);
        state.viewports = [Viewport::new(4.0, 0.0, 0.0, 8.0, 0.0, 1.0)]
            .into_iter()
            .collect();
        assert_eq!(state.vk_scissors(false)[0].extent.width, 0);
        assert_eq!(
            EncoderState::vk_viewports(state.used_viewports(false), None, None)[0].width,
            1.0
        );
        state.viewports = [Viewport::new(2.0, 2.0, 4.0, 4.0, 0.0, 1.0)]
            .into_iter()
            .collect();
        state.scissors = [ScissorRect::new(3, 0, 5, 8)].into_iter().collect();
        let clipped = state.vk_scissors(false)[0];
        assert_eq!(
            (
                clipped.offset.x,
                clipped.offset.y,
                clipped.extent.width,
                clipped.extent.height
            ),
            (3, 2, 3, 4)
        );
    }

    /// The vertex stage's compensation: none for Vulkan's viewports; a mirror's `sx` of -1 over the
    /// same pixels; clip mode's `[n, f - n]` into a Vulkan range of `[0, 1]`; clamp mode's one
    /// clamp as the Vulkan range; none for a range a device takes natively; and the refusals.
    #[test]
    fn viewports_vulkan_cannot_be_are_compensated() {
        let gap = |gap| Err(Error::NotImplemented(gap));
        let mut state = EncoderState::metal_defaults([4, 4]);
        assert_eq!(
            state.compensation(
                false,
                false,
                None,
                OneDepth::default(),
                VertexClamp::default()
            ),
            Ok(None)
        );
        state.viewports = [Viewport::new(4.0, 0.0, -4.0, 4.0, 0.0, 1.0)]
            .into_iter()
            .collect();
        let mirror = state
            .compensation(
                false,
                false,
                None,
                OneDepth::default(),
                VertexClamp::default(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(mirror.viewport.values, [-1.0, 0.0, 1.0, 0.0]);
        assert!(!mirror.viewport.compensation.depth);
        assert_eq!(mirror.depth, None);
        let vk = EncoderState::vk_viewports(state.used_viewports(false), mirror.depth, None);
        assert_eq!([vk[0].x, vk[0].width, vk[0].min_depth], [0.0, 4.0, 0.0]);
        state.viewports = [Viewport::new(0.0, 0.0, 4.0, 4.0, 0.5, 2.0)]
            .into_iter()
            .collect();
        let clip = state
            .compensation(
                false,
                false,
                None,
                OneDepth::default(),
                VertexClamp::default(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(clip.viewport.values, [1.0, 0.5, 1.5, 1.0]);
        assert!(clip.viewport.compensation.depth);
        assert_eq!(clip.depth, Some([0.0, 1.0]));
        assert_eq!(
            state.compensation(
                false,
                true,
                None,
                OneDepth::default(),
                VertexClamp::default()
            ),
            Ok(None),
            "native"
        );
        let vk = EncoderState::vk_viewports(state.used_viewports(false), clip.depth, None);
        assert_eq!([vk[0].min_depth, vk[0].max_depth], [0.0, 1.0]);
        state.clip_mode = DepthClipMode::Clamp;
        let clamp = state
            .compensation(
                false,
                false,
                None,
                OneDepth::default(),
                VertexClamp::default(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(clamp.viewport.values, [1.0, 0.0, 3.0, 0.0]);
        assert_eq!(clamp.depth, Some([0.5, 1.0]));
        state.viewports = [Viewport::new(4.0, 0.0, -4.0, 4.0, 2.0, -1.0)]
            .into_iter()
            .collect();
        let both = state
            .compensation(
                false,
                false,
                None,
                OneDepth::default(),
                VertexClamp::default(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(both.viewport.values, [-1.0, 2.0, -3.0, 0.0]);
        assert_eq!(both.depth, Some([0.0, 1.0]));
        // One depth past `[0, 1]` without `VK_EXT_depth_range_unrestricted` (one inside it, 0 of
        // a range below 0 included, collapses: `one_depth_clamps_collapse_the_vulkan_range`).
        for (znear, zfar) in [(1.5, 3.0), (2.0, 2.0)] {
            state.viewports = [Viewport::new(0.0, 0.0, 4.0, 4.0, znear, zfar)]
                .into_iter()
                .collect();
            assert_eq!(
                state.compensation(
                    false,
                    false,
                    None,
                    OneDepth::default(),
                    VertexClamp::default()
                ),
                gap(RuntimeGap::ViewportDepthRange),
                "{znear} {zfar}"
            );
        }
        state.clip_mode = DepthClipMode::Clip;
        let whole = Viewport::new(0.0, 0.0, 4.0, 4.0, 0.0, 1.0);
        state.viewports = [Viewport::new(0.0, 0.0, 4.0, 4.0, 0.5, 2.0), whole]
            .into_iter()
            .collect();
        assert!(
            state
                .compensation(
                    false,
                    false,
                    None,
                    OneDepth::default(),
                    VertexClamp::default()
                )
                .unwrap()
                .is_some(),
            "the first"
        );
        assert_eq!(
            state.compensation(
                true,
                false,
                None,
                OneDepth::default(),
                VertexClamp::default()
            ),
            gap(RuntimeGap::ViewportDepthRange)
        );
        // Mirrored and unmirrored viewports: `sx` 1 under the mirrors' variant, viewport 0's bit
        // set; the first alone where the pipeline selects none.
        let mirrored = Viewport::new(4.0, 0.0, -4.0, 4.0, 0.0, 1.0);
        state.viewports = [mirrored, whole].into_iter().collect();
        let mixed = state
            .compensation(
                true,
                false,
                None,
                OneDepth::default(),
                VertexClamp::default(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            mixed.viewport.compensation,
            ViewportCompensation::new(false).with_mirrors(true)
        );
        assert_eq!(
            (mixed.viewport.values, mixed.viewport.mirrors, mixed.depth),
            ([1.0, 0.0, 1.0, 0.0], 0b01, None)
        );
        let first = state
            .compensation(
                false,
                false,
                None,
                OneDepth::default(),
                VertexClamp::default(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            (first.viewport.values, first.viewport.mirrors),
            ([-1.0, 0.0, 1.0, 0.0], 0)
        );
        // One compensated depth through both, viewport 1's bit set.
        let range = Viewport::new(0.0, 0.0, 4.0, 4.0, 0.5, 2.0);
        state.viewports = [range, Viewport::new(4.0, 0.0, -4.0, 4.0, 0.5, 2.0)]
            .into_iter()
            .collect();
        let both = state
            .compensation(
                true,
                false,
                None,
                OneDepth::default(),
                VertexClamp::default(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            both.viewport.compensation,
            ViewportCompensation::new(true).with_mirrors(true)
        );
        assert_eq!(
            (both.viewport.values, both.viewport.mirrors, both.depth),
            ([1.0, 0.5, 1.5, 1.0], 0b10, Some([0.0, 1.0]))
        );
        // Mirrors beside different depths stay refused.
        state.viewports = [range, mirrored].into_iter().collect();
        assert_eq!(
            state.compensation(
                true,
                false,
                None,
                OneDepth::default(),
                VertexClamp::default()
            ),
            gap(RuntimeGap::ViewportDepthRange)
        );
    }

    /// In depth clamp mode a one clamp of one depth takes no compensation of the depth: the draw's
    /// Vulkan range collapses to it (0 for a range below 0, a Metal probe on the M3 Max), past `[0,
    /// 1]` only under `VK_EXT_depth_range_unrestricted`, where the fragment function reads no depth
    /// of its position and the bias moves no sample off it, natively too; a mirror keeps its `sx`;
    /// natively the range stays and the clamp is the one depth.
    #[test]
    fn one_depth_clamps_collapse_the_vulkan_range() {
        use metal2vulkan::vk::PrimitiveTopology as T;
        use DepthBiasEffect::{Factors, Offset, Saturated};
        // The repository's profile: this test runs only from the workspace checkout.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../metal2vulkan-validation/profiles/moltenvk-m3.json"
        );
        let text = std::fs::read_to_string(path).unwrap();
        let description =
            metal2vulkan::device::DeviceDescription::from_khronos_json(&text, None).unwrap();
        let device = metal2vulkan::device::Device::new(
            description,
            metal2vulkan::device::DevicePolicy::default(),
        )
        .unwrap();
        let profile = device.profile();
        let range = || Err(Error::NotImplemented(RuntimeGap::ViewportDepthRange));
        let bias_gap = || Err(Error::NotImplemented(RuntimeGap::NonFiniteDepthBias));
        let plain = OneDepth::default();
        let unrestricted = OneDepth {
            unrestricted: true,
            ..plain
        };
        let reads = OneDepth {
            reads: true,
            ..unrestricted
        };
        let mut state = EncoderState::metal_defaults([4, 4]);
        state.clip_mode = DepthClipMode::Clamp;
        let at = |znear, zfar| vec![Viewport::new(0.0, 0.0, 4.0, 4.0, znear, zfar)];
        let depths = |state: &EncoderState, native| {
            state
                .draw_viewports(false, T::TRIANGLE_LIST, profile, native)
                .map(|v| [v[0].znear, v[0].zfar])
        };
        // 0 of -1 to 0, -2 to -1 and -1 to -2, and 1 of 1 to 2, on any device; 1.5 of 1.5 to 3
        // and 2 of 2 to 2 past [0, 1] under the extension alone.
        for ((znear, zfar), depth) in [
            ((-1.0, 0.0), 0.0),
            ((-2.0, -1.0), 0.0),
            ((-1.0, -2.0), 0.0),
            ((1.0, 2.0), 1.0),
            ((1.5, 3.0), 1.5),
            ((2.0, 2.0), 2.0),
        ] {
            state.viewports = at(znear, zfar).into_iter().collect();
            let case = format!("{znear} {zfar}");
            let want = match depth <= 1.0 {
                true => Ok(None),
                false => range(),
            };
            assert_eq!(
                state.compensation(false, false, None, plain, VertexClamp::default()),
                want,
                "{case}"
            );
            for one in [unrestricted, reads] {
                let want = if one.reads { range() } else { Ok(None) };
                assert_eq!(
                    state.compensation(false, false, None, one, VertexClamp::default()),
                    want,
                    "{case}"
                );
                assert_eq!(
                    state.compensation(false, true, None, one, VertexClamp::default()),
                    Ok(None),
                    "{case}"
                );
            }
            assert_eq!(depths(&state, false), Ok([depth; 2]), "{case}");
            assert_eq!(depths(&state, true), Ok([znear, zfar]), "{case}: native");
            let one = [depth as f32; 2].map(f32::to_bits);
            assert_eq!(
                state.depth_range(false, None).map(|r| r.map(|r| r.clamp)),
                Ok(Some(one)),
                "{case}: the native clamp"
            );
        }
        // Factors move a sample of 1.5 to 3 off 1.5 (to 1.75 under an offset of 0.75), natively
        // too, but not one of -1 to 0 off 0 or of 2 to 2 off 2, and factors of no offset none;
        // a saturated depth of the one depth collapses, an offset is refused, and the range's
        // larger depth, 3, stores through a Vulkan range of Metal's clamp of each sample, 1.5 to 3,
        // under the extension.
        let factors = Factors {
            constant: 1.0,
            slope: 0.0,
            clamp: 0.75,
        };
        let none = Factors {
            constant: 0.0,
            slope: 0.0,
            clamp: 0.25,
        };
        let with = |state: &EncoderState, native, bias| {
            state.compensation(
                false,
                native,
                Some(bias),
                unrestricted,
                VertexClamp::default(),
            )
        };
        state.viewports = at(1.5, 3.0).into_iter().collect();
        assert_eq!(with(&state, false, factors), range());
        assert_eq!(with(&state, true, factors), range(), "natively");
        assert_eq!(with(&state, false, none), Ok(None));
        assert_eq!(with(&state, false, Saturated { high: false }), Ok(None));
        assert_eq!(
            with(&state, false, Saturated { high: true })
                .map(|c| c.map(|c| (c.viewport.values, c.depth))),
            Ok(Some(([1.0, 1.0, 0.0, 0.0], Some([1.5, 3.0]))))
        );
        assert_eq!(with(&state, true, Saturated { high: true }), Ok(None));
        assert_eq!(with(&state, false, Offset(0.25)), bias_gap());
        for (znear, zfar) in [(-1.0, 0.0), (2.0, 2.0)] {
            state.viewports = at(znear, zfar).into_iter().collect();
            assert_eq!(with(&state, false, factors), Ok(None), "{znear} {zfar}");
            assert_eq!(with(&state, true, factors), Ok(None), "{znear} {zfar}");
        }
        // A mirror keeps its `sx` beside the collapsed depth.
        state.viewports = [Viewport::new(4.0, 0.0, -4.0, 4.0, 1.5, 3.0)]
            .into_iter()
            .collect();
        let mirror = state
            .compensation(false, false, None, unrestricted, VertexClamp::default())
            .unwrap()
            .unwrap();
        assert_eq!(
            (mirror.viewport.values, mirror.depth),
            ([-1.0, 0.0, 1.0, 0.0], None)
        );
        assert_eq!(depths(&state, false), Ok([1.5, 1.5]));
        // Clip mode clamps no range to one depth.
        state.clip_mode = DepthClipMode::Clip;
        state.viewports = at(1.5, 3.0).into_iter().collect();
        assert_eq!(depths(&state, false), Ok([1.5, 3.0]));
    }

    /// A bias's offset through a range whose vertices Metal clamps adds after the clamp (Metal
    /// probes on an M4 and the M3 Max): in clip mode after the vertex stage's clamp at 0; in clamp
    /// mode through Metal's clamp of each sample where the device holds a Vulkan range past 1;
    /// inside `[0, 1]` in clamp mode without a vertex clamp. Bias factors take the same range, and
    /// a saturated depth past the one clamp the range up to it.
    #[test]
    fn offsets_add_after_the_vertex_clamp() {
        use DepthBiasEffect::{Factors, Offset, Saturated};
        type Taken = ([f32; 4], bool, [f32; 2], Option<[f32; 2]>);
        let viewport = |n, f| Viewport::new(0.0, 0.0, 4.0, 4.0, n, f);
        let takes = VertexClamp { takes: true };
        let unclamped = VertexClamp::default();
        let wide = OneDepth {
            unrestricted: true,
            reads: false,
        };
        let narrow = OneDepth::default();
        let unbounded = [f32::NEG_INFINITY, f32::INFINITY];
        let gap = Err(Error::NotImplemented(RuntimeGap::NonFiniteDepthBias));
        let of = |state: &EncoderState, native, bias, one_depth, clamp| {
            state
                .compensation(false, native, Some(bias), one_depth, clamp)
                .map(|c| {
                    c.map(|c| -> Taken {
                        let v = c.viewport;
                        (
                            v.values,
                            v.compensation.vertex_clamp,
                            v.vertex_clamp,
                            c.depth,
                        )
                    })
                })
        };
        let third = |x: f64| (x / 1.5) as f32;
        let factors = Factors {
            constant: 1.0,
            slope: 0.0,
            clamp: 0.0,
        };
        // Clip mode through -0.5 to 1.5: `a` is `n + o`, each vertex's depth plus `o` at least
        // `o`, natively too.
        let mut state = EncoderState::metal_defaults([4, 4]);
        state.viewports = [viewport(-0.5, 1.5)].into_iter().collect();
        for native in [false, true] {
            assert_eq!(
                of(&state, native, Offset(0.25), narrow, takes),
                Ok(Some((
                    [1.0, -0.25, 2.0, 1.0],
                    true,
                    [0.25, f32::INFINITY],
                    Some([0.0, 1.0])
                ))),
                "{native}"
            );
        }
        assert_eq!(of(&state, false, Offset(0.25), narrow, unclamped), gap);
        // Clamp mode through -0.5 to 1.5: Metal's clamp of each sample, [0, 1.5], under the
        // extension; without it an offset is refused and factors keep the one clamp [0, 1].
        state.clip_mode = DepthClipMode::Clamp;
        assert_eq!(
            of(&state, false, Offset(0.25), wide, takes),
            Ok(Some((
                [1.0, third(-0.25), third(2.0), 0.0],
                true,
                [third(0.25), third(1.25)],
                Some([0.0, 1.5])
            )))
        );
        assert_eq!(of(&state, false, Offset(0.25), narrow, takes), gap);
        assert_eq!(
            of(&state, true, factors, wide, takes),
            Ok(Some((
                [1.0, third(-0.5), third(2.0), 0.0],
                true,
                [0.0, third(1.0)],
                Some([0.0, 1.5])
            )))
        );
        assert_eq!(
            of(&state, true, factors, narrow, takes),
            Ok(Some((
                [1.0, -0.5, 2.0, 0.0],
                true,
                [0.0, 1.0],
                Some([0.0, 1.0])
            )))
        );
        // A saturated depth past the one clamp: the range up to it, under the extension.
        let up = Saturated { high: true };
        assert_eq!(
            of(&state, false, up, wide, unclamped),
            Ok(Some((
                [1.0, 1.0, 0.0, 0.0],
                false,
                unbounded,
                Some([0.0, 1.5])
            )))
        );
        assert_eq!(of(&state, false, up, narrow, unclamped), gap);
        // Clamp mode through 1.5 to 3, whose one clamp is 1.5: biased, its vertices clamped to
        // `[0, 1]`, `z / w` to [-1, -1/3], through [1.5, 3]; without the extension refused.
        state.viewports = [viewport(1.5, 3.0)].into_iter().collect();
        assert_eq!(
            of(&state, false, factors, wide, takes),
            Ok(Some((
                [1.0, 0.0, 1.0, 0.0],
                true,
                [-1.0, third(-0.5)],
                Some([1.5, 3.0])
            )))
        );
        assert_eq!(
            of(&state, false, factors, narrow, takes),
            Err(Error::NotImplemented(RuntimeGap::ViewportDepthRange))
        );
        // Clamp mode inside [0, 1]: no vertex clamp, the offset folded through the range,
        // natively too; one depth alone where the range is one.
        state.viewports = [viewport(0.25, 0.75)].into_iter().collect();
        for (native, clamp) in [(false, unclamped), (true, takes)] {
            assert_eq!(
                of(&state, native, Offset(0.25), narrow, clamp),
                Ok(Some((
                    [1.0, 0.5, 1.0, 0.0],
                    false,
                    unbounded,
                    Some([0.25, 0.75])
                ))),
                "{native}"
            );
        }
        state.viewports = [viewport(0.5, 0.5)].into_iter().collect();
        assert_eq!(
            of(&state, false, Offset(0.25), narrow, takes),
            Ok(Some((
                [1.0, 0.0, 0.0, 0.0],
                false,
                unbounded,
                Some([0.5, 0.5])
            )))
        );
        // Factors inside [0, 1] stay Vulkan's bias: no compensation.
        assert_eq!(of(&state, false, factors, narrow, takes), Ok(None));
    }

    /// A saturated or offset depth bias folds into a compensated depth as Metal biases (a Metal
    /// probe on an M4): in clip mode at 1 or 0, an offset through a range of no negative depth; in
    /// clamp mode at the larger depth, or the smaller but at least 0, where the one clamp holds it;
    /// the rest refused, and factors left to Vulkan's bias.
    #[test]
    fn saturated_and_offset_biases_fold_into_the_compensation() {
        use DepthBiasEffect::{Factors, Offset, Saturated};
        let gap = Err(Error::NotImplemented(RuntimeGap::NonFiniteDepthBias));
        let values = |state: &EncoderState, bias| {
            state
                .compensation(
                    false,
                    false,
                    Some(bias),
                    OneDepth::default(),
                    VertexClamp::default(),
                )
                .map(|c| c.map(|c| c.viewport.values))
        };
        let mut state = EncoderState::metal_defaults([4, 4]);
        state.viewports = [Viewport::new(0.0, 0.0, 4.0, 4.0, 0.5, 1.5)]
            .into_iter()
            .collect();
        let up = Saturated { high: true };
        let down = Saturated { high: false };
        assert_eq!(values(&state, up), Ok(Some([1.0, 1.0, 0.0, 1.0])));
        assert_eq!(values(&state, down), Ok(Some([1.0, 0.0, 0.0, 1.0])));
        assert_eq!(
            values(&state, Offset(0.25)),
            Ok(Some([1.0, 0.75, 1.0, 1.0]))
        );
        assert_eq!(
            values(&state, Offset(-0.25)),
            Ok(Some([1.0, 0.25, 1.0, 1.0]))
        );
        state.viewports = [Viewport::new(0.0, 0.0, 4.0, 4.0, -0.5, 1.5)]
            .into_iter()
            .collect();
        assert_eq!(values(&state, up), Ok(Some([1.0, 1.0, 0.0, 1.0])));
        assert_eq!(values(&state, Offset(0.25)), gap, "a negative depth");
        // The one clamp is [0, 1]: the larger depth, 1.5, is past it; the smaller is 0.
        state.clip_mode = DepthClipMode::Clamp;
        assert_eq!(values(&state, up), gap, "past the clamp");
        assert_eq!(values(&state, down), Ok(Some([1.0, 0.0, 0.0, 0.0])));
        assert_eq!(values(&state, Offset(0.25)), gap, "an offset in clamp mode");
        // The one clamp is [0, 0.5]: the larger depth is its top.
        state.viewports = [Viewport::new(0.0, 0.0, 4.0, 4.0, 0.5, -0.5)]
            .into_iter()
            .collect();
        assert_eq!(values(&state, up), Ok(Some([1.0, 1.0, 0.0, 0.0])));
        let factors = Factors {
            constant: 1.0,
            slope: 0.0,
            clamp: 0.0,
        };
        assert_eq!(
            state.compensation(
                false,
                false,
                Some(factors),
                OneDepth::default(),
                VertexClamp::default()
            ),
            state.compensation(
                false,
                false,
                None,
                OneDepth::default(),
                VertexClamp::default()
            )
        );
    }

    /// Metal's clamp of each vertex's depth (Metal probes on the M3 Max and an M4): the viewports
    /// it differs through ([`clamps_vertices`]), and the compensation that takes it, natively too:
    /// at 0 in clip mode (`z / w` to `[0, +inf]` in a Vulkan range of `[0, 1]`), to `[0, 1]` in
    /// clamp mode (`z / w` of the one clamp `[L, H]` to `[-L / (H - L), (1 - L) / (H - L)]`); none
    /// where the draw does not take it, through a range it does not differ through, or under a
    /// saturated bias; viewports of different clamped depths refused, mirrors alone kept.
    #[test]
    fn vertex_clamps_clamp_each_vertexs_depth() {
        let viewport = |n, f| Viewport::new(0.0, 0.0, 4.0, 4.0, n, f);
        for (n, f, clip, clamp, offset) in [
            (-0.5, 1.5, true, true, true),
            (-1.0, 0.5, true, true, true),
            (2.0, -1.0, true, true, true),
            (0.5, 2.0, false, true, true),
            (0.0, 1.0, false, false, false),
            (0.25, 0.75, false, false, false),
            // One depth in clamp mode: 1.5, and 0; biased, 1.5 to 3 of two.
            (1.5, 3.0, false, false, true),
            (-2.0, -1.0, true, false, false),
        ] {
            for offsets in [false, true] {
                assert_eq!(
                    clamps_vertices(&viewport(n, f), false, offsets),
                    clip,
                    "{n} {f} clip {offsets}"
                );
            }
            assert_eq!(
                clamps_vertices(&viewport(n, f), true, false),
                clamp,
                "{n} {f} clamp"
            );
            assert_eq!(
                clamps_vertices(&viewport(n, f), true, true),
                offset,
                "{n} {f} clamp, offset"
            );
        }
        let takes = VertexClamp { takes: true };
        let plain = OneDepth::default();
        let unclamped = VertexClamp::default();
        let mut state = EncoderState::metal_defaults([4, 4]);
        state.viewports = [viewport(-0.5, 1.5)].into_iter().collect();
        for native in [false, true] {
            let c = state
                .compensation(false, native, None, plain, takes)
                .unwrap()
                .unwrap();
            let variant = c.viewport.compensation;
            assert!(variant.depth && variant.vertex_clamp, "{native}");
            assert_eq!(c.viewport.values, [1.0, -0.5, 2.0, 1.0], "{native}");
            assert_eq!(c.viewport.vertex_clamp, [0.0, f32::INFINITY], "{native}");
            assert_eq!(c.depth, Some([0.0, 1.0]), "{native}");
        }
        // Without the clamp: the depth's compensation alone, and none natively.
        let c = state
            .compensation(false, false, None, plain, unclamped)
            .unwrap()
            .unwrap();
        assert!(!c.viewport.compensation.vertex_clamp);
        assert_eq!(c.viewport.vertex_clamp, [f32::NEG_INFINITY, f32::INFINITY]);
        assert_eq!(
            state.compensation(false, true, None, plain, unclamped),
            Ok(None)
        );
        // Factors keep the clamp; a saturated bias, one depth, takes none.
        let factors = DepthBiasEffect::Factors {
            constant: 1.0,
            slope: 1.0,
            clamp: 0.0,
        };
        assert_eq!(
            state.compensation(false, true, Some(factors), plain, takes),
            state.compensation(false, true, None, plain, takes)
        );
        let up = DepthBiasEffect::Saturated { high: true };
        let c = state
            .compensation(false, false, Some(up), plain, takes)
            .unwrap()
            .unwrap();
        assert!(!c.viewport.compensation.vertex_clamp);
        assert_eq!(
            state.compensation(false, true, Some(up), plain, takes),
            Ok(None)
        );
        // Clamp mode: the one clamp [0.5, 1] of 0.5 to 2, `z / w` to [-1, 1]; [0, 1] of -0.5 to
        // 1.5, `z / w` to [0, 1].
        state.clip_mode = DepthClipMode::Clamp;
        state.viewports = [viewport(0.5, 2.0)].into_iter().collect();
        let c = state
            .compensation(false, true, None, plain, takes)
            .unwrap()
            .unwrap();
        assert_eq!(c.viewport.values, [1.0, 0.0, 3.0, 0.0]);
        assert_eq!(c.viewport.vertex_clamp, [-1.0, 1.0]);
        assert_eq!(c.depth, Some([0.5, 1.0]));
        state.viewports = [viewport(-0.5, 1.5)].into_iter().collect();
        let c = state
            .compensation(false, false, None, plain, takes)
            .unwrap()
            .unwrap();
        assert_eq!(c.viewport.values, [1.0, -0.5, 2.0, 0.0]);
        assert_eq!(c.viewport.vertex_clamp, [0.0, 1.0]);
        // Clip mode through 0.5 to 2, no negative depth: nothing to clamp, and none natively.
        state.clip_mode = DepthClipMode::Clip;
        state.viewports = [viewport(0.5, 2.0)].into_iter().collect();
        assert_eq!(
            state.compensation(false, true, None, plain, takes),
            Ok(None)
        );
        let c = state
            .compensation(false, false, None, plain, takes)
            .unwrap()
            .unwrap();
        assert!(!c.viewport.compensation.vertex_clamp);
        // Viewports of different clamped depths: refused, natively too, where the caller keeps
        // the device's ranges unclamped.
        state.viewports = [viewport(-0.5, 1.5), viewport(-1.0, 1.0)]
            .into_iter()
            .collect();
        assert_eq!(
            state.compensation(true, true, None, plain, takes),
            Err(Error::NotImplemented(RuntimeGap::ViewportDepthRange))
        );
        assert_eq!(
            state.compensation(true, true, None, plain, unclamped),
            Ok(None)
        );
        // Mirrored and unmirrored viewports of one range keep the clamp's bounds.
        let mirrored = Viewport::new(4.0, 0.0, -4.0, 4.0, -0.5, 1.5);
        state.viewports = [viewport(-0.5, 1.5), mirrored].into_iter().collect();
        let c = state
            .compensation(true, true, None, plain, takes)
            .unwrap()
            .unwrap();
        let variant = c.viewport.compensation;
        assert!(variant.mirrors && variant.vertex_clamp);
        assert_eq!(c.viewport.mirrors, 0b10);
        assert_eq!(c.viewport.vertex_clamp, [0.0, f32::INFINITY]);
    }

    /// A draw of points widens its Vulkan viewports by the least power of two from 2 holding
    /// Metal's guard band and a pixel past each edge, about the same center; `None` past the
    /// device's limits or where the widened origin is not a `float`.
    #[test]
    fn points_widen_their_viewports_about_their_centers() {
        let limits = RegionLimits::new([16384, 16384], [-32768.0, 32767.0]);
        let mut state = EncoderState::metal_defaults([8, 8]);
        assert_eq!(state.point_guard(false, &limits), Some([2.0, 2.0]));
        let vk = EncoderState::vk_viewports(state.used_viewports(false), None, Some([2.0, 2.0]));
        assert_eq!(
            [vk[0].x, vk[0].y, vk[0].width, vk[0].height],
            [-4.0, 12.0, 16.0, -16.0]
        );
        // A one-pixel column: 1 + 2 / 1 needs 4; a quarter pixel's 9 needs 16.
        state.viewports = [Viewport::new(4.5, 0.0, 1.0, 8.0, 0.0, 1.0)]
            .into_iter()
            .collect();
        assert_eq!(state.point_guard(false, &limits), Some([4.0, 2.0]));
        let vk = EncoderState::vk_viewports(state.used_viewports(false), None, Some([4.0, 2.0]));
        assert_eq!([vk[0].x, vk[0].width], [3.0, 4.0]);
        assert_eq!(vk[0].x + vk[0].width / 2.0, 5.0, "the center stays");
        state.viewports = [Viewport::new(4.0, 0.0, 0.25, 8.0, 0.0, 1.0)]
            .into_iter()
            .collect();
        assert_eq!(state.point_guard(false, &limits), Some([16.0, 2.0]));
        // Mirrored and flipped viewports widen alike.
        state.viewports = [Viewport::new(8.0, 8.0, -8.0, -8.0, 0.0, 1.0)]
            .into_iter()
            .collect();
        assert_eq!(state.point_guard(false, &limits), Some([2.0, 2.0]));
        // Past the device's dimensions or bounds, or an origin no `float` holds.
        state.viewports = [Viewport::new(0.0, 0.0, 16384.0, 8.0, 0.0, 1.0)]
            .into_iter()
            .collect();
        assert_eq!(state.point_guard(false, &limits), None);
        state.viewports = [Viewport::new(30000.0, 0.0, 4000.0, 8.0, 0.0, 1.0)]
            .into_iter()
            .collect();
        assert_eq!(state.point_guard(false, &limits), None);
        state.viewports = [Viewport::new(0.1, 0.0, 64.0, 8.0, 0.0, 1.0)]
            .into_iter()
            .collect();
        assert_eq!(state.point_guard(false, &limits), None);
        // Every viewport a selecting pipeline uses takes the widest scale.
        state.viewports = [
            Viewport::new(0.0, 0.0, 8.0, 8.0, 0.0, 1.0),
            Viewport::new(4.0, 4.0, 1.0, 2.0, 0.0, 1.0),
        ]
        .into_iter()
        .collect();
        assert_eq!(state.point_guard(true, &limits), Some([4.0, 2.0]));
    }

    /// References saturate at 255 (probe 4: 0x1a5 replaced stores 0xff).
    #[test]
    fn stencil_references_saturate() {
        let mut state = EncoderState::metal_defaults([4, 4]);
        state.set_stencil_references(0x1a5, 0x22);
        assert_eq!(state.stencil_reference(), [0xff, 0x22]);
        state.set_stencil_references(u32::MAX, 0x100);
        assert_eq!(state.stencil_reference(), [0xff, 0xff]);
    }

    /// Metal's measured answers for NaN and infinite factors (Metal probes on an M3 Max and an M4
    /// alike), each `(constant, slope, clamp)` with the effect Metal's stored depths show, or
    /// `None` where flat and sloped triangles differ.
    #[test]
    fn non_finite_depth_biases_take_metal_s_measured_effect() {
        use DepthBiasEffect::{Factors, Offset, Saturated};
        let (inf, nan) = (f32::INFINITY, f32::NAN);
        let high = Some(Saturated { high: true });
        let low = Some(Saturated { high: false });
        let cases = [
            // Probe 2.
            ((nan, 0.0, 0.0), high),
            ((inf, 0.0, 0.0), high),
            ((-inf, 0.0, 0.0), low),
            ((inf, 0.0, 0.125), Some(Offset(0.125))),
            ((-inf, 0.0, -0.125), Some(Offset(-0.125))),
            ((nan, 0.0, 0.125), Some(Offset(0.125))),
            ((0.0, nan, 0.0), high),
            ((0.0, inf, 0.0), high),
            ((0.0, inf, 0.0625), Some(Offset(0.0625))),
            (
                (1000.0, 0.0, nan),
                Some(Factors {
                    constant: 1000.0,
                    slope: 0.0,
                    clamp: 0.0,
                }),
            ),
            (
                (1000.0, 0.0, inf),
                Some(Factors {
                    constant: 1000.0,
                    slope: 0.0,
                    clamp: 0.0,
                }),
            ),
            (
                (-1000.0, 0.0, -inf),
                Some(Factors {
                    constant: -1000.0,
                    slope: 0.0,
                    clamp: 0.0,
                }),
            ),
            // Probe 3: a NaN constant is +inf against a negative clamp; a slope's NaN takes the
            // clamp of either sign.
            ((nan, 0.0, -0.125), high),
            ((inf, 0.0, -0.125), high),
            ((-inf, 0.0, 0.125), low),
            ((0.0, nan, -0.125), Some(Offset(-0.125))),
            ((0.0, -inf, -0.125), Some(Offset(-0.125))),
            ((inf, -inf, 0.0), high),
            ((-inf, inf, 0.0), high),
            ((nan, 0.0, inf), high),
            ((nan, 0.0, -inf), high),
            ((inf, 0.0, nan), high),
            ((-inf, 0.0, nan), low),
            // Probe 4.
            ((0.0, nan, 0.1), Some(Offset(0.1))),
            ((inf, 0.0, 0.4), Some(Offset(0.4))),
            // Flat and sloped triangles differ (flat: 1, 0.625, 0.375; sloped: 0, 0, 1).
            ((0.0, -inf, 0.0), None),
            ((0.0, -inf, 0.125), None),
            ((0.0, inf, -0.125), None),
            ((-inf, -inf, 0.0), None),
        ];
        for ((constant, slope, clamp), effect) in cases {
            assert_eq!(
                depth_bias_effect(constant, slope, clamp),
                effect,
                "({constant}, {slope}, {clamp})"
            );
        }
        let mut state = EncoderState::metal_defaults([4, 4]);
        assert_eq!(
            state.set_depth_bias(0.0, -inf, 0.0),
            Err(Error::NotImplemented(RuntimeGap::NonFiniteDepthBias))
        );
        assert!(state.set_depth_bias(nan, 0.0, 0.0).is_ok());
    }

    /// A saturated bias collapses each viewport's depth range at 1 or 0, or in depth clamp mode at
    /// its larger or smaller depth; an offset moves it where it stays in `[0, 1]` in clip mode
    /// and is refused otherwise; points and lines take no bias (probes 2 to 4).
    #[test]
    fn biased_depths_move_the_draw_s_viewports() {
        use metal2vulkan::vk::PrimitiveTopology as T;
        // The repository's profile: this test runs only from the workspace checkout.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../metal2vulkan-validation/profiles/moltenvk-m3.json"
        );
        let text = std::fs::read_to_string(path).unwrap();
        let description =
            metal2vulkan::device::DeviceDescription::from_khronos_json(&text, None).unwrap();
        let device = metal2vulkan::device::Device::new(
            description,
            metal2vulkan::device::DevicePolicy::default(),
        )
        .unwrap();
        let profile = device.profile();
        let mut state = EncoderState::metal_defaults([4, 4]);
        state.viewports = [Viewport::new(0.0, 0.0, 4.0, 4.0, 0.75, 0.25)]
            .into_iter()
            .collect();
        let depths = |state: &EncoderState, topology| {
            state
                .draw_viewports(false, topology, profile, false)
                .map(|v| [v[0].znear, v[0].zfar])
        };
        let gap = Err(Error::NotImplemented(RuntimeGap::NonFiniteDepthBias));
        state.set_depth_bias(f32::INFINITY, 0.0, 0.0).unwrap();
        assert_eq!(depths(&state, T::TRIANGLE_LIST), Ok([1.0, 1.0]));
        assert_eq!(depths(&state, T::LINE_LIST), Ok([0.75, 0.25]), "no bias");
        state.set_depth_bias(f32::NEG_INFINITY, 0.0, 0.0).unwrap();
        assert_eq!(depths(&state, T::TRIANGLE_STRIP), Ok([0.0, 0.0]));
        state.set_depth_bias(0.0, 0.0, 0.0).unwrap();
        assert_eq!(depths(&state, T::TRIANGLE_LIST), Ok([0.75, 0.25]));
        state.set_depth_bias(f32::INFINITY, 0.0, 0.125).unwrap();
        assert_eq!(depths(&state, T::TRIANGLE_LIST), Ok([0.875, 0.375]));
        state.set_depth_bias(f32::INFINITY, 0.0, 0.4).unwrap();
        assert_eq!(depths(&state, T::TRIANGLE_LIST), gap, "moved past 1");
        // Natively, the native clamp to [0, 1] after the move is Metal's, through a range of no
        // negative depth.
        let native = |state: &EncoderState| {
            let bias = state.draw_bias(T::TRIANGLE_LIST, profile).unwrap();
            let moved = state
                .draw_viewports(false, T::TRIANGLE_LIST, profile, true)
                .map(|v| [v[0].znear, v[0].zfar]);
            (moved, state.depth_range(false, bias))
        };
        state.set_depth_bias(f32::INFINITY, 0.0, 0.375).unwrap();
        let clipped = DepthRange {
            clip: true,
            clamp: [0.0f32, 1.0].map(f32::to_bits),
        };
        assert_eq!(native(&state), (Ok([1.125, 0.625]), Ok(Some(clipped))));
        state.viewports = [Viewport::new(0.0, 0.0, 4.0, 4.0, -0.25, 0.25)]
            .into_iter()
            .collect();
        assert_eq!(native(&state).0, gap, "a negative depth clamps at 0 first");
        state.viewports = [Viewport::new(0.0, 0.0, 4.0, 4.0, 0.75, 0.25)]
            .into_iter()
            .collect();
        state.clip_mode = DepthClipMode::Clamp;
        state.set_depth_bias(f32::INFINITY, 0.0, 0.0).unwrap();
        assert_eq!(depths(&state, T::TRIANGLE_LIST), Ok([0.75, 0.75]));
        state.set_depth_bias(f32::NEG_INFINITY, 0.0, 0.0).unwrap();
        assert_eq!(depths(&state, T::TRIANGLE_LIST), Ok([0.25, 0.25]));
        state.set_depth_bias(0.0, f32::NAN, 0.0).unwrap();
        assert_eq!(depths(&state, T::TRIANGLE_LIST), Ok([0.75, 0.75]));
        state.set_depth_bias(f32::INFINITY, 0.0, 0.125).unwrap();
        assert_eq!(
            depths(&state, T::TRIANGLE_LIST),
            gap,
            "an offset in clamp mode"
        );
        let bias = state.vk_depth_bias(T::TRIANGLE_LIST, profile).unwrap();
        assert_eq!(
            [bias.constant_factor, bias.slope_factor, bias.clamp],
            [0.0; 3],
            "the viewports take it"
        );
    }

    /// The range a function that runs its depth test itself clamps by: the draw's own, refused
    /// under a compensated depth, but for a returned depth through ranges inside `[0, 1]` (the
    /// written depth's clamp on a device that takes no range natively), which clamps to the
    /// viewport's own range.
    #[test]
    fn a_returned_depth_keeps_its_range_under_the_written_clamp_s_compensation() {
        use metal2vulkan::vk::PrimitiveTopology as T;
        // The repository's profile: this test runs only from the workspace checkout.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../metal2vulkan-validation/profiles/intel-irisxe-rplp-anv.json"
        );
        let text = std::fs::read_to_string(path).unwrap();
        let description =
            metal2vulkan::device::DeviceDescription::from_khronos_json(&text, None).unwrap();
        let device = metal2vulkan::device::Device::new(
            description,
            metal2vulkan::device::DevicePolicy::default(),
        )
        .unwrap();
        let profile = device.profile();
        let gap = Err(Error::NotImplemented(RuntimeGap::ViewportDepthRange));
        let mut state = EncoderState::metal_defaults([4, 4]);
        state.clip_mode = DepthClipMode::Clamp;
        state.viewports = [Viewport::new(0.0, 0.0, 4.0, 4.0, 0.25, 0.75)]
            .into_iter()
            .collect();
        let range = |state: &EncoderState, compensated, returned| {
            state.shader_tests_range(
                false,
                T::TRIANGLE_LIST,
                profile,
                (false, compensated),
                returned,
            )
        };
        assert_eq!(range(&state, false, false), Ok([0.25, 0.75]));
        assert_eq!(range(&state, false, true), Ok([0.25, 0.75]));
        assert_eq!(range(&state, true, false), gap, "the window depth");
        assert_eq!(range(&state, true, true), Ok([0.25, 0.75]), "returned");
        state.viewports = [Viewport::new(0.0, 0.0, 4.0, 4.0, 0.5, 2.0)]
            .into_iter()
            .collect();
        assert_eq!(range(&state, true, true), gap, "past [0, 1]");
    }
}
