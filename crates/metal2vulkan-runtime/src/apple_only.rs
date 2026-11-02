//! The Apple-only calls of the consumer's trait: each is accepted where it is exact on Vulkan, and
//! otherwise refused with [`AppleOnly`] naming the missing feature, and [`AppleOnly::gap`] saying
//! why: Metal's answer is not measured, or the runtime has not built it. None is a device's gap:
//! each is the runtime's own.
//!
//! Exact means one of:
//! - the call states Metal's default, which every runtime object already has: the viewport
//!   transform on;
//! - the call sets encoder state the runtime draws as Metal does: a provoking vertex mode of 0
//!   or 1, `render/provoking.rs`; a line width, `render/line_width.rs` (the device's own lines
//!   where they differ from Metal's are announced, `device-wide-lines`, and those at the nearest
//!   width they draw where neither they nor the band draw Metal's, `lines-at-nearest-width`);
//! - the call acts on state no runtime object has: Vulkan shows no texture compression, so a
//!   reinterpretation flush has nothing to flush, and after an invalidation Metal leaves the
//!   contents undefined, so keeping them is one of the outcomes Metal allows;
//! - the call sets encoder state a runtime pipeline reads as Metal's does: the alpha test
//!   reference, which a pipeline with the private alpha test compares against (`BindState`'s
//!   `alpha_test_reference`).
//!
//! - `setPointSize:`, which Metal measured makes a no-op: the point is the vertex function's
//!   `[[point_size]]`; and `setDepthCleared` and `setStencilCleared`, which change nothing Metal
//!   draws (the consumer's suite case on the M3 Max, pixel for pixel).
//! - `setVertexAmplificationMode:value:`, which sets the amplification count
//!   `setVertexAmplificationCount:viewMappings:` sets and keeps the view mappings that call set,
//!   whichever of the two comes last deciding (the consumer's measurements on the M3 Max, modes 0
//!   to 8, 15, 16, 255 and the largest values): the count is the value's set bits in mode 1 and the
//!   value in every other mode, exact from 1 to 8 (`amplification_mode_count`).
//!
//! A provoking vertex mode past 1, which Metal draws a third way, is accepted and drawn as mode 1
//! (patches as mode 0), announced at each draw whose shape and pipeline state Metal draws
//! otherwise there (`provoking-vertex-unmatched`).
//!
//! Every other call is refused, and records nothing:
//! - a vertex amplification mode whose count is 0, which draws a number of copies that changes
//!   from run to run on Metal, or above 8 (9 draws one copy and 16 and 32 eight, the rest
//!   unmeasured), and one whose count passes the view mappings kept (unmeasured);
//! - turning the viewport transform off, which Vulkan has no switch for and whose meaning on
//!   Metal is not measured;
//! - a texture over an IOSurface made with `kIOSurfaceClientDirection`, a key in no header of the
//!   public macOS SDK, refused by the IOSurface plane's stand-in
//!   (`Buffer::new_texture_with_iosurface_plane`), which is no call of the table.
//!
//! Texture fills (`blit_fill.rs`) refuse nothing here: a multisample texture's samples, a texture
//! buffer's bytes and a color packed in a format Metal packs no color in (the command buffer's
//! last packed texel, announced as `unpacked-fill-color`) are built. A multisample fill a device
//! cannot copy into the texture is refused there, `Unsupported`, as the device's gap.
//!
//! [`AppleOnlyCall`] is the table, and every call of it is here: the compute, render and blit
//! encoders', and the texture's protection options. The command buffer's protection options are
//! `command_buffer/protection.rs`'s, as Metal's GPU answers them, but the forms not measured
//! ([`AppleOnly::ProtectedContent`]). A texture's
//! rotation is the runtime's (`TextureDescriptor::rotation`), and command predication, loops
//! included, is `predication.rs`'s.

use super::blit::BlitCommandEncoder;
use super::blit_texture::check_slice_level;
use super::capabilities::MAX_VERTEX_AMPLIFICATION;
use super::compute::ComputeCommandEncoder;
use super::error::Error;
use super::render::RenderCall;
use super::render::RenderCommandEncoder;
use super::render::RenderLowering;
use super::replay::BlitCall;
use super::texture::Texture;

/// An Apple-only feature the runtime refuses, each for the reason [`AppleOnly::gap`] gives (No
/// device lacks any of them; each is a gap of the runtime's own).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AppleOnly {
    /// Protection options whose answer on Metal is not measured
    /// (`command_buffer/protection.rs`): options changed once an encoder was made under some, or
    /// after options from 0x10 up, and a resource state encoder made under 1, 3 or 7.
    ProtectedContent,
    /// The viewport transform turned off (`setViewportTransformEnabled:NO`).
    ViewportTransformDisabled,
    /// `setVertexAmplificationMode:value:` of a count (the value's set bits in mode 1, else the
    /// value) of 0, whose copies change from run to run on Metal, or above 8, or above the view
    /// mappings kept.
    VertexAmplificationMode,
    /// A texture over an IOSurface made with `kIOSurfaceClientDirection`
    /// (`newTextureWithDescriptor:iosurface:plane:` of such a surface): IOSurface SPI, a key
    /// IOSurface.framework exports and no header of the public macOS SDK declares, so the
    /// surface is the consumer's to build. The consumer reports that Metal drops a GPU write into
    /// a surface made with value 2 (read-only), without a fault; the runtime does not model the
    /// key's values.
    SurfaceClientDirection,
}

/// Why the runtime refuses an [`AppleOnly`] feature. None is a device's gap: no measured Vulkan
/// device lacks what an exact form would need. Each is the runtime's own, and a measurement can
/// lift it. A color fill whose bytes Metal leaves undefined, once refused as having no one
/// answer, is built: the runtime gives one of Metal's answers.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AppleOnlyGap {
    /// What Metal does on Apple's GPU is not measured, so the runtime does not guess it: the
    /// runtime's gap, which a measurement would let it build.
    Unmeasured,
}

impl AppleOnly {
    /// Why the runtime refuses the feature ([`AppleOnlyGap`]).
    pub const fn gap(self) -> AppleOnlyGap {
        match self {
            Self::ProtectedContent
            | Self::ViewportTransformDisabled
            | Self::VertexAmplificationMode
            | Self::SurfaceClientDirection => AppleOnlyGap::Unmeasured,
        }
    }
}

/// Metal's `protectionOptions` of a resource made without any, and of every runtime texture.
pub const DEFAULT_PROTECTION_OPTIONS: u64 = 0;

/// One Apple-only call (class A) with the arguments its answer depends on: the table each
/// encoder's, descriptor's and command buffer's method of that call answers by.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AppleOnlyCall {
    /// `invalidateCompressedTexture:` (blit encoder).
    InvalidateCompressedTexture,
    /// `invalidateCompressedTexture:slice:level:` (blit encoder).
    InvalidateCompressedTextureSlice,
    /// `insertCompressedTextureReinterpretationFlush` (compute encoder).
    CompressedTextureReinterpretationFlush,
    /// `setAlphaTestReferenceValue:` (render encoder).
    SetAlphaTestReferenceValue(f32),
    /// `setLineWidth:` (render encoder).
    SetLineWidth(f32),
    /// `setPointSize:` (render encoder).
    SetPointSize(f32),
    /// `setProvokingVertexMode:` (render encoder).
    SetProvokingVertexMode(u64),
    /// `setViewportTransformEnabled:` (render encoder).
    SetViewportTransformEnabled(bool),
    /// `setVertexAmplificationMode:value:` (render encoder).
    SetVertexAmplificationMode {
        /// The mode given.
        mode: u64,
        /// The value given.
        value: u64,
    },
    /// `setDepthCleared` (render encoder).
    SetDepthCleared,
    /// `setStencilCleared` (render encoder).
    SetStencilCleared,
}

impl AppleOnlyCall {
    /// Metal's selector.
    pub fn selector(self) -> &'static str {
        match self {
            AppleOnlyCall::InvalidateCompressedTexture => "invalidateCompressedTexture:",
            AppleOnlyCall::InvalidateCompressedTextureSlice => {
                "invalidateCompressedTexture:slice:level:"
            }
            AppleOnlyCall::CompressedTextureReinterpretationFlush => {
                "insertCompressedTextureReinterpretationFlush"
            }
            AppleOnlyCall::SetAlphaTestReferenceValue(_) => "setAlphaTestReferenceValue:",
            AppleOnlyCall::SetLineWidth(_) => "setLineWidth:",
            AppleOnlyCall::SetPointSize(_) => "setPointSize:",
            AppleOnlyCall::SetProvokingVertexMode(_) => "setProvokingVertexMode:",
            AppleOnlyCall::SetViewportTransformEnabled(_) => "setViewportTransformEnabled:",
            AppleOnlyCall::SetVertexAmplificationMode { .. } => "setVertexAmplificationMode:value:",
            AppleOnlyCall::SetDepthCleared => "setDepthCleared",
            AppleOnlyCall::SetStencilCleared => "setStencilCleared",
        }
    }

    /// The runtime's answer: `Ok` where it takes the call (it states Metal's default, acts on
    /// state no runtime object has, so it records nothing, or sets the encoder's provoking
    /// vertex or line width, [`RenderCommandEncoder::set_provoking_vertex_mode`] and
    /// [`RenderCommandEncoder::set_line_width`]), else the refusal naming the missing feature.
    pub fn check(self) -> Result<(), AppleOnly> {
        match self {
            AppleOnlyCall::InvalidateCompressedTexture
            | AppleOnlyCall::InvalidateCompressedTextureSlice
            | AppleOnlyCall::CompressedTextureReinterpretationFlush
            | AppleOnlyCall::SetAlphaTestReferenceValue(_)
            | AppleOnlyCall::SetLineWidth(_)
            | AppleOnlyCall::SetPointSize(_)
            | AppleOnlyCall::SetProvokingVertexMode(_)
            | AppleOnlyCall::SetDepthCleared
            | AppleOnlyCall::SetStencilCleared => Ok(()),
            AppleOnlyCall::SetViewportTransformEnabled(enabled) => {
                default_or(enabled, AppleOnly::ViewportTransformDisabled)
            }
            AppleOnlyCall::SetVertexAmplificationMode { mode, value } => default_or(
                (1..=MAX_VERTEX_AMPLIFICATION).contains(&amplification_mode_count(mode, value)),
                AppleOnly::VertexAmplificationMode,
            ),
        }
    }
}

/// The amplification count `setVertexAmplificationMode:value:` sets (the consumer's measurements
/// on the M3 Max): the number of `value`'s set bits in mode 1, and `value` in every other mode.
/// Metal draws that many copies, amplification IDs from 0 up, for a count from 1 to 8.
pub(crate) fn amplification_mode_count(mode: u64, value: u64) -> u64 {
    match mode {
        1 => u64::from(value.count_ones()),
        _ => value,
    }
}

/// `Ok` where the call states Metal's default, else the refusal of `feature`.
fn default_or(is_default: bool, feature: AppleOnly) -> Result<(), AppleOnly> {
    if is_default {
        Ok(())
    } else {
        Err(feature)
    }
}

/// The runtime error of `call`'s answer.
fn answer(call: AppleOnlyCall) -> Result<(), Error> {
    call.check().map_err(Error::AppleOnly)
}

impl Texture {
    /// `protectionOptions` (SPI): [`DEFAULT_PROTECTION_OPTIONS`], the only protection options a
    /// runtime texture has.
    pub fn protection_options(&self) -> u64 {
        DEFAULT_PROTECTION_OPTIONS
    }
}

impl BlitCommandEncoder {
    /// `invalidateCompressedTexture:` (SPI): a no-op. Vulkan shows no texture compression, and
    /// after an invalidation Metal leaves the contents undefined, so keeping them is an outcome
    /// Metal allows. Refused (`Invalid`, `ForeignObject`) for another device's texture, as every
    /// blit is.
    pub fn invalidate_compressed_texture(&mut self, texture: &Texture) -> Result<(), Error> {
        self.logged(
            |_| BlitCall::InvalidateCompressedTexture(texture.clone()),
            |this| {
                this.core()
                    .check_device("invalidate_compressed_texture", texture.device())?;
                answer(AppleOnlyCall::InvalidateCompressedTexture)
            },
        )
    }

    /// `invalidateCompressedTexture:slice:level:` (SPI): a no-op, as
    /// [`invalidate_compressed_texture`](Self::invalidate_compressed_texture) is, for one slice
    /// and level. Refused (`Invalid`) also for a slice or level the texture does not have, as the
    /// blit encoder's hints of a slice and level are.
    pub fn invalidate_compressed_texture_slice(
        &mut self,
        texture: &Texture,
        slice: u64,
        level: u64,
    ) -> Result<(), Error> {
        self.logged(
            |_| BlitCall::InvalidateCompressedTextureSlice(texture.clone(), slice, level),
            |this| {
                const CALL: &str = "invalidate_compressed_texture_slice";
                this.core().check_device(CALL, texture.device())?;
                check_slice_level(CALL, texture, slice, level)?;
                answer(AppleOnlyCall::InvalidateCompressedTextureSlice)
            },
        )
    }
}

impl ComputeCommandEncoder {
    /// `insertCompressedTextureReinterpretationFlush` (SPI): a no-op. Vulkan shows no texture
    /// compression, so there is nothing to flush.
    pub fn insert_compressed_texture_reinterpretation_flush(&mut self) {}
}

impl RenderLowering {
    /// The lowering of [`RenderCommandEncoder::set_alpha_test_reference_value`].
    pub(crate) fn set_alpha_test_reference_value(&mut self, value: f32) -> Result<(), Error> {
        self.log(|_| RenderCall::AlphaTestReferenceValue(value));
        answer(AppleOnlyCall::SetAlphaTestReferenceValue(value))?;
        self.set_alpha_test_reference(value);
        Ok(())
    }

    /// The lowering of [`RenderCommandEncoder::set_line_width`].
    pub(crate) fn set_line_width(&mut self, width: f32) -> Result<(), Error> {
        self.log(|_| RenderCall::LineWidth(width));
        answer(AppleOnlyCall::SetLineWidth(width))?;
        self.set_encoder_line_width(width);
        Ok(())
    }

    /// The lowering of [`RenderCommandEncoder::set_point_size`].
    pub(crate) fn set_point_size(&mut self, size: f32) -> Result<(), Error> {
        self.log(|_| RenderCall::PointSize(size));
        answer(AppleOnlyCall::SetPointSize(size))
    }

    /// The lowering of [`RenderCommandEncoder::set_provoking_vertex_mode`].
    pub(crate) fn set_provoking_vertex_mode(&mut self, mode: u64) -> Result<(), Error> {
        self.log(|_| RenderCall::ProvokingVertexMode(mode));
        self.set_provoking_vertex(mode);
        Ok(())
    }

    /// The lowering of [`RenderCommandEncoder::set_viewport_transform_enabled`].
    pub(crate) fn set_viewport_transform_enabled(&mut self, enabled: bool) -> Result<(), Error> {
        self.log(|_| RenderCall::ViewportTransformEnabled(enabled));
        answer(AppleOnlyCall::SetViewportTransformEnabled(enabled))
    }

    /// The lowering of [`RenderCommandEncoder::set_vertex_amplification_mode`].
    pub(crate) fn set_vertex_amplification_mode(
        &mut self,
        mode: u64,
        value: u64,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::VertexAmplificationMode(mode, value));
        answer(AppleOnlyCall::SetVertexAmplificationMode { mode, value })?;
        self.amplify_with_kept_mappings(amplification_mode_count(mode, value))
            .ok_or(Error::AppleOnly(AppleOnly::VertexAmplificationMode))
    }

    /// The lowering of [`RenderCommandEncoder::set_depth_cleared`].
    pub(crate) fn set_depth_cleared(&mut self) -> Result<(), Error> {
        self.log(|_| RenderCall::DepthCleared);
        answer(AppleOnlyCall::SetDepthCleared)
    }

    /// The lowering of [`RenderCommandEncoder::set_stencil_cleared`].
    pub(crate) fn set_stencil_cleared(&mut self) -> Result<(), Error> {
        self.log(|_| RenderCall::StencilCleared);
        answer(AppleOnlyCall::SetStencilCleared)
    }
}

impl RenderCommandEncoder {
    /// `setAlphaTestReferenceValue:` (SPI): the reference a pipeline with the private alpha test
    /// (`PrivateRenderState::alpha_test_enabled`) compares its fragments' color 0 alpha against,
    /// both as Metal's 8-bit unorm codes; each encoder starts at Metal's 1. Accepted for every
    /// value, as on Metal.
    pub fn set_alpha_test_reference_value(&mut self, value: f32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_alpha_test_reference_value(value)),
            false => self.defer(|_| RenderCall::AlphaTestReferenceValue(value)),
        }
    }

    /// `setLineWidth:` (SPI): the width the encoder's lines are drawn at after the call, until the
    /// next call or the encoder's end, each encoder starting at 1, as Metal draws it (the M3 Max,
    /// macOS 27.0.1): `floor(16 width) / 16` clamped to `[0, 16]`, NaN and every width not
    /// above 0 as 0. At 1 a line is Metal's ordinary one; at any other width the parallelogram
    /// whose minor-axis edges pass through the endpoints, each that width long, a strip's
    /// segments each their own; at 0 as at 1/16. Points and filled
    /// triangles do not change. Every width is accepted. A draw at another width than 1 runs the
    /// device's own lines where they draw it (announced `device-wide-lines` where they differ
    /// from Metal's, or under `WideLinePolicy::Exact` the band below where they do), else the
    /// pipeline's band variant, two triangles over the parallelogram, which needs
    /// `tessellationShader`. What neither draws runs the device's own lines at the width nearest
    /// Metal's they draw (1 without `wideLines`), announced `lines-at-nearest-width`: a
    /// triangle's edges in fill mode `Lines` and a mesh draw where the device's lines do not draw
    /// the width, a line without `tessellationShader`, a line in a rasterization-rate-mapped
    /// pass or of an execution's command, and a band under vertex amplification above 1. A draw
    /// of no complete primitive draws nothing, as Metal's. Refused at the draw
    /// (`NotImplemented(WideLines)`): a strip's band whose list of segments the runtime cannot
    /// draw. The runtime's README states the route each device takes.
    pub fn set_line_width(&mut self, width: f32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_line_width(width)),
            false => self.defer(|_| RenderCall::LineWidth(width)),
        }
    }

    /// `setPointSize:` (SPI): a no-op. On an M4 it changes nothing Metal draws at any size (0,
    /// 0.5, 1, 3, 5, 100, -2, NaN, infinity), whether the vertex function writes `[[point_size]]`
    /// or not and whether the pipeline's private `pointSizeOutputVS` is set.
    pub fn set_point_size(&mut self, size: f32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_point_size(size)),
            false => self.defer(|_| RenderCall::PointSize(size)),
        }
    }

    /// `setProvokingVertexMode:` (SPI): the vertex the encoder's draws after the call take each
    /// primitive's flat inputs from, until the next call or the encoder's end, as Metal draws it
    /// (the M3 Max, macOS 27.0.1): mode 0, Metal's default, the first vertex; mode 1 the last, a
    /// triangle strip's odd triangle then in Vulkan's last-vertex order for its barycentric
    /// weights too. An execution of indirect render commands takes the first vertex whatever the
    /// mode, as on Metal. Mode 1 runs a pipeline of the last vertex where the device enables
    /// `VK_EXT_provoking_vertex`'s `provokingVertexLast` with `provokingVertexModePerPipeline`,
    /// and elsewhere draws each primitive whose vertex order a pipeline state reads from a list
    /// led by its last vertex, which a GPU pass writes (the CPU for a direct draw without
    /// indices); what that list draws otherwise than Metal is announced
    /// (`provoking-vertex-lines`, `provoking-vertex-unmatched`). Metal draws every other mode,
    /// its top bit set or not, as 2, a third way: a triangle list's, a fan's and the private
    /// triangle types' primitives as mode 1 (but a triangle of a pipeline that selects its slice
    /// or viewport from its middle vertex), a strip's odd triangle from its first vertex, every
    /// line's flat inputs and `[[primitive_id]]` 0, and patches as mode 0. The runtime draws it as
    /// mode 1, and patches as mode 0, announcing each draw whose pipeline state reads what Metal
    /// draws otherwise there (`provoking-vertex-unmatched`). No mode is refused.
    pub fn set_provoking_vertex_mode(&mut self, mode: u64) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_provoking_vertex_mode(mode)),
            false => self.defer(|_| RenderCall::ProvokingVertexMode(mode)),
        }
    }

    /// `setViewportTransformEnabled:` (SPI): accepted on, Metal's default and every runtime
    /// draw's; off is refused ([`AppleOnly::ViewportTransformDisabled`]): Vulkan has no form for
    /// it.
    pub fn set_viewport_transform_enabled(&mut self, enabled: bool) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_viewport_transform_enabled(enabled)),
            false => self.defer(|_| RenderCall::ViewportTransformEnabled(enabled)),
        }
    }

    /// `setVertexAmplificationMode:value:` (SPI): sets the amplification count
    /// [`set_vertex_amplification_count`](Self::set_vertex_amplification_count) sets, the number
    /// of `value`'s set bits in mode 1 and `value` in every other mode, and keeps the view mappings
    /// that call set (none at the encoder's start), as Metal does (the consumer's measurements on
    /// the M3 Max). Refused
    /// ([`AppleOnly::VertexAmplificationMode`]) for a count of 0, whose copies Metal draws change
    /// from run to run, above 8 (unmeasured but for three values), or above the view mappings kept
    /// (unmeasured); a refused call keeps the amplification set before.
    pub fn set_vertex_amplification_mode(&mut self, mode: u64, value: u64) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_vertex_amplification_mode(mode, value)),
            false => self.defer(|_| RenderCall::VertexAmplificationMode(mode, value)),
        }
    }

    /// `setDepthCleared` (SPI): a no-op. It changes nothing Metal draws (the consumer's suite case
    /// on the M3 Max, pixel for pixel).
    pub fn set_depth_cleared(&mut self) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_depth_cleared()),
            false => self.defer(|_| RenderCall::DepthCleared),
        }
    }

    /// `setStencilCleared` (SPI): a no-op, as [`set_depth_cleared`](Self::set_depth_cleared) is.
    pub fn set_stencil_cleared(&mut self) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_stencil_cleared()),
            false => self.defer(|_| RenderCall::StencilCleared),
        }
    }
}
