//! Render pass descriptors: what a render command encoder renders into
//! ([`RenderPassDescriptor`]), checked against its textures into the one dynamic rendering the
//! encoder begins ([`Pass`]).
//!
//! Each attachment is one level and slice of its texture, viewed through an image view of that
//! level and layer with the identity swizzle, which the texture keeps. Its load actions are
//! Vulkan's of the same name: `Load`, `Clear` and `DontCare`. Every store action the runtime builds
//! stores: a pass may end its rendering before the encoder ends and begin it again, loading what it
//! stored (a memory barrier after a draw that wrote memory, or a copy a draw needs first), so a
//! `DontCare` store keeps the texels too, one of the outcomes Metal allows. An `Unknown` one, which
//! the encoder sets later, stores too, as Vulkan fixes the store as the rendering begins: on a
//! tile-based device that stores a target Metal would have discarded.
//!
//! A pass has two areas. Metal's render area (`Pass::area`) is the smallest attachment's level, cut
//! to `renderTargetWidth` and `renderTargetHeight` where they are set: every scissor, the default
//! viewport and scissor, the areas a draw reads of its own attachments and the shadows are cut to
//! it, so no draw rasterises past it. Vulkan's render area (`Pass::reach`), where the load, store
//! and resolve operations act, is Metal's rounded up to [`REACH`] pixels a side and cut to the
//! smallest attachment's level, in every pass, with no call, barrier or pass of its own. A Mac
//! clears that band too: Apple's page for the `Clear` load says "The GPU writes a value to every
//! pixel in the attachment at the start of the render pass", and on an M3 Max and an M4 a `Clear`
//! pass that draws cleared its area rounded up to 32 pixels a side, clipped to the attachment. So a
//! `Clear` pass's band holds its clear value, a `Load` keeps the band, and a `DontCare` leaves it
//! undefined, as Metal allows any value there. A resolving pass resolves the band too, from the
//! samples its load left there. A pass whose attachments differ in size reaches the smallest level
//! only.
//!
//! A store action that resolves (`MultisampleResolve`, `StoreAndMultisampleResolve`, and
//! `Unknown` where the attachment names a resolve texture) resolves in the same rendering, by
//! Vulkan's own resolve: the attachment's `VkRenderingAttachmentInfo` names a view of the resolve
//! texture's `resolveLevel` and `resolveSlice` and a resolve mode. A color attachment resolves by
//! `AVERAGE`, the device's average ([`PrecisionNote::DeviceRoundedResolve`]); a depth attachment
//! by its [`MultisampleDepthResolveFilter`]'s mode and a stencil attachment by its
//! [`MultisampleStencilResolveFilter`]'s, where the device's resolve modes offer them. Either
//! action stores the multisample texels too, as a `DontCare` store does: a rendering the encoder
//! ends early and begins again resolves again as it ends, and its last resolve holds the pass's
//! whole result.
//!
//! A surface plane that renders as a linear colour attachment alone ([`Attachment::linear`]:
//! `VK_NV_linear_color_attachment`'s `LINEAR` image, `crate::surface`) renders only in a pass whose
//! colour attachments are all such planes, with no depth or stencil attachment and no resolve
//! into one; any other pass over one is refused as it begins ([`Gap::LinearPlanePass`], decided
//! by [`linear_shape`] from one mark an attachment).

use crate::buffer::{Buffer, BufferObject};
use crate::device::Core;
use crate::error::{Error, Gap, LinearPlanePassKind, Refusal, RenderPassKind};
use crate::texture::{Attachment, Texture, TextureObject};
use ash::vk;
use metal2vulkan::metal::format::{self as formats, FormatEmulation};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{PrecisionNote, SampledType};
use std::sync::Arc;

/// The color attachments a pass has (`MTLRenderPassDescriptor.colorAttachments`).
pub const MAX_COLOR_ATTACHMENTS: usize = 8;

/// `MTLLoadAction`: what an attachment holds as its pass starts.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LoadAction {
    /// `MTLLoadActionDontCare`: texels the pass does not read, `VK_ATTACHMENT_LOAD_OP_DONT_CARE`.
    #[default]
    DontCare,
    /// `MTLLoadActionLoad`: the attachment's texels, `VK_ATTACHMENT_LOAD_OP_LOAD`.
    Load,
    /// `MTLLoadActionClear`: the attachment's clear value, `VK_ATTACHMENT_LOAD_OP_CLEAR`.
    Clear,
}

/// `MTLVisibilityResultType`: what a pass's visibility results do with the words of the
/// visibility result buffer they are written to.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VisibilityResultType {
    /// `MTLVisibilityResultTypeReset`, Metal's default: each offset a mode of the encoder takes
    /// is written its result; every other word keeps its bytes, as an Apple M3 Max's do (macOS
    /// 27.0.1).
    #[default]
    Reset,
    /// `MTLVisibilityResultTypeAccumulate`: each result is added to its word. Not built: a
    /// `Boolean` or `Counting` mode in such a pass is
    /// [`DrawKind::AccumulatedVisibilityResults`](crate::DrawKind::AccumulatedVisibilityResults);
    /// a pass whose encoder sets none leaves the buffer as it was, as Metal's does.
    Accumulate,
}

/// `MTLStoreAction`: what an attachment keeps as its pass ends.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum StoreAction {
    /// `MTLStoreActionDontCare`: the attachment's texels are undefined after the pass. The
    /// runtime keeps them, `VK_ATTACHMENT_STORE_OP_STORE`: a pass that ends its rendering early
    /// and begins it again loads what it stored.
    DontCare,
    /// `MTLStoreActionStore`: the pass's texels are kept, `VK_ATTACHMENT_STORE_OP_STORE`.
    #[default]
    Store,
    /// `MTLStoreActionMultisampleResolve`: the pass resolves the attachment into its resolve
    /// texture as its rendering ends (the module's documentation), and keeps the multisample
    /// texels too, `VK_ATTACHMENT_STORE_OP_STORE`, as `DontCare` does. Refused with no resolve
    /// texture, [`Refusal::ResolveTexture`].
    MultisampleResolve,
    /// `MTLStoreActionStoreAndMultisampleResolve`: the pass keeps the texels and resolves them, as
    /// `MultisampleResolve` does.
    StoreAndMultisampleResolve,
    /// `MTLStoreActionUnknown`, left to the encoder to decide before it ends
    /// ([`RenderCommandEncoder::set_color_store_action`](crate::RenderCommandEncoder::set_color_store_action)
    /// and its depth and stencil siblings): `VK_ATTACHMENT_STORE_OP_STORE`, as Vulkan fixes the
    /// store as the rendering begins. A `DontCare` set later keeps the texels, as `DontCare` does;
    /// on a tile-based device this stores a target Metal would have discarded. Where the
    /// attachment names a resolve texture the rendering resolves into it, as Vulkan fixes the
    /// resolve as the rendering begins too: the encoder may then set only an action that resolves
    /// ([`RenderPassKind::LateStore`]).
    Unknown,
    /// `MTLStoreActionCustomSampleDepthStore`: refused, [`Gap::RenderPass`] of
    /// [`RenderPassKind::StoreAction`].
    CustomSampleDepthStore,
}

/// `MTLMultisampleDepthResolveFilter`: the depth a resolve of a multisample depth attachment
/// keeps of each pixel's samples, by the Vulkan resolve mode of the same choice. Sample 0 is on
/// every device (`VK_RESOLVE_MODE_SAMPLE_ZERO_BIT`, which Vulkan requires); `Min` and `Max` where
/// the device's `supportedDepthResolveModes` has `VK_RESOLVE_MODE_MIN_BIT` or
/// `VK_RESOLVE_MODE_MAX_BIT`, else [`Gap::RenderPass`] of [`RenderPassKind::ResolveFilter`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MultisampleDepthResolveFilter {
    /// `MTLMultisampleDepthResolveFilterSample0`: sample 0's depth.
    #[default]
    Sample0,
    /// `MTLMultisampleDepthResolveFilterMin`: the least of the samples' depths.
    Min,
    /// `MTLMultisampleDepthResolveFilterMax`: the greatest of the samples' depths.
    Max,
}

/// `MTLMultisampleStencilResolveFilter`: the stencil a resolve of a multisample stencil
/// attachment keeps of each pixel's samples.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MultisampleStencilResolveFilter {
    /// `MTLMultisampleStencilResolveFilterSample0`: sample 0's stencil,
    /// `VK_RESOLVE_MODE_SAMPLE_ZERO_BIT`, which Vulkan requires of every device.
    #[default]
    Sample0,
    /// `MTLMultisampleStencilResolveFilterDepthResolvedSample`: the stencil of the sample whose
    /// depth the depth resolve keeps. No Vulkan resolve mode picks it: refused, [`Gap::RenderPass`]
    /// of [`RenderPassKind::ResolveFilter`].
    DepthResolvedSample,
}

/// `MTLClearColor`: the color a `Clear` load action writes, in the attachment format's channels
/// (red, green, blue, alpha).
///
/// A float, normalized or sRGB channel takes the value as a `float`, which the device converts to
/// the format by its own rules (Vulkan clamps a normalized value and asks the device to round to
/// nearest; Metal rounds to nearest even); an integer channel takes it truncated toward zero and
/// saturated to the channel's range, NaN 0, as Metal does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClearColor {
    /// `red`.
    pub red: f64,
    /// `green`.
    pub green: f64,
    /// `blue`.
    pub blue: f64,
    /// `alpha`.
    pub alpha: f64,
}

impl Default for ClearColor {
    /// Metal's default: opaque black.
    fn default() -> ClearColor {
        ClearColor::new(0.0, 0.0, 0.0, 1.0)
    }
}

impl ClearColor {
    /// `MTLClearColorMake(red, green, blue, alpha)`.
    pub fn new(red: f64, green: f64, blue: f64, alpha: f64) -> ClearColor {
        ClearColor {
            red,
            green,
            blue,
            alpha,
        }
    }
}

/// `MTLRenderPassColorAttachmentDescriptor`, with `MTLRenderPassAttachmentDescriptor`'s
/// properties.
///
/// A store action that resolves resolves into `resolve_texture` by the device's average (the
/// module's documentation). Refused, as [`Gap::RenderPass`] of [`RenderPassKind::Resolve`]: a
/// resolve of an extended-range, bfloat or YCbCr format, which Vulkan's average would take in the
/// image's own encoding. An integer format has no resolve, [`Refusal::ResolveFormat`].
///
/// Not built: a 3D texture's `depthPlane` (refused by the texture's type) and
/// `storeActionOptions`, whose one option needs more than one sample.
///
/// Its default is Metal's: no texture or resolve texture, levels, slices and planes 0, `DontCare`
/// load, `Store` store, opaque black clear.
#[derive(Clone, Default)]
pub struct RenderPassColorAttachmentDescriptor {
    /// `texture`: the texture rendered into; `None` for no attachment at this index. It needs
    /// `RenderTarget` usage and a color format; a 2D texture, a 2D array, a cube, a cube array, or
    /// a 2D multisample texture or array.
    pub texture: Option<Texture>,
    /// `level`: the mipmap level rendered into.
    pub level: u32,
    /// `slice`: the array slice or cube face rendered into, the first of a layered pass's
    /// ([`RenderPassDescriptor::render_target_array_length`]).
    pub slice: u32,
    /// `loadAction`: `DontCare` by default.
    pub load_action: LoadAction,
    /// `storeAction`: `Store` by default.
    pub store_action: StoreAction,
    /// `clearColor`: opaque black by default.
    pub clear_color: ClearColor,
    /// `resolveTexture`: the texture a store action that resolves resolves into; `None` by
    /// default, and read only where the store action resolves, as Metal reads it. It takes the
    /// attachment's pixel format and one sample, and its level the attachment level's width and
    /// height; another is [`Gap::RenderPass`] of [`RenderPassKind::Resolve`]. Metal resolves into
    /// a texture of any usage, and so does the runtime into a colour texture whose format the
    /// device's optimal image renders; into one made without `RenderTarget` usage whose image has
    /// no attachment usage, such as a texture over a buffer in place, it is [`Gap::RenderPass`]
    /// of [`RenderPassKind::ResolveUsage`].
    pub resolve_texture: Option<Texture>,
    /// `resolveLevel`: the resolve texture's mipmap level resolved into.
    pub resolve_level: u32,
    /// `resolveSlice`: the resolve texture's array slice or cube face resolved into, the first of a
    /// layered pass's.
    pub resolve_slice: u32,
    /// `resolveDepthPlane`: a 3D resolve texture's depth plane. A plane but 0 is
    /// [`Gap::RenderPass`] of [`RenderPassKind::TextureType`], as a 3D attachment is.
    pub resolve_depth_plane: u32,
}

/// `MTLRenderPassDepthAttachmentDescriptor`.
///
/// Loaded, cleared and stored as a color attachment is, an optimal image in its real layout, of
/// the Vulkan format the translator's format table chooses by the device's format query; draws
/// test and write it as their depth-stencil state says. With a stencil attachment beside it,
/// both are one texture's same level and slice, a format of depth and stencil, as Vulkan's
/// rendering takes them; two others are refused, [`Gap::RenderPass`] of
/// [`RenderPassKind::DepthStencilViews`]. A `Clear` load's depth lies in `[0, 1]`; another is
/// refused, [`RenderPassKind::ClearDepth`].
///
/// A store action that resolves resolves into `resolve_texture` by `depth_resolve_filter`, as a
/// color attachment's resolves (the module's documentation).
///
/// Not built: a 3D texture's `depthPlane`, as a color attachment's.
#[derive(Clone)]
pub struct RenderPassDepthAttachmentDescriptor {
    /// `texture`: the texture whose depth is rendered into; `None` for no depth attachment. It
    /// needs `RenderTarget` usage, a format with depth and one sample.
    pub texture: Option<Texture>,
    /// `level`: the mipmap level rendered into.
    pub level: u32,
    /// `slice`: the array slice or cube face rendered into, the first of a layered pass's
    /// ([`RenderPassDescriptor::render_target_array_length`]).
    pub slice: u32,
    /// `loadAction`: `Clear` by default.
    pub load_action: LoadAction,
    /// `storeAction`: `DontCare` by default.
    pub store_action: StoreAction,
    /// `clearDepth`: 1 by default.
    pub clear_depth: f64,
    /// `resolveTexture`, as a color attachment's.
    pub resolve_texture: Option<Texture>,
    /// `resolveLevel`, as a color attachment's.
    pub resolve_level: u32,
    /// `resolveSlice`, as a color attachment's.
    pub resolve_slice: u32,
    /// `resolveDepthPlane`, as a color attachment's.
    pub resolve_depth_plane: u32,
    /// `depthResolveFilter`: `Sample0` by default.
    pub depth_resolve_filter: MultisampleDepthResolveFilter,
}

impl Default for RenderPassDepthAttachmentDescriptor {
    /// Metal's defaults: no texture or resolve texture, levels, slices and planes 0, `Clear` load,
    /// `DontCare` store, a clear depth of 1, the `Sample0` resolve filter.
    fn default() -> RenderPassDepthAttachmentDescriptor {
        RenderPassDepthAttachmentDescriptor {
            texture: None,
            level: 0,
            slice: 0,
            load_action: LoadAction::Clear,
            store_action: StoreAction::DontCare,
            clear_depth: 1.0,
            resolve_texture: None,
            resolve_level: 0,
            resolve_slice: 0,
            resolve_depth_plane: 0,
            depth_resolve_filter: MultisampleDepthResolveFilter::Sample0,
        }
    }
}

/// `MTLRenderPassStencilAttachmentDescriptor`.
///
/// A store action that resolves resolves into `resolve_texture` by `stencil_resolve_filter`, as
/// a color attachment's resolves (the module's documentation). With a depth attachment that
/// resolves too, both resolve into one view (`VUID-VkRenderingInfo-pDepthAttachment-06086`), else
/// [`Gap::RenderPass`] of [`RenderPassKind::Resolve`].
///
/// Not built: a 3D texture's `depthPlane`, as a color attachment's.
#[derive(Clone)]
pub struct RenderPassStencilAttachmentDescriptor {
    /// `texture`: the texture whose stencil is rendered into; `None` for no stencil attachment. It
    /// needs `RenderTarget` usage, a format with stencil and one sample.
    pub texture: Option<Texture>,
    /// `level`: the mipmap level rendered into.
    pub level: u32,
    /// `slice`: the array slice or cube face rendered into, the first of a layered pass's
    /// ([`RenderPassDescriptor::render_target_array_length`]).
    pub slice: u32,
    /// `loadAction`: `Clear` by default.
    pub load_action: LoadAction,
    /// `storeAction`: `DontCare` by default.
    pub store_action: StoreAction,
    /// `clearStencil`: 0 by default; its low 8 bits are the stencil's, as Metal keeps them.
    pub clear_stencil: u32,
    /// `resolveTexture`, as a color attachment's.
    pub resolve_texture: Option<Texture>,
    /// `resolveLevel`, as a color attachment's.
    pub resolve_level: u32,
    /// `resolveSlice`, as a color attachment's.
    pub resolve_slice: u32,
    /// `resolveDepthPlane`, as a color attachment's.
    pub resolve_depth_plane: u32,
    /// `stencilResolveFilter`: `Sample0` by default.
    pub stencil_resolve_filter: MultisampleStencilResolveFilter,
}

impl Default for RenderPassStencilAttachmentDescriptor {
    /// Metal's defaults: no texture or resolve texture, levels, slices and planes 0, `Clear` load,
    /// `DontCare` store, a clear stencil of 0, the `Sample0` resolve filter.
    fn default() -> RenderPassStencilAttachmentDescriptor {
        RenderPassStencilAttachmentDescriptor {
            texture: None,
            level: 0,
            slice: 0,
            load_action: LoadAction::Clear,
            store_action: StoreAction::DontCare,
            clear_stencil: 0,
            resolve_texture: None,
            resolve_level: 0,
            resolve_slice: 0,
            resolve_depth_plane: 0,
            stencil_resolve_filter: MultisampleStencilResolveFilter::Sample0,
        }
    }
}

/// `MTLRenderPassDescriptor`: the attachments a render command encoder renders into
/// ([`CommandBuffer::render_command_encoder`](crate::CommandBuffer::render_command_encoder)).
///
/// Not built, and so not fields: a rasterization rate map, tile sizes, imageblock and threadgroup
/// memory lengths and sample buffer attachments.
#[derive(Clone, Default)]
pub struct RenderPassDescriptor {
    /// `colorAttachments`.
    pub color_attachments: [RenderPassColorAttachmentDescriptor; MAX_COLOR_ATTACHMENTS],
    /// `depthAttachment`.
    pub depth_attachment: RenderPassDepthAttachmentDescriptor,
    /// `stencilAttachment`.
    pub stencil_attachment: RenderPassStencilAttachmentDescriptor,
    /// `renderTargetWidth`: where nonzero, the width of Metal's render area, cut to the smallest
    /// attachment's. The pass's draws are cut to that area; its load, store and resolve act on it
    /// rounded up to 32 pixels and cut to the smallest attachment, as a Mac's `Clear` reaches. A
    /// pass with no attachment is that wide.
    pub render_target_width: u32,
    /// `renderTargetHeight`: where nonzero, the height of Metal's render area, cut to the smallest
    /// attachment's, rounded up as the width is for the load, store and resolve. A pass with no
    /// attachment is that tall.
    pub render_target_height: u32,
    /// `defaultRasterSampleCount`: the samples of a pass with no attachment, 0 by default as in
    /// Metal, where such a pass is refused. It draws with pipeline states of that
    /// `raster_sample_count`
    /// ([`DrawKind::AttachmentFormats`](crate::DrawKind::AttachmentFormats) otherwise), which
    /// reach their answers through memory alone. A pass with an attachment takes its
    /// attachments' samples and reads none of this, as Metal's does.
    pub default_raster_sample_count: u32,
    /// `renderTargetArrayLength`: the layers the pass renders, 0 by default. At 2 or more every
    /// attachment and every resolve texture renders that many slices from its own `slice` (a
    /// cube's faces are slices), the pass's load, store and resolve act on each of them, and a
    /// vertex function's `[[render_target_array_index]]` `i` renders slice `slice + i`, an index at
    /// or past the length rendering the first; at 0 and 1 the pass renders the attachment's slice
    /// alone, whatever index the vertex function writes, as Metal's does. A fragment function reads
    /// the index as the vertex function wrote it. An attachment or resolve texture with fewer
    /// slices from its own is refused, [`Refusal::AttachmentPlace`] naming the last slice the pass
    /// would render; a length past the device's `maxFramebufferLayers`, [`Gap::RenderPass`] of
    /// [`RenderPassKind::ArrayLength`].
    pub render_target_array_length: u32,
    /// `visibilityResultBuffer`: `None` by default. The command buffer keeps it until it retires.
    /// The pass writes the 8 bytes at each offset a `Boolean` or `Counting` mode of its encoder
    /// takes
    /// ([`RenderCommandEncoder::set_visibility_result_mode`](crate::RenderCommandEncoder::set_visibility_result_mode)),
    /// after its rendering and before its second barrier, ordered as a blit's write is; every other
    /// byte keeps its value, as Metal's pass keeps it under either `visibilityResultType` where its
    /// encoder enables no mode (an Apple M3 Max, macOS 27.0.1).
    pub visibility_result_buffer: Option<Buffer>,
    /// `visibilityResultType`: [`VisibilityResultType::Reset`] by default.
    pub visibility_result_type: VisibilityResultType,
}

impl RenderPassDescriptor {
    /// The static note of the pass's color resolves: [`PrecisionNote::DeviceRoundedResolve`] where
    /// a color attachment's store action resolves (or is `Unknown` with a resolve texture), as
    /// Vulkan's `AVERAGE` leaves the average's rounding, and an sRGB format's conversion, to the
    /// device. `None` where no color attachment resolves: a depth or stencil resolve keeps one
    /// sample's value, or the least or greatest, exactly.
    pub fn precision_note(&self) -> Option<PrecisionNote> {
        let resolves = |c: &RenderPassColorAttachmentDescriptor| {
            c.texture.is_some()
                && matches!(
                    resolving(c.store_action, c.resolve_texture.is_some()),
                    Ok(true)
                )
        };
        (self.color_attachments.iter().any(resolves)).then_some(PrecisionNote::DeviceRoundedResolve)
    }
}

/// The device's depth and stencil resolve modes
/// (`VkPhysicalDeviceVulkan12Properties`, Vulkan 1.2's depth-stencil resolve properties).
#[derive(Clone, Copy)]
pub(crate) struct ResolveModes {
    /// `supportedDepthResolveModes`.
    pub(crate) depth: vk::ResolveModeFlags,
    /// `supportedStencilResolveModes`.
    pub(crate) stencil: vk::ResolveModeFlags,
    /// `independentResolveNone`: one aspect of a depth-stencil attachment may resolve while the
    /// other does not.
    pub(crate) independent_none: bool,
    /// `independentResolve`: the two aspects may resolve by different modes.
    pub(crate) independent: bool,
}

/// The resolve of an attachment of a checked pass: a view of its resolve texture's level and
/// the pass's layers from its slice, and the mode Vulkan resolves by.
pub(crate) struct Resolve {
    pub(crate) texture: Texture,
    pub(crate) attachment: Attachment,
    /// The layout the pass resolves into it in: its home, or the attachment layout for an image
    /// whose home is shader-read-only, which the pass moves there and back.
    pub(crate) layout: vk::ImageLayout,
    pub(crate) mode: vk::ResolveModeFlags,
}

impl Resolve {
    /// Whether the pass moves its image out of home and back.
    pub(crate) fn moves(&self) -> bool {
        self.layout != self.attachment.home
    }

    /// The barrier that moves its level and layers, every aspect, from `old` to `new`, after
    /// every earlier command and before every later one.
    pub(crate) fn barrier(
        &self,
        old: vk::ImageLayout,
        new: vk::ImageLayout,
    ) -> vk::ImageMemoryBarrier2<'static> {
        let a = &self.attachment;
        layout_barrier(a.image, range(a), old, new)
    }
}

/// The level and layers of `a`, every aspect.
fn range(a: &Attachment) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: a.aspects,
        base_mip_level: a.level,
        level_count: 1,
        base_array_layer: a.layer,
        layer_count: a.layers,
    }
}

/// One attachment of a checked pass.
pub(crate) struct Target {
    pub(crate) texture: Texture,
    pub(crate) attachment: Attachment,
    /// The layout the pass renders it in: its home, or the attachment layout for an image whose
    /// home is shader-read-only, which the pass moves there and back; the feedback layout once a
    /// draw of the pass read it through a texture (`crate::render_encoder`).
    pub(crate) layout: vk::ImageLayout,
    load: vk::AttachmentLoadOp,
    store: vk::AttachmentStoreOp,
    clear: vk::ClearValue,
    /// Where its store action resolves, its resolve, boxed: a pass that resolves nothing moves
    /// no resolve's room.
    pub(crate) resolve: Option<Box<Resolve>>,
}

impl Target {
    /// Whether the pass moves its image out of home and back.
    pub(crate) fn moves(&self) -> bool {
        self.layout != self.attachment.home
    }

    /// The colour the pass clears it to, as its rendering's load clears it: `None` for an
    /// attachment the pass loads or does not care for, and for the stencil attachment.
    pub(crate) fn clear_color(&self) -> Option<vk::ClearColorValue> {
        let color = self.attachment.aspects == vk::ImageAspectFlags::COLOR;
        if !color || self.load != vk::AttachmentLoadOp::CLEAR {
            return None;
        }
        // SAFETY: a colour attachment's clear value is its colour (`Pass::check`).
        Some(unsafe { self.clear.color })
    }

    /// Its `VkRenderingAttachmentInfo`; `resumed`, for a rendering the pass begins again, which
    /// loads what the pass stored.
    /// Where it resolves, the rendering resolves it as it ends, each time it begins.
    pub(crate) fn info(&self, resumed: bool) -> vk::RenderingAttachmentInfo<'static> {
        let info = vk::RenderingAttachmentInfo::default()
            .image_view(self.attachment.view)
            .image_layout(self.layout)
            .load_op(match resumed {
                true => vk::AttachmentLoadOp::LOAD,
                false => self.load,
            })
            .store_op(self.store)
            .clear_value(self.clear);
        match &self.resolve {
            Some(r) => info
                .resolve_mode(r.mode)
                .resolve_image_view(r.attachment.view)
                .resolve_image_layout(r.layout),
            None => info,
        }
    }

    /// The barrier that moves its level and layers, every aspect, from `old` to `new`, after
    /// every earlier command and before every later one.
    pub(crate) fn barrier(
        &self,
        old: vk::ImageLayout,
        new: vk::ImageLayout,
    ) -> vk::ImageMemoryBarrier2<'static> {
        let a = &self.attachment;
        layout_barrier(a.image, self.range(), old, new)
    }

    /// Its level and layers, every aspect.
    pub(crate) fn range(&self) -> vk::ImageSubresourceRange {
        range(&self.attachment)
    }
}

/// The barrier that moves `range` of `image` from `old` to `new`, after every earlier command and
/// before every later one.
pub(crate) fn layout_barrier(
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range)
}

/// The granularity, in pixels a side, to which a pass's Vulkan render area rounds Metal's area up
/// (`Pass::reach`).
pub(crate) const REACH: u32 = 32;

/// A pass descriptor checked against its textures: its attachments, by index for the colors,
/// and its two areas.
pub(crate) struct Pass {
    /// Up to the last color attachment with a texture.
    pub(crate) colors: Vec<Option<Target>>,
    /// The depth attachment and the stencil attachment: one view of one image where both are set
    /// (`Pass::check`), each with its own load, store and clear.
    pub(crate) depth: Option<Target>,
    pub(crate) stencil: Option<Target>,
    /// Metal's area: the smallest attachment's level, cut to `renderTargetWidth` and
    /// `renderTargetHeight` where they are set. Every scissor, the default viewport and scissor,
    /// the areas a draw fetches and the shadows are cut to it.
    pub(crate) area: [u32; 2],
    /// The Vulkan render area, where the load, store and resolve operations act: `area` rounded up
    /// to [`REACH`] pixels a side, cut to the smallest attachment's level.
    pub(crate) reach: [u32; 2],
    /// Every attachment's samples, one bit: the pass refuses attachments of two counts.
    pub(crate) samples: vk::SampleCountFlags,
    /// The layers every attachment and resolve renders from its own slice:
    /// `renderTargetArrayLength`, at least 1. The rendering's `layerCount`, and the length each
    /// draw pushes to a vertex function that writes `[[render_target_array_index]]`
    /// (`PushField::RenderTargetArrayLength`), which stores `Layer` 0 for an index at or past it.
    pub(crate) layers: u32,
    /// Whether any attachment resolves: a pass that resolves nothing walks no resolve as its
    /// encoder begins and ends ([`Pass::resolves`]).
    pub(crate) resolving: bool,
}

#[cfg(test)]
thread_local! {
    /// The walks of a pass's resolves ([`Pass::resolves`]) on this thread, which a test reads.
    pub(crate) static RESOLVE_WALKS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

impl Pass {
    /// Every attachment's image once, colors first: the stencil attachment's only where there is
    /// no depth attachment, whose image it otherwise is.
    pub(crate) fn targets(&self) -> impl Iterator<Item = &Target> {
        let stencil = self.stencil.as_ref().filter(|_| self.depth.is_none());
        (self.colors.iter().flatten())
            .chain(&self.depth)
            .chain(stencil)
    }

    /// Every attachment's resolve, colors first: the stencil attachment's only where the depth
    /// attachment does not resolve, whose resolve view it otherwise shares. Walked only where the
    /// pass is [`resolving`](Pass::resolving).
    pub(crate) fn resolves(&self) -> impl Iterator<Item = &Resolve> {
        #[cfg(test)]
        RESOLVE_WALKS.with(|walks| walks.set(walks.get() + 1));
        let depth = self.depth.as_ref().and_then(|t| t.resolve.as_deref());
        let stencil =
            (self.stencil.as_ref().and_then(|t| t.resolve.as_deref())).filter(|_| depth.is_none());
        (self.colors.iter().flatten())
            .filter_map(|t| t.resolve.as_deref())
            .chain(depth)
            .chain(stencil)
    }

    /// Records `vkCmdBeginRendering` of the pass into `command`; `resumed`, again after the
    /// encoder ended it early, loading what it stored.
    ///
    /// # Safety
    ///
    /// `command` records outside any rendering and keeps each attachment's texture until it
    /// retires; each attachment is in the layout the pass renders it in.
    pub(crate) unsafe fn begin(
        &self,
        device: &ash::Device,
        command: vk::CommandBuffer,
        resumed: bool,
    ) {
        let colors: Vec<vk::RenderingAttachmentInfo> = (self.colors.iter())
            .map(|c| {
                c.as_ref()
                    .map_or_else(Default::default, |t| t.info(resumed))
            })
            .collect();
        let depth = self.depth.as_ref().map(|t| t.info(resumed));
        let stencil = self.stencil.as_ref().map(|t| t.info(resumed));
        let [width, height] = self.reach;
        let mut info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            })
            .layer_count(self.layers)
            .color_attachments(&colors);
        if let Some(depth) = &depth {
            info = info.depth_attachment(depth);
        }
        if let Some(stencil) = &stencil {
            info = info.stencil_attachment(stencil);
        }
        // SAFETY: as the caller states; the render area (`reach`) lies inside every attachment,
        // and every draw's scissor inside Metal's area, which lies inside it; each attachment and
        // resolve is a view of one level and the pass's layers of its image, at most the device's
        // `maxFramebufferLayers` (`Pass::check`), and a depth and a stencil attachment both set are
        // one view.
        unsafe { device.cmd_begin_rendering(command, &info) };
    }

    /// Checks `d` on the device of `core`.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's texture or visibility result buffer;
    /// [`Refusal::AttachmentUsage`]; [`Refusal::AttachmentPlace`]; [`Refusal::AttachmentFormat`];
    /// [`Refusal::ResolveTexture`]; [`Refusal::ResolveFormat`]; [`Refusal::NoAttachmentPass`];
    /// [`Gap::RenderPass`];
    /// [`Gap::LinearPlanePass`] for a shape [`linear_shape`] names; [`Error::Vulkan`] where an
    /// attachment's view cannot be made.
    pub(crate) fn check(core: &Arc<Core>, d: &RenderPassDescriptor) -> Result<Pass, Error> {
        let gap = |kind| Error::Unsupported(Gap::RenderPass { kind });
        // Metal's 0 and 1 both render the attachment's slice alone.
        let layers = d.render_target_array_length.max(1);
        if layers > core.limits.max_framebuffer_layers {
            return Err(gap(RenderPassKind::ArrayLength));
        }
        let mine = |texture: &Texture| match texture.is_of(core) {
            true => Ok(()),
            false => Err(Error::Refused(Refusal::OtherDevice)),
        };
        if let Some(buffer) = &d.visibility_result_buffer {
            if !buffer.is_of(core) {
                return Err(Error::Refused(Refusal::OtherDevice));
            }
        }
        let mut colors = Vec::new();
        let mut any_resolve = false;
        for (index, c) in d.color_attachments.iter().enumerate() {
            let Some(texture) = &c.texture else {
                continue;
            };
            mine(texture)?;
            let pf = texture.descriptor().pixel_format;
            if formats::is_depth_stencil(pf) {
                return Err(Error::Refused(Refusal::AttachmentFormat {
                    pixel_format: pf,
                }));
            }
            let attachment = texture.attachment(c.level, c.slice, layers)?;
            let resolve = match (
                resolving(c.store_action, c.resolve_texture.is_some())?,
                &c.resolve_texture,
            ) {
                (true, Some(into)) => {
                    // Metal resolves no integer format; an emulated one would average the
                    // image's own encoding.
                    if !matches!(formats::sampled_type(pf), Some(SampledType::Float)) {
                        return Err(Error::Refused(Refusal::ResolveFormat { pixel_format: pf }));
                    }
                    if !matches!(
                        texture.emulation(),
                        FormatEmulation::Direct
                            | FormatEmulation::RotatedView
                            | FormatEmulation::AlphaInRed
                    ) {
                        return Err(gap(RenderPassKind::Resolve));
                    }
                    let into = ResolveInto {
                        texture: into,
                        level: c.resolve_level,
                        slice: c.resolve_slice,
                        depth_plane: c.resolve_depth_plane,
                    };
                    Some(Box::new(resolve(
                        core,
                        texture,
                        &attachment,
                        into,
                        vk::ResolveModeFlags::AVERAGE,
                    )?))
                }
                _ => None,
            };
            let clear = match c.load_action {
                LoadAction::Clear => vk::ClearValue {
                    color: clear_color(pf, texture.emulation(), c.clear_color)
                        .ok_or_else(|| gap(RenderPassKind::ClearFormat))?,
                },
                LoadAction::Load | LoadAction::DontCare => vk::ClearValue::default(),
            };
            any_resolve |= resolve.is_some();
            colors.resize_with(index + 1, || None);
            colors[index] = Some(target(
                texture,
                attachment,
                c.load_action,
                c.store_action,
                clear,
                resolve,
            )?);
        }
        let z = &d.depth_attachment;
        let depth = match &z.texture {
            Some(texture) => {
                mine(texture)?;
                let pf = texture.descriptor().pixel_format;
                let depth = metal2vulkan::vk::ImageAspectFlags::DEPTH;
                if !formats::aspects(pf).is_some_and(|aspects| aspects.contains(depth)) {
                    return Err(Error::Refused(Refusal::AttachmentFormat {
                        pixel_format: pf,
                    }));
                }
                let attachment = texture.attachment(z.level, z.slice, layers)?;
                let resolve = match (
                    resolving(z.store_action, z.resolve_texture.is_some())?,
                    &z.resolve_texture,
                ) {
                    (true, Some(into)) => {
                        let mode = match z.depth_resolve_filter {
                            MultisampleDepthResolveFilter::Sample0 => {
                                vk::ResolveModeFlags::SAMPLE_ZERO
                            }
                            MultisampleDepthResolveFilter::Min => vk::ResolveModeFlags::MIN,
                            MultisampleDepthResolveFilter::Max => vk::ResolveModeFlags::MAX,
                        };
                        if !core.resolve_modes.depth.contains(mode) {
                            return Err(gap(RenderPassKind::ResolveFilter));
                        }
                        let into = ResolveInto {
                            texture: into,
                            level: z.resolve_level,
                            slice: z.resolve_slice,
                            depth_plane: z.resolve_depth_plane,
                        };
                        Some(Box::new(resolve(core, texture, &attachment, into, mode)?))
                    }
                    _ => None,
                };
                let clear = match z.load_action {
                    LoadAction::Clear if !(0.0..=1.0).contains(&z.clear_depth) => {
                        return Err(gap(RenderPassKind::ClearDepth))
                    }
                    LoadAction::Clear => z.clear_depth as f32,
                    LoadAction::Load | LoadAction::DontCare => 0.0,
                };
                any_resolve |= resolve.is_some();
                let clear = vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: clear,
                        stencil: 0,
                    },
                };
                Some(target(
                    texture,
                    attachment,
                    z.load_action,
                    z.store_action,
                    clear,
                    resolve,
                )?)
            }
            None => None,
        };
        let s = &d.stencil_attachment;
        let stencil = match &s.texture {
            Some(texture) => {
                mine(texture)?;
                let pf = texture.descriptor().pixel_format;
                let stencil = metal2vulkan::vk::ImageAspectFlags::STENCIL;
                if !formats::aspects(pf).is_some_and(|aspects| aspects.contains(stencil)) {
                    return Err(Error::Refused(Refusal::AttachmentFormat {
                        pixel_format: pf,
                    }));
                }
                let attachment = texture.attachment(s.level, s.slice, layers)?;
                let resolve = match (
                    resolving(s.store_action, s.resolve_texture.is_some())?,
                    &s.resolve_texture,
                ) {
                    (true, Some(into)) => {
                        let mode = match s.stencil_resolve_filter {
                            MultisampleStencilResolveFilter::Sample0 => {
                                vk::ResolveModeFlags::SAMPLE_ZERO
                            }
                            MultisampleStencilResolveFilter::DepthResolvedSample => {
                                return Err(gap(RenderPassKind::ResolveFilter))
                            }
                        };
                        if !core.resolve_modes.stencil.contains(mode) {
                            return Err(gap(RenderPassKind::ResolveFilter));
                        }
                        let into = ResolveInto {
                            texture: into,
                            level: s.resolve_level,
                            slice: s.resolve_slice,
                            depth_plane: s.resolve_depth_plane,
                        };
                        Some(Box::new(resolve(core, texture, &attachment, into, mode)?))
                    }
                    _ => None,
                };
                let clear = vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: s.clear_stencil & 0xff,
                    },
                };
                any_resolve |= resolve.is_some();
                Some(target(
                    texture,
                    attachment,
                    s.load_action,
                    s.store_action,
                    clear,
                    resolve,
                )?)
            }
            None => None,
        };
        if let (Some(depth), Some(stencil)) = (&depth, &stencil) {
            if depth.attachment.view != stencil.attachment.view {
                return Err(gap(RenderPassKind::DepthStencilViews));
            }
            // One view's two aspects resolve as the device pairs them
            // (VUID-VkRenderingInfo-pDepthAttachment-06086, -06104 and -06105).
            let modes = &core.resolve_modes;
            match (&depth.resolve, &stencil.resolve) {
                _ if !any_resolve => {}
                (Some(d), Some(s)) if d.attachment.view != s.attachment.view => {
                    return Err(gap(RenderPassKind::Resolve))
                }
                (Some(d), Some(s)) if d.mode != s.mode && !modes.independent => {
                    return Err(gap(RenderPassKind::ResolveFilter))
                }
                (Some(_), None) | (None, Some(_)) if !modes.independent_none => {
                    return Err(gap(RenderPassKind::ResolveFilter))
                }
                _ => {}
            }
        }
        let mut pass = Pass {
            colors,
            depth,
            stencil,
            area: [u32::MAX; 2],
            reach: [u32::MAX; 2],
            samples: vk::SampleCountFlags::empty(),
            layers,
            resolving: any_resolve,
        };
        let colors = pass.colors.iter().flatten();
        let resolves = (any_resolve.then(|| colors.clone()).into_iter().flatten())
            .filter_map(|t| t.resolve.as_ref().map(|r| r.attachment.linear));
        let shape = linear_shape(
            colors.map(|t| t.attachment.linear),
            resolves,
            pass.depth.is_some() || pass.stencil.is_some(),
        );
        if let Some(kind) = shape {
            return Err(Error::Unsupported(Gap::LinearPlanePass { kind }));
        }
        // Vulkan renders one sample count into every attachment of a pass
        // (VUID-VkRenderingInfo-multisampledRenderToSingleSampled-06857).
        let mut samples = vk::SampleCountFlags::empty();
        for target in pass.targets() {
            let own = target.attachment.samples;
            if !samples.is_empty() && samples != own {
                return Err(gap(RenderPassKind::SampleCounts));
            }
            samples = own;
        }
        pass.samples = samples;
        let mut level = pass.targets().fold([u32::MAX; 2], |level, t| {
            let extent = t.attachment.extent;
            [level[0].min(extent[0]), level[1].min(extent[1])]
        });
        if level[0] == u32::MAX {
            // No attachment: Metal's area is `renderTargetWidth` by `renderTargetHeight`, and its
            // samples `defaultRasterSampleCount`, which a draw's pipeline state matches; Vulkan's
            // rendering of no attachment takes the pipeline's samples alone.
            let size = [d.render_target_width, d.render_target_height];
            let count = d.default_raster_sample_count;
            if size.contains(&0) || count == 0 {
                return Err(Error::Refused(Refusal::NoAttachmentPass {
                    width: size[0],
                    height: size[1],
                    samples: count,
                }));
            }
            // VUID-VkRenderingInfo-pNext-07815 and -07816.
            if (size.iter().zip(core.limits.max_framebuffer)).any(|(&side, max)| side > max) {
                return Err(gap(RenderPassKind::NoAttachment));
            }
            pass.samples = vk::SampleCountFlags::from_raw(count);
            level = size;
        }
        for (axis, set) in [d.render_target_width, d.render_target_height]
            .into_iter()
            .enumerate()
        {
            let area = match set {
                0 => level[axis],
                set => level[axis].min(set),
            };
            pass.area[axis] = area;
            pass.reach[axis] = area.next_multiple_of(REACH).min(level[axis]);
        }
        Ok(pass)
    }
}

/// What a pass sets that Vulkan's rule for images that render as linear colour attachments alone
/// does not render ([`Attachment::linear`], a surface plane's: `VK_NV_linear_color_attachment`
/// renders such a format "as long as all the color attachments in the render pass instance have
/// `VK_IMAGE_TILING_LINEAR` tiling"), from each colour attachment's mark (`colors`), each colour
/// resolve's (`resolves`) and whether the pass has a depth or a stencil attachment: a resolve
/// into a marked image, a marked colour attachment beside a depth or stencil attachment (the
/// driver lost the device on that shape), or beside an unmarked colour attachment. `None` where
/// nothing is marked, and where every colour attachment is marked with nothing else set.
pub(crate) fn linear_shape(
    colors: impl IntoIterator<Item = bool>,
    resolves: impl IntoIterator<Item = bool>,
    depth_or_stencil: bool,
) -> Option<LinearPlanePassKind> {
    if resolves.into_iter().any(|linear| linear) {
        return Some(LinearPlanePassKind::Resolve);
    }
    let (marked, unmarked) = colors
        .into_iter()
        .fold((false, false), |(marked, unmarked), linear| {
            (marked || linear, unmarked || !linear)
        });
    match (marked, depth_or_stencil, unmarked) {
        (false, _, _) => None,
        (true, true, _) => Some(LinearPlanePassKind::DepthStencil),
        (true, false, true) => Some(LinearPlanePassKind::OtherColor),
        (true, false, false) => None,
    }
}

/// The Vulkan load and store operations of Metal's `load` and `store` actions: the load one to
/// one, and `STORE` for `Store`, `DontCare` ([`StoreAction::DontCare`]), `Unknown`
/// ([`StoreAction::Unknown`]) and the two that resolve, whose resolve is the attachment's
/// ([`Target::info`]).
///
/// Err: [`Gap::RenderPass`] of [`RenderPassKind::StoreAction`].
pub(crate) fn ops(
    load: LoadAction,
    store: StoreAction,
) -> Result<(vk::AttachmentLoadOp, vk::AttachmentStoreOp), Error> {
    let load = match load {
        LoadAction::DontCare => vk::AttachmentLoadOp::DONT_CARE,
        LoadAction::Load => vk::AttachmentLoadOp::LOAD,
        LoadAction::Clear => vk::AttachmentLoadOp::CLEAR,
    };
    let gap = |kind| Err(Error::Unsupported(Gap::RenderPass { kind }));
    let store = match store {
        StoreAction::DontCare
        | StoreAction::Store
        | StoreAction::Unknown
        | StoreAction::MultisampleResolve
        | StoreAction::StoreAndMultisampleResolve => vk::AttachmentStoreOp::STORE,
        StoreAction::CustomSampleDepthStore => return gap(RenderPassKind::StoreAction),
    };
    Ok((load, store))
}

/// Whether an attachment of store action `store` resolves, `named` where it names a resolve
/// texture: an action that resolves, and `Unknown` with a resolve texture, which the encoder is
/// left to set to one (Vulkan fixes the resolve as the rendering begins).
///
/// Err: [`Refusal::ResolveTexture`] for an action that resolves with no resolve texture.
pub(crate) fn resolving(store: StoreAction, named: bool) -> Result<bool, Error> {
    let resolves = matches!(
        store,
        StoreAction::MultisampleResolve | StoreAction::StoreAndMultisampleResolve
    );
    match (resolves, named) {
        (true, false) => Err(Error::Refused(Refusal::ResolveTexture)),
        (true, true) => Ok(true),
        (false, named) => Ok(named && store == StoreAction::Unknown),
    }
}

/// Where an attachment resolves: the resolve texture and its level, slice and depth plane.
struct ResolveInto<'a> {
    texture: &'a Texture,
    level: u32,
    slice: u32,
    depth_plane: u32,
}

/// The resolve of `texture`'s attachment `attachment` into `into` by `mode`: a view of the
/// resolve texture's level and the attachment's layers from its slice, as an attachment's.
///
/// Err: [`Refusal::OtherDevice`] for another device's resolve texture; [`Gap::RenderPass`] of
/// [`RenderPassKind::Resolve`] for an attachment of one sample, and for a resolve texture of
/// another pixel format, of more than one sample, or whose level is not the attachment level's
/// width and height; [`Gap::RenderPass`] of [`RenderPassKind::TextureType`] for a depth plane
/// but 0; what the resolve texture's resolve attachment returns, [`Gap::RenderPass`] of
/// [`RenderPassKind::ResolveUsage`] for a resolve texture whose image has no attachment usage
/// among it.
fn resolve(
    core: &Arc<Core>,
    texture: &Texture,
    attachment: &Attachment,
    into: ResolveInto<'_>,
    mode: vk::ResolveModeFlags,
) -> Result<Resolve, Error> {
    let gap = |kind| Err(Error::Unsupported(Gap::RenderPass { kind }));
    if !into.texture.is_of(core) {
        return Err(Error::Refused(Refusal::OtherDevice));
    }
    let d = into.texture.descriptor();
    if attachment.samples == vk::SampleCountFlags::TYPE_1
        || d.sample_count != 1
        || d.pixel_format != texture.descriptor().pixel_format
    {
        return gap(RenderPassKind::Resolve);
    }
    let view = (into.texture).resolve_attachment(into.level, into.slice, attachment.layers)?;
    if into.depth_plane != 0 {
        return gap(RenderPassKind::TextureType);
    }
    if view.extent != attachment.extent || view.format != attachment.format {
        return gap(RenderPassKind::Resolve);
    }
    let layout = match view.home {
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => vk::ImageLayout::ATTACHMENT_OPTIMAL,
        home => home,
    };
    Ok(Resolve {
        texture: into.texture.clone(),
        attachment: view,
        layout,
        mode,
    })
}

/// One attachment of `texture`, viewed as `attachment`, with its load and store actions, its
/// clear value and its resolve.
///
/// Err: as [`ops`].
fn target(
    texture: &Texture,
    attachment: Attachment,
    load: LoadAction,
    store: StoreAction,
    clear: vk::ClearValue,
    resolve: Option<Box<Resolve>>,
) -> Result<Target, Error> {
    let (load, store) = ops(load, store)?;
    let layout = match attachment.home {
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => vk::ImageLayout::ATTACHMENT_OPTIMAL,
        home => home,
    };
    Ok(Target {
        texture: texture.clone(),
        attachment,
        layout,
        load,
        store,
        clear,
        resolve,
    })
}

/// The Vulkan clear value of `color` for a color attachment of `pf` held as `emulation`: an
/// integer channel's value truncated toward zero and saturated to its range, NaN 0; any other
/// channel's as a `float`, in the image's channels: a rotated format's, and `A8Unorm`'s held in
/// `R8_UNORM`, rotated (`to_rotated_image`), so its alpha lands in red. `None` where only a shader
/// clears the format: an extended-range, bfloat or YCbCr format.
pub(crate) fn clear_color(
    pf: PixelFormat,
    emulation: FormatEmulation,
    color: ClearColor,
) -> Option<vk::ClearColorValue> {
    let rgba = [color.red, color.green, color.blue, color.alpha];
    if !matches!(
        emulation,
        FormatEmulation::Direct | FormatEmulation::RotatedView | FormatEmulation::AlphaInRed
    ) {
        return None;
    }
    // A channel narrower than 32 bits: byte `c` of the word holds `32 - bits`.
    let word = formats::integer_channels_word(pf);
    let bits: [u32; 4] = std::array::from_fn(|c| 32 - ((word >> (8 * c)) & 0xff));
    match formats::sampled_type(pf)? {
        SampledType::Uint => Some(vk::ClearColorValue {
            uint32: std::array::from_fn(|c| {
                let max = ((1u64 << bits[c]) - 1) as f64;
                match rgba[c].is_nan() {
                    true => 0,
                    false => rgba[c].trunc().clamp(0.0, max) as u32,
                }
            }),
        }),
        SampledType::Sint => Some(vk::ClearColorValue {
            int32: std::array::from_fn(|c| {
                let half = (1u64 << (bits[c] - 1)) as f64;
                match rgba[c].is_nan() {
                    true => 0,
                    false => rgba[c].trunc().clamp(-half, half - 1.0) as i32,
                }
            }),
        }),
        SampledType::Float => {
            let float32 = rgba.map(|x| x as f32);
            Some(vk::ClearColorValue {
                float32: match emulation {
                    FormatEmulation::RotatedView | FormatEmulation::AlphaInRed => {
                        formats::to_rotated_image(float32)
                    }
                    _ => float32,
                },
            })
        }
        _ => None,
    }
}
