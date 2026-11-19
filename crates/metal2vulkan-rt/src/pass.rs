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
//! tile-based device that stores a target Metal would have discarded. The render area is the
//! smallest attachment's level, cut to `renderTargetWidth` and `renderTargetHeight` where they are
//! set.

use crate::buffer::{Buffer, BufferObject};
use crate::device::Core;
use crate::error::{Error, Gap, Refusal, RenderPassKind};
use crate::texture::{Attachment, Texture, TextureObject};
use ash::vk;
use metal2vulkan::metal::format::{self as formats, FormatEmulation};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::SampledType;
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
    /// `MTLStoreActionMultisampleResolve`: refused, [`Gap::RenderPass`] of
    /// [`RenderPassKind::Resolve`].
    MultisampleResolve,
    /// `MTLStoreActionStoreAndMultisampleResolve`: refused, as `MultisampleResolve`.
    StoreAndMultisampleResolve,
    /// `MTLStoreActionUnknown`, left to the encoder to decide before it ends
    /// ([`RenderCommandEncoder::set_color_store_action`](crate::RenderCommandEncoder::set_color_store_action)
    /// and its depth and stencil siblings): `VK_ATTACHMENT_STORE_OP_STORE`, as Vulkan fixes the
    /// store as the rendering begins. A `DontCare` set later keeps the texels, as `DontCare` does;
    /// on a tile-based device this stores a target Metal would have discarded.
    Unknown,
    /// `MTLStoreActionCustomSampleDepthStore`: refused, [`Gap::RenderPass`] of
    /// [`RenderPassKind::StoreAction`].
    CustomSampleDepthStore,
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
/// Not built: a resolve (`resolveTexture` with a resolving store action, refused by the store
/// action), a 3D texture's `depthPlane` (refused by the texture's type) and
/// `storeActionOptions`, whose one option needs more than one sample.
///
/// Its default is Metal's: no texture, level and slice 0, `DontCare` load, `Store` store, opaque
/// black clear.
#[derive(Clone, Default)]
pub struct RenderPassColorAttachmentDescriptor {
    /// `texture`: the texture rendered into; `None` for no attachment at this index. It needs
    /// `RenderTarget` usage and a color format, and one sample; a 2D texture, a 2D array, a cube
    /// or a cube array.
    pub texture: Option<Texture>,
    /// `level`: the mipmap level rendered into.
    pub level: u32,
    /// `slice`: the array slice or cube face rendered into.
    pub slice: u32,
    /// `loadAction`: `DontCare` by default.
    pub load_action: LoadAction,
    /// `storeAction`: `Store` by default.
    pub store_action: StoreAction,
    /// `clearColor`: opaque black by default.
    pub clear_color: ClearColor,
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
/// Not built: a resolve and a 3D texture's `depthPlane`, as a color attachment's.
#[derive(Clone)]
pub struct RenderPassDepthAttachmentDescriptor {
    /// `texture`: the texture whose depth is rendered into; `None` for no depth attachment. It
    /// needs `RenderTarget` usage, a format with depth and one sample.
    pub texture: Option<Texture>,
    /// `level`: the mipmap level rendered into.
    pub level: u32,
    /// `slice`: the array slice or cube face rendered into.
    pub slice: u32,
    /// `loadAction`: `Clear` by default.
    pub load_action: LoadAction,
    /// `storeAction`: `DontCare` by default.
    pub store_action: StoreAction,
    /// `clearDepth`: 1 by default.
    pub clear_depth: f64,
}

impl Default for RenderPassDepthAttachmentDescriptor {
    /// Metal's defaults: no texture, level and slice 0, `Clear` load, `DontCare` store, a clear
    /// depth of 1.
    fn default() -> RenderPassDepthAttachmentDescriptor {
        RenderPassDepthAttachmentDescriptor {
            texture: None,
            level: 0,
            slice: 0,
            load_action: LoadAction::Clear,
            store_action: StoreAction::DontCare,
            clear_depth: 1.0,
        }
    }
}

/// `MTLRenderPassStencilAttachmentDescriptor`.
///
/// Not built: a resolve and a 3D texture's `depthPlane`, as a color attachment's.
#[derive(Clone)]
pub struct RenderPassStencilAttachmentDescriptor {
    /// `texture`: the texture whose stencil is rendered into; `None` for no stencil attachment. It
    /// needs `RenderTarget` usage, a format with stencil and one sample.
    pub texture: Option<Texture>,
    /// `level`: the mipmap level rendered into.
    pub level: u32,
    /// `slice`: the array slice or cube face rendered into.
    pub slice: u32,
    /// `loadAction`: `Clear` by default.
    pub load_action: LoadAction,
    /// `storeAction`: `DontCare` by default.
    pub store_action: StoreAction,
    /// `clearStencil`: 0 by default; its low 8 bits are the stencil's, as Metal keeps them.
    pub clear_stencil: u32,
}

impl Default for RenderPassStencilAttachmentDescriptor {
    /// Metal's defaults: no texture, level and slice 0, `Clear` load, `DontCare` store, a clear
    /// stencil of 0.
    fn default() -> RenderPassStencilAttachmentDescriptor {
        RenderPassStencilAttachmentDescriptor {
            texture: None,
            level: 0,
            slice: 0,
            load_action: LoadAction::Clear,
            store_action: StoreAction::DontCare,
            clear_stencil: 0,
        }
    }
}

/// `MTLRenderPassDescriptor`: the attachments a render command encoder renders into
/// ([`CommandBuffer::render_command_encoder`](crate::CommandBuffer::render_command_encoder)).
///
/// Not built, and so not fields: a rasterization rate map, tile sizes, imageblock and threadgroup
/// memory lengths, sample buffer attachments, layered rendering (`renderTargetArrayLength`) and a
/// pass with no attachment (`defaultRasterSampleCount`); a pass with no attachment is refused,
/// [`Gap::RenderPass`] of [`RenderPassKind::NoAttachment`]. `visibilityResultType` is not a field:
/// it decides only what an enabled visibility result mode writes, and no mode but `Disabled` is
/// built.
#[derive(Clone, Default)]
pub struct RenderPassDescriptor {
    /// `colorAttachments`.
    pub color_attachments: [RenderPassColorAttachmentDescriptor; MAX_COLOR_ATTACHMENTS],
    /// `depthAttachment`.
    pub depth_attachment: RenderPassDepthAttachmentDescriptor,
    /// `stencilAttachment`.
    pub stencil_attachment: RenderPassStencilAttachmentDescriptor,
    /// `renderTargetWidth`: where nonzero, the render area's width, cut to the attachments'.
    pub render_target_width: u32,
    /// `renderTargetHeight`: where nonzero, the render area's height, cut to the attachments'.
    pub render_target_height: u32,
    /// `visibilityResultBuffer`: `None` by default. The command buffer keeps it until it retires,
    /// and the pass leaves its bytes as they were: Metal's pass does so under either
    /// `visibilityResultType` where its encoder enables no mode (an Apple M3 Max, macOS 27.0.1),
    /// and `Disabled` is the only mode
    /// [`RenderCommandEncoder::set_visibility_result_mode`](crate::RenderCommandEncoder::set_visibility_result_mode)
    /// builds. It joins no hazard tracking: no command of the pass reaches it.
    pub visibility_result_buffer: Option<Buffer>,
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
    pub(crate) fn info(&self, resumed: bool) -> vk::RenderingAttachmentInfo<'static> {
        vk::RenderingAttachmentInfo::default()
            .image_view(self.attachment.view)
            .image_layout(self.layout)
            .load_op(match resumed {
                true => vk::AttachmentLoadOp::LOAD,
                false => self.load,
            })
            .store_op(self.store)
            .clear_value(self.clear)
    }

    /// The barrier that moves its level and layer, every aspect, from `old` to `new`, after every
    /// earlier command and before every later one.
    pub(crate) fn barrier(
        &self,
        old: vk::ImageLayout,
        new: vk::ImageLayout,
    ) -> vk::ImageMemoryBarrier2<'static> {
        let a = &self.attachment;
        layout_barrier(a.image, self.range(), old, new)
    }

    /// Its level and layer, every aspect.
    pub(crate) fn range(&self) -> vk::ImageSubresourceRange {
        let a = &self.attachment;
        vk::ImageSubresourceRange {
            aspect_mask: a.aspects,
            base_mip_level: a.level,
            level_count: 1,
            base_array_layer: a.layer,
            layer_count: 1,
        }
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

/// A pass descriptor checked against its textures: its attachments, by index for the colors,
/// and its render area.
pub(crate) struct Pass {
    /// Up to the last color attachment with a texture.
    pub(crate) colors: Vec<Option<Target>>,
    /// The depth attachment and the stencil attachment: one view of one image where both are set
    /// (`Pass::check`), each with its own load, store and clear.
    pub(crate) depth: Option<Target>,
    pub(crate) stencil: Option<Target>,
    pub(crate) area: [u32; 2],
    /// Every attachment's samples, one bit: the pass refuses attachments of two counts.
    pub(crate) samples: vk::SampleCountFlags,
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
        let [width, height] = self.area;
        let mut info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            })
            .layer_count(1)
            .color_attachments(&colors);
        if let Some(depth) = &depth {
            info = info.depth_attachment(depth);
        }
        if let Some(stencil) = &stencil {
            info = info.stencil_attachment(stencil);
        }
        // SAFETY: as the caller states; the render area lies inside every attachment, each is a
        // view of one level and layer of its image, and a depth and a stencil attachment both set
        // are one view.
        unsafe { device.cmd_begin_rendering(command, &info) };
    }

    /// Checks `d` on the device of `core`.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's texture or visibility result buffer;
    /// [`Refusal::AttachmentUsage`]; [`Refusal::AttachmentPlace`]; [`Refusal::AttachmentFormat`];
    /// [`Gap::RenderPass`]; [`Error::Vulkan`] where an attachment's view cannot be made.
    pub(crate) fn check(core: &Arc<Core>, d: &RenderPassDescriptor) -> Result<Pass, Error> {
        let gap = |kind| Error::Unsupported(Gap::RenderPass { kind });
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
            let attachment = texture.attachment(c.level, c.slice)?;
            let clear = match c.load_action {
                LoadAction::Clear => vk::ClearValue {
                    color: clear_color(pf, texture.emulation(), c.clear_color)
                        .ok_or_else(|| gap(RenderPassKind::ClearFormat))?,
                },
                LoadAction::Load | LoadAction::DontCare => vk::ClearValue::default(),
            };
            colors.resize_with(index + 1, || None);
            colors[index] = Some(target(
                texture,
                attachment,
                c.load_action,
                c.store_action,
                clear,
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
                let attachment = texture.attachment(z.level, z.slice)?;
                let clear = match z.load_action {
                    LoadAction::Clear if !(0.0..=1.0).contains(&z.clear_depth) => {
                        return Err(gap(RenderPassKind::ClearDepth))
                    }
                    LoadAction::Clear => z.clear_depth as f32,
                    LoadAction::Load | LoadAction::DontCare => 0.0,
                };
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
                let attachment = texture.attachment(s.level, s.slice)?;
                let clear = vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: s.clear_stencil & 0xff,
                    },
                };
                Some(target(
                    texture,
                    attachment,
                    s.load_action,
                    s.store_action,
                    clear,
                )?)
            }
            None => None,
        };
        if let (Some(depth), Some(stencil)) = (&depth, &stencil) {
            if depth.attachment.view != stencil.attachment.view {
                return Err(gap(RenderPassKind::DepthStencilViews));
            }
        }
        let mut pass = Pass {
            colors,
            depth,
            stencil,
            area: [u32::MAX; 2],
            samples: vk::SampleCountFlags::empty(),
        };
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
        pass.area = pass.targets().fold(pass.area, |area, t| {
            let extent = t.attachment.extent;
            [area[0].min(extent[0]), area[1].min(extent[1])]
        });
        if pass.area[0] == u32::MAX {
            return Err(gap(RenderPassKind::NoAttachment));
        }
        for (axis, set) in [d.render_target_width, d.render_target_height]
            .into_iter()
            .enumerate()
        {
            if set > 0 {
                pass.area[axis] = pass.area[axis].min(set);
            }
        }
        Ok(pass)
    }
}

/// The Vulkan load and store operations of Metal's `load` and `store` actions: the load one to
/// one, and `STORE` for `Store`, `DontCare` ([`StoreAction::DontCare`]) and `Unknown`
/// ([`StoreAction::Unknown`]).
///
/// Err: [`Gap::RenderPass`] of [`RenderPassKind::Resolve`] or [`RenderPassKind::StoreAction`].
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
        StoreAction::DontCare | StoreAction::Store | StoreAction::Unknown => {
            vk::AttachmentStoreOp::STORE
        }
        StoreAction::MultisampleResolve | StoreAction::StoreAndMultisampleResolve => {
            return gap(RenderPassKind::Resolve)
        }
        StoreAction::CustomSampleDepthStore => return gap(RenderPassKind::StoreAction),
    };
    Ok((load, store))
}

/// One attachment of `texture`, viewed as `attachment`, with its load and store actions and its
/// clear value.
///
/// Err: as [`ops`].
fn target(
    texture: &Texture,
    attachment: Attachment,
    load: LoadAction,
    store: StoreAction,
    clear: vk::ClearValue,
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
