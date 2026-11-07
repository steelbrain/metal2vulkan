//! The `half` shadow of a pass's extended-range colors: Metal's tile memory holds a `BGR10_XR`,
//! `BGRA10_XR` or `RGB10A8_2P_XR10` color (and the `_sRGB` twins) as `half` through a render pass
//! and rounds it to the codes only at the store, where the encoder stores the codes at every draw.
//! A draw that blends into or reads such a color an earlier draw of the pass wrote, or whose clear
//! its codes cannot hold, would see the codes; from that draw on the pass holds the color's shadow
//! instead, and every draw that writes or reads the color takes the variant that reads the shadow
//! in place of the codes and writes both (`BindState::extended_shadows`, `lower::fragment::fetch`),
//! so the codes always hold the encode of the last `half` written to each written channel and no
//! store pass runs.
//!
//! - **The shadow** is an `R16G16B16A16_SFLOAT` image of the pass's extent, in `GENERAL` like
//!   every runtime image, holding pixel `(x, y)`'s `half` values at texel `(x, y)`, and in a
//!   multisampled pass each sample's as the color's storage form holds the samples
//!   (`framebuffer_fetch::through_planes`): a multisampled image of the pass's samples, or a 2D
//!   array of one layer per sample (sample `s` at texel `(x, y, s)`); made at the draw where it
//!   starts, one per color and pass, bound where the plan writes `WriteSource::ExtendedShadow`.
//!   Each draw that binds it reads and writes it in the fragment shader, inside the pixel
//!   interlock that orders the color's own storage form, so what orders that form between draws
//!   orders it too (`InterlockedStorage`'s colors: a barrier, or the interlock alone on a driver
//!   measured to order such draws). In a pass that fetches through input attachments the color
//!   stays the subpass's, its codes read and written through its input attachment in
//!   rasterization order, and the draw reads and writes the shadow beside it inside the pixel
//!   interlock (`lower::fragment`'s `input_shadow`), behind the same barrier; there a multisampled
//!   pass holds the samples in a multisampled image.
//! - **The start.** The pass's instances keep their draws for recording again
//!   (`render/reroute.rs`) on every device while the pass has such a color, of one layer, without
//!   a rasterization-rate map, fetching in the pixel interlock, or through input attachments on a
//!   device whose functions read and write the shadow beside them (the pixel interlock, fragment
//!   stores, the shadow's format as a storage image and, at more than one sample, multisampled
//!   storage images: `framebuffer_fetch::holds_extended_shadows`); a multisampled pass's draws do not
//!   record again, so there a color starts at the first draw that reads it (blends into it,
//!   fetches it), not only at one that reads what an earlier draw wrote. At the draw
//!   (the gate above), a color that no instance before the current one wrote (no draw, tile
//!   dispatch or execution of render commands) starts: where the current instance's draws wrote
//!   it, they record again first into an instance that leaves the color out of its subpass (in a
//!   pass that fetches through input attachments, one of the same shape), in its place, holding
//!   the shadow (`record_leaving_out`); else the draw's own instance holds it.
//!   The instance that first holds it sets it before it, between barriers: to the clear color's
//!   halves where the pass clears the color (`ExtendedRange::clear_tile_value`, every sample),
//!   else, in a single-sampled pass, by the shadow load of the codes
//!   (`bind::extended_shadow_module`), which writes each texel's tile value, the fetch's own
//!   decode, reading the codes as a storage image (in a pass that fetches through input
//!   attachments, where the texture's image has that usage). A color an earlier instance wrote, or whose instance's draws cannot record again,
//!   keeps the codes (the announcement), as does one whose start fails (its draws' shadowed
//!   variants refused, a load the device cannot build, a multisampled pass that does not clear
//!   it), for the rest of the pass.
//! - **The end.** A draw whose shadowed variant is refused (a mesh pipeline's, an amplified
//!   draw's, and over input attachments one whose shadow stores the tests or the coverage could
//!   leave apart from its codes: `lower::fragment`'s `input_shadow`) gives up the shadows of the colors it writes or reads, and an execution of render
//!   commands or a tile dispatch, whose commands write the codes alone, gives up every shadow:
//!   the codes hold the encode of the last `half` written, and the draws after it see them, as
//!   before the shadow. None starts again in the pass. Not while the encoder records an
//!   execution's commands, which bind no shadow.
//! - **The announcement**. A draw that held a color's shadow read what Metal's tile memory holds,
//!   so `extended-range-blends-stored-codes` names only the colors a draw read from the codes
//!   (`degrade_extended_range_blends`, `ExtendedShadows::held`).

use super::RenderLowering;
use crate::blit::TRANSFER;
use crate::command_buffer::CommandBuffer;
use crate::device::functions::Vk;
use crate::error::{internal, Error};
use crate::framebuffer_fetch;
use crate::image_plan::ImagePlan;
use crate::owned_kernel::{OwnedKernel, OwnedModule};
use crate::pipeline::GraphicsState as DrawingState;
use crate::render_pass::{PassLayout, RenderPassDescriptor, Target};
use crate::render_target;
use crate::render_values::LoadAction;
use crate::resource::ResourceKey;
use crate::stencil_copy::PassObjects;
use crate::texture::Texture;
use crate::texture_descriptor::TextureDescriptor;
use crate::tracker::{Access, Barrier, Use};
use crate::values::ResourceOptions;
use crate::Device;
use ash::vk;
use metal2vulkan::bind::{BindState, ExtendedShadowModule};
use metal2vulkan::metal::format::{ExtendedRange, FormatMapping};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::pipeline::GraphicsState;
use metal2vulkan::vk::{
    Format, ImageAspectFlags, ImageCreateFlags, ImageType, ImageUsageFlags, SampleCountFlags,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// The shadow's format.
const SHADOW_FORMAT: Format = Format::R16G16B16A16_SFLOAT;

/// A pass's extended-range colors and their shadows (the module docs).
pub(super) struct ExtendedShadows {
    /// The pass's colors of a measured extended-range form, by index: the form, and the halves
    /// its tile holds after the pass's clear where it clears.
    colors: BTreeMap<u32, (ExtendedRange, Option<[f32; 4]>)>,
    /// The pass holds shadows: of one layer, without a rasterization-rate map, with such a color,
    /// fetching in the pixel interlock, or through input attachments on a device whose functions
    /// read and write the shadow beside them (`framebuffer_fetch::holds_extended_shadows`).
    holds: bool,
    /// The first instance, by index, whose draws, tile dispatches or executed commands may have
    /// written each color.
    first_written: BTreeMap<u32, usize>,
    /// The shadows made, by color: one per color and pass.
    made: BTreeMap<u32, Target>,
    /// The colors whose shadow the draws hold.
    live: BTreeSet<u32>,
    /// The colors whose shadow ended or did not start: none starts again in the pass.
    ended: BTreeSet<u32>,
    /// The colors whose shadow the next draw's instance sets before it.
    pending: Vec<u32>,
    /// The settings before the instances: the instance's index and the color.
    fills: Vec<(usize, u32)>,
}

impl ExtendedShadows {
    /// The shadows of the pass `descriptor` describes, of `layout`, on `device`, with a
    /// rasterization-rate map where `rate_mapped`: none made yet.
    pub(super) fn of(
        device: &Device,
        descriptor: &RenderPassDescriptor,
        layout: &PassLayout,
        rate_mapped: bool,
    ) -> ExtendedShadows {
        let colors: BTreeMap<u32, (ExtendedRange, Option<[f32; 4]>)> = (0u32..)
            .zip(&descriptor.color_attachments)
            .filter_map(|(index, color)| {
                let texture = color.texture.as_ref()?;
                // Not the padded and bfloat forms, whose tile memory is unmeasured.
                let form = ExtendedRange::of(texture.pixel_format())
                    .filter(|form| !form.padded() && !form.bfloat())?;
                let tile = match color.load_action {
                    LoadAction::Clear => Some(form.clear_tile_value(color.clear_color.channels())?),
                    _ => None,
                };
                Some((index, (form, tile)))
            })
            .collect();
        let profile = device.compile_device().profile();
        let holds = !colors.is_empty()
            && framebuffer_fetch::holds_extended_shadows(
                profile,
                layout.fetch_inputs,
                layout.samples,
            )
            && layout.layers == 1
            && !rate_mapped;
        ExtendedShadows {
            colors,
            holds,
            first_written: BTreeMap::new(),
            made: BTreeMap::new(),
            live: BTreeSet::new(),
            ended: BTreeSet::new(),
            pending: Vec::new(),
            fills: Vec::new(),
        }
    }

    /// The shadows of an encoder that holds none: one that replays an execution's commands.
    pub(super) fn none() -> ExtendedShadows {
        ExtendedShadows {
            colors: BTreeMap::new(),
            holds: false,
            first_written: BTreeMap::new(),
            made: BTreeMap::new(),
            live: BTreeSet::new(),
            ended: BTreeSet::new(),
            pending: Vec::new(),
            fills: Vec::new(),
        }
    }

    /// Whether the pass's instances keep their draws for recording again: it may hold shadows.
    pub(super) fn keeps_draws(&self) -> bool {
        self.holds
    }

    /// Whether the pass made `color`'s shadow, whose draws took a view of the color.
    pub(super) fn made(&self, color: u32) -> bool {
        self.made.contains_key(&color)
    }

    /// Whether the draws hold color `color`'s shadow: a draw that bound it read the tile's
    /// `half`, not the codes (`degrade_extended_range_blends` announces nothing for it).
    pub(super) fn held(&self, color: u32) -> bool {
        self.live.contains(&color)
    }

    /// Give up the shadows of `colors` for the rest of the pass.
    fn end(&mut self, colors: &[u32]) {
        for c in colors {
            self.live.remove(c);
            self.ended.insert(*c);
        }
    }

    /// Note that what instance `at` records may write `colors`.
    fn written(&mut self, at: usize, colors: impl IntoIterator<Item = u32>) {
        for c in colors {
            if self.colors.contains_key(&c) {
                self.first_written.entry(c).or_insert(at);
            }
        }
    }

    /// The storage views of the shadows the draws hold, by color.
    pub(super) fn views(&self) -> Result<Vec<(u32, vk::ImageView)>, Error> {
        self.live
            .iter()
            .map(|&c| {
                let target = self
                    .made
                    .get(&c)
                    .ok_or_else(|| internal("a held shadow never made"))?;
                Ok((c, render_target::storage_view(target)?))
            })
            .collect()
    }

    /// A draw's uses of the shadows of `colors` its plan binds (ascending, `PlanFacts::extended`):
    /// the fragment shader reads and writes each.
    pub(super) fn draw_uses(&self, colors: &[u32]) -> Result<Vec<Use<ResourceKey>>, Error> {
        let stage = vk::PipelineStageFlags2::FRAGMENT_SHADER;
        let read = Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_READ);
        let write = Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_WRITE);
        let mut uses = Vec::new();
        for &c in colors {
            let target = self
                .made
                .get(&c)
                .filter(|_| self.live.contains(&c))
                .ok_or_else(|| internal("a draw of a shadow the pass does not hold"))?;
            let layers = 0..target.layers;
            uses.push(
                target
                    .texture
                    .image_use(0..1, layers.clone(), read, false)?,
            );
            uses.push(target.texture.image_use(0..1, layers, write, true)?);
        }
        Ok(uses)
    }

    /// The settings before each of the pass's `count` instances, of the colors of `layout` (the
    /// pass's at the end of its encoder) that instance first holds the shadows of, and their
    /// uses: no list at all where the pass sets no shadow, every instance's settings then empty.
    pub(super) fn settings(
        &self,
        device: &Device,
        layout: &PassLayout,
        count: usize,
    ) -> Result<(Vec<Vec<ShadowFill>>, Vec<Use<ResourceKey>>), Error> {
        if self.fills.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut settings: Vec<Vec<ShadowFill>> = (0..count).map(|_| Vec::new()).collect();
        let mut uses = Vec::new();
        for &(at, c) in &self.fills {
            let shadow = self
                .made
                .get(&c)
                .ok_or_else(|| internal("a shadow set but never made"))?;
            let &(form, tile) = self
                .colors
                .get(&c)
                .ok_or_else(|| internal("a shadow of no extended-range color"))?;
            let [width, height] = shadow.extent;
            let kind = match tile {
                Some(values) => {
                    let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
                    let layers = 0..shadow.layers;
                    uses.push(shadow.texture.image_use(0..1, layers, write, true)?);
                    FillKind::Clear(values, shadow.layers)
                }
                None => {
                    let codes = layout
                        .colors
                        .get(c as usize)
                        .and_then(Option::as_ref)
                        .map(|color| &color.target)
                        .filter(|target| target.format == form.raw_format() && target.layers == 1)
                        .ok_or_else(|| internal("a shadow of codes the pass does not hold"))?;
                    let kernel = device.owned_kernel(OwnedModule::ExtendedShadowLoad(form))?;
                    let (objects, set) = PassObjects::with_set(
                        device,
                        kernel.set_layout,
                        &[
                            (
                                vk::DescriptorType::STORAGE_IMAGE,
                                render_target::storage_view(codes)?,
                            ),
                            (
                                vk::DescriptorType::STORAGE_IMAGE,
                                render_target::storage_view(shadow)?,
                            ),
                        ],
                    )?;
                    let compute = vk::PipelineStageFlags2::COMPUTE_SHADER;
                    let read = Access::new(compute, vk::AccessFlags2::SHADER_STORAGE_READ);
                    let write = Access::new(compute, vk::AccessFlags2::SHADER_STORAGE_WRITE);
                    let (level, layer) = (codes.level, codes.layer);
                    uses.push(codes.texture.image_use(
                        level..level + 1,
                        layer..layer + 1,
                        read,
                        false,
                    )?);
                    uses.push(shadow.texture.image_use(0..1, 0..1, write, true)?);
                    let size = ExtendedShadowModule::LOCAL_SIZE;
                    FillKind::Load {
                        _objects: Arc::new(objects),
                        kernel,
                        set,
                        groups: [width.div_ceil(size), height.div_ceil(size)],
                    }
                }
            };
            let fill = ShadowFill {
                shadow: shadow.texture.raw_image()?,
                kind,
            };
            settings
                .get_mut(at)
                .ok_or_else(|| internal("a shadow set before an instance the pass lacks"))?
                .push(fill);
        }
        Ok((settings, uses))
    }

    /// The shadows made, which the command buffer holds until it completes.
    pub(super) fn textures(&self) -> Vec<Texture> {
        self.made.values().map(|t| t.texture.clone()).collect()
    }
}

/// How a shadow is set before the instance that first holds it (the module docs).
enum FillKind {
    /// To the clear color's halves, in every layer of the shadow (one per sample where it holds
    /// the samples as planes).
    Clear([f32; 4], u32),
    /// By the shadow load of the codes: its set's pool, held until the command buffer completes,
    /// the kernel, its set, and the workgroups it runs.
    Load {
        _objects: Arc<PassObjects>,
        kernel: OwnedKernel,
        set: vk::DescriptorSet,
        groups: [u32; 2],
    },
}

/// The setting of one shadow before an instance.
pub(super) struct ShadowFill {
    shadow: vk::Image,
    kind: FillKind,
}

impl ShadowFill {
    /// Record, outside a render pass, the setting between a barrier from everything before and
    /// one to everything after.
    ///
    /// # Safety
    ///
    /// `raw` is the encoder's command buffer, recording outside a render pass, which only the
    /// encoder records; the shadow, the codes and the setting's objects are live until it
    /// completes and rest in `GENERAL`.
    pub(super) unsafe fn record_in(&self, device: &Vk, raw: vk::CommandBuffer) {
        // SAFETY: the caller's.
        unsafe {
            Barrier::everything().record(device, raw);
            match &self.kind {
                FillKind::Clear(values, layers) => {
                    let colour = vk::ClearColorValue { float32: *values };
                    let range = vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: *layers,
                    };
                    let general = vk::ImageLayout::GENERAL;
                    device.cmd_clear_color_image(raw, self.shadow, general, &colour, &[range]);
                }
                FillKind::Load {
                    kernel,
                    set,
                    groups,
                    ..
                } => {
                    let compute = vk::PipelineBindPoint::COMPUTE;
                    device.cmd_bind_pipeline(raw, compute, kernel.pipeline);
                    device.cmd_bind_descriptor_sets(raw, compute, kernel.layout, 0, &[*set], &[]);
                    let [x, y] = *groups;
                    device.cmd_dispatch(raw, x, y, 1);
                }
            }
            Barrier::everything().record(device, raw);
        }
    }
}

/// Whether a draw of `graphics` reads color `index` (the extended-range announcement's and the
/// gate's test): in the interlocked storage form, as an input attachment, or as a blend's
/// destination.
pub(super) fn reads_color(graphics: &GraphicsState, index: u32) -> bool {
    stores(graphics, index)
        || graphics
            .input_attachments
            .iter()
            .any(|a| a.color_index == index)
        || color(graphics, index).is_some_and(|a| a.blend.is_some())
}

/// The colors whose shadows a state's creation predicts its first draw holds
/// (`pipeline/predicted.rs`), for a pipeline of `graphics` whose color attachments are `colors`
/// (each index's pixel format) at `samples`, in a pass that `holds` shadows: in a multisampled pass, each color of a measured extended-range form the draw
/// reads ([`reads_color`]), as such a pass starts that color's shadow at the first draw that reads
/// it where it clears the color ([`RenderLowering::start_extended_shadows`]), and a multisampled
/// color's only contents at the pass's start that are not undefined are its clear's or its
/// load's, a load being what a pass of memoryless samples cannot take. None in a single-sampled
/// pass, which starts the shadow only where an earlier draw of the pass wrote the color or its
/// clear's codes cannot hold the clear, which no creation sees: of the suite's single-sampled
/// draws that blend into such a color, the `xr-blend-planes-*` ones blend once over a loaded
/// target and hold no shadow. None where the pass would hold no shadow (not `holds`,
/// [`ExtendedShadows::of`]: a device whose pass fetches through input attachments and whose
/// functions cannot read and write the shadow beside them); the creation predicts the pass of one
/// layer without a rasterization-rate map, Metal's defaults.
pub(crate) fn predicted_shadows(
    graphics: &GraphicsState,
    colors: &[Option<PixelFormat>],
    samples: u32,
    holds: bool,
) -> BTreeSet<u32> {
    if samples <= 1 || !holds {
        return BTreeSet::new();
    }
    (0u32..)
        .zip(colors)
        .filter(|(index, format)| {
            format
                .and_then(ExtendedRange::of)
                .is_some_and(|form| !form.padded() && !form.bfloat())
                && reads_color(graphics, *index)
        })
        .map(|(index, _)| index)
        .collect()
}

/// Whether a draw of `graphics` may write color `index`: in the interlocked storage form, or
/// through a write mask that names a channel.
pub(super) fn writes_color(graphics: &GraphicsState, index: u32) -> bool {
    stores(graphics, index) || color(graphics, index).is_some_and(|a| !a.write_mask.is_empty())
}

/// Whether `graphics` reads and writes color `index` in the interlocked storage form.
fn stores(graphics: &GraphicsState, index: u32) -> bool {
    graphics
        .storage_attachments
        .iter()
        .any(|a| a.color_index == index)
}

/// `graphics`'s state of color `index`.
fn color(
    graphics: &GraphicsState,
    index: u32,
) -> Option<&metal2vulkan::pipeline::ColorAttachmentState> {
    graphics
        .color_attachments
        .get(index as usize)
        .and_then(Option::as_ref)
}

impl RenderLowering {
    /// The shadows a draw binds: the held ones, none while the encoder records an execution's
    /// commands.
    pub(super) fn extended_shadows_bound(&self) -> &BTreeSet<u32> {
        static NONE: BTreeSet<u32> = BTreeSet::new();
        match self.replaying {
            true => &NONE,
            false => &self.extended.live,
        }
    }

    /// Before a draw of `pipeline`, start the shadows of the colors it blends into or reads that
    /// an earlier draw of the pass wrote or whose clear their codes cannot hold (the module
    /// docs), in a multisampled pass every color it blends into or reads: those no earlier
    /// instance wrote, the current instance's draws recorded again first into one that leaves
    /// them out where they wrote them.
    pub(super) fn start_extended_shadows(&mut self, pipeline: &DrawingState) -> Result<(), Error> {
        if !self.extended.holds
            || self.replaying
            || self.rerouting
            || pipeline.descriptor().vertex().is_none()
        {
            return Ok(());
        }
        let graphics = &pipeline.base().graphics;
        let here = self.earlier.len();
        // In a multisampled pass no draw records again (`render/reroute.rs`), so a shadow starts at
        // the first draw that reads the color, before any draw of its instance wrote it, rather
        // than at the first that reads what an earlier draw wrote.
        let eager = self.layout.samples > 1;
        let wanted: Vec<u32> = self
            .extended
            .colors
            .keys()
            .copied()
            .filter(|c| {
                reads_color(graphics, *c)
                    && (eager || self.extended_written.contains(c))
                    && !self.extended.live.contains(c)
                    && !self.extended.ended.contains(c)
            })
            .collect();
        // Where the current instance's draws wrote a starting color, they record again.
        let mut again = false;
        let mut starting = Vec::new();
        for c in wanted {
            let written_here = match self.extended.first_written.get(&c) {
                None => Some(false),
                Some(&at) => (at == here).then_some(true),
            };
            match written_here {
                Some(here) if self.shadow_loads(c)? => {
                    again |= here;
                    starting.push(c);
                }
                _ => self.extended.end(&[c]),
            }
        }
        if starting.is_empty() {
            return Ok(());
        }
        let profile = self.core.device().compile_device().profile();
        let samples = match framebuffer_fetch::through_planes(profile, self.layout.samples) {
            true => ShadowSamples::Planes(self.layout.samples),
            false => ShadowSamples::Multisampled(self.layout.samples),
        };
        for &c in &starting {
            if !self.extended.made.contains_key(&c) {
                let shadow = new_shadow(self.core.command_buffer(), self.layout.extent, samples)?;
                self.extended.made.insert(c, shadow);
            }
        }
        self.extended.live.extend(starting.iter().copied());
        if !again {
            self.extended.pending.extend(starting);
            return Ok(());
        }
        let mut colors: Vec<u32> = self.current.shape.storage.to_vec();
        colors.extend(starting.iter().copied());
        colors.sort_unstable();
        colors.dedup();
        match self.record_leaving_out(colors, pipeline)? {
            true => self
                .extended
                .fills
                .extend(starting.iter().map(|&c| (here, c))),
            false => self.extended.end(&starting),
        }
        Ok(())
    }

    /// Whether the shadow of color `c` can be set before its instance: from the clear, or, in a
    /// single-sampled pass, by the shadow load of codes in the form's raw format, which the
    /// device builds. A multisampled pass that loads the color keeps the codes: the load reads
    /// one sample a texel.
    fn shadow_loads(&self, c: u32) -> Result<bool, Error> {
        let Some(&(form, tile)) = self.extended.colors.get(&c) else {
            return Ok(false);
        };
        if tile.is_some() {
            return Ok(true);
        }
        if self.layout.samples > 1 {
            return Ok(false);
        }
        // The load reads the codes as a storage image: in a pass that fetches through input
        // attachments, only where the texture's image has that usage.
        let raw = self
            .layout
            .colors
            .get(c as usize)
            .and_then(Option::as_ref)
            .is_some_and(|color| {
                color.target.format == form.raw_format()
                    && (!self.layout.fetch_inputs
                        || color
                            .target
                            .texture
                            .image_plan()
                            .is_ok_and(|plan| plan.usage.contains(ImageUsageFlags::STORAGE)))
            });
        if !raw {
            return Ok(false);
        }
        match self
            .core
            .device()
            .owned_kernel(OwnedModule::ExtendedShadowLoad(form))
        {
            Ok(_) => Ok(true),
            Err(Error::Compile(_)) => Ok(false),
            Err(other) => Err(other),
        }
    }

    /// After settling a draw of `pipeline` in `state`: where its shadowed variant was refused
    /// (`settle_draw` bound no shadow where the pass holds some), give up the shadows of the
    /// colors it writes or reads.
    pub(super) fn give_up_extended_shadows(&mut self, pipeline: &DrawingState, state: &BindState) {
        let bound = self.extended_shadows_bound();
        if bound.is_empty() || !state.extended_shadows.is_empty() {
            return;
        }
        let graphics = &pipeline.base().graphics;
        let touched: Vec<u32> = (bound.iter().copied())
            .filter(|&c| reads_color(graphics, c) || writes_color(graphics, c))
            .collect();
        self.extended.end(&touched);
    }

    /// Once a draw of `pipeline` installed its instance: the shadows that instance first holds
    /// are set before it, and what it may write is noted.
    pub(super) fn installed_extended_draw(&mut self, pipeline: &DrawingState) {
        if self.replaying {
            return;
        }
        let at = self.earlier.len();
        let ExtendedShadows { pending, fills, .. } = &mut self.extended;
        fills.extend(pending.drain(..).map(|c| (at, c)));
        let graphics = &pipeline.base().graphics;
        let written: Vec<u32> = self
            .extended
            .colors
            .keys()
            .copied()
            .filter(|&c| writes_color(graphics, c))
            .collect();
        self.extended.written(at, written);
    }

    /// At an execution of render commands or a tile dispatch, whose commands write the codes
    /// alone: give up every shadow, and note that the current instance may have written every
    /// extended-range color.
    pub(super) fn end_extended_shadows(&mut self) {
        let at = self.earlier.len();
        let colors: Vec<u32> = self.extended.colors.keys().copied().collect();
        self.extended.written(at, colors.iter().copied());
        let live: Vec<u32> = self.extended.live.iter().copied().collect();
        self.extended.end(&live);
    }
}

/// How a shadow holds a multisampled pass's samples: as the color's storage image holds them in
/// the interlocked storage form (`framebuffer_fetch::through_planes`), so the shadowed draws read
/// and write each sample alike.
#[derive(Clone, Copy)]
enum ShadowSamples {
    /// A multisampled image of the pass's samples (one at one sample).
    Multisampled(u32),
    /// A single-sample 2D array of one layer per sample of the pass's.
    Planes(u32),
}

/// A shadow of the pass's `extent` holding its samples as `samples` says (the module docs).
fn new_shadow(
    maker: &CommandBuffer,
    extent: [u32; 2],
    samples: ShadowSamples,
) -> Result<Target, Error> {
    let [width, height] = extent;
    let (count, layers) = match samples {
        ShadowSamples::Multisampled(count) => (count, 1),
        ShadowSamples::Planes(count) => (1, count),
    };
    let descriptor = TextureDescriptor {
        texture_type: match samples {
            ShadowSamples::Multisampled(1) => TextureType::Type2D,
            ShadowSamples::Multisampled(_) => TextureType::Type2DMultisample,
            ShadowSamples::Planes(_) => TextureType::Type2DArray,
        },
        pixel_format: PixelFormat::RGBA16Float,
        width: width.into(),
        height: height.into(),
        array_length: layers.into(),
        sample_count: count.into(),
        usage: TextureUsage::RENDER_TARGET,
        resource_options: ResourceOptions::private(),
        ..TextureDescriptor::default()
    };
    let plan = ImagePlan {
        pixel_format: PixelFormat::RGBA16Float,
        flags: ImageCreateFlags::empty(),
        image_type: ImageType::TYPE_2D,
        format: SHADOW_FORMAT,
        extent: [width, height, 1],
        levels: 1,
        layers,
        samples: SampleCountFlags::from_raw(count),
        usage: ImageUsageFlags::STORAGE | ImageUsageFlags::TRANSFER_DST,
        view_formats: Vec::new(),
        mapping: FormatMapping::direct(SHADOW_FORMAT, ImageAspectFlags::COLOR),
        unavailable: TextureUsage::SHADER_READ
            | TextureUsage::SHADER_WRITE
            | TextureUsage::SHADER_ATOMIC
            | TextureUsage::RENDER_TARGET
            | TextureUsage::PIXEL_FORMAT_VIEW,
    };
    let texture = Device::new_planned_texture(maker, &descriptor, plan)?;
    Ok(Target {
        texture,
        level: 0,
        layer: 0,
        layers,
        format: SHADOW_FORMAT,
        samples: count,
        extent,
    })
}
