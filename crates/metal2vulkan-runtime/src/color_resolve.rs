//! The crate-owned multisample color resolve, drawn after the pass, where the device's resolve in
//! the pass cannot write the answer at all.
//!
//! A color attachment resolves by the device's `AVERAGE`, which leaves the average's rounding, and
//! an sRGB average's space, to the device: Metal's own resolve where the driver's is Metal's
//! ([`average_is_metals`], `DriverFact::ColorResolveAverageIsMetals`: MoltenVK from 1.4.1 on
//! Apple's GPUs resolves every format by Metal's multisample resolve store action, measured at 2
//! and 4 samples), elsewhere the device's native answer, announced as `device-rounded-resolve`. So
//! does an attachment a fetching pass's sample planes hold as it ends: the planes are merged back
//! and one more instance of no draws resolves it (`sample_planes.rs`). The crate-owned resolve
//! (`bind::color_resolve_modules`) draws only the two resolves no device's average writes:
//!
//! An extended-range attachment (`BGR10_XR`, `BGRA10_XR`, their `_sRGB` twins and
//! `RGB10A2Unorm_sRGB`, the forms Metal multisamples) is held in its raw format's codes, which no
//! device's resolve averages, so it resolves by the crate-owned resolve at every count
//! ([`extended`]: `bind::ColorResolveFormat::extended_range`, whose docs state
//! Metal's measured rule and the near-equivalents of the channels the codes cannot give, announced
//! as `extended-range-resolve`), from any source below; an image that takes neither input
//! attachment usage nor a copy is refused, as no other resolve reads it.
//!
//! **A resolve into the attachment's format's sRGB twin** ([`twin_rules`], `resolve.rs`), which
//! Metal resolves in the resolve texture's encoding, takes the crate-owned resolve on every
//! device: a subpass's resolve attachment is of its color attachment's own format
//! (`VUID-VkSubpassDescription-pResolveAttachments-00850`), so no device's average writes the
//! twin's encoding. Its rules are the resolve attachment's format's. An sRGB attachment resolving
//! into its UNORM twin is read through its image's view in that twin, whose codes the UNORM rules
//! average. A UNORM attachment resolving into its sRGB twin takes the device's own sRGB decode and
//! encode where the device renders the sRGB format ([`decodes`],
//! `bind::ColorResolveFormat::device_srgb`): the resolve reads the samples through a view in the
//! sRGB format (a UNORM texture's image lists its twin's views, `image_plan.rs`) and writes their
//! average through the resolve attachment's own view, announced as `device-rounded-resolve` on
//! every device, Metal's resolve taking its own 12-bit tables; a device without the format reads
//! the codes in the attachment's own format and takes those tables. Such a resolve reads the
//! attachment itself, never its sample planes, whose copy format holds the codes under the
//! attachment's own format's rules; the attachment's image, and a stand-in a pass renders it into,
//! take the input attachment usage it reads them through where that is free ([`twin_resolved`],
//! [`twin_read_view`]), and samples it can neither read nor copy are not built
//! (`NotImplemented(Render)`).
//!
//! The choice is made with the attachment's resolve ([`of_resolve`], `resolve.rs`), so the pass's
//! shape (`render_target.rs`) leaves such a resolve out of its subpass from its first instance:
//! the pass's last instance stores the attachment's samples, and after the pass one render pass
//! instance of the resolve's own pass per attachment draws a triangle over the render area that
//! reads each pixel's samples and writes the resolve attachment (the resolve texture's level and
//! slice, or the scratch image copied there after, `resolve.rs`):
//!
//! - **From the sample planes** (`sample_planes.rs`) where they hold an extended-range
//!   attachment's samples as the pass ends (its last instance's draws fetch it): read through a
//!   sampled 2D-array view in their copy format. The planes are not merged back into the
//!   attachment unless Metal keeps its samples (`Store`, `StoreAndMultisampleResolve`).
//! - **From the attachment** otherwise: its level and layer as the resolve pass's input
//!   attachment 0, loaded and stored. An 8-bit sRGB attachment whose image takes its UNORM twin's
//!   view (an sRGB texture's image lists its twin's formats) is read through that view
//!   ([`raw_view`], `bind::ResolveSource::RawInputAttachment`), so the resolve reads each sample's
//!   code as a unorm code with no table lookup. Its image needs input attachment usage, which a
//!   multisampled color render target the runtime makes takes where the device may resolve it so
//!   and the usage costs the image nothing, and a memoryless one wherever the device makes the
//!   image (`framebuffer_fetch.rs`), as does a stand-in (`attachment_stand_in.rs`).
//! - **From a copy of the attachment's samples** where its image lacks that usage (a caller's
//!   image without it, or a device where it would change the image's memory requirements) but
//!   takes transfer reads ([`copied`]): its level and the pass's layers over the render area are
//!   copied (`vkCmdCopyImage`, one format and sample count) into a scratch image of its format and
//!   samples that has the usage (and, for the device's sRGB decode, takes the sRGB twin's view),
//!   held by the command buffer until it completes, and the resolve reads the scratch as it reads
//!   an attachment. The copy moves each sample's bits; the one
//!   device measured to change them, RADV, writes a 16-bit float NaN as `0xfe00`
//!   (`RuntimeDriverDefect::MultisampledCopiesConvertFloatNans`), which the resolve counts as 0
//!   whatever its bits. A caller's image has the transfer usage the runtime's texture of its
//!   descriptor has (`texture/wrapped.rs`), so only an image of a format the device copies no
//!   image of has neither usage.
//!
//! From any source, an 8-bit sRGB resolve by the tables writes its result through the resolve
//! attachment's view in its UNORM twin where that image takes one ([`raw_view`],
//! `bind::ColorResolveFormat::written_raw`): each code goes out as a unorm code's float, which the
//! twin's store keeps as the code, with no table lookup and no sRGB encode. One by the device's
//! decode writes through the attachment's own view, whose store encodes.
//!
//! A layered pass (`renderTargetArrayLength` 2 or more) resolves each of its layers in a render
//! pass instance of its own, through views of that layer. Its resolve from the sample planes, which
//! hold one layer, is `NotImplemented(LayeredPass)`.
//!
//! The resolve's pipelines and passes are made once per device, format, sample count and source,
//! at the first pass that needs them, and destroyed with the device; each pass's descriptor sets
//! and framebuffers are held by its command buffer until it completes, and, where every resolve of
//! the pass reads its attachment through the attachment's own views, kept by the image it first
//! resolves into for its next pass of the same views ([`KeptResolves`]). Each resolve runs between
//! barriers from everything before and to everything after, as does each copy before them. The
//! runtime announces `shader-resolve` (performance) for the extra pass (naming the copy where the
//! resolve reads one), and `unmeasured-resolve` (precision) at more than 4 samples.

use super::binding_view::{views_as, VkView};
use super::blit::TRANSFER;
use super::command_buffer::CommandBuffer;
use super::device::{device_error, lock};
use super::encoder::EncoderCore;
use super::error::{internal, vulkan, Error, RuntimeGap};
use super::exit_guard;
use super::image_plan::{srgb_twin, ImagePlan};
use super::release;
use super::render_pass::{PassLayout, Target};
use super::render_target::{takes_input, ATTACHMENT_STAGES};
use super::resolve;
use super::resource::ResourceKey;
use super::sample_planes::{pipeline_layout, set_layout, shader_module};
use super::tracker::{Access, Barrier, Use};
use super::Device;
use crate::shader_cache::driver::Making;
use ash::vk;
use metal2vulkan::bind::{color_resolve_modules, ColorResolveFormat, PlanesCopy, ResolveSource};
use metal2vulkan::device::{DeviceProfile, DriverFact, FormatUsage};
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::format::{direct_format, ExtendedRange, WordAttachment};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::vk::{
    ComponentMapping, Format, FormatFeatureFlags2, ImageAspectFlags, ImageUsageFlags, ImageViewType,
};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

/// The most samples of a texture on Apple's GPUs, so the most whose resolve Metal measures.
pub(super) const METAL_SAMPLES: u32 = 4;

/// Whether `device`'s own `AVERAGE` resolve of a color attachment of `samples` samples writes
/// Metal's resolve, whatever its format: `DriverFact::ColorResolveAverageIsMetals`, at a count
/// Apple's GPUs make (MoltenVK's resolve is Metal's multisample resolve store action).
pub(super) fn average_is_metals(device: &Device, samples: u32) -> bool {
    samples <= METAL_SAMPLES
        && device
            .properties()
            .driver_facts
            .contains(&DriverFact::ColorResolveAverageIsMetals)
}

/// The resolve of a multisampled color attachment of the extended-range format `pf`, held in its
/// raw format's codes, which no device's resolve averages: the crate-owned one at every count (the
/// module docs). `None` for another format.
pub(super) fn extended(pf: PixelFormat) -> Option<ColorResolveFormat> {
    ExtendedRange::of(pf).and_then(ColorResolveFormat::extended_range)
}

/// The crate-owned resolve of a color attachment of pixel format `pf` held in `format` at
/// `samples`, where it resolves the attachment's own samples into its own format, at every count
/// past one: an extended-range format's ([`extended`]), and a word's ([`word`]). `None` where the
/// device resolves it in the pass.
pub(super) fn shader_rules(
    pf: PixelFormat,
    format: Format,
    samples: u32,
) -> Option<ColorResolveFormat> {
    (samples > 1).then(|| extended(pf).or_else(|| word(pf, format)))?
}

/// The resolve of a multisampled color attachment of `pf` held in `format`, its word's format
/// (`metal::format::WordAttachment`, `image_plan::word_samples`), which no device's average reads:
/// a packed 16-bit or `RGB9E5Float` word's own ([`ColorResolveFormat::word`]), and an 8-bit sRGB
/// word's format's sRGB rules ([`ColorResolveFormat::of`]), whose codes it reads and writes
/// through the views of its UNORM twin ([`raw_view`]). `None` for another format.
fn word(pf: PixelFormat, format: Format) -> Option<ColorResolveFormat> {
    let word = WordAttachment::layout(pf).filter(|w| w.word_format() == format)?;
    ColorResolveFormat::word(word).or_else(|| direct_format(pf).and_then(ColorResolveFormat::of))
}

/// Whether `target` is an attachment rendered through the view of its word
/// (`metal::format::WordAttachment`): its view is in the word's format.
fn word_view(target: &Target) -> bool {
    WordAttachment::layout(target.texture.pixel_format())
        .is_some_and(|w| w.word_format() == target.format)
}

/// The crate-owned resolve of the color attachment `source`, where the runtime resolves it so:
/// [`shader_rules`] of its format, view and samples, where the resolve may read its samples:
/// through its image's input attachment usage, or from a copy of them into a scratch image that
/// has it ([`copied`]), unless its sample planes hold them (the module docs). An extended-range or
/// word attachment's image that takes neither is refused: no other resolve reads its codes.
pub(super) fn of_attachment(
    device: &Device,
    source: &Target,
) -> Result<Option<ColorResolveFormat>, Error> {
    let pf = source.texture.pixel_format();
    match shader_rules(pf, source.format, source.samples) {
        Some(rules) if takes_input(source)? || copied(device, source)? => Ok(Some(rules)),
        Some(_) => Err(Error::Unsupported(Unsupported::not_implemented(
            Feature::Format,
        ))),
        None => Ok(None),
    }
}

/// The crate-owned resolve of a color attachment resolving into its format's sRGB twin, whose
/// encoding Metal resolves in (`resolve.rs`): the resolve attachment's format's rules, on every
/// device, since a subpass's resolve attachment is of its color attachment's own format and so no
/// device's average writes the twin's encoding. `None` where that format has no rule.
fn twin_rules(into: &Target) -> Option<ColorResolveFormat> {
    ColorResolveFormat::of(into.format)
}

/// The rules of the crate-owned resolve `resolve` takes from the color attachment `source`: the
/// resolve attachment's format's where it is the attachment's format's sRGB twin
/// ([`twin_rules`], which no device's average writes, so a format with no rule is refused here and
/// samples the resolve cannot read where it plans the pass), their sRGB channels by the device's
/// own decode and encode where it reaches the samples through a view in the sRGB format
/// ([`decodes`]), else the attachment's own ([`of_attachment`]).
pub(super) fn of_resolve(
    device: &Device,
    source: &Target,
    resolve: &resolve::Resolve,
) -> Result<Option<ColorResolveFormat>, Error> {
    if !resolve.twin {
        return of_attachment(device, source);
    }
    let rules = twin_rules(&resolve.attachment).ok_or(Error::NotImplemented(RuntimeGap::Render))?;
    let decoded = rules.device_srgb();
    match decoded.decodes_on_device() && decodes(device, source, rules.format())? {
        true => Ok(Some(decoded)),
        false => Ok(Some(rules)),
    }
}

/// Whether a twin resolve into the sRGB format `srgb` reads the color attachment `source` through
/// the device's own sRGB decode (`bind::ColorResolveFormat::device_srgb`, the module docs): the
/// device renders `srgb`, and the resolve reaches the samples through a view in it, the
/// attachment's where its image takes input attachment usage (an 8-bit UNORM texture's image
/// lists its sRGB twin's views where the device has them, `image_plan.rs`), else a copy's
/// ([`copied`]), whose scratch image takes the view ([`copy_out`]).
fn decodes(device: &Device, source: &Target, srgb: Format) -> Result<bool, Error> {
    let profile = device.compile_device().profile();
    if !profile.supports_format(srgb, FormatUsage::ColorAttachment) {
        return Ok(false);
    }
    match takes_input(source)? {
        true => Ok(views_as(source.texture.image_plan()?, srgb, false)),
        false => copied(device, source),
    }
}

/// Whether the crate-owned resolve may read the color attachment `source`, whose image takes no
/// input attachment usage, from a copy of its samples (the module docs): its image takes transfer
/// reads, and the device copies images of its format both ways, which the scratch image the copy
/// writes needs.
fn copied(device: &Device, source: &Target) -> Result<bool, Error> {
    let usage = source.texture.image_plan()?.usage;
    let profile = device.compile_device().profile();
    let copies = FormatFeatureFlags2::TRANSFER_SRC | FormatFeatureFlags2::TRANSFER_DST;
    Ok(usage.contains(ImageUsageFlags::TRANSFER_SRC)
        && device
            .shared()
            .textures
            .facts(profile)
            .optimal(source.format)
            .contains(copies))
}

/// Whether a multisampled color attachment of `pf` may resolve into its format's sRGB twin, which
/// the crate-owned resolve reads it for on every device (`resolve.rs`): the twin's format has
/// Metal's rules (`bind::ColorResolveFormat::of`). Its image, and the stand-in a pass renders it into
/// (`attachment_stand_in.rs`), take the input attachment usage that resolve reads them through
/// where it is free.
pub(super) fn twin_resolved(pf: PixelFormat, samples: u32) -> bool {
    samples > 1
        && srgb_twin(pf)
            .and_then(direct_format)
            .and_then(ColorResolveFormat::of)
            .is_some()
}

/// The view format a twin resolve reads a multisampled color attachment of `pf`, held in
/// `format`, through, where its image must list one: an 8-bit sRGB attachment's UNORM twin, whose
/// codes the resolve texture's UNORM rules average, and an 8-bit UNORM attachment's sRGB twin
/// where `profile` renders it, through which the resolve into that twin takes the device's own
/// decode ([`decodes`]). `None` where the resolve reads the attachment in its own format.
pub(super) fn twin_read_view(
    profile: &DeviceProfile,
    pf: PixelFormat,
    format: Format,
    samples: u32,
) -> Option<Format> {
    use metal2vulkan::metal::format::srgb_raw_twin;
    if !twin_resolved(pf, samples) {
        return None;
    }
    let decoded = srgb_twin(pf).and_then(direct_format).filter(|&twin| {
        ColorResolveFormat::of(twin).is_some_and(|rules| rules.device_srgb().decodes_on_device())
            && profile.supports_format(twin, FormatUsage::ColorAttachment)
    });
    let read = decoded.or_else(|| srgb_raw_twin(pf).and_then(direct_format))?;
    (read != format).then_some(read)
}

/// `plan` with input attachment usage, where the runtime may resolve its image by the crate-owned
/// resolve reading it as an input attachment: a multisampled color attachment whose format and
/// samples [`shader_rules`] resolves, or one whose format's sRGB twin it resolves into whatever
/// the device ([`twin_resolved`]). `None` where it has the usage already, or needs none.
pub(super) fn input_candidate(plan: &ImagePlan) -> Option<ImagePlan> {
    let usage = plan.usage;
    let samples = plan.samples.as_raw();
    let resolved = shader_rules(plan.pixel_format, plan.format, samples).is_some()
        || twin_resolved(plan.pixel_format, samples);
    let candidate = usage.contains(ImageUsageFlags::COLOR_ATTACHMENT)
        && !usage.contains(ImageUsageFlags::INPUT_ATTACHMENT)
        && resolved;
    candidate.then(|| ImagePlan {
        usage: usage | ImageUsageFlags::INPUT_ATTACHMENT,
        ..plan.clone()
    })
}

/// Whether the exact resolve of `rules` reaches `target` (the color attachment it reads, or the
/// resolve attachment it writes) through a view in `rules`' raw twin
/// (`ColorResolveFormat::raw_format`, the module docs): an 8-bit sRGB format's UNORM twin, where
/// the image takes a view of it (an sRGB texture's image lists its twin's formats,
/// `image_plan.rs`) and `device` renders it. A target already in the raw twin, which a twin
/// resolve's attachment is (`resolve::Resolve::twin`: the UNORM attachment of sRGB rules holds
/// their codes itself), is reached through a view in its own format. So is a target an 8-bit
/// sRGB texture renders through the view of its word ([`word_view`]), whose image lists its
/// UNORM twin's view (`image_plan.rs`): the word's codes are the twin's.
fn raw_view(device: &Device, target: &Target, rules: ColorResolveFormat) -> Result<bool, Error> {
    let raw = rules.raw_format();
    let reached = target.format == rules.format() || target.format == raw || word_view(target);
    if raw == rules.format() || !reached {
        return Ok(false);
    }
    let profile = device.compile_device().profile();
    Ok(views_as(target.texture.image_plan()?, raw, false)
        && profile.supports_format(raw, FormatUsage::ColorAttachment))
}

/// Where a resolve reads its samples (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Source {
    /// The attachment, as the resolve pass's input attachment 0: through its raw twin's view where
    /// `raw` ([`raw_view`]), else in its own format.
    Attachment { raw: bool },
    /// The attachment's sample planes, viewed in their copy format.
    Planes(PlanesCopy),
}

impl Source {
    fn bind(self) -> ResolveSource {
        match self {
            Source::Attachment { raw: false } => ResolveSource::InputAttachment,
            Source::Attachment { raw: true } => ResolveSource::RawInputAttachment,
            Source::Planes(copy) => ResolveSource::Planes(copy),
        }
    }

    /// The descriptor type of the source's binding.
    fn descriptor(self) -> vk::DescriptorType {
        match self {
            Source::Attachment { .. } => vk::DescriptorType::INPUT_ATTACHMENT,
            Source::Planes(_) => vk::DescriptorType::SAMPLED_IMAGE,
        }
    }

    /// Whether the source is the attachment, as input attachment 0.
    fn attachment(self) -> bool {
        matches!(self, Source::Attachment { .. })
    }

    /// The format of the view of the attachment the source reads, whose format is `rules`'s: its
    /// raw twin through the raw view, else its own.
    fn read_format(self, rules: ColorResolveFormat) -> Format {
        match self {
            Source::Attachment { raw: true } => rules.raw_format(),
            Source::Attachment { raw: false } | Source::Planes(_) => rules.format(),
        }
    }
}

/// One resolve's objects, the device's: for one format, sample count and source.
#[derive(Clone, Copy, Debug)]
struct ResolvePipeline {
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
}

impl ResolvePipeline {
    const NULL: ResolvePipeline = ResolvePipeline {
        vertex: vk::ShaderModule::null(),
        fragment: vk::ShaderModule::null(),
        set_layout: vk::DescriptorSetLayout::null(),
        layout: vk::PipelineLayout::null(),
        render_pass: vk::RenderPass::null(),
        pipeline: vk::Pipeline::null(),
    };

    /// Destroy every object.
    ///
    /// # Safety
    ///
    /// No pending command uses them; a null handle is a no-op.
    unsafe fn destroy(&self, raw: &ash::Device) {
        // SAFETY: the caller's contract.
        unsafe {
            release::object(raw, self.pipeline);
            release::object(raw, self.render_pass);
            release::object(raw, self.layout);
            release::object(raw, self.set_layout);
            release::object(raw, self.fragment);
            release::object(raw, self.vertex);
        }
    }
}

/// The resolves' objects one device has made (the module docs): bounded by the formats with
/// closed rules, the sample counts and the planes' copy formats.
#[derive(Default)]
pub(super) struct ColorResolveObjects {
    pipelines: Vec<((ColorResolveFormat, u32, Source), ResolvePipeline)>,
}

impl ColorResolveObjects {
    /// Destroy every object; the device is idle, and is destroyed next.
    pub(super) fn destroy(&self, raw: &ash::Device) {
        for (_, pipeline) in &self.pipelines {
            // SAFETY: the device is idle, so no pending command uses them.
            unsafe { pipeline.destroy(raw) };
        }
    }
}

impl Device {
    /// The resolve's objects for `format` at `samples` from `source`, made on first use.
    fn color_resolve_pipeline(
        &self,
        format: ColorResolveFormat,
        samples: u32,
        source: Source,
    ) -> Result<ResolvePipeline, Error> {
        let key = (format, samples, source);
        // A making's turn before the lock (`budget::helper_turn`): none where it is made.
        let _turn =
            crate::pipeline::budget::helper_turn(&self.shared().color_resolves, |objects| {
                objects.pipelines.iter().any(|(k, _)| *k == key)
            });
        let mut objects = lock(&self.shared().color_resolves);
        if let Some((_, pipeline)) = objects.pipelines.iter().find(|(k, _)| *k == key) {
            return Ok(*pipeline);
        }
        let modules = color_resolve_modules(
            self.compile_device(),
            format,
            samples,
            source.bind(),
            Feature::Format,
        )
        .map_err(device_error)?;
        let raw = self.shared().vk();
        let mut made = ResolvePipeline::NULL;
        let result = (|| {
            made.vertex = shader_module(raw, &modules.vertex)?;
            made.fragment = shader_module(raw, &modules.fragment)?;
            made.set_layout =
                set_layout(raw, &[source.descriptor()], vk::ShaderStageFlags::FRAGMENT)?;
            made.layout = pipeline_layout(raw, made.set_layout)?;
            made.render_pass = resolve_pass(raw, format, samples, source)?;
            let making = Making::new("color resolve")
                .words(&modules.vertex)
                .words(&modules.fragment)
                .text(&key);
            made.pipeline = resolve_pipeline(self, &made, making)?;
            Ok(())
        })();
        match result {
            Ok(()) => {
                objects.pipelines.push((key, made));
                Ok(made)
            }
            Err(error) => {
                // SAFETY: nothing used the objects.
                unsafe { made.destroy(raw) };
                Err(error)
            }
        }
    }
}

/// The resolve's render pass for `rules`'s format at `samples` from `source`: the multisampled
/// attachment as input attachment 0 (loaded and stored, `GENERAL`, in the format of the view the
/// source reads) where it reads it, then the single-sampled resolve attachment in the format of
/// the view the result is written through (`ColorResolveFormat::written_format`), written whole
/// over the render area (`DONT_CARE`, then `STORE`).
fn resolve_pass(
    raw: &ash::Device,
    rules: ColorResolveFormat,
    samples: u32,
    source: Source,
) -> Result<vk::RenderPass, Error> {
    let general = vk::ImageLayout::GENERAL;
    let samples_attachment = vk::AttachmentDescription::default()
        .format(source.read_format(rules).into())
        .samples(vk::SampleCountFlags::from_raw(samples))
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(general)
        .final_layout(general);
    let resolved = vk::AttachmentDescription::default()
        .format(rules.written_format().into())
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(general)
        .final_layout(general);
    let (attachments, color) = match source {
        Source::Attachment { .. } => (vec![samples_attachment, resolved], 1),
        Source::Planes(_) => (vec![resolved], 0),
    };
    let inputs = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(general)];
    let colors = [vk::AttachmentReference::default()
        .attachment(color)
        .layout(general)];
    let mut subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&colors);
    if source.attachment() {
        subpass = subpass.input_attachments(&inputs);
    }
    let subpasses = [subpass];
    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);
    // SAFETY: `info` and its arrays are live for the call.
    unsafe { crate::creation::render_pass(raw, &info) }
}

/// The resolve's pipeline in its pass: the triangle over the viewport at one sample, without
/// blending, writing every component. Its record is the one `making` names.
fn resolve_pipeline(
    device: &Device,
    made: &ResolvePipeline,
    making: Making,
) -> Result<vk::Pipeline, Error> {
    let shared = device.shared();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(made.vertex)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(made.fragment)
            .name(c"main"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let attachments = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(made.layout)
        .render_pass(made.render_pass)
        .subpass(0);
    shared.make_graphics_pipeline(making, info)
}

/// The objects one pass's resolves hold until its command buffer completes, and, kept
/// ([`KeptResolves`]), until the image keeping them lets them go and every command buffer that
/// took them completed: their descriptor sets' pool and, for each resolve and layer, its
/// framebuffer and set.
struct PassObjects {
    device: Device,
    pool: vk::DescriptorPool,
    resolves: Vec<Planned>,
}

impl Drop for PassObjects {
    fn drop(&mut self) {
        // Once the process is exiting, the objects are left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let raw = self.device.shared().vk();
        // SAFETY: no pending command uses them (a command buffer retains them until it completes);
        // the sets are freed with their pool, and a null handle is a no-op. The views they were
        // made of may have gone first: an object is destroyed in any order but after its children,
        // and its destruction accesses no other object (the Vulkan specification's object
        // lifetime: "Applications can: free or destroy objects in any order, except that parent
        // objects must: be freed only after all child objects are freed", the parent being the
        // first object of the child's creation, here the device; "When destroying or freeing an
        // object, implementations must: not access any memory or other objects that may:
        // otherwise be accessed when the object is accessed").
        unsafe {
            for resolve in &self.resolves {
                release::object(raw, resolve.framebuffer);
            }
            release::object(raw, self.pool);
        }
    }
}

/// One attachment's resolve of one layer, planned.
struct Planned {
    pipeline: ResolvePipeline,
    framebuffer: vk::Framebuffer,
    set: vk::DescriptorSet,
}

/// A pass's exact color resolves, planned before the pass records and recorded after it.
pub(super) struct ShaderResolves {
    extent: [u32; 2],
    uses: Vec<Use<ResourceKey>>,
    /// The copies of the attachments the resolves read from a scratch image ([`copied`]):
    /// `(attachment, scratch, region)` each, and their uses.
    copies: Vec<(vk::Image, vk::Image, vk::ImageCopy)>,
    copy_uses: Vec<Use<ResourceKey>>,
    objects: Arc<PassObjects>,
}

/// The most passes' resolve objects an image keeps ([`KeptResolves`]).
const KEPT_PASSES: usize = 4;

/// The resolve objects of the passes that resolved into one image ([`ShaderResolves::plan`]),
/// kept for its next pass of the same views, and the image's key in the passes' keys.
///
/// A pass whose every resolve reads its attachment through the attachment's own views (not the
/// sample planes, nor a copy of its samples, whose views are the pass's own) is kept by the image
/// it first resolves into, by everything its objects are made of ([`PassKey`]), and a later pass
/// of the same key binds the kept sets and begins the kept framebuffers instead of making its
/// own: the same descriptors of the same views, never written again. A set or framebuffer serves
/// several pending command buffers at once: neither is mutated by the commands that use it (the
/// Vulkan specification: "Any object parameters that are not labeled as externally synchronized
/// are either not mutated by the command or are internally synchronized"; `vkCmdBindDescriptorSets`
/// and `vkCmdBeginRenderPass` label only their command buffer), and a set's contents must only not
/// be altered while a command that bound it may consume them ("Once all pending uses have
/// completed, it is legal to update and reuse a descriptor set").
///
/// A key names each image by one no other image had or will have ([`KeptResolves::new`]), so a
/// view of an image that dropped is never taken for a view of another, whatever handle the
/// driver gave either. A view an image lets go while it lives (a rebind's or a storage form's new
/// `VkImage`, a detach's stand-in, a reattach) moves the device's views epoch
/// (`DeviceShared::views_epoch`): a pass kept under an older epoch is never taken, and an image's
/// kept passes all go at its next pass after the epoch moved. A kept pass's objects go once the
/// image lets go of them (as it drops, before its views; as the epoch moves; as a newer pass
/// takes the place of the oldest past [`KEPT_PASSES`]) and every command buffer that took them
/// completed.
pub(super) struct KeptResolves {
    image: ResourceKey,
    passes: Mutex<Kept<PassObjects>>,
}

impl KeptResolves {
    /// No passes kept yet, for an image made now.
    pub(super) fn new() -> KeptResolves {
        KeptResolves {
            image: ResourceKey::new(),
            passes: Mutex::new(Kept::default()),
        }
    }

    /// Let every kept pass go.
    pub(super) fn clear(&self) {
        lock(&self.passes).values.clear();
    }
}

/// What a kept pass's objects are made of ([`KeptResolves`]): the pass's extent and, for each
/// resolve and layer in the pass's order, the resolve's pipeline (format, samples and source),
/// the view it reads and the view it writes, each as its image's key and the view's description.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PassKey {
    extent: [u32; 2],
    resolves: Vec<ResolveKey>,
}

/// One resolve and layer of a [`PassKey`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResolveKey {
    pipeline: (ColorResolveFormat, u32, Source),
    read: (ResourceKey, VkView),
    written: (ResourceKey, VkView),
}

/// Values kept by key, each under the views epoch it was made in: at most [`KEPT_PASSES`], the
/// newest first, all of one epoch.
struct Kept<T> {
    epoch: u64,
    values: Vec<(PassKey, Arc<T>)>,
}

impl<T> Default for Kept<T> {
    fn default() -> Kept<T> {
        Kept {
            epoch: 0,
            values: Vec::new(),
        }
    }
}

impl<T> Kept<T> {
    /// The value kept for `key`, where the views epoch is `epoch` as it was when the value was
    /// made; a newer epoch lets every kept value go.
    fn find(&mut self, epoch: u64, key: &PassKey) -> Option<Arc<T>> {
        self.at(epoch);
        if epoch != self.epoch {
            return None;
        }
        self.values
            .iter()
            .find(|(kept, _)| kept == key)
            .map(|(_, value)| Arc::clone(value))
    }

    /// Keep `value` for `key`, made under views epoch `epoch`: not where the epoch moved since,
    /// and letting the oldest go past [`KEPT_PASSES`].
    fn keep(&mut self, epoch: u64, key: PassKey, value: Arc<T>) {
        self.at(epoch);
        if epoch != self.epoch {
            return;
        }
        if self.values.len() == KEPT_PASSES {
            self.values.pop();
        }
        self.values.insert(0, (key, value));
    }

    /// Let every kept value go where `epoch` is newer than theirs.
    fn at(&mut self, epoch: u64) {
        if epoch > self.epoch {
            self.values.clear();
            self.epoch = epoch;
        }
    }
}

/// The key of a pass resolving each of `wanted` (its attachment, its resolve attachment, its
/// format's rules and its source) over `extent` in `layers` layers ([`KeptResolves`]); `None`
/// where a resolve reads the sample planes or a copy of its samples, or its targets have other
/// layers than the pass (which [`ShaderResolves::plan`] refuses).
fn pass_key<'a>(
    wanted: impl Iterator<Item = (&'a Target, &'a Target, ColorResolveFormat, Source)>,
    extent: [u32; 2],
    layers: u32,
) -> Result<Option<PassKey>, Error> {
    let mut resolves = Vec::new();
    for (attachment, into, format, source) in wanted {
        let own_views = source.attachment()
            && attachment.layers == layers
            && into.layers == layers
            && takes_input(attachment)?;
        if !own_views {
            return Ok(None);
        }
        let read = attachment.texture.kept_resolves()?.image;
        let written = into.texture.kept_resolves()?.image;
        for layer in 0..layers {
            resolves.push(ResolveKey {
                pipeline: (format, attachment.samples, source),
                read: (
                    read,
                    view_of(attachment, layer, source.read_format(format), INPUT)?,
                ),
                written: (
                    written,
                    view_of(into, layer, format.written_format(), WRITTEN)?,
                ),
            });
        }
    }
    Ok(Some(PassKey { extent, resolves }))
}

/// The usage of a resolve's view of the samples it reads as its input attachment.
const INPUT: ImageUsageFlags = ImageUsageFlags::INPUT_ATTACHMENT;
/// The usage of a resolve's view of the attachment it writes.
const WRITTEN: ImageUsageFlags = ImageUsageFlags::COLOR_ATTACHMENT;

/// The use of `samples`' level and `layers` layers by a resolve that reads them as its input
/// attachment.
fn read_use(samples: &Target, layers: u32) -> Result<Use<ResourceKey>, Error> {
    samples.texture.image_use(
        samples.level..samples.level + 1,
        samples.layer..samples.layer + layers,
        Access::new(
            vk::PipelineStageFlags2::FRAGMENT_SHADER
                | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::INPUT_ATTACHMENT_READ
                | vk::AccessFlags2::COLOR_ATTACHMENT_READ
                | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        ),
        true,
    )
}

/// The use of `into`'s level and `layers` layers by the resolve that writes them.
fn written_use(into: &Target, layers: u32) -> Result<Use<ResourceKey>, Error> {
    into.texture.image_use(
        into.level..into.level + 1,
        into.layer..into.layer + layers,
        Access::new(
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        ),
        true,
    )
}

/// The copy of `samples`' level and `layers` layers over the render area `extent` into a scratch
/// image of its format and samples that the crate-owned resolve reads as its input attachment
/// ([`copied`]), through a view in `view` where one is named (the sRGB twin the device decodes,
/// [`decodes`]): the scratch, the copy, and its uses (the read of the attachment and the write of
/// the scratch). The scratch is `maker`'s, held until it completes.
fn copy_out(
    maker: &CommandBuffer,
    samples: &Target,
    layers: u32,
    extent: [u32; 2],
    view: Option<Format>,
) -> Result<
    (
        Target,
        (vk::Image, vk::Image, vk::ImageCopy),
        [Use<ResourceKey>; 2],
    ),
    Error,
> {
    let usage = ImageUsageFlags::COLOR_ATTACHMENT
        | ImageUsageFlags::INPUT_ATTACHMENT
        | ImageUsageFlags::TRANSFER_DST;
    let area = Target {
        extent,
        ..samples.clone()
    };
    let scratch = resolve::scratch_viewed(maker, &area, usage, samples.samples, view)?;
    let at = |t: &Target| vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: t.level,
        base_array_layer: t.layer,
        layer_count: layers,
    };
    let [width, height] = extent;
    let region = vk::ImageCopy {
        src_subresource: at(samples),
        src_offset: vk::Offset3D::default(),
        dst_subresource: at(&scratch),
        dst_offset: vk::Offset3D::default(),
        extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
    };
    let level = |t: &Target| t.level..t.level + 1;
    let on = |t: &Target| t.layer..t.layer + layers;
    let uses = [
        samples.texture.image_use(
            level(samples),
            on(samples),
            Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ),
            false,
        )?,
        scratch.texture.image_use(
            level(&scratch),
            on(&scratch),
            Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE),
            true,
        )?,
    ];
    let copy = (
        samples.texture.raw_image()?,
        scratch.texture.raw_image()?,
        region,
    );
    Ok((scratch, copy, uses))
}

/// The single-level view of `target`'s layer `layer` (counted from its first) in `format`,
/// serving `usage`.
fn view(
    target: &Target,
    layer: u32,
    format: Format,
    usage: ImageUsageFlags,
) -> Result<vk::ImageView, Error> {
    target
        .texture
        .image_view(&view_of(target, layer, format, usage)?)
}

/// The description of [`view`]'s view.
fn view_of(
    target: &Target,
    layer: u32,
    format: Format,
    usage: ImageUsageFlags,
) -> Result<VkView, Error> {
    // A 1D image's layer is a 1D view (a 3D image's planes take 2D ones).
    let view_type = match target.texture.image_plan()?.image_type {
        metal2vulkan::vk::ImageType::TYPE_1D => ImageViewType::TYPE_1D,
        _ => ImageViewType::TYPE_2D,
    };
    Ok(VkView {
        view_type,
        format,
        components: ComponentMapping::IDENTITY,
        aspect: ImageAspectFlags::COLOR,
        base_level: target.level,
        level_count: 1,
        base_layer: target.layer + layer,
        layer_count: 1,
        usage,
    })
}

impl ShaderResolves {
    /// The exact resolves of `layout`'s color attachments (its final layout) whose last render
    /// pass instance leaves `last_storage` out of its draws' subpass: each from its sample
    /// planes where they hold its samples, else from the attachment, or from a copy of its
    /// samples where its image takes no input attachment usage (the module docs). `None` where
    /// no attachment resolves so. Scratch images are `maker`'s.
    pub(super) fn plan(
        maker: &CommandBuffer,
        layout: &PassLayout,
        last_storage: &[u32],
    ) -> Result<Option<ShaderResolves>, Error> {
        let device = maker.device();
        let mut wanted = Vec::new();
        for (index, color) in layout.colors.iter().enumerate() {
            let Some(color) = color else {
                continue;
            };
            let Some(resolve) = &color.resolve else {
                continue;
            };
            let Some(format) = resolve.shader else {
                continue;
            };
            // A twin resolve's rules are the resolve attachment's format's, which the sample
            // planes' copy format does not hold the codes under: it reads the attachment itself
            // (`resolve::Resolve::twin`).
            let planes = color
                .planes
                .as_ref()
                .filter(|_| last_storage.contains(&(index as u32)) && !resolve.twin);
            // An attachment read from a copy of its samples ([`copied`]) is read in its own
            // format, or in the sRGB twin the device decodes ([`decodes`]): the scratch image
            // takes no other view. Its own is the rules' raw form where the attachment is already
            // in it, which a twin resolve's UNORM attachment under its sRGB resolve texture's
            // tables is. Under the device's decode the samples are read in the rules' format.
            let decoded = format.decodes_on_device();
            let raw = !decoded
                && raw_view(device, &color.target, format)?
                && (takes_input(&color.target)? || color.target.format == format.raw_format());
            let source = match planes {
                Some(planes) => Source::Planes(planes.copy),
                None => Source::Attachment { raw },
            };
            // The view the source reads the samples through, the rules' own form or its raw
            // twin, must be one their image takes: a twin resolve reads an sRGB attachment
            // through the UNORM twin its rules are in, and a UNORM one through its own format or,
            // under the device's decode, its sRGB twin's. A copy of the samples ([`copied`]) is
            // read in the attachment's own format, or in the sRGB twin its scratch image takes
            // for the device's decode ([`copy_out`]).
            let read_format = source.read_format(format);
            let reaches = match source.attachment() {
                false => true,
                true if takes_input(&color.target)? => {
                    views_as(color.target.texture.image_plan()?, read_format, false)
                }
                true => read_format == color.target.format || decoded,
            };
            if !reaches {
                return Err(Error::NotImplemented(RuntimeGap::Render));
            }
            // The result goes out through the resolve attachment's raw view where it has one,
            // but for the device's encode, which its own view stores.
            let format = match !decoded && raw_view(device, &resolve.attachment, format)? {
                true => format.written_raw(),
                false => format,
            };
            wanted.push((color, resolve, format, source, planes));
        }
        if wanted.is_empty() {
            return Ok(None);
        }
        // A layered pass resolves each of its layers in a render pass instance of its own. The
        // sample planes hold one layer (`sample_planes.rs`), so such a pass's resolve from them
        // is not built.
        let layers = layout.layers;
        if layers > 1 && wanted.iter().any(|w| !w.3.attachment()) {
            return Err(Error::NotImplemented(RuntimeGap::LayeredPass));
        }
        // A pass whose resolves read their attachments through their own views takes the objects
        // the image it first resolves into keeps for a pass of the same key, or keeps the ones it
        // makes ([`KeptResolves`]). The views epoch is read before any view is taken, so a pass is
        // kept under an epoch no newer than its views'.
        let epoch = device.shared().views_epoch.load(Ordering::Acquire);
        let targets = wanted
            .iter()
            .map(|w| (&w.0.target, &w.1.attachment, w.2, w.3));
        let kept = match pass_key(targets, layout.extent, layers)? {
            Some(key) => Some((wanted[0].1.attachment.texture.kept_resolves()?, key)),
            None => None,
        };
        if let Some((memo, key)) = &kept {
            if let Some(objects) = lock(&memo.passes).find(epoch, key) {
                let mut uses = Vec::with_capacity(2 * wanted.len());
                for (color, resolve, ..) in &wanted {
                    uses.push(read_use(&color.target, layers)?);
                    uses.push(written_use(&resolve.attachment, layers)?);
                }
                device.shared().counts.color_resolve_pass_kept();
                return Ok(Some(ShaderResolves {
                    extent: layout.extent,
                    uses,
                    copies: Vec::new(),
                    copy_uses: Vec::new(),
                    objects,
                }));
            }
        }
        let raw = device.shared().vk();
        let count = wanted.len() as u32 * layers;
        let mut objects = PassObjects {
            device: device.clone(),
            pool: vk::DescriptorPool::null(),
            resolves: Vec::with_capacity(count as usize),
        };
        let inputs = wanted.iter().filter(|w| w.3.attachment()).count() as u32 * layers;
        let mut sizes = Vec::new();
        for (ty, n) in [
            (vk::DescriptorType::INPUT_ATTACHMENT, inputs),
            (vk::DescriptorType::SAMPLED_IMAGE, count - inputs),
        ] {
            if n > 0 {
                sizes.push(vk::DescriptorPoolSize::default().ty(ty).descriptor_count(n));
            }
        }
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(count)
            .pool_sizes(&sizes);
        // SAFETY: `info` and its array are live for the call.
        objects.pool = unsafe { raw.create_descriptor_pool(&info, None) }.map_err(vulkan)?;
        let general = vk::ImageLayout::GENERAL;
        let mut uses = Vec::new();
        let mut copies = Vec::new();
        let mut copy_uses = Vec::new();
        for (color, resolve, format, source, planes) in wanted {
            let attachment = &color.target;
            let into = &resolve.attachment;
            if attachment.layers != layers || into.layers != layers {
                return Err(internal("an exact resolve of other layers than its pass"));
            }
            // An attachment whose image takes no input attachment usage is read from a copy of
            // its samples ([`copied`]).
            let read_copy = source.attachment() && !takes_input(attachment)?;
            let scratch = match read_copy {
                true if copied(device, attachment)? => {
                    let read_format = source.read_format(format);
                    let view = (read_format != attachment.format).then_some(read_format);
                    let (scratch, copy, used) =
                        copy_out(maker, attachment, layers, layout.extent, view)?;
                    copies.push(copy);
                    copy_uses.extend(used);
                    Some(scratch)
                }
                // A twin resolve is the one the runtime draws whatever the device's average, so
                // an attachment it can neither read nor copy is a resolve the runtime does not
                // build, not an invariant ([`of_resolve`] gates every other one).
                true if resolve.twin => return Err(Error::NotImplemented(RuntimeGap::Render)),
                true => {
                    return Err(internal(
                        "an exact resolve of an attachment it can neither read nor copy",
                    ))
                }
                false => None,
            };
            let samples = scratch.as_ref().unwrap_or(attachment);
            let pipeline = device.color_resolve_pipeline(format, samples.samples, source)?;
            match (source, planes) {
                (Source::Planes(_), Some(planes)) => uses.push(planes.image_use(
                    Access::new(
                        vk::PipelineStageFlags2::FRAGMENT_SHADER,
                        vk::AccessFlags2::SHADER_SAMPLED_READ,
                    ),
                    false,
                )?),
                (Source::Attachment { .. }, _) => uses.push(read_use(samples, layers)?),
                (Source::Planes(_), None) => {
                    return Err(internal("a planes resolve without planes"))
                }
            }
            uses.push(written_use(into, layers)?);
            // Each layer of the pass: one set and framebuffer of that layer's views.
            for layer in 0..layers {
                let layouts = [pipeline.set_layout];
                let allocate = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(objects.pool)
                    .set_layouts(&layouts);
                // SAFETY: the pool holds a set per resolve and layer, of its source's descriptor
                // (counted above).
                let set = unsafe { raw.allocate_descriptor_sets(&allocate) }
                    .map_err(vulkan)?
                    .first()
                    .copied()
                    .ok_or_else(|| internal("the driver allocated no set"))?;
                let written = view(into, layer, format.written_format(), WRITTEN)?;
                let (read, attachments) = match (source, planes) {
                    (Source::Planes(copy), Some(planes)) => {
                        let read = planes.view(copy.format(), ImageUsageFlags::SAMPLED)?;
                        (read, vec![written])
                    }
                    (Source::Attachment { .. }, _) => {
                        let read = view(samples, layer, source.read_format(format), INPUT)?;
                        (read, vec![read, written])
                    }
                    (Source::Planes(_), None) => {
                        return Err(internal("a planes resolve without planes"))
                    }
                };
                let info = [vk::DescriptorImageInfo::default()
                    .image_view(read)
                    .image_layout(general)];
                // SAFETY: the set was just allocated and no command uses it; the view is live
                // (the command buffer retains its texture).
                unsafe {
                    raw.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(0)
                            .descriptor_type(source.descriptor())
                            .image_info(&info)],
                        &[],
                    )
                };
                let [width, height] = layout.extent;
                let info = vk::FramebufferCreateInfo::default()
                    .render_pass(pipeline.render_pass)
                    .attachments(&attachments)
                    .width(width)
                    .height(height)
                    .layers(1);
                // SAFETY: `info` and its array are live for the call; the views are live.
                let framebuffer = unsafe { crate::creation::framebuffer(raw, &info) }?;
                objects.resolves.push(Planned {
                    pipeline,
                    framebuffer,
                    set,
                });
            }
        }
        device.shared().counts.color_resolve_pass_made();
        let objects = Arc::new(objects);
        if let Some((memo, key)) = kept {
            lock(&memo.passes).keep(epoch, key, Arc::clone(&objects));
        }
        Ok(Some(ShaderResolves {
            extent: layout.extent,
            uses,
            copies,
            copy_uses,
            objects,
        }))
    }

    /// Record the resolves in `core`'s command buffer after the pass, between barriers from
    /// everything before and to everything after, each copy of an attachment's samples first,
    /// between barriers too, and hold their objects until the command buffer completes.
    pub(super) fn record(self, core: &mut EncoderCore) {
        let ShaderResolves {
            extent,
            uses,
            copies,
            copy_uses,
            objects,
        } = self;
        let [width, height] = extent;
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        core.barrier(Barrier::everything());
        if !copies.is_empty() {
            core.command(TRANSFER, &copy_uses, |raw, command_buffer| {
                let general = vk::ImageLayout::GENERAL;
                for &(from, to, region) in &copies {
                    // SAFETY: the encoder's command buffer is recording outside a render pass;
                    // both images are live until it completes (the encoder retains the
                    // attachment's texture, and the command buffer the scratch's) and rest in
                    // `GENERAL`; the region is the attachment's level and the pass's layers over
                    // the render area, in both images, of one format and sample count.
                    unsafe {
                        raw.cmd_copy_image(command_buffer, from, general, to, general, &[region])
                    };
                }
            });
            core.barrier(Barrier::everything());
        }
        core.command(ATTACHMENT_STAGES, &uses, |raw, command_buffer| {
            for resolve in &objects.resolves {
                let begin = vk::RenderPassBeginInfo::default()
                    .render_pass(resolve.pipeline.render_pass)
                    .framebuffer(resolve.framebuffer)
                    .render_area(area);
                // SAFETY: the encoder's command buffer is recording outside a render pass; the
                // pipeline, framebuffer, set and the images they view are live until it completes
                // (the device keeps the pipeline, this the framebuffer and set, the encoder the
                // textures and planes); every image rests in `GENERAL`. A kept framebuffer and
                // set are of views the textures still present ([`KeptResolves`]).
                unsafe {
                    raw.cmd_begin_render_pass(command_buffer, &begin, vk::SubpassContents::INLINE);
                    raw.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        resolve.pipeline.pipeline,
                    );
                    raw.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        resolve.pipeline.layout,
                        0,
                        &[resolve.set],
                        &[],
                    );
                    raw.cmd_set_viewport(command_buffer, 0, &[viewport]);
                    raw.cmd_set_scissor(command_buffer, 0, &[area]);
                    raw.cmd_draw(command_buffer, 3, 1, 0, 0);
                    raw.cmd_end_render_pass(command_buffer);
                }
            }
        });
        core.barrier(Barrier::everything());
        core.retain(objects);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pass resolving a view of image `read` into one of image `written`, the same views but
    /// for their images.
    fn key(read: ResourceKey, written: ResourceKey) -> PassKey {
        let view = VkView {
            view_type: ImageViewType::TYPE_2D,
            format: Format::R8G8B8A8_UNORM,
            components: ComponentMapping::IDENTITY,
            aspect: ImageAspectFlags::COLOR,
            base_level: 0,
            level_count: 1,
            base_layer: 0,
            layer_count: 1,
            usage: INPUT,
        };
        let format = ColorResolveFormat::of(Format::R8G8B8A8_UNORM).expect("RGBA8's rules");
        PassKey {
            extent: [4, 4],
            resolves: vec![ResolveKey {
                pipeline: (format, 4, Source::Attachment { raw: false }),
                read: (read, view),
                written: (
                    written,
                    VkView {
                        usage: WRITTEN,
                        ..view
                    },
                ),
            }],
        }
    }

    /// A pass kept of a view of an image that dropped is never taken for the same view of an
    /// image made after it, whatever handle the driver gives the new view (it may give the
    /// dropped one's again): a key names an image by a key no other image had, and no handle.
    #[test]
    fn a_new_image_s_view_never_takes_a_dropped_image_s_kept_pass() {
        let written = KeptResolves::new();
        let dropped = KeptResolves::new();
        let dropped_key = dropped.image;
        let mut kept = Kept::default();
        kept.keep(1, key(dropped_key, written.image), Arc::new(()));
        assert!(kept.find(1, &key(dropped_key, written.image)).is_some());
        drop(dropped);
        let made_after = KeptResolves::new();
        assert_ne!(made_after.image, dropped_key);
        assert!(kept
            .find(1, &key(made_after.image, written.image))
            .is_none());
    }

    /// Once an image let views go (the views epoch moved), a kept pass is never taken but made
    /// again, every kept one is let go, and one made under the older epoch is not kept.
    #[test]
    fn a_moved_views_epoch_makes_the_pass_again() {
        let (read, written) = (KeptResolves::new().image, KeptResolves::new().image);
        let objects = Arc::new(());
        let mut kept = Kept::default();
        kept.keep(3, key(read, written), Arc::clone(&objects));
        assert!(kept.find(3, &key(read, written)).is_some());
        assert!(kept.find(4, &key(read, written)).is_none());
        assert_eq!(
            Arc::strong_count(&objects),
            1,
            "the kept pass was not let go"
        );
        kept.keep(3, key(read, written), Arc::clone(&objects));
        assert!(kept.find(4, &key(read, written)).is_none());
        kept.keep(4, key(read, written), Arc::clone(&objects));
        assert!(kept.find(4, &key(read, written)).is_some());
    }

    /// An image keeps at most `KEPT_PASSES` passes, letting the oldest go first.
    #[test]
    fn an_image_keeps_its_newest_passes() {
        let written = KeptResolves::new().image;
        let reads: Vec<_> = (0..=KEPT_PASSES)
            .map(|_| KeptResolves::new().image)
            .collect();
        let mut kept = Kept::default();
        for &read in &reads {
            kept.keep(0, key(read, written), Arc::new(()));
        }
        assert!(kept.find(0, &key(reads[0], written)).is_none());
        for &read in &reads[1..] {
            assert!(kept.find(0, &key(read, written)).is_some());
        }
    }
}
