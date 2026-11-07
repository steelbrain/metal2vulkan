//! Framebuffer fetch (its fourth part).
//!
//! Metal orders a fragment function's read of its color attachment after every earlier fragment
//! of the pixel, within a draw and across draws. The compile layer emits one of two exact forms
//! (`lower::fragment::fetch`), by what the compile device has:
//!
//! - **Input attachments** (`GraphicsState::input_attachments`) with rasterization-order
//!   attachment access (`GraphicsState::rasterization_order_attachment_access`), where the device
//!   has `rasterizationOrderColorAttachmentAccess` (both lavapipes of the venues).
//! - **Storage attachments in the ordered pixel interlock** (`GraphicsState::storage_attachments`),
//!   where it has `fragmentShaderPixelInterlock` alone (RADV, ANV, NVIDIA and MoltenVK).
//!
//! Where neither form is open to the pipeline (a device with neither feature, or a pipeline state
//! the storage form cannot keep exact, such as a blending fetched attachment), the compile layer
//! refuses the pipeline: an unordered read could see a pixel before an earlier fragment's write.
//!
//! The input attachment form ([`through_inputs`]): a pass cannot foresee which of its draws
//! fetch, so on such a device every color render target is also an input attachment
//! (`image_plan.rs`; a caller's image made without that usage is not, `texture/wrapped.rs`, and
//! a pass that fetches it renders it into a scratch image that has it, `attachment_stand_in.rs`),
//! every pass's draws' subpass reads each color attachment as the input attachment of its index
//! with rasterization-order access (`render_target.rs`), and every
//! graphics pipeline made for such a subpass orders its color attachment accesses so
//! (`pipeline/graphics.rs`): a fetch then reads what each earlier fragment of the pixel wrote,
//! whichever pipeline drew it. Every attachment rests in `GENERAL`, the layout an attachment
//! read as both needs.
//!
//! The storage form ([`through_storage`]): a draw's fetched attachments are storage images it
//! reads and writes in the interlock, which its subpass leaves out (`render_target.rs`); a pass
//! whose draws change an attachment's role splits into render pass instances (`render.rs`). The
//! fetched attachment's image needs `STORAGE` usage, which a texture cannot foresee either, and
//! which costs some devices: on Apple's GPU (MoltenVK) it drops the image's lossless compression,
//! measured as a sampled read three to five times slower, while RADV and NVIDIA lay out every
//! target measured alike, and ANV its 2048x2048 `RGBA8Unorm` and `RGBA16Float` targets but not a
//! 16x16 `R32Float` one. So each device answers for each image: a color render target takes
//! `STORAGE` where its image's memory requirements are the same with it and without it
//! ([`Device::with_free_usage`], remembered by description in [`UsageAnswers`]), and a pass that
//! fetches a texture without it (a caller's image among them) renders the texture into a scratch
//! image that has it (`attachment_stand_in.rs`). A texture Metal may write from a shader has
//! `STORAGE` anyway.
//!
//! The interlock orders the overlapping fragments of a draw, and Vulkan orders those of
//! consecutive draws of a subpass alike (`render.rs`'s `InterlockedStorage`, which quotes the
//! sentences). Even so, between two draws of one render pass instance, a draw that reads or writes
//! a color in the storage form after an earlier draw of the instance wrote one there records after
//! a barrier from the fragment shader's writes (`render_target::between_storage_draws`), as a draw
//! that reaches the fragment imageblock's storage does: RADV on an AMD 780M (Mesa 26.2.4) does not
//! order such draws by the interlock alone, and a memoryless color's copy through a fetch read
//! texels as they were before the blends of the draws just before it. Not on a driver measured to
//! order them (`DriverFact::InterlockOrdersDraws`: ANV on the Iris Xe), where the colors' barrier
//! goes, about 15,000 GPU cycles a draw there.
//!
//! An 8-bit sRGB attachment (with the compile form) is read and written through a storage view in
//! the unsigned-integer format of its texel size ([`storage_view_format`]): Metal's tile keeps the
//! attachment's 8-bit codes between draws, and the compile layer decodes and encodes them itself. A
//! texture is not made mutable to that view, so a pass that fetches one renders it into a scratch
//! image that is, with extended usage (its `STORAGE` is the view's), and a memoryless one into such
//! a backing; the pass's draws that do not fetch it render it through a view in its own format. A
//! multisampled pass's sample planes take the view too (`sample_planes.rs`). A device that stores
//! no image of the view's format, or does not reinterpret the attachment's components as the
//! view's, refuses the fetching draw, typed `Unsupported`.
//!
//! A BGRA-order attachment (`BGRA8Unorm`, `BGR10A2Unorm`) on a device that accesses its format
//! without one in neither direction (ANV; SPIR-V names no image format for it) is read and written
//! through a storage view in its RGBA-order twin (`R8G8B8A8_UNORM`, `A2B10G10R10_UNORM_PACK32`),
//! the compile layer exchanging red and blue: one decision for both layers
//! (`StorageAttachmentUse::view_alias_of`, with the device's profile), so a device that reads the
//! format without one (RADV, NVIDIA, MoltenVK) views it in its own format, as before. A color
//! render target's free `STORAGE` ([`storage_candidate`]) is mutable to the twin as well where
//! that costs the image nothing, so the common pass fetches it in place; a texture Metal may write
//! lists the twin already (`metal::format::storage_view_formats`); any other renders into a
//! scratch, backing or sample planes made mutable to it, as the sRGB view's. Through input attachments the device
//! decodes the codes the function reads, and the compile layer takes that value back to its code by
//! Metal's encode and decodes the code by Metal's table, so the function reads Metal's value there
//! too.
//!
//! In a multisampled pass the compile layer reads the attachment per sample, in either form. The
//! runtime draws the input attachment form where the attachment's image takes input attachment
//! usage. On a device without multisampled storage images ([`through_planes`]: ANV, MoltenVK) the
//! storage form reads and writes the attachment's sample planes, which the runtime keeps for the
//! pass and copies the samples into and out of (`sample_planes.rs`). Those copies read and write
//! the attachment's image sampled, through a view in the unsigned-integer format of its texel size;
//! a multisampled color render target takes that usage by the same rule as `STORAGE`
//! ([`planes_candidate`]), and a pass that fetches a texture without it renders into a multisampled
//! scratch image that has it. A device with multisampled storage images
//! ([`through_multisampled_storage`]: RADV, NVIDIA) takes the storage form on the attachment's own
//! multisampled image: a multisampled color render target takes `STORAGE` by the single-sampled
//! rule where the device stores its format at its sample count ([`storage_candidate`]), and a pass
//! that fetches a texture without it, or a memoryless one, renders into a multisampled scratch
//! image or backing that has it, which an image copy fills and empties at the same sample count
//! (`attachment_stand_in.rs`).

use super::color_resolve;
use super::depth_stencil_scratch;
use super::device::lock;
use super::error::Error;
use super::image_plan::{hiz_candidate, ImagePlan};
use super::key_hash::KeyMap;
use super::Device;
use metal2vulkan::bind::PlanesCopy;
use metal2vulkan::device::{DeviceProfile, DriverFact, FormatUsage, VkFeature, VkProperty};
use metal2vulkan::error::Capability;
use metal2vulkan::metal::format::WordAttachment;
use metal2vulkan::metal::TextureType;
use metal2vulkan::pipeline::StorageAttachmentUse;
use metal2vulkan::vk::{Format, ImageCreateFlags, ImageUsageFlags, NumericFormat};
use std::collections::BTreeMap;

/// Whether the compile layer reads `profile`'s fetched color attachments as input attachments
/// in rasterization order (`lower::fragment::fetch`): where the device has
/// `rasterizationOrderColorAttachmentAccess`.
pub(crate) fn through_inputs(profile: &DeviceProfile) -> bool {
    profile.fragment().rasterization_order_color_access
}

/// Whether a render pass on `profile` of `samples` samples, fetching through input attachments
/// where `fetch_inputs`, may hold an extended-range color's `half` shadow
/// (`render/extended_shadow.rs`): every pass that fetches in the pixel interlock, whose storage
/// form reads and writes it; and one that fetches through input attachments on a device with
/// `fragmentShaderPixelInterlock` and `fragmentStoresAndAtomics` that stores the shadow's format,
/// and at more than one sample multisampled storage images, whose functions read and write the
/// shadow beside the input attachment in the interlock (`lower::fragment`'s `input_shadow`).
pub(crate) fn holds_extended_shadows(
    profile: &DeviceProfile,
    fetch_inputs: bool,
    samples: u32,
) -> bool {
    if !fetch_inputs {
        return true;
    }
    profile.fragment().pixel_interlock
        && profile
            .enabled()
            .features
            .contains(&VkFeature::FragmentStoresAndAtomics)
        && profile.supports_format(Format::R16G16B16A16_SFLOAT, FormatUsage::StorageImage)
        && (samples <= 1 || profile.images().storage_multisample)
}

/// Whether the compile layer reads `profile`'s fetched color attachments as storage images in the
/// ordered pixel interlock (`lower::fragment::fetch`): where it has `fragmentShaderPixelInterlock`
/// without rasterization-order color attachment access.
pub(crate) fn through_storage(profile: &DeviceProfile) -> bool {
    let fragment = profile.fragment();
    !fragment.rasterization_order_color_access && fragment.pixel_interlock
}

/// Whether `profile`'s device renders some color format only through the view of its word
/// (`WordAttachment::of`) and the compile layer does not build that word's fetch
/// (`NotImplemented(Format)`): an 8-bit sRGB word (the one- and two-channel 8-bit sRGB formats on
/// NVIDIA and ANV) or an `RGB9E5Float` one, whose decodes a read does not take.
///
/// A packed 16-bit word's fetch is built: the read decodes the word into the value Metal's
/// programmable blending hands the function (`lower::fragment::fetch`,
/// `stage_io::word_color::decode`), so a format the device renders only through such a word
/// (`BGRA4Unorm` on MoltenVK) leaves the device's framebuffer fetch whole. The question the
/// `ProgrammableBlending` family row asks is this one, not whether any format renders through a
/// word at all.
pub(crate) fn renders_through_words(profile: &DeviceProfile) -> bool {
    metal2vulkan::metal::PixelFormat::ALL.iter().any(|&pf| {
        matches!(
            WordAttachment::of(pf, profile),
            Some(WordAttachment::Srgb8 { .. } | WordAttachment::Rgb9e5)
        )
    })
}

/// Whether the compile layer reads `profile`'s fetched color attachments of a pass of `samples`
/// samples as sample planes (`StorageAttachmentView::SamplePlanes`): in the pixel interlock, at
/// more than one sample, on a device without multisampled storage images.
pub(crate) fn through_planes(profile: &DeviceProfile, samples: u32) -> bool {
    through_storage(profile) && samples > 1 && !profile.images().storage_multisample
}

/// Whether the compile layer reads `profile`'s fetched color attachments of a pass of `samples`
/// samples as the attachment's own multisampled storage image (`StorageAttachmentView::Attachment`
/// at more than one sample): in the pixel interlock, at more than one sample, on a device with
/// multisampled storage images. Link refuses the pipeline where the count is not among
/// `storageImageSampleCounts`.
pub(crate) fn through_multisampled_storage(profile: &DeviceProfile, samples: u32) -> bool {
    through_storage(profile) && samples > 1 && profile.images().storage_multisample
}

/// Whether `profile` stores images of `samples` samples: every device at one sample; at more,
/// one with multisampled storage images whose `storageImageSampleCounts` has the count.
pub(crate) fn stores_samples(profile: &DeviceProfile, samples: u32) -> bool {
    samples == 1
        || (profile.images().storage_multisample
            && profile
                .enabled()
                .properties
                .contains(&VkProperty::StorageImageSampleCount(samples)))
}

/// The format of a fetching draw's storage view of a color attachment of `format` in the pixel
/// interlock on `profile` (`StorageAttachmentUse::view_format`), the compile layer's own choice
/// (`StorageAttachmentUse::view_alias_of`): the unsigned-integer format of an 8-bit sRGB
/// format's texel size, through which the compile layer reads and writes the attachment's codes
/// and decodes and encodes them itself, as Metal's tile keeps them; a BGRA-order format's
/// RGBA-order twin on a device that accesses the format without one in neither direction (ANV),
/// through which it reads and writes the texels with red and blue exchanged; `format` itself for
/// any other, and on any other device. A pass renders such an attachment, wherever a draw fetches
/// it, into an image that takes this view: its own where it does ([`views_in`]), else a stand-in
/// made mutable to it (`attachment_stand_in.rs`).
pub(crate) fn storage_view_format(profile: &DeviceProfile, format: Format) -> Format {
    StorageAttachmentUse::view_alias_of(format, profile).unwrap_or(format)
}

/// Whether an image of `plan` takes a view in `format`: its own format, or a mutable image's
/// view format (any of its texel size where it lists none).
pub(crate) fn views_in(plan: &ImagePlan, format: Format) -> bool {
    format == plan.format
        || (plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT)
            && (plan.view_formats.is_empty() || plan.view_formats.contains(&format)))
}

/// Whether a fetching draw in the interlock reads and writes an attachment that renders in
/// `format` through a storage view of its image of `plan` ([`storage_view_format`]): the image is
/// that format, or the attachment a view of it in a format `profile` stores, or the image takes
/// the view ([`views_in`]).
pub(crate) fn stores_in_place(profile: &DeviceProfile, format: Format, plan: &ImagePlan) -> bool {
    let view = storage_view_format(profile, format);
    if view == format {
        format == plan.format || profile.supports_format(format, FormatUsage::StorageImage)
    } else {
        views_in(plan, view)
    }
}

/// The answers remembered at most: past this many, the least recently asked goes.
pub(crate) const ANSWERS: usize = 1024;

/// A device's answers, by image description and the description with a usage added, to whether
/// the usage costs a color render target's image nothing (the module docs): what a framebuffer
/// fetch's form needs, `STORAGE` ([`storage_candidate`]) or what the sample planes' copies need
/// ([`planes_candidate`]), and the input attachment usage the exact resolve reads through
/// (`color_resolve::input_candidate`); for a memoryless attachment's input attachment usage,
/// whether the device makes the image ([`Device::with_free_usage`]). Found by a hash of the
/// description and the candidate. Bounded: past [`ANSWERS`], the least recently asked goes first,
/// so the descriptions a program makes every frame keep their answers through a run of
/// descriptions made once.
#[derive(Default)]
pub(crate) struct UsageAnswers {
    /// Each answer and when it was last asked, by description and candidate.
    answers: KeyMap<ImagePlan, KeyMap<ImagePlan, (bool, u64)>>,
    /// The descriptions and candidates answered, by when each was last asked.
    asked: BTreeMap<u64, (ImagePlan, ImagePlan)>,
    /// When the next ask is.
    clock: u64,
}

impl UsageAnswers {
    /// The answer for an image of `plan` given `candidate`'s usage, if remembered, now the most
    /// recently asked.
    pub(crate) fn answer(&mut self, plan: &ImagePlan, candidate: &ImagePlan) -> Option<bool> {
        let (free, asked) = self.answers.get_mut(plan)?.get_mut(candidate)?;
        self.clock += 1;
        if let Some(pair) = self.asked.remove(&*asked) {
            self.asked.insert(self.clock, pair);
        }
        *asked = self.clock;
        Some(*free)
    }

    /// Remember `free` for an image of `plan` given `candidate`'s usage, the most recently asked.
    pub(crate) fn remember(&mut self, plan: ImagePlan, candidate: ImagePlan, free: bool) {
        // Another thread may have remembered it since this one asked.
        if self.answer(&plan, &candidate).is_some() {
            if let Some((kept, _)) =
                (self.answers.get_mut(&plan)).and_then(|candidates| candidates.get_mut(&candidate))
            {
                *kept = free;
            }
            return;
        }
        self.clock += 1;
        if !self.answers.contains_key(&plan) {
            self.answers.insert(plan.clone(), KeyMap::default());
        }
        if let Some(candidates) = self.answers.get_mut(&plan) {
            candidates.insert(candidate.clone(), (free, self.clock));
        }
        self.asked.insert(self.clock, (plan, candidate));
        if self.asked.len() > ANSWERS {
            let Some((_, (plan, candidate))) = self.asked.pop_first() else {
                return;
            };
            if let Some(candidates) = self.answers.get_mut(&plan) {
                candidates.remove(&candidate);
                if candidates.is_empty() {
                    self.answers.remove(&plan);
                }
            }
        }
    }
}

/// `plan` with the usage its fetch's form needs, where a framebuffer fetch through the interlock
/// may read its image (the module docs): [`storage_candidate`] for a single-sampled color
/// attachment and a multisampled one of a device with multisampled storage images,
/// [`planes_candidate`] for a multisampled one of a device without them. `None` where the plan
/// has it already or cannot take it.
pub(crate) fn fetch_candidate(plan: &ImagePlan, profile: &DeviceProfile) -> Option<ImagePlan> {
    storage_candidate(plan, profile).or_else(|| planes_candidate(plan, profile))
}

/// `plan` with `STORAGE` usage, where a framebuffer fetch through the interlock may read its
/// image (the module docs): a color attachment that is not transient, of a format the device
/// stores, on a device whose fetch is that form; single-sampled, or multisampled where the fetch
/// reads the attachment's own multisampled image ([`through_multisampled_storage`]) and the device
/// stores images of its samples ([`stores_samples`]). Where a fetching draw views the image in
/// another format ([`storage_view_format`]: a BGRA-order format's RGBA-order twin on ANV) that
/// the device stores and the image does not take ([`views_in`]), the candidate is mutable to it
/// too, its own format and the view's listed (beside those it lists already), so a pass fetches
/// it in place. `None` where the plan has `STORAGE` already (a texture Metal may write) or cannot
/// take it.
pub(crate) fn storage_candidate(plan: &ImagePlan, profile: &DeviceProfile) -> Option<ImagePlan> {
    let usage = plan.usage;
    let samples = plan.samples.as_raw();
    let own_image = samples == 1 || through_multisampled_storage(profile, samples);
    // A packed 16-bit render target the device renders through its word's view
    // (`image_plan.rs`), which a fetching draw stores through too: `STORAGE` is that view's, which
    // the image's extended usage lets it hold.
    let word = WordAttachment::of(plan.pixel_format, profile)
        .map(WordAttachment::word_format)
        .filter(|&w| {
            plan.flags.contains(ImageCreateFlags::EXTENDED_USAGE)
                && views_in(plan, w)
                && profile.supports_format(w, FormatUsage::StorageImage)
        });
    let candidate = through_storage(profile)
        && usage.contains(ImageUsageFlags::COLOR_ATTACHMENT)
        && !usage.intersects(ImageUsageFlags::STORAGE | ImageUsageFlags::TRANSIENT_ATTACHMENT)
        && own_image
        && stores_samples(profile, samples)
        && (profile.supports_format(plan.format, FormatUsage::StorageImage) || word.is_some());
    if !candidate {
        return None;
    }
    let mut candidate = ImagePlan {
        usage: usage | ImageUsageFlags::STORAGE,
        ..plan.clone()
    };
    // An 8-bit sRGB format's unsigned-integer view stays off a texture (the module docs).
    let srgb = plan
        .format
        .block()
        .is_some_and(|b| b.color == Some(NumericFormat::Srgb));
    let view = storage_view_format(profile, plan.format);
    if !srgb && !views_in(plan, view) && profile.supports_format(view, FormatUsage::StorageImage) {
        candidate.flags |= ImageCreateFlags::MUTABLE_FORMAT;
        if candidate.view_formats.is_empty() {
            candidate.view_formats.push(plan.format);
        }
        candidate.view_formats.push(view);
    }
    Some(candidate)
}

/// Whether the sample planes' copies (`sample_planes.rs`) read and write `plan`'s image directly:
/// it is sampled and may be viewed in the [`PlanesCopy`] format of its texel size (mutable, with
/// no format list or one that lists it); an `A8_UNORM` image, whose copy is in its own format
/// ([`PlanesCopy::Alpha8`]), needs no other view.
pub(crate) fn planes_copies(plan: &ImagePlan) -> bool {
    let Some(copy) = PlanesCopy::for_planes(plan.format) else {
        return false;
    };
    plan.usage.contains(ImageUsageFlags::SAMPLED)
        && (!copy.moves_bits()
            || (plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT)
                && (plan.view_formats.is_empty() || plan.view_formats.contains(&copy.format()))))
}

/// What a multisampled color image of `format` lacks on `profile` for the sample planes' copies
/// to read and write it (`sample_planes.rs`), the first of: its own format sampled (no extended
/// usage is asked), and the [`PlanesCopy`] format of its texel size sampled and a color
/// attachment. `None` where it lacks nothing, and where `format` has no copy format.
pub(crate) fn planes_copies_gap(format: Format, profile: &DeviceProfile) -> Option<Capability> {
    let copy = PlanesCopy::for_planes(format)?;
    [
        (format, FormatUsage::SampledImage),
        (copy.format(), FormatUsage::SampledImage),
        (copy.format(), FormatUsage::ColorAttachment),
    ]
    .into_iter()
    .find(|&(format, usage)| !profile.supports_format(format, usage))
    .map(|(format, usage)| Capability::Format { format, usage })
}

/// The usage and format list a multisampled color image takes for the sample planes' copies
/// (`sample_planes.rs`): `plan` sampled and mutable with the [`PlanesCopy`] format of its texel
/// size among its view formats, or sampled alone where the copy is in its own format
/// (`A8_UNORM`'s). `None` where the device cannot run the copies on it: no copy format, or a
/// format usage it lacks ([`planes_copies_gap`]).
pub(crate) fn with_planes_copies(plan: &ImagePlan, profile: &DeviceProfile) -> Option<ImagePlan> {
    let copy = PlanesCopy::for_planes(plan.format)?;
    if planes_copies_gap(plan.format, profile).is_some() {
        return None;
    }
    let mut candidate = plan.clone();
    candidate.usage |= ImageUsageFlags::SAMPLED;
    // `A8_UNORM`'s copy is in its own format, which views it as it is.
    if !copy.moves_bits() {
        return Some(candidate);
    }
    let any_format =
        plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT) && plan.view_formats.is_empty();
    candidate.flags |= ImageCreateFlags::MUTABLE_FORMAT;
    if !any_format {
        if candidate.view_formats.is_empty() {
            candidate.view_formats.push(plan.format);
        }
        if !candidate.view_formats.contains(&copy.format()) {
            candidate.view_formats.push(copy.format());
        }
    }
    Some(candidate)
}

/// `plan` with what the sample planes' copies need ([`with_planes_copies`]), where a framebuffer
/// fetch reads its image as sample planes (the module docs): a multisampled color attachment
/// that is not transient, on a device that reads it so ([`through_planes`]). `None` where the
/// plan has it already or cannot take it.
pub(crate) fn planes_candidate(plan: &ImagePlan, profile: &DeviceProfile) -> Option<ImagePlan> {
    let usage = plan.usage;
    let candidate = through_planes(profile, plan.samples.as_raw())
        && usage.contains(ImageUsageFlags::COLOR_ATTACHMENT)
        && !usage.contains(ImageUsageFlags::TRANSIENT_ATTACHMENT)
        && !planes_copies(plan);
    if !candidate {
        return None;
    }
    with_planes_copies(plan, profile)
}

impl Device {
    /// `plan`, an image of Metal's `texture_type`, with the sampled usage that keeps a
    /// single-sample depth attachment's hierarchical depth in `GENERAL` on a driver measured to
    /// need it (`image_plan::hiz_candidate`), wherever the device makes the image, whatever its
    /// size; then with the usage its framebuffer fetch's form
    /// needs ([`fetch_candidate`]), or a multisampled depth/stencil attachment's copies without
    /// colour copies need (`depth_stencil_scratch::sampled_candidate`), then the unsigned-integer
    /// view a copy keeps a multisampled attachment's NaN bits through on a device whose copies
    /// convert its NaNs (`nan_clear::view_candidate`), then the input attachment usage the
    /// exact color resolve reads a multisampled color attachment through
    /// (`color_resolve::input_candidate`), and then the feedback loop usage a draw that reads the
    /// texture its pass renders through a snapshot takes (`texture::feedback_candidate`), each
    /// where it may take it and it costs the image
    /// nothing: the image's memory requirements, size and alignment, are the same with it as
    /// without it (the module docs). A memoryless attachment takes the input attachment usage
    /// wherever the device makes the image: its image takes no copies, so the exact resolve reads
    /// it through that usage or not at all (`color_resolve.rs`).
    pub(crate) fn with_free_usage(
        &self,
        plan: ImagePlan,
        texture_type: TextureType,
    ) -> Result<ImagePlan, Error> {
        let keeps_hiz =
            (self.properties().driver_facts).contains(&DriverFact::SampledDepthKeepsHiZ);
        let plan = match hiz_candidate(&plan, self.compile_device().profile(), keeps_hiz) {
            Some(candidate) => self.if_made(plan, candidate, texture_type)?,
            None => plan,
        };
        let fetched = fetch_candidate(&plan, self.compile_device().profile())
            .or_else(|| depth_stencil_scratch::sampled_candidate(&plan, self));
        let plan = match fetched {
            Some(candidate) => self.if_free(plan, candidate, texture_type)?,
            None => plan,
        };
        let plan = match super::nan_clear::view_candidate(&plan, self)? {
            Some(candidate) => self.if_free(plan, candidate, texture_type)?,
            None => plan,
        };
        let plan = match color_resolve::input_candidate(&plan) {
            Some(candidate) if plan.usage.contains(ImageUsageFlags::TRANSIENT_ATTACHMENT) => {
                self.if_made(plan, candidate, texture_type)?
            }
            Some(candidate) => self.if_free(plan, candidate, texture_type)?,
            None => plan,
        };
        match crate::texture::feedback_candidate(&plan, self) {
            Some(candidate) => self.if_free(plan, candidate, texture_type),
            None => Ok(plan),
        }
    }

    /// `candidate`, `plan` with a usage added, where the device makes an image of it (its limits
    /// for the image take it, `Device::image_requirements`), else `plan`; remembered as
    /// [`Device::if_free`]'s answers are.
    fn if_made(
        &self,
        plan: ImagePlan,
        candidate: ImagePlan,
        texture_type: TextureType,
    ) -> Result<ImagePlan, Error> {
        let answers = &self.shared().textures.usage_answers;
        let known = lock(answers).answer(&plan, &candidate);
        let made = match known {
            Some(made) => made,
            None => {
                let made = self.image_requirements(&candidate, texture_type)?.is_some();
                lock(answers).remember(plan.clone(), candidate.clone(), made);
                made
            }
        };
        Ok(if made { candidate } else { plan })
    }

    /// `candidate`, `plan` with a usage added, where that costs the image nothing (as
    /// [`Device::with_free_usage`] says), else `plan`.
    pub(crate) fn if_free(
        &self,
        plan: ImagePlan,
        candidate: ImagePlan,
        texture_type: TextureType,
    ) -> Result<ImagePlan, Error> {
        #[cfg(test)]
        if (self.shared().storage_costs_forced).load(std::sync::atomic::Ordering::Relaxed)
            && candidate.usage.contains(ImageUsageFlags::STORAGE)
            && !plan.usage.contains(ImageUsageFlags::STORAGE)
        {
            return Ok(plan);
        }
        let answers = &self.shared().textures.usage_answers;
        let known = lock(answers).answer(&plan, &candidate);
        let free = match known {
            Some(free) => free,
            None => {
                let with = self.image_requirements(&candidate, texture_type)?;
                let without = self.image_requirements(&plan, texture_type)?;
                let free = match (with, without) {
                    (Some(with), Some(without)) => {
                        (with.size, with.alignment) == (without.size, without.alignment)
                    }
                    _ => false,
                };
                lock(answers).remember(plan.clone(), candidate.clone(), free);
                free
            }
        };
        Ok(if free { candidate } else { plan })
    }
}
