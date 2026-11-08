//! Multisample resolves: `MTLStoreActionMultisampleResolve` and
//! `MTLStoreActionStoreAndMultisampleResolve` (step 6's fourth part).
//!
//! **Metal**, measured on an M4 with macOS 27.0.1 (with and without API validation):
//!
//! - A color resolve averages each pixel's stored samples: unorm values round to nearest (the
//!   samples 0, 85, 170 and 255 give 128), sRGB ones are averaged in linear space and encoded
//!   again, float and half ones exactly. Integer formats are no resolve target ("cannot be a
//!   MSAA resolve target"). A pixel no draw covered resolves to the clear color.
//! - The resolve writes only the resolve texture's `resolveLevel` and `resolveSlice`, whose
//!   extent is the attachment's (a level of a larger texture serves). The resolve texture needs
//!   no `RenderTarget` usage.
//! - **A level of another extent** (probes, an M4, four fills alike; Metal's validation stops it,
//!   "size mismatch"): the overlap with the attachment resolves. A depth or stencil level keeps its
//!   texels past it; a color level takes the attachment's clear color past it over its whole level
//!   and slice after a `Clear` load, and what memory held after a `Load` or `DontCare` one
//!   (refused, `ResolveSize`). The runtime resolves into a scratch image of the attachment's
//!   extent, clears a color level reaching past it, and copies the overlap.
//! - **A multisampled color resolve texture of another sample count** than the attachment's,
//!   of its extent, is left untouched (probes 13, 13b and 14: 4 into 2, 2 into 4, drawn and
//!   not): the attachment resolves nowhere. One of the attachment's count takes a copy of its
//!   samples, which the runtime does not build (refused, `ResolveSamples`), and a depth or
//!   stencil one a layout pattern (refused).
//! - **A resolve texture of a format the attachment's tile is not stored into** (our probe `xres`
//!   on the M3 Max under macOS 27.0.1, plain Metal: 36 non-integer color formats at 2 and 4
//!   samples into 55 color formats, a Private attachment cleared and drawn or not, a Shared
//!   resolve texture prefilled; the consumer's M3 Max measurements of `BGRA8Unorm` and
//!   `BGRA8Unorm_sRGB` into `RGBA16Float`, `BGR10_XR` and its twin, macOS 26.6.2's Maps among
//!   them, over every load action, storage modes, sizes up to 1920 by 1080 and IOSurfaces): Metal
//!   holds the samples of an 8-bit norm, sRGB or packed 32-bit format as a word of codes, and of a
//!   16-bit norm format as 16-bit codes, and stores them only into a texture of a format held so
//!   (or a 64-bit extended-range one); a resolve into any other format writes nothing
//!   ([`untouched_across_formats`]). `MultisampleResolve` then leaves the samples as they were
//!   before the pass, as `DontCare` does, and `StoreAndMultisampleResolve` stores them as `Store`
//!   does: the runtime does not resolve, and stores the attachment so (in a pass without a draw
//!   call that clears it, through a backing its texture keeps, `render.rs`). A half, float,
//!   packed 16-bit or extended-range attachment's resolve writes every format measured.
//! - **A resolve texture of another pixel format** (probes, an M4 with macOS 27.0.1; Metal's
//!   validation stops it, "does not match Resolve PixelFormat"): a multisampled attachment
//!   resolving into its format's sRGB twin resolves in the **resolve texture's** encoding, a
//!   texture or a view alike: the attachment's sample codes are read as codes of the resolve
//!   texture's format, averaged in that format's space and encoded again there, and alpha's codes
//!   averaged as a linear channel's. Measured on an M3 Max under macOS 27.0.1 with samples that
//!   disagree, which the reading needs (0x80 is a fixed point of both encodings' averages, and so
//!   are 0x00 and 0xff, so a pixel whose samples agree reads the same under either rule): half a
//!   pixel's samples at code 0x40 and the rest at 0x00 store 0x2c into an `RGBA8Unorm_sRGB` resolve
//!   texture from an `RGBA8Unorm` attachment, where the attachment's own encoding would store 0x20,
//!   and 0x20 into an `RGBA8Unorm` one from an `RGBA8Unorm_sRGB` attachment, where the attachment's
//!   own encoding would store 0x2c; alpha takes 0x20 either way. Both `BGRA8` pairs and both
//!   directions alike, at 2 and 4 samples (the counts an M3 Max makes of these formats), each case
//!   deterministic over four to six processes. So the runtime resolves it by the crate-owned
//!   resolve in the resolve texture's format (`color_resolve.rs`): a
//!   subpass resolve writes only the attachment's own format
//!   (`VUID-VkSubpassDescription-pResolveAttachments-00850`), so no device's average resolves
//!   across the pair. The other channel order of the family, whose probes covered samples that
//!   agree and so read neither encoding, is not built (`NotImplemented(Render)`). The other pairs
//!   Metal writes (the M3 Max and an M4 alike) follow no rule the runtime builds: a word of codes
//!   is stored into another such format as its codes in that format's channel order, 0x80 as 0x81
//!   into a snorm channel, a shared-exponent word normalized, into `BGR10A2Unorm` as 10-bit fields
//!   swapped (`RGBA8Unorm` into `RGBA8Snorm` writes its bytes, 0x80 as 0x81); `RGBA16Float` into
//!   `RGBA8Unorm` converts, `R32Float` into `R32Uint` writes its bits, into `RGBA8Unorm` its red
//!   with zeros, and `RGBA16Float` into `RG32Float` a pattern of neither; each stays refused
//!   (`ResolveFormat`). An integer format's resolve writes values no sample holds, which change
//!   from run to run (`RGBA8Uint`, `R32Uint`): refused. Measured over every integer color format
//!   (an M4 with macOS 27.0.1, without API validation, which stops it; the 19 formats, 4 samples,
//!   every subset of covered samples over a clear, two runs): a pixel whose samples differ resolves
//!   to a value no sample holds, not sample 0's, mostly another one each run, and a 32-bit channel
//!   takes such a value even where the samples agree; only an 8- or 16-bit channel (`RGB10A2Uint`'s
//!   too) whose samples agree resolves to their value. No rule that holds for every pixel: refused.
//! - An extended-range attachment (`BGR10_XR`, `BGRA10_XR`, their `_sRGB` twins and
//!   `RGB10A2Unorm_sRGB`; our probe `xr`, the M3 Max under macOS 27.0.1) resolves a linear form's
//!   colors to the average of the samples' codes rounded half up; its alpha and an sRGB form's
//!   colors average values finer than the codes. The runtime holds the codes and resolves by the
//!   crate-owned resolve at every count (`color_resolve.rs`), Metal's rule where the codes give
//!   it and a near-equivalent elsewhere, announced as `extended-range-resolve`.
//! - An attachment the device renders through the view of its word
//!   (`metal::format::WordAttachment`: a multisampled texture held in its word's format,
//!   `image_plan.rs`) holds each sample's stored word, which no device's average reads: it
//!   resolves by the crate-owned resolve at every count (`color_resolve.rs`). Metal resolves
//!   `RGB9E5Float` and an 8-bit sRGB format from the stored codes (our probe `msaa_word_formats.m`
//!   on the M3 Max under macOS 27.0.1: the pass that drew the samples and one that loaded them
//!   resolve alike), by rules the crate-owned resolve applies; a packed 16-bit format from its
//!   tile's values, finer than the codes, where the resolve averages the codes, a near-equivalent
//!   within one code, announced as `word-resolve`.
//! - The depth filters `Sample0`, `Min` and `Max` take sample 0's depth, the smallest and the
//!   largest. The stencil filter `Sample0` takes sample 0's stencil, and `DepthResolvedSample`
//!   the stencil of the sample whose depth the depth filter took: of equal depths, the last
//!   sample after `Min` and the first after `Max` (a probe, an M3 Max and an M4 alike).
//! - `StoreAndMultisampleResolve` keeps the samples: a later pass loads them. One texture's
//!   depth or stencil stored and resolved beside the other aspect stored keeps both aspects'
//!   samples and resolves that aspect alone, the resolve texture's other aspect untouched
//!   (a probe, hazard-tracked textures, an M3 Max and an M4 alike).
//! - **A depth or stencil `StoreAndMultisampleResolve` of an untracked texture** (a heap's by
//!   default, or one made `HazardTrackingMode::Untracked`; a probe, an M3 Max with macOS 26.7 and
//!   an M4 with 27.0.1): Metal's resolve takes the samples the texture held before the pass in most
//!   runs (every probe-16 run on the M4) and the pass's in the others; a tracked texture's, a color
//!   attachment's and `MultisampleResolve` resolve the pass's. Memory- and timing-dependent, inside
//!   one pass where no fence or event reaches: the runtime resolves the pass's samples, as Metal
//!   does for a tracked texture.
//! - **A layered pass** (`renderTargetArrayLength` 2 or more; a Metal probe on an M3 Max with macOS
//!   26.7 and an M4 with 27.0.1, alike, with and without API validation): layer `i` of the pass,
//!   from the attachment's slice, resolves into slice `resolveSlice + i`, and a layer past the
//!   resolve texture's slices is not resolved (a 2D resolve texture takes layer 0 alone), which API
//!   validation accepts. A layer no draw reached resolves its clear.
//!
//! **Refused** (`Invalid`, each measured): a depth or stencil attachment's resolving store action
//! without a resolve texture; a memoryless resolve texture (a Metal probe on an M4); a
//! single-sampled depth or stencil attachment, or a color one's late target, a multisampled resolve
//! texture (but a color one of another sample count, above), another pixel format (but a
//! multisampled color attachment's sRGB twin and a format its tile is not stored into, above), an
//! integer color format, a level or slice
//! past the resolve texture, and a color level reaching past the attachment after a load other than
//! `Clear`; `DepthResolvedSample` without a depth attachment, or with a separate depth texture
//! stored beside it; and depth and stencil attachments of one texture that mix storing and
//! resolving as plain Metal leaves undetermined ([`mixed`]). A color attachment's resolve without a
//! texture, or a texture under a store action that does not resolve, the descriptor's or the
//! encoder's, stops Metal's validation, and plain Metal resolves nothing and leaves that texture
//! untouched (a Metal probe on an M4), as the runtime does; so does a depth or stencil attachment's
//! texture under such an action, or under `Unknown` still at `endEncoding`, and the attachment is
//! stored by the action (`DontCare` for `Unknown`; an M4 with macOS 27.0.1,
//! color, depth and stencil alike). So does the resolve of a single-sampled color attachment, and
//! plain Metal resolves its one sample (probe 6m's `ref_single_sampled_source`): the runtime copies
//! the attachment's texels into the resolve texture after the pass (`color_copy`), of the
//! attachment's extent.
//!
//! **Plain Metal's answers to what its validation stops** (probes on an M4, four fills alike): an
//! index the resolve texture's type does not have is ignored (`render_pass::used_indices`: a 2D
//! texture's `resolveSlice`, a 3D texture's slice, a depth plane of any other);
//! `DepthResolvedSample` with a depth attachment that does not resolve takes the sample its filter
//! picks, drawn after the pass as for a resolved depth; and one texture's depth resolved alone
//! (`MultisampleResolve`) with its stencil stored resolves the depth and keeps the stencil's
//! samples, the resolve texture's stencil untouched (a probe alike on hazard-tracked textures). One
//! texture's depth stored and resolved beside its stencil resolved alone, which plain Metal answers
//! with a layout pattern of the resolved depth in the depth's samples and its resolve texture
//! (probe 17, an M4), the runtime resolves both and keeps both aspects' samples, as it ran them
//! previously (the stencil's stored too, so MoltenVK runs Metal's valid form): a recorded
//! near-equivalent, announced once per device (`Degradation::MixedDepthStencilResolve`,
//! [`layout_patterned`]).
//!
//! **The mechanism.** The pass resolves at the end of its subpass, in render pass 2 terms:
//!
//! - **Color** through the subpass's resolve attachments. Vulkan averages a resolve of a
//!   normalized or floating-point format (`VK_RESOLVE_MODE_AVERAGE_BIT`) and leaves the
//!   average's rounding to the device, so a device may differ from Metal by one unit at a tie
//!   (`device-rounded-resolve`). A resolve into the attachment's format's sRGB twin
//!   takes no subpass resolve at all: the resolve attachment of a subpass is of the color
//!   attachment's own format, so only the crate-owned resolve writes the twin's encoding.
//! - **The samples a resolve after the pass reads.** A resolve the runtime draws after the pass
//!   (`color_resolve.rs`: the twin's and an extended-range attachment's) reads the samples the
//!   pass stores, so the pass stores them
//!   (`render_target::color_ops`) where Metal's `MultisampleResolve` stores nothing of the
//!   attachment and leaves its texture as it was. Such a pass renders the attachment through the
//!   image its texture keeps for the queue, the samples copied into it where the pass loads them
//!   and never back ([`keep_samples`]): the resolve reads what the pass rendered and the texture
//!   keeps what it held, as Metal's does. In place where the attachment's image is one a draw
//!   took a view of or the device copies no samples of, which `dont-care-store` announces.
//! - **Depth and stencil** through `VkSubpassDescriptionDepthStencilResolve` (core in Vulkan
//!   1.2, which every runtime device has): `Sample0`, `Min` and `Max` are `SAMPLE_ZERO`, `MIN`
//!   and `MAX`, and `DepthResolvedSample` after `Sample0` is `SAMPLE_ZERO`. An aspect the pass
//!   does not resolve is `NONE`, and the resolve texture keeps its texels there.
//! - **One aspect at a time.** A device that resolves a combined format's aspects only in one
//!   mode (`independentResolve`), or both or neither (`independentResolveNone`), resolves each
//!   aspect apart: the first in the draws' subpass, the second after the pass in a render pass of
//!   its own (`render_target::later_resolve`), each into a scratch image whose resolved aspect
//!   alone is copied into the resolve texture after the pass, through a staging buffer
//!   (`ResolveCopies`). A resolve's other aspect takes `NONE` where the device allows, else the
//!   same mode, which writes only the scratch image.
//! - **`DepthResolvedSample` after `Min` or `Max`**: the depth resolves in the subpass by `MIN` or
//!   `MAX`, its stencil `NONE` there, and the stencil is drawn after the pass into the same resolve
//!   attachment by `depth_stencil_resolve.rs`, which reads the attachment's samples (both aspects,
//!   which the last instance stores) through sampled views and picks Metal's sample by depth. The
//!   attachment's image must be sampled, and the pass of one layer.
//! - **`Min` or `Max` on a device without the mode** (lavapipe has neither): the depth is drawn
//!   after the pass the same way, as `FragDepth`, where that stores the picked value unchanged
//!   (`depth_copy::copies_exactly`: a 32-bit float depth on a driver measured to,
//!   `DriverFact::FragmentDepthStoresUnchanged`, or a 16-bit one, whose code is kept where
//!   `DriverFact::FragmentDepthKeepsDepth16Codes` holds); its stencil then too after
//!   `DepthResolvedSample`. Where the subpass resolves no aspect, the draws write the resolve
//!   texture (or a scratch image copied there) themselves. Elsewhere the missing mode stays
//!   refused, `NotImplemented(DepthStencilResolve)`.
//! - A resolve texture whose image cannot be an attachment (Metal's texture has no
//!   `RenderTarget` usage) is resolved into a scratch image of the pass, whose render area is
//!   copied into the texture after the pass.
//! - **Layers.** A layered pass resolves every layer in its subpass, into a view of the resolve
//!   texture's slices from `resolveSlice` where they hold the pass's layers, else into a scratch
//!   image of the pass's layers whose layers that fit in the texture's slices are copied into
//!   them after the pass. A depth or stencil attachment resolves its first layer alone, into slice
//!   `resolveSlice`, as Metal does (a probe): through a scratch image of the pass's layers whose
//!   first is copied, and its drawn aspects from the first layer's samples.
//! - **A render pass that resolves depth and stencil from the first layer and level alone**
//!   (lavapipe, `RuntimeDriverDefect::DepthStencilResolveFirstSubresource`): its render pass
//!   resolves a depth/stencil attachment from its image's first layer and level into the resolve
//!   attachment image's first, whatever layer and level the views begin at. There a resolve slice
//!   or level other than the first takes the scratch image, copied there after the pass; and an
//!   attachment that begins at another slice is copied after the pass's instances, the pass's
//!   layers of it, into a scratch image of its format and samples from the first
//!   ([`ResolveSource`]), which an instance of no draws, and each resolve after the pass, resolves
//!   from in its place. A memoryless attachment there, which takes no copy, is not built
//!   (`NotImplemented(DepthStencilResolve)`).
//! - The multisampled attachment of `MultisampleResolve` is stored `DONT_CARE`: Metal leaves its
//!   texels undefined (the recorded near-equivalent of a `DontCare` store).
//!
//! **Not built** (`NotImplemented(DepthStencilResolve)`): a filter the device's resolve modes
//! lack (`supportedDepthResolveModes`, `supportedStencilResolveModes`, or the other aspect's
//! modes where a subpass resolving one aspect must name the same mode for both), as the compile
//! device's description states them (`VkProperty::{DepthResolveMode, StencilResolveMode,
//! IndependentResolveNone, IndependentResolve}`), for which a resolve draw is the exact path; and
//! `DepthResolvedSample` after `Min` or `Max` of an attachment whose image is not sampled, or
//! whose stencil the draws cannot write; and a split resolve (below) with both aspects drawn after
//! the pass.
//!
//! **Split resolves**. Where the aspects resolve into different textures (separate depth and
//! stencil textures, which render through a combined scratch image, `depth_stencil_scratch.rs`, or
//! one combined attachment whose depth and stencil resolve textures differ), each resolve texture
//! is checked against the attachment that names it, and each aspect resolves in a subpass of its
//! own (the first in the draws' subpass, the second in a later pass) into a scratch image of the
//! attachment's format, whose aspect alone is copied into its texture after the pass through a
//! staging buffer, as one aspect at a time does. An aspect drawn after the pass (a `Min` or `Max`
//! depth on a device without the mode, or a `DepthResolvedSample` stencil after either) is drawn
//! into its own resolve texture, or into a scratch image copied there, from the attachment's
//! samples: separate textures' from the image they render through, which is then sampled
//! (`depth_stencil_scratch::scratch`). An aspect the pass does not resolve keeps its resolve
//! texture's texels. Announced as `one-aspect-resolve`.

use super::attachment_stand_in::{self, Cause, StandIn};
use super::blit::{reads, writes, TRANSFER};
use super::buffer::Buffer;
use super::color_resolve;
use super::command_buffer::CommandBuffer;
use super::degradation::Degradation;
use super::depth_stencil_scratch::Separate;
use super::encoder::EncoderCore;
use super::error::{internal, invalid, Error, Problem, RuntimeGap};
use super::image_plan::{block_family, sample_count, srgb_twin, ImagePlan};
use super::linear_attachment;
use super::render_pass::{Attachment, ColorTarget, Facts, PassLayout, RenderPassProblem, Target};
use super::render_values::{
    LoadAction, MultisampleDepthResolveFilter, MultisampleStencilResolveFilter, StoreAction,
};
use super::resource::ResourceKey;
use super::texture::{copy_units, image_aspects, Texture};
use super::texture_descriptor::TextureDescriptor;
use super::tracker::{Access, Barrier, Use};
use super::values::{ResourceOptions, StorageMode};
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::{ColorResolveFormat, DepthPick};
use metal2vulkan::device::DeviceProfile;
use metal2vulkan::error::{Feature, RuntimeDriverDefect, Unsupported};
use metal2vulkan::metal::format::{aspects, sampled_type, ExtendedRange, FormatMapping};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::pipeline::SampledType;
use metal2vulkan::vk::{Format, ImageAspectFlags, ImageCreateFlags, ImageType, ImageUsageFlags};

/// A Vulkan resolve mode of a depth or stencil aspect: `NONE` where the aspect does not resolve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum ResolveMode {
    None,
    SampleZero,
    Min,
    Max,
    /// Drawn after the pass by the pick (`depth_stencil_resolve.rs`), so no subpass resolves it:
    /// as a stencil, the stencil of the sample the pick takes (`DepthResolvedSample` after `Min`
    /// or `Max`), which no Vulkan mode states; as a depth, the pick's depth, where the device has
    /// no `MIN` or `MAX` depth mode and its fragment depth stores the value unchanged.
    Picked(DepthPick),
}

impl ResolveMode {
    /// The mode's Vulkan bit: `NONE` for `ResolveMode::Picked`, which no subpass makes.
    pub(super) fn vk(self) -> vk::ResolveModeFlags {
        match self {
            ResolveMode::None | ResolveMode::Picked(_) => vk::ResolveModeFlags::NONE,
            ResolveMode::SampleZero => vk::ResolveModeFlags::SAMPLE_ZERO,
            ResolveMode::Min => vk::ResolveModeFlags::MIN,
            ResolveMode::Max => vk::ResolveModeFlags::MAX,
        }
    }

    /// The mode a subpass resolves the aspect in: `NONE` for one drawn after the pass.
    fn in_subpass(self) -> ResolveMode {
        match self {
            ResolveMode::Picked(_) => ResolveMode::None,
            mode => mode,
        }
    }
}

/// The depth and stencil resolve modes a device has (`VkPhysicalDeviceDepthStencilResolveProperties`,
/// as the compile device's description states them).
#[derive(Clone, Copy)]
pub(super) struct ResolveModes {
    depth: vk::ResolveModeFlags,
    stencil: vk::ResolveModeFlags,
    /// One aspect may resolve while the other does not.
    independent_none: bool,
    /// The aspects may resolve in different modes.
    independent: bool,
}

impl ResolveModes {
    /// The modes the compile device's description states (its `supportedDepthResolveModes`,
    /// `supportedStencilResolveModes`, `independentResolveNone` and `independentResolve`), so
    /// the runtime resolves by what the compile layer planned for, and a description that
    /// states fewer modes than the physical device has resolves by those alone.
    pub(super) fn of(profile: &DeviceProfile) -> ResolveModes {
        let limits = profile.limits();
        let flags = |modes: metal2vulkan::vk::ResolveModeFlags| {
            vk::ResolveModeFlags::from_raw(modes.as_raw())
        };
        ResolveModes {
            depth: flags(limits.supported_depth_resolve_modes),
            stencil: flags(limits.supported_stencil_resolve_modes),
            independent_none: limits.independent_resolve_none,
            independent: limits.independent_resolve,
        }
    }

    /// Whether the device resolves a format with `both` aspects in `depth` and `stencil` mode
    /// (`NONE` for an aspect not resolved).
    fn allows(&self, depth: ResolveMode, stencil: ResolveMode, both: bool) -> bool {
        let none = ResolveMode::None;
        let supported = (depth == none || self.depth.contains(depth.vk()))
            && (stencil == none || self.stencil.contains(stencil.vk()));
        let paired = !both
            || depth == stencil
            || self.independent
            || (self.independent_none && (depth == none || stencil == none));
        supported && paired
    }
}

/// What an attachment resolves into.
#[derive(Clone, Debug)]
pub(super) struct Resolve {
    /// The pass's resolve attachment: the resolve texture's level and slice, or a scratch
    /// image's.
    pub(super) attachment: Target,
    /// The resolve texture's level and slice, when the pass resolves into a scratch image whose
    /// render area is copied there after the pass.
    pub(super) copy_to: Option<Target>,
    /// The rules of a color resolve the runtime draws after the pass, in place of the subpass's
    /// (`color_resolve.rs`): `None` where the device resolves it, and for depth and stencil.
    pub(super) shader: Option<ColorResolveFormat>,
    /// The rules are the resolve attachment's format's rather than the attachment's: a
    /// multisampled color attachment resolving into its format's sRGB twin, which Metal resolves
    /// in the resolve texture's encoding (the module docs). Such a resolve reads the attachment's
    /// codes through a view in the rules' own form, never its sample planes, whose copy format
    /// holds them under the attachment's format.
    pub(super) twin: bool,
    /// The clear color (its bits) the resolve texture's level and layers take before the
    /// scratch image's copy, where they reach past the attachment that clears them: plain Metal
    /// writes the attachment's clear color there (the module docs). `None` elsewhere.
    pub(super) clear: Option<[u32; 4]>,
}

/// The depth/stencil attachment's resolve.
#[derive(Clone, Debug)]
pub(super) struct DepthStencilResolve {
    /// Each aspect's mode as Metal's filters state it, `None` for an aspect not resolved, and
    /// `ResolveMode::Picked` for an aspect drawn after the pass (the module docs).
    pub(super) depth_mode: ResolveMode,
    pub(super) stencil_mode: ResolveMode,
    /// The resolves that make it, the first in the draws' subpass and the other in a pass of its
    /// own after the pass (`render_target::later_resolve`): one, or one per aspect on a device
    /// that cannot resolve the pair's modes in one subpass (the module docs); none where every
    /// resolved aspect is drawn.
    pub(super) steps: Vec<ResolveStep>,
    /// The resolve attachment the draws write where no step resolves into one (every resolved
    /// aspect drawn), or where the aspects resolve into different textures (`split`): the drawn
    /// aspect's resolve texture's level and slice, or a scratch image copied there.
    pub(super) drawn_into: Option<Resolve>,
    /// Its aspects resolve into different textures (separate ones, or two for one combined
    /// texture): each step resolves one aspect into a scratch image copied into its own.
    pub(super) split: bool,
}

impl DepthStencilResolve {
    /// The pick of the aspects drawn after the pass, and which: `None` where none is.
    pub(super) fn picked(&self) -> Option<(DepthPick, ImageAspectFlags)> {
        picked(self.depth_mode, self.stencil_mode)
    }

    /// The resolve attachment the drawn aspects are written into: the draws' own, or the first
    /// step's.
    pub(super) fn drawn_target(&self) -> Option<&Resolve> {
        self.drawn_into
            .as_ref()
            .or(self.steps.first().map(|step| &step.resolve))
    }

    /// Each resolve attachment it writes, with the aspects of the resolve texture written there
    /// (those a scratch image's copy takes): its steps', then the draws' own.
    pub(super) fn written(&self) -> impl Iterator<Item = (&Resolve, ImageAspectFlags)> {
        let drawn = self
            .picked()
            .map_or(ImageAspectFlags::empty(), |(_, aspects)| aspects);
        let steps = self.steps.iter().map(|step| (&step.resolve, step.aspects));
        steps.chain(self.drawn_into.iter().map(move |resolve| (resolve, drawn)))
    }
}

/// One depth/stencil resolve of a subpass: the draws' subpass's, or a later pass's.
#[derive(Clone, Debug)]
pub(super) struct ResolveStep {
    pub(super) resolve: Resolve,
    /// The subpass's modes, `None` for an aspect it does not resolve.
    pub(super) depth: ResolveMode,
    pub(super) stencil: ResolveMode,
    /// The aspects of the resolve texture it writes: those a scratch image's copy takes.
    pub(super) aspects: ImageAspectFlags,
    /// It resolves one aspect into a scratch image, the device resolving the pair's modes in no
    /// one subpass (the module docs).
    pub(super) one_aspect: bool,
}

/// One attachment's resolve as its descriptor states it: the resolve texture with its level,
/// slice and depth plane, and the store action.
pub(super) struct Wanted<'a> {
    pub(super) texture: Option<&'a Texture>,
    pub(super) level: u64,
    pub(super) slice: u64,
    pub(super) depth_plane: u64,
    pub(super) store: StoreAction,
}

impl<'a> Wanted<'a> {
    /// The resolve of an attachment with a texture (`attached`) whose descriptor names
    /// `texture` at `(level, slice, depth_plane)` under `store`; `None` without one.
    pub(super) fn of(
        attached: bool,
        texture: &'a Option<Texture>,
        (level, slice, depth_plane): (u64, u64, u64),
        store: StoreAction,
    ) -> Option<Wanted<'a>> {
        attached.then_some(Wanted {
            texture: texture.as_ref(),
            level,
            slice,
            depth_plane,
            store,
        })
    }
}

/// The refusal of a resolve of a multisampled attachment of `pf` that no resolve reads: one of
/// Apple's private forms held in raw codes of which Metal makes no multisampled texture (bfloat,
/// padded and two-plane ones; our probe `xrms`, the M3 Max under macOS 27.0.1), whose codes the
/// crate-owned resolve has no rule for (`color_resolve::extended`). The format's Resolve capability
/// answers it too (`families::formats`).
pub(crate) fn unmodelled_resolve(pf: PixelFormat) -> Option<Unsupported> {
    (ExtendedRange::of(pf).is_some() && super::color_resolve::extended(pf).is_none())
        .then(|| Unsupported::not_implemented(Feature::Format))
}

/// What Apple's GPU holds of a multisampled color attachment of a pixel format as its resolve
/// stores it, by the formats it stores that into ([`untouched_across_formats`]; our probe `xres`,
/// the M3 Max under macOS 27.0.1, plain Metal: each of the 36 non-integer color formats below at
/// 2 and 4 samples, resolved into a texture of each of the 55 color formats below).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResolveTile {
    /// A 32-bit word of codes: the 8-bit unorm, sRGB and snorm formats of one, two and four
    /// channels and `A8Unorm`, and the packed 32-bit `RGB10A2Unorm`, `BGR10A2Unorm`,
    /// `RG11B10Float` and `RGB9E5Float`. Stored into a texture of another such format as its codes
    /// (and into `BGRA10_XR` and its twin as a pattern of neither).
    Codes32,
    /// 16-bit unorm and snorm codes: `R16Unorm`, `RG16Unorm`, `RGBA16Unorm` and their snorm
    /// formats. Stored into a texture of another such format as its codes (and into `BGRA10_XR`
    /// and its twin as a pattern).
    Norm16,
    /// Values: the half and float formats, the packed 16-bit ones and the extended-range ones.
    /// Stored into a texture of every format measured, converted or as a pattern.
    Values,
    /// The integer formats, which are no resolve source: as a resolve texture, measured.
    Integers,
}

impl ResolveTile {
    /// The tile of `pf`; `None` for a format the probe did not resolve into (a depth, stencil,
    /// compressed, three-channel, two-plane or YCbCr format, `R16Float`'s private kin).
    fn of(pf: PixelFormat) -> Option<ResolveTile> {
        use PixelFormat as P;
        Some(match pf {
            P::A8Unorm
            | P::R8Unorm
            | P::R8UnormSRGB
            | P::R8Snorm
            | P::RG8Unorm
            | P::RG8UnormSRGB
            | P::RG8Snorm
            | P::RGBA8Unorm
            | P::RGBA8UnormSRGB
            | P::RGBA8Snorm
            | P::BGRA8Unorm
            | P::BGRA8UnormSRGB
            | P::RGB10A2Unorm
            | P::BGR10A2Unorm
            | P::RG11B10Float
            | P::RGB9E5Float => ResolveTile::Codes32,
            P::R16Unorm
            | P::R16Snorm
            | P::RG16Unorm
            | P::RG16Snorm
            | P::RGBA16Unorm
            | P::RGBA16Snorm => ResolveTile::Norm16,
            P::R16Float
            | P::RG16Float
            | P::RGBA16Float
            | P::R32Float
            | P::RG32Float
            | P::RGBA32Float
            | P::B5G6R5Unorm
            | P::A1BGR5Unorm
            | P::ABGR4Unorm
            | P::BGR5A1Unorm
            | P::BGR10XR
            | P::BGR10XRSRGB
            | P::BGRA10XR
            | P::BGRA10XRSRGB => ResolveTile::Values,
            P::R8Uint
            | P::R8Sint
            | P::R16Uint
            | P::R16Sint
            | P::RG8Uint
            | P::RG8Sint
            | P::R32Uint
            | P::R32Sint
            | P::RG16Uint
            | P::RG16Sint
            | P::RGBA8Uint
            | P::RGBA8Sint
            | P::RGB10A2Uint
            | P::RG32Uint
            | P::RG32Sint
            | P::RGBA16Uint
            | P::RGBA16Sint
            | P::RGBA32Uint
            | P::RGBA32Sint => ResolveTile::Integers,
            _ => return None,
        })
    }
}

/// Whether the multisampled color attachment `source` names a resolve texture, as `wanted`
/// states it, that Metal leaves untouched across formats ([`untouched_across_formats`]): its
/// store action resolves or is `Unknown`. [`color`] gives it no resolve.
pub(super) fn unresolved_across(source: &Target, wanted: &Wanted<'_>) -> bool {
    source.samples > 1
        && (wanted.store.resolves() || wanted.store == StoreAction::Unknown)
        && wanted.texture.is_some_and(|into| {
            untouched_across_formats(source.texture.pixel_format(), into.pixel_format())
        })
}

/// Whether Metal leaves a resolve texture of pixel format `into` untouched by the resolve of a
/// multisampled color attachment of `from`, another format (the module docs): a word of codes, or
/// 16-bit norm codes ([`ResolveTile`]), into a format of another tile but the 64-bit
/// extended-range ones. Every pair measured gave one answer at 2 and 4 samples, under both
/// resolving store actions, drawn and not.
fn untouched_across_formats(from: PixelFormat, into: PixelFormat) -> bool {
    use PixelFormat as P;
    let (Some(source), Some(destination)) = (ResolveTile::of(from), ResolveTile::of(into)) else {
        return false;
    };
    matches!(source, ResolveTile::Codes32 | ResolveTile::Norm16)
        && destination != source
        && !matches!(into, P::BGRA10XR | P::BGRA10XRSRGB)
}

fn problem(call: &'static str, p: RenderPassProblem) -> Error {
    invalid(call, Problem::RenderPass(p))
}

/// Whether `which` resolves: its descriptor's resolve texture and store action, checked (a depth or
/// stencil attachment's resolving action needs a texture). An attachment does not resolve without a
/// texture and an action that resolves or is `Unknown`: plain Metal leaves a resolve texture under
/// `Store` or `DontCare` untouched and stores the attachment by the action, and a color
/// attachment's resolving action without a texture resolves nothing (Metal probe cases
/// `ref_resolve_without_texture` and `ref_texture_with_store` on an M4; depth and stencil alike in
/// our probe; its validation stops each).
fn resolves(call: &'static str, which: Attachment, wanted: &Wanted<'_>) -> Result<bool, Error> {
    let resolving = wanted.store.resolves() || wanted.store == StoreAction::Unknown;
    if let Attachment::Color(_) = which {
        return Ok(wanted.texture.is_some() && resolving);
    }
    match wanted.texture {
        None if wanted.store.resolves() => Err(problem(
            call,
            RenderPassProblem::ResolveWithoutTexture(which),
        )),
        None => Ok(false),
        Some(_) => Ok(resolving),
    }
}

/// Whether one texture's depth and stencil store actions mix storing and resolving as Metal
/// leaves undetermined: the stencil resolved alone (`MultisampleResolve`) beside a stored depth,
/// whose samples plain Metal overwrites with a layout pattern of the resolved depth (probes, an M4;
/// its validation stops it). An `Unknown` action decides at its late store action. The other forms
/// are Metal's answers (probe 17 on hazard-tracked textures, an M3 Max and an M4): an aspect stored
/// and resolved (`StoreAndMultisampleResolve`) beside a stored one keeps both aspects' samples and
/// resolves that aspect alone, the resolve texture's other aspect untouched, and the depth resolved
/// alone beside a stored stencil resolves the depth by its filter and keeps the stencil's samples.
pub(super) fn mixed(depth: StoreAction, stencil: StoreAction) -> bool {
    depth == StoreAction::Store && stencil == StoreAction::MultisampleResolve
}

/// Whether one texture's depth and stencil store actions are the depth stored and resolved
/// beside the stencil resolved alone, which Metal's validation stops and plain Metal answers with
/// a layout pattern of the resolved depth in the depth's samples and its resolve texture (a probe,
/// an M4). The runtime, which ran it previously, resolves both and keeps both aspects' samples
/// (`render_target::patterned_aspects`: so MoltenVK runs Metal's valid `StoreAndMultisampleResolve`
/// of both), and announces `Degradation::MixedDepthStencilResolve` (a recorded near-equivalent).
pub(super) fn layout_patterned(depth: StoreAction, stencil: StoreAction) -> bool {
    depth == StoreAction::StoreAndMultisampleResolve && stencil == StoreAction::MultisampleResolve
}

/// The resolve of color attachment `which`, `source`, as `wanted` states it; `None` when it does
/// not resolve. Refused as the module docs say.
/// The attachment's `load` action and `clear` color decide a resolve texture level reaching past
/// it.
pub(super) fn color(
    call: &'static str,
    maker: &CommandBuffer,
    which: Attachment,
    source: &Target,
    wanted: &Wanted<'_>,
    (load, clear): (LoadAction, vk::ClearColorValue),
) -> Result<Option<Resolve>, Error> {
    let device = maker.device();
    if !resolves(call, which, wanted)? {
        return Ok(None);
    }
    let texture = wanted
        .texture
        .ok_or_else(|| internal("a resolve without its texture"))?;
    if matches!(
        sampled_type(source.texture.pixel_format()),
        Some(SampledType::Uint | SampledType::Sint | SampledType::Uint64)
    ) {
        return Err(problem(call, RenderPassProblem::ResolveFormat(which)));
    }
    // A private form's codes no resolve reads (Metal makes no multisampled texture of it).
    if let Some(unsupported) = unmodelled_resolve(source.texture.pixel_format()) {
        return Err(Error::Unsupported(unsupported));
    }
    single_sampled(call, which, source)?;
    let Some(destination) = destination(call, device, which, source, texture, wanted)? else {
        return Ok(None);
    };
    // A level reaching past the attachment takes its clear color there; after a `Load` or
    // `DontCare` load plain Metal writes what memory held there (the module docs): refused.
    let [width, height] = destination.target.extent;
    let past = width > source.extent[0] || height > source.extent[1];
    if past && load != LoadAction::Clear {
        return Err(problem(call, RenderPassProblem::ResolveSize(which)));
    }
    let mut resolve =
        destination.attach(maker, source, ImageUsageFlags::COLOR_ATTACHMENT, false)?;
    if past {
        // SAFETY: every member of the union is four 32-bit values; any bits are a valid `u32`.
        resolve.clear = Some(unsafe { clear.uint32 });
    }
    resolve.shader = color_resolve::of_resolve(device, source, &resolve)?;
    Ok(Some(resolve))
}

/// Whether a resolve the runtime draws after the pass reads `color`'s samples (the crate-owned
/// color resolve, `color_resolve.rs`) while the pass stores nothing of the attachment
/// (`MultisampleResolve`, which leaves Metal's texture as it was, the module docs). The pass must
/// store the samples for that resolve to read (`render_target::color_ops`), so it would write the
/// texture in place: [`keep_samples`] renders it through an image its texture keeps instead.
fn samples_read_after(color: &ColorTarget) -> bool {
    color.store == StoreAction::MultisampleResolve
        && color.target.samples > 1
        && color.resolve.as_ref().is_some_and(|r| r.shader.is_some())
}

/// Render each multisampled color attachment among `colors`, a pass's as it ends, whose samples a
/// resolve the runtime draws after the pass reads while the pass stores nothing of it
/// ([`samples_read_after`]), through the image its texture keeps for `maker`'s queue
/// (`attachment_stand_in::kept_backing`, [`Cause::KeptSamples`]): the samples are copied into it
/// where the pass loads them and never back, so the texture keeps the samples it held before the
/// pass, as Metal's does, while the resolve still reads the samples the pass rendered.
///
/// In place where the pass executes render commands (`executes`, whose commands record later) or a
/// draw took a view of the attachment's image (`viewed`, as `kept_bytes::settle` asks it of the
/// attachment it renders apart: a framebuffer fetch in place, its sample planes, a view of another
/// format, its extended-range shadow), where it already renders through a stand-in (whose copy
/// back Metal's store action decides alike, `attachment_stand_in.rs`), where its texture keeps no
/// texels (memoryless), and where the device copies no image of the format: there the pass stores
/// the samples it rendered, Metal's answer where its tiler overflows, which `dont-care-store`
/// announces (`Device::degrade_pass`). On a device whose framebuffer fetch reads input
/// attachments (`PassLayout::fetch_inputs`), whose subpass names every color as an input
/// attachment too, a color is viewed only where a draw read it so: the input reference takes the
/// framebuffer's view, the kept image's, as the color reference does. The linear rule's `taken`
/// counts every color of a pass there (`render.rs`), which would keep no samples on such a device
/// at all. A pass whose color attachments are multisampled has no linear one (no texture over a
/// buffer is multisampled), so none of this moves what `VK_NV_linear_color_attachment`'s rule
/// binds (`linear_attachment.rs`).
pub(super) fn keep_samples(
    maker: &CommandBuffer,
    colors: &mut [Option<ColorTarget>],
    executes: bool,
    viewed: impl Fn(usize) -> bool,
) -> Result<(), Error> {
    if executes {
        return Ok(());
    }
    let queue = maker.queue_place().id();
    for (index, color) in colors.iter_mut().enumerate() {
        let Some(color) = color else {
            continue;
        };
        if !samples_read_after(color)
            || color.stand_in.is_some()
            || color.memoryless
            || viewed(index)
        {
            continue;
        }
        // A load other than `Clear` copies the attachment's samples into the image first: an
        // image or a format the device copies no samples of keeps the pass in place.
        if color.load != LoadAction::Clear
            && attachment_stand_in::check_copies(&color.target).is_err()
        {
            continue;
        }
        let own = color.target.clone();
        let backing = match attachment_stand_in::kept_backing(maker, &own, queue, true) {
            Ok(backing) => backing,
            Err(Error::NotImplemented(RuntimeGap::Render)) => continue,
            Err(error) => return Err(error),
        };
        color.target = backing;
        color.stand_in = Some(StandIn {
            own,
            cause: Cause::KeptSamples,
            area: None,
        });
    }
    Ok(())
}

/// The resolve of the depth/stencil attachment `source` (one texture for both aspects, one of
/// them alone, or the combined scratch image `separate` textures render through) with the depth
/// and stencil attachments' `depth` and `stencil` and filters; `None` when neither resolves.
/// Refused as the module docs say.
pub(super) fn depth_stencil(
    call: &'static str,
    maker: &CommandBuffer,
    source: &Target,
    separate: Option<&Separate>,
    depth: Option<&Wanted<'_>>,
    stencil: Option<&Wanted<'_>>,
    depth_filter: MultisampleDepthResolveFilter,
    stencil_filter: MultisampleStencilResolveFilter,
) -> Result<Option<DepthStencilResolve>, Error> {
    let device = maker.device();
    let depth_resolves = match depth {
        Some(d) => resolves(call, Attachment::Depth, d)?,
        None => false,
    };
    let stencil_resolves = match stencil {
        Some(s) => resolves(call, Attachment::Stencil, s)?,
        None => false,
    };
    // One texture whose depth and stencil mix storing and resolving as Metal leaves undetermined
    // ([`mixed`]).
    if let (Some(d), Some(s)) = (depth, stencil) {
        if separate.is_none() && mixed(d.store, s.store) {
            return Err(problem(call, RenderPassProblem::DepthStencilResolveMix));
        }
    }
    // `DepthResolvedSample` takes the sample the depth attachment's filter picks whether or not
    // the depth resolves (probes, an M4): refused without a depth attachment, and where a separate
    // depth texture that does not resolve is stored (plain Metal stores a layout pattern of its
    // samples, probe 14) or may be (`Unknown`).
    let unpicked = match depth {
        None => true,
        Some(d) => separate.is_some() && !depth_resolves && d.store != StoreAction::DontCare,
    };
    if stencil_resolves
        && stencil_filter == MultisampleStencilResolveFilter::DepthResolvedSample
        && unpicked
    {
        return Err(problem(
            call,
            RenderPassProblem::DepthResolvedSampleWithoutDepthResolve,
        ));
    }
    if !depth_resolves && !stencil_resolves {
        return Ok(None);
    }
    // Each resolved aspect's destination, checked as Metal checks it against the attachment
    // that names it: a separate texture's own (its format, samples and extent), else `source`.
    let checked = |which: Attachment| match (separate, which) {
        (Some(separate), Attachment::Depth) => separate
            .depth
            .as_ref()
            .ok_or_else(|| internal("a separate depth resolve without its texture")),
        (Some(separate), _) => Ok(&separate.stencil),
        (None, _) => Ok(source),
    };
    let aspect_destination =
        |which: Attachment, resolves: bool, wanted: Option<&Wanted<'_>>| match (resolves, wanted) {
            (true, Some(wanted)) => {
                let texture = wanted
                    .texture
                    .ok_or_else(|| internal("a resolve without its texture"))?;
                let source = checked(which)?;
                single_sampled(call, which, source)?;
                destination(call, device, which, source, texture, wanted)?
                    .ok_or_else(|| internal("a depth or stencil resolve left untouched"))
                    .map(Some)
            }
            _ => Ok(None),
        };
    let depth_destination = aspect_destination(Attachment::Depth, depth_resolves, depth)?;
    let stencil_destination = aspect_destination(Attachment::Stencil, stencil_resolves, stencil)?;
    // Both aspects resolve into one texture's level and slice, or each into its own: separate
    // textures, or one combined texture's aspects resolving into two (the split resolve).
    let same = match (&depth_destination, &stencil_destination) {
        (Some(d), Some(s)) => {
            d.target.texture.same_storage(&s.target.texture)
                && (d.target.level, d.target.layer) == (s.target.level, s.target.layer)
        }
        _ => true,
    };
    let split = separate.is_some() || !same;
    let destination = depth_destination
        .clone()
        .or_else(|| stencil_destination.clone())
        .ok_or_else(|| internal("a depth/stencil resolve without its attachment"))?;
    let modes = ResolveModes::of(device.compile_device().profile());
    // A `Min` or `Max` depth the device has no mode for is drawn after the pass where its
    // fragment depth stores the picked value unchanged (`depth_copy::copies_exactly`).
    let exact_depth = super::depth_copy::copies_exactly(device, source.format)
        && super::depth_copy::copies_exactly(device, destination.target.format);
    let (depth_mode, stencil_mode) = aspect_modes(
        &modes,
        exact_depth,
        (depth_resolves, depth_filter),
        (stencil_resolves, stencil_filter),
    );
    let subpass_stencil = stencil_mode.in_subpass();
    // A drawn depth's subpass mode is free, as the draws overwrite it: `NONE`, or `SAMPLE_ZERO`
    // where the device pairs no `NONE` with the stencil's mode (lavapipe), which spares the
    // stencil a step of its own.
    let both =
        image_aspects(source.format).contains(ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL);
    let none = ResolveMode::None;
    let subpass_depth = match depth_mode {
        ResolveMode::Picked(_)
            if subpass_stencil != none
                && !modes.allows(none, subpass_stencil, both)
                && modes.allows(ResolveMode::SampleZero, subpass_stencil, both) =>
        {
            ResolveMode::SampleZero
        }
        mode => mode.in_subpass(),
    };
    // Every resolved aspect drawn: no subpass resolves, and the draws write their own resolve
    // attachment.
    let (mut steps, drawn_into) = if split {
        // A split resolve resolves each aspect a subpass resolves into a scratch image of
        // `source` in a subpass of its own, and copies that aspect alone into its texture after
        // the pass; an aspect drawn after the pass is drawn into its own texture (or a scratch
        // image copied there). Both aspects drawn, each into its own texture: not built.
        let usage = ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        let drawn = match (depth_mode, stencil_mode) {
            (ResolveMode::Picked(_), ResolveMode::Picked(_)) => {
                return Err(Error::NotImplemented(RuntimeGap::DepthStencilResolve));
            }
            (ResolveMode::Picked(_), _) => depth_destination.clone(),
            (_, ResolveMode::Picked(_)) => stencil_destination.clone(),
            _ => None,
        };
        let drawn_into = match drawn {
            Some(drawn) => Some(drawn.attach(maker, source, usage, false)?),
            None => None,
        };
        let aspects = [
            (
                depth_mode.in_subpass(),
                ImageAspectFlags::DEPTH,
                depth_destination,
            ),
            (
                subpass_stencil,
                ImageAspectFlags::STENCIL,
                stencil_destination,
            ),
        ];
        (split_steps(maker, source, aspects, &modes)?, drawn_into)
    } else if subpass_depth == none && subpass_stencil == none {
        let usage = ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        (
            Vec::new(),
            Some(destination.attach(maker, source, usage, false)?),
        )
    } else {
        let steps = steps(
            maker,
            source,
            destination,
            subpass_depth,
            subpass_stencil,
            &modes,
        )?;
        (steps, None)
    };
    // A drawn aspect is written into the draws' own resolve attachment, or else into the first
    // step's, which then holds it too (a scratch image's copy takes it).
    if let (Some((_, aspects)), Some(step), None) = (
        picked(depth_mode, stencil_mode),
        steps.first_mut(),
        &drawn_into,
    ) {
        step.aspects |= aspects;
    }
    let resolve = DepthStencilResolve {
        depth_mode,
        stencil_mode,
        steps,
        drawn_into,
        split,
    };
    if let Some((pick, aspects)) = resolve.picked() {
        let into = resolve
            .drawn_target()
            .ok_or_else(|| internal("a drawn resolve without its attachment"))?;
        super::depth_stencil_resolve::draws_run(device, source, &into.attachment, pick, aspects)?;
    }
    Ok(Some(resolve))
}

/// Each aspect's mode as Metal's filters state it on a device with `modes`, `ResolveMode::None`
/// for an aspect that does not resolve (`depth.0`, `stencil.0`): `Sample0`, `Min` and `Max` are
/// `SAMPLE_ZERO`, `MIN` and `MAX`, a `Min` or `Max` depth the device has no mode for picked by
/// the draws after the pass where they store it unchanged (`exact_depth`,
/// `depth_copy::copies_exactly`); a `DepthResolvedSample` stencil is sample 0's after `Sample0`,
/// and picked after `Min` or `Max`, which no Vulkan mode states, the depth resolved or not.
fn aspect_modes(
    modes: &ResolveModes,
    exact_depth: bool,
    (depth_resolves, depth_filter): (bool, MultisampleDepthResolveFilter),
    (stencil_resolves, stencil_filter): (bool, MultisampleStencilResolveFilter),
) -> (ResolveMode, ResolveMode) {
    let pick_or = |mode: ResolveMode, pick: DepthPick| match exact_depth
        && !modes.depth.contains(mode.vk())
    {
        true => ResolveMode::Picked(pick),
        false => mode,
    };
    let depth = match (depth_resolves, depth_filter) {
        (false, _) => ResolveMode::None,
        (true, MultisampleDepthResolveFilter::Sample0) => ResolveMode::SampleZero,
        (true, MultisampleDepthResolveFilter::Min) => pick_or(ResolveMode::Min, DepthPick::Min),
        (true, MultisampleDepthResolveFilter::Max) => pick_or(ResolveMode::Max, DepthPick::Max),
    };
    let stencil = match (stencil_resolves, stencil_filter) {
        (false, _) => ResolveMode::None,
        (true, MultisampleStencilResolveFilter::Sample0) => ResolveMode::SampleZero,
        (true, MultisampleStencilResolveFilter::DepthResolvedSample) => match depth_filter {
            MultisampleDepthResolveFilter::Sample0 => ResolveMode::SampleZero,
            MultisampleDepthResolveFilter::Min => ResolveMode::Picked(DepthPick::Min),
            MultisampleDepthResolveFilter::Max => ResolveMode::Picked(DepthPick::Max),
        },
    };
    (depth, stencil)
}

/// Whether the resolve of a depth/stencil attachment whose depth is of `format` (`depth` and
/// `stencil` its descriptors' resolves, under `depth_filter` and `stencil_filter`) may be drawn
/// after the pass on `device`, the draws reading the attachment's samples through sampled views
/// (`depth_stencil_resolve.rs`): the image the runtime renders separate textures through then
/// takes `SAMPLED` usage (`depth_stencil_scratch::scratch`). It answers before the pass's store
/// actions and resolve textures are checked, so it errs toward yes: an `Unknown` store action
/// may resolve, and [`depth_stencil`] decides by the resolve texture's format too.
pub(super) fn may_draw(
    device: &Device,
    format: Format,
    depth: Option<&Wanted<'_>>,
    stencil: Option<&Wanted<'_>>,
    depth_filter: MultisampleDepthResolveFilter,
    stencil_filter: MultisampleStencilResolveFilter,
) -> bool {
    let may_resolve = |wanted: Option<&Wanted<'_>>| {
        wanted.is_some_and(|w| {
            w.texture.is_some() && (w.store.resolves() || w.store == StoreAction::Unknown)
        })
    };
    let modes = ResolveModes::of(device.compile_device().profile());
    let exact_depth = super::depth_copy::copies_exactly(device, format);
    let (depth, stencil) = aspect_modes(
        &modes,
        exact_depth,
        (may_resolve(depth), depth_filter),
        (may_resolve(stencil), stencil_filter),
    );
    picked(depth, stencil).is_some()
}

/// The pick of the aspects of `depth` and `stencil` mode drawn after the pass, and which: `None`
/// where neither is.
fn picked(depth: ResolveMode, stencil: ResolveMode) -> Option<(DepthPick, ImageAspectFlags)> {
    let mut drawn = None;
    for (mode, aspect) in [
        (depth, ImageAspectFlags::DEPTH),
        (stencil, ImageAspectFlags::STENCIL),
    ] {
        if let ResolveMode::Picked(pick) = mode {
            let aspects = drawn.map_or(ImageAspectFlags::empty(), |(_, a)| a);
            drawn = Some((pick, aspects | aspect));
        }
    }
    drawn
}

/// The subpass resolves of `source` into `destination` in `depth` and `stencil` mode on a device
/// with `modes`: one where the device resolves the pair in one subpass, else one per resolved
/// aspect into a scratch image (the module docs).
fn steps(
    maker: &CommandBuffer,
    source: &Target,
    destination: Destination,
    depth: ResolveMode,
    stencil: ResolveMode,
    modes: &ResolveModes,
) -> Result<Vec<ResolveStep>, Error> {
    let none = ResolveMode::None;
    let named = |mode: ResolveMode, aspect: ImageAspectFlags| {
        if mode == none {
            ImageAspectFlags::empty()
        } else {
            aspect
        }
    };
    let aspects = named(depth, ImageAspectFlags::DEPTH) | named(stencil, ImageAspectFlags::STENCIL);
    // The attachment's Vulkan format (a `Depth32Float` texture may be held combined) decides
    // whether its aspects resolve as a pair.
    let both =
        image_aspects(source.format).contains(ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL);
    if modes.allows(depth, stencil, both) {
        let needed = ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        return Ok(vec![ResolveStep {
            resolve: destination.attach(maker, source, needed, true)?,
            depth,
            stencil,
            aspects,
            one_aspect: false,
        }]);
    }
    // One subpass per aspect, each into a scratch image copied into the resolve texture.
    if !both || !destination.usage.contains(ImageUsageFlags::TRANSFER_DST) {
        return Err(Error::NotImplemented(RuntimeGap::DepthStencilResolve));
    }
    let mut steps = Vec::new();
    for (mode, aspect) in [
        (depth, ImageAspectFlags::DEPTH),
        (stencil, ImageAspectFlags::STENCIL),
    ] {
        if mode == none {
            continue;
        }
        let pair = |other| {
            if aspect == ImageAspectFlags::DEPTH {
                (mode, other)
            } else {
                (other, mode)
            }
        };
        let (depth, stencil) = [pair(none), pair(mode)]
            .into_iter()
            .find(|&(d, s)| modes.allows(d, s, true))
            .ok_or(Error::NotImplemented(RuntimeGap::DepthStencilResolve))?;
        steps.push(ResolveStep {
            resolve: Resolve {
                attachment: scratch(maker, source, ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, 1)?,
                copy_to: Some(destination.target.clone()),
                shader: None,
                twin: false,
                clear: None,
            },
            depth,
            stencil,
            aspects: aspect,
            one_aspect: true,
        });
    }
    Ok(steps)
}

/// The steps of a split resolve of `source`: for each aspect of `aspects` resolved (its mode not
/// `None`), a subpass resolving it in that mode (the other aspect `NONE` where the device allows,
/// else the same mode) into a scratch image of `source`, whose aspect alone is copied into the
/// aspect's destination after the pass. Refused `NotImplemented(DepthStencilResolve)` where the
/// device has no such pair of modes or a destination's image takes no copy.
fn split_steps(
    maker: &CommandBuffer,
    source: &Target,
    aspects: [(ResolveMode, ImageAspectFlags, Option<Destination>); 2],
    modes: &ResolveModes,
) -> Result<Vec<ResolveStep>, Error> {
    let none = ResolveMode::None;
    let both =
        image_aspects(source.format).contains(ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL);
    let mut steps = Vec::new();
    for (mode, aspect, destination) in aspects {
        if mode == none {
            continue;
        }
        let destination =
            destination.ok_or_else(|| internal("a resolved aspect without its destination"))?;
        if !destination.usage.contains(ImageUsageFlags::TRANSFER_DST)
            || super::render_pass::not_2d(&destination.target)
        {
            return Err(Error::NotImplemented(RuntimeGap::DepthStencilResolve));
        }
        let pair = |other| {
            if aspect == ImageAspectFlags::DEPTH {
                (mode, other)
            } else {
                (other, mode)
            }
        };
        let (depth, stencil) = [pair(none), pair(mode)]
            .into_iter()
            .find(|&(d, s)| modes.allows(d, s, both))
            .ok_or(Error::NotImplemented(RuntimeGap::DepthStencilResolve))?;
        steps.push(ResolveStep {
            resolve: Resolve {
                attachment: scratch(maker, source, ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, 1)?,
                copy_to: Some(destination.target),
                shader: None,
                twin: false,
                clear: None,
            },
            depth,
            stencil,
            aspects: aspect,
            one_aspect: true,
        });
    }
    Ok(steps)
}

/// A resolve texture's level and slice, with its image's usage.
#[derive(Clone)]
struct Destination {
    target: Target,
    usage: ImageUsageFlags,
    /// The destination is the attachment's format's sRGB twin, whose encoding the resolve works
    /// in (`Resolve::twin`).
    twin: bool,
}

impl Destination {
    /// The resolve into the destination: the pass's attachment where its image can be one with
    /// `needed` usage, else a scratch image whose render area is copied there after the pass. A
    /// render pass's depth/stencil resolve (`depth_stencil`) on a device that writes its resolve
    /// attachment image's first layer and level alone ([`first_subresource_only`]) takes the
    /// scratch image where the destination begins at another.
    fn attach(
        self,
        maker: &CommandBuffer,
        source: &Target,
        needed: ImageUsageFlags,
        depth_stencil: bool,
    ) -> Result<Resolve, Error> {
        let device = maker.device();
        // A layered pass's resolve attachment has every layer of the pass: a texture whose slices
        // hold them all is one, else the pass resolves into a scratch image of its layers whose
        // layers that fit are copied. A level of another extent than the attachment's is
        // resolved into through a scratch image too, whose overlap with it is copied.
        let misplaced =
            depth_stencil && first_subresource_only(device) && !at_first_subresource(&self.target);
        // Nor is a linear image the device renders only where every color attachment is linear
        // (`linear_attachment.rs`): the pass's are multisampled, never linear.
        let bound = !depth_stencil && linear_attachment::bound(device, &self.target)?;
        if bound {
            let format = self.target.texture.pixel_format();
            device.degrade(Degradation::LinearAttachmentCopies, || {
                (
                    format!("a {format:?} texture's linear image, the resolve of a render pass"),
                    "the device renders a linear image of the format only where every color \
                     attachment of the render pass instance is linear \
                     (VK_NV_linear_color_attachment), and the pass's are multisampled"
                        .to_owned(),
                )
            });
        }
        if self.usage.contains(needed)
            && self.target.layers == source.layers
            && self.target.extent == source.extent
            && !misplaced
            && !bound
        {
            return Ok(Resolve {
                attachment: self.target,
                copy_to: None,
                shader: None,
                twin: self.twin,
                clear: None,
            });
        }
        // The scratch's copy into a 3D texture's planes, or into a 1D texture, is not built
        // (`render_pass::not_2d`).
        if !self.usage.contains(ImageUsageFlags::TRANSFER_DST)
            || super::render_pass::not_2d(&self.target)
        {
            return Err(Error::NotImplemented(RuntimeGap::Render));
        }
        // A twin resolve writes the resolve texture's format, so its scratch image is of that
        // format, whose bytes the copy after the pass moves into the texture; every other
        // resolve's is of the attachment's.
        let written = match self.twin {
            true => Target {
                texture: self.target.texture.clone(),
                format: self.target.format,
                ..source.clone()
            },
            false => source.clone(),
        };
        Ok(Resolve {
            attachment: scratch(maker, &written, needed, 1)?,
            copy_to: Some(self.target),
            shader: None,
            twin: self.twin,
            clear: None,
        })
    }
}

/// Refuse the resolve of a single-sampled `source` (measured: Metal's validation stops it), but
/// for a color attachment's, a copy (`color_copy`).
fn single_sampled(call: &'static str, which: Attachment, source: &Target) -> Result<(), Error> {
    if source.samples < 2 {
        return Err(problem(
            call,
            RenderPassProblem::ResolveSourceSamples(which),
        ));
    }
    Ok(())
}

/// The resolve of single-sampled color attachment `source` (`which`) as `wanted` states it: the
/// resolve texture's level and slice that the attachment's texels are copied into after the pass,
/// over the render area and the pass's layers that fit (`ResolveCopies`), as plain Metal resolves
/// one sample (Metal probe case `ref_single_sampled_source` on an M4, which its validation stops:
/// the resolve texture holds the draw); `None` where it does not resolve. Refused as `color`
/// refuses a resolve but for the sample count, and `NotImplemented(Render)` where the copy is not
/// built: a memoryless attachment, or an attachment or destination whose image takes no copy or is
/// not 2D.
pub(super) fn color_copy(
    call: &'static str,
    device: &Device,
    which: Attachment,
    source: &Target,
    wanted: &Wanted<'_>,
) -> Result<Option<Target>, Error> {
    if !resolves(call, which, wanted)? {
        return Ok(None);
    }
    let texture = wanted
        .texture
        .ok_or_else(|| internal("a resolve without its texture"))?;
    if matches!(
        sampled_type(source.texture.pixel_format()),
        Some(SampledType::Uint | SampledType::Sint | SampledType::Uint64)
    ) {
        return Err(problem(call, RenderPassProblem::ResolveFormat(which)));
    }
    let destination = destination(call, device, which, source, texture, wanted)?
        .ok_or_else(|| internal("a single-sampled resolve left untouched"))?;
    // A level of another extent, measured of a multisampled attachment alone (`color`), stays
    // refused.
    if destination.target.extent != source.extent {
        return Err(problem(call, RenderPassProblem::ResolveSize(which)));
    }
    let copied = source.texture.storage_mode() != StorageMode::Memoryless
        && source
            .texture
            .image_plan()?
            .usage
            .contains(ImageUsageFlags::TRANSFER_SRC)
        && !super::render_pass::not_2d(source);
    if !copied
        || !destination.usage.contains(ImageUsageFlags::TRANSFER_DST)
        || super::render_pass::not_2d(&destination.target)
    {
        return Err(Error::NotImplemented(RuntimeGap::Render));
    }
    Ok(Some(destination.target))
}

/// The resolve destination of `source` in `texture` at `wanted`'s level, slice and depth plane,
/// checked as Metal checks it (but for `source`'s sample count, `single_sampled`); `None` for a
/// multisampled color resolve texture of another sample count than a multisampled `source`'s,
/// and for a resolve texture of a format a multisampled `source`'s tile is not stored into
/// ([`untouched_across_formats`]), which plain Metal leaves untouched (the module docs).
fn destination(
    call: &'static str,
    device: &Device,
    which: Attachment,
    source: &Target,
    texture: &Texture,
    wanted: &Wanted<'_>,
) -> Result<Option<Destination>, Error> {
    if !texture.device().same(device) {
        return Err(texture.device().foreign(call, device));
    }
    if texture.storage_mode() == StorageMode::Memoryless {
        return Err(problem(call, RenderPassProblem::MemorylessResolve(which)));
    }
    let facts = Facts::of(texture.record());
    // A multisampled color resolve texture of another sample count is left untouched; one of
    // the attachment's count (plain Metal copies the samples, not built) and a depth or stencil
    // one (a layout pattern) are refused.
    let untouched = facts.samples > 1
        && source.samples > 1
        && facts.samples != source.samples
        && matches!(which, Attachment::Color(_));
    if facts.samples > 1 && !untouched {
        return Err(problem(call, RenderPassProblem::ResolveSamples(which)));
    }
    // A multisampled color attachment resolving into its format's sRGB twin resolves in the
    // resolve texture's encoding (the module docs): it resolves in the resolve texture's format,
    // by the crate-owned resolve. One whose tile Metal does not store into the resolve texture's
    // format leaves that texture untouched.
    let pixel_format = source.texture.pixel_format();
    let twin = source.samples > 1 && srgb_twin(pixel_format) == Some(facts.pixel_format);
    let across = source.samples > 1
        && facts.samples == 1
        && matches!(which, Attachment::Color(_))
        && untouched_across_formats(pixel_format, facts.pixel_format);
    if facts.pixel_format != pixel_format && !twin && !across {
        // The other channel order of the 8-bit four-channel family: measured, not built.
        if source.samples > 1 && other_channel_order(pixel_format, facts.pixel_format) {
            return Err(Error::NotImplemented(RuntimeGap::Render));
        }
        return Err(problem(call, RenderPassProblem::ResolveFormat(which)));
    }
    match facts.texture_type {
        TextureType::Type2DMultisample | TextureType::Type2DMultisampleArray if untouched => {}
        TextureType::Type2D
        | TextureType::Type2DArray
        | TextureType::Cube
        | TextureType::CubeArray => {}
        // A texture buffer is resolved into through its attachment twin, as it renders
        // (`Texture::attachment_twin`; Metal resolves into one plain, the probe's `tb_resolve`).
        TextureType::TextureBuffer => {}
        // A 3D one is resolved into from its depth plane (the probe's `t3d_resolve_plane3`),
        // through a 2D view of its planes, and a 1D one through a 1D view (`t1d_resolve`: Metal
        // resolves plain, its validation stops the pass).
        TextureType::Type3D | TextureType::Type1D | TextureType::Type1DArray => {}
        _ => return Err(Error::NotImplemented(RuntimeGap::Render)),
    }
    let volume = facts.texture_type == TextureType::Type3D;
    let planes = |level: u64| (facts.extent[2] >> level.min(31)).max(1);
    // An index the texture's type does not have is ignored.
    let (slice, depth_plane) =
        super::render_pass::used_indices(&facts, wanted.slice, wanted.depth_plane);
    let past = if volume {
        wanted.level >= u64::from(facts.levels)
            || slice != 0
            || depth_plane >= u64::from(planes(wanted.level))
    } else {
        wanted.level >= u64::from(facts.levels)
            || slice >= u64::from(facts.slices)
            || depth_plane != 0
    };
    if past {
        return Err(problem(call, RenderPassProblem::ResolveSubresource(which)));
    }
    // The layer the resolve begins at, and the layers there are from it: a 3D texture's planes.
    let (first, available) = if volume {
        (depth_plane as u32, planes(wanted.level))
    } else {
        (
            facts.base_layer + slice as u32,
            facts.base_layer + facts.slices,
        )
    };
    let level = wanted.level as u32;
    let extent = [
        (facts.extent[0] >> level).max(1),
        (facts.extent[1] >> level).max(1),
    ];
    // A level of another extent resolves the overlap (`color` refuses a color one reaching past
    // the attachment without a clear); one of another sample count or a format the attachment's
    // tile is not stored into is left untouched where its extent is the attachment's
    // (measured), and refused elsewhere (unmeasured).
    if untouched || across {
        return match extent == source.extent {
            true => Ok(None),
            false => Err(problem(call, RenderPassProblem::ResolveSize(which))),
        };
    }
    // A view of another format is resolved into in that format, as it renders
    // (`render_pass::attachment_format`), and the sRGB twin of the attachment's format in the
    // twin's own, the twin among the attachment's image's view formats. A 2D texture over a buffer
    // is resolved into through its aliasing image or its shadow, as it renders
    // (`render/shadows.rs`), a texture buffer through its attachment twin, and a rotated texture
    // through its twin of the pass's frame.
    let texture = &texture.pass_texture()?;
    let plan = texture.image_plan()?;
    let wanted_format = match twin {
        true => facts.pixel_format,
        false => pixel_format,
    };
    let (format, usage) =
        match super::render_pass::attachment_format(which, plan, wanted_format, device) {
            Ok(format) => (format, plan.usage),
            // A view in a format the device does not render resolves through a scratch of the
            // source's format (the same Metal format), copied in bytes (`Destination::attach`).
            // A twin resolve has no such scratch: the twin's own format is the one format Metal's
            // resolve encodes into, so a device that does not render it resolves nothing exactly.
            Err(Error::Unsupported(_)) if !twin => (
                source.format,
                plan.usage.difference(
                    ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                ),
            ),
            Err(error) => return Err(error),
        };
    // A subpass's resolve attachment is of its color attachment's format
    // (`VUID-VkSubpassDescription-pResolveAttachments-00850`), so every resolve but a twin's is
    // resolved in the attachment's format. A twin's is drawn after the pass in the resolve
    // texture's, which Metal encodes into (the module docs).
    if !twin && format != source.format {
        return Err(Error::NotImplemented(RuntimeGap::Render));
    }
    // Layer `i` of a layered pass resolves into slice `resolveSlice + i`, and a layer past the
    // texture's slices is not resolved (a Metal probe, an M3 Max and an M4 alike): the resolve
    // writes the layers that fit, at least the first. A depth or stencil attachment resolves its
    // first layer alone, into slice `resolveSlice`; the later layers are not resolved (a probe, an
    // M3 Max and an M4 alike).
    let layers = match which {
        Attachment::Color(_) => source.layers,
        Attachment::Depth | Attachment::Stencil => 1,
    };
    let fitting = (available - first).min(layers);
    let resolved = Target {
        texture: texture.clone(),
        level: facts.base_level + level,
        layer: first,
        layers: fitting,
        format,
        samples: 1,
        extent,
    };
    Ok(Some(Destination {
        target: resolved,
        usage,
        twin,
    }))
}

/// Whether `a` and `b` are `RGBA8Unorm` and `BGRA8Unorm`, either of them its sRGB twin: one
/// resolves into the other in the destination's channel order and the attachment's encoding
/// (the module docs).
fn other_channel_order(a: PixelFormat, b: PixelFormat) -> bool {
    use PixelFormat as P;
    matches!(
        (block_family(a), block_family(b)),
        (P::RGBA8Unorm, P::BGRA8Unorm) | (P::BGRA8Unorm, P::RGBA8Unorm)
    )
}

/// An image of `source`'s format, extent and layers, of `samples`, that the pass resolves into,
/// then copies from (one sample), or that a copy of `source` is resolved from (its samples,
/// [`ResolveSource`], and the exact color resolve's copy, `color_resolve.rs`).
pub(super) fn scratch(
    maker: &CommandBuffer,
    source: &Target,
    usage: ImageUsageFlags,
    samples: u32,
) -> Result<Target, Error> {
    scratch_viewed(maker, source, usage, samples, None)
}

/// [`scratch`], its image also taking views in `view` where one is named (mutable to it alone):
/// the color resolve's copy that it reads through the device's sRGB decode (`color_resolve.rs`).
pub(super) fn scratch_viewed(
    maker: &CommandBuffer,
    source: &Target,
    usage: ImageUsageFlags,
    samples: u32,
    view: Option<Format>,
) -> Result<Target, Error> {
    let format = source.format;
    let [width, height] = source.extent;
    let layers = source.layers;
    let pixel_format = source.texture.pixel_format();
    let descriptor = TextureDescriptor {
        texture_type: match (samples > 1, layers > 1) {
            (false, false) => TextureType::Type2D,
            (false, true) => TextureType::Type2DArray,
            (true, false) => TextureType::Type2DMultisample,
            (true, true) => TextureType::Type2DMultisampleArray,
        },
        pixel_format,
        width: width.into(),
        height: height.into(),
        array_length: layers.into(),
        sample_count: samples.into(),
        usage: TextureUsage::RENDER_TARGET,
        resource_options: ResourceOptions::private(),
        ..TextureDescriptor::default()
    };
    let aspects = aspects(pixel_format).unwrap_or(ImageAspectFlags::COLOR);
    let plan = ImagePlan {
        pixel_format,
        flags: match view {
            Some(_) => ImageCreateFlags::MUTABLE_FORMAT,
            None => ImageCreateFlags::empty(),
        },
        image_type: ImageType::TYPE_2D,
        format,
        extent: [width, height, 1],
        levels: 1,
        layers,
        samples: sample_count(samples)
            .ok_or_else(|| internal("an attachment of a sample count Vulkan lacks"))?,
        usage: usage | ImageUsageFlags::TRANSFER_SRC,
        view_formats: view.map_or_else(Vec::new, |view| vec![format, view]),
        mapping: FormatMapping::direct(format, aspects),
        unavailable: TextureUsage::SHADER_READ
            | TextureUsage::SHADER_WRITE
            | TextureUsage::SHADER_ATOMIC
            | TextureUsage::PIXEL_FORMAT_VIEW,
    };
    let texture = Device::new_planned_texture(maker, &descriptor, plan)?;
    Ok(Target {
        texture,
        level: 0,
        layer: 0,
        layers,
        format,
        samples,
        extent: source.extent,
    })
}

/// Whether `device`'s render pass resolves a depth/stencil attachment from its image's first
/// layer and level, and into its resolve attachment image's, whatever layer and level their
/// views begin at (`RuntimeDriverDefect::DepthStencilResolveFirstSubresource`).
pub(super) fn first_subresource_only(device: &Device) -> bool {
    device
        .properties()
        .driver_defects
        .contains(&RuntimeDriverDefect::DepthStencilResolveFirstSubresource)
}

/// Whether `target` begins at its image's first layer and level.
fn at_first_subresource(target: &Target) -> bool {
    target.layer == 0 && target.level == 0
}

/// The copy a pass's depth/stencil resolves read on a device whose render pass resolves from the
/// attachment image's first layer and level ([`first_subresource_only`]), where the pass's
/// depth/stencil attachment begins at another: the pass's layers of the attachment, every aspect
/// of its format, copied after the pass's instances into a scratch image of its format, samples,
/// extent and layers, from the first (`vkCmdCopyImage` of one format and sample count). The pass
/// then resolves in an instance of no draws whose depth/stencil attachment is the scratch image,
/// as do its resolves after the pass (`render.rs`).
pub(super) struct ResolveSource {
    /// The scratch image: the depth/stencil attachment of the instances that resolve.
    pub(super) target: Target,
    from: vk::Image,
    to: vk::Image,
    region: vk::ImageCopy,
    /// The copy's read of the attachment and write of the scratch image.
    pub(super) uses: Vec<Use<ResourceKey>>,
}

impl ResolveSource {
    /// The copy `layout`'s depth/stencil resolves read on `device`: `None` where the device
    /// resolves where the attachment lies, the attachment begins at its image's first layer and
    /// level, or no render pass resolves it (the draws of `depth_stencil_resolve.rs` read it
    /// through views). Refused `NotImplemented(DepthStencilResolve)` where the attachment's image
    /// takes no copy (a memoryless texture's).
    pub(super) fn plan(
        maker: &CommandBuffer,
        layout: &PassLayout,
    ) -> Result<Option<ResolveSource>, Error> {
        let device = maker.device();
        let Some(ds) = &layout.depth_stencil else {
            return Ok(None);
        };
        let source = &ds.target;
        let resolved = ds.resolve.as_ref().is_some_and(|r| !r.steps.is_empty());
        if !resolved || !first_subresource_only(device) || at_first_subresource(source) {
            return Ok(None);
        }
        let copied = source.texture.storage_mode() != StorageMode::Memoryless
            && source
                .texture
                .image_plan()?
                .usage
                .contains(ImageUsageFlags::TRANSFER_SRC);
        if !copied {
            return Err(Error::NotImplemented(RuntimeGap::DepthStencilResolve));
        }
        let usage = ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | ImageUsageFlags::TRANSFER_DST;
        let target = scratch(maker, source, usage, source.samples)?;
        let aspect_mask = image_aspects(source.format);
        let layers = |t: &Target| vk::ImageSubresourceLayers {
            aspect_mask: aspect_mask.into(),
            mip_level: t.level,
            base_array_layer: t.layer,
            layer_count: t.layers,
        };
        let [width, height] = source.extent;
        let region = vk::ImageCopy {
            src_subresource: layers(source),
            src_offset: vk::Offset3D::default(),
            dst_subresource: layers(&target),
            dst_offset: vk::Offset3D::default(),
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };
        let read = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let at = |t: &Target| t.level..t.level + 1;
        let on = |t: &Target| t.layer..t.layer + t.layers;
        let uses = vec![
            source
                .texture
                .image_use(at(source), on(source), read, false)?,
            target
                .texture
                .image_use(at(&target), on(&target), write, true)?,
        ];
        Ok(Some(ResolveSource {
            from: source.texture.raw_image()?,
            to: target.texture.raw_image()?,
            region,
            uses,
            target,
        }))
    }

    /// `layout` with the scratch image as its depth/stencil attachment: the layout of the
    /// instances that resolve.
    pub(super) fn resolving(&self, layout: &PassLayout) -> PassLayout {
        let mut resolving = layout.clone();
        if let Some(ds) = resolving.depth_stencil.as_mut() {
            ds.target = self.target.clone();
        }
        resolving
    }

    /// Record the copy, after a barrier from everything before.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only the caller records it; both
    /// images are live until it completes (the encoder retains the attachment, and this the
    /// scratch image), and rest in `GENERAL`.
    pub(super) unsafe fn record(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        let general = vk::ImageLayout::GENERAL;
        // SAFETY: the caller's contract; the region is the attachment's level and the pass's
        // layers of both images, which have one format, sample count and extent.
        unsafe {
            Barrier::everything().record(raw, command_buffer);
            raw.cmd_copy_image(
                command_buffer,
                self.from,
                general,
                self.to,
                general,
                &[self.region],
            );
        }
    }
}

/// The copies of a pass's scratch resolves into their textures, over the render area within the
/// texture's level (a smaller level takes the overlap) and the texture's layers (those of a layered
/// pass that fit in its slices), each after the clear its resolve names of those layers.
///
/// A copy of every aspect its format has is one image copy. A copy of one aspect of a
/// depth/stencil format with two (a resolve of one aspect at a time, or one aspect resolved into a
/// scratch image) goes out of the scratch image into a staging buffer and from it into the
/// texture: a buffer copy's bytes are the aspect's alone by definition, while lavapipe's image
/// copy of `D32_SFLOAT_S8_UINT`'s stencil aspect also wrote the depth aspect
/// (`runtime_lane::resolves`, `ds_unresolved_sample0`, at 1ae4ebbc).
pub(super) struct ResolveCopies {
    /// `(texture, clear color bits, its level and layers)` of each clear before the copies.
    clears: Vec<(vk::Image, [u32; 4], vk::ImageSubresourceRange)>,
    /// `(scratch, texture, region)` of each image copy.
    copies: Vec<(vk::Image, vk::Image, vk::ImageCopy)>,
    /// The aspect copies through a staging buffer.
    staged: Option<Staged>,
    uses: Vec<Use<ResourceKey>>,
}

/// One-aspect copies through a staging buffer: each into the buffer, then each out of it.
struct Staged {
    staging: Buffer,
    into_buffer: Vec<(vk::Image, vk::BufferImageCopy)>,
    out_of_buffer: Vec<(vk::Image, vk::BufferImageCopy)>,
}

impl ResolveCopies {
    /// The copies of `resolves` (each with the aspects it resolved) over the render area
    /// `extent`; `None` when no resolve went through a scratch image.
    pub(super) fn plan(
        device: &Device,
        resolves: &[(&Resolve, ImageAspectFlags)],
        extent: [u32; 2],
    ) -> Result<Option<ResolveCopies>, Error> {
        let read = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let levels = |t: &Target| t.level..t.level + 1;
        // A copy takes the texture's layers (those of a layered pass that fit in its slices)
        // from the scratch image's first.
        let on = |t: &Target, count: u32| t.layer..t.layer + count;
        let layers =
            |t: &Target, aspect: ImageAspectFlags, count: u32| vk::ImageSubresourceLayers {
                aspect_mask: aspect.into(),
                mip_level: t.level,
                base_array_layer: t.layer,
                layer_count: count,
            };
        let mut clears = Vec::new();
        let mut copies = Vec::new();
        let mut uses = Vec::new();
        // `(scratch, texture, aspect, offset, layers)` of each staged copy.
        let mut aspect_copies = Vec::new();
        let mut length = 0u64;
        for &(resolve, aspect) in resolves {
            let Some(to) = &resolve.copy_to else {
                continue;
            };
            let from = &resolve.attachment;
            let count = to.layers;
            let [width, height] = [extent[0].min(to.extent[0]), extent[1].min(to.extent[1])];
            uses.push(
                from.texture
                    .image_use(levels(from), on(from, count), read, false)?,
            );
            uses.push(
                to.texture
                    .image_use(levels(to), on(to, count), write, true)?,
            );
            if let Some(bits) = resolve.clear {
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: to.level,
                    level_count: 1,
                    base_array_layer: to.layer,
                    layer_count: count,
                };
                clears.push((to.texture.raw_image()?, bits, range));
            }
            if aspect == image_aspects(from.format) {
                let region = vk::ImageCopy {
                    src_subresource: layers(from, aspect, count),
                    src_offset: vk::Offset3D::default(),
                    dst_subresource: layers(to, aspect, count),
                    dst_offset: vk::Offset3D::default(),
                    extent: vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    },
                };
                copies.push((from.texture.raw_image()?, to.texture.raw_image()?, region));
                continue;
            }
            let bytes = copy_units(from.format, from.texture.pixel_format())
                .into_iter()
                .find(|unit| unit.0 == aspect)
                .map(|unit| u64::from(unit.1))
                .ok_or_else(|| internal("a resolve copy of an aspect its format lacks"))?;
            // A depth/stencil copy's buffer offset is a multiple of 4; its layers lie one after
            // another.
            let offset = length.next_multiple_of(4);
            let texels = u64::from(width) * u64::from(height);
            length = offset + bytes * texels * u64::from(count);
            aspect_copies.push((from, to, aspect, offset, count, [width, height]));
        }
        let staged = if aspect_copies.is_empty() {
            None
        } else {
            let staging = device.own_buffer(length, ResourceOptions::private())?;
            uses.push(writes(&staging, 0..length));
            uses.push(reads(&staging, 0..length));
            let (mut into_buffer, mut out_of_buffer) = (Vec::new(), Vec::new());
            for (from, to, aspect, offset, count, [width, height]) in aspect_copies {
                let region = |target: &Target| vk::BufferImageCopy {
                    buffer_offset: staging.at(offset).1,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: layers(target, aspect, count),
                    image_offset: vk::Offset3D::default(),
                    image_extent: vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    },
                };
                into_buffer.push((from.texture.raw_image()?, region(from)));
                out_of_buffer.push((to.texture.raw_image()?, region(to)));
            }
            Some(Staged {
                staging,
                into_buffer,
                out_of_buffer,
            })
        };
        Ok(
            (!copies.is_empty() || staged.is_some()).then_some(ResolveCopies {
                clears,
                copies,
                staged,
                uses,
            }),
        )
    }

    /// Record the copies in `core`'s command buffer after the pass, between barriers from
    /// everything before and to everything after, and hold the staging buffer until the
    /// command buffer completes.
    pub(super) fn record(self, core: &mut EncoderCore) {
        let ResolveCopies {
            clears,
            copies,
            staged,
            uses,
        } = self;
        core.barrier(Barrier::everything());
        core.command(TRANSFER, &uses, |device, raw| {
            let general = vk::ImageLayout::GENERAL;
            // SAFETY: the encoder's command buffer is recording outside a render pass; the images
            // and the staging buffer are live until it completes (the encoder retains the
            // textures and the scratch images, and this the buffer); every image rests in
            // `GENERAL`, and each region lies in its image and in the buffer (the scratch image is
            // the attachment's extent and has the pass's layers, the region the render area
            // within the resolve level, which has the copied layers; the buffer as planned); a
            // cleared image has transfer-destination usage, as its copy needs.
            unsafe {
                for (image, bits, range) in &clears {
                    let color = vk::ClearColorValue { uint32: *bits };
                    device.cmd_clear_color_image(raw, *image, general, &color, &[*range]);
                }
                if !clears.is_empty() {
                    Barrier::everything().record(device, raw);
                }
                for (from, to, region) in &copies {
                    device.cmd_copy_image(raw, *from, general, *to, general, &[*region]);
                }
                if let Some(staged) = &staged {
                    let (buffer, _) = staged.staging.at(0);
                    for (image, region) in &staged.into_buffer {
                        device.cmd_copy_image_to_buffer(raw, *image, general, buffer, &[*region]);
                    }
                    Barrier::everything().record(device, raw);
                    for (image, region) in &staged.out_of_buffer {
                        device.cmd_copy_buffer_to_image(raw, buffer, *image, general, &[*region]);
                    }
                }
            }
        });
        core.barrier(Barrier::everything());
        if let Some(staged) = staged {
            core.retain(staged.staging);
        }
    }
}
