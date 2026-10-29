//! Framebuffer fetch through the ordered pixel interlock.
//!
//! **Where.** Metal orders a fetch after every earlier fragment of the pixel, across draws.
//! Vulkan orders an input-attachment read so only under rasterization-order attachment access
//! (`color`). A device without that access but with `fragmentShaderPixelInterlock`
//! ([`interlock_orders`]: RADV, ANV, and MoltenVK where measured) orders a fetch instead through
//! the ordered pixel interlock, the mechanism of raster order groups and imageblocks: each color
//! attachment the function reads becomes a storage image of the attachment
//! (`ResourceKind::TileAttachment(n)`), read at the fragment's pixel (the integer part of
//! `FragCoord.xy`) at each use, and, when the function writes that attachment through its first
//! (index 0) output, written there at the return in place of the output
//! (`GraphicsState::storage_attachments`; colors it writes but does not read stay ordinary outputs,
//! but those the pipeline's state names: an extended-range attachment it blends, below, and a color
//! its draw's render pass instance keeps out of its subpass for other draws' storage form,
//! `VariantKey::interlocked_colors`). The whole body runs inside the critical section
//! (`fragment::wrap_entry`), so the read, the body, and the write of every fragment of a pixel are
//! one step in primitive order, across the draws and instances of a subpass, and the device-scope
//! coherence of every access inside the critical section (`sync::memory_model`, the image is
//! `Coherent` under `GLSL450`, each access carries `MakeTexel*` under the Vulkan model, as the
//! body's other accesses do) makes each step's write visible to the next: Metal's order. A
//! discarded fragment writes nothing: a demoted invocation's image writes have no effect, a
//! terminate-form discard returns before the write (the entry kills after the critical section),
//! and a predicated one skips it.
//!
//! **Exactness, from the pipeline's state** (`LoweringInputs::framebuffer_fetch`, which the
//! render pipeline compile sets on such a device; without it the function is
//! `NeedsInput(FramebufferFetch)`). The storage write happens inside the shader, so it is exact
//! only where nothing after the shader would drop or change Metal's color write ([`storage_form`],
//! which link applies again to the pipeline it links into):
//!
//! - one view and one layer: no vertex amplification (the binder also refuses a pass of more
//!   than one layer);
//! - for a written attachment whose write mask covers some of its format's channels (one that
//!   covers none stores nothing, so none of this bears on it): no blending but one and zero, added,
//!   for color and alpha, which stores the source as no blending does (a signed normalized format's
//!   aside: its NaN stores another code under it), or a blend the shader computes as Metal does,
//!   an extended-range, a 16-bit float or an 8-bit unorm attachment's (all below), or the generic
//!   blend of another format the function reads (a near-equivalent, below), no alpha to coverage
//!   (it can drop the
//!   fragment's samples), no signed normalized format under a write mask that keeps some of its
//!   channels (its two codes of -1 read alike, so a channel written back would lose the texel's
//!   bits), and, with a depth or stencil attachment, early fragment tests: the module declares them
//!   there when they keep the function's writes Metal's (Metal's function declares them, or it runs
//!   no discard, exports no depth, stencil, or coverage, and writes no other memory), so a fragment
//!   the tests fail never runs. Without a depth or stencil attachment no test can fail, and the
//!   module declares none (unless Metal's function does). A function that only its discard keeps
//!   from them stores before the tests, which each draw's plan checks
//!   (`RuntimeFact::TestedStorageWrites`): a draw whose tests drop no fragment runs it, one whose
//!   tests write neither attachment and count nothing runs the variant that declares them
//!   (`LoweringInputs::read_only_tests`), and any other is refused. A function that writes other
//!   memory, or returns depth, stencil, or a sample mask, stores before the tests too, and states
//!   the fact without that variant: Metal runs its memory writes for the fragments its tests then
//!   drop and tests what it returns, which early tests would not, so only a draw
//!   whose tests drop no fragment runs it, and any other is refused at the depth (or stencil)
//!   attachment;
//! - a direct format, declared without a format where the device grants the format's storage
//!   access without one, so that one module serves every format the function reads alike, else with
//!   its SPIR-V image format where it has one (no format feature beyond storage,
//!   `shaderStorageImageExtendedFormats` for an extended one); a BGRA-order format (`BGRA8Unorm`,
//!   `BGR10A2Unorm`), which has no SPIR-V image format, on a device that accesses it without one
//!   in neither direction (ANV), through a view of the attachment in its RGBA-order twin
//!   (`R8G8B8A8_UNORM`, `A2B10G10R10_UNORM_PACK32`; `StorageView::rgba_view`,
//!   `StorageAttachmentUse::view_alias_of`), declared as the twin is, every texel read from the
//!   image or written to it with red and blue exchanged ([`metal_order`]): the device converts
//!   each channel as it converts the attachment's own, so a read, a write, the channels a write
//!   mask keeps and the per-sample forms hold the values the attachment's own view would; an
//!   `A8Unorm` attachment whose image holds Metal's alpha in red (`R8_UNORM` on a device without
//!   `A8_UNORM`, `StorageView::alpha_in_red`) through that image, every texel read from it taken
//!   to Metal's `(0, 0, 0, r)` and every one written to it holding Metal's alpha in red
//!   ([`metal_order`], [`image_order`]), so a read, a merge of the channels a write mask keeps
//!   (in Metal's channels, of which only alpha is stored) and the per-sample forms are Metal's
//!   (probe `a8`'s uses on an M3 Max under macOS 27.0.1); an
//!   8-bit sRGB format the device renders (`R8Unorm_sRGB`, `RG8Unorm_sRGB`, `RGBA8Unorm_sRGB`,
//!   `BGRA8Unorm_sRGB`) through a view of the attachment in the unsigned integer format of its
//!   texel (`R8_UINT`, `R16_UINT`, `R32_UINT`; `StorageView::uint_view`), whose bytes the function
//!   converts itself by the device's conversion (`arith::pack`'s `unpack` and `pack` of
//!   `unorm4x8_srgb`: the 8-bit instruction and the sRGB transfer by the device's `Pow`,
//!   `PrecisionNote::DevicePackConversion`): a read decodes each byte, a channel the format lacks
//!   0 and alpha 1, a write encodes and keeps the format's bytes; a blended write encodes Metal's blend, its
//!   8-bit unorm blend of the format's unorm twin over the texel as a fetch decodes it
//!   (`extended_range::Blend::srgb8`, [`srgb_texel`]: on the M3 Max every channel of the blends
//!   measured, both maths, `half4` and `float4`, each format), and where that blend is not built
//!   (a blend of the second source; under fast math a format without alpha's lerps, which Metal
//!   computes in forms of its own) or the output is fewer than four lanes (blended as four whose
//!   missing lanes are zero), the rounded forms: Metal's without fast math for four lanes, else a
//!   near-equivalent the function notes (`PrecisionNote::UnmeasuredFetchBlend`); a packed 16-bit
//!   format the device samples but does not render
//!   (`metal::format::WordAttachment`: `A1BGR5Unorm`, `ABGR4Unorm` on the RTX 5080) through a view
//!   of the attachment in its word's unsigned integer format (`R16_UINT`, `StorageView::word`): a
//!   write stores Metal's word of the output (`stage_io::word_color::encode`), the fields of the
//!   channels its write mask keeps taken from the word it reads first, so they keep their codes as
//!   Metal's tile keeps them ([`word_texel`]). The pipeline sends such an attachment here where it
//!   writes it unblended (or blended one and zero) under a mask of some of its channels, or
//!   blended otherwise (`color_writes::stored_word`). A blended write computes Metal's packed
//!   16-bit blend of a `half4` or `float4` output over the word it reads
//!   (`extended_range::Blend::packed16`, `stage_io::word_color::decode`): exact without fast math
//!   over an attachment holding the codes Metal's tile does, within one code of Metal's under fast
//!   math (Metal regroups some sums otherwise, 3,527 of 94,732,288 channels measured off by one),
//!   and over a clear's or an earlier blend's result in the same pass, which Metal's tile holds as
//!   a `half` finer than the codes the attachment holds (`PrecisionNote::DeviceColorConversion`,
//!   noted for every such blend). An output of fewer lanes and a factor of the second source are
//!   not built. A read of one decodes the word's channels as Metal's programmable blending hands
//!   them to the function, each code over its maximum code rounded to `half`, nearest even
//!   ([`value`], the same decode; measured for a `float4` and a `half4` color input,
//!   `metal::format::word`), which Metal's tile holds finer after a clear or an earlier draw of the
//!   same pass, as it holds a blend's destination. A read of an 8-bit sRGB word and of an
//!   `RGB9E5Float` one is not built (`NotImplemented(Format)`).
//!
//! **Per sample**. Metal runs a function that reads a color attachment of a multisampled pass once
//! per covered sample, reading its own sample, in draw order per sample, with `[[position]]`'s x
//! and y at the pixel center (measured). Compiled for such a pipeline
//! (`LoweringInputs::multisampled`), the function reads (and writes) the attachment's sample at its
//! `SampleId` ([`plan_sample`]), which runs it once per covered sample under Vulkan's sample
//! shading; link's sample invocations then read its pixel inputs where Metal does or refuse them
//! ([`StorageSamples`]). Where the device has multisampled storage images, the image is
//! multisampled, of the pipeline's samples (`GraphicsState::rasterization_samples`, which link
//! checks against `storageImageSampleCounts`), read and written with the `Sample` operand.
//! Elsewhere (ANV, MoltenVK) it holds the samples as planes: a single-sample 2D-array image of one
//! layer per sample, sample `s` of pixel `(x, y)` at texel `(x, y, s)`, which the consumer fills
//! from the multisampled attachment before the draws and writes back after them
//! (`StorageAttachmentUse::view`). Either needs `sampleRateShading`. The pixel interlock orders
//! every invocation covering the pixel, and so each sample's. A returned sample mask keeps an
//! invocation's write exactly when it sets the invocation's sample's bit (measured), so each write
//! is gated on that bit ([`sample_bit`]); at one sample on bit 0 (measured). A driver that drops
//! the image writes of an early-tests fragment shader in a multisampled pass
//! (`DriverWorkaround::EarlyTestsDropMultisampledImageWrites`: MoltenVK on Apple's GPU) refuses the
//! per-sample form's writes where the module declares early tests (`Needs::DriverDefect`). Where
//! only a depth or stencil attachment would have the module declare them, it declares none there:
//! it stores before the tests and states `TestedStorageWrites` without the early-tests variant, so
//! only a draw whose tests drop no fragment runs it; a function whose Metal source declares early
//! tests keeps the refusal.
//!
//! **A write mask** short of the format's channels: Metal keeps the channels the mask leaves out.
//! The write reads the texel at the pixel first and writes those channels back as it read them
//! ([`StorageView::kept`]); inside the interlock no other fragment of the pixel writes the texel in
//! between, so every channel keeps its value (a `half` texel's NaN in the device's decode,
//! `PrecisionNote::DeviceNanEncoding`). A mask that covers none of the format's channels writes
//! nothing. Link checks the kept channels against the pipeline it links into. Every mask that keeps
//! some of a format's channels compiles alike: the function compiles under the mask's class
//! ([`compiled_write_mask`]: none, every channel, or some), and a write of the third class keeps
//! the lanes whose bits the draw's `PushField::StorageKeptChannels` sets
//! ([`StorageView::kept_at_draw`]), which the binder writes from the pipeline's
//! `StorageAttachmentUse::kept`, so the masks of a class share one module and one fragment key
//! (Metal's own backend compiles each new mask). An extended-range attachment's kept
//! channels feed its blend and encode, and an sRGB one merges its bytes, so those compile each mask
//! (no draw word).
//!
//! **An extended-range attachment** (`metal::format::ExtendedRange`): the image holds Metal's raw
//! texels in the raw unsigned format, so the storage image is declared of that format's
//! `uint` texels ([`StorageView::extended`]), and a read decodes the texel into Metal's tile value
//! (`extended_range::tile_value`: each code as the tile holds it, rounded to `half`, a `float` read
//! alike). Such an attachment the function writes under blending takes
//! this form whether or not the function reads it, since no device blends the raw texels: the write
//! reads the texel's tile value, blends the output over it as Metal does, and stores the kernel
//! write's encode of the result ([`StorageView::blend`]), the channels a write mask keeps as Metal
//! keeps them (`extended_range::encode_keeping`). Metal's model (`extended_range::Blend`) holds a
//! `half4` output under every factor and operation, the blend color read as push data
//! (`PushField::BlendColor`), and a `float4` output under the blends whose color lanes take one
//! term alone, rounded to `half` first; the rest is refused by name
//! (`Feature::ExtendedRangeBlend`). A write of every channel with blending disabled stores the
//! texel Metal's unblended write stores, a `float` output rounded to `half` and encoded as a
//! kernel's write (`stage_io::extended_color::texel`). An unblended write that stores some of its
//! channels is not built; one whose mask covers none of them stores nothing, as above.
//!
//! **The `half` shadow**: Metal's tile memory holds such an attachment as `half` through a render
//! pass and rounds it to the codes only at the store, so a blend or a fetch onto what an earlier
//! draw of the pass wrote sees that draw's `half`, not its codes. A function compiled for an
//! instance that holds the attachment's shadow (`LoweringInputs::extended_shadows`, one sample)
//! declares a second storage image of the attachment's extent in `Rgba16f`
//! (`ResourceKind::ExtendedShadow`), each texel what the tile holds. A read takes Metal's read of
//! the shadow's texel (`extended_range::tile_read`: the color lanes clamped to the blend's source
//! range, alpha to `[0, 1]`, a NaN to the low end, `BGR10_XR`'s alpha 1), a blend reads its
//! destination so, and every write stores the value it computed, unclamped, the blend's result or
//! the unblended output's `half`, in the shadow and its encode in the codes, both inside the
//! interlock (the tile holds a write past the range and stores its encode, and every
//! read clamps); channels a write mask keeps stay as the shadow and the codes held them. Every
//! value the shadow holds is a `half`, so the device's conversions both ways are exact, and the
//! codes always hold the encode of the last `half` written to each channel. A shadowed attachment
//! the storage form does not take, or of a padded or bfloat form, or of more than one sample, is
//! refused whatever the policy (the runtime keeps such a draw off the shadow).
//!
//! **A 16-bit float attachment's blend** (`R16Float`, `RG16Float`, `RGBA16Float`): no Vulkan blend
//! applies to a storage write, so the shader computes Metal's: the write reads the texel at the pixel inside the interlock, blends the function's `half`
//! or `float` output of one to four lanes over it (a second (index 1) output beside it, and a
//! factor of the second source, take the generic blend, below), and stores the
//! result (`extended_range::Blend::half_float`, [`half_blended`]). Metal's compiler appends the
//! blend to the function, in the output's type: a `half` output blends in `half` arithmetic, a
//! `float` one in `float`, its result converted to `half` by the storage image's store, as Metal's
//! write of a `float` converts it (toward zero) where the profile takes
//! `DriverDeviation::MetalHalfStorageWrites` and as the device does elsewhere
//! (`PrecisionNote::DeviceColorConversion`), and to nearest even in the shader under
//! `TextureWriteRoundingMode::NearestEven`, which no measured store gives. An output of fewer
//! than four lanes (Metal takes none with fewer than the format's channels, so it writes `R16Float`
//! or `RG16Float`, which have no alpha) blends as four whose missing lanes play no part, but where
//! the color channels' factors read the source alpha (`SourceAlpha`, `OneMinusSourceAlpha`): that
//! alpha is undefined to Metal's compiler, which stores `+0` in each color channel under fast math
//! and its NaN (`0x7e00`) without, whatever the operands (`SourceAlphaSaturated` is its minimum
//! with one minus the destination alpha, 0 in such a format). Our probe: plain Metal on the M3 Max
//! under macOS 27.0.1, our own MSL, `half` and `float` outputs of one, two and three lanes into
//! `R16Float` and `RG16Float` (each width the format takes), the 677 and 1,350 states of our probe
//! below, and the 677 under `RG16Float`'s partial write masks, fast and safe math: every
//! channel the model's rounded, fitted or constant forms name holds Metal's value (some zero signs
//! differ, a known gap of `docs/FAST-PATHS.md`). Without fast math every product and sum rounds on
//! its own (the rounded forms); under fast math Metal's passes fuse and regroup it (the fast
//! forms). Measured on plain Metal on the M3 Max under macOS 27.0.1, our own MSL, a function
//! reading `[[color(0)]]` beside an unwritten second attachment, 16,384 random pixels a draw
//! (`RGBA16Float`, 677 blend states of one key for color and alpha and 1,350 of a key beside one
//! minus the source alpha: the same states into `R16Float` and `RG16Float`, and the 677 under four
//! write masks that keep some channels): without fast math the rounded forms store Metal's value in
//! every channel of every state, for loaded `half4` and `float4` outputs; under fast math every
//! class whose form is no term, one term, one fused multiply-add, a minimum or maximum, or a blend
//! constant's difference (`BlendColor` with `OneMinusBlendColor`, `BlendAlpha` with
//! `OneMinusBlendAlpha`, subtracted: the minuend's product fused, the subtrahend's rounded first)
//! does (`extended_range::Blend::fitted`), the One over OneMinusSourceAlpha blend included, with
//! the infinities, NaNs, signed zeros and values past `half`'s range. Metal's compiler folds the
//! output's own fast-math arithmetic into the blend (a returned product fused into the blend's one
//! minus the source alpha; a sum that reads the fetched color regrouped by a common factor), so the
//! blend is Metal's exactly where its forms are fitted, the module states its math, and every lane
//! of the output is opaque to it ([`folds_into_blend`]). Elsewhere it is a near-equivalent, and the
//! function records `PrecisionNote::HalfOutputFastMathBlend` (`HalfOutputPrecision` for a module
//! that names no math mode): measured within 2 `half` units in the last place at the operands'
//! magnitude, 16 at the output's under cancellation (below one at the function's own largest term).
//! A `half` output's blend stores a `half` in every channel, so no conversion of the device's
//! bears on it, and a `float` one's only by the store's rounding above; a NaN of the blend is the
//! device's arithmetic's, which the storage image's store converts, noted
//! `PrecisionNote::DeviceNanEncoding` (`stage_io::color_write`; Metal stores each as `0x7e00`). An
//! attachment the function does not read takes the form only where a draw moved it into the
//! storage form (`VariantKey::interlocked_colors`) and its blend is exact; elsewhere the variant
//! is refused and the render pass splits around the draw, whose blend is the device's.
//!
//! **An 8-bit unorm attachment's blend** (`R8Unorm`, `RG8Unorm`, `RGBA8Unorm`, `BGRA8Unorm`) of a
//! `half4` or `float4` output the shader computes as Metal does ([`unorm8_blended`]): the write
//! reads the texel at the pixel inside the interlock in Metal's channel order, clamps the output to
//! `[0, 1]` (a NaN to 0), reads each code `c` as `h(c / 255)` (`f(c / 255)` for a `float4` output)
//! and the blend color clamped to `[0, 1]` (pushed as `PushField::HalfBlendColor`), blends in the
//! output's type in the forms Metal's compiler builds (`extended_range::Blend::unorm8`: fused and
//! regrouped under fast math, each step rounded without) (`extended_range::blend::blend_unorm8`),
//! stores the result, which the storage image's store converts to its code by the device's
//! conversion (`PrecisionNote::DeviceColorConversion`), and keeps the codes of the channels its
//! write mask leaves out. Measured with plain Metal on the M3 Max under macOS 27.0.1: our probe
//! (premultiplied source-over, every `half` alpha in `[0, 1]` against every code of each channel,
//! `half4` and `float4`, fast and safe math, each of the four formats: Metal's code in every
//! channel); our probes (every factor and operation pair for color beside source-over alpha and the
//! reverse, the same pair for both, and random pairs, over every destination code times every
//! source alpha code); our probes (write masks, one color channel and alpha alone among them); our
//! probe (blend colors past both ends). The clamp keeps the output's own arithmetic out of the
//! blend (products, sums and a regroup over the fetched color, each the model's over the function's
//! value), so the blend is Metal's wherever the module states its math (`HalfOutputPrecision` for
//! one naming none). A format without alpha's fast-math lerp, common factor or contracted pair
//! (Metal sums those in forms of their own) and an output of fewer lanes take the generic blend
//! (below), as do a factor of the second source and a second (index 1) output.
//!
//! **A 32-bit float attachment's blend** (`R32Float`, `RG32Float`, `RGBA32Float`) of a `half4` or
//! `float4` output of a module with fast math (or naming no math mode), which the shader computes
//! as Metal does ([`float32_blended`], `extended_range::Blend::float32`): the write reads the texel
//! at the pixel inside the interlock and blends the output over it in `float` throughout, in the
//! forms Metal's compiler folds into the function -- the 16-bit float family's fast forms,
//! every rounding to `float` -- and stores the `f32` the blend gives, the channels its write mask
//! keeps taken back from the texel. Nothing is clamped, as nothing is for a 16-bit float
//! attachment. Measured against the committed recordings of plain Metal on the M3 Max under macOS
//! 27.0.1 of `half-output-blends-r32f`, `-rg32f` and `-rgba32f`, each case 256 pixels of a `half4`
//! output over destinations its own generator draws, every word of all twenty-one of their blend
//! states: of the seventeen states whose factors read no blend color, every word of the two- and
//! four-channel formats is Metal's under these forms, ten of them in forms
//! `extended_range::Blend::fitted` holds without a measurement and two by one
//! (`extended_range::blend::MEASURED_THIRTY_TWO`). A state the measurement does not fit stays with
//! the device's blend and its note, as a 16-bit float one's does: a one-channel attachment's
//! `SourceColor` over `DestinationColor` (Metal rounds the destination's product first there, and
//! the device's blend is Metal's), every state whose factors read the blend color, an output of
//! fewer than four lanes, and a module without fast math, all of which take the generic blend
//! below: both forms are near-equivalents at an unfitted state, and the generic blend's are
//! measured the closer at the one-channel state above. The blend is Metal's only where no lane of
//! the output folds into it ([`folds_into_blend`]) and the module states its math, as a 16-bit
//! float one's is; where a lane folds, the forms stay and the function notes
//! `PrecisionNote::HalfOutputFastMathBlend`. An attachment the function does not read takes the
//! form on the same terms a 16-bit float one does: only where a draw moved it into the storage
//! form (`VariantKey::interlocked_colors`) and its blend is exact.
//!
//! **Another format's blend** (a format held directly whose channels read as floats, but a 16-bit
//! float, a 32-bit float, an sRGB, a word or an extended-range one: a 16-bit normalized, 8-bit
//! signed normalized, 10-bit, packed float or `A8Unorm` format, among others; and an 8-bit unorm
//! one of an output of fewer than four lanes, or under a blend Metal's model there does not hold),
//! one of an output of fewer than four lanes, and a 32-bit float one under a state the
//! measurement above does not fit), whose Metal arithmetic is not built, of an attachment the
//! function reads: no Vulkan blend
//! applies to the storage write, so the shader computes the pipeline's blend itself, the generic
//! float blend ([`generic_blended`], `extended_range::Blend::generic`): the write reads the texel
//! inside the interlock, blends the output's four lanes (a missing one zero) over it in `float`,
//! each factor, product and sum rounded to nearest even, a factor of zero keeping its product, the
//! source and the blend color (`PushField::HalfBlendColor`) clamped first to a normalized format's
//! range as Vulkan's blend clamps them, and stores the result through the device's conversion
//! (`PrecisionNote::DeviceColorConversion`), the channels a write mask keeps taken back from the
//! texel. Metal's blend arithmetic for these formats is not measured: the function records
//! `PrecisionNote::UnmeasuredFetchBlend`. An attachment the function does not read (one a draw
//! moved into the storage form) does not take it: the variant is refused and the render pass
//! splits around the draw, whose blend is the device's. A factor of the second source reads the
//! function's second (index 1) output, four `half` or `float` lanes, clamped as the first (Metal
//! makes and draws a pipeline that fetches and blends so; one whose function writes no second
//! output Metal refuses, and so is it refused, `DescriptorError::ColorAttachment`). The generic
//! blend also takes a 16-bit float, 32-bit float or 8-bit unorm attachment's blend of the second
//! source, or
//! beside a second output, whose Metal arithmetic is not measured.
//!
//! A blocked attachment is refused with the blocker's reason (what the device lacks, or
//! `NotImplemented`): no ordered path exists there. The storage form
//! stores Vulkan's texel output conversion of the output, the rule a color attachment write
//! states for the same formats.

use std::collections::{BTreeMap, BTreeSet};

use crate::air::abi::{AbiInt, OutputRole, ParamRole};
use crate::air::intrinsics::{AirIntrinsic, Fragment, PackDir, PackFormat};
use crate::air::ir::{Callee, InstKind, Type as AirType, TypeId as AirTypeId, Value};
use crate::device::profile::DriverWorkaround;
use crate::device::requirements::{FormatUsage, VkFeature};
use crate::device::DeviceProfile;
use crate::error::{
    Attachment, Capability, DescriptorError, Error, ExtendedRangeBlend, Feature, RuntimeInput,
    Unsupported,
};
use crate::lower::arith::{extended_range, narrow, ops, pack, FloatFmt, Form, Held};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::stage_io::{self, Returned};
use crate::lower::sync::memory_model;
use crate::metal::format::{
    accessed_without_format, color_output_components, rgba_twin, rgba_twin_gaps, sampled_type,
    spirv_image_format, without_format_gaps, ExtendedRange, FormatEmulation, StorageUse,
    WordAttachment,
};
use crate::metal::ColorWriteMask;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::lower_inputs::{BlendEquation, FetchedAttachment, FramebufferFetchInputs};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, SampledType, Stage};
use crate::pipeline::StorageAttachmentUse;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConvOp, Dim, ImageOperands, ImageType, ImageUsage, MemAccess, StorageClass,
    Type as SType, UnOp,
};
use crate::vk;

use super::{not_lowered, ColorInput};

/// Whether `p` orders framebuffer fetch only through the ordered pixel interlock: no
/// rasterization-order color attachment access, with `fragmentShaderPixelInterlock`.
pub(crate) fn interlock_orders(p: &DeviceProfile) -> bool {
    let f = p.fields().fragment;
    !f.rasterization_order_color_access && f.pixel_interlock
}

/// Whether `p` orders framebuffer fetch through rasterization-order input attachments (lavapipe):
/// rasterization-order color attachment access, which [`interlock_orders`] excludes.
pub(crate) fn inputs_order(p: &DeviceProfile) -> bool {
    p.fields().fragment.rasterization_order_color_access
}

/// Whether `pf` is a 16-bit float color format, whose blend of a `half` output of a module
/// without fast math Metal computes in `half` arithmetic, and of one with fast math in the fast
/// forms it folds into the function ([`extended_range::blend::BlendMath`]).
pub(crate) fn half_float_format(pf: crate::metal::PixelFormat) -> bool {
    use crate::metal::PixelFormat as P;
    matches!(pf, P::R16Float | P::RG16Float | P::RGBA16Float)
}

/// Whether `pf` is a 32-bit float color format, whose blend of an output of a module with fast
/// math Metal computes in `float` in the fast forms it folds into the function
/// ([`extended_range::Blend::float32`]); without fast math its arithmetic there is not measured.
pub(crate) fn float32_format(pf: crate::metal::PixelFormat) -> bool {
    use crate::metal::PixelFormat as P;
    matches!(pf, P::R32Float | P::RG32Float | P::RGBA32Float)
}

/// Whether `pf` is an 8-bit unorm color format held directly, whose blend Metal computes in the
/// arithmetic of the output's type over the clamped output ([`extended_range::Blend::unorm8`]).
pub(crate) fn unorm8_format(pf: crate::metal::PixelFormat) -> bool {
    use crate::metal::PixelFormat as P;
    matches!(pf, P::R8Unorm | P::RG8Unorm | P::RGBA8Unorm | P::BGRA8Unorm)
}

/// Whether `pf` is an 8-bit sRGB color format, whose blend Metal computes as its unorm twin's over
/// the decoded texel, encoded ([`extended_range::Blend::srgb8`]).
pub(crate) fn srgb8_format(pf: crate::metal::PixelFormat) -> bool {
    use crate::metal::PixelFormat as P;
    matches!(
        pf,
        P::R8UnormSRGB | P::RG8UnormSRGB | P::RGBA8UnormSRGB | P::BGRA8UnormSRGB
    )
}

/// Whether `pf` is a packed 16-bit unorm format (`metal::format::WordAttachment::Packed16`), whose
/// blend Metal computes in `half` over its decoded codes ([`extended_range::Blend::packed16`]).
pub(crate) fn packed16_format(pf: crate::metal::PixelFormat) -> bool {
    matches!(
        WordAttachment::layout(pf),
        Some(WordAttachment::Packed16 { .. })
    )
}

/// Metal's blend `eq` of a format held directly, which the shader computes in the storage form,
/// storing the channels `written` of four under math `math`: an 8-bit unorm attachment's in
/// `arith` ([`extended_range::Blend::unorm8`]), an 8-bit sRGB one's in `arith`
/// ([`extended_range::Blend::srgb8`], whether Metal's or not), a packed 16-bit one's
/// ([`extended_range::Blend::packed16`], in `half` whatever the output), else a 16-bit float one's
/// ([`extended_range::Blend::half_float`], which takes its arithmetic when it computes). The
/// planning callers pass `BlendArith::Half`: what is refused, what reads the blend color, and the
/// words are the same in either arithmetic.
fn direct_blend(
    eq: &BlendEquation,
    pf: crate::metal::PixelFormat,
    written: ColorWriteMask,
    (math, arith): (
        extended_range::blend::BlendMath,
        extended_range::blend::BlendArith,
    ),
) -> Result<extended_range::Blend, Error> {
    if packed16_format(pf) {
        return extended_range::Blend::packed16(eq, written, math);
    }
    if srgb8_format(pf) {
        return extended_range::Blend::srgb8(eq, pf, written, (math, arith)).map(|(b, _)| b);
    }
    if float32_format(pf) {
        return match math {
            extended_range::blend::BlendMath::Fast => {
                extended_range::Blend::float32(eq, pf, written)
            }
            // Metal's arithmetic for the format without fast math is measured in no state, so
            // the planner leaves it to the generic blend, whose rounded forms these are.
            extended_range::blend::BlendMath::Rounded => generic(eq, pf, written),
        };
    }
    match unorm8_format(pf) {
        true => extended_range::Blend::unorm8(eq, pf, written, (math, arith)),
        false => extended_range::Blend::half_float(eq, pf, written, math),
    }
}

/// Whether Metal's blend `eq` into the 8-bit sRGB attachment of `pf` storing the channels `written`
/// of four under math `math` is its 8-bit unorm blend's, or its rounded forms without fast math
/// ([`extended_range::Blend::srgb8`]'s flag): elsewhere the shader computes a near-equivalent.
fn srgb8_exact(
    eq: &BlendEquation,
    pf: crate::metal::PixelFormat,
    written: ColorWriteMask,
    math: extended_range::blend::BlendMath,
) -> bool {
    let half = extended_range::blend::BlendArith::Half;
    extended_range::Blend::srgb8(eq, pf, written, (math, half)).is_ok_and(|(_, exact)| exact)
}

/// Whether the output a blend into the 8-bit sRGB attachment of `pf` takes is four lanes, its
/// blended output `output` ([`blended_output`]): every output to a four-channel format is
/// ([`select_output`]); another format's where its blended output is, `None` (beside a second
/// output) counting as fewer.
fn srgb_output4(
    pf: crate::metal::PixelFormat,
    output: Option<(extended_range::blend::BlendArith, u32)>,
) -> bool {
    color_output_components(pf) == Some(4) || output.is_some_and(|(_, lanes)| lanes == 4)
}

/// The generic float blend `eq` into an attachment of `pf` storing the channels `written` of four
/// ([`extended_range::Blend::generic`]): a format without alpha reads a destination alpha of one.
fn generic(
    eq: &BlendEquation,
    pf: crate::metal::PixelFormat,
    written: ColorWriteMask,
) -> Result<extended_range::Blend, Error> {
    let has_alpha = channels(pf).contains(ColorWriteMask::ALPHA);
    extended_range::Blend::generic(eq, has_alpha, written)
}

/// The arithmetic of Metal's 16-bit float blend of entry `e`'s write of color attachment `color`,
/// and the output's lanes: that of its first (index 0) output there, where it is one to four lanes
/// of `half` or `float`, and the entry writes no second (index 1) output there; `None` for another
/// output, or none.
pub(crate) fn blended_output(
    e: &crate::air::Entry,
    color: u32,
) -> Option<(extended_range::blend::BlendArith, u32)> {
    let types = e.module.types();
    let func = e.module.func(e.func)?;
    let Some(AirType::Function { ret, .. }) = types.get(func.ty) else {
        return None;
    };
    let field = |i: u32| match types.get(*ret) {
        Some(AirType::Struct { fields, .. }) => fields.get(i as usize).copied(),
        Some(AirType::Void) | None => None,
        Some(_) => Some(*ret),
    };
    let of = |ty: AirTypeId| match types.get(ty) {
        Some(AirType::Half) => Some(extended_range::blend::BlendArith::Half),
        Some(AirType::Float) => Some(extended_range::blend::BlendArith::Float),
        _ => None,
    };
    let arith = |ty: AirTypeId| match types.get(ty) {
        Some(AirType::Vector { elem, len }) if (1..=4).contains(len) => Some((of(*elem)?, *len)),
        Some(_) => Some((of(ty)?, 1)),
        None => None,
    };
    let to = |o: &crate::air::abi::OutputBinding, index: u32| {
        matches!(
            o.role,
            OutputRole::RenderTarget {
                attachment: AbiInt::Const(a),
                index: i,
                ..
            } if a == color && i == index
        )
    };
    let outputs = &e.ep.outputs;
    if outputs.iter().any(|o| to(o, 1)) {
        return None;
    }
    outputs
        .iter()
        .find(|o| to(o, 0))
        .and_then(|o| field(o.field))
        .and_then(arith)
}

/// Whether Metal's compiler may fold the arithmetic of the function's first output to color
/// attachment `color` into the blend it computes there, under blend math `math`: the blend is
/// code Metal's compiler appends to the function before its fast-math passes, so a lane of the
/// output the function computes by a floating-point operation carrying a fast-math flag (an
/// addition, subtraction, multiply, division or remainder, an `llvm.fmuladd`), a constant, a
/// select or phi of one, or the attachment's own color input, is contracted, regrouped or folded
/// with the blend's arithmetic (plain Metal on the M3 Max under macOS 27.0.1: a
/// returned product fused into the blend's one minus the source alpha, `D·k + D·A` regrouped as
/// `D·(k + A)`). A lane of a load, a call, a conversion, a parameter or an operation without
/// fast-math flags is opaque to it: the blend reads the value the function computes. Each lane
/// is followed back through `insertvalue`, `insertelement`, `shufflevector`, `extractelement` and
/// `fneg`; a return whose field cannot be followed so counts as folded. Without fast math
/// (`BlendMath::Rounded`) nothing folds but exactly: a module without fast-math flags allows only
/// exact rewrites.
fn folds_into_blend(
    cx: &SelectCx<'_>,
    color: u32,
    math: extended_range::blend::BlendMath,
) -> Result<bool, Error> {
    use crate::air::ir::BinOp as AirBin;
    if math == extended_range::blend::BlendMath::Rounded {
        return Ok(false);
    }
    let Some(field) = cx.entry.ep.outputs.iter().find_map(|o| match o.role {
        OutputRole::RenderTarget {
            attachment: AbiInt::Const(a),
            index: 0,
            ..
        } if a == color => Some(o.field),
        _ => None,
    }) else {
        return Ok(true);
    };
    // The output's lanes; the blend reads a missing one as zero.
    let lanes = blended_output(cx.entry, color).map_or(4, |(_, n)| n);
    let types = cx.entry.module.types();
    let consts = cx.entry.module.consts();
    let structured = cx
        .entry
        .module
        .func(cx.entry.func)
        .and_then(|f| match types.get(f.ty) {
            Some(AirType::Function { ret, .. }) => types.get(*ret),
            _ => None,
        })
        .is_some_and(|t| matches!(t, AirType::Struct { .. }));
    let color_input = |a: u32| {
        cx.entry.ep.params.iter().any(|b| {
            b.param == a
                && matches!(
                    b.role,
                    ParamRole::ColorInput {
                        index: AbiInt::Const(c),
                        ..
                    } if c == color
                )
        })
    };
    let const_index = |v: Value| match v {
        Value::Const(c) => match consts.get(c).map(|k| &k.value) {
            Some(crate::air::ir::Const::Int(b)) => u32::try_from(*b).ok(),
            _ => None,
        },
        _ => None,
    };
    let lanes_of = |v: Value| {
        cx.value_ty(v).and_then(|t| match types.get(t) {
            Some(AirType::Vector { len, .. }) => Some(*len),
            _ => None,
        })
    };
    let limit = cx.body.insts.len().saturating_add(8);
    // Whether lane `lane` of `v` folds into the blend.
    let folds = |mut v: Value, mut lane: u32| -> bool {
        for _ in 0..limit {
            let at = match v {
                Value::Const(_) => return true,
                Value::Arg(a) => return color_input(a.raw()),
                Value::Inst(at) => at,
            };
            let Some(inst) = cx.body.insts.get(at) else {
                return true;
            };
            match &inst.kind {
                InstKind::BinOp { op, fmf, .. } => {
                    let float = matches!(
                        op,
                        AirBin::FAdd | AirBin::FSub | AirBin::FMul | AirBin::FDiv | AirBin::FRem
                    );
                    return float && !fmf.is_empty();
                }
                InstKind::FNeg { v: x, .. } => v = *x,
                InstKind::InsertElement { vec, elt, idx } => match const_index(*idx) {
                    Some(k) if k == lane => {
                        v = *elt;
                        lane = 0;
                    }
                    Some(_) => v = *vec,
                    None => return true,
                },
                InstKind::ShuffleVector { a, b, mask } => {
                    let Some(Some(pick)) = mask.get(lane as usize).copied() else {
                        return true;
                    };
                    let Some(n) = lanes_of(*a) else { return true };
                    (v, lane) = if pick < n { (*a, pick) } else { (*b, pick - n) };
                }
                InstKind::ExtractElement { vec, idx } => match const_index(*idx) {
                    Some(k) => {
                        v = *vec;
                        lane = k;
                    }
                    None => return true,
                },
                InstKind::Select { t, f, .. } => {
                    return matches!(t, Value::Const(_)) || matches!(f, Value::Const(_));
                }
                InstKind::Phi { incoming, .. } => {
                    return incoming.iter().any(|(x, _)| matches!(x, Value::Const(_)));
                }
                InstKind::Call(call) => {
                    return matches!(
                        call.callee,
                        Callee::Llvm {
                            op: crate::air::intrinsics::LlvmIntrinsic::Float {
                                op: crate::air::intrinsics::LlvmFloatOp::FMulAdd,
                                ..
                            },
                            ..
                        }
                    );
                }
                _ => return false,
            }
        }
        true
    };
    // The field's value at each return.
    let field_of = |mut v: Value| -> Option<Value> {
        if !structured {
            return Some(v);
        }
        for _ in 0..limit {
            let Value::Inst(at) = v else { return None };
            let InstKind::InsertValue { agg, val, indices } = &cx.body.insts.get(at)?.kind else {
                return None;
            };
            if indices.as_ref() == [field] {
                return Some(*val);
            }
            v = *agg;
        }
        None
    };
    for (_, inst) in cx.body.insts.iter() {
        let InstKind::Ret(Some(r)) = &inst.kind else {
            continue;
        };
        let Some(v) = field_of(*r) else {
            return Ok(true);
        };
        if (0..lanes).any(|lane| folds(v, lane)) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// What the function does that the storage form of one color depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ColorUse {
    /// The function writes the attachment through a first (index 0) output.
    pub writes: bool,
    /// The module declares early fragment tests: Metal's function does, or the pipeline has a
    /// depth or stencil attachment and the function's writes are Metal's under them (see the
    /// module docs).
    pub early_tests: bool,
    /// The module declares no early tests only because the function discards, and each draw's
    /// plan checks that its depth and stencil tests drop no fragment
    /// (`RuntimeFact::TestedStorageWrites`): a write then stores before them as Metal's does
    /// after them.
    pub draw_tested: bool,
    /// The function's math, which picks the extended-range blends Metal's model holds
    /// (`extended_range::blend::BlendMath`).
    pub math: extended_range::blend::BlendMath,
}

/// How the storage form's image holds the attachment's samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageSamples {
    /// A pipeline of one sample: a single-sample 2D image, pixel `(x, y)` at texel `(x, y)`.
    One,
    /// The pipeline's samples in a multisampled 2D image, sample `s` of pixel `(x, y)` read and
    /// written at texel `(x, y)` with the `Sample` operand `s`.
    Multisampled,
    /// The pipeline's samples as planes, on a device without multisampled storage images: a
    /// single-sample 2D-array image of one layer per sample, sample `s` of pixel `(x, y)` at
    /// texel `(x, y, s)`.
    Planes,
}

/// How the storage form declares and views one attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageView {
    /// The attachment's Metal pixel format.
    pub pixel_format: crate::metal::PixelFormat,
    /// The attachment's Vulkan format, which the view takes (but an sRGB attachment's,
    /// [`uint_view`](Self::uint_view), and a BGRA-order one's [`rgba_view`](Self::rgba_view)).
    pub format: vk::Format,
    /// The declared SPIR-V image format: `Unknown` for a formatless declaration.
    pub declared: crate::spv::ImageFormat,
    /// The numeric type of the attachment's texels, the declaration's sampled type.
    pub numeric: SampledType,
    /// How the image holds the attachment's samples.
    pub samples: StorageSamples,
    /// The format's channels a write of the function keeps from the texel it overwrites: those
    /// outside the descriptor's write mask, which the storage write takes from the texel it reads
    /// first. Empty where the function does not write the attachment, or the mask covers every
    /// channel of the format; all four (`R`, `G`, `B` and `A`) where it covers none of them, so
    /// that the write stores nothing. Where [`kept_at_draw`](Self::kept_at_draw), the class's: the
    /// channels the descriptor's mask keeps are the draw's.
    pub kept: vk::ColorComponentFlags,
    /// The write keeps some channels and takes which from the draw: the bits of
    /// `PushField::StorageKeptChannels` for the attachment's color (`GraphicsState`'s
    /// `StorageAttachmentUse::kept`), not [`kept`](Self::kept), which names the class.
    pub kept_at_draw: bool,
    /// The attachment's extended-range form, whose image holds Metal's raw texels: the
    /// declaration's texels are the raw format's `uint`s ([`numeric`](Self::numeric)), and a read
    /// decodes Metal's tile value. `None` for a format held directly.
    pub extended: Option<ExtendedRange>,
    /// The blend a write computes in the shader, from the texel it reads first: the pipeline's,
    /// where it enables blending there, of an extended-range attachment
    /// (`extended_range::blend::blend`), Metal's of a 16-bit float, 32-bit float, 8-bit unorm or
    /// packed 16-bit one, or the generic one of another format
    /// ([`generic_blend`](Self::generic_blend)).
    /// `None` for an unblended write or no write.
    pub blend: Option<BlendEquation>,
    /// The shader computes [`blend`](Self::blend) by the generic float blend
    /// ([`extended_range::Blend::generic`], [`generic_blended`]): the blend of a format held
    /// directly whose Metal arithmetic is not built, a near-equivalent the function notes
    /// (`PrecisionNote::UnmeasuredFetchBlend`).
    pub generic_blend: bool,
    /// The unsigned-integer format of the texel size an sRGB attachment is viewed in
    /// (`StorageAttachmentUse::view_alias`): the declaration's texels are its `uint`s, which a
    /// read unpacks and decodes and a write encodes and packs, by the device's conversion.
    /// `None` for another format.
    pub uint_view: Option<vk::Format>,
    /// The RGBA-order twin a BGRA-order attachment is viewed in (`StorageAttachmentUse::
    /// view_alias`, `metal::format::rgba_twin`) on a device that accesses its own format without
    /// one in neither direction: the declaration is the twin's, and every texel read from the
    /// image or written to it has red and blue exchanged ([`metal_order`]), so the function sees
    /// Metal's channels. `None` for another format and device.
    pub rgba_view: Option<vk::Format>,
    /// The word layout of a packed 16-bit attachment the device renders through a view of its
    /// words (`metal::format::WordAttachment`): the view and the declaration are the word's
    /// unsigned integer format, and a write encodes Metal's word ([`word_texel`]). `None` for
    /// another format and device.
    pub word: Option<WordAttachment>,
    /// The attachment's image holds Metal's alpha in red (`metal::format::alpha_in_red`:
    /// `A8Unorm` in `R8_UNORM` on a device without `A8_UNORM`): the view and the declaration are
    /// `R8_UNORM`'s, a texel read from the image, `(r, 0, 0, 1)`, is Metal's `(0, 0, 0, r)`, and a
    /// texel written to it holds Metal's alpha in red ([`metal_order`], [`image_order`]).
    pub alpha_in_red: bool,
}

impl StorageView {
    /// The numeric class of the function's reads and writes: an extended-range attachment's
    /// values are floats, whatever its raw texels.
    pub fn read_class(&self) -> SampledType {
        match (self.extended, self.uint_view, self.word) {
            (Some(_), _, _) | (_, Some(_), _) | (_, _, Some(_)) => SampledType::Float,
            (None, None, None) => self.numeric,
        }
    }

    /// Whether a write stores nothing: the mask covers none of the format's channels.
    pub fn stores_nothing(&self) -> bool {
        self.kept == vk::ColorComponentFlags::known_bits()
    }

    /// Whether a write merges: it keeps some of the format's channels and stores the others.
    pub fn merges(&self) -> bool {
        !self.kept.is_empty() && !self.stores_nothing()
    }

    /// Whether the declaration names no format.
    pub fn formatless(&self) -> bool {
        self.declared == crate::spv::ImageFormat::Unknown
    }
}

/// One color attachment the function reads in the storage form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageColor {
    pub view: StorageView,
    /// The function writes the attachment.
    pub writes: bool,
    /// The returned value's field that holds the write (0 for a returned value itself) and the
    /// output's form, once the stage interface planned it.
    pub output: Option<(u32, Form)>,
    /// The draw's render pass instance holds the extended-range attachment's `half` shadow
    /// (`LoweringInputs::extended_shadows`): a read takes the shadow's texel as the tile value,
    /// and a write stores the value it computes there as well as the codes ([`declare_shadow`]).
    pub shadowed: bool,
    /// The returned value's field that holds the function's second (index-1) output to the
    /// attachment and its form, once the stage interface planned it: the second source of an
    /// extended-range blend the shader computes ([`select_second_output`]).
    pub second: Option<(u32, Form)>,
    /// The 16-bit float blend the shader computes ([`StorageView::blend`] of a format held
    /// directly) is Metal's exactly: its forms are fitted (`extended_range::Blend::fitted`), the
    /// module states its math, and Metal cannot fold the output's arithmetic into it
    /// ([`folds_into_blend`]). Where not, the function notes the near-equivalent. True for
    /// another attachment.
    pub exact_blend: bool,
    /// The pipeline rounds its half-float channels to nearest even
    /// (`TextureWriteRoundingMode::NearestEven`), which a 16-bit float blend of a `float4` output
    /// rounds its result by, as Metal's blended write does; else toward zero.
    pub nearest_even: bool,
}

fn not_implemented() -> Unsupported {
    Unsupported::not_implemented(Feature::FramebufferFetch)
}

/// Whether `format`'s float channels are narrower than 32 bits (a half-float or a packed float
/// format), whose texels a read decodes into `float`.
fn narrow_float(format: vk::Format) -> bool {
    format.block().is_some_and(|b| match b.color {
        Some(vk::NumericFormat::Sfloat) => b.component_bits != Some(32),
        Some(vk::NumericFormat::Ufloat) => true,
        _ => false,
    })
}

/// The channels of `mask` a write of every channel of a color attachment of `pf` must keep: its
/// first `color_output_components(pf)` of red, green, blue, and alpha (all four for a format of
/// none).
fn channels(pf: crate::metal::PixelFormat) -> ColorWriteMask {
    let n = color_output_components(pf).unwrap_or(4);
    let order = [
        ColorWriteMask::RED,
        ColorWriteMask::GREEN,
        ColorWriteMask::BLUE,
        ColorWriteMask::ALPHA,
    ];
    order
        .iter()
        .take(usize::from(n))
        .fold(ColorWriteMask::NONE, |m, &c| m.union(c))
}

/// Whether a storage-form write to an attachment of `pf` that keeps some of its channels takes
/// them from the draw (`PushField::StorageKeptChannels`), so that the write masks of a class
/// compile alike ([`compiled_write_mask`]): every format but an extended-range one, whose kept channels Metal's blend model and
/// encode read (`extended_range::Blend`, `encode_keeping`), and an sRGB one, whose write merges the
/// bytes of its `uint` view.
pub(crate) fn kept_at_draw(pf: crate::metal::PixelFormat) -> bool {
    let srgb = crate::metal::format::direct_format(pf)
        .and_then(|f| f.block())
        .is_some_and(|b| b.color == Some(vk::NumericFormat::Srgb));
    ExtendedRange::of(pf).is_none() && !srgb
}

/// The channels a write of the attachment of `pf` under write mask `mask` keeps from its texel:
/// none where the function does not write it (`writes` false) or the mask covers every channel of
/// the format, all four (`R`, `G`, `B` and `A`) where it covers none of them (the write stores
/// nothing), else those of the format the mask leaves out.
pub(crate) fn kept_channels(
    pf: crate::metal::PixelFormat,
    mask: ColorWriteMask,
    writes: bool,
) -> ColorWriteMask {
    let all = channels(pf);
    if !writes {
        ColorWriteMask::NONE
    } else if all.intersection(mask).is_empty() {
        ColorWriteMask::ALL
    } else {
        all.difference(mask)
    }
}

/// The write mask a fragment function compiles the storage form of an attachment of `pf` under,
/// for the descriptor's `mask` (`FramebufferFetchInputs::of`), blended where `blended`: where the
/// write takes its kept channels from the draw ([`kept_at_draw`]), the class of `mask` alone (none
/// where it covers none of the format's channels, all where it covers every one, and for a mask
/// covering some, every channel of the format but the first, the one mask that stands for them
/// all), so the write masks of a class share one key and one module; any other format's `mask` as
/// it is. A blended 8-bit unorm attachment with alpha, or a blended packed 16-bit one, splits the
/// partial class in three, as Metal's blend forms do (`extended_range::Blend::unorm8`,
/// `extended_range::Blend::packed16`): alpha alone (`ALPHA`), one color channel with or without
/// alpha (`RED` and `ALPHA`), and two or more color channels.
pub(crate) fn compiled_write_mask(
    pf: crate::metal::PixelFormat,
    mask: ColorWriteMask,
    blended: bool,
) -> ColorWriteMask {
    if !kept_at_draw(pf) || mask.contains(ColorWriteMask::UNSPECIALIZED) {
        return mask;
    }
    let all = channels(pf);
    let written = all.intersection(mask);
    let colors = [
        ColorWriteMask::RED,
        ColorWriteMask::GREEN,
        ColorWriteMask::BLUE,
    ]
    .into_iter()
    .filter(|&c| written.contains(c))
    .count();
    // Metal's blend forms differ by the class (`extended_range::Blend::unorm8`, `packed16`).
    let unorm8 = blended
        && ((unorm8_format(pf) && all.contains(ColorWriteMask::ALPHA)) || packed16_format(pf));
    if written.is_empty() {
        ColorWriteMask::NONE
    } else if written == all {
        ColorWriteMask::ALL
    } else if unorm8 && colors == 0 {
        ColorWriteMask::ALPHA
    } else if unorm8 && colors == 1 {
        ColorWriteMask::RED.union(ColorWriteMask::ALPHA)
    } else {
        all.difference(ColorWriteMask::RED)
    }
}

/// The storage form of the read of color attachment `color` of a pipeline whose state is
/// `inputs` (`multisampled`: it rasterizes more than one sample, so the form is per sample) by a
/// function whose use of it is `use_`, on `p` (see the module docs): `Ok(Ok(view))` where the
/// form is exact, `Ok(Err(why))`
/// where the pipeline's state, the function, or the device blocks it, and `Err` for a
/// descriptor error that refuses the pipeline whatever the form (no attachment at `color`, a
/// pixel format no color attachment can have).
pub(crate) fn storage_form(
    color: u32,
    inputs: &FramebufferFetchInputs,
    multisampled: bool,
    use_: ColorUse,
    p: &DeviceProfile,
) -> Result<Result<StorageView, Unsupported>, Error> {
    let attachment: FetchedAttachment =
        inputs
            .attachments
            .get(&color)
            .copied()
            .flatten()
            .ok_or(Error::InvalidDescriptor(DescriptorError::ColorAttachment {
                index: color,
            }))?;
    let pf = attachment.pixel_format;
    // A packed 16-bit attachment the device samples but does not render renders through a view of
    // its words, which the function stores as Metal's word and merges by its fields.
    let word = WordAttachment::of(pf, p);
    let usages = BTreeSet::from([FormatUsage::ColorAttachment]);
    let (format, emulation) = match word {
        Some(word) => (word.word_format(), FormatEmulation::Direct),
        None => {
            let mapping = crate::metal::format::vk_format(pf, &usages, p)?;
            (mapping.format, mapping.emulation)
        }
    };
    // An extended-range attachment holds Metal's raw texels in its raw format, which a read
    // decodes.
    let extended = ExtendedRange::of(pf);
    let srgb = format
        .block()
        .is_some_and(|b| b.color == Some(vk::NumericFormat::Srgb));
    // An 8-bit sRGB attachment is viewed in the unsigned integer format of its texel (`R8_UINT`,
    // `R16_UINT` or `R32_UINT`), whose bytes the function decodes and encodes by the device's
    // conversion (`arith::pack`'s, a channel the format lacks 0 and alpha 1, a write's encode of
    // Metal's blend where it blends).
    // A BGRA-order attachment on a device that accesses its format without one in neither
    // direction (SPIR-V names no image format for it) is viewed in its RGBA-order twin, whose red
    // and blue the function exchanges on every read and write: the device converts each channel
    // as it converts the attachment's, so the values are those of the attachment's own view.
    let alias = StorageAttachmentUse::view_alias_of(format, p)
        .filter(|_| emulation == FormatEmulation::Direct && word.is_none());
    let uint_view = alias.filter(|&u| {
        srgb && matches!(
            u,
            vk::Format::R8_UINT | vk::Format::R16_UINT | vk::Format::R32_UINT
        )
    });
    let rgba_view = alias.filter(|_| !srgb);
    let viewed = uint_view.or(rgba_view).unwrap_or(format);
    let writes = use_.writes;
    // The channels a write keeps from the texel: those of the format the descriptor's mask leaves
    // out, or all four where it leaves out every one (the write stores nothing). An `Unspecialized`
    // mask is a placeholder, which no form serves.
    let all = channels(pf);
    let kept = kept_channels(pf, attachment.write_mask, writes);
    let merged = !kept.is_empty() && kept != ColorWriteMask::ALL;
    // A write that keeps some channels of a format whose kept channels are the draw's: the
    // write mask is its class's (`compiled_write_mask`), and the draw's word names them.
    let kept_draw = merged && kept_at_draw(pf);
    let snorm = format
        .block()
        .is_some_and(|b| b.color == Some(vk::NumericFormat::Snorm));
    // A returned sample mask keeps the invocation's own sample when it sets its bit: the per-sample
    // form gates its write on that bit, and a write at one sample on bit 0. A kept channel of a
    // signed normalized format beside a written one is not modelled either: its two codes of -1
    // read alike, so writing back what was read would not keep the texel's bits. A write that
    // stores some of an extended-range attachment's channels is built unblended where it stores
    // every one, and under a blend Metal's model covers, which the shader computes: no device
    // blends the raw texels. A write whose mask covers none of the format's channels stores
    // nothing, so nothing a store depends on (the blend, coverage, the sample mask, the depth and
    // stencil tests) bears on it; and a blend of one and zero, added, for color and alpha stores
    // the source as blending disabled does, a signed normalized NaN aside, and a 16-bit float
    // attachment's blend without fast math, which keeps the destination's product by zero (a NaN
    // of an infinite or NaN texel, the sum's zero sign).
    let stores = writes && kept != ColorWriteMask::ALL;
    // The blend of an extended-range attachment the write stores, which the shader computes
    // where Metal's measured model holds it (`extended_range::Blend`), its gap refused by name.
    // And the blend of a 16-bit float attachment, which no Vulkan blend applies to a storage
    // write: the shader computes Metal's, in the arithmetic of a four-lane `half` or `float`
    // output, which [`plan`] checks (`extended_range::Blend::half_float`: every factor and
    // operation but the second source's, which the generic blend below takes). A channel the
    // format lacks counts as written, as [`plan`] plans it.
    let kept_zero = use_.math == extended_range::blend::BlendMath::Rounded;
    let written_of_four = ColorWriteMask::ALL.difference(kept);
    let half_blend = half_float_format(pf)
        && attachment.blending_enabled
        && (!attachment.replaces || kept_zero)
        && attachment.equation.is_some_and(|eq| {
            extended_range::Blend::half_float(&eq, pf, written_of_four, use_.math).is_ok()
        });
    // And the blend of a 32-bit float attachment of a module with fast math (or naming no math
    // mode) at a state whose forms are Metal's (`extended_range::Blend::float32`, whose
    // `Blend::fitted` is the one sweep that measured the family). Elsewhere -- a state the sweep
    // does not fit, a module without fast math, an output of fewer than four lanes (below) -- the
    // generic blend keeps the attachment: both are near-equivalents there, and the generic one is
    // measured the closer at a one-channel attachment's `SourceColor` over `DestinationColor`
    // (`half-output-blends-r32f`'s `sc-dc`, where Metal rounds the destination's product first and
    // the generic blend's mirror forms are Metal's at every word). A 16-bit float attachment's
    // blend takes the in-shader form at an unfitted state instead, its forms having been swept
    // over hundreds of states rather than one.
    let float_blend = float32_format(pf)
        && attachment.blending_enabled
        && !attachment.replaces
        && use_.math == extended_range::blend::BlendMath::Fast
        && attachment.equation.is_some_and(|eq| {
            extended_range::Blend::float32(&eq, pf, written_of_four).is_ok_and(|b| b.fitted())
        });
    // And the blend of an 8-bit unorm attachment where Metal's measured model holds it
    // (`extended_range::Blend::unorm8`: every factor and operation but the second source's, any
    // write mask); its other blends take the generic blend below.
    let half = extended_range::blend::BlendArith::Half;
    let unorm8_blend = unorm8_format(pf)
        && attachment.blending_enabled
        && !attachment.replaces
        && attachment.equation.is_some_and(|eq| {
            extended_range::Blend::unorm8(&eq, pf, written_of_four, (use_.math, half)).is_ok()
        });
    // And the blend of a packed 16-bit attachment rendered through its word view, which no
    // device blends (`extended_range::Blend::packed16`: every factor and operation but the second
    // source's, any write mask).
    let word_blend = packed16_format(pf)
        && word.is_some()
        && attachment.blending_enabled
        && !attachment.replaces
        && attachment.equation.is_some_and(|eq| {
            extended_range::Blend::packed16(&eq, written_of_four, use_.math).is_ok()
        });
    // And the blend of an 8-bit sRGB attachment viewed in its unsigned integer format, which no
    // device blends: the shader computes Metal's, the 8-bit unorm blend over the decoded texel,
    // encoded (`extended_range::Blend::srgb8`: its rounded forms where the 8-bit unorm blend is
    // not built, a near-equivalent [`plan`] notes under fast math).
    let srgb_blend = uint_view.is_some()
        && attachment.blending_enabled
        && !attachment.replaces
        && attachment.equation.is_some_and(|eq| {
            extended_range::Blend::srgb8(&eq, pf, written_of_four, (use_.math, half)).is_ok()
        });
    // And any other blend of a float-class format held directly (not an sRGB or a word one),
    // whose Metal arithmetic is not built: the shader computes the generic float blend
    // (`extended_range::Blend::generic`), a near-equivalent [`plan`] takes only where the function
    // reads the attachment, and notes.
    let float_class = sampled_type(pf) == Some(SampledType::Float);
    let generic_blend = stores
        && attachment.blending_enabled
        && !(attachment.replaces && !snorm)
        && float_class
        && !srgb
        && extended.is_none()
        && word.is_none()
        && !(half_blend || unorm8_blend || float_blend);
    let blend = attachment.equation.filter(|_| {
        (extended.is_some_and(|e| !e.padded())
            || half_blend
            || float_blend
            || unorm8_blend
            || word_blend
            || srgb_blend
            || generic_blend)
            && stores
    });
    if let Some(eq) = blend {
        let planned = match extended {
            Some(form) => {
                let written = all.intersection(attachment.write_mask);
                extended_range::Blend::of(&eq, form, written, use_.math)
            }
            None if generic_blend => generic(&eq, pf, written_of_four),
            None => direct_blend(&eq, pf, written_of_four, (use_.math, half)),
        };
        match planned {
            Ok(_) => {}
            Err(Error::Unsupported(gap)) => return Ok(Err(gap)),
            Err(e) => return Err(e),
        }
    }
    let shader_blends = blend.is_some();
    let blends = attachment.blending_enabled && !(attachment.replaces && !snorm) && !shader_blends;
    // An `A8Unorm` attachment whose image holds Metal's alpha in red, for every use of the
    // texture (`metal::format::alpha_in_red`), is read and written through its `R8_UNORM` image,
    // the function moving Metal's alpha into and out of red.
    let alpha_in_red = emulation == FormatEmulation::AlphaInRed
        && crate::metal::format::alpha_in_red(pf, p)
        && word.is_none();
    // An unblended write of every channel stores Metal's write of the output. A padded 10-bit
    // attachment's tile value and blends are not measured: no storage form.
    let not_modelled = inputs.amplified
        || extended.is_some_and(ExtendedRange::padded)
        || (emulation != FormatEmulation::Direct && extended.is_none() && !alpha_in_red)
        || (extended.is_some()
            && stores
            && !shader_blends
            && (attachment.blending_enabled || !kept.is_empty()))
        || (srgb && uint_view.is_none())
        // An `RGB9E5Float` word's mantissas share its exponent: no field is kept apart.
        || (word == Some(WordAttachment::Rgb9e5) && stores && merged)
        || (writes
            && attachment
                .write_mask
                .contains(ColorWriteMask::UNSPECIALIZED))
        || (stores
            && (blends
                || (snorm && merged)
                || inputs.alpha_to_coverage
                || (inputs.depth_or_stencil && !use_.early_tests && !use_.draw_tested)));
    if not_modelled {
        return Ok(Err(not_implemented()));
    }
    let kept = crate::metal::state::color_write_mask(kept)?;
    let f = p.fields();
    // The per-sample form's writes are storage image writes of the fragment shader; where it
    // declares early tests, a driver that drops such writes in a pass of more than one sample
    // would lose every one.
    let defect = DriverWorkaround::EarlyTestsDropMultisampledImageWrites;
    if multisampled && writes && use_.early_tests && f.workarounds.contains(&defect) {
        return Ok(Err(Unsupported::driver_defect(
            Feature::FramebufferFetch,
            defect,
        )));
    }
    // What the form needs whatever view it takes, and what the view's own route lacks.
    let mut common = Vec::new();
    let mut route = Vec::new();
    // The per-sample form: the image read and written at the invocation's `SampleId`, which needs
    // sample-rate shading. It is multisampled where the device has multisampled storage images (its
    // sample count is the pipeline's, which link checks against `storageImageSampleCounts`), and
    // planes elsewhere.
    let samples = match (multisampled, f.images.storage_multisample) {
        (false, _) => StorageSamples::One,
        (true, true) => StorageSamples::Multisampled,
        (true, false) => StorageSamples::Planes,
    };
    if multisampled && !f.fragment.sample_rate_shading {
        common.push(Capability::Feature(VkFeature::SampleRateShading));
    }
    if writes && !f.leaf.fragment_stores_and_atomics {
        common.push(Capability::Feature(VkFeature::FragmentStoresAndAtomics));
    }
    if !p.supports_format(viewed, FormatUsage::StorageImage) {
        route.push(Capability::Format {
            format: viewed,
            usage: FormatUsage::StorageImage,
        });
    }
    // Declared without a format where the device's features grant the format that access without
    // one, so that the declaration, and with it the module, is the same for every format the
    // function reads alike (a `BGRA8Unorm` attachment's and an `RGBA16Float` one's); else with its
    // SPIR-V image format where the device can (no format feature beyond storage); else without a
    // format where the format's own bits allow it (`without_format_gaps`: Vulkan 1.3 or
    // `VK_KHR_format_feature_flags2`).
    let use_without = StorageUse::texture(true, writes);
    let typed = spirv_image_format(viewed).filter(|d| {
        !crate::spirv::requirements::extended_format(*d) || f.images.storage_extended_formats
    });
    let declared = match typed {
        _ if accessed_without_format(viewed, use_without, p) => {
            Some(crate::spv::ImageFormat::Unknown)
        }
        Some(d) => Some(d),
        None => {
            let gaps = without_format_gaps(viewed, use_without, p);
            if gaps.is_empty() {
                Some(crate::spv::ImageFormat::Unknown)
            } else {
                match spirv_image_format(viewed) {
                    Some(_) => route.push(Capability::Feature(
                        VkFeature::ShaderStorageImageExtendedFormats,
                    )),
                    None => route.extend(gaps),
                }
                None
            }
        }
    };
    let numeric = match (extended, uint_view, word) {
        (Some(_), _, _) | (_, Some(_), _) | (_, _, Some(_)) => SampledType::Uint,
        (None, None, None) => sampled_type(pf).ok_or_else(not_implemented)?,
    };
    match declared {
        Some(declared) if common.is_empty() && route.is_empty() => Ok(Ok(StorageView {
            pixel_format: pf,
            format,
            declared,
            numeric,
            samples,
            kept,
            kept_at_draw: kept_draw,
            extended,
            blend,
            generic_blend: generic_blend && blend.is_some(),
            uint_view,
            rgba_view,
            word,
            alpha_in_red,
        })),
        _ => {
            // A BGRA-order format no declaration reaches, whose RGBA-order twin lacks something
            // too: either route's gaps.
            let mut alternatives = vec![[common.clone(), route].concat()];
            let twin_gaps = rgba_twin(viewed)
                .filter(|_| declared.is_none())
                .map(|twin| rgba_twin_gaps(twin, p))
                .filter(|gaps| !gaps.is_empty());
            if let Some(twin_gaps) = twin_gaps {
                alternatives.push([common, twin_gaps].concat());
            }
            Ok(Err(Unsupported::needs_any_of(
                Feature::FramebufferFetch,
                alternatives,
            )))
        }
    }
}

/// Whether a render pipeline on `p` blends into a color attachment of `pf`, one of Apple's
/// extended-range formats (`metal::format::extended_range_blend`, its public door): the decision
/// [`plan`] and [`storage_form`] make for such an attachment, written whole by the function's
/// first output under a blend Metal's model holds (premultiplied source-over, one), in a pipeline
/// of one sample with no depth or stencil attachment. `Ok` where the device orders fetch through
/// the pixel interlock alone ([`interlock_orders`]) and that storage form is exact there; `Ok`
/// where it orders fetch through rasterization-order input attachments and the function's blend
/// over the attachment's input attachment holds (`input_blend`: a form Metal's model blends, not a
/// padded or bfloat one, which stays `NotImplemented(Format)` as `stage_io::extended_color` refuses
/// it); else what blocks it, without either ordering the interlock it needs.
pub(crate) fn extended_range_blend(
    pf: crate::metal::PixelFormat,
    p: &DeviceProfile,
) -> Result<(), Unsupported> {
    use crate::metal::{BlendFactor as F, BlendOperation as O};
    let Some(form) = ExtendedRange::of(pf) else {
        return Err(Unsupported::not_implemented(Feature::Format));
    };
    let equation = BlendEquation {
        source_rgb: F::One,
        destination_rgb: F::OneMinusSourceAlpha,
        rgb_operation: O::Add,
        source_alpha: F::One,
        destination_alpha: F::OneMinusSourceAlpha,
        alpha_operation: O::Add,
    };
    if !interlock_orders(p) {
        let f = p.fields().fragment;
        if !f.rasterization_order_color_access {
            return Err(Unsupported::needs(
                Feature::FramebufferFetch,
                Capability::Feature(VkFeature::FragmentShaderPixelInterlock),
            ));
        }
        // The blend over the input attachment: the attachment's raw format a color attachment,
        // and the blend Metal's model's.
        if form.padded() || form.bfloat() {
            return Err(Unsupported::not_implemented(Feature::Format));
        }
        let usages = BTreeSet::from([FormatUsage::ColorAttachment]);
        match crate::metal::format::vk_format(pf, &usages, p) {
            Ok(_) => {}
            Err(Error::Unsupported(gap)) => return Err(gap),
            Err(_) => return Err(Unsupported::not_representable(Feature::Format)),
        }
        let math = extended_range::blend::BlendMath::Fast;
        return match extended_range::Blend::of(&equation, form, channels(pf), math) {
            Ok(_) => Ok(()),
            Err(Error::Unsupported(gap)) => Err(gap),
            Err(_) => Err(Unsupported::not_representable(Feature::Format)),
        };
    }
    let attachment = FetchedAttachment {
        pixel_format: pf,
        blending_enabled: true,
        write_mask: ColorWriteMask::ALL,
        replaces: false,
        equation: Some(equation),
    };
    let inputs = FramebufferFetchInputs {
        attachments: BTreeMap::from([(0, Some(attachment))]),
        depth_or_stencil: false,
        alpha_to_coverage: false,
        amplified: false,
    };
    let use_ = ColorUse {
        writes: true,
        early_tests: false,
        draw_tested: false,
        math: extended_range::blend::BlendMath::Fast,
    };
    match storage_form(0, &inputs, false, use_, p) {
        Ok(form) => form.map(drop),
        Err(_) => Err(Unsupported::not_representable(Feature::Format)),
    }
}

/// What the function's body holds that the storage form depends on, from one walk.
#[derive(Default)]
pub(super) struct BodyFacts {
    /// The LLVM parameters the body reads.
    pub(super) used: BTreeSet<u32>,
    /// The body discards the fragment.
    pub(super) discards: bool,
    /// The body writes memory (`discard::side_effect`).
    pub(super) side_effects: bool,
}

/// How the depth and stencil tests bear on the function's storage writes, which it makes before
/// them where the module declares no early tests (module docs).
#[derive(Clone, Copy)]
struct StoredBeforeTests {
    /// Only its discard keeps the function from early tests: its stores are Metal's where a
    /// draw's tests drop no fragment, and under early tests where they write and count nothing.
    tested: bool,
    /// It writes other memory, or returns depth, stencil, or a sample mask, or early tests would
    /// lose its writes (`dropped`): its stores are Metal's only where a draw's tests drop no
    /// fragment.
    untested: bool,
}

/// [`StoredBeforeTests`] of a function whose body holds `body`, that Metal runs under early tests
/// where `declared_early` and returns depth, stencil, or a sample mask where `exports`, in a
/// pipeline with a depth or stencil attachment where `depth_or_stencil`, where early tests would
/// drop its storage writes where `dropped`.
fn stored_before_tests(
    body: &BodyFacts,
    (declared_early, exports): (bool, bool),
    (depth_or_stencil, dropped): (bool, bool),
) -> StoredBeforeTests {
    let testable = depth_or_stencil && !(exports || body.side_effects);
    StoredBeforeTests {
        tested: !declared_early && testable && body.discards,
        untested: !declared_early && depth_or_stencil && (exports || body.side_effects || dropped),
    }
}

pub(super) fn body_facts(cx: &SelectCx<'_>) -> BodyFacts {
    let mut out = BodyFacts::default();
    for (id, inst) in cx.body.insts.iter() {
        inst.kind.for_each_value(|v| {
            if let Value::Arg(a) = v {
                out.used.insert(a.raw());
            }
        });
        if let InstKind::Call(call) = &inst.kind {
            if matches!(
                call.callee,
                Callee::Air {
                    op: AirIntrinsic::Fragment(Fragment::Discard),
                    ..
                }
            ) {
                out.discards = true;
            }
        }
        if super::discard::side_effect(cx, id, inst) {
            out.side_effects = true;
        }
    }
    out
}

/// The color attachments the function reads: those of its used `[[color(n)]]` parameters of a
/// constant index (another index is refused when the parameter is planned). Each such
/// parameter's type is checked here, so a function no pipeline state would make lowerable is
/// refused before it needs one.
fn colors_read(cx: &SelectCx<'_>, used: &BTreeSet<u32>) -> Result<BTreeSet<u32>, Error> {
    let mut out = BTreeSet::new();
    for b in cx
        .entry
        .ep
        .params
        .iter()
        .filter(|b| used.contains(&b.param))
    {
        if let ParamRole::ColorInput {
            index: AbiInt::Const(color),
            ..
        } = b.role
        {
            super::color::param_form(cx, b)
                .map_err(|e| e.at_attachment(Attachment::Color(color)))?;
            out.insert(color);
        }
    }
    Ok(out)
}

/// The extended-range attachments the function writes through a first output where the
/// pipeline blends them (`LoweringInputs::extended_range_colors`), which only the storage form
/// writes, and the AIR type of each output.
pub(super) fn blended_extended_range(cx: &SelectCx<'_>) -> Result<Vec<(u32, AirTypeId)>, Error> {
    let mut out = Vec::new();
    for (o, ty) in cx
        .entry
        .ep
        .outputs
        .iter()
        .zip(stage_io::output_types(cx)?.1)
    {
        if let OutputRole::RenderTarget {
            attachment: AbiInt::Const(color),
            index: 0,
            ..
        } = o.role
        {
            if cx
                .inputs
                .extended_range_colors
                .get(&color)
                .is_some_and(|e| e.blended)
            {
                out.push((color, ty));
            }
        }
    }
    Ok(out)
}

/// Decide the form of every color attachment the function reads, of every extended-range
/// attachment it writes under blending, and of every other attachment the pipeline's state
/// names that it writes (a color its draw's instance keeps out of its subpass), before the stage
/// interface is planned (its outputs to a storage-form attachment are not interface variables): on
/// a device that orders fetch only through the interlock ([`interlock_orders`]), the storage form
/// where it is exact; elsewhere the input attachment (`color`), or the output an extended-range
/// attachment's blend refuses. A blocked attachment refuses the function.
pub(super) fn plan(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if cx.stage != Stage::Fragment || !interlock_orders(cx.profile) {
        return Ok(());
    }
    let body = body_facts(cx);
    let outputs = &cx.entry.ep.outputs;
    let returns = |role: fn(&OutputRole) -> bool| outputs.iter().any(|o| role(&o.role));
    let exports = returns(|r| {
        matches!(
            r,
            OutputRole::Depth(_) | OutputRole::Stencil | OutputRole::SampleMask
        )
    });
    let declared_early = matches!(
        cx.entry.ep.stage,
        crate::air::abi::Stage::Fragment {
            early_fragment_tests: true
        }
    );
    // A draw may move a color the function writes through its first output into the storage
    // form: a color its render pass instance keeps out of its subpass for other draws'
    // interlocked form (`VariantKey::interlocked_colors`), or an extended-range color whose `half`
    // shadow the instance holds (`VariantKey::extended_shadows`), both single-sampled, so no
    // driver drops their writes under early tests. The plan of the pipeline as compiled, which
    // stores no color so, selects the tests such a draw's variant needs from this.
    let movable = stored_before_tests(&body, (declared_early, exports), (true, false));
    if (movable.tested || movable.untested)
        && returns(|r| matches!(r, OutputRole::RenderTarget { index: 0, .. }))
    {
        cx.facts.insert(RuntimeFact::MovableStorageWrites {
            early_variant: movable.tested && !movable.untested,
        });
    }
    let blended = blended_extended_range(cx)?;
    // A bfloat attachment's blend has no model (Metal's validation refuses one): no storage form
    // blends it.
    let bfloat = blended.iter().find(|&&(color, _)| {
        cx.inputs
            .extended_range_colors
            .get(&color)
            .is_some_and(|e| e.form.bfloat())
    });
    if let Some(&(color, _)) = bfloat {
        let refused = Unsupported::not_implemented(Feature::Format);
        return Err(refused.at_attachment(Attachment::Color(color)).into());
    }
    let fetched = colors_read(cx, &body.used)?;
    let mut read = fetched.clone();
    read.extend(blended.iter().map(|&(color, _)| color));
    // Every other attachment the pipeline's state names that the function writes through its
    // first output: a color the draw moved into the storage form (above), written through it as a
    // fetched one is, or one named for Metal's 16-bit float blend alone.
    if let Some(inputs) = &cx.inputs.framebuffer_fetch {
        read.extend(outputs.iter().filter_map(|o| match o.role {
            OutputRole::RenderTarget {
                attachment: AbiInt::Const(color),
                index: 0,
                ..
            } if inputs.attachments.contains_key(&color) => Some(color),
            _ => None,
        }));
    }
    if read.is_empty() {
        return Ok(());
    }
    // Metal's blend into an extended-range attachment is modelled for a `half4` output, and for a
    // `float4` one under a blend that takes it rounded to `half` (`extended_range::Blend::takes`);
    // another `float` one is rounded otherwise, unfitted.
    let mut outputs4 = BTreeMap::new();
    for (color, ty) in blended {
        let f =
            super::color::form(cx, ty).map_err(|e| e.at_attachment(Attachment::Color(color)))?;
        if f.lanes == 4 {
            if let Some(fmt) = f.float_fmt() {
                outputs4.insert(color, fmt);
            }
        }
    }
    let Some(inputs) = cx.inputs.framebuffer_fetch.clone() else {
        return Err(Error::NeedsInput(RuntimeInput::FramebufferFetch));
    };
    // The module declares early fragment tests so that a depth or stencil test the fragment
    // fails drops its storage writes, as it drops Metal's color write; it may only where they
    // keep the function's writes Metal's (no discard, no depth, stencil, or coverage export, no
    // other memory write). Without a depth or stencil attachment no test can fail, so the
    // declaration would change nothing, and it is left out: Apple's GPU drops the image writes
    // of an early-tests fragment shader in a pass of more than one sample.
    let testable = inputs.depth_or_stencil && !(exports || body.side_effects);
    // A driver that drops the image writes of an early-tests fragment shader in a pass of more
    // than one sample would lose every storage write of a function that writes a fetched color
    // beside a depth or stencil attachment under early tests: such a function declares none (unless
    // Metal's does), and stores before the tests, Metal's where its draws' tests drop no fragment;
    // it has no early-tests variant. (its form checks the driver's defect itself, and the device
    // blends it where that blocks it)
    let writes_read = read.iter().any(|&color| {
        outputs.iter().any(|o| {
            matches!(
                o.role,
                OutputRole::RenderTarget {
                    attachment: AbiInt::Const(a),
                    index: 0,
                    ..
                } if a == color
            )
        })
    });
    let dropped = cx.inputs.multisampled
        && inputs.depth_or_stencil
        && writes_read
        && cx
            .profile
            .fields()
            .workarounds
            .contains(&DriverWorkaround::EarlyTestsDropMultisampledImageWrites);
    // A function that only its discard keeps from early tests, on a device that requires ordered
    // fetch (`tested`): its storage writes are Metal's where its draws' tests drop no fragment,
    // before the tests, and where they write neither attachment and count no visibility result,
    // under early tests (`LoweringInputs::read_only_tests`), which then drop no fragment Metal's
    // later tests keep and leave the attachments as Metal's do; the plan of each draw checks
    // which (`RuntimeFact::TestedStorageWrites`). One that writes other memory, or returns depth,
    // stencil, or a sample mask, beside a depth or stencil attachment, or whose driver would drop
    // its writes under early tests (`untested`): Metal runs its memory writes for every fragment,
    // those its tests then drop included, and tests what it returns (measured on Apple's M4), which
    // early tests would not, so its storage writes before the tests are Metal's where its draws'
    // tests drop no fragment, which the plan of each draw checks; it has no early-tests variant.
    let StoredBeforeTests { tested, untested } = stored_before_tests(
        &body,
        (declared_early, exports),
        (inputs.depth_or_stencil, dropped),
    );
    let early_tests = declared_early
        || (testable && !body.discards && !untested)
        || (tested && cx.inputs.read_only_tests && !untested);
    for color in read {
        // Compiled under the state of a pipeline for another function.
        if !inputs.attachments.contains_key(&color) {
            return Err(Error::NeedsInput(RuntimeInput::FramebufferFetch));
        }
        let writes = outputs.iter().any(|o| {
            matches!(
                o.role,
                OutputRole::RenderTarget {
                    attachment: AbiInt::Const(a),
                    index: 0,
                    ..
                } if a == color
            )
        });
        let use_ = ColorUse {
            writes,
            early_tests,
            draw_tested: (tested || untested) && !early_tests,
            math: blend_math(cx),
        };
        let tied = |e: Error| e.at_attachment(Attachment::Color(color));
        let form =
            storage_form(color, &inputs, cx.inputs.multisampled, use_, cx.profile).map_err(tied)?;
        // Metal's blend of a 16-bit float attachment, computed by the shader: exact where its
        // forms are fitted and Metal cannot fold the output's arithmetic into it
        // ([`folds_into_blend`]), else a near-equivalent the function notes. An attachment the
        // function does not read (one a draw moved into the storage form) takes it only where it
        // is exact: elsewhere the device blends it, as the render pass splits around the draw.
        let mut sixteen_exact = true;
        // The generic float blend of a format whose Metal arithmetic is not built, a
        // near-equivalent the function notes: taken only where the function reads the
        // attachment. Elsewhere the device blends it, as the render pass splits around the draw.
        // A blend of the second source reads the function's second (index 1) output, four `half`
        // or `float` lanes as the first's.
        let second = second_output4(cx, color)?;
        // A blend of the second source whose function writes no second output: Metal refuses
        // the pipeline ("Fragment shader does not write to render target color(0), index(1) that
        // is required for blending"), as link refuses it where the device blends.
        let reads_second = inputs
            .attachments
            .get(&color)
            .copied()
            .flatten()
            .is_some_and(|a| {
                a.blending_enabled && a.equation.is_some_and(|eq| eq.reads_second_source())
            });
        if reads_second && second.is_none() {
            let invalid = DescriptorError::ColorAttachment { index: color };
            return Err(Error::InvalidDescriptor(invalid));
        }
        let generic_of = |view: StorageView, exact: &mut bool| {
            let reads_second = view
                .blend
                .and_then(|eq| generic(&eq, view.pixel_format, written(view.kept)).ok())
                .is_some_and(|b| b.reads_second());
            match (fetched.contains(&color), reads_second, second) {
                (false, _, _) => Err(not_implemented()),
                (true, true, Some(None)) => Err(Unsupported::not_implemented(
                    Feature::ExtendedRangeBlend(ExtendedRangeBlend::Output),
                )),
                (true, _, _) => {
                    *exact = false;
                    Ok((
                        StorageView {
                            generic_blend: true,
                            ..view
                        },
                        None,
                    ))
                }
            }
        };
        let form = match form {
            Ok(view) if view.generic_blend => generic_of(view, &mut sixteen_exact),
            // Metal's 16-bit float blend, of a `half` or `float` output of one to four lanes, or
            // its 8-bit unorm blend, of four.
            Ok(view) if view.blend.is_some() && view.extended.is_none() => {
                let math = blend_math(cx);
                let planned = view.blend.map(|eq| {
                    let written = written(view.kept);
                    let half = extended_range::blend::BlendArith::Half;
                    direct_blend(&eq, view.pixel_format, written, (math, half))
                });
                let unorm8 = unorm8_format(view.pixel_format);
                let srgb = view.uint_view.is_some();
                match (planned, blended_output(cx.entry, color)) {
                    // An 8-bit sRGB blend (`extended_range::Blend::srgb8`), whose second source,
                    // where it reads one, is the function's second output: Metal's where its 8-bit
                    // unorm blend is built, the output is four lanes and the module states its
                    // math (Metal clamps the output first, so none of the output's arithmetic
                    // folds into it), else a near-equivalent the function notes (an output of
                    // fewer lanes blends as four whose missing lanes are zero, alpha among them),
                    // taken only where the function reads the attachment.
                    (Some(Ok(b)), output) if srgb => {
                        let stated = cx.entry.compile.fast_math.is_some();
                        let four = srgb_output4(view.pixel_format, output);
                        let exact = four
                            && view.blend.is_some_and(|eq| {
                                srgb8_exact(&eq, view.pixel_format, written(view.kept), math)
                            });
                        sixteen_exact = stated && exact;
                        match (b.reads_second(), second) {
                            (true, Some(None)) => Err(Unsupported::not_implemented(
                                Feature::ExtendedRangeBlend(ExtendedRangeBlend::Output),
                            )),
                            _ if sixteen_exact || fetched.contains(&color) => Ok((view, None)),
                            _ => Err(not_implemented()),
                        }
                    }
                    // An 8-bit unorm blend of fewer lanes is not measured: the generic blend.
                    (Some(Ok(_)), Some((_, lanes))) if unorm8 && lanes < 4 => {
                        generic_of(view, &mut sixteen_exact)
                    }
                    // A packed 16-bit one is not built.
                    (Some(Ok(_)), Some((_, lanes))) if view.word.is_some() && lanes < 4 => {
                        Err(not_implemented())
                    }
                    // A packed 16-bit blend reads the attachment's codes, where Metal's tile holds
                    // a clear's or an earlier blend's value finer, and under fast math sums some
                    // states otherwise (`extended_range::Blend::packed16`): a near-equivalent the
                    // function notes, taken whether or not the function reads the attachment, as
                    // no device blends the format.
                    (Some(Ok(_)), Some(_)) if view.word.is_some() => {
                        sixteen_exact = false;
                        Ok((view, None))
                    }
                    // A 32-bit float attachment's blend of an output of fewer than four lanes is
                    // not measured (the recordings' output is a `half4`): the generic blend.
                    (Some(Ok(_)), Some((_, lanes)))
                        if float32_format(view.pixel_format) && lanes < 4 =>
                    {
                        generic_of(view, &mut sixteen_exact)
                    }
                    (Some(Ok(b)), Some((_, lanes))) => {
                        let stated = cx.entry.compile.fast_math.is_some();
                        // An output of fewer lanes whose color blend reads the source alpha stores
                        // Metal's constant ([`half_blended`]), whatever the output.
                        let constant = lanes < 4 && b.color_reads_source_alpha();
                        // Metal clamps the output before an 8-bit unorm blend, so none of the
                        // output's arithmetic folds into it: the blend is
                        // Metal's wherever the module states its math.
                        sixteen_exact = stated
                            && (unorm8
                                || constant
                                || (b.fitted() && !folds_into_blend(cx, color, math)?));
                        match sixteen_exact || fetched.contains(&color) {
                            true => Ok((view, None)),
                            false => Err(not_implemented()),
                        }
                    }
                    (Some(Err(Error::Unsupported(gap))), _) => Err(gap),
                    (Some(Err(e)), _) => return Err(tied(e)),
                    // A second (index 1) output beside the first, which Metal's measured blends
                    // do not read: the generic blend, which stores it nowhere.
                    (Some(Ok(_)), None) if second.is_some() && view.word.is_none() => {
                        generic_of(view, &mut sixteen_exact)
                    }
                    _ => Err(not_implemented()),
                }
            }
            Ok(view) => match blend_of(&view, (outputs4.get(&color).copied(), blend_math(cx))) {
                // A blend of the second source reads the function's index-1 output (which it
                // writes: Metal refuses the pipeline otherwise, above), four `half` or `float`
                // lanes as the first's.
                Ok(Some(b)) if b.reads_second() => match second {
                    Some(Some(_)) => Ok((view, Some(b))),
                    _ => Err(Unsupported::not_implemented(Feature::ExtendedRangeBlend(
                        ExtendedRangeBlend::Output,
                    ))),
                },
                Ok(b) => Ok((view, b)),
                Err(Error::Unsupported(gap)) => Err(gap),
                Err(e) => return Err(tied(e)),
            },
            Err(blocked) => Err(blocked),
        };
        // The instance's `half` shadow of an extended-range attachment, which holds one sample
        // per pixel, of a form whose tile memory is measured (not a padded or bfloat one). The
        // shadow is the attachment's tile value only where every write of it keeps it, so a
        // shadowed attachment the storage form does not take is refused whatever the policy: an
        // unordered input attachment would neither read nor write it.
        let shadowed = cx.inputs.extended_shadows.contains(&color);
        let form = match form {
            Ok((view, _))
                if shadowed
                    && (view.extended.is_none_or(|e| e.padded() || e.bfloat())
                        || view.samples != StorageSamples::One) =>
            {
                return Err(tied(not_implemented().into()));
            }
            Err(blocked) if shadowed => return Err(tied(blocked.into())),
            form => form,
        };
        match form {
            Ok((view, b)) => {
                // The blend color, as push data, where the blend reads it.
                if b.as_ref()
                    .is_some_and(extended_range::Blend::reads_constant)
                {
                    cx.push.insert(PushField::BlendColor);
                }
                // The blend color as the encoder set it, where Metal's 16-bit float blend reads
                // it. Where that blend is not Metal's exactly, the function notes it.
                if let (Some(eq), None) = (view.blend, view.extended) {
                    let arith = extended_range::blend::BlendArith::Half;
                    let math = (blend_math(cx), arith);
                    let (pf, written) = (view.pixel_format, written(view.kept));
                    let half = match view.generic_blend {
                        true => generic(&eq, pf, written),
                        false => direct_blend(&eq, pf, written, math),
                    };
                    if half.is_ok_and(|h| h.reads_constant()) {
                        cx.push.insert(PushField::HalfBlendColor);
                    }
                    let four = srgb_output4(pf, blended_output(cx.entry, color));
                    let srgb_unmeasured = view.uint_view.is_some()
                        && !(four && srgb8_exact(&eq, pf, written, blend_math(cx)));
                    if view.generic_blend || srgb_unmeasured {
                        cx.precision.insert(PrecisionNote::UnmeasuredFetchBlend);
                    } else if view.word.is_some() {
                        cx.precision.insert(PrecisionNote::DeviceColorConversion);
                    } else if !sixteen_exact {
                        let note = crate::lower::arith::output_precision::half_output_note(cx);
                        cx.precision.insert(note);
                    }
                }
                // The channels a partial write mask keeps, as the draw's word.
                if view.kept_at_draw {
                    cx.push.insert(PushField::StorageKeptChannels);
                }
                let nearest_even = cx.inputs.texture_write_rounding
                    == crate::metal::TextureWriteRoundingMode::NearestEven;
                cx.families.fragment.fetch.insert(
                    color,
                    StorageColor {
                        view,
                        writes,
                        output: None,
                        shadowed,
                        second: None,
                        exact_blend: sixteen_exact,
                        nearest_even,
                    },
                );
                // An image holding Metal's alpha in red is read and written as Metal's alpha; the
                // interface records the attachment, which link holds the pipeline's to
                // (`color_writes::check`).
                if view.alpha_in_red {
                    cx.families
                        .stage_io
                        .interface
                        .alpha_in_red_colors
                        .insert(color);
                }
                cx.facts.insert(RuntimeFact::InterlockedFramebufferFetch {
                    color,
                    format: view.format,
                    writes,
                    early_tests,
                    kept: view.kept,
                    blend: view.blend,
                    rounded: view.blend.is_some()
                        && use_.math == extended_range::blend::BlendMath::Rounded,
                });
                cx.descriptors.storage_images = cx
                    .descriptors
                    .storage_images
                    .saturating_add(1 + u32::from(shadowed));
                // A merged write stores the kept channels as it read them: a `half` (or a packed
                // float) texel's NaN is the device's decode, as for any read of one.
                if view.merges() && narrow_float(view.format) {
                    cx.precision.insert(PrecisionNote::DeviceNanEncoding);
                }
            }
            Err(blocked) => return Err(tied(blocked.into())),
        }
    }
    if !cx.families.fragment.fetch.is_empty() {
        cx.families.fragment.fetch_early_tests = early_tests && !declared_early;
    }
    // Whether a write stores is the draw's tests' to decide. A write whose mask covers none of
    // its attachment's channels stores nothing, so no test bears on it: a moved color's neither,
    // for which the plan of the pipeline before the move selects no tests either
    // (`RuntimeFact::MovableStorageWrites`).
    let stores = cx
        .families
        .fragment
        .fetch
        .values()
        .any(|sc| sc.writes && !sc.view.stores_nothing());
    if (tested || untested) && stores {
        cx.facts.insert(RuntimeFact::TestedStorageWrites {
            early_tests,
            early_variant: tested && !untested,
        });
        // Compiled for a draw whose depth test alone can drop a fragment, the function runs that
        // test itself, before its store (`depth_test`).
        if cx.inputs.shader_tests {
            // A returned depth is tested, a returned stencil is the stencil test's reference, and
            // a returned sample mask keeps the fragment's samples before the tests.
            let field = |role: fn(&OutputRole) -> bool| {
                outputs.iter().find(|o| role(&o.role)).map(|o| o.field)
            };
            let exports = super::depth_test::Exports {
                depth: field(|r| matches!(r, OutputRole::Depth(_))),
                stencil: field(|r| matches!(r, OutputRole::Stencil)),
                mask: field(|r| matches!(r, OutputRole::SampleMask)),
            };
            super::depth_test::plan(cx, exports)?;
        }
    }
    // A per-sample function's returned mask gates its storage writes. At one sample too, on the
    // mask's bit 0 (measured: Metal stores no color there without it).
    let written = cx.families.fragment.fetch.values().any(|sc| sc.writes);
    if written {
        cx.families.fragment.fetch_mask = outputs
            .iter()
            .find(|o| matches!(o.role, OutputRole::SampleMask))
            .map(|o| o.field);
    }
    Ok(())
}

/// Plan the invocation's sample for a function compiled for a multisampled pipeline
/// (`LoweringInputs::multisampled`), which reads each color it fetches, in either form, at its
/// `SampleId`: Metal runs a function that reads a color attachment once per covered sample, reading
/// its own sample, and a `SampleId` input runs a Vulkan fragment shader once per covered sample.
/// Nothing for one sample.
pub(super) fn plan_sample(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if !cx.inputs.multisampled || cx.families.fragment.fetch_sample.is_some() {
        return Ok(());
    }
    if !cx.profile.fields().fragment.sample_rate_shading {
        return Err(Unsupported::needs(
            Feature::FramebufferFetch,
            Capability::Feature(VkFeature::SampleRateShading),
        )
        .into());
    }
    cx.families.fragment.fetch_sample = Some(stage_io::sample_id(cx));
    Ok(())
}

/// Whether the function reads color attachment `color` in the storage form.
pub(crate) fn is_storage(cx: &SelectCx<'_>, color: u32) -> bool {
    cx.families.fragment.fetch.contains_key(&color)
}

/// Whether a value of `form` has the numeric class of `numeric` (a float of `Float`, an integer
/// of `Uint` or `Sint`). A narrower integer written to a signed-integer attachment is not
/// lowered: which extension fills the attachment's lanes is not established (as for a color
/// output). A read takes the texel's low bits, as an input attachment's does.
fn class_of(form: Form, numeric: SampledType, color: u32, write: bool) -> Result<(), Error> {
    let ok = match numeric {
        SampledType::Float => form.is_float(),
        SampledType::Uint => !form.is_float(),
        SampledType::Sint => !form.is_float(),
        // No color attachment's format reads as `ulong`.
        SampledType::Uint64 => false,
    };
    if !ok {
        return Err(Error::InvalidDescriptor(DescriptorError::ColorAttachment {
            index: color,
        }));
    }
    if write && numeric == SampledType::Sint && !matches!(form.held, Held::Int { bits: 32, .. }) {
        return Err(not_lowered());
    }
    Ok(())
}

/// Plan the read of storage-form color input `c`: its class against the attachment's, and the
/// pixel's `FragCoord`.
pub(super) fn select_read(cx: &mut SelectCx<'_>, c: ColorInput) -> Result<(), Error> {
    let Some(sc) = cx.families.fragment.fetch.get(&c.color).copied() else {
        return Err(cx.invariant("a storage-form read of an attachment of another form"));
    };
    // A packed 16-bit word attachment's read decodes the word, Metal's value
    // (`stage_io::word_color::decode`, [`value`]). An 8-bit sRGB word's decode of its bytes, and
    // an `RGB9E5Float` word's of its shared exponent, are not built.
    if matches!(
        sc.view.word,
        Some(WordAttachment::Srgb8 { .. } | WordAttachment::Rgb9e5)
    ) {
        let refused = Unsupported::not_implemented(Feature::Format);
        return Err(refused.at_attachment(Attachment::Color(c.color)).into());
    }
    class_of(c.form, sc.view.read_class(), c.color, false)?;
    // An `_sRGB` extended-range texel decodes through Metal's table.
    if let Some(e) = sc.view.extended {
        cx.families.arith.converts(e);
    }
    // An sRGB texel decodes by the device's conversion.
    if sc.view.uint_view.is_some() {
        cx.precision.insert(PrecisionNote::DevicePackConversion);
    }
    plan_sample(cx)?;
    let coord = stage_io::frag_coord(cx);
    cx.families.fragment.fetch_coord = Some(coord);
    Ok(())
}

/// The function's second (index-1) output to attachment `color`: `None` where it writes none,
/// `Some(Some(fmt))` where it is four lanes of `fmt`, a `half` or a `float`, and `Some(None)` for
/// another.
pub(super) fn second_output4(
    cx: &SelectCx<'_>,
    color: u32,
) -> Result<Option<Option<FloatFmt>>, Error> {
    for (o, ty) in cx
        .entry
        .ep
        .outputs
        .iter()
        .zip(stage_io::output_types(cx)?.1)
    {
        if let OutputRole::RenderTarget {
            attachment: AbiInt::Const(c),
            index: 1,
            ..
        } = o.role
        {
            if c == color {
                let f = super::color::form(cx, ty)?;
                let fmt = f.float_fmt().filter(|_| f.lanes == 4);
                return Ok(Some(
                    fmt.filter(|f| matches!(f, FloatFmt::Half | FloatFmt::F32)),
                ));
            }
        }
    }
    Ok(None)
}

/// Whether the function's second (index-1) output to attachment `color` is the second source of
/// an extended-range or generic blend the shader computes in the storage form
/// ([`select_second_output`]), which no interface variable holds.
pub(crate) fn blends_second(cx: &SelectCx<'_>, color: u32) -> bool {
    cx.families.fragment.fetch.get(&color).is_some_and(|sc| {
        sc.view.blend.is_some()
            && (sc.view.extended.is_some() || sc.view.generic_blend || sc.view.uint_view.is_some())
    })
}

/// Plan the second (index-1) output of AIR type `ty` held in field `field` of the returned value
/// to storage-form attachment `color`, whose extended-range or generic blend the shader computes
/// ([`blends_second`]): the blend reads it where its factors name the second source (an
/// extended-range one's rounded to `half`), clamped as the first output; else nothing stores it,
/// as Metal's blend there reads none.
pub fn select_second_output(
    cx: &mut SelectCx<'_>,
    field: u32,
    color: u32,
    ty: AirTypeId,
) -> Result<(), Error> {
    let form = super::color::form(cx, ty)?;
    let Some(sc) = cx.families.fragment.fetch.get_mut(&color) else {
        return Err(cx.invariant("a second output of an attachment of another form"));
    };
    sc.second = Some((field, form));
    Ok(())
}

/// Plan the output of AIR type `ty` held in field `field` of the returned value to storage-form
/// attachment `color` (`stage_io` plans the entry's other outputs): its form has the
/// attachment's class and at least its format's channels, as a color output must.
pub fn select_output(
    cx: &mut SelectCx<'_>,
    field: u32,
    color: u32,
    ty: AirTypeId,
) -> Result<(), Error> {
    let form = super::color::form(cx, ty)?;
    let Some(sc) = cx.families.fragment.fetch.get(&color).copied() else {
        return Err(cx.invariant("a storage-form output of an attachment of another form"));
    };
    class_of(form, sc.view.read_class(), color, true)?;
    // An extended-range attachment's blend is of a `half4` output, or of a `float4` one where
    // the blend takes it ([`plan`]), and an `_sRGB` one's encodes through Metal's table.
    let output4 = (form.lanes == 4).then(|| form.float_fmt()).flatten();
    if blend_of(&sc.view, (output4, blend_math(cx))).is_err() {
        return Err(cx.invariant("an extended-range blend of an output outside its model"));
    }
    if let Some(e) = sc.view.extended {
        // An unblended write stores Metal's write of a `float` or `half` output; a `bfloat` one's
        // is not measured.
        let measured = matches!(
            form.float_fmt(),
            Some(crate::lower::arith::FloatFmt::F32 | crate::lower::arith::FloatFmt::Half)
        );
        if !measured {
            return Err(Unsupported::not_implemented(Feature::Format).into());
        }
        cx.families.arith.converts(e);
    }
    let pf = cx
        .inputs
        .framebuffer_fetch
        .as_ref()
        .and_then(|i| i.attachments.get(&color).copied().flatten())
        .map(|a| a.pixel_format)
        .ok_or_else(|| cx.invariant("a storage-form output without its attachment"))?;
    if color_output_components(pf).is_some_and(|n| form.lanes < n) {
        return Err(Error::InvalidDescriptor(DescriptorError::ColorAttachment {
            index: color,
        }));
    }
    plan_sample(cx)?;
    // The write rounds to the code Metal stores where the format converts it; an sRGB attachment's
    // write encodes its bytes itself (`arith::pack::pack` of a `float` or `half` output), and a word
    // attachment's Metal's word of a `float` or `half` output ([`word_texel`]): each records the
    // pipeline's map, which link checks, without rounding a lane.
    if sc.view.word.is_some() {
        let measured = matches!(
            form.float_fmt(),
            Some(crate::lower::arith::FloatFmt::F32 | crate::lower::arith::FloatFmt::Half)
        );
        if !measured {
            return Err(Unsupported::not_implemented(Feature::Format).into());
        }
        // An 8-bit sRGB word encodes by the device's conversion.
        if let Some(WordAttachment::Srgb8 { .. }) = sc.view.word {
            cx.precision.insert(PrecisionNote::DevicePackConversion);
        }
        if let Some(rules) = cx.inputs.color_writes.get(&color).copied() {
            cx.families
                .stage_io
                .interface
                .color_writes
                .insert(color, rules);
        }
    } else if sc.view.uint_view.is_some() {
        let measured = matches!(
            form.float_fmt(),
            Some(crate::lower::arith::FloatFmt::F32 | crate::lower::arith::FloatFmt::Half)
        );
        if !measured {
            return Err(Unsupported::not_implemented(Feature::Format).into());
        }
        cx.precision.insert(PrecisionNote::DevicePackConversion);
        if let Some(rules) = cx.inputs.color_writes.get(&color).copied() {
            cx.families
                .stage_io
                .interface
                .color_writes
                .insert(color, rules);
        }
    } else {
        stage_io::color_write::plan(cx, color, form);
        // An 8-bit unorm blend's result is the storage image's to convert, the device's
        // conversion ([`unorm8_blended`]).
        if sc.view.blend.is_some() && !sc.view.generic_blend && unorm8_format(sc.view.pixel_format)
        {
            cx.precision.insert(PrecisionNote::DeviceColorConversion);
        }
    }
    let coord = stage_io::frag_coord(cx);
    let fp = &mut cx.families.fragment;
    fp.fetch_coord = Some(coord);
    match fp.fetch.get_mut(&color) {
        Some(sc) if sc.writes && sc.output.is_none() => {
            sc.output = Some((field, form));
            Ok(())
        }
        _ => Err(invariant("a storage-form output the plan does not expect")),
    }
}

fn invariant(detail: &str) -> Error {
    Error::Internal(crate::error::InternalError::new(
        crate::error::InternalKind::Invariant,
        detail,
    ))
}

/// Check the plan once every output is planned: each written storage-form attachment has its
/// output.
pub(super) fn finish(cx: &SelectCx<'_>) -> Result<(), Error> {
    if cx
        .families
        .fragment
        .fetch
        .values()
        .any(|sc| sc.writes != sc.output.is_some())
    {
        return Err(invariant(
            "a written storage-form attachment without its output",
        ));
    }
    Ok(())
}

/// An upper estimate of the words the storage form adds: one object per attachment, the pixel's
/// coordinate (and sample) and a read per use, and at each of `rets` returns the coordinate, the
/// guard (with the returned mask's bit), and a write per written attachment (a read first where
/// it merges kept channels, and a decode, blend, and encode where it blends an extended-range
/// texel; an exchange of red and blue per read and write of an attachment viewed in its
/// RGBA-order twin).
pub(super) fn words(cx: &SelectCx<'_>, uses: u64, rets: u64) -> u64 {
    const OBJECT: u64 = 40;
    const READ: u64 = 56;
    const WRITE: u64 = 56;
    let fetch = &cx.families.fragment.fetch;
    if fetch.is_empty() {
        return 0;
    }
    let written = fetch.values().filter(|sc| sc.writes).count() as u64;
    let merged = fetch.values().filter(|sc| sc.view.merges()).count() as u64;
    // A merge of the draw's kept channels tests four bits and selects.
    const KEPT_AT_DRAW: u64 = 40;
    let at_draw = fetch.values().filter(|sc| sc.view.kept_at_draw).count() as u64;
    // An extended-range attachment's blended write reads, decodes, blends, and encodes its
    // texel, its blend's words by its arithmetic, the larger of a `half4` and a `float4` output's;
    // an unblended one rounds and encodes the output.
    let blend: u64 = fetch
        .values()
        .filter_map(|sc| {
            [FloatFmt::Half, FloatFmt::F32]
                .into_iter()
                .filter_map(|o| match blend_of(&sc.view, (Some(o), blend_math(cx))) {
                    Ok(Some(b)) => Some(u64::from(b.words())),
                    _ => None,
                })
                .max()
        })
        .map(|b| {
            READ.saturating_add(b).saturating_add(u64::from(
                extended_range::DECODE_WORDS + extended_range::ENCODE_WORDS,
            ))
        })
        .fold(0u64, u64::saturating_add);
    // A 16-bit float attachment's blended write (and a generic one, in `float`, its source and
    // blend color clamped) reads its texel and blends in `half`, or in `float` with exact
    // multiply-adds (twice the words) and Metal's conversion of the result; an
    // 8-bit unorm one's clamps the output and decodes the codes (the storage image's store
    // converts the result); a
    // packed 16-bit one's clamps the output and rounds it to `half`, and decodes the word.
    let sixteen: u64 = fetch
        .values()
        .filter(|sc| sc.view.extended.is_none())
        .filter_map(|sc| {
            let eq = sc.view.blend?;
            let kept = self::written(sc.view.kept);
            let pf = sc.view.pixel_format;
            let half = extended_range::blend::BlendArith::Half;
            let b = match sc.view.generic_blend {
                true => generic(&eq, pf, kept).ok()?,
                false => direct_blend(&eq, pf, kept, (blend_math(cx), half)).ok()?,
            };
            let unorm8 = match unorm8_format(pf) {
                true => 48 * 6,
                false => 0,
            };
            let packed = match sc.view.word {
                Some(_) => 48 * 6 + stage_io::word_color::DECODE_WORDS,
                None => 0,
            };
            Some(
                READ.saturating_add(2 * u64::from(b.words()))
                    .saturating_add(32 * 6)
                    .saturating_add(unorm8)
                    .saturating_add(packed),
            )
        })
        .fold(0u64, u64::saturating_add);
    let unblended = fetch
        .values()
        .filter(|sc| sc.writes && sc.view.extended.is_some() && sc.view.blend.is_none())
        .count() as u64;
    // A shadowed attachment's object, and a read and a write of its shadow at each return.
    let shadowed = fetch.values().filter(|sc| sc.shadowed).count() as u64;
    // An extended-range attachment's read decodes its texel, an sRGB one's reorders its bytes and
    // decodes them (`arith::pack::unpack`).
    // An RGBA-order view's read and write exchange red and blue: one shuffle each.
    const EXCHANGE: u64 = 10;
    let exchanged = fetch
        .values()
        .filter(|sc| sc.view.rgba_view.is_some())
        .count() as u64;
    // An alpha-in-red image's read moves its red to alpha, and its write alpha to red.
    let moved = fetch.values().filter(|sc| sc.view.alpha_in_red).count() as u64;
    let read = match (
        fetch.values().any(|sc| sc.view.extended.is_some()),
        fetch.values().any(|sc| sc.view.uint_view.is_some()),
    ) {
        (true, _) => READ.saturating_add(u64::from(extended_range::DECODE_WORDS) + 120),
        (false, true) => READ.saturating_add(u64::from(srgb_words(PackDir::Unpack)) + 120),
        (false, false) => READ,
    }
    // A packed 16-bit word attachment's read decodes the word's four fields. Added, not chosen:
    // a function may read such an attachment beside one of another kind, and every use takes the
    // larger shape.
    .saturating_add(match fetch.values().any(|sc| sc.view.word.is_some()) {
        true => stage_io::word_color::DECODE_WORDS,
        false => 0,
    })
    .saturating_add(EXCHANGE.saturating_mul(exchanged))
    .saturating_add(super::color::ALPHA_IN_RED_WORDS.saturating_mul(moved));
    // An sRGB attachment's write encodes four channels (`arith::pack::pack`), reorders its bytes,
    // and merges kept ones from the texel it reads first.
    let srgb_writes = fetch
        .values()
        .filter(|sc| sc.writes && sc.view.uint_view.is_some())
        .count() as u64;
    let srgb_write = READ.saturating_add(u64::from(srgb_words(PackDir::Pack)) + 120);
    // A word attachment's write encodes Metal's word and merges the kept fields: its kept mask
    // from the draw's four bits (a test and a select per field).
    let word_write: u64 = fetch
        .values()
        .filter(|sc| sc.writes)
        .filter_map(|sc| sc.view.word)
        .map(|w| stage_io::word_color::encode_words(w) + 4 * 24 + 40)
        .fold(0u64, u64::saturating_add);
    let write = WRITE
        .saturating_mul(written)
        .saturating_add(READ.saturating_mul(merged))
        .saturating_add(KEPT_AT_DRAW.saturating_mul(at_draw))
        .saturating_add(blend)
        .saturating_add(sixteen)
        .saturating_add(stage_io::extended_color::WORDS.saturating_mul(unblended))
        .saturating_add(srgb_write.saturating_mul(srgb_writes))
        .saturating_add(word_write)
        .saturating_add((READ + WRITE + 8).saturating_mul(shadowed))
        .saturating_add((2 * EXCHANGE).saturating_mul(exchanged))
        .saturating_add((EXCHANGE + super::color::ALPHA_IN_RED_WORDS).saturating_mul(moved))
        .saturating_add(40);
    OBJECT
        .saturating_mul(fetch.len() as u64 + shadowed)
        .saturating_add(read.saturating_mul(uses))
        .saturating_add(rets.saturating_mul(write))
}

/// Declare storage-form attachment `color`'s storage image: a 2D storage image of its declared
/// format and its numeric type, multisampled or arrayed as it holds the samples
/// ([`StorageSamples`]).
pub(super) fn declare(mb: &mut ModuleBuilder, color: u32, sc: &StorageColor) -> BResult<GlobalId> {
    let sampled_type = scalar_type(mb, sc.view.numeric)?;
    let ty = mb.ty(SType::Image(ImageType {
        sampled_type,
        dim: Dim::D2,
        depth: false,
        arrayed: sc.view.samples == StorageSamples::Planes,
        multisampled: sc.view.samples == StorageSamples::Multisampled,
        usage: ImageUsage::Storage,
        format: sc.view.declared,
    }))?;
    let slot = ResourceSlot {
        stage: Stage::Fragment,
        kind: ResourceKind::TileAttachment(color),
        index: color,
    };
    mb.resource(slot, StorageClass::UniformConstant, ty)
}

/// Declare the `half` shadow of extended-range attachment `color` that the draw's render pass
/// instance holds (`StorageColor::shadowed`): a single-sample 2D `Rgba16f` storage image of
/// `float` texels (`ResourceKind::ExtendedShadow`), each Metal's tile value of the texel. Every
/// value it holds is a `half`, so the device's conversions both ways are exact.
pub(super) fn declare_shadow(mb: &mut ModuleBuilder, color: u32) -> BResult<GlobalId> {
    let sampled_type = mb.ty(SType::Float { width: 32 })?;
    let ty = mb.ty(SType::Image(ImageType {
        sampled_type,
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: false,
        usage: ImageUsage::Storage,
        format: crate::spv::ImageFormat::Rgba16f,
    }))?;
    let slot = ResourceSlot {
        stage: Stage::Fragment,
        kind: ResourceKind::ExtendedShadow(color),
        index: color,
    };
    mb.resource(slot, StorageClass::UniformConstant, ty)
}

/// The shadow of attachment `color` and the invocation's pixel, its texel there
/// ([`declare_shadow`]).
fn shadow_and_texel(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
) -> BResult<(SValue, SValue)> {
    let Some(g) = cx
        .decl
        .families
        .fragment
        .extended_shadows
        .get(&color)
        .copied()
    else {
        return Err(cx.fail(format!("the shadow of attachment {color} is not declared")));
    };
    let p = fb.module().global_value(g)?;
    let image = fb.load(p, MemAccess::default())?;
    let pixel = pixel(cx, fb)?;
    Ok((image, pixel))
}

/// The tile value the shadow of attachment `color`, held in `range`, holds at the invocation's
/// pixel, read inside the interlock: Metal's read of what the tile holds
/// (`extended_range::tile_read`: the color lanes clamped to the blend's source range, alpha to
/// `[0, 1]`, a NaN to the low end, `BGR10_XR`'s alpha one).
fn shadow_read(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
    range: ExtendedRange,
) -> BResult<SValue> {
    let (image, at) = shadow_and_texel(cx, fb, color)?;
    let mem = memory_model::coherence(cx.plan).texel(fb)?;
    let texel = fb.read(image, at, &ImageOperands::default(), mem)?;
    extended_range::tile_read(fb, range, texel)
}

/// The 32-bit scalar type of `numeric`.
fn scalar_type(mb: &mut ModuleBuilder, numeric: SampledType) -> BResult<crate::spirv::ids::TypeId> {
    match numeric {
        SampledType::Float => mb.ty(SType::Float { width: 32 }),
        // A `ulong` texture's image is `RG32` unsigned.
        SampledType::Uint | SampledType::Uint64 => mb.ty(SType::Int {
            width: 32,
            signed: false,
        }),
        SampledType::Sint => mb.ty(SType::Int {
            width: 32,
            signed: true,
        }),
    }
}

/// The storage image of attachment `color` held as `samples`, the texel of the invocation's
/// sample, and the access's image operands: the invocation's [`pixel`] with its [`operands`],
/// or, for planes, the pixel and the invocation's `SampleId` as the layer.
fn image_and_texel(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
    samples: StorageSamples,
) -> BResult<(SValue, SValue, ImageOperands)> {
    let Some(g) = cx.decl.families.fragment.attachments.get(&color).copied() else {
        return Err(cx.fail(format!("storage attachment {color} is not declared")));
    };
    let p = fb.module().global_value(g)?;
    let image = fb.load(p, MemAccess::default())?;
    let pixel = pixel(cx, fb)?;
    if samples != StorageSamples::Planes {
        let ops = operands(cx, fb)?;
        return Ok((image, pixel, ops));
    }
    let Some(layer) = sample(cx, fb)? else {
        return Err(cx.fail("a planes access without its SampleId"));
    };
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = fb.module().ty(SType::Vector { elem: u32_, len: 3 })?;
    let x = fb.composite_extract(pixel, &[0])?;
    let y = fb.composite_extract(pixel, &[1])?;
    let texel = fb.composite_construct(uvec3, &[x, y, layer])?;
    Ok((image, texel, ImageOperands::default()))
}

/// The texel of storage-form attachment `sc`'s image at the invocation's pixel (and sample) of
/// attachment `color`, read inside the interlock, in Metal's channel order ([`metal_order`]):
/// every read of the attachment's image goes through here, so a read, a merge's kept channels, a
/// blend's destination and the per-sample forms see Metal's channels alike.
fn read_texel(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
    sc: &StorageColor,
) -> BResult<SValue> {
    let (image, at, ops) = image_and_texel(cx, fb, color, sc.view.samples)?;
    let mem = memory_model::coherence(cx.plan).texel(fb)?;
    let texel = fb.read(image, at, &ops, mem)?;
    metal_order(fb, sc, texel)
}

/// Texel `texel` of storage-form attachment `sc`'s image, in Metal's channel order: an attachment
/// viewed in its RGBA-order twin ([`StorageView::rgba_view`]) has red and blue exchanged, and one
/// whose image holds Metal's alpha in red ([`StorageView::alpha_in_red`]) reads `(r, 0, 0, 1)` as
/// Metal's `(0, 0, 0, r)`; any other is as it is. Every texel read from the attachment's image
/// goes through here ([`read_texel`]), and every one written to it through [`image_order`].
fn metal_order(fb: &mut FunctionBuilder<'_>, sc: &StorageColor, texel: SValue) -> BResult<SValue> {
    match (sc.view.rgba_view, sc.view.alpha_in_red) {
        (Some(_), _) => fb.vector_shuffle(texel, texel, &[2, 1, 0, 3]),
        (None, true) => super::color::alpha_in_red(fb, texel),
        (None, false) => Ok(texel),
    }
}

/// Metal-ordered texel `texel` of storage-form attachment `sc`, in its image's channel order:
/// [`metal_order`]'s exchange of red and blue, which is its own inverse, and for an image holding
/// Metal's alpha in red, Metal's alpha in every lane (the image keeps its red). Every texel
/// written to the attachment's image goes through here ([`emit_return`]).
fn image_order(fb: &mut FunctionBuilder<'_>, sc: &StorageColor, texel: SValue) -> BResult<SValue> {
    match (sc.view.rgba_view, sc.view.alpha_in_red) {
        (Some(_), _) => fb.vector_shuffle(texel, texel, &[2, 1, 0, 3]),
        (None, true) => fb.vector_shuffle(texel, texel, &[3, 3, 3, 3]),
        (None, false) => Ok(texel),
    }
}

/// The image operands of a fetched color's read or write in a multisampled image (an input
/// attachment, or a multisampled storage image): the invocation's `SampleId` for a per-sample
/// function (`FragmentPlan::fetch_sample`), none for one sample.
pub(super) fn operands(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
) -> BResult<ImageOperands> {
    let Some(sample) = sample(cx, fb)? else {
        return Ok(ImageOperands::default());
    };
    Ok(ImageOperands {
        sample: Some(sample),
        ..ImageOperands::default()
    })
}

/// The invocation's `SampleId`, loaded, for a per-sample function; `None` for one sample.
pub(super) fn sample(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<Option<SValue>> {
    let Some(object) = cx.plan.families.fragment.fetch_sample else {
        return Ok(None);
    };
    let g = stage_io::object_global(cx, object)?;
    let p = fb.module().global_value(g)?;
    fb.load(p, MemAccess::default()).map(Some)
}

/// The invocation's pixel, a `uvec2` from the integer part of `FragCoord.xy` (the pixel's
/// center, single-sampled): the texel of a storage image the interlocked forms hold per pixel.
pub(super) fn pixel(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
    let Some(coord) = cx.plan.families.fragment.fetch_coord else {
        return Err(cx.fail("a storage-form access without its FragCoord"));
    };
    let frag = stage_io::object_global(cx, coord)?;
    let frag = fb.module().global_value(frag)?;
    let frag = fb.load(frag, MemAccess::default())?;
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let uvec2 = fb.module().ty(SType::Vector { elem: u32_, len: 2 })?;
    let fx = fb.composite_extract(frag, &[0])?;
    let fy = fb.composite_extract(frag, &[1])?;
    let x = fb.convert(ConvOp::ConvertFToU, fx, u32_)?;
    let y = fb.convert(ConvOp::ConvertFToU, fy, u32_)?;
    fb.composite_construct(uvec2, &[x, y])
}

/// A `uvec4` of a signed-integer texel's bits, or the texel itself.
fn as_held(fb: &mut FunctionBuilder<'_>, texel: SValue, numeric: SampledType) -> BResult<SValue> {
    if numeric != SampledType::Sint {
        return Ok(texel);
    }
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let uvec4 = fb.module().ty(SType::Vector { elem: u32_, len: 4 })?;
    fb.convert(ConvOp::Bitcast, texel, uvec4)
}

/// The value of storage-form color input `c` in its lowered representation, read at its pixel
/// (as `color::value` reads an input attachment).
pub(super) fn value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    c: ColorInput,
) -> BResult<SValue> {
    let Some(sc) = cx.plan.families.fragment.fetch.get(&c.color).copied() else {
        return Err(cx.fail(format!("attachment {} is not in the storage form", c.color)));
    };
    let texel = match (sc.shadowed, sc.view.extended) {
        // The instance's shadow holds the tile value itself.
        (true, Some(range)) => Some(shadow_read(cx, fb, c.color, range)?),
        (true, None) => return Err(cx.fail("a shadow of an attachment of another form")),
        (false, _) => None,
    };
    let texel = match texel {
        Some(held) => held,
        None => read_texel(cx, fb, c.color, &sc)?,
    };
    let texel = match (sc.view.extended, sc.view.uint_view) {
        (Some(_), _) if sc.shadowed => texel,
        (Some(form), _) => {
            let table = crate::lower::texture::extended_table(cx, form)?;
            extended_range::tile_value(fb, form, texel, table)?
        }
        (None, Some(_)) => {
            let word = fb.composite_extract(texel, &[0])?;
            srgb_decode(fb, sc.view.format, word)?
        }
        // A packed 16-bit word attachment's texel is the word, whose channels the read decodes as
        // Metal's programmable blending hands them to the function, and as its blend reads them
        // (`stage_io::word_color::decode`; the measurement is `metal::format::word`'s). The other
        // word kinds' reads are refused in the plan ([`select_read`]).
        (None, None) if sc.view.word.is_some() => {
            let Some(word) = sc.view.word else {
                return Err(cx.fail("a word read of an attachment without a word"));
            };
            let w = fb.composite_extract(texel, &[0])?;
            stage_io::word_color::decode(fb, word, w)?
        }
        (None, None) => as_held(fb, texel, sc.view.numeric)?,
    };
    let x = match c.form.lanes {
        4 => texel,
        1 => fb.composite_extract(texel, &[0])?,
        n => {
            let lanes: Vec<u32> = (0..u32::from(n)).collect();
            fb.vector_shuffle(texel, texel, &lanes)?
        }
    };
    let x = ops::mask(fb, c.form, x)?;
    narrow::round(fb, c.form, x)
}

/// Write each written storage-form attachment's output of returned value `v` (in the entry's
/// returned shape) at its pixel, unless a discard was recorded (`discarded`).
pub(super) fn emit_return(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    v: Option<crate::air::ir::ValueId>,
) -> BResult<()> {
    let writes: Vec<(u32, StorageColor, u32, Form)> = cx
        .plan
        .families
        .fragment
        .fetch
        .iter()
        .filter(|(_, sc)| !sc.view.stores_nothing())
        .filter_map(|(&color, sc)| sc.output.map(|(field, form)| (color, *sc, field, form)))
        .collect();
    if writes.is_empty() {
        return Ok(());
    }
    let Some(v) = v else {
        return Err(cx.fail("a storage-form output without a returned value"));
    };
    let x = cx.operand(fb, v)?;
    let mut texels = Vec::with_capacity(writes.len());
    for (color, sc, field, form) in writes {
        let part = match cx.plan.families.stage_io.returned {
            Returned::Struct => fb.composite_extract(x, &[field])?,
            Returned::Value => x,
            Returned::Void => return Err(cx.fail("a storage-form output of a void entry")),
        };
        // The tile value the write leaves, which the instance's shadow stores.
        let mut held = None;
        // The second (index-1) output, the blend's second source where it reads one.
        let second = match (sc.second, cx.plan.families.stage_io.returned) {
            (Some((field, form)), Returned::Struct) => {
                Some((form, fb.composite_extract(x, &[field])?))
            }
            (Some(_), _) => return Err(cx.fail("a second output outside a returned structure")),
            (None, _) => None,
        };
        let texel = match (sc.view.extended, sc.view.blend) {
            (Some(range), Some(_)) => {
                let (texel, shadow) = blended(cx, fb, color, &sc, range, (form, part), second)?;
                held = shadow;
                texel
            }
            (Some(range), None) if !sc.shadowed => {
                stage_io::extended_color::texel(cx, fb, range, form, part)?
            }
            // The output's `half`, which the tile holds and the codes encode.
            (Some(range), None) => {
                let v = stage_io::extended_color::held(fb, range, form, part)?;
                held = Some(v);
                let table = crate::lower::texture::extended_table(cx, range)?;
                extended_range::encode(fb, range, v, table)?
            }
            (None, _) if sc.view.uint_view.is_some() => {
                srgb_texel(cx, fb, color, &sc, (form, part), second)?
            }
            (None, _) if sc.view.word.is_some() => word_texel(cx, fb, color, &sc, form, part)?,
            (None, _) => {
                // Metal's 16-bit float blend of an output of one to four lanes, its 8-bit unorm
                // blend of four, or the generic blend, over the texel.
                let part = match sc.view.blend {
                    Some(eq) if sc.view.generic_blend => {
                        generic_blended(cx, fb, color, &sc, eq, (form, part), second)?
                    }
                    Some(eq) if unorm8_format(sc.view.pixel_format) => {
                        unorm8_blended(cx, fb, color, &sc, eq, (form, part))?
                    }
                    Some(eq) if float32_format(sc.view.pixel_format) => {
                        float32_blended(cx, fb, color, &sc, eq, (form, part))?
                    }
                    Some(eq) => half_blended(cx, fb, color, &sc, eq, (form, part))?,
                    None => part,
                };
                let part = if sc.view.merges() {
                    merge(cx, fb, color, &sc, form, part)?
                } else {
                    part
                };
                let part = stage_io::color_write::round(cx, fb, color, form, part)?;
                texel_of(fb, part, form, sc.view.numeric)?
            }
        };
        // The image's texel of the Metal-ordered one.
        let texel = image_order(fb, &sc, texel)?;
        let (image, at, ops) = image_and_texel(cx, fb, color, sc.view.samples)?;
        texels.push((image, at, texel, ops));
        if let Some(held) = held {
            let (image, at) = shadow_and_texel(cx, fb, color)?;
            texels.push((image, at, held, ImageOperands::default()));
        }
    }
    let mem = memory_model::coherence(cx.plan).texel(fb)?;
    let write = move |fb: &mut FunctionBuilder<'_>, texels: &mut Vec<Write>| {
        for (image, at, texel, ops) in texels.iter() {
            fb.write(*image, *at, *texel, ops, mem)?;
        }
        Ok(())
    };
    // The guard: not discarded, and, for a function that returns a sample mask, the mask's bit of
    // the invocation's own sample (bit 0 at one sample).
    let discarded = if cx.plan.families.fragment.records_discards() {
        let Some(g) = cx.decl.families.fragment.discarded else {
            return Err(cx.fail("a recorded discard without its predicate"));
        };
        let p = fb.module().global_value(g)?;
        let discarded = fb.load(p, MemAccess::default())?;
        Some(fb.unary(UnOp::LogicalNot, discarded)?)
    } else {
        None
    };
    let masked = match cx.plan.families.fragment.fetch_mask {
        Some(field) => Some(sample_bit(cx, fb, x, field)?),
        None => None,
    };
    let kept = match (discarded, masked) {
        (None, None) => return write(fb, &mut texels),
        (Some(k), None) | (None, Some(k)) => k,
        (Some(a), Some(b)) => fb.binary(BinOp::LogicalAnd, a, b)?,
    };
    let arm: ArmFn<'_, Vec<Write>> = Box::new(move |fb, texels| {
        write(fb, texels)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut texels, kept, arm, None)?;
    Ok(())
}

/// One storage write: the image, the texel's coordinate, the texel, and the image operands.
type Write = (SValue, SValue, SValue, ImageOperands);

/// The output `part` of `form` (one to four `f32` lanes: a `half` output's `half`s, or a `float`
/// one's `float`s) blended into 16-bit float attachment `color` by Metal's blend `eq`
/// (`extended_range::Blend::half_float`, of the function's math): the texel read first at the
/// invocation's pixel (and sample), inside the interlock, as [`merge`] reads it, and Metal's
/// arithmetic of the output over it in the output's type
/// (`extended_range::blend::blend_half_float`; the blend color from `PushField::HalfBlendColor`
/// where the blend reads it), a `float` result converted to `half` by the storage image's store,
/// the device's conversion (Metal's blended write's, toward zero, the largest finite `half` past
/// it, where the profile takes `DriverDeviation::MetalHalfStorageWrites`), or rounded to nearest
/// even under `TextureWriteRoundingMode::NearestEven` (`color_writes::HALF_NEAREST`), which no
/// measured device's store gives. A NaN of the blend is the device's arithmetic's, which the
/// store converts (`PrecisionNote::DeviceNanEncoding`).
/// An output of fewer than four lanes blends as four whose missing lanes are zero, alpha among
/// them, and the result keeps the output's lanes. The channels a write
/// mask keeps are [`merge`]'s to take back from the texel.
fn half_blended(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
    sc: &StorageColor,
    eq: BlendEquation,
    (form, part): (Form, SValue),
) -> BResult<SValue> {
    let math = extended_range::blend::BlendMath::of(cx.entry.compile.fast_math);
    let (pf, written) = (sc.view.pixel_format, written(sc.view.kept));
    let Ok(b) = extended_range::Blend::half_float(&eq, pf, written, math) else {
        return Err(cx.fail("a 16-bit float blend outside Metal's model"));
    };
    let arith = match (form.lanes, form.float_fmt()) {
        (1..=4, Some(FloatFmt::Half)) => extended_range::blend::BlendArith::Half,
        (1..=4, Some(FloatFmt::F32)) => extended_range::blend::BlendArith::Float,
        _ => return Err(cx.fail("a 16-bit float blend of another output")),
    };
    // The lanes an output of fewer than four lacks are undefined to Metal's compiler, which folds
    // color channels whose blend reads the source alpha as a factor to `+0` under fast math and
    // stores its NaN without; such an output's format has no alpha channel.
    if form.lanes < 4 && b.color_reads_source_alpha() {
        let bits = match math {
            extended_range::blend::BlendMath::Fast => 0,
            extended_range::blend::BlendMath::Rounded => 0x7FC0_0000,
        };
        return ops::splat(fb, form.with(Held::Float(FloatFmt::F32)), bits);
    }
    let texel = read_texel(cx, fb, color, sc)?;
    let d = as_held(fb, texel, sc.view.numeric)?;
    let constant = match b.reads_constant() {
        true => {
            let field = cx.push_field(fb, PushField::HalfBlendColor)?;
            Some(fb.composite_extract(field, &[0])?)
        }
        false => None,
    };
    let s = four_lanes(fb, form, part)?;
    let v = extended_range::blend::blend_half_float(fb, &b, arith, (s, d), constant)?;
    let v = output_lanes(fb, form, v)?;
    if arith == extended_range::blend::BlendArith::Half || !sc.nearest_even {
        // The blend's result is the storage image's store's to convert, its NaN the device's
        // (`PrecisionNote::DeviceNanEncoding`): a `float` result as Metal's blended write converts
        // it toward zero where the profile takes `MetalHalfStorageWrites`.
        return Ok(v);
    }
    let rule = crate::pipeline::color_writes::HALF_NEAREST;
    crate::lower::arith::color_write::small_float(fb, form, v, rule)
}

/// The output `part` of `form` (four `f32` lanes: a `half4` output's `half`s, or a `float4` one's
/// `float`s) blended into 8-bit unorm attachment `color` by Metal's blend `eq`
/// (`extended_range::Blend::unorm8`, of the function's math and the output's type), stored as
/// Metal's unorm write stores it: the texel read first at the invocation's pixel (and sample),
/// inside the interlock, in Metal's channel order ([`read_texel`]: an attachment viewed in its
/// RGBA-order twin has red and blue exchanged, and the write exchanges them back,
/// [`emit_return`]), Metal's arithmetic of the clamped output over the destination's code decoded
/// in the output's type, the blend color from `PushField::HalfBlendColor` where the blend reads it
/// (`extended_range::blend::blend_unorm8`), the result written as it is: the storage image's
/// store converts it to its code, the device's conversion (`PrecisionNote::DeviceColorConversion`);
/// a channel the write mask keeps takes the texel's after (the module docs).
fn unorm8_blended(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
    sc: &StorageColor,
    eq: BlendEquation,
    (form, part): (Form, SValue),
) -> BResult<SValue> {
    let math = extended_range::blend::BlendMath::of(cx.entry.compile.fast_math);
    let (pf, written) = (sc.view.pixel_format, written(sc.view.kept));
    let arith = match (form.lanes, form.float_fmt()) {
        (4, Some(FloatFmt::Half)) => extended_range::blend::BlendArith::Half,
        (4, Some(FloatFmt::F32)) => extended_range::blend::BlendArith::Float,
        _ => return Err(cx.fail("an 8-bit unorm blend of another output")),
    };
    let Ok(b) = direct_blend(&eq, pf, written, (math, arith)) else {
        return Err(cx.fail("an 8-bit unorm blend outside Metal's model"));
    };
    let texel = read_texel(cx, fb, color, sc)?;
    let constant = match b.reads_constant() {
        true => {
            let field = cx.push_field(fb, PushField::HalfBlendColor)?;
            Some(fb.composite_extract(field, &[0])?)
        }
        false => None,
    };
    extended_range::blend::blend_unorm8(fb, &b, arith, (part, texel), constant)
}

/// The four-lane output `part` of `form` (a `half4` output's `half`s, or a `float4` one's)
/// blended into 32-bit float attachment `color` by Metal's blend `eq`
/// (`extended_range::Blend::float32`, of a module with fast math): the texel read first at the
/// invocation's pixel (and sample), inside the interlock, as [`merge`] reads it, and Metal's
/// arithmetic of the output over it in `float` throughout
/// (`extended_range::blend::blend_float32`; the blend color from `PushField::HalfBlendColor`
/// where the blend reads it), the `float` the attachment stores. Nothing is clamped and nothing is
/// converted after: the storage image's store holds the `f32` the blend gives. The channels a
/// write mask keeps are [`merge`]'s to take back from the texel.
fn float32_blended(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
    sc: &StorageColor,
    eq: BlendEquation,
    (form, part): (Form, SValue),
) -> BResult<SValue> {
    let (pf, written) = (sc.view.pixel_format, written(sc.view.kept));
    if !matches!(form.float_fmt(), Some(FloatFmt::Half | FloatFmt::F32)) || form.lanes != 4 {
        return Err(cx.fail("a 32-bit float blend of another output"));
    }
    let Ok(b) = extended_range::Blend::float32(&eq, pf, written) else {
        return Err(cx.fail("a 32-bit float blend outside Metal's model"));
    };
    let texel = read_texel(cx, fb, color, sc)?;
    let d = as_held(fb, texel, sc.view.numeric)?;
    let constant = match b.reads_constant() {
        true => {
            let field = cx.push_field(fb, PushField::HalfBlendColor)?;
            Some(fb.composite_extract(field, &[0])?)
        }
        false => None,
    };
    extended_range::blend::blend_float32(fb, &b, (part, d), constant)
}

/// The output `part` of `form` (one to four `f32` lanes: a `half` output's `half`s, or a `float`
/// one's `float`s) blended into attachment `color` by the generic float blend `eq`
/// ([`StorageView::generic_blend`], `extended_range::Blend::generic`): the texel read first at the
/// invocation's pixel (and sample), inside the interlock, in Metal's channel order
/// ([`read_texel`]), and the blend of the output's four lanes (a missing one zero) over it in
/// `float`, each factor, product and sum rounded, the source, the second source (`second`, the
/// function's second output, where the blend reads it) and the blend color (from
/// `PushField::HalfBlendColor` where the blend reads it) clamped first to a normalized format's
/// range, `[0, 1]` or `[-1, 1]` (`extended_range::blend::blend_generic`). The result keeps the
/// output's lanes; the device's conversion stores it, and the channels a write mask keeps are
/// [`merge`]'s to take back from the texel. Metal's arithmetic for the format is not built: the
/// function notes `PrecisionNote::UnmeasuredFetchBlend`.
fn generic_blended(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
    sc: &StorageColor,
    eq: BlendEquation,
    (form, part): (Form, SValue),
    second: Option<(Form, SValue)>,
) -> BResult<SValue> {
    if !matches!(form.float_fmt(), Some(FloatFmt::Half | FloatFmt::F32)) {
        return Err(cx.fail("a generic blend of another output"));
    }
    let Ok(b) = generic(&eq, sc.view.pixel_format, written(sc.view.kept)) else {
        return Err(cx.fail("a generic blend outside its forms"));
    };
    let s1 = match (b.reads_second(), second) {
        (true, Some((form, s1)))
            if matches!(form.float_fmt(), Some(FloatFmt::Half | FloatFmt::F32)) =>
        {
            Some(four_lanes(fb, form, s1)?)
        }
        (true, _) => return Err(cx.fail("a blend of the second source without its output")),
        (false, _) => None,
    };
    let texel = read_texel(cx, fb, color, sc)?;
    let d = as_held(fb, texel, sc.view.numeric)?;
    let constant = match b.reads_constant() {
        true => {
            let field = cx.push_field(fb, PushField::HalfBlendColor)?;
            Some(fb.composite_extract(field, &[0])?)
        }
        false => None,
    };
    let range = match sc.view.format.block().and_then(|b| b.color) {
        Some(vk::NumericFormat::Unorm) => Some((0.0, 1.0)),
        Some(vk::NumericFormat::Snorm) => Some((-1.0, 1.0)),
        _ => None,
    };
    let s = four_lanes(fb, form, part)?;
    let v = extended_range::blend::blend_generic(fb, &b, range, (s, s1, d), constant)?;
    output_lanes(fb, form, v)
}

/// Output `part` of `form` (one to four `f32` lanes) as the four lanes Metal's blend reads: its
/// own, then zeros.
fn four_lanes(fb: &mut FunctionBuilder<'_>, form: Form, part: SValue) -> BResult<SValue> {
    if form.lanes == 4 {
        return Ok(part);
    }
    let f32_ = scalar_type(fb.module(), SampledType::Float)?;
    let vec4 = fb.module().ty(SType::Vector { elem: f32_, len: 4 })?;
    let f32_form = Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 1,
    };
    let zero = ops::fsplat(fb, f32_form, 0.0)?;
    let mut lanes = Vec::with_capacity(4);
    for lane in 0..4u32 {
        lanes.push(match lane {
            0 if form.lanes == 1 => part,
            l if l < u32::from(form.lanes) => fb.composite_extract(part, &[l])?,
            _ => zero,
        });
    }
    fb.composite_construct(vec4, &lanes)
}

/// The lanes of `form` (one to four) of the four `f32` lanes `v`.
fn output_lanes(fb: &mut FunctionBuilder<'_>, form: Form, v: SValue) -> BResult<SValue> {
    match form.lanes {
        4 => Ok(v),
        1 => fb.composite_extract(v, &[0]),
        n => {
            let lanes: Vec<u32> = (0..u32::from(n)).collect();
            fb.vector_shuffle(v, v, &lanes)
        }
    }
}

/// Whether the function writes color attachment `color` through the storage form under a blend
/// the shader computes of a format held directly, Metal's 16-bit float, 8-bit unorm or 8-bit sRGB
/// blend, so that the write stores the value the blend gives, whatever the output and the
/// device's conversion (`exact`: and that blend is Metal's exactly, [`StorageColor::exact_blend`]).
/// Not the generic blend ([`StorageView::generic_blend`]), whose result the device's conversion
/// stores as it stores an output.
pub(super) fn blends_in_shader(cx: &SelectCx<'_>, color: u32, exact: bool) -> bool {
    cx.families.fragment.fetch.get(&color).is_some_and(|sc| {
        sc.view.blend.is_some()
            && sc.view.extended.is_none()
            && !sc.view.generic_blend
            && (sc.exact_blend || !exact)
    })
}

/// Whether the function writes color attachment `color` through the storage form under Metal's
/// 16-bit float blend, which the shader computes ([`blends_in_shader`]), and so stores `half`s.
pub(super) fn blends_half(cx: &SelectCx<'_>, color: u32) -> bool {
    blends_in_shader(cx, color, false)
        && cx
            .families
            .fragment
            .fetch
            .get(&color)
            .is_some_and(|sc| half_float_format(sc.view.pixel_format))
}

/// Output `part` (of `form`, in its lowered representation) of storage-form attachment `color`
/// whose write keeps some channels (`StorageView::kept`): each kept channel's lane taken from the
/// texel the write overwrites, read first at the invocation's pixel (and sample), as `value` reads
/// it. Inside the interlock no other fragment of the pixel writes the texel between the read and
/// the write, so the write stores the kept channels as they were (in the device's decode of the
/// texel: a `half` NaN's bits are the device's, `PrecisionNote::DeviceNanEncoding`).
fn merge(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
    sc: &StorageColor,
    form: Form,
    part: SValue,
) -> BResult<SValue> {
    let texel = read_texel(cx, fb, color, sc)?;
    let texel = as_held(fb, texel, sc.view.numeric)?;
    let n = u32::from(form.lanes);
    if n < 2 {
        return Err(cx.fail("a merged storage write of one lane"));
    }
    let channels = [
        vk::ColorComponentFlags::R,
        vk::ColorComponentFlags::G,
        vk::ColorComponentFlags::B,
        vk::ColorComponentFlags::A,
    ];
    // The kept channels the draw's word names: bit `c` of the color's four takes channel `c`
    // from the texel.
    if sc.view.kept_at_draw {
        let word = cx.push_field(fb, PushField::StorageKeptChannels)?;
        let shift = fb.u32_const(4 * color)?;
        let shift = fb.module().const_value(shift)?;
        let bits = fb.binary(BinOp::ShiftRightLogical, word, shift)?;
        let zero = fb.u32_const(0)?;
        let zero = fb.module().const_value(zero)?;
        let mut keeps = Vec::with_capacity(n as usize);
        for lane in 0..n {
            let bit = fb.u32_const(1 << lane)?;
            let bit = fb.module().const_value(bit)?;
            let set = fb.binary(BinOp::BitwiseAnd, bits, bit)?;
            keeps.push(fb.binary(BinOp::INotEqual, set, zero)?);
        }
        let bvec = fb.bool_ty(form.lanes)?;
        let keep = fb.composite_construct(bvec, &keeps)?;
        // The texel's lanes the output has.
        let lanes: Vec<u32> = (0..n).collect();
        let texel = fb.vector_shuffle(texel, texel, &lanes)?;
        return fb.select(keep, texel, part);
    }
    let lanes: Vec<u32> = (0..n)
        .zip(channels)
        .map(|(lane, c)| {
            if sc.view.kept.contains(c) {
                n + lane
            } else {
                lane
            }
        })
        .collect();
    fb.vector_shuffle(part, texel, &lanes)
}

/// The sources Metal's blend `b` of an extended-range attachment reads, from the function's
/// output `part` (of `form`) and its second (index-1) output `second` where the blend reads the
/// second source: a `float4` output rounded to `half`, nearest even, first, but where the blend's
/// own model reads it unrounded, and the second source, a `float4` one rounded to `half` first, as
/// the first under the rounded model.
pub(super) fn blend_sources(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: &extended_range::Blend,
    (form, part): (Form, SValue),
    second: Option<(Form, SValue)>,
) -> BResult<(SValue, Option<SValue>)> {
    let part = match form.float_fmt() {
        Some(FloatFmt::F32) if !b.float => extended_range::blend::half_of(fb, form, part)?,
        _ => part,
    };
    let s1 = match (b.reads_second(), second) {
        (true, Some((form, s1))) => Some(match form.float_fmt() {
            Some(FloatFmt::F32) => extended_range::blend::half_of(fb, form, s1)?,
            _ => s1,
        }),
        (true, None) => return Err(cx.fail("a blend of the second source without its output")),
        (false, _) => None,
    };
    Ok((part, s1))
}

/// The blend color words Metal's blend `b` reads (`PushField::BlendColor`, the clamped color
/// and its row, `blend_color::constant_words`), where it reads the blend color.
pub(super) fn blend_constant(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: &extended_range::Blend,
) -> BResult<Option<[SValue; 2]>> {
    if !b.reads_constant() {
        return Ok(None);
    }
    let field = cx.push_field(fb, PushField::BlendColor)?;
    let c = fb.composite_extract(field, &[0])?;
    let m = fb.composite_extract(field, &[1])?;
    Ok(Some([c, m]))
}

/// The raw texel a write of output `part` (of `form`: `half4`, or `float4` under a blend that
/// takes it, rounded to `half`, nearest even, first) to extended-range attachment `color` (held in
/// `range`) stores under the pipeline's blend `eq`: the texel read first at the invocation's pixel
/// (and sample), as [`merge`] reads it, inside the interlock, its tile value
/// (`extended_range::tile_value`), Metal's blend of the output over that
/// (`extended_range::blend::blend`, the second source from `second`, the function's index-1 output,
/// where it reads it; the blend color from `PushField::BlendColor` where it reads it), and the
/// kernel write's encode of the result, the channels a write mask keeps taken from the texel as
/// Metal keeps them (`extended_range::encode_keeping`).
///
/// Under the instance's shadow (`StorageColor::shadowed`) the destination is the shadow's texel,
/// Metal's tile value, and the write stores the blend's result there too (the second texel), the
/// kept channels as the shadow held them; the codes texel is read only to keep channels.
fn blended(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
    sc: &StorageColor,
    range: ExtendedRange,
    (form, part): (Form, SValue),
    second: Option<(Form, SValue)>,
) -> BResult<(SValue, Option<SValue>)> {
    let output = (form.lanes == 4).then(|| form.float_fmt()).flatten();
    let mode = extended_range::blend::BlendMath::of(cx.entry.compile.fast_math);
    let Ok(Some(b)) = blend_of(&sc.view, (output, mode)) else {
        return Err(cx.fail("an extended-range blend outside Metal's model"));
    };
    let (part, s1) = blend_sources(cx, fb, &b, (form, part), second)?;
    let table = crate::lower::texture::extended_table(cx, range)?;
    let mut codes = None;
    let mut read_codes = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>| -> BResult<SValue> {
        if let Some(texel) = codes {
            return Ok(texel);
        }
        let texel = read_texel(cx, fb, color, sc)?;
        codes = Some(texel);
        Ok(texel)
    };
    let shadow = match sc.shadowed {
        true => Some(shadow_read(cx, fb, color, range)?),
        false => None,
    };
    let d = match shadow {
        Some(held) => held,
        None => {
            let texel = read_codes(cx, fb)?;
            extended_range::tile_value(fb, range, texel, table)?
        }
    };
    let constant = blend_constant(cx, fb, &b)?;
    let v = extended_range::blend::blend(fb, range, &b, (part, s1, d), constant)?;
    if !sc.view.merges() {
        let texel = extended_range::encode(fb, range, v, table)?;
        return Ok((texel, shadow.map(|_| v)));
    }
    let kept = [
        vk::ColorComponentFlags::R,
        vk::ColorComponentFlags::G,
        vk::ColorComponentFlags::B,
        vk::ColorComponentFlags::A,
    ]
    .map(|c| sc.view.kept.contains(c));
    let texel = read_codes(cx, fb)?;
    let stored = extended_range::encode_keeping(fb, range, v, table, (texel, kept))?;
    let held = match shadow {
        // The shadow keeps the channels the codes keep, as the tile held them.
        Some(held) => {
            let lanes: Vec<u32> = kept
                .iter()
                .zip(0..4u32)
                .map(|(&kept, lane)| match kept {
                    true => 4 + lane,
                    false => lane,
                })
                .collect();
            Some(fb.vector_shuffle(v, held, &lanes)?)
        }
        None => None,
    };
    Ok((stored, held))
}

/// An upper estimate of the SPIR-V words of an sRGB texel's conversion in direction `dir`
/// ([`srgb_decode`], [`srgb_texel`]'s encode).
fn srgb_words(dir: PackDir) -> u32 {
    let f4 = Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 4,
    };
    pack::words(PackFormat::Unorm4x8Srgb, dir, f4)
}

/// The bytes of a texel of the sRGB format `format` held in one `u32` (`word`, the texel's bytes
/// in memory order), in red, green, blue, alpha order, as `arith::pack`'s `pack` and `unpack` hold
/// `unorm4x8_srgb`'s: `B8G8R8A8_SRGB`'s blue and red bytes swapped, `R8G8B8A8_SRGB`'s as they
/// are. The swap is its own inverse.
fn rgba_order(fb: &mut FunctionBuilder<'_>, format: vk::Format, word: SValue) -> BResult<SValue> {
    if format != vk::Format::B8G8R8A8_SRGB {
        return Ok(word);
    }
    let k = |fb: &mut FunctionBuilder<'_>, v: u32| -> BResult<SValue> {
        let c = fb.u32_const(v)?;
        fb.module().const_value(c)
    };
    let (ga, low, sixteen) = (k(fb, 0xff00_ff00)?, k(fb, 0xff)?, k(fb, 16)?);
    let kept = fb.binary(BinOp::BitwiseAnd, word, ga)?;
    let down = fb.binary(BinOp::ShiftRightLogical, word, sixteen)?;
    let down = fb.binary(BinOp::BitwiseAnd, down, low)?;
    let up = fb.binary(BinOp::BitwiseAnd, word, low)?;
    let up = fb.binary(BinOp::ShiftLeftLogical, up, sixteen)?;
    let swapped = fb.binary(BinOp::BitwiseOr, kept, down)?;
    fb.binary(BinOp::BitwiseOr, swapped, up)
}

/// The `vec4` a fetch reads from the sRGB texel `word` of `format`: the device's decode
/// (`arith::pack::unpack` of `unorm4x8_srgb`, `PrecisionNote::DevicePackConversion`): each byte
/// by the 8-bit instruction, the color channels through the sRGB transfer by the device's `Pow`,
/// a one- or two-channel format's missing channels 0 and its alpha 1 (its word's bytes are its
/// channels', the rest zero).
fn srgb_decode(fb: &mut FunctionBuilder<'_>, format: vk::Format, word: SValue) -> BResult<SValue> {
    let word = match format {
        vk::Format::R8_SRGB | vk::Format::R8G8_SRGB => {
            let alpha = fb.u32_const(0xff00_0000)?;
            let alpha = fb.module().const_value(alpha)?;
            fb.binary(BinOp::BitwiseOr, word, alpha)?
        }
        _ => word,
    };
    let rgba = rgba_order(fb, format, word)?;
    let f4 = Form {
        held: Held::Float(crate::lower::arith::FloatFmt::F32),
        lanes: 4,
    };
    pack::unpack(fb, PackFormat::Unorm4x8Srgb, f4, rgba)
}

/// The `uvec4` texel (its first lane the texel's word, in the view's `R8_UINT`, `R16_UINT` or
/// `R32_UINT`) a write of output `part` (one to four `float` or `half` lanes of `form`, a missing
/// lane 0) to the sRGB attachment `color` stores: the device's encode
/// (`arith::pack::pack` of `unorm4x8_srgb`, `PrecisionNote::DevicePackConversion`: the color
/// channels through the sRGB transfer by the device's `Pow`, then the 8-bit instruction), its
/// bytes in `format`'s order; a write that keeps channels takes their bytes from the texel read
/// first at the invocation's pixel (and sample), inside the interlock, so they stay as they were,
/// bit for bit. Under a blend (`StorageView::blend`) the encode is of Metal's blend of the output
/// over that texel, decoded as a fetch reads it ([`srgb_decode`]), in the output's type, the second
/// source `second` (the function's second output) where the blend reads one, and the blend color
/// from `PushField::HalfBlendColor` where it reads that (`extended_range::Blend::srgb8`,
/// `extended_range::blend::blend_srgb8`).
fn srgb_texel(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
    sc: &StorageColor,
    (form, part): (Form, SValue),
    second: Option<(Form, SValue)>,
) -> BResult<SValue> {
    if form
        .float_fmt()
        .map(crate::lower::arith::FloatFmt::container)
        != Some(32)
    {
        return Err(cx.fail("an sRGB storage write of another output than float lanes"));
    }
    // The output's four lanes, a missing one 0, which the encode packs and the word cuts at the
    // format's bytes.
    let part = four_lanes(fb, form, part)?;
    let form = Form { lanes: 4, ..form };
    // The texel the write reads first: the blend's destination, and the bytes a mask keeps.
    let old = match sc.view.blend.is_some() || sc.view.merges() {
        true => {
            let texel = read_texel(cx, fb, color, sc)?;
            Some(fb.composite_extract(texel, &[0])?)
        }
        false => None,
    };
    let part = match (sc.view.blend, old) {
        (Some(eq), Some(old)) => {
            let arith = match form.float_fmt() {
                Some(FloatFmt::Half) => extended_range::blend::BlendArith::Half,
                _ => extended_range::blend::BlendArith::Float,
            };
            let math = extended_range::blend::BlendMath::of(cx.entry.compile.fast_math);
            let (pf, written) = (sc.view.pixel_format, written(sc.view.kept));
            let Ok((b, _)) = extended_range::Blend::srgb8(&eq, pf, written, (math, arith)) else {
                return Err(cx.fail("an sRGB blend outside its forms"));
            };
            let s1 = match (b.reads_second(), second) {
                (true, Some((form, s1)))
                    if form.lanes == 4
                        && matches!(form.float_fmt(), Some(FloatFmt::Half | FloatFmt::F32)) =>
                {
                    Some(s1)
                }
                (true, _) => return Err(cx.fail("a blend of the second source without its output")),
                (false, _) => None,
            };
            let decoded = srgb_decode(fb, sc.view.format, old)?;
            let constant = match b.reads_constant() {
                true => {
                    let field = cx.push_field(fb, PushField::HalfBlendColor)?;
                    Some(fb.composite_extract(field, &[0])?)
                }
                false => None,
            };
            let parts = (part, s1, decoded);
            extended_range::blend::blend_srgb8(fb, &b, arith, parts, constant)?
        }
        (Some(_), None) => return Err(cx.fail("an sRGB blend without its texel")),
        (None, _) => part,
    };
    let f4 = form.with(Held::Float(crate::lower::arith::FloatFmt::F32));
    let rgba = pack::pack(fb, PackFormat::Unorm4x8Srgb, f4, part)?;
    let mut word = rgba_order(fb, sc.view.format, rgba)?;
    // A one- or two-channel format's word is its channels' bytes alone: the view's integer format
    // holds no other.
    let bytes = match sc.view.format {
        vk::Format::R8_SRGB => Some(0xffu32),
        vk::Format::R8G8_SRGB => Some(0xffff),
        _ => None,
    };
    if let Some(bytes) = bytes {
        let bytes = fb.u32_const(bytes)?;
        let bytes = fb.module().const_value(bytes)?;
        word = fb.binary(BinOp::BitwiseAnd, word, bytes)?;
    }
    if sc.view.merges() {
        // The kept channels' bytes, in memory order.
        let bytes = match sc.view.format {
            vk::Format::B8G8R8A8_SRGB => [16u32, 8, 0, 24],
            _ => [0u32, 8, 16, 24],
        };
        let channels = [
            vk::ColorComponentFlags::R,
            vk::ColorComponentFlags::G,
            vk::ColorComponentFlags::B,
            vk::ColorComponentFlags::A,
        ];
        let mask = channels
            .iter()
            .zip(bytes)
            .filter(|(c, _)| sc.view.kept.contains(**c))
            .fold(0u32, |m, (_, at)| m | 0xff << at);
        let Some(old) = old else {
            return Err(cx.fail("an sRGB merge without its texel"));
        };
        let keep = fb.u32_const(mask)?;
        let keep = fb.module().const_value(keep)?;
        let write = fb.u32_const(!mask)?;
        let write = fb.module().const_value(write)?;
        let old = fb.binary(BinOp::BitwiseAnd, old, keep)?;
        let new = fb.binary(BinOp::BitwiseAnd, word, write)?;
        word = fb.binary(BinOp::BitwiseOr, old, new)?;
    }
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let uvec4 = fb.module().ty(SType::Vector { elem: u32_, len: 4 })?;
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    fb.composite_construct(uvec4, &[word, zero, zero, zero])
}

/// The `uvec4` texel that writes output `part` of `form` (`float` or `half` lanes) to word
/// attachment `color` (`StorageView::word`): Metal's word of the output
/// (`stage_io::word_color::encode`), or under a blend (`StorageView::blend`) of Metal's blend of
/// the four-lane output over the word read first at the pixel, inside the interlock, as Metal's
/// blend reads it (`stage_io::word_color::decode`, `extended_range::blend::blend_packed16`, the
/// blend color from `PushField::HalfBlendColor` where the blend reads it); and where the write
/// keeps some of the format's channels, the fields of those channels taken from the word read
/// first, so they keep their codes as Metal's tile keeps them. The kept channels are the draw's
/// (`PushField::StorageKeptChannels`) where the write takes them from it
/// (`StorageView::kept_at_draw`), else the view's.
fn word_texel(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
    sc: &StorageColor,
    form: Form,
    part: SValue,
) -> BResult<SValue> {
    let Some(word) = sc.view.word else {
        return Err(cx.fail("a word texel of an attachment without a word"));
    };
    // The word read first, where the blend reads it.
    let mut read = None;
    let mut value = match sc.view.blend {
        Some(eq) => {
            if form.lanes != 4 {
                return Err(cx.fail("a packed 16-bit blend of another output"));
            }
            let math = extended_range::blend::BlendMath::of(cx.entry.compile.fast_math);
            let Ok(b) = extended_range::Blend::packed16(&eq, written(sc.view.kept), math) else {
                return Err(cx.fail("a packed 16-bit blend outside Metal's model"));
            };
            let texel = read_texel(cx, fb, color, sc)?;
            let old = fb.composite_extract(texel, &[0])?;
            read = Some(old);
            let d = stage_io::word_color::decode(fb, word, old)?;
            let constant = match b.reads_constant() {
                true => {
                    let field = cx.push_field(fb, PushField::HalfBlendColor)?;
                    Some(fb.composite_extract(field, &[0])?)
                }
                false => None,
            };
            let v = extended_range::blend::blend_packed16(fb, &b, (part, d), constant)?;
            let f4 = form.with(Held::Float(FloatFmt::F32));
            stage_io::word_color::encode(fb, word, (f4, v))?
        }
        None => stage_io::word_color::encode(fb, word, (form, part))?,
    };
    if sc.view.merges() {
        let channels = [
            vk::ColorComponentFlags::R,
            vk::ColorComponentFlags::G,
            vk::ColorComponentFlags::B,
            vk::ColorComponentFlags::A,
        ];
        let field_bits = |f: crate::metal::format::WordField| ((1u32 << f.bits) - 1) << f.shift;
        let keep = match sc.view.kept_at_draw {
            // Bit `c` of the color's four in the draw's word keeps channel `c`'s field.
            true => {
                let draw = cx.push_field(fb, PushField::StorageKeptChannels)?;
                let shift = fb.u32_const(4 * color)?;
                let shift = fb.module().const_value(shift)?;
                let bits = fb.binary(BinOp::ShiftRightLogical, draw, shift)?;
                let zero = fb.u32_const(0)?;
                let zero = fb.module().const_value(zero)?;
                let mut keep = zero;
                for (lane, field) in word.fields().into_iter().enumerate() {
                    let Some(field) = field else {
                        continue;
                    };
                    let bit = fb.u32_const(1 << lane)?;
                    let bit = fb.module().const_value(bit)?;
                    let set = fb.binary(BinOp::BitwiseAnd, bits, bit)?;
                    let set = fb.binary(BinOp::INotEqual, set, zero)?;
                    let mask = fb.u32_const(field_bits(field))?;
                    let mask = fb.module().const_value(mask)?;
                    let mask = fb.select(set, mask, zero)?;
                    keep = fb.binary(BinOp::BitwiseOr, keep, mask)?;
                }
                keep
            }
            false => {
                let mask = channels
                    .iter()
                    .zip(word.fields())
                    .filter(|(c, _)| sc.view.kept.contains(**c))
                    .filter_map(|(_, f)| f)
                    .fold(0u32, |m, f| m | field_bits(f));
                let mask = fb.u32_const(mask)?;
                fb.module().const_value(mask)?
            }
        };
        let old = match read {
            Some(old) => old,
            None => {
                let texel = read_texel(cx, fb, color, sc)?;
                fb.composite_extract(texel, &[0])?
            }
        };
        let old = fb.binary(BinOp::BitwiseAnd, old, keep)?;
        let write = fb.unary(UnOp::Not, keep)?;
        let new = fb.binary(BinOp::BitwiseAnd, value, write)?;
        value = fb.binary(BinOp::BitwiseOr, old, new)?;
    }
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let uvec4 = fb.module().ty(SType::Vector { elem: u32_, len: 4 })?;
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    fb.composite_construct(uvec4, &[value, zero, zero, zero])
}

/// The blend an extended-range attachment's write computes in the shader
/// (`StorageView::blend`) of an output of four lanes of `output` (`None` for another output)
/// of a function of math `math`, as Metal's model holds it: `None` for no blend; `Err` with the gap
/// (`Feature::ExtendedRangeBlend`) where the model does not hold it ([`extended_range::Blend`]).
fn blend_of(
    view: &StorageView,
    (output, math): (Option<FloatFmt>, extended_range::blend::BlendMath),
) -> Result<Option<extended_range::Blend>, Error> {
    let (Some(eq), Some(form)) = (view.blend, view.extended) else {
        return Ok(None);
    };
    extended_range::Blend::for_output(&eq, form, written(view.kept), (output, math)).map(Some)
}

/// The blend math of the function `cx` selects ([`extended_range::blend::BlendMath`]).
pub(super) fn blend_math(cx: &SelectCx<'_>) -> extended_range::blend::BlendMath {
    extended_range::blend::BlendMath::of(cx.entry.compile.fast_math)
}

/// The channels a write that keeps `kept` stores.
pub(super) fn written(kept: vk::ColorComponentFlags) -> ColorWriteMask {
    [
        (vk::ColorComponentFlags::R, ColorWriteMask::RED),
        (vk::ColorComponentFlags::G, ColorWriteMask::GREEN),
        (vk::ColorComponentFlags::B, ColorWriteMask::BLUE),
        (vk::ColorComponentFlags::A, ColorWriteMask::ALPHA),
    ]
    .into_iter()
    .filter(|(k, _)| !kept.contains(*k))
    .fold(ColorWriteMask::NONE, |m, (_, c)| m.union(c))
}

/// Whether returned value `x`'s `[[sample_mask]]` output, in field `field`, sets the bit of the
/// invocation's own sample: `((mask >> SampleId) & 1) != 0`. Under sample shading an invocation
/// shades its one sample, and Metal keeps its write exactly when the mask sets that sample's bit.
/// At one sample, where no `SampleId` is planned, bit 0 (Metal stores no color without it).
fn sample_bit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    x: SValue,
    field: u32,
) -> BResult<SValue> {
    let mask = match cx.plan.families.stage_io.returned {
        Returned::Struct => fb.composite_extract(x, &[field])?,
        Returned::Value | Returned::Void => {
            return Err(cx.fail("a returned sample mask outside a returned struct"))
        }
    };
    let sample = match sample(cx, fb)? {
        Some(sample) => sample,
        None => {
            let zero = fb.u32_const(0)?;
            fb.module().const_value(zero)?
        }
    };
    let shifted = fb.binary(BinOp::ShiftRightLogical, mask, sample)?;
    let one = fb.u32_const(1)?;
    let one = fb.module().const_value(one)?;
    let bit = fb.binary(BinOp::BitwiseAnd, shifted, one)?;
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    fb.binary(BinOp::INotEqual, bit, zero)
}

/// The `vec4` texel of the attachment's sampled type that writes output `x` of `form`: its lanes
/// then zeros (lanes past the format's channels are ignored), a signed-integer one's bits as
/// `int`.
fn texel_of(
    fb: &mut FunctionBuilder<'_>,
    x: SValue,
    form: Form,
    numeric: SampledType,
) -> BResult<SValue> {
    let elem = scalar_type(
        fb.module(),
        if form.is_float() {
            SampledType::Float
        } else {
            SampledType::Uint
        },
    )?;
    let vec4 = fb.module().ty(SType::Vector { elem, len: 4 })?;
    let texel = if form.lanes == 4 {
        x
    } else {
        let zero = ops::splat(
            fb,
            Form {
                held: form.held,
                lanes: 1,
            },
            0,
        )?;
        let mut parts = Vec::with_capacity(4);
        for lane in 0..4u32 {
            parts.push(match lane {
                0 if form.lanes == 1 => x,
                l if l < u32::from(form.lanes) => fb.composite_extract(x, &[l])?,
                _ => zero,
            });
        }
        fb.composite_construct(vec4, &parts)?
    };
    if numeric != SampledType::Sint {
        return Ok(texel);
    }
    let i32_ = scalar_type(fb.module(), SampledType::Sint)?;
    let ivec4 = fb.module().ty(SType::Vector { elem: i32_, len: 4 })?;
    fb.convert(ConvOp::Bitcast, texel, ivec4)
}

#[cfg(test)]
mod tests;
