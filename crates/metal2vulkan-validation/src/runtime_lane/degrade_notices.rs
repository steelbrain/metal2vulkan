//! Device tests of degrade notices on the device the cfg selects, under the validation layer. Each
//! degradation the runtime takes is announced once per device and kind through the device's message
//! sink, however often the device takes it, and its notice decodes by its kind: the code is its
//! `id_name`, the payload its `degradation`.
//!
//! Every kind of the closed vocabulary (`Degradation::ALL`) has a site a lane test here, in
//! `host_bytes`, in `framebuffer_fetch`, in `presenting`, in `shader_stencil`, in
//! `tile_dispatches`, in `fragment_imageblocks`, in `gpu_times`, in `resource_ids`, in
//! `command_queues`, in
//! `absent_reads`, in
//! `indirect_host_steps`, in `indirect_dispatches`, in `index_fetches`, in `texture_blits`, in
//! `sparse_textures`, in `buffer_texture_blits`, in `amplified_draws`, in `color_writes`, in
//! `binary_archives`, in `view_reinterpretation`, in `mipmaps_exact`, in `clear_reach`, in
//! `resolves`, in `mesh_draws`, in `provoking`, in `provoking_mesh`, in `visibility`, in
//! `pass_start_reads`, in `unorm_depth_bias`, in `pvrtc_textures`, in `heaps`, in `rate_maps`, in
//! `rebind_reach`, in `predication` or in `blits` takes, each test checking that
//! its kinds are announced exactly once: the buffers over
//! host bytes ([`HOST_BYTES`]), blits ([`BLITS`]),
//! dispatches ([`DISPATCHES`]), render passes ([`PASSES`]), framebuffer fetch's passes
//! ([`FETCHES`]), its targets' storage forms ([`STORAGE_FORMS`]) and its sample planes
//! ([`PLANES`]), textures over the caller's images
//! ([`WRAPPED`]), the shader-side stencil ([`SHADOW`]), tile dispatches ([`TILE_DISPATCHES`]),
//! fragment imageblocks ([`FRAGMENT_IMAGEBLOCKS`]), command buffers' GPU times ([`GPU_TIMES`],
//! [`UNTIMED`], [`FROM_HOST`]), color fills packed where fills ignore the color
//! ([`UNPACKED_FILL`]), texture buffers' fills reading bytes past their pattern's copies in
//! `texture_buffer_fills` ([`TEXTURE_BUFFER_FILL`]), draws of points no state sizes in
//! `topology_classes` ([`UNSIZED_POINTS`]),
//! resource IDs ([`UNLISTED`]),
//! indirect command buffers' commands naming a released state ([`RELEASED_STATES`]) and their host
//! steps ([`HOST_STEPS`]), indirect dispatches' host reads ([`INDIRECT_DISPATCHES`]), indexed draws
//! past their index buffer ([`INDEX_FETCHES`]), depth twins ([`TWINS`]) and uploads into depth
//! images ([`UPLOADS`]), a MoltenVK whose
//! argument-buffer state is unread ([`ARGUMENT_BUFFERS`]), a device answering at a Metal level
//! below its host's macOS release here ([`METAL_LEVELS`]), mipmaps within Metal's
//! spread ([`MIPMAPS`]), a heap's depth or stencil texture laid apart from the heap ([`APART`]),
//! a multisample depth or stencil texture without GPU-optimized contents that stays optimized
//! ([`OPTIMIZED`]),
//! a heap's colour texture on a driver whose compressed images hide aliased writes ([`ALIASES`]),
//! one Metal compresses that keeps the driver's compression there ([`COMPRESSED`]),
//! sparse textures on a device without strict residency ([`SPARSE`]), overlapping copies between textures over
//! buffers ([`OVERLAPS`]), commands writing a shadowed texture's bytes two ways ([`MERGES`]),
//! `Shared` buffers in system memory on a device without unified memory ([`SYSTEM_MEMORY`]),
//! amplified draws drawn one amplification at a time ([`AMPLIFICATION`]), render pipeline states
//! whose fragment rounds its color writes in `color_writes` ([`COLOR_WRITES`]), render pipeline
//! states whose stages an archive holds from two adds in `binary_archives` ([`ARCHIVES`]),
//! texture views of another component layout shadowed in `view_reinterpretation`
//! ([`VIEW_SHADOWS`]), mipmaps computed as an Apple M4's where an Apple M3 Max's differ in
//! `mipmaps_exact` ([`M3_MIPMAPS`]) and mipmaps at a rounding the chain does not fit there
//! ([`TIE_MIPMAPS`]), executions of an indirect command buffer from one command
//! with two lengths in `visibility` ([`REPEATED_STARTS`]), resources made with Metal's private
//! option bit here ([`PRIVATE_OPTION`]), indexed indirect draws of bytes a draw wrote with no
//! barrier in `indirect_draws` ([`UNORDERED_DRAWS`]), pipeline states whose functions read an
//! argument their function constants removed in `absent_reads` ([`ABSENT_READS`]), a slot's
//! neutral bound for a texture of another numeric class in `cross_class_reads` ([`CROSS_CLASS`]),
//! passes that
//! draw over an attachment loaded with `DontCare` in `clear_reach` ([`DONT_CARE_LOADS`]), passes
//! that draw and load a rotated texture of rotation 4 in `rotated_textures` ([`ROTATED_LOADS`]),
//! draws over a rate map of no layers made after a map of layers was released in
//! `rate_mapped_passes` ([`ZERO_LAYER_AFTER_RELEASE`]), and passes over a map reset after they took
//! it that the runtime cannot encode again there ([`RESET_UNENCODED`]),
//! IOSurface planes under 16 bytes a row in `buffer_texture_blits` ([`NARROW_ROWS`]), one
//! texture's depth stored and resolved beside its stencil resolved alone in `resolves` ([`MIXED_RESOLVES`]), mesh
//! draws emulated on a device without `VK_EXT_mesh_shader` in `mesh_draws` ([`EMULATED_MESH`]),
//! draws under Apple's last provoking vertex on a device without its own in `provoking` and
//! `provoking_mesh` ([`PROVOKING`]),
//! render pipeline states whose draws run a synthesized geometry stage in `barycentric_weights`
//! ([`SYNTHESIZED_GEOMETRY`]), tessellated draws emulated on a device without
//! `tessellationShader` in `patches` ([`EMULATED_TESSELLATION`]), render and compute pipeline
//! states whose exact forms a stage past the device's compiler word bound dropped here
//! ([`EXACT_FORMS_OVER_LIMIT`]),
//! vertex-stage reads in place of bytes a fragment stage of the pass wrote in `pass_start_reads`
//! ([`IN_PLACE_VERTEX_READS`]) and a vertex stage's write of bytes one wrote there
//! ([`FRAGMENTS_BEFORE_GEOMETRY`]), constant biases of 16-bit unorm depths the device converts
//! past a fragment stage that tests first, where the profile is not measured to store Metal's
//! biased codes, in `unorm_depth_bias` ([`UNORM_DEPTH_BIAS`]), 16-bit unorm depths the device converts without the profile's
//! deviation ([`UNORM_DEPTH_ROUNDING`]) and those the fragment stage writes Metal's codes of
//! ([`SHADER_UNORM_DEPTH`]),
//! draws of points in a rate-mapped pass, which keep Vulkan's clipping of points, in
//! `wide_points` ([`POINT_GUARD_BAND`]), draws whose fragment function writes a depth in depth
//! clamp mode through a range above 0 that keep the device's clamp, in `viewport_ranges`
//! ([`NAN_DEPTH`]), draws of triangles in depth clip mode under a bias that takes the depth past
//! `[0, 1]` on a device that stores it unclamped and cannot clamp it ([`UNCLAMPED_BIAS`]), draws
//! through depth ranges past `[0, 1]` whose vertex stage clamps each
//! vertex's depth, or cannot, and bias factors in depth clamp mode through a range past 1 there
//! ([`VERTEX_CLAMPS`]),
//! sampler states of an anisotropy past the device's largest here ([`CAPPED_ANISOTROPY`]),
//! anisotropic sampler states without a mip filter through Metal's own sampler
//! ([`NOT_MIPMAPPED_ANISOTROPY`]), draws that read an attachment's bytes in place in
//! `attachment_reads` ([`IN_PLACE`]), textures
//! held for copies alone that a kernel's texture handles reach in `pvrtc_textures`
//! ([`UNREADABLE`]), the device's texture table and heap sets in `resource_ids` (an unviewable
//! texture, [`UNVIEWABLE`]; a shadowed one, [`UNNAMED_SHADOWS`]; an ID past a pipeline's arrays,
//! [`PAST_CAPACITY`]; a heap set made past its driver's support answer, [`PAST_LAYOUT_SUPPORT`];
//! each command buffer's own copy on a device withholding
//! `descriptorBindingUpdateUnusedWhilePending`, [`OWN_COPIES`]; a no-copy buffer's deallocator
//! that waits for its pages' import, which a texture's image a submitted command buffer may read
//! holds, [`DELAYED_DEALLOCATORS`]), a no-copy buffer over pages the process maps read-only,
//! made from its dma-buf source in `dma_buf_imports` on Linux ([`READ_ONLY_PAGES`]),
//! pass colors a draw keeps where Metal's compiler may fold the output
//! and zero them in `color_reformat` ([`KEPT_COLORS`]), multisampled depth copied in by an
//! inexact fragment-depth draw in `sample_copies` ([`DRAWN_DEPTH`]), textures bound beyond their
//! declared usage through an image of their own in `buffer_texture_order` ([`USAGE_SHADOWS`]),
//! command buffers whose named uses of a buffer rebound before their first submission fail in
//! `rebind_reach` ([`REBOUND_USES`]), textures over bytes detached after they were resident,
//! and writable ones whose writes their stand-ins' sinks keep, reached unnamed in
//! `detach_textures` ([`DETACHED_REACHES`]), textures in heaps laid out otherwise than Metal's
//! tiles in `heap_layouts` and `heap_textures` ([`HEAP_LAYOUTS`]), fetches under a written depth test
//! run in the shader in `framebuffer_fetch::shader_depth` ([`SHADER_DEPTH`]),
//! heap textures whose layout in Metal's heaps the runtime does not model in `heap_metal_sizes`
//! ([`UNMODELLED_SIZES`]) and those whose images take more of a heap than Metal's answer there
//! ([`PAST_METAL_SIZES`]), BC1 to BC5 textures on a driver not measured to decode them as Metal
//! does in `compressed_decode` ([`BLOCK_DECODES`]) and writes into their decoded twins behind
//! work in flight there ([`BEHIND_WRITES`]), ASTC textures on a device without their format in
//! `astc_textures` ([`ASTC_TWINS`]; PVRTC ones take the first in `pvrtc_textures`), passes rendering a linear image apart in
//! `linear_attachments` ([`LINEAR_COPIES`]) and those rendering one through copies as they clear
//! it and no command of them may read it, there too ([`CHOSEN_COPIES`]), passes rendering a texture
//! over a buffer they do not store through the image its texture keeps in `kept_buffer_bytes`
//! ([`KEPT_BYTES`]), the device's
//! first sparse texture on a driver whose first sparse resource waits for every queue's work in
//! `first_use` ([`FIRST_SPARSE`]), command buffers a submission of several batches completes
//! together on a driver that merges submissions in `grouped_submissions` ([`GROUPED`]),
//! visibility
//! counts of draws writing several colors on a driver that counts each sample once per color in
//! `color_groups` ([`PER_COLOR_COUNTS`]), passes that load and draw a signed normalized attachment
//! holding a minimum code, or an `RGB9E5Float` one holding an unnormalized word, in
//! `snorm_tile_store` ([`SNORM_MINIMUM`], [`SHARED_EXPONENT`]), condition words of command
//! predication past their buffer that read memory unrelated to it in `predication`
//! ([`WORD_PAST_BUFFER`]),
//! and
//! pipeline states, whose stages' precision notes and reconvergence are announced where the
//! runtime makes their objects (one site, which maps every note to its kind; a fragment function's
//! interpolated varyings on a device not measured to interpolate as Metal does are one such note,
//! which `interpolated_varyings_announce_the_device_interpolation_once` takes; a vertex function's
//! simdgroups grouped otherwise than Apple's GPUs group vertices are one such note, which
//! `vertex_simdgroup_functions_announce_the_grouping_once` takes where the driver regroups them,
//! and the same test takes the render variant's vertex simdgroups on a driver that runs them
//! without advertising the stage, [`UNADVERTISED_VERTEX_SIMDGROUPS`]).
//! Implementation-defined reconvergence
//! fires only on a device without maximal reconvergence, which no venue here is, so a device that
//! withholds the feature takes it on every venue. A kind a venue does not take (emulated simdgroups
//! where the compile layer runs the dispatch on no emulated variant, a one-aspect resolve where the
//! device resolves the aspects independently, both properties no device option withholds,
//! uncalibrated GPU times where the device calibrates its timestamps, an unlisted sampler state
//! where the device's and the pipeline's sampler arrays hold its ID, a depth twin where the
//! device's depth images keep every `float`, an upload into a depth image where the device enabled
//! `VK_EXT_depth_range_unrestricted` or its driver keeps a value outside `[0, 1]` without it, and a
//! `Shared` buffer in system memory where the device has unified memory, a heap's depth or stencil
//! texture where the driver lays no such image apart, and a depth or stencil texture without
//! GPU-optimized contents where the driver drops the layout, emulated mesh draws where the
//! device has `VK_EXT_mesh_shader`, a capped anisotropy where the device's samplers take 32
//! or none, and a vertex simdgroup grouping where the driver groups vertices as Apple's GPUs do or
//! the vertex stage has no subgroups) is expected absent there, as the
//! compile layer, the device's description or its driver facts settle it.

use super::absent_reads::ABSENT_READS;
use super::amplified_draws::AMPLIFICATION;
use super::astc_textures::{decodes_astc, ASTC_TWINS};
use super::attachment_reads::IN_PLACE;
use super::barycentric_weights::SYNTHESIZED_GEOMETRY;
use super::binary_archives::ARCHIVES;
use super::blits::SYSTEM_MEMORY;
use super::buffer_texture_blits::{NARROW_ROWS, OVERLAPS};
use super::buffer_texture_order::USAGE_SHADOWS;
use super::buffer_texture_renders::MERGES;
use super::clear_reach::DONT_CARE_LOADS;
use super::color_groups::PER_COLOR_COUNTS;
use super::color_reformat::KEPT_COLORS;
use super::color_writes::COLOR_WRITES;
use super::command_queues::SHARED_PRESENT_QUEUES;
use super::compressed_decode::{BEHIND_WRITES, BLOCK_DECODES};
use super::cross_class_reads::CROSS_CLASS;
use super::depth_denormals::{DEPTH_DENORMALS, SIGNED_DEPTH_TESTS};
use super::detach_textures::DETACHED_REACHES;
use super::extended_range_blends::{EXTENDED_RANGE_BITS, STORED_CODES, UNREACHED_TILES};
use super::extended_range_resolve::XR_RESOLVE;
use super::extended_range_two_planes::HEAP_PLANES;
use super::first_use::FIRST_SPARSE;
use super::format_copies::MEMMOVES;
use super::fragment_imageblocks::FRAGMENT_IMAGEBLOCKS;
use super::framebuffer_fetch::{FETCHES, PLANES, SHADER_DEPTH, STORAGE_FORMS};
use super::fresh_contents::FRESH_PATTERNS;
use super::grouped_submissions::GROUPED;
use super::heap_layouts::HEAP_LAYOUTS;
use super::heap_metal_sizes::{PAST_METAL_SIZES, UNMODELLED_SIZES};
use super::heap_textures::{ALIASES, APART, COMPRESSED, OPTIMIZED};
use super::host_memory::{not_importable, Mapping, GUEST_PAGE, READ_ONLY_PAGES};
use super::index_fetches::INDEX_FETCHES;
use super::indirect_command_buffers::UNMEASURED_IDENTIFIERS;
use super::indirect_dispatches::INDIRECT_DISPATCHES;
use super::indirect_executions::{NIL_OFFSETS, RELEASED_STATES};
use super::indirect_host_steps::HOST_STEPS;
use super::kept_buffer_bytes::KEPT_BYTES;
use super::linear_attachments::{CHOSEN_COPIES, LINEAR_COPIES};
use super::mesh_draws::EMULATED_MESH;
use super::mipmaps_exact::{M3_MIPMAPS, TIE_MIPMAPS};
use super::no_copy_heaps::HEAP;
use super::pass_start_reads::{FRAGMENTS_BEFORE_GEOMETRY, IN_PLACE_VERTEX_READS};
use super::patches::EMULATED_TESSELLATION;
use super::predication::WORD_PAST_BUFFER;
use super::provoking::PROVOKING;
use super::pvrtc_textures::UNREADABLE;
use super::rate_mapped_passes::{RESET_UNENCODED, ZERO_LAYER_AFTER_RELEASE};
use super::rebind_reach::REBOUND_USES;
use super::resolves::{color_pass, lane, quiet, MIXED_RESOLVES, ONE};
use super::resource_ids::{
    DELAYED_DEALLOCATORS, OWN_COPIES, PAST_CAPACITY, PAST_LAYOUT_SUPPORT, UNLISTED,
    UNNAMED_SHADOWS, UNVIEWABLE,
};
use super::rotated_textures::ROTATED_LOADS;
use super::sample_copies::DRAWN_DEPTH;
use super::shader_stencil::SHADOW;
use super::snorm_tile_store::{SHARED_EXPONENT, SNORM_MINIMUM};
use super::sparse_textures::SPARSE;
use super::texture_blits::{MIPMAPS, TWINS, UPLOADS};
use super::tile_dispatches::TILE_DISPATCHES;
use super::topology_classes::UNSIZED_POINTS;
use super::two_plane_copies::TWO_PLANE_COPIES;
use super::unorm_depth_bias::{SHADER_UNORM_DEPTH, UNORM_DEPTH_BIAS, UNORM_DEPTH_ROUNDING};
use super::view_reinterpretation::VIEW_SHADOWS;
use super::viewport_ranges::{NAN_DEPTH, UNCLAMPED_BIAS, VERTEX_CLAMPS};
use super::visibility::REPEATED_STARTS;
use super::wide_points::POINT_GUARD_BAND;
use super::ycbcr_passes::YCBCR_PASSES;
use super::ycbcr_textures::YCBCR_FILTERS;
use super::{
    case_making_kinds, creation_heard_checked, made_lane_device, pipeline_state, public_case,
    run_case, run_with, shared_kinds, validated_device_with, Messages, PublicCase, ScratchCache,
};
use crate::air_library::AirLibrary;
use crate::candidate::engine::compute_shape;
use crate::candidate::engine::resources::pixel_format;
use crate::candidate::engine::variants::compile_compute_for_shape;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::{AstcDecode, SamplerDescriptor};
use metal2vulkan::cache::VariantKey;
use metal2vulkan::device::VkExtension;
use metal2vulkan::device::{DecodedTexelsPolicy, ResolvePolicy};
use metal2vulkan::device::{DriverFact, DriverWorkaround, OpBits, VkFeature, VkProperty};
use metal2vulkan::device::{SimdClass, Tier};
use metal2vulkan::error::{DriverDeviation, Feature};
use metal2vulkan::library::Library;
use metal2vulkan::metal::format::bound_texel_format;
use metal2vulkan::metal::{
    PixelFormat, SamplerMinMagFilter, SamplerMipFilter, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::MeshPipelineDescriptor;
use metal2vulkan::pipeline::PipelineLayout;
use metal2vulkan::pipeline::StageModule;
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, Stage, StageSource,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, Reconvergence};
use metal2vulkan::pipeline::{PrecisionNote, SlotRef, SpecKey};
use metal2vulkan::vk::DriverId;
use metal2vulkan::vk::ResolveModeFlags as Mode;
use metal2vulkan::Budget;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CpuCacheMode, DebugMessage, Degradation, DegradationCategory, Device,
    DeviceOptions, DriverSetting, Error, HazardTrackingMode, HeapDescriptor, HeapType, HostMemory,
    LayerSetting, LayerSettingValue, LoadAction, MessageSeverity, MetalLevel,
    MultisampleStencilResolveFilter, Origin, PipelineOptions, Problem, RenderPassDescriptor,
    ResourceOptions, RuntimeGap, Size, StorageMode, StoreAction, TextureDescriptor,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Duration;

/// The longest a command buffer here may take.
const WAIT: Duration = Duration::from_secs(60);

/// The kinds `host_bytes` takes over buffers whose bytes start past their `VkBuffer`'s start.
pub(super) const HOST_BYTES: [Degradation; 4] = [
    Degradation::ShadowedTexture,
    Degradation::StagedTextureBuffer,
    Degradation::CopiedDrawBytes,
    Degradation::LandedVisibilityResults,
];

/// The kinds [`blits_degrade_once_per_kind`] takes.
const BLITS: [Degradation; 3] = [
    Degradation::OverlappingBufferCopy,
    Degradation::SplitBufferFill,
    Degradation::StagedTextureCopy,
];

/// The kinds [`dispatches_degrade_once_per_kind`] takes (the second where the device pins the
/// kernel's subgroup size).
const DISPATCHES: [Degradation; 2] = [Degradation::SplitDispatch, Degradation::EmulatedSimdgroups];

/// The kinds [`passes_degrade_once_per_kind`] takes (the second where the device resolves a
/// combined format's aspects one at a time; the exact resolve and the device's rounded one where
/// the driver's average of `RGBA8Unorm` at 4 samples is not measured as Metal's, or where the
/// device makes a texture of 8 samples; the unmeasured resolve where it makes one).
const PASSES: [Degradation; 7] = [
    Degradation::SeparateDepthStencil,
    Degradation::OneAspectResolve,
    Degradation::ResolveScratch,
    Degradation::DeviceRoundedResolve,
    Degradation::DontCareStore,
    Degradation::ShaderResolve,
    Degradation::UnmeasuredResolve,
];

/// The kind `presenting`'s cross-queue test takes: a texture over the test's image that rests in
/// `TRANSFER_DST_OPTIMAL` between batches.
pub(super) const WRAPPED: [Degradation; 1] = [Degradation::WrappedImageLayoutTransitions];

/// The kind `sample_positions`' test of a caller's image made without the custom-location flag
/// takes: its pass at custom positions renders into a stand-in made with it.
pub(super) const POSITIONED: [Degradation; 1] = [Degradation::PositionsThroughScratch];

/// The kinds a device announces as it is made, which the lane's constructor takes and checks
/// exactly (`runtime_lane::creation_kinds`); `metal-level-below-host` is [`METAL_LEVELS`].
/// `float-tininess-after-rounding` is announced by the devices `native_maths` makes with native
/// device maths, on every venue but MoltenVK, and by none made with the lane's exact policy.
pub(super) const CREATED: [Degradation; 3] = [
    Degradation::MetalLevelNotMacOS,
    Degradation::DriverPipelineCacheKept,
    Degradation::FloatTininessAfterRounding,
];

/// The kind `capabilities`' private answers take, read once.
pub(super) const PRIVATE_PROPERTIES: [Degradation; 1] = [Degradation::PrivatePropertiesUnknown];

/// The kind `gpu_times` takes on a venue whose device calibrates no timestamps against a host
/// clock the runtime reads (and expects absent where it does), as does `first_use`'s busy work,
/// whose measurement reads GPU times (`gpu_times::read_kinds`).
pub(super) const GPU_TIMES: [Degradation; 1] = [Degradation::UncalibratedGpuTimes];

/// The kind `gpu_times` takes on every venue that writes timestamps, with more command buffers in
/// flight than the device's timestamp pools have slots.
pub(super) const UNTIMED: [Degradation; 1] = [Degradation::UntimedCommandBuffer];

/// The kind `color_fills` takes on every venue, with a color packed in a padded 10-bit format.
pub(super) const UNPACKED_FILL: [Degradation; 1] = [Degradation::UnpackedFillColor];

/// The kind `texture_buffer_fills` takes on every venue, with fills whose chunks read bytes past
/// their pattern's whole copies.
pub(super) const TEXTURE_BUFFER_FILL: [Degradation; 1] = [Degradation::TextureBufferFillLeftover];

/// The kind `gpu_times` takes on every venue, on a device made with `gpu_times_from_host`, and
/// `first_use`'s busy work where the driver answers GPU times from the host
/// (`gpu_times::read_kinds`).
pub(super) const FROM_HOST: [Degradation; 1] = [Degradation::GpuTimesFromHost];

/// A non-uniform grid: 10 by 3 threads in threadgroups of 8 by 4.
const RAGGED: &str = "a-ragged-thread-grid-observes-each-threadgroups-size-and-position";
/// Shuffles in threadgroups of 8 threads: partial simdgroups.
const PARTIAL_SIMDGROUPS: &str = "dsm2-shuffle-past-the-thread-count-eight";
/// Kernels of float functions and arithmetic, and one of a nearest cube sample, whose stages
/// record precision notes (the cube's `CubeFaceReciprocal` on a device without
/// `VkProperty::MetalSampler`).
const NOTED: [&str; 3] = [
    "anchor-three-transcendentals-where-rounding-cannot-reach",
    "nan-model-float-denormals-per-operation",
    "cube-major-axis-ties-at-8x8",
];

/// How often each kind was announced among `notices`, each notice checked for its form: a
/// warning named by its kind's code, of the performance kind for a performance degradation and
/// the general kind for a precision one, its text the payload's.
pub(super) fn counts(notices: &[DebugMessage]) -> BTreeMap<Degradation, usize> {
    let mut counts = BTreeMap::new();
    for message in notices {
        let notice = message.degradation.as_ref().expect("a degrade notice");
        let kind = notice.degradation;
        assert_eq!(message.severity, MessageSeverity::Warning, "{message:?}");
        assert_eq!(message.id_name, kind.code(), "{message:?}");
        assert_eq!(Degradation::from_code(&message.id_name), Some(kind));
        let performance = kind.category() == DegradationCategory::Performance;
        assert_eq!(message.kind.performance, performance, "{message:?}");
        assert_eq!(message.kind.general, !performance, "{message:?}");
        assert!(!message.kind.validation, "{message:?}");
        assert_eq!(notice.cost, kind.cost());
        assert!(
            !notice.object.is_empty() && !notice.reason.is_empty(),
            "{message:?}"
        );
        let text = format!(
            "metal2vulkan: {}: {}; {} [{}]",
            notice.object,
            notice.reason,
            notice.cost,
            kind.code()
        );
        assert_eq!(message.text, text);
        eprintln!("notice: {}", message.text);
        *counts.entry(kind).or_insert(0) += 1;
    }
    counts
}

/// The kinds `indirect_draws` takes: an indexed indirect draw of bytes a draw of its encoder
/// wrote with no memory barrier in the encoder.
pub(super) const UNORDERED_DRAWS: [Degradation; 1] = [Degradation::UnorderedDrawBytes];

/// Every kind among `notices` is announced once, and the kinds announced are exactly `kinds`:
/// their [`counts`]. A kind past `kinds` fails as one missing does (before, any other kind passed,
/// so an unforced degradation went unseen). A device's creation notices are its constructor's
/// (`runtime_lane::validated_device_with`), so `notices` holds only the test's own work's.
pub(super) fn assert_once(
    notices: &[DebugMessage],
    kinds: &[Degradation],
) -> BTreeMap<Degradation, usize> {
    let counts = counts(notices);
    let twice: Vec<_> = counts.iter().filter(|(_, &n)| n > 1).collect();
    assert!(twice.is_empty(), "announced more than once: {twice:?}");
    let missing: Vec<_> = kinds.iter().filter(|k| !counts.contains_key(k)).collect();
    assert!(
        missing.is_empty(),
        "not announced: {missing:?} (counts {counts:?})"
    );
    let unexpected: Vec<_> = counts.keys().filter(|k| !kinds.contains(k)).collect();
    assert!(
        unexpected.is_empty(),
        "announced past {kinds:?}: {unexpected:?} (counts {counts:?})"
    );
    counts
}

/// [`assert_once`] of `kinds`, but each kind of `allowed` may be announced, once, or not: the
/// kinds a step's values decide where the test does not model them ([`bound_value_kinds`]).
pub(super) fn assert_once_allowing(
    notices: &[DebugMessage],
    kinds: &[Degradation],
    allowed: &BTreeSet<Degradation>,
) -> BTreeMap<Degradation, usize> {
    let announced = counts(notices);
    let mut expected: Vec<Degradation> = kinds.to_vec();
    expected.extend(
        allowed
            .iter()
            .filter(|k| announced.contains_key(k) && !kinds.contains(k)),
    );
    assert_once(notices, &expected)
}

/// A device under the validation layer reporting to `messages`, for `test`.
pub(super) fn on_device(test: impl FnOnce(&Device, &Messages)) {
    on_device_with(|_| {}, test);
}

/// [`on_device`] with the options `adjust` changes.
pub(super) fn on_device_with(
    adjust: impl FnOnce(&mut DeviceOptions),
    test: impl FnOnce(&Device, &Messages),
) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
        test(&device, &messages);
    }
    assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
}

/// Every kind is taken by a lane test's site: listed above, or a stage's precision note or
/// reconvergence; no kind is in two lists, and no list names a kind twice. `misplaced-notice`
/// ([`MISPLACED`]) is taken by the runtime's own tests instead, as no lane can meet it.
#[test]
fn every_kind_has_a_site_a_lane_test_takes() {
    let stage_kinds = Degradation::ALL.iter().copied().filter(|kind| {
        kind.precision_note().is_some() || *kind == Degradation::ImplementationDefinedReconvergence
    });
    let mut listed = BTreeSet::new();
    let lists = [
        &HOST_BYTES[..],
        &BLITS,
        &DISPATCHES,
        &PASSES,
        &FETCHES,
        &STORAGE_FORMS,
        &POSITIONED,
        &PLANES,
        &WRAPPED,
        &SHADOW,
        &TILE_DISPATCHES,
        &FRAGMENT_IMAGEBLOCKS,
        &GPU_TIMES,
        &UNTIMED,
        &UNPACKED_FILL,
        &TEXTURE_BUFFER_FILL,
        &FROM_HOST,
        &UNLISTED,
        &RELEASED_STATES,
        &HOST_STEPS,
        &INDIRECT_DISPATCHES,
        &INDEX_FETCHES,
        &TWINS,
        &UPLOADS,
        &MIPMAPS,
        &APART,
        &OPTIMIZED,
        &ALIASES,
        &COMPRESSED,
        &SPARSE,
        &OVERLAPS,
        &MERGES,
        &SYSTEM_MEMORY,
        &AMPLIFICATION,
        &COLOR_WRITES,
        &ARCHIVES,
        &VIEW_SHADOWS,
        &M3_MIPMAPS,
        &REPEATED_STARTS,
        &PRIVATE_OPTION,
        &IN_PLACE_VERTEX_READS,
        &FRAGMENTS_BEFORE_GEOMETRY,
        &UNORM_DEPTH_BIAS,
        &UNORM_DEPTH_ROUNDING,
        &SHADER_UNORM_DEPTH,
        &POINT_GUARD_BAND,
        &NAN_DEPTH,
        &UNCLAMPED_BIAS,
        &VERTEX_CLAMPS,
        &UNORDERED_DRAWS,
        &ABSENT_READS,
        &DONT_CARE_LOADS,
        &ROTATED_LOADS,
        &NARROW_ROWS,
        &MIXED_RESOLVES,
        &EMULATED_MESH,
        &PROVOKING,
        &CAPPED_ANISOTROPY,
        &NOT_MIPMAPPED_ANISOTROPY,
        &IN_PLACE,
        &ARGUMENT_BUFFERS,
        &METAL_LEVELS,
        &UNREADABLE,
        &UNVIEWABLE,
        &UNNAMED_SHADOWS,
        &PAST_CAPACITY,
        &PAST_LAYOUT_SUPPORT,
        &OWN_COPIES,
        &KEPT_COLORS,
        &PER_COLOR_COUNTS,
        &DRAWN_DEPTH,
        &USAGE_SHADOWS,
        &HEAP_PLANES,
        &TWO_PLANE_COPIES,
        &EXTENDED_RANGE_BITS,
        &UNREACHED_TILES,
        &SNORM_MINIMUM,
        &SHARED_EXPONENT,
        &STORED_CODES,
        &UNADVERTISED_VERTEX_SIMDGROUPS,
        &YCBCR_FILTERS,
        &REBOUND_USES,
        &DETACHED_REACHES,
        &ZERO_LAYER_AFTER_RELEASE,
        &RESET_UNENCODED,
        &HEAP_LAYOUTS,
        &SHADER_DEPTH,
        &TIE_MIPMAPS,
        &MEMMOVES,
        &FRESH_PATTERNS,
        &UNMODELLED_SIZES,
        &UNMEASURED_IDENTIFIERS,
        &CROSS_CLASS,
        &NIL_OFFSETS,
        &BLOCK_DECODES,
        &BEHIND_WRITES,
        &ASTC_TWINS,
        &SHARED_PRESENT_QUEUES,
        &PAST_METAL_SIZES,
        &LINEAR_COPIES,
        &CHOSEN_COPIES,
        &KEPT_BYTES,
        &DELAYED_DEALLOCATORS,
        &READ_ONLY_PAGES,
        &MISPLACED,
        &DEPTH_DENORMALS,
        &SIGNED_DEPTH_TESTS,
        &FIRST_SPARSE,
        &CREATED,
        &PRIVATE_PROPERTIES,
        &UNSIZED_POINTS,
        &SYNTHESIZED_GEOMETRY,
        &EMULATED_TESSELLATION,
        &EXACT_FORMS_OVER_LIMIT,
        &GROUPED,
        &WORD_PAST_BUFFER,
        &XR_RESOLVE,
        &YCBCR_PASSES,
    ];
    for kind in lists.into_iter().flatten().copied().chain(stage_kinds) {
        assert!(listed.insert(kind), "{kind:?} is listed twice");
    }
    let all: BTreeSet<Degradation> = Degradation::ALL.iter().copied().collect();
    assert_eq!(listed, all);
}

/// The kind a MoltenVK device whose argument-buffer state the runtime does not read takes
/// ([`moltenvk_argument_buffers_are_announced_where_unread_and_refused_off`]).
const ARGUMENT_BUFFERS: [Degradation; 1] = [Degradation::MoltenVkArgumentBuffersUnread];

/// A device opens announcing [`ARGUMENT_BUFFERS`] exactly where it is MoltenVK and its
/// argument-buffer state is not read: a release other than 1.4.1 and 1.4.2, or one of them whose
/// `maxPerStageDescriptorUpdateAfterBindSampledImages` is neither 1,000,000 (argument buffers on)
/// nor, on an Apple GPU, the per-stage limit (off). Every lane venue's MoltenVK is 1.4.1 or 1.4.2
/// on an Apple GPU with argument buffers on, so no venue announces it today; each checks the rule
/// on its own device. On MoltenVK (the M3 and M4 venues) a device opened with the consumer's own
/// layer setting `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS` false, which MoltenVK applies after the
/// runtime's, is refused at device open, `Error::DriverSetting(MoltenVkArgumentBuffersOff)`, with
/// nothing for the validation layer to report.
#[test]
fn moltenvk_argument_buffers_are_announced_where_unread_and_refused_off() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let moltenvk = {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        let name = device.name().to_owned();
        let description = &device.properties().description;
        let limits = &description.limits;
        let moltenvk = description.driver_id == Some(DriverId::MOLTENVK);
        let uab = limits.max_per_stage_descriptor_update_after_bind_sampled_images;
        let read = super::MOLTENVK_ARGUMENT_BUFFERS_READ.contains(&description.driver_version)
            && (uab == 1_000_000
                || (description.vendor_id == 0x106b
                    && uab == limits.max_per_stage_descriptor_sampled_images));
        // The constructor checked the creation notices against `creation_kinds`.
        let made = super::creation_kinds(&device, false);
        assert!(
            messages.take_notices().is_empty(),
            "announced past the creation"
        );
        eprintln!(
            "{name}: driverID {:?}, driverVersion {}, vendorID {:#x}, \
             maxPerStageDescriptorUpdateAfterBindSampledImages {uab}; made announcing {made:?}",
            description.driver_id, description.driver_version, description.vendor_id
        );
        for kind in ARGUMENT_BUFFERS {
            assert_eq!(
                made.contains(&kind),
                moltenvk && !read,
                "{kind:?} on {name} exactly where its MoltenVK's state is unread: {made:?}"
            );
        }
        moltenvk
    };
    assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
    if !moltenvk {
        return;
    }
    let off = Messages::default();
    let mut options = super::validated_options(&off, cache.path());
    options.layer_settings.push(LayerSetting::new(
        c"MoltenVK".to_owned(),
        c"MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS".to_owned(),
        LayerSettingValue::Bool(false),
    ));
    match crate::candidate::runtime_case::open_device(TEST_DEVICE, options) {
        Err(crate::candidate::runtime_case::CaseError::Runtime { error, .. })
            if matches!(
                *error,
                Error::DriverSetting(DriverSetting::MoltenVkArgumentBuffersOff)
            ) => {}
        Err(other) => panic!("argument buffers off: {other}"),
        Ok(_) => panic!("argument buffers off: the device opened"),
    }
    assert_eq!(off.take(), Vec::new(), "the validation layer reported");
}

/// The kind a pipeline state takes where a stage of its translation, with the device's exact
/// forms, is longer than the device's compiler word bound, and the translation dropped exact
/// forms to fit it ([`exact_forms_past_the_compiler_bound_are_announced_exactly_there`]).
const EXACT_FORMS_OVER_LIMIT: [Degradation; 1] = [Degradation::ExactFormsOverCompilerLimit];

/// A render pipeline state, and a compute pipeline state, announces [`EXACT_FORMS_OVER_LIMIT`]
/// exactly where a stage of its translation with the lane's exact forms
/// (`runtime_lane::validated_options`) is longer than the device's compiler word bound
/// (`DeviceProfile::module_word_bound`: ANV's, `None` elsewhere). No owned fixture's stage reaches
/// ANV's bound, so no venue announces it today; each checks the rule on its own device, with the
/// owned `runtime_examples` render pipeline and kernel. The runtime's own tests
/// (`fitted_variant_notice`) take the announcement of a fitted variant of every kind of state,
/// and the translator's (`pipeline::compile::fit`) the fit of every kind of pipeline.
#[test]
fn exact_forms_past_the_compiler_bound_are_announced_exactly_there() {
    on_device(|device, messages| {
        let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/public/examples/runtime_examples.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = |name| {
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor()
        };
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function("colored_vertex")));
        descriptor.fragment = Some(StageSource::Function(function("colored_fragment")));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let compile = device.compile_device();
        let bound = compile.profile().module_word_bound();
        // The notices a state's creation made, against whether its longest stage passes the bound.
        let check = |what: &str, longest: usize| {
            let past = bound.is_some_and(|bound| longest > bound as usize);
            let announced = counts(&messages.take_notices())
                .get(&Degradation::ExactFormsOverCompilerLimit)
                .copied()
                .unwrap_or(0);
            eprintln!(
                "{}: {what}: longest stage {longest} words, compiler word bound {bound:?}",
                device.name()
            );
            assert_eq!(
                announced,
                usize::from(past),
                "{what}: {:?} exactly past the bound",
                EXACT_FORMS_OVER_LIMIT
            );
        };
        let compiled = compile
            .compile_render_pipeline(&descriptor, &Budget::standard())
            .unwrap();
        let longest = compiled
            .stages
            .iter()
            .map(|stage| stage.spirv.len())
            .max()
            .unwrap_or(0);
        let _state = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        check("render", longest);
        let kernel =
            ComputePipelineDescriptor::new(StageSource::Function(function("scale_and_offset")));
        let longest = compile
            .compile_compute_pipeline(&kernel, &Budget::standard())
            .unwrap()
            .stage
            .spirv
            .len();
        let _state = device
            .new_compute_pipeline_state(&kernel, &PipelineOptions::default())
            .unwrap();
        check("compute", longest);
    });
}

/// The kind a device answering at a Metal level below its host's macOS release takes
/// ([`a_metal_level_below_the_host_release_is_announced_once`]).
const METAL_LEVELS: [Degradation; 1] = [Degradation::MetalLevelBelowHost];

/// A device opens at its host's Metal level, the device's site of
/// [`METAL_LEVELS`]: on macOS, the measured level of the host's release, or else the newest
/// measured below it, announcing [`METAL_LEVELS`] once, exactly where the release is not a
/// measured one; off macOS, the newest level, announcing `metal-level-not-macos`. Every lane
/// venue's macOS release is a measured level (26.7 on the M3, 27.0.1 on the M4) or none, so no
/// venue announces [`METAL_LEVELS`] today; each checks the rule on its own host. A device made at a
/// named level (`DeviceOptions::metal_level`) answers at it and announces neither. The lane's
/// constructor checks each device's creation notices exactly (`creation_kinds`), so the test checks
/// the level, that the constructor expects the notice exactly where the host's release is not a
/// measured one, and that nothing more is announced.
#[test]
fn a_metal_level_below_the_host_release_is_announced_once() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let host = super::host_release();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
    let level = device.properties().metal_level;
    let made = assert_once(&messages.take_notices(), &[]);
    let created = super::creation_kinds(&device, false);
    let measured =
        |level: MetalLevel| super::macos_release(level.macos()).expect("a level's release reads");
    let (expected, below) = match host {
        Some(version) => {
            let at = MetalLevel::ALL
                .into_iter()
                .rev()
                .find(|&level| measured(level) <= version)
                .expect("the host's release is at or above a measured level");
            (at, measured(at) != version)
        }
        None => (MetalLevel::NEWEST, false),
    };
    eprintln!(
        "{}: host macOS {host:?}, Metal level {level}; made announcing {created:?}",
        device.name()
    );
    assert_eq!(
        level, expected,
        "the device's Metal level on macOS {host:?}"
    );
    for kind in METAL_LEVELS {
        assert_eq!(
            created.contains(&kind),
            below,
            "{kind:?} exactly where macOS {host:?} is not a measured release: {created:?}"
        );
    }
    assert_eq!(
        created.contains(&Degradation::MetalLevelNotMacOS),
        host.is_none(),
        "metal-level-not-macos exactly off macOS: {created:?}"
    );
    assert!(made.is_empty(), "announced past the creation: {made:?}");
    drop(device);
    for named in MetalLevel::ALL {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.metal_level = Some(named);
        })
        .unwrap();
        assert_eq!(device.properties().metal_level, named, "the level named");
        let made = assert_once(&messages.take_notices(), &[]);
        let created = super::creation_kinds(&device, true);
        for kind in METAL_LEVELS
            .into_iter()
            .chain([Degradation::MetalLevelNotMacOS])
        {
            assert!(
                !created.contains(&kind),
                "{kind:?} at the named level {named}: {created:?}"
            );
        }
        assert!(made.is_empty(), "announced past the creation: {made:?}");
    }
    assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
}

/// The kind an anisotropy past the device's largest takes
/// ([`an_anisotropy_past_the_devices_largest_is_announced_once`]).
const CAPPED_ANISOTROPY: [Degradation; 1] = [Degradation::CappedAnisotropy];

/// Sampler states of a `maxAnisotropy` plain Metal samples with 32 (17 to 31 in its low five
/// bits, which Metal's validation stops) are made, filtering with the device's largest anisotropy
/// where that is less, announced once; a state of 16 is not announced. A device without anisotropy
/// refuses each, `Unsupported`, and announces nothing.
#[test]
fn an_anisotropy_past_the_devices_largest_is_announced_once() {
    on_device(|device, messages| {
        let limits = device.compile_device().profile().limits();
        let largest = f32::from_bits(limits.max_sampler_anisotropy_bits);
        let mut d = SamplerDescriptor::new();
        d.max_anisotropy = 16;
        let anisotropic = match device.new_sampler_state(&d) {
            Ok(_) => true,
            Err(Error::Unsupported(u)) => {
                eprintln!("anisotropy 16: {u:?}");
                false
            }
            Err(other) => panic!("anisotropy 16: {other}"),
        };
        assert_eq!(counts(&messages.take_notices()), BTreeMap::new());
        for value in [17, 31, 63, u32::MAX] {
            d.max_anisotropy = value;
            match device.new_sampler_state(&d) {
                Ok(_) => assert!(anisotropic, "anisotropy {value} made where 16 is not"),
                Err(Error::Unsupported(u)) if !anisotropic => {
                    eprintln!("anisotropy {value}: {u:?}")
                }
                Err(other) => panic!("anisotropy {value}: {other}"),
            }
        }
        let notices = messages.take_notices();
        if anisotropic && largest < 32.0 {
            assert_eq!(assert_once(&notices, &CAPPED_ANISOTROPY).len(), 1);
        } else {
            assert_eq!(counts(&notices), BTreeMap::new());
        }
    });
}

/// The kind an anisotropic sampler state without a mip filter whose filters differ takes through
/// Metal's own sampler ([`an_anisotropic_state_without_mips_is_announced_through_metals_sampler`]).
const NOT_MIPMAPPED_ANISOTROPY: [Degradation; 1] = [Degradation::NotMipmappedAnisotropicSampler];

/// Sampler states of `maxAnisotropy` 8 without a mip filter are made; the one whose minification
/// and magnification filters differ is announced once, exactly on a device with anisotropy that
/// samples through Metal's own sampler (`DriverDeviation::MetalSampler`, the M3 and M4 venues'
/// MoltenVK). Equal filters, a mip filter, or no anisotropy announce nothing anywhere.
#[test]
fn an_anisotropic_state_without_mips_is_announced_through_metals_sampler() {
    on_device(|device, messages| {
        let profile = device.compile_device().profile();
        let own = profile.takes_driver_deviation(DriverDeviation::MetalSampler)
            && profile
                .enabled()
                .features
                .contains(&VkFeature::SamplerAnisotropy);
        let state = |min, mag, mip, anisotropy| {
            let mut d = SamplerDescriptor::new();
            d.min_filter = min;
            d.mag_filter = mag;
            d.mip_filter = mip;
            d.max_anisotropy = anisotropy;
            d
        };
        use SamplerMinMagFilter::{Linear, Nearest};
        use SamplerMipFilter::NotMipmapped;
        let unannounced = [
            state(Linear, Linear, NotMipmapped, 8),
            state(Linear, Nearest, SamplerMipFilter::Linear, 8),
            state(Linear, Nearest, NotMipmapped, 1),
        ];
        for d in unannounced {
            match device.new_sampler_state(&d) {
                Ok(_) | Err(Error::Unsupported(_)) => {}
                Err(other) => panic!("{d:?}: {other}"),
            }
        }
        assert_eq!(counts(&messages.take_notices()), BTreeMap::new());
        let mixed = state(Linear, Nearest, NotMipmapped, 8);
        for _ in 0..2 {
            match device.new_sampler_state(&mixed) {
                Ok(_) => {}
                Err(Error::Unsupported(u)) if !own => eprintln!("{mixed:?}: {u:?}"),
                Err(other) => panic!("{mixed:?}: {other}"),
            }
        }
        let notices = messages.take_notices();
        if own {
            assert_eq!(assert_once(&notices, &NOT_MIPMAPPED_ANISOTROPY).len(), 1);
        } else {
            assert_eq!(counts(&notices), BTreeMap::new());
        }
    });
}

/// The kind a build without debug assertions announces after a notice announced against the
/// runtime's rule. No lane meets it: every lane builds with debug assertions, where such an
/// announcement ends the process instead (so a lane that met one would fail). The runtime's own
/// test `a_misplaced_notice_reaches_the_sink_unchanged_from_the_notice_thread_and_is_recorded_once`
/// takes its site, as a build without debug assertions does.
const MISPLACED: [Degradation; 1] = [Degradation::MisplacedNotice];

/// The kind Metal's private option bit takes where the GPU's writes reach the memory here
/// ([`the_private_option_bit_is_announced_on_a_heap_of_the_runtime_s_memory`],
/// [`the_private_option_bit_drops_the_gpu_s_writes_to_no_copy_memory`]).
const PRIVATE_OPTION: [Degradation; 1] = [Degradation::PrivateResourceOption];

/// `options` with Metal's private option bit 24 (0x1000000).
fn with_private_bit(options: ResourceOptions) -> ResourceOptions {
    ResourceOptions::from_raw(options.raw() | ResourceOptions::PRIVATE_BIT)
        .expect("the runtime takes the private bit")
}

/// Whether `made` is refused `Invalid` for the undefined option bits `raw`.
fn refused_bits<T>(made: Result<T, Error>, raw: u64) -> bool {
    matches!(
        made,
        Err(Error::Invalid(invalid)) if invalid.problem
            == Problem::UndefinedValue { ty: "MTLResourceOptions", raw }
    )
}

/// Where Metal ends the process at Metal's private option bit 24 (the M3 Max under macOS 27.0.1:
/// a buffer of a length or of bytes, `Shared` or `Private`, asserts in its suballocator, and a
/// texture descriptor in its validation), the runtime refuses the call `Invalid`; a buffer with it
/// placed in a heap without it is made as without it and reports its options without it, as
/// Metal's is. Nothing is announced.
#[test]
fn the_private_option_bit_is_refused_where_metal_ends_and_ignored_in_a_plain_heap() {
    on_device(|device, messages| {
        let shared = with_private_bit(ResourceOptions::shared());
        let private = with_private_bit(ResourceOptions::private());
        assert!(refused_bits(device.new_buffer(256, shared), shared.raw()));
        assert!(refused_bits(device.new_buffer(256, private), private.raw()));
        assert!(refused_bits(
            device.new_buffer_with_bytes(&[7; 16], shared),
            shared.raw()
        ));
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
        d.resource_options = shared;
        assert!(refused_bits(device.new_texture(&d), shared.raw()));
        d.resource_options = private;
        assert!(refused_bits(device.new_texture(&d), private.raw()));
        let plain = ResourceOptions::private();
        let heap = device
            .new_heap(&HeapDescriptor::new(HeapType::Automatic, 1 << 16, plain))
            .unwrap();
        assert!(refused_bits(heap.new_texture(&d), private.raw()));
        let placed = heap.new_buffer(256, private).unwrap();
        assert_eq!(
            placed.resource_options(),
            heap.new_buffer(256, plain).unwrap().resource_options()
        );
        assert_eq!(
            placed.resource_options().raw() & ResourceOptions::PRIVATE_BIT,
            0
        );
        assert_made_once(messages, &[], &BTreeSet::new());
    });
}

/// A heap of the runtime's memory made with Metal's private option bit 24, which Metal makes and
/// keeps the GPU's writes from the memory of, is made as without it here, the GPU's writes
/// reaching the memory, the heap and a buffer placed in it reporting their options without the
/// bit; it is announced once.
#[test]
fn the_private_option_bit_is_announced_on_a_heap_of_the_runtime_s_memory() {
    on_device(|device, messages| {
        let private = with_private_bit(ResourceOptions::private());
        let heap = device
            .new_heap(&HeapDescriptor::new(HeapType::Automatic, 1 << 16, private))
            .unwrap();
        assert_eq!(
            heap.resource_options().raw() & ResourceOptions::PRIVATE_BIT,
            0
        );
        let placed = heap.new_buffer(256, private).unwrap();
        assert_eq!(
            placed.resource_options().raw() & ResourceOptions::PRIVATE_BIT,
            0
        );
        assert_made_once(messages, &PRIVATE_OPTION, &BTreeSet::new());
    });
}

/// Whether the GPU's writes to no-copy memory made with Metal's private option bit 24 are
/// dropped here, as Metal drops them: on macOS, where the runtime imports its pages through a
/// read-only view of them (the runtime's `read_only_view.rs`).
const DROPS_WRITES: bool = cfg!(target_os = "macos");

/// Of `buffer`, a no-copy buffer over host pages whose bytes are its `contents`: the CPU writes
/// 0x11 to its first 16 bytes, a command buffer fills them with 0x5a, and later command buffers
/// copy them out, before and after the CPU writes 0x33 there. Whether they dropped the fill: the
/// pages and the first copy kept the CPU's 0x11, and the second read its 0x33. Where they took
/// it, the pages and the first copy hold 0x5a, and the second the CPU's 0x33.
fn drops_the_fill(device: &Device, buffer: &Buffer) -> bool {
    let contents = buffer.contents().unwrap();
    let write = |byte: u8| {
        // SAFETY: the buffer's first 16 bytes are host memory of this process, and no command
        // uses them now.
        unsafe { std::ptr::write_bytes(contents.as_ptr(), byte, 16) };
    };
    // SAFETY: as above.
    let pages = || unsafe { std::slice::from_raw_parts(contents.as_ptr(), 16) }.to_vec();
    let queue = device.new_command_queue().unwrap();
    let run = |encode: &dyn Fn(&mut metal2vulkan_runtime::BlitCommandEncoder)| {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        encode(&mut blit);
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    };
    let out = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let copied = || {
        run(&|blit| {
            blit.copy_from_buffer_to_buffer(buffer, 0, &out, 0, 16)
                .unwrap()
        });
        // SAFETY: `out` holds 16 bytes, and no command uses them now.
        unsafe { std::slice::from_raw_parts(out.contents().unwrap().as_ptr(), 16) }.to_vec()
    };
    write(0x11);
    run(&|blit| blit.fill_buffer(buffer, 0..16, 0x5a).unwrap());
    let (kept, first) = (pages(), copied());
    write(0x33);
    let second = copied();
    assert_eq!(second, [0x33; 16], "a later copy reads the CPU's bytes");
    assert_eq!(kept, first, "a later copy reads the pages");
    if kept == [0x11; 16] {
        return true;
    }
    assert_eq!(kept, [0x5a; 16]);
    false
}

/// A no-copy buffer, a buffer placed in a no-copy heap and that buffer rebound to other pages,
/// made with Metal's private option bit 24, which Metal makes and keeps the GPU's writes from the
/// memory of (the M3 Max under macOS 27.0.1: a blit's fill reaches neither the CPU nor a later
/// command buffer, which reads the CPU's bytes): on macOS each drops a fill so, and nothing is
/// announced; elsewhere the fill reaches the pages, announced once. Each reports its options
/// without the bit, as Metal's does.
#[test]
fn the_private_option_bit_drops_the_gpu_s_writes_to_no_copy_memory() {
    on_device(|device, messages| {
        let bit = ResourceOptions::PRIVATE_BIT;
        let shared = with_private_bit(ResourceOptions::shared());
        let plain = device
            .new_buffer(256, ResourceOptions::shared())
            .unwrap()
            .resource_options();
        let memory = Mapping::anonymous(GUEST_PAGE);
        let other = Mapping::anonymous(GUEST_PAGE);
        // SAFETY: the mapping outlives the buffer, which is dropped first.
        let no_copy = unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::new(memory.pointer()),
                GUEST_PAGE as u64,
                shared,
                None,
            )
        };
        let no_copy = match no_copy {
            Err(error) if not_importable(&error) => {
                eprintln!("PRIVATEBIT: the device imports no host memory: {error:?}");
                return;
            }
            no_copy => no_copy.unwrap(),
        };
        assert_eq!(no_copy.resource_options(), plain);
        assert_eq!(drops_the_fill(device, &no_copy), DROPS_WRITES);
        // Rebound to other pages, it drops the GPU's writes to them too.
        // SAFETY: the other mapping outlives the buffer, and no other thread encodes or commits.
        unsafe { no_copy.rebind_bytes_no_copy(HostMemory::new(other.pointer()), None, None) }
            .unwrap();
        assert_eq!(no_copy.contents(), Some(other.pointer()));
        assert_eq!(drops_the_fill(device, &no_copy), DROPS_WRITES);
        let heap_memory = Mapping::anonymous(HEAP);
        let descriptor = HeapDescriptor::new(HeapType::Automatic, HEAP as u64, shared);
        // SAFETY: the mapping outlives the heap, which is dropped first.
        let heap = unsafe {
            device.new_heap_with_bytes_no_copy(
                HostMemory::new(heap_memory.pointer()),
                &descriptor,
                None,
                None,
            )
        }
        .unwrap();
        assert_eq!(heap.resource_options().raw() & bit, 0);
        let placed = heap.new_buffer(256, ResourceOptions::shared()).unwrap();
        assert_eq!(drops_the_fill(device, &placed), DROPS_WRITES);
        drop((placed, heap, no_copy));
        // Beside them, what the `Shared` resources force on the device.
        let kinds: &[Degradation] = match DROPS_WRITES {
            true => &[],
            false => &PRIVATE_OPTION,
        };
        assert_made_once(messages, kinds, &shared_kinds(device));
    });
}

/// Two command buffers each copy between overlapping ranges of one buffer, fill bytes that are
/// not whole words, and copy rows into a `Depth16Unorm` texture from an offset of 2, which
/// Vulkan takes only at multiples of 4 for a depth format: each kind is announced once.
#[test]
fn blits_degrade_once_per_kind() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::Depth16Unorm, 4, 4, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::new(
            StorageMode::Private,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        let depth = device.new_texture(&descriptor).unwrap();
        for _ in 0..2 {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(&buffer, 0, &buffer, 8, 64)
                .unwrap();
            blit.fill_buffer(&buffer, 1..63, 0x5a).unwrap();
            blit.copy_from_buffer_to_texture(
                &buffer,
                2,
                8,
                0,
                Size::new(4, 4, 1),
                &depth,
                0,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
            .unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
        }
        // Beside them, what the `Shared` buffer forces on the device.
        assert_made_once(messages, &BLITS, &shared_kinds(device));
    });
}

/// Whether `case`'s dispatch runs on the kernel's emulated-simdgroups variant on `device`, as the
/// compile layer settles it: where it does, the dispatch announces
/// `Degradation::EmulatedSimdgroups`.
fn emulates_simdgroups(device: &Device, case: &PublicCase) -> bool {
    let compile = device.compile_device();
    let addressed = compile.profile().descriptors().buffer_device_address;
    let shape = compute_shape(&case.compute, addressed).unwrap();
    let library = Library::from_air(&case.bitcode).unwrap();
    let function = FunctionDescriptor::new(library.function(&case.compute.entry).unwrap());
    let dispatch = Some(&case.compute.dispatch);
    let budget = Budget::standard();
    let (_, key) =
        compile_compute_for_shape(compile, &function, &shape, dispatch, None, &budget).unwrap();
    key.emulated_simdgroups
}

/// A non-uniform grid runs twice (split into regions), and a kernel of simdgroup shuffles twice
/// in threadgroups of 8 threads: each kind is announced once, the emulated simdgroups exactly
/// where the compile layer settles the dispatch on that variant (a device whose kernels' simdgroups
/// run on pinned subgroups wider than 8).
#[test]
fn dispatches_degrade_once_per_kind() {
    on_device(|device, messages| {
        let ragged = public_case(RAGGED).unwrap();
        let partial = public_case(PARTIAL_SIMDGROUPS).unwrap();
        for case in [&ragged, &ragged, &partial, &partial] {
            run_case(device, case).unwrap();
        }
        let emulated = emulates_simdgroups(device, &partial);
        eprintln!("{PARTIAL_SIMDGROUPS}: emulated simdgroups: {emulated}");
        let expected: &[Degradation] = if emulated {
            &DISPATCHES
        } else {
            &DISPATCHES[..1]
        };
        // Beside them, what the cases' makings and `Shared` buffers force on the device.
        let mut first = shared_kinds(device);
        first.extend(case_making_kinds(device, &ragged));
        first.extend(case_making_kinds(device, &partial));
        let counts = assert_made_once(messages, expected, &first);
        assert_eq!(
            counts.contains_key(&Degradation::EmulatedSimdgroups),
            emulated,
            "{counts:?}"
        );
    });
}

/// The kinds a pipeline state of `case` announces on `device`: the precision notes its kernel's
/// stage records that no specialization removes, and reconvergence that is not Metal's, as the
/// device's compile layer compiles it under the empty variant key. A note the stage keeps only
/// under some specializations (`StageModule::specialized_notes`) is announced where the runtime
/// makes the `VkPipeline` of a specialization that keeps it, at a dispatch, not by the state.
fn stage_kinds(device: &Device, case: &PublicCase) -> BTreeSet<Degradation> {
    let compile = device.compile_device();
    let budget = Budget::standard();
    let inputs = VariantKey::new().lowering_inputs(Stage::Kernel);
    let library = Library::from_air(&case.bitcode).unwrap();
    let function = FunctionDescriptor::new(library.function(&case.compute.entry).unwrap());
    let compiled = compile
        .compile_function(&function, &inputs, &budget)
        .unwrap();
    let descriptor = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
    let stage = compile
        .compile_compute_pipeline(&descriptor, &budget)
        .unwrap()
        .stage;
    let mut kinds: BTreeSet<Degradation> = stage
        .precision_notes
        .iter()
        .filter(|&&note| !stage.specialized_notes.iter().any(|s| s.note == note))
        .map(|&note| Degradation::from_precision_note(note))
        .collect();
    if stage.reconvergence == Reconvergence::ImplementationDefined {
        kinds.insert(Degradation::ImplementationDefinedReconvergence);
    }
    kinds
}

/// Each of `cases`' pipeline states made twice on `device` announces [`stage_kinds`] once each,
/// and nothing else: their union.
fn assert_stage_kinds_once(
    device: &Device,
    messages: &Messages,
    cases: &[PublicCase],
) -> BTreeSet<Degradation> {
    let mut expected = BTreeSet::new();
    for case in cases {
        expected.extend(stage_kinds(device, case));
        for _ in 0..2 {
            pipeline_state(device, case).unwrap();
        }
    }
    let counts = counts(&messages.take_notices());
    let announced: BTreeSet<Degradation> = counts.keys().copied().collect();
    assert_eq!(announced, expected, "{counts:?}");
    assert!(counts.values().all(|&n| n == 1), "{counts:?}");
    expected
}

/// Pipeline states of kernels with float functions and arithmetic, and a nearest cube sample
/// ([`NOTED`]), each made twice: the precision notes their stages record (and reconvergence that
/// is not Metal's), as the compile layer compiles them under the empty variant key, are announced
/// once each, and nothing else.
#[test]
fn pipeline_states_announce_their_stages_notes_once() {
    on_device(|device, messages| {
        let cases: Vec<PublicCase> = NOTED.map(|name| public_case(name).unwrap()).into();
        let expected = assert_stage_kinds_once(device, messages, &cases);
        assert!(!expected.is_empty(), "no stage of {NOTED:?} records a note");
    });
}

/// Kernels whose `half` samples and reads keep a near-equivalent only for some bound formats: an
/// 8-bit unorm texture's linear filter and `half` conversion, and packed 32-bit formats' reads and
/// samples, every arm of which is exact under the formats they bind (a packed class names its
/// measured family).
const SPECIALIZED: [&str; 2] = [
    "linear-filter-float-and-half-results-in-1d-and-2d-of-rgba8-unorm",
    "half-conversion-of-packed-32-bit-formats-by-read-and-sample",
];

/// The precision kinds a run of `case` on `device` announces, and the specialized notes its stage
/// records: the notes of the stage its bind shape compiles (`compile_compute_for_shape`, the
/// runtime's own route) under its plan's specialization and the values its plan pushes
/// (`StageModule::precision_notes_bound`, `BindingPlan::specialized_value`), but a note kept by
/// clauses of the bound formats alone (`SpecClause::tests_bound_formats`) only where one holds of
/// the formats `case` binds (`StageModule::bound_format_clauses`; a clause testing a slot `case`
/// binds nothing at does not hold, as the runtime reads it: the slot reads zero), and reconvergence
/// that is not Metal's.
fn run_kinds(device: &Device, case: &PublicCase) -> (BTreeSet<Degradation>, BTreeSet<Degradation>) {
    let compile = device.compile_device();
    let addressed = compile.profile().descriptors().buffer_device_address;
    let shape = compute_shape(&case.compute, addressed).unwrap();
    let library = Library::from_air(&case.bitcode).unwrap();
    let function = FunctionDescriptor::new(library.function(&case.compute.entry).unwrap());
    let dispatch = Some(&case.compute.dispatch);
    let budget = Budget::standard();
    let (pipeline, _) =
        compile_compute_for_shape(compile, &function, &shape, dispatch, None, &budget).unwrap();
    let plan = pipeline.plan(compile, &shape).unwrap();
    let stage = &pipeline.stage;
    let class = |slot: SlotRef| {
        let texture = case.compute.textures.get(&slot.index)?;
        Some(bound_texel_format(pixel_format(texture.format)))
    };
    let bound = |key| plan.specialized_value(key);
    let mut kinds: BTreeSet<Degradation> = stage
        .precision_notes_bound(&plan.specialization, &bound)
        .into_iter()
        .filter(|&note| {
            let held = stage.note_clauses_bound(note, &plan.specialization, &bound);
            held.is_empty()
                || held.iter().any(|c| !c.tests_bound_formats())
                || stage.bound_format_clauses(note, &class).iter().any(|c| {
                    c.all.iter().all(|t| match t.key {
                        SpecKey::BoundTexelFormat(slot) => {
                            case.compute.textures.contains_key(&slot.index)
                        }
                        _ => true,
                    })
                })
        })
        .map(Degradation::from_precision_note)
        .collect();
    if pipeline.stage.reconvergence == Reconvergence::ImplementationDefined {
        kinds.insert(Degradation::ImplementationDefinedReconvergence);
    }
    let specialized = pipeline
        .stage
        .specialized_notes
        .iter()
        .map(|s| Degradation::from_precision_note(s.note))
        .collect();
    (kinds, specialized)
}

/// Runs of kernels whose texture notes their bound formats decide ([`SPECIALIZED`]), each run
/// twice, announce once each exactly the precision kinds their stages keep under the
/// specialization their runs bind, and no other precision kind: the `half` conversion their stages
/// specialize (`HalfTexelConversion`, kept for a format of no measured family) is exact under an
/// 8-bit unorm texture and under the packed formats, so neither run announces it.
#[test]
fn runs_announce_the_notes_their_specialization_keeps_once() {
    on_device(|device, messages| {
        let mut expected = BTreeSet::new();
        let mut held = Vec::new();
        for name in SPECIALIZED {
            let case = public_case(name).unwrap();
            let (kinds, specialized) = run_kinds(device, &case);
            eprintln!("{name}: specialized {specialized:?}, held {kinds:?}");
            held.push((kinds.clone(), specialized));
            expected.extend(kinds);
            for _ in 0..2 {
                run_case(device, &case).unwrap();
            }
        }
        let counts = counts(&messages.take_notices());
        let announced: BTreeSet<Degradation> = counts
            .keys()
            .copied()
            .filter(|kind| {
                kind.precision_note().is_some()
                    || *kind == Degradation::ImplementationDefinedReconvergence
            })
            .collect();
        assert_eq!(announced, expected, "{counts:?}");
        assert!(counts.values().all(|&n| n == 1), "{counts:?}");
        let conversion = Degradation::from_precision_note(PrecisionNote::HalfTexelConversion);
        for (kinds, specialized) in &held {
            assert!(specialized.contains(&conversion), "{specialized:?}");
            assert!(!kinds.contains(&conversion), "{kinds:?}");
        }
    });
}

/// A kind a variant's making announces, asked for at every occurrence, is announced at each
/// dispatch that binds the variant, where the making alone would announce it once (the variant
/// is made once). One pipeline state of each kernel of float functions and arithmetic, and of a
/// nearest cube sample ([`NOTED`]), run twice: on a device of the default options each
/// precision kind its stage keeps under its run's specialization ([`run_kinds`]) is announced
/// once; on a device whose options list those kinds (`DeviceOptions::announce_each`), twice for
/// each kernel keeping it. The second run's bind announces the notices the first made: it makes
/// none (`WorkCounts::bound_notices_made`).
#[test]
fn a_kind_asked_for_at_each_occurrence_is_announced_at_each_bind_of_a_variant() {
    let cases: Vec<PublicCase> = NOTED.map(|name| public_case(name).unwrap()).into();
    // How many kernels keep each kind, and how often the runs announced it.
    let run = |device: &Device, messages: &Messages| {
        let mut kept: BTreeMap<Degradation, usize> = BTreeMap::new();
        for (case, name) in cases.iter().zip(NOTED) {
            for kind in run_kinds(device, case).0 {
                *kept.entry(kind).or_default() += 1;
            }
            let pipeline = pipeline_state(device, case).unwrap();
            run_with(device, case, &pipeline).unwrap();
            let made = device.work_counts().bound_notices_made;
            run_with(device, case, &pipeline).unwrap();
            assert_eq!(
                device.work_counts().bound_notices_made,
                made,
                "{name}: the second bind made its notices again"
            );
        }
        let announced: BTreeMap<Degradation, usize> = counts(&messages.take_notices())
            .into_iter()
            .filter(|(kind, _)| kept.contains_key(kind))
            .collect();
        (kept, announced)
    };
    let mut asked = BTreeSet::new();
    on_device(|device, messages| {
        let (kept, announced) = run(device, messages);
        assert!(!kept.is_empty(), "no stage of {NOTED:?} keeps a note");
        let once: BTreeMap<Degradation, usize> = kept.keys().map(|&kind| (kind, 1)).collect();
        assert_eq!(announced, once);
        asked = kept.into_keys().collect();
    });
    let each = |options: &mut DeviceOptions| options.announce_each = asked.clone();
    on_device_with(each, |device, messages| {
        let (kept, announced) = run(device, messages);
        let twice: BTreeMap<Degradation, usize> =
            kept.into_iter().map(|(kind, n)| (kind, 2 * n)).collect();
        assert_eq!(announced, twice);
    });
}

/// A device that withholds maximal reconvergence (`shaderMaximalReconvergence`, which every venue
/// here has) enables no such feature, and its compile device states none. A pipeline state of a
/// kernel of simdgroup shuffles ([`PARTIAL_SIMDGROUPS`]), made twice, announces its stage's kinds
/// once each and nothing else: implementation-defined reconvergence exactly where the device's
/// subgroups run the shuffles (the `Direct` or `Cluster32` tier: the GPUs). Where its kernels'
/// subgroups cannot hold Metal's 32-lane simdgroups (lavapipe's), the shuffles run emulated,
/// through threadgroup memory in uniform control flow, which keeps Metal's reconvergence.
#[test]
fn a_device_without_maximal_reconvergence_announces_it_once() {
    let feature = VkFeature::ShaderMaximalReconvergence;
    on_device_with(
        |options| {
            options.withheld_features.insert(feature);
        },
        |device, messages| {
            assert!(!device.enabled().features.contains(&feature));
            let profile = device.compile_device().profile();
            assert!(!profile.subgroups().maximal_reconvergence);
            let case = public_case(PARTIAL_SIMDGROUPS).unwrap();
            let expected = assert_stage_kinds_once(device, messages, &[case]);
            let defined = expected.contains(&Degradation::ImplementationDefinedReconvergence);
            let shuffles = SimdClass::CrossLane(OpBits::SHUFFLE);
            let tier = profile.simd_tier(&shuffles, Stage::Kernel);
            eprintln!(
                "{PARTIAL_SIMDGROUPS} without maximal reconvergence, shuffles on the {tier:?} \
                 tier: implementation-defined reconvergence: {defined}"
            );
            assert_eq!(defined, tier != Ok(Tier::Emulated), "{expected:?}");
        },
    );
}

/// The owned fixture whose vertex function runs simdgroup functions over 40 points: a sum, a
/// shuffle, a shuffle by XOR and a ballot.
const VERTEX_SIMDGROUPS: &str = "rvc_simd_vertex.ll";

/// The kind a render pipeline state of a vertex function of simdgroup functions takes on a
/// driver that runs them as Metal does without advertising the vertex stage
/// (`DriverDeviation::MetalVertexSimdgroups`).
pub(super) const UNADVERTISED_VERTEX_SIMDGROUPS: [Degradation; 1] =
    [Degradation::UnadvertisedVertexSimdgroups];

/// A render pipeline state of a vertex function of simdgroup functions ([`VERTEX_SIMDGROUPS`]),
/// made twice, announces its stage's kinds once each and nothing else:
/// `vertex-simdgroup-grouping` exactly where nothing pins the vertex stage's subgroups to Apple's
/// 32 consecutive vertices and the driver is measured to group vertices otherwise
/// (`DriverWorkaround::VertexGroupingNotApples`: ANV, 8 interleaved vertices a subgroup), and
/// [`UNADVERTISED_VERTEX_SIMDGROUPS`] exactly where its pipeline requires
/// `VkProperty::MetalVertexSimdgroups`, the profile taking the deviation and its device not
/// listing the vertex stage (MoltenVK on the measured Apple GPUs). A device without subgroup
/// operations in the vertex stage (lavapipe's) refuses the state, typed, and announces nothing.
#[test]
fn vertex_simdgroup_functions_announce_the_grouping_once() {
    on_device(|device, messages| {
        let bitcode = llvm_assemble(&engine_dir().join(VERTEX_SIMDGROUPS)).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = library
            .new_function_with_name("simd_functions_in_a_vertex_function")
            .unwrap();
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        descriptor.rasterization_enabled = false;
        let compile = device.compile_device();
        let profile = compile.profile();
        let vertex = &profile.subgroups().vertex;
        let options = PipelineOptions::default();
        if let Some(error) = device
            .new_render_pipeline_state(&descriptor, &options)
            .err()
        {
            let Error::Unsupported(u) = &error else {
                panic!("{VERTEX_SIMDGROUPS}: {error}");
            };
            assert_eq!(u.feature, Feature::SubgroupOperations, "{u:?}");
            assert!(!vertex.supported, "refused with vertex subgroups: {u:?}");
            eprintln!("{}: vertex simdgroups refused, typed: {u:?}", device.name());
            assert_eq!(counts(&messages.take_notices()), BTreeMap::new());
            return;
        }
        let compiled = compile
            .compile_render_pipeline(&descriptor, &Budget::standard())
            .unwrap();
        let mut expected: BTreeSet<Degradation> = compiled
            .stages
            .iter()
            .flat_map(|s| s.precision_notes.iter())
            .map(|&note| Degradation::from_precision_note(note))
            .collect();
        if compiled
            .stages
            .iter()
            .any(|s| s.reconvergence == Reconvergence::ImplementationDefined)
        {
            expected.insert(Degradation::ImplementationDefinedReconvergence);
        }
        let unadvertised = profile.takes_driver_deviation(DriverDeviation::MetalVertexSimdgroups)
            && !profile
                .enabled()
                .properties
                .contains(&VkProperty::SubgroupStage(Stage::Vertex));
        assert_eq!(
            compiled
                .requirements
                .properties
                .contains(&VkProperty::MetalVertexSimdgroups),
            unadvertised,
            "{:?}",
            compiled.requirements.properties
        );
        if unadvertised {
            expected.extend(UNADVERTISED_VERTEX_SIMDGROUPS);
        }
        let regroups = !vertex.pinnable
            && profile
                .workarounds()
                .contains(&DriverWorkaround::VertexGroupingNotApples);
        let grouping = Degradation::VertexSimdgroupGrouping;
        assert_eq!(expected.contains(&grouping), regroups, "{expected:?}");
        device
            .new_render_pipeline_state(&descriptor, &options)
            .unwrap();
        let counts = counts(&messages.take_notices());
        let announced: BTreeSet<Degradation> = counts.keys().copied().collect();
        assert_eq!(announced, expected, "{counts:?}");
        assert!(counts.values().all(|&n| n == 1), "{counts:?}");
        eprintln!(
            "{}: vertex simdgroups, pinnable {}, grouping announced: {regroups}, unadvertised \
             stage announced: {unadvertised}",
            device.name(),
            vertex.pinnable
        );
    });
}

/// Render pipeline states of the owned fixture `zwd_shapes`, made in turn on one device: one whose
/// fragment function reads no varying (`zwd_panel`, its position alone), then two whose fragment
/// functions read varyings the device interpolates (`zwd_path_exterior`, perspective;
/// `zwd_path_exterior_linear`, no-perspective). `device-interpolation` is announced exactly once,
/// by the first state reading one, where the profile does not take
/// `DriverDeviation::MetalInterpolation` (every driver but MoltenVK), and never where it does
/// (MoltenVK, which hands the varyings to Metal); there each such state's pipeline requires
/// `VkProperty::MetalInterpolation`, and its stages record no note. The state reading none
/// records and requires neither.
#[test]
fn interpolated_varyings_announce_the_device_interpolation_once() {
    use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
    use metal2vulkan::metal::DataType;
    use metal2vulkan::pipeline::ColorAttachmentDescriptor;
    on_device(|device, messages| {
        let bitcode = llvm_assemble(&engine_dir().join("zwd_shapes.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let constants = FunctionConstantValues::defaults().with_index(
            0,
            FunctionConstantValue::new(DataType::UInt, 0u32.to_le_bytes().to_vec()),
        );
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name_constant_values(name, &constants)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let compile = device.compile_device();
        let by_metal = compile
            .profile()
            .takes_driver_deviation(DriverDeviation::MetalInterpolation);
        let note = PrecisionNote::DeviceInterpolation;
        let options = PipelineOptions::default();
        for (vertex, fragment, interpolates) in [
            ("zwd_quad", "zwd_panel", false),
            ("zwd_path_v0", "zwd_path_exterior", true),
            ("zwd_path_v0", "zwd_path_exterior_linear", true),
        ] {
            let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
            descriptor.fragment = Some(function(fragment));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
            descriptor.color_attachments[1] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RG16Float));
            let compiled = compile
                .compile_render_pipeline(&descriptor, &Budget::standard())
                .unwrap();
            let noted = compiled
                .stages
                .iter()
                .any(|s| s.precision_notes.contains(&note));
            assert_eq!(noted, interpolates && !by_metal, "{fragment}");
            let requires = compiled
                .requirements
                .properties
                .contains(&VkProperty::MetalInterpolation);
            assert_eq!(requires, interpolates && by_metal, "{fragment}");
            device
                .new_render_pipeline_state(&descriptor, &options)
                .unwrap();
            let announced = counts(&messages.take_notices())
                .get(&Degradation::from_precision_note(note))
                .copied()
                .unwrap_or(0);
            let first = fragment == "zwd_path_exterior";
            assert_eq!(announced, usize::from(first && !by_metal), "{fragment}");
        }
        eprintln!(
            "{}: interpolates as Metal {by_metal}, device-interpolation announced: {}",
            device.name(),
            !by_metal
        );
    });
}

/// How `device` resolves a combined depth/stencil format's stencil by `SAMPLE_ZERO` while its
/// depth is not resolved, as its compile device's description states the modes (`resolve.rs`):
/// `Some(false)` in the draws' subpass, `Some(true)` one aspect at a time (a subpass whose depth
/// takes the same mode, into a scratch image), `None` not at all.
fn resolves_stencil_alone(device: &Device) -> Option<bool> {
    let limits = device.compile_device().profile().limits();
    let zero = Mode::SAMPLE_ZERO;
    if !limits.supported_stencil_resolve_modes.contains(zero) {
        return None;
    }
    if limits.independent_resolve_none || limits.independent_resolve {
        return Some(false);
    }
    limits
        .supported_depth_resolve_modes
        .contains(zero)
        .then_some(true)
}

/// Passes, each encoded twice: separate `Depth32Float` and `Stencil8` textures cleared and
/// stored; a 4x `RGBA8Unorm` target resolved (its samples stored by `MultisampleResolve`, Vulkan's
/// `DONT_CARE`) into a `RenderTarget` texture, and into a `ShaderRead` one; a 4x `RGBA16Float`
/// target resolved, and an 8x one and an 8x `RGBA8Unorm` one where the device makes them; and a
/// 4x `Depth32Float_Stencil8` target whose stencil alone resolves. Each kind is announced once and
/// nothing else, the one-aspect resolve exactly where the device's description resolves that
/// stencil one aspect at a time (lavapipe, which resolves no aspects independently), and
/// `device-rounded-resolve` exactly where the 8x half-float target, which no rule covers, is made.
#[test]
fn passes_degrade_once_per_kind() {
    let (messages, _cache, lane) = lane();
    let stencil_alone = resolves_stencil_alone(&lane.device);
    eprintln!("stencil resolved alone, one aspect at a time: {stencil_alone:?}");
    let rgba8 = PixelFormat::RGBA8Unorm;
    let combined = PixelFormat::Depth32FloatStencil8;
    let mut eight = TextureDescriptor::texture_2d(rgba8, 4, 4, false);
    eight.texture_type = TextureType::Type2DMultisample;
    eight.sample_count = 8;
    eight.usage = TextureUsage::RENDER_TARGET;
    eight.resource_options = ResourceOptions::private();
    let eight_samples = lane.device.new_texture(&eight).ok();
    eight.pixel_format = PixelFormat::RGBA16Float;
    let half_eight = lane.device.new_texture(&eight).ok();
    let exact = !lane
        .device
        .properties()
        .driver_facts
        .contains(&DriverFact::ColorResolveAverageIsMetals);
    eprintln!(
        "8 samples made: {} (RGBA16Float: {}); exact resolve at 4 samples: {exact}",
        eight_samples.is_some(),
        half_eight.is_some()
    );
    for _ in 0..2 {
        let mut separate = RenderPassDescriptor::new();
        separate.depth_attachment.texture =
            Some(lane.texture(PixelFormat::Depth32Float, ONE, TextureUsage::RENDER_TARGET));
        separate.depth_attachment.load_action = LoadAction::Clear;
        separate.depth_attachment.store_action = StoreAction::Store;
        separate.stencil_attachment.texture =
            Some(lane.texture(PixelFormat::Stencil8, ONE, TextureUsage::RENDER_TARGET));
        separate.stencil_attachment.load_action = LoadAction::Clear;
        separate.stencil_attachment.store_action = StoreAction::Store;
        lane.run(&separate, |_, _, _| {}).unwrap();
        let msaa = lane.msaa(rgba8);
        for usage in [TextureUsage::RENDER_TARGET, TextureUsage::SHADER_READ] {
            let resolve = lane.texture(rgba8, ONE, usage);
            let descriptor = color_pass(
                &msaa,
                Some(&resolve),
                StoreAction::MultisampleResolve,
                (0, 0),
            );
            lane.run(&descriptor, |_, _, _| {}).unwrap();
        }
        // A half-float resolve at 4 samples, Metal's closed rule (the exact resolve), and at 8
        // where the device makes such a texture, which no rule covers: the device's average.
        let half = PixelFormat::RGBA16Float;
        let descriptor = color_pass(
            &lane.msaa(half),
            Some(&lane.single(half)),
            StoreAction::MultisampleResolve,
            (0, 0),
        );
        lane.run(&descriptor, |_, _, _| {}).unwrap();
        if let Some(eight) = &half_eight {
            let descriptor = color_pass(
                eight,
                Some(&lane.single(half)),
                StoreAction::MultisampleResolve,
                (0, 0),
            );
            lane.run(&descriptor, |_, _, _| {}).unwrap();
        }
        // A resolve of 8 samples, where the device makes such a texture.
        if let Some(eight) = &eight_samples {
            let descriptor = color_pass(
                eight,
                Some(&lane.single(rgba8)),
                StoreAction::MultisampleResolve,
                (0, 0),
            );
            lane.run(&descriptor, |_, _, _| {}).unwrap();
        }
        let source = lane.msaa(combined);
        let mut stencil = RenderPassDescriptor::new();
        stencil.depth_attachment.texture = Some(source.clone());
        stencil.depth_attachment.load_action = LoadAction::Clear;
        stencil.depth_attachment.store_action = StoreAction::DontCare;
        let attachment = &mut stencil.stencil_attachment;
        attachment.texture = Some(source);
        attachment.load_action = LoadAction::Clear;
        attachment.store_action = StoreAction::MultisampleResolve;
        attachment.resolve_texture = Some(lane.single(combined));
        attachment.stencil_resolve_filter = MultisampleStencilResolveFilter::Sample0;
        let ran = lane.run(&stencil, |_, _, _| {});
        match stencil_alone {
            Some(_) => ran.unwrap(),
            None => assert_eq!(
                ran,
                Err(Error::NotImplemented(RuntimeGap::DepthStencilResolve))
            ),
        }
    }
    let one_aspect = stencil_alone == Some(true);
    let expected: Vec<Degradation> = PASSES
        .into_iter()
        .filter(|&kind| match kind {
            Degradation::OneAspectResolve => one_aspect,
            // Where the device's average is Metal's (MoltenVK's is Metal's own resolve) and it
            // makes no texture of 8 samples (an Apple GPU: Metal makes none past 4), every resolve
            // here is Metal's own and none rounds by the device.
            Degradation::ShaderResolve => exact || eight_samples.is_some(),
            // Only a format without a rule at its count rounds by the device: the half-float
            // resolve of 8 samples.
            Degradation::DeviceRoundedResolve => half_eight.is_some(),
            Degradation::UnmeasuredResolve => eight_samples.is_some() || half_eight.is_some(),
            _ => true,
        })
        .collect();
    let counts = assert_once(&messages.take_notices(), &expected);
    let announced: Vec<Degradation> = counts.keys().copied().collect();
    let mut expected = expected;
    expected.sort();
    assert_eq!(announced, expected, "{counts:?}");
    quiet(lane, &messages);
}

thread_local! {
    /// What the test's asking thread marks itself with before its creations; 0 elsewhere.
    static MARK: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// What a sink heard: the thread it ran on, that thread's [`MARK`], and the message.
type Heard = std::sync::Arc<std::sync::Mutex<Vec<(std::thread::ThreadId, u32, DebugMessage)>>>;

/// `f`'s answer on a new thread of the test's, a program's thread at the utility quality of
/// service on macOS, marked 0xA5 ([`MARK`]) first; with the thread's ID.
fn on_a_utility_thread<T: Send>(f: impl FnOnce() -> T + Send) -> (std::thread::ThreadId, T) {
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                #[cfg(target_os = "macos")]
                // SAFETY: sets the calling thread's own class; no pointer.
                assert_eq!(
                    unsafe {
                        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_UTILITY, 0)
                    },
                    0
                );
                MARK.with(|mark| mark.set(0xA5));
                (std::thread::current().id(), f())
            })
            .join()
            .unwrap()
    })
}

/// A program's sink that keeps what it heard in its thread's state reads it on the thread that
/// made the creation, after the call (the consumer's invariant): a creation asked for on a
/// program's thread below the user-initiated class, whose pipeline the runtime makes on a
/// thread of its own on macOS, delivers its degrade notices and the driver's and layer's messages
/// raised in it on the asking thread, the notices first.
/// - Kernels whose stages record notes ([`NOTED`], every device): each notice heard on the
///   asking thread, marked.
/// - A vertex function's simdgroup functions where the driver leaves the vertex stage out of
///   its subgroup stages (MoltenVK): the notice, then the layer's messages about the module,
///   all on the asking thread, marked; a device that refuses the state, typed, is reported.
///
/// Every other validation message fails it.
#[test]
fn a_creation_s_notices_and_messages_reach_the_sink_on_the_asking_thread() {
    let heard: Heard = std::sync::Arc::default();
    let sink = {
        let heard = std::sync::Arc::clone(&heard);
        metal2vulkan_runtime::MessageSink::new(move |message| {
            let me = std::thread::current().id();
            let mark = MARK.with(std::cell::Cell::get);
            heard.lock().unwrap().push((me, mark, message.clone()));
        })
    };
    // The device's creation notices reach this sink, not the lane's: they are checked here.
    let cache = ScratchCache::new().unwrap();
    let lane = Messages::default();
    let made = made_lane_device(TEST_DEVICE, &lane, cache.path(), |options| {
        options.messages = Some(sink);
    });
    let (device, level_named) = made.unwrap();
    {
        let device = &device;
        let take = || -> Vec<(std::thread::ThreadId, u32, DebugMessage)> {
            heard.lock().unwrap().drain(..).collect()
        };
        let created: Vec<DebugMessage> = take().into_iter().map(|(_, _, m)| m).collect();
        creation_heard_checked(device, level_named, &created, &lane).unwrap();
        // Notes announced as each state is made.
        let cases: Vec<PublicCase> = NOTED.map(|name| public_case(name).unwrap()).into();
        let (asker, ()) = on_a_utility_thread(|| {
            for case in &cases {
                pipeline_state(device, case).unwrap();
            }
        });
        let noted = take();
        assert!(
            noted.iter().any(|(_, _, m)| m.degradation.is_some()),
            "no notice heard for {NOTED:?}: {noted:?}"
        );
        for (thread, mark, message) in &noted {
            assert_eq!(
                (*thread, *mark),
                (asker, 0xA5),
                "off the asking thread: {message:?}"
            );
            assert!(message.degradation.is_some(), "not a notice: {message:?}");
        }
        // The notice, then the layer's messages about the vertex module.
        let bitcode = llvm_assemble(&engine_dir().join(VERTEX_SIMDGROUPS)).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = library
            .new_function_with_name("simd_functions_in_a_vertex_function")
            .unwrap();
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        descriptor.rasterization_enabled = false;
        let options = PipelineOptions::default();
        let (asker, made) =
            on_a_utility_thread(|| device.new_render_pipeline_state(&descriptor, &options));
        let vertex = take();
        if let Err(error) = made {
            assert!(matches!(error, Error::Unsupported(_)), "{error}");
            eprintln!(
                "{}: vertex simdgroups refused, typed: {error}; {} message(s) heard",
                device.name(),
                vertex.len()
            );
        }
        let notice = vertex.iter().position(|(_, _, m)| m.degradation.is_some());
        let layer: Vec<usize> = (0..vertex.len())
            .filter(|&at| vertex[at].2.degradation.is_none())
            .collect();
        for (thread, mark, message) in &vertex {
            assert_eq!(
                (*thread, *mark),
                (asker, 0xA5),
                "off the asking thread: {message:?}"
            );
        }
        for &at in &layer {
            let message = &vertex[at].2;
            assert!(
                message.id_name.contains("VUID-RuntimeSpirv-None-06343"),
                "a validation message: {message:?}"
            );
            assert!(
                notice.is_some_and(|n| n < at),
                "a layer message before the notice: {vertex:?}"
            );
        }
        eprintln!(
            "ASKING-THREAD {}: {} note notice(s); vertex simdgroups: notice at {notice:?}, \
                 layer messages at {layer:?}, all on the asking thread",
            device.name(),
            noted.len()
        );
    }
    drop(device);
    assert_eq!(lane.take(), Vec::new(), "the validation layer reported");
}

/// The kinds a pipeline state of `stages` announces as the runtime makes it
/// (`Device::degrade_stages` at the making), each where the device has not announced it before:
/// every precision note a stage records that no specialization removes (a specialized one is
/// announced where a draw or dispatch binds a variant keeping it), but
/// `ArgumentPixelSamplerScaling`, which a draw or dispatch listing the state announces;
/// `shader-half-color-bits` where a stage writes a `half`'s bits to a color attachment,
/// `shader-color-conversion` where one rounds a color write to Metal's code; and reconvergence
/// that is not Metal's.
fn making_kinds<'a>(stages: impl IntoIterator<Item = &'a StageModule>) -> BTreeSet<Degradation> {
    let mut kinds = BTreeSet::new();
    for stage in stages {
        for &note in &stage.precision_notes {
            let specialized = stage.specialized_notes.iter().any(|s| s.note == note);
            if !specialized && note != PrecisionNote::ArgumentPixelSamplerScaling {
                kinds.insert(Degradation::from_precision_note(note));
            }
        }
        if !stage.half_bit_colors.is_empty() {
            kinds.insert(Degradation::ShaderHalfColorBits);
        }
        if !stage.rounded_colors.is_empty() {
            kinds.insert(Degradation::ShaderColorConversion);
        }
        if stage.reconvergence == Reconvergence::ImplementationDefined {
            kinds.insert(Degradation::ImplementationDefinedReconvergence);
        }
    }
    kinds
}

/// [`making_kinds`] of a render pipeline state of `descriptor` on `device`, as the device's
/// compile layer compiles it, apart from the runtime (`compile_render_pipeline`, as
/// `color_writes::pipeline_states_announce_the_shader_conversion_once` holds the runtime to), and
/// its layout's ([`layout_kinds`]).
pub(super) fn made_render_kinds(
    device: &Device,
    descriptor: &RenderPipelineDescriptor,
) -> BTreeSet<Degradation> {
    let compiled = device
        .compile_device()
        .compile_render_pipeline(descriptor, &Budget::standard())
        .unwrap();
    let mut kinds = making_kinds(&compiled.stages);
    kinds.extend(layout_kinds(device, &compiled.layout));
    kinds
}

/// [`making_kinds`] of a mesh render pipeline state of `descriptor` on `device`, as its compile
/// layer compiles it (`compile_mesh_pipeline`), and its layout's ([`layout_kinds`]).
pub(super) fn made_mesh_kinds(
    device: &Device,
    descriptor: &MeshPipelineDescriptor,
) -> BTreeSet<Degradation> {
    let compiled = device
        .compile_device()
        .compile_mesh_pipeline(descriptor, &Budget::standard())
        .unwrap();
    let mut kinds = making_kinds(&compiled.stages);
    kinds.extend(layout_kinds(device, &compiled.layout));
    kinds
}

/// [`making_kinds`] of a compute pipeline state of kernel `function` on `device`, as its compile
/// layer compiles it (`compile_compute_pipeline`), and its layout's ([`layout_kinds`]).
pub(super) fn made_compute_kinds(
    device: &Device,
    function: FunctionDescriptor,
) -> BTreeSet<Degradation> {
    let descriptor = ComputePipelineDescriptor::new(StageSource::Function(function));
    let compiled = device
        .compile_device()
        .compile_compute_pipeline(&descriptor, &Budget::standard())
        .unwrap();
    let mut kinds = making_kinds([&compiled.stage]);
    kinds.extend(layout_kinds(device, &compiled.layout));
    kinds
}

/// The kinds the making of a pipeline state of `layout` announces on `device` by its layout:
/// `set-past-layout-support` where a set of it is one the device's driver refuses
/// ([`refused_sets`]), which the runtime makes only where the driver runs what it refuses
/// (`RuntimeDriverDefect::SetLayoutSupportUnderstated`; elsewhere it refuses the state, typed).
pub(super) fn layout_kinds(device: &Device, layout: &PipelineLayout) -> BTreeSet<Degradation> {
    forced(&[(
        Degradation::SetPastLayoutSupport,
        !refused_sets(device, layout).is_empty(),
    )])
}

/// The sets of `layout`, each by its number and descriptor count, whose layout `device`'s driver
/// refuses through `vkGetDescriptorSetLayoutSupport`, asked here apart from the runtime of each
/// set's binding types, counts, stages and flags as the layout states them.
pub(super) fn refused_sets(device: &Device, layout: &PipelineLayout) -> Vec<(u32, u64)> {
    // SAFETY: a device-level query on the device's own handle, made while it lives.
    let raw = unsafe { device.raw() };
    layout
        .sets
        .iter()
        .filter_map(|set| {
            let bindings: Vec<ash::vk::DescriptorSetLayoutBinding<'_>> = set
                .bindings
                .iter()
                .map(|binding| {
                    ash::vk::DescriptorSetLayoutBinding::default()
                        .binding(binding.binding)
                        .descriptor_type(binding.ty.into())
                        .descriptor_count(binding.count)
                        .stage_flags(binding.stages.into())
                })
                .collect();
            let flags: Vec<ash::vk::DescriptorBindingFlags> =
                set.bindings.iter().map(|b| b.flags.into()).collect();
            let mut flags_info =
                ash::vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
            let mut info = ash::vk::DescriptorSetLayoutCreateInfo::default()
                .flags(set.flags.into())
                .bindings(&bindings);
            if flags.iter().any(|f| !f.is_empty()) {
                info = info.push_next(&mut flags_info);
            }
            let mut support = ash::vk::DescriptorSetLayoutSupport::default();
            // SAFETY: `info` and every array it points at are live for the call, and `support`
            // heads an empty chain.
            unsafe {
                raw.device
                    .get_descriptor_set_layout_support(&info, &mut support)
            };
            let total = set.bindings.iter().map(|b| u64::from(b.count)).sum();
            (support.supported != ash::vk::TRUE).then_some((set.set, total))
        })
        .collect()
}

/// The kinds a dispatch of kernel `function`'s compute pipeline state on `device` may announce
/// beyond its making's ([`made_compute_kinds`]): those of the specialized notes its stage records
/// with a clause that tests a value the function reads as data (`SpecClause::tests_bound_values`:
/// a bound texture's format class). No pipeline's key carries those values, so the runtime
/// announces such a kind at a dispatch whose pushed values hold one of its clauses
/// (`degrade_bound_values`), and only there; a test that dispatches with values it does not model
/// (slots it binds nothing at) takes each as allowed, not forced ([`assert_once_allowing`]).
pub(super) fn bound_value_kinds(
    device: &Device,
    function: FunctionDescriptor,
) -> BTreeSet<Degradation> {
    let descriptor = ComputePipelineDescriptor::new(StageSource::Function(function));
    let compiled = device
        .compile_device()
        .compile_compute_pipeline(&descriptor, &Budget::standard())
        .unwrap();
    compiled
        .stage
        .specialized_notes
        .iter()
        .filter(|s| s.when.iter().any(|c| c.tests_bound_values()))
        .map(|s| Degradation::from_precision_note(s.note))
        .collect()
}

/// The notices `messages` holds announce each kind once: exactly `kinds`, and those of `first`
/// the device had not announced before they were taken ([`Messages::taken_kinds`]), a device
/// announcing a kind once. `first` holds the kinds the steps since the last take announce on
/// their first occurrence: their pipeline state makings' ([`made_render_kinds`],
/// [`made_compute_kinds`]) and the runtime's, each where the steps or a
/// device fact force it ([`forced`]). [`assert_once`] of the expectation.
pub(super) fn assert_made_once(
    messages: &Messages,
    kinds: &[Degradation],
    first: &BTreeSet<Degradation>,
) -> BTreeMap<Degradation, usize> {
    let expected = expected_once(messages, kinds, first);
    assert_once(&messages.take_notices(), &expected)
}

/// [`assert_made_once`], the notices it took returned, for their words.
pub(super) fn made_once(
    messages: &Messages,
    kinds: &[Degradation],
    first: &BTreeSet<Degradation>,
) -> Vec<DebugMessage> {
    let expected = expected_once(messages, kinds, first);
    let notices = messages.take_notices();
    assert_once(&notices, &expected);
    notices
}

/// `kinds`, and those of `first` not announced before.
pub(super) fn expected_once(
    messages: &Messages,
    kinds: &[Degradation],
    first: &BTreeSet<Degradation>,
) -> Vec<Degradation> {
    let before = messages.taken_kinds();
    let mut expected: Vec<Degradation> = kinds.to_vec();
    for kind in first.difference(&before) {
        if !expected.contains(kind) {
            expected.push(*kind);
        }
    }
    expected
}

/// The kinds of `kinds` whose condition holds: each paired with the device fact, or the test's
/// own step, that forces the runtime to announce it.
pub(super) fn forced(kinds: &[(Degradation, bool)]) -> BTreeSet<Degradation> {
    kinds
        .iter()
        .filter(|(_, holds)| *holds)
        .map(|(kind, _)| *kind)
        .collect()
}

/// Whether `device` announces `depth-denormals` for a write that may store a denormal 32-bit
/// float depth in a depth image (`on_host`: a host copy, which stores the bits as they are): its
/// driver is not measured to store one as it is (`DriverFact::DepthImageKeepsDenormals`) or to
/// test a stored one as Metal does (`DriverFact::DepthTestKeepsDenormals`).
pub(super) fn loses_depth_denormals(device: &Device, on_host: bool) -> bool {
    let facts = &device.properties().driver_facts;
    let stored = on_host || facts.contains(&DriverFact::DepthImageKeepsDenormals);
    !(stored && facts.contains(&DriverFact::DepthTestKeepsDenormals))
}

/// Whether `device` announces `restricted-depth-upload` for an upload into a 32-bit float depth
/// image: it lacks `VK_EXT_depth_range_unrestricted`, where Vulkan leaves a value outside
/// `[0, 1]` undefined, and its driver is not measured to keep one
/// (`DriverFact::DepthImageKeepsValuesOutsideUnitRange`).
pub(super) fn clamps_depth_uploads(device: &Device) -> bool {
    let unrestricted = device
        .enabled()
        .extensions
        .contains(&VkExtension::ExtDepthRangeUnrestricted);
    let kept = device
        .properties()
        .driver_facts
        .contains(&DriverFact::DepthImageKeepsValuesOutsideUnitRange);
    !unrestricted && !kept
}

/// Whether `device` keeps a negative or NaN 32-bit float depth a write stores in a render
/// target's depth image (not [`clamps_depth_uploads`]) and its depth test compares it as it is,
/// where Metal's compares its magnitude (no `DriverFact::DepthTestComparesMagnitudes`): a
/// single-sample `Depth32Float` target then renders a pass that loads it over its magnitude
/// stand-in (`texture/signed_depth.rs`), and a write into a multisampled or combined
/// depth/stencil one announces `signed-depth-test`.
pub(super) fn signs_depth_tests(device: &Device) -> bool {
    let magnitudes = device
        .properties()
        .driver_facts
        .contains(&DriverFact::DepthTestComparesMagnitudes);
    !clamps_depth_uploads(device) && !magnitudes
}

/// Whether `device` announces `shared-in-system-memory` for the first `Shared` buffer of the
/// default CPU cache mode a test makes: it has no unified memory (`blits::SYSTEM_MEMORY`).
pub(super) fn shared_in_system_memory(device: &Device) -> bool {
    !device.properties().has_unified_memory
}

/// Whether `device` gives a single-sample `Depth32Float` texture of shader-read usage without
/// render target usage a depth twin (`depth-twin`): its depth images may store a value otherwise
/// than Metal's ([`clamps_depth_uploads`], or a driver not measured to keep a denormal there,
/// `DriverFact::DepthImageKeepsDenormals`).
pub(super) fn twins_depth(device: &Device) -> bool {
    let keeps = device
        .properties()
        .driver_facts
        .contains(&DriverFact::DepthImageKeepsDenormals);
    clamps_depth_uploads(device) || !keeps
}

/// The kind `device` announces for a resolve of a color attachment of `format` at `samples`, a
/// texture the runtime made: `shader-resolve` under `ResolvePolicy::Exact` where the format has
/// Metal's rules at that count (`ColorResolveFormat::of` and `resolves`) and the device's
/// `AVERAGE` is not measured to be Metal's there (`DriverFact::ColorResolveAverageIsMetals`, at
/// most 4 samples): the runtime gives such a target the input attachment usage the exact resolve
/// reads it through where that costs the image nothing, which every driver measured answers so
/// (lavapipe, RADV, ANV and NVIDIA); `device-rounded-resolve` where the device's
/// `AVERAGE` resolves it (one with rules under `ResolvePolicy::Device`, the default, and one
/// without rules there under either) and is not measured to resolve as Metal's; none where it is
/// (the fact holds for every format).
pub(super) fn resolve_kind(
    device: &Device,
    format: PixelFormat,
    samples: u32,
) -> Option<Degradation> {
    // Rules that hold at that count: the float formats' at 2 and 4 samples only.
    let rules = metal2vulkan::metal::format::direct_format(format)
        .and_then(metal2vulkan::bind::ColorResolveFormat::of)
        .is_some_and(|rules| rules.resolves(samples));
    let metals = samples <= 4
        && device
            .properties()
            .driver_facts
            .contains(&DriverFact::ColorResolveAverageIsMetals);
    let policy = device.compile_device().policy().resolve;
    // A format without rules, or any under the device's resolve, resolves by the device's
    // average, Metal's own where the fact holds.
    match (
        metals,
        rules && samples > 1 && policy == ResolvePolicy::Exact,
    ) {
        (true, _) => None,
        (false, true) => Some(Degradation::ShaderResolve),
        (false, false) => Some(Degradation::DeviceRoundedResolve),
    }
}

/// Whether `device` copies bytes into a multisampled 32-bit float depth (from a buffer, or from a
/// colour texture) by a fragment-depth draw its driver is not measured to store exactly
/// (`drawn-depth-copy`): it lacks maintenance8's copies of such images, or its driver copies them
/// wrong (`RuntimeDriverDefect::MultisampledDepthStencilColourCopies`), and its fragment depth is
/// not measured to store every value unchanged (`DriverFact::FragmentDepthStoresUnchanged`).
pub(super) fn draws_depth_copies(device: &Device) -> bool {
    let properties = device.properties();
    let defect = metal2vulkan::error::RuntimeDriverDefect::MultisampledDepthStencilColourCopies;
    let copies = device.enabled().features.contains(&VkFeature::Maintenance8)
        && !properties.driver_defects.contains(&defect);
    let exact = properties
        .driver_facts
        .contains(&DriverFact::FragmentDepthStoresUnchanged);
    !copies && !exact
}

/// Whether `device` holds a texture of `pf` of its own in a carrier with a decoded twin
/// (`carried-decoded-twin`), and whether the twin's codes announce `decoded-texel-precision`: an
/// ASTC format the device has no block for (`astc_textures::decodes_astc`), a PVRTC, ETC2 or
/// EAC format it has no format for, or a 4:2:2 format (every device); the notice for an ASTC LDR
/// format's 8-bit codes and an EAC 11-bit format's 16-bit ones under
/// `DecodedTexelsPolicy::Compact`, and for no other twin, whose codes are exact (an ASTC sRGB
/// format's, a PVRTC format's, an 8-bit ETC2 or EAC format's, a 4:2:2 format's).
pub(super) fn twins_carried(device: &Device, pf: PixelFormat) -> [bool; 2] {
    let astc = AstcDecode::of(pf).filter(|_| !decodes_astc(device, pf));
    let usage = [metal2vulkan::device::FormatUsage::SampledImage].into();
    let profile = device.compile_device().profile();
    let decoded = metal2vulkan::metal::format::vk_format(pf, &usage, profile)
        .is_ok_and(|m| m.emulation == metal2vulkan::metal::format::FormatEmulation::Decoded);
    let pvrtc = metal2vulkan::bind::PvrtcDecode::of(pf).is_some() && decoded;
    let etc = metal2vulkan::bind::EtcDecode::of(pf).filter(|_| decoded);
    let chroma = metal2vulkan::bind::ChromaDecode::of(pf).is_some() && decoded;
    let eleven = etc.is_some_and(metal2vulkan::bind::EtcDecode::eleven_bit);
    let compact = device.compile_device().policy().decoded_texels == DecodedTexelsPolicy::Compact;
    let ldr = astc.is_some_and(|decode| !decode.srgb);
    [
        astc.is_some() || pvrtc || etc.is_some() || chroma,
        (ldr || eleven) && compact,
    ]
}

/// Whether `device` gives a single-sample texture of `pf` a decoded twin (`decoded-block-twin`): a
/// BC1 to BC5 format whose decode its driver is not measured to make as Metal's (BC1 to BC3
/// `DriverFact::ColorBlockDecodeIsMetals`, BC4 and BC5 `DriverFact::ChannelBlockDecodeIsMetals`).
/// Every venue's twin formats take sampling, linear filtering and copies, which the twin needs.
pub(super) fn twins_blocks(device: &Device, pf: PixelFormat) -> bool {
    use metal2vulkan::bind::BlockDecode as B;
    let fact = match B::of(pf) {
        Some(B::Bc1 | B::Bc2 | B::Bc3) => DriverFact::ColorBlockDecodeIsMetals,
        Some(_) => DriverFact::ChannelBlockDecodeIsMetals,
        None => return false,
    };
    !device.properties().driver_facts.contains(&fact)
}

/// Whether `device` announces `compressed-texel-decode` for a texture of `pf` it makes in memory
/// of its own: a BC6H or BC7 texture, which takes no decoded twin, on a driver not measured to
/// decode it as Metal does (`DriverFact::Bc6hBlockDecodeIsMetals`, `Bc7BlockDecodeIsMetals`). A BC1
/// to BC5 texture takes a twin instead ([`twins_blocks`]).
pub(super) fn decodes_blocks_its_way(device: &Device, pf: PixelFormat) -> bool {
    use PixelFormat as P;
    let fact = match pf {
        P::BC6HRGBFloat | P::BC6HRGBUfloat => DriverFact::Bc6hBlockDecodeIsMetals,
        P::BC7RGBAUnorm | P::BC7RGBAUnormSRGB => DriverFact::Bc7BlockDecodeIsMetals,
        _ => return false,
    };
    !device.properties().driver_facts.contains(&fact)
}

/// Whether `device` announces `biased-depth-past-unit-range` for a draw of triangles in depth
/// clip mode under a depth bias that can move their depth past `[0, 1]`: it enabled
/// `VK_EXT_depth_range_unrestricted`, so it stores such a depth unclamped, without the native
/// depth range (`viewport_ranges::native_depth_range`), whose clamp would keep Metal's.
pub(super) fn unclamped_bias(device: &Device) -> bool {
    device
        .enabled()
        .extensions
        .contains(&VkExtension::ExtDepthRangeUnrestricted)
        && !super::viewport_ranges::native_depth_range(device)
}
