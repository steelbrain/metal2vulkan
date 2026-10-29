//! The Metal API on Vulkan (`docs/RUNTIME-PORTING.md`): Metal's objects and calls, run on a Vulkan
//! device through `ash`.
//!
//! Unstable until its pinned table is frozen. It builds only on `metal2vulkan`'s public API
//! (`device`, `library`, `pipeline`, `bind`, `cache`, `metal`, `vk`, `error`, `ash_interop`), as a
//! separate crate must.
//!
//! - Objects keep Metal's names without the `MTL` prefix, and methods are Metal's selectors in
//!   snake_case. Handles ([`Device`], [`CommandQueue`], [`CommandBuffer`], [`Buffer`],
//!   [`Library`], [`Function`], [`ComputePipelineState`], [`Event`], [`SharedEvent`]) are cheap
//!   clonable `Send + Sync` values; encoders are owned, `Send` and not `Sync`.
//! - Where Metal raises, aborts, or answers nil, a call returns a typed [`Error`].
//! - Each file owns one object or mechanism; a later step adds its own files and one `mod` and
//!   one `pub use` line here.

// As in `metal2vulkan`: `too_many_arguments` and `type_complexity` are threshold heuristics that
// fire pervasively here, where encoding and planning functions take many typed parameters and
// their return and table shapes are nested domain types. Factoring every such signature behind a
// wrapper struct or a one-off `type` alias would add indirection without improving clarity, so
// both are accepted crate-wide rather than scattered per-site.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]
// Every public item says what it is and its Metal counterpart: a new undocumented one fails the
// warnings-denied gates.
#![warn(missing_docs)]

mod address;
mod address_map;
mod address_pool;
mod amplification;
mod apple_only;
mod argument_encoder;
mod argument_table;
mod astc;
mod attachment_stand_in;
mod attachmentless;
mod attributes;
mod batch_order;
mod binary_archive;
mod binding;
mod binding_view;
mod blit;
mod blit_fill;
mod blit_indirect;
mod blit_plan;
mod blit_texture;
mod buffer;
mod buffer_descriptor;
mod buffer_texture;
mod capabilities;
mod chroma_pairs;
mod churn_counts;
mod clear_reach;
mod clear_value;
mod color_list;
mod color_reformat;
mod color_resolve;
mod command_buffer;
mod compressed_blit;
mod compute;
mod contexts;
mod copied;
mod copy_plan;
mod creation;
mod deferred;
mod degradation;
mod depth_copy;
mod depth_stand_in;
mod depth_stencil;
mod depth_stencil_resolve;
mod depth_stencil_scratch;
mod derived;
mod descriptor_arena;
mod detach;
mod device;
#[cfg(unix)]
mod dirty_pages;
mod dispatch_execute;
mod dma_buf;
mod draw_source;
mod driver_defects;
mod driver_facts;
#[doc(hidden)]
mod dynamic_library;
mod dynamic_state;
mod enables;
mod encoder;
mod error;
mod etc;
mod event;
mod event_timeline;
mod exactness;
mod execution;
mod families;
#[doc(hidden)]
pub use execution::{ExecutedPipeline, Execution};
mod exit_guard;
pub use exit_guard::process_exiting;
#[doc(hidden)]
pub use exit_guard::{registrations as exit_registrations, Registrations as ExitRegistrations};
mod executed_commands;
mod fence;
mod fetch_reach;
mod fill_value;
mod fragment_imageblock;
mod framebuffer_fetch;
mod freed_memory;
mod fresh_pages;
mod fresh_unwritten;
mod function_table;
#[doc(hidden)]
pub mod generated_commands;
mod generated_execute;
mod gpu_execute;
mod gpu_pools;
mod gpu_time;
mod heap;
mod heap_sets;
mod host_aliases;
mod host_bytes;
mod host_memory;
#[cfg(unix)]
mod host_residency;
#[cfg(unix)]
mod host_shared_pointers;
mod image_barrier;
mod image_plan;
mod in_place;
mod in_place_check;
mod index_bounds;
mod index_rewrite;
mod indirect_command_buffer;
mod indirect_dispatch;
mod indirect_execute;
mod interop;
mod kept_bytes;
mod key_hash;
mod library;
mod linear_attachment;
mod linear_merge;
mod linear_plan;
mod live_bind;
mod live_buffers;
mod loaded;
mod loader_word;
mod managed;
mod memory;
mod memory_log;
mod metal_release;
mod mipmap;
mod mipmap_filter;
mod mipmap_pick;
mod mipmap_plan;
mod mipmap_scratch;
mod misplaced_notices;
mod nan_clear;
mod neutral;
mod noted_apart;
mod owned_kernel;
mod pending_dispatch;
mod pipeline;
mod pixel_format_info;
mod predication;
mod private_properties;
mod program_objects;
mod pvrtc;
mod queue;
mod range;
mod rate_map;
mod rate_mapped_pass;
mod read_only_view;
mod reader_sizes;
mod rebind;
mod recorder;
mod recycle;
mod release;
mod released_programs;
mod render;
mod render_pass;
mod render_state;
mod render_target;
mod render_values;
mod replay;
mod residency;
mod resolve;
mod resource;
mod resource_ids;
mod retained;
mod rotated;
mod row_copies;
mod sample_copy;
mod sample_interleave;
mod sample_planes;
mod sample_positions;
mod sampler;
mod scheduler;
mod shader_cache;
mod shadow;
mod shadow_copy_back;
mod shadow_copy_in;
mod shadow_merge;
mod shadow_stencil;
mod sharing;
mod sorted_chunks;
mod spares;
mod sparse;
mod stencil_copy;
mod storage_placeholder;
mod tessellation;
mod texture;
mod texture_descriptor;
mod tile_pass;
mod tracker;
mod trails;
mod two_planes;
mod unique_identifier;
mod upload;
mod used_resources;
mod values;
mod versioned;
mod view_plan;
mod visibility;
mod word_fill;
mod work_counts;
mod wrapped_buffer;
mod ycbcr;
mod ycbcr_device;

#[cfg(test)]
mod tests;
mod written_commands;

pub use address::{AddressPlacement, AddressRange, GpuAddress, PlacementConflict, RangeToken};
pub use address_map::{AddressModeMap, MappedSampling};
pub use address_pool::AddressPool;
pub use amplification::VertexAmplificationViewMapping;
pub use apple_only::{AppleOnly, AppleOnlyCall, AppleOnlyGap, DEFAULT_PROTECTION_OPTIONS};
pub use argument_encoder::{ArgumentEncoder, ArgumentEncoderProblem};
pub use attributes::PurgeableState;
pub use binary_archive::{BinaryArchive, BinaryArchiveDescriptor};
pub use blit::BlitCommandEncoder;
pub use buffer::Buffer;
pub use buffer_descriptor::{BufferContents, BufferDescriptor};
pub use buffer_texture::IoSurfacePlane;
pub use capabilities::{DeviceProperties, Query};
pub use churn_counts::{ObjectChurn, ShadowKeeps};
pub use command_buffer::checks::{Submission, SubmissionCheck};
pub use command_buffer::{CommandBuffer, CommitFailure, FailedEncoders, Handler};
pub use compute::ComputeCommandEncoder;
pub use copy_plan::{BlitOption, CopyProblem};
pub use degradation::{Degradation, DegradationCategory, DegradeNotice};
pub use degradation::{MisplacedNotice, NoticePlace};
pub use depth_stencil::DepthStencilState;
pub use device::{
    DebugMessage, Device, DeviceHandles, DeviceKind, DeviceOptions, LayerSetting,
    LayerSettingValue, MessageKind, MessageSeverity, MessageSink, PhysicalDeviceIdentity,
    RawDevice,
};
pub use dirty_pages::{DirtyPages, DirtyStamp};
#[cfg(unix)]
pub use dma_buf::{DmaBufChunk, DmaBufRequest, DmaBufSource};
pub use dynamic_library::DynamicLibrary;
pub use error::DynamicLibraryError;
pub use error::ExtensionFeature;
pub use error::FailureOrigin;
pub use error::InstanceExtensionRequired;
pub use error::MetalError;
pub use error::Nil;
pub use error::Unproven;
pub use error::{
    ArchivedPipeline, BinaryArchiveError, BinaryArchiveMiss, CommandBufferError, DeadlockCause,
    DriverSetting, EncoderCall, Error, InstanceApiVersion, Invalid, LoaderNotFound, MemoryKind,
    MeshRecordsOverflow, Problem, RuntimeGap, SamplerLimit, ShaderCacheError,
    TessellationRecordsOverflow, TimedOut, VkResultCode,
};
pub use error::{FailedPipeline, FailedState, PipelineCreationCause};
pub use event::{
    Event, EventRef, SharedEvent, SharedEventHandle, SharedEventListener, SharedEventNotification,
};
pub use exactness::Exactness;
pub use executed_commands::{
    ExecutedCommand, ExecutedCommands, ExecutedCommandsHandler, RecordsRead,
};
pub use families::{
    FamilyRow, FeatureRow, FormatCapabilities, FormatCapability, FormatRow, Framework, GpuFamily,
    LimitRow, Refusal, RowAudit, RowState,
};
pub use fence::Fence;
pub use function_table::{
    FunctionHandle, FunctionTableProblem, VisibleFunctionTable, VisibleFunctionTableDescriptor,
};
pub use gpu_time::host_time;
pub use heap::{Heap, HeapDescriptor, HeapProblem, HeapType, SizeAndAlign};
pub use host_bytes::HostBytes;
pub use host_memory::{Deallocator, HostMemory, ImportGate, ImportRequest, KeepAlive};
pub use indirect_command_buffer::{
    EncodedComputeCommand, EncodedRenderCommand, IndirectBufferBinding, IndirectCommand,
    IndirectCommandBuffer, IndirectCommandBufferDescriptor, IndirectCommandBufferProblem,
    IndirectCommandCheck, IndirectCommandCheckNeeds, IndirectComputeCommand, IndirectDispatch,
    IndirectDraw, IndirectRenderCommand, NIL_PATCH_BUFFER_ADDRESS,
};
pub use interop::SemaphoreProblem;
pub use library::{Function, FunctionDescriptor, Library};
pub use linear_plan::BufferTextureProblem;
pub use metal2vulkan::device::DriverFact;
pub use metal_release::MACOS_RELEASE;
pub use mipmap_plan::MipmapProblem;
#[doc(hidden)]
pub use pipeline::budget::{set_making_budget_for_measurement, MakingBudgetStarted};
#[doc(hidden)]
pub use pipeline::objects::makes_layouts_past_support;
pub use pipeline::{
    BinaryFunctionProblem, BinaryFunctionStage, RenderPipelineFunctionsDescriptor,
    RequiredThreadsProblem,
};
pub use pipeline::{ComputePipelineState, PipelineOption, PipelineOptions, RenderPipelineState};
pub use pixel_format_info::{PixelFormatInfo, TextureLevelInfo, TextureLevelQuery};
pub use private_properties::{PrivateAnswer, PrivateCount, PrivateFlag, PrivateProperties};
pub use queue::CommandQueue;
pub use rate_map::{RasterizationRateMap, RateMapProblem};
pub use rate_mapped_pass::RegionRefusal;
pub use rebind::RebindProblem;
#[doc(hidden)]
pub use recorder::CommandsRecorded;
pub use render::{DrawProblem, RenderCommandEncoder, ResolveTarget};
pub use render_pass::{
    Attachment, RenderPassColorAttachmentDescriptor, RenderPassDepthAttachmentDescriptor,
    RenderPassDescriptor, RenderPassProblem, RenderPassStencilAttachmentDescriptor,
    MAX_COLOR_ATTACHMENTS,
};
pub use render_state::RenderStateProblem;
pub use render_values::{
    ClearColor, DrawPrimitiveType, LoadAction, MultisampleDepthResolveFilter,
    MultisampleStencilResolveFilter, SamplePosition, ScissorRect, StoreAction, StoreActionOptions,
    Viewport,
};
pub use resource::{ObjectId, ResourceRef};
pub use resource_ids::{IdPlacement, ResourceId, ResourceIdKind};
pub use sampler::{sampler_border_color, SamplerState, SamplerStateOptions};
pub use shader_cache::{
    ShaderCacheOptions, ShaderCacheStatistics, DEFAULT_CAPACITY, DEFAULT_MEMORY_CAPACITY,
};
pub use sparse::device::SparseFeatures;
pub use sparse::encoder::ResourceStateCommandEncoder;
pub use sparse::{SparsePageSize, SparseProblem, SparseTextureRegionAlignmentMode};
pub use texture::{SharedTextureHandle, Texture, VkImageLayouts, WrappedImageProblem};
pub use texture_descriptor::{TextureDescriptor, TextureProblem, TextureSwizzleChannels};
pub use used_resources::{EncoderKind, EndedEncoder, UsedResources};
pub use values::{
    CommandBufferStatus, Coordinate2D, CpuCacheMode, HazardTrackingMode, Origin, Region,
    ResourceOptions, Size, StorageMode,
};
pub use view_plan::ViewProblem;
pub use visibility::{VisibilityProblem, VisibilityResultMode};
pub use work_counts::{CopySite, MemoryFrees, WorkCounts};
pub use wrapped_buffer::{VkBufferMemory, WrappedBufferProblem};
pub use ycbcr::{IoSurfaceYCbCr, SurfaceYCbCrMatrix};
