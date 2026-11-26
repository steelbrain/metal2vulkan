//! The Metal API on Vulkan, with one path for every device.
//!
//! Every difference between devices comes from a Vulkan query: a feature, a property, a format, a
//! limit or a memory type. The crate holds no vendor, device or driver identity, no environment
//! gate and no runtime policy switch; `tests::one_path` enforces this over its sources.
//!
//! It builds only on `metal2vulkan`'s public API, as a separate crate must.

#![warn(missing_docs)]

mod alias;
mod arena;
mod arrays;
mod blit;
mod block;
mod bound;
mod buffer;
mod cache;
mod capture;
mod coherence;
mod command;
mod complete;
mod copy_out;
mod counters;
#[cfg(windows)]
mod d3d11;
mod depth_stencil;
mod device;
mod dgc;
mod encoder;
mod error;
mod event;
mod export;
mod extension;
mod fault;
mod fence;
mod fill;
pub mod floor;
mod function_table;
mod hazard;
mod host_bytes;
mod import;
mod indirect;
mod instance;
mod kept;
mod kernel_map;
mod memory;
mod over;
mod own;
mod pass;
mod pipeline;
mod pixel_format_info;
mod places;
mod queue;
mod readback;
mod reaper;
mod record;
mod render;
mod render_encoder;
mod ring;
mod sampler;
mod shadow;
mod stage;
mod staging;
mod submit;
mod submit_calls;
mod surface;
#[cfg(test)]
mod tests;
mod texture;
mod times;
mod visibility;

pub use arrays::{IndexClass, IndexGrant, ResourceId};
pub use blit::{BlitCommandEncoder, BlitOption, BufferLayout, TexturePlace};
pub use block::Block;
pub use buffer::{Buffer, MAX_BUFFER_LENGTH};
pub use command::{CommandBuffer, CommandBufferError, CommandBufferStatus, DeviceLoss};
pub use counters::{
    Counters, Gauges, Metrics, OverCause, OverCopyKind, SiteAllocations, Timed, VariantMake,
    VariantStart,
};
#[cfg(windows)]
pub use d3d11::D3d11Plane;
pub use depth_stencil::DepthStencilState;
pub use device::{Device, DeviceOptions, RawDevice};
pub use encoder::{ComputeCommandEncoder, Dispatch, DispatchType, Resource, ResourceUsage};
pub use error::{
    DrawKind, Error, Gap, IndirectCommandsKind, KernelInputKind, Limit, LinearPlanePassKind,
    PartialThreadgroupsSite, Refusal, RefusalBasis, RenderPassKind, RenderPipelineKind, StageGap,
    StageInKind, StorageElementCase, TextureBacking, TextureViewKind,
};
pub use event::{Event, SharedEvent};
pub use export::{Export, ExportHandle};
pub use extension::{ConsumerFeature, DeviceExtension};
pub use fault::{DeviceFault, FaultAddress, FaultPlace, FaultRange, RangeFreed, RangeKind};
pub use fence::Fence;
pub use function_table::{VisibleFunctionTable, VisibleFunctionTableDescriptor};
pub use host_bytes::HostBytes;
pub use import::ImportHandle;
pub use indirect::{IndirectCommandBuffer, IndirectCommandBufferDescriptor, IndirectCommandType};
pub use instance::{Instance, PhysicalDevice};
pub use memory::{AllocationId, AllocationInfo, BlockId, BlockInfo, BlockMappings, MemoryClass};
/// `MTLLibrary`: the translator's, read from AIR or a metallib.
/// `MTLSamplerDescriptor`: the translator's.
pub use metal2vulkan::bind::SamplerDescriptor;
pub use metal2vulkan::library::Library;
/// `MTLDepthStencilDescriptor` and its `MTLStencilDescriptor`: the translator's.
pub use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
/// `MTLBindingAccess`, which a pipeline state's texture slots answer: the translator's.
pub use metal2vulkan::metal::BindingAccess;
/// `MTLReadWriteTextureTier` and `MTLArgumentBuffersTier`: the translator's.
pub use metal2vulkan::metal::{ArgumentBuffersTier, ReadWriteTextureTier};
/// `MTLSamplerMinMagFilter`, `MTLSamplerMipFilter`, `MTLSamplerAddressMode`,
/// `MTLSamplerBorderColor`, `MTLCompareFunction` and `MTLSamplerReductionMode`: the translator's.
pub use metal2vulkan::metal::{
    CompareFunction, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter, SamplerMipFilter,
    SamplerReductionMode,
};
/// `MTLPrimitiveType`, `MTLIndexType`, `MTLCullMode`, `MTLWinding`, `MTLStencilOperation`,
/// `MTLTriangleFillMode` and `MTLDepthClipMode`:
/// the translator's.
pub use metal2vulkan::metal::{
    CullMode, DepthClipMode, IndexType, PrimitiveType, StencilOperation, TriangleFillMode, Winding,
};
/// `MTLPixelFormat`, `MTLTextureType`, `MTLTextureUsage` and `MTLTextureSwizzleChannels` with its
/// `MTLTextureSwizzle`: the translator's.
pub use metal2vulkan::metal::{
    PixelFormat, TextureSwizzle, TextureSwizzleChannels, TextureType, TextureUsage,
};
/// A function's stage, which a render pipeline state's texture slots are asked by: the
/// translator's.
pub use metal2vulkan::pipeline::Stage;
/// `MTLRenderPipelineDescriptor`, its `MTLRenderPipelineColorAttachmentDescriptor` and its
/// `MTLVertexDescriptor`: the translator's.
pub use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, VertexDescriptor,
};
/// `MTLComputePipelineDescriptor`, its kernel, a pipeline's precision notes, and the samplers a
/// pipeline state numbers: the translator's.
pub use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, FunctionDescriptor, NumberedSampler, PrecisionNote, StageSource,
};
/// `MTLFunctionHandle`, the word a visible-function table's entry holds, and a function of
/// `MTLLinkedFunctions`, which a compute pipeline state's handle is asked of: the translator's.
pub use metal2vulkan::pipeline::{FunctionHandle, LinkedFunction};
pub use pass::{
    ClearColor, LoadAction, MultisampleDepthResolveFilter, MultisampleStencilResolveFilter,
    RenderPassColorAttachmentDescriptor, RenderPassDepthAttachmentDescriptor, RenderPassDescriptor,
    RenderPassStencilAttachmentDescriptor, StoreAction, VisibilityResultType,
    MAX_COLOR_ATTACHMENTS,
};
pub use pipeline::{ComputePipelineState, DispatchVariant};
pub use pixel_format_info::{PixelFormatInfo, TextureLevelInfo, TextureLevelQuery};
pub use queue::CommandQueue;
pub use render::RenderPipelineState;
pub use render_encoder::{
    IndexBuffer, RenderCommandEncoder, ScissorRect, Viewport, VisibilityResultMode,
};
pub use sampler::SamplerState;
pub use stage::{serve_stage, StageBreach, StageFill, StageRequest, StageSpans};
pub use surface::{
    PlaneLayout, PlaneSpec, Surface, SurfaceCopy, SurfaceExport, SurfaceForm, SurfaceSpec,
};
pub use texture::{
    BytesPlace, HeapRange, Region, SizeAndAlign, StorageMode, Texture, TextureDescriptor,
};
pub use times::{host_time, HOST_DOMAIN};
