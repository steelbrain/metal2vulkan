//! The Metal side of the case vocabulary: native Metal through `objc2-metal`, on macOS only.
//!
//! This is what record mode runs. It exists so a case's Metal answer is taken by running the case
//! rather than copied off a terminal: the same case body, the same sites, the same transcript
//! shape.
//!
//! A shader reaches Metal as its fixture's typed-pointer AIR text: `xcrun metal-as` reassembles it
//! and `xcrun metallib` packages the library, which `newLibraryWithURL:` loads. Apple's assembler
//! reads that dialect and not the one a current `llvm-as` writes, which is why a fixture commits
//! both forms.
//!
//! Metal announces no degrade notice, so this side announces none: a notice in a transcript is
//! ours, and a case that holds one must name what justifies it.
use crate::archives::{
    Added, Bytes, Created, DynamicSource, LibraryAt, Observation, Program, Scratch, Step,
};
use crate::argument_encoders::{ArgumentDescriptorSpec, ArgumentWrite, EncoderFacts};
use crate::case::{
    Answer, ArgumentSlot, Bindings, CommandDraw, CoordinateProbe, Failure, FragmentFill,
    FragmentFilled, Handle, MappedCoordinates, MeshCommand, MeshFunctions, MeshReflectionSpec,
    MeshStep, PassOutcome, RateMapFacts, RateMapForm, RateMapLayer, RateMapRequest, SamplerSpec,
    ShaderSource, Side, StageInRegion, StageInput, TextureRole, TextureSpec, TileFacts,
    TilePipelineSpec, TileStep, Venue,
};
use crate::command_buffer_facts::CommandBufferFacts;
use crate::compute_descriptors::{
    ComputeDescriptorFacts, DescriptorSpec as ComputeDescriptorSpec,
    SuppliedLists as ComputeSuppliedLists, AWAY as COMPUTE_AWAY,
};
use crate::compute_pass::{
    ComputePass, ComputePassOutcome, ComputeStep, Dispatching, PassResource,
};
use crate::device_queries::{Box6, HeapBufferQuery, RegionQuery};
use crate::function_bindings::{
    ArrayFacts, BindingFacts, BufferFacts, MemberFacts, PayloadFacts, PointerFacts, ReferenceFacts,
    StageBindings, StageReflection, StructFacts, TextureFacts, ThreadgroupFacts, TypeFacts,
};
use crate::function_facts::{
    AttributeFacts, ConstantAt, ConstantFacts, ConstantValue, FunctionFacts,
};
use crate::heap_query::{HeapBufferSpec, HeapOwnFacts, HeapSpec, QueriedResource, ResourceFacts};
use crate::indirect_commands::{
    CommandsMade, ComputeCommandStep, EncodedCommand, IcbDescriptor, RenderCommandStep,
};
use crate::library_facts::{FunctionDescriptorSpec, LibraryFacts};
use crate::mesh_descriptors::{
    DescriptorSpec as MeshDescriptorSpec, MeshDescriptorFacts, SuppliedLists as MeshSuppliedLists,
    AWAY as MESH_AWAY,
};
use crate::pass_descriptors::{
    AttachmentFacts, Indices, Numbers, PassDescriptorFacts, PassSpec, Reads, Row, Shape,
    ASSIGNED_COLOR, DEPTH_AT, MUTATED_AFTER, READ_COLORS as PASS_COLORS, STENCIL_AT, SWEPT_COLOR,
};
use crate::pipeline_facts::{ComputeStateFacts, ComputeStateSpec, RenderStateFacts};
use crate::pipeline_functions::{FunctionsDescriptorFacts, FunctionsDescriptorSpec, Write};
use crate::pipeline_handlers::{
    BothPaths, Delivered, Form as PipelineForm, Handled, StateFacts,
    IMAGEBLOCK as HANDLER_IMAGEBLOCK, WAIT as HANDLER_WAIT,
};
use crate::pipeline_option_reflection::UnderOptions;
use crate::pipeline_options::{OptionOutcome, FRAGMENT_STAGE, KERNEL_STAGE, VERTEX_STAGE};
use crate::queue_slots::QueueSlots;
use crate::render::{
    Dispatch, Plane, RenderPass, RenderPipelineSpec, RenderStep, RenderWork, StageFunction,
    StepResource, TargetSpec, VertexStride,
};
use crate::render_descriptors::{
    AttributeFacts as DescriptorAttributeFacts, ColorAway, ColorFacts, DescriptorSpec, LayoutFacts,
    LinkedCounts, RenderDescriptorFacts, SuppliedLists, VertexDescriptorFacts, AWAY, READ_BUFFERS,
    READ_COLORS, READ_VERTEX_SLOTS,
};
use crate::sparse_mapping::{
    MappingMode, MappingStep, SparseMapped, SparseMappingProgram, TileRect,
};
use crate::sparse_query::{
    HeapFacts, QueryPage, SparseHeapSpec, SparseTextureFacts, SparseTextureSpec, SparseTileQuery,
    TileSize,
};
use crate::texture_query::{HeapTextureQuery, SizeAndAlign, Storage};
use crate::texture_views::{SharedHandleFacts, ViewCall, ViewFacts, ViewRequest};
use crate::tile_descriptors::{
    DescriptorSpec as TileDescriptorSpec, SuppliedLists as TileSuppliedLists, TileDescriptorFacts,
    AWAY as TILE_AWAY,
};
use crate::transcript::Value;
use crate::visible_tables::{
    Asked, Binding, Entry, HandleFacts, LinkedList as SpecLinkedList, Made as MadeTable, Maker,
    TableFacts, TableOutcome, TableSpec, Writes,
};
use metal2vulkan::library::LibraryType;
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{
    BindingAccess, BindingType, FunctionOptions, FunctionType, PatchType, PixelFormat, RenderStages,
};
use metal2vulkan_runtime::{PurgeableState, SparsePageSize};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{AnyThread, ClassType};
use objc2_foundation::{
    NSArray, NSDictionary, NSError, NSObjectProtocol, NSRange, NSString, NSUInteger, NSURL,
};
use objc2_metal::MTLBlitOption;
use objc2_metal::{
    MTLArgumentDescriptor, MTLArgumentEncoder, MTLArrayType, MTLAttribute, MTLAttributeFormat,
    MTLBarrierScope, MTLBinaryArchive, MTLBinaryArchiveDescriptor, MTLBinding, MTLBindingAccess,
    MTLBindingType, MTLBlendFactor, MTLBlendOperation, MTLBlitCommandEncoder, MTLBuffer,
    MTLBufferBinding, MTLCPUCacheMode, MTLClearColor, MTLColorWriteMask, MTLCommandBuffer,
    MTLCommandBufferHandler, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompareFunction, MTLComputeCommandEncoder, MTLComputePipelineDescriptor,
    MTLComputePipelineReflection, MTLComputePipelineState, MTLCreateSystemDefaultDevice,
    MTLCullMode, MTLDataType, MTLDepthClipMode, MTLDepthStencilDescriptor, MTLDepthStencilState,
    MTLDevice, MTLDispatchType, MTLDynamicLibrary, MTLEvent, MTLFence, MTLFunction,
    MTLFunctionConstant, MTLFunctionConstantValues, MTLFunctionDescriptor, MTLFunctionHandle,
    MTLFunctionOptions, MTLHazardTrackingMode, MTLHeap, MTLHeapDescriptor, MTLHeapType,
    MTLIndexType, MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor,
    MTLIndirectCommandType, MTLIndirectComputeCommand, MTLIndirectRenderCommand, MTLLibrary,
    MTLLinkedFunctions, MTLLoadAction, MTLMeshRenderPipelineDescriptor,
    MTLMultisampleDepthResolveFilter, MTLMultisampleStencilResolveFilter, MTLMutability,
    MTLObjectPayloadBinding, MTLOrigin, MTLPipelineBufferDescriptor,
    MTLPipelineBufferDescriptorArray, MTLPipelineOption, MTLPixelFormat, MTLPointerType,
    MTLPrimitiveTopologyClass, MTLPrimitiveType, MTLPurgeableState,
    MTLRasterizationRateLayerDescriptor, MTLRasterizationRateMap,
    MTLRasterizationRateMapDescriptor, MTLRegion, MTLRenderCommandEncoder,
    MTLRenderPassAttachmentDescriptor, MTLRenderPassColorAttachmentDescriptor,
    MTLRenderPassColorAttachmentDescriptorArray, MTLRenderPassDepthAttachmentDescriptor,
    MTLRenderPassDescriptor, MTLRenderPassStencilAttachmentDescriptor,
    MTLRenderPipelineColorAttachmentDescriptor, MTLRenderPipelineColorAttachmentDescriptorArray,
    MTLRenderPipelineDescriptor, MTLRenderPipelineFunctionsDescriptor, MTLRenderPipelineReflection,
    MTLRenderPipelineState, MTLRenderStages, MTLResource, MTLResourceOptions,
    MTLResourceStateCommandEncoder, MTLResourceUsage, MTLSamplePosition, MTLSamplerAddressMode,
    MTLSamplerBorderColor, MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerMipFilter,
    MTLSamplerReductionMode, MTLSamplerState, MTLScissorRect, MTLSharedEvent, MTLSharedEventHandle,
    MTLSize, MTLSparsePageSize, MTLSparseTextureMappingMode, MTLSparseTextureRegionAlignmentMode,
    MTLStageInputOutputDescriptor, MTLStencilDescriptor, MTLStencilOperation, MTLStepFunction,
    MTLStorageMode, MTLStoreAction, MTLStoreActionOptions, MTLStructMember, MTLStructType,
    MTLTessellationControlPointIndexType, MTLTessellationFactorFormat,
    MTLTessellationFactorStepFunction, MTLTessellationPartitionMode, MTLTexture, MTLTextureBinding,
    MTLTextureDescriptor, MTLTextureReferenceType, MTLTextureSwizzle, MTLTextureSwizzleChannels,
    MTLTextureType, MTLTextureUsage, MTLTextureViewDescriptor, MTLThreadgroupBinding,
    MTLTileRenderPipelineColorAttachmentDescriptor,
    MTLTileRenderPipelineColorAttachmentDescriptorArray, MTLTileRenderPipelineDescriptor,
    MTLTriangleFillMode, MTLVertexAttribute, MTLVertexAttributeDescriptor,
    MTLVertexAttributeDescriptorArray, MTLVertexBufferLayoutDescriptor,
    MTLVertexBufferLayoutDescriptorArray, MTLVertexDescriptor, MTLVertexFormat,
    MTLVertexStepFunction, MTLViewport, MTLVisibilityResultMode, MTLVisibleFunctionTable,
    MTLVisibleFunctionTableDescriptor, MTLWinding,
};
/// The Metal level this side answers at: the host's own, since Metal answers as the release it runs
/// on.
const METAL_LEVEL: &str = crate::recording::ORACLE_OS_VERSION;

/// Native Metal, with the objects the case has made.
pub struct MetalSide {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    libraries: Vec<Retained<ProtocolObject<dyn MTLLibrary>>>,
    pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    textures: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
    /// The sampler states the case has made. Its own pool: a sampler state is its own Metal
    /// object, not another name for one of the pools above.
    samplers: Vec<Retained<ProtocolObject<dyn MTLSamplerState>>>,
    command_buffers: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    /// The render pipeline states the case has made, each with the reflection its creation
    /// answered where the case asked for one. One pool, because a mesh state, a tile state and a
    /// graphics state are one Metal object (`MTLRenderPipelineState`) made from three descriptors.
    render_pipelines: Vec<RenderPipeline>,
    /// The indirect command buffers the case has made.
    command_buffers_indirect: Vec<Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>>,
    /// The rasterization-rate maps the case has made. Its own pool: a rate map is its own
    /// Metal object, not another name for one of the pools above.
    rate_maps: Vec<Retained<ProtocolObject<dyn MTLRasterizationRateMap>>>,
    /// For each argument buffer the case has made, the textures it names: Metal keeps a resource a
    /// kernel reaches through an argument buffer resident only where the encoder was told so, so
    /// binding the buffer makes them resident. Not a pool — the buffer itself lives in `buffers`.
    argument_residents: Vec<(Handle, Vec<Handle>)>,
    /// The depth and stencil states the case has made. Its own pool: an `MTLDepthStencilState` is
    /// its own Metal object, not another name for one of the pools above.
    depth_stencil_states: Vec<Retained<ProtocolObject<dyn MTLDepthStencilState>>>,
    /// The heaps the case has made. Its own pool: an `MTLHeap` is its own Metal object, not
    /// another name for one of the pools above, and not another name for the depth and stencil
    /// states either.
    heaps: Vec<Retained<ProtocolObject<dyn MTLHeap>>>,
    /// The argument encoders the case has made, each with the argument buffer it writes into. Its
    /// own pool: an `MTLArgumentEncoder` is its own Metal object, not another name for one of the
    /// pools above.
    argument_encoders: Vec<ArgumentEncoderAt>,
    /// **Every IOSurface this side made, and the only thing that owns one.** Not a pool a handle
    /// names: a surface is not a case's object at all — the case states a buffer, or a plane's
    /// rotation and stride, and this side turns it into the surface Metal needs. They are kept so
    /// a surface outlives the texture that reads it.
    ///
    /// The two families that need a surface — a two-plane YCbCr plane texture and a rotated
    /// framebuffer-only target — are **one mechanism here**, which is why there is one pool. What
    /// differs between them is the layout asked for and the geometry it comes from, and that lives
    /// in [`crate::metal_surface`]'s two constructors, not in a second owner.
    surfaces: Vec<crate::metal_surface::Surface>,
    /// Which surface a texture handle's plane reads and writes, and how long that plane is.
    ///
    /// **A lookup, not a pool**: it owns nothing and releases nothing, and the surface it names
    /// lives in `surfaces` above. It exists because a rotated target's case writes the plane
    /// before its pass and reads it after, so that side has to find the surface again from the
    /// texture handle — which a plane texture's case never does, since its bytes go in before the
    /// surface is handed to Metal and come back out through the texture.
    planes: Vec<PlaneAt>,
    /// The scratch directory the metallibs are built in, removed when the side is dropped.
    scratch: Scratch,
    /// The host pages this side mapped for a no-copy buffer ([`Side::host_buffer`]), one per such
    /// buffer, in the order they were made.
    ///
    /// **Declared after every resource pool on purpose**, because fields drop in declaration
    /// order: a no-copy buffer's pages have to stay mapped for as long as the buffer, and
    /// `buffers` above is dropped before this is. Not a pool a handle names — the buffer itself
    /// lives in `buffers`, and what this adds is the mapping under it.
    host_pages: Vec<crate::host_pages::HostPages>,
}

/// Where one texture handle's plane lies: which surface in [`MetalSide::surfaces`] holds it, and
/// how many bytes of that surface the plane is.
///
/// The index is into that vector, which only ever grows while a side lives, so an index stays the
/// surface it named. The length is the **case's** — its stride times its rows — and not
/// `IOSurfaceGetAllocSize`, which is routinely larger: measured here a 16x64 plane asked for a
/// 272-byte stride came back in 32,768 bytes, 15,360 more than the plane holds. Comparing the
/// allocation would compare bytes no case ever stated.
#[derive(Clone, Copy)]
struct PlaneAt {
    handle: Handle,
    at: usize,
    length: usize,
}

/// One render pipeline state of the case's, with the reflection its creation answered.
///
/// The reflection lives here, beside the state, rather than in a pool of its own, because **Metal
/// answers a reflection only at the state's creation**: `MTLRenderPipelineState`'s own
/// `reflection` property is nil for a state an `MTLDevice` made. So a handle has to stand for the
/// pair, and the state is kept so it outlives its own reflection. It is `None` for a state whose
/// creation was not asked for one, which is every state but
/// [`Side::render_pipeline_with_reflection`]'s.
struct RenderPipeline {
    state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    reflection: Option<Retained<MTLRenderPipelineReflection>>,
}

impl RenderPipeline {
    /// A state whose creation was not asked for a reflection.
    fn unreflected(state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>) -> Self {
        Self {
            state,
            reflection: None,
        }
    }
}

/// One argument encoder of the case's, with where its arguments go.
///
/// The target is kept beside the encoder because `constantDataAtIndex:` answers an **address**,
/// and the fact a case compares is its offset inside the arguments: to take that offset the side
/// needs the argument buffer the encoder was given and where in it the arguments start.
struct ArgumentEncoderAt {
    encoder: Retained<ProtocolObject<dyn MTLArgumentEncoder>>,
    /// The argument buffer and the offset of the arguments in it; `None` until one is set.
    target: Option<(Handle, u64)>,
}

/// The array of Objective-C pointers one of `MTLArgumentEncoder`'s range setters takes, with a
/// null for each nil the case named, and the range it covers.
///
/// Its own function so the eight range setters share one spelling: each takes
/// `NonNull<*const ProtocolObject<dyn T>>` over the whole range, and a hand-written array per
/// setter is eight chances to get a length or a first index wrong.
fn pointer_array<T: ?Sized>(
    objects: &[Option<Retained<ProtocolObject<T>>>],
) -> Vec<*const ProtocolObject<T>> {
    objects
        .iter()
        .map(|object| match object {
            Some(object) => &**object as *const ProtocolObject<T>,
            None => std::ptr::null(),
        })
        .collect()
}

/// The range a range setter writes: `elements.len()` arguments from `first`.
///
/// Refused where the case named no element at all: Metal's array setters read `range.length`
/// pointers from the array they are handed, and an empty slice has no address to hand them. No
/// case states an empty range, and this says so rather than passing a dangling pointer.
fn range_of<T>(first: u32, elements: &[T]) -> Answer<NSRange> {
    if elements.is_empty() {
        return Err(Failure::named(
            "EmptyRange",
            "a range setter of no element".to_owned(),
        ));
    }
    Ok(NSRange::new(first as usize, elements.len()))
}

/// `count` commands from `first` of `commands` as Metal's range, refused where they leave the
/// buffer.
///
/// The blit encoder's three indirect-command calls take a range past the buffer's commands without
/// complaint — a copy that runs past the destination corrupts its command 0 on plain Metal, and
/// Metal's own validation stops neither — so the side refuses such a range here rather than asking
/// a device what a corruption reads as. `what` names which buffer a refusal is about.
fn commands_range(
    commands: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
    first: u64,
    count: u64,
    what: &str,
) -> Answer<NSRange> {
    let held = commands.size() as u64;
    if first.checked_add(count).is_none_or(|end| end > held) {
        return Err(Failure::named(
            "PastTheCommands",
            format!("{count} commands at {first} of {what}, of {held}"),
        ));
    }
    Ok(NSRange::new(first as usize, count as usize))
}

/// A binding as the subprotocol it conforms to, with the conformance **asked of the object**
/// rather than assumed from its `type`.
///
/// `objc2` offers no cast between two protocol objects: `ProtocolObject::from_ref` needs a static
/// conformance, which `dyn MTLBinding` to `dyn MTLBufferBinding` is not, and a protocol object
/// cannot be downcast. The alternative is `msg_send!`, which sends a selector with no receiver type
/// at all — and that is how `MTLTextureBinding`'s `arrayLength` came to be read by
/// `texture_accesses` while the coverage document still counted every one of its items unexercised:
/// a selector sent without a receiver is a call nothing, including that derivation, can check. So
/// the cast is spelled once, here, and `conformsToProtocol:` makes its precondition a test rather
/// than a comment.
fn conforming<'a, P: ?Sized>(
    binding: &'a ProtocolObject<dyn MTLBinding>,
    protocol: &std::ffi::CStr,
) -> Answer<&'a ProtocolObject<P>> {
    let declared = objc2::runtime::AnyProtocol::get(protocol)
        .ok_or_else(|| Failure::named("NoSuchProtocol", protocol.to_string_lossy().into_owned()))?;
    if !binding.conformsToProtocol(declared) {
        return Err(Failure::named(
            "NotConforming",
            format!(
                "a binding of MTLBindingType({}) does not conform to {}",
                binding.r#type().0,
                protocol.to_string_lossy()
            ),
        ));
    }
    // SAFETY: `ProtocolObject<P>` is a `repr(transparent)` wrapper over the same Objective-C
    // object whatever `P` is, and the object was just asked, and answered, that it conforms to
    // `P`'s protocol, so every selector `P` declares is one it responds to.
    Ok(unsafe { &*std::ptr::from_ref(binding).cast::<ProtocolObject<P>>() })
}

/// The translator's `MTLDataType` for a raw Metal ordinal.
fn our_data_type(raw: usize) -> Answer<metal2vulkan::metal::DataType> {
    metal2vulkan::metal::DataType::try_from(raw as u32)
        .map_err(|_| Failure::named("UnknownDataType", format!("MTLDataType({raw})")))
}

/// The translator's `MTLFunctionType` for a raw Metal ordinal.
fn our_function_type(raw: usize) -> Answer<FunctionType> {
    FunctionType::try_from(raw as u32)
        .map_err(|_| Failure::named("UnknownFunctionType", format!("MTLFunctionType({raw})")))
}

/// The translator's `MTLPatchType` for a raw Metal ordinal.
fn our_patch_type(raw: usize) -> Answer<PatchType> {
    PatchType::try_from(raw as u32)
        .map_err(|_| Failure::named("UnknownPatchType", format!("MTLPatchType({raw})")))
}

/// One `MTLVertexAttribute`'s six facts, in the vocabulary both sides answer in.
///
/// # Why this and [`stage_input_attribute_facts`] are two functions, each taking its own class
///
/// `MTLVertexAttribute` and `MTLAttribute` are two Objective-C classes that state the same six
/// facts and share no protocol, so there is nothing to be generic over, and one `macro_rules!` over
/// the six getters would read better than two near-identical bodies. It would also hide them: the
/// coverage derivation reads the leg's calls out of `fn` bodies, and a call written inside a macro
/// sits in no function at all, so both classes would be exercised by a case and credited to
/// nothing.
///
/// And the attribute arrives as a **parameter** rather than as a `for` loop's variable, which is
/// the shape [`metal_binding`] already uses for the same reason: the derivation types a receiver
/// from a `let`, a parameter or a closure's parameter, and a loop variable types as nothing, so
/// six getters read straight off one would resolve to no item either.
fn vertex_attribute_facts(attribute: &MTLVertexAttribute) -> Answer<AttributeFacts> {
    Ok(AttributeFacts {
        name: attribute.name().to_string(),
        attribute_index: attribute.attributeIndex() as u32,
        attribute_type: our_data_type(attribute.attributeType().0)?,
        active: attribute.isActive(),
        patch_data: attribute.isPatchData(),
        patch_control_point_data: attribute.isPatchControlPointData(),
    })
}

/// One `MTLAttribute`'s six facts. [`vertex_attribute_facts`] says why the two are not one.
fn stage_input_attribute_facts(attribute: &MTLAttribute) -> Answer<AttributeFacts> {
    Ok(AttributeFacts {
        name: attribute.name().to_string(),
        attribute_index: attribute.attributeIndex() as u32,
        attribute_type: our_data_type(attribute.attributeType().0)?,
        active: attribute.isActive(),
        patch_data: attribute.isPatchData(),
        patch_control_point_data: attribute.isPatchControlPointData(),
    })
}

/// `vertexAttributes`' list in the vocabulary both sides answer in; `None` is Metal's nil.
fn our_vertex_attributes(
    list: Option<&NSArray<MTLVertexAttribute>>,
) -> Answer<Option<Vec<AttributeFacts>>> {
    let Some(list) = list else { return Ok(None) };
    let mut found = Vec::new();
    for attribute in list {
        found.push(vertex_attribute_facts(&attribute)?);
    }
    Ok(Some(found))
}

/// `stageInputAttributes`' list in the vocabulary both sides answer in; `None` is Metal's nil.
fn our_stage_input_attributes(
    list: Option<&NSArray<MTLAttribute>>,
) -> Answer<Option<Vec<AttributeFacts>>> {
    let Some(list) = list else { return Ok(None) };
    let mut found = Vec::new();
    for attribute in list {
        found.push(stage_input_attribute_facts(&attribute)?);
    }
    Ok(Some(found))
}

/// The `MTLFunctionConstantValues` the case's values build, each through the setter its
/// [`ConstantAt`] names.
fn constant_values(values: &[ConstantValue]) -> Retained<MTLFunctionConstantValues> {
    let built = MTLFunctionConstantValues::new();
    for value in values {
        let bytes = std::ptr::NonNull::from(value.bytes.as_slice()).cast::<std::ffi::c_void>();
        let data_type = MTLDataType(value.data_type.as_raw() as usize);
        // SAFETY: each pointer is to `value.bytes`, which outlives the call, and each holds the
        // bytes the case states for the type and the count it states. The index, the range and
        // the name are the case's own; Metal checks them against the function's own constants
        // when the pipeline state compiles, and a value naming no constant is ignored.
        unsafe {
            match &value.at {
                ConstantAt::Index(index) => {
                    built.setConstantValue_type_atIndex(bytes, data_type, *index as usize);
                }
                ConstantAt::Range { first, count } => {
                    built.setConstantValues_type_withRange(
                        bytes,
                        data_type,
                        NSRange::new(*first as usize, *count as usize),
                    );
                }
                ConstantAt::Name(name) => {
                    built.setConstantValue_type_withName(
                        bytes,
                        data_type,
                        &NSString::from_str(name),
                    );
                }
            }
        }
    }
    built
}

/// Every `MTLFunction` getter the vocabulary reads, off a function however it was made.
///
/// Shared by `newFunctionWithName:`, `newFunctionWithName:constantValues:error:` and
/// `newFunctionWithDescriptor:error:`, so the three ways of making a function are read through one
/// walk and a case comparing them against each other compares the same getters.
fn our_function_facts(
    function: &ProtocolObject<dyn MTLFunction>,
    registry: u64,
) -> Answer<FunctionFacts> {
    // Each list bound to a local of its declared type, as the bindings walk does: the coverage
    // derivation types a receiver from what bound it, so a call made inside a `for` header or on
    // an `Option`'s contents names no type at all.
    let vertex: Option<Retained<NSArray<MTLVertexAttribute>>> = function.vertexAttributes();
    let stage_in: Option<Retained<NSArray<MTLAttribute>>> = function.stageInputAttributes();
    let dictionary: Retained<NSDictionary<NSString, MTLFunctionConstant>> =
        function.functionConstantsDictionary();
    let mut found = Vec::new();
    for key in dictionary.allKeys() {
        let constant: Retained<MTLFunctionConstant> = dictionary
            .objectForKey(&key)
            .ok_or_else(|| Failure::named("NoSuchConstant", key.to_string()))?;
        found.push(ConstantFacts {
            name: constant.name().to_string(),
            data_type: our_data_type(constant.r#type().0)?,
            index: constant.index() as u32,
            required: constant.required(),
        });
    }
    // A dictionary has no order, so the facts are sorted here and the comparison is of two sorted
    // lists: `allKeys` is free to answer in any order and a case still compares the same constant
    // against the same constant.
    found.sort_by(|left, right| left.name.cmp(&right.name));
    let count = function.patchControlPointCount();
    Ok(FunctionFacts {
        name: function.name().to_string(),
        function_type: our_function_type(function.functionType().0)?,
        patch_type: our_patch_type(function.patchType().0)?,
        // Metal answers -1 where there is no count, which the vocabulary spells `None`.
        patch_control_point_count: u32::try_from(count).ok(),
        vertex_attributes: our_vertex_attributes(vertex.as_deref())?,
        stage_input_attributes: our_stage_input_attributes(stage_in.as_deref())?,
        constants: found,
        options: FunctionOptions::try_from(function.options().0 as u32)
            .map_err(|error| Failure::named("UnknownFunctionOptions", error.to_string()))?,
        // The registry ID is each host's own number, so only the **comparison** is portable: the
        // two sides agree that the function came from the case's device, never on the number that
        // says so.
        device_is_the_cases: function.device().registryID() == registry,
    })
}

/// The translator's `MTLLibraryType` for a raw Metal ordinal.
///
/// The ordinals are `MTLLibraryType`'s own (0 and 1), not the metallib header's `file_type` byte
/// (0 and 2) that our side reads the type out of.
fn our_library_type(raw: isize) -> Answer<LibraryType> {
    match raw {
        0 => Ok(LibraryType::Executable),
        1 => Ok(LibraryType::Dynamic),
        other => Err(Failure::named(
            "UnknownLibraryType",
            format!("MTLLibraryType({other})"),
        )),
    }
}

/// The translator's `MTLBindingAccess` for a raw Metal ordinal.
fn our_access(raw: usize) -> Answer<BindingAccess> {
    BindingAccess::try_from(raw as u32)
        .map_err(|_| Failure::named("UnknownBindingAccess", format!("MTLBindingAccess({raw})")))
}

/// The translator's `MTLTextureType` for a raw Metal ordinal.
fn our_texture_type(raw: usize) -> Answer<metal2vulkan::metal::TextureType> {
    metal2vulkan::metal::TextureType::try_from(raw as u32)
        .map_err(|_| Failure::named("UnknownTextureType", format!("MTLTextureType({raw})")))
}

/// One binding of Metal's reflection, in the vocabulary both sides answer in.
///
/// Which subprotocol to read is decided by the binding's own `type`, which is how Metal states it:
/// `MTLBindingTypeBuffer` is the fact that the binding conforms to `MTLBufferBinding`.
/// The `name` of each function of one of an `MTLRenderPipelineFunctionsDescriptor`'s three
/// properties, in order; a nil property has none, which is the fold
/// [`crate::pipeline_functions`] states.
fn function_names(list: Option<&NSArray<ProtocolObject<dyn MTLFunction>>>) -> Vec<String> {
    let Some(list) = list else {
        return Vec::new();
    };
    list.iter()
        .map(|function| function.name().to_string())
        .collect()
}

fn metal_binding(binding: &ProtocolObject<dyn MTLBinding>) -> Answer<BindingFacts> {
    let kind = binding.r#type();
    let mut facts = BindingFacts {
        name: binding.name().to_string(),
        binding_type: BindingType::try_from(kind.0 as u32).map_err(|_| {
            Failure::named("UnknownBindingType", format!("MTLBindingType({})", kind.0))
        })?,
        access: our_access(binding.access().0)?,
        index: binding.index() as u32,
        used: binding.isUsed(),
        argument: binding.isArgument(),
        buffer: None,
        threadgroup: None,
        texture: None,
        payload: None,
    };
    if kind == MTLBindingType::Buffer {
        let held: &ProtocolObject<dyn MTLBufferBinding> = conforming(binding, c"MTLBufferBinding")?;
        let struct_type = held.bufferStructType();
        let pointer_type = held.bufferPointerType();
        facts.buffer = Some(BufferFacts {
            alignment: held.bufferAlignment() as u64,
            data_size: held.bufferDataSize() as u64,
            data_type: our_data_type(held.bufferDataType().0)?,
            struct_type: struct_type
                .as_deref()
                .map(|held| metal_struct(held, crate::function_bindings::DEPTH))
                .transpose()?,
            pointer_type: pointer_type
                .as_deref()
                .map(|held| metal_pointer(held, crate::function_bindings::DEPTH))
                .transpose()?,
        });
    }
    if kind == MTLBindingType::ThreadgroupMemory {
        let held: &ProtocolObject<dyn MTLThreadgroupBinding> =
            conforming(binding, c"MTLThreadgroupBinding")?;
        facts.threadgroup = Some(ThreadgroupFacts {
            alignment: held.threadgroupMemoryAlignment() as u64,
            data_size: held.threadgroupMemoryDataSize() as u64,
        });
    }
    if kind == MTLBindingType::Texture {
        let held: &ProtocolObject<dyn MTLTextureBinding> =
            conforming(binding, c"MTLTextureBinding")?;
        facts.texture = Some(TextureFacts {
            texture_type: our_texture_type(held.textureType().0)?,
            data_type: our_data_type(held.textureDataType().0)?,
            depth: held.isDepthTexture(),
            array_length: held.arrayLength() as u32,
        });
    }
    if kind == MTLBindingType::ObjectPayload {
        let held: &ProtocolObject<dyn MTLObjectPayloadBinding> =
            conforming(binding, c"MTLObjectPayloadBinding")?;
        facts.payload = Some(PayloadFacts {
            alignment: held.objectPayloadAlignment() as u64,
            data_size: held.objectPayloadDataSize() as u64,
        });
    }
    Ok(facts)
}

/// All five stage lists of one `MTLRenderPipelineReflection`, whichever creation answered it.
///
/// Each list is bound to a local rather than called in the `for` header, for the reason
/// [`Side::compute_pipeline_bindings`](crate::case::Side::compute_pipeline_bindings) states: the
/// coverage derivation reads a receiver by walking back from the dot, and in a `for` header that
/// walk swallows the loop variable.
fn metal_stage_bindings(reflection: &MTLRenderPipelineReflection) -> Answer<StageBindings> {
    let read = |bindings: &NSArray<ProtocolObject<dyn MTLBinding>>| -> Answer<Vec<BindingFacts>> {
        let mut found = Vec::new();
        for binding in bindings {
            found.push(metal_binding(&binding)?);
        }
        Ok(found)
    };
    let vertex = reflection.vertexBindings();
    let fragment = reflection.fragmentBindings();
    let tile = reflection.tileBindings();
    let object = reflection.objectBindings();
    let mesh = reflection.meshBindings();
    Ok(StageBindings {
        vertex: read(&vertex)?,
        fragment: read(&fragment)?,
        tile: read(&tile)?,
        object: read(&object)?,
        mesh: read(&mesh)?,
    })
}

/// A struct type of Metal's: its members, and whether `memberByName:` finds each of them.
fn metal_struct(held: &MTLStructType, depth: u32) -> Answer<StructFacts> {
    let members = held.members();
    let mut found = Vec::new();
    let mut by_name = true;
    for member in &members {
        let name = member.name();
        by_name &= held
            .memberByName(&name)
            .is_some_and(|other| other.name().to_string() == name.to_string());
        found.push(metal_member(&member, depth)?);
    }
    Ok(StructFacts {
        members: found,
        found_by_name: by_name,
    })
}

fn metal_member(member: &MTLStructMember, depth: u32) -> Answer<MemberFacts> {
    Ok(MemberFacts {
        name: member.name().to_string(),
        offset: member.offset() as u64,
        data_type: our_data_type(member.dataType().0)?,
        argument_index: member.argumentIndex() as u32,
        kind: match depth {
            0 => TypeFacts::Deeper,
            _ => metal_detail(
                member.structType().as_deref(),
                member.arrayType().as_deref(),
                member.textureReferenceType().as_deref(),
                member.pointerType().as_deref(),
                depth - 1,
            )?,
        },
    })
}

/// Which of the four type getters answered, in the order Metal declares them.
fn metal_detail(
    held_struct: Option<&MTLStructType>,
    held_array: Option<&MTLArrayType>,
    held_reference: Option<&MTLTextureReferenceType>,
    held_pointer: Option<&MTLPointerType>,
    depth: u32,
) -> Answer<TypeFacts> {
    if let Some(held) = held_struct {
        return Ok(TypeFacts::Struct(metal_struct(held, depth)?));
    }
    if let Some(held) = held_array {
        return Ok(TypeFacts::Array(Box::new(metal_array(held, depth)?)));
    }
    if let Some(held) = held_reference {
        return Ok(TypeFacts::TextureReference(metal_reference(held)?));
    }
    if let Some(held) = held_pointer {
        return Ok(TypeFacts::Pointer(Box::new(metal_pointer(held, depth)?)));
    }
    Ok(TypeFacts::Plain)
}

/// What a texture an argument buffer holds says it is.
///
/// A function of its own, like the three beside it, rather than four getters read where the type
/// getter answered: the receiver of a call is typed by what declares it, and a binding an `if let`
/// pattern made carries no declared type, so reading these four there named their container to
/// nobody. The same four calls, with the receiver a parameter, resolve to `MTLTextureReferenceType`.
fn metal_reference(held: &MTLTextureReferenceType) -> Answer<ReferenceFacts> {
    Ok(ReferenceFacts {
        data_type: our_data_type(held.textureDataType().0)?,
        texture_type: our_texture_type(held.textureType().0)?,
        access: our_access(held.access().0)?,
        depth: held.isDepthTexture(),
    })
}

fn metal_array(held: &MTLArrayType, depth: u32) -> Answer<ArrayFacts> {
    Ok(ArrayFacts {
        element_type: our_data_type(held.elementType().0)?,
        array_length: held.arrayLength() as u32,
        stride: held.stride() as u64,
        argument_index_stride: held.argumentIndexStride() as u32,
        element: match depth {
            0 => TypeFacts::Deeper,
            _ => metal_detail(
                held.elementStructType().as_deref(),
                held.elementArrayType().as_deref(),
                held.elementTextureReferenceType().as_deref(),
                held.elementPointerType().as_deref(),
                depth - 1,
            )?,
        },
    })
}

fn metal_pointer(held: &MTLPointerType, depth: u32) -> Answer<PointerFacts> {
    Ok(PointerFacts {
        element_type: our_data_type(held.elementType().0)?,
        access: our_access(held.access().0)?,
        alignment: held.alignment() as u64,
        data_size: held.dataSize() as u64,
        element_is_argument_buffer: held.elementIsArgumentBuffer(),
        element: match depth {
            0 => TypeFacts::Deeper,
            // A pointer declares no texture-reference getter of its own, which is the nil here.
            _ => metal_detail(
                held.elementStructType().as_deref(),
                held.elementArrayType().as_deref(),
                None,
                None,
                depth - 1,
            )?,
        },
    })
}

/// The head of a non-empty array, as the pointer a range setter takes.
fn array_head<T>(elements: &[T]) -> Answer<std::ptr::NonNull<T>> {
    elements
        .first()
        .map(std::ptr::NonNull::from)
        .ok_or_else(|| Failure::named("EmptyRange", "a range setter of no element".to_owned()))
}

impl MetalSide {
    /// The system default device and a queue on it.
    pub fn new() -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| "MTLCreateSystemDefaultDevice returned nil".to_owned())?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| "MTLDevice::newCommandQueue returned nil".to_owned())?;
        Ok(Self {
            device,
            queue,
            libraries: Vec::new(),
            pipelines: Vec::new(),
            buffers: Vec::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
            command_buffers: Vec::new(),
            render_pipelines: Vec::new(),
            command_buffers_indirect: Vec::new(),
            rate_maps: Vec::new(),
            argument_residents: Vec::new(),
            depth_stencil_states: Vec::new(),
            heaps: Vec::new(),
            argument_encoders: Vec::new(),
            surfaces: Vec::new(),
            planes: Vec::new(),
            scratch: Scratch::new()?,
            host_pages: Vec::new(),
        })
    }

    /// The buffer `handle` names.
    fn held(&self, handle: Handle) -> Answer<&ProtocolObject<dyn MTLBuffer>> {
        self.buffers
            .get(handle.0 as usize)
            .map(|buffer| &**buffer)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("buffer {}", handle.0)))
    }

    /// The texture `handle` names.
    fn held_texture(&self, handle: Handle) -> Answer<&ProtocolObject<dyn MTLTexture>> {
        self.textures
            .get(handle.0 as usize)
            .map(|texture| &**texture)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("texture {}", handle.0)))
    }

    /// The sampler state `handle` names.
    fn held_sampler(&self, handle: Handle) -> Answer<&ProtocolObject<dyn MTLSamplerState>> {
        self.samplers
            .get(handle.0 as usize)
            .map(|sampler| &**sampler)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("sampler {}", handle.0)))
    }

    /// The argument encoder `handle` names.
    ///
    /// Retained rather than borrowed, because a case's write reaches the encoder and the pools of
    /// the resources it writes in one call, and the residency it then records needs the side
    /// mutably.
    fn held_encoder(
        &self,
        handle: Handle,
    ) -> Answer<Retained<ProtocolObject<dyn MTLArgumentEncoder>>> {
        self.argument_encoders
            .get(handle.0 as usize)
            .map(|at| at.encoder.clone())
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("argument encoder {}", handle.0)))
    }

    /// The heap `handle` names.
    fn held_heap(&self, handle: Handle) -> Answer<&ProtocolObject<dyn MTLHeap>> {
        self.heaps
            .get(handle.0 as usize)
            .map(|heap| &**heap)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("heap {}", handle.0)))
    }

    /// Encode step `at` of `program`, answering whether it closed its command buffer.
    ///
    /// Each arm is one Metal call, with the program's indices resolved through the handles it
    /// made. The open encoder is passed in rather than held on the side, because a program's
    /// encoders live exactly as long as its own run.
    #[allow(clippy::too_many_arguments)]
    fn sparse_step(
        &mut self,
        program: &SparseMappingProgram,
        at: usize,
        textures: &[Handle],
        buffers: &[Handle],
        fences: &[Retained<ProtocolObject<dyn MTLFence>>],
        scratch: &[Handle],
        scratch_at: &mut usize,
        pending: &mut Vec<(usize, usize)>,
        command_buffer: &mut Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
        open: &mut OpenEncoder,
    ) -> Answer<Commits> {
        let texture_at = |at: usize| -> Answer<Handle> {
            textures
                .get(at)
                .copied()
                .ok_or_else(|| Failure::named("NoSuchHandle", format!("texture {at}")))
        };
        let step = &program.steps[at];
        match step {
            MappingStep::Commit => {
                open.end();
                if command_buffer.is_none() {
                    return Err(Failure::named("NoCommandBuffer", "a commit of nothing"));
                }
                Ok(Commits::Yes)
            }
            MappingStep::NextEncoder => {
                open.end();
                Ok(Commits::No)
            }
            MappingStep::FillRegion {
                texture,
                level,
                slice,
                region,
                ..
            } => {
                let held = Retained::from(self.held_texture(texture_at(*texture)?)?);
                let source = Retained::from(self.held(scratch[*scratch_at])?);
                let (per_row, bytes) = program.tile.bytes_of(region);
                let (origin, extent) = program.tile.texels_of(region);
                let encoder = self.open_blit(command_buffer, open)?;
                // SAFETY: the buffer holds the region's rows at `per_row`, the side just made it
                // that long, and the region lies inside the level's own tiles, which the case's
                // guard holds against the grid this side checked against the device.
                unsafe {
                    encoder.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &source,
                        0,
                        per_row as usize,
                        bytes as usize,
                        long_size(extent),
                        &held,
                        *slice as usize,
                        *level as usize,
                        long_origin(origin),
                    );
                }
                *scratch_at += 1;
                Ok(Commits::No)
            }
            MappingStep::ReadRegion {
                texture,
                level,
                slice,
                region,
            } => {
                let held = Retained::from(self.held_texture(texture_at(*texture)?)?);
                let into = Retained::from(self.held(scratch[*scratch_at])?);
                let (per_row, bytes) = program.tile.bytes_of(region);
                let (origin, extent) = program.tile.texels_of(region);
                let encoder = self.open_blit(command_buffer, open)?;
                // SAFETY: as the fill's, the other way round.
                unsafe {
                    encoder.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                        &held,
                        *slice as usize,
                        *level as usize,
                        long_origin(origin),
                        long_size(extent),
                        &into,
                        0,
                        per_row as usize,
                        bytes as usize,
                    );
                }
                pending.push((*scratch_at, bytes as usize));
                *scratch_at += 1;
                Ok(Commits::No)
            }
            MappingStep::One {
                texture,
                mode,
                region,
                level,
                slice,
            } => {
                let held = Retained::from(self.held_texture(texture_at(*texture)?)?);
                let encoder = self.open_state(command_buffer, open)?;
                // SAFETY: the region lies inside the level's own tiles, which the case's guard
                // holds against the grid this side checked against the device, and the encoder
                // outlives the call.
                unsafe {
                    encoder.updateTextureMapping_mode_region_mipLevel_slice(
                        &held,
                        mapping_mode(*mode),
                        tile_region(region),
                        *level as usize,
                        *slice as usize,
                    );
                }
                Ok(Commits::No)
            }
            MappingStep::Many {
                texture,
                mode,
                regions,
            } => {
                let held = Retained::from(self.held_texture(texture_at(*texture)?)?);
                let rects: Vec<MTLRegion> = regions
                    .iter()
                    .map(|(region, _, _)| tile_region(region))
                    .collect();
                let levels: Vec<usize> = regions
                    .iter()
                    .map(|(_, level, _)| *level as usize)
                    .collect();
                let slices: Vec<usize> = regions
                    .iter()
                    .map(|(_, _, slice)| *slice as usize)
                    .collect();
                let encoder = self.open_state(command_buffer, open)?;
                // SAFETY: the three arrays each hold `numRegions` elements, every region lies
                // inside its level's tiles, and the encoder outlives the call.
                unsafe {
                    encoder.updateTextureMappings_mode_regions_mipLevels_slices_numRegions(
                        &held,
                        mapping_mode(*mode),
                        array_head(&rects)?,
                        array_head(&levels)?,
                        array_head(&slices)?,
                        rects.len(),
                    );
                }
                Ok(Commits::No)
            }
            MappingStep::Indirect {
                texture,
                mode,
                buffer,
                offset,
            } => {
                let held = Retained::from(self.held_texture(texture_at(*texture)?)?);
                let named = buffers
                    .get(*buffer)
                    .copied()
                    .ok_or_else(|| Failure::named("NoSuchHandle", format!("buffer {buffer}")))?;
                let arguments = Retained::from(self.held(named)?);
                let encoder = self.open_state(command_buffer, open)?;
                // SAFETY: the buffer holds a 32-bit count and that many
                // `MTLMapIndirectArguments` at `offset`, which the case states as its own input
                // and its guard checks, and the encoder outlives the call.
                unsafe {
                    encoder.updateTextureMapping_mode_indirectBuffer_indirectBufferOffset(
                        &held,
                        mapping_mode(*mode),
                        &arguments,
                        *offset as usize,
                    );
                }
                Ok(Commits::No)
            }
            MappingStep::Move {
                from,
                from_slice,
                from_level,
                from_origin,
                size,
                to,
                to_slice,
                to_level,
                to_origin,
            } => {
                let source = Retained::from(self.held_texture(texture_at(*from)?)?);
                let destination = Retained::from(self.held_texture(texture_at(*to)?)?);
                let encoder = self.open_state(command_buffer, open)?;
                // SAFETY: both regions lie inside their level's own tiles, which the case's guard
                // holds, and the encoder outlives the call.
                unsafe {
                    encoder.moveTextureMappingsFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &source,
                        *from_slice as usize,
                        *from_level as usize,
                        long_origin(*from_origin),
                        long_size(*size),
                        &destination,
                        *to_slice as usize,
                        *to_level as usize,
                        long_origin(*to_origin),
                    );
                }
                Ok(Commits::No)
            }
            MappingStep::UpdateFence { fence } => {
                let held = fences
                    .get(*fence)
                    .cloned()
                    .ok_or_else(|| Failure::named("NoSuchHandle", format!("fence {fence}")))?;
                let encoder = self.open_state(command_buffer, open)?;
                encoder.updateFence(&held);
                Ok(Commits::No)
            }
            MappingStep::WaitForFence { fence } => {
                let held = fences
                    .get(*fence)
                    .cloned()
                    .ok_or_else(|| Failure::named("NoSuchHandle", format!("fence {fence}")))?;
                let encoder = self.open_state(command_buffer, open)?;
                encoder.waitForFence(&held);
                Ok(Commits::No)
            }
        }
    }

    /// The command buffer a sparse mapping program's next command goes into, opened on first use.
    fn open_command_buffer(
        &self,
        command_buffer: &mut Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    ) -> Answer<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if command_buffer.is_none() {
            *command_buffer = Some(
                self.queue
                    .commandBuffer()
                    .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?,
            );
        }
        Ok(command_buffer
            .clone()
            .expect("the command buffer was just opened"))
    }

    /// The program's blit encoder, ending a resource state encoder first where one is open.
    fn open_blit(
        &self,
        command_buffer: &mut Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
        open: &mut OpenEncoder,
    ) -> Answer<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>> {
        if let OpenEncoder::Blit(encoder) = open {
            return Ok(encoder.clone());
        }
        open.end();
        let buffer = self.open_command_buffer(command_buffer)?;
        let encoder = buffer
            .blitCommandEncoder()
            .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil".to_owned()))?;
        *open = OpenEncoder::Blit(encoder.clone());
        Ok(encoder)
    }

    /// The program's resource state encoder, ending a blit encoder first where one is open.
    fn open_state(
        &self,
        command_buffer: &mut Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
        open: &mut OpenEncoder,
    ) -> Answer<Retained<ProtocolObject<dyn MTLResourceStateCommandEncoder>>> {
        if let OpenEncoder::State(encoder) = open {
            return Ok(encoder.clone());
        }
        open.end();
        let buffer = self.open_command_buffer(command_buffer)?;
        let encoder = buffer.resourceStateCommandEncoder().ok_or_else(|| {
            Failure::named("ResourceStateCommandEncoder", "returned nil".to_owned())
        })?;
        *open = OpenEncoder::State(encoder.clone());
        Ok(encoder)
    }

    /// The rasterization-rate map `handle` names.
    fn held_rate_map(
        &self,
        handle: Handle,
    ) -> Answer<&ProtocolObject<dyn MTLRasterizationRateMap>> {
        self.rate_maps
            .get(handle.0 as usize)
            .map(|map| &**map)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("rate map {}", handle.0)))
    }

    /// Level 0 of `texture` read back through a blit into a shared buffer of its own, at
    /// `row_pitch` bytes a row: what a private texture's bytes are, since the host reaches none of
    /// them directly.
    fn blit_out(
        &mut self,
        texture: &ProtocolObject<dyn MTLTexture>,
        size: [u32; 2],
        row_pitch: u32,
    ) -> Answer<Vec<u8>> {
        let length = row_pitch as usize * size[1] as usize;
        let back = self
            .device
            .newBufferWithLength_options(length.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                Failure::named(
                    "NewBufferWithLength",
                    format!("{length} bytes returned nil"),
                )
            })?;
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        let blit = command_buffer
            .blitCommandEncoder()
            .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil".to_owned()))?;
        // SAFETY: the buffer holds `size[1]` rows of `row_pitch` bytes, which is what the copy
        // writes, and the texture holds level 0 of that extent.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                texture,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize {
                    width: size[0] as usize,
                    height: size[1] as usize,
                    depth: 1,
                },
                &back,
                0,
                row_pitch as usize,
                length,
            );
        }
        blit.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let mut bytes = vec![0u8; length];
        // SAFETY: the buffer is shared storage of at least `length` bytes and the copy into it has
        // completed.
        unsafe {
            std::ptr::copy_nonoverlapping(
                back.contents().as_ptr().cast::<u8>(),
                bytes.as_mut_ptr(),
                length,
            );
        }
        keep(&mut self.buffers, back);
        keep(&mut self.command_buffers, command_buffer);
        Ok(bytes)
    }

    /// One command buffer with one compute encoder: the pipeline set, `bindings` bound at 0 up,
    /// each of `stage_in` set in order, `dispatch` encoded, and `commit`. The two dispatch calls
    /// share this body, so neither side can pick up a different binding order with one of them.
    fn encode(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        stage_in: &[StageInRegion],
        dispatch: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) -> Answer<Handle> {
        let bound: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = bindings
            .buffers
            .iter()
            .map(|handle| self.held(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        let sampled: Vec<Retained<ProtocolObject<dyn MTLTexture>>> = bindings
            .textures
            .iter()
            .map(|handle| self.held_texture(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        let states: Vec<Retained<ProtocolObject<dyn MTLSamplerState>>> = bindings
            .samplers
            .iter()
            .map(|handle| self.held_sampler(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        // The textures the bound argument buffers name: Metal keeps a resource reached through an
        // argument buffer resident only where the encoder was told so.
        let resident: Vec<Retained<ProtocolObject<dyn MTLResource>>> = self
            .argument_residents
            .iter()
            .filter(|(buffer, _)| bindings.buffers.contains(buffer))
            .flat_map(|(_, textures)| textures.clone())
            .map(|handle| {
                self.held_texture(handle).map(|texture| {
                    let resource: &ProtocolObject<dyn MTLResource> =
                        ProtocolObject::from_ref(texture);
                    Retained::from(resource)
                })
            })
            .collect::<Answer<Vec<_>>>()?;
        let state = self
            .pipelines
            .get(pipeline.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", pipeline.0)))?
            .clone();
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| Failure::named("ComputeCommandEncoder", "returned nil".to_owned()))?;
        encoder.setComputePipelineState(&state);
        for (at, buffer) in bound.iter().enumerate() {
            // SAFETY: the buffer is at least as long as the kernel's binding, and the encoder
            // outlives the call.
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, at) };
        }
        for (at, texture) in sampled.iter().enumerate() {
            // SAFETY: the texture is of the kind and access the kernel declares at that slot, and
            // the encoder outlives the call.
            unsafe { encoder.setTexture_atIndex(Some(texture), at) };
        }
        for (at, sampler) in states.iter().enumerate() {
            // SAFETY: the kernel declares a sampler at that slot, and the encoder outlives the
            // call.
            unsafe { encoder.setSamplerState_atIndex(Some(sampler), at) };
        }
        for resource in &resident {
            encoder.useResource_usage(resource, MTLResourceUsage::Read);
        }
        for region in stage_in {
            match region {
                StageInRegion::Direct { origin, size } => {
                    encoder.setStageInRegion(MTLRegion {
                        origin: MTLOrigin {
                            x: origin[0] as usize,
                            y: origin[1] as usize,
                            z: origin[2] as usize,
                        },
                        size: MTLSize {
                            width: size[0] as usize,
                            height: size[1] as usize,
                            depth: size[2] as usize,
                        },
                    });
                }
                StageInRegion::Indirect { buffer, offset } => {
                    let held = self.held(*buffer)?;
                    // SAFETY: the buffer holds the region's six words from `offset`, which the
                    // case states as one of its inputs, and the encoder outlives the call.
                    unsafe {
                        encoder.setStageInRegionWithIndirectBuffer_indirectBufferOffset(
                            held,
                            *offset as usize,
                        );
                    }
                }
            }
        }
        dispatch(&encoder);
        encoder.endEncoding();
        command_buffer.commit();
        Ok(keep(&mut self.command_buffers, command_buffer))
    }
}

/// Bytes the command buffer whose timings [`Side::command_buffer_facts`] reads fills: a few, so
/// the GPU ran and nothing else did.
const TIMED_BYTES: usize = 256;

/// The labels the two queues of [`Side::command_buffer_facts`] take: distinct, so a command
/// buffer that answered the other queue reads as the other label.
const QUEUE_LABELS: [&str; 2] = ["conformance-queue-a", "conformance-queue-b"];

/// The byte a read-back buffer of a sparse mapping program is made holding, so a byte no copy
/// wrote is a value the side put there rather than whatever the allocator left.
const READ_GUARD: u8 = 0x7e;

/// Which encoder of a sparse mapping program's command buffer is open.
///
/// Metal takes one encoder at a time of a command buffer, so a program whose next command needs
/// the other kind ends this one first. The variants hold the encoder itself rather than a flag,
/// because ending it is a call on the object.
enum OpenEncoder {
    None,
    Blit(Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>),
    State(Retained<ProtocolObject<dyn MTLResourceStateCommandEncoder>>),
}

impl OpenEncoder {
    /// End whichever encoder is open, leaving none.
    fn end(&mut self) {
        match self {
            OpenEncoder::None => {}
            OpenEncoder::Blit(encoder) => encoder.endEncoding(),
            OpenEncoder::State(encoder) => encoder.endEncoding(),
        }
        *self = OpenEncoder::None;
    }
}

/// Whether a step of a sparse mapping program closed its command buffer.
enum Commits {
    No,
    Yes,
}

/// Metal's `MTLSparseTextureMappingMode` for a case's mode, by its raw value, so the two spellings
/// cannot drift apart.
fn mapping_mode(mode: MappingMode) -> MTLSparseTextureMappingMode {
    MTLSparseTextureMappingMode(mode.as_raw() as usize)
}

/// Three axes as Metal's size.
fn long_size(axes: [u64; 3]) -> MTLSize {
    MTLSize {
        width: axes[0] as usize,
        height: axes[1] as usize,
        depth: axes[2] as usize,
    }
}

/// Three axes as Metal's origin.
fn long_origin(axes: [u64; 3]) -> MTLOrigin {
    MTLOrigin {
        x: axes[0] as usize,
        y: axes[1] as usize,
        z: axes[2] as usize,
    }
}

/// A region of tiles as the `MTLRegion` the mapping calls take: Metal states a mapping region in
/// **tiles**, not texels, so the numbers pass through unchanged.
fn tile_region(region: &TileRect) -> MTLRegion {
    MTLRegion {
        origin: long_origin(region.origin),
        size: long_size(region.size),
    }
}

/// Three axes as Metal's size.
fn size(axes: [u32; 3]) -> MTLSize {
    MTLSize {
        width: axes[0] as usize,
        height: axes[1] as usize,
        depth: axes[2] as usize,
    }
}

/// Metal's region for a case's six numbers.
fn metal_region(held: &Box6) -> MTLRegion {
    MTLRegion {
        origin: MTLOrigin {
            x: held[0] as usize,
            y: held[1] as usize,
            z: held[2] as usize,
        },
        size: MTLSize {
            width: held[3] as usize,
            height: held[4] as usize,
            depth: held[5] as usize,
        },
    }
}

/// A case's six numbers for Metal's region.
fn our_region(held: &MTLRegion) -> Box6 {
    [
        held.origin.x as u64,
        held.origin.y as u64,
        held.origin.z as u64,
        held.size.width as u64,
        held.size.height as u64,
        held.size.depth as u64,
    ]
}

/// `MTLBufferLayoutStrideDynamic`, the stride a layout takes from the encoder's binding.
const STRIDE_DYNAMIC: usize = usize::MAX;

/// Set every field [`AWAY`] states on `descriptor`, each away from Metal's own default.
///
/// The label is set only because the table states one: `setLabel:nil` ends Metal's process, and
/// nothing here ever passes nil.
fn state_descriptor(descriptor: &Retained<MTLRenderPipelineDescriptor>) {
    descriptor.setLabel(Some(&NSString::from_str(AWAY.label)));
    // `sampleCount` is the **deprecated** spelling of the raster sample count, and this is the
    // only one this side sends: the case reads both getters back, which is what says the two are
    // one field of the runtime's. The deprecation is Apple's and the coverage map still lists the
    // property as implemented, so there is no non-deprecated way to send the selector a case must
    // send; the attribute is on the two call sites alone, with no `deprecated` anywhere else in
    // this file inside it.
    #[allow(deprecated)]
    descriptor.setSampleCount(AWAY.sample_count as usize);
    descriptor.setAlphaToCoverageEnabled(AWAY.alpha_to_coverage);
    descriptor.setAlphaToOneEnabled(AWAY.alpha_to_one);
    descriptor.setRasterizationEnabled(AWAY.rasterization_enabled);
    descriptor.setDepthAttachmentPixelFormat(MTLPixelFormat(AWAY.depth_format.as_raw() as usize));
    descriptor
        .setStencilAttachmentPixelFormat(MTLPixelFormat(AWAY.stencil_format.as_raw() as usize));
    descriptor.setTessellationFactorScaleEnabled(AWAY.tessellation_factor_scale_enabled);
    descriptor.setTessellationFactorFormat(MTLTessellationFactorFormat(
        AWAY.tessellation_factor_format.as_raw() as usize,
    ));
    descriptor.setTessellationFactorStepFunction(MTLTessellationFactorStepFunction(
        AWAY.tessellation_factor_step_function.as_raw() as usize,
    ));
    descriptor.setTessellationOutputWindingOrder(MTLWinding(
        AWAY.tessellation_output_winding_order.as_raw() as usize,
    ));
    // SAFETY: the amplification count, the topology class, the partition mode, the tessellation
    // factor and the control-point index type are the stated table's own values, each a named
    // value of Metal's own enumeration or a small positive number inside Metal's documented range
    // (an amplification count of 2 of the device's 8, a factor of 12 of Metal's 64). Nothing is
    // compiled from the descriptor, so no value here reaches a pipeline.
    unsafe {
        descriptor.setMaxVertexAmplificationCount(AWAY.max_vertex_amplification_count as usize);
        descriptor.setInputPrimitiveTopology(MTLPrimitiveTopologyClass(
            AWAY.input_primitive_topology.as_raw() as usize,
        ));
        descriptor.setTessellationPartitionMode(MTLTessellationPartitionMode(
            AWAY.tessellation_partition_mode.as_raw() as usize,
        ));
        descriptor.setMaxTessellationFactor(AWAY.max_tessellation_factor as usize);
        descriptor.setTessellationControlPointIndexType(MTLTessellationControlPointIndexType(
            AWAY.tessellation_control_point_index_type.as_raw() as usize,
        ));
    }
    descriptor.setSupportIndirectCommandBuffers(AWAY.support_indirect_command_buffers);
    descriptor.setSupportAddingVertexBinaryFunctions(AWAY.support_adding_vertex_binary_functions);
    descriptor
        .setSupportAddingFragmentBinaryFunctions(AWAY.support_adding_fragment_binary_functions);
    descriptor.setMaxVertexCallStackDepth(AWAY.max_vertex_call_stack_depth as usize);
    descriptor.setMaxFragmentCallStackDepth(AWAY.max_fragment_call_stack_depth as usize);
    let colors: Retained<MTLRenderPipelineColorAttachmentDescriptorArray> =
        descriptor.colorAttachments();
    for color in AWAY.colors {
        // SAFETY: the index is below the eight colour attachments every Metal pipeline descriptor
        // holds, which `every_stated_slot_is_read_back` keeps it to.
        let attachment: Retained<MTLRenderPipelineColorAttachmentDescriptor> =
            unsafe { colors.objectAtIndexedSubscript(color.index as usize) };
        state_color_attachment(&attachment, color);
    }
    let vertex_buffers: Retained<MTLPipelineBufferDescriptorArray> = descriptor.vertexBuffers();
    let fragment_buffers: Retained<MTLPipelineBufferDescriptorArray> = descriptor.fragmentBuffers();
    state_mutability(&vertex_buffers, AWAY.vertex_buffers);
    state_mutability(&fragment_buffers, AWAY.fragment_buffers);
    let vertex_layout = MTLVertexDescriptor::vertexDescriptor();
    let attributes: Retained<MTLVertexAttributeDescriptorArray> = vertex_layout.attributes();
    for attribute in AWAY.attributes {
        // SAFETY: as the attachments', for the attribute array.
        let slot: Retained<MTLVertexAttributeDescriptor> =
            unsafe { attributes.objectAtIndexedSubscript(attribute.index as usize) };
        slot.setFormat(MTLVertexFormat(attribute.format.as_raw() as usize));
        // SAFETY: the offset and the buffer index are the table's own numbers, inside the record
        // and the buffer array the descriptor holds.
        unsafe {
            slot.setOffset(attribute.offset as usize);
            slot.setBufferIndex(attribute.buffer_index as usize);
        }
    }
    let layouts: Retained<MTLVertexBufferLayoutDescriptorArray> = vertex_layout.layouts();
    for layout in AWAY.layouts {
        // SAFETY: as the attributes', for the layout array.
        let slot: Retained<MTLVertexBufferLayoutDescriptor> =
            unsafe { layouts.objectAtIndexedSubscript(layout.index as usize) };
        slot.setStepFunction(MTLVertexStepFunction(layout.step_function.as_raw() as usize));
        // SAFETY: the stride and the rate are the table's own numbers.
        unsafe {
            slot.setStride(layout.stride as usize);
            slot.setStepRate(layout.step_rate as usize);
        }
    }
    descriptor.setVertexDescriptor(Some(&vertex_layout));
}

/// What `descriptor` answers of itself, every getter of its own surface.
///
/// Each list bound to a local of its declared type, as the bindings walk does: the coverage
/// derivation types a receiver from what bound it, so a call made on a chained getter's answer
/// names no type at all.
fn read_descriptor(
    descriptor: &Retained<MTLRenderPipelineDescriptor>,
    supplied: bool,
) -> RenderDescriptorFacts {
    let colors: Retained<MTLRenderPipelineColorAttachmentDescriptorArray> =
        descriptor.colorAttachments();
    let mut attachments = Vec::new();
    for index in 0..READ_COLORS {
        // SAFETY: `READ_COLORS` is below the eight attachments every descriptor holds.
        let attachment: Retained<MTLRenderPipelineColorAttachmentDescriptor> =
            unsafe { colors.objectAtIndexedSubscript(index as usize) };
        attachments.push(read_color_attachment(&attachment));
    }
    let vertex_buffers: Retained<MTLPipelineBufferDescriptorArray> = descriptor.vertexBuffers();
    let fragment_buffers: Retained<MTLPipelineBufferDescriptorArray> = descriptor.fragmentBuffers();
    let archives = descriptor.binaryArchives();
    let vertex: Option<Retained<ProtocolObject<dyn MTLFunction>>> = descriptor.vertexFunction();
    let fragment: Option<Retained<ProtocolObject<dyn MTLFunction>>> = descriptor.fragmentFunction();
    let layout: Option<Retained<MTLVertexDescriptor>> = descriptor.vertexDescriptor();
    RenderDescriptorFacts {
        label: descriptor.label().map(|label| label.to_string()),
        vertex_function: vertex.map(|held| held.name().to_string()),
        fragment_function: fragment.map(|held| held.name().to_string()),
        vertex_descriptor: layout.as_ref().map_or_else(
            || VertexDescriptorFacts::empty(READ_VERTEX_SLOTS),
            read_vertex_descriptor,
        ),
        // `sampleCount` is Apple's deprecated spelling; see `state_descriptor`.
        #[allow(deprecated)]
        sample_count: descriptor.sampleCount() as u64,
        raster_sample_count: descriptor.rasterSampleCount() as u64,
        alpha_to_coverage_enabled: descriptor.isAlphaToCoverageEnabled(),
        alpha_to_one_enabled: descriptor.isAlphaToOneEnabled(),
        rasterization_enabled: descriptor.isRasterizationEnabled(),
        max_vertex_amplification_count: descriptor.maxVertexAmplificationCount() as u64,
        colors: attachments,
        depth_attachment_pixel_format: descriptor.depthAttachmentPixelFormat().0 as u32,
        stencil_attachment_pixel_format: descriptor.stencilAttachmentPixelFormat().0 as u32,
        input_primitive_topology: descriptor.inputPrimitiveTopology().0 as u32,
        tessellation_partition_mode: descriptor.tessellationPartitionMode().0 as u32,
        max_tessellation_factor: descriptor.maxTessellationFactor() as u64,
        tessellation_factor_scale_enabled: descriptor.isTessellationFactorScaleEnabled(),
        tessellation_factor_format: descriptor.tessellationFactorFormat().0 as u32,
        tessellation_control_point_index_type: descriptor.tessellationControlPointIndexType().0
            as u32,
        tessellation_factor_step_function: descriptor.tessellationFactorStepFunction().0 as u32,
        tessellation_output_winding_order: descriptor.tessellationOutputWindingOrder().0 as u32,
        vertex_buffers: read_mutability(&vertex_buffers),
        fragment_buffers: read_mutability(&fragment_buffers),
        support_indirect_command_buffers: descriptor.supportIndirectCommandBuffers(),
        // `len` and not `count`: the two are one Objective-C selector, and `count` is a selector
        // `objc2-metal` binds on other types, so the coverage derivation would read it as a Metal
        // call on the element type a preloaded-library getter answers.
        binary_archives: archives.map_or(0, |held| held.len() as u64),
        supplied: supplied.then(|| read_supplied(descriptor)),
        support_adding_vertex_binary_functions: descriptor.supportAddingVertexBinaryFunctions(),
        support_adding_fragment_binary_functions: descriptor.supportAddingFragmentBinaryFunctions(),
        max_vertex_call_stack_depth: descriptor.maxVertexCallStackDepth() as u64,
        max_fragment_call_stack_depth: descriptor.maxFragmentCallStackDepth() as u64,
    }
}

/// The four per-stage lists of supplied objects, by how many each holds.
///
/// Read only where the shape asks for it: `reset` leaves `vertexPreloadedLibraries` nil and
/// `objc2-metal`'s accessor for it panics on a nil ([`SuppliedLists`]).
fn read_supplied(descriptor: &Retained<MTLRenderPipelineDescriptor>) -> SuppliedLists {
    let vertex_libraries = descriptor.vertexPreloadedLibraries();
    let fragment_libraries = descriptor.fragmentPreloadedLibraries();
    SuppliedLists {
        vertex_preloaded_libraries: vertex_libraries.len() as u64,
        fragment_preloaded_libraries: fragment_libraries.len() as u64,
        vertex_linked_functions: linked_counts(&descriptor.vertexLinkedFunctions()),
        fragment_linked_functions: linked_counts(&descriptor.fragmentLinkedFunctions()),
    }
}

/// What a vertex descriptor's first [`READ_VERTEX_SLOTS`] attribute and layout slots answer.
fn read_vertex_descriptor(layout: &Retained<MTLVertexDescriptor>) -> VertexDescriptorFacts {
    let attributes: Retained<MTLVertexAttributeDescriptorArray> = layout.attributes();
    let layouts: Retained<MTLVertexBufferLayoutDescriptorArray> = layout.layouts();
    VertexDescriptorFacts {
        attributes: (0..READ_VERTEX_SLOTS)
            .map(|index| {
                // SAFETY: `READ_VERTEX_SLOTS` is below the attribute slots a vertex descriptor
                // holds.
                let slot: Retained<MTLVertexAttributeDescriptor> =
                    unsafe { attributes.objectAtIndexedSubscript(index as usize) };
                DescriptorAttributeFacts {
                    format: slot.format().0 as u32,
                    offset: slot.offset() as u64,
                    buffer_index: slot.bufferIndex() as u64,
                }
            })
            .collect(),
        layouts: (0..READ_VERTEX_SLOTS)
            .map(|index| {
                // SAFETY: as the attributes', for the layout array.
                let slot: Retained<MTLVertexBufferLayoutDescriptor> =
                    unsafe { layouts.objectAtIndexedSubscript(index as usize) };
                LayoutFacts {
                    stride: slot.stride() as u64,
                    step_function: slot.stepFunction().0 as u32,
                    step_rate: slot.stepRate() as u64,
                }
            })
            .collect(),
    }
}

/// The mutability of a pipeline buffer array's first [`READ_BUFFERS`] slots.
///
/// One function for the four arrays that answer it — a render descriptor's two, a compute
/// descriptor's `buffers` and a tile descriptor's `tileBuffers` — because the array type is one
/// Metal class and the question is one question.
fn read_mutability(array: &Retained<MTLPipelineBufferDescriptorArray>) -> Vec<u32> {
    (0..READ_BUFFERS)
        .map(|index| {
            // SAFETY: `READ_BUFFERS` is below the buffer slots a pipeline descriptor holds.
            let slot: Retained<MTLPipelineBufferDescriptor> =
                unsafe { array.objectAtIndexedSubscript(index as usize) };
            slot.mutability().0 as u32
        })
        .collect()
}

/// Set the stated mutability of each slot `stated` names on a pipeline buffer array.
fn state_mutability(
    array: &Retained<MTLPipelineBufferDescriptorArray>,
    stated: &[(u32, metal2vulkan::metal::Mutability)],
) {
    for (index, mutability) in stated {
        // SAFETY: the index is below the buffer slots a pipeline descriptor holds, which each
        // family's own `every_stated_slot_is_read_back` keeps it to.
        let slot: Retained<MTLPipelineBufferDescriptor> =
            unsafe { array.objectAtIndexedSubscript(*index as usize) };
        slot.setMutability(MTLMutability(mutability.as_raw() as usize));
    }
}

/// Set every field [`ComputeAway`] states on a compute pipeline descriptor, each away from Metal's
/// own default.
///
/// The label is set only because the table states one: `setLabel:nil` ends Metal's process.
fn state_compute_descriptor(descriptor: &Retained<MTLComputePipelineDescriptor>) {
    descriptor.setLabel(Some(&NSString::from_str(COMPUTE_AWAY.label)));
    // SAFETY: the flag is a promise to Metal's optimizer about the threadgroup sizes a dispatch
    // will use, and nothing of this family dispatches or compiles the descriptor.
    unsafe {
        descriptor.setThreadGroupSizeIsMultipleOfThreadExecutionWidth(
            COMPUTE_AWAY.threadgroup_size_is_multiple_of_execution_width,
        );
    }
    descriptor
        .setMaxTotalThreadsPerThreadgroup(COMPUTE_AWAY.max_total_threads_per_threadgroup as usize);
    descriptor.setSupportIndirectCommandBuffers(COMPUTE_AWAY.support_indirect_command_buffers);
    descriptor.setSupportAddingBinaryFunctions(COMPUTE_AWAY.support_adding_binary_functions);
    descriptor.setMaxCallStackDepth(COMPUTE_AWAY.max_call_stack_depth as usize);
    descriptor
        .setRequiredThreadsPerThreadgroup(size(COMPUTE_AWAY.required_threads_per_threadgroup));
    let buffers: Retained<MTLPipelineBufferDescriptorArray> = descriptor.buffers();
    state_mutability(&buffers, COMPUTE_AWAY.buffers);
}

/// What a compute pipeline descriptor answers of itself, every getter of its own compared surface.
///
/// Each array bound to a local of its declared type, as the bindings walk does: the coverage
/// derivation types a receiver from what bound it, so a call made on a chained getter's answer
/// names no type at all.
fn read_compute_descriptor(
    descriptor: &Retained<MTLComputePipelineDescriptor>,
    supplied: bool,
) -> ComputeDescriptorFacts {
    let buffers: Retained<MTLPipelineBufferDescriptorArray> = descriptor.buffers();
    let archives = descriptor.binaryArchives();
    let kernel: Option<Retained<ProtocolObject<dyn MTLFunction>>> = descriptor.computeFunction();
    let required = descriptor.requiredThreadsPerThreadgroup();
    ComputeDescriptorFacts {
        label: descriptor.label().map(|label| label.to_string()),
        compute_function: kernel.map(|held| held.name().to_string()),
        threadgroup_size_is_multiple_of_execution_width: descriptor
            .threadGroupSizeIsMultipleOfThreadExecutionWidth(),
        max_total_threads_per_threadgroup: descriptor.maxTotalThreadsPerThreadgroup() as u64,
        buffers: read_mutability(&buffers),
        support_indirect_command_buffers: descriptor.supportIndirectCommandBuffers(),
        // `len` and not `count`: the two are one Objective-C selector, and `count` is a selector
        // `objc2-metal` binds on other types, so the coverage derivation would read it as a Metal
        // call on the element type the getter answers.
        binary_archives: archives.map_or(0, |held| held.len() as u64),
        supplied: supplied.then(|| read_compute_supplied(descriptor)),
        support_adding_binary_functions: descriptor.supportAddingBinaryFunctions(),
        max_call_stack_depth: descriptor.maxCallStackDepth() as u64,
        required_threads_per_threadgroup: [
            required.width as u64,
            required.height as u64,
            required.depth as u64,
        ],
    }
}

/// A compute descriptor's three lists of supplied objects, by how many each holds.
///
/// Read only where the shape asks for it: `reset` leaves `preloadedLibraries` nil and
/// `objc2-metal`'s accessor for it panics on a nil
/// ([`ComputeSuppliedLists`](crate::compute_descriptors::SuppliedLists)).
fn read_compute_supplied(
    descriptor: &Retained<MTLComputePipelineDescriptor>,
) -> ComputeSuppliedLists {
    let preloaded: Retained<NSArray<ProtocolObject<dyn MTLDynamicLibrary>>> =
        descriptor.preloadedLibraries();
    // `insertLibraries` is Apple's **deprecated** spelling of the same property, and this reads it
    // beside `preloadedLibraries` so that the two answering alike is a recorded fact rather than an
    // assumption. The deprecation is Apple's and the coverage map still lists the property as
    // implemented, so there is no undeprecated way to send the selector a case must send.
    #[allow(deprecated)]
    let inserted: Option<Retained<NSArray<ProtocolObject<dyn MTLDynamicLibrary>>>> =
        descriptor.insertLibraries();
    let linked: Option<Retained<MTLLinkedFunctions>> = descriptor.linkedFunctions();
    ComputeSuppliedLists {
        preloaded_libraries: preloaded.len() as u64,
        insert_libraries: inserted.map_or(0, |held| held.len() as u64),
        linked_functions: linked
            .as_ref()
            .map_or_else(LinkedCounts::default, linked_counts),
    }
}

/// Set every field [`TileAway`] states on a tile render pipeline descriptor.
///
/// `resets` says whether the caller will send `reset` afterwards, because the buffer mutability is
/// stated only where it will not: subscripting `tileBuffers` and then resetting the descriptor
/// leaves a freed array behind that the descriptor's own `dealloc` releases again
/// ([`TileDescriptorFacts::buffers`](crate::tile_descriptors::TileDescriptorFacts::buffers)).
fn state_tile_descriptor(descriptor: &Retained<MTLTileRenderPipelineDescriptor>, resets: bool) {
    descriptor.setLabel(Some(&NSString::from_str(TILE_AWAY.label)));
    // SAFETY: the sample count is the stated table's own 4, a count every Metal device supports,
    // and nothing of this family compiles the descriptor.
    unsafe { descriptor.setRasterSampleCount(TILE_AWAY.raster_sample_count as usize) };
    descriptor.setThreadgroupSizeMatchesTileSize(TILE_AWAY.threadgroup_size_matches_tile_size);
    descriptor
        .setMaxTotalThreadsPerThreadgroup(TILE_AWAY.max_total_threads_per_threadgroup as usize);
    descriptor.setSupportAddingBinaryFunctions(TILE_AWAY.support_adding_binary_functions);
    descriptor.setMaxCallStackDepth(TILE_AWAY.max_call_stack_depth as usize);
    descriptor.setRequiredThreadsPerThreadgroup(size(TILE_AWAY.required_threads_per_threadgroup));
    let colors: Retained<MTLTileRenderPipelineColorAttachmentDescriptorArray> =
        descriptor.colorAttachments();
    for (index, format) in TILE_AWAY.colors {
        // SAFETY: the index is below the eight colour attachments every tile descriptor holds,
        // which the family's `every_stated_slot_is_read_back` keeps it to.
        let attachment: Retained<MTLTileRenderPipelineColorAttachmentDescriptor> =
            unsafe { colors.objectAtIndexedSubscript(*index as usize) };
        attachment.setPixelFormat(MTLPixelFormat(format.as_raw() as usize));
    }
    if !resets {
        let buffers: Retained<MTLPipelineBufferDescriptorArray> = descriptor.tileBuffers();
        state_mutability(&buffers, TILE_AWAY.buffers);
    }
}

/// What a tile render pipeline descriptor answers of itself.
fn read_tile_descriptor(
    descriptor: &Retained<MTLTileRenderPipelineDescriptor>,
    supplied: bool,
) -> TileDescriptorFacts {
    let colors: Retained<MTLTileRenderPipelineColorAttachmentDescriptorArray> =
        descriptor.colorAttachments();
    let mut formats = Vec::new();
    for index in 0..READ_COLORS {
        // SAFETY: `READ_COLORS` is below the eight attachments every tile descriptor holds.
        let attachment: Retained<MTLTileRenderPipelineColorAttachmentDescriptor> =
            unsafe { colors.objectAtIndexedSubscript(index as usize) };
        formats.push(attachment.pixelFormat().0 as u32);
    }
    let archives = descriptor.binaryArchives();
    // Non-null in `objc2-metal` although a fresh descriptor's is nil, so every shape of the family
    // sets it before this runs ([`crate::tile_descriptors`]).
    let tile: Retained<ProtocolObject<dyn MTLFunction>> = descriptor.tileFunction();
    let required = descriptor.requiredThreadsPerThreadgroup();
    TileDescriptorFacts {
        label: descriptor.label().map(|label| label.to_string()),
        tile_function: tile.name().to_string(),
        raster_sample_count: descriptor.rasterSampleCount() as u64,
        colors: formats,
        threadgroup_size_matches_tile_size: descriptor.threadgroupSizeMatchesTileSize(),
        // Read only where the shape asks for it, which is every shape that does not reset: a reset
        // descriptor's `tileBuffers` is memory Metal has freed
        // ([`TileDescriptorFacts::buffers`](crate::tile_descriptors::TileDescriptorFacts::buffers)).
        buffers: supplied.then(|| {
            let buffers: Retained<MTLPipelineBufferDescriptorArray> = descriptor.tileBuffers();
            read_mutability(&buffers)
        }),
        max_total_threads_per_threadgroup: descriptor.maxTotalThreadsPerThreadgroup() as u64,
        binary_archives: archives.map_or(0, |held| held.len() as u64),
        supplied: supplied.then(|| read_tile_supplied(descriptor)),
        support_adding_binary_functions: descriptor.supportAddingBinaryFunctions(),
        max_call_stack_depth: descriptor.maxCallStackDepth() as u64,
        required_threads_per_threadgroup: [
            required.width as u64,
            required.height as u64,
            required.depth as u64,
        ],
    }
}

/// A tile descriptor's two lists of supplied objects, by how many each holds.
///
/// Read only where the shape asks for it: both accessors are non-null and `reset` leaves each nil
/// ([`TileSuppliedLists`](crate::tile_descriptors::SuppliedLists)).
fn read_tile_supplied(descriptor: &Retained<MTLTileRenderPipelineDescriptor>) -> TileSuppliedLists {
    let preloaded: Retained<NSArray<ProtocolObject<dyn MTLDynamicLibrary>>> =
        descriptor.preloadedLibraries();
    let linked: Retained<MTLLinkedFunctions> = descriptor.linkedFunctions();
    TileSuppliedLists {
        preloaded_libraries: preloaded.len() as u64,
        linked_functions: linked_counts(&linked),
    }
}

/// Set every field [`MeshAway`] states on a mesh render pipeline descriptor.
fn state_mesh_descriptor(descriptor: &Retained<MTLMeshRenderPipelineDescriptor>) {
    descriptor.setLabel(Some(&NSString::from_str(MESH_AWAY.label)));
    // SAFETY: the sample count is the stated table's own 4, a count every Metal device supports,
    // and nothing of this family compiles the descriptor.
    unsafe { descriptor.setRasterSampleCount(MESH_AWAY.raster_sample_count as usize) };
    descriptor.setAlphaToCoverageEnabled(MESH_AWAY.alpha_to_coverage);
    descriptor.setAlphaToOneEnabled(MESH_AWAY.alpha_to_one);
    descriptor.setRasterizationEnabled(MESH_AWAY.rasterization_enabled);
    descriptor
        .setDepthAttachmentPixelFormat(MTLPixelFormat(MESH_AWAY.depth_format.as_raw() as usize));
    descriptor.setStencilAttachmentPixelFormat(MTLPixelFormat(
        MESH_AWAY.stencil_format.as_raw() as usize
    ));
    descriptor.setSupportIndirectCommandBuffers(MESH_AWAY.support_indirect_command_buffers);
    let colors: Retained<MTLRenderPipelineColorAttachmentDescriptorArray> =
        descriptor.colorAttachments();
    for color in MESH_AWAY.colors {
        // SAFETY: the index is below the eight colour attachments every mesh descriptor holds,
        // which the family's `every_stated_slot_is_read_back` keeps it to.
        let attachment: Retained<MTLRenderPipelineColorAttachmentDescriptor> =
            unsafe { colors.objectAtIndexedSubscript(color.index as usize) };
        state_color_attachment(&attachment, color);
    }
}

/// What a mesh render pipeline descriptor answers of its implemented surface.
fn read_mesh_descriptor(
    descriptor: &Retained<MTLMeshRenderPipelineDescriptor>,
    supplied: bool,
) -> MeshDescriptorFacts {
    let colors: Retained<MTLRenderPipelineColorAttachmentDescriptorArray> =
        descriptor.colorAttachments();
    let mut attachments = Vec::new();
    for index in 0..READ_COLORS {
        // SAFETY: `READ_COLORS` is below the eight attachments every mesh descriptor holds.
        let attachment: Retained<MTLRenderPipelineColorAttachmentDescriptor> =
            unsafe { colors.objectAtIndexedSubscript(index as usize) };
        attachments.push(read_color_attachment(&attachment));
    }
    let archives = descriptor.binaryArchives();
    let object: Option<Retained<ProtocolObject<dyn MTLFunction>>> = descriptor.objectFunction();
    let mesh: Option<Retained<ProtocolObject<dyn MTLFunction>>> = descriptor.meshFunction();
    let fragment: Option<Retained<ProtocolObject<dyn MTLFunction>>> = descriptor.fragmentFunction();
    MeshDescriptorFacts {
        label: descriptor.label().map(|label| label.to_string()),
        object_function: object.map(|held| held.name().to_string()),
        mesh_function: mesh.map(|held| held.name().to_string()),
        fragment_function: fragment.map(|held| held.name().to_string()),
        raster_sample_count: descriptor.rasterSampleCount() as u64,
        alpha_to_coverage_enabled: descriptor.isAlphaToCoverageEnabled(),
        alpha_to_one_enabled: descriptor.isAlphaToOneEnabled(),
        rasterization_enabled: descriptor.isRasterizationEnabled(),
        colors: attachments,
        depth_attachment_pixel_format: descriptor.depthAttachmentPixelFormat().0 as u32,
        stencil_attachment_pixel_format: descriptor.stencilAttachmentPixelFormat().0 as u32,
        support_indirect_command_buffers: descriptor.supportIndirectCommandBuffers(),
        binary_archives: archives.map_or(0, |held| held.len() as u64),
        supplied: supplied.then(|| read_mesh_supplied(descriptor)),
    }
}

/// A mesh descriptor's three per-stage linked-function lists, by how many objects each holds.
///
/// Read only where the shape asks for it: all three accessors are non-null and `reset` leaves each
/// nil ([`MeshSuppliedLists`](crate::mesh_descriptors::SuppliedLists)).
fn read_mesh_supplied(descriptor: &Retained<MTLMeshRenderPipelineDescriptor>) -> MeshSuppliedLists {
    let object: Retained<MTLLinkedFunctions> = descriptor.objectLinkedFunctions();
    let mesh: Retained<MTLLinkedFunctions> = descriptor.meshLinkedFunctions();
    let fragment: Retained<MTLLinkedFunctions> = descriptor.fragmentLinkedFunctions();
    MeshSuppliedLists {
        object_linked_functions: linked_counts(&object),
        mesh_linked_functions: linked_counts(&mesh),
        fragment_linked_functions: linked_counts(&fragment),
    }
}

/// Set one colour attachment's nine fields from a stated row.
///
/// One function for the two descriptors whose attachments are
/// `MTLRenderPipelineColorAttachmentDescriptor` — the render descriptor's and the mesh
/// descriptor's — so the two families cannot disagree about which setter a stated field reaches.
fn state_color_attachment(
    attachment: &Retained<MTLRenderPipelineColorAttachmentDescriptor>,
    stated: &ColorAway,
) {
    attachment.setPixelFormat(MTLPixelFormat(stated.pixel_format.as_raw() as usize));
    attachment.setBlendingEnabled(stated.blending_enabled);
    attachment.setSourceRGBBlendFactor(MTLBlendFactor(stated.source_rgb.as_raw() as usize));
    attachment
        .setDestinationRGBBlendFactor(MTLBlendFactor(stated.destination_rgb.as_raw() as usize));
    attachment.setRgbBlendOperation(MTLBlendOperation(stated.rgb_operation.as_raw() as usize));
    attachment.setSourceAlphaBlendFactor(MTLBlendFactor(stated.source_alpha.as_raw() as usize));
    attachment
        .setDestinationAlphaBlendFactor(MTLBlendFactor(stated.destination_alpha.as_raw() as usize));
    attachment.setAlphaBlendOperation(MTLBlendOperation(stated.alpha_operation.as_raw() as usize));
    attachment.setWriteMask(MTLColorWriteMask(stated.write_mask.as_raw() as usize));
}

/// What one colour attachment answers, every getter of its own surface.
fn read_color_attachment(
    attachment: &Retained<MTLRenderPipelineColorAttachmentDescriptor>,
) -> ColorFacts {
    ColorFacts {
        pixel_format: attachment.pixelFormat().0 as u32,
        blending_enabled: attachment.isBlendingEnabled(),
        source_rgb: attachment.sourceRGBBlendFactor().0 as u32,
        destination_rgb: attachment.destinationRGBBlendFactor().0 as u32,
        rgb_operation: attachment.rgbBlendOperation().0 as u32,
        source_alpha: attachment.sourceAlphaBlendFactor().0 as u32,
        destination_alpha: attachment.destinationAlphaBlendFactor().0 as u32,
        alpha_operation: attachment.alphaBlendOperation().0 as u32,
        write_mask: attachment.writeMask().0 as u32,
    }
}

/// `MTLLinkedFunctions`' `functions`, `privateFunctions` and `binaryFunctions`, which answer the
/// same shape.
type Functions = Option<Retained<NSArray<ProtocolObject<dyn MTLFunction>>>>;

/// `MTLLinkedFunctions`' `groups`, by call-site group name.
type Groups = Option<Retained<NSDictionary<NSString, NSArray<ProtocolObject<dyn MTLFunction>>>>>;

/// How many functions a stage's `MTLLinkedFunctions` holds, by kind; a list the object answers nil
/// for is none, for the reason [`crate::render_descriptors`] gives.
fn linked_counts(linked: &Retained<MTLLinkedFunctions>) -> LinkedCounts {
    let functions: Functions = linked.functions();
    let private: Functions = linked.privateFunctions();
    let binary: Functions = linked.binaryFunctions();
    let groups: Groups = linked.groups();
    LinkedCounts {
        functions: functions.map_or(0, |held| held.len() as u64),
        private_functions: private.map_or(0, |held| held.len() as u64),
        binary_functions: binary.map_or(0, |held| held.len() as u64),
        groups: groups.map_or(0, |held| held.len() as u64),
    }
}

/// Metal's vertex descriptor for `layout`. The case's enumerations are Metal's own, so each one
/// crosses as its raw value.
fn vertex_descriptor(layout: &crate::render::VertexLayout<'_>) -> Retained<MTLVertexDescriptor> {
    let descriptor = MTLVertexDescriptor::vertexDescriptor();
    for attribute in layout.attributes {
        // SAFETY: the index is below the attribute array every descriptor holds, which the case's
        // own guard keeps it to.
        let held = unsafe {
            descriptor
                .attributes()
                .objectAtIndexedSubscript(attribute.index as usize)
        };
        held.setFormat(MTLVertexFormat(attribute.format.as_raw() as usize));
        // SAFETY: the offset and the buffer index are the case's own numbers, both inside the
        // record the case binds and the buffer array the descriptor holds.
        unsafe {
            held.setOffset(attribute.offset as usize);
            held.setBufferIndex(attribute.buffer as usize);
        }
    }
    for buffer in layout.layouts {
        // SAFETY: as the attributes', for the layout array.
        let held = unsafe {
            descriptor
                .layouts()
                .objectAtIndexedSubscript(buffer.index as usize)
        };
        held.setStepFunction(MTLVertexStepFunction(buffer.step.as_raw() as usize));
        // SAFETY: the stride is the case's own record size, or Metal's own
        // `MTLBufferLayoutStrideDynamic`, and the rate is the step function's own, 0 for the
        // constant step and 1 otherwise.
        unsafe {
            held.setStride(match buffer.stride {
                VertexStride::Bytes(stride) => stride as usize,
                VertexStride::Dynamic => STRIDE_DYNAMIC,
            });
            held.setStepRate(buffer.rate as usize);
        }
    }
    descriptor
}

/// Metal's stage-input descriptor for `stage_input`. The case's enumerations are Metal's own, so
/// each reaches `MTLAttributeFormat`, `MTLStepFunction` and `MTLIndexType` by its raw ordinal.
fn stage_input_descriptor(stage_input: &StageInput) -> Retained<MTLStageInputOutputDescriptor> {
    let descriptor = MTLStageInputOutputDescriptor::stageInputOutputDescriptor();
    let attributes = descriptor.attributes();
    for attribute in &stage_input.attributes {
        // SAFETY: `objectAtIndexedSubscript:` is Metal's own accessor for an attribute index the
        // device's attribute limit covers, which the case's indices are inside.
        let written = unsafe { attributes.objectAtIndexedSubscript(attribute.index as usize) };
        written.setFormat(MTLAttributeFormat(attribute.format.as_raw() as usize));
        written.setOffset(attribute.offset as usize);
        // SAFETY: the buffer index is inside the device's buffer argument table, which the
        // case's indices are.
        unsafe { written.setBufferIndex(attribute.buffer_index as usize) };
    }
    let layouts = descriptor.layouts();
    for layout in &stage_input.layouts {
        // SAFETY: as above, for a buffer index inside the device's buffer limit.
        let written = unsafe { layouts.objectAtIndexedSubscript(layout.buffer_index as usize) };
        written.setStride(
            layout
                .stride
                .map_or(STRIDE_DYNAMIC, |stride| stride as usize),
        );
        written.setStepFunction(MTLStepFunction(layout.step_function.as_raw() as usize));
        written.setStepRate(layout.step_rate as usize);
    }
    descriptor.setIndexType(MTLIndexType(stage_input.index_type.as_raw() as usize));
    // SAFETY: as above, for the index buffer's own slot.
    unsafe { descriptor.setIndexBufferIndex(stage_input.index_buffer_index as usize) };
    descriptor
}

/// `kept` with `value` pushed on, answering its index.
/// The `MTLTextureSwizzle` a channel's source is, by the raw ordinal both enumerations share.
fn swizzle_of(source: metal2vulkan::metal::TextureSwizzle) -> MTLTextureSwizzle {
    MTLTextureSwizzle(source.as_raw() as u8)
}

fn keep<T>(kept: &mut Vec<T>, value: T) -> Handle {
    kept.push(value);
    Handle(u32::try_from(kept.len() - 1).expect("fewer than 2^32 objects"))
}

/// Metal's texture descriptor for the one a case states: every field of
/// [`TextureRequest`](crate::texture::TextureRequest) reaches it.
///
/// A function of its own, and not inlined in the call, so that
/// `both_sides_state_every_field_of_a_texture_request` can hold the two sides' descriptors to the
/// same request field by field with no device. Making an `MTLTextureDescriptor` asks nothing of a
/// device, so that guard runs wherever Metal's frameworks are.
///
/// [`crate::texture::Over`] is **not** here: it is which object the call is sent to, not a field of
/// the descriptor.
pub(crate) fn texture_descriptor(
    request: &crate::texture::TextureRequest<'_>,
) -> Retained<MTLTextureDescriptor> {
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setPixelFormat(MTLPixelFormat(request.format as usize));
    descriptor.setTextureType(MTLTextureType(request.texture_type.as_raw() as usize));
    // SAFETY: the setters take the descriptor's own words; nothing is borrowed.
    unsafe {
        descriptor.setWidth(request.size[0] as usize);
        descriptor.setHeight(request.size[1] as usize);
        descriptor.setDepth(request.size[2].max(1) as usize);
        descriptor.setMipmapLevelCount(request.levels.max(1) as usize);
        descriptor.setArrayLength(request.slices.max(1) as usize);
    }
    descriptor.setUsage(MTLTextureUsage(request.usage.as_raw() as usize));
    descriptor.setStorageMode(if request.private {
        MTLStorageMode::Private
    } else {
        MTLStorageMode::Shared
    });
    descriptor.setAllowGPUOptimizedContents(request.gpu_optimized);
    descriptor.setSwizzle(MTLTextureSwizzleChannels {
        red: swizzle_of(request.swizzle.red),
        green: swizzle_of(request.swizzle.green),
        blue: swizzle_of(request.swizzle.blue),
        alpha: swizzle_of(request.swizzle.alpha),
    });
    descriptor
}

/// Run `xcrun tool args`, answering why it could not.
fn xcrun(tool: &str, args: &[&std::ffi::OsStr]) -> Result<(), String> {
    let output = std::process::Command::new("/usr/bin/xcrun")
        .arg(tool)
        .args(args)
        .output()
        .map_err(|error| format!("xcrun {tool}: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "xcrun {tool} failed: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

impl MetalSide {
    /// The pipeline state each distinct kernel a program's per-sample read-backs name, in the
    /// order they are first named.
    ///
    /// Empty for a program with no such read-back, so a program of copies alone compiles nothing.
    fn read_back_pipelines(
        &mut self,
        work: &crate::work::Work,
    ) -> Answer<Vec<crate::metal_work::Kernel>> {
        let mut wanted: Vec<&'static str> = Vec::new();
        for step in &work.steps {
            if let crate::work::Step::ReadSamples { kernel, .. } = step {
                if !wanted.contains(kernel) {
                    wanted.push(kernel);
                }
            }
        }
        if wanted.is_empty() {
            return Ok(Vec::new());
        }
        let fixture = work.kernels.ok_or_else(|| {
            Failure::named("NoKernels", "the program states no fixture for its kernels")
        })?;
        let loaded = crate::fixture::Fixture::new(fixture)
            .loaded()
            .map_err(|why| Failure::named("NoFixture", why))?;
        let library = self.library(&loaded.source())?;
        let mut made = Vec::with_capacity(wanted.len());
        for kernel in wanted {
            let handle = self.compute_pipeline(library, kernel)?;
            let state = self
                .pipelines
                .get(handle.0 as usize)
                .cloned()
                .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", handle.0)))?;
            made.push((kernel, state));
        }
        Ok(made)
    }
}

impl Side for MetalSide {
    fn venue(&self) -> Venue {
        Venue {
            device: self.device.name().to_string(),
            driver: "Metal".to_owned(),
            api: "Metal".to_owned(),
            metal_level: METAL_LEVEL.to_owned(),
        }
    }

    fn library(&mut self, source: &ShaderSource<'_>) -> Answer<Handle> {
        let serial = self.libraries.len();
        let text = self.scratch.0.join(format!("{serial}.air.ll"));
        let air = self.scratch.0.join(format!("{serial}.air"));
        let metallib = self.scratch.0.join(format!("{serial}.metallib"));
        let build = || -> Result<(), String> {
            std::fs::write(&text, source.air_text)
                .map_err(|error| format!("{}: {error}", text.display()))?;
            xcrun(
                "metal-as",
                &[text.as_os_str(), "-o".as_ref(), air.as_os_str()],
            )?;
            xcrun(
                "metallib",
                &[air.as_os_str(), "-o".as_ref(), metallib.as_os_str()],
            )
        };
        build().map_err(|why| Failure::named("MetalToolFailed", why))?;
        let path = metallib
            .to_str()
            .ok_or_else(|| Failure::named("NonUtf8Path", metallib.display().to_string()))?;
        let library = self
            .device
            .newLibraryWithURL_error(&NSURL::fileURLWithPath(&NSString::from_str(path)))
            .map_err(|error| Failure::named("NewLibraryWithURL", error.to_string()))?;
        Ok(keep(&mut self.libraries, library))
    }

    fn compute_pipeline(&mut self, library: Handle, entry: &str) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", entry.to_owned()))?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| Failure::named("NewComputePipelineState", error.to_string()))?;
        Ok(keep(&mut self.pipelines, pipeline))
    }

    fn visible_function_table(
        &mut self,
        caller: Handle,
        functions: Handle,
        spec: &TableSpec<'_>,
    ) -> Answer<TableOutcome> {
        let caller = self
            .libraries
            .get(caller.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", caller.0)))?
            .clone();
        let defining = self
            .libraries
            .get(functions.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", functions.0)))?
            .clone();
        let named = |library: &ProtocolObject<dyn MTLLibrary>,
                     name: &str|
         -> Answer<Retained<ProtocolObject<dyn MTLFunction>>> {
            library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| Failure::named("NewFunctionWithName", name.to_owned()))
        };
        // `MTLLinkedFunctions`: each list in the order the spec states, because a handle's word is
        // the function's position among the callable ones.
        let linked = MTLLinkedFunctions::linkedFunctions();
        for list in [SpecLinkedList::Functions, SpecLinkedList::PrivateFunctions] {
            let mut held = Vec::new();
            for (name, of) in spec.linked.functions {
                if *of == list {
                    held.push(named(&defining, name)?);
                }
            }
            if held.is_empty() {
                continue;
            }
            let array = NSArray::from_retained_slice(&held);
            match list {
                SpecLinkedList::Functions => linked.setFunctions(Some(&array)),
                SpecLinkedList::PrivateFunctions => linked.setPrivateFunctions(Some(&array)),
            }
        }
        let entry = named(&caller, spec.entry)?;
        let maker = match spec.maker {
            Maker::Compute => {
                let descriptor = MTLComputePipelineDescriptor::new();
                descriptor.setComputeFunction(Some(&entry));
                descriptor.setLinkedFunctions(Some(&linked));
                let state = self
                    .device
                    .newComputePipelineStateWithDescriptor_options_reflection_error(
                        &descriptor,
                        MTLPipelineOption::None,
                        None,
                    )
                    .map_err(|error| {
                        Failure::named(
                            "NewComputePipelineState",
                            error.localizedDescription().to_string(),
                        )
                    })?;
                MetalMaker::Compute(state)
            }
            Maker::DrawingStage(stage) | Maker::TileStage(stage) => {
                let descriptor = MTLRenderPipelineDescriptor::new();
                descriptor.setVertexFunction(Some(&entry));
                descriptor.setVertexLinkedFunctions(Some(&linked));
                // The caller's vertex function returns nothing and the state has no fragment
                // function, so the state rasterizes nothing and declares no attachment.
                descriptor.setRasterizationEnabled(false);
                let state = self
                    .device
                    .newRenderPipelineStateWithDescriptor_error(&descriptor)
                    .map_err(|error| {
                        Failure::named(
                            "NewRenderPipelineState",
                            error.localizedDescription().to_string(),
                        )
                    })?;
                MetalMaker::Render(state, MTLRenderStages(stage.stages().as_raw() as usize))
            }
        };
        let registry = self.device.registryID();
        let mut handles = Vec::new();
        let mut function_types = Vec::new();
        let mut held: Vec<AskedHandle<'_>> = Vec::new();
        for name in spec.handles {
            let function = named(&defining, name)?;
            function_types.push(our_function_type(function.functionType().0)?);
            // The receivers below name their Metal type where a pattern or an inference would
            // leave it unspelled. The coverage derivation types a receiver from a declaration
            // alone, and an untyped one costs its selector the credit and reads in the generated
            // document exactly like a call no case makes.
            let handle: Option<Retained<ProtocolObject<dyn MTLFunctionHandle>>> =
                maker.function_handle(&function);
            let facts =
                handle
                    .as_ref()
                    .map(
                        |handle: &Retained<ProtocolObject<dyn MTLFunctionHandle>>| HandleFacts {
                            name: handle.name().to_string(),
                            device_is_the_cases: handle.device().registryID() == registry,
                        },
                    );
            handles.push(Asked::answered(facts));
            held.push((*name, handle));
        }
        let handle_of = |name: &str| {
            held.iter()
                .find(|(of, _)| *of == name)
                .and_then(|(_, handle)| handle.as_deref())
        };
        let wanted = MTLVisibleFunctionTableDescriptor::visibleFunctionTableDescriptor();
        // SAFETY: the count is the case's own, which the family caps at 3 — a count near the
        // device's maximum ends the process, which `crate::visible_tables` records.
        unsafe { wanted.setFunctionCount(spec.function_count as NSUInteger) };
        let table: Option<Retained<ProtocolObject<dyn MTLVisibleFunctionTable>>> =
            maker.new_visible_function_table(&wanted);
        let second: Option<Option<Retained<ProtocolObject<dyn MTLVisibleFunctionTable>>>> = spec
            .second_table
            .then(|| maker.new_visible_function_table(&wanted));
        let descriptor_function_count = wanted.functionCount() as u64;
        let facts = table.as_ref().map(
            |table: &Retained<ProtocolObject<dyn MTLVisibleFunctionTable>>| TableFacts {
                resource_id_is_non_zero: table.gpuResourceID().to_raw() != 0,
                a_second_tables_id_differs: second.map(
                    |second: Option<Retained<ProtocolObject<dyn MTLVisibleFunctionTable>>>| {
                        second.is_some_and(
                            |second: Retained<ProtocolObject<dyn MTLVisibleFunctionTable>>| {
                                second.gpuResourceID().to_raw() != table.gpuResourceID().to_raw()
                            },
                        )
                    },
                ),
                device_is_the_cases: table.device().registryID() == registry,
            },
        );
        // The entries, by whichever setter the case names. Metal's setters answer nothing, so a
        // side that took them records `Ok`.
        if let Some(table) = &table {
            let table: &ProtocolObject<dyn MTLVisibleFunctionTable> = table;
            let taken: Vec<Option<&ProtocolObject<dyn MTLFunctionHandle>>> = spec
                .entries
                .iter()
                .map(|entry| match entry {
                    Entry::Function(name) => handle_of(name),
                    Entry::Nil => None,
                })
                .collect();
            match spec.writes {
                Writes::OneAtATime => {
                    for (at, handle) in taken.iter().enumerate() {
                        // SAFETY: the handle is the table's own state's, and `at` is below the
                        // count the table was made with (the case states no more entries than its
                        // `function_count`), so Metal bounds-checks nothing this call needs.
                        unsafe { table.setFunction_atIndex(*handle, at as NSUInteger) };
                    }
                }
                Writes::WholeRange if taken.is_empty() => {}
                Writes::WholeRange => {
                    let raw: Vec<*const ProtocolObject<dyn MTLFunctionHandle>> = taken
                        .iter()
                        .map(|handle| match handle {
                            Some(handle) => std::ptr::from_ref(*handle),
                            None => std::ptr::null(),
                        })
                        .collect();
                    let range = NSRange::new(0, raw.len());
                    // SAFETY: `raw` holds `range.length` pointers, each a handle of the table's
                    // own state or null, and the range lies inside the table's count.
                    unsafe {
                        table.setFunctions_withRange(
                            std::ptr::NonNull::new(raw.as_ptr().cast_mut()).unwrap_unchecked(),
                            range,
                        );
                    }
                }
            }
        }
        let Some(threads) = spec.threads else {
            return Ok(TableOutcome::read_only(
                handles,
                descriptor_function_count,
                function_types,
                MadeTable::answered(facts),
            ));
        };
        let MetalMaker::Compute(state) = &maker else {
            return Err(Failure::named(
                "NoDispatch",
                "a render stage's table has no dispatch in this family".to_owned(),
            ));
        };
        let out = self
            .device
            .newBufferWithLength_options(
                threads as NSUInteger * 4,
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| Failure::named("NewBuffer", "returned nil".to_owned()))?;
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| Failure::named("ComputeCommandEncoder", "returned nil".to_owned()))?;
        encoder.setComputePipelineState(state);
        // SAFETY: the buffer is `threads * 4` bytes, which is what the kernel writes at slot 0,
        // and the encoder outlives the call.
        unsafe { encoder.setBuffer_offset_atIndex(Some(&out), 0, 0) };
        if let (Binding::Bound, Some(table)) = (spec.binding, &table) {
            // SAFETY: the table is this state's own and `TABLE_INDEX` is the index the fixture
            // declares its `visible_function_table` at; the encoder outlives the call.
            unsafe { encoder.setVisibleFunctionTable_atBufferIndex(Some(table), TABLE_INDEX) };
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(size([threads, 1, 1]), size([1, 1, 1]));
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let status = Value::unit_variant(status_word(command_buffer.status()));
        // SAFETY: the buffer is `threads * 4` bytes of shared storage the host reaches, and the
        // wait above returned before they are read.
        let bytes = unsafe {
            std::slice::from_raw_parts(out.contents().as_ptr().cast::<u8>(), threads as usize * 4)
        };
        let words = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        Ok(TableOutcome {
            handles,
            descriptor_function_count,
            function_types,
            table: MadeTable::answered(facts),
            entries_written: Ok(()),
            words,
            status,
        })
    }

    fn static_threadgroup_memory_length(&mut self, pipeline: Handle) -> Answer<u64> {
        let state = self
            .pipelines
            .get(pipeline.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", pipeline.0)))?;
        Ok(state.staticThreadgroupMemoryLength() as u64)
    }

    fn buffer(&mut self, bytes: &[u8]) -> Answer<Handle> {
        let handle = self.zeroed_buffer(bytes.len())?;
        let buffer = self.held(handle)?;
        // SAFETY: the buffer was just made with at least `bytes.len()` bytes of shared storage,
        // and no command uses it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.contents().as_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
        Ok(handle)
    }

    fn zeroed_buffer(&mut self, length: usize) -> Answer<Handle> {
        let buffer = self
            .device
            .newBufferWithLength_options(length.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                Failure::named(
                    "NewBufferWithLength",
                    format!("{length} bytes returned nil"),
                )
            })?;
        Ok(keep(&mut self.buffers, buffer))
    }

    fn host_buffer(&mut self, spec: &crate::host_pages::HostBufferSpec) -> Answer<Handle> {
        if let Some(why) = spec.unmappable() {
            return Err(Failure::named("UnmappableHostBuffer", why));
        }
        let length = usize::try_from(spec.bytes)
            .map_err(|_| Failure::named("UnmappableHostBuffer", "more bytes than a usize"))?;
        let pages = crate::host_pages::HostPages::of(length, spec.pattern)
            .ok_or_else(|| Failure::named("MmapFailed", format!("{length} bytes")))?;
        // Metal's private `MTLResourceOptions` bit 24: the memory takes the GPU's writes nowhere.
        // Written as the raw bit because no public header names it, which is also why the suite
        // reaches it as a number rather than through a binding.
        let private_bit = MTLResourceOptions(1 << 24);
        let mut options = MTLResourceOptions::StorageModeShared;
        if spec.read_only {
            options |= private_bit;
        }
        // SAFETY: the pages are this side's own mapping, `length` bytes long, and they stay mapped
        // for as long as the buffer: `host_pages` is declared after `buffers`, so the buffer is
        // released first. The deallocator is nil, so Metal frees nothing of ours.
        let buffer = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    pages.pointer().cast(),
                    length,
                    options,
                    None,
                )
        }
        .ok_or_else(|| {
            Failure::named(
                "NewBufferWithBytesNoCopy",
                format!("{length} bytes of options {:#x} returned nil", options.0),
            )
        })?;
        self.host_pages.push(pages);
        Ok(keep(&mut self.buffers, buffer))
    }

    fn texture_2d(
        &mut self,
        format: PixelFormat,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<Handle> {
        // SAFETY: the descriptor constructor reads only the format and the size it is given.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat(format.as_raw() as usize),
                size[0] as usize,
                size[1] as usize,
                false,
            )
        };
        descriptor.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        descriptor.setStorageMode(MTLStorageMode::Shared);
        let texture = self
            .device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!("{format:?} {}x{} returned nil", size[0], size[1]),
                )
            })?;
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: size[0] as usize,
                height: size[1] as usize,
                depth: 1,
            },
        };
        if bytes.len() < row_pitch as usize * size[1] as usize {
            return Err(Failure::named(
                "TooFewBytes",
                format!("{} bytes for {} rows of {row_pitch}", bytes.len(), size[1]),
            ));
        }
        // SAFETY: `bytes` holds `size[1]` rows of `row_pitch` bytes, checked just above, which is
        // the region the call writes.
        unsafe {
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                0,
                std::ptr::NonNull::from(bytes).cast(),
                row_pitch as usize,
            );
        }
        Ok(keep(&mut self.textures, texture))
    }

    fn read_texture(&mut self, texture: Handle, size: [u32; 2], row_pitch: u32) -> Answer<Vec<u8>> {
        let texture = self.held_texture(texture)?;
        let mut bytes = vec![0u8; row_pitch as usize * size[1] as usize];
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: size[0] as usize,
                height: size[1] as usize,
                depth: 1,
            },
        };
        // SAFETY: `bytes` is `size[1]` rows of `row_pitch` bytes, which is the region the call
        // reads, and the caller waited for the command buffer that wrote the texture.
        unsafe {
            texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                std::ptr::NonNull::from(bytes.as_mut_slice()).cast(),
                row_pitch as usize,
                region,
                0,
            );
        }
        Ok(bytes)
    }

    fn texture_2d_in_role(&mut self, spec: &TextureSpec, bytes: &[u8]) -> Answer<Handle> {
        // SAFETY: the descriptor constructor reads only the format and the size it is given.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat(spec.format.as_raw() as usize),
                spec.size[0] as usize,
                spec.size[1] as usize,
                false,
            )
        };
        descriptor.setUsage(match spec.role {
            TextureRole::Read => MTLTextureUsage::ShaderRead,
            TextureRole::Write => MTLTextureUsage::ShaderWrite,
            TextureRole::ReadWrite => MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
            TextureRole::RenderTarget => MTLTextureUsage::RenderTarget,
        });
        descriptor.setStorageMode(MTLStorageMode::Shared);
        let texture = self
            .device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!(
                        "{:?} {}x{} returned nil",
                        spec.format, spec.size[0], spec.size[1]
                    ),
                )
            })?;
        if !bytes.is_empty() {
            if bytes.len() < spec.row_pitch as usize * spec.size[1] as usize {
                return Err(Failure::named(
                    "TooFewBytes",
                    format!(
                        "{} bytes for {} rows of {}",
                        bytes.len(),
                        spec.size[1],
                        spec.row_pitch
                    ),
                ));
            }
            let region = MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize {
                    width: spec.size[0] as usize,
                    height: spec.size[1] as usize,
                    depth: 1,
                },
            };
            // SAFETY: `bytes` holds the rows of the pitch the region writes, checked just above.
            unsafe {
                texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                    region,
                    0,
                    std::ptr::NonNull::from(bytes).cast(),
                    spec.row_pitch as usize,
                );
            }
        }
        Ok(keep(&mut self.textures, texture))
    }

    fn texture(&mut self, request: &crate::texture::TextureRequest<'_>) -> Answer<Handle> {
        let descriptor = texture_descriptor(request);
        let swizzle = descriptor.swizzle();
        // One of the three Metal calls that answers a texture of this descriptor, as the request
        // says: the device's own, a buffer's, or a parent texture's view.
        let made = match request.over {
            None => self.device.newTextureWithDescriptor(&descriptor),
            Some(crate::texture::Over::Buffer {
                buffer,
                offset,
                bytes_per_row,
            }) => {
                let held: Retained<ProtocolObject<dyn MTLBuffer>> =
                    Retained::from(self.held(buffer)?);
                held.newTextureWithDescriptor_offset_bytesPerRow(
                    &descriptor,
                    offset as usize,
                    bytes_per_row as usize,
                )
            }
            Some(crate::texture::Over::View { texture }) => {
                let held: Retained<ProtocolObject<dyn MTLTexture>> =
                    Retained::from(self.held_texture(texture)?);
                // SAFETY: the level and slice ranges lie inside the parent, which the case's own
                // request for the parent states, and the format and type are a view Metal makes of
                // it or nil.
                unsafe {
                    held.newTextureViewWithPixelFormat_textureType_levels_slices_swizzle(
                        MTLPixelFormat(request.format as usize),
                        MTLTextureType(request.texture_type.as_raw() as usize),
                        objc2_foundation::NSRange::new(0, request.levels.max(1) as usize),
                        objc2_foundation::NSRange::new(0, request.slices.max(1) as usize),
                        swizzle,
                    )
                }
            }
        };
        let texture = made.ok_or_else(|| {
            Failure::named(
                "NewTextureWithDescriptor",
                format!(
                    "format {} {:?} {}x{}x{} returned nil",
                    request.format,
                    request.texture_type,
                    request.size[0],
                    request.size[1],
                    request.size[2]
                ),
            )
        })?;
        for upload in request.uploads {
            if upload.bytes.len() < upload.needed() {
                return Err(Failure::named(
                    "TooFewBytes",
                    format!(
                        "{} bytes for {} rows of {}",
                        upload.bytes.len(),
                        upload.rows,
                        upload.row_pitch
                    ),
                ));
            }
            let region = MTLRegion {
                origin: MTLOrigin {
                    x: upload.origin[0] as usize,
                    y: upload.origin[1] as usize,
                    z: upload.origin[2] as usize,
                },
                size: MTLSize {
                    width: upload.size[0] as usize,
                    height: upload.size[1] as usize,
                    depth: upload.size[2].max(1) as usize,
                },
            };
            // SAFETY: `upload.bytes` holds the rows of the pitch the region writes, checked just
            // above, and the region lies in the level and slice the texture was made with.
            unsafe {
                texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    region,
                    upload.level as usize,
                    upload.slice as usize,
                    std::ptr::NonNull::from(upload.bytes).cast(),
                    upload.row_pitch as usize,
                    upload.image_pitch as usize,
                );
            }
        }
        Ok(keep(&mut self.textures, texture))
    }

    fn replace_region(
        &mut self,
        texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<()> {
        if bytes.len() < row_pitch as usize * size[1] as usize {
            return Err(Failure::named(
                "TooFewBytes",
                format!("{} bytes for {} rows of {row_pitch}", bytes.len(), size[1]),
            ));
        }
        let texture: Retained<ProtocolObject<dyn MTLTexture>> =
            Retained::from(self.held_texture(texture)?);
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: size[0] as usize,
                height: size[1] as usize,
                depth: 1,
            },
        };
        // SAFETY: `bytes` holds `size[1]` rows of `row_pitch` bytes, checked just above, which is
        // the region the call writes.
        unsafe {
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                0,
                std::ptr::NonNull::from(bytes).cast(),
                row_pitch as usize,
            );
        }
        Ok(())
    }

    fn sampler(&mut self, spec: &SamplerSpec) -> Answer<Handle> {
        let filter = |filter: metal2vulkan::metal::SamplerMinMagFilter| {
            MTLSamplerMinMagFilter(filter.as_raw() as usize)
        };
        let address = |mode: metal2vulkan::metal::SamplerAddressMode| {
            MTLSamplerAddressMode(mode.as_raw() as usize)
        };
        let descriptor = MTLSamplerDescriptor::new();
        descriptor.setMinFilter(filter(spec.min_filter));
        descriptor.setMagFilter(filter(spec.mag_filter));
        descriptor.setMipFilter(MTLSamplerMipFilter(spec.mip_filter.as_raw() as usize));
        descriptor.setSAddressMode(address(spec.s_address_mode));
        descriptor.setTAddressMode(address(spec.t_address_mode));
        descriptor.setRAddressMode(address(spec.r_address_mode));
        descriptor.setBorderColor(MTLSamplerBorderColor(spec.border_color.as_raw() as usize));
        descriptor.setNormalizedCoordinates(spec.normalized_coordinates);
        descriptor.setLodMinClamp(spec.lod_min_clamp);
        descriptor.setLodMaxClamp(spec.lod_max_clamp);
        descriptor.setLodAverage(spec.lod_average);
        descriptor.setMaxAnisotropy(spec.max_anisotropy as usize);
        descriptor.setCompareFunction(compare_function(spec.compare_function));
        descriptor.setReductionMode(MTLSamplerReductionMode(
            spec.reduction_mode.as_raw() as usize
        ));
        descriptor.setSupportArgumentBuffers(spec.support_argument_buffers);
        let sampler = self
            .device
            .newSamplerStateWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewSamplerStateWithDescriptor",
                    format!("{spec:?} returned nil"),
                )
            })?;
        Ok(keep(&mut self.samplers, sampler))
    }

    fn argument_buffer(&mut self, slots: &[ArgumentSlot]) -> Answer<Handle> {
        let mut words = Vec::with_capacity(slots.len() * 8);
        let mut named = Vec::new();
        for slot in slots {
            let id = match *slot {
                ArgumentSlot::Texture(handle) => {
                    named.push(handle);
                    self.held_texture(handle)?.gpuResourceID().to_raw()
                }
                ArgumentSlot::Sampler(handle) => {
                    self.held_sampler(handle)?.gpuResourceID().to_raw()
                }
            };
            words.extend(id.to_le_bytes());
        }
        let buffer = self.buffer(&words)?;
        self.argument_residents.push((buffer, named));
        Ok(buffer)
    }

    fn argument_encoder(
        &mut self,
        library: Handle,
        entry: &str,
        buffer_index: u32,
    ) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", entry.to_owned()))?;
        // SAFETY: the case names a buffer index the function really takes an argument buffer at,
        // which is what its own fixture declares; Metal raises for any other index, so a case of
        // that shape is not written here.
        let encoder = unsafe { function.newArgumentEncoderWithBufferIndex(buffer_index as usize) };
        Ok(keep(
            &mut self.argument_encoders,
            ArgumentEncoderAt {
                encoder,
                target: None,
            },
        ))
    }

    fn argument_encoder_with_arguments(
        &mut self,
        arguments: &[ArgumentDescriptorSpec],
    ) -> Answer<Handle> {
        let descriptors: Vec<Retained<MTLArgumentDescriptor>> = arguments
            .iter()
            .map(|spec| {
                let descriptor = MTLArgumentDescriptor::argumentDescriptor();
                descriptor.setDataType(MTLDataType(spec.data_type.as_raw() as usize));
                descriptor.setIndex(spec.index as usize);
                descriptor.setArrayLength(spec.array_length as usize);
                descriptor.setAccess(MTLBindingAccess(spec.access.as_raw() as usize));
                descriptor.setTextureType(MTLTextureType(spec.texture_type.as_raw() as usize));
                descriptor.setConstantBlockAlignment(spec.constant_block_alignment as usize);
                descriptor
            })
            .collect();
        let encoder = self
            .device
            .newArgumentEncoderWithArguments(&NSArray::from_retained_slice(&descriptors))
            .ok_or_else(|| {
                Failure::named(
                    "NewArgumentEncoderWithArguments",
                    format!("{} descriptor(s) returned nil", descriptors.len()),
                )
            })?;
        Ok(keep(
            &mut self.argument_encoders,
            ArgumentEncoderAt {
                encoder,
                target: None,
            },
        ))
    }

    fn argument_encoder_facts(&mut self, encoder: Handle, label: &str) -> Answer<EncoderFacts> {
        let registry = self.device.registryID();
        let held = self.held_encoder(encoder)?;
        held.setLabel(Some(&NSString::from_str(label)));
        Ok(EncoderFacts {
            encoded_length: held.encodedLength() as u64,
            alignment: held.alignment() as u64,
            label: held.label().map(|label| label.to_string()),
            // The registry ID is each host's own number, so only the **comparison** is portable:
            // the two sides agree that the encoder came from the case's device, never on the
            // number that says so.
            device_is_the_cases: held.device().registryID() == registry,
        })
    }

    fn set_argument_buffer(
        &mut self,
        encoder: Handle,
        buffer: Handle,
        offset: u64,
        element: Option<u64>,
    ) -> Answer<()> {
        let held_buffer: Retained<ProtocolObject<dyn MTLBuffer>> =
            Retained::from(self.held(buffer)?);
        let held = self.held_encoder(encoder)?;
        let arguments = offset + element.unwrap_or(0) * held.encodedLength() as u64;
        // SAFETY: the buffer outlives the encoder's use of it (this side's own pool owns it), and
        // the case states an offset whose arguments lie inside its own buffer.
        unsafe {
            match element {
                None => held.setArgumentBuffer_offset(Some(&held_buffer), offset as usize),
                Some(element) => held.setArgumentBuffer_startOffset_arrayElement(
                    Some(&held_buffer),
                    offset as usize,
                    element as usize,
                ),
            }
        }
        self.argument_encoders[encoder.0 as usize].target = Some((buffer, arguments));
        Ok(())
    }

    fn nested_argument_encoder(&mut self, encoder: Handle, index: u32) -> Answer<Option<Handle>> {
        let held = self.held_encoder(encoder)?;
        // SAFETY: Metal answers nil for every index that names no nested argument buffer rather
        // than raising, which the probe measured over a plain pointer, a pointer to a struct of
        // constants, an inline struct of resources, a texture, two arrays and an index past the
        // layout, so no index a case can name ends the process here.
        let nested = unsafe { held.newArgumentEncoderForBufferAtIndex(index as usize) };
        Ok(nested.map(|nested| {
            keep(
                &mut self.argument_encoders,
                ArgumentEncoderAt {
                    encoder: nested,
                    target: None,
                },
            )
        }))
    }

    fn encode_argument(&mut self, encoder: Handle, write: &ArgumentWrite) -> Answer<()> {
        let target = self
            .argument_encoders
            .get(encoder.0 as usize)
            .and_then(|at| at.target)
            .map(|(buffer, _)| buffer);
        let textures = |handles: &[Option<Handle>]| -> Answer<Vec<_>> {
            handles
                .iter()
                .map(|handle| {
                    handle
                        .map(|h| self.held_texture(h).map(Retained::from))
                        .transpose()
                })
                .collect()
        };
        let samplers = |handles: &[Option<Handle>]| -> Answer<Vec<_>> {
            handles
                .iter()
                .map(|handle| {
                    handle
                        .map(|h| self.held_sampler(h).map(Retained::from))
                        .transpose()
                })
                .collect()
        };
        let buffers = |handles: &[Option<Handle>]| -> Answer<Vec<_>> {
            handles
                .iter()
                .map(|handle| handle.map(|h| self.held(h).map(Retained::from)).transpose())
                .collect()
        };
        let computes = |handles: &[Option<Handle>]| -> Answer<Vec<_>> {
            handles
                .iter()
                .map(|handle| {
                    handle
                        .map(|h| {
                            self.pipelines
                                .get(h.0 as usize)
                                .map(|state| Retained::from(&**state))
                                .ok_or_else(|| {
                                    Failure::named("NoSuchHandle", format!("pipeline {}", h.0))
                                })
                        })
                        .transpose()
                })
                .collect()
        };
        let renders = |handles: &[Option<Handle>]| -> Answer<Vec<_>> {
            handles
                .iter()
                .map(|handle| {
                    handle
                        .map(|h| {
                            self.render_pipelines
                                .get(h.0 as usize)
                                .map(|held| Retained::from(&*held.state))
                                .ok_or_else(|| {
                                    Failure::named(
                                        "NoSuchHandle",
                                        format!("render pipeline {}", h.0),
                                    )
                                })
                        })
                        .transpose()
                })
                .collect()
        };
        let commands = |handles: &[Option<Handle>]| -> Answer<Vec<_>> {
            handles
                .iter()
                .map(|handle| {
                    handle
                        .map(|h| {
                            self.command_buffers_indirect
                                .get(h.0 as usize)
                                .map(|held| Retained::from(&**held))
                                .ok_or_else(|| {
                                    Failure::named(
                                        "NoSuchHandle",
                                        format!("indirect command buffer {}", h.0),
                                    )
                                })
                        })
                        .transpose()
                })
                .collect()
        };
        // Every range setter takes a pointer to an array of Objective-C pointers over the whole
        // range, so each is written through this one shape.
        let held = self.held_encoder(encoder)?;
        match write {
            ArgumentWrite::Texture { texture, index } => {
                let held_textures = textures(&[*texture])?;
                // SAFETY: the texture outlives the call (this side's pool owns it), and the index
                // is the case's own, which Metal answers for or complains about itself.
                unsafe {
                    held.setTexture_atIndex(held_textures[0].as_deref(), *index as usize);
                }
            }
            ArgumentWrite::Textures {
                textures: ts,
                first,
            } => {
                let held_textures = textures(ts)?;
                let pointers = pointer_array(&held_textures);
                let range = range_of(*first, &pointers)?;
                // SAFETY: `pointers` is an array of `pointers.len()` Objective-C pointers, which
                // is the range's length, and each object it names outlives the call.
                unsafe {
                    held.setTextures_withRange(array_head(&pointers)?, range);
                }
            }
            ArgumentWrite::Sampler { sampler, index } => {
                let held_samplers = samplers(&[*sampler])?;
                // SAFETY: as for `setTexture:atIndex:`.
                unsafe {
                    held.setSamplerState_atIndex(held_samplers[0].as_deref(), *index as usize);
                }
            }
            ArgumentWrite::Samplers {
                samplers: ss,
                first,
            } => {
                let held_samplers = samplers(ss)?;
                let pointers = pointer_array(&held_samplers);
                let range = range_of(*first, &pointers)?;
                // SAFETY: as for `setTextures:withRange:`.
                unsafe {
                    held.setSamplerStates_withRange(array_head(&pointers)?, range);
                }
            }
            ArgumentWrite::Buffer {
                buffer,
                offset,
                index,
            } => {
                let held_buffers = buffers(&[*buffer])?;
                // SAFETY: as for `setTexture:atIndex:`; the offset is the case's own and lies
                // inside the buffer it names.
                unsafe {
                    held.setBuffer_offset_atIndex(
                        held_buffers[0].as_deref(),
                        *offset as usize,
                        *index as usize,
                    );
                }
            }
            ArgumentWrite::Buffers { buffers: bs, first } => {
                let handles: Vec<Option<Handle>> = bs.iter().map(|(handle, _)| *handle).collect();
                let held_buffers = buffers(&handles)?;
                let pointers = pointer_array(&held_buffers);
                let offsets: Vec<usize> = bs.iter().map(|(_, at)| *at as usize).collect();
                let range = range_of(*first, &pointers)?;
                // SAFETY: both arrays hold the range's length of elements, and each buffer they
                // name outlives the call.
                unsafe {
                    held.setBuffers_offsets_withRange(
                        array_head(&pointers)?,
                        std::ptr::NonNull::from(&offsets[0]),
                        range,
                    );
                }
            }
            ArgumentWrite::ComputePipelineState { state, index } => {
                let held_states = computes(&[*state])?;
                // SAFETY: as for `setTexture:atIndex:`.
                unsafe {
                    held.setComputePipelineState_atIndex(
                        held_states[0].as_deref(),
                        *index as usize,
                    );
                }
            }
            ArgumentWrite::ComputePipelineStates { states, first } => {
                let held_states = computes(states)?;
                let pointers = pointer_array(&held_states);
                let range = range_of(*first, &pointers)?;
                // SAFETY: as for `setTextures:withRange:`.
                unsafe {
                    held.setComputePipelineStates_withRange(array_head(&pointers)?, range);
                }
            }
            ArgumentWrite::RenderPipelineState { state, index } => {
                let held_states = renders(&[*state])?;
                // SAFETY: as for `setTexture:atIndex:`.
                unsafe {
                    held.setRenderPipelineState_atIndex(held_states[0].as_deref(), *index as usize);
                }
            }
            ArgumentWrite::RenderPipelineStates { states, first } => {
                let held_states = renders(states)?;
                let pointers = pointer_array(&held_states);
                let range = range_of(*first, &pointers)?;
                // SAFETY: as for `setTextures:withRange:`.
                unsafe {
                    held.setRenderPipelineStates_withRange(array_head(&pointers)?, range);
                }
            }
            ArgumentWrite::IndirectCommandBuffer { commands: c, index } => {
                let held_commands = commands(&[*c])?;
                // SAFETY: as for `setTexture:atIndex:`.
                unsafe {
                    held.setIndirectCommandBuffer_atIndex(
                        held_commands[0].as_deref(),
                        *index as usize,
                    );
                }
            }
            ArgumentWrite::IndirectCommandBuffers { commands: c, first } => {
                let held_commands = commands(c)?;
                let pointers = pointer_array(&held_commands);
                let range = range_of(*first, &pointers)?;
                // SAFETY: as for `setTextures:withRange:`.
                unsafe {
                    held.setIndirectCommandBuffers_withRange(array_head(&pointers)?, range);
                }
            }
            ArgumentWrite::VisibleFunctionTable { table, index } => match table {
                // SAFETY: nil is what the case states, and the index is its own.
                None => unsafe { held.setVisibleFunctionTable_atIndex(None, *index as usize) },
                // No case states one, and neither side holds a pool of them, so both answer the
                // same "no such call" rather than one of them inventing an object.
                Some(_) => return Err(Failure::unavailable("visible_function_table_handle")),
            },
            ArgumentWrite::VisibleFunctionTables { tables, first } => {
                if tables.iter().any(Option::is_some) {
                    return Err(Failure::unavailable("visible_function_table_handle"));
                }
                let pointers: Vec<*const ProtocolObject<dyn MTLVisibleFunctionTable>> =
                    tables.iter().map(|_| std::ptr::null()).collect();
                let range = range_of(*first, &pointers)?;
                // SAFETY: as for `setTextures:withRange:`, over an array of nils.
                unsafe {
                    held.setVisibleFunctionTables_withRange(array_head(&pointers)?, range);
                }
            }
            ArgumentWrite::Constant { index, bytes } => {
                // SAFETY: the index is the case's own; the call answers where the constant goes,
                // and Metal complains itself where no argument has it.
                let at = unsafe { held.constantDataAtIndex(*index as usize) };
                // SAFETY: the address is the constant's inside the argument buffer's contents, and
                // the case states a constant no longer than the argument it names, which its own
                // layout facts compare.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        at.as_ptr().cast::<u8>(),
                        bytes.len(),
                    );
                }
            }
        }
        // A texture a function reaches through an argument buffer is resident only where the
        // encoder was told so, so binding the buffer makes it resident.
        if let Some(buffer) = target {
            let named = write.textures();
            if !named.is_empty() {
                self.argument_residents.push((buffer, named));
            }
        }
        Ok(())
    }

    fn compute_pipeline_bindings(
        &mut self,
        library: Handle,
        entry: &str,
    ) -> Answer<Vec<BindingFacts>> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", entry.to_owned()))?;
        let mut reflection = None;
        // SAFETY: the function is of this library, and `reflection` is this call's own
        // out-parameter.
        let state = unsafe {
            self.device
                .newComputePipelineStateWithFunction_options_reflection_error(
                    &function,
                    MTLPipelineOption::BindingInfo,
                    Some(&mut reflection),
                )
        }
        .map_err(|error| {
            Failure::named(
                "NewComputePipelineState",
                error.localizedDescription().to_string(),
            )
        })?;
        // Declared, not inferred: the coverage derivation types a receiver from what bound it,
        // and an `Option` the creation filled in through an out-parameter names no type at all.
        let reflection: Retained<MTLComputePipelineReflection> = reflection.ok_or_else(|| {
            Failure::named(
                "NoReflection",
                "the creation answered no reflection".to_owned(),
            )
        })?;
        // Bound to a local rather than called in the `for` header: the coverage derivation reads
        // a receiver by walking back from the dot, and in a `for` header that walk swallows the
        // loop variable, so the call would resolve to no type at all.
        let bindings = reflection.bindings();
        let mut found = Vec::new();
        for binding in &bindings {
            found.push(metal_binding(&binding)?);
        }
        // The state is kept so it outlives the reflection its creation answered, as the render
        // pipelines' pool does.
        keep(&mut self.pipelines, state);
        Ok(found)
    }

    fn compute_pipeline_reflection_under(
        &mut self,
        library: Handle,
        entry: &str,
        options: u64,
    ) -> Answer<UnderOptions> {
        let function = self.function_of(library, entry)?;
        let mut reflection = None;
        // SAFETY: the function is of this device's library, and `reflection` is this call's own
        // out-parameter. The options are bits Apple's header names
        // ([`crate::pipeline_option_reflection::MASKS`]), so nothing undefined reaches Metal here.
        let made = unsafe {
            self.device
                .newComputePipelineStateWithFunction_options_reflection_error(
                    &function,
                    MTLPipelineOption(options as usize),
                    Some(&mut reflection),
                )
        };
        let Ok(state) = made else {
            return Ok(UnderOptions::refused());
        };
        // Declared, not inferred: the coverage derivation types a receiver from what bound it, and
        // an `Option` the creation filled in through an out-parameter names no type at all.
        let found = match reflection {
            Some(reflection) => {
                let reflection: Retained<MTLComputePipelineReflection> = reflection;
                // Bound to a local rather than called in the `for` header, for the reason
                // [`Side::compute_pipeline_bindings`] states.
                let bindings = reflection.bindings();
                let mut held = Vec::new();
                for binding in &bindings {
                    held.push(metal_binding(&binding)?);
                }
                Some(held)
            }
            None => None,
        };
        // The state is kept so it outlives the reflection its creation answered.
        keep(&mut self.pipelines, state);
        Ok(UnderOptions {
            refused: false,
            bindings: found,
        })
    }

    fn archives(&mut self, program: &Program<'_>) -> Answer<Vec<Observation>> {
        metal_program(&self.device, &self.scratch, program)
    }

    fn compute_pipeline_under_options(
        &mut self,
        library: Handle,
        entry: &str,
        options: u64,
    ) -> Answer<OptionOutcome> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", entry.to_owned()))?;
        let mut reflection = None;
        // SAFETY: the function is of this library, and `reflection` is this call's own
        // out-parameter. The option is the case's own raw value, which Metal accepts whole: it
        // reads the bits it defines and keeps the rest.
        let state = unsafe {
            self.device
                .newComputePipelineStateWithFunction_options_reflection_error(
                    &function,
                    MTLPipelineOption(options as NSUInteger),
                    Some(&mut reflection),
                )
        }
        .map_err(|error| {
            Failure::named(
                "NewComputePipelineState",
                error.localizedDescription().to_string(),
            )
        })?;
        let outcome = match reflection {
            // Declared, not inferred: the coverage derivation types a receiver from what bound
            // it, and an `Option` the creation filled in through an out-parameter names no type.
            Some(reflection) => {
                let reflection: Retained<MTLComputePipelineReflection> = reflection;
                let bindings = reflection.bindings();
                let mut found = Vec::new();
                for binding in &bindings {
                    found.push(metal_binding(&binding)?);
                }
                OptionOutcome {
                    reflected: true,
                    stages: vec![(KERNEL_STAGE, Some(found))],
                }
            }
            None => OptionOutcome::without_reflection(&[KERNEL_STAGE]),
        };
        // The state is kept so it outlives the reflection its creation answered.
        keep(&mut self.pipelines, state);
        Ok(outcome)
    }

    fn render_pipeline_under_options(
        &mut self,
        library: Handle,
        vertex: &str,
        fragment: &str,
        attachment: PixelFormat,
        options: u64,
    ) -> Answer<OptionOutcome> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = |name: &str| -> Answer<Retained<ProtocolObject<dyn MTLFunction>>> {
            library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| Failure::named("NewFunctionWithName", name.to_owned()))
        };
        let descriptor = MTLRenderPipelineDescriptor::new();
        let (vertex, fragment) = (function(vertex)?, function(fragment)?);
        descriptor.setVertexFunction(Some(&vertex));
        descriptor.setFragmentFunction(Some(&fragment));
        let attachments = descriptor.colorAttachments();
        // SAFETY: attachment 0 is inside the fixed-size array every descriptor holds.
        let first = unsafe { attachments.objectAtIndexedSubscript(0) };
        first.setPixelFormat(MTLPixelFormat(attachment.as_raw() as usize));
        let mut reflection = None;
        let state = self
            .device
            .newRenderPipelineStateWithDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption(options as NSUInteger),
                Some(&mut reflection),
            )
            .map_err(|error| {
                Failure::named(
                    "NewRenderPipelineState",
                    error.localizedDescription().to_string(),
                )
            })?;
        let outcome = match reflection {
            Some(reflection) => {
                let reflection: Retained<MTLRenderPipelineReflection> = reflection;
                let mut stages = Vec::new();
                for (stage, bindings) in [
                    (VERTEX_STAGE, reflection.vertexBindings()),
                    (FRAGMENT_STAGE, reflection.fragmentBindings()),
                ] {
                    let mut found = Vec::new();
                    for binding in &bindings {
                        found.push(metal_binding(&binding)?);
                    }
                    stages.push((stage, Some(found)));
                }
                OptionOutcome {
                    reflected: true,
                    stages,
                }
            }
            None => OptionOutcome::without_reflection(&[VERTEX_STAGE, FRAGMENT_STAGE]),
        };
        // The state is kept so it outlives the reflection its creation answered.
        keep(
            &mut self.render_pipelines,
            RenderPipeline {
                state,
                reflection: None,
            },
        );
        Ok(outcome)
    }

    fn render_pipeline_functions_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &FunctionsDescriptorSpec<'_>,
    ) -> Answer<FunctionsDescriptorFacts> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let mut made: Vec<(&str, Retained<ProtocolObject<dyn MTLFunction>>)> = Vec::new();
        for name in spec.names() {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| Failure::named("NewFunctionWithName", name.to_owned()))?;
            made.push((name, function));
        }
        let held = |name: &str| -> Answer<Retained<ProtocolObject<dyn MTLFunction>>> {
            made.iter()
                .find(|(held, _)| *held == name)
                .map(|(_, function)| function.clone())
                .ok_or_else(|| Failure::named("NoSuchFunction", name.to_owned()))
        };
        let descriptor = MTLRenderPipelineFunctionsDescriptor::new();
        // Each property's writes in the order the case states them, so "set and then set again"
        // and "set and then put back to nil" are what the descriptor actually saw.
        for writes in [spec.vertex, spec.fragment, spec.tile].iter().enumerate() {
            let (stage, writes) = writes;
            for write in *writes {
                let functions = match write {
                    Write::Functions(names) => {
                        let mut objects = Vec::new();
                        for name in *names {
                            objects.push(held(name)?);
                        }
                        Some(NSArray::from_retained_slice(&objects))
                    }
                    Write::Nil => None,
                };
                let list = functions.as_deref();
                // SAFETY: the array holds this library's own functions, which is what the
                // property takes; Metal copies it.
                unsafe {
                    match stage {
                        0 => descriptor.setVertexAdditionalBinaryFunctions(list),
                        1 => descriptor.setFragmentAdditionalBinaryFunctions(list),
                        _ => descriptor.setTileAdditionalBinaryFunctions(list),
                    }
                }
            }
        }
        // Each getter bound to a local, so the coverage derivation types the receiver.
        let vertex = descriptor.vertexAdditionalBinaryFunctions();
        let fragment = descriptor.fragmentAdditionalBinaryFunctions();
        let tile = descriptor.tileAdditionalBinaryFunctions();
        Ok(FunctionsDescriptorFacts {
            vertex: function_names(vertex.as_deref()),
            fragment: function_names(fragment.as_deref()),
            tile: function_names(tile.as_deref()),
        })
    }

    fn compute_pipeline_with_descriptor(
        &mut self,
        library: Handle,
        spec: &ComputeStateSpec<'_>,
    ) -> Answer<Handle> {
        let descriptor = self.compute_descriptor_from(library, spec)?;
        let state = self
            .device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|error| {
                Failure::named(
                    "NewComputePipelineState",
                    error.localizedDescription().to_string(),
                )
            })?;
        Ok(keep(&mut self.pipelines, state))
    }

    fn compute_pipeline_facts(
        &mut self,
        state: Handle,
        imageblock: [u32; 3],
    ) -> Answer<ComputeStateFacts> {
        let registry = self.device.registryID();
        // Declared, so the coverage derivation types the receiver: the pool's own element type is
        // this side's `Vec`, which names no Metal type the walk can read off a field.
        let state: &Retained<ProtocolObject<dyn MTLComputePipelineState>> = self
            .pipelines
            .get(state.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", state.0)))?;
        Ok(compute_state_facts(state, registry, imageblock))
    }

    fn render_pipeline_facts(
        &mut self,
        state: Handle,
        imageblock: [u32; 3],
    ) -> Answer<RenderStateFacts> {
        let registry = self.device.registryID();
        let state: &Retained<ProtocolObject<dyn MTLRenderPipelineState>> = &self
            .render_pipelines
            .get(state.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", state.0)))?
            .state;
        Ok(render_state_facts(state, registry, imageblock))
    }

    fn render_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &DescriptorSpec<'_>,
    ) -> Answer<RenderDescriptorFacts> {
        let held: &Retained<ProtocolObject<dyn MTLLibrary>> = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = |entry: &str| {
            held.newFunctionWithName(&NSString::from_str(entry))
                .ok_or_else(|| Failure::named("NoSuchFunction", entry.to_owned()))
        };
        let descriptor = MTLRenderPipelineDescriptor::new();
        let stages = |descriptor: &Retained<MTLRenderPipelineDescriptor>| -> Answer<()> {
            descriptor.setVertexFunction(Some(&*function(spec.vertex)?));
            if let Some(fragment) = spec.fragment {
                descriptor.setFragmentFunction(Some(&*function(fragment)?));
            }
            Ok(())
        };
        stages(&descriptor)?;
        if spec.shape.sets() {
            state_descriptor(&descriptor);
        }
        if spec.shape.resets() {
            // Metal's own reset, which is the item under test. The stage functions go back
            // afterwards: `reset` clears them too, and a descriptor whose vertex function is nil
            // answers a different shape from the one the defaults case recorded.
            descriptor.reset();
            stages(&descriptor)?;
        }
        Ok(read_descriptor(&descriptor, !spec.shape.resets()))
    }

    fn compute_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &ComputeDescriptorSpec<'_>,
    ) -> Answer<ComputeDescriptorFacts> {
        let held: &Retained<ProtocolObject<dyn MTLLibrary>> = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let kernel = || {
            held.newFunctionWithName(&NSString::from_str(spec.kernel))
                .ok_or_else(|| Failure::named("NoSuchFunction", spec.kernel.to_owned()))
        };
        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&*kernel()?));
        if spec.shape.sets() {
            state_compute_descriptor(&descriptor);
        }
        if spec.shape.resets() {
            // Metal's own reset, which is the item under test. The kernel goes back afterwards:
            // `reset` clears it too, and a descriptor whose compute function is nil answers a
            // different shape from the one the defaults case recorded.
            descriptor.reset();
            descriptor.setComputeFunction(Some(&*kernel()?));
        }
        Ok(read_compute_descriptor(&descriptor, !spec.shape.resets()))
    }

    fn tile_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &TileDescriptorSpec<'_>,
    ) -> Answer<TileDescriptorFacts> {
        let held: &Retained<ProtocolObject<dyn MTLLibrary>> = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let tile = || {
            held.newFunctionWithName(&NSString::from_str(spec.tile))
                .ok_or_else(|| Failure::named("NoSuchFunction", spec.tile.to_owned()))
        };
        let descriptor = MTLTileRenderPipelineDescriptor::new();
        // SAFETY: the function is of this library, and the descriptor is this call's own. Set
        // before anything reads the descriptor and again after the reset, because `tileFunction`
        // is non-null in `objc2-metal` and a descriptor with none would panic rather than answer.
        unsafe { descriptor.setTileFunction(&*tile()?) };
        if spec.shape.sets() {
            state_tile_descriptor(&descriptor, spec.shape.resets());
        }
        if spec.shape.resets() {
            descriptor.reset();
            // SAFETY: as above.
            unsafe { descriptor.setTileFunction(&*tile()?) };
        }
        Ok(read_tile_descriptor(&descriptor, !spec.shape.resets()))
    }

    fn mesh_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &MeshDescriptorSpec<'_>,
    ) -> Answer<MeshDescriptorFacts> {
        let held: &Retained<ProtocolObject<dyn MTLLibrary>> = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = |entry: &str| {
            held.newFunctionWithName(&NSString::from_str(entry))
                .ok_or_else(|| Failure::named("NoSuchFunction", entry.to_owned()))
        };
        let descriptor = MTLMeshRenderPipelineDescriptor::new();
        let stages = |descriptor: &Retained<MTLMeshRenderPipelineDescriptor>| -> Answer<()> {
            // SAFETY: each function is of this library, and the descriptor is this call's own.
            unsafe {
                descriptor.setMeshFunction(Some(&*function(spec.mesh)?));
                descriptor.setFragmentFunction(Some(&*function(spec.fragment)?));
                if let Some(object) = spec.object {
                    descriptor.setObjectFunction(Some(&*function(object)?));
                }
            }
            Ok(())
        };
        stages(&descriptor)?;
        if spec.shape.sets() {
            state_mesh_descriptor(&descriptor);
        }
        if spec.shape.resets() {
            descriptor.reset();
            stages(&descriptor)?;
        }
        Ok(read_mesh_descriptor(&descriptor, !spec.shape.resets()))
    }

    fn pass_descriptor_facts(&mut self, spec: &PassSpec) -> Answer<PassDescriptorFacts> {
        let descriptor: Retained<MTLRenderPassDescriptor> =
            MTLRenderPassDescriptor::renderPassDescriptor();
        if spec.shape.sets() {
            for index in 0..PASS_COLORS {
                let color: Retained<MTLRenderPassColorAttachmentDescriptor> =
                    pass_color(&descriptor, index);
                let stated = crate::pass_descriptors::stated_row(index);
                state_pass_attachment(&color, &stated);
                color.setClearColor(clear_color_of(&stated));
            }
            let depth: Retained<MTLRenderPassDepthAttachmentDescriptor> =
                descriptor.depthAttachment();
            let stated = crate::pass_descriptors::stated_row(DEPTH_AT);
            state_pass_attachment(&depth, &stated);
            depth.setClearDepth(stated.clear_depth);
            depth.setDepthResolveFilter(MTLMultisampleDepthResolveFilter(
                stated.filters.depth as usize,
            ));
            let stencil: Retained<MTLRenderPassStencilAttachmentDescriptor> =
                descriptor.stencilAttachment();
            let stated = crate::pass_descriptors::stated_row(STENCIL_AT);
            state_pass_attachment(&stencil, &stated);
            stencil.setClearStencil(stated.clear_stencil);
            stencil.setStencilResolveFilter(MTLMultisampleStencilResolveFilter(
                stated.filters.stencil as usize,
            ));
        }
        if spec.shape == Shape::Assigned {
            assign_pass_attachments(&descriptor);
        }
        if let Some(row) = spec.row {
            let shifted = |shift: u64| match spec.reads.rotates() {
                true => row.rotated(shift),
                false => row,
            };
            let color: Retained<MTLRenderPassColorAttachmentDescriptor> =
                pass_color(&descriptor, SWEPT_COLOR);
            let stated = shifted(0);
            state_pass_attachment(&color, &stated);
            color.setClearColor(clear_color_of(&stated));
            let depth: Retained<MTLRenderPassDepthAttachmentDescriptor> =
                descriptor.depthAttachment();
            let stated = shifted(1);
            state_pass_attachment(&depth, &stated);
            depth.setClearDepth(stated.clear_depth);
            depth.setDepthResolveFilter(MTLMultisampleDepthResolveFilter(
                stated.filters.depth as usize,
            ));
            let stencil: Retained<MTLRenderPassStencilAttachmentDescriptor> =
                descriptor.stencilAttachment();
            let stated = shifted(2);
            state_pass_attachment(&stencil, &stated);
            stencil.setClearStencil(stated.clear_stencil);
            stencil.setStencilResolveFilter(MTLMultisampleStencilResolveFilter(
                stated.filters.stencil as usize,
            ));
        }
        if let Some(numbers) = spec.numbers {
            descriptor.setRenderTargetArrayLength(numbers.0[0] as usize);
            descriptor.setRenderTargetWidth(numbers.0[1] as usize);
            descriptor.setRenderTargetHeight(numbers.0[2] as usize);
            descriptor.setDefaultRasterSampleCount(numbers.0[3] as usize);
            descriptor.setTileWidth(numbers.0[4] as usize);
            descriptor.setTileHeight(numbers.0[5] as usize);
            descriptor.setImageblockSampleLength(numbers.0[6] as usize);
            descriptor.setThreadgroupMemoryLength(numbers.0[7] as usize);
        }
        if let Some(stated) = spec.positions {
            let positions: Vec<metal2vulkan_runtime::SamplePosition> = stated
                .iter()
                .map(|(x, y)| metal2vulkan_runtime::SamplePosition::new(*x, *y))
                .collect();
            // The suite's one guard for the shapes Metal asserts on, which every family that sets
            // sample positions goes through. The count is held against the row's own length: this
            // descriptor has no attachment, so Metal counts the positions against nothing until an
            // encoder opens, and the coordinate rule is the one the setter itself asserts on.
            if let Some(refusal) =
                crate::render::forbidden_positions(&positions, positions.len() as u32)
            {
                return Err(refusal);
            }
            let held: Vec<MTLSamplePosition> = stated
                .iter()
                .map(|(x, y)| MTLSamplePosition { x: *x, y: *y })
                .collect();
            // SAFETY: the pointer is to `held.len()` positions that live for the call, and the
            // guard above has just held the count and every coordinate.
            unsafe {
                descriptor.setSamplePositions_count(held.as_ptr(), held.len());
            }
        }
        if spec.shape == Shape::Cleared {
            clear_pass_attachments(&descriptor);
        }
        Ok(read_pass_descriptor(&descriptor, spec.reads))
    }

    fn function_facts(
        &mut self,
        library: Handle,
        entry: &str,
        constants: &[ConstantValue],
    ) -> Answer<FunctionFacts> {
        let registry = self.device.registryID();
        let library: &Retained<ProtocolObject<dyn MTLLibrary>> = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let name = NSString::from_str(entry);
        let function: Retained<ProtocolObject<dyn MTLFunction>> = match constants.is_empty() {
            true => library
                .newFunctionWithName(&name)
                .ok_or_else(|| Failure::named("NoSuchFunction", entry.to_owned()))?,
            false => {
                let values = constant_values(constants);
                library
                    .newFunctionWithName_constantValues_error(&name, &values)
                    .map_err(|error| {
                        Failure::named(
                            "NewFunctionWithConstantValues",
                            error.localizedDescription().to_string(),
                        )
                    })?
            }
        };
        our_function_facts(&function, registry)
    }

    fn library_facts(&mut self, library: Handle) -> Answer<LibraryFacts> {
        let registry = self.device.registryID();
        let library: &Retained<ProtocolObject<dyn MTLLibrary>> = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        // Bound to a local of its declared type, as the facts walks do: the coverage derivation
        // types a receiver from what bound it, so a call made inside a `for` header names no type.
        let names: Retained<NSArray<NSString>> = library.functionNames();
        let install: Option<Retained<NSString>> = library.installName();
        Ok(LibraryFacts {
            function_names: names.iter().map(|name| name.to_string()).collect(),
            library_type: our_library_type(library.r#type().0)?,
            install_name: install.map(|name| name.to_string()),
            // The registry ID is each host's own number, so only the **comparison** is portable,
            // as it is for a function's own device.
            device_is_the_cases: library.device().registryID() == registry,
        })
    }

    fn function_facts_from_descriptor(
        &mut self,
        library: Handle,
        descriptor: &FunctionDescriptorSpec,
    ) -> Answer<FunctionFacts> {
        let registry = self.device.registryID();
        let library: &Retained<ProtocolObject<dyn MTLLibrary>> = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let made: Retained<MTLFunctionDescriptor> = MTLFunctionDescriptor::functionDescriptor();
        made.setName(Some(&NSString::from_str(&descriptor.name)));
        if !descriptor.constants.is_empty() {
            made.setConstantValues(Some(&constant_values(&descriptor.constants)));
        }
        made.setOptions(MTLFunctionOptions(descriptor.options.as_raw() as usize));
        let function: Retained<ProtocolObject<dyn MTLFunction>> = library
            .newFunctionWithDescriptor_error(&made)
            .map_err(|error| {
                Failure::named(
                    "NewFunctionWithDescriptor",
                    error.localizedDescription().to_string(),
                )
            })?;
        our_function_facts(&function, registry)
    }

    fn constant_data_offset(&mut self, encoder: Handle, index: u32) -> Answer<u64> {
        let (buffer, arguments) = self
            .argument_encoders
            .get(encoder.0 as usize)
            .ok_or_else(|| {
                Failure::named("NoSuchHandle", format!("argument encoder {}", encoder.0))
            })?
            .target
            .ok_or_else(|| {
                Failure::named("NoArgumentBuffer", "no argument buffer is set".to_owned())
            })?;
        let base = self.held(buffer)?.contents().as_ptr() as usize + arguments as usize;
        let held = self.held_encoder(encoder)?;
        // SAFETY: the index is the case's own; the call answers an address inside the argument
        // buffer's contents, and nothing is read or written through it here.
        let at = unsafe { held.constantDataAtIndex(index as usize) };
        (at.as_ptr() as usize)
            .checked_sub(base)
            .map(|offset| offset as u64)
            .ok_or_else(|| {
                Failure::named(
                    "ConstantBeforeTheArguments",
                    "constantDataAtIndex: answered an address below the arguments".to_owned(),
                )
            })
    }

    fn dispatch(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        threadgroups: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle> {
        self.encode(pipeline, bindings, &[], |encoder| {
            encoder.dispatchThreadgroups_threadsPerThreadgroup(size(threadgroups), size(threads));
        })
    }

    fn compute_pipeline_with_stage_input(
        &mut self,
        library: Handle,
        entry: &str,
        stage_input: &StageInput,
    ) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", entry.to_owned()))?;
        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&function));
        descriptor.setStageInputDescriptor(Some(&stage_input_descriptor(stage_input)));
        let pipeline = self
            .device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|error| Failure::named("NewComputePipelineState", error.to_string()))?;
        Ok(keep(&mut self.pipelines, pipeline))
    }

    fn dispatch_threads(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        stage_in: &[StageInRegion],
        grid: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle> {
        self.encode(pipeline, bindings, stage_in, |encoder| {
            encoder.dispatchThreads_threadsPerThreadgroup(size(grid), size(threads));
        })
    }

    fn wait(&mut self, command_buffer: Handle) -> Answer<Value> {
        let command_buffer = self
            .command_buffers
            .get(command_buffer.0 as usize)
            .ok_or_else(|| {
                Failure::named(
                    "NoSuchHandle",
                    format!("command buffer {}", command_buffer.0),
                )
            })?;
        command_buffer.waitUntilCompleted();
        Ok(Value::unit_variant(status_word(command_buffer.status())))
    }

    fn read_buffer(&mut self, buffer: Handle, offset: u64, length: usize) -> Answer<Vec<u8>> {
        let buffer = self.held(buffer)?;
        let past = offset
            .checked_add(length as u64)
            .is_none_or(|end| end > buffer.length() as u64);
        if past {
            return Err(Failure::named(
                "PastTheBuffer",
                format!(
                    "{length} bytes at {offset} of a buffer of {}",
                    buffer.length()
                ),
            ));
        }
        // SAFETY: `[offset, offset + length)` is inside the buffer's shared storage, and the
        // caller waited for the command buffer that wrote it.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                buffer.contents().as_ptr().cast::<u8>().add(offset as usize),
                length,
            )
        };
        Ok(bytes.to_vec())
    }

    /// A copy-and-fill program, through public Metal alone ([`crate::metal_work`]).
    ///
    /// The colour fill (`fillTextureColor:`) is Apple SPI with no arm here, so a case built on one
    /// still carries its Metal answer. Everything else a program states -- heaps, placed textures,
    /// textures over a buffer, views, every blit, `replaceRegion:`, `getBytes:`, a clearing render
    /// pass and a per-sample read-back -- is public Metal whatever pixel format the case names,
    /// including Apple's two-plane extended-range 550 and 551; the pattern fill is reached by
    /// selector after `respondsToSelector:`.
    fn work(&mut self, work: &crate::work::Work) -> Answer<crate::work::Done> {
        // A per-sample read-back's pipeline states are made here and handed to the interpreter,
        // because making one needs the scratch directory, `xcrun metal-as`, `xcrun metallib` and
        // the library pool this side owns, none of which the interpreter sees.
        let kernels = self.read_back_pipelines(work)?;
        crate::metal_work::run(&self.device, &self.queue, work, &kernels)
    }

    fn announced(&mut self) -> Vec<String> {
        Vec::new()
    }

    fn mesh_pipeline_for_commands(
        &mut self,
        library: Handle,
        functions: &MeshFunctions<'_>,
    ) -> Answer<Handle> {
        let descriptor = self.mesh_descriptor(library, functions)?;
        descriptor.setSupportIndirectCommandBuffers(true);
        self.made_mesh_pipeline(&descriptor)
    }

    fn mesh_command_buffer(&mut self, commands: &[MeshCommand]) -> Answer<Handle> {
        let descriptor = MTLIndirectCommandBufferDescriptor::new();
        descriptor.setCommandTypes(
            MTLIndirectCommandType::DrawMeshThreadgroups | MTLIndirectCommandType::DrawMeshThreads,
        );
        descriptor.setInheritPipelineState(false);
        descriptor.setInheritBuffers(false);
        // SAFETY: one object and one mesh binding, which is what every command of a case binds.
        unsafe {
            descriptor.setMaxObjectBufferBindCount(1);
            descriptor.setMaxMeshBufferBindCount(1);
        }
        // SAFETY: the count is the commands the case encodes, and every command below is inside
        // it.
        let made = unsafe {
            self.device
                .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                    &descriptor,
                    commands.len().max(1),
                    MTLResourceOptions::StorageModeShared,
                )
        }
        .ok_or_else(|| {
            Failure::named(
                "NewIndirectCommandBuffer",
                format!("{} commands returned nil", commands.len()),
            )
        })?;
        for (at, command) in commands.iter().enumerate() {
            // SAFETY: `at` is below the buffer's command count, just set to `commands.len()`.
            // The type is written out so the receiver of the setters below is a typed name: the
            // coverage derivation types a local from what made it, and the chain that makes this
            // one passes through an `MTLResourceOptions` constant it cannot read past, so without
            // the annotation every setter here is attributed to no item at all.
            let encoded: Retained<ProtocolObject<dyn MTLIndirectRenderCommand>> =
                unsafe { made.indirectRenderCommandAtIndex(at) };
            let state = &self
                .render_pipelines
                .get(command.pipeline.0 as usize)
                .ok_or_else(|| {
                    Failure::named(
                        "NoSuchHandle",
                        format!("mesh pipeline {}", command.pipeline.0),
                    )
                })?
                .state;
            encoded.setRenderPipelineState(state);
            for (buffer, index) in &command.object_buffers {
                let buffer = self.held(*buffer)?;
                // SAFETY: the buffer is at least as long as the function's binding, and the
                // command outlives the call.
                unsafe { encoded.setObjectBuffer_offset_atIndex(buffer, 0, *index as usize) };
            }
            for (buffer, index) in &command.mesh_buffers {
                let buffer = self.held(*buffer)?;
                // SAFETY: as the object buffer's.
                unsafe { encoded.setMeshBuffer_offset_atIndex(buffer, 0, *index as usize) };
            }
            let size = |axes: [u32; 3]| MTLSize {
                width: axes[0] as usize,
                height: axes[1] as usize,
                depth: axes[2] as usize,
            };
            match command.draw {
                CommandDraw::Threadgroups {
                    groups,
                    object,
                    mesh,
                } => encoded
                    .drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                        size(groups),
                        size(object),
                        size(mesh),
                    ),
                CommandDraw::Threads {
                    threads,
                    object,
                    mesh,
                } => encoded.drawMeshThreads_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                    size(threads),
                    size(object),
                    size(mesh),
                ),
            }
        }
        Ok(keep(&mut self.command_buffers_indirect, made))
    }

    fn compute_pipeline_for_commands(
        &mut self,
        library: Handle,
        entry: &str,
        stage_input: Option<&StageInput>,
    ) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", entry.to_owned()))?;
        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&function));
        descriptor.setSupportIndirectCommandBuffers(true);
        if let Some(stage_input) = stage_input {
            descriptor.setStageInputDescriptor(Some(&stage_input_descriptor(stage_input)));
        }
        let pipeline = self
            .device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|error| Failure::named("NewComputePipelineState", error.to_string()))?;
        Ok(keep(&mut self.pipelines, pipeline))
    }

    fn indirect_command_buffer(
        &mut self,
        spec: &IcbDescriptor,
        commands: &[EncodedCommand],
    ) -> Answer<CommandsMade> {
        let descriptor = MTLIndirectCommandBufferDescriptor::new();
        descriptor.setCommandTypes(MTLIndirectCommandType::from_bits_retain(
            spec.command_types.as_raw() as usize,
        ));
        descriptor.setInheritPipelineState(spec.inherit_pipeline_state);
        descriptor.setInheritBuffers(spec.inherit_buffers);
        descriptor.setMaxVertexBufferBindCount(spec.max_vertex_buffer_bind_count as usize);
        descriptor.setMaxFragmentBufferBindCount(spec.max_fragment_buffer_bind_count as usize);
        descriptor.setMaxKernelBufferBindCount(spec.max_kernel_buffer_bind_count as usize);
        descriptor.setSupportDynamicAttributeStride(spec.support_dynamic_attribute_stride);
        descriptor.setSupportColorAttachmentMapping(spec.support_color_attachment_mapping);
        descriptor.setInheritDepthStencilState(spec.inherit_depth_stencil_state);
        descriptor.setInheritDepthBias(spec.inherit_depth_bias);
        descriptor.setInheritDepthClipMode(spec.inherit_depth_clip_mode);
        descriptor.setInheritCullMode(spec.inherit_cull_mode);
        descriptor.setInheritFrontFacingWinding(spec.inherit_front_facing_winding);
        descriptor.setInheritTriangleFillMode(spec.inherit_triangle_fill_mode);
        // SAFETY: each count is the bindings the case's own commands set, at or below Metal's 31,
        // and a count of 0 is Metal's own default for the three object and mesh properties.
        unsafe {
            descriptor.setMaxKernelThreadgroupMemoryBindCount(
                spec.max_kernel_threadgroup_memory_bind_count as usize,
            );
            descriptor.setMaxObjectBufferBindCount(spec.max_object_buffer_bind_count as usize);
            descriptor.setMaxMeshBufferBindCount(spec.max_mesh_buffer_bind_count as usize);
            descriptor.setMaxObjectThreadgroupMemoryBindCount(
                spec.max_object_threadgroup_memory_bind_count as usize,
            );
        }
        let count = commands.len().max(1);
        // SAFETY: the count is the commands the case encodes, and every command below is inside
        // it.
        let made = unsafe {
            self.device
                .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                    &descriptor,
                    count,
                    MTLResourceOptions::StorageModeShared,
                )
        }
        .ok_or_else(|| {
            Failure::named(
                "NewIndirectCommandBuffer",
                format!("{count} commands returned nil"),
            )
        })?;
        let mut refused = Vec::with_capacity(commands.len());
        for (at, command) in commands.iter().enumerate() {
            match command {
                EncodedCommand::Compute(steps) => {
                    // SAFETY: `at` is below the buffer's command count, just set to `count`.
                    let encoded: Retained<ProtocolObject<dyn MTLIndirectComputeCommand>> =
                        unsafe { made.indirectComputeCommandAtIndex(at) };
                    for step in steps {
                        self.encode_compute_command(&encoded, step)?;
                    }
                }
                EncodedCommand::Render(steps) => {
                    // SAFETY: as the compute command's.
                    let encoded: Retained<ProtocolObject<dyn MTLIndirectRenderCommand>> =
                        unsafe { made.indirectRenderCommandAtIndex(at) };
                    for step in steps {
                        self.encode_render_command(&encoded, step)?;
                    }
                }
            }
            // Metal's setters answer nothing, so no step of this side can refuse.
            refused.push(vec![None; command.steps()]);
        }
        Ok(CommandsMade {
            commands: keep(&mut self.command_buffers_indirect, made),
            refused,
        })
    }

    fn indirect_commands_size(&mut self, commands: Handle) -> Answer<u64> {
        let icb = self.held_commands(commands)?;
        Ok(icb.size() as u64)
    }

    fn reset_indirect_commands(&mut self, commands: Handle, first: u64, count: u64) -> Answer<()> {
        let icb = self.held_commands(commands)?;
        let range = NSRange::new(first as usize, count as usize);
        // SAFETY: the range is inside the buffer's own command count, which the case states and
        // `indirect_commands_size` reads back beside it.
        unsafe { icb.resetWithRange(range) };
        Ok(())
    }

    fn execute_mesh_commands(
        &mut self,
        targets: &[Handle],
        commands: Handle,
        used: &[Handle],
    ) -> Answer<PassOutcome> {
        let icb = self
            .command_buffers_indirect
            .get(commands.0 as usize)
            .ok_or_else(|| {
                Failure::named("NoSuchHandle", format!("command buffer {}", commands.0))
            })?
            .clone();
        let resident: Vec<&ProtocolObject<dyn MTLResource>> = used
            .iter()
            .map(|handle| self.held(*handle).map(ProtocolObject::from_ref))
            .collect::<Answer<Vec<_>>>()?;
        let (command_buffer, encoder) = self.pass(targets)?;
        let stages = MTLRenderStages::Object | MTLRenderStages::Mesh | MTLRenderStages::Fragment;
        for resource in &resident {
            encoder.useResource_usage_stages(
                resource,
                MTLResourceUsage::Read | MTLResourceUsage::Write,
                stages,
            );
        }
        // SAFETY: the range is the buffer's own command count.
        unsafe {
            encoder.executeCommandsInBuffer_withRange(
                &icb,
                objc2_foundation::NSRange::new(0, icb.size()),
            );
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        // Metal's calls answer nothing, so no step of this side can refuse: one `None` per
        // resident resource, one for the execution and one for the commit, as the runtime side
        // answers them.
        let refused = vec![None; resident.len() + 2];
        let error = command_buffer
            .error()
            .map(|error| Value::unit_variant(format!("code {}", error.code())));
        let status = Value::unit_variant(status_word(command_buffer.status()));
        keep(&mut self.command_buffers, command_buffer);
        Ok(PassOutcome {
            refused,
            status,
            error,
        })
    }

    fn fragment_fill(&mut self, fill: &FragmentFill<'_>) -> Answer<FragmentFilled> {
        let format = MTLPixelFormat(fill.format.as_raw() as usize);
        let (vertex, fragment) = {
            let library = self.libraries.get(fill.library.0 as usize).ok_or_else(|| {
                Failure::named("NoSuchHandle", format!("library {}", fill.library.0))
            })?;
            let function = |name: &str| {
                library
                    .newFunctionWithName(&NSString::from_str(name))
                    .ok_or_else(|| Failure::named("NoSuchFunction", name.to_owned()))
            };
            (function(fill.vertex)?, function(fill.fragment)?)
        };
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&vertex));
        descriptor.setFragmentFunction(Some(&fragment));
        // SAFETY: attachment 0 is inside the eight every Metal pipeline descriptor holds.
        let attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
        attachment.setPixelFormat(format);
        let state = self
            .device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(|error| Failure::named("NewRenderPipelineState", error.to_string()))?;
        // SAFETY: the descriptor constructor reads only the format and the size it is given.
        let target_descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format,
                fill.size[0] as usize,
                fill.size[1] as usize,
                false,
            )
        };
        target_descriptor.setUsage(MTLTextureUsage::RenderTarget);
        target_descriptor.setStorageMode(MTLStorageMode::Private);
        let target = self
            .device
            .newTextureWithDescriptor(&target_descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!(
                        "{:?} {}x{} render target returned nil",
                        fill.format, fill.size[0], fill.size[1]
                    ),
                )
            })?;
        let bound: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = fill
            .buffers
            .iter()
            .map(|handle| self.held(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        // SAFETY: as the pipeline's attachment 0.
        let color = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        color.setTexture(Some(&target));
        color.setLoadAction(MTLLoadAction::DontCare);
        color.setStoreAction(MTLStoreAction::Store);
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or_else(|| Failure::named("RenderCommandEncoder", "returned nil".to_owned()))?;
        encoder.setRenderPipelineState(&state);
        for (index, buffer) in bound.iter().enumerate() {
            // SAFETY: the buffer is at least as long as the function's binding, and the encoder
            // outlives the call.
            unsafe { encoder.setFragmentBuffer_offset_atIndex(Some(buffer), 0, index) };
        }
        // SAFETY: the pipeline is set, the vertex function reads no buffer, and the count is the
        // case's own small one.
        unsafe {
            encoder.drawPrimitives_vertexStart_vertexCount(
                MTLPrimitiveType::Triangle,
                0,
                fill.vertices as usize,
            );
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let status = Value::unit_variant(status_word(command_buffer.status()));
        keep(&mut self.command_buffers, command_buffer);
        let bytes = self.blit_out(&target, fill.size, fill.row_pitch)?;
        keep(&mut self.textures, target);
        Ok(FragmentFilled { status, bytes })
    }

    fn mesh_pipeline(&mut self, library: Handle, functions: &MeshFunctions<'_>) -> Answer<Handle> {
        let descriptor = self.mesh_descriptor(library, functions)?;
        self.made_mesh_pipeline(&descriptor)
    }

    fn mesh_pipeline_bindings(
        &mut self,
        library: Handle,
        spec: &MeshReflectionSpec<'_>,
        options: u64,
    ) -> Answer<StageReflection> {
        let descriptor = self.mesh_reflection_descriptor(library, spec)?;
        let mut reflection = None;
        // The options are the named bits, for the reason `tile_pipeline_bindings` states.
        let made = self
            .device
            .newRenderPipelineStateWithMeshDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption(options as NSUInteger),
                Some(&mut reflection),
            );
        let Ok(state) = made else {
            return Ok(StageReflection::refused());
        };
        // Declared, not inferred, for the reason `tile_pipeline_bindings` states.
        let found = match reflection {
            Some(reflection) => {
                let reflection: Retained<MTLRenderPipelineReflection> = reflection;
                StageReflection::reflected(metal_stage_bindings(&reflection)?)
            }
            None => StageReflection::unreflected(),
        };
        // The state is kept so it outlives the reflection its creation answered.
        keep(
            &mut self.render_pipelines,
            RenderPipeline::unreflected(state),
        );
        Ok(found)
    }

    fn render_target(&mut self, width: u32, height: u32) -> Answer<Handle> {
        // SAFETY: the extent is a case's own small one, and the descriptor is this call's own.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                TARGET_PIXEL_FORMAT,
                width as usize,
                height as usize,
                false,
            )
        };
        descriptor.setUsage(MTLTextureUsage::RenderTarget);
        descriptor.setStorageMode(MTLStorageMode::Shared);
        let texture = self
            .device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!("{width}x{height} returned nil"),
                )
            })?;
        Ok(keep(&mut self.textures, texture))
    }

    fn mesh_pass(&mut self, targets: &[Handle], steps: &[MeshStep]) -> Answer<PassOutcome> {
        let (command_buffer, encoder) = self.pass(targets)?;
        // Metal's encoder methods answer nothing, so no step of this side can refuse. A refusal
        // of ours then compares against Metal's answer and reads as a refusal.
        let refused = vec![None; steps.len()];
        for step in steps {
            self.encode_step(&encoder, step)?;
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let error = command_buffer
            .error()
            .map(|error| Value::unit_variant(format!("code {}", error.code())));
        let status = Value::unit_variant(status_word(command_buffer.status()));
        keep(&mut self.command_buffers, command_buffer);
        Ok(PassOutcome {
            refused,
            status,
            error,
        })
    }

    fn rate_map(&mut self, request: &RateMapRequest<'_>) -> Answer<Handle> {
        if !request.form.takes(request.layers.len()) {
            return Err(Failure::named(
                "FormTakesAnotherLayerCount",
                format!(
                    "{:?} cannot build a descriptor of {} layers",
                    request.form,
                    request.layers.len()
                ),
            ));
        }
        let screen = MTLSize {
            width: request.screen[0] as usize,
            height: request.screen[1] as usize,
            depth: 0,
        };
        let layers = request
            .layers
            .iter()
            .map(layer_descriptor)
            .collect::<Answer<Vec<_>>>()?;
        let descriptor = match request.form {
            // The two class methods that take their layers with them. Each is one call, so there
            // is no intermediate state for a setter to leave wrong, which is what makes them worth
            // comparing against the setter forms at all.
            RateMapForm::ScreenSizeAndLayer => {
                let layer = layers
                    .first()
                    .ok_or_else(|| Failure::named("NoLayer", "the one-layer form has no layer"))?;
                // SAFETY: the screen size is inside Metal's own validation bound, which
                // `rate_map_space` holds every shape to.
                unsafe {
                    MTLRasterizationRateMapDescriptor::
                        rasterizationRateMapDescriptorWithScreenSize_layer(screen, layer)
                }
            }
            RateMapForm::ScreenSizeLayerCountLayers => {
                // `objc2-metal` 0.3.2 binds neither this constructor nor
                // `MTLRasterizationRateLayerDescriptor`'s `sampleCount` getter, though the SDK
                // declares both, so the selector goes by hand. Apple's signature is
                // `(MTLSize, NSUInteger, MTLRasterizationRateLayerDescriptor * const *)`, and the
                // array must hold `layerCount` non-null pointers.
                let pointers: Vec<*const MTLRasterizationRateLayerDescriptor> =
                    layers.iter().map(|layer| &**layer as *const _).collect();
                let class = MTLRasterizationRateMapDescriptor::class();
                // SAFETY: the selector is the SDK's, the receiver is its class, and `pointers`
                // holds exactly `layers.len()` non-null pointers to live layer descriptors that
                // outlive the call.
                let made: Option<Retained<MTLRasterizationRateMapDescriptor>> = unsafe {
                    objc2::msg_send![
                        class,
                        rasterizationRateMapDescriptorWithScreenSize: screen,
                        layerCount: pointers.len(),
                        layers: pointers.as_ptr(),
                    ]
                };
                made.ok_or_else(|| {
                    Failure::named("RateMapDescriptorWithLayers", "returned nil".to_owned())
                })?
            }
            // The three forms that start from an empty descriptor and set their layers into it.
            _ => {
                // SAFETY: as above, the screen size is inside Metal's validation bound.
                let descriptor = unsafe {
                    MTLRasterizationRateMapDescriptor::rasterizationRateMapDescriptorWithScreenSize(
                        screen,
                    )
                };
                let holed = request.form == RateMapForm::Holed;
                for (at, layer) in layers.iter().enumerate() {
                    // The holed form sets every layer one index up, leaving index 0 nil: the shape
                    // whose `layerCount` Metal answers 0 for.
                    let index = if holed { at + 1 } else { at };
                    match request.form {
                        // Apple's header calls `layers[i] = layer` identical to
                        // `setLayer:atIndex:`; comparing the compiled map of each is what holds it
                        // to that.
                        RateMapForm::LayersSubscript => {
                            let array = descriptor.layers();
                            // SAFETY: the index is inside the array the descriptor grows on
                            // demand.
                            unsafe { array.setObject_atIndexedSubscript(Some(layer), index) };
                        }
                        // SAFETY: as above.
                        _ => unsafe { descriptor.setLayer_atIndex(Some(layer), index) },
                    }
                }
                descriptor
            }
        };
        if let Some(label) = request.label {
            descriptor.setLabel(Some(&NSString::from_str(label)));
        }
        // What the form actually built, read off the descriptor itself. This is a check on the
        // leg, not a compared site: a form that reached another state than the case asked for
        // would compile another map, and the comparison would then read as a difference of
        // Metal's. `layerCount` counts the layers from index 0 contiguously, so the holed form's
        // own answer is 0 with a layer still in it.
        let built = descriptor.layerCount();
        let wanted = match request.form {
            RateMapForm::Holed => 0,
            _ => request.layers.len(),
        };
        if built != wanted {
            return Err(Failure::named(
                "FormBuiltAnotherDescriptor",
                format!(
                    "{:?} built a descriptor of {built} layers, not {wanted}",
                    request.form
                ),
            ));
        }
        let map = self
            .device
            .newRasterizationRateMapWithDescriptor(&descriptor)
            .ok_or_else(|| Failure::named("NewRasterizationRateMap", "returned nil".to_owned()))?;
        Ok(keep(&mut self.rate_maps, map))
    }

    fn rate_map_coordinates(
        &mut self,
        map: Handle,
        probes: &[CoordinateProbe],
    ) -> Answer<Vec<MappedCoordinates>> {
        let map = self.held_rate_map(map)?;
        let mut out = Vec::with_capacity(probes.len());
        for probe in probes {
            // SAFETY: neither call is bounds-checked on the layer index, and a case asks only
            // inside the map's own `layerCount`. The two shapes where the screen-to-physical call
            // ends Metal's process — a map of no layers and a map of a zero screen extent — are
            // the case's to keep away from, and `rate_map_space` holds every swept shape to that.
            let physical = unsafe {
                map.mapScreenToPhysicalCoordinates_forLayer(
                    // `MTLCoordinate2D` is `MTLSamplePosition` in `objc2-metal`, one struct of two
                    // `f32` under two SDK names.
                    MTLSamplePosition {
                        x: probe.screen[0],
                        y: probe.screen[1],
                    },
                    probe.layer as usize,
                )
            };
            // SAFETY: as above.
            let back = unsafe {
                map.mapPhysicalToScreenCoordinates_forLayer(
                    MTLSamplePosition {
                        x: probe.physical[0],
                        y: probe.physical[1],
                    },
                    probe.layer as usize,
                )
            };
            out.push(MappedCoordinates {
                screen_to_physical: [physical.x.to_bits(), physical.y.to_bits()],
                physical_to_screen: [back.x.to_bits(), back.y.to_bits()],
            });
        }
        Ok(out)
    }

    fn rate_map_device_is_the_sides(&mut self, map: Handle) -> Answer<bool> {
        let map = self.held_rate_map(map)?;
        let device = map.device();
        Ok(std::ptr::eq(&*device, &*self.device))
    }

    fn rate_map_facts(&mut self, map: Handle, layers: &[u64]) -> Answer<RateMapFacts> {
        let map = self.held_rate_map(map)?;
        let size = |size: MTLSize| [size.width as u64, size.height as u64, size.depth as u64];
        let sized = map.parameterBufferSizeAndAlign();
        Ok(RateMapFacts {
            label: map.label().map(|label| label.to_string()),
            screen_size: size(map.screenSize()),
            physical_granularity: size(map.physicalGranularity()),
            layer_count: map.layerCount() as u64,
            // SAFETY: `physicalSizeForLayer:` is not bounds-checked, and the case asks only about
            // layer indices Metal's own answers are measured at.
            physical_sizes: layers
                .iter()
                .map(|layer| size(unsafe { map.physicalSizeForLayer(*layer as usize) }))
                .collect(),
            parameter_size: sized.size as u64,
            parameter_align: sized.align as u64,
        })
    }

    fn copy_rate_map_parameters(&mut self, map: Handle, into: Handle, offset: u64) -> Answer<()> {
        let buffer: Retained<ProtocolObject<dyn MTLBuffer>> = Retained::from(self.held(into)?);
        let map = self.held_rate_map(map)?;
        if u64::try_from(buffer.length()).unwrap_or(u64::MAX)
            < offset + map.parameterBufferSizeAndAlign().size as u64
        {
            // Metal writes past the buffer's length here, so the case never asks it to: a short
            // buffer is refused before the copy rather than measured.
            return Err(Failure::named(
                "PastTheBuffer",
                format!(
                    "{} parameter bytes at {offset} of a buffer of {}",
                    map.parameterBufferSizeAndAlign().size,
                    buffer.length()
                ),
            ));
        }
        // SAFETY: the buffer is of shared storage, is long enough for the parameter data at the
        // offset, is kept alive by `self.buffers`, and no command uses it.
        unsafe { map.copyParameterDataToBuffer_offset(&buffer, offset as usize) };
        Ok(())
    }

    fn read_texture_over(
        &mut self,
        texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
        before: &[u8],
    ) -> Answer<Vec<u8>> {
        let wanted = row_pitch as usize * size[1] as usize;
        if before.len() != wanted {
            return Err(Failure::named(
                "TooFewBytes",
                format!(
                    "{} bytes before the read, and it reads {wanted}",
                    before.len()
                ),
            ));
        }
        let texture = self.held_texture(texture)?;
        let mut bytes = before.to_vec();
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: size[0] as usize,
                height: size[1] as usize,
                depth: 1,
            },
        };
        // SAFETY: `bytes` is `size[1]` rows of `row_pitch` bytes, checked just above, which is the
        // region the call reads, and no command buffer of this case is in flight.
        unsafe {
            texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                std::ptr::NonNull::from(bytes.as_mut_slice()).cast(),
                row_pitch as usize,
                region,
                0,
            );
        }
        Ok(bytes)
    }

    fn render_pipeline_with_reflection(
        &mut self,
        library: Handle,
        vertex: &str,
        fragment: &str,
        attachment: PixelFormat,
    ) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = |name: &str| -> Answer<Retained<ProtocolObject<dyn MTLFunction>>> {
            library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| Failure::named("NewFunctionWithName", name.to_owned()))
        };
        let descriptor = MTLRenderPipelineDescriptor::new();
        let (vertex, fragment) = (function(vertex)?, function(fragment)?);
        descriptor.setVertexFunction(Some(&vertex));
        descriptor.setFragmentFunction(Some(&fragment));
        let attachments = descriptor.colorAttachments();
        // SAFETY: attachment 0 is inside the fixed-size array every descriptor holds.
        let first = unsafe { attachments.objectAtIndexedSubscript(0) };
        first.setPixelFormat(MTLPixelFormat(attachment.as_raw() as usize));
        let mut reflection = None;
        let state = self
            .device
            .newRenderPipelineStateWithDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::BindingInfo,
                Some(&mut reflection),
            )
            .map_err(|error| {
                Failure::named(
                    "NewRenderPipelineState",
                    error.localizedDescription().to_string(),
                )
            })?;
        let reflection = reflection.ok_or_else(|| {
            Failure::named(
                "NoReflection",
                "the creation answered no reflection".to_owned(),
            )
        })?;
        Ok(keep(
            &mut self.render_pipelines,
            RenderPipeline {
                state,
                reflection: Some(reflection),
            },
        ))
    }

    fn texture_accesses(
        &mut self,
        pipeline: Handle,
        stage: RenderStages,
        indices: &[u64],
    ) -> Answer<Vec<Option<BindingAccess>>> {
        let reflection = self
            .render_pipelines
            .get(pipeline.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", pipeline.0)))?
            .reflection
            .as_ref()
            .ok_or_else(|| {
                // Not a Metal gap and not a venue's: the case asked a state whose creation never
                // asked Metal for a reflection, and Metal cannot be asked afterwards.
                Failure::named(
                    "NoReflection",
                    format!("pipeline {} was not created with one", pipeline.0),
                )
            })?;
        // One stage, because that is how Metal states it: a binding list a stage. A union has no
        // Metal answer, only a composition rule of ours.
        let bindings = match stage.as_raw() {
            raw if raw == RenderStages::VERTEX.as_raw() => reflection.vertexBindings(),
            raw if raw == RenderStages::FRAGMENT.as_raw() => reflection.fragmentBindings(),
            raw if raw == RenderStages::TILE.as_raw() => reflection.tileBindings(),
            raw if raw == RenderStages::OBJECT.as_raw() => reflection.objectBindings(),
            raw if raw == RenderStages::MESH.as_raw() => reflection.meshBindings(),
            raw => {
                return Err(Failure::unavailable(&format!(
                    "texture_accesses of MTLRenderStages({raw:#x}), which Metal's reflection \
                     states no binding list for"
                )))
            }
        };
        // Each texture binding covers `arrayLength` slots from its own index, and an argument the
        // function never references is listed with `isUsed` false.
        let mut covered: Vec<(u64, u64, BindingAccess, bool)> = Vec::new();
        for binding in &bindings {
            if binding.r#type() != MTLBindingType::Texture {
                continue;
            }
            let access = match binding.access() {
                MTLBindingAccess::ReadOnly => BindingAccess::ReadOnly,
                MTLBindingAccess::ReadWrite => BindingAccess::ReadWrite,
                MTLBindingAccess::WriteOnly => BindingAccess::WriteOnly,
                other => {
                    return Err(Failure::named(
                        "UnknownBindingAccess",
                        format!("MTLBindingAccess({})", other.0),
                    ))
                }
            };
            // SAFETY: `MTLTextureBinding` extends `MTLBinding` with `arrayLength`, and the
            // binding answered `MTLBindingTypeTexture` just above, so it conforms to it. objc2
            // offers no safe cast between two protocol objects, so the selector is sent directly.
            let length: usize = unsafe { objc2::msg_send![&*binding, arrayLength] };
            covered.push((
                binding.index() as u64,
                length as u64,
                access,
                binding.isUsed(),
            ));
        }
        Ok(indices
            .iter()
            .map(|index| {
                covered
                    .iter()
                    .find(|(first, length, _, _)| *index >= *first && *index < *first + *length)
                    .and_then(|(_, _, access, used)| used.then_some(*access))
            })
            .collect())
    }

    fn tile_pipeline(&mut self, library: Handle, spec: &TilePipelineSpec<'_>) -> Answer<Handle> {
        let descriptor = self.tile_descriptor_from(library, spec)?;
        let state = self
            .device
            .newRenderPipelineStateWithTileDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|error| Failure::named("NewTileRenderPipelineState", error.to_string()))?;
        Ok(keep(
            &mut self.render_pipelines,
            RenderPipeline::unreflected(state),
        ))
    }

    fn tile_pipeline_bindings(
        &mut self,
        library: Handle,
        spec: &TilePipelineSpec<'_>,
        options: u64,
    ) -> Answer<StageReflection> {
        let descriptor = self.tile_descriptor_from(library, spec)?;
        let mut reflection = None;
        // The options are bits Apple's header names
        // ([`crate::pipeline_option_reflection::MASKS`]), so nothing undefined reaches Metal here.
        let made = self
            .device
            .newRenderPipelineStateWithTileDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption(options as NSUInteger),
                Some(&mut reflection),
            );
        let Ok(state) = made else {
            return Ok(StageReflection::refused());
        };
        // Declared, not inferred: the coverage derivation types a receiver from what bound it, and
        // an `Option` the creation filled in through an out-parameter names no type at all.
        let found = match reflection {
            Some(reflection) => {
                let reflection: Retained<MTLRenderPipelineReflection> = reflection;
                StageReflection::reflected(metal_stage_bindings(&reflection)?)
            }
            None => StageReflection::unreflected(),
        };
        // The state is kept so it outlives the reflection its creation answered, as the compute
        // half's is.
        keep(
            &mut self.render_pipelines,
            RenderPipeline::unreflected(state),
        );
        Ok(found)
    }

    fn pipeline_both_paths(
        &mut self,
        library: Handle,
        form: &PipelineForm<'_>,
    ) -> Answer<BothPaths> {
        self.both_paths(library, form)
    }

    fn tile_facts(&mut self, state: Handle) -> Answer<TileFacts> {
        // Declared, so the coverage derivation types the receiver: the pool's element is this
        // side's own struct, which names no Metal type the walk can read off a field, and these
        // three getters were therefore attributed to no item while being compared exactly.
        let state: &Retained<ProtocolObject<dyn MTLRenderPipelineState>> = &self
            .render_pipelines
            .get(state.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("tile pipeline {}", state.0)))?
            .state;
        Ok(tile_state_facts(state))
    }

    fn tile_pass(
        &mut self,
        targets: &[Handle],
        tile: [u32; 2],
        steps: &[TileStep],
    ) -> Answer<PassOutcome> {
        let (command_buffer, encoder) = self.loading_pass(targets, tile)?;
        // Metal's encoder methods answer nothing, so no step of this side can refuse.
        let refused = vec![None; steps.len()];
        for step in steps {
            self.encode_tile_step(&encoder, step)?;
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let error = command_buffer
            .error()
            .map(|error| Value::unit_variant(format!("code {}", error.code())));
        let status = Value::unit_variant(status_word(command_buffer.status()));
        keep(&mut self.command_buffers, command_buffer);
        Ok(PassOutcome {
            refused,
            status,
            error,
        })
    }

    fn compute_pass(&mut self, pass: &ComputePass<'_>) -> Answer<ComputePassOutcome> {
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        // The type is written out rather than inferred from the `match`, so that the coverage
        // derivation can resolve the receiver of the getter below: a call whose receiver does not
        // type is attributed to no item at all, which is how `dispatchType` came to read
        // unexercised while this very line read it.
        let encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
            match pass.dispatching {
                Dispatching::Serial => command_buffer.computeCommandEncoder(),
                Dispatching::Concurrent => command_buffer
                    .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent),
            }
            .ok_or_else(|| Failure::named("ComputeCommandEncoder", pass.dispatching.call()))?;
        // Read into its own binding, rather than as the scrutinee of the `match`: the coverage
        // derivation reads a call's receiver as the expression before the dot, and `match encoder`
        // is not a receiver it can type — which is how this very line left `dispatchType` reading
        // unexercised.
        let answered = encoder.dispatchType();
        let dispatch_type = Value::unit_variant(match answered {
            MTLDispatchType::Concurrent => "Concurrent",
            _ => "Serial",
        });
        // Metal's encoder methods answer nothing, so no step of this side can refuse. A refusal of
        // ours then compares against Metal's answer at that step and reads as a refusal.
        let refused = vec![None; pass.steps.len()];
        for step in pass.steps {
            self.encode_compute_step(&encoder, step)?;
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let error = command_buffer
            .error()
            .map(|error| Value::unit_variant(format!("code {}", error.code())));
        let status = Value::unit_variant(status_word(command_buffer.status()));
        keep(&mut self.command_buffers, command_buffer);
        Ok(ComputePassOutcome {
            refused,
            dispatch_type,
            status,
            error,
        })
    }

    fn target(&mut self, spec: &TargetSpec) -> Answer<Handle> {
        if let Some(plane) = &spec.surface {
            return self.surface_target(spec, plane);
        }
        if let Some(over) = &spec.over_buffer {
            return self.buffer_backed_target(spec, over);
        }
        // SAFETY: the descriptor constructor reads only the format and the size it is given.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat(spec.format.as_raw() as usize),
                spec.size[0] as usize,
                spec.size[1] as usize,
                false,
            )
        };
        let mut usage = MTLTextureUsage::RenderTarget;
        if spec.shader_read {
            usage |= MTLTextureUsage::ShaderRead;
        }
        if spec.shader_write {
            usage |= MTLTextureUsage::ShaderWrite;
        }
        descriptor.setUsage(usage);
        descriptor.setStorageMode(match spec.shared {
            true => MTLStorageMode::Shared,
            false => MTLStorageMode::Private,
        });
        // The shape before the samples, so a multisampled array target keeps the array type the
        // shape states rather than the plain multisample type the arm below would give it.
        if let Some(shape) = &spec.shape {
            descriptor.setTextureType(MTLTextureType(shape.texture_type.as_raw() as usize));
            // SAFETY: the three counts are the case's own small numbers, each at least 1, and the
            // descriptor is this call's own. Metal answers nil for a shape it will not make,
            // which the caller reports.
            unsafe {
                descriptor.setMipmapLevelCount(shape.levels as usize);
                descriptor.setArrayLength(shape.array_length as usize);
                descriptor.setDepth(shape.depth as usize);
            }
        }
        if spec.samples > 1 {
            // The plain multisample type only where the shape named no type of its own: a
            // multisampled array target keeps the array type its shape states. The **count** is
            // set either way, since a descriptor of a multisample type at one sample fails
            // `-[MTLTextureDescriptorInternal validateWithDevice:]` and ends Metal's process.
            if spec.shape.is_none() {
                descriptor.setTextureType(MTLTextureType::Type2DMultisample);
            }
            // SAFETY: the count is 2 or 4, which the case's own guard keeps it to, and Metal takes
            // both on every device; the descriptor is this call's own.
            unsafe { descriptor.setSampleCount(spec.samples as usize) };
        }
        let texture = self
            .device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!(
                        "a {:?} {}x{} target at {} samples of {:?} returned nil",
                        spec.format, spec.size[0], spec.size[1], spec.samples, spec.shape
                    ),
                )
            })?;
        Ok(keep(&mut self.textures, texture))
    }

    fn render_pipeline(
        &mut self,
        library: Handle,
        spec: &RenderPipelineSpec<'_>,
    ) -> Answer<Handle> {
        let descriptor = self.render_descriptor_from(library, spec)?;
        let state = self
            .device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(|error| Failure::named("NewRenderPipelineState", error.to_string()))?;
        Ok(keep(
            &mut self.render_pipelines,
            RenderPipeline::unreflected(state),
        ))
    }

    fn depth_stencil_state(&mut self, spec: &DepthStencilDescriptor) -> Answer<Handle> {
        let descriptor = MTLDepthStencilDescriptor::new();
        descriptor.setDepthCompareFunction(compare_function(spec.depth_compare_function));
        descriptor.setDepthWriteEnabled(spec.depth_write_enabled);
        descriptor.setFrontFaceStencil(Some(&stencil_descriptor(&spec.front_face_stencil)));
        descriptor.setBackFaceStencil(Some(&stencil_descriptor(&spec.back_face_stencil)));
        let state = self
            .device
            .newDepthStencilStateWithDescriptor(&descriptor)
            .ok_or_else(|| Failure::named("NewDepthStencilState", "returned nil".to_owned()))?;
        Ok(keep(&mut self.depth_stencil_states, state))
    }

    fn write_plane(&mut self, target: Handle, bytes: &[u8]) -> Answer<()> {
        let (surface, length) = self.held_plane(target)?;
        if bytes.len() != length {
            return Err(Failure::named(
                "PlaneLength",
                format!("{} bytes for a plane of {length}", bytes.len()),
            ));
        }
        // No command buffer is in flight: a case writes the plane before it opens its pass and
        // reads it after the wait.
        surface
            .fill(bytes)
            .map_err(|why| Failure::named("WritePlane", why))
    }

    fn read_plane(&mut self, target: Handle) -> Answer<Vec<u8>> {
        let (surface, length) = self.held_plane(target)?;
        // The caller waited for the command buffer that wrote the plane before asking for it.
        surface
            .read(length)
            .map_err(|why| Failure::named("ReadPlane", why))
    }

    fn render(&mut self, work: &[RenderWork]) -> Answer<PassOutcome> {
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        // Metal's encoder methods answer nothing, so no piece of this side's program can refuse. A
        // refusal of ours then compares against Metal's answer and reads as a refusal.
        let refused = vec![None; work.len()];
        for piece in work {
            match piece {
                RenderWork::Pass(pass) => self.encode_pass(&command_buffer, pass)?,
                RenderWork::ReadTexture {
                    texture,
                    into,
                    offset,
                    row_pitch,
                    size,
                    plane,
                    at,
                } => self.encode_read_texture(
                    &command_buffer,
                    (*texture, *at, *size),
                    *into,
                    *offset,
                    *row_pitch,
                    *plane,
                )?,
                RenderWork::Dispatch(dispatch) => {
                    self.encode_dispatch(&command_buffer, dispatch)?;
                }
                RenderWork::CopyBuffer {
                    from,
                    from_offset,
                    into,
                    into_offset,
                    length,
                } => {
                    let source: Retained<ProtocolObject<dyn MTLBuffer>> =
                        Retained::from(self.held(*from)?);
                    let destination: Retained<ProtocolObject<dyn MTLBuffer>> =
                        Retained::from(self.held(*into)?);
                    for (buffer, offset, what) in [
                        (&source, *from_offset, "the source"),
                        (&destination, *into_offset, "the destination"),
                    ] {
                        let held = u64::try_from(buffer.length()).unwrap_or(0);
                        if offset.checked_add(*length).is_none_or(|end| end > held) {
                            return Err(Failure::named(
                                "PastTheBuffer",
                                format!("{length} bytes at {offset} of {what}, of {held}"),
                            ));
                        }
                    }
                    let blit = command_buffer.blitCommandEncoder().ok_or_else(|| {
                        Failure::named("BlitCommandEncoder", "returned nil".to_owned())
                    })?;
                    // SAFETY: both ranges lie inside their own buffer, checked just above.
                    unsafe {
                        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                            &source,
                            *from_offset as usize,
                            &destination,
                            *into_offset as usize,
                            *length as usize,
                        );
                    }
                    blit.endEncoding();
                }
                RenderWork::ComputeCommands {
                    commands,
                    first,
                    count,
                    used,
                } => {
                    self.encode_compute_commands(&command_buffer, *commands, *first, *count, used)?;
                }
                RenderWork::ResetCommands {
                    commands,
                    first,
                    count,
                } => {
                    let icb = self.held_commands(*commands)?;
                    let range = commands_range(&icb, *first, *count, "the buffer")?;
                    let blit = command_buffer.blitCommandEncoder().ok_or_else(|| {
                        Failure::named("BlitCommandEncoder", "returned nil".to_owned())
                    })?;
                    // SAFETY: the range lies inside the buffer's commands, checked just above,
                    // and the buffer outlives the call.
                    unsafe { blit.resetCommandsInBuffer_withRange(&icb, range) };
                    blit.endEncoding();
                }
                RenderWork::CopyCommands {
                    from,
                    first,
                    count,
                    into,
                    into_index,
                } => {
                    let source = self.held_commands(*from)?;
                    let destination = self.held_commands(*into)?;
                    let range = commands_range(&source, *first, *count, "the source")?;
                    // A copy that runs past the destination's end corrupts its command 0 on plain
                    // Metal and its validation stops neither, so the side refuses it here rather
                    // than asking a device what a corruption reads as.
                    commands_range(&destination, *into_index, *count, "the destination")?;
                    let blit = command_buffer.blitCommandEncoder().ok_or_else(|| {
                        Failure::named("BlitCommandEncoder", "returned nil".to_owned())
                    })?;
                    // SAFETY: the source range lies inside the source and the same count from
                    // `into_index` lies inside the destination, both checked just above, and both
                    // buffers outlive the call.
                    unsafe {
                        blit.copyIndirectCommandBuffer_sourceRange_destination_destinationIndex(
                            &source,
                            range,
                            &destination,
                            *into_index as usize,
                        );
                    }
                    blit.endEncoding();
                }
                RenderWork::OptimizeCommands {
                    commands,
                    first,
                    count,
                } => {
                    let icb = self.held_commands(*commands)?;
                    let range = commands_range(&icb, *first, *count, "the buffer")?;
                    let blit = command_buffer.blitCommandEncoder().ok_or_else(|| {
                        Failure::named("BlitCommandEncoder", "returned nil".to_owned())
                    })?;
                    // SAFETY: the range lies inside the buffer's commands, checked just above,
                    // and the buffer outlives the call.
                    unsafe { blit.optimizeIndirectCommandBuffer_withRange(&icb, range) };
                    blit.endEncoding();
                }
            }
        }
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let error = command_buffer
            .error()
            .map(|error| Value::unit_variant(format!("code {}", error.code())));
        let status = Value::unit_variant(status_word(command_buffer.status()));
        keep(&mut self.command_buffers, command_buffer);
        Ok(PassOutcome {
            refused,
            status,
            error,
        })
    }

    fn heap_texture_size_and_align(&mut self, query: &HeapTextureQuery) -> Answer<SizeAndAlign> {
        would_end_metals_process(query)?;
        let descriptor = queried_descriptor(query);
        let answered = self
            .device
            .heapTextureSizeAndAlignWithDescriptor(&descriptor);
        Ok(SizeAndAlign {
            size: answered.size as u64,
            align: answered.align as u64,
        })
    }

    fn heap_buffer_size_and_align(&mut self, query: &HeapBufferQuery) -> Answer<SizeAndAlign> {
        let answered = self.device.heapBufferSizeAndAlignWithLength_options(
            query.length as usize,
            MTLResourceOptions(query.options as usize),
        );
        Ok(SizeAndAlign {
            size: answered.size as u64,
            align: answered.align as u64,
        })
    }

    fn supports_vertex_amplification_count(&mut self, count: u64) -> Answer<bool> {
        Ok(self.device.supportsVertexAmplificationCount(count as usize))
    }

    fn supports_rasterization_rate_map_layers(&mut self, layers: u64) -> Answer<bool> {
        // SAFETY: the layer count is the case's own small positive number, which the sweep keeps
        // between 1 and 4; the query allocates nothing and makes no map.
        Ok(unsafe {
            self.device
                .supportsRasterizationRateMapWithLayerCount(layers as usize)
        })
    }

    fn minimum_linear_texture_alignment(&mut self, format: PixelFormat) -> Answer<u64> {
        Ok(self
            .device
            .minimumLinearTextureAlignmentForPixelFormat(MTLPixelFormat(format.as_raw() as usize))
            as u64)
    }

    fn convert_sparse_pixel_regions(&mut self, query: &RegionQuery) -> Answer<Vec<Box6>> {
        let mut pixels: Vec<MTLRegion> = query.regions.iter().map(metal_region).collect();
        let mut tiles: Vec<MTLRegion> = pixels.clone();
        let mode = match query.inward {
            true => MTLSparseTextureRegionAlignmentMode::Inward,
            false => MTLSparseTextureRegionAlignmentMode::Outward,
        };
        let (Some(from), Some(into)) = (
            std::ptr::NonNull::new(pixels.as_mut_ptr()),
            std::ptr::NonNull::new(tiles.as_mut_ptr()),
        ) else {
            return Err(Failure::named(
                "NoRegions",
                "the query states none".to_owned(),
            ));
        };
        // SAFETY: both pointers are to `query.regions.len()` elements of a `Vec` this call owns,
        // which is the count handed as `numRegions`; the tile size is the case's own, inside the
        // shapes the family's prose names.
        unsafe {
            self.device
                .convertSparsePixelRegions_toTileRegions_withTileSize_alignmentMode_numRegions(
                    from,
                    into,
                    size([
                        query.tile[0] as u32,
                        query.tile[1] as u32,
                        query.tile[2] as u32,
                    ]),
                    mode,
                    query.regions.len(),
                );
        }
        Ok(tiles.iter().map(our_region).collect())
    }

    fn convert_sparse_tile_regions(&mut self, query: &RegionQuery) -> Answer<Vec<Box6>> {
        let mut tiles: Vec<MTLRegion> = query.regions.iter().map(metal_region).collect();
        let mut pixels: Vec<MTLRegion> = tiles.clone();
        let (Some(from), Some(into)) = (
            std::ptr::NonNull::new(tiles.as_mut_ptr()),
            std::ptr::NonNull::new(pixels.as_mut_ptr()),
        ) else {
            return Err(Failure::named(
                "NoRegions",
                "the query states none".to_owned(),
            ));
        };
        // SAFETY: as the other direction's.
        unsafe {
            self.device
                .convertSparseTileRegions_toPixelRegions_withTileSize_numRegions(
                    from,
                    into,
                    size([
                        query.tile[0] as u32,
                        query.tile[1] as u32,
                        query.tile[2] as u32,
                    ]),
                    query.regions.len(),
                );
        }
        Ok(pixels.iter().map(our_region).collect())
    }

    fn queue_slots(&mut self, max: Option<u32>, cap: u32) -> Answer<QueueSlots> {
        let queue = match max {
            None => self.device.newCommandQueue(),
            Some(max) => self
                .device
                .newCommandQueueWithMaxCommandBufferCount(max as usize),
        }
        .ok_or_else(|| Failure::named("NewCommandQueue", format!("{max:?} returned nil")))?;
        let other = self
            .device
            .newCommandQueue()
            .ok_or_else(|| Failure::named("NewCommandQueue", "the second queue returned nil"))?;
        crate::queue_slots::probe(
            cap,
            move || {
                // Each make gets an autorelease pool of its own, and that is load-bearing rather
                // than tidy. `commandBuffer` answers an autoreleased object; the `Retained` is a
                // second reference, so without a pool that drains, the probe's own release frees
                // nothing, the slot never comes back, and the blocked make it is meant to release
                // stays blocked (measured: the whole probe timed out). A worker thread has no pool
                // of its own, so one is made here.
                objc2::rc::autoreleasepool(|_| {
                    queue
                        .commandBuffer()
                        .map(Made)
                        .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))
                })
            },
            || objc2::rc::autoreleasepool(|_| other.commandBuffer().is_some()),
        )
    }

    fn shared_event_handle_round_trip(
        &mut self,
        label: &str,
    ) -> Answer<crate::shared_events::EventHandleRoundTrip> {
        use crate::shared_events::{
            EventHandleRoundTrip, HANDLE_FIRST_VALUE, HANDLE_SECOND_VALUE, HANDLE_UNREACHED_VALUE,
            WAIT_MS,
        };
        let registry = self.device.registryID();
        let event = self
            .device
            .newSharedEvent()
            .ok_or_else(|| Failure::named("NewSharedEvent", "returned nil".to_owned()))?;
        event.setLabel(Some(&NSString::from_str(label)));
        let read_label = event.label().map(|label| label.to_string());
        // The registry ID is each host's own number, so only the **comparison** is portable; an
        // event that answers no device at all is recorded as not being the case's.
        let device_is_the_cases = event
            .device()
            .is_some_and(|held| held.registryID() == registry);
        let value_at_first = event.signaledValue();
        event.setSignaledValue(HANDLE_FIRST_VALUE);
        let value_after_the_host_set_it = event.signaledValue();
        let handle: Retained<MTLSharedEventHandle> = event.newSharedEventHandle();
        let handle_label = handle.label().map(|label| label.to_string());
        let reopened = self
            .device
            .newSharedEventWithHandle(&handle)
            .ok_or_else(|| Failure::named("NewSharedEventWithHandle", "returned nil".to_owned()))?;
        let reopened_label = reopened.label().map(|label| label.to_string());
        let reopened_device_is_the_cases = reopened
            .device()
            .is_some_and(|held| held.registryID() == registry);
        let reopened_value = reopened.signaledValue();
        reopened.setSignaledValue(HANDLE_SECOND_VALUE);
        let value_on_the_first_after_the_second_was_set = event.signaledValue();
        let reopened_waits = [
            reopened.waitUntilSignaledValue_timeoutMS(HANDLE_SECOND_VALUE, WAIT_MS),
            reopened.waitUntilSignaledValue_timeoutMS(HANDLE_UNREACHED_VALUE, WAIT_MS),
        ];
        Ok(EventHandleRoundTrip {
            label: read_label,
            device_is_the_cases,
            value_at_first,
            value_after_the_host_set_it,
            handle_label,
            reopened_label,
            reopened_device_is_the_cases,
            reopened_value,
            value_on_the_first_after_the_second_was_set,
            reopened_waits,
        })
    }

    fn shared_event_rises(&mut self, label: &str) -> Answer<crate::shared_events::EventRise> {
        use crate::shared_events::{EventRise, WAIT_MS};
        let event = self
            .device
            .newSharedEvent()
            .ok_or_else(|| Failure::named("NewSharedEvent", "returned nil".to_owned()))?;
        event.setLabel(Some(&NSString::from_str(label)));
        let read_label = event.label().map(|label| label.to_string());
        let initial = event.signaledValue();
        // The host's sets: a lower value leaves the event where it was.
        event.setSignaledValue(5);
        event.setSignaledValue(3);
        event.setSignaledValue(5);
        let after_host_sets = event.signaledValue();
        let as_event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event);
        let signal = |value: u64| -> Answer<u64> {
            // One autorelease pool per signal: the command buffer this answers is autoreleased,
            // and a loop that makes several without a pool that drains keeps every one of them
            // alive.
            objc2::rc::autoreleasepool(|_| {
                let command_buffer = self
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
                command_buffer.encodeSignalEvent_value(as_event, value);
                command_buffer.commit();
                command_buffer.waitUntilCompleted();
                Ok(event.signaledValue())
            })
        };
        let after_gpu_signals = [signal(3)?, signal(5)?];
        let waits = [
            event.waitUntilSignaledValue_timeoutMS(3, WAIT_MS),
            event.waitUntilSignaledValue_timeoutMS(5, WAIT_MS),
            event.waitUntilSignaledValue_timeoutMS(6, WAIT_MS),
        ];
        let after_max_signal = signal(u64::MAX)?;
        let waited_for_max = event.waitUntilSignaledValue_timeoutMS(u64::MAX, WAIT_MS);
        Ok(EventRise {
            label: read_label,
            initial,
            after_host_sets,
            after_gpu_signals,
            waits,
            after_max_signal,
            waited_for_max,
        })
    }

    fn event_cross_queue_wait(&mut self) -> Answer<crate::event_orders::CrossQueueWait> {
        use crate::event_orders::{
            CrossQueueWait, Status, HALF, SCRATCH, SIGNALLING_BYTE, WAITING_BYTE,
        };
        // One pool around the whole probe: `commandBuffer`, `newEvent` and `newCommandQueue` each
        // answer an autoreleased object, and a probe's thread has no pool of its own. Without this
        // the command buffers would live in the never-drained top-level pool and hold their
        // queues' slots for good.
        objc2::rc::autoreleasepool(|_| {
            let queue = || {
                self.device
                    .newCommandQueue()
                    .ok_or_else(|| Failure::named("NewCommandQueue", "returned nil"))
            };
            let (waiting_queue, signalling_queue) = (queue()?, queue()?);
            let event = self
                .device
                .newEvent()
                .ok_or_else(|| Failure::named("NewEvent", "returned nil"))?;
            let scratch = self
                .device
                .newBufferWithLength_options(
                    SCRATCH as usize,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| Failure::named("NewBuffer", "returned nil"))?;
            let fill = |command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
                        offset: u64,
                        length: u64,
                        byte: u8|
             -> Result<(), Failure> {
                let encoder = command_buffer
                    .blitCommandEncoder()
                    .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil"))?;
                encoder.fillBuffer_range_value(
                    &scratch,
                    objc2_foundation::NSRange::new(offset as usize, length as usize),
                    byte,
                );
                encoder.endEncoding();
                Ok(())
            };
            let waiting = waiting_queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            waiting.encodeWaitForEvent_value(&event, 1);
            fill(&waiting, 0, HALF, WAITING_BYTE)?;
            let signalling = signalling_queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            fill(&signalling, HALF, HALF, SIGNALLING_BYTE)?;
            signalling.encodeSignalEvent_value(&event, 1);
            // Bounded completion, rather than `waitUntilCompleted`, which Metal has no bounded
            // form of: the handler is added before the commit and the probe waits on its channel.
            let waiting_done = completion_channel(&waiting);
            let signalling_done = completion_channel(&signalling);
            // The waiting one first, which is the whole point: its signal is not committed yet.
            waiting.commit();
            let waiting_after_its_commit = Status::read(status_word(waiting.status()));
            signalling.commit();
            let signalling_at_the_end = settled(&signalling_done, &signalling);
            let waiting_at_the_end = settled(&waiting_done, &waiting);
            // SAFETY: `SCRATCH` bytes of the shared buffer, whose writers have both completed.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    scratch.contents().as_ptr().cast::<u8>(),
                    SCRATCH as usize,
                )
            }
            .to_vec();
            Ok(CrossQueueWait {
                waiting_after_its_commit,
                waiting_at_the_end,
                signalling_at_the_end,
                bytes,
            })
        })
    }

    fn event_highest_signal(&mut self, values: [u64; 2]) -> Answer<u64> {
        use crate::event_orders::SCRATCH;
        objc2::rc::autoreleasepool(|_| {
            let queue = || {
                self.device
                    .newCommandQueue()
                    .ok_or_else(|| Failure::named("NewCommandQueue", "returned nil"))
            };
            let queues = [queue()?, queue()?];
            let event = self
                .device
                .newSharedEvent()
                .ok_or_else(|| Failure::named("NewSharedEvent", "returned nil"))?;
            let as_event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event);
            let scratch = self
                .device
                .newBufferWithLength_options(
                    SCRATCH as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| Failure::named("NewBuffer", "returned nil"))?;
            let mut done = Vec::new();
            for (queue, &value) in queues.iter().zip(&values) {
                let command_buffer = queue
                    .commandBuffer()
                    .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
                let encoder = command_buffer
                    .blitCommandEncoder()
                    .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil"))?;
                encoder.fillBuffer_range_value(
                    &scratch,
                    objc2_foundation::NSRange::new(0, SCRATCH as usize),
                    value as u8,
                );
                encoder.endEncoding();
                command_buffer.encodeSignalEvent_value(as_event, value);
                let channel = completion_channel(&command_buffer);
                command_buffer.commit();
                done.push(channel);
            }
            for channel in &done {
                if channel
                    .done
                    .recv_timeout(crate::event_orders::COMPLETION)
                    .is_err()
                {
                    return Err(Failure::named(
                        "NeverCompleted",
                        "a signalling command buffer",
                    ));
                }
            }
            Ok(event.signaledValue())
        })
    }

    fn held_command_buffer(&mut self) -> Answer<crate::event_orders::HeldCommandBuffer> {
        use crate::event_orders::{
            HeldCommandBuffer, Status, COMPLETED, COMPLETION, HELD_FOR, SCHEDULED, SCRATCH,
        };
        objc2::rc::autoreleasepool(|_| {
            let queue = || {
                self.device
                    .newCommandQueue()
                    .ok_or_else(|| Failure::named("NewCommandQueue", "returned nil"))
            };
            let (held_queue, other_queue) = (queue()?, queue()?);
            let event = self
                .device
                .newSharedEvent()
                .ok_or_else(|| Failure::named("NewSharedEvent", "returned nil"))?;
            let as_event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event);
            let scratch = self
                .device
                .newBufferWithLength_options(
                    SCRATCH as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| Failure::named("NewBuffer", "returned nil"))?;
            let fill = |command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
                        byte: u8|
             -> Result<(), Failure> {
                let encoder = command_buffer
                    .blitCommandEncoder()
                    .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil"))?;
                encoder.fillBuffer_range_value(
                    &scratch,
                    objc2_foundation::NSRange::new(0, SCRATCH as usize),
                    byte,
                );
                encoder.endEncoding();
                Ok(())
            };
            let mut log = HandlerLog::new();
            let held = held_queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            held.encodeWaitForEvent_value(as_event, 1);
            fill(&held, 1)?;
            // SAFETY: each block is a valid heap block the log keeps alive for as long as this
            // probe runs, which outlives every call Metal makes of it.
            log.add(SCHEDULED, |block| unsafe {
                held.addScheduledHandler(block)
            });
            log.add(COMPLETED, |block| unsafe {
                held.addCompletedHandler(block)
            });
            held.commit();
            let behind = held_queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            fill(&behind, 2)?;
            let behind_done = completion_channel(&behind);
            behind.commit();
            let other = other_queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            fill(&other, 3)?;
            let other_done = completion_channel(&other);
            other.commit();
            let other_queue_status = settled(&other_done, &other);
            // The window the negative facts are false in: a submission that was going to happen
            // has had its chance by now.
            std::thread::sleep(HELD_FOR);
            let held_while_waiting = Status::read(status_word(held.status()));
            let behind_while_waiting = Status::read(status_word(behind.status()));
            let handlers_while_waiting = log.names();
            event.setSignaledValue(1);
            // The held command buffer's own completed handler is the bound: it is the last of its
            // handlers to run, so once it has, the log is whole and the status is settled.
            log.wait_for(COMPLETED, COMPLETION);
            let handlers_at_the_end = log.names();
            let held_at_the_end = if handlers_at_the_end.iter().any(|name| name == COMPLETED) {
                Status::read(status_word(held.status()))
            } else {
                Status::NeverSettled
            };
            let behind_at_the_end = settled(&behind_done, &behind);
            Ok(HeldCommandBuffer {
                other_queue: other_queue_status,
                held_while_waiting,
                behind_while_waiting,
                handlers_while_waiting,
                handlers_at_the_end,
                held_at_the_end,
                behind_at_the_end,
            })
        })
    }

    fn signal_before_a_wait(&mut self) -> Answer<crate::event_orders::SignalBeforeWait> {
        use crate::event_orders::{Status, HELD_FOR, SCRATCH, SIGNAL_WAIT};
        objc2::rc::autoreleasepool(|_| {
            let queue = self
                .device
                .newCommandQueue()
                .ok_or_else(|| Failure::named("NewCommandQueue", "returned nil"))?;
            let event = || {
                self.device
                    .newSharedEvent()
                    .ok_or_else(|| Failure::named("NewSharedEvent", "returned nil"))
            };
            let (before, gate, after) = (event()?, event()?, event()?);
            let scratch = self
                .device
                .newBufferWithLength_options(
                    SCRATCH as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| Failure::named("NewBuffer", "returned nil"))?;
            let command_buffer = queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            let fill = |byte: u8| -> Result<(), Failure> {
                let encoder = command_buffer
                    .blitCommandEncoder()
                    .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil"))?;
                encoder.fillBuffer_range_value(
                    &scratch,
                    objc2_foundation::NSRange::new(0, SCRATCH as usize),
                    byte,
                );
                encoder.endEncoding();
                Ok(())
            };
            let before_event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*before);
            let gate_event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*gate);
            let after_event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*after);
            fill(1)?;
            command_buffer.encodeSignalEvent_value(before_event, 1);
            command_buffer.encodeWaitForEvent_value(gate_event, 1);
            fill(2)?;
            command_buffer.encodeSignalEvent_value(after_event, 1);
            let done = completion_channel(&command_buffer);
            command_buffer.commit();
            let before_landed =
                before.waitUntilSignaledValue_timeoutMS(1, SIGNAL_WAIT.as_millis() as u64);
            // The window the two negative facts are false in: the signal behind the wait has not
            // landed, and the command buffer is no further along than committed.
            std::thread::sleep(HELD_FOR);
            let after_while_pending = after.signaledValue();
            let status_while_pending = Status::read(status_word(command_buffer.status()));
            gate.setSignaledValue(1);
            let status_at_the_end = settled(&done, &command_buffer);
            Ok(crate::event_orders::SignalBeforeWait {
                before_landed,
                after_while_pending,
                status_while_pending,
                after_at_the_end: after.signaledValue(),
                status_at_the_end,
            })
        })
    }

    fn detach_in_a_completed_handler(
        &mut self,
        buffer: Handle,
        after: std::time::Duration,
    ) -> Answer<crate::host_detach::DetachedInHandler> {
        use crate::event_orders::Status;
        use crate::host_detach::{DetachedInHandler, FILL_BYTE, FILL_BYTES};
        use std::sync::mpsc;
        let detaching = self
            .buffers
            .get(buffer.0 as usize)
            .cloned()
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("buffer {}", buffer.0)))?;
        // Apple SPI: no public header names `detachBacking`, so it is sent by selector and only
        // after the buffer answered that it responds. A release that stops answering it gives a
        // typed refusal naming the selector rather than an unrecognised-selector crash.
        require_selector(&*detaching, objc2::sel!(detachBacking), "detachBacking")?;
        objc2::rc::autoreleasepool(|_| {
            let queue = self
                .device
                .newCommandQueue()
                .ok_or_else(|| Failure::named("NewCommandQueue", "returned nil"))?;
            let command_buffer = queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            let encoder = command_buffer
                .blitCommandEncoder()
                .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil"))?;
            // The command buffer has to *use* the buffer: a submission that named it is the only
            // thing that can hold it, and a detach asked about a buffer nothing took is a
            // different question.
            let length = detaching
                .length()
                .min(usize::try_from(FILL_BYTES).unwrap_or(usize::MAX));
            encoder.fillBuffer_range_value(&detaching, NSRange::new(0, length), FILL_BYTE);
            encoder.endEncoding();
            let (sender, answered) = mpsc::channel();
            let handler = {
                let detaching = detaching.clone();
                block2::RcBlock::new(
                    move |_: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                        // Past the race between Metal's own release of what the command buffer
                        // used and this handler's start, which is the region where Metal's answer
                        // is deterministic (`crate::host_detach`).
                        std::thread::sleep(after);
                        let _ = sender.send(detach_backing(&detaching));
                    },
                )
            };
            // SAFETY: the block is a valid heap block held in `handler` for the rest of this
            // probe, which does not return before the handler answered or its bound ran out.
            unsafe { command_buffer.addCompletedHandler(block2::RcBlock::as_ptr(&handler)) };
            command_buffer.commit();
            // The handler's own channel, with a bound. `waitUntilCompleted` is deliberately not
            // called: Metal has no bounded form of it, and an answer that did not arrive is a
            // recorded fact rather than a hang.
            let (detached, status) = match answered.recv_timeout(crate::host_detach::WAIT) {
                Ok(detached) => (
                    Some(detached),
                    Status::read(status_word(command_buffer.status())),
                ),
                Err(_) => (None, Status::NeverSettled),
            };
            Ok(DetachedInHandler { detached, status })
        })
    }

    fn scheduled_handler_order(
        &mut self,
    ) -> Answer<crate::scheduled_handlers::ScheduledHandlerOrder> {
        use crate::event_orders::{Status, COMPLETED, SCHEDULED, SCRATCH};
        use crate::scheduled_handlers::{
            Returned, ScheduledHandlerOrder, ARRIVAL, HOLD_LIMIT, WINDOW,
        };
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{mpsc, Arc, Mutex};
        objc2::rc::autoreleasepool(|_| {
            let queue = self
                .device
                .newCommandQueue()
                .ok_or_else(|| Failure::named("NewCommandQueue", "returned nil"))?;
            let scratch = self
                .device
                .newBufferWithLength_options(
                    SCRATCH as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| Failure::named("NewBuffer", "returned nil"))?;
            let command_buffer = queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            let encoder = command_buffer
                .blitCommandEncoder()
                .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil"))?;
            encoder.fillBuffer_range_value(
                &scratch,
                objc2_foundation::NSRange::new(0, SCRATCH as usize),
                1,
            );
            encoder.endEncoding();
            let order: Arc<Mutex<Vec<String>>> = Arc::default();
            let (entered, entering) = mpsc::channel();
            let (release, released) = mpsc::channel::<()>();
            let held = {
                let order = Arc::clone(&order);
                block2::RcBlock::new(
                    move |own: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                        if let Ok(mut order) = order.lock() {
                            order.push(SCHEDULED.to_owned());
                        }
                        // SAFETY: Metal hands the handler its own command buffer, which is alive
                        // for the call.
                        let status = unsafe { own.as_ref() }.status();
                        let _ = entered.send(status_word(status).to_owned());
                        // Held, so the probe can read the status and the wait while this handler
                        // is inside itself. Bounded, so the handler leaves Metal's handler thread
                        // whatever the probe does.
                        let _ = released.recv_timeout(HOLD_LIMIT);
                    },
                )
            };
            let (done, completing) = mpsc::channel();
            let finished = {
                let order = Arc::clone(&order);
                block2::RcBlock::new(
                    move |own: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                        if let Ok(mut order) = order.lock() {
                            order.push(COMPLETED.to_owned());
                        }
                        // SAFETY: as above.
                        let status = unsafe { own.as_ref() }.status();
                        let _ = done.send(status_word(status).to_owned());
                    },
                )
            };
            // SAFETY: both blocks are valid heap blocks held in `held` and `finished` for the rest
            // of this probe, which outlives every call Metal makes of them: the probe does not
            // return before the completed handler ran or its bound ran out.
            unsafe {
                command_buffer.addScheduledHandler(block2::RcBlock::as_ptr(&held));
                command_buffer.addCompletedHandler(block2::RcBlock::as_ptr(&finished));
            }
            command_buffer.commit();
            let inside_the_handler = match entering.recv_timeout(ARRIVAL) {
                Ok(word) => Status::read(word),
                Err(_) => Status::NeverSettled,
            };
            let while_the_handler_ran = Status::read(status_word(command_buffer.status()));
            let returned = Arc::new(AtomicBool::new(false));
            let (answered, answer) = mpsc::channel();
            {
                let (waited, returned) = (Waited(command_buffer.clone()), Arc::clone(&returned));
                // The wait runs on a thread of its own because the fact under test is that it does
                // **not** return; the thread is left to finish rather than joined, so a wait that
                // never came back leaves that thread and not the suite.
                std::thread::spawn(move || {
                    waited.wait_until_scheduled();
                    returned.store(true, Ordering::SeqCst);
                    let _ = answered.send(());
                });
            }
            // The window the negative fact is false in.
            std::thread::sleep(WINDOW);
            let wait_returned_while_the_handler_ran = returned.load(Ordering::SeqCst);
            let _ = release.send(());
            // Metal's `waitUntilScheduled` answers nothing, so a return is the whole answer.
            let wait_answered = match answer.recv_timeout(ARRIVAL) {
                Ok(()) => Returned::Ok,
                Err(_) => Returned::NotWithinTheWindow,
            };
            let inside_the_completed_handler = match completing.recv_timeout(ARRIVAL) {
                Ok(word) => Status::read(word),
                Err(_) => Status::NeverSettled,
            };
            let handler_order = order.lock().map(|order| order.clone()).unwrap_or_default();
            Ok(ScheduledHandlerOrder {
                inside_the_handler,
                while_the_handler_ran,
                wait_returned_while_the_handler_ran,
                wait_answered,
                inside_the_completed_handler,
                handler_order,
            })
        })
    }

    /// A texture over one plane of a real IOSurface laid out as the case says its stand-in buffer
    /// is (`newTextureWithDescriptor:iosurface:plane:`), or over the buffer itself.
    ///
    /// The surface is the only way Metal makes a texture of a two-plane YCbCr format at all: the
    /// descriptor alone answers nil for every one of the 69, which is what
    /// `ycbcr-two-plane-format-descriptors` records. So this is where the buffer the case states
    /// becomes the thing Metal takes — same plane offsets, same row pitches, same bytes from byte
    /// 0 — and the surface is kept beside the texture so it outlives it.
    fn plane_texture(
        &mut self,
        buffer: Handle,
        over: &crate::case::PlaneTexture,
    ) -> Answer<Handle> {
        let bytes = {
            let held = self.held(buffer)?;
            let length = held.length();
            // SAFETY: the buffer's own shared storage, whole, and no command uses it yet.
            unsafe {
                std::slice::from_raw_parts(held.contents().as_ptr().cast::<u8>(), length).to_vec()
            }
        };
        let [width, height] = over.size;
        // The descriptor and the texture are autoreleased, and a case may make several over a 24
        // MiB surface, so they are made inside a pool rather than left to a thread that drains
        // none.
        let made = objc2::rc::autoreleasepool(|_| {
            let descriptor = MTLTextureDescriptor::new();
            descriptor.setTextureType(MTLTextureType::Type2D);
            descriptor.setPixelFormat(MTLPixelFormat(over.format as usize));
            // SAFETY: the case's own extent; the setters read nothing but the value given.
            unsafe {
                descriptor.setWidth(width as usize);
                descriptor.setHeight(height as usize);
            }
            descriptor.setStorageMode(MTLStorageMode::Shared);
            match over.ostype {
                Some(ostype) => {
                    // The usage the runtime side's plane texture takes: a surface's plane is read.
                    descriptor.setUsage(MTLTextureUsage::ShaderRead);
                    let format = PixelFormat::try_from(over.format).map_err(|problem| {
                        Failure::named("NoSuchPixelFormat", format!("{}: {problem}", over.format))
                    })?;
                    let planes = crate::metal_surface::planes_of(format, over.size, over.planes)
                        .map_err(|why| Failure::named("NoSuchPlaneLayout", why))?;
                    crate::metal_surface::planes_hold_their_texels(&planes, bytes.len())
                        .map_err(|why| Failure::named("PlanesPastTheBuffer", why))?;
                    let surface = crate::metal_surface::Surface::two_plane(
                        ostype,
                        over.size,
                        &planes,
                        bytes.len(),
                    )
                    .map_err(|why| Failure::named("NoSurface", why))?;
                    surface
                        .planes_are_as_asked(&planes)
                        .map_err(|why| Failure::named("SurfaceLaidOtherwise", why))?;
                    surface
                        .fill(&bytes)
                        .map_err(|why| Failure::named("SurfaceNotFilled", why))?;
                    let device: &ProtocolObject<dyn MTLDevice> = &self.device;
                    let descriptor: &MTLTextureDescriptor = &descriptor;
                    let plane = over.plane as usize;
                    require_selector(
                        device,
                        objc2::sel!(newTextureWithDescriptor:iosurface:plane:),
                        "MTLDevice newTextureWithDescriptor:iosurface:plane:",
                    )?;
                    // SAFETY: `newTextureWithDescriptor:iosurface:plane:` of a live surface whose
                    // planes it states, at one of them; the surface is kept below, so it outlives
                    // the texture that reads it.
                    let texture: Option<Retained<ProtocolObject<dyn MTLTexture>>> = unsafe {
                        objc2::msg_send![device, newTextureWithDescriptor: descriptor, iosurface: surface.as_ptr(), plane: plane]
                    };
                    let texture = texture.ok_or_else(|| {
                        Failure::named(
                            "NewTextureWithIOSurfacePlane",
                            format!(
                                "format {} over plane {} of a {width}x{height} '{}' surface \
                                 returned nil",
                                over.format,
                                over.plane,
                                crate::metal_surface::four_cc(ostype)
                            ),
                        )
                    })?;
                    Ok((texture, Some(surface)))
                }
                None => {
                    // The usage the runtime side's texture over a buffer takes.
                    descriptor.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
                    let (offset, row) = over.planes[0];
                    let held = self.held(buffer)?;
                    let texture = held
                        .newTextureWithDescriptor_offset_bytesPerRow(
                            &descriptor,
                            offset as usize,
                            row as usize,
                        )
                        .ok_or_else(|| {
                            Failure::named(
                                "NewTextureWithDescriptorOffsetBytesPerRow",
                                format!(
                                    "format {} {width}x{height} at {offset} of {row} a row \
                                     returned nil",
                                    over.format
                                ),
                            )
                        })?;
                    Ok((texture, None))
                }
            }
        })?;
        let (texture, surface) = made;
        if let Some(surface) = surface {
            self.surfaces.push(surface);
        }
        Ok(keep(&mut self.textures, texture))
    }

    fn buffer_texture(
        &mut self,
        request: &crate::buffer_texture::BufferTextureRequest,
    ) -> Answer<crate::buffer_texture::BufferTextureFacts> {
        // The guard first, and before any object is made: a row pitch below the texture's own row
        // fails AGX's `bytes_per_row >= used_bytes_per_row` and ends this process, so the call is
        // never made. A case of that shape says `MetalSource::did_not_return` and record mode
        // refuses it; this is the second line, for a row somebody adds to a recorded table by
        // mistake.
        if let Some(why) = crate::buffer_texture::ends_metals_process(request) {
            return Err(Failure::named("WouldEndMetalsProcess", why));
        }
        let (_, options) = storage_mode(request.buffer_storage);
        let buffer = self
            .device
            .newBufferWithLength_options(request.buffer_length as usize, options)
            .ok_or_else(|| {
                Failure::named(
                    "NewBufferWithLength",
                    format!("{} bytes returned nil", request.buffer_length),
                )
            })?;
        let descriptor = queried_descriptor(&request.descriptor);
        let texture = buffer
            .newTextureWithDescriptor_offset_bytesPerRow(
                &descriptor,
                request.offset as usize,
                request.bytes_per_row as usize,
            )
            .ok_or_else(|| {
                // Metal's own nil: the descriptor and the three numbers describe no texture over
                // the buffer, and the process goes on. The runtime answers `Error::Nil` for the
                // same shape, so the two sides say the same thing in their own spellings.
                Failure::named(
                    "Nil",
                    format!(
                        "newTextureWithDescriptor:offset:{}:bytesPerRow:{} returned nil",
                        request.offset, request.bytes_per_row
                    ),
                )
            })?;
        let facts = buffer_texture_facts(&texture);
        // The buffer and the texture are kept, so neither is freed while the sweep goes on and a
        // later row cannot read memory a dropped texture handed back.
        keep(&mut self.buffers, buffer);
        keep(&mut self.textures, texture);
        Ok(facts)
    }

    fn queried_texture_facts(
        &mut self,
        query: &HeapTextureQuery,
    ) -> Answer<Option<crate::buffer_texture::BufferTextureFacts>> {
        would_end_metals_process(query)?;
        let descriptor = queried_descriptor(query);
        let Some(texture) = self.device.newTextureWithDescriptor(&descriptor) else {
            return Ok(None);
        };
        let facts = buffer_texture_facts(&texture);
        self.textures.push(texture);
        Ok(Some(facts))
    }

    fn makes_texture(&mut self, query: &HeapTextureQuery) -> Answer<bool> {
        would_end_metals_process(query)?;
        let descriptor = queried_descriptor(query);
        // The texture, where Metal makes one, is released as this statement ends: the case asks
        // this only of descriptors Metal makes none of, and nothing here keeps a handle, so no
        // sweep of the family holds a texture it does not read.
        Ok(self.device.newTextureWithDescriptor(&descriptor).is_some())
    }

    fn sparse_tile_size(&mut self, query: &SparseTileQuery) -> Answer<TileSize> {
        let texture_type = MTLTextureType(query.texture_type.as_raw() as usize);
        let format = MTLPixelFormat(query.format.as_raw() as usize);
        let samples = query.samples as usize;
        // SAFETY: both getters read the three scalars they are given and answer an `MTLSize`.
        // Neither allocates, compiles or reaches the GPU, and neither takes an object.
        let size = unsafe {
            match query.page {
                None => self
                    .device
                    .sparseTileSizeWithTextureType_pixelFormat_sampleCount(
                        texture_type,
                        format,
                        samples,
                    ),
                Some(page) => self
                    .device
                    .sparseTileSizeWithTextureType_pixelFormat_sampleCount_sparsePageSize(
                        texture_type,
                        format,
                        samples,
                        metal_page(page),
                    ),
            }
        };
        Ok(TileSize {
            width: size.width as u64,
            height: size.height as u64,
            depth: size.depth as u64,
        })
    }

    fn sparse_tile_size_in_bytes(&mut self, page: QueryPage) -> Answer<u64> {
        Ok(match page {
            None => self.device.sparseTileSizeInBytes() as u64,
            Some(page) => {
                self.device
                    .sparseTileSizeInBytesForSparsePageSize(metal_page(page)) as u64
            }
        })
    }

    fn sparse_heap(&mut self, spec: &SparseHeapSpec) -> Answer<Handle> {
        let descriptor = MTLHeapDescriptor::new();
        descriptor.setType(MTLHeapType::Sparse);
        descriptor.setStorageMode(MTLStorageMode::Private);
        descriptor.setSparsePageSize(metal_page(spec.page));
        descriptor.setSize(spec.size as usize);
        let heap = self
            .device
            .newHeapWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewHeapWithDescriptor",
                    format!("a sparse heap of {} bytes returned nil", spec.size),
                )
            })?;
        self.heaps.push(heap);
        Ok(Handle(self.heaps.len() as u32 - 1))
    }

    fn heap_facts(&mut self, heap: Handle) -> Answer<HeapFacts> {
        let heap = self.held_heap(heap)?;
        Ok(HeapFacts {
            heap_type: heap.r#type().0 as u64,
            size: heap.size() as u64,
            used_size: heap.usedSize() as u64,
        })
    }

    fn heap_texture(&mut self, heap: Handle, spec: &SparseTextureSpec) -> Answer<Handle> {
        let descriptor = sparse_texture_descriptor(spec);
        let texture = self
            .held_heap(heap)?
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!("{} on a heap returned nil", spec.site()),
                )
            })?;
        self.textures.push(texture);
        Ok(Handle(self.textures.len() as u32 - 1))
    }

    fn sparse_texture_facts(&mut self, texture: Handle) -> Answer<SparseTextureFacts> {
        let texture = self.held_texture(texture)?;
        Ok(SparseTextureFacts {
            is_sparse: texture.isSparse(),
            first_mipmap_in_tail: texture.firstMipmapInTail() as u64,
            tail_size_in_bytes: texture.tailSizeInBytes() as u64,
        })
    }

    fn heap(&mut self, spec: &HeapSpec) -> Answer<Handle> {
        // The guard first, and before any object is made: the shapes it names are ones Metal's own
        // validation stops, which ends this process. A case of such a shape is refused by its own
        // table's guard before record mode runs; this is the second line, for a row somebody adds
        // to a recorded table by mistake.
        if let Some(why) = spec.is_well_formed() {
            return Err(Failure::named("WouldEndMetalsProcess", why));
        }
        let descriptor = MTLHeapDescriptor::new();
        descriptor.setType(metal_heap_type(spec.heap_type));
        descriptor.setSize(spec.size as usize);
        descriptor.setStorageMode(storage_mode(spec.storage).0);
        descriptor.setCpuCacheMode(MTLCPUCacheMode(spec.cache.raw() as usize));
        descriptor.setHazardTrackingMode(MTLHazardTrackingMode(spec.hazard.raw() as usize));
        let heap = self
            .device
            .newHeapWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewHeapWithDescriptor",
                    format!("{} returned nil", spec.site()),
                )
            })?;
        self.heaps.push(heap);
        Ok(Handle(self.heaps.len() as u32 - 1))
    }

    fn heap_own_facts(&mut self, heap: Handle) -> Answer<HeapOwnFacts> {
        let registry = self.device.registryID();
        let heap = self.held_heap(heap)?;
        Ok(HeapOwnFacts {
            heap_type: heap.r#type().0 as u64,
            size: heap.size() as u64,
            used_size: heap.usedSize() as u64,
            current_allocated_size: heap.currentAllocatedSize() as u64,
            storage_mode: heap.storageMode().0 as u64,
            cpu_cache_mode: heap.cpuCacheMode().0 as u64,
            hazard_tracking_mode: heap.hazardTrackingMode().0 as u64,
            resource_options: heap.resourceOptions().0 as u64,
            label: heap.label().map(|label| label.to_string()),
            // The relation, not the pointer: a recorded message holds no address, and what Metal
            // settles is whether a heap answers the device that made it. The registry ID is the
            // spelling, because object identity is not a question both APIs can be asked
            // (`HeapOwnFacts::own_device`).
            own_device: heap.device().registryID() == registry,
        })
    }

    fn heap_max_available_size(&mut self, heap: Handle, alignment: u64) -> Answer<u64> {
        Ok(self
            .held_heap(heap)?
            .maxAvailableSizeWithAlignment(alignment as usize) as u64)
    }

    fn set_heap_label(&mut self, heap: Handle, label: &str) -> Answer<()> {
        self.held_heap(heap)?
            .setLabel(Some(&NSString::from_str(label)));
        Ok(())
    }

    fn heap_purgeable_state(&mut self, heap: Handle, state: PurgeableState) -> Answer<u64> {
        Ok(self
            .held_heap(heap)?
            .setPurgeableState(metal_purgeable(state))
            .0 as u64)
    }

    fn heap_buffer(&mut self, heap: Handle, spec: &HeapBufferSpec) -> Answer<Handle> {
        let held = self.held_heap(heap)?;
        // The options are the heap's own, which Metal requires of a placed resource: any other is
        // a call its validation stops.
        let options = held.resourceOptions();
        let buffer = match spec.offset {
            None => held.newBufferWithLength_options(spec.length as usize, options),
            // SAFETY: the length and the offset are the case's own, the offset is a multiple of
            // Metal's 256-byte placement grain and the bytes lie inside the heap — which the
            // case's own guard decides from the row's shape before record mode runs.
            Some(offset) => unsafe {
                held.newBufferWithLength_options_offset(
                    spec.length as usize,
                    options,
                    offset as usize,
                )
            },
        }
        .ok_or_else(|| {
            // Metal's own nil: it places no buffer of this length at this offset, and the process
            // goes on. The runtime answers `Error::Nil` for the same shape.
            Failure::named("Nil", format!("{} returned nil", spec.call()))
        })?;
        Ok(keep(&mut self.buffers, buffer))
    }

    fn resource_facts(
        &mut self,
        of: QueriedResource,
        heap: Option<Handle>,
    ) -> Answer<ResourceFacts> {
        let registry = self.device.registryID();
        let placed_in: Option<*const ProtocolObject<dyn MTLHeap>> = match heap {
            None => None,
            Some(heap) => Some(self.held_heap(heap)? as *const _),
        };
        let resource: &ProtocolObject<dyn MTLResource> = match of {
            QueriedResource::Buffer(handle) => ProtocolObject::from_ref(self.held(handle)?),
            QueriedResource::Texture(handle) => {
                ProtocolObject::from_ref(self.held_texture(handle)?)
            }
        };
        let answered = resource.heap();
        let answered_ptr = answered
            .as_ref()
            .map(|heap| &**heap as *const ProtocolObject<dyn MTLHeap>);
        Ok(ResourceFacts {
            label: resource.label().map(|label| label.to_string()),
            own_device: resource.device().registryID() == registry,
            storage_mode: resource.storageMode().0 as u64,
            cpu_cache_mode: resource.cpuCacheMode().0 as u64,
            hazard_tracking_mode: resource.hazardTrackingMode().0 as u64,
            resource_options: resource.resourceOptions().0 as u64,
            in_a_heap: answered_ptr.is_some(),
            expected_heap: answered_ptr.is_some() && answered_ptr == placed_in,
            heap_offset: resource.heapOffset() as u64,
            aliasable: resource.isAliasable(),
        })
    }

    fn set_resource_label(&mut self, of: QueriedResource, label: &str) -> Answer<()> {
        let resource: &ProtocolObject<dyn MTLResource> = match of {
            QueriedResource::Buffer(handle) => ProtocolObject::from_ref(self.held(handle)?),
            QueriedResource::Texture(handle) => {
                ProtocolObject::from_ref(self.held_texture(handle)?)
            }
        };
        resource.setLabel(Some(&NSString::from_str(label)));
        Ok(())
    }

    fn make_resource_aliasable(&mut self, of: QueriedResource) -> Answer<()> {
        let resource: &ProtocolObject<dyn MTLResource> = match of {
            QueriedResource::Buffer(handle) => ProtocolObject::from_ref(self.held(handle)?),
            QueriedResource::Texture(handle) => {
                ProtocolObject::from_ref(self.held_texture(handle)?)
            }
        };
        // SAFETY: `makeAliasable` takes no argument and reads nothing but the receiver. Its own
        // contract is that the resource's memory may be reused by another resource of the same
        // heap from now on — so a case asks it of a resource it reads getters of afterwards and
        // never of one it has bound, dispatched through or drawn with. No case of this suite does
        // either: the family that calls it encodes nothing at all.
        unsafe { resource.makeAliasable() };
        Ok(())
    }

    fn resource_purgeable_state(
        &mut self,
        of: QueriedResource,
        state: PurgeableState,
    ) -> Answer<u64> {
        let resource: &ProtocolObject<dyn MTLResource> = match of {
            QueriedResource::Buffer(handle) => ProtocolObject::from_ref(self.held(handle)?),
            QueriedResource::Texture(handle) => {
                ProtocolObject::from_ref(self.held_texture(handle)?)
            }
        };
        Ok(resource.setPurgeableState(metal_purgeable(state)).0 as u64)
    }

    fn texture_view(&mut self, of: Handle, request: &ViewRequest) -> Answer<Handle> {
        let texture = self.held_texture(of)?;
        let format = MTLPixelFormat(request.format.as_raw() as usize);
        let texture_type = MTLTextureType(request.texture_type.as_raw() as usize);
        let levels = NSRange::new(request.first_level as usize, request.level_count as usize);
        let slices = NSRange::new(request.first_slice as usize, request.slice_count as usize);
        let swizzle = metal_swizzle(request.swizzle);
        // SAFETY: every one of the five values is the case's own, and the ranges lie inside the
        // parent's own levels and slices with a format of the parent's size class -- which
        // `ViewRequest::is_well_formed` decides from the request's shape and the parent's, and
        // which the case's own guard holds every row of every table to before record mode runs.
        // A range outside the parent's, or a format of another size class, is what Metal's
        // validation stops, and the selectors read nothing but the values they are given.
        let view = unsafe {
            match request.call {
                ViewCall::PixelFormat => texture.newTextureViewWithPixelFormat(format),
                ViewCall::LevelsSlices => texture
                    .newTextureViewWithPixelFormat_textureType_levels_slices(
                        format,
                        texture_type,
                        levels,
                        slices,
                    ),
                ViewCall::Swizzled => texture
                    .newTextureViewWithPixelFormat_textureType_levels_slices_swizzle(
                        format,
                        texture_type,
                        levels,
                        slices,
                        swizzle,
                    ),
                ViewCall::Descriptor => {
                    let descriptor = MTLTextureViewDescriptor::new();
                    descriptor.setPixelFormat(format);
                    descriptor.setTextureType(texture_type);
                    descriptor.setLevelRange(levels);
                    descriptor.setSliceRange(slices);
                    descriptor.setSwizzle(swizzle);
                    texture.newTextureViewWithDescriptor(&descriptor)
                }
            }
        }
        .ok_or_else(|| {
            Failure::named(
                "Nil",
                format!(
                    "{} returned nil for {}",
                    request.call.selector(),
                    request.site()
                ),
            )
        })?;
        Ok(keep(&mut self.textures, view))
    }

    fn texture_view_facts(&mut self, texture: Handle, parent: Option<Handle>) -> Answer<ViewFacts> {
        let of_parent: Option<*const ProtocolObject<dyn MTLTexture>> = match parent {
            None => None,
            Some(parent) => Some(self.held_texture(parent)? as *const _),
        };
        let texture = self.held_texture(texture)?;
        let answered = texture.parentTexture();
        let answered_ptr = answered
            .as_ref()
            .map(|parent| &**parent as *const ProtocolObject<dyn MTLTexture>);
        Ok(ViewFacts {
            has_parent: answered_ptr.is_some(),
            expected_parent: answered_ptr.is_some() && answered_ptr == of_parent,
            parent_relative_level: texture.parentRelativeLevel() as u64,
            parent_relative_slice: texture.parentRelativeSlice() as u64,
            swizzle: raw_swizzle(texture.swizzle()),
            shareable: texture.isShareable(),
            texture_type: texture.textureType().0 as u64,
            pixel_format: texture.pixelFormat().0 as u64,
            mipmap_level_count: texture.mipmapLevelCount() as u64,
            array_length: texture.arrayLength() as u64,
        })
    }

    fn shared_texture(&mut self, query: &HeapTextureQuery) -> Answer<Handle> {
        let descriptor = queried_descriptor(query);
        let texture = self
            .device
            .newSharedTextureWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "Nil",
                    format!(
                        "newSharedTextureWithDescriptor: returned nil for {}",
                        query.site()
                    ),
                )
            })?;
        Ok(keep(&mut self.textures, texture))
    }

    fn shared_texture_handle(&mut self, texture: Handle) -> Answer<SharedHandleFacts> {
        let registry = self.device.registryID();
        let handle = self
            .held_texture(texture)?
            .newSharedTextureHandle()
            .ok_or_else(|| {
                Failure::named("Nil", "newSharedTextureHandle returned nil".to_owned())
            })?;
        let own_device = handle.device().registryID() == registry;
        let opened = self.device.newSharedTextureWithHandle(&handle);
        let facts = match &opened {
            None => SharedHandleFacts {
                own_device,
                opens: false,
                opened_shareable: false,
                opened_width: 0,
                opened_height: 0,
                opened_levels: 0,
            },
            Some(opened) => SharedHandleFacts {
                own_device,
                opens: true,
                opened_shareable: opened.isShareable(),
                opened_width: opened.width() as u64,
                opened_height: opened.height() as u64,
                opened_levels: opened.mipmapLevelCount() as u64,
            },
        };
        // Kept as the runtime side keeps it: a texture opened from a handle and released here
        // would be a texture over memory a later row could still be asked about.
        if let Some(opened) = opened {
            keep(&mut self.textures, opened);
        }
        Ok(facts)
    }

    fn sparse_mapping(&mut self, program: &SparseMappingProgram) -> Answer<SparseMapped> {
        // The venue's own tile size, against the grid the program's regions are stated in. A
        // disagreement stops the program **before anything is encoded**: a region in tiles means
        // nothing without the grid it was written for, and a region past a level's tiles is the
        // one shape in this family that could fault a device.
        for spec in &program.textures {
            let query = SparseTileQuery {
                texture_type: spec.texture_type,
                format: spec.format,
                samples: spec.samples,
                page: Some(program.heap.page),
            };
            let held = self.sparse_tile_size(&query)?;
            let stated = program.tile.tile;
            if [held.width, held.height, held.depth] != stated {
                return Err(Failure::named(
                    "TileSize",
                    format!(
                        "{} tiles {}x{}x{} where the program states {}x{}x{}",
                        spec.site(),
                        held.width,
                        held.height,
                        held.depth,
                        stated[0],
                        stated[1],
                        stated[2]
                    ),
                ));
            }
        }
        let heap_handle = self.sparse_heap(&program.heap)?;
        let mut textures = Vec::new();
        for spec in &program.textures {
            textures.push(self.heap_texture(heap_handle, spec)?);
        }
        let mut buffers = Vec::new();
        for bytes in &program.buffers {
            buffers.push(self.buffer(bytes)?);
        }
        let mut fences = Vec::new();
        for at in 0..program.fences {
            fences.push(
                self.device.newFence().ok_or_else(|| {
                    Failure::named("NewFence", format!("fence {at} returned nil"))
                })?,
            );
        }
        let mut tails = Vec::new();
        for texture in &textures {
            let facts = self.sparse_texture_facts(*texture)?;
            tails.push((facts.first_mipmap_in_tail, facts.tail_size_in_bytes));
        }
        // A scratch buffer for each fill and each read, in the order the steps name them, so a
        // read's bytes are the side's own and never one of the program's.
        let mut scratch: Vec<Handle> = Vec::new();
        for step in &program.steps {
            match step {
                MappingStep::FillRegion { region, value, .. } => {
                    let (_, bytes) = program.tile.bytes_of(region);
                    scratch.push(self.buffer(&vec![*value; bytes as usize])?);
                }
                MappingStep::ReadRegion { region, .. } => {
                    let (_, bytes) = program.tile.bytes_of(region);
                    scratch.push(self.buffer(&vec![READ_GUARD; bytes as usize])?);
                }
                _ => {}
            }
        }
        // The baseline: the heap with its textures made and nothing mapped.
        let mut used_size = vec![self.heap_facts(heap_handle)?.used_size];
        let mut refused: Vec<Option<Failure>> = Vec::with_capacity(program.steps.len());
        let mut statuses = Vec::new();
        let mut reads = Vec::new();
        let mut pending: Vec<(usize, usize)> = Vec::new();
        let mut scratch_at = 0usize;
        let mut command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>> = None;
        let mut open = OpenEncoder::None;
        for at in 0..program.steps.len() {
            let answered = self.sparse_step(
                program,
                at,
                &textures,
                &buffers,
                &fences,
                &scratch,
                &mut scratch_at,
                &mut pending,
                &mut command_buffer,
                &mut open,
            );
            refused.push(match answered {
                Ok(Commits::No) => None,
                Ok(Commits::Yes) => {
                    let status = match command_buffer.take() {
                        Some(buffer) => {
                            buffer.commit();
                            buffer.waitUntilCompleted();
                            Value::unit_variant(status_word(buffer.status()))
                        }
                        None => Value::Absent,
                    };
                    statuses.push(status);
                    used_size.push(self.heap_facts(heap_handle)?.used_size);
                    for &(slot, length) in &pending {
                        reads.push(self.read_buffer(scratch[slot], 0, length)?);
                    }
                    pending.clear();
                    None
                }
                Err(failure) => Some(failure),
            });
        }
        // A program that left an encoder open closes it and does not commit: its steps' answers
        // stand, and the case's own guard holds every program to ending on a commit.
        open.end();
        Ok(SparseMapped {
            refused,
            used_size,
            tails,
            statuses,
            reads,
        })
    }

    fn command_buffer_facts(&mut self) -> Answer<CommandBufferFacts> {
        let scratch = self.buffer(&[0u8; TIMED_BYTES])?;
        let target = Retained::from(self.held(scratch)?);
        let made = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        // Each getter is read through a binding of its own declared type, never as the scrutinee
        // of an expression: the coverage derivation types a call's receiver from the expression
        // before the dot, and a receiver it cannot type is attributed to no item at all.
        let held: Retained<ProtocolObject<dyn MTLCommandBuffer>> = made;
        let answered_device = held.device();
        let queues_device = self.queue.device();
        let device_is_the_queues = answered_device.registryID() == queues_device.registryID();
        // Two queues with distinct labels, a command buffer on each: whether each command
        // buffer's `commandQueue` answers **its own** queue's label. Neither API compares two
        // queues for identity publicly, so this is the strongest fact both legs compute alike.
        let mut queue_label_is_its_own = [false; 2];
        for (at, label) in QUEUE_LABELS.iter().enumerate() {
            let made = self.device.newCommandQueue().ok_or_else(|| {
                Failure::named("NewCommandQueue", format!("{label} returned nil"))
            })?;
            let labelled: Retained<ProtocolObject<dyn MTLCommandQueue>> = made;
            labelled.setLabel(Some(&NSString::from_str(label)));
            let buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
                labelled.commandBuffer().ok_or_else(|| {
                    Failure::named("CommandBuffer", format!("on {label} returned nil"))
                })?;
            let answered = buffer.commandQueue();
            queue_label_is_its_own[at] =
                answered.label().map(|held| held.to_string()).as_deref() == Some(*label);
        }
        let kernel_before = [held.kernelStartTime(), held.kernelEndTime()];
        let gpu_before = [held.GPUStartTime(), held.GPUEndTime()];
        // One blit of a few bytes of the side's own buffer: enough work that the GPU ran, and no
        // shader, draw or dispatch at all.
        let encoder = held
            .blitCommandEncoder()
            .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil".to_owned()))?;
        encoder.fillBuffer_range_value(&target, NSRange::new(0, TIMED_BYTES), 0xa5);
        encoder.endEncoding();
        held.commit();
        held.waitUntilCompleted();
        let status = Value::unit_variant(status_word(held.status()));
        let kernel_after = [held.kernelStartTime(), held.kernelEndTime()];
        let gpu_after = [held.GPUStartTime(), held.GPUEndTime()];

        // A command buffer of its own for the encoder, so the timings above are of a command
        // buffer that only blitted and this one only opens an encoder.
        let state_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>> = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        let opened = state_buffer.resourceStateCommandEncoder();
        let resource_state_encoder_opened = opened.is_some();
        if let Some(encoder) = opened {
            // Ended with nothing encoded into it: it maps nothing and takes no heap memory.
            encoder.endEncoding();
        }
        state_buffer.commit();
        state_buffer.waitUntilCompleted();
        let state_encoder_status = Value::unit_variant(status_word(state_buffer.status()));
        Ok(CommandBufferFacts::new(
            device_is_the_queues,
            queue_label_is_its_own,
            kernel_before,
            kernel_after,
            Ok((gpu_before, gpu_after)),
            resource_state_encoder_opened,
            status,
            state_encoder_status,
        ))
    }
}

/// Metal's `MTLHeapType` for a case's heap type, by its raw value, so the two spellings cannot
/// drift apart.
fn metal_heap_type(heap_type: metal2vulkan_runtime::HeapType) -> MTLHeapType {
    MTLHeapType(heap_type.raw() as isize)
}

/// Metal's `MTLPurgeableState` for a case's state, by its raw value.
fn metal_purgeable(state: PurgeableState) -> MTLPurgeableState {
    MTLPurgeableState(state.raw() as usize)
}

/// Metal's `MTLTextureSwizzleChannels` for a case's swizzle, each channel by its raw ordinal.
fn metal_swizzle(
    swizzle: metal2vulkan::metal::TextureSwizzleChannels,
) -> MTLTextureSwizzleChannels {
    let channel =
        |held: metal2vulkan::metal::TextureSwizzle| MTLTextureSwizzle(held.as_raw() as u8);
    MTLTextureSwizzleChannels {
        red: channel(swizzle.red),
        green: channel(swizzle.green),
        blue: channel(swizzle.blue),
        alpha: channel(swizzle.alpha),
    }
}

/// The four raw ordinals of a swizzle Metal answered, in Metal's order.
///
/// Raw and not the translator's own enumeration, for the reason
/// [`ViewFacts::swizzle`](crate::texture_views::ViewFacts::swizzle) gives: mapping into a
/// `#[non_exhaustive]` enumeration would coerce a channel Metal names and it does not.
fn raw_swizzle(swizzle: MTLTextureSwizzleChannels) -> [u32; 4] {
    [
        u32::from(swizzle.red.0),
        u32::from(swizzle.green.0),
        u32::from(swizzle.blue.0),
        u32::from(swizzle.alpha.0),
    ]
}

/// Metal's `MTLSparsePageSize` for a case's page, by its raw value, so the two spellings cannot
/// drift apart.
fn metal_page(page: SparsePageSize) -> MTLSparsePageSize {
    MTLSparsePageSize(page.raw() as isize)
}

/// Metal's descriptor for a case's sparse texture: every field the case names, by its raw ordinal.
///
/// The storage is the heap's (`Private`) and the usage is shader-read, with the render-target bit
/// where the case asks for it. Metal stops any other storage for a heap's texture even without its
/// validation layer, so neither is a field a case may set.
fn sparse_texture_descriptor(spec: &SparseTextureSpec) -> Retained<MTLTextureDescriptor> {
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setTextureType(MTLTextureType(spec.texture_type.as_raw() as usize));
    descriptor.setPixelFormat(MTLPixelFormat(spec.format.as_raw() as usize));
    // SAFETY: each extent and count is a case's own, and the setters read nothing but the value
    // they are given.
    unsafe {
        descriptor.setWidth(spec.size[0] as usize);
        descriptor.setHeight(spec.size[1] as usize);
        descriptor.setDepth(spec.size[2] as usize);
        descriptor.setMipmapLevelCount(spec.levels as usize);
        descriptor.setSampleCount(spec.samples as usize);
        descriptor.setArrayLength(spec.array_length as usize);
    }
    let usage = if spec.render_target {
        MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget
    } else {
        MTLTextureUsage::ShaderRead
    };
    descriptor.setUsage(usage);
    descriptor.setStorageMode(MTLStorageMode::Private);
    descriptor
}

/// A command buffer the slot probe holds, so the thread that frees a slot need not be the thread
/// that made it.
///
/// `objc2` gives `MTLCommandBuffer` no `Send`, because Metal documents a command buffer as unsafe
/// to *encode into* from more than one thread. The probe encodes into none: it makes each command
/// buffer on one thread and only ever releases it from another, and releasing an Objective-C object
/// off the thread that made it is sound, since the retain count is atomic.
pub struct Made(pub Retained<ProtocolObject<dyn MTLCommandBuffer>>);

// SAFETY: as the type's documentation states: the probe never encodes into a held command buffer,
// and the only operation that crosses threads is the release.
unsafe impl Send for Made {}

impl MetalSide {
    /// The plane behind the texture `target`.
    fn held_plane(&self, target: Handle) -> Answer<(&crate::metal_surface::Surface, usize)> {
        self.planes
            .iter()
            .find(|plane| plane.handle == target)
            .map(|plane| (&self.surfaces[plane.at], plane.length))
            .ok_or_else(|| {
                Failure::named(
                    "NotOverASurface",
                    format!("texture {} lies over no IOSurface", target.0),
                )
            })
    }

    /// A framebuffer-only texture of `spec`'s format and extent over one plane of an IOSurface this
    /// side makes for it, at `plane`'s rotation and stride.
    ///
    /// Three things here are not ordinary Metal, and each is guarded:
    ///
    /// - **The stride rule is a safety rule.** Metal asserts, *ending the process*, on an IOSurface
    ///   texture whose `bytesPerRow` is below the texture's own row
    ///   ([`crate::surface`]), so the plane is refused here before any descriptor reaches Metal.
    /// - **`setFramebufferOnly:` and `setRotation:` are Apple SPI.** Both are reached by selector
    ///   and only after `respondsToSelector:`, so a release that stops answering them gives a typed
    ///   refusal naming the selector instead of an unrecognised-selector crash.
    /// - **`newTextureWithDescriptor:iosurface:plane:` is public**, but its `IOSurfaceRef`
    ///   argument belongs to a framework this crate does not depend on, so it is sent by selector
    ///   with the raw pointer.
    ///
    /// A target as a texture over a buffer the case made:
    /// `newTextureWithDescriptor:offset:bytesPerRow:`.
    ///
    /// The stride is checked **before any object is made**, because a `bytesPerRow` below the
    /// target's own row ends Metal's process rather than answering nil
    /// ([`crate::render::OverBuffer::unsafe_stride`]). That is the same order the surface target
    /// takes for the same reason, and it is what lets a case that would stop the oracle be
    /// refused on a machine with no device at all.
    fn buffer_backed_target(
        &mut self,
        spec: &TargetSpec,
        over: &crate::render::OverBuffer,
    ) -> Answer<Handle> {
        if let Some(why) = over.unsafe_stride() {
            return Err(Failure::named("WouldEndMetalsProcess", why));
        }
        let buffer = self.held(over.buffer)?;
        // SAFETY: the descriptor constructor reads only the format and the size it is given.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat(spec.format.as_raw() as usize),
                spec.size[0] as usize,
                spec.size[1] as usize,
                false,
            )
        };
        let mut usage = MTLTextureUsage::RenderTarget;
        if spec.shader_read {
            usage |= MTLTextureUsage::ShaderRead;
        }
        if spec.shader_write {
            usage |= MTLTextureUsage::ShaderWrite;
        }
        descriptor.setUsage(usage);
        // A texture over a buffer holds the buffer's own bytes, so its storage is the buffer's.
        descriptor.setStorageMode(MTLStorageMode::Shared);
        let texture = buffer
            .newTextureWithDescriptor_offset_bytesPerRow(
                &descriptor,
                over.offset as usize,
                over.bytes_per_row as usize,
            )
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptorOffsetBytesPerRow",
                    format!(
                        "a {:?} {}x{} target over a buffer at offset {} and {} bytes a row \
                         returned nil",
                        spec.format, spec.size[0], spec.size[1], over.offset, over.bytes_per_row
                    ),
                )
            })?;
        Ok(keep(&mut self.textures, texture))
    }

    /// Metal answering nil is a refusal, which is safe and is what the suite reads a venue's gap
    /// out of.
    fn surface_target(
        &mut self,
        spec: &TargetSpec,
        plane: &crate::surface::PlaneSpec,
    ) -> Answer<Handle> {
        if plane.unsafe_stride(spec.size) {
            return Err(Failure::named(
                "WouldEndMetalsProcess",
                format!(
                    "a plane of {} bytes a row behind a {} texel row: Metal's descriptor \
                     validation asserts (\"IOSurface texture: bytesPerRow must be greater or equal \
                     to\"), which ends the process",
                    plane.row_pitch,
                    spec.size[0] * plane.texel
                ),
            ));
        }
        // SAFETY: the descriptor constructor reads only the format and the size it is given.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat(spec.format.as_raw() as usize),
                spec.size[0] as usize,
                spec.size[1] as usize,
                false,
            )
        };
        let mut usage = MTLTextureUsage::RenderTarget;
        if spec.shader_read {
            usage |= MTLTextureUsage::ShaderRead;
        }
        if spec.shader_write {
            usage |= MTLTextureUsage::ShaderWrite;
        }
        descriptor.setUsage(usage);
        descriptor.setStorageMode(MTLStorageMode::Shared);
        let framebuffer_only = objc2::sel!(setFramebufferOnly:);
        let rotation = objc2::sel!(setRotation:);
        for (selector, name) in [
            (
                framebuffer_only,
                "MTLTextureDescriptor setFramebufferOnly: (Apple SPI)",
            ),
            (rotation, "MTLTextureDescriptor setRotation: (Apple SPI)"),
        ] {
            require_selector(&*descriptor, selector, name)?;
        }
        // SAFETY: both selectors were just checked, and each takes one integral argument and
        // answers nothing, as the private setters do. Verified answered on an Apple M3 Max under
        // macOS 27.0.1, where the descriptor also reads the rotation back.
        unsafe {
            let _: () = objc2::msg_send![&*descriptor, setFramebufferOnly: true];
            let _: () = objc2::msg_send![&*descriptor, setRotation: plane.rotation as objc2_foundation::NSUInteger];
        }
        let surface = crate::metal_surface::Surface::single_plane(
            plane.plane,
            plane.texel,
            plane.row_pitch,
            plane.pad,
        )
        .map_err(|why| Failure::named("IOSurfaceCreate", why))?;
        // The layout was a request, not an instruction, so it is read back before a texture lies
        // over it: a surface whose rows are not where the case says they are would make every
        // recorded byte a comparison at the wrong offset.
        surface
            .is_as_asked(plane.plane, plane.row_pitch)
            .map_err(|why| Failure::named("SurfaceNotAsAsked", why))?;
        let texture = self.texture_over_surface(&descriptor, &surface, plane)?;
        let handle = keep(&mut self.textures, texture);
        let at = self.surfaces.len();
        self.surfaces.push(surface);
        self.planes.push(PlaneAt {
            handle,
            at,
            length: plane.length(),
        });
        Ok(handle)
    }

    /// `newTextureWithDescriptor:iosurface:plane:` of plane 0, by selector.
    ///
    /// The plane's bytes are already the pad byte: [`crate::metal_surface::Surface::single_plane`]
    /// sets them, so there is no window in which Metal could read a surface this side had not
    /// written.
    fn texture_over_surface(
        &self,
        descriptor: &MTLTextureDescriptor,
        surface: &crate::metal_surface::Surface,
        plane: &crate::surface::PlaneSpec,
    ) -> Answer<Retained<ProtocolObject<dyn MTLTexture>>> {
        let selector = objc2::sel!(newTextureWithDescriptor:iosurface:plane:);
        require_selector(
            &*self.device,
            selector,
            "MTLDevice newTextureWithDescriptor:iosurface:plane:",
        )?;
        // SAFETY: the selector was just checked; it takes the descriptor, an `IOSurfaceRef` and a
        // plane index, and answers a `+1` texture, which `Retained` takes ownership of. The stride
        // rule the caller checked is what keeps this call from asserting.
        let texture: Option<Retained<ProtocolObject<dyn MTLTexture>>> = unsafe {
            objc2::msg_send![
                &*self.device,
                newTextureWithDescriptor: descriptor,
                iosurface: surface.as_ptr(),
                plane: 0usize,
            ]
        };
        texture.ok_or_else(|| {
            Failure::named("NewTextureWithIoSurface", format!("{plane:?} returned nil"))
        })
    }
}

/// Check `object` answers `selector` before anything is sent to it, answering
/// [`Failure::unavailable`] naming the selector where it does not.
///
/// **Every Apple SPI this suite sends goes through here**: the two texture fills
/// ([`crate::metal_work`]), `MTLTextureDescriptor`'s `setFramebufferOnly:` and `setRotation:`, and
/// the `IOSurfaceRef` creator. One lookup rather than one per call site, so a release that stops
/// answering a private selector gives the same typed refusal naming it wherever it is reached,
/// instead of an unrecognised-selector crash in whichever module happened to forget the check.
pub(crate) fn require_selector(
    object: &(impl NSObjectProtocol + objc2::Message),
    selector: objc2::runtime::Sel,
    name: &'static str,
) -> Answer<()> {
    match object.respondsToSelector(selector) {
        true => Ok(()),
        false => Err(Failure::unavailable(name)),
    }
}

/// The guard every call that hands a queried descriptor to Metal runs first
/// ([`HeapTextureQuery::forbidden`]): the shapes Metal's own validation stops, which ends this
/// process rather than answering nil.
///
/// A case's own table is held to the same list with no device at all, so this is the second line,
/// for a row somebody adds to a recorded table by mistake.
fn would_end_metals_process(query: &HeapTextureQuery) -> Answer<()> {
    match query.forbidden() {
        None => Ok(()),
        Some(why) => Err(Failure::named("WouldEndMetalsProcess", why)),
    }
}

/// Metal's descriptor for a queried one: every field the case names, by its raw ordinal, so a
/// private format and a private usage bit reach Metal as readily as a public one.
fn queried_descriptor(query: &HeapTextureQuery) -> Retained<MTLTextureDescriptor> {
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setTextureType(MTLTextureType(query.texture_type.as_raw() as usize));
    descriptor.setPixelFormat(MTLPixelFormat(query.format.as_raw() as usize));
    // SAFETY: each extent, count and index is a case's own, and the setters read nothing but the
    // value they are given.
    unsafe {
        descriptor.setWidth(query.size[0] as usize);
        descriptor.setHeight(query.size[1] as usize);
        descriptor.setDepth(query.size[2] as usize);
        descriptor.setMipmapLevelCount(query.levels as usize);
        descriptor.setSampleCount(query.samples as usize);
        descriptor.setArrayLength(query.array_length as usize);
    }
    descriptor.setUsage(MTLTextureUsage(query.usage.as_raw() as usize));
    descriptor.setStorageMode(storage_mode(query.storage).0);
    descriptor.setAllowGPUOptimizedContents(query.gpu_optimized);
    descriptor
}

/// What a texture Metal laid over a buffer answers of itself: every getter the runtime's `Texture`
/// has too, so a difference names the field rather than the object.
fn buffer_texture_facts(
    texture: &ProtocolObject<dyn MTLTexture>,
) -> crate::buffer_texture::BufferTextureFacts {
    crate::buffer_texture::BufferTextureFacts {
        texture_type: texture.textureType().0 as u64,
        pixel_format: texture.pixelFormat().0 as u64,
        width: texture.width() as u64,
        height: texture.height() as u64,
        depth: texture.depth() as u64,
        mipmap_level_count: texture.mipmapLevelCount() as u64,
        array_length: texture.arrayLength() as u64,
        sample_count: texture.sampleCount() as u64,
        usage: texture.usage().0 as u64,
        storage_mode: texture.storageMode().0 as u64,
        cpu_cache_mode: texture.cpuCacheMode().0 as u64,
        hazard_tracking_mode: texture.hazardTrackingMode().0 as u64,
        gpu_optimized: texture.allowGPUOptimizedContents(),
    }
}

/// The case's enumerations reach Metal's by their raw ordinals, which are Metal's own.
fn compare_function(function: metal2vulkan::metal::CompareFunction) -> MTLCompareFunction {
    MTLCompareFunction(function.as_raw() as usize)
}

fn stencil_operation(operation: metal2vulkan::metal::StencilOperation) -> MTLStencilOperation {
    MTLStencilOperation(operation.as_raw() as usize)
}

/// One face's stencil state as Metal's descriptor.
fn stencil_descriptor(face: &StencilDescriptor) -> Retained<MTLStencilDescriptor> {
    let descriptor = MTLStencilDescriptor::new();
    descriptor.setStencilCompareFunction(compare_function(face.stencil_compare_function));
    descriptor.setStencilFailureOperation(stencil_operation(face.stencil_failure_operation));
    descriptor.setDepthFailureOperation(stencil_operation(face.depth_failure_operation));
    descriptor.setDepthStencilPassOperation(stencil_operation(face.depth_stencil_pass_operation));
    descriptor.setReadMask(face.read_mask);
    descriptor.setWriteMask(face.write_mask);
    descriptor
}

/// The storage mode and resource options a [`Storage`] names: the Metal side's one spelling of the
/// four modes, which [`crate::metal_work`] reads too, so a case's storage mode cannot mean one
/// thing to a queried descriptor and another to a program's buffer.
pub fn storage_mode(storage: Storage) -> (MTLStorageMode, MTLResourceOptions) {
    match storage {
        Storage::Shared => (
            MTLStorageMode::Shared,
            MTLResourceOptions::StorageModeShared,
        ),
        Storage::Managed => (
            MTLStorageMode::Managed,
            MTLResourceOptions::StorageModeManaged,
        ),
        Storage::Private => (
            MTLStorageMode::Private,
            MTLResourceOptions::StorageModePrivate,
        ),
        Storage::Memoryless => (
            MTLStorageMode::Memoryless,
            MTLResourceOptions::StorageModeMemoryless,
        ),
    }
}

/// The pixel format [`crate::case::TARGET_FORMAT`] names, as Metal spells it.
const TARGET_PIXEL_FORMAT: MTLPixelFormat = MTLPixelFormat::RGBA32Uint;

impl MetalSide {
    /// The descriptor of a mesh render pipeline of `library`'s `functions`.
    /// One pipeline descriptor through Metal's `completionHandler:` form and through that form's
    /// synchronous sibling ([`crate::pipeline_handlers`]).
    ///
    /// The descriptor is built **once** and handed to both calls, which is what makes the two
    /// answers comparable. The handler path is waited for on its own channel with a bound
    /// ([`HANDLER_WAIT`]) and a bound that runs out is recorded, never retried: no call here is
    /// `waitUntilCompleted`, which Metal has no bounded form of.
    fn both_paths(&mut self, library: Handle, form: &PipelineForm<'_>) -> Answer<BothPaths> {
        let registry = self.device.registryID();
        let (handled, synchronous) = match form {
            PipelineForm::Render {
                spec,
                options: None,
            } => {
                let descriptor = self.render_descriptor_from(library, spec)?;
                let (sender, answered) = std::sync::mpsc::channel();
                let handler = block2::RcBlock::new(
                    move |state: *mut ProtocolObject<dyn MTLRenderPipelineState>,
                          error: *mut NSError| {
                        let _ = sender.send(render_delivery(
                            state,
                            None,
                            error,
                            registry,
                            RenderKind::Drawing,
                        ));
                    },
                );
                // SAFETY: the block is a valid heap block held in `handler` until the wait below
                // has ended, and the descriptor is this call's own.
                unsafe {
                    self.device
                        .newRenderPipelineStateWithDescriptor_completionHandler(
                            &descriptor,
                            block2::RcBlock::as_ptr(&handler),
                        );
                }
                let handled = handler_answer(&answered);
                let synchronous = match self
                    .device
                    .newRenderPipelineStateWithDescriptor_error(&descriptor)
                {
                    Ok(state) => Delivered::made(StateFacts::Render(render_state_facts(
                        &state,
                        registry,
                        HANDLER_IMAGEBLOCK,
                    ))),
                    Err(_) => Delivered::refused(false),
                };
                (handled, synchronous)
            }
            PipelineForm::Render {
                spec,
                options: Some(bits),
            } => {
                let descriptor = self.render_descriptor_from(library, spec)?;
                let options = MTLPipelineOption(*bits as usize);
                let (sender, answered) = std::sync::mpsc::channel();
                let handler = block2::RcBlock::new(
                    move |state: *mut ProtocolObject<dyn MTLRenderPipelineState>,
                          reflection: *mut MTLRenderPipelineReflection,
                          error: *mut NSError| {
                        let _ = sender.send(render_delivery(
                            state,
                            Some(!reflection.is_null()),
                            error,
                            registry,
                            RenderKind::Drawing,
                        ));
                    },
                );
                // SAFETY: as the option-less form's.
                unsafe {
                    self.device
                        .newRenderPipelineStateWithDescriptor_options_completionHandler(
                            &descriptor,
                            options,
                            block2::RcBlock::as_ptr(&handler),
                        );
                }
                let handled = handler_answer(&answered);
                let mut reflection = None;
                let synchronous = match self
                    .device
                    .newRenderPipelineStateWithDescriptor_options_reflection_error(
                        &descriptor,
                        options,
                        Some(&mut reflection),
                    ) {
                    Ok(state) => Delivered::reflected(
                        StateFacts::Render(render_state_facts(
                            &state,
                            registry,
                            HANDLER_IMAGEBLOCK,
                        )),
                        reflection.is_some(),
                    ),
                    Err(_) => Delivered::refused(true),
                };
                (handled, synchronous)
            }
            PipelineForm::ComputeFunction {
                entry,
                options: None,
            } => {
                let function = self.function_of(library, entry)?;
                let (sender, answered) = std::sync::mpsc::channel();
                let handler = block2::RcBlock::new(
                    move |state: *mut ProtocolObject<dyn MTLComputePipelineState>,
                          error: *mut NSError| {
                        let _ = sender.send(compute_delivery(state, None, error, registry));
                    },
                );
                // SAFETY: the function is of this device's library and the block is held until the
                // wait below has ended.
                unsafe {
                    self.device
                        .newComputePipelineStateWithFunction_completionHandler(
                            &function,
                            block2::RcBlock::as_ptr(&handler),
                        );
                }
                let handled = handler_answer(&answered);
                let synchronous = match self
                    .device
                    .newComputePipelineStateWithFunction_error(&function)
                {
                    Ok(state) => Delivered::made(StateFacts::Compute(compute_state_facts(
                        &state,
                        registry,
                        HANDLER_IMAGEBLOCK,
                    ))),
                    Err(_) => Delivered::refused(false),
                };
                (handled, synchronous)
            }
            PipelineForm::ComputeFunction {
                entry,
                options: Some(bits),
            } => {
                let function = self.function_of(library, entry)?;
                let options = MTLPipelineOption(*bits as usize);
                let (sender, answered) = std::sync::mpsc::channel();
                let handler = block2::RcBlock::new(
                    move |state: *mut ProtocolObject<dyn MTLComputePipelineState>,
                          reflection: *mut MTLComputePipelineReflection,
                          error: *mut NSError| {
                        let _ = sender.send(compute_delivery(
                            state,
                            Some(!reflection.is_null()),
                            error,
                            registry,
                        ));
                    },
                );
                // SAFETY: as the option-less form's.
                unsafe {
                    self.device
                        .newComputePipelineStateWithFunction_options_completionHandler(
                            &function,
                            options,
                            block2::RcBlock::as_ptr(&handler),
                        );
                }
                let handled = handler_answer(&answered);
                let mut reflection = None;
                // SAFETY: the function is of this device's library and `reflection` is this
                // call's own out-parameter.
                let made = unsafe {
                    self.device
                        .newComputePipelineStateWithFunction_options_reflection_error(
                            &function,
                            options,
                            Some(&mut reflection),
                        )
                };
                let synchronous = match made {
                    Ok(state) => Delivered::reflected(
                        StateFacts::Compute(compute_state_facts(
                            &state,
                            registry,
                            HANDLER_IMAGEBLOCK,
                        )),
                        reflection.is_some(),
                    ),
                    Err(_) => Delivered::refused(true),
                };
                (handled, synchronous)
            }
            PipelineForm::ComputeDescriptor { spec, options } => {
                let descriptor = self.compute_descriptor_from(library, spec)?;
                let options = MTLPipelineOption(*options as usize);
                let (sender, answered) = std::sync::mpsc::channel();
                let handler = block2::RcBlock::new(
                    move |state: *mut ProtocolObject<dyn MTLComputePipelineState>,
                          reflection: *mut MTLComputePipelineReflection,
                          error: *mut NSError| {
                        let _ = sender.send(compute_delivery(
                            state,
                            Some(!reflection.is_null()),
                            error,
                            registry,
                        ));
                    },
                );
                // SAFETY: as the function forms'.
                unsafe {
                    self.device
                        .newComputePipelineStateWithDescriptor_options_completionHandler(
                            &descriptor,
                            options,
                            block2::RcBlock::as_ptr(&handler),
                        );
                }
                let handled = handler_answer(&answered);
                let mut reflection = None;
                let synchronous = match self
                    .device
                    .newComputePipelineStateWithDescriptor_options_reflection_error(
                        &descriptor,
                        options,
                        Some(&mut reflection),
                    ) {
                    Ok(state) => Delivered::reflected(
                        StateFacts::Compute(compute_state_facts(
                            &state,
                            registry,
                            HANDLER_IMAGEBLOCK,
                        )),
                        reflection.is_some(),
                    ),
                    Err(_) => Delivered::refused(true),
                };
                (handled, synchronous)
            }
            PipelineForm::Tile { spec, options } => {
                let descriptor = self.tile_descriptor_from(library, spec)?;
                let options = MTLPipelineOption(*options as usize);
                let (sender, answered) = std::sync::mpsc::channel();
                let handler = block2::RcBlock::new(
                    move |state: *mut ProtocolObject<dyn MTLRenderPipelineState>,
                          reflection: *mut MTLRenderPipelineReflection,
                          error: *mut NSError| {
                        let _ = sender.send(render_delivery(
                            state,
                            Some(!reflection.is_null()),
                            error,
                            registry,
                            RenderKind::Tile,
                        ));
                    },
                );
                // SAFETY: as the drawing forms'.
                unsafe {
                    self.device
                        .newRenderPipelineStateWithTileDescriptor_options_completionHandler(
                            &descriptor,
                            options,
                            block2::RcBlock::as_ptr(&handler),
                        );
                }
                let handled = handler_answer(&answered);
                let mut reflection = None;
                let synchronous = match self
                    .device
                    .newRenderPipelineStateWithTileDescriptor_options_reflection_error(
                        &descriptor,
                        options,
                        Some(&mut reflection),
                    ) {
                    Ok(state) => Delivered::reflected(
                        StateFacts::Tile(tile_state_facts(&state)),
                        reflection.is_some(),
                    ),
                    Err(_) => Delivered::refused(true),
                };
                (handled, synchronous)
            }
            PipelineForm::Mesh { functions, options } => {
                let descriptor = self.mesh_descriptor(library, functions)?;
                let options = MTLPipelineOption(*options as usize);
                let (sender, answered) = std::sync::mpsc::channel();
                let handler = block2::RcBlock::new(
                    move |state: *mut ProtocolObject<dyn MTLRenderPipelineState>,
                          reflection: *mut MTLRenderPipelineReflection,
                          error: *mut NSError| {
                        let _ = sender.send(render_delivery(
                            state,
                            Some(!reflection.is_null()),
                            error,
                            registry,
                            RenderKind::Drawing,
                        ));
                    },
                );
                // SAFETY: as the drawing forms'.
                unsafe {
                    self.device
                        .newRenderPipelineStateWithMeshDescriptor_options_completionHandler(
                            &descriptor,
                            options,
                            block2::RcBlock::as_ptr(&handler),
                        );
                }
                let handled = handler_answer(&answered);
                let mut reflection = None;
                let synchronous = match self
                    .device
                    .newRenderPipelineStateWithMeshDescriptor_options_reflection_error(
                        &descriptor,
                        options,
                        Some(&mut reflection),
                    ) {
                    Ok(state) => Delivered::reflected(
                        StateFacts::Render(render_state_facts(
                            &state,
                            registry,
                            HANDLER_IMAGEBLOCK,
                        )),
                        reflection.is_some(),
                    ),
                    Err(_) => Delivered::refused(true),
                };
                (handled, synchronous)
            }
        };
        Ok(BothPaths::new(handled, synchronous))
    }

    /// `newFunctionWithName:` of `entry` in the library `library` names.
    fn function_of(
        &self,
        library: Handle,
        entry: &str,
    ) -> Answer<Retained<ProtocolObject<dyn MTLFunction>>> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", entry.to_owned()))
    }

    /// The `MTLComputePipelineDescriptor` `spec` describes, of `library`'s kernel.
    ///
    /// Its own method rather than part of a creation, because the same descriptor is handed to
    /// two creations: a case of [`crate::pipeline_handlers`] takes one descriptor through the
    /// `completionHandler:` call and through its synchronous sibling, and the two paths are only
    /// comparable where the value they are given is the same.
    fn compute_descriptor_from(
        &self,
        library: Handle,
        spec: &ComputeStateSpec<'_>,
    ) -> Answer<Retained<MTLComputePipelineDescriptor>> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(spec.entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", spec.entry.to_owned()))?;
        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&function));
        // Set only where the case states one. `setLabel:nil` on a pipeline descriptor fails an
        // assertion inside Metal and ends the process ("label must not be nil", measured on this
        // M3 Max while recording this family), and Metal's own default is already nil, so a case
        // that means a nil label is a case that makes this call not at all.
        if let Some(label) = spec.label {
            descriptor.setLabel(Some(&NSString::from_str(label)));
        }
        descriptor.setSupportIndirectCommandBuffers(spec.support_indirect_command_buffers);
        if let Some(threads) = spec.required_threads_per_threadgroup {
            descriptor.setRequiredThreadsPerThreadgroup(size(threads));
        }
        Ok(descriptor)
    }

    /// The `MTLRenderPipelineDescriptor` `spec` describes, of `library`'s stage functions.
    ///
    /// Its own method for the reason [`MetalSide::compute_descriptor_from`]'s is.
    fn render_descriptor_from(
        &self,
        library: Handle,
        spec: &RenderPipelineSpec<'_>,
    ) -> Answer<Retained<MTLRenderPipelineDescriptor>> {
        let function = |stage: &StageFunction<'_>| {
            let named = stage.library.unwrap_or(library);
            let held = self
                .libraries
                .get(named.0 as usize)
                .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", named.0)))?;
            held.newFunctionWithName(&NSString::from_str(stage.entry))
                .ok_or_else(|| Failure::named("NoSuchFunction", stage.entry.to_owned()))
        };
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&*function(&spec.vertex)?));
        if let Some(fragment) = &spec.fragment {
            descriptor.setFragmentFunction(Some(&*function(fragment)?));
        }
        // Set only where the case states one: `setLabel:nil` ends Metal's process, as the
        // compute descriptor's does.
        if let Some(label) = spec.label {
            descriptor.setLabel(Some(&NSString::from_str(label)));
        }
        descriptor.setSupportIndirectCommandBuffers(spec.support_indirect_command_buffers);
        descriptor.setRasterSampleCount(spec.samples as usize);
        descriptor.setRasterizationEnabled(spec.rasterization);
        descriptor.setAlphaToCoverageEnabled(spec.alpha_to_coverage);
        descriptor.setAlphaToOneEnabled(spec.alpha_to_one);
        // SAFETY: the value is an enumerator of the SDK's own `MTLPrimitiveTopologyClass` table,
        // and the descriptor is this call's own. The setter takes a raw `NSUInteger`, which is why
        // `objc2` marks it unsafe.
        unsafe {
            descriptor.setInputPrimitiveTopology(MTLPrimitiveTopologyClass(
                spec.input_primitive_topology.as_raw() as usize,
            ));
        }
        descriptor.setSupportIndirectCommandBuffers(spec.support_indirect_command_buffers);
        // Every tessellation field, and only where the case states them: a pipeline that names no
        // tessellation state leaves Metal's own defaults alone, which is what every other family's
        // pipeline has always been made with.
        if let Some(state) = &spec.tessellation {
            // SAFETY: each value is an enumerator of the SDK's own table or, for the factor, a
            // count the case keeps inside `MTLDevice`'s documented 1..=64, and the descriptor is
            // this call's own. The three setters `objc2` marks unsafe take a raw `NSUInteger`.
            unsafe {
                descriptor.setMaxTessellationFactor(state.max_factor as usize);
                descriptor.setTessellationPartitionMode(MTLTessellationPartitionMode(
                    state.partition.as_raw() as usize,
                ));
                descriptor.setTessellationControlPointIndexType(
                    MTLTessellationControlPointIndexType(
                        state.control_point_index_type.as_raw() as usize
                    ),
                );
            }
            descriptor.setTessellationFactorFormat(MTLTessellationFactorFormat(
                state.factor_format.as_raw() as usize,
            ));
            descriptor.setTessellationFactorStepFunction(MTLTessellationFactorStepFunction(
                state.factor_step.as_raw() as usize,
            ));
            descriptor
                .setTessellationOutputWindingOrder(MTLWinding(state.winding.as_raw() as usize));
            descriptor.setTessellationFactorScaleEnabled(state.factor_scale_enabled);
        }
        if let Some(layout) = &spec.vertex_layout {
            descriptor.setVertexDescriptor(Some(&vertex_descriptor(layout)));
        }
        if let Some(format) = spec.depth_format {
            descriptor.setDepthAttachmentPixelFormat(MTLPixelFormat(format.as_raw() as usize));
        }
        if let Some(format) = spec.stencil_format {
            descriptor.setStencilAttachmentPixelFormat(MTLPixelFormat(format.as_raw() as usize));
        }
        for (at, color) in spec.colors.iter().enumerate() {
            let Some(color) = color else { continue };
            // SAFETY: `at` is below the eight color attachments every Metal pipeline holds, which
            // the case's own guard keeps it to.
            let attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(at) };
            attachment.setPixelFormat(MTLPixelFormat(color.format.as_raw() as usize));
            attachment.setWriteMask(MTLColorWriteMask(color.write_mask.as_raw() as usize));
            if let Some(blend) = &color.blend {
                let factor = |factor: metal2vulkan::metal::BlendFactor| {
                    MTLBlendFactor(factor.as_raw() as usize)
                };
                let operation = |operation: metal2vulkan::metal::BlendOperation| {
                    MTLBlendOperation(operation.as_raw() as usize)
                };
                attachment.setBlendingEnabled(true);
                attachment.setSourceRGBBlendFactor(factor(blend.source_rgb));
                attachment.setDestinationRGBBlendFactor(factor(blend.destination_rgb));
                attachment.setRgbBlendOperation(operation(blend.rgb));
                attachment.setSourceAlphaBlendFactor(factor(blend.source_alpha));
                attachment.setDestinationAlphaBlendFactor(factor(blend.destination_alpha));
                attachment.setAlphaBlendOperation(operation(blend.alpha));
            }
        }
        Ok(descriptor)
    }

    /// The `MTLTileRenderPipelineDescriptor` `spec` describes, of `library`'s tile function.
    ///
    /// Its own method for the reason [`MetalSide::compute_descriptor_from`]'s is.
    fn tile_descriptor_from(
        &self,
        library: Handle,
        spec: &TilePipelineSpec<'_>,
    ) -> Answer<Retained<MTLTileRenderPipelineDescriptor>> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(spec.entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", spec.entry.to_owned()))?;
        let descriptor = MTLTileRenderPipelineDescriptor::new();
        // SAFETY: the function is of this library, and the descriptor is this call's own.
        unsafe { descriptor.setTileFunction(&function) };
        for (at, format) in spec.attachments.iter().enumerate() {
            // SAFETY: `at` is below the eight attachments the descriptor's array holds.
            unsafe {
                descriptor
                    .colorAttachments()
                    .objectAtIndexedSubscript(at)
                    .setPixelFormat(MTLPixelFormat(format.as_raw() as usize));
            }
        }
        if let Some(limit) = spec.max_total_threads_per_threadgroup {
            descriptor.setMaxTotalThreadsPerThreadgroup(limit as usize);
        }
        descriptor.setThreadgroupSizeMatchesTileSize(spec.threadgroup_size_matches_tile_size);
        Ok(descriptor)
    }

    fn mesh_descriptor(
        &self,
        library: Handle,
        functions: &MeshFunctions<'_>,
    ) -> Answer<Retained<MTLMeshRenderPipelineDescriptor>> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = |name: &str| {
            library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| Failure::named("NoSuchFunction", name.to_owned()))
        };
        let descriptor = MTLMeshRenderPipelineDescriptor::new();
        // SAFETY: each function is of this library, and the descriptor is this call's own.
        unsafe {
            descriptor.setMeshFunction(Some(&*function(functions.mesh)?));
            if let Some(object) = functions.object {
                descriptor.setObjectFunction(Some(&*function(object)?));
            }
            descriptor.setFragmentFunction(Some(&*function(functions.fragment)?));
            let attachments = descriptor.colorAttachments();
            for at in 0..functions.attachments as usize {
                attachments
                    .objectAtIndexedSubscript(at)
                    .setPixelFormat(TARGET_PIXEL_FORMAT);
            }
        }
        Ok(descriptor)
    }

    /// The mesh descriptor `spec` describes, whose attachments are the formats the case names.
    fn mesh_reflection_descriptor(
        &self,
        library: Handle,
        spec: &MeshReflectionSpec<'_>,
    ) -> Answer<Retained<MTLMeshRenderPipelineDescriptor>> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = |name: &str| {
            library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| Failure::named("NoSuchFunction", name.to_owned()))
        };
        let descriptor = MTLMeshRenderPipelineDescriptor::new();
        // SAFETY: each function is of this library, and the descriptor is this call's own.
        unsafe {
            descriptor.setMeshFunction(Some(&*function(spec.mesh)?));
            if let Some(object) = spec.object {
                descriptor.setObjectFunction(Some(&*function(object)?));
            }
            descriptor.setFragmentFunction(Some(&*function(spec.fragment)?));
            let attachments = descriptor.colorAttachments();
            for (at, format) in spec.attachments.iter().enumerate() {
                // SAFETY: `at` is below the eight attachments the descriptor's array holds.
                attachments
                    .objectAtIndexedSubscript(at)
                    .setPixelFormat(MTLPixelFormat(format.as_raw() as usize));
            }
        }
        Ok(descriptor)
    }

    /// The state `descriptor` makes, kept by handle.
    fn made_mesh_pipeline(
        &mut self,
        descriptor: &MTLMeshRenderPipelineDescriptor,
    ) -> Answer<Handle> {
        let state = self
            .device
            .newRenderPipelineStateWithMeshDescriptor_options_reflection_error(
                descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|error| Failure::named("NewMeshRenderPipelineState", error.to_string()))?;
        Ok(keep(
            &mut self.render_pipelines,
            RenderPipeline::unreflected(state),
        ))
    }

    /// One command buffer and one render encoder over `targets`, each attachment cleared to zero
    /// and stored.
    #[allow(clippy::type_complexity)]
    fn pass(
        &self,
        targets: &[Handle],
    ) -> Answer<(
        Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>,
    )> {
        let attachments: Vec<Retained<ProtocolObject<dyn MTLTexture>>> = targets
            .iter()
            .map(|handle| self.held_texture(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        for (at, target) in attachments.iter().enumerate() {
            // SAFETY: `at` is below the eight attachments every Metal pass holds.
            let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(at) };
            attachment.setTexture(Some(target));
            attachment.setLoadAction(MTLLoadAction::Clear);
            attachment.setClearColor(MTLClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            });
            attachment.setStoreAction(MTLStoreAction::Store);
        }
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or_else(|| Failure::named("RenderCommandEncoder", "returned nil".to_owned()))?;
        Ok((command_buffer, encoder))
    }

    /// One command buffer and one render encoder over `targets` in `tile`-sized tiles, each
    /// attachment loaded and stored.
    #[allow(clippy::type_complexity)]
    fn loading_pass(
        &self,
        targets: &[Handle],
        tile: [u32; 2],
    ) -> Answer<(
        Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>,
    )> {
        let attachments: Vec<Retained<ProtocolObject<dyn MTLTexture>>> = targets
            .iter()
            .map(|handle| self.held_texture(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        for (at, target) in attachments.iter().enumerate() {
            // SAFETY: `at` is below the eight attachments every Metal pass holds.
            let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(at) };
            attachment.setTexture(Some(target));
            attachment.setLoadAction(MTLLoadAction::Load);
            attachment.setStoreAction(MTLStoreAction::Store);
        }
        pass.setTileWidth(tile[0] as usize);
        pass.setTileHeight(tile[1] as usize);
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or_else(|| Failure::named("RenderCommandEncoder", "returned nil".to_owned()))?;
        Ok((command_buffer, encoder))
    }

    /// Encode one of a case's tile-pass steps into `encoder`.
    fn encode_tile_step(
        &self,
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        step: &TileStep,
    ) -> Answer<()> {
        match step {
            TileStep::Pipeline(state) => {
                let state = &self
                    .render_pipelines
                    .get(state.0 as usize)
                    .ok_or_else(|| {
                        Failure::named("NoSuchHandle", format!("tile pipeline {}", state.0))
                    })?
                    .state;
                encoder.setRenderPipelineState(state);
            }
            TileStep::TileBuffer {
                buffer,
                offset,
                index,
            } => {
                let buffer = self.held(*buffer)?;
                // SAFETY: the buffer is at least as long as the function's binding, and the
                // encoder outlives the call.
                unsafe {
                    encoder.setTileBuffer_offset_atIndex(
                        Some(buffer),
                        *offset as usize,
                        *index as usize,
                    )
                };
            }
            TileStep::TileBytes { bytes, index } => {
                let pointer =
                    std::ptr::NonNull::new(bytes.as_ptr().cast::<std::ffi::c_void>().cast_mut())
                        .ok_or_else(|| Failure::named("NoBytes", "an empty inline binding"))?;
                // SAFETY: `bytes` lives for the call, which copies them.
                unsafe {
                    encoder.setTileBytes_length_atIndex(pointer, bytes.len(), *index as usize)
                };
            }
            TileStep::TileTexture { texture, index } => {
                let texture = self.held_texture(*texture)?;
                // SAFETY: the texture is kept alive by `self.textures`, and the encoder outlives
                // the call.
                unsafe { encoder.setTileTexture_atIndex(Some(texture), *index as usize) };
            }
            TileStep::DispatchThreadsPerTile(threads) => {
                encoder.dispatchThreadsPerTile(MTLSize {
                    width: threads[0] as usize,
                    height: threads[1] as usize,
                    depth: threads[2] as usize,
                });
            }
        }
        Ok(())
    }

    /// Encode one of a case's render-pass steps into `encoder`.
    fn encode_step(
        &self,
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        step: &MeshStep,
    ) -> Answer<()> {
        let size = |axes: [u32; 3]| MTLSize {
            width: axes[0] as usize,
            height: axes[1] as usize,
            depth: axes[2] as usize,
        };
        match step {
            MeshStep::Pipeline(state) => {
                let state = &self
                    .render_pipelines
                    .get(state.0 as usize)
                    .ok_or_else(|| {
                        Failure::named("NoSuchHandle", format!("mesh pipeline {}", state.0))
                    })?
                    .state;
                encoder.setRenderPipelineState(state);
            }
            MeshStep::ObjectBuffer {
                buffer,
                offset,
                index,
            } => {
                let buffer = self.held(*buffer)?;
                // SAFETY: the buffer is at least as long as the function's binding, and the
                // encoder outlives the call.
                unsafe {
                    encoder.setObjectBuffer_offset_atIndex(
                        Some(buffer),
                        *offset as usize,
                        *index as usize,
                    )
                };
            }
            MeshStep::ObjectBytes { bytes, index } => {
                let pointer =
                    std::ptr::NonNull::new(bytes.as_ptr().cast::<std::ffi::c_void>().cast_mut())
                        .ok_or_else(|| Failure::named("NoBytes", "an empty inline binding"))?;
                // SAFETY: `bytes` lives for the call, which copies them.
                unsafe {
                    encoder.setObjectBytes_length_atIndex(pointer, bytes.len(), *index as usize)
                };
            }
            MeshStep::MeshBuffer {
                buffer,
                offset,
                index,
            } => {
                let buffer = self.held(*buffer)?;
                // SAFETY: as the object buffer's.
                unsafe {
                    encoder.setMeshBuffer_offset_atIndex(
                        Some(buffer),
                        *offset as usize,
                        *index as usize,
                    )
                };
            }
            MeshStep::FragmentBuffer {
                buffer,
                offset,
                index,
            } => {
                let buffer = self.held(*buffer)?;
                // SAFETY: as the object buffer's.
                unsafe {
                    encoder.setFragmentBuffer_offset_atIndex(
                        Some(buffer),
                        *offset as usize,
                        *index as usize,
                    )
                };
            }
            MeshStep::DrawThreadgroups {
                groups,
                object,
                mesh,
            } => encoder
                .drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                    size(*groups),
                    size(*object),
                    size(*mesh),
                ),
            MeshStep::DrawThreads {
                threads,
                object,
                mesh,
            } => encoder.drawMeshThreads_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                size(*threads),
                size(*object),
                size(*mesh),
            ),
            MeshStep::DrawIndirect {
                buffer,
                offset,
                object,
                mesh,
            } => {
                let buffer = self.held(*buffer)?;
                // SAFETY: the buffer holds three 32-bit counts at the offset, which the case
                // wrote before the pass.
                unsafe {
                    encoder.drawMeshThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                        buffer,
                        *offset as usize,
                        size(*object),
                        size(*mesh),
                    );
                }
            }
            MeshStep::BarrierScope {
                scope,
                after,
                before,
            } => encoder.memoryBarrierWithScope_afterStages_beforeStages(
                MTLBarrierScope(*scope as usize),
                MTLRenderStages(*after as usize),
                MTLRenderStages(*before as usize),
            ),
            MeshStep::BarrierResources {
                buffers,
                after,
                before,
            } => {
                let over: Vec<&ProtocolObject<dyn MTLResource>> = buffers
                    .iter()
                    .map(|handle| self.held(*handle).map(ProtocolObject::from_ref))
                    .collect::<Answer<Vec<_>>>()?;
                let mut pointers: Vec<std::ptr::NonNull<ProtocolObject<dyn MTLResource>>> =
                    over.iter().map(|resource| (*resource).into()).collect();
                let first = std::ptr::NonNull::new(pointers.as_mut_ptr())
                    .ok_or_else(|| Failure::named("NoResources", "an empty barrier"))?;
                // SAFETY: `pointers` holds exactly `pointers.len()` resource pointers, each
                // retained by `over` for the whole call.
                unsafe {
                    encoder.memoryBarrierWithResources_count_afterStages_beforeStages(
                        first,
                        pointers.len(),
                        MTLRenderStages(*after as usize),
                        MTLRenderStages(*before as usize),
                    );
                }
            }
        }
        Ok(())
    }

    /// The samples of the pass's own attachments, which is the count Metal holds its custom
    /// sample positions to: the first attachment the pass states, and 1 for a pass with none.
    fn pass_samples(&self, pass: &RenderPass) -> Answer<u32> {
        let attached = pass
            .colors
            .iter()
            .filter_map(|color| color.texture)
            .chain(pass.depth.as_ref().map(|depth| depth.texture))
            .chain(pass.stencil.as_ref().map(|stencil| stencil.texture))
            .next();
        match attached {
            Some(handle) => Ok(self.held_texture(handle)?.sampleCount() as u32),
            None => Ok(1),
        }
    }

    /// Encode one render pass of a program into `command_buffer`.
    fn encode_pass(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        pass: &RenderPass,
    ) -> Answer<()> {
        let descriptor = MTLRenderPassDescriptor::renderPassDescriptor();
        for (at, color) in pass.colors.iter().enumerate() {
            let Some(texture) = color.texture else {
                continue;
            };
            let texture: Retained<ProtocolObject<dyn MTLTexture>> =
                Retained::from(self.held_texture(texture)?);
            // SAFETY: `at` is below the eight color attachments every Metal pass holds, which the
            // case's own guard keeps it to.
            let attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(at) };
            attachment.setTexture(Some(&texture));
            attachment.setLevel(color.place.level as usize);
            attachment.setSlice(color.place.slice as usize);
            attachment.setDepthPlane(color.place.depth_plane as usize);
            attachment.setResolveLevel(color.place.resolve_level as usize);
            attachment.setResolveSlice(color.place.resolve_slice as usize);
            attachment.setResolveDepthPlane(color.place.resolve_depth_plane as usize);
            attachment.setLoadAction(MTLLoadAction(color.load.raw() as usize));
            attachment.setStoreAction(MTLStoreAction(color.store.raw() as usize));
            attachment.setClearColor(MTLClearColor {
                red: color.clear[0],
                green: color.clear[1],
                blue: color.clear[2],
                alpha: color.clear[3],
            });
            attachment.setStoreActionOptions(MTLStoreActionOptions::from_bits_retain(
                color.options.raw() as usize,
            ));
            if let Some(resolve) = color.resolve {
                let resolve: Retained<ProtocolObject<dyn MTLTexture>> =
                    Retained::from(self.held_texture(resolve)?);
                attachment.setResolveTexture(Some(&resolve));
            }
        }
        if let Some(depth) = &pass.depth {
            let texture: Retained<ProtocolObject<dyn MTLTexture>> =
                Retained::from(self.held_texture(depth.texture)?);
            let attachment = descriptor.depthAttachment();
            attachment.setTexture(Some(&texture));
            attachment.setLevel(depth.place.level as usize);
            attachment.setSlice(depth.place.slice as usize);
            attachment.setDepthPlane(depth.place.depth_plane as usize);
            attachment.setResolveLevel(depth.place.resolve_level as usize);
            attachment.setResolveSlice(depth.place.resolve_slice as usize);
            attachment.setResolveDepthPlane(depth.place.resolve_depth_plane as usize);
            attachment.setLoadAction(MTLLoadAction(depth.load.raw() as usize));
            attachment.setStoreAction(MTLStoreAction(depth.store.raw() as usize));
            attachment.setClearDepth(depth.clear);
            attachment.setStoreActionOptions(MTLStoreActionOptions::from_bits_retain(
                depth.options.raw() as usize,
            ));
            attachment.setDepthResolveFilter(MTLMultisampleDepthResolveFilter(
                depth.filter.raw() as usize
            ));
            if let Some(resolve) = depth.resolve {
                let resolve: Retained<ProtocolObject<dyn MTLTexture>> =
                    Retained::from(self.held_texture(resolve)?);
                attachment.setResolveTexture(Some(&resolve));
            }
        }
        if let Some(stencil) = &pass.stencil {
            let texture: Retained<ProtocolObject<dyn MTLTexture>> =
                Retained::from(self.held_texture(stencil.texture)?);
            let attachment = descriptor.stencilAttachment();
            attachment.setTexture(Some(&texture));
            attachment.setLevel(stencil.place.level as usize);
            attachment.setSlice(stencil.place.slice as usize);
            attachment.setDepthPlane(stencil.place.depth_plane as usize);
            attachment.setResolveLevel(stencil.place.resolve_level as usize);
            attachment.setResolveSlice(stencil.place.resolve_slice as usize);
            attachment.setResolveDepthPlane(stencil.place.resolve_depth_plane as usize);
            attachment.setLoadAction(MTLLoadAction(stencil.load.raw() as usize));
            attachment.setStoreAction(MTLStoreAction(stencil.store.raw() as usize));
            attachment.setClearStencil(stencil.clear);
            attachment.setStoreActionOptions(MTLStoreActionOptions::from_bits_retain(
                stencil.options.raw() as usize,
            ));
            attachment.setStencilResolveFilter(MTLMultisampleStencilResolveFilter(
                stencil.filter.raw() as usize,
            ));
            if let Some(resolve) = stencil.resolve {
                let resolve: Retained<ProtocolObject<dyn MTLTexture>> =
                    Retained::from(self.held_texture(resolve)?);
                attachment.setResolveTexture(Some(&resolve));
            }
        }
        if let Some(buffer) = pass.visibility {
            descriptor.setVisibilityResultBuffer(Some(self.held(buffer)?));
        }
        // Written whatever it is, since 0 is Metal's own default and the property takes it.
        descriptor.setRenderTargetArrayLength(pass.array_length as usize);
        if let Some(size) = pass.size {
            descriptor.setRenderTargetWidth(size[0] as usize);
            descriptor.setRenderTargetHeight(size[1] as usize);
            descriptor.setDefaultRasterSampleCount(1);
        }
        if !pass.positions.is_empty() {
            // The one place the Metal leg encodes `setSamplePositions:count:`, so the guard that
            // keeps the suite off Metal's fatal shapes is reached by every case that states
            // positions, whatever else it does. The samples are the pass's own attachments', which
            // is what Metal counts the positions against.
            let samples = self.pass_samples(pass)?;
            if let Some(refusal) = crate::render::forbidden_positions(&pass.positions, samples) {
                return Err(refusal);
            }
            let positions: Vec<MTLSamplePosition> = pass
                .positions
                .iter()
                .map(|position| MTLSamplePosition {
                    x: position.x,
                    y: position.y,
                })
                .collect();
            // SAFETY: the pointer is to `positions.len()` positions that live for the call, and
            // the count is at or below the pass's samples, which the guard above has just held.
            unsafe {
                descriptor.setSamplePositions_count(positions.as_ptr(), positions.len());
            }
        }
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&descriptor)
            .ok_or_else(|| Failure::named("RenderCommandEncoder", "returned nil".to_owned()))?;
        for step in &pass.steps {
            self.encode_step_of(&encoder, step)?;
        }
        encoder.endEncoding();
        Ok(())
    }

    /// Encode one of a case's render-pass steps into `encoder`.
    fn encode_step_of(
        &self,
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        step: &RenderStep,
    ) -> Answer<()> {
        let primitive =
            |kind: metal2vulkan::metal::PrimitiveType| MTLPrimitiveType(kind.as_raw() as usize);
        match step {
            RenderStep::Pipeline(state) => {
                let state = self.held_render_state(*state)?;
                encoder.setRenderPipelineState(&state);
            }
            RenderStep::DepthStencilState(state) => {
                let state = self
                    .depth_stencil_states
                    .get(state.0 as usize)
                    .ok_or_else(|| {
                        Failure::named("NoSuchHandle", format!("depth/stencil state {}", state.0))
                    })?;
                encoder.setDepthStencilState(Some(state));
            }
            RenderStep::StencilReference(value) => encoder.setStencilReferenceValue(*value),
            RenderStep::VertexBuffer {
                buffer,
                offset,
                index,
            } => {
                let held = self.held(*buffer)?;
                // SAFETY: the buffer is at least as long as the vertex function's binding from
                // that offset, which the case states as one of its inputs, and the encoder
                // outlives the call.
                unsafe {
                    encoder.setVertexBuffer_offset_atIndex(
                        Some(held),
                        *offset as usize,
                        *index as usize,
                    );
                }
            }
            RenderStep::FragmentBuffer {
                buffer,
                offset,
                index,
            } => {
                let held = self.held(*buffer)?;
                // SAFETY: as the vertex buffer's, for the fragment function's binding.
                unsafe {
                    encoder.setFragmentBuffer_offset_atIndex(
                        Some(held),
                        *offset as usize,
                        *index as usize,
                    );
                }
            }
            RenderStep::FragmentTexture { texture, index } => {
                let held = self.held_texture(*texture)?;
                // SAFETY: the texture is of the kind and access the fragment function declares at
                // that slot, and the encoder outlives the call.
                unsafe { encoder.setFragmentTexture_atIndex(Some(held), *index as usize) };
            }
            RenderStep::FragmentSampler { sampler, index } => {
                let held = self.held_sampler(*sampler)?;
                // SAFETY: the fragment function declares a sampler at that slot, and the encoder
                // outlives the call.
                unsafe { encoder.setFragmentSamplerState_atIndex(Some(held), *index as usize) };
            }
            RenderStep::VertexBytes { bytes, index } => {
                let pointer =
                    std::ptr::NonNull::new(bytes.as_ptr().cast::<std::ffi::c_void>().cast_mut())
                        .ok_or_else(|| Failure::named("NoBytes", "an empty inline binding"))?;
                // SAFETY: `bytes` lives for the call, which copies them, and is at least as long
                // as the vertex function's binding at that slot.
                unsafe {
                    encoder.setVertexBytes_length_atIndex(pointer, bytes.len(), *index as usize)
                };
            }
            RenderStep::Viewport(rect) => encoder.setViewport(MTLViewport {
                originX: rect.origin[0],
                originY: rect.origin[1],
                width: rect.size[0],
                height: rect.size[1],
                znear: rect.depth[0],
                zfar: rect.depth[1],
            }),
            RenderStep::BlendColor(channels) => {
                let [red, green, blue, alpha] = channels.0;
                encoder.setBlendColorRed_green_blue_alpha(red, green, blue, alpha);
            }
            RenderStep::VisibilityResultMode { mode, offset } => {
                encoder.setVisibilityResultMode_offset(
                    MTLVisibilityResultMode(mode.raw() as usize),
                    *offset as usize,
                );
            }
            RenderStep::Draw {
                primitive: kind,
                first,
                count,
                instances,
            } => {
                // SAFETY: the draw names a vertex range of the case's own triangle, and every
                // buffer the vertex function reads was bound by a step before it.
                unsafe {
                    encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                        primitive(*kind),
                        *first as usize,
                        *count as usize,
                        *instances as usize,
                        0,
                    );
                }
            }
            RenderStep::DrawIndexed {
                primitive: kind,
                count,
                index_type,
                indices,
                offset,
                instances,
                base_vertex,
                base_instance,
            } => {
                let buffer = self.held(*indices)?;
                // SAFETY: the index buffer holds the case's own indices from that offset, which
                // the case states as one of its inputs, and the counts are its own.
                unsafe {
                    encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                        primitive(*kind),
                        *count as usize,
                        MTLIndexType(index_type.as_raw() as usize),
                        buffer,
                        *offset as usize,
                        *instances as usize,
                        *base_vertex as isize,
                        *base_instance as usize,
                    );
                }
            }
            RenderStep::DrawIndirect {
                primitive: kind,
                arguments,
                offset,
            } => {
                let buffer = self.held(*arguments)?;
                // SAFETY: the buffer holds the draw's four 32-bit arguments at that offset, which
                // the case wrote before the pass.
                unsafe {
                    encoder.drawPrimitives_indirectBuffer_indirectBufferOffset(
                        primitive(*kind),
                        buffer,
                        *offset as usize,
                    );
                }
            }
            RenderStep::DrawPlain {
                primitive: kind,
                first,
                count,
            } => {
                // SAFETY: the draw names a vertex range of the case's own positions, and every
                // buffer the vertex function reads was bound by a step before it.
                unsafe {
                    encoder.drawPrimitives_vertexStart_vertexCount(
                        primitive(*kind),
                        *first as usize,
                        *count as usize,
                    );
                }
            }
            RenderStep::DrawInstanced {
                primitive: kind,
                first,
                count,
                instances,
            } => {
                // SAFETY: as `DrawPlain`'s, for `instances` instances of the same range.
                unsafe {
                    encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                        primitive(*kind),
                        *first as usize,
                        *count as usize,
                        *instances as usize,
                    );
                }
            }
            RenderStep::DrawIndexedPlain {
                primitive: kind,
                count,
                index_type,
                indices,
                offset,
            } => {
                let buffer = self.held(*indices)?;
                // SAFETY: the index buffer holds `count` indices of that type from that offset,
                // which the case states as one of its inputs, and every index names a vertex of
                // the case's own positions.
                unsafe {
                    encoder
                        .drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                            primitive(*kind),
                            *count as usize,
                            MTLIndexType(index_type.as_raw() as usize),
                            buffer,
                            *offset as usize,
                        );
                }
            }
            RenderStep::DrawIndexedInstanced {
                primitive: kind,
                count,
                index_type,
                indices,
                offset,
                instances,
            } => {
                let buffer = self.held(*indices)?;
                // SAFETY: as `DrawIndexedPlain`'s, for `instances` instances of the same indices.
                unsafe {
                    encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount(
                        primitive(*kind),
                        *count as usize,
                        MTLIndexType(index_type.as_raw() as usize),
                        buffer,
                        *offset as usize,
                        *instances as usize,
                    );
                }
            }
            RenderStep::DrawIndexedIndirect {
                primitive: kind,
                index_type,
                indices,
                index_offset,
                arguments,
                offset,
            } => {
                let index_buffer = self.held(*indices)?;
                let argument_buffer = self.held(*arguments)?;
                // SAFETY: the argument buffer holds the draw's five 32-bit arguments at that
                // offset and the index buffer holds the indices they name, both written by the
                // case before the pass.
                unsafe {
                    encoder.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                        primitive(*kind),
                        MTLIndexType(index_type.as_raw() as usize),
                        index_buffer,
                        *index_offset as usize,
                        argument_buffer,
                        *offset as usize,
                    );
                }
            }
            RenderStep::UseBuffer {
                buffer,
                write,
                stages,
            } => {
                let held = self.held(*buffer)?;
                let resource: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(held);
                encoder.useResource_usage_stages(
                    resource,
                    commands_usage(*write),
                    MTLRenderStages::from_bits_retain(*stages as usize),
                );
            }
            RenderStep::ExecuteCommands {
                commands,
                first,
                count,
            } => {
                let icb = self.held_commands(*commands)?;
                let range = NSRange::new(*first as usize, *count as usize);
                // SAFETY: the range is inside the buffer's own command count, which the case
                // states and reads back beside it, and every buffer the commands' functions reach
                // was made resident by a step before this one.
                unsafe { encoder.executeCommandsInBuffer_withRange(&icb, range) };
            }
            RenderStep::Scissor(rect) => encoder.setScissorRect(scissor(rect)),
            RenderStep::Scissors(rects) => {
                let mut held: Vec<MTLScissorRect> = rects.iter().map(scissor).collect();
                let count = held.len();
                let at = array_pointer(&mut held, "NoScissorRects")?;
                // SAFETY: `at` points to `count` rectangles that live for the call, and every one
                // lies inside the pass's target, which the case's own guard holds.
                unsafe { encoder.setScissorRects_count(at, count) };
            }
            RenderStep::Viewports(rects) => {
                let mut held: Vec<MTLViewport> = rects.iter().map(viewport).collect();
                let count = held.len();
                let at = array_pointer(&mut held, "NoViewports")?;
                // SAFETY: `at` points to `count` viewports that live for the call.
                unsafe { encoder.setViewports_count(at, count) };
            }
            RenderStep::Cull(mode) => encoder.setCullMode(MTLCullMode(mode.as_raw() as usize)),
            RenderStep::FrontFacing(winding) => {
                encoder.setFrontFacingWinding(MTLWinding(winding.as_raw() as usize));
            }
            RenderStep::FillMode(mode) => {
                encoder.setTriangleFillMode(MTLTriangleFillMode(mode.as_raw() as usize));
            }
            RenderStep::DepthClip(mode) => {
                encoder.setDepthClipMode(MTLDepthClipMode(mode.as_raw() as usize));
            }
            RenderStep::DepthBias(values) => {
                encoder.setDepthBias_slopeScale_clamp(
                    values.bias,
                    values.slope_scale,
                    values.clamp,
                );
            }
            RenderStep::StencilReferences { front, back } => {
                encoder.setStencilFrontReferenceValue_backReferenceValue(*front, *back);
            }
            RenderStep::VertexBufferOffset { offset, index } => {
                // SAFETY: the index holds a buffer a step before this one bound, and the offset
                // leaves the stage's binding inside it, which the case states as its own input.
                unsafe {
                    encoder.setVertexBufferOffset_atIndex(*offset as usize, *index as usize);
                }
            }
            RenderStep::FragmentBufferOffset { offset, index } => {
                // SAFETY: as the vertex offset's, for the fragment function's binding.
                unsafe {
                    encoder.setFragmentBufferOffset_atIndex(*offset as usize, *index as usize);
                }
            }
            RenderStep::FragmentBytes { bytes, index } => {
                let pointer =
                    std::ptr::NonNull::new(bytes.as_ptr().cast::<std::ffi::c_void>().cast_mut())
                        .ok_or_else(|| Failure::named("NoBytes", "an empty inline binding"))?;
                // SAFETY: `bytes` lives for the call, which copies them, and is at least as long
                // as the fragment function's binding at that slot.
                unsafe {
                    encoder.setFragmentBytes_length_atIndex(pointer, bytes.len(), *index as usize)
                };
            }
            RenderStep::VertexBuffers { buffers, first } => {
                let mut range = self.buffer_range(buffers)?;
                let mut pointers = raw_pointers(&range.held);
                let count = pointers.len();
                let at = array_pointer(&mut pointers, "NoBuffers")?;
                let lengths = array_pointer(&mut range.offsets, "NoBuffers")?;
                // SAFETY: both pointers are to `count` entries that live for the call, over the
                // buffers `range.held` borrows from the side's pool, and each binding lies inside
                // its buffer, which the case states as one of its inputs.
                unsafe {
                    encoder.setVertexBuffers_offsets_withRange(
                        at,
                        lengths,
                        NSRange::new(*first as usize, count),
                    );
                }
            }
            RenderStep::FragmentBuffers { buffers, first } => {
                let mut range = self.buffer_range(buffers)?;
                let mut pointers = raw_pointers(&range.held);
                let count = pointers.len();
                let at = array_pointer(&mut pointers, "NoBuffers")?;
                let lengths = array_pointer(&mut range.offsets, "NoBuffers")?;
                // SAFETY: as the vertex range's, for the fragment function's bindings.
                unsafe {
                    encoder.setFragmentBuffers_offsets_withRange(
                        at,
                        lengths,
                        NSRange::new(*first as usize, count),
                    );
                }
            }
            RenderStep::VertexTexture { texture, index } => {
                let held = self.held_texture(*texture)?;
                // SAFETY: the texture is of the kind and access the vertex function declares at
                // that slot, and the encoder outlives the call.
                unsafe { encoder.setVertexTexture_atIndex(Some(held), *index as usize) };
            }
            RenderStep::VertexTextures { textures, first } => {
                let held = self.texture_range(textures)?;
                let mut pointers = raw_pointers(&held);
                let count = pointers.len();
                let at = array_pointer(&mut pointers, "NoTextures")?;
                // SAFETY: the pointer is to `count` texture pointers that live for the call, over
                // the textures `held` borrows from the side's pool.
                unsafe {
                    encoder.setVertexTextures_withRange(at, NSRange::new(*first as usize, count));
                }
            }
            RenderStep::FragmentTextures { textures, first } => {
                let held = self.texture_range(textures)?;
                let mut pointers = raw_pointers(&held);
                let count = pointers.len();
                let at = array_pointer(&mut pointers, "NoTextures")?;
                // SAFETY: as the vertex range's, for the fragment function's slots.
                unsafe {
                    encoder.setFragmentTextures_withRange(at, NSRange::new(*first as usize, count));
                }
            }
            RenderStep::VertexSampler { sampler, index } => {
                let held = self.held_sampler(*sampler)?;
                // SAFETY: the vertex function declares a sampler at that slot, and the encoder
                // outlives the call.
                unsafe { encoder.setVertexSamplerState_atIndex(Some(held), *index as usize) };
            }
            RenderStep::VertexSamplers { samplers, first } => {
                let held = self.sampler_range(samplers)?;
                let mut pointers = raw_pointers(&held);
                let count = pointers.len();
                let at = array_pointer(&mut pointers, "NoSamplers")?;
                // SAFETY: the pointer is to `count` sampler pointers that live for the call, over
                // the states `held` borrows from the side's pool.
                unsafe {
                    encoder
                        .setVertexSamplerStates_withRange(at, NSRange::new(*first as usize, count));
                }
            }
            RenderStep::FragmentSamplers { samplers, first } => {
                let held = self.sampler_range(samplers)?;
                let mut pointers = raw_pointers(&held);
                let count = pointers.len();
                let at = array_pointer(&mut pointers, "NoSamplers")?;
                // SAFETY: as the vertex range's, for the fragment function's slots.
                unsafe {
                    encoder.setFragmentSamplerStates_withRange(
                        at,
                        NSRange::new(*first as usize, count),
                    );
                }
            }
            RenderStep::VertexSamplerClamped {
                sampler,
                clamp,
                index,
            } => {
                let held = self.held_sampler(*sampler)?;
                // SAFETY: the vertex function declares a sampler at that slot, and the encoder
                // outlives the call.
                unsafe {
                    encoder.setVertexSamplerState_lodMinClamp_lodMaxClamp_atIndex(
                        Some(held),
                        clamp.min,
                        clamp.max,
                        *index as usize,
                    );
                }
            }
            RenderStep::FragmentSamplerClamped {
                sampler,
                clamp,
                index,
            } => {
                let held = self.held_sampler(*sampler)?;
                // SAFETY: as the vertex binding's, for the fragment function's slot.
                unsafe {
                    encoder.setFragmentSamplerState_lodMinClamp_lodMaxClamp_atIndex(
                        Some(held),
                        clamp.min,
                        clamp.max,
                        *index as usize,
                    );
                }
            }
            RenderStep::VertexSamplersClamped { samplers, first } => {
                let mut range = self.clamped_sampler_range(samplers)?;
                let mut pointers = raw_pointers(&range.held);
                let count = pointers.len();
                let at = array_pointer(&mut pointers, "NoSamplers")?;
                let lowest = array_pointer(&mut range.minimums, "NoSamplers")?;
                let highest = array_pointer(&mut range.maximums, "NoSamplers")?;
                // SAFETY: the three pointers are to `count` entries each, all living for the call,
                // over the states `range.held` borrows from the side's pool.
                unsafe {
                    encoder.setVertexSamplerStates_lodMinClamps_lodMaxClamps_withRange(
                        at,
                        lowest,
                        highest,
                        NSRange::new(*first as usize, count),
                    );
                }
            }
            RenderStep::FragmentSamplersClamped { samplers, first } => {
                let mut range = self.clamped_sampler_range(samplers)?;
                let mut pointers = raw_pointers(&range.held);
                let count = pointers.len();
                let at = array_pointer(&mut pointers, "NoSamplers")?;
                let lowest = array_pointer(&mut range.minimums, "NoSamplers")?;
                let highest = array_pointer(&mut range.maximums, "NoSamplers")?;
                // SAFETY: as the vertex range's, for the fragment function's slots.
                unsafe {
                    encoder.setFragmentSamplerStates_lodMinClamps_lodMaxClamps_withRange(
                        at,
                        lowest,
                        highest,
                        NSRange::new(*first as usize, count),
                    );
                }
            }
            RenderStep::BarrierScope {
                scope,
                after,
                before,
            } => encoder.memoryBarrierWithScope_afterStages_beforeStages(
                MTLBarrierScope(scope.as_raw() as usize),
                MTLRenderStages(after.as_raw() as usize),
                MTLRenderStages(before.as_raw() as usize),
            ),
            RenderStep::BarrierResources {
                resources,
                after,
                before,
            } => {
                let held = self.resource_range(resources)?;
                let mut pointers = held_pointers(&held);
                let count = pointers.len();
                let at = array_pointer(&mut pointers, "NoResources")?;
                // SAFETY: `at` points to `count` resource pointers, over the resources `held`
                // borrows from the side's pools for the whole call.
                unsafe {
                    encoder.memoryBarrierWithResources_count_afterStages_beforeStages(
                        at,
                        count,
                        MTLRenderStages(after.as_raw() as usize),
                        MTLRenderStages(before.as_raw() as usize),
                    );
                }
            }
            // Metal deprecated the four calls below in favour of the ones that name the stages,
            // and `docs/METAL-COVERAGE.md` records the runtime implementing each as every stage.
            // They are items of the SDK a program still calls, so the suite exercises them; the
            // deprecation is Metal's and there is no other selector that makes this call.
            #[allow(deprecated)]
            RenderStep::UseResource { resource, usage } => {
                let held = self.held_resource(*resource)?;
                encoder.useResource_usage(held, MTLResourceUsage(usage.as_raw() as usize));
            }
            RenderStep::UseResourceInStages {
                resource,
                usage,
                stages,
            } => {
                let held = self.held_resource(*resource)?;
                encoder.useResource_usage_stages(
                    held,
                    MTLResourceUsage(usage.as_raw() as usize),
                    MTLRenderStages(stages.as_raw() as usize),
                );
            }
            #[allow(deprecated)]
            RenderStep::UseResources { resources, usage } => {
                let held = self.resource_range(resources)?;
                let mut pointers = held_pointers(&held);
                let count = pointers.len();
                let at = array_pointer(&mut pointers, "NoResources")?;
                // SAFETY: `at` points to `count` resource pointers, over the resources `held`
                // borrows from the side's pools for the whole call.
                unsafe {
                    encoder.useResources_count_usage(
                        at,
                        count,
                        MTLResourceUsage(usage.as_raw() as usize),
                    );
                }
            }
            RenderStep::UseResourcesInStages {
                resources,
                usage,
                stages,
            } => {
                let held = self.resource_range(resources)?;
                let mut pointers = held_pointers(&held);
                let count = pointers.len();
                let at = array_pointer(&mut pointers, "NoResources")?;
                // SAFETY: as the unstaged range's.
                unsafe {
                    encoder.useResources_count_usage_stages(
                        at,
                        count,
                        MTLResourceUsage(usage.as_raw() as usize),
                        MTLRenderStages(stages.as_raw() as usize),
                    );
                }
            }
            #[allow(deprecated)]
            RenderStep::UseHeap(heap) => encoder.useHeap(self.held_heap(*heap)?),
            RenderStep::UseHeapInStages { heap, stages } => encoder.useHeap_stages(
                self.held_heap(*heap)?,
                MTLRenderStages(stages.as_raw() as usize),
            ),
            #[allow(deprecated)]
            RenderStep::UseHeaps(heaps) => {
                let held = self.heap_range(heaps)?;
                let mut pointers = held_pointers(&held);
                let count = pointers.len();
                let at = array_pointer(&mut pointers, "NoHeaps")?;
                // SAFETY: `at` points to `count` heap pointers, over the heaps `held` borrows from
                // the side's pool for the whole call.
                unsafe { encoder.useHeaps_count(at, count) };
            }
            RenderStep::UseHeapsInStages { heaps, stages } => {
                let held = self.heap_range(heaps)?;
                let mut pointers = held_pointers(&held);
                let count = pointers.len();
                let at = array_pointer(&mut pointers, "NoHeaps")?;
                // SAFETY: as the unstaged range's.
                unsafe {
                    encoder.useHeaps_count_stages(
                        at,
                        count,
                        MTLRenderStages(stages.as_raw() as usize),
                    );
                }
            }
            RenderStep::TessellationFactorBuffer {
                buffer,
                offset,
                instance_stride,
            } => {
                let held = buffer.map(|handle| self.held(handle)).transpose()?;
                // SAFETY: the buffer is one of this side's own, and the case states a range inside
                // it; a nil buffer is Metal's own "no factors bound" and takes no range.
                unsafe {
                    encoder.setTessellationFactorBuffer_offset_instanceStride(
                        held,
                        *offset as usize,
                        *instance_stride as usize,
                    );
                }
            }
            // Metal's own validation complains at a scale that is 0, negative, NaN, infinite or
            // a subnormal `float`; plain Metal takes each, which is what these cases ask about,
            // and the validated leg runs in a child of the command.
            RenderStep::TessellationFactorScale { bits } => {
                encoder.setTessellationFactorScale(f32::from_bits(*bits));
            }
            RenderStep::DrawPatches {
                control_points,
                first,
                count,
                patch_indices,
                patch_index_offset,
                instances,
                base_instance,
            } => {
                let indices = patch_indices.map(|handle| self.held(handle)).transpose()?;
                // SAFETY: the patch index buffer is one of this side's own and the case states a
                // range inside it; the factor buffer the draw reads was bound by a step before it.
                unsafe {
                    encoder.drawPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_instanceCount_baseInstance(
                        *control_points as usize,
                        *first as usize,
                        *count as usize,
                        indices,
                        *patch_index_offset as usize,
                        *instances as usize,
                        *base_instance as usize,
                    );
                }
            }
            RenderStep::DrawIndexedPatches {
                control_points,
                first,
                count,
                patch_indices,
                patch_index_offset,
                control_point_indices,
                control_point_index_offset,
                instances,
                base_instance,
            } => {
                let indices = patch_indices.map(|handle| self.held(handle)).transpose()?;
                let control = self.held(*control_point_indices)?;
                // SAFETY: both index buffers are this side's own, and the case states a range
                // inside each: `(first + count) · control_points` control-point indices from the
                // control-point offset, which is what Metal reads.
                unsafe {
                    encoder.drawIndexedPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_controlPointIndexBuffer_controlPointIndexBufferOffset_instanceCount_baseInstance(
                        *control_points as usize,
                        *first as usize,
                        *count as usize,
                        indices,
                        *patch_index_offset as usize,
                        control,
                        *control_point_index_offset as usize,
                        *instances as usize,
                        *base_instance as usize,
                    );
                }
            }
        }
        Ok(())
    }

    /// The buffers a range step names, with the offsets its call reads beside them.
    fn buffer_range(&self, buffers: &[(Handle, u64)]) -> Answer<BufferRange<'_>> {
        Ok(BufferRange {
            held: buffers
                .iter()
                .map(|(handle, _)| self.held(*handle))
                .collect::<Answer<Vec<_>>>()?,
            offsets: buffers.iter().map(|(_, offset)| *offset as usize).collect(),
        })
    }

    /// The textures a range step names.
    fn texture_range(&self, textures: &[Handle]) -> Answer<Vec<&ProtocolObject<dyn MTLTexture>>> {
        textures
            .iter()
            .map(|handle| self.held_texture(*handle))
            .collect()
    }

    /// The sampler states a range step names.
    fn sampler_range(
        &self,
        samplers: &[Handle],
    ) -> Answer<Vec<&ProtocolObject<dyn MTLSamplerState>>> {
        samplers
            .iter()
            .map(|handle| self.held_sampler(*handle))
            .collect()
    }

    /// The sampler states a clamped range step names, with the two clamp arrays its call reads
    /// beside them.
    fn clamped_sampler_range(
        &self,
        samplers: &[(Handle, crate::render::LodClamp)],
    ) -> Answer<ClampedSamplerRange<'_>> {
        Ok(ClampedSamplerRange {
            held: samplers
                .iter()
                .map(|(handle, _)| self.held_sampler(*handle))
                .collect::<Answer<Vec<_>>>()?,
            minimums: samplers.iter().map(|(_, clamp)| clamp.min).collect(),
            maximums: samplers.iter().map(|(_, clamp)| clamp.max).collect(),
        })
    }

    /// The resources a residency or barrier step names, as `MTLResource`s of the side's pools.
    fn resource_range(
        &self,
        resources: &[StepResource],
    ) -> Answer<Vec<&ProtocolObject<dyn MTLResource>>> {
        resources
            .iter()
            .map(|resource| self.held_resource(*resource))
            .collect()
    }

    /// The one resource a residency step names.
    fn held_resource(&self, resource: StepResource) -> Answer<&ProtocolObject<dyn MTLResource>> {
        Ok(match resource {
            StepResource::Buffer(handle) => ProtocolObject::from_ref(self.held(handle)?),
            StepResource::Texture(handle) => ProtocolObject::from_ref(self.held_texture(handle)?),
        })
    }

    /// The heaps a range step names.
    fn heap_range(&self, heaps: &[Handle]) -> Answer<Vec<&ProtocolObject<dyn MTLHeap>>> {
        heaps.iter().map(|handle| self.held_heap(*handle)).collect()
    }
}

/// The buffers one `setVertexBuffers:offsets:withRange:` or `setFragmentBuffers:offsets:withRange:`
/// binds, and the offsets beside them.
///
/// A named carrier rather than a tuple: the call reads two arrays that must be the same length,
/// which is what the type says and a pair of `Vec`s does not.
struct BufferRange<'a> {
    held: Vec<&'a ProtocolObject<dyn MTLBuffer>>,
    offsets: Vec<usize>,
}

/// The sampler states one `set*SamplerStates:lodMinClamps:lodMaxClamps:withRange:` binds, and its
/// two clamp arrays, which are the same length as the states for the same reason.
struct ClampedSamplerRange<'a> {
    held: Vec<&'a ProtocolObject<dyn MTLSamplerState>>,
    minimums: Vec<f32>,
    maximums: Vec<f32>,
}

/// The raw pointers a range-taking binding call reads its objects through. They are valid as long
/// as `held` borrows them, which is the side's own pool for the whole call.
fn raw_pointers<T: ?Sized>(held: &[&T]) -> Vec<*const T> {
    held.iter().map(|object| *object as *const T).collect()
}

/// The pointers a count-taking residency or barrier call reads its objects through, valid as
/// `raw_pointers`' are.
fn held_pointers<T: ?Sized>(held: &[&T]) -> Vec<std::ptr::NonNull<T>> {
    held.iter().map(|object| (*object).into()).collect()
}

/// The pointer a count-taking or range-taking call reads an array of `at` through, or the failure
/// an empty range is: Metal's calls take a pointer and a count, and a zero-length range has no
/// pointer to give.
fn array_pointer<T>(at: &mut [T], what: &'static str) -> Answer<std::ptr::NonNull<T>> {
    std::ptr::NonNull::new(at.as_mut_ptr()).ok_or_else(|| Failure::named(what, "an empty range"))
}

/// The `MTLScissorRect` a case's rectangle is.
fn scissor(rect: &crate::render::ScissorBox) -> MTLScissorRect {
    MTLScissorRect {
        x: rect.x as usize,
        y: rect.y as usize,
        width: rect.width as usize,
        height: rect.height as usize,
    }
}

/// The `MTLViewport` a case's viewport is.
fn viewport(rect: &crate::render::ViewportRect) -> MTLViewport {
    MTLViewport {
        originX: rect.origin[0],
        originY: rect.origin[1],
        width: rect.size[0],
        height: rect.size[1],
        znear: rect.depth[0],
        zfar: rect.depth[1],
    }
}

/// The usage an indirect command buffer's residency declaration states: `MTLResourceUsageRead`,
/// and `MTLResourceUsageWrite` beside it where `write`.
fn commands_usage(write: bool) -> MTLResourceUsage {
    if write {
        MTLResourceUsage::Read | MTLResourceUsage::Write
    } else {
        MTLResourceUsage::Read
    }
}

/// The indirect-command-buffer vocabulary ([`crate::indirect_commands`]) on native Metal.
///
/// Each command is handed to its own function so the command object arrives as a **typed
/// parameter**: that is what the setters are attributed to, and it keeps the two kinds' calls
/// apart in one place each.
impl MetalSide {
    /// The indirect command buffer `handle` names.
    fn held_commands(
        &self,
        handle: Handle,
    ) -> Answer<Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>> {
        self.command_buffers_indirect
            .get(handle.0 as usize)
            .cloned()
            .ok_or_else(|| {
                Failure::named("NoSuchHandle", format!("indirect commands {}", handle.0))
            })
    }

    /// Encode one step onto an indirect compute command.
    fn encode_compute_command(
        &self,
        encoder: &ProtocolObject<dyn MTLIndirectComputeCommand>,
        step: &ComputeCommandStep,
    ) -> Answer<()> {
        match step {
            ComputeCommandStep::Pipeline(state) => {
                let state = self
                    .pipelines
                    .get(state.0 as usize)
                    .ok_or_else(|| {
                        Failure::named("NoSuchHandle", format!("compute pipeline {}", state.0))
                    })?
                    .clone();
                encoder.setComputePipelineState(&state);
            }
            ComputeCommandStep::KernelBuffer {
                buffer,
                offset,
                index,
            } => {
                let held = self.held(*buffer)?;
                // SAFETY: the buffer is at least as long as the kernel's binding from that
                // offset, which the case states as one of its inputs, and the index is below the
                // descriptor's own kernel buffer count.
                unsafe {
                    encoder.setKernelBuffer_offset_atIndex(held, *offset as usize, *index as usize);
                }
            }
            ComputeCommandStep::KernelBufferWithStride {
                buffer,
                offset,
                stride,
                index,
            } => {
                let held = self.held(*buffer)?;
                // SAFETY: as the plain binding's, and the buffer's descriptor supports a dynamic
                // attribute stride, which the case states.
                unsafe {
                    encoder.setKernelBuffer_offset_attributeStride_atIndex(
                        held,
                        *offset as usize,
                        *stride as usize,
                        *index as usize,
                    );
                }
            }
            ComputeCommandStep::ThreadgroupMemoryLength { length, index } => {
                // SAFETY: the index is below the descriptor's own threadgroup length count and
                // the length is the case's own, inside the device's threadgroup memory.
                unsafe {
                    encoder.setThreadgroupMemoryLength_atIndex(*length as usize, *index as usize);
                }
            }
            ComputeCommandStep::StageInRegion { origin, size } => {
                encoder.setStageInRegion(MTLRegion {
                    origin: MTLOrigin {
                        x: origin[0] as usize,
                        y: origin[1] as usize,
                        z: origin[2] as usize,
                    },
                    size: MTLSize {
                        width: size[0] as usize,
                        height: size[1] as usize,
                        depth: size[2] as usize,
                    },
                });
            }
            ComputeCommandStep::ImageblockSize { width, height } => {
                encoder.setImageblockWidth_height(*width as usize, *height as usize);
            }
            ComputeCommandStep::DispatchThreadgroups { groups, threads } => {
                encoder.concurrentDispatchThreadgroups_threadsPerThreadgroup(
                    size(*groups),
                    size(*threads),
                );
            }
            ComputeCommandStep::DispatchThreads { threads, per_group } => {
                encoder.concurrentDispatchThreads_threadsPerThreadgroup(
                    size(*threads),
                    size(*per_group),
                );
            }
            ComputeCommandStep::Barrier => encoder.setBarrier(),
            ComputeCommandStep::ClearBarrier => encoder.clearBarrier(),
            ComputeCommandStep::Reset => encoder.reset(),
        }
        Ok(())
    }

    /// Encode one step onto an indirect render command.
    fn encode_render_command(
        &self,
        encoder: &ProtocolObject<dyn MTLIndirectRenderCommand>,
        step: &RenderCommandStep,
    ) -> Answer<()> {
        let primitive =
            |kind: metal2vulkan::metal::PrimitiveType| MTLPrimitiveType(kind.as_raw() as usize);
        match step {
            RenderCommandStep::Pipeline(state) => {
                let state = self.held_render_state(*state)?;
                encoder.setRenderPipelineState(&state);
            }
            RenderCommandStep::VertexBuffer {
                buffer,
                offset,
                index,
            } => {
                let held = self.held(*buffer)?;
                // SAFETY: the buffer is at least as long as the function's binding from that
                // offset, and the index is below the descriptor's own vertex buffer count.
                unsafe {
                    encoder.setVertexBuffer_offset_atIndex(held, *offset as usize, *index as usize);
                }
            }
            RenderCommandStep::VertexBufferWithStride {
                buffer,
                offset,
                stride,
                index,
            } => {
                let held = self.held(*buffer)?;
                // SAFETY: as the plain binding's, and the buffer's descriptor supports a dynamic
                // attribute stride.
                unsafe {
                    encoder.setVertexBuffer_offset_attributeStride_atIndex(
                        held,
                        *offset as usize,
                        *stride as usize,
                        *index as usize,
                    );
                }
            }
            RenderCommandStep::FragmentBuffer {
                buffer,
                offset,
                index,
            } => {
                let held = self.held(*buffer)?;
                // SAFETY: as the vertex binding's, below the descriptor's fragment buffer count.
                unsafe {
                    encoder.setFragmentBuffer_offset_atIndex(
                        held,
                        *offset as usize,
                        *index as usize,
                    );
                }
            }
            RenderCommandStep::ObjectBuffer {
                buffer,
                offset,
                index,
            } => {
                let held = self.held(*buffer)?;
                // SAFETY: as the vertex binding's, below the descriptor's object buffer count.
                unsafe {
                    encoder.setObjectBuffer_offset_atIndex(held, *offset as usize, *index as usize);
                }
            }
            RenderCommandStep::MeshBuffer {
                buffer,
                offset,
                index,
            } => {
                let held = self.held(*buffer)?;
                // SAFETY: as the vertex binding's, below the descriptor's mesh buffer count.
                unsafe {
                    encoder.setMeshBuffer_offset_atIndex(held, *offset as usize, *index as usize);
                }
            }
            RenderCommandStep::ObjectThreadgroupMemoryLength { length, index } => {
                // SAFETY: the length is the case's own, inside the device's threadgroup memory.
                unsafe {
                    encoder.setObjectThreadgroupMemoryLength_atIndex(
                        *length as usize,
                        *index as usize,
                    );
                }
            }
            RenderCommandStep::Draw {
                primitive: kind,
                first,
                count,
                instances,
                base_instance,
            } => {
                // SAFETY: the draw names a vertex range of the case's own geometry, and every
                // buffer the vertex function reads was bound by a step before it.
                unsafe {
                    encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                        primitive(*kind),
                        *first as usize,
                        *count as usize,
                        *instances as usize,
                        *base_instance as usize,
                    );
                }
            }
            RenderCommandStep::DrawIndexed {
                primitive: kind,
                count,
                index_type,
                indices,
                offset,
                instances,
                base_vertex,
                base_instance,
            } => {
                let buffer = self.held(*indices)?;
                // SAFETY: the index buffer holds the case's own indices from that offset, which
                // the case states as one of its inputs, and the counts are its own.
                unsafe {
                    encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                        primitive(*kind),
                        *count as usize,
                        MTLIndexType(index_type.as_raw() as usize),
                        buffer,
                        *offset as usize,
                        *instances as usize,
                        *base_vertex as isize,
                        *base_instance as usize,
                    );
                }
            }
            RenderCommandStep::DrawPatches {
                control_points,
                first,
                count,
                patch_indices,
                patch_index_offset,
                instances,
                base_instance,
                factors,
                factor_offset,
                factor_instance_stride,
            } => {
                let indices = match patch_indices {
                    Some(handle) => Some(self.held(*handle)?),
                    None => None,
                };
                let factor_buffer = self.held(*factors)?;
                // SAFETY: the patch range lies inside the case's own patch index buffer where it
                // has one, the factor buffer holds a record for every patch and instance the draw
                // reaches, and the pipeline a step before this one set is a tessellating one of
                // these control points.
                unsafe {
                    encoder.drawPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_instanceCount_baseInstance_tessellationFactorBuffer_tessellationFactorBufferOffset_tessellationFactorBufferInstanceStride(
                        *control_points as usize,
                        *first as usize,
                        *count as usize,
                        indices,
                        *patch_index_offset as usize,
                        *instances as usize,
                        *base_instance as usize,
                        factor_buffer,
                        *factor_offset as usize,
                        *factor_instance_stride as usize,
                    );
                }
            }
            RenderCommandStep::DrawIndexedPatches {
                control_points,
                first,
                count,
                patch_indices,
                patch_index_offset,
                control_point_indices,
                control_point_index_offset,
                instances,
                base_instance,
                factors,
                factor_offset,
                factor_instance_stride,
            } => {
                let indices = match patch_indices {
                    Some(handle) => Some(self.held(*handle)?),
                    None => None,
                };
                let points = self.held(*control_point_indices)?;
                let factor_buffer = self.held(*factors)?;
                // SAFETY: as the plain patch draw's, and the control-point index buffer holds an
                // index of the pipeline's own type for every control point the patches reach.
                unsafe {
                    encoder.drawIndexedPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_controlPointIndexBuffer_controlPointIndexBufferOffset_instanceCount_baseInstance_tessellationFactorBuffer_tessellationFactorBufferOffset_tessellationFactorBufferInstanceStride(
                        *control_points as usize,
                        *first as usize,
                        *count as usize,
                        indices,
                        *patch_index_offset as usize,
                        points,
                        *control_point_index_offset as usize,
                        *instances as usize,
                        *base_instance as usize,
                        factor_buffer,
                        *factor_offset as usize,
                        *factor_instance_stride as usize,
                    );
                }
            }
            RenderCommandStep::DrawMeshThreadgroups {
                groups,
                object,
                mesh,
            } => {
                encoder.drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                    size(*groups),
                    size(*object),
                    size(*mesh),
                );
            }
            RenderCommandStep::DrawMeshThreads {
                threads,
                object,
                mesh,
            } => {
                encoder.drawMeshThreads_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                    size(*threads),
                    size(*object),
                    size(*mesh),
                );
            }
            RenderCommandStep::Barrier => encoder.setBarrier(),
            RenderCommandStep::ClearBarrier => encoder.clearBarrier(),
            RenderCommandStep::Reset => encoder.reset(),
        }
        Ok(())
    }
}

/// The render vocabulary ([`crate::render`]) on native Metal.
impl MetalSide {
    /// The render pipeline state `handle` names.
    fn held_render_state(
        &self,
        handle: Handle,
    ) -> Answer<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        self.render_pipelines
            .get(handle.0 as usize)
            .map(|held| held.state.clone())
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("render pipeline {}", handle.0)))
    }

    /// Encode one compute dispatch of a program into `command_buffer`.
    fn encode_dispatch(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        dispatch: &Dispatch,
    ) -> Answer<()> {
        let state = self
            .pipelines
            .get(dispatch.pipeline.0 as usize)
            .ok_or_else(|| {
                Failure::named(
                    "NoSuchHandle",
                    format!("compute pipeline {}", dispatch.pipeline.0),
                )
            })?
            .clone();
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| Failure::named("ComputeCommandEncoder", "returned nil".to_owned()))?;
        encoder.setComputePipelineState(&state);
        for (at, (buffer, offset)) in dispatch.buffers.iter().enumerate() {
            let held = self.held(*buffer)?;
            // SAFETY: the buffer is at least as long as the kernel's binding from that offset,
            // which the case states as one of its inputs, and the encoder outlives the call.
            unsafe { encoder.setBuffer_offset_atIndex(Some(held), *offset as usize, at) };
        }
        for (at, texture) in dispatch.textures.iter().enumerate() {
            let held = self.held_texture(*texture)?;
            // SAFETY: the texture is of the kind and access the kernel declares at that slot, and
            // the encoder outlives the call.
            unsafe { encoder.setTexture_atIndex(Some(held), at) };
        }
        encoder.dispatchThreads_threadsPerThreadgroup(size(dispatch.grid), size(dispatch.threads));
        encoder.endEncoding();
        Ok(())
    }

    /// Encode one command of a compute pass into `encoder`.
    ///
    /// Each arm is one Metal call, with the handles resolved through the side's own pools, so a
    /// case's command list reads as the calls it makes.
    fn encode_compute_step(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        step: &ComputeStep,
    ) -> Answer<()> {
        // A resource of either kind as `id<MTLResource>`: what the residency declarations and the
        // resource barrier take.
        let resource = |named: &PassResource| -> Answer<Retained<ProtocolObject<dyn MTLResource>>> {
            match named {
                PassResource::Buffer(handle) => {
                    let held = self.held(*handle)?;
                    Ok(Retained::from(ProtocolObject::<dyn MTLResource>::from_ref(
                        held,
                    )))
                }
                PassResource::Texture(handle) => {
                    let held = self.held_texture(*handle)?;
                    Ok(Retained::from(ProtocolObject::<dyn MTLResource>::from_ref(
                        held,
                    )))
                }
            }
        };
        match step {
            ComputeStep::Pipeline(pipeline) => {
                let state = self
                    .pipelines
                    .get(pipeline.0 as usize)
                    .ok_or_else(|| {
                        Failure::named("NoSuchHandle", format!("pipeline {}", pipeline.0))
                    })?
                    .clone();
                encoder.setComputePipelineState(&state);
            }
            ComputeStep::Buffer {
                buffer,
                offset,
                index,
            } => {
                let held = self.held(*buffer)?;
                // SAFETY: the buffer is at least as long as the kernel's binding from that
                // offset, which the case states as one of its inputs, and the encoder outlives
                // the call.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(held), *offset as usize, *index as usize)
                };
            }
            ComputeStep::BufferOffset { offset, index } => {
                // SAFETY: a buffer is bound at that index already — the case states the binding
                // before this command — and the offset leaves the kernel's binding inside it.
                unsafe { encoder.setBufferOffset_atIndex(*offset as usize, *index as usize) };
            }
            ComputeStep::Buffers { buffers, first } => {
                let held: Vec<Option<Retained<ProtocolObject<dyn MTLBuffer>>>> = buffers
                    .iter()
                    .map(|(handle, _)| self.held(*handle).map(|held| Some(Retained::from(held))))
                    .collect::<Answer<Vec<_>>>()?;
                let pointers = pointer_array(&held);
                let offsets: Vec<usize> = buffers.iter().map(|(_, at)| *at as usize).collect();
                let range = range_of(*first, &pointers)?;
                // SAFETY: both arrays hold the range's length of elements, each buffer outlives
                // the call, and each binding lies inside its buffer from its offset.
                unsafe {
                    encoder.setBuffers_offsets_withRange(
                        array_head(&pointers)?,
                        array_head(&offsets)?,
                        range,
                    );
                }
            }
            ComputeStep::Bytes { bytes, index } => {
                // `bytes.as_ptr()`, never `NonNull::from(bytes)`: the latter is the address of the
                // `Vec` itself — its pointer, capacity and length — and Metal would copy those
                // words into the kernel's binding. Written as the three inline bindings beside it
                // are (`setTileBytes:`, `setObjectBytes:`, `setVertexBytes:`); taking the other
                // one first made this case's own Metal leg read a kernel's coordinates out of a
                // `Vec` header, which read exactly like a product bug.
                let pointer =
                    std::ptr::NonNull::new(bytes.as_ptr().cast::<std::ffi::c_void>().cast_mut())
                        .ok_or_else(|| Failure::named("NoBytes", "an empty inline binding"))?;
                // SAFETY: `bytes` lives for the call, which copies them, and is at least as long
                // as the kernel's binding at that slot.
                unsafe { encoder.setBytes_length_atIndex(pointer, bytes.len(), *index as usize) };
            }
            ComputeStep::BufferWithStride {
                buffer,
                offset,
                stride,
                index,
            } => {
                let held = self.held(*buffer)?;
                // SAFETY: as for `setBuffer:offset:atIndex:`; the stride is a number Metal reads
                // by value, and the case states the buffer's own length beside it.
                unsafe {
                    encoder.setBuffer_offset_attributeStride_atIndex(
                        held,
                        *offset as usize,
                        *stride as usize,
                        *index as usize,
                    );
                }
            }
            ComputeStep::BuffersWithStrides { buffers, first } => {
                let held: Vec<Option<Retained<ProtocolObject<dyn MTLBuffer>>>> = buffers
                    .iter()
                    .map(|(handle, _, _)| self.held(*handle).map(|held| Some(Retained::from(held))))
                    .collect::<Answer<Vec<_>>>()?;
                let pointers = pointer_array(&held);
                let offsets: Vec<usize> = buffers.iter().map(|(_, at, _)| *at as usize).collect();
                let strides: Vec<usize> = buffers
                    .iter()
                    .map(|(_, _, stride)| *stride as usize)
                    .collect();
                let range = range_of(*first, &pointers)?;
                // SAFETY: the three arrays hold the range's length of elements, each buffer
                // outlives the call, and each binding lies inside its buffer from its offset.
                unsafe {
                    encoder.setBuffers_offsets_attributeStrides_withRange(
                        array_head(&pointers)?,
                        array_head(&offsets)?,
                        array_head(&strides)?,
                        range,
                    );
                }
            }
            ComputeStep::BufferOffsetWithStride {
                offset,
                stride,
                index,
            } => {
                // SAFETY: a buffer is bound at that index already — the case states the binding
                // before this command — and the offset leaves the kernel's binding inside it.
                unsafe {
                    encoder.setBufferOffset_attributeStride_atIndex(
                        *offset as usize,
                        *stride as usize,
                        *index as usize,
                    );
                }
            }
            ComputeStep::BytesWithStride {
                bytes,
                stride,
                index,
            } => {
                // `bytes.as_ptr()`, never `NonNull::from(bytes)`: see `ComputeStep::Bytes`.
                let pointer =
                    std::ptr::NonNull::new(bytes.as_ptr().cast::<std::ffi::c_void>().cast_mut())
                        .ok_or_else(|| Failure::named("NoBytes", "an empty inline binding"))?;
                // SAFETY: `bytes` lives for the call, which copies them, and is at least as long
                // as the kernel's binding at that slot.
                unsafe {
                    encoder.setBytes_length_attributeStride_atIndex(
                        pointer,
                        bytes.len(),
                        *stride as usize,
                        *index as usize,
                    );
                }
            }
            ComputeStep::StageInRegion { origin, size } => {
                encoder.setStageInRegion(MTLRegion {
                    origin: MTLOrigin {
                        x: origin[0] as usize,
                        y: origin[1] as usize,
                        z: origin[2] as usize,
                    },
                    size: MTLSize {
                        width: size[0] as usize,
                        height: size[1] as usize,
                        depth: size[2] as usize,
                    },
                });
            }
            ComputeStep::Texture { texture, index } => {
                let held = self.held_texture(*texture)?;
                // SAFETY: the texture is of the kind and access the kernel declares at that slot,
                // and the encoder outlives the call.
                unsafe { encoder.setTexture_atIndex(Some(held), *index as usize) };
            }
            ComputeStep::Textures { textures, first } => {
                let held: Vec<Option<Retained<ProtocolObject<dyn MTLTexture>>>> = textures
                    .iter()
                    .map(|handle| {
                        self.held_texture(*handle)
                            .map(|held| Some(Retained::from(held)))
                    })
                    .collect::<Answer<Vec<_>>>()?;
                let pointers = pointer_array(&held);
                let range = range_of(*first, &pointers)?;
                // SAFETY: the array holds the range's length of pointers and each texture
                // outlives the call.
                unsafe { encoder.setTextures_withRange(array_head(&pointers)?, range) };
            }
            ComputeStep::Sampler { sampler, index } => {
                let held = self.held_sampler(*sampler)?;
                // SAFETY: the kernel declares a sampler at that slot, and the encoder outlives
                // the call.
                unsafe { encoder.setSamplerState_atIndex(Some(held), *index as usize) };
            }
            ComputeStep::Samplers { samplers, first } => {
                let held: Vec<Option<Retained<ProtocolObject<dyn MTLSamplerState>>>> = samplers
                    .iter()
                    .map(|handle| {
                        self.held_sampler(*handle)
                            .map(|held| Some(Retained::from(held)))
                    })
                    .collect::<Answer<Vec<_>>>()?;
                let pointers = pointer_array(&held);
                let range = range_of(*first, &pointers)?;
                // SAFETY: as for `setTextures:withRange:`.
                unsafe { encoder.setSamplerStates_withRange(array_head(&pointers)?, range) };
            }
            ComputeStep::ClampedSampler {
                sampler,
                lod_min,
                lod_max,
                index,
            } => {
                let held = self.held_sampler(*sampler)?;
                // SAFETY: as for `setSamplerState:atIndex:`; the clamps are two floats Metal
                // reads by value.
                unsafe {
                    encoder.setSamplerState_lodMinClamp_lodMaxClamp_atIndex(
                        Some(held),
                        *lod_min,
                        *lod_max,
                        *index as usize,
                    );
                }
            }
            ComputeStep::ClampedSamplers { samplers, first } => {
                let held: Vec<Option<Retained<ProtocolObject<dyn MTLSamplerState>>>> = samplers
                    .iter()
                    .map(|(handle, _, _)| {
                        self.held_sampler(*handle)
                            .map(|held| Some(Retained::from(held)))
                    })
                    .collect::<Answer<Vec<_>>>()?;
                let pointers = pointer_array(&held);
                let mins: Vec<f32> = samplers.iter().map(|(_, min, _)| *min).collect();
                let maxes: Vec<f32> = samplers.iter().map(|(_, _, max)| *max).collect();
                let range = range_of(*first, &pointers)?;
                // SAFETY: the three arrays hold the range's length of elements and each state
                // outlives the call.
                unsafe {
                    encoder.setSamplerStates_lodMinClamps_lodMaxClamps_withRange(
                        array_head(&pointers)?,
                        array_head(&mins)?,
                        array_head(&maxes)?,
                        range,
                    );
                }
            }
            ComputeStep::ThreadgroupMemory { length, index } => {
                // SAFETY: the length is at least the bytes the kernel's `[[threadgroup(n)]]`
                // allocation reads, which the case states as one of its inputs, and the index is
                // one the kernel declares.
                unsafe {
                    encoder.setThreadgroupMemoryLength_atIndex(*length as usize, *index as usize);
                }
            }
            ComputeStep::UseResource {
                resource: one,
                usage,
            } => {
                let held = resource(one)?;
                encoder.useResource_usage(&held, MTLResourceUsage(*usage as usize));
            }
            ComputeStep::UseResources { resources, usage } => {
                let held: Vec<Retained<ProtocolObject<dyn MTLResource>>> =
                    resources.iter().map(resource).collect::<Answer<Vec<_>>>()?;
                let pointers: Vec<std::ptr::NonNull<ProtocolObject<dyn MTLResource>>> = held
                    .iter()
                    .map(|one| std::ptr::NonNull::from(&**one))
                    .collect();
                // SAFETY: the array holds `count` pointers and each resource outlives the call.
                unsafe {
                    encoder.useResources_count_usage(
                        array_head(&pointers)?,
                        pointers.len(),
                        MTLResourceUsage(*usage as usize),
                    );
                }
            }
            ComputeStep::UseHeap { heap } => {
                let held = self.held_heap(*heap)?;
                encoder.useHeap(held);
            }
            ComputeStep::UseHeaps { heaps } => {
                let held = self.heap_range(heaps)?;
                let pointers: Vec<std::ptr::NonNull<ProtocolObject<dyn MTLHeap>>> = held
                    .iter()
                    .map(|one| std::ptr::NonNull::from(&**one))
                    .collect();
                // SAFETY: the array holds `count` pointers and each heap outlives the call.
                unsafe { encoder.useHeaps_count(array_head(&pointers)?, pointers.len()) };
            }
            ComputeStep::BarrierScope { scope } => {
                encoder.memoryBarrierWithScope(MTLBarrierScope(*scope as usize));
            }
            ComputeStep::BarrierResources { resources } => {
                let held: Vec<Retained<ProtocolObject<dyn MTLResource>>> =
                    resources.iter().map(resource).collect::<Answer<Vec<_>>>()?;
                let pointers: Vec<std::ptr::NonNull<ProtocolObject<dyn MTLResource>>> = held
                    .iter()
                    .map(|one| std::ptr::NonNull::from(&**one))
                    .collect();
                // SAFETY: as for `useResources:count:usage:`.
                unsafe {
                    encoder
                        .memoryBarrierWithResources_count(array_head(&pointers)?, pointers.len());
                }
            }
            ComputeStep::DispatchThreadgroups { groups, threads } => {
                encoder.dispatchThreadgroups_threadsPerThreadgroup(size(*groups), size(*threads));
            }
            ComputeStep::DispatchThreadgroupsIndirect {
                buffer,
                offset,
                threads,
            } => {
                let held = self.held(*buffer)?;
                // SAFETY: the buffer holds the grid's three 32-bit words at that offset, which
                // the case states as one of its inputs, and the encoder outlives the call.
                unsafe {
                    encoder
                        .dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                            held,
                            *offset as usize,
                            size(*threads),
                        );
                }
            }
        }
        Ok(())
    }

    /// Encode one compute encoder that makes `used` resident and runs a range of an indirect
    /// command buffer's commands into `command_buffer`.
    fn encode_compute_commands(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        commands: Handle,
        first: u64,
        count: u64,
        used: &[(Handle, bool)],
    ) -> Answer<()> {
        let icb = self.held_commands(commands)?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| Failure::named("ComputeCommandEncoder", "returned nil".to_owned()))?;
        for (buffer, write) in used {
            let held = self.held(*buffer)?;
            let resource: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(held);
            encoder.useResource_usage(resource, commands_usage(*write));
        }
        let range = NSRange::new(first as usize, count as usize);
        // SAFETY: the range is inside the buffer's own command count, which the case states and
        // reads back beside it, and every buffer the commands' kernels reach was just made
        // resident.
        unsafe { encoder.executeCommandsInBuffer_withRange(&icb, range) };
        encoder.endEncoding();
        Ok(())
    }

    /// Encode one texture-to-buffer blit of a program into `command_buffer`, through Metal's
    /// un-optioned selector at [`Plane::Whole`] and its optioned one for a stated plane.
    ///
    /// `source` is the texture read, the level, slice and plane of it the read copies, and that
    /// place's own extent, which for a level past the first is the level's and not the target's.
    fn encode_read_texture(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        source: (Handle, crate::render::ReadAt, [u32; 2]),
        into: Handle,
        offset: u64,
        row_pitch: u32,
        plane: Plane,
    ) -> Answer<()> {
        let (texture, at, extent) = source;
        let texture: Retained<ProtocolObject<dyn MTLTexture>> =
            Retained::from(self.held_texture(texture)?);
        let buffer: Retained<ProtocolObject<dyn MTLBuffer>> = Retained::from(self.held(into)?);
        let bytes = row_pitch as u64 * u64::from(extent[1]);
        if u64::try_from(buffer.length()).unwrap_or(0) < offset + bytes {
            return Err(Failure::named(
                "PastTheBuffer",
                format!(
                    "{bytes} bytes at {offset} of a buffer of {}",
                    buffer.length()
                ),
            ));
        }
        let encoder = command_buffer
            .blitCommandEncoder()
            .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil".to_owned()))?;
        let origin = MTLOrigin {
            x: 0,
            y: 0,
            z: at.depth_plane as usize,
        };
        let size = MTLSize {
            width: extent[0] as usize,
            height: extent[1] as usize,
            depth: 1,
        };
        // Metal binds **two** selectors here and a plane at the default takes the one every
        // recording was taken through, not the optioned one with `MTLBlitOptionNone`. Apple's
        // header says the two are the same call, and the suite does not take that on trust: the
        // blit family's own step vocabulary states both
        // ([`crate::work::Step::CopyTextureToBufferWithoutOptions`]), and routing every render
        // read through the optioned selector would have moved thirty modules off the un-optioned
        // one and left Metal's two selectors agreeing untested.
        //
        // One arm a plane and no `A | B` arm or binding: the coverage generator gates a call by
        // the variants of the arms around it, and a call under several variants at once is only
        // attributed to a module that names **all** of them, so an arm over three planes would
        // claim the optioned selector for every module and name none of them.
        match plane {
            // SAFETY: the buffer holds `bytes` from `offset`, checked just above, and the region
            // is the whole extent of the level, slice and
            // plane the case named, of a texture the case made with that place in it.
            Plane::Whole => unsafe {
                encoder
                    .copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                        &texture,
                        at.slice as usize,
                        at.level as usize,
                        origin,
                        size,
                        &buffer,
                        offset as usize,
                        row_pitch as usize,
                        bytes as usize,
                        );
            },
            Plane::Depth => Self::read_texture_plane(
                &encoder,
                &texture,
                &buffer,
                at,
                origin,
                size,
                offset,
                row_pitch,
                bytes,
                MTLBlitOption::DepthFromDepthStencil,
            ),
            Plane::Stencil => Self::read_texture_plane(
                &encoder,
                &texture,
                &buffer,
                at,
                origin,
                size,
                offset,
                row_pitch,
                bytes,
                MTLBlitOption::StencilFromDepthStencil,
            ),
            Plane::RowLinearPvrtc => Self::read_texture_plane(
                &encoder,
                &texture,
                &buffer,
                at,
                origin,
                size,
                offset,
                row_pitch,
                bytes,
                MTLBlitOption::RowLinearPVRTC,
            ),
        }
        encoder.endEncoding();
        Ok(())
    }

    /// One texture-to-buffer blit through Metal's **optioned** selector, under `option`.
    ///
    /// Its own function rather than a third arm of the `match` above because each arm there is
    /// one plane: the coverage generator gates a call by the variants of the arms around it, and
    /// a call under `A | B | C`, or under a closure defined outside the `match`, would claim the
    /// optioned selector for every module that reads a texture at all.
    #[allow(clippy::too_many_arguments)]
    fn read_texture_plane(
        encoder: &ProtocolObject<dyn MTLBlitCommandEncoder>,
        texture: &ProtocolObject<dyn MTLTexture>,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        at: crate::render::ReadAt,
        origin: MTLOrigin,
        size: MTLSize,
        offset: u64,
        row_pitch: u32,
        bytes: u64,
        option: MTLBlitOption,
    ) {
        // SAFETY: as [`MetalSide::encode_read_texture`] states: the buffer holds `bytes` from
        // `offset`, which it checked, and the region is the whole extent of
        // the level, slice and plane the case named.
        unsafe {
            encoder
                .copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                    texture,
                    at.slice as usize,
                    at.level as usize,
                    origin,
                    size,
                    buffer,
                    offset as usize,
                    row_pitch as usize,
                    bytes as usize,
                    option,
                );
        }
    }
}

/// A command-buffer handler block the probes keep alive: the owning half of
/// [`MTLCommandBufferHandler`], which is the raw pointer Metal's setters take.
type HandlerBlock =
    block2::RcBlock<dyn Fn(std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>)>;

/// A command buffer a probe waits for the scheduling of on a thread of its own.
///
/// `objc2` gives `MTLCommandBuffer` no `Send`, because Metal documents a command buffer as unsafe
/// to **encode into** from more than one thread. The probe encodes into none on that thread: it
/// calls `waitUntilScheduled`, which Metal documents as a wait any thread may make, and then
/// releases the reference. Both are sound off the thread that made it, since the retain count is
/// atomic.
struct Waited(Retained<ProtocolObject<dyn MTLCommandBuffer>>);

// SAFETY: as the type's documentation states: the probe never encodes into it on the waiting
// thread, and the only calls that cross threads are `waitUntilScheduled` and the release.
unsafe impl Send for Waited {}

impl Waited {
    /// `waitUntilScheduled`.
    ///
    /// A method rather than a field read at the call site, and that is load-bearing: a closure
    /// reaching `waited.0` captures the `Retained` on its own, which is not `Send`, so the whole
    /// wrapper — the thing this type's `Send` is about — would never be what crossed the thread.
    fn wait_until_scheduled(&self) {
        self.0.waitUntilScheduled();
    }
}

/// A command buffer's completion, as a bound the probe's own thread can wait on.
///
/// Metal has **no bounded wait**: `waitUntilCompleted` either returns or does not. So an event
/// probe, whose whole subject is a command buffer held back by a value that has not arrived, never
/// calls it — it adds a completed handler before the commit and waits on this channel with a bound.
/// A bound that runs out is a recorded fact ([`crate::event_orders::Status::NeverSettled`]), not a
/// hang.
struct Completion {
    done: std::sync::mpsc::Receiver<()>,
    /// The block Metal calls, kept alive for as long as the probe runs. `addCompletedHandler:`
    /// takes a pointer, and keeping our own reference means the block's lifetime does not rest on
    /// what Metal does with it.
    _block: HandlerBlock,
}

/// A completed handler on `command_buffer` that signals its channel, added before the commit.
fn completion_channel(command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> Completion {
    let (sender, done) = std::sync::mpsc::channel();
    let block = block2::RcBlock::new(
        move |_: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            let _ = sender.send(());
        },
    );
    // SAFETY: the block is a valid heap block, kept alive in the `Completion` this answers for as
    // long as the probe runs, which outlives every call Metal makes of it.
    unsafe { command_buffer.addCompletedHandler(block2::RcBlock::as_ptr(&block)) };
    Completion {
        done,
        _block: block,
    }
}

/// The status `command_buffer` reads once its completion arrived, or that it never did.
fn settled(
    completion: &Completion,
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
) -> crate::event_orders::Status {
    if completion
        .done
        .recv_timeout(crate::event_orders::COMPLETION)
        .is_err()
    {
        crate::event_orders::Status::NeverSettled
    } else {
        crate::event_orders::Status::read(status_word(command_buffer.status()))
    }
}

/// The names a command buffer's handlers logged, in the order Metal called them, with each name
/// also sent so the probe can wait for one rather than guess when the log is whole.
///
/// The channel is what makes the log readable: Metal fixes no order between two handlers of one
/// command buffer that this probe could rely on, so the probe waits for the name it cares about to
/// arrive instead of assuming the handler it added first is the one that ran first.
struct HandlerLog {
    called: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    sender: std::sync::mpsc::Sender<String>,
    arrived: std::sync::mpsc::Receiver<String>,
    /// The blocks Metal calls, kept alive for as long as the probe runs.
    blocks: Vec<HandlerBlock>,
}

impl HandlerLog {
    /// An empty log.
    fn new() -> Self {
        let (sender, arrived) = std::sync::mpsc::channel();
        Self {
            called: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            sender,
            arrived,
            blocks: Vec::new(),
        }
    }

    /// Add a handler that logs itself as `name` to `command_buffer`, as the kind `add` names.
    ///
    /// `add` is the raw setter so one body serves both `addScheduledHandler:` and
    /// `addCompletedHandler:`.
    fn add(&mut self, name: &'static str, add: impl FnOnce(MTLCommandBufferHandler)) {
        let (called, sender) = (std::sync::Arc::clone(&self.called), self.sender.clone());
        let block = block2::RcBlock::new(
            move |_: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                if let Ok(mut called) = called.lock() {
                    called.push(name.to_owned());
                }
                let _ = sender.send(name.to_owned());
            },
        );
        add(block2::RcBlock::as_ptr(&block));
        self.blocks.push(block);
    }

    /// Wait until the handler named `name` has been called, or `bound` runs out.
    fn wait_for(&self, name: &str, bound: std::time::Duration) {
        let deadline = std::time::Instant::now() + bound;
        loop {
            if self
                .called
                .lock()
                .is_ok_and(|called| called.iter().any(|called| called == name))
            {
                return;
            }
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() || self.arrived.recv_timeout(left).is_err() {
                return;
            }
        }
    }

    /// The names logged so far, in order.
    fn names(&self) -> Vec<String> {
        self.called
            .lock()
            .map(|called| called.clone())
            .unwrap_or_default()
    }
}

/// `detachBacking` of `buffer` (Apple SPI): whether the buffer now lies in no host pages.
///
/// A function rather than a bare `msg_send!` at the call site, for the reason [`conforming`]
/// states: a selector sent with no receiver type is a call the coverage derivation cannot read.
/// One place also states why the send is sound.
fn detach_backing(buffer: &ProtocolObject<dyn MTLBuffer>) -> bool {
    // SAFETY: the caller asked the buffer for this selector first
    // (`require_selector`, which every Apple SPI this suite sends goes through); it takes no
    // argument and answers a `BOOL`.
    unsafe { objc2::msg_send![buffer, detachBacking] }
}

/// The word a command buffer's status records as, in the runtime's spelling so the two sides'
/// transcripts hold one vocabulary.
pub(crate) fn status_word(status: MTLCommandBufferStatus) -> &'static str {
    match status {
        MTLCommandBufferStatus::NotEnqueued => "NotEnqueued",
        MTLCommandBufferStatus::Enqueued => "Enqueued",
        MTLCommandBufferStatus::Committed => "Committed",
        MTLCommandBufferStatus::Scheduled => "Scheduled",
        MTLCommandBufferStatus::Completed => "Completed",
        MTLCommandBufferStatus::Error => "Error",
        _ => "Unknown",
    }
}

/// State every field of one attachment descriptor from `row`: the three actions and the six index
/// fields, which `MTLRenderPassAttachmentDescriptor` declares for all three kinds.
///
/// The parameter is the superclass the kinds share, so the calls here are the superclass's items
/// once rather than three times over its subclasses.
fn state_pass_attachment(attachment: &MTLRenderPassAttachmentDescriptor, row: &Row) {
    attachment.setLoadAction(MTLLoadAction(row.actions.load as usize));
    attachment.setStoreAction(MTLStoreAction(row.actions.store as usize));
    attachment.setStoreActionOptions(MTLStoreActionOptions::from_bits_retain(
        row.actions.options as usize,
    ));
    attachment.setLevel(row.indices.0[0] as usize);
    attachment.setSlice(row.indices.0[1] as usize);
    attachment.setDepthPlane(row.indices.0[2] as usize);
    attachment.setResolveLevel(row.indices.0[3] as usize);
    attachment.setResolveSlice(row.indices.0[4] as usize);
    attachment.setResolveDepthPlane(row.indices.0[5] as usize);
}

/// What one attachment descriptor answers, through the superclass all three kinds share.
fn read_pass_attachment(attachment: &MTLRenderPassAttachmentDescriptor) -> AttachmentFacts {
    // Each getter's answer is bound to a local of its declared type, as the other facts walks do:
    // the coverage derivation types a receiver from what bound it.
    let texture: Option<Retained<ProtocolObject<dyn MTLTexture>>> = attachment.texture();
    let resolve: Option<Retained<ProtocolObject<dyn MTLTexture>>> = attachment.resolveTexture();
    AttachmentFacts {
        has_texture: texture.is_some(),
        has_resolve_texture: resolve.is_some(),
        indices: Indices([
            attachment.level() as u64,
            attachment.slice() as u64,
            attachment.depthPlane() as u64,
            attachment.resolveLevel() as u64,
            attachment.resolveSlice() as u64,
            attachment.resolveDepthPlane() as u64,
        ]),
        actions: crate::pass_descriptors::Actions {
            load: attachment.loadAction().0 as u64,
            store: attachment.storeAction().0 as u64,
            options: attachment.storeActionOptions().bits() as u64,
        },
    }
}

/// The colour attachment at `index` of `descriptor`'s array.
fn pass_color(
    descriptor: &MTLRenderPassDescriptor,
    index: u32,
) -> Retained<MTLRenderPassColorAttachmentDescriptor> {
    let array: Retained<MTLRenderPassColorAttachmentDescriptorArray> =
        descriptor.colorAttachments();
    // SAFETY: `index` is below `READ_COLORS`, the eight attachments Metal's array holds, which
    // every caller here keeps it to.
    unsafe { array.objectAtIndexedSubscript(index as usize) }
}

/// Metal's number of sample positions, and the positions themselves.
///
/// `getSamplePositions:count:` answers the count for a count of 0 and writes the array for a count
/// long enough, which is the only way to ask Metal how many the descriptor holds.
fn pass_positions(descriptor: &MTLRenderPassDescriptor) -> (u64, Vec<(u32, u32)>) {
    // SAFETY: a count of 0 with a null pointer is the documented way to ask for the count alone,
    // and nothing is written.
    let count = unsafe { descriptor.getSamplePositions_count(std::ptr::null_mut(), 0) };
    let mut held: Vec<MTLSamplePosition> = vec![MTLSamplePosition { x: 0.0, y: 0.0 }; count];
    if count > 0 {
        // SAFETY: the pointer is to `count` positions that live for the call, and the count is the
        // one Metal has just answered.
        unsafe {
            descriptor.getSamplePositions_count(held.as_mut_ptr(), count);
        }
    }
    (
        count as u64,
        held.iter()
            .map(|position| (position.x.to_bits(), position.y.to_bits()))
            .collect(),
    )
}

/// The pass's own eight numbers, in [`Numbers`]' order.
fn pass_numbers(descriptor: &MTLRenderPassDescriptor) -> Numbers {
    Numbers([
        descriptor.renderTargetArrayLength() as u64,
        descriptor.renderTargetWidth() as u64,
        descriptor.renderTargetHeight() as u64,
        descriptor.defaultRasterSampleCount() as u64,
        descriptor.tileWidth() as u64,
        descriptor.tileHeight() as u64,
        descriptor.imageblockSampleLength() as u64,
        descriptor.threadgroupMemoryLength() as u64,
    ])
}

/// What `descriptor` answers under `reads`.
fn read_pass_descriptor(descriptor: &MTLRenderPassDescriptor, reads: Reads) -> PassDescriptorFacts {
    let mut facts = PassDescriptorFacts::empty();
    for index in 0..reads.colors() {
        let color: Retained<MTLRenderPassColorAttachmentDescriptor> = pass_color(descriptor, index);
        let clear: MTLClearColor = color.clearColor();
        if matches!(reads, Reads::Whole | Reads::Attachments) {
            facts.colors.push(read_pass_attachment(&color));
        }
        facts.color_clears.push([
            clear.red.to_bits(),
            clear.green.to_bits(),
            clear.blue.to_bits(),
            clear.alpha.to_bits(),
        ]);
    }
    if matches!(reads, Reads::Whole | Reads::Attachments | Reads::ClearDepth) {
        let depth: Retained<MTLRenderPassDepthAttachmentDescriptor> = descriptor.depthAttachment();
        facts.clear_depth = Some(depth.clearDepth().to_bits());
        if reads != Reads::ClearDepth {
            facts.depth = Some(read_pass_attachment(&depth));
            facts.depth_resolve_filter = Some(depth.depthResolveFilter().0 as u64);
        }
    }
    if matches!(
        reads,
        Reads::Whole | Reads::Attachments | Reads::ClearStencil
    ) {
        let stencil: Retained<MTLRenderPassStencilAttachmentDescriptor> =
            descriptor.stencilAttachment();
        facts.clear_stencil = Some(stencil.clearStencil());
        if reads != Reads::ClearStencil {
            facts.stencil = Some(read_pass_attachment(&stencil));
            facts.stencil_resolve_filter = Some(stencil.stencilResolveFilter().0 as u64);
        }
    }
    if matches!(reads, Reads::Whole | Reads::Numbers) {
        facts.numbers = Some(pass_numbers(descriptor));
        let visibility: Option<Retained<ProtocolObject<dyn MTLBuffer>>> =
            descriptor.visibilityResultBuffer();
        let map: Option<Retained<ProtocolObject<dyn MTLRasterizationRateMap>>> =
            descriptor.rasterizationRateMap();
        facts.has_visibility_buffer = Some(visibility.is_some());
        facts.has_rate_map = Some(map.is_some());
    }
    if matches!(reads, Reads::Whole | Reads::Positions) {
        let (count, positions) = pass_positions(descriptor);
        facts.position_count = Some(count);
        facts.positions = Some(positions);
    }
    facts
}

/// `row`'s clear colour as Metal's structure.
fn clear_color_of(row: &Row) -> MTLClearColor {
    MTLClearColor {
        red: row.clear_color[0],
        green: row.clear_color[1],
        blue: row.clear_color[2],
        alpha: row.clear_color[3],
    }
}

/// [`Shape::Assigned`]: build each attachment apart from the pass, state it, assign it in, and
/// then mutate it **through the object that was assigned**.
///
/// What the pass answers afterwards says whether the assignment copied the descriptor. The
/// mutation states the values of another attachment of the stated table, so the two answers are
/// values the record already holds elsewhere and a reader can tell which one came back.
fn assign_pass_attachments(descriptor: &MTLRenderPassDescriptor) {
    let array: Retained<MTLRenderPassColorAttachmentDescriptorArray> =
        descriptor.colorAttachments();
    let color: Retained<MTLRenderPassColorAttachmentDescriptor> =
        MTLRenderPassColorAttachmentDescriptor::new();
    let assigned = crate::pass_descriptors::stated_row(ASSIGNED_COLOR);
    state_pass_attachment(&color, &assigned);
    color.setClearColor(clear_color_of(&assigned));
    // SAFETY: the slot is below the eight attachments the array holds.
    unsafe { array.setObject_atIndexedSubscript(Some(&color), ASSIGNED_COLOR as usize) };
    let after = crate::pass_descriptors::stated_row(MUTATED_AFTER);
    state_pass_attachment(&color, &after);
    color.setClearColor(clear_color_of(&after));

    let depth: Retained<MTLRenderPassDepthAttachmentDescriptor> =
        MTLRenderPassDepthAttachmentDescriptor::new();
    let assigned = crate::pass_descriptors::stated_row(DEPTH_AT);
    state_pass_attachment(&depth, &assigned);
    depth.setClearDepth(assigned.clear_depth);
    depth.setDepthResolveFilter(MTLMultisampleDepthResolveFilter(
        assigned.filters.depth as usize,
    ));
    descriptor.setDepthAttachment(Some(&depth));
    state_pass_attachment(&depth, &after);
    depth.setClearDepth(crate::pass_descriptors::BASE.clear_depth);

    let stencil: Retained<MTLRenderPassStencilAttachmentDescriptor> =
        MTLRenderPassStencilAttachmentDescriptor::new();
    let assigned = crate::pass_descriptors::stated_row(STENCIL_AT);
    state_pass_attachment(&stencil, &assigned);
    stencil.setClearStencil(assigned.clear_stencil);
    stencil.setStencilResolveFilter(MTLMultisampleStencilResolveFilter(
        assigned.filters.stencil as usize,
    ));
    descriptor.setStencilAttachment(Some(&stencil));
    state_pass_attachment(&stencil, &after);
    stencil.setClearStencil(crate::pass_descriptors::BASE.clear_stencil);
}

/// [`Shape::Cleared`]: set every attachment of the pass to nil.
///
/// `objc2-metal` 0.3.2 declares the three getters non-null, so a nil answered here would panic
/// rather than compare. That it does not is this shape's measurement: Metal answers a fresh
/// descriptor, which the read then records field by field.
fn clear_pass_attachments(descriptor: &MTLRenderPassDescriptor) {
    let array: Retained<MTLRenderPassColorAttachmentDescriptorArray> =
        descriptor.colorAttachments();
    for index in 0..PASS_COLORS {
        // SAFETY: the slot is below the eight attachments the array holds.
        unsafe { array.setObject_atIndexedSubscript(None, index as usize) };
    }
    descriptor.setDepthAttachment(None);
    descriptor.setStencilAttachment(None);
}

/// One `MTLRasterizationRateLayerDescriptor` holding a layer's quality samples.
///
/// `initWithSampleCount:horizontal:vertical:` reads `width` samples from `horizontal` and `height`
/// from `vertical` and answers nil for an axis of none, which is why [`RateMapLayer`] holds at
/// least one sample on each.
///
/// The other initializer, `initWithSampleCount:`, zero-fills both axes (measured on an Apple M3
/// Max: `maxSampleCount` and `sampleCount` both read the count it was given, and every sample of
/// `horizontalSampleStorage` and of `horizontal` reads 0). Nothing here uses it, because a layer
/// built that way and then filled through `horizontal[i] = q` reaches the same state this reaches
/// in one call, and a case comparing the two would be comparing two spellings of one Metal state
/// rather than anything Metal derives.
fn layer_descriptor(layer: &RateMapLayer) -> Answer<Retained<MTLRasterizationRateLayerDescriptor>> {
    let count = MTLSize {
        width: layer.horizontal.len(),
        height: layer.vertical.len(),
        depth: 0,
    };
    // SAFETY: each pointer is to at least as many floats as its axis's count, and Metal copies the
    // samples during the call.
    Ok(unsafe {
        MTLRasterizationRateLayerDescriptor::initWithSampleCount_horizontal_vertical(
            MTLRasterizationRateLayerDescriptor::alloc(),
            count,
            std::ptr::NonNull::new(layer.horizontal.as_ptr().cast_mut())
                .ok_or_else(|| Failure::named("NoSamples", "a layer axis of none"))?,
            std::ptr::NonNull::new(layer.vertical.as_ptr().cast_mut())
                .ok_or_else(|| Failure::named("NoSamples", "a layer axis of none"))?,
        )
    })
}

/// What a compute pipeline state answers of itself, read off the state rather than off a handle.
///
/// A free function because a creation handler reads these getters **inside the block**, on
/// Metal's own thread, from a raw pointer it was given: there is no handle there and nothing to
/// look up. [`Side::compute_pipeline_facts`] is the same reading through the case's pool, so the
/// two can never drift.
fn compute_state_facts(
    state: &ProtocolObject<dyn MTLComputePipelineState>,
    registry: u64,
    imageblock: [u32; 3],
) -> ComputeStateFacts {
    let required = state.requiredThreadsPerThreadgroup();
    ComputeStateFacts {
        label: state.label().map(|label| label.to_string()),
        // The registry ID is each host's own number, so only the **comparison** is portable.
        device_is_the_cases: state.device().registryID() == registry,
        static_threadgroup_memory_length: state.staticThreadgroupMemoryLength() as u64,
        support_indirect_command_buffers: state.supportIndirectCommandBuffers(),
        // SAFETY: the dimensions are the case's own three positive extents, which the call
        // multiplies by the kernel's imageblock cell size and nothing else.
        imageblock_memory_length: unsafe {
            state.imageblockMemoryLengthForDimensions(size(imageblock))
        } as u64,
        required_threads_per_threadgroup: [
            required.width as u64,
            required.height as u64,
            required.depth as u64,
        ],
    }
}

/// What a drawing render pipeline state answers of itself, read off the state.
///
/// A free function for the reason [`compute_state_facts`] gives.
fn render_state_facts(
    state: &ProtocolObject<dyn MTLRenderPipelineState>,
    registry: u64,
    imageblock: [u32; 3],
) -> RenderStateFacts {
    let required = state.requiredThreadsPerTileThreadgroup();
    RenderStateFacts {
        label: state.label().map(|label| label.to_string()),
        device_is_the_cases: state.device().registryID() == registry,
        max_total_threads_per_threadgroup: state.maxTotalThreadsPerThreadgroup() as u64,
        threadgroup_size_matches_tile_size: state.threadgroupSizeMatchesTileSize(),
        imageblock_sample_length: Some(state.imageblockSampleLength() as u64),
        imageblock_memory_length: Some(
            state.imageblockMemoryLengthForDimensions(size(imageblock)) as u64
        ),
        support_indirect_command_buffers: state.supportIndirectCommandBuffers(),
        required_threads_per_tile_threadgroup: [
            required.width as u64,
            required.height as u64,
            required.depth as u64,
        ],
    }
}

/// The three getters a **tile** state answers, read off the state.
///
/// A free function for the reason [`compute_state_facts`] gives.
fn tile_state_facts(state: &ProtocolObject<dyn MTLRenderPipelineState>) -> TileFacts {
    TileFacts {
        max_total_threads_per_threadgroup: state.maxTotalThreadsPerThreadgroup() as u64,
        threadgroup_size_matches_tile_size: state.threadgroupSizeMatchesTileSize(),
        imageblock_sample_length: Some(state.imageblockSampleLength() as u64),
    }
}

/// Which kind of state a `MTLRenderPipelineState` handler was given, which is what decides the
/// getters its facts are read from.
///
/// A tile state and a drawing state are one Metal object made from two descriptors, and
/// [`crate::pipeline_facts`] states which getters each answers, so the handler has to be told
/// which descriptor it came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RenderKind {
    Drawing,
    Tile,
}

/// What a render-state creation handler was given, as the vocabulary records it.
///
/// `reflected` is `Some(whether a reflection came)` for a form with an options parameter and
/// `None` for one without, which is the distinction
/// [`Delivered::reflection`](crate::pipeline_handlers::Delivered::reflection) holds.
fn render_delivery(
    state: *mut ProtocolObject<dyn MTLRenderPipelineState>,
    reflected: Option<bool>,
    error: *mut NSError,
    registry: u64,
    kind: RenderKind,
) -> Delivered {
    // SAFETY: the pointer is the one Metal handed this handler. Metal's contract for these blocks
    // is that it is either null or a state that is valid for the call, and nothing here keeps a
    // reference past the facts it reads.
    match unsafe { state.as_ref() } {
        Some(state) => {
            let facts = match kind {
                RenderKind::Drawing => {
                    StateFacts::Render(render_state_facts(state, registry, HANDLER_IMAGEBLOCK))
                }
                RenderKind::Tile => StateFacts::Tile(tile_state_facts(state)),
            };
            match reflected {
                Some(held) => Delivered::reflected(facts, held),
                None => Delivered::made(facts),
            }
        }
        None => Delivered {
            state: false,
            error: !error.is_null(),
            reflection: reflected,
            facts: None,
        },
    }
}

/// What a compute-state creation handler was given, as the vocabulary records it.
fn compute_delivery(
    state: *mut ProtocolObject<dyn MTLComputePipelineState>,
    reflected: Option<bool>,
    error: *mut NSError,
    registry: u64,
) -> Delivered {
    // SAFETY: as [`render_delivery`]'s.
    match unsafe { state.as_ref() } {
        Some(state) => {
            let facts =
                StateFacts::Compute(compute_state_facts(state, registry, HANDLER_IMAGEBLOCK));
            match reflected {
                Some(held) => Delivered::reflected(facts, held),
                None => Delivered::made(facts),
            }
        }
        None => Delivered {
            state: false,
            error: !error.is_null(),
            reflection: reflected,
            facts: None,
        },
    }
}

/// The handler's answer, waited for on its **own channel** with a bound.
///
/// A timeout is recorded as
/// [`Arrival::NeverArrived`](crate::pipeline_handlers::Arrival::NeverArrived) and never waited for
/// again: this is the one place the family could hang, and it cannot.
fn handler_answer(answered: &std::sync::mpsc::Receiver<Delivered>) -> Handled {
    match answered.recv_timeout(HANDLER_WAIT) {
        Ok(delivered) => Handled::answered(delivered),
        Err(_) => Handled::never_arrived(),
    }
}

/// One of a program's libraries as Metal's side built it: the `MTLLibrary`, and the AIR object
/// `xcrun metal -dynamiclib` makes its dynamic form from.
struct BuiltLibrary {
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    /// The `.air` object `xcrun metal-as` assembled.
    air: std::path::PathBuf,
    /// The executable metallib `xcrun metallib` packaged.
    metallib: std::path::PathBuf,
}

/// The objects a program made, by the index its steps name them with. `None` where the creation
/// was refused, which later steps that name it then observe as their own refusal.
struct Held {
    archives: Vec<Option<Retained<ProtocolObject<dyn MTLBinaryArchive>>>>,
    from_data: Vec<Option<Retained<ProtocolObject<dyn MTLLibrary>>>>,
    dynamics: Vec<Option<Retained<ProtocolObject<dyn MTLDynamicLibrary>>>>,
}

/// Run an archive program on Metal ([`Side::archives`]).
///
/// A free function rather than a method, because every step needs only the device and the scratch
/// directory and the objects it makes are the program's own: nothing here joins one of the side's
/// handle pools, since no handle in this vocabulary names an archive or a dynamic library.
fn metal_program(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    scratch: &Scratch,
    program: &Program<'_>,
) -> Answer<Vec<Observation>> {
    let mut built = Vec::with_capacity(program.libraries.len());
    for (at, entry) in program.libraries.iter().enumerate() {
        built.push(metal_build_library(device, scratch, at, &entry.source)?);
    }
    let registry = device.registryID();
    let mut held = Held {
        archives: Vec::new(),
        from_data: Vec::new(),
        dynamics: Vec::new(),
    };
    // A serial for the dynamic-library files a `Bytes::DynamicLibraryFile` step writes, so two
    // steps of one program never build over each other's output.
    let mut serial = 0usize;
    let mut out = Vec::with_capacity(program.steps.len());
    for (_, step) in program.steps {
        out.push(metal_step(
            device,
            scratch,
            program,
            &built,
            registry,
            &mut held,
            &mut serial,
            step,
        )?);
    }
    Ok(out)
}

/// One of a program's libraries, assembled and packaged with Apple's own tools, as
/// [`Side::library`] packages a case's shader.
fn metal_build_library(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    scratch: &Scratch,
    at: usize,
    source: &ShaderSource<'_>,
) -> Answer<BuiltLibrary> {
    let text = scratch.0.join(format!("program-{at}.air.ll"));
    let air = scratch.0.join(format!("program-{at}.air"));
    let metallib = scratch.0.join(format!("program-{at}.metallib"));
    let build = || -> Result<(), String> {
        std::fs::write(&text, source.air_text)
            .map_err(|error| format!("{}: {error}", text.display()))?;
        xcrun(
            "metal-as",
            &[text.as_os_str(), "-o".as_ref(), air.as_os_str()],
        )?;
        xcrun(
            "metallib",
            &[air.as_os_str(), "-o".as_ref(), metallib.as_os_str()],
        )
    };
    build().map_err(|why| Failure::named("MetalToolFailed", why))?;
    let url = file_url(&metallib)?;
    let library = device
        .newLibraryWithURL_error(&url)
        .map_err(|error| Failure::named("NewLibraryWithURL", error.to_string()))?;
    Ok(BuiltLibrary {
        library,
        air,
        metallib,
    })
}

/// The `NSURL` of a path.
fn file_url(path: &std::path::Path) -> Answer<Retained<NSURL>> {
    let text = path
        .to_str()
        .ok_or_else(|| Failure::named("NonUtf8Path", path.display().to_string()))?;
    Ok(NSURL::fileURLWithPath(&NSString::from_str(text)))
}

/// The bytes a step names, built with Apple's own tools where it names a container.
fn metal_bytes(
    scratch: &Scratch,
    built: &[BuiltLibrary],
    serial: &mut usize,
    bytes: Bytes,
) -> Answer<Vec<u8>> {
    let library = |at: usize| -> Answer<&BuiltLibrary> {
        built
            .get(at)
            .ok_or_else(|| Failure::named("NoSuchProgramLibrary", at.to_string()))
    };
    match bytes {
        Bytes::Metallib { library: at } => {
            let path = &library(at)?.metallib;
            std::fs::read(path).map_err(|error| {
                Failure::named("MetallibNotRead", format!("{}: {error}", path.display()))
            })
        }
        Bytes::DynamicLibraryFile {
            library: at,
            install_name,
        } => {
            let air = library(at)?.air.clone();
            let out = scratch.0.join(format!("dynamic-{serial}.metallib"));
            *serial += 1;
            xcrun(
                "metal",
                &[
                    "-dynamiclib".as_ref(),
                    "-install_name".as_ref(),
                    install_name.as_ref(),
                    air.as_os_str(),
                    "-o".as_ref(),
                    out.as_os_str(),
                ],
            )
            .map_err(|why| Failure::named("MetalToolFailed", why))?;
            std::fs::read(&out).map_err(|error| {
                Failure::named(
                    "DynamicLibraryNotRead",
                    format!("{}: {error}", out.display()),
                )
            })
        }
        Bytes::Text(text) => Ok(text.as_bytes().to_vec()),
        Bytes::Empty => Ok(Vec::new()),
    }
}

/// Metal's compute pipeline descriptor for what a step names, with `archives` listed.
fn metal_compute_descriptor(
    built: &[BuiltLibrary],
    archives: &NSArray<ProtocolObject<dyn MTLBinaryArchive>>,
    at: usize,
    kernel: &str,
) -> Answer<Retained<MTLComputePipelineDescriptor>> {
    let descriptor = MTLComputePipelineDescriptor::new();
    let function = metal_function(built, at, kernel)?;
    descriptor.setComputeFunction(Some(&function));
    descriptor.setBinaryArchives(Some(archives));
    Ok(descriptor)
}

/// Metal's render pipeline descriptor for what a step names, with `archives` listed and
/// `attachments` colour attachments of the program's format.
fn metal_render_descriptor(
    built: &[BuiltLibrary],
    program: &Program<'_>,
    archives: &NSArray<ProtocolObject<dyn MTLBinaryArchive>>,
    at: usize,
    vertex: &str,
    fragment: &str,
    attachments: usize,
) -> Answer<Retained<MTLRenderPipelineDescriptor>> {
    let descriptor = MTLRenderPipelineDescriptor::new();
    let (vertex, fragment) = (
        metal_function(built, at, vertex)?,
        metal_function(built, at, fragment)?,
    );
    descriptor.setVertexFunction(Some(&vertex));
    descriptor.setFragmentFunction(Some(&fragment));
    let colors = descriptor.colorAttachments();
    for index in 0..attachments {
        // SAFETY: every index a case states is inside the fixed-size array each descriptor holds,
        // which `tests::archive_programs_declare_attachments_metal_holds` keeps true.
        let color = unsafe { colors.objectAtIndexedSubscript(index) };
        color.setPixelFormat(MTLPixelFormat(program.attachment.as_raw() as usize));
    }
    descriptor.setBinaryArchives(Some(archives));
    Ok(descriptor)
}

/// `newFunctionWithName:` of one of the program's libraries.
fn metal_function(
    built: &[BuiltLibrary],
    at: usize,
    name: &str,
) -> Answer<Retained<ProtocolObject<dyn MTLFunction>>> {
    built
        .get(at)
        .ok_or_else(|| Failure::named("NoSuchProgramLibrary", at.to_string()))?
        .library
        .newFunctionWithName(&NSString::from_str(name))
        .ok_or_else(|| Failure::named("NoSuchFunction", name.to_owned()))
}

/// One step of a program on Metal.
#[allow(clippy::too_many_arguments)]
fn metal_step(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    scratch: &Scratch,
    program: &Program<'_>,
    built: &[BuiltLibrary],
    registry: u64,
    held: &mut Held,
    serial: &mut usize,
    step: &Step,
) -> Answer<Observation> {
    let listed = |held: &Held,
                  indices: &[usize]|
     -> Option<Retained<NSArray<ProtocolObject<dyn MTLBinaryArchive>>>> {
        let mut found = Vec::with_capacity(indices.len());
        for index in indices {
            found.push(held.archives.get(*index)?.clone()?);
        }
        Some(NSArray::from_retained_slice(&found))
    };
    match *step {
        Step::NewArchive { from } => {
            let descriptor = MTLBinaryArchiveDescriptor::new();
            if let Some(source) = from {
                let url = file_url(&scratch.source(source))?;
                descriptor.setUrl(Some(&url));
            }
            let made = device
                .newBinaryArchiveWithDescriptor_error(&descriptor)
                .ok();
            let refused = made.is_none();
            held.archives.push(made);
            Ok(Observation::Made { refused })
        }
        Step::ArchiveLabel { archive, set } => {
            let Some(Some(archive)) = held.archives.get(archive) else {
                return Ok(Observation::refused_for(step));
            };
            let archive: &Retained<ProtocolObject<dyn MTLBinaryArchive>> = archive;
            if let Some(label) = set {
                archive.setLabel(Some(&NSString::from_str(label)));
            }
            Ok(Observation::Label(
                archive.label().map(|label| label.to_string()),
            ))
        }
        Step::ArchiveDevice { archive } => {
            let Some(Some(archive)) = held.archives.get(archive) else {
                return Ok(Observation::refused_for(step));
            };
            let archive: &Retained<ProtocolObject<dyn MTLBinaryArchive>> = archive;
            Ok(Observation::Flag(archive.device().registryID() == registry))
        }
        Step::Add { archive, what } => {
            let Some(Some(archive)) = held.archives.get(archive).cloned() else {
                return Ok(Observation::refused_for(step));
            };
            let none = NSArray::from_retained_slice(&[]);
            let refused = match what {
                Added::Compute { library, kernel } => {
                    let descriptor = metal_compute_descriptor(built, &none, library, kernel)?;
                    archive
                        .addComputePipelineFunctionsWithDescriptor_error(&descriptor)
                        .is_err()
                }
                Added::Render {
                    library,
                    vertex,
                    fragment,
                    attachments,
                } => {
                    let descriptor = metal_render_descriptor(
                        built,
                        program,
                        &none,
                        library,
                        vertex,
                        fragment,
                        attachments,
                    )?;
                    archive
                        .addRenderPipelineFunctionsWithDescriptor_error(&descriptor)
                        .is_err()
                }
                Added::Tile { library, tile } => {
                    let descriptor = MTLTileRenderPipelineDescriptor::new();
                    let tile = metal_function(built, library, tile)?;
                    // SAFETY: the function is of one of the program's own libraries, and the
                    // descriptor is this step's own; nothing else holds it.
                    unsafe { descriptor.setTileFunction(&tile) };
                    let colors: Retained<MTLTileRenderPipelineColorAttachmentDescriptorArray> =
                        descriptor.colorAttachments();
                    // SAFETY: attachment 0 is inside the fixed-size array the descriptor holds.
                    let color = unsafe { colors.objectAtIndexedSubscript(0) };
                    color.setPixelFormat(MTLPixelFormat(program.attachment.as_raw() as usize));
                    archive
                        .addTileRenderPipelineFunctionsWithDescriptor_error(&descriptor)
                        .is_err()
                }
                Added::Mesh {
                    library,
                    object,
                    mesh,
                    fragment,
                } => {
                    let descriptor = MTLMeshRenderPipelineDescriptor::new();
                    let (object, mesh, fragment) = (
                        metal_function(built, library, object)?,
                        metal_function(built, library, mesh)?,
                        metal_function(built, library, fragment)?,
                    );
                    // SAFETY: the three functions are of one of the program's own libraries, and
                    // the descriptor is this step's own; nothing else holds it.
                    unsafe {
                        descriptor.setObjectFunction(Some(&object));
                        descriptor.setMeshFunction(Some(&mesh));
                        descriptor.setFragmentFunction(Some(&fragment));
                    }
                    let colors: Retained<MTLRenderPipelineColorAttachmentDescriptorArray> =
                        descriptor.colorAttachments();
                    // SAFETY: attachment 0 is inside the fixed-size array the descriptor holds.
                    let color = unsafe { colors.objectAtIndexedSubscript(0) };
                    color.setPixelFormat(MTLPixelFormat(program.attachment.as_raw() as usize));
                    archive
                        .addMeshRenderPipelineFunctionsWithDescriptor_error(&descriptor)
                        .is_err()
                }
            };
            Ok(Observation::Made { refused })
        }
        Step::Serialize { archive, to } => {
            let Some(Some(archive)) = held.archives.get(archive) else {
                return Ok(Observation::refused_for(step));
            };
            let archive: &Retained<ProtocolObject<dyn MTLBinaryArchive>> = archive;
            let url = file_url(&scratch.path(to))?;
            let refused = archive.serializeToURL_error(&url).is_err();
            Ok(Observation::Made { refused })
        }
        Step::Create {
            archives,
            options,
            what,
        } => {
            let Some(archives) = listed(held, archives) else {
                return Ok(Observation::refused_for(step));
            };
            let option = MTLPipelineOption(options as NSUInteger);
            let refused = match what {
                Created::Compute { library, kernel } => {
                    let descriptor = metal_compute_descriptor(built, &archives, library, kernel)?;
                    device
                        .newComputePipelineStateWithDescriptor_options_reflection_error(
                            &descriptor,
                            option,
                            None,
                        )
                        .is_err()
                }
                Created::Render {
                    library,
                    vertex,
                    fragment,
                    attachments,
                } => {
                    let descriptor = metal_render_descriptor(
                        built,
                        program,
                        &archives,
                        library,
                        vertex,
                        fragment,
                        attachments,
                    )?;
                    device
                        .newRenderPipelineStateWithDescriptor_options_reflection_error(
                            &descriptor,
                            option,
                            None,
                        )
                        .is_err()
                }
            };
            Ok(Observation::Made { refused })
        }
        Step::FileState { file } => Ok(scratch.state(file)),
        Step::Write { file, bytes } => {
            let written = metal_bytes(scratch, built, serial, bytes)?;
            let refused = std::fs::write(scratch.path(file), &written).is_err();
            Ok(Observation::Made { refused })
        }
        Step::NewLibraryWithData { bytes } => {
            let data = metal_bytes(scratch, built, serial, bytes)?;
            let made = device
                .newLibraryWithData_error(&dispatch2::DispatchData::from_bytes(&data))
                .ok();
            let facts = match &made {
                Some(library) => Some(metal_library_facts(library, registry)?),
                None => None,
            };
            let refused = made.is_none();
            held.from_data.push(made);
            Ok(Observation::Library { refused, facts })
        }
        Step::NewDynamicLibrary { from } => {
            let made = match from {
                DynamicSource::Library(at) => {
                    let library = match at {
                        LibraryAt::Program(index) => built
                            .get(index)
                            .map(|built| built.library.clone())
                            .ok_or_else(|| {
                                Failure::named("NoSuchProgramLibrary", index.to_string())
                            })?,
                        LibraryAt::FromData(index) => {
                            match held.from_data.get(index).cloned().flatten() {
                                Some(library) => library,
                                None => {
                                    held.dynamics.push(None);
                                    return Ok(Observation::refused_for(step));
                                }
                            }
                        }
                    };
                    device.newDynamicLibrary_error(&library).ok()
                }
                DynamicSource::Url(source) => {
                    let url = file_url(&scratch.source(source))?;
                    device.newDynamicLibraryWithURL_error(&url).ok()
                }
            };
            let install_name = match &made {
                Some(dynamic) => {
                    let dynamic: &Retained<ProtocolObject<dyn MTLDynamicLibrary>> = dynamic;
                    Some(dynamic.installName().to_string())
                }
                None => None,
            };
            let refused = made.is_none();
            held.dynamics.push(made);
            Ok(Observation::Dynamic {
                refused,
                install_name,
            })
        }
        Step::DynamicLabel { dynamic, set } => {
            let Some(Some(dynamic)) = held.dynamics.get(dynamic) else {
                return Ok(Observation::refused_for(step));
            };
            let dynamic: &Retained<ProtocolObject<dyn MTLDynamicLibrary>> = dynamic;
            if let Some(label) = set {
                dynamic.setLabel(Some(&NSString::from_str(label)));
            }
            Ok(Observation::Label(
                dynamic.label().map(|label| label.to_string()),
            ))
        }
        Step::DynamicDevice { dynamic } => {
            let Some(Some(dynamic)) = held.dynamics.get(dynamic) else {
                return Ok(Observation::refused_for(step));
            };
            let dynamic: &Retained<ProtocolObject<dyn MTLDynamicLibrary>> = dynamic;
            Ok(Observation::Flag(dynamic.device().registryID() == registry))
        }
        Step::SerializeDynamic { dynamic, to } => {
            let Some(Some(dynamic)) = held.dynamics.get(dynamic) else {
                return Ok(Observation::refused_for(step));
            };
            let dynamic: &Retained<ProtocolObject<dyn MTLDynamicLibrary>> = dynamic;
            let url = file_url(&scratch.path(to))?;
            let refused = dynamic.serializeToURL_error(&url).is_err();
            Ok(Observation::Made { refused })
        }
    }
}

/// A library's own four portable facts, as [`crate::library_facts`] compares them.
///
/// `functionNames` and nothing of a function itself: `newFunctionWithName:` of a **dynamic**
/// library's export stops at an assertion inside Metal (`TYPE` 3, measured on an M3 Max and an
/// M4), and a library made from data here may be a dynamic one, so this reads the names and never
/// asks for a function.
fn metal_library_facts(
    library: &Retained<ProtocolObject<dyn MTLLibrary>>,
    registry: u64,
) -> Answer<LibraryFacts> {
    let names = library.functionNames();
    let install = library.installName();
    Ok(LibraryFacts {
        function_names: names.iter().map(|name| name.to_string()).collect(),
        library_type: our_library_type(library.r#type().0)?,
        install_name: install.map(|name| name.to_string()),
        device_is_the_cases: library.device().registryID() == registry,
    })
}

/// The Metal buffer index the owned callers declare their `visible_function_table` at, which is
/// the index the encoder binds the case's table at. It is the fixtures' own `[[buffer(1)]]`.
const TABLE_INDEX: NSUInteger = 1;

/// One function a case asked a handle of, and what the state answered: nil where the stage does
/// not link it.
type AskedHandle<'a> = (
    &'a str,
    Option<Retained<ProtocolObject<dyn MTLFunctionHandle>>>,
);

/// The pipeline state a case's handles and table come from, on this side: a compute state, or a
/// render state and the one stage the case asks about.
///
/// Both of Metal's spellings behind one pair of methods, for the reason the runtime side's own
/// maker gives: the case's answer must not depend on which selector took it.
///
/// Each arm names its receiver's Metal type in a `let` rather than taking it straight out of the
/// pattern. That is for the coverage derivation, which types a receiver from a declaration and
/// attributes a selector two Metal types bind — `functionHandleWithFunction:`, which
/// `MTLComputePipelineState` and `MTLDevice` both bind — only where it can: an untyped receiver
/// costs the item its credit and reads in the generated document exactly like a selector no case
/// calls.
enum MetalMaker {
    Compute(Retained<ProtocolObject<dyn MTLComputePipelineState>>),
    Render(
        Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
        MTLRenderStages,
    ),
}

impl MetalMaker {
    /// `functionHandleWithFunction:`, or its `:stage:` spelling.
    fn function_handle(
        &self,
        function: &ProtocolObject<dyn MTLFunction>,
    ) -> Option<Retained<ProtocolObject<dyn MTLFunctionHandle>>> {
        match self {
            MetalMaker::Compute(state) => {
                let state: &ProtocolObject<dyn MTLComputePipelineState> = state;
                state.functionHandleWithFunction(function)
            }
            // SAFETY: the function is a `[[visible]]` function of the case's own library, whose
            // argument and return types are the table's own, and `stage` is one stage or two —
            // never the empty set, on which plain Metal faults.
            MetalMaker::Render(state, stage) => unsafe {
                let state: &ProtocolObject<dyn MTLRenderPipelineState> = state;
                state.functionHandleWithFunction_stage(function, *stage)
            },
        }
    }

    /// `newVisibleFunctionTableWithDescriptor:`, or its `:stage:` spelling.
    fn new_visible_function_table(
        &self,
        descriptor: &MTLVisibleFunctionTableDescriptor,
    ) -> Option<Retained<ProtocolObject<dyn MTLVisibleFunctionTable>>> {
        match self {
            MetalMaker::Compute(state) => {
                let state: &ProtocolObject<dyn MTLComputePipelineState> = state;
                state.newVisibleFunctionTableWithDescriptor(descriptor)
            }
            MetalMaker::Render(state, stage) => {
                let state: &ProtocolObject<dyn MTLRenderPipelineState> = state;
                state.newVisibleFunctionTableWithDescriptor_stage(descriptor, *stage)
            }
        }
    }
}
