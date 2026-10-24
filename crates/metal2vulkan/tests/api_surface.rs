//! The frozen public surface of the new API.
//!
//! The consumer surface is the items named at the root of each public module (`error`, `device`,
//! `library`, `reflection`, `pipeline`, `bind`, `cache`, `metal` with its four mapping modules, `vk`, and
//! `ash_interop` under feature `ash`), plus `Budget` and `CancelToken` at the crate root. An item
//! at a module root, or a submodule, marked `#[doc(hidden)]` is public but not API, and the table
//! names it `hidden` (there is none today; the crate root's hidden `env_vars` is the binary's). A
//! crate-private item (`pub(crate)`) is not a declaration of the surface: no cfg makes one
//! reachable.
//!
//! The `surface!` table below is hand-maintained. Two checks pin it:
//!
//! - At compile time, every public entry is named at its path, and every public type is
//!   `Send + Sync`.
//! - At run time, the module roots are read from source: the set of public items, hidden items,
//!   and public submodules must equal the table, and every public struct and enum must have the
//!   construction class the table states. A new or removed public item, or a type that gains or
//!   loses `#[non_exhaustive]`, fails here until the table says so.
//!
//! Construction classes: `N` is `#[non_exhaustive]` (fields and variants may grow within a minor
//! version); `O` is opaque (a private field, so consumers cannot build or destructure it); `E` is
//! exhaustive by design: a value or key whose fields are its identity, so a new field would be a
//! breaking change whatever the attribute, or an enum every consumer must interpret completely, so
//! a new variant is a compile error there.
//!
//! Method signatures are not pinned here; `compiled_functions_round_trip_through_the_public_api`
//! and `compiled_render_stages_round_trip_through_the_public_api` drive the main consumer flow
//! (compile, cache, link, plan, instantiate) through the public paths.
//!
//! The consumer reference (`scripts/api-docs`) renders exactly this table:
//! `the_reference_build_renders_exactly_the_surface` checks the crate root's documentation
//! switch, and `every_public_entry_is_documented` that every entry it renders has docs.
//!
//! The file reaches the crate only through its public paths, so it stays an integration test: from
//! outside the crate, naming an item at its path proves it public.
#![recursion_limit = "1024"]

mod common;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// How a consumer may build a struct or enum from outside the crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Class {
    /// `#[non_exhaustive]`.
    N,
    /// A private field (and no `#[non_exhaustive]`).
    O,
    /// Exhaustive by design.
    E,
}

/// What a root name is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    Struct(Class),
    Enum(Class),
    Trait,
    Fn,
    Const,
    Static,
    Type,
    /// A public submodule (a namespace of the surface).
    Mod,
    /// A `#[doc(hidden)]` root item or submodule: public, but not API.
    Hidden,
}

/// One root name: (module path below the crate root, name, kind).
type Entry = (&'static str, &'static str, Kind);

fn send_sync<T: Send + Sync + ?Sized>() {}

/// Expands the surface table into `surface_entries` and the compile-time checks.
macro_rules! surface {
    ($(
        $(#[$cfg:meta])*
        $module:literal => $($seg:ident)::+ { $($body:tt)* }
    )*) => {
        /// The frozen surface, module by module.
        fn surface_entries() -> Vec<Entry> {
            let mut v: Vec<Entry> = Vec::new();
            $(
                $(#[$cfg])*
                { surface!(@entries v $module; $($body)*); }
            )*
            v
        }

        /// Every public entry is named at its path, and every public type is `Send + Sync`.
        #[test]
        fn every_public_entry_is_nameable_and_types_are_send_and_sync() {
            $(
                $(#[$cfg])*
                { surface!(@check [$($seg)::+]; $($body)*); }
            )*
        }
    };

    (@entries $v:ident $m:literal; ) => {};
    (@entries $v:ident $m:literal; struct $n:ident $(<$($g:tt),+>)? $c:ident; $($rest:tt)*) => {
        $v.push(($m, stringify!($n), Kind::Struct(Class::$c)));
        surface!(@entries $v $m; $($rest)*);
    };
    (@entries $v:ident $m:literal; enum $n:ident $c:ident; $($rest:tt)*) => {
        $v.push(($m, stringify!($n), Kind::Enum(Class::$c)));
        surface!(@entries $v $m; $($rest)*);
    };
    (@entries $v:ident $m:literal; trait $n:ident; $($rest:tt)*) => {
        $v.push(($m, stringify!($n), Kind::Trait));
        surface!(@entries $v $m; $($rest)*);
    };
    (@entries $v:ident $m:literal; fn $n:ident; $($rest:tt)*) => {
        $v.push(($m, stringify!($n), Kind::Fn));
        surface!(@entries $v $m; $($rest)*);
    };
    (@entries $v:ident $m:literal; const $n:ident; $($rest:tt)*) => {
        $v.push(($m, stringify!($n), Kind::Const));
        surface!(@entries $v $m; $($rest)*);
    };
    (@entries $v:ident $m:literal; static $n:ident; $($rest:tt)*) => {
        $v.push(($m, stringify!($n), Kind::Static));
        surface!(@entries $v $m; $($rest)*);
    };
    (@entries $v:ident $m:literal; type $n:ident; $($rest:tt)*) => {
        $v.push(($m, stringify!($n), Kind::Type));
        surface!(@entries $v $m; $($rest)*);
    };
    (@entries $v:ident $m:literal; mod $n:ident; $($rest:tt)*) => {
        $v.push(($m, stringify!($n), Kind::Mod));
        surface!(@entries $v $m; $($rest)*);
    };
    (@entries $v:ident $m:literal; hidden $($n:ident),+; $($rest:tt)*) => {
        $( $v.push(($m, stringify!($n), Kind::Hidden)); )+
        surface!(@entries $v $m; $($rest)*);
    };

    (@check [$($p:ident)::+]; ) => {};
    (@check [$($p:ident)::+]; struct $n:ident $(<$($g:tt),+>)? $c:ident; $($rest:tt)*) => {
        send_sync::<$($p)::+::$n $(<$($g),+>)?>();
        surface!(@check [$($p)::+]; $($rest)*);
    };
    (@check [$($p:ident)::+]; enum $n:ident $c:ident; $($rest:tt)*) => {
        send_sync::<$($p)::+::$n>();
        surface!(@check [$($p)::+]; $($rest)*);
    };
    (@check [$($p:ident)::+]; type $n:ident; $($rest:tt)*) => {
        send_sync::<$($p)::+::$n>();
        surface!(@check [$($p)::+]; $($rest)*);
    };
    (@check [$($p:ident)::+]; hidden $($n:ident),+; $($rest:tt)*) => {
        surface!(@check [$($p)::+]; $($rest)*);
    };
    (@check [$($p:ident)::+]; $kind:tt $n:ident; $($rest:tt)*) => {
        {
            #[allow(unused_imports)]
            use $($p)::+::$n as _;
        }
        surface!(@check [$($p)::+]; $($rest)*);
    };
}

surface! {
    "" => metal2vulkan {
        struct Budget N;
        struct CancelToken O;
    }
    "error" => metal2vulkan::error {
        struct DiagName O;
        struct InputError N;
        enum InputErrorKind N;
        struct Location O;
        enum Attachment N;
        enum AbiError N;
        enum BitcodeError N;
        enum BitstreamError N;
        enum CanonError N;
        enum ContainerError N;
        enum ErrorClass N;
        enum IntrinsicError N;
        enum VerifyError N;
        enum Error N;
        enum Retry N;
        enum Permanence N;
        struct Unsupported N;
        enum Needs N;
        enum RuntimeDriverDefect N;
        enum DriverDeviation N;
        enum Capability N;
        enum Feature N;
        enum SparseNeed N;
        enum LinearNeed N;
        enum LowerFamily N;
        enum PrivateRenderField N;
        enum ExtendedRangeBlend N;
        enum RuntimeInput N;
        enum DescriptorError N;
        enum DynamicLibraryProblem N;
        enum LinkProblem N;
        enum Limit N;
        enum BudgetExceeded N;
        struct InternalError N;
        enum InternalKind N;
        enum BindError N;
    }
    "device" => metal2vulkan::device {
        fn vk_stage;
        struct Device O;
        struct DeviceDescription N;
        struct FormatProperties N;
        struct FormatSupport N;
        struct SubgroupProperties N;
        struct GeneratedCommandsProperties N;
        struct AllocationLimits N;
        struct EnableSet N;
        enum Exposure N;
        struct MetalDeviceProperties N;
        struct SimdSupport N;
        struct StageSimd N;
        enum ContainmentPolicy N;
        struct DevicePolicy N;
        enum Unorm8LinearFilterPolicy N;
        enum SamplingPolicy N;
        enum YCbCrFilterPolicy N;
        enum FloatMathsPolicy N;
        enum TexelConversionPolicy N;
        enum HalfRoundingPolicy N;
        enum NanMinMaxPolicy N;
        enum RoundingOrderPolicy N;
        enum HalfBlendPolicy N;
        enum HalfTexelBitsPolicy N;
        enum MultisampleDepthTwinPolicy N;
        enum DecodedTexelsPolicy N;
        enum HeapCompressionPolicy N;
        enum CompletionPolicy N;
        enum TileStorePolicy N;
        enum ResolvePolicy N;
        enum BatchOrderPolicy N;
        enum RenderLoweringPolicy N;
        struct DeviceProfile O;
        struct SpirvValidation N;
        struct EnabledCaps N;
        struct FragmentCaps N;
        struct ImageCaps N;
        struct FloatControls N;
        struct FloatWidthControls N;
        enum DriverWorkaround N;
        enum DriverFact N;
        struct DriverDiagnostic N;
        const NVIDIA_AMPERE_GA10X;
        enum GuestIsolation N;
        enum HandleDomain N;
        enum Independence N;
        struct Limits N;
        struct MeshLimits N;
        enum PointerDomain N;
        enum Containment N;
        struct DescriptorCaps N;
        enum FunctionStorage N;
        enum Robustness N;
        enum SimdClass N;
        struct StageSubgroup N;
        struct SubgroupModel N;
        struct SubgroupSizes E;
        enum Tier N;
        struct ApiVersion E;
        enum FloatControl N;
        enum FloatWidth N;
        enum FormatUsage N;
        struct Requirements N;
        enum ResolveMode N;
        struct SpirvCapability O;
        enum SpirvExtension N;
        enum StageCreateFlag N;
        enum SubgroupOp N;
        enum VkExtension N;
        enum VkFeature N;
        enum VkLimit N;
        enum VkProperty N;
        struct OpBits E;
    }
    "library" => metal2vulkan::library {
        struct CompiledFunction N;
        struct FunctionCompileKey N;
        struct Function N;
        struct FunctionConstant N;
        struct FunctionConstantValue N;
        struct FunctionConstantValues N;
        struct FunctionInfo<'static> N;
        enum FunctionStage N;
        struct Library N;
        enum LibraryType N;
        struct DynamicLibrary N;
    }
    "reflection" => metal2vulkan::reflection {
        struct ArrayType N;
        struct Attribute N;
        struct Binding N;
        struct BufferBinding N;
        struct ComputePipelineReflection N;
        struct DirectArgumentUse N;
        struct FunctionConstant N;
        struct FunctionProperties N;
        struct FunctionReflection N;
        struct LinkedUseAnswers N;
        enum LinkedUseGap N;
        struct ObjectPayloadBinding N;
        struct PointerType N;
        struct RenderPipelineReflection N;
        struct SlotWrite N;
        enum SlotWriteGap N;
        enum SlotWritability N;
        struct SlotWrites N;
        struct StructMember N;
        struct StructType N;
        struct TextureBinding N;
        struct TextureReferenceType N;
        struct ThreadgroupBinding N;
        struct UnwrittenFragmentInputs N;
    }
    "pipeline" => metal2vulkan::pipeline {
        fn heap_unused_while_pending;
        struct ComputePipeline N;
        struct ComputePipelineMetalInfo N;
        struct CompileReport N;
        struct EntryDescriptor N;
        enum EntryMath E;
        struct EntryPlan N;
        enum EntryRef E;
        struct LinkedSource O;
        struct StageEntries O;
        struct StandaloneModule N;
        enum PrecisionNote E;
        enum Reconvergence N;
        struct RenderPipeline N;
        struct RuntimeDependency N;
        struct SpecConstant N;
        struct SpecClause N;
        struct SpecTest N;
        struct SpecializationLayout N;
        struct SpecializedNote N;
        struct StageModule N;
        struct StrategyCensus N;
        struct StrategyUse N;
        struct ColorAttachmentDescriptor N;
        struct ComputePipelineDescriptor N;
        struct DynamicLibraries N;
        struct FunctionDescriptor N;
        struct LinkedFunction N;
        struct LinkedFunctions N;
        enum LinkedList N;
        struct MeshPipelineDescriptor N;
        struct PrivateRenderState N;
        struct RenderPipelineDescriptor N;
        struct StageInputAttributeDescriptor N;
        struct StageInputDescriptor N;
        struct StageInputLayoutDescriptor N;
        enum StageSource E;
        struct TessellationDescriptor N;
        struct TileRenderPipelineDescriptor N;
        struct VertexAttributeDescriptor N;
        struct VertexBufferLayoutDescriptor N;
        struct VertexDescriptor N;
        struct BlendState N;
        struct ColorAttachmentState N;
        struct FragmentImageblock N;
        struct GraphicsState N;
        enum ImageblockMechanism N;
        struct InputAttachmentUse N;
        struct ShaderTestsUse N;
        struct ShadowStencilUse N;
        struct StandInUse N;
        enum StandInView N;
        struct StorageAttachmentUse N;
        enum StorageAttachmentView E;
        struct TessellationState N;
        struct EmulatedTessellation N;
        struct AccessSet N;
        enum AttributeFetch E;
        enum BufferForm N;
        enum BufferKind N;
        struct BufferSlot N;
        struct CommandWrites E;
        enum DescriptorView N;
        struct MeshRecords N;
        enum NumberedSampler N;
        enum PatchDomain N;
        struct PipelineInterface N;
        struct PlacedField N;
        enum PushLoc N;
        struct PushPlacement N;
        struct RecordField N;
        enum RecordValue N;
        struct SamplerSlot N;
        enum SlotInterface N;
        struct TextureAccess N;
        struct TextureSlot N;
        struct ThreadgroupArgument N;
        enum Window N;
        enum BindingSource N;
        struct DescriptorBinding N;
        struct DescriptorSetLayout N;
        struct PipelineLayout N;
        struct PushConstantRange N;
        struct AliasView N;
        struct AmplificationDraws N;
        struct BlendEquation N;
        enum CaptureForm N;
        enum ChannelWrite N;
        struct ColorGroup N;
        struct ExtendedRangeColor N;
        enum FactorLane N;
        struct FetchedAttachment N;
        struct FramebufferFetchInputs N;
        enum HalfOutputLane N;
        struct LoweringInputs N;
        enum RotatedBlendForm N;
        struct StorageImageChoice N;
        enum StorageImageClass N;
        enum VertexFetch N;
        struct VertexFetchProjection N;
        struct ViewportCompensation N;
        struct FunctionHandle O;
        struct RenderPipelineKeys N;
        enum Mechanism N;
        enum RuntimeFact N;
        enum UniformViewChoice N;
        enum AccelerationStructureLevel N;
        struct AirUserKey E;
        enum Axis N;
        struct CombinedSampler N;
        enum HandleClass N;
        enum ImageAccess N;
        enum ImageDim N;
        struct ImageHandleClass N;
        enum PushField N;
        enum ResourceKind N;
        struct ResourceSlot E;
        enum SampledType N;
        enum SamplerRef N;
        enum SlotClass E;
        struct SlotRef E;
        enum SnapshotForm N;
        enum SpecKey N;
        enum Stage N;
        enum StorageFormatClass N;
        enum Symbol N;
        enum SyntheticBinding N;
        struct VaryingKey E;
        struct PulledAttribute N;
        struct VertexAttribute N;
        struct VertexBinding N;
        struct VertexInputState N;
        enum VertexStride E;
    }
    "bind" => metal2vulkan::bind {
        struct ArgumentBufferLayout N;
        struct ArgumentDescriptor N;
        enum ArgumentLayoutError N;
        struct ArgumentMember N;
        struct ArgumentPlace N;
        enum ArgumentType N;
        const ARGUMENT_HANDLE_BYTES;
        enum Dispatch N;
        struct DispatchCommands N;
        struct DispatchRegion N;
        enum IndirectArgs N;
        enum DrawCall N;
        struct DrawCommands N;
        enum IndexOverrun N;
        struct MeshDispatch N;
        enum MeshGroups N;
        enum MeshRecordsRegion N;
        struct TessellationDispatch N;
        enum TessellationRecords N;
        fn heap_classes;
        struct HeapElements O;
        enum HeapSource E;
        fn handle_entry;
        fn handle_entry_flags;
        fn heap_element;
        fn heap_raw_form;
        fn heap_write_nans_converted;
        enum HeapElement N;
        const HANDLE_ENTRY_BYTES;
        const HANDLE_TABLE_CAPACITY;
        const NEUTRAL_HEAP_ELEMENT;
        fn indirect_clamp_module;
        struct IndirectClamp N;
        struct IndirectClampModule N;
        struct BufferExtent N;
        struct BufferRange N;
        struct DescriptorBytes N;
        struct DescriptorRange N;
        struct IndexBufferRef N;
        struct IndirectRef N;
        struct PushConstantWrite N;
        enum PushHole E;
        struct ResidentRange N;
        struct ResidentSet O;
        struct RootTable N;
        enum NeutralResource N;
        fn sampler_desc;
        enum AccessClass N;
        struct BindingPlan N;
        struct ByteRange N;
        struct ComputeTemplate N;
        struct DescriptorWrite N;
        enum FieldLocation N;
        struct GraphicsTemplate N;
        enum ImageSource E;
        struct ImageViewDesc N;
        enum PlanDiagnostic N;
        struct PushFieldPlacement N;
        struct PushTemplate N;
        struct ResourceUse N;
        enum SamplerSource E;
        enum UseTarget E;
        struct VertexBufferBind N;
        struct ViewSetup N;
        enum WriteSource N;
        struct BindShape N;
        struct BindState N;
        struct BoundAccelerationStructure N;
        enum BoundBuffer N;
        struct BoundSampler N;
        struct BoundSamplerShape N;
        enum BoundTexture N;
        enum BufferShape N;
        enum HeapPlacement N;
        enum ResourceClass N;
        struct SamplerDescriptor N;
        struct SamplerShape N;
        struct StageShape N;
        struct StageTable N;
        struct TextureFacts N;
        enum TextureShape N;
        enum UsedClass N;
        struct UsedResource N;
        struct UsedShape N;
        struct VertexAmplification N;
        struct ViewMapping N;
        struct ImageblockSize N;
        struct StageInRegion N;
        struct TessellationFactors N;
        struct TessellationFactorBuffer N;
        struct ArgumentSampler N;
        struct ArgumentSamplersShape N;
        struct ArgumentFunctionTable N;
        struct FunctionTablesShape N;
        struct TileThreadgroupMemory N;
        struct TileThreadgroupArgument N;
        struct CompensatedViewport N;
        enum FragmentTests N;
        struct ExecutionStatus E;
        struct MeshRecordsFault N;
        struct TessellationRecordsFault N;
        struct StatusBits E;
        struct StatusReport N;
        enum StatusWord N;
        struct ReplayFaults E;
        fn replay_preprocess;
        fn replay_preprocess_module;
        struct ReplayExecution N;
        struct ReplayLayout N;
        struct ReplayPreprocess N;
        struct ReplayPreprocessModule N;
        enum ReplayRefusal N;
        struct ReplayVariant N;
        const MAX_REPLAY_DISPATCHES;
        const MAX_REPLAY_REGIONS;
        const MAX_REPLAY_SLOTS;
        const MAX_REPLAY_VARIANTS;
        fn replay_preprocess_generated;
        struct GeneratedLayout N;
        enum GeneratedRefusal N;
        struct GeneratedReplay N;
        struct GeneratedVariant N;
        fn draw_replay_module;
        fn draw_replay_preprocess;
        struct DrawReplay N;
        struct DrawReplayLayout N;
        struct DrawReplayModule N;
        struct DrawReplayPreprocess N;
        const DRAW_ARGUMENT_BYTES;
        const MAX_DRAW_REPLAY_TYPES;
        fn draw_replay_draw;
        fn draw_replay_merges;
        fn draw_replay_primitive_type;
        fn draw_replay_slots;
        fn draw_replay_state;
        enum DrawReplayRule N;
        const DRAW_REPLAY_WRITES;
        fn draw_merge_module;
        struct DrawMergeModule N;
        enum MergePhase E;
        struct MergedLayout N;
        const MERGED_CHUNK;
        const MERGED_HEADER_BYTES;
        const MERGED_REST_FIRST;
        fn flat_dispatch_module;
        enum FlatArguments N;
        struct FlatDispatchModule N;
        struct FlatIndirect N;
        fn predication_step_module;
        enum PredicationCompare N;
        enum PredicationOp N;
        struct PredicationStep N;
        enum PredicationStepForm N;
        struct PredicationStepModule N;
        struct FunctionTableSnapshot O;
        struct FunctionTableWords O;
        fn geometry_flags;
        fn geometry_table_offset;
        fn instance_record;
        fn primitive_wrap;
        struct InstanceRecord O;
        struct MetalInstance N;
        const GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION;
        const GEOMETRY_OPAQUE;
        struct RasterizationRateMap O;
        struct RateMapAxis O;
        struct RateMapDescriptor N;
        struct RateMapLayerDescriptor N;
        enum RateMapError N;
        const MAX_SCREEN_EXTENT;
        enum UnpinnedCompile N;
        struct RateMapRegions O;
        struct RateMapRegion O;
        struct RegionLimits N;
        enum RegionError N;
        struct SparseTileMap O;
        enum TileMapError N;
        enum TileMapping E;
        fn visibility_combine_module;
        struct ContinuedResult N;
        struct VisibilityCombine N;
        struct VisibilityCombineModule N;
        fn visibility_resolve_module;
        struct VisibilityResolve N;
        struct VisibilityResolveModule N;
        fn shadow_copy_back_module;
        struct ShadowCopyBackModule N;
        enum ShadowTexelView E;
        fn shadow_copy_in_module;
        struct ShadowCopyInModule N;
        fn rows_copy_module;
        struct RowsCopyModule N;
        fn shadow_merge_module;
        struct ShadowMergeModule N;
        fn shadow_merge_image_module;
        fn shadow_merge_kept_image_module;
        struct ShadowMergeImageModule N;
        enum ShadowMergeRead E;
        fn depth_magnitude_module;
        struct DepthMagnitudeModule N;
        fn unused_bits_module;
        struct UnusedBitsModule N;
        fn extended_shadow_module;
        struct ExtendedShadowModule N;
        fn tile_reach_module;
        struct TileReachModule N;
        fn capture_assembly_module;
        struct CaptureAssemblyModule N;
        fn capture_settle_module;
        struct CaptureSettleModule N;
        enum TexelWords N;
        fn compressed_blit_module;
        enum CompressedBlit N;
        struct CompressedBlitModule N;
        fn block_decode_module;
        enum BlockDecode N;
        struct BlockDecodeModule N;
        fn astc_decode_module;
        struct AstcDecode N;
        enum AstcTexels N;
        fn pvrtc_decode_module;
        struct PvrtcDecode N;
        fn etc_decode_module;
        enum EtcDecode N;
        enum EtcTexels N;
        fn chroma_decode_module;
        enum ChromaDecode N;
        fn mesh_scan_module;
        fn mesh_object_arguments_module;
        struct MeshScanModule N;
        fn mesh_arguments_module;
        fn fold_object_grid;
        fn metal_object_grid;
        struct MeshArguments N;
        enum MeshArgumentsForm E;
        struct MeshArgumentsModule N;
        struct MeshGridLimits N;
        struct MeshGridRule N;
        struct ObjectFold N;
        const METAL_MESH_GRID_AXIS;
        const METAL_MESH_GRID_TOTAL;
        const METAL_OBJECT_GRID_TOTAL;
        fn tessellation_scan_module;
        struct TessellationScanKernel N;
        struct TessellationScanModule N;
        fn mipmap_filter_module;
        struct MipmapFilterChain N;
        struct MipmapFilterKernel O;
        struct MipmapFilterLevel N;
        struct MipmapFilterModule N;
        enum MipmapRule N;
        fn plane_copy_module;
        struct PlaneCopyModule N;
        struct PlaneRows N;
        fn plane_bytes_copy_module;
        struct PlaneBytes N;
        struct PlaneBytesCopyModule N;
        enum PlaneSource N;
        fn plane_words_module;
        struct PlaneWordsModule N;
        fn rotate_texels_module;
        struct RotateTexelsModule N;
        fn two_plane_texels_module;
        struct TwoPlaneTexelsModule N;
        struct PlaneLayout N;
        fn tiled_texels_module;
        struct TiledTexelsModule N;
        struct TiledLevelMove N;
        fn ycbcr_convert_module;
        fn ycbcr_convert_floats_module;
        fn ycbcr_texel;
        fn ycbcr_read;
        struct YCbCrConvertModule N;
        struct YCbCrPlanes N;
        struct YCbCrCodes N;
        enum YCbCrOutput N;
        enum YCbCrRange N;
        enum YCbCrMatrix N;
        fn ycbcr_write_module;
        fn ycbcr_write;
        fn ycbcr_store;
        struct YCbCrWriteModule N;
        fn ycbcr_refresh_module;
        struct YCbCrRefreshModule N;
        fn planes_merge_modules;
        fn planes_split_module;
        enum PlanesCopy N;
        struct PlanesMergeModules N;
        struct PlanesSplitModule N;
        fn interleave_merge_modules;
        fn interleave_split_module;
        struct InterleaveMergeModules N;
        struct InterleaveSplitModule N;
        fn color_resolve_modules;
        struct ColorResolveFormat O;
        struct ColorResolveModules N;
        struct ResolveChannel N;
        enum ResolveRule N;
        enum ResolveSource N;
        const SRGB_RESOLVE_LINEAR;
        const SRGB_RESOLVE_THRESHOLDS;
        fn color_write_table;
        fn depth_stencil_copy_modules;
        fn stencil_split_module;
        fn depth_split_module;
        enum CopySource E;
        enum DepthStandInForm E;
        struct DepthSplitModule N;
        enum DepthStencilCopy N;
        struct DepthStencilCopyModules N;
        struct StencilBit O;
        struct StencilSplitModule N;
        fn depth_stencil_resolve_modules;
        enum DepthPick N;
        struct DepthStencilResolveDraw N;
        struct DepthStencilResolveModules N;
        enum ResolveWrite N;
        fn index_bounds_fill_module;
        fn index_bounds_fill_run_module;
        fn index_bounds_plan_module;
        fn index_bounds_plan_run_module;
        struct BoundsRunDraw E;
        struct IndexBoundsFillModule N;
        struct IndexBoundsPass N;
        struct IndexBoundsPlanModule N;
        struct IndexBoundsRun O;
        fn index_gather_module;
        fn index_gather_plan_module;
        fn index_rewrite_module;
        fn index_rewrite_run_module;
        struct GatherDraw E;
        struct IndexGather O;
        struct IndexGatherModule N;
        struct IndexGatherPlanModule N;
        struct IndexRewriteModule N;
        struct IndexRewritePass N;
        struct IndexRewriteRun O;
        struct IndirectGatherDraw E;
        enum RewriteOrder N;
        enum RewriteOutput N;
        enum RewritePhase N;
        enum RewriteProgram N;
        struct RewriteRunDraw E;
        enum RewriteSource N;
        mod indirect_commands;
    }
    "bind::indirect_commands" => metal2vulkan::bind::indirect_commands {
        enum Action E;
        enum AddressError N;
        enum BindingError E;
        struct Bindings O;
        struct BufferBinding E;
        enum BufferStage E;
        struct CommandBuffer O;
        enum CommandBufferError E;
        enum CommandDescriptor E;
        struct CommandRecord E;
        enum ComputeDispatch E;
        struct Inheritance E;
        enum RecordError E;
        enum RecordField E;
        struct RecordView<'static> O;
        struct RegisteredBuffer<'static, u8> O;
        struct RenderDraws E;
        struct ResolvedBindings O;
        struct ResolvedBuffer<'static, u8> N;
        enum StateSource E;
        const BINDING_COUNT;
        const COMMAND_BUFFER_HEADER_BYTES;
        const COMMAND_BYTES;
        const RECORDS_ADDRESS_BASE;
        const RECORDS_ID_LIMIT;
        const THREADGROUP_BUFFER_ARGUMENT_COUNT;
        fn records_address;
    }
    "cache" => metal2vulkan::cache {
        type Hash256;
        const SEMANTICS_VERSION;
        const ARTIFACT_FORMAT_VERSION;
        const RUNTIME_VERSION;
        const SOURCE_DIGEST;
        struct PipelineKey N;
        struct VariantKey N;
        struct SpecializationKey N;
        enum ArtifactError N;
        trait PipelineArtifact;
    }
    "metal" => metal2vulkan::metal {
        enum ArgumentBuffersTier N;
        enum AttributeFormat N;
        struct BarrierScope O;
        enum BindingAccess N;
        enum BindingType N;
        enum BlendFactor N;
        enum BlendOperation N;
        struct ColorWriteMask O;
        enum CompareFunction N;
        enum CullMode E;
        enum DataType N;
        enum DepthClipMode N;
        enum DispatchType N;
        struct FunctionOptions O;
        enum FunctionType N;
        enum IndexType E;
        struct IndirectCommandType O;
        enum Mutability N;
        enum PatchType N;
        enum PixelFormat N;
        enum PrimitiveTopologyClass N;
        enum PrimitiveType E;
        enum ReadWriteTextureTier N;
        struct RenderStages O;
        struct ResourceUsage O;
        enum SamplerAddressMode N;
        enum SamplerBorderColor N;
        enum SamplerMinMagFilter N;
        enum SamplerMipFilter N;
        enum SamplerReductionMode N;
        enum StencilOperation N;
        enum StepFunction N;
        enum TessellationControlPointIndexType N;
        enum TessellationFactorFormat N;
        enum TessellationFactorStepFunction N;
        enum TessellationPartitionMode N;
        enum TextureSwizzle N;
        struct TextureSwizzleChannels E;
        enum TextureType E;
        struct TextureUsage O;
        enum TextureWriteRoundingMode N;
        enum TriangleFillMode N;
        enum VertexFormat N;
        enum VertexStepFunction N;
        enum Winding E;
        mod format;
        mod sampler;
        mod state;
        mod vertex;
    }
    "metal::format" => metal2vulkan::metal::format {
        struct FormatMapping N;
        enum FormatEmulation N;
        enum ExtendedRange N;
        enum WordAttachment N;
        struct WordField E;
        enum ClearRoute N;
        fn vk_format;
        fn swizzle_over;
        fn direct_format;
        fn byte_format;
        fn srgb_raw_twin;
        fn block_extent;
        fn imageblock_sample_length;
        fn color_record;
        fn explicit_imageblock_sample_length;
        fn aspects;
        fn is_depth_stencil;
        fn sampled_type;
        fn texture_usages;
        fn image_view_type;
        fn image_type;
        fn storage_image_choice;
        fn half_bits_read_view;
        fn storage_view_formats;
        fn ycbcr;
        fn writes_dropped;
        fn ycbcr_writes_stored;
        fn ycbcr_renders;
        fn ycbcr_pass_refused;
        fn padded_words;
        fn extended_range_blend;
        fn x_channel;
        fn half_write_nans_converted;
        fn bound_texel_format;
        struct YCbCrFormat N;
        enum ChromaSubsampling N;
        enum YCbCrReads N;
        fn ycbcr_format;
        fn rotated_channels;
        fn to_rotated_image;
        fn alpha_in_red;
        fn alpha_in_red_format;
    }
    "metal::sampler" => metal2vulkan::metal::sampler {
        struct SamplerUse N;
        struct SamplerWord E;
        struct AddressModeMap O;
        enum OpaqueBlackBorder N;
        const NOT_MIPMAPPED_MAX_LOD;
        fn check_sampler;
        fn vk_sampler;
        fn check_border_swizzle;
        fn check_border_view;
        fn opaque_black_reads_transparent;
        fn opaque_black_border;
        fn border_color;
        fn compare_op;
    }
    "metal::state" => metal2vulkan::metal::state {
        fn blend_factor;
        fn blend_op;
        fn color_write_mask;
        fn blend_state;
        fn stencil_op;
        struct StencilDescriptor N;
        struct DepthStencilDescriptor N;
        struct StencilOpState N;
        struct DepthStencilState N;
        fn depth_stencil_state;
        fn input_assembly;
        fn index_type;
        fn polygon_mode;
        fn depth_clamp_enable;
        struct DepthBias N;
        enum DepthBiasUnit N;
        fn depth_bias;
        fn depth16_bias;
        fn depth16_bias_side;
    }
    "metal::vertex" => metal2vulkan::metal::vertex {
        fn vk_vertex_format;
        fn vertex_alignment;
        const MAX_VERTEX_ALIGNMENT;
        enum AttributeFetch N;
        fn attribute_fetch;
    }
    "vk" => metal2vulkan::vk {
        enum DrawCommand E;
        struct AccessFlags O;
        struct AccessFlags2 O;
        struct BlendFactor O;
        struct BlendOp O;
        struct BorderColor O;
        struct BufferUsageFlags O;
        struct ColorComponentFlags O;
        struct CompareOp O;
        struct ComponentSwizzle O;
        struct CullModeFlags O;
        struct DescriptorBindingFlags O;
        struct DescriptorSetLayoutCreateFlags O;
        struct DescriptorType O;
        struct DriverId O;
        struct DynamicState O;
        struct Filter O;
        struct Format O;
        struct FormatFeatureFlags O;
        struct FormatFeatureFlags2 O;
        struct FrontFace O;
        struct ImageAspectFlags O;
        struct ImageCreateFlags O;
        struct ImageLayout O;
        struct ImageTiling O;
        struct ImageType O;
        struct ImageUsageFlags O;
        struct ImageViewType O;
        struct IndexType O;
        struct LogicOp O;
        struct PhysicalDeviceType O;
        struct PipelineShaderStageCreateFlags O;
        struct PipelineStageFlags O;
        struct PipelineStageFlags2 O;
        struct PolygonMode O;
        struct PrimitiveTopology O;
        struct ResolveModeFlags O;
        struct SampleCountFlags O;
        struct SamplerAddressMode O;
        struct SamplerMipmapMode O;
        struct SamplerReductionMode O;
        struct ShaderFloatControlsIndependence O;
        struct ShaderStage O;
        struct ShaderStageFlags O;
        struct StencilOp O;
        struct SubgroupFeatureFlags O;
        struct VertexInputRate O;
        struct ComponentMapping E;
        struct FormatBlock N;
        struct FormatExposure N;
        enum NumericFormat N;
        enum NumericType N;
        struct SamplerDesc N;
        struct SpecializationMapEntry E;
        struct ApiVersion E;
    }
    #[cfg(feature = "ash")]
    "ash_interop" => metal2vulkan::ash_interop {
        fn sampler_create_info;
    }
}

// ---- The consumer flow through the public paths (M5 "`CompiledFunction` round trip") ----

use metal2vulkan::bind::{BindState, BoundBuffer, Dispatch, DrawCall, StageTable};
use metal2vulkan::cache::{Hash256, PipelineArtifact, PipelineKey};
use metal2vulkan::device::{
    Device, DeviceDescription, DevicePolicy, FloatMathsPolicy, FloatWidth, NanMinMaxPolicy,
};
use metal2vulkan::library::{CompiledFunction, Library};
use metal2vulkan::metal::{PixelFormat, VertexFormat};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipeline, ComputePipelineDescriptor, EntryDescriptor,
    EntryMath, EntryRef, FunctionDescriptor, LoweringInputs, PrecisionNote, RenderPipeline,
    RenderPipelineDescriptor, Stage, StageSource, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};
use metal2vulkan::Budget;

/// `inputs` with the color writes `writes` (attachment, its channels' rules), the half-float
/// channels alone where `dev`'s driver converts a fragment's float writes as Metal does, and
/// elsewhere each attachment whose rules are each none, the device's, an 8-bit unorm channel's or a
/// half-float channel's, one of them a rounding, selected at the draw (`ChannelWrite::Selected`):
/// the inputs a fragment of a pipeline with those attachments compiles under.
fn with_color_writes(
    dev: &Device,
    mut inputs: LoweringInputs,
    writes: &[(u32, [Option<metal2vulkan::pipeline::ChannelWrite>; 4])],
) -> LoweringInputs {
    use metal2vulkan::pipeline::ChannelWrite;
    let metal = metal2vulkan::error::DriverDeviation::MetalColorWrites;
    let half = ChannelWrite::FloatTowardZero {
        mantissa: 10,
        signed: true,
    };
    let metal = dev.profile().takes_driver_deviation(metal);
    // Whether a channel's rule rounds, or `None` where no code selects it.
    let rounds = |r: Option<ChannelWrite>| match r {
        None | Some(ChannelWrite::Device) => Some(false),
        Some(r) => (r == ChannelWrite::Unorm { bits: 8 } || r == half).then_some(true),
    };
    let selects = |rules: &[Option<ChannelWrite>; 4]| {
        let each: Option<Vec<bool>> = rules.iter().map(|&r| rounds(r)).collect();
        each.is_some_and(|each| each.contains(&true))
    };
    inputs.color_writes = writes
        .iter()
        .map(|&(a, rules)| match metal {
            true => (a, rules.map(|r| r.filter(|&r| r == half))),
            false if selects(&rules) => (a, [Some(ChannelWrite::Selected); 4]),
            false => (a, rules),
        })
        .filter(|(_, rules)| rules.iter().any(Option::is_some))
        .collect();
    inputs
}

/// Every channel of an `RGBA8Unorm` attachment.
const UNORM8: [Option<metal2vulkan::pipeline::ChannelWrite>; 4] =
    [Some(metal2vulkan::pipeline::ChannelWrite::Unorm { bits: 8 }); 4];

/// Owned fixture `stem` as bitcode, or `None` (skipped visibly) without `llvm-as`. Assembling is
/// test infrastructure; everything after it uses the public surface.
fn fixture(stem: &str) -> Option<Vec<u8>> {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine")
        .join(format!("{stem}.ll"));
    let text =
        std::fs::read_to_string(&ll).unwrap_or_else(|e| panic!("read {}: {e}", ll.display()));
    common::llvm_as(&text, &format!("{stem}.ll"))
}

/// Committed profile `stem` as a device with the default policy.
fn device(stem: &str) -> Device {
    device_under(stem, DevicePolicy::default())
}

/// [`device`] under `policy`.
fn device_under(stem: &str, policy: DevicePolicy) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, policy).expect("device")
}

/// A buffer bound at `offset` in a `length`-byte allocation.
fn buffer(length: u64, offset: u64) -> BoundBuffer {
    BoundBuffer::Buffer {
        length,
        offset,
        gpu_address: 0x1_0000,
        device_address: None,
        attribute_stride: None,
    }
}

/// `f` compiled on `dev`, encoded, and decoded: the decoded function equals the original and
/// re-encodes to the same bytes.
fn round_tripped(
    dev: &Device,
    f: &FunctionDescriptor,
    inputs: &LoweringInputs,
) -> CompiledFunction {
    let compiled = dev
        .compile_function(f, inputs, &Budget::standard())
        .expect("compile");
    assert_eq!(*compiled.key(), dev.function_compile_key(f, inputs));
    let bytes = compiled.to_bytes();
    let back = CompiledFunction::from_bytes_for_key(&bytes, compiled.key(), dev).expect("decode");
    assert_eq!(back, compiled);
    assert_eq!(back.to_bytes(), bytes, "the encoding is canonical");
    back
}

/// A compute pipeline linked from a decoded `CompiledFunction` is the pipeline compiled from the
/// function, it round-trips as an artifact, and plans and dispatches the same.
#[test]
fn compiled_functions_round_trip_through_the_public_api() {
    let Some(bc) = fixture("store_const") else {
        return;
    };
    for stem in ["moltenvk-m3", "vk12-floor"] {
        let dev = device(stem);
        let f = FunctionDescriptor::new(
            Library::from_air(&bc)
                .expect("library")
                .function("store_const")
                .expect("function"),
        );
        let from_source = dev
            .compile_compute_pipeline(
                &ComputePipelineDescriptor::new(StageSource::Function(f.clone())),
                &Budget::standard(),
            )
            .expect("pipeline from the function");
        let back = round_tripped(&dev, &f, &LoweringInputs::new());
        let d = ComputePipelineDescriptor::new(StageSource::Compiled(back));
        assert_eq!(dev.compute_pipeline_key(&d), from_source.key, "{stem}");
        let linked = dev
            .compile_compute_pipeline(&d, &Budget::standard())
            .expect("pipeline from the decoded function");
        assert_eq!(linked, from_source, "{stem}");
        let cached = ComputePipeline::from_bytes_for_key(&linked.to_bytes(), &linked.key, &dev)
            .expect("pipeline decode");
        assert_eq!(cached, from_source, "{stem}");

        let mut table = StageTable::default();
        table.buffers.insert(0, buffer(4096, 256));
        let mut state = BindState::default();
        state.stages.insert(Stage::Kernel, table);
        let dispatch = Dispatch::Threads {
            threads: [100, 1, 1],
            threads_per_group: [32, 1, 1],
        };
        let plan = cached.plan(&dev, &state.shape()).expect("plan");
        assert_eq!(plan, from_source.plan(&dev, &state.shape()).expect("plan"));
        assert_eq!(
            plan.instantiate_dispatch(&state, &dispatch),
            from_source
                .plan(&dev, &state.shape())
                .expect("plan")
                .instantiate_dispatch(&state, &dispatch)
        );
    }
}

/// The content digests are public: a pipeline's names what it says, not the build that compiled it.
/// Another build of the crate, whose versions differ and so move every cache key, compiles the same
/// content under another `key`; that key and another profile fingerprint leave the digest equal,
/// while one SPIR-V word moves it. A binding plan's digest is its content's: equal for equal plans,
/// moved by one changed value.
#[test]
fn content_digests_name_what_a_pipeline_says_not_its_build() {
    let Some(bc) = fixture("store_const") else {
        return;
    };
    let dev = device("moltenvk-m3");
    let f = FunctionDescriptor::new(
        Library::from_air(&bc)
            .expect("library")
            .function("store_const")
            .expect("function"),
    );
    let pipeline = dev
        .compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Function(f)),
            &Budget::standard(),
        )
        .expect("pipeline");
    let digest: Hash256 = pipeline.content_digest();
    let decoded = ComputePipeline::from_bytes(&pipeline.to_bytes(), &dev).expect("decode");
    assert_eq!(decoded.content_digest(), digest);

    let mut other_build = pipeline.clone();
    other_build.key = PipelineKey::new([7; 32]);
    other_build.profile = [9; 32];
    assert_eq!(other_build.content_digest(), digest);

    let mut edited = pipeline.clone();
    let mut words = edited.stage.spirv.to_vec();
    let last = words.len() - 1;
    words[last] ^= 1;
    edited.stage.spirv = words.into();
    assert_ne!(edited.content_digest(), digest);

    // The render pipeline's digest, by the same definition.
    let _: fn(&RenderPipeline) -> Hash256 = RenderPipeline::content_digest;

    #[cfg(feature = "serde")]
    {
        let mut table = StageTable::default();
        table.buffers.insert(0, buffer(4096, 256));
        let mut state = BindState::default();
        state.stages.insert(Stage::Kernel, table);
        let plan = pipeline.plan(&dev, &state.shape()).expect("plan");
        let plan_digest = plan.content_digest().expect("plan digest");
        let replanned = other_build.plan(&dev, &state.shape()).expect("plan");
        assert_eq!(replanned.content_digest(), Ok(plan_digest));
        let mut edited = plan.clone();
        edited.compute = None;
        assert_ne!(edited.content_digest(), Ok(plan_digest));
    }
}

/// The precision contract is public: a fast `float` division records the device's arithmetic,
/// Metal's precise division records the device's division and NaN where the policy takes both
/// (`NativeDivision`, `NativeNanEncoding`, native device maths), the division alone under
/// `NanMinMaxPolicy::Exact` and nothing under `FloatMathsPolicy::Exact`, an integer operator has no
/// notes, and a fast `float` `tanh`, which the crate computes with a division of its own, records
/// the device's arithmetic, on a profile whose `float` functions are the device's (lavapipe's); on
/// one whose functions are Metal's own (MoltenVK's), both are Metal's own in a fast-math entry and
/// record nothing there, and the division records its note in an entry that keeps NaNs.
#[test]
fn the_precision_contract_names_each_float_operations_notes() {
    use metal2vulkan_air::intrinsics::AirIntrinsic;
    use metal2vulkan_air::ir::{BinOp, FastMath};
    let fast = FastMath::ALL
        .iter()
        .fold(FastMath::NONE, |all, (flag, _)| all.union(*flag));
    let lavapipe = device("lavapipe");
    let moltenvk = device("moltenvk-m3");
    let (lavapipe, moltenvk) = (lavapipe.profile(), moltenvk.profile());
    let fdiv = |p: &metal2vulkan::device::DeviceProfile, math| {
        PrecisionNote::of_float_operator(BinOp::FDiv, FloatWidth::F32, fast, p, math)
    };
    assert!(fdiv(lavapipe, EntryMath::Fast)
        .expect("a float operator")
        .contains(&PrecisionNote::DeviceArithmetic));
    assert_eq!(fdiv(moltenvk, EntryMath::Fast), Some(BTreeSet::new()));
    assert!(fdiv(moltenvk, EntryMath::Precise)
        .expect("a float operator")
        .contains(&PrecisionNote::DeviceArithmetic));
    let mut policy = DevicePolicy::default();
    policy.float_maths = FloatMathsPolicy::Native;
    policy.nan_and_min_max = NanMinMaxPolicy::Native;
    let native = device_under("lavapipe", policy);
    policy.nan_and_min_max = NanMinMaxPolicy::Exact;
    let metal_nans = device_under("lavapipe", policy);
    policy.float_maths = FloatMathsPolicy::Exact;
    let exact = device_under("lavapipe", policy);
    for (p, notes) in [
        (
            native.profile(),
            BTreeSet::from([
                PrecisionNote::NativeDivision,
                PrecisionNote::NativeNanEncoding,
            ]),
        ),
        (
            metal_nans.profile(),
            BTreeSet::from([PrecisionNote::NativeDivision]),
        ),
        (exact.profile(), BTreeSet::new()),
    ] {
        assert_eq!(
            PrecisionNote::of_float_operator(
                BinOp::FDiv,
                FloatWidth::F32,
                FastMath::NONE,
                p,
                EntryMath::Precise
            ),
            Some(notes)
        );
    }
    assert_eq!(
        PrecisionNote::of_float_operator(
            BinOp::Add,
            FloatWidth::F32,
            fast,
            lavapipe,
            EntryMath::Fast
        ),
        None
    );
    let Some(AirIntrinsic::Math(tanh)) = AirIntrinsic::parse("air.fast_tanh.f32") else {
        panic!("air.fast_tanh.f32 is a math call");
    };
    let notes = PrecisionNote::of_math_call(tanh, fast, lavapipe, EntryMath::Fast)
        .expect("a float function");
    assert!(notes.contains(&PrecisionNote::DeviceArithmetic));
    let notes = PrecisionNote::of_math_call(tanh, fast, moltenvk, EntryMath::Fast)
        .expect("a float function");
    assert!(notes.is_empty(), "{notes:?}");
}

/// A render pipeline with a signed-integer color attachment compiles its fragment function with the
/// attachments it writes in `LoweringInputs::signed_color_attachments`, the inputs
/// `render_pipeline_key` hands out; a fragment compiled on its own under them links into the same
/// pipeline, and one compiled without them is another pipeline's stage. A pipeline without such
/// an attachment keeps the fragment's common-path key.
#[test]
fn a_signed_integer_attachment_compiles_the_fragment_for_it() {
    let Some(bc) = fixture("signed_color") else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let vertex = FunctionDescriptor::new(lib.function("signed_color_v").expect("vertex"));
    let fragment = FunctionDescriptor::new(lib.function("signed_targets_f").expect("fragment"));
    let descriptor = |f: StageSource, ids: PixelFormat| {
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(vertex.clone()));
        d.fragment = Some(f);
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        d.color_attachments[1] = Some(ColorAttachmentDescriptor::new(ids));
        d.color_attachments[2] = Some(ColorAttachmentDescriptor::new(PixelFormat::RG32Uint));
        // An attachment the fragment does not write is not an input.
        d.color_attachments[5] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Sint));
        d
    };
    for stem in ["moltenvk-m3", "lavapipe"] {
        let dev = device(stem);
        let signed = descriptor(
            StageSource::Function(fragment.clone()),
            PixelFormat::RGBA32Sint,
        );
        let keys = dev
            .render_pipeline_key(&signed, &Budget::standard())
            .expect("keys");
        let inputs = keys.fragment_inputs.clone().expect("fragment inputs");
        assert_eq!(
            inputs.signed_color_attachments,
            BTreeSet::from([1]),
            "{stem}"
        );
        let pipeline = dev
            .compile_render_pipeline(&signed, &Budget::standard())
            .expect("signed pipeline");
        assert_eq!(pipeline.key, keys.pipeline, "{stem}");
        let f = round_tripped(&dev, &fragment, &inputs);
        assert_eq!(Some(*f.key()), keys.fragment, "{stem}");
        let from_compiled = descriptor(StageSource::Compiled(f), PixelFormat::RGBA32Sint);
        assert_eq!(
            dev.compile_render_pipeline(&from_compiled, &Budget::standard())
                .expect("pipeline from the compiled fragment"),
            pipeline,
            "{stem}"
        );
        let plain = round_tripped(&dev, &fragment, &LoweringInputs::new());
        let foreign = descriptor(
            StageSource::Compiled(plain.clone()),
            PixelFormat::RGBA32Sint,
        );
        assert_eq!(
            dev.compile_render_pipeline(&foreign, &Budget::standard())
                .unwrap_err(),
            metal2vulkan::error::Error::InvalidDescriptor(
                metal2vulkan::error::DescriptorError::ColorAttachment { index: 1 }
            ),
            "{stem}"
        );
        // An unsigned attachment there: the common path, no decode, the plain key.
        let unsigned = descriptor(
            StageSource::Function(fragment.clone()),
            PixelFormat::RGBA32Uint,
        );
        let keys = dev
            .render_pipeline_key(&unsigned, &Budget::standard())
            .expect("keys");
        let common = with_color_writes(&dev, LoweringInputs::new(), &[(0, UNORM8)]);
        assert_eq!(keys.fragment_inputs.as_ref(), Some(&common), "{stem}");
        let plain = round_tripped(&dev, &fragment, &common);
        assert_eq!(keys.fragment, Some(*plain.key()), "{stem}");
    }
}

/// A render pipeline that rasterizes more than one sample compiles its fragment function under
/// `LoweringInputs::multisampled`, the input `render_pipeline_key` hands out (also where a
/// signed-integer attachment makes it decode the function); a fragment compiled on its own under it
/// links into the same pipeline, and one compiled without it, which would read `[[position]]` and
/// its center varyings where Vulkan shades, is refused as needing the input. A pipeline of one
/// sample keeps the fragment's common-path key. `moltenvk-m3`, whose `InterpolateAtOffset` reads
/// off the pixel center, reads the center varyings at the pixel center itself, as Metal does, so
/// its pipeline is as lavapipe's. At MoltenVK 1.4.0, not measured so, the key is the same and the
/// sample-rate function's center varyings have no pixel form: the pipeline is refused naming the
/// property it lacks, compiled under the input or not.
#[test]
fn a_multisampled_pipeline_compiles_the_fragment_for_it() {
    let Some(bc) = fixture("multisample") else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let vertex = FunctionDescriptor::new(lib.function("multisample_vertex").expect("vertex"));
    let fragment = FunctionDescriptor::new(
        lib.function("multisample_center_at_sample")
            .expect("fragment"),
    );
    let descriptor = |f: StageSource, samples: u32| {
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(vertex.clone()));
        d.fragment = Some(f);
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
        d.raster_sample_count = samples;
        d
    };
    let half = [Some(metal2vulkan::pipeline::ChannelWrite::FloatTowardZero {
        mantissa: 10,
        signed: true,
    }); 4];
    // MoltenVK 1.4.0 as its `driverVersion` encodes it.
    const MOLTENVK_1_4_0: u32 = 10400;
    for (stem, release) in [
        ("moltenvk-m3", None),
        ("moltenvk-m3", Some(MOLTENVK_1_4_0)),
        ("lavapipe", None),
    ] {
        let dev = match release {
            None => device(stem),
            Some(version) => {
                let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../metal2vulkan-validation/profiles")
                    .join(format!("{stem}.json"));
                let text = std::fs::read_to_string(&path).expect("profile");
                let mut desc =
                    DeviceDescription::from_khronos_json(&text, None).expect("description");
                desc.driver_version = version;
                Device::new(desc, DevicePolicy::default()).expect("device")
            }
        };
        let mut multisampled = with_color_writes(&dev, LoweringInputs::new(), &[(0, half)]);
        multisampled.multisampled = true;
        let four = descriptor(StageSource::Function(fragment.clone()), 4);
        let keys = dev
            .render_pipeline_key(&four, &Budget::standard())
            .expect("keys");
        assert_eq!(keys.fragment_inputs.as_ref(), Some(&multisampled), "{stem}");
        let common = with_color_writes(&dev, LoweringInputs::new(), &[(0, half)]);
        let plain = round_tripped(&dev, &fragment, &common);
        let foreign = descriptor(StageSource::Compiled(plain.clone()), 4);
        if release.is_some() {
            let needs_centered: metal2vulkan::error::Error =
                metal2vulkan::error::Unsupported::needs(
                    metal2vulkan::error::Feature::SampleRateShading,
                    metal2vulkan::error::Capability::Property(
                        metal2vulkan::device::VkProperty::InterpolateAtOffsetCentered,
                    ),
                )
                .into();
            for d in [&four, &foreign] {
                assert_eq!(
                    dev.compile_render_pipeline(d, &Budget::standard())
                        .unwrap_err(),
                    needs_centered,
                    "{stem}"
                );
            }
            continue;
        }
        let pipeline = dev
            .compile_render_pipeline(&four, &Budget::standard())
            .expect("multisampled pipeline");
        assert_eq!(pipeline.key, keys.pipeline, "{stem}");
        let f = round_tripped(&dev, &fragment, &multisampled);
        assert_eq!(Some(*f.key()), keys.fragment, "{stem}");
        let from_compiled = descriptor(StageSource::Compiled(f.clone()), 4);
        assert_eq!(
            dev.compile_render_pipeline(&from_compiled, &Budget::standard())
                .expect("pipeline from the compiled fragment"),
            pipeline,
            "{stem}"
        );
        assert_eq!(
            dev.compile_render_pipeline(&foreign, &Budget::standard())
                .unwrap_err(),
            metal2vulkan::error::Error::NeedsInput(metal2vulkan::error::RuntimeInput::Multisampled),
            "{stem}"
        );
        // A signed-integer attachment the fragment does not write: the key decodes the function,
        // finds no signed output, and keeps the multisampled input.
        let mut signed = four.clone();
        signed.color_attachments[3] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Sint));
        let keys = dev
            .render_pipeline_key(&signed, &Budget::standard())
            .expect("keys");
        assert_eq!(keys.fragment_inputs.as_ref(), Some(&multisampled), "{stem}");
        assert_eq!(keys.fragment, Some(*f.key()), "{stem}");
        // One sample: the common path, the plain key.
        let one = descriptor(StageSource::Function(fragment.clone()), 1);
        let keys = dev
            .render_pipeline_key(&one, &Budget::standard())
            .expect("keys");
        assert_eq!(keys.fragment_inputs, Some(common), "{stem}");
        assert_eq!(keys.fragment, Some(*plain.key()), "{stem}");
    }
}

/// The vertex layout `stage_io_vertex` reads its attributes through: one 24-byte buffer.
fn stage_io_layout() -> VertexDescriptor {
    let mut layout = VertexDescriptor::new();
    layout
        .layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(24)));
    for (index, format, offset) in [
        (0, VertexFormat::Float3, 0),
        (1, VertexFormat::Half2, 12),
        (4, VertexFormat::UShort2, 16),
        (7, VertexFormat::UChar4, 20),
    ] {
        layout
            .attributes
            .insert(index, VertexAttributeDescriptor::new(format, offset, 0));
    }
    layout
}

/// One entry of a bare module translates through its profile as a library's function compiles
/// on the profile's device: the same compiled function at the same cost, linked alone into a SPIR-V
/// module, with its plan, its specialized entry, a vertex stage's inputs, and its interface facts
/// beside it.
#[test]
fn one_entry_of_a_bare_module_translates_through_the_public_api() {
    let (Some(kernel), Some(stages)) = (fixture("store_const"), fixture("stage_io")) else {
        return;
    };
    let budget = Budget::standard();
    let none = LoweringInputs::new();
    let layout = stage_io_layout();
    for stem in ["moltenvk-m3", "lavapipe"] {
        let dev = device(stem);
        let profile = dev.profile();
        let entry = EntryDescriptor::new(&kernel, EntryRef::Name("store_const"));
        let (compiled, report) = profile.compile_entry(&entry, &none, &budget);
        let compiled = compiled.expect("compile");
        let function = Library::from_air(&kernel)
            .expect("library")
            .function("store_const")
            .expect("function");
        let (from_library, library_report) =
            dev.compile_function_with_report(&FunctionDescriptor::new(function), &none, &budget);
        assert_eq!(Ok(&compiled), from_library.as_ref(), "{stem}");
        assert_eq!(report, library_report, "{stem}");
        assert_eq!(compiled.stage(), Stage::Kernel, "{stem}");
        assert!(
            !compiled.slots().is_empty(),
            "{stem}: it stores to a buffer"
        );
        assert!(compiled.attribute_fetches().is_empty(), "{stem}");

        let alone = profile.link_standalone(&compiled).expect("link");
        assert_eq!(
            alone.words.first(),
            Some(&0x0723_0203),
            "{stem}: SPIR-V's magic"
        );
        let (plan, _) = profile.plan_entry(&entry, &none, &budget);
        let plan = plan.expect("plan");
        assert_eq!(plan.stage, Stage::Kernel, "{stem}");
        assert!(
            plan.estimated_words >= alone.words.len() as u64,
            "{stem}: the estimate bounds the module"
        );
        let specialized = profile
            .specialized_entry(&entry, &none, &budget)
            .expect("entry");
        assert_eq!(specialized.ep.name.to_string(), "store_const", "{stem}");

        // A vertex function's inputs are its render pipeline's vertex stage's, and it links
        // alone too.
        let vertex = EntryDescriptor::new(&stages, EntryRef::Name("stage_io_vertex"));
        let inputs = profile
            .vertex_entry_inputs(&vertex, Some(&layout), &budget)
            .expect("vertex inputs");
        let function = Library::from_air(&stages)
            .expect("library")
            .function("stage_io_vertex")
            .expect("vertex");
        let mut pipeline =
            RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(function)));
        pipeline.vertex_descriptor = Some(layout.clone());
        let keys = dev.render_pipeline_key(&pipeline, &budget).expect("keys");
        assert_eq!(Some(&inputs), keys.vertex_inputs.as_ref(), "{stem}");
        let (v, _) = profile.compile_entry(&vertex, &inputs, &budget);
        let v = v.expect("vertex");
        assert_eq!(*v.key(), keys.vertex, "{stem}");
        let fetches = v.attribute_fetches();
        assert!(!fetches.is_empty(), "{stem}");
        assert!(
            fetches.keys().all(|a| layout.attributes.contains_key(a)),
            "{stem}: {fetches:?}"
        );
        profile.link_standalone(&v).expect("vertex alone");

        // A function compiled for another profile does not link here.
        let other = device(if stem == "lavapipe" {
            "moltenvk-m3"
        } else {
            "lavapipe"
        });
        assert!(
            other.profile().link_standalone(&compiled).is_err(),
            "{stem}"
        );
    }
}

/// A render pipeline linked from decoded vertex and fragment `CompiledFunction`s, each compiled
/// under the inputs `render_pipeline_key` hands out, is the pipeline compiled from the
/// functions, round-trips as an artifact, and plans and draws the same.
#[test]
fn compiled_render_stages_round_trip_through_the_public_api() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let vertex = FunctionDescriptor::new(lib.function("stage_io_vertex").expect("vertex"));
    let fragment = FunctionDescriptor::new(lib.function("stage_io_fragment").expect("fragment"));
    let layout = stage_io_layout();
    let descriptor = |v: StageSource, f: StageSource| {
        let mut d = RenderPipelineDescriptor::new(v);
        d.fragment = Some(f);
        d.vertex_descriptor = Some(layout.clone());
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        d.color_attachments[2] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
        d.depth_format = Some(PixelFormat::Depth32Float);
        d
    };
    for stem in ["moltenvk-m3", "lavapipe"] {
        let dev = device(stem);
        let from_functions = descriptor(
            StageSource::Function(vertex.clone()),
            StageSource::Function(fragment.clone()),
        );
        let keys = dev
            .render_pipeline_key(&from_functions, &Budget::standard())
            .expect("keys");
        let from_source = dev
            .compile_render_pipeline(&from_functions, &Budget::standard())
            .expect("pipeline from the functions");
        assert_eq!(keys.pipeline, from_source.key, "{stem}");

        // Each stage compiles on its own under the inputs its pipeline's keys hand out: a
        // vertex function's are the vertex-fetch projection over the attributes it reads, a
        // fragment function's on the common path only how its `RGBA8Unorm` attachment takes its
        // float output where the device does not convert as Metal does.
        let vertex_inputs = keys.vertex_inputs.clone().expect("vertex inputs");
        assert!(vertex_inputs.vertex_fetch.is_some(), "{stem}");
        let common = with_color_writes(&dev, LoweringInputs::new(), &[(0, UNORM8)]);
        assert_eq!(keys.fragment_inputs.as_ref(), Some(&common), "{stem}");
        let v = round_tripped(&dev, &vertex, &vertex_inputs);
        assert_eq!(*v.key(), keys.vertex, "{stem}");
        let f = round_tripped(&dev, &fragment, &common);
        assert_eq!(Some(*f.key()), keys.fragment, "{stem}");
        let from_compiled = descriptor(StageSource::Compiled(v), StageSource::Compiled(f));
        let compiled_keys = dev
            .render_pipeline_key(&from_compiled, &Budget::standard())
            .expect("keys of the decoded functions");
        assert_eq!(compiled_keys.pipeline, keys.pipeline, "{stem}");
        assert_eq!(compiled_keys.vertex_inputs, None, "{stem}");
        let linked = dev
            .compile_render_pipeline(&from_compiled, &Budget::standard())
            .expect("pipeline from the decoded functions");
        assert_eq!(linked, from_source, "{stem}");
        let cached = RenderPipeline::from_bytes_for_key(&linked.to_bytes(), &linked.key, &dev)
            .expect("pipeline decode");
        assert_eq!(cached, from_source, "{stem}");

        let mut table = StageTable::default();
        table.buffers.insert(0, buffer(24 * 64, 48));
        table.buffers.insert(1, buffer(256, 16));
        let mut state = BindState::default();
        state.stages.insert(Stage::Vertex, table);
        let draw = DrawCall::Primitives {
            vertex_start: 3,
            vertex_count: 6,
            instance_count: 2,
            base_instance: 5,
        };
        let plan = cached.plan(&dev, &state.shape()).expect("plan");
        let original = from_source.plan(&dev, &state.shape()).expect("plan");
        assert_eq!(plan, original, "{stem}");
        assert_eq!(
            plan.instantiate_draw(&state, &draw),
            original.instantiate_draw(&state, &draw),
            "{stem}"
        );
    }
}

// ---- The source scan ----

fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// The AIR front end's crate, whose modules the product re-exports from: a module path that starts
/// with its name is a module of that crate, below its `src/` (`crates/metal2vulkan-air/src/`).
const FRONT_END: &str = "metal2vulkan_air";

/// The file of module `path` (`["bind"]`, `["metal", "format"]`, `["metal2vulkan_air",
/// "metal"]`), and the directory its children live in.
fn module_file(path: &[String]) -> (PathBuf, PathBuf) {
    let (mut dir, path) = match path.split_first() {
        Some((first, rest)) if first == FRONT_END => (
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../metal2vulkan-air")
                .join("src"),
            rest,
        ),
        _ => (src_dir(), path),
    };
    if path.is_empty() {
        return (dir.join("lib.rs"), dir);
    }
    for seg in &path[..path.len() - 1] {
        dir = dir.join(seg);
    }
    let last = &path[path.len() - 1];
    let mod_rs = dir.join(last).join("mod.rs");
    if mod_rs.is_file() {
        return (mod_rs, dir.join(last));
    }
    let file = dir.join(format!("{last}.rs"));
    assert!(file.is_file(), "no source file for module {path:?}");
    (file, dir.join(last))
}

/// One top-level public declaration of a source file.
#[derive(Clone, Debug)]
enum Decl {
    /// `pub <kind> <name>` defined here, at line `line` (0-based) of the file.
    Item {
        name: String,
        kind: Kind,
        hidden: bool,
        line: usize,
    },
    /// `pub use <path>` of `name` (the last path segment, or the `as` name), or of every item
    /// (`*`).
    Use {
        path: Vec<String>,
        name: String,
        hidden: bool,
    },
    /// `pub mod <name>;`.
    Mod { name: String, hidden: bool },
}

/// The attribute and doc lines directly above line `i`.
fn attrs_above(lines: &[&str], i: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut j = i;
    while j > 0 {
        let l = lines[j - 1];
        let attr_like = l.starts_with("#[")
            || l.starts_with("//")
            || l.starts_with(")]")
            || (l.starts_with(' ') && !l.trim().is_empty());
        if !attr_like {
            break;
        }
        out.push(l.trim().to_owned());
        j -= 1;
    }
    out
}

fn ident_prefix(s: &str) -> String {
    s.chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

/// The class of the struct or enum declared at line `i`.
fn class_at(lines: &[&str], i: usize, is_struct: bool) -> Class {
    if attrs_above(lines, i)
        .iter()
        .any(|a| a == "#[non_exhaustive]")
    {
        return Class::N;
    }
    if !is_struct {
        return Class::E;
    }
    let line = lines[i];
    let after_name = line
        .split_once("struct ")
        .map(|(_, r)| r)
        .unwrap_or_default();
    let rest = after_name.trim_start_matches(|c: char| c.is_alphanumeric() || c == '_');
    if let Some(tuple) = rest.strip_prefix('(') {
        let body = tuple.rsplit_once(')').map(|(b, _)| b).unwrap_or(tuple);
        let private = body
            .split(',')
            .map(str::trim)
            .filter(|f| !f.is_empty())
            .any(|f| !f.starts_with("pub ") || f.starts_with("pub("));
        return if private { Class::O } else { Class::E };
    }
    if rest.trim_start().starts_with(';') {
        return Class::E;
    }
    // A braced struct: its fields are the four-space-indented `name: Type` lines.
    for l in &lines[i + 1..] {
        if *l == "}" {
            break;
        }
        let Some(field) = l.strip_prefix("    ") else {
            continue;
        };
        if field.starts_with(' ') || field.starts_with("//") || field.starts_with('#') {
            continue;
        }
        if field.starts_with("pub(") || (!field.starts_with("pub ") && field.contains(':')) {
            return Class::O;
        }
    }
    Class::E
}

/// Splits a use tree (`a::{b::{C,D},E as F}`: no spaces but around `as`) into (path, name)
/// leaves.
fn use_leaves(prefix: &[String], tree: &str, out: &mut Vec<(Vec<String>, String)>) {
    if let Some(open) = tree.find('{') {
        let head: Vec<String> = tree[..open]
            .trim_end_matches("::")
            .split("::")
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();
        let inner = &tree[open + 1..tree.len() - 1];
        let mut depth = 0;
        let mut start = 0;
        let mut parts = Vec::new();
        for (k, c) in inner.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => depth -= 1,
                ',' if depth == 0 => {
                    parts.push(&inner[start..k]);
                    start = k + 1;
                }
                _ => {}
            }
        }
        parts.push(&inner[start..]);
        let mut p = prefix.to_vec();
        p.extend(head);
        for part in parts.into_iter().filter(|s| !s.is_empty()) {
            use_leaves(&p, part, out);
        }
        return;
    }
    let (path, alias) = match tree.split_once(" as ") {
        Some((p, a)) => (p, Some(a)),
        None => (tree, None),
    };
    let mut segs: Vec<String> = prefix.to_vec();
    segs.extend(path.split("::").map(str::to_owned));
    let name = alias
        .map(str::to_owned)
        .unwrap_or_else(|| segs.last().cloned().unwrap_or_default());
    out.push((segs, name));
}

/// The name a generated-enum macro invocation starting at line `i` defines, and its kind. A
/// `metal_enum!` states `#[non_exhaustive]` itself, or is exhaustive.
fn generated_item(lines: &[&str], i: usize) -> Option<(String, Kind)> {
    let header = || {
        lines[i + 1..]
            .iter()
            .map(|l| l.trim())
            .take_while(|l| l.starts_with("///") || l.starts_with("#["))
    };
    let kind = match lines[i] {
        "vk_enum! {" | "vk_flags! {" | "vk_bit! {" => Kind::Struct(Class::O),
        "metal_enum! {" if header().any(|l| l == "#[non_exhaustive]") => Kind::Enum(Class::N),
        "metal_enum! {" => Kind::Enum(Class::E),
        "metal_flags! {" => Kind::Struct(Class::O),
        _ => return None,
    };
    let name = lines[i + 1..]
        .iter()
        .map(|l| l.trim())
        .find(|l| !l.starts_with("///") && !l.starts_with("#["))?;
    Some((ident_prefix(name), kind))
}

/// The (path, name) leaves of the `pub use` statement starting at line `i`, and its last line.
fn use_statement(lines: &[&str], i: usize) -> (Vec<(Vec<String>, String)>, usize) {
    let mut tree = lines[i]["pub use ".len()..].to_owned();
    let mut j = i;
    while !tree.trim_end().ends_with(';') {
        j += 1;
        tree.push_str(lines[j]);
    }
    let tree = tree
        .trim_end()
        .trim_end_matches(';')
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace(":: ", "::")
        .replace(" ::", "::")
        .replace("{ ", "{")
        .replace(" }", "}")
        .replace(", ", ",")
        .replace(",}", "}");
    let mut leaves = Vec::new();
    use_leaves(&[], &tree, &mut leaves);
    (leaves, j)
}

/// Every top-level public declaration of `file`.
fn declarations(file: &Path) -> Vec<Decl> {
    let text = std::fs::read_to_string(file).unwrap_or_else(|e| panic!("{file:?}: {e}"));
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let hidden = || attrs_above(&lines, i).iter().any(|a| a == HIDDEN);
        if let Some((name, kind)) = generated_item(&lines, i) {
            out.push(Decl::Item {
                name,
                kind,
                hidden: false,
                line: i,
            });
        } else if line.starts_with("pub use ") {
            let (leaves, j) = use_statement(&lines, i);
            let h = hidden();
            for (path, name) in leaves {
                out.push(Decl::Use {
                    path,
                    name,
                    hidden: h,
                });
            }
            i = j;
        } else if let Some(rest) = line.strip_prefix("pub mod ") {
            out.push(Decl::Mod {
                name: ident_prefix(rest),
                hidden: hidden(),
            });
        } else if let Some(rest) = line.strip_prefix("pub ") {
            let rest = rest
                .strip_prefix("const fn ")
                .map(|r| ("fn", r))
                .or_else(|| rest.strip_prefix("unsafe fn ").map(|r| ("fn", r)))
                .or_else(|| rest.split_once(' '));
            if let Some((word, name)) = rest {
                let kind = match word {
                    "struct" => Some(Kind::Struct(class_at(&lines, i, true))),
                    "enum" => Some(Kind::Enum(class_at(&lines, i, false))),
                    "trait" => Some(Kind::Trait),
                    "fn" => Some(Kind::Fn),
                    "const" => Some(Kind::Const),
                    "static" => Some(Kind::Static),
                    "type" => Some(Kind::Type),
                    _ => None,
                };
                if let Some(kind) = kind {
                    out.push(Decl::Item {
                        name: ident_prefix(name),
                        kind,
                        hidden: hidden(),
                        line: i,
                    });
                }
            }
        }
        i += 1;
    }
    out
}

/// The kind of `name` as module `module` defines or re-exports it.
fn resolve(module: &[String], name: &str, depth: usize) -> Kind {
    definition(module, name, depth).kind
}

/// Where an item is defined: its file, the line its declaration starts on, and its kind.
struct Definition {
    file: PathBuf,
    line: usize,
    kind: Kind,
}

/// The definition of `name` as module `module` defines or re-exports it.
fn definition(module: &[String], name: &str, depth: usize) -> Definition {
    assert!(depth < 16, "re-export cycle at {module:?}::{name}");
    let (file, _) = module_file(module);
    for d in declarations(&file) {
        match d {
            Decl::Item {
                name: n,
                kind,
                line,
                ..
            } if n == name => return Definition { file, line, kind },
            Decl::Use { path, name: n, .. } if n == name => {
                let (target, item) = split_use(module, &path);
                return definition(&target, &item, depth + 1);
            }
            Decl::Use { path, name: n, .. } if n == "*" => {
                let (target, _) = split_use(module, &path);
                if root_names(&target).contains(name) {
                    return definition(&target, name, depth + 1);
                }
            }
            _ => {}
        }
    }
    panic!("{module:?} names no `{name}`");
}

/// The module and item a use path names, from inside `module`. A path into the product's import
/// of the front end's crate (`crate::air::...`, from `use metal2vulkan_air as air;` at the root)
/// continues in that crate.
fn split_use(module: &[String], path: &[String]) -> (Vec<String>, String) {
    let item = path.last().cloned().expect("a path");
    let segs = &path[..path.len() - 1];
    let in_front_end = module.first().is_some_and(|first| first == FRONT_END);
    let mut target: Vec<String> = match segs.first().map(String::as_str) {
        Some("crate") if in_front_end => {
            let mut t = vec![FRONT_END.to_owned()];
            t.extend_from_slice(&segs[1..]);
            t
        }
        Some("crate") => segs[1..].to_vec(),
        Some(FRONT_END) => segs.to_vec(),
        Some("super") => {
            let mut t = module[..module.len() - 1].to_vec();
            t.extend_from_slice(&segs[1..]);
            t
        }
        Some("self") => {
            let mut t = module.to_vec();
            t.extend_from_slice(&segs[1..]);
            t
        }
        _ => {
            let mut t = module.to_vec();
            t.extend_from_slice(segs);
            t
        }
    };
    if target.is_empty() {
        target = module.to_vec();
    }
    if target.first().is_some_and(|first| first != FRONT_END) {
        if let Some(imported) = front_end_imports().get(&target[0]) {
            target.splice(..1, imported.iter().cloned());
        }
    }
    (target, item)
}

/// The product root's single-line imports from the front end's crate, by the name each binds:
/// `use metal2vulkan_air as air;` binds `air` to the crate's root, `use metal2vulkan_air::budget;`
/// binds `budget` to its module, whatever their visibility.
fn front_end_imports() -> std::collections::BTreeMap<String, Vec<String>> {
    let file = src_dir().join("lib.rs");
    let text = std::fs::read_to_string(&file).unwrap_or_else(|e| panic!("{file:?}: {e}"));
    let mut imports = std::collections::BTreeMap::new();
    for line in text.lines() {
        let rest = line
            .strip_prefix("pub ")
            .or_else(|| line.strip_prefix("pub(crate) "))
            .unwrap_or(line);
        let Some(tree) = rest.strip_prefix("use ").and_then(|t| t.strip_suffix(';')) else {
            continue;
        };
        let mut leaves = Vec::new();
        use_leaves(&[], &tree.replace(", ", ","), &mut leaves);
        for (path, name) in leaves {
            if path.first().is_some_and(|first| first == FRONT_END) {
                imports.insert(name, path);
            }
        }
    }
    imports
}

/// Every name module `module` makes public at its root.
fn root_names(module: &[String]) -> BTreeSet<String> {
    let (file, _) = module_file(module);
    let mut names = BTreeSet::new();
    for d in declarations(&file) {
        match d {
            Decl::Item { name, .. } | Decl::Mod { name, .. } => {
                names.insert(name);
            }
            Decl::Use { path, name, .. } if name == "*" => {
                let (target, _) = split_use(module, &path);
                names.extend(root_names(&target));
            }
            Decl::Use { name, .. } => {
                names.insert(name);
            }
        }
    }
    names
}

/// The surface of module `module` as its source declares it.
fn scanned(module: &'static str) -> Vec<Entry> {
    let path: Vec<String> = module.split("::").map(str::to_owned).collect();
    let (file, _) = module_file(&path);
    let mut out: Vec<Entry> = Vec::new();
    let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
    for d in declarations(&file) {
        match d {
            Decl::Item {
                name, kind, hidden, ..
            } => {
                out.push((module, leak(name), if hidden { Kind::Hidden } else { kind }));
            }
            Decl::Mod { name, hidden } => {
                out.push((
                    module,
                    leak(name),
                    if hidden { Kind::Hidden } else { Kind::Mod },
                ));
            }
            Decl::Use {
                path, name, hidden, ..
            } if name == "*" => {
                let (target, _) = split_use(&path_of(module), &path);
                for n in root_names(&target) {
                    let kind = if hidden {
                        Kind::Hidden
                    } else {
                        resolve(&target, &n, 0)
                    };
                    out.push((module, leak(n), kind));
                }
            }
            Decl::Use {
                path, name, hidden, ..
            } => {
                let kind = if hidden {
                    Kind::Hidden
                } else {
                    let (target, item) = split_use(&path_of(module), &path);
                    resolve(&target, &item, 0)
                };
                out.push((module, leak(name), kind));
            }
        }
    }
    out
}

/// The new API's items at the crate root (`Budget`, `CancelToken`): the root's re-exports that are
/// not hidden, read with their construction classes.
fn scanned_root() -> Vec<Entry> {
    let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
    let mut out: Vec<Entry> = Vec::new();
    for d in declarations(&src_dir().join("lib.rs")) {
        if let Decl::Use {
            path,
            name,
            hidden: false,
        } = d
        {
            let (target, item) = split_use(&[], &path);
            out.push(("", leak(name), resolve(&target, &item, 0)));
        }
    }
    out
}

fn path_of(module: &str) -> Vec<String> {
    module.split("::").map(str::to_owned).collect()
}

/// The module roots and public submodules the scan reads (the crate root is read by
/// [`scanned_root`]: it also declares the internal modules).
fn scanned_modules() -> Vec<&'static str> {
    let mut m = vec![
        "error",
        "device",
        "library",
        "reflection",
        "pipeline",
        "bind",
        "bind::indirect_commands",
        "cache",
        "metal",
        "metal::format",
        "metal::sampler",
        "metal::state",
        "metal::vertex",
        "vk",
    ];
    if cfg!(feature = "ash") {
        m.push("ash_interop");
    }
    m
}

/// The table equals the source: same names, kinds, construction classes, hidden items, and
/// submodules. On a mismatch the message lists both sides' extra entries.
#[test]
fn the_table_matches_the_source() {
    let table: BTreeSet<Entry> = surface_entries().into_iter().collect();
    let mut source: BTreeSet<Entry> = scanned_root().into_iter().collect();
    for m in scanned_modules() {
        source.extend(scanned(m));
    }
    let only_table: Vec<_> = table.difference(&source).collect();
    let only_source: Vec<_> = source.difference(&table).collect();
    if !only_table.is_empty() || !only_source.is_empty() {
        let mut msg = String::from("the public surface changed: update tests/api_surface.rs\n");
        msg.push_str("in the table, not the source:\n");
        for e in &only_table {
            msg.push_str(&format!("  {e:?}\n"));
        }
        msg.push_str("in the source, not the table:\n");
        for e in &only_source {
            msg.push_str(&format!("  {e:?}\n"));
        }
        panic!("{msg}");
    }
}

/// Prints the scanned surface in table syntax, to start a table update from (run with
/// `--run-ignored ignored-only`); the classes of new types and the entry are still decided by hand.
#[test]
#[ignore = "prints the scanned surface; run by hand when the table changes"]
fn print_the_scanned_surface() {
    let modules = scanned_modules().into_iter().map(|m| (m, scanned(m)));
    for (m, entries) in std::iter::once(("", scanned_root())).chain(modules) {
        let path = if m.is_empty() {
            "metal2vulkan".to_owned()
        } else {
            format!("metal2vulkan::{m}")
        };
        println!("    {m:?} => {path} {{");
        for (_, name, kind) in entries {
            let line = match kind {
                Kind::Struct(c) => format!("struct {name} {c:?};"),
                Kind::Enum(c) => format!("enum {name} {c:?};"),
                Kind::Trait => format!("trait {name};"),
                Kind::Fn => format!("fn {name};"),
                Kind::Const => format!("const {name};"),
                Kind::Static => format!("static {name};"),
                Kind::Type => format!("type {name};"),
                Kind::Mod => format!("mod {name};"),
                Kind::Hidden => format!("hidden {name};"),
            };
            println!("        {line}");
        }
        println!("    }}");
    }
}

// ---- The consumer reference ----
//
// `scripts/api-docs` renders the reference of this surface: the library's ordinary rustdoc build
// with every feature. At the crate root it shows the new API's module roots, `Budget`, and
// `CancelToken`, and hides `env_vars` (which the binary reads). Inside the roots,
// `#[doc(hidden)]` items and submodules stay hidden, and `the_table_matches_the_source` pins
// those. With the two tests below, the reference renders exactly the table, and every entry it
// renders has documentation.

/// A hidden item: public, but not API (the crate root's `env_vars`).
const HIDDEN: &str = "#[doc(hidden)]";

/// Every top-level `pub` declaration of the crate root: the names it makes public, and its
/// attribute lines.
fn crate_root_items() -> Vec<(Vec<String>, Vec<String>)> {
    let file = src_dir().join("lib.rs");
    let text = std::fs::read_to_string(&file).unwrap_or_else(|e| panic!("{file:?}: {e}"));
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let attrs = || attrs_above(&lines, i);
        if lines[i].starts_with("pub use ") {
            let (leaves, j) = use_statement(&lines, i);
            out.push((leaves.into_iter().map(|(_, n)| n).collect(), attrs()));
            i = j;
        } else if let Some(rest) = lines[i].strip_prefix("pub ") {
            let name = rest
                .split_whitespace()
                .find(|w| !matches!(*w, "mod" | "struct" | "enum" | "fn" | "const" | "static"))
                .map(ident_prefix)
                .unwrap_or_default();
            out.push((vec![name], attrs()));
        }
        i += 1;
    }
    out
}

/// The reference build renders exactly the table's module roots and root items at the crate
/// root: the public crate-root items that do not carry [`HIDDEN`].
#[test]
fn the_reference_build_renders_exactly_the_surface() {
    let mut rendered = BTreeSet::new();
    for (names, attrs) in crate_root_items() {
        if !attrs.iter().any(|a| a == HIDDEN) {
            rendered.extend(names);
        }
    }
    let mut surface: BTreeSet<String> = surface_entries()
        .into_iter()
        .filter(|&(_, _, kind)| kind != Kind::Hidden)
        .filter_map(|(m, n, _)| match m {
            "" => Some(n.to_owned()),
            m if !m.contains("::") => Some(m.to_owned()),
            _ => None,
        })
        .collect();
    // The crate root declares `ash_interop` whatever the features; the table names it under `ash`.
    surface.insert("ash_interop".to_owned());
    assert_eq!(
        rendered, surface,
        "the crate-root items the reference renders are the table's module roots and root items"
    );
}

/// Whether the declaration at line `line` of `file` carries a doc comment: `///` or `#[doc = ...]`
/// above it, or, for an item a `vk_*!`/`metal_*!` generator emits, inside the invocation before
/// its name.
fn has_docs(file: &Path, line: usize) -> bool {
    let text = std::fs::read_to_string(file).unwrap_or_else(|e| panic!("{file:?}: {e}"));
    let lines: Vec<&str> = text.lines().collect();
    let is_doc = |l: &str| {
        let l = l.trim();
        (l.starts_with("///") && !l.starts_with("////")) || l.starts_with("#[doc =")
    };
    if generated_item(&lines, line).is_some() {
        return lines[line + 1..]
            .iter()
            .take_while(|l| is_doc(l) || l.trim().starts_with("#["))
            .any(|l| is_doc(l));
    }
    attrs_above(&lines, line).iter().any(|a| is_doc(a))
}

/// Whether module `path`'s file opens with an inner doc comment (`//!`).
fn module_has_docs(path: &[String]) -> bool {
    let (file, _) = module_file(path);
    let text = std::fs::read_to_string(&file).unwrap_or_else(|e| panic!("{file:?}: {e}"));
    text.lines()
        .find(|l| !l.trim().is_empty())
        .is_some_and(|l| l.starts_with("//!"))
}

/// Every module root, public submodule, and public entry of the table has documentation where it
/// is defined: the reference build renders each. Documenting fields, variants, and methods is not
/// checked here.
#[test]
fn every_public_entry_is_documented() {
    let mut missing = Vec::new();
    for m in scanned_modules() {
        if !module_has_docs(&path_of(m)) {
            missing.push(format!("module {m}: no `//!` docs"));
        }
    }
    let root = declarations(&src_dir().join("lib.rs"));
    for (m, name, kind) in surface_entries() {
        if matches!(kind, Kind::Hidden | Kind::Mod) {
            continue;
        }
        let def = if m.is_empty() {
            let path = root
                .iter()
                .find_map(|d| match d {
                    Decl::Use { path, name: n, .. } if n == name => Some(path.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("the crate root names no `{name}`"));
            let (target, item) = split_use(&[], &path);
            definition(&target, &item, 0)
        } else {
            definition(&path_of(m), name, 0)
        };
        if !has_docs(&def.file, def.line) {
            missing.push(format!(
                "{m}::{name}: undocumented at {}:{}",
                def.file.display(),
                def.line + 1
            ));
        }
    }
    assert!(
        missing.is_empty(),
        "the consumer reference renders these undocumented:\n  {}",
        missing.join("\n  ")
    );
}
