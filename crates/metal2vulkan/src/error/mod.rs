//! Errors: every way compiling can fail, as typed values to match on.
//!
//! Compiling returns [`Error`]: `Device::from_queries`, `Library::parse`, every `Device::compile_*`
//! call, `Device::render_pipeline_key`, and the [`metal`](crate::metal) mappings. Decoding cached
//! bytes returns
//! [`ArtifactError`](crate::cache::ArtifactError), which is always a cache miss. Match variants,
//! not messages: `Display` is for logs.
//!
//! What each [`Error`] asks of the caller:
//!
//! - [`Error::Unsupported`]: a typed refusal ([`Unsupported`]) of valid Metal. Its [`Feature`]
//!   names the Metal construct, and its [`Needs`] says what would change the answer:
//!   [`Needs::AnyOf`] lists alternative sets of device [`Capability`]s (a device that enables
//!   every capability of one set compiles it), [`Needs::NotImplemented`] is a gap of this
//!   translator version on every device, and [`Needs::NotRepresentable`] has no Vulkan 1.2+
//!   mechanism. The translator does not return different output in place of a refusal; where an
//!   output accepts a documented difference from Metal, its stage lists it as a
//!   [`PrecisionNote`](crate::pipeline::PrecisionNote).
//! - [`Error::NeedsInput`]: the compile needs a pipeline input the call did not supply
//!   ([`RuntimeInput`]): a function-constant value, a storage texture's format, a linked
//!   function, a vertex or stage-input descriptor entry. Supply it and compile again.
//! - [`Error::InvalidInput`] (malformed or ABI-inconsistent AIR: [`InputError`]),
//!   [`Error::InvalidDescriptor`] (an inconsistent descriptor or request: [`DescriptorError`]),
//!   and [`Error::LimitExceeded`] (a deterministic cap: [`Limit`]): the request cannot succeed as
//!   made.
//! - [`Error::Budget`] and [`Error::Cancelled`]: the attempt passed its
//!   [`Budget`](crate::Budget)'s deadline, or its [`CancelToken`](crate::CancelToken) was
//!   cancelled. Transient: retry later.
//! - [`Error::Internal`]: a translator bug ([`InternalError`]). A panic inside a compile entry
//!   point is caught and returned as one.
//!
//! **Caching failures.** [`Error::retry`] says whether the same request can succeed later.
//!
//! Every enum here is `#[non_exhaustive]`, and so are [`Unsupported`], [`InputError`], and
//! [`InternalError`]: new refusals and variants arrive additively, so match with a wildcard arm.

// Every variant is enumerated here up front; a variant discovered later is a one-line append.
// Validation outcome classes are a `From<&Error>` in `validation/`, never a second vocabulary.

use std::fmt;

pub use crate::air::error::{Attachment, DiagName, InputError, InputErrorKind, Location};
// The detail vocabulary of `InputErrorKind`.
pub use crate::air::error::{
    AbiError, BitcodeError, BitstreamError, CanonError, ContainerError, ErrorClass, IntrinsicError,
    VerifyError,
};
// The vocabulary the front end and the translator share: the caps, the deadline, internal
// invariants, and why a linked function cannot resolve a reference.
pub use crate::air::error::{BudgetExceeded, InternalError, InternalKind, Limit, LinkProblem};
// The constants-block records a refusal names (`Feature::Constant`).
pub use crate::air::error::ConstantRecord;
// How a refused generic pointer holds a space only the run knows (`Feature::GenericPointer`).
pub use crate::air::error::GenericPointer;
use crate::air::error::{MissingInput, RequestError};
use crate::device::requirements::{
    ApiVersion, FormatUsage, VkExtension, VkFeature, VkLimit, VkProperty,
};
use crate::pipeline::sym::{Axis, SlotRef, SpecKey, Stage};
use crate::vk;

/// Why a translation or compile did not produce output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Valid Metal that is not representable on this device, or not implemented yet.
    Unsupported(Unsupported),
    /// A pipeline input the caller did not supply.
    NeedsInput(RuntimeInput),
    /// Malformed or ABI-inconsistent input, or input that fails IR verification.
    InvalidInput(InputError),
    /// The caller's descriptor or request is inconsistent.
    InvalidDescriptor(DescriptorError),
    /// A deterministic cap was exceeded (structural caps, work units, ledger bytes). Permanent for
    /// the input.
    LimitExceeded(Limit),
    /// The wall-clock deadline passed. Transient: never cached or recorded as an observation.
    Budget(BudgetExceeded),
    /// The caller cancelled the attempt. Transient.
    Cancelled,
    /// An internal invariant failed: builder error, verifier failure, or a caught panic.
    Internal(InternalError),
}

/// Whether retrying the same request can succeed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Retry {
    /// The outcome is deterministic for this request.
    Never,
    /// The outcome depended on host load or the caller (deadline, cancellation).
    Later,
}

impl Error {
    /// A refusal ([`Error::Unsupported`]) tied to render pipeline attachment `attachment`
    /// ([`Unsupported::at_attachment`]); any other error as it is.
    pub(crate) fn at_attachment(self, attachment: Attachment) -> Error {
        match self {
            Error::Unsupported(u) => Error::Unsupported(u.at_attachment(attachment)),
            other => other,
        }
    }

    /// Whether retrying the same request can succeed: `Cancelled`, `Budget{Time}`, and
    /// `Internal{AllocationFailed}` are `Later`. The last is the host allocator failing a
    /// reservation the byte ledger admitted, which depends on the host's memory, not on the input.
    pub fn retry(&self) -> Retry {
        match self {
            Error::Cancelled | Error::Budget(_) => Retry::Later,
            Error::Internal(internal) if internal.kind == InternalKind::AllocationFailed => {
                Retry::Later
            }
            Error::Unsupported(_)
            | Error::NeedsInput(_)
            | Error::InvalidInput(_)
            | Error::InvalidDescriptor(_)
            | Error::LimitExceeded(_)
            | Error::Internal(_) => Retry::Never,
        }
    }

    /// Where the belief that the input must be refused comes from: a refusal's
    /// ([`Needs::basis`]); [`RefusalBasis::Ours`] for a cap, the deadline or a translator bug
    /// ([`Error::LimitExceeded`], [`Error::Budget`], [`Error::Internal`]); and `None` where no
    /// valid input is refused: malformed input, an inconsistent descriptor, an input the call
    /// did not supply, or the caller's cancellation.
    pub fn basis(&self) -> Option<RefusalBasis> {
        match self {
            Error::Unsupported(refusal) => Some(refusal.needs.basis()),
            Error::LimitExceeded(_) | Error::Budget(_) | Error::Internal(_) => {
                Some(RefusalBasis::Ours)
            }
            Error::NeedsInput(_)
            | Error::InvalidInput(_)
            | Error::InvalidDescriptor(_)
            | Error::Cancelled => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported(u) => write!(f, "unsupported: {u}"),
            Error::NeedsInput(i) => write!(f, "needs pipeline input: {i:?}"),
            Error::InvalidInput(e) => write!(f, "invalid input: {e}"),
            Error::InvalidDescriptor(e) => write!(f, "invalid descriptor: {e:?}"),
            Error::LimitExceeded(l) => write!(f, "limit exceeded: {l:?}"),
            Error::Budget(b) => write!(f, "budget exceeded: {b:?}"),
            Error::Cancelled => f.write_str("cancelled"),
            Error::Internal(e) => write!(f, "internal error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<InputError> for Error {
    /// The mapping of input error kinds to outcome classes, which the front end's own `Error`
    /// makes.
    fn from(e: InputError) -> Error {
        Error::from(crate::air::error::Error::from(e))
    }
}

impl From<crate::air::error::Error> for Error {
    /// The front end's outcome, one variant to one variant: its refusals are
    /// [`Needs::NotImplemented`], and an inconsistent request is an inconsistent descriptor.
    fn from(e: crate::air::error::Error) -> Error {
        use crate::air::error::Error as Air;
        match e {
            Air::Unsupported(u) => Error::Unsupported(u.into()),
            Air::NeedsInput(i) => Error::NeedsInput(i.into()),
            Air::InvalidInput(e) => Error::InvalidInput(e),
            Air::InvalidRequest(r) => Error::InvalidDescriptor(r.into()),
            Air::LimitExceeded(l) => Error::LimitExceeded(l),
            Air::Budget(b) => Error::Budget(b),
            Air::Cancelled => Error::Cancelled,
            Air::Internal(e) => Error::Internal(e),
        }
    }
}

impl From<crate::air::error::Unsupported> for Unsupported {
    fn from(u: crate::air::error::Unsupported) -> Unsupported {
        let crate::air::error::Unsupported {
            feature,
            location,
            symbol,
        } = u;
        Unsupported {
            feature: feature.into(),
            needs: Needs::NotImplemented,
            location,
            symbol,
        }
    }
}

impl From<crate::air::error::Unsupported> for Error {
    fn from(u: crate::air::error::Unsupported) -> Error {
        Error::Unsupported(u.into())
    }
}

impl From<crate::air::error::Feature> for Feature {
    /// Each front-end feature is the translator's feature of the same name.
    fn from(f: crate::air::error::Feature) -> Feature {
        use crate::air::error::Feature as Air;
        match f {
            Air::LibraryKind(kind) => Feature::LibraryKind(kind),
            Air::Stage(tag) => Feature::Stage(tag),
            Air::AddressSpace(space) => Feature::AddressSpace(space),
            Air::ModuleRecord { code } => Feature::ModuleRecord { code },
            Air::Instruction { code } => Feature::Instruction { code },
            Air::Type { code } => Feature::Type { code },
            Air::CallingConvention(cc) => Feature::CallingConvention(cc),
            Air::UnknownIntrinsic => Feature::UnknownIntrinsic,
            Air::StaticInitializer => Feature::StaticInitializer,
            Air::Recursion => Feature::Recursion,
            Air::DynamicCall => Feature::DynamicCall,
            Air::InlineAsm => Feature::InlineAsm,
            Air::Linking => Feature::Linking,
            Air::Constant { record } => Feature::Constant { record },
            Air::Attribute { kind } => Feature::Attribute { kind },
            Air::BitcodeRecord { block, code } => Feature::BitcodeRecord { block, code },
            Air::SyncScope => Feature::SyncScope,
            Air::DirectCall => Feature::DirectCall,
            Air::FunctionConstantArgument => Feature::FunctionConstantArgument,
            Air::IntersectionFunctionTable => Feature::IntersectionFunctionTable,
            Air::Tessellation => Feature::Tessellation,
            Air::RayTracing => Feature::RayTracing,
            Air::Tensors => Feature::Tensors,
            Air::GpuEncodedIndirectCommands => Feature::GpuEncodedIndirectCommands,
            Air::Imageblocks => Feature::Imageblocks,
            Air::ProgramScopeResource => Feature::ProgramScopeResource,
            Air::ProgramScopeSharedSlot => Feature::ProgramScopeSharedSlot,
            Air::ProgramScopeLinked => Feature::ProgramScopeLinked,
            Air::ProgramScopeRead => Feature::ProgramScopeRead,
            Air::GenericPointer(how) => Feature::GenericPointer(how),
        }
    }
}

impl From<MissingInput> for RuntimeInput {
    fn from(i: MissingInput) -> RuntimeInput {
        match i {
            MissingInput::LinkedFunction { symbol } => RuntimeInput::LinkedFunction { symbol },
            MissingInput::DynamicSymbol { symbol } => RuntimeInput::DynamicSymbol { symbol },
            MissingInput::DynamicLibrary { install_name } => {
                RuntimeInput::DynamicLibrary { install_name }
            }
        }
    }
}

impl From<RequestError> for DescriptorError {
    fn from(r: RequestError) -> DescriptorError {
        match r {
            RequestError::EntryNotFound {
                requested,
                available,
            } => DescriptorError::EntryNotFound {
                requested,
                available,
            },
            RequestError::AmbiguousEntry { requested } => {
                DescriptorError::AmbiguousEntry { requested }
            }
            RequestError::FunctionConstantIndex { index } => {
                DescriptorError::FunctionConstantIndex { index }
            }
            RequestError::FunctionConstantType { index } => {
                DescriptorError::FunctionConstantType { index }
            }
            RequestError::DuplicateFunctionConstant { index } => {
                DescriptorError::DuplicateFunctionConstant { index }
            }
            RequestError::UnknownEnumValue { ty, value } => {
                DescriptorError::UnknownEnumValue { ty, value }
            }
            RequestError::LinkedFunction { name, problem } => {
                DescriptorError::LinkedFunction { name, problem }
            }
            RequestError::DuplicateColorOutput { color, index } => {
                DescriptorError::DuplicateColorOutput { color, index }
            }
            RequestError::OverlappingArgument {
                param,
                index,
                count,
            } => DescriptorError::OverlappingArgument {
                param,
                index,
                count,
            },
        }
    }
}

impl From<Unsupported> for Error {
    fn from(u: Unsupported) -> Error {
        Error::Unsupported(u)
    }
}

/// A typed refusal: which Metal-level feature, and what would make it translatable.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unsupported {
    pub feature: Feature,
    pub needs: Needs,
    pub location: Option<Location>,
    /// The symbol the refused construct names, for diagnostics only: an unknown intrinsic, the
    /// callee of a call that needs linking, the pixel format a refused texture, view or copy
    /// holds. Opaque like every [`DiagName`].
    pub symbol: Option<DiagName>,
}

impl Unsupported {
    /// Not implemented by this translator version on any device.
    pub fn not_implemented(feature: Feature) -> Unsupported {
        Unsupported {
            feature,
            needs: Needs::NotImplemented,
            location: None,
            symbol: None,
        }
    }

    /// No Vulkan 1.2+ mechanism exists for these semantics.
    pub fn not_representable(feature: Feature) -> Unsupported {
        Unsupported {
            feature,
            needs: Needs::NotRepresentable,
            location: None,
            symbol: None,
        }
    }

    /// Translatable if the device had every capability of any one alternative.
    pub fn needs_any_of(feature: Feature, alternatives: Vec<Vec<Capability>>) -> Unsupported {
        Unsupported {
            feature,
            needs: Needs::AnyOf(alternatives),
            location: None,
            symbol: None,
        }
    }

    /// Translatable if the device had this one capability.
    pub fn needs(feature: Feature, capability: Capability) -> Unsupported {
        Unsupported::needs_any_of(feature, vec![vec![capability]])
    }

    /// Attach the input location.
    pub fn at(mut self, location: Location) -> Unsupported {
        self.location = Some(location);
        self
    }

    /// Tie the refusal to render pipeline attachment `attachment`, whose state it refuses: its
    /// location gains the attachment, the render pipeline's descriptor where it has none yet. A
    /// refusal already tied to an attachment keeps it.
    pub(crate) fn at_attachment(self, attachment: Attachment) -> Unsupported {
        let location = match &self.location {
            Some(at) if at.attachment().is_some() => return self,
            Some(at) => at.clone(),
            None => Location::render_pipeline(),
        };
        self.at(location.with_attachment(attachment))
    }

    /// Whether the refusal is placed in the input: not where it has no location, nor where its
    /// location is only the render pipeline's descriptor ([`Location::render_pipeline`]), which a
    /// stage compiled under the pipeline places at its own function or metadata
    /// ([`Unsupported::placed_at`]).
    pub(crate) fn is_placed(&self) -> bool {
        self.location
            .as_ref()
            .is_some_and(|at| !at.is_render_pipeline())
    }

    /// Place the refusal at `at`, keeping the render pipeline attachment it is tied to.
    pub(crate) fn placed_at(self, at: Location) -> Unsupported {
        let at = match self.location.as_ref().and_then(Location::attachment) {
            Some(attachment) => at.with_attachment(attachment),
            None => at,
        };
        self.at(at)
    }

    /// Name the symbol the refused construct names (diagnostics only).
    pub fn naming(mut self, symbol: DiagName) -> Unsupported {
        self.symbol = Some(symbol);
        self
    }
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.feature)?;
        if let Some(symbol) = &self.symbol {
            write!(f, " ({symbol})")?;
        }
        write!(f, "; needs {:?}", self.needs)?;
        if let Some(at) = &self.location {
            write!(f, " at {at}")?;
        }
        Ok(())
    }
}

/// What would make a refused input translatable.
///
/// A single device feature or extension (`Needs::Feature(..)`, `Needs::Extension(..)` in prose) is
/// `AnyOf` with one one-capability alternative; see [`Unsupported::needs`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Needs {
    /// Enabling every capability of any one alternative would make it translatable.
    AnyOf(Vec<Vec<Capability>>),
    /// Not implemented by this translator version on any device.
    NotImplemented,
    /// No Vulkan 1.2+ mechanism exists for these semantics.
    NotRepresentable,
}

impl Needs {
    /// Where the belief that the input must be refused comes from: the device's lack
    /// ([`Needs::AnyOf`]) or every Vulkan device's ([`Needs::NotRepresentable`]), or this
    /// translator's gap ([`Needs::NotImplemented`]).
    pub fn basis(&self) -> RefusalBasis {
        match self {
            Needs::AnyOf(_) | Needs::NotRepresentable => RefusalBasis::Device,
            Needs::NotImplemented => RefusalBasis::Ours,
        }
    }
}

/// Where the belief that an input or a call must be refused comes from ([`Error::basis`]), by
/// which a census of the refusals programs meet tells Metal's own refusals from a device's lack
/// and this project's gaps. `docs/REFUSALS.md` lists each refusal's.
///
/// Metal has no counterpart: Metal's `NSError` carries no basis.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RefusalBasis {
    /// Metal itself refuses it, and Apple's own words say so: a quoted assertion, `NSError` or
    /// validation message, a header, or Apple's documentation.
    MetalDocumented,
    /// Metal itself refuses the input, and a run of ours on a named device or release says so.
    MetalMeasured,
    /// The translator's or the runtime's own gap, cap or deadline, or the misuse of a runtime
    /// extension Metal has no counterpart for.
    Ours,
    /// The mechanism is absent: from this Vulkan device, from its driver, or from every Vulkan
    /// device.
    Device,
}

/// One device capability a refusal can name.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Capability {
    Feature(VkFeature),
    Extension(VkExtension),
    Property(VkProperty),
    Limit {
        limit: VkLimit,
        required: u64,
        available: u64,
    },
    Format {
        format: vk::Format,
        usage: FormatUsage,
    },
    ApiVersion(ApiVersion),
}

/// A Metal-level feature (or input construct) that was refused.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Feature {
    // ---- Input constructs outside the typed model ----
    /// A metallib of a kind the container does not read: its header's `file_type` is none of 0 (an
    /// executable library), 1 (a Core Image kernel library) and 2 (a dynamic library).
    LibraryKind(u8),
    /// A function stage the model does not cover.
    Stage(u8),
    /// An LLVM address space the typed model does not define (it defines AIR's `thread` 0
    /// through `threadgroup_imageblock` 4, `object_data` 6, the mesh output handle's 7 and MSL
    /// 4.1's generic 8). The 5 of Apple's AGX ray-tracing and tessellation library modules has no
    /// established meaning and stays refused; a pointer in 6 or 7 outside an object or mesh
    /// function's payload or mesh handle is refused as this feature too, at analysis.
    AddressSpace(u32),
    /// A module-level record outside the accepted subset (alias, ifunc, comdat, …).
    ModuleRecord {
        code: u32,
    },
    /// An instruction record outside the closed instruction set.
    Instruction {
        code: u32,
    },
    /// A type outside the typed model.
    Type {
        code: u32,
    },
    /// A calling convention other than AIR's.
    CallingConvention(u32),
    /// An intrinsic outside the closed set.
    UnknownIntrinsic,
    /// An AGX intrinsic whose semantics are not modeled.
    AgxUnmodeled,
    /// A static initializer that cannot be evaluated.
    StaticInitializer,
    /// Recursion (one variant).
    Recursion,
    /// Indirect calls, function pointers, visible/intersection function tables (one variant).
    DynamicCall,
    /// A call to inline assembly (`asm`) that an entry reaches, other than the empty
    /// `asm sideeffect "", ""` decode drops: the IR gives no assembly a meaning. The refusal names
    /// the expression (`asm "nop", ""`).
    InlineAsm,
    /// Linkage signatures / stitching / dynamic libraries.
    Linking,

    // ---- Control flow and simdgroups ----
    /// Simdgroup width 32 is not achievable for this usage, stage, and device.
    SimdWidth,
    /// A local size beyond a device limit on this axis (`None`: the total).
    LocalSize {
        axis: Option<Axis>,
    },
    /// A kernel's partial-threadgroups variant (`LoweringInputs::partial_threadgroups`) that the
    /// lowering cannot make: a tile function, a kernel that reads a stage-input region, one with
    /// compute derivatives or a cooperative imageblock slice write; a kernel whose simdgroups run
    /// on the emulated tier (its profile's subgroups, its lockstep order or its large body's
    /// pinned width give it that tier), whose exchanges are not walked past the grid; a body whose
    /// uniform barriers need a vote word that the stage's workgroup memory cannot hold beside its
    /// static arena; or the input beside `flat_grid`, `folded_groups`, `replay_bindings` or
    /// `emulated_simdgroups`. The kernel's dispatches keep the plain module's exact split.
    PartialThreadgroups,
    /// An irreducible region containing a convergent instruction.
    IrreducibleConvergent,
    /// Compute derivatives in a flattened simd tier.
    ComputeDerivativesWithSimd,
    /// Derivatives outside fragment without compute-derivative support.
    ComputeDerivatives,
    /// Quad operations in a stage without quad support.
    QuadOperations,
    /// Simdgroup matrix operations.
    SimdgroupMatrix,
    /// A 16x16x16 simdgroup matrix product
    /// (`air.simdgroup_matrix_16x16x16_[widening_]multiply_accumulate`), in any stage. Metal makes
    /// a pipeline of one only on a GPU of the Apple10 family and later: an M3 Max under macOS
    /// 27.0.1 refuses every one ("simdgroup_matrix<T,16,16x16> operations are supported by
    /// GPUFamily10 and later"). The translation refuses it on every device
    /// ([`Needs::NotImplemented`]): its fragment layout, which only such a GPU can show, is not
    /// measured.
    SimdgroupMatrix16x16x16,

    // ---- Memory ----
    /// Argument buffers holding buffer pointers (dynamic roots).
    ArgumentBufferPointers,
    /// Argument buffers holding textures or samplers (handle domain).
    ArgumentBufferTextures,
    /// Argument-buffer members that are pipeline states, ICBs, function tables, or acceleration
    /// structures.
    ArgumentBufferObjects,
    /// Dereferencing a pointer loaded from memory or made from an integer.
    DereferenceLoadedPointer,
    /// Storing a pointer or converting it to an integer.
    PointerToInteger,
    /// Comparing pointers with different roots.
    PointerCompare,
    /// A resource handle used as a byte pointer, or conflicting handle provenance.
    HandleAsPointer,
    /// A constant global whose address escapes.
    ConstantGlobalEscaping,
    /// Threadgroup memory above the device limit.
    ThreadgroupMemorySize,
    /// Threadgroup memory that reads zero until the program writes it, on
    /// `Device::from_queries`' profile: a null initializer of each `Workgroup` variable that holds
    /// it, which needs `shaderZeroInitializeWorkgroupMemory`.
    ZeroedThreadgroupMemory,
    /// `coherent(device)` accesses (Vulkan memory model).
    CoherentMemory,
    /// 64-bit atomics.
    Int64Atomics,
    /// Image atomics beyond 32-bit integers.
    ImageAtomics,
    /// Too many descriptors of one kind in a stage or pipeline.
    DescriptorCount,
    /// A buffer window beyond the descriptor range limits.
    BufferRange,

    // ---- Arithmetic ----
    /// 64-bit integer arithmetic.
    Int64Arithmetic,
    /// 64-bit float arithmetic.
    Float64Arithmetic,

    // ---- Textures and samplers (lowering B2, B3) ----
    /// A pixel or vertex format not supported for the required use.
    Format,
    /// Divergent implicit-LOD sampling or LOD query through a multi-candidate handle.
    DivergentImplicitLod,
    /// Implicit-LOD sampling outside fragment.
    ImplicitLodOutsideFragment,
    /// `min_lod_clamp` without `shaderResourceMinLod`.
    MinLodClamp,
    /// A texel offset the device cannot apply: outside its range, dynamic without the features
    /// `Offset` needs, or on a cube (no offset exists there in Metal or SPIR-V).
    TexelOffset,
    /// Gather with non-constant offsets or components.
    GatherExtended,
    /// Multisampled storage images.
    StorageImageMultisample,
    /// Storage images in the vertex stage.
    StorageTextureInVertexStage,
    /// Cube arrays.
    CubeArray,
    /// Sampler anisotropy an argument buffer lists, on a device without `samplerAnisotropy`.
    SamplerAnisotropy,
    /// Sampler border colors Vulkan cannot express.
    SamplerBorderColor,
    /// Sampler LOD bias.
    SamplerLodBias,
    /// Sampler compare functions outside what Vulkan allows for the image.
    SamplerCompare,
    /// A constexpr sampler with pixel coordinates and a compare function, which Apple's compiler
    /// rejects ("invalid values combination in sampler initialization"), so no AIR Metal's
    /// toolchain writes carries one and Metal gives it no answer.
    PixelSamplerCompare,
    /// Texture buffers beyond the device's texel-buffer limits.
    TexelBuffer,

    // ---- Stages and fragment (lowering B7, B9) ----
    /// Raster order groups without fragment shader interlock.
    RasterOrderGroups,
    /// Framebuffer fetch in a shape not implemented (a write the function would compute over its
    /// own read of the attachment), or that needs a feature the device lacks (sample-rate shading
    /// for a multisampled pass's read).
    FramebufferFetch,
    /// `[[primitive_id]]` in a fragment function: behind a mesh function on a device without
    /// geometry or tessellation shaders, whose fragment stage reads it from a varying no mesh
    /// function writes; and, of a pipeline whose vertex function writes that varying, the
    /// indexed draws its de-indexed form (`VariantKey::primitive_id_indices`) does not run: a
    /// strip, an indirect draw, indices past their buffer, and a vertex function that writes
    /// memory (`NotImplemented`).
    PrimitiveId,
    /// One of Metal's private primitive types (a line loop, a triangle fan, quads, a quad strip)
    /// drawn in the pulled list form (`VariantKey::private_list`), whose vertex function reads the
    /// consumer's list at its device address: on a device without device addresses.
    PrivatePrimitives,
    /// Vertex amplification.
    VertexAmplification,
    /// Layered rendering (`render_target_array_index`, `viewport_array_index`).
    LayeredRendering,
    /// Clip or cull distances.
    ClipDistance,
    /// Dual-source blending.
    DualSourceBlending,
    /// Barycentric coordinates.
    Barycentrics,
    /// Stencil export.
    StencilExport,
    /// Per-sample shading.
    SampleRateShading,
    /// Metal's pull-model interpolation (`interpolant` fragment inputs and their
    /// `interpolate_at_center`, `_centroid`, `_sample` and `_offset` functions) in a form not
    /// lowered: an interpolant read other than as the interpolant of one of its functions (chosen
    /// at run time, stored or passed on), or one a mesh function writes per primitive. On a device
    /// without `sampleRateShading` (which the `InterpolationFunction` capability needs), refused as
    /// needing it. A read at a sample or an offset where some of the quad may have left is
    /// [`Feature::DivergentInterpolation`].
    PullModelInterpolation,
    /// Metal's pull-model interpolation at a sample or an offset (`interpolate_at_sample`,
    /// `interpolate_at_offset`) outside the fragment function's first block, or after a
    /// `discard_fragment` in it, where some of the quad's invocations may have left the control
    /// flow: not lowered, on any device. A device's `InterpolateAtSample` and `InterpolateAtOffset`
    /// may read the quad's other invocations, and RADV's read garbage where those have left (in a
    /// 1x1 framebuffer, whose quad holds three helper invocations), so the shape is a gap rather
    /// than any device's answer. The same reads in the first block, before any discard, are
    /// lowered.
    DivergentInterpolation,
    /// The pixel's coverage (`[[sample_mask]]` read) in a fragment function that runs once per
    /// sample: Metal hands every invocation the whole pixel's coverage, while a Vulkan invocation
    /// under sample shading sees only the bits of the samples it shades, and nothing gathers a
    /// pixel's sample invocations.
    PixelCoverageAtSampleRate,
    /// Too many interface components or locations.
    InterfaceComponents,
    /// A fragment input without a vertex producer (a Metal link error).
    UnmatchedVarying,
    /// Kernel `[[stage_in]]`: a stage-input descriptor combination whose Metal meaning is not
    /// established.
    KernelStageIn,
    /// A vertex buffer layout stepped per instance at a rate above 1, on a profile that fetches
    /// every attribute in fixed function (the identity domains, where the pulled fetch is not
    /// taken): fixed-function input divides the instance index only with
    /// `vertexAttributeInstanceRateDivisor`, a rate within `maxVertexAttribDivisor`, and
    /// `supportsNonZeroFirstInstance`, as a Metal draw may carry a base instance. The refusal
    /// names the feature, else the limit, the profile lacks.
    VertexStepRate,
    /// A vertex attribute whose layout fixed-function input cannot fetch, on a profile that fetches
    /// every attribute in fixed function (the identity domains, where the pulled fetch is not
    /// taken): an offset or static stride that is not a multiple of the format's component size
    /// (Metal takes any byte multiple), an offset or stride past `maxVertexInputAttributeOffset` or
    /// `maxVertexInputBindingStride`, more attributes or layouts than `maxVertexInputAttributes` or
    /// `maxVertexInputBindings`, or an attribute that ends past its layout's static stride where
    /// the strides are dynamic state.
    VertexInputLayout,

    // ---- the refused families, not implemented yet ----
    /// Tessellation: a post-tessellation vertex function and its pipeline's patch draws, through
    /// the device's tessellation stages (`tessellationShader`).
    Tessellation,
    MeshShading,
    TileShading,
    /// An imageblock outside the forms implemented: a multisample imageblock's operation (its
    /// sample count, the colors a pixel holds, a color's coverage mask, or a store to the samples
    /// a coverage mask selects); a fragment imageblock on a layered or amplified pass, written
    /// under alpha to coverage, or whose write a test may drop where no form runs the test first;
    /// or an explicit layout, imageblock data operand or slice write of a shape not modelled. Tile
    /// functions run on single-sample passes; fragment imageblocks hold a block per sample.
    Imageblocks,
    RayTracing,
    Tensors,
    RasterizationRateMap,
    GpuEncodedIndirectCommands,

    // ---- Device ----
    /// The device's Vulkan version is below the 1.2 floor.
    VulkanVersion,
    /// The description's `maxPushConstantsSize` is below 128 bytes, Vulkan's least and the least
    /// push window a stage is lowered for: a compute stage's dispatch region and buffer addresses,
    /// a render pipeline's two stages together.
    PushWindow,
    /// The identity pointer and handle domains (`PointerDomain::Identity`,
    /// `HandleDomain::Identity`): on a device without `bufferDeviceAddress` and `shaderInt64`,
    /// and, on any device, a compile of a function they do not lower yet: anything but a kernel
    /// that reaches memory only through its buffer slots.
    IdentityDomains,
    /// A constants-block record LLVM defines that the typed model has no form for (a block
    /// address, a `select` and the other constant expressions of [`ConstantRecord`]), by LLVM's
    /// name for it.
    Constant {
        record: ConstantRecord,
    },
    /// An LLVM attribute kind that changes the calling ABI and that AIR does not use
    /// (`inalloca`, `preallocated`, `byref`, `nest`, the `swift*` kinds).
    Attribute {
        kind: u32,
    },
    /// A bitcode record LLVM defines in `block` that the typed model does not cover and that has
    /// no more specific variant (legacy spellings predating module version 2, such as the old
    /// metadata node and string records and pre-group attribute entries).
    BitcodeRecord {
        block: u32,
        code: u32,
    },
    /// An atomic access or fence in a synchronization scope other than LLVM's built-in
    /// `singlethread` and system scopes.
    SyncScope,
    /// Simdgroup operations of a class the stage's subgroups do not support (the stage is not in
    /// `subgroupSupportedStages`, or an operation class is missing from
    /// `subgroupSupportedOperations`).
    SubgroupOperations,
    /// A call to a defined function the inliner cannot take: a variadic callee, a `byval`
    /// argument, whose private copy inlining would have to materialize, or a call at another
    /// function type whose arguments match the parameters (another return type, or identically
    /// shaped structs of other names). A call at a function type whose arguments differ from the
    /// parameters in count or shape is `InvalidInput(Canon(CallTypeMismatch))`.
    DirectCall,
    /// A switch case literal wider than 32 bits: the structurizer emits 32-bit `OpSwitch`
    /// selectors.
    WideSwitchLiteral,
    /// An instruction, intrinsic, or interface element of a lowering family whose lowering has
    /// not landed yet (each family narrows its refusals as it lands).
    Lowering(LowerFamily),
    /// A sampler address mode the device lacks (`MirrorClampToEdge`).
    SamplerAddressMode,
    /// `alphaToOneEnabled` on a device without Vulkan's `alphaToOne` feature, the exact
    /// counterpart (`VkPipelineMultisampleStateCreateInfo::alphaToOneEnable`). A device that has
    /// the feature draws the pipeline.
    AlphaToOne,
    /// Buffer or image stores or atomics in a vertex or fragment function without the stage's
    /// store feature (`vertexPipelineStoresAndAtomics`, `fragmentStoresAndAtomics`).
    StageStoresAndAtomics,
    /// An observed read of a pointer entry argument that a function-constant predicate removes,
    /// other than an absent buffer (which reads as null) or an absent texture or sampler read only
    /// as the texture or sampler operand of texture calls (an absent read): a texture used another
    /// way, an element of a texture table (a read, query, or sample; a write to or fence of the
    /// texture itself is dropped, as Metal drops it), threadgroup memory, an object, a
    /// pointer-typed imageblock or control-point input, or a sampler merged with a present one. The
    /// argument is not declared (MSL 5.8.1.2) and Metal defines no value for it (its compiler drops
    /// a store of such a read and gives arithmetic on it NaN), and no contained answer for a
    /// dereference is established. An absent read yields zero.
    /// An absent argument passed by value (its parameter is not a pointer: a vertex attribute, a
    /// stage input, a varying, a color input, a builtin, imageblock data, an indirect constant, a
    /// per-patch attribute) reads zero instead. A read nothing observes is `undef`, or goes; a
    /// return observes only the outputs the predicates keep, so a read that feeds only a removed
    /// output is unobserved.
    FunctionConstantArgument,
    /// A use of Metal's bicubic filter that no Metal reading establishes: `min_lod_clamp` (Apple's
    /// device compiler fails to build the pipeline), `mirrored_repeat`, a dynamic offset, a
    /// comparison, an LOD query, a 3D, cube, depth, integer, or heap texture, min/max reduction, or
    /// anisotropy; at bind time, a bound pixel format of no measured family. Every other bicubic
    /// sample is Metal's measured filter emulated by fetches (`lower::sampler::bicubic`).
    SamplerBicubicFilter,
    /// A dispatch whose threadgroup count on `axis` exceeds `maxComputeWorkGroupCount` by more
    /// than the binder splits into regions (`None`: the indirect arguments' offset), or whose grid does not
    /// fit Metal's 32-bit `threads_per_grid`.
    GroupCount {
        axis: Option<Axis>,
    },
    /// A ray intersection through intersection functions: an `intersect` call with a non-null
    /// intersection function table or an intersection function buffer, whose candidates Metal
    /// hands to the table's functions.
    IntersectionFunctionTable,
    /// Color attachments whose write masks or blend states differ, as Metal sets them per
    /// attachment, on a device without `independentBlend`, where no identical set of states
    /// leaves every attachment as Metal would.
    IndependentBlending,
    /// A `rasterSampleCount` outside the device's framebuffer sample counts for the pipeline's
    /// attachments (`framebuffer*SampleCounts`), and, for a color attachment, outside its own
    /// format's.
    RasterSampleCount,
    /// The render encoder's `MTLTriangleFillModeLines` (`polygonMode` `LINE`) on a device
    /// without `fillModeNonSolid`.
    TriangleFillLines,
    /// The render encoder's `TriangleFillMode::Points` (`polygonMode` `POINT`) on a device
    /// without `fillModeNonSolid`, or on a portability-subset device without `pointPolygons`.
    TriangleFillPoints,
    /// The render encoder's `MTLDepthClipModeClamp` (`depthClampEnable`) on a device without
    /// `depthClamp`.
    DepthClipClamp,
    /// The render encoder's depth bias clamp, nonzero with a nonzero bias, on a device without
    /// `depthBiasClamp`.
    DepthBiasClamp,
    /// `CONSTANT_ALPHA` or `ONE_MINUS_CONSTANT_ALPHA` (Metal's `BlendAlpha` factors) as a color
    /// blend factor on a portability-subset device without `constantAlphaColorBlendFactors`.
    ConstantAlphaBlendFactor,
    /// A sparse texture (`MTLTexture.isSparse`) bound at `slot`, where a function observes the
    /// residency of its accesses (`sparse_color::resident()`) and the module has no sparse form
    /// for them (`RuntimeFact::SparseTexture`): it needs `shaderResourceResidency`, or, where the
    /// device has it, an access has none (a bicubic sample, not implemented). A texture that is
    /// not sparse is answered on every device.
    SparseTexture {
        slot: SlotRef,
    },
    // The runtime's refusals of texture objects and texture commands. Each depends on the Vulkan
    // device or on a Metal measurement, so another device or a later measurement can lift it;
    // what no Vulkan device can ever have is the runtime's Apple-only class instead.
    /// A texture swizzle whose Metal behavior is not measured.
    TextureSwizzle,
    /// A stage whose private memory (its `thread` variables and arrays, and the constants it
    /// holds per thread) takes more Function storage a thread than the device holds correctly
    /// (`VkLimit::FunctionStorageBytesPerThread`): its measured value, or on a device not measured
    /// the floor of 16 KiB, and for a vertex or fragment function there the 192 KiB Metal was
    /// measured to hold. A stage with a call stack is refused as `Recursion` instead.
    PrivateMemory,
    /// Reflection of an argument whose answer Metal's has not been measured for (`reflection`):
    /// a type the MSL type grammar does not cover, a binding kind no probe has asked Metal about
    /// (explicit imageblocks and imageblock data, argument-buffer function tables, acceleration
    /// structures and render pipeline states), or, in a library's answer, an index a function
    /// constant computes. The symbol names the argument's type.
    Reflection,
    /// A `depth_stencil_state` a function reads from an argument buffer or takes as a parameter,
    /// the state of an indirect render command, which no record holds. (A render command's state
    /// setters, `render_command::set_cull_mode` and the rest, are `GpuEncodedIndirectCommands`, as
    /// every encoder operation is.) Metal's Apple10 family alone lists the feature, and Metal's
    /// compiler refuses such a function on an Apple9 GPU. The translation refuses it on every
    /// device ([`Needs::NotImplemented`]).
    IndirectRenderState,
    /// A fragment function's read of a Metal 3.1 program-scope built-in variable that Metal does
    /// not lower there: a global `thread_index_in_quadgroup` (`air.get_thread_index_in_quadgroup`),
    /// which Apple's compiler takes in a fragment function but whose render pipeline the M4 and
    /// the M3 Max refuse, while they run the same built-in as the fragment's parameter. The
    /// translation refuses it on every device ([`Needs::NotImplemented`]).
    GlobalBuiltinInFragment,
    /// A private fixed-function property of Metal's pipeline descriptor (Apple SPI, OpenGL on
    /// Metal's) set to a value whose effect the translation has not built: the property it names
    /// ([`PrivateRenderField`]). Metal's answer for each was measured; building it is open work.
    PrivateRenderState(PrivateRenderField),
    /// A texture, a sampler or threadgroup memory bound at program scope (MSL 3.2's
    /// `!air.global_bindings`), one or an array, that the entry reads: only a bound buffer becomes
    /// a parameter. Metal takes each: an Apple M3 Max under macOS 27.0.1 runs kernels that read a
    /// bound texture and sampler, a bound sampler array and bound threadgroup memory (m3-max-70306).
    /// Basis: metal-takes (an Apple M3 Max under macOS 27.0.1, potluck job m3-max-70306)
    ProgramScopeResource,
    /// A buffer bound at program scope at a slot that a parameter of the entry, or another binding
    /// it reads, binds with another role or type. Metal takes it: on an Apple M3 Max under macOS
    /// 27.0.1, a `device float *` and a `constant uint *` bound at one index read the one buffer
    /// bound there (m3-max-70268).
    /// Basis: metal-takes (an Apple M3 Max under macOS 27.0.1, potluck job m3-max-70268)
    ProgramScopeSharedSlot,
    /// A program-scope binding in the module of a function a pipeline links (a visible function,
    /// or a dynamic library's). Metal takes it: on an Apple M3 Max under macOS 27.0.1, a kernel's
    /// pipeline links another library's visible function, which reads the buffer the kernel's
    /// stage binds at the global's index (m3-max-70306).
    /// Basis: metal-takes (an Apple M3 Max under macOS 27.0.1, potluck job m3-max-70306)
    ProgramScopeLinked,
    /// A read of a buffer bound at program scope other than a load of the pointer its global
    /// holds. `xcrun metal` 32023.922 writes every read so, `&p` included, so only other AIR reaches
    /// it.
    ProgramScopeRead,
    /// A generic pointer (MSL 4.1's `addrspace(8)`) that canonicalization cannot make a pointer of
    /// one address space, by how it comes to hold a space only the run knows
    /// ([`GenericPointer`]). Vulkan's logical addressing has no pointer that may point into either
    /// of two storage classes.
    GenericPointer(GenericPointer),
    /// A sequentially consistent fence (`seq_cst` `atomic_thread_fence`, an LLVM `fence seq_cst`,
    /// or a `seq_cst` barrier ordering device memory), whose total order with the command
    /// buffer's other such fences the translation keeps through an atomic of the command buffer's
    /// `ExecutionStatus` word: in a vertex or fragment stage without its stores and atomics.
    SeqCstFence,
    /// The coverage after the depth and stencil tests (`[[sample_mask, post_depth_coverage]]`).
    /// Vulkan's one form reads `SampleMask` under `VK_EXT_post_depth_coverage`, which no runtime
    /// enables; no form is built.
    PostDepthCoverage,
    /// Lines at a width other than 1 (Apple's `setLineWidth:`), drawn by the device's own wide
    /// lines (`wideLines`) or as bands of triangles through synthesized tessellation stages
    /// (`tessellationShader`, `pipeline::link::band`).
    WideLines,
}

/// A private property of Metal's render pipeline descriptor that
/// [`Feature::PrivateRenderState`] refuses at a value other than Metal's default
/// (`PrivateRenderState`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrivateRenderField {
    /// `sampleCoverage` other than 1 at a sample count past 4, where Metal was not measured.
    SampleCoverage,
    /// `alphaTestEnabled` with a test the translation has not built: an `alphaTestFunction` past
    /// `MTLCompareFunctionAlways`, a pipeline without a fragment function, a fragment function
    /// whose color 0 is not four `float` or `half` lanes (Metal's compiler fails a function that
    /// writes no color 0 under the test), or one that loops over its covered samples.
    AlphaTest,
    /// `logicOperationEnabled` with an operation other than copy and no-op: on a device without
    /// `logicOp` (`Unsupported::needs` naming it), or where Vulkan's operation is not Metal's (an
    /// sRGB attachment, a float attachment that blends, a format whose operation Metal was not
    /// measured with).
    LogicOperation,
    /// `pointSmoothEnabled`.
    PointSmooth,
    /// `explicitVisibilityGroupID` other than the default in a pipeline whose
    /// `maxVertexAmplificationCount` is past 1.
    ExplicitVisibilityGroup,
    /// `fragmentDepthCompareClampMask` other than 0 in a pipeline whose fragment function makes a
    /// depth comparison through an argument buffer's sampler, which no bit of the mask names
    /// (Metal's answer unmeasured).
    FragmentDepthCompareClamp,
    /// `textureWriteRoundingMode` 2 (`TextureWriteRoundingMode::NearestEven`) in a pipeline whose
    /// fragment function writes a texel that Metal's mode 2 stores otherwise than its default (a
    /// `float` texel into a half-float or packed small-float texture, `RG11B10Float` or
    /// `RGB9E5Float`, or a `half` texel into a packed small-float one), whose mode-2 rounding the
    /// translation has not built: refused at the pipeline's making where the write's slot
    /// declares such a format, and at the draw where the bound texture decides and is one.
    TextureWriteRounding,
}

/// A family of AIR operations the translator lowers, as `Feature::Lowering` names it in a
/// refusal of an operation it does not lower yet.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LowerFamily {
    /// Math, conversions, comparisons, composites (`lower::arith`).
    Arith,
    /// Loads, stores, address arithmetic, atomics, memory intrinsics (`lower::access`).
    Access,
    /// Image operations (`lower::texture`).
    Texture,
    /// Sampling (`lower::sampler`).
    Sampler,
    /// Simdgroup, quad, and matrix operations (`lower::simd`).
    Simd,
    /// Barriers and derivatives (`lower::sync::barrier`).
    Barrier,
    /// Fences and the memory model (`lower::sync::memory_model`).
    MemoryModel,
    /// Fragment-only operations (`lower::fragment`).
    Fragment,
    /// Stage inputs, outputs, and varyings (`lower::stage_io`).
    StageIo,
    /// Dispatch builtins and regions (`lower::dispatch`).
    Dispatch,
    /// A post-tessellation vertex function's patch inputs and builtins (`lower::tessellation`).
    Tessellation,
    /// Ray intersection and acceleration structures (`lower::raytracing`).
    RayTracing,
    /// An object function's mesh grid and a mesh function's outputs (`lower::mesh`).
    Mesh,
}

/// A pipeline input that the caller did not supply.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RuntimeInput {
    /// The format of a storage texture on a profile that cannot access it without one: the
    /// slot's `StorageImageChoice`.
    StorageTextureFormat { slot: SlotRef },
    /// A call through a visible-function reference (`!air.visible_function_references`)
    /// survived specialization, and no linked function of the descriptor
    /// (`MTLLinkedFunctions`) has the name it references. `symbol` is that name.
    LinkedFunction { symbol: DiagName },
    /// The vertex descriptor entry of an attribute that only an in-shader fetch can read: a
    /// post-tessellation vertex function's patch or control-point attribute, which has no
    /// fixed-function input to fall back on.
    VertexDescriptor { attribute: u32 },
    /// The stage-input descriptor entry of a kernel `[[stage_in]]` attribute
    /// (`MTLComputePipelineDescriptor.stageInputDescriptor`): a kernel fetches its stage inputs in
    /// the shader, so without the descriptor there is nothing to fetch them by.
    StageInputDescriptor { attribute: u32 },
    /// A kernel reaches the implicit imageblock, which only a tile function has
    /// (`LoweringInputs::tile_attachments`): its cells are the tile render pipeline's color
    /// attachments, whose formats a tile pipeline descriptor gives.
    TileAttachments,
    /// A fragment function linked into a render pipeline that rasterizes more than one sample
    /// runs per sample and reads `[[position]]` or a center-sampled varying, which it reads at the
    /// pixel center only when compiled for such a pipeline (`LoweringInputs::multisampled`):
    /// compiled without it, a sample-rate invocation would read them at its sample. A shader-side
    /// stencil is compiled for the pipeline's kind either way: its shadow is multisampled exactly
    /// when the pipeline is.
    Multisampled,
    /// A fragment function reads `[[viewport_array_index]]`, which it reads from the raw varying
    /// a vertex function that writes the index adds (`LoweringInputs::viewport_index_varying`), and
    /// was compiled for the other kind of vertex function: compiled without the field behind such a
    /// vertex function it would read the clamped `ViewportIndex`, and compiled under it behind
    /// another a varying no stage writes. It is compiled for the pipeline again.
    ViewportIndex,
    /// A fragment function reads `[[render_target_array_index]]`, which it reads from the raw
    /// varying a vertex function that writes the index adds
    /// (`LoweringInputs::render_target_array_index_varying`), and was compiled for the other kind
    /// of vertex function: compiled without the field behind such a vertex function it would read
    /// `Layer`, which holds 0 for an index at or past the pass's length (or for any index in a pass
    /// of one layer), and compiled under it behind another a varying no stage writes. It is
    /// compiled for the pipeline again.
    RenderTargetArrayIndex,
    /// A fragment function reads its `[[primitive_id]]` from the varying its pipeline's vertex
    /// function writes (`IoRole::PrimitiveId`, on a device that cannot declare a fragment
    /// `PrimitiveId`), and the vertex function was compiled without
    /// `LoweringInputs::primitive_id_varying`, so it writes none; or the vertex function writes it
    /// and the fragment function, compiled for another vertex function, reads Vulkan's
    /// `PrimitiveId`. It is compiled for the pipeline again.
    PrimitiveId,
    /// A fragment function writes a color attachment whose format converts its float output (a
    /// normalized, sRGB or half-float format) and was compiled for other attachments, another
    /// blend state, or another device's conversion: its map of how each such output reaches the
    /// attachment (`LoweringInputs::color_writes`: which conversion records a note, and a
    /// half-float channel's rounding under `TextureWriteRoundingMode::NearestEven`) is not the
    /// pipeline's. It is compiled for the pipeline again.
    ColorWrites,
    /// A call to a function, or a use of a variable, that the stage's module imports from a
    /// dynamic library (a declaration its `air.dyld_flat_table` names) survived specialization,
    /// and no dynamic library the stage links exports it: neither its preloaded libraries nor
    /// those loaded for the install names its search reaches (`FunctionDescriptor::libraries`,
    /// `MTLDynamicLibrary`). `symbol` is its name. Metal's pipeline creation fails the same way
    /// ("Undefined symbols").
    DynamicSymbol { symbol: DiagName },
    /// An import's use survived specialization, and an install name the stage's search reaches
    /// has no loaded library (`DynamicLibraries::loaded`; Metal could not load its file):
    /// Metal's creation fails then even when a preloaded library exports every symbol.
    /// `install_name` is the first such name.
    DynamicLibrary { install_name: DiagName },
    /// A fragment function reads a varying that its mesh render pipeline's mesh function writes
    /// per primitive, which it reads per primitive (`PerPrimitiveEXT`) only when compiled for
    /// that mesh function's primitive outputs (`LoweringInputs::primitive_varyings`), or reads per
    /// primitive one that its pipeline's vertex or mesh function writes per vertex: it was
    /// compiled for another pipeline, and is compiled for this one again.
    PrimitiveVaryings,
    /// A mesh function's payload arena is smaller than its mesh render pipeline's object
    /// function's, or an object function's (a mesh function's without one) is smaller than its
    /// descriptor's `payloadMemoryLength`: compiled on its own or for another pipeline, it would
    /// read zeros past its own payload where Metal reads the pipeline's
    /// (`LoweringInputs::object_payload`). It is compiled for this pipeline again.
    ObjectPayload,
    /// A fragment function in a render pipeline whose private sample mask drops a sample writes
    /// memory, reads `[[sample_mask]]` or declares `[[early_fragment_tests]]`, so it applies the
    /// mask itself (`LoweringInputs::sample_mask`), as Metal applies it after the function runs;
    /// or it applies another mask. It was compiled for another pipeline, and is compiled for this
    /// one again.
    SampleMask,
    /// A fragment function in a render pipeline with the private alpha test on runs the test
    /// itself (`LoweringInputs::alpha_test`), as Metal tests its color 0's alpha after it runs;
    /// or it runs another comparison. It was compiled for another pipeline, and is compiled for
    /// this one again.
    AlphaTest,
    /// A fragment function in a render pipeline with a private `fragmentDepthCompareClampMask`
    /// clamps the references of its depth comparisons through the mask's samplers itself
    /// (`LoweringInputs::depth_compare_clamp_mask`), as Metal clamps them; or it clamps others. It
    /// was compiled for another pipeline, and is compiled for this one again.
    DepthCompareClamp,
    /// A fragment function reads flat an interpolated varying that its pipeline's vertex function
    /// does not write the same at every vertex of a primitive: compiled for another vertex
    /// function, which wrote it so (`LoweringInputs::uniform_varyings`), it would read the
    /// provoking vertex's value where Metal interpolates. It is compiled for this pipeline again.
    UniformVaryings,
}

/// How Metal answers a pipeline it refuses, where a [`DescriptorError`] carries it
/// ([`DescriptorError::ColorAttachment`]): measured with plain Metal, no API validation layer, on
/// an Apple M3 Max under macOS 27.0.1.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MetalAnswer {
    /// The creation call answers nil and an `NSError` (code 3, in the GPU driver's error domain,
    /// `AGXMetalG15X_M1` on the device measured), and the process goes on.
    NilWithError,
    /// Metal's descriptor validation asserts ("Render Pipeline Descriptor Validation") and the
    /// process ends (`SIGABRT`). Also the answer of a case whose Metal answer is not measured.
    EndsProcess,
}

/// The caller's descriptor or request is inconsistent.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DescriptorError {
    /// The requested entry point is not in the library.
    EntryNotFound { requested: DiagName, available: u32 },
    /// More than one entry matches the request.
    AmbiguousEntry { requested: DiagName },
    /// The entry exists but has a different stage than the descriptor slot it was given.
    StageMismatch { expected: Stage, found: Stage },
    /// A function-constant index that the function does not declare, in a request made to the
    /// canonicalizer directly (`air::SpecRequest`, a linked module's constants). A descriptor's
    /// values never raise it: the compile ignores a value for a constant the function does not
    /// declare, as Metal's `newFunctionWithName:constantValues:` does.
    FunctionConstantIndex { index: u32 },
    /// A function-constant value of the wrong type.
    FunctionConstantType { index: u32 },
    /// The same function constant supplied twice.
    DuplicateFunctionConstant { index: u32 },
    /// A vertex descriptor attribute the vertex function does not declare, or of the wrong type.
    VertexAttribute { attribute: u32 },
    /// A vertex descriptor's buffer `layout` whose layout and attributes disagree as Metal stops
    /// at: a set attribute names it and it has no layout, or it has a layout of a nonzero or
    /// dynamic stride and no set attribute names it.
    VertexLayout { layout: u32 },
    /// Color attachment `index` and the pipeline's function or blend disagree, and Metal refuses
    /// the pipeline. `metal` is how, as plain Metal answers each case on an Apple M3 Max under
    /// macOS 27.0.1 (`metal2vulkan-conformance/probes/color_attachment_answers.m`):
    ///
    /// - [`MetalAnswer::NilWithError`]: an output, or a `[[color(n)]]` input, of another numeric
    ///   class than the attachment's format ("output of type half4 is not compatible with a
    ///   MTLPixelFormatRGBA8Uint color attachment."; "input of type uint4 is not compatible
    ///   ..."); an output with fewer components than the format's channels ("fragment shader
    ///   color output does not have enough components for the pixel format"); a blend of the
    ///   second source whose function writes no second output, or no first ("Fragment shader
    ///   does not write to render target color(0), index(1) that is required for blending"); and
    ///   a `[[color(n)]]` input, or a tile function's imageblock member, of an attachment without
    ///   a format ("Shaders reads from a color attachment whose pixel format is
    ///   MTLPixelFormatInvalid"; "Invalid pixel format for render target 1").
    /// - [`MetalAnswer::EndsProcess`]: blending enabled on a bfloat attachment (`R16Bfloat`,
    ///   `RG16Bfloat`, `RGBA16Bfloat`), whatever the function writes and with no function ("the
    ///   pixelformat MTLPixelFormatRGBA16Bfloat for this render target is not blendable"); and
    ///   alpha to coverage over an integer attachment 0, rasterizing or not
    ///   ("MTLRenderPipelineDescriptor with alphaToCoverageEnabled requires a float/unorm pixel
    ///   format"). Also a `StageSource::Compiled` tile function compiled under another format of
    ///   the attachment than the descriptor gives it: Metal compiles a tile function for its
    ///   pipeline, so the case has no Metal answer to measure, and is answered as unmeasured.
    ColorAttachment { index: u32, metal: MetalAnswer },
    /// A raw Metal enum value outside the known set.
    UnknownEnumValue { ty: &'static str, value: u32 },
    /// A pipeline descriptor without a required stage.
    MissingStage { stage: Stage },
    /// A render pipeline that rasterizes whose vertex function writes no `[[position]]`: one
    /// returning `void`, as every other vertex function Metal's compiler builds returns its
    /// position. `newRenderPipelineStateWithDescriptor:` refuses it ("RasterizationEnabled is true
    /// but the vertex shader's return type is void", on an M3 Max under macOS 27.0.1), and makes it
    /// under `PrivateRenderState::open_gl_mode_enabled` as a state that rasterizes nothing, which
    /// is the descriptor with rasterization disabled: the link refuses it whatever the mode.
    RasterizedVoidVertex,
    /// A device description (or the Khronos profile JSON it was read from) is malformed or
    /// inconsistent.
    DeviceDescription { detail: String },
    /// A Metal enum value that is a placeholder, not a concrete value (`MTLPixelFormatInvalid`,
    /// `...Unspecialized`), where a concrete one is required.
    PlaceholderEnumValue { ty: &'static str, value: u32 },
    /// A per-instance vertex buffer layout with a step rate of 0.
    VertexStepRate { step_rate: u32 },
    /// A sampler whose LOD clamps are NaN, or whose minimum exceeds its maximum.
    SamplerLodClamp,
    /// Two stages of one pipeline declare specialization constant `key` with a different size or
    /// default: they were compiled for different pipelines or profiles.
    SpecializationConflict { key: SpecKey },
    /// A `StageSource::Compiled` function compiled for another device profile.
    ForeignCompiledFunction,
    /// A pixel format (raw `MTLPixelFormat` ordinal) used in a way its aspects do not allow: a
    /// color attachment of a depth or stencil format, or a depth/stencil attachment of a color
    /// format (Metal rejects either descriptor).
    PixelFormatUsage { format: u32, usage: FormatUsage },
    /// A linked function (`MTLLinkedFunctions`) that cannot resolve the visible-function
    /// reference naming it. `name` is the function's name.
    LinkedFunction {
        name: DiagName,
        problem: LinkProblem,
    },
    /// A render or mesh render pipeline descriptor whose `maxVertexAmplificationCount` is 0: its
    /// encoders draw with a count in `[1, maxVertexAmplificationCount]`, so the maximum is at
    /// least 1. Metal's descriptor validation ends the process at a mesh descriptor's
    /// ("maxVertexAmplificationCount (0) cannot be set to zero.").
    VertexAmplificationCount { count: u32 },
    /// A kernel `[[stage_in]]` attribute the compute pipeline's stage-input descriptor does not
    /// describe.
    StageInputAttribute { attribute: u32 },
    /// A stage-input descriptor's buffer `layout` whose layout and attributes disagree, as for
    /// [`DescriptorError::VertexLayout`]: a set attribute names it and the descriptor lacks its
    /// layout, or its layout has a nonzero or dynamic stride and no set attribute names it.
    StageInputLayout { layout: u32 },
    /// A stage-input buffer layout that steps by thread position in the grid with a step rate of
    /// 0: no element is every 0th thread; or, but for Metal's unset layout, a constant step at a
    /// step rate other than 0, which Metal's pipeline creation stops at an assertion for, with and
    /// without API validation.
    StageInputStepRate { layout: u32, step_rate: u32 },
    /// A render pipeline descriptor whose `maxTessellationFactor` is not one Metal accepts for
    /// its partition mode: between 1 and 64, a power of two for `pow2`, and even for the
    /// fractional modes.
    MaxTessellationFactor {
        factor: u32,
        partition_mode: crate::metal::TessellationPartitionMode,
    },
    /// Two color outputs of the function that remain under its function-constant values write
    /// color attachment `color` at blend index `index` (`[[color(color), index(index)]]`): Metal
    /// refuses the specialization when it makes the function. MSL refuses such a pair outright, so
    /// only constants that leave both outputs of a gated pair present make one.
    DuplicateColorOutput { color: u32, index: u32 },
    /// Parameter `param` of the function remains under its function-constant values in an
    /// argument table (buffer, threadgroup memory, texture, or sampler) at slots `index .. index +
    /// count` that overlap those of an earlier parameter that remains: Metal refuses the
    /// specialization when it makes the function ("argument has invalid location
    /// (<index,count>)", measured on an M3 Max for every table). MSL refuses such a pair outright,
    /// so only constants that set an argument's slot (`[[texture(c)]]`) make one.
    OverlappingArgument { param: u32, index: u32, count: u32 },
    /// A `StageSource::Compiled` stage whose recursion's call stack was sized for
    /// `maxCallStackDepth` `compiled` (`RuntimeFact::CallStackDepth`) in a pipeline whose depth
    /// for that stage is `given` (0 taken as 1): a compute or tile pipeline's, or a render
    /// pipeline's vertex or fragment depth. The stack would hold another number of frames than the
    /// pipeline's Metal stage has. Compile the stage under the pipeline's depth.
    MaxCallStackDepth { compiled: u32, given: u32 },
    /// A `StageSource::Compiled` fragment function whose push fields start at `fragment` in a
    /// render pipeline whose vertex function's push fields end past it, at `vertex_end`: on
    /// `Device::from_queries`' profile a render pipeline's two stages share one push range, and a
    /// fragment function compiles under its vertex function's
    /// [`CompiledFunction::fragment_push_base`](crate::library::CompiledFunction::fragment_push_base)
    /// (`LoweringInputs::push_base`). Compile the fragment under that base.
    PushBase { fragment: u32, vertex_end: u32 },
    /// A `StageSource::Compiled` kernel compiled in another OpenCL mode than its compute
    /// pipeline's (`RuntimeFact::OpenClMode`; `compiled` says whether it was compiled in OpenCL
    /// mode): its arguments sit at other indices than the pipeline's Metal kernel takes them at.
    /// Compile the kernel under the pipeline's `open_cl_mode_enabled`.
    OpenClMode { compiled: bool },
    /// A tile function that declares threadgroup variables: Metal refuses its tile pipeline
    /// state ("Tile shaders do not support shader declared threadgroup variables", measured on an
    /// M4 and an M3 Max). Its threadgroup memory is its `[[threadgroup(n)]]` arguments alone.
    TileThreadgroupVariable,
    /// A library `newDynamicLibrary:error:` refuses as a dynamic library
    /// ([`DynamicLibrary::new`](crate::library::DynamicLibrary::new)).
    DynamicLibrary { problem: DynamicLibraryProblem },
    /// A stage-input buffer `layout` other than Metal's unset one (stride 0, per vertex, step
    /// rate 1) whose step function is a vertex stage's (per vertex, instance, patch or control
    /// point): Metal's compute pipeline creation stops at an assertion ("invalid stepFunction
    /// value"), with and without API validation, measured on an M4.
    StageInputStepFunction {
        layout: u32,
        step_function: crate::metal::StepFunction,
    },
    /// A `StageSource::Compiled` tile function compiled under `threadgroupSizeMatchesTileSize`
    /// `compiled` (`RuntimeFact::TileThreadgroup`) in a tile pipeline whose descriptor gives the
    /// flag the other value: the builtins it reads would answer the other pipeline's threadgroup.
    /// Compile the function under the pipeline's flag
    /// (`LoweringInputs::threadgroup_size_matches_tile_size`).
    TileThreadgroupSize { compiled: bool },
    /// A mesh render pipeline descriptor's thread, threadgroup or payload field that Metal
    /// refuses ([`MeshPipelineProblem`]).
    MeshPipeline { problem: MeshPipelineProblem },
    /// A compute pipeline descriptor's `maxTotalThreadsPerThreadgroup` that Metal refuses
    /// ([`ComputePipelineProblem`]).
    ComputePipeline { problem: ComputePipelineProblem },
}

/// A compute pipeline descriptor's `maxTotalThreadsPerThreadgroup` that Metal's creation fails
/// over, an error of code 3, measured on an M3 Max under macOS 27.0.1
/// (`compute_pipeline_limits.m`, plain and with `MTL_DEBUG_LAYER=1`, which agree). The limit is
/// Metal's 1024, not a lower one of this device's (the compiled pipeline's
/// [`ComputePipelineMetalInfo`](crate::pipeline::ComputePipelineMetalInfo) answers that one). A
/// required threadgroup size beside the maximum is the runtime's check.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComputePipelineProblem {
    /// The descriptor's maximum `max` above Metal's `limit`, for a kernel without a maximum of
    /// its own: "Specified total max threads per threadgroup (1025) exceeds the maximum total
    /// threads per threadgroup supported (1024)".
    MaxThreadsOverLimit { max: u32, limit: u32 },
    /// The descriptor's maximum `max` beside the kernel's own
    /// `[[max_total_threads_per_threadgroup(n)]]`, `specified`, another number of threads, whether
    /// or not `max` is above Metal's limit: "Kernel specified max total threads per threadgroup
    /// (64) must match compute pipeline specified max total threads per threadgroup (32)". A
    /// descriptor of the kernel's own maximum is made.
    MaxThreadsDiffer { specified: u32, max: u32 },
}

/// A field of a mesh render pipeline descriptor (`MTLMeshRenderPipelineDescriptor`) that Metal
/// refuses, measured on an M3 Max under macOS 27.0.1 (`mesh_pipeline_limits.m`, plain and with
/// `MTL_DEBUG_LAYER=1`, which agree). Metal's creation fails over some, an error, and its
/// descriptor validation ends the process at the rest, with or without API validation; each
/// variant says which. The thread limit is Metal's 1024, not a lower one of this device's.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MeshPipelineProblem {
    /// `stage`'s `maxTotalThreadsPerObjectThreadgroup` or `maxTotalThreadsPerMeshThreadgroup`
    /// `max` above Metal's `limit`: the creation fails, code 3, "Specified total max threads per
    /// threadgroup (2048) exceeds the maximum total threads per threadgroup supported (1024)",
    /// with or without an object function.
    MaxThreadsOverLimit { stage: Stage, max: u32, limit: u32 },
    /// `stage`'s descriptor maximum `max` beside other `specified` threads: those of its
    /// function's `[[max_total_threads_per_threadgroup(n)]]`, or of the stage's required size.
    /// The creation fails, code 3, "Kernel specified max total threads per threadgroup (64)
    /// must match compute pipeline specified max total threads per threadgroup (32)". Metal
    /// refuses a required size so on its first compile of the stage's function; it makes it
    /// once it compiled that function under the same maximum and a required size of the
    /// maximum's threads, which this refusal does not follow.
    MaxThreadsDiffer {
        stage: Stage,
        specified: u32,
        max: u32,
    },
    /// `stage`'s required threads per threadgroup with an axis of 0, though not every axis (all
    /// three 0 is no requirement): the process ends, "(4 * 0 * 1)(0) must not be 0.".
    RequiredEmptyAxis { stage: Stage, required: [u32; 3] },
    /// `stage`'s required threads per threadgroup, `threads` in all, above Metal's `limit`: the
    /// process ends, "must be <= 1024 (device limit)".
    RequiredOverLimit {
        stage: Stage,
        threads: u64,
        limit: u32,
    },
    /// `stage`'s required threads per threadgroup, `threads` in all, above its descriptor maximum
    /// `max`: the process ends, "must be <= maxTotalThreadsPerThreadgroup (32)".
    RequiredOverMax {
        stage: Stage,
        threads: u64,
        max: u32,
    },
    /// A required object threadgroup size `required` without an object function: the process
    /// ends, "If objectFunction is nil, requiredThreadsPerObjectThreadgroup must be [0, 0, 0]".
    RequiredWithoutObjectFunction { required: [u32; 3] },
    /// `payloadMemoryLength` `length` above Metal's `limit` of 16384 bytes: the creation fails,
    /// code 3, "Object shader payload size (16385) exceeds the maximum payload size allowed
    /// (16384)" with an object function, and code 2, "Payload size > 16384", without one.
    PayloadOverLimit { length: u32, limit: u32 },
}

/// Why a library is no dynamic library (`newDynamicLibrary:error:`), as Metal refuses it (probes,
/// an M3 Max and an M4).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DynamicLibraryProblem {
    /// The library's type is not dynamic: Metal's "Expected a library with type
    /// MTLLibraryTypeDynamic".
    NotDynamic,
    /// A dynamic library without an install name, which no dependency could name.
    NoInstallName,
    /// Its code reads a function constant: Metal's "cannot lower module with unresolved
    /// function constants" (a constant the code does not read is not in its AIR).
    FunctionConstants,
}

/// Why the view of a texture a heap array may hold cannot be made (`bind::heap`): the storage-read
/// check of `lower::texture::slot` skips such a member.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum BindError {
    /// The view needs something the device cannot do (a format, a view type).
    Unsupported(Unsupported),
    /// A view the derivation cannot state; always a bug, never the guest's.
    Internal(InternalError),
    /// A bound resource's descriptor is invalid (e.g. a sampler whose LOD clamps are NaN or
    /// inverted).
    InvalidDescriptor(DescriptorError),
}

impl fmt::Display for BindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for BindError {}

#[cfg(test)]
mod tests;
