//! Errors: every way compiling or binding can fail, as typed values to match on.
//!
//! Compiling returns [`Error`]: `Device::from_queries`, `Library::parse`, every `Device::compile_*`
//! call, `Device::render_pipeline_key`, and the [`metal`](crate::metal) mappings. Planning and
//! instantiating a binding returns [`BindError`]. Decoding cached bytes returns
//! [`ArtifactError`](crate::cache::ArtifactError), which is always a cache miss. Match variants,
//! not messages: `Display` is for logs.
//!
//! What each [`Error`] asks of the caller:
//!
//! - [`Error::Unsupported`]: a typed refusal ([`Unsupported`]) of valid Metal. Its [`Feature`]
//!   names the Metal construct, and its [`Needs`] says what would change the answer:
//!   [`Needs::AnyOf`] lists alternative sets of device [`Capability`]s (a device that enables
//!   every capability of one set compiles it), [`Needs::NotImplemented`] is a gap of this
//!   translator version on every device, [`Needs::NotRepresentable`] has no Vulkan 1.2+
//!   mechanism, and
//!   [`Needs::StageBound`] is a stage the device's plan writes heavier than the translator hands
//!   any driver of a device [`Device::from_queries`](crate::device::Device::from_queries) made.
//!   The runtime adds
//!   [`Needs::MetalGpuFamily`]: Metal's own refusal
//!   on a device without the family, and [`Needs::MetalRefuses`]: Metal's own refusal of the
//!   form on the devices it was measured on. The translator does not return different output in place
//!   of a refusal; where an output accepts a documented difference from Metal, its stage lists it
//!   as a [`PrecisionNote`](crate::pipeline::PrecisionNote).
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
//! **Caching failures.** [`Error::retry`] says whether the same request can succeed later. A
//! [`BindError`] from a pipeline's `plan` depends only on the pipeline, the
//! device, and the [`BindShape`](crate::bind::BindShape), so it memoizes beside the plan on
//! `(PipelineKey, BindShape)`. One from `instantiate_dispatch` or `instantiate_draw` can also
//! depend on that dispatch or draw and on the bind state's per-draw values (offsets, lengths,
//! addresses, the stage-in region, the tessellation factor buffer), so it does not memoize on the
//! shape. [`BindError::NeedsVariant`] names the pipeline variant to plan or instantiate on
//! instead.
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
            Air::ProgramScopeBinding => Feature::ProgramScopeBinding,
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
    /// holds, or, for a stage past the size the
    /// translator hands a driver ([`Feature::StageSize`]), the stage's entry weight and the bound
    /// it passed (`entry weight 532213, past the bound 99109`). Opaque like every [`DiagName`].
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

    /// Refused as Metal refuses it on a device that does not claim the Metal GPU `family`
    /// (`MTLGPUFamily`'s raw value), as this device does not ([`Needs::MetalGpuFamily`]).
    pub fn metal_gpu_family(feature: Feature, family: i64) -> Unsupported {
        Unsupported {
            feature,
            needs: Needs::MetalGpuFamily(family),
            location: None,
            symbol: None,
        }
    }

    /// Run on a device with `capability`, which this device does not have and which the compile
    /// layer's vocabulary does not name ([`Needs::DeviceCapability`]).
    pub fn device_capability(feature: Feature, capability: &'static str) -> Unsupported {
        Unsupported {
            feature,
            needs: Needs::DeviceCapability(capability),
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
    /// A device capability the compile layer's vocabulary does not name, which this device does
    /// not have: named by its Vulkan member name (`"sparseResidencyImage2D"`) or, where it is not
    /// one member, in words. **The device's own gap**, not unbuilt work of ours: a device with the
    /// capability runs the input.
    ///
    /// Why a variant of its own rather than [`Needs::AnyOf`]. `AnyOf` names its capabilities as
    /// [`Capability`], and `Capability::Feature`'s vocabulary — [`VkFeature::ALL`], generated from
    /// the exposure table — *is* the compile profile's feature set, so anything named there enters
    /// every profile, cache key and digest. A capability only the **runtime** needs is deliberately
    /// outside it: a sparse heap's `sparseBinding` and `sparseResidency*`, its queue family's
    /// sparse bit, and the sparse image formats a device lists, as `timelineSemaphore` would be
    /// (`sparse::device`). Before this variant there was no way for such a refusal to say "the
    /// device lacks it": the remaining words were [`Needs::NotImplemented`], which claims we have
    /// not built the thing, and [`Needs::NotRepresentable`], which claims no Vulkan mechanism
    /// exists for it on any device. Both are false of a device that simply does not have a Vulkan
    /// 1.0 feature, and both read to a consumer as our unbuilt work.
    ///
    /// No compile profile, cache key or digest holds it; only the runtime refuses so.
    DeviceCapability(&'static str),
    /// Not implemented by this translator version on any device.
    NotImplemented,
    /// No Vulkan 1.2+ mechanism exists for these semantics.
    NotRepresentable,
    /// The stage, as this device's plan writes it, is heavier than the translator hands any
    /// driver: its entry weight passes the heaviest stage measured to compile within host-sized memory
    /// ([`Unsupported::symbol`] names the weight and the bound). The weight is the plan's, so a
    /// device whose plan writes the stage lighter translates it: one that keeps its widest
    /// simdgroup mode pinned where this one emulates simdgroups. A gap of the translator rather
    /// than of the device: no device is handed a stage past the bound.
    StageBound,
    /// A device that claims this Metal GPU family (`MTLGPUFamily`'s raw value, 1010 for Apple10),
    /// which the device does not: Metal itself refuses the input on a device without the family,
    /// so the refusal is Metal's own answer rather than a gap. Only the runtime refuses so, by the
    /// families its device claims (`supportsFamily:`); no compile profile, cache key or digest
    /// holds it.
    MetalGpuFamily(i64),
    /// Metal itself refuses this form on every device named here, each as its GPU and macOS
    /// version (`"Apple M4, macOS 27.0.1"`; the devices it was measured on), so the refusal is
    /// Metal's own answer rather than a gap. No GPU family is known to take the form. Only the
    /// runtime refuses so, on a device whose Metal answer the measured ones stand for (one that
    /// does not claim a family past theirs); elsewhere the form is [`Needs::NotImplemented`]. No
    /// compile profile, cache key or digest holds it.
    MetalRefuses(&'static [&'static str]),
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
    /// GPUFamily10 and later"), so the runtime answers it as Metal does on a device that does not
    /// claim Apple10 ([`Needs::MetalGpuFamily`]). Its fragment layout, which only such a GPU can
    /// show, is not measured.
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
    /// A stage module heavier than the translator hands the device's driver: past the heaviest
    /// measured to compile within host-sized memory, on a device
    /// [`Device::from_queries`](crate::device::Device::from_queries) made ([`Needs::StageBound`]).
    StageSize,
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
    /// Framebuffer fetch without an ordered form on the device (rasterization-order attachment
    /// access, or the pixel interlock's storage form where the pipeline's state lets it be
    /// exact), or in a shape not implemented.
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
    /// A render pass's custom sample positions (`setSamplePositions:count:`) on a device that
    /// takes no custom sample locations at the pass's sample count, which is the count the grid
    /// is programmed at (`VkProperty::SampleLocationSampleCount`, named by the refusal). Only the
    /// runtime refuses so; the device's own lack, not the caller's error and not a gap.
    SamplePositions,
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
    /// The device's `maxPushConstantsSize` is below the 256 bytes of the push window every stage
    /// is lowered for: a compute stage's dispatch region and buffer addresses, a render
    /// pipeline's two stages together.
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
    /// A texture whose extent, array layers, or level count is past what the device takes for its
    /// format, type, and usage (`Capability::Limit` names the device limit where one applies).
    TextureSize,
    /// A texture sample count the device does not take for its format, type, and usage.
    TextureSampleCount,
    /// A texture view Vulkan cannot make exactly on this image: a format reinterpretation outside
    /// Vulkan's compatibility classes, or a view type the image cannot take.
    TextureView,
    /// A texture swizzle whose Metal behavior is not measured.
    TextureSwizzle,
    /// `generateMipmapsForTexture:` that cannot run on the device: a format neither the device's
    /// blits filter nor the compute filter computes, or limits the filter's chain does not fit.
    MipmapGeneration,
    /// A buffer over the caller's host memory without a copy (`newBufferWithBytesNoCopy:`): it
    /// needs `VK_EXT_external_memory_host`, a memory type the pointer and the buffer both allow,
    /// and a range aligned to the device's import alignment; a driver may also refuse the memory.
    HostMemory,
    /// A command buffer's GPU times (`GPUStartTime`, `GPUEndTime`) on a device whose queue
    /// family writes no timestamps (`timestampValidBits` 0).
    GpuTimes,
    /// A `Counting` visibility result (`MTLVisibilityResultModeCounting`) on a device without
    /// `occlusionQueryPrecise`: the samples that passed are a precise occlusion query's count.
    CountingVisibility,
    /// A texture view whose format has other components, or components of other widths, than
    /// its texture's (`RGBA8Unorm` as `R32Uint`), on a device whose `VK_KHR_portability_subset`
    /// withholds `imageViewFormatReinterpretation`, or that the runtime adopted without being
    /// told it enabled it (the runtime's `RawDevice::view_format_reinterpretation`), where the
    /// runtime's shadow of the view's texels cannot serve it: a view of a sparse texture or a
    /// heap's texture, formats whose copies would not keep every bit, or a command that writes
    /// the texture both through such a view and through another alias of it.
    ViewFormatReinterpretation,
    /// A render pass whose depth and stencil attachments are separate multisampled textures that
    /// it loads or may store, on a device without maintenance8: Vulkan copies a multisampled depth or stencil aspect between images of different formats only through a
    /// colour image of maintenance8's depth-stencil/colour table.
    SeparateMultisampledDepthStencil,
    /// A stage whose private memory (its `thread` variables and arrays, and the constants it
    /// holds per thread) takes more Function storage a thread than the device holds correctly
    /// (`VkLimit::FunctionStorageBytesPerThread`): its measured value, or on a device not measured
    /// the floor of 16 KiB, and for a vertex or fragment function there the 192 KiB Metal was
    /// measured to hold. A stage with a call stack is refused as `Recursion` instead.
    PrivateMemory,
    /// A sparse heap (`MTLHeapTypeSparse`), or a texture made from one, on a device without what
    /// a sparse resident image of it needs, which `missing` names.
    SparseTextures {
        missing: SparseNeed,
    },
    /// Command buffers and events, which the runtime completes and signals through timeline
    /// semaphores (core in Vulkan 1.2), on a device whose `maxTimelineSemaphoreValueDifference`
    /// (`max_value_difference`) is below the `2^31 - 1` Vulkan requires of every device: the
    /// device is not conformant, and the runtime refuses it at creation, before its first signal.
    TimelineSemaphores {
        max_value_difference: u64,
    },
    /// Reflection of an argument whose answer Metal's has not been measured for (`reflection`):
    /// a type the MSL type grammar does not cover, a binding kind no probe has asked Metal about
    /// (explicit imageblocks and imageblock data, argument-buffer function tables, acceleration
    /// structures and render pipeline states), or, in a library's answer, an index a function
    /// constant computes. The symbol names the argument's type.
    Reflection,
    /// A texture of a heap whose memory the device's optimal images do not take (a `Shared` heap
    /// on a device whose optimal images take no host-visible memory, as the RTX 3090's), which
    /// the runtime makes a linear image in the heap's memory, on a device whose linear images
    /// lack what the texture needs: `missing` names it.
    HeapTextureTiling {
        missing: LinearNeed,
    },
    /// A shader write to a texture made without `ShaderWrite` (or `ShaderAtomic`), or a read of
    /// one made without `ShaderRead`, which plain Metal runs, where the texture's image lacks the
    /// storage or sampled usage: the runtime adds it only where it costs the image nothing (the
    /// same memory size and alignment), so that no valid program's texture loses its
    /// compression. The exact route is a shadow image with the usage, copied around the commands
    /// that use the texture so.
    UndeclaredTextureUsage,
    /// A render pass whose render area or layers are past the device's framebuffer limits
    /// (`Capability::Limit` names `MaxFramebufferWidth`, `MaxFramebufferHeight` or
    /// `MaxFramebufferLayers`). Metal renders passes up to 16384 pixels a side and 2048 layers.
    RenderTargetSize,
    /// A render state a function sets on an indirect render command
    /// (`render_command::set_cull_mode`, `set_front_facing_winding`, `set_triangle_fill_mode`,
    /// `set_depth_bias`, `set_depth_clip_mode`, `set_depth_stencil_state`), or a
    /// `depth_stencil_state` it reads from an argument buffer: the record holds no render state.
    /// Metal's Apple10 family alone lists the feature, and Metal's compiler refuses such a
    /// function on an Apple9 GPU, so the runtime answers it as that compiler does on a device that
    /// does not claim Apple10 ([`Needs::MetalGpuFamily`]).
    IndirectRenderState,
    /// A fragment function's read of a Metal 3.1 program-scope built-in variable that Metal does
    /// not lower there: a global `thread_index_in_quadgroup` (`air.get_thread_index_in_quadgroup`),
    /// which Apple's compiler takes in a fragment function but whose render pipeline the M4 and
    /// the M3 Max refuse, while they run the same built-in as the fragment's parameter. The
    /// translation refuses it `NotImplemented`; the runtime answers it as Metal does
    /// ([`Needs::MetalRefuses`]).
    GlobalBuiltinInFragment,
    /// A private fixed-function property of Metal's pipeline descriptor (Apple SPI, OpenGL on
    /// Metal's) set to a value whose effect the translation has not built: the property it names
    /// ([`PrivateRenderField`]). Metal's answer for each was measured; building it is open work.
    PrivateRenderState(PrivateRenderField),
    /// A blend the function computes in the shader, into an extended-range attachment
    /// (`BGRA10_XR`, `BGR10_XR`, `RGB10A8_2P_XR10` and their `_sRGB` twins) or into a 16-bit
    /// float attachment in the interlocked storage form, of the part it names
    /// ([`ExtendedRangeBlend`]).
    ExtendedRangeBlend(ExtendedRangeBlend),
    /// A program-scope buffer binding of MSL 3.2 (macOS 15 and later), `constant T* constant p
    /// [[buffer(n)]];` in Apple's `!air.global_bindings` list, valid AIR this version does not
    /// implement yet.
    ProgramScopeBinding,
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
    /// The coverage after the depth and stencil tests (`[[sample_mask, post_depth_coverage]]`),
    /// read through `VK_EXT_post_depth_coverage`.
    PostDepthCoverage,
    /// Lines at a width other than 1 (Apple's `setLineWidth:`), drawn by the device's own wide
    /// lines (`wideLines`) or as bands of triangles through synthesized tessellation stages
    /// (`tessellationShader`, `pipeline::link::band`).
    WideLines,
}

/// The part of a blend the shader computes that [`Feature::ExtendedRangeBlend`] refuses. An
/// extended-range attachment's blend, which the shader computes by the device's arithmetic, is
/// built for every factor, operation and write mask, and refuses only [`Output`](Self::Output);
/// a 16-bit float attachment's blend in the interlocked storage form refuses the other two, the
/// generic blend taking their place where the function reads the attachment.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExtendedRangeBlend {
    /// Into a 16-bit float attachment in the storage form, a factor of the second source
    /// (`Source1Color`, `Source1Alpha` and their one-minus forms): Metal's blend is not built.
    /// (A function that writes no second output to the attachment Metal refuses:
    /// [`DescriptorError::ColorAttachment`].)
    DualSource,
    /// Into a 16-bit float attachment in the storage form under fast math, a write mask of exactly
    /// one color channel: Metal blends that channel otherwise than the two or three it blends as
    /// one vector, by a rule not known. (Without fast math it is built.)
    OneColorChannel,
    /// An output other than four `half` or `float` lanes, or a second output other than those the
    /// blend reads: not built.
    Output,
}

/// A private property of Metal's render pipeline descriptor that
/// [`Feature::PrivateRenderState`] refuses at a value other than Metal's default
/// (`PrivateRenderState`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrivateRenderField {
    /// `sampleMask` dropping a sample, in a pipeline whose fragment function writes an
    /// attachment itself (a storage attachment, a shadow stencil or an imageblock), whose writes
    /// were not measured under the mask.
    SampleMask,
    /// `sampleCoverage` other than 1 at a sample count past 4, where Metal was not measured, or
    /// in a pipeline whose fragment function writes an attachment itself (a storage attachment,
    /// a shadow stencil or an imageblock), whose writes were not measured under it.
    SampleCoverage,
    /// `alphaTestEnabled` with a test the translation has not built: an `alphaTestFunction` past
    /// `MTLCompareFunctionAlways`, a pipeline without a fragment function, a fragment function
    /// whose color 0 is not four `float` or `half` lanes (Metal's compiler fails a function that
    /// writes no color 0 under the test), or one that loops over its covered samples.
    AlphaTest,
    /// `logicOperationEnabled` with an operation other than copy and no-op: on a device without
    /// `logicOp` (`Unsupported::needs` naming it), or where Vulkan's operation is not Metal's (an
    /// sRGB attachment, a float attachment that blends, a format whose operation Metal was not
    /// measured with, a fragment function that writes an attachment itself).
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

/// What a device's linear images lack for a texture of a heap whose memory its optimal images do
/// not take (`Feature::HeapTextureTiling`). The limits are those
/// `vkGetPhysicalDeviceImageFormatProperties2` reports for a linear image of the texture's
/// format, type, usage and flags.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LinearNeed {
    /// A memory type of the heap's storage that the device's linear images take: neither its
    /// optimal nor its linear images take one.
    MemoryType,
    /// A linear image of the texture's format, type, usage and flags at all: the device reports
    /// none (the RTX 3090 makes no linear cube, 3D or depth image).
    Image,
    /// `features` of `format`'s linear tiling, which its optimal tiling has and the texture's
    /// usage relies on (`linearTilingFeatures`).
    Features {
        format: vk::Format,
        features: vk::FormatFeatureFlags2,
    },
    /// An extent of `required` texels along an axis where the device's linear images hold
    /// `available`.
    Extent { required: u32, available: u32 },
    /// `required` mip levels where the device's linear images hold `available`.
    MipLevels { required: u32, available: u32 },
    /// `required` array layers (slices, or six faces a cube) where the device's linear images
    /// hold `available`.
    ArrayLayers { required: u32, available: u32 },
    /// The texture's sample count.
    SampleCount(u32),
}

/// What a device lacks for a sparse resident image (`Feature::SparseTextures`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SparseNeed {
    /// `sparseBinding`.
    SparseBinding,
    /// A queue family with `VK_QUEUE_SPARSE_BINDING_BIT` where the runtime submits.
    SparseBindingQueue,
    /// `sparseResidencyImage2D`: a 2D, 2D array, cube, cube array or 2D multisample texture.
    SparseResidencyImage2D,
    /// `sparseResidencyImage3D`: a 3D texture.
    SparseResidencyImage3D,
    /// `sparseResidency2Samples` to `sparseResidency16Samples`, for this sample count.
    SparseResidencySamples(u32),
    /// Sparse residency of the texture's format, type, sample count and usage:
    /// `vkGetPhysicalDeviceSparseImageFormatProperties` lists none for its aspects.
    SparseImageFormat,
    /// A sparse image whose memory needs a metadata aspect bound
    /// (`VK_SPARSE_MEMORY_BIND_METADATA_BIT`), which the runtime does not bind.
    SparseMetadata,
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
    /// A fragment function reads a color attachment (framebuffer fetch) on a device that orders
    /// the read only through the pixel interlock, where the read's form depends on the render
    /// pipeline's attachments and state (`LoweringInputs::framebuffer_fetch`): compiled without
    /// that state, or for a pipeline whose state chose another form, it is compiled for the
    /// pipeline again.
    FramebufferFetch,
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
    /// A fragment function in a render pipeline whose private logic operation it computes itself
    /// on a device without Vulkan's `logicOp` (`LoweringInputs::logic_operation`), as Metal
    /// operates on its attachments' codes; or it computes another operation, or one its pipeline
    /// does not take. It was compiled for another pipeline, and is compiled for this one again.
    LogicOperation,
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

/// A binder planning or instantiation error. A pipeline's `plan` sees only the pipeline, the
/// device, and the `BindShape`, so a planning error is permanent for its `(PipelineKey,
/// BindShape)` pair. An instantiation error can also depend on the dispatch or draw and on the
/// bind state's per-draw values (`MissingStageInRegion`, `UniformViewAliased`, the patch-draw
/// errors, a descriptor range or texel buffer beyond its limit, ...), so it is not permanent for
/// the pair. `NeedsVariant` names the pipeline a dispatch or draw runs on instead.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindError {
    /// A slot index beyond the Metal maximum for its table.
    SlotOutOfRange { slot: SlotRef },
    /// The bind shape needs something the device cannot do (local size limits, formats,
    /// sampler reduction, GPU-encoded ICBs, …).
    Unsupported(Unsupported),
    /// A device-address slot received a buffer without a device address.
    MissingDeviceAddress { slot: SlotRef },
    /// A descriptor-form slot's window exceeds the descriptor range limit (Profile F).
    DescriptorRangeExceeded {
        slot: SlotRef,
        needed: u64,
        limit: u64,
    },
    /// A texture buffer bound through one texel buffer view exceeds `maxTexelBufferElements`:
    /// one wider than that view binds through views of its parts (`bind::texel_parts`) except at
    /// a slot whose function reads it through a sampler, on a device that indexes no array of
    /// the slot's kind of texel buffer by a `NonUniform` value, or in a format whose block holds
    /// several texels.
    TexelBufferTooLarge {
        slot: SlotRef,
        elements: u64,
        limit: u64,
    },
    /// The pipeline and the binder disagree (a pipeline built by another device, or a binder
    /// bug); always a bug, never the guest's.
    Internal(InternalError),
    /// A bound resource's descriptor is invalid (e.g. a sampler whose LOD clamps are NaN or
    /// inverted).
    InvalidDescriptor(DescriptorError),
    /// A render pipeline fetches a fixed-function vertex buffer that nothing is bound to.
    UnboundVertexBuffer { slot: SlotRef },
    /// The set-1 heap's elements (one per resident texture a handle can name, plus the null
    /// element) exceed the pipeline's heap capacity: the heap arrays' count, at most the handle
    /// table's reach.
    HeapTooLarge { count: u64, limit: u64 },
    /// A pipeline reads the Metal address of a buffer slot (`SlotGuestBase`: a pointer compare,
    /// `ptrtoint`, or stored pointer) that is bound with `set*Bytes`, which gives the binder no
    /// Metal address to write.
    MissingGuestAddress { slot: SlotRef },
    /// The encoder's vertex amplification (`setVertexAmplificationCount:viewMappings:`) is not
    /// one Metal draws this pipeline with: a count of 0, one above 1 for a pipeline whose
    /// `maxVertexAmplificationCount` (`max`) is 1, or view mappings given for fewer
    /// amplifications than the count.
    InvalidVertexAmplification {
        count: u32,
        max: u32,
        view_mappings: u32,
    },
    /// The dispatch runs on another variant of the pipeline than the one planned: a threadgroup
    /// with a partial simdgroup, which the kernel's flattened simd tier cannot run, runs on its
    /// emulated-simdgroups variant (`VariantKey::emulated_simdgroups`). The consumer compiles the
    /// pipeline under `variant` (the plan's key and what the dispatch adds, also
    /// `BindingPlan::dispatch_variant`), plans it for the same shape, whose plan states `variant`,
    /// and instantiates the dispatch there. Nothing is dispatched on the plan asked. A draw answers
    /// it when the pipeline planned is not the variant its vertex function's render target slice
    /// asks for at the shape's render target array length (`VariantKey::dropped_vertex_layer`);
    /// `variant` is then the plan's own. Boxed: a key holds a set per kind of variant, and the
    /// error stays small.
    NeedsVariant {
        variant: Box<crate::cache::VariantKey>,
    },
    /// The compute encoder's imageblock size (`setImageblockWidth:height:`) is not one a kernel
    /// with an imageblock dispatches with: a width or height of 0 (the encoder never set it) or
    /// above 65535, which `get_imageblock_width` and `get_imageblock_height` cannot answer.
    InvalidImageblockSize { width: u32, height: u32 },
    /// The kernel reads the compute encoder's stage-in region (a `[[stage_in]]` attribute
    /// stepping by thread position, or `[[grid_origin]]` or `[[grid_size]]`) and the encoder
    /// never set one (`setStageInRegion:`): what Metal reads then is not established.
    MissingStageInRegion,
    /// A `device` buffer slot the pipeline reads through a uniform view (Profile F's per-stage
    /// storage-buffer fallback, `RuntimeFact::UniformViewAliasing`) is bound where a buffer slot
    /// the pipeline writes (`written`) may write: their Metal windows overlap. A uniform view need
    /// not observe the dispatch's or draw's own writes, so nothing is bound. Depends on the
    /// per-draw Metal addresses, not on the shape.
    UniformViewAliased { slot: SlotRef, written: SlotRef },
    /// A draw Metal does not make with this pipeline: patches (`patches`) with a pipeline whose
    /// vertex function is not post-tessellation, or primitives with one whose vertex function is.
    InvalidDrawKind { patches: bool },
    /// A patch draw's `numberOfPatchControlPoints` (`given`) differs from the `N` the
    /// post-tessellation vertex function's `[[patch(_, N)]]` states (`expected`).
    PatchControlPoints { given: u32, expected: u32 },
    /// The tessellation factors a draw reads span more bytes (`needed`) than one storage-buffer
    /// descriptor can bind (`limit`, `maxStorageBufferRange`), or an instance stride does.
    TessellationFactorRange { needed: u64, limit: u64 },
    /// The encoder's tessellation factor scale (`bits`, the `float`'s) rounded to `half` is not a
    /// positive `half` (it is negative, zero, or NaN). Metal's validation stops
    /// `setTessellationFactorScale:` for a scale that is not a positive normal `float`, and
    /// culls every patch for one that is 0 as a `half` (2^-30), which
    /// a consumer draws as nothing.
    InvalidTessellationFactorScale { bits: u32 },
    /// A tile function's render area (`BindState::render_area`) has a width or height of 0: the
    /// encoder's render pass was never set, so the dispatch has no tile to run.
    InvalidRenderArea { width: u32, height: u32 },
    /// The binder sets up no device replay of GPU-encoded compute commands for this execution,
    /// variant set, or plan: `reason` says why. Nothing is recorded.
    InvalidReplay {
        reason: crate::bind::replay::ReplayRefusal,
    },
    /// A patch draw's patch index buffer is bound at an offset that is not a multiple of 4, which
    /// Metal requires (a Metal probe on an M4: "patchIndexBufferOffset(2) must be a multiple of 4
    /// bytes").
    InvalidPatchIndexBuffer { offset: u64 },
    /// The patch indices a draw reads span more bytes (`needed`) than one storage-buffer
    /// descriptor can bind (`limit`, `maxStorageBufferRange`).
    PatchIndexRange { needed: u64, limit: u64 },
    /// A patch draw with a control-point index buffer (`indexed`, `drawIndexedPatches`) for a
    /// pipeline whose `tessellationControlPointIndexType` is `None`, or one without it for a
    /// pipeline whose type is not, which Metal's validation stops (a Metal probe on an M4:
    /// "tessellationControlPointIndexType of MTLTessellationControlPointIndexTypeNone not valid for
    /// drawIndexedPatches", and of `UInt32` for `drawPatches`).
    ControlPointIndexing { indexed: bool },
    /// A patch draw's control-point index buffer is bound at an offset that is not a multiple of
    /// its pipeline's index size, which Metal requires (a Metal probe on an M4:
    /// "controlPointIndexBufferOffset(1) must be a multiple of 2 bytes").
    InvalidControlPointIndexBuffer { offset: u64 },
    /// The control-point indices a draw reads span more bytes (`needed`) than one storage-buffer
    /// descriptor can bind (`limit`, `maxStorageBufferRange`).
    ControlPointIndexRange { needed: u64, limit: u64 },
    /// A tile function reads its threadgroup argument `index`, which the encoder never set
    /// (`BindState::tile_threadgroup`, `setThreadgroupMemoryLength:offset:atIndex:`): Metal's
    /// validation stops the dispatch ("missing Threadgroup Memory binding").
    MissingThreadgroupMemory { index: u32 },
    /// A draw Metal does not make with this pipeline: a mesh draw (`mesh`) with a pipeline that is
    /// not a mesh render pipeline, or a draw of primitives or patches with one (Metal ends the
    /// process on `drawPrimitives` with a mesh pipeline state).
    InvalidMeshDraw { mesh: bool },
    /// A draw of a pipeline whose vertex function writes the primitive ID
    /// (`LoweringInputs::primitive_id_varying`) whose bind state names no primitive type
    /// (`BindState::primitive_type`): its vertices per primitive, and whether an indexed draw
    /// runs de-indexed (a strip's does not), depend on it.
    MissingPrimitiveType,
}

impl fmt::Display for BindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for BindError {}

#[cfg(test)]
mod tests;
