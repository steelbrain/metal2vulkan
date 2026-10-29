//! Errors: every way compiling or binding can fail, as typed values to match on.
//!
//! Compiling returns [`Error`]: `Device::new`, `Library::parse`, every `Device::compile_*` call,
//! `Device::render_pipeline_key`, and the [`metal`](crate::metal) mappings. Planning and
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
//!   mechanism, [`Needs::DriverDefect`] is a mechanism the device's driver is measured to run
//!   wrong ([`Needs::RuntimeDriverDefect`] one the runtime records for itself). The runtime adds
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
//! **Caching failures.** [`Error::retry`] says whether the same request can succeed later, and
//! [`Error::is_permanent_for`] which key may memoize the failure: [`Permanence::Input`] (the
//! input, on every device), [`Permanence::InputAndDevice`] (the input on this device profile;
//! every compile key includes the profile's fingerprint), or [`Permanence::Never`] (transient:
//! never cache it). A [`BindError`] from a pipeline's `plan` depends only on the pipeline, the
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
use crate::air::error::{MissingInput, RequestError};
use crate::device::requirements::{
    ApiVersion, FormatUsage, VkExtension, VkFeature, VkLimit, VkProperty,
};
use crate::device::DriverWorkaround;
use crate::metal::PixelFormat;
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

/// What the outcome is permanent for, i.e. which cache key may memoize it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Permanence {
    /// Permanent for the input (bytes, entry, constants, descriptor) on every device.
    Input,
    /// Permanent for the input on this device profile.
    InputAndDevice,
    /// Not permanent; never memoize.
    Never,
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

    /// What this outcome is permanent for. Transient outcomes (those [`Error::retry`] calls
    /// `Later`) are `Never` and may not enter a memo, artifact cache, or validation observation.
    pub fn is_permanent_for(&self) -> Permanence {
        match self {
            Error::Cancelled | Error::Budget(_) => Permanence::Never,
            Error::Internal(internal) if internal.kind == InternalKind::AllocationFailed => {
                Permanence::Never
            }
            Error::Unsupported(u) => match u.needs {
                Needs::AnyOf(_) => Permanence::InputAndDevice,
                Needs::NotImplemented | Needs::NotRepresentable => Permanence::Input,
                Needs::DriverDefect(_)
                | Needs::RuntimeDriverDefect(_)
                | Needs::DeviceCapability(_)
                | Needs::MetalGpuFamily(_)
                | Needs::MetalRefuses(_) => Permanence::InputAndDevice,
            },
            Error::NeedsInput(_)
            | Error::InvalidInput(_)
            | Error::InvalidDescriptor(_)
            | Error::LimitExceeded(_) => Permanence::Input,
            // The emitted module depends on the profile, so an invariant failure is keyed on both.
            Error::Internal(_) => Permanence::InputAndDevice,
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
            Air::Linking => Feature::Linking,
            Air::Constant { code } => Feature::Constant { code },
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
        }
    }
}

impl From<MissingInput> for RuntimeInput {
    fn from(i: MissingInput) -> RuntimeInput {
        match i {
            MissingInput::FunctionConstant { index } => RuntimeInput::FunctionConstant { index },
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
    /// callee of a call that needs linking, or the pixel format a refused texture, view or copy
    /// holds ([`naming_format`](Self::naming_format)). Opaque like every [`DiagName`].
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

    /// Translatable on a device whose driver does not have `defect`, which this device's does.
    pub fn driver_defect(feature: Feature, defect: DriverWorkaround) -> Unsupported {
        Unsupported {
            feature,
            needs: Needs::DriverDefect(defect),
            location: None,
            symbol: None,
        }
    }

    /// Run by the runtime on a device whose driver does not have `defect`, which this device's
    /// does.
    pub fn runtime_driver_defect(feature: Feature, defect: RuntimeDriverDefect) -> Unsupported {
        Unsupported {
            feature,
            needs: Needs::RuntimeDriverDefect(defect),
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

    /// Name the pixel format a refused texture, view, copy or attachment holds, by its Metal
    /// enumerator name (`MTLPixelFormatRGB10A8_2P_XR10`), as [`symbol`](Self::symbol)
    /// (diagnostics only).
    pub fn naming_format(self, pixel_format: PixelFormat) -> Unsupported {
        self.naming(DiagName::new(pixel_format.name()))
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
    /// The mechanism exists, but the device's driver is measured to run it wrong: the profile
    /// carries the named [`DriverWorkaround`], derived from the driver's identity, and the
    /// translation refuses rather than hand back output the driver would change. A device
    /// without the defect translates the input.
    DriverDefect(DriverWorkaround),
    /// The mechanism exists, but the device's driver runs wrong a command the runtime would
    /// record for itself: the runtime refuses rather than hand back what the driver would
    /// change. A device without the defect runs the input.
    RuntimeDriverDefect(RuntimeDriverDefect),
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

/// A driver's measured answer where the specifications leave it to the device, per driver and
/// GPU (`device::driver_facts`): the profile takes it on every release of a driver measured with
/// it. Where the profile takes it a module relies on it (and requires its property) or records no
/// note; elsewhere each variant names the route taken: the device's own answer, recorded as a
/// precision note, or, where the answer decides a capability, its emulation or a refusal.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DriverDeviation {
    /// How the driver hands a kernel's cube sample's explicit gradients to the GPU
    /// (`ProfileFields::cube_gradients`'s compute route, `CubeGradientRoute::Metal`): as given, to
    /// Metal's own reading. Unmeasured, the route is Vulkan's rule (`CubeGradientRoute::Vulkan`):
    /// the function hands Metal's footprint in the form that rule reads, and the device's reading
    /// of it is recorded as `PrecisionNote::CubeGradientFootprint`. A vertex or fragment
    /// function's route is [`DriverDeviation::GraphicsCubeGradientTangents`].
    CubeGradientRoute,
    /// The device runs each 32 consecutive invocations of a compute workgroup in lockstep, as
    /// an Apple GPU runs a Metal simdgroup (`VkProperty::SimdgroupLockstep`). Unmeasured, a kernel
    /// that needs that order takes emulated simdgroups, exact.
    SimdgroupLockstep,
    /// The device computes each `GLSL.std.450` function and `OpFDiv` of `float` as Metal's own,
    /// in the math mode Metal compiles the module with (`FloatFunctions::Metal`). Unmeasured, the
    /// module takes the device's instructions all the same, each kind recorded with its bound
    /// (`PrecisionNote::NativeTranscendentals`, and `NativeDivision`, `NativeDot` and `NativeMix`
    /// for a division, a `dot` and a `mix`).
    MetalFloatFunctions,
    /// The device's `float` and `half` arithmetic answers Metal's canonical NaN
    /// (`NanEncoding::MetalCanonical`). The module leaves every NaN's bits to the device either
    /// way; unmeasured, a NaN Metal answers records `PrecisionNote::NativeNanEncoding`.
    MetalCanonicalNans,
    /// The device samples, filters and converts texels through Metal's own sampler
    /// (`VkProperty::MetalSampler`), an 8-bit sRGB texel's decode included. Unmeasured, every
    /// sample, gather, comparison and level query is the device's own sampler's all the same
    /// (`VkProperty::DeviceSampling`), recorded as `PrecisionNote::DeviceSampling`, and every
    /// read, sample, gather and input-attachment read of an sRGB texel the device's own decode,
    /// recorded as `PrecisionNote::SrgbTexelDecode`. A `half` result is the device's `float`
    /// texel rounded to nearest even either way; where the profile takes it the module records
    /// `PrecisionNote::HalfTexelConversion` only for a class whose Metal `half` is not that
    /// `float` rounded so.
    MetalSampler,
    /// How the driver hands a vertex or fragment function's cube sample's explicit gradients to
    /// the GPU under macOS 27 (the graphics route, `CubeGradientRoute::AppleProjected`): through
    /// SPIRV-Cross's `spvGradientCube`, whose result Metal reads by its macOS 27 rule, the face's
    /// two tangents: an `x` or `z` face's gradient reaches it as Metal's own, a `y` face's `z`
    /// tangent is lost. Unmeasured, as [`DriverDeviation::CubeGradientRoute`].
    GraphicsCubeGradientTangents,
    /// The device stores a fragment function's float color output into a normalized, sRGB or
    /// half-float attachment as Metal does (`VkProperty::MetalColorWrites`): the exact product
    /// rounded to nearest, Metal's twelve-bit sRGB encode, and `half` rounded toward zero. The
    /// fragment function writes each such output as computed either way, but a half-float channel
    /// under `TextureWriteRoundingMode::NearestEven`, which it rounds on every profile; unmeasured,
    /// a channel the device converts records `PrecisionNote::DeviceColorConversion`
    /// (`LoweringInputs::color_writes`).
    MetalColorWrites,
    /// The device stores the NaN `0x7fc00000` a fragment function writes for a `float` output to
    /// a half-float color channel as `0x7e00`, the NaN Metal stores for any `float` NaN
    /// (`VkProperty::MetalFloatColorNans`): the NaN of a lane the function rounds to nearest even
    /// under `TextureWriteRoundingMode::NearestEven`. Where the profile takes it, such a lane
    /// records no note. Unmeasured, it records `PrecisionNote::DeviceNanEncoding`, as an output
    /// written as computed does on every device but one of `MetalColorWrites`.
    MetalFloatColorNans,
    /// The device stores a `half` NaN a fragment function writes (held widened to `float`, its
    /// payload in the top bits of the significand) to a half-float color channel as its own
    /// bits, as Metal stores a `half` output's NaN (`VkProperty::MetalHalfColorNans`). Unmeasured,
    /// a module that writes a `half` output to such a channel records
    /// `PrecisionNote::DeviceNanEncoding`.
    MetalHalfColorNans,
    /// The device's `GLSL.std.450 Fma` on 16-bit floats is Metal's own `half` `fma`, which
    /// rounds the exact `a * b + c` once (`VkProperty::MetalHalfFma`). Where the profile also has
    /// `shaderFloat16`, a `half` fused multiply-add converts its operands (each a `half`, so
    /// exactly) to 16 bits, takes the device's `Fma` and widens its result. Unmeasured, it takes
    /// `OpFmaKHR` on 16 bits where the profile has `shaderFmaFloat16` and rounds to `half` under
    /// the 16-bit float controls, and elsewhere the device's `float` `Fma` rounded to `half`,
    /// recorded as `PrecisionNote::NativeFma`.
    MetalHalfFma,
    /// The device's `float` arithmetic reads an `f32` denormal operand as the zero of its sign
    /// and flushes a denormal result to one, as Metal does (`VkProperty::MetalFloatDenormals`):
    /// `OpFAdd`, `OpFSub`, `OpFMul`, `OpFmaKHR`, `OpDot`, the comparisons, and `GLSL.std.450`'s
    /// `Ldexp`, `Sqrt`, `InverseSqrt`, `Trunc` and `RoundEven`. Unmeasured, such operands and
    /// results are the device's own, recorded as `PrecisionNote::DeviceDenormals`.
    MetalFloatDenormals,
    /// The device reads a center-sampled varying (a smooth input decorated neither `Centroid`
    /// nor `Sample`) of a multisampled pipeline at the pixel center in every fragment
    /// invocation, a sample-rate one's included, as Metal reads it
    /// (`VkProperty::MetalCenterVaryings`). Vulkan evaluates such an input at an
    /// implementation-defined position in the area the invocation shades. Where the profile takes
    /// it, the input's plain read is its pixel form (`StageIo::varyings_at_pixel`), with no
    /// `InterpolateAtOffset`. Unmeasured, the pixel form is `InterpolateAtOffset` at offset 0 where
    /// that reads from the center (`VkProperty::InterpolateAtOffsetCentered`), and a function that
    /// runs per sample is refused elsewhere.
    MetalCenterVaryings,
    /// The device stores a fragment function's `float` color output into a half-float channel,
    /// or one of `RG11B10Float`'s unsigned 11- and 10-bit floats, as Metal does for every value
    /// but a NaN, plain and through blending with factors one and zero
    /// (`VkProperty::MetalSmallFloatColorWrites`): rounded toward zero, a finite value past the
    /// largest stored as the largest, an infinity as itself, and where unsigned a negative value
    /// as 0. The function writes such an output as computed either way. Where the profile takes
    /// it, the lane records no `PrecisionNote::DeviceColorConversion` and the module requires the
    /// property; its NaN, as computed, is the device's (`PrecisionNote::DeviceNanEncoding`).
    /// Unmeasured, the lane records `DeviceColorConversion` (`LoweringInputs::color_writes`).
    MetalSmallFloatColorWrites,
    /// The device stores a fragment function's float color output into a 16-bit unorm or snorm
    /// channel as Metal does, the exact product with the format's largest code rounded to
    /// nearest, for every value, a NaN's included, plain and through blending with factors one
    /// and zero, except a blended snorm channel's NaN, which Metal stores as -1.0's code
    /// (`VkProperty::MetalNorm16ColorWrites`). The function writes such an output as computed
    /// either way. Where the profile takes it, a lane to such a channel but a blended snorm one
    /// records no `PrecisionNote::DeviceColorConversion` and the module requires the property;
    /// a blended snorm channel's lane, and every such lane on a profile without it, records the
    /// note.
    MetalNorm16ColorWrites,
    /// The device's `GLSL.std.450` `NMin`, `NMax` and `NClamp` on 32-bit floats are Metal's own
    /// `fmin`, `fmax` and `clamp` (`VkProperty::MetalMinMax`): a NaN operand loses, two NaNs and a
    /// tie answer the second operand's bits, an `f32` denormal compares as a zero but is returned
    /// itself, and `clamp(x, lo, hi)` is `fmin(fmax(x, lo), hi)`, in every math mode, of every
    /// operand class, and of every `half` held widened. Metal's `fmin`, `fmax`, `min3`, `max3`,
    /// `clamp` and `saturate` on `float` and `half` are those instructions on every profile: where
    /// the profile takes it the module requires the property, and unmeasured it records
    /// `PrecisionNote::NativeMinMax`.
    MetalMinMax,
    /// The device's 16-bit `OpFAdd` and `OpFSub` decorated `NoContraction`, and its 16-bit
    /// `Fma(a, b, -0.0)` decorated `NoContraction`, are Metal's own `half` `+`, `-` and `*`
    /// (`VkProperty::MetalHalfArithmetic`), with operands narrowed from `float` and the result
    /// widened back: the exact result rounded once to nearest even on the `half` grid, `half`
    /// denormals kept, an exact zero sum `+0` unless both addends are `-0`, and every NaN result
    /// `0x7E00`, in every math mode. Where the profile also has `shaderFloat16`, a `half` `fadd`,
    /// `fsub` or `fmul` converts its operands (each a `half`, so exactly) to 16 bits, takes that
    /// kept instruction and widens its result. Unmeasured, it takes the same instruction under the
    /// 16-bit float controls that make it Metal's by the Vulkan spec (`QuantizeForm::Controlled`),
    /// and elsewhere computes the result in `float` and rounds it to `half` as every `half`
    /// rounds (`narrow::round`, noted `NativeHalfConversion` or `NativeHalfQuantize` where that
    /// rounding is not Metal's). The measurement reads every `half` through the device's
    /// widening (`x + -0` of every `x`), exact but for a NaN, denormals included: a module that
    /// reads a `half` from its bits by that widening (every profile with `shaderFloat16` does)
    /// requires the property for its denormals, and records `PrecisionNote::NativeNanEncoding`
    /// for its NaNs' bits.
    MetalHalfArithmetic,
    /// The device stores every `float` but a NaN that a shader writes to a half-float storage
    /// image (`R16Float`, `RG16Float`, `RGBA16Float`; a storage texel buffer of one too) as Metal
    /// stores a kernel's write (`VkProperty::MetalHalfStorageWrites`): rounded toward zero, a
    /// finite value past 65504 stored as 65504, an infinity as itself. The shader writes the
    /// value as computed either way (`StorageImageClass::Identity`), its NaN as
    /// [`MetalHalfStorageNans`] states: a fragment function's `float` blend into a 16-bit float
    /// attachment in the storage form among them, which Metal's blended write rounds so too.
    /// Where the profile takes it the module requires the property; unmeasured, such a write
    /// records `PrecisionNote::DeviceColorConversion`. An integer texel written into a half-float
    /// image (its bits as the `float`), which Metal's integer store rounds to nearest even, is the
    /// store's to convert either way, and records the note.
    ///
    /// [`MetalHalfStorageNans`]: DriverDeviation::MetalHalfStorageNans
    MetalHalfStorageWrites,
    /// The device stores every NaN a shader writes to a half-float storage image as Metal stores
    /// a kernel's, `0x7e00` (`VkProperty::MetalHalfStorageNans`). The shader writes the NaN as
    /// computed either way; where the profile takes it the module requires the property, and
    /// unmeasured such a write records `PrecisionNote::DeviceNanEncoding`.
    MetalHalfStorageNans,
    /// The device's fragment subgroups are Metal's simdgroups
    /// (`VkProperty::MetalFragmentSimdgroups`): the driver hands the fragment function to Metal, so
    /// a fragment subgroup holds the pixels (at sample rate, the samples) Apple's rasterizer packs
    /// into one simdgroup, and its `SubgroupLocalInvocationId` is Metal's
    /// `thread_index_in_simdgroup`. Where the profile takes it, a fragment function's
    /// `[[thread_index_in_simdgroup]]` is that builtin. Unmeasured, the lane and the simdgroup
    /// operations run over the device's own grouping, which Metal does not rule out, and record
    /// `PrecisionNote::FragmentSimdgroupGrouping`: which pixels share a simdgroup, and in what
    /// order, is the rasterizer's packing, which nothing in Vulkan states or pins.
    MetalFragmentSimdgroups,
    /// The device runs a vertex function's subgroup operations as Metal runs its simdgroup
    /// functions, though its `subgroupSupportedStages` leaves the vertex stage out
    /// (`VkProperty::MetalVertexSimdgroups`): the driver hands the vertex function to Metal, which
    /// runs it in simdgroups of 32 consecutive vertices. Where the profile takes it, the profile's
    /// vertex stage has subgroups, at the device's subgroup size, of the operation classes measured
    /// exact there: basic, vote, arithmetic, ballot and shuffle. A vertex module that uses them
    /// requires the property in place of `VkProperty::SubgroupStage(Vertex)`. Unmeasured, the stage
    /// has none, as the device states.
    MetalVertexSimdgroups,
    /// In a fragment invocation of a multisampled pipeline that runs once per sample, the device
    /// gives `FragCoord.z` at the pixel centre, the draw's depth bias included, as Metal gives
    /// `[[position]].z`, reads a no-perspective varying at the pixel centre where it is read as it
    /// is and at the sample through `InterpolateAtSample`, and its depth test stores each sample's
    /// own depth (`VkProperty::CenterFragCoordDepth`). Where the profile takes it, a fragment
    /// function that runs its draw's depth test itself at more than one sample computes each
    /// sample's depth: its vertex function's window depth read at the sample, plus `FragCoord.z`
    /// less that varying read at the centre, which carries the bias
    /// (`PrecisionNote::ComputedSampleDepth`). Unmeasured, that test is refused at more than one
    /// sample.
    CenterFragCoordDepth,
    /// In a fragment invocation of a multisampled pipeline that runs once per sample, the device
    /// gives `FragCoord.z` at the invocation's sample, the draw's depth bias included: the depth
    /// its own depth test stores at that sample (`VkProperty::SampleFragCoordDepth`). Where the
    /// profile takes it, a fragment function that runs its draw's depth test itself at more than
    /// one sample tests `FragCoord.z` at each sample, exact as at one sample. Unmeasured, that test
    /// is refused at more than one sample, or computes each sample's depth where the profile takes
    /// [`DriverDeviation::CenterFragCoordDepth`].
    SampleFragCoordDepth,
    /// The device stores a `half` texel's write as Metal stores it once the shader hands it the
    /// half widened to `float` in a 16-bit float storage image, and the NaN `0x7fc00000` in a
    /// packed small float one (`VkProperty::MetalHalfTexelWrites`): the half's own bits, a
    /// signalling NaN's unquieted, in the first, and the format's canonical NaN in the second.
    /// Where the profile takes it, a write to a slot whose storage choice declares a 16-bit float
    /// format records no `PrecisionNote::DeviceNanEncoding` and the module requires the property.
    /// Unmeasured, such a write records the note, which the runtime announces at the bind of such
    /// a texture to a slot whose storage choice declares no format. A packed small float channel's NaN, the half widened, records
    /// the note on every device.
    MetalHalfTexelWrites,
    /// The device stores a `float` texel a shader writes to a unorm or snorm storage image (not
    /// an sRGB one; a `half` texel's, widened, alike) as Metal's code
    /// (`VkProperty::MetalNormStorageWrites`): each channel's value clamped to its range (a NaN 0)
    /// times its maximum code exactly, rounded to nearest with a tie away from zero, a packed
    /// 16-bit format's channel likewise and its 1-bit channel 1 only above one half. The shader
    /// writes the value as computed either way. Where the profile takes it the module requires
    /// the property, and a write that may reach a 1-bit alpha records
    /// `PrecisionNote::DeviceColorConversion`; unmeasured, every write that may reach such a format
    /// records it.
    MetalNormStorageWrites,
    /// The device stores every depth a draw writes to a `Depth16Unorm` attachment, unbiased, as
    /// Metal's code: the depth scaled by 65535 exactly and rounded to nearest, a tie up
    /// (`VkProperty::MetalDepth16Stores`; Metal stored 32768 for 0.5 and 6554 for 0.1's
    /// `0x3dcccccd`). Vulkan leaves the conversion's rounding to the device. Every depth is the
    /// device's own conversion; no module reads the deviation. Unmeasured, the runtime announces
    /// `unorm-depth-rounding` at a draw that tests or writes such a depth.
    MetalDepth16Stores,
    /// The device stores every depth a draw biases on a `Depth16Unorm` attachment as Metal's code,
    /// Metal's constant recorded in the device's unit as [`depth_bias`](crate::metal::state::depth_bias)
    /// records it (`VkProperty::MetalDepth16BiasedStores`): Metal biases by `c ± 1` units of
    /// 1/65535 as `float` (one unit for a slope alone), adds the bias to the depth rounding toward
    /// zero, and stores the sum rounded to nearest, a tie up. Vulkan leaves the unit (at most twice
    /// 2^-16) and the sum's rounding to the device. Every biased depth is the device's own, the
    /// attachment's fixed-function bias applied; no module reads the deviation. Unmeasured, the
    /// runtime announces `unorm-depth-bias` at a draw that biases such a depth by a constant, or
    /// by a slope alone.
    MetalDepth16BiasedStores,
    /// The device interpolates a varying as Metal does, bit for bit
    /// (`VkProperty::MetalInterpolation`): the driver hands the varying and its qualifiers to
    /// Metal, whose rasterizer interpolates it. Vulkan states no precision for interpolation, and
    /// every other device measured rounds otherwise. Where the profile takes it, a fragment
    /// function reading a varying the device interpolates records no note. Unmeasured, such a
    /// function records `PrecisionNote::DeviceInterpolation`.
    MetalInterpolation,
    /// The device takes a primitive's `Layer` and `ViewportIndex` from its first vertex, whatever
    /// its other vertices store, as Metal takes a primitive's slice and viewport
    /// (`VkProperty::MetalFirstVertexIndices`). Vulkan takes them only where every vertex of a
    /// primitive stores the same. Where the profile takes it, a vertex function whose index differs
    /// among a primitive's vertices stores it to its builtin. Unmeasured, it stores it through the
    /// pipeline's synthesized geometry stages, which store the first vertex's at every vertex,
    /// and is refused on a device without `geometryShader`.
    MetalFirstVertexIndices,
    /// The device runs a fragment function's quads whole, its helper invocations taking part in
    /// every group operation, as Metal runs a fragment function's helper threads
    /// (`VkProperty::MetalFullQuads`): the driver hands the fragment function to Metal, whose
    /// simdgroup functions read, combine and count a quad's helpers wherever the primitive or the
    /// target cuts the quad. Vulkan gives that only under `RequireFullQuadsKHR`
    /// (`shaderQuadControl`). Where the profile takes it and lacks `shaderQuadControl`, a
    /// fragment function whose group operations observe other lanes declares no such mode and
    /// requires the property in place of the feature. Unmeasured, such a function needs the
    /// feature.
    MetalFullQuads,
    /// The device's fixed-function vertex input converts the codes of a normalized format (the
    /// 8- and 16-bit `Normalized` formats of either sign, `UChar4Normalized_BGRA`,
    /// `Int1010102Normalized`, `UInt1010102Normalized`, through their `UNORM` and `SNORM` vertex
    /// formats) to Metal's `float`, at every code (`VkProperty::MetalNormalizedVertexFetch`): the
    /// code over its maximum correctly rounded, `c / (2^b - 1)` unsigned and
    /// `max(c / (2^(b-1) - 1), -1)` signed, but an `Int1010102Normalized` 10-bit field's, which
    /// Metal reads as the code times `float(1 / 511)` rounded to `float` (one `f32` ulp toward
    /// zero at 96 of the 1,024 codes). Vulkan states the correctly rounded quotient and lets a
    /// driver compute it otherwise. Where the profile takes it, a render pipeline whose vertex
    /// descriptor gives such a format to an attribute fixed-function input fetches records no
    /// `PrecisionNote::DeviceVertexConversion` for it and requires the property, but for a
    /// 16-bit format the function reads as `half`: Metal rounds the code's quotient to `half`
    /// once, where the module rounds the device's `float` to `half`, so that attribute records
    /// the note on every profile. Unmeasured, every such attribute records the note.
    MetalNormalizedVertexFetch,
    /// The device's alpha to coverage covers Metal's samples at every pixel
    /// (`VkProperty::MetalAlphaCoverage`): the driver hands `alphaToCoverageEnable` to Metal's own
    /// `alphaToCoverageEnabled`, whose coverage of an alpha depends on the pixel's place in a 2x2
    /// ordered dither at two and four samples. Vulkan specifies no algorithm for converting the
    /// alpha to a coverage mask. Where the profile takes it, a render pipeline that covers by
    /// alpha records no `PrecisionNote::DeviceAlphaCoverage` and requires the property.
    /// Unmeasured, such a pipeline's fragment stage records the note.
    MetalAlphaCoverage,
}

/// A defect of a device's driver in a command the runtime records for itself, never in a
/// translation (which [`DriverWorkaround`] covers). The runtime keys it on the driver's
/// identity and release (`driverID`, `driverVersion`; `runtime::driver_defects`), lists the
/// device's in its properties, and takes another way where an exact one exists, else refuses,
/// what the driver would run wrong ([`Needs::RuntimeDriverDefect`]), or runs it announced as a
/// degradation where a refusal would withhold a feature of the families the device claims. No
/// compile profile, cache key or digest holds it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuntimeDriverDefect {
    /// `vkCmdCopyImage` between a multisampled depth/stencil image and a multisampled colour
    /// image (maintenance8's depth-stencil/colour copies) does not copy every sample, or loses
    /// the device. The runtime copies so the separate multisampled depth and stencil textures
    /// (`Feature::SeparateMultisampledDepthStencil`) and the shadow of a multisampled combined
    /// depth/stencil attachment's stencil a fragment function writes without stencil export
    /// (`Feature::StencilExport`).
    MultisampledDepthStencilColourCopies,
    /// A primitive whose normalized depths reach past `[0, 1]` rasterizes inexact depths, where
    /// Metal's own viewport of the same depth range draws them exactly. The runtime draws a
    /// viewport depth range past `[0, 1]` that the device does not take natively through such
    /// depths (the vertex stage's compensation, `Feature::ViewportDepthRange`).
    NormalizedDepthPastUnitRange,
    /// A compressed image's memory requirements answer a size short of the memory the GPU
    /// writes its compression data through, so a resource bound right after the image loses what
    /// a command buffer that also writes the image writes into its first bytes. Nothing is
    /// refused: the runtime rounds each image's size up past that memory (whole 256-byte chunks
    /// of the aux map's on ANV's Xe-LP GPUs), so none of its resources lies there.
    ImageSizeShortOfCompressionData,
    /// A render pass's depth/stencil resolve (`VkSubpassDescriptionDepthStencilResolve`) reads
    /// its attachment's image, and writes its resolve attachment's, from their first array layer
    /// and mip level, whatever layer and level their views begin at. Nothing is refused but a
    /// memoryless attachment at another layer: the runtime resolves into a scratch image copied
    /// where the resolve belongs, and from a copy of an attachment at another layer.
    DepthStencilResolveFirstSubresource,
    /// A 2D view of a 3D image bound to lazily allocated memory (a memoryless 3D texture's depth
    /// plane attached to a pass) crashes the process: the driver makes the view by aliasing the
    /// image's memory through a placement heap, which lazily allocated memory lacks. Nothing is
    /// refused: the runtime gives such an image device-local memory, where, as in a memoryless
    /// texture, nothing a pass leaves is observable.
    LazilyAllocatedVolumePlaneViews,
    /// A `VkPipelineCache` made from saved data compiles every shader the data holds before
    /// `vkCreatePipelineCache` returns, used or not, so a device seeded with a large cache waits
    /// for all of them at its creation. Nothing is refused: the runtime neither seeds nor saves
    /// the driver's pipeline cache there, and each pipeline compiles its own shaders when it is
    /// made (the OS's shader cache, keyed by the shader's source, keeps what an earlier process
    /// compiled).
    PipelineCacheDataCompiledAtLoad,
    /// A copy into or out of an image of `B10G11R11_UFLOAT_PACK32` or `E5B9G9R9_UFLOAT_PACK32`
    /// that the driver keeps losslessly compressed rewrites some texels' bytes, as Metal's GPU
    /// blits of such a texture do (its NaNs made quiet, its shared exponents renormalized),
    /// where Metal's CPU copies keep them. Nothing is refused: the runtime makes the image of a
    /// texture a CPU copy can reach uncompressed there, whose copies keep the bytes as every
    /// other device's do.
    CompressedImageCopiesRewriteTexels,
    /// A copy into a multisampled image of a float format draws the samples through that format,
    /// so a NaN in a 16-bit float channel lands as the device's own NaN, its sign and payload
    /// lost, where Vulkan's copies keep every bit. Nothing is refused: the runtime writes a
    /// multisampled attachment's NaN clear through a view in the unsigned-integer format of its
    /// texel size (an integer clear converts nothing), which a multisampled color render target
    /// of such a format the runtime makes takes there where it costs the image nothing, and
    /// announces `device-nan-encoding` where the image takes no such view.
    MultisampledCopiesConvertFloatNans,
    /// The driver signals a queue's timeline semaphore before its own completion work for what
    /// was submitted ahead of the signal has run, and that work still reads objects the command
    /// buffers used (the query pools they reset or began): one released at the signal's value
    /// crashed. Nothing is refused: the runtime ends each command buffer there with a fence as
    /// well, which the driver signals after that work, and releases what the command buffer used
    /// only once both are signaled. Elsewhere a command buffer ends at its queue's value alone,
    /// and its commit makes no fence.
    ReadsCommandBuffersAfterTheirSignal,
    /// `vkCmdCopyImage` between the stencil aspect of a multisampled depth/stencil image and a
    /// multisampled colour image (maintenance8's depth-stencil/colour copies) loses the device,
    /// where the depth aspect's copies are right. The runtime copies such stencils as it does
    /// under [`RuntimeDriverDefect::MultisampledDepthStencilColourCopies`] (a kernel and draws, or
    /// a refusal), and their depths through the colour copies.
    MultisampledStencilColourCopies,
    /// A `vkCmdExecuteGeneratedCommandsEXT` the driver preprocesses itself, recorded after an
    /// execution whose execution set bound pipelines, runs that preprocessing with the shader the
    /// earlier execution bound last rather than the driver's own, which then reads and writes
    /// memory by the wrong arguments: the device faults or its queue hangs. Nothing is refused:
    /// the runtime preprocesses every execution of an indirect command buffer's range apart
    /// (`vkCmdPreprocessGeneratedCommandsEXT`) before the first of them runs.
    ImplicitPreprocessAfterExecutionSet,
    /// A render pass over an image of `VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT` whose one plane
    /// the caller laid out (`VkImageDrmFormatModifierExplicitCreateInfoEXT`) writes the rows past
    /// its height up to a multiple of 4, past its subresource and its memory requirements (into
    /// another resource's bytes, or unmapped memory). Nothing is refused: the runtime makes no
    /// such image of a height that is no multiple of 4 there, and a texture over a buffer or a
    /// heap's pages takes its shadow (`shadowed-texture`).
    ExplicitPlaneRowsPastHeight,
    /// A precise occlusion query counts each sample that passes once per color output the
    /// fragment shader writes, where Vulkan counts it once. Nothing is refused: a draw counted by
    /// a `Counting` visibility result whose fragment stage writes more than one color attachment
    /// runs, announced (`visibility-count-per-color`); `Boolean` results are Metal's.
    OcclusionCountPerColorOutput,
    /// Freeing memory that a buffer of `SHADER_DEVICE_ADDRESS` usage lay in fails a submission
    /// still pending whose pipelines read device addresses, though none of its commands refers
    /// to that memory. Nothing is refused: the runtime destroys such a buffer at once and frees
    /// its memory once the batches and its own submissions made before it completed, a no-copy
    /// buffer's deallocator waiting with it, announced (`host-deallocator-delayed`). The
    /// program's own submissions to the runtime's queues are not waited for.
    FreedAddressMemoryFailsPendingWork,
    /// Destroying a buffer of `SHADER_DEVICE_ADDRESS` usage while another thread is inside a
    /// `vkQueueSubmit` whose pipelines read device addresses fails that submission, though none of
    /// its commands refers to the buffer. Nothing is refused: the runtime destroys such a buffer
    /// with no submission of its own inside its `vkQueueSubmit`. The program's own submissions to
    /// the runtime's queues are not waited for.
    DestroyedAddressBufferFailsSubmission,
    /// `vkCmdPushDescriptorSet` writes the descriptors it pushes but not the sizes of their
    /// buffers that a shader reads back (a storage buffer's runtime array length,
    /// `OpArrayLength`), which the driver takes from `vkCmdBindDescriptorSets` alone: a pushed
    /// buffer's length reads 0, so the accesses a shader bounds by it read and write nothing.
    /// Nothing is refused: the runtime binds set 0 there, as on a device without the push
    /// command.
    PushedBufferSizesUnset,
    /// Descriptors of set 0 pushed (`vkCmdPushDescriptorSet`) before a bind of another pipeline
    /// of the same layout, which Vulkan keeps for it (Pipeline Layout Compatibility), are not
    /// what that pipeline reads where no push follows the bind: a uniform buffer it reads
    /// through its descriptor reads zeros there after a pipeline that read the same buffer's
    /// constants at fixed offsets. Nothing is refused: the first draw after each pipeline bind
    /// there pushes at least one of set 0's descriptors, those that changed or else the first.
    PushedDescriptorsStaleAfterPipelineBinds,
    /// A compute dispatch recorded after a copy in the same `VkCommandBuffer` samples a binding
    /// its pipeline layout bakes an immutable sampler into (a constexpr sampler's) as Repeat, not
    /// with the baked sampler's modes. Nothing is refused: an encoder's dispatch of a variant
    /// that bakes samplers records in a `VkCommandBuffer` of its own there, as one whose pipeline
    /// is still being made does.
    ImmutableSamplersLostAfterCopies,
    /// `vkGetDescriptorSetLayoutSupport` answers `VK_FALSE` for a descriptor set layout the driver
    /// makes and runs: one of exactly `maxPerSetDescriptors` descriptors, which Vulkan guarantees
    /// `VK_TRUE`, and every one past it, however far within the update-after-bind limits the
    /// driver reports. Nothing is refused: the runtime makes such a layout there (a heap's set 1)
    /// as the driver runs it, announced (`set-past-layout-support`), and Vulkan's validation layer
    /// reports it (`VUID-vkCreateDescriptorSetLayout-support-09582`).
    SetLayoutSupportUnderstated,
    /// A view of some of the levels or layers of an image bound to lazily allocated memory (a
    /// memoryless texture's level, slice or cube face attached to a pass) has no texture: the
    /// driver makes such a view as a Metal texture view, and Metal makes no view of a memoryless
    /// texture, so the pass renders without the attachment (its resolve too). Nothing is
    /// refused: the runtime gives such an image, one of more than one level or layer,
    /// device-local memory, where, as in a memoryless texture, nothing a pass leaves is
    /// observable.
    LazilyAllocatedSubresourceViews,
    /// Binding memory to an image with host-transfer usage copies the whole image **out of that
    /// memory on the CPU** before anything has written it, through a row pitch of the driver's
    /// own, and the driver's own check of that copy fails for an image whose texel block holds
    /// more than one texel: it reports the pitch as short of the row it reads. Nothing is
    /// refused: a block-compressed texture's image takes no host-transfer usage there and so
    /// stays in memory of its own class, and its CPU copies go through a submission, as on a
    /// device without `VK_EXT_host_image_copy`.
    HostTransferImagesCopiedAtBind,
    /// A host wait for any of several semaphores or fences to be signaled (`vkWaitSemaphores`
    /// with `VK_SEMAPHORE_WAIT_ANY_BIT`, `vkWaitForFences` with `waitAll` false) loses a signal
    /// that lands while the call is still registering a later one of them: it sleeps on to its
    /// timeout although the condition it waited for was met. A wait that names one object loses
    /// none. Nothing is refused: every host wait of the runtime names one object there — each
    /// queue's completion waiter its queue's next fence or timeline value — and the scheduler's
    /// thread sleeps on its condition variable, woken by the completion turns, in place of
    /// waiting for the signals its held batches wait for to land on the GPU itself.
    WaitForAnyObjectLosesASignal,
    /// A sampler Y′CbCr conversion of a two-plane 4:2:0 or 4:2:2 format reconstructs the chroma
    /// away from where its chroma offsets place it: each chroma sample's weights, of the right
    /// shape, lie one luma texel up and to the left of the midpoint's (or, cosited, are no
    /// interpolation at all), and a linear filter filters the chroma twice, which Vulkan's
    /// reconstruction (`VkChromaLocation`, `VkFilter`) forbids. Nothing is refused: the runtime
    /// converts such a texture's planes with its own kernel there, as on a device without the
    /// conversion.
    TwoPlaneChromaMisplaced,
    /// A sampler Y′CbCr conversion of a 10-bit multi-planar format (`..._3PACK16`) reads each
    /// code's 16-bit word, its six unused low bits included, as a 16-bit unsigned normalized
    /// value, where Vulkan converts the 10-bit code as `c / 1023`: a code reads up to one code
    /// low (1023 as 0.999039). Nothing is refused: the runtime converts such a texture's planes
    /// with its own kernel there, as on a device without the conversion.
    TenBitCodesReadAsWords,
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
    /// A metallib that is not an executable library (dynamic library, companion).
    LibraryKind(u8),
    /// A function stage the model does not cover.
    Stage(u8),
    /// An LLVM address space the typed model does not define (it defines AIR's `thread` 0
    /// through `threadgroup_imageblock` 4, `object_data` 6 and the mesh output handle's 7). The 5
    /// of Apple's AGX ray-tracing and tessellation library modules has no established meaning and
    /// stays refused; a pointer in 6 or 7 outside an object or mesh function's payload or mesh
    /// handle is refused as this feature too, at analysis.
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
    /// A pixel format not supported for the required use.
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

    // ---- the refused families, not implemented yet ----
    /// Tessellation: a post-tessellation vertex function and its pipeline's patch draws, through
    /// the device's tessellation stages, or emulated on a device without `tessellationShader`.
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
    /// A bitcode constant record outside the typed model (block address, inline asm, `select`
    /// and other constant expressions the IR has no form for).
    Constant {
        code: u32,
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
    /// dereference is established. An absent read sets `StatusBits::ABSENT_ARGUMENT_READ` where it
    /// runs and yields zero (`RuntimeFact::AbsentArgumentRead`); in a vertex or fragment stage
    /// without its stores and atomics it is refused with what it needs.
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
    /// it loads or may store, on a device without maintenance8, or whose driver copies such
    /// aspects wrong ([`RuntimeDriverDefect::MultisampledDepthStencilColourCopies`]): Vulkan copies
    /// a multisampled depth or stencil aspect between images of different formats only through a
    /// colour image of maintenance8's depth-stencil/colour table.
    SeparateMultisampledDepthStencil,
    /// The fragment's coverage (`[[sample_mask]]` read) in a fragment function of a pipeline of
    /// more than one sample that runs once per pixel.
    PixelCoverage,
    /// A centroid varying (`[[centroid_perspective]]`, `[[centroid_no_perspective]]`) in a
    /// fragment function of a pipeline of more than one sample.
    CentroidVarying,
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
    /// A viewport depth range past `[0, 1]` (`setViewport:`), which Metal takes and the runtime
    /// draws natively (`VK_EXT_depth_range_unrestricted`) or through the vertex stage's
    /// compensation.
    ViewportDepthRange,
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
    /// A sequentially consistent fence (`seq_cst` `atomic_thread_fence`, an LLVM `fence seq_cst`,
    /// or a `seq_cst` barrier ordering device memory), whose total order with the command
    /// buffer's other such fences the translation keeps through an atomic of the command buffer's
    /// `ExecutionStatus` word: in a vertex or fragment stage without its stores and atomics.
    SeqCstFence,
    /// The coverage after the depth and stencil tests (`[[sample_mask, post_depth_coverage]]`),
    /// read through `VK_EXT_post_depth_coverage`.
    PostDepthCoverage,
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
    /// A value use of an unsupplied function constant survived SCCP.
    FunctionConstant { index: u32 },
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
    /// function writes (`IoRole::PrimitiveId`: on a device that cannot declare a fragment
    /// `PrimitiveId`, or behind a post-tessellation vertex function emulated on a device without
    /// `tessellationShader`), and the vertex function was compiled without
    /// `LoweringInputs::primitive_id_varying`, so it writes none; or the vertex function writes it
    /// and the fragment function, compiled for another vertex function, reads Vulkan's
    /// `PrimitiveId`, which counts the tessellator's triangles there. It is compiled for the
    /// pipeline again.
    PrimitiveId,
    /// A post-tessellation vertex function compiled for a device without `tessellationShader`
    /// bakes its render pipeline descriptor's tessellation state in
    /// (`LoweringInputs::tessellation`), and was compiled for another state than its pipeline's:
    /// it is compiled for the pipeline again.
    TessellationDescriptor,
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
    /// function's: compiled on its own or for another pipeline, it would read zeros past its own
    /// payload where Metal reads the object function's (`LoweringInputs::object_payload`). It is
    /// compiled for this pipeline again.
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
    /// A color attachment format without a fragment output, or of the wrong type class; one a
    /// fragment output has fewer components for than its channels; one whose blend reads a
    /// second output the function does not write, or writes without the first; or an integer
    /// attachment 0 under alpha to coverage (Metal refuses each pipeline).
    ColorAttachment { index: u32 },
    /// A raw Metal enum value outside the known set.
    UnknownEnumValue { ty: &'static str, value: u32 },
    /// A pipeline descriptor without a required stage.
    MissingStage { stage: Stage },
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
    /// A render pipeline descriptor whose `maxVertexAmplificationCount` is 0: its encoders draw
    /// with a count in `[1, maxVertexAmplificationCount]`, so the maximum is at least 1.
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
    /// A `StageSource::Compiled` stage whose recursion's call stack was sized for
    /// `maxCallStackDepth` `compiled` (`RuntimeFact::CallStackDepth`) in a pipeline whose depth
    /// for that stage is `given` (0 taken as 1): a compute or tile pipeline's, or a render
    /// pipeline's vertex or fragment depth. The stack would hold another number of frames than the
    /// pipeline's Metal stage has. Compile the stage under the pipeline's depth.
    MaxCallStackDepth { compiled: u32, given: u32 },
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
    /// A texel buffer exceeds `maxTexelBufferElements`.
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
    /// A vertex buffer layout with a per-bind stride (`MTLBufferLayoutStrideDynamic`) reads
    /// inline bytes (`setVertexBytes`) bound at `MTLAttributeStrideStatic`
    /// (`BoundBuffer::Bytes::attribute_stride` `Some(u32::MAX)`), which an M4 read as element 0
    /// and zeros past it (Metal's debug layer stops the draw): not emulated.
    /// Inline bytes and a buffer bound without a stride read at stride 0, as plain Metal's encoder
    /// reads them, and at the stride bound with one.
    MissingVertexStride { slot: SlotRef },
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
