//! The front end's errors: why reading or canonicalizing AIR stopped.
//!
//! Every reader and canonicalization entry returns [`Error`]. Its variants are the translator's
//! outcome classes as the front end meets them: a typed refusal of valid AIR the front end does not
//! cover ([`Unsupported`], naming a [`Feature`]), an input the request did not supply
//! ([`MissingInput`]), malformed or ABI-inconsistent input ([`InputError`]), an inconsistent
//! request ([`RequestError`]), a deterministic cap ([`Limit`]), the attempt's deadline or
//! cancellation, and an internal invariant ([`InternalError`]). The translator's own error type
//! converts each variant into its own one to one, so a translation reports the front end's
//! outcome unchanged.
//!
//! Every input error is deterministic for the same bytes and request. [`Error`] wraps
//! [`InputError`] and maps each [`InputErrorKind`] to one outcome class.
//!
//! [`DiagName`] and [`Location`] are opaque: they render for diagnostics and compare for tests,
//! but expose no string or position a caller could branch on (no name-keyed behavior, AGENTS.md).
//! A location's one typed part is the render pipeline [`Attachment`] a refusal is tied to
//! ([`Location::attachment`]): a slot of the caller's own descriptor, not a place in the input.
//!
//! The outcome vocabulary is exhaustive: [`Error`], [`Unsupported`], [`Feature`],
//! [`MissingInput`], and [`RequestError`] are what every consumer interprets completely (the
//! translator maps each variant to its own, one to one), so a new variant or field is a compile
//! error there rather than a silently unmapped outcome, and arrives in a minor (0.x) release. The
//! detail vocabulary is `#[non_exhaustive]`, every other enum here, [`InputError`], and
//! [`InternalError`]: new details arrive additively, so match those with a wildcard arm.

use std::fmt;

/// A symbol or metadata name kept for diagnostics only.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct DiagName(Box<str>);

impl DiagName {
    /// Wrap a name for diagnostics.
    pub fn new(name: impl Into<Box<str>>) -> DiagName {
        DiagName(name.into())
    }
}

impl fmt::Debug for DiagName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.0, f)
    }
}

impl fmt::Display for DiagName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Where in the input an error was found, and the render pipeline attachment it is tied to. Opaque
/// but for the attachment; see the module docs.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Location {
    kind: LocationKind,
    attachment: Option<Attachment>,
}

/// A render pipeline's attachment a refusal is tied to ([`Location::attachment`]): one of its
/// state (its pixel format, blend, write mask or sample count), or of the fragment function's
/// output to it or input from it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Attachment {
    /// The color attachment of this index.
    Color(u32),
    /// The depth attachment.
    Depth,
    /// The stencil attachment.
    Stencil,
}

impl fmt::Display for Attachment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Attachment::Color(index) => write!(f, "color attachment {index}"),
            Attachment::Depth => f.write_str("depth attachment"),
            Attachment::Stencil => f.write_str("stencil attachment"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum LocationKind {
    Container {
        offset: u64,
    },
    Bitstream {
        bit: u64,
        blocks: Box<[u32]>,
        record: Option<u32>,
    },
    Function {
        func: DiagName,
        block: Option<u32>,
        inst: Option<u32>,
    },
    Metadata {
        entry: Option<DiagName>,
        arg: Option<u32>,
        key: Option<&'static str>,
    },
    Module,
    /// A render pipeline's descriptor, where linking its stages refuses it.
    RenderPipeline,
}

impl Location {
    /// A location of `kind`, tied to no attachment.
    fn of(kind: LocationKind) -> Location {
        Location {
            kind,
            attachment: None,
        }
    }

    /// A byte offset in the container.
    pub fn container(offset: u64) -> Location {
        Location::of(LocationKind::Container { offset })
    }

    /// A bit position in the bitstream, with the enclosing block ids (outermost first) and the
    /// record index within the innermost block.
    pub fn bitstream(bit: u64, blocks: &[u32], record: Option<u32>) -> Location {
        Location::of(LocationKind::Bitstream {
            bit,
            blocks: blocks.into(),
            record,
        })
    }

    /// A function, optionally narrowed to a block and an instruction index.
    pub fn function(func: DiagName, block: Option<u32>, inst: Option<u32>) -> Location {
        Location::of(LocationKind::Function { func, block, inst })
    }

    /// ABI metadata, optionally narrowed to an entry point, argument, and key.
    pub fn metadata(
        entry: Option<DiagName>,
        arg: Option<u32>,
        key: Option<&'static str>,
    ) -> Location {
        Location::of(LocationKind::Metadata { entry, arg, key })
    }

    /// The module as a whole.
    pub fn module() -> Location {
        Location::of(LocationKind::Module)
    }

    /// A render pipeline's descriptor, where linking its stages refuses it (no stage's
    /// function or metadata is at fault).
    pub fn render_pipeline() -> Location {
        Location::of(LocationKind::RenderPipeline)
    }

    /// This location, tied to render pipeline attachment `attachment`.
    pub fn with_attachment(self, attachment: Attachment) -> Location {
        Location {
            kind: self.kind,
            attachment: Some(attachment),
        }
    }

    /// The render pipeline attachment the error is tied to, if any.
    pub fn attachment(&self) -> Option<Attachment> {
        self.attachment
    }

    /// Whether this is a render pipeline's descriptor ([`Location::render_pipeline`]), which a
    /// stage that compiles under it places at its own function or metadata.
    pub fn is_render_pipeline(&self) -> bool {
        matches!(self.kind, LocationKind::RenderPipeline)
    }

    /// The function, block, and instruction index of a function location (for diagnostics and
    /// census tooling).
    pub fn in_function(&self) -> Option<(&DiagName, Option<u32>, Option<u32>)> {
        match &self.kind {
            LocationKind::Function { func, block, inst } => Some((func, *block, *inst)),
            _ => None,
        }
    }

    /// The entry, argument index, and key of an ABI-metadata location (for diagnostics and census
    /// tooling): a refusal of an entry parameter is at its argument, one of the entry's interface
    /// at the entry.
    pub fn in_metadata(&self) -> Option<(Option<&DiagName>, Option<u32>, Option<&'static str>)> {
        match &self.kind {
            LocationKind::Metadata { entry, arg, key } => Some((entry.as_ref(), *arg, *key)),
            _ => None,
        }
    }
}

impl fmt::Debug for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.kind, f)?;
        if let Some(attachment) = self.attachment {
            write!(f, " at {attachment:?}")?;
        }
        Ok(())
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            LocationKind::Container { offset } => write!(f, "container offset {offset}"),
            LocationKind::Bitstream {
                bit,
                blocks,
                record,
            } => {
                write!(f, "bitstream bit {bit} in blocks {blocks:?}")?;
                if let Some(record) = record {
                    write!(f, ", record {record}")?;
                }
                Ok(())
            }
            LocationKind::Function { func, block, inst } => {
                write!(f, "function {func}")?;
                if let Some(block) = block {
                    write!(f, ", block {block}")?;
                }
                if let Some(inst) = inst {
                    write!(f, ", instruction {inst}")?;
                }
                Ok(())
            }
            LocationKind::Metadata { entry, arg, key } => {
                f.write_str("metadata")?;
                if let Some(entry) = entry {
                    write!(f, " of entry {entry}")?;
                }
                if let Some(arg) = arg {
                    write!(f, ", argument {arg}")?;
                }
                if let Some(key) = key {
                    write!(f, ", key {key}")?;
                }
                Ok(())
            }
            LocationKind::Module => f.write_str("module"),
            LocationKind::RenderPipeline => f.write_str("render pipeline"),
        }?;
        match self.attachment {
            Some(attachment) => write!(f, ", {attachment}"),
            None => Ok(()),
        }
    }
}

/// An error decoding or canonicalizing AIR input.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputError {
    /// What went wrong.
    pub kind: InputErrorKind,
    /// Where in the input it was found.
    pub at: Location,
}

impl InputError {
    /// An error of `kind` at `at`.
    pub fn new(kind: InputErrorKind, at: Location) -> InputError {
        InputError { kind, at }
    }

    /// The coarse class of this error.
    pub fn class(&self) -> ErrorClass {
        match &self.kind {
            InputErrorKind::Container(_)
            | InputErrorKind::Bitstream(_)
            | InputErrorKind::Bitcode(_)
            | InputErrorKind::Verify(_)
            | InputErrorKind::Abi(_)
            | InputErrorKind::Canon(CanonError::UndefCondition | CanonError::CallTypeMismatch) => {
                ErrorClass::Malformed
            }
            InputErrorKind::Intrinsic(IntrinsicError::UnknownIntrinsic { .. })
            | InputErrorKind::Unsupported(_)
            | InputErrorKind::Canon(_) => ErrorClass::Unsupported,
            InputErrorKind::Intrinsic(_) => ErrorClass::Malformed,
            InputErrorKind::Request(_) => ErrorClass::InvalidRequest,
            InputErrorKind::Limit(_) => ErrorClass::LimitExceeded,
        }
    }
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} at {}", self.kind, self.at)
    }
}

impl std::error::Error for InputError {}

/// The coarse class of an [`InputError`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorClass {
    /// Malformed or ABI-inconsistent input.
    Malformed,
    /// Valid input the front end does not cover.
    Unsupported,
    /// A request inconsistent with the library.
    InvalidRequest,
    /// A deterministic cap was exceeded.
    LimitExceeded,
}

/// What went wrong with the input.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputErrorKind {
    /// The metallib container is malformed.
    Container(ContainerError),
    /// The LLVM bitstream is malformed.
    Bitstream(BitstreamError),
    /// A bitcode record is malformed or outside the accepted subset.
    Bitcode(BitcodeError),
    /// The decoded IR breaks a construction invariant.
    Verify(VerifyError),
    /// Valid input that the typed model does not cover (maps to `Unsupported{NotImplemented}`).
    Unsupported(Feature),
    /// An intrinsic call is unknown or ill-formed.
    Intrinsic(IntrinsicError),
    /// AIR metadata is inconsistent with the ABI.
    Abi(AbiError),
    /// Canonicalization cannot complete.
    Canon(CanonError),
    /// The request (entry name, function-constant values) is invalid for this library.
    Request(RequestError),
    /// A structural cap was exceeded (checked before allocation).
    Limit(Limit),
}

/// Container-layer errors.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContainerError {
    /// The input starts with no metallib, bitcode-wrapper, or bitcode magic, or a wrapper's
    /// stream does not start with the bitcode magic.
    BadMagic,
    /// The input ends before a structure it declares.
    Truncated {
        /// The input length, in bytes, the structure needs.
        need: u64,
        /// The input's length in bytes.
        have: u64,
    },
    /// The metallib header states a container version the reader does not support.
    UnsupportedContainerVersion {
        /// The header's major version.
        major: u16,
        /// The header's minor version.
        minor: u16,
    },
    /// A function's bitcode does not hash to its `HASH` tag (SHA-256).
    HashMismatch,
    /// An offset or size that points outside the file or its section, or ranges of two
    /// functions' bitcode or metadata that overlap.
    BadOffset,
    /// A malformed tag or tag list: it overruns its region, has the wrong length for its tag, or
    /// disagrees with its record's size word.
    BadTag,
    /// A tag that may appear once per function appears twice.
    DuplicateTag,
    /// A function entry lacks a required tag (`NAME`, `TYPE`, `HASH`, `MDSZ`, `OFFT`, `VERS`).
    MissingTag,
    /// A universal metallib without the slice of its AIR.
    NoAirSlice,
}

/// Bitstream-layer errors. Excess block nesting is `Limit::BitstreamNesting`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BitstreamError {
    /// A malformed abbreviation definition (no operands, a misplaced array or blob, an unknown
    /// encoding, a bad width), or a field width out of range.
    BadAbbrev,
    /// A VBR value that does not fit 64 bits.
    BadVbr,
    /// A read past the end of the stream or of the enclosing block.
    UnexpectedEnd,
    /// An abbreviation id that no definition in scope gives (at the top level, any id but a
    /// block's).
    UnknownAbbrevId {
        /// The abbreviation id read.
        id: u32,
    },
    /// A `BLOCKINFO` block with a nested block, an abbreviation before its first `SETBID`, or a
    /// `SETBID` without an operand.
    BadBlockInfo,
    /// A blob that does not start at a byte boundary.
    BadBlob,
    /// A block ends at a different position than its length word declares, or overruns its parent.
    BadBlockLength,
    /// A field exceeds the range of what it encodes (block id, record code, abbrev width).
    ValueOutOfRange,
}

/// Bitcode-record errors.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BitcodeError {
    /// The module's `VERSION` record states a version other than 2.
    UnsupportedBitcodeVersion {
        /// The version the record states.
        version: u64,
    },
    /// An `IDENTIFICATION` block that repeats, follows the module, or has an epoch other than 0.
    BadIdentification,
    /// A block where none of that id is accepted: an unknown block, a second module, or a
    /// sub-block of a block that holds only records.
    UnknownBlock {
        /// The block's id.
        block: u32,
    },
    /// A record code its block does not define.
    UnknownRecord {
        /// The enclosing block's id.
        block: u32,
        /// The record's code.
        code: u32,
    },
    /// A record with missing, out-of-range, or inconsistent operands, or a block whose structure
    /// is wrong (a block ending early, a record at the top level, no module block).
    MalformedRecord {
        /// The enclosing block's id.
        block: u32,
        /// The record's code (0 when the block's structure is wrong).
        code: u32,
    },
    /// A record names a value that does not exist or is of the wrong kind.
    BadValueRef,
    /// A malformed or cyclic type, or a record that names a missing type or one of the wrong
    /// kind.
    BadType,
    /// An attribute kind, encoding, or `memory(..)` version that LLVM does not define.
    UnknownAttribute,
    /// Metadata that refers to a missing node, a named-metadata name without its operands, or an
    /// attachment that is not a node.
    BadMetadata,
    /// A string table (`STRTAB`) that is missing, repeated, or not exactly one blob.
    BadStringTable,
    /// The data layout string is not UTF-8 or does not parse.
    BadDataLayout,
}

/// IR construction invariants.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// A function body has no block, or a block is empty, does not tile the instruction arena,
    /// or does not end in a terminator.
    MissingTerminator,
    /// A terminator before the last instruction of its block.
    TerminatorNotLast,
    /// A branch targets the function's entry block.
    EntryBlockHasPredecessors,
    /// A branch names a block the function does not have, or a block's name is not a symbol.
    BadBlockRef,
    /// An operand names an argument, instruction, constant, function, or symbol that does not
    /// exist, or the result of a `void` instruction.
    BadValueRef,
    /// An operand's type does not fit its instruction (or an intrinsic callee has a body).
    OperandType,
    /// An instruction's result type does not exist or does not fit its operation.
    ResultType,
    /// A PHI after a non-PHI in its block, or whose incoming blocks are not its block's
    /// predecessors.
    PhiPredecessors,
    /// A reachable use that its definition does not dominate.
    UseNotDominated,
    /// A type breaks the type-construction rules (bad width, unsized member, missing child).
    MalformedType,
    /// A constant breaks the constant typing rules.
    MalformedConstant,
    /// Metadata refers to a missing node, string, or constant.
    MalformedMetadata,
    /// A function's signature, attributes, or linkage are inconsistent.
    MalformedFunction,
    /// A global's type, initializer, or linkage are inconsistent.
    MalformedGlobal,
}

/// Intrinsic decode errors.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntrinsicError {
    /// A declared intrinsic outside the closed set (maps to `Unsupported{NotImplemented}`).
    UnknownIntrinsic {
        /// The declared intrinsic's name, for diagnostics.
        name: DiagName,
    },
    /// The call's signature does not match the intrinsic's typed signature.
    SignatureMismatch,
    /// An operand that must be an immediate is not a constant.
    ImmediateNotConstant,
}

/// AIR metadata ABI errors (the ABI's cross-checks).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AbiError {
    /// A named list, key, option, role, or stage attribute the ABI does not define, or one the
    /// role or stage at hand does not accept.
    UnknownKey,
    /// A required key or operand is absent (for example a buffer's `air.location_index`).
    MissingKey,
    /// A value of the wrong type or range, a repeated single-valued key, or a node of the wrong
    /// shape.
    BadValue,
    /// The metadata disagrees with a function's signature: not one binding per parameter,
    /// outputs that do not match the return type, or a control-point function of the wrong type.
    SignatureMismatch,
    /// A metallib function's container metadata (`TYPE`, `CNST`) disagrees with its bitcode's
    /// ABI (`AirAbi::check_container`).
    ContainerMismatch,
    /// A parameter's role disagrees with its type.
    RoleTypeMismatch,
    /// `air.arg_type_size` disagrees with the DataLayout size of the pointee.
    TypeSizeMismatch,
    /// `air.struct_type_info` disagrees with the DataLayout of the pointee.
    StructInfoMismatch,
    /// A constexpr sampler word does not decode.
    BadSamplerState,
    /// An entry parameter's argument-table slots run past Metal's table for its kind: 31 buffer
    /// entries (a buffer or argument buffer, a buffer table's elements, or an acceleration
    /// structure or function table, which bind at a buffer index), 31 threadgroup memory
    /// entries, 128 texture entries, or 16 sampler entries per function, every GPU family's
    /// maximum in Metal's feature set tables. No Metal compiler emits such a binding.
    SlotPastTable,
    /// An entry's root carries a trailing attribute its stage does not read: one no stage models,
    /// or one another stage models (`early_fragment_tests` on a kernel). Named by the key the root
    /// states: the string operand itself, or a keyed node's first string.
    UnknownStageAttribute(DiagName),
}

/// Canonicalization errors. Recursion and dynamic calls are `Unsupported(Feature::Recursion |
/// Feature::DynamicCall)`; an unsupplied function constant with a surviving value use is
/// `Error::NeedsInput`; the inline cap is `Limit::InlinedInstructions`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CanonError {
    /// A static initializer cannot be evaluated to constant bytes.
    StaticInitNotEvaluable,
    /// A reachable branch or switch tests a literal `undef` or `poison` (undefined behavior).
    UndefCondition,
    /// A reachable call to a function the module defines (or a linked function a stub resolves
    /// to), at a function type whose arguments differ from the function's parameters in count or
    /// in shape (identified structs matched by body). No rule binds such a call's arguments to the
    /// callee's parameters: an LLVM call's arguments match its own function type, and a call
    /// through an incompatible function type is undefined behavior in C and C++ (C11 6.5.2.2p9),
    /// which MSL inherits; which argument a parameter receives would be Apple's backend's calling
    /// convention, which AIR does not carry. A call at another type whose arguments match (another
    /// return type, identically shaped structs of other names) or of a variadic type is
    /// `Unsupported{DirectCall}`.
    CallTypeMismatch,
}

/// Why reading or canonicalizing AIR did not produce a module or an entry.
///
/// The translator's error type converts each variant into its own one to one
/// (`Unsupported` with `Needs::NotImplemented`, `InvalidRequest` as `InvalidDescriptor`), so
/// nothing here is lost or reclassified on the way out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Valid AIR that the front end does not cover: a construct of a typed [`Feature`] not
    /// implemented yet.
    Unsupported(Unsupported),
    /// An input the request did not supply: a function-constant value or a linked function.
    NeedsInput(MissingInput),
    /// Malformed or ABI-inconsistent input, or input that fails IR verification.
    InvalidInput(InputError),
    /// The request (entry, function-constant values, linked functions) is inconsistent with the
    /// library.
    InvalidRequest(RequestError),
    /// A deterministic cap was exceeded (structural caps, work units, ledger bytes). Permanent for
    /// the input.
    LimitExceeded(Limit),
    /// The wall-clock deadline passed. Transient: never cached or recorded as an observation.
    Budget(BudgetExceeded),
    /// The caller cancelled the attempt. Transient.
    Cancelled,
    /// An internal invariant failed: a builder or verifier rejection, or a host allocation the
    /// ledger admitted that failed.
    Internal(InternalError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported(u) => write!(f, "unsupported: {u}"),
            Error::NeedsInput(i) => write!(f, "needs input: {i:?}"),
            Error::InvalidInput(e) => write!(f, "invalid input: {e}"),
            Error::InvalidRequest(e) => write!(f, "invalid request: {e:?}"),
            Error::LimitExceeded(l) => write!(f, "limit exceeded: {l:?}"),
            Error::Budget(b) => write!(f, "budget exceeded: {b:?}"),
            Error::Cancelled => f.write_str("cancelled"),
            Error::Internal(e) => write!(f, "internal error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<InputError> for Error {
    /// The mapping of input error kinds to outcome classes.
    fn from(e: InputError) -> Error {
        let InputError { kind, at } = e;
        match kind {
            InputErrorKind::Unsupported(feature) => {
                Error::Unsupported(Unsupported::not_implemented(feature).at(at))
            }
            InputErrorKind::Intrinsic(IntrinsicError::UnknownIntrinsic { name }) => {
                Error::Unsupported(
                    Unsupported::not_implemented(Feature::UnknownIntrinsic)
                        .at(at)
                        .naming(name),
                )
            }
            InputErrorKind::Canon(CanonError::StaticInitNotEvaluable) => {
                Error::Unsupported(Unsupported::not_implemented(Feature::StaticInitializer).at(at))
            }
            kind @ InputErrorKind::Canon(
                CanonError::UndefCondition | CanonError::CallTypeMismatch,
            ) => Error::InvalidInput(InputError { kind, at }),
            InputErrorKind::Request(r) => Error::InvalidRequest(r),
            InputErrorKind::Limit(l) => Error::LimitExceeded(l),
            kind @ (InputErrorKind::Container(_)
            | InputErrorKind::Bitstream(_)
            | InputErrorKind::Bitcode(_)
            | InputErrorKind::Verify(_)
            | InputErrorKind::Intrinsic(_)
            | InputErrorKind::Abi(_)) => Error::InvalidInput(InputError { kind, at }),
        }
    }
}

impl From<Unsupported> for Error {
    fn from(u: Unsupported) -> Error {
        Error::Unsupported(u)
    }
}

/// A typed refusal of valid AIR: which construct ([`Feature`]), where, and the symbol it names.
/// The front end refuses only what this version does not implement, on every device; the
/// translator reports it as `Needs::NotImplemented`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unsupported {
    /// The refused construct.
    pub feature: Feature,
    /// Where in the input it was found (`None`: not located).
    pub location: Option<Location>,
    /// The symbol the refused construct names, for diagnostics only: an unknown intrinsic, or the
    /// callee of a call that needs linking. Opaque like every [`DiagName`].
    pub symbol: Option<DiagName>,
}

impl Unsupported {
    /// `feature`, not implemented by this version.
    pub fn not_implemented(feature: Feature) -> Unsupported {
        Unsupported {
            feature,
            location: None,
            symbol: None,
        }
    }

    /// Attach the input location.
    pub fn at(mut self, location: Location) -> Unsupported {
        self.location = Some(location);
        self
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
        if let Some(at) = &self.location {
            write!(f, " at {at}")?;
        }
        Ok(())
    }
}

/// An AIR construct the front end refuses ([`Unsupported`]). Each is also a feature of the
/// translator's own refusals, under the same name.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Feature {
    // ---- Input constructs outside the typed model ----
    /// A metallib that is not an executable library (dynamic library, companion).
    LibraryKind(u8),
    /// A function stage the model does not cover.
    Stage(u8),
    /// An LLVM address space the typed model does not define (it defines AIR's `thread` 0
    /// through `threadgroup_imageblock` 4). The 5 and 6 of Apple's AGX ray-tracing and
    /// tessellation library modules have no established meaning and stay refused.
    AddressSpace(u32),
    /// A module-level record outside the accepted subset (alias, ifunc, comdat, …).
    ModuleRecord {
        /// The module block's record code.
        code: u32,
    },
    /// An instruction record outside the closed instruction set.
    Instruction {
        /// The function block's record code.
        code: u32,
    },
    /// A type outside the typed model.
    Type {
        /// The type block's record code.
        code: u32,
    },
    /// A calling convention other than AIR's.
    CallingConvention(u32),
    /// An intrinsic outside the closed set.
    UnknownIntrinsic,
    /// A static initializer that cannot be evaluated.
    StaticInitializer,
    /// Recursion (one variant).
    Recursion,
    /// Indirect calls, function pointers, visible/intersection function tables (one variant).
    DynamicCall,
    /// Linkage signatures / stitching / dynamic libraries.
    Linking,
    /// A bitcode constant record outside the typed model (block address, inline asm, `select`
    /// and other constant expressions the IR has no form for).
    Constant {
        /// The constants block's record code.
        code: u32,
    },
    /// An LLVM attribute kind that changes the calling ABI and that AIR does not use
    /// (`inalloca`, `preallocated`, `byref`, `nest`, the `swift*` kinds).
    Attribute {
        /// The LLVM attribute kind number.
        kind: u32,
    },
    /// A bitcode record LLVM defines in `block` that the typed model does not cover and that has
    /// no more specific variant (legacy spellings predating module version 2, such as the old
    /// metadata node and string records and pre-group attribute entries).
    BitcodeRecord {
        /// The block's id.
        block: u32,
        /// The record's code within the block.
        code: u32,
    },
    /// An atomic access or fence in a synchronization scope other than LLVM's built-in
    /// `singlethread` and system scopes.
    SyncScope,
    /// A call to a defined function the inliner cannot take: a variadic callee, a `byval`
    /// argument, whose private copy inlining would have to materialize, or a call at another
    /// function type whose arguments match the parameters (another return type, or identically
    /// shaped structs of other names). A call at a function type whose arguments differ from the
    /// parameters in count or shape is `InvalidInput(Canon(CallTypeMismatch))`.
    DirectCall,
    /// An observed read of a pointer entry argument that a function-constant predicate removes,
    /// other than an absent buffer (which reads as null) or an absent texture or sampler read
    /// only as the texture or sampler operand of texture calls (an absent read,
    /// `intrinsics::FunctionConstant::AbsentRead`): Metal defines no value for it.
    FunctionConstantArgument,
    /// A ray intersection through intersection functions: an intersection function table bound
    /// or called.
    IntersectionFunctionTable,
    // ---- the refused families, not implemented yet ----
    /// A post-tessellation vertex function's control-point input used other than through its
    /// fetch function, or that function called other than on the input.
    Tessellation,
    /// A ray tracing intrinsic that does not decode: an intersector whose tags the grammar
    /// rejects, or an `intersection_query` operation.
    RayTracing,
    /// A `matmul2d` or tensor form the canonical expansion does not model.
    Tensors,
    /// An undecoded indirect-command-buffer encoding intrinsic.
    GpuEncodedIndirectCommands,
    /// A multisample imageblock's operation, which does not decode: its sample count, the colors
    /// its pixel holds or a color's coverage mask, or a store to the samples of a coverage mask.
    /// Tile functions and fragment imageblocks run on single-sample passes only.
    Imageblocks,
    /// A program-scope buffer binding of MSL 3.2 (macOS 15 and later), `constant T* constant p
    /// [[buffer(n)]];` in Apple's `!air.global_bindings` list, valid AIR this version does not
    /// implement yet.
    ProgramScopeBinding,
}

/// An input the request did not supply.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MissingInput {
    /// A value use of an unsupplied function constant survived SCCP.
    FunctionConstant {
        /// The constant's index.
        index: u32,
    },
    /// A call through a visible-function reference (`!air.visible_function_references`)
    /// survived specialization, and no linked function has the name it references. `symbol` is that
    /// name.
    LinkedFunction {
        /// The symbol the reference names.
        symbol: DiagName,
    },
    /// A call to a function, or a use of a variable, that the module imports from a dynamic
    /// library (a declaration its `air.dyld_flat_table` names) survived specialization, and no
    /// dynamic library the stage links exports it (`MTLDynamicLibrary`).
    DynamicSymbol {
        /// The import's symbol.
        symbol: DiagName,
    },
    /// An import's use survived specialization while a dynamic library an install name names
    /// was not supplied (it could not be loaded): Metal's creation fails then, whatever the
    /// other libraries export.
    DynamicLibrary {
        /// The first install name of the search with no library.
        install_name: DiagName,
    },
}

/// The request is inconsistent with the library: its entry, its function-constant values, or a
/// linked function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequestError {
    /// The requested entry point is not in the library.
    EntryNotFound {
        /// The entry name the request gave.
        requested: DiagName,
        /// How many entry points the library declares.
        available: u32,
    },
    /// More than one entry matches the request.
    AmbiguousEntry {
        /// The entry name the request gave.
        requested: DiagName,
    },
    /// A function-constant index that the function does not declare, in a request made to the
    /// canonicalizer directly (a `SpecRequest`, a linked module's constants).
    FunctionConstantIndex {
        /// The index the request names.
        index: u32,
    },
    /// A function-constant value of the wrong type.
    FunctionConstantType {
        /// The constant's index.
        index: u32,
    },
    /// The same function constant supplied twice (by index, or by index and by name).
    DuplicateFunctionConstant {
        /// The constant's index.
        index: u32,
    },
    /// A raw Metal enum value outside the known set.
    UnknownEnumValue {
        /// The Metal enum's type name (for example `MTLDataType`).
        ty: &'static str,
        /// The raw value given.
        value: u32,
    },
    /// A linked function that cannot resolve the visible-function reference naming it. `name` is
    /// the function's name.
    LinkedFunction {
        /// The linked function's name.
        name: DiagName,
        /// Why it cannot resolve the reference.
        problem: LinkProblem,
    },
    /// Two color outputs of the function that remain under its function-constant values write
    /// color attachment `color` at blend index `index` (`[[color(color), index(index)]]`): Metal
    /// refuses the specialization when it makes the function.
    DuplicateColorOutput {
        /// The color attachment both write.
        color: u32,
        /// The blend index both write.
        index: u32,
    },
}

/// Why a linked function cannot resolve a visible-function reference.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LinkProblem {
    /// Two different linked functions have this name.
    Ambiguous,
    /// The function is not a visible function of its module (`!air.visible`).
    NotVisible,
    /// The reference's function type is not the linked function's.
    Signature,
    /// The linked function's module has another data layout than the module that references it.
    DataLayout,
}

/// A deterministic structural or accounting cap. Values live in `budget`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Limit {
    /// Instructions after inlining.
    InlinedInstructions,
    /// Structurizer output blocks, `4(n+e)+1024`.
    OutputBlocks,
    /// Irreducible nesting depth.
    IrreducibleDepth,
    /// Emitted SPIR-V words.
    SpirvWords,
    /// Deterministic work units of one attempt.
    WorkUnits,
    /// Ledger bytes of one attempt.
    LedgerBytes,
    /// Types in the bitcode type table.
    TypeCount,
    /// Metadata nodes.
    MetadataNodes,
    /// Bitstream block nesting.
    BitstreamNesting,
    /// Instructions of the decoded module.
    InstructionCount,
    /// Nesting depth of `air.struct_type_info` layouts and the argument-buffer members inside
    /// them.
    LayoutNesting,
    /// Nesting depth of structured constructs the structurizer emits (P8 recurses once per
    /// level).
    ConstructNesting,
    /// Bytes of a function's private arena, whose pointers are `u32` byte offsets.
    PrivateArenaBytes,
    /// Bytes of a kernel's call stack: its `maxCallStackDepth` times the frame of its recursion
    /// through visible functions, as Metal reserves it, passes the stack a thread has on Metal,
    /// where pipeline creation fails with "Compute pipeline exceeds available stack space".
    CallStackBytes,
}

/// Wall-clock budget outcomes. Transient.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BudgetExceeded {
    /// The attempt's deadline passed.
    Time,
}

/// An internal invariant failure. Always a bug, but for a host allocation that failed after the
/// ledger admitted it ([`InternalKind::AllocationFailed`]).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InternalError {
    /// Which invariant failed.
    pub kind: InternalKind,
    /// A human-readable description, for diagnostics.
    pub detail: String,
}

impl InternalError {
    /// An internal error of `kind` with a human-readable detail.
    pub fn new(kind: InternalKind, detail: impl Into<String>) -> InternalError {
        InternalError {
            kind,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for InternalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.detail)
    }
}

/// Which internal invariant failed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InternalKind {
    /// The typed builder rejected an instruction (`BuildError`).
    Build,
    /// The finalize verifier rejected the module.
    Verify,
    /// The module requires more than the profile provides.
    RequirementsExceedProfile,
    /// A panic was caught at a public entry point.
    Panic,
    /// The host allocator failed after the ledger admitted the reservation. Transient: it
    /// depends on the host's memory, not on the input.
    AllocationFailed,
    /// Another invariant, named in the detail.
    Invariant,
}

#[cfg(test)]
mod tests {
    use super::{Attachment, DiagName, Location};

    /// A location tied to an attachment keeps its place and names the attachment after it; the
    /// render pipeline's descriptor is a place of its own.
    #[test]
    fn a_location_names_the_attachment_it_is_tied_to() {
        let entry = Location::metadata(Some(DiagName::new("frag")), None, None);
        assert_eq!(entry.attachment(), None);
        let tied = entry.clone().with_attachment(Attachment::Color(3));
        assert_eq!(tied.attachment(), Some(Attachment::Color(3)));
        assert_eq!(tied.in_metadata(), entry.in_metadata());
        assert_ne!(tied, entry);
        assert_eq!(tied.to_string(), format!("{entry}, color attachment 3"));
        assert_eq!(format!("{tied:?}"), format!("{entry:?} at Color(3)"));
        let pipeline = Location::render_pipeline().with_attachment(Attachment::Depth);
        assert!(pipeline.is_render_pipeline() && !tied.is_render_pipeline());
        assert_eq!(pipeline.to_string(), "render pipeline, depth attachment");
        let stencil = Location::render_pipeline().with_attachment(Attachment::Stencil);
        assert_eq!(stencil.to_string(), "render pipeline, stencil attachment");
    }
}
