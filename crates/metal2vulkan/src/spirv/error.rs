//! `BuildError`: the typed builder's rejection of an ill-formed construction.
//!
//! A correct lowering never triggers a `BuildError`; it is an internal-bug class. It is typed rather
//! than a panic so that a lowering bug becomes an honest translation failure
//! (`Error::Internal(Build)`) instead of an invalid module.

use std::fmt;

use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{ResourceSlot, SpecKey, Symbol, VaryingKey};
use crate::spirv::ids::{BlockId, ConstId, FuncId, GlobalId, TypeId, Value, VarId};
use crate::spirv::ir::{GlslOp, ImageOperandKind, IoDir, StorageClass};

/// The builder's result type.
pub type BResult<T> = Result<T, BuildError>;

/// Why the builder refused a construction.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuildError {
    /// An id that this builder did not produce.
    Dangling(Dangling),
    /// A dense id space (types, constants, values, …) would exceed `u32`.
    IdSpaceExhausted,
    /// A type that violates a typing rule.
    Type(TypeRule),
    /// An explicit layout that violates its layout rule, at member `member` of the struct being
    /// interned (or `None` for an array stride).
    Layout {
        member: Option<u32>,
        violation: LayoutViolation,
    },
    /// A constant that violates a typing rule.
    Constant(ConstRule),
    /// A module-scope variable that violates a rule.
    Global(GlobalRule),
    /// An entry point or execution mode that violates a rule.
    Entry(EntryRule),
    /// An instruction operand that violates the opcode's typing rule.
    Operand {
        op: crate::spv::Op,
        rule: OperandRule,
    },
    /// A `GLSL.std.450` operand that violates the instruction's typing rule.
    ExtOperand { op: GlslOp, rule: OperandRule },
    /// A control-flow or function-structure rule.
    Control(ControlRule),
    /// A relocation that cannot be applied.
    Reloc(RelocRule),
    /// An instruction of more than 0xFFFF words, which the SPIR-V word-count field cannot encode
    /// (the serializer refuses it rather than truncating).
    InstructionTooLong { op: crate::spv::Op, words: u64 },
}

impl From<Dangling> for BuildError {
    fn from(d: Dangling) -> BuildError {
        BuildError::Dangling(d)
    }
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SPIR-V builder rejected construction: {self:?}")
    }
}

impl std::error::Error for BuildError {}

impl From<BuildError> for Error {
    fn from(e: BuildError) -> Error {
        Error::Internal(InternalError::new(InternalKind::Build, e.to_string()))
    }
}

/// An id outside the builder's arenas.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dangling {
    Type(TypeId),
    Const(ConstId),
    Value(Value),
    Global(GlobalId),
    Func(FuncId),
    Block(BlockId),
    Var(VarId),
}

/// A violated type rule.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeRule {
    /// Integer width outside 8/16/32/64 or float width outside 16/32/64.
    ScalarWidth,
    /// Vector length outside 2..=4, or a non-scalar element.
    VectorShape,
    /// Matrix column count outside 2..=4, or a column that is not a float vector.
    MatrixShape,
    /// An array length that is not a positive integer constant (or integer spec constant).
    ArrayLength,
    /// An aggregate element or member that cannot be one (void, function, runtime array, …).
    Element,
    /// A struct with no members.
    EmptyStruct,
    /// A runtime array that is not the last member of a `Block` struct.
    RuntimeArrayPlacement,
    /// Explicit-layout and implicit-layout aggregates mixed in one aggregate.
    MixedLayout,
    /// A `Block` struct nested inside another aggregate other than a descriptor array.
    NestedBlock,
    /// `bool` (or an opaque type) inside an explicitly laid-out aggregate.
    NotLayoutable,
    /// The pointee is not allowed in the pointer's storage class.
    PointeeForClass(StorageClass),
    /// A function type with a void parameter or a function-typed return.
    FunctionSignature,
    /// An image type whose operands are inconsistent.
    ImageShape,
    /// A sampled image whose image is not a sampled, non-buffer image.
    SampledImageShape,
}

/// A violated explicit-layout rule.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutViolation {
    /// The layout does not list exactly one offset per member.
    MemberCount,
    /// The offset is not a multiple of the member's alignment under the rule.
    Misaligned { offset: u32, align: u32 },
    /// The member starts before the end (and required padding) of the previous one.
    Overlap { offset: u32, min: u32 },
    /// A matrix member without a matrix stride, or a non-matrix member with one.
    MatrixStridePresence,
    /// A matrix stride that is too small or misaligned for the column.
    MatrixStride { stride: u32 },
    /// An array stride that is too small or misaligned for the element.
    ArrayStride { stride: u32 },
    /// A nested laid-out aggregate follows a less strict rule than its container.
    NestedRule,
    /// The size of an element is not known (a spec-constant-sized array).
    UnsizedElement,
    /// The layout arithmetic overflowed `u32`.
    Overflow,
}

/// A violated constant rule.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstRule {
    /// A scalar constant whose type is not an integer or float scalar.
    ScalarType,
    /// Bits set above the scalar's width.
    BitsExceedWidth,
    /// A composite whose type is not a vector, matrix, sized array, or struct.
    CompositeType,
    /// A composite whose parts do not match the type's members.
    CompositeParts,
    /// A specialization constant used as a part of a plain composite constant.
    SpecInComposite,
    /// `OpConstantNull` of a type that does not allow it.
    NullType,
    /// A specialization constant whose type is not a bool, integer, or float scalar, or whose
    /// default does not fit it.
    SpecType,
    /// The same specialization key declared twice with different type or default.
    SpecKeyConflict(SpecKey),
    /// An `OpSpecConstantOp` whose operands are not scalar constants of the types its operation
    /// requires, or none of which is specialization-dependent.
    SpecOpOperands,
}

/// A violated module-scope variable rule.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GlobalRule {
    /// A resource declared in a storage class resources cannot use.
    ResourceClass(StorageClass),
    /// A resource whose type does not fit its storage class (`Block` struct, descriptor array,
    /// opaque type).
    ResourceType,
    /// The same resource slot declared twice.
    DuplicateResource(ResourceSlot),
    /// The same varying declared twice in one direction.
    DuplicateVarying(VaryingKey, IoDir),
    /// A user varying of a type the interface cannot carry.
    VaryingType,
    /// The same fixed `Location` declared twice in one direction.
    DuplicateLocation(u32, IoDir),
    /// A builtin not in the supported set.
    UnsupportedBuiltIn(crate::spv::BuiltIn),
    /// A builtin used in a direction it does not have.
    BuiltInDirection(crate::spv::BuiltIn, IoDir),
    /// A builtin declared with the wrong type.
    BuiltInType(crate::spv::BuiltIn),
    /// A push member that overlaps the previous one or is placed out of order.
    PushOrder { offset: u32 },
    /// A push member added after the push block was sealed.
    PushSealed,
    /// The push block was requested but has no members.
    PushEmpty,
    /// A plain global in a storage class other than `Private` or `Workgroup`.
    PlainClass(StorageClass),
    /// An initializer whose type differs from the variable's.
    InitializerType,
    /// `Invariant` on a variable other than the `Position`, `ClipDistance`, `FragDepth`,
    /// `SampleMask` and `FragStencilRefEXT` outputs.
    InvariantTarget,
    /// `PerPrimitiveEXT` on a variable other than a user varying.
    PerPrimitiveTarget,
    /// More than one `TaskPayloadWorkgroupEXT` variable in an entry's interface, or an
    /// `OpEmitMeshTasksEXT` payload that is not one.
    Payload,
    /// `Coherent` resources in a module under the Vulkan memory model, which bans the decoration,
    /// or `Coherent` on a variable other than a storage-image resource.
    CoherentTarget,
    /// `Volatile` on a variable other than a storage-image resource or a descriptor array of
    /// them, or in a module under the Vulkan memory model, which bans the decoration.
    VolatileTarget,
    /// An enabling capability that is not one of the builtin's alternatives
    /// (`requirements::builtin_enablers`), a second, different choice for one builtin, or a
    /// choice for a variable that is not a builtin (`None`).
    BuiltInEnabler(Option<crate::spv::BuiltIn>),
}

/// A violated entry-point rule.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryRule {
    /// A second entry point: modules are compiled per function.
    Duplicate,
    /// No entry point was declared before finalize.
    Missing,
    /// The entry function has parameters or a non-void return.
    Signature,
    /// An execution mode that the execution model does not allow.
    ModeForModel,
    /// Two modes of one exclusive group (depth, interlock, derivative group, local size), or the
    /// same mode twice.
    ConflictingModes,
    /// A compute entry without a local size.
    MissingLocalSize,
    /// A local size of zero, or a `LocalSizeId` operand that is not a 32-bit integer constant.
    LocalSize,
    /// A float-controls width outside 16/32/64.
    FloatWidth,
    /// A tessellation evaluation entry without its patch domain, or a tessellation control entry
    /// without an output patch size of 1 to 32.
    TessellationModes,
    /// A mesh entry without a nonzero vertex and primitive count and its primitive kind.
    MeshModes,
    /// An instruction the entry's execution model (and modes) cannot run: implicit derivatives
    /// outside fragment (or derivative-group compute), fragment-only operations elsewhere, or
    /// interlock without an interlock mode.
    InstructionForModel(crate::spv::Op),
}

/// A violated instruction operand rule.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandRule {
    /// The operand must be a pointer.
    NotPointer,
    /// Stores are not allowed through this storage class.
    ReadOnlyClass(StorageClass),
    /// The stored value's type differs from the pointee.
    StoredType,
    /// Opaque values cannot be stored.
    Opaque,
    /// `PhysicalStorageBuffer` accesses need an `Aligned` operand; alignments must be powers of two.
    Alignment,
    /// Operand `n` (0-based, in the constructor's argument order) is not of the type category the
    /// opcode needs (integer, float, bool, vector, pointer, …).
    Category(u8),
    /// Operand `n`'s type does not match the type the opcode requires from the other operands
    /// (same type, component count, or element type).
    Mismatch(u8),
    /// Operand `n` has a component width the opcode does not accept.
    Width(u8),
    /// Operand `n` must be a constant.
    NotConstant(u8),
    /// Operand `n` is a constant (or literal) out of the opcode's range.
    Range(u8),
    /// The number of operands, arguments, parts, or indices is wrong.
    Arity,
    /// The requested result type does not fit the opcode.
    ResultType,
    /// The pointer's storage class does not allow this opcode (e.g. loads through an
    /// `Image`-class texel pointer, which is valid only as an atomic operand).
    Class(StorageClass),
    /// A scope the opcode does not allow.
    Scope,
    /// Memory semantics the opcode does not allow.
    Semantics,
    /// An image-instruction rule.
    Image(ImageRule),
    /// A sampled image consumed outside the block of the `OpSampledImage` that created it, or one
    /// that does not come from an `OpSampledImage`.
    SampledImageBlock,
    /// A call to a function whose signature does not match the arguments.
    Callee,
    /// `NoContraction` requested on an opcode that is not float arithmetic, or on a `GLSL.std.450`
    /// instruction other than `Fma`.
    NoContraction,
    /// An atomic's pointer is not a place: a variable, an access chain, or a texel pointer. A
    /// device address becomes one through `FunctionBuilder::device_place`.
    NotPlace,
    /// Operand `n` carries a non-uniform descriptor element where it would lose its `NonUniform`
    /// decoration (a call argument: the callee's parameter is not decorated).
    NonUniform(u8),
}

/// A violated image-instruction rule.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageRule {
    /// The image type (dimensionality, usage, multisampling, depth, arrayed, format) does not
    /// allow this opcode.
    ImageType,
    /// The coordinate has the wrong component type or count.
    Coordinate,
    /// An image operand the opcode or image does not allow.
    NotAllowed(ImageOperandKind),
    /// A required image operand is missing.
    Missing(ImageOperandKind),
    /// An image operand of the wrong type, or a `ConstOffset` that is not a constant.
    OperandType(ImageOperandKind),
    /// A `Dref`, gather component, or texel operand of the wrong type or value.
    Value,
}

/// Why `relocate` refused a module.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RelocRule {
    /// The link table assigns no number to this symbol.
    MissingSymbol(Symbol),
    /// A relocation offset outside the module's words.
    Offset(u32),
}

/// A violated control-flow or function-structure rule.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlRule {
    /// An instruction or terminator was added with no open block (after a terminator).
    NoCurrentBlock,
    /// The returned value does not match the function's return type.
    ReturnType,
    /// A function was finished with an open (unterminated) block.
    UnterminatedBlock(BlockId),
    /// The module was finalized while a function was still being built.
    UnfinishedFunction(FuncId),
    /// A block of another function.
    ForeignBlock(BlockId),
    /// A local variable of a type `Function` storage cannot hold.
    LocalType,
    /// `break_`/`continue_` of a loop that is not the innermost loop open in this function
    /// (SPIR-V structured exits reach only the innermost loop's merge or continue target).
    NotInnermostLoop,
    /// A construct arm returned `ArmEnd::FallThrough` with no open block, or `ArmEnd::Exits` with
    /// its block still open.
    ArmEnd,
    /// The arms that fall through to a merge disagree on the number or types of their values, or
    /// an `if_` without an else arm falls through with values (the header edge has none).
    ArmValues,
    /// A merged value of a type an `OpPhi` cannot carry (logical pointers, opaque or
    /// runtime-sized values).
    PhiType,
    /// A switch case with no literal, or a literal listed twice.
    CaseLiterals,
    /// A loop's `exit_sel` is a variable of another function, or not of a 32-bit integer type.
    /// (An id that names no variable is `Dangling::Var`.)
    ExitSelector,
    /// `OpKill`/`OpTerminateInvocation` outside the entry function while the entry declares an
    /// interlock mode.
    KillOutsideEntry(FuncId),
    /// A tangled instruction inside a switch-at-use.
    TangledInSwitchAtUse,
    /// A switch-at-use calls a function that transitively executes a tangled instruction.
    TangledCallInSwitchAtUse(FuncId),
    /// A switch too large to split into `OpSwitch`es within SPIR-V's limit of `(literal, label)`
    /// pairs: more chunks of arms than one `OpSwitch` carries, which no module within the word
    /// budget reaches.
    SwitchSplit,
}
