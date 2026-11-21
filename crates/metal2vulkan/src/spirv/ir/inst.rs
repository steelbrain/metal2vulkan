//! Typed operand vocabulary of the instruction families.
//!
//! [`Inst`](super::Inst) groups opcodes by family (`Binary { op: BinOp, .. }`, `Atomic*`,
//! `Image*`, `Group`, `Glsl`, …). Each family enum below is closed, and each variant names exactly
//! one SPIR-V opcode (or `GLSL.std.450` instruction), so the set of emitted opcodes stays a closed,
//! typed list with no generic operand-list escape hatch. The typing rule of every variant is
//! enforced by its constructor in `builder/ops/`.

use crate::spirv::ids::{ConstId, Value};

/// Unary operations whose operand and result types are checked by their family constructor.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnOp {
    SNegate,
    FNegate,
    Not,
    LogicalNot,
    BitReverse,
    BitCount,
    IsNan,
    IsInf,
    Any,
    All,
    CopyObject,
    QuantizeToF16,
}

impl UnOp {
    /// The SPIR-V opcode.
    pub fn to_spirv(self) -> crate::spv::Op {
        use crate::spv::Op;
        match self {
            UnOp::SNegate => Op::SNegate,
            UnOp::FNegate => Op::FNegate,
            UnOp::Not => Op::Not,
            UnOp::LogicalNot => Op::LogicalNot,
            UnOp::BitReverse => Op::BitReverse,
            UnOp::BitCount => Op::BitCount,
            UnOp::IsNan => Op::IsNan,
            UnOp::IsInf => Op::IsInf,
            UnOp::Any => Op::Any,
            UnOp::All => Op::All,
            UnOp::CopyObject => Op::CopyObject,
            UnOp::QuantizeToF16 => Op::QuantizeToF16,
        }
    }
}

/// Binary operations: arithmetic, bitwise, shifts, comparisons, extended arithmetic, and linear
/// algebra.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    IAdd,
    ISub,
    IMul,
    UDiv,
    SDiv,
    UMod,
    SRem,
    SMod,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
    ShiftRightLogical,
    ShiftRightArithmetic,
    ShiftLeftLogical,
    BitwiseOr,
    BitwiseXor,
    BitwiseAnd,
    LogicalEqual,
    LogicalNotEqual,
    LogicalOr,
    LogicalAnd,
    IEqual,
    INotEqual,
    UGreaterThan,
    SGreaterThan,
    UGreaterThanEqual,
    SGreaterThanEqual,
    ULessThan,
    SLessThan,
    ULessThanEqual,
    SLessThanEqual,
    FOrdEqual,
    FUnordEqual,
    FOrdNotEqual,
    FUnordNotEqual,
    FOrdLessThan,
    FUnordLessThan,
    FOrdGreaterThan,
    FUnordGreaterThan,
    FOrdLessThanEqual,
    FUnordLessThanEqual,
    FOrdGreaterThanEqual,
    FUnordGreaterThanEqual,
    /// Result: `struct { T, T }` (sum, carry).
    IAddCarry,
    /// Result: `struct { T, T }` (difference, borrow).
    ISubBorrow,
    /// Result: `struct { T, T }` (low, high).
    UMulExtended,
    /// Result: `struct { T, T }` (low, high).
    SMulExtended,
    VectorTimesScalar,
    Dot,
}

/// Whether a float arithmetic result may be fused with other operations by the device
/// (SPIR-V `NoContraction`). Vulkan lets an implementation contract an `OpFMul` feeding an
/// `OpFAdd`/`OpFSub` into one fused operation unless one of them is decorated `NoContraction`;
/// LLVM allows that only when both carry the `contract` fast-math flag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Contraction {
    /// The device may contract the operation (no decoration). Every non-float operation.
    #[default]
    Allowed,
    /// `OpDecorate %result NoContraction`: the operation is rounded on its own. Only float
    /// arithmetic ([`BinOp::allows_no_contraction`]) and the `GLSL.std.450` `Fma` a multiply
    /// rounded on its own becomes carry it.
    Forbidden,
}

impl BinOp {
    /// Whether `NoContraction` applies to the operation: the float arithmetic opcodes.
    pub fn allows_no_contraction(self) -> bool {
        matches!(
            self,
            BinOp::FAdd
                | BinOp::FSub
                | BinOp::FMul
                | BinOp::FDiv
                | BinOp::FRem
                | BinOp::VectorTimesScalar
        )
    }

    /// The SPIR-V opcode.
    pub fn to_spirv(self) -> crate::spv::Op {
        use crate::spv::Op;
        match self {
            BinOp::IAdd => Op::IAdd,
            BinOp::ISub => Op::ISub,
            BinOp::IMul => Op::IMul,
            BinOp::UDiv => Op::UDiv,
            BinOp::SDiv => Op::SDiv,
            BinOp::UMod => Op::UMod,
            BinOp::SRem => Op::SRem,
            BinOp::SMod => Op::SMod,
            BinOp::FAdd => Op::FAdd,
            BinOp::FSub => Op::FSub,
            BinOp::FMul => Op::FMul,
            BinOp::FDiv => Op::FDiv,
            BinOp::FRem => Op::FRem,
            BinOp::ShiftRightLogical => Op::ShiftRightLogical,
            BinOp::ShiftRightArithmetic => Op::ShiftRightArithmetic,
            BinOp::ShiftLeftLogical => Op::ShiftLeftLogical,
            BinOp::BitwiseOr => Op::BitwiseOr,
            BinOp::BitwiseXor => Op::BitwiseXor,
            BinOp::BitwiseAnd => Op::BitwiseAnd,
            BinOp::LogicalEqual => Op::LogicalEqual,
            BinOp::LogicalNotEqual => Op::LogicalNotEqual,
            BinOp::LogicalOr => Op::LogicalOr,
            BinOp::LogicalAnd => Op::LogicalAnd,
            BinOp::IEqual => Op::IEqual,
            BinOp::INotEqual => Op::INotEqual,
            BinOp::UGreaterThan => Op::UGreaterThan,
            BinOp::SGreaterThan => Op::SGreaterThan,
            BinOp::UGreaterThanEqual => Op::UGreaterThanEqual,
            BinOp::SGreaterThanEqual => Op::SGreaterThanEqual,
            BinOp::ULessThan => Op::ULessThan,
            BinOp::SLessThan => Op::SLessThan,
            BinOp::ULessThanEqual => Op::ULessThanEqual,
            BinOp::SLessThanEqual => Op::SLessThanEqual,
            BinOp::FOrdEqual => Op::FOrdEqual,
            BinOp::FUnordEqual => Op::FUnordEqual,
            BinOp::FOrdNotEqual => Op::FOrdNotEqual,
            BinOp::FUnordNotEqual => Op::FUnordNotEqual,
            BinOp::FOrdLessThan => Op::FOrdLessThan,
            BinOp::FUnordLessThan => Op::FUnordLessThan,
            BinOp::FOrdGreaterThan => Op::FOrdGreaterThan,
            BinOp::FUnordGreaterThan => Op::FUnordGreaterThan,
            BinOp::FOrdLessThanEqual => Op::FOrdLessThanEqual,
            BinOp::FUnordLessThanEqual => Op::FUnordLessThanEqual,
            BinOp::FOrdGreaterThanEqual => Op::FOrdGreaterThanEqual,
            BinOp::FUnordGreaterThanEqual => Op::FUnordGreaterThanEqual,
            BinOp::IAddCarry => Op::IAddCarry,
            BinOp::ISubBorrow => Op::ISubBorrow,
            BinOp::UMulExtended => Op::UMulExtended,
            BinOp::SMulExtended => Op::SMulExtended,
            BinOp::VectorTimesScalar => Op::VectorTimesScalar,
            BinOp::Dot => Op::Dot,
        }
    }
}

/// Numeric conversions and pointer/integer conversions. The result type is part of the value.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConvOp {
    ConvertFToU,
    ConvertFToS,
    ConvertSToF,
    ConvertUToF,
    UConvert,
    SConvert,
    FConvert,
    /// Integer to `PhysicalStorageBuffer` pointer.
    ConvertUToPtr,
    Bitcast,
}

impl ConvOp {
    /// The SPIR-V opcode.
    pub fn to_spirv(self) -> crate::spv::Op {
        use crate::spv::Op;
        match self {
            ConvOp::ConvertFToU => Op::ConvertFToU,
            ConvOp::ConvertFToS => Op::ConvertFToS,
            ConvOp::ConvertSToF => Op::ConvertSToF,
            ConvOp::ConvertUToF => Op::ConvertUToF,
            ConvOp::UConvert => Op::UConvert,
            ConvOp::SConvert => Op::SConvert,
            ConvOp::FConvert => Op::FConvert,
            ConvOp::ConvertUToPtr => Op::ConvertUToPtr,
            ConvOp::Bitcast => Op::Bitcast,
        }
    }
}

/// A memory or execution scope. Vulkan admits no other scope for the operations the core emits.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scope {
    Device,
    Workgroup,
    Subgroup,
    Invocation,
}

impl Scope {
    /// The SPIR-V enumerant.
    pub fn to_spirv(self) -> crate::spv::Scope {
        match self {
            Scope::Device => crate::spv::Scope::Device,
            Scope::Workgroup => crate::spv::Scope::Workgroup,
            Scope::Subgroup => crate::spv::Scope::Subgroup,
            Scope::Invocation => crate::spv::Scope::Invocation,
        }
    }
}

/// The ordering part of a memory-semantics operand. `SequentiallyConsistent` is not offered:
/// Metal exposes relaxed atomics only, and Vulkan treats it as acquire-release anyway.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MemoryOrder {
    #[default]
    Relaxed,
    Acquire,
    Release,
    AcquireRelease,
}

/// A memory-semantics operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Semantics {
    pub order: MemoryOrder,
    /// `UniformMemory`: `StorageBuffer`, `Uniform`, and `PhysicalStorageBuffer` memory.
    pub uniform_memory: bool,
    /// `WorkgroupMemory`.
    pub workgroup_memory: bool,
    /// `ImageMemory`.
    pub image_memory: bool,
    /// `MakeAvailable` (Vulkan memory model; needs a release ordering).
    pub make_available: bool,
    /// `MakeVisible` (Vulkan memory model; needs an acquire ordering).
    pub make_visible: bool,
    /// `Volatile` (atomics only).
    pub volatile: bool,
}

impl Semantics {
    /// Relaxed, with no storage classes.
    pub const RELAXED: Semantics = Semantics {
        order: MemoryOrder::Relaxed,
        uniform_memory: false,
        workgroup_memory: false,
        image_memory: false,
        make_available: false,
        make_visible: false,
        volatile: false,
    };

    /// The operand's bits.
    pub fn bits(self) -> u32 {
        let order = match self.order {
            MemoryOrder::Relaxed => 0,
            MemoryOrder::Acquire => 0x2,
            MemoryOrder::Release => 0x4,
            MemoryOrder::AcquireRelease => 0x8,
        };
        let flag = |on: bool, bit: u32| if on { bit } else { 0 };
        order
            | flag(self.uniform_memory, 0x40)
            | flag(self.workgroup_memory, 0x100)
            | flag(self.image_memory, 0x800)
            | flag(self.make_available, 0x2000)
            | flag(self.make_visible, 0x4000)
            | flag(self.volatile, 0x8000)
    }

    /// Whether any storage class is named.
    pub fn has_storage(self) -> bool {
        self.uniform_memory || self.workgroup_memory || self.image_memory
    }
}

/// A scope operand: the typed scope and the 32-bit integer constant that carries it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScopeId {
    pub scope: Scope,
    pub id: ConstId,
}

/// A memory-semantics operand: the typed semantics and the 32-bit integer constant that carries
/// them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SemanticsId {
    pub sem: Semantics,
    pub id: ConstId,
}

/// Read-modify-write atomics with one value operand.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomicOp {
    Exchange,
    IAdd,
    ISub,
    SMin,
    UMin,
    SMax,
    UMax,
    And,
    Or,
    Xor,
    /// `OpAtomicFAddEXT` (`SPV_EXT_shader_atomic_float_add`).
    FAdd,
}

impl AtomicOp {
    /// The SPIR-V opcode.
    pub fn to_spirv(self) -> crate::spv::Op {
        use crate::spv::Op;
        match self {
            AtomicOp::Exchange => Op::AtomicExchange,
            AtomicOp::IAdd => Op::AtomicIAdd,
            AtomicOp::ISub => Op::AtomicISub,
            AtomicOp::SMin => Op::AtomicSMin,
            AtomicOp::UMin => Op::AtomicUMin,
            AtomicOp::SMax => Op::AtomicSMax,
            AtomicOp::UMax => Op::AtomicUMax,
            AtomicOp::And => Op::AtomicAnd,
            AtomicOp::Or => Op::AtomicOr,
            AtomicOp::Xor => Op::AtomicXor,
            AtomicOp::FAdd => Op::AtomicFAddEXT,
        }
    }
}

/// Derivative operations. All need implicit derivatives (fragment, or compute with a derivative
/// group mode) and are tangled.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DerivOp {
    DPdx,
    DPdy,
    Fwidth,
    /// `OpDPdxFine`: the difference along x within the invocation's own row of its quad
    /// (`DerivativeControl`, which every Vulkan device has).
    DPdxFine,
    /// `OpDPdyFine`: the difference along y within the invocation's own column of its quad.
    DPdyFine,
    /// `OpFwidthFine`: `|DPdxFine| + |DPdyFine|`.
    FwidthFine,
}

impl DerivOp {
    /// The SPIR-V opcode.
    pub fn to_spirv(self) -> crate::spv::Op {
        use crate::spv::Op;
        match self {
            DerivOp::DPdx => Op::DPdx,
            DerivOp::DPdy => Op::DPdy,
            DerivOp::Fwidth => Op::Fwidth,
            DerivOp::DPdxFine => Op::DPdxFine,
            DerivOp::DPdyFine => Op::DPdyFine,
            DerivOp::FwidthFine => Op::FwidthFine,
        }
    }
}

/// The kind of an image operand, for error reporting.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageOperandKind {
    Bias,
    Lod,
    Grad,
    ConstOffset,
    Offset,
    Sample,
    MinLod,
    MakeTexelVisible,
    NonPrivateTexel,
    VolatileTexel,
}

/// Optional image operands. Each constructor admits only the operands its opcode and image allow.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct ImageOperands {
    /// Float scalar; implicit-LOD sampling only.
    pub bias: Option<Value>,
    /// Float scalar for explicit-LOD sampling, integer scalar for `OpImageFetch`.
    pub lod: Option<Value>,
    /// `(dx, dy)`: float vectors with one component per non-array coordinate.
    pub grad: Option<(Value, Value)>,
    /// A constant integer offset with one component per non-array coordinate.
    pub const_offset: Option<Value>,
    /// A non-constant offset (`ImageGatherExtended`): on gathers, and on samples, where Vulkan
    /// also needs `maintenance8` (VUID-RuntimeSpirv-Offset-10213).
    pub offset: Option<Value>,
    /// The sample index of a multisampled image.
    pub sample: Option<Value>,
    /// Float scalar; implicit-LOD sampling, or explicit sampling with `grad`.
    pub min_lod: Option<Value>,
}

/// The Vulkan memory model's texel operands of a storage-image access: an `OpImageRead`,
/// `OpImageSparseRead`, or `OpImageWrite`, the only instructions that carry one. They are kept out
/// of [`ImageOperands`], which every image instruction holds: an IR instruction is as large as its
/// largest form, so a field there grows every instruction. It is eight bytes, which keeps both
/// storage accesses no larger than the sampling instruction.
///
/// `volatile` is `VolatileTexel`: the access is performed, once, and a read never returns an
/// earlier write of the invocation that it did not perform (the Vulkan memory model's form of
/// the `Volatile` decoration, which that model bans).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexelMemory {
    /// No `NonPrivateTexel`: the access is ordered only within its invocation.
    Private { volatile: bool },
    /// `NonPrivateTexel`: the access takes part in inter-invocation ordering.
    NonPrivate { volatile: bool },
    /// `NonPrivateTexel` with `MakeTexelVisible` (a read) or `MakeTexelAvailable` (a write) at
    /// `scope`, carried by the constant `id` ([`TexelMemory::made`] is the [`ScopeId`]; the two
    /// are separate fields because a [`ScopeId`] field leaves no room for `volatile` in eight
    /// bytes). SPIR-V admits either operand only with `NonPrivateTexel`, and an access's opcode
    /// decides which of the two it is, so neither misuse is representable.
    Made {
        scope: Scope,
        id: ConstId,
        volatile: bool,
    },
}

impl Default for TexelMemory {
    fn default() -> Self {
        Self::PRIVATE
    }
}

impl TexelMemory {
    /// No texel operand: a plain storage access.
    pub const PRIVATE: Self = Self::Private { volatile: false };

    /// `NonPrivateTexel` with `MakeTexel*` at `scope`.
    pub fn made(scope: ScopeId, volatile: bool) -> Self {
        Self::Made {
            scope: scope.scope,
            id: scope.id,
            volatile,
        }
    }

    /// The scope of `MakeTexelAvailable` or `MakeTexelVisible`, if the access makes its texel
    /// available or visible.
    pub fn made_scope(self) -> Option<ScopeId> {
        match self {
            Self::Made { scope, id, .. } => Some(ScopeId { scope, id }),
            Self::Private { .. } | Self::NonPrivate { .. } => None,
        }
    }

    /// `NonPrivateTexel`.
    pub fn non_private(self) -> bool {
        !matches!(self, Self::Private { .. })
    }

    /// `VolatileTexel`.
    pub fn volatile(self) -> bool {
        match self {
            Self::Private { volatile }
            | Self::NonPrivate { volatile }
            | Self::Made { volatile, .. } => volatile,
        }
    }

    /// These operands with `VolatileTexel` set to `volatile`.
    pub fn with_volatile(self, volatile: bool) -> Self {
        match self {
            Self::Private { .. } => Self::Private { volatile },
            Self::NonPrivate { .. } => Self::NonPrivate { volatile },
            Self::Made { scope, id, .. } => Self::Made {
                scope,
                id,
                volatile,
            },
        }
    }

    /// Whether the operands need the Vulkan memory model (any of them does).
    pub fn vulkan_memory_model(self) -> bool {
        self != Self::PRIVATE
    }
}

/// The shape of an image-sampling opcode. The eight combinations without `sparse` are the eight
/// `OpImage[Sample|SampleProj][Dref]{Implicit,Explicit}Lod` opcodes; with `sparse` (never
/// projective: SPIR-V reserves `OpImageSparseSampleProj*`) the four
/// `OpImageSparseSample[Dref]{Implicit,Explicit}Lod` opcodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SampleForm {
    /// Implicit LOD (needs implicit derivatives; tangled).
    pub implicit: bool,
    /// Projective coordinates (one extra component).
    pub proj: bool,
    /// Depth comparison.
    pub dref: bool,
    /// The sparse form: the result is the residency code and the texel (`SparseResidency`).
    pub sparse: bool,
}

impl SampleForm {
    /// The SPIR-V opcode. The builder never makes a sparse projective form, whose opcodes SPIR-V
    /// reserves; it maps to the non-projective sparse opcode.
    pub fn to_spirv(self) -> crate::spv::Op {
        use crate::spv::Op;
        match (self.sparse, self.proj, self.dref, self.implicit) {
            (false, false, false, true) => Op::ImageSampleImplicitLod,
            (false, false, false, false) => Op::ImageSampleExplicitLod,
            (false, false, true, true) => Op::ImageSampleDrefImplicitLod,
            (false, false, true, false) => Op::ImageSampleDrefExplicitLod,
            (false, true, false, true) => Op::ImageSampleProjImplicitLod,
            (false, true, false, false) => Op::ImageSampleProjExplicitLod,
            (false, true, true, true) => Op::ImageSampleProjDrefImplicitLod,
            (false, true, true, false) => Op::ImageSampleProjDrefExplicitLod,
            (true, _, false, true) => Op::ImageSparseSampleImplicitLod,
            (true, _, false, false) => Op::ImageSparseSampleExplicitLod,
            (true, _, true, true) => Op::ImageSparseSampleDrefImplicitLod,
            (true, _, true, false) => Op::ImageSparseSampleDrefExplicitLod,
        }
    }
}

/// Image queries on an image value.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageQuery {
    /// `OpImageQuerySizeLod` with an integer level.
    SizeLod(Value),
    Size,
    Levels,
    Samples,
}

impl ImageQuery {
    /// The SPIR-V opcode.
    pub fn to_spirv(self) -> crate::spv::Op {
        use crate::spv::Op;
        match self {
            ImageQuery::SizeLod(_) => Op::ImageQuerySizeLod,
            ImageQuery::Size => Op::ImageQuerySize,
            ImageQuery::Levels => Op::ImageQueryLevels,
            ImageQuery::Samples => Op::ImageQuerySamples,
        }
    }
}

/// The `GroupOperation` of a subgroup reduction or scan.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroupOperation {
    Reduce,
    InclusiveScan,
    ExclusiveScan,
    /// Reduce within clusters of `size` lanes (a power of two, carried by constant `id`).
    ClusteredReduce {
        size: u32,
        id: ConstId,
    },
}

impl GroupOperation {
    /// The SPIR-V enumerant.
    pub fn to_spirv(self) -> crate::spv::GroupOperation {
        match self {
            GroupOperation::Reduce => crate::spv::GroupOperation::Reduce,
            GroupOperation::InclusiveScan => crate::spv::GroupOperation::InclusiveScan,
            GroupOperation::ExclusiveScan => crate::spv::GroupOperation::ExclusiveScan,
            GroupOperation::ClusteredReduce { .. } => crate::spv::GroupOperation::ClusteredReduce,
        }
    }
}

/// Subgroup arithmetic, bitwise, and logical reductions and scans.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroupArith {
    IAdd,
    FAdd,
    IMul,
    FMul,
    SMin,
    UMin,
    FMin,
    SMax,
    UMax,
    FMax,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
}

impl GroupArith {
    /// The SPIR-V opcode.
    pub fn to_spirv(self) -> crate::spv::Op {
        use crate::spv::Op;
        match self {
            GroupArith::IAdd => Op::GroupNonUniformIAdd,
            GroupArith::FAdd => Op::GroupNonUniformFAdd,
            GroupArith::IMul => Op::GroupNonUniformIMul,
            GroupArith::FMul => Op::GroupNonUniformFMul,
            GroupArith::SMin => Op::GroupNonUniformSMin,
            GroupArith::UMin => Op::GroupNonUniformUMin,
            GroupArith::FMin => Op::GroupNonUniformFMin,
            GroupArith::SMax => Op::GroupNonUniformSMax,
            GroupArith::UMax => Op::GroupNonUniformUMax,
            GroupArith::FMax => Op::GroupNonUniformFMax,
            GroupArith::BitwiseAnd => Op::GroupNonUniformBitwiseAnd,
            GroupArith::BitwiseOr => Op::GroupNonUniformBitwiseOr,
            GroupArith::BitwiseXor => Op::GroupNonUniformBitwiseXor,
            GroupArith::LogicalAnd => Op::GroupNonUniformLogicalAnd,
            GroupArith::LogicalOr => Op::GroupNonUniformLogicalOr,
            GroupArith::LogicalXor => Op::GroupNonUniformLogicalXor,
        }
    }
}

/// Subgroup shuffles.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShuffleOp {
    /// Read lane `x`.
    Shuffle,
    /// Read lane `self ^ x`.
    Xor,
}

/// Direction of a quad swap.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuadDirection {
    Horizontal,
    Vertical,
    Diagonal,
}

impl QuadDirection {
    /// The operand's value.
    pub fn bits(self) -> u32 {
        match self {
            QuadDirection::Horizontal => 0,
            QuadDirection::Vertical => 1,
            QuadDirection::Diagonal => 2,
        }
    }
}

/// A `GroupNonUniform*` operation. The execution scope is always `Subgroup`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GroupInst {
    Elect,
    All(Value),
    Any(Value),
    BroadcastFirst(Value),
    Ballot(Value),
    BallotBitExtract {
        ballot: Value,
        index: Value,
    },
    Shuffle {
        op: ShuffleOp,
        value: Value,
        x: Value,
    },
    Arith {
        op: GroupArith,
        group: GroupOperation,
        value: Value,
    },
    QuadBroadcast {
        value: Value,
        index: Value,
    },
    QuadSwap {
        value: Value,
        dir: QuadDirection,
        dir_id: ConstId,
    },
}

impl GroupInst {
    /// Call `f` on every SSA value operand, in operand order (constant operands carried as
    /// [`ConstId`]s are not visited).
    pub fn for_each_value(&self, mut f: impl FnMut(Value)) {
        match self {
            GroupInst::Elect => {}
            GroupInst::All(v)
            | GroupInst::Any(v)
            | GroupInst::BroadcastFirst(v)
            | GroupInst::Ballot(v)
            | GroupInst::Arith { value: v, .. }
            | GroupInst::QuadSwap { value: v, .. } => f(*v),
            GroupInst::BallotBitExtract {
                ballot: a,
                index: b,
            }
            | GroupInst::Shuffle { value: a, x: b, .. }
            | GroupInst::QuadBroadcast { value: a, index: b } => {
                f(*a);
                f(*b);
            }
        }
    }

    /// [`GroupInst::for_each_value`] with each operand open to change, in the same order.
    pub fn for_each_value_mut(&mut self, mut f: impl FnMut(&mut Value)) {
        match self {
            GroupInst::Elect => {}
            GroupInst::All(v)
            | GroupInst::Any(v)
            | GroupInst::BroadcastFirst(v)
            | GroupInst::Ballot(v)
            | GroupInst::Arith { value: v, .. }
            | GroupInst::QuadSwap { value: v, .. } => f(v),
            GroupInst::BallotBitExtract {
                ballot: a,
                index: b,
            }
            | GroupInst::Shuffle { value: a, x: b, .. }
            | GroupInst::QuadBroadcast { value: a, index: b } => {
                f(a);
                f(b);
            }
        }
    }

    /// Call `f` on every constant operand carried as a [`ConstId`]: a cluster size or a quad
    /// direction.
    pub fn for_each_const(&self, mut f: impl FnMut(ConstId)) {
        match self {
            GroupInst::Arith { group: g, .. } => {
                if let GroupOperation::ClusteredReduce { id, .. } = g {
                    f(*id);
                }
            }
            GroupInst::QuadSwap { dir_id, .. } => f(*dir_id),
            GroupInst::Elect
            | GroupInst::All(_)
            | GroupInst::Any(_)
            | GroupInst::BroadcastFirst(_)
            | GroupInst::Ballot(_)
            | GroupInst::BallotBitExtract { .. }
            | GroupInst::Shuffle { .. }
            | GroupInst::QuadBroadcast { .. } => {}
        }
    }

    /// The SPIR-V opcode.
    pub fn to_spirv(&self) -> crate::spv::Op {
        use crate::spv::Op;
        match self {
            GroupInst::Elect => Op::GroupNonUniformElect,
            GroupInst::All(_) => Op::GroupNonUniformAll,
            GroupInst::Any(_) => Op::GroupNonUniformAny,
            GroupInst::BroadcastFirst(_) => Op::GroupNonUniformBroadcastFirst,
            GroupInst::Ballot(_) => Op::GroupNonUniformBallot,
            GroupInst::BallotBitExtract { .. } => Op::GroupNonUniformBallotBitExtract,
            GroupInst::Shuffle { op, .. } => match op {
                ShuffleOp::Shuffle => Op::GroupNonUniformShuffle,
                ShuffleOp::Xor => Op::GroupNonUniformShuffleXor,
            },
            GroupInst::Arith { op, .. } => op.to_spirv(),
            GroupInst::QuadBroadcast { .. } => Op::GroupNonUniformQuadBroadcast,
            GroupInst::QuadSwap { .. } => Op::GroupNonUniformQuadSwap,
        }
    }
}

/// A value of an intersection a ray query reports: `OpRayQueryGetIntersection*KHR`
/// (`SPV_KHR_ray_query`). Each names one opcode; its result type is fixed by the opcode.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RayQueryValue {
    /// `OpRayQueryGetIntersectionTypeKHR`: `u32` (committed: none 0, triangle 1, generated 2;
    /// candidate: triangle 0, AABB 1).
    Type,
    /// `OpRayQueryGetIntersectionTKHR`: `f32`.
    T,
    /// `OpRayQueryGetIntersectionInstanceCustomIndexKHR`: `u32`.
    InstanceCustomIndex,
    /// `OpRayQueryGetIntersectionInstanceIdKHR`: `u32`.
    InstanceId,
    /// `OpRayQueryGetIntersectionGeometryIndexKHR`: `u32`.
    GeometryIndex,
    /// `OpRayQueryGetIntersectionPrimitiveIndexKHR`: `u32`.
    PrimitiveIndex,
    /// `OpRayQueryGetIntersectionBarycentricsKHR`: `vec2<f32>`.
    Barycentrics,
    /// `OpRayQueryGetIntersectionFrontFaceKHR`: `bool`.
    FrontFace,
    /// `OpRayQueryGetIntersectionObjectToWorldKHR`: a 4-column matrix of `vec3<f32>`.
    ObjectToWorld,
    /// `OpRayQueryGetIntersectionWorldToObjectKHR`: a 4-column matrix of `vec3<f32>`.
    WorldToObject,
    /// `OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR`: `u32`, the
    /// instance's 24-bit shader-binding-table record offset.
    InstanceShaderBindingTableRecordOffset,
}

impl RayQueryValue {
    /// The SPIR-V opcode.
    pub fn to_spirv(self) -> crate::spv::Op {
        use crate::spv::Op;
        match self {
            RayQueryValue::Type => Op::RayQueryGetIntersectionTypeKHR,
            RayQueryValue::T => Op::RayQueryGetIntersectionTKHR,
            RayQueryValue::InstanceCustomIndex => Op::RayQueryGetIntersectionInstanceCustomIndexKHR,
            RayQueryValue::InstanceId => Op::RayQueryGetIntersectionInstanceIdKHR,
            RayQueryValue::GeometryIndex => Op::RayQueryGetIntersectionGeometryIndexKHR,
            RayQueryValue::PrimitiveIndex => Op::RayQueryGetIntersectionPrimitiveIndexKHR,
            RayQueryValue::Barycentrics => Op::RayQueryGetIntersectionBarycentricsKHR,
            RayQueryValue::FrontFace => Op::RayQueryGetIntersectionFrontFaceKHR,
            RayQueryValue::ObjectToWorld => Op::RayQueryGetIntersectionObjectToWorldKHR,
            RayQueryValue::WorldToObject => Op::RayQueryGetIntersectionWorldToObjectKHR,
            RayQueryValue::InstanceShaderBindingTableRecordOffset => {
                Op::RayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR
            }
        }
    }
}

/// A ray query instruction (`SPV_KHR_ray_query`). `query` is always a pointer to a
/// `Function`- or `Private`-storage `OpTypeRayQueryKHR` variable.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RayQueryInst {
    /// `OpRayQueryInitializeKHR`.
    Initialize {
        query: Value,
        accel: Value,
        flags: Value,
        cull_mask: Value,
        origin: Value,
        tmin: Value,
        direction: Value,
        tmax: Value,
    },
    /// `OpRayQueryProceedKHR`: `bool`, whether traversal stopped at a candidate.
    Proceed { query: Value },
    /// `OpRayQueryConfirmIntersectionKHR`: commit the current (triangle) candidate.
    Confirm { query: Value },
    /// `OpRayQueryGetIntersection*KHR` of the committed (`committed`) or the candidate
    /// intersection, whose `Intersection` operand is the `u32` constant `which` (1 or 0).
    Get {
        query: Value,
        value: RayQueryValue,
        committed: bool,
        which: ConstId,
    },
}

impl RayQueryInst {
    /// Call `f` on every SSA value operand, in operand order (the `Intersection` constant of a
    /// get is carried as a [`ConstId`] and not visited).
    pub fn for_each_value(&self, mut f: impl FnMut(Value)) {
        match self {
            RayQueryInst::Initialize {
                query,
                accel,
                flags,
                cull_mask,
                origin,
                tmin,
                direction,
                tmax,
            } => {
                for v in [
                    *query, *accel, *flags, *cull_mask, *origin, *tmin, *direction, *tmax,
                ] {
                    f(v);
                }
            }
            RayQueryInst::Proceed { query }
            | RayQueryInst::Confirm { query }
            | RayQueryInst::Get { query, .. } => f(*query),
        }
    }

    /// [`RayQueryInst::for_each_value`] with each operand open to change, in the same order.
    pub fn for_each_value_mut(&mut self, mut f: impl FnMut(&mut Value)) {
        match self {
            RayQueryInst::Initialize {
                query,
                accel,
                flags,
                cull_mask,
                origin,
                tmin,
                direction,
                tmax,
            } => {
                for v in [
                    query, accel, flags, cull_mask, origin, tmin, direction, tmax,
                ] {
                    f(v);
                }
            }
            RayQueryInst::Proceed { query }
            | RayQueryInst::Confirm { query }
            | RayQueryInst::Get { query, .. } => f(query),
        }
    }

    /// Call `f` on the constant operand carried as a [`ConstId`]: a get's `Intersection`.
    pub fn for_each_const(&self, mut f: impl FnMut(ConstId)) {
        match self {
            RayQueryInst::Get { which, .. } => f(*which),
            RayQueryInst::Initialize { .. }
            | RayQueryInst::Proceed { .. }
            | RayQueryInst::Confirm { .. } => {}
        }
    }

    /// The SPIR-V opcode.
    pub fn to_spirv(&self) -> crate::spv::Op {
        use crate::spv::Op;
        match self {
            RayQueryInst::Initialize { .. } => Op::RayQueryInitializeKHR,
            RayQueryInst::Proceed { .. } => Op::RayQueryProceedKHR,
            RayQueryInst::Confirm { .. } => Op::RayQueryConfirmIntersectionKHR,
            RayQueryInst::Get { value, .. } => value.to_spirv(),
        }
    }
}

/// The `GLSL.std.450` instructions the core emits. Each has a typing contract checked by
/// `FunctionBuilder::glsl`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GlslOp {
    Round,
    RoundEven,
    Trunc,
    FAbs,
    SAbs,
    FSign,
    SSign,
    Floor,
    Ceil,
    Fract,
    Radians,
    Degrees,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Atan2,
    Pow,
    Exp,
    Log,
    Exp2,
    Log2,
    Sqrt,
    InverseSqrt,
    Determinant,
    MatrixInverse,
    ModfStruct,
    FMin,
    UMin,
    SMin,
    FMax,
    UMax,
    SMax,
    FClamp,
    UClamp,
    SClamp,
    FMix,
    Step,
    SmoothStep,
    Fma,
    FrexpStruct,
    Ldexp,
    PackSnorm4x8,
    PackUnorm4x8,
    PackSnorm2x16,
    PackUnorm2x16,
    PackHalf2x16,
    UnpackSnorm2x16,
    UnpackUnorm2x16,
    UnpackHalf2x16,
    UnpackSnorm4x8,
    UnpackUnorm4x8,
    Length,
    Distance,
    Cross,
    Normalize,
    FaceForward,
    Reflect,
    Refract,
    FindILsb,
    FindSMsb,
    FindUMsb,
    InterpolateAtCentroid,
    InterpolateAtSample,
    InterpolateAtOffset,
    NMin,
    NMax,
    NClamp,
}

impl GlslOp {
    /// The `GLSL.std.450` instruction number.
    pub fn to_spirv(self) -> crate::spv::GlslStd450Op {
        use crate::spv::GlslStd450Op as G;
        match self {
            GlslOp::Round => G::Round,
            GlslOp::RoundEven => G::RoundEven,
            GlslOp::Trunc => G::Trunc,
            GlslOp::FAbs => G::FAbs,
            GlslOp::SAbs => G::SAbs,
            GlslOp::FSign => G::FSign,
            GlslOp::SSign => G::SSign,
            GlslOp::Floor => G::Floor,
            GlslOp::Ceil => G::Ceil,
            GlslOp::Fract => G::Fract,
            GlslOp::Radians => G::Radians,
            GlslOp::Degrees => G::Degrees,
            GlslOp::Sin => G::Sin,
            GlslOp::Cos => G::Cos,
            GlslOp::Tan => G::Tan,
            GlslOp::Asin => G::Asin,
            GlslOp::Acos => G::Acos,
            GlslOp::Atan => G::Atan,
            GlslOp::Sinh => G::Sinh,
            GlslOp::Cosh => G::Cosh,
            GlslOp::Tanh => G::Tanh,
            GlslOp::Asinh => G::Asinh,
            GlslOp::Acosh => G::Acosh,
            GlslOp::Atanh => G::Atanh,
            GlslOp::Atan2 => G::Atan2,
            GlslOp::Pow => G::Pow,
            GlslOp::Exp => G::Exp,
            GlslOp::Log => G::Log,
            GlslOp::Exp2 => G::Exp2,
            GlslOp::Log2 => G::Log2,
            GlslOp::Sqrt => G::Sqrt,
            GlslOp::InverseSqrt => G::InverseSqrt,
            GlslOp::Determinant => G::Determinant,
            GlslOp::MatrixInverse => G::MatrixInverse,
            GlslOp::ModfStruct => G::ModfStruct,
            GlslOp::FMin => G::FMin,
            GlslOp::UMin => G::UMin,
            GlslOp::SMin => G::SMin,
            GlslOp::FMax => G::FMax,
            GlslOp::UMax => G::UMax,
            GlslOp::SMax => G::SMax,
            GlslOp::FClamp => G::FClamp,
            GlslOp::UClamp => G::UClamp,
            GlslOp::SClamp => G::SClamp,
            GlslOp::FMix => G::FMix,
            GlslOp::Step => G::Step,
            GlslOp::SmoothStep => G::SmoothStep,
            GlslOp::Fma => G::Fma,
            GlslOp::FrexpStruct => G::FrexpStruct,
            GlslOp::Ldexp => G::Ldexp,
            GlslOp::PackSnorm4x8 => G::PackSnorm4x8,
            GlslOp::PackUnorm4x8 => G::PackUnorm4x8,
            GlslOp::PackSnorm2x16 => G::PackSnorm2x16,
            GlslOp::PackUnorm2x16 => G::PackUnorm2x16,
            GlslOp::PackHalf2x16 => G::PackHalf2x16,
            GlslOp::UnpackSnorm2x16 => G::UnpackSnorm2x16,
            GlslOp::UnpackUnorm2x16 => G::UnpackUnorm2x16,
            GlslOp::UnpackHalf2x16 => G::UnpackHalf2x16,
            GlslOp::UnpackSnorm4x8 => G::UnpackSnorm4x8,
            GlslOp::UnpackUnorm4x8 => G::UnpackUnorm4x8,
            GlslOp::Length => G::Length,
            GlslOp::Distance => G::Distance,
            GlslOp::Cross => G::Cross,
            GlslOp::Normalize => G::Normalize,
            GlslOp::FaceForward => G::FaceForward,
            GlslOp::Reflect => G::Reflect,
            GlslOp::Refract => G::Refract,
            GlslOp::FindILsb => G::FindILsb,
            GlslOp::FindSMsb => G::FindSMsb,
            GlslOp::FindUMsb => G::FindUMsb,
            GlslOp::InterpolateAtCentroid => G::InterpolateAtCentroid,
            GlslOp::InterpolateAtSample => G::InterpolateAtSample,
            GlslOp::InterpolateAtOffset => G::InterpolateAtOffset,
            GlslOp::NMin => G::NMin,
            GlslOp::NMax => G::NMax,
            GlslOp::NClamp => G::NClamp,
        }
    }
}

/// What an instruction needs from the entry point's execution model. Checked once the entry point
/// and every function are known (`ModuleBuilder::finalize`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelNeed {
    /// Implicit derivatives: fragment, or compute with a derivative group mode.
    ImplicitDerivatives,
    /// The fragment execution model.
    Fragment,
    /// The fragment execution model with a fragment-interlock mode.
    Interlock,
    /// A `Workgroup` execution or memory scope: compute only (VUID-StandaloneSpirv-None-04637,
    /// VUID-StandaloneSpirv-None-07321). With the builder's `Workgroup`-or-`Subgroup` execution
    /// scope, this also keeps a vertex or fragment `OpControlBarrier` at `Subgroup` execution
    /// scope (VUID-StandaloneSpirv-OpControlBarrier-04682). A task or mesh entry has it too.
    WorkgroupScope,
    /// The mesh execution model (`OpSetMeshOutputsEXT`).
    Mesh,
}
