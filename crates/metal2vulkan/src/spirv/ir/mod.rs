//! Typed module data model and interners.

pub mod constant;
pub mod edit;
pub mod inst;
pub(crate) mod intern;
pub mod module;
pub mod types;

pub use constant::{ConstBits, ConstDef, ConstKind, Constant, Consts, SpecOp};
pub use edit::{InstAt, Promotion};
pub use inst::{
    AtomicOp, BinOp, Contraction, ConvOp, DerivOp, GlslOp, GroupArith, GroupInst, GroupOperation,
    ImageOperandKind, ImageOperands, ImageQuery, MemoryOrder, ModelNeed, QuadDirection,
    RayQueryInst, RayQueryValue, SampleForm, Scope, ScopeId, Semantics, SemanticsId, ShuffleOp,
    TexelMemory, UnOp,
};
pub use module::{
    Block, Decl, EntryPoint, ExecMode, ExecModel, Function, GlobalKind, GlobalVar, Inst, InstNode,
    Interpolation, IoDecor, LocalVar, LoopMeta, MemAccess, Merge, Phi, PushMember, Sampling,
    Terminator, TypedModule, Unrolling, ValueDef, ValueOrigin,
};
pub use types::{
    natural_layout, Dim, ImageType, ImageUsage, IoDir, LayoutRule, MemberLayout, Shape,
    StorageClass, StructLayout, Type, TypeFacts, Types,
};
