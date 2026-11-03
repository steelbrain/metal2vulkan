//! The exhaustive typed AIR IR.
//!
//! Arena storage with `u32` newtype ids, interned types, constants, and strings, and no `Rc` or
//! string-keyed lookups in the hot structure. Ids follow insertion order, which the decoder makes
//! bitcode order, so identical bytes produce identical IR. Pointer types carry only an address
//! space. Every AIR value has one dense [`ValueId`].

pub mod attrs;
// The IR builder: synthetic modules for tests and tooling.
pub mod build;
pub mod cfg;
pub mod constant;
pub mod id;
pub mod inst;
pub(crate) mod intern;
pub mod layout;
pub mod metadata;
pub mod module;
// The IR printer: the IR as text, for diagnostics, tests, and tooling.
pub mod print;
pub mod symbols;
pub mod types;
pub mod value;
pub mod verify;

pub use attrs::{CallAttrs, FnAttrs, MemoryEffects, ModRef, ParamAttrs};
pub use constant::{Const, ConstExpr, ConstTable, Constant, InlineAsm};
pub use id::{Arena, ArgIdx, BlockId, ConstId, FuncId, GlobalId, Idx, InstId, MdId, Sym, TypeId};
#[cfg(test)]
pub use inst::Opcode;
pub use inst::{
    Align, Atomic, AtomicRmwOp, BinOp, Call, CallConv, Callee, CastOp, FastMath, FloatPredicate,
    Inst, InstKind, IntFlags, IntPredicate, LoopHints, Ordering, RangeMd, SyncScope, TailKind,
    Unroll, Value,
};
pub use layout::{DataLayout, Layout, TypeLayouts};
pub use metadata::{Md, NamedMd};
pub use module::{
    Block, Body, Function, Global, Linkage, ModuleParts, Param, Section, UnnamedAddr, Visibility,
};
pub use symbols::Symbols;
pub use types::{AddrSpace, Type, TypeTable};
pub use value::{ValueId, ValueIndex};
pub use verify::Module;

#[cfg(test)]
mod print_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod verify_tests;
