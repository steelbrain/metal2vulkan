//! Typed ids shared by the builder, verifier, structurizer, and lowering.
//!
//! Each id is a dense `u32` index into the arena of its kind inside one `TypedModule`; ids of
//! different kinds never mix. The builder is the only producer of ids for a real module.

macro_rules! id_type {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u32);

        impl $name {
            /// The id with this dense index.
            pub const fn new(index: u32) -> $name {
                $name(index)
            }

            /// The dense index, for indexing side tables.
            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

id_type!(
    /// An SSA value: an instruction result, constant, global, or parameter.
    Value
);
id_type!(
    /// A `Function`-storage variable (demotion vars, loop-carried values).
    VarId
);
id_type!(
    /// A basic block.
    BlockId
);
id_type!(
    /// An interned type.
    TypeId
);
id_type!(
    /// An interned constant.
    ConstId
);
id_type!(
    /// A module-scope variable.
    GlobalId
);
id_type!(
    /// A function.
    FuncId
);

/// A scalar type a demoted value can have.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScalarTy {
    Bool,
    Int { bits: u8 },
    Float { bits: u8 },
}

/// The type of a structurizer-demoted value: the Plan's lowered representation, which is always
/// demotable.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DemoteTy {
    /// A legalized scalar.
    Scalar(ScalarTy),
    /// A legalized vector.
    Vector { elem: ScalarTy, len: u8 },
    /// A legalized aggregate of demotable members.
    Composite(Box<[DemoteTy]>),
    /// A fat buffer pointer `(root u32, offset u32 | u64)`.
    FatPtr { wide: bool },
    /// A device address `u64`: a wide buffer pointer under the identity domains, where a pointer
    /// is the address it holds.
    Address,
    /// A workgroup or private arena offset `u32`.
    ArenaOffset,
    /// A handle-table index `u32`.
    HandleIndex,
}
