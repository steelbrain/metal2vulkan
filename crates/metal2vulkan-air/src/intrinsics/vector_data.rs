//! OpenCL C's vector data functions, as Apple's OpenCL compiler spells them:
//! `air.vload.<space>.<T>` and `air.vstore.<space>.<T>` (`vload<n>`, `vstore<n>`), and over `half`
//! data `air.vload_half.<space>.<T>`, `air.vloada_half.<space>.<T>`,
//! `air.vstore_half.<space>.<rounding>.<T>` and `air.vstorea_half.<space>.<rounding>.<T>`
//! (`vload_half[<n>]`, `vloada_half<n>`, `vstore_half[<n>][_<rounding>]`,
//! `vstorea_half<n>[_<rounding>]`; the compiler spells a store without a rounding `rte`).
//!
//! A load is `T (i64 offset, ptr p)` and a store `void (T v, i64 offset, ptr p)`, `p` a pointer to
//! the data's elements in `<space>`: `global` (device), `constant` (loads only), `local`
//! (threadgroup) or `private` (thread). `T` is `v<n><E>` for `vload` and `vstore`, `E` one of `i8`,
//! `i16`, `i32`, `i64` and `f32` (OpenCL's signed and unsigned types alike) and `n` one of 2, 3, 4,
//! 8 and 16; it is `f32` or `v<n>f32` for the `half` forms (`v<n>f32` for `vloada_half` and
//! `vstorea_half`), whose data is `half`. Canonicalization makes each call its element accesses
//! (`canon::vector_data`).

use super::sig::{Operand, Signature};
use super::ty::{Elem, NumTy};
use crate::ir::AddrSpace;

/// The rounding a store to `half` data names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HalfRounding {
    /// `rte`: to nearest, ties to even.
    Rte,
    /// `rtz`: toward zero.
    Rtz,
    /// `rtp`: toward positive infinity.
    Rtp,
    /// `rtn`: toward negative infinity.
    Rtn,
}

impl HalfRounding {
    fn parse(token: &str) -> Option<HalfRounding> {
        Some(match token {
            "rte" => HalfRounding::Rte,
            "rtz" => HalfRounding::Rtz,
            "rtp" => HalfRounding::Rtp,
            "rtn" => HalfRounding::Rtn,
            _ => return None,
        })
    }
}

/// What a vector data function does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VectorDataOp {
    /// `air.vload`: `T`'s elements from `p + offset * n`.
    Load,
    /// `air.vstore`: `v`'s elements to `p + offset * n`.
    Store,
    /// `air.vload_half` and (`aligned`) `air.vloada_half`: `T`'s lanes, each a `half` widened to
    /// `float`, from `p + offset * n` (a `vloada_half3`'s from `p + offset * 4`).
    LoadHalf {
        /// `vloada_half`.
        aligned: bool,
    },
    /// `air.vstore_half` and (`aligned`) `air.vstorea_half`: `v`'s lanes, each narrowed to `half`,
    /// to `p + offset * n` (a `vstorea_half3`'s to `p + offset * 4`).
    StoreHalf {
        /// `vstorea_half`.
        aligned: bool,
        /// The rounding the name spells.
        rounding: HalfRounding,
    },
}

/// A decoded vector data function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VectorData {
    /// The operation.
    pub op: VectorDataOp,
    /// The data's address space, the name's `<space>`.
    pub space: AddrSpace,
    /// The value type the name spells: the load's result, the store's value.
    pub ty: NumTy,
}

impl VectorData {
    /// Parse the part of a name after `air.`. `None` is no vector data function.
    pub(super) fn parse(rest: &str) -> Option<VectorData> {
        let mut t = rest.split('.');
        let (head, space) = (t.next()?, t.next()?);
        let load = head.starts_with("vload");
        let op = match head {
            "vload" => VectorDataOp::Load,
            "vstore" => VectorDataOp::Store,
            "vload_half" | "vloada_half" => VectorDataOp::LoadHalf {
                aligned: head == "vloada_half",
            },
            "vstore_half" | "vstorea_half" => VectorDataOp::StoreHalf {
                aligned: head == "vstorea_half",
                rounding: HalfRounding::parse(t.next()?)?,
            },
            _ => return None,
        };
        let space = match space {
            "global" => AddrSpace::Device,
            "constant" if load => AddrSpace::Constant,
            "local" => AddrSpace::Threadgroup,
            "private" => AddrSpace::Thread,
            _ => return None,
        };
        let ty = NumTy::parse(t.next()?)?;
        if t.next().is_some() || !matches!(ty.lanes, 1 | 2 | 3 | 4 | 8 | 16) {
            return None;
        }
        let ok = match op {
            VectorDataOp::Load | VectorDataOp::Store => {
                ty.lanes > 1
                    && matches!(
                        ty.elem,
                        Elem::I8 | Elem::I16 | Elem::I32 | Elem::I64 | Elem::F32
                    )
            }
            VectorDataOp::LoadHalf { aligned } | VectorDataOp::StoreHalf { aligned, .. } => {
                ty.elem == Elem::F32 && !(aligned && ty.lanes == 1)
            }
        };
        ok.then_some(VectorData { op, space, ty })
    }

    /// `T (i64, ptr)` or `void (T, i64, ptr)`.
    pub fn signature(&self) -> Signature {
        let (t, offset, p) = (
            Operand::Num(self.ty),
            Operand::Num(NumTy::scalar(Elem::I64)),
            Operand::Ptr(self.space),
        );
        match self.op {
            VectorDataOp::Load | VectorDataOp::LoadHalf { .. } => Signature::new(t, &[offset, p]),
            VectorDataOp::Store | VectorDataOp::StoreHalf { .. } => {
                Signature::new(Operand::Void, &[t, offset, p])
            }
        }
    }

    /// The element the data holds: `half` for the `half` forms, else `T`'s.
    pub fn data(&self) -> Elem {
        match self.op {
            VectorDataOp::Load | VectorDataOp::Store => self.ty.elem,
            VectorDataOp::LoadHalf { .. } | VectorDataOp::StoreHalf { .. } => Elem::F16,
        }
    }

    /// The elements one step of `offset` moves `p`: `n`, but 4 for a `vloada_half3` or a
    /// `vstorea_half3`.
    pub fn stride(&self) -> u32 {
        match self.op {
            VectorDataOp::LoadHalf { aligned: true }
            | VectorDataOp::StoreHalf { aligned: true, .. }
                if self.ty.lanes == 3 =>
            {
                4
            }
            _ => self.ty.lanes,
        }
    }

    /// Whether the function stores.
    pub fn stores(&self) -> bool {
        matches!(
            self.op,
            VectorDataOp::Store | VectorDataOp::StoreHalf { .. }
        )
    }
}
