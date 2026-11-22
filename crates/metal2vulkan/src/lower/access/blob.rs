//! Pointers into the pipeline constant blob and loads through them.
//!
//! A buffer pointer whose every root is a constant global placed in the blob
//! (`ConstGlobalForm::Blob`, `memory::plan`) is the `u32` byte offset from the blob's start
//! (`DemoteTy::ArenaOffset`): a global's address is its blob offset, a GEP adds its datalayout
//! offsets with the buffer family's index rule (`access::gep_offset`), and `bitcast`,
//! `addrspacecast`, `freeze`, and `select` copy or select the offset. Several blob globals in one
//! root set share the representation, since their offsets are all blob offsets. A load is a
//! word-composed leaf from the blob's words (`memory::blob`): a bound blob's storage view, under
//! the normative check against the bound size, or under the identity domains the module's
//! `Private` words, checked against their length on every load.
//!
//! The blob is constant: a store through a blob pointer is refused, as is a root set that mixes
//! the blob with a parameter slot or null (not implemented: it needs the blob's root in the
//! switch-at-use of `access`), and a phi with a constant pointer operand.

use std::sync::Arc;

use crate::air::ir::{CastOp, Const, Inst, InstId, InstKind, Value, ValueId};
use crate::error::Error;
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::memory::blob::{BlobDecl, BlobHome, BlobUse};
use crate::memory::contain;
use crate::memory::leaf::{self, k32, Leaf, WordMem, WordOps};
use crate::memory::plan::{ConstGlobalForm, MemoryPlan};
use crate::memory::provenance::{Repr, RootSetId};
use crate::memory::views;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, Value as SValue};
use crate::spirv::ir::BinOp;

use super::emit::{index_u32, ViewMem};
use super::local::LocalPtr;
use super::root_sets::SetFacts;
use super::{is_ptr_ty, operand, refuse, GepTerms};

/// A leaf load from the blob, checked in the shader against the blob's length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlobLoad {
    pub ptr: LocalPtr,
    pub leaf: Leaf,
}

/// The blob strategies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlobStrategy {
    /// `getelementptr` on a blob offset: `base + konst + Σ index · stride` (wrapping `u32`).
    Gep {
        base: LocalPtr,
        konst: u32,
        terms: GepTerms,
    },
    /// `bitcast`, `addrspacecast`, or `freeze` of a blob pointer.
    Copy {
        of: LocalPtr,
    },
    /// `select c, t, f` of two blob pointers.
    Select {
        c: ValueId,
        t: ValueId,
        f: ValueId,
    },
    Load(BlobLoad),
}

impl BlobStrategy {
    /// The census name (`access.blob.<op>`).
    pub fn name(&self) -> &'static str {
        match self {
            BlobStrategy::Gep { .. } => "access.blob.gep",
            BlobStrategy::Copy { .. } => "access.blob.pointer_copy",
            BlobStrategy::Select { .. } => "access.blob.pointer_select",
            BlobStrategy::Load(_) => "access.blob.load.checked",
        }
    }

    /// An upper estimate of the SPIR-V words one use emits.
    pub fn words(&self) -> u32 {
        match self {
            BlobStrategy::Gep { terms, .. } => {
                12u32.saturating_add(16u32.saturating_mul(terms.len() as u32))
            }
            BlobStrategy::Copy { .. } => 4,
            BlobStrategy::Select { .. } => 6,
            BlobStrategy::Load(l) => l.leaf.cost().saturating_add(64).saturating_add(16),
        }
    }
}

/// The blob offset of constant global `g`, when it is in the blob.
pub(super) fn global_offset(mp: &MemoryPlan, g: crate::air::ir::GlobalId) -> Option<u64> {
    mp.const_global(g).and_then(|c| match c.form {
        ConstGlobalForm::Blob { offset } => Some(offset),
        ConstGlobalForm::Private => None,
    })
}

/// Whether pointer `v` points into the blob: every root is a blob-form constant global (its set's
/// facts, read once for the function).
pub fn is_blob(cx: &SelectCx<'_>, v: Value) -> Result<bool, Error> {
    let Some(Repr::BufPtr { roots, wide: false }) = cx.memory.repr(v) else {
        return Ok(false);
    };
    Ok(set_facts(cx, roots)?.blob)
}

/// A root set holding a blob global and anything else: a static fat pointer under
/// `access::BLOB_ROOT`, unless it is wide.
fn mixed(cx: &SelectCx<'_>, v: Value) -> Result<bool, Error> {
    let Some(Repr::BufPtr { roots, .. }) = cx.memory.repr(v) else {
        return Ok(false);
    };
    Ok(set_facts(cx, roots)?.any_blob && !is_blob(cx, v)?)
}

/// The facts of a buffer pointer's root set; one selection did not read is an invariant
/// violation, as in `access::statics`.
fn set_facts<'c>(cx: &'c SelectCx<'_>, roots: RootSetId) -> Result<&'c SetFacts, Error> {
    cx.root_sets
        .get(roots)
        .ok_or_else(|| cx.invariant("a buffer pointer's root set was not read"))
}

/// The lowered representation of blob pointer `v` (`DemoteTy::ArenaOffset`), `None` for any other
/// pointer. A phi with a constant incoming pointer is refused.
pub fn lowered_pointer(cx: &SelectCx<'_>, v: Value) -> Result<Option<DemoteTy>, Error> {
    if mixed(cx, v)? {
        // A static fat pointer (`access::lowered_pointer`); the blob in a wide root set is not
        // implemented.
        return match cx.memory.repr(v) {
            Some(Repr::BufPtr { wide: false, .. }) => Ok(None),
            _ => Err(refuse()),
        };
    }
    if !is_blob(cx, v)? {
        return Ok(None);
    }
    if super::phis::incoming(cx, v)?.is_some_and(|i| i.constant) {
        return Err(refuse());
    }
    Ok(Some(DemoteTy::ArenaOffset))
}

/// Record that the function reads the blob, its bytes shared with the memory plan's: under the
/// identity domains the module's words, with no descriptor; otherwise bound, the artifact carrying
/// its bytes and its storage descriptor counting in the per-stage limits.
pub fn use_blob(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if cx.families.access.blob.is_some() {
        return Ok(());
    }
    let bytes = match &cx.memory.blob {
        Some(b) => Arc::clone(&b.bytes),
        None => return Err(cx.invariant("a blob pointer without a blob")),
    };
    let home = match super::identity(cx.profile) {
        true => BlobHome::Module,
        false => BlobHome::Bound,
    };
    cx.families.access.blob = Some(BlobUse { bytes, home });
    if home == BlobHome::Bound {
        cx.descriptors.storage_buffers = cx.descriptors.storage_buffers.saturating_add(1);
    }
    Ok(())
}

/// The operand form of blob pointer `v`: a value, or a global's address as its blob offset.
pub fn blob_ptr(cx: &SelectCx<'_>, v: Value) -> Result<LocalPtr, Error> {
    let Value::Const(c) = v else {
        return Ok(LocalPtr::Value(operand(cx, v)?));
    };
    match cx.entry.module.consts().get(c).map(|k| &k.value) {
        Some(Const::GlobalAddr(g)) => {
            let off = global_offset(cx.memory, *g).ok_or_else(refuse)?;
            Ok(LocalPtr::Offset(u32::try_from(off).map_err(|_| refuse())?))
        }
        _ => Err(refuse()),
    }
}

/// A leaf load of AIR type `ty` with alignment `align` through blob pointer `ptr`.
pub fn load(
    cx: &mut SelectCx<'_>,
    ptr: Value,
    ty: crate::air::ir::TypeId,
    align: Option<crate::air::ir::Align>,
) -> Result<BlobLoad, Error> {
    use_blob(cx)?;
    let align = align.map_or(1, |a| a.bytes());
    let proven = cx.memory.proven_align(cx.entry, ptr);
    let leaf = leaf::leaf_of(cx.entry, cx.profile, &cx.memory.layouts, ty, align, proven)?;
    Ok(BlobLoad {
        ptr: blob_ptr(cx, ptr)?,
        leaf,
    })
}

/// Select the blob strategy of `inst`, or `None` when it involves no blob pointer.
pub fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
) -> Result<Option<BlobStrategy>, Error> {
    let here = Value::Inst(at);
    let blob = |cx: &SelectCx<'_>, v: Value| -> Result<(), Error> {
        if is_blob(cx, v)? {
            Ok(())
        } else {
            Err(refuse())
        }
    };
    Ok(Some(match &inst.kind {
        InstKind::Gep {
            src_ty,
            base,
            indices,
            ..
        } => {
            if !is_blob(cx, here)? {
                return Ok(None);
            }
            blob(cx, *base)?;
            use_blob(cx)?;
            let base = blob_ptr(cx, *base)?;
            let (konst, terms) = super::gep_offset(cx, *src_ty, indices)?;
            BlobStrategy::Gep { base, konst, terms }
        }
        InstKind::Cast {
            op: CastOp::Bitcast | CastOp::AddrSpaceCast,
            v,
        }
        | InstKind::Freeze(v)
            if is_ptr_ty(cx, inst.ty) =>
        {
            if !is_blob(cx, here)? {
                return Ok(None);
            }
            blob(cx, *v)?;
            use_blob(cx)?;
            BlobStrategy::Copy {
                of: blob_ptr(cx, *v)?,
            }
        }
        InstKind::Select { c, t, f, .. } if is_ptr_ty(cx, inst.ty) => {
            if !is_blob(cx, here)? {
                return Ok(None);
            }
            blob(cx, *t)?;
            blob(cx, *f)?;
            use_blob(cx)?;
            BlobStrategy::Select {
                c: operand(cx, *c)?,
                t: operand(cx, *t)?,
                f: operand(cx, *f)?,
            }
        }
        InstKind::Load {
            ptr,
            align,
            volatile: false,
            atomic: None,
            ..
        } if !is_ptr_ty(cx, inst.ty) => {
            if !is_blob(cx, *ptr)? {
                return Ok(None);
            }
            BlobStrategy::Load(load(cx, *ptr, inst.ty, *align)?)
        }
        // The blob is constant: a store through it has no Metal meaning here.
        InstKind::Store { ptr, .. } => {
            return match is_blob(cx, *ptr)? {
                true => Err(refuse()),
                false => Ok(None),
            }
        }
        _ => return Ok(None),
    }))
}

/// The offset value of blob pointer operand `p`.
pub fn ptr_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    p: LocalPtr,
) -> BResult<SValue> {
    match p {
        LocalPtr::Value(v) => cx.operand(fb, v),
        LocalPtr::Offset(o) => k32(fb, o),
    }
}

/// Load `leaf` at blob byte offset `off`, contained.
pub fn load_at(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    leaf: &Leaf,
    off: SValue,
) -> BResult<SValue> {
    let Some(decl) = cx.decl.families.access.blob else {
        return Err(cx.fail("the constant blob is not declared"));
    };
    load_decl(fb, decl, leaf, off)
}

/// Load `leaf` at byte offset `off` of the declared blob `decl`, contained: a bound blob's view
/// and the module's words alike.
pub fn load_decl(
    fb: &mut FunctionBuilder<'_>,
    decl: BlobDecl,
    leaf: &Leaf,
    off: SValue,
) -> BResult<SValue> {
    // The blob is read-only: its words are the same on every load, which stay plain.
    match decl {
        BlobDecl::Bound(view) => {
            let mut mem = ViewMem(view, WordOps::default(), views::UnitViews::NONE);
            let size = views::storage_bytes(fb, view)?;
            contained(fb, &mut mem, leaf, off, size)
        }
        BlobDecl::Module(mut words) => {
            let size = k32(fb, words.words.saturating_mul(4))?;
            contained(fb, &mut words, leaf, off, size)
        }
    }
}

/// Load `leaf` at byte offset `off` of `mem`, whose bytes number `size`: the leaf, or zero where
/// it does not lie wholly inside.
fn contained(
    fb: &mut FunctionBuilder<'_>,
    mem: &mut dyn WordMem,
    leaf: &Leaf,
    off: SValue,
    size: SValue,
) -> BResult<SValue> {
    let inb = contain::in_bounds(fb, off, size, u64::from(leaf.bytes))?;
    let zero = k32(fb, 0)?;
    let at = fb.select(inb, off, zero)?;
    let v = leaf::load(fb, mem, leaf, at)?;
    leaf::contained(fb, leaf, inb, v)
}

/// Emit blob strategy `s`, defining value `vid`.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    vid: ValueId,
    s: &BlobStrategy,
) -> BResult<()> {
    let r = match s {
        BlobStrategy::Gep { base, konst, terms } => {
            let mut off = ptr_value(cx, fb, *base)?;
            if *konst != 0 {
                let c = k32(fb, *konst)?;
                off = fb.binary(BinOp::IAdd, off, c)?;
            }
            for (v, stride, form) in terms {
                let x = cx.operand(fb, *v)?;
                let mut x = index_u32(fb, x, *form)?;
                if *stride != 1 {
                    let s = k32(fb, *stride)?;
                    x = fb.binary(BinOp::IMul, x, s)?;
                }
                off = fb.binary(BinOp::IAdd, off, x)?;
            }
            off
        }
        BlobStrategy::Copy { of } => ptr_value(cx, fb, *of)?,
        BlobStrategy::Select { c, t, f } => {
            let c = cx.operand(fb, *c)?;
            let t = cx.operand(fb, *t)?;
            let f = cx.operand(fb, *f)?;
            fb.select(c, t, f)?
        }
        BlobStrategy::Load(l) => {
            let off = ptr_value(cx, fb, l.ptr)?;
            load_at(cx, fb, &l.leaf, off)?
        }
    };
    cx.define(vid, r);
    Ok(())
}
