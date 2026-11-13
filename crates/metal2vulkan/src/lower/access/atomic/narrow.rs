//! Narrow atomics: LLVM atomics on `i8` and `i16`. No memory this family reaches has a word
//! narrower than 32 bits (descriptor views and the arena are `u32` arrays, and `Int8`/`Int16`
//! atomics are not core Vulkan), so a narrow atomic operates on the field of its containing `u32`
//! word: the word at `offset & ~3`, the field `bits` wide at bit `(offset & 3) * 8` (an LLVM
//! atomic's alignment is at least its size, so the field never straddles two words; selection
//! refuses an under-aligned one). Values are held zero-extended in a `u32`; the operand is masked
//! to the field first (a poison operand, a `nuw` operation's, `ops::int_binary`, may carry bits
//! above its width), so an operand shifted into place sets no bit outside its field.
//!
//! - `load` is `OpAtomicLoad` of the word and the field extracted from it.
//! - `and`, `or`, and `xor` are one word atomic each, whose operand leaves the other fields
//!   unchanged (`and` with the other fields' bits set, `or` and `xor` with them clear).
//! - Every other operation (`add`, `sub`, the `min`/`max` family, `xchg`, `store`, the
//!   compare-and-swap operators, and `cmpxchg`) is a compare-and-swap loop on the word, as a
//!   32-bit operation SPIR-V lacks is (`super::cas_loop`): it reads the word, computes the new
//!   field, and exchanges the word with the other fields as read. A neighbouring field's change
//!   fails the exchange, so the loop retries until no other invocation wrote the word in between;
//!   it is lock-free but not statically bounded, so it opens a builder loop. `cmpxchg` leaves the
//!   loop without writing when the field differs from the comparator; the read then carries the
//!   failure order.
//!
//! The field's new value is computed at its width: `add` and `sub` wrap, signed `min`/`max`
//! compare the sign-extended fields, and the result is masked to the field.

use crate::memory::leaf::k32;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{
    AtomicOp, BinOp, ConstBits, Constant, GlslOp, MemAccess, MemoryOrder, Semantics, UnOp,
};

use super::{cas_new, dangling, load_sem, store_sem, AtomicKind, Op};

/// A narrow field of a `u32` word.
#[derive(Clone, Copy)]
struct Field {
    /// The field's bit offset in the word (a `u32`).
    shift: SValue,
    /// The field's width in bits (8 or 16).
    bits: u8,
    /// `2^bits - 1`.
    lane: SValue,
    /// `lane << shift`.
    mask: SValue,
}

impl Field {
    /// The field of `word`, zero-extended.
    fn get(self, fb: &mut FunctionBuilder<'_>, word: SValue) -> BResult<SValue> {
        let s = fb.binary(BinOp::ShiftRightLogical, word, self.shift)?;
        fb.binary(BinOp::BitwiseAnd, s, self.lane)
    }

    /// `word` with the field set to `v` (a value of the field's width).
    fn put(self, fb: &mut FunctionBuilder<'_>, word: SValue, v: SValue) -> BResult<SValue> {
        let keep = fb.unary(UnOp::Not, self.mask)?;
        let rest = fb.binary(BinOp::BitwiseAnd, word, keep)?;
        let placed = fb.binary(BinOp::ShiftLeftLogical, v, self.shift)?;
        fb.binary(BinOp::BitwiseOr, rest, placed)
    }

    /// `v` sign-extended from the field's width to 32 bits.
    fn sext(self, fb: &mut FunctionBuilder<'_>, v: SValue) -> BResult<SValue> {
        let up = k32(fb, 32 - u32::from(self.bits))?;
        let hi = fb.binary(BinOp::ShiftLeftLogical, v, up)?;
        fb.binary(BinOp::ShiftRightArithmetic, hi, up)
    }
}

/// The bit offset (a `u32`) of the field at byte offset `off` (a `u32`) in its word:
/// `(off & 3) * 8`.
pub(super) fn shift(fb: &mut FunctionBuilder<'_>, off: SValue) -> BResult<SValue> {
    let three = k32(fb, 3)?;
    let byte = fb.binary(BinOp::BitwiseAnd, off, three)?;
    let k = k32(fb, 3)?;
    fb.binary(BinOp::ShiftLeftLogical, byte, k)
}

/// Whether a narrow atomic of `kind` is a compare-and-swap loop.
pub(super) fn loops(kind: AtomicKind) -> bool {
    !matches!(
        kind,
        AtomicKind::Load | AtomicKind::Rmw(AtomicOp::And | AtomicOp::Or | AtomicOp::Xor)
    )
}

/// Apply narrow atomic `o` of width `bits` to the field at bit `shift` of word pointer `p`: the
/// field's old value, zero-extended (`None` for a store).
pub(super) fn apply(
    fb: &mut FunctionBuilder<'_>,
    p: SValue,
    shift: SValue,
    bits: u8,
    o: Op,
) -> BResult<Option<SValue>> {
    let lane = k32(fb, (1u32 << bits) - 1)?;
    let mask = fb.binary(BinOp::ShiftLeftLogical, lane, shift)?;
    let value = match o.value {
        Some(x) => Some(fb.binary(BinOp::BitwiseAnd, x, lane)?),
        None => None,
    };
    let o = Op { value, ..o };
    let f = Field {
        shift,
        bits,
        lane,
        mask,
    };
    match o.kind {
        AtomicKind::Load => {
            let w = fb.atomic_load(p, o.scope, load_sem(o.sem))?;
            f.get(fb, w).map(Some)
        }
        AtomicKind::Rmw(op @ (AtomicOp::And | AtomicOp::Or | AtomicOp::Xor)) => {
            let x = o.value.ok_or_else(dangling)?;
            let mut operand = fb.binary(BinOp::ShiftLeftLogical, x, shift)?;
            if op == AtomicOp::And {
                let keep = fb.unary(UnOp::Not, mask)?;
                operand = fb.binary(BinOp::BitwiseOr, operand, keep)?;
            }
            let w = fb.atomic(op, p, o.scope, o.sem, operand)?;
            f.get(fb, w).map(Some)
        }
        AtomicKind::Store => {
            cas(fb, p, f, o)?;
            Ok(None)
        }
        _ => cas(fb, p, f, o).map(Some),
    }
}

/// The field's new value for old value `cur` and operand `x` (both of the field's width).
fn next(fb: &mut FunctionBuilder<'_>, f: Field, o: Op, cur: SValue) -> BResult<SValue> {
    let x = o.value.ok_or_else(dangling)?;
    let r = match o.kind {
        AtomicKind::Store | AtomicKind::CompareExchange | AtomicKind::Rmw(AtomicOp::Exchange) => {
            return Ok(x)
        }
        AtomicKind::Rmw(AtomicOp::IAdd) => fb.binary(BinOp::IAdd, cur, x)?,
        AtomicKind::Rmw(AtomicOp::ISub) => fb.binary(BinOp::ISub, cur, x)?,
        // Both are zero-extended, so the unsigned order is the field's.
        AtomicKind::Rmw(AtomicOp::UMin) => return fb.glsl(GlslOp::UMin, &[cur, x]),
        AtomicKind::Rmw(AtomicOp::UMax) => return fb.glsl(GlslOp::UMax, &[cur, x]),
        AtomicKind::Rmw(op @ (AtomicOp::SMin | AtomicOp::SMax)) => {
            let a = f.sext(fb, cur)?;
            let b = f.sext(fb, x)?;
            let g = if op == AtomicOp::SMin {
                GlslOp::SMin
            } else {
                GlslOp::SMax
            };
            fb.glsl(g, &[a, b])?
        }
        AtomicKind::Cas(op) if !op.is_float() => cas_new(fb, op, o.word, cur, x)?,
        // `and`/`or`/`xor` and `load` are no loop; float kinds are never narrow (selection).
        _ => return Err(dangling()),
    };
    fb.binary(BinOp::BitwiseAnd, r, f.lane)
}

/// The compare-and-swap loop of `o` on field `f` of word pointer `p`: the field's old value.
fn cas(fb: &mut FunctionBuilder<'_>, p: SValue, f: Field, o: Op) -> BResult<SValue> {
    let zero = fb.module().constant(Constant::Scalar {
        ty: o.word,
        bits: ConstBits(0),
    })?;
    // The old field leaves the loop through a variable.
    let var = fb.local_var(o.word, Some(zero))?;
    let out = fb.var_ptr(var)?;
    let compare = o.kind == AtomicKind::CompareExchange;
    let (read, success) = match o.kind {
        // The read is an atomic load of its own, so it keeps the failure's acquire whatever
        // the success's order.
        AtomicKind::CompareExchange => (load_sem(o.failure), o.sem),
        AtomicKind::Store => (Semantics::RELAXED, store_sem(o.sem)),
        _ => (Semantics::RELAXED, o.sem),
    };
    // The loop's exit depends on the read (a spin read): a relaxed one acquires the word's own
    // memory.
    let read = if read.order == MemoryOrder::Relaxed {
        o.reload
    } else {
        read
    };
    let body: LoopFn<'_, ()> = Box::new(move |fb, ctx, lc| {
        let old = fb.atomic_load(p, o.scope, read)?;
        let cur = f.get(fb, old)?;
        if compare {
            let c = o.comparator.ok_or_else(dangling)?;
            let differs = fb.binary(BinOp::INotEqual, cur, c)?;
            let fail: ArmFn<'_, ()> = Box::new(move |fb, _| {
                fb.store(out, cur, MemAccess::default())?;
                lc.break_(fb)?;
                Ok(ArmEnd::Exits)
            });
            fb.if_(ctx, differs, fail, None)?;
        }
        let v = next(fb, f, o, cur)?;
        let new = f.put(fb, old, v)?;
        let seen = fb.atomic_compare_exchange(p, o.scope, success, Semantics::RELAXED, new, old)?;
        let done = fb.binary(BinOp::IEqual, seen, old)?;
        let exit: ArmFn<'_, ()> = Box::new(move |fb, _| {
            fb.store(out, cur, MemAccess::default())?;
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(ctx, done, exit, None)?;
        Ok(())
    });
    fb.loop_(&mut (), LoopSpec::default(), body)?;
    fb.load(out, MemAccess::default())
}
