//! The narrow integers whose high container bits no use reads ([`drop_unread_wraps`]).
//!
//! A narrow integer (`i8`, `i16`) is held zero-extended in a wider container (`legal::Held::Int`).
//! An add, subtract, multiply or shift left can carry into the container's high bits, and a
//! signed divide, remainder or arithmetic shift right leaves them sign-extended, so
//! `ops::int_binary` clears them with a mask after each. The low bits of each such result depend on
//! nothing but the low bits of its operands (and a shift's amount), so where no use of the result
//! reads its high bits the mask is work no answer depends on.
//!
//! A use reads only its operand's low bits where its own lowering does: an add, subtract,
//! multiply or shift left that masks its result, a signed divide, remainder or arithmetic shift
//! right (which sign-extend from the low bits first), a truncation or a sign extension (a
//! shift-left and arithmetic-shift-right pair, then a mask), and a `freeze` of a narrow integer
//! (a mask). A use passes the bits through where its result holds them as they are: a
//! bitwise and, or and xor, an add, subtract, multiply or shift left marked `nuw` (unmasked), a
//! `select`'s two values, a copy, a lane's extraction or insertion at a constant index, a
//! shuffle, and a phi; such a use reads its operand's high bits exactly where its own result's
//! are read. Every other use reads them: a comparison, a zero extension, a logical shift right,
//! an unsigned divide or remainder, a shift's amount, a conversion to float, a store, a call, a
//! return, an index, and any instruction another family lowers.

use crate::air::ir::{BinOp, Body, CastOp, InstId, InstKind, IntFlags, Value, ValueIndex};
use crate::budget::Meter;
use crate::error::Error;
use crate::lower::arith::ArithStrategy;
use crate::lower::strategy::Strategy;

/// How a use treats its operand's high container bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Use {
    /// It reads them.
    Reads,
    /// It holds them in its own result as they are.
    Passes,
    /// It reads only the low bits.
    Low,
}

/// The selected arith strategy of instruction `id`, if it has one.
fn arith(strategy: &[Option<Strategy>], values: ValueIndex, id: InstId) -> Option<&ArithStrategy> {
    let vid = values.inst(id)?;
    match strategy.get(vid.index())?.as_ref()? {
        Strategy::Arith(s) => Some(s),
        _ => None,
    }
}

/// Each value operand of instruction `kind` (selected as `s`; `None` for a phi) with its use.
fn uses(kind: &InstKind, s: Option<&ArithStrategy>, mut f: impl FnMut(Value, Use)) {
    use ArithStrategy as S;
    match (kind, s) {
        (InstKind::Phi { incoming, .. }, _) => {
            for &(v, _) in incoming.iter() {
                f(v, Use::Passes);
            }
        }
        (
            InstKind::BinOp {
                op, flags, a, b, ..
            },
            Some(S::Int { .. }),
        ) => {
            let nuw = flags.contains(IntFlags::NUW);
            let (x, y) = match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul if nuw => (Use::Passes, Use::Passes),
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::SDiv | BinOp::SRem => {
                    (Use::Low, Use::Low)
                }
                BinOp::Shl if nuw => (Use::Passes, Use::Reads),
                BinOp::Shl | BinOp::AShr => (Use::Low, Use::Reads),
                BinOp::And | BinOp::Or | BinOp::Xor => (Use::Passes, Use::Passes),
                _ => (Use::Reads, Use::Reads),
            };
            f(*a, x);
            f(*b, y);
        }
        (
            InstKind::Cast {
                op: CastOp::Trunc | CastOp::SExt,
                v,
            },
            Some(S::Cast { .. }),
        )
        | (InstKind::Freeze(v), Some(S::Freeze(_))) => f(*v, Use::Low),
        (InstKind::Freeze(v), Some(S::Copy)) => f(*v, Use::Passes),
        (InstKind::Select { c, t, f: e, .. }, Some(S::Select)) => {
            f(*c, Use::Reads);
            f(*t, Use::Passes);
            f(*e, Use::Passes);
        }
        (InstKind::ExtractElement { vec, idx }, Some(S::ExtractConst(_))) => {
            f(*vec, Use::Passes);
            f(*idx, Use::Reads);
        }
        (InstKind::InsertElement { vec, elt, idx }, Some(S::InsertConst(_))) => {
            f(*vec, Use::Passes);
            f(*elt, Use::Passes);
            f(*idx, Use::Reads);
        }
        (InstKind::ShuffleVector { a, b, .. }, Some(S::Shuffle(_))) => {
            f(*a, Use::Passes);
            f(*b, Use::Passes);
        }
        _ => kind.for_each_value(|v| f(v, Use::Reads)),
    }
}

/// Mark `v` read if it is an instruction's result, queueing it when it was not already.
fn mark(read: &mut [bool], work: &mut Vec<InstId>, v: Value) {
    let Value::Inst(i) = v else { return };
    if let Some(r) = read.get_mut(i.raw() as usize) {
        if !*r {
            *r = true;
            work.push(i);
        }
    }
}

/// Whether the high container bits of each instruction's result are read, by `InstId`, from the
/// selected strategies: read by a use that reads them, or passed through by one whose own result's
/// are read (the module docs). One pass over the body's instructions, then one over the results
/// it finds read, each queued once, when first found; the bitmap is charged to `m` and released by
/// the caller (`insts.len()` bytes).
fn high_bits_read(
    body: &Body,
    values: ValueIndex,
    strategy: &[Option<Strategy>],
    m: &mut Meter,
) -> Result<Vec<bool>, Error> {
    let n = body.insts.len();
    let work_bytes = n.saturating_mul(std::mem::size_of::<InstId>());
    m.charge(n.saturating_add(work_bytes))?;
    let mut read = vec![false; n];
    let mut work: Vec<InstId> = Vec::with_capacity(n);
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        uses(&inst.kind, arith(strategy, values, id), |v, u| {
            if u == Use::Reads {
                mark(&mut read, &mut work, v);
            }
        });
    }
    // A result read reads what its use passes through to it.
    while let Some(id) = work.pop() {
        m.tick(1)?;
        let Some(inst) = body.insts.get(id) else {
            continue;
        };
        uses(&inst.kind, arith(strategy, values, id), |v, u| {
            if u == Use::Passes {
                mark(&mut read, &mut work, v);
            }
        });
    }
    drop(work);
    m.release(work_bytes);
    Ok(read)
}

/// Drop the wrap of every narrow add, subtract, multiply and shift left not marked `nuw`, and of
/// every narrow signed divide, remainder and arithmetic shift right, whose result's high
/// container bits no use reads ([`high_bits_read`]): its `ArithStrategy::Int` keeps `wrap` false,
/// so `ops::int_binary` leaves the container's high bits as the operation left them.
pub fn drop_unread_wraps(
    body: &Body,
    values: ValueIndex,
    strategy: &mut [Option<Strategy>],
    m: &mut Meter,
) -> Result<(), Error> {
    let read = high_bits_read(body, values, strategy, m)?;
    for id in body.insts.ids() {
        m.tick(1)?;
        if read.get(id.raw() as usize).copied().unwrap_or(true) {
            continue;
        }
        let Some(vid) = values.inst(id) else {
            continue;
        };
        if let Some(Some(Strategy::Arith(ArithStrategy::Int {
            op,
            flags,
            form,
            wrap,
        }))) = strategy.get_mut(vid.index())
        {
            let wraps = match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Shl => !flags.contains(IntFlags::NUW),
                BinOp::SDiv | BinOp::SRem | BinOp::AShr => true,
                _ => false,
            };
            if wraps && form.legalized() {
                *wrap = false;
            }
        }
    }
    drop(read);
    m.release(body.insts.len());
    Ok(())
}
