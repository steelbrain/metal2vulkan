//! What a pointer phi's incoming values are, read once per phi.
//!
//! A pointer phi's lowered representation refuses some incoming values, and each family that owns
//! a representation asks about its own: a static fat pointer refuses a constant other than null
//! and a blob pointer (`access::lowered_pointer`), a wide pointer anything neither null nor wide
//! (`dynamic::lowered_pointer`), and a handle, an arena or typed-object offset, and a blob offset
//! any constant (`handle`, `local`, `blob`). Every read of the phi's representation (each
//! operand, each block-crossing use) asked again, a walk of the incoming values per read, which
//! was quadratic in a function whose many uses read a phi of many incoming values. Each answer is a
//! function of the body, the memory plan, and the root sets selection read before any instruction
//! (`root_sets`), so [`plan`] reads every pointer phi's incoming values once, before any
//! instruction is selected: one work unit per phi and per incoming value, the table charged while
//! selection runs.

use std::collections::BTreeMap;

use crate::air::ir::{InstId, InstKind, Type, Value};
use crate::budget::{tree_bytes, Meter};
use crate::error::Error;
use crate::lower::select::SelectCx;

use super::{blob, dynamic, pointer};

/// The incoming values of every pointer phi of the function, by phi.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PhiFacts {
    phis: BTreeMap<InstId, Incoming>,
    /// The ledger bytes the table holds.
    charged: usize,
}

/// What one pointer phi's incoming values hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Incoming {
    /// Some incoming value is a constant.
    pub(super) constant: bool,
    /// Some incoming value is a constant other than the null pointer (or an undefined one),
    /// whatever its space.
    pub(super) not_null: bool,
    /// The first incoming value, in order, that a static fat pointer refuses (a constant other
    /// than null, or a blob pointer), or whose root set selection did not read.
    pub(super) fat: Fat,
    /// Some incoming value is neither the null constant nor a wide pointer (a constant, or a
    /// pointer that is not wide).
    pub(super) not_wide: bool,
}

/// What a static fat pointer phi meets first among its incoming values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fat {
    /// No refused incoming value.
    Plain,
    /// A constant other than null, or a blob pointer: refused.
    Refused,
    /// A buffer pointer whose root set selection did not read: an invariant violation.
    Unread,
}

impl PhiFacts {
    /// The ledger bytes the table holds, which selection returns when it ends.
    pub fn charged(&self) -> usize {
        self.charged
    }
}

/// Read every pointer phi's incoming values in the entry function, once each.
pub fn plan(cx: &SelectCx<'_>, m: &mut Meter) -> Result<PhiFacts, Error> {
    let mut facts = PhiFacts::default();
    let result = fill(cx, &mut facts, m);
    match result {
        Ok(()) => Ok(facts),
        Err(err) => {
            m.release(facts.charged);
            Err(err)
        }
    }
}

fn fill(cx: &SelectCx<'_>, facts: &mut PhiFacts, m: &mut Meter) -> Result<(), Error> {
    let entry = std::mem::size_of::<(InstId, Incoming)>();
    for (id, inst) in cx.body.insts.iter() {
        let InstKind::Phi { incoming, .. } = &inst.kind else {
            continue;
        };
        if !matches!(cx.entry.module.types().get(inst.ty), Some(Type::Ptr { .. })) {
            continue;
        }
        m.tick(1)?;
        let mut constant = false;
        let mut not_null = false;
        let mut fat = None;
        let mut not_wide = false;
        for (x, _) in incoming.iter() {
            m.tick(1)?;
            let is_const = matches!(x, Value::Const(_));
            let null = pointer::null_constant(cx, *x);
            constant |= is_const;
            not_null |= is_const && !pointer::null_pointer_constant(cx, *x);
            not_wide |= !null && (is_const || !dynamic::is_wide(cx, *x));
            if fat.is_none() {
                if is_const && !null {
                    fat = Some(Fat::Refused);
                } else {
                    match blob::is_blob(cx, *x) {
                        Ok(true) => fat = Some(Fat::Refused),
                        Ok(false) => {}
                        Err(_) => fat = Some(Fat::Unread),
                    }
                }
            }
        }
        let bytes = tree_bytes(1, entry);
        m.charge(bytes)?;
        facts.charged = facts.charged.saturating_add(bytes);
        facts.phis.insert(
            id,
            Incoming {
                constant,
                not_null,
                fat: fat.unwrap_or(Fat::Plain),
                not_wide,
            },
        );
    }
    Ok(())
}

/// The incoming values of `v` when it is a phi (`None` for any other value). A phi of a pointer
/// representation the table does not hold is an invariant violation: every representation is a
/// pointer's, and [`plan`] read every pointer phi.
pub(super) fn incoming<'c>(cx: &'c SelectCx<'_>, v: Value) -> Result<Option<&'c Incoming>, Error> {
    let Value::Inst(i) = v else {
        return Ok(None);
    };
    if !matches!(
        cx.body.insts.get(i).map(|x| &x.kind),
        Some(InstKind::Phi { .. })
    ) {
        return Ok(None);
    }
    cx.phis
        .phis
        .get(&i)
        .map(Some)
        .ok_or_else(|| cx.invariant("a pointer phi's incoming values were not read"))
}

/// The error a static fat pointer phi meets among its incoming values (`access::lowered_pointer`),
/// if any.
pub(super) fn fat_refusal(cx: &SelectCx<'_>, i: &Incoming) -> Option<Error> {
    match i.fat {
        Fat::Plain => None,
        Fat::Refused => Some(super::refuse()),
        // `blob::is_blob`'s error for a set selection did not read.
        Fat::Unread => Some(cx.invariant("a buffer pointer's root set was not read")),
    }
}
