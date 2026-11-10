//! Who reads each instruction of an AIR body, as a table indexed by instruction: the selection's
//! and the regroupings' users, read by instruction number where a sorted map was searched for each;
//! and which of the entry's parameters the body reads ([`params_used`]).

use std::collections::BTreeSet;

use crate::air::ir::{Body, Idx, InstId, Value};

/// The parameters `body` uses: each argument an instruction names, a call's operand included. A
/// parameter of a specialized entry whose reads its constants fold away (deleted with what nothing
/// observes) is not among them. This is the lowering's reading of a used parameter: what it plans,
/// and what the compile decides an input's attachment state, ordering and refusals from.
pub(crate) fn params_used(body: &Body) -> BTreeSet<u32> {
    let mut used = BTreeSet::new();
    for inst in body.insts.values() {
        inst.kind.for_each_value(|v| {
            if let Value::Arg(a) = v {
                used.insert(a.raw());
            }
        });
    }
    used
}

/// The users of each instruction of a body: `list[start[i]..start[i + 1]]` are instruction `i`'s,
/// each listed once per operand naming it, in the body's instruction order (the order a sorted
/// map of lists filled in that order holds them in).
pub(crate) struct Users {
    start: Vec<u32>,
    list: Vec<InstId>,
}

impl Users {
    /// The users of every instruction of `body`, counted, then placed.
    pub(crate) fn of(body: &Body) -> Users {
        let n = body.insts.len();
        let mut start = vec![0u32; n + 1];
        for inst in body.insts.values() {
            inst.kind.for_each_value(|v| {
                if let Value::Inst(d) = v {
                    if let Some(c) = start.get_mut(d.index() + 1) {
                        *c += 1;
                    }
                }
            });
        }
        for k in 1..start.len() {
            let prev = start.get(k - 1).copied().unwrap_or(0);
            if let Some(c) = start.get_mut(k) {
                *c += prev;
            }
        }
        let mut fill = start.get(..n).unwrap_or(&[]).to_vec();
        let mut list = vec![InstId::new(0); start.last().copied().unwrap_or(0) as usize];
        for (at, inst) in body.insts.iter() {
            inst.kind.for_each_value(|v| {
                if let Value::Inst(d) = v {
                    if let Some(next) = fill.get_mut(d.index()) {
                        if let Some(slot) = list.get_mut(*next as usize) {
                            *slot = at;
                        }
                        *next += 1;
                    }
                }
            });
        }
        Users { start, list }
    }

    /// The users of `id`, in instruction order (none for an instruction outside the body).
    pub(crate) fn get(&self, id: InstId) -> &[InstId] {
        let at = |k: usize| self.start.get(k).copied().unwrap_or(0) as usize;
        let (lo, hi) = (at(id.index()), at(id.index() + 1));
        self.list.get(lo..hi).unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::Users;
    use crate::air::ir::{
        ArgIdx, BinOp, Body, FastMath, Inst, InstId, InstKind, IntFlags, TypeId, Value,
    };

    #[test]
    fn users_are_listed_per_operand_in_instruction_order() {
        let mut m = crate::budget::Meter::new(&crate::budget::Budget::default());
        let mut body = Body::with_capacity(3, 1, &mut m).expect("body");
        let ty = TypeId::new(0);
        let add = |a: Value, b: Value| Inst {
            ty,
            kind: InstKind::BinOp {
                op: BinOp::Add,
                flags: IntFlags::NONE,
                fmf: FastMath::NONE,
                a,
                b,
            },
        };
        let arg = Value::Arg(ArgIdx::new(0));
        let i0 = body.push_inst(add(arg, arg), &mut m).expect("i0");
        let i1 = body
            .push_inst(add(Value::Inst(i0), Value::Inst(i0)), &mut m)
            .expect("i1");
        let i2 = body
            .push_inst(add(Value::Inst(i1), Value::Inst(i0)), &mut m)
            .expect("i2");
        let users = Users::of(&body);
        assert_eq!(users.get(i0), &[i1, i1, i2]);
        assert_eq!(users.get(i1), &[i2]);
        assert!(users.get(i2).is_empty());
        assert!(users.get(InstId::new(7)).is_empty());
    }
}
