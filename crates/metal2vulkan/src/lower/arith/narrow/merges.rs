//! The merges of a widened `half` that a select took against a value it does not hold widened
//! with another widened value, followed through the composites that carry them
//! ([`merges_mixed_selects`], `narrow::moves_unfolded`).
//!
//! A select carried inside a composite (a texel read's `{T, i8}`, a struct, an array, a vector)
//! and merged with it is taken for the measured merge of its members: not measured through a
//! composite, but its members' merges are the measured shape once a compiler splits it into them.
//! The analysis therefore asks, member by member, what a value holds: a path of
//! literal indices names a member (a struct's member, an array's element, a matrix's column, a
//! vector's lane), the empty path the value itself. A construct, an insert, an extract and a copy
//! carry each member through unchanged; a select and a phi take each member from their operands.

use super::has_bit;
use crate::budget::Meter;
use crate::error::Error;
use crate::spirv::ids::{BlockId, Value};
use crate::spirv::ir::{Inst, Type, TypedModule, UnOp};
use std::collections::HashMap;

/// The most steps one question of a value's member follows ([`Members::widened_at`],
/// [`Members::mixed_at`]): a bound on the walk. Past it a member counts as widened wherever its
/// value is widened, so a merge is never missed for it.
const STEPS: usize = 256;

/// The most members of one value's type the merge checks look at; a type with more counts as
/// merging wherever two of its incoming values are widened.
const NODES: usize = 256;

/// The bytes one value's record holds while the analysis runs, as the meter charges it: the
/// entry and the map's slack.
const RECORD_BYTES: usize = std::mem::size_of::<(Value, Made<'static>)>() + 16;

/// How module `m`'s value was made, where it carries its operands' members.
enum Made<'a> {
    Construct(&'a [Value]),
    Insert {
        object: Value,
        composite: Value,
        indices: &'a [u32],
    },
    Extract {
        composite: Value,
        indices: &'a [u32],
    },
    Copy(Value),
    Select {
        t: Value,
        f: Value,
    },
    Phi(&'a [(Value, BlockId)]),
}

/// The values of a module that carry members, and its widened values.
struct Members<'a> {
    m: &'a TypedModule,
    made: HashMap<Value, Made<'a>>,
    widened: &'a [u64],
}

impl<'a> Members<'a> {
    fn new(m: &'a TypedModule, widened: &'a [u64]) -> Members<'a> {
        let mut made = HashMap::new();
        for b in m.blocks() {
            for phi in &b.phis {
                made.insert(phi.result, Made::Phi(&phi.incoming));
            }
            for n in &b.insts {
                let Some(r) = n.result else { continue };
                let carried = match &n.inst {
                    Inst::CompositeConstruct { parts } => Made::Construct(parts),
                    Inst::CompositeInsert {
                        object,
                        composite,
                        indices,
                    } => Made::Insert {
                        object: *object,
                        composite: *composite,
                        indices,
                    },
                    Inst::CompositeExtract { composite, indices } => Made::Extract {
                        composite: *composite,
                        indices,
                    },
                    Inst::Unary {
                        op: UnOp::CopyObject,
                        a,
                    } => Made::Copy(*a),
                    Inst::Select { t, f, .. } => Made::Select { t: *t, f: *f },
                    _ => continue,
                };
                made.insert(r, carried);
            }
        }
        Members { m, made, widened }
    }

    /// The type of value `v`.
    fn ty(&self, v: Value) -> Option<&'a Type> {
        let ty = self.m.value(v).ok()?.ty;
        self.m.types().get(ty).ok()
    }

    /// The lanes of value `v`: a vector's, 1 for a scalar.
    fn lanes(&self, v: Value) -> u32 {
        match self.ty(v) {
            Some(Type::Vector { len, .. }) => u32::from(*len),
            _ => 1,
        }
    }

    /// The part of construct `v` (of `parts`) that holds member `path`, and the path inside it:
    /// a vector's lane is in the part that supplies it, every other composite's member is its
    /// part. `None` for the empty path or one out of range.
    fn part(&self, v: Value, parts: &[Value], path: &[u32]) -> Option<(Value, Vec<u32>)> {
        let (&first, rest) = path.split_first()?;
        if !matches!(self.ty(v), Some(Type::Vector { .. })) {
            return Some((*parts.get(first as usize)?, rest.to_vec()));
        }
        let mut lane = first;
        for &p in parts {
            let n = self.lanes(p);
            if lane < n {
                let inside = match matches!(self.ty(p), Some(Type::Vector { .. })) {
                    true => [&[lane][..], rest].concat(),
                    false => rest.to_vec(),
                };
                return Some((p, inside));
            }
            lane -= n;
        }
        None
    }

    /// Whether member `path` of `v` holds a widened value, wholly or in part: the member of a
    /// construct's part, an insert's object or composite, an extract's composite, a copy's
    /// operand, a select's or a phi's operands, else (a value made otherwise) `v`'s own mark.
    fn widened_at(&self, v: Value, path: &[u32]) -> bool {
        let mut seen = Vec::new();
        let mut steps = STEPS;
        self.widened_in(v, path.to_vec(), &mut seen, &mut steps)
    }

    fn widened_in(
        &self,
        v: Value,
        path: Vec<u32>,
        seen: &mut Vec<Value>,
        steps: &mut usize,
    ) -> bool {
        let own = has_bit(self.widened, v);
        if *steps == 0 {
            return own;
        }
        *steps -= 1;
        match self.made.get(&v) {
            Some(Made::Construct(parts)) => match self.part(v, parts, &path) {
                Some((p, inside)) => self.widened_in(p, inside, seen, steps),
                None => own,
            },
            Some(Made::Insert {
                object,
                composite,
                indices,
            }) => {
                let inside = path.strip_prefix(*indices).map(<[u32]>::to_vec);
                if path.is_empty() {
                    own
                } else if let Some(inside) = inside {
                    self.widened_in(*object, inside, seen, steps)
                } else if indices.starts_with(&path) {
                    self.widened_in(*object, Vec::new(), seen, steps)
                        || self.widened_in(*composite, path, seen, steps)
                } else {
                    self.widened_in(*composite, path, seen, steps)
                }
            }
            Some(Made::Extract { composite, indices }) => {
                let full = [&indices[..], &path[..]].concat();
                self.widened_in(*composite, full, seen, steps)
            }
            Some(Made::Copy(a)) => self.widened_in(*a, path, seen, steps),
            Some(Made::Select { t, f }) => {
                self.widened_in(*t, path.clone(), seen, steps)
                    || self.widened_in(*f, path, seen, steps)
            }
            Some(Made::Phi(incoming)) => {
                if seen.contains(&v) {
                    return false;
                }
                seen.push(v);
                incoming
                    .iter()
                    .any(|(i, _)| self.widened_in(*i, path.clone(), seen, steps))
            }
            None => own,
        }
    }

    /// Whether member `path` of `v` is a select of a widened value against a value it does not
    /// hold widened, carried there by constructs, inserts, extracts and copies (a phi carries
    /// none: a merge then a select keeps every NaN's bits, measured). A select of composites
    /// selects each member: its member `path` is such a select where one operand's member is
    /// widened and the other's is not.
    fn mixed_at(&self, v: Value, path: &[u32]) -> bool {
        let mut steps = STEPS;
        let mut at = (v, path.to_vec());
        loop {
            let (v, path) = at;
            if steps == 0 {
                return has_bit(self.widened, v);
            }
            steps -= 1;
            at = match self.made.get(&v) {
                Some(Made::Select { t, f }) => {
                    return self.widened_at(*t, &path) != self.widened_at(*f, &path);
                }
                Some(Made::Construct(parts)) => match self.part(v, parts, &path) {
                    Some(inside) => inside,
                    None => return false,
                },
                Some(Made::Insert {
                    object,
                    composite,
                    indices,
                }) => {
                    let inside = path.strip_prefix(*indices).map(<[u32]>::to_vec);
                    if let (false, Some(inside)) = (path.is_empty(), inside) {
                        (*object, inside)
                    } else if indices.starts_with(&path) {
                        return false;
                    } else {
                        (*composite, path)
                    }
                }
                Some(Made::Extract { composite, indices }) => {
                    (*composite, [&indices[..], &path[..]].concat())
                }
                Some(Made::Copy(a)) => (*a, path),
                Some(Made::Phi(_)) | None => return false,
            };
        }
    }

    /// The paths of every member of `v`'s type, itself (the empty path) first: a struct's
    /// members, a sized array's elements, a matrix's columns and a vector's lanes, and theirs.
    /// `None` past [`NODES`].
    fn paths(&self, v: Value) -> Option<Vec<Vec<u32>>> {
        let ty = self.m.value(v).ok()?.ty;
        let mut out = Vec::new();
        let mut open = vec![(ty, Vec::new())];
        while let Some((ty, path)) = open.pop() {
            if out.len() == NODES {
                return None;
            }
            let types = self.m.types();
            let members: Vec<_> = match types.get(ty).ok()? {
                Type::Struct { members, .. } => members.to_vec(),
                Type::Vector { elem, len } => vec![*elem; usize::from(*len)],
                Type::Matrix { col, cols } => vec![*col; usize::from(*cols)],
                Type::Array { elem, len, .. } => match self.m.consts().array_len(*len, types) {
                    Ok(Some(n)) if n as usize <= NODES => vec![*elem; n as usize],
                    Ok(Some(_)) => return None,
                    _ => Vec::new(),
                },
                _ => Vec::new(),
            };
            for (i, member) in members.into_iter().enumerate().rev() {
                open.push((member, [&path[..], &[i as u32]].concat()));
            }
            out.push(path);
        }
        Some(out)
    }

    /// Whether one of `values`, at index `at`, holds a mixed select at a member where another of
    /// them holds a widened value: their merge is the measured one. The others are every other
    /// index, or with `distinct` every other value (a phi's incoming value twice is no merge with
    /// itself). Only widened values take part: a value holding a widened member is widened. A
    /// type of more than [`NODES`] members merges wherever two of them are widened.
    fn merges(&self, values: &[Value], at: usize, distinct: bool) -> bool {
        let Some(&s) = values.get(at).filter(|s| has_bit(self.widened, **s)) else {
            return false;
        };
        let others = || {
            values.iter().enumerate().filter(move |&(j, v)| {
                has_bit(self.widened, *v) && if distinct { *v != s } else { j != at }
            })
        };
        let Some(paths) = self.paths(s) else {
            return others().next().is_some();
        };
        paths
            .iter()
            .any(|path| self.mixed_at(s, path) && others().any(|(_, v)| self.widened_at(*v, path)))
    }
}

/// Whether module `m` merges a widened value (one of `widened`) that a select took against a value
/// it does not hold widened with another widened value (`narrow::moves_unfolded`): a phi with
/// such a select among its incoming values and another widened one, or a function variable
/// (`local_float`) such a select is stored to and another widened value too, member by member
/// (the module docs). Measured on the M3 Max under macOS 27.0.1 (2026-11-03), every `half` bit
/// pattern widened, moved and narrowed by Metal's compiler, with and without fast math: a merge of
/// `b ? float(h) : 0.0` with `float(h2)`, or with `b ? float(h2) : 0.0`, answers 2045 of the 2046
/// NaNs `0x7E00`, where the select alone, the merge of widened values alone, a select of two
/// widened values merged with a third, the select merged with a constant, a merge then a select,
/// two selects, two merges, the select carried to a later block, and a select of the `half` bits
/// before the widening each keep every NaN's bits. One pass over the instructions to record the
/// values that carry members and the stores, [`RECORD_BYTES`] per instruction and phi charged to
/// `meter` while it runs, then the questions of each phi's and each such variable's values.
pub(super) fn merges_mixed_selects(
    m: &TypedModule,
    widened: &[u64],
    local_float: &dyn Fn(Value) -> bool,
    meter: &mut Meter,
) -> Result<bool, Error> {
    let records: usize = m
        .blocks()
        .iter()
        .map(|b| b.phis.len() + b.insts.len())
        .sum();
    let bytes = records.saturating_mul(RECORD_BYTES);
    meter.charge(bytes)?;
    let members = Members::new(m, widened);
    // The widened values stored to each function variable, in order. Each store is one of the
    // merged values, where each of a phi's incoming values is: two stores of one select to a
    // variable merge it with itself, and a phi's two edges of one value do not.
    let mut stores: HashMap<Value, Vec<Value>> = HashMap::new();
    for n in m.blocks().iter().flat_map(|b| &b.insts) {
        if let Inst::Store { ptr, value, .. } = &n.inst {
            if has_bit(widened, *value) && local_float(*ptr) {
                stores.entry(*ptr).or_default().push(*value);
            }
        }
    }
    let stored = stores
        .values()
        .any(|values| (0..values.len()).any(|i| members.merges(values, i, false)));
    let merged = stored
        || m.blocks().iter().flat_map(|b| &b.phis).any(|phi| {
            if !has_bit(widened, phi.result) {
                return false;
            }
            let values: Vec<Value> = phi.incoming.iter().map(|(v, _)| *v).collect();
            (0..values.len()).any(|i| members.merges(&values, i, true))
        });
    drop((members, stores));
    meter.release(bytes);
    Ok(merged)
}
