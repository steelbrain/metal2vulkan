//! What the access family reads from a buffer pointer's root set, read once per set.
//!
//! Every access, operand, and representation of a buffer pointer asks what its root set holds:
//! its candidate slots and null-ness (`access::buffer_roots`), whether it points into the
//! constant blob (`blob::is_blob`), a wide pointer's static slots (`dynamic::plan_entry`), and
//! the elements a buffer-table load reads (`buffer_table::element`). The memory plan interns the
//! sets, and each answer is a function of the set and the plan alone, so [`plan`] reads every
//! distinct set a buffer pointer of the function names once, before any instruction is
//! selected: one work unit per pointer value and per root read, the table charged to the ledger
//! while selection runs. Each question is then one lookup instead of a walk of the roots per
//! use, which was quadratic in a function whose many accesses go through a pointer of many roots.
//!
//! A texture or sampler handle's root set is read the same way: its domain (a heap handle, or its
//! static candidates, `memory::handles::domain`) is a function of the set, the handle's class, and,
//! for a handle-array element, the value's offset range and handle size, so [`plan`] computes it
//! once per distinct key, metered as the walk is (`handles::domain`), and keeps its [`DomainKind`]
//! (or its refusal). A representation, operand, or heap check of a handle is then one lookup; only
//! a use that holds the candidates (a switch-at-use) walks them, metered, again.

use std::collections::BTreeMap;

use crate::air::ir::Value;
use crate::air::Entry;
use crate::budget::{tree_bytes, Meter};
use crate::error::Error;
use crate::lower::select::SelectCx;
use crate::memory::handles::{self, DomainKind};
use crate::memory::plan::MemoryPlan;
use crate::memory::provenance::{self, HandleClass, OffsetRange, Repr, Root, RootSetId};

use super::{blob, refuse, slot_root, Target};

/// The facts of every root set a buffer pointer of the function names, by set.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RootSetFacts {
    sets: Vec<Option<SetFacts>>,
    /// The domain of every texture or sampler handle an instruction of the function yields, by
    /// [`HandleKey`].
    handles: BTreeMap<HandleKey, Result<DomainKind, Error>>,
    /// The ledger bytes the table holds.
    charged: usize,
}

/// What a handle instruction's domain is a function of: its root set and class, its handle size,
/// and, when its set is one root, its offset range (an array element's range names its
/// elements), as `(tag, lo, hi)`.
type HandleKey = (RootSetId, HandleClass, Option<u64>, (u8, i64, i64));

/// The [`HandleKey`] of instruction value `v`, a `class` handle of root set `cands`.
fn handle_key(
    e: &Entry,
    mp: &MemoryPlan,
    v: Value,
    class: HandleClass,
    cands: RootSetId,
) -> HandleKey {
    let range = match mp.roots(cands) {
        [_] => mp.offset(v).unwrap_or(OffsetRange::Top),
        _ => OffsetRange::Top,
    };
    let range = match range {
        OffsetRange::Empty => (0, 0, 0),
        OffsetRange::Span { lo, hi } => (1, lo, hi),
        OffsetRange::Top => (2, 0, 0),
    };
    (cands, class, handles::handle_size(e, mp, v), range)
}

/// Whether this module lowers handles of `class` (`handle::lowered_class`).
fn lowered_handle(class: HandleClass) -> bool {
    matches!(class, HandleClass::Texture | HandleClass::Sampler)
}

/// What one root set holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SetFacts {
    /// The candidates of a static-root pointer of the set (`access::buffer_roots`), or the
    /// refusal its first root that is none meets.
    pub(super) statics: Result<Statics, StaticRefusal>,
    /// Every root is a blob-form constant global, and there is one (`blob::is_blob`).
    pub(super) blob: bool,
    /// Some root is a blob-form constant global.
    pub(super) any_blob: bool,
    /// The static slots of a wide pointer of the set, every one with a window; `None` when a
    /// root is neither null, dynamic, nor such a slot (`dynamic::plan_entry`).
    pub(super) wide_slots: Option<Box<[usize]>>,
    /// Whether `wide_slots` were added to the dynamic plan's slots.
    pub(super) merged: bool,
    /// The buffer-table elements the set names (`buffer_table::element`).
    pub(super) table: TableSet,
}

/// The candidate slots of a static-root pointer: every buffer root's slot, sorted by root
/// number, whether the pointer may be null, and whether it may point into the blob.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Statics {
    pub(super) targets: Box<[Target]>,
    pub(super) null: bool,
    pub(super) in_blob: bool,
}

/// Why a set's pointer has no static candidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StaticRefusal {
    /// A root with no static slot: not lowered.
    NotLowered,
    /// A buffer root's slot index out of the plan: an invariant violation.
    MissingSlot,
    /// An intersection's primitive-data pointer.
    PrimitiveData,
}

impl StaticRefusal {
    /// The error a selection that needs the set's candidates returns.
    pub(super) fn error(self, cx: &SelectCx<'_>) -> Error {
        match self {
            StaticRefusal::NotLowered => refuse(),
            StaticRefusal::MissingSlot => cx.invariant("a buffer root names a missing slot"),
            StaticRefusal::PrimitiveData => crate::lower::raytracing::primitive_data_read(),
        }
    }
}

/// The buffer-table elements of a set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TableSet {
    /// No root, or the null root alone: a read wholly outside the table.
    Null,
    /// Elements `lo..=hi` (`count` of them, sorted) of the table parameter `param`, and nothing
    /// else.
    Elements {
        param: u32,
        lo: u32,
        hi: u32,
        count: usize,
    },
    /// Any other set.
    Other,
}

impl RootSetFacts {
    /// The facts of set `id`, when a buffer pointer names it.
    pub(super) fn get(&self, id: RootSetId) -> Option<&SetFacts> {
        self.sets.get(id.index()).and_then(Option::as_ref)
    }

    /// [`RootSetFacts::get`], mutably.
    pub(super) fn get_mut(&mut self, id: RootSetId) -> Option<&mut SetFacts> {
        self.sets.get_mut(id.index()).and_then(Option::as_mut)
    }

    /// The ledger bytes the table holds, which selection returns when it ends.
    pub fn charged(&self) -> usize {
        self.charged
    }

    /// The domain of instruction value `v`, a texture or sampler handle of root set `cands` and
    /// class `class`, as [`plan`] read it; `None` when it did not (no such handle).
    pub(super) fn handle(
        &self,
        e: &Entry,
        mp: &MemoryPlan,
        v: Value,
        class: HandleClass,
        cands: RootSetId,
    ) -> Option<&Result<DomainKind, Error>> {
        self.handles.get(&handle_key(e, mp, v, class, cands))
    }
}

/// Read every root set a buffer pointer of the function names, once each ([`RootSetFacts`]).
pub fn plan(e: &Entry, mp: &MemoryPlan, m: &mut Meter) -> Result<RootSetFacts, Error> {
    let mut facts = RootSetFacts::default();
    let result = fill(e, mp, &mut facts, m);
    match result {
        Ok(()) => Ok(facts),
        Err(err) => {
            m.release(facts.charged);
            Err(err)
        }
    }
}

fn fill(e: &Entry, mp: &MemoryPlan, facts: &mut RootSetFacts, m: &mut Meter) -> Result<(), Error> {
    let n = mp.sets.len();
    let table = n.saturating_mul(std::mem::size_of::<Option<SetFacts>>());
    m.charge(table)?;
    facts.charged = table;
    facts.sets.resize(n, None);
    for (i, repr) in mp.repr.iter().enumerate() {
        m.tick(1)?;
        if let Some(Repr::Handle { class, cands }) = repr {
            read_handle(e, mp, (i, *class, *cands), facts, m)?;
            continue;
        }
        let Some(Repr::BufPtr { roots, .. }) = repr else {
            continue;
        };
        if facts.get(*roots).is_some() {
            continue;
        }
        let set = read(e, mp, mp.roots(*roots), m)?;
        let bytes = set
            .statics
            .as_ref()
            .map_or(0, |s| s.targets.len() * std::mem::size_of::<Target>())
            + set
                .wide_slots
                .as_ref()
                .map_or(0, |s| s.len() * std::mem::size_of::<usize>());
        m.charge(bytes)?;
        facts.charged = facts.charged.saturating_add(bytes);
        if let Some(slot) = facts.sets.get_mut(roots.index()) {
            *slot = Some(set);
        }
    }
    Ok(())
}

/// Read the domain of the value of index `i` in the memory plan, a `class` handle of root set
/// `cands`, when it is an instruction's texture or sampler handle and its key is not read yet.
fn read_handle(
    e: &Entry,
    mp: &MemoryPlan,
    (i, class, cands): (usize, HandleClass, RootSetId),
    facts: &mut RootSetFacts,
    m: &mut Meter,
) -> Result<(), Error> {
    let v = crate::air::ir::ValueId::from_index(i).and_then(|id| mp.values.value(id));
    let Some(v @ Value::Inst(_)) = v else {
        return Ok(());
    };
    if !lowered_handle(class) {
        return Ok(());
    }
    let key = handle_key(e, mp, v, class, cands);
    if facts.handles.contains_key(&key) {
        return Ok(());
    }
    let kind = match handles::domain(e, mp, v, class, m) {
        Ok(d) => Ok(d.kind()),
        // The meter's stops end the attempt; a refusal is the handle's, kept for its uses.
        Err(err @ (Error::LimitExceeded(_) | Error::Cancelled | Error::Budget(_))) => {
            return Err(err)
        }
        Err(err) => Err(err),
    };
    let bytes = tree_bytes(
        1,
        std::mem::size_of::<(HandleKey, Result<DomainKind, Error>)>(),
    );
    m.charge(bytes)?;
    facts.charged = facts.charged.saturating_add(bytes);
    facts.handles.insert(key, kind);
    Ok(())
}

/// The facts of one set, read in its order: one work unit per root.
fn read(e: &Entry, mp: &MemoryPlan, roots: &[Root], m: &mut Meter) -> Result<SetFacts, Error> {
    let mut targets = Vec::new();
    let (mut null, mut in_blob) = (false, false);
    let mut statics: Option<StaticRefusal> = None;
    let mut wide: Option<Vec<usize>> = Some(Vec::new());
    let mut blob_globals = 0usize;
    let mut table = TableSet::Null;
    for (i, &r) in roots.iter().enumerate() {
        m.tick(1)?;
        let blob_global = matches!(r, Root::Global(g) if blob::global_offset(mp, g).is_some());
        blob_globals += usize::from(blob_global);
        let slot = mp.slot_of_root(r);
        // The static candidates, up to the first root that has none.
        if statics.is_none() {
            match r {
                Root::Null => null = true,
                Root::Global(_) if blob_global => in_blob = true,
                Root::Param(_) | Root::BufferElement { .. } => match slot {
                    None => statics = Some(StaticRefusal::NotLowered),
                    Some(slot) => match mp.slots.get(slot) {
                        None => statics = Some(StaticRefusal::MissingSlot),
                        Some(plan) => targets.push(Target {
                            slot,
                            root: slot_root(plan.slot.index),
                        }),
                    },
                },
                Root::Call(_) if provenance::primitive_data_root(e, r) => {
                    statics = Some(StaticRefusal::PrimitiveData)
                }
                _ => statics = Some(StaticRefusal::NotLowered),
            }
        }
        // A wide pointer's static slots: each with a window.
        if let Some(slots) = wide.as_mut() {
            match r {
                Root::Null | Root::Dynamic => {}
                Root::Param(_) | Root::BufferElement { .. } => {
                    match slot.filter(|s| mp.slots.get(*s).is_some_and(|p| p.window.is_some())) {
                        Some(s) => slots.push(s),
                        None => wide = None,
                    }
                }
                _ => wide = None,
            }
        }
        // The table's elements: one table's, sorted as the set is.
        table = match (table, r) {
            (TableSet::Null, Root::Null) if roots.len() == 1 => TableSet::Null,
            (TableSet::Null, Root::BufferElement { param, element }) if i == 0 => {
                TableSet::Elements {
                    param,
                    lo: element,
                    hi: element,
                    count: 1,
                }
            }
            (
                TableSet::Elements {
                    param, lo, count, ..
                },
                Root::BufferElement { param: p, element },
            ) if p == param => TableSet::Elements {
                param,
                lo,
                hi: element,
                count: count + 1,
            },
            _ => TableSet::Other,
        };
    }
    targets.sort_by_key(|t| t.root);
    Ok(SetFacts {
        statics: match statics {
            None => Ok(Statics {
                targets: targets.into_boxed_slice(),
                null,
                in_blob,
            }),
            Some(refusal) => Err(refusal),
        },
        blob: !roots.is_empty() && blob_globals == roots.len(),
        any_blob: blob_globals > 0,
        wide_slots: wide.map(Vec::into_boxed_slice),
        merged: false,
        table,
    })
}
