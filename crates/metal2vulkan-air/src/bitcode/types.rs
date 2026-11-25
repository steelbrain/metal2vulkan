//! The `TYPE_BLOCK_NEW` type table.
//!
//! Records are read into a raw table first, because identified structs and typed pointers may
//! refer to later entries. Each entry is then interned into the IR's [`TypeTable`] with its
//! pointee erased, in dependency order. Erasure removes the only legal source of cycles (a struct
//! reaching itself through a pointer), so a cycle that remains is malformed input. The pointee of
//! each typed pointer is kept here, by bitcode type index, for the decode-only `PointeeTable`.
//!
//! [`TypeIds`] extends the table with the type ids LLVM's reader gives values the table has no
//! entry for (`getVirtualTypeID`, `llvm/lib/Bitcode/Reader/BitcodeReader.cpp:1532` at
//! `llvmorg-23.1.2`): a pointer to a known element, as a global, a function, an `alloca` or a
//! `getelementptr` yields. Each value of a function body carries such an id, so a record that
//! states no type of its own (a typed-pointer `load`, `call` or `getelementptr` without an
//! explicit type, pre-3.7 writers) takes it from its pointer operand's contained type, as LLVM's
//! reader does (`getContainedTypeID`, line 1510).

use std::collections::HashMap;

use super::codes::*;
use super::stream::Stream;
use crate::budget::Meter;
use crate::error::{BitcodeError, InputError, InputErrorKind, Location};
use crate::error::{Error, Feature, Limit};
use crate::ir::{AddrSpace, Symbols, Type, TypeId, TypeTable};

/// One bitcode type: its erased IR type and, for a typed pointer, the bitcode index of its
/// pointee.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BcType {
    /// The erased IR type; `None` when the IR keeps none, because only a marker declaration
    /// decode removed used it (`markers`).
    pub ty: Option<TypeId>,
    /// `Some` only for a typed (non-opaque) pointer.
    pub pointee: Option<u32>,
}

/// The module's type table by bitcode type index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BitcodeTypes {
    entries: Vec<BcType>,
    /// The bitcode return and parameter type indices of every function type, by bitcode index
    /// (sorted).
    fn_sigs: Vec<(u32, u32, Box<[u32]>)>,
    /// The element of every array and vector, and the fields of every struct, by bitcode index
    /// (sorted).
    aggregates: Vec<(u32, Box<[u32]>)>,
}

impl BitcodeTypes {
    /// The entry at bitcode index `idx`.
    pub fn get(&self, idx: u64) -> Option<BcType> {
        usize::try_from(idx)
            .ok()
            .and_then(|i| self.entries.get(i))
            .copied()
    }

    /// The erased IR type at bitcode index `idx`.
    pub fn ty(&self, idx: u64) -> Option<TypeId> {
        self.get(idx).and_then(|t| t.ty)
    }

    /// The bitcode index of the pointee of the typed pointer at `idx`.
    pub fn pointee(&self, idx: u64) -> Option<u32> {
        self.get(idx).and_then(|t| t.pointee)
    }

    /// The bitcode parameter type indices of the function type at `idx` (typed pointers kept,
    /// so a parameter's pointee is `pointee(params[i])`).
    pub fn params(&self, idx: u64) -> Option<&[u32]> {
        self.sig(idx).map(|(p, _)| p)
    }

    /// The bitcode return type index of the function type at `idx`.
    pub fn ret(&self, idx: u64) -> Option<u32> {
        self.sig(idx).map(|(_, r)| r)
    }

    /// Contained type `i` of the entry at `idx`, as LLVM's reader records it: a typed pointer's
    /// pointee (0), a function type's return (0) and parameters (1..), an array's or vector's
    /// element (0), a struct's fields.
    pub(crate) fn contained(&self, idx: u32, i: u32) -> Option<u32> {
        let i = usize::try_from(i).ok()?;
        if let Some(p) = self.pointee(u64::from(idx)) {
            return (i == 0).then_some(p);
        }
        if let Some((params, ret)) = self.sig(u64::from(idx)) {
            return match i.checked_sub(1) {
                None => Some(ret),
                Some(p) => params.get(p).copied(),
            };
        }
        let at = self
            .aggregates
            .binary_search_by_key(&idx, |(i, _)| *i)
            .ok()?;
        self.aggregates.get(at)?.1.get(i).copied()
    }

    fn sig(&self, idx: u64) -> Option<(&[u32], u32)> {
        let idx = u32::try_from(idx).ok()?;
        let at = self
            .fn_sigs
            .binary_search_by_key(&idx, |(i, _, _)| *i)
            .ok()?;
        self.fn_sigs.get(at).map(|(_, r, p)| (&**p, *r))
    }

    /// The number of entries.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Every entry in bitcode order.
    #[cfg(test)]
    pub fn iter(&self) -> impl Iterator<Item = &BcType> + '_ {
        self.entries.iter()
    }

    /// The table with every IR type renumbered by `new` (`None`: the IR no longer has it).
    pub(super) fn renumbered(self, new: impl Fn(TypeId) -> Option<TypeId>) -> BitcodeTypes {
        let BitcodeTypes {
            mut entries,
            fn_sigs,
            aggregates,
        } = self;
        for e in entries.iter_mut() {
            e.ty = e.ty.and_then(&new);
        }
        BitcodeTypes {
            entries,
            fn_sigs,
            aggregates,
        }
    }
}

/// A type record before interning; child references are bitcode indices.
#[derive(Clone, Debug)]
enum Raw {
    Leaf(Type),
    Ptr {
        space: AddrSpace,
        pointee: Option<u32>,
    },
    Vector {
        elem: u32,
        len: u32,
    },
    Array {
        elem: u32,
        len: u64,
    },
    Struct {
        fields: Box<[u32]>,
        packed: bool,
        name: Option<crate::ir::Sym>,
    },
    Function {
        ret: u32,
        params: Box<[u32]>,
        vararg: bool,
    },
}

impl Raw {
    /// Call `f` on every child the erased type needs interned first (pointees excluded: an
    /// erased pointer depends on nothing).
    fn for_each_child(&self, mut f: impl FnMut(u32) -> Result<(), Error>) -> Result<(), Error> {
        match self {
            Raw::Vector { elem, .. } | Raw::Array { elem, .. } => f(*elem),
            Raw::Struct { fields, .. } => fields.iter().try_for_each(|&c| f(c)),
            Raw::Function { ret, params, .. } => {
                f(*ret)?;
                params.iter().try_for_each(|&c| f(c))
            }
            Raw::Leaf(_) | Raw::Ptr { .. } => Ok(()),
        }
    }
}

/// Decode a `TYPE_BLOCK_NEW` that was just entered, interning every entry into `types`.
pub(crate) fn decode(
    s: &mut Stream<'_>,
    types: &mut TypeTable,
    symbols: &mut Symbols,
    m: &mut Meter,
) -> Result<BitcodeTypes, Error> {
    const B: u32 = TYPE_BLOCK;
    let mut raw: Vec<Raw> = Vec::new();
    let mut declared: Option<u64> = None;
    let mut pending_name = None;
    s.records(m, B, |s, code, _, m| {
        let entry = match code {
            TYPE_NUMENTRY => {
                let n = s.op(B, code, 0)?;
                m.cap(Limit::TypeCount, n)?;
                if declared.is_some() || !raw.is_empty() {
                    return Err(s.malformed(B, code));
                }
                declared = Some(n);
                return Ok(());
            }
            TYPE_STRUCT_NAME => {
                let name = s.chars(m, B, code, 0)?;
                pending_name = Some(symbols.intern(&name, m)?);
                return Ok(());
            }
            TYPE_VOID => Raw::Leaf(Type::Void),
            TYPE_FLOAT => Raw::Leaf(Type::Float),
            TYPE_DOUBLE => Raw::Leaf(Type::Double),
            TYPE_LABEL => Raw::Leaf(Type::Label),
            TYPE_HALF => Raw::Leaf(Type::Half),
            TYPE_BFLOAT => Raw::Leaf(Type::BFloat),
            TYPE_METADATA => Raw::Leaf(Type::Metadata),
            TYPE_INTEGER => {
                let width = s.op(B, code, 0)?;
                match u8::try_from(width) {
                    Ok(w @ 1..=64) => Raw::Leaf(Type::Int(w)),
                    // LLVM integers are 1..=2^23 bits; wider than 64 is valid but not modeled.
                    _ if (65..=(1 << 23)).contains(&width) => {
                        return Err(s.unsupported(Feature::Type { code }))
                    }
                    _ => return Err(s.malformed(B, code)),
                }
            }
            TYPE_POINTER => Raw::Ptr {
                pointee: Some(s.op32(B, code, 0)?),
                space: space(s, code, s.ops().get(1).copied().unwrap_or(0))?,
            },
            TYPE_OPAQUE_POINTER => Raw::Ptr {
                pointee: None,
                space: space(s, code, s.op(B, code, 0)?)?,
            },
            TYPE_OPAQUE => Raw::Leaf(Type::OpaqueStruct {
                name: pending_name.take(),
            }),
            TYPE_ARRAY => Raw::Array {
                len: s.op(B, code, 0)?,
                elem: s.op32(B, code, 1)?,
            },
            TYPE_VECTOR => {
                if s.ops().get(2).is_some_and(|&scalable| scalable != 0) {
                    return Err(s.unsupported(Feature::Type { code }));
                }
                Raw::Vector {
                    len: s.op32(B, code, 0)?,
                    elem: s.op32(B, code, 1)?,
                }
            }
            TYPE_STRUCT_ANON | TYPE_STRUCT_NAMED => {
                let packed = s.op(B, code, 0)? != 0;
                let fields = indices(s, m, code, 1)?;
                let name = if code == TYPE_STRUCT_NAMED {
                    pending_name.take()
                } else {
                    None
                };
                Raw::Struct {
                    fields,
                    packed,
                    name,
                }
            }
            TYPE_FUNCTION => Raw::Function {
                vararg: s.op(B, code, 0)? != 0,
                ret: s.op32(B, code, 1)?,
                params: indices(s, m, code, 2)?,
            },
            c if TYPE_UNSUPPORTED.contains(&c) => return Err(s.unsupported(Feature::Type { code })),
            _ => return Err(s.err(BitcodeError::UnknownRecord { block: B, code })),
        };
        if declared.is_none_or(|n| raw.len() as u64 >= n) {
            return Err(s.malformed(B, code));
        }
        grow(m, &mut raw)?;
        raw.push(entry);
        Ok(())
    })?;
    if declared.is_some_and(|n| raw.len() as u64 != n) {
        return Err(bad_type(s.location()));
    }
    let at = s.location();
    let erased = intern_all(&raw, types, m, &at)?;
    let mut entries = Vec::new();
    m.reserve(&mut entries, raw.len())?;
    let mut fn_sigs = Vec::new();
    let mut aggregates = Vec::new();
    for (i, (r, ty)) in raw.iter().zip(erased).enumerate() {
        if let Raw::Function { ret, params, .. } = r {
            grow(m, &mut fn_sigs)?;
            m.charge(std::mem::size_of_val::<[u32]>(params))?;
            fn_sigs.push((i as u32, *ret, params.clone()));
        }
        let children = match r {
            Raw::Vector { elem, .. } | Raw::Array { elem, .. } => Some(Box::from([*elem])),
            Raw::Struct { fields, .. } => Some(fields.clone()),
            _ => None,
        };
        if let Some(children) = children {
            grow(m, &mut aggregates)?;
            m.charge(std::mem::size_of_val::<[u32]>(&children))?;
            aggregates.push((i as u32, children));
        }
        let pointee = match r {
            Raw::Ptr {
                pointee: Some(p), ..
            } => {
                if *p as usize >= raw.len() {
                    return Err(bad_type(at));
                }
                Some(*p)
            }
            _ => None,
        };
        entries.push(BcType {
            ty: Some(ty),
            pointee,
        });
    }
    Ok(BitcodeTypes {
        entries,
        fn_sigs,
        aggregates,
    })
}

/// No type id: what LLVM's reader calls `InvalidTypeID`.
pub(crate) const NO_TYPE_ID: u32 = u32::MAX;

/// LLVM's reader's type ids: the table's indices, then the ids it makes for types the table has
/// no entry for (`getVirtualTypeID`), one per erased type and first contained id, numbered after
/// the table. Only a contained id and its erased type are ever read back, so a type with nothing
/// contained (a comparison's `i1` result) needs no id of its own: it is [`NO_TYPE_ID`], whose
/// contained ids are none, as are those of LLVM's id for it.
#[derive(Debug, Default)]
pub(crate) struct TypeIds {
    /// The erased type and the contained ids of each id past the table.
    made: Vec<(TypeId, Box<[u32]>)>,
    /// The id made for each `(erased type, first contained id)`.
    by_key: HashMap<(TypeId, u32), u32>,
}

impl TypeIds {
    /// Contained id `i` of id `id` ([`NO_TYPE_ID`] where there is none).
    pub(crate) fn contained(&self, table: &BitcodeTypes, id: u32, i: u32) -> u32 {
        let len = table.entries.len() as u32;
        let found = match id.checked_sub(len) {
            None => table.contained(id, i),
            Some(k) => self
                .made
                .get(k as usize)
                .and_then(|(_, c)| c.get(i as usize))
                .copied(),
        };
        found.unwrap_or(NO_TYPE_ID)
    }

    /// The erased type of id `id`.
    pub(crate) fn ty(&self, table: &BitcodeTypes, id: u32) -> Option<TypeId> {
        let len = table.entries.len() as u32;
        match id.checked_sub(len) {
            None => table.ty(u64::from(id)),
            Some(k) => self.made.get(k as usize).map(|(t, _)| *t),
        }
    }

    /// The id of erased type `ty` whose contained ids are `children`, made on first use. With
    /// nothing contained, or a first contained id that is none, it is [`NO_TYPE_ID`].
    pub(crate) fn id(
        &mut self,
        table: &BitcodeTypes,
        ty: TypeId,
        children: &[u32],
        m: &mut Meter,
    ) -> Result<u32, Error> {
        let Some(&first) = children.first().filter(|&&c| c != NO_TYPE_ID) else {
            return Ok(NO_TYPE_ID);
        };
        if let Some(&id) = self.by_key.get(&(ty, first)) {
            return Ok(id);
        }
        let id = u32::try_from(table.entries.len() + self.made.len())
            .ok()
            .filter(|&id| id != NO_TYPE_ID)
            .ok_or(Error::LimitExceeded(Limit::TypeCount))?;
        grow(m, &mut self.made)?;
        m.charge(std::mem::size_of_val::<[u32]>(children))?;
        m.charge(std::mem::size_of::<((TypeId, u32), u32)>().saturating_mul(2))?;
        self.made.push((ty, children.into()));
        self.by_key.insert((ty, first), id);
        Ok(id)
    }
}

/// Intern every raw entry in dependency order (iterative post-order; no recursion on untrusted
/// nesting). Returns the erased id of each entry.
fn intern_all(
    raw: &[Raw],
    types: &mut TypeTable,
    m: &mut Meter,
    at: &Location,
) -> Result<Vec<TypeId>, Error> {
    const NEW: u8 = 0;
    const OPEN: u8 = 1;
    const DONE: u8 = 2;
    let n = raw.len();
    let mut state: Vec<u8> = Vec::new();
    m.reserve(&mut state, n)?;
    state.resize(n, NEW);
    let mut out: Vec<Option<TypeId>> = Vec::new();
    m.reserve(&mut out, n)?;
    out.resize(n, None);
    let mut stack: Vec<(u32, bool)> = Vec::new();
    for root in 0..n {
        if state.get(root) == Some(&DONE) {
            continue;
        }
        grow(m, &mut stack)?;
        stack.push((root as u32, false));
        while let Some((i, expanded)) = stack.pop() {
            m.tick(1)?;
            let r = raw.get(i as usize).ok_or_else(|| bad_type(at.clone()))?;
            let slot = state
                .get_mut(i as usize)
                .ok_or_else(|| bad_type(at.clone()))?;
            if expanded {
                *slot = DONE;
                let ty = erase(r, &out, at)?;
                let id = types.intern(ty, m).map_err(|e| match e {
                    Error::InvalidInput(_) => bad_type(at.clone()),
                    other => other,
                })?;
                if let Some(o) = out.get_mut(i as usize) {
                    *o = Some(id);
                }
                continue;
            }
            match *slot {
                DONE => continue,
                OPEN => return Err(bad_type(at.clone())),
                _ => {}
            }
            *slot = OPEN;
            grow(m, &mut stack)?;
            stack.push((i, true));
            r.for_each_child(|c| {
                match state.get(c as usize) {
                    None => return Err(bad_type(at.clone())),
                    Some(&DONE) => {}
                    Some(_) => {
                        grow(m, &mut stack)?;
                        stack.push((c, false));
                    }
                }
                Ok(())
            })?;
        }
    }
    let mut ids = Vec::new();
    m.reserve(&mut ids, n)?;
    for t in out {
        ids.push(t.ok_or_else(|| bad_type(at.clone()))?);
    }
    Ok(ids)
}

/// The erased IR type of `r`, whose children are already interned in `out`.
fn erase(r: &Raw, out: &[Option<TypeId>], at: &Location) -> Result<Type, Error> {
    let get = |i: u32| {
        out.get(i as usize)
            .copied()
            .flatten()
            .ok_or_else(|| bad_type(at.clone()))
    };
    Ok(match r {
        Raw::Leaf(t) => t.clone(),
        Raw::Ptr { space, .. } => Type::Ptr { space: *space },
        Raw::Vector { elem, len } => Type::Vector {
            elem: get(*elem)?,
            len: *len,
        },
        Raw::Array { elem, len } => Type::Array {
            elem: get(*elem)?,
            len: *len,
        },
        Raw::Struct {
            fields,
            packed,
            name,
        } => Type::Struct {
            fields: fields.iter().map(|&f| get(f)).collect::<Result<_, _>>()?,
            packed: *packed,
            name: *name,
        },
        Raw::Function {
            ret,
            params,
            vararg,
        } => Type::Function {
            ret: get(*ret)?,
            params: params.iter().map(|&p| get(p)).collect::<Result<_, _>>()?,
            vararg: *vararg,
        },
    })
}

fn space(s: &Stream<'_>, code: u32, raw: u64) -> Result<AddrSpace, Error> {
    let n = u32::try_from(raw).map_err(|_| s.malformed(TYPE_BLOCK, code))?;
    AddrSpace::from_raw(n).map_err(|u| s.unsupported(u.feature))
}

/// Operands from `start` as `u32` type indices.
fn indices(s: &Stream<'_>, m: &mut Meter, code: u32, start: usize) -> Result<Box<[u32]>, Error> {
    let ops = s.ops().get(start..).unwrap_or(&[]);
    let mut out = Vec::new();
    m.reserve(&mut out, ops.len())?;
    for &o in ops {
        out.push(u32::try_from(o).map_err(|_| s.malformed(TYPE_BLOCK, code))?);
    }
    Ok(out.into_boxed_slice())
}

fn bad_type(at: Location) -> Error {
    InputError::new(InputErrorKind::Bitcode(BitcodeError::BadType), at).into()
}

/// Charge for and reserve room for one more element when `v` is full (doubling).
pub(crate) fn grow<T>(m: &mut Meter, v: &mut Vec<T>) -> Result<(), Error> {
    if v.len() == v.capacity() {
        let additional = v.len().max(4);
        m.reserve(v, additional)?;
    }
    Ok(())
}
