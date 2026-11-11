//! Typed allocas and closed constant `Private` globals, with sized-array clamps.
//!
//! A **typed object** is an alloca the memory plan types (every access is a plain load or store
//! of exactly the subobject at its offset shape) or a closed, type-consistent constant global
//! (`ConstGlobalForm::Private`). It is one `Private` variable of its own type, an array of `count`
//! elements (1 for a global) of the allocated type rebuilt from the AIR type ([`TypedTy`]: arrays
//! stay arrays, so a dynamic index is an access-chain index; leaves are the arithmetic family's
//! held forms). A constant global's variable carries its initializer ([`Init`]). An alloca whose
//! simdgroup matrices hold only fragments holds each matrix as its lane's two elements
//! ([`TypedNode::Fragment`], `memory::fragment`).
//!
//! AIR types and constants are interned, so a type's members share types and an initializer's
//! parts share constants. The shape and the initializer keep one node per distinct type (and
//! constant), and are built, declared, and charged per node, never per path: a struct holding
//! another twice, nested `d` deep, is `d + 1` nodes, not `2^d` leaves.
//!
//! A pointer into a typed object is its `u32` byte offset from the object's start, like an arena
//! pointer (`DemoteTy::ArenaOffset`). An access resolves its offset **shape** (a constant, or a
//! constant plus a multiple of the element size of each of up to [`MAX_DYNAMIC`] nested array
//! levels) to a [`TypedPath`]: constant indices and dynamic levels, outermost first. Each dynamic
//! index is recovered from the pointer's offset level by level, as `(rem - start) / stride`, with
//! `start` the byte offset of the level's element 0 inside the enclosing level's element (or the
//! object) and `rem` the offset inside that element; a negative or wrapped index is a huge unsigned
//! one.
//!
//! **Clamps** (`GuestIsolation::On`): an access never leaves its object. By construction, not a
//! pass. An index within its level is used as is, so an in-range access is the access Metal makes.
//! An index past its level's length (a negative or wrapped one is a huge unsigned one) is
//! undefined in Metal, whose own answer is whatever its memory holds at the address:
//!
//! - in a thread's own object (an alloca) the access is checked, every dynamic level's `idx < len`:
//!   a load past a level reads zero, and a store past one is dropped, so it changes no element of
//!   the object, as Metal's store past an array changes none of the array's elements (it lands
//!   past them);
//! - in a threadgroup or constant object each dynamic level's index is `min(idx, len - 1)`: an
//!   index past it reaches the level's last element, a load reading that element and a store
//!   writing it.
//!
//! Without isolation the indices are used as is.
//!
//! Where the access's `getelementptr` indexes the levels itself, its own index operands are the
//! levels' indices (`lower::access::local::lined_up`): within their levels they are the indices
//! the offset recovers, and a loop-invariant one stays loop-invariant, so its clamp or check
//! hoists out of the loop.

use std::collections::BTreeMap;

use super::fragment;
use crate::air::ir::{Const, ConstId as AirConstId, Type, TypeId as AirTypeId, TypeLayouts, Value};
use crate::air::Entry;
use crate::budget::{tree_bytes, Meter};
use crate::device::DeviceProfile;
use crate::error::{Error, Feature, LowerFamily, Unsupported};
use crate::lower::arith::{self, LoweredConst};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::{BResult, BuildError, Dangling};
use crate::spirv::ids::{ConstId, DemoteTy, ScalarTy, TypeId};
use crate::spirv::ir::{ConstBits, Constant, Type as SType};

fn not_lowered() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::Access)).into()
}

/// The deepest type nesting a typed object may have.
pub(super) const MAX_DEPTH: u32 = 64;

/// The most unshared type size the typed objects of one function may declare: the sum, over the
/// nodes of each object's [`TypedTy`], of the node's size as a tree whose arrays are counted once
/// ([`TypedTy::unshared`]). SPIR-V declares each type once, but a consumer may walk a declared
/// type's members without remembering what it has walked: `spirv-val`'s type pass does (SPIRV-Tools
/// 2026.3), taking 1.2 s on a 20-level struct DAG and four times longer per two levels. An object
/// past this size is held untyped (the private arena, or the constant blob).
pub const MAX_UNSHARED: u64 = 1 << 16;

/// One distinct AIR type of a [`TypedTy`] in SPIR-V shape. Its members are nodes of the same
/// `TypedTy`, by index, each before it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypedNode {
    /// A scalar or vector in the arithmetic family's held form.
    Leaf(DemoteTy),
    /// A simdgroup matrix's `<64 x T>` held as its lane's fragment, elements 0 and 1, in this
    /// 2-lane vector of the matrix's held element (`memory::fragment`).
    Fragment(DemoteTy),
    /// An array of `len` elements of node `elem`.
    Array { elem: u32, len: u32 },
    /// A struct of these member nodes, in member order.
    Struct(Box<[u32]>),
}

/// The SPIR-V shape of a typed object's type, rebuilt from its AIR type.
///
/// AIR types are interned, so an aggregate's members share types: `%e{k} = { %e{k-1}, %e{k-1} }`
/// nested `d` deep has `2^d` paths through `d + 1` types. The shape keeps one [`TypedNode`] per
/// distinct AIR type the type reaches, each after the members it refers to, and the type itself
/// last. Building it ([`typed_ty`]), declaring its SPIR-V types ([`spirv_types`]), and reading and
/// declaring an initializer ([`init_of`], [`declare_init`]) cost its types and member edges, never
/// its paths. Arrays stay arrays, so a dynamic index is an access-chain index.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TypedTy {
    nodes: Vec<TypedNode>,
    /// The sum of the nodes' unshared sizes ([`TypedTy::unshared`]).
    unshared: u64,
    /// The ledger bytes the nodes hold.
    bytes: usize,
}

impl TypedTy {
    /// The nodes, each after the members it refers to; the type itself is the last.
    #[cfg(test)]
    pub fn nodes(&self) -> &[TypedNode] {
        &self.nodes
    }

    /// The sum, over the nodes, of each node's size as a tree whose arrays are counted once (a
    /// scalar or vector is 1, an array 1 more than its element, a struct 1 more than its
    /// members together), saturating: what walking each declared type's members without a record
    /// of the types walked costs ([`MAX_UNSHARED`]).
    pub fn unshared(&self) -> u64 {
        self.unshared
    }

    /// Return the ledger bytes the shape holds, dropping it.
    pub fn release(self, m: &mut Meter) {
        m.release(self.bytes);
    }

    /// Whether some node is a simdgroup matrix held as its fragment.
    pub fn has_fragments(&self) -> bool {
        self.nodes
            .iter()
            .any(|n| matches!(n, TypedNode::Fragment(_)))
    }

    /// The index of the type itself.
    fn root(&self) -> Option<u32> {
        u32::try_from(self.nodes.len().checked_sub(1)?).ok()
    }

    /// An upper bound on the bytes one object of this shape takes, its leaves in their held forms
    /// (saturating): a scalar its width, at least 4 bytes (a `bool` too), a vector its scalar times
    /// its lanes rounded up to a power of two, an array its length times its element, and a struct
    /// its members each at the next multiple of its alignment, the whole rounded up to the largest;
    /// a leaf is aligned to its size, up to 16, as MSL lays such values out. One pass over the
    /// nodes, each after its members, its record charged to the ledger while it lives.
    pub fn held_bytes(&self, m: &mut Meter) -> Result<u64, Error> {
        fn leaf(d: &DemoteTy) -> u64 {
            let scalar = |s: &ScalarTy| match s {
                ScalarTy::Bool => 4,
                ScalarTy::Int { bits } | ScalarTy::Float { bits } => {
                    u64::from(bits.div_ceil(8)).max(4)
                }
            };
            match d {
                DemoteTy::Scalar(s) => scalar(s),
                DemoteTy::Vector { elem, len } => {
                    scalar(elem).saturating_mul(u64::from(len.next_power_of_two()))
                }
                DemoteTy::Composite(parts) => parts.iter().fold(0u64, |sum, p| {
                    let size = leaf(p);
                    round_to(sum, size.next_power_of_two().min(16)).saturating_add(size)
                }),
                DemoteTy::FatPtr { .. } => 16,
                DemoteTy::ArenaOffset | DemoteTy::HandleIndex => 4,
            }
        }
        fn round_to(v: u64, align: u64) -> u64 {
            v.checked_next_multiple_of(align.max(1)).unwrap_or(u64::MAX)
        }
        // (size, alignment) of each node.
        let mut sized: Vec<(u64, u64)> = Vec::new();
        m.reserve_exact(&mut sized, self.nodes.len())?;
        let at =
            |sized: &[(u64, u64)], i: u32| sized.get(i as usize).copied().unwrap_or((u64::MAX, 16));
        for n in &self.nodes {
            let s = match n {
                TypedNode::Leaf(d) | TypedNode::Fragment(d) => {
                    let size = leaf(d);
                    (size, size.next_power_of_two().min(16))
                }
                TypedNode::Array { elem, len } => {
                    let (size, align) = at(&sized, *elem);
                    (round_to(size, align).saturating_mul(u64::from(*len)), align)
                }
                TypedNode::Struct(members) => {
                    let mut end = 0u64;
                    let mut align = 1u64;
                    for m in members.iter() {
                        let (size, a) = at(&sized, *m);
                        end = round_to(end, a).saturating_add(size);
                        align = align.max(a);
                    }
                    (round_to(end, align), align)
                }
            };
            sized.push(s);
        }
        let bytes = sized.last().map_or(0, |(size, _)| *size);
        m.release(crate::budget::vec_bytes(&sized));
        Ok(bytes)
    }
}

/// What [`typed_ty`] knows of an AIR type it has built.
#[derive(Clone, Copy, Debug)]
struct Built {
    node: u32,
    /// The most types on one path from this type down, itself included.
    height: u32,
    /// Its size as a tree whose arrays are counted once (saturating).
    size: u64,
}

/// The ledger bytes one entry of a build's record of built types holds.
fn built_entry_bytes() -> usize {
    tree_bytes(1, std::mem::size_of::<(AirTypeId, Built)>())
}

/// One build of a [`TypedTy`]: the nodes so far, and which AIR type each is.
struct Shaper<'a> {
    e: &'a Entry,
    p: &'a DeviceProfile,
    /// A simdgroup matrix's `<64 x T>` is a [`TypedNode::Fragment`].
    fragments: bool,
    nodes: Vec<TypedNode>,
    /// The sum of the built nodes' sizes.
    unshared: u64,
    /// The ledger bytes `nodes` holds.
    bytes: usize,
    built: BTreeMap<AirTypeId, Built>,
}

/// The typed shape of AIR type `ty` on profile `p`. `Ok(Err(refusal))` is a type with a member no
/// typed object can hold (a pointer, an opaque type, an empty or over-long array, a chunked vector
/// but, with `fragments`, a simdgroup matrix's `<64 x T>`, held as its fragment) or nested deeper
/// than `MAX_DEPTH` (64 types below `ty`); an outer error is a budget stop.
///
/// Each distinct type is built once, and each type reached (the type itself, and every member of
/// a type built) is one work unit. The nodes stay on the ledger while the shape lives
/// ([`TypedTy::release`]); the record of built types is released on return. A type reached again
/// keeps its node, and is refused, as building it again would be, where its deepest member would
/// pass `MAX_DEPTH`: the walk meets the refusal the tree walk met, at the same first type.
pub fn typed_ty(
    e: &Entry,
    p: &DeviceProfile,
    ty: AirTypeId,
    fragments: bool,
    m: &mut Meter,
) -> Result<Result<TypedTy, Error>, Error> {
    let mut s = Shaper {
        e,
        p,
        fragments,
        nodes: Vec::new(),
        unshared: 0,
        bytes: 0,
        built: BTreeMap::new(),
    };
    let built = s.at(ty, 0, m);
    m.release(s.built.len().saturating_mul(built_entry_bytes()));
    match built {
        Ok(Ok(_)) => Ok(Ok(TypedTy {
            nodes: s.nodes,
            unshared: s.unshared,
            bytes: s.bytes,
        })),
        Ok(Err(refusal)) => {
            m.release(s.bytes);
            Ok(Err(refusal))
        }
        Err(stop) => {
            m.release(s.bytes);
            Err(stop)
        }
    }
}

impl Shaper<'_> {
    /// Build `ty`, reached `depth` types below the object's type.
    fn at(
        &mut self,
        ty: AirTypeId,
        depth: u32,
        m: &mut Meter,
    ) -> Result<Result<Built, Error>, Error> {
        m.tick(1)?;
        if depth > MAX_DEPTH {
            return Ok(Err(not_lowered()));
        }
        if let Some(b) = self.built.get(&ty) {
            // Its deepest member is `height - 1` types below it.
            return Ok(if depth + (b.height - 1) > MAX_DEPTH {
                Err(not_lowered())
            } else {
                Ok(*b)
            });
        }
        let Some(t) = self.e.module.types().get(ty) else {
            return Ok(Err(not_lowered()));
        };
        let (node, height, size) = match t {
            Type::Array { elem, len } => {
                let Some(len) = u32::try_from(*len).ok().filter(|n| *n > 0) else {
                    return Ok(Err(not_lowered()));
                };
                let elem = match self.at(*elem, depth + 1, m)? {
                    Ok(b) => b,
                    Err(refusal) => return Ok(Err(refusal)),
                };
                (
                    TypedNode::Array {
                        elem: elem.node,
                        len,
                    },
                    elem.height + 1,
                    elem.size.saturating_add(1),
                )
            }
            Type::Struct { fields, .. } if !fields.is_empty() => {
                let held = fields.len().saturating_mul(std::mem::size_of::<u32>());
                m.charge(held)?;
                self.bytes = self.bytes.saturating_add(held);
                let mut members = Vec::with_capacity(fields.len());
                let (mut height, mut size) = (0, 1u64);
                for f in fields.iter() {
                    let b = match self.at(*f, depth + 1, m)? {
                        Ok(b) => b,
                        Err(refusal) => return Ok(Err(refusal)),
                    };
                    members.push(b.node);
                    height = height.max(b.height);
                    size = size.saturating_add(b.size);
                }
                (
                    TypedNode::Struct(members.into_boxed_slice()),
                    height + 1,
                    size,
                )
            }
            Type::Int(_)
            | Type::Half
            | Type::BFloat
            | Type::Float
            | Type::Double
            | Type::Vector { .. } => match arith::lowered_type(self.e, self.p, ty) {
                Ok(d @ (DemoteTy::Scalar(_) | DemoteTy::Vector { .. })) => {
                    (TypedNode::Leaf(d), 1, 1)
                }
                Ok(d) if self.fragments && fragment::is_matrix(self.e.module.types(), ty) => {
                    match fragment::fragment_form(&d) {
                        Some(f) => (TypedNode::Fragment(f), 1, 1),
                        None => return Ok(Err(not_lowered())),
                    }
                }
                Ok(_) => return Ok(Err(not_lowered())),
                Err(refusal) => return Ok(Err(refusal)),
            },
            _ => return Ok(Err(not_lowered())),
        };
        let Ok(index) = u32::try_from(self.nodes.len()) else {
            return Ok(Err(not_lowered()));
        };
        let before = m.ledger();
        m.grow(&mut self.nodes, 1)?;
        let grown = usize::try_from(m.ledger().saturating_sub(before)).unwrap_or(usize::MAX);
        self.bytes = self.bytes.saturating_add(grown);
        m.charge(built_entry_bytes())?;
        self.nodes.push(node);
        self.unshared = self.unshared.saturating_add(size);
        let b = Built {
            node: index,
            height,
            size,
        };
        self.built.insert(ty, b);
        Ok(Ok(b))
    }
}

fn missing() -> BuildError {
    BuildError::Dangling(Dangling::Value(crate::spirv::ids::Value::new(u32::MAX)))
}

fn u32_const(mb: &mut ModuleBuilder, v: u32) -> BResult<ConstId> {
    let ty = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    mb.constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(v)),
    })
}

/// The SPIR-V type of `t`, and of each of its nodes in their order (no explicit layout: the
/// variables are `Private`). Each node is declared once, after its members.
pub fn spirv_types(mb: &mut ModuleBuilder, t: &TypedTy) -> BResult<(TypeId, Vec<TypeId>)> {
    let mut ids: Vec<TypeId> = Vec::with_capacity(t.nodes.len());
    for n in &t.nodes {
        let id = match n {
            TypedNode::Leaf(d) | TypedNode::Fragment(d) => {
                arith::spirv_type(mb, d)?.ok_or_else(missing)?
            }
            TypedNode::Array { elem, len } => {
                let elem = *ids.get(*elem as usize).ok_or_else(missing)?;
                let len = u32_const(mb, *len)?;
                mb.ty(SType::Array {
                    elem,
                    len,
                    stride: None,
                })?
            }
            TypedNode::Struct(members) => {
                let members = members
                    .iter()
                    .map(|i| ids.get(*i as usize).copied().ok_or_else(missing))
                    .collect::<BResult<Box<[TypeId]>>>()?;
                mb.ty(SType::Struct {
                    members,
                    layout: None,
                })?
            }
        };
        ids.push(id);
    }
    let root = *ids.last().ok_or_else(missing)?;
    Ok((root, ids))
}

/// One part of an [`Init`], in the held form of its node of the object's [`TypedTy`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum InitNode {
    Leaf(LoweredConst),
    /// The member parts, by index in the same `Init`, each before it.
    Parts(Box<[u32]>),
    /// All zero (`null`, `zeroinitializer`, `undef`, `poison`: an undefined initializer byte may
    /// be anything).
    Zero,
}

/// A constant initializer in the held forms of a [`TypedTy`]. Constants are interned like types, so
/// an aggregate's parts share constants; the initializer keeps one part per distinct constant and
/// node it reaches, each after its member parts, and the initializer itself last.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Init {
    /// Each part's node in the object's shape, and the part.
    nodes: Vec<(u32, InitNode)>,
}

impl Init {
    /// The parts, each after its member parts; the initializer itself is the last.
    #[cfg(test)]
    pub fn nodes(&self) -> &[(u32, InitNode)] {
        &self.nodes
    }
}

/// The ledger bytes one entry of a read's record of read parts holds.
fn read_entry_bytes() -> usize {
    tree_bytes(1, std::mem::size_of::<((AirConstId, u32), u32)>())
}

/// One read of an [`Init`]: the parts so far, and which constant and node each is.
struct InitReader<'a> {
    e: &'a Entry,
    t: &'a TypedTy,
    nodes: Vec<(u32, InitNode)>,
    /// The ledger bytes `nodes` holds.
    bytes: usize,
    read: BTreeMap<(AirConstId, u32), u32>,
}

/// The initializer `c` of a typed object of shape `t`. Each distinct pair of a constant and a node
/// is read once, and each pair reached (the initializer itself, and every member of a part read) is
/// one work unit. The parts stay on the ledger while the plan holds them; the record of read parts
/// is released on return, and everything on a refusal.
pub fn init_of(e: &Entry, c: AirConstId, t: &TypedTy, m: &mut Meter) -> Result<Init, Error> {
    let root = t.root().ok_or_else(not_lowered)?;
    let mut r = InitReader {
        e,
        t,
        nodes: Vec::new(),
        bytes: 0,
        read: BTreeMap::new(),
    };
    let read = r.at(c, root, m);
    m.release(r.read.len().saturating_mul(read_entry_bytes()));
    match read {
        Ok(_) => Ok(Init { nodes: r.nodes }),
        Err(err) => {
            m.release(r.bytes);
            Err(err)
        }
    }
}

impl InitReader<'_> {
    /// Read constant `c` as node `node` of the shape.
    fn at(&mut self, c: AirConstId, node: u32, m: &mut Meter) -> Result<u32, Error> {
        m.tick(1)?;
        if let Some(&i) = self.read.get(&(c, node)) {
            return Ok(i);
        }
        let (e, t) = (self.e, self.t);
        let k = e.module.consts().get(c).ok_or_else(not_lowered)?;
        let shape = t.nodes.get(node as usize).ok_or_else(not_lowered)?;
        let part = match (&k.value, shape) {
            // Only an alloca holds fragments, and an alloca has no initializer.
            (_, TypedNode::Fragment(_)) => return Err(not_lowered()),
            (Const::Null | Const::Undef | Const::Poison, _) => InitNode::Zero,
            (_, TypedNode::Leaf(d)) => {
                // A vector's lanes are held while the part lives, charged before they are built.
                let lanes = match d {
                    DemoteTy::Vector { len, .. } => {
                        usize::from(*len).saturating_mul(std::mem::size_of::<u64>())
                    }
                    _ => 0,
                };
                m.charge(lanes)?;
                self.bytes = self.bytes.saturating_add(lanes);
                match arith::constant_as(e, Value::Const(c), d) {
                    Some(LoweredConst::Undef(_)) => InitNode::Zero,
                    Some(l @ (LoweredConst::Scalar { .. } | LoweredConst::Vector { .. })) => {
                        InitNode::Leaf(l)
                    }
                    _ => return Err(not_lowered()),
                }
            }
            (Const::Aggregate(elems), TypedNode::Array { elem, len }) => {
                if elems.len() != *len as usize {
                    return Err(not_lowered());
                }
                InitNode::Parts(self.parts(elems, |_| Some(*elem), m)?)
            }
            (Const::Aggregate(elems), TypedNode::Struct(members)) => {
                if elems.len() != members.len() {
                    return Err(not_lowered());
                }
                InitNode::Parts(self.parts(elems, |i| members.get(i).copied(), m)?)
            }
            _ => return Err(not_lowered()),
        };
        let index = u32::try_from(self.nodes.len()).map_err(|_| not_lowered())?;
        let before = m.ledger();
        m.grow(&mut self.nodes, 1)?;
        let grown = usize::try_from(m.ledger().saturating_sub(before)).unwrap_or(usize::MAX);
        self.bytes = self.bytes.saturating_add(grown);
        m.charge(read_entry_bytes())?;
        self.nodes.push((node, part));
        self.read.insert((c, node), index);
        Ok(index)
    }

    /// Read the elements `elems` of an aggregate, element `i` as node `member(i)`.
    fn parts(
        &mut self,
        elems: &[AirConstId],
        member: impl Fn(usize) -> Option<u32>,
        m: &mut Meter,
    ) -> Result<Box<[u32]>, Error> {
        let held = elems.len().saturating_mul(std::mem::size_of::<u32>());
        m.charge(held)?;
        self.bytes = self.bytes.saturating_add(held);
        let mut parts = Vec::with_capacity(elems.len());
        for (i, x) in elems.iter().enumerate() {
            let node = member(i).ok_or_else(not_lowered)?;
            parts.push(self.at(*x, node, m)?);
        }
        Ok(parts.into_boxed_slice())
    }
}

/// The SPIR-V constant of initializer `i` of shape `t`, whose nodes' types are `types` (from
/// [`spirv_types`]). Each part is declared once, after its member parts.
pub fn declare_init(
    mb: &mut ModuleBuilder,
    t: &TypedTy,
    types: &[TypeId],
    i: &Init,
) -> BResult<ConstId> {
    let mut ids: Vec<ConstId> = Vec::with_capacity(i.nodes.len());
    for (node, part) in &i.nodes {
        let ty = *types.get(*node as usize).ok_or_else(missing)?;
        let id = match (part, t.nodes.get(*node as usize)) {
            (InitNode::Zero, _) => mb.constant(Constant::Null { ty })?,
            (InitNode::Leaf(LoweredConst::Scalar { bits, .. }), Some(TypedNode::Leaf(_))) => mb
                .constant(Constant::Scalar {
                    ty,
                    bits: ConstBits(*bits),
                })?,
            (InitNode::Leaf(LoweredConst::Vector { elem, lanes }), Some(TypedNode::Leaf(_))) => {
                let et = arith::spirv_type(mb, &DemoteTy::Scalar(*elem))?.ok_or_else(missing)?;
                let mut parts = Vec::with_capacity(lanes.len());
                for b in lanes.iter() {
                    parts.push(mb.constant(Constant::Scalar {
                        ty: et,
                        bits: ConstBits(*b),
                    })?);
                }
                mb.constant(Constant::Composite {
                    ty,
                    parts: parts.into(),
                })?
            }
            (InitNode::Parts(ps), Some(TypedNode::Array { .. } | TypedNode::Struct(_))) => {
                let parts = ps
                    .iter()
                    .map(|p| ids.get(*p as usize).copied().ok_or_else(missing))
                    .collect::<BResult<Box<[ConstId]>>>()?;
                mb.constant(Constant::Composite { ty, parts })?
            }
            _ => return Err(missing()),
        };
        ids.push(id);
    }
    ids.last().copied().ok_or_else(missing)
}

/// The most dynamic indices an offset shape keeps: one per nested array level a typed access
/// indexes dynamically.
pub const MAX_DYNAMIC: usize = 4;

/// The dynamic strides of an offset shape: distinct nonzero byte strides, largest first (at most
/// [`MAX_DYNAMIC`]). Two dynamic terms of one stride are one term (their indices add).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Strides([u64; MAX_DYNAMIC]);

impl Strides {
    /// The strides of one dynamic term.
    pub fn one(stride: u64) -> Option<Strides> {
        Strides([0; MAX_DYNAMIC]).with(stride)
    }

    /// These strides and `stride`, or `None` for a zero stride or more than [`MAX_DYNAMIC`].
    pub fn with(self, stride: u64) -> Option<Strides> {
        if stride == 0 {
            return None;
        }
        if self.0.contains(&stride) {
            return Some(self);
        }
        let mut out = self.0;
        *out.iter_mut().find(|s| **s == 0)? = stride;
        out.sort_unstable_by(|a, b| b.cmp(a));
        Some(Strides(out))
    }

    /// Every stride of `o` added to these, or `None` past [`MAX_DYNAMIC`].
    pub fn union(self, o: Strides) -> Option<Strides> {
        o.iter().try_fold(self, |acc, s| acc.with(s))
    }

    /// The strides, largest first.
    pub fn iter(self) -> impl Iterator<Item = u64> {
        self.0.into_iter().take_while(|s| *s != 0)
    }
}

/// A pointer's offset shape from its object's start: a constant, or a constant plus one dynamic
/// multiple of each of its strides (one per dynamically indexed array level).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Shape {
    Exact(i64),
    Strided { base: i64, strides: Strides },
}

/// A dynamic level of a [`TypedPath`]: an array whose element 0 starts `start` bytes into the
/// enclosing dynamic level's element (or into the object, for the first level), with elements of
/// `stride` bytes, `len` of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DynLevel {
    pub start: u32,
    pub stride: u32,
    pub len: u32,
}

/// One index of a [`TypedPath`]'s access chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Step {
    Const(u32),
    Dyn(DynLevel),
}

/// The access chain of a typed access, outermost index first. Each dynamic index is recovered
/// from the pointer's offset level by level: with `rem` the offset (the object's start is 0),
/// a level's index is `(rem - start) / stride` and the next level's `rem` is `(rem - start) %
/// stride`, the offset inside the element. For in-range indices that is exact: a level's
/// constant displacement and every inner level's contribution stay inside its element. Every
/// array level that encloses a dynamic one is dynamic too ([`path`]), so an offset that a
/// flattened index carries past an inner array's end, still inside the object, recovers the
/// element it addresses.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TypedPath {
    pub steps: Vec<Step>,
}

impl TypedPath {
    /// The dynamic levels, outermost first.
    pub fn dynamic(&self) -> impl Iterator<Item = DynLevel> + '_ {
        self.steps.iter().filter_map(|s| match s {
            Step::Dyn(d) => Some(*d),
            Step::Const(_) => None,
        })
    }
}

/// One step into `ty` at byte `off`: the child type, the offset inside it, the child's index,
/// and the child's byte offset inside `ty`.
fn step(
    e: &Entry,
    layouts: &TypeLayouts,
    ty: AirTypeId,
    off: u64,
) -> Option<(AirTypeId, u64, u32, u64)> {
    match e.module.types().get(ty)? {
        Type::Struct { fields, .. } => {
            let offs = layouts.field_offsets(ty)?;
            fields
                .iter()
                .zip(offs.iter())
                .enumerate()
                .find_map(|(i, (f, &o))| {
                    let s = layouts.alloc_size(*f)?;
                    (o <= off && off < o.saturating_add(s))
                        .then(|| Some((*f, off - o, u32::try_from(i).ok()?, o)))
                        .flatten()
                })
        }
        Type::Array { elem, len } => {
            let es = layouts.alloc_size(*elem).filter(|&s| s > 0)?;
            let i = off / es;
            (i < *len).then(|| Some((*elem, off % es, u32::try_from(i).ok()?, i * es)))?
        }
        Type::Vector { elem, len } => {
            let es = layouts.alloc_size(*elem).filter(|&s| s > 0)?;
            let i = off / es;
            (i < u64::from(*len))
                .then(|| Some((*elem, off % es, u32::try_from(i).ok()?, i * es)))?
        }
        _ => None,
    }
}

/// The constant indices from `ty` at `off` down to the first subobject `stop` accepts (its type
/// and the offset inside it), and what `stop` answers there.
fn walk<R>(
    e: &Entry,
    layouts: &TypeLayouts,
    mut ty: AirTypeId,
    mut off: u64,
    stop: &dyn Fn(AirTypeId, u64) -> Option<R>,
    out: &mut Vec<Step>,
) -> Option<R> {
    for _ in 0..=MAX_DEPTH {
        if let Some(r) = stop(ty, off) {
            return Some(r);
        }
        let (t, o, i, _) = step(e, layouts, ty, off)?;
        out.push(Step::Const(i));
        (ty, off) = (t, o);
    }
    None
}

/// The access chain of an access of type `a` at `shape` into an object of `count` elements of
/// type `obj` (the memory plan's typed-access rule), or `None` when the shape does not name a
/// subobject of type `a` (the caller refuses).
///
/// A strided shape's strides are matched to array levels outermost first, along the path its
/// constant part `base` takes: the object itself (an array of `count` elements of `obj`), then
/// each array level inside it whose element size is the next stride. Every stride must match a
/// level; the constant indices between levels, and below the last, locate `base` inside each
/// element (the displacement a dynamic index of 0 leaves). An array level whose element is larger
/// than the next stride, the object itself among them where it holds more than one element, is
/// dynamic as well, its index recovered from the offset: a flattened index (a `float *` into a
/// `float[8][8]`) may carry the offset across its elements, as Metal's byte address does.
pub fn path(
    e: &Entry,
    layouts: &TypeLayouts,
    obj: AirTypeId,
    count: u64,
    shape: Shape,
    a: AirTypeId,
) -> Option<TypedPath> {
    path_to(e, layouts, obj, count, shape, &|ty, off| {
        (ty == a && off == 0).then_some(())
    })
    .map(|(p, ())| p)
}

/// Which held elements of a simdgroup matrix's fragment an access reaches: both, as a `<2 x T>` at
/// the matrix's start (MSL's `thread_elements()` as one vector), or one, as a `T` at element 0
/// or 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FragmentPart {
    Pair,
    Lane(u32),
}

/// The access chain of an access of type `a` at `shape` into an object of `count` elements of
/// type `obj` that reaches a simdgroup matrix member's held elements ([`FragmentPart`]): the path
/// to the matrix member itself, the part, and the matrix's type. `None` for any other access:
/// the whole matrix (see [`path`]), an element past 1, or a part of another type than the
/// matrix's element.
pub fn fragment_path(
    e: &Entry,
    layouts: &TypeLayouts,
    obj: AirTypeId,
    count: u64,
    shape: Shape,
    a: AirTypeId,
) -> Option<(TypedPath, FragmentPart, AirTypeId)> {
    let types = e.module.types();
    let (elem, pair) = match types.get(a)? {
        Type::Vector { elem, len: 2 } => (*elem, true),
        Type::Vector { .. } => return None,
        _ => (a, false),
    };
    let size = layouts.alloc_size(elem).filter(|&s| s > 0)?;
    path_to(e, layouts, obj, count, shape, &|ty, off| {
        match types.get(ty) {
            Some(Type::Vector { elem: m, .. }) if fragment::is_matrix(types, ty) && *m == elem => {}
            _ => return None,
        }
        let part = match (pair, off) {
            (true, 0) => FragmentPart::Pair,
            (false, o) if o == 0 || o == size => FragmentPart::Lane(u32::from(o != 0)),
            _ => return None,
        };
        Some((part, ty))
    })
    .map(|(p, (part, ty))| (p, part, ty))
}

/// [`path`]'s walk, ending where `stop` accepts the subobject reached.
fn path_to<R>(
    e: &Entry,
    layouts: &TypeLayouts,
    obj: AirTypeId,
    count: u64,
    shape: Shape,
    stop: &dyn Fn(AirTypeId, u64) -> Option<R>,
) -> Option<(TypedPath, R)> {
    let size = layouts.alloc_size(obj).filter(|&s| s > 0)?;
    let mut steps = Vec::new();
    let (base, strides) = match shape {
        Shape::Exact(c) => (c, None),
        Shape::Strided { base, strides } => (base, Some(strides)),
    };
    let base = u64::try_from(base).ok()?;
    let mut pending = strides.into_iter().flat_map(Strides::iter).peekable();
    // The object level: a dynamic index over its `count` elements, or element `base / size`.
    // `start` is the byte offset, inside the innermost dynamic level's element (or the object),
    // of the type `ty` being walked; `off` is `base`'s offset inside `ty`.
    let deeper = |p: Option<&u64>, size: u64| p.is_some_and(|&s| s < size);
    let mut start = if pending.peek() == Some(&size) || (count > 1 && deeper(pending.peek(), size))
    {
        if pending.peek() == Some(&size) {
            pending.next();
        }
        steps.push(Step::Dyn(DynLevel {
            start: 0,
            stride: u32::try_from(size).ok()?,
            len: u32::try_from(count).ok()?,
        }));
        0
    } else {
        let first = base / size;
        if first >= count {
            return None;
        }
        steps.push(Step::Const(u32::try_from(first).ok()?));
        first * size
    };
    let (mut ty, mut off) = (obj, base % size);
    for _ in 0..=MAX_DEPTH {
        let Some(&stride) = pending.peek() else {
            let r = walk(e, layouts, ty, off, stop, &mut steps)?;
            return Some((TypedPath { steps }, r));
        };
        if let Some(Type::Array { elem, len }) = e.module.types().get(ty) {
            let es = layouts.alloc_size(*elem).filter(|&s| s > 0)?;
            if es >= stride {
                // The stride's own level, or an array enclosing a deeper dynamic level.
                if es == stride {
                    pending.next();
                }
                steps.push(Step::Dyn(DynLevel {
                    start: u32::try_from(start).ok()?,
                    stride: u32::try_from(es).ok()?,
                    len: u32::try_from(*len).ok()?,
                }));
                (ty, off, start) = (*elem, off % es, 0);
                continue;
            }
        }
        let (t, o, i, at) = step(e, layouts, ty, off)?;
        steps.push(Step::Const(i));
        start = start.checked_add(at)?;
        (ty, off) = (t, o);
    }
    None
}
