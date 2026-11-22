//! Runtime values and type queries.

use std::collections::BTreeMap;

use crate::pipeline::sym::SpecKey;
use crate::spirv::ids::{ConstId, TypeId, Value};
use crate::spirv::ir::{ConstKind, Constant, StorageClass, Type, TypedModule};

use super::{malformed, IResult, InterpError, Unmodelled};

/// A runtime value. Scalars hold their bits zero-extended from their width (`bool` is 0 or 1);
/// vectors, matrices (by column), arrays, and structs hold their parts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Val {
    S(u64),
    C(Box<[Val]>),
    P(Ptr),
}

impl Val {
    pub(super) fn s(&self) -> IResult<u64> {
        match self {
            Val::S(x) => Ok(*x),
            _ => Err(malformed("expected a scalar value")),
        }
    }

    pub(super) fn parts(&self) -> IResult<&[Val]> {
        match self {
            Val::C(p) => Ok(p),
            _ => Err(malformed("expected a composite value")),
        }
    }

    pub(super) fn ptr(&self) -> IResult<Ptr> {
        match self {
            Val::P(p) => Ok(*p),
            _ => Err(malformed("expected a pointer value")),
        }
    }

    /// The components of a scalar (one) or a vector of scalars.
    pub(super) fn comps(&self) -> IResult<Vec<u64>> {
        match self {
            Val::S(x) => Ok(vec![*x]),
            Val::C(p) => p.iter().map(Val::s).collect(),
            Val::P(_) => Err(malformed("expected a numeric value")),
        }
    }

    /// A scalar (one component) or a vector.
    pub(super) fn from_comps(c: Vec<u64>) -> Val {
        match c.as_slice() {
            [x] => Val::S(*x),
            _ => Val::C(c.into_iter().map(Val::S).collect()),
        }
    }

    /// A scalar (one component) or a vector, from components already held as scalar values:
    /// [`Val::from_comps`] without the intermediate integers.
    pub(super) fn from_lanes(c: Vec<Val>) -> Val {
        match <[Val; 1]>::try_from(c) {
            Ok([x]) => x,
            Err(c) => Val::C(c.into()),
        }
    }
}

/// The memory object a pointer points into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Root {
    /// Allocation `n` of the executing invocation (`Function`, `Private`, `Input`).
    Lane(u32),
    /// `Workgroup` variable region `n` of the current workgroup.
    Workgroup(u32),
    /// Buffer `n` of the resources.
    Buffer(u32),
    /// A descriptor array of buffers whose element 0 is buffer `n` of the resources:
    /// `OpAccessChain` selects an element, buffer `n + e`.
    BufferArray(u32),
    Push,
    /// `PhysicalStorageBuffer`: the offset is the device address.
    Physical,
    /// Bound image `n` (`UniformConstant`): loading the variable yields the image itself.
    Image(u32),
    /// A texel of bound image `n` (`OpImageTexelPointer`, an atomic's operand): the offset is the
    /// texel's index in its texels, and the pointer reaches the texel's first component, a 32-bit
    /// integer.
    Texel(u32),
}

/// A pointer: its memory object, byte offset, and the matrix stride of the matrix (or array of
/// matrices) it points to, when it came from an explicit layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Ptr {
    pub root: Root,
    pub off: u64,
    pub ms: Option<u32>,
}

/// The component class of a scalar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Bool,
    Int { signed: bool },
    Float,
}

/// The shape of a scalar or vector type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Num {
    pub kind: Kind,
    /// Bits per component (1 for `bool`).
    pub width: u8,
    /// Components (1 for a scalar).
    pub n: u8,
}

/// Type queries over one module.
#[derive(Clone, Copy)]
pub(super) struct Tys<'m> {
    pub m: &'m TypedModule,
    /// The run's specialization values (`Resources::spec`).
    pub spec: &'m BTreeMap<SpecKey, u64>,
}

impl<'m> Tys<'m> {
    pub(super) fn get(&self, t: TypeId) -> IResult<&'m Type> {
        self.m
            .types()
            .get(t)
            .map_err(|e| malformed(format!("type {t:?}: {e:?}")))
    }

    pub(super) fn value_ty(&self, v: Value) -> IResult<TypeId> {
        self.m
            .value(v)
            .map(|d| d.ty)
            .map_err(|e| malformed(format!("value {v:?}: {e:?}")))
    }

    /// The scalar shape of a scalar type, if it is one.
    pub(super) fn scalar(&self, t: TypeId) -> IResult<Option<Num>> {
        Ok(match self.get(t)? {
            Type::Bool => Some(Num {
                kind: Kind::Bool,
                width: 1,
                n: 1,
            }),
            Type::Int { width, signed } => Some(Num {
                kind: Kind::Int { signed: *signed },
                width: *width,
                n: 1,
            }),
            Type::Float { width } => Some(Num {
                kind: Kind::Float,
                width: *width,
                n: 1,
            }),
            _ => None,
        })
    }

    /// The shape of a scalar or vector type.
    pub(super) fn num(&self, t: TypeId) -> IResult<Num> {
        if let Some(s) = self.scalar(t)? {
            return Ok(s);
        }
        match self.get(t)? {
            Type::Vector { elem, len } => {
                let s = self
                    .scalar(*elem)?
                    .ok_or_else(|| malformed("vector of a non-scalar"))?;
                Ok(Num { n: *len, ..s })
            }
            _ => Err(malformed(format!("type {t:?} is not a scalar or vector"))),
        }
    }

    /// The shape of a value's scalar or vector type.
    pub(super) fn num_of(&self, v: Value) -> IResult<Num> {
        self.num(self.value_ty(v)?)
    }

    /// The pointee of a pointer type.
    pub(super) fn pointee(&self, t: TypeId) -> IResult<(StorageClass, TypeId)> {
        match self.get(t)? {
            Type::Pointer { class, pointee } => Ok((*class, *pointee)),
            _ => Err(malformed(format!("type {t:?} is not a pointer"))),
        }
    }

    /// The length of a sized array (a specialization length takes its default).
    pub(super) fn array_len(&self, len: ConstId) -> IResult<u64> {
        let def = self
            .m
            .consts()
            .get(len)
            .map_err(|e| malformed(format!("{e:?}")))?;
        match &def.kind {
            ConstKind::Plain(Constant::Scalar { bits, .. }) => Ok(bits.0),
            ConstKind::Spec { .. } | ConstKind::SpecOp { .. } => self.spec_scalar(len, 0),
            _ => Err(malformed("array length is not a scalar constant")),
        }
    }

    /// The value of a scalar plain, specialization, or `OpSpecConstantOp` constant under the
    /// run's specialization values (Vulkan sizes a specialization-length array by the specialized
    /// value). `depth` bounds the operand chain.
    pub(super) fn spec_scalar(&self, c: ConstId, depth: u32) -> IResult<u64> {
        if depth > 64 {
            return Err(malformed("a specialization operation chain too deep"));
        }
        let def = self
            .m
            .consts()
            .get(c)
            .map_err(|e| malformed(format!("{e:?}")))?;
        match &def.kind {
            ConstKind::Plain(Constant::Scalar { bits, .. }) => Ok(bits.0),
            ConstKind::Spec { key, default } => {
                Ok(self.spec.get(key).copied().unwrap_or(default.0))
            }
            ConstKind::SpecOp { op, operands } => {
                let mut vals = Vec::with_capacity(operands.len());
                for o in operands.iter() {
                    vals.push(self.spec_scalar(*o, depth + 1)?);
                }
                // The operation's width is its operands' (for `Select`, its arms'), not its
                // result's: `ULessThan` compares integers but yields a 1-bit `bool`.
                let last = operands
                    .last()
                    .ok_or_else(|| malformed("a specialization operation without operands"))?;
                let last_ty = self
                    .m
                    .consts()
                    .get(*last)
                    .map_err(|e| malformed(format!("{e:?}")))?
                    .ty;
                let width = self
                    .scalar(last_ty)?
                    .ok_or_else(|| malformed("a non-scalar specialization operand"))?
                    .width;
                op.eval(width, &vals)
                    .ok_or_else(|| malformed("a specialization operation of the wrong arity"))
            }
            _ => Err(malformed("not a scalar constant")),
        }
    }

    /// The type of part `i` of a composite type.
    pub(super) fn part(&self, t: TypeId, i: usize) -> IResult<TypeId> {
        match self.get(t)? {
            Type::Vector { elem, .. } | Type::Array { elem, .. } => Ok(*elem),
            Type::Matrix { col, .. } => Ok(*col),
            Type::Struct { members, .. } => members
                .get(i)
                .copied()
                .ok_or_else(|| malformed("struct member out of range")),
            _ => Err(malformed(format!("type {t:?} is not a composite"))),
        }
    }

    /// The number of parts of a composite type.
    pub(super) fn part_count(&self, t: TypeId) -> IResult<u64> {
        match self.get(t)? {
            Type::Vector { len, .. } => Ok(u64::from(*len)),
            Type::Matrix { cols, .. } => Ok(u64::from(*cols)),
            Type::Array { len, .. } => self.array_len(*len),
            Type::Struct { members, .. } => Ok(members.len() as u64),
            _ => Err(malformed(format!("type {t:?} is not a composite"))),
        }
    }

    /// The zero value of a type (`OpConstantNull`, and the model of undefined values).
    pub(super) fn zero(&self, t: TypeId) -> IResult<Val> {
        match self.get(t)? {
            Type::Bool | Type::Int { .. } | Type::Float { .. } => Ok(Val::S(0)),
            Type::Pointer {
                class: StorageClass::PhysicalStorageBuffer,
                ..
            } => Ok(Val::P(Ptr {
                root: Root::Physical,
                off: 0,
                ms: None,
            })),
            Type::Vector { .. }
            | Type::Matrix { .. }
            | Type::Array { .. }
            | Type::Struct { .. } => {
                let n = self.part_count(t)?;
                if n > 1 << 20 {
                    return Err(Unmodelled::Type(t).err());
                }
                let mut parts = Vec::with_capacity(n as usize);
                for i in 0..n as usize {
                    parts.push(self.zero(self.part(t, i)?)?);
                }
                Ok(Val::C(parts.into()))
            }
            _ => Err(Unmodelled::Type(t).err()),
        }
    }

    /// Every constant's value, indexed by [`ConstId`]: a specialization constant takes its value
    /// from `spec` (the bits in its type's width, else a configuration error) or its default.
    pub(super) fn constants(&self, spec: &BTreeMap<SpecKey, u64>) -> IResult<Vec<Val>> {
        let mut out: Vec<Val> = Vec::with_capacity(self.m.consts().len());
        for (_, def) in self.m.consts().iter() {
            let v = match &def.kind {
                ConstKind::Plain(Constant::Scalar { bits, .. }) => Val::S(bits.0),
                ConstKind::Spec { key, default } => match spec.get(key) {
                    None => Val::S(default.0),
                    Some(&bits) => {
                        let width = self
                            .scalar(def.ty)?
                            .ok_or_else(|| malformed("a non-scalar specialization constant"))?
                            .width;
                        if bits & !mask(width) != 0 {
                            return Err(InterpError::Config(format!(
                                "specialization value {bits:#x} for {key:?} exceeds {width} bits"
                            )));
                        }
                        Val::S(bits)
                    }
                },
                ConstKind::Plain(Constant::Null { ty }) => self.zero(*ty)?,
                // The local-size composite reads its three specialization constants' defaults,
                // like any other composite.
                ConstKind::Plain(Constant::Composite { parts, .. }) => composite(&out, parts)?,
                ConstKind::WorkgroupSize { parts } => composite(&out, parts)?,
                ConstKind::SpecOp { .. } => {
                    let id = ConstId::new(out.len() as u32);
                    Val::S(self.spec_scalar(id, 0)?)
                }
            };
            out.push(v);
        }
        Ok(out)
    }
}

/// A composite constant from its already-evaluated parts.
fn composite(out: &[Val], parts: &[ConstId]) -> IResult<Val> {
    let mut vs = Vec::with_capacity(parts.len());
    for p in parts {
        vs.push(
            out.get(p.index())
                .cloned()
                .ok_or_else(|| malformed("constant part declared later"))?,
        );
    }
    Ok(Val::C(vs.into()))
}

/// All-ones in the low `width` bits.
pub(super) fn mask(width: u8) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// `bits` (a `width`-bit two's complement integer) as `i64`.
pub(super) fn sext(bits: u64, width: u8) -> i64 {
    if width >= 64 {
        bits as i64
    } else {
        let sh = 64 - u32::from(width);
        ((bits << sh) as i64) >> sh
    }
}
