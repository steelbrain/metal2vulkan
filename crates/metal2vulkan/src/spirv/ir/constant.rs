//! Constants, specialization constants, and the constant interner.
//!
//! Plain constants are hash-consed. Specialization constants are keyed by their [`SpecKey`] (link
//! assigns the `SpecId`) and never merged with plain constants of equal value.

use crate::pipeline::sym::SpecKey;
use crate::spirv::error::{BResult, BuildError, ConstRule, Dangling, TypeRule};
use crate::spirv::ids::{ConstId, TypeId};
use crate::spirv::ir::intern::FxHashMap;
use crate::spirv::ir::types::{Type, Types};

/// Raw bits of a scalar in its type's width: `0`/`1` for `bool`, two's complement for integers,
/// IEEE-754 for floats. No bit above the width may be set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConstBits(pub u64);

/// A plain (non-specialization) constant.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Constant {
    /// A `bool`, integer, or float scalar (`OpConstantTrue/False`, `OpConstant`).
    Scalar { ty: TypeId, bits: ConstBits },
    /// `OpConstantComposite` of plain constants.
    Composite { ty: TypeId, parts: Box<[ConstId]> },
    /// `OpConstantNull`.
    Null { ty: TypeId },
}

impl Constant {
    /// The constant's type.
    pub fn ty(&self) -> TypeId {
        match self {
            Constant::Scalar { ty, .. }
            | Constant::Composite { ty, .. }
            | Constant::Null { ty } => *ty,
        }
    }
}

/// What an interned constant is.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstKind {
    Plain(Constant),
    /// `OpSpecConstant{True,False,}` whose `SpecId` is the relocatable symbol `key`.
    Spec {
        key: SpecKey,
        default: ConstBits,
    },
    /// The `OpSpecConstantComposite` of three positive `u32` constants, at least one of them a
    /// specialization constant, decorated `BuiltIn WorkgroupSize`: a compute module's local size
    /// from specialization constants without `LocalSizeId` (which needs `maintenance4`). The
    /// flattened simd tiers use `(sn, 1, 1)` with plain `1`s. At most one per module.
    WorkgroupSize {
        parts: [ConstId; 3],
    },
    /// `OpSpecConstantOp` of `op` over `operands` (integer or `bool` scalars, at least one of them
    /// a specialization constant): a value fixed at pipeline creation, e.g. the length of the
    /// threadgroup arena from its specialized argument lengths. Never merged with a plain constant
    /// of equal default value.
    SpecOp {
        op: SpecOp,
        operands: Box<[ConstId]>,
    },
}

/// The operations an [`ConstKind::SpecOp`] may apply: the subset of the `Shader`-capability
/// `OpSpecConstantOp` opcodes the new core needs, all on integer or `bool` scalars.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpecOp {
    /// `a + b` (wrapping), both operands of the result type.
    IAdd,
    /// `a · b` (wrapping), both operands of the result type.
    IMul,
    /// `a & b`, both operands of the result type.
    BitwiseAnd,
    /// `a >> b` (logical), both operands of the result type.
    ShiftRightLogical,
    /// `a < b` (unsigned) of two integers of one type; the result is `bool`.
    ULessThan,
    /// `c ? a : b` for a `bool` `c` and `a`, `b` of the result type.
    Select,
}

impl SpecOp {
    /// The number of operands.
    pub fn arity(self) -> usize {
        match self {
            SpecOp::Select => 3,
            _ => 2,
        }
    }

    /// The operation on scalar bits. `width` is the width of the integer operands (for
    /// [`SpecOp::Select`], of the arms), never the result's: [`SpecOp::ULessThan`] compares two
    /// `width`-bit integers and yields a `bool`. `bool` is 0 or 1. `None` for a wrong operand
    /// count.
    #[cfg(test)]
    pub fn eval(self, width: u8, operands: &[u64]) -> Option<u64> {
        let mask = if width >= 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        Some(match (self, operands) {
            (SpecOp::IAdd, [a, b]) => a.wrapping_add(*b) & mask,
            (SpecOp::IMul, [a, b]) => a.wrapping_mul(*b) & mask,
            (SpecOp::BitwiseAnd, [a, b]) => a & b & mask,
            // SPIR-V leaves a shift by the width or more undefined; the model yields 0.
            (SpecOp::ShiftRightLogical, [a, b]) => {
                if *b >= u64::from(width) {
                    0
                } else {
                    (a & mask) >> b
                }
            }
            (SpecOp::ULessThan, [a, b]) => u64::from((a & mask) < (b & mask)),
            (SpecOp::Select, [c, a, b]) => {
                if *c != 0 {
                    *a
                } else {
                    *b
                }
            }
            _ => return None,
        })
    }
}

/// One interned constant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstDef {
    pub ty: TypeId,
    pub kind: ConstKind,
}

/// The constant interner (insertion-ordered; the hash indexes are lookup-only).
#[derive(Clone, Debug, Default)]
pub struct Consts {
    list: Vec<ConstDef>,
    index: FxHashMap<Constant, ConstId>,
    spec: FxHashMap<SpecKey, ConstId>,
    spec_ops: FxHashMap<(SpecOp, TypeId, Box<[ConstId]>), ConstId>,
    /// The heap bytes of the operand lists inside `list` and the index keys, counted as they are
    /// interned.
    boxed: usize,
}

impl Constant {
    /// The heap bytes of the constant's boxed parts.
    pub fn heap_bytes(&self) -> usize {
        match self {
            Constant::Composite { parts, .. } => std::mem::size_of_val::<[ConstId]>(parts),
            Constant::Scalar { .. } | Constant::Null { .. } => 0,
        }
    }
}

impl Consts {
    /// The heap bytes of the interner's list and indexes, with the boxed operand lists of the
    /// constants and of the index keys, for the byte ledger.
    pub fn footprint(&self) -> usize {
        use crate::budget::{map_bytes, vec_bytes};
        vec_bytes(&self.list)
            + map_bytes(&self.index)
            + map_bytes(&self.spec)
            + map_bytes(&self.spec_ops)
            + self.boxed
    }

    /// The constant with this id.
    pub fn get(&self, id: ConstId) -> BResult<&ConstDef> {
        // A match, not `ok_or`, as `TypedModule::value`: no error built at every lookup.
        match self.list.get(id.index()) {
            Some(def) => Ok(def),
            None => Err(BuildError::Dangling(Dangling::Const(id))),
        }
    }

    /// Every constant, in interning order.
    pub fn iter(&self) -> impl Iterator<Item = (ConstId, &ConstDef)> {
        self.list
            .iter()
            .enumerate()
            .map(|(i, c)| (ConstId::new(i as u32), c))
    }

    /// Stop interning plain constant `id`: a later [`Consts::intern`] of an equal constant makes a
    /// new one. Its definition stays, so its id stays valid; nothing declares it once its module
    /// dropped its declaration. Nothing changes for any other constant.
    pub fn forget(&mut self, id: ConstId) {
        if let Some(ConstDef {
            kind: ConstKind::Plain(k),
            ..
        }) = self.list.get(id.index())
        {
            if self.index.get(k) == Some(&id) {
                if let Some((key, _)) = self.index.remove_entry(k) {
                    self.boxed = self.boxed.saturating_sub(key.heap_bytes());
                }
            }
        }
    }

    /// The number of interned constants.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.list.len()
    }

    /// The bits of a plain scalar constant.
    pub fn scalar_bits(&self, id: ConstId) -> Option<ConstBits> {
        match self.list.get(id.index()) {
            Some(ConstDef {
                kind: ConstKind::Plain(Constant::Scalar { bits, .. }),
                ..
            }) => Some(*bits),
            _ => None,
        }
    }

    /// The length an array type with length operand `id` has: `Some(n)` for a plain integer
    /// constant, `None` for an integer specialization constant.
    pub fn array_len(&self, id: ConstId, types: &Types) -> BResult<Option<u32>> {
        let def = self.get(id)?;
        if !matches!(types.get(def.ty)?, Type::Int { .. }) {
            return Err(BuildError::Type(TypeRule::ArrayLength));
        }
        let (bits, spec) = match &def.kind {
            ConstKind::Plain(Constant::Scalar { bits, .. }) => (*bits, false),
            ConstKind::Spec { default, .. } => (*default, true),
            // Its value is known only at pipeline creation (and checked there).
            ConstKind::SpecOp { .. } => {
                return match types.get(def.ty)? {
                    Type::Int { .. } => Ok(None),
                    _ => Err(BuildError::Type(TypeRule::ArrayLength)),
                };
            }
            ConstKind::Plain(_) | ConstKind::WorkgroupSize { .. } => {
                return Err(BuildError::Type(TypeRule::ArrayLength))
            }
        };
        // Signed lengths must be positive too: their sign bit is above `u32` for 64-bit and a
        // negative 32-bit value exceeds `i32::MAX`.
        let signed = matches!(types.get(def.ty)?, Type::Int { signed: true, .. });
        let limit = if signed {
            i32::MAX as u64
        } else {
            u32::MAX as u64
        };
        if bits.0 == 0 || bits.0 > limit {
            return Err(BuildError::Type(TypeRule::ArrayLength));
        }
        if spec {
            Ok(None)
        } else {
            Ok(Some(bits.0 as u32))
        }
    }

    /// Intern a plain constant after checking it against its type. Returns `(id, fresh)`.
    pub fn intern(&mut self, c: Constant, types: &Types) -> BResult<(ConstId, bool)> {
        if let Some(id) = self.index.get(&c) {
            return Ok((*id, false));
        }
        self.check(&c, types)?;
        let id = self.next_id()?;
        // The list and the index each hold the constant's parts.
        self.boxed = self.boxed.saturating_add(c.heap_bytes().saturating_mul(2));
        self.index.insert(c.clone(), id);
        self.list.push(ConstDef {
            ty: c.ty(),
            kind: ConstKind::Plain(c),
        });
        Ok((id, true))
    }

    /// Declare the specialization constant `key`. Declaring the same key again with the same type
    /// and default returns the first declaration.
    pub fn spec(
        &mut self,
        key: SpecKey,
        ty: TypeId,
        default: ConstBits,
        types: &Types,
    ) -> BResult<(ConstId, bool)> {
        if let Some(id) = self.spec.get(&key) {
            let def = self.get(*id)?;
            return match &def.kind {
                ConstKind::Spec { default: d, .. } if def.ty == ty && *d == default => {
                    Ok((*id, false))
                }
                _ => Err(BuildError::Constant(ConstRule::SpecKeyConflict(key))),
            };
        }
        match scalar_width(types.get(ty)?) {
            Some(w) if fits(default, w) => {}
            _ => return Err(BuildError::Constant(ConstRule::SpecType)),
        }
        let id = self.next_id()?;
        self.spec.insert(key, id);
        self.list.push(ConstDef {
            ty,
            kind: ConstKind::Spec { key, default },
        });
        Ok((id, true))
    }

    /// Declare `OpSpecConstantOp` `op` of type `ty` over `operands`. Every operand must be a scalar
    /// plain or specialization constant (or another spec op) of the type the operation requires,
    /// and at least one must be specialization-dependent. The same operation over the same operands
    /// returns the first declaration.
    pub fn spec_op(
        &mut self,
        op: SpecOp,
        ty: TypeId,
        operands: &[ConstId],
        types: &Types,
    ) -> BResult<(ConstId, bool)> {
        let key = (op, ty, Box::<[ConstId]>::from(operands));
        if let Some(id) = self.spec_ops.get(&key) {
            return Ok((*id, false));
        }
        let err = || BuildError::Constant(ConstRule::SpecOpOperands);
        if operands.len() != op.arity() {
            return Err(err());
        }
        let result = types.get(ty)?;
        let mut specs = 0;
        let mut tys = Vec::with_capacity(operands.len());
        for c in operands {
            let def = self.get(*c)?;
            match &def.kind {
                ConstKind::Spec { .. } | ConstKind::SpecOp { .. } => specs += 1,
                ConstKind::Plain(Constant::Scalar { .. }) => {}
                _ => return Err(err()),
            }
            tys.push(def.ty);
        }
        if specs == 0 {
            return Err(err());
        }
        let is_int = |t: TypeId| -> BResult<bool> { Ok(matches!(types.get(t)?, Type::Int { .. })) };
        let ok = match (op, tys.as_slice()) {
            (
                SpecOp::IAdd | SpecOp::IMul | SpecOp::BitwiseAnd | SpecOp::ShiftRightLogical,
                [a, b],
            ) => is_int(ty)? && *a == ty && *b == ty,
            (SpecOp::ULessThan, [a, b]) => matches!(result, Type::Bool) && is_int(*a)? && a == b,
            (SpecOp::Select, [c, a, b]) => {
                matches!(types.get(*c)?, Type::Bool)
                    && *a == ty
                    && *b == ty
                    && scalar_width(result).is_some()
            }
            _ => false,
        };
        if !ok {
            return Err(err());
        }
        let id = self.next_id()?;
        // The list and the index key each hold the operands.
        let operand_bytes = std::mem::size_of_val::<[ConstId]>(operands);
        self.boxed = self.boxed.saturating_add(operand_bytes.saturating_mul(2));
        self.spec_ops.insert(key, id);
        self.list.push(ConstDef {
            ty,
            kind: ConstKind::SpecOp {
                op,
                operands: operands.into(),
            },
        });
        Ok((id, true))
    }

    /// Declare the module's `WorkgroupSize` composite of `ty` (a `u32` 3-vector) from three
    /// positive `u32` constants ([`Consts::positive`]), at least one of them a specialization
    /// constant or an `OpSpecConstantOp` of them. Declaring it again with the same parts returns
    /// the first declaration; different parts are a conflict.
    pub fn workgroup_size(
        &mut self,
        ty: TypeId,
        parts: [ConstId; 3],
        types: &Types,
    ) -> BResult<(ConstId, bool)> {
        if let Some((id, def)) = self
            .iter()
            .find(|(_, d)| matches!(d.kind, ConstKind::WorkgroupSize { .. }))
        {
            return match def.kind {
                ConstKind::WorkgroupSize { parts: p } if p == parts && def.ty == ty => {
                    Ok((id, false))
                }
                _ => Err(BuildError::Constant(ConstRule::CompositeParts)),
            };
        }
        let u32_elem = match types.get(ty)? {
            Type::Vector { elem, len: 3 } => *elem,
            _ => return Err(BuildError::Constant(ConstRule::CompositeType)),
        };
        if !matches!(
            types.get(u32_elem)?,
            Type::Int {
                width: 32,
                signed: false
            }
        ) {
            return Err(BuildError::Constant(ConstRule::CompositeType));
        }
        let mut specs = 0;
        for p in parts {
            let def = self.get(p)?;
            if matches!(def.kind, ConstKind::Spec { .. } | ConstKind::SpecOp { .. }) {
                specs += 1;
            }
            // A local size of zero is invalid on every axis, so a part must be positive.
            if def.ty != u32_elem || !self.positive(p)? {
                return Err(BuildError::Constant(ConstRule::CompositeParts));
            }
        }
        // A size of plain constants only is a `LocalSize` mode, not a specialization composite.
        if specs == 0 {
            return Err(BuildError::Constant(ConstRule::CompositeParts));
        }
        let id = self.next_id()?;
        self.list.push(ConstDef {
            ty,
            kind: ConstKind::WorkgroupSize { parts },
        });
        Ok((id, true))
    }

    /// Whether constant `c` is positive at its defaults: a plain scalar or a specialization
    /// constant's default above 0, or a product or select of such. An `OpSpecConstantOp` of any
    /// other operation is not known to be.
    fn positive(&self, c: ConstId) -> BResult<bool> {
        Ok(match &self.get(c)?.kind {
            ConstKind::Spec { default, .. } => default.0 != 0,
            ConstKind::Plain(Constant::Scalar { bits, .. }) => bits.0 != 0,
            ConstKind::SpecOp {
                op: SpecOp::IMul,
                operands,
            } => match **operands {
                [a, b] => self.positive(a)? && self.positive(b)?,
                _ => false,
            },
            ConstKind::SpecOp {
                op: SpecOp::Select,
                operands,
            } => match **operands {
                [_, a, b] => self.positive(a)? && self.positive(b)?,
                _ => false,
            },
            _ => false,
        })
    }

    /// The module's `WorkgroupSize` composite, if declared.
    pub fn workgroup_size_const(&self) -> Option<ConstId> {
        self.iter()
            .find(|(_, d)| matches!(d.kind, ConstKind::WorkgroupSize { .. }))
            .map(|(id, _)| id)
    }

    fn next_id(&self) -> BResult<ConstId> {
        u32::try_from(self.list.len())
            .map(ConstId::new)
            .map_err(|_| BuildError::IdSpaceExhausted)
    }

    fn check(&self, c: &Constant, types: &Types) -> BResult<()> {
        let err = |r| Err(BuildError::Constant(r));
        let ty = types.get(c.ty())?;
        match c {
            Constant::Scalar { bits, .. } => match scalar_width(ty) {
                None => err(ConstRule::ScalarType),
                Some(w) if !fits(*bits, w) => err(ConstRule::BitsExceedWidth),
                Some(_) => Ok(()),
            },
            Constant::Composite { parts, .. } => {
                let expected: Vec<TypeId> = match ty {
                    Type::Vector { elem, len } => vec![*elem; usize::from(*len)],
                    Type::Matrix { col, cols } => vec![*col; usize::from(*cols)],
                    Type::Array { elem, len, .. } => match self.array_len(*len, types)? {
                        Some(n) if n as usize == parts.len() => vec![*elem; parts.len()],
                        Some(_) => return err(ConstRule::CompositeParts),
                        None => return err(ConstRule::CompositeType),
                    },
                    Type::Struct { members, .. } => members.to_vec(),
                    _ => return err(ConstRule::CompositeType),
                };
                if types.facts(c.ty())?.runtime_sized {
                    return err(ConstRule::CompositeType);
                }
                if expected.len() != parts.len() {
                    return err(ConstRule::CompositeParts);
                }
                for (p, want) in parts.iter().zip(expected) {
                    let def = self.get(*p)?;
                    if matches!(def.kind, ConstKind::Spec { .. } | ConstKind::SpecOp { .. }) {
                        return err(ConstRule::SpecInComposite);
                    }
                    if def.ty != want {
                        return err(ConstRule::CompositeParts);
                    }
                }
                Ok(())
            }
            Constant::Null { ty: id } => {
                let ok = match ty {
                    Type::Bool
                    | Type::Int { .. }
                    | Type::Float { .. }
                    | Type::Vector { .. }
                    | Type::Matrix { .. }
                    | Type::Struct { .. } => true,
                    Type::Array { len, .. } => self.array_len(*len, types)?.is_some(),
                    _ => false,
                };
                if ok && !types.facts(*id)?.runtime_sized {
                    Ok(())
                } else {
                    err(ConstRule::NullType)
                }
            }
        }
    }
}

/// The bit width of a scalar type, with `bool` as width 1.
fn scalar_width(t: &Type) -> Option<u8> {
    match t {
        Type::Bool => Some(1),
        Type::Int { width, .. } | Type::Float { width } => Some(*width),
        _ => None,
    }
}

fn fits(bits: ConstBits, width: u8) -> bool {
    width >= 64 || bits.0 >> width == 0
}
