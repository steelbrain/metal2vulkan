//! Composite constructors: construct, extract, insert, dynamic vector access, shuffles, and
//! logical copies.

use crate::spirv::builder::function::FunctionBuilder;
use crate::spirv::builder::module::StructOp;
use crate::spirv::builder::ops::{err, index};
use crate::spirv::error::{BResult, OperandRule};
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{Inst, Type};

impl FunctionBuilder<'_> {
    /// `OpCompositeConstruct` of type `ty`: a vector from at least two scalars or vectors of its
    /// component type totalling its length; a matrix from its columns; a sized array from its
    /// elements; a struct from its members.
    pub fn composite_construct(&mut self, ty: TypeId, parts: &[Value]) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::CompositeConstruct;
        if self.mb().types().facts(ty)?.runtime_sized {
            return Err(err(OP, OperandRule::ResultType));
        }
        let mut part_tys = Vec::with_capacity(parts.len());
        for p in parts {
            part_tys.push(self.value_ty(*p)?);
        }
        let ty_def = self.mb().types().get(ty)?.clone();
        let is_struct = matches!(ty_def, Type::Struct { .. });
        let expected: Vec<TypeId> = match ty_def {
            Type::Vector { elem, len } => {
                if parts.len() < 2 {
                    return Err(err(OP, OperandRule::Arity));
                }
                let mut total = 0u32;
                for (i, t) in part_tys.iter().enumerate() {
                    let n = match self.num_of(*t)? {
                        Some(x) if x.elem == elem => x.n,
                        _ => return Err(err(OP, OperandRule::Mismatch(index(i)))),
                    };
                    total += u32::from(n);
                }
                if total != u32::from(len) {
                    return Err(err(OP, OperandRule::Arity));
                }
                part_tys.clone()
            }
            Type::Matrix { col, cols } => vec![col; usize::from(cols)],
            Type::Array { elem, len, .. } => {
                match self.mb().consts().array_len(len, self.mb().types())? {
                    Some(n) => vec![elem; n as usize],
                    None => return Err(err(OP, OperandRule::ResultType)),
                }
            }
            Type::Struct { members, .. } => members.to_vec(),
            _ => return Err(err(OP, OperandRule::ResultType)),
        };
        if expected.len() != parts.len() {
            return Err(err(OP, OperandRule::Arity));
        }
        for (i, (got, want)) in part_tys.iter().zip(&expected).enumerate() {
            if got != want {
                return Err(err(OP, OperandRule::Mismatch(index(i))));
            }
        }
        // A part carrying a non-uniform descriptor element would reach its consumer through an
        // extract, undecorated.
        if let Some(i) = parts.iter().position(|p| self.mb().non_uniform.contains(p)) {
            return Err(err(OP, OperandRule::NonUniform(index(i))));
        }
        let y = self.emit(
            Inst::CompositeConstruct {
                parts: parts.into(),
            },
            ty,
        )?;
        if is_struct {
            self.note_struct(y, StructOp::Construct(parts.into()));
        }
        Ok(y)
    }

    /// The type reached by walking `indices` (literal indices) from composite type `ty`. Every
    /// index must be in range of a vector, matrix, sized array with a known length, or struct.
    fn walk_literal(&self, op: crate::spv::Op, ty: TypeId, indices: &[u32]) -> BResult<TypeId> {
        if indices.is_empty() {
            return Err(err(op, OperandRule::Arity));
        }
        let types = self.mb().types();
        let mut cur = ty;
        for (i, idx) in indices.iter().enumerate() {
            let range = err(op, OperandRule::Range(index(i + 1)));
            let (next, len) = match types.get(cur)? {
                Type::Vector { elem, len } => (*elem, u32::from(*len)),
                Type::Matrix { col, cols } => (*col, u32::from(*cols)),
                Type::Array { elem, len, .. } => match self.mb().consts().array_len(*len, types)? {
                    Some(n) => (*elem, n),
                    None => return Err(range),
                },
                Type::Struct { members, .. } => {
                    let m = members.get(*idx as usize).copied().ok_or(range)?;
                    (m, u32::MAX)
                }
                _ => return Err(err(op, OperandRule::Category(0))),
            };
            if *idx >= len {
                return Err(err(op, OperandRule::Range(index(i + 1))));
            }
            cur = next;
        }
        Ok(cur)
    }

    /// `OpCompositeExtract`: the result has the type at `indices`.
    pub fn composite_extract(&mut self, composite: Value, indices: &[u32]) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::CompositeExtract;
        let of = self.value_ty(composite)?;
        let ty = self.walk_literal(OP, of, indices)?;
        let y = self.emit(
            Inst::CompositeExtract {
                composite,
                indices: indices.into(),
            },
            ty,
        )?;
        if matches!(self.mb().types().get(of)?, Type::Struct { .. }) {
            let indices = indices.into();
            self.note_struct(y, StructOp::Extract { composite, indices });
        }
        Ok(y)
    }

    /// `OpCompositeInsert`: `object` must have the type at `indices`; the result has the
    /// composite's type.
    pub fn composite_insert(
        &mut self,
        object: Value,
        composite: Value,
        indices: &[u32],
    ) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::CompositeInsert;
        let ty = self.value_ty(composite)?;
        let at = self.walk_literal(OP, ty, indices).map_err(|e| match e {
            crate::spirv::error::BuildError::Operand {
                rule: OperandRule::Category(0),
                ..
            } => err(OP, OperandRule::Category(1)),
            e => e,
        })?;
        if self.value_ty(object)? != at {
            return Err(err(OP, OperandRule::Mismatch(0)));
        }
        if self.mb().non_uniform.contains(&object) {
            return Err(err(OP, OperandRule::NonUniform(0)));
        }
        let y = self.emit(
            Inst::CompositeInsert {
                object,
                composite,
                indices: indices.into(),
            },
            ty,
        )?;
        if matches!(self.mb().types().get(ty)?, Type::Struct { .. }) {
            let indices = indices.into();
            self.note_struct(
                y,
                StructOp::Insert {
                    object,
                    composite,
                    indices,
                },
            );
        }
        Ok(y)
    }

    /// `OpVectorExtractDynamic`: `index` is an integer scalar. Result: the component type.
    pub fn vector_extract_dynamic(&mut self, vector: Value, index: Value) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::VectorExtractDynamic;
        let v = self.want(OP, 0, vector, |n| n.is_vector())?;
        self.want(OP, 1, index, |n| n.is_int() && n.is_scalar())?;
        self.emit(Inst::VectorExtractDynamic { vector, index }, v.elem)
    }

    /// `OpVectorInsertDynamic`: `component` has the vector's component type, `index` is an integer
    /// scalar. Result: the vector's type.
    pub fn vector_insert_dynamic(
        &mut self,
        vector: Value,
        component: Value,
        index: Value,
    ) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::VectorInsertDynamic;
        let v = self.want(OP, 0, vector, |n| n.is_vector())?;
        if self.value_ty(component)? != v.elem {
            return Err(err(OP, OperandRule::Mismatch(1)));
        }
        self.want(OP, 2, index, |n| n.is_int() && n.is_scalar())?;
        self.emit(
            Inst::VectorInsertDynamic {
                vector,
                component,
                index,
            },
            v.ty,
        )
    }

    /// `OpVectorShuffle`: two vectors of one component type; 2 to 4 components, each selecting a
    /// component of the concatenation (undefined components are not offered). Result: a vector of
    /// the component type.
    pub fn vector_shuffle(&mut self, a: Value, b: Value, components: &[u32]) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::VectorShuffle;
        let x = self.want(OP, 0, a, |n| n.is_vector())?;
        let y = self.want(OP, 1, b, |n| n.is_vector())?;
        if y.elem != x.elem {
            return Err(err(OP, OperandRule::Mismatch(1)));
        }
        let len = u8::try_from(components.len())
            .ok()
            .filter(|l| (2..=4).contains(l))
            .ok_or(err(OP, OperandRule::Arity))?;
        let total = u32::from(x.n) + u32::from(y.n);
        if components.iter().any(|c| *c >= total) {
            return Err(err(OP, OperandRule::Range(2)));
        }
        let ty = self.vec_of(x.elem, len)?;
        self.emit(
            Inst::VectorShuffle {
                a,
                b,
                components: components.into(),
            },
            ty,
        )
    }
}
