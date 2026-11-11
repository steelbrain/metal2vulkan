//! Arithmetic, bitwise, logical, comparison, linear-algebra, select, and bit-field constructors.

use crate::spirv::builder::function::FunctionBuilder;
use crate::spirv::builder::module::ProductForm;
use crate::spirv::builder::ops::{err, Num};
use crate::spirv::error::{BResult, OperandRule};
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConstBits, ConstKind, Constant, Contraction, GlslOp, Inst, Shape, Type, UnOp,
    ValueOrigin,
};

impl FunctionBuilder<'_> {
    /// A unary operation. Result types: `IsNan`/`IsInf` give `bool` of the operand's component
    /// count, `Any`/`All` a `bool`, everything else the operand's type.
    pub fn unary(&mut self, op: UnOp, a: Value) -> BResult<Value> {
        let o = op.to_spirv();
        let ty = match op {
            UnOp::SNegate | UnOp::Not => self.want(o, 0, a, Num::is_int)?.ty,
            UnOp::FNegate => self.want(o, 0, a, Num::is_float)?.ty,
            UnOp::LogicalNot => self.want(o, 0, a, Num::is_bool)?.ty,
            UnOp::BitReverse | UnOp::BitCount => {
                // Vulkan: the base must be 32-bit (VUID-StandaloneSpirv-Base-04781).
                let x = self.want(o, 0, a, Num::is_int)?;
                if x.width != 32 {
                    return Err(err(o, OperandRule::Width(0)));
                }
                x.ty
            }
            UnOp::QuantizeToF16 => {
                let x = self.want(o, 0, a, Num::is_float)?;
                if x.width != 32 {
                    return Err(err(o, OperandRule::Width(0)));
                }
                x.ty
            }
            UnOp::IsNan | UnOp::IsInf => {
                let x = self.want(o, 0, a, Num::is_float)?;
                self.bool_ty(x.n)?
            }
            UnOp::Any | UnOp::All => {
                self.want(o, 0, a, |n| n.is_bool() && n.is_vector())?;
                self.bool_ty(1)?
            }
            UnOp::CopyObject => {
                let ty = self.value_ty(a)?;
                let f = *self.mb().types().facts(ty)?;
                if !matches!(f.shape, Shape::Plain | Shape::Implicit | Shape::Explicit)
                    || f.runtime_sized
                {
                    return Err(err(o, OperandRule::Category(0)));
                }
                ty
            }
        };
        self.emit(Inst::Unary { op, a }, ty)
    }

    /// A binary operation. Integer arithmetic and bitwise operations take integers of equal width
    /// and component count (signedness may differ) and give the first operand's type; float
    /// arithmetic takes two operands of one float type; shifts take integer operands of equal
    /// component count and give the base's type; comparisons give `bool` of the operands'
    /// component count; extended arithmetic gives `struct { T, T }`; `VectorTimesScalar` gives the
    /// vector's type and `Dot` the component type.
    pub fn binary(&mut self, op: BinOp, a: Value, b: Value) -> BResult<Value> {
        self.binary_with(op, a, b, Contraction::Allowed)
    }

    /// [`binary`](Self::binary) of float arithmetic decorated `NoContraction`: the device rounds
    /// the operation on its own and never fuses it with another. `op` must allow the decoration
    /// ([`BinOp::allows_no_contraction`]).
    ///
    /// A multiply (`FMul`) takes the module's [`ProductForm`]: an `OpFMul` decorated
    /// `NoContraction` where the device's driver compiles the SPIR-V itself, or `GLSL.std.450
    /// Fma(a, b, -0.0)` decorated `NoContraction` where it reaches Metal through SPIRV-Cross. `a *
    /// b + (-0.0)` is `a * b` exactly whether the device fuses it or rounds the product and then
    /// the sum (`x + (-0.0)` is `x` for every `x`, either zero included). SPIRV-Cross, which
    /// MoltenVK uses, renders a `NoContraction` `OpFMul` as `fma(l, r, 0)`, which answers `+0` for
    /// a `-0` product; it renders the `Fma` as `fma(a, b, -0.0)`. Elsewhere the `-0.0` addend is a
    /// value the device may answer: Mesa folds `Fma(0, b, -0.0)` to it under `nsz`, where Metal's
    /// multiply answers `+0`, and no per-instruction decoration keeps it from doing so on Mesa
    /// 26.2. The `Fma`'s decoration is still needed: Mesa splits a contractable `Fma` into a
    /// multiply and an add and folds the `-0.0` away, leaving a multiply that may be fused with a
    /// following add allowing contraction (NIR's `fuse` option on older Mesa, or a backend such as
    /// ACO). Mesa's SPIR-V reader marks a `NoContraction` result exact, so it is not split.
    pub fn binary_no_contraction(&mut self, op: BinOp, a: Value, b: Value) -> BResult<Value> {
        self.binary_with(op, a, b, Contraction::Forbidden)
    }

    /// A binary operation with contraction `contract` (see [`binary`](Self::binary)).
    pub fn binary_with(
        &mut self,
        op: BinOp,
        a: Value,
        b: Value,
        contract: Contraction,
    ) -> BResult<Value> {
        let o = op.to_spirv();
        if contract == Contraction::Forbidden && !op.allows_no_contraction() {
            return Err(err(o, OperandRule::NoContraction));
        }
        let mismatch = err(o, OperandRule::Mismatch(1));
        let ty = match op {
            BinOp::IAdd
            | BinOp::ISub
            | BinOp::IMul
            | BinOp::UDiv
            | BinOp::SDiv
            | BinOp::UMod
            | BinOp::SRem
            | BinOp::SMod
            | BinOp::BitwiseOr
            | BinOp::BitwiseXor
            | BinOp::BitwiseAnd => {
                // The result has the first operand's type; `OpUDiv`/`OpUMod` require an unsigned
                // result type (found by the P9a spirv-val sweep).
                let unsigned = matches!(op, BinOp::UDiv | BinOp::UMod);
                let x = self.want(o, 0, a, |n| n.is_int() && (n.is_uint() || !unsigned))?;
                let y = self.want(o, 1, b, Num::is_int)?;
                if !x.same_shape(&y) {
                    return Err(mismatch);
                }
                x.ty
            }
            BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem => {
                let x = self.want(o, 0, a, Num::is_float)?;
                if self.value_ty(b)? != x.ty {
                    return Err(mismatch);
                }
                x.ty
            }
            BinOp::ShiftRightLogical | BinOp::ShiftRightArithmetic | BinOp::ShiftLeftLogical => {
                let x = self.want(o, 0, a, Num::is_int)?;
                let y = self.want(o, 1, b, Num::is_int)?;
                if x.n != y.n {
                    return Err(mismatch);
                }
                x.ty
            }
            BinOp::LogicalEqual | BinOp::LogicalNotEqual | BinOp::LogicalOr | BinOp::LogicalAnd => {
                let x = self.want(o, 0, a, Num::is_bool)?;
                if self.value_ty(b)? != x.ty {
                    return Err(mismatch);
                }
                x.ty
            }
            BinOp::IEqual
            | BinOp::INotEqual
            | BinOp::UGreaterThan
            | BinOp::SGreaterThan
            | BinOp::UGreaterThanEqual
            | BinOp::SGreaterThanEqual
            | BinOp::ULessThan
            | BinOp::SLessThan
            | BinOp::ULessThanEqual
            | BinOp::SLessThanEqual => {
                let x = self.want(o, 0, a, Num::is_int)?;
                let y = self.want(o, 1, b, Num::is_int)?;
                if !x.same_shape(&y) {
                    return Err(mismatch);
                }
                self.bool_ty(x.n)?
            }
            BinOp::FOrdEqual
            | BinOp::FUnordEqual
            | BinOp::FOrdNotEqual
            | BinOp::FUnordNotEqual
            | BinOp::FOrdLessThan
            | BinOp::FUnordLessThan
            | BinOp::FOrdGreaterThan
            | BinOp::FUnordGreaterThan
            | BinOp::FOrdLessThanEqual
            | BinOp::FUnordLessThanEqual
            | BinOp::FOrdGreaterThanEqual
            | BinOp::FUnordGreaterThanEqual => {
                let x = self.want(o, 0, a, Num::is_float)?;
                if self.value_ty(b)? != x.ty {
                    return Err(mismatch);
                }
                self.bool_ty(x.n)?
            }
            BinOp::IAddCarry | BinOp::ISubBorrow | BinOp::UMulExtended | BinOp::SMulExtended => {
                // The struct members, and so both operands, share one type; all but
                // `SMulExtended` require it unsigned.
                let unsigned = op != BinOp::SMulExtended;
                let x = self.want(o, 0, a, |n| n.is_int() && (n.is_uint() || !unsigned))?;
                if self.value_ty(b)? != x.ty {
                    return Err(mismatch);
                }
                self.module().ty(Type::Struct {
                    members: [x.ty, x.ty].into(),
                    layout: None,
                })?
            }
            BinOp::VectorTimesScalar => {
                let x = self.want(o, 0, a, |n| n.is_float() && n.is_vector())?;
                if self.value_ty(b)? != x.elem {
                    return Err(mismatch);
                }
                x.ty
            }
            BinOp::Dot => {
                let x = self.want(o, 0, a, |n| n.is_float() && n.is_vector())?;
                if self.value_ty(b)? != x.ty {
                    return Err(mismatch);
                }
                x.elem
            }
        };
        if let Some(r) = self.fold_binary(op, a, b, ty)? {
            return Ok(r);
        }
        if matches!(op, BinOp::FMul | BinOp::VectorTimesScalar)
            && self.mb().product_form == ProductForm::PositiveZeroFactors
        {
            if let Some(r) = self.product_by_positive_zeros(op, a, b, contract, ty)? {
                return Ok(r);
            }
        }
        if op == BinOp::FMul
            && contract == Contraction::Forbidden
            && self.mb().product_form == ProductForm::NegativeZeroFma
        {
            let z = self.negative_zero(a)?;
            return self.glsl_with(GlslOp::Fma, &[a, b, z], Contraction::Forbidden);
        }
        self.emit(Inst::Binary { op, a, b, contract }, ty)
    }

    /// Under [`ProductForm::PositiveZeroFactors`], the product `op` of `a` and `b` where one of them
    /// is an `f32` constant with a `-0.0` component: the product by that constant with each such
    /// component `+0.0`, negated in those components of the result (`OpFNegate`, and an
    /// `OpVectorShuffle` of the product and its negation where only some components are). `-(x ·
    /// +0)` is `x · (−0)` for every `x`. `None` where neither factor is such a constant.
    fn product_by_positive_zeros(
        &mut self,
        op: BinOp,
        a: Value,
        b: Value,
        contract: Contraction,
        ty: TypeId,
    ) -> BResult<Option<Value>> {
        for (index, factor) in [(1, b), (0, a)] {
            let Some(negative) = self.negative_zero_components(factor)? else {
                continue;
            };
            let positive = self.positive_zeros(factor, &negative)?;
            let (x, y) = if index == 1 {
                (a, positive)
            } else {
                (positive, b)
            };
            let product = self.emit(
                Inst::Binary {
                    op,
                    a: x,
                    b: y,
                    contract,
                },
                ty,
            )?;
            let negated = self.unary(UnOp::FNegate, product)?;
            // A scalar factor of `OpVectorTimesScalar` scales every component.
            if negative.iter().all(|n| *n) {
                return Ok(Some(negated));
            }
            // A vector's components, at most four: the negation's are numbered after them.
            let len = negative.len() as u32;
            let pick: Vec<u32> = (0..len)
                .zip(&negative)
                .map(|(i, n)| if *n { len + i } else { i })
                .collect();
            return self.vector_shuffle(product, negated, &pick).map(Some);
        }
        Ok(None)
    }

    /// Which components of `v` are `-0.0`, when `v` is a plain `f32` scalar or vector constant
    /// with at least one; `None` otherwise.
    fn negative_zero_components(&self, v: Value) -> BResult<Option<Vec<bool>>> {
        const NEGATIVE_ZERO: u64 = 0x8000_0000;
        let ValueOrigin::Const(c) = self.origin(v)? else {
            return Ok(None);
        };
        let Some(n) = self.num(v)? else {
            return Ok(None);
        };
        if !n.is_float() || n.width != 32 {
            return Ok(None);
        }
        let consts = self.mb().consts();
        let negative = match &consts.get(c)?.kind {
            ConstKind::Plain(Constant::Scalar { bits, .. }) => vec![bits.0 == NEGATIVE_ZERO],
            ConstKind::Plain(Constant::Composite { parts, .. }) => parts
                .iter()
                .map(|p| consts.scalar_bits(*p).is_some_and(|b| b.0 == NEGATIVE_ZERO))
                .collect(),
            _ => return Ok(None),
        };
        Ok(negative.contains(&true).then_some(negative))
    }

    /// The constant `v` (a plain `f32` scalar or vector constant) with its components where
    /// `negative` holds made `+0.0`.
    fn positive_zeros(&mut self, v: Value, negative: &[bool]) -> BResult<Value> {
        let ValueOrigin::Const(c) = self.origin(v)? else {
            return Ok(v);
        };
        let def = self.mb().consts().get(c)?.clone();
        let positive = match def.kind {
            ConstKind::Plain(Constant::Scalar { ty, .. }) => Constant::Scalar {
                ty,
                bits: ConstBits(0),
            },
            ConstKind::Plain(Constant::Composite { ty, parts }) => {
                let elem = self.want(crate::spv::Op::FMul, 0, v, Num::is_float)?.elem;
                let zero = self.module().constant(Constant::Scalar {
                    ty: elem,
                    bits: ConstBits(0),
                })?;
                let parts = parts
                    .iter()
                    .zip(negative)
                    .map(|(p, n)| if *n { zero } else { *p })
                    .collect();
                Constant::Composite { ty, parts }
            }
            _ => return Ok(v),
        };
        let id = self.module().constant(positive)?;
        self.module().const_value(id)
    }

    /// The `-0.0` of `v`'s float scalar or vector type: the sign bit alone in each component.
    fn negative_zero(&mut self, v: Value) -> BResult<Value> {
        let x = self.want(crate::spv::Op::FMul, 0, v, Num::is_float)?;
        let s = self.module().constant(Constant::Scalar {
            ty: x.elem,
            bits: ConstBits(1u64 << (x.width - 1)),
        })?;
        let c = if x.n == 1 {
            s
        } else {
            self.module().constant(Constant::Composite {
                ty: x.ty,
                parts: vec![s; usize::from(x.n)].into(),
            })?
        };
        self.module().const_value(c)
    }

    /// The value of `v` when it is a plain `bool` constant.
    pub fn const_bool(&self, v: Value) -> BResult<Option<bool>> {
        Ok(match self.scalar_const(v)? {
            Some((b, 1, _)) => Some(b != 0),
            _ => None,
        })
    }

    /// The plain scalar constant bits and width of `v` (a `bool` is width 1), if it is one.
    fn scalar_const(&self, v: Value) -> BResult<Option<(u64, u8, bool)>> {
        let ValueOrigin::Const(c) = self.origin(v)? else {
            return Ok(None);
        };
        let def = self.mb().consts().get(c)?;
        let bits = match &def.kind {
            ConstKind::Plain(Constant::Scalar { bits, .. }) => bits.0,
            ConstKind::Plain(Constant::Null { .. }) => 0,
            _ => return Ok(None),
        };
        Ok(match self.mb().types().get(def.ty)? {
            Type::Bool => Some((bits, 1, false)),
            Type::Int { width, signed } => Some((bits, *width, *signed)),
            _ => None,
        })
    }

    /// `op` on scalar integer or `bool` constants `a` and `b`, computed at translation time as
    /// the device computes it: integer arithmetic wraps at the operands' width, a shift folds only
    /// by less than the width (a wider shift's result is undefined, so it is left to the device),
    /// and divisions are not folded. `None` when either operand is not such a constant or `op` does
    /// not fold.
    fn fold_binary(&mut self, op: BinOp, a: Value, b: Value, ty: TypeId) -> BResult<Option<Value>> {
        let (Some((x, w, _)), Some((y, _, _))) = (self.scalar_const(a)?, self.scalar_const(b)?)
        else {
            return Ok(None);
        };
        let mask = if w >= 64 { u64::MAX } else { (1u64 << w) - 1 };
        let sx = |v: u64| -> i64 {
            if w >= 64 {
                v as i64
            } else {
                ((v << (64 - u32::from(w))) as i64) >> (64 - u32::from(w))
            }
        };
        let flag = |c: bool| Some(u64::from(c));
        let r = match op {
            BinOp::IAdd => Some(x.wrapping_add(y) & mask),
            BinOp::ISub => Some(x.wrapping_sub(y) & mask),
            BinOp::IMul => Some(x.wrapping_mul(y) & mask),
            BinOp::BitwiseAnd => Some(x & y),
            BinOp::BitwiseOr => Some(x | y),
            BinOp::BitwiseXor => Some(x ^ y),
            BinOp::ShiftLeftLogical if y < u64::from(w) => Some((x << y) & mask),
            BinOp::ShiftRightLogical if y < u64::from(w) => Some(x >> y),
            BinOp::ShiftRightArithmetic if y < u64::from(w) => Some((sx(x) >> y) as u64 & mask),
            BinOp::IEqual | BinOp::LogicalEqual => flag(x == y),
            BinOp::INotEqual | BinOp::LogicalNotEqual => flag(x != y),
            BinOp::UGreaterThan => flag(x > y),
            BinOp::UGreaterThanEqual => flag(x >= y),
            BinOp::ULessThan => flag(x < y),
            BinOp::ULessThanEqual => flag(x <= y),
            BinOp::SGreaterThan => flag(sx(x) > sx(y)),
            BinOp::SGreaterThanEqual => flag(sx(x) >= sx(y)),
            BinOp::SLessThan => flag(sx(x) < sx(y)),
            BinOp::SLessThanEqual => flag(sx(x) <= sx(y)),
            BinOp::LogicalAnd => flag(x != 0 && y != 0),
            BinOp::LogicalOr => flag(x != 0 || y != 0),
            _ => None,
        };
        let Some(r) = r else {
            return Ok(None);
        };
        let c = self.module().constant(Constant::Scalar {
            ty,
            bits: ConstBits(r),
        })?;
        self.module().const_value(c).map(Some)
    }

    /// `OpSelect`: `cond` is a `bool` or a `bool` vector with the objects' component count; the
    /// objects share one data type (no opaque value or logical pointer). Result: that type.
    pub fn select(&mut self, cond: Value, t: Value, f: Value) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::Select;
        let c = self.want(OP, 0, cond, Num::is_bool)?;
        let ty = self.value_ty(t)?;
        let facts = *self.mb().types().facts(ty)?;
        if !matches!(
            facts.shape,
            Shape::Plain | Shape::Implicit | Shape::Explicit
        ) || facts.runtime_sized
        {
            return Err(err(OP, OperandRule::Category(1)));
        }
        if self.value_ty(f)? != ty {
            return Err(err(OP, OperandRule::Mismatch(2)));
        }
        if c.is_vector() {
            match self.num_of(ty)? {
                Some(x) if x.n == c.n => {}
                _ => return Err(err(OP, OperandRule::Mismatch(0))),
            }
        }
        // A constant condition selects at translation time.
        if let Some((b, 1, _)) = self.scalar_const(cond)? {
            return Ok(if b != 0 { t } else { f });
        }
        self.emit(Inst::Select { cond, t, f }, ty)
    }

    /// `OpFmaKHR` (`SPV_KHR_fma`): `a * b + c` with one rounding, which Vulkan's precision table
    /// makes correctly rounded. The three operands share one 16- or 32-bit float scalar or vector
    /// type, the widths the core emits it at (its feature is `shaderFmaFloat16` or
    /// `shaderFmaFloat32`). Result: that type.
    pub fn fma(&mut self, a: Value, b: Value, c: Value) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::FmaKHR;
        let x = self.want(OP, 0, a, Num::is_float)?;
        if x.width != 32 && x.width != 16 {
            return Err(err(OP, OperandRule::Width(0)));
        }
        if self.value_ty(b)? != x.ty {
            return Err(err(OP, OperandRule::Mismatch(1)));
        }
        if self.value_ty(c)? != x.ty {
            return Err(err(OP, OperandRule::Mismatch(2)));
        }
        self.emit(Inst::Fma { a, b, c }, x.ty)
    }

    /// `OpBitFieldInsert`: `base` and `insert` share one 32-bit integer type (Vulkan), `offset`
    /// and `count` are integer scalars. Result: the base's type.
    pub fn bit_field_insert(
        &mut self,
        base: Value,
        insert: Value,
        offset: Value,
        count: Value,
    ) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::BitFieldInsert;
        let ty = self.bit_field_base(OP, base)?;
        if self.value_ty(insert)? != ty {
            return Err(err(OP, OperandRule::Mismatch(1)));
        }
        self.want(OP, 2, offset, |n| n.is_int() && n.is_scalar())?;
        self.want(OP, 3, count, |n| n.is_int() && n.is_scalar())?;
        self.emit(
            Inst::BitFieldInsert {
                base,
                insert,
                offset,
                count,
            },
            ty,
        )
    }

    /// `OpBitFieldSExtract` (`signed`) or `OpBitFieldUExtract`: `base` is a 32-bit integer scalar
    /// or vector (Vulkan), `offset` and `count` integer scalars. Result: the base's type.
    pub fn bit_field_extract(
        &mut self,
        signed: bool,
        base: Value,
        offset: Value,
        count: Value,
    ) -> BResult<Value> {
        let op = if signed {
            crate::spv::Op::BitFieldSExtract
        } else {
            crate::spv::Op::BitFieldUExtract
        };
        let ty = self.bit_field_base(op, base)?;
        self.want(op, 1, offset, |n| n.is_int() && n.is_scalar())?;
        self.want(op, 2, count, |n| n.is_int() && n.is_scalar())?;
        self.emit(
            Inst::BitFieldExtract {
                signed,
                base,
                offset,
                count,
            },
            ty,
        )
    }

    fn bit_field_base(&self, op: crate::spv::Op, base: Value) -> BResult<TypeId> {
        let x = self.want(op, 0, base, Num::is_int)?;
        if x.width != 32 {
            return Err(err(op, OperandRule::Width(0)));
        }
        Ok(x.ty)
    }
}
