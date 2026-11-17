//! Memory and call constructors: access chains, runtime-array length, memory copies, and function
//! calls. Loads and stores are in `function.rs`.

use crate::spirv::builder::function::{not_texel_pointer, FunctionBuilder};
use crate::spirv::builder::ops::{err, index};
use crate::spirv::error::{BResult, BuildError, ControlRule, OperandRule};
use crate::spirv::ids::{FuncId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, ConvOp, Inst, LayoutRule, Shape, StorageClass, StructLayout, Type, ValueOrigin,
};

impl FunctionBuilder<'_> {
    /// `OpAccessChain`: walk `indices` (integer scalars) from `base`'s pointee. Struct indices
    /// must be 32-bit integer constants in range; constant indices into vectors, matrices, and
    /// sized arrays must be in range. Result: a pointer of `base`'s storage class to the reached
    /// type. `Image`-class texel pointers cannot be indexed.
    pub fn access_chain(&mut self, base: Value, indices: &[Value]) -> BResult<Value> {
        self.chain(base, indices, false)
    }

    /// `OpAccessChain` of element `index` of the descriptor array `base` points to, where `index`
    /// may differ between invocations (a non-uniform index), and the module requires the array
    /// kind's non-uniform indexing capability and feature. `base` must point to a `UniformConstant`
    /// array of images (not subpass data) or samplers, or a `StorageBuffer` array of blocks. The
    /// serializer decorates `NonUniform` the result and every value that carries the element to the
    /// instruction consuming it (VUID-RuntimeSpirv-NonUniform-06274): a load of an image or sampler
    /// element, an image texel pointer to it, a sampled image of a loaded element, the image of
    /// such a sampled image, and a further access chain into a block element, which a load, a
    /// store, an atomic, or `OpArrayLength` then reads through. The builder refuses every other use
    /// of those values that would carry the element on (a phi, select, copy, store, composite part,
    /// or call argument), and an image or sampler element is not indexable further. A value loaded
    /// from a block element is data, which carries no element.
    pub fn access_chain_non_uniform(&mut self, base: Value, index: Value) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::AccessChain;
        let (class, pointee, f) = self.pointer(OP, base)?;
        if !matches!(f.shape, Shape::DescriptorArray { .. }) {
            return Err(err(OP, OperandRule::Category(0)));
        }
        let elem = match self.mb().types().get(pointee)? {
            Type::Array { elem, .. } | Type::RuntimeArray { elem, .. } => *elem,
            _ => return Err(err(OP, OperandRule::Category(0))),
        };
        match (class, self.mb().types().get(elem)?) {
            (StorageClass::UniformConstant, Type::Image(img))
                if img.dim != crate::spirv::ir::Dim::SubpassData => {}
            (StorageClass::UniformConstant, Type::Sampler) => {}
            (StorageClass::StorageBuffer, Type::Struct { .. })
                if self.mb().types().facts(elem)?.block => {}
            (StorageClass::UniformConstant | StorageClass::StorageBuffer, _) => {
                return Err(err(OP, OperandRule::Category(0)))
            }
            (other, _) => return Err(err(OP, OperandRule::Class(other))),
        }
        self.chain(base, &[index], true)
    }

    fn chain(&mut self, base: Value, indices: &[Value], non_uniform: bool) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::AccessChain;
        let (class, pointee, _) = self.pointer(OP, base)?;
        not_texel_pointer(OP, class)?;
        if indices.is_empty() {
            return Err(err(OP, OperandRule::Arity));
        }
        let mut cur = pointee;
        for (i, idx) in indices.iter().enumerate() {
            let pos = index(i + 1);
            let x = self.want(OP, pos, *idx, |n| n.is_int() && n.is_scalar())?;
            let constant = self.const_scalar(*idx)?;
            let types = self.mb().types();
            let (next, len) = match types.get(cur)? {
                Type::Struct { members, .. } => {
                    let Some(k) = constant else {
                        return Err(err(OP, OperandRule::NotConstant(pos)));
                    };
                    if x.width != 32 {
                        return Err(err(OP, OperandRule::Width(pos)));
                    }
                    let m = usize::try_from(k)
                        .ok()
                        .and_then(|k| members.get(k))
                        .copied()
                        .ok_or(err(OP, OperandRule::Range(pos)))?;
                    (m, None)
                }
                Type::Vector { elem, len } => (*elem, Some(u64::from(*len))),
                Type::Matrix { col, cols } => (*col, Some(u64::from(*cols))),
                Type::Array { elem, len, .. } => (
                    *elem,
                    self.mb().consts().array_len(*len, types)?.map(u64::from),
                ),
                Type::RuntimeArray { elem, .. } => (*elem, None),
                _ => return Err(err(OP, OperandRule::Category(pos))),
            };
            if let (Some(k), Some(len)) = (constant, len) {
                // A signed index constant is compared by its two's-complement bits sign-extended:
                // a negative index is out of range.
                let k = sign_extend(k, x.width, x.is_uint());
                if k < 0 || k as u64 >= len {
                    return Err(err(OP, OperandRule::Range(pos)));
                }
            }
            cur = next;
        }
        let ty = self.module().ty(Type::Pointer {
            class,
            pointee: cur,
        })?;
        self.emit(
            Inst::AccessChain {
                base,
                indices: indices.into(),
                non_uniform,
            },
            ty,
        )
    }

    /// A place of type `pointee` at device address `addr` (a 64-bit integer scalar), as a
    /// `PhysicalStorageBuffer` pointer to `pointee`: `addr` converted to a pointer to the
    /// one-member `Block { pointee }` at offset 0, and the access chain to that member. Both
    /// pointers denote `addr`. The member is GLSL's `buffer_reference` shape, which SPIRV-Cross
    /// writes as a place (`ptr->m`); a bare `OpConvertUToPtr` result is an rvalue in its MSL, and
    /// an atomic needs its operand's address. Every device-address pointer the core builds comes
    /// from here. The block takes the pointee's own layout rule, at most std430, so it needs no
    /// layout feature the pointee does not.
    pub fn device_place(&mut self, addr: Value, pointee: TypeId) -> BResult<Value> {
        self.place_at(addr, pointee, false)
    }

    /// [`device_place`](Self::device_place) for memory nothing writes while the module runs: its
    /// block's member is `NonWritable`, so the device may read it as read-only memory (RADV reads a
    /// uniform address of it with a scalar load). Only loads go through it.
    pub fn device_place_read_only(&mut self, addr: Value, pointee: TypeId) -> BResult<Value> {
        self.place_at(addr, pointee, true)
    }

    fn place_at(&mut self, addr: Value, pointee: TypeId, non_writable: bool) -> BResult<Value> {
        let rule = self
            .mb()
            .types()
            .facts(pointee)?
            .max_rule()
            .min(LayoutRule::Std430);
        let layout = StructLayout {
            non_writable,
            ..natural_layout(self.mb().types(), rule, true, &[pointee])?
        };
        let block = self.module().ty(Type::Struct {
            members: [pointee].into(),
            layout: Some(layout),
        })?;
        let ptr = self.module().ty(Type::Pointer {
            class: StorageClass::PhysicalStorageBuffer,
            pointee: block,
        })?;
        let base = self.convert(ConvOp::ConvertUToPtr, addr, ptr)?;
        let zero = self.u32_const(0)?;
        let zero = self.module().const_value(zero)?;
        self.access_chain(base, &[zero])
    }

    /// `OpArrayLength`: `ptr` points (in `StorageBuffer`) to a `Block` whose last member,
    /// `member`, is a runtime array. Result: `u32`.
    pub fn array_length(&mut self, ptr: Value, member: u32) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::ArrayLength;
        let (class, pointee, f) = self.pointer(OP, ptr)?;
        if class != StorageClass::StorageBuffer || !f.block {
            return Err(err(OP, OperandRule::Category(0)));
        }
        let last_is_rta = match self.mb().types().get(pointee)? {
            Type::Struct { members, .. } => match members.last() {
                Some(m) if members.len() - 1 == member as usize => {
                    matches!(self.mb().types().get(*m)?, Type::RuntimeArray { .. })
                }
                _ => false,
            },
            _ => false,
        };
        if !last_is_rta {
            return Err(err(OP, OperandRule::Range(1)));
        }
        let ty = self.uint_ty(32)?;
        self.emit(Inst::ArrayLength { ptr, member }, ty)
    }

    /// `OpFunctionCall` of finished function `func` (not this one) with `args` of its parameter
    /// types. A logical-pointer argument must be the pointer of a `UniformConstant`, `Function`,
    /// `Private`, or `Workgroup` variable (a memory object declaration), and no argument may carry
    /// a non-uniform descriptor element. Returns the result, or `None` for a void callee.
    pub fn call(&mut self, func: FuncId, args: &[Value]) -> BResult<Option<Value>> {
        const OP: crate::spv::Op = crate::spv::Op::FunctionCall;
        if func == self.id() {
            return Err(err(OP, OperandRule::Callee));
        }
        let d = self.mb().draft(func)?;
        if d.fn_ty.is_none() {
            return Err(BuildError::Control(ControlRule::UnfinishedFunction(func)));
        }
        let (ret, params) = (d.ret, d.param_tys.clone());
        if params.len() != args.len() {
            return Err(err(OP, OperandRule::Arity));
        }
        for (i, (a, p)) in args.iter().zip(&params).enumerate() {
            if self.value_ty(*a)? != *p {
                return Err(err(OP, OperandRule::Mismatch(index(i))));
            }
            if self.mb().non_uniform.contains(a) {
                return Err(err(OP, OperandRule::NonUniform(index(i))));
            }
            self.check_pointer_arg(OP, index(i), *a, *p)?;
        }
        let inst = Inst::Call {
            func,
            args: args.into(),
        };
        if matches!(self.mb().types().get(ret)?, Type::Void) {
            self.emit_void(inst)?;
            Ok(None)
        } else {
            self.emit(inst, ret).map(Some)
        }
    }

    fn check_pointer_arg(&self, op: crate::spv::Op, i: u8, a: Value, ty: TypeId) -> BResult<()> {
        let Type::Pointer { class, .. } = self.mb().types().get(ty)? else {
            return Ok(());
        };
        let ok = match class {
            StorageClass::PhysicalStorageBuffer => true,
            StorageClass::UniformConstant
            | StorageClass::Function
            | StorageClass::Private
            | StorageClass::Workgroup => {
                matches!(
                    self.origin(a)?,
                    ValueOrigin::Global(_) | ValueOrigin::Var(_)
                )
            }
            _ => false,
        };
        if ok {
            Ok(())
        } else {
            Err(err(op, OperandRule::Category(i)))
        }
    }
}

/// `bits` of a `width`-bit integer as `i128`, sign-extended unless `unsigned`.
fn sign_extend(bits: u64, width: u8, unsigned: bool) -> i128 {
    if unsigned {
        return i128::from(bits);
    }
    let shift = 64 - u32::from(width.min(64));
    i128::from(((bits << shift) as i64) >> shift)
}
