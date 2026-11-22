//! Atomic, texel-pointer, and barrier constructors.
//!
//! Atomics operate only on pointers in `StorageBuffer`, `Workgroup`, `PhysicalStorageBuffer`,
//! `TaskPayloadWorkgroupEXT`, or `Image` (a texel pointer from
//! [`FunctionBuilder::image_texel_pointer`]). Scopes and semantics
//! are typed; the builder interns the 32-bit constants that carry them.

use crate::spirv::builder::function::FunctionBuilder;
use crate::spirv::builder::ops::{err, Num};
use crate::spirv::error::{BResult, BuildError, Dangling, ImageRule, OperandRule};
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    AtomicOp, Dim, ImageUsage, Inst, MemoryOrder, Scope, ScopeId, Semantics, SemanticsId,
    StorageClass, Type, ValueOrigin,
};

impl FunctionBuilder<'_> {
    /// The scope operand for `scope`.
    pub fn scope_id(&mut self, scope: Scope) -> BResult<ScopeId> {
        let id = self.u32_const(scope.to_spirv() as u32)?;
        Ok(ScopeId { scope, id })
    }

    /// The semantics operand for `sem`.
    pub fn semantics_id(&mut self, sem: Semantics) -> BResult<SemanticsId> {
        let id = self.u32_const(sem.bits())?;
        Ok(SemanticsId { sem, id })
    }

    /// The pointee shape of atomic pointer operand `ptr`: a 32- or 64-bit integer or a 16-, 32-,
    /// or 64-bit float scalar in an atomic-capable storage class, reached through a place.
    fn atomic_ptr(&self, op: crate::spv::Op, ptr: Value) -> BResult<Num> {
        let (class, pointee, _) = self.pointer(op, ptr)?;
        // `SPV_EXT_mesh_shader` adds an object function's payload to the atomics' classes.
        if !matches!(
            class,
            StorageClass::StorageBuffer
                | StorageClass::Workgroup
                | StorageClass::Image
                | StorageClass::PhysicalStorageBuffer
                | StorageClass::TaskPayloadWorkgroupEXT
        ) {
            return Err(err(op, OperandRule::Class(class)));
        }
        let n = match self.num_of(pointee)? {
            Some(n) if n.is_scalar() && n.is_int() && matches!(n.width, 32 | 64) => n,
            Some(n) if n.is_scalar() && n.is_float() => n,
            _ => return Err(err(op, OperandRule::Category(0))),
        };
        if !self.is_place(ptr)? {
            return Err(err(op, OperandRule::NotPlace));
        }
        Ok(n)
    }

    /// Whether pointer `ptr` is a place: a variable, or the result of an access chain or a texel
    /// pointer. SPIRV-Cross writes an atomic's pointer as `&expr`, which MSL accepts only for a
    /// place; a converted, selected, loaded, or parameter pointer is an rvalue there. The
    /// definition is looked up from the end of its block, where a lowering builds the pointer an
    /// atomic uses.
    ///
    /// **The bound**. The walk is the instructions between the atomic and its pointer's definition,
    /// and each lowering builds that pointer (an access chain, a texel pointer, or a device place)
    /// a few instructions before its atomics. Measured over all 14,780 indexed rows on
    /// `moltenvk-m3`, `vk12-floor`, and `lavapipe` (release, 232,436 checks), the walk is at most
    /// 5 instructions and 1.11 on average, so a per-value place fact would save nothing; the
    /// public places tests pin it (`atomic_place_scan_steps`). A lowering that reuses one place
    /// across a long block should add that fact instead.
    fn is_place(&self, ptr: Value) -> BResult<bool> {
        Ok(match self.origin(ptr)? {
            ValueOrigin::Global(_) | ValueOrigin::Var(_) => true,
            ValueOrigin::Inst(b) => self
                .mb()
                .blocks
                .get(b.index())
                .ok_or(BuildError::Dangling(Dangling::Block(b)))?
                .insts
                .iter()
                .rev()
                .find(|n| n.result == Some(ptr))
                .is_some_and(|n| n.inst.is_place()),
            _ => false,
        })
    }

    /// Semantics rules shared by atomics and barriers: an ordering names at least one storage
    /// class and a relaxed ordering none (VUID-StandaloneSpirv-MemorySemantics-10870/10871),
    /// `MakeAvailable` needs a release ordering and `MakeVisible` an acquire ordering, only
    /// atomics may be `Volatile`, and an `Invocation` memory scope takes no semantics at all
    /// (VUID-StandaloneSpirv-None-04641).
    fn check_semantics(
        &self,
        op: crate::spv::Op,
        mem: Scope,
        sem: Semantics,
        orders: &[MemoryOrder],
        atomic: bool,
    ) -> BResult<()> {
        let release = matches!(
            sem.order,
            MemoryOrder::Release | MemoryOrder::AcquireRelease
        );
        let acquire = matches!(
            sem.order,
            MemoryOrder::Acquire | MemoryOrder::AcquireRelease
        );
        let ok = orders.contains(&sem.order)
            && ((sem.order == MemoryOrder::Relaxed) != sem.has_storage())
            && (mem != Scope::Invocation || sem.bits() == 0)
            && (!sem.make_available || release)
            && (!sem.make_visible || acquire)
            && (!sem.volatile || atomic);
        if ok {
            Ok(())
        } else {
            Err(err(op, OperandRule::Semantics))
        }
    }

    /// `OpAtomicLoad` (relaxed or acquire). Result: the pointee type.
    pub fn atomic_load(&mut self, ptr: Value, scope: Scope, sem: Semantics) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::AtomicLoad;
        let p = self.atomic_ptr(OP, ptr)?;
        self.check_semantics(
            OP,
            scope,
            sem,
            &[MemoryOrder::Relaxed, MemoryOrder::Acquire],
            true,
        )?;
        let (scope, sem) = (self.scope_id(scope)?, self.semantics_id(sem)?);
        self.emit(Inst::AtomicLoad { ptr, scope, sem }, p.ty)
    }

    /// `OpAtomicStore` (relaxed or release) of a value of the pointee type.
    pub fn atomic_store(
        &mut self,
        ptr: Value,
        scope: Scope,
        sem: Semantics,
        value: Value,
    ) -> BResult<()> {
        const OP: crate::spv::Op = crate::spv::Op::AtomicStore;
        let p = self.atomic_ptr(OP, ptr)?;
        self.check_semantics(
            OP,
            scope,
            sem,
            &[MemoryOrder::Relaxed, MemoryOrder::Release],
            true,
        )?;
        if self.value_ty(value)? != p.ty {
            return Err(err(OP, OperandRule::Mismatch(3)));
        }
        let (scope, sem) = (self.scope_id(scope)?, self.semantics_id(sem)?);
        self.emit_void(Inst::AtomicStore {
            ptr,
            scope,
            sem,
            value,
        })
    }

    /// A read-modify-write atomic: `Exchange` on integers or floats, the float operations on
    /// floats, the rest on integers. `value` has the pointee type. Result: the pointee type (the
    /// old value).
    pub fn atomic(
        &mut self,
        op: AtomicOp,
        ptr: Value,
        scope: Scope,
        sem: Semantics,
        value: Value,
    ) -> BResult<Value> {
        let o = op.to_spirv();
        let p = self.atomic_ptr(o, ptr)?;
        let ok = match op {
            AtomicOp::Exchange => true,
            _ => p.is_int(),
        };
        if !ok {
            return Err(err(o, OperandRule::Category(0)));
        }
        self.check_semantics(o, scope, sem, ALL_ORDERS, true)?;
        if self.value_ty(value)? != p.ty {
            return Err(err(o, OperandRule::Mismatch(3)));
        }
        let (scope, sem) = (self.scope_id(scope)?, self.semantics_id(sem)?);
        self.emit(
            Inst::Atomic {
                op,
                ptr,
                scope,
                sem,
                value,
            },
            p.ty,
        )
    }

    /// `OpAtomicCompareExchange` on an integer: store `value` when the old value equals
    /// `comparator`. The unequal semantics are relaxed or acquire and never stronger than the
    /// equal semantics. Result: the old value.
    pub fn atomic_compare_exchange(
        &mut self,
        ptr: Value,
        scope: Scope,
        equal: Semantics,
        unequal: Semantics,
        value: Value,
        comparator: Value,
    ) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::AtomicCompareExchange;
        let p = self.atomic_ptr(OP, ptr)?;
        if !p.is_int() {
            return Err(err(OP, OperandRule::Category(0)));
        }
        self.check_semantics(OP, scope, equal, ALL_ORDERS, true)?;
        self.check_semantics(
            OP,
            scope,
            unequal,
            &[MemoryOrder::Relaxed, MemoryOrder::Acquire],
            true,
        )?;
        if unequal.order == MemoryOrder::Acquire
            && !matches!(
                equal.order,
                MemoryOrder::Acquire | MemoryOrder::AcquireRelease
            )
        {
            return Err(err(OP, OperandRule::Semantics));
        }
        if self.value_ty(value)? != p.ty {
            return Err(err(OP, OperandRule::Mismatch(4)));
        }
        if self.value_ty(comparator)? != p.ty {
            return Err(err(OP, OperandRule::Mismatch(5)));
        }
        let scope = self.scope_id(scope)?;
        let (equal, unequal) = (self.semantics_id(equal)?, self.semantics_id(unequal)?);
        self.emit(
            Inst::AtomicCompareExchange {
                ptr,
                scope,
                equal,
                unequal,
                value,
                comparator,
            },
            p.ty,
        )
    }

    /// `OpImageTexelPointer`: `image` points (in `UniformConstant`) to a storage or buffer image
    /// (storage texel buffers included) of format `R32ui`, `R32i`, `R32f`, `R64ui`, or `R64i` (Vulkan); `coord` is an integer with
    /// the image's texel coordinate count (cubes and cube arrays take 3); `sample` is an integer
    /// scalar, the constant 0 unless the image is multisampled. Result: an `Image`-class pointer
    /// to the sampled type, valid only as an atomic operand.
    pub fn image_texel_pointer(
        &mut self,
        image: Value,
        coord: Value,
        sample: Value,
    ) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::ImageTexelPointer;
        let (class, pointee, _) = self.pointer(OP, image)?;
        let img = match self.mb().types().get(pointee)? {
            Type::Image(img) if class == StorageClass::UniformConstant => *img,
            _ => return Err(err(OP, OperandRule::Category(0))),
        };
        let format_ok = matches!(
            img.format,
            crate::spv::ImageFormat::R32ui
                | crate::spv::ImageFormat::R32i
                | crate::spv::ImageFormat::R32f
                | crate::spv::ImageFormat::R64ui
                | crate::spv::ImageFormat::R64i
        );
        let usage_ok = img.usage == ImageUsage::Storage;
        if !format_ok || !usage_ok || img.dim == Dim::SubpassData {
            return Err(err(OP, OperandRule::Image(ImageRule::ImageType)));
        }
        let want = match img.dim {
            Dim::D1 => 1 + u8::from(img.arrayed),
            Dim::D2 => 2 + u8::from(img.arrayed),
            Dim::D3 | Dim::Cube => 3,
            Dim::Buffer | Dim::SubpassData => 1,
        };
        match self.num(coord)? {
            Some(c) if c.is_int() && c.n == want => {}
            _ => return Err(err(OP, OperandRule::Image(ImageRule::Coordinate))),
        }
        self.want(OP, 2, sample, |n| n.is_int() && n.is_scalar())?;
        if !img.multisampled && self.const_scalar(sample)? != Some(0) {
            return Err(err(OP, OperandRule::Range(2)));
        }
        let ty: TypeId = self.module().ty(Type::Pointer {
            class: StorageClass::Image,
            pointee: img.sampled_type,
        })?;
        self.emit(
            Inst::ImageTexelPointer {
                image,
                coord,
                sample,
            },
            ty,
        )
    }

    /// `OpControlBarrier`: execution scope `Workgroup` or `Subgroup` (Vulkan).
    pub fn control_barrier(&mut self, exec: Scope, mem: Scope, sem: Semantics) -> BResult<()> {
        const OP: crate::spv::Op = crate::spv::Op::ControlBarrier;
        if !matches!(exec, Scope::Workgroup | Scope::Subgroup) {
            return Err(err(OP, OperandRule::Scope));
        }
        self.check_semantics(OP, mem, sem, ALL_ORDERS, false)?;
        let (exec, mem) = (self.scope_id(exec)?, self.scope_id(mem)?);
        let sem = self.semantics_id(sem)?;
        self.emit_void(Inst::ControlBarrier { exec, mem, sem })
    }

    /// `OpMemoryBarrier`: the semantics carry an ordering and a storage class (Vulkan).
    pub fn memory_barrier(&mut self, mem: Scope, sem: Semantics) -> BResult<()> {
        const OP: crate::spv::Op = crate::spv::Op::MemoryBarrier;
        self.check_semantics(
            OP,
            mem,
            sem,
            &[
                MemoryOrder::Acquire,
                MemoryOrder::Release,
                MemoryOrder::AcquireRelease,
            ],
            false,
        )?;
        let mem = self.scope_id(mem)?;
        let sem = self.semantics_id(sem)?;
        self.emit_void(Inst::MemoryBarrier { mem, sem })
    }
}

const ALL_ORDERS: &[MemoryOrder] = &[
    MemoryOrder::Relaxed,
    MemoryOrder::Acquire,
    MemoryOrder::Release,
    MemoryOrder::AcquireRelease,
];
