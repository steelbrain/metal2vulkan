//! Texture atomics: `atomic_<op>_explicit` on a `read_write` 1D, 1D array, 2D, 2D array, 3D, or
//! buffer texture.
//!
//! The operand layouts are the decoder's (`air::intrinsics::texture`), read from Apple's
//! compiler: a 2D or 3D texture passes the zero vector at the position a read passes its offset,
//! and every operation the memory order `0` (`memory_order_relaxed`, MSL's only order for
//! texture atomics), a compare-exchange its failure order too. Any other value of either is not
//! modelled.
//!
//! Metal's texture atomics need an `R32Uint` or `R32Sint` texture of the texel's numeric type,
//! and use the value's first component. Each is one SPIR-V atomic at `Device` scope, relaxed, on
//! the `OpImageTexelPointer` of the texel, so the slot is declared `R32ui` / `R32i` (the
//! choice's format, or Metal's rule without one): the read-modify-writes `OpAtomic{IAdd, ISub,
//! And, Or, Xor}`, `OpAtomic{S,U}{Min,Max}` by the texel's sign, `exchange` `OpAtomicExchange`,
//! and `load` `OpAtomicLoad`. A `store` is an `OpAtomicExchange` whose old texel is never read:
//! the same relaxed write, and a form every device's compiler takes (SPIRV-Cross's MSL for an
//! image texel pointer's `OpAtomicStore` does not compile, measured on MoltenVK). The result is
//! Metal's `(old, 0, 0, 1)`, the old texel's first component then an `R32` texel's read
//! expansion, inside the texture and outside it (measured on Apple M3 Max, authored cases
//! `be2717e3` and `4d76953a`).
//!
//! `compare_exchange_weak` reads its comparator, lane 0 of the texel `expected` points to (a
//! private-arena leaf, as a memory compare-exchange's), is `OpAtomicCompareExchange` (SPIR-V's
//! exchange is strong, a valid weak one), succeeds when the old texel equals the comparator, and
//! writes the old texel back to lane 0 of `expected`, its other lanes as passed (Metal writes
//! lane 0 only, measured in the same cases).
//!
//! **`ulong`**. Metal's `atomic_{min,max}` of a `ulong` texture compare the texel's unsigned 64-bit
//! value, the low word in R and the high word in G of the `RG32Uint` texture Metal binds, with lane
//! 0 of the operand, and return nothing (measured on Apple M3 Max, 4096 contending threads exact).
//! Each is refused on every device (`Unsupported{ImageAtomics}`, not implemented): its one exact
//! form is a 64-bit image atomic (`shaderImageInt64Atomics`), which no runtime enables. A 64-bit
//! compare-and-update of an image texel is one atomic, which neither two 32-bit word atomics nor a
//! lock (Vulkan promises no forward progress to a spinning invocation) can make, and an optimally
//! tiled image has no memory a buffer atomic could reach.
//!
//! **Out of bounds**. Metal returns 0 from an atomic outside the texture and changes nothing, and a
//! compare-exchange there fails and leaves `expected` as passed. A device under robust image access
//! (texel buffer access for a texture buffer) does the former, so there an atomic is left to it;
//! elsewhere it is guarded and yields 0 (`slot::write_guard`). A compare-exchange always tests the
//! texel against the image's size: the device's zero would compare equal to a zero comparator.

use crate::air::intrinsics::{AtomicOp as AirOp, ImageDim, ImageShape, MemFlags, TexRole};
use crate::air::ir::{Inst, InstId, Value, ValueId};
use crate::error::{Error, Feature, Unsupported};
use crate::lower::access::atomic::spin;
use crate::lower::access::local::{self, LocalAccess};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::sync::barrier;
use crate::memory::arena::Arena;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, Value as SValue};
use crate::spirv::ir::{
    AtomicOp, BinOp, ConstBits, Constant, ConvOp, MemAccess, MemoryOrder, Scope, Semantics,
    Type as SType, UnOp,
};

use super::{
    const_int, coord, id, image_access, is_zero, mark, refuse, slot, Coord, ImageRef, Operands,
    TexelClass, TexelConv, TexelForm,
};

/// A texture atomic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TexAtomic {
    pub image: ImageRef,
    pub at: Coord,
    pub op: AirOp,
    /// The operand, or a compare-exchange's desired texel (a held `<4 x i32>`; its lane 0 is
    /// used); `None` for a load.
    pub value: Option<ValueId>,
    /// A compare-exchange's `expected` texel, read for the comparator and written back.
    pub expected: Option<LocalAccess>,
    /// The texel is signed (`OpAtomicS{Min,Max}` on an `R32i` image).
    pub signed: bool,
    /// Leave a texel outside the image alone (no robustness discards the access).
    pub guard: bool,
    /// The result's lowered type: a texel, of which only lane 0 is observed, or a
    /// compare-exchange's `bool`; `None` for a store.
    pub result: Option<DemoteTy>,
    /// A load an exit of a loop holding it depends on (`access::atomic::spin`): it acquires image
    /// memory.
    pub spin: bool,
}

impl TexAtomic {
    /// The census name: `texture.atomic_<op>[_guarded]`, the operation as MSL spells it.
    pub fn name(&self) -> String {
        let op = match self.op {
            AirOp::Add => "add",
            AirOp::Sub => "sub",
            AirOp::And => "and",
            AirOp::Or => "or",
            AirOp::Xor => "xor",
            AirOp::Min => "min",
            AirOp::Max => "max",
            AirOp::Xchg => "exchange",
            AirOp::Load if self.spin => "spin_load",
            AirOp::Load => "load",
            AirOp::Store => "store",
            AirOp::CmpXchgWeak => "compare_exchange",
        };
        let guarded = if self.guard { "_guarded" } else { "" };
        format!("texture.atomic_{op}{guarded}")
    }

    /// The estimated words: the texel pointer, the atomic, the operand and result conversions, and
    /// the guard; a compare-exchange's in-bounds test, comparison, and the accesses of `expected`.
    pub fn words(&self) -> u32 {
        let guard = if self.guard { 48 } else { 0 };
        let expected = self
            .expected
            .as_ref()
            .map_or(0, |e| e.leaf.cost().saturating_mul(2).saturating_add(160));
        40u32.saturating_add(guard).saturating_add(expected)
    }
}

/// Select texture atomic `inst` (operation `op`) on `image`.
pub(super) fn select(
    cx: &mut SelectCx<'_>,
    (at, inst): (InstId, &Inst),
    op: AirOp,
    ops: &Operands<'_>,
    (image, shape): (ImageRef, ImageShape),
    texel: (TexelForm, TexelClass),
) -> Result<super::TextureStrategy, Error> {
    let ImageRef::Slot(_) = image else {
        // A heap storage class is formatless; an atomic needs an `R32` one.
        return Err(Unsupported::not_implemented(Feature::ArgumentBufferTextures).into());
    };
    // Metal's texture atomics take 32-bit integer texels, and a `ulong` texel's `min` and `max`,
    // which no form is built for (the module's docs).
    match texel.0.conv {
        TexelConv::I32 => {}
        TexelConv::U64Words { .. } if matches!(op, AirOp::Min | AirOp::Max) => {
            return Err(Unsupported::not_implemented(Feature::ImageAtomics).into());
        }
        _ => return Err(refuse()),
    }
    let signed = texel.1 == TexelClass::Sint;
    // The positions whose only established value is zero.
    let zero_offset = ops.get(TexRole::Offset).is_none_or(|v| is_zero(cx, v));
    let relaxed = |role| ops.get(role).is_none_or(|v| const_int(cx, v) == Some(0));
    let ordered = ops.get(TexRole::Order).is_some() && relaxed(TexRole::Order);
    if !(zero_offset && ordered && relaxed(TexRole::FailureOrder)) || !image_access(cx, image)?.0 {
        return Err(refuse());
    }
    let expected = ops.get(TexRole::Expected);
    slot::stage_stores(cx)?;
    let value = match ops.get(TexRole::Value) {
        Some(v) if matches!(v, Value::Const(_)) && cx.constant(v).is_none() => {
            return Err(refuse());
        }
        Some(v) => Some(v),
        None if op == AirOp::Load => None,
        None => return Err(cx.invariant("a texture atomic without a value")),
    };
    let expected = match (expected, value) {
        (Some(p), Some(v)) => {
            let ty = cx
                .value_ty(v)
                .ok_or_else(|| cx.invariant("a texture compare-exchange value without a type"))?;
            Some(local::access(cx, Arena::Private, p, None, ty, None)?)
        }
        _ => None,
    };
    let at_ = coord(cx, ops, shape)?;
    let guard = slot::write_guard(cx, shape.dim == ImageDim::Buffer);
    let result = if op == AirOp::Store {
        None
    } else {
        Some(cx.lowered_repr(Value::Inst(at), inst.ty)?)
    };
    let value = match value {
        Some(v) => Some(id(cx, v)?),
        None => None,
    };
    mark(cx, image, |u| u.atomic = true);
    let spin = op == AirOp::Load && spin::is_spin(cx, at);
    Ok(super::TextureStrategy::Atomic(TexAtomic {
        image,
        at: at_,
        op,
        value,
        expected,
        signed,
        guard,
        result,
        spin,
    }))
}

/// The SPIR-V operation of read-modify-write `op` on a texel of `signed`ness.
fn rmw(op: AirOp, signed: bool) -> Option<AtomicOp> {
    Some(match op {
        AirOp::Add => AtomicOp::IAdd,
        AirOp::Sub => AtomicOp::ISub,
        AirOp::And => AtomicOp::And,
        AirOp::Or => AtomicOp::Or,
        AirOp::Xor => AtomicOp::Xor,
        AirOp::Min if signed => AtomicOp::SMin,
        AirOp::Min => AtomicOp::UMin,
        AirOp::Max if signed => AtomicOp::SMax,
        AirOp::Max => AtomicOp::UMax,
        AirOp::Xchg => AtomicOp::Exchange,
        AirOp::Load | AirOp::Store | AirOp::CmpXchgWeak => return None,
    })
}

/// Emit texture atomic `a`: its result value (`None` for a store).
pub(super) fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    a: &TexAtomic,
) -> BResult<Option<SValue>> {
    let ImageRef::Slot(index) = a.image else {
        return Err(cx.fail("a texture atomic on a heap image"));
    };
    let decl = cx
        .plan
        .families
        .texture
        .slots
        .get(&index)
        .and_then(|s| s.decl);
    let global = cx.decl.families.texture.images.get(&index).copied();
    let (Some(decl), Some((g, _))) = (decl, global) else {
        return Err(cx.fail(format!("texture slot {index} is not declared")));
    };
    let base = fb.module().global_value(g)?;
    // Under the identity domains the slot's image is the element of its class's set-1 array at
    // the index its resource ID holds (`emit::identity_id`).
    let ptr = match cx.plan.families.access.identity {
        true => {
            let (element, _) = super::emit::identity_id(cx, fb, index)?;
            fb.access_chain(base, &[element])?
        }
        false => base,
    };
    let coord = super::emit::coordinate(cx, fb, &decl, &a.at)?;
    // A texture buffer's parts: the part that holds the texel, at its index there; the guard
    // tests the texel against the whole width.
    let (part, at) = match decl.parts {
        Some(p) => super::emit::part_pointer(fb, ptr, p, coord)?,
        None => (ptr, coord),
    };
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let sampled = fb.module().ty(SType::Int {
        width: 32,
        signed: a.signed,
    })?;
    // Lane 0 of a held `<4 x i32>`, as the image's sampled type.
    let lane = |fb: &mut FunctionBuilder<'_>, v: SValue| -> BResult<SValue> {
        let x = fb.composite_extract(v, &[0])?;
        if a.signed {
            fb.convert(ConvOp::Bitcast, x, sampled)
        } else {
            Ok(x)
        }
    };
    let x = match a.value {
        Some(v) => {
            let v = cx.operand(fb, v)?;
            Some(lane(fb, v)?)
        }
        None => None,
    };
    let held = match &a.expected {
        Some(e) => match local::emit_access(cx, fb, e, None)? {
            Some(h) => Some(h),
            None => return Err(cx.fail("a texture compare-exchange without its expected texel")),
        },
        None => None,
    };
    let comparator = match held {
        Some(h) => Some(lane(fb, h)?),
        None => None,
    };
    let op = a.op;
    let signed = a.signed;
    // Whether the operation has a result: a store has none.
    let returns = a.result.is_some();
    // A volatile image's atomic under the Vulkan model is `Volatile`. A spin read acquires image
    // memory.
    let order = if a.spin {
        barrier::semantics(
            MemFlags {
                texture: true,
                ..MemFlags::default()
            },
            MemoryOrder::Acquire,
            barrier::memory_model(cx.plan),
        )
    } else {
        Semantics::RELAXED
    };
    let sem = Semantics {
        volatile: super::volatile_texel(cx.plan, &a.image),
        ..order
    };
    let access = move |fb: &mut FunctionBuilder<'_>| -> BResult<Vec<SValue>> {
        let zero = fb.u32_const(0)?;
        let sample = fb.module().const_value(zero)?;
        let p = fb.image_texel_pointer(part, at, sample)?;
        Ok(match (op, x, comparator) {
            (AirOp::Load, _, _) => vec![fb.atomic_load(p, Scope::Device, sem)?],
            // A store is an exchange whose old texel is never read: the same relaxed write.
            (AirOp::Store, Some(x), _) => {
                fb.atomic(AtomicOp::Exchange, p, Scope::Device, sem, x)?;
                Vec::new()
            }
            (AirOp::CmpXchgWeak, Some(x), Some(c)) => {
                let old = fb.atomic_compare_exchange(p, Scope::Device, sem, sem, x, c)?;
                let equal = fb.binary(BinOp::IEqual, old, c)?;
                vec![old, equal]
            }
            (op, Some(x), _) => match rmw(op, signed) {
                Some(o) => {
                    let old = fb.atomic(o, p, Scope::Device, sem, x)?;
                    if returns {
                        vec![old]
                    } else {
                        Vec::new()
                    }
                }
                None => Vec::new(),
            },
            _ => Vec::new(),
        })
    };
    let expected_n = usize::from(returns) + usize::from(comparator.is_some());
    let compare = comparator.is_some();
    let out = if a.guard || compare {
        let img = match decl.parts {
            Some(p) => {
                let last = fb.u32_const(p.count.saturating_sub(1))?;
                let last = fb.module().const_value(last)?;
                let element = fb.access_chain(ptr, &[last])?;
                fb.load(element, MemAccess::default())?
            }
            None => fb.load(ptr, MemAccess::default())?,
        };
        let size = super::emit::texel_width(fb, &decl, img)?;
        let lt = fb.binary(BinOp::ULessThan, coord, size)?;
        let inb = if super::emit::coord_len(&decl) == 1 {
            lt
        } else {
            fb.unary(UnOp::All, lt)?
        };
        let then: ArmFn<'_, ()> = Box::new(move |fb, _| Ok(ArmEnd::FallThrough(access(fb)?)));
        // Outside the image: 0 and no change; a compare-exchange fails with `expected` as
        // passed (its old texel is the comparator, which writes `expected` back unchanged).
        let other: ArmFn<'_, ()> = Box::new(move |fb, _| {
            let values = match (op, comparator) {
                _ if !returns && comparator.is_none() => Vec::new(),
                (AirOp::CmpXchgWeak, Some(c)) => {
                    let b = fb.module().ty(SType::Bool)?;
                    let f = fb.module().constant(Constant::Scalar {
                        ty: b,
                        bits: ConstBits(0),
                    })?;
                    vec![c, fb.module().const_value(f)?]
                }
                _ => {
                    let z = fb.module().constant(Constant::Null { ty: sampled })?;
                    vec![fb.module().const_value(z)?]
                }
            };
            Ok(ArmEnd::FallThrough(values))
        });
        fb.if_(&mut (), inb, then, Some(other))?.0
    } else {
        access(fb)?
    };
    if out.len() != expected_n {
        return Err(cx.fail("a texture atomic without its results"));
    }
    let Some(&old) = out.first() else {
        return Ok(None);
    };
    let old = if signed {
        fb.convert(ConvOp::Bitcast, old, u32_)?
    } else {
        old
    };
    if let (Some(e), Some(h), Some(&ok)) = (&a.expected, held, out.get(1)) {
        // The old texel's first component back to lane 0 of `expected`; the other lanes as
        // passed, as Metal leaves them.
        let back = fb.composite_insert(old, h, &[0])?;
        local::emit_access(cx, fb, e, Some(back))?;
        return Ok(Some(ok));
    }
    // Metal's `(old, 0, 0, 1)`: an `R32` texel's read expansion.
    let Some(result) = &a.result else {
        return Err(cx.fail("a texture atomic result without a lowered type"));
    };
    let Some(ty) = crate::lower::arith::spirv_type(fb.module(), result)? else {
        return Err(cx.fail("a texture atomic result without a lowered type"));
    };
    if !matches!(fb.module().types().get(ty)?, SType::Vector { len: 4, .. }) {
        return Err(cx.fail("a texture atomic result that is no four-lane vector"));
    }
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    let one = fb.u32_const(1)?;
    let one = fb.module().const_value(one)?;
    fb.composite_construct(ty, &[old, zero, zero, one])
        .map(Some)
}
