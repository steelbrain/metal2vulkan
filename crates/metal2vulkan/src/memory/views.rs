//! Descriptor views of a static-root buffer slot.
//!
//! A descriptor-form slot has exactly one view, a word array over the slot's descriptor range:
//!
//! - [`ViewKind::Storage`]: a `StorageBuffer` block `{ u32 words[]; }` (std430, `ArrayStride`
//!   4). It is valid on every profile, reads and writes every leaf through word composition
//!   (`memory::leaf`), and its `OpArrayLength` is the bound range in words, so the padded byte
//!   size is available to the in-shader containment check. The serializer decorates it `Aliased`:
//!   two Metal buffer slots may be bound to the same memory.
//! - [`ViewKind::UniformWords`]: a `Uniform` block `{ u32 words[N]; }` (std430 in `Uniform`, which
//!   needs `uniformBufferStandardLayout`).
//! - [`ViewKind::UniformVec4`]: a `Uniform` block `{ uvec4 vecs[N]; }` (std140, valid everywhere);
//!   a word is one lane of a 4-vector.
//!
//! Uniform views exist only for read-only slots with a `Bounded` window whose descriptor range
//! fits `maxUniformBufferRange` ([`uniform_range`]), and only on profiles whose containment
//! encoding emits no check. Their fixed arrays span the device's aligned uniform range limit,
//! so a shared descriptor's residual stays inside the declared array. A `Uniform` block cannot
//! end in a runtime array, while the range actually bound may be shorter
//! (the binder clamps it to the allocation) and no in-shader check can read it.
//!
//! A storage slot may also have **narrow views** ([`NarrowView`]): `StorageBuffer { u16[] }` and `{
//! u8[] }` blocks at the slot's binding (`ModuleBuilder::resource_view`), declared only where the
//! profile has the width's storage access and arithmetic type (`storageBuffer16BitAccess` and
//! `shaderInt16`, `storageBuffer8BitAccess` and `shaderInt8`) and an access of the slot reads or
//! writes a leaf in one such unit. A unit holds exactly the leaf's bytes, so it is loaded and
//! stored plainly, without the word's shift or the sub-word store's two atomics (`memory::leaf`).
//! The binder contract (descriptor bases congruent to the Metal address modulo 16) carries a leaf's
//! alignment of 1 or 2 to the unit index, and the word view's `OpArrayLength` still contains every
//! access.
//!
//! A storage or uniform slot may also have a **`uvec4` view** ([`VecView`]): `StorageBuffer {
//! uvec4[] }` or `Uniform { uvec4[N] }` at the slot's binding, declared where an access of the slot
//! moves a 16-byte leaf whose AIR alignment is at least 16 at an offset `residual + m` proven a
//! multiple of 16 (`memory::leaf`'s placement: the residual's alignment is the bind offset's, and
//! `m`'s is its structure's; a `UniformVec4` word view is its own). A storage slot may likewise
//! have a **`uvec2` view**, `StorageBuffer { uvec2[] }`, where an access moves an 8-byte leaf whose
//! AIR alignment is at least 8 at an offset proven a multiple of 8 (a uniform slot has none: a
//! std140 array's stride is 16). The binder's 16-byte-aligned descriptor bases carry that alignment
//! to the unit index `(residual + m) / 16` or `/ 8`, and such a leaf lies wholly inside or wholly
//! outside a bound range (every range is a whole number of 16-byte units), so the word view's
//! containment check of the leaf's bytes is the unit's. `u64[]` views are not declared.

use crate::pipeline::sym::{ResourceKind, ResourceSlot, SlotRef};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, ConvOp, LayoutRule, MemAccess, StorageClass, Type,
};

/// The view kind of a descriptor-form slot (the plan's `DescriptorView` with its size).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ViewKind {
    /// `StorageBuffer { u32[] }`.
    Storage,
    /// `Uniform { u32[words] }` (std430; `uniformBufferStandardLayout`).
    UniformWords { words: u32 },
    /// `Uniform { uvec4[vecs] }` (std140).
    UniformVec4 { vecs: u32 },
}

/// A declared view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct View {
    pub global: GlobalId,
    pub kind: ViewKind,
    /// The `u32` type.
    pub u32_ty: TypeId,
}

/// The least alignment of a Metal buffer slot's descriptor base: the binder starts every storage
/// and uniform view's range at a multiple of `max(min*BufferOffsetAlignment, 16)` bytes from the
/// allocation's start ([`base_alignment`]), so with the allocation's first byte at a
/// 16-byte-aligned Metal address, a descriptor byte is congruent to its Metal address modulo 16
/// and a 16-byte leaf whose Metal address is a multiple of 16 is one `uvec4` element.
pub const BASE_ALIGNMENT: u64 = 16;

/// The alignment of a Metal buffer slot's descriptor base for a view whose minimum offset
/// alignment is `min_offset` (`minStorageBufferOffsetAlignment` or
/// `minUniformBufferOffsetAlignment`): the larger of it and [`BASE_ALIGNMENT`].
pub fn base_alignment(min_offset: u64) -> u64 {
    min_offset.max(BASE_ALIGNMENT)
}

/// The byte size of the descriptor range a view of a `Bounded` window of `len` bytes must cover,
/// for a descriptor base alignment of `align` bytes ([`base_alignment`] of the view's minimum
/// offset alignment): the binder binds from `align_down(bind_offset + start, align)` to
/// `round_up(bind_offset + start + len, 16)`, so a shader offset (bind residual plus Metal
/// offset) stays below `align - 1 + len`, and a word read ends at most at the next multiple of 16.
/// `None` on overflow.
///
/// The binder states the bound as `round_up(len + align - 4, 16)`, which assumes `bind_offset +
/// start` is a multiple of 4; a window may start at any byte (a `char` at offset 5), so the bound
/// here uses `align - 1`.
pub fn uniform_range(len: u64, align: u64) -> Option<u64> {
    len.checked_add(align.max(4))?
        .checked_sub(1)?
        .checked_next_multiple_of(16)
}

fn u32_type(mb: &mut ModuleBuilder) -> BResult<TypeId> {
    mb.ty(Type::Int {
        width: 32,
        signed: false,
    })
}

/// The `Buffer` descriptor binding symbol of `slot`, which every view of the slot is declared at.
pub fn binding_symbol(slot: SlotRef) -> ResourceSlot {
    ResourceSlot {
        stage: slot.stage,
        kind: ResourceKind::Buffer,
        index: slot.index,
    }
}

/// Declare the view of `slot` (its `Buffer` descriptor binding symbol).
pub fn declare(mb: &mut ModuleBuilder, slot: SlotRef, kind: ViewKind) -> BResult<View> {
    declare_at(mb, binding_symbol(slot), kind)
}

/// Declare a word view of kind `kind` at binding symbol `sym` (a Metal buffer slot, or a
/// synthetic binding such as the constant blob, `memory::blob`).
pub fn declare_at(mb: &mut ModuleBuilder, sym: ResourceSlot, kind: ViewKind) -> BResult<View> {
    let u32_ty = u32_type(mb)?;
    let (class, rule, member) = match kind {
        ViewKind::Storage => (
            StorageClass::StorageBuffer,
            LayoutRule::Std430,
            mb.ty(Type::RuntimeArray {
                elem: u32_ty,
                stride: Some(4),
            })?,
        ),
        ViewKind::UniformWords { words } => {
            let len = mb.constant(Constant::Scalar {
                ty: u32_ty,
                bits: ConstBits(u64::from(words)),
            })?;
            (
                StorageClass::Uniform,
                LayoutRule::Std430,
                mb.ty(Type::Array {
                    elem: u32_ty,
                    len,
                    stride: Some(4),
                })?,
            )
        }
        ViewKind::UniformVec4 { vecs } => {
            let v4 = mb.ty(Type::Vector {
                elem: u32_ty,
                len: 4,
            })?;
            let len = mb.constant(Constant::Scalar {
                ty: u32_ty,
                bits: ConstBits(u64::from(vecs)),
            })?;
            (
                StorageClass::Uniform,
                LayoutRule::Std140,
                mb.ty(Type::Array {
                    elem: v4,
                    len,
                    stride: Some(16),
                })?,
            )
        }
    };
    let layout = natural_layout(mb.types(), rule, true, &[member])?;
    let block = mb.ty(Type::Struct {
        members: [member].into(),
        layout: Some(layout),
    })?;
    let global = mb.resource(sym, class, block)?;
    Ok(View {
        global,
        kind,
        u32_ty,
    })
}

fn k(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

/// A pointer to word `word` (a `u32` index) of a storage view, or `None` for a uniform view.
pub fn word_ptr(fb: &mut FunctionBuilder<'_>, view: View, word: Value) -> BResult<Option<Value>> {
    match view.kind {
        ViewKind::Storage => {
            let base = fb.module().global_value(view.global)?;
            let zero = k(fb, 0)?;
            Ok(Some(fb.access_chain(base, &[zero, word])?))
        }
        ViewKind::UniformWords { .. } | ViewKind::UniformVec4 { .. } => Ok(None),
    }
}

/// Load word `word` (a `u32` index) of `view`.
pub fn read_word(fb: &mut FunctionBuilder<'_>, view: View, word: Value) -> BResult<Value> {
    let base = fb.module().global_value(view.global)?;
    let zero = k(fb, 0)?;
    match view.kind {
        ViewKind::Storage | ViewKind::UniformWords { .. } => {
            let p = fb.access_chain(base, &[zero, word])?;
            fb.load(p, MemAccess::default())
        }
        ViewKind::UniformVec4 { .. } => {
            let two = k(fb, 2)?;
            let three = k(fb, 3)?;
            let vi = fb.binary(BinOp::ShiftRightLogical, word, two)?;
            let lane = fb.binary(BinOp::BitwiseAnd, word, three)?;
            let p = fb.access_chain(base, &[zero, vi])?;
            let v = fb.load(p, MemAccess::default())?;
            fb.vector_extract_dynamic(v, lane)
        }
    }
}

/// A typed narrow view of a storage slot: `StorageBuffer { u<8·bytes>[] }` at the slot's binding,
/// beside its word view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NarrowView {
    pub global: GlobalId,
    /// The unit's bytes: 1 or 2.
    pub bytes: u32,
    /// The unit's type (`u8` or `u16`).
    pub unit_ty: TypeId,
    /// The `u32` type.
    pub u32_ty: TypeId,
}

/// A `uvec4` or `uvec2` view of a descriptor slot: `StorageBuffer { uvec4[] }`, `StorageBuffer {
/// uvec2[] }` or `Uniform { uvec4[N] }` at the slot's binding beside its word view, or a
/// `UniformVec4` word view itself. A unit is vector element `index` of member 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VecView {
    pub global: GlobalId,
    /// The view's storage class: a uniform view's vectors are read-only, read plainly.
    pub class: StorageClass,
}

/// The unit views of a descriptor slot beside its word view: a storage slot's narrow and `uvec2`
/// views and a storage or uniform slot's `uvec4` view.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnitViews {
    pub u16: Option<NarrowView>,
    pub u8: Option<NarrowView>,
    pub vec2: Option<VecView>,
    pub vec4: Option<VecView>,
}

impl UnitViews {
    /// No unit view.
    pub const NONE: UnitViews = UnitViews {
        u16: None,
        u8: None,
        vec2: None,
        vec4: None,
    };

    /// The units these views load and store.
    pub fn units(&self) -> crate::memory::leaf::Units {
        crate::memory::leaf::Units {
            u16: self.u16.is_some(),
            u8: self.u8.is_some(),
            vec2: self.vec2.is_some(),
            vec4: self.vec4.is_some(),
        }
    }

    /// The vector view of `lanes`-word units (2 or 4).
    pub fn vec(&self, lanes: u32) -> Option<VecView> {
        match lanes {
            2 => self.vec2,
            4 => self.vec4,
            _ => None,
        }
    }

    /// The view of `bytes`-byte units (1 or 2).
    pub fn of(&self, bytes: u32) -> Option<NarrowView> {
        match bytes {
            1 => self.u8,
            2 => self.u16,
            _ => None,
        }
    }

    /// Declare the unit views of `units` for `slot`, whose word view [`declare`] declared as
    /// `view`: a storage view's narrow, `uvec2` and `uvec4` views, a `UniformWords` view's `uvec4`
    /// view; a `UniformVec4` view is its own `uvec4` view, declared or not. A uniform slot has no
    /// narrow or `uvec2` view.
    pub fn declare(
        mb: &mut ModuleBuilder,
        slot: SlotRef,
        view: View,
        units: crate::memory::leaf::Units,
    ) -> BResult<UnitViews> {
        let storage = view.kind == ViewKind::Storage;
        Ok(UnitViews {
            u16: if storage && units.u16 {
                Some(declare_narrow(mb, slot, 2)?)
            } else {
                None
            },
            u8: if storage && units.u8 {
                Some(declare_narrow(mb, slot, 1)?)
            } else {
                None
            },
            vec2: if storage && units.vec2 {
                Some(declare_vec(mb, slot, view, 2)?)
            } else {
                None
            },
            vec4: match view.kind {
                ViewKind::UniformVec4 { .. } => Some(VecView {
                    global: view.global,
                    class: StorageClass::Uniform,
                }),
                _ if units.vec4 => Some(declare_vec(mb, slot, view, 4)?),
                _ => None,
            },
        })
    }
}

/// Declare the `uvecN` view (`N = lanes`, 2 or 4) of `slot`, whose word view [`declare`] declared
/// as `view`: `StorageBuffer { uvecN[] }` (std430, `ArrayStride` `4N`) beside a storage view, or
/// `Uniform { uvec4[words / 4] }` (std140, valid without `uniformBufferStandardLayout`) beside a
/// `UniformWords` view, whose size is a multiple of 16 bytes ([`uniform_range`]). The serializer
/// decorates a storage one `Aliased` as every storage view of a Metal buffer. A uniform view has no
/// `uvec2` view: a std140 array's stride is 16.
pub fn declare_vec(
    mb: &mut ModuleBuilder,
    slot: SlotRef,
    view: View,
    lanes: u32,
) -> BResult<VecView> {
    let sym = binding_symbol(slot);
    let u32_ty = u32_type(mb)?;
    let len = u8::try_from(lanes)
        .ok()
        .filter(|&l| matches!(l, 2 | 4))
        .ok_or(crate::spirv::error::BuildError::Global(
            crate::spirv::error::GlobalRule::ResourceType,
        ))?;
    let vec = mb.ty(Type::Vector { elem: u32_ty, len })?;
    let (class, rule, member) = match view.kind {
        ViewKind::Storage => (
            StorageClass::StorageBuffer,
            LayoutRule::Std430,
            mb.ty(Type::RuntimeArray {
                elem: vec,
                stride: Some(4 * lanes),
            })?,
        ),
        ViewKind::UniformWords { words } if lanes == 4 => {
            let len = mb.constant(Constant::Scalar {
                ty: u32_ty,
                bits: ConstBits(u64::from(words / 4)),
            })?;
            (
                StorageClass::Uniform,
                LayoutRule::Std140,
                mb.ty(Type::Array {
                    elem: vec,
                    len,
                    stride: Some(16),
                })?,
            )
        }
        // The word view is already one.
        ViewKind::UniformVec4 { .. } if lanes == 4 => {
            return Ok(VecView {
                global: view.global,
                class: StorageClass::Uniform,
            })
        }
        ViewKind::UniformWords { .. } | ViewKind::UniformVec4 { .. } => {
            return Err(crate::spirv::error::BuildError::Global(
                crate::spirv::error::GlobalRule::ResourceType,
            ))
        }
    };
    let layout = natural_layout(mb.types(), rule, true, &[member])?;
    let block = mb.ty(Type::Struct {
        members: [member].into(),
        layout: Some(layout),
    })?;
    let global = mb.resource_view(sym, class, block)?;
    Ok(VecView { global, class })
}

/// Load vector unit `index` of `view` with memory operands `ops`.
pub fn read_vec(
    fb: &mut FunctionBuilder<'_>,
    view: VecView,
    index: Value,
    ops: MemAccess,
) -> BResult<Value> {
    let base = fb.module().global_value(view.global)?;
    let zero = k(fb, 0)?;
    let p = fb.access_chain(base, &[zero, index])?;
    fb.load(p, ops)
}

/// Store the vector `value` to unit `index` of storage view `view` with memory operands `ops`.
pub fn write_vec(
    fb: &mut FunctionBuilder<'_>,
    view: VecView,
    index: Value,
    value: Value,
    ops: MemAccess,
) -> BResult<()> {
    let base = fb.module().global_value(view.global)?;
    let zero = k(fb, 0)?;
    let p = fb.access_chain(base, &[zero, index])?;
    fb.store(p, value, ops)
}

/// Declare the `bytes`-byte (1 or 2) narrow view of `slot`, whose word view [`declare`] declared.
pub fn declare_narrow(mb: &mut ModuleBuilder, slot: SlotRef, bytes: u32) -> BResult<NarrowView> {
    let sym = binding_symbol(slot);
    let u32_ty = u32_type(mb)?;
    let width = match bytes {
        1 => 8,
        2 => 16,
        _ => {
            return Err(crate::spirv::error::BuildError::Global(
                crate::spirv::error::GlobalRule::ResourceType,
            ))
        }
    };
    let unit_ty = mb.ty(Type::Int {
        width,
        signed: false,
    })?;
    let member = mb.ty(Type::RuntimeArray {
        elem: unit_ty,
        stride: Some(bytes),
    })?;
    let layout = natural_layout(mb.types(), LayoutRule::Std430, true, &[member])?;
    let block = mb.ty(Type::Struct {
        members: [member].into(),
        layout: Some(layout),
    })?;
    let global = mb.resource_view(sym, StorageClass::StorageBuffer, block)?;
    Ok(NarrowView {
        global,
        bytes,
        unit_ty,
        u32_ty,
    })
}

/// A pointer to unit `index` (a `u32` index of `view.bytes`-byte units) of narrow view `view`.
fn unit_ptr(fb: &mut FunctionBuilder<'_>, view: NarrowView, index: Value) -> BResult<Value> {
    let base = fb.module().global_value(view.global)?;
    let zero = k(fb, 0)?;
    fb.access_chain(base, &[zero, index])
}

/// Load unit `index` of `view` with memory operands `ops`, zero-extended to a `u32`; a 16-bit
/// unit's extension is noted (`FunctionBuilder::note_widened16`), so a `half` read from it may
/// take the unit itself (`narrow::from_word_halves`).
pub fn read_unit(
    fb: &mut FunctionBuilder<'_>,
    view: NarrowView,
    index: Value,
    ops: MemAccess,
) -> BResult<Value> {
    let p = unit_ptr(fb, view, index)?;
    let x = fb.load(p, ops)?;
    let y = fb.convert(ConvOp::UConvert, x, view.u32_ty)?;
    if view.bytes == 2 {
        fb.note_widened16(y, x);
    }
    Ok(y)
}

/// Store the low `8 · view.bytes` bits of `value` (a `u32`) to unit `index` of `view` with
/// memory operands `ops`.
pub fn write_unit(
    fb: &mut FunctionBuilder<'_>,
    view: NarrowView,
    index: Value,
    value: Value,
    ops: MemAccess,
) -> BResult<()> {
    let x = fb.convert(ConvOp::UConvert, value, view.unit_ty)?;
    let p = unit_ptr(fb, view, index)?;
    fb.store(p, x, ops)
}

/// The byte size of a storage view's bound range, as a `u32` (`OpArrayLength · 4`): the padded
/// size. A range of `2^30` words or more wraps to a smaller size, which only makes a containment
/// check stricter. `None` for a uniform view, which has no in-shader check.
pub fn storage_bytes(fb: &mut FunctionBuilder<'_>, view: View) -> BResult<Option<Value>> {
    match view.kind {
        ViewKind::Storage => {
            let base = fb.module().global_value(view.global)?;
            let len = fb.array_length(base, 0)?;
            let four = k(fb, 4)?;
            Ok(Some(fb.binary(BinOp::IMul, len, four)?))
        }
        ViewKind::UniformWords { .. } | ViewKind::UniformVec4 { .. } => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_uniform_range_covers_any_start_residue() {
        // A 1-byte window at an odd start with 16-byte alignment: the residue can be 15, and the
        // word read ends at 16.
        assert_eq!(uniform_range(1, 16), Some(16));
        // The `- 4` form would give 16 here, but a window of 4 bytes starting at residue 15 reads
        // up to byte 19 (word 4): the range must be 32.
        assert_eq!(uniform_range(4, 16), Some(32));
        assert_eq!(uniform_range(6, 256), Some(272));
        assert_eq!(uniform_range(u64::MAX, 16), None);
    }
}
