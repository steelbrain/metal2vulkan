//! A kernel's explicit imageblock.
//!
//! Metal gives each threadgroup of a kernel dispatch its own imageblock: `width · height` cells of
//! the explicit layout's `air.imageblock_data_size` bytes, the extent the compute encoder sets
//! (`setImageblockWidth:height:`), its contents undefined when the threadgroup starts. The kernel
//! reaches it only through `air.imageblock_data(coord, 0, 0)`, the cell at `coord` (a `ushort2`),
//! the extent queries `air.get_imageblock_{width,height}`, and slice writes to a texture
//! (`lower::texture::slice`). The explicit layout is opaque: nothing may observe where one cell
//! is relative to another (an imageblock pointer compared or converted to an integer is refused).
//!
//! **Workgroup staging.** The imageblock is the threadgroup arena's last region
//! (`memory::arena`): cell `(x, y)` is at `region + (y · width + x) · cell_bytes`, and the
//! region's length `width · height · cell_bytes` is specialized (`SpecKey::ImageblockWidth`,
//! `SpecKey::ImageblockHeight`, [`RuntimeFact::ImageblockSize`]), so the binder charges it,
//! rounded to 16 bytes, to `maxComputeSharedMemorySize` with the threadgroup memory it shares. A
//! cell pointer is an arena offset (`Repr::TgPtr`, `memory::provenance`): GEPs, loads, stores,
//! atomics, memory intrinsics, and containment are the threadgroup arena's (`access::local`), and
//! a `mem_threadgroup_imageblock` barrier is a workgroup-memory barrier (`sync::barrier`). The
//! extent queries read the specialization constants, which the binder fixes below 2¹⁶ (Metal's
//! `get_width` is a `ushort`).
//!
//! A tile function's implicit imageblock, and an explicit one aliased onto it
//! (`air.alias_implicit_imageblock`), are the same region holding the tile's attachment texels
//! (`access::tile`); a kernel compiled without tile inputs that reaches one is
//! `NeedsInput(TileAttachments)`.
//!
//! Refused: an imageblock outside a kernel, a kernel with two explicit imageblocks
//! (`imageblock_data` names none), one without a data size, and `imageblock_data` operands other
//! than `(coord, 0, 0)` are `Unsupported{Imageblocks, NotImplemented}`.

use crate::air::abi::{ImageblockKind, ParamRole};
use crate::air::intrinsics::{AirIntrinsic, Imageblock};
use crate::air::ir::{Callee, Const, Inst, InstKind, Value, ValueId};
use crate::error::{Error, Feature, LowerFamily, Unsupported};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::memory::arena::Arena;
use crate::memory::leaf::{k32, Leaf};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{SpecKey, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{ConstId, Value as SValue};
use crate::spirv::ir::{BinOp, ConstBits, SpecOp, Type as SType};

use super::local::{self, LocalAccess, LocalPtr};

/// An imageblock intrinsic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageblockStrategy {
    /// `air.imageblock_data(coord, 0, 0)`: the arena offset of cell `coord`.
    Cell { coord: ValueId, cell_bytes: u32 },
    /// `air.get_imageblock_width` (`height: false`) or `_height`: the specialized extent.
    Extent { height: bool },
    /// A tile function's `air.load.implicit_imageblock` (`value: None`) or
    /// `air.store.implicit_imageblock` of `value` (`access::tile`): the attachment texel `leaf` at
    /// `offset` in cell `coord`, contained when `contained`; a normalized attachment's bits,
    /// converted by `norm`.
    Texel {
        coord: ValueId,
        cell_bytes: u32,
        offset: u32,
        leaf: Leaf,
        value: Option<ValueId>,
        contained: bool,
        norm: Option<super::tile::NormAccess>,
    },
}

impl ImageblockStrategy {
    pub fn name(&self) -> &'static str {
        match self {
            ImageblockStrategy::Cell { .. } => "access.imageblock.cell",
            ImageblockStrategy::Extent { .. } => "access.imageblock.extent",
            ImageblockStrategy::Texel { value: None, .. } => "access.imageblock.texel.load",
            ImageblockStrategy::Texel { value: Some(_), .. } => "access.imageblock.texel.store",
        }
    }

    pub fn words(&self) -> u32 {
        match self {
            // Two extracts, a multiply-add, the scale, and the region's offset.
            ImageblockStrategy::Cell { .. } => 40,
            ImageblockStrategy::Extent { .. } => 4,
            // The cell's offset, the leaf, and its containment; and a normalized texel's
            // conversion.
            ImageblockStrategy::Texel {
                leaf, value, norm, ..
            } => 120u32
                .saturating_add(leaf.cost())
                .saturating_add(norm.map_or(0, |n| n.words(leaf.form.lanes, value.is_some()))),
        }
    }
}

fn imageblocks() -> Error {
    Unsupported::not_implemented(Feature::Imageblocks).into()
}

/// What a use of the imageblock reaches: a tile function stages the attachments whose cells it
/// reads, and stores those it may write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CellUse {
    /// The extent only.
    Extent,
    /// Every cell, read only (a slice write).
    Read,
    /// Every cell, through a cell pointer that may be written.
    ReadWrite,
}

/// Use the kernel's explicit imageblock, or a tile function's cells (`access::tile`), as `cells`:
/// the arena's last region, whose cell bytes this returns. The first use places it in the
/// threadgroup arena ([`place_region`]).
pub(crate) fn use_imageblock(cx: &mut SelectCx<'_>, cells: CellUse) -> Result<u32, Error> {
    if super::tile::is_tile(cx) {
        let cell_bytes = match cells {
            CellUse::Extent => super::tile::cell_bytes(cx)?,
            CellUse::Read => super::tile::use_cells(cx, false)?,
            CellUse::ReadWrite => super::tile::use_cells(cx, true)?,
        };
        place_region(cx, cell_bytes)?;
        return Ok(cell_bytes);
    }
    let params = &cx.entry.ep.params;
    let implicit = params.iter().any(|b| match &b.role {
        ParamRole::Imageblock(arg) => arg.kind == ImageblockKind::Implicit || arg.alias_implicit,
        _ => false,
    });
    if implicit && cx.stage == Stage::Kernel {
        return Err(super::tile::needs_tile());
    }
    if cx.stage != Stage::Kernel {
        return Err(imageblocks());
    }
    let cell_bytes = cx
        .memory
        .tg_arena
        .imageblock
        .map(|r| r.cell_bytes)
        .ok_or_else(imageblocks)?;
    place_region(cx, cell_bytes)?;
    Ok(cell_bytes)
}

/// Place the imageblock region of `cell_bytes`-byte cells in the threadgroup arena on its first
/// use: check the arena's static part again, and record [`RuntimeFact::ImageblockSize`].
pub(crate) fn place_region(cx: &mut SelectCx<'_>, cell_bytes: u32) -> Result<(), Error> {
    local::use_arena(cx, Arena::Threadgroup)?;
    local::composed(cx, Arena::Threadgroup);
    let limit = crate::lower::mesh::workgroup_limit(cx);
    let Some(shape) = cx.families.access.local.tg.as_mut() else {
        return Err(cx.invariant("a used threadgroup arena without a shape"));
    };
    if shape.imageblock.is_none() {
        shape.imageblock = Some(cell_bytes);
        shape.check(limit)?;
        cx.facts.insert(RuntimeFact::ImageblockSize { cell_bytes });
    }
    Ok(())
}

/// Whether `v` is the integer constant 0.
fn is_zero(cx: &SelectCx<'_>, v: Value) -> bool {
    let Value::Const(c) = v else { return false };
    matches!(
        cx.entry.module.consts().get(c).map(|k| &k.value),
        Some(Const::Int(0) | Const::Null)
    )
}

/// The value id of operand `v`; a constant must be one the arith family materializes.
pub(crate) fn operand(cx: &SelectCx<'_>, v: Value) -> Result<ValueId, Error> {
    if matches!(v, Value::Const(_)) && cx.constant(v).is_none() {
        return Err(Unsupported::not_implemented(Feature::Lowering(LowerFamily::Arith)).into());
    }
    cx.values
        .id(v)
        .ok_or_else(|| cx.invariant("an imageblock operand out of the value range"))
}

/// Select the strategy of an imageblock intrinsic call, or `None` for any other instruction.
pub fn select(cx: &mut SelectCx<'_>, inst: &Inst) -> Result<Option<ImageblockStrategy>, Error> {
    let InstKind::Call(call) = &inst.kind else {
        return Ok(None);
    };
    let Callee::Air {
        op: AirIntrinsic::Imageblock(op),
        ..
    } = &call.callee
    else {
        return Ok(None);
    };
    Ok(Some(match op {
        Imageblock::Data => {
            let [coord, index, tail] = call.args[..] else {
                return Err(cx.invariant("an imageblock_data call of another arity"));
            };
            // Every `imageblock::data(ushort2)` passes `(coord, 0, 0)`; other values have no
            // Metal meaning this lowering knows.
            if !is_zero(cx, index) || !is_zero(cx, tail) {
                return Err(imageblocks());
            }
            let cell_bytes = use_imageblock(cx, CellUse::ReadWrite)?;
            ImageblockStrategy::Cell {
                coord: operand(cx, coord)?,
                cell_bytes,
            }
        }
        Imageblock::Width | Imageblock::Height => {
            use_imageblock(cx, CellUse::Extent)?;
            ImageblockStrategy::Extent {
                height: *op == Imageblock::Height,
            }
        }
        // The implicit imageblock's accesses read and write a tile function's cells.
        Imageblock::Load(_) | Imageblock::Store(_) => {
            super::tile::select_texel(cx, call, inst.ty, op)?
        }
    }))
}

/// The imageblock's extent in `u32` specialization constants (default 0 by 0: the binder
/// always specializes them).
pub fn extent_constants(mb: &mut ModuleBuilder) -> BResult<(ConstId, ConstId)> {
    let u32_ = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let w = mb.spec(SpecKey::ImageblockWidth, u32_, ConstBits(0))?;
    let h = mb.spec(SpecKey::ImageblockHeight, u32_, ConstBits(0))?;
    Ok((w, h))
}

/// The imageblock region's byte length, `width · height · cell_bytes`, an `OpSpecConstantOp`
/// (`memory::arena::declare_threadgroup` places it after the regions). The binder keeps it within
/// `maxComputeSharedMemorySize`, so it does not wrap.
pub fn region_length(mb: &mut ModuleBuilder, cell_bytes: u32) -> BResult<ConstId> {
    let u32_ = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let (w, h) = extent_constants(mb)?;
    let cell = mb.constant(crate::spirv::ir::Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(cell_bytes)),
    })?;
    let cells = mb.spec_op(SpecOp::IMul, u32_, &[w, h])?;
    mb.spec_op(SpecOp::IMul, u32_, &[cells, cell])
}

/// The imageblock's width and height as `u32` values.
pub(crate) fn extent(fb: &mut FunctionBuilder<'_>) -> BResult<(SValue, SValue)> {
    let (w, h) = extent_constants(fb.module())?;
    Ok((fb.module().const_value(w)?, fb.module().const_value(h)?))
}

/// The byte offset of the imageblock region in the threadgroup arena.
pub(crate) fn region(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
    let at = cx
        .decl
        .families
        .access
        .local
        .tg
        .as_ref()
        .and_then(|t| t.imageblock);
    match at {
        Some(c) => fb.module().const_value(c),
        None => Err(cx.fail("the imageblock region is not declared")),
    }
}

/// The arena offset of cell `(x, y)` (`u32` values): `region + (y · width + x) · cell_bytes`.
pub(crate) fn cell_offset(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    x: SValue,
    y: SValue,
    cell_bytes: u32,
) -> BResult<SValue> {
    let base = region(cx, fb)?;
    let (w, _) = extent(fb)?;
    let row = fb.binary(BinOp::IMul, y, w)?;
    let index = fb.binary(BinOp::IAdd, row, x)?;
    let size = k32(fb, cell_bytes)?;
    let rel = fb.binary(BinOp::IMul, index, size)?;
    fb.binary(BinOp::IAdd, base, rel)
}

/// Whether an imageblock access is contained: as every threadgroup arena access.
pub(crate) fn contained(cx: &SelectCx<'_>) -> bool {
    local::contained(cx, Arena::Threadgroup)
}

/// Load `leaf` from the threadgroup arena at byte offset `off`, contained when `contained`.
pub(crate) fn load(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    leaf: Leaf,
    contained: bool,
    off: SValue,
) -> BResult<SValue> {
    let a = LocalAccess {
        arena: Arena::Threadgroup,
        ptr: LocalPtr::Offset(0),
        value: None,
        leaf,
        contained,
        volatile: false,
    };
    match local::emit_access_at(cx, fb, &a, off, None)? {
        Some(v) => Ok(v),
        None => Err(cx.fail("an imageblock load without a value")),
    }
}

/// Emit imageblock strategy `s`, defining value `vid`.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    vid: ValueId,
    s: &ImageblockStrategy,
) -> BResult<()> {
    let r = match s {
        ImageblockStrategy::Cell { coord, cell_bytes } => {
            // The coordinate's lanes are held zero-extended (`ushort2`).
            let c = cx.operand(fb, *coord)?;
            let x = fb.composite_extract(c, &[0])?;
            let y = fb.composite_extract(c, &[1])?;
            cell_offset(cx, fb, x, y, *cell_bytes)?
        }
        ImageblockStrategy::Extent { height } => {
            let (w, h) = extent(fb)?;
            if *height {
                h
            } else {
                w
            }
        }
        ImageblockStrategy::Texel { .. } => match super::tile::emit_texel(cx, fb, s)? {
            Some(v) => v,
            None => return Ok(()),
        },
    };
    cx.define(vid, r);
    Ok(())
}
