//! Imageblock slice writes (`air.write_imageblock_slice_to_texture_2d[_array]`).
//!
//! `texture.write(img.slice(cell->field[, size]), coord[, layer])` copies one field of a block of
//! the kernel's explicit imageblock (`access::imageblock`) into the texture, one texel per cell.
//! The operands are the texture, the field pointer, an explicit-size flag, the block's origin, its
//! size, the texture coordinate, the layer (arrays), the level, a trailing flag, and the access.
//! Metal's semantics, as the authored cases measured them:
//!
//! - the block starts at the imageblock's own origin; the pointer selects the **field** (its byte
//!   offset within the cell) and its cell is ignored;
//! - the block is the size operand when the flag is set, else the whole imageblock
//!   (`width · height`): an implicit slice is the imageblock's extent, not the threadgroup's;
//! - cell `(x, y)` of the block lands at `coord + (x, y)` on the level, and a texel outside the
//!   texture is discarded, like any write (the guard `write_guard` decides); a level other than
//!   the constant 0 is written through the slot's level views, and a level the texture lacks
//!   drops the slice;
//! - the element `E` the intrinsic names (`f32`, `f16`, `i32`, `i16`, 1, 2, or 4 lanes) is the
//!   texel's leading lanes and every later lane is 0, alpha included, in every pixel format: a
//!   4-lane element is the whole texel, a 1-lane one `(v, 0, 0, 0)`, a 2-lane one `(x, y, 0, 0)`,
//!   which the format then keeps as a texture write keeps any texel (an `R` format the first
//!   lane, `A8Unorm` the alpha of 0). Measured on an M4 and the M3 Max (a probe `slice`) for
//!   `half`, `float`, `short` and `int` elements of each width into formats of one, two and four
//!   components, through a texture slot and through an argument buffer, with Metal's validation
//!   layer on and off alike.
//!
//! The copy is one counted loop over the block's cells in the writing invocation (a builder
//! loop), each iteration a contained arena leaf load
//! (`access::imageblock::load`) and a texture write. A block
//! past the imageblock's extent in either axis is clipped to it (the cells past it are not the
//! imageblock's; Metal leaves them undefined).
//!
//! Only these are pinned (else `Unsupported{Imageblocks, NotImplemented}`): the origin operand the
//! constant zero, the trailing flag the constant `false`, the flag a constant, a field pointer
//! whose roots are all imageblock cells at one exact in-cell offset (in a tile function whose pass
//! keeps fewer bytes of its cells, `LoweringInputs::kept_imageblock_bytes`, a field within them),
//! and a static texture slot or a heap element (a texture loaded from an argument buffer), not a
//! switch. An integer element spells no signedness (neither the AIR symbol nor the texture
//! argument's ABI metadata carries one): its texel class is the slot's from another operation, else
//! the numeric type of the format bound there, the slot's `StorageTextureFormat` input on every
//! profile (a formatless choice keeps the type, `RuntimeFact::StorageTextureNumericType`).
//!
//! A heap element is written as a heap `write` writes it: through its group's formatless storage
//! class, at level 0, the write dropped through the null element, a narrower element as a slot's is
//! (the texel does not depend on the member's format, so no binder check is needed). An integer
//! element's class is the group's from another operation (no slot has a format input to give it).

use crate::air::intrinsics::{Elem, TexRole, Texture};
use crate::air::ir::{InstId, Value, ValueId};
use crate::budget::Meter;
use crate::error::{Error, Feature, RuntimeInput, Unsupported};
use crate::lower::access::imageblock;
use crate::lower::arith::legal::held_on;
use crate::lower::arith::{FloatFmt, Form, Held};
use crate::lower::select::SelectCx;
use crate::memory::leaf::{Leaf, Path};
use crate::memory::provenance::imageblock_cell;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::lower_inputs::StorageImageClass;
use crate::pipeline::sym::{SampledType, SlotClass, SlotRef};

use super::{
    coord, image_access, is_zero, mark, observe, slot, storage_level, Coord, ImageRef, Operands,
    TexelClass, TexelConv, TexelForm, TextureStrategy,
};

/// A slice write of the kernel's imageblock to a texture slot or heap element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slice {
    pub image: ImageRef,
    /// The texture coordinate of the block's first cell.
    pub at: Coord,
    /// The level of the slot's level views when it is not the constant 0.
    pub level: Option<ValueId>,
    /// The block's size (a held `ushort2`) when the slice names one; else the imageblock's.
    pub size: Option<ValueId>,
    /// The field's byte offset within a cell.
    pub field: u32,
    pub cell_bytes: u32,
    /// The element in the cell.
    pub leaf: Leaf,
    /// The element's texel conversion; its lanes are the element's.
    pub texel: TexelForm,
    /// Discard a texel outside the texture in the shader (no robustness discards it).
    pub guard: bool,
    /// The cell loads are contained.
    pub contained: bool,
    /// The slot's storage class (`super::storage_class`): a converting one moves or encodes each
    /// texel's channels (`RotateArgb`'s rotation, `SrgbEncode`'s codes).
    pub class: StorageImageClass,
    /// Every invocation of the workgroup writes a strided share of the cells, then a workgroup
    /// barrier (`super::cooperative`); else the writing invocation writes them all.
    pub cooperative: bool,
}

impl Slice {
    pub(super) fn words(&self) -> u32 {
        self.leaf.cost()
            + if self.contained { 64 } else { 0 }
            + if self.guard { 48 } else { 0 }
            + super::emit::class_write_words(self.class)
            + if self.cooperative { 24 } else { 0 }
            + 320
    }
}

fn imageblocks() -> Error {
    Unsupported::not_implemented(Feature::Imageblocks).into()
}

/// Whether `v` is the constant `i1` `b`.
fn is_flag(cx: &SelectCx<'_>, v: Value, b: bool) -> bool {
    super::const_int(cx, v) == Some(u64::from(b))
}

/// The texel class of an integer element written to `slot`: the slot's from another operation,
/// else the numeric type of the format bound there, which the caller states in the slot's
/// `StorageImageChoice` (an input on every profile: `NeedsInput(StorageTextureFormat)` without
/// one). The choice's declared format gives the type, or, for a formatless choice, the type it
/// keeps (`StorageImageChoice::numeric`); the function then depends on it
/// (`RuntimeFact::StorageTextureNumericType`), so the binder keys the slot's choice even where a
/// formatless declaration serves its format. A bound float format leaves the element's texels
/// undefined: refused.
fn integer_class(cx: &mut SelectCx<'_>, slot: SlotRef) -> Result<TexelClass, Error> {
    let seen = cx
        .families
        .texture
        .slots
        .get(&slot.index)
        .and_then(|s| s.texel);
    let class = match seen {
        Some(c) => Some(c),
        None => {
            let Some(choice) = cx.inputs.storage_images.get(&slot) else {
                return Err(Error::NeedsInput(RuntimeInput::StorageTextureFormat {
                    slot,
                }));
            };
            let numeric = if choice.vk_format == crate::vk::Format::UNDEFINED {
                choice.numeric
            } else {
                crate::bind::neutral::numeric_type(choice.vk_format)
            };
            cx.facts
                .insert(RuntimeFact::StorageTextureNumericType { slot });
            match numeric {
                Some(SampledType::Sint) => Some(TexelClass::Sint),
                Some(SampledType::Uint) => Some(TexelClass::Uint),
                Some(SampledType::Uint64) => Some(TexelClass::Uint64),
                Some(SampledType::Float) | None => None,
            }
        }
    };
    match class {
        Some(c @ (TexelClass::Sint | TexelClass::Uint)) => Ok(c),
        _ => Err(imageblocks()),
    }
}

/// The texel class of an integer element written to heap group `group`: the group's from another
/// operation (its class is then that numeric type's). A group no other operation types has no
/// format input to give it one: refused.
fn heap_integer_class(cx: &SelectCx<'_>, group: u32) -> Result<TexelClass, Error> {
    match cx.families.texture.heap.get(&group).and_then(|h| h.texel) {
        Some(c @ (TexelClass::Sint | TexelClass::Uint)) => Ok(c),
        _ => Err(imageblocks()),
    }
}

/// Select slice write `call` to `image`, a texture slot or a heap element.
pub(super) fn select(
    cx: &mut SelectCx<'_>,
    (call, t): (InstId, &Texture),
    ops: &Operands<'_>,
    image: ImageRef,
    m: &mut Meter,
) -> Result<TextureStrategy, Error> {
    let elem = t
        .texel
        .ok_or_else(|| cx.invariant("a slice write without an element type"))?
        .ty;
    let operand = |role| {
        ops.get(role)
            .ok_or_else(|| cx.invariant("a slice write without an operand"))
    };
    let (flag, origin, size, tail, ptr) = (
        operand(TexRole::Opaque(2))?,
        operand(TexRole::Opaque(3))?,
        operand(TexRole::Opaque(4))?,
        operand(TexRole::Opaque(8))?,
        operand(TexRole::Imageblock)?,
    );
    if !is_zero(cx, origin) || !is_flag(cx, tail, false) {
        return Err(imageblocks());
    }
    let size = if is_flag(cx, flag, true) {
        Some(imageblock::operand(cx, size)?)
    } else if is_flag(cx, flag, false) {
        None
    } else {
        return Err(imageblocks());
    };
    let level = storage_level(cx, ops, image)?;
    let cell_bytes = imageblock::use_imageblock(cx, imageblock::CellUse::Read)?;
    let roots_ok = cx
        .memory
        .roots_of(ptr)
        .is_some_and(|r| !r.is_empty() && r.iter().all(|r| imageblock_cell(cx.entry, *r)));
    let lanes = u8::try_from(elem.lanes).map_err(|_| imageblocks())?;
    let (held, conv, lane_bytes) = match elem.elem {
        Elem::F32 => (Held::Float(FloatFmt::F32), Some(TexelConv::F32), 4),
        Elem::F16 => (Held::Float(FloatFmt::Half), Some(TexelConv::F16), 2),
        Elem::I32 => (
            Held::Int {
                bits: 32,
                container: 32,
            },
            None,
            4,
        ),
        Elem::I16 => (
            Held::Int {
                bits: 16,
                container: 32,
            },
            None,
            2,
        ),
        _ => return Err(imageblocks()),
    };
    let bytes = lane_bytes * u32::from(lanes);
    // A tile function's slice of words its pass does not keep is unmeasured
    // (`access::tile::KeptCells`).
    let limit = cx
        .inputs
        .kept_imageblock_bytes
        .map_or(cell_bytes, |kept| kept.min(cell_bytes));
    let field = cx
        .memory
        .offset(ptr)
        .and_then(|o| o.exact())
        .and_then(|o| u32::try_from(o).ok())
        .filter(|o| roots_ok && o.checked_add(bytes).is_some_and(|end| end <= limit))
        .ok_or_else(imageblocks)?;
    let slot = match image {
        ImageRef::Slot(index) => Some(SlotRef {
            stage: cx.stage,
            class: SlotClass::Texture,
            index,
        }),
        ImageRef::Heap { .. } => None,
    };
    let (conv, class) = match conv {
        Some(c) => (c, TexelClass::Float),
        None => {
            let class = match (slot, image) {
                (Some(slot), _) => integer_class(cx, slot)?,
                (None, ImageRef::Heap { group, .. }) => heap_integer_class(cx, group)?,
                (None, ImageRef::Slot(_)) => return Err(cx.invariant("a slot without its ref")),
            };
            let conv = match elem.elem {
                Elem::I32 => TexelConv::I32,
                _ => TexelConv::I16 {
                    signed: class == TexelClass::Sint,
                },
            };
            (conv, class)
        }
    };
    observe(cx, image, t.image, Some(class))?;
    if !image_access(cx, image)?.0 {
        return Err(imageblocks());
    }
    // A slice written to a storage slot bound to a swizzled texture is not measured.
    if super::swizzled_storage(cx, image)? {
        return Err(crate::error::Unsupported::not_implemented(
            crate::error::Feature::TextureSwizzle,
        )
        .into());
    }
    slot::stage_stores(cx)?;
    let at = coord(cx, ops, t.image)?;
    // Every cell's offset is `region + k · cell_bytes + field`, the region 16-byte aligned.
    let align = 1u32 << (cell_bytes | field | 16).trailing_zeros();
    let path = if align >= 4 {
        Path::Aligned
    } else if bytes <= 2 && bytes <= align {
        Path::OneWord
    } else {
        Path::Bytes
    };
    let leaf = Leaf {
        form: Form {
            held: held_on(cx.profile, held)?,
            lanes,
        },
        bytes,
        path,
    };
    let guard = slot::write_guard(cx, false);
    let class = super::storage_class(cx, image)?;
    if conv == TexelConv::F32 {
        super::half_storage_facts(cx, image, (class, false, true), m)?;
    }
    if matches!(conv, TexelConv::F32 | TexelConv::F16) {
        super::norm_storage_facts(cx, image, t.image, TexelForm { conv, lanes })?;
    }
    if conv == TexelConv::F16 {
        super::half_write_nans(cx, image, m)?;
    }
    mark(cx, image, |u| u.write = true);
    // Every invocation shares the write where its guard selects local id 0 at a uniform point.
    let cooperative = match super::cooperative::guard(cx, call, m)? {
        Some(cond) => {
            let v = cx
                .values
                .inst(cond)
                .ok_or_else(|| cx.invariant("a guard without a value"))?;
            cx.families.texture.cooperative.insert(v);
            true
        }
        None => false,
    };
    Ok(TextureStrategy::ImageblockSlice(Box::new(Slice {
        image,
        at,
        level,
        size,
        field,
        cell_bytes,
        leaf,
        texel: TexelForm { conv, lanes },
        guard,
        contained: imageblock::contained(cx),
        class,
        cooperative,
    })))
}
