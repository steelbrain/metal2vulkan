//! Push placement: the complete algorithm. Families only contribute fields.
//!
//! Each stage has a push window of the device's window bytes, `W`: its `maxPushConstantsSize` up
//! to [`MOST_BYTES`] (`DeviceProfile::push_window`), 256 on every device that reports 256 or
//! more, and at least [`LEAST_BYTES`]. A compute window is `[0, W)` (a dispatch region and 24
//! buffer addresses below its tail at 256 bytes, 8 at 128), and a render
//! pipeline's two stages share one range `[0, W)`, dense by link: the vertex function's fields
//! from its start ([`base`]), the fragment function's from its `LoweringInputs::push_base`, the
//! vertex function's push end, so only the fields the pair places take space (a four-byte field
//! the fragment function places before its eight-byte ones takes the four bytes the vertex
//! function's end may leave). A vertex function starts at 0, and a post-tessellation one at
//! [`TESSELLATION_BASE`], past the push range link places for the pipeline's control stage at 0,
//! so the pair's fields never take its bytes: where they pass the range they spill, as any
//! function's do, and the pipeline is made on every device.
//! The vertex function's window there ends at `W - 8`, 8 bytes short of the range's end
//! ([`vertex_end`]): a vertex function is lowered without knowing its fragment function, and
//! a fragment function whose fields spill must push its parameter block's address (8 bytes,
//! 8-aligned) from the vertex function's end, so the vertex function leaves that room. The
//! reserved 8 bytes cost only a vertex function whose own fields pass `W - 8`, which spills where
//! it would have filled them. `select` places the function's own [`PushField`]s in the fixed kind
//! order (the derived `Ord` of [`PushField`]; within a kind, by slot). Fields fill the
//! stage window in that order, each at the next offset aligned to the field; from the first field
//! that does not fit, every field goes to the stage's `ArgumentTable` uniform buffer
//! (`ResourceSlot{stage, Synthetic(ArgumentTable)}`), at offsets fixed by the same order. The
//! placement depends only on the function's own field set, so each field has one access form,
//! emitted once, and link never changes it. Under the identity domains the `ArgumentTable` is no
//! descriptor but a parameter block at a device address, which the function pushes first
//! (`PushField::ArgumentTableAddress`) whenever its fields spill, and reads its spilled fields
//! from at the same offsets.
//!
//! On the floor each stage's pushed fields end with its tail ([`tail`]): its highest field of a
//! tail kind ([`tail_kind`], one or two words) moves above the others, [`HOLE_BYTES`] past their
//! end, or, where the pushed fields hold no tail kind, the zero word `PushField::TailWord` does
//! (`PushPlacement::tail_word`).
//! A driver that uploads a stage's push bytes up to the highest one it reads then uploads at least
//! 28 bytes past the other fields, which holds every load it widens from them, and the tail's own
//! load, one or two words, is never widened. The stage reads the tail in its first block and
//! returns where it holds a value the consumer never pushes there ([`emit_tail_guard`]), so no
//! optimizer drops the read. A stage that pushes one field of one or two words has no other load
//! to widen and no tail. The hole moves no field the stage reads and adds no read where the tail
//! is a field of its own, so the words a driver keeps in registers are the same.
//!
//! Field shapes are chosen so that the `ArgumentTable`'s std140 and std430 layouts agree (scalars,
//! 2-vectors, and 16-byte-strided arrays only), so the table is a valid uniform block on every
//! profile without `uniformBufferStandardLayout`.

use std::collections::BTreeSet;

use crate::air::abi::Stage as AirStage;
use crate::lower::select::SelectCx;
use crate::pipeline::interface::{PlacedField, PushLoc, PushPlacement};
use crate::pipeline::sym::{PushField, Stage};

/// The byte range `[start, end)` of `stage`'s push window on a device of window `bytes`
/// (`DeviceProfile::push_window`); `base` is a fragment function's `LoweringInputs::push_base`,
/// or a vertex function's start ([`base`]).
pub const fn window(stage: Stage, base: u32, bytes: u32) -> (u32, u32) {
    match stage {
        Stage::Vertex => (base, vertex_end(bytes)),
        Stage::Fragment => (base, bytes),
        Stage::Kernel => (0, bytes),
        // A mesh render pipeline's object, mesh and fragment functions share the 128 bytes: the
        // fragment function keeps its window, and the vertex function's is split.
        Stage::Object => (0, 32),
        Stage::Mesh => (32, MESH_END),
    }
}

/// The end of a mesh function's window, where a mesh render pipeline's fragment function's
/// fields start.
pub const MESH_END: u32 = 64;

/// The most bytes of a device's push window: a device's window is its `maxPushConstantsSize` up
/// to these (`DeviceProfile::push_window`).
pub const MOST_BYTES: u32 = 256;

/// The least bytes of a device's push window: Vulkan's least `maxPushConstantsSize`, below which
/// `Device::from_queries` refuses a description.
pub const LEAST_BYTES: u32 = 128;

/// Where a post-tessellation vertex function's window starts: past the most bytes the pipeline's
/// control stage pushes in either form (`control::PUSH_BYTES`), whose range link places at 0
/// (`tessellation::push_offset`, the first free offset), so the control stage's range fits beside
/// the vertex and fragment functions' whatever their fields.
pub const TESSELLATION_BASE: u32 = crate::pipeline::link::tessellation::control::PUSH_BYTES;

/// Where the fields of `cx`'s function start in its window ([`window`]): a post-tessellation
/// vertex function's at [`TESSELLATION_BASE`], any other function's at its
/// `LoweringInputs::push_base`, a fragment function's place behind its vertex function and 0 for
/// the rest.
pub fn base(cx: &SelectCx<'_>) -> u32 {
    match cx.entry.ep.stage {
        AirStage::Vertex { patch: Some(_) } => TESSELLATION_BASE,
        _ => cx.inputs.push_base,
    }
}

/// Where a fragment function's fields start on the floor behind vertex function placement
/// `vertex`, of a post-tessellation function or not (`tessellated`): the vertex function's push
/// end, or where it pushes nothing, its window's start ([`base`]).
pub fn fragment_base(vertex: &PushPlacement, tessellated: bool) -> u32 {
    let start = match tessellated {
        true => TESSELLATION_BASE,
        false => 0,
    };
    (vertex.push_range).map_or(start, |(offset, size)| offset + size)
}

/// The end of a vertex function's window on a device of window `bytes`: the window's end less one
/// address, so the fragment function's window, from the vertex function's end, always holds its
/// parameter block's address (`PushField::ArgumentTableAddress`, which the order places first, at
/// the next multiple of 8).
pub const fn vertex_end(bytes: u32) -> u32 {
    bytes - 8
}

/// The bytes between the end of a floor stage's other pushed fields and its tail ([`tail`]). A
/// driver widens a push load past its last byte by at most 28 bytes, so a load from those fields
/// ends inside the tail's first word, and it never merges a load with one 24 bytes or more away.
pub const HOLE_BYTES: u32 = 24;

/// Whether a field of kind `field` can be a floor stage's tail: one or two words whose value the
/// consumer never pushes with every bit set, which the stage returns on ([`emit_tail_guard`]). A
/// slot's bounds hold a length of at most 2 GiB, and the zero range is at a multiple of 16
/// (`PushField::SlotBounds`, `PushField::ZeroAddress`).
pub const fn tail_kind(field: PushField) -> bool {
    matches!(field, PushField::SlotBounds(_) | PushField::ZeroAddress)
}

/// The SPIR-V shape of a push field's value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FieldShape {
    /// A `u32`.
    U32,
    /// A `u64` (device addresses; only placed on profiles with buffer device addresses).
    U64,
    /// A `u32` 2-vector.
    UVec2,
    /// An array of `n` `u32` 4-vectors (stride 16).
    UVec4Array(u32),
}

impl FieldShape {
    /// Size in bytes.
    pub const fn size(self) -> u32 {
        match self {
            FieldShape::U32 => 4,
            FieldShape::U64 | FieldShape::UVec2 => 8,
            FieldShape::UVec4Array(n) => 16 * n,
        }
    }

    /// Alignment in bytes (the same under std140 and std430 for these shapes).
    pub const fn align(self) -> u32 {
        match self {
            FieldShape::U32 => 4,
            FieldShape::U64 | FieldShape::UVec2 => 8,
            FieldShape::UVec4Array(_) => 16,
        }
    }
}

/// The shape of field `f`.
pub const fn shape(f: PushField) -> FieldShape {
    match f {
        PushField::ArgumentTableAddress
        | PushField::RootTableAddress
        | PushField::ExecutionStatusAddress
        | PushField::CaptureIndexAddress
        | PushField::MergedDrawAddress
        | PushField::CommandRecordsAddress
        | PushField::ReplayRegionAddress
        | PushField::TessellationFactorAddress
        | PushField::PrivateArenaAddress => FieldShape::U64,
        // A Metal address, low word first: the same eight bytes as a `u64`, readable without
        // `shaderInt64`.
        PushField::SlotGuestBase(_) => FieldShape::UVec2,
        // Region base `xyz` and grid `xyz`, one 4-vector each (`w` unused); a stage-in region's
        // origin and size the same way.
        PushField::DispatchRegion | PushField::StageInRegion => FieldShape::UVec4Array(2),
        // A draw's depth clamp range, two `f32`s' bits.
        PushField::DepthClampRange => FieldShape::UVec2,
        // The alpha test reference's code.
        PushField::AlphaTestReference => FieldShape::U32,
        // The blend color and one minus it, four `f32`s' bits each.
        // The blend color as the encoder set it, four `f32`s' bits.
        PushField::HalfBlendColor => FieldShape::UVec4Array(1),
        // A draw's depth bias: its constant, slope and clamp as `f32`s' bits, and a zero word.
        // Base vertex, base instance.
        PushField::BaseVertexInstance => FieldShape::UVec2,
        // The draw's vertices per primitive.
        PushField::VerticesPerPrimitive => FieldShape::U32,
        // A zero word, the index count, the first instance, and the base vertex.
        PushField::CaptureLayout => FieldShape::UVec4Array(1),
        // A constexpr sampler's index in its use's sampler array, and the sampled-texture array's
        // element a fetched color attachment is read through.
        PushField::ConstexprSampler(_) | PushField::FetchedColor { .. } => FieldShape::U32,
        // The residual is below `min{Storage,Uniform}BufferOffsetAlignment`.
        PushField::SlotBindOffset(_) | PushField::VertexBufferStride(_) => FieldShape::U32,
        // The bytes of a slot's buffer before its bind offset, and its length: each at most 2 GiB
        // under the identity domains.
        PushField::SlotBounds(_) => FieldShape::UVec2,
        // The identity domains' zero range, and an indirect command buffer's execution range.
        PushField::ZeroAddress | PushField::IndirectRange => FieldShape::U64,
        // The command of an indirect command buffer's execution, and its part's first.
        PushField::IndirectSequence | PushField::IndirectBase => FieldShape::U32,
        // A texture's mipmap level count, an acceleration structure's instance count, the render
        // encoder's viewport count, the render pass's array length, a kernel stage-in buffer's
        // bound stride, a storage texture's integer channels' widths, a bicubic slot's format
        // treatment.
        PushField::TextureLevels(_)
        | PushField::TextureIntegerChannels(_)
        | PushField::BicubicTexels(_)
        | PushField::InstanceCount(_)
        | PushField::ViewportCount
        | PushField::RenderTargetArrayLength
        | PushField::StageInStride(_) => FieldShape::U32,
        // A draw of one amplification's id and offsets.
        PushField::AmplificationId
        | PushField::AmplificationLayerOffset
        | PushField::AmplificationViewportOffset => FieldShape::U32,
        // A synthesized tessellation control stage's words: the scale is a `float`'s bits.
        PushField::TessellationFactorOffset
        | PushField::TessellationFactorInstanceStride
        | PushField::TessellationFactorScale
        | PushField::PatchIndexOffset
        | PushField::PatchIndexBound
        | PushField::ControlPointIndexOffset => FieldShape::U32,
        // A band's viewport half extent and half width, four `f32`s.
        PushField::LineBand => FieldShape::UVec4Array(1),
        // The whole indices from the bound index offset to the buffer's end, unsaturated:
        // `firstIndex + indexCount` reaches `2^33`.
        PushField::IndexCapacity => FieldShape::U64,
        // The floor's tail word, pushed zero.
        PushField::TailWord => FieldShape::U32,
    }
}

fn align_up(v: u32, a: u32) -> u32 {
    v.div_ceil(a).saturating_mul(a)
}

/// Place `fields` for `stage` in its push window of a device of window `bytes`, from a fragment
/// function's `base` there ([`window`]), with its tail ([`tail`]).
pub fn place(stage: Stage, base: u32, bytes: u32, fields: &BTreeSet<PushField>) -> PushPlacement {
    let (start, end) = window(stage, base, bytes);
    let (pushed, offsets, word, push_end) = floor(start, end, fields);
    let mut table = 0u32;
    let mut placed = Vec::with_capacity(fields.len());
    // `BTreeSet` iterates in the derived `Ord`: exactly the kind order, then slot order.
    for (i, &field) in fields.iter().enumerate() {
        let sh = shape(field);
        let loc = match offsets.get(i) {
            Some(&offset) if i < pushed => PushLoc::Push { offset },
            _ => {
                let off = align_up(table, sh.align());
                table = off.saturating_add(sh.size());
                PushLoc::ArgumentTable { offset: off }
            }
        };
        placed.push(PlacedField {
            field,
            loc,
            size: sh.size(),
        });
    }
    PushPlacement {
        stage,
        push_range: (push_end > start).then_some((start, push_end - start)),
        argument_table_size: align_up(table, 16),
        fields: placed,
        tail_word: word,
    }
}

/// How many of `fields`, a prefix of the kind order, `[start, end)` holds packed: each at the
/// next offset aligned to it, until the first that does not fit; their offsets, and their end.
/// Reads `fields` only up to that first one.
fn dense(start: u32, end: u32, fields: &BTreeSet<PushField>) -> (usize, Vec<u32>, u32) {
    let mut cursor = start;
    let mut offsets = Vec::new();
    for &field in fields {
        let sh = shape(field);
        let at = align_up(cursor, sh.align());
        if at.saturating_add(sh.size()) > end {
            break;
        }
        cursor = at + sh.size();
        offsets.push(at);
    }
    (offsets.len(), offsets, cursor)
}

/// The floor's placement of `fields` in `[start, end)`: the longest prefix of the kind order that
/// fits with its tail ([`tail_layout`]); how many it pushes, their offsets, the tail word's offset
/// where it takes one, and the push's end. The prefix the window holds packed bounds it: the hole
/// only takes room, so `fields` is read only up to the first field that does not fit packed.
fn floor(
    start: u32,
    end: u32,
    fields: &BTreeSet<PushField>,
) -> (usize, Vec<u32>, Option<u32>, u32) {
    let (most, _, _) = dense(start, end, fields);
    let prefix: Vec<PushField> = fields.iter().copied().take(most).collect();
    (0..=most)
        .rev()
        .find_map(|n| {
            let pushed = prefix.get(..n)?;
            tail_layout(start, end, pushed)
                .map(|(offsets, word, push_end)| (n, offsets, word, push_end))
        })
        .unwrap_or((0, Vec::new(), None, start))
}

/// `pushed`, in kind order, laid out from `start` with its tail, where they fit by `end`: every
/// field but the highest of a tail kind packed in order, then that one [`HOLE_BYTES`] above their
/// end, or, where none is of a tail kind, the tail word there. One field of one or two words, or
/// none, is placed alone. The offsets by index in `pushed`, the tail word's, and the push's end.
fn tail_layout(start: u32, end: u32, pushed: &[PushField]) -> Option<(Vec<u32>, Option<u32>, u32)> {
    match pushed {
        [] => return Some((Vec::new(), None, start)),
        [only] if shape(*only).size() <= 8 => {
            let sh = shape(*only);
            let at = align_up(start, sh.align());
            return (at.saturating_add(sh.size()) <= end).then(|| (vec![at], None, at + sh.size()));
        }
        _ => {}
    }
    let tail = pushed.iter().rposition(|&f| tail_kind(f));
    let mut offsets = Vec::with_capacity(pushed.len());
    let mut cursor = start;
    for (i, &field) in pushed.iter().enumerate() {
        if Some(i) == tail {
            // Its offset comes once the others' end is known.
            offsets.push(0);
            continue;
        }
        let sh = shape(field);
        let at = align_up(cursor, sh.align());
        offsets.push(at);
        cursor = at + sh.size();
    }
    let top = tail.and_then(|i| pushed.get(i).copied());
    let sh = shape(top.unwrap_or(PushField::TailWord));
    let at = align_up(cursor.saturating_add(HOLE_BYTES), sh.align());
    let push_end = at.saturating_add(sh.size());
    if push_end > end {
        return None;
    }
    let word = match tail.and_then(|i| offsets.get_mut(i)) {
        Some(offset) => {
            *offset = at;
            None
        }
        None => Some(at),
    };
    Some((offsets, word, push_end))
}

/// Whether `stage`'s placement of `fields` spills into the `ArgumentTable`: whether [`place`]
/// gives it a size. That is the first field with a size that does not fit, so the fields are read
/// only up to it, at most the window's worth plus one, however many there are; a step that asks
/// once per slot it moves (`access::finish_slots`) then costs no walk of every slot's field.
pub fn spills(stage: Stage, base: u32, bytes: u32, fields: &BTreeSet<PushField>) -> bool {
    let (start, end) = window(stage, base, bytes);
    let pushed = floor(start, end, fields).0;
    // Every shape has a size, so any field past the pushed ones gives the table one.
    pushed < fields.len()
}

/// The tail of `placement` ([`place`] on the floor): its tail word (`PushField::TailWord`), or
/// its highest pushed field, where it lies at least [`HOLE_BYTES`] above every other pushed
/// field's end.
pub fn tail(placement: &PushPlacement) -> Option<PushField> {
    let pushed = || {
        (placement.fields.iter()).filter_map(|f| match f.loc {
            PushLoc::Push { offset } => Some((f.field, offset, f.size)),
            PushLoc::ArgumentTable { .. } => None,
        })
    };
    let below = |skip: Option<PushField>| {
        pushed()
            .filter(|&(f, _, _)| Some(f) != skip)
            .map(|(_, offset, size)| offset + size)
            .max()
    };
    let (field, at, skip) = match placement.tail_word {
        Some(at) => (PushField::TailWord, at, None),
        None => {
            let (field, at, _) = pushed().max_by_key(|&(_, offset, _)| offset)?;
            (field, at, Some(field))
        }
    };
    (at >= below(skip)? + HOLE_BYTES).then_some(field)
}

/// The floor's bound on `placement`, a stage's: a stage that pushes more than one field, or one
/// of more than two words, ends with its tail ([`tail`]), a field of a tail kind or the tail word,
/// of one or two words; its push range ends with its highest pushed byte, by the end of a device
/// window of `bytes`. `Err` names the placement that passes it.
pub fn check_floor(placement: &PushPlacement, bytes: u32) -> Result<(), String> {
    let pushed: Vec<(PushField, u32, u32)> = (placement.fields.iter())
        .filter_map(|f| match f.loc {
            PushLoc::Push { offset } => Some((f.field, offset, f.size)),
            PushLoc::ArgumentTable { .. } => None,
        })
        .collect();
    let word = placement
        .tail_word
        .map(|at| at + shape(PushField::TailWord).size());
    let top = (pushed.iter())
        .map(|&(_, offset, size)| offset + size)
        .chain(word)
        .max();
    let single = word.is_none() && matches!(pushed.as_slice(), [(_, _, size)] if *size <= 8);
    let tailed = tail(placement)
        .is_some_and(|t| (tail_kind(t) || t == PushField::TailWord) && shape(t).size() <= 8);
    let range = placement.push_range.map(|(offset, size)| offset + size);
    let ok =
        (pushed.is_empty() || single || tailed) && range == top && top.is_none_or(|t| t <= bytes);
    match ok {
        true => Ok(()),
        false => Err(format!(
            "{:?} push range {:?} with fields {pushed:?} and tail word {:?} passes the floor's \
             tail bound",
            placement.stage, placement.push_range, placement.tail_word
        )),
    }
}

/// In the open block of a function whose push block holds the stage's tail `field` ([`tail`]),
/// `value` its member as the block read it: return from the function where it holds the value the
/// consumer never pushes there: the tail word non-zero, a two-word field with both words all
/// ones, an address all ones. The branch is never taken; the return keeps the read from any
/// optimizer, as everything the function does after it depends on it.
pub(crate) fn emit_tail_guard(
    fb: &mut crate::spirv::builder::FunctionBuilder<'_>,
    value: crate::spirv::ids::Value,
    field: PushField,
) -> crate::spirv::error::BResult<()> {
    use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
    use crate::spirv::ir::{BinOp, ConstBits, Constant};
    let set = match (field, shape(field)) {
        (PushField::TailWord, _) => {
            let zero = fb.u32_const(0)?;
            let zero = fb.module().const_value(zero)?;
            fb.binary(BinOp::INotEqual, value, zero)?
        }
        (_, FieldShape::UVec2) => {
            let low = fb.composite_extract(value, &[0])?;
            let high = fb.composite_extract(value, &[1])?;
            let both = fb.binary(BinOp::BitwiseAnd, low, high)?;
            let ones = fb.u32_const(u32::MAX)?;
            let ones = fb.module().const_value(ones)?;
            fb.binary(BinOp::IEqual, both, ones)?
        }
        (_, FieldShape::U64) => {
            let ty = fb.uint_ty(64)?;
            let ones = fb.module().constant(Constant::Scalar {
                ty,
                bits: ConstBits(u64::MAX),
            })?;
            let ones = fb.module().const_value(ones)?;
            fb.binary(BinOp::IEqual, value, ones)?
        }
        (_, FieldShape::U32 | FieldShape::UVec4Array(_)) => return Ok(()),
    };
    let leave: ArmFn<'_, ()> = Box::new(|fb, _| {
        fb.ret(None)?;
        Ok(ArmEnd::Exits)
    });
    fb.if_(&mut (), set, leave, None)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::sym::{SlotClass, SlotRef};

    fn slot(stage: Stage, index: u32) -> SlotRef {
        SlotRef {
            stage,
            class: SlotClass::Buffer,
            index,
        }
    }

    /// The placement invariants every field set satisfies: placement order, alignment, push fields
    /// inside the window without overlap, the prefix property (nothing is pushed after the first
    /// spill), and a push range and table size that cover their fields. On the floor the tail
    /// ([`tail`]) lies above every other pushed field and the floor's bound holds
    /// ([`check_floor`]); the guaranteed window has no tail. In a window of [`MOST_BYTES`].
    fn check(stage: Stage, fields: &BTreeSet<PushField>) -> PushPlacement {
        check_in(stage, 0, MOST_BYTES, fields)
    }

    /// [`check`] from a fragment function's push `base`.
    fn check_at(stage: Stage, base: u32, fields: &BTreeSet<PushField>) -> PushPlacement {
        check_in(stage, base, MOST_BYTES, fields)
    }

    /// [`check_at`] in a device window of `bytes`.
    fn check_in(
        stage: Stage,
        base: u32,
        bytes: u32,
        fields: &BTreeSet<PushField>,
    ) -> PushPlacement {
        let p = place(stage, base, bytes, fields);
        assert_eq!(
            spills(stage, base, bytes, fields),
            p.argument_table_size > 0
        );
        let (start, end) = window(stage, base, bytes);
        assert_eq!(p.stage, stage);
        let order: Vec<PushField> = p.fields.iter().map(|f| f.field).collect();
        assert_eq!(order, fields.iter().copied().collect::<Vec<_>>());
        assert_eq!(check_floor(&p, bytes), Ok(()));
        let top = tail(&p);
        let (mut push_end, mut table_end, mut spilled) = (start, 0u32, false);
        let mut tail_end = p.tail_word.map(|at| (at, at + 4));
        for f in &p.fields {
            let sh = shape(f.field);
            assert_eq!(f.size, sh.size());
            match f.loc {
                PushLoc::Push { offset } if Some(f.field) == top => {
                    assert_eq!(offset % sh.align(), 0);
                    assert!(offset + f.size <= end);
                    tail_end = Some((offset, offset + f.size));
                }
                PushLoc::Push { offset } => {
                    assert!(!spilled, "{:?} pushed after a spill", f.field);
                    assert_eq!(offset % sh.align(), 0);
                    assert!(offset >= push_end && offset + f.size <= end);
                    push_end = offset + f.size;
                }
                PushLoc::ArgumentTable { offset } => {
                    spilled = true;
                    assert_eq!(offset % sh.align(), 0);
                    assert!(offset >= table_end);
                    table_end = offset + f.size;
                }
            }
        }
        if let Some((at, tail_end)) = tail_end {
            assert!(at >= push_end + HOLE_BYTES, "the tail above the hole");
            push_end = tail_end;
        }
        match p.push_range {
            Some((off, size)) => {
                assert_eq!(off, start);
                assert_eq!(off + size, push_end);
                assert_eq!(size % 4, 0);
            }
            None => assert_eq!(push_end, start),
        }
        assert!(p.argument_table_size >= table_end);
        assert_eq!(p.argument_table_size % 16, 0);
        p
    }

    #[test]
    fn an_empty_field_set_places_nothing() {
        for stage in [Stage::Vertex, Stage::Fragment, Stage::Kernel] {
            let p = check(stage, &BTreeSet::new());
            assert!(p.fields.is_empty());
            assert_eq!(p.push_range, None);
            assert_eq!(p.argument_table_size, 0);
        }
    }

    /// A kernel's dispatch region and its buffers' addresses. The addresses hold no tail kind, so
    /// the tail word ends the push, 24 bytes above them. In a window of 256 bytes 24 addresses end
    /// at 224 and the word at 248, and a 25th spills; with a spill, the table's address and the
    /// region's alignment take the first 16 bytes, and 22 addresses stay in the window. In a window
    /// of 128 bytes, 8 addresses end at 96 and the word at 120, a 9th spills, and with a spill 6
    /// stay.
    #[test]
    fn the_window_holds_a_region_and_24_or_8_addresses_below_its_tail() {
        let fields = |n: u32| -> BTreeSet<PushField> {
            let mut f: BTreeSet<PushField> = (0..n)
                .map(|i| PushField::SlotGuestBase(slot(Stage::Kernel, i)))
                .collect();
            f.insert(PushField::DispatchRegion);
            f
        };
        let pushed = |p: &PushPlacement| {
            p.fields
                .iter()
                .filter(|f| matches!(f.loc, PushLoc::Push { .. }))
                .count()
        };
        for (bytes, most, kept) in [(MOST_BYTES, 24, 22), (LEAST_BYTES, 8, 6)] {
            let floor = check_in(Stage::Kernel, 0, bytes, &fields(most));
            assert_eq!(
                (floor.push_range, floor.argument_table_size),
                (Some((0, bytes - 4)), 0),
                "{bytes}"
            );
            assert_eq!(pushed(&floor), most as usize + 1, "the region and {most}");
            assert_eq!(
                (tail(&floor), floor.tail_word),
                (Some(PushField::TailWord), Some(bytes - 8))
            );
            assert!(spills(Stage::Kernel, 0, bytes, &fields(most + 1)));
            let mut spilled = fields(most + 1);
            spilled.insert(PushField::ArgumentTableAddress);
            let p = check_in(Stage::Kernel, 0, bytes, &spilled);
            assert_eq!(
                (p.push_range, p.argument_table_size),
                (Some((0, bytes - 4)), 32)
            );
            assert_eq!(
                (pushed(&p), p.tail_word),
                (kept as usize + 2, Some(bytes - 8)),
                "the address, the region and {kept} buffer addresses"
            );
        }
    }

    /// The floor's tail: a kernel's highest field of a tail kind moves 24 bytes above its other
    /// fields, the rest packed as before; the tail word ends a push of no tail kind, and a single
    /// field of one or two words has none.
    #[test]
    fn the_floor_places_each_stage_s_tail_above_a_hole() {
        let at = |p: &PushPlacement, field: PushField| {
            (p.fields.iter())
                .find(|f| f.field == field)
                .map(|f| f.loc)
                .expect("placed")
        };
        let (a, b) = (slot(Stage::Kernel, 0), slot(Stage::Kernel, 1));
        // A region, two addresses, the zero range and one slot's bounds.
        let mut fields: BTreeSet<PushField> = [
            PushField::DispatchRegion,
            PushField::SlotGuestBase(a),
            PushField::SlotGuestBase(b),
            PushField::ZeroAddress,
            PushField::SlotBounds(a),
        ]
        .into();
        let p = check(Stage::Kernel, &fields);
        assert_eq!(tail(&p), Some(PushField::SlotBounds(a)));
        assert_eq!(at(&p, PushField::ZeroAddress), PushLoc::Push { offset: 48 });
        assert_eq!(
            at(&p, PushField::SlotBounds(a)),
            PushLoc::Push { offset: 80 }
        );
        assert_eq!(p.push_range, Some((0, 88)));
        assert_eq!(p.tail_word, None);
        // A second slot's bounds: the higher one is the tail, the first stays packed.
        fields.insert(PushField::SlotBounds(b));
        let p = check(Stage::Kernel, &fields);
        assert_eq!(tail(&p), Some(PushField::SlotBounds(b)));
        assert_eq!(
            at(&p, PushField::SlotBounds(a)),
            PushLoc::Push { offset: 56 }
        );
        assert_eq!(
            at(&p, PushField::SlotBounds(b)),
            PushLoc::Push { offset: 88 }
        );
        assert_eq!(p.push_range, Some((0, 96)));
        // A region alone: the tail word, 24 bytes above it.
        let region = check(Stage::Kernel, &[PushField::DispatchRegion].into());
        assert_eq!(tail(&region), Some(PushField::TailWord));
        assert_eq!(region.tail_word, Some(56));
        assert_eq!(region.push_range, Some((0, 60)));
        // One address alone: no tail.
        let one = check(Stage::Kernel, &[PushField::SlotGuestBase(a)].into());
        assert_eq!((tail(&one), one.push_range), (None, Some((0, 8))));
        assert!(check(Stage::Kernel, &BTreeSet::new()).fields.is_empty());
    }

    /// The floor's bound, which `select` checks on every placement: a stage that pushes more than
    /// one short field without a tail 24 bytes above the others, a long field on top, a range that
    /// does not end with the highest pushed byte or passes the window, and the tail word off the
    /// floor are refused.
    #[test]
    fn the_floor_refuses_a_placement_without_its_tail() {
        let a = slot(Stage::Kernel, 0);
        let fields: BTreeSet<PushField> = [
            PushField::DispatchRegion,
            PushField::SlotGuestBase(a),
            PushField::SlotBounds(a),
        ]
        .into();
        let p = place(Stage::Kernel, 0, MOST_BYTES, &fields);
        assert_eq!(check_floor(&p, MOST_BYTES), Ok(()));
        let mut short = p.clone();
        short.push_range = Some((0, 80));
        assert!(
            check_floor(&short, MOST_BYTES).is_err(),
            "a range that does not end with the tail"
        );
        let mut moved = p.clone();
        for f in &mut moved.fields {
            if f.field == PushField::SlotBounds(a) {
                f.loc = PushLoc::Push { offset: 56 };
            }
        }
        moved.push_range = Some((0, 64));
        assert!(check_floor(&moved, MOST_BYTES).is_err(), "a hole of 16");
        let region = place(
            Stage::Kernel,
            0,
            MOST_BYTES,
            &[PushField::DispatchRegion].into(),
        );
        assert_eq!(check_floor(&region, MOST_BYTES), Ok(()));
        let mut bare = region.clone();
        bare.tail_word = None;
        bare.push_range = Some((0, 32));
        assert!(
            check_floor(&bare, MOST_BYTES).is_err(),
            "a long field alone"
        );
        let mut past = region;
        past.push_range = Some((0, 260));
        assert!(check_floor(&past, MOST_BYTES).is_err());
        // A placement of a 256-byte window that passes 128 bytes is past a 128-byte window.
        let wide = place(Stage::Kernel, 0, MOST_BYTES, &fields_of(20));
        assert_eq!(check_floor(&wide, MOST_BYTES), Ok(()));
        assert!(check_floor(&wide, LEAST_BYTES).is_err());
    }

    /// A kernel's region and `n` buffer addresses.
    fn fields_of(n: u32) -> BTreeSet<PushField> {
        (0..n)
            .map(|i| PushField::SlotGuestBase(slot(Stage::Kernel, i)))
            .chain([PushField::DispatchRegion])
            .collect()
    }

    /// The windows a pipeline of each kind pushes on a device of window `W`, 256 or 128 bytes,
    /// pinned: a compute kernel's `[0, W)`, and a render pipeline's one range `[0, W)`, the vertex
    /// function's from its start (0, or 24 for a post-tessellation function) to `W - 8` and the
    /// fragment function's from its base. A compute kernel's window does not move with a render
    /// base, and a mesh pipeline's windows do not move with `W`.
    #[test]
    fn each_kind_pushes_its_own_window() {
        assert_eq!(TESSELLATION_BASE, 24);
        for (bytes, vertex) in [(256, 248), (128, 120)] {
            for base in [0, 8, 40, vertex] {
                assert_eq!(window(Stage::Kernel, base, bytes), (0, bytes));
                assert_eq!(window(Stage::Fragment, base, bytes), (base, bytes));
            }
            for base in [0, TESSELLATION_BASE] {
                assert_eq!(window(Stage::Vertex, base, bytes), (base, vertex));
            }
            assert_eq!(window(Stage::Object, 0, bytes), (0, 32));
            assert_eq!(window(Stage::Mesh, 0, bytes), (32, MESH_END));
        }
    }

    /// The control stage's push range fits below a post-tessellation vertex function's window in
    /// either form, with every factor step and scale, so link places it at 0 beside any pair of
    /// functions; a fragment function behind such a vertex function that pushes nothing starts
    /// past it too, and behind any other at 0.
    #[test]
    fn the_control_stage_fits_below_a_post_tessellation_window() {
        use crate::metal::TessellationFactorStepFunction as Step;
        use crate::pipeline::link::tessellation::control::Control;
        let steps = [
            Step::Constant,
            Step::PerPatch,
            Step::PerInstance,
            Step::PerPatchAndPerInstance,
        ];
        for addressed in [false, true] {
            for step in steps {
                for scale in [false, true] {
                    let (bytes, align) = Control::push_bytes(addressed, step, scale);
                    assert!(bytes <= TESSELLATION_BASE, "{addressed} {step:?} {scale}");
                    assert_eq!(TESSELLATION_BASE % align, 0);
                }
            }
        }
        let empty = place(
            Stage::Vertex,
            TESSELLATION_BASE,
            LEAST_BYTES,
            &BTreeSet::new(),
        );
        assert_eq!(empty.push_range, None);
        assert_eq!(fragment_base(&empty, true), TESSELLATION_BASE);
        assert_eq!(fragment_base(&empty, false), 0);
        let one = BTreeSet::from([PushField::ArgumentTableAddress]);
        let pushed = place(Stage::Vertex, TESSELLATION_BASE, LEAST_BYTES, &one);
        assert_eq!(pushed.push_range, Some((TESSELLATION_BASE, 8)));
        assert_eq!(fragment_base(&pushed, true), TESSELLATION_BASE + 8);
    }

    /// On the floor a render pipeline's fields are dense by link: a fragment function placed from
    /// the vertex function's push end starts there, its fields fill the rest of the window `W`
    /// below its own tail, and from the first that does not fit it spills. Each stage has its own
    /// tail: here the addresses hold no tail kind, so each ends with the tail word. A vertex
    /// function's window ends at `W - 8`, so a fragment function from its end can always push its
    /// block's address. From 24, a fragment function pushes 25 addresses at 256 bytes and 9 at
    /// 128; a vertex function 27 and 11.
    #[test]
    fn the_floor_places_a_fragment_function_from_its_base() {
        let vertex: BTreeSet<PushField> = (0..3)
            .map(|i| PushField::SlotGuestBase(slot(Stage::Vertex, i)))
            .collect();
        let v = check_at(Stage::Vertex, 0, &vertex);
        assert_eq!(v.push_range, Some((0, 52)), "the tail word at 48");
        let fragment = |n: u32| -> BTreeSet<PushField> {
            (0..n)
                .map(|i| PushField::SlotGuestBase(slot(Stage::Fragment, i)))
                .collect()
        };
        let f = check_at(Stage::Fragment, 52, &fragment(2));
        assert_eq!(
            f.push_range,
            Some((52, 48)),
            "two addresses from 56, the word at 96"
        );
        let wide = |n: u32| -> BTreeSet<PushField> {
            (0..n)
                .map(|i| PushField::SlotGuestBase(slot(Stage::Vertex, i)))
                .collect()
        };
        for (bytes, most, widest) in [(MOST_BYTES, 25, 27), (LEAST_BYTES, 9, 11)] {
            // From 24, `most` addresses end below the word at `bytes - 8`, and one more spills.
            let full = check_in(Stage::Fragment, 24, bytes, &fragment(most));
            assert_eq!(full.push_range, Some((24, 8 * most + 28)), "{bytes}");
            assert_eq!(full.tail_word, Some(bytes - 8));
            assert_eq!(full.argument_table_size, 0);
            assert!(spills(Stage::Fragment, 24, bytes, &fragment(most + 1)));
            // A vertex function's window ends at `bytes - 8`: `widest` addresses end below the
            // word at `bytes - 16`, and one more spills.
            let v = check_in(Stage::Vertex, 0, bytes, &wide(widest));
            assert_eq!(
                (v.push_range, v.argument_table_size),
                (Some((0, 8 * widest + 28)), 0)
            );
            assert_eq!(v.tail_word, Some(bytes - 16));
            assert!(spills(Stage::Vertex, 0, bytes, &wide(widest + 1)));
            // The fragment function from the fullest vertex function's end, `bytes - 12`, pushes
            // its block's address, 8-aligned, at `bytes - 8`, alone, and spills the rest; from
            // `bytes - 8` the same.
            let mut spilled = fragment(2);
            spilled.insert(PushField::ArgumentTableAddress);
            let (last, fullest) = (bytes - 8, bytes - 12);
            for (base, range) in [(last, (last, 8)), (fullest, (fullest, 12))] {
                let f = check_in(Stage::Fragment, base, bytes, &spilled);
                assert_eq!((f.push_range, f.argument_table_size), (Some(range), 16));
                assert_eq!(
                    (f.fields[0].field, f.fields[0].loc),
                    (
                        PushField::ArgumentTableAddress,
                        PushLoc::Push { offset: last }
                    ),
                    "from {base} in {bytes}"
                );
            }
        }
    }

    /// The slot words an indirect command buffer's command reaches: a vertex function of an
    /// indirect draw pushes the execution range's address, its sequence word and its part's first
    /// command (16 bytes), then its buffer slots' addresses, below the tail word. Its window ends at
    /// `W - 8`, so it pushes 25 slot addresses at 256 bytes and 9 at 128, and where its fields
    /// spill (its block's address first) 24 and 8. An execution refuses a command whose slot's
    /// address is in the block (`IndirectCommandsKind::DrawInput`, in the runtime).
    #[test]
    fn an_indirect_draw_s_vertex_function_pushes_25_or_9_slot_words() {
        let indirect = |n: u32, spilled: bool| -> BTreeSet<PushField> {
            (0..n)
                .map(|i| PushField::SlotGuestBase(slot(Stage::Vertex, i)))
                .chain([
                    PushField::IndirectRange,
                    PushField::IndirectSequence,
                    PushField::IndirectBase,
                ])
                .chain(spilled.then_some(PushField::ArgumentTableAddress))
                .collect()
        };
        let slots = |p: &PushPlacement| {
            (p.fields.iter())
                .filter(|f| matches!(f.field, PushField::SlotGuestBase(_)))
                .filter(|f| matches!(f.loc, PushLoc::Push { .. }))
                .count() as u32
        };
        for (bytes, most, spilling) in [(MOST_BYTES, 25, 24), (LEAST_BYTES, 9, 8)] {
            let p = check_in(Stage::Vertex, 0, bytes, &indirect(most, false));
            assert_eq!((slots(&p), p.argument_table_size), (most, 0), "{bytes}");
            assert!(spills(Stage::Vertex, 0, bytes, &indirect(most + 1, false)));
            let p = check_in(Stage::Vertex, 0, bytes, &indirect(31, true));
            assert_eq!(slots(&p), spilling, "{bytes}, spilled");
        }
    }

    /// A synthetic sweep over field sets built from every kind: the invariants hold for each.
    #[test]
    fn placement_invariants_hold_over_synthetic_field_sets() {
        let kinds = |stage: Stage, i: u32| {
            [
                PushField::RootTableAddress,
                PushField::ExecutionStatusAddress,
                PushField::DispatchRegion,
                PushField::BaseVertexInstance,
                PushField::SlotBindOffset(slot(stage, i)),
                PushField::SlotGuestBase(slot(stage, i)),
                PushField::VertexBufferStride(i),
                PushField::ZeroAddress,
                PushField::SlotBounds(slot(stage, i)),
            ]
        };
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for stage in [Stage::Vertex, Stage::Fragment, Stage::Kernel] {
            for _ in 0..200 {
                let mut fields = BTreeSet::new();
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let n = (state % 40) as u32;
                for i in 0..n {
                    let kinds = kinds(stage, i % 31);
                    let pick = ((state >> (i % 50)) % kinds.len() as u64) as usize;
                    fields.insert(kinds[pick]);
                }
                for bytes in [MOST_BYTES, LEAST_BYTES] {
                    check_in(stage, 0, bytes, &fields);
                }
            }
        }
    }

    /// `ExecutionStatusAddress` is always in the push block, where its readers find it: the
    /// resident-miss count, and `spirv::requirements::writes`, which knows the `ExecutionStatus`
    /// buffer by a load of this member and so needs no `ArgumentTable` path. The kind order (the
    /// derived `Ord` of `PushField`) puts it after `ArgumentTableAddress` and `RootTableAddress`,
    /// the only kinds that sort before it, of which a function places one at most (the first is
    /// the identity domains', the second the table domains'), and placement fills the window as a
    /// prefix of that order, so the field sits in the first 16 bytes of its stage window whatever
    /// else the function pushes; every window holds 64. Checked for every stage, with neither
    /// address and with each, alone and with every later kind at its largest count (every buffer
    /// slot's residual and guest base, 31 strides), which spills.
    #[test]
    fn the_status_address_is_always_placed_in_the_push_block() {
        for stage in [Stage::Vertex, Stage::Fragment, Stage::Kernel] {
            let slot = |index| SlotRef {
                stage,
                class: SlotClass::Buffer,
                index,
            };
            let kinds = [
                PushField::ArgumentTableAddress,
                PushField::RootTableAddress,
                PushField::ExecutionStatusAddress,
                PushField::DispatchRegion,
                PushField::BaseVertexInstance,
                PushField::SlotBindOffset(slot(0)),
                PushField::SlotGuestBase(slot(0)),
                PushField::VertexBufferStride(0),
            ];
            let before: Vec<PushField> = kinds
                .into_iter()
                .filter(|k| *k < PushField::ExecutionStatusAddress)
                .collect();
            assert_eq!(
                before,
                [PushField::ArgumentTableAddress, PushField::RootTableAddress]
            );
            let mut later =
                BTreeSet::from([PushField::DispatchRegion, PushField::BaseVertexInstance]);
            for i in 0..31 {
                later.insert(PushField::SlotBindOffset(slot(i)));
                later.insert(PushField::SlotGuestBase(slot(i)));
                later.insert(PushField::VertexBufferStride(i));
            }
            let (start, end) = window(stage, 0, LEAST_BYTES);
            let firsts = [
                None,
                Some(PushField::RootTableAddress),
                Some(PushField::ArgumentTableAddress),
            ];
            for first in firsts {
                for rest in [BTreeSet::new(), later.clone()] {
                    let spills = !rest.is_empty();
                    let mut fields = rest;
                    fields.insert(PushField::ExecutionStatusAddress);
                    fields.extend(first);
                    let placement = place(stage, 0, LEAST_BYTES, &fields);
                    let status = placement
                        .fields
                        .iter()
                        .find(|f| f.field == PushField::ExecutionStatusAddress)
                        .expect("placed");
                    let at = start + 8 * u32::from(first.is_some());
                    assert_eq!(
                        status.loc,
                        PushLoc::Push { offset: at },
                        "{stage:?} {first:?}"
                    );
                    assert!(at + status.size <= end);
                    assert_eq!(placement.argument_table_size > 0, spills, "{stage:?}");
                }
            }
        }
    }

    /// A vertex function's index address is a hole the consumer fills, which must be in the push
    /// block: the placement order puts it right after the two other addresses, so the vertex
    /// window holds it whatever else the function reads, and its layout follows the window's
    /// other fields.
    #[test]
    fn the_index_address_is_always_placed_in_the_vertex_window() {
        let mut later = BTreeSet::from([
            PushField::AmplificationId,
            PushField::AmplificationLayerOffset,
            PushField::AmplificationViewportOffset,
            PushField::BaseVertexInstance,
            PushField::CaptureLayout,
        ]);
        for i in 0..31 {
            later.insert(PushField::SlotBindOffset(slot(Stage::Vertex, i)));
            later.insert(PushField::SlotGuestBase(slot(Stage::Vertex, i)));
            later.insert(PushField::VertexBufferStride(i));
        }
        let (start, end) = window(Stage::Vertex, 0, LEAST_BYTES);
        for (root, status) in [(false, false), (true, false), (false, true), (true, true)] {
            let mut fields = later.clone();
            fields.insert(PushField::CaptureIndexAddress);
            if root {
                fields.insert(PushField::RootTableAddress);
            }
            if status {
                fields.insert(PushField::ExecutionStatusAddress);
            }
            let placement = check_in(Stage::Vertex, 0, LEAST_BYTES, &fields);
            let at = start + 8 * (u32::from(root) + u32::from(status));
            let index = placement
                .fields
                .iter()
                .find(|f| f.field == PushField::CaptureIndexAddress)
                .expect("placed");
            assert_eq!(index.loc, PushLoc::Push { offset: at }, "{root} {status}");
            assert!(at + index.size <= end);
            let layout = placement
                .fields
                .iter()
                .find(|f| f.field == PushField::CaptureLayout)
                .expect("placed");
            assert_eq!(layout.size, 16);
        }
    }
}
