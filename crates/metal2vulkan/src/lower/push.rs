//! Push placement: the complete algorithm. Families only contribute fields.
//!
//! Each stage has a push window: vertex `[0, 64)`, fragment `[64, 128)`, compute `[0, 128)`.
//! Vulkan guarantees `maxPushConstantsSize ≥ 128`, so those windows depend neither on the other
//! stage nor on the device. On a profile whose push window is the new runtime's floor
//! (`PushWindow::Floor`, `Device::from_queries`'), which requires `maxPushConstantsSize ≥ 256`, a
//! compute window is `[0, 256)` (a dispatch region and 28 buffer addresses), and a render
//! pipeline's two stages share one range `[0, 256)`, dense by link: the vertex function's fields
//! from 0, the fragment function's from its `LoweringInputs::push_base`, the vertex function's push
//! end rounded up to 8, so only the fields the pair places take space. The vertex function's
//! window there is `[0, 248)`, 8 bytes short of the range's end ([`VERTEX_FLOOR_END`]): a vertex
//! function is lowered without knowing its fragment function, and a fragment function whose
//! fields spill must push its parameter block's address (8 bytes, 8-aligned) from the vertex
//! function's end, so the vertex function leaves that room. The reserved 8 bytes cost only a vertex
//! function whose own fields pass 248, which spills where it would have filled them.
//! `select` places the function's own [`PushField`]s in the fixed kind order (the derived `Ord` of [`PushField`]; within a kind, by slot). Fields fill the
//! stage window in that order, each at the next offset aligned to the field; from the first field
//! that does not fit, every field goes to the stage's `ArgumentTable` uniform buffer
//! (`ResourceSlot{stage, Synthetic(ArgumentTable)}`), at offsets fixed by the same order. The
//! placement depends only on the function's own field set, so each field has one access form,
//! emitted once, and link never changes it. Under the identity domains the `ArgumentTable` is no
//! descriptor but a parameter block at a device address, which the function pushes first
//! (`PushField::ArgumentTableAddress`) whenever its fields spill, and reads its spilled fields
//! from at the same offsets.
//!
//! Field shapes are chosen so that the `ArgumentTable`'s std140 and std430 layouts agree (scalars,
//! 2-vectors, and 16-byte-strided arrays only), so the table is a valid uniform block on every
//! profile without `uniformBufferStandardLayout`.

use std::collections::BTreeSet;

use crate::device::profile::PushWindow;
use crate::pipeline::interface::{PlacedField, PushLoc, PushPlacement};
use crate::pipeline::sym::{PushField, Stage};

/// The byte range `[start, end)` of `stage`'s push window on a profile whose push window is `w`;
/// `base` is a fragment function's `LoweringInputs::push_base`, which only the floor reads.
pub const fn window(stage: Stage, w: PushWindow, base: u32) -> (u32, u32) {
    match stage {
        Stage::Vertex => match w {
            PushWindow::Guaranteed => (0, 64),
            PushWindow::Floor => (0, VERTEX_FLOOR_END),
        },
        Stage::Fragment => match w {
            PushWindow::Guaranteed => (64, 128),
            PushWindow::Floor => (base, FLOOR_BYTES),
        },
        Stage::Kernel => match w {
            PushWindow::Guaranteed => (0, 128),
            PushWindow::Floor => (0, FLOOR_BYTES),
        },
        // A mesh render pipeline's object, mesh and fragment functions share the 128 bytes: the
        // fragment function keeps its window, and the vertex function's is split.
        Stage::Object => (0, 32),
        Stage::Mesh => (32, 64),
    }
}

/// The push bytes of the new runtime's floor (`PushWindow::Floor`): its `maxPushConstantsSize`.
pub const FLOOR_BYTES: u32 = 256;

/// The end of a vertex function's window on the floor: the floor's end less one address, so the
/// fragment function's window, from the vertex function's end rounded up to 8, always holds its
/// parameter block's address (`PushField::ArgumentTableAddress`, which the order places first).
pub const VERTEX_FLOOR_END: u32 = FLOOR_BYTES - 8;

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
        | PushField::TessellationFactorAddress => FieldShape::U64,
        // A Metal address, low word first: the same eight bytes as a `u64`, readable without
        // `shaderInt64`.
        PushField::SlotGuestBase(_) => FieldShape::UVec2,
        // Region base `xyz` and grid `xyz`, one 4-vector each (`w` unused); a stage-in region's
        // origin and size the same way.
        PushField::DispatchRegion | PushField::StageInRegion => FieldShape::UVec4Array(2),
        // A draw's viewport compensation, four `f32`s' bits.
        PushField::ViewportCompensation => FieldShape::UVec4Array(1),
        // A draw's point guard band, two `f32`s' bits.
        PushField::PointGuard => FieldShape::UVec2,
        PushField::DepthClampRange => FieldShape::UVec2,
        // A draw's mirrored viewports, one bit each.
        PushField::ViewportMirrors => FieldShape::U32,
        // Which of a snapshot variant's draws runs.
        PushField::SnapshotPhase => FieldShape::U32,
        // The alpha test reference's code.
        PushField::AlphaTestReference => FieldShape::U32,
        // A draw's clamp of each vertex's depth, two `f32`s' bits.
        PushField::VertexDepthClamp => FieldShape::UVec2,
        // The blend color and one minus it, four `f32`s' bits each.
        // The blend color as the encoder set it, four `f32`s' bits.
        PushField::HalfBlendColor => FieldShape::UVec4Array(1),
        // A draw's depth bias: its constant, slope and clamp as `f32`s' bits, and a zero word.
        // Base vertex, base instance.
        PushField::BaseVertexInstance => FieldShape::UVec2,
        // The draw's vertices per primitive.
        PushField::VerticesPerPrimitive => FieldShape::U32,
        // The front face's word, then the back face's.
        PushField::ShadowStencilState => FieldShape::UVec2,
        // The depth word, viewport 0's near and far, and a zero word.
        PushField::ShaderTestsState => FieldShape::UVec4Array(1),
        // The faces' stencil words, the references, and a zero word.
        PushField::ShaderStencilState => FieldShape::UVec4Array(1),
        // A zero word, the index count, the first instance, and the base vertex.
        PushField::CaptureLayout => FieldShape::UVec4Array(1),
        // Four bits of kept channels per color attachment.
        PushField::StorageKeptChannels => FieldShape::U32,
        // A specialization key's value carried as data: every such key is a `u32` or a boolean.
        PushField::Specialized(_) => FieldShape::U32,
        // A constexpr sampler's index in its use's sampler array, and the sampled-texture array's
        // element a fetched color attachment is read through.
        PushField::ConstexprSampler(_) | PushField::FetchedColor { .. } => FieldShape::U32,
        // An in-place YCbCr texture's size, planes and conversion.
        PushField::YCbCrInPlace(_) => FieldShape::UVec4Array(2),
        // A texture's size, rows and first texel in its texel buffer view.
        PushField::TexelRows(_) => FieldShape::UVec4Array(1),
        // The residual is below `min{Storage,Uniform}BufferOffsetAlignment`.
        PushField::SlotBindOffset(_) | PushField::VertexBufferStride(_) => FieldShape::U32,
        // The bytes from a slot's bind offset to its buffer's end, at most 2 GiB under the
        // identity domains.
        PushField::SlotRemaining(_) => FieldShape::U32,
        // The identity domains' zero range.
        PushField::ZeroAddress => FieldShape::U64,
        // A texture's mipmap level count, an acceleration structure's instance count, the render
        // encoder's viewport count, the render pass's array length, a kernel stage-in buffer's
        // bound stride, a storage texture's integer channels' widths.
        PushField::TextureLevels(_)
        | PushField::TextureIntegerChannels(_)
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
        // An emulated tessellation draw's chunk.
        PushField::TessellationChunk => FieldShape::U32,
        // The row of regions a recording of a rate-mapped pass's regions draws: its first region's
        // segments of x and y.
        PushField::RegionBox => FieldShape::UVec2,
        // A band's viewport half extent and half width, four `f32`s.
        PushField::LineBand => FieldShape::UVec4Array(1),
        // A fan-out probe's flag index.
        PushField::RegionProbe => FieldShape::U32,
        // The whole indices from the bound index offset to the buffer's end, unsaturated:
        // `firstIndex + indexCount` reaches `2^33`.
        PushField::IndexCapacity => FieldShape::U64,
    }
}

fn align_up(v: u32, a: u32) -> u32 {
    v.div_ceil(a).saturating_mul(a)
}

/// Place `fields` for `stage` in its push window on a profile whose push window is `w`, from a
/// fragment function's `base` there ([`window`]).
pub fn place(
    stage: Stage,
    w: PushWindow,
    base: u32,
    fields: &BTreeSet<PushField>,
) -> PushPlacement {
    let (start, end) = window(stage, w, base);
    let mut cursor = start;
    let mut table = 0u32;
    let mut spilled = false;
    let mut placed = Vec::with_capacity(fields.len());
    // `BTreeSet` iterates in the derived `Ord`: exactly the kind order, then slot order.
    for &field in fields {
        let sh = shape(field);
        let at = align_up(cursor, sh.align());
        let loc = if !spilled && at.saturating_add(sh.size()) <= end {
            cursor = at + sh.size();
            PushLoc::Push { offset: at }
        } else {
            spilled = true;
            let off = align_up(table, sh.align());
            table = off.saturating_add(sh.size());
            PushLoc::ArgumentTable { offset: off }
        };
        placed.push(PlacedField {
            field,
            loc,
            size: sh.size(),
        });
    }
    PushPlacement {
        stage,
        push_range: (cursor > start).then_some((start, cursor - start)),
        argument_table_size: align_up(table, 16),
        fields: placed,
    }
}

/// Whether `stage`'s placement of `fields` spills into the `ArgumentTable`: whether [`place`]
/// gives it a size. That is the first field with a size that does not fit, so the fields are read
/// only up to it, at most the window's worth plus one, however many there are; a step that asks
/// once per slot it moves (`access::finish_slots`) then costs no walk of every slot's field.
pub fn spills(stage: Stage, w: PushWindow, base: u32, fields: &BTreeSet<PushField>) -> bool {
    let (start, end) = window(stage, w, base);
    let mut cursor = start;
    let mut spilled = false;
    for &field in fields {
        let sh = shape(field);
        let at = align_up(cursor, sh.align());
        if !spilled && at.saturating_add(sh.size()) <= end {
            cursor = at + sh.size();
        } else if sh.size() > 0 {
            return true;
        } else {
            spilled = true;
        }
    }
    false
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
    /// spill), and a push range and table size that cover their fields.
    fn check(stage: Stage, fields: &BTreeSet<PushField>) -> PushPlacement {
        check_in(stage, PushWindow::Guaranteed, fields)
    }

    /// [`check`] on a profile whose push window is `w`.
    fn check_in(stage: Stage, w: PushWindow, fields: &BTreeSet<PushField>) -> PushPlacement {
        check_at(stage, w, 0, fields)
    }

    /// [`check_in`] from a fragment function's push `base`.
    fn check_at(
        stage: Stage,
        w: PushWindow,
        base: u32,
        fields: &BTreeSet<PushField>,
    ) -> PushPlacement {
        let p = place(stage, w, base, fields);
        assert_eq!(spills(stage, w, base, fields), p.argument_table_size > 0);
        let (start, end) = window(stage, w, base);
        assert_eq!(p.stage, stage);
        let order: Vec<PushField> = p.fields.iter().map(|f| f.field).collect();
        assert_eq!(order, fields.iter().copied().collect::<Vec<_>>());
        let (mut push_end, mut table_end, mut spilled) = (start, 0u32, false);
        for f in &p.fields {
            let sh = shape(f.field);
            assert_eq!(f.size, sh.size());
            match f.loc {
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

    #[test]
    fn one_bind_offset_opens_the_stage_window() {
        for (stage, base) in [
            (Stage::Vertex, 0),
            (Stage::Fragment, 64),
            (Stage::Kernel, 0),
        ] {
            let fields = [PushField::SlotBindOffset(slot(stage, 0))].into();
            let p = check(stage, &fields);
            assert_eq!(p.fields[0].loc, PushLoc::Push { offset: base });
            assert_eq!(p.push_range, Some((base, 4)));
        }
    }

    /// Fields arrive in any order and are placed in kind order, each aligned: a `u32` residual,
    /// then an 8-aligned guest base after 4 bytes of padding.
    #[test]
    fn fields_are_placed_in_kind_order_with_alignment() {
        let s = slot(Stage::Kernel, 3);
        let fields = [
            PushField::SlotGuestBase(s),
            PushField::VertexBufferStride(0),
            PushField::SlotBindOffset(s),
            PushField::DispatchRegion,
        ]
        .into();
        let p = check(Stage::Kernel, &fields);
        let locs: Vec<(PushField, PushLoc)> = p.fields.iter().map(|f| (f.field, f.loc)).collect();
        assert_eq!(
            locs,
            vec![
                (PushField::DispatchRegion, PushLoc::Push { offset: 0 }),
                (PushField::SlotBindOffset(s), PushLoc::Push { offset: 32 }),
                (PushField::SlotGuestBase(s), PushLoc::Push { offset: 40 }),
                (
                    PushField::VertexBufferStride(0),
                    PushLoc::Push { offset: 48 }
                ),
            ]
        );
        assert_eq!(p.push_range, Some((0, 52)));
    }

    /// 31 residuals fill 124 bytes of the compute window; the first guest base would end past
    /// 128, so it and every later field go to the `ArgumentTable`.
    #[test]
    fn the_first_field_that_does_not_fit_spills_it_and_the_rest() {
        let mut fields = BTreeSet::new();
        for i in 0..31 {
            fields.insert(PushField::SlotBindOffset(slot(Stage::Kernel, i)));
            fields.insert(PushField::SlotGuestBase(slot(Stage::Kernel, i)));
        }
        let p = check(Stage::Kernel, &fields);
        let pushed = p
            .fields
            .iter()
            .filter(|f| matches!(f.loc, PushLoc::Push { .. }))
            .count();
        assert_eq!(pushed, 31);
        assert_eq!(p.push_range, Some((0, 124)));
        assert_eq!(p.argument_table_size, 31 * 8 + 8);
        assert_eq!(
            p.fields[31].loc,
            PushLoc::ArgumentTable { offset: 0 },
            "the first spilled field starts the table"
        );
    }

    /// A kernel's dispatch region and its buffers' addresses: 12 addresses fill the guaranteed
    /// window and 28 the floor's, which places the first 12 as the guaranteed one does, and a 29th
    /// spills; with a spill, the table's address and the region's alignment take the first 16
    /// bytes, and 26 addresses stay in the window. The render windows are the same on both.
    #[test]
    fn the_floor_window_holds_a_region_and_twenty_eight_addresses() {
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
        let guaranteed = check_in(Stage::Kernel, PushWindow::Guaranteed, &fields(12));
        assert_eq!(
            (guaranteed.push_range, guaranteed.argument_table_size),
            (Some((0, 128)), 0)
        );
        assert!(spills(
            Stage::Kernel,
            PushWindow::Guaranteed,
            0,
            &fields(13)
        ));
        let floor = check_in(Stage::Kernel, PushWindow::Floor, &fields(28));
        assert_eq!(
            (floor.push_range, floor.argument_table_size),
            (Some((0, 256)), 0)
        );
        assert_eq!(pushed(&floor), 29);
        let same = check_in(Stage::Kernel, PushWindow::Floor, &fields(12));
        assert_eq!(
            same, guaranteed,
            "within 128 bytes the two windows place alike"
        );
        assert!(spills(Stage::Kernel, PushWindow::Floor, 0, &fields(29)));
        let mut spilled = fields(29);
        spilled.insert(PushField::ArgumentTableAddress);
        let p = check_in(Stage::Kernel, PushWindow::Floor, &spilled);
        assert_eq!((p.push_range, p.argument_table_size), (Some((0, 256)), 32));
        assert_eq!(
            pushed(&p),
            28,
            "the address, the region and 26 buffer addresses"
        );
        for stage in [Stage::Object, Stage::Mesh] {
            assert_eq!(
                window(stage, PushWindow::Floor, 0),
                window(stage, PushWindow::Guaranteed, 0)
            );
        }
    }

    /// The windows a pipeline of each kind pushes, pinned: on the guaranteed window the fixed
    /// ones, whatever a fragment function's base; on the floor a compute kernel's `[0, 256)`, and a
    /// render pipeline's one range `[0, 256)`, the vertex function's from 0 and the fragment
    /// function's from its base. A compute kernel's window does not move with a render base.
    #[test]
    fn each_kind_pushes_its_own_window() {
        for base in [0, 8, 40, 248] {
            assert_eq!(window(Stage::Vertex, PushWindow::Guaranteed, base), (0, 64));
            assert_eq!(
                window(Stage::Fragment, PushWindow::Guaranteed, base),
                (64, 128)
            );
            assert_eq!(
                window(Stage::Kernel, PushWindow::Guaranteed, base),
                (0, 128)
            );
            assert_eq!(window(Stage::Kernel, PushWindow::Floor, base), (0, 256));
            assert_eq!(window(Stage::Vertex, PushWindow::Floor, base), (0, 248));
            assert_eq!(
                window(Stage::Fragment, PushWindow::Floor, base),
                (base, 256)
            );
        }
    }

    /// On the floor a render pipeline's fields are dense by link: a fragment function placed from
    /// the vertex function's push end rounded up to 8 starts there, its fields fill the rest of
    /// the 256 bytes, and from the first that does not fit it spills. A vertex function's window
    /// ends at 248, so a fragment function from its end can always push its block's address.
    #[test]
    fn the_floor_places_a_fragment_function_from_its_base() {
        let vertex: BTreeSet<PushField> = (0..3)
            .map(|i| PushField::SlotGuestBase(slot(Stage::Vertex, i)))
            .collect();
        let v = check_at(Stage::Vertex, PushWindow::Floor, 0, &vertex);
        assert_eq!(v.push_range, Some((0, 24)));
        let fragment = |n: u32| -> BTreeSet<PushField> {
            (0..n)
                .map(|i| PushField::SlotGuestBase(slot(Stage::Fragment, i)))
                .collect()
        };
        let f = check_at(Stage::Fragment, PushWindow::Floor, 24, &fragment(2));
        assert_eq!(f.push_range, Some((24, 16)));
        let full = check_at(Stage::Fragment, PushWindow::Floor, 24, &fragment(29));
        assert_eq!(full.push_range, Some((24, 232)), "to the floor's end");
        assert_eq!(full.argument_table_size, 0);
        assert!(spills(
            Stage::Fragment,
            PushWindow::Floor,
            24,
            &fragment(30)
        ));
        // A vertex function's window ends at 248: 31 addresses fill it, and a 32nd spills.
        let wide = |n: u32| -> BTreeSet<PushField> {
            (0..n)
                .map(|i| PushField::SlotGuestBase(slot(Stage::Vertex, i)))
                .collect()
        };
        let v = check_at(Stage::Vertex, PushWindow::Floor, 0, &wide(31));
        assert_eq!((v.push_range, v.argument_table_size), (Some((0, 248)), 0));
        assert!(spills(Stage::Vertex, PushWindow::Floor, 0, &wide(32)));
        // Fields that would reach 252, 4 bytes into what was the fragment function's, spill.
        let mut past = wide(31);
        past.insert(PushField::SlotRemaining(slot(Stage::Vertex, 0)));
        assert!(spills(Stage::Vertex, PushWindow::Floor, 0, &past));
        // The fragment function from the fullest vertex function's end pushes its block's
        // address there, at 248, and spills the rest; from an end of 244 (a base rounded to 4)
        // the address, 8-aligned, is still at 248.
        let mut spilled = fragment(2);
        spilled.insert(PushField::ArgumentTableAddress);
        for (base, range) in [(248, (248, 8)), (244, (244, 12))] {
            let f = check_at(Stage::Fragment, PushWindow::Floor, base, &spilled);
            assert_eq!((f.push_range, f.argument_table_size), (Some(range), 16));
            assert_eq!(
                (f.fields[0].field, f.fields[0].loc),
                (
                    PushField::ArgumentTableAddress,
                    PushLoc::Push { offset: 248 }
                ),
                "from {base}"
            );
        }
    }

    /// The vertex window holds 16 residuals; the 17th spills even though a later, smaller field
    /// would still fit nowhere else: placement is a prefix of the order.
    #[test]
    fn the_vertex_window_holds_sixty_four_bytes() {
        let mut fields: BTreeSet<PushField> = (0..17)
            .map(|i| PushField::SlotBindOffset(slot(Stage::Vertex, i)))
            .collect();
        fields.insert(PushField::VertexBufferStride(0));
        let p = check(Stage::Vertex, &fields);
        assert_eq!(p.push_range, Some((0, 64)));
        assert_eq!(p.fields[16].loc, PushLoc::ArgumentTable { offset: 0 });
        assert_eq!(p.fields[17].loc, PushLoc::ArgumentTable { offset: 4 });
        assert_eq!(p.argument_table_size, 16);
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
                PushField::Specialized(crate::pipeline::sym::SpecKey::TexelFormat(SlotRef {
                    stage,
                    class: SlotClass::Texture,
                    index: i,
                })),
            ]
        };
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let windows = [PushWindow::Guaranteed, PushWindow::Floor];
        for (stage, w) in [Stage::Vertex, Stage::Fragment, Stage::Kernel]
            .into_iter()
            .flat_map(|stage| windows.map(|w| (stage, w)))
        {
            for _ in 0..200 {
                let mut fields = BTreeSet::new();
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let n = (state % 40) as u32;
                for i in 0..n {
                    let pick = ((state >> (i % 50)) % 8) as usize;
                    fields.insert(kinds(stage, i % 31)[pick]);
                }
                check_in(stage, w, &fields);
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
            let (start, end) = window(stage, PushWindow::Guaranteed, 0);
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
                    let placement = place(stage, PushWindow::Guaranteed, 0, &fields);
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
            PushField::ViewportCompensation,
            PushField::CaptureLayout,
        ]);
        for i in 0..31 {
            later.insert(PushField::SlotBindOffset(slot(Stage::Vertex, i)));
            later.insert(PushField::SlotGuestBase(slot(Stage::Vertex, i)));
            later.insert(PushField::VertexBufferStride(i));
        }
        let (start, end) = window(Stage::Vertex, PushWindow::Guaranteed, 0);
        for (root, status) in [(false, false), (true, false), (false, true), (true, true)] {
            let mut fields = later.clone();
            fields.insert(PushField::CaptureIndexAddress);
            if root {
                fields.insert(PushField::RootTableAddress);
            }
            if status {
                fields.insert(PushField::ExecutionStatusAddress);
            }
            let placement = check(Stage::Vertex, &fields);
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
