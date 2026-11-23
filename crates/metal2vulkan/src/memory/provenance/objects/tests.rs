//! The object layouts and overlap test are linear in the layout graph: a layout shared by many
//! members is visited once, and an access descends only into the elements it covers partly.

use super::*;
use crate::air::abi::{
    AbiInt, ArgTypeInfo, BufferArg, IndirectArg, ObjectArg, ParamRole, Slot, UserKey,
};
use crate::air::error::DiagName;
use crate::budget::Budget;
use crate::error::Limit;

fn member(offset: u32, size: u32, layout: Option<StructRef>, kind: MemberKind) -> Member {
    Member {
        offset,
        size,
        array_len: 0,
        layout,
        kind,
        render_target: None,
        raster_order_group: None,
        type_name: DiagName::new("t"),
        name: DiagName::new("m"),
        key: UserKey(Box::new([])),
    }
}

fn indirect(role: ParamRole) -> MemberKind {
    MemberKind::Indirect(IndirectArg {
        ordinal: 0,
        role,
        ty: ArgTypeInfo::default(),
        name: None,
    })
}

fn slot() -> Slot {
    Slot {
        index: AbiInt::Const(0),
        count: AbiInt::Const(1),
    }
}

fn pipeline(offset: u32) -> Member {
    member(
        offset,
        8,
        None,
        indirect(ParamRole::ComputePipelineState(ObjectArg {
            location: slot(),
            access: None,
        })),
    )
}

fn word(offset: u32) -> Member {
    member(offset, 4, None, MemberKind::Plain)
}

fn by_value(offset: u32, size: u32, inner: u32) -> Member {
    member(offset, size, Some(StructRef(inner)), MemberKind::Plain)
}

/// `S_k { S_{k+1} a; S_{k+1} b; }` for `k < depth`, with `b` at `b_at(size of S_{k+1})`, over
/// the leaf `{ uint x; uint y; compute_pipeline_state p; }` (16 bytes, the object at 8) or, without
/// `object`, `{ uint x; uint y; }` padded to 16.
fn shared_graph(depth: u32, b_at: impl Fn(u32) -> u32, object: bool) -> Vec<StructInfo> {
    let mut structs = Vec::new();
    for k in 0..depth {
        let size = 16u32 << (depth - k - 1).min(27);
        structs.push(StructInfo {
            members: vec![by_value(0, size, k + 1), by_value(b_at(size), size, k + 1)],
        });
    }
    let mut leaf = vec![word(0), word(4)];
    if object {
        leaf.push(pipeline(8));
    }
    structs.push(StructInfo { members: leaf });
    structs
}

fn meter(work: u64) -> Meter {
    Meter::with_caps(&Budget::default(), work, 1 << 30)
}

#[test]
fn a_shared_layout_is_classified_once() {
    // 2^60 paths from the root to the leaf; one pass over 121 members and 120 edges.
    for object in [true, false] {
        let structs = shared_graph(60, |_| 0, object);
        let mut m = meter(u64::MAX);
        let layouts = ObjectLayouts::of(&structs, false, &mut m).expect("layouts");
        assert!((0..=60).all(|s| layouts.holds(StructRef(s)) == object));
        assert!((0..=60).all(|s| !layouts.behind(StructRef(s))));
        assert!(m.work() <= 4 * 121, "work {}", m.work());
    }
}

#[test]
fn an_access_descends_only_into_the_elements_it_covers_partly() {
    // Disjoint halves: S_k is twice S_{k+1}, so element 0 of the root is 16 << 26 bytes.
    let depth = 27;
    let structs = shared_graph(depth, |size| size, true);
    let mut m = meter(u64::MAX);
    let layouts = ObjectLayouts::of(&structs, false, &mut m).expect("layouts");
    let contains = |s: StructRef| layouts.holds(s);
    let leaf = |mem: &Member| mem.handle().is_some_and(is_object);
    let size = 16u64 << depth;
    // `y` of the first leaf, the object of the first leaf, and the span from the first leaf's
    // `y` to the last leaf's `x`.
    for (range, hit) in [((4, 8), false), ((8, 12), true), ((4, size - 12), true)] {
        let before = m.work();
        let found = overlaps(&structs, &contains, &leaf, StructRef(0), range, 0, &mut m);
        assert_eq!(found.expect("overlap"), hit, "{range:?}");
        // Two members per visit, at most two partly covered paths down 28 levels.
        assert!(
            m.work() - before <= 2 * 2 * (u64::from(depth) + 1),
            "{range:?}: work {}",
            m.work() - before
        );
    }
}

#[test]
fn overlapping_members_end_at_the_work_cap() {
    // Malformed metadata: `a` and `b` both at offset 0, so a partly covered access has two
    // partly covered paths per level. The meter ends the search instead of 2^60 visits.
    let structs = shared_graph(60, |_| 0, true);
    let mut m = meter(1 << 20);
    let layouts = ObjectLayouts::of(&structs, false, &mut m).expect("layouts");
    let contains = |s: StructRef| layouts.holds(s);
    let leaf = |mem: &Member| mem.handle().is_some_and(is_object);
    let found = overlaps(&structs, &contains, &leaf, StructRef(0), (4, 8), 0, &mut m);
    assert!(matches!(found, Err(Error::LimitExceeded(Limit::WorkUnits))));
}

fn command(offset: u32) -> Member {
    member(
        offset,
        8,
        None,
        indirect(ParamRole::CommandBuffer(ObjectArg {
            location: slot(),
            access: None,
        })),
    )
}

/// A command buffer's field is opaque only where it does not hold its word: `{ command_buffer c;
/// uint w[2]; }` holds an object outside the identity domains and none under them, where an
/// access of its first eight bytes is plain data; a pipeline state's field is opaque under both.
#[test]
fn a_command_buffers_field_is_opaque_only_without_its_word() {
    let structs = vec![
        StructInfo {
            members: vec![command(0), word(8), word(12)],
        },
        StructInfo {
            members: vec![pipeline(0), word(8)],
        },
    ];
    for words in [false, true] {
        let mut m = meter(u64::MAX);
        let layouts = ObjectLayouts::of(&structs, words, &mut m).expect("layouts");
        let holds: Vec<bool> = (0..2).map(|s| layouts.holds(StructRef(s))).collect();
        assert_eq!(holds, [!words, true], "words {words}");
        let leaf = |mem: &Member| layouts.opaque(mem);
        let contains = |s: StructRef| layouts.holds(s);
        let found = overlaps(&structs, &contains, &leaf, StructRef(0), (0, 8), 0, &mut m);
        assert_eq!(found.expect("overlap"), !words, "words {words}");
    }
}

#[test]
fn an_object_behind_a_pointer_member_is_behind() {
    // Outer { Pipelines* inner; uint x; } -> Pipelines { compute_pipeline_state p; uint s; },
    // and Wrapper { Outer o; } by value.
    let pointer = member(
        0,
        8,
        None,
        indirect(ParamRole::ArgumentBuffer(BufferArg {
            location: slot(),
            access: crate::air::abi::Access::Read,
            space: crate::air::ir::AddrSpace::Constant,
            buffer_size: None,
            layout: Some(StructRef(1)),
            raster_order_group: None,
        })),
    );
    let structs = vec![
        StructInfo {
            members: vec![pointer, word(8)],
        },
        StructInfo {
            members: vec![pipeline(0), word(8)],
        },
        StructInfo {
            members: vec![by_value(0, 16, 0)],
        },
    ];
    let mut m = meter(u64::MAX);
    let layouts = ObjectLayouts::of(&structs, false, &mut m).expect("layouts");
    let holds: Vec<bool> = (0..3).map(|s| layouts.holds(StructRef(s))).collect();
    let behind: Vec<bool> = (0..3).map(|s| layouts.behind(StructRef(s))).collect();
    assert_eq!(holds, [false, true, false]);
    assert_eq!(behind, [true, false, true]);
}

/// `{ uint data[n]; compute_pipeline_state p; }` passed as a pointer argument buffer with no
/// stated size, whose elements repeat every `4n + 8` bytes.
fn data_then_object(n: u32) -> (Vec<StructInfo>, ObjectBuffer) {
    let mut data = word(0);
    data.array_len = n;
    let structs = vec![StructInfo {
        members: vec![data, pipeline(4 * n)],
    }];
    let buffer = ObjectBuffer {
        param: 0,
        layout: StructRef(0),
        first: u64::from(4 * n + 8),
        stride: Some(u64::from(4 * n + 8)),
        held: None,
        holds: true,
        behind: false,
    };
    (structs, buffer)
}

/// An access at least as wide as its step is one contiguous range, tested once: a word at any
/// offset of `data` (every byte, as a joined shape admits, or every fourth) is data, and one more
/// offset reaches the object. Each answer costs a few units, not one per offset (8,192 or more,
/// past the 4,096 the rule lists) or per residue modulo the 32,776-byte stride.
#[test]
fn an_access_as_wide_as_its_step_is_one_test() {
    let n = 8192;
    let (structs, buffer) = data_then_object(n);
    let mut m = meter(u64::MAX);
    let layouts = ObjectLayouts::of(&structs, false, &mut m).expect("layouts");
    let words = crate::memory::alloca::Strides::one(4).expect("stride");
    let data_end = 4 * i64::from(n);
    for shape in [
        Shape::Top,
        Shape::Strided {
            base: 0,
            strides: words,
        },
    ] {
        for (last, hit) in [(data_end - 4, false), (data_end, true)] {
            let fact = PtrFact {
                roots: super::super::RootSetId::EMPTY,
                range: OffsetRange::Span { lo: 0, hi: last },
                shape,
                align: 0,
            };
            let before = m.work();
            let touched =
                buffer.touches_object(&structs, &layouts, fact, Some(Reach::at(4)), &mut m);
            assert_eq!(touched.expect("touches"), hit, "{shape:?} {last}");
            assert!(
                m.work() - before <= 8,
                "{shape:?} {last}: work {}",
                m.work() - before
            );
        }
    }
}
