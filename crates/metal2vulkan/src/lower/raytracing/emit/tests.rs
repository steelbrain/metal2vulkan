//! The settings' ray flags and the ray check on the CPU interpreter: every combination of Metal's
//! winding, triangle cull, geometry cull, opacity cull, forced opacity, and accept-any, and rays at
//! the edges of Vulkan's validity rules.
//!
//! A compute harness reads the settings (or the ray) from buffer 0 and writes what [`flags`]
//! (or [`ray_valid`]) computes to buffer 1. The traversal itself cannot run here: the interpreter
//! has no acceleration structures and reports every ray query instruction unmodelled.

use super::*;
use crate::memory::views::{self, View};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::interp::{self, Buffer, Config, Resources};
use crate::spirv::ir::{ConvOp, ExecMode, ExecModel, MemAccess, TypedModule};

fn kslot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

/// Word `i` of `view`.
fn word(fb: &mut FunctionBuilder<'_>, view: View, i: u32) -> SValue {
    let w = u32c(fb, i).expect("index");
    views::read_word(fb, view, w).expect("read")
}

/// Store `x` (a `u32`) to word `i` of `view`.
fn put(fb: &mut FunctionBuilder<'_>, view: View, i: u32, x: SValue) {
    let w = u32c(fb, i).expect("index");
    let p = views::word_ptr(fb, view, w).expect("pointer");
    fb.store(p, x, MemAccess::default()).expect("store");
}

/// `c ? 1 : 0`.
fn bit(fb: &mut FunctionBuilder<'_>, c: SValue) -> SValue {
    bit_if(fb, c, 1).expect("select")
}

/// A one-invocation compute module around `body`, reading buffer 0 and writing buffer 1.
fn harness(body: impl FnOnce(&mut FunctionBuilder<'_>, View, View)) -> TypedModule {
    let mut mb = ModuleBuilder::new();
    let input = views::declare_at(&mut mb, kslot(0)).expect("input");
    let out = views::declare_at(&mut mb, kslot(1)).expect("out");
    let void = mb.ty(SType::Void).expect("void");
    let mut f = mb.function(void).expect("function");
    body(&mut f, input, out);
    f.ret(None).expect("ret");
    let id = f.finish().expect("finish");
    mb.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .expect("entry");
    mb.finalize().expect("finalize")
}

/// Run `m` over input words `input`; the first `n` output words.
fn run(m: &TypedModule, input: &[u32], n: usize) -> Vec<u32> {
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(kslot(0)),
                address: None,
                bytes: input.iter().flat_map(|w| w.to_le_bytes()).collect(),
            },
            Buffer {
                slot: Some(kslot(1)),
                address: None,
                bytes: vec![0; n * 4],
            },
        ],
        push: Vec::new(),
        spec: Default::default(),
    };
    let o = interp::run(m, &Config::default(), res).expect("run");
    o.buffers
        .get(1)
        .expect("output")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The flags Metal's settings map to, and whether they cull everything, written independently of
/// the emitter.
fn reference(
    winding: u32,
    tcull: u32,
    gcull: u32,
    ocull: u32,
    forced: u32,
    any: bool,
) -> (u32, bool) {
    let ccw = winding == 1;
    let (front, back) = match tcull {
        1 => (true, false),
        2 => (false, true),
        _ => (false, false),
    };
    // Clockwise is Vulkan's front face; counterclockwise swaps the faces.
    let (vk_front, vk_back) = if ccw { (back, front) } else { (front, back) };
    let skip_tri = gcull & 1 != 0;
    let skip_box = gcull & 2 != 0;
    let mut word = 0;
    if !skip_tri {
        word |= if vk_front { CULL_FRONT } else { 0 } | if vk_back { CULL_BACK } else { 0 };
    }
    word |= if skip_tri { SKIP_TRIANGLES } else { 0 } | if skip_box { SKIP_AABBS } else { 0 };
    word |= match (forced, ocull) {
        (1, _) => OPAQUE,
        (2, _) => NO_OPAQUE,
        (_, 1) => CULL_OPAQUE,
        (_, 2) => CULL_NO_OPAQUE,
        _ => 0,
    };
    if any {
        word |= TERMINATE_ON_FIRST_HIT;
    }
    let cull_all =
        (skip_tri && skip_box) || (forced == 1 && ocull == 1) || (forced == 2 && ocull == 2);
    (word, cull_all)
}

/// Every combination of the settings: the flags of a traversal match [`reference`] and satisfy
/// VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06889 (not both skips), -06890 (at most one of
/// the triangle skip and the two facing culls), and -06891 (at most one opacity flag); a
/// combination that culls everything is found so and never traverses.
#[test]
fn every_setting_combination_maps_to_valid_flags() {
    let m = harness(|fb, input, out| {
        let s = Settings {
            winding: word(fb, input, 0),
            triangle_cull: word(fb, input, 1),
            geometry_cull: word(fb, input, 2),
            opacity_cull: word(fb, input, 3),
            forced_opacity: word(fb, input, 4),
            accept_any: {
                let a = word(fb, input, 5);
                let zero = u32c(fb, 0).expect("zero");
                fb.binary(BinOp::INotEqual, a, zero).expect("any")
            },
        };
        let (flags, cull_all, ccw) = flags(fb, &s).expect("flags");
        put(fb, out, 0, flags);
        let c = bit(fb, cull_all);
        put(fb, out, 1, c);
        let w = bit(fb, ccw);
        put(fb, out, 2, w);
    });
    let mut traced = 0;
    for winding in 0..2 {
        for tcull in 0..3 {
            for gcull in 0..8 {
                for ocull in 0..3 {
                    for forced in 0..3 {
                        for any in [false, true] {
                            let input = [winding, tcull, gcull, ocull, forced, u32::from(any)];
                            let got = run(&m, &input, 3);
                            let (want, cull_all) =
                                reference(winding, tcull, gcull, ocull, forced, any);
                            let case = format!("{input:?}: {got:x?}");
                            assert_eq!(got[1] == 1, cull_all, "{case}");
                            assert_eq!(got[2] == 1, winding == 1, "{case}");
                            if cull_all {
                                continue;
                            }
                            traced += 1;
                            let flags = got[0];
                            assert_eq!(flags, want, "{case}");
                            assert!(
                                flags & (SKIP_TRIANGLES | SKIP_AABBS)
                                    != SKIP_TRIANGLES | SKIP_AABBS,
                                "06889 {case}"
                            );
                            assert!(
                                (flags & (SKIP_TRIANGLES | CULL_FRONT | CULL_BACK)).count_ones()
                                    <= 1,
                                "06890 {case}"
                            );
                            assert!(
                                (flags & (OPAQUE | NO_OPAQUE | CULL_OPAQUE | CULL_NO_OPAQUE))
                                    .count_ones()
                                    <= 1,
                                "06891 {case}"
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(traced > 0);
}

/// The ray check at the edges of VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06348 to -06351: a
/// finite origin and direction, `0 <= min_distance <= max_distance`, no NaN; an infinite
/// maximum distance is valid.
#[test]
fn rays_outside_vulkans_rules_are_invalid() {
    let m = harness(|fb, input, out| {
        let f32_ = fb.module().ty(SType::Float { width: 32 }).expect("f32");
        let vec3 = fb
            .module()
            .ty(SType::Vector { elem: f32_, len: 3 })
            .expect("vec3");
        let mut floats = Vec::new();
        for i in 0..8 {
            let w = word(fb, input, i);
            floats.push(fb.convert(ConvOp::Bitcast, w, f32_).expect("float"));
        }
        let origin = fb.composite_construct(vec3, &floats[0..3]).expect("origin");
        let direction = fb
            .composite_construct(vec3, &floats[3..6])
            .expect("direction");
        let valid = ray_valid(fb, origin, direction, floats[6], floats[7]).expect("valid");
        let v = bit(fb, valid);
        put(fb, out, 0, v);
    });
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    let cases: [([f32; 8], bool); 12] = [
        ([0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 100.0], true),
        ([0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, inf], true),
        ([0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 5.0, 5.0], true),
        ([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0], true),
        ([0.0, 0.0, -1.0, 0.0, 0.0, 1.0, -0.0, 1.0], true),
        ([inf, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 1.0], false),
        ([0.0, 0.0, -1.0, 0.0, -inf, 1.0, 0.0, 1.0], false),
        ([0.0, nan, -1.0, 0.0, 0.0, 1.0, 0.0, 1.0], false),
        ([0.0, 0.0, -1.0, 0.0, 0.0, nan, 0.0, 1.0], false),
        ([0.0, 0.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0], false),
        ([0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 2.0, 1.0], false),
        ([0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, nan], false),
    ];
    for (ray, want) in cases {
        let input = ray.map(f32::to_bits);
        assert_eq!(run(&m, &input, 1), [u32::from(want)], "{ray:?}");
    }
    let m_nan_min = [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, nan, 1.0].map(f32::to_bits);
    assert_eq!(run(&m, &m_nan_min, 1), [0], "NaN min_distance");
}

/// The table a candidate is checked against reads the words the consumer's encoder
/// (`bind::table::FunctionTableWords`) writes: the table starts at word 2 of buffer 0, after the
/// candidate's record offset (word 0) and a word that would accept every candidate if read as an
/// entry (word 1), so a read of any other word than the present word, the size word, or entry
/// `index` changes an answer. A triangle is accepted exactly when the table is unbound or its
/// entry is the opaque triangle function; a null entry, a function's word, an index at or past
/// the size, and an entry past the bound range (read as null, never out of bounds) reject it.
#[test]
fn a_table_accepts_a_triangle_at_its_opaque_entries_or_unbound() {
    use crate::bind::table::FunctionTableWords;
    use crate::pipeline::table::FunctionHandle;
    const BASE: u32 = 2;
    let m = harness(|fb, input, out| {
        let index = word(fb, input, 0);
        let base = u32c(fb, BASE).expect("base");
        let accepts = entry_accepts(fb, (input, base), index).expect("accepts");
        let v = bit(fb, accepts);
        put(fb, out, 0, v);
    });
    let buffer = |index: u32, table: &[u32]| -> Vec<u32> {
        let mut words = vec![index, FunctionHandle::RESERVED_WORD];
        words.extend_from_slice(table);
        words
    };
    // Entries: null, the opaque triangle, a function's word, the opaque triangle.
    let mut encoded = FunctionTableWords::new(4);
    assert!(encoded.set_opaque_triangle(1));
    assert!(encoded.set_opaque_triangle(3));
    let mut bound = encoded.words().to_vec();
    bound[FIRST_ENTRY_WORD as usize + 2] = 7;
    for (index, want) in [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (u32::MAX, 0)] {
        assert_eq!(run(&m, &buffer(index, &bound), 1), [want], "entry {index}");
    }
    // A size past the words the buffer holds: the entries past its range are null.
    let mut short = bound.clone();
    short[SIZE_WORD as usize] = u32::MAX;
    for (index, want) in [(3, 1), (4, 0), (1000, 0), (u32::MAX - 1, 0)] {
        assert_eq!(run(&m, &buffer(index, &short), 1), [want], "short {index}");
    }
    // An unbound table (a zero-filled neutral buffer) accepts every triangle.
    let unbound = vec![0; bound.len()];
    for index in [0, 1, 4, u32::MAX] {
        assert_eq!(run(&m, &buffer(index, &unbound), 1), [1], "unbound {index}");
    }
    // A table past the buffer's end is unbound.
    assert_eq!(run(&m, &buffer(0, &[]), 1), [1], "no table words");
    // The present word alone decides: cleared, the table's other words are not read; set, an
    // empty table rejects every triangle.
    let mut cleared = bound.clone();
    cleared[PRESENT_WORD as usize] = FunctionHandle::NULL_WORD;
    for index in [0, 2, 4] {
        assert_eq!(run(&m, &buffer(index, &cleared), 1), [1], "cleared {index}");
    }
    let empty = FunctionTableWords::new(0);
    assert_eq!(run(&m, &buffer(0, empty.words()), 1), [0], "empty table");
}
