//! The continuation combine module: valid on every committed profile (L3, `G-val`), and exact
//! on the CPU interpreter (each result's word ORed with its continuation words half by half, or
//! summed with them as 64-bit integers), in place, for the listed words only, across workgroups
//! and short ranges; and its plan's windows.

use super::*;
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Run the combine of `table` over `dest` and `sources` (each bound whole) in `groups`
/// workgroups, in both invocation orders; the destination's words after it, the sources
/// unchanged.
fn combine(table: &[u32], dest: &[u32], sources: &[u32], groups: u32) -> Vec<u32> {
    let m = typed_module().expect("module");
    let mut out = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let res = Resources {
            buffers: vec![
                Buffer {
                    slot: Some(TABLE),
                    address: None,
                    bytes: bytes(table),
                },
                Buffer {
                    slot: Some(DEST),
                    address: None,
                    bytes: bytes(dest),
                },
                Buffer {
                    slot: Some(SOURCES),
                    address: None,
                    bytes: bytes(sources),
                },
            ],
            push: Vec::new(),
            spec: Default::default(),
        };
        let cfg = Config {
            order,
            workgroups: [groups, 1, 1],
            ..Config::default()
        };
        out.push(run(&m, &cfg, res).expect("run").buffers);
    }
    assert_eq!(out[0], out[1], "order independent");
    assert_eq!(words(&out[0][2]), sources, "the sources are only read");
    words(&out[0][1])
}

#[test]
fn the_combine_module_is_valid_on_every_committed_profile() {
    for (_, p) in committed_profiles().expect("profiles") {
        assert_valid(typed_module().expect("module"), &p);
    }
}

/// A result's word becomes the OR of its count and its continuations', half by half: nonzero
/// exactly when any of them is, whatever counts an imprecise query gives. Words not listed keep
/// their bytes.
#[test]
fn continuations_are_ored_into_their_result_in_place() {
    // Sources: four 64-bit continuation words.
    let sources = [0, 0, 4, 0, 0, 0x8000_0000, 0, 0];
    for (result, first, count, want) in [
        ([0, 0], 0, 1, [0, 0]),
        ([0, 0], 0, 2, [4, 0]),
        ([0, 0], 2, 2, [0, 0x8000_0000]),
        ([1, 0], 3, 1, [1, 0]),
        ([2, 0], 1, 3, [6, 0x8000_0000]),
        ([0, 0], 1, 0, [0, 0]),
    ] {
        let dest = [0xAAAA_AAAA, 0xAAAA_AAAA, result[0], result[1], 7, 9];
        let out = combine(&[1, first, count, 0], &dest, &sources, 1);
        assert_eq!(
            out,
            [0xAAAA_AAAA, 0xAAAA_AAAA, want[0], want[1], 7, 9],
            "{result:?} with {count} from {first}"
        );
    }
}

/// A `Counting` result's word becomes the 64-bit sum of its count and its continuations',
/// carrying from the low half into the high one and wrapping past 2^64.
#[test]
fn counting_continuations_are_summed_into_their_result() {
    // Sources: 16, 2^32 - 1, 1 and 2^64 - 1, as 64-bit words.
    let sources = [16, 0, u32::MAX, 0, 1, 0, u32::MAX, u32::MAX];
    for (result, first, count, want) in [
        ([16, 0], 0, 1, [32, 0]),
        ([16, 0], 0, 3, [32, 1]),
        ([0, 0], 1, 2, [0, 1]),
        ([5, 7], 3, 1, [4, 7]),
        ([u32::MAX, 0], 2, 1, [0, 1]),
        ([9, 0], 1, 0, [9, 0]),
    ] {
        let dest = [0xAAAA_AAAA, 0xAAAA_AAAA, result[0], result[1], 7, 9];
        let out = combine(&[1, first, count, 1], &dest, &sources, 1);
        assert_eq!(
            out,
            [0xAAAA_AAAA, 0xAAAA_AAAA, want[0], want[1], 7, 9],
            "{result:?} plus {count} from {first}"
        );
    }
}

/// Every listed result combines, across workgroups, each with its own continuations; a word
/// between them is left as copied, and an invocation past the table does nothing.
#[test]
fn every_listed_result_combines_across_workgroups() {
    let n = VisibilityCombineModule::LOCAL_SIZE + 3;
    // Result k (at word 2k) has k % 3 continuations; continuation j of it holds 1 << (j % 32).
    let mut dest = Vec::new();
    for _ in 0..2 * n {
        dest.extend([0, 0]);
    }
    let (mut table, mut sources, mut want) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..n {
        let first = (sources.len() / 2) as u32;
        let count = k % 3;
        let mut low = 0;
        for j in 0..count {
            sources.extend([1 << ((k + j) % 32), 0]);
            low |= 1 << ((k + j) % 32);
        }
        table.extend([2 * k, first, count, 0]);
        want.push(low);
    }
    let out = combine(&table, &dest, &sources, 2);
    for k in 0..n as usize {
        assert_eq!(out[4 * k..4 * k + 2], [want[k], 0], "result {k}");
        assert_eq!(out[4 * k + 2..4 * k + 4], [0, 0], "word {}", 2 * k + 1);
    }
}

/// A result whose high half lies past the bound range is skipped, and so is a table entry past
/// it; continuations past the sources' end are not read. Nothing faults.
#[test]
fn words_past_the_bound_ranges_are_skipped() {
    let sources = [2, 0, 4, 0];
    assert_eq!(
        combine(
            &[0, 0, 2, 0, 1, 1, 1, 0, 9, 0, 1, 0],
            &[1, 0, 6],
            &sources,
            1
        ),
        [7, 0, 6]
    );
    assert_eq!(combine(&[0, 1, 5, 0], &[1, 0], &sources, 1), [5, 0]);
    assert_eq!(combine(&[0, 1, 5, 1], &[1, 0], &sources, 1), [5, 0]);
    assert_eq!(combine(&[0, 7, 1, 0], &[1, 0], &sources, 1), [1, 0]);
}

#[test]
fn the_plan_binds_aligned_windows() {
    let at = ContinuedResult::new;
    let plan = VisibilityCombine::plan(
        &[at(72, 1, 2, true), at(24, 0, 1, false)],
        64,
        1 << 16,
        65_535,
    )
    .expect("plan");
    assert_eq!(plan.len(), 1);
    assert_eq!(
        (plan[0].dest_range.offset, plan[0].dest_range.range),
        (0, 80)
    );
    assert_eq!(plan[0].table, [3, 0, 1, 0, 9, 1, 2, 1], "sorted by offset");
    assert_eq!(plan[0].group_count, 1);
    assert_eq!(plan[0].table_bytes()[..8], [3, 0, 0, 0, 0, 0, 0, 0]);
    // Results farther apart than one binding reaches take a dispatch each.
    let plan = VisibilityCombine::plan(&[at(8, 0, 1, false), at(200, 1, 1, false)], 16, 64, 65_535)
        .expect("plan");
    let ranges: Vec<_> = plan
        .iter()
        .map(|c| (c.dest_range.offset, c.dest_range.range, c.table.clone()))
        .collect();
    assert_eq!(
        ranges,
        [(0, 16, vec![1, 0, 1, 0]), (192, 16, vec![1, 1, 1, 0])]
    );
    // Too many for one dispatch.
    let results: Vec<_> = (0..65).map(|k| at(8 * k, k as u32, 1, false)).collect();
    let plan = VisibilityCombine::plan(&results, 8, 1 << 16, 1).expect("plan");
    assert_eq!(plan.len(), 2);
    assert_eq!(plan[0].table.len(), 64 * VisibilityCombine::ENTRY_WORDS);
    assert_eq!(plan[1].table, [0, 64, 1, 0]);
    assert_eq!(VisibilityCombine::plan(&[], 64, 64, 1), Some(Vec::new()));
}

#[test]
fn the_plan_refuses_what_it_cannot_bind() {
    let at = ContinuedResult::new;
    assert_eq!(
        VisibilityCombine::plan(&[at(12, 0, 1, false)], 8, 64, 1),
        None,
        "misaligned"
    );
    assert_eq!(
        VisibilityCombine::plan(&[at(8, 0, 1, false), at(8, 1, 1, true)], 8, 64, 1),
        None,
        "one offset twice"
    );
    assert_eq!(
        VisibilityCombine::plan(&[at(8, 7, 2, true)], 8, 64, 1),
        None,
        "continuations past the sources one binding reaches"
    );
    assert_eq!(
        VisibilityCombine::plan(&[at(0, 0, 1, false)], 8, 64, 0),
        None,
        "no groups"
    );
    assert_eq!(
        VisibilityCombine::plan(&[at(u64::MAX - 7, 0, 1, false)], 8, 64, 1),
        None,
        "an overflow"
    );
    assert_eq!(
        VisibilityCombine::plan(&[at(56, 0, 1, false)], 64, 32, 1),
        None,
        "a word past the range from its aligned offset"
    );
}

#[test]
fn the_device_module_is_numbered_with_the_fixed_bindings() {
    let (_, p) = committed_profiles()
        .expect("profiles")
        .into_iter()
        .find(|(s, _)| s == "vk12-floor")
        .expect("vk12-floor");
    let r = crate::spirv::serialize::serialize(typed_module().expect("module")).expect("serialize");
    let words = crate::spirv::serialize::relocate(&r, &binding_of).expect("relocate");
    crate::spirv::testing::check_words_valid(&words, &r.requirements, &p).expect("L3");
    let mut bindings: Vec<u32> = r
        .relocs
        .iter()
        .filter(|(s, _)| matches!(s, Symbol::Binding(_)))
        .map(|(_, at)| words[*at as usize])
        .collect();
    bindings.sort_unstable();
    assert_eq!(
        bindings,
        [
            VisibilityCombineModule::TABLE_BINDING,
            VisibilityCombineModule::DEST_BINDING,
            VisibilityCombineModule::SOURCES_BINDING
        ]
    );
}
