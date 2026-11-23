//! A thread pointer that may be null, end to end.
//!
//! The owned fixture `engine/nullable_private`'s `closure_alloc` hands out the next element of a
//! thread's array at an index only the run knows, or null once the array's room is used up (a
//! `phi` of the element's address and null), and its `noinline` callees compare that pointer with
//! null and write through it where it is not. The private arena keeps its first word for null, so
//! the pointer is an arena offset that is 0 where it is null.
//!
//! - The kernel compiles and passes L3 on every committed profile, comparing the pointer's offset
//!   with null's (`access.private.pointer_compare`).
//! - On the CPU interpreter it writes what the M3 Max writes (`nullable_private.m`), on every
//!   committed profile.
//!
//! `llvm-as` assembles the fixture; a missing tool skips visibly.

use std::collections::BTreeMap;

use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::testing::committed_profiles;

const KERNEL: &str = "nullable_private";
/// The threads, all in one threadgroup.
const THREADS: u32 = 8;
/// The words each thread writes.
const WORDS: usize = 8;

/// What the M3 Max answers (macOS 27.0.1) over `in[i] = i + 0.5`, eight words a thread: four
/// weights (-1 where the pointer is null), the closures handed out, the sum of their types, the sum
/// of their data, and 1 where the fifth request is null.
#[rustfmt::skip]
const METAL: [[f32; WORDS]; THREADS as usize] = [
    [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    [3.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.5, 1.0],
    [5.0, 7.0, -1.0, -1.0, 2.0, 3.0, 6.0, 1.0],
    [7.0, 9.0, 11.0, -1.0, 3.0, 6.0, 13.5, 1.0],
    [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    [11.0, -1.0, -1.0, -1.0, 1.0, 1.0, 5.5, 1.0],
    [13.0, 15.0, -1.0, -1.0, 2.0, 3.0, 14.0, 1.0],
    [15.0, 17.0, 19.0, -1.0, 3.0, 6.0, 25.5, 1.0],
];

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{KERNEL}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {KERNEL}: {why}"),
    }
}

fn buffer(index: u32, bytes: Vec<u8>) -> Buffer {
    Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        }),
        address: None,
        bytes,
    }
}

/// A pointer that is an array element or null compares with null, and reads and writes the
/// element where it is not null, as Metal's does, on every committed profile.
#[test]
fn a_thread_pointer_that_may_be_null_answers_as_metal() {
    let Some(bc) = fixture() else { return };
    let input: Vec<u8> = (0..16u16)
        .flat_map(|i| (f32::from(i) + 0.5).to_le_bytes())
        .collect();
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = compile(&bc, KERNEL, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let (plan, m) = typed_module(&bc, KERNEL, &p).expect("module");
        assert!(
            plan.census()
                .get("access.private.pointer_compare")
                .is_some_and(|(n, _)| *n > 0),
            "{stem}: {:?}",
            plan.census()
        );
        // Zero bind offsets, and a region the single-region dispatch does not read.
        for f in &plan.push.fields {
            match (f.field, f.loc) {
                (
                    PushField::SlotBindOffset(_) | PushField::DispatchRegion,
                    PushLoc::Push { .. },
                ) => {}
                other => panic!("{stem}: unexpected push field {other:?}"),
            }
        }
        let res = Resources {
            buffers: vec![
                buffer(0, vec![0; 4 * WORDS * THREADS as usize]),
                buffer(1, input.clone()),
            ],
            push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
            spec: BTreeMap::from(crate::pipeline::threadgroup_constants([THREADS, 1, 1])),
        };
        let cfg = Config {
            workgroups: [1, 1, 1],
            ..Config::default()
        };
        let out = run(&m, &cfg, res).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        let got: Vec<f32> = out.buffers[0]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect();
        for (t, (got, want)) in got.chunks(WORDS).zip(METAL.iter()).enumerate() {
            assert_eq!(got, want, "{stem}: thread {t}");
        }
    }
}
