//! Thread pointers held in a thread's array of structs at indices only the run knows, among the
//! structs' data, end to end.
//!
//! The owned fixture `engine/pooled_cells` holds a renderer's closure array: `closure_alloc` hands
//! out the next element or null, and `closure_alloc_extra` hands out a thread pointer to room at
//! the array's end, which the element keeps. The array's slots are not disciplined (its pointers
//! and its data meet at offsets the run picks), so it forwards as a pool
//! (`memory::provenance::cells`): a pointer loaded from it is any pointer that leaks, or null.
//!
//! - `pooled_closures` and `pooled_bytes` (a pointer into another array copied into the pool as
//!   bytes) compile and pass L3 on every committed profile, reading their pointers through cell
//!   loads, and on the CPU interpreter write what the M3 Max writes (`pooled_cells.m`, macOS
//!   27.0.1).
//! - `pooled_spill` (the pointer's bits stored into device memory and read back into the pool) is
//!   refused on every committed profile, `Lowering(Access)`: a thread pointer converted to an
//!   integer is not lowered. The pool's loads do count the pointer (`memory::plan`'s tests), so
//!   what refuses it is the conversion, never a wrong answer.
//! - `pooled_context` (the same closures beside a context struct holding a constant pointer to an
//!   argument buffer of textures) compiles and passes L3 on the profiles of
//!   `generic_pointers`' tests; the runtime lane runs it against Metal's words.
//!
//! `llvm-as` assembles the fixture; a missing tool skips visibly.

use std::collections::BTreeMap;

use crate::device::DevicePolicy;
use crate::error::{Error, Feature, LowerFamily};
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, profile, typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::testing::committed_profiles;

/// The threads, all in one threadgroup.
const THREADS: u32 = 8;
/// The words each thread writes.
const WORDS: usize = 8;

/// What the M3 Max answers for `pooled_closures`, eight words a thread (the fixture's header).
#[rustfmt::skip]
const CLOSURES: [[f32; WORDS]; THREADS as usize] = [
    [10.5, -1.0, -1.0, 1.0, 0.0, 1.0, 0.5, 1.0],
    [11.5, 2.5, -1.0, 2.0, 0.0, 3.0, 1.5, 1.0],
    [12.5, 23.5, -1.0, 2.0, 0.0, 3.0, 5.0, 2.0],
    [13.5, 24.5, 5.5, 3.0, 0.0, 6.0, 7.0, 2.0],
    [14.5, -1.0, -1.0, 1.0, 0.0, 1.0, 4.5, 1.0],
    [15.5, 6.5, -1.0, 2.0, 0.0, 3.0, 5.5, 1.0],
    [16.5, 27.5, -1.0, 2.0, 0.0, 3.0, 13.0, 2.0],
    [17.5, 28.5, 9.5, 3.0, 0.0, 6.0, 15.0, 2.0],
];

/// What the M3 Max answers for `pooled_bytes` and `pooled_spill` alike.
#[rustfmt::skip]
const COPIED: [[f32; WORDS]; THREADS as usize] = [
    [0.5, 100.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [2.5, 102.5, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    [4.5, 104.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [6.5, 106.5, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    [4.5, 104.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [6.5, 106.5, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    [8.5, 108.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [10.5, 110.5, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
];

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("pooled_cells.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected pooled_cells: {why}"),
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

/// Kernel `entry` on every committed profile: compiled, past L3, reading pointers through cell
/// loads, and run on the CPU interpreter over `in[i] = i + 0.5` with `third` at buffer 2 when
/// given; its words checked against `metal`.
fn answers_as_metal(bc: &[u8], entry: &str, third: Option<Vec<u8>>, metal: &[[f32; WORDS]]) {
    let input: Vec<u8> = (0..16u16)
        .flat_map(|i| (f32::from(i) + 0.5).to_le_bytes())
        .collect();
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = compile(bc, entry, &p).unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {entry}: L3: {e}"));
        let (plan, m) = typed_module(bc, entry, &p).expect("module");
        assert!(
            plan.census()
                .get("access.cell.load")
                .is_some_and(|(n, _)| *n > 0),
            "{stem} {entry}: {:?}",
            plan.census()
        );
        for f in &plan.push.fields {
            match (f.field, f.loc) {
                (
                    PushField::SlotBindOffset(_) | PushField::DispatchRegion,
                    PushLoc::Push { .. },
                ) => {}
                other => panic!("{stem} {entry}: unexpected push field {other:?}"),
            }
        }
        let mut buffers = vec![
            buffer(0, vec![0; 4 * WORDS * THREADS as usize]),
            buffer(1, input.clone()),
        ];
        if let Some(b) = &third {
            buffers.push(buffer(2, b.clone()));
        }
        let res = Resources {
            buffers,
            push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
            spec: BTreeMap::from([
                (SpecKey::LocalSize(Axis::X), u64::from(THREADS)),
                (SpecKey::LocalSize(Axis::Y), 1),
                (SpecKey::LocalSize(Axis::Z), 1),
                (SpecKey::FlatLocalSize, u64::from(THREADS)),
            ]),
        };
        let cfg = Config {
            workgroups: [1, 1, 1],
            ..Config::default()
        };
        let out = run(&m, &cfg, res).unwrap_or_else(|e| panic!("{stem} {entry}: {e:?}"));
        let got: Vec<f32> = out.buffers[0]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect();
        for (t, (got, want)) in got.chunks(WORDS).zip(metal.iter()).enumerate() {
            assert_eq!(got, want, "{stem} {entry}: thread {t}");
        }
    }
}

/// Elements handed out at run-time indices, each holding a thread pointer to room at the array's
/// end, read back through the pool as Metal reads them.
#[test]
fn pointers_a_closure_array_holds_answer_as_metal() {
    let Some(bc) = fixture() else { return };
    answers_as_metal(&bc, "pooled_closures", None, &CLOSURES);
}

/// A pointer into another array copied into the pool as bytes is read back and written through as
/// Metal's is; one whose bits pass through device memory as an integer is refused at the
/// conversion.
#[test]
fn a_pointer_copied_into_a_pool_as_data_answers_as_metal_or_is_refused() {
    let Some(bc) = fixture() else { return };
    answers_as_metal(
        &bc,
        "pooled_bytes",
        Some(8u32.to_le_bytes().to_vec()),
        &COPIED,
    );
    for (stem, p) in committed_profiles().expect("profiles") {
        match compile(&bc, "pooled_spill", &p).map(|_| ()) {
            Err(Error::Unsupported(u)) if u.feature == Feature::Lowering(LowerFamily::Access) => {}
            other => panic!("{stem} pooled_spill: {other:?}"),
        }
    }
}

/// The closures beside a context struct holding a constant pointer to an argument buffer of
/// textures compile and pass L3.
#[test]
fn a_context_s_argument_buffer_beside_a_pool_compiles() {
    let Some(bc) = fixture() else { return };
    for stem in ["lavapipe", "amd-rdna3-radv", "moltenvk-m3"] {
        let p = profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile");
        let a = compile(&bc, "pooled_context", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
    }
}
