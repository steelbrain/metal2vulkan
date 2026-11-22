//! A buffer argument's offsets are 32-bit under the identity domains, and the runtime makes no
//! buffer longer than 2 GiB (`metal2vulkan_runtime::MAX_BUFFER_LENGTH`), so every byte of a bound buffer
//! lies within 2^31 bytes of the bind address. On the CPU interpreter, which traps on an address
//! that lies in no buffer, `ob1_storage_buffer` loads and stores at word indices up to the last
//! word below 2 GiB through slots whose bytes are two buffers: one at the bind address and one
//! ending at 2 GiB past it. Each access lands at its own word.
//!
//! `llvm-as` assembles the fixture; without it the test skips visibly.

use std::collections::BTreeMap;

use super::engine_word_phases::{bounds_word, identity_profiles, slot_word, ZERO_UNMAPPED};
use crate::lower::dispatch::region_bytes;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, SpecKey};
use crate::pipeline::testing::{compile_kept, engine_dir, llvm_assemble, AssembleError};
use crate::spirv::interp::{run, Buffer, Config, Resources};

const FIXTURE: &str = "ob1_storage_buffer";
const THREADS: u32 = 8;
/// The word at byte 2 GiB of the slot, one past its last.
const AT_2_GIB: u32 = 1 << 29;
/// The device address of the index buffer, and the one `src` and `out` are both bound at.
const IDX: u64 = 0x1_0000_0000;
const SLOT: u64 = 0x10_0000_0000;
/// The slot's words in the interpreter's buffers: the first 1024 at `SLOT`, and the last 1024
/// below 2 GiB at `SLOT + 4 * HIGH`.
const WORDS: u32 = 1024;
const HIGH: u32 = AT_2_GIB - WORDS;

/// Thread `t` loads `src[LOADS[t]]` into `out[t]` and stores `0xA0000000 | t` to
/// `out[STORES[t]]`: one load and one store near the bind address, the rest below 2 GiB up to its
/// last word.
const LOADS: [u32; 8] = [
    70,
    HIGH,
    HIGH + 1,
    AT_2_GIB - 300,
    AT_2_GIB - 4,
    AT_2_GIB - 3,
    AT_2_GIB - 2,
    AT_2_GIB - 1,
];
const STORES: [u32; 8] = [
    128,
    HIGH + 2,
    HIGH + 3,
    AT_2_GIB - 400,
    AT_2_GIB - 8,
    AT_2_GIB - 7,
    AT_2_GIB - 6,
    AT_2_GIB - 5,
];

/// The interpreter buffer (1 low, 2 high) and byte offset in it of slot word `word`.
fn at(word: u32) -> (usize, usize) {
    match word {
        w if w < WORDS => (1, 4 * w as usize),
        w => (2, 4 * (w - HIGH) as usize),
    }
}

#[test]
fn accesses_up_to_2_gib_land_at_their_own_words() {
    let bc = match llvm_assemble(&engine_dir().join(format!("{FIXTURE}.ll"))) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(e) => panic!("{FIXTURE}: {e:?}"),
    };
    for (stem, p) in identity_profiles()
        .iter()
        .filter(|(s, _)| s == "lavapipe" || s == "amd-780m-radv")
    {
        let (_, plan, module) =
            compile_kept(&bc, FIXTURE, p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
        let region = region_bytes([0; 3], [THREADS, 1, 1], [THREADS, 1, 1]).expect("packs");
        for f in &plan.push.fields {
            let PushLoc::Push { offset } = f.loc else {
                panic!("{stem}: {f:?} spilled");
            };
            let o = offset as usize;
            match f.field {
                PushField::SlotGuestBase(s) => {
                    let address = if s.index == 0 { IDX } else { SLOT };
                    push[o..o + 8].copy_from_slice(&slot_word(address).to_le_bytes());
                }
                // The index buffer's words, and the slot's 2 GiB.
                PushField::SlotBounds(s) => {
                    let bytes = match s.index {
                        0 => 4 * (LOADS.len() + STORES.len()),
                        _ => 4 * AT_2_GIB as usize,
                    };
                    push[o..o + 8].copy_from_slice(&bounds_word(bytes, 0));
                }
                PushField::ZeroAddress => {
                    push[o..o + 8].copy_from_slice(&ZERO_UNMAPPED.to_le_bytes())
                }
                PushField::DispatchRegion => push[o..o + 32].copy_from_slice(&region),
                _ => panic!("{stem}: unexpected push field {f:?}"),
            }
        }
        let mut bytes = [
            LOADS
                .iter()
                .chain(&STORES)
                .flat_map(|w| w.to_le_bytes())
                .collect(),
            vec![0u8; 4 * WORDS as usize],
            vec![0u8; 4 * WORDS as usize],
        ];
        for (t, &word) in LOADS.iter().enumerate() {
            let (b, o) = at(word);
            bytes[b][o..o + 4].copy_from_slice(&(0x5000_0000 | t as u32).to_le_bytes());
        }
        let spec = BTreeMap::from([
            (SpecKey::LocalSize(Axis::X), u64::from(THREADS)),
            (SpecKey::LocalSize(Axis::Y), 1),
            (SpecKey::LocalSize(Axis::Z), 1),
            (SpecKey::FlatLocalSize, u64::from(THREADS)),
        ]);
        let res = Resources {
            buffers: [IDX, SLOT, SLOT + 4 * u64::from(HIGH)]
                .into_iter()
                .zip(bytes)
                .map(|(address, bytes)| Buffer {
                    slot: None,
                    address: Some(address),
                    bytes,
                })
                .collect(),
            push,
            spec,
        };
        let got = run(&module, &Config::default(), res).unwrap_or_else(|e| {
            panic!("{stem}: loads at {LOADS:#x?}, stores at {STORES:#x?}: {e}")
        });
        let word = |w: u32| {
            let (b, o) = at(w);
            u32::from_le_bytes(got.buffers[b][o..o + 4].try_into().expect("a word"))
        };
        let loaded: Vec<u32> = (0..THREADS).map(word).collect();
        let want: Vec<u32> = (0..THREADS).map(|t| 0x5000_0000 | t).collect();
        assert_eq!(loaded, want, "{stem}: the loads at {LOADS:#x?}");
        for (t, &w) in STORES.iter().enumerate() {
            assert_eq!(
                word(w),
                0xA000_0000 | t as u32,
                "{stem}: the store to word {w:#x}"
            );
        }
    }
}
