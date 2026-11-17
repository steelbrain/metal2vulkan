//! A buffer argument's offsets are Metal's unsigned 32-bit ones under the identity domains, which
//! reach every buffer through its device address (`lower::access::psb`): an access 2 GiB and more
//! past the bind offset lands there, not 4 GiB below, where a signed 32-bit access-chain index
//! would place it. The CPU interpreter takes an access chain's index as signed, as SPIR-V does,
//! and traps on an address that lies in no buffer. On it, `ob1_storage_buffer` loads and stores at
//! word indices on both sides of 2^29 through slots whose bytes are two buffers, one at the bind
//! address and one around 2 GiB past it.
//!
//! `llvm-as` assembles the fixture; without it the test skips visibly.

use std::collections::BTreeMap;

use super::engine_word_phases::identity_profiles;
use crate::lower::dispatch::region_bytes;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, SpecKey};
use crate::pipeline::testing::{compile_kept, engine_dir, llvm_assemble, AssembleError};
use crate::spirv::interp::{run, Buffer, Config, Resources};

const FIXTURE: &str = "ob1_storage_buffer";
const THREADS: u32 = 8;
/// The word at byte 2 GiB of the slot.
const AT_2_GIB: u32 = 1 << 29;
/// The device address of the index buffer, and the one `src` and `out` are both bound at.
const IDX: u64 = 0x1_0000_0000;
const SLOT: u64 = 0x2_0000_0000;
/// The slot's words in the interpreter's buffers: the first 1024 at `SLOT`, and those from
/// `HIGH` on, 16 below 2 GiB to 64 KiB past it, at `SLOT + 4 * HIGH`.
const LOW_WORDS: u32 = 1024;
const HIGH: u32 = AT_2_GIB - 16;
const HIGH_WORDS: u32 = 16 + (64 << 10) / 4;

/// Thread `t` loads `src[LOADS[t]]` into `out[t]` and stores `0xA0000000 | t` to
/// `out[STORES[t]]`: one load and one store below 2 GiB, the rest from the word below it to the
/// last word of the high range.
const LOADS: [u32; 8] = [
    64,
    AT_2_GIB - 1,
    AT_2_GIB,
    AT_2_GIB + 1,
    AT_2_GIB + 2,
    AT_2_GIB + 3,
    AT_2_GIB + 4096,
    HIGH + HIGH_WORDS - 1,
];
const STORES: [u32; 8] = [
    128,
    AT_2_GIB - 2,
    AT_2_GIB + 8,
    AT_2_GIB + 9,
    AT_2_GIB + 10,
    AT_2_GIB + 11,
    AT_2_GIB + 8000,
    HIGH + HIGH_WORDS - 2,
];

/// The interpreter buffer (1 low, 2 high) and byte offset in it of slot word `word`.
fn at(word: u32) -> (usize, usize) {
    match word {
        w if w < LOW_WORDS => (1, 4 * w as usize),
        w => (2, 4 * (w - HIGH) as usize),
    }
}

#[test]
fn accesses_past_2_gib_land_at_their_own_words() {
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
        .filter(|(s, _)| s == "lavapipe" || s == "moltenvk-m3")
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
                    push[o..o + 8].copy_from_slice(&address.to_le_bytes());
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
            vec![0u8; 4 * LOW_WORDS as usize],
            vec![0u8; 4 * HIGH_WORDS as usize],
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
