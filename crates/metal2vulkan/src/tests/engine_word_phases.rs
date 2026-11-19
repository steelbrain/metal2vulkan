//! The vector word-phase fixture (`engine/vector_word_phases`) under the identity domains, which
//! reach every buffer through its device address (`lower::access::psb`), as `Device::from_queries`
//! derives each committed profile that has device addresses and 64-bit integers.
//!
//! Metal places a `uint4` or `float4` access at its own address, a multiple of 4 that need not be
//! one of 16, whatever the `align 16` its AIR states (`memory::leaf`'s placement). So no
//! device-address load or store claims more alignment than the translator proves for its address
//! (`MemoryPlan::proven_align`): 4 through `vwp_words`'s `uint` buffers, which Metal binds at any
//! multiple of 4, and 16 through `vwp_vec16`'s `float4` ones. A form that took the AIR's `align`
//! fails here without a GPU. On the CPU interpreter, which traps a physical access whose address
//! its `Aligned` operand does not divide, both kernels answer Metal's words with `vwp_words`'s
//! buffers bound at offsets 0, 4, 8 and 12 (`metal2vulkan_devtools::vector_word_phases`).
//!
//! `llvm-as` assembles the fixture; without it the tests skip visibly.

use std::collections::{BTreeMap, BTreeSet};

use crate::air::ir::InstKind;
use crate::device::profile::{DeviceProfile, HandleDomain, PointerDomain};
use crate::device::{Device, DevicePolicy};
use crate::library::FunctionConstantValues;
use crate::lower::dispatch::region_bytes;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, SpecKey};
use crate::pipeline::testing::{
    compile, compile_kept, description, engine_dir, link_and_validate, llvm_assemble, meter,
    specialized_entry, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::testing::committed_profiles;
use crate::spirv_binary::{parse_bytes, Operand};
use crate::spv::{Op, StorageClass};
use metal2vulkan_devtools::vector_word_phases::{
    answer, inputs, vec16_offsets, FIXTURE, OFFSETS, THREADS, VEC16, WORDS,
};

/// `MemoryAccess`'s `Aligned` bit.
const ALIGNED: u32 = 0x2;

fn identity() -> DevicePolicy {
    DevicePolicy {
        pointer_domain: PointerDomain::Identity,
        handle_domain: HandleDomain::Identity,
        ..DevicePolicy::default()
    }
}

fn bitcode() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{FIXTURE}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(e) => panic!("{FIXTURE}: {e:?}"),
    }
}

/// The word the runtime pushes for a buffer slot's `SlotGuestBase` at `address` under the identity
/// domains: the address less 2^31 (`lower::access::SLOT_ADDRESS_BIAS`).
pub(super) fn slot_word(address: u64) -> u64 {
    address.wrapping_sub(crate::lower::access::SLOT_ADDRESS_BIAS)
}

/// Every committed profile `Device::from_queries` derives under the identity domains (those with
/// `bufferDeviceAddress` and `shaderInt64`), by stem.
pub(super) fn identity_profiles() -> Vec<(String, DeviceProfile)> {
    committed_profiles()
        .expect("profiles")
        .into_iter()
        .filter_map(|(stem, _)| {
            let desc = description(&stem).expect("description");
            let device = Device::from_queries(desc, identity()).ok()?;
            Some((stem, device.profile().clone()))
        })
        .collect()
}

/// The alignments `MemoryPlan::proven_align` gives the addresses of kernel `name`'s loads and
/// stores.
fn proven(bc: &[u8], name: &str, p: &DeviceProfile) -> BTreeSet<u64> {
    let m = &mut meter();
    let consts = FunctionConstantValues::defaults();
    let e = specialized_entry(bc, name, p, &consts, &[], m).expect("entry");
    let memory = crate::memory::plan::analyze_memory(&e, p, m).expect("memory plan");
    let body = e
        .module
        .func(e.func)
        .and_then(|f| f.body.as_ref())
        .expect("a body");
    body.insts
        .iter()
        .filter_map(|(_, inst)| match &inst.kind {
            InstKind::Load { ptr, .. } | InstKind::Store { ptr, .. } => {
                Some(memory.proven_align(&e, *ptr))
            }
            _ => None,
        })
        .collect()
}

/// The `Aligned` operand of every load and store of serialized module `words` through a
/// `PhysicalStorageBuffer` pointer (0 for one without it).
pub(super) fn device_alignments(words: &[u32]) -> Vec<u32> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let module = parse_bytes(&bytes).expect("parse");
    let physical = StorageClass::PhysicalStorageBuffer as u32;
    let pointers: BTreeSet<u32> = module
        .instructions
        .iter()
        .filter(|i| i.opcode == Op::TypePointer)
        .filter(|i| i.operands.first() == Some(&Operand::Word(physical)))
        .filter_map(|i| i.result_id)
        .collect();
    let types: BTreeMap<u32, u32> = module
        .instructions
        .iter()
        .filter_map(|i| Some((i.result_id?, i.result_type?)))
        .collect();
    let through = |ptr: &Operand| match ptr {
        Operand::IdRef(id) => types.get(id).is_some_and(|t| pointers.contains(t)),
        _ => false,
    };
    let aligned = |access: &[Operand]| match access {
        [Operand::Word(mask), Operand::Word(align), ..] if mask & ALIGNED != 0 => *align,
        _ => 0,
    };
    module
        .instructions
        .iter()
        .filter_map(|i| match (i.opcode, i.operands.as_slice()) {
            (Op::Load, [ptr, access @ ..]) if through(ptr) => Some(aligned(access)),
            (Op::Store, [ptr, _, access @ ..]) if through(ptr) => Some(aligned(access)),
            _ => None,
        })
        .collect()
}

/// Every device-address load and store of the two kernels claims at most the alignment the
/// translator proves for its address: 4 where the buffer is a `uint` one Metal binds at any
/// multiple of 4, whatever the `align 16` of the AIR's `uint4` and `float4` accesses through it,
/// and 16 where it is a `float4` one, where the access is one `uvec4`: a slot held to at least 4
/// is reached from its address as it is, whatever its window. Each argument states its alignment
/// (`RuntimeFact::AddressAlignment`). Each module passes L3.
#[test]
fn device_accesses_claim_at_most_the_proven_alignment() {
    let Some(bc) = bitcode() else {
        return;
    };
    let profiles = identity_profiles();
    for stem in [
        "lavapipe",
        "moltenvk-m3",
        "amd-rdna3-radv",
        "nvidia-rtx50-linux",
    ] {
        assert!(
            profiles.iter().any(|(s, _)| s == stem),
            "{stem} is not among the identity profiles"
        );
    }
    for (stem, p) in &profiles {
        for (name, align) in [(WORDS, 4), (VEC16, 16)] {
            let what = format!("{name} on {stem}");
            assert_eq!(
                proven(&bc, name, p),
                BTreeSet::from([align]),
                "{what}: proven"
            );
            let a = compile(&bc, name, p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let claimed = device_alignments(&a.module.words);
            assert!(!claimed.is_empty(), "{what}: no device-address access");
            assert!(
                claimed.iter().all(|&c| (4..=align).contains(&u64::from(c))),
                "{what}: claimed {claimed:?}, proven {align}"
            );
            // A `float4` access at an offset proven a multiple of 16 is one `uvec4` access.
            assert_eq!(
                claimed.iter().max().copied().map(u64::from),
                Some(align),
                "{what}: claimed {claimed:?}, proven {align}"
            );
            // Both arguments state their alignment, which the binder holds their addresses to.
            let stated: Vec<u32> = (a.facts.iter())
                .filter_map(|f| match f {
                    RuntimeFact::AddressAlignment { align, .. } => Some(*align),
                    _ => None,
                })
                .collect();
            assert_eq!(stated, [align as u32; 2], "{what}: stated alignments");
        }
    }
}

/// On the CPU interpreter, with `vwp_words`'s buffers bound at each word phase and `vwp_vec16`'s
/// at multiples of 16, every buffer above 4 GiB, both kernels leave Metal's words in all four
/// buffers, and no access traps on its `Aligned` operand.
#[test]
fn vector_accesses_answer_metal_at_every_word_phase() {
    let Some(bc) = bitcode() else {
        return;
    };
    let before = inputs();
    for (stem, p) in identity_profiles()
        .iter()
        .filter(|(s, _)| s == "lavapipe" || s == "moltenvk-m3")
    {
        for offset in OFFSETS {
            let what = format!("{stem} at offset {offset}");
            let (a_at, b_at) = vec16_offsets(offset);
            let [src, dst] = dispatch(
                &bc,
                WORDS,
                p,
                [
                    (0x1_0000_0000, offset, before.src.clone()),
                    (0x2_0000_1000, offset, before.dst.clone()),
                ],
            )
            .unwrap_or_else(|e| panic!("{what}: {WORDS}: {e}"));
            let [a, b] = dispatch(
                &bc,
                VEC16,
                p,
                [
                    (0x3_0000_2000, a_at, before.a.clone()),
                    (0x4_0000_3000, b_at, before.b.clone()),
                ],
            )
            .unwrap_or_else(|e| panic!("{what}: {VEC16}: {e}"));
            let want = answer(offset);
            for (buffer, got, want) in [
                ("src", &src, &want.src),
                ("dst", &dst, &want.dst),
                ("a", &a, &want.a),
                ("b", &b, &want.b),
            ] {
                let words = |b: &[u8]| b.as_chunks::<4>().0.to_vec();
                let first = words(got)
                    .iter()
                    .zip(&words(want))
                    .position(|(x, y)| x != y);
                assert!(
                    got.len() == want.len() && first.is_none(),
                    "{what}: {buffer} differs from Metal's words first at word {first:?}"
                );
            }
        }
    }
}

/// Kernel `name` run once over [`THREADS`] threads in one threadgroup on the interpreter, its
/// buffer `n` at device address `buffers[n].0` holding `buffers[n].2`, bound at offset
/// `buffers[n].1`: the buffers' bytes afterwards.
fn dispatch(
    bc: &[u8],
    name: &str,
    p: &DeviceProfile,
    buffers: [(u64, u64, Vec<u8>); 2],
) -> Result<[Vec<u8>; 2], String> {
    let (a, plan, module) = compile_kept(bc, name, p).map_err(|e| e.to_string())?;
    assert!(a.interface.descriptors.is_empty(), "{name}: a descriptor");
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    let region = region_bytes([0; 3], [THREADS, 1, 1], [THREADS, 1, 1]).expect("packs");
    for f in &plan.push.fields {
        let PushLoc::Push { offset } = f.loc else {
            return Err(format!("{f:?} spilled"));
        };
        let o = offset as usize;
        match f.field {
            PushField::SlotGuestBase(s) => {
                let (address, bind, _) = &buffers[s.index as usize];
                push[o..o + 8].copy_from_slice(&slot_word(address + bind).to_le_bytes());
            }
            PushField::DispatchRegion => push[o..o + 32].copy_from_slice(&region),
            _ => return Err(format!("unexpected push field {f:?}")),
        }
    }
    let spec = BTreeMap::from([
        (SpecKey::LocalSize(Axis::X), u64::from(THREADS)),
        (SpecKey::LocalSize(Axis::Y), 1),
        (SpecKey::LocalSize(Axis::Z), 1),
        (SpecKey::FlatLocalSize, u64::from(THREADS)),
    ]);
    let [first, second] = buffers;
    let res = Resources {
        buffers: [first, second]
            .into_iter()
            .map(|(address, _, bytes)| Buffer {
                slot: None,
                address: Some(address),
                bytes,
            })
            .collect(),
        push,
        spec,
    };
    let got = run(&module, &Config::default(), res).map_err(|e| e.to_string())?;
    let [x, y]: [Vec<u8>; 2] = got
        .buffers
        .try_into()
        .map_err(|_| "two buffers".to_owned())?;
    Ok([x, y])
}
