//! Sequentially consistent fences: a `seq_cst` `air.atomic.fence`, an LLVM `fence seq_cst`, and a
//! program's `seq_cst` barrier whose memory scope is the device are each followed by an
//! acquire-release increment of the command buffer's `StatusWord::SeqCstFences` at the fence's
//! scope, which keeps them in one total order, as Metal's are: Vulkan's `SequentiallyConsistent` is
//! `AcquireRelease`, which lets a store-buffering pair both read the old values. An `acq_rel` fence
//! takes no increment. The increment reaches the status word by its device address where the device
//! addresses buffers, and through the stage's binding of the status buffer elsewhere.
//!
//! The kernel is the store-buffering litmus of our probe
//! (`metal2vulkan_devtools::litmus::store_buffering`).

use crate::device::profile::BufferAccess;
use crate::error::{Error, Feature};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::testing::{
    compile, device_f, link_and_validate, status_reach, status_reach_on, StatusReach, PROFILE_F,
};
use crate::spirv::testing::committed_profiles;
use metal2vulkan_devtools::litmus::store_buffering;
use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};

const OP_CONSTANT: u32 = 43;
const OP_MEMORY_BARRIER: u32 = 225;
const OP_ATOMIC_IADD: u32 = 234;
const SCOPE_DEVICE: u32 = 1;
const SCOPE_WORKGROUP: u32 = 2;

/// Bitcode of `text` through `llvm-as`, or `None` (printing why) where it is missing.
fn assemble(text: &str) -> Option<Vec<u8>> {
    match llvm_assemble_text(text) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the kernel: {why}"),
    }
}

/// The scope constant of every `OpAtomicIAdd` of `words`, and the count of `OpMemoryBarrier`s.
fn increments_and_fences(words: &[u32]) -> (Vec<u32>, usize) {
    let mut constants = std::collections::BTreeMap::new();
    let (mut scopes, mut fences) = (Vec::new(), 0);
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let (len, op) = ((w >> 16) as usize, w & 0xFFFF);
        let operand = |k: usize| words.get(i + k).copied();
        match op {
            OP_CONSTANT if len == 4 => {
                if let (Some(id), Some(v)) = (operand(2), operand(3)) {
                    constants.insert(id, v);
                }
            }
            OP_ATOMIC_IADD => {
                let scope = operand(4).and_then(|id| constants.get(&id).copied());
                scopes.push(scope.expect("an atomic's scope is a constant"));
            }
            OP_MEMORY_BARRIER => fences += 1,
            _ => {}
        }
        if len == 0 {
            break;
        }
        i += len;
    }
    (scopes, fences)
}

/// On every committed profile and on [`PROFILE_F`] (below the runtime's floor), each `seq_cst` fence of the store-buffering kernel
/// (`air.atomic.fence` at device and at threadgroup scope, and an LLVM `fence seq_cst`) is an
/// `OpMemoryBarrier` and an increment of the status word at the fence's scope, the module reaching
/// the `ExecutionStatus` buffer by address where the profile addresses buffers and through the
/// stage's binding of it elsewhere; an `acq_rel` fence is the barrier alone, reaching no status.
#[test]
fn a_seq_cst_fence_is_ordered_with_the_command_buffers_others() {
    let device = "  tail call void @air.atomic.fence(i32 1, i32 5, i32 2)";
    let threadgroup = "  tail call void @air.atomic.fence(i32 1, i32 5, i32 1)";
    let llvm = "  fence seq_cst";
    let acq_rel = "  tail call void @air.atomic.fence(i32 1, i32 4, i32 2)";
    let mut kernels = Vec::new();
    for (fence, scope) in [
        (device, Some(SCOPE_DEVICE)),
        (threadgroup, Some(SCOPE_WORKGROUP)),
        (llvm, Some(SCOPE_DEVICE)),
        (acq_rel, None),
    ] {
        let Some(bc) = assemble(&store_buffering(fence)) else {
            return;
        };
        kernels.push((fence, scope, bc));
    }
    let (mut addressed, mut floor) = (0, 0);
    let mut profiles = committed_profiles().expect("committed profiles");
    let f = device_f(crate::device::DevicePolicy::default()).expect(PROFILE_F);
    profiles.push((PROFILE_F.into(), f.profile().clone()));
    for (stem, p) in profiles {
        for (fence, scope, bc) in &kernels {
            let what = format!("{stem}, {}", fence.trim());
            let a = match compile(bc, "sb", &p) {
                Ok(a) => a,
                Err(Error::Unsupported(u)) if u.feature != Feature::SeqCstFence => {
                    eprintln!("{what}: refused for {u:?}");
                    continue;
                }
                Err(e) => panic!("{what}: {e}"),
            };
            let (scopes, fences) = increments_and_fences(&a.module.words);
            assert_eq!(fences, 2, "{what}: one barrier a side");
            match scope {
                Some(s) => {
                    assert_eq!(scopes, [*s, *s], "{what}: one increment a side");
                    assert!(a.facts.contains(&RuntimeFact::ExecutionStatus), "{what}");
                    assert_eq!(status_reach(&a), status_reach_on(&p), "{what}");
                }
                None => {
                    assert!(scopes.is_empty(), "{what}: {scopes:?}");
                    assert_eq!(status_reach(&a), StatusReach::None, "{what}");
                }
            }
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        }
        match p.fields().buffer_access == BufferAccess::Addressed {
            true => addressed += 1,
            false => floor += 1,
        }
    }
    assert!(
        addressed >= 10 && floor >= 1,
        "{addressed} with addresses, {floor} without"
    );
}
