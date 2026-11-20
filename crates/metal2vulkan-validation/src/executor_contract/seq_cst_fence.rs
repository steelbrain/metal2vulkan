//! A sequentially consistent fence.
//!
//! The translator orders a `seq_cst` fence, and a program's `seq_cst` barrier whose memory scope is
//! the device, with every other such fence of the command buffer through an acquire-release
//! increment of one word of its execution status buffer (`lower::sync::barrier`). The runtime's
//! identity domains reach no such word: its pipeline is refused as `Unsupported{IdentityDomains}`,
//! naming the `ExecutionStatusAddress` encoder value, so the engine's candidate does not run the
//! case ([`ToolingRequirement::SeqCstFence`]). The Metal oracle runs it.
//!
//! The scan reads the entry's function and the functions it reaches by direct calls, which its
//! pipeline compiles, as the `min_lod_clamp` scan does. A call counts in the forms the translator
//! takes the word for: `air.atomic.fence` of order `seq_cst` (5) at a scope other than the thread
//! (a threadgroup scope only in a stage with a threadgroup), and a barrier in the MSL 4.1 form (its
//! three operands carry an order) of order `seq_cst` whose memory scope is the device or the
//! legacy all-devices scope; each names a memory (`mem_device`, `mem_threadgroup`, `mem_texture`
//! or `mem_threadgroup_imageblock`), without which it orders nothing and emits no fence. A
//! barrier of the AIR 2.8 form orders memory at its own execution scope, never the device's, and
//! is no claim. A call whose operands are not constant integers is no claim.

use super::min_lod_clamp::{direct_callee, function_bodies};
use crate::air::reflect::{ShaderInterface, ShaderStage};
use crate::requirement::ToolingRequirement;
use metal2vulkan_air::intrinsics::{AirIntrinsic, Sync};
use std::collections::BTreeSet;

/// The `seq_cst` memory order's immediate.
const SEQ_CST: u64 = 5;
/// The memories a fence orders that the translator emits a fence for (`mem_device`,
/// `mem_threadgroup`, `mem_texture`, `mem_threadgroup_imageblock`).
const ORDERED_MEMORIES: u64 = 0xf;
/// The `thread_scope` immediates: thread, threadgroup, device, all devices.
const THREAD: u64 = 0;
const THREADGROUP: u64 = 1;
const DEVICE: u64 = 2;
const ALL_DEVICES: u64 = 3;

/// [`ToolingRequirement::SeqCstFence`] when a call the pipeline of `reflection`'s entry compiles
/// is a sequentially consistent fence ([`takes_the_status_word`]).
pub fn seq_cst_fence_requirement(
    source_ll: &str,
    reflection: &ShaderInterface,
) -> Option<ToolingRequirement> {
    let threadgroup = matches!(
        reflection.stage,
        ShaderStage::Kernel | ShaderStage::Object | ShaderStage::Mesh
    );
    reflection
        .entry_point
        .as_deref()
        .is_some_and(|entry| fences(source_ll, entry, threadgroup))
        .then_some(ToolingRequirement::SeqCstFence)
}

/// Whether a call in the function `entry` of `source_ll`, or in a function it reaches by direct
/// calls, takes the execution status word ([`takes_the_status_word`]); `threadgroup` says whether
/// the stage has a threadgroup. A module that does not define `entry` has none.
pub fn fences(source_ll: &str, entry: &str, threadgroup: bool) -> bool {
    if !source_ll.contains("@air.atomic.fence")
        && !source_ll.contains("@air.wg.barrier")
        && !source_ll.contains("@air.simdgroup.barrier")
    {
        return false;
    }
    let bodies = function_bodies(source_ll);
    let Some((&entry, _)) = bodies.get_key_value(entry) else {
        return false;
    };
    let mut reached = BTreeSet::from([entry]);
    let mut pending = vec![entry];
    while let Some(function) = pending.pop() {
        for callee in bodies[function]
            .iter()
            .filter_map(|text| direct_callee(text))
        {
            if let Some((&callee, _)) = bodies.get_key_value(callee) {
                if reached.insert(callee) {
                    pending.push(callee);
                }
            }
        }
    }
    reached
        .iter()
        .flat_map(|function| &bodies[function])
        .any(|text| takes_the_status_word(text, threadgroup))
}

/// Whether the instruction `text` is a barrier or fence call in a form the translator orders
/// through the execution status word (the module documentation's).
pub fn takes_the_status_word(text: &str, threadgroup: bool) -> bool {
    let Some(at) = text.find("call ") else {
        return false;
    };
    let call = &text[at..];
    let Some(name_start) = call.find("@air.").map(|at| at + 1) else {
        return false;
    };
    let Some(name_length) = call[name_start..].find('(') else {
        return false;
    };
    let Some(AirIntrinsic::Sync(sync)) =
        AirIntrinsic::parse(&call[name_start..name_start + name_length])
    else {
        return false;
    };
    let arguments = &call[name_start + name_length + 1..];
    let Some(arguments) = arguments.rfind(')').map(|end| &arguments[..end]) else {
        return false;
    };
    let immediates: Option<Vec<u64>> = super::split_top_level(arguments)
        .into_iter()
        .map(|operand| operand.trim().strip_prefix("i32 ")?.trim().parse().ok())
        .collect();
    let Some(&[flags, order, scope]) = immediates.as_deref() else {
        return false;
    };
    if order != SEQ_CST || flags & ORDERED_MEMORIES == 0 {
        return false;
    }
    match sync {
        Sync::Fence { .. } => scope != THREAD && (scope != THREADGROUP || threadgroup),
        Sync::Barrier { .. } => matches!(scope, DEVICE | ALL_DEVICES),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(body: &str) -> String {
        format!(
            "declare void @air.atomic.fence(i32, i32, i32)\n\
             declare void @air.wg.barrier(i32, i32)\n\
             define void @k() {{\nentry:\n{body}\n  ret void\n}}\n"
        )
    }

    /// The forms the translator orders through the status word are claimed, at each scope the
    /// fence orders beyond its thread, and in a function the entry calls.
    #[test]
    fn a_seq_cst_fence_beyond_its_thread_is_claimed() {
        for scope in [1, 2, 3, 4] {
            let text = module(&format!(
                "  call void @air.atomic.fence(i32 1, i32 5, i32 {scope})"
            ));
            assert!(fences(&text, "k", true), "scope {scope}");
        }
        let barrier = module("  call void @air.wg.barrier(i32 2, i32 5, i32 2)");
        assert!(fences(&barrier, "k", true));
        let callee = "declare void @air.atomic.fence(i32, i32, i32)\n\
                      define internal void @f() {\nentry:\n  call void @air.atomic.fence(i32 4, i32 5, i32 2)\n  ret void\n}\n\
                      define void @k() {\nentry:\n  call void @f()\n  ret void\n}\n";
        assert!(fences(callee, "k", true));
    }

    /// No claim for a fence the translator emits without the word: another order, the thread's
    /// scope, no memory, a threadgroup scope in a stage without one, a barrier of the AIR 2.8 form
    /// or of a threadgroup memory scope, a call outside the entry's reach, or a run-time operand.
    #[test]
    fn a_fence_the_translator_orders_without_the_word_is_no_claim() {
        let unclaimed = [
            "  call void @air.atomic.fence(i32 1, i32 4, i32 2)",
            "  call void @air.atomic.fence(i32 1, i32 0, i32 2)",
            "  call void @air.atomic.fence(i32 1, i32 5, i32 0)",
            "  call void @air.atomic.fence(i32 0, i32 5, i32 2)",
            "  call void @air.atomic.fence(i32 16, i32 5, i32 2)",
            "  call void @air.wg.barrier(i32 1, i32 1)",
            "  call void @air.wg.barrier(i32 1, i32 5, i32 1)",
            "  call void @air.wg.barrier(i32 1, i32 4, i32 2)",
            "  call void @air.atomic.fence(i32 1, i32 %order, i32 2)",
        ];
        for body in unclaimed {
            assert!(!fences(&module(body), "k", true), "{body}");
        }
        let threadgroup = module("  call void @air.atomic.fence(i32 2, i32 5, i32 1)");
        assert!(fences(&threadgroup, "k", true));
        assert!(!fences(&threadgroup, "k", false));
        let unreached = "declare void @air.atomic.fence(i32, i32, i32)\n\
                         define internal void @f() {\nentry:\n  call void @air.atomic.fence(i32 1, i32 5, i32 2)\n  ret void\n}\n\
                         define void @k() {\nentry:\n  ret void\n}\n";
        assert!(!fences(unreached, "k", true));
        assert!(!fences(&module(""), "missing", true));
    }

    /// The two public fixtures whose runs the runtime refuses for the word carry it: a
    /// device-scope `seq_cst` fence between two stores, and MSL 4.1's barrier of `seq_cst` order
    /// at the device's memory scope.
    #[test]
    fn the_public_fixtures_the_runtime_refuses_for_the_word_are_claimed() {
        let engine = metal2vulkan_devtools::engine_dir();
        for (name, entry) in [
            ("a4_device_fence.ll", "a4_device_fence"),
            ("a27_memory_order.ll", "a27_memory_order"),
        ] {
            let text = std::fs::read_to_string(engine.join(name)).unwrap();
            assert!(fences(&text, entry, true), "{name}");
        }
    }
}
