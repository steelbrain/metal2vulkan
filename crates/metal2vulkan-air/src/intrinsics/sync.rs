//! Barriers and fences: `air.wg.barrier`, `air.simdgroup.barrier`, `air.atomic.fence`.
//!
//! A barrier is `void (i32 mem_flags, i32 scope)` in AIR 2.8 and `void (i32 mem_flags, i32 order,
//! i32 scope)` in AIR 2.9 (deployment target macOS 27); a fence is `void (i32 mem_flags, i32 order,
//! i32 scope)`; every operand is an immediate. `mem_flags` is Metal's `mem_flags` mask
//! (`mem_device` 1, `mem_threadgroup` 2, `mem_texture` 4, `mem_threadgroup_imageblock` 8,
//! `mem_object_data` 16). A barrier's order is the `memory_order` MSL 4.1's barriers take (MSL
//! §6.10.1); Metal's headers pass `memory_order_seq_cst` to the barrier builtin in every earlier
//! language version, and AIR 2.8 drops the operand, so the 2-operand form is a `seq_cst` barrier.
//! Which form a declaration has is read from its type.
//!
//! The scopes differ by form. In AIR 2.8 a barrier's scope operand is both its **execution** and
//! its memory scope, read from the operand, never from the name: AIR also spells a
//! threadgroup-wide barrier through `air.simdgroup.barrier`. In AIR 2.9 the operand is MSL 4.1's
//! `thread_scope` argument, the **memory** scope alone (the threads that observe the ordered
//! accesses: MSL §6.10.1), and the name is the execution scope: `threadgroup_barrier`
//! (`air.wg.barrier`) synchronizes the threadgroup and `simdgroup_barrier`
//! (`air.simdgroup.barrier`) the simdgroup whatever scope they are given (Apple's compiler writes
//! `threadgroup_barrier(..., thread_scope_simdgroup)` as `air.wg.barrier` with scope 4). Metal's
//! headers pass the name's own scope in every language before MSL 4.1, so the two readings agree on
//! every barrier those spell. A value outside the modeled enumerations leaves the call unknown.
//!
//! In a kernel of Metal's OpenCL mode an `air.wg.barrier`'s flags are OpenCL C's
//! `cl_mem_fence_flags`, not Metal's mask ([`MemFlags::from_cl_mem_fence_flags`]): decode reads
//! every barrier as Metal's, and specialization reads such a kernel's again (`canon::opencl`).

use super::atomic::{MemoryOrder, ThreadScope};
use super::sig::{ImmKind, Immediates, Operand, Signature};
use crate::error::IntrinsicError;

/// Metal's `mem_flags`: which memories a barrier, fence, or AIR 2.9 atomic orders.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemFlags {
    /// `mem_device` (1): device memory.
    pub device: bool,
    /// `mem_threadgroup` (2): threadgroup memory.
    pub threadgroup: bool,
    /// `mem_texture` (4): texture memory.
    pub texture: bool,
    /// `mem_threadgroup_imageblock` (8): threadgroup imageblock memory.
    pub threadgroup_imageblock: bool,
    /// `mem_object_data`: an object function's payload (MSL §6.10.1, §6.16.3). No stage this
    /// translation compiles has that memory; `mem_flags`' every-flag value (31), which MSL 4.1's
    /// non-explicit atomics pass, includes it.
    pub object_data: bool,
}

impl MemFlags {
    /// The flags of mask `v`; `None` when a bit outside the five is set.
    pub fn from_raw(v: u64) -> Option<MemFlags> {
        (v & !0x1f == 0).then_some(MemFlags {
            device: v & 1 != 0,
            threadgroup: v & 2 != 0,
            texture: v & 4 != 0,
            threadgroup_imageblock: v & 8 != 0,
            object_data: v & 16 != 0,
        })
    }

    /// The flags of OpenCL C's `cl_mem_fence_flags` mask `v`, which a threadgroup barrier of a
    /// kernel of Metal's OpenCL mode takes in place of Metal's mask (`canon::opencl`):
    /// `CLK_LOCAL_MEM_FENCE` (1) is `mem_threadgroup` and `CLK_GLOBAL_MEM_FENCE` (2) is
    /// `mem_device`. `None` when a bit outside the two is set: Apple's `cl_kernel.h` names no
    /// other.
    pub(crate) fn from_cl_mem_fence_flags(v: u64) -> Option<MemFlags> {
        (v & !0x3 == 0).then_some(MemFlags {
            device: v & 2 != 0,
            threadgroup: v & 1 != 0,
            ..MemFlags::default()
        })
    }
}

/// Which intrinsic a barrier was spelled with: its execution scope in the AIR 2.9 form (in the
/// AIR 2.8 form that is the `scope` operand).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BarrierName {
    /// `air.wg.barrier`.
    Threadgroup,
    /// `air.simdgroup.barrier`.
    Simdgroup,
}

/// A decoded barrier or fence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Sync {
    /// An execution barrier over `scope` (threadgroup or simdgroup) that orders `flags`' memories
    /// among the threads of `memory`.
    Barrier {
        /// The intrinsic the barrier was spelled with.
        name: BarrierName,
        /// The `mem_flags` immediate: the memories the barrier orders.
        flags: MemFlags,
        /// The AIR 2.9 form's `memory_order` immediate; `None` for the AIR 2.8 form, which has
        /// no such operand and orders as `seq_cst` ([`Sync::barrier_order`]).
        order: Option<MemoryOrder>,
        /// The execution scope: the operand in the AIR 2.8 form, the name's in the AIR 2.9 form.
        scope: ThreadScope,
        /// The memory scope: the operand in either form (in the AIR 2.8 form, also `scope`).
        memory: ThreadScope,
    },
    /// A memory fence (`atomic_thread_fence`).
    Fence {
        /// The `mem_flags` immediate: the memories the fence orders.
        flags: MemFlags,
        /// The `memory_order` immediate.
        order: MemoryOrder,
        /// The `thread_scope` immediate: the threads that observe the ordering.
        scope: ThreadScope,
    },
}

impl Sync {
    pub(super) fn parse(rest: &str) -> Option<Sync> {
        let placeholder = (MemFlags::default(), ThreadScope::Threadgroup);
        let name = match rest {
            "wg.barrier" => BarrierName::Threadgroup,
            "simdgroup.barrier" => BarrierName::Simdgroup,
            "atomic.fence" => {
                return Some(Sync::Fence {
                    flags: placeholder.0,
                    order: MemoryOrder::Relaxed,
                    scope: placeholder.1,
                })
            }
            _ => return None,
        };
        Some(Sync::Barrier {
            name,
            flags: placeholder.0,
            order: None,
            scope: placeholder.1,
            memory: placeholder.1,
        })
    }

    /// The AIR 2.9 form of an AIR 2.8 barrier, the other declaration its name may have.
    pub(super) fn alternative(self) -> Option<Sync> {
        match self {
            Sync::Barrier {
                name,
                flags,
                order: None,
                scope,
                memory,
            } => Some(Sync::Barrier {
                name,
                flags,
                order: Some(MemoryOrder::SeqCst),
                scope,
                memory,
            }),
            _ => None,
        }
    }

    /// The order of a barrier: its `memory_order` operand, or `seq_cst` for the AIR 2.8 form
    /// (Metal's headers pass `memory_order_seq_cst` there). `None` for a fence.
    pub fn barrier_order(&self) -> Option<MemoryOrder> {
        match self {
            Sync::Barrier { order, .. } => Some(order.unwrap_or(MemoryOrder::SeqCst)),
            Sync::Fence { .. } => None,
        }
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        let i32s = Operand::Imm(ImmKind::I32);
        match self {
            Sync::Barrier { order: None, .. } => Signature::new(Operand::Void, &[i32s, i32s]),
            Sync::Barrier { order: Some(_), .. } | Sync::Fence { .. } => {
                Signature::new(Operand::Void, &[i32s, i32s, i32s])
            }
        }
    }

    /// Read the flags, order, and scopes. `Ok(None)`: a value outside the enumerations, or an AIR
    /// 2.8 barrier's scope other than threadgroup or simdgroup.
    pub(super) fn bind(self, imm: &Immediates<'_>) -> Result<Option<Sync>, IntrinsicError> {
        let Some(flags) = MemFlags::from_raw(imm.get(0)?) else {
            return Ok(None);
        };
        Ok(match self {
            Sync::Barrier {
                name, order: None, ..
            } => match ThreadScope::from_raw(imm.get(1)?) {
                Some(scope @ (ThreadScope::Threadgroup | ThreadScope::Simdgroup)) => {
                    Some(Sync::Barrier {
                        name,
                        flags,
                        order: None,
                        scope,
                        memory: scope,
                    })
                }
                _ => None,
            },
            // The AIR 2.9 form: the order sits between the flags and the memory scope, and the
            // execution scope is the name's.
            Sync::Barrier {
                name,
                order: Some(_),
                ..
            } => match (
                MemoryOrder::from_raw(imm.get(1)?),
                ThreadScope::from_raw(imm.get(2)?),
            ) {
                (Some(order), Some(memory)) => Some(Sync::Barrier {
                    name,
                    flags,
                    order: Some(order),
                    scope: match name {
                        BarrierName::Threadgroup => ThreadScope::Threadgroup,
                        BarrierName::Simdgroup => ThreadScope::Simdgroup,
                    },
                    memory,
                }),
                _ => None,
            },
            Sync::Fence { .. } => {
                let (order, scope) = (imm.get(1)?, imm.get(2)?);
                match (MemoryOrder::from_raw(order), ThreadScope::from_raw(scope)) {
                    (Some(order), Some(scope)) => Some(Sync::Fence {
                        flags,
                        order,
                        scope,
                    }),
                    _ => None,
                }
            }
        })
    }
}
