//! What a function reads and writes through anything but its bound slots, decided from the
//! specialized entry's AIR and its memory plan: `StageInterface::reads_unbound_resources`,
//! `StageInterface::writes_unbound_resources` and `StageInterface::unbound_storage_textures`.
//!
//! A bound slot is what the encoder sets: a buffer, texture or sampler argument, an element of a
//! buffer-table argument, and the words of an argument buffer bound at a buffer index. An access
//! is through something else where the pointer or texture it goes through may come from memory or
//! an integer (a root set holding `Root::Dynamic`: an argument buffer's buffers and textures, an
//! address stored in a buffer, a resource ID, `inttoptr`), or from an intrinsic call
//! (`Root::Call`). Metal requires such resources to be declared with `useResource` or `useHeap`;
//! the runtime orders them by these two facts. The accesses are:
//!
//! - loads, stores and atomics of device and constant memory, and the coherent, simdgroup-matrix,
//!   asynchronous-copy and `llvm.mem*` forms of them;
//! - texture operations that touch texels: samples, gathers, reads and block reads (reads),
//!   writes and imageblock slices (writes), atomics (both). Queries, null tests and fences touch
//!   no texel.
//!
//! Some operations are both, whatever their operands: an indirect command buffer's operations
//! (its records are only reached through an argument buffer), ray tracing, and a call the
//! entry cannot see into (a visible function's, an unknown intrinsic).
//!
//! Of the texture operations through such a texture, those that reach it through its storage
//! image are storage accesses: a write, an atomic, and a read of an `access::write` or
//! `access::read_write` texture (the access the operation carries). An intersection and a call
//! the entry cannot see into are storage accesses too; an indirect command buffer's operations
//! and the other ray tracing operations touch no texture.

use crate::air;
use crate::air::intrinsics::{
    Access, AccessKind, AirIntrinsic, AtomicOp, LlvmIntrinsic, SimdMatrix, TexOp,
};
use crate::air::ir::{AddrSpace, Callee, InstKind, Type, Value};
use crate::memory::plan::MemoryPlan;
use crate::memory::provenance::{HandleClass, Repr, Root};

/// What a function reads and writes through anything but its bound slots (see the module).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Unbound {
    pub(super) reads: bool,
    pub(super) writes: bool,
    /// Some access through such a texture reaches it through its storage image.
    pub(super) storage_textures: bool,
}

impl Unbound {
    const ALL: Unbound = Unbound {
        reads: true,
        writes: true,
        storage_textures: true,
    };

    fn add(&mut self, reads: bool, writes: bool) {
        self.reads |= reads;
        self.writes |= writes;
    }
}

/// What entry `e` reads and writes through anything but its bound slots (see the module).
pub(super) fn unbound(e: &air::Entry, memory: &MemoryPlan) -> Unbound {
    let Some(body) = e.module.func(e.func).and_then(|f| f.body.as_ref()) else {
        return Unbound::default();
    };
    let space = |v: Value| match e
        .module
        .value_ty(e.func, v)
        .map(|t| e.module.types().get(t))
    {
        Some(Some(Type::Ptr { space })) => Some(*space),
        _ => None,
    };
    let buffer = |v: Value| matches!(space(v), Some(AddrSpace::Device | AddrSpace::Constant));
    let texture = |v: Value| {
        matches!(
            memory.repr(v),
            Some(Repr::Handle {
                class: HandleClass::Texture,
                ..
            })
        )
    };
    // A pointer or handle that may come from memory, an integer or an intrinsic call.
    let loaded = |v: Value| {
        memory
            .roots_of(v)
            .is_some_and(|r| r.iter().any(|r| matches!(r, Root::Dynamic | Root::Call(_))))
    };
    let through = |v: Value| buffer(v) && loaded(v);
    let mut out = Unbound::default();
    for inst in body.insts.values() {
        match &inst.kind {
            InstKind::Load { ptr, .. } => out.add(through(*ptr), false),
            InstKind::Store { ptr, .. } => out.add(false, through(*ptr)),
            InstKind::AtomicRmw { ptr, .. } | InstKind::CmpXchg { ptr, .. } => {
                let t = through(*ptr);
                out.add(t, t);
            }
            InstKind::Call(call) => {
                let args = &call.args;
                let any = |f: &dyn Fn(Value) -> bool| args.iter().any(|a| f(*a));
                let buffers = any(&through);
                match &call.callee {
                    Callee::Air { op, .. } => match op {
                        AirIntrinsic::Atomic(a) => {
                            out.add(buffers, buffers && a.op != AtomicOp::Load)
                        }
                        AirIntrinsic::Coherent(c) => {
                            let store = c.kind == AccessKind::Store;
                            out.add(buffers && !store, buffers && store);
                        }
                        AirIntrinsic::SimdMatrix(m) => {
                            let store = matches!(
                                m,
                                SimdMatrix::Store8x8 { .. } | SimdMatrix::Store8x8Legacy { .. }
                            );
                            out.add(buffers && !store, buffers && store);
                        }
                        AirIntrinsic::Texture(t) => {
                            let handle = any(&|a| texture(a) && loaded(a));
                            let (reads, writes) = match t.op {
                                TexOp::Write | TexOp::WriteImageblockSlice => (false, true),
                                TexOp::Atomic(_) => (true, true),
                                TexOp::Query(_)
                                | TexOp::IsNull
                                | TexOp::GetNull
                                | TexOp::Fence
                                | TexOp::CalculateLod { .. } => (false, false),
                                TexOp::Sample { .. }
                                | TexOp::SampleGrad { .. }
                                | TexOp::Gather { .. }
                                | TexOp::Read
                                | TexOp::BlockRead => (true, false),
                            };
                            out.add(handle && reads, handle && writes);
                            let storage =
                                matches!(t.access, Some(Access::Write | Access::ReadWrite));
                            out.storage_textures |= handle && (writes || (reads && storage));
                        }
                        AirIntrinsic::IndirectCommand(_)
                        | AirIntrinsic::AccelerationStructure(_) => out.add(true, true),
                        AirIntrinsic::Intersect(_) => out = Unbound::ALL,
                        // An asynchronous copy writes one side and reads the other; any other
                        // intrinsic given such a pointer may only read through it.
                        AirIntrinsic::AsyncCopy(_) => out.add(buffers, buffers),
                        _ => out.add(buffers, false),
                    },
                    Callee::Llvm { op, .. } => match op {
                        LlvmIntrinsic::Memcpy { .. } | LlvmIntrinsic::Memmove { .. } => {
                            let at = |i: usize| args.get(i).is_some_and(|a| through(*a));
                            out.add(at(1), at(0));
                        }
                        LlvmIntrinsic::Memset { .. } => {
                            out.add(false, args.first().is_some_and(|a| through(*a)))
                        }
                        _ => out.add(buffers, buffers),
                    },
                    Callee::Direct(_)
                    | Callee::Indirect { .. }
                    | Callee::UnknownIntrinsic(_)
                    | Callee::Unbound { .. } => out = Unbound::ALL,
                }
            }
            _ => {}
        }
        if out == Unbound::ALL {
            break;
        }
    }
    out
}
