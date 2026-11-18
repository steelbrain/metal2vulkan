//! Whether a kernel's memory work is images alone, decided from the specialized entry's AIR alone:
//! `KernelFacts::image_only`.
//!
//! A kernel is image-only when it reads an image and loads, stores and updates atomically nothing
//! in device memory:
//!
//! - it reads an image: a texture sample, gather, read or block read;
//! - no load, store, read-modify-write or compare-exchange goes through a pointer into the
//!   `device` address space, or a generic one;
//! - no call takes such a pointer, but a texture or sampler operation, whose texture operand is
//!   one: an atomic, coherent, simdgroup-matrix or asynchronous-copy intrinsic, an `llvm.mem*`
//!   copy or fill, and any call the entry cannot see into.
//!
//! Loads of `constant` memory, a thread's own memory and texture writes are allowed. A runtime may
//! run such a kernel that does not observe its threadgroup partition in square workgroups.

use crate::air;
use crate::air::abi::Stage as AbiStage;
use crate::air::intrinsics::{AirIntrinsic, TexOp};
use crate::air::ir::{AddrSpace, Callee, InstKind, Type, Value};

/// Whether entry `e` is image-only (see the module). Any other stage, and an entry without a body,
/// is not.
pub(crate) fn image_only(e: &air::Entry) -> bool {
    if !matches!(e.ep.stage, AbiStage::Kernel { .. }) {
        return false;
    }
    let Some(body) = e.module.func(e.func).and_then(|f| f.body.as_ref()) else {
        return false;
    };
    let device = |v: Value| {
        matches!(
            e.module
                .value_ty(e.func, v)
                .and_then(|t| e.module.types().get(t)),
            Some(Type::Ptr {
                space: AddrSpace::Device | AddrSpace::Generic
            })
        )
    };
    let mut reads = false;
    for inst in body.insts.values() {
        let touches = match &inst.kind {
            InstKind::Load { ptr, .. }
            | InstKind::Store { ptr, .. }
            | InstKind::AtomicRmw { ptr, .. }
            | InstKind::CmpXchg { ptr, .. } => device(*ptr),
            InstKind::Call(call) => match &call.callee {
                Callee::Air {
                    op: AirIntrinsic::Texture(t),
                    ..
                } => {
                    reads |= matches!(
                        t.op,
                        TexOp::Sample { .. }
                            | TexOp::SampleGrad { .. }
                            | TexOp::Gather { .. }
                            | TexOp::Read
                            | TexOp::BlockRead
                    );
                    false
                }
                Callee::Air {
                    op: AirIntrinsic::Sampler(_),
                    ..
                } => false,
                Callee::Air { .. } | Callee::Llvm { .. } => call.args.iter().any(|&a| device(a)),
                _ => true,
            },
            _ => false,
        };
        if touches {
            return false;
        }
    }
    reads
}
