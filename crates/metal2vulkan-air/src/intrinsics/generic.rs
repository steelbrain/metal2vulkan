//! MSL 4.1's address-space queries on generic pointers.

use super::sig::{Operand, Signature};
use super::ty::{Elem, NumTy};
use crate::ir::AddrSpace;

/// `air.is_{device,thread,threadgroup}_pointer`: `i1 (ptr addrspace(8))`, whether a generic pointer
/// points into one address space (MSL 4.1's `is_device_addrspace`, `is_thread_addrspace` and
/// `is_threadgroup_addrspace`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PointerSpace {
    /// The space asked about: `Device`, `Thread` or `Threadgroup`.
    pub space: AddrSpace,
}

impl PointerSpace {
    pub(super) fn parse(rest: &str) -> Option<PointerSpace> {
        let space = match rest.strip_prefix("is_")?.strip_suffix("_pointer")? {
            "device" => AddrSpace::Device,
            "thread" => AddrSpace::Thread,
            "threadgroup" => AddrSpace::Threadgroup,
            _ => return None,
        };
        Some(PointerSpace { space })
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        Signature::new(
            Operand::Num(NumTy::scalar(Elem::I1)),
            &[Operand::Ptr(AddrSpace::Generic)],
        )
    }
}
