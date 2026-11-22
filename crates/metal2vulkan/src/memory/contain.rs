//! The bounds of a buffer-slot access at a run-time offset, and the side-effect guard.
//!
//! A descriptor view's access stays inside the range the binder supplied, by the in-shader check
//! on every profile: loads read at `select(inb, off, 0)` and yield `select(inb, v, 0)`; stores and
//! atomics sit in a builder-closed `if_(inb)`. No buffer robustness is enabled, so none bounds the
//! access in its place.
//!
//! An image or texel buffer is contained by the texture family: its reads by
//! [`device_reads_texels`], its writes by [`device_drops_texel_writes`], each in the shader where
//! the device's image robustness does not give Metal's answer, and a texel buffer's always.
//! Validation labels its runs by the same two predicates.
//!
//! The bounds check is normative ([`in_bounds`]): `inb = (size >= n) && (off <= size - n)`,
//! evaluated in the offset's width, never `off + n <= size`, which wraps. An offset is unsigned and
//! relative to the containment domain's base, so a wrapped GEP result is a huge offset and fails
//! the check.

use crate::device::DeviceProfile;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, ConstBits, Constant};

/// How one leaf access is contained.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Check {
    /// No in-shader check: a device-address access under the identity domains, which nothing
    /// bounds (`AccessPlan::psb_contained`).
    None,
    /// The normative in-shader check.
    InShader,
}

/// Whether the device's robustness on `p` returns Metal's out-of-bounds texel for a read of an
/// image, so the read is left to it: `robustImageAccess2`. Elsewhere, and for a texel buffer
/// (`buffer`), which no enabled buffer robustness bounds, the shader checks the read:
/// `robustImageAccess` alone reads a missing alpha as 0 or 1 and a fetch past the view's levels as
/// undefined.
pub fn device_reads_texels(p: &DeviceProfile, buffer: bool) -> bool {
    !buffer && p.fields().images.robust_image_access2
}

/// Whether the device's robustness on `p` discards an out-of-bounds write of an image, as Metal
/// does, so the write is left to it: `robustImageAccess` at either level. Elsewhere, and for a
/// texel buffer (`buffer`), which no enabled buffer robustness bounds, the shader guards the
/// write.
pub fn device_drops_texel_writes(p: &DeviceProfile, buffer: bool) -> bool {
    !buffer && p.fields().images.robust_image_access
}

/// An integer constant of the type of `like` (a 32- or 64-bit unsigned scalar).
pub fn int_like(fb: &mut FunctionBuilder<'_>, like: Value, v: u64) -> BResult<Value> {
    let ty = fb.value_ty(like)?;
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(v),
    })?;
    fb.module().const_value(c)
}

/// The normative check `(size >= n) && (off <= size - n)` for an `n`-byte access at `off` of a
/// domain of `size` bytes; `off` and `size` share one unsigned integer type (`u32` or `u64`).
pub fn in_bounds(fb: &mut FunctionBuilder<'_>, off: Value, size: Value, n: u64) -> BResult<Value> {
    let n = int_like(fb, size, n)?;
    let fits = fb.binary(BinOp::UGreaterThanEqual, size, n)?;
    let room = fb.binary(BinOp::ISub, size, n)?;
    let within = fb.binary(BinOp::ULessThanEqual, off, room)?;
    fb.binary(BinOp::LogicalAnd, fits, within)
}

#[cfg(test)]
mod tests {
    use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
    use crate::spirv::interp::harness::{
        address, check_containment, int_const, AccessArgs, ACCESS_SIZES,
    };
    use crate::spirv::ir::MemAccess;

    use super::*;

    /// The device-address form of the bounds guard over the interpreter harness: `select(inb, addr,
    /// sink)` then `select(inb, v, 0)`, with the buffer base standing in for the sink.
    fn psb_contained(fb: &mut FunctionBuilder<'_>, a: &AccessArgs) -> BResult<Value> {
        let inb = in_bounds(fb, a.offset, a.size, u64::from(a.n))?;
        let zero = int_const(fb, a, 0)?;
        let off = fb.select(inb, a.offset, zero)?;
        let p = address(fb, &AccessArgs { offset: off, ..*a })?;
        let v = fb.load(
            p,
            MemAccess {
                aligned: Some(1),
                ..MemAccess::default()
            },
        )?;
        let null = fb.module().constant(Constant::Null { ty: a.elem })?;
        let null = fb.module().const_value(null)?;
        fb.select(inb, v, null)
    }

    /// The store form: a store inside `if_(inb)`, here recording the loaded value in a variable
    /// that stays zero when the access is out of bounds.
    fn guarded(fb: &mut FunctionBuilder<'_>, a: &AccessArgs) -> BResult<Value> {
        let inb = in_bounds(fb, a.offset, a.size, u64::from(a.n))?;
        let null = fb.module().constant(Constant::Null { ty: a.elem })?;
        let var = fb.local_var(a.elem, Some(null))?;
        let result = fb.var_ptr(var)?;
        let args = *a;
        let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
            let p = address(fb, &args)?;
            let v = fb.load(
                p,
                MemAccess {
                    aligned: Some(1),
                    ..MemAccess::default()
                },
            )?;
            fb.store(result, v, MemAccess::default())?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        fb.if_(&mut (), inb, then, None)?;
        fb.load(result, MemAccess::default())
    }

    /// The L0 test: offsets around the buffer end and at `0xFFFF_FFF0..=0xFFFF_FFFF` (and the top
    /// of the 64-bit range), `n ∈ {1, 2, 4, 16}`, in both offset widths, never trap and yield the
    /// in-bounds bytes exactly when the normative check holds, for both encodings.
    #[test]
    fn both_encodings_contain_offsets_near_2_to_the_32() {
        assert_eq!(ACCESS_SIZES, [1, 2, 4, 16]);
        for size in [16u64, 22, 64] {
            check_containment(&psb_contained, size).expect("select encoding contains");
            check_containment(&guarded, size).expect("if_ encoding contains");
        }
    }
}
