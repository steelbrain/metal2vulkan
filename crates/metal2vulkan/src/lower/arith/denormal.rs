//! How Metal reads `f32` denormals, per operation, against the device: Metal reads an `f32`
//! denormal operand as the zero of its sign and flushes a denormal result to one, where the device
//! keeps or flushes them by its own float controls, noted (`PrecisionNote::DeviceDenormals`). Only
//! `float` operations are listed: a widened `half` is never an `f32` denormal, and Metal's `bfloat`
//! and `double` environments are not measured.

use crate::lower::arith::math::FloatFn;

/// Whether Metal's reading of the `f32` denormals of float function `op` (on `float` lanes, where
/// the lowering leaves `op` to the device's instruction) is measured: Metal reads each `float`
/// operand of these as the zero of its sign and flushes a denormal result (the denormal probes,
/// `nan_denormals` and `nan_denormals_fast`, every lane), where the device's reading is its own,
/// noted `DeviceDenormals`. False for a function whose lowering reads its denormals as Metal does
/// itself (`FloatFn::metal_denormals`) or whose reading is not measured. The functions: `floor`,
/// `ceil`, `trunc`, `rint`, `sqrt`, `rsqrt`, `fast::rsqrt`, `dot`, `fma` and `llvm.fmuladd`
/// (`OpFmaKHR`), `fast::atan2`, and `ldexp` on `float`.
///
/// `fast::atan2(y, x)` reads a denormal `y` or `x` as the zero of its sign and answers no denormal
/// (an Apple M3 Max under macOS 27.0.1: every pair of 28 patterns, in entries with and without fast
/// math); its quotient is the device's division, whose operands are those two, or, in a call
/// without every fast-math flag, `y` times the device's reciprocal of `x`, a denormal Metal flushes
/// past `2^126`; a denormal quotient reaches the fast `atan`, which reads it as a zero.
///
/// `ldexp` on `float` (`math::ldexp_bits`) reads a denormal operand and rounds a result below
/// `2^-126` by the device's `float` multiply: a denormal operand times
/// `2^64`, and a normal times `2^-126`.
///
/// `trunc` and `rint` (`Trunc`, `RoundEven`) of a denormal answer the zero of its sign on every
/// venue measured, as Metal does (the probes' words, both modes, all seven venues), but Vulkan
/// states no sign for a flushed denormal, so the device's is noted.
pub fn covered(op: FloatFn) -> bool {
    use FloatFn as F;
    matches!(
        op,
        F::Floor
            | F::Ceil
            | F::Trunc
            | F::Rint
            | F::Sqrt
            | F::Rsqrt
            | F::FastRsqrt
            | F::Dot
            | F::Fma
            | F::MulAdd
            | F::FastAtan2 { .. }
            | F::Ldexp
    )
}
