//! How Metal reads `f32` denormals, per operation, against the device: where the device reads them
//! as Metal does, an `f32` denormal operand as the zero of its sign and a denormal result flushed
//! to one (`DriverDeviation::MetalFloatDenormals`, a measured driver fact), the module leaves them
//! to it and requires the property; elsewhere they are the device's own, noted
//! (`PrecisionNote::DeviceDenormals`). Only `float` operations are listed: a widened `half` is
//! never an `f32` denormal, and Metal's `bfloat` and `double` environments are not measured.

use crate::device::DeviceProfile;
use crate::error::DriverDeviation;
use crate::lower::arith::math::FloatFn;

/// Whether profile `p`'s device reads the denormals of the operations [`Covered::Deviation`]
/// names as Metal does: an `f32` denormal operand as the zero of its sign, a denormal result
/// flushed to one (`DriverDeviation::MetalFloatDenormals`, a measured driver fact). Elsewhere
/// they are the device's own (`PrecisionNote::DeviceDenormals`).
pub fn device_flushes(p: &DeviceProfile) -> bool {
    p.takes_driver_deviation(DriverDeviation::MetalFloatDenormals)
}

/// Where an operation the device computes reads its `f32` denormals as Metal does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Covered {
    /// On a device with the measured deviation ([`device_flushes`]); elsewhere they are the
    /// device's own.
    Deviation,
    /// On no device measured: the device's own on every profile. `Floor` and `Ceil` of a denormal
    /// give `-1` and `1` on NVIDIA's 615.71.09 and on lavapipe, where Metal reads the operand as a
    /// zero (`floor(-2^-149)` is `-0`, `ceil(2^-149)` is `+0`).
    Never,
}

/// Where the device's instruction for float function `op` (on `float` lanes, where the lowering
/// leaves `op` to it) reads `f32` denormals as Metal, which reads each `float` operand of these
/// as the zero of its sign and flushes a denormal result (the denormal probes, `nan_denormals`
/// and `nan_denormals_fast`, every lane). `None` for a function whose lowering reads its
/// denormals as Metal does itself (`FloatFn::metal_denormals`) or whose reading is not measured.
///
/// | function | where the device's reading is Metal's |
/// |---|---|
/// | `floor`, `ceil` | no device measured |
/// | `trunc`, `rint` | the deviation |
/// | `sqrt`, `rsqrt`, `fast::rsqrt` | the deviation |
/// | `dot` | the deviation |
/// | `fma`, `llvm.fmuladd` (`OpFmaKHR`) | the deviation |
/// | `fast::atan2` | the deviation |
/// | `ldexp` on `float` | the deviation |
///
/// `fast::atan2(y, x)` reads a denormal `y` or `x` as the zero of its sign and answers no denormal
/// (an Apple M3 Max under macOS 27.0.1: every pair of 28 patterns, in entries with and without fast
/// math); its quotient is the device's division, whose operands are those two, and a denormal
/// quotient reaches the fast `atan`, which reads it as a zero.
///
/// `ldexp` on `float` (`math::ldexp_bits`) reads a denormal operand and rounds a result below
/// `2^-126` by the device's `float` multiply, which the deviation covers: a denormal operand times
/// `2^64`, and a normal times `2^-126`.
///
/// `trunc` and `rint` (`Trunc`, `RoundEven`) of a denormal answer the zero of its sign on every
/// venue measured, as Metal does (the probes' words, both modes, all seven venues), but Vulkan
/// states no sign for a flushed denormal, so only the deviation covers them.
pub fn coverage(op: FloatFn) -> Option<Covered> {
    use FloatFn as F;
    Some(match op {
        F::Floor | F::Ceil => Covered::Never,
        F::Trunc
        | F::Rint
        | F::Sqrt
        | F::Rsqrt
        | F::FastRsqrt
        | F::Dot
        | F::Fma
        | F::MulAdd
        | F::FastAtan2 { .. }
        | F::Ldexp => Covered::Deviation,
        _ => return None,
    })
}
