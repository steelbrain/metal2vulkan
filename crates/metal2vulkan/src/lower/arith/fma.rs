//! Fused multiply-add on `float` and `half`.
//!
//! Metal's `fma` (`air.fma`, `llvm.fma`) rounds the exact `a * b + c` once. Vulkan's
//! `GLSL.std.450 Fma` does not promise that (its precision is inherited from an `OpFMul`
//! followed by an `OpFAdd`). A `float` fused multiply-add is the device's `GLSL.std.450 Fma`
//! ([`Route::Native`]) on every profile, within Vulkan's precision rather than rounded once
//! (`PrecisionNote::NativeFma`).
//!
//! A `half` one takes [`Route::F16Native`] where the profile computes `half` under the 16-bit
//! float controls
//! ([`narrow::half_controls`]): the device's `Fma` on 16-bit floats between the same exact
//! conversions, which the module's 16-bit holding then removes (`spirv::half_hold`), so the
//! operands and the result stay 16-bit floats. Its precision is Vulkan's for `Fma`, inherited from
//! an `OpFMul` followed by an `OpFAdd`, each correctly rounded to `half` under the controls'
//! `RoundingModeRTE`, where Metal's rounds once: noted `PrecisionNote::NativeFma`, as
//! [`Route::Native`] is. Every other `half` one takes [`Route::Native`]: the device's `float`
//! `Fma` of the operands held in `float`, whose product is exact there, rounded to `half`.

use crate::device::DeviceProfile;
use crate::lower::arith::legal::{FloatFmt, Form};
use crate::lower::arith::narrow;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::Value;
use crate::spirv::ir::{ConvOp, GlslOp, Type};

/// How a fused multiply-add computes its result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Route {
    /// A `half` fused multiply-add by the device's `Fma` on 16-bit floats under the 16-bit float
    /// controls: within Vulkan's precision rather than rounded once (`PrecisionNote::NativeFma`)
    /// ([`half_route`]).
    F16Native,
    /// The device's `GLSL.std.450 Fma` on the `float` lanes (a `half`'s held in `float`, its
    /// result rounded to `half`), within Vulkan's precision rather than rounded once
    /// (`PrecisionNote::NativeFma`).
    Native,
}

impl Route {
    /// Whether the route is the device's own `float` instruction (its `Fma`), at the device's
    /// denormals and NaN encoding.
    pub fn device_float(self) -> bool {
        self == Route::Native
    }
}

/// The census names of the fused multiply-add strategies (one per [`Route`]).
pub const F16_NATIVE: &str = "arith.fma.f16.native";
/// See [`F16_NATIVE`].
pub const NATIVE: &str = "arith.fma.native";

/// The `half` route profile `p` offers: [`Route::F16Native`] where the profile computes under the
/// 16-bit float controls (`narrow::half_controls`), else the device's `float` `Fma` rounded to
/// `half` ([`Route::Native`]).
pub fn half_route(p: &DeviceProfile) -> Route {
    match narrow::half_controls(p) {
        true => Route::F16Native,
        false => Route::Native,
    }
}

/// The routes a profile's fused multiply-adds take, by format ([`routes`]): a `float` one's is
/// [`Route::Native`] on every profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Routes {
    /// A `half` one's: [`half_route`].
    pub half: Route,
}

/// The routes profile `p` offers ([`half_route`]).
pub fn routes(p: &DeviceProfile) -> Routes {
    Routes {
        half: half_route(p),
    }
}

/// A fused multiply-add on floats of `form` (`float` or `half` lanes) by `route`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fma {
    pub form: Form,
    pub route: Route,
}

impl Fma {
    /// The fused multiply-add on `form` by `routes` (a profile's [`routes`]): `half` by its
    /// `half` route, `float` by the device's `Fma`. `None` for any other form.
    pub fn by(routes: Routes, form: Form) -> Option<Fma> {
        let route = match form.float_fmt()? {
            FloatFmt::Half => routes.half,
            FloatFmt::F32 => Route::Native,
            FloatFmt::BFloat | FloatFmt::F64 => return None,
        };
        Some(Fma { form, route })
    }

    /// The census name.
    pub fn name(self) -> &'static str {
        match self.route {
            Route::F16Native => F16_NATIVE,
            Route::Native => NATIVE,
        }
    }

    /// An upper estimate of the SPIR-V words per use: the instruction; on [`Route::F16Native`] the
    /// three conversions, the instruction and the widening.
    pub fn words(self) -> u32 {
        match self.route {
            Route::F16Native => 32,
            // The instruction, and a `half` result's rounding.
            Route::Native => 64,
        }
    }
}

/// Emit `fma` on `args` (three floats of `fma.form`). The result is a value of the form's format.
pub fn emit(fb: &mut FunctionBuilder<'_>, fma: Fma, args: &[Value]) -> BResult<Value> {
    let [a, b, c] = *args else {
        return Err(BuildError::Type(TypeRule::FunctionSignature));
    };
    match (fma.route, fma.form.float_fmt()) {
        (Route::F16Native, Some(FloatFmt::Half)) => f16(fb, fma.form.lanes, [a, b, c]),
        // The device's `Fma`; on `half` operands held in `float` the product is exact, so only
        // the sum's rounding and the rounding to `half` differ from Metal's one rounding.
        (Route::Native, Some(FloatFmt::F32)) => fb.glsl(GlslOp::Fma, &[a, b, c]),
        (Route::Native, Some(FloatFmt::Half)) => {
            let r = fb.glsl(GlslOp::Fma, &[a, b, c])?;
            narrow::round(fb, fma.form, r)
        }
        _ => Err(BuildError::Type(TypeRule::FunctionSignature)),
    }
}

/// The `half` fused multiply-add of `a`, `b`, `c` (`half` values held in `float` lanes of
/// `lanes` lanes) by the device's `Fma` on 16-bit floats ([`Route::F16Native`]): each operand
/// converted to 16 bits, exactly since it is a `half`, the instruction, and its result widened,
/// exactly, within Vulkan's precision for `Fma`, noted.
fn f16(fb: &mut FunctionBuilder<'_>, lanes: u8, [a, b, c]: [Value; 3]) -> BResult<Value> {
    let lanes_of = |fb: &mut FunctionBuilder<'_>, width: u8| -> BResult<_> {
        let e = fb.module().ty(Type::Float { width })?;
        match lanes {
            1 => Ok(e),
            len => fb.module().ty(Type::Vector { elem: e, len }),
        }
    };
    let (h, w) = (lanes_of(fb, 16)?, lanes_of(fb, 32)?);
    let narrow = |fb: &mut FunctionBuilder<'_>, x: Value| fb.convert(ConvOp::FConvert, x, h);
    let (a, b, c) = (narrow(fb, a)?, narrow(fb, b)?, narrow(fb, c)?);
    let r = fb.glsl(GlslOp::Fma, &[a, b, c])?;
    fb.convert(ConvOp::FConvert, r, w)
}
