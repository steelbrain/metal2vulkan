//! The decode of a heap texture of a form a heap reads: an argument buffer's `BGR10_XR`,
//! `BGRA10_XR`, `R10Unorm_X6`, `RG10Unorm_X12`, `R16Bfloat`, `RG16Bfloat` or `RGBA16Bfloat` texture
//! (`ExtendedRange::HEAP_READ`).
//!
//! No Vulkan format holds these texels, so a slot reads them through a view of their raw unsigned
//! integer format and decodes them under its variant. A heap's element is any texture an argument
//! buffer names, known at run time alone, so the binder holds such a texture's raw view in the
//! heap's unsigned integer class of its shape and its float class's neutral (`bind::heap`,
//! `Fit::Raw`), and its handle-table entry's class index names its form past the heap's own
//! classes (`metal::texel_format::heap_extended_base`). A float read of a heap group that may
//! meet one ([`HeapPlan::reads_raw`](super::slot::HeapPlan)) tests that index; where it names a
//! form, it fetches the raw texel from that class's array at the same element and calls this
//! module's routine, which decodes it as a slot's read does (`extended_range::decode`, its `half`
//! rounding included) and selects Metal's out-of-bounds texel outside the texture: zero, alpha 1
//! for a form without alpha.

use crate::metal::format::ExtendedRange;
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::FuncId;
use crate::spirv::ir::{BinOp, ConstBits, Constant, Type as SType};

/// The census name of the routine's words, once per module and result type.
pub const CENSUS_NAME: &str = "texture.read.heap.raw_routine";

/// The routine's words: each form's decode, its select, and the out-of-bounds texel.
pub const ROUTINE_WORDS: u64 = ExtendedRange::HEAP_READ.len() as u64
    * (crate::lower::arith::extended_range::DECODE_WORDS as u64 + 24)
    + 160;

/// The words a read adds to call it: the class index's test, the branch and its merge, the unsigned
/// element's access chain and load, its level and size queries, tests and clamps, the fetch, and
/// the call.
pub const READ_WORDS: u32 = 256;

/// The words a query adds: the class index's test, the branch and its merge, and the unsigned
/// element's access chain, load and query.
pub const QUERY_WORDS: u32 = 128;

/// Define the routine decoding raw texel `raw` (a `uvec4`, the raw format's lanes) of the form at
/// index `form` (a `u32`) of `ExtendedRange::HEAP_READ`, inside the texture where `inb` (a
/// `bool`) holds, else Metal's out-of-bounds texel, as a `vec4` of `f32` (a `half` result's lanes
/// rounded as a slot's read rounds them, where `half`).
pub fn define(mb: &mut ModuleBuilder, half: bool) -> BResult<FuncId> {
    let f32_ = mb.ty(SType::Float { width: 32 })?;
    let u32_ = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let bool_ = mb.ty(SType::Bool)?;
    let vf = mb.ty(SType::Vector { elem: f32_, len: 4 })?;
    let vu = mb.ty(SType::Vector { elem: u32_, len: 4 })?;
    let mut fb = mb.routine(vf)?;
    let raw = fb.param(vu)?;
    let form = fb.param(u32_)?;
    let inb = fb.param(bool_)?;
    let lanes = |fb: &mut crate::spirv::builder::FunctionBuilder<'_>, alpha: f32| {
        let mut parts = Vec::with_capacity(4);
        for v in [0.0f32, 0.0, 0.0, alpha] {
            parts.push(fb.module().constant(Constant::Scalar {
                ty: f32_,
                bits: ConstBits(u64::from(v.to_bits())),
            })?);
        }
        let c = fb.module().constant(Constant::Composite {
            ty: vf,
            parts: parts.into(),
        })?;
        fb.module().const_value(c)
    };
    let zero = lanes(&mut fb, 0.0)?;
    let one_alpha = lanes(&mut fb, 1.0)?;
    let mut v = zero;
    let mut alpha_one = None;
    for (k, &f) in (0u64..).zip(ExtendedRange::HEAP_READ.iter()) {
        let decoded = crate::lower::arith::extended_range::decode(&mut fb, f, raw, half)?;
        let c = fb.module().constant(Constant::Scalar {
            ty: u32_,
            bits: ConstBits(k),
        })?;
        let c = fb.module().const_value(c)?;
        let is = fb.binary(BinOp::IEqual, form, c)?;
        v = fb.select(is, decoded, v)?;
        // A form without alpha reads alpha 1 outside the texture, as inside.
        if !f.channels().contains(crate::metal::ColorWriteMask::ALPHA) {
            alpha_one = Some(match alpha_one {
                None => is,
                Some(any) => fb.binary(BinOp::LogicalOr, any, is)?,
            });
        }
    }
    let oob = match alpha_one {
        Some(a) => fb.select(a, one_alpha, zero)?,
        None => zero,
    };
    let r = fb.select(inb, v, oob)?;
    fb.ret(Some(r))?;
    fb.finish()
}
