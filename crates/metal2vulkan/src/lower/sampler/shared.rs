//! The shared emulation of a switch-at-use (SEMANTICS 1012).
//!
//! A switch-at-use whose every arm takes the same emulated sampling on its own texture slot
//! ([`Sampling::shared_shape`]): a bicubic sample. Emitted per arm, each arm repeats the whole
//! emulation (its position, weights, sums, and routine calls), so the switch's words grow with
//! arms × emulation, and so does the driver's work once the bound formats make every arm's
//! emulation live (an Intel ANV pipeline of 32 such arms took 6 s to create).
//!
//! **The shared form** runs the emulation once, outside the dispatch, on the named candidate's
//! facts, and dispatches only what reads a candidate's own image:
//!
//! - a facts dispatch, whose arms yield their level-0 size (`OpImageQuerySizeLod` 0), the bound
//!   format's bits (the slot's `BicubicTexels` specialization) and `true`, and
//!   whose null arm yields a size of ones, bits 0 and `false`;
//! - the emulation on those facts (its footprint, weights and routine calls once), whose taps a
//!   dispatch reads from each arm's image (the null arm's are zero).
//!
//! Under the identity domains an arm's image is its texture slot's element of the set-1 array at
//! the index its pushed resource ID holds, and its bits the slot's `PushField::BicubicTexels`, as
//! a candidate's own sample reads them.
//!
//! Where a handle names a candidate the facts are that candidate's, so every value is the one its
//! own arm computed; elsewhere the switch's null value is selected. The emulation of a null handle
//! runs on a size of ones and zero taps, whose value is discarded.

use crate::lower::emit::EmitCx;
use crate::lower::texture::{dispatch, with_word, NullLeaf};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{ConstBits, Constant, ImageQuery, Type as SType};

use super::bicubic::Bicubic;
use super::{BindingKey, CombinedImage, SampleOp, Sampling, Source};

/// The candidates of a switch-at-use that emulates once: the handles' values dispatched on,
/// outermost first, and each arm's case literals and sampling.
pub(crate) struct Candidates<'a> {
    pub(crate) sels: Vec<SValue>,
    pub(crate) arms: Vec<(&'a [u32], &'a Sampling)>,
}

impl Candidates<'_> {
    /// One dispatch on the handles: `read` yields each arm's values from its sampling in the arm,
    /// `null` the null arm's; the merged values, in order.
    pub(super) fn each(
        &self,
        cx: &mut EmitCx<'_>,
        fb: &mut FunctionBuilder<'_>,
        read: &dyn Fn(&mut EmitCx<'_>, &mut FunctionBuilder<'_>, &Sampling) -> BResult<Vec<SValue>>,
        null: &NullLeaf<'_>,
    ) -> BResult<Vec<SValue>> {
        let leaf =
            |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, a: &Sampling| read(cx, fb, a);
        dispatch(cx, fb, &self.sels, &self.arms, &leaf, null)
    }

    /// The image of arm sampling `a`'s binding, loaded in the open block: its combined binding's,
    /// or under the identity domains its texture slot's element of the set-1 array
    /// (`emit::identity_image`).
    pub(super) fn image(
        &self,
        cx: &mut EmitCx<'_>,
        fb: &mut FunctionBuilder<'_>,
        a: &Sampling,
    ) -> BResult<SValue> {
        let key = key(cx, a)?;
        if cx.plan.families.access.identity {
            return Ok(super::emit::identity_image(cx, fb, key)?.0);
        }
        match cx.decl.families.sampler.combined.get(&key) {
            Some(&(b, _)) => b.image(fb),
            None => Err(cx.fail(format!("combined binding {key:?} is not declared"))),
        }
    }
}

/// The combined binding sampling `a` reads.
fn key(cx: &mut EmitCx<'_>, a: &Sampling) -> BResult<BindingKey> {
    match a.source {
        Source::Combined(key) => Ok(key),
        _ => Err(cx.fail("a shared sampling of a heap element")),
    }
}

/// The image facts of the binding sampling `a` reads: its combined binding's, or under the
/// identity domains its identity sampling's.
fn image_facts(cx: &mut EmitCx<'_>, a: &Sampling) -> BResult<CombinedImage> {
    let key = key(cx, a)?;
    let image = match cx.plan.families.access.identity {
        true => (cx.plan.families.sampler.identity.get(&key)).map(|i| i.combined()),
        false => (cx.decl.families.sampler.combined.get(&key)).map(|&(_, image)| image),
    };
    image.ok_or_else(|| cx.fail(format!("combined binding {key:?} is not declared")))
}

/// The bicubic sample every arm of a shared switch takes ([`Sampling::shared_shape`]).
fn bicubic(s: &Sampling) -> Option<&Bicubic> {
    match &s.op {
        SampleOp::Bicubic(b) => Some(b),
        _ => None,
    }
}

/// The bound format's bits of arm sampling `a`'s slot, its bicubic sample's, read in the open
/// block (`bicubic::format_bits`).
fn bits(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, a: &Sampling) -> BResult<SValue> {
    let Some(b) = bicubic(a) else {
        return Err(cx.fail("a shared switch's arm without its bicubic sample"));
    };
    super::bicubic::format_bits(cx, fb, b.texture)?.value(fb)
}

/// The value of the shared switch over `c` (module docs): its sampling's value where a handle
/// names a candidate, else `null`'s.
pub(crate) fn shared_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    c: &Candidates<'_>,
    null: &NullLeaf<'_>,
) -> BResult<SValue> {
    let Some(&(_, first)) = c.arms.first() else {
        return Err(cx.fail("a shared switch without arms"));
    };
    let Some(b) = bicubic(first) else {
        return Err(cx.fail("a shared switch without an emulation"));
    };
    let image = image_facts(cx, first)?;
    for &(_, a) in &c.arms {
        if image_facts(cx, a)? != image {
            return Err(cx.fail("a shared switch's candidates of different images"));
        }
    }
    let facts = facts(cx, fb, c, image)?;
    let [size, bits, named] = facts[..] else {
        return Err(cx.fail("a shared switch's facts without their values"));
    };
    let x = super::bicubic::emit_shared(cx, fb, first, (size, image), b, bits, c)?;
    let x = with_word(cx, fb, &first.result, x, None)?;
    let [n] = null(cx, fb)?[..] else {
        return Err(cx.fail("a shared switch without its null value"));
    };
    fb.select(named, x, n)
}

/// The facts dispatch (module docs): each arm's level-0 size, its format's bits and `true`; the
/// null arm's ones, 0, and `false`.
fn facts(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    c: &Candidates<'_>,
    image: CombinedImage,
) -> BResult<Vec<SValue>> {
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let bool_ = fb.module().ty(SType::Bool)?;
    let scalar = |fb: &mut FunctionBuilder<'_>, ty, bits: u64| {
        fb.module().constant(Constant::Scalar {
            ty,
            bits: ConstBits(bits),
        })
    };
    let one = scalar(fb, u32_, 1)?;
    let lanes = image.plane + u8::from(image.arrayed);
    let ones = if lanes == 1 {
        one
    } else {
        let ty = fb.module().ty(SType::Vector {
            elem: u32_,
            len: lanes,
        })?;
        let parts = vec![one; usize::from(lanes)].into_boxed_slice();
        fb.module().constant(Constant::Composite { ty, parts })?
    };
    let zero = scalar(fb, u32_, 0)?;
    let yes = scalar(fb, bool_, 1)?;
    let no = scalar(fb, bool_, 0)?;
    let read = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, a: &Sampling| {
        let img = c.image(cx, fb, a)?;
        let level = fb.module().const_value(zero)?;
        let size = fb.query(ImageQuery::SizeLod(level), img)?;
        let bits = bits(cx, fb, a)?;
        Ok(vec![size, bits, fb.module().const_value(yes)?])
    };
    let null_facts = |_: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>| {
        Ok(vec![
            fb.module().const_value(ones)?,
            fb.module().const_value(zero)?,
            fb.module().const_value(no)?,
        ])
    };
    c.each(cx, fb, &read, &null_facts)
}
