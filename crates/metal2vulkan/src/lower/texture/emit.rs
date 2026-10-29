//! Emission of the texture strategies.

use crate::air::ir::{InstId, ValueId};
use crate::lower::access::imageblock;
use crate::lower::arith::narrow;
use crate::lower::emit::EmitCx;
use crate::lower::sync::barrier::{memory_model, BarrierModel};
use crate::pipeline::lower_inputs::StorageImageClass;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec, Merged};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{ConstId, FuncId, TypeId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, Dim, GlslOp, ImageOperands, ImageQuery, ImageUsage,
    MemAccess, MemoryOrder, Scope, Semantics, TexelMemory, Type as SType, UnOp,
};

use super::half::{TexelLanes, TexelOrigin};
use super::slice::Slice;
use super::{
    Coord, ImageDecl, ImageRef, QueryKind, QueryOp, Read, TexelClass, TexelConv, TexelForm,
};
use super::{NullArm, Prelude, Switch, TextureStrategy, Write};
use crate::memory::handles::{HANDLE_ALPHA_ONE, NEUTRAL_ELEMENT, NULL_HANDLE};

/// Emit strategy `s` of instruction `at` into the open block.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    at: InstId,
    s: &TextureStrategy,
) -> BResult<()> {
    match value(cx, fb, s, None)? {
        Some(v) => define(cx, at, v),
        None => Ok(()),
    }
}

/// The value of strategy `s` (`None` for one without a result), emitted into the open block.
/// `grads` are the gradients a switch-at-use hoisted for its sampling arms.
fn value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &TextureStrategy,
    grads: Option<(SValue, SValue)>,
) -> BResult<Option<SValue>> {
    Ok(Some(match s {
        TextureStrategy::Read(r) => read_crossed(cx, fb, r)?,
        TextureStrategy::Write(w) => {
            write(cx, fb, w)?;
            return Ok(None);
        }
        TextureStrategy::Atomic(a) => return super::atomic::emit(cx, fb, a),
        TextureStrategy::Query(q) => query(cx, fb, q)?,
        TextureStrategy::Fence => {
            fence(cx, fb)?;
            return Ok(None);
        }
        TextureStrategy::Sampling(x) => crate::lower::sampler::sampling_value(cx, fb, x, grads)?,
        TextureStrategy::IsNull { index } => {
            let Some(p) = cx.decl.families.texture.presence.get(index).copied() else {
                return Err(cx.fail("an is_null of a slot without a presence constant"));
            };
            fb.unary(UnOp::LogicalNot, p)?
        }
        TextureStrategy::HeapIsNull { handle } => {
            let h = cx.operand(fb, *handle)?;
            let e = heap_element_index(cx, fb, h)?;
            let null = u32s(fb, 1, NEUTRAL_ELEMENT)?;
            fb.binary(BinOp::IEqual, e, null)?
        }
        TextureStrategy::NullHandle => {
            let c = fb.u32_const(NULL_HANDLE)?;
            fb.module().const_value(c)?
        }
        TextureStrategy::Switch(sw) => return switch(cx, fb, sw),
        TextureStrategy::ImageblockSlice(sl) => {
            slice(cx, fb, sl)?;
            return Ok(None);
        }
        TextureStrategy::ExtendedSample(e) => crate::lower::sampler::extended::emit(cx, fb, e)?,
        TextureStrategy::InPlaceSample(e) => crate::lower::sampler::in_place::emit(cx, fb, e)?,
        TextureStrategy::AbsentArgument(a) => {
            if a.marks {
                crate::lower::access::absent::mark(cx, fb)?;
            }
            return Ok(null_value(cx, fb, &a.value)?.first().copied());
        }
    }))
}

/// A switch-at-use over the candidates of a multi-candidate texture or sampler: nested
/// `switch_at_use`s on each handle in turn, one arm per candidate combination, and the null arm for
/// every other handle value. A rule-2 sampling switch computes its gradients first, in the use's
/// own position; an LOD query's switch computes every arm's value there, and its arms only select
/// one. A sampling switch whose arms take one emulation on their own texture slots
/// ([`Switch::shared`]) emulates once, its arms reading only their images
/// (`sampler::shared_value`).
fn switch(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    sw: &Switch,
) -> BResult<Option<SValue>> {
    let grads = match &sw.prelude {
        Prelude::Gradients(h) => Some(crate::lower::sampler::hoisted_gradients(cx, fb, h)?),
        Prelude::Nothing | Prelude::Arms => None,
    };
    if let Some(shared) = sw.shared() {
        let sels = selectors(cx, fb, &sw.handles)?;
        let c = crate::lower::sampler::Candidates { sels, arms: shared };
        let null = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>| null_value(cx, fb, &sw.null);
        return Ok(Some(crate::lower::sampler::shared_value(
            cx, fb, &c, &null,
        )?));
    }
    let mut arms: Vec<(&[u32], Leaf<'_>)> = Vec::with_capacity(sw.arms.len());
    for (l, s) in &sw.arms {
        let leaf = match sw.prelude {
            // The absent handle's arm marks the status where it runs, so it stays in its arm; it
            // has no tangled instruction.
            Prelude::Arms if matches!(s, TextureStrategy::AbsentArgument(_)) => Leaf::Strategy(s),
            Prelude::Arms => match value(cx, fb, s, None)? {
                Some(v) => Leaf::Ready(v),
                None => return Err(cx.fail("an arm evaluated before its switch without a value")),
            },
            Prelude::Nothing | Prelude::Gradients(_) => Leaf::Strategy(s),
        };
        arms.push((l.as_slice(), leaf));
    }
    let sels = selectors(cx, fb, &sw.handles)?;
    let leaf = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, l: Leaf<'_>| match l {
        Leaf::Strategy(s) => Ok(value(cx, fb, s, grads)?.into_iter().collect()),
        Leaf::Ready(v) => Ok(vec![v]),
    };
    let null = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>| null_value(cx, fb, &sw.null);
    let merged = dispatch(cx, fb, &sels, &arms, &leaf, &null)?;
    Ok(merged.first().copied())
}

/// The values of a switch-at-use's `handles`, outermost first.
fn selectors(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    handles: &[ValueId],
) -> BResult<Vec<SValue>> {
    let mut sels = Vec::with_capacity(handles.len());
    for h in handles {
        sels.push(cx.operand(fb, *h)?);
    }
    Ok(sels)
}

/// What an arm of a switch-at-use yields: its strategy, emitted inside the arm, or the value
/// the switch computed before its dispatch.
#[derive(Clone, Copy)]
enum Leaf<'a> {
    Strategy(&'a TextureStrategy),
    Ready(SValue),
}

/// The value of the null arm.
fn null_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    null: &NullArm,
) -> BResult<Vec<SValue>> {
    Ok(match null {
        NullArm::Nothing => Vec::new(),
        NullArm::True => {
            let b = fb.module().ty(SType::Bool)?;
            let t = fb.module().constant(Constant::Scalar {
                ty: b,
                bits: ConstBits(1),
            })?;
            vec![fb.module().const_value(t)?]
        }
        NullArm::Zero(d) => {
            let Some(ty) = crate::lower::arith::spirv_type(fb.module(), d)? else {
                return Err(cx.fail("a null arm without a lowered result type"));
            };
            let z = fb.module().constant(Constant::Null { ty })?;
            vec![fb.module().const_value(z)?]
        }
    })
}

/// What a switch-at-use's arm yields in its own arm: an arm's values from its leaf `L`.
pub(crate) type ArmLeaf<'a, L> =
    dyn Fn(&mut EmitCx<'_>, &mut FunctionBuilder<'_>, L) -> BResult<Vec<SValue>> + 'a;

/// What a switch-at-use's null arm yields.
pub(crate) type NullLeaf<'a> =
    dyn Fn(&mut EmitCx<'_>, &mut FunctionBuilder<'_>) -> BResult<Vec<SValue>> + 'a;

/// Dispatch `arms` (each with one case literal per selector left in `sels`) on `sels[0]`,
/// recursing on the rest within each arm: `leaf` emits an arm's values in the arm, and `null`
/// the null arm's, which every other handle value takes; with no selector and no arm, the null
/// arm. The arms' values are merged in order.
pub(crate) fn dispatch<L: Copy>(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    sels: &[SValue],
    arms: &[(&[u32], L)],
    leaf: &ArmLeaf<'_, L>,
    null: &NullLeaf<'_>,
) -> BResult<Vec<SValue>> {
    let Some((&sel, rest)) = sels.split_first() else {
        return match arms {
            [(_, l)] => leaf(cx, fb, *l),
            // A constant null handle: nothing to dispatch on, the null arm alone.
            [] => null(cx, fb),
            _ => Err(cx.fail("a switch-at-use arm without exactly one strategy")),
        };
    };
    // The arms grouped by their first literal, in order.
    let mut groups: Vec<(u32, Vec<(&[u32], L)>)> = Vec::new();
    for (lits, s) in arms {
        let Some((&first, tail)) = lits.split_first() else {
            return Err(cx.fail("a switch-at-use arm without a literal"));
        };
        match groups.last_mut() {
            Some((l, g)) if *l == first => g.push((tail, *s)),
            _ => groups.push((first, vec![(tail, *s)])),
        }
    }
    let mut cases: Vec<(Vec<u32>, ArmFn<'_, EmitCx<'_>>)> = Vec::with_capacity(groups.len());
    for (lit, group) in groups {
        let arm: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
            let v = dispatch(cx, fb, rest, &group, leaf, null)?;
            Ok(ArmEnd::FallThrough(v))
        });
        cases.push((vec![lit], arm));
    }
    let default: ArmFn<'_, EmitCx<'_>> =
        Box::new(move |fb, cx| Ok(ArmEnd::FallThrough(null(cx, fb)?)));
    let merged = fb.switch_at_use(cx, sel, cases, default)?;
    Ok(merged.0)
}

fn define(cx: &mut EmitCx<'_>, at: InstId, x: SValue) -> BResult<()> {
    match cx.plan.values.inst(at) {
        Some(v) => {
            cx.define(v, x);
            Ok(())
        }
        None => Err(cx.fail(format!("{at:?} is out of the value range"))),
    }
}

fn u32_ty(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })
}

fn f32_ty(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    fb.module().ty(SType::Float { width: 32 })
}

fn u64_ty(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    fb.module().ty(SType::Int {
        width: 64,
        signed: false,
    })
}

/// Metal's `ulong` texel `(R | G << 32, 0, 0, 1)` of `RG32` image texel `x`: the low word in R and
/// the high word in G (measured on Apple M3 Max), and Metal's constant lanes past the first, which
/// out of bounds, where R and G are 0, are the same. `wide`: held as the low words `(R, 0, 0, 1)`
/// and the high words `(G, 0, 0, 0)`.
fn join_words(fb: &mut FunctionBuilder<'_>, x: SValue, wide: bool) -> BResult<SValue> {
    let lo = fb.composite_extract(x, &[0])?;
    let hi = fb.composite_extract(x, &[1])?;
    if wide {
        let u = u32_ty(fb)?;
        let uvec = vec(fb, u, 4)?;
        let zero = u32s(fb, 1, 0)?;
        let one = u32s(fb, 1, 1)?;
        let low = fb.composite_construct(uvec, &[lo, zero, zero, one])?;
        let high = fb.composite_construct(uvec, &[hi, zero, zero, zero])?;
        let ty = fb.module().ty(SType::Struct {
            members: vec![uvec, uvec].into(),
            layout: None,
        })?;
        return fb.composite_construct(ty, &[low, high]);
    }
    let u64_ = u64_ty(fb)?;
    let lo = fb.convert(ConvOp::UConvert, lo, u64_)?;
    let hi = fb.convert(ConvOp::UConvert, hi, u64_)?;
    let shift = splat(fb, u64_, 1, 32)?;
    let hi = fb.binary(BinOp::ShiftLeftLogical, hi, shift)?;
    let v = fb.binary(BinOp::BitwiseOr, lo, hi)?;
    let zero = splat(fb, u64_, 1, 0)?;
    let one = splat(fb, u64_, 1, 1)?;
    let ty = vec(fb, u64_, 4)?;
    fb.composite_construct(ty, &[v, zero, zero, one])
}

/// The `RG32` image texel `(low, high, 0, 0)` of lane 0 of held `ulong` texel `v`: Metal's write
/// stores that lane only, its low word in R and its high word in G. `wide`: `v` is held as its low
/// words and its high words.
fn split_words(fb: &mut FunctionBuilder<'_>, v: SValue, wide: bool) -> BResult<SValue> {
    let u = u32_ty(fb)?;
    let (lo, hi) = if wide {
        (
            fb.composite_extract(v, &[0, 0])?,
            fb.composite_extract(v, &[1, 0])?,
        )
    } else {
        let u64_ = u64_ty(fb)?;
        let x = fb.composite_extract(v, &[0])?;
        let lo = fb.convert(ConvOp::UConvert, x, u)?;
        let shift = splat(fb, u64_, 1, 32)?;
        let hi = fb.binary(BinOp::ShiftRightLogical, x, shift)?;
        (lo, fb.convert(ConvOp::UConvert, hi, u)?)
    };
    let zero = u32s(fb, 1, 0)?;
    let ty = vec(fb, u, 4)?;
    fb.composite_construct(ty, &[lo, hi, zero, zero])
}

/// Whether `decl` is a `ulong` slot declared `R64ui` (a `ulong` atomic updates it): its image
/// texels are 64-bit, one `ulong` each.
fn r64(decl: &ImageDecl) -> bool {
    decl.texel == TexelClass::Uint64 && decl.format == crate::spv::ImageFormat::R64ui
}

/// Metal's `ulong` texel `(R, 0, 0, 1)` of `R64` image texel `x`: the device's lanes past the
/// first, inside the image and outside it, are never read.
fn ulong_texel(fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<SValue> {
    let v = fb.composite_extract(x, &[0])?;
    let u64_ = u64_ty(fb)?;
    let zero = splat(fb, u64_, 1, 0)?;
    let one = splat(fb, u64_, 1, 1)?;
    let ty = vec(fb, u64_, 4)?;
    fb.composite_construct(ty, &[v, zero, zero, one])
}

/// A vector of `n` elements of `elem`, or `elem` itself for `n = 1`.
fn vec(fb: &mut FunctionBuilder<'_>, elem: TypeId, n: u8) -> BResult<TypeId> {
    if n == 1 {
        Ok(elem)
    } else {
        fb.module().ty(SType::Vector { elem, len: n })
    }
}

/// The constant `bits` of scalar type `elem`, splatted to `n` lanes.
fn splat(fb: &mut FunctionBuilder<'_>, elem: TypeId, n: u8, bits: u64) -> BResult<SValue> {
    let s = fb.module().constant(Constant::Scalar {
        ty: elem,
        bits: ConstBits(bits),
    })?;
    if n == 1 {
        return fb.module().const_value(s);
    }
    let ty = vec(fb, elem, n)?;
    let c = fb.module().constant(Constant::Composite {
        ty,
        parts: vec![s; usize::from(n)].into(),
    })?;
    fb.module().const_value(c)
}

fn u32s(fb: &mut FunctionBuilder<'_>, n: u8, v: u32) -> BResult<SValue> {
    let u = u32_ty(fb)?;
    splat(fb, u, n, u64::from(v))
}

fn f32s(fb: &mut FunctionBuilder<'_>, n: u8, v: f32) -> BResult<SValue> {
    let f = f32_ty(fb)?;
    splat(fb, f, n, u64::from(v.to_bits()))
}

/// A loaded image.
struct Loaded {
    decl: ImageDecl,
    img: SValue,
    /// A heap element's handle is not null (the null arm's value is then the operation's, as for a
    /// static null handle).
    live: Option<SValue>,
    /// The level of a slot's level views is one of the bound texture's.
    within: Option<SValue>,
    /// A heap element's handle, where it carries the out-of-bounds-alpha flag.
    flagged: Option<SValue>,
}

/// The declaration of `image`.
fn decl_of(cx: &EmitCx<'_>, image: &ImageRef) -> Option<ImageDecl> {
    let t = &cx.plan.families.texture;
    match *image {
        ImageRef::Slot(index) => t.slots.get(&index).and_then(|s| s.decl),
        ImageRef::Heap { group, .. } => t.heap.get(&group).and_then(|h| h.decl),
    }
}

/// Whether `decl` is a slot's 3D texture whose variant binds a two-plane extended-range format
/// (`ExtendedRange::Planes`). Metal addresses such a texture's texels as one 2D slice's: every
/// read and sample of it, at every coordinate and level, in bounds or not, takes its first
/// texel; a write at `(x, y, z)` of level 0 stores texel `(x, y)` of its first slice whatever
/// `z`, and one past the level's width or height or of another level stores nothing; its depth
/// query answers `⌈w / 4⌉` of its first level's width `w` at level 0, and at level `l`
/// [`level_extent`] of that (`max(1, ⌈w / 4⌉ >> (l & 255))`), and a one-level texture's
/// level count `1 - ⌊(w - 1) / 16⌋` (as a `uint`). Measured on an Apple M3 Max under macOS 27.0.1
/// (`Shared` and `Private`, in a heap and of its own memory; depths 1 to 64; every width from 1
/// to 130 at heights 1, 5, 64 and 300 for the queries). A mipmapped one's reads take bytes that
/// vary from run to run there; they read here as a one-level texture's.
fn planes_volume(decl: &ImageDecl) -> bool {
    decl.dim == Dim::D3
        && !decl.arrayed
        && matches!(
            decl.extended,
            Some(crate::metal::format::ExtendedRange::Planes { .. })
        )
}

/// The loaded image of `image`: for a slot's level views, the element of level `level` (`None`: the
/// constant 0, the only level of every other storage access).
fn image(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    image: &ImageRef,
    level: Option<ValueId>,
) -> BResult<Loaded> {
    image_in(cx, fb, image, level, None)
}

/// [`image`], a heap element read in numeric class `class` where given: through the group's
/// array of that class (`HeapPlan::cross`).
fn image_in(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    image: &ImageRef,
    level: Option<ValueId>,
    class: Option<TexelClass>,
) -> BResult<Loaded> {
    let index = match *image {
        ImageRef::Slot(index) => index,
        ImageRef::Heap { group, handle } => {
            let h = match class {
                Some(k) => heap_cross_element(cx, fb, (group, k), handle)?,
                None => heap_element(cx, fb, (group, false), handle)?,
            };
            return Ok(Loaded {
                decl: h.decl,
                img: h.img,
                live: Some(h.live),
                within: None,
                flagged: h.flagged,
            });
        }
    };
    let slot = cx.plan.families.texture.slots.get(&index);
    let (decl, at) = (slot.and_then(|s| s.decl), slot.map(|s| s.slot));
    let global = cx.decl.families.texture.images.get(&index).copied();
    let (Some(decl), Some(at), Some((g, _))) = (decl, at, global) else {
        return Err(cx.fail(format!("texture slot {index} is not declared")));
    };
    let ptr = fb.module().global_value(g)?;
    let Some(n) = decl.level_views else {
        if level.is_some() {
            return Err(cx.fail(format!(
                "a level of texture slot {index} without level views"
            )));
        }
        let img = fb.load(ptr, MemAccess::default())?;
        return Ok(Loaded {
            decl,
            img,
            live: None,
            within: None,
            flagged: None,
        });
    };
    let (img, within) = level_element(cx, fb, at, ptr, n, level)?;
    Ok(Loaded {
        decl,
        img,
        live: None,
        within: Some(within),
        flagged: None,
    })
}

/// Texture slot `index`'s image, loaded in the open block, with its declaration: an
/// extended-range sample fetches its raw texels from it (`sampler::extended`).
pub(crate) fn slot_image(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    index: u32,
) -> BResult<(ImageDecl, SValue)> {
    let Loaded { decl, img, .. } = image(cx, fb, &ImageRef::Slot(index), None)?;
    Ok((decl, img))
}

/// Element `level` (`None`: 0) of the `n` level views `ptr` points to, slot `slot`'s, and whether
/// the level is below the bound texture's level count (`PushField::TextureLevels`). A constant
/// level below `n` indexes the array as is (that element is a neutral where the texture lacks the
/// level); any other level is clamped below the count first (element 0, a neutral, for a count of
/// 0), so the access never leaves the array, and indexes it `NonUniform`.
fn level_element(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    slot: crate::pipeline::sym::SlotRef,
    ptr: SValue,
    n: u32,
    level: Option<ValueId>,
) -> BResult<(SValue, SValue)> {
    let count = cx.push_field(fb, crate::pipeline::sym::PushField::TextureLevels(slot))?;
    let l = match level {
        Some(l) => cx.operand(fb, l)?,
        None => u32s(fb, 1, 0)?,
    };
    let within = fb.binary(BinOp::ULessThan, l, count)?;
    let element = match fb.const_scalar(l)? {
        Some(k) if k < u64::from(n) => fb.access_chain(ptr, &[l])?,
        _ => {
            let one = u32s(fb, 1, 1)?;
            let some = fb.glsl(GlslOp::UMax, &[count, one])?;
            let top = fb.binary(BinOp::ISub, some, one)?;
            let index = fb.glsl(GlslOp::UMin, &[l, top])?;
            fb.access_chain_non_uniform(ptr, index)?
        }
    };
    let img = fb.load(element, MemAccess::default())?;
    Ok((img, within))
}

/// A heap element.
pub(crate) struct HeapElement {
    pub decl: ImageDecl,
    /// The image, loaded through a `NonUniform` access chain.
    pub img: SValue,
    /// The handle is live (not the null element).
    pub live: SValue,
    /// The handle, where the function's handle-table search carries the element's
    /// out-of-bounds-alpha flag in it (`memory::handles::Lookup::Flagged`): `heap_alpha_one` reads
    /// the flag.
    pub flagged: Option<SValue>,
}

/// The element of heap handle `h`: the handle masked to its element where the function's
/// handle-table search carries flags in it (`memory::handles::Lookup::element_mask`), else the
/// handle itself.
fn heap_element_index(cx: &EmitCx<'_>, fb: &mut FunctionBuilder<'_>, h: SValue) -> BResult<SValue> {
    let Some(element) = cx.plan.families.access.lookup().element_mask() else {
        return Ok(h);
    };
    let mask = u32s(fb, 1, element)?;
    fb.binary(BinOp::BitwiseAnd, h, mask)
}

/// Whether the flagged heap handle `h` ([`HeapElement::flagged`]) has [`HANDLE_ALPHA_ONE`] set:
/// its texture's pixel format has no alpha, so Metal reads alpha 1 out of bounds.
fn heap_alpha_one(fb: &mut FunctionBuilder<'_>, h: SValue) -> BResult<SValue> {
    let bit = u32s(fb, 1, HANDLE_ALPHA_ONE)?;
    let flag = fb.binary(BinOp::BitwiseAnd, h, bit)?;
    let zero = u32s(fb, 1, 0)?;
    fb.binary(BinOp::INotEqual, flag, zero)
}

/// Element `handle` of heap group `group`'s array: its declaration, the image loaded through a
/// `NonUniform` access chain, whether the handle is live (not the null element), and the handle
/// where it carries the out-of-bounds-alpha flag. A read without comparison (`plain`) of a group
/// whose comparisons the shader makes reads the group's float class (`HeapPlan::plain`); every
/// other access the group's own class.
pub(crate) fn heap_element(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    (group, plain): (u32, bool),
    handle: ValueId,
) -> BResult<HeapElement> {
    let h = cx.plan.families.texture.heap.get(&group);
    let declared = h.and_then(|h| h.plain.filter(|_| plain).or(h.class.zip(h.decl)));
    let Some(declared) = declared else {
        return Err(cx.fail(format!("heap group {group} is not declared")));
    };
    heap_element_of(cx, fb, declared, handle)
}

/// [`heap_element`] of heap group `group`'s array of numeric class `class`
/// (`HeapPlan::cross`), which holds the members of that class.
fn heap_cross_element(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    (group, class): (u32, TexelClass),
    handle: ValueId,
) -> BResult<HeapElement> {
    let h = cx.plan.families.texture.heap.get(&group);
    let Some(declared) = h.and_then(|h| h.cross.get(&class).copied()) else {
        return Err(cx.fail(format!(
            "heap group {group} has no {class:?} class declared"
        )));
    };
    heap_element_of(cx, fb, declared, handle)
}

/// The element of heap handle `handle` in the array of heap class `class`, declared `decl`.
fn heap_element_of(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    (class, decl): (crate::pipeline::sym::ImageHandleClass, ImageDecl),
    handle: ValueId,
) -> BResult<HeapElement> {
    let Some(g) = cx.decl.families.texture.heap.get(&class).copied() else {
        return Err(cx.fail(format!("heap class {class:?} is not declared")));
    };
    let base = fb.module().global_value(g)?;
    let h = cx.operand(fb, handle)?;
    let element = heap_element_index(cx, fb, h)?;
    let flagged = cx.plan.families.access.heap_alpha.then_some(h);
    let p = fb.access_chain_non_uniform(base, element)?;
    let img = fb.load(p, MemAccess::default())?;
    let null = u32s(fb, 1, NEUTRAL_ELEMENT)?;
    let live = fb.binary(BinOp::INotEqual, element, null)?;
    Ok(HeapElement {
        decl,
        img,
        live,
        flagged,
    })
}

/// `v` where `live` holds (a heap element's handle is not null), else the zero of `v`'s type.
pub(crate) fn live_or_zero(
    fb: &mut FunctionBuilder<'_>,
    live: Option<SValue>,
    v: SValue,
) -> BResult<SValue> {
    let Some(live) = live else { return Ok(v) };
    let ty = fb.value_ty(v)?;
    let z = fb.module().constant(Constant::Null { ty })?;
    let z = fb.module().const_value(z)?;
    fb.select(live, v, z)
}

/// The number of coordinate components of `decl`: its plane plus the array layer.
pub(super) fn coord_len(decl: &ImageDecl) -> u8 {
    let plane = match decl.dim {
        Dim::D1 | Dim::Buffer => 1,
        Dim::D2 | Dim::Cube | Dim::SubpassData => 2,
        Dim::D3 => 3,
    };
    plane + u8::from(decl.arrayed)
}

/// The greatest cube index whose layers `6 · cube + face` stay below 2^32.
const LAST_CUBE: u32 = (u32::MAX - 5) / 6;

/// The image coordinate of `c` (a `u32` or `uvecN`), with the array layer appended; a cube face
/// becomes the layer `face + 6 · cube` of the 2D-array view.
///
/// A cube's face at or past 6 is a layer at or past the cube view's 6. A cube array's face at or
/// past 6, or a cube index whose layer would pass 2^32, names no layer: Metal's access is out of
/// bounds there, so its layer is `u32::MAX`, past every view, which the bounds check or the
/// device's robustness takes as out of bounds, never as another cube's face.
pub(super) fn coordinate(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    decl: &ImageDecl,
    c: &Coord,
) -> BResult<SValue> {
    let plane = cx.operand(fb, c.plane)?;
    compose(cx, fb, decl, c, plane)
}

/// [`coordinate`] with the plane `plane`.
fn compose(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    decl: &ImageDecl,
    c: &Coord,
    plane: SValue,
) -> BResult<SValue> {
    let layer = match c.layer {
        Some(l) => Some(cx.operand(fb, l)?),
        None => None,
    };
    let face = match c.face {
        Some(f) => Some(cx.operand(fb, f)?),
        None => None,
    };
    let extra = match (face, layer) {
        (Some(f), Some(l)) => {
            let six = u32s(fb, 1, 6)?;
            let base = fb.binary(BinOp::IMul, l, six)?;
            let at = fb.binary(BinOp::IAdd, base, f)?;
            let face_ok = fb.binary(BinOp::ULessThan, f, six)?;
            let last = u32s(fb, 1, LAST_CUBE)?;
            let cube_ok = fb.binary(BinOp::ULessThanEqual, l, last)?;
            let ok = fb.binary(BinOp::LogicalAnd, face_ok, cube_ok)?;
            let none = u32s(fb, 1, u32::MAX)?;
            Some(fb.select(ok, at, none)?)
        }
        (Some(f), None) => Some(f),
        (None, l) => l,
    };
    let n = coord_len(decl);
    let coord = match extra {
        Some(e) => {
            let u = u32_ty(fb)?;
            let ty = vec(fb, u, n)?;
            fb.composite_construct(ty, &[plane, e])?
        }
        None => plane,
    };
    Ok(coord)
}

/// The size a clamp-to-edge read clamped its coordinate into: the image's size at the read's
/// level clamped into its levels (with the level count and that level), or its size.
#[derive(Clone, Copy)]
struct EdgeSize {
    size: SValue,
    levels: Option<(SValue, SValue)>,
}

/// The coordinate of clamp-to-edge read `r` of `img` (MSL 4.1 §6.13): its plane, a signed
/// coordinate plus its offset, clamped lane-wise into `[0, size - 1]` of the image's size at
/// `level` (clamped into the image's levels: an out-of-range level reads Metal's out-of-bounds
/// texel by its own check), then the layer and face as [`coordinate`] composes them, unclamped.
fn edge_coordinate(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    decl: &ImageDecl,
    img: SValue,
    r: &Read,
    level: Option<SValue>,
) -> BResult<(SValue, EdgeSize)> {
    let (Some(e), at) = (r.edge, &r.at) else {
        return Err(cx.fail("a clamp-to-edge coordinate of a read without its sampler"));
    };
    let edge = match level {
        Some(l) if decl.usage == ImageUsage::Sampled => {
            let levels = fb.query(ImageQuery::Levels, img)?;
            let clamped = clamp_below(fb, l, levels, 1)?;
            EdgeSize {
                size: fb.query(ImageQuery::SizeLod(clamped), img)?,
                levels: Some((levels, clamped)),
            }
        }
        _ => EdgeSize {
            size: fb.query(ImageQuery::Size, img)?,
            levels: None,
        },
    };
    let mut plane = cx.operand(fb, at.plane)?;
    if let Some(o) = e.offset {
        let o = cx.operand(fb, o)?;
        plane = fb.binary(BinOp::IAdd, plane, o)?;
    }
    let n = coord_len(decl);
    let lanes = at.lanes;
    let extent = if lanes == n {
        edge.size
    } else if lanes == 1 {
        fb.composite_extract(edge.size, &[0])?
    } else {
        let first: Vec<u32> = (0..u32::from(lanes)).collect();
        fb.vector_shuffle(edge.size, edge.size, &first)?
    };
    let one = u32s(fb, lanes, 1)?;
    let top = fb.binary(BinOp::ISub, extent, one)?;
    let below = fb.glsl(GlslOp::SMin, &[plane, top])?;
    let zero = u32s(fb, lanes, 0)?;
    let plane = fb.glsl(GlslOp::SMax, &[below, zero])?;
    Ok((compose(cx, fb, decl, at, plane)?, edge))
}

/// `min(x, bound - 1)` lane-wise, for unsigned `x` and a nonzero `bound` of the same shape.
fn clamp_below(fb: &mut FunctionBuilder<'_>, x: SValue, bound: SValue, n: u8) -> BResult<SValue> {
    let one = u32s(fb, n, 1)?;
    let top = fb.binary(BinOp::ISub, bound, one)?;
    fb.glsl(GlslOp::UMin, &[x, top])
}

/// Swap red and blue (`SwizzleBgra`).
fn swizzle_bgra(fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<SValue> {
    fb.vector_shuffle(x, x, &[2, 1, 0, 3])
}

/// Rotate `ARGB8Unorm`'s channels (`RotateArgb`): its bytes are alpha, red, green, and blue, which
/// an `R8G8B8A8_UNORM` image holds in its red, green, blue, and alpha. A read takes Metal's red,
/// green, blue, and alpha from the image's green, blue, alpha, and red; a write stores the inverse.
fn rotate_argb(fb: &mut FunctionBuilder<'_>, x: SValue, read: bool) -> BResult<SValue> {
    match read {
        true => fb.vector_shuffle(x, x, &[1, 2, 3, 0]),
        false => fb.vector_shuffle(x, x, &[3, 0, 1, 2]),
    }
}

/// `A8Unorm`'s byte held in an `R8_UNORM` image's red (`AlphaInRed`): a read (image texel `x`,
/// `(r, 0, 0, 1)`) is `(0, 0, 0, r)`, Metal's alpha-only texel; a write stores the texel's alpha
/// in red (the image's other lanes are none).
fn alpha_in_red(fb: &mut FunctionBuilder<'_>, x: SValue, read: bool) -> BResult<SValue> {
    match read {
        true => {
            let zero = f32s(fb, 4, 0.0)?;
            fb.vector_shuffle(zero, x, &[0, 1, 2, 4])
        }
        false => fb.vector_shuffle(x, x, &[3, 3, 3, 3]),
    }
}

/// The words [`heap_alpha_in_red`] emits, at most: the class field's shift and mask, the
/// comparison, the shuffle and the select.
pub(crate) const HEAP_ALPHA_IN_RED_WORDS: u32 = 40;

/// Image texel `x` (four `f32` lanes) of a storage access through heap handle `handle`, as
/// [`alpha_in_red`] converts it (`read`: the image's to Metal's; else Metal's to the image's) where
/// the handle's class index is `A8Unorm`'s (`metal::texel_format::heap_alpha_in_red_index`), as
/// it is elsewhere (`Read::heap_alpha_red`, `Write::heap_alpha_red`).
fn heap_alpha_in_red(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    handle: ValueId,
    x: SValue,
    read: bool,
) -> BResult<SValue> {
    let index = crate::metal::texel_format::heap_alpha_in_red_index();
    let is = heap_class_among(cx, fb, handle, &[index])?;
    let held = alpha_in_red(fb, x, read)?;
    fb.select(is, held, x)
}

/// The words [`srgb_codes`] emits, at most: `pack::pack`'s `unorm4x8_srgb` (the transfer's steps
/// and their constants, the 8-bit instruction) and the codes' extraction and scale.
pub(crate) const SRGB_WRITE_WORDS: u32 = 16 + 9 * 6 + 5 * 8 + 12 + 24 * 4 + 96;

/// The values an 8-bit sRGB format's UNORM alias (`SrgbEncode`) stores as the codes of held texel
/// `x` (four `f32` lanes, a `half` texel's widened exactly): `pack_float_to_srgb_unorm4x8` by the
/// device's conversion (`pack::pack`: each colour channel through the MSL transfer by the device's
/// `Pow`, then `PackUnorm4x8`; `PrecisionNote::DevicePackConversion`), which a kernel's write
/// takes too. Metal encodes each colour channel through its twelve-bit table. Each code `c` is
/// written as `c` times the `f32` nearest `1 / 255`, whose product with 255 lies within `2^-7` of
/// `c`, so every device's conversion stores `c`.
fn srgb_codes(fb: &mut FunctionBuilder<'_>, form: TexelForm, x: SValue) -> BResult<SValue> {
    use crate::air::intrinsics::PackFormat;
    use crate::lower::arith::legal::{FloatFmt, Form, Held};
    if form.lanes != 4 {
        return Err(crate::spirv::error::BuildError::Type(
            crate::spirv::error::TypeRule::FunctionSignature,
        ));
    }
    let f = Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 4,
    };
    let word = crate::lower::arith::pack::pack(fb, PackFormat::Unorm4x8Srgb, f, x)?;
    let u = u32_ty(fb)?;
    let vu = vec(fb, u, 4)?;
    let spread = fb.composite_construct(vu, &[word; 4])?;
    let mut shifts = Vec::with_capacity(4);
    for lane in 0..4u32 {
        shifts.push(u32s(fb, 1, 8 * lane)?);
    }
    let shifts = fb.composite_construct(vu, &shifts)?;
    let codes = fb.binary(BinOp::ShiftRightLogical, spread, shifts)?;
    let byte = u32s(fb, 4, 0xff)?;
    let codes = fb.binary(BinOp::BitwiseAnd, codes, byte)?;
    let f32_ = f32_ty(fb)?;
    let vf = vec(fb, f32_, 4)?;
    let c = fb.convert(ConvOp::ConvertUToF, codes, vf)?;
    let scale = f32s(fb, 4, 1.0 / 255.0)?;
    fb.binary(BinOp::FMul, c, scale)
}

/// The held AIR texel of image texel `x` (a `vec4` of the image's sampled type), read as one
/// texel: a `half` one converted by the bound format's class (`half`), a `float` one where the
/// read takes `float`; an 8-bit sRGB format's codes decoded by the device's transfer
/// (`SrgbEncode`, [`srgb_texel`]).
fn from_image(
    fb: &mut FunctionBuilder<'_>,
    decl: &ImageDecl,
    form: TexelForm,
    x: SValue,
    half: Option<HalfClass>,
) -> BResult<SValue> {
    if r64(decl) {
        return ulong_texel(fb, x);
    }
    let x = match decl.class {
        StorageImageClass::Identity => x,
        StorageImageClass::SwizzleBgra => swizzle_bgra(fb, x)?,
        // An 8-bit sRGB format's codes, through its UNORM alias: decoded by the device's
        // transfer, as a UNORM twin's are (alpha 1 for an RGBX format's).
        StorageImageClass::SrgbEncode => {
            let x = match decl.x_channel {
                true => x_channel_read(fb, decl, x)?,
                false => x,
            };
            return srgb_texel(fb, form, x, SRGB_TEXEL);
        }
        // `BGRA8Unorm_sRGB`'s codes through an rgba8 view: red and blue exchanged, then decoded.
        StorageImageClass::SrgbEncodeBgra => {
            let x = swizzle_bgra(fb, x)?;
            return srgb_texel(fb, form, x, SRGB_TEXEL);
        }
        StorageImageClass::RotateArgb => rotate_argb(fb, x, true)?,
        StorageImageClass::AlphaInRed => alpha_in_red(fb, x, true)?,
        // A `YCbCrWrite` slot is only written (`StorageImageClass::YCbCrWrite`).
        StorageImageClass::YCbCrWrite => x,
        // An `RGB9E5Float` word, through its `R32_UINT` alias, decoded exactly.
        StorageImageClass::SharedExponentBits => return shared_exponent_texel(fb, form, x),
    };
    let x = match decl.x_channel {
        true => x_channel_read(fb, decl, x)?,
        false => x,
    };
    // A `half` texel of a slot that converts by the bound format's class: a read's lanes are one
    // texel's.
    let x = match half {
        Some(h) => {
            let lanes = TexelLanes::Texel.value(fb)?;
            h.convert(fb, x, lanes, TexelOrigin::Image)?
        }
        None => x,
    };
    let x = if form.lanes == 1 {
        fb.composite_extract(x, &[0])?
    } else {
        x
    };
    match half {
        Some(_) => Ok(x),
        None => held_texel(fb, decl.held_class(), form, x),
    }
}

/// The held texel (`form`, a `half` or `float` result) of texel `x` (a `vec4` of `f32`) of an
/// 8-bit sRGB texture that the device has no sRGB view of (a storage image's UNORM alias, a
/// format's UNORM twin): each lane an 8-bit code `c` as the UNORM view reads it, `c / 255`, or the
/// device's filter of such codes. A color lane is decoded by the MSL transfer through the device's
/// division and `Pow` (`pack::srgb_decode_texel`), an alpha lane kept, and a `half` result is the
/// `float` value rounded to nearest even. Metal decodes each code through its twelve-bit table
/// and filters the decoded values (`PrecisionNote::SrgbTexelDecode`). `color` says which lanes are
/// color: a texel's first three, every lane of a gather of a color channel, none of a gather of
/// alpha. A format of fewer channels reads 0 for green and blue and 1 for alpha through its view,
/// which decode to the same.
pub(crate) fn srgb_texel(
    fb: &mut FunctionBuilder<'_>,
    form: TexelForm,
    x: SValue,
    color: [bool; 4],
) -> BResult<SValue> {
    use crate::lower::arith::legal::{FloatFmt, Form, Held};
    let d = crate::lower::arith::pack::srgb_decode_texel(fb, x, color)?;
    let d = match form.conv {
        TexelConv::F16 => {
            let half = Form {
                held: Held::Float(FloatFmt::Half),
                lanes: 4,
            };
            narrow::round(fb, half, d)?
        }
        _ => d,
    };
    if form.lanes == 1 {
        fb.composite_extract(d, &[0])
    } else {
        Ok(d)
    }
}

/// Texel `x` of codes (a `vec4` of `f32`, `c / 255` as in [`srgb_texel`]) of an 8-bit sRGB
/// texture held as its UNORM twin, as a bicubic filter's tap (`BicubicTexels::SRGB`): each color
/// lane the device's decode of its code (`pack::srgb_decode_texel`), which the filter sums as it
/// is, alpha's `c / 255` as it is. A border color the filter substitutes (each lane 0 or 1) reads
/// the same either way.
pub(crate) fn srgb_raw_tap(fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<SValue> {
    crate::lower::arith::pack::srgb_decode_texel(fb, x, SRGB_TEXEL)
}

/// The words [`srgb_raw_tap`] emits, at most: the transfer's steps on the color lanes
/// (`the_srgb_estimate_bounds_its_emission`).
pub(crate) const SRGB_TAP_WORDS: u32 = crate::lower::arith::pack::SRGB_DECODE_TEXEL_WORDS;

/// The color lanes of an sRGB texel ([`srgb_texel`]): red, green, and blue.
pub(crate) const SRGB_TEXEL: [bool; 4] = [true, true, true, false];

/// The words [`srgb_texel`] emits, at most: the transfer's steps, a `half` result's rounding, and
/// the lane's extraction (`the_srgb_estimate_bounds_its_emission`).
pub(crate) const SRGB_DECODE_WORDS: u32 =
    crate::lower::arith::pack::SRGB_DECODE_TEXEL_WORDS + 64 + 4 * 4 + 8;

/// The conversion of a slot's `half` texels by the bound format's class: the module's routine and
/// the slot's `TexelFormat` specialization constant; or of a heap color texture's by the class its
/// handle carries, read from the handle.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HalfClass {
    routine: FuncId,
    class: ClassOf,
    /// The slot's `PlainHalfTexels` specialization, where it declares one.
    plain: Option<SValue>,
}

/// Where a [`HalfClass`]'s class comes from.
#[derive(Clone, Copy, Debug)]
enum ClassOf {
    /// A slot's `TexelFormat` specialization constant.
    Slot(ConstId),
    /// A heap texture's class, read from its handle (`heap_texel_class`).
    Heap(SValue),
}

impl HalfClass {
    /// The held `half` texel of image texel `x` (a `vec4` of `f32`) of lanes `lanes`
    /// (`half::TexelLanes::code`). An image instruction's texel (`origin`) of a slot that declares
    /// `PlainHalfTexels` converts by the plain arm where the specialization holds and by the
    /// routine where it does not, one selection construct on it, so specialization keeps one of
    /// them (`half` module docs).
    pub(crate) fn convert(
        self,
        fb: &mut FunctionBuilder<'_>,
        x: SValue,
        lanes: SValue,
        origin: TexelOrigin,
    ) -> BResult<SValue> {
        let class = match self.class {
            ClassOf::Slot(c) => fb.module().const_value(c)?,
            ClassOf::Heap(v) => v,
        };
        use crate::lower::arith::legal::{FloatFmt, Form, Held};
        use crate::spirv::error::{BuildError, OperandRule};
        let routine = self.routine;
        let plain = match (origin, self.plain) {
            (TexelOrigin::Image, Some(plain)) => plain,
            _ => return super::half::convert(fb, routine, x, class, lanes),
        };
        let half = Form {
            held: Held::Float(FloatFmt::Half),
            lanes: 4,
        };
        let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
            let v = narrow::round_texel(fb, half, x)?;
            Ok(ArmEnd::FallThrough(vec![v]))
        });
        let other: ArmFn<'_, ()> = Box::new(move |fb, _| {
            let v = super::half::convert(fb, routine, x, class, lanes)?;
            Ok(ArmEnd::FallThrough(vec![v]))
        });
        let Merged(merged) = fb.if_(&mut (), plain, then, Some(other))?;
        merged.first().copied().ok_or(BuildError::Operand {
            op: crate::spv::Op::Phi,
            rule: OperandRule::Arity,
        })
    }
}

/// How the `half` texels of a read, gather, or sample of `image` in form `texel` convert: by the
/// bound format's class where the slot takes its `TexelFormat` specialization, else (any other
/// result, a heap element, a depth texture) through [`held_texel`].
pub(crate) fn half_class(cx: &EmitCx<'_>, image: &ImageRef, texel: TexelForm) -> Option<HalfClass> {
    if texel.conv != TexelConv::F16 {
        return None;
    }
    let ImageRef::Slot(index) = *image else {
        return None;
    };
    let t = &cx.decl.families.texture;
    Some(HalfClass {
        routine: t.half_routine?,
        class: ClassOf::Slot(t.half_format.get(&index).copied()?),
        plain: t.plain_half.get(&index).copied(),
    })
}

/// How the `half` texels of a sample or gather of the heap color texture of handle `handle`
/// convert, in form `texel`: by the class the handle carries, read from it, where the module
/// defines the routine and the class table (`TexturePlan::heap_half`), else `None`.
pub(crate) fn heap_half_class(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    handle: ValueId,
    texel: TexelForm,
) -> BResult<Option<HalfClass>> {
    let t = &cx.decl.families.texture;
    let (true, Some(routine), Some(_)) =
        (texel.conv == TexelConv::F16, t.half_routine, t.heap_classes)
    else {
        return Ok(None);
    };
    let h = cx.operand(fb, handle)?;
    let class = super::heap_texel_class(cx, fb, h)?;
    Ok(Some(HalfClass {
        routine,
        class: ClassOf::Heap(class),
        plain: None,
    }))
}

/// The held AIR texel of `x`, `form.lanes` components of an image of sampled type `texel`: a
/// `half` rounded exactly, half subnormals and the decoded NaN kept (`narrow::round_texel`), a
/// signed integer bitcast to its held `u32`s, a 16-bit integer masked, and a `ulong` joined from
/// its two words. Shared with sampling.
pub fn held_texel(
    fb: &mut FunctionBuilder<'_>,
    texel: TexelClass,
    form: TexelForm,
    x: SValue,
) -> BResult<SValue> {
    let n = form.lanes;
    let u = u32_ty(fb)?;
    let held = vec(fb, u, n)?;
    match form.conv {
        TexelConv::F32 => Ok(x),
        TexelConv::F16 => narrow::round_texel(fb, form.half(), x),
        TexelConv::I32 if texel == TexelClass::Sint => fb.convert(ConvOp::Bitcast, x, held),
        TexelConv::I32 => Ok(x),
        TexelConv::I16 { .. } => {
            let x = if texel == TexelClass::Sint {
                fb.convert(ConvOp::Bitcast, x, held)?
            } else {
                x
            };
            let mask = u32s(fb, n, 0xFFFF)?;
            fb.binary(BinOp::BitwiseAnd, x, mask)
        }
        TexelConv::U64Words { wide } => join_words(fb, x, wide),
    }
}

/// Held texel `x` of form `native`, a texture's own result in its numeric class (`F32` or
/// `I32`, `texture::native_form`), as the result `declared` a slot of another class declares
/// (`LoweringInputs::cross_class`), as plain Metal hands it back: each 32-bit word as it is, a
/// `float` or 32-bit integer one bitcast, a `half` one the low 16 bits of each word, held widened
/// with its bits noted (`narrow::from_bits`: every bit pattern exactly but a NaN's, whose encoding
/// is the device's where the module widens a `half` by the device, as its
/// `PrecisionNote::NativeHalfConversion` or `NativeNanEncoding` records). A conversion
/// `texture::native_form` makes no form for is an emission failure.
pub(crate) fn reinterpret(
    fb: &mut FunctionBuilder<'_>,
    native: TexelForm,
    declared: TexelConv,
    x: SValue,
) -> BResult<SValue> {
    let n = native.lanes;
    match (native.conv, declared) {
        (TexelConv::F32, TexelConv::F32) | (TexelConv::I32, TexelConv::I32) => Ok(x),
        (TexelConv::F32, TexelConv::I32) => {
            let u = u32_ty(fb)?;
            let t = vec(fb, u, n)?;
            fb.convert(ConvOp::Bitcast, x, t)
        }
        (TexelConv::I32, TexelConv::F32) => {
            let f = f32_ty(fb)?;
            let t = vec(fb, f, n)?;
            fb.convert(ConvOp::Bitcast, x, t)
        }
        (TexelConv::I32, TexelConv::F16) => {
            let half = TexelForm {
                conv: TexelConv::F16,
                lanes: n,
            };
            narrow::from_bits(fb, half.half(), x)
        }
        // A `short` or `ushort` result: an integer word's low 16 bits, a float texel's `half`
        // bits, held zero-extended.
        (TexelConv::I32, TexelConv::I16 { .. }) => {
            let mask = u32s(fb, n, 0xFFFF)?;
            fb.binary(BinOp::BitwiseAnd, x, mask)
        }
        (TexelConv::F16, TexelConv::I16 { .. }) => narrow::to_bits(fb, native.half(), x),
        _ => Err(crate::spirv::error::BuildError::Type(
            crate::spirv::error::TypeRule::FunctionSignature,
        )),
    }
}

/// Written lanes `v`, held as `c.declared` declares them, as the texel `native` of the bound
/// texture's own class (`texture::native_write_form`), as plain Metal stores them in format
/// `c.format` (on an Apple M3 Max, 28 formats, texture and texture-buffer writes):
///
/// - into an integer format, the lane's word (a 32-bit lane's bits, a 16-bit lane's, a `half`'s
///   bits, extended by the format's signedness), which the write then saturates to each
///   channel's range as every Metal integer write does ([`saturated`]);
/// - into a float format, a 32-bit lane's bits as a `float`, a 16-bit lane's as a `half`, which
///   the float class then writes as a `float` or `half` texel, the device's store converting a
///   `float` onto a half-float or small-float channel (`RG11B10Float`) as it converts every
///   `float` write, where Metal's integer store rounds to nearest even
///   (`PrecisionNote::DeviceColorConversion`, `texture::half_storage_facts`).
pub(crate) fn cross_write(
    fb: &mut FunctionBuilder<'_>,
    c: super::CrossWrite,
    native: TexelForm,
    v: SValue,
) -> BResult<SValue> {
    use crate::vk::NumericFormat as N;
    let n = native.lanes;
    let malformed =
        || crate::spirv::error::BuildError::Type(crate::spirv::error::TypeRule::FunctionSignature);
    let half = TexelForm {
        conv: TexelConv::F16,
        lanes: n,
    };
    match native.conv {
        TexelConv::I32 => {
            let block = c.format.block().ok_or_else(malformed)?;
            let signed = block.color == Some(N::Sint);
            let (word, sixteen) = match c.declared {
                TexelConv::F32 => {
                    let u = u32_ty(fb)?;
                    let t = vec(fb, u, n)?;
                    (fb.convert(ConvOp::Bitcast, v, t)?, false)
                }
                TexelConv::I32 => (v, false),
                TexelConv::F16 => (narrow::to_bits(fb, half.half(), v)?, true),
                TexelConv::I16 { .. } => (v, true),
                TexelConv::U64Words { .. } => return Err(malformed()),
            };
            // A 16-bit lane, held zero-extended, sign-extended into a signed format.
            match sixteen && signed {
                true => {
                    let i32_ = fb.module().ty(SType::Int {
                        width: 32,
                        signed: true,
                    })?;
                    let ivec = vec(fb, i32_, n)?;
                    let u = u32_ty(fb)?;
                    let uvec = vec(fb, u, n)?;
                    let k = u32s(fb, n, 16)?;
                    let up = fb.binary(BinOp::ShiftLeftLogical, word, k)?;
                    let s = fb.convert(ConvOp::Bitcast, up, ivec)?;
                    let s = fb.binary(BinOp::ShiftRightArithmetic, s, k)?;
                    fb.convert(ConvOp::Bitcast, s, uvec)
                }
                false => Ok(word),
            }
        }
        TexelConv::F32 => {
            let x = match c.declared {
                TexelConv::I32 => {
                    let f = f32_ty(fb)?;
                    let t = vec(fb, f, n)?;
                    fb.convert(ConvOp::Bitcast, v, t)?
                }
                TexelConv::I16 { .. } => narrow::from_bits(fb, half.half(), v)?,
                _ => return Err(malformed()),
            };
            Ok(x)
        }
        // A 16-bit lane's bits as the `half` the float class writes.
        TexelConv::F16 => match c.declared {
            TexelConv::I16 { .. } => narrow::from_bits(fb, half.half(), v),
            _ => Err(malformed()),
        },
        _ => Err(malformed()),
    }
}

/// Image texel `x` (`form`'s lanes, the image's signed or unsigned words after its class's
/// conversion) saturated to each channel's range under `range`, as plain Metal stores an integer
/// write: each lane clamped to `[-2^(bits-1), 2^(bits-1) - 1]` in a signed image and to
/// `[0, 2^bits - 1]` in an unsigned one. A slot's word holds `32 - bits` in each channel's byte,
/// so the bounds are a shift of the 32-bit ones.
fn saturated(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    decl: &ImageDecl,
    (range, form): (super::IntRange, TexelForm),
    x: SValue,
) -> BResult<SValue> {
    use super::IntRange;
    let n = form.lanes;
    let malformed =
        || crate::spirv::error::BuildError::Type(crate::spirv::error::TypeRule::FunctionSignature);
    let signed = match decl.texel {
        TexelClass::Sint => true,
        TexelClass::Uint => false,
        _ => return Ok(x),
    };
    let i32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: true,
    })?;
    let u = u32_ty(fb)?;
    match range {
        IntRange::Full => Ok(x),
        IntRange::Widths(bits) => {
            let lanes = bits.get(..usize::from(n)).ok_or_else(malformed)?;
            if signed {
                let lo: Vec<u32> = lanes
                    .iter()
                    .map(|b| (-(1i64 << ((*b).clamp(1, 32) - 1))) as u32)
                    .collect();
                let hi: Vec<u32> = lanes
                    .iter()
                    .map(|b| ((1i64 << ((*b).clamp(1, 32) - 1)) - 1) as u32)
                    .collect();
                let lo = int_lanes(fb, i32_, &lo)?;
                let hi = int_lanes(fb, i32_, &hi)?;
                fb.glsl(GlslOp::SClamp, &[x, lo, hi])
            } else {
                let hi: Vec<u32> = lanes
                    .iter()
                    .map(|b| ((1u64 << (*b).clamp(1, 32)) - 1) as u32)
                    .collect();
                let hi = int_lanes(fb, u, &hi)?;
                fb.glsl(GlslOp::UMin, &[x, hi])
            }
        }
        IntRange::SlotWord(slot) => {
            let word = cx.push_field(
                fb,
                crate::pipeline::sym::PushField::TextureIntegerChannels(slot),
            )?;
            let at: [u32; 4] = [0, 8, 16, 24];
            let at = at.get(..usize::from(n)).ok_or_else(malformed)?;
            let words = match n {
                1 => word,
                _ => {
                    let t = vec(fb, u, n)?;
                    fb.composite_construct(t, &vec![word; usize::from(n)])?
                }
            };
            let at = int_lanes(fb, u, at)?;
            let shifts = fb.binary(BinOp::ShiftRightLogical, words, at)?;
            let mask = u32s(fb, n, 0xFF)?;
            let shifts = fb.binary(BinOp::BitwiseAnd, shifts, mask)?;
            if signed {
                let top = u32s(fb, n, 0x7FFF_FFFF)?;
                let hi = fb.binary(BinOp::ShiftRightLogical, top, shifts)?;
                let lo = fb.unary(UnOp::Not, hi)?;
                let ivec = vec(fb, i32_, n)?;
                let hi = fb.convert(ConvOp::Bitcast, hi, ivec)?;
                let lo = fb.convert(ConvOp::Bitcast, lo, ivec)?;
                fb.glsl(GlslOp::SClamp, &[x, lo, hi])
            } else {
                let all = u32s(fb, n, u32::MAX)?;
                let hi = fb.binary(BinOp::ShiftRightLogical, all, shifts)?;
                fb.glsl(GlslOp::UMin, &[x, hi])
            }
        }
    }
}

/// The integer constant of scalar type `elem` holding `lanes` (one lane: the scalar).
fn int_lanes(fb: &mut FunctionBuilder<'_>, elem: TypeId, lanes: &[u32]) -> BResult<SValue> {
    let parts = lanes
        .iter()
        .map(|b| {
            fb.module().constant(Constant::Scalar {
                ty: elem,
                bits: ConstBits(u64::from(*b)),
            })
        })
        .collect::<BResult<Vec<ConstId>>>()?;
    if let [one] = parts.as_slice() {
        return fb.module().const_value(*one);
    }
    let n = u8::try_from(parts.len()).unwrap_or(4);
    let ty = vec(fb, elem, n)?;
    let c = fb.module().constant(Constant::Composite {
        ty,
        parts: parts.into(),
    })?;
    fb.module().const_value(c)
}

/// The image texel (a `vec4` of the image's sampled type) of held AIR texel `v`; an 8-bit sRGB
/// format's codes encoded by the device's conversion (`SrgbEncode`, [`srgb_codes`]).
pub fn to_image(
    fb: &mut FunctionBuilder<'_>,
    decl: &ImageDecl,
    form: TexelForm,
    v: SValue,
) -> BResult<SValue> {
    let signed = decl.texel == TexelClass::Sint;
    let i32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: true,
    })?;
    let ivec = vec(fb, i32_, form.lanes)?;
    let x = match form.conv {
        TexelConv::F32 | TexelConv::F16 => v,
        TexelConv::I32 if signed => fb.convert(ConvOp::Bitcast, v, ivec)?,
        TexelConv::I32 => v,
        TexelConv::I16 { .. } if signed => {
            // Sign-extend the held 16-bit value: an out-of-range integer texel is undefined.
            let sixteen = u32s(fb, form.lanes, 16)?;
            let up = fb.binary(BinOp::ShiftLeftLogical, v, sixteen)?;
            let s = fb.convert(ConvOp::Bitcast, up, ivec)?;
            fb.binary(BinOp::ShiftRightArithmetic, s, sixteen)?
        }
        TexelConv::I16 { .. } => v,
        // An `R64` image stores lane 0 of the held `ulong` texel as it is.
        TexelConv::U64Words { .. } if r64(decl) => v,
        TexelConv::U64Words { wide } => split_words(fb, v, wide)?,
    };
    let x = match decl.class {
        StorageImageClass::Identity => x,
        StorageImageClass::SwizzleBgra => swizzle_bgra(fb, x)?,
        StorageImageClass::SrgbEncode => srgb_codes(fb, form, x)?,
        StorageImageClass::SrgbEncodeBgra => {
            let codes = srgb_codes(fb, form, x)?;
            swizzle_bgra(fb, codes)?
        }
        StorageImageClass::RotateArgb => rotate_argb(fb, x, false)?,
        StorageImageClass::AlphaInRed if form.lanes == 4 => alpha_in_red(fb, x, false)?,
        StorageImageClass::AlphaInRed => {
            return Err(crate::spirv::error::BuildError::Type(
                crate::spirv::error::TypeRule::FunctionSignature,
            ))
        }
        StorageImageClass::YCbCrWrite => ycbcr_write_texel(fb, form, x)?,
        StorageImageClass::SharedExponentBits => shared_exponent_word(fb, form, x)?,
    };
    match decl.x_channel && form.lanes == 4 {
        true => x_channel_write(fb, x),
        false => Ok(x),
    }
}

/// An RGBX texel as Metal reads it (`metal::format::x_channel`): image texel `x` (a `vec4` of
/// the image's sampled type, after the class's conversion) with alpha 1, `1.0` or the integer 1.
fn x_channel_read(fb: &mut FunctionBuilder<'_>, decl: &ImageDecl, x: SValue) -> BResult<SValue> {
    let one = match decl.texel {
        TexelClass::Float => f32s(fb, 1, 1.0)?,
        TexelClass::Sint => {
            let i32_ = fb.module().ty(SType::Int {
                width: 32,
                signed: true,
            })?;
            splat(fb, i32_, 1, 1)?
        }
        TexelClass::Uint | TexelClass::Uint64 => u32s(fb, 1, 1)?,
    };
    fb.composite_insert(one, x, &[3])
}

/// An RGBX texel as Metal's write stores it (`metal::format::x_channel`): image texel `x` (four
/// lanes, after the class's conversion, so a converted green's bits) with its green in the fourth
/// channel, Metal's X field.
fn x_channel_write(fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<SValue> {
    fb.vector_shuffle(x, x, &[0, 1, 2, 1])
}

/// The words of [`ycbcr_write_texel`] on four lanes: three extracts and their bitcasts, the
/// constants, and the construction.
pub(crate) const YCBCR_WRITE_WORDS: u32 = 32;

/// The `uvec4` a `YCbCrWrite` slot stores for held texel `x` (`f32` lanes of `form`, a `half`
/// texel's widened exactly): the bits of its red, green and blue (0 for a lane the texel lacks)
/// and the mark 1 of a written texel (`StorageImageClass::YCbCrWrite`). Its slot is a `float`
/// one's (`slot::declare_slot`).
fn ycbcr_write_texel(fb: &mut FunctionBuilder<'_>, form: TexelForm, x: SValue) -> BResult<SValue> {
    let u = u32_ty(fb)?;
    let zero = u32s(fb, 1, 0)?;
    let mut words = Vec::with_capacity(4);
    for lane in 0..3u32 {
        let word = match (form.lanes, lane) {
            (1, 0) => fb.convert(ConvOp::Bitcast, x, u)?,
            (n, lane) if lane < u32::from(n) && n > 1 => {
                let channel = fb.composite_extract(x, &[lane])?;
                fb.convert(ConvOp::Bitcast, channel, u)?
            }
            _ => zero,
        };
        words.push(word);
    }
    words.push(u32s(fb, 1, 1)?);
    let ty = vec(fb, u, 4)?;
    fb.composite_construct(ty, &words)
}

/// The held texel (`form`, a `half` or `float` result) of a `SharedExponentBits` slot's image texel
/// `x` (a `uvec4`, the `RGB9E5Float` word in its first lane): each of red, green and blue its
/// mantissa `m` times `2^(e - 24)` for the word's exponent `e`, exact in `f32` and in `half` (nine
/// significant bits, from `2^-24` to 65408), and alpha 1, as Metal reads the texel.
fn shared_exponent_texel(
    fb: &mut FunctionBuilder<'_>,
    form: TexelForm,
    x: SValue,
) -> BResult<SValue> {
    let f = f32_ty(fb)?;
    let word = fb.composite_extract(x, &[0])?;
    let twenty_seven = fb.u32_const(27)?;
    let twenty_seven = fb.module().const_value(twenty_seven)?;
    let e = fb.binary(BinOp::ShiftRightLogical, word, twenty_seven)?;
    // `2^(e - 24)`: the `f32` of biased exponent `e + 103`, a normal for every `e` in `[0, 31]`.
    let bias = fb.u32_const(103)?;
    let bias = fb.module().const_value(bias)?;
    let biased = fb.binary(BinOp::IAdd, e, bias)?;
    let twenty_three = fb.u32_const(23)?;
    let twenty_three = fb.module().const_value(twenty_three)?;
    let scale = fb.binary(BinOp::ShiftLeftLogical, biased, twenty_three)?;
    let scale = fb.convert(ConvOp::Bitcast, scale, f)?;
    let nine_bits = fb.u32_const(0x1ff)?;
    let nine_bits = fb.module().const_value(nine_bits)?;
    let mut lanes = Vec::with_capacity(4);
    for c in 0..3u32 {
        let shift = fb.u32_const(9 * c)?;
        let shift = fb.module().const_value(shift)?;
        let m = fb.binary(BinOp::ShiftRightLogical, word, shift)?;
        let m = fb.binary(BinOp::BitwiseAnd, m, nine_bits)?;
        let m = fb.convert(ConvOp::ConvertUToF, m, f)?;
        lanes.push(fb.binary(BinOp::FMul, m, scale)?);
    }
    lanes.push(f32s(fb, 1, 1.0)?);
    let v4 = vec(fb, f, 4)?;
    let v = fb.composite_construct(v4, &lanes)?;
    match form.lanes {
        1 => fb.composite_extract(v, &[0]),
        _ => Ok(v),
    }
}

/// The `uvec4` a `SharedExponentBits` slot stores for held texel `x` (`f32` lanes of `form`, a
/// `half` texel's widened exactly): the `RGB9E5Float` word Metal stores in its first lane
/// (`StorageImageClass::SharedExponentBits`, [`rgb9e5_word`]); alpha is not stored. An integer
/// texel is not one the format takes.
fn shared_exponent_word(
    fb: &mut FunctionBuilder<'_>,
    form: TexelForm,
    x: SValue,
) -> BResult<SValue> {
    if form.lanes != 4 || !matches!(form.conv, TexelConv::F32 | TexelConv::F16) {
        return Err(crate::spirv::error::BuildError::Type(
            crate::spirv::error::TypeRule::FunctionSignature,
        ));
    }
    let u = u32_ty(fb)?;
    let word = rgb9e5_word(fb, x)?;
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    let v4 = vec(fb, u, 4)?;
    fb.composite_construct(v4, &[word, zero, zero, zero])
}

/// An upper estimate of the SPIR-V words of one [`rgb9e5_word`].
pub(crate) const RGB9E5_WORDS: u64 = 3 * 8 * 7 + 3 * 7 * 7 + 16 * 7;

/// The `RGB9E5Float` word, a `u32`, Metal stores for the red, green and blue `f32` lanes of `x` (a
/// vector of at least three; a kernel's write, a `float` output's store into a word attachment,
/// `metal::format::WordAttachment::Rgb9e5`): each clamped to `[0, 65408]` on its bits (a NaN, a
/// negative value and `-0` 0; an infinity 65408), the exponent `e = max(⌊log2 m⌋, -16) + 16` from
/// the largest clamped channel `m`'s exponent field, and each mantissa the clamped channel times
/// `2^(24 - e)` (exact: a power of two, the product below 512) converted toward zero.
pub(crate) fn rgb9e5_word(fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<SValue> {
    let u = u32_ty(fb)?;
    let f = f32_ty(fb)?;
    let constant = |fb: &mut FunctionBuilder<'_>, v: u32| -> BResult<SValue> {
        let c = fb.u32_const(v)?;
        fb.module().const_value(c)
    };
    let zero = constant(fb, 0)?;
    let top = constant(fb, 65408.0f32.to_bits())?;
    let magnitude = constant(fb, 0x7fff_ffff)?;
    let infinity = constant(fb, 0x7f80_0000)?;
    let mut clamped = [zero; 3];
    for (c, out) in (0u32..).zip(clamped.iter_mut()) {
        let v = fb.composite_extract(x, &[c])?;
        let bits = fb.convert(ConvOp::Bitcast, v, u)?;
        // A NaN, or any value with its sign bit set, is 0; a positive value's bits order as its
        // value does, so the clamp is an unsigned minimum.
        let mag = fb.binary(BinOp::BitwiseAnd, bits, magnitude)?;
        let nan = fb.binary(BinOp::UGreaterThan, mag, infinity)?;
        let negative = fb.binary(BinOp::UGreaterThan, bits, magnitude)?;
        let dropped = fb.binary(BinOp::LogicalOr, nan, negative)?;
        let below = fb.glsl(GlslOp::UMin, &[bits, top])?;
        *out = fb.select(dropped, zero, below)?;
    }
    let [red, green, blue] = clamped;
    let largest = fb.glsl(GlslOp::UMax, &[red, green])?;
    let largest = fb.glsl(GlslOp::UMax, &[largest, blue])?;
    // `max(⌊log2 m⌋, -16) + 16` is `max(field, 111) - 111` of the exponent field (a zero or a
    // denormal `m` taking 0).
    let twenty_three = constant(fb, 23)?;
    let field = fb.binary(BinOp::ShiftRightLogical, largest, twenty_three)?;
    let floor = constant(fb, 111)?;
    let field = fb.glsl(GlslOp::UMax, &[field, floor])?;
    let e = fb.binary(BinOp::ISub, field, floor)?;
    // `2^(24 - e)`: the `f32` of biased exponent `151 - e`.
    let top_bias = constant(fb, 151)?;
    let biased = fb.binary(BinOp::ISub, top_bias, e)?;
    let scale = fb.binary(BinOp::ShiftLeftLogical, biased, twenty_three)?;
    let scale = fb.convert(ConvOp::Bitcast, scale, f)?;
    let most = constant(fb, 0x1ff)?;
    let exponent_at = constant(fb, 27)?;
    let mut word = fb.binary(BinOp::ShiftLeftLogical, e, exponent_at)?;
    for (c, bits) in clamped.into_iter().enumerate() {
        let v = fb.convert(ConvOp::Bitcast, bits, f)?;
        let scaled = fb.binary(BinOp::FMul, v, scale)?;
        let m = fb.convert(ConvOp::ConvertFToU, scaled, u)?;
        let m = fb.glsl(GlslOp::UMin, &[m, most])?;
        let shift = constant(fb, 9 * c as u32)?;
        let m = fb.binary(BinOp::ShiftLeftLogical, m, shift)?;
        word = fb.binary(BinOp::BitwiseOr, word, m)?;
    }
    Ok(word)
}

/// Whether the texel class index heap handle `handle` carries is one of `indices` (a `bool`).
fn heap_class_among(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    handle: ValueId,
    indices: &[u32],
) -> BResult<SValue> {
    use crate::memory::handles::HANDLE_TEXEL_CLASS;
    let h = cx.operand(fb, handle)?;
    let shift = u32s(fb, 1, HANDLE_TEXEL_CLASS.trailing_zeros())?;
    let mask = u32s(
        fb,
        1,
        HANDLE_TEXEL_CLASS >> HANDLE_TEXEL_CLASS.trailing_zeros(),
    )?;
    let index = fb.binary(BinOp::ShiftRightLogical, h, shift)?;
    let index = fb.binary(BinOp::BitwiseAnd, index, mask)?;
    let mut among = None;
    for &i in indices {
        let k = u32s(fb, 1, i)?;
        let equal = fb.binary(BinOp::IEqual, index, k)?;
        among = Some(match among {
            None => equal,
            Some(any) => fb.binary(BinOp::LogicalOr, any, equal)?,
        });
    }
    among.ok_or_else(|| cx.fail("no heap class of the family"))
}

/// The estimated words of a write's conversion by storage class `class` on four lanes: an 8-bit
/// sRGB format's codes ([`srgb_codes`]), an `RGB9E5Float` word ([`shared_exponent_word`]), or a
/// YCbCr texel's bits and mark ([`ycbcr_write_texel`]), none for the other classes' swizzles,
/// which the write's base estimate covers.
pub(crate) fn class_write_words(class: StorageImageClass) -> u32 {
    match class {
        StorageImageClass::YCbCrWrite => YCBCR_WRITE_WORDS,
        StorageImageClass::SrgbEncode | StorageImageClass::SrgbEncodeBgra => SRGB_WRITE_WORDS,
        StorageImageClass::SharedExponentBits => SHARED_EXPONENT_WORDS,
        _ => 0,
    }
}

/// The words of [`shared_exponent_word`], at most: three channels' extract, clamp and mantissa
/// (about twelve instructions each), the exponent's six, the word's placement, and the constants.
pub(crate) const SHARED_EXPONENT_WORDS: u32 = 3 * 12 * 6 + 8 * 6 + 3 * 6 + 12 * 4;

/// `a && b`, or `b` alone.
fn and(fb: &mut FunctionBuilder<'_>, a: Option<SValue>, b: SValue) -> BResult<SValue> {
    match a {
        Some(a) => fb.binary(BinOp::LogicalAnd, a, b),
        None => Ok(b),
    }
}

/// `texel`, a read of `decl` as the image's `vec4`, swizzled by `word` (a slot's
/// `TextureSwizzle` specialization): each channel `c` is the source its three bits at `3c` name,
/// `MTLTextureSwizzle`'s raw value: 0 zero, 1 one (1.0 for a float image, else the integer 1), 2 to
/// 5 the texel's red, green, blue, or alpha. Selects on the word's fields, so one module serves
/// every swizzle.
fn swizzled(
    fb: &mut FunctionBuilder<'_>,
    decl: &ImageDecl,
    texel: SValue,
    word: SValue,
) -> BResult<SValue> {
    let (elem, one) = match decl.held_class() {
        TexelClass::Float => (f32_ty(fb)?, u64::from(1.0f32.to_bits())),
        TexelClass::Sint => (
            fb.module().ty(SType::Int {
                width: 32,
                signed: true,
            })?,
            1,
        ),
        TexelClass::Uint | TexelClass::Uint64 => (u32_ty(fb)?, 1),
    };
    swizzle_lanes(fb, (elem, one), texel, word)
}

/// The element type of a held texel of `form` ([`slice_texel`]'s), and the bits of its one: an
/// `f32`'s 1.0 for a `float` or `half` (held widened), a `u32`'s 1 for an integer. A `ulong`
/// texel has none here.
fn held_lane(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    form: TexelForm,
) -> BResult<(TypeId, u64)> {
    match form.conv {
        TexelConv::F32 | TexelConv::F16 => Ok((f32_ty(fb)?, u64::from(1.0f32.to_bits()))),
        TexelConv::I32 | TexelConv::I16 { .. } => Ok((u32_ty(fb)?, 1)),
        TexelConv::U64Words { .. } => Err(cx.fail("a swizzled `ulong` texel")),
    }
}

/// `texel`, a four-lane held texel of `form` read from a storage slot of
/// `LoweringInputs::swizzled_storage`, swizzled by `word` (the slot's `TextureSwizzle`
/// specialization) as plain Metal swizzles a storage read of a swizzled texture: each lane `c` is
/// the source its three bits at `3c` name, as [`swizzled`] takes it, of the texel as the slot
/// converts it, the out-of-bounds texel included.
fn swizzled_texel(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    form: TexelForm,
    texel: SValue,
    word: SValue,
) -> BResult<SValue> {
    let lane = held_lane(cx, fb, form)?;
    swizzle_lanes(fb, lane, texel, word)
}

/// `v`, a four-lane held texel of `form` written to a storage slot of
/// `LoweringInputs::swizzled_storage`, with the swizzle `word` undone as plain Metal undoes it
/// (an M4 under macOS 27.0.1): channel `c` takes the lane `i` of the first of the
/// swizzle's four sources that is channel `c` (raw value `2 + c`), and its own lane `c` where no
/// source is. A permutation's inverse, and for a swizzle with a zero, a one or a repeated source
/// the same rule: `RRRR` stores `v` as it is, `GR01` on two channels stores `(v.y, v.x)`.
fn unswizzled(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    form: TexelForm,
    v: SValue,
    word: SValue,
) -> BResult<SValue> {
    let (elem, _) = held_lane(cx, fb, form)?;
    let ty = vec(fb, elem, 4)?;
    let mut lanes = Vec::with_capacity(4);
    let mut sources = Vec::with_capacity(4);
    let seven = u32s(fb, 1, 7)?;
    for i in 0..4u32 {
        lanes.push(fb.composite_extract(v, &[i])?);
        let shift = u32s(fb, 1, 3 * i)?;
        let field = fb.binary(BinOp::ShiftRightLogical, word, shift)?;
        sources.push(fb.binary(BinOp::BitwiseAnd, field, seven)?);
    }
    let mut out = Vec::with_capacity(4);
    for (c, &own) in (0u32..).zip(&lanes) {
        let channel = u32s(fb, 1, 2 + c)?;
        let mut stored = own;
        // The first source that is the channel wins: select from the last to the first.
        for (&lane, &source) in lanes.iter().zip(&sources).rev() {
            let is = fb.binary(BinOp::IEqual, source, channel)?;
            stored = fb.select(is, lane, stored)?;
        }
        out.push(stored);
    }
    fb.composite_construct(ty, &out)
}

/// `texel`, a `vec4` of `f32` an extended-range sampled slot of
/// `LoweringInputs::swizzled_storage` decodes for a sample or a gather tap (a border texel
/// included), swizzled by `word` (the slot's `TextureSwizzle` specialization) as plain Metal
/// swizzles a sampled texture's decoded texels ([`swizzled`]).
pub(crate) fn swizzled_float(
    fb: &mut FunctionBuilder<'_>,
    texel: SValue,
    word: SValue,
) -> BResult<SValue> {
    let f = f32_ty(fb)?;
    swizzle_lanes(fb, (f, u64::from(1.0f32.to_bits())), texel, word)
}

/// The element type of `texel`, a four-lane held texel of a sampling, and the bits of its one:
/// an `f32`'s or `f16`'s 1.0, an integer's 1.
fn sampled_lane(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    texel: SValue,
) -> BResult<(TypeId, u64)> {
    let ty = fb.value_ty(texel)?;
    let elem = match fb.module().types().get(ty)? {
        SType::Vector { elem, len: 4 } => *elem,
        _ => return Err(cx.fail("a swizzled texel that is not four lanes")),
    };
    let one = match fb.module().types().get(elem)? {
        SType::Float { width: 32 } => u64::from(1.0f32.to_bits()),
        SType::Float { width: 16 } => 0x3c00,
        SType::Int { width: 32, .. } => 1,
        _ => return Err(cx.fail("a swizzled texel of no 32-bit or `half` lane")),
    };
    Ok((elem, one))
}

/// `texel`, the four-lane held texel a sample of a packed normalized slot of
/// `LoweringInputs::swizzled_storage` converts by its class, swizzled by `word` (the slot's
/// `TextureSwizzle` specialization) as plain Metal swizzles a sampled texture's texels
/// ([`swizzled`]).
pub(crate) fn swizzled_held(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    texel: SValue,
    word: SValue,
) -> BResult<SValue> {
    let lane = sampled_lane(cx, fb, texel)?;
    swizzle_lanes(fb, lane, texel, word)
}

/// The gather of `component` of a packed normalized slot of `LoweringInputs::swizzled_storage`,
/// from `taps`, the unswizzled texture's gathers of its four components (held texels, red's
/// first): the gather of the component the swizzle `word` names in its three bits at
/// `3 · component` (the slot's `TextureSwizzle` specialization), or 0 or 1 at every tap for its
/// `Zero` or `One`, as plain Metal gathers a swizzled texture.
pub(crate) fn swizzled_gather(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    taps: [SValue; 4],
    word: SValue,
    component: u32,
) -> BResult<SValue> {
    let (elem, one) = sampled_lane(cx, fb, taps[0])?;
    let shift = u32s(fb, 1, 3 * component)?;
    let field = fb.binary(BinOp::ShiftRightLogical, word, shift)?;
    let seven = u32s(fb, 1, 7)?;
    let field = fb.binary(BinOp::BitwiseAnd, field, seven)?;
    let ones = splat(fb, elem, 4, one)?;
    let zeros = splat(fb, elem, 4, 0)?;
    let from_one = {
        let k = u32s(fb, 1, 1)?;
        fb.binary(BinOp::IEqual, field, k)?
    };
    let mut out = fb.select(from_one, ones, zeros)?;
    for (c, tap) in (0u32..).zip(taps) {
        let k = u32s(fb, 1, 2 + c)?;
        let from = fb.binary(BinOp::IEqual, field, k)?;
        out = fb.select(from, tap, out)?;
    }
    Ok(out)
}

/// `texel`, a `vec4` of `elem` (one: the bits of its 1), swizzled by `word` ([`swizzled`]).
fn swizzle_lanes(
    fb: &mut FunctionBuilder<'_>,
    (elem, one): (TypeId, u64),
    texel: SValue,
    word: SValue,
) -> BResult<SValue> {
    let u = u32_ty(fb)?;
    let u4 = vec(fb, u, 4)?;
    let word = fb.composite_construct(u4, &[word, word, word, word])?;
    let shifts = {
        let mut parts = Vec::with_capacity(4);
        for v in [0u64, 3, 6, 9] {
            parts.push(fb.module().constant(Constant::Scalar {
                ty: u,
                bits: ConstBits(v),
            })?);
        }
        let c = fb.module().constant(Constant::Composite {
            ty: u4,
            parts: parts.into(),
        })?;
        fb.module().const_value(c)?
    };
    let fields = fb.binary(BinOp::ShiftRightLogical, word, shifts)?;
    let seven = u32s(fb, 4, 7)?;
    let fields = fb.binary(BinOp::BitwiseAnd, fields, seven)?;
    let is = |fb: &mut FunctionBuilder<'_>, v: u32| -> BResult<SValue> {
        let k = u32s(fb, 4, v)?;
        fb.binary(BinOp::IEqual, fields, k)
    };
    let ones = splat(fb, elem, 4, one)?;
    let zeros = splat(fb, elem, 4, 0)?;
    let from_one = is(fb, 1)?;
    let mut out = fb.select(from_one, ones, zeros)?;
    let ty = vec(fb, elem, 4)?;
    for c in 0..4u32 {
        let channel = fb.composite_extract(texel, &[c])?;
        let spread = fb.composite_construct(ty, &[channel, channel, channel, channel])?;
        let from = is(fb, 2 + c)?;
        out = fb.select(from, spread, out)?;
    }
    Ok(out)
}

/// Metal's out-of-bounds texel of a read of `decl` through `image`, as the image's `vec4`: zero,
/// with an alpha of 1 where `alpha` asks for the bound format's and it has none, else 0: a slot's
/// `OutOfBoundsAlpha` specialization says so, and a heap element's handle-table flag, which its
/// `flagged` handle carries. A depth read takes its first component, 0.
pub(super) fn out_of_bounds_texel(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    decl: &ImageDecl,
    image: &ImageRef,
    alpha: bool,
    flagged: Option<SValue>,
) -> BResult<SValue> {
    // An extended-range slot's out-of-bounds texel is Metal's value, selected after the decode.
    let (elem, one) = match decl.held_class() {
        TexelClass::Float => (f32_ty(fb)?, u64::from(1.0f32.to_bits())),
        TexelClass::Sint => (
            fb.module().ty(SType::Int {
                width: 32,
                signed: true,
            })?,
            1,
        ),
        // A `ulong` slot's image is `RG32` unsigned, or `R64` unsigned.
        TexelClass::Uint64 if r64(decl) => (u64_ty(fb)?, 1),
        TexelClass::Uint | TexelClass::Uint64 => (u32_ty(fb)?, 1),
    };
    let ty = vec(fb, elem, 4)?;
    if !alpha {
        let z = fb.module().constant(Constant::Null { ty })?;
        return fb.module().const_value(z);
    }
    let spec = match *image {
        ImageRef::Slot(index) => match cx.decl.families.texture.oob_alpha.get(&index).copied() {
            Some(spec) => spec,
            None => {
                return Err(cx.fail(format!(
                    "texture slot {index} has no out-of-bounds alpha constant"
                )))
            }
        },
        ImageRef::Heap { .. } => match flagged {
            Some(h) => heap_alpha_one(fb, h)?,
            None => {
                return Err(cx.fail("a heap read's alpha without the handle's flag"));
            }
        },
    };
    let zero = splat(fb, elem, 1, 0)?;
    let one = splat(fb, elem, 1, one)?;
    let alpha = fb.select(spec, one, zero)?;
    fb.composite_construct(ty, &[zero, zero, zero, alpha])
}

/// The `SparseTexture` specialization of texture slot `index`, where `residency` says the access's
/// status word is observed and the slot has a sparse arm or a tile-map arm; `None` otherwise (the
/// word is 0). A written slot without either arm declares the specialization only to drop its
/// writes ([`sparse_writes_kept`]).
pub(crate) fn sparse_spec(cx: &EmitCx<'_>, index: u32, residency: bool) -> Option<SValue> {
    let armed = cx
        .plan
        .families
        .texture
        .slots
        .get(&index)
        .is_some_and(|s| s.sparse_arm || s.tile_map);
    if !residency || !armed {
        return None;
    }
    cx.decl.families.texture.sparse.get(&index).copied()
}

/// Whether a write to `image` lands: the negation of its slot's `SparseTexture` specialization,
/// since Metal drops every write a function makes to a sparse texture; `None` where every write
/// lands (a heap element, or a slot whose writes are not dropped).
fn sparse_writes_kept(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    image: &ImageRef,
) -> BResult<Option<SValue>> {
    let ImageRef::Slot(index) = *image else {
        return Ok(None);
    };
    let drops = cx
        .plan
        .families
        .texture
        .slots
        .get(&index)
        .is_some_and(|s| s.sparse_writes);
    let spec = cx.decl.families.texture.sparse.get(&index).copied();
    match (drops, spec) {
        (true, Some(spec)) => Ok(Some(fb.unary(UnOp::LogicalNot, spec)?)),
        (true, None) => Err(cx.fail(format!(
            "texture slot {index} drops sparse writes without its specialization"
        ))),
        (false, _) => Ok(None),
    }
}

/// `a && b` of two optional conditions (`None`: true).
fn and_kept(
    fb: &mut FunctionBuilder<'_>,
    a: Option<SValue>,
    b: Option<SValue>,
) -> BResult<Option<SValue>> {
    Ok(match (a, b) {
        (None, c) | (c, None) => c,
        (Some(a), Some(b)) => Some(fb.binary(BinOp::LogicalAnd, a, b)?),
    })
}

/// The result and status word of an access whose residency the function observes, under slot
/// specialization `spec` (true: the bound texture is sparse). There `access(fb, true)`, the
/// access's sparse form, returns the struct of the residency code and the result, and the word is
/// 1 where `OpImageSparseTexelsResident` says a texel it read is not resident (bit 0 of Metal's
/// word, `!sparse_color::resident()`), else 0; otherwise `access(fb, false)`, the plain access,
/// and 0, Metal's word for a texture that is not sparse, which is resident everywhere. A
/// specialization constant is uniform, so an implicit-LOD sample in either arm keeps its
/// derivatives. `access` builds everything it reads that must share its block (a sampled image).
pub(crate) fn resident_access<'a, F>(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    spec: SValue,
    access: F,
) -> BResult<(SValue, SValue)>
where
    F: Fn(&mut FunctionBuilder<'_>, bool) -> BResult<SValue> + Copy + 'a,
{
    let sparse: ArmFn<'a, ()> = Box::new(move |fb, _| {
        let r = access(fb, true)?;
        let code = fb.composite_extract(r, &[0])?;
        let value = fb.composite_extract(r, &[1])?;
        let resident = fb.texels_resident(code)?;
        let zero = u32s(fb, 1, 0)?;
        let one = u32s(fb, 1, 1)?;
        let word = fb.select(resident, zero, one)?;
        Ok(ArmEnd::FallThrough(vec![value, word]))
    });
    let plain: ArmFn<'a, ()> = Box::new(move |fb, _| {
        let value = access(fb, false)?;
        let word = u32s(fb, 1, 0)?;
        Ok(ArmEnd::FallThrough(vec![value, word]))
    });
    let Merged(merged) = fb.if_(&mut (), spec, sparse, Some(plain))?;
    match merged[..] {
        [value, word] => Ok((value, word)),
        _ => Err(cx.fail("a sparse arm without its result and word")),
    }
}

/// The tile map of texture slot `index`, where the slot has a tile-map arm.
pub(crate) fn tile_map(cx: &EmitCx<'_>, index: u32) -> Option<crate::memory::views::View> {
    cx.decl.families.texture.tile_maps.get(&index).copied()
}

/// The texel `(x, y, z)` and slice of image coordinate `coord` of an image declared `decl`, for a
/// tile map: a 2D coordinate's `z` is 0, a 2D array's last component is the slice.
pub(crate) fn tile_parts(
    fb: &mut FunctionBuilder<'_>,
    decl: &ImageDecl,
    coord: SValue,
) -> BResult<([SValue; 3], SValue)> {
    let zero = u32s(fb, 1, 0)?;
    let x = fb.composite_extract(coord, &[0])?;
    let y = fb.composite_extract(coord, &[1])?;
    Ok(match (decl.dim, decl.arrayed) {
        (Dim::D3, _) => ([x, y, fb.composite_extract(coord, &[2])?], zero),
        (_, true) => ([x, y, zero], fb.composite_extract(coord, &[2])?),
        _ => ([x, y, zero], zero),
    })
}

/// The words of a tile map that every texel of one level and slice shares, read once: the level's
/// entry (clamped below 16), each axis's tile extent and tile count, and the first bit of the slice
/// (clamped below the map's count).
#[derive(Clone, Copy)]
pub(crate) struct TileLevel {
    map: crate::memory::views::View,
    extents: [SValue; 3],
    counts: [SValue; 3],
    /// The slice's first bit: `slice · B + base`.
    first: SValue,
}

impl TileLevel {
    /// Level `level` and slice `slice` of `map`, as `pipeline::tile_map` lays it out.
    pub(crate) fn new(
        fb: &mut FunctionBuilder<'_>,
        map: crate::memory::views::View,
        level: SValue,
        slice: SValue,
    ) -> BResult<TileLevel> {
        use crate::memory::views::read_word;
        use crate::pipeline::tile_map::{
            LEVELS_WORD, LEVEL_WORDS, MAX_LEVELS, SLICES_WORD, SLICE_BITS_WORD, TILE_WORD,
        };
        let one = u32s(fb, 1, 1)?;
        let word = |fb: &mut FunctionBuilder<'_>, w: u32| -> BResult<SValue> {
            let at = u32s(fb, 1, w)?;
            read_word(fb, map, at)
        };
        let top = u32s(fb, 1, MAX_LEVELS - 1)?;
        let level = fb.glsl(GlslOp::UMin, &[level, top])?;
        let stride = u32s(fb, 1, LEVEL_WORDS)?;
        let first = u32s(fb, 1, LEVELS_WORD)?;
        let entry = fb.binary(BinOp::IMul, level, stride)?;
        let entry = fb.binary(BinOp::IAdd, entry, first)?;
        let base = read_word(fb, map, entry)?;
        let mut extents = [one; 3];
        let mut counts = [one; 3];
        for (axis, (extent, count)) in (0u32..).zip(extents.iter_mut().zip(counts.iter_mut())) {
            let e = word(fb, TILE_WORD + axis)?;
            *extent = fb.glsl(GlslOp::UMax, &[e, one])?;
            let k = u32s(fb, 1, 1 + axis)?;
            let at = fb.binary(BinOp::IAdd, entry, k)?;
            *count = read_word(fb, map, at)?;
        }
        let slices = word(fb, SLICES_WORD)?;
        let last = fb.binary(BinOp::ISub, slices, one)?;
        let slice = fb.glsl(GlslOp::UMin, &[slice, last])?;
        let slice_bits = word(fb, SLICE_BITS_WORD)?;
        let first = fb.binary(BinOp::IMul, slice, slice_bits)?;
        let first = fb.binary(BinOp::IAdd, first, base)?;
        Ok(TileLevel {
            map,
            extents,
            counts,
            first,
        })
    }

    /// The tile along `axis` (x, y, or z, from 0) of texel coordinate `c`, clamped below the
    /// level's tiles, so the lookup stays in the map for any access.
    pub(crate) fn tile(
        &self,
        fb: &mut FunctionBuilder<'_>,
        axis: usize,
        c: SValue,
    ) -> BResult<SValue> {
        let ([ex, ey, ez], [nx, ny, nz]) = (self.extents, self.counts);
        let (extent, count) = match axis {
            0 => (ex, nx),
            1 => (ey, ny),
            _ => (ez, nz),
        };
        let one = u32s(fb, 1, 1)?;
        let last = fb.binary(BinOp::ISub, count, one)?;
        let t = fb.binary(BinOp::UDiv, c, extent)?;
        fb.glsl(GlslOp::UMin, &[t, last])
    }

    /// Whether the tile at `tiles` (each from [`TileLevel::tile`]) is mapped: its bit,
    /// `(tz · ny + ty) · nx + tx` past the slice's first.
    pub(crate) fn mapped(
        &self,
        fb: &mut FunctionBuilder<'_>,
        tiles: [SValue; 3],
    ) -> BResult<SValue> {
        use crate::memory::views::read_word;
        use crate::pipeline::tile_map::BITS_WORD;
        let [tx, ty, tz] = tiles;
        let [nx, ny, _] = self.counts;
        let row = fb.binary(BinOp::IMul, tz, ny)?;
        let row = fb.binary(BinOp::IAdd, row, ty)?;
        let within = fb.binary(BinOp::IMul, row, nx)?;
        let within = fb.binary(BinOp::IAdd, within, tx)?;
        let bit = fb.binary(BinOp::IAdd, self.first, within)?;
        let five = u32s(fb, 1, 5)?;
        let index = fb.binary(BinOp::ShiftRightLogical, bit, five)?;
        let bits = u32s(fb, 1, BITS_WORD)?;
        let index = fb.binary(BinOp::IAdd, index, bits)?;
        let w = read_word(fb, self.map, index)?;
        let mask = u32s(fb, 1, 31)?;
        let shift = fb.binary(BinOp::BitwiseAnd, bit, mask)?;
        let w = fb.binary(BinOp::ShiftRightLogical, w, shift)?;
        let one = u32s(fb, 1, 1)?;
        let w = fb.binary(BinOp::BitwiseAnd, w, one)?;
        let zero = u32s(fb, 1, 0)?;
        fb.binary(BinOp::INotEqual, w, zero)
    }
}

/// Whether texel `texel` of level `level` and slice `slice` is in a mapped tile of `map`, as
/// `pipeline::tile_map` decodes it, through [`TileLevel`].
pub(crate) fn tile_mapped(
    fb: &mut FunctionBuilder<'_>,
    map: crate::memory::views::View,
    texel: [SValue; 3],
    level: SValue,
    slice: SValue,
) -> BResult<SValue> {
    let l = TileLevel::new(fb, map, level, slice)?;
    let [x, y, z] = texel;
    let tiles = [l.tile(fb, 0, x)?, l.tile(fb, 1, y)?, l.tile(fb, 2, z)?];
    l.mapped(fb, tiles)
}

/// The status word of an access of texel `texel` of level `level` and slice `slice` on a slot
/// with a tile-map arm, under slot specialization `spec` (true: the bound texture is sparse): 1
/// where the texel's tile is unmapped (bit 0 of Metal's word, `!sparse_color::resident()`), else 0;
/// and 0 where the texture is not sparse, which reads no word of the map (a neutral buffer stands
/// in for it).
pub(crate) fn tiled_word(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    spec: SValue,
    map: crate::memory::views::View,
    texel: [SValue; 3],
    level: SValue,
    slice: SValue,
) -> BResult<SValue> {
    let sparse: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let mapped = tile_mapped(fb, map, texel, level, slice)?;
        let zero = u32s(fb, 1, 0)?;
        let one = u32s(fb, 1, 1)?;
        let word = fb.select(mapped, zero, one)?;
        Ok(ArmEnd::FallThrough(vec![word]))
    });
    let plain: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let word = u32s(fb, 1, 0)?;
        Ok(ArmEnd::FallThrough(vec![word]))
    });
    let Merged(merged) = fb.if_(&mut (), spec, sparse, Some(plain))?;
    match merged[..] {
        [word] => Ok(word),
        _ => Err(cx.fail("a tile-map arm without its word")),
    }
}

/// `x`, a four-lane held texel (32-bit lanes: a `float`, or an integer in a `u32`) of an access
/// of texture slot `index`, with each lane its `ConstantLanes` specialization names set to that
/// lane's constant, 1 (a float's 1.0, an integer's 1) or 0: where the bound texture is a sparse
/// image, a block of it that holds no mapped tile reads the device's value, where Metal reads a
/// channel the format lacks as its constant. A gather of component `gathered` reads that component
/// in every lane, so each lane takes its constant.
pub(crate) fn constant_lanes(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    index: u32,
    x: SValue,
    gathered: Option<u32>,
) -> BResult<SValue> {
    let Some(tests) = cx.decl.families.texture.constant_lanes.get(&index).copied() else {
        return Err(cx.fail(format!("texture slot {index} has no constant lanes")));
    };
    let ty = fb.value_ty(x)?;
    let elem = match fb.module().types().get(ty)? {
        SType::Vector { elem, len: 4 } => *elem,
        _ => return Err(cx.fail("constant lanes of a texel that is not four lanes")),
    };
    let one = match fb.module().types().get(elem)? {
        SType::Float { width: 32 } => u64::from(1.0f32.to_bits()),
        SType::Int { width: 32, .. } => 1,
        _ => return Err(cx.fail("constant lanes of a texel that is no 32-bit lane")),
    };
    let zero = fb.module().constant(Constant::Null { ty: elem })?;
    let zero = fb.module().const_value(zero)?;
    let one = fb.module().constant(Constant::Scalar {
        ty: elem,
        bits: ConstBits(one),
    })?;
    let one = fb.module().const_value(one)?;
    // Each source lane's constant, made once: a gather's one component, else each lane's.
    let mut made: Vec<(u32, SValue)> = Vec::with_capacity(4);
    let mut lanes = Vec::with_capacity(4);
    for lane in 0..4u32 {
        let source = gathered.unwrap_or(lane);
        let at = source as usize;
        let (Some(&constant), Some(&is_one)) = (tests.constant.get(at), tests.one.get(at)) else {
            return Err(cx.fail(format!("constant lanes of component {source}")));
        };
        let value = match made.iter().find(|m| m.0 == source) {
            Some(&(_, v)) => v,
            None => {
                let v = fb.select(is_one, one, zero)?;
                made.push((source, v));
                v
            }
        };
        let v = fb.composite_extract(x, &[lane])?;
        lanes.push(fb.select(constant, value, v)?);
    }
    fb.composite_construct(ty, &lanes)
}

/// The `{T, i8}` value of `texel` with status word `word` (0 where none is given), or the texel
/// itself where `result` is no such struct.
pub(crate) fn with_word(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    result: &crate::spirv::ids::DemoteTy,
    texel: SValue,
    word: Option<SValue>,
) -> BResult<SValue> {
    let Some(ty) = crate::lower::arith::spirv_type(fb.module(), result)? else {
        return Err(cx.fail("a texel result without a lowered type"));
    };
    match fb.module().types().get(ty)? {
        SType::Struct { .. } => {
            let word = match word {
                Some(w) => w,
                None => u32s(fb, 1, 0)?,
            };
            fb.composite_construct(ty, &[texel, word])
        }
        _ => Ok(texel),
    }
}

fn read(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, r: &Read) -> BResult<SValue> {
    // A two-plane 3D texture's read takes its first texel ([`planes_volume`]).
    let volume = decl_of(cx, &r.image).is_some_and(|d| planes_volume(&d));
    // A storage read's level is its level views'; a sampled read's, its `Lod`.
    let (view, lod) = if r.level_views {
        (r.level.filter(|_| !volume), None)
    } else {
        (None, r.level)
    };
    let Loaded {
        decl,
        img,
        live,
        within,
        flagged,
    } = image_in(cx, fb, &r.image, view, r.heap_class)?;
    let plain = match (volume, r.edge) {
        (true, _) => Some(u32s(fb, coord_len(&decl), 0)?),
        (false, None) => Some(coordinate(cx, fb, &decl, &r.at)?),
        (false, Some(_)) => None,
    };
    let mut level = match (lod, volume) {
        (Some(_), true) => Some(u32s(fb, 1, 0)?),
        (Some(l), false) => Some(cx.operand(fb, l)?),
        (None, _) => None,
    };
    let mut sample = match r.sample {
        Some(s) => Some(cx.operand(fb, s)?),
        None => None,
    };
    let sampled = decl.usage == ImageUsage::Sampled;
    let spec = match r.image {
        ImageRef::Slot(index) => sparse_spec(cx, index, r.residency),
        ImageRef::Heap { .. } => None,
    };
    // A clamp-to-edge read's plane coordinate is clamped into the image first.
    let (mut coord, edge) = match (plain, r.edge) {
        (Some(c), _) => (c, None),
        (None, Some(_)) => {
            let (c, at) = edge_coordinate(cx, fb, &decl, img, r, level)?;
            (c, Some(at))
        }
        (None, None) => return Err(cx.fail("a read without its coordinate")),
    };
    // Whether every operand of the access is in range: under the shader's check, each operand is
    // then clamped into the image so the access stays inside it; under a sparse arm, the test
    // decides the status word alone, the access left to the device's robustness otherwise. A
    // clamp-to-edge read is tested as it accesses, with its clamped coordinate against the size it
    // clamped into, so only its level, layer, or sample index can fail.
    let mut in_range = None;
    if (r.checked() || spec.is_some()) && !volume {
        let size = match (edge, level) {
            // The clamp-to-edge size query's, at the level it clamped.
            (Some(at), l) => {
                if let (Some((levels, clamped)), Some(l)) = (at.levels, l) {
                    let ok = fb.binary(BinOp::ULessThan, l, levels)?;
                    in_range = Some(and(fb, in_range, ok)?);
                    if r.checked() {
                        level = Some(clamped);
                    }
                }
                at.size
            }
            (None, Some(l)) if sampled => {
                let levels = fb.query(ImageQuery::Levels, img)?;
                let ok = fb.binary(BinOp::ULessThan, l, levels)?;
                in_range = Some(and(fb, in_range, ok)?);
                // A size query's level must be the image's, so it reads the clamped one.
                let clamped = clamp_below(fb, l, levels, 1)?;
                if r.checked() {
                    level = Some(clamped);
                }
                fb.query(ImageQuery::SizeLod(clamped), img)?
            }
            (None, _) => fb.query(ImageQuery::Size, img)?,
        };
        let n = coord_len(&decl);
        let lt = fb.binary(BinOp::ULessThan, coord, size)?;
        let ok = if n == 1 { lt } else { fb.unary(UnOp::All, lt)? };
        in_range = Some(and(fb, in_range, ok)?);
        if r.checked() {
            coord = clamp_below(fb, coord, size, n)?;
        }
        if let Some(s) = sample {
            let samples = fb.query(ImageQuery::Samples, img)?;
            let ok = fb.binary(BinOp::ULessThan, s, samples)?;
            in_range = Some(and(fb, in_range, ok)?);
            if r.checked() {
                sample = Some(clamp_below(fb, s, samples, 1)?);
            }
        }
    }
    // A level the bound texture lacks is out of bounds whatever the device's robustness: the
    // element read is another level's view, or a neutral.
    let within = if r.level_views {
        let Some(within) = within else {
            return Err(cx.fail("a level-view read of a slot without level views"));
        };
        Some(within)
    } else {
        None
    };
    // Where the shader selects Metal's out-of-bounds texel: under its check, and past a level-view
    // read's level count.
    let inb = match (r.checked(), within) {
        (true, Some(w)) => Some(and(fb, in_range, w)?),
        (true, None) => in_range,
        (false, w) => w,
    };
    let ops = ImageOperands {
        lod: level,
        sample,
        ..ImageOperands::default()
    };
    // A storage read takes the function's coherence; a sampled image is read-only.
    let mem = if sampled {
        TexelMemory::PRIVATE
    } else {
        texel_memory(cx, fb, &r.image)?
    };
    // A slot the draw reads through a snapshot (`LoweringInputs::snapshot_textures`): its
    // interface fact admits only the plain fetch of a level the snapshot's copy makes, through the
    // slot's image, which the copy fetches from.
    let snapshot = match r.image {
        ImageRef::Slot(index) => {
            let plain = sampled
                && spec.is_none()
                && sample.is_none()
                && r.edge.is_none()
                && !r.level_views
                && within.is_none();
            crate::lower::fragment::snapshot_prepare(cx, fb, index, img, plain)?
        }
        ImageRef::Heap { .. } => None,
    };
    let access = move |fb: &mut FunctionBuilder<'_>, sparse: bool| -> BResult<SValue> {
        match (sampled, sparse) {
            (true, false) => fb.fetch(img, coord, &ops),
            (true, true) => fb.sparse_fetch(img, coord, &ops),
            (false, false) => fb.read(img, coord, &ops, mem),
            (false, true) => fb.sparse_read(img, coord, &ops, mem),
        }
    };
    let tiled = match r.image {
        ImageRef::Slot(index) if spec.is_some() => tile_map(cx, index),
        _ => None,
    };
    let mut inb = inb;
    let (texel, word) = match spec {
        Some(spec) => {
            let (texel, word) = match tiled {
                // A tile map: the plain access of the ordinary image, whose unmapped texels the
                // consumer keeps at zero, and the residency of the texel's tile.
                Some(map) => {
                    let texel = access(fb, false)?;
                    let at = match (level, view) {
                        (Some(l), _) => l,
                        (None, Some(v)) => cx.operand(fb, v)?,
                        (None, None) => u32s(fb, 1, 0)?,
                    };
                    let (xyz, slice) = tile_parts(fb, &decl, coord)?;
                    (texel, tiled_word(cx, fb, spec, map, xyz, at, slice)?)
                }
                None => resident_access(cx, fb, spec, access)?,
            };
            // A read outside the image is resident, as Metal answers for every texture, sparse or
            // not, mapped or not: a device's code for it is not Vulkan's to state (lavapipe reports
            // it not resident), and the shader's clamp would read another texel's.
            let resident = match (r.checked(), in_range, within) {
                // The shader's own test.
                (true, ..) => inb,
                (false, Some(ok), Some(w)) => Some(fb.binary(BinOp::LogicalAnd, ok, w)?),
                (false, Some(ok), None) => Some(ok),
                (false, None, w) => w,
            };
            let word = match resident {
                Some(ok) => {
                    let zero = u32s(fb, 1, 0)?;
                    fb.select(ok, word, zero)?
                }
                None => word,
            };
            // A texel that is not resident reads Metal's unmapped value: zero memory as the
            // bound format reads it, the out-of-bounds texel. Under the sparse arm a device reads
            // that only under `residencyNonResidentStrict`; under a tile map the image holds it,
            // but where the texture is a sparse image, a block of it that holds no mapped tile
            // reads the device's value.
            let zero = u32s(fb, 1, 0)?;
            let kept = fb.binary(BinOp::IEqual, word, zero)?;
            inb = Some(and(fb, inb, kept)?);
            (texel, Some(word))
        }
        None => match snapshot {
            Some(s) => (
                crate::lower::fragment::snapshot_read(cx, fb, s, coord)?,
                None,
            ),
            None => (access(fb, false)?, None),
        },
    };
    // A heap member of a form a heap reads: its raw texel's value, Metal's out-of-bounds texel
    // selected against its own image, which the float array's test below then leaves.
    let texel = match r.heap_raw {
        true => {
            let (texel, member) = heap_raw_texel(cx, fb, r, texel)?;
            if let Some(i) = inb {
                inb = Some(fb.binary(BinOp::LogicalOr, member, i)?);
            }
            texel
        }
        false => texel,
    };
    // An `A8Unorm` storage member's texel held in red, Metal's `(0, 0, 0, r)` before Metal's
    // out-of-bounds texel is selected.
    let texel = match (r.heap_alpha_red, r.image) {
        (true, ImageRef::Heap { handle, .. }) => heap_alpha_in_red(cx, fb, handle, texel, true)?,
        _ => texel,
    };
    // An extended-range texel's value, decoded from its raw code before Metal's out-of-bounds
    // texel is selected.
    let texel = match r.extended {
        Some(form) => {
            let table = extended_table(cx, form)?;
            let half = r.texel.conv == TexelConv::F16;
            crate::lower::arith::extended_range::decode(fb, form, texel, table, half)?
        }
        None => texel,
    };
    let texel = match inb {
        Some(inb) => {
            let oob = out_of_bounds_texel(cx, fb, &decl, &r.image, r.alpha, flagged)?;
            fb.select(inb, texel, oob)?
        }
        None => texel,
    };
    // A texture buffer's swizzle, on the texel Metal reads, out of bounds too.
    let texel = match (r.swizzle, r.image) {
        (true, ImageRef::Slot(index)) => {
            let Some(word) = cx.decl.families.texture.swizzle.get(&index).copied() else {
                return Err(cx.fail(format!("texture slot {index} has no swizzle constant")));
            };
            swizzled(fb, &decl, texel, word)?
        }
        _ => texel,
    };
    // An 8-bit sRGB texture held as its UNORM twin: the codes the slot's image reads, decoded by
    // the device's transfer.
    let texel = match r.raw {
        true => srgb_texel(fb, r.texel, texel, SRGB_TEXEL)?,
        false => {
            let half = half_class(cx, &r.image, r.texel);
            from_image(fb, &decl, r.texel, texel, half)?
        }
    };
    // A swizzled texture's swizzle at a storage slot, on the converted texel.
    let texel = match (r.storage_swizzle, r.image) {
        (true, ImageRef::Slot(index)) => {
            let Some(word) = cx.decl.families.texture.swizzle.get(&index).copied() else {
                return Err(cx.fail(format!("texture slot {index} has no swizzle constant")));
            };
            swizzled_texel(cx, fb, r.texel, texel, word)?
        }
        _ => texel,
    };
    // The lanes Metal reads as constants, on the texel it reads.
    let texel = match (r.constant_lanes, r.image) {
        (true, ImageRef::Slot(index)) => constant_lanes(cx, fb, index, texel, None)?,
        _ => texel,
    };
    // A texture of another class: its own texel's words, out of bounds too, as declared.
    let texel = match r.declared {
        Some(declared) => reinterpret(fb, r.texel, declared, texel)?,
        None => texel,
    };
    let texel = live_or_zero(fb, live, texel)?;
    // `{T, i8}`: the status word is the sparse arm's where the function observes it, else 0,
    // Metal's for a resident access.
    with_word(cx, fb, &r.result, texel, word)
}

/// The texel of heap read `r` (`Read::heap_raw`) where its handle's class index names a form a
/// heap reads (`ExtendedRange::HEAP_READ`, past `metal::texel_format::heap_extended_base`), else
/// `texel`, the float array's: that element's raw texel fetched from the group's unsigned integer
/// class (`HeapPlan::raw`), its level and coordinate (a clamp-to-edge read's clamped into it)
/// tested against that image and clamped into it, decoded by the form, Metal's out-of-bounds
/// texel outside it (`heap_raw::define`). With whether the handle names such a form (a `bool`).
fn heap_raw_texel(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    r: &Read,
    texel: SValue,
) -> BResult<(SValue, SValue)> {
    let ImageRef::Heap { group, handle } = r.image else {
        return Err(cx.fail("a raw heap read of a slot"));
    };
    let Some((_, decl)) = cx
        .plan
        .families
        .texture
        .heap
        .get(&group)
        .and_then(|h| h.raw)
    else {
        return Err(cx.fail(format!("heap group {group} has no unsigned class")));
    };
    let half = r.texel.conv == TexelConv::F16;
    let Some(routine) = cx.decl.families.texture.heap_raw.get(&half).copied() else {
        return Err(cx.fail("a raw heap read without its routine"));
    };
    let h = cx.operand(fb, handle)?;
    let (form, member) = heap_raw_form(fb, h)?;
    let raw_arm: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let img = heap_raw_image(cx, fb, group, h)?;
        let level = match r.level {
            Some(l) => Some(cx.operand(fb, l)?),
            None => None,
        };
        let n = coord_len(&decl);
        let (coord, level, ok) = match r.edge {
            // A clamp-to-edge read's coordinate, clamped into this image.
            Some(_) => {
                let (c, at) = edge_coordinate(cx, fb, &decl, img, r, level)?;
                match (at.levels, level) {
                    (Some((levels, clamped)), Some(l)) => {
                        let ok = fb.binary(BinOp::ULessThan, l, levels)?;
                        (c, Some(clamped), Some(ok))
                    }
                    _ => (c, level, None),
                }
            }
            None => {
                let c = coordinate(cx, fb, &decl, &r.at)?;
                let (size, level, ok) = match level {
                    Some(l) => {
                        let levels = fb.query(ImageQuery::Levels, img)?;
                        let ok = fb.binary(BinOp::ULessThan, l, levels)?;
                        let clamped = clamp_below(fb, l, levels, 1)?;
                        let size = fb.query(ImageQuery::SizeLod(clamped), img)?;
                        (size, Some(clamped), Some(ok))
                    }
                    None => (fb.query(ImageQuery::Size, img)?, None, None),
                };
                let lt = fb.binary(BinOp::ULessThan, c, size)?;
                let inside = if n == 1 { lt } else { fb.unary(UnOp::All, lt)? };
                let ok = and(fb, ok, inside)?;
                (clamp_below(fb, c, size, n)?, level, Some(ok))
            }
        };
        let ops = ImageOperands {
            lod: level,
            ..ImageOperands::default()
        };
        let texel = fb.fetch(img, coord, &ops)?;
        let ok = match ok {
            Some(ok) => ok,
            None => {
                let bool_ = fb.module().ty(SType::Bool)?;
                let t = fb.module().constant(Constant::Scalar {
                    ty: bool_,
                    bits: ConstBits(1),
                })?;
                fb.module().const_value(t)?
            }
        };
        let v = fb
            .call(routine, &[texel, form, ok])?
            .ok_or_else(|| cx.fail("the raw heap routine returned nothing"))?;
        Ok(ArmEnd::FallThrough(vec![v]))
    });
    let own_arm: ArmFn<'_, EmitCx<'_>> =
        Box::new(move |_fb, _cx| Ok(ArmEnd::FallThrough(vec![texel])));
    let Merged(merged) = fb.if_(cx, member, raw_arm, Some(own_arm))?;
    match merged[..] {
        [v] => Ok((v, member)),
        _ => Err(cx.fail("a raw heap read's branch without its texel")),
    }
}

/// The form heap handle `h` (a `u32`) names by its class index past the heap's own classes
/// (`metal::texel_format::heap_extended_base`): its index in `ExtendedRange::HEAP_READ` (a `u32`,
/// wrapped past the forms where it names none), and whether it names one (a `bool`).
pub(crate) fn heap_raw_form(fb: &mut FunctionBuilder<'_>, h: SValue) -> BResult<(SValue, SValue)> {
    use crate::memory::handles::HANDLE_TEXEL_CLASS;
    let shift = u32s(fb, 1, HANDLE_TEXEL_CLASS.trailing_zeros())?;
    let mask = u32s(
        fb,
        1,
        HANDLE_TEXEL_CLASS >> HANDLE_TEXEL_CLASS.trailing_zeros(),
    )?;
    let index = fb.binary(BinOp::ShiftRightLogical, h, shift)?;
    let index = fb.binary(BinOp::BitwiseAnd, index, mask)?;
    let base = u32s(fb, 1, crate::metal::texel_format::heap_extended_base())?;
    let form = fb.binary(BinOp::ISub, index, base)?;
    let forms = u32s(
        fb,
        1,
        crate::metal::format::ExtendedRange::HEAP_READ.len() as u32,
    )?;
    let member = fb.binary(BinOp::ULessThan, form, forms)?;
    Ok((form, member))
}

/// The index heap handle `h` (a `u32`) carries in its class field
/// (`memory::handles::HANDLE_TEXEL_CLASS`).
fn heap_class_field(fb: &mut FunctionBuilder<'_>, h: SValue) -> BResult<SValue> {
    use crate::memory::handles::HANDLE_TEXEL_CLASS;
    let shift = u32s(fb, 1, HANDLE_TEXEL_CLASS.trailing_zeros())?;
    let mask = u32s(
        fb,
        1,
        HANDLE_TEXEL_CLASS >> HANDLE_TEXEL_CLASS.trailing_zeros(),
    )?;
    let index = fb.binary(BinOp::ShiftRightLogical, h, shift)?;
    fb.binary(BinOp::BitwiseAnd, index, mask)
}

/// Whether class index `index` (a `u32`, [`heap_class_field`]) names a member of numeric class
/// `class` (a `bool`): a signed or unsigned integer color format's marker
/// (`metal::texel_format::heap_numeric_base`), or for the float class one of the heap's own
/// classes, a float format's (`heap_extended_base`: an extended-range form's, past them, is read
/// by the group's own read).
fn names_class(fb: &mut FunctionBuilder<'_>, index: SValue, class: TexelClass) -> BResult<SValue> {
    use crate::metal::texel_format::{heap_extended_base, heap_numeric_base};
    match class {
        TexelClass::Float => {
            let base = u32s(fb, 1, heap_extended_base())?;
            fb.binary(BinOp::ULessThan, index, base)
        }
        TexelClass::Sint => {
            let k = u32s(fb, 1, heap_numeric_base())?;
            fb.binary(BinOp::IEqual, index, k)
        }
        TexelClass::Uint => {
            let k = u32s(fb, 1, heap_numeric_base() + 1)?;
            fb.binary(BinOp::IEqual, index, k)
        }
        TexelClass::Uint64 => {
            let bool_ = fb.module().ty(SType::Bool)?;
            let f = fb.module().constant(Constant::Scalar {
                ty: bool_,
                bits: ConstBits(0),
            })?;
            fb.module().const_value(f)
        }
    }
}

/// Read `r`; for a heap read with siblings (`Read::heap_cross`), the sibling of the numeric class
/// the handle's class index names ([`names_class`]), its words handed back as `r`'s, else `r`
/// itself: Metal reads a member of another class in its own class and hands back the words, as
/// at a slot.
fn read_crossed(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, r: &Read) -> BResult<SValue> {
    let ImageRef::Heap { handle, .. } = r.image else {
        return read(cx, fb, r);
    };
    if r.heap_cross.is_empty() {
        return read(cx, fb, r);
    }
    let h = cx.operand(fb, handle)?;
    let index = heap_class_field(fb, h)?;
    read_arms(cx, fb, index, &r.heap_cross, r)
}

/// The first of `siblings` whose class `index` names, read, else the rest's, else `own`.
fn read_arms(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    index: SValue,
    siblings: &[Read],
    own: &Read,
) -> BResult<SValue> {
    let Some((first, rest)) = siblings.split_first() else {
        return read(cx, fb, own);
    };
    let Some(class) = first.heap_class else {
        return Err(cx.fail("a heap read's sibling without its class"));
    };
    let named = names_class(fb, index, class)?;
    let then: ArmFn<'_, EmitCx<'_>> =
        Box::new(move |fb, cx| Ok(ArmEnd::FallThrough(vec![read(cx, fb, first)?])));
    let other: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        Ok(ArmEnd::FallThrough(vec![read_arms(
            cx, fb, index, rest, own,
        )?]))
    });
    let Merged(merged) = fb.if_(cx, named, then, Some(other))?;
    match merged[..] {
        [v] => Ok(v),
        _ => Err(cx.fail("a heap read's class branch without its texel")),
    }
}

/// The image of heap handle `h`'s element in heap group `group`'s unsigned integer class
/// (`HeapPlan::raw`), where the binder holds the raw texels of a texture a heap reads by them
/// alone, loaded through a `NonUniform` access chain.
pub(crate) fn heap_raw_image(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    group: u32,
    h: SValue,
) -> BResult<SValue> {
    let raw = cx
        .plan
        .families
        .texture
        .heap
        .get(&group)
        .and_then(|h| h.raw);
    let Some((class, _)) = raw else {
        return Err(cx.fail(format!("heap group {group} has no unsigned class")));
    };
    let Some(g) = cx.decl.families.texture.heap.get(&class).copied() else {
        return Err(cx.fail(format!("heap class {class:?} is not declared")));
    };
    let element = heap_element_index(cx, fb, h)?;
    let ptr = fb.module().global_value(g)?;
    let p = fb.access_chain_non_uniform(ptr, element)?;
    fb.load(p, MemAccess::default())
}

/// The table an extended-range texel of `form` converts through: the module's for an `_sRGB`
/// form (`ArithPlan::extended_range_srgb`), the form's own for a padded `_sRGB` one
/// (`ArithPlan::padded_srgb`), none for a linear one.
pub(crate) fn extended_table(
    cx: &mut EmitCx<'_>,
    form: crate::metal::format::ExtendedRange,
) -> BResult<Option<crate::spirv::ids::GlobalId>> {
    let arith = &cx.decl.families.arith;
    let table = match form.padded_srgb() {
        true => arith.padded_srgb.get(&form).copied(),
        false => arith.extended_range_srgb,
    };
    match (form.srgb() || form.padded_srgb(), table) {
        (false, _) => Ok(None),
        (true, Some(table)) => Ok(Some(table)),
        (true, None) => Err(cx.fail("an sRGB extended-range texel without Metal's table")),
    }
}

fn write(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, w: &Write) -> BResult<()> {
    let Loaded {
        decl,
        img,
        live,
        within,
        ..
    } = image(cx, fb, &w.image, w.level)?;
    let coord = coordinate(cx, fb, &decl, &w.at)?;
    // A two-plane 3D texture's write stores level 0's texel `(x, y)` of its first slice, and
    // nothing past the level's width or height or at another level ([`planes_volume`]).
    let volume = match planes_volume(&decl) {
        true => Some(volume_write(cx, fb, img, coord, w.level)?),
        false => None,
    };
    let coord = volume.map_or(coord, |(at, _)| at);
    let v = cx.operand(fb, w.value)?;
    // A texture of another numeric class: the declared lanes as its own class's texel.
    let v = match w.cross {
        Some(c) => cross_write(fb, c, w.texel, v)?,
        None => v,
    };
    // A swizzled texture's swizzle at a storage slot, undone on the value.
    let v = match (w.swizzle, w.image) {
        (true, ImageRef::Slot(index)) => {
            let Some(word) = cx.decl.families.texture.swizzle.get(&index).copied() else {
                return Err(cx.fail(format!("texture slot {index} has no swizzle constant")));
            };
            unswizzled(cx, fb, w.texel, v, word)?
        }
        _ => v,
    };
    // An extended-range texel's raw code, encoded as Metal writes it.
    let texel = match w.extended {
        Some(form) => {
            let table = extended_table(cx, form)?;
            crate::lower::arith::extended_range::encode(fb, form, v, table)?
        }
        None => to_image(fb, &decl, w.texel, v)?,
    };
    // An integer texel's lanes, saturated to the format's channels as Metal stores them.
    let texel = saturated(cx, fb, &decl, (w.saturate, w.texel), texel)?;
    // An `A8Unorm` storage member's alpha, stored in its image's red.
    let texel = match (w.heap_alpha_red, w.image) {
        (true, ImageRef::Heap { handle, .. }) => heap_alpha_in_red(cx, fb, handle, texel, false)?,
        _ => texel,
    };
    let mem = texel_memory(cx, fb, &w.image)?;
    // A write to a sparse texture is dropped.
    let kept = sparse_writes_kept(cx, fb, &w.image)?;
    let within = and_kept(fb, within, kept)?;
    if let Some((_, stored)) = volume {
        let mut cond = stored;
        for c in [live, within].into_iter().flatten() {
            cond = fb.binary(BinOp::LogicalAnd, cond, c)?;
        }
        let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
            fb.write(img, coord, texel, &ImageOperands::default(), mem)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        fb.if_(&mut (), cond, then, None)?;
        return Ok(());
    }
    if !w.guard {
        // A write through the null heap element, or at a level the texture lacks, is dropped.
        let cond = match (live, within) {
            (None, None) => return fb.write(img, coord, texel, &ImageOperands::default(), mem),
            (Some(c), None) | (None, Some(c)) => c,
            (Some(a), Some(b)) => fb.binary(BinOp::LogicalAnd, a, b)?,
        };
        let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
            fb.write(img, coord, texel, &ImageOperands::default(), mem)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        fb.if_(&mut (), cond, then, None)?;
        return Ok(());
    }
    let n = coord_len(&decl);
    let size = fb.query(ImageQuery::Size, img)?;
    let lt = fb.binary(BinOp::ULessThan, coord, size)?;
    let mut inb = if n == 1 { lt } else { fb.unary(UnOp::All, lt)? };
    if let Some(live) = live {
        inb = fb.binary(BinOp::LogicalAnd, inb, live)?;
    }
    if let Some(within) = within {
        inb = fb.binary(BinOp::LogicalAnd, inb, within)?;
    }
    let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
        fb.write(img, coord, texel, &ImageOperands::default(), mem)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut (), inb, then, None)?;
    Ok(())
}

/// A two-plane 3D texture's write at `coord` of level `level` ([`planes_volume`]): the coordinate
/// of texel `(x, y)` of its first slice, and whether Metal stores it there (`x` and `y` within the
/// level, which `img` is, and the level 0).
fn volume_write(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    img: SValue,
    coord: SValue,
    level: Option<ValueId>,
) -> BResult<(SValue, SValue)> {
    let size = fb.query(ImageQuery::Size, img)?;
    let lt = fb.binary(BinOp::ULessThan, coord, size)?;
    let in_x = fb.composite_extract(lt, &[0])?;
    let in_y = fb.composite_extract(lt, &[1])?;
    let mut stored = fb.binary(BinOp::LogicalAnd, in_x, in_y)?;
    if let Some(l) = level {
        let l = cx.operand(fb, l)?;
        let zero = u32s(fb, 1, 0)?;
        let first = fb.binary(BinOp::IEqual, l, zero)?;
        stored = fb.binary(BinOp::LogicalAnd, stored, first)?;
    }
    let zeros = u32s(fb, 3, 0)?;
    let at = fb.vector_shuffle(coord, zeros, &[0, 1, 5])?;
    Ok((at, stored))
}

/// The texel operands of a storage-image read or write of `image`: the function's coherence
/// (`sync::memory_model`) and, for a volatile image under the Vulkan model, `VolatileTexel`
/// ([`super::volatile_texel`]).
fn texel_memory(
    cx: &EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    image: &ImageRef,
) -> BResult<TexelMemory> {
    let mem = crate::lower::sync::memory_model::coherence(cx.plan).texel(fb)?;
    Ok(mem.with_volatile(super::volatile_texel(cx.plan, image)))
}

/// `min(a, b)` of two `u32`s.
fn umin(fb: &mut FunctionBuilder<'_>, a: SValue, b: SValue) -> BResult<SValue> {
    let lt = fb.binary(BinOp::ULessThan, a, b)?;
    fb.select(lt, a, b)
}

/// The four-lane held texel of a slice element `v` of `lanes` lanes: the element's lanes, then 0
/// in every later lane, alpha included, whatever the texture's format (measured: a 1-lane element
/// is `(v, 0, 0, 0)`, a 2-lane one `(x, y, 0, 0)`, a 4-lane one the texel).
fn slice_texel(fb: &mut FunctionBuilder<'_>, form: TexelForm, v: SValue) -> BResult<SValue> {
    let elem = match form.conv {
        TexelConv::F32 | TexelConv::F16 => f32_ty(fb)?,
        TexelConv::I32 | TexelConv::I16 { .. } => u32_ty(fb)?,
        TexelConv::U64Words { .. } => u64_ty(fb)?,
    };
    let ty = vec(fb, elem, 4)?;
    let zero = |fb: &mut FunctionBuilder<'_>| -> BResult<SValue> {
        let z = fb.module().constant(Constant::Null { ty: elem })?;
        fb.module().const_value(z)
    };
    match form.lanes {
        1 => {
            let z = zero(fb)?;
            fb.composite_construct(ty, &[v, z, z, z])
        }
        2 => {
            let x = fb.composite_extract(v, &[0])?;
            let y = fb.composite_extract(v, &[1])?;
            let z = zero(fb)?;
            fb.composite_construct(ty, &[x, y, z, z])
        }
        _ => Ok(v),
    }
}

/// Copy imageblock slice `s` to its texture: a counted loop over the block's cells, `k` the cell
/// `(k mod bw, k / bw)` of a `bw · bh` block clipped to the imageblock, whose element at the
/// field's offset is written at `coord + (x, y)`, discarded outside the texture. A cooperative
/// slice (`super::cooperative`) runs in every invocation: cells `k = lin, lin + n, ...`, `lin`
/// the invocation's index in the threadgroup and `n` its threads, then a workgroup barrier over
/// the arena, so that no invocation stores a cell before every share is written.
fn slice<'c>(cx: &mut EmitCx<'c>, fb: &mut FunctionBuilder<'_>, s: &Slice) -> BResult<()> {
    let Loaded {
        decl,
        img,
        live,
        within,
        ..
    } = image(cx, fb, &s.image, s.level)?;
    // A write through the null heap element, at a level the texture lacks, or to a sparse texture
    // is dropped.
    let within = and_kept(fb, live, within)?;
    let kept = sparse_writes_kept(cx, fb, &s.image)?;
    let within = and_kept(fb, within, kept)?;
    let origin = coordinate(cx, fb, &decl, &s.at)?;
    let (w, h) = imageblock::extent(fb)?;
    let (bw, bh) = match s.size {
        Some(v) => {
            let v = cx.operand(fb, v)?;
            let x = fb.composite_extract(v, &[0])?;
            let y = fb.composite_extract(v, &[1])?;
            (umin(fb, x, w)?, umin(fb, y, h)?)
        }
        None => (w, h),
    };
    let n = fb.binary(BinOp::IMul, bw, bh)?;
    let u = u32_ty(fb)?;
    let (first, step) = match s.cooperative {
        true => shares(cx, fb)?,
        false => (u32s(fb, 1, 0)?, u32s(fb, 1, 1)?),
    };
    let var = fb.local_var(u, None)?;
    let counter = fb.var_ptr(var)?;
    fb.store(counter, first, MemAccess::default())?;
    let dims = coord_len(&decl);
    let s = *s;
    let mem = texel_memory(cx, fb, &s.image)?;
    let body: LoopFn<'_, EmitCx<'c>> = Box::new(move |fb, cx, lc| {
        let k = fb.load(counter, MemAccess::default())?;
        let done = fb.binary(BinOp::UGreaterThanEqual, k, n)?;
        let exit: ArmFn<'_, EmitCx<'c>> = Box::new(move |fb, _| {
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(cx, done, exit, None)?;
        let next = fb.binary(BinOp::IAdd, k, step)?;
        fb.store(counter, next, MemAccess::default())?;
        let x = fb.binary(BinOp::UMod, k, bw)?;
        let y = fb.binary(BinOp::UDiv, k, bw)?;
        let cell = imageblock::cell_offset(cx, fb, x, y, s.cell_bytes)?;
        let field = u32s(fb, 1, s.field)?;
        let off = fb.binary(BinOp::IAdd, cell, field)?;
        let v = imageblock::load(cx, fb, s.leaf, s.contained, off)?;
        let held = slice_texel(fb, s.texel, v)?;
        let form = TexelForm {
            conv: s.texel.conv,
            lanes: 4,
        };
        let texel = to_image(fb, &decl, form, held)?;
        let ty = vec(fb, u, dims)?;
        let delta = if dims == 2 {
            fb.composite_construct(ty, &[x, y])?
        } else {
            let z = u32s(fb, 1, 0)?;
            fb.composite_construct(ty, &[x, y, z])?
        };
        let at = fb.binary(BinOp::IAdd, origin, delta)?;
        if !s.guard {
            // A level the texture lacks drops the slice, as the null heap element does.
            let Some(within) = within else {
                return fb.write(img, at, texel, &ImageOperands::default(), mem);
            };
            let then: ArmFn<'_, EmitCx<'c>> = Box::new(move |fb, _| {
                fb.write(img, at, texel, &ImageOperands::default(), mem)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            fb.if_(cx, within, then, None)?;
            return Ok(());
        }
        let size = fb.query(ImageQuery::Size, img)?;
        let lt = fb.binary(BinOp::ULessThan, at, size)?;
        let mut inb = fb.unary(UnOp::All, lt)?;
        if let Some(within) = within {
            inb = fb.binary(BinOp::LogicalAnd, inb, within)?;
        }
        let then: ArmFn<'_, EmitCx<'c>> = Box::new(move |fb, _| {
            fb.write(img, at, texel, &ImageOperands::default(), mem)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        fb.if_(cx, inb, then, None)?;
        Ok(())
    });
    fb.loop_(cx, LoopSpec::default(), body)?;
    if s.cooperative {
        let flags = crate::air::intrinsics::MemFlags {
            threadgroup_imageblock: true,
            ..Default::default()
        };
        let sem = crate::lower::sync::barrier::semantics(
            flags,
            MemoryOrder::AcquireRelease,
            memory_model(cx.plan),
        );
        fb.control_barrier(Scope::Workgroup, Scope::Workgroup, sem)?;
    }
    Ok(())
}

/// A cooperative slice's first cell and step: the invocation's index in the threadgroup and the
/// threadgroup's threads.
fn shares(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<(SValue, SValue)> {
    use crate::lower::dispatch::KernelBuiltin;
    let decl: &crate::lower::emit::Declared = cx.decl;
    let Some(k) = decl.families.dispatch.kernel.as_ref() else {
        return Err(cx.fail("a cooperative slice write outside a kernel"));
    };
    let lin = k.value(fb, KernelBuiltin::ThreadIndexInThreadgroup, &mut |fb| {
        cx.push_field(fb, crate::pipeline::sym::PushField::DispatchRegion)
    })?;
    let threads = k.running_threads(fb)?;
    let tx = fb.composite_extract(threads, &[0])?;
    let ty_ = fb.composite_extract(threads, &[1])?;
    let tz = fb.composite_extract(threads, &[2])?;
    let txy = fb.binary(BinOp::IMul, tx, ty_)?;
    let n = fb.binary(BinOp::IMul, txy, tz)?;
    Ok((lin, n))
}

fn query(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, q: &QueryOp) -> BResult<SValue> {
    // A size query at a level answers from the image's first level ([`level_extent`]): a storage
    // slot's first level view, a sampled image's level 0.
    let storage = decl_of(cx, &q.image).is_some_and(|d| d.usage == ImageUsage::Storage);
    let Loaded {
        decl, img, live, ..
    } = image(cx, fb, &q.image, None)?;
    // The unsigned integer class of a sampled float group, which holds its members of a form a
    // heap reads; an integer, depth, or storage group has none, and holds no such member. And
    // the group's arrays of the other numeric types, which hold the members of those classes.
    let mut arms: Vec<(SValue, crate::spirv::ids::GlobalId)> = Vec::new();
    let mut held = None;
    if let (true, ImageRef::Heap { group, handle }) = (q.heap_raw || q.heap_cross, q.image) {
        let plan = cx.plan.families.texture.heap.get(&group);
        let raw = plan.and_then(|h| h.raw).filter(|_| q.heap_raw);
        let cross: Vec<_> = match (q.heap_cross, plan) {
            (true, Some(h)) => h.cross.iter().map(|(k, (c, _))| (*k, *c)).collect(),
            _ => Vec::new(),
        };
        if raw.is_some() || !cross.is_empty() {
            let h = cx.operand(fb, handle)?;
            let index = heap_class_field(fb, h)?;
            let global = |cx: &mut EmitCx<'_>, class| {
                cx.decl
                    .families
                    .texture
                    .heap
                    .get(&class)
                    .copied()
                    .ok_or_else(|| cx.fail(format!("heap class {class:?} is not declared")))
            };
            for (k, class) in cross {
                let named = names_class(fb, index, k)?;
                arms.push((named, global(cx, class)?));
            }
            // A heap member of a form a heap reads is queried in the group's unsigned integer
            // class, which holds it: one arm with the unsigned integer members' where both are.
            if let Some((class, _)) = raw {
                let (_, member) = heap_raw_form(fb, h)?;
                let g = global(cx, class)?;
                match arms.iter_mut().find(|(_, at)| *at == g) {
                    Some(arm) => arm.0 = fb.binary(BinOp::LogicalOr, arm.0, member)?,
                    None => arms.push((member, g)),
                }
            }
            held = Some(h);
        }
    }
    let value = query_arms(cx, fb, q, (&decl, img, storage), held, &arms)?;
    live_or_zero(fb, live, value)
}

/// The value of query `q` of the element of heap handle `h` in the array of the first of `arms`
/// whose test holds, else the rest's, else of `img`.
fn query_arms(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    q: &QueryOp,
    (decl, img, storage): (&ImageDecl, SValue, bool),
    h: Option<SValue>,
    arms: &[(SValue, crate::spirv::ids::GlobalId)],
) -> BResult<SValue> {
    let (Some(h), Some((&(test, g), rest))) = (h, arms.split_first()) else {
        return query_value(cx, fb, q, (decl, img, storage));
    };
    let decl = *decl;
    let then: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let element = heap_element_index(cx, fb, h)?;
        let ptr = fb.module().global_value(g)?;
        let p = fb.access_chain_non_uniform(ptr, element)?;
        let other = fb.load(p, MemAccess::default())?;
        let v = query_value(cx, fb, q, (&decl, other, storage))?;
        Ok(ArmEnd::FallThrough(vec![v]))
    });
    let other: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let v = query_arms(cx, fb, q, (&decl, img, storage), Some(h), rest)?;
        Ok(ArmEnd::FallThrough(vec![v]))
    });
    let Merged(merged) = fb.if_(cx, test, then, Some(other))?;
    match merged[..] {
        [v] => Ok(v),
        _ => Err(cx.fail("a heap query's class branch without its value")),
    }
}

/// The value of query `q` of image `img` of `decl` (a storage image's where `storage`, its first
/// level's view).
fn query_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    q: &QueryOp,
    (decl, img, storage): (&ImageDecl, SValue, bool),
) -> BResult<SValue> {
    if decl.in_place {
        return in_place_query(cx, fb, q);
    }
    let sampled_mips =
        decl.usage == ImageUsage::Sampled && !decl.multisampled && decl.dim != Dim::Buffer;
    let n = coord_len(decl);
    // The size of the image's first level.
    let size = |fb: &mut FunctionBuilder<'_>| -> BResult<SValue> {
        match sampled_mips {
            true => {
                let zero = u32s(fb, 1, 0)?;
                fb.query(ImageQuery::SizeLod(zero), img)
            }
            false => fb.query(ImageQuery::Size, img),
        }
    };
    let level = match q.level {
        Some(l) => Some(cx.operand(fb, l)?),
        None => None,
    };
    let component = |fb: &mut FunctionBuilder<'_>, v: SValue, i: u8| -> BResult<SValue> {
        if n == 1 {
            Ok(v)
        } else {
            fb.composite_extract(v, &[u32::from(i)])
        }
    };
    let volume = q.volume && planes_volume(decl);
    Ok(match q.kind {
        // A two-plane 3D texture's depth: `⌈w / 4⌉` at its first level ([`planes_volume`]).
        QueryKind::Size { component: 2 } if volume => {
            let w = volume_width(cx, fb, q, decl, img)?;
            let three = u32s(fb, 1, 3)?;
            let w = fb.binary(BinOp::IAdd, w, three)?;
            let two = u32s(fb, 1, 2)?;
            let quarter = fb.binary(BinOp::ShiftRightLogical, w, two)?;
            match level {
                Some(l) => level_extent(fb, quarter, l)?,
                None => quarter,
            }
        }
        QueryKind::Size { component: c } => {
            let s = size(fb)?;
            let e = component(fb, s, c)?;
            match level {
                Some(l) => level_extent(fb, e, l)?,
                None => e,
            }
        }
        QueryKind::ArraySize => {
            let s = size(fb)?;
            let layers = component(fb, s, n.saturating_sub(1))?;
            if decl.cube_as_array {
                let six = u32s(fb, 1, 6)?;
                fb.binary(BinOp::UDiv, layers, six)?
            } else {
                layers
            }
        }
        QueryKind::Levels => {
            // A storage image's level count is the binder's.
            let n = match storage {
                true => {
                    let ImageRef::Slot(index) = q.image else {
                        return Err(cx.fail("a heap storage image's level count"));
                    };
                    let Some(slot) = cx.plan.families.texture.slots.get(&index).map(|s| s.slot)
                    else {
                        return Err(cx.fail(format!("texture slot {index} is not planned")));
                    };
                    cx.push_field(fb, crate::pipeline::sym::PushField::TextureLevels(slot))?
                }
                false => fb.query(ImageQuery::Levels, img)?,
            };
            match volume {
                // A two-plane 3D texture of one level: `1 - ⌊(w - 1) / 16⌋`
                // ([`planes_volume`]).
                true => {
                    let w = volume_width(cx, fb, q, decl, img)?;
                    let one = u32s(fb, 1, 1)?;
                    let below = fb.binary(BinOp::ISub, w, one)?;
                    let four = u32s(fb, 1, 4)?;
                    let sixteenths = fb.binary(BinOp::ShiftRightLogical, below, four)?;
                    let answer = fb.binary(BinOp::ISub, one, sixteenths)?;
                    let single = fb.binary(BinOp::IEqual, n, one)?;
                    fb.select(single, answer, n)?
                }
                false => n,
            }
        }
        // A 1D texture has no mipmaps in Metal: always 0.
        QueryKind::NoLevels => u32s(fb, 1, 0)?,
        QueryKind::Samples => fb.query(ImageQuery::Samples, img)?,
    })
}

/// The answer of query `q` of a slot under its in-place variant (`sampler::in_place`): its
/// texture's width or height from the slot's push field, at a level as at any texture's
/// ([`level_extent`]), and its one level.
fn in_place_query(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    q: &QueryOp,
) -> BResult<SValue> {
    let ImageRef::Slot(index) = q.image else {
        return Err(cx.fail("an in-place query of a heap element"));
    };
    let Some(slot) = cx.plan.families.texture.slots.get(&index).map(|s| s.slot) else {
        return Err(cx.fail(format!("texture slot {index} is not planned")));
    };
    match q.kind {
        QueryKind::Size { component } => {
            let e = crate::lower::sampler::in_place::extent(cx, fb, slot, component)?;
            match q.level {
                Some(l) => {
                    let l = cx.operand(fb, l)?;
                    level_extent(fb, e, l)
                }
                None => Ok(e),
            }
        }
        QueryKind::Levels => u32s(fb, 1, 1),
        _ => Err(cx.fail("an in-place query the form does not answer")),
    }
}

/// The extent Metal answers at level `l` of a texture whose first level's extent is `e`:
/// `max(1, e >> (l & 255))`, at every level, past the texture's level count too, a shift of 32 or
/// more answering 1. Measured on an Apple M3 Max under macOS 27.0.1: the width, height, and depth
/// of 2D, 3D, cube, cube array, 2D array, and depth textures, sampled and `read` and
/// `read_write`, of one level and of partial and full chains, at levels 0 to 1025 and up to
/// 2^32 - 1, and a two-plane 3D texture's (whose depth is `⌈w / 4⌉`, [`planes_volume`]).
fn level_extent(fb: &mut FunctionBuilder<'_>, e: SValue, l: SValue) -> BResult<SValue> {
    let low = u32s(fb, 1, 255)?;
    let l = fb.binary(BinOp::BitwiseAnd, l, low)?;
    let last = u32s(fb, 1, 31)?;
    let l = umin(fb, l, last)?;
    let shifted = fb.binary(BinOp::ShiftRightLogical, e, l)?;
    let one = u32s(fb, 1, 1)?;
    fb.glsl(GlslOp::UMax, &[shifted, one])
}

/// The width of the first level of `q`'s two-plane 3D texture ([`planes_volume`]), a sampled
/// slot's image `img`: its level 0's size; a storage slot's, its first level view's.
fn volume_width(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    q: &QueryOp,
    decl: &ImageDecl,
    img: SValue,
) -> BResult<SValue> {
    let size = match (decl.usage, decl.level_views) {
        (ImageUsage::Sampled, _) => {
            let zero = u32s(fb, 1, 0)?;
            fb.query(ImageQuery::SizeLod(zero), img)?
        }
        (_, Some(_)) => {
            let Loaded { img, .. } = image(cx, fb, &q.image, None)?;
            fb.query(ImageQuery::Size, img)?
        }
        (_, None) => fb.query(ImageQuery::Size, img)?,
    };
    fb.composite_extract(size, &[0])
}

/// `fence`: the thread's texture writes become visible to its later reads.
fn fence(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
    let vulkan = memory_model(cx.plan) == BarrierModel::Vulkan;
    fb.memory_barrier(
        Scope::Device,
        Semantics {
            order: MemoryOrder::AcquireRelease,
            image_memory: true,
            make_available: vulkan,
            make_visible: vulkan,
            ..Semantics::RELAXED
        },
    )
}
