//! The blends a fragment function computes in the shader, inside the pixel interlock, where no
//! fixed-function blend applies to the attachment.
//!
//! **An extended-range attachment** ([`NativeBlend`], [`native`]) holds raw codes no device
//! blends: the function reads the texel's tile value ([`super::tile_value`]), blends, and stores
//! the kernel write's encode of the result ([`super::encode`], [`super::encode_keeping`] under a
//! write mask). The blend is a blend unit's arithmetic, the device's own: `S·Fs op D·Fd` in
//! `float`, in the blend equation's order, for the color lanes the write stores and for alpha;
//! `Min` and `Max` read no factor; `SourceAlphaSaturated` is `min(sa, 1 - da)` in color and one in
//! alpha; each `1 - x` is computed in `float`. The source, the second source and the blend color
//! are taken as they are: the attachment is a padded form's, which Metal clamps none of (a blend
//! into one of Apple's extended-range forms is refused, `fragment::fetch::codes_round_the_tile`).
//! Metal's compiler instead fuses, regroups and rounds each step to `half`;
//! the stored value can differ from Metal's in the last bits of the encode, which the artifact
//! records (`PrecisionNote::ExtendedRangeBlendSource`, `ExtendedRangeFloatBlendSource`).
//!
//! **An 8-bit unorm attachment held with its channels rotated** ([`Blend`], [`Blend::unorm8`],
//! the blend `fragment::input_blend` computes over the input attachment) takes the forms Metal's
//! compiler builds. The forms, for the color lanes the write stores (one vector) and alpha (a
//! scalar), each computing `S op D`, `S = s Fs` and `D = d Fd`, each sum and product rounded once
//! to the arithmetic's type, nearest even (`h`):
//!
//! 1. `Min` and `Max`: `min(s, d)` and `max(s, d)`, the factors unread.
//! 2. A zero factor drops its term: `0`, or `h(±S)` or `h(±D)` alone.
//! 3. A factor of one: `h(±S ± D)`, one fused multiply-add.
//! 4. Two products, added, where one's factor is `1 - z` and the other's operands hold `z`
//!    (a lerp): `h(y + z · inner)`, `y` the first product's operand and `inner` the other
//!    product's other operand minus `y`.
//! 5. Two products sharing an operand `z`: `h(z · inner)`, `inner` the sum of the other two.
//! 6. Otherwise one product is rounded first and the other fused into the sum: the source's
//!    under `ReverseSubtract`, the destination's under `Subtract`, and under `Add` by the
//!    factors' ranks ([`unorm8_rank`]).
//!
//! An inner sum expands each `1 - z` into `1` and `-z` (but a blend color's, a color lane's of
//! an alpha, and, in alpha, the one-minus-blend-alpha the color factors share once it comes
//! second), merges equal terms, and adds them in a fixed order ([`inner_sum`]), each addition
//! rounded. `SourceAlphaSaturated` is `min(sa, h(1 - da))` in color and one in alpha, and `1 - x`
//! is `h(1 - x)`.
//!
//! **A module without fast math** (`air.compile.fast_math_disable`, [`BlendMath::Rounded`]):
//! Metal's compiler folds the blend into the function without fusing it: each term's product
//! rounded (a factor of one its operand, of zero no term), and the two terms' sum rounded; `min`
//! and `max` as the fast forms. Every `1 - x` is `h(1 - x)` of the value it reads
//! ([`Node::OneMinus`]).

use super::{floats, per_lane_f};
use crate::error::{Error, ExtendedRangeBlend, Feature, Unsupported};
use crate::lower::arith::fma::{float_fma_exact, half_sum};
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::narrow;
use crate::lower::arith::ops::{fsplat, splat, ty};
use crate::lower::arith::texel_round::{f32_to_half, HalfRounding};
use crate::metal::{BlendFactor, BlendOperation, ColorWriteMask, PixelFormat};
use crate::pipeline::lower_inputs::BlendEquation;
use crate::spirv::builder::{FunctionBuilder, QuantizeForm};
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, ConvOp, GlslOp, UnOp};

// ---- The device's blend arithmetic ----

/// The blend `eq` a write computes in the shader by the device's own arithmetic, where the device
/// cannot blend the attachment's format itself (an extended-range attachment's raw texels:
/// [`native`]). Every factor, operation and write mask is built, the second source's too.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeBlend {
    /// The pipeline's blend at the attachment.
    pub(crate) eq: BlendEquation,
    /// The write stores some color channel.
    color: bool,
    /// The write stores alpha.
    alpha: bool,
}

/// Whether `f` is one of the second source's factors.
fn second_factor(f: BlendFactor) -> bool {
    use BlendFactor as F;
    matches!(
        f,
        F::Source1Color | F::OneMinusSource1Color | F::Source1Alpha | F::OneMinusSource1Alpha
    )
}

/// Whether `f` reads the blend color.
fn constant_factor(f: BlendFactor) -> bool {
    use BlendFactor as F;
    matches!(
        f,
        F::BlendColor | F::OneMinusBlendColor | F::BlendAlpha | F::OneMinusBlendAlpha
    )
}

impl NativeBlend {
    /// The blend `eq` of a write storing the channels `written` of four (a channel the format
    /// lacks counts as unwritten): `Err` with the placeholder a factor or operation is
    /// (`DescriptorError::PlaceholderEnumValue`).
    pub fn of(eq: &BlendEquation, written: ColorWriteMask) -> Result<NativeBlend, Error> {
        for f in [
            eq.source_rgb,
            eq.destination_rgb,
            eq.source_alpha,
            eq.destination_alpha,
        ] {
            if f == BlendFactor::Unspecialized {
                return Err(placeholder(BlendFactor::METAL_NAME, f.as_raw()));
            }
        }
        for op in [eq.rgb_operation, eq.alpha_operation] {
            if op == BlendOperation::Unspecialized {
                return Err(placeholder(BlendOperation::METAL_NAME, op.as_raw()));
            }
        }
        let color = [
            ColorWriteMask::RED,
            ColorWriteMask::GREEN,
            ColorWriteMask::BLUE,
        ]
        .into_iter()
        .any(|c| written.contains(c));
        Ok(NativeBlend {
            eq: *eq,
            color,
            alpha: written.contains(ColorWriteMask::ALPHA),
        })
    }

    /// The factors the stored classes read: none under `Min` or `Max`.
    fn factors(&self) -> Vec<BlendFactor> {
        let minmax = |op: BlendOperation| matches!(op, BlendOperation::Min | BlendOperation::Max);
        let mut fs = Vec::with_capacity(4);
        if self.color && !minmax(self.eq.rgb_operation) {
            fs.extend([self.eq.source_rgb, self.eq.destination_rgb]);
        }
        if self.alpha && !minmax(self.eq.alpha_operation) {
            fs.extend([self.eq.source_alpha, self.eq.destination_alpha]);
        }
        fs
    }

    /// Whether the blend reads the blend color (`PushField::HalfBlendColor`).
    pub fn reads_constant(&self) -> bool {
        self.factors().into_iter().any(constant_factor)
    }

    /// Whether the blend reads the second source (the function's index-1 output).
    pub fn reads_second(&self) -> bool {
        self.factors().into_iter().any(second_factor)
    }
}

/// The values a native blend's factors read, `vec4`s of `f32`.
struct Operands {
    s: Value,
    d: Value,
    s1: Option<Value>,
    c: Option<Value>,
}

/// `x`'s alpha in every lane.
fn alpha_splat(fb: &mut FunctionBuilder<'_>, x: Value) -> BResult<Value> {
    fb.vector_shuffle(x, x, &[3, 3, 3, 3])
}

/// Factor `f` of a lane class (alpha's where `alpha`) over `o`, every lane: `None` for a factor
/// of one, whose product is its operand.
fn native_factor(
    fb: &mut FunctionBuilder<'_>,
    f: BlendFactor,
    alpha: bool,
    o: &Operands,
) -> BResult<Option<Value>> {
    use BlendFactor as F;
    let v = floats();
    let missing = || BuildError::Type(TypeRule::FunctionSignature);
    let one_minus = |fb: &mut FunctionBuilder<'_>, x: Value| -> BResult<Value> {
        let one = fsplat(fb, v, 1.0)?;
        fb.binary(BinOp::FSub, one, x)
    };
    let s1 = || o.s1.ok_or_else(missing);
    let c = || o.c.ok_or_else(missing);
    Ok(Some(match f {
        F::Zero => fsplat(fb, v, 0.0)?,
        F::One => return Ok(None),
        F::SourceAlphaSaturated if alpha => return Ok(None),
        F::SourceColor => o.s,
        F::OneMinusSourceColor => one_minus(fb, o.s)?,
        F::SourceAlpha => alpha_splat(fb, o.s)?,
        F::OneMinusSourceAlpha => {
            let a = alpha_splat(fb, o.s)?;
            one_minus(fb, a)?
        }
        F::DestinationColor => o.d,
        F::OneMinusDestinationColor => one_minus(fb, o.d)?,
        F::DestinationAlpha => alpha_splat(fb, o.d)?,
        F::OneMinusDestinationAlpha => {
            let a = alpha_splat(fb, o.d)?;
            one_minus(fb, a)?
        }
        F::SourceAlphaSaturated => {
            let sa = alpha_splat(fb, o.s)?;
            let da = alpha_splat(fb, o.d)?;
            let m = one_minus(fb, da)?;
            fb.glsl(GlslOp::FMin, &[sa, m])?
        }
        F::BlendColor => c()?,
        F::OneMinusBlendColor => one_minus(fb, c()?)?,
        F::BlendAlpha => alpha_splat(fb, c()?)?,
        F::OneMinusBlendAlpha => {
            let a = alpha_splat(fb, c()?)?;
            one_minus(fb, a)?
        }
        F::Source1Color => s1()?,
        F::OneMinusSource1Color => one_minus(fb, s1()?)?,
        F::Source1Alpha => alpha_splat(fb, s1()?)?,
        F::OneMinusSource1Alpha => {
            let a = alpha_splat(fb, s1()?)?;
            one_minus(fb, a)?
        }
        F::Unspecialized => return Err(missing()),
    }))
}

/// One lane class of a native blend (alpha's where `alpha`, read from lane 3) over `o`, every
/// lane: `S·Fs op D·Fd` in the equation's order, each product and the sum the device's `OpFMul`
/// and `OpFAdd` or `OpFSub` (a factor of one its operand); `Min` and `Max` the device's `FMin`
/// and `FMax` of the source and the destination, the factors unread.
fn native_class(
    fb: &mut FunctionBuilder<'_>,
    (fs, fd, op): (BlendFactor, BlendFactor, BlendOperation),
    alpha: bool,
    o: &Operands,
) -> BResult<Value> {
    match op {
        BlendOperation::Min => return fb.glsl(GlslOp::FMin, &[o.s, o.d]),
        BlendOperation::Max => return fb.glsl(GlslOp::FMax, &[o.s, o.d]),
        _ => {}
    }
    let source = match native_factor(fb, fs, alpha, o)? {
        Some(f) => fb.binary(BinOp::FMul, o.s, f)?,
        None => o.s,
    };
    let destination = match native_factor(fb, fd, alpha, o)? {
        Some(f) => fb.binary(BinOp::FMul, o.d, f)?,
        None => o.d,
    };
    match op {
        BlendOperation::Subtract => fb.binary(BinOp::FSub, source, destination),
        BlendOperation::ReverseSubtract => fb.binary(BinOp::FSub, destination, source),
        _ => fb.binary(BinOp::FAdd, source, destination),
    }
}

/// The blend `b` ([`NativeBlend`]) of output `s` (a `vec4` of `f32`: the function's `half4` or
/// `float4` output as it computed it) over `d` (a `vec4` of `f32`, the destination as the write
/// reads it) by the device's arithmetic: a `vec4` of `f32` holding each blended lane, which the
/// caller converts to the attachment. The source, the second source `s1` (the function's index-1
/// output, where `b` reads it) and the blend color `color` (the encoder's blend color, a `uvec4`
/// of its `f32`s' bits, `PushField::HalfBlendColor`, where `b` reads it) are taken as they are, as
/// Metal takes a padded form's. Each class computes `S·Fs op D·Fd` in `f32`, in the equation's
/// order, each
/// factor (`1 - x`, `SourceAlphaSaturated`'s `min(sa, 1 - da)`), product and sum one device
/// instruction, nothing rounded to `half` between them, the device free to contract a product into
/// its sum as it contracts the module's own arithmetic. A lane the write does not store holds
/// `d`'s.
pub fn native(
    fb: &mut FunctionBuilder<'_>,
    b: &NativeBlend,
    (s, s1, d): (Value, Option<Value>, Value),
    color: Option<Value>,
) -> BResult<Value> {
    let f = floats();
    let s1 = s1.filter(|_| b.reads_second());
    let c = match color {
        Some(words) if b.reads_constant() => {
            let ft = ty(fb, f)?;
            Some(fb.convert(ConvOp::Bitcast, words, ft)?)
        }
        _ => None,
    };
    let o = Operands { s, d, s1, c };
    let eq = &b.eq;
    let rgb = match b.color {
        true => native_class(
            fb,
            (eq.source_rgb, eq.destination_rgb, eq.rgb_operation),
            false,
            &o,
        )?,
        false => d,
    };
    let alpha = match b.alpha {
        true => native_class(
            fb,
            (eq.source_alpha, eq.destination_alpha, eq.alpha_operation),
            true,
            &o,
        )?,
        false => d,
    };
    fb.vector_shuffle(rgb, alpha, &[0, 1, 2, 7])
}

/// A value a blend lane reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Atom {
    /// The source's lane: a color lane's own channel, alpha's alpha.
    S,
    /// The source's alpha.
    Sa,
    /// The destination's lane (the tile value's).
    D,
    /// The destination's alpha.
    Da,
    /// The blend color's lane.
    C,
    /// The blend color's alpha.
    Ca,
    /// The second source's lane (the function's index-1 output), rounded and clamped as the
    /// source's: a color lane's own channel, alpha's alpha ([`BlendMath::Rounded`] alone).
    S1,
    /// The second source's alpha.
    S1a,
}

impl Atom {
    /// An alpha a color lane reads as a splat of one value.
    fn splat(self) -> bool {
        matches!(self, Atom::Sa | Atom::Da | Atom::Ca | Atom::S1a)
    }

    /// An inner sum's order among its atoms under `scheme`.
    fn order(self, scheme: Order) -> u8 {
        match (self, scheme) {
            (Atom::C | Atom::Ca, _) => 9,
            (Atom::S1 | Atom::S1a, _) => 1,
            (Atom::S | Atom::Sa, Order::Destination) => 2,
            (Atom::D | Atom::Da, Order::Destination) => 0,
            (Atom::Da, Order::Scalar) => 0,
            (Atom::D, Order::Scalar) => 1,
            (Atom::Sa, Order::Scalar) => 2,
            (Atom::S, Order::Scalar) => 3,
        }
    }
}

/// An inner sum's order among its atoms ([`Atom::order`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    /// The destination's before the source's: an 8-bit unorm attachment's blend.
    Destination,
    /// The destination's before the source's, and each alpha before its lane: an 8-bit unorm
    /// attachment's color channel stored alone.
    Scalar,
}

/// A factor or operand of a blend lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Node {
    Zero,
    One,
    Atom(Atom),
    /// `h(1 - x)`.
    OneMinus(Atom),
    /// `min(sa, h(1 - da))`, a color lane's `SourceAlphaSaturated`.
    Saturate,
}

/// An operand and its factor: their product.
pub(crate) type Term = (Atom, Node);

/// A summand of an inner sum: the constant one (`None`) or a node, with its coefficient.
pub(crate) type Summand = (i8, Option<Node>);

/// The arithmetic of one lane class, in the module docs' forms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Arith {
    /// `min(s, d)` or `max(s, d)`, `s` the `source` atom.
    MinMax {
        max: bool,
        source: Atom,
    },
    Zero,
    /// `h(sign · term)`.
    Term {
        sign: i8,
        term: Term,
    },
    /// `h(sign · product + sign · addend)`, the addend's factor one: one fused multiply-add.
    Fused {
        product: (i8, Term),
        addend: (i8, Term),
    },
    /// `h(y + z · inner)`.
    Lerp {
        y: Node,
        z: Node,
        inner: Vec<Summand>,
    },
    /// `h(z · inner)`.
    Common {
        z: Node,
        inner: Vec<Summand>,
    },
    /// `h(sign · h(rounded) + sign · fused)`, the fused product added exactly.
    Contracted {
        rounded: (i8, Term),
        fused: (i8, Term),
    },
    /// `h(h(sign · source) + h(sign · destination))`, a term of factor one its operand: a module
    /// without fast math's ([`BlendMath::Rounded`]).
    Rounded {
        source: (i8, Term),
        destination: (i8, Term),
    },
}

/// Metal's blend of one attachment's write, as the shader computes it ([`blend_unorm8`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Blend {
    /// The color lanes' arithmetic, where the write stores two or three of them.
    pub(crate) color: Option<Arith>,
    /// Alpha's, where the write stores it.
    pub(crate) alpha: Option<Arith>,
    /// The function's math mode: every one minus is `h(1 - x)` of the `half` it reads under
    /// [`BlendMath::Rounded`], the blend color's too.
    pub(crate) math: BlendMath,
}

/// How Metal's compiler computes a blend it folds into the function, by the function's math mode
/// (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMath {
    /// Fast math, or a module that names no mode: the fast forms, a product fused into a sum by
    /// the factors' structure.
    Fast,
    /// A module without fast math (`air.compile.fast_math_disable`): each term's product rounded
    /// to `half`, then the terms' sum ([`Arith::Rounded`]).
    Rounded,
}

impl BlendMath {
    /// The blend math of a module compiled with `fast_math` (`CompileOptions::fast_math`).
    pub fn of(fast_math: Option<bool>) -> BlendMath {
        match fast_math {
            Some(false) => BlendMath::Rounded,
            Some(true) | None => BlendMath::Fast,
        }
    }
}

fn refused(gap: ExtendedRangeBlend) -> Error {
    Unsupported::not_implemented(Feature::ExtendedRangeBlend(gap)).into()
}

fn placeholder(ty: &'static str, value: u32) -> Error {
    Error::InvalidDescriptor(crate::error::DescriptorError::PlaceholderEnumValue { ty, value })
}

/// Factor `f` of a color lane, or of alpha where `alpha`, under `math`: the second source's
/// factors are refused under fast math.
fn factor(f: BlendFactor, alpha: bool, math: BlendMath) -> Result<Node, Error> {
    use BlendFactor as F;
    let own = |color: Atom, a: Atom| if alpha { a } else { color };
    Ok(match f {
        F::Zero => Node::Zero,
        F::One => Node::One,
        F::SourceColor => Node::Atom(own(Atom::S, Atom::Sa)),
        F::OneMinusSourceColor => Node::OneMinus(own(Atom::S, Atom::Sa)),
        F::SourceAlpha => Node::Atom(Atom::Sa),
        F::OneMinusSourceAlpha => Node::OneMinus(Atom::Sa),
        F::DestinationColor => Node::Atom(own(Atom::D, Atom::Da)),
        F::OneMinusDestinationColor => Node::OneMinus(own(Atom::D, Atom::Da)),
        F::DestinationAlpha => Node::Atom(Atom::Da),
        F::OneMinusDestinationAlpha => Node::OneMinus(Atom::Da),
        F::SourceAlphaSaturated if alpha => Node::One,
        F::SourceAlphaSaturated => Node::Saturate,
        F::BlendColor => Node::Atom(own(Atom::C, Atom::Ca)),
        F::OneMinusBlendColor => Node::OneMinus(own(Atom::C, Atom::Ca)),
        F::BlendAlpha => Node::Atom(Atom::Ca),
        F::OneMinusBlendAlpha => Node::OneMinus(Atom::Ca),
        F::Source1Color | F::OneMinusSource1Color | F::Source1Alpha | F::OneMinusSource1Alpha
            if math == BlendMath::Fast =>
        {
            return Err(refused(ExtendedRangeBlend::DualSource))
        }
        F::Source1Color => Node::Atom(own(Atom::S1, Atom::S1a)),
        F::OneMinusSource1Color => Node::OneMinus(own(Atom::S1, Atom::S1a)),
        F::Source1Alpha => Node::Atom(Atom::S1a),
        F::OneMinusSource1Alpha => Node::OneMinus(Atom::S1a),
        F::Unspecialized => return Err(placeholder(BlendFactor::METAL_NAME, f.as_raw())),
    })
}

/// An 8-bit unorm attachment's factor rank, of a color factor or alpha's where `alpha`, in the
/// arithmetic `arith`: of an added pair, the destination's product is rounded first where the
/// source factor's rank is at least the destination factor's, else the source's.
///
/// In a color lane the one minus the destination alpha ranks as the one minus the source alpha
/// does. Plain Metal on the M3 Max under macOS 27.0.1 blends `DestinationAlpha` over
/// `OneMinusDestinationAlpha` with the source's product rounded first, into `RGBA8Unorm` from a
/// `float4` output under fast math: `f(fma(d, f(1 - da), f(s · da)))` gives Metal's code in every
/// color channel of the 8,192 texels of the `argb8-targets-destination-alpha` recording, and the
/// destination's product rounded first misses 10 of them. The same rank in `half` arithmetic is
/// not measured.
fn unorm8_rank(f: BlendFactor, alpha: bool, arith: BlendArith) -> u8 {
    use BlendFactor as F;
    let float = arith == BlendArith::Float;
    match f {
        F::SourceColor
        | F::SourceAlpha
        | F::DestinationColor
        | F::OneMinusDestinationColor
        | F::DestinationAlpha => 0,
        F::OneMinusDestinationAlpha if alpha => 0,
        F::OneMinusSourceColor | F::OneMinusSourceAlpha if alpha && !float => 0,
        F::OneMinusSourceColor
        | F::OneMinusSourceAlpha
        | F::OneMinusDestinationAlpha
        | F::SourceAlphaSaturated => 1,
        F::BlendColor | F::BlendAlpha if float => 1,
        F::BlendColor | F::BlendAlpha => 2,
        F::OneMinusBlendColor | F::OneMinusBlendAlpha => 3,
        _ => 9,
    }
}

/// The factor `n` of a format without alpha (`BGR10_XR`, `R16Float`, `RG16Float`): its destination
/// alpha is one.
fn opaque(n: Node) -> Node {
    match n {
        Node::Atom(Atom::Da) => Node::One,
        Node::OneMinus(Atom::Da) => Node::Zero,
        n => n,
    }
}

/// The one-minus nodes of the color factors that alpha's sums and lerps take apart ([`plan`]):
/// `values` (one value the color lanes compute too: a lerp's one minus is none of them where
/// alpha's model says so, and a sum keeps one whole where added second).
#[derive(Clone, Debug, Default)]
struct Shared {
    values: Vec<Node>,
}

/// Whether an inner sum keeps the one-minus node `n`, added with `sign` (from the destination's
/// product where `destination`), whole beyond the nodes that always stay whole ([`inner_sum`]): a
/// shared value added from the destination's product.
fn kept_whole((n, sign, _, destination): (Node, i8, usize, bool), shared: &Shared) -> bool {
    shared.values.contains(&n) && sign > 0 && destination
}

/// The summands of `pairs` (each a node, its sign, and whether it comes from the destination's
/// product, in the IR's order) as Metal's compiler adds them: each `1 - z` expanded into `1` and
/// `-z` but where it stays one value (a blend color's; an alpha's in a color lane of a vector
/// class, `scalar` false; where `keep` says so, [`kept_whole`]), equal terms merged, and the terms
/// ordered: the constant last, but between the two others where they are not both subtracted,
/// after the one of the lower [`Atom::order`] under `order`.
fn inner_sum(
    pairs: [(Node, i8, bool); 2],
    (scalar, order): (bool, Order),
    keep: &dyn Fn((Node, i8, usize, bool)) -> bool,
) -> Vec<Summand> {
    let mut terms: Vec<Summand> = Vec::with_capacity(4);
    for (i, (n, sign, destination)) in pairs.into_iter().enumerate() {
        match n {
            Node::OneMinus(z) => {
                let leaf = matches!(z, Atom::C | Atom::Ca)
                    || (!scalar && z.splat())
                    || keep((n, sign, i, destination));
                if leaf {
                    terms.push((sign, Some(n)));
                } else {
                    terms.push((sign, None));
                    terms.push((-sign, Some(Node::Atom(z))));
                }
            }
            Node::One => terms.push((sign, None)),
            n => terms.push((sign, Some(n))),
        }
    }
    let mut merged: Vec<Summand> = Vec::with_capacity(terms.len());
    for (c, k) in terms {
        match merged.iter_mut().find(|m| m.1 == k) {
            Some(m) => m.0 += c,
            None => merged.push((c, k)),
        }
    }
    merged.retain(|m| m.0 != 0);
    let constant = merged.iter().copied().find(|m| m.1.is_none());
    let rest: Vec<Summand> = merged.iter().copied().filter(|m| m.1.is_some()).collect();
    match (constant, rest.as_slice()) {
        (Some(one), &[a, b]) if a.0 > 0 || b.0 > 0 => {
            let rank = |m: Summand| match m.1 {
                Some(Node::Atom(x)) => x.order(order),
                _ => 9,
            };
            match rank(b) < rank(a) {
                true => vec![b, one, a],
                false => vec![a, one, b],
            }
        }
        _ => rest.into_iter().chain(constant).collect(),
    }
}

/// The arithmetic of a lane class (alpha where `alpha`) of factors `fs` and `fd` under `op`,
/// `shared` the one-minus nodes of the color factors that alpha's sums and lerps take apart,
/// `opaque_` for a format without alpha, in the arithmetic `arith`. `scalar` marks a color class of one channel (an 8-bit unorm attachment's write mask
/// of one color channel), which Metal plans as a scalar: an alpha read there is no splat, but a
/// one minus an alpha that its destination factor is or that alpha's factors compute (`shared`)
/// stays whole in its sums and takes no lerp.
fn plan(
    (fs, fd, op): (BlendFactor, BlendFactor, BlendOperation),
    (alpha, scalar): (bool, bool),
    shared: &Shared,
    opaque_: bool,
    arith: BlendArith,
) -> Result<Arith, Error> {
    let order = match scalar {
        false => Order::Destination,
        true => Order::Scalar,
    };
    let (s, da) = match alpha {
        true => (Atom::Sa, Atom::Da),
        false => (Atom::S, Atom::D),
    };
    let (ps, pd): (i8, i8) = match op {
        BlendOperation::Add => (1, 1),
        BlendOperation::Subtract => (1, -1),
        BlendOperation::ReverseSubtract => (-1, 1),
        BlendOperation::Min => {
            return Ok(Arith::MinMax {
                max: false,
                source: s,
            })
        }
        BlendOperation::Max => {
            return Ok(Arith::MinMax {
                max: true,
                source: s,
            })
        }
        BlendOperation::Unspecialized => {
            return Err(placeholder(BlendOperation::METAL_NAME, op.as_raw()))
        }
    };
    let fast = BlendMath::Fast;
    let (mut nfs, mut nfd) = (factor(fs, alpha, fast)?, factor(fd, alpha, fast)?);
    if opaque_ {
        (nfs, nfd) = (opaque(nfs), opaque(nfd));
    }
    let st: Term = (s, nfs);
    let dt: Term = (da, nfd);
    match (nfs == Node::Zero, nfd == Node::Zero) {
        (true, true) => return Ok(Arith::Zero),
        (true, false) => return Ok(Arith::Term { sign: pd, term: dt }),
        (false, true) => return Ok(Arith::Term { sign: ps, term: st }),
        (false, false) => {}
    }
    if nfs == Node::One || nfd == Node::One {
        let (product, addend) = match nfs == Node::One && nfd != Node::One {
            true => ((pd, dt), (ps, st)),
            false => ((ps, st), (pd, dt)),
        };
        return Ok(Arith::Fused { product, addend });
    }
    // Each product's operand and factor.
    let sops = (Node::Atom(s), nfs);
    let dops = (Node::Atom(da), nfd);
    let both_orders = |p: (Node, Node)| [(p.0, p.1), (p.1, p.0)];
    // A color channel stored alone keeps whole the one minus an alpha its destination factor is
    // and those alpha's factors compute.
    let mut alone = shared.values.clone();
    if matches!(nfd, Node::OneMinus(z) if z.splat()) {
        alone.push(nfd);
    }
    let alone = &alone;
    let keep_in = || {
        move |k: (Node, i8, usize, bool)| match scalar {
            true => alone.contains(&k.0),
            false => kept_whole(k, shared),
        }
    };
    // The lerp takes the destination's product for `y`'s first.
    let sides = [(false, (dops, sops)), (true, (sops, dops))];
    if op == BlendOperation::Add {
        for (source_side, (yt, zo)) in sides {
            for (y, f) in both_orders(yt) {
                let Node::OneMinus(za) = f else {
                    continue;
                };
                // No lerp of a one minus the color factors computed, the destination alpha's
                // among them.
                if shared.values.contains(&f) && (za == Atom::Ca || za == Atom::Da) {
                    continue;
                }
                let both = [sops.0, sops.1].contains(&f) && [dops.0, dops.1].contains(&f);
                if !alpha && !scalar && (za.splat() || y == Node::Atom(za) || both) {
                    continue;
                }
                if scalar && (alone.contains(&f) || y == Node::Atom(za) || both) {
                    continue;
                }
                // Alpha takes no lerp of the destination alpha beside itself or held by both
                // products: Metal's alpha there is the common factor's form.
                if alpha && za == Atom::Da && (y == Node::Atom(za) || both) {
                    continue;
                }
                let z = Node::Atom(za);
                if let Some((_, x)) = both_orders(zo).into_iter().find(|&(zj, _)| zj == z) {
                    let keep = keep_in();
                    // `x` comes from the other product than `y`.
                    let pairs = [(x, 1, source_side), (y, -1, !source_side)];
                    let inner = inner_sum(pairs, (alpha || scalar, order), &keep);
                    return Ok(Arith::Lerp { y, z, inner });
                }
            }
        }
    }
    let keep = keep_in();
    for (z, x) in both_orders(sops) {
        for (zd, y) in both_orders(dops) {
            if z == zd && !matches!(z, Node::Zero | Node::One) {
                let pair = match op {
                    BlendOperation::ReverseSubtract => [(y, pd, true), (x, ps, false)],
                    _ => [(x, ps, false), (y, pd, true)],
                };
                let inner = inner_sum(pair, (alpha || scalar, order), &keep);
                return Ok(Arith::Common { z, inner });
            }
        }
    }
    let source_first = Arith::Contracted {
        rounded: (ps, st),
        fused: (pd, dt),
    };
    let destination_first = Arith::Contracted {
        rounded: (pd, dt),
        fused: (ps, st),
    };
    Ok(match op {
        BlendOperation::ReverseSubtract => source_first,
        BlendOperation::Subtract => destination_first,
        _ => {
            let rank = |f| unorm8_rank(f, alpha, arith);
            match rank(fs) >= rank(fd) {
                true => destination_first,
                false => source_first,
            }
        }
    })
}

/// The arithmetic of a lane class (alpha where `alpha`) of factors `fs` and `fd` under `op` in a
/// module without fast math ([`BlendMath::Rounded`]), `opaque_` for a format without alpha:
/// `min` or `max`; no term, one, or both terms' rounded products added and rounded
/// ([`Arith::Rounded`]). A zero factor leaves its term out.
fn plan_rounded(
    (fs, fd, op): (BlendFactor, BlendFactor, BlendOperation),
    alpha: bool,
    opaque_: bool,
) -> Result<Arith, Error> {
    let (s, d) = match alpha {
        true => (Atom::Sa, Atom::Da),
        false => (Atom::S, Atom::D),
    };
    let (ps, pd): (i8, i8) = match op {
        BlendOperation::Add => (1, 1),
        BlendOperation::Subtract => (1, -1),
        BlendOperation::ReverseSubtract => (-1, 1),
        BlendOperation::Min | BlendOperation::Max => {
            return Ok(Arith::MinMax {
                max: op == BlendOperation::Max,
                source: s,
            })
        }
        BlendOperation::Unspecialized => {
            return Err(placeholder(BlendOperation::METAL_NAME, op.as_raw()))
        }
    };
    let rounded = BlendMath::Rounded;
    let (mut nfs, mut nfd) = (factor(fs, alpha, rounded)?, factor(fd, alpha, rounded)?);
    if opaque_ {
        (nfs, nfd) = (opaque(nfs), opaque(nfd));
    }
    let (st, dt): (Term, Term) = ((s, nfs), (d, nfd));
    Ok(match (nfs == Node::Zero, nfd == Node::Zero) {
        (true, true) => Arith::Zero,
        (true, false) => Arith::Term { sign: pd, term: dt },
        (false, true) => Arith::Term { sign: ps, term: st },
        (false, false) => Arith::Rounded {
            source: (ps, st),
            destination: (pd, dt),
        },
    })
}

/// Metal's blend `eq` of a write storing the channels `written` of four into an 8-bit unorm
/// format with an alpha channel where `has_alpha`, in the arithmetic `arith`, in a function of
/// math `math`: a write mask of one color channel Metal plans as a scalar.
fn planned_for(
    eq: &BlendEquation,
    (has_alpha, arith): (bool, BlendArith),
    written: ColorWriteMask,
    math: BlendMath,
) -> Result<Blend, Error> {
    let colors = [
        ColorWriteMask::RED,
        ColorWriteMask::GREEN,
        ColorWriteMask::BLUE,
    ]
    .into_iter()
    .filter(|&c| written.contains(c))
    .count();
    let alpha_written = has_alpha && written.contains(ColorWriteMask::ALPHA);
    let opaque_ = !has_alpha;
    let color = match colors {
        0 => None,
        _ => {
            let f = (eq.source_rgb, eq.destination_rgb, eq.rgb_operation);
            let scalar = colors == 1;
            // A color channel stored alone shares the one minus an alpha that alpha's factors
            // compute, stored or not.
            let mut computed = Shared::default();
            let alpha_factors = !matches!(
                eq.alpha_operation,
                BlendOperation::Min | BlendOperation::Max
            );
            if scalar && has_alpha && alpha_factors {
                use BlendFactor as F;
                let uses = |f: [F; 2]| {
                    f.iter()
                        .any(|f| [eq.source_alpha, eq.destination_alpha].contains(f))
                };
                for (f, z) in [
                    ([F::OneMinusSourceAlpha, F::OneMinusSourceColor], Atom::Sa),
                    (
                        [F::OneMinusDestinationAlpha, F::OneMinusDestinationColor],
                        Atom::Da,
                    ),
                    ([F::OneMinusBlendAlpha, F::OneMinusBlendColor], Atom::Ca),
                ] {
                    if uses(f) {
                        computed.values.push(Node::OneMinus(z));
                    }
                }
            }
            Some(match math {
                BlendMath::Fast => plan(f, (false, scalar), &computed, opaque_, arith)?,
                BlendMath::Rounded => plan_rounded(f, false, opaque_)?,
            })
        }
    };
    let alpha = match alpha_written {
        false => None,
        true => {
            let uses = |f: BlendFactor| [eq.source_rgb, eq.destination_rgb].contains(&f);
            // Alpha shares what color lanes it stores compute: one minus the blend alpha, the
            // source alpha, and the destination alpha (which `SourceAlphaSaturated` computes
            // too); with no color channel stored, the one minus an alpha the color channels'
            // destination factor is, which Metal computes still. A minimum or maximum computes no
            // factor.
            use BlendFactor as F;
            let mut shared = Shared::default();
            let factors = !matches!(eq.rgb_operation, BlendOperation::Min | BlendOperation::Max);
            if colors == 0 && factors {
                let z = match eq.destination_rgb {
                    F::OneMinusBlendAlpha => Some(Atom::Ca),
                    F::OneMinusSourceAlpha => Some(Atom::Sa),
                    F::OneMinusDestinationAlpha => Some(Atom::Da),
                    _ => None,
                };
                shared.values = z.map(Node::OneMinus).into_iter().collect();
            }
            if colors > 0 && factors {
                let minus = [
                    (uses(F::OneMinusBlendAlpha), Atom::Ca),
                    (uses(F::OneMinusSourceAlpha), Atom::Sa),
                    (
                        uses(F::OneMinusDestinationAlpha) || uses(F::SourceAlphaSaturated),
                        Atom::Da,
                    ),
                ];
                shared.values = minus
                    .into_iter()
                    .filter(|&(u, _)| u)
                    .map(|(_, z)| Node::OneMinus(z))
                    .collect();
            }
            let f = (eq.source_alpha, eq.destination_alpha, eq.alpha_operation);
            Some(match math {
                BlendMath::Rounded => plan_rounded(f, true, opaque_)?,
                BlendMath::Fast => plan(f, (true, false), &shared, opaque_, arith)?,
            })
        }
    };
    Ok(Blend { color, alpha, math })
}

impl Blend {
    /// Metal's blend `eq` of a four-lane output of a module of math `math`, blended in `arith`
    /// (the output's type), into an 8-bit unorm attachment of format `pf` (`R8Unorm`, `RG8Unorm`,
    /// `RGBA8Unorm`, `BGRA8Unorm`) storing the channels `written` of four (a channel the format
    /// lacks counts as written), computed by [`blend_unorm8`] over the output clamped to `[0, 1]`
    /// and the destination's code decoded in that type, the result stored by the device's
    /// conversion.
    /// Without fast math the rounded forms (the module docs). Under fast math the fast forms, as
    /// Metal's compiler builds an 8-bit unorm attachment's blend (plain Metal on the M3 Max under
    /// macOS 27.0.1):
    ///
    /// - an inner sum adds the destination's atoms before the source's ([`Atom::order`]);
    /// - a lerp takes `y` from the destination's product first, and alpha takes no lerp of the
    ///   destination alpha beside itself or held by both products;
    /// - of an added pair the product rounded first follows [`unorm8_rank`];
    /// - alpha shares the one minus the blend alpha, the source alpha and the destination alpha
    ///   (`SourceAlphaSaturated`'s too) that stored color lanes compute, and with no color
    ///   channel stored the one minus an alpha the color channels' destination factor is, where the
    ///   color channels' operation is no minimum or maximum, which computes no factor: it takes no
    ///   lerp of a shared blend or destination alpha, and its sums keep a shared one minus whole
    ///   where they add it from the destination's product;
    /// - a write mask of one color channel plans that channel as a scalar: its
    ///   alpha reads are no splats, but it keeps whole, and takes no lerp of, a one minus an alpha
    ///   that its destination factor is or that alpha's factors compute, alpha stored or not; its
    ///   sums add the destination alpha before the destination's lane, then the source alpha
    ///   before the source's ([`Order::Scalar`]).
    ///
    /// A factor of the second source is not built (`Feature::FramebufferFetch` `NotImplemented`).
    /// Nor, under fast math, is a
    /// format without alpha's (`R8Unorm`, `RG8Unorm`) lerp, common factor or contracted pair: Metal
    /// sums those in forms of their own, not measured.
    pub fn unorm8(
        eq: &BlendEquation,
        pf: PixelFormat,
        written: ColorWriteMask,
        (math, arith): (BlendMath, BlendArith),
    ) -> Result<Blend, Error> {
        use BlendFactor as F;
        let has_alpha = match pf {
            PixelFormat::R8Unorm | PixelFormat::RG8Unorm => false,
            // `ARGB8Unorm` blends as `RGBA8Unorm`, its channels rotated (`fetch::unorm8_format`).
            PixelFormat::RGBA8Unorm | PixelFormat::BGRA8Unorm | PixelFormat::ARGB8Unorm => true,
            _ => return Err(Unsupported::not_implemented(Feature::FramebufferFetch).into()),
        };
        let factors = [
            eq.source_rgb,
            eq.destination_rgb,
            eq.source_alpha,
            eq.destination_alpha,
        ];
        let second = |f: &BlendFactor| {
            matches!(
                f,
                F::Source1Color
                    | F::OneMinusSource1Color
                    | F::Source1Alpha
                    | F::OneMinusSource1Alpha
            )
        };
        if factors.iter().any(second) {
            return Err(Unsupported::not_implemented(Feature::FramebufferFetch).into());
        }
        let b = planned_for(eq, (has_alpha, arith), written, math)?;
        // A format without alpha sums its two products in forms of its own: of
        // its fast-math forms only a single or fused product, zero, and a minimum or maximum are
        // measured.
        let measured = |a: &Arith| {
            matches!(
                a,
                Arith::MinMax { .. } | Arith::Zero | Arith::Term { .. } | Arith::Fused { .. }
            )
        };
        if !has_alpha && math == BlendMath::Fast && !b.color.iter().all(measured) {
            return Err(Unsupported::not_implemented(Feature::FramebufferFetch).into());
        }
        Ok(b)
    }

    /// Whether the blend reads the blend color (`PushField::HalfBlendColor`).
    pub fn reads_constant(&self) -> bool {
        [&self.color, &self.alpha]
            .into_iter()
            .flatten()
            .any(|a| nodes(a).into_iter().any(reads_constant))
    }

    /// Whether the blend reads the second source (the function's index-1 output).
    pub fn reads_second(&self) -> bool {
        let second = |n: Node| {
            matches!(
                n,
                Node::Atom(Atom::S1 | Atom::S1a) | Node::OneMinus(Atom::S1 | Atom::S1a)
            )
        };
        [&self.color, &self.alpha]
            .into_iter()
            .flatten()
            .any(|a| nodes(a).into_iter().any(second))
    }
}

/// Whether `n` reads the blend color.
fn reads_constant(n: Node) -> bool {
    matches!(
        n,
        Node::Atom(Atom::C | Atom::Ca) | Node::OneMinus(Atom::C | Atom::Ca)
    )
}

/// Every node `a` reads.
fn nodes(a: &Arith) -> Vec<Node> {
    let term = |t: &Term| [Node::Atom(t.0), t.1];
    let summands = |s: &[Summand]| s.iter().filter_map(|m| m.1).collect::<Vec<_>>();
    match a {
        Arith::MinMax { .. } | Arith::Zero => Vec::new(),
        Arith::Term { term: t, .. } => term(t).to_vec(),
        Arith::Fused { product, addend } => [term(&product.1), term(&addend.1)].concat(),
        Arith::Contracted { rounded, fused } => [term(&rounded.1), term(&fused.1)].concat(),
        Arith::Rounded {
            source,
            destination,
        } => [term(&source.1), term(&destination.1)].concat(),
        Arith::Lerp { y, z, inner } => [vec![*y, *z], summands(inner)].concat(),
        Arith::Common { z, inner, .. } => [vec![*z], summands(inner)].concat(),
    }
}

// ---- The shader's arithmetic ----

/// How a lane class's values are held: the color lanes in a `vec4` (lane 3 unread), alpha a
/// scalar.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lanes {
    Vector,
    Scalar,
}

impl Lanes {
    fn floats(self) -> Form {
        Form {
            held: Held::Float(FloatFmt::F32),
            lanes: match self {
                Lanes::Vector => 4,
                Lanes::Scalar => 1,
            },
        }
    }
}

/// The `vec4`s a blend reads: the source, the destination, and the blend color's reads where the
/// blend reads them; the function's math; and the second source where the blend reads it
/// ([`Atom::S1`]).
struct Inputs {
    s: Value,
    d: Value,
    /// The blend color as alpha's `BlendAlpha` reads it, its one minus as `OneMinusBlendColor`
    /// reads it, and the color as the color lanes' `BlendColor` reads it.
    constant: Option<(Value, Value, Value)>,
    math: BlendMath,
    s1: Option<Value>,
    /// The arithmetic every rounding is to ([`BlendArith`]).
    arith: BlendArith,
}

/// The values of one lane class, each node computed once.
struct Class<'a> {
    inputs: &'a Inputs,
    lanes: Lanes,
    nodes: Vec<(Node, Value)>,
    /// The class reads `BlendColor` and `OneMinusBlendColor` both, so its one minus is of
    /// `BlendColor`'s read.
    one_minus_of_read: bool,
}

impl<'a> Class<'a> {
    /// The class of `lanes` computing `a` over `inputs`.
    fn of(inputs: &'a Inputs, lanes: Lanes, a: &Arith) -> Self {
        let n = nodes(a);
        Class {
            inputs,
            lanes,
            nodes: Vec::new(),
            one_minus_of_read: n.contains(&Node::Atom(Atom::C))
                && n.contains(&Node::OneMinus(Atom::C)),
        }
    }

    /// `v` (a `vec4`) as this class holds it: the vector, or its lane 3.
    fn own(&self, fb: &mut FunctionBuilder<'_>, v: Value) -> BResult<Value> {
        match self.lanes {
            Lanes::Vector => Ok(v),
            Lanes::Scalar => fb.composite_extract(v, &[3]),
        }
    }

    /// Lane 3 of `v` (a `vec4`), in every lane of this class.
    fn splat(&self, fb: &mut FunctionBuilder<'_>, v: Value) -> BResult<Value> {
        match self.lanes {
            Lanes::Vector => fb.vector_shuffle(v, v, &[3, 3, 3, 3]),
            Lanes::Scalar => fb.composite_extract(v, &[3]),
        }
    }

    fn constant(&self) -> BResult<(Value, Value, Value)> {
        self.inputs
            .constant
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))
    }

    fn atom(&mut self, fb: &mut FunctionBuilder<'_>, a: Atom) -> BResult<Value> {
        let i = self.inputs;
        match a {
            Atom::S => self.own(fb, i.s),
            Atom::Sa => self.splat(fb, i.s),
            Atom::D => self.own(fb, i.d),
            Atom::Da => self.splat(fb, i.d),
            Atom::C => {
                let (_, _, direct) = self.constant()?;
                self.own(fb, direct)
            }
            Atom::Ca => {
                let (c, _, _) = self.constant()?;
                self.splat(fb, c)
            }
            Atom::S1 | Atom::S1a => {
                let s1 = i.s1.ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
                match a {
                    Atom::S1 => self.own(fb, s1),
                    _ => self.splat(fb, s1),
                }
            }
        }
    }

    /// `h(1 - x)` of `half` `x`: `1 - x` is exact in `f32` but for `|x|` below `2^-13`, where its
    /// rounding stays clear of a `half` midpoint, so it rounds once.
    fn one_minus(&mut self, fb: &mut FunctionBuilder<'_>, x: Value) -> BResult<Value> {
        let f = self.lanes.floats();
        let one = fsplat(fb, f, 1.0)?;
        if self.inputs.arith == BlendArith::Float {
            return fb.binary_no_contraction(BinOp::FSub, one, x);
        }
        let m = fb.binary(BinOp::FSub, one, x)?;
        half_of(fb, f, m)
    }

    /// `x · y`: in `float` arithmetic rounded to `float` on its own, never fused into a sum, its
    /// zero signed on the bits ([`signed_product`]); a product of two `half`s is exact in `f32`.
    fn mul(&self, fb: &mut FunctionBuilder<'_>, x: Value, y: Value) -> BResult<Value> {
        match self.inputs.arith {
            BlendArith::Float => signed_product(fb, self.lanes.floats(), x, y),
            BlendArith::Half => fb.binary(BinOp::FMul, x, y),
        }
    }

    fn node(&mut self, fb: &mut FunctionBuilder<'_>, n: Node) -> BResult<Value> {
        if let Some(&(_, v)) = self.nodes.iter().find(|(m, _)| *m == n) {
            return Ok(v);
        }
        let f = self.lanes.floats();
        let v = match n {
            Node::Zero => fsplat(fb, f, 0.0)?,
            Node::One => fsplat(fb, f, 1.0)?,
            Node::Atom(a) => self.atom(fb, a)?,
            // Under fast math the precomputed one minus, but beside `BlendColor` in its class (a
            // 16-bit float attachment's difference, [`constant_difference`]) the one minus of
            // `BlendColor`'s read, as below.
            Node::OneMinus(Atom::C)
                if self.inputs.math == BlendMath::Fast && !self.one_minus_of_read =>
            {
                let (_, m, _) = self.constant()?;
                self.own(fb, m)?
            }
            Node::OneMinus(Atom::Ca) if self.inputs.math == BlendMath::Fast => {
                let (_, m, _) = self.constant()?;
                self.splat(fb, m)?
            }
            // The blend color's one minus reads the clamped color, but beside `BlendColor` in its
            // class `BlendColor`'s read.
            Node::OneMinus(Atom::C) => {
                let (c, _, direct) = self.constant()?;
                let x = match self.one_minus_of_read {
                    true => self.own(fb, direct)?,
                    false => self.own(fb, c)?,
                };
                self.one_minus(fb, x)?
            }
            Node::OneMinus(a) => {
                let x = self.atom(fb, a)?;
                self.one_minus(fb, x)?
            }
            Node::Saturate => {
                let sa = self.atom(fb, Atom::Sa)?;
                let da = self.atom(fb, Atom::Da)?;
                let m = self.one_minus(fb, da)?;
                metal_min(fb, f, sa, m)?
            }
        };
        self.nodes.push((n, v));
        Ok(v)
    }

    /// `x` times the small integer `c` (exact).
    fn scaled(&self, fb: &mut FunctionBuilder<'_>, c: i8, x: Value) -> BResult<Value> {
        match c {
            1 => Ok(x),
            -1 => fb.unary(UnOp::FNegate, x),
            c => {
                let k = fsplat(fb, self.lanes.floats(), f64::from(c))?;
                fb.binary(BinOp::FMul, x, k)
            }
        }
    }

    fn round(&self, fb: &mut FunctionBuilder<'_>, x: Value) -> BResult<Value> {
        match self.inputs.arith {
            // Every `float` operation here is rounded already.
            BlendArith::Float => Ok(x),
            BlendArith::Half => half_of(fb, self.lanes.floats(), x),
        }
    }

    /// `h(a · b + c)`: of `half` values exactly ([`half_sum`]); in `float` arithmetic correctly
    /// rounded ([`float_fma_exact`]).
    fn fma(&self, fb: &mut FunctionBuilder<'_>, [a, b, c]: [Value; 3]) -> BResult<Value> {
        let f = self.lanes.floats();
        if self.inputs.arith == BlendArith::Float {
            return float_fma_exact(fb, f, a, b, c);
        }
        let s = half_sum(fb, f.with(Held::Float(FloatFmt::Half)), a, b, c)?;
        half_of(fb, f, s)
    }

    /// `sign · operand`, and the factor, of `(sign, term)`.
    fn signed(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        (sign, t): (i8, Term),
    ) -> BResult<[Value; 2]> {
        let x = self.atom(fb, t.0)?;
        let x = self.scaled(fb, sign, x)?;
        let f = self.node(fb, t.1)?;
        Ok([x, f])
    }

    /// The value of inner sum `inner`: one summand as it is, more added in order, each addition
    /// rounded.
    fn inner(&mut self, fb: &mut FunctionBuilder<'_>, inner: &[Summand]) -> BResult<Value> {
        let f = self.lanes.floats();
        let mut acc: Option<Value> = None;
        for &(c, k) in inner {
            let x = match k {
                None => fsplat(fb, f, f64::from(c))?,
                Some(n) => {
                    let v = self.node(fb, n)?;
                    self.scaled(fb, c, v)?
                }
            };
            acc = Some(match acc {
                None => x,
                Some(a) => {
                    let one = fsplat(fb, f, 1.0)?;
                    self.fma(fb, [a, one, x])?
                }
            });
        }
        match acc {
            Some(v) => Ok(v),
            None => fsplat(fb, f, 0.0),
        }
    }

    fn eval(&mut self, fb: &mut FunctionBuilder<'_>, a: &Arith) -> BResult<Value> {
        match a {
            Arith::MinMax { max, source } => {
                let d = match self.lanes {
                    Lanes::Vector => Atom::D,
                    Lanes::Scalar => Atom::Da,
                };
                let s = self.atom(fb, *source)?;
                let d = self.atom(fb, d)?;
                let op = match max {
                    true => GlslOp::FMax,
                    false => GlslOp::FMin,
                };
                fb.glsl(op, &[s, d])
            }
            Arith::Zero => fsplat(fb, self.lanes.floats(), 0.0),
            Arith::Term { sign, term } => self.product(fb, (*sign, *term)),
            Arith::Fused { product, addend } => {
                let [x, f] = self.signed(fb, *product)?;
                let a = self.atom(fb, addend.1 .0)?;
                let a = self.scaled(fb, addend.0, a)?;
                self.fma(fb, [x, f, a])
            }
            Arith::Lerp { y, z, inner } => {
                let i = self.inner(fb, inner)?;
                let y = self.node(fb, *y)?;
                let z = self.node(fb, *z)?;
                self.fma(fb, [z, i, y])
            }
            Arith::Common { z, inner } => {
                let i = self.inner(fb, inner)?;
                let z = self.node(fb, *z)?;
                // A product of two `half`s is exact in `f32`.
                let p = self.mul(fb, z, i)?;
                self.round(fb, p)
            }
            Arith::Contracted { rounded, fused } => {
                let [x, f] = self.signed(fb, *rounded)?;
                let p = self.mul(fb, x, f)?;
                let r = self.round(fb, p)?;
                let [x, f] = self.signed(fb, *fused)?;
                self.fma(fb, [x, f, r])
            }
            Arith::Rounded {
                source,
                destination,
            } => {
                let s = self.product(fb, *source)?;
                let d = self.product(fb, *destination)?;
                let one = fsplat(fb, self.lanes.floats(), 1.0)?;
                self.fma(fb, [s, one, d])
            }
        }
    }

    /// `h(sign · operand · factor)` of `(sign, term)`: the operand alone where the factor is one;
    /// a product of two `half`s is exact in `f32`.
    fn product(&mut self, fb: &mut FunctionBuilder<'_>, (sign, t): (i8, Term)) -> BResult<Value> {
        let [x, f] = self.signed(fb, (sign, t))?;
        match t.1 {
            Node::One => Ok(x),
            _ => {
                let p = self.mul(fb, x, f)?;
                self.round(fb, p)
            }
        }
    }
}

/// Metal's `min(x, y)` in `SourceAlphaSaturated` (lanes of form `f`): a NaN operand gives the other,
/// and two zeros `+0` (a 16-bit float attachment's blends, every lane of
/// `min(sa, 1 - da)` where `sa` is `-0` beside the `+0` of a destination alpha of one, or either a
/// NaN, under safe math; under fast math the same rule leaves no lane of `RGBA16Float` but a
/// zero's sign off Metal's, and fewer of those than the device's `FMin` of `-0` first). The
/// device's `FMin` is the minimum of two numbers, its answer for a NaN or a zero's sign the
/// device's own (Vulkan leaves both undefined), so those lanes are decided on the bits and a
/// select.
fn metal_min(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, y: Value) -> BResult<Value> {
    let u = crate::lower::arith::narrow::u32_form(f);
    let ut = ty(fb, u)?;
    let m = fb.glsl(GlslOp::FMin, &[x, y])?;
    let xb = fb.convert(ConvOp::Bitcast, x, ut)?;
    let yb = fb.convert(ConvOp::Bitcast, y, ut)?;
    let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
    let infinity = splat(fb, u, 0x7F80_0000)?;
    let zero = splat(fb, u, 0)?;
    let either = fb.binary(BinOp::BitwiseOr, xb, yb)?;
    let either = fb.binary(BinOp::BitwiseAnd, either, magnitude)?;
    let zeros = fb.binary(BinOp::IEqual, either, zero)?;
    let plus = fsplat(fb, f, 0.0)?;
    let m = fb.select(zeros, plus, m)?;
    let xm = fb.binary(BinOp::BitwiseAnd, xb, magnitude)?;
    let x_nan = fb.binary(BinOp::UGreaterThan, xm, infinity)?;
    let m = fb.select(x_nan, y, m)?;
    let ym = fb.binary(BinOp::BitwiseAnd, yb, magnitude)?;
    let y_nan = fb.binary(BinOp::UGreaterThan, ym, infinity)?;
    fb.select(y_nan, x, m)
}

/// `x · y` (`float` lanes of form `f`) rounded to `float` on its own (`NoContraction`), a zero
/// product taking the sign IEEE 754 gives it (the factors' signs differing), put on the bits as an
/// integer: SPIRV-Cross renders a `NoContraction` `OpFMul` as `fma(x, y, 0)`, whose zero is `+0`
/// for a `-0` product (MoltenVK; see `fma::half_sum`), and a module without
/// `SignedZeroInfNanPreserve` lets the device ignore the sign of a zero.
fn signed_product(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, y: Value) -> BResult<Value> {
    let p = fb.binary_no_contraction(BinOp::FMul, x, y)?;
    let u = crate::lower::arith::narrow::u32_form(f);
    let (ut, ft) = (ty(fb, u)?, ty(fb, f)?);
    let xb = fb.convert(ConvOp::Bitcast, x, ut)?;
    let yb = fb.convert(ConvOp::Bitcast, y, ut)?;
    let pb = fb.convert(ConvOp::Bitcast, p, ut)?;
    let mag = splat(fb, u, 0x7FFF_FFFF)?;
    let sign = splat(fb, u, 0x8000_0000)?;
    let z = splat(fb, u, 0)?;
    let m = fb.binary(BinOp::BitwiseAnd, pb, mag)?;
    let zero = fb.binary(BinOp::IEqual, m, z)?;
    let s = fb.binary(BinOp::BitwiseXor, xb, yb)?;
    let s = fb.binary(BinOp::BitwiseAnd, s, sign)?;
    let w = fb.select(zero, s, pb)?;
    fb.convert(ConvOp::Bitcast, w, ft)
}

/// `s` (a `vec4`) clamped: its color lanes to `(lo, hi)`, alpha to `[0, 1]`, a NaN to the low
/// end.
pub(super) fn clamped(
    fb: &mut FunctionBuilder<'_>,
    s: Value,
    (lo, hi): (f32, f32),
) -> BResult<Value> {
    let low = per_lane_f(fb, [lo, lo, lo, 0.0])?;
    let high = per_lane_f(fb, [hi, hi, hi, 1.0])?;
    let nan = fb.unary(UnOp::IsNan, s)?;
    let s = fb.select(nan, low, s)?;
    fb.glsl(GlslOp::FClamp, &[s, low, high])
}

/// The arithmetic Metal blends a 16-bit float attachment in: the type of the function's output
/// (a `half4` output's blend rounds each step to `half`, a `float4` one's to
/// `float`, and the result converts to `half` as Metal's unblended write of a `float` does).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendArith {
    /// A `half4` output: each rounding is to `half`, nearest even.
    Half,
    /// A `float4` output: each rounding is to `float`, nearest even, and the result is a `float`,
    /// which the caller converts to the format.
    Float,
}

/// Metal's blend `b` ([`Blend::unorm8`]) of a four-lane output `s` (a `vec4` of `f32`: a `half4`
/// output's `half`s, or a `float4` one) over `texel`, the 8-bit unorm attachment's texel as the
/// storage image reads it (a `vec4` of `f32`, each channel its code over 255; a channel the format
/// lacks reads 0, alpha 1), in `arith`: a `vec4` of `f32` holding each blended lane, a `half`
/// (`BlendArith::Half`) or a `float` (`BlendArith::Float`), which the caller stores by the
/// device's conversion (`PrecisionNote::DeviceColorConversion`). Metal clamps the output to `[0, 1]`
/// first, a NaN to 0, every lane ([`clamped`]), and reads the destination as its code over 255
/// correctly rounded to the arithmetic's type ([`unorm8_tile_value`]), the value a fetch reads.
/// `color`, the encoder's blend color (a `uvec4` of its `f32`s' bits, `PushField::HalfBlendColor`,
/// where `b` reads it, [`Blend::reads_constant`]), is what `BlendColor` reads, clamped to `[0, 1]`
/// and rounded to `half` in `half` arithmetic (blend colors past both ends of every channel), and
/// `OneMinusBlendColor` reads the one minus of that read under either math, `h(1 - h(c))`
/// (`f(1 - c)` in `float`). A NaN blend color, which no probe drew, reads 0.
pub fn blend_unorm8(
    fb: &mut FunctionBuilder<'_>,
    b: &Blend,
    arith: BlendArith,
    (s, texel): (Value, Value),
    color: Option<Value>,
) -> BResult<Value> {
    let s = clamped(fb, s, (0.0, 1.0))?;
    let d = unorm8_tile_value(fb, texel, arith)?;
    clamped_blend(fb, b, arith, (s, None, d), color)
}

/// Blend `b` of the clamped output `s` (and second source `s1`, where `b` reads it) over the
/// destination `d`, `vec4`s of `f32` in `arith`, the blend color `color` clamped to `[0, 1]` (a
/// NaN 0) and rounded to `arith`, and its one minus that of the read: [`blend_unorm8`]'s
/// arithmetic.
fn clamped_blend(
    fb: &mut FunctionBuilder<'_>,
    b: &Blend,
    arith: BlendArith,
    (s, s1, d): (Value, Option<Value>, Value),
    color: Option<Value>,
) -> BResult<Value> {
    let f = floats();
    let constant = match color {
        Some(words) => {
            let ft = ty(fb, f)?;
            let c = fb.convert(ConvOp::Bitcast, words, ft)?;
            let zero = fsplat(fb, f, 0.0)?;
            let one = fsplat(fb, f, 1.0)?;
            let nan = fb.unary(UnOp::IsNan, c)?;
            let c = fb.select(nan, zero, c)?;
            let c = fb.glsl(GlslOp::FClamp, &[c, zero, one])?;
            let c = match arith {
                BlendArith::Half => half_of(fb, f, c)?,
                BlendArith::Float => c,
            };
            let m = fb.binary_no_contraction(BinOp::FSub, one, c)?;
            let m = match arith {
                BlendArith::Half => half_of(fb, f, m)?,
                BlendArith::Float => m,
            };
            Some((c, m, c))
        }
        None => None,
    };
    let inputs = Inputs {
        s,
        d,
        constant,
        math: b.math,
        s1,
        arith,
    };
    lanes(fb, b, &inputs)
}

/// Metal's read of each lane of `texel` (a `vec4` of `f32`, an 8-bit unorm texel as the storage
/// image reads it, `c / 255` for each code `c`, Vulkan's conversion of a normalized fixed-point
/// value) in `arith`: the code `c` (the lane times 255, rounded to the nearest integer, which any
/// decode within half a code of `c / 255` gives) over 255, correctly rounded to `float`
/// ([`div_exact`](crate::lower::arith::exact_division::div_exact)) for `BlendArith::Float`, and to
/// `half` for `BlendArith::Half` (the M3 Max under macOS 27.0.1: a fetch of every
/// code reads `h(c / 255)` as a `half4` and `f(c / 255)` as a `float4`; the blend reads the same).
/// The `half` takes no exact quotient: `c / 255` repeats the byte `c` without end, which keeps it
/// at least `2^-9` of a `half`'s step from a `half` midpoint (at least `2^-20` of the value, every
/// nonzero `c / 255` being a normal `half`), and `c` times the `f32` nearest `1 / 255`, one
/// correctly rounded `OpFMul` (Vulkan's precision table), lies within `2^-23` of `c / 255`
/// relatively (that constant's error and the product's rounding, `2^-24` each), so it rounds to
/// the same `half` ([`half_of`]); every code is checked
/// (`lower::arith::tests::blend8::a_code_s_half_is_its_product_s`).
pub fn unorm8_tile_value(
    fb: &mut FunctionBuilder<'_>,
    texel: Value,
    arith: BlendArith,
) -> BResult<Value> {
    let f = floats();
    let k255 = fsplat(fb, f, 255.0)?;
    let scaled = fb.binary(BinOp::FMul, texel, k255)?;
    let rounded = fb.glsl(GlslOp::RoundEven, &[scaled])?;
    match arith {
        BlendArith::Float => {
            let u = crate::lower::arith::narrow::u32_form(f);
            let ut = ty(fb, u)?;
            let code = fb.convert(ConvOp::ConvertFToU, rounded, ut)?;
            crate::lower::arith::exact_division::div_exact(fb, u, code, 255)
        }
        BlendArith::Half => {
            let inverse = fsplat(fb, f, f64::from(1.0f32 / 255.0))?;
            let v = fb.binary_no_contraction(BinOp::FMul, rounded, inverse)?;
            half_of(fb, f, v)
        }
    }
}

/// `x` (`f32` lanes of form `f`) rounded to `half`, nearest even, half denormals kept, an infinity
/// from 65520 up: each rounding of Metal's blend arithmetic. Where the module rounds to `half` by
/// the device's conversion under the 16-bit float controls ([`QuantizeForm::Controlled`]: the
/// profile has `shaderFloat16`, `RoundingModeRTE`, `DenormPreserve` and `SignedZeroInfNanPreserve`
/// for 16 bits; [`narrow::quantize_form`]), that conversion and
/// its widening ([`narrow::round`]): two instructions, Metal's `half(float)` by the Vulkan spec
/// (conversions are correctly rounded, to nearest even under `RoundingModeRTE`, denormals kept
/// under `DenormPreserve`, zero signs and infinities under `SignedZeroInfNanPreserve`), as the
/// module's own `half` results round on such a profile (exact, no note). Elsewhere the integer
/// sequence ([`f32_to_half`]), about fifty instructions, the same on every device. The two agree on
/// every `x` but a NaN's bits, which the conversion leaves to the device: the blends clamp a NaN
/// source to zero, and a 16-bit float blend stores every NaN as Metal's own.
pub fn half_of(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    match fb.module().quantize_form() {
        QuantizeForm::Controlled => narrow::round(fb, f.with(Held::Float(FloatFmt::Half)), x),
        _ => f32_to_half(fb, f, x, HalfRounding::NearestEven),
    }
}

/// The `vec4` of blend `b`'s lanes over `inputs`: the color class's three, and alpha's, each
/// lane the write does not store `d`'s.
fn lanes(fb: &mut FunctionBuilder<'_>, b: &Blend, inputs: &Inputs) -> BResult<Value> {
    let f = floats();
    let d = inputs.d;
    let color = match &b.color {
        Some(a) => Class::of(inputs, Lanes::Vector, a).eval(fb, a)?,
        None => d,
    };
    let alpha = match &b.alpha {
        Some(a) => Some(Class::of(inputs, Lanes::Scalar, a).eval(fb, a)?),
        None => None,
    };
    let ft = ty(fb, f)?;
    let mut lanes = Vec::with_capacity(4);
    for k in 0..3u32 {
        lanes.push(fb.composite_extract(color, &[k])?);
    }
    lanes.push(match alpha {
        Some(a) => a,
        None => fb.composite_extract(d, &[3])?,
    });
    fb.composite_construct(ft, &lanes)
}
