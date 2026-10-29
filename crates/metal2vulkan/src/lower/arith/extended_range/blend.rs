//! Metal's blend into an extended-range attachment, which no device computes on the raw texels: the
//! function computes it in the shader, inside the pixel interlock, from the texel's tile value
//! ([`super::tile_value`]), and stores the kernel write's encode of the result ([`super::encode`],
//! [`super::encode_keeping`] under a write mask).
//!
//! **The measurement.** Our probes (an Apple M4, macOS 27.0.1, plain Metal): configurations of
//! every factor and operation of the color channels and of alpha, the blend color, write masks, and
//! `half` and `float` outputs, each over random sources (NaNs, the infinities, denormals, values
//! past every range) and random texels of `BGRA10_XR`, `BGR10_XR` and `RGB10A8_2P_XR10`; the M3 Max
//! under macOS 26.7 (25G229) gave every configuration of our probe the M4's bytes (15,762 of
//! 15,762), so one model holds on both chips and both OS levels.
//!
//! **The rule** for a `half4` output into a linear form. Metal clamps the source first, a color
//! lane to [`super::SOURCE_RANGE`] (a NaN to its low end) and alpha to `[0, 1]` (a NaN to 0), and
//! the blend color likewise, rounded to `half` ([`super::blend_color::constant_words`]); the
//! destination is the tile
//! value, so every value the blend reads is a `half`. The color lanes the write stores (two or
//! three of them, one vector) and alpha (a scalar) each compute `S op D`, `S = s Fs` and
//! `D = d Fd`, in one form the factors' structure picks ([`Blend::of`]), each sum and product
//! rounded once to `half`, nearest even (`h`):
//!
//! 1. `Min` and `Max`: `min(s, d)` and `max(s, d)`, the factors unread.
//! 2. A zero factor drops its term: `0`, or `h(±S)` or `h(±D)` alone (but a 16-bit float
//!    attachment's blend without fast math, and one minus the source subtracted alone into one of
//!    two or more channels, [`Blend::half_float`]).
//! 3. A factor of one: `h(±S ± D)`, one fused multiply-add.
//! 4. Two products, added, where one's factor is `1 - z` and the other's operands hold `z`
//!    (a lerp): `h(y + z · inner)`, `y` the first product's operand and `inner` the other
//!    product's other operand minus `y`.
//! 5. Two products sharing an operand `z`: `h(z · inner)`, `inner` the sum of the other two.
//! 6. Otherwise one product is rounded first and the other fused into the sum: the source's
//!    under `ReverseSubtract`, the destination's under `Subtract`, and under `Add` by the
//!    factors' ranks ([`color_rank`], [`alpha_rank`]).
//!
//! An inner sum expands each `1 - z` into `1` and `-z` (but a blend color's, a color lane's of
//! an alpha, and, in alpha, the one-minus-blend-alpha the color factors share once it comes
//! second), merges equal terms, and adds them in a fixed order ([`inner_sum`]), each addition
//! rounded to `half`. `SourceAlphaSaturated` is `min(sa, h(1 - da))` in color and one in alpha,
//! and `1 - x` is `h(1 - x)`. `BGR10_XR`'s destination alpha is one: its factors fold
//! (`DestinationAlpha` one, `OneMinusDestinationAlpha` zero).
//!
//! **A `float4` output into a linear form** (the same sweeps' `float4` configurations, and every
//! `float4` row of the lane table's linear forms: [`Blend::for_output`]). Metal
//! clamps its color lanes to [`FLOAT_SOURCE_RANGE`] (a NaN to its low end) and blends them
//! unrounded ([`Atom::Sf`]); its alpha, and a color lane's read of the source's alpha, are its
//! `half` (nearest even), clamped as a `half4` output's. A product, a sum, or `1 - x` that reads an
//! unrounded lane is computed exactly and rounded to `float` before its rounding to `half`; `min`
//! and `max` round their result to `half`. The factors' structure picks the `half4` rule's forms
//! but where an unrounded lane moves Metal's choice:
//!
//! - one product kept a `float` and the other rounded to `half` first, added in `float`
//!   ([`Arith::Kept`]: a source factor of one minus the source color beside a destination factor
//!   of an alpha, the destination color, or one minus the source's alpha);
//! - the source times the destination twice over: `h(d · h(2 sf))`; a difference of the
//!   destination's products by a splat alpha: its inner sum rounded to `half` first
//!   ([`Arith::Common`]'s `round_inner`);
//! - alpha keeps more one-minus nodes whole in its sums (one minus the destination alpha; the
//!   color factors' one minus the source alpha added first in a difference), and ranks a few
//!   pairs otherwise ([`kept_whole`], the `Add` ranks).
//!
//! **A module without fast math** (`air.compile.fast_math_disable`, [`BlendMath::Rounded`]; the
//! sweeps built under `-fmetal-math-mode=safe`, every key of each, and a computed `float4`'s 21
//! tuples into each linear form). Metal's compiler folds the blend into the function without fusing
//! it: the output is rounded to `half` (a `float4` one too) and clamped as a `half4` output's, each
//! term's product rounded to `half` (a factor of one its operand, of zero no term), and the two
//! terms' sum rounded; `min` and `max` as the fast forms. Every `1 - x` is `h(1 - x)` of the `half`
//! it reads, the blend color's color lanes' too ([`Node::OneMinus`]). An `_sRGB` form's blends take
//! the same forms, of a `float4` output too, rounded to `half` first (the lane table's probe built
//! under safe math, and our probe under relaxed math, the same bytes: every configuration made
//! into each of the six forms).
//!
//! Without fast math the model also holds where the fast forms are not built: a mask of one color
//! channel, the color channels' `BlendColor` with `OneMinusBlendColor` under a difference, and any
//! blend into an `_sRGB` form, masked too (for the linear forms, for every form), and the second
//! source's factors, which read the function's index-1 output rounded to `half` and clamped as the
//! first ([`Atom::S1`]: every key that names one, and our probe's table of every form). A blend
//! into `BGRA10_XR_sRGB` clamps the blend color's color channels to that form's source range, as it
//! does the source ([`super::SRGB_SOURCE_RANGE`]; the binder's row holds the color clamped to both
//! ranges' hull, [`super::blend_color::constant_words`]), where `OneMinusBlendColor` reads it; but
//! `BlendColor` itself reads a color channel at or past `super::blend_color::SRGB_DIRECT_BOUND` (an
//! `f32` 45 steps below the range's top) as `1.66796875`, a `half` below the top, which the binder
//! marks past the hull's top (`super::blend_color::PAST_DIRECT_BOUND`). A color class that reads
//! `BlendColor` and `OneMinusBlendColor` both reads the one minus of `BlendColor`'s read
//! (`h(1 - 1.66796875)` past the same bound; the safe table's configurations 17 and 40 and the
//! second source's 20, our probes' sweep), while a color class reading the one minus alone reads it
//! of the clamped color beside an alpha class reading `BlendColor` too.
//!
//! **The `_sRGB` forms under fast math.** A blend into `BGR10_XR_sRGB` or
//! `RGB10A8_2P_XR10_sRGB` takes the linear forms' models, a `half4` output's and a `float4`
//! one's: the lane table's every row of the two forms (each configuration, mask and output type)
//! is Metal's bytes under them. `BGRA10_XR_sRGB` takes them for the narrow blends alone
//! ([`Blend::narrow`]: a `float4` output among them only unmasked, where its color lanes take one
//! term alone, rounded to `half` first): beyond them the table's rows differ, 7 of its `half4`
//! rows (each reading `OneMinusBlendColor`) and 76 of its `float4` ones ([`fast_narrow_only`]).
//!
//! **A 32-bit float attachment's blend** ([`Blend::float32`]) is these fast forms evaluated in
//! `float` rather than in `half`, measured over the committed recordings of
//! `half-output-blends-r32f`, `-rg32f` and `-rgba32f` ([`MEASURED_THIRTY_TWO`]). It is planned
//! under fast math alone, and no blend color's difference is a fitted form there.
//!
//! **Not built** ([`ExtendedRangeBlend`]) under fast math: a factor of the second source (under a `float4`
//! output's model, in any of the four factors); the color channels' `BlendColor` with
//! `OneMinusBlendColor` under `Subtract` or `ReverseSubtract` (Metal's result there follows
//! neither form for every blend color the sweeps drew); a write mask of exactly one color channel
//! (Metal blends that channel otherwise than the vector form); a `BGRA10_XR_sRGB` blend beyond the
//! narrow ones (`SrgbForm`), and a `float4` output into it beyond them (`FloatOutput`), both
//! measured only; and any other output.

use super::blend_color::SRGB_DIRECT_TOP;
use super::{floats, per_lane_f, source_range, SRGB_SOURCE_RANGE};
use crate::error::{Error, ExtendedRangeBlend, Feature, Unsupported};
use crate::lower::arith::fma::{float_fma_exact, half_sum};
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::narrow;
use crate::lower::arith::ops::{fsplat, splat, ty};
use crate::lower::arith::texel_round::{f32_to_half, HalfRounding};
use crate::metal::format::ExtendedRange;
use crate::metal::{BlendFactor, BlendOperation, ColorWriteMask, PixelFormat};
use crate::pipeline::lower_inputs::BlendEquation;
use crate::spirv::builder::{FunctionBuilder, QuantizeForm};
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, ConvOp, GlslOp, UnOp};

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
    /// A `float4` output's color lane unrounded: clamped to [`FLOAT_SOURCE_RANGE`] (a NaN to its
    /// low end), a `float`.
    Sf,
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
            (Atom::S | Atom::Sa | Atom::Sf, Order::Source) => 0,
            (Atom::D | Atom::Da, Order::Source) => 2,
            (Atom::S | Atom::Sa | Atom::Sf, Order::Destination) => 2,
            (Atom::D | Atom::Da, Order::Destination) => 0,
            (Atom::Da, Order::Scalar) => 0,
            (Atom::D, Order::Scalar) => 1,
            (Atom::Sa, Order::Scalar) => 2,
            (Atom::S | Atom::Sf, Order::Scalar) => 3,
        }
    }
}

/// An inner sum's order among its atoms ([`Atom::order`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    /// The source's before the destination's.
    Source,
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
    /// `h(1 - x)`; the blend color's from [`super::blend_color::constant_words`].
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
    /// `min(s, d)` or `max(s, d)`, `s` the `source` atom (rounded to `half` where a `float`).
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
    /// `h(z · inner)`, `inner` rounded to `half` first where `round_inner`.
    Common {
        z: Node,
        inner: Vec<Summand>,
        round_inner: bool,
    },
    /// `h(sign · h(rounded) + sign · fused)`, the fused product added exactly.
    Contracted {
        rounded: (i8, Term),
        fused: (i8, Term),
    },
    /// `h(f(f(sign · kept) + sign · h(rounded)))`, `f` rounding to `float`: the kept product a
    /// `float`'s, added in `float` (a `float4` output's color lanes).
    Kept {
        kept: (i8, Term),
        rounded: (i8, Term),
    },
    /// `h(h(sign · source) + h(sign · destination))`, a term of factor one its operand: a module
    /// without fast math's ([`BlendMath::Rounded`]).
    Rounded {
        source: (i8, Term),
        destination: (i8, Term),
    },
}

/// Metal's blend of one attachment's write, as the shader computes it ([`blend`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Blend {
    /// The color lanes' arithmetic, where the write stores two or three of them.
    pub(crate) color: Option<Arith>,
    /// Alpha's, where the write stores it.
    pub(crate) alpha: Option<Arith>,
    /// The blends: added, source factor zero or one and destination factor zero, one, or one minus
    /// the source's alpha, for color and for alpha.
    narrow: bool,
    /// A blend whose color lanes take one term alone, which takes a `float` output rounded to
    /// `half` first.
    takes_float: bool,
    /// The write keeps a channel of the format.
    masked: bool,
    /// The blend of a `float4` output into a linear form: its color lanes read the source
    /// unrounded ([`Atom::Sf`]).
    pub(crate) float: bool,
    /// The function's math mode: every one minus is `h(1 - x)` of the `half` it reads under
    /// [`BlendMath::Rounded`], the blend color's too.
    pub(crate) math: BlendMath,
    /// A 16-bit float attachment of two or more channels under fast math, whose destination
    /// product subtracted alone by one minus a source value Metal computes with the factor negated
    /// ([`Blend::half_float`]).
    negates_one_minus: bool,
    /// A blend of a state measured word for word against Metal's own stored codes
    /// ([`MEASURED_SIXTEEN`], [`MEASURED_THIRTY_TWO`]), whose forms are Metal's whether or not the
    /// sweeps fit them in every state ([`Blend::fitted`]).
    measured: bool,
    /// A 32-bit float attachment's blend ([`Blend::float32`]), whose fast forms were measured over
    /// one sweep of states rather than the 16-bit family's many: a blend color's difference is no
    /// fitted form there ([`Blend::fitted`]).
    wide: bool,
}

/// The range Metal clamps a `float4` output's color lanes to before blending into an attachment
/// of a linear form: the decimal literals `-0.752941` and `1.25098` as `float`, three `float`
/// steps inside the codes' values `-384 / 510` and `638 / 510` (every source beyond them or a NaN,
/// which lands on the low end, of either sweep).
pub(crate) const FLOAT_SOURCE_RANGE: (f32, f32) =
    (f32::from_bits(0xBF40_C0BE), f32::from_bits(0x3FA0_201D));

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

/// The blend states a 16-bit float attachment's fast-math blend was measured in word for word
/// against Metal's own stored codes, color and alpha factors alike, whose forms [`Blend::fitted`]
/// holds although the state sweeps do not fit a contracted pair or a common factor in every state:
/// the source and destination factors, under `Add`.
///
/// Measured against the committed recordings of plain Metal on an Apple M3 Max (macOS 27.0.1) of
/// `half-output-blends-r16f`, `-rg16f` and `-rgba16f` ("conformance"): every word of each state's
/// site in each of the three formats, 1,792 a state, the destinations each case's own generator
/// draws and the sources its sweep. `half-output-blends-rgba16f`'s `da-1mda` needs the alpha
/// class's common factor ([`plan`]) to hold at its alpha.
///
/// A state is here only where no word of any of the three formats differs, which is what says the
/// form is Metal's and not one width's coincidence. Two states the sites measure are out:
/// `SourceAlpha` over `OneMinusSourceAlpha`, whose contracted pair holds at every word of
/// `-rg16f` and differs at 5 of `-r16f` and 13 of `-rgba16f`; and `SourceAlphaSaturated` over
/// `OneMinusSourceAlpha`, where Metal rounds the destination's product first at one and at four
/// channels (every word of `-r16f` and `-rgba16f`) and the source's at two (every word of
/// `-rg16f`, the other order differing at 47): no form the state decides is Metal's at every
/// width, so the device blends it and the pipeline announces
/// `PrecisionNote::HalfOutputFastMathBlend`.
const MEASURED_SIXTEEN: [(BlendFactor, BlendFactor); 2] = [
    (BlendFactor::SourceColor, BlendFactor::DestinationColor),
    (
        BlendFactor::DestinationAlpha,
        BlendFactor::OneMinusDestinationAlpha,
    ),
];

/// The blend states a 32-bit float attachment's fast-math blend was measured in word for word
/// against Metal's own stored codes, color and alpha factors alike, whose forms [`Blend::fitted`]
/// holds for an attachment of two or more channels ([`Blend::float32`]): the same two pairs
/// [`MEASURED_SIXTEEN`] names, under `Add`.
///
/// Measured against the committed recordings of plain Metal on an Apple M3 Max (macOS 27.0.1) of
/// `half-output-blends-r32f`, `-rg32f` and `-rgba32f` ("conformance"): every word of every one of
/// those cases' twenty-one states, 5,376 between them, the destinations each case's own generator
/// draws (multiples of `2^-20` in `[0, 1)`) and the sources its sweep. Of the seventeen states
/// whose factors read no blend color, every word of `RG32Float` and of `RGBA32Float` is Metal's
/// under these forms, `RGBA32Float`'s `da-1mda` alpha included, where ten of the seventeen take a
/// form [`Blend::fitted`] holds without a measurement (no term, one term, one fused multiply-add,
/// a minimum or maximum) and these two need one.
///
/// **One channel is out.** `SourceColor` over `DestinationColor` into `R32Float` differs at 18 of
/// its 256 words, by a code, where Metal rounds the **destination's** product first: the device's
/// own blend is Metal's there and keeps it, as `half-output-blends-r32f`'s `sc-dc` site records.
/// So a one-channel attachment's state is not measured here, which is why the form that holds at
/// two and four channels is not taken at one. The width-dependence is the 16-bit family's too
/// ([`MEASURED_SIXTEEN`]'s `SourceAlphaSaturated` note).
///
/// A state whose factors read the blend color is out of this list and of [`Blend::fitted`]: a
/// 32-bit float attachment's blend color difference is not measured, and two of those states do
/// differ from Metal under these forms (`BlendAlpha` over `OneMinusBlendAlpha` into `R32Float` at
/// 77 of 256 words, `BlendColor` over `OneMinusSourceAlpha` into `RGBA32Float` at 10 of 1,024), so
/// the device blends them and the pipeline announces `PrecisionNote::HalfOutputFastMathBlend`.
const MEASURED_THIRTY_TWO: [(BlendFactor, BlendFactor); 2] = MEASURED_SIXTEEN;

/// Whether the blend `eq` is one of the measured states `states` ([`MEASURED_SIXTEEN`],
/// [`MEASURED_THIRTY_TWO`]): its color and its alpha factors the one pair, added, as each measured
/// state's are.
fn measured_state(eq: &BlendEquation, states: &[(BlendFactor, BlendFactor)]) -> bool {
    let added = eq.rgb_operation == BlendOperation::Add
        && eq.alpha_operation == BlendOperation::Add
        && (eq.source_rgb, eq.destination_rgb) == (eq.source_alpha, eq.destination_alpha);
    added && states.contains(&(eq.source_rgb, eq.destination_rgb))
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

/// A color factor's rank, which picks the contracted form of an added pair (the module docs'
/// form 6): the source's product is rounded first where its rank is at most the destination's,
/// but the destination's at equal ranks 4 and 5.
fn color_rank(f: BlendFactor) -> u8 {
    use BlendFactor as F;
    match f {
        F::SourceColor | F::OneMinusSourceColor => 0,
        F::SourceAlpha | F::OneMinusSourceAlpha | F::DestinationAlpha => 1,
        F::DestinationColor => 2,
        F::OneMinusDestinationColor | F::OneMinusDestinationAlpha => 3,
        F::SourceAlphaSaturated | F::BlendAlpha => 4,
        F::BlendColor | F::OneMinusBlendAlpha => 5,
        F::OneMinusBlendColor => 6,
        _ => 9,
    }
}

/// Alpha's factor rank ([`color_rank`]'s): the source's product is rounded first where its rank
/// is at most the destination's.
fn alpha_rank(f: BlendFactor) -> u8 {
    use BlendFactor as F;
    match f {
        F::SourceAlphaSaturated => 0,
        F::SourceColor | F::OneMinusSourceColor | F::SourceAlpha | F::OneMinusSourceAlpha => 1,
        F::DestinationColor
        | F::OneMinusDestinationColor
        | F::DestinationAlpha
        | F::OneMinusDestinationAlpha => 3,
        F::BlendColor | F::BlendAlpha => 4,
        F::OneMinusBlendColor | F::OneMinusBlendAlpha => 5,
        _ => 9,
    }
}

/// An 8-bit unorm attachment's factor rank, of a color factor or alpha's where `alpha`, in the
/// arithmetic `arith`: of an added pair, the destination's product is rounded first where the
/// source factor's rank is at least the destination factor's, else the source's.
fn unorm8_rank(f: BlendFactor, alpha: bool, arith: BlendArith) -> u8 {
    use BlendFactor as F;
    let float = arith == BlendArith::Float;
    match f {
        F::SourceColor
        | F::SourceAlpha
        | F::DestinationColor
        | F::OneMinusDestinationColor
        | F::DestinationAlpha
        | F::OneMinusDestinationAlpha => 0,
        F::OneMinusSourceColor | F::OneMinusSourceAlpha if alpha && !float => 0,
        F::OneMinusSourceColor | F::OneMinusSourceAlpha | F::SourceAlphaSaturated => 1,
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
/// alpha's model says so, and a sum keeps one whole where added second) and `leaves` (kept whole
/// alone, a `float4` output's one minus the destination alpha).
#[derive(Clone, Debug, Default)]
struct Shared {
    values: Vec<Node>,
    leaves: Vec<Node>,
}

/// Whether an inner sum keeps the one-minus node `n`, added with `sign` as its `i`th pair (from
/// the destination's product where `destination`), whole beyond the nodes that always stay whole
/// ([`inner_sum`]): a shared value added second; under a `float4` output's alpha
/// (`float_alpha`), a shared value or leaf added second or first in a difference (`op`), and in a
/// lerp's destination form (`lerp_destination`) a shared value added anywhere. An 8-bit unorm
/// attachment's blend (`unorm8`) keeps a shared value added from the destination's product.
fn kept_whole(
    (n, sign, i, destination): (Node, i8, usize, bool),
    shared: &Shared,
    (float_alpha, op, lerp_destination, unorm8): (bool, BlendOperation, bool, bool),
) -> bool {
    let added = sign > 0;
    if unorm8 {
        return shared.values.contains(&n) && added && destination;
    }
    if !float_alpha {
        return shared.values.contains(&n) && added && i > 0;
    }
    let any = shared.values.contains(&n) || shared.leaves.contains(&n);
    (any && added && (i > 0 || op == BlendOperation::Subtract))
        || (lerp_destination && shared.values.contains(&n) && added)
}

/// The summands of `pairs` (each a node, its sign, and whether it comes from the destination's
/// product, in the IR's order) as Metal's compiler adds them: each `1 - z` expanded into `1` and
/// `-z` but where it stays one value (a blend color's; an alpha's in a color lane of a vector
/// class, `scalar` false; where `keep` says so, [`kept_whole`]), equal terms merged, and the terms
/// ordered: the constant last, but between the two others where they are not both subtracted,
/// after the one of the lower [`Atom::order`] under `order`. Where
/// `half_reads` (a `float4` output's color lanes), the first two summands read an unrounded source
/// lane ([`Atom::Sf`]) beside a splat alpha as `half` ([`Atom::S`]).
fn inner_sum(
    pairs: [(Node, i8, bool); 2],
    (scalar, order): (bool, Order),
    keep: &dyn Fn((Node, i8, usize, bool)) -> bool,
    half_reads: bool,
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
    let mut ordered = match (constant, rest.as_slice()) {
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
    };
    if half_reads {
        let splat = |m: &Summand| matches!(m.1, Some(Node::Atom(Atom::Sa | Atom::Da)));
        if let [first, second, ..] = ordered.as_mut_slice() {
            let beside = (splat(second), splat(first));
            for (m, splat_beside) in [(first, beside.0), (second, beside.1)] {
                if splat_beside && m.1 == Some(Node::Atom(Atom::Sf)) {
                    m.1 = Some(Node::Atom(Atom::S));
                }
            }
        }
    }
    ordered
}

/// Whether term `t` reads an unrounded source lane ([`Atom::Sf`]): its product is a `float`'s.
fn raw_term(t: &Term) -> bool {
    t.0 == Atom::Sf || t.1 == Node::Atom(Atom::Sf)
}

/// The arithmetic of a lane class (alpha where `alpha`) of factors `fs` and `fd` under `op`,
/// `shared` the one-minus nodes of the color factors that alpha's sums and lerps take apart,
/// `opaque_` for a format without alpha, `float` for a `float4` output's, in the rules of
/// `family`. `scalar` marks a color class of one channel (an 8-bit unorm attachment's write mask
/// of one color channel), which Metal plans as a scalar: an alpha read there is no splat, but a
/// one minus an alpha that its destination factor is or that alpha's factors compute (`shared`)
/// stays whole in its sums and takes no lerp.
fn plan(
    (fs, fd, op): (BlendFactor, BlendFactor, BlendOperation),
    (alpha, scalar): (bool, bool),
    shared: &Shared,
    (opaque_, float): (bool, bool),
    family: Family,
) -> Result<Arith, Error> {
    let unorm8 = matches!(family, Family::Unorm8(_));
    let order = match (unorm8, scalar) {
        (false, _) => Order::Source,
        (true, false) => Order::Destination,
        (true, true) => Order::Scalar,
    };
    // A `float4` output's color lanes read their source lane unrounded.
    let raw = float && !alpha;
    let float_alpha = float && alpha;
    let (sa, da) = match alpha {
        true => (Atom::Sa, Atom::Da),
        false => (Atom::S, Atom::D),
    };
    let s = if raw { Atom::Sf } else { sa };
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
    let unrounded = |n: Node| match (raw, n) {
        (true, Node::Atom(Atom::S)) => Node::Atom(Atom::Sf),
        (true, Node::OneMinus(Atom::S)) => Node::OneMinus(Atom::Sf),
        (_, n) => n,
    };
    let fast = BlendMath::Fast;
    let (mut nfs, mut nfd) = (
        unrounded(factor(fs, alpha, fast)?),
        unrounded(factor(fd, alpha, fast)?),
    );
    if opaque_ {
        (nfs, nfd) = (opaque(nfs), opaque(nfd));
    }
    // The source's product reads it as `half` beside a splat alpha factor.
    let s_operand = match nfs {
        Node::Atom(Atom::Sa | Atom::Da) if raw => Atom::S,
        _ => s,
    };
    let st: Term = (s_operand, nfs);
    let dt: Term = (da, nfd);
    match (nfs == Node::Zero, nfd == Node::Zero) {
        (true, true) => return Ok(Arith::Zero),
        (true, false) => return Ok(Arith::Term { sign: pd, term: dt }),
        (false, true) => {
            // The reversed source times the source alpha reads the source unrounded.
            let reversed = op == BlendOperation::ReverseSubtract && nfs == Node::Atom(Atom::Sa);
            let term = if raw && reversed { (s, nfs) } else { st };
            return Ok(Arith::Term { sign: ps, term });
        }
        (false, false) => {}
    }
    if nfs == Node::One || nfd == Node::One {
        let (product, mut addend) = match nfs == Node::One && nfd != Node::One {
            true => ((pd, dt), (ps, st)),
            false => ((ps, st), (pd, dt)),
        };
        // A lone source added beside a splat alpha factor's product is read as `half`.
        let splat = matches!(product.1 .1, Node::Atom(Atom::Sa | Atom::Da));
        if raw && op != BlendOperation::Subtract && splat && addend.1 .0 == Atom::Sf {
            addend.1 .0 = Atom::S;
        }
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
    let keep_in = |lerp_destination: bool| {
        move |k: (Node, i8, usize, bool)| match scalar {
            true => alone.contains(&k.0),
            false => kept_whole(k, shared, (float_alpha, op, lerp_destination, unorm8)),
        }
    };
    // An 8-bit unorm attachment's lerp takes the destination's product for `y`'s first.
    let mut sides = [(true, (sops, dops)), (false, (dops, sops))];
    if unorm8 {
        sides.reverse();
    }
    if op == BlendOperation::Add {
        for (source_side, (yt, zo)) in sides {
            for (y, f) in both_orders(yt) {
                let Node::OneMinus(za) = f else {
                    continue;
                };
                // No lerp of a one minus the color factors computed (8-bit unorm: the destination
                // alpha's too).
                if shared.values.contains(&f) && (za == Atom::Ca || (unorm8 && za == Atom::Da)) {
                    continue;
                }
                let both = [sops.0, sops.1].contains(&f) && [dops.0, dops.1].contains(&f);
                if !alpha && !scalar && (za.splat() || y == Node::Atom(za) || both) {
                    continue;
                }
                if scalar && (alone.contains(&f) || y == Node::Atom(za) || both) {
                    continue;
                }
                // An 8-bit unorm attachment's alpha takes no lerp of the destination alpha beside
                // itself or held by both products, and nor does a 16-bit float one's: Metal's
                // alpha there is the common factor's form, `h(da · h(h(1 + sa) - da))`
                // (`half-output-blends-rgba16f`'s `da-1mda` and `dc-1mdc`, whose alpha factors are
                // both the destination alpha and its one minus: every one of the 512 words of the
                // two sites, against the committed Metal recordings, where the lerp holds 480).
                let no_destination_lerp =
                    unorm8 || matches!(family, Family::HalfFloat | Family::Float32);
                if no_destination_lerp && alpha && za == Atom::Da && (y == Node::Atom(za) || both) {
                    continue;
                }
                // A `float4` output's alpha takes no lerp of a one minus the source alpha the
                // color factors share or both products hold, nor of the source's own product
                // `sa (1 - sa)`.
                let kept = za == Atom::Sa && (shared.values.contains(&f) || both);
                if float_alpha && (kept || (source_side && y == Node::Atom(za))) {
                    continue;
                }
                let z = Node::Atom(za);
                if let Some((_, x)) = both_orders(zo).into_iter().find(|&(zj, _)| zj == z) {
                    let keep = keep_in(!source_side);
                    // `x` comes from the other product than `y`.
                    let pairs = [(x, 1, source_side), (y, -1, !source_side)];
                    let inner = inner_sum(pairs, (alpha || scalar, order), &keep, raw);
                    return Ok(Arith::Lerp { y, z, inner });
                }
            }
        }
    }
    // Both products the source times the destination: the destination times the source
    // doubled, rounded.
    let source_by_destination = [Node::Atom(Atom::Sf), Node::Atom(Atom::D)];
    let is_sd = |p: (Node, Node)| {
        source_by_destination.contains(&p.0) && source_by_destination.contains(&p.1) && p.0 != p.1
    };
    if raw && op == BlendOperation::Add && is_sd(sops) && is_sd(dops) {
        return Ok(Arith::Common {
            z: Node::Atom(Atom::D),
            inner: vec![(2, Some(Node::Atom(Atom::Sf)))],
            round_inner: true,
        });
    }
    let keep = keep_in(false);
    for (z, x) in both_orders(sops) {
        for (zd, y) in both_orders(dops) {
            if z == zd && !matches!(z, Node::Zero | Node::One) {
                let pair = match op {
                    BlendOperation::ReverseSubtract => [(y, pd, true), (x, ps, false)],
                    _ => [(x, ps, false), (y, pd, true)],
                };
                // A difference of the destination's products by a splat alpha reads the source
                // unrounded and rounds its sum.
                let splat_difference = op == BlendOperation::Subtract
                    && matches!(x, Node::Atom(_))
                    && matches!(y, Node::Atom(Atom::Sa | Atom::Da));
                if raw && z == Node::Atom(Atom::D) && splat_difference {
                    let inner = inner_sum(pair, (alpha || scalar, order), &keep, false);
                    return Ok(Arith::Common {
                        z,
                        inner,
                        round_inner: true,
                    });
                }
                let inner = inner_sum(pair, (alpha || scalar, order), &keep, raw);
                return Ok(Arith::Common {
                    z,
                    inner,
                    round_inner: false,
                });
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
    use BlendFactor as F;
    if let (Family::Unorm8(arith), BlendOperation::Add) = (family, op) {
        let rank = |f| unorm8_rank(f, alpha, arith);
        return Ok(match rank(fs) >= rank(fd) {
            true => destination_first,
            false => source_first,
        });
    }
    Ok(match op {
        BlendOperation::ReverseSubtract => source_first,
        BlendOperation::Subtract => destination_first,
        _ => {
            let (rs, rd) = match alpha {
                true => (alpha_rank(fs), alpha_rank(fd)),
                false => (color_rank(fs), color_rank(fd)),
            };
            // A `float4` output's alpha: the source alpha times itself beside the destination's
            // one minus the shared source alpha rounds the source's product first.
            let shared_minus = nfd == Node::OneMinus(Atom::Sa) && shared.values.contains(&nfd);
            let one_minus_source = matches!(fs, F::OneMinusSourceColor);
            let alpha_one_minus = matches!(fs, F::OneMinusSourceAlpha | F::OneMinusSourceColor);
            if float_alpha && shared_minus && nfs == Node::Atom(Atom::Sa) {
                source_first
            } else if rs > rd || (!alpha && rs == rd && matches!(rs, 4 | 5)) {
                destination_first
            } else if raw
                && one_minus_source
                && raw_term(&st)
                && matches!(
                    fd,
                    F::DestinationAlpha
                        | F::DestinationColor
                        | F::OneMinusDestinationAlpha
                        | F::OneMinusSourceAlpha
                        | F::SourceAlpha
                )
            {
                // The source's product kept a `float`, the destination's rounded first.
                Arith::Kept {
                    kept: (ps, st),
                    rounded: (pd, dt),
                }
            } else if float_alpha
                && alpha_one_minus
                && matches!(fd, F::DestinationAlpha | F::DestinationColor)
            {
                destination_first
            } else {
                source_first
            }
        }
    })
}

/// The arithmetic of a lane class (alpha where `alpha`) of factors `fs` and `fd` under `op` in a
/// module without fast math ([`BlendMath::Rounded`]), `opaque_` for a format without alpha:
/// `min` or `max`; no term, one, or both terms' rounded products added and rounded
/// ([`Arith::Rounded`]). Where `keep_zero` (a 16-bit float attachment's blend) a zero factor keeps
/// its product, `x · 0`, in the sum ([`Class::zero_kept`]); elsewhere it leaves its term out,
/// which an extended-range form's clamped operands and encode cannot tell apart.
fn plan_rounded(
    (fs, fd, op): (BlendFactor, BlendFactor, BlendOperation),
    alpha: bool,
    (opaque_, keep_zero): (bool, bool),
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
        (true, true) if !keep_zero => Arith::Zero,
        (true, false) if !keep_zero => Arith::Term { sign: pd, term: dt },
        (false, true) if !keep_zero => Arith::Term { sign: ps, term: st },
        _ => Arith::Rounded {
            source: (ps, st),
            destination: (pd, dt),
        },
    })
}

/// The blends ([`Blend::narrow`]).
fn narrow(eq: &BlendEquation) -> bool {
    use BlendFactor as F;
    let source = |f: BlendFactor| matches!(f, F::Zero | F::One);
    let destination = |f: BlendFactor| matches!(f, F::Zero | F::One | F::OneMinusSourceAlpha);
    eq.rgb_operation == BlendOperation::Add
        && eq.alpha_operation == BlendOperation::Add
        && source(eq.source_rgb)
        && source(eq.source_alpha)
        && destination(eq.destination_rgb)
        && destination(eq.destination_alpha)
}

/// Metal's blend `eq` of a write into an attachment of `form` storing the channels `written` of
/// its format, a `float4` output's (into a linear form) where `float` (see [`Blend::of`]), in a
/// function of math `math`.
fn planned(
    eq: &BlendEquation,
    form: ExtendedRange,
    written: ColorWriteMask,
    (float, math): (bool, BlendMath),
) -> Result<Blend, Error> {
    let has_alpha = !matches!(form, ExtendedRange::Bgr10 { .. });
    let family = Family::Extended {
        narrow_only: fast_narrow_only(form),
    };
    planned_for(eq, (has_alpha, family), written, (float, math))
}

/// Whether Metal's fast-math blends into `form` follow the models for the narrow blends alone
/// ([`Blend::narrow`], unmasked): `BGRA10_XR_sRGB`, the one `_sRGB` form the lane table measures
/// otherwise beyond them (the module docs). The other two `_sRGB` forms take the linear forms'
/// models for every blend, as the linear forms do.
fn fast_narrow_only(form: ExtendedRange) -> bool {
    form == ExtendedRange::Bgra10 { srgb: true }
}

/// The attachment whose blend Metal's compiler builds, whose forms differ: an extended-range
/// form's (one whose fast-math blends the models hold for the narrow blends alone where
/// `narrow_only`, [`fast_narrow_only`]), a 16-bit float attachment's ([`Blend::half_float`]), or
/// an 8-bit unorm one's in the arithmetic of the output's type ([`Blend::unorm8`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Extended {
        narrow_only: bool,
    },
    HalfFloat,
    /// A 32-bit float attachment's ([`Blend::float32`]), which plans as a 16-bit float one's: the
    /// forms the measurement fitted are the same, computed in `float` rather than in `half`.
    Float32,
    Unorm8(BlendArith),
}

/// [`planned`] for a format with an alpha channel where `has_alpha`, of `family`: a 16-bit float
/// or 8-bit unorm attachment's blend takes fast math's color channels' blend color difference
/// (an extended-range form refuses it), and an 8-bit unorm one's a write mask of one color
/// channel, which Metal plans as a scalar.
fn planned_for(
    eq: &BlendEquation,
    (has_alpha, family): (bool, Family),
    written: ColorWriteMask,
    (float, math): (bool, BlendMath),
) -> Result<Blend, Error> {
    let narrow_only = family == Family::Extended { narrow_only: true };
    let sixteen = !matches!(family, Family::Extended { .. });
    let unorm8 = matches!(family, Family::Unorm8(_));
    // A zero factor keeps its product without fast math in a 16-bit float attachment's blend;
    // a 32-bit float one's is planned under fast math alone ([`Blend::float32`]).
    let keep_zero = family == Family::HalfFloat;
    let colors = [
        ColorWriteMask::RED,
        ColorWriteMask::GREEN,
        ColorWriteMask::BLUE,
    ]
    .into_iter()
    .filter(|&c| written.contains(c))
    .count();
    let alpha_written = has_alpha && written.contains(ColorWriteMask::ALPHA);
    let masked = colors < 3 || (has_alpha && !alpha_written);
    let narrow = narrow(eq);
    // The rounded model holds for any blend into an `_sRGB` form, a mask of one color channel,
    // and the color channels' blend color difference; the fast models for `BGRA10_XR_sRGB`'s
    // narrow blends alone.
    let fast = math == BlendMath::Fast;
    if fast && narrow_only && (!narrow || masked) {
        return Err(refused(ExtendedRangeBlend::SrgbForm));
    }
    let opaque_ = !has_alpha;
    let color = match colors {
        0 => None,
        1 if fast && !unorm8 => return Err(refused(ExtendedRangeBlend::OneColorChannel)),
        _ => {
            let pair = [eq.source_rgb, eq.destination_rgb];
            let differs = matches!(
                eq.rgb_operation,
                BlendOperation::Subtract | BlendOperation::ReverseSubtract
            ) && pair.contains(&BlendFactor::BlendColor)
                && pair.contains(&BlendFactor::OneMinusBlendColor);
            if fast && differs && !sixteen {
                return Err(refused(ExtendedRangeBlend::ConstantColorDifference));
            }
            let f = (eq.source_rgb, eq.destination_rgb, eq.rgb_operation);
            let scalar = unorm8 && colors == 1;
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
                BlendMath::Fast => plan(f, (false, scalar), &computed, (opaque_, float), family)?,
                BlendMath::Rounded => plan_rounded(f, false, (opaque_, keep_zero))?,
            })
        }
    };
    let alpha = match alpha_written {
        false => None,
        true => {
            // The one-minus-blend-alpha the color factors use, which alpha's sums keep whole once
            // it comes second; under a `float4` output also their one minus the source alpha,
            // and, kept whole alone, their one minus the destination alpha.
            let uses = |f: BlendFactor| [eq.source_rgb, eq.destination_rgb].contains(&f);
            let mut shared = Shared::default();
            if uses(BlendFactor::OneMinusBlendAlpha) {
                shared.values.push(Node::OneMinus(Atom::Ca));
            }
            if float && uses(BlendFactor::OneMinusSourceAlpha) {
                shared.values.push(Node::OneMinus(Atom::Sa));
            }
            if float && uses(BlendFactor::OneMinusDestinationAlpha) {
                shared.leaves.push(Node::OneMinus(Atom::Da));
            }
            // An 8-bit unorm attachment's alpha shares what color lanes it stores compute: one
            // minus the blend alpha, the source alpha, and the destination alpha (which
            // `SourceAlphaSaturated` computes too); with no color channel stored, the one minus
            // an alpha the color channels' destination factor is, which Metal computes still. A
            // minimum or maximum computes no factor.
            if unorm8 {
                use BlendFactor as F;
                shared = Shared::default();
                let factors =
                    !matches!(eq.rgb_operation, BlendOperation::Min | BlendOperation::Max);
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
            }
            let f = (eq.source_alpha, eq.destination_alpha, eq.alpha_operation);
            Some(match math {
                BlendMath::Rounded => plan_rounded(f, true, (opaque_, keep_zero))?,
                BlendMath::Fast => plan(f, (true, false), &shared, (opaque_, float), family)?,
            })
        }
    };
    let takes_float =
        narrow && (eq.source_rgb == BlendFactor::Zero || eq.destination_rgb == BlendFactor::Zero);
    Ok(Blend {
        color,
        alpha,
        narrow,
        takes_float,
        masked,
        float,
        math,
        negates_one_minus: false,
        measured: false,
        wide: false,
    })
}

impl Blend {
    /// Metal's blend `eq` of a `half4` output's write into an attachment of `form` storing the
    /// channels `written` of its format in a function of math `math`, where the model holds it
    /// (see the module docs): `Err` with the gap (`Feature::ExtendedRangeBlend`) where it does
    /// not, or with the placeholder a factor or operation is
    /// (`DescriptorError::PlaceholderEnumValue`).
    pub fn of(
        eq: &BlendEquation,
        form: ExtendedRange,
        written: ColorWriteMask,
        math: BlendMath,
    ) -> Result<Blend, Error> {
        planned(eq, form, written, (false, math))
    }

    /// [`Blend::of`] for an output of four lanes of `output` (`None` for another output) of a
    /// function of math `math`: a `float4` one into a linear form, `BGR10_XR_sRGB` or
    /// `RGB10A8_2P_XR10_sRGB` takes its own model (the module docs), which refuses a factor of the
    /// second source in any of the four factors; into `BGRA10_XR_sRGB`, the blends that take it
    /// ([`Blend::takes`]). Without fast math
    /// ([`BlendMath::Rounded`]) a blend takes a `half4` output or a `float4` one rounded to `half`
    /// first alike.
    pub fn for_output(
        eq: &BlendEquation,
        form: ExtendedRange,
        written: ColorWriteMask,
        (output, math): (Option<FloatFmt>, BlendMath),
    ) -> Result<Blend, Error> {
        if math == BlendMath::Rounded {
            return match output {
                Some(FloatFmt::Half | FloatFmt::F32) => {
                    planned(eq, form, written, (false, BlendMath::Rounded))
                }
                _ => Err(refused(ExtendedRangeBlend::Output)),
            };
        }
        if output != Some(FloatFmt::F32) || fast_narrow_only(form) {
            let b = Blend::of(eq, form, written, BlendMath::Fast)?;
            b.takes(output)?;
            return Ok(b);
        }
        let b = planned(eq, form, written, (true, BlendMath::Fast))?;
        let factors = [
            eq.source_rgb,
            eq.destination_rgb,
            eq.source_alpha,
            eq.destination_alpha,
        ];
        let second = |f: &BlendFactor| {
            use BlendFactor as F;
            matches!(
                f,
                F::Source1Color
                    | F::OneMinusSource1Color
                    | F::Source1Alpha
                    | F::OneMinusSource1Alpha
            )
        };
        match factors.iter().any(second) {
            true => Err(refused(ExtendedRangeBlend::DualSource)),
            false => Ok(b),
        }
    }

    /// Metal's blend `eq` of a `half` or `float` output of a module of math `math` into a 16-bit
    /// float attachment of format `pf` (`R16Float`, `RG16Float`, `RGBA16Float`) storing the
    /// channels `written` of four (a channel the format lacks counts as written: the store drops
    /// it), in the arithmetic of the output's type ([`BlendArith`]), computed by
    /// [`blend_half_float`]. `R16Float` and `RG16Float` have no alpha: their destination alpha is
    /// one, so `DestinationAlpha` is a factor of one and `OneMinusDestinationAlpha` of zero, which
    /// under fast math drops its term (a lone term `h(±S)` or `h(±D)`, both dropped `+0`). Under
    /// fast math into `RG16Float` or `RGBA16Float`, a destination's product subtracted alone
    /// (`Subtract` of a zero source factor) by one minus the source's color or alpha is
    /// `h(d · h(z - 1))`, the factor negated, whose zero has `d`'s sign where `z` is one; so is any
    /// one minus in alpha of a `float4` output; every other lone term is `h(±S)` or `h(±D)`, the
    /// operand negated, `R16Float`'s too (our probes `rb16b`, `rb16c` and `rb16e`, the M3 Max under
    /// macOS 27.0.1: `half4` and `float4` outputs, sources and texels of `±0`, `±1`, `2.5`,
    /// `-0.75`, `±inf` and NaN in each of color and alpha, every lane of 20 states with a zero
    /// factor into each format, and the blend color one). Without fast math
    /// the model's rounded forms ([`BlendMath::Rounded`], the module docs): each factor, each
    /// product and the sum rounded to the arithmetic's type, nearest even, the blend color rounded
    /// to `half` (a `half4` output) before its one minus (and our probes, plain Metal on the M3 Max
    /// under macOS 27.0.1: every one of 677 blend states of one key for color and alpha and 1,350
    /// of a key beside one minus the source alpha's, over 16,384 pixels of loaded `half4` and
    /// `float4` outputs, every channel of each 16-bit float format, and the 677 under four write
    /// masks that keep some channels of `RGBA16Float`). There a factor of zero keeps its product
    /// `x · 0` in the sum, as IEEE 754 adds: a NaN where `x` is an infinity or a NaN, else the
    /// other term, a zero sum `-0` only where both terms are `-0` (two zero factors alike), every
    /// NaN the one Metal stores, `0x7e00` (plain Metal on the M3 Max under macOS 27.0.1, each
    /// operation with a zero factor beside each factor, color and alpha, into `RGBA16Float` and
    /// `RG16Float` from `half4` and `float4` outputs, three seeds of 16,384 pixels whose sources
    /// and texels hold signed zeros, infinities and NaNs among them: every lane).
    /// `SourceAlphaSaturated` takes Metal's minimum under either math, a NaN operand giving the
    /// other and two zeros `+0` ([`metal_min`]). Under fast math the fast forms of the module docs,
    /// with nothing clamped: the blend color's one minus `h(1 - c)` of a color channel and
    /// `h(1 - h(a))` of alpha in `half`, `1 - c` in `float` (the same sweeps built under
    /// `MTLMathModeFast`: every class of the forms [`Blend::fitted`] names, each channel of every
    /// state; the sum and lerp forms fit most states but not all). A blend constant's difference
    /// (`BlendColor` with `OneMinusBlendColor`, `BlendAlpha` with `OneMinusBlendAlpha`, under
    /// `Subtract` or `ReverseSubtract`) fuses the minuend's product and rounds the subtrahend's
    /// first, the color channels' one minus beside `BlendColor` being of its read, `h(1 - h(c))` in
    /// `half` (each such state of both sweeps, every channel of `half4` and `float4` outputs,
    /// signed zeros included; and our probe through narrower outputs). A factor of the second
    /// source is no form here (`Feature::ExtendedRangeBlend` `DualSource`): its output is not
    /// planned beside the storage form's write.
    pub fn half_float(
        eq: &BlendEquation,
        pf: PixelFormat,
        written: ColorWriteMask,
        math: BlendMath,
    ) -> Result<Blend, Error> {
        let channels = match pf {
            PixelFormat::R16Float => 1,
            PixelFormat::RG16Float => 2,
            _ => 4,
        };
        let b = planned_for(
            eq,
            (channels == 4, Family::HalfFloat),
            written,
            (false, math),
        )?;
        let b = Blend {
            negates_one_minus: channels > 1 && math == BlendMath::Fast,
            measured: math == BlendMath::Fast && !b.masked && measured_state(eq, &MEASURED_SIXTEEN),
            ..b
        };
        match b.reads_second() {
            true => Err(refused(ExtendedRangeBlend::DualSource)),
            false => Ok(b),
        }
    }

    /// Metal's blend `eq` of a `half` or `float` output of a module **with** fast math (or naming
    /// no math mode) into a 32-bit float attachment of format `pf` (`R32Float`, `RG32Float`,
    /// `RGBA32Float`) storing the channels `written` of four (a channel the format lacks counts as
    /// written: the store drops it), in `float` arithmetic throughout, computed by
    /// [`blend_float32`]. `R32Float` and `RG32Float` have no alpha: their destination alpha is
    /// one, so `DestinationAlpha` is a factor of one and `OneMinusDestinationAlpha` of zero.
    ///
    /// The forms are a 16-bit float attachment's fast forms ([`Blend::half_float`],
    /// [`Family::Float32`]) evaluated in `float`: each factor, each product rounded to `float`
    /// nearest even, and, of an added pair the factors' ranks contract, the source's product
    /// rounded to the attachment and the destination's carried into the sum unrounded,
    /// `f(f(Sf · S) + Df · D)` ([`Arith::Contracted`] under [`BlendArith::Float`], one fused
    /// multiply-add on the destination term), where a device's own blend is the mirror image. The
    /// states whose forms were measured word for word are [`MEASURED_THIRTY_TWO`]; the rest hold
    /// only where [`Blend::fitted`] says so without one.
    ///
    /// A zero factor drops its term, as it does under a 16-bit float attachment's fast math: the
    /// four zero-factor states of those cases (`one-zero`, `zero-1msa`, `dc-zero`, `1mbc-zero`)
    /// are Metal's at every word here, but their sweep draws no NaN, infinite or denormal source,
    /// so the sign and NaN of a dropped term rest on the 16-bit family's measurement of the same
    /// fold ([`Blend::half_float`]'s probes) and not on this width's.
    ///
    /// Only fast math: a module without fast math (`air.compile.fast_math_disable`) is not
    /// planned here, Metal's arithmetic for a 32-bit float attachment there having been measured
    /// in no state, and its blend stays the generic one ([`Blend::generic`], whose rounded forms
    /// are a fixed-function blend's) with its note. A factor of the second source is no form here
    /// (`Feature::ExtendedRangeBlend` `DualSource`): its output is not planned beside the storage
    /// form's write.
    pub fn float32(
        eq: &BlendEquation,
        pf: PixelFormat,
        written: ColorWriteMask,
    ) -> Result<Blend, Error> {
        let channels = match pf {
            PixelFormat::R32Float => 1,
            PixelFormat::RG32Float => 2,
            _ => 4,
        };
        let b = planned_for(
            eq,
            (channels == 4, Family::Float32),
            written,
            (false, BlendMath::Fast),
        )?;
        let b = Blend {
            // A one-channel attachment's `SourceColor` over `DestinationColor` is Metal's in the
            // other order, so no state of one channel is measured ([`MEASURED_THIRTY_TWO`]).
            measured: channels > 1 && !b.masked && measured_state(eq, &MEASURED_THIRTY_TWO),
            wide: true,
            ..b
        };
        match b.reads_second() {
            true => Err(refused(ExtendedRangeBlend::DualSource)),
            false => Ok(b),
        }
    }

    /// The generic blend `eq` into a format whose Metal blend is not built (see
    /// `lower::fragment::fetch`'s module docs), of a format with an alpha channel where
    /// `has_alpha` (else the destination alpha is one), storing the channels `written` of four (a
    /// channel the format lacks counts as written), computed by [`blend_generic`]: in `float`, each
    /// factor, product and sum rounded to `float`, nearest even, a factor of zero keeping its
    /// product as IEEE 754 adds, the forms of [`BlendMath::Rounded`] whatever the module's math. A
    /// factor of the second source reads the function's second (index 1) output.
    pub fn generic(
        eq: &BlendEquation,
        has_alpha: bool,
        written: ColorWriteMask,
    ) -> Result<Blend, Error> {
        planned_for(
            eq,
            (has_alpha, Family::HalfFloat),
            written,
            (false, BlendMath::Rounded),
        )
    }

    /// Metal's blend `eq` of a four-lane output of a module of math `math`, blended in `arith`
    /// (the output's type), into an 8-bit unorm attachment of format `pf` (`R8Unorm`, `RG8Unorm`,
    /// `RGBA8Unorm`, `BGRA8Unorm`) storing the channels `written` of four (a channel the format
    /// lacks counts as written), computed by [`blend_unorm8`] over the output clamped to `[0, 1]`
    /// and the destination's code decoded in that type, the result stored by the device's
    /// conversion.
    /// Without fast math the rounded forms (the module docs). Under fast math the 16-bit float
    /// blend's forms, but as Metal's compiler builds an 8-bit unorm attachment's blend (plain Metal
    /// on the M3 Max under macOS 27.0.1, `lower::fragment::fetch`'s module docs):
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
    /// A factor of the second source is not built (`Feature::FramebufferFetch` `NotImplemented`):
    /// its output is not planned beside the storage form's write. Nor, under fast math, is a
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
            PixelFormat::RGBA8Unorm | PixelFormat::BGRA8Unorm => true,
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
        let family = Family::Unorm8(arith);
        let b = planned_for(eq, (has_alpha, family), written, (false, math))?;
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

    /// Metal's blend `eq` of a four-lane output of a module of math `math`, blended in `arith`
    /// (the output's type), into an 8-bit sRGB attachment of format `pf` (`R8Unorm_sRGB`,
    /// `RG8Unorm_sRGB`, `RGBA8Unorm_sRGB`, `BGRA8Unorm_sRGB`) storing the channels `written` of
    /// four, computed by [`blend_srgb8`]: the 8-bit unorm blend of the format's unorm twin
    /// ([`Blend::unorm8`]) over the destination as a fetch reads it, Metal's twelve-bit decode of
    /// each color byte and alpha's byte over 255, the result stored by Metal's sRGB encode. Plain
    /// Metal on the M3 Max under macOS 27.0.1 (our own MSL, random sources and texel bytes, `half4`
    /// and `float4` outputs, fast and safe math, each of the four formats): every channel of
    /// premultiplied and source-alpha over, the blend color's lerps, a blue-only and an
    /// alpha-only write mask, a sum, and a minimum and maximum is that arithmetic's. Where the
    /// 8-bit unorm blend is not built (a factor of the second source; under fast math, a format
    /// without alpha's lerp, common factor or contracted pair), the rounded forms
    /// ([`BlendMath::Rounded`]): Metal's without fast math (a second source's and the formats
    /// without alpha's measured so), a near-equivalent under it. The flag is whether the blend is
    /// Metal's: the 8-bit unorm blend, or the rounded forms without fast math.
    pub fn srgb8(
        eq: &BlendEquation,
        pf: PixelFormat,
        written: ColorWriteMask,
        (math, arith): (BlendMath, BlendArith),
    ) -> Result<(Blend, bool), Error> {
        let (twin, has_alpha) = match pf {
            PixelFormat::R8UnormSRGB => (PixelFormat::R8Unorm, false),
            PixelFormat::RG8UnormSRGB => (PixelFormat::RG8Unorm, false),
            PixelFormat::RGBA8UnormSRGB => (PixelFormat::RGBA8Unorm, true),
            PixelFormat::BGRA8UnormSRGB => (PixelFormat::BGRA8Unorm, true),
            _ => return Err(Unsupported::not_implemented(Feature::FramebufferFetch).into()),
        };
        if let Ok(b) = Blend::unorm8(eq, twin, written, (math, arith)) {
            return Ok((b, true));
        }
        let family = Family::Unorm8(arith);
        let rounded = BlendMath::Rounded;
        let b = planned_for(eq, (has_alpha, family), written, (false, rounded))?;
        Ok((b, math == rounded))
    }

    /// Metal's blend `eq` of a four-lane output of a module of math `math` into a packed 16-bit
    /// unorm attachment (`B5G6R5Unorm`, `A1BGR5Unorm`, `ABGR4Unorm`, `BGR5A1Unorm`) storing the
    /// channels `written` of four, computed by [`blend_packed16`] in `half` arithmetic whatever the
    /// output's type, over the output clamped to `[0, 1]` and rounded to `half`, and the
    /// destination's code decoded to `half` (alpha one in a format without it), the result stored
    /// by Metal's write of the format's codes. Without fast math the 8-bit unorm blend's rounded
    /// forms in `half` ([`Blend::unorm8`]); under fast math the 16-bit float blend's forms of four
    /// channels ([`Blend::half_float`]), and a write mask of one color channel the 8-bit unorm
    /// blend's scalar plan. Measured on plain Metal on the M3 Max under macOS 27.0.1, our own MSL:
    /// 2,891 states (every factor and operation pair for color beside source-over alpha, the same
    /// pair for both, and random states under random write masks, one blend color) over 8,192
    /// random pixels of `half4` and `float4` outputs into each of the four formats, under fast and
    /// safe math. Without fast math every channel holds Metal's code; under fast math every channel
    /// lies within one code of Metal's, and 3,527 of 94,732,288 channels miss it (`B5G6R5Unorm`;
    /// at most 1,232 in each other format): Metal regroups some sums otherwise than either form.
    /// A factor of the second source is not built (`Feature::FramebufferFetch` `NotImplemented`).
    pub fn packed16(
        eq: &BlendEquation,
        written: ColorWriteMask,
        math: BlendMath,
    ) -> Result<Blend, Error> {
        use BlendFactor as F;
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
        let unorm8 = (true, Family::Unorm8(BlendArith::Half));
        match math {
            BlendMath::Rounded => planned_for(eq, unorm8, written, (false, math)),
            BlendMath::Fast => planned_for(eq, (true, Family::HalfFloat), written, (false, math))
                .or_else(|_| planned_for(eq, unorm8, written, (false, math))),
        }
    }

    /// Whether every class the blend stores takes a form the 16-bit float sweeps fit in every state
    /// ([`Blend::half_float`]): the rounded forms without fast math; under fast math no term, one
    /// term, one fused multiply-add, a minimum or maximum, or a blend constant's difference
    /// ([`constant_difference`], a 16-bit float attachment's alone), and every form of a state
    /// measured word for word against Metal's stored codes ([`MEASURED_SIXTEEN`],
    /// [`MEASURED_THIRTY_TWO`]). A fast-math blend of another form (a sum regrouped by a lerp or a
    /// common factor, or another product rounded beside a fused one) fits most states but not every
    /// one, so it is a near-equivalent.
    pub fn fitted(&self) -> bool {
        let fits = |a: &Arith| match self.math {
            BlendMath::Rounded => true,
            BlendMath::Fast => {
                self.measured
                    || matches!(
                        a,
                        Arith::Zero
                            | Arith::Term { .. }
                            | Arith::Fused { .. }
                            | Arith::MinMax { .. }
                    )
                    || (!self.wide && constant_difference(a))
            }
        };
        [&self.color, &self.alpha].into_iter().flatten().all(fits)
    }

    /// Whether the blend of an output of four lanes of `output` (`None` for another output) is
    /// Metal's model's: a `half4` one; a `float4` one under a blend whose color lanes take one term
    /// alone, unmasked, rounded to `half` first; another is the gap it is.
    pub fn takes(&self, output: Option<FloatFmt>) -> Result<(), Error> {
        match output {
            Some(FloatFmt::Half) => Ok(()),
            Some(FloatFmt::F32) if self.narrow && self.takes_float && !self.masked => Ok(()),
            Some(FloatFmt::F32) => Err(refused(ExtendedRangeBlend::FloatOutput)),
            _ => Err(refused(ExtendedRangeBlend::Output)),
        }
    }

    /// Whether the blend reads the blend color (`PushField::BlendColor`).
    pub fn reads_constant(&self) -> bool {
        [&self.color, &self.alpha]
            .into_iter()
            .flatten()
            .any(|a| nodes(a).into_iter().any(reads_constant))
    }

    /// Whether the color channels' arithmetic reads the source alpha as a factor (`SourceAlpha` or
    /// `OneMinusSourceAlpha`; `SourceAlphaSaturated` reads it beside one minus the destination
    /// alpha).
    pub fn color_reads_source_alpha(&self) -> bool {
        self.color.as_ref().is_some_and(|a| {
            nodes(a)
                .into_iter()
                .any(|n| matches!(n, Node::Atom(Atom::Sa) | Node::OneMinus(Atom::Sa)))
        })
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

    /// An upper estimate of the SPIR-V words of [`blend`]: the source's clamp and the lanes'
    /// combination, the blend color's clamp, and each class's node reads, roundings to `half`,
    /// and exact sums.
    pub fn words(&self) -> u32 {
        const BASE: u32 = 60 * 6;
        const NODE: u32 = 6 * 6;
        const ROUND: u32 = 48 * 6;
        const SUM: u32 = 36 * 6;
        // A `float4` output's: the source's clamp and its rounding to `half`, and per class an
        // exact `float` multiply-add and a rounding more.
        const FLOAT_BASE: u32 = 12 * 6 + ROUND;
        const FLOAT_FMA: u32 = 92 * 6 + ROUND;
        let mut words = match self.float {
            true => BASE + FLOAT_BASE,
            false => BASE,
        };
        // The second source's clamp, as the source's.
        if self.reads_second() {
            words = words.saturating_add(BASE / 2);
        }
        const CONSTANT: u32 = 18 * 6;
        if self.reads_constant() {
            words = words.saturating_add(CONSTANT);
        }
        for a in [&self.color, &self.alpha].into_iter().flatten() {
            let (rounds, sums) = match a {
                Arith::MinMax { .. } | Arith::Zero => (1, 0),
                Arith::Term { .. } | Arith::Common { .. } => (1, 0),
                Arith::Fused { .. } | Arith::Lerp { .. } => (1, 1),
                Arith::Contracted { .. } | Arith::Kept { .. } => (2, 1),
                Arith::Rounded { .. } => (3, 1),
            };
            if self.float {
                words = words.saturating_add(FLOAT_FMA);
            }
            let inner = match a {
                Arith::Lerp { inner, .. } | Arith::Common { inner, .. } => {
                    inner.len().saturating_sub(1) as u32
                }
                _ => 0,
            };
            let ns = nodes(a);
            // A one-minus or saturated node rounds once.
            let node_rounds = ns
                .iter()
                .filter(|n| matches!(n, Node::OneMinus(_) | Node::Saturate))
                .count() as u32;
            words = words
                .saturating_add(NODE.saturating_mul(ns.len() as u32))
                .saturating_add(ROUND.saturating_mul(rounds + inner + node_rounds))
                .saturating_add(SUM.saturating_mul(sums + inner));
        }
        words
    }
}

/// Whether `a` is a difference of a blend constant's product and its one minus's (`BlendColor` with
/// `OneMinusBlendColor`, `BlendAlpha` with `OneMinusBlendAlpha`, subtracted either way): the
/// minuend's product fused, the subtrahend's rounded first ([`Arith::Contracted`] of terms of
/// opposite signs), which a 16-bit float attachment's fast-math sweeps fit in every state
/// ([`Blend::fitted`]; added, the same products are no such form).
fn constant_difference(a: &Arith) -> bool {
    let Arith::Contracted { rounded, fused } = a else {
        return false;
    };
    let factors = [rounded.1 .1, fused.1 .1];
    rounded.0 != fused.0
        && [Atom::C, Atom::Ca]
            .into_iter()
            .any(|x| factors.contains(&Node::Atom(x)) && factors.contains(&Node::OneMinus(x)))
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
        Arith::Kept { kept, rounded } => [term(&kept.1), term(&rounded.1)].concat(),
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

/// The `vec4`s a blend reads: the clamped source, the tile value, and the blend color's two
/// rows ([`super::blend_color::constant_words`]) where it reads them; and a `float4` output's
/// source unrounded, clamped ([`Atom::Sf`]); and the function's math; and the second source,
/// clamped, where the blend reads it ([`Atom::S1`]).
struct Inputs {
    s: Value,
    d: Value,
    /// The blend color's first row clamped to the form's range, the second row, and the first as
    /// `BlendColor` reads it (`BGRA10_XR_sRGB`'s past its bound).
    constant: Option<(Value, Value, Value)>,
    sf: Option<Value>,
    math: BlendMath,
    s1: Option<Value>,
    /// The arithmetic every rounding is to: `half` but a 16-bit float attachment's blend of a
    /// `float4` output ([`BlendArith`]).
    arith: BlendArith,
    /// [`Blend`]'s `negates_one_minus`.
    negates_one_minus: bool,
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
            Atom::Sf => {
                let sf = i.sf.ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
                self.own(fb, sf)
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

    /// Whether Metal computes a destination's product subtracted alone by `1 - z` as `d · (z - 1)`
    /// ([`Blend::half_float`]): into a 16-bit float attachment of two or more channels under fast
    /// math, where `z` is a source value; in alpha of a `float4` output, any `z`.
    fn negates_one_minus(&self, z: Atom) -> bool {
        let float_alpha = self.lanes == Lanes::Scalar && self.inputs.arith == BlendArith::Float;
        self.inputs.negates_one_minus && (matches!(z, Atom::S | Atom::Sf | Atom::Sa) || float_alpha)
    }

    /// `x - 1`, rounded as [`one_minus`](Self::one_minus) rounds `1 - x`: `+0` where `x` is one.
    fn minus_one(&mut self, fb: &mut FunctionBuilder<'_>, x: Value) -> BResult<Value> {
        let f = self.lanes.floats();
        let one = fsplat(fb, f, 1.0)?;
        if self.inputs.arith == BlendArith::Float {
            return fb.binary_no_contraction(BinOp::FSub, x, one);
        }
        let m = fb.binary(BinOp::FSub, x, one)?;
        half_of(fb, f, m)
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

    /// `h(a · b + c)`: of `half` values exactly ([`half_sum`]); where the product or the addend
    /// (`raw`) reads an unrounded source ([`Atom::Sf`]), the exact value rounded to `float` first,
    /// as Metal's `float` arithmetic rounds it ([`float_fma_exact`]; a product of `half`s is
    /// exact in `float`, so a `float` addend's sum is one addition).
    fn fma(
        &self,
        fb: &mut FunctionBuilder<'_>,
        (raw_product, raw_addend): (bool, bool),
        [a, b, c]: [Value; 3],
    ) -> BResult<Value> {
        let f = self.lanes.floats();
        if self.inputs.arith == BlendArith::Float {
            return float_fma_exact(fb, f, a, b, c);
        }
        let s = match (raw_product, raw_addend) {
            (true, _) => float_fma_exact(fb, f, a, b, c)?,
            (false, true) => {
                let p = fb.binary(BinOp::FMul, a, b)?;
                fb.binary(BinOp::FAdd, p, c)?
            }
            (false, false) => half_sum(fb, f.with(Held::Float(FloatFmt::Half)), a, b, c)?,
        };
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
    /// rounded to `half` (through `float` where it adds an unrounded source); and whether it is an
    /// unrounded `float` (one such summand).
    fn inner(&mut self, fb: &mut FunctionBuilder<'_>, inner: &[Summand]) -> BResult<(Value, bool)> {
        let f = self.lanes.floats();
        let mut acc: Option<(Value, bool)> = None;
        for &(c, k) in inner {
            let (x, raw) = match k {
                None => (fsplat(fb, f, f64::from(c))?, false),
                Some(n) => {
                    let v = self.node(fb, n)?;
                    (self.scaled(fb, c, v)?, n == Node::Atom(Atom::Sf))
                }
            };
            acc = Some(match acc {
                None => (x, raw),
                Some((a, raw_a)) => {
                    let one = fsplat(fb, f, 1.0)?;
                    (self.fma(fb, (false, raw_a || raw), [a, one, x])?, false)
                }
            });
        }
        match acc {
            Some(v) => Ok(v),
            None => Ok((fsplat(fb, f, 0.0)?, false)),
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
                let m = fb.glsl(op, &[s, d])?;
                match source {
                    Atom::Sf => self.round(fb, m),
                    _ => Ok(m),
                }
            }
            Arith::Zero => fsplat(fb, self.lanes.floats(), 0.0),
            Arith::Term {
                sign: -1,
                term: (operand @ (Atom::D | Atom::Da), Node::OneMinus(z)),
            } if self.negates_one_minus(*z) => {
                // `h(d · h(z - 1))`: where `z` is one, the zero of `d`'s sign.
                let x = self.atom(fb, *operand)?;
                let z = self.atom(fb, *z)?;
                let f = self.minus_one(fb, z)?;
                let p = self.mul(fb, x, f)?;
                self.round(fb, p)
            }
            Arith::Term { sign, term } => self.product(fb, (*sign, *term)),
            Arith::Fused { product, addend } => {
                let [x, f] = self.signed(fb, *product)?;
                let a = self.atom(fb, addend.1 .0)?;
                let a = self.scaled(fb, addend.0, a)?;
                let raw = (raw_term(&product.1), addend.1 .0 == Atom::Sf);
                self.fma(fb, raw, [x, f, a])
            }
            Arith::Lerp { y, z, inner } => {
                let (i, raw_inner) = self.inner(fb, inner)?;
                let raw = (
                    *z == Node::Atom(Atom::Sf) || raw_inner,
                    *y == Node::Atom(Atom::Sf),
                );
                let y = self.node(fb, *y)?;
                let z = self.node(fb, *z)?;
                self.fma(fb, raw, [z, i, y])
            }
            Arith::Common {
                z,
                inner,
                round_inner,
            } => {
                let (i, _) = self.inner(fb, inner)?;
                let i = match round_inner {
                    true => self.round(fb, i)?,
                    false => i,
                };
                let z = self.node(fb, *z)?;
                // A product of two `half`s is exact in `f32`, one of an unrounded source rounded
                // to `float` as Metal's is.
                let p = self.mul(fb, z, i)?;
                self.round(fb, p)
            }
            Arith::Contracted { rounded, fused } => {
                let [x, f] = self.signed(fb, *rounded)?;
                let p = self.mul(fb, x, f)?;
                let r = self.round(fb, p)?;
                let [x, f] = self.signed(fb, *fused)?;
                self.fma(fb, (raw_term(&fused.1), false), [x, f, r])
            }
            Arith::Kept { kept, rounded } => {
                let [x, f] = self.signed(fb, *rounded)?;
                let p = fb.binary(BinOp::FMul, x, f)?;
                let r = self.round(fb, p)?;
                let [x, f] = self.signed(fb, *kept)?;
                // The kept product rounded to `float` on its own, never fused into the sum.
                let k = fb.binary_no_contraction(BinOp::FMul, x, f)?;
                let s = fb.binary(BinOp::FAdd, k, r)?;
                self.round(fb, s)
            }
            Arith::Rounded {
                source,
                destination,
            } if source.1 .1 == Node::Zero || destination.1 .1 == Node::Zero => {
                self.zero_kept(fb, *source, *destination)
            }
            Arith::Rounded {
                source,
                destination,
            } => {
                let s = self.product(fb, *source)?;
                let d = self.product(fb, *destination)?;
                let one = fsplat(fb, self.lanes.floats(), 1.0)?;
                self.fma(fb, (false, false), [s, one, d])
            }
        }
    }

    /// `h(sign · operand · factor)` of `(sign, term)`: the operand alone where the factor is one
    /// (a `half`, but an unrounded source rounded); a product of two `half`s is exact in `f32`,
    /// one of an unrounded source rounded to `float` as Metal's is.
    fn product(&mut self, fb: &mut FunctionBuilder<'_>, (sign, t): (i8, Term)) -> BResult<Value> {
        let [x, f] = self.signed(fb, (sign, t))?;
        match t.1 {
            Node::One if t.0 == Atom::Sf => self.round(fb, x),
            Node::One => Ok(x),
            _ => {
                let p = self.mul(fb, x, f)?;
                self.round(fb, p)
            }
        }
    }

    /// The bits of `sign · operand · 0` of `(sign, term)`, a term of factor zero, and whether it is
    /// a NaN: the operand an infinity or a NaN, whose product with zero is a NaN; else the zero of
    /// the signed operand's sign. Decided on the bits: a device may fold a product by a constant
    /// zero to `+0` (a module without `SignedZeroInfNanPreserve`, MoltenVK's fast math), which
    /// keeps neither the NaN nor the sign.
    fn zero_product(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        (sign, t): (i8, Term),
    ) -> BResult<(Value, Value)> {
        let f = self.lanes.floats();
        let u = crate::lower::arith::narrow::u32_form(f);
        let ut = ty(fb, u)?;
        let x = self.atom(fb, t.0)?;
        let xb = fb.convert(ConvOp::Bitcast, x, ut)?;
        let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
        let infinity = splat(fb, u, 0x7F80_0000)?;
        let sign_bit = splat(fb, u, 0x8000_0000)?;
        let m = fb.binary(BinOp::BitwiseAnd, xb, magnitude)?;
        let nan = fb.binary(BinOp::UGreaterThanEqual, m, infinity)?;
        let z = fb.binary(BinOp::BitwiseAnd, xb, sign_bit)?;
        let z = match sign < 0 {
            true => fb.binary(BinOp::BitwiseXor, z, sign_bit)?,
            false => z,
        };
        Ok((z, nan))
    }

    /// The bits of term `t`: a zero factor's product and whether it is a NaN
    /// ([`zero_product`](Self::zero_product)), or the term rounded as
    /// [`product`](Self::product) rounds it and `None`.
    fn zero_or_product(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        t: (i8, Term),
    ) -> BResult<(Value, Option<Value>)> {
        if t.1 .1 == Node::Zero {
            let (z, nan) = self.zero_product(fb, t)?;
            return Ok((z, Some(nan)));
        }
        let u = crate::lower::arith::narrow::u32_form(self.lanes.floats());
        let ut = ty(fb, u)?;
        let p = self.product(fb, t)?;
        Ok((fb.convert(ConvOp::Bitcast, p, ut)?, None))
    }

    /// The sum of `a` and `b` where a factor of either is zero ([`Arith::Rounded`] of a 16-bit
    /// float attachment's blend, [`plan_rounded`]'s `keep_zero`): a NaN where a zero factor's
    /// product is one (its operand an infinity or a NaN) or the other term is one, the quiet NaN
    /// of positive sign Metal stores (`0x7e00`) whatever the NaN's sign and payload; else the
    /// other term, rounded as [`product`](Self::product) rounds it, but where that is a zero, the
    /// zero IEEE 754 gives the sum: `-0` only where both are `-0`. Two zero factors sum their
    /// zeros alike. The sum of a zero and a number is the number, so nothing rounds again. On the
    /// bits, as [`zero_product`](Self::zero_product), and the NaN on a select, which no float mode
    /// changes.
    fn zero_kept(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        a: (i8, Term),
        b: (i8, Term),
    ) -> BResult<Value> {
        let f = self.lanes.floats();
        let u = crate::lower::arith::narrow::u32_form(f);
        let ft = ty(fb, f)?;
        let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
        let zero = splat(fb, u, 0)?;
        let (za, na) = self.zero_or_product(fb, a)?;
        let (zb, nb) = self.zero_or_product(fb, b)?;
        // Each zero factor's product is a signed zero (its magnitude bits clear): `x & z` is the
        // zero both signs share where `x` is a zero too, so a zero other takes it and a number
        // stays itself.
        let (sum, nan) = match (na, nb) {
            (Some(na), Some(nb)) => (
                fb.binary(BinOp::BitwiseAnd, za, zb)?,
                fb.binary(BinOp::LogicalOr, na, nb)?,
            ),
            (Some(n), None) | (None, Some(n)) => {
                let (z, other) = match na.is_some() {
                    true => (za, zb),
                    false => (zb, za),
                };
                let shared = fb.binary(BinOp::BitwiseAnd, other, z)?;
                let m = fb.binary(BinOp::BitwiseAnd, other, magnitude)?;
                let is_zero = fb.binary(BinOp::IEqual, m, zero)?;
                let infinity = splat(fb, u, 0x7F80_0000)?;
                let other_nan = fb.binary(BinOp::UGreaterThan, m, infinity)?;
                let nan = fb.binary(BinOp::LogicalOr, n, other_nan)?;
                (fb.select(is_zero, shared, other)?, nan)
            }
            (None, None) => return Err(BuildError::Type(TypeRule::FunctionSignature)),
        };
        let quiet = splat(fb, u, 0x7FC0_0000)?;
        let w = fb.select(nan, quiet, sum)?;
        fb.convert(ConvOp::Bitcast, w, ft)
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
/// end. A read of the `half` shadow clamps so too (`super::tile_read`).
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

/// Metal's blend `b` of output `s` (a `vec4` of `f32`, red first: a `half4` output's, or a
/// `float4` one's, rounded to `half`, nearest even, first but under a `float4` output's own model,
/// [`Blend::float`]) over `d`, the attachment's tile value
/// ([`super::tile_value`]), into an attachment of `form`: a `vec4` of `f32` holding each blended
/// lane, a `half` value, which [`super::encode`] stores (see the module docs). `constant` holds
/// `PushField::BlendColor`'s two `uvec4`s ([`super::blend_color::constant_words`]) where `b`
/// reads the blend color ([`Blend::reads_constant`]), the color clamped to the form's source range
/// as its source is. `s1` holds the second source, as `s` a `half`'s, where `b` reads it
/// ([`Blend::reads_second`]). A lane the write does not store holds `d`'s.
pub fn blend(
    fb: &mut FunctionBuilder<'_>,
    form: ExtendedRange,
    b: &Blend,
    (s, s1, d): (Value, Option<Value>, Value),
    constant: Option<[Value; 2]>,
) -> BResult<Value> {
    let f = floats();
    // A `float4` output's color lanes read it unrounded, clamped to its own range, and every
    // other read is of its `half`.
    let (sf, s) = match b.float {
        true => (
            Some(clamped(fb, s, FLOAT_SOURCE_RANGE)?),
            half_of(fb, f, s)?,
        ),
        false => (None, s),
    };
    let s = clamped(fb, s, source_range(form))?;
    let s1 = match s1 {
        Some(s1) => Some(clamped(fb, s1, source_range(form))?),
        None => None,
    };
    let constant = match constant {
        Some([c, m]) => {
            let ft = ty(fb, f)?;
            let c = fb.convert(ConvOp::Bitcast, c, ft)?;
            // The binder clamps the color to the forms' hull; each form clamps it to its own range.
            let clamped_c = clamped(fb, c, source_range(form))?;
            let m = fb.convert(ConvOp::Bitcast, m, ft)?;
            // `BGRA10_XR_sRGB`'s `BlendColor` reads a color channel at or past `SRGB_DIRECT_BOUND`,
            // which the binder marks past the hull's top, as `SRGB_DIRECT_TOP`; its one minus reads
            // the clamped value as every form's does.
            let direct = match form {
                ExtendedRange::Bgra10 { srgb: true } => {
                    let top = fsplat(fb, f, f64::from(SRGB_SOURCE_RANGE.1))?;
                    let past = fb.binary(BinOp::FOrdGreaterThan, c, top)?;
                    let read = fsplat(fb, f, f64::from(SRGB_DIRECT_TOP))?;
                    fb.select(past, read, clamped_c)?
                }
                _ => clamped_c,
            };
            Some((clamped_c, m, direct))
        }
        None => None,
    };
    let inputs = Inputs {
        s,
        d,
        constant,
        sf,
        math: b.math,
        s1,
        arith: BlendArith::Half,
        negates_one_minus: b.negates_one_minus,
    };
    lanes(fb, b, &inputs)
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

/// Metal's blend `b` ([`Blend::half_float`]) of a four-lane output `s` (a `vec4` of `f32`: a
/// `half4` output's `half`s, or a `float4` one) over `d`, the 16-bit float attachment's texel as
/// the storage image reads it (a `vec4` of `f32`; a channel the format lacks reads 0, alpha 1), in
/// `arith`: a `vec4` of `f32` holding each blended lane, a `half` (`BlendArith::Half`) or a `float`
/// (`BlendArith::Float`, which the caller converts to `half` by Metal's write). Nothing is clamped:
/// the source is the value the function returns, and `color`, the encoder's blend color (a `uvec4`
/// of its `f32`s' bits, `PushField::HalfBlendColor`, where `b` reads it,
/// [`Blend::reads_constant`]), is what `BlendColor` reads, rounded to `half`, nearest even, in
/// `half` arithmetic (the one fit of `RGBA16Float`'s blue lane under a blend color of 0.9), and
/// `OneMinusBlendColor` its one minus: without fast math of that read, under fast math `h(1 - c)`
/// of a color channel's `f32` and `h(1 - h(a))` of alpha, but of that read beside `BlendColor` in a
/// color class. The `Min` and `Max` operations are the device's `FMin` and `FMax`, whose answer at
/// a NaN is not Metal's measured one; `SourceAlphaSaturated`'s minimum is Metal's ([`metal_min`]).
/// A `float` multiply-add is correctly rounded where no part of it underflows
/// (`fma::float_fma_exact`).
pub fn blend_half_float(
    fb: &mut FunctionBuilder<'_>,
    b: &Blend,
    arith: BlendArith,
    (s, d): (Value, Value),
    color: Option<Value>,
) -> BResult<Value> {
    let f = floats();
    let constant = match color {
        Some(words) => {
            let ft = ty(fb, f)?;
            let c = fb.convert(ConvOp::Bitcast, words, ft)?;
            let one = fsplat(fb, f, 1.0)?;
            match arith {
                BlendArith::Half => {
                    let h = half_of(fb, f, c)?;
                    // Under fast math the color channels' one minus reads the `f32`, alpha's
                    // its `half` (a color class beside `BlendColor` computes its own of the
                    // `half`).
                    let minus_of = fb.vector_shuffle(c, h, &[0, 1, 2, 7])?;
                    let m = fb.binary(BinOp::FSub, one, minus_of)?;
                    let m = half_of(fb, f, m)?;
                    Some((h, m, h))
                }
                BlendArith::Float => {
                    let m = fb.binary_no_contraction(BinOp::FSub, one, c)?;
                    Some((c, m, c))
                }
            }
        }
        None => None,
    };
    let inputs = Inputs {
        s,
        d,
        constant,
        sf: None,
        math: b.math,
        s1: None,
        arith,
        negates_one_minus: b.negates_one_minus,
    };
    lanes(fb, b, &inputs)
}

/// Metal's blend `b` ([`Blend::float32`]) of a four-lane output `s` (a `vec4` of `f32`: a `half4`
/// output's `half`s, or a `float4` one's) over `d`, the 32-bit float attachment's texel as the
/// storage image reads it (a `vec4` of `f32`; a channel the format lacks reads 0, alpha 1), in
/// `float` arithmetic: a `vec4` of `f32` holding each blended lane, the `float` the attachment
/// stores. Nothing is clamped, as nothing is for a 16-bit float attachment: the source is the
/// value the function returns.
///
/// The same arithmetic as [`blend_half_float`] in [`BlendArith::Float`], which is what a 32-bit
/// float attachment's blend is: every rounding of the forms is to `float`, so `color`, the
/// encoder's blend color (a `uvec4` of its `f32`s' bits, `PushField::HalfBlendColor`, where `b`
/// reads it, [`Blend::reads_constant`]), is what `BlendColor` reads unrounded and
/// `OneMinusBlendColor` its `f(1 - c)`. A `float` multiply-add is correctly rounded where no part
/// of it underflows (`fma::float_fma_exact`).
pub fn blend_float32(
    fb: &mut FunctionBuilder<'_>,
    b: &Blend,
    (s, d): (Value, Value),
    color: Option<Value>,
) -> BResult<Value> {
    blend_half_float(fb, b, BlendArith::Float, (s, d), color)
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

/// Metal's blend `b` ([`Blend::srgb8`]) of a four-lane output `s` (a `vec4` of `f32`: a `half4`
/// output's `half`s, or a `float4` one) over `decoded`, the 8-bit sRGB attachment's texel as a
/// fetch reads it (a `vec4` of `f32`: each color byte's twelve-bit step over 4095 and the alpha
/// byte over 255, correctly rounded, `math::unpack` of `unorm4x8_srgb`; a channel the format lacks
/// 0, alpha 1), in `arith`: a `vec4` of `f32` holding each blended lane, a `half`
/// (`BlendArith::Half`) or a `float` (`BlendArith::Float`), which the caller stores by Metal's
/// sRGB encode. The output, and `s1`, the second source (the function's second output, where `b`
/// reads it, [`Blend::reads_second`]), are clamped to `[0, 1]` first, a NaN to 0, and the
/// destination read rounded to the arithmetic's type: each `f32` the decode gives rounds to the
/// `half` its exact quotient does (no step over 4095 and no byte over 255 lies within an `f32`
/// rounding of a `half` midpoint; every one is checked,
/// `lower::arith::tests::blend8::an_srgb_step_s_half_is_its_quotient_s`). `color`, the blend
/// color, reads as [`blend_unorm8`]'s.
pub fn blend_srgb8(
    fb: &mut FunctionBuilder<'_>,
    b: &Blend,
    arith: BlendArith,
    (s, s1, decoded): (Value, Option<Value>, Value),
    color: Option<Value>,
) -> BResult<Value> {
    let f = floats();
    let s = clamped(fb, s, (0.0, 1.0))?;
    let s1 = match s1 {
        Some(v) => Some(clamped(fb, v, (0.0, 1.0))?),
        None => None,
    };
    let d = match arith {
        BlendArith::Half => half_of(fb, f, decoded)?,
        BlendArith::Float => decoded,
    };
    clamped_blend(fb, b, arith, (s, s1, d), color)
}

/// The generic blend `b` ([`Blend::generic`]) of a four-lane output `s` (a `vec4` of `f32`) over
/// `d`, the attachment's texel as the storage image reads it (a `vec4` of `f32`; a channel the
/// format lacks reads 0, alpha 1), in `float`: a `vec4` of `f32` holding each blended lane, which
/// the caller stores through the device's conversion. Where the format is normalized, the source
/// and the blend color are clamped to its `range` first (`[0, 1]` unorm, `[-1, 1]` snorm, every
/// lane, a NaN to the low end), as a fixed-function blend clamps them; `color`, the encoder's blend
/// color (a `uvec4` of its `f32`s' bits, `PushField::HalfBlendColor`, where `b` reads it,
/// [`Blend::reads_constant`]), is what `BlendColor` reads, and `OneMinusBlendColor` reads one minus
/// it; `s1`, the second source (a `vec4` of `f32`, the function's second output, where `b` reads
/// it, [`Blend::reads_second`]), is clamped as `s`. Not Metal's arithmetic for the format, which is
/// not built: a near-equivalent.
pub fn blend_generic(
    fb: &mut FunctionBuilder<'_>,
    b: &Blend,
    range: Option<(f32, f32)>,
    (s, s1, d): (Value, Option<Value>, Value),
    color: Option<Value>,
) -> BResult<Value> {
    let f = floats();
    let clamp = |fb: &mut FunctionBuilder<'_>, v: Value| -> BResult<Value> {
        let Some((lo, hi)) = range else {
            return Ok(v);
        };
        let low = per_lane_f(fb, [lo; 4])?;
        let high = per_lane_f(fb, [hi; 4])?;
        let nan = fb.unary(UnOp::IsNan, v)?;
        let v = fb.select(nan, low, v)?;
        fb.glsl(GlslOp::FClamp, &[v, low, high])
    };
    let s = clamp(fb, s)?;
    let s1 = match s1 {
        Some(v) => Some(clamp(fb, v)?),
        None => None,
    };
    let constant = match color {
        Some(words) => {
            let ft = ty(fb, f)?;
            let c = fb.convert(ConvOp::Bitcast, words, ft)?;
            let c = clamp(fb, c)?;
            let one = fsplat(fb, f, 1.0)?;
            let m = fb.binary_no_contraction(BinOp::FSub, one, c)?;
            Some((c, m, c))
        }
        None => None,
    };
    let inputs = Inputs {
        s,
        d,
        constant,
        sf: None,
        math: BlendMath::Rounded,
        s1,
        arith: BlendArith::Float,
        negates_one_minus: false,
    };
    lanes(fb, b, &inputs)
}

/// Blend `b` of the clamped output `s` (and second source `s1`, where `b` reads it) over the
/// destination `d`, `vec4`s of `f32` in `arith`, the blend color `color` clamped to `[0, 1]` (a
/// NaN 0) and rounded to `arith`, and its one minus that of the read: [`blend_unorm8`]'s,
/// [`blend_srgb8`]'s and [`blend_packed16`]'s arithmetic.
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
        sf: None,
        math: b.math,
        s1,
        arith,
        negates_one_minus: false,
    };
    lanes(fb, b, &inputs)
}

/// Metal's blend `b` ([`Blend::packed16`]) of a four-lane output `s` (a `vec4` of `f32`: a
/// `half4` output's `half`s, or a `float4` one) over `d`, a packed 16-bit attachment's texel as
/// Metal's blend reads it (a `vec4` of `f32`, each channel its code over its maximum rounded to
/// `half`, a channel the format lacks 0, alpha 1), in `half` arithmetic: a `vec4` of `f32` holding
/// each blended lane, a `half`, which the caller stores as the format's codes. Metal clamps the
/// output to `[0, 1]` first, a NaN to 0, and rounds it to `half`, nearest even, a `float4` one too;
/// `color`, the blend color, reads as [`blend_unorm8`]'s in `half`.
pub fn blend_packed16(
    fb: &mut FunctionBuilder<'_>,
    b: &Blend,
    (s, d): (Value, Value),
    color: Option<Value>,
) -> BResult<Value> {
    let f = floats();
    let s = clamped(fb, s, (0.0, 1.0))?;
    let s = half_of(fb, f, s)?;
    clamped_blend(fb, b, BlendArith::Half, (s, None, d), color)
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
/// for 16 bits, and a driver that keeps the conversion and its widening, not one with
/// `DriverWorkaround::HalfConversionPairFolded`; [`narrow::controlled_modes`]), that conversion
/// and its widening ([`narrow::round`]): two instructions, Metal's `half(float)` by the Vulkan spec
/// (conversions are correctly rounded, to nearest even under `RoundingModeRTE`, denormals kept
/// under `DenormPreserve`, zero signs and infinities under `SignedZeroInfNanPreserve`), as the
/// module's own `half` results round on such a profile (docs/FAST-PATHS.md's `half` rounding row:
/// exact under either policy, no note). Elsewhere the integer sequence ([`f32_to_half`]), about
/// fifty instructions, the same on every device. The two agree on every `x` but a NaN's bits,
/// which the conversion leaves to the device: the blends clamp a NaN source to zero, and a 16-bit
/// float blend stores every NaN as Metal's own.
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
