//! Conversions: `air.convert.<kind>.<type>.<kind>.<type>[.<rounding>[.sat]]` and
//! `air.{pack,unpack}.<format>.<type>`.
//!
//! The rounding (`rte`, `rtz`, `rtp`, `rtn`) and `.sat` are the suffixes Apple's OpenCL compiler
//! spells on every `convert_<type>[_sat][_<mode>]` it emits ([`Convert::rounding`],
//! [`Convert::saturate`]). Whether a conversion honors them is the lowering's: Metal honors them
//! in a compute pipeline of OpenCL mode and answers most of them as the conversion without the
//! suffix elsewhere.

use super::sig::{Operand, Signature};
use super::ty::{Elem, NumTy};

/// How a conversion operand's bits are read: the kind letter before each type token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NumKind {
    /// `f`: floating point.
    Float,
    /// `s`: signed integer.
    Signed,
    /// `u`: unsigned integer, or a boolean (`u.i1`).
    Unsigned,
}

impl NumKind {
    fn parse(token: &str) -> Option<NumKind> {
        Some(match token {
            "f" => NumKind::Float,
            "s" => NumKind::Signed,
            "u" => NumKind::Unsigned,
            _ => return None,
        })
    }

    /// Whether `elem` may be read with this kind. Booleans are spelled only `u.i1`.
    fn admits(self, elem: Elem) -> bool {
        match self {
            NumKind::Float => elem.is_float(),
            NumKind::Signed => elem.is_int(),
            NumKind::Unsigned => elem.is_int() || elem == Elem::I1,
        }
    }
}

/// An 8-bit float format, carried in `i8` lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Fp8 {
    /// `f8e4m3`: 4 exponent bits (bias 7) and 3 mantissa bits, IEEE-like (an all-ones exponent is
    /// infinity or NaN).
    E4M3,
    /// `f8e4m3fn`: 4 exponent bits (bias 7) and 3 mantissa bits, finite (only `S.1111.111` is NaN).
    E4M3Fn,
    /// `f8e5m2`: 5 exponent bits (bias 15) and 2 mantissa bits, IEEE-like.
    E5M2,
}

impl Fp8 {
    /// `[v<N>]f8<format>`: the lane count and format.
    fn parse(token: &str) -> Option<(u32, Fp8)> {
        let (lanes, fmt) = match token.strip_prefix('v') {
            Some(rest) => {
                let n = rest.bytes().take_while(u8::is_ascii_digit).count();
                let (count, fmt) = rest.split_at_checked(n)?;
                if count.starts_with('0') {
                    return None;
                }
                let lanes: u32 = count.parse().ok()?;
                (lanes >= 2).then_some((lanes, fmt))?
            }
            None => (1, token),
        };
        let fmt = match fmt {
            "f8e4m3" => Fp8::E4M3,
            "f8e4m3fn" => Fp8::E4M3Fn,
            "f8e5m2" => Fp8::E5M2,
            _ => return None,
        };
        Some((lanes, fmt))
    }
}

/// The rounding a conversion's name spells after its types (`.rte`, `.rtz`, `.rtp`, `.rtn`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rounding {
    /// `rte`: to nearest, ties to even.
    Even,
    /// `rtz`: toward zero.
    Zero,
    /// `rtp`: toward positive infinity.
    Up,
    /// `rtn`: toward negative infinity.
    Down,
}

impl Rounding {
    fn parse(token: &str) -> Option<Rounding> {
        Some(match token {
            "rte" => Rounding::Even,
            "rtz" => Rounding::Zero,
            "rtp" => Rounding::Up,
            "rtn" => Rounding::Down,
            _ => return None,
        })
    }
}

/// A numeric conversion between equal lane counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Convert {
    /// The destination value type.
    pub dst: NumTy,
    /// How the destination is read: the first kind letter.
    pub dst_kind: NumKind,
    /// The source value type; an 8-bit float source is its `i8` carrier.
    pub src: NumTy,
    /// How the source is read: the second kind letter (`f` for an 8-bit float source).
    pub src_kind: NumKind,
    /// The format of an 8-bit float source (`<f|s|u>.<dst>.f.[v<N>]f8<format>`): to a float
    /// or to an integer other than `i1`. An 8-bit float destination does not decode
    /// ([`super::UNDECODED`]).
    pub src_fp8: Option<Fp8>,
    /// The rounding spelled after the types, between numbers (no `bool`, no 8-bit float).
    pub rounding: Option<Rounding>,
    /// `.sat` after the rounding: to an integer, out-of-range values take the destination's bound.
    pub saturate: bool,
}

impl Convert {
    /// Parse the part of a name after `air.convert.`.
    pub(super) fn parse(rest: &str) -> Option<Convert> {
        let mut t = rest.split('.');
        let (dk, dt, sk, st) = (t.next()?, t.next()?, t.next()?, t.next()?);
        let rounding = match t.next() {
            Some(token) => Some(Rounding::parse(token)?),
            None => None,
        };
        let saturate = match t.next() {
            Some("sat") => true,
            Some(_) => return None,
            None => false,
        };
        if t.next().is_some() {
            return None;
        }
        let (src, src_fp8) = match Fp8::parse(st) {
            // An 8-bit float converts to a float or to an integer (not to `bool`, checked below).
            Some((lanes, fmt)) if sk == "f" => (
                NumTy {
                    elem: Elem::I8,
                    lanes,
                },
                Some(fmt),
            ),
            Some(_) => return None,
            None => (NumTy::parse(st)?, None),
        };
        let c = Convert {
            dst_kind: NumKind::parse(dk)?,
            dst: NumTy::parse(dt)?,
            src_kind: NumKind::parse(sk)?,
            src,
            src_fp8,
            rounding,
            saturate,
        };
        let src_ok = match src_fp8 {
            Some(_) => c.dst.elem != Elem::I1,
            None => c.src_kind.admits(c.src.elem),
        };
        // A rounding is spelled between numbers (no `bool`, no 8-bit float), and `.sat` only after
        // one, to an integer: the forms Apple's OpenCL runtime declares.
        let numbers = c.dst.elem != Elem::I1 && c.src.elem != Elem::I1 && src_fp8.is_none();
        let suffix_ok = match (rounding.is_some(), saturate) {
            (false, false) => true,
            (true, false) => numbers,
            (true, true) => numbers && c.dst_kind != NumKind::Float,
            (false, true) => false,
        };
        let ok = c.dst_kind.admits(c.dst.elem) && src_ok && suffix_ok && c.dst.lanes == c.src.lanes;
        ok.then_some(c)
    }

    /// `dst (src)`.
    pub fn signature(&self) -> Signature {
        Signature::new(Operand::Num(self.dst), &[Operand::Num(self.src)])
    }
}

/// A packed format of `air.pack.*` / `air.unpack.*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PackFormat {
    /// `unorm4x8`: four 8-bit unsigned normalized values in a `uint`.
    Unorm4x8,
    /// `unorm4x8.srgb`: four 8-bit unsigned normalized values in a `uint`, the first three
    /// sRGB-encoded.
    Unorm4x8Srgb,
    /// `snorm4x8`: four 8-bit signed normalized values in a `uint`.
    Snorm4x8,
    /// `unorm2x16`: two 16-bit unsigned normalized values in a `uint`.
    Unorm2x16,
    /// `snorm2x16`: two 16-bit signed normalized values in a `uint`.
    Snorm2x16,
    /// `unorm.rgb10a2`: three 10-bit and one 2-bit unsigned normalized values in a `uint`.
    UnormRgb10a2,
    /// `air.{pack,unpack}.snorm.rgb10a2`: MSL 4's `pack_float_to_snorm10a2` and
    /// `unpack_snorm10a2_to_float` (and their `half` forms).
    SnormRgb10a2,
    /// `unorm.rg11b10f`: three unsigned floats of 11, 11, and 10 bits in a `uint`.
    UnormRg11b10f,
    /// `unorm.rgb9e5`: three 9-bit mantissas with a shared 5-bit exponent in a `uint`.
    UnormRgb9e5,
    /// `unorm.rgb565`: three unsigned normalized values of 5, 6, and 5 bits in a `ushort`.
    UnormRgb565,
}

/// Every format: its name tokens, lane count, and packed integer element.
pub(super) const FORMATS: &[(&str, PackFormat, u32, Elem)] = &[
    ("unorm4x8", PackFormat::Unorm4x8, 4, Elem::I32),
    ("unorm4x8.srgb", PackFormat::Unorm4x8Srgb, 4, Elem::I32),
    ("snorm4x8", PackFormat::Snorm4x8, 4, Elem::I32),
    ("unorm2x16", PackFormat::Unorm2x16, 2, Elem::I32),
    ("snorm2x16", PackFormat::Snorm2x16, 2, Elem::I32),
    ("unorm.rgb10a2", PackFormat::UnormRgb10a2, 4, Elem::I32),
    ("snorm.rgb10a2", PackFormat::SnormRgb10a2, 4, Elem::I32),
    ("unorm.rg11b10f", PackFormat::UnormRg11b10f, 3, Elem::I32),
    ("unorm.rgb9e5", PackFormat::UnormRgb9e5, 3, Elem::I32),
    ("unorm.rgb565", PackFormat::UnormRgb565, 3, Elem::I16),
];

impl PackFormat {
    fn entry(self) -> (u32, Elem) {
        FORMATS
            .iter()
            .find(|(_, f, _, _)| *f == self)
            .map_or((0, Elem::I32), |&(_, _, lanes, packed)| (lanes, packed))
    }
}

/// Which way a [`Pack`] converts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PackDir {
    /// Floats to the packed integer.
    Pack,
    /// The packed integer to floats.
    Unpack,
}

/// A pack or unpack of a float vector to or from a packed integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pack {
    /// Whether this packs (`air.pack`) or unpacks (`air.unpack`).
    pub dir: PackDir,
    /// The packed format the name spells.
    pub format: PackFormat,
    /// The float vector side.
    pub ty: NumTy,
}

impl Pack {
    /// Parse the part of a name after `air.pack.` / `air.unpack.`.
    pub(super) fn parse(dir: PackDir, rest: &str) -> Option<Pack> {
        let (format, ty) = rest.rsplit_once('.')?;
        let &(_, format, lanes, _) = FORMATS.iter().find(|(name, ..)| *name == format)?;
        let ty = NumTy::parse(ty)?;
        let ok = matches!(ty.elem, Elem::F16 | Elem::F32) && ty.lanes == lanes;
        ok.then_some(Pack { dir, format, ty })
    }

    /// `packed (floats)` or `floats (packed)`.
    pub fn signature(&self) -> Signature {
        let packed = Operand::Num(NumTy::scalar(self.format.entry().1));
        let floats = Operand::Num(self.ty);
        match self.dir {
            PackDir::Pack => Signature::new(packed, &[floats]),
            PackDir::Unpack => Signature::new(floats, &[packed]),
        }
    }
}
