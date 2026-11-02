//! The GPU's decode of ASTC LDR blocks into a decoded twin's texels: the module
//! [`astc_decode_module`](super::astc_decode_module) serializes, the same decode as the CPU's
//! ([`AstcDecode::decode`]) in integer operations alone.
//!
//! One invocation per texel decodes its block's header, its partition's endpoints and its own
//! weights, straight-line: each reading of the integer sequence encoding (a colour value or a
//! weight) computes the plain, trit and quint forms of its value and selects the range's, and
//! every error condition joins one flag, which turns the texel magenta. The tables of the trits,
//! the quints and the unquantizations are the CPU decode's own functions, tabulated as the module
//! is built.

use super::{
    quints, trits, unquantize_color, unquantize_weight, AstcDecode, AstcTexels, Range, RANGES,
    WEIGHT_RANGES,
};
use crate::bind::block_decode::{table, BlockDecodeModule, BLOCKS, SHAPE, TEXELS};
use crate::bind::ycbcr_convert::{scalar, words_buffer};
use crate::lower::arith::exact_division::div_exact;
use crate::lower::arith::legal::{Form, Held};
use crate::lower::arith::srgb::SRGB_DECODE_STEPS;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, ExecMode, ExecModel, IoDir, MemAccess, Type, TypedModule,
    UnOp,
};

mod hdr;

/// The module's tables (the module docs), as `Private` arrays.
#[derive(Clone, Copy)]
struct Tables {
    /// The five trits of each 8-bit trit block, 2 bits each.
    trits: GlobalId,
    /// The three quints of each 7-bit quint block, 3 bits each.
    quints: GlobalId,
    /// Each range of [`RANGES`]: its bits, and its base at bit 8.
    ranges: GlobalId,
    /// Each range of [`WEIGHT_RANGES`], likewise.
    weight_ranges: GlobalId,
    /// The first entry of each range of [`RANGES`] in `colors`.
    color_at: GlobalId,
    /// Every colour value of every range of 6 levels or more, unquantized (0 elsewhere).
    colors: GlobalId,
    /// The first entry of each range of [`WEIGHT_RANGES`] in `weights`.
    weight_at: GlobalId,
    /// Every weight of every weight range, unquantized.
    weights: GlobalId,
    /// Metal's sRGB decode step of each 8-bit code.
    srgb_steps: GlobalId,
}

impl Tables {
    fn new(b: &mut ModuleBuilder, u32_: TypeId) -> BResult<Tables> {
        let info = |r: &Range| r.bits | r.base << 8;
        let mut color_at = Vec::with_capacity(RANGES.len());
        let mut colors = Vec::new();
        for range in RANGES {
            color_at.push(colors.len() as u32);
            for v in 0..range.levels() {
                colors.push(match range.levels() >= 6 {
                    true => unquantize_color(range, v),
                    false => 0,
                });
            }
        }
        let mut weight_at = Vec::with_capacity(WEIGHT_RANGES.len());
        let mut weights = Vec::new();
        for range in WEIGHT_RANGES {
            weight_at.push(weights.len() as u32);
            weights.extend((0..range.levels()).map(|v| unquantize_weight(range, v)));
        }
        let trit_words: Vec<u32> = (0..256)
            .map(|t| {
                (trits(t).into_iter())
                    .enumerate()
                    .fold(0, |w, (k, v)| w | v << (2 * k))
            })
            .collect();
        let quint_words: Vec<u32> = (0..128)
            .map(|q| {
                (quints(q).into_iter())
                    .enumerate()
                    .fold(0, |w, (k, v)| w | v << (3 * k))
            })
            .collect();
        let ranges: Vec<u32> = RANGES.iter().map(info).collect();
        let weight_ranges: Vec<u32> = WEIGHT_RANGES.iter().map(info).collect();
        let steps: Vec<u32> = SRGB_DECODE_STEPS.iter().map(|&s| u32::from(s)).collect();
        Ok(Tables {
            trits: table(b, u32_, &trit_words)?,
            quints: table(b, u32_, &quint_words)?,
            ranges: table(b, u32_, &ranges)?,
            weight_ranges: table(b, u32_, &weight_ranges)?,
            color_at: table(b, u32_, &color_at)?,
            colors: table(b, u32_, &colors)?,
            weight_at: table(b, u32_, &weight_at)?,
            weights: table(b, u32_, &weights)?,
            srgb_steps: table(b, u32_, &steps)?,
        })
    }
}

/// A 128-bit stream of four words, its lowest bit first.
type Stream = [Value; 4];

/// A range of the integer sequence encoding, read from a table at decode time: its low bits and
/// its base (1, 3 or 5).
#[derive(Clone, Copy)]
struct Ise {
    bits: Value,
    base: Value,
}

/// The decode of one texel: the function it builds in, and the values every step reads.
struct K<'a, 'b> {
    f: &'a mut FunctionBuilder<'b>,
    u32_: TypeId,
    tables: Tables,
}

impl K<'_, '_> {
    fn k(&mut self, v: u32) -> BResult<Value> {
        scalar(self.f, self.u32_, u64::from(v))
    }

    /// The `bool` constant `v`.
    fn boolean(&mut self, v: bool) -> BResult<Value> {
        let ty = self.f.module().ty(Type::Bool)?;
        let c = self.f.module().constant(Constant::Scalar {
            ty,
            bits: ConstBits(u64::from(v)),
        })?;
        self.f.module().const_value(c)
    }

    /// Whether `v` is one of `values`.
    fn one_of(&mut self, v: Value, values: &[u32]) -> BResult<Value> {
        let mut any = self.boolean(false)?;
        for &value in values {
            let is = self.eqk(v, value)?;
            any = self.lor(any, is)?;
        }
        Ok(any)
    }

    fn op(&mut self, op: BinOp, a: Value, b: Value) -> BResult<Value> {
        self.f.binary(op, a, b)
    }

    fn opk(&mut self, op: BinOp, a: Value, b: u32) -> BResult<Value> {
        let b = self.k(b)?;
        self.f.binary(op, a, b)
    }

    fn add(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::IAdd, a, b)
    }

    fn addk(&mut self, a: Value, b: u32) -> BResult<Value> {
        self.opk(BinOp::IAdd, a, b)
    }

    fn sub(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::ISub, a, b)
    }

    fn mul(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::IMul, a, b)
    }

    fn mulk(&mut self, a: Value, b: u32) -> BResult<Value> {
        self.opk(BinOp::IMul, a, b)
    }

    fn and(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::BitwiseAnd, a, b)
    }

    fn andk(&mut self, a: Value, b: u32) -> BResult<Value> {
        self.opk(BinOp::BitwiseAnd, a, b)
    }

    fn or(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::BitwiseOr, a, b)
    }

    fn shl(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::ShiftLeftLogical, a, b)
    }

    fn shlk(&mut self, a: Value, b: u32) -> BResult<Value> {
        self.opk(BinOp::ShiftLeftLogical, a, b)
    }

    fn shr(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::ShiftRightLogical, a, b)
    }

    fn shrk(&mut self, a: Value, b: u32) -> BResult<Value> {
        self.opk(BinOp::ShiftRightLogical, a, b)
    }

    fn sel(&mut self, c: Value, a: Value, b: Value) -> BResult<Value> {
        self.f.select(c, a, b)
    }

    fn eqk(&mut self, a: Value, b: u32) -> BResult<Value> {
        self.opk(BinOp::IEqual, a, b)
    }

    fn lor(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::LogicalOr, a, b)
    }

    fn land(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::LogicalAnd, a, b)
    }

    fn not(&mut self, a: Value) -> BResult<Value> {
        self.f.unary(UnOp::LogicalNot, a)
    }

    /// `(v >> at) & (2^n - 1)` for a constant field.
    fn field(&mut self, v: Value, at: u32, n: u32) -> BResult<Value> {
        let shifted = self.shrk(v, at)?;
        self.andk(shifted, (1 << n) - 1)
    }

    /// Entry `index` of the `Private` table `table`.
    fn entry(&mut self, table: GlobalId, index: Value) -> BResult<Value> {
        let table = self.f.module().global_value(table)?;
        let p = self.f.access_chain(table, &[index])?;
        self.f.load(p, MemAccess::default())
    }

    /// Word `i` of `stream`, 0 past its fourth.
    fn word(&mut self, stream: &Stream, i: Value) -> BResult<Value> {
        let mut word = self.k(0)?;
        for (k, &w) in (0..4).zip(stream).rev() {
            let at = self.eqk(i, k)?;
            word = self.sel(at, w, word)?;
        }
        Ok(word)
    }

    /// `2^n - 1` for `n` of 0 to 31.
    fn mask(&mut self, n: Value) -> BResult<Value> {
        let one = self.k(1)?;
        let top = self.shl(one, n)?;
        self.sub(top, one)
    }

    /// The `n` bits (0 to 31) of `stream` from bit `at`, a bit past 127 read as 0.
    fn read(&mut self, stream: &Stream, at: Value, n: Value) -> BResult<Value> {
        let i = self.shrk(at, 5)?;
        let shift = self.andk(at, 31)?;
        let low = self.word(stream, i)?;
        let low = self.shr(low, shift)?;
        let next = self.addk(i, 1)?;
        let high = self.word(stream, next)?;
        let thirty_two = self.k(32)?;
        let up = self.sub(thirty_two, shift)?;
        let up = self.andk(up, 31)?;
        let high = self.shl(high, up)?;
        let aligned = self.eqk(shift, 0)?;
        let zero = self.k(0)?;
        let high = self.sel(aligned, zero, high)?;
        let bits = self.or(low, high)?;
        let mask = self.mask(n)?;
        self.and(bits, mask)
    }

    /// [`read`](Self::read) of a stream whose bits from `end` on are zero.
    fn read_to(&mut self, stream: &Stream, at: Value, n: Value, end: Value) -> BResult<Value> {
        let bits = self.read(stream, at, n)?;
        let past = self.op(BinOp::UGreaterThanEqual, at, end)?;
        let left = self.sub(end, at)?;
        let left = self.op(BinOp::ULessThan, left, n)?;
        let available = self.sub(end, at)?;
        let available = self.sel(left, available, n)?;
        let zero = self.k(0)?;
        let available = self.sel(past, zero, available)?;
        let mask = self.mask(available)?;
        self.and(bits, mask)
    }

    /// [`read_to`](Self::read_to) of a constant width.
    fn read_tok(&mut self, stream: &Stream, at: Value, n: u32, end: Value) -> BResult<Value> {
        let n = self.k(n)?;
        self.read_to(stream, at, n, end)
    }

    /// The range of entry `index` of a range table.
    fn ise(&mut self, ranges: GlobalId, index: Value) -> BResult<Ise> {
        let info = self.entry(ranges, index)?;
        let bits = self.andk(info, 0xff)?;
        let base = self.shrk(info, 8)?;
        Ok(Ise { bits, base })
    }

    /// The bits `count` values of `range` take in the encoding.
    fn ise_bits(&mut self, range: Ise, count: Value) -> BResult<Value> {
        let plain = self.mul(count, range.bits)?;
        let eight = self.mulk(count, 8)?;
        let eight = self.addk(eight, 4)?;
        let trit_bits = self.opk(BinOp::UDiv, eight, 5)?;
        let seven = self.mulk(count, 7)?;
        let seven = self.addk(seven, 2)?;
        let quint_bits = self.opk(BinOp::UDiv, seven, 3)?;
        let zero = self.k(0)?;
        let is_trit = self.eqk(range.base, 3)?;
        let is_quint = self.eqk(range.base, 5)?;
        let extra = self.sel(is_quint, quint_bits, zero)?;
        let extra = self.sel(is_trit, trit_bits, extra)?;
        self.add(plain, extra)
    }

    /// Value `i` of the integer sequence encoding of `range` from bit `start` of `stream`, whose
    /// bits from `end` on are zero: its trit or quint times `2^bits` plus its low bits.
    fn ise_value(
        &mut self,
        stream: &Stream,
        (start, end): (Value, Value),
        range: Ise,
        i: Value,
    ) -> BResult<Value> {
        let n = range.bits;
        // Plain: `bits` bits at `start + i · bits`.
        let at = self.mul(i, n)?;
        let at = self.add(start, at)?;
        let plain = self.read_to(stream, at, n, end)?;
        // Trits: groups of five values in `5 bits + 8` bits.
        let group = self.opk(BinOp::UDiv, i, 5)?;
        let first = self.mulk(group, 5)?;
        let k = self.sub(i, first)?;
        let size = self.mulk(n, 5)?;
        let size = self.addk(size, 8)?;
        let size = self.mul(group, size)?;
        let base = self.add(start, size)?;
        let mut t = self.k(0)?;
        // The trit block's bits: 2 after the first value, 2 after the second, 1, 2, 1.
        for (values, gaps, width, at) in [
            (1, 0, 2, 0),
            (2, 2, 2, 2),
            (3, 4, 1, 4),
            (4, 5, 2, 5),
            (5, 7, 1, 7),
        ] {
            let pos = self.mulk(n, values)?;
            let pos = self.addk(pos, gaps)?;
            let pos = self.add(base, pos)?;
            let piece = self.read_tok(stream, pos, width, end)?;
            let piece = self.shlk(piece, at)?;
            t = self.or(t, piece)?;
        }
        let mut gap = self.k(0)?;
        for (index, before) in [(1, 2), (2, 4), (3, 5), (4, 7)] {
            let here = self.eqk(k, index)?;
            let before = self.k(before)?;
            gap = self.sel(here, before, gap)?;
        }
        let at = self.mul(k, n)?;
        let at = self.add(at, gap)?;
        let at = self.add(base, at)?;
        let low = self.read_to(stream, at, n, end)?;
        let trit_word = self.entry(self.tables.trits, t)?;
        let shift = self.shlk(k, 1)?;
        let trit = self.shr(trit_word, shift)?;
        let trit = self.andk(trit, 3)?;
        let trit = self.shl(trit, n)?;
        let trit_value = self.or(trit, low)?;
        // Quints: groups of three values in `3 bits + 7` bits.
        let group = self.opk(BinOp::UDiv, i, 3)?;
        let first = self.mulk(group, 3)?;
        let k = self.sub(i, first)?;
        let size = self.mulk(n, 3)?;
        let size = self.addk(size, 7)?;
        let size = self.mul(group, size)?;
        let base = self.add(start, size)?;
        let mut q = self.k(0)?;
        // The quint block's bits: 3 after the first value, 2 after the second, 2 after the third.
        for (values, gaps, width, at) in [(1, 0, 3, 0), (2, 3, 2, 3), (3, 5, 2, 5)] {
            let pos = self.mulk(n, values)?;
            let pos = self.addk(pos, gaps)?;
            let pos = self.add(base, pos)?;
            let piece = self.read_tok(stream, pos, width, end)?;
            let piece = self.shlk(piece, at)?;
            q = self.or(q, piece)?;
        }
        let mut gap = self.k(0)?;
        for (index, before) in [(1, 3), (2, 5)] {
            let here = self.eqk(k, index)?;
            let before = self.k(before)?;
            gap = self.sel(here, before, gap)?;
        }
        let at = self.mul(k, n)?;
        let at = self.add(at, gap)?;
        let at = self.add(base, at)?;
        let low = self.read_to(stream, at, n, end)?;
        let quint_word = self.entry(self.tables.quints, q)?;
        let shift = self.mulk(k, 3)?;
        let quint = self.shr(quint_word, shift)?;
        let quint = self.andk(quint, 7)?;
        let quint = self.shl(quint, n)?;
        let quint_value = self.or(quint, low)?;
        let is_trit = self.eqk(range.base, 3)?;
        let is_quint = self.eqk(range.base, 5)?;
        let value = self.sel(is_quint, quint_value, plain)?;
        self.sel(is_trit, trit_value, value)
    }

    /// `v` clamped into 0 to 255, as a signed value.
    fn clamp8(&mut self, v: Value) -> BResult<Value> {
        let zero = self.k(0)?;
        let top = self.k(255)?;
        let low = self.op(BinOp::SLessThan, v, zero)?;
        let v = self.sel(low, zero, v)?;
        let high = self.op(BinOp::SGreaterThan, v, top)?;
        self.sel(high, top, v)
    }

    /// `bit_transfer_signed(a, b)`: the signed 6-bit offset and the base.
    fn bit_transfer(&mut self, a: Value, b: Value) -> BResult<(Value, Value)> {
        let b = self.shrk(b, 1)?;
        let top = self.andk(a, 0x80)?;
        let b = self.or(b, top)?;
        let a = self.shrk(a, 1)?;
        let a = self.andk(a, 0x3f)?;
        let negative = self.andk(a, 0x20)?;
        let negative = self.opk(BinOp::INotEqual, negative, 0)?;
        let lowered = self.opk(BinOp::ISub, a, 0x40)?;
        let a = self.sel(negative, lowered, a)?;
        Ok((a, b))
    }

    /// `blue_contract(r, g, b, a)`, signed.
    fn blue_contract(&mut self, [r, g, b, a]: [Value; 4]) -> BResult<[Value; 4]> {
        let r = self.add(r, b)?;
        let r = self.opk(BinOp::ShiftRightArithmetic, r, 1)?;
        let g = self.add(g, b)?;
        let g = self.opk(BinOp::ShiftRightArithmetic, g, 1)?;
        Ok([r, g, b, a])
    }

    fn clamp4(&mut self, e: [Value; 4]) -> BResult<[Value; 4]> {
        Ok([
            self.clamp8(e[0])?,
            self.clamp8(e[1])?,
            self.clamp8(e[2])?,
            self.clamp8(e[3])?,
        ])
    }

    fn sel4(&mut self, c: Value, a: [Value; 4], b: [Value; 4]) -> BResult<[Value; 4]> {
        Ok([
            self.sel(c, a[0], b[0])?,
            self.sel(c, a[1], b[1])?,
            self.sel(c, a[2], b[2])?,
            self.sel(c, a[3], b[3])?,
        ])
    }

    /// The two endpoints of colour endpoint mode `cem` from its unquantized values `v`
    /// (`super::endpoints`), and whether the mode is an HDR one.
    fn endpoints(&mut self, cem: Value, v: [Value; 8]) -> BResult<([[Value; 4]; 2], Value)> {
        let opaque = self.k(255)?;
        let zero = self.k(0)?;
        // 0: luminance.
        let lum = [[v[0], v[0], v[0], opaque], [v[1], v[1], v[1], opaque]];
        // 1: luminance, base and offset.
        let l0 = self.shrk(v[0], 2)?;
        let high = self.andk(v[1], 0xc0)?;
        let l0 = self.or(l0, high)?;
        let offset = self.andk(v[1], 0x3f)?;
        let l1 = self.add(l0, offset)?;
        let past = self.opk(BinOp::UGreaterThan, l1, 255)?;
        let l1 = self.sel(past, opaque, l1)?;
        let lum_delta = [[l0, l0, l0, opaque], [l1, l1, l1, opaque]];
        // 4: luminance and alpha.
        let lum_alpha = [[v[0], v[0], v[0], v[2]], [v[1], v[1], v[1], v[3]]];
        // 5: luminance and alpha, base and offset.
        let (o1, b0) = self.bit_transfer(v[1], v[0])?;
        let (o3, b2) = self.bit_transfer(v[3], v[2])?;
        let l1 = self.add(b0, o1)?;
        let a1 = self.add(b2, o3)?;
        let lum_alpha_delta = [
            self.clamp4([b0, b0, b0, b2])?,
            self.clamp4([l1, l1, l1, a1])?,
        ];
        // 6 and 10: RGB scaled, and alpha.
        let is_ten = self.eqk(cem, 10)?;
        let a0 = self.sel(is_ten, v[4], opaque)?;
        let a1 = self.sel(is_ten, v[5], opaque)?;
        let mut scaled = [zero; 3];
        for (c, slot) in scaled.iter_mut().enumerate() {
            let product = self.mul(v[c], v[3])?;
            *slot = self.shrk(product, 8)?;
        }
        let rgb_scale = [
            [scaled[0], scaled[1], scaled[2], a0],
            [v[0], v[1], v[2], a1],
        ];
        // 8 and 12: RGB, and alpha.
        let is_twelve = self.eqk(cem, 12)?;
        let a0 = self.sel(is_twelve, v[6], opaque)?;
        let a1 = self.sel(is_twelve, v[7], opaque)?;
        let s1 = self.add(v[1], v[3])?;
        let s1 = self.add(s1, v[5])?;
        let s0 = self.add(v[0], v[2])?;
        let s0 = self.add(s0, v[4])?;
        let forward = self.op(BinOp::UGreaterThanEqual, s1, s0)?;
        let e0 = self.clamp4([v[0], v[2], v[4], a0])?;
        let e1 = self.clamp4([v[1], v[3], v[5], a1])?;
        let c0 = self.blue_contract([v[1], v[3], v[5], a1])?;
        let c0 = self.clamp4(c0)?;
        let c1 = self.blue_contract([v[0], v[2], v[4], a0])?;
        let c1 = self.clamp4(c1)?;
        let rgb = [self.sel4(forward, e0, c0)?, self.sel4(forward, e1, c1)?];
        // 9 and 13: RGB, and alpha, base and offset.
        let is_thirteen = self.eqk(cem, 13)?;
        let (o1, b0) = self.bit_transfer(v[1], v[0])?;
        let (o3, b2) = self.bit_transfer(v[3], v[2])?;
        let (o5, b4) = self.bit_transfer(v[5], v[4])?;
        let (o7, b6) = self.bit_transfer(v[7], v[6])?;
        let o7 = self.sel(is_thirteen, o7, zero)?;
        let b6 = self.sel(is_thirteen, b6, opaque)?;
        let sum = self.add(o1, o3)?;
        let sum = self.add(sum, o5)?;
        let forward = self.op(BinOp::SGreaterThanEqual, sum, zero)?;
        let ends = [
            self.add(b0, o1)?,
            self.add(b2, o3)?,
            self.add(b4, o5)?,
            self.add(b6, o7)?,
        ];
        let e0 = self.clamp4([b0, b2, b4, b6])?;
        let e1 = self.clamp4(ends)?;
        let c0 = self.blue_contract(ends)?;
        let c0 = self.clamp4(c0)?;
        let c1 = self.blue_contract([b0, b2, b4, b6])?;
        let c1 = self.clamp4(c1)?;
        let rgb_delta = [self.sel4(forward, e0, c0)?, self.sel4(forward, e1, c1)?];
        // The mode's.
        let mut out = lum;
        let modes: [(&[u32], _); 6] = [
            (&[1], lum_delta),
            (&[4], lum_alpha),
            (&[5], lum_alpha_delta),
            (&[6, 10], rgb_scale),
            (&[8, 12], rgb),
            (&[9, 13], rgb_delta),
        ];
        for (modes, ends) in modes {
            let here = self.one_of(cem, modes)?;
            out = [
                self.sel4(here, ends[0], out[0])?,
                self.sel4(here, ends[1], out[1])?,
            ];
        }
        let hdr = self.one_of(cem, &[2, 3, 7, 11, 14, 15])?;
        Ok((out, hdr))
    }

    /// `hash52(p)`.
    fn hash52(&mut self, p: Value) -> BResult<Value> {
        let xor = |k: &mut Self, a: Value, b: Value| k.op(BinOp::BitwiseXor, a, b);
        let t = self.shrk(p, 15)?;
        let p = xor(self, p, t)?;
        let t = self.shlk(p, 17)?;
        let p = self.sub(p, t)?;
        let t = self.shlk(p, 7)?;
        let p = self.add(p, t)?;
        let t = self.shlk(p, 4)?;
        let p = self.add(p, t)?;
        let t = self.shrk(p, 5)?;
        let p = xor(self, p, t)?;
        let t = self.shlk(p, 16)?;
        let p = self.add(p, t)?;
        let t = self.shrk(p, 7)?;
        let p = xor(self, p, t)?;
        let t = self.shrk(p, 3)?;
        let p = xor(self, p, t)?;
        let t = self.shlk(p, 6)?;
        let p = xor(self, p, t)?;
        let t = self.shrk(p, 17)?;
        xor(self, p, t)
    }

    /// The partition of texel `(x, y)` of a block of `texels` texels, `count` partitions (2 to
    /// 4) and partition index `seed` (`super::Partitions`).
    fn partition(
        &mut self,
        seed: Value,
        count: Value,
        texels: u32,
        [x, y]: [Value; 2],
    ) -> BResult<Value> {
        let seed = {
            let less = self.addk(count, u32::MAX)?;
            let shifted = self.shlk(less, 10)?;
            self.add(seed, shifted)?
        };
        let rnum = self.hash52(seed)?;
        let mut s = [rnum; 8];
        for (i, slot) in s.iter_mut().enumerate() {
            *slot = self.field(rnum, 4 * i as u32, 4)?;
        }
        let four = self.k(4)?;
        let five = self.k(5)?;
        let six = self.k(6)?;
        let three = self.eqk(count, 3)?;
        let odd = self.andk(seed, 1)?;
        let odd = self.opk(BinOp::INotEqual, odd, 0)?;
        let two = self.andk(seed, 2)?;
        let two = self.opk(BinOp::INotEqual, two, 0)?;
        let by_two = self.sel(two, four, five)?;
        let by_three = self.sel(three, six, five)?;
        let sh1 = self.sel(odd, by_two, by_three)?;
        let sh2 = self.sel(odd, by_three, by_two)?;
        for (i, slot) in s.iter_mut().enumerate() {
            let square = self.mul(*slot, *slot)?;
            let shift = if i % 2 == 0 { sh1 } else { sh2 };
            *slot = self.shr(square, shift)?;
        }
        let (x, y) = match texels < 31 {
            true => (self.shlk(x, 1)?, self.shlk(y, 1)?),
            false => (x, y),
        };
        let mut lines = [rnum; 4];
        for (line, (slot, shift)) in lines.iter_mut().zip([14, 10, 6, 2]).enumerate() {
            let sx = self.mul(s[2 * line], x)?;
            let sy = self.mul(s[2 * line + 1], y)?;
            let sum = self.add(sx, sy)?;
            let r = self.shrk(rnum, shift)?;
            let sum = self.add(sum, r)?;
            *slot = self.andk(sum, 0x3f)?;
        }
        let zero = self.k(0)?;
        let under_three = self.opk(BinOp::ULessThan, count, 3)?;
        let under_four = self.opk(BinOp::ULessThan, count, 4)?;
        let [a, b, c, d] = lines;
        let c = self.sel(under_three, zero, c)?;
        let d = self.sel(under_four, zero, d)?;
        let ge = |k: &mut Self, p: Value, q: Value| k.op(BinOp::UGreaterThanEqual, p, q);
        let ab = ge(self, a, b)?;
        let ac = ge(self, a, c)?;
        let ad = ge(self, a, d)?;
        let bc = ge(self, b, c)?;
        let bd = ge(self, b, d)?;
        let cd = ge(self, c, d)?;
        let first = self.land(ab, ac)?;
        let first = self.land(first, ad)?;
        let second = self.land(bc, bd)?;
        let one = self.k(1)?;
        let two = self.k(2)?;
        let three = self.k(3)?;
        let p = self.sel(cd, two, three)?;
        let p = self.sel(second, one, p)?;
        self.sel(first, zero, p)
    }

    /// Entry `v` of the unquantization table of `range` (an index into its range table) whose
    /// ranges start at `at` in `values`.
    fn unquantize(
        &mut self,
        (at, values): (GlobalId, GlobalId),
        range: Value,
        v: Value,
    ) -> BResult<Value> {
        let first = self.entry(at, range)?;
        let index = self.add(first, v)?;
        self.entry(values, index)
    }
}

/// The texel's channels, each its 16-bit interpolation result `C` (an HDR format's half-precision
/// bits), and whether the texel is an error one: the decode of the block `words` at texel
/// `(s, t)` of a block of `decode`'s footprint (`super::decode_block` and `super::void_extent`).
fn texel(
    k: &mut K<'_, '_>,
    decode: AstcDecode,
    words: Stream,
    [s, t]: [Value; 2],
) -> BResult<([Value; 4], Value)> {
    let (bw, bh) = (decode.width, decode.height);
    let zero = k.k(0)?;
    let no = k.boolean(false)?;
    let w0 = words[0];
    let mode = k.andk(w0, 0x7ff)?;
    let bit = |k: &mut K<'_, '_>, i: u32| k.field(mode, i, 1);
    let bits = |k: &mut K<'_, '_>, i: u32, n: u32| k.field(mode, i, n);
    // The block mode (`super::block_mode`): the first form where its low two bits are set.
    let low2 = k.andk(mode, 3)?;
    let first_form = k.opk(BinOp::INotEqual, low2, 0)?;
    let b4 = bit(k, 4)?;
    let b8 = bit(k, 8)?;
    let b9 = bit(k, 9)?;
    let b10 = bit(k, 10)?;
    let a = bits(k, 5, 2)?;
    let b = bits(k, 7, 2)?;
    let low2_up = k.shlk(low2, 1)?;
    let range_first = k.or(b4, low2_up)?;
    let a2 = k.addk(a, 2)?;
    let b4w = k.addk(b, 4)?;
    let b8w = k.addk(b, 8)?;
    let b_low = k.andk(b, 1)?;
    let b6 = k.addk(b_low, 6)?;
    let b2 = k.addk(b_low, 2)?;
    let layout = bits(k, 2, 2)?;
    let is0 = k.eqk(layout, 0)?;
    let is1 = k.eqk(layout, 1)?;
    let is2 = k.eqk(layout, 2)?;
    let high8 = k.opk(BinOp::INotEqual, b8, 0)?;
    let gw3 = k.sel(high8, b2, a2)?;
    let gh3 = k.sel(high8, a2, b6)?;
    let gw = k.sel(is2, a2, gw3)?;
    let gh = k.sel(is2, b8w, gh3)?;
    let gw = k.sel(is1, b8w, gw)?;
    let gh = k.sel(is1, a2, gh)?;
    let gw_first = k.sel(is0, b4w, gw)?;
    let gh_first = k.sel(is0, a2, gh)?;
    // The second form.
    let layout_up = k.shlk(layout, 1)?;
    let range_second = k.or(b4, layout_up)?;
    let twelve = k.k(12)?;
    let a6 = k.addk(a, 6)?;
    let c9 = bits(k, 9, 2)?;
    let c6 = k.addk(c9, 6)?;
    let six = k.k(6)?;
    let ten = k.k(10)?;
    let c0 = k.eqk(b, 0)?;
    let c1 = k.eqk(b, 1)?;
    let c2 = k.eqk(b, 2)?;
    let a0 = k.eqk(a, 0)?;
    let gw3 = k.sel(a0, six, ten)?;
    let gh3 = k.sel(a0, ten, six)?;
    let gw = k.sel(c2, a6, gw3)?;
    let gh = k.sel(c2, c6, gh3)?;
    let gw = k.sel(c1, a2, gw)?;
    let gh = k.sel(c1, twelve, gh)?;
    let gw_second = k.sel(c0, twelve, gw)?;
    let gh_second = k.sel(c0, a2, gh)?;
    let plain_high = k.sel(c2, zero, b9)?;
    let plain_dual = k.sel(c2, zero, b10)?;
    let low4 = k.andk(mode, 0xf)?;
    let reserved_low = k.eqk(low4, 0)?;
    let a_past = k.opk(BinOp::UGreaterThan, a, 1)?;
    let c3 = k.eqk(b, 3)?;
    let reserved_c3 = k.land(c3, a_past)?;
    let reserved_second = k.lor(reserved_low, reserved_c3)?;
    let range_bits = k.sel(first_form, range_first, range_second)?;
    let high = k.sel(first_form, b9, plain_high)?;
    let dual = k.sel(first_form, b10, plain_dual)?;
    let gw = k.sel(first_form, gw_first, gw_second)?;
    let gh = k.sel(first_form, gh_first, gh_second)?;
    let reserved = k.sel(first_form, no, reserved_second)?;
    let few_range = k.opk(BinOp::ULessThan, range_bits, 2)?;
    let mut bad = k.lor(reserved, few_range)?;
    let high_at = k.mulk(high, 6)?;
    let weight_range = k.addk(range_bits, u32::MAX - 1)?;
    let weight_range = k.add(weight_range, high_at)?;
    let weight_range = k.andk(weight_range, 15)?;
    let eleven = k.k(11)?;
    let past_table = k.op(BinOp::UGreaterThan, weight_range, eleven)?;
    let weight_range = k.sel(past_table, zero, weight_range)?;
    let wise = k.ise(k.tables.weight_ranges, weight_range)?;
    let dual = k.opk(BinOp::INotEqual, dual, 0)?;
    let one = k.k(1)?;
    let two = k.k(2)?;
    let planes = k.sel(dual, two, one)?;
    // The grid fits the block, holds at most 64 weights in 24 to 96 bits.
    let wide = k.opk(BinOp::UGreaterThan, gw, bw)?;
    let tall = k.opk(BinOp::UGreaterThan, gh, bh)?;
    bad = k.lor(bad, wide)?;
    bad = k.lor(bad, tall)?;
    let grid = k.mul(gw, gh)?;
    let weight_count = k.mul(grid, planes)?;
    let many = k.opk(BinOp::UGreaterThan, weight_count, 64)?;
    bad = k.lor(bad, many)?;
    let weight_bits = k.ise_bits(wise, weight_count)?;
    let few = k.opk(BinOp::ULessThan, weight_bits, 24)?;
    let lots = k.opk(BinOp::UGreaterThan, weight_bits, 96)?;
    bad = k.lor(bad, few)?;
    bad = k.lor(bad, lots)?;
    // Partitions and colour endpoint modes.
    let count = k.field(w0, 11, 2)?;
    let count = k.addk(count, 1)?;
    let four_parts = k.eqk(count, 4)?;
    let dual_four = k.land(dual, four_parts)?;
    bad = k.lor(bad, dual_four)?;
    let single = k.eqk(count, 1)?;
    let below = k.k(128)?;
    let below = k.sub(below, weight_bits)?;
    let cem_single = k.field(w0, 13, 4)?;
    let seed = k.field(w0, 13, 10)?;
    let low = k.field(w0, 23, 6)?;
    let low_class = k.andk(low, 3)?;
    let shared = k.eqk(low_class, 0)?;
    let three_count = k.mulk(count, 3)?;
    let extra_config = k.addk(three_count, u32::MAX - 3)?;
    let extra_multi = k.sel(shared, zero, extra_config)?;
    let extra = k.sel(single, zero, extra_multi)?;
    let past_weights = k.op(BinOp::UGreaterThan, extra, below)?;
    bad = k.lor(bad, past_weights)?;
    let config_at = k.sub(below, extra)?;
    let config_high = k.read(&words, config_at, extra)?;
    let config_high = k.shlk(config_high, 6)?;
    let config = k.or(low, config_high)?;
    let class_base = k.andk(config, 3)?;
    let class_base = k.addk(class_base, u32::MAX)?;
    let shared_cem = k.shrk(low, 2)?;
    let mut cems = [zero; 4];
    for (i, cem) in (0u32..).zip(cems.iter_mut()) {
        let class_shift = k.k(2 + i)?;
        let class = k.shr(config, class_shift)?;
        let class = k.andk(class, 1)?;
        let class = k.add(class_base, class)?;
        let class = k.shlk(class, 2)?;
        let m_shift = k.addk(count, 2 + 2 * i)?;
        let m = k.shr(config, m_shift)?;
        let m = k.andk(m, 3)?;
        let own = k.or(class, m)?;
        let multi = k.sel(shared, shared_cem, own)?;
        *cem = match i {
            0 => k.sel(single, cem_single, multi)?,
            _ => multi,
        };
    }
    let color_start = {
        let seventeen = k.k(17)?;
        let twenty_nine = k.k(29)?;
        k.sel(single, seventeen, twenty_nine)?
    };
    let plane_bits = k.sel(dual, two, zero)?;
    let reserved_below = k.add(extra, plane_bits)?;
    let short = k.op(BinOp::UGreaterThan, reserved_below, below)?;
    bad = k.lor(bad, short)?;
    let color_end = k.sub(below, reserved_below)?;
    let two_bits = k.k(2)?;
    let ccs = k.read(&words, color_end, two_bits)?;
    let mut value_count = zero;
    let mut counts = [zero; 4];
    for (i, cem) in cems.into_iter().enumerate() {
        let class = k.shrk(cem, 2)?;
        let n = k.addk(class, 1)?;
        let n = k.shlk(n, 1)?;
        let inside = k.opk(BinOp::UGreaterThan, count, i as u32)?;
        counts[i] = k.sel(inside, n, zero)?;
        value_count = k.add(value_count, counts[i])?;
    }
    let too_many = k.opk(BinOp::UGreaterThan, value_count, 18)?;
    bad = k.lor(bad, too_many)?;
    let backwards = k.op(BinOp::UGreaterThan, color_start, color_end)?;
    bad = k.lor(bad, backwards)?;
    let color_bits = k.sub(color_end, color_start)?;
    // The largest colour range of 6 levels or more whose encoding fits.
    let mut color_range = zero;
    let mut color_used = zero;
    for (index, range) in RANGES.iter().enumerate().filter(|(_, r)| r.levels() >= 6) {
        let info = Ise {
            bits: k.k(range.bits)?,
            base: k.k(range.base)?,
        };
        let used = k.ise_bits(info, value_count)?;
        let fits = k.op(BinOp::ULessThanEqual, used, color_bits)?;
        let index = k.k(index as u32)?;
        color_range = k.sel(fits, index, color_range)?;
        color_used = k.sel(fits, used, color_used)?;
    }
    let no_range = k.eqk(color_range, 0)?;
    bad = k.lor(bad, no_range)?;
    let cise = k.ise(k.tables.ranges, color_range)?;
    let color_stream_end = k.add(color_start, color_used)?;
    // The texel's partition, its mode and its values.
    let multi = k.not(single)?;
    let texels = bw * bh;
    let p = k.partition(seed, count, texels, [s, t])?;
    let p = k.sel(multi, p, zero)?;
    let mut cem = cems[0];
    let mut first_value = zero;
    let mut before = zero;
    for i in 1..4u32 {
        before = k.add(before, counts[i as usize - 1])?;
        let here = k.eqk(p, i)?;
        cem = k.sel(here, cems[i as usize], cem)?;
        first_value = k.sel(here, before, first_value)?;
    }
    let mut v = [zero; 8];
    for (i, slot) in (0u32..).zip(v.iter_mut()) {
        let index = k.addk(first_value, i)?;
        let raw = k.ise_value(&words, (color_start, color_stream_end), cise, index)?;
        let tables = (k.tables.color_at, k.tables.colors);
        *slot = k.unquantize(tables, color_range, raw)?;
    }
    let ([l0, l1], hdr_cem) = k.endpoints(cem, v)?;
    // The endpoints widened to 16 bits (`super::endpoints`): an LDR mode's 8-bit ones, and an HDR
    // format's HDR mode's 12-bit ones, logarithmic but for mode 14's alpha.
    let widen = |k: &mut K<'_, '_>, e: Value| match decode.srgb {
        true => {
            let up = k.shlk(e, 8)?;
            k.opk(BinOp::BitwiseOr, up, 0x80)
        }
        false => k.mulk(e, 257),
    };
    let mut e0 = [zero; 4];
    let mut e1 = [zero; 4];
    for ch in 0..4 {
        e0[ch] = widen(k, l0[ch])?;
        e1[ch] = widen(k, l1[ch])?;
    }
    let (partition_bad, lns) = match decode.hdr {
        true => {
            let ([h0, h1], alpha_lns) = k.hdr_endpoints(cem, v)?;
            for ch in 0..4 {
                for (end, hdr_end) in [(&mut e0[ch], h0[ch]), (&mut e1[ch], h1[ch])] {
                    let log = k.shlk(hdr_end, 4)?;
                    let wide = match ch {
                        3 => {
                            let linear = k.mulk(hdr_end, 257)?;
                            k.sel(alpha_lns, log, linear)?
                        }
                        _ => log,
                    };
                    *end = k.sel(hdr_cem, wide, *end)?;
                }
            }
            let alpha = k.land(hdr_cem, alpha_lns)?;
            (bad, Some([hdr_cem, hdr_cem, hdr_cem, alpha]))
        }
        false => (k.lor(bad, hdr_cem)?, None),
    };
    // The weights, from bit 127 down: the stream of the block's bits reversed.
    let mut reversed = words;
    for (slot, &w) in reversed.iter_mut().zip(words.iter().rev()) {
        *slot = k.f.unary(UnOp::BitReverse, w)?;
    }
    let ds = (1024 + bw / 2) / (bw - 1);
    let dt = (1024 + bh / 2) / (bh - 1);
    let gw_less = k.addk(gw, u32::MAX)?;
    let gh_less = k.addk(gh, u32::MAX)?;
    let gs = k.mulk(s, ds)?;
    let gs = k.mul(gs, gw_less)?;
    let gs = k.addk(gs, 32)?;
    let gs = k.shrk(gs, 6)?;
    let gt = k.mulk(t, dt)?;
    let gt = k.mul(gt, gh_less)?;
    let gt = k.addk(gt, 32)?;
    let gt = k.shrk(gt, 6)?;
    let js = k.shrk(gs, 4)?;
    let fs = k.andk(gs, 15)?;
    let jt = k.shrk(gt, 4)?;
    let ft = k.andk(gt, 15)?;
    let w11 = k.mul(fs, ft)?;
    let w11 = k.addk(w11, 8)?;
    let w11 = k.shrk(w11, 4)?;
    let w10 = k.sub(ft, w11)?;
    let w01 = k.sub(fs, w11)?;
    let sixteen = k.k(16)?;
    let w00 = k.sub(sixteen, fs)?;
    let w00 = k.sub(w00, ft)?;
    let w00 = k.add(w00, w11)?;
    let row = k.mul(jt, gw)?;
    let v0 = k.add(js, row)?;
    let v1 = k.addk(v0, 1)?;
    let v2 = k.add(v0, gw)?;
    let v3 = k.addk(v2, 1)?;
    let mut plane_weights = [zero; 2];
    for (plane, slot) in (0u32..).zip(plane_weights.iter_mut()) {
        let mut sum = k.k(8)?;
        for (index, factor) in [(v0, w00), (v1, w01), (v2, w10), (v3, w11)] {
            let at = k.mul(index, planes)?;
            let at = k.addk(at, plane)?;
            let raw = k.ise_value(&reversed, (zero, weight_bits), wise, at)?;
            let tables = (k.tables.weight_at, k.tables.weights);
            let weight = k.unquantize(tables, weight_range, raw)?;
            let inside = k.op(BinOp::ULessThan, at, weight_count)?;
            let weight = k.sel(inside, weight, zero)?;
            let term = k.mul(weight, factor)?;
            sum = k.add(sum, term)?;
        }
        *slot = k.shrk(sum, 4)?;
    }
    let second = k.sel(dual, plane_weights[1], plane_weights[0])?;
    let mut channels = [zero; 4];
    for (ch, slot) in (0u32..).zip(channels.iter_mut()) {
        let is_ccs = k.eqk(ccs, ch)?;
        let use_second = k.land(dual, is_ccs)?;
        let w = k.sel(use_second, second, plane_weights[0])?;
        let sixty_four = k.k(64)?;
        let rest = k.sub(sixty_four, w)?;
        let a = k.mul(e0[ch as usize], rest)?;
        let b = k.mul(e1[ch as usize], w)?;
        let sum = k.add(a, b)?;
        let sum = k.addk(sum, 32)?;
        let c = k.shrk(sum, 6)?;
        *slot = match lns {
            Some(lns) => {
                let log = k.lns_to_half(c)?;
                let linear = k.unorm16_to_half(c)?;
                k.sel(lns[ch as usize], log, linear)?
            }
            None => c,
        };
    }
    // A void-extent block (`super::void_extent`).
    let void_mode = k.andk(mode, 0x1ff)?;
    let void = k.eqk(void_mode, 0x1fc)?;
    let hdr_void = bit(k, 9)?;
    let hdr_void = k.opk(BinOp::INotEqual, hdr_void, 0)?;
    let reserved_void = k.field(w0, 10, 2)?;
    let reserved_void = k.opk(BinOp::INotEqual, reserved_void, 3)?;
    let mut coords = [zero; 4];
    let mut all_ones = k.boolean(true)?;
    for (i, slot) in (0u32..).zip(coords.iter_mut()) {
        let at = k.k(12 + 13 * i)?;
        let n = k.k(13)?;
        *slot = k.read(&words, at, n)?;
        let ones = k.eqk(*slot, 0x1fff)?;
        all_ones = k.land(all_ones, ones)?;
    }
    let empty_s = k.op(BinOp::UGreaterThanEqual, coords[0], coords[1])?;
    let empty_t = k.op(BinOp::UGreaterThanEqual, coords[2], coords[3])?;
    let empty = k.lor(empty_s, empty_t)?;
    let some = k.not(all_ones)?;
    let empty = k.land(some, empty)?;
    // An HDR format's HDR block holds halves, its LDR one 16-bit values read as an LDR channel.
    let void_bad = match decode.hdr {
        true => reserved_void,
        false => k.lor(hdr_void, reserved_void)?,
    };
    let void_bad = k.lor(void_bad, empty)?;
    let mut color = [zero; 4];
    for (i, slot) in color.iter_mut().enumerate() {
        let word = words[2 + i / 2];
        *slot = k.field(word, 16 * (i as u32 % 2), 16)?;
        if decode.hdr {
            let half = k.quiet_nan(*slot)?;
            let linear = k.unorm16_to_half(*slot)?;
            *slot = k.sel(hdr_void, half, linear)?;
        }
    }
    let channels = k.sel4(void, color, channels)?;
    let bad = k.sel(void, void_bad, partition_bad)?;
    Ok((channels, bad))
}

/// The module of `decode` writing `texels` as a typed module, before serialization: the module
/// [`astc_decode_module`](super::astc_decode_module) serializes, which the CPU interpreter runs.
/// Its bindings are the block decode's (`BlockDecodeModule`); it reads the shape's first three
/// words, the region's size.
pub(super) fn typed_module(decode: AstcDecode, texels: AstcTexels) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let shape = words_buffer(&mut b, SHAPE, u32_)?;
    let blocks = words_buffer(&mut b, BLOCKS, u32_)?;
    let texel_words = words_buffer(&mut b, TEXELS, u32_)?;
    let tables = Tables::new(&mut b, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = scalar(&mut f, u32_, 0)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let y = f.composite_extract(invocation, &[1])?;
    let z = f.composite_extract(invocation, &[2])?;
    let shape = f.module().global_value(shape)?;
    let field = |f: &mut FunctionBuilder<'_>, index: u32| -> BResult<Value> {
        let index = scalar(f, u32_, u64::from(index))?;
        let p = f.access_chain(shape, &[zero, index])?;
        f.load(p, MemAccess::default())
    };
    let width = field(&mut f, 0)?;
    let height = field(&mut f, 1)?;
    let depth = field(&mut f, 2)?;
    let in_x = f.binary(BinOp::ULessThan, x, width)?;
    let in_y = f.binary(BinOp::ULessThan, y, height)?;
    let in_z = f.binary(BinOp::ULessThan, z, depth)?;
    let inside = f.binary(BinOp::LogicalAnd, in_x, in_y)?;
    let inside = f.binary(BinOp::LogicalAnd, inside, in_z)?;
    let blocks = f.module().global_value(blocks)?;
    let texel_words = f.module().global_value(texel_words)?;
    let decoded: ArmFn<'_, ()> = Box::new(move |f, _| {
        let mut k = K { f, u32_, tables };
        let (bw, bh) = (decode.width, decode.height);
        // The texel's block, `(z · down + y / bh) · across + x / bw`, and its place in it.
        let across = k.addk(width, bw - 1)?;
        let across = k.opk(BinOp::UDiv, across, bw)?;
        let down = k.addk(height, bh - 1)?;
        let down = k.opk(BinOp::UDiv, down, bh)?;
        let bx = k.opk(BinOp::UDiv, x, bw)?;
        let by = k.opk(BinOp::UDiv, y, bh)?;
        let row = k.mul(z, down)?;
        let row = k.add(row, by)?;
        let block = k.mul(row, across)?;
        let block = k.add(block, bx)?;
        let base = k.shlk(block, 2)?;
        let mut words = [zero; 4];
        for (i, slot) in (0u32..).zip(words.iter_mut()) {
            let index = k.addk(base, i)?;
            let p = k.f.access_chain(blocks, &[zero, index])?;
            *slot = k.f.load(p, MemAccess::default())?;
        }
        let x0 = k.mulk(bx, bw)?;
        let s = k.sub(x, x0)?;
        let y0 = k.mulk(by, bh)?;
        let t = k.sub(y, y0)?;
        let (channels, bad) = texel(&mut k, decode, words, [s, t])?;
        // An error texel is magenta (each channel's `C` all ones, green 0), an HDR format's NaN.
        let error = match decode.hdr {
            true => [k.k(super::HDR_ERROR.into())?; 4],
            false => {
                let all = k.k(0xffff)?;
                [all, zero, all, all]
            }
        };
        let channels = k.sel4(bad, error, channels)?;
        let out: Vec<Value> = match decode.form(texels) {
            AstcTexels::Half => {
                let mut words = Vec::with_capacity(2);
                for pair in channels.as_chunks::<2>().0 {
                    let high = k.shlk(pair[1], 16)?;
                    words.push(k.or(pair[0], high)?);
                }
                words
            }
            AstcTexels::Float if decode.hdr => {
                let mut words = Vec::with_capacity(4);
                for c in channels {
                    words.push(k.half_bits_to_f32(c)?);
                }
                words
            }
            AstcTexels::Unorm8 => {
                let mut packed = zero;
                for (ch, c) in (0u32..).zip(channels) {
                    let code = match decode.srgb {
                        true => k.shrk(c, 8)?,
                        false => {
                            let twelve = k.shrk(c, 4)?;
                            let scaled = k.mulk(twelve, 255)?;
                            let scaled = k.addk(scaled, 2047)?;
                            k.opk(BinOp::UDiv, scaled, 4095)?
                        }
                    };
                    let placed = k.shlk(code, 8 * ch)?;
                    packed = k.or(packed, placed)?;
                }
                vec![packed]
            }
            AstcTexels::Float => {
                let u = Form {
                    held: Held::Int {
                        bits: 32,
                        container: 32,
                    },
                    lanes: 1,
                };
                let mut words = Vec::with_capacity(4);
                for (ch, c) in channels.into_iter().enumerate() {
                    let value = match (decode.srgb, ch) {
                        (true, 3) => {
                            let code = k.shrk(c, 8)?;
                            div_exact(k.f, u, code, 255)?
                        }
                        (true, _) => {
                            let code = k.shrk(c, 8)?;
                            let step = k.entry(k.tables.srgb_steps, code)?;
                            div_exact(k.f, u, step, 4095)?
                        }
                        (false, _) => {
                            let twelve = k.shrk(c, 4)?;
                            div_exact(k.f, u, twelve, 4095)?
                        }
                    };
                    words.push(k.f.convert(ConvOp::Bitcast, value, u32_)?);
                }
                words
            }
        };
        // The texel's first word, `((z · height + y) · width + x) · words`.
        let at = k.mul(z, height)?;
        let at = k.add(at, y)?;
        let at = k.mul(at, width)?;
        let at = k.add(at, x)?;
        let at = k.mulk(at, out.len() as u32)?;
        for (i, word) in (0u32..).zip(out) {
            let index = k.addk(at, i)?;
            let p = k.f.access_chain(texel_words, &[zero, index])?;
            k.f.store(p, word, MemAccess::default())?;
        }
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, decoded, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: BlockDecodeModule::LOCAL_SIZE,
            y: BlockDecodeModule::LOCAL_SIZE,
            z: 1,
        }],
    )?;
    b.finalize()
}
