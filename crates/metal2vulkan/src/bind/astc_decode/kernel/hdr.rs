//! The GPU's HDR half of the ASTC decode, in integer operations: the HDR colour endpoint modes'
//! endpoints and the 16-bit interpolation results' half-precision values, the CPU's
//! (`super::super::hdr`) step for step.

use super::K;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, GlslOp};

impl K<'_, '_> {
    fn sub_k(&mut self, a: Value, b: u32) -> BResult<Value> {
        self.opk(BinOp::ISub, a, b)
    }

    /// `v` clamped into 0 to `0xFFF`, as a signed value.
    fn clamp12(&mut self, v: Value) -> BResult<Value> {
        let zero = self.k(0)?;
        let top = self.k(0xFFF)?;
        self.f.glsl(GlslOp::SClamp, &[v, zero, top])
    }

    /// `v`'s low `bits` bits (a value) as a signed value.
    fn sign_extend(&mut self, v: Value, bits: Value) -> BResult<Value> {
        let thirty_two = self.k(32)?;
        let shift = self.sub(thirty_two, bits)?;
        let up = self.shl(v, shift)?;
        self.op(BinOp::ShiftRightArithmetic, up, shift)
    }

    /// Bit `at` of `v`.
    fn bit(&mut self, v: Value, at: u32) -> BResult<Value> {
        self.field(v, at, 1)
    }

    /// `x << at` where the one-hot mode `ohm` is in `mask`, else 0.
    fn place(&mut self, ohm: Value, mask: u32, x: Value, at: u32) -> BResult<Value> {
        let hit = self.andk(ohm, mask)?;
        let hit = self.opk(BinOp::INotEqual, hit, 0)?;
        let shifted = self.shlk(x, at)?;
        let zero = self.k(0)?;
        self.sel(hit, shifted, zero)
    }

    /// `v | place(...)` for each of `places` (mask, bit, place).
    fn placed(&mut self, ohm: Value, v: Value, places: &[(u32, Value, u32)]) -> BResult<Value> {
        let mut v = v;
        for &(mask, x, at) in places {
            let p = self.place(ohm, mask, x, at)?;
            v = self.or(v, p)?;
        }
        Ok(v)
    }

    /// Entry `index` (0 to 7) of the constant `table`.
    fn pick(&mut self, table: &[u32], index: Value) -> BResult<Value> {
        let mut out = self.k(table[0])?;
        for (i, &v) in (0u32..).zip(table).skip(1) {
            let here = self.eqk(index, i)?;
            let v = self.k(v)?;
            out = self.sel(here, v, out)?;
        }
        Ok(out)
    }

    /// Swaps channel 0 with channel `with` of `e` where `c` holds.
    fn swap_where(&mut self, c: Value, e: [Value; 3], with: usize) -> BResult<[Value; 3]> {
        let mut out = e;
        out[0] = self.sel(c, e[with], e[0])?;
        out[with] = self.sel(c, e[0], e[with])?;
        Ok(out)
    }

    /// Mode 7: HDR RGB, a base and a scale (`hdr::rgb_scale`).
    fn rgb_scale(&mut self, v: [Value; 8]) -> BResult<[[Value; 3]; 2]> {
        let a = self.andk(v[0], 0xC0)?;
        let a = self.shrk(a, 6)?;
        let b = self.andk(v[1], 0x80)?;
        let b = self.shrk(b, 5)?;
        let c = self.andk(v[2], 0x80)?;
        let c = self.shrk(c, 4)?;
        let modeval = self.or(a, b)?;
        let modeval = self.or(modeval, c)?;
        let high = self.andk(modeval, 0xC)?;
        let first = self.opk(BinOp::INotEqual, high, 0xC)?;
        let last = self.eqk(modeval, 0xF)?;
        let major_first = self.shrk(modeval, 2)?;
        let low = self.andk(modeval, 3)?;
        let zero = self.k(0)?;
        let four = self.k(4)?;
        let five = self.k(5)?;
        let major = self.sel(last, zero, low)?;
        let major = self.sel(first, major_first, major)?;
        let mode = self.sel(last, five, four)?;
        let mode = self.sel(first, low, mode)?;
        let red = self.andk(v[0], 0x3F)?;
        let green = self.andk(v[1], 0x1F)?;
        let blue = self.andk(v[2], 0x1F)?;
        let scale = self.andk(v[3], 0x1F)?;
        let x0 = self.bit(v[1], 6)?;
        let x1 = self.bit(v[1], 5)?;
        let x2 = self.bit(v[2], 6)?;
        let x3 = self.bit(v[2], 5)?;
        let x4 = self.bit(v[3], 7)?;
        let x5 = self.bit(v[3], 6)?;
        let x6 = self.bit(v[3], 5)?;
        let one = self.k(1)?;
        let ohm = self.shl(one, mode)?;
        let green = self.placed(ohm, green, &[(0x30, x0, 6), (0x3A, x1, 5)])?;
        let blue = self.placed(ohm, blue, &[(0x30, x2, 6), (0x3A, x3, 5)])?;
        let scale = self.placed(ohm, scale, &[(0x3D, x6, 5), (0x2D, x5, 6), (0x04, x4, 7)])?;
        let red = self.placed(
            ohm,
            red,
            &[
                (0x3B, x4, 6),
                (0x04, x3, 6),
                (0x10, x5, 7),
                (0x0F, x2, 7),
                (0x05, x1, 8),
                (0x0A, x0, 8),
                (0x05, x0, 9),
                (0x02, x6, 9),
                (0x01, x3, 10),
                (0x02, x5, 10),
            ],
        )?;
        let shift = self.pick(&[1, 1, 2, 3, 4, 5], mode)?;
        let red = self.shl(red, shift)?;
        let green = self.shl(green, shift)?;
        let blue = self.shl(blue, shift)?;
        let scale = self.shl(scale, shift)?;
        let not_five = self.opk(BinOp::INotEqual, mode, 5)?;
        let green_less = self.sub(red, green)?;
        let blue_less = self.sub(red, blue)?;
        let green = self.sel(not_five, green_less, green)?;
        let blue = self.sel(not_five, blue_less, blue)?;
        let e = [red, green, blue];
        let is_one = self.eqk(major, 1)?;
        let e = self.swap_where(is_one, e, 1)?;
        let is_two = self.eqk(major, 2)?;
        let e = self.swap_where(is_two, e, 2)?;
        let mut e0 = e;
        let mut e1 = e;
        for (low, high) in e0.iter_mut().zip(e1.iter_mut()) {
            let less = self.sub(*low, scale)?;
            *low = self.clamp12(less)?;
            *high = self.clamp12(*high)?;
        }
        Ok([e0, e1])
    }

    /// Modes 11, 14 and 15's colour: HDR RGB, direct or a base and offsets (`hdr::rgb`).
    fn rgb(&mut self, v: [Value; 8]) -> BResult<[[Value; 3]; 2]> {
        let m4 = self.andk(v[4], 0x80)?;
        let m4 = self.shrk(m4, 7)?;
        let m5 = self.andk(v[5], 0x80)?;
        let m5 = self.shrk(m5, 6)?;
        let major = self.or(m4, m5)?;
        // Direct (`major` 3).
        let low4 = self.andk(v[4], 0x7F)?;
        let low5 = self.andk(v[5], 0x7F)?;
        let direct = [
            [
                self.shlk(v[0], 4)?,
                self.shlk(v[2], 4)?,
                self.shlk(low4, 5)?,
            ],
            [
                self.shlk(v[1], 4)?,
                self.shlk(v[3], 4)?,
                self.shlk(low5, 5)?,
            ],
        ];
        // A base and offsets.
        let b1 = self.andk(v[1], 0x80)?;
        let b1 = self.shrk(b1, 7)?;
        let b2 = self.andk(v[2], 0x80)?;
        let b2 = self.shrk(b2, 6)?;
        let b3 = self.andk(v[3], 0x80)?;
        let b3 = self.shrk(b3, 5)?;
        let mode = self.or(b1, b2)?;
        let mode = self.or(mode, b3)?;
        let high = self.andk(v[1], 0x40)?;
        let high = self.shlk(high, 2)?;
        let a = self.or(v[0], high)?;
        let b0 = self.andk(v[2], 0x3F)?;
        let b1 = self.andk(v[3], 0x3F)?;
        let c = self.andk(v[1], 0x3F)?;
        let d_bits = self.pick(&[7, 6, 7, 6, 5, 6, 5, 6], mode)?;
        let d0 = self.sign_extend(low4, d_bits)?;
        let d1 = self.sign_extend(low5, d_bits)?;
        let x0 = self.bit(v[2], 6)?;
        let x1 = self.bit(v[3], 6)?;
        let x2 = self.bit(v[4], 6)?;
        let x3 = self.bit(v[5], 6)?;
        let x4 = self.bit(v[4], 5)?;
        let x5 = self.bit(v[5], 5)?;
        let one = self.k(1)?;
        let ohm = self.shl(one, mode)?;
        let a = self.placed(
            ohm,
            a,
            &[
                (0xA4, x0, 9),
                (0x08, x2, 9),
                (0x50, x4, 9),
                (0x50, x5, 10),
                (0xA0, x1, 10),
                (0xC0, x2, 11),
            ],
        )?;
        let c = self.placed(ohm, c, &[(0x04, x1, 6), (0xE8, x3, 6), (0x20, x2, 7)])?;
        let b0 = self.placed(ohm, b0, &[(0x5B, x0, 6), (0x12, x2, 7)])?;
        let b1 = self.placed(ohm, b1, &[(0x5B, x1, 6), (0x12, x3, 7)])?;
        let shift = self.shrk(mode, 1)?;
        let shift = self.opk(BinOp::BitwiseXor, shift, 3)?;
        let [a, b0, b1, c, d0, d1] = [a, b0, b1, c, d0, d1].map(|x| (x, shift));
        let a = self.shl(a.0, a.1)?;
        let b0 = self.shl(b0.0, b0.1)?;
        let b1 = self.shl(b1.0, b1.1)?;
        let c = self.shl(c.0, c.1)?;
        let d0 = self.shl(d0.0, d0.1)?;
        let d1 = self.shl(d1.0, d1.1)?;
        let g1 = self.sub(a, b0)?;
        let bl1 = self.sub(a, b1)?;
        let r0 = self.sub(a, c)?;
        let g0 = self.sub(g1, c)?;
        let g0 = self.sub(g0, d0)?;
        let bl0 = self.sub(bl1, c)?;
        let bl0 = self.sub(bl0, d1)?;
        let mut e0 = [r0, g0, bl0];
        let mut e1 = [a, g1, bl1];
        for ch in e0.iter_mut().chain(e1.iter_mut()) {
            *ch = self.clamp12(*ch)?;
        }
        let is_one = self.eqk(major, 1)?;
        e0 = self.swap_where(is_one, e0, 1)?;
        e1 = self.swap_where(is_one, e1, 1)?;
        let is_two = self.eqk(major, 2)?;
        e0 = self.swap_where(is_two, e0, 2)?;
        e1 = self.swap_where(is_two, e1, 2)?;
        let is_direct = self.eqk(major, 3)?;
        let mut out = [e0, e1];
        for (end, direct) in out.iter_mut().zip(direct) {
            for (ch, d) in end.iter_mut().zip(direct) {
                *ch = self.sel(is_direct, d, *ch)?;
            }
        }
        Ok(out)
    }

    /// Mode 15's alpha: HDR, direct or a base and an offset (`hdr::alpha`).
    fn hdr_alpha(&mut self, v6: Value, v7: Value) -> BResult<[Value; 2]> {
        let m6 = self.bit(v6, 7)?;
        let m7 = self.shrk(v7, 6)?;
        let m7 = self.andk(m7, 2)?;
        let mode = self.or(m6, m7)?;
        let a0 = self.andk(v6, 0x7F)?;
        let a1 = self.andk(v7, 0x7F)?;
        let direct = [self.shlk(a0, 5)?, self.shlk(a1, 5)?];
        let up = self.addk(mode, 1)?;
        let high = self.shl(a1, up)?;
        let high = self.andk(high, 0x780)?;
        let base = self.or(a0, high)?;
        let x3f = self.k(0x3F)?;
        let x20 = self.k(0x20)?;
        let keep = self.shr(x3f, mode)?;
        let flip = self.shr(x20, mode)?;
        let off = self.and(a1, keep)?;
        let off = self.op(BinOp::BitwiseXor, off, flip)?;
        let off = self.sub(off, flip)?;
        let four = self.k(4)?;
        let shift = self.sub(four, mode)?;
        let base = self.shl(base, shift)?;
        let off = self.shl(off, shift)?;
        let end = self.add(off, base)?;
        let end = self.clamp12(end)?;
        let is_direct = self.eqk(mode, 3)?;
        Ok([
            self.sel(is_direct, direct[0], base)?,
            self.sel(is_direct, direct[1], end)?,
        ])
    }

    /// The two endpoints of an HDR colour endpoint mode `cem` from its unquantized values `v`
    /// (`hdr::endpoints`): each channel a 12-bit logarithmic value but mode 14's alpha, an 8-bit
    /// LDR one; and whether alpha is logarithmic (not mode 14).
    pub(super) fn hdr_endpoints(
        &mut self,
        cem: Value,
        v: [Value; 8],
    ) -> BResult<([[Value; 4]; 2], Value)> {
        let one = self.k(0x780)?;
        // 2: luminance, large range.
        let up0 = self.shlk(v[0], 4)?;
        let up1 = self.shlk(v[1], 4)?;
        let ordered = self.op(BinOp::UGreaterThanEqual, v[1], v[0])?;
        let swapped0 = self.addk(up1, 8)?;
        let swapped1 = self.sub_k(up0, 8)?;
        let y0 = self.sel(ordered, up0, swapped0)?;
        let y1 = self.sel(ordered, up1, swapped1)?;
        let large = [[y0, y0, y0, one], [y1, y1, y1, one]];
        // 3: luminance, small range.
        let wide = self.andk(v[0], 0x80)?;
        let wide = self.opk(BinOp::INotEqual, wide, 0)?;
        let hi = self.andk(v[1], 0xE0)?;
        let hi = self.shlk(hi, 4)?;
        let lo = self.andk(v[0], 0x7F)?;
        let lo2 = self.shlk(lo, 2)?;
        let y_wide = self.or(hi, lo2)?;
        let d_wide = self.andk(v[1], 0x1F)?;
        let d_wide = self.shlk(d_wide, 2)?;
        let hi = self.andk(v[1], 0xF0)?;
        let hi = self.shlk(hi, 4)?;
        let lo1 = self.shlk(lo, 1)?;
        let y_narrow = self.or(hi, lo1)?;
        let d_narrow = self.andk(v[1], 0x0F)?;
        let d_narrow = self.shlk(d_narrow, 1)?;
        let y0 = self.sel(wide, y_wide, y_narrow)?;
        let d = self.sel(wide, d_wide, d_narrow)?;
        let y1 = self.add(y0, d)?;
        let top = self.k(0xFFF)?;
        let y1 = self.f.glsl(GlslOp::UMin, &[y1, top])?;
        let small = [[y0, y0, y0, one], [y1, y1, y1, one]];
        // 7: RGB, a base and a scale.
        let [s0, s1] = self.rgb_scale(v)?;
        let scaled = [[s0[0], s0[1], s0[2], one], [s1[0], s1[1], s1[2], one]];
        // 11, 14 and 15: RGB, and LDR or HDR alpha.
        let [c0, c1] = self.rgb(v)?;
        let [h0, h1] = self.hdr_alpha(v[6], v[7])?;
        let is_eleven = self.eqk(cem, 11)?;
        let is_fourteen = self.eqk(cem, 14)?;
        let a0 = self.sel(is_fourteen, v[6], h0)?;
        let a1 = self.sel(is_fourteen, v[7], h1)?;
        let a0 = self.sel(is_eleven, one, a0)?;
        let a1 = self.sel(is_eleven, one, a1)?;
        let color = [[c0[0], c0[1], c0[2], a0], [c1[0], c1[1], c1[2], a1]];
        let mut out = color;
        for (cem_value, ends) in [(7, scaled), (3, small), (2, large)] {
            let here = self.eqk(cem, cem_value)?;
            out = [
                self.sel4(here, ends[0], out[0])?,
                self.sel4(here, ends[1], out[1])?,
            ];
        }
        let alpha_lns = self.not(is_fourteen)?;
        Ok((out, alpha_lns))
    }

    /// `hdr::lns_to_half` of the 16-bit `c`.
    pub(super) fn lns_to_half(&mut self, c: Value) -> BResult<Value> {
        let e = self.shrk(c, 11)?;
        let m = self.andk(c, 0x7FF)?;
        let three = self.mulk(m, 3)?;
        let five = self.mulk(m, 5)?;
        let five = self.sub_k(five, 2048)?;
        let four = self.mulk(m, 4)?;
        let four = self.sub_k(four, 512)?;
        let low = self.opk(BinOp::ULessThan, m, 512)?;
        let high = self.opk(BinOp::UGreaterThanEqual, m, 1536)?;
        let mt = self.sel(high, five, four)?;
        let mt = self.sel(low, three, mt)?;
        let e = self.shlk(e, 10)?;
        let mt = self.shrk(mt, 3)?;
        let half = self.add(e, mt)?;
        let top = self.k(0x7BFF)?;
        self.f.glsl(GlslOp::UMin, &[half, top])
    }

    /// `hdr::unorm16_to_half` of the 16-bit `c`.
    pub(super) fn unorm16_to_half(&mut self, c: Value) -> BResult<Value> {
        let lead = self.f.glsl(GlslOp::FindUMsb, &[c])?;
        let exponent = self.sub_k(lead, 1)?;
        let exponent = self.shlk(exponent, 10)?;
        let thirty_one = self.k(31)?;
        let up = self.sub(thirty_one, lead)?;
        let up = self.andk(up, 31)?;
        let mantissa = self.shl(c, up)?;
        let mantissa = self.shrk(mantissa, 21)?;
        let mantissa = self.andk(mantissa, 0x3FF)?;
        let normal = self.or(exponent, mantissa)?;
        let subnormal = self.shlk(c, 8)?;
        let small = self.opk(BinOp::ULessThan, c, 4)?;
        let half = self.sel(small, subnormal, normal)?;
        let one = self.k(0x3C00)?;
        let all = self.eqk(c, 0xFFFF)?;
        self.sel(all, one, half)
    }

    /// The half `h` (Metal's quiet NaN for every NaN): a void-extent block's HDR value.
    pub(super) fn quiet_nan(&mut self, h: Value) -> BResult<Value> {
        let magnitude = self.andk(h, 0x7FFF)?;
        let nan = self.opk(BinOp::UGreaterThan, magnitude, 0x7C00)?;
        let quiet = self.k(super::super::HDR_ERROR.into())?;
        self.sel(nan, quiet, h)
    }

    /// The `f32` bits of the half `h`, exactly (`hdr::half_value`).
    pub(super) fn half_bits_to_f32(&mut self, h: Value) -> BResult<Value> {
        let sign = self.andk(h, 0x8000)?;
        let sign = self.shlk(sign, 16)?;
        let exponent = self.field(h, 10, 5)?;
        let mantissa = self.andk(h, 0x3FF)?;
        let wide = self.shlk(mantissa, 13)?;
        // A normal half: the exponent rebased.
        let rebased = self.addk(exponent, 112)?;
        let rebased = self.shlk(rebased, 23)?;
        let normal = self.or(rebased, wide)?;
        // An infinity or a NaN.
        let special = self.opk(BinOp::BitwiseOr, wide, 0x7F80_0000)?;
        // A subnormal (or zero): `mantissa · 2^-24`, normalized by its leading one.
        let lead = self.f.glsl(GlslOp::FindUMsb, &[mantissa])?;
        let exponent_bits = self.addk(lead, 103)?;
        let exponent_bits = self.shlk(exponent_bits, 23)?;
        let ten = self.k(10)?;
        let up = self.sub(ten, lead)?;
        let up = self.andk(up, 31)?;
        let fraction = self.shl(mantissa, up)?;
        let fraction = self.andk(fraction, 0x3FF)?;
        let fraction = self.shlk(fraction, 13)?;
        let subnormal = self.or(exponent_bits, fraction)?;
        let zero = self.k(0)?;
        let none = self.eqk(mantissa, 0)?;
        let subnormal = self.sel(none, zero, subnormal)?;
        let is_zero = self.eqk(exponent, 0)?;
        let is_special = self.eqk(exponent, 31)?;
        let bits = self.sel(is_special, special, normal)?;
        let bits = self.sel(is_zero, subnormal, bits)?;
        self.or(sign, bits)
    }
}
