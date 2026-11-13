//! Rasterization-rate map decoding (Apple's layout).
//!
//! `rasterization_rate_map_decoder`'s `map_screen_to_physical_coordinates` and
//! `map_physical_to_screen_coordinates` are AIR's opaque `air.map_*_coordinates.v2f32.p2i8.i32`
//! over the `constant rasterization_rate_map_data &` buffer, which holds Metal's parameter data
//! in Apple's layout (`rate_map_data`). Each call becomes Apple's GPU decoder in plain AIR, here,
//! after SCCP: word loads through the call's own map pointer (`getelementptr i32` and `load`, so
//! memory planning and containment treat them as any other constant-buffer read), the table and
//! entry arithmetic (a horizontal table `layer · 1026` words past layer 0's, unclamped), the
//! granule index (screen to physical: of `c + 1/2`), the boundary byte's test, and per axis an
//! `fmul` and `fadd` (screen to physical) or an `fsub` and `fmul` (physical to screen), all without
//! fast-math flags, so the module's result is `rate_map_data`'s `shader_screen_to_physical` and
//! `shader_physical_to_screen` bit for bit: Apple's GPU decoder's, measured on an Apple M4 under
//! fast and safe math alike, where the device reads `float` denormals and answers `NaN` as Metal
//! does. The lowering gives these operations, as every `float` `+ - *`, the device's denormals and
//! `NaN`: Metal's on a device measured to read them so, the device's own elsewhere, noted. The
//! call's fast-math flags are not carried: the decoder's arithmetic is Apple's, not the source's.
//!
//! The `uint2` overloads (`.v2i32`) become Apple's GPU answers too, measured on an Apple M3 Max
//! (`rate_map_data`'s `shader_screen_to_physical_integer` and `shader_physical_to_screen_integer`):
//! screen to physical, per axis, the `float2` decoder at the pixel's centre (`uitofp` of the
//! coordinate plus 1/2) and the saturating conversion of its answer to `uint`; physical to screen,
//! the entry of granule `c >> 5` and the bits of `(0 - offset) · reciprocal`, the coordinate's
//! value not read, as Apple's GPU answers.

use crate::budget::Meter;
use crate::error::Error;
use crate::intrinsics::{AirIntrinsic, Fragment};
use crate::ir::{
    AddrSpace, Align, BinOp, Block, Body, Call, Callee, CastOp, Const, ConstId, Constant, FastMath,
    FloatPredicate, Idx, Inst, InstId, InstKind, IntFlags, IntPredicate, ModuleParts, Type, TypeId,
    Value,
};

use super::rate_map_data as format;
use super::rewrite::invariant;

/// The direction of a decoded call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    ToPhysical,
    ToScreen,
}

/// What a decoded call maps: its direction, and whether it is a `uint2` overload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    dir: Direction,
    integer: bool,
}

impl Shape {
    const ALL: [Shape; 4] = [
        Shape {
            dir: Direction::ToPhysical,
            integer: false,
        },
        Shape {
            dir: Direction::ToScreen,
            integer: false,
        },
        Shape {
            dir: Direction::ToPhysical,
            integer: true,
        },
        Shape {
            dir: Direction::ToScreen,
            integer: true,
        },
    ];
}

/// The call's decoded shape, when it is a rate-map call.
fn decoded(kind: &InstKind) -> Option<(Shape, &Call)> {
    let InstKind::Call(call) = kind else {
        return None;
    };
    match call.callee {
        Callee::Air {
            op:
                AirIntrinsic::Fragment(Fragment::MapCoordinates {
                    to_physical,
                    integer,
                }),
            ..
        } => {
            let dir = if to_physical {
                Direction::ToPhysical
            } else {
                Direction::ToScreen
            };
            Some((Shape { dir, integer }, call))
        }
        _ => None,
    }
}

/// The integer constants the decoder uses.
const INTS: [u32; 12] = [
    0,
    1,
    u32::MAX,
    format::ENTRY_WORDS,
    format::GRANULE_SHIFT,
    format::OFFSET_SHIFT,
    format::NO_BOUNDARY,
    format::BOUNDARY_MASK,
    format::TABLE_WORDS,
    format::HORIZONTAL_TO_PHYSICAL,
    format::VERTICAL_TO_SCREEN,
    format::VERTICAL_TO_PHYSICAL,
];

/// The types and constants the decoder uses.
struct Vocab {
    i32t: TypeId,
    i64t: TypeId,
    f32t: TypeId,
    i1t: TypeId,
    ptr: TypeId,
    v2f32: TypeId,
    v2i32: TypeId,
    ints: Vec<(u32, ConstId)>,
    zero_f: ConstId,
    saturation_f: ConstId,
    half_f: ConstId,
    inv_granularity_f: ConstId,
    granularity_f: ConstId,
    two_f: ConstId,
    offset_unit_f: ConstId,
    poison_v2: ConstId,
    poison_v2i: ConstId,
}

impl Vocab {
    fn new(parts: &mut ModuleParts, m: &mut Meter) -> Result<Vocab, Error> {
        let i32t = parts.types.intern(Type::Int(32), m)?;
        let i64t = parts.types.intern(Type::Int(64), m)?;
        let f32t = parts.types.intern(Type::Float, m)?;
        let i1t = parts.types.intern(Type::Int(1), m)?;
        let ptr = parts.types.intern(
            Type::Ptr {
                space: AddrSpace::Constant,
            },
            m,
        )?;
        let v2f32 = parts.types.intern(Type::Vector { elem: f32t, len: 2 }, m)?;
        let v2i32 = parts.types.intern(Type::Vector { elem: i32t, len: 2 }, m)?;
        let mut c = |ty, value| parts.consts.intern(Constant { ty, value }, m);
        let mut ints = Vec::new();
        for v in INTS {
            ints.push((v, c(i32t, Const::Int(u64::from(v)))?));
        }
        let mut f = |v: f32| c(f32t, Const::Float(u64::from(v.to_bits())));
        Ok(Vocab {
            i32t,
            i64t,
            f32t,
            i1t,
            ptr,
            v2f32,
            v2i32,
            ints,
            zero_f: f(0.0)?,
            saturation_f: f(format::SHADER_INDEX_SATURATION)?,
            half_f: f(format::SHADER_GRANULE_OFFSET)?,
            inv_granularity_f: f(format::INV_GRANULARITY)?,
            granularity_f: f(format::GRANULARITY as f32)?,
            two_f: f(2.0)?,
            offset_unit_f: f(format::OFFSET_UNIT)?,
            poison_v2: c(v2f32, Const::Poison)?,
            poison_v2i: c(v2i32, Const::Poison)?,
        })
    }

    fn int(&self, v: u32) -> Result<Value, Error> {
        self.ints
            .iter()
            .find(|&&(w, _)| w == v)
            .map(|&(_, c)| Value::Const(c))
            .ok_or_else(|| invariant("a decoder constant not interned"))
    }
}

/// An instruction sink: the output arena.
struct Sink<'a> {
    out: &'a mut Body,
    v: &'a Vocab,
}

impl Sink<'_> {
    fn push(&mut self, ty: TypeId, kind: InstKind, m: &mut Meter) -> Result<Value, Error> {
        let id = self.out.push_inst(Inst { ty, kind }, m)?;
        Ok(Value::Inst(id))
    }

    fn bin(
        &mut self,
        ty: TypeId,
        op: BinOp,
        a: Value,
        b: Value,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let kind = InstKind::BinOp {
            op,
            flags: IntFlags::NONE,
            fmf: FastMath::NONE,
            a,
            b,
        };
        self.push(ty, kind, m)
    }

    /// An `i32` operation of `a` and the constant `k`.
    fn int_k(&mut self, op: BinOp, a: Value, k: u32, m: &mut Meter) -> Result<Value, Error> {
        let k = self.v.int(k)?;
        self.bin(self.v.i32t, op, a, k, m)
    }

    /// An `f32` operation of `a` and `b`.
    fn float(&mut self, op: BinOp, a: Value, b: Value, m: &mut Meter) -> Result<Value, Error> {
        self.bin(self.v.f32t, op, a, b, m)
    }

    fn cast(&mut self, op: CastOp, ty: TypeId, v: Value, m: &mut Meter) -> Result<Value, Error> {
        self.push(ty, InstKind::Cast { op, v }, m)
    }

    fn select(
        &mut self,
        ty: TypeId,
        c: Value,
        t: Value,
        f: Value,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let kind = InstKind::Select {
            c,
            t,
            f,
            fmf: FastMath::NONE,
        };
        self.push(ty, kind, m)
    }

    fn icmp(
        &mut self,
        pred: IntPredicate,
        a: Value,
        b: Value,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        self.push(self.v.i1t, InstKind::ICmp { pred, a, b }, m)
    }

    fn fcmp(
        &mut self,
        pred: FloatPredicate,
        a: Value,
        b: Value,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let kind = InstKind::FCmp {
            pred,
            fmf: FastMath::NONE,
            a,
            b,
        };
        self.push(self.v.i1t, kind, m)
    }

    /// Word `index` of the map, as `ty`.
    fn load(
        &mut self,
        map: Value,
        index: Value,
        ty: TypeId,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let wide = self.cast(CastOp::ZExt, self.v.i64t, index, m)?;
        let gep = InstKind::Gep {
            src_ty: self.v.i32t,
            base: map,
            indices: Box::new([wide]),
            inbounds: false,
        };
        let at = self.push(self.v.ptr, gep, m)?;
        let load = InstKind::Load {
            ptr: at,
            align: Align::from_bytes(4),
            volatile: false,
            atomic: None,
            range: None,
        };
        self.push(ty, load, m)
    }

    /// `rate_map_data::shader_index(t)`: `NaN` and below 0 are 0, from `2^32` on `u32::MAX`, and
    /// the conversion sees only values it converts exactly.
    fn index(&mut self, t: Value, m: &mut Meter) -> Result<Value, Error> {
        let (f32t, zero, saturation) = (
            self.v.f32t,
            Value::Const(self.v.zero_f),
            Value::Const(self.v.saturation_f),
        );
        let ge = self.fcmp(FloatPredicate::Oge, t, zero, m)?;
        let a = self.select(f32t, ge, t, zero, m)?;
        let saturated = self.fcmp(FloatPredicate::Oge, a, saturation, m)?;
        let a = self.select(f32t, saturated, zero, a, m)?;
        let i = self.cast(CastOp::FPToUI, self.v.i32t, a, m)?;
        let max = self.v.int(u32::MAX)?;
        self.select(self.v.i32t, saturated, max, i, m)
    }

    /// The word of entry `j`'s first word in the table at word `table`.
    fn entry(&mut self, table: Value, j: Value, m: &mut Meter) -> Result<Value, Error> {
        let at = self.int_k(BinOp::Mul, j, format::ENTRY_WORDS, m)?;
        self.bin(self.v.i32t, BinOp::Add, table, at, m)
    }

    /// One axis of the decoder over the table at word `table`.
    fn axis(
        &mut self,
        dir: Direction,
        map: Value,
        table: Value,
        c: Value,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let (i32t, f32t) = (self.v.i32t, self.v.f32t);
        let inv_g = Value::Const(self.v.inv_granularity_f);
        match dir {
            Direction::ToPhysical => {
                // A shader looks up the granule of the screen coordinate plus half a pixel.
                let half = Value::Const(self.v.half_f);
                let h = self.float(BinOp::FAdd, c, half, m)?;
                let t = self.float(BinOp::FMul, h, inv_g, m)?;
                let j = self.index(t, m)?;
                // The boundary byte: past it, the next entry's line.
                let at = self.entry(table, j, m)?;
                let at = self.int_k(BinOp::Add, at, 1, m)?;
                let packed = self.load(map, at, i32t, m)?;
                let byte = self.int_k(BinOp::And, packed, format::NO_BOUNDARY, m)?;
                let none = self.v.int(format::NO_BOUNDARY)?;
                let some = self.icmp(IntPredicate::Ne, byte, none, m)?;
                // Its low seven bits are compared (`rate_map_data::BOUNDARY_MASK`).
                let boundary = self.int_k(BinOp::And, packed, format::BOUNDARY_MASK, m)?;
                let jf = self.cast(CastOp::UIToFP, f32t, j, m)?;
                let g = Value::Const(self.v.granularity_f);
                let start = self.float(BinOp::FMul, jf, g, m)?;
                let into = self.float(BinOp::FSub, c, start, m)?;
                let two = Value::Const(self.v.two_f);
                let halves = self.float(BinOp::FMul, into, two, m)?;
                let bf = self.cast(CastOp::UIToFP, f32t, boundary, m)?;
                let past = self.fcmp(FloatPredicate::Oge, halves, bf, m)?;
                let next = self.bin(self.v.i1t, BinOp::And, past, some, m)?;
                let j1 = self.int_k(BinOp::Add, j, 1, m)?;
                let j = self.select(i32t, next, j1, j, m)?;
                let at = self.entry(table, j, m)?;
                let scale = self.load(map, at, f32t, m)?;
                let at = self.int_k(BinOp::Add, at, 1, m)?;
                let packed = self.load(map, at, i32t, m)?;
                let units = self.int_k(BinOp::AShr, packed, format::OFFSET_SHIFT, m)?;
                let units = self.cast(CastOp::SIToFP, f32t, units, m)?;
                let unit = Value::Const(self.v.offset_unit_f);
                let offset = self.float(BinOp::FMul, units, unit, m)?;
                let q = self.float(BinOp::FMul, c, scale, m)?;
                self.float(BinOp::FAdd, q, offset, m)
            }
            Direction::ToScreen => {
                let t = self.float(BinOp::FMul, c, inv_g, m)?;
                let k = self.index(t, m)?;
                let at = self.entry(table, k, m)?;
                let reciprocal = self.load(map, at, f32t, m)?;
                let at = self.int_k(BinOp::Add, at, 1, m)?;
                let offset = self.load(map, at, f32t, m)?;
                let d = self.float(BinOp::FSub, c, offset, m)?;
                self.float(BinOp::FMul, d, reciprocal, m)
            }
        }
    }

    /// One axis of a `uint2` overload over the table at word `table`, of the `uint` coordinate
    /// `c` (`rate_map_data`'s `shader_screen_to_physical_integer` and
    /// `shader_physical_to_screen_integer`).
    fn integer_axis(
        &mut self,
        dir: Direction,
        map: Value,
        table: Value,
        c: Value,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let (i32t, f32t) = (self.v.i32t, self.v.f32t);
        match dir {
            Direction::ToPhysical => {
                // The `float2` decoder at the pixel's centre, its answer converted saturating.
                let cf = self.cast(CastOp::UIToFP, f32t, c, m)?;
                let half = Value::Const(self.v.half_f);
                let centre = self.float(BinOp::FAdd, cf, half, m)?;
                let r = self.axis(Direction::ToPhysical, map, table, centre, m)?;
                self.index(r, m)
            }
            Direction::ToScreen => {
                // The granule's entry; the coordinate's value is not read.
                let k = self.int_k(BinOp::LShr, c, format::GRANULE_SHIFT, m)?;
                let at = self.entry(table, k, m)?;
                let reciprocal = self.load(map, at, f32t, m)?;
                let at = self.int_k(BinOp::Add, at, 1, m)?;
                let offset = self.load(map, at, f32t, m)?;
                let zero = Value::Const(self.v.zero_f);
                let d = self.float(BinOp::FSub, zero, offset, m)?;
                let r = self.float(BinOp::FMul, d, reciprocal, m)?;
                self.cast(CastOp::Bitcast, i32t, r, m)
            }
        }
    }

    /// The decoder of one call of `shape`: coordinate `c`, map pointer `map`, layer `layer`. Its
    /// last instruction is the result.
    fn decode(
        &mut self,
        shape: Shape,
        c: Value,
        map: Value,
        layer: Value,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let Shape { dir, integer } = shape;
        let to_physical = dir == Direction::ToPhysical;
        // The layer's horizontal table, unclamped (`rate_map_data::table`); the vertical one is
        // every layer's.
        let offset = self.int_k(BinOp::Mul, layer, format::TABLE_WORDS, m)?;
        let horizontal = self.int_k(BinOp::Add, offset, format::table(0, 0, to_physical), m)?;
        let vertical = self.v.int(format::table(0, 1, to_physical))?;
        let (elem, vec, mut v) = match integer {
            false => (self.v.f32t, self.v.v2f32, Value::Const(self.v.poison_v2)),
            true => (self.v.i32t, self.v.v2i32, Value::Const(self.v.poison_v2i)),
        };
        for (axis, table) in [(0u32, horizontal), (1, vertical)] {
            let lane = self.v.int(axis)?;
            let ca = self.push(elem, InstKind::ExtractElement { vec: c, idx: lane }, m)?;
            let r = match integer {
                false => self.axis(dir, map, table, ca, m)?,
                true => self.integer_axis(dir, map, table, ca, m)?,
            };
            v = self.push(
                vec,
                InstKind::InsertElement {
                    vec: v,
                    elt: r,
                    idx: lane,
                },
                m,
            )?;
        }
        Ok(v)
    }
}

/// The number of instructions one call of `shape` becomes.
fn decoder_len(shape: Shape, v: &Vocab, m: &mut Meter) -> Result<u32, Error> {
    let mut scratch = Body::default();
    let mut s = Sink {
        out: &mut scratch,
        v,
    };
    let any = Value::Const(v.poison_v2);
    s.decode(shape, any, any, any, m)?;
    let len = scratch.insts.len();
    scratch.release(m);
    u32::try_from(len).map_err(|_| invariant("decoder length"))
}

/// Replace every rate-map call of `body` by its decoder.
pub(super) fn expand(body: Body, parts: &mut ModuleParts, m: &mut Meter) -> Result<Body, Error> {
    if !body.insts.iter().any(|(_, i)| decoded(&i.kind).is_some()) {
        return Ok(body);
    }
    let v = Vocab::new(parts, m)?;
    let mut lens = [0u32; Shape::ALL.len()];
    for (len, shape) in lens.iter_mut().zip(Shape::ALL) {
        *len = decoder_len(shape, &v, m)?;
    }
    let len_of = |shape: Shape| {
        (Shape::ALL.iter().zip(lens))
            .find(|&(s, _)| *s == shape)
            .map(|(_, len)| len)
            .ok_or_else(|| invariant("a decoder shape not measured"))
    };
    // Each old instruction's id in the output: a decoded call's is its decoder's last.
    let mut result = Vec::new();
    m.reserve(&mut result, body.insts.len())?;
    let mut next = 0u32;
    for (_, inst) in body.insts.iter() {
        let n = match decoded(&inst.kind) {
            Some((shape, _)) => len_of(shape)?,
            None => 1,
        };
        next = next
            .checked_add(n)
            .ok_or_else(|| invariant("instruction ids"))?;
        result.push(InstId::new(next - 1));
    }
    let map = |val: Value| -> Result<Value, Error> {
        match val {
            Value::Inst(i) => result
                .get(i.index())
                .map(|&n| Value::Inst(n))
                .ok_or_else(|| invariant("an operand out of range")),
            other => Ok(other),
        }
    };
    let mut out = Body::default();
    for (_, block) in body.blocks.iter() {
        m.tick(1)?;
        let start = InstId::new(u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?);
        for i in block.inst_ids() {
            let inst = body
                .insts
                .get(i)
                .ok_or_else(|| invariant("instruction out of range"))?;
            match decoded(&inst.kind) {
                Some((shape, call)) => {
                    let [c, ptr, layer] = call.args.as_ref() else {
                        return Err(invariant("a rate-map call without three operands"));
                    };
                    let (c, ptr, layer) = (map(*c)?, map(*ptr)?, map(*layer)?);
                    let mut s = Sink {
                        out: &mut out,
                        v: &v,
                    };
                    s.decode(shape, c, ptr, layer, m)?;
                }
                None => {
                    let mut kind = inst.kind.clone();
                    let mut failed = None;
                    kind.for_each_value_mut(|val| match map(*val) {
                        Ok(n) => *val = n,
                        Err(e) => {
                            failed.get_or_insert(e);
                        }
                    });
                    if let Some(e) = failed {
                        return Err(e);
                    }
                    out.push_inst(Inst { ty: inst.ty, kind }, m)?;
                }
            }
            let want = result.get(i.index()).map(|r| r.index() + 1);
            if want != Some(out.insts.len()) {
                return Err(invariant("the decoder's length differs from its measure"));
            }
        }
        let end = InstId::new(u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?);
        out.blocks.push(
            Block {
                start,
                end,
                name: block.name,
            },
            m,
        )?;
    }
    m.release(result.len().saturating_mul(std::mem::size_of::<InstId>()));
    // The body the decoders replaced is freed.
    body.release(m);
    Ok(out)
}
