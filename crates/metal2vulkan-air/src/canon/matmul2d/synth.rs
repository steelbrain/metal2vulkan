//! The bodies [`super::expand`] gives the calls it lowers, built as plain AIR.
//!
//! Each lowered call becomes a call to a function built here, whose parameters are the call's
//! arguments followed, when the body reads them ([`needs_lanes`]), by the lane's index in its
//! simdgroup and the simdgroup's index in the threadgroup. An argument the call site knows (a
//! constant, or an integer [`super::eval`] reads) is that constant in the body. The inliner then
//! takes the function like any other callee. What a body decides from the call's constant
//! arguments (the descriptor, the operand, the element types, the scope) is decided by
//! [`Helper::of_call`] and [`Run::of_call`], from the caller's values.

use crate::budget::Meter;
use crate::error::DiagName;
use crate::error::{Error, Feature, Unsupported};
use crate::intrinsics::Tensor;
use crate::ir::{
    AddrSpace, ArgIdx, BinOp, CallConv, CastOp, Const, FnAttrs, FuncId, Function, IntPredicate,
    Linkage, ModuleParts, Param, ParamAttrs, Type, TypeId, UnnamedAddr, Value, Visibility,
};
use crate::tensor_ops::{
    DataType, Descriptor, DescriptorType, Inner, MatMul2dFn, Mode, OperandForm, OperandIndex,
    RunForm, TensorSpace, DYNAMIC_EXTENT, NO_SCALE, RESERVED_INDEX,
};

use super::super::out::Out;
use super::super::rewrite::invariant;
use super::layout::{lane_place, Arith, Holds, Layout};
use super::tensor;

/// A refusal of a `matmul2d` or tensor form this lowering does not model.
pub(super) fn refuse(what: impl Into<String>) -> Error {
    let what: String = what.into();
    Unsupported::not_implemented(Feature::Tensors)
        .naming(DiagName::new(what))
        .into()
}

/// The argument positions of a cooperative-tensor helper (the header's 26.2 declarations): the
/// element index, the three element types (left, right, destination; `None` where the form
/// passes fewer), and the scope's thread count.
struct Positions {
    index: Option<usize>,
    types: [Option<usize>; 3],
    threads: Option<usize>,
}

/// The storage argument of every helper that takes one.
const STORAGE: usize = 8;

/// `get_coordinate`'s coordinates pointer and coordinate type.
const COORDINATES: usize = 10;
const COORDINATE_TYPE: usize = 11;

fn positions(f: MatMul2dFn) -> Option<Positions> {
    let p = |index, types, threads| Positions {
        index,
        types,
        threads,
    };
    Some(match f {
        MatMul2dFn::TensorDataSize => p(None, [Some(8), Some(9), Some(10)], Some(11)),
        MatMul2dFn::TensorNumElements => p(None, [Some(9), Some(10), None], Some(11)),
        MatMul2dFn::TensorGetElementPointer => p(Some(9), [Some(10), Some(11), Some(12)], None),
        MatMul2dFn::TensorGetCoordinate => p(Some(9), [Some(13), Some(14), Some(15)], Some(12)),
        MatMul2dFn::TensorInit => p(None, [Some(9), Some(10), Some(11)], Some(12)),
        MatMul2dFn::TensorIsValidElement => p(Some(9), [Some(10), Some(11), Some(12)], Some(13)),
        MatMul2dFn::TensorGetElementIndex | MatMul2dFn::Run(_) | MatMul2dFn::Unmodeled => {
            return None
        }
    })
}

/// The bytes of one element of `t`, for the element types modeled.
fn elem_bytes(t: DataType) -> Option<u64> {
    match t {
        DataType::Float32 | DataType::Int32 => Some(4),
        DataType::Float16 | DataType::BFloat16 => Some(2),
        DataType::Int8 | DataType::Fp8E4M3 | DataType::Fp8E5M2 => Some(1),
        _ => None,
    }
}

/// Whether a product of these element types (left, right, destination) is modeled, as measured:
/// float, half, or bfloat inputs into float, 8-bit float inputs (both e4m3 or both e5m2) into
/// float, and `int8_t` inputs into `int32_t`.
fn modeled_types(types: [DataType; 3]) -> bool {
    matches!(
        types,
        [DataType::Float32, DataType::Float32, DataType::Float32]
            | [DataType::Float16, DataType::Float16, DataType::Float32]
            | [DataType::BFloat16, DataType::BFloat16, DataType::Float32]
            | [DataType::Fp8E4M3, DataType::Fp8E4M3, DataType::Float32]
            | [DataType::Fp8E5M2, DataType::Fp8E5M2, DataType::Float32]
            | [DataType::Int8, DataType::Int8, DataType::Int32]
    )
}

/// Refuse a product of 8-bit floats (`left` the left input's element type) beyond what was
/// measured (`tns_mm_fp8`): the descriptor 32 x 32 x 32, the left input not transposed, without
/// relaxed precision, over one simdgroup (`simdgroups`, when the form names its scope), and no
/// cooperative input (`cooperative_input`: none was measured, and no `run` form takes one).
fn fp8_measured(
    left: DataType,
    desc: &Descriptor,
    simdgroups: Option<u32>,
    cooperative_input: bool,
) -> Result<(), Error> {
    if !matches!(left, DataType::Fp8E4M3 | DataType::Fp8E5M2) {
        return Ok(());
    }
    let measured = (desc.m, desc.n, desc.k) == (32, 32, Inner::Static(32))
        && !desc.transpose_left
        && !desc.relaxed_precision
        && simdgroups.is_none_or(|s| s == 1);
    if !measured {
        let over = simdgroups.map_or(String::new(), |s| format!(" over {s} simdgroups"));
        return Err(refuse(format!("matmul2d {desc} over {left:?}{over}")));
    }
    if cooperative_input {
        return Err(refuse(format!("a cooperative matmul2d input of {left:?}")));
    }
    Ok(())
}

/// The simdgroups of a scope of `threads` threads (`air.get_simdgroup_size` is 32).
fn scope(threads: u64) -> Result<u32, Error> {
    match threads {
        32 => Ok(1),
        64 => Ok(2),
        128 => Ok(4),
        _ => Err(refuse(format!("a matmul2d scope of {threads} threads"))),
    }
}

/// What a cooperative-tensor helper call names, read from the caller's constant arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Helper {
    f: MatMul2dFn,
    holds: Holds,
    desc: Descriptor,
    /// The operand's element type, when the form passes it.
    elem: Option<DataType>,
    /// The layout, when the form passes the scope.
    layout: Option<Layout>,
}

impl Helper {
    /// The helper `f` called with `args`, whose integer values `eval` reads.
    pub fn of_call(
        f: MatMul2dFn,
        args: &[Value],
        eval: &dyn Fn(Value) -> Option<u64>,
    ) -> Result<Helper, Error> {
        let pos = positions(f).ok_or_else(|| refuse(format!("matmul2d {f:?}")))?;
        if Some(args.len()) != f.arity() {
            return Err(refuse(format!("matmul2d {f:?} of another arity")));
        }
        let arg = |i: usize| args.get(i).copied().and_then(eval);
        let holds = match arg(0).and_then(OperandIndex::from_raw) {
            Some(OperandIndex::Destination) => Holds::Destination,
            Some(OperandIndex::Right) => Holds::Right,
            Some(OperandIndex::Left) => Holds::Left,
            None => return Err(refuse("a cooperative tensor of an unknown operand")),
        };
        let mut fields = [0u64; 7];
        for (i, slot) in fields.iter_mut().enumerate() {
            *slot = arg(1 + i).ok_or_else(|| refuse("a matmul2d descriptor not constant"))?;
        }
        let desc = Descriptor::from_fields(fields)
            .ok_or_else(|| refuse("a malformed matmul2d descriptor"))?;
        let ty = |slot: Option<usize>| -> Result<Option<DataType>, Error> {
            match slot {
                None => Ok(None),
                Some(i) => arg(i)
                    .and_then(DataType::from_raw)
                    .map(Some)
                    .ok_or_else(|| refuse("a matmul2d element type not constant")),
            }
        };
        let [left, right, dest] = [ty(pos.types[0])?, ty(pos.types[1])?, ty(pos.types[2])?];
        if let (Some(l), Some(r)) = (left, right) {
            // `num_elements` names no destination type: the inputs' own destination stands in.
            let d = dest.unwrap_or(if l.is_integer() {
                DataType::Int32
            } else {
                DataType::Float32
            });
            if !modeled_types([l, r, d]) {
                return Err(refuse(format!("matmul2d over {l:?} x {r:?} into {d:?}")));
            }
        }
        let elem = match holds {
            Holds::Left => left,
            Holds::Destination => dest,
            Holds::Right => right,
        };
        if f == MatMul2dFn::TensorGetCoordinate {
            let coord = arg(COORDINATE_TYPE).and_then(DataType::from_raw);
            if !matches!(coord, Some(DataType::Int32 | DataType::UInt32)) {
                return Err(refuse(
                    "a matmul2d coordinate of a type other than int or uint",
                ));
            }
        }
        let simdgroups = match pos.threads {
            None => None,
            Some(i) => {
                let threads = arg(i).ok_or_else(|| refuse("a matmul2d scope not constant"))?;
                Some(scope(threads)?)
            }
        };
        if let Some(l) = left {
            fp8_measured(l, &desc, simdgroups, holds != Holds::Destination)?;
        }
        let layout = match simdgroups {
            None => None,
            Some(simdgroups) => Some(Layout::of(holds, &desc, simdgroups).ok_or_else(|| {
                refuse(format!(
                    "matmul2d {desc} {holds:?} over {simdgroups} simdgroups"
                ))
            })?),
        };
        Ok(Helper {
            f,
            holds,
            desc,
            elem,
            layout,
        })
    }

    fn layout(&self) -> Result<Layout, Error> {
        self.layout
            .ok_or_else(|| invariant("a matmul2d helper without its scope"))
    }

    fn elem_bytes(&self) -> Result<u64, Error> {
        self.elem
            .and_then(elem_bytes)
            .ok_or_else(|| refuse("a cooperative tensor of an unmodeled element type"))
    }

    /// The constant a size helper answers, `None` for another helper.
    pub fn size(&self) -> Result<Option<u64>, Error> {
        Ok(match self.f {
            MatMul2dFn::TensorNumElements => Some(u64::from(self.layout()?.capacity())),
            MatMul2dFn::TensorDataSize => {
                Some(u64::from(self.layout()?.capacity()) * self.elem_bytes()?)
            }
            _ => None,
        })
    }
}

/// A `run` call: its form, the descriptor its cooperative operands name, and the destination's
/// layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Run {
    form: RunForm,
    desc: Descriptor,
    dest: Layout,
    /// A cooperative left input's layout.
    left: Option<Layout>,
    /// A cooperative right input's layout.
    right: Option<Layout>,
}

impl Run {
    /// The `run` form `form` called with `args`, whose descriptor argument points at `desc`
    /// (when it is a constant) and whose destination's (and a cooperative input's) storage
    /// `storage` reads as the helper call that made it. A refusal names the first of: the form,
    /// the descriptor, the shape, then the storage. A `_v2` run whose tensor operands have no
    /// scale factors and only dynamic extents passes them as the earlier form does, so it is that
    /// form's run.
    pub fn of_call(
        form: RunForm,
        args: &[Value],
        desc: Option<Descriptor>,
        eval: &dyn Fn(Value) -> Option<u64>,
        storage: &dyn Fn(Value) -> Option<Helper>,
    ) -> Result<Run, Error> {
        if form.single_thread {
            return Err(refuse("a matmul2d run of the single-thread form"));
        }
        if args.len() != form.arity() {
            return Err(refuse("a matmul2d run of another arity"));
        }
        let types = [form.left.elem(), form.right.elem(), form.destination.elem()];
        if !modeled_types(types) {
            return Err(refuse(format!(
                "matmul2d over {:?} x {:?} into {:?}",
                types[0], types[1], types[2]
            )));
        }
        if !form.destination.is_cooperative() {
            return Err(refuse("a matmul2d tensor destination"));
        }
        let [left, right, dest] = form.operand_args();
        for (operand, at) in [(form.left, left), (form.right, right)] {
            if let OperandForm::Tensor { space, .. } = operand {
                if space == TensorSpace::Thread {
                    return Err(refuse("a matmul2d tensor in thread memory"));
                }
                let kind = args.get(at + 1).copied().and_then(eval);
                if kind.and_then(DescriptorType::from_raw) != Some(DescriptorType::Inline) {
                    return Err(refuse("a matmul2d tensor other than tensor_inline"));
                }
                if let Some([scale, block0, block1, extent0, extent1]) = form.v2_tensor_args(at) {
                    let word = |i: usize| args.get(i).copied().and_then(eval);
                    if word(scale) != Some(NO_SCALE)
                        || word(block0) != Some(0)
                        || word(block1) != Some(0)
                    {
                        return Err(refuse("a matmul2d tensor with scale factors"));
                    }
                    if word(extent0) != Some(DYNAMIC_EXTENT)
                        || word(extent1) != Some(DYNAMIC_EXTENT)
                    {
                        return Err(refuse("a matmul2d tensor of a static extent"));
                    }
                }
            }
        }
        let desc =
            desc.ok_or_else(|| refuse("a matmul2d run whose descriptor is not a constant"))?;
        if desc.k == Inner::Dynamic {
            return Err(refuse(format!("matmul2d {desc}")));
        }
        let threads = args
            .last()
            .copied()
            .and_then(eval)
            .ok_or_else(|| refuse("a matmul2d scope not constant"))?;
        let simdgroups = scope(threads)?;
        fp8_measured(
            types[0],
            &desc,
            Some(simdgroups),
            form.left.is_cooperative() || form.right.is_cooperative(),
        )?;
        let unmeasured = || refuse(format!("matmul2d {desc} over {simdgroups} simdgroups"));
        let dest_layout =
            Layout::of(Holds::Destination, &desc, simdgroups).ok_or_else(unmeasured)?;
        let layout_of = |cooperative: bool, holds: Holds| -> Result<Option<Layout>, Error> {
            match cooperative {
                true => Ok(Some(
                    Layout::of(holds, &desc, simdgroups).ok_or_else(unmeasured)?,
                )),
                false => Ok(None),
            }
        };
        let left_layout = layout_of(form.left.is_cooperative(), Holds::Left)?;
        let right_layout = layout_of(form.right.is_cooperative(), Holds::Right)?;
        // The cooperative operands' storage is the header's `get_element_pointer` of the
        // reserved index, made for this descriptor.
        let made_for = |at: usize, holds: Holds| -> Result<(), Error> {
            let h = args
                .get(at)
                .copied()
                .and_then(storage)
                .ok_or_else(|| refuse("a matmul2d run whose storage is not a helper's"))?;
            if h.holds != holds || h.desc != desc {
                return Err(refuse(
                    "a matmul2d run whose storage is another operand's or descriptor's",
                ));
            }
            Ok(())
        };
        made_for(dest, Holds::Destination)?;
        if form.left.is_cooperative() {
            made_for(left, Holds::Left)?;
        }
        if form.right.is_cooperative() {
            made_for(right, Holds::Right)?;
        }
        Ok(Run {
            form,
            desc,
            dest: dest_layout,
            left: left_layout,
            right: right_layout,
        })
    }
}

/// How one call is lowered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Lowered {
    /// The call answers a constant of `bits` bits (a size).
    Const {
        value: u64,
        bits: u8,
    },
    /// The call does nothing (a copy nothing reads).
    Nothing,
    Helper(Helper),
    Run(Run),
    /// A tensor operation of [`tensor::form`] `rank`.
    Tensor {
        op: Tensor,
        rank: u64,
    },
}

/// Whether `l`'s body reads 8-bit float elements (`metal_fp8_e4m3_format`,
/// `metal_fp8_e5m2_format`), each widened to float through integer arithmetic on its byte
/// ([`fp8_wide`]): a `run` whose left or right input holds them.
pub(super) fn reads_float8(l: &Lowered) -> bool {
    let float8 = |t: DataType| matches!(t, DataType::Fp8E4M3 | DataType::Fp8E5M2);
    match l {
        Lowered::Run(r) => float8(r.form.left.elem()) || float8(r.form.right.elem()),
        Lowered::Const { .. } | Lowered::Nothing | Lowered::Helper(_) | Lowered::Tensor { .. } => {
            false
        }
    }
}

/// Whether `l`'s body reads the lane's and the simdgroup's index.
pub(super) fn needs_lanes(l: &Lowered) -> bool {
    match l {
        Lowered::Run(_) => true,
        Lowered::Helper(h) => h.f == MatMul2dFn::TensorGetCoordinate,
        Lowered::Const { .. } | Lowered::Nothing | Lowered::Tensor { .. } => false,
    }
}

/// Build `l`'s function, of parameters `params` (the call's argument types, then two `i32` lane
/// parameters when [`needs_lanes`]) returning `ret`. The body reads each parameter as the
/// constant `params` gives it, where it gives one.
pub(super) fn function(
    parts: &mut ModuleParts,
    l: &Lowered,
    params: &[(TypeId, Option<Value>)],
    ret: TypeId,
    m: &mut Meter,
) -> Result<FuncId, Error> {
    let n = params.len();
    let mut types = Vec::new();
    let mut args = Vec::new();
    m.reserve(&mut types, n)?;
    m.reserve(&mut args, n)?;
    for (i, &(ty, known)) in params.iter().enumerate() {
        let arg = u32::try_from(i)
            .map(|i| Value::Arg(ArgIdx::new(i)))
            .map_err(|_| invariant("parameters"))?;
        types.push(ty);
        args.push(known.unwrap_or(arg));
    }
    let fn_ty = parts.types.intern(
        Type::Function {
            ret,
            params: types.as_slice().into(),
            vararg: false,
        },
        m,
    )?;
    m.release(types.capacity() * std::mem::size_of::<TypeId>());
    drop(types);
    let lanes = match (needs_lanes(l), n.checked_sub(2)) {
        (true, Some(at)) => Some(Lanes {
            lane: arg_at(&args, at)?,
            simdgroup: arg_at(&args, at + 1)?,
        }),
        (true, None) => return Err(invariant("a lowering without its lanes")),
        (false, _) => None,
    };
    let mut out = Out::new(parts);
    out.open(None)?;
    let result = match l {
        Lowered::Const { value, bits } => Some(out.int(*bits, *value, m)?),
        Lowered::Nothing => None,
        Lowered::Helper(h) => helper(&mut out, h, &args, lanes, m)?,
        Lowered::Run(r) => {
            let lanes = lanes.ok_or_else(|| invariant("a run without its lanes"))?;
            run(&mut out, r, &args, lanes, m)?;
            None
        }
        Lowered::Tensor { op, rank } => tensor_op(&mut out, *op, *rank, &args, m)?,
    };
    out.ret(result, m)?;
    let Out { body, parts, .. } = out;
    m.release(args.capacity() * std::mem::size_of::<Value>());
    let name = format!("__m2v.matmul2d.{}", parts.funcs.len());
    let name = parts.symbols.intern(name.as_bytes(), m)?;
    let mut ps = Vec::new();
    m.reserve(&mut ps, n)?;
    ps.resize(n, Param::default());
    let f = Function {
        name,
        ty: fn_ty,
        linkage: Linkage::Internal,
        visibility: Visibility::Default,
        unnamed_addr: UnnamedAddr::Significant,
        cconv: CallConv::C,
        attrs: FnAttrs::default(),
        ret_attrs: ParamAttrs::default(),
        params: ps.into_boxed_slice(),
        body: Some(body),
        section: None,
        align: None,
    };
    parts.funcs.push(f, m)
}

/// The lane's index in its simdgroup and the simdgroup's index in the threadgroup.
#[derive(Clone, Copy)]
struct Lanes {
    lane: Value,
    simdgroup: Value,
}

/// The argument at `i`.
fn arg_at(args: &[Value], i: usize) -> Result<Value, Error> {
    args.get(i)
        .copied()
        .ok_or_else(|| invariant("a lowered call missing an argument"))
}

/// [`Arith`] emitted into `out`, metering `m`.
struct Emit<'a, 'p> {
    out: &'a mut Out<'p>,
    m: &'a mut Meter,
}

impl Arith for Emit<'_, '_> {
    type V = Value;
    fn k(&mut self, v: u32) -> Result<Value, Error> {
        self.out.k32(v, self.m)
    }
    fn op(&mut self, op: BinOp, a: Value, b: Value) -> Result<Value, Error> {
        self.out.i32op(op, a, b, self.m)
    }
}

/// `v` (an `i16` element index) as an `i32`.
fn index32(out: &mut Out<'_>, v: Value, m: &mut Meter) -> Result<Value, Error> {
    let i32t = out.int_ty(32, m)?;
    out.cast(CastOp::ZExt, v, i32t, m)
}

/// `base + i * size` in thread memory, `i` an `i32`.
fn element_at(
    out: &mut Out<'_>,
    base: Value,
    i: Value,
    size: u64,
    m: &mut Meter,
) -> Result<Value, Error> {
    let i64t = out.int_ty(64, m)?;
    let i = out.cast(CastOp::ZExt, i, i64t, m)?;
    let size = out.k64(size, m)?;
    let off = out.bin(i64t, BinOp::Mul, i, size, m)?;
    out.byte_gep(AddrSpace::Thread, base, off, m)
}

fn helper(
    out: &mut Out<'_>,
    h: &Helper,
    args: &[Value],
    lanes: Option<Lanes>,
    m: &mut Meter,
) -> Result<Option<Value>, Error> {
    let pos = positions(h.f).ok_or_else(|| invariant("a lowered matmul2d helper"))?;
    let index = |out: &mut Out<'_>, m: &mut Meter| -> Result<Value, Error> {
        let at = pos
            .index
            .ok_or_else(|| invariant("a helper without an index"))?;
        index32(out, arg_at(args, at)?, m)
    };
    match h.f {
        // Construction zeroes the elements the thread holds (`+0` over floats), a destination's,
        // a right input's and a left input's alike: Metal's answers to `tns_init` and
        // `tns_init_left`, whose loops construct each round's tensor over the storage the round
        // before set, and to the gather kernels, which construct their destination in a loop's
        // body and accumulate into it.
        MatMul2dFn::TensorInit => {
            let elem = h
                .elem
                .ok_or_else(|| invariant("a cooperative tensor without its type"))?;
            let size = h.elem_bytes()?;
            let base = arg_at(args, STORAGE)?;
            let ty = elem_ty(out, elem, m)?;
            let bits = match elem {
                DataType::Float32 | DataType::Float16 => Const::Float(0),
                _ => Const::Int(0),
            };
            let zero = out.constant(ty, bits, m)?;
            let cap = out.k32(h.layout()?.capacity(), m)?;
            out.counted_loop(cap, &[], m, &mut |out, e, _, m| {
                let slot = element_at(out, base, e, size, m)?;
                out.store(zero, slot, size, m)?;
                Ok(Vec::new())
            })?;
            Ok(None)
        }
        MatMul2dFn::TensorIsValidElement => {
            // Every element of every measured layout is valid.
            let cap = h.layout()?.capacity();
            let e = index(out, m)?;
            let cap = out.k32(cap, m)?;
            Ok(Some(out.icmp(IntPredicate::Ult, e, cap, m)?))
        }
        MatMul2dFn::TensorGetElementPointer => {
            // Element `i` of the storage; the reserved index names the storage itself.
            let size = h.elem_bytes()?;
            let base = arg_at(args, STORAGE)?;
            let i = index(out, m)?;
            let reserved = out.k32(u32::from(RESERVED_INDEX), m)?;
            let is_reserved = out.icmp(IntPredicate::Eq, i, reserved, m)?;
            let zero = out.k32(0, m)?;
            let i32t = out.int_ty(32, m)?;
            let i = out.select(i32t, is_reserved, zero, i, m)?;
            Ok(Some(element_at(out, base, i, size, m)?))
        }
        MatMul2dFn::TensorGetCoordinate => {
            let l = h.layout()?;
            let coords = arg_at(args, COORDINATES)?;
            let lanes = lanes.ok_or_else(|| invariant("a coordinate without its lanes"))?;
            let e = index(out, m)?;
            let a = &mut Emit { out, m };
            let lp = lane_place(a, lanes.lane)?;
            let (row, col) = l.place(a, e, lp, lanes.simdgroup)?;
            let (c0, c1) = if l.row_first { (row, col) } else { (col, row) };
            out.store(c0, coords, 4, m)?;
            let second = out.byte_gep_k(AddrSpace::Thread, coords, 4, m)?;
            out.store(c1, second, 4, m)?;
            Ok(None)
        }
        MatMul2dFn::TensorDataSize
        | MatMul2dFn::TensorNumElements
        | MatMul2dFn::TensorGetElementIndex
        | MatMul2dFn::Run(_)
        | MatMul2dFn::Unmodeled => Err(invariant("a matmul2d helper lowered as a body")),
    }
}

/// The type the product reads elements of `t` as: `float`, or `i32` for an integer.
fn wide_ty(out: &mut Out<'_>, t: DataType, m: &mut Meter) -> Result<TypeId, Error> {
    if t.is_integer() {
        out.int_ty(32, m)
    } else {
        out.ty(Type::Float, m)
    }
}

/// The zero of [`wide_ty`].
fn wide_zero(out: &mut Out<'_>, t: DataType, m: &mut Meter) -> Result<Value, Error> {
    if t.is_integer() {
        out.k32(0, m)
    } else {
        out.f32k(0.0, m)
    }
}

/// The type the product loads an element of `t` as: `bfloat` as its `i16` bits.
fn elem_ty(out: &mut Out<'_>, t: DataType, m: &mut Meter) -> Result<TypeId, Error> {
    match t {
        DataType::Float32 => out.ty(Type::Float, m),
        DataType::Float16 => out.ty(Type::Half, m),
        DataType::Int32 => out.int_ty(32, m),
        DataType::BFloat16 => out.int_ty(16, m),
        DataType::Int8 => out.int_ty(8, m),
        _ => Err(invariant("a lowered matmul2d element type")),
    }
}

/// Load the element of type `t` at `ptr` and widen it exactly to [`wide_ty`].
fn load_wide(out: &mut Out<'_>, t: DataType, ptr: Value, m: &mut Meter) -> Result<Value, Error> {
    let i32t = out.int_ty(32, m)?;
    let f32t = out.ty(Type::Float, m)?;
    match t {
        DataType::Float32 => out.load(f32t, ptr, 4, m),
        DataType::Int32 => out.load(i32t, ptr, 4, m),
        DataType::Float16 => {
            let half = out.ty(Type::Half, m)?;
            let h = out.load(half, ptr, 2, m)?;
            out.cast(CastOp::FPExt, h, f32t, m)
        }
        DataType::BFloat16 => {
            // A bfloat is the high half of the float of the same value.
            let i16t = out.int_ty(16, m)?;
            let bits = out.load(i16t, ptr, 2, m)?;
            let wide = out.cast(CastOp::ZExt, bits, i32t, m)?;
            let wide = out.i32k(BinOp::Shl, wide, 16, m)?;
            out.cast(CastOp::Bitcast, wide, f32t, m)
        }
        DataType::Int8 => {
            let i8t = out.int_ty(8, m)?;
            let b = out.load(i8t, ptr, 1, m)?;
            out.cast(CastOp::SExt, b, i32t, m)
        }
        DataType::Fp8E4M3 | DataType::Fp8E5M2 => {
            let i8t = out.int_ty(8, m)?;
            let b = out.load(i8t, ptr, 1, m)?;
            let b = out.cast(CastOp::ZExt, b, i32t, m)?;
            fp8_wide(out, t, b, m)
        }
        _ => Err(invariant("a lowered matmul2d element type")),
    }
}

/// The float of the 8-bit float `t` whose byte is `b` (an `i32`, zero-extended), exactly, as
/// Metal reads it (`tns_mm_fp8`, the M3 Max): a subnormal kept, a signed zero kept; e4m3 has no
/// infinity or NaN (`0x7F` and `0xFF` are 480 and -480); e5m2's infinities are infinities and its
/// NaNs the quiet NaN `0x7FC00000`. Integer arithmetic on the byte's fields, so no device needs an
/// 8-bit or 16-bit float type for it.
fn fp8_wide(out: &mut Out<'_>, t: DataType, b: Value, m: &mut Meter) -> Result<Value, Error> {
    // The mantissa's bits, float's exponent bias less the format's, the least normal magnitude
    // (its bits), and the least subnormal's value.
    let (mantissa, rebias, least_normal, least_subnormal) = match t {
        DataType::Fp8E4M3 => (3, 127 - 7, 0x08, 2f32.powi(-9)),
        DataType::Fp8E5M2 => (2, 127 - 15, 0x04, 2f32.powi(-16)),
        _ => return Err(invariant("an 8-bit float of another format")),
    };
    let i32t = out.int_ty(32, m)?;
    let f32t = out.ty(Type::Float, m)?;
    let sign = out.i32k(BinOp::And, b, 0x80, m)?;
    let sign = out.i32k(BinOp::Shl, sign, 24, m)?;
    let magnitude = out.i32k(BinOp::And, b, 0x7F, m)?;
    // A normal magnitude: its exponent and mantissa fields moved to float's, the exponent
    // rebiased.
    let normal = out.i32k(BinOp::Shl, magnitude, 23 - mantissa, m)?;
    let normal = out.i32k(BinOp::Add, normal, rebias << 23, m)?;
    // A subnormal one (the exponent field zero, so the magnitude is the mantissa): the mantissa
    // times the least subnormal, exactly.
    let sub = out.cast(CastOp::UIToFP, magnitude, f32t, m)?;
    let scale = out.f32k(least_subnormal, m)?;
    let sub = out.bin(f32t, BinOp::FMul, sub, scale, m)?;
    let sub = out.cast(CastOp::Bitcast, sub, i32t, m)?;
    let least_normal = out.k32(least_normal, m)?;
    let is_sub = out.icmp(IntPredicate::Ult, magnitude, least_normal, m)?;
    let bits = out.select(i32t, is_sub, sub, normal, m)?;
    let mut bits = out.i32op(BinOp::Or, bits, sign, m)?;
    if t == DataType::Fp8E5M2 {
        // The exponent field all ones: an infinity, or (with a mantissa) a NaN.
        let infinite = out.k32(0x7C, m)?;
        let is_inf = out.icmp(IntPredicate::Eq, magnitude, infinite, m)?;
        let inf = out.k32(0x7F80_0000, m)?;
        let inf = out.i32op(BinOp::Or, inf, sign, m)?;
        bits = out.select(i32t, is_inf, inf, bits, m)?;
        let is_nan = out.icmp(IntPredicate::Ugt, magnitude, infinite, m)?;
        let nan = out.k32(0x7FC0_0000, m)?;
        bits = out.select(i32t, is_nan, nan, bits, m)?;
    }
    out.cast(CastOp::Bitcast, bits, f32t, m)
}

/// A tensor operand's fields, read once from its object ([`tensor`]).
#[derive(Clone, Copy)]
struct TensorView {
    space: AddrSpace,
    data: Value,
    extents: [Value; 2],
    strides: [Value; 2],
    elem: DataType,
}

impl TensorView {
    fn read(
        out: &mut Out<'_>,
        obj: Value,
        space: TensorSpace,
        elem: DataType,
        m: &mut Meter,
    ) -> Result<TensorView, Error> {
        let space = match space {
            TensorSpace::Device => AddrSpace::Device,
            TensorSpace::Threadgroup => AddrSpace::Threadgroup,
            TensorSpace::Thread => return Err(invariant("a lowered tensor in thread memory")),
        };
        let ptr = out.ptr_ty(space, m)?;
        let data = out.load(ptr, obj, tensor::ALIGN, m)?;
        let i32t = out.int_ty(32, m)?;
        let mut field = |off: u64, m: &mut Meter| -> Result<Value, Error> {
            let p = out.byte_gep_k(AddrSpace::Thread, obj, off, m)?;
            out.load(i32t, p, tensor::INDEX_BYTES, m)
        };
        let extents = [
            field(tensor::extent_at(0), m)?,
            field(tensor::extent_at(1), m)?,
        ];
        let strides = [
            field(tensor::stride_at(2, 0), m)?,
            field(tensor::stride_at(2, 1), m)?,
        ];
        Ok(TensorView {
            space,
            data,
            extents,
            strides,
            elem,
        })
    }

    /// The element at tensor index `(i0, i1)`, widened, or zero past the extents (loaded only
    /// within them).
    fn element(
        self,
        out: &mut Out<'_>,
        (i0, i1): (Value, Value),
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let ty = wide_ty(out, self.elem, m)?;
        let zero = wide_zero(out, self.elem, m)?;
        let in0 = out.icmp(IntPredicate::Slt, i0, self.extents[0], m)?;
        let in1 = out.icmp(IntPredicate::Slt, i1, self.extents[1], m)?;
        let i1t = out.int_ty(1, m)?;
        let inside = out.bin(i1t, BinOp::And, in0, in1, m)?;
        let size = elem_bytes(self.elem).ok_or_else(|| invariant("a lowered element type"))?;
        out.cond_value(inside, ty, zero, m, &mut |out, m| {
            let i64t = out.int_ty(64, m)?;
            let mut off = out.k64(0, m)?;
            for (i, s) in [(i0, self.strides[0]), (i1, self.strides[1])] {
                let i = out.cast(CastOp::SExt, i, i64t, m)?;
                let s = out.cast(CastOp::SExt, s, i64t, m)?;
                let t = out.bin(i64t, BinOp::Mul, i, s, m)?;
                off = out.bin(i64t, BinOp::Add, off, t, m)?;
            }
            let size = out.k64(size, m)?;
            let off = out.bin(i64t, BinOp::Mul, off, size, m)?;
            let p = out.byte_gep(self.space, self.data, off, m)?;
            load_wide(out, self.elem, p, m)
        })
    }
}

/// An input of a `run`.
enum Input {
    Tensor(TensorView),
    /// A cooperative input: its storage, element type, and layout.
    Cooperative {
        storage: Value,
        elem: DataType,
        layout: Layout,
    },
}

impl Input {
    /// The input of `form` at argument `at` of `args`, `layout` a cooperative one's.
    fn read(
        out: &mut Out<'_>,
        form: OperandForm,
        args: &[Value],
        at: usize,
        layout: Option<Layout>,
        m: &mut Meter,
    ) -> Result<Input, Error> {
        Ok(match form {
            OperandForm::Tensor { space, elem } => {
                Input::Tensor(TensorView::read(out, arg_at(args, at)?, space, elem, m)?)
            }
            OperandForm::Cooperative { elem } => Input::Cooperative {
                storage: arg_at(args, at)?,
                elem,
                layout: layout.ok_or_else(|| invariant("a cooperative input's layout"))?,
            },
        })
    }
}

/// `run`: every destination element the thread holds is, for `k` ascending below the
/// descriptor's `k`, `acc = fma(a, b, acc)` over floats (one rounding per step, the order
/// Metal's sums were measured to take, with or without relaxed precision) and `acc += a * b` over
/// integers (wrapping), from `acc = C` in `multiply_accumulate` mode and from zero in `multiply`
/// mode (`+0` over floats), where `a` and `b` read zero past their tensors' extents. A
/// cooperative input's element is another lane's, read with a shuffle every lane runs before any
/// branch.
fn run(
    out: &mut Out<'_>,
    r: &Run,
    args: &[Value],
    lanes: Lanes,
    m: &mut Meter,
) -> Result<(), Error> {
    let [left_at, right_at, dest_at] = r.form.operand_args();
    let a = Input::read(out, r.form.left, args, left_at, r.left, m)?;
    let b = Input::read(out, r.form.right, args, right_at, r.right, m)?;
    let dest = arg_at(args, dest_at)?;
    let dest_ty = r.form.destination.elem();
    let acc_ty = wide_ty(out, dest_ty, m)?;
    let float = !dest_ty.is_integer();
    let Inner::Static(k_total) = r.desc.k else {
        return Err(invariant("a lowered run with a dynamic k"));
    };
    let (tl, tr) = (r.desc.transpose_left, r.desc.transpose_right);
    let accumulate = r.desc.mode == Mode::MultiplyAccumulate;
    let l = r.dest;
    let lp = lane_place(&mut Emit { out, m }, lanes.lane)?;
    let cap = out.k32(l.capacity(), m)?;
    out.counted_loop(cap, &[], m, &mut |out, e, _, m| {
        let (row, col) = l.place(&mut Emit { out, m }, e, lp, lanes.simdgroup)?;
        let slot = element_at(out, dest, e, 4, m)?;
        let start = if accumulate {
            out.load(acc_ty, slot, 4, m)?
        } else if float {
            // Metal's float `multiply` starts from `+0`, so a sum of `-0` products is `+0`
            // (`matmul2d-edge-float-multiply-mode`).
            out.f32k(0.0, m)?
        } else {
            out.k32(0, m)?
        };
        let k_n = out.k32(k_total, m)?;
        let acc = out.counted_loop(k_n, &[(acc_ty, start)], m, &mut |out, k, carried, m| {
            let [acc] = carried else {
                return Err(invariant("a matmul2d accumulator"));
            };
            // The cooperative inputs' elements first, A's (row, k) and B's (k, col): each is
            // another lane's, read with a shuffle every lane runs, before any branch.
            let av = shuffled(out, &a, true, (row, k), lanes.lane, m)?;
            let bv = shuffled(out, &b, false, (k, col), lanes.lane, m)?;
            let bv = match (&b, bv) {
                (Input::Tensor(b), _) => b.element(out, if tr { (k, col) } else { (col, k) }, m)?,
                (Input::Cooperative { .. }, Some(v)) => v,
                (Input::Cooperative { .. }, None) => return Err(invariant("a shuffled input")),
            };
            let av = match (&a, av) {
                (Input::Tensor(a), _) => a.element(out, if tl { (row, k) } else { (k, row) }, m)?,
                (Input::Cooperative { .. }, Some(v)) => v,
                (Input::Cooperative { .. }, None) => return Err(invariant("a shuffled input")),
            };
            let next = if float {
                let f32t = out.ty(Type::Float, m)?;
                out.intrinsic(
                    "llvm.fma.f32",
                    f32t,
                    &[f32t, f32t, f32t],
                    &[av, bv, *acc],
                    m,
                )?
            } else {
                let p = out.i32op(BinOp::Mul, av, bv, m)?;
                out.i32op(BinOp::Add, *acc, p, m)?
            };
            Ok(vec![next])
        })?;
        let [acc] = acc.as_slice() else {
            return Err(invariant("a matmul2d accumulator"));
        };
        out.store(*acc, slot, 4, m)?;
        Ok(Vec::new())
    })?;
    Ok(())
}

/// A cooperative input's element, `None` for a tensor: the left input's at `(row, k)` when `left`
/// ([`Layout::left_holder`]), else the right input's at `(k, col)` ([`Layout::right_holder`]),
/// `row` and `col` the place of one of the lane's own destination elements.
fn shuffled(
    out: &mut Out<'_>,
    input: &Input,
    left: bool,
    at: (Value, Value),
    lane: Value,
    m: &mut Meter,
) -> Result<Option<Value>, Error> {
    let Input::Cooperative {
        storage,
        elem,
        layout,
    } = input
    else {
        return Ok(None);
    };
    let a = &mut Emit { out, m };
    let held = match left {
        true => layout.left_holder(a, at, lane)?,
        false => layout.right_holder(a, at, lane)?,
    };
    cooperative_element(out, *storage, *elem, held, m).map(Some)
}

/// A cooperative input's element held by lane `src` at element `e` ([`Layout::left_holder`],
/// [`Layout::right_holder`]), an index the same in every lane, so every lane loads its own
/// element there and shuffles.
fn cooperative_element(
    out: &mut Out<'_>,
    storage: Value,
    elem: DataType,
    (src, e): (Value, Value),
    m: &mut Meter,
) -> Result<Value, Error> {
    let size = elem_bytes(elem).ok_or_else(|| invariant("a lowered element type"))?;
    let mine = element_at(out, storage, e, size, m)?;
    let v = load_wide(out, elem, mine, m)?;
    let i16t = out.int_ty(16, m)?;
    let src = out.cast(CastOp::Trunc, src, i16t, m)?;
    if elem.is_integer() {
        let i32t = out.int_ty(32, m)?;
        out.intrinsic("air.simd_shuffle.u.i32", i32t, &[i32t, i16t], &[v, src], m)
    } else {
        let f32t = out.ty(Type::Float, m)?;
        out.intrinsic("air.simd_shuffle.f32", f32t, &[f32t, i16t], &[v, src], m)
    }
}

/// A tensor operation of [`tensor::form`] `rank` over this translation's tensor objects.
fn tensor_op(
    out: &mut Out<'_>,
    t: Tensor,
    rank: u64,
    args: &[Value],
    m: &mut Meter,
) -> Result<Option<Value>, Error> {
    match t {
        Tensor::Extent { .. } => tensor::extent(out, args, m).map(Some),
        Tensor::InitStrided { .. } => tensor::init(out, args, rank, m).map(|()| None),
        Tensor::Slice => tensor::copy(out, args, rank, m).map(|()| None),
        Tensor::DescriptorSize => Err(invariant("a tensor size lowered as a body")),
    }
}
