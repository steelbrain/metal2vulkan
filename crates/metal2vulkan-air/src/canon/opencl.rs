//! Metal's OpenCL mode: a kernel of Apple's OpenCL compiler linked as a compute pipeline made with
//! `openCLModeEnabled` links it ([`SpecRequest::open_cl_mode`](super::SpecRequest)).
//!
//! Apple's OpenCL compiler (`newLibraryWithSource` with `sourceLanguage` 1) writes each kernel as
//! an entry `__entry_<name>` that loads each value passed by value from its constant pointer and
//! calls the kernel, an internal function, which calls OpenCL C's work-item functions as internal
//! wrappers of `air.get_*` intrinsics: `get_global_id(d)` is `air.get_global_id.i32(d)` widened,
//! `get_global_offset(d)` `air.get_global_offset.i32(d)` and `get_work_dim()`
//! `air.get_work_dim.i32()` ([`WorkItem`]); the others are the reads of a global built-in
//! ([`GlobalBuiltin`]). What a pipeline of OpenCL mode answers, measured on an M3 Max (macOS
//! 27.0.1; Metal refuses a plain pipeline of the same function, "Internal compiler error"):
//!
//! - **Hidden buffers.** Three buffers come before the kernel's own: `__global_offset_and_num_dims`
//!   at index 0 (`uint`, read-only, constant), four words the dispatcher fills with the dispatch's
//!   global offset in each of its dimensions and its dimension count; `__printf_buffer` at 1
//!   (`char`, read-write); and `__image_order_data_type_table` at 2 (`uint`, read-only,
//!   constant). The pipeline's reflection lists them after the kernel's arguments, each used.
//! - **Indices.** Each argument bound at a buffer index sits 3 past the index its node states;
//!   threadgroup memory (`__local`), textures and samplers keep theirs.
//! - **Work-item functions.** `get_global_id(d)` is the thread's position in the grid plus the
//!   offset in `d`, `get_global_offset(d)` the offset, `get_work_dim()` the count, and
//!   `get_local_id`, `get_group_id`, `get_local_size`, `get_global_size` and `get_num_groups` the
//!   built-ins they read as program-scope variables. A grid of 8 by 4 threads in threadgroups of 4
//!   by 2 at offset (3, 5), of two dimensions, answers each value of every thread, for constant
//!   dimensions and for a dimension the kernel is given.
//! - **Image formats.** `get_image_channel_order(image)` is `air.get_channel_order_texture_<kind>`
//!   and `get_image_channel_data_type(image)` `air.get_channel_data_type_texture_<kind>`
//!   ([`Query::ChannelOrder`], [`Query::ChannelDataType`]), each of the image argument. The third
//!   hidden buffer holds two words per texture index: of the image at texture index `t`, word
//!   `2t` is its data type and word `2t + 1` its order, read as they are. Measured on an M3 Max
//!   (macOS 27.0.1) and a Radeon Pro 5300M (macOS 26.7.1) with the buffer's word `i` set to
//!   `i * 0x10000 + 0x1234`, over one image of each kind beside a 2D image at index 0. Two
//!   queries of one kind's wrapper (`get_image_channel_order` of two read-only 2D images) answer
//!   the same word on both, the M3 Max's and the 5300M's of different calls; here each query
//!   answers its own image's.
//! - **Images in the reflection.** An image argument reflects as the texture bound for it: an
//!   `image1d_t` a 2D texture and an `image1d_array_t` a 2D array ([`entry`]), each of data type
//!   `MTLDataTypeNone` (AIR names its component `unknown`), whatever the kernel reads or writes it
//!   as. Measured on an M3 Max (macOS 27.0.1) over `image1d_t`, `image1d_array_t`, `image2d_t` and
//!   `image3d_t` arguments read as float, int, uint and half, written, only queried and unused.
//!
//! `expand` runs after SCCP, whose constants name each read's dimension, and before the global
//! built-ins (`global_builtin`) take the reads it leaves. It appends the three buffers to the entry
//! and rewrites each read of a dimension `c` below 3: `get_global_id(c)` becomes the built-in's
//! read of lane `c` plus word `c` of the first buffer, `get_global_offset(c)` that word, and
//! `get_work_dim()` word 3. A read of a dimension past 2 is OpenCL's answer for it, 0 for an id
//! or an offset and 1 for a size or a count (the M3 Max leaves the store of such a value
//! unwritten). A read of a dimension no constant names selects among its reads of the three lanes,
//! with OpenCL's answer past them. Each image format query of an image argument at a constant
//! texture index becomes the load of its word of the third buffer; one of any other texture is
//! left, and lowering refuses it. [`entry`] then moves the entry's buffer-indexed arguments 3 on
//! and adds the three buffers' bindings.

use crate::abi::{
    AbiInt, Access, ArgTypeInfo, BufferArg, EntryPoint, ParamBinding, ParamRole, Slot,
};
use crate::budget::Meter;
use crate::error::{DiagName, Error};
use crate::intrinsics::{AirIntrinsic, Elem, GlobalBuiltin, GlobalVar, Query, TexOp, WorkItem};
use crate::ir::{
    AddrSpace, Align, ArgIdx, BinOp, Block, Body, Call, Callee, CastOp, Const, Constant, FastMath,
    FuncId, Idx, Inst, InstId, InstKind, IntFlags, IntPredicate, ModuleParts, Type, TypeId, Value,
};

use super::rewrite::{append_params, invariant};

/// How far past the index its node states a buffer-indexed argument sits: past the three hidden
/// buffers.
const HIDDEN_BUFFERS: u32 = 3;

/// The hidden buffers, in index order: their names, element types, sizes, accesses and address
/// spaces, as the M3 Max's reflection states them.
const BUFFERS: [(&str, &str, u32, Access, AddrSpace); 3] = [
    (
        "__global_offset_and_num_dims",
        "uint",
        4,
        Access::Read,
        AddrSpace::Constant,
    ),
    (
        "__printf_buffer",
        "char",
        1,
        Access::ReadWrite,
        AddrSpace::Device,
    ),
    (
        "__image_order_data_type_table",
        "uint",
        4,
        Access::Read,
        AddrSpace::Constant,
    ),
];

/// The word of the first hidden buffer that holds the dispatch's dimension count.
const WORK_DIM_WORD: u64 = 3;

/// The hidden buffer that holds the images' formats, after the first.
const TABLE: u32 = 2;

/// The entry's image arguments: each parameter bound at a constant texture index, with that
/// index.
#[derive(Debug, Default)]
pub(super) struct Images(Vec<(u32, u32)>);

impl Images {
    /// The image arguments of entry `ep`.
    pub(super) fn of(ep: &EntryPoint, m: &mut Meter) -> Result<Images, Error> {
        let mut images = Vec::new();
        for p in &ep.params {
            m.tick(1)?;
            if let ParamRole::Texture(t) = &p.role {
                if let (AbiInt::Const(index), AbiInt::Const(1)) =
                    (t.location.index, t.location.count)
                {
                    m.reserve(&mut images, 1)?;
                    images.push((p.param, index));
                }
            }
        }
        Ok(Images(images))
    }

    /// The texture index of the image `v` is, where it is an image argument.
    fn index(&self, v: Value) -> Option<u32> {
        let Value::Arg(a) = v else {
            return None;
        };
        let param = u32::try_from(a.index()).ok()?;
        self.0.iter().find(|(p, _)| *p == param).map(|&(_, t)| t)
    }

    /// The bytes [`Images::of`] charged.
    fn release(self, m: &mut Meter) {
        m.release(self.0.capacity() * std::mem::size_of::<(u32, u32)>());
    }
}

/// An OpenCL work-item read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Read {
    /// A built-in's lane, read as a global built-in.
    Builtin(GlobalBuiltin),
    /// `get_global_offset(d)`.
    Offset,
    /// `get_work_dim()`.
    WorkDim,
    /// `get_image_channel_order` or `get_image_channel_data_type` of an image argument: word
    /// `word` of the third hidden buffer.
    Format { word: u64 },
}

impl Read {
    /// The read's element: the built-in's width, or `i32`.
    fn elem(self) -> Elem {
        match self {
            Read::Builtin(g) => g.ty,
            Read::Offset | Read::WorkDim | Read::Format { .. } => Elem::I32,
        }
    }

    /// OpenCL's answer for a dimension past 2: 0 for an id or an offset, 1 for a size or a count.
    fn past(self) -> u64 {
        match self {
            Read::Builtin(GlobalBuiltin {
                var:
                    GlobalVar::ThreadsPerThreadgroup
                    | GlobalVar::ThreadsPerGrid
                    | GlobalVar::ThreadgroupsPerGrid,
                ..
            }) => 1,
            _ => 0,
        }
    }

    /// Whether the read takes a dimension.
    fn dimensioned(self) -> bool {
        matches!(self, Read::Builtin(_) | Read::Offset)
    }
}

/// The OpenCL read `kind` is, with its call: a read of one of the six built-ins OpenCL C names,
/// `get_global_offset`, `get_work_dim`, or an image format query of one of `images`.
fn read<'k>(kind: &'k InstKind, images: &Images) -> Option<(Read, &'k Call)> {
    let InstKind::Call(call) = kind else {
        return None;
    };
    let Callee::Air { op, .. } = call.callee else {
        return None;
    };
    let read = match op {
        AirIntrinsic::GlobalBuiltin(
            g @ GlobalBuiltin {
                var:
                    GlobalVar::ThreadPositionInGrid
                    | GlobalVar::ThreadPositionInThreadgroup
                    | GlobalVar::ThreadgroupPositionInGrid
                    | GlobalVar::ThreadsPerThreadgroup
                    | GlobalVar::ThreadsPerGrid
                    | GlobalVar::ThreadgroupsPerGrid,
                ..
            },
        ) => Read::Builtin(g),
        AirIntrinsic::WorkItem(WorkItem::GlobalOffset) => Read::Offset,
        AirIntrinsic::WorkItem(WorkItem::WorkDim) => Read::WorkDim,
        AirIntrinsic::Texture(t) => {
            let order = match t.op {
                TexOp::Query(Query::ChannelOrder) => 1,
                TexOp::Query(Query::ChannelDataType) => 0,
                _ => return None,
            };
            let index = images.index(*call.args.first()?)?;
            Read::Format {
                word: 2 * u64::from(index) + order,
            }
        }
        _ => return None,
    };
    Some((read, call))
}

/// The dimension a read names: a constant, or a value no constant gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dim {
    Const(u64),
    Value(Value),
}

/// The dimension call `call` of `r` names (`Const(WORK_DIM_WORD)` for `get_work_dim` and an image
/// format's word for its query, which name none).
fn dim(parts: &ModuleParts, r: Read, call: &Call) -> Result<Dim, Error> {
    match r {
        Read::WorkDim => return Ok(Dim::Const(WORK_DIM_WORD)),
        Read::Format { word } => return Ok(Dim::Const(word)),
        Read::Builtin(_) | Read::Offset => {}
    }
    let [v] = call.args.as_ref() else {
        return Err(invariant("a work-item read without its dimension"));
    };
    Ok(match v {
        Value::Const(c) => match parts.consts.get(*c).map(|k| &k.value) {
            Some(Const::Int(n)) => Dim::Const(*n),
            _ => Dim::Value(*v),
        },
        _ => Dim::Value(*v),
    })
}

/// An instruction sink over the output body.
struct Sink<'a> {
    out: &'a mut Body,
    parts: &'a mut ModuleParts,
    /// The first hidden buffer's parameter.
    offsets: Value,
    /// The third hidden buffer's parameter.
    table: Value,
}

impl Sink<'_> {
    fn ty(&mut self, t: Type, m: &mut Meter) -> Result<TypeId, Error> {
        self.parts.types.intern(t, m)
    }

    fn elem_ty(&mut self, e: Elem, m: &mut Meter) -> Result<TypeId, Error> {
        let bits = match e {
            Elem::I16 => 16,
            _ => 32,
        };
        self.ty(Type::Int(bits), m)
    }

    fn int(&mut self, e: Elem, v: u64, m: &mut Meter) -> Result<Value, Error> {
        let ty = self.elem_ty(e, m)?;
        let c = self.parts.consts.intern(
            Constant {
                ty,
                value: Const::Int(v),
            },
            m,
        )?;
        Ok(Value::Const(c))
    }

    fn push(&mut self, ty: TypeId, kind: InstKind, m: &mut Meter) -> Result<Value, Error> {
        let id = self.out.push_inst(Inst { ty, kind }, m)?;
        Ok(Value::Inst(id))
    }

    /// Word `w` of the hidden buffer `base`, in `e`.
    fn word(&mut self, base: Value, w: u64, e: Elem, m: &mut Meter) -> Result<Value, Error> {
        let i32t = self.elem_ty(Elem::I32, m)?;
        let ptr = self.ty(
            Type::Ptr {
                space: AddrSpace::Constant,
            },
            m,
        )?;
        let index = self.int(Elem::I32, w, m)?;
        let at = self.push(
            ptr,
            InstKind::Gep {
                src_ty: i32t,
                base,
                indices: Box::new([index]),
                inbounds: true,
            },
            m,
        )?;
        let word = self.push(
            i32t,
            InstKind::Load {
                ptr: at,
                align: Align::from_bytes(4),
                volatile: false,
                atomic: None,
                range: None,
            },
            m,
        )?;
        if e == Elem::I32 {
            return Ok(word);
        }
        let to = self.elem_ty(e, m)?;
        self.push(
            to,
            InstKind::Cast {
                op: CastOp::Trunc,
                v: word,
            },
            m,
        )
    }

    /// Read `r` of the constant dimension `c`, `call` (of result type `ty`) the original read.
    fn at(
        &mut self,
        r: Read,
        call: &Call,
        ty: TypeId,
        c: u64,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let e = r.elem();
        if r.dimensioned() && c > 2 {
            return self.int(e, r.past(), m);
        }
        match r {
            Read::Offset | Read::WorkDim => self.word(self.offsets, c, e, m),
            Read::Format { .. } => self.word(self.table, c, e, m),
            Read::Builtin(g) => {
                let lane = self.int(e, c, m)?;
                let mut lane_read = call.clone();
                lane_read.args = Box::new([lane]);
                let v = self.push(ty, InstKind::Call(Box::new(lane_read)), m)?;
                if g.var != GlobalVar::ThreadPositionInGrid {
                    return Ok(v);
                }
                let offset = self.word(self.offsets, c, e, m)?;
                self.push(
                    ty,
                    InstKind::BinOp {
                        op: BinOp::Add,
                        flags: IntFlags::NONE,
                        fmf: FastMath::NONE,
                        a: v,
                        b: offset,
                    },
                    m,
                )
            }
        }
    }

    /// Whether dimension `d` of read `r` is `c`.
    fn is(&mut self, r: Read, d: Value, c: u64, m: &mut Meter) -> Result<Value, Error> {
        let i1 = self.ty(Type::Int(1), m)?;
        let b = self.int(r.elem(), c, m)?;
        self.push(
            i1,
            InstKind::ICmp {
                pred: IntPredicate::Eq,
                a: d,
                b,
            },
            m,
        )
    }

    /// Read `r` (of result type `ty`) of dimension `d`, `call` the original read. A dimension no
    /// constant names selects among the three lanes' reads, OpenCL's answer past them; the last
    /// instruction pushed is the result, or the result is a constant and none is pushed.
    fn emit(
        &mut self,
        r: Read,
        call: &Call,
        ty: TypeId,
        d: Dim,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let d = match d {
            Dim::Const(c) => return self.at(r, call, ty, c, m),
            Dim::Value(d) => d,
        };
        let lanes = [
            self.at(r, call, ty, 0, m)?,
            self.at(r, call, ty, 1, m)?,
            self.at(r, call, ty, 2, m)?,
        ];
        let is = [
            self.is(r, d, 0, m)?,
            self.is(r, d, 1, m)?,
            self.is(r, d, 2, m)?,
        ];
        let mut v = self.int(r.elem(), r.past(), m)?;
        for (lane, cmp) in lanes.into_iter().zip(is).rev() {
            v = self.push(
                ty,
                InstKind::Select {
                    c: cmp,
                    t: lane,
                    f: v,
                    fmf: FastMath::NONE,
                },
                m,
            )?;
        }
        Ok(v)
    }
}

/// What one instruction of the old body becomes: its read's expansion, `len` instructions whose
/// last is its value, or (`len` 0) a constant.
#[derive(Clone, Copy, Debug)]
struct Expansion {
    len: u32,
    constant: Option<Value>,
}

/// The expansion of read `r` of dimension `d` (`call`, of result type `ty`), measured by
/// emitting it into a scratch body.
fn measure(
    parts: &mut ModuleParts,
    (offsets, table): (Value, Value),
    (r, call, ty, d): (Read, &Call, TypeId, Dim),
    m: &mut Meter,
) -> Result<Expansion, Error> {
    let mut scratch = Body::default();
    let mut s = Sink {
        out: &mut scratch,
        parts,
        offsets,
        table,
    };
    let v = s.emit(r, call, ty, d, m)?;
    let len = u32::try_from(scratch.insts.len()).map_err(|_| invariant("expansion length"))?;
    scratch.release(m);
    Ok(Expansion {
        len,
        constant: (len == 0).then_some(v),
    })
}

/// Append the three hidden buffers to the entry `func` and make each OpenCL work-item read of its
/// solved `body` a read of a built-in or of the first buffer, and each image format query of one
/// of its `images` a read of the third (see the module), returning the new body and the first
/// buffer's parameter index.
pub(super) fn expand(
    body: Body,
    parts: &mut ModuleParts,
    (func, images): (FuncId, Images),
    m: &mut Meter,
) -> Result<(Body, u32), Error> {
    let out = rewrite(body, parts, func, &images, m);
    images.release(m);
    out
}

/// [`expand`] over `images`.
fn rewrite(
    body: Body,
    parts: &mut ModuleParts,
    func: FuncId,
    images: &Images,
    m: &mut Meter,
) -> Result<(Body, u32), Error> {
    let [a, b, c] = BUFFERS.map(|(.., space)| Type::Ptr { space });
    let tys = [
        parts.types.intern(a, m)?,
        parts.types.intern(b, m)?,
        parts.types.intern(c, m)?,
    ];
    let first = append_params(parts, func, &tys, m)?;
    let offsets = Value::Arg(ArgIdx::new(first));
    let table = Value::Arg(ArgIdx::new(
        first
            .checked_add(TABLE)
            .ok_or_else(|| invariant("parameter indices"))?,
    ));
    let buffers = (offsets, table);
    if !body
        .insts
        .iter()
        .any(|(_, i)| read(&i.kind, images).is_some())
    {
        return Ok((body, first));
    }
    // Each old instruction's value in the output: an expansion's last instruction, or its
    // constant.
    let mut result: Vec<Value> = Vec::new();
    m.reserve(&mut result, body.insts.len())?;
    let mut next = 0u32;
    for (_, inst) in body.insts.iter() {
        m.tick(1)?;
        let e = match read(&inst.kind, images) {
            Some((r, call)) => {
                let d = dim(parts, r, call)?;
                measure(parts, buffers, (r, call, inst.ty, d), m)?
            }
            None => Expansion {
                len: 1,
                constant: None,
            },
        };
        next = next
            .checked_add(e.len)
            .ok_or_else(|| invariant("instruction ids"))?;
        result.push(match e.constant {
            Some(c) => c,
            None => Value::Inst(InstId::new(next - 1)),
        });
    }
    let map = |val: Value| -> Result<Value, Error> {
        match val {
            Value::Inst(i) => result
                .get(i.index())
                .copied()
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
            let found = read(&kind, images).map(|(r, call)| (r, call.clone()));
            let value = match found {
                Some((r, call)) => {
                    let d = dim(parts, r, &call)?;
                    let mut s = Sink {
                        out: &mut out,
                        parts,
                        offsets,
                        table,
                    };
                    s.emit(r, &call, inst.ty, d, m)?
                }
                None => {
                    let id = out.push_inst(Inst { ty: inst.ty, kind }, m)?;
                    Value::Inst(id)
                }
            };
            if result.get(i.index()) != Some(&value) {
                return Err(invariant(
                    "a work-item read's expansion differs from its measure",
                ));
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
    m.release(result.len().saturating_mul(std::mem::size_of::<Value>()));
    // The body the expansions replaced is freed.
    body.release(m);
    Ok((out, first))
}

/// Entry point `ep` as a pipeline of OpenCL mode binds it, the three hidden buffers its parameters
/// `first` to `first + 2`: each argument bound at a buffer index 3 past its own (an argument
/// buffer, a function table and an acceleration structure as a buffer), threadgroup memory,
/// textures and samplers at theirs, a 1D image's texture of the 2D type Metal's OpenCL binds
/// ([`bound_texture`]), and the buffers' bindings after the entry's own.
pub fn entry(mut ep: EntryPoint, first: u32) -> Result<EntryPoint, Error> {
    for p in &mut ep.params {
        if let ParamRole::Texture(_) = p.role {
            let bound = (p.ty.name.as_ref()).and_then(|name| bound_texture(&name.to_string()));
            if let Some(bound) = bound {
                p.ty.name = Some(DiagName::new(bound));
            }
            continue;
        }
        let slot = match &mut p.role {
            ParamRole::Buffer(b) | ParamRole::ArgumentBuffer(b) => &mut b.location,
            ParamRole::AccelerationStructure { arg, .. }
            | ParamRole::FunctionTable(arg)
            | ParamRole::IntersectionTable(arg) => &mut arg.location,
            _ => continue,
        };
        if let AbiInt::Const(index) = &mut slot.index {
            *index = index
                .checked_add(HIDDEN_BUFFERS)
                .ok_or_else(|| invariant("a buffer index past the index space"))?;
        }
    }
    for (index, (&(name, ty, size, access, space), param)) in
        (0u32..).zip(BUFFERS.iter().zip(first..))
    {
        ep.params.push(ParamBinding {
            param,
            role: ParamRole::Buffer(BufferArg {
                location: Slot {
                    index: AbiInt::Const(index),
                    count: AbiInt::Const(1),
                },
                access,
                space,
                buffer_size: None,
                layout: None,
                raster_order_group: None,
            }),
            unused: false,
            predicate: None,
            ty: ArgTypeInfo {
                size: Some(size),
                align: Some(size),
                name: Some(DiagName::new(ty)),
            },
            name: Some(DiagName::new(name)),
        });
    }
    Ok(ep)
}

/// The MSL type of the texture Metal's OpenCL binds for an image argument whose type AIR names
/// `name`, where that is another: a 1D image's texture is 2D (`texture1d<..>` a `texture2d<..>`,
/// `texture1d_array<..>` a `texture2d_array<..>`), as [`opencl_1d`](super::opencl_1d) reads it,
/// and as a pipeline of OpenCL mode reflects it (`MTLTextureType2D` and `MTLTextureType2DArray`,
/// measured on an M3 Max under macOS 27.0.1 and a Radeon Pro 5300M under macOS 26.7.1).
fn bound_texture(name: &str) -> Option<String> {
    let (template, rest) = name.split_once('<')?;
    let bound = match template {
        "texture1d" => "texture2d",
        "texture1d_array" => "texture2d_array",
        _ => return None,
    };
    Some(format!("{bound}<{rest}"))
}
