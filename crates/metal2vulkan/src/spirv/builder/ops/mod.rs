//! Typed instruction constructors, one file per family.
//!
//! Every constructor checks the opcode's typing rules against its operands' types, derives the
//! result type (interning it when new), and only then appends the instruction, so a rejected call
//! leaves the function unchanged apart from possibly-interned helper types and constants. Errors
//! are [`BuildError::Operand`] naming the opcode and the violated [`OperandRule`]; operand
//! positions in the rule are the constructor's argument positions (0-based).
//!
//! Families: [`arith`] (unary, binary, select, bit fields), [`convert`], [`composite`],
//! [`memory`] (access chains, array length, copies, calls), [`atomic`] (atomics, texel pointers,
//! barriers), [`image`] (sampling, fetch, gather, read/write, queries, derivatives), [`group`]
//! (subgroup operations), [`glsl`] (`GLSL.std.450`), [`misc`] (undef, helper invocation,
//! interlock), [`mesh`] (a mesh pipeline's output counts and launch), and [`ray`] (ray queries).

pub mod arith;
pub mod atomic;
pub mod composite;
pub mod convert;
pub mod glsl;
pub mod group;
pub mod image;
pub mod memory;
pub mod mesh;
pub mod misc;
pub mod ray;

#[cfg(test)]
mod tests;

use crate::spirv::builder::function::FunctionBuilder;
use crate::spirv::builder::module::BlockDraft;
use crate::spirv::error::{BResult, BuildError, EntryRule, OperandRule};
use crate::spirv::ids::{ConstId, TypeId, Value};
use crate::spirv::ir::{
    ConstBits, ConstKind, Constant, EntryPoint, ExecMode, ExecModel, Inst, ModelNeed, Terminator,
    Type, ValueOrigin,
};

/// The component class of a scalar or vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Bool,
    Int { signed: bool },
    Float,
}

/// The numeric shape of a scalar or vector type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Num {
    pub kind: Kind,
    /// Component width in bits (1 for `bool`).
    pub width: u8,
    /// Component count (1 for a scalar).
    pub n: u8,
    /// The component type.
    pub elem: TypeId,
    /// The whole type.
    pub ty: TypeId,
}

impl Num {
    pub fn is_int(&self) -> bool {
        matches!(self.kind, Kind::Int { .. })
    }
    pub fn is_uint(&self) -> bool {
        self.kind == Kind::Int { signed: false }
    }
    pub fn is_sint(&self) -> bool {
        self.kind == Kind::Int { signed: true }
    }
    pub fn is_float(&self) -> bool {
        self.kind == Kind::Float
    }
    pub fn is_bool(&self) -> bool {
        self.kind == Kind::Bool
    }
    pub fn is_scalar(&self) -> bool {
        self.n == 1
    }
    pub fn is_vector(&self) -> bool {
        self.n > 1
    }
    /// Same class (ignoring integer signedness), width, and component count.
    pub fn same_shape(&self, o: &Num) -> bool {
        self.class() == o.class() && self.width == o.width && self.n == o.n
    }
    fn class(&self) -> u8 {
        match self.kind {
            Kind::Bool => 0,
            Kind::Int { .. } => 1,
            Kind::Float => 2,
        }
    }
}

pub(crate) fn err(op: crate::spv::Op, rule: OperandRule) -> BuildError {
    BuildError::Operand { op, rule }
}

/// An operand position, saturated into the rule's `u8`.
pub(crate) fn index(i: usize) -> u8 {
    u8::try_from(i).unwrap_or(u8::MAX)
}

impl FunctionBuilder<'_> {
    /// The numeric shape of type `ty`, if it is a scalar or vector.
    pub(crate) fn num_of(&self, ty: TypeId) -> BResult<Option<Num>> {
        let types = self.mb().types();
        let (elem, n) = match types.get(ty)? {
            Type::Vector { elem, len } => (*elem, *len),
            _ => (ty, 1),
        };
        let (kind, width) = match types.get(elem)? {
            Type::Bool => (Kind::Bool, 1),
            Type::Int { width, signed } => (Kind::Int { signed: *signed }, *width),
            Type::Float { width } => (Kind::Float, *width),
            _ => return Ok(None),
        };
        Ok(Some(Num {
            kind,
            width,
            n,
            elem,
            ty,
        }))
    }

    /// The numeric shape of value `v`, if it is a scalar or vector.
    pub(crate) fn num(&self, v: Value) -> BResult<Option<Num>> {
        self.num_of(self.value_ty(v)?)
    }

    /// Operand `i` (`v`) must be a scalar or vector satisfying `pred`, else `Category(i)`.
    pub(crate) fn want(
        &self,
        op: crate::spv::Op,
        i: u8,
        v: Value,
        pred: impl Fn(&Num) -> bool,
    ) -> BResult<Num> {
        match self.num(v)? {
            Some(n) if pred(&n) => Ok(n),
            _ => Err(err(op, OperandRule::Category(i))),
        }
    }

    /// Intern `bool` (`n == 1`) or a `bool` vector.
    pub(crate) fn bool_ty(&mut self, n: u8) -> BResult<TypeId> {
        let b = self.module().ty(Type::Bool)?;
        self.vec_of(b, n)
    }

    /// `elem` itself when `n == 1`, else the vector of `n` of it.
    pub(crate) fn vec_of(&mut self, elem: TypeId, n: u8) -> BResult<TypeId> {
        if n == 1 {
            Ok(elem)
        } else {
            self.module().ty(Type::Vector { elem, len: n })
        }
    }

    /// Intern the unsigned integer of `width` bits.
    pub(crate) fn uint_ty(&mut self, width: u8) -> BResult<TypeId> {
        self.module().ty(Type::Int {
            width,
            signed: false,
        })
    }

    /// Intern the 32-bit float.
    pub(crate) fn f32_ty(&mut self) -> BResult<TypeId> {
        self.module().ty(Type::Float { width: 32 })
    }

    /// Intern the unsigned 32-bit constant `v`.
    pub(crate) fn u32_const(&mut self, v: u32) -> BResult<ConstId> {
        let ty = self.uint_ty(32)?;
        self.module().constant(Constant::Scalar {
            ty,
            bits: ConstBits(u64::from(v)),
        })
    }

    /// The bits of `v` when it is a plain (non-specialization) scalar constant.
    pub(crate) fn const_scalar(&self, v: Value) -> BResult<Option<u64>> {
        match self.origin(v)? {
            ValueOrigin::Const(c) => Ok(self.mb().consts().scalar_bits(c).map(|b| b.0)),
            _ => Ok(None),
        }
    }

    /// Whether `v` is a plain (non-specialization) constant of any shape.
    pub(crate) fn is_plain_const(&self, v: Value) -> BResult<bool> {
        match self.origin(v)? {
            ValueOrigin::Const(c) => Ok(matches!(
                self.mb().consts().get(c)?.kind,
                ConstKind::Plain(_)
            )),
            _ => Ok(false),
        }
    }

    /// Where `v` comes from.
    pub(crate) fn origin(&self, v: Value) -> BResult<ValueOrigin> {
        self.mb()
            .values
            .get(v.index())
            .map(|d| d.origin)
            .ok_or(BuildError::Dangling(crate::spirv::error::Dangling::Value(
                v,
            )))
    }

    /// Append `inst` with a result of type `ty`.
    pub(crate) fn emit(&mut self, inst: Inst, ty: TypeId) -> BResult<Value> {
        self.append(inst, Some(ty))?
            .ok_or(BuildError::IdSpaceExhausted)
    }

    /// Append `inst`, which has no result.
    pub(crate) fn emit_void(&mut self, inst: Inst) -> BResult<()> {
        self.append(inst, None).map(|_| ())
    }
}

/// Check every instruction's [`ModelNeed`] against the entry point (called by `finalize`, when the
/// entry point and every function are known). Modules are compiled per entry, so every block
/// belongs to the entry's call tree or is dead code that must be valid for it anyway. `is_subpass`
/// says whether an image value is a `SubpassData` image, whose reads are fragment-only
/// (VUID-StandaloneSpirv-SubpassData-04660).
pub(super) fn check_entry_model(
    blocks: &[BlockDraft],
    entry: &EntryPoint,
    is_subpass: impl Fn(Value) -> BResult<bool>,
) -> BResult<()> {
    let frag = entry.model == ExecModel::Fragment;
    let derivatives = frag
        || (entry.model == ExecModel::GLCompute
            && entry
                .modes
                .iter()
                .any(|m| matches!(m, ExecMode::DerivativeGroupLinear)));
    let interlock = frag && entry.modes.iter().any(|m| m.is_interlock());
    for b in blocks {
        // A task entry launches its mesh workgroups from the entry function itself.
        let launch = matches!(b.term, Some(Terminator::EmitMeshTasks { .. }));
        if launch && (entry.model != ExecModel::TaskEXT || b.func != entry.func) {
            return Err(BuildError::Entry(EntryRule::InstructionForModel(
                crate::spv::Op::EmitMeshTasksEXT,
            )));
        }
        for node in &b.insts {
            let need = match &node.inst {
                Inst::ImageRead { image, .. } if is_subpass(*image)? => Some(ModelNeed::Fragment),
                inst => inst.model_need(),
            };
            let Some(need) = need else {
                continue;
            };
            let ok = match need {
                ModelNeed::ImplicitDerivatives => derivatives,
                ModelNeed::Fragment => frag,
                ModelNeed::Interlock => interlock,
                ModelNeed::WorkgroupScope => matches!(
                    entry.model,
                    ExecModel::GLCompute | ExecModel::TaskEXT | ExecModel::MeshEXT
                ),
                ModelNeed::Mesh => entry.model == ExecModel::MeshEXT,
            };
            if !ok {
                return Err(BuildError::Entry(EntryRule::InstructionForModel(model_op(
                    &node.inst,
                ))));
            }
        }
    }
    Ok(())
}

/// The opcode of an instruction that has a [`ModelNeed`], for error reporting.
fn model_op(inst: &Inst) -> crate::spv::Op {
    use crate::spv::Op;
    match inst {
        Inst::ImageSample { form, .. } => form.to_spirv(),
        Inst::ImageQueryLod { .. } => Op::ImageQueryLod,
        Inst::Derivative { op, .. } => op.to_spirv(),
        Inst::DemoteToHelperInvocation => Op::DemoteToHelperInvocation,
        Inst::IsHelperInvocation => Op::IsHelperInvocationEXT,
        Inst::BeginInterlock => Op::BeginInvocationInterlockEXT,
        Inst::ImageRead { .. } => Op::ImageRead,
        Inst::EndInterlock => Op::EndInvocationInterlockEXT,
        Inst::AtomicLoad { .. } => Op::AtomicLoad,
        Inst::AtomicStore { .. } => Op::AtomicStore,
        Inst::Atomic { op, .. } => op.to_spirv(),
        Inst::AtomicCompareExchange { .. } => Op::AtomicCompareExchange,
        Inst::ControlBarrier { .. } => Op::ControlBarrier,
        Inst::MemoryBarrier { .. } => Op::MemoryBarrier,
        Inst::Group { op, .. } => op.to_spirv(),
        Inst::SetMeshOutputs { .. } => Op::SetMeshOutputsEXT,
        _ => Op::ExtInst,
    }
}
