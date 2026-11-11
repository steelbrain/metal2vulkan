//! Typed decode of `air.*` / `llvm.*` / AGX calls.
//!
//! Every call whose callee is a declaration named `air.*` or `llvm.*` is decoded once, at module
//! decode, into [`AirIntrinsic`] / [`LlvmIntrinsic`]; afterwards no code inspects intrinsic
//! names. Decoding has two halves:
//!
//! 1. **Declaration** ([`classify`]): the name grammar of the family its prefix selects gives a
//!    variant, the variant derives its typed [`Signature`], and the declaration's function type
//!    must match it. This runs once per declaration.
//! 2. **Call** ([`bind`]): the operands the signature marks [`Operand::Imm`] must be integer
//!    constants and are read into the variant's fields.
//!
//! A name outside the closed set is not an error at decode: its calls stay
//! `Callee::UnknownIntrinsic` and the post-canonical check refuses them only if they survive
//! function-constant specialization. The same holds for a known name whose immediate names an
//! enumerator the model does not define. A known name whose declaration disagrees with its
//! signature is [`IntrinsicError::SignatureMismatch`] (malformed input). A call whose immediate is
//! not an integer constant in the function that makes it keeps its [`Template`]
//! (`Callee::Unbound`): an internal function's parameter passed on to the intrinsic, as Metal's
//! compiler leaves OpenCL C's `barrier(flags)`, is each call site's constant once inlining gives
//! the parameter its argument, and the inliner binds the call there. An immediate still not a
//! constant after inlining is [`IntrinsicError::ImmediateNotConstant`]. The `llvm.*` markers
//! [`Dropped`] lists are dropped at decode.
//!
//! A [`Signature`]'s `Operand::Handle` parameters and [`Signature::ret_handle`] give the typed handle positions that
//! seed memory's provenance lattice. Families decoded here: math, conversions and packing, memory
//! atomics and coherent accesses, derivatives, textures and samplers, barriers and fences,
//! simdgroup and quad operations, simdgroup matrices, function constants, fragment operations,
//! pull-model interpolation, imageblocks, async copies, function tables, acceleration-structure
//! queries, ray intersection (`air.intersect`), tensor descriptors, indirect commands, OpenCL C's
//! vector data loads and stores, and the
//! closed `llvm.*` set with the AGX family. Every other `air.*` name (names whose declarations have not been observed) stays
//! unknown. [`UNDECODED`] lists the known families among them, and the intrinsic census of the
//! `bitcode-reader` audit reports every unknown name with row counts. [`UNDECODED_FEATURES`] names
//! the undecoded families that belong to a typed feature not implemented yet (intersection function
//! tables), so their calls are refused as that feature.

mod atomic;
mod convert;
mod intersect;
mod layout;
mod llvm;
mod math;
mod matrix;
mod mesh;
mod objects;
mod sig;
mod simd;
mod stage;
mod sync;
mod texture;
mod ty;
mod vector_data;

#[cfg(test)]
mod families_tests;
#[cfg(test)]
mod intersect_tests;
#[cfg(test)]
mod tests;

pub use atomic::{AccessKind, Atomic, AtomicOp, AtomicSpace, Coherent, MemoryOrder, ThreadScope};
pub use convert::{Convert, Fp8, NumKind, Pack, PackDir, PackFormat};
pub use intersect::{Instancing, Intersect, IntersectTags, ResultField, Role as IntersectRole};
pub use layout::Layout;
pub use llvm::{dropped_family, AgxSpace};
pub use llvm::{AgxOp, Dropped, LlvmFloatOp, LlvmIntOp, LlvmIntrinsic};
pub use math::{Math, MathFn, Precision};
pub use matrix::{MatElem, SimdMatrix};
pub use mesh::Mesh;
pub use objects::{
    AccelerationStructure, AsyncCopy, CommandStage, FunctionTable, IndirectCommand, RenderState,
    Tensor,
};
#[cfg(test)]
pub use sig::ImmKind;
pub use sig::{HandleKind, ImageDim, ImageShape, Immediates, Operand, Signature};
pub use simd::{Group, GroupOp, GroupScope, IsUniform, SimdgroupSize};
pub use stage::Imageblock;
pub use stage::{AbsentAccess, BoolReduce, Derivative, DerivativeOp, Fragment, FunctionConstant};
pub use stage::{GlobalBuiltin, GlobalVar, WorkItem};
pub use stage::{Interpolate, InterpolateAt};
pub use sync::{BarrierName, MemFlags, Sync};
#[cfg(test)]
pub use texture::{Access, Form};
pub use texture::{Component, LodMode, Query, SamplerOp, TexOp, TexRole, Texture};
pub use ty::NumTy;
pub use ty::{Elem, Sign};
pub use vector_data::{HalfRounding, VectorData, VectorDataOp};

use crate::error::Feature;
use crate::error::IntrinsicError;
use crate::ir::{ConstTable, Sym, TypeId, TypeTable, Value};

/// A decoded `air.*` intrinsic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AirIntrinsic {
    /// The math family: `air.[fast_]<function>[.<s|u>].<type>`, and the Metal runtime routines
    /// `frexp` and `ilogb`.
    Math(Math),
    /// `air.convert.<kind>.<type>.<kind>.<type>`: a numeric conversion.
    Convert(Convert),
    /// `air.{pack,unpack}.<format>.<type>`: floats to or from a packed integer.
    Pack(Pack),
    /// `air.atomic.{global,local,private}.<op>.*`: an atomic on memory.
    Atomic(Atomic),
    /// `air.{load,store}.{device,system}_coherent.*`: a plain access with coherent semantics.
    Coherent(Coherent),
    /// `air.[fast_]{dfdx,dfdy}.<type>` and `air.fwidth.<type>`: a screen-space derivative.
    Derivative(Derivative),
    /// `air.{all,any}.v<N>i1`: a boolean vector reduction.
    BoolReduce(BoolReduce),
    /// A texture operation: `air.<op>_{texture,depth}_<dim>[.i16][.<s|u>].<type>`.
    Texture(Texture),
    /// A sampler operation: `air.get_read_sampler`.
    Sampler(SamplerOp),
    /// A barrier or fence: `air.wg.barrier`, `air.simdgroup.barrier`, or `air.atomic.fence`.
    Sync(Sync),
    /// A simdgroup or quad operation: `air.{simd,quad}_<op>[.<s|u>].<type>`.
    Group(Group),
    /// `air.is_uniform.<type>`: whether a value is the same across the simdgroup.
    IsUniform(IsUniform),
    /// `air.get_simdgroup_size.<i16|i32>`: the simdgroup's execution width.
    SimdgroupSize(SimdgroupSize),
    /// `air.get_<built-in>.<i16|i32>`: a kernel built-in a program-scope variable declares.
    GlobalBuiltin(GlobalBuiltin),
    /// `air.get_global_offset.i32` and `air.get_work_dim.i32`: an OpenCL C work-item function no
    /// Metal built-in answers.
    WorkItem(WorkItem),
    /// A simdgroup-matrix operation: `air.simdgroup_matrix_{8x8,16x16x16}_*`.
    SimdMatrix(SimdMatrix),
    /// A function-constant query, resolved by canonicalization's specialization.
    FunctionConstant(FunctionConstant),
    /// A fragment operation: a discard, a rate-map coordinate mapping, or the sample count.
    Fragment(Fragment),
    /// `air.interpolate_<at>_<perspective>.<type>`: an interpolation function of an
    /// `interpolant` fragment input (Metal's pull-model interpolation).
    Interpolate(Interpolate),
    /// An imageblock operation: its data pointer, its size, or an implicit-imageblock load or
    /// store.
    Imageblock(Imageblock),
    /// A simdgroup async copy, or an operation on its events.
    AsyncCopy(AsyncCopy),
    /// An operation on a visible or intersection function table.
    FunctionTable(FunctionTable),
    /// A query of an instance acceleration structure.
    AccelerationStructure(AccelerationStructure),
    /// `air.intersect[.<tag>]*`: an intersector's `intersect` call.
    Intersect(Intersect),
    /// A tensor descriptor operation (Apple's opaque tensor ABI).
    Tensor(Tensor),
    /// An indirect-command-buffer encoding operation on a compute or render command.
    IndirectCommand(IndirectCommand),
    /// An object or mesh function's operation.
    Mesh(Mesh),
    /// OpenCL C's vector data loads and stores: `air.vload[a][_half].<space>.<type>` and
    /// `air.vstore[a][_half].<space>[.<rounding>].<type>`.
    VectorData(VectorData),
}

impl AirIntrinsic {
    /// Parse a full `air.*` name. `None`: not in the decoded families. A name may also be declared
    /// with another operand form ([`classify`] tries each against the declaration's type); this is
    /// the first. Validation reads a texture call's operand roles from it.
    pub fn parse(name: &str) -> Option<AirIntrinsic> {
        let rest = name.strip_prefix("air.")?;
        if let Some(r) = rest.strip_prefix("convert.") {
            return Convert::parse(r).map(AirIntrinsic::Convert);
        }
        if let Some(r) = rest.strip_prefix("pack.") {
            return Pack::parse(PackDir::Pack, r).map(AirIntrinsic::Pack);
        }
        if let Some(r) = rest.strip_prefix("unpack.") {
            return Pack::parse(PackDir::Unpack, r).map(AirIntrinsic::Pack);
        }
        if let Some(s) = Sync::parse(rest) {
            return Some(AirIntrinsic::Sync(s));
        }
        if let Some(r) = rest.strip_prefix("atomic.") {
            return Atomic::parse(r).map(AirIntrinsic::Atomic);
        }
        use AirIntrinsic as A;
        if let Some(i) = Intersect::parse(rest) {
            return Some(A::Intersect(i));
        }
        Imageblock::parse(rest)
            .map(A::Imageblock)
            .or_else(|| {
                let r = rest.strip_prefix("load.")?;
                Coherent::parse(AccessKind::Load, r).map(A::Coherent)
            })
            .or_else(|| {
                let r = rest.strip_prefix("store.")?;
                Coherent::parse(AccessKind::Store, r).map(A::Coherent)
            })
            .or_else(|| Texture::parse(rest).map(A::Texture))
            .or_else(|| SamplerOp::parse(rest).map(A::Sampler))
            .or_else(|| Derivative::parse(rest).map(A::Derivative))
            .or_else(|| BoolReduce::parse(rest).map(A::BoolReduce))
            .or_else(|| Group::parse(rest).map(A::Group))
            .or_else(|| IsUniform::parse(rest).map(A::IsUniform))
            .or_else(|| SimdgroupSize::parse(rest).map(A::SimdgroupSize))
            .or_else(|| GlobalBuiltin::parse(rest).map(A::GlobalBuiltin))
            .or_else(|| WorkItem::parse(rest).map(A::WorkItem))
            .or_else(|| SimdMatrix::parse(rest).map(A::SimdMatrix))
            .or_else(|| FunctionConstant::parse(rest).map(A::FunctionConstant))
            .or_else(|| Fragment::parse(rest).map(A::Fragment))
            .or_else(|| Interpolate::parse(rest).map(A::Interpolate))
            .or_else(|| AsyncCopy::parse(rest).map(A::AsyncCopy))
            .or_else(|| FunctionTable::parse(rest).map(A::FunctionTable))
            .or_else(|| AccelerationStructure::parse(rest).map(A::AccelerationStructure))
            .or_else(|| Tensor::parse(rest).map(A::Tensor))
            .or_else(|| IndirectCommand::parse(rest).map(A::IndirectCommand))
            .or_else(|| Mesh::parse(rest).map(A::Mesh))
            .or_else(|| VectorData::parse(rest).map(A::VectorData))
            .or_else(|| Math::parse(rest).map(A::Math))
    }

    /// Another operand form the same name may be declared with (a texture read's legacy form, a
    /// 64-bit atomic min or max without its result, the AIR 2.9 form of an atomic or a barrier,
    /// the AIR 2.7 form of a simdgroup-matrix block load or store, the AIR 2.5 kernel or vertex
    /// buffer setter without its attribute stride).
    fn alternative(self) -> Option<AirIntrinsic> {
        match self {
            AirIntrinsic::Texture(t) => t.alternative().map(AirIntrinsic::Texture),
            AirIntrinsic::Atomic(a) => a.alternative().map(AirIntrinsic::Atomic),
            AirIntrinsic::Sync(s) => s.alternative().map(AirIntrinsic::Sync),
            AirIntrinsic::SimdMatrix(m) => m.alternative().map(AirIntrinsic::SimdMatrix),
            AirIntrinsic::IndirectCommand(c) => c.alternative().map(AirIntrinsic::IndirectCommand),
            _ => None,
        }
    }

    /// The typed signature: result and operand positions, including the immediate and handle
    /// positions (the handle typing's seeds).
    pub fn signature(&self) -> Signature {
        match self {
            AirIntrinsic::Math(m) => m.signature(),
            AirIntrinsic::Convert(c) => c.signature(),
            AirIntrinsic::Pack(p) => p.signature(),
            AirIntrinsic::Atomic(a) => a.signature(),
            AirIntrinsic::Coherent(c) => c.signature(),
            AirIntrinsic::Derivative(d) => d.signature(),
            AirIntrinsic::BoolReduce(b) => b.signature(),
            AirIntrinsic::Texture(t) => t.signature(),
            AirIntrinsic::Sampler(s) => s.signature(),
            AirIntrinsic::Sync(s) => s.signature(),
            AirIntrinsic::Group(g) => g.signature(),
            AirIntrinsic::IsUniform(u) => u.signature(),
            AirIntrinsic::SimdgroupSize(s) => s.signature(),
            AirIntrinsic::GlobalBuiltin(g) => g.signature(),
            AirIntrinsic::WorkItem(w) => w.signature(),
            AirIntrinsic::SimdMatrix(m) => m.signature(),
            AirIntrinsic::FunctionConstant(f) => f.signature(),
            AirIntrinsic::Fragment(f) => f.signature(),
            AirIntrinsic::Interpolate(i) => i.signature(),
            AirIntrinsic::Imageblock(i) => i.signature(),
            AirIntrinsic::AsyncCopy(a) => a.signature(),
            AirIntrinsic::FunctionTable(f) => f.signature(),
            AirIntrinsic::AccelerationStructure(a) => a.signature(),
            AirIntrinsic::Intersect(i) => i.signature(),
            AirIntrinsic::Tensor(t) => t.signature(),
            AirIntrinsic::IndirectCommand(c) => c.signature(),
            AirIntrinsic::Mesh(m) => m.signature(),
            AirIntrinsic::VectorData(v) => v.signature(),
        }
    }

    /// Read one call's immediates. `Ok(None)`: an immediate outside the modeled enumerators.
    fn bind(self, imm: &Immediates<'_>) -> Result<Option<AirIntrinsic>, IntrinsicError> {
        Ok(Some(match self {
            AirIntrinsic::Math(m) => match m.bind(imm)? {
                Some(m) => AirIntrinsic::Math(m),
                None => return Ok(None),
            },
            AirIntrinsic::Atomic(a) => match a.bind(imm)? {
                Some(a) => AirIntrinsic::Atomic(a),
                None => return Ok(None),
            },
            AirIntrinsic::Texture(t) => match t.bind(imm)? {
                Some(t) => AirIntrinsic::Texture(t),
                None => return Ok(None),
            },
            AirIntrinsic::Sync(s) => match s.bind(imm)? {
                Some(s) => AirIntrinsic::Sync(s),
                None => return Ok(None),
            },
            other => other,
        }))
    }
}

/// The `air.*` families the decoder knows of but does not decode, each with the reason: their
/// calls stay `Callee::UnknownIntrinsic`. This is the "listed" half of the census contract: every
/// corpus name decodes or matches an entry here. Each entry is a name prefix.
pub const UNDECODED: &[(&str, &str)] = &[
    (
        "air.function_constant_predicate",
        "no declaration observed in the corpus or the fixtures",
    ),
    (
        "air.set_intersection_function_table.",
        "no declaration observed in the corpus or the fixtures",
    ),
    // A conversion to an 8-bit float. On Apple M3 (AGXMetalG15X, macOS 26.7), Metal does not
    // compile one from a float (its compiler service is interrupted); one from an integer compiles
    // and runs but returns bytes that are not the format's encodings and are the same for every
    // format. MSL defines no rounding, overflow, or saturation for these `air.convert`s (MSL 4.1
    // has 8-bit floats only through `pack` with explicit modes), and no device here can pin them.
    // The prefixes cover the scalar destinations, the only ones observed; a vector destination
    // (`air.convert.f.v<N>f8<format>.`) matches none of them, so such a name would stay an unlisted
    // unknown intrinsic.
    ("air.convert.f.f8e4m3.", FP8_NARROWING),
    ("air.convert.f.f8e4m3fn.", FP8_NARROWING),
    ("air.convert.f.f8e5m2.", FP8_NARROWING),
    // A multisample imageblock's operations: its sample count (`get_num_samples`), the colors its
    // pixel holds and each one's coverage mask (`get_num_colors`, `get_color_coverage_mask`), and
    // a store to the samples a coverage mask names. Typed as [`UNDECODED_FEATURES`].
    ("air.get_imageblock_samples", MULTISAMPLE_IMAGEBLOCK),
    ("air.get_imageblock_num_colors", MULTISAMPLE_IMAGEBLOCK),
    ("air.get_color_coverage_mask", MULTISAMPLE_IMAGEBLOCK),
    (
        "air.store.implicit_imageblock.mask.",
        MULTISAMPLE_IMAGEBLOCK,
    ),
];

/// The reason a multisample imageblock's operations are [`UNDECODED`].
const MULTISAMPLE_IMAGEBLOCK: &str = "a multisample imageblock's sample count, colors, coverage \
                                      masks and masked stores: tile functions and fragment \
                                      imageblocks run on single-sample passes only, and how many \
                                      colors Metal keeps for a pixel is not probed";

/// The reason the conversions to an 8-bit float are [`UNDECODED`].
const FP8_NARROWING: &str = "Metal on M3 (AGXMetalG15X, macOS 26.7) does not compile a float \
                             source (XPC interruption), and for an integer source returns bytes \
                             that are not fp8 encodings and ignore the format; MSL defines no \
                             air.convert narrowing rule; no device here can pin it";

/// The [`UNDECODED`] entry `name` belongs to, if any.
pub fn undecoded_family(name: &str) -> Option<&'static (&'static str, &'static str)> {
    UNDECODED
        .iter()
        .find(|(prefix, _)| name.starts_with(prefix))
}

/// The undecoded `air.*` families that belong to a typed feature not implemented yet, by the name
/// prefix of the stable AIR ABI, with that feature. A call to one of them that survives
/// canonicalization is refused as that feature (`NotImplemented`) rather than as an unknown
/// intrinsic; so is a name of the ray tracing families that does not decode
/// ([`undecoded_feature`]); every other undecoded name stays `Feature::UnknownIntrinsic`. Every
/// prefix here is also an [`UNDECODED`] prefix, so no such name ever decodes.
pub const UNDECODED_FEATURES: &[(&str, Feature)] = &[
    // Binding an intersection function table, as its decoded sibling
    // `air.set_buffer_intersection_function_table` is.
    (
        "air.set_intersection_function_table.",
        Feature::IntersectionFunctionTable,
    ),
    // A multisample imageblock's operations: imageblocks, on a single-sample pass only.
    ("air.get_imageblock_samples", Feature::Imageblocks),
    ("air.get_imageblock_num_colors", Feature::Imageblocks),
    ("air.get_color_coverage_mask", Feature::Imageblocks),
    ("air.store.implicit_imageblock.mask.", Feature::Imageblocks),
];

/// The typed feature of the undecoded intrinsic `name` ([`UNDECODED_FEATURES`]), if it belongs
/// to such a family, or `RayTracing` for an undecoded name of the ray tracing families (an
/// intersector name the grammar rejects, or an `intersection_query` operation). `name` is a
/// declaration's symbol bytes.
pub fn undecoded_feature(name: &[u8]) -> Option<Feature> {
    UNDECODED_FEATURES
        .iter()
        .find(|(prefix, _)| name.starts_with(prefix.as_bytes()))
        .map(|(_, feature)| feature.clone())
        .or_else(|| undecoded_ray_tracing(name).then_some(Feature::RayTracing))
}

/// Whether `name` is a ray tracing intrinsic that does not decode, by the first component of its
/// stable AIR name: an intersector's `air.intersect[.<tag>]*` whose tags the grammar rejects (a
/// later tag, a new combination), or an operation of Metal's `intersection_query`,
/// `air.<op>_intersection_query[.<tag>]*` (`next`, `reset`, `commit_triangle_intersection`, the
/// candidate and committed getters), which is not decoded yet.
fn undecoded_ray_tracing(name: &[u8]) -> bool {
    let Some(rest) = name.strip_prefix(b"air.") else {
        return false;
    };
    let head = rest.split(|&b| b == b'.').next().unwrap_or_default();
    if head == b"intersect" {
        return std::str::from_utf8(rest).map_or(true, |r| Intersect::parse(r).is_none());
    }
    head.ends_with(b"_intersection_query")
}

/// What a declaration named `air.*` / `llvm.*` decodes to. The `Air` / `Llvm` variants hold the
/// declaration's template: its immediate fields are placeholders until [`bind`] reads a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeclClass {
    /// An `air.*` declaration (or a runtime routine the math family models), as its template.
    Air(AirIntrinsic),
    /// A kept `llvm.*` declaration, as its template.
    Llvm(LlvmIntrinsic),
    /// A marker decode drops (its calls produce no instruction).
    Drop(Dropped),
    /// Outside the closed set: calls stay `Callee::UnknownIntrinsic`.
    Unknown,
    /// A known name whose declaration disagrees with its typed signature; any call is
    /// [`IntrinsicError::SignatureMismatch`].
    Mismatch,
}

/// The Metal runtime routine a declaration named `name` is, when the math family models it:
/// `___metal_frexp_*` or `___metal_ilogb_*`.
fn routine(name: &[u8]) -> Option<Math> {
    let rest = name.strip_prefix(b"___metal_")?;
    Math::parse_routine(std::str::from_utf8(rest).ok()?)
}

/// Whether a declaration named `name` is an intrinsic: its name has the `air.` or the `llvm.`
/// prefix of the stable ABI, or it is a Metal runtime routine the math family models (`frexp`,
/// `ilogb`). Any other declaration is an ordinary function.
pub fn is_intrinsic_name(name: &[u8]) -> bool {
    name.starts_with(b"air.") || name.starts_with(b"llvm.") || routine(name).is_some()
}

/// Classify the declaration `name` (symbol `sym`) of function type `fn_ty`. `None`: the name
/// has neither the `air.` nor the `llvm.` prefix and is no modeled runtime routine, so it is an
/// ordinary function.
pub fn classify(name: &[u8], sym: Sym, fn_ty: TypeId, types: &TypeTable) -> Option<DeclClass> {
    if let Some(op) = routine(name) {
        let op = AirIntrinsic::Math(op);
        return Some(if op.signature().matches(types, fn_ty) {
            DeclClass::Air(op)
        } else {
            DeclClass::Mismatch
        });
    }
    if !is_intrinsic_name(name) {
        return None;
    }
    // Every modeled name is ASCII; anything else is outside the set.
    let Ok(name) = std::str::from_utf8(name) else {
        return Some(DeclClass::Unknown);
    };
    let class = if let Some(rest) = name.strip_prefix("llvm.") {
        match llvm::parse(rest, sym, types, fn_ty) {
            Ok(llvm::Parsed::Dropped(d)) => DeclClass::Drop(d),
            Ok(llvm::Parsed::Kept(op)) => match op.signature() {
                Some(sig) if !sig.matches(types, fn_ty) => DeclClass::Mismatch,
                _ => DeclClass::Llvm(op),
            },
            Err(llvm::Fail::Unknown) => DeclClass::Unknown,
            Err(llvm::Fail::Mismatch) => DeclClass::Mismatch,
        }
    } else {
        match AirIntrinsic::parse(name) {
            None => DeclClass::Unknown,
            Some(op) => std::iter::successors(Some(op), |o| o.alternative())
                .find(|o| o.signature().matches(types, fn_ty))
                .map_or(DeclClass::Mismatch, DeclClass::Air),
        }
    };
    Some(class)
}

impl DeclClass {
    /// The template of an intrinsic in the closed set: `None` for a dropped marker, an unknown
    /// name or a mismatched declaration.
    pub fn template(self) -> Option<Template> {
        match self {
            DeclClass::Air(op) => Some(Template::Air(op)),
            DeclClass::Llvm(op) => Some(Template::Llvm(op)),
            DeclClass::Drop(_) | DeclClass::Unknown | DeclClass::Mismatch => None,
        }
    }
}

/// An intrinsic in the closed set before a call's immediates are read in: the template its
/// declaration classifies as. A call whose immediate operand is not an integer constant in the
/// function that makes it keeps its template (`Callee::Unbound`) until canonicalization reads
/// the immediates where inlining gives them each call site's arguments ([`Template::bind`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Template {
    /// An `air.*` intrinsic's template.
    Air(AirIntrinsic),
    /// An `llvm.*` intrinsic's template.
    Llvm(LlvmIntrinsic),
}

impl Template {
    /// Bind one call with `args` to an intrinsic of this template, as [`bind`] binds a call to
    /// its declaration's class.
    pub fn bind(self, args: &[Value], consts: &ConstTable) -> Result<Bound, IntrinsicError> {
        let class = match self {
            Template::Air(op) => DeclClass::Air(op),
            Template::Llvm(op) => DeclClass::Llvm(op),
        };
        bind(class, args, consts)
    }
}

/// A call's decoded callee, from its declaration's class and its arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bound {
    /// An `air.*` intrinsic with the call's immediates read in.
    Air(AirIntrinsic),
    /// An `llvm.*` intrinsic with the call's immediates read in.
    Llvm(LlvmIntrinsic),
    /// Keep the call as `Callee::UnknownIntrinsic`.
    Unknown,
}

/// Bind one call with `args` to a declaration of class `class` (not [`DeclClass::Drop`], whose
/// calls decode never builds; it binds as unknown).
pub fn bind(
    class: DeclClass,
    args: &[Value],
    consts: &ConstTable,
) -> Result<Bound, IntrinsicError> {
    match class {
        DeclClass::Air(op) => {
            let imm = Immediates::new(op.signature(), args, consts);
            Ok(op.bind(&imm)?.map_or(Bound::Unknown, Bound::Air))
        }
        DeclClass::Llvm(op) => match op.signature() {
            Some(sig) => Ok(Bound::Llvm(op.bind(&Immediates::new(sig, args, consts))?)),
            None => Ok(Bound::Llvm(op)),
        },
        DeclClass::Mismatch => Err(IntrinsicError::SignatureMismatch),
        DeclClass::Drop(_) | DeclClass::Unknown => Ok(Bound::Unknown),
    }
}
