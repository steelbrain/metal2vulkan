//! The typed module: values, globals, functions, blocks, instructions, entry point.
//!
//! A [`TypedModule`] is produced only by `ModuleBuilder::finalize`, so every one is built through
//! the typed constructors. Every id kind is dense and module-wide: a [`BlockId`] or [`VarId`]
//! names one block or variable of the whole module, so module-level tables (`loops`, relocations)
//! need no function qualifier. SPIR-V result ids are assigned only at serialization.

use crate::device::requirements::Requirements;
use crate::device::subgroup::SubgroupRequirement;
use crate::pipeline::sym::{PushField, ResourceSlot, VaryingKey};
use crate::spirv::error::{BResult, BuildError, Dangling};
use crate::spirv::ids::{BlockId, ConstId, FuncId, GlobalId, TypeId, Value, VarId};
use crate::spirv::ir::constant::Consts;
use crate::spirv::ir::inst::{
    AtomicOp, BinOp, Contraction, ConvOp, DerivOp, GlslOp, GroupInst, ImageOperands, ImageQuery,
    ModelNeed, RayQueryInst, SampleForm, Scope, ScopeId, SemanticsId, TexelMemory, UnOp,
};
use crate::spirv::ir::types::{IoDir, StorageClass, Types};

/// Where an SSA value comes from.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueOrigin {
    /// A constant (plain or specialization).
    Const(ConstId),
    /// The pointer of a module-scope variable.
    Global(GlobalId),
    /// The pointer of a `Function`-storage variable.
    Var(VarId),
    /// Parameter `index` of `func`.
    Param { func: FuncId, index: u32 },
    /// The result of an instruction in `block`.
    Inst(BlockId),
    /// The result of an `OpPhi` in `block`.
    Phi(BlockId),
    /// A parameter, variable, or result of a function that left the module because no call
    /// reaches it ([`TypedModule::remove_dead`]): nothing may read it.
    Removed,
}

/// One SSA value: its type and origin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueDef {
    pub ty: TypeId,
    pub origin: ValueOrigin,
}

/// Interpolation of a user varying.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Interpolation {
    Smooth,
    Flat,
    NoPerspective,
}

/// Sampling of a user varying.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sampling {
    Center,
    Centroid,
    Sample,
}

/// Decorations of a user varying. `Location` (and `Component` when `packed`) are relocatable
/// symbols of the varying's key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IoDecor {
    pub interpolation: Interpolation,
    pub sampling: Sampling,
    /// Link may pack this varying into a component of a shared location.
    pub packed: bool,
    /// `Index` for dual-source blending outputs.
    pub index: Option<u32>,
}

impl Default for IoDecor {
    fn default() -> IoDecor {
        IoDecor {
            interpolation: Interpolation::Smooth,
            sampling: Sampling::Center,
            packed: false,
            index: None,
        }
    }
}

/// What a module-scope variable is for; the serializer derives its decorations from this.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GlobalKind {
    /// A descriptor binding (relocatable `DescriptorSet`/`Binding`).
    Resource(ResourceSlot),
    /// The function's push-constant block.
    PushBlock,
    /// A user varying.
    Io {
        key: VaryingKey,
        dir: IoDir,
        deco: IoDecor,
    },
    /// A builtin variable.
    BuiltIn {
        builtin: crate::spv::BuiltIn,
        dir: IoDir,
    },
    /// A `Private` or `Workgroup` variable.
    Plain,
    /// A user variable at a fixed `Location` between stages the crate synthesizes, which no
    /// Metal varying reaches (the tessellation chain); `patch` decorates it `Patch`.
    FixedIo {
        location: u32,
        dir: IoDir,
        patch: bool,
    },
}

/// A module-scope variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlobalVar {
    pub class: StorageClass,
    /// The variable's data type.
    pub pointee: TypeId,
    /// The interned `Pointer { class, pointee }`.
    pub ptr_ty: TypeId,
    /// The variable's pointer value.
    pub value: Value,
    pub init: Option<ConstId>,
    pub kind: GlobalKind,
}

/// A `Function`-storage variable. It is declared in its function's entry block, whatever block
/// requested it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalVar {
    pub func: FuncId,
    pub pointee: TypeId,
    pub ptr_ty: TypeId,
    pub value: Value,
    pub init: Option<ConstId>,
}

/// One member of the push-constant block, placed by `select`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PushMember {
    pub field: PushField,
    /// The member index in the push block.
    pub index: u32,
    pub ty: TypeId,
    pub offset: u32,
}

/// Memory operands of a load or store.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MemAccess {
    pub volatile: bool,
    /// `Aligned` (a power of two). Required through `PhysicalStorageBuffer` pointers.
    pub aligned: Option<u32>,
    pub nontemporal: bool,
    /// `MakePointerAvailable` at this scope (stores only; Vulkan memory model; needs
    /// `non_private`).
    pub make_available: Option<ScopeId>,
    /// `MakePointerVisible` at this scope (loads only; Vulkan memory model; needs
    /// `non_private`).
    pub make_visible: Option<ScopeId>,
    /// `NonPrivatePointer`: the access takes part in inter-invocation ordering (Vulkan memory
    /// model; `StorageBuffer`, `PhysicalStorageBuffer`, `Uniform`, or `Workgroup` pointers).
    pub non_private: bool,
}

impl MemAccess {
    /// Whether the operands need the Vulkan memory model.
    pub fn vulkan_memory_model(&self) -> bool {
        self.non_private || self.make_available.is_some() || self.make_visible.is_some()
    }
}

/// An instruction. A closed enum over the opcodes the new core emits; there is no generic
/// operand-list escape hatch. Opcodes are grouped by family; each family enum in
/// [`inst`](super::inst) names one opcode per variant.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Inst {
    /// `OpLoad`: the result has the pointee type.
    Load { ptr: Value, access: MemAccess },
    /// `OpStore`.
    Store {
        ptr: Value,
        value: Value,
        access: MemAccess,
    },
    // ---- Arithmetic, bitwise, logical, conversion (builder/ops/{arith,convert}.rs) ----
    /// A unary operation.
    Unary { op: UnOp, a: Value },
    /// A binary operation. `contract` is [`Contraction::Forbidden`] only for float arithmetic
    /// the serializer decorates `NoContraction`.
    Binary {
        op: BinOp,
        a: Value,
        b: Value,
        contract: Contraction,
    },
    /// `OpSelect`.
    Select { cond: Value, t: Value, f: Value },
    /// `OpBitFieldInsert`.
    BitFieldInsert {
        base: Value,
        insert: Value,
        offset: Value,
        count: Value,
    },
    /// `OpFmaKHR` (`SPV_KHR_fma`): `a * b + c` rounded once (Vulkan: "Correctly rounded").
    Fma { a: Value, b: Value, c: Value },
    /// `OpBitFieldSExtract` (`signed`) or `OpBitFieldUExtract`.
    BitFieldExtract {
        signed: bool,
        base: Value,
        offset: Value,
        count: Value,
    },
    /// A conversion; the target type is the result's type. `contract` is
    /// [`Contraction::Forbidden`] only on an `FConvert`: a conversion decorated `NoContraction`,
    /// which the device may not remove or merge with another operation (without it Vulkan assumes
    /// `AllowTransform`, under which `f32 -> f16 -> f32` may become `f32` alone).
    Convert {
        op: ConvOp,
        a: Value,
        contract: Contraction,
    },
    // ---- Composites (builder/ops/composite.rs) ----
    /// `OpCompositeConstruct`.
    CompositeConstruct { parts: Box<[Value]> },
    /// `OpCompositeExtract`.
    CompositeExtract {
        composite: Value,
        indices: Box<[u32]>,
    },
    /// `OpCompositeInsert`.
    CompositeInsert {
        object: Value,
        composite: Value,
        indices: Box<[u32]>,
    },
    /// `OpVectorExtractDynamic`.
    VectorExtractDynamic { vector: Value, index: Value },
    /// `OpVectorInsertDynamic`.
    VectorInsertDynamic {
        vector: Value,
        component: Value,
        index: Value,
    },
    /// `OpVectorShuffle` (no undefined components).
    VectorShuffle {
        a: Value,
        b: Value,
        components: Box<[u32]>,
    },
    /// `OpUndef` of the result type.
    Undef,
    // ---- Memory (builder/ops/memory.rs) ----
    /// `OpAccessChain`. `non_uniform` marks an element of a descriptor array chosen by an index
    /// that may differ between invocations: the result and every value that carries the element
    /// on ([`Inst::carries_non_uniform`]) are decorated `NonUniform`
    /// (VUID-RuntimeSpirv-NonUniform-06274), and derivation requires the array kind's non-uniform
    /// indexing capability instead of its dynamic one.
    AccessChain {
        base: Value,
        indices: Box<[Value]>,
        non_uniform: bool,
    },
    /// `OpArrayLength` of the runtime array member `member` of the pointed-to `Block`.
    ArrayLength { ptr: Value, member: u32 },
    /// `OpFunctionCall`.
    Call { func: FuncId, args: Box<[Value]> },
    // ---- Atomics and barriers (builder/ops/atomic.rs) ----
    /// `OpAtomicLoad`.
    AtomicLoad {
        ptr: Value,
        scope: ScopeId,
        sem: SemanticsId,
    },
    /// `OpAtomicStore`.
    AtomicStore {
        ptr: Value,
        scope: ScopeId,
        sem: SemanticsId,
        value: Value,
    },
    /// A read-modify-write atomic with one value operand.
    Atomic {
        op: AtomicOp,
        ptr: Value,
        scope: ScopeId,
        sem: SemanticsId,
        value: Value,
    },
    /// `OpAtomicCompareExchange`.
    AtomicCompareExchange {
        ptr: Value,
        scope: ScopeId,
        equal: SemanticsId,
        unequal: SemanticsId,
        value: Value,
        comparator: Value,
    },
    /// `OpImageTexelPointer`: an `Image`-class pointer, valid only as an atomic operand.
    ImageTexelPointer {
        image: Value,
        coord: Value,
        sample: Value,
    },
    /// `OpControlBarrier`.
    ControlBarrier {
        exec: ScopeId,
        mem: ScopeId,
        sem: SemanticsId,
    },
    /// `OpMemoryBarrier`.
    MemoryBarrier { mem: ScopeId, sem: SemanticsId },
    // ---- Images and derivatives (builder/ops/image.rs) ----
    /// `OpSampledImage`.
    SampledImage { image: Value, sampler: Value },
    /// `OpImage`: the image of a sampled image.
    Image(Value),
    /// One of the eight `OpImageSample*` opcodes.
    ImageSample {
        form: SampleForm,
        sampled_image: Value,
        coord: Value,
        dref: Option<Value>,
        ops: ImageOperands,
    },
    /// `OpImageFetch`, or `OpImageSparseFetch` (`sparse`).
    ImageFetch {
        image: Value,
        coord: Value,
        ops: ImageOperands,
        sparse: bool,
    },
    /// `OpImageGather` of component `component` (carried by constant `component_id`), or
    /// `OpImageSparseGather` (`sparse`).
    ImageGather {
        sampled_image: Value,
        coord: Value,
        component: u32,
        component_id: ConstId,
        ops: ImageOperands,
        sparse: bool,
    },
    /// `OpImageDrefGather`, or `OpImageSparseDrefGather` (`sparse`).
    ImageDrefGather {
        sampled_image: Value,
        coord: Value,
        dref: Value,
        ops: ImageOperands,
        sparse: bool,
    },
    /// `OpImageRead`, or `OpImageSparseRead` (`sparse`).
    ImageRead {
        image: Value,
        coord: Value,
        ops: ImageOperands,
        sparse: bool,
        mem: TexelMemory,
    },
    /// `OpImageSparseTexelsResident`: whether the residency code of a sparse access says every
    /// texel it accessed was resident.
    ImageSparseTexelsResident(Value),
    /// `OpImageWrite`.
    ImageWrite {
        image: Value,
        coord: Value,
        texel: Value,
        ops: ImageOperands,
        mem: TexelMemory,
    },
    /// An image query.
    ImageQuery { query: ImageQuery, image: Value },
    /// `OpImageQueryLod`.
    ImageQueryLod { sampled_image: Value, coord: Value },
    /// A derivative.
    Derivative { op: DerivOp, a: Value },
    // ---- Subgroup operations (builder/ops/group.rs) ----
    /// A `GroupNonUniform*` operation at `Subgroup` scope.
    Group { scope: ScopeId, op: GroupInst },
    /// A `GLSL.std.450` instruction. `contract` is [`Contraction::Forbidden`] only for an `Fma`
    /// the serializer decorates `NoContraction`: a multiply rounded on its own.
    Glsl {
        op: GlslOp,
        args: Box<[Value]>,
        contract: Contraction,
    },
    /// `OpBeginInvocationInterlockEXT`.
    BeginInterlock,
    /// `OpEndInvocationInterlockEXT`.
    EndInterlock,
    /// `OpDemoteToHelperInvocation` (`SPV_EXT_demote_to_helper_invocation` in SPIR-V 1.5): the
    /// invocation keeps executing as a helper. Only `FunctionBuilder::demote` creates it.
    DemoteToHelperInvocation,
    /// `OpIsHelperInvocationEXT` (`SPV_EXT_demote_to_helper_invocation`): whether the invocation
    /// is a helper, a demoted one included. Only `FunctionBuilder::is_helper_invocation` creates
    /// it.
    IsHelperInvocation,
    /// `OpSetMeshOutputsEXT`: the vertices and primitives a mesh workgroup outputs.
    SetMeshOutputs { vertices: Value, primitives: Value },
    /// `OpEmitVertex`: a geometry entry emits a vertex of its output values.
    EmitVertex,
    /// `OpEndPrimitive`: a geometry entry ends its output primitive.
    EndPrimitive,
    // ---- Ray queries (builder/ops/ray.rs) ----
    /// A `SPV_KHR_ray_query` instruction.
    RayQuery(RayQueryInst),
}

impl Inst {
    /// The heap bytes of the instruction's boxed operand list, for the byte ledger.
    pub fn heap_bytes(&self) -> usize {
        match self {
            Inst::CompositeConstruct { parts: v }
            | Inst::AccessChain { indices: v, .. }
            | Inst::Call { args: v, .. }
            | Inst::Glsl { args: v, .. } => std::mem::size_of_val::<[Value]>(v),
            Inst::CompositeExtract { indices: v, .. }
            | Inst::CompositeInsert { indices: v, .. }
            | Inst::VectorShuffle { components: v, .. } => std::mem::size_of_val::<[u32]>(v),
            _ => 0,
        }
    }

    /// Whether the instruction is tangled: its result depends on which invocations execute it
    /// together (implicit-LOD sampling, `OpImageQueryLod`, derivatives, subgroup operations), so
    /// the structurizer may not duplicate it or move it into divergent control flow.
    pub fn is_tangled(&self) -> bool {
        match self {
            Inst::ImageSample { form, .. } => form.implicit,
            Inst::ImageQueryLod { .. } | Inst::Derivative { .. } | Inst::Group { .. } => true,
            _ => false,
        }
    }

    /// Whether the result carries a non-uniform descriptor element to the instruction that consumes
    /// it, given which operands do (`marked`) and whether the result is `opaque` (an image,
    /// sampler, or sampled image): a non-uniform access chain or a chain into a block element one
    /// reached, an opaque load of one (the image or sampler), a texel pointer into one (an image
    /// atomic's pointer), a sampled image of such an image or sampler, and the image of such a
    /// sampled image. Each result is decorated `NonUniform` (VUID-RuntimeSpirv-NonUniform-06274);
    /// the builder refuses the other uses that would carry the element on. Data loaded from a block
    /// element carries none.
    pub fn carries_non_uniform(&self, marked: impl Fn(Value) -> bool, opaque: bool) -> bool {
        match self {
            Inst::AccessChain {
                non_uniform, base, ..
            } => *non_uniform || marked(*base),
            Inst::Load { ptr, .. } => opaque && marked(*ptr),
            Inst::ImageTexelPointer { image, .. } => marked(*image),
            Inst::SampledImage { image, sampler } => marked(*image) || marked(*sampler),
            Inst::Image(sampled_image) => marked(*sampled_image),
            _ => false,
        }
    }

    /// Whether the result is a place: a pointer an atomic may operate on. Variables are places too;
    /// they are not instructions.
    pub fn is_place(&self) -> bool {
        matches!(
            self,
            Inst::AccessChain { .. } | Inst::ImageTexelPointer { .. }
        )
    }

    /// The pointer operand of an atomic, `None` for every other instruction.
    pub fn atomic_pointer(&self) -> Option<Value> {
        match self {
            Inst::AtomicLoad { ptr, .. }
            | Inst::AtomicStore { ptr, .. }
            | Inst::Atomic { ptr, .. }
            | Inst::AtomicCompareExchange { ptr, .. } => Some(*ptr),
            _ => None,
        }
    }

    /// Call `f` on every SSA value operand, in operand order. Constant operands carried as
    /// [`ConstId`]s (scopes, semantics, gather components, cluster sizes, quad directions) are not
    /// SSA values of the instruction and are not visited.
    pub fn for_each_value(&self, mut f: impl FnMut(Value)) {
        fn ops(o: &ImageOperands, f: &mut dyn FnMut(Value)) {
            let ImageOperands {
                bias,
                lod,
                grad,
                const_offset,
                offset,
                sample,
                min_lod,
            } = *o;
            for v in [bias, lod].into_iter().flatten() {
                f(v);
            }
            if let Some((dx, dy)) = grad {
                f(dx);
                f(dy);
            }
            for v in [const_offset, offset, sample, min_lod]
                .into_iter()
                .flatten()
            {
                f(v);
            }
        }
        match self {
            Inst::Load { ptr, .. } | Inst::ArrayLength { ptr, .. } => f(*ptr),
            Inst::Store { ptr, value, .. } => {
                f(*ptr);
                f(*value);
            }
            Inst::Unary { a, .. }
            | Inst::Convert { a, .. }
            | Inst::Image(a)
            | Inst::Derivative { a, .. } => f(*a),
            Inst::Binary { a, b, .. } => {
                f(*a);
                f(*b);
            }
            Inst::Select { cond, t, f: e } => {
                f(*cond);
                f(*t);
                f(*e);
            }
            Inst::Fma { a, b, c } => {
                f(*a);
                f(*b);
                f(*c);
            }
            Inst::BitFieldInsert {
                base,
                insert,
                offset,
                count,
            } => {
                for v in [*base, *insert, *offset, *count] {
                    f(v);
                }
            }
            Inst::BitFieldExtract {
                base,
                offset,
                count,
                ..
            } => {
                for v in [*base, *offset, *count] {
                    f(v);
                }
            }
            Inst::CompositeConstruct { parts } => parts.iter().copied().for_each(f),
            Inst::CompositeExtract { composite, .. } => f(*composite),
            Inst::CompositeInsert {
                object, composite, ..
            } => {
                f(*object);
                f(*composite);
            }
            Inst::VectorExtractDynamic { vector, index } => {
                f(*vector);
                f(*index);
            }
            Inst::VectorInsertDynamic {
                vector,
                component,
                index,
            } => {
                f(*vector);
                f(*component);
                f(*index);
            }
            Inst::VectorShuffle { a, b, .. } => {
                f(*a);
                f(*b);
            }
            Inst::Undef
            | Inst::BeginInterlock
            | Inst::EndInterlock
            | Inst::DemoteToHelperInvocation
            | Inst::IsHelperInvocation
            | Inst::ControlBarrier { .. }
            | Inst::MemoryBarrier { .. }
            | Inst::EmitVertex
            | Inst::EndPrimitive => {}
            Inst::SetMeshOutputs {
                vertices,
                primitives,
            } => {
                f(*vertices);
                f(*primitives);
            }
            Inst::AccessChain { base, indices, .. } => {
                f(*base);
                indices.iter().copied().for_each(f);
            }
            Inst::Call { args, .. } => args.iter().copied().for_each(f),
            Inst::AtomicLoad { ptr, .. } => f(*ptr),
            Inst::AtomicStore { ptr, value, .. } | Inst::Atomic { ptr, value, .. } => {
                f(*ptr);
                f(*value);
            }
            Inst::AtomicCompareExchange {
                ptr,
                value,
                comparator,
                ..
            } => {
                f(*ptr);
                f(*value);
                f(*comparator);
            }
            Inst::ImageTexelPointer {
                image,
                coord,
                sample,
            } => {
                f(*image);
                f(*coord);
                f(*sample);
            }
            Inst::SampledImage { image, sampler } => {
                f(*image);
                f(*sampler);
            }
            Inst::ImageSample {
                sampled_image,
                coord,
                dref,
                ops: o,
                ..
            } => {
                f(*sampled_image);
                f(*coord);
                if let Some(d) = dref {
                    f(*d);
                }
                ops(o, &mut f);
            }
            Inst::ImageFetch {
                image,
                coord,
                ops: o,
                ..
            }
            | Inst::ImageRead {
                image,
                coord,
                ops: o,
                ..
            } => {
                f(*image);
                f(*coord);
                ops(o, &mut f);
            }
            Inst::ImageSparseTexelsResident(code) => f(*code),
            Inst::ImageGather {
                sampled_image,
                coord,
                ops: o,
                ..
            } => {
                f(*sampled_image);
                f(*coord);
                ops(o, &mut f);
            }
            Inst::ImageDrefGather {
                sampled_image,
                coord,
                dref,
                ops: o,
                ..
            } => {
                f(*sampled_image);
                f(*coord);
                f(*dref);
                ops(o, &mut f);
            }
            Inst::ImageWrite {
                image,
                coord,
                texel,
                ops: o,
                ..
            } => {
                f(*image);
                f(*coord);
                f(*texel);
                ops(o, &mut f);
            }
            Inst::ImageQuery { query, image } => {
                f(*image);
                if let ImageQuery::SizeLod(lod) = query {
                    f(*lod);
                }
            }
            Inst::ImageQueryLod {
                sampled_image,
                coord,
            } => {
                f(*sampled_image);
                f(*coord);
            }
            Inst::Group { op, .. } => op.for_each_value(f),
            Inst::Glsl { args, .. } => args.iter().copied().for_each(f),
            Inst::RayQuery(op) => op.for_each_value(f),
        }
    }

    /// [`Inst::for_each_value`] with each operand open to change, in the same order: a transform
    /// that renames values (`spirv::roll`) rewrites an instruction in place.
    pub fn for_each_value_mut(&mut self, mut f: impl FnMut(&mut Value)) {
        fn ops(o: &mut ImageOperands, f: &mut dyn FnMut(&mut Value)) {
            let ImageOperands {
                bias,
                lod,
                grad,
                const_offset,
                offset,
                sample,
                min_lod,
            } = o;
            for v in [bias, lod].into_iter().flatten() {
                f(v);
            }
            if let Some((dx, dy)) = grad {
                f(dx);
                f(dy);
            }
            for v in [const_offset, offset, sample, min_lod]
                .into_iter()
                .flatten()
            {
                f(v);
            }
        }
        match self {
            Inst::Load { ptr, .. } | Inst::ArrayLength { ptr, .. } => f(ptr),
            Inst::Store { ptr, value, .. } => {
                f(ptr);
                f(value);
            }
            Inst::Unary { a, .. }
            | Inst::Convert { a, .. }
            | Inst::Image(a)
            | Inst::Derivative { a, .. } => f(a),
            Inst::Binary { a, b, .. } => {
                f(a);
                f(b);
            }
            Inst::Select { cond, t, f: e } => {
                f(cond);
                f(t);
                f(e);
            }
            Inst::Fma { a, b, c } => {
                f(a);
                f(b);
                f(c);
            }
            Inst::BitFieldInsert {
                base,
                insert,
                offset,
                count,
            } => {
                for v in [base, insert, offset, count] {
                    f(v);
                }
            }
            Inst::BitFieldExtract {
                base,
                offset,
                count,
                ..
            } => {
                for v in [base, offset, count] {
                    f(v);
                }
            }
            Inst::CompositeConstruct { parts } => parts.iter_mut().for_each(f),
            Inst::CompositeExtract { composite, .. } => f(composite),
            Inst::CompositeInsert {
                object, composite, ..
            } => {
                f(object);
                f(composite);
            }
            Inst::VectorExtractDynamic { vector, index } => {
                f(vector);
                f(index);
            }
            Inst::VectorInsertDynamic {
                vector,
                component,
                index,
            } => {
                f(vector);
                f(component);
                f(index);
            }
            Inst::VectorShuffle { a, b, .. } => {
                f(a);
                f(b);
            }
            Inst::Undef
            | Inst::BeginInterlock
            | Inst::EndInterlock
            | Inst::DemoteToHelperInvocation
            | Inst::IsHelperInvocation
            | Inst::ControlBarrier { .. }
            | Inst::MemoryBarrier { .. }
            | Inst::EmitVertex
            | Inst::EndPrimitive => {}
            Inst::SetMeshOutputs {
                vertices,
                primitives,
            } => {
                f(vertices);
                f(primitives);
            }
            Inst::AccessChain { base, indices, .. } => {
                f(base);
                indices.iter_mut().for_each(f);
            }
            Inst::Call { args, .. } => args.iter_mut().for_each(f),
            Inst::AtomicLoad { ptr, .. } => f(ptr),
            Inst::AtomicStore { ptr, value, .. } | Inst::Atomic { ptr, value, .. } => {
                f(ptr);
                f(value);
            }
            Inst::AtomicCompareExchange {
                ptr,
                value,
                comparator,
                ..
            } => {
                f(ptr);
                f(value);
                f(comparator);
            }
            Inst::ImageTexelPointer {
                image,
                coord,
                sample,
            } => {
                f(image);
                f(coord);
                f(sample);
            }
            Inst::SampledImage { image, sampler } => {
                f(image);
                f(sampler);
            }
            Inst::ImageSample {
                sampled_image,
                coord,
                dref,
                ops: o,
                ..
            } => {
                f(sampled_image);
                f(coord);
                if let Some(d) = dref {
                    f(d);
                }
                ops(o, &mut f);
            }
            Inst::ImageFetch {
                image,
                coord,
                ops: o,
                ..
            }
            | Inst::ImageRead {
                image,
                coord,
                ops: o,
                ..
            } => {
                f(image);
                f(coord);
                ops(o, &mut f);
            }
            Inst::ImageSparseTexelsResident(code) => f(code),
            Inst::ImageGather {
                sampled_image,
                coord,
                ops: o,
                ..
            } => {
                f(sampled_image);
                f(coord);
                ops(o, &mut f);
            }
            Inst::ImageDrefGather {
                sampled_image,
                coord,
                dref,
                ops: o,
                ..
            } => {
                f(sampled_image);
                f(coord);
                f(dref);
                ops(o, &mut f);
            }
            Inst::ImageWrite {
                image,
                coord,
                texel,
                ops: o,
                ..
            } => {
                f(image);
                f(coord);
                f(texel);
                ops(o, &mut f);
            }
            Inst::ImageQuery { query, image } => {
                f(image);
                if let ImageQuery::SizeLod(lod) = query {
                    f(lod);
                }
            }
            Inst::ImageQueryLod {
                sampled_image,
                coord,
            } => {
                f(sampled_image);
                f(coord);
            }
            Inst::Group { op, .. } => op.for_each_value_mut(f),
            Inst::Glsl { args, .. } => args.iter_mut().for_each(f),
            Inst::RayQuery(op) => op.for_each_value_mut(f),
        }
    }

    /// Call `f` on every constant operand carried as a [`ConstId`] rather than an SSA value: the
    /// scopes and memory semantics of atomics, barriers, subgroup operations, and Vulkan memory
    /// model accesses, a gather's component, a cluster size, a quad direction, and a ray query's
    /// intersection. [`Inst::for_each_value`] visits none of them.
    pub fn for_each_const(&self, mut f: impl FnMut(ConstId)) {
        fn access(a: &MemAccess, f: &mut dyn FnMut(ConstId)) {
            for s in [a.make_available, a.make_visible].into_iter().flatten() {
                f(s.id);
            }
        }
        match self {
            Inst::Load { access: a, .. } | Inst::Store { access: a, .. } => access(a, &mut f),
            Inst::ImageRead { mem, .. } | Inst::ImageWrite { mem, .. } => {
                if let Some(s) = mem.made_scope() {
                    f(s.id);
                }
            }
            Inst::AtomicLoad { scope, sem, .. }
            | Inst::AtomicStore { scope, sem, .. }
            | Inst::Atomic { scope, sem, .. }
            | Inst::MemoryBarrier {
                mem: scope, sem, ..
            } => {
                f(scope.id);
                f(sem.id);
            }
            Inst::AtomicCompareExchange {
                scope,
                equal,
                unequal,
                ..
            } => {
                f(scope.id);
                f(equal.id);
                f(unequal.id);
            }
            Inst::ControlBarrier { exec, mem, sem } => {
                f(exec.id);
                f(mem.id);
                f(sem.id);
            }
            Inst::ImageGather { component_id, .. } => f(*component_id),
            Inst::Group { scope, op } => {
                f(scope.id);
                op.for_each_const(f);
            }
            Inst::RayQuery(op) => op.for_each_const(f),
            Inst::Unary { .. }
            | Inst::Binary { .. }
            | Inst::Select { .. }
            | Inst::BitFieldInsert { .. }
            | Inst::BitFieldExtract { .. }
            | Inst::Fma { .. }
            | Inst::Convert { .. }
            | Inst::CompositeConstruct { .. }
            | Inst::CompositeExtract { .. }
            | Inst::CompositeInsert { .. }
            | Inst::VectorExtractDynamic { .. }
            | Inst::VectorInsertDynamic { .. }
            | Inst::VectorShuffle { .. }
            | Inst::Undef
            | Inst::AccessChain { .. }
            | Inst::ArrayLength { .. }
            | Inst::Call { .. }
            | Inst::ImageTexelPointer { .. }
            | Inst::SampledImage { .. }
            | Inst::Image(_)
            | Inst::ImageSample { .. }
            | Inst::ImageFetch { .. }
            | Inst::ImageDrefGather { .. }
            | Inst::ImageSparseTexelsResident(_)
            | Inst::ImageQuery { .. }
            | Inst::ImageQueryLod { .. }
            | Inst::Derivative { .. }
            | Inst::Glsl { .. }
            | Inst::BeginInterlock
            | Inst::EndInterlock
            | Inst::DemoteToHelperInvocation
            | Inst::IsHelperInvocation
            | Inst::SetMeshOutputs { .. }
            | Inst::EmitVertex
            | Inst::EndPrimitive => {}
        }
    }

    /// What the instruction needs from the entry point's execution model, if anything.
    pub fn model_need(&self) -> Option<ModelNeed> {
        match self {
            Inst::ImageSample { form, .. } if form.implicit => Some(ModelNeed::ImplicitDerivatives),
            Inst::ImageQueryLod { .. } | Inst::Derivative { .. } => {
                Some(ModelNeed::ImplicitDerivatives)
            }
            Inst::DemoteToHelperInvocation | Inst::IsHelperInvocation => Some(ModelNeed::Fragment),
            Inst::Glsl {
                op:
                    GlslOp::InterpolateAtCentroid
                    | GlslOp::InterpolateAtSample
                    | GlslOp::InterpolateAtOffset,
                ..
            } => Some(ModelNeed::Fragment),
            Inst::BeginInterlock | Inst::EndInterlock => Some(ModelNeed::Interlock),
            Inst::SetMeshOutputs { .. } => Some(ModelNeed::Mesh),
            Inst::EmitVertex | Inst::EndPrimitive => Some(ModelNeed::Geometry),
            Inst::AtomicLoad { scope, .. }
            | Inst::AtomicStore { scope, .. }
            | Inst::Atomic { scope, .. }
            | Inst::AtomicCompareExchange { scope, .. }
            | Inst::MemoryBarrier { mem: scope, .. }
            | Inst::Group { scope, .. }
                if scope.scope == Scope::Workgroup =>
            {
                Some(ModelNeed::WorkgroupScope)
            }
            Inst::ControlBarrier { exec, mem, .. }
                if exec.scope == Scope::Workgroup || mem.scope == Scope::Workgroup =>
            {
                Some(ModelNeed::WorkgroupScope)
            }
            _ => None,
        }
    }
}

/// An instruction with its result, if it has one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstNode {
    pub result: Option<Value>,
    pub inst: Inst,
}

// Every instruction is as large as the largest variant of [`Inst`], so a field added to one form,
// or to an operand struct several forms hold, grows every instruction of every module and the
// translation's peak memory with it (four texel fields in `ImageOperands` made it 104 bytes, and
// the ledger peak of about 9,600 corpus rows grew). The size is pinned here; growing it is a
// decision to record, not a side effect. (Pointer-width bound: the boxed slices of `Inst::Call` and
// others make it 8-byte aligned.)
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(std::mem::size_of::<Inst>() == 88);
    assert!(std::mem::size_of::<InstNode>() == 96);
    assert!(std::mem::size_of::<TexelMemory>() == 8);
};

/// An `OpPhi`. Only builder-closed `if_`/`switch_` merges create phis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Phi {
    pub result: Value,
    pub incoming: Vec<(Value, BlockId)>,
}

/// A structured merge declaration: a property of its header block, not an instruction.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Merge {
    /// `OpSelectionMerge`.
    Selection { merge: BlockId },
    /// `OpLoopMerge`. Only `FunctionBuilder::loop_` creates one, and it records a [`LoopMeta`].
    Loop { merge: BlockId, cont: BlockId },
}

/// A block terminator.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Terminator {
    Branch(BlockId),
    CondBranch {
        cond: Value,
        t: BlockId,
        f: BlockId,
    },
    Switch {
        sel: Value,
        default: BlockId,
        cases: Vec<(u64, BlockId)>,
    },
    Return,
    ReturnValue(Value),
    Kill,
    TerminateInvocation,
    Unreachable,
    /// `OpEmitMeshTasksEXT`: an object entry launches `x * y * z` mesh workgroups, which read
    /// `payload`, its `TaskPayloadWorkgroupEXT` variable. Only `FunctionBuilder:: emit_mesh_tasks`
    /// creates it, in a task entry.
    EmitMeshTasks {
        x: Value,
        y: Value,
        z: Value,
        payload: Option<Value>,
    },
}

impl Terminator {
    /// Call `f` on every SSA value operand.
    pub fn for_each_value(&self, mut f: impl FnMut(Value)) {
        match self {
            Terminator::CondBranch { cond, .. } => f(*cond),
            Terminator::Switch { sel, .. } => f(*sel),
            Terminator::ReturnValue(v) => f(*v),
            Terminator::EmitMeshTasks { x, y, z, payload } => {
                for v in [*x, *y, *z].into_iter().chain(*payload) {
                    f(v);
                }
            }
            Terminator::Branch(_)
            | Terminator::Return
            | Terminator::Kill
            | Terminator::TerminateInvocation
            | Terminator::Unreachable => {}
        }
    }
}

impl Terminator {
    /// [`Terminator::for_each_value`] with each operand open to change, in the same order.
    pub fn for_each_value_mut(&mut self, mut f: impl FnMut(&mut Value)) {
        match self {
            Terminator::CondBranch { cond, .. } => f(cond),
            Terminator::Switch { sel, .. } => f(sel),
            Terminator::ReturnValue(v) => f(v),
            Terminator::EmitMeshTasks { x, y, z, payload } => {
                for v in [x, y, z].into_iter().chain(payload.as_mut()) {
                    f(v);
                }
            }
            Terminator::Branch(_)
            | Terminator::Return
            | Terminator::Kill
            | Terminator::TerminateInvocation
            | Terminator::Unreachable => {}
        }
    }

    /// Call `f` on every block the terminator may branch to, in operand order.
    pub fn for_each_target_mut(&mut self, mut f: impl FnMut(&mut BlockId)) {
        match self {
            Terminator::Branch(t) => f(t),
            Terminator::CondBranch { t, f: e, .. } => {
                f(t);
                f(e);
            }
            Terminator::Switch { default, cases, .. } => {
                f(default);
                for (_, c) in cases {
                    f(c);
                }
            }
            Terminator::Return
            | Terminator::ReturnValue(_)
            | Terminator::Kill
            | Terminator::TerminateInvocation
            | Terminator::Unreachable
            | Terminator::EmitMeshTasks { .. } => {}
        }
    }
}

/// A basic block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub func: FuncId,
    pub phis: Vec<Phi>,
    pub insts: Vec<InstNode>,
    pub merge: Option<Merge>,
    pub term: Terminator,
}

/// A function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function {
    pub ret: TypeId,
    /// The interned `Function { ret, params }` type.
    pub fn_ty: TypeId,
    pub params: Vec<Value>,
    /// Blocks in layout order; the first is the entry block.
    pub blocks: Vec<BlockId>,
    /// `Function`-storage variables, declared at the top of the entry block.
    pub vars: Vec<VarId>,
    /// Whether the function is a routine its callers share (`ModuleBuilder::routine`), which
    /// `serialize::serialize_for` declares `DontInline` for a profile whose driver converts every
    /// function to one inlined at each call.
    pub routine: bool,
    /// Whether the routine is a texel routine (`ModuleBuilder::texel_routine`): a linear filter
    /// emulation's sum, which `serialize` declares `DontInline` for every profile, so a driver
    /// that honours the control compiles it once and calls it from each site.
    pub kept: bool,
    /// Whether the routine is a sampling routine (`ModuleBuilder::sampling_routine`): one
    /// combined binding's sample emulation, which `serialize::serialize_for` declares
    /// `DontInline` for a profile whose driver keeps such a call a call
    /// (`DriverWorkaround::DontInlineKeepsCalls`) or inlines every function at each call.
    pub sampling: bool,
}

/// Metadata of every loop, recorded by `FunctionBuilder::loop_`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopMeta {
    pub header: BlockId,
    pub merge: BlockId,
    pub cont: BlockId,
    pub preheader: BlockId,
    pub exit_sel: Option<VarId>,
    /// The loop's `OpLoopMerge` asks `DontUnroll`: a loop `spirv::roll` wrote to compile a
    /// repeated region once, which a downstream compiler that unrolled it would write out again.
    pub dont_unroll: bool,
}

/// The execution models the new core emits.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExecModel {
    Vertex,
    Fragment,
    GLCompute,
    /// A post-tessellation vertex function.
    TessellationEvaluation,
    /// The tessellation control stage link synthesizes for a tessellated pipeline.
    TessellationControl,
    /// An object function (`SPV_EXT_mesh_shader`).
    TaskEXT,
    /// A mesh function (`SPV_EXT_mesh_shader`).
    MeshEXT,
    /// The pass-through geometry stage link synthesizes for a fragment function that reads
    /// `[[barycentric_coord]]` on a device without `fragmentShaderBarycentric`.
    Geometry,
}

impl ExecModel {
    /// Whether the model is a tessellation stage (`Tessellation` capability).
    pub fn is_tessellation(self) -> bool {
        matches!(
            self,
            ExecModel::TessellationEvaluation | ExecModel::TessellationControl
        )
    }
}

/// An execution mode with typed operands.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExecMode {
    /// Required on every fragment entry; `entry` adds it when absent.
    OriginUpperLeft,
    EarlyFragmentTests,
    DepthReplacing,
    DepthGreater,
    DepthLess,
    StencilRefReplacing,
    PixelInterlockOrdered,
    SampleInterlockOrdered,
    LocalSize {
        x: u32,
        y: u32,
        z: u32,
    },
    DerivativeGroupLinear,
    SignedZeroInfNanPreserve {
        width: u8,
    },
    /// `DenormPreserve` (`SPV_KHR_float_controls`): denormal results and operands of this float
    /// width are kept.
    DenormPreserve {
        width: u8,
    },
    /// `RoundingModeRTE` (`SPV_KHR_float_controls`): float operations and conversions of this
    /// width round to nearest even.
    RoundingModeRte {
        width: u8,
    },
    SubgroupUniformControlFlow,
    MaximallyReconverges,
    /// `RequireFullQuadsKHR` (`SPV_KHR_quad_control`, fragment only): every quad starts with its
    /// four invocations, helpers included, and they take part in every group operation.
    RequireFullQuads,
    /// `PostDepthCoverage` (`SPV_KHR_post_depth_coverage`, fragment only): `SampleMask` reads the
    /// coverage after the early depth and stencil tests.
    PostDepthCoverage,
    // ---- Tessellation ----
    /// The triangle patch domain.
    Triangles,
    /// The quad patch domain.
    Quads,
    SpacingEqual,
    SpacingFractionalEven,
    SpacingFractionalOdd,
    VertexOrderCw,
    VertexOrderCcw,
    /// The tessellation control stage's output patch size (1..=32 on every device with
    /// `tessellationShader`), or the vertices a mesh entry writes at most.
    OutputVertices {
        n: u32,
    },
    // ---- Mesh ----
    /// The primitives a mesh entry writes at most (`OutputPrimitivesEXT`).
    OutputPrimitives {
        n: u32,
    },
    /// A mesh entry's primitive kind: points (`OutputPoints`).
    OutputPoints,
    /// Lines (`OutputLinesEXT`).
    OutputLines,
    /// Triangles (`OutputTrianglesEXT`).
    OutputTriangles,
    // ---- Geometry ----
    /// A geometry entry's input primitive: points (`InputPoints`). Its triangles are
    /// [`ExecMode::Triangles`] and its output points [`ExecMode::OutputPoints`].
    InputPoints,
    /// Lines (`InputLines`).
    InputLines,
    /// A geometry entry's output primitive: line strips (`OutputLineStrip`).
    OutputLineStrip,
    /// Triangle strips (`OutputTriangleStrip`).
    OutputTriangleStrip,
    /// A geometry entry's invocations per input primitive (`Invocations`).
    Invocations {
        n: u32,
    },
}

impl ExecMode {
    /// Whether this mode declares fragment shader interlock.
    pub fn is_interlock(self) -> bool {
        matches!(
            self,
            ExecMode::PixelInterlockOrdered | ExecMode::SampleInterlockOrdered
        )
    }
}

/// The module's single entry point (per-function compilation).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryPoint {
    pub model: ExecModel,
    pub func: FuncId,
    pub modes: Vec<ExecMode>,
}

/// A module-scope declaration, in creation order. Every declaration refers only to earlier ones,
/// so this is a valid SPIR-V declaration order.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decl {
    Type(TypeId),
    Const(ConstId),
    Global(GlobalId),
}

/// A finalized typed SPIR-V 1.5 module. Built only by `ModuleBuilder::finalize`.
///
/// `finalize` derives the entry interface and the requirements. The loop table is filled by
/// `FunctionBuilder::loop_` (its barrier facts by `finalize`).
#[derive(Clone, Debug)]
pub struct TypedModule {
    pub(in crate::spirv) types: Types,
    pub(in crate::spirv) consts: Consts,
    /// The SSA value of each constant, indexed by [`ConstId`].
    pub(in crate::spirv) const_values: Vec<Value>,
    pub(in crate::spirv) values: Vec<ValueDef>,
    pub(in crate::spirv) globals: Vec<GlobalVar>,
    pub(in crate::spirv) push: Vec<PushMember>,
    pub(in crate::spirv) functions: Vec<Function>,
    pub(in crate::spirv) blocks: Vec<Block>,
    pub(in crate::spirv) vars: Vec<LocalVar>,
    pub(in crate::spirv) loops: Vec<LoopMeta>,
    /// The header of every switch-at-use (`FunctionBuilder::switch_at_use`), in creation order.
    pub(in crate::spirv) switches_at_use: Vec<BlockId>,
    pub(in crate::spirv) entry: EntryPoint,
    pub(in crate::spirv) decls: Vec<Decl>,
    /// The entry point's interface: every module-scope variable the entry's static call tree
    /// references, in declaration order (the SPIR-V 1.4+ rule; derived by `finalize`, never
    /// listed by hand).
    pub(in crate::spirv) interface: Vec<GlobalId>,
    /// Everything the module needs from the device, recorded by `finalize` from what the module
    /// contains and what its subgroup requirement needs.
    pub(in crate::spirv) requirements: Requirements,
    /// The subgroup requirement the plan chose (`ModuleBuilder::subgroup`): the only source of the
    /// requirements' plan facts.
    pub(in crate::spirv) subgroup: SubgroupRequirement,
    /// Builtin outputs decorated `Invariant` (`ModuleBuilder::invariant`).
    pub(in crate::spirv) invariant: std::collections::BTreeSet<GlobalId>,
    /// User varyings decorated `PerPrimitiveEXT` (`ModuleBuilder::per_primitive`): a mesh entry's
    /// primitive outputs, and the fragment inputs that read them.
    pub(in crate::spirv) per_primitive: std::collections::BTreeSet<GlobalId>,
    /// The `InterpolateAtOffset` offsets are Metal's moved by 7/16
    /// (`ModuleBuilder::offsets_from_seven_sixteenths`).
    pub(in crate::spirv) offsets_from_seven_sixteenths: bool,
    /// Resources decorated `Coherent`: every storage buffer and storage image resource of a
    /// module built with `ModuleBuilder::coherent_resources`.
    pub(in crate::spirv) coherent: std::collections::BTreeSet<GlobalId>,
    /// Storage-image resources decorated `Volatile` (`ModuleBuilder::volatile`).
    pub(in crate::spirv) volatile: std::collections::BTreeSet<GlobalId>,
    /// The enabling capability chosen for a builtin whose registry offers alternatives
    /// (`ModuleBuilder::enable_builtin`).
    pub(in crate::spirv) enabling: std::collections::BTreeMap<GlobalId, crate::spv::Capability>,
    /// The values decorated `NonUniform`: every result that carries a non-uniform descriptor
    /// element ([`Inst::carries_non_uniform`]), recorded as the builder appends it.
    pub(in crate::spirv) non_uniform: std::collections::BTreeSet<Value>,
    /// The narrowings to 16 bits of the conversion pairs that round a `half` in a form measured
    /// to answer Metal's where the driver takes `DriverDeviation::MetalHalfConversion`
    /// (`ModuleBuilder::measured_roundings`), recorded as the builder emits them.
    pub(in crate::spirv) measured_roundings: std::collections::BTreeSet<Value>,
    /// The texels the texture family's writes store, whose stored NaN the write's own notes
    /// account for (`ModuleBuilder::stored_texels`), recorded as the builder emits them.
    pub(in crate::spirv) stored_texels: std::collections::BTreeSet<Value>,
    /// The `Function` variables the emitter holds workgroup uniform by construction: the `cont`
    /// flag of a predicated loop that iterates together, which the lowering's uniformity analysis
    /// proved. An in-memory record for checks (`spirv::testing::barriers`), kept only where they
    /// build; it changes no emitted word.
    #[cfg(test)]
    pub(in crate::spirv) uniform_vars: std::collections::BTreeSet<VarId>,
}

impl TypedModule {
    /// The `Function` variables the emitter holds workgroup uniform by construction.
    #[cfg(test)]
    pub fn uniform_vars(&self) -> &std::collections::BTreeSet<VarId> {
        &self.uniform_vars
    }

    /// The type table.
    pub fn types(&self) -> &Types {
        &self.types
    }

    /// The constant table.
    pub fn consts(&self) -> &Consts {
        &self.consts
    }

    /// The narrowings to 16 bits of the conversion pairs that round a `half` in a form measured to
    /// answer Metal's (`ModuleBuilder::note_measured_rounding`).
    pub fn measured_roundings(&self) -> &std::collections::BTreeSet<Value> {
        &self.measured_roundings
    }

    /// The texels the texture family's writes store (`ModuleBuilder::note_stored_texel`).
    pub fn stored_texels(&self) -> &std::collections::BTreeSet<Value> {
        &self.stored_texels
    }

    /// The SSA value of a constant.
    pub fn const_value(&self, c: ConstId) -> BResult<Value> {
        self.const_values
            .get(c.index())
            .copied()
            .ok_or(BuildError::Dangling(Dangling::Const(c)))
    }

    /// The definition of a value.
    pub fn value(&self, v: Value) -> BResult<&ValueDef> {
        // A match, not `ok_or`: unoptimized (dev and test) builds would build and drop the
        // error at every lookup, and this is among the hottest (the interpreter's every operand).
        match self.values.get(v.index()) {
            Some(def) => Ok(def),
            None => Err(BuildError::Dangling(Dangling::Value(v))),
        }
    }

    /// Every value definition, indexed by [`Value`].
    pub fn values(&self) -> &[ValueDef] {
        &self.values
    }

    /// Every module-scope variable, indexed by [`GlobalId`].
    pub fn globals(&self) -> &[GlobalVar] {
        &self.globals
    }

    /// The push-constant block's members, in member order.
    pub fn push_members(&self) -> &[PushMember] {
        &self.push
    }

    /// Every function, indexed by [`FuncId`].
    pub fn functions(&self) -> &[Function] {
        &self.functions
    }

    /// Every block, indexed by [`BlockId`].
    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    /// Every `Function`-storage variable, indexed by [`VarId`].
    pub fn vars(&self) -> &[LocalVar] {
        &self.vars
    }

    /// Every loop's metadata.
    #[cfg(test)]
    pub fn loops(&self) -> &[LoopMeta] {
        &self.loops
    }

    /// The header of every switch-at-use, in creation order. No tangled instruction sits inside
    /// one, directly or through a call; `finalize` checks it.
    #[cfg(test)]
    pub fn switches_at_use(&self) -> &[BlockId] {
        &self.switches_at_use
    }

    /// The entry point.
    pub fn entry(&self) -> &EntryPoint {
        &self.entry
    }

    /// Module-scope declarations in declaration order.
    pub fn decls(&self) -> &[Decl] {
        &self.decls
    }

    /// The entry point's interface variables, in declaration order.
    #[cfg(test)]
    pub fn interface(&self) -> &[GlobalId] {
        &self.interface
    }

    /// The module's derived device requirements.
    pub fn requirements(&self) -> &Requirements {
        &self.requirements
    }
}
