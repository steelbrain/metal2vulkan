//! `ModuleBuilder`: module-scope declarations and `finalize`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::device::subgroup::SubgroupRequirement;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, SpecKey, VaryingKey};
use crate::spirv::builder::constructs;
use crate::spirv::builder::function::FunctionBuilder;
use crate::spirv::dead::Offer;
use crate::spirv::error::{
    BResult, BuildError, ControlRule, Dangling, EntryRule, GlobalRule, LayoutViolation, TypeRule,
};
use crate::spirv::ids::{BlockId, ConstId, FuncId, GlobalId, TypeId, Value, VarId};
use crate::spirv::ir::{
    Block, ConstBits, Constant, Consts, Decl, Dim, EntryPoint, ExecMode, ExecModel, Function,
    GlobalKind, GlobalVar, ImageType, ImageUsage, InstNode, IoDecor, IoDir, LayoutRule, LocalVar,
    LoopMeta, MemberLayout, Merge, Phi, PushMember, Shape, SpecOp, StorageClass, StructLayout,
    Terminator, Type, TypedModule, Types, ValueDef, ValueOrigin,
};
use crate::spirv::requirements;

/// How a function was started: a plain function, a routine, a texel routine, or a sampling routine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Started {
    Plain,
    Routine,
    Texel,
    Sampling,
}

/// A function under construction.
#[derive(Clone, Debug)]
pub(super) struct FunctionDraft {
    pub(super) ret: TypeId,
    pub(super) params: Vec<Value>,
    pub(super) param_tys: Vec<TypeId>,
    pub(super) blocks: Vec<BlockId>,
    pub(super) vars: Vec<VarId>,
    /// Set by `FunctionBuilder::finish`.
    pub(super) fn_ty: Option<TypeId>,
    /// Started by [`ModuleBuilder::routine`].
    pub(super) routine: bool,
    /// Started by [`ModuleBuilder::texel_routine`].
    pub(super) kept: bool,
    /// Started by [`ModuleBuilder::sampling_routine`].
    pub(super) sampling: bool,
}

/// A block under construction: `term` is `None` until the block is terminated.
#[derive(Clone, Debug)]
pub(super) struct BlockDraft {
    pub(super) func: FuncId,
    pub(super) phis: Vec<Phi>,
    pub(super) insts: Vec<InstNode>,
    pub(super) merge: Option<Merge>,
    pub(super) term: Option<Terminator>,
}

/// Builds one SPIR-V 1.5 module through typed constructors. Every method checks its typing rules
/// before recording anything, so a rejected call leaves the builder unchanged.
#[derive(Debug, Default)]
pub struct ModuleBuilder {
    types: Types,
    consts: Consts,
    const_values: Vec<Value>,
    pub(super) values: Vec<ValueDef>,
    globals: Vec<GlobalVar>,
    resources: HashMap<ResourceSlot, GlobalId>,
    varyings: HashMap<(VaryingKey, IoDir), GlobalId>,
    builtins: HashMap<(crate::spv::BuiltIn, IoDir), GlobalId>,
    push: Vec<PushMember>,
    push_global: Option<GlobalId>,
    pub(super) functions: Vec<FunctionDraft>,
    pub(super) blocks: Vec<BlockDraft>,
    pub(super) vars: Vec<LocalVar>,
    pub(super) loops: Vec<LoopMeta>,
    /// The header of every switch-at-use, in creation order.
    pub(super) switches_at_use: Vec<BlockId>,
    entry: Option<EntryPoint>,
    decls: Vec<Decl>,
    /// Builtin outputs decorated `Invariant`.
    invariant: BTreeSet<GlobalId>,
    /// User varyings decorated `PerPrimitiveEXT` (`ModuleBuilder::per_primitive`).
    per_primitive: BTreeSet<GlobalId>,
    /// The module's `InterpolateAtOffset` offsets are Metal's moved by 7/16
    /// (`ModuleBuilder::offsets_from_seven_sixteenths`).
    offsets_from_seven_sixteenths: bool,
    /// Every storage buffer and storage image resource is decorated `Coherent`
    /// (`ModuleBuilder::coherent_resources`).
    coherent: bool,
    /// Storage-image resources decorated `Coherent` one by one (`ModuleBuilder::coherent`).
    coherent_images: BTreeSet<GlobalId>,
    /// Storage-image resources decorated `Volatile` (`ModuleBuilder::volatile`).
    volatile: BTreeSet<GlobalId>,
    /// The enabling capability lowering chose for a builtin whose registry offers alternatives
    /// (`requirements::builtin_enablers`).
    enabling: BTreeMap<GlobalId, crate::spv::Capability>,
    /// The subgroup requirement the plan chose (`ModuleBuilder::subgroup`); none when the lowering
    /// states none.
    subgroup: Option<SubgroupRequirement>,
    /// Every result that carries a non-uniform descriptor element (`Inst::carries_non_uniform`):
    /// decorated `NonUniform`, and refused where it would reach an undecorated operand.
    pub(super) non_uniform: BTreeSet<Value>,
    /// The narrowings to 16 bits of the conversion pairs that round a `half` in a form measured
    /// to answer Metal's on the devices whose driver takes `DriverDeviation::MetalHalfConversion`
    /// (`ModuleBuilder::note_measured_rounding`). Nothing emitted depends on it: the compile
    /// reads it to decide `PrecisionNote::NativeHalfConversion`.
    pub(super) measured_roundings: BTreeSet<Value>,
    /// The `Function` variables the emitter holds workgroup uniform by construction
    /// (`FunctionBuilder::note_uniform_var`). Nothing emitted depends on it: a record for the
    /// checks of `spirv::testing::barriers`, kept only where they build.
    #[cfg(test)]
    pub(super) uniform_vars: BTreeSet<VarId>,
    /// Heap bytes inside the blocks (their instruction and phi storage, and the instructions'
    /// boxed operand lists), counted as they grow so [`ModuleBuilder::footprint`] need not walk the
    /// blocks.
    pub(super) block_bytes: usize,
    /// The bytes of every switch split's working tables: each literal with its arm's chunk, each
    /// arm's chunk, the groups, each group's literals by chunk, and the cases regrouped by chunk. A
    /// split frees them when it returns, so [`ModuleBuilder::footprint`] keeps them for the rest of
    /// the module, an upper bound of their peak that the emitters' polls charge like the blocks'
    /// growth.
    pub(super) split_bytes: usize,
    /// The narrow format bits the lowering noted for values it holds widened, by format and value:
    /// `FunctionBuilder::note_bits` and `bits_of`. Nothing here changes the IR.
    pub(super) bits: HashMap<(NarrowFmt, Value), NotedBits>,
    /// The vector and lane each lane extract the lowering noted reads (`FunctionBuilder::note_lane`).
    pub(super) lanes: HashMap<Value, (Value, u32)>,
    /// The vector, lane value, and lane index of each narrow lane insert the lowering noted
    /// (`FunctionBuilder::note_insert`).
    pub(super) inserts: HashMap<Value, (Value, Value, u32)>,
    /// The 16-bit integer each zero extension to 32 bits the lowering noted widens
    /// (`FunctionBuilder::note_widened16`).
    pub(super) widened16: HashMap<Value, Value>,
    /// The 16-bit float (scalar or vector) each `OpFConvert` to 32 bits widens, by its result
    /// (`FunctionBuilder::convert`): a conversion of the result back to 16 bits is that float,
    /// exactly. A widening of a conversion from 32 bits is not listed: that pair is a rounding to
    /// `half` (`lower::arith::narrow::rounds_by_conversion`), kept whole.
    pub(super) widened_halves: HashMap<Value, Value>,
    /// The result of every `OpFConvert` from 32 to 16 bits (`FunctionBuilder::convert`).
    pub(super) narrowed_floats: HashSet<Value>,
    /// Every `OpUndef` result (`FunctionBuilder::is_undef`).
    pub(super) undefs: HashSet<Value>,
    /// The dynamic texel offsets the lowering asked to be taken as constants, by offset value
    /// (`FunctionBuilder::arm_offset`).
    pub(super) offset_arms: HashMap<Value, crate::spirv::builder::ops::image::ArmedOffset>,
    /// The equal values the lowering offered (`FunctionBuilder::offer`), in offer order.
    pub(super) offers: Vec<Offer>,
    /// The offer groups handed out (`FunctionBuilder::offer_group`).
    pub(super) offer_groups: u32,
    /// The encode of a narrow vector computed for its lanes' stores in a block, with its offer
    /// group (`FunctionBuilder::note_encode`).
    pub(super) encodes: HashMap<(NarrowFmt, Value), (Value, BlockId, u32)>,
    /// How a float multiply the device must round on its own is emitted
    /// ([`ModuleBuilder::product_form`]).
    pub(super) product_form: ProductForm,
    /// How the lowering clears the bits above a narrow integer's width
    /// ([`ModuleBuilder::mask_form`]).
    mask_form: MaskForm,
    /// How the lowering rounds an `f32` to `half` ([`ModuleBuilder::quantize_form`]).
    quantize_form: QuantizeForm,
    /// Whether that rounding is the device's own rather than one measured to be Metal's
    /// ([`ModuleBuilder::native_half`]).
    native_half: bool,
    /// Whether a `half`'s NaN bits are the device's conversion's under the float controls
    /// ([`ModuleBuilder::native_half_nans`]).
    native_half_nans: bool,
}

/// How a float multiply the device must round on its own and never fuse is emitted
/// ([`FunctionBuilder::binary_no_contraction`]). Lowering takes it from the profile's `float`
/// functions fact (`ArithPlan::product`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ProductForm {
    /// `OpFMul` decorated `NoContraction`, for a device whose driver compiles the SPIR-V itself:
    /// the device rounds the product once, keeps a `-0` product, and folds a zero factor as
    /// Metal does (`x · 0` to `0` where the multiply carries `nsz`). It has no addend a device
    /// could answer for a zero product.
    #[default]
    Multiply,
    /// `GLSL.std.450 Fma(a, b, -0.0)` decorated `NoContraction`, for a device whose SPIR-V reaches
    /// Metal's compiler through SPIRV-Cross (MoltenVK, the profile whose `float` functions are
    /// Metal's): SPIRV-Cross renders a `NoContraction` `OpFMul` as `fma(l, r, 0)`, which answers
    /// `+0` for a `-0` product, and this form as `fma(a, b, -0.0)`, which Metal's compiler treats
    /// as Metal treats its own multiply. `a · b + (-0.0)` is `a · b` exactly whether the device
    /// fuses it or not.
    NegativeZeroFma,
}

/// How the lowering clears the bits above a narrow integer's width `N` in a 32-bit container
/// (`lower::arith::ops::mask`). Lowering takes it from the profile's driver workarounds
/// (`ArithPlan::mask`). Both forms are the same bits on every device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MaskForm {
    /// `OpBitwiseAnd` with `2^N - 1`.
    #[default]
    And,
    /// `OpBitFieldUExtract(x, 0, N)`, for a driver whose compiler drops the mask after a signed
    /// clamp to exactly the narrow range (`DriverWorkaround::NarrowMaskAfterSignedClamp`).
    Extract,
}

/// How the lowering rounds an `f32` to `half` (`lower::arith::narrow`): the device's own
/// rounding. Lowering takes it from the profile's `shaderFloat16` and float-controls properties
/// (`ArithPlan::quantize`). `Controlled` gives Metal's conversion to `half` (to nearest even on the
/// half grid, half denormals kept, infinity from 65520 up) by the Vulkan spec; `Device` and
/// `Converted` are within `PrecisionNote::NativeHalfQuantize`'s or `NativeHalfConversion`'s bound.
/// It also chooses how a `half` is widened from its bits (`narrow::from_bits`): `OpFConvert` of
/// the bits read as a 16-bit float in `Converted` and `Controlled`, `UnpackHalf2x16` in `Device`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum QuantizeForm {
    /// The device's `OpQuantizeToF16` alone, on a profile without `shaderFloat16`: a result
    /// below the smallest normal half may be a zero.
    #[default]
    Device,
    /// No quantize: `OpFConvert` to a 16-bit float and back, on a profile with `shaderFloat16`
    /// whose float controls do not make the conversion Metal's.
    Converted,
    /// No quantize: `OpFConvert` to a 16-bit float and back, each decorated `NoContraction`, in
    /// an entry that declares `RoundingModeRTE`, `DenormPreserve` and `SignedZeroInfNanPreserve`
    /// for 16 bits (and the modes the device's independence ties to them), where the profile has
    /// `shaderFloat16` and those float controls: Metal's `half(float)` by the Vulkan spec, which
    /// makes such a conversion correctly rounded to nearest even with denormals, signed zeros,
    /// infinities and NaNs kept, and forbids removing it (`narrow::controlled_modes`).
    Controlled,
}

/// A 16-bit float format the lowering holds widened in `f32`, whose bits it may note.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NarrowFmt {
    Half,
    BFloat,
}

/// Noted format bits of a value: each lane's low 16 bits are the format bits of that lane of
/// the value (the bits above them are zero but in the lanes `dirty` has a bit for), valid
/// wherever the value is when `at` is `None`, else only in block `at`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NotedBits {
    pub(super) bits: Value,
    pub(super) dirty: u32,
    pub(super) at: Option<BlockId>,
}

impl ModuleBuilder {
    /// An empty module.
    pub fn new() -> ModuleBuilder {
        ModuleBuilder::default()
    }

    /// Emit every float multiply the device must round on its own in `form`. Emission sets it
    /// before any instruction, from the profile's `float` functions fact; a builder that never sets
    /// it emits [`ProductForm::Multiply`].
    pub fn product_form(&mut self, form: ProductForm) {
        self.product_form = form;
    }

    /// Clear the bits above a narrow integer's width in `form`. Emission sets it before any
    /// instruction, from the profile's driver workarounds; a builder that never sets it emits
    /// [`MaskForm::And`].
    pub fn set_mask_form(&mut self, form: MaskForm) {
        self.mask_form = form;
    }

    /// How the lowering clears the bits above a narrow integer's width
    /// ([`ModuleBuilder::set_mask_form`]).
    pub fn mask_form(&self) -> MaskForm {
        self.mask_form
    }

    /// Round an `f32` to `half` in `form`. Emission sets it before any instruction, from the
    /// profile; a builder that never sets it emits [`QuantizeForm::Device`].
    pub fn set_quantize_form(&mut self, form: QuantizeForm) {
        self.quantize_form = form;
    }

    /// How the lowering rounds an `f32` to `half` ([`ModuleBuilder::set_quantize_form`]).
    pub fn quantize_form(&self) -> QuantizeForm {
        self.quantize_form
    }

    /// Whether the [`ModuleBuilder::quantize_form`] is the device's own rounding, native half
    /// rounding, rather than one controlled to be Metal's. Emission sets it with the
    /// form; a builder that never sets it takes its form as Metal's.
    pub fn set_native_half(&mut self, native: bool) {
        self.native_half = native;
    }

    /// Record `narrowed`, the narrowing to 16 bits of a conversion pair that rounds a `half` in a
    /// form measured to answer Metal's where the driver takes
    /// `DriverDeviation::MetalHalfConversion` (`narrow::round_measured`).
    pub fn note_measured_rounding(&mut self, narrowed: Value) {
        self.measured_roundings.insert(narrowed);
    }

    /// Whether the module's `half` rounding is the device's own
    /// ([`ModuleBuilder::set_native_half`]).
    pub fn native_half(&self) -> bool {
        self.native_half
    }

    /// Whether a `half`'s NaN bits are left to the device's conversion where the module rounds
    /// through it under the float controls ([`QuantizeForm::Controlled`]): a `half` texel takes the
    /// rounding's NaN. Emission sets it with the form; a builder that never sets it keeps a
    /// texel's NaN.
    pub fn set_native_half_nans(&mut self, native: bool) {
        self.native_half_nans = native;
    }

    /// Whether a `half`'s NaN bits are the device's conversion's
    /// ([`ModuleBuilder::set_native_half_nans`]).
    pub fn native_half_nans(&self) -> bool {
        self.native_half_nans
    }

    /// The heap bytes the builder holds, for the byte ledger: every input-scaled container's
    /// capacity, with the blocks' contents counted as they grow. The emitters charge its growth at
    /// their poll points ([`crate::budget::Growth`]). Constant time in the size of the module (one
    /// term per function).
    pub fn footprint(&self) -> usize {
        use crate::budget::{map_bytes, tree_bytes, vec_bytes};
        let functions: usize = self
            .functions
            .iter()
            .map(|f| {
                vec_bytes(&f.params)
                    + vec_bytes(&f.param_tys)
                    + vec_bytes(&f.blocks)
                    + vec_bytes(&f.vars)
            })
            .sum();
        let bytes = self.types.footprint()
            + self.consts.footprint()
            + vec_bytes(&self.const_values)
            + vec_bytes(&self.values)
            + vec_bytes(&self.globals)
            + map_bytes(&self.resources)
            + map_bytes(&self.varyings)
            + map_bytes(&self.builtins)
            + vec_bytes(&self.push)
            + vec_bytes(&self.functions)
            + functions
            + vec_bytes(&self.blocks)
            + self.block_bytes
            + self.split_bytes
            + vec_bytes(&self.vars)
            + vec_bytes(&self.loops)
            + vec_bytes(&self.switches_at_use)
            + vec_bytes(&self.decls)
            + tree_bytes(self.invariant.len(), std::mem::size_of::<GlobalId>())
            + tree_bytes(self.volatile.len(), std::mem::size_of::<GlobalId>())
            + tree_bytes(
                self.enabling.len(),
                std::mem::size_of::<(GlobalId, crate::spv::Capability)>(),
            )
            + tree_bytes(self.non_uniform.len(), std::mem::size_of::<Value>())
            + tree_bytes(self.measured_roundings.len(), std::mem::size_of::<Value>())
            + map_bytes(&self.bits)
            + map_bytes(&self.lanes)
            + map_bytes(&self.inserts)
            + map_bytes(&self.widened16)
            + map_bytes(&self.widened_halves)
            + self.narrowed_floats.capacity() * std::mem::size_of::<Value>()
            + vec_bytes(&self.offers)
            + map_bytes(&self.encodes)
            + self.undefs.capacity() * std::mem::size_of::<Value>();
        #[cfg(test)]
        let bytes = bytes + tree_bytes(self.uniform_vars.len(), std::mem::size_of::<VarId>());
        bytes
    }

    /// Count the capacity a block's instruction or phi storage gained (`before` → `after`
    /// elements of `T`) into [`ModuleBuilder::footprint`].
    pub(super) fn count_block_growth<T>(&mut self, before: usize, after: usize) {
        self.block_bytes = self.block_bytes.saturating_add(
            after
                .saturating_sub(before)
                .saturating_mul(std::mem::size_of::<T>()),
        );
    }

    /// Count a switch split's working tables, `bytes`, into [`ModuleBuilder::footprint`].
    pub(super) fn count_split_tables(&mut self, bytes: usize) {
        self.split_bytes = self.split_bytes.saturating_add(bytes);
    }

    // ---- Types and constants ----

    /// Intern a type after checking its typing and layout rules.
    pub fn ty(&mut self, t: Type) -> BResult<TypeId> {
        let (id, fresh) = self.types.intern(t, &self.consts)?;
        if fresh {
            self.decls.push(Decl::Type(id));
        }
        Ok(id)
    }

    /// The type table.
    pub fn types(&self) -> &Types {
        &self.types
    }

    /// The equal values the lowering offered, for [`crate::spirv::dead::remove_dead_keeping`]
    /// once the module is finalized.
    pub fn offers(&self) -> &[Offer] {
        &self.offers
    }

    /// The constant table.
    pub fn consts(&self) -> &Consts {
        &self.consts
    }

    /// Intern a plain constant after checking it against its type.
    pub fn constant(&mut self, c: Constant) -> BResult<ConstId> {
        let ty = c.ty();
        let (id, fresh) = self.consts.intern(c, &self.types)?;
        if fresh {
            self.record_const(id, ty)?;
        }
        Ok(id)
    }

    /// Declare the specialization constant `key` of scalar type `ty`.
    pub fn spec(&mut self, key: SpecKey, ty: TypeId, default: ConstBits) -> BResult<ConstId> {
        let (id, fresh) = self.consts.spec(key, ty, default, &self.types)?;
        if fresh {
            self.record_const(id, ty)?;
        }
        Ok(id)
    }

    /// Declare `OpSpecConstantOp` `op` of type `ty` over `operands`: a value fixed at pipeline
    /// creation from specialization constants, such as the threadgroup arena's length. The operands
    /// are checked by [`Consts::spec_op`].
    pub fn spec_op(&mut self, op: SpecOp, ty: TypeId, operands: &[ConstId]) -> BResult<ConstId> {
        let (id, fresh) = self.consts.spec_op(op, ty, operands, &self.types)?;
        if fresh {
            self.record_const(id, ty)?;
        }
        Ok(id)
    }

    /// Declare the module's local size as the constants `x`, `y`, `z` (positive `u32` scalars, at
    /// least one a specialization constant from [`ModuleBuilder::spec`] and the others plain, as
    /// in the flattened `(sn, 1, 1)`): an `OpSpecConstantComposite` decorated `BuiltIn
    /// WorkgroupSize`. Valid under Vulkan 1.2 without `maintenance4`, unlike `LocalSizeId`. A
    /// `GLCompute` entry declared after it needs no local-size mode.
    pub fn workgroup_size(&mut self, x: ConstId, y: ConstId, z: ConstId) -> BResult<ConstId> {
        let u32_ = self.ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let ty = self.ty(Type::Vector { elem: u32_, len: 3 })?;
        let (id, fresh) = self.consts.workgroup_size(ty, [x, y, z], &self.types)?;
        if fresh {
            self.record_const(id, ty)?;
        }
        Ok(id)
    }

    fn record_const(&mut self, id: ConstId, ty: TypeId) -> BResult<()> {
        let v = self.new_value(ty, ValueOrigin::Const(id))?;
        self.const_values.push(v);
        self.decls.push(Decl::Const(id));
        Ok(())
    }

    /// The SSA value of a constant.
    pub fn const_value(&self, c: ConstId) -> BResult<Value> {
        self.const_values
            .get(c.index())
            .copied()
            .ok_or(BuildError::Dangling(Dangling::Const(c)))
    }

    /// The pointer value of a module-scope variable.
    pub fn global_value(&self, g: GlobalId) -> BResult<Value> {
        self.global(g).map(|g| g.value)
    }

    /// A module-scope variable.
    pub fn global(&self, g: GlobalId) -> BResult<&GlobalVar> {
        self.globals
            .get(g.index())
            .ok_or(BuildError::Dangling(Dangling::Global(g)))
    }

    /// The type of a value.
    pub fn value_ty(&self, v: Value) -> BResult<TypeId> {
        self.values
            .get(v.index())
            .map(|d| d.ty)
            .ok_or(BuildError::Dangling(Dangling::Value(v)))
    }

    pub(super) fn new_value(&mut self, ty: TypeId, origin: ValueOrigin) -> BResult<Value> {
        let v = u32::try_from(self.values.len())
            .map(Value::new)
            .map_err(|_| BuildError::IdSpaceExhausted)?;
        self.values.push(ValueDef { ty, origin });
        Ok(v)
    }

    // ---- Module-scope variables ----

    fn add_global(
        &mut self,
        class: StorageClass,
        pointee: TypeId,
        init: Option<ConstId>,
        kind: GlobalKind,
    ) -> BResult<GlobalId> {
        if let Some(c) = init {
            if self.consts.get(c)?.ty != pointee {
                return Err(BuildError::Global(GlobalRule::InitializerType));
            }
        }
        let ptr_ty = self.ty(Type::Pointer { class, pointee })?;
        let id = u32::try_from(self.globals.len())
            .map(GlobalId::new)
            .map_err(|_| BuildError::IdSpaceExhausted)?;
        let value = self.new_value(ptr_ty, ValueOrigin::Global(id))?;
        self.globals.push(GlobalVar {
            class,
            pointee,
            ptr_ty,
            value,
            init,
            kind,
        });
        self.decls.push(Decl::Global(id));
        Ok(id)
    }

    /// Declare the descriptor binding `sym`. `ty` is the variable's data type: a `Block` struct
    /// (or descriptor array of them) for `StorageBuffer`/`Uniform`, an opaque type (or descriptor
    /// array of them) for `UniformConstant`. Link assigns its set and binding.
    pub fn resource(
        &mut self,
        sym: ResourceSlot,
        class: StorageClass,
        ty: TypeId,
    ) -> BResult<GlobalId> {
        if self.resources.contains_key(&sym) {
            return Err(BuildError::Global(GlobalRule::DuplicateResource(sym)));
        }
        let f = *self.types.facts(ty)?;
        let ok = match class {
            StorageClass::StorageBuffer | StorageClass::Uniform => {
                f.block || f.shape == Shape::DescriptorArray { opaque: false }
            }
            StorageClass::UniformConstant => {
                matches!(
                    f.shape,
                    Shape::Opaque | Shape::DescriptorArray { opaque: true }
                )
            }
            other => return Err(BuildError::Global(GlobalRule::ResourceClass(other))),
        };
        // An input attachment is exactly one `SubpassData` image, which serialization decorates
        // with its `InputAttachmentIndex`.
        let subpass = matches!(
            self.types.get(ty)?,
            Type::Image(ImageType {
                dim: Dim::SubpassData,
                ..
            })
        );
        let ok = ok && subpass == matches!(sym.kind, ResourceKind::InputAttachment(_));
        // A combined image sampler is exactly one `OpTypeSampledImage`: a texture and sampler's,
        // or a planes image's through the device's conversion.
        let combined = matches!(self.types.get(ty)?, Type::SampledImage { .. });
        let ok = ok
            && combined
                == matches!(
                    sym.kind,
                    ResourceKind::CombinedTextureSampler(_) | ResourceKind::YCbCrPlanes(_)
                );
        // An acceleration structure binding is exactly one acceleration structure.
        let accel = matches!(self.types.get(ty)?, Type::AccelerationStructure);
        let ok = ok && accel == matches!(sym.kind, ResourceKind::AccelerationStructure(_));
        // A separate texture is one sampled image of a sampled shape, and a separate sampler one
        // sampler.
        let ok = ok
            && match sym.kind {
                ResourceKind::SeparateTexture(_) => matches!(
                    self.types.get(ty)?,
                    Type::Image(ImageType {
                        usage: ImageUsage::Sampled,
                        dim,
                        ..
                    }) if !matches!(dim, Dim::Buffer | Dim::SubpassData)
                ),
                ResourceKind::SeparateSampler(_) => matches!(self.types.get(ty)?, Type::Sampler),
                _ => true,
            };
        if !ok {
            return Err(BuildError::Global(GlobalRule::ResourceType));
        }
        let id = self.add_global(class, ty, None, GlobalKind::Resource(sym))?;
        self.resources.insert(sym, id);
        Ok(id)
    }

    /// Declare a second view of the declared `StorageBuffer` or `Uniform` binding `sym`: a
    /// variable of block type `ty` in the same storage class, bound to the same set and binding as
    /// the first, so both reach the same memory (a typed narrow view of a Metal buffer slot beside
    /// its word view; its `uvec4` view). Vulkan accepts several variables at one binding when each
    /// is compatible with its descriptor type; the serializer decorates every storage view of a
    /// Metal buffer `Aliased`, and link counts the binding once. [`resource`](Self::resource) must
    /// have declared `sym` in `class`.
    pub fn resource_view(
        &mut self,
        sym: ResourceSlot,
        class: StorageClass,
        ty: TypeId,
    ) -> BResult<GlobalId> {
        let first = self
            .resources
            .get(&sym)
            .copied()
            .ok_or(BuildError::Global(GlobalRule::ResourceType))?;
        let views = matches!(class, StorageClass::StorageBuffer | StorageClass::Uniform);
        if !views || self.global(first)?.class != class || !self.types.facts(ty)?.block {
            return Err(BuildError::Global(GlobalRule::ResourceType));
        }
        self.add_global(class, ty, None, GlobalKind::Resource(sym))
    }

    /// Declare a `Private`, `Workgroup`, or `TaskPayloadWorkgroupEXT` variable.
    pub fn plain_global(
        &mut self,
        class: StorageClass,
        ty: TypeId,
        init: Option<ConstId>,
    ) -> BResult<GlobalId> {
        match class {
            StorageClass::Workgroup
                if self.types.facts(ty)?.shape == Shape::Explicit
                    && !self.types.facts(ty)?.block =>
            {
                // An explicitly laid-out `Workgroup` variable must be a `Block`
                // (`SPV_KHR_workgroup_memory_explicit_layout`).
                Err(BuildError::Type(TypeRule::PointeeForClass(class)))
            }
            StorageClass::Private | StorageClass::Workgroup => {
                self.add_global(class, ty, init, GlobalKind::Plain)
            }
            // An object function's payload: implicitly laid out, never initialized.
            StorageClass::TaskPayloadWorkgroupEXT
                if init.is_none() && self.types.facts(ty)?.shape != Shape::Explicit =>
            {
                self.add_global(class, ty, init, GlobalKind::Plain)
            }
            other => Err(BuildError::Global(GlobalRule::PlainClass(other))),
        }
    }

    /// Add push field `f` of type `ty` at byte `offset` of the push block. `select` fixes the
    /// placement; fields arrive in increasing, non-overlapping offset order, so the member index is
    /// the call order. Fails once the block is sealed by [`ModuleBuilder::push_global`].
    pub fn push_member(&mut self, f: PushField, ty: TypeId, offset: u32) -> BResult<PushMember> {
        if self.push_global.is_some() {
            return Err(BuildError::Global(GlobalRule::PushSealed));
        }
        let facts = *self.types.facts(ty)?;
        if facts.block || facts.runtime_sized || !facts.layoutable {
            return Err(BuildError::Type(TypeRule::PointeeForClass(
                StorageClass::PushConstant,
            )));
        }
        if facts.matrix {
            return Err(BuildError::Layout {
                member: Some(self.push.len() as u32),
                violation: LayoutViolation::MatrixStridePresence,
            });
        }
        let min = match self.push.last() {
            None => 0,
            Some(prev) => {
                let size = self.types.facts(prev.ty)?.size().unwrap_or(u32::MAX);
                prev.offset.saturating_add(size)
            }
        };
        if offset < min || !offset.is_multiple_of(facts.align(LayoutRule::Scalar)) {
            return Err(BuildError::Global(GlobalRule::PushOrder { offset }));
        }
        let index = u32::try_from(self.push.len()).map_err(|_| BuildError::IdSpaceExhausted)?;
        let m = PushMember {
            field: f,
            index,
            ty,
            offset,
        };
        self.push.push(m);
        Ok(m)
    }

    /// The push-constant block variable. Seals the block: its type is interned now, under the
    /// strictest layout rule the placed offsets satisfy (std430, else scalar).
    pub fn push_global(&mut self) -> BResult<GlobalId> {
        if let Some(g) = self.push_global {
            return Ok(g);
        }
        if self.push.is_empty() {
            return Err(BuildError::Global(GlobalRule::PushEmpty));
        }
        let members: Box<[TypeId]> = self.push.iter().map(|m| m.ty).collect();
        let offsets: Box<[MemberLayout]> = self
            .push
            .iter()
            .map(|m| MemberLayout {
                offset: m.offset,
                matrix_stride: None,
            })
            .collect();
        let block = |rule| Type::Struct {
            members: members.clone(),
            layout: Some(StructLayout {
                rule,
                block: true,
                members: offsets.clone(),
            }),
        };
        let ty = match self.ty(block(LayoutRule::Std430)) {
            Ok(ty) => ty,
            Err(BuildError::Layout { .. }) => self.ty(block(LayoutRule::Scalar))?,
            Err(e) => return Err(e),
        };
        let g = self.add_global(StorageClass::PushConstant, ty, None, GlobalKind::PushBlock)?;
        self.push_global = Some(g);
        Ok(g)
    }

    /// Declare the user varying `key` in direction `dir`. Link assigns its `Location` (and
    /// `Component` when packed).
    pub fn io(
        &mut self,
        key: VaryingKey,
        dir: IoDir,
        ty: TypeId,
        deco: IoDecor,
    ) -> BResult<GlobalId> {
        if self.varyings.contains_key(&(key.clone(), dir)) {
            return Err(BuildError::Global(GlobalRule::DuplicateVarying(key, dir)));
        }
        if !self.varying_type_ok(ty)? || (deco.index.is_some() && dir == IoDir::Input) {
            return Err(BuildError::Global(GlobalRule::VaryingType));
        }
        let id = self.add_global(
            dir.class(),
            ty,
            None,
            GlobalKind::Io {
                key: key.clone(),
                dir,
                deco,
            },
        )?;
        self.varyings.insert((key, dir), id);
        Ok(id)
    }

    /// Declare a user variable at the fixed `Location` `location` in direction `dir`, decorated
    /// `Patch` when `patch`: an interface between stages the crate synthesizes, which no Metal
    /// varying reaches and link does not number (the tessellation chain). A per-vertex input of a
    /// tessellation control stage is an array whose outer level is the vertex.
    pub fn fixed_io(
        &mut self,
        location: u32,
        dir: IoDir,
        ty: TypeId,
        patch: bool,
    ) -> BResult<GlobalId> {
        let taken = self.globals.iter().any(|g| {
            matches!(g.kind, GlobalKind::FixedIo { location: l, dir: d, .. } if l == location && d == dir)
        });
        if taken {
            return Err(BuildError::Global(GlobalRule::DuplicateLocation(
                location, dir,
            )));
        }
        if !self.varying_type_ok(ty)? {
            return Err(BuildError::Global(GlobalRule::VaryingType));
        }
        self.add_global(
            dir.class(),
            ty,
            None,
            GlobalKind::FixedIo {
                location,
                dir,
                patch,
            },
        )
    }

    /// A varying holds numeric scalars, vectors, matrices, and implicit-layout aggregates of
    /// them: no `bool`, pointer, or opaque value anywhere inside.
    fn varying_type_ok(&self, ty: TypeId) -> BResult<bool> {
        let mut stack = vec![ty];
        while let Some(t) = stack.pop() {
            match self.types.get(t)? {
                Type::Int { .. } | Type::Float { .. } | Type::Matrix { .. } => {}
                Type::Vector { elem, .. } => stack.push(*elem),
                Type::Array {
                    elem, stride: None, ..
                } => stack.push(*elem),
                Type::Struct {
                    members,
                    layout: None,
                } => stack.extend(members.iter().copied()),
                _ => return Ok(false),
            }
        }
        Ok(true)
    }

    /// Decorate builtin output `g` `Invariant`: SPIR-V's "expressions computing its value be
    /// computed invariantly with respect to other shaders computing the same expressions". Only
    /// the `Position` output (Metal's `[[invariant]]` position) and a fragment's per-fragment
    /// outputs, `FragDepth`, `SampleMask` and `FragStencilRefEXT` (the variants of one render
    /// pipeline's color groups, `LoweringInputs::color_group`), can be invariant.
    pub fn invariant(&mut self, g: GlobalId) -> BResult<()> {
        use crate::spv::BuiltIn as B;
        match self.global(g)?.kind {
            GlobalKind::BuiltIn {
                builtin: B::Position | B::FragDepth | B::SampleMask | B::FragStencilRefEXT,
                dir: IoDir::Output,
            } => {
                self.invariant.insert(g);
                Ok(())
            }
            _ => Err(BuildError::Global(GlobalRule::InvariantTarget)),
        }
    }

    /// Mark the module's `GLSL.std.450 InterpolateAtOffset` offsets as Metal's offsets moved by
    /// 7/16, for a device whose instruction reads Metal's `interpolate_at_offset` at its offset
    /// plus 7/16: the module then requires
    /// `VkProperty::InterpolateAtOffsetFromSevenSixteenths`, where every other use of the
    /// instruction requires `VkProperty::InterpolateAtOffsetCentered`.
    pub fn offsets_from_seven_sixteenths(&mut self) {
        self.offsets_from_seven_sixteenths = true;
    }

    /// Decorate user varying `g` `PerPrimitiveEXT`: a mesh entry's primitive output, or a fragment
    /// input that reads one. A mesh entry's primitive builtins need no call: the serializer
    /// decorates them.
    pub fn per_primitive(&mut self, g: GlobalId) -> BResult<()> {
        match self.global(g)?.kind {
            GlobalKind::Io { .. } => {
                self.per_primitive.insert(g);
                Ok(())
            }
            _ => Err(BuildError::Global(GlobalRule::PerPrimitiveTarget)),
        }
    }

    /// Decorate every storage buffer and storage image resource of the module `Coherent`, those
    /// declared later included: under the `GLSL450` memory model each access through them is
    /// made available and visible at device scope, which is what an ordered fragment-interlock
    /// critical section orders its accesses by. The Vulkan memory model bans the decoration
    /// ([`ModuleBuilder::finalize`]); its accesses carry the per-access operands instead.
    pub fn coherent_resources(&mut self) {
        self.coherent = true;
    }

    /// Decorate storage-image resource `g` `Coherent`: under the `GLSL450` memory model each access
    /// to its texels is made available and visible at device scope, as
    /// [`coherent_resources`](ModuleBuilder::coherent_resources) makes every shared resource's,
    /// for one image whose accesses an ordered fragment-interlock critical section orders while
    /// the module's other resources keep plain accesses. Only a `UniformConstant` resource whose
    /// type is a storage image can be; the Vulkan memory model bans the decoration
    /// ([`ModuleBuilder::finalize`]).
    pub fn coherent(&mut self, g: GlobalId) -> BResult<()> {
        let gv = self.global(g)?;
        let storage = gv.class == StorageClass::UniformConstant
            && matches!(gv.kind, GlobalKind::Resource(_))
            && matches!(
                self.types.get(gv.pointee)?,
                Type::Image(ImageType {
                    usage: ImageUsage::Storage,
                    ..
                })
            );
        if !storage {
            return Err(BuildError::Global(GlobalRule::CoherentTarget));
        }
        self.coherent_images.insert(g);
        Ok(())
    }

    /// Decorate storage-image resource `g` `Volatile`: under the `GLSL450` memory model every
    /// access to its texels is a volatile access, so a load never returns a value the invocation
    /// stored to that out-of-bounds location and the device's robustness discarded. Vulkan lets
    /// robust image and texel-buffer access return such a value, and names the decoration as what
    /// prevents it. Only a `UniformConstant` resource whose type is a storage image (a storage
    /// texel buffer included), or a descriptor array of them, can be volatile; the Vulkan memory
    /// model bans the decoration ([`ModuleBuilder::finalize`]).
    pub fn volatile(&mut self, g: GlobalId) -> BResult<()> {
        let gv = self.global(g)?;
        let mut ty = gv.pointee;
        if let Type::Array { elem, .. } | Type::RuntimeArray { elem, .. } = self.types.get(ty)? {
            ty = *elem;
        }
        let storage = gv.class == StorageClass::UniformConstant
            && matches!(gv.kind, GlobalKind::Resource(_))
            && matches!(
                self.types.get(ty)?,
                Type::Image(ImageType {
                    usage: ImageUsage::Storage,
                    ..
                })
            );
        if !storage {
            return Err(BuildError::Global(GlobalRule::VolatileTarget));
        }
        self.volatile.insert(g);
        Ok(())
    }

    /// Declare builtin variable `g` through enabling capability `cap`, one of the alternatives
    /// the registry offers for it (`requirements::builtin_enablers`): requirement derivation then
    /// declares `cap` and its feature in place of the default alternative. Lowering chooses the
    /// alternative the profile enables; choosing a different one for the same variable again is
    /// refused.
    pub fn enable_builtin(&mut self, g: GlobalId, cap: crate::spv::Capability) -> BResult<()> {
        let GlobalKind::BuiltIn { builtin, dir } = self.global(g)?.kind else {
            return Err(BuildError::Global(GlobalRule::BuiltInEnabler(None)));
        };
        let offered = requirements::builtin_enablers(builtin, dir)
            .iter()
            .any(|(c, _)| *c == cap);
        if !offered {
            return Err(BuildError::Global(GlobalRule::BuiltInEnabler(Some(
                builtin,
            ))));
        }
        match self.enabling.insert(g, cap) {
            Some(prev) if prev != cap => Err(BuildError::Global(GlobalRule::BuiltInEnabler(Some(
                builtin,
            )))),
            _ => Ok(()),
        }
    }

    /// Declare builtin `b` in direction `dir` with data type `ty`. Declaring the same builtin and
    /// direction again returns the first variable. A mesh entry's outputs are arrays of the
    /// builtin's type, one element per vertex or primitive; a mesh-only builtin
    /// (`CullPrimitiveEXT`, the primitive indices) exists only so. A geometry entry's per-vertex
    /// inputs are arrays of the builtin's type, one element per vertex of its input primitive.
    /// Which model may declare which form is the verifier's.
    pub fn builtin(&mut self, b: crate::spv::BuiltIn, dir: IoDir, ty: TypeId) -> BResult<GlobalId> {
        let (dirs, shape) =
            builtin_contract(b).ok_or(BuildError::Global(GlobalRule::UnsupportedBuiltIn(b)))?;
        if !dirs.contains(&dir) {
            return Err(BuildError::Global(GlobalRule::BuiltInDirection(b, dir)));
        }
        if let Some(g) = self.builtins.get(&(b, dir)) {
            let g = *g;
            return if self.global(g)?.pointee == ty {
                Ok(g)
            } else {
                Err(BuildError::Global(GlobalRule::BuiltInType(b)))
            };
        }
        let arrayed = ((dir == IoDir::Output && mesh_arrayed(b))
            || (dir == IoDir::Input && geometry_per_vertex(b)))
            && match *self.types.get(ty)? {
                Type::Array {
                    elem, stride: None, ..
                } => self.builtin_shape_ok(shape, elem)?,
                _ => false,
            };
        if !arrayed && (mesh_only(b) || !self.builtin_shape_ok(shape, ty)?) {
            return Err(BuildError::Global(GlobalRule::BuiltInType(b)));
        }
        let id = self.add_global(
            dir.class(),
            ty,
            None,
            GlobalKind::BuiltIn { builtin: b, dir },
        )?;
        self.builtins.insert((b, dir), id);
        Ok(id)
    }

    fn builtin_shape_ok(&self, shape: BuiltInShape, ty: TypeId) -> BResult<bool> {
        let t = self.types.get(ty)?;
        let int32 = |id: TypeId| -> BResult<bool> {
            Ok(matches!(self.types.get(id)?, Type::Int { width: 32, .. }))
        };
        let f32 = |id: TypeId| -> BResult<bool> {
            Ok(matches!(self.types.get(id)?, Type::Float { width: 32 }))
        };
        Ok(match (shape, t) {
            (BuiltInShape::Int32, _) => int32(ty)?,
            (BuiltInShape::F32, _) => f32(ty)?,
            (BuiltInShape::Bool, Type::Bool) => true,
            (BuiltInShape::UVec3, Type::Vector { elem, len: 3 })
            | (BuiltInShape::UVec2, Type::Vector { elem, len: 2 }) => {
                matches!(
                    self.types.get(*elem)?,
                    Type::Int {
                        width: 32,
                        signed: false
                    }
                )
            }
            (BuiltInShape::Vec(n), Type::Vector { elem, len }) => *len == n && f32(*elem)?,
            (
                BuiltInShape::F32Array,
                Type::Array {
                    elem, stride: None, ..
                },
            ) => f32(*elem)?,
            (
                BuiltInShape::Int32Array,
                Type::Array {
                    elem, stride: None, ..
                },
            ) => int32(*elem)?,
            _ => false,
        })
    }

    // ---- Functions and the entry point ----

    /// Start a function returning `ret`. Its entry block is open.
    pub fn function(&mut self, ret: TypeId) -> BResult<FunctionBuilder<'_>> {
        self.start_function(ret, Started::Plain)
    }

    /// Start a routine returning `ret`: a function the lowering defines once and calls from
    /// every site that needs it. Written for a profile whose driver converts every function to
    /// one inlined at each call (`DriverWorkaround::FunctionsInlinedAtEveryCall`: MoltenVK's
    /// SPIRV-Cross 1.4.1, which still does so), it is declared `DontInline` (`OpFunction`'s
    /// function control), the standard SPIR-V statement that each call should stay a call; for
    /// any other profile it takes no control ([`crate::spirv::serialize::serialize_for`]). Its
    /// entry block is open.
    pub fn routine(&mut self, ret: TypeId) -> BResult<FunctionBuilder<'_>> {
        self.start_function(ret, Started::Routine)
    }

    /// Start a texel routine returning `ret`: a [`routine`](Self::routine) that sums a linear
    /// filter emulation's taps, declared `DontInline` for every profile
    /// ([`crate::spirv::serialize::serialize_for`]). A driver that honours the control (NVIDIA's)
    /// compiles it once where it would otherwise compile it again inside every sample that calls
    /// it, which made a fragment of a few hundred emulated samples take tens of seconds to create
    /// on an RTX 3090; one that inlines every function (Mesa's) ignores it. Its entry block is
    /// open.
    pub fn texel_routine(&mut self, ret: TypeId) -> BResult<FunctionBuilder<'_>> {
        self.start_function(ret, Started::Texel)
    }

    /// Start a sampling routine returning `ret`: a [`routine`](Self::routine) that holds the
    /// emulation of one combined binding's sample shape, which every sample of that shape calls
    /// with its plain data. It is declared `DontInline` where the profile states that its driver
    /// keeps such a call a call (`DriverWorkaround::DontInlineKeepsCalls`, a measured cost fact),
    /// and, as every routine, where its driver inlines every function at each call
    /// (`DriverWorkaround::FunctionsInlinedAtEveryCall`); elsewhere it takes no control
    /// ([`crate::spirv::serialize::serialize_for`]). Its entry block is open.
    pub fn sampling_routine(&mut self, ret: TypeId) -> BResult<FunctionBuilder<'_>> {
        self.start_function(ret, Started::Sampling)
    }

    fn start_function(&mut self, ret: TypeId, started: Started) -> BResult<FunctionBuilder<'_>> {
        match self.types.facts(ret)?.shape {
            Shape::Plain | Shape::Implicit => {}
            Shape::NotData if matches!(self.types.get(ret)?, Type::Void) => {}
            _ => return Err(BuildError::Type(TypeRule::FunctionSignature)),
        }
        let func = u32::try_from(self.functions.len())
            .map(FuncId::new)
            .map_err(|_| BuildError::IdSpaceExhausted)?;
        let entry = self.new_block(func)?;
        self.functions.push(FunctionDraft {
            ret,
            params: Vec::new(),
            param_tys: Vec::new(),
            blocks: vec![entry],
            vars: Vec::new(),
            fn_ty: None,
            routine: started != Started::Plain,
            kept: started == Started::Texel,
            sampling: started == Started::Sampling,
        });
        Ok(FunctionBuilder::new(self, func, entry))
    }

    pub(super) fn new_block(&mut self, func: FuncId) -> BResult<BlockId> {
        let id = u32::try_from(self.blocks.len())
            .map(BlockId::new)
            .map_err(|_| BuildError::IdSpaceExhausted)?;
        self.blocks.push(BlockDraft {
            func,
            phis: Vec::new(),
            insts: Vec::new(),
            merge: None,
            term: None,
        });
        Ok(id)
    }

    pub(super) fn draft(&self, f: FuncId) -> BResult<&FunctionDraft> {
        self.functions
            .get(f.index())
            .ok_or(BuildError::Dangling(Dangling::Func(f)))
    }

    /// Record the subgroup requirement the plan chose for the entry: its pinned size and full
    /// subgroups are stated outside the SPIR-V, so `finalize` adds what they need of the device to
    /// the derived requirements (`requirements::recorded`).
    pub fn subgroup(&mut self, requirement: SubgroupRequirement) {
        self.subgroup = Some(requirement);
    }

    /// Declare the module's entry point: finished function `f`, which takes no parameters and
    /// returns void. Fragment entries get `OriginUpperLeft` when `modes` lacks it.
    pub fn entry(&mut self, model: ExecModel, f: FuncId, mut modes: Vec<ExecMode>) -> BResult<()> {
        if self.entry.is_some() {
            return Err(BuildError::Entry(EntryRule::Duplicate));
        }
        let d = self.draft(f)?;
        if d.fn_ty.is_none() {
            return Err(BuildError::Control(ControlRule::UnfinishedFunction(f)));
        }
        if !d.params.is_empty() || !matches!(self.types.get(d.ret)?, Type::Void) {
            return Err(BuildError::Entry(EntryRule::Signature));
        }
        if model == ExecModel::Fragment && !modes.contains(&ExecMode::OriginUpperLeft) {
            modes.insert(0, ExecMode::OriginUpperLeft);
        }
        self.check_modes(model, &modes)?;
        self.entry = Some(EntryPoint {
            model,
            func: f,
            modes,
        });
        Ok(())
    }

    fn check_modes(&self, model: ExecModel, modes: &[ExecMode]) -> BResult<()> {
        let err = |r| Err(BuildError::Entry(r));
        let mut groups = BTreeSet::new();
        let mut has_local_size = false;
        let mesh = model == ExecModel::MeshEXT;
        for m in modes {
            let (group, width, models) = mode_contract(*m);
            if !models.contains(&model) {
                return err(EntryRule::ModeForModel);
            }
            if let Some(w) = width {
                if !matches!(w, 16 | 32 | 64) {
                    return err(EntryRule::FloatWidth);
                }
            }
            if !groups.insert((group, width)) {
                return err(EntryRule::ConflictingModes);
            }
            match m {
                ExecMode::LocalSize { x, y, z } => {
                    has_local_size = true;
                    if *x == 0 || *y == 0 || *z == 0 {
                        return err(EntryRule::LocalSize);
                    }
                }
                ExecMode::OutputVertices { n }
                    if model == ExecModel::TessellationControl && (*n == 0 || *n > 32) =>
                {
                    return err(EntryRule::TessellationModes);
                }
                ExecMode::OutputVertices { n: 0 } | ExecMode::Invocations { n: 0 }
                    if model == ExecModel::Geometry =>
                {
                    return err(EntryRule::GeometryModes);
                }
                ExecMode::OutputVertices { n: 0 } | ExecMode::OutputPrimitives { n: 0 } => {
                    return err(EntryRule::MeshModes);
                }
                _ => {}
            }
        }
        // The crate declares the domain in both tessellation stages and the output patch size in
        // the control stage, so each module is complete on its own.
        let domain = modes
            .iter()
            .any(|m| matches!(m, ExecMode::Triangles | ExecMode::Quads));
        let patch_size = modes
            .iter()
            .any(|m| matches!(m, ExecMode::OutputVertices { .. }));
        let tessellation_ok = match model {
            ExecModel::TessellationEvaluation => domain,
            ExecModel::TessellationControl => domain && patch_size,
            _ => true,
        };
        if !tessellation_ok {
            return err(EntryRule::TessellationModes);
        }
        // A mesh entry states the most vertices and primitives it writes and their kind.
        let mesh_ok = !mesh
            || (patch_size
                && modes
                    .iter()
                    .any(|m| matches!(m, ExecMode::OutputPrimitives { .. }))
                && modes.iter().any(|m| {
                    matches!(
                        m,
                        ExecMode::OutputPoints | ExecMode::OutputLines | ExecMode::OutputTriangles
                    )
                }));
        if !mesh_ok {
            return err(EntryRule::MeshModes);
        }
        // A geometry entry states its input and output primitives and the most vertices it emits.
        let geometry_ok = model != ExecModel::Geometry
            || (patch_size
                && modes.iter().any(|m| {
                    matches!(
                        m,
                        ExecMode::InputPoints | ExecMode::InputLines | ExecMode::Triangles
                    )
                })
                && modes.iter().any(|m| {
                    matches!(
                        m,
                        ExecMode::OutputPoints
                            | ExecMode::OutputLineStrip
                            | ExecMode::OutputTriangleStrip
                    )
                }));
        if !geometry_ok {
            return err(EntryRule::GeometryModes);
        }
        let has_local_size = has_local_size || self.consts.workgroup_size_const().is_some();
        let workgroups = matches!(
            model,
            ExecModel::GLCompute | ExecModel::TaskEXT | ExecModel::MeshEXT
        );
        if workgroups && !has_local_size {
            return err(EntryRule::MissingLocalSize);
        }
        Ok(())
    }

    /// Finish the module. Every function must be finished and the entry point declared; an unsealed
    /// push block is sealed. Loop barrier facts, the entry interface, and the device requirements
    /// are derived here.
    pub fn finalize(mut self) -> BResult<TypedModule> {
        if !self.push.is_empty() {
            self.push_global()?;
        }
        let entry = self.entry.ok_or(BuildError::Entry(EntryRule::Missing))?;
        let (values, types) = (&self.values, &self.types);
        super::ops::check_entry_model(&self.blocks, &entry, |v| {
            let ty = values
                .get(v.index())
                .map(|d| d.ty)
                .ok_or(BuildError::Dangling(Dangling::Value(v)))?;
            Ok(matches!(
                types.get(ty)?,
                Type::Image(ImageType {
                    dim: Dim::SubpassData,
                    ..
                })
            ))
        })?;
        constructs::check_kills(&self.blocks, &entry)?;
        let mut functions = Vec::with_capacity(self.functions.len());
        for (i, d) in self.functions.into_iter().enumerate() {
            let fn_ty = d
                .fn_ty
                .ok_or(BuildError::Control(ControlRule::UnfinishedFunction(
                    FuncId::new(i as u32),
                )))?;
            functions.push(Function {
                ret: d.ret,
                fn_ty,
                params: d.params,
                blocks: d.blocks,
                vars: d.vars,
                routine: d.routine,
                kept: d.kept,
                sampling: d.sampling,
            });
        }
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for (i, b) in self.blocks.into_iter().enumerate() {
            // `finish` terminates every block of a finished function, so this only fires for a
            // block of an unfinished one, which the loop above already refused.
            let term = b
                .term
                .ok_or(BuildError::Control(ControlRule::UnterminatedBlock(
                    BlockId::new(i as u32),
                )))?;
            blocks.push(Block {
                func: b.func,
                phis: b.phis,
                insts: b.insts,
                merge: b.merge,
                term,
            });
        }
        let loops = self.loops;
        constructs::check_switches_at_use(&functions, &blocks, &self.switches_at_use)?;
        let view = requirements::ModuleView {
            types: &self.types,
            consts: &self.consts,
            values: &self.values,
            globals: &self.globals,
            functions: &functions,
            blocks: &blocks,
            entry: &entry,
            enabling: &self.enabling,
            per_primitive: &self.per_primitive,
            offsets_from_seven_sixteenths: self.offsets_from_seven_sixteenths,
        };
        let interface = requirements::interface(&view)?;
        let subgroup = self.subgroup.unwrap_or(SubgroupRequirement::NONE);
        let requirements = requirements::recorded(&view, &subgroup)?;
        // The Vulkan memory model bans the `Volatile` and `Coherent` decorations (its accesses
        // carry the `VolatileTexel`, `NonPrivate*`, and `Make*` operands instead).
        let vulkan = requirements::uses_vulkan_memory_model(&requirements);
        if !self.volatile.is_empty() && vulkan {
            return Err(BuildError::Global(GlobalRule::VolatileTarget));
        }
        if (self.coherent || !self.coherent_images.is_empty()) && vulkan {
            return Err(BuildError::Global(GlobalRule::CoherentTarget));
        }
        let mut coherent = if self.coherent {
            shared_resources(&self.types, &self.globals)?
        } else {
            BTreeSet::new()
        };
        coherent.extend(self.coherent_images);
        Ok(TypedModule {
            types: self.types,
            consts: self.consts,
            const_values: self.const_values,
            values: self.values,
            globals: self.globals,
            push: self.push,
            functions,
            blocks,
            vars: self.vars,
            loops,
            switches_at_use: self.switches_at_use,
            entry,
            decls: self.decls,
            interface,
            requirements,
            subgroup,
            invariant: self.invariant,
            per_primitive: self.per_primitive,
            offsets_from_seven_sixteenths: self.offsets_from_seven_sixteenths,
            coherent,
            volatile: self.volatile,
            enabling: self.enabling,
            non_uniform: self.non_uniform,
            measured_roundings: self.measured_roundings,
            #[cfg(test)]
            uniform_vars: self.uniform_vars,
        })
    }
}

/// The value shape a builtin variable must have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BuiltInShape {
    /// A 32-bit integer of either signedness.
    Int32,
    F32,
    Bool,
    /// `uvec3`.
    UVec3,
    /// `uvec2`.
    UVec2,
    /// A `f32` vector of this length.
    Vec(u8),
    /// An array of `f32`.
    F32Array,
    /// An array of 32-bit integers.
    Int32Array,
}

/// The directions and shape of each builtin the new core declares. Stage legality is the
/// verifier's.
fn builtin_contract(b: crate::spv::BuiltIn) -> Option<(&'static [IoDir], BuiltInShape)> {
    use crate::spv::BuiltIn as B;
    const IN: &[IoDir] = &[IoDir::Input];
    const OUT: &[IoDir] = &[IoDir::Output];
    const BOTH: &[IoDir] = &[IoDir::Input, IoDir::Output];
    Some(match b {
        B::GlobalInvocationId | B::LocalInvocationId | B::WorkgroupId | B::NumWorkgroups => {
            (IN, BuiltInShape::UVec3)
        }
        B::LocalInvocationIndex
        | B::SubgroupSize
        | B::SubgroupLocalInvocationId
        | B::NumSubgroups
        | B::SubgroupId
        | B::VertexIndex
        | B::InstanceIndex
        | B::BaseVertex
        | B::BaseInstance
        | B::DrawIndex
        | B::SampleId
        | B::ViewIndex => (IN, BuiltInShape::Int32),
        // A fragment's input, or a mesh entry's per-primitive output.
        B::PrimitiveId => (BOTH, BuiltInShape::Int32),
        B::FragCoord => (IN, BuiltInShape::Vec(4)),
        // Tessellation.
        B::TessCoord => (IN, BuiltInShape::Vec(3)),
        B::InvocationId | B::PatchVertices => (IN, BuiltInShape::Int32),
        B::TessLevelOuter | B::TessLevelInner => (OUT, BuiltInShape::F32Array),
        B::PointCoord | B::SamplePosition => (IN, BuiltInShape::Vec(2)),
        B::FrontFacing | B::HelperInvocation => (IN, BuiltInShape::Bool),
        // A geometry entry's per-vertex inputs, as well as outputs.
        B::Position => (BOTH, BuiltInShape::Vec(4)),
        B::PointSize => (BOTH, BuiltInShape::F32),
        B::FragDepth => (OUT, BuiltInShape::F32),
        B::FragStencilRefEXT => (OUT, BuiltInShape::Int32),
        B::Layer | B::ViewportIndex => (BOTH, BuiltInShape::Int32),
        B::ClipDistance | B::CullDistance => (BOTH, BuiltInShape::F32Array),
        B::SampleMask => (BOTH, BuiltInShape::Int32Array),
        B::BaryCoordKHR | B::BaryCoordNoPerspKHR => (IN, BuiltInShape::Vec(3)),
        // A mesh entry's, arrayed per primitive.
        B::PrimitivePointIndicesEXT => (OUT, BuiltInShape::Int32),
        B::PrimitiveLineIndicesEXT => (OUT, BuiltInShape::UVec2),
        B::PrimitiveTriangleIndicesEXT => (OUT, BuiltInShape::UVec3),
        B::CullPrimitiveEXT => (OUT, BuiltInShape::Bool),
        _ => return None,
    })
}

/// Whether a geometry entry declares builtin input `b` as an array, one element per vertex of its
/// input primitive (Vulkan's `gl_in`).
pub(crate) fn geometry_per_vertex(b: crate::spv::BuiltIn) -> bool {
    use crate::spv::BuiltIn as B;
    matches!(
        b,
        B::Position | B::PointSize | B::ClipDistance | B::CullDistance
    )
}

/// Whether a mesh entry declares builtin output `b` as an array, one element per vertex or
/// primitive.
pub(crate) fn mesh_arrayed(b: crate::spv::BuiltIn) -> bool {
    use crate::spv::BuiltIn as B;
    matches!(
        b,
        B::Position
            | B::PointSize
            | B::ClipDistance
            | B::CullDistance
            | B::PrimitiveId
            | B::Layer
            | B::ViewportIndex
            | B::CullPrimitiveEXT
            | B::PrimitivePointIndicesEXT
            | B::PrimitiveLineIndicesEXT
            | B::PrimitiveTriangleIndicesEXT
    )
}

/// Whether builtin `b` exists only as a mesh entry's arrayed output.
fn mesh_only(b: crate::spv::BuiltIn) -> bool {
    use crate::spv::BuiltIn as B;
    matches!(
        b,
        B::CullPrimitiveEXT
            | B::PrimitivePointIndicesEXT
            | B::PrimitiveLineIndicesEXT
            | B::PrimitiveTriangleIndicesEXT
    )
}

/// Whether a mesh entry's builtin output `b` is per primitive, and so decorated
/// `PerPrimitiveEXT` (VUID-PrimitiveId-PrimitiveId-07040, VUID-Layer-Layer-07039,
/// VUID-ViewportIndex-ViewportIndex-07060, VUID-CullPrimitiveEXT-CullPrimitiveEXT-07038).
pub(crate) fn mesh_per_primitive(b: crate::spv::BuiltIn) -> bool {
    use crate::spv::BuiltIn as B;
    matches!(
        b,
        B::PrimitiveId | B::Layer | B::ViewportIndex | B::CullPrimitiveEXT
    )
}

/// For an execution mode: its exclusivity group, its float width (if any), and the models that
/// allow it. Two modes with the same (group, width) conflict.
fn mode_contract(m: ExecMode) -> (u8, Option<u8>, &'static [ExecModel]) {
    const FRAG: &[ExecModel] = &[ExecModel::Fragment];
    const COMP: &[ExecModel] = &[ExecModel::GLCompute];
    // The models that run in workgroups of a local size.
    const WORKGROUPS: &[ExecModel] =
        &[ExecModel::GLCompute, ExecModel::TaskEXT, ExecModel::MeshEXT];
    const MESH: &[ExecModel] = &[ExecModel::MeshEXT];
    const GEOMETRY: &[ExecModel] = &[ExecModel::Geometry];
    const MESH_OR_GEOMETRY: &[ExecModel] = &[ExecModel::MeshEXT, ExecModel::Geometry];
    const ALL: &[ExecModel] = &[
        ExecModel::Vertex,
        ExecModel::Fragment,
        ExecModel::GLCompute,
        ExecModel::TessellationEvaluation,
        ExecModel::TessellationControl,
        ExecModel::TaskEXT,
        ExecModel::MeshEXT,
        ExecModel::Geometry,
    ];
    // Vulkan lets either tessellation stage declare the domain, spacing, and vertex order; the
    // output patch size is the control stage's.
    const TESS: &[ExecModel] = &[
        ExecModel::TessellationEvaluation,
        ExecModel::TessellationControl,
    ];
    // The triangle domain, or a geometry entry's triangle input.
    const TRIANGLES: &[ExecModel] = &[
        ExecModel::TessellationEvaluation,
        ExecModel::TessellationControl,
        ExecModel::Geometry,
    ];
    // The output patch size, or a mesh or geometry entry's most vertices.
    const OUTPUT_VERTICES: &[ExecModel] = &[
        ExecModel::TessellationControl,
        ExecModel::MeshEXT,
        ExecModel::Geometry,
    ];
    match m {
        ExecMode::OriginUpperLeft => (0, None, FRAG),
        ExecMode::EarlyFragmentTests => (1, None, FRAG),
        ExecMode::DepthReplacing => (2, None, FRAG),
        ExecMode::DepthGreater | ExecMode::DepthLess => (3, None, FRAG),
        ExecMode::StencilRefReplacing => (4, None, FRAG),
        ExecMode::PixelInterlockOrdered | ExecMode::SampleInterlockOrdered => (5, None, FRAG),
        ExecMode::LocalSize { .. } => (6, None, WORKGROUPS),
        ExecMode::DerivativeGroupLinear => (7, None, COMP),
        ExecMode::SignedZeroInfNanPreserve { width } => (9, Some(width), ALL),
        ExecMode::DenormPreserve { width } => (8, Some(width), ALL),
        ExecMode::RoundingModeRte { width } => (10, Some(width), ALL),
        ExecMode::SubgroupUniformControlFlow => (11, None, ALL),
        ExecMode::MaximallyReconverges => (12, None, ALL),
        ExecMode::Triangles => (13, None, TRIANGLES),
        ExecMode::Quads => (13, None, TESS),
        ExecMode::InputPoints | ExecMode::InputLines => (13, None, GEOMETRY),
        ExecMode::SpacingEqual
        | ExecMode::SpacingFractionalEven
        | ExecMode::SpacingFractionalOdd => (14, None, TESS),
        ExecMode::VertexOrderCw | ExecMode::VertexOrderCcw => (15, None, TESS),
        ExecMode::OutputVertices { .. } => (16, None, OUTPUT_VERTICES),
        ExecMode::OutputPrimitives { .. } => (17, None, MESH),
        ExecMode::OutputPoints => (18, None, MESH_OR_GEOMETRY),
        ExecMode::OutputLines | ExecMode::OutputTriangles => (18, None, MESH),
        ExecMode::OutputLineStrip | ExecMode::OutputTriangleStrip => (18, None, GEOMETRY),
        ExecMode::RequireFullQuads => (19, None, FRAG),
        ExecMode::PostDepthCoverage => (21, None, FRAG),
        ExecMode::Invocations { .. } => (20, None, GEOMETRY),
    }
}

/// The resources other invocations may write: every `StorageBuffer` resource, and every
/// `UniformConstant` resource whose type is a storage image (a storage texel buffer
/// included) or a descriptor array of them.
fn shared_resources(types: &Types, globals: &[GlobalVar]) -> BResult<BTreeSet<GlobalId>> {
    let mut out = BTreeSet::new();
    for (i, gv) in globals.iter().enumerate() {
        if !matches!(gv.kind, GlobalKind::Resource(_)) {
            continue;
        }
        let shared = match gv.class {
            StorageClass::StorageBuffer => true,
            StorageClass::UniformConstant => {
                let mut ty = gv.pointee;
                if let Type::Array { elem, .. } | Type::RuntimeArray { elem, .. } = types.get(ty)? {
                    ty = *elem;
                }
                matches!(
                    types.get(ty)?,
                    Type::Image(ImageType {
                        usage: ImageUsage::Storage,
                        dim,
                        ..
                    }) if *dim != Dim::SubpassData
                )
            }
            _ => false,
        };
        if shared {
            let id = u32::try_from(i).map_err(|_| BuildError::IdSpaceExhausted)?;
            out.insert(GlobalId::new(id));
        }
    }
    Ok(out)
}
