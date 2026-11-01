//! Capability / extension / feature derivation (spirv-and-lowering spec A4) and the entry
//! interface (SPIR-V 1.4+ rule).
//!
//! No lowering writes `OpCapability` or `OpExtension`. `ModuleBuilder::finalize` calls
//! [`recorded`]: [`derive()`], one exhaustive walk over what the module actually contains (types,
//! module-scope variables and builtins, execution modes, instructions, terminators), plus the
//! facts of the module's subgroup requirement (below). It records the resulting [`Requirements`] on
//! the `TypedModule`, and so do the edits that change a finalized module; the serializer declares
//! exactly its SPIR-V capabilities and extensions, and `DeviceProfile::check` compares the Vulkan
//! half with the profile.
//!
//! Conventions:
//!
//! - **Features imply their exposure; extensions are what every device needs.** A requirement
//!   lists the gating *feature* (e.g. `shaderDemoteToHelperInvocation`, which exists through
//!   `VK_EXT_shader_demote_to_helper_invocation` or Vulkan 1.3), and `extensions` lists exactly
//!   the device extensions the module needs on **every** device: the extension of each required
//!   feature that no core version exposes (`VkFeature::exposure`; e.g.
//!   `VK_EXT_fragment_shader_interlock`, `VK_KHR_maintenance8`), added in one closing step, and
//!   the extension of a declaration the registry gates by exposure alone with no feature and no
//!   core version (`VK_EXT_shader_stencil_export`). An extension a core version promoted
//!   (`VK_KHR_shader_terminate_invocation`) depends on the device's version, so it is not listed:
//!   the description lists the feature only where it is exposed (`Device::new` checks it), and
//!   `DeviceDescription::required_enables` names the extension below that version.
//! - **A capability with a version or an extension beside its feature is enabled on the device.**
//!   `StorageImageReadWithoutFormat` and `StorageImageWriteWithoutFormat` are enabled by
//!   `shaderStorageImage{Read,Write}WithoutFormat`, by Vulkan 1.3, or by
//!   `VK_KHR_format_feature_flags2`, under which each format's own bits govern its access. The
//!   module lists the capability alone, and `DeviceProfile::check` and
//!   `DeviceDescription::required_enables` resolve it on the device
//!   (`device::requirements::WITHOUT_FORMAT_CAPABILITIES`).
//! - **Contextual choices are structural.** Where the Vulkan registry offers alternatives for one
//!   capability (buffer vs shared vs image float atomics, 8/16-bit access per storage class), the
//!   walk picks the one the instruction's storage class and type select. A test checks every
//!   derived capability and extension against the generated registry table
//!   (`registry_generated`, emitted by `scripts/regen-vulkan-requirements`) and every serialized family against `spirv-val`, so the hand-written `match` stays honest.
//! - **Arithmetic widths are conservative.** A declared 8-, 16-, or 64-bit scalar type requires
//!   its arithmetic capability (`Int8`, `Int16`, `Float16`, `Int64`, `Float64`) and feature even
//!   when the module only loads and stores such values. Over-requiring can refuse a profile; it
//!   never produces an invalid module. (Relaxing this to the storage-only use sets of
//!   `SPV_KHR_8bit_storage`/`SPV_KHR_16bit_storage` is a recorded follow-up.)
//! - **The subgroup policy is a plan fact.** A pinned subgroup size and full subgroups are stated
//!   outside the SPIR-V (the pipeline's `required_subgroup_size` and stage flags), so derivation
//!   cannot find what they need of the device. The lowering hands the module its plan's
//!   [`SubgroupRequirement`] (`ModuleBuilder::subgroup`), and [`recorded`] adds that
//!   requirement's facts to what [`derive()`] finds: `subgroupSizeControl`, the stage in
//!   `requiredSubgroupSizeStages`, the size range, and `REQUIRE_FULL_SUBGROUPS`. The verifier
//!   accepts exactly those, and only where the module's own requirement justifies them.
//! - Device *format* usages are bind facts: the binder checks each format it binds against the
//!   device, so no module's requirements name one.

use std::collections::BTreeSet;

use crate::spv::Capability as Cap;

use crate::device::profile::Independence;
use crate::device::requirements::{
    ApiVersion, FloatControl, FloatWidth, Requirements, SpirvCapability, SpirvExtension,
    SubgroupOp, VkExtension, VkFeature, VkLimit, VkProperty,
};
use crate::device::subgroup::SubgroupRequirement;
use crate::pipeline::sym::Stage;
use crate::spirv::error::{BResult, BuildError, Dangling, GlobalRule, TypeRule};
use crate::spirv::ids::{FuncId, GlobalId, TypeId, Value};
use crate::spirv::ir::{
    AtomicOp, DerivOp, Dim, ExecMode, ExecModel, GlobalKind, GlslOp, GroupInst, GroupOperation,
    ImageOperands, ImageType, ImageUsage, Inst, IoDir, LayoutRule, RayQueryInst, Scope, ScopeId,
    Semantics, SemanticsId, Shape, StorageClass, Terminator, TexelMemory, Type, ValueOrigin,
};
use crate::spirv::ir::{
    Block, Consts, EntryPoint, Function, GlobalVar, MemAccess, Types, ValueDef,
};

/// `SkipTrianglesKHR | SkipAABBsKHR`.
const SKIP_PRIMITIVES: u64 = 0x100 | 0x200;

/// The independence an entry needs whose one denormal (or rounding) mode names `widths`
/// (RuntimeSpirv VUIDs 06289-06292): none where every width or none is named; 32-bit only where 16
/// and 64 bits agree and 32 bits does not; all where 16 and 64 bits differ.
fn independence_needed(widths: &BTreeSet<u8>) -> Independence {
    let named = |w: u8| widths.contains(&w);
    if named(16) != named(64) {
        Independence::All
    } else if named(32) != named(16) {
        Independence::Only32Bit
    } else {
        Independence::None
    }
}

mod own_pixel;
mod snapshot;
mod writes;

pub use own_pixel::own_pixel_textures;
pub use snapshot::{snapshot_copies_alone, snapshot_textures};
pub use writes::writes_memory;

#[cfg(test)]
mod registry_generated;
#[cfg(test)]
pub(crate) mod tests;

/// The parts of a module that derivation reads, borrowed before the `TypedModule` exists:
/// `ModuleBuilder::finalize` computes the interface and requirements from this view and then
/// constructs the module whole.
#[derive(Clone, Copy)]
pub struct ModuleView<'a> {
    pub(crate) types: &'a Types,
    pub(crate) consts: &'a Consts,
    pub(crate) values: &'a [ValueDef],
    pub(crate) globals: &'a [GlobalVar],
    pub(crate) functions: &'a [Function],
    pub(crate) blocks: &'a [Block],
    pub(crate) entry: &'a EntryPoint,
    /// The enabling capability lowering chose for a builtin with alternatives
    /// ([`builtin_enablers`]).
    pub(crate) enabling: &'a std::collections::BTreeMap<GlobalId, Cap>,
    /// The user varyings decorated `PerPrimitiveEXT`.
    pub(crate) per_primitive: &'a BTreeSet<GlobalId>,
}

/// The enabling capabilities of builtin `b` in direction `dir` that lowering chooses among by
/// the profile, each with the Vulkan feature that enables it. The SPIR-V registry lists each as an
/// enabling capability of the builtin; a module that records no choice for a variable
/// (`ModuleBuilder::enable_builtin`) declares the first. Empty for every other builtin, whose
/// capability derivation picks structurally.
///
/// - A fragment `PrimitiveId` input: `Geometry` (`geometryShader`) or `Tessellation`
///   (`tessellationShader`).
/// - A fragment `Layer` input: `Geometry` or `ShaderLayer` (`shaderOutputLayer`).
/// - A fragment `ViewportIndex` input: `MultiViewport` (`multiViewport`) or
///   `ShaderViewportIndex` (`shaderOutputViewportIndex`).
pub fn builtin_enablers(b: crate::spv::BuiltIn, dir: IoDir) -> &'static [(Cap, VkFeature)] {
    use crate::spv::BuiltIn as B;
    match (b, dir) {
        (B::PrimitiveId, IoDir::Input) => &[
            (Cap::Geometry, VkFeature::GeometryShader),
            (Cap::Tessellation, VkFeature::TessellationShader),
        ],
        (B::Layer, IoDir::Input) => &[
            (Cap::Geometry, VkFeature::GeometryShader),
            (Cap::ShaderLayer, VkFeature::ShaderOutputLayer),
        ],
        (B::ViewportIndex, IoDir::Input) => &[
            (Cap::MultiViewport, VkFeature::MultiViewport),
            (
                Cap::ShaderViewportIndex,
                VkFeature::ShaderOutputViewportIndex,
            ),
        ],
        _ => &[],
    }
}

/// Whether each function is in the entry point's static call tree.
pub(crate) fn reachable_functions(m: &ModuleView<'_>) -> BResult<Vec<bool>> {
    let mut seen = vec![false; m.functions.len()];
    let mut stack = vec![m.entry.func];
    while let Some(f) = stack.pop() {
        let slot = seen
            .get_mut(f.index())
            .ok_or(BuildError::Dangling(Dangling::Func(f)))?;
        if *slot {
            continue;
        }
        *slot = true;
        for b in &function(m, f)?.blocks {
            for n in &block(m, *b)?.insts {
                if let Inst::Call { func, .. } = &n.inst {
                    stack.push(*func);
                }
            }
        }
    }
    Ok(seen)
}

/// The entry point's interface: every module-scope variable referenced by a function of the
/// entry's static call tree, in declaration (global id) order, each once (SPIR-V 1.4+).
pub fn interface(m: &ModuleView<'_>) -> BResult<Vec<GlobalId>> {
    let reachable = reachable_functions(m)?;
    let mut used = vec![false; m.globals.len()];
    for (fi, f) in m.functions.iter().enumerate() {
        if !reachable.get(fi).copied().unwrap_or(false) {
            continue;
        }
        let mut mark = |v: Value| {
            if let Some(d) = m.values.get(v.index()) {
                if let ValueOrigin::Global(g) = d.origin {
                    if let Some(u) = used.get_mut(g.index()) {
                        *u = true;
                    }
                }
            }
        };
        for b in &f.blocks {
            let b = block(m, *b)?;
            for p in &b.phis {
                for (v, _) in &p.incoming {
                    mark(*v);
                }
            }
            for n in &b.insts {
                n.inst.for_each_value(&mut mark);
            }
            b.term.for_each_value(&mut mark);
        }
    }
    let interface: Vec<GlobalId> = used
        .iter()
        .enumerate()
        .filter(|(_, u)| **u)
        .map(|(i, _)| GlobalId::new(i as u32))
        .collect();
    // Explicitly laid-out `Workgroup` memory is all or nothing per entry point, and an entry has
    // at most one payload (`TaskPayloadWorkgroupEXT`).
    let (mut blocks, mut implicit, mut payloads) = (false, false, 0usize);
    for g in &interface {
        let gv = m
            .globals
            .get(g.index())
            .ok_or(BuildError::Dangling(Dangling::Global(*g)))?;
        if gv.class == StorageClass::Workgroup {
            if m.types.facts(gv.pointee)?.block {
                blocks = true;
            } else {
                implicit = true;
            }
        }
        payloads += usize::from(gv.class == StorageClass::TaskPayloadWorkgroupEXT);
    }
    if blocks && implicit {
        return Err(BuildError::Global(GlobalRule::MixedWorkgroupLayout));
    }
    if payloads > 1 {
        return Err(BuildError::Global(GlobalRule::Payload));
    }
    Ok(interface)
}

/// The Metal stage of an execution model: both tessellation stages and the synthesized geometry
/// stage belong to Metal's vertex stage.
pub(crate) fn metal_stage(model: ExecModel) -> Stage {
    match model {
        ExecModel::Vertex
        | ExecModel::TessellationEvaluation
        | ExecModel::TessellationControl
        | ExecModel::Geometry => Stage::Vertex,
        ExecModel::Fragment => Stage::Fragment,
        ExecModel::GLCompute => Stage::Kernel,
        ExecModel::TaskEXT => Stage::Object,
        ExecModel::MeshEXT => Stage::Mesh,
    }
}

/// What a module records it needs from the device: what it contains ([`derive()`]) and what its
/// plan's subgroup requirement needs beyond it ([`SubgroupRequirement::add_device_facts`]).
pub fn recorded(m: &ModuleView<'_>, subgroup: &SubgroupRequirement) -> BResult<Requirements> {
    let mut r = derive(m)?;
    subgroup.add_device_facts(metal_stage(m.entry.model), &mut r);
    Ok(r)
}

/// Derive everything `m` needs from the device.
pub fn derive(m: &ModuleView<'_>) -> BResult<Requirements> {
    let mut d = Deriver {
        m,
        r: Requirements {
            api_version: Some(ApiVersion::V1_2),
            ..Requirements::default()
        },
        model: m.entry.model,
        stage: metal_stage(m.entry.model),
        vulkan_memory_model: false,
        device_scope: false,
    };
    d.cap(Cap::Shader);
    if m.entry.model.is_tessellation() {
        d.cap(Cap::Tessellation);
        d.feat(VkFeature::TessellationShader);
    }
    if m.entry.model == ExecModel::Geometry {
        d.cap(Cap::Geometry);
        d.feat(VkFeature::GeometryShader);
    }
    // A task or mesh entry, and a fragment entry reading a mesh entry's primitive outputs
    // (`PerPrimitiveEXT`), declare mesh shading.
    let stage_feature = match m.entry.model {
        ExecModel::TaskEXT => Some(VkFeature::TaskShader),
        ExecModel::MeshEXT => Some(VkFeature::MeshShader),
        _ if !m.per_primitive.is_empty() => Some(VkFeature::MeshShader),
        _ => None,
    };
    if let Some(feature) = stage_feature {
        d.cap(Cap::MeshShadingEXT);
        d.ext(SpirvExtension::ExtMeshShader);
        d.feat(feature);
    }
    d.types()?;
    d.globals()?;
    d.modes()?;
    for b in m.blocks {
        for n in &b.insts {
            d.inst(&n.inst)?;
        }
        if b.term == Terminator::TerminateInvocation {
            d.ext(SpirvExtension::KhrTerminateInvocation);
            d.feat(VkFeature::ShaderTerminateInvocation);
        }
        // `OpEmitMeshTasksEXT` is the task entry's: its `MeshShadingEXT` covers it.
    }
    if d.vulkan_memory_model {
        d.cap(Cap::VulkanMemoryModel);
        d.feat(VkFeature::VulkanMemoryModel);
        if d.device_scope {
            d.cap(Cap::VulkanMemoryModelDeviceScope);
            d.feat(VkFeature::VulkanMemoryModelDeviceScope);
        }
    }
    close_extensions(&mut d.r);
    Ok(d.r)
}

/// Adds the extension of every required feature that no core version exposes: such a feature
/// is enabled only with its extension on every device.
pub(crate) fn close_extensions(r: &mut Requirements) {
    for &f in &r.features {
        if let Some(e) = f.exposure().only_extension() {
            r.extensions.insert(e);
        }
    }
}

/// Whether a module with these requirements uses the Vulkan memory model: derived from the memory
/// semantics and scopes it contains.
pub fn uses_vulkan_memory_model(r: &Requirements) -> bool {
    r.spirv_capabilities
        .contains(&SpirvCapability::of(Cap::VulkanMemoryModel))
}

/// Whether a module with these requirements uses `PhysicalStorageBuffer` pointers (addressing
/// model `PhysicalStorageBuffer64`).
pub fn uses_physical_storage_buffer(r: &Requirements) -> bool {
    r.spirv_capabilities
        .contains(&SpirvCapability::of(Cap::PhysicalStorageBufferAddresses))
}

fn function<'a>(m: &ModuleView<'a>, f: FuncId) -> BResult<&'a Function> {
    m.functions
        .get(f.index())
        .ok_or(BuildError::Dangling(Dangling::Func(f)))
}

fn block<'a>(m: &ModuleView<'a>, b: crate::spirv::ids::BlockId) -> BResult<&'a Block> {
    m.blocks
        .get(b.index())
        .ok_or(BuildError::Dangling(Dangling::Block(b)))
}

/// The 8- and 16-bit scalars inside a type (not through pointers).
#[derive(Clone, Copy, Debug, Default)]
struct Narrow {
    w8: bool,
    w16: bool,
}

struct Deriver<'m> {
    m: &'m ModuleView<'m>,
    r: Requirements,
    model: ExecModel,
    stage: Stage,
    vulkan_memory_model: bool,
    device_scope: bool,
}

impl Deriver<'_> {
    fn cap(&mut self, c: Cap) {
        self.r.spirv_capabilities.insert(SpirvCapability::of(c));
    }

    fn feat(&mut self, f: VkFeature) {
        self.r.features.insert(f);
    }

    fn ext(&mut self, e: SpirvExtension) {
        self.r.spirv_extensions.insert(e);
    }

    fn prop(&mut self, p: VkProperty) {
        self.r.properties.insert(p);
    }

    /// A `Max*` limit must be at least `v`.
    fn at_least(&mut self, l: VkLimit, v: u64) {
        let e = self.r.limits.entry(l).or_insert(0);
        *e = (*e).max(v);
    }

    fn ray_query(&mut self) {
        self.cap(Cap::RayQueryKHR);
        self.ext(SpirvExtension::KhrRayQuery);
        self.feat(VkFeature::RayQuery);
    }

    /// The bits of `v` when it is a plain scalar constant.
    fn const_bits(&self, v: Value) -> BResult<Option<u64>> {
        let def = self
            .m
            .values
            .get(v.index())
            .ok_or(BuildError::Dangling(Dangling::Value(v)))?;
        Ok(match def.origin {
            ValueOrigin::Const(c) => self.m.consts.scalar_bits(c).map(|b| b.0),
            _ => None,
        })
    }

    fn ty(&self, t: TypeId) -> BResult<&Type> {
        self.m.types.get(t)
    }

    fn value_ty(&self, v: Value) -> BResult<TypeId> {
        self.m
            .values
            .get(v.index())
            .map(|d| d.ty)
            .ok_or(BuildError::Dangling(Dangling::Value(v)))
    }

    /// The storage class and pointee of pointer value `v`.
    fn pointer_of(&self, v: Value) -> BResult<(StorageClass, TypeId)> {
        match self.ty(self.value_ty(v)?)? {
            Type::Pointer { class, pointee } => Ok((*class, *pointee)),
            _ => Err(BuildError::Dangling(Dangling::Value(v))),
        }
    }

    /// The scalar component type of a scalar or vector type.
    fn scalar_of(&self, t: TypeId) -> BResult<&Type> {
        match self.ty(t)? {
            Type::Vector { elem, .. } => self.ty(*elem),
            other => Ok(other),
        }
    }

    /// The 8- and 16-bit scalars inside `t`, each type read once: members share types, so a walk
    /// of every path would be exponential in a struct DAG's nesting.
    fn narrow(&self, t: TypeId) -> BResult<Narrow> {
        let mut out = Narrow::default();
        let mut seen = BTreeSet::new();
        let mut stack = vec![t];
        while let Some(t) = stack.pop() {
            if !seen.insert(t) {
                continue;
            }
            match self.ty(t)? {
                Type::Int { width: 8, .. } => out.w8 = true,
                Type::Int { width: 16, .. } | Type::Float { width: 16 } => out.w16 = true,
                Type::Vector { elem, .. } => stack.push(*elem),
                Type::Matrix { col, .. } => stack.push(*col),
                Type::Array { elem, .. } | Type::RuntimeArray { elem, .. } => stack.push(*elem),
                Type::Struct { members, .. } => stack.extend(members.iter().copied()),
                _ => {}
            }
        }
        Ok(out)
    }

    /// The feature that stores and atomics through descriptors need in this stage: Vulkan's
    /// `vertexPipelineStoresAndAtomics` in the vertex pipeline, `fragmentStoresAndAtomics` in a
    /// fragment shader.
    fn stage_stores(&mut self) {
        match self.model {
            ExecModel::Vertex
            | ExecModel::TessellationEvaluation
            | ExecModel::TessellationControl
            | ExecModel::Geometry => self.feat(VkFeature::VertexPipelineStoresAndAtomics),
            ExecModel::Fragment => self.feat(VkFeature::FragmentStoresAndAtomics),
            // Vulkan's stage store features name no task or mesh stage.
            ExecModel::GLCompute | ExecModel::TaskEXT | ExecModel::MeshEXT => {}
        }
    }

    // ---- Types ----

    fn types(&mut self) -> BResult<()> {
        let m = self.m;
        for (_, t) in m.types.iter() {
            match t {
                Type::Int { width: 8, .. } => {
                    self.cap(Cap::Int8);
                    self.feat(VkFeature::ShaderInt8);
                }
                Type::Int { width: 16, .. } => {
                    self.cap(Cap::Int16);
                    self.feat(VkFeature::ShaderInt16);
                }
                Type::Int { width: 64, .. } => {
                    self.cap(Cap::Int64);
                    self.feat(VkFeature::ShaderInt64);
                }
                Type::Float { width: 16 } => {
                    self.cap(Cap::Float16);
                    self.feat(VkFeature::ShaderFloat16);
                }
                Type::Float { width: 64 } => {
                    self.cap(Cap::Float64);
                    self.feat(VkFeature::ShaderFloat64);
                }
                Type::RuntimeArray { stride: None, .. } => {
                    self.cap(Cap::RuntimeDescriptorArray);
                    self.feat(VkFeature::RuntimeDescriptorArray);
                }
                Type::Pointer { class, pointee } => self.pointer_type(*class, *pointee)?,
                Type::Image(img) => self.image_type(img)?,
                // Both ray query types are declared under `RayQueryKHR`.
                Type::AccelerationStructure | Type::RayQuery => self.ray_query(),
                _ => {}
            }
        }
        Ok(())
    }

    fn pointer_type(&mut self, class: StorageClass, pointee: TypeId) -> BResult<()> {
        if class == StorageClass::PhysicalStorageBuffer {
            self.cap(Cap::PhysicalStorageBufferAddresses);
            self.feat(VkFeature::BufferDeviceAddress);
        }
        let n = self.narrow(pointee)?;
        let (c8, f8, c16, f16) = match class {
            StorageClass::StorageBuffer | StorageClass::PhysicalStorageBuffer => (
                Cap::StorageBuffer8BitAccess,
                VkFeature::StorageBuffer8BitAccess,
                Cap::StorageBuffer16BitAccess,
                VkFeature::StorageBuffer16BitAccess,
            ),
            StorageClass::Uniform => (
                Cap::UniformAndStorageBuffer8BitAccess,
                VkFeature::UniformAndStorageBuffer8BitAccess,
                Cap::UniformAndStorageBuffer16BitAccess,
                VkFeature::UniformAndStorageBuffer16BitAccess,
            ),
            StorageClass::PushConstant => (
                Cap::StoragePushConstant8,
                VkFeature::StoragePushConstant8,
                Cap::StoragePushConstant16,
                VkFeature::StoragePushConstant16,
            ),
            StorageClass::Input | StorageClass::Output => {
                if n.w8 {
                    // Vulkan has no 8-bit interface storage.
                    return Err(BuildError::Type(TypeRule::PointeeForClass(class)));
                }
                if n.w16 {
                    self.cap(Cap::StorageInputOutput16);
                    self.feat(VkFeature::StorageInputOutput16);
                }
                return Ok(());
            }
            // Workgroup narrow access is a property of explicitly laid-out `Workgroup` blocks
            // (see `globals`); other classes need only the arithmetic capability.
            _ => return self.layout(class, pointee),
        };
        if n.w8 {
            self.cap(c8);
            self.feat(f8);
        }
        if n.w16 {
            self.cap(c16);
            self.feat(f16);
        }
        self.layout(class, pointee)
    }

    /// The layout feature an explicitly laid-out pointee needs in `class` (the P8a layout rules:
    /// scalar layout needs `scalarBlockLayout`, std430 in `Uniform` needs
    /// `uniformBufferStandardLayout`, scalar `Workgroup` blocks need
    /// `workgroupMemoryExplicitLayoutScalarBlockLayout`).
    fn layout(&mut self, class: StorageClass, pointee: TypeId) -> BResult<()> {
        // A descriptor array's layout is its element's.
        let laid = match self.ty(pointee)? {
            Type::Array {
                elem, stride: None, ..
            }
            | Type::RuntimeArray { elem, stride: None } => *elem,
            _ => pointee,
        };
        let f = *self.m.types.facts(laid)?;
        if f.shape != Shape::Explicit {
            return Ok(());
        }
        match (class, f.max_rule()) {
            (
                StorageClass::StorageBuffer
                | StorageClass::PhysicalStorageBuffer
                | StorageClass::PushConstant
                | StorageClass::Uniform,
                LayoutRule::Scalar,
            ) => self.feat(VkFeature::ScalarBlockLayout),
            (StorageClass::Uniform, LayoutRule::Std430) => {
                self.feat(VkFeature::UniformBufferStandardLayout)
            }
            (StorageClass::Workgroup, LayoutRule::Scalar) => {
                self.feat(VkFeature::WorkgroupMemoryExplicitLayoutScalarBlockLayout)
            }
            _ => {}
        }
        Ok(())
    }

    fn image_type(&mut self, img: &ImageType) -> BResult<()> {
        if matches!(self.ty(img.sampled_type)?, Type::Int { width: 64, .. }) {
            self.cap(Cap::Int64ImageEXT);
            self.ext(SpirvExtension::ExtShaderImageInt64);
            self.feat(VkFeature::ShaderImageInt64Atomics);
        }
        let storage = img.usage == ImageUsage::Storage;
        match img.dim {
            Dim::D1 => self.cap(if storage {
                Cap::Image1D
            } else {
                Cap::Sampled1D
            }),
            Dim::Buffer => self.cap(if storage {
                Cap::ImageBuffer
            } else {
                Cap::SampledBuffer
            }),
            Dim::Cube if img.arrayed => {
                self.cap(if storage {
                    Cap::ImageCubeArray
                } else {
                    Cap::SampledCubeArray
                });
                self.feat(VkFeature::ImageCubeArray);
            }
            Dim::SubpassData => self.cap(Cap::InputAttachment),
            Dim::D2 | Dim::D3 | Dim::Cube => {}
        }
        if storage && img.multisampled && img.dim != Dim::SubpassData {
            self.cap(Cap::StorageImageMultisample);
            self.feat(VkFeature::ShaderStorageImageMultisample);
            if img.arrayed {
                self.cap(Cap::ImageMSArray);
            }
        }
        if storage && extended_format(img.format) {
            self.cap(Cap::StorageImageExtendedFormats);
            self.feat(VkFeature::ShaderStorageImageExtendedFormats);
        }
        Ok(())
    }

    // ---- Module-scope variables ----

    fn globals(&mut self) -> BResult<()> {
        let m = self.m;
        for (i, g) in m.globals.iter().enumerate() {
            match &g.kind {
                GlobalKind::Plain if g.class == StorageClass::Workgroup => {
                    if g.init.is_some() {
                        self.feat(VkFeature::ShaderZeroInitializeWorkgroupMemory);
                    }
                    if m.types.facts(g.pointee)?.block {
                        self.cap(Cap::WorkgroupMemoryExplicitLayoutKHR);
                        self.ext(SpirvExtension::KhrWorkgroupMemoryExplicitLayout);
                        self.feat(VkFeature::WorkgroupMemoryExplicitLayout);
                        let n = self.narrow(g.pointee)?;
                        if n.w8 {
                            self.cap(Cap::WorkgroupMemoryExplicitLayout8BitAccessKHR);
                            self.feat(VkFeature::WorkgroupMemoryExplicitLayout8BitAccess);
                        }
                        if n.w16 {
                            self.cap(Cap::WorkgroupMemoryExplicitLayout16BitAccessKHR);
                            self.feat(VkFeature::WorkgroupMemoryExplicitLayout16BitAccess);
                        }
                    }
                }
                GlobalKind::BuiltIn { builtin, dir } => {
                    let id =
                        GlobalId::new(u32::try_from(i).map_err(|_| BuildError::IdSpaceExhausted)?);
                    self.builtin(id, *builtin, *dir, g.pointee)?
                }
                // The `Sample` decoration needs `SampleRateShading`, which Vulkan gates on
                // `sampleRateShading`, with or without a `SampleId` beside it.
                GlobalKind::Io { deco, .. }
                    if matches!(deco.sampling, crate::spirv::ir::Sampling::Sample) =>
                {
                    self.cap(Cap::SampleRateShading);
                    self.feat(VkFeature::SampleRateShading);
                }
                GlobalKind::PushBlock => {
                    if let Some(size) = m.types.facts(g.pointee)?.size() {
                        self.at_least(VkLimit::MaxPushConstantsSize, u64::from(size));
                    }
                }
                // An acceleration structure descriptor is bound to a structure the device
                // built, which needs `accelerationStructure` and the device addresses its build
                // reads (`bufferDeviceAddress`).
                GlobalKind::Resource(_)
                    if matches!(m.types.get(g.pointee)?, Type::AccelerationStructure) =>
                {
                    self.feat(VkFeature::AccelerationStructure);
                    self.feat(VkFeature::BufferDeviceAddress);
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn builtin(
        &mut self,
        g: GlobalId,
        b: crate::spv::BuiltIn,
        dir: IoDir,
        ty: TypeId,
    ) -> BResult<()> {
        use crate::spv::BuiltIn as B;
        // A mesh entry's outputs are arrayed per vertex or per primitive, and a geometry or
        // evaluation entry's per-vertex inputs and a control entry's per-vertex inputs and
        // outputs per vertex: a builtin's own array, a clip or cull distance's, is that array's
        // element.
        let arrays_vertices = match (self.model, dir) {
            (ExecModel::Geometry | ExecModel::TessellationEvaluation, IoDir::Input) => true,
            (ExecModel::TessellationControl, _) => true,
            _ => false,
        };
        let arrayed = (self.model == ExecModel::MeshEXT && dir == IoDir::Output)
            || (arrays_vertices && crate::spirv::builder::geometry_per_vertex(b));
        let array_len = |d: &Self| -> BResult<Option<u32>> {
            let own = match (arrayed, d.ty(ty)?) {
                (false, _) => ty,
                (true, Type::Array { elem, .. }) => *elem,
                (true, _) => return Ok(None),
            };
            match d.ty(own)? {
                Type::Array { len, .. } => d.m.consts.array_len(*len, d.m.types),
                _ => Ok(None),
            }
        };
        // A tessellation stage's `PrimitiveId` is enabled by `Tessellation`, and a geometry stage's
        // by `Geometry`, which the entry declared.
        if b == B::PrimitiveId
            && (self.model.is_tessellation() || self.model == ExecModel::Geometry)
        {
            return Ok(());
        }
        // A geometry stage's `Layer` output is enabled by `Geometry`, which the entry declared, and
        // its `ViewportIndex` output by `MultiViewport` (`multiViewport`).
        if self.model == ExecModel::Geometry && dir == IoDir::Output {
            match b {
                B::Layer => return Ok(()),
                B::ViewportIndex => {
                    self.cap(Cap::MultiViewport);
                    self.feat(VkFeature::MultiViewport);
                    return Ok(());
                }
                _ => {}
            }
        }
        // A task or mesh entry's `DrawIndex` is enabled by `MeshShadingEXT` (as the grammar lists
        // it), which the entry declared: it needs no `DrawParameters`.
        if b == B::DrawIndex && matches!(self.model, ExecModel::TaskEXT | ExecModel::MeshEXT) {
            return Ok(());
        }
        // A geometry stage's point size, input or output, is `GeometryPointSize`'s.
        if b == B::PointSize && self.model == ExecModel::Geometry {
            self.cap(Cap::GeometryPointSize);
            self.feat(VkFeature::ShaderTessellationAndGeometryPointSize);
            return Ok(());
        }
        // A mesh entry's per-primitive outputs (`PrimitiveId`, `Layer`, `ViewportIndex`) and its
        // own builtins are enabled by `MeshShadingEXT`, which the entry declared. Vulkan still asks
        // `shaderOutputLayer` of a mesh entry's `Layer` and `shaderOutputViewportIndex` of its
        // `ViewportIndex` (VUID-Layer-Layer-04273, VUID-ViewportIndex-ViewportIndex-04405).
        if self.model == ExecModel::MeshEXT
            && dir == IoDir::Output
            && matches!(
                b,
                B::PrimitiveId
                    | B::Layer
                    | B::ViewportIndex
                    | B::CullPrimitiveEXT
                    | B::PrimitivePointIndicesEXT
                    | B::PrimitiveLineIndicesEXT
                    | B::PrimitiveTriangleIndicesEXT
            )
        {
            match b {
                B::Layer => self.feat(VkFeature::ShaderOutputLayer),
                B::ViewportIndex => self.feat(VkFeature::ShaderOutputViewportIndex),
                _ => {}
            }
            return Ok(());
        }
        // A builtin with alternatives declares the one lowering chose, or the first.
        let enablers = builtin_enablers(b, dir);
        if let Some(&(default, _)) = enablers.first() {
            let cap = self.m.enabling.get(&g).copied().unwrap_or(default);
            let (cap, feat) = enablers
                .iter()
                .copied()
                .find(|(c, _)| *c == cap)
                .ok_or(BuildError::Global(GlobalRule::BuiltInEnabler(Some(b))))?;
            self.cap(cap);
            self.feat(feat);
            return Ok(());
        }
        match b {
            B::SampleId | B::SamplePosition => {
                self.cap(Cap::SampleRateShading);
                self.feat(VkFeature::SampleRateShading);
            }
            B::BaryCoordKHR | B::BaryCoordNoPerspKHR => {
                self.cap(Cap::FragmentBarycentricKHR);
                self.ext(SpirvExtension::KhrFragmentShaderBarycentric);
                self.feat(VkFeature::FragmentShaderBarycentric);
            }
            B::Layer => {
                self.cap(Cap::ShaderLayer);
                self.feat(VkFeature::ShaderOutputLayer);
            }
            B::ViewportIndex => {
                self.cap(Cap::ShaderViewportIndex);
                self.feat(VkFeature::ShaderOutputViewportIndex);
            }
            B::ClipDistance => {
                self.cap(Cap::ClipDistance);
                self.feat(VkFeature::ShaderClipDistance);
                if let Some(n) = array_len(self)? {
                    self.at_least(VkLimit::MaxClipDistances, u64::from(n));
                }
            }
            B::CullDistance => {
                self.cap(Cap::CullDistance);
                self.feat(VkFeature::ShaderCullDistance);
                if let Some(n) = array_len(self)? {
                    self.at_least(VkLimit::MaxCullDistances, u64::from(n));
                }
            }
            B::FragStencilRefEXT => {
                self.cap(Cap::StencilExportEXT);
                self.ext(SpirvExtension::ExtShaderStencilExport);
                self.r
                    .extensions
                    .insert(VkExtension::ExtShaderStencilExport);
            }
            B::BaseVertex | B::BaseInstance | B::DrawIndex => {
                self.cap(Cap::DrawParameters);
                self.feat(VkFeature::ShaderDrawParameters);
            }
            B::ViewIndex => {
                self.cap(Cap::MultiView);
                self.feat(VkFeature::Multiview);
            }
            B::SubgroupSize | B::SubgroupLocalInvocationId | B::NumSubgroups | B::SubgroupId => {
                self.cap(Cap::GroupNonUniform);
                self.subgroup(SubgroupOp::Basic);
            }
            _ => {}
        }
        Ok(())
    }

    // ---- Execution modes ----

    fn modes(&mut self) -> BResult<()> {
        let m = self.m;
        for mode in &m.entry.modes {
            match *mode {
                ExecMode::SignedZeroInfNanPreserve { width } => self.float_control(
                    Cap::SignedZeroInfNanPreserve,
                    FloatControl::SignedZeroInfNanPreserve,
                    width,
                ),
                ExecMode::DenormPreserve { width } => {
                    self.float_control(Cap::DenormPreserve, FloatControl::DenormPreserve, width)
                }
                ExecMode::RoundingModeRte { width } => {
                    self.float_control(Cap::RoundingModeRTE, FloatControl::RoundingModeRte, width)
                }
                ExecMode::PixelInterlockOrdered => {
                    self.cap(Cap::FragmentShaderPixelInterlockEXT);
                    self.ext(SpirvExtension::ExtFragmentShaderInterlock);
                    self.feat(VkFeature::FragmentShaderPixelInterlock);
                }
                ExecMode::SampleInterlockOrdered => {
                    self.cap(Cap::FragmentShaderSampleInterlockEXT);
                    self.ext(SpirvExtension::ExtFragmentShaderInterlock);
                    self.feat(VkFeature::FragmentShaderSampleInterlock);
                }
                ExecMode::StencilRefReplacing => {
                    self.cap(Cap::StencilExportEXT);
                    self.ext(SpirvExtension::ExtShaderStencilExport);
                    self.r
                        .extensions
                        .insert(VkExtension::ExtShaderStencilExport);
                }
                ExecMode::DerivativeGroupLinear => {
                    self.cap(Cap::ComputeDerivativeGroupLinearKHR);
                    self.ext(SpirvExtension::KhrComputeShaderDerivatives);
                    self.feat(VkFeature::ComputeDerivativeGroupLinear);
                }
                ExecMode::SubgroupUniformControlFlow => {
                    self.ext(SpirvExtension::KhrSubgroupUniformControlFlow);
                    self.feat(VkFeature::ShaderSubgroupUniformControlFlow);
                }
                ExecMode::MaximallyReconverges => {
                    self.ext(SpirvExtension::KhrMaximalReconvergence);
                    self.feat(VkFeature::ShaderMaximalReconvergence);
                }
                ExecMode::RequireFullQuads => {
                    self.cap(Cap::QuadControlKHR);
                    self.ext(SpirvExtension::KhrQuadControl);
                    self.feat(VkFeature::ShaderQuadControl);
                }
                ExecMode::PostDepthCoverage => {
                    self.cap(Cap::SampleMaskPostDepthCoverage);
                    self.ext(SpirvExtension::KhrPostDepthCoverage);
                    self.r.extensions.insert(VkExtension::ExtPostDepthCoverage);
                }
                // A task or mesh entry's local size is bounded by the mesh shader limits, which no
                // profile records yet: the stage's feature, which none enables, refuses it first.
                ExecMode::LocalSize { x, y, z } if self.model == ExecModel::GLCompute => {
                    for (axis, v) in [x, y, z].into_iter().enumerate() {
                        self.at_least(VkLimit::MaxComputeWorkGroupSize(axis as u8), u64::from(v));
                    }
                    let total = u64::from(x) * u64::from(y) * u64::from(z);
                    self.at_least(VkLimit::MaxComputeWorkGroupInvocations, total);
                }
                ExecMode::LocalSize { .. } => {}
                ExecMode::OriginUpperLeft
                | ExecMode::EarlyFragmentTests
                | ExecMode::DepthReplacing
                | ExecMode::DepthGreater
                | ExecMode::DepthLess => {}
                // The entry's `Tessellation` capability covers the tessellation modes; an output
                // patch size of at most 32 is within every device's `maxTessellationPatchSize`.
                ExecMode::Triangles
                | ExecMode::Quads
                | ExecMode::SpacingEqual
                | ExecMode::SpacingFractionalEven
                | ExecMode::SpacingFractionalOdd
                | ExecMode::VertexOrderCw
                | ExecMode::VertexOrderCcw
                | ExecMode::OutputVertices { .. } => {}
                // The entry's `MeshShadingEXT` covers a mesh entry's modes; its counts are bounded
                // by the mesh shader limits, as its local size is.
                ExecMode::OutputPrimitives { .. }
                | ExecMode::OutputPoints
                | ExecMode::OutputLines
                | ExecMode::OutputTriangles => {}
                // The entry's `Geometry` covers a geometry entry's modes. The synthesized stage
                // emits at most 3 vertices in one invocation, within every device's
                // `maxGeometryOutputVertices` (at least 256) and `maxGeometryShaderInvocations`
                // (at least 32) where `geometryShader` is.
                ExecMode::InputPoints
                | ExecMode::InputLines
                | ExecMode::OutputLineStrip
                | ExecMode::OutputTriangleStrip
                | ExecMode::Invocations { .. } => {}
            }
        }
        // The widths the denormal and rounding modes name: a width without one keeps the
        // device's default, which is another mode for the independence rules.
        let widths = |pick: fn(&ExecMode) -> Option<u8>| -> BTreeSet<u8> {
            m.entry.modes.iter().filter_map(pick).collect()
        };
        self.r.denorm_independence = independence_needed(&widths(|m| match m {
            ExecMode::DenormPreserve { width } => Some(*width),
            _ => None,
        }));
        self.r.rounding_independence = independence_needed(&widths(|m| match m {
            ExecMode::RoundingModeRte { width } => Some(*width),
            _ => None,
        }));
        Ok(())
    }

    fn float_control(&mut self, c: Cap, kind: FloatControl, width: u8) {
        self.cap(c);
        let w = match width {
            16 => FloatWidth::F16,
            32 => FloatWidth::F32,
            _ => FloatWidth::F64,
        };
        self.prop(VkProperty::FloatControl(kind, w));
    }

    // ---- Instructions ----

    fn subgroup(&mut self, op: SubgroupOp) {
        self.prop(VkProperty::SubgroupStage(self.stage));
        self.prop(VkProperty::SubgroupOperation(SubgroupOp::Basic));
        self.prop(VkProperty::SubgroupOperation(op));
    }

    fn scope(&mut self, s: ScopeId) {
        if s.scope == Scope::Device {
            self.device_scope = true;
        }
    }

    fn semantics(&mut self, s: SemanticsId) {
        let Semantics {
            make_available,
            make_visible,
            volatile,
            ..
        } = s.sem;
        if make_available || make_visible || volatile {
            self.vulkan_memory_model = true;
        }
    }

    /// A load's or store's memory operands: `NonPrivatePointer` and `MakePointer*` need the
    /// Vulkan memory model, and their scopes count like an atomic's.
    fn mem_access(&mut self, a: &MemAccess) {
        if a.vulkan_memory_model() {
            self.vulkan_memory_model = true;
        }
        for s in [a.make_available, a.make_visible].into_iter().flatten() {
            self.scope(s);
        }
    }

    fn inst(&mut self, inst: &Inst) -> BResult<()> {
        if let Inst::Load { access, .. } | Inst::Store { access, .. } = inst {
            self.mem_access(access);
        }
        match inst {
            Inst::Store { ptr, .. } => {
                if matches!(
                    self.pointer_of(*ptr)?.0,
                    StorageClass::StorageBuffer | StorageClass::PhysicalStorageBuffer
                ) {
                    self.stage_stores();
                }
            }
            Inst::AccessChain {
                base,
                indices,
                non_uniform,
            } => self.access_chain(*base, indices, *non_uniform)?,
            Inst::AtomicLoad { ptr, scope, sem } => {
                self.scope(*scope);
                self.semantics(*sem);
                self.atomic(*ptr, None)?;
            }
            Inst::AtomicStore {
                ptr, scope, sem, ..
            } => {
                self.scope(*scope);
                self.semantics(*sem);
                self.atomic(*ptr, None)?;
            }
            Inst::Atomic {
                op,
                ptr,
                scope,
                sem,
                ..
            } => {
                self.scope(*scope);
                self.semantics(*sem);
                self.atomic(*ptr, Some(*op))?;
            }
            Inst::AtomicCompareExchange {
                ptr,
                scope,
                equal,
                unequal,
                ..
            } => {
                self.scope(*scope);
                self.semantics(*equal);
                self.semantics(*unequal);
                self.atomic(*ptr, None)?;
            }
            Inst::ControlBarrier { exec, mem, sem } => {
                self.scope(*exec);
                self.scope(*mem);
                self.semantics(*sem);
            }
            Inst::MemoryBarrier { mem, sem } => {
                self.scope(*mem);
                self.semantics(*sem);
            }
            Inst::ImageSample { ops, form, .. } => {
                self.image_operands(ops, false);
                if form.sparse {
                    self.sparse_residency();
                }
            }
            Inst::ImageFetch { ops, sparse, .. } => {
                self.image_operands(ops, false);
                if *sparse {
                    self.sparse_residency();
                }
            }
            Inst::ImageDrefGather { ops, sparse, .. } | Inst::ImageGather { ops, sparse, .. } => {
                self.image_operands(ops, true);
                if *sparse {
                    self.sparse_residency();
                }
            }
            Inst::ImageSparseTexelsResident(_) => self.sparse_residency(),
            Inst::ImageRead {
                image,
                ops,
                sparse,
                mem,
                ..
            } => {
                self.image_operands(ops, false);
                self.texel_memory(*mem);
                if *sparse {
                    self.sparse_residency();
                }
                let img = self.image_of(*image)?;
                // Enabled on the device (`DeviceProfile::check`), not by a feature listed here.
                if img.format == crate::spv::ImageFormat::Unknown && img.dim != Dim::SubpassData {
                    self.cap(Cap::StorageImageReadWithoutFormat);
                }
            }
            Inst::ImageWrite {
                image, ops, mem, ..
            } => {
                self.image_operands(ops, false);
                self.texel_memory(*mem);
                self.stage_stores();
                if self.image_of(*image)?.format == crate::spv::ImageFormat::Unknown {
                    self.cap(Cap::StorageImageWriteWithoutFormat);
                }
            }
            Inst::ImageQuery { .. } | Inst::ImageQueryLod { .. } => self.cap(Cap::ImageQuery),
            Inst::Derivative { op, .. } => {
                if !matches!(op, DerivOp::DPdx | DerivOp::DPdy | DerivOp::Fwidth) {
                    self.cap(Cap::DerivativeControl);
                }
            }
            Inst::Group { scope, op } => {
                self.scope(*scope);
                self.group(op)?;
            }
            Inst::Glsl {
                op:
                    op @ (GlslOp::InterpolateAtCentroid
                    | GlslOp::InterpolateAtSample
                    | GlslOp::InterpolateAtOffset),
                ..
            } => {
                // Vulkan gates `InterpolationFunction` on `sampleRateShading`.
                self.cap(Cap::InterpolationFunction);
                self.feat(VkFeature::SampleRateShading);
                // An offset is read from the pixel center only where the device does as
                // GLSL.std.450 states.
                if *op == GlslOp::InterpolateAtOffset {
                    self.prop(VkProperty::InterpolateAtOffsetCentered);
                }
            }
            Inst::RayQuery(RayQueryInst::Initialize { flags, .. }) => {
                // `SkipTrianglesKHR` or `SkipAABBsKHR` need `RayTraversalPrimitiveCullingKHR`,
                // which `rayQuery` enables; a runtime flags word may hold either.
                if self
                    .const_bits(*flags)?
                    .is_none_or(|f| f & SKIP_PRIMITIVES != 0)
                {
                    self.cap(Cap::RayTraversalPrimitiveCullingKHR);
                }
            }
            // The builder admits 16- and 32-bit floats (`FunctionBuilder::fma`): `shaderFmaFloat16`
            // or `shaderFmaFloat32` by the operands' width, the result's
            // (VUID-RuntimeSpirv-shaderFmaFloat16-10977, VUID-RuntimeSpirv-shaderFmaFloat32-10978).
            Inst::Fma { a, .. } => {
                self.cap(Cap::Fmakhr);
                self.ext(SpirvExtension::KhrFma);
                let half = matches!(
                    self.scalar_of(self.value_ty(*a)?)?,
                    Type::Float { width: 16 }
                );
                self.feat(if half {
                    VkFeature::ShaderFmaFloat16
                } else {
                    VkFeature::ShaderFmaFloat32
                });
            }
            Inst::DemoteToHelperInvocation | Inst::IsHelperInvocation => {
                self.cap(Cap::DemoteToHelperInvocation);
                self.ext(SpirvExtension::ExtDemoteToHelperInvocation);
                self.feat(VkFeature::ShaderDemoteToHelperInvocation);
            }
            _ => {}
        }
        Ok(())
    }

    /// Dynamic indexing of a descriptor array needs the array kind's dynamic-indexing
    /// capability and feature; a `NonUniform` element needs `ShaderNonUniform` and the kind's
    /// non-uniform indexing capability and feature instead.
    fn access_chain(&mut self, base: Value, indices: &[Value], non_uniform: bool) -> BResult<()> {
        let (class, pointee) = self.pointer_of(base)?;
        let elem = match self.ty(pointee)? {
            Type::Array {
                elem, stride: None, ..
            }
            | Type::RuntimeArray { elem, stride: None }
                if matches!(
                    self.m.types.facts(pointee)?.shape,
                    Shape::DescriptorArray { .. }
                ) =>
            {
                *elem
            }
            _ => return Ok(()),
        };
        let Some(first) = indices.first() else {
            return Ok(());
        };
        if non_uniform {
            let (c, f) = match (class, self.ty(elem)?) {
                (StorageClass::StorageBuffer, _) => (
                    Cap::StorageBufferArrayNonUniformIndexing,
                    VkFeature::ShaderStorageBufferArrayNonUniformIndexing,
                ),
                (_, Type::Image(img)) => match (img.dim, img.usage) {
                    (Dim::Buffer, ImageUsage::Sampled) => (
                        Cap::UniformTexelBufferArrayNonUniformIndexing,
                        VkFeature::ShaderUniformTexelBufferArrayNonUniformIndexing,
                    ),
                    (Dim::Buffer, ImageUsage::Storage) => (
                        Cap::StorageTexelBufferArrayNonUniformIndexing,
                        VkFeature::ShaderStorageTexelBufferArrayNonUniformIndexing,
                    ),
                    (_, ImageUsage::Storage) => (
                        Cap::StorageImageArrayNonUniformIndexing,
                        VkFeature::ShaderStorageImageArrayNonUniformIndexing,
                    ),
                    (_, ImageUsage::Sampled) => (
                        Cap::SampledImageArrayNonUniformIndexing,
                        VkFeature::ShaderSampledImageArrayNonUniformIndexing,
                    ),
                },
                // Samplers count as sampled images (`shaderSampledImageArrayNonUniformIndexing`).
                _ => (
                    Cap::SampledImageArrayNonUniformIndexing,
                    VkFeature::ShaderSampledImageArrayNonUniformIndexing,
                ),
            };
            self.cap(Cap::ShaderNonUniform);
            self.cap(c);
            self.feat(f);
            return Ok(());
        }
        let dynamic = !matches!(
            self.m.values.get(first.index()).map(|d| d.origin),
            Some(ValueOrigin::Const(_))
        );
        if !dynamic {
            return Ok(());
        }
        let (c, f) = match (class, self.ty(elem)?) {
            (StorageClass::Uniform, _) => (
                Cap::UniformBufferArrayDynamicIndexing,
                VkFeature::ShaderUniformBufferArrayDynamicIndexing,
            ),
            (StorageClass::StorageBuffer, _) => (
                Cap::StorageBufferArrayDynamicIndexing,
                VkFeature::ShaderStorageBufferArrayDynamicIndexing,
            ),
            (_, Type::Image(img)) => match (img.dim, img.usage) {
                (Dim::SubpassData, _) => (
                    Cap::InputAttachmentArrayDynamicIndexing,
                    VkFeature::ShaderInputAttachmentArrayDynamicIndexing,
                ),
                (Dim::Buffer, ImageUsage::Sampled) => (
                    Cap::UniformTexelBufferArrayDynamicIndexing,
                    VkFeature::ShaderUniformTexelBufferArrayDynamicIndexing,
                ),
                (Dim::Buffer, ImageUsage::Storage) => (
                    Cap::StorageTexelBufferArrayDynamicIndexing,
                    VkFeature::ShaderStorageTexelBufferArrayDynamicIndexing,
                ),
                (_, ImageUsage::Storage) => (
                    Cap::StorageImageArrayDynamicIndexing,
                    VkFeature::ShaderStorageImageArrayDynamicIndexing,
                ),
                (_, ImageUsage::Sampled) => (
                    Cap::SampledImageArrayDynamicIndexing,
                    VkFeature::ShaderSampledImageArrayDynamicIndexing,
                ),
            },
            _ => (
                Cap::SampledImageArrayDynamicIndexing,
                VkFeature::ShaderSampledImageArrayDynamicIndexing,
            ),
        };
        self.cap(c);
        self.feat(f);
        Ok(())
    }

    fn image_of(&self, v: Value) -> BResult<ImageType> {
        match self.ty(self.value_ty(v)?)? {
            Type::Image(img) => Ok(*img),
            _ => Err(BuildError::Dangling(Dangling::Value(v))),
        }
    }

    /// An `OpImageSparse*` instruction: the `SparseResidency` capability, which Vulkan gates on
    /// `shaderResourceResidency`.
    fn sparse_residency(&mut self) {
        self.cap(Cap::SparseResidency);
        self.feat(VkFeature::ShaderResourceResidency);
    }

    /// A storage access's texel operands (`NonPrivateTexel`, `VolatileTexel`, `MakeTexel*`):
    /// they need the Vulkan memory model, and their scope counts like an atomic's.
    fn texel_memory(&mut self, mem: TexelMemory) {
        if mem.vulkan_memory_model() {
            self.vulkan_memory_model = true;
        }
        if let Some(s) = mem.made_scope() {
            self.scope(s);
        }
    }

    /// The operands of an image instruction; `gather` for `OpImage*Gather`.
    fn image_operands(&mut self, ops: &ImageOperands, gather: bool) {
        if ops.min_lod.is_some() {
            self.cap(Cap::MinLod);
            self.feat(VkFeature::ShaderResourceMinLod);
        }
        if ops.offset.is_some() {
            self.cap(Cap::ImageGatherExtended);
            self.feat(VkFeature::ShaderImageGatherExtended);
            // `Offset` outside a gather needs `maintenance8` (VUID-RuntimeSpirv-Offset-10213).
            if !gather {
                self.feat(VkFeature::Maintenance8);
            }
        }
    }

    /// Features an atomic on `ptr` needs: 64-bit integer and float atomics per storage class, and
    /// the stage's store feature for descriptor-backed memory. `op` is the read-modify-write
    /// operation (`None` for loads, stores, exchanges of the plain kind, steps, and
    /// compare-exchange).
    fn atomic(&mut self, ptr: Value, op: Option<AtomicOp>) -> BResult<()> {
        let (class, pointee) = self.pointer_of(ptr)?;
        let buffer = matches!(
            class,
            StorageClass::StorageBuffer | StorageClass::PhysicalStorageBuffer
        );
        if buffer || class == StorageClass::Image {
            self.stage_stores();
        }
        match *self.ty(pointee)? {
            Type::Int { width: 64, .. } => {
                self.cap(Cap::Int64Atomics);
                self.feat(match class {
                    StorageClass::Workgroup => VkFeature::ShaderSharedInt64Atomics,
                    StorageClass::Image => VkFeature::ShaderImageInt64Atomics,
                    _ => VkFeature::ShaderBufferInt64Atomics,
                });
            }
            Type::Float { width } => self.float_atomic(class, width, op)?,
            _ => {}
        }
        Ok(())
    }

    fn float_atomic(
        &mut self,
        class: StorageClass,
        width: u8,
        op: Option<AtomicOp>,
    ) -> BResult<()> {
        use VkFeature as F;
        // [plain (load/store/exchange), add] per (class, width).
        let set = match (class, width) {
            (StorageClass::Workgroup, 16) => [
                F::ShaderSharedFloat16Atomics,
                F::ShaderSharedFloat16AtomicAdd,
            ],
            (StorageClass::Workgroup, 32) => [
                F::ShaderSharedFloat32Atomics,
                F::ShaderSharedFloat32AtomicAdd,
            ],
            (StorageClass::Workgroup, 64) => [
                F::ShaderSharedFloat64Atomics,
                F::ShaderSharedFloat64AtomicAdd,
            ],
            (StorageClass::Image, 32) => {
                [F::ShaderImageFloat32Atomics, F::ShaderImageFloat32AtomicAdd]
            }
            (StorageClass::StorageBuffer | StorageClass::PhysicalStorageBuffer, 16) => [
                F::ShaderBufferFloat16Atomics,
                F::ShaderBufferFloat16AtomicAdd,
            ],
            (StorageClass::StorageBuffer | StorageClass::PhysicalStorageBuffer, 32) => [
                F::ShaderBufferFloat32Atomics,
                F::ShaderBufferFloat32AtomicAdd,
            ],
            (StorageClass::StorageBuffer | StorageClass::PhysicalStorageBuffer, 64) => [
                F::ShaderBufferFloat64Atomics,
                F::ShaderBufferFloat64AtomicAdd,
            ],
            _ => return Err(BuildError::Type(TypeRule::PointeeForClass(class))),
        };
        let [plain, add] = set;
        match op {
            Some(AtomicOp::FAdd) => {
                self.feat(add);
                let (c, e) = match width {
                    16 => (
                        Cap::AtomicFloat16AddEXT,
                        SpirvExtension::ExtShaderAtomicFloat16Add,
                    ),
                    32 => (
                        Cap::AtomicFloat32AddEXT,
                        SpirvExtension::ExtShaderAtomicFloatAdd,
                    ),
                    _ => (
                        Cap::AtomicFloat64AddEXT,
                        SpirvExtension::ExtShaderAtomicFloatAdd,
                    ),
                };
                self.cap(c);
                self.ext(e);
            }
            _ => self.feat(plain),
        }
        Ok(())
    }

    fn group(&mut self, op: &GroupInst) -> BResult<()> {
        let (c, class) = match op {
            GroupInst::Elect => (Cap::GroupNonUniform, SubgroupOp::Basic),
            GroupInst::All(_) | GroupInst::Any(_) => (Cap::GroupNonUniformVote, SubgroupOp::Vote),
            GroupInst::BroadcastFirst(_)
            | GroupInst::Ballot(_)
            | GroupInst::BallotBitExtract { .. } => {
                (Cap::GroupNonUniformBallot, SubgroupOp::Ballot)
            }
            GroupInst::Shuffle { .. } => (Cap::GroupNonUniformShuffle, SubgroupOp::Shuffle),
            GroupInst::Arith {
                group: GroupOperation::ClusteredReduce { .. },
                ..
            } => (Cap::GroupNonUniformClustered, SubgroupOp::Clustered),
            GroupInst::Arith { .. } => (Cap::GroupNonUniformArithmetic, SubgroupOp::Arithmetic),
            GroupInst::QuadBroadcast { .. } | GroupInst::QuadSwap { .. } => {
                if self.model == ExecModel::Vertex {
                    self.prop(VkProperty::SubgroupQuadOperationsInAllStages);
                }
                (Cap::GroupNonUniformQuad, SubgroupOp::Quad)
            }
        };
        self.cap(Cap::GroupNonUniform);
        self.cap(c);
        self.subgroup(class);
        // Subgroup operations on 8-, 16-, or 64-bit components need extended types (the ballot
        // operands and lane indices are 32-bit by construction).
        let mut extended = false;
        let mut err = None;
        op.for_each_value(|v| match self.value_ty(v).and_then(|t| self.scalar_of(t)) {
            Ok(Type::Int { width, .. } | Type::Float { width }) if *width != 32 => extended = true,
            Ok(_) => {}
            Err(e) => err = Some(e),
        });
        if let Some(e) = err {
            return Err(e);
        }
        if extended {
            self.feat(VkFeature::ShaderSubgroupExtendedTypes);
        }
        Ok(())
    }
}

/// Whether a storage-image format needs `StorageImageExtendedFormats` (every format outside the
/// `Shader`-capability set; `Unknown` needs the without-format capabilities at its uses, and
/// the 64-bit formats `Int64ImageEXT`, which the sampled type already requires).
pub(crate) fn extended_format(f: crate::spv::ImageFormat) -> bool {
    use crate::spv::ImageFormat as F;
    !matches!(
        f,
        F::Unknown
            | F::Rgba32f
            | F::Rgba16f
            | F::R32f
            | F::Rgba8
            | F::Rgba8Snorm
            | F::Rgba32i
            | F::Rgba16i
            | F::Rgba8i
            | F::R32i
            | F::Rgba32ui
            | F::Rgba16ui
            | F::Rgba8ui
            | F::R32ui
            | F::R64ui
            | F::R64i
    )
}

/// Every [`SpirvExtension`], for exhaustiveness tests.
#[cfg(test)]
pub const ALL_SPIRV_EXTENSIONS: &[SpirvExtension] = &[
    SpirvExtension::KhrTerminateInvocation,
    SpirvExtension::ExtDemoteToHelperInvocation,
    SpirvExtension::ExtFragmentShaderInterlock,
    SpirvExtension::KhrFragmentShaderBarycentric,
    SpirvExtension::ExtShaderStencilExport,
    SpirvExtension::ExtShaderViewportIndexLayer,
    SpirvExtension::ExtShaderAtomicFloatAdd,
    SpirvExtension::ExtShaderAtomicFloatMinMax,
    SpirvExtension::ExtShaderImageInt64,
    SpirvExtension::KhrWorkgroupMemoryExplicitLayout,
    SpirvExtension::KhrComputeShaderDerivatives,
    SpirvExtension::KhrSubgroupUniformControlFlow,
    SpirvExtension::KhrMaximalReconvergence,
    SpirvExtension::KhrFloatControls,
    SpirvExtension::KhrVulkanMemoryModel,
    SpirvExtension::KhrShaderDrawParameters,
    SpirvExtension::KhrPhysicalStorageBuffer,
    SpirvExtension::ExtDescriptorIndexing,
    SpirvExtension::ExtShaderAtomicFloat16Add,
    SpirvExtension::KhrSubgroupRotate,
    SpirvExtension::KhrRayQuery,
    SpirvExtension::KhrFma,
    SpirvExtension::ExtMeshShader,
    SpirvExtension::KhrQuadControl,
    SpirvExtension::KhrPostDepthCoverage,
];

/// The `OpExtension` name of a SPIR-V extension.
pub fn spirv_extension_name(e: SpirvExtension) -> &'static str {
    match e {
        SpirvExtension::KhrTerminateInvocation => "SPV_KHR_terminate_invocation",
        SpirvExtension::ExtDemoteToHelperInvocation => "SPV_EXT_demote_to_helper_invocation",
        SpirvExtension::ExtFragmentShaderInterlock => "SPV_EXT_fragment_shader_interlock",
        SpirvExtension::KhrFragmentShaderBarycentric => "SPV_KHR_fragment_shader_barycentric",
        SpirvExtension::ExtShaderStencilExport => "SPV_EXT_shader_stencil_export",
        SpirvExtension::ExtShaderViewportIndexLayer => "SPV_EXT_shader_viewport_index_layer",
        SpirvExtension::ExtShaderAtomicFloatAdd => "SPV_EXT_shader_atomic_float_add",
        SpirvExtension::ExtShaderAtomicFloatMinMax => "SPV_EXT_shader_atomic_float_min_max",
        SpirvExtension::ExtShaderImageInt64 => "SPV_EXT_shader_image_int64",
        SpirvExtension::KhrWorkgroupMemoryExplicitLayout => {
            "SPV_KHR_workgroup_memory_explicit_layout"
        }
        SpirvExtension::KhrComputeShaderDerivatives => "SPV_KHR_compute_shader_derivatives",
        SpirvExtension::KhrSubgroupUniformControlFlow => "SPV_KHR_subgroup_uniform_control_flow",
        SpirvExtension::KhrMaximalReconvergence => "SPV_KHR_maximal_reconvergence",
        SpirvExtension::KhrFloatControls => "SPV_KHR_float_controls",
        SpirvExtension::KhrVulkanMemoryModel => "SPV_KHR_vulkan_memory_model",
        SpirvExtension::KhrShaderDrawParameters => "SPV_KHR_shader_draw_parameters",
        SpirvExtension::KhrPhysicalStorageBuffer => "SPV_KHR_physical_storage_buffer",
        SpirvExtension::ExtDescriptorIndexing => "SPV_EXT_descriptor_indexing",
        SpirvExtension::ExtShaderAtomicFloat16Add => "SPV_EXT_shader_atomic_float16_add",
        SpirvExtension::KhrSubgroupRotate => "SPV_KHR_subgroup_rotate",
        SpirvExtension::KhrRayQuery => "SPV_KHR_ray_query",
        SpirvExtension::KhrFma => "SPV_KHR_fma",
        SpirvExtension::ExtMeshShader => "SPV_EXT_mesh_shader",
        SpirvExtension::KhrQuadControl => "SPV_KHR_quad_control",
        SpirvExtension::KhrPostDepthCoverage => "SPV_KHR_post_depth_coverage",
    }
}
