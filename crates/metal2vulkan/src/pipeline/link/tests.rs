//! Link tests: the link contract read from typed modules, numbering and the layout, pipeline-wide
//! limits (the sweep over every committed profile), stage-interface locations and packing, vertex
//! input, graphics state, and the property over random pipelines.

mod geometry;
mod property;
mod render;
mod sweep;

use std::collections::BTreeSet;

use super::*;
use crate::device::derive::vk_stage;
use crate::device::subgroup::SubgroupRequirement;
use crate::device::DevicePolicy;
use crate::library::FunctionCompileKey;
use crate::pipeline::artifact::{Reconvergence, StrategyCensus};
use crate::pipeline::interface::{StageDescriptor, StageSpec};
use crate::pipeline::stage_io::{
    Interpolation, IoNumeric, IoRole, IoType, IoVariable, Sampling, StageIo,
};
use crate::pipeline::sym::{AirUserKey, CombinedSampler, PushField, SamplerRef, SyntheticBinding};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::ids::{GlobalId, TypeId};
use crate::spirv::ir::{
    natural_layout, ConstBits, Constant, Dim, ExecModel, ImageType, ImageUsage, IoDecor, IoDir,
    LayoutRule, StorageClass, Type, TypedModule,
};
use crate::spirv::serialize::serialize;

// ---- Fixtures ----

pub(super) fn profile(stem: &str) -> DeviceProfile {
    crate::pipeline::testing::profile(stem, DevicePolicy::default()).expect("profile")
}

/// The key of user varying `name`.
pub(super) fn key(name: &str) -> VaryingKey {
    VaryingKey::user(name.as_bytes())
}

/// `components` lanes of `numeric` from a 32-bit source.
pub(super) fn io_ty(numeric: IoNumeric, components: u8) -> IoType {
    IoType {
        numeric,
        components,
        source_bits: 32,
        bits: 32,
    }
}

/// Vertex attribute `index` of a format that is not a narrow float.
pub(super) fn attr(index: u32, ty: IoType) -> IoVariable {
    attr_of(index, ty, false)
}

/// Vertex attribute `index`, `narrow_float` as its projection entry states.
pub(super) fn attr_of(index: u32, ty: IoType, narrow_float: bool) -> IoVariable {
    IoVariable {
        key: VaryingKey::attribute(index),
        role: IoRole::Attribute {
            index,
            narrow_float,
        },
        ty,
    }
}

/// User varying `name`: a vertex output (`interpolation` `None`) or a fragment input.
pub(super) fn varying(name: &str, ty: IoType, interpolation: Option<Interpolation>) -> IoVariable {
    IoVariable {
        key: key(name),
        role: IoRole::Varying {
            user: AirUserKey(name.as_bytes().into()),
            interpolation,
        },
        ty,
    }
}

/// Color output `attachment` with dual-source index `index`.
pub(super) fn color(attachment: u32, index: u32, ty: IoType) -> IoVariable {
    IoVariable {
        key: VaryingKey::color(attachment, index),
        role: IoRole::Color { attachment, index },
        ty,
    }
}

/// Perspective-correct center interpolation.
pub(super) const SMOOTH: Option<Interpolation> = Some(Interpolation::Smooth {
    sampling: Sampling::Center,
    perspective: true,
});

/// A stage-IO contract of `inputs` and `outputs`.
pub(super) fn stage_io(inputs: Vec<IoVariable>, outputs: Vec<IoVariable>) -> StageIo {
    StageIo {
        inputs,
        outputs,
        ..StageIo::default()
    }
}

pub(super) fn res(stage: Stage, kind: ResourceKind, index: u32) -> ResourceSlot {
    ResourceSlot { stage, kind, index }
}

/// A stage interface with no slots, push fields, or stage IO and these link symbols.
pub(super) fn iface(
    stage: Stage,
    descriptors: Vec<StageDescriptor>,
    spec: Vec<StageSpec>,
) -> StageInterface {
    StageInterface {
        stage,
        slots: BTreeMap::new(),
        threadgroup: Vec::new(),
        push: crate::lower::push::place(stage, &BTreeSet::new()),
        descriptors,
        spec,
        needs_pointer_translation: false,
        needs_handle_translation: false,
        io: Default::default(),
        tessellation: None,
        writes_memory: false,
        mesh_records: None,
        emulated_payload: None,
        payload: None,
        reaches_command_buffers: false,
        command_writes: Default::default(),
        writes_textures: false,
        numbered_samplers: Vec::new(),
        own_pixel_textures: Vec::new(),
        snapshot_textures: Vec::new(),
        private_bytes: 0,
    }
}

/// An artifact of `interface` whose module is `module` (or an empty one).
pub(super) fn artifact(
    interface: StageInterface,
    module: Option<crate::spirv::serialize::RelocatableModule>,
) -> FunctionArtifact {
    let module = module.unwrap_or_else(|| crate::spirv::serialize::RelocatableModule {
        words: Vec::new(),
        relocs: Vec::new(),
        requirements: Default::default(),
    });
    let requirements = module.requirements.clone();
    FunctionArtifact {
        key: FunctionCompileKey::new([0; 32]),
        module,
        interface,
        facts: Vec::new(),
        subgroup: SubgroupRequirement::NONE,
        reconvergence: Reconvergence::Guaranteed,
        precision: Vec::new(),
        strategies: StrategyCensus::default(),
        blob: None,
        requirements,
        variant_bits: 0,
        profile: [0; 32],
        kernel: Default::default(),
        reflection: None,
        slot_writes: Vec::new(),
        specialized: Vec::new(),
    }
}

/// An artifact serialized from `m`, with the interface its link contract gives and stage IO
/// `io`.
pub(super) fn from_module(stage: Stage, m: TypedModule, io: StageIo) -> FunctionArtifact {
    let s = contract::stage_symbols(&m, &io).expect("contract");
    let module = serialize(m).expect("serialize");
    let mut i = iface(stage, s.descriptors, s.spec);
    i.io = io;
    artifact(i, Some(module))
}

pub(super) fn u32_ty(b: &mut ModuleBuilder) -> TypeId {
    b.ty(Type::Int {
        width: 32,
        signed: false,
    })
    .expect("u32")
}

pub(super) fn f32_ty(b: &mut ModuleBuilder) -> TypeId {
    b.ty(Type::Float { width: 32 }).expect("f32")
}

pub(super) fn u32_const(b: &mut ModuleBuilder, v: u64) -> crate::spirv::ids::ConstId {
    let u = u32_ty(b);
    b.constant(Constant::Scalar {
        ty: u,
        bits: ConstBits(v),
    })
    .expect("constant")
}

/// A `Block { u32 data[]; }` storage buffer (std430) bound to `slot`.
pub(super) fn ssbo(b: &mut ModuleBuilder, slot: ResourceSlot) -> GlobalId {
    let u = u32_ty(b);
    let rta = b
        .ty(Type::RuntimeArray {
            elem: u,
            stride: Some(4),
        })
        .expect("runtime array");
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta]).expect("layout");
    let block = b
        .ty(Type::Struct {
            members: [rta].into(),
            layout: Some(layout),
        })
        .expect("block");
    b.resource(slot, StorageClass::StorageBuffer, block)
        .expect("ssbo")
}

/// A `Block { uvec4 data[n]; }` uniform buffer (std140) bound to `slot`.
pub(super) fn ubo(b: &mut ModuleBuilder, slot: ResourceSlot, n: u64) -> GlobalId {
    let u = u32_ty(b);
    let uvec4 = b.ty(Type::Vector { elem: u, len: 4 }).expect("uvec4");
    let len = u32_const(b, n);
    let arr = b
        .ty(Type::Array {
            elem: uvec4,
            len,
            stride: Some(16),
        })
        .expect("array");
    let layout = natural_layout(b.types(), LayoutRule::Std140, true, &[arr]).expect("layout");
    let block = b
        .ty(Type::Struct {
            members: [arr].into(),
            layout: Some(layout),
        })
        .expect("block");
    b.resource(slot, StorageClass::Uniform, block).expect("ubo")
}

/// An image type over `f32`.
pub(super) fn image(b: &mut ModuleBuilder, dim: Dim, usage: ImageUsage) -> TypeId {
    let f = f32_ty(b);
    b.ty(Type::Image(ImageType {
        sampled_type: f,
        dim,
        depth: false,
        arrayed: false,
        multisampled: false,
        usage,
        format: match usage {
            ImageUsage::Sampled => crate::spv::ImageFormat::Unknown,
            _ => crate::spv::ImageFormat::Rgba32f,
        },
    }))
    .expect("image")
}

/// Declare an opaque resource of type `ty` (or a descriptor array of `count` of them).
pub(super) fn opaque(b: &mut ModuleBuilder, slot: ResourceSlot, ty: TypeId, count: u64) {
    let ty = if count > 1 {
        let len = u32_const(b, count);
        b.ty(Type::Array {
            elem: ty,
            len,
            stride: None,
        })
        .expect("descriptor array")
    } else {
        ty
    };
    b.resource(slot, StorageClass::UniformConstant, ty)
        .expect("opaque resource");
}

/// Finish `b` with an empty entry of `model` (a compute entry takes its local size from the
/// `LocalSize` spec composite).
pub(super) fn finish(mut b: ModuleBuilder, model: ExecModel) -> TypedModule {
    if model == ExecModel::GLCompute {
        let u = u32_ty(&mut b);
        let x = b
            .spec(
                SpecKey::LocalSize(crate::pipeline::sym::Axis::X),
                u,
                ConstBits(1),
            )
            .expect("sx");
        let y = b
            .spec(
                SpecKey::LocalSize(crate::pipeline::sym::Axis::Y),
                u,
                ConstBits(1),
            )
            .expect("sy");
        let z = b
            .spec(
                SpecKey::LocalSize(crate::pipeline::sym::Axis::Z),
                u,
                ConstBits(1),
            )
            .expect("sz");
        b.workgroup_size(x, y, z).expect("workgroup size");
    }
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish");
    b.entry(model, main, Vec::new()).expect("entry");
    b.finalize().expect("finalize")
}

fn stage_of(d: &StageDescriptor) -> vk::ShaderStageFlags {
    vk_stage(d.slot.stage).flags()
}

// ---- The link contract ----

/// Every descriptor kind reads back with the type its storage class and data type declare, a
/// descriptor array with its length; spec constants carry their size and default.
#[test]
fn the_contract_reads_descriptor_types_counts_and_spec_constants() {
    let k = Stage::Kernel;
    let mut b = ModuleBuilder::new();
    ssbo(&mut b, res(k, ResourceKind::Buffer, 3));
    ubo(
        &mut b,
        res(
            k,
            ResourceKind::Synthetic(SyntheticBinding::ArgumentTable),
            0,
        ),
        2,
    );
    let sampled = image(&mut b, Dim::D2, ImageUsage::Sampled);
    opaque(&mut b, res(k, ResourceKind::Texture, 0), sampled, 1);
    let storage = image(&mut b, Dim::D2, ImageUsage::Storage);
    opaque(&mut b, res(k, ResourceKind::Texture, 1), storage, 1);
    let texel = image(&mut b, Dim::Buffer, ImageUsage::Sampled);
    opaque(&mut b, res(k, ResourceKind::Texture, 2), texel, 1);
    let stexel = image(&mut b, Dim::Buffer, ImageUsage::Storage);
    opaque(&mut b, res(k, ResourceKind::Texture, 3), stexel, 1);
    opaque(&mut b, res(k, ResourceKind::Texture, 4), sampled, 3);
    let sampler = b.ty(Type::Sampler).expect("sampler");
    opaque(&mut b, res(k, ResourceKind::Sampler, 0), sampler, 1);
    let combined = b
        .ty(Type::SampledImage { image: sampled })
        .expect("combined");
    opaque(&mut b, res(k, combined_kind(), 5), combined, 1);
    let u = u32_ty(&mut b);
    b.spec(SpecKey::TgLength(0), u, ConstBits(64))
        .expect("spec");
    let bool_ = b.ty(Type::Bool).expect("bool");
    b.spec(SpecKey::MultiRegion, bool_, ConstBits(0))
        .expect("spec");
    let m = finish(b, ExecModel::GLCompute);
    let s = contract::stage_symbols(&m, &StageIo::default()).expect("contract");
    use vk::DescriptorType as T;
    let got: Vec<(ResourceSlot, vk::DescriptorType, u32)> = s
        .descriptors
        .iter()
        .map(|d| (d.slot, d.ty, d.count))
        .collect();
    assert_eq!(
        got,
        vec![
            (res(k, ResourceKind::Buffer, 3), T::STORAGE_BUFFER, 1),
            (res(k, ResourceKind::Texture, 0), T::SAMPLED_IMAGE, 1),
            (res(k, ResourceKind::Texture, 1), T::STORAGE_IMAGE, 1),
            (res(k, ResourceKind::Texture, 2), T::UNIFORM_TEXEL_BUFFER, 1),
            (res(k, ResourceKind::Texture, 3), T::STORAGE_TEXEL_BUFFER, 1),
            (res(k, ResourceKind::Texture, 4), T::SAMPLED_IMAGE, 3),
            (res(k, ResourceKind::Sampler, 0), T::SAMPLER, 1),
            (
                res(
                    k,
                    ResourceKind::Synthetic(SyntheticBinding::ArgumentTable),
                    0
                ),
                T::UNIFORM_BUFFER,
                1
            ),
            (res(k, combined_kind(), 5), T::COMBINED_IMAGE_SAMPLER, 1),
        ]
    );
    use crate::pipeline::sym::Axis;
    assert_eq!(
        s.spec,
        vec![
            StageSpec::new(SpecKey::LocalSize(Axis::X), 4, 1),
            StageSpec::new(SpecKey::LocalSize(Axis::Y), 4, 1),
            StageSpec::new(SpecKey::LocalSize(Axis::Z), 4, 1),
            StageSpec::new(SpecKey::MultiRegion, 4, 0),
            StageSpec::new(SpecKey::TgLength(0), 4, 64),
        ]
    );
}

/// A module of `model` declaring exactly the interface variables of `io`.
pub(super) fn io_module(model: ExecModel, io: &StageIo) -> TypedModule {
    let mut b = ModuleBuilder::new();
    declare_io(&mut b, io);
    finish(b, model)
}

/// Declare the interface variables of `io` (`float`/`uint` scalars and vectors; varyings
/// packable, integer fragment inputs flat, colors with their dual-source `Index`).
pub(super) fn declare_io(b: &mut ModuleBuilder, io: &StageIo) {
    for (vars, dir) in [(&io.inputs, IoDir::Input), (&io.outputs, IoDir::Output)] {
        for v in vars {
            let ty = io_decl_ty(b, v.ty);
            let deco = IoDecor {
                packed: matches!(v.role, IoRole::Varying { .. } | IoRole::ViewportIndex),
                interpolation: match v.role {
                    IoRole::Varying {
                        interpolation: Some(Interpolation::Flat),
                        ..
                    } => crate::spirv::ir::Interpolation::Flat,
                    IoRole::ViewportIndex if dir == IoDir::Input => {
                        crate::spirv::ir::Interpolation::Flat
                    }
                    _ => crate::spirv::ir::Interpolation::Smooth,
                },
                index: match v.role {
                    IoRole::Color { index, .. } if index > 0 => Some(index),
                    _ => None,
                },
                ..IoDecor::default()
            };
            b.io(v.key.clone(), dir, ty, deco).expect("io");
        }
    }
}

/// The declared SPIR-V type of an interface variable of type `t`.
pub(super) fn io_decl_ty(b: &mut ModuleBuilder, t: IoType) -> TypeId {
    let elem = match t.numeric {
        IoNumeric::Float => f32_ty(b),
        _ => u32_ty(b),
    };
    if t.components == 1 {
        elem
    } else {
        b.ty(Type::Vector {
            elem,
            len: t.components,
        })
        .expect("vector")
    }
}

/// The contract accepts a module whose interface variables are exactly the stage-IO contract's,
/// and refuses (as an invariant failure) one that declares a variable the contract omits, lacks
/// one it lists, or declares one with another direction or type.
#[test]
fn the_contract_checks_interface_variables_against_the_stage_io() {
    let f4 = io_ty(IoNumeric::Float, 4);
    let io = stage_io(
        vec![attr(2, io_ty(IoNumeric::Uint, 2))],
        vec![varying("uv", f4, None)],
    );
    let m = io_module(ExecModel::Vertex, &io);
    assert!(contract::stage_symbols(&m, &io).is_ok());
    let internal = |r: Result<contract::StageSymbols, Error>| matches!(r, Err(Error::Internal(ref i)) if i.kind == InternalKind::Invariant);
    // The contract omits a declared variable.
    assert!(internal(contract::stage_symbols(
        &m,
        &stage_io(vec![], vec![varying("uv", f4, None)])
    )));
    // The contract lists a variable the module lacks.
    let mut extra = io.clone();
    extra.outputs.push(varying("more", f4, None));
    assert!(internal(contract::stage_symbols(&m, &extra)));
    // Another type or direction.
    let other_ty = stage_io(
        vec![attr(2, io_ty(IoNumeric::Float, 2))],
        vec![varying("uv", f4, None)],
    );
    assert!(internal(contract::stage_symbols(&m, &other_ty)));
    let other_dir = stage_io(
        vec![attr(2, io_ty(IoNumeric::Uint, 2)), varying("uv", f4, None)],
        vec![],
    );
    assert!(internal(contract::stage_symbols(&m, &other_dir)));
}

// ---- Numbering and layout ----

/// Bindings are dense in set 0 in `ResourceSlot` order across stages, each listing exactly its
/// stage and its Metal or synthetic source; spec ids are dense over the pipeline, and a key two
/// stages share has one id and appears in both stages' layouts.
#[test]
fn bindings_and_spec_ids_are_dense_and_exact() {
    use vk::DescriptorType as T;
    let p = profile("moltenvk-m3");
    let (v, fr) = (Stage::Vertex, Stage::Fragment);
    let vs = iface(
        v,
        vec![
            StageDescriptor::new(res(v, ResourceKind::Buffer, 30), T::STORAGE_BUFFER, 1),
            StageDescriptor::new(res(v, ResourceKind::Buffer, 0), T::STORAGE_BUFFER, 1),
        ],
        vec![StageSpec::new(SpecKey::SampleCount, 4, 1)],
    );
    let fs = iface(
        fr,
        vec![
            StageDescriptor::new(res(fr, ResourceKind::Buffer, 0), T::UNIFORM_BUFFER, 1),
            StageDescriptor::new(res(fr, ResourceKind::Texture, 2), T::SAMPLED_IMAGE, 4),
            StageDescriptor::new(
                res(
                    fr,
                    ResourceKind::Synthetic(SyntheticBinding::ArgumentTable),
                    0,
                ),
                T::UNIFORM_BUFFER,
                1,
            ),
        ],
        vec![
            StageSpec::new(SpecKey::SampleCount, 4, 1),
            StageSpec::new(SpecKey::TgLength(0), 4, 0),
        ],
    );
    let (va, fa) = (artifact(vs, None), artifact(fs, None));
    let l = link(&[&va, &fa], &render(), &p).expect("link");
    let set = &l.layout.sets[0];
    assert_eq!(l.layout.sets.len(), 1);
    let order: Vec<ResourceSlot> = l.table.bindings.keys().copied().collect();
    let mut sorted = order.clone();
    sorted.sort();
    assert_eq!(order, sorted);
    for (n, b) in set.bindings.iter().enumerate() {
        assert_eq!(b.binding, n as u32, "dense");
    }
    let all: Vec<&StageDescriptor> = va
        .interface
        .descriptors
        .iter()
        .chain(&fa.interface.descriptors)
        .collect();
    for d in all {
        let (s, n) = l.table.bindings[&d.slot];
        assert_eq!(s, 0);
        let b = &set.bindings[n as usize];
        assert_eq!((b.ty, b.count, b.stages), (d.ty, d.count, stage_of(d)));
        assert_eq!(b.source, binding_source(d.slot));
    }
    assert_eq!(
        set.bindings[4].source,
        BindingSource::Synthetic {
            stage: fr,
            binding: SyntheticBinding::ArgumentTable
        }
    );
    assert_eq!(
        l.table.spec,
        BTreeMap::from([(SpecKey::TgLength(0), 0), (SpecKey::SampleCount, 1)])
    );
    assert_eq!(l.specialization.len(), 2);
    let ids = |i: usize| -> Vec<(u32, SpecKey)> {
        l.specialization[i]
            .1
            .constants
            .iter()
            .map(|c| (c.id, c.key))
            .collect()
    };
    assert_eq!(ids(0), vec![(1, SpecKey::SampleCount)]);
    assert_eq!(
        ids(1),
        vec![(0, SpecKey::TgLength(0)), (1, SpecKey::SampleCount)]
    );
}

/// Push constant ranges are each stage's window, trimmed to its fields.
#[test]
fn push_ranges_are_the_stage_windows() {
    let p = profile("moltenvk-m3");
    let sb = |stage| {
        PushField::SlotBindOffset(SlotRef {
            stage,
            class: SlotClass::Buffer,
            index: 0,
        })
    };
    let mut vs = iface(Stage::Vertex, Vec::new(), Vec::new());
    vs.push = crate::lower::push::place(Stage::Vertex, &BTreeSet::from([sb(Stage::Vertex)]));
    let mut fs = iface(Stage::Fragment, Vec::new(), Vec::new());
    fs.push = crate::lower::push::place(
        Stage::Fragment,
        &BTreeSet::from([PushField::RootTableAddress, sb(Stage::Fragment)]),
    );
    let (va, fa) = (artifact(vs, None), artifact(fs, None));
    let l = link(&[&va, &fa], &render(), &p).expect("link");
    let ranges: Vec<(vk::ShaderStageFlags, u32, u32)> = l
        .layout
        .push_constants
        .iter()
        .map(|r| (r.stages, r.offset, r.size))
        .collect();
    assert_eq!(
        ranges,
        vec![
            (vk::ShaderStageFlags::VERTEX, 0, 4),
            (vk::ShaderStageFlags::FRAGMENT, 64, 12),
        ]
    );
    assert_eq!(l.interface.push.len(), 2);
}

/// The stage list must match the pipeline kind.
#[test]
fn stages_must_match_the_pipeline_kind() {
    let p = profile("moltenvk-m3");
    let k = artifact(iface(Stage::Kernel, vec![], vec![]), None);
    let v = artifact(iface(Stage::Vertex, vec![], vec![]), None);
    let f = artifact(iface(Stage::Fragment, vec![], vec![]), None);
    let bad = |r: Result<Linked, Error>| matches!(r, Err(Error::InvalidDescriptor(_)));
    assert!(link(&[&k], &LinkInputs::compute(), &p).is_ok());
    assert!(bad(link(&[&v], &LinkInputs::compute(), &p)));
    assert!(bad(link(&[], &LinkInputs::compute(), &p)));
    assert!(bad(link(&[&k, &k], &LinkInputs::compute(), &p)));
    let render = render();
    assert!(link(&[&v, &f], &render, &p).is_ok());
    assert!(link(&[&v], &render, &p).is_ok(), "no fragment function");
    assert!(bad(link(&[&f], &render, &p)));
    assert!(bad(link(&[&v, &k], &render, &p)));
    assert!(bad(link(&[&v, &f, &f], &render, &p)));
    let mut discard = RenderLinkInputs::new();
    discard.rasterization_enabled = false;
    let discard = LinkInputs::Render(Box::new(discard));
    assert!(link(&[&v], &discard, &p).is_ok());
    assert!(bad(link(&[&v, &f], &discard, &p)));
    for a in [&k, &v, &f] {
        assert!(link(&[a], &LinkInputs::standalone(), &p).is_ok());
    }
    assert!(bad(link(&[&v, &f], &LinkInputs::standalone(), &p)));
}

/// A mesh render pipeline takes an optional object stage, its mesh stage, and a fragment stage
/// exactly when it rasterizes: Metal asserts on one without its mesh function, or without a
/// fragment function where it rasterizes.
#[test]
fn a_mesh_pipeline_takes_its_stages_in_order() {
    let p = profile("lavapipe");
    let o = artifact(iface(Stage::Object, vec![], vec![]), None);
    let m = artifact(iface(Stage::Mesh, vec![], vec![]), None);
    let v = artifact(iface(Stage::Vertex, vec![], vec![]), None);
    let f = artifact(iface(Stage::Fragment, vec![], vec![]), None);
    let bad = |r: Result<Linked, Error>| matches!(r, Err(Error::InvalidDescriptor(_)));
    let mesh = LinkInputs::Mesh(Box::default());
    assert!(link(&[&m, &f], &mesh, &p).is_ok());
    assert!(link(&[&o, &m, &f], &mesh, &p).is_ok());
    assert!(bad(link(&[&m], &mesh, &p)), "no fragment function");
    assert!(bad(link(&[&o, &f], &mesh, &p)), "no mesh function");
    assert!(bad(link(&[&v, &f], &mesh, &p)));
    assert!(bad(link(&[&m, &o, &f], &mesh, &p)));
    assert!(bad(link(&[&m, &f, &f], &mesh, &p)));
    assert!(bad(link(&[&m, &f], &render(), &p)));
    let mut discard = RenderLinkInputs::new();
    discard.rasterization_enabled = false;
    let discard = LinkInputs::Mesh(Box::new(discard));
    assert!(link(&[&m], &discard, &p).is_ok());
    assert!(link(&[&o, &m], &discard, &p).is_ok());
    assert!(bad(link(&[&m, &f], &discard, &p)));
}

/// An artifact of `stage` with no slots and stage IO `io`.
fn with_io(stage: Stage, io: StageIo) -> FunctionArtifact {
    let mut i = iface(stage, vec![], vec![]);
    i.io = io;
    artifact(i, None)
}

/// A mesh output read per primitive.
fn primitive_varying(name: &str, ty: IoType) -> IoVariable {
    IoVariable {
        key: key(name),
        role: IoRole::PrimitiveVarying {
            user: AirUserKey(name.as_bytes().into()),
        },
        ty,
    }
}

/// In a mesh render pipeline each fragment input takes the location of the mesh output of its key
/// and type; the fragment function reads per primitive exactly the inputs whose keys the mesh
/// function writes per primitive, and a render pipeline's none; an input no mesh output writes is
/// `UnmatchedVarying`.
#[test]
fn a_mesh_pipelines_fragment_reads_its_primitive_outputs_per_primitive() {
    use crate::error::RuntimeInput;
    let p = profile("lavapipe");
    let f4 = io_ty(IoNumeric::Float, 4);
    let u1 = io_ty(IoNumeric::Uint, 1);
    let flat = Some(Interpolation::Flat);
    let mesh_io = stage_io(
        vec![],
        vec![
            varying("shade", f4, None),
            primitive_varying("id", u1),
            primitive_varying("weight", f4),
        ],
    );
    let m = with_io(Stage::Mesh, mesh_io);
    let fragment = |inputs: Vec<IoVariable>, primitive: &[&str]| {
        let mut io = stage_io(inputs, vec![color(0, 0, f4)]);
        io.primitive_inputs = primitive.iter().map(|n| key(n)).collect();
        with_io(Stage::Fragment, io)
    };
    let mesh = LinkInputs::Mesh(Box::default());
    let f = fragment(
        vec![varying("weight", f4, flat), varying("shade", f4, SMOOTH)],
        &["weight"],
    );
    let l = link(&[&m, &f], &mesh, &p).expect("link");
    assert_eq!(
        l.table.locations,
        BTreeMap::from([
            (key("shade"), (0, 0)),
            (key("id"), (1, 0)),
            (key("weight"), (2, 0)),
            (VaryingKey::color(0, 0), (0, 0)),
        ])
    );
    let needs = |r: Result<Linked, Error>| {
        assert_eq!(
            r.err(),
            Some(Error::NeedsInput(RuntimeInput::PrimitiveVaryings))
        )
    };
    // Compiled for another pipeline: reading the primitive's weight per vertex, or the vertex's
    // shade per primitive.
    needs(link(
        &[&m, &fragment(vec![varying("weight", f4, flat)], &[])],
        &mesh,
        &p,
    ));
    needs(link(
        &[&m, &fragment(vec![varying("shade", f4, flat)], &["shade"])],
        &mesh,
        &p,
    ));
    // A render pipeline's fragment function reads nothing per primitive.
    let v = with_io(
        Stage::Vertex,
        stage_io(vec![], vec![varying("weight", f4, None)]),
    );
    needs(link(&[&v, &f], &render(), &p));
    // An input the mesh function does not write, or writes with another type.
    for input in [varying("other", f4, SMOOTH), varying("id", f4, flat)] {
        let primitive: &[&str] = if input.key == key("id") { &["id"] } else { &[] };
        let r = link(&[&m, &fragment(vec![input], primitive)], &mesh, &p);
        assert_eq!(
            r.err(),
            Some(Unsupported::not_representable(Feature::UnmatchedVarying).into())
        );
    }
}

/// `push_descriptor_eligible` is set 0 fitting `maxPushDescriptors` on a profile with push
/// descriptors; the layout never splits.
#[test]
fn push_descriptor_eligibility_follows_the_profile() {
    let mut p = profile("moltenvk-m3");
    let k = Stage::Kernel;
    let descs = |n: u32| {
        (0..n)
            .map(|i| {
                StageDescriptor::new(
                    res(k, ResourceKind::Buffer, i),
                    vk::DescriptorType::STORAGE_BUFFER,
                    1,
                )
            })
            .collect::<Vec<_>>()
    };
    p.fields.descriptors.push_descriptor = true;
    p.fields.limits.max_push_descriptors = 4;
    for (n, eligible) in [(0, true), (4, true), (5, false)] {
        let a = artifact(iface(k, descs(n), vec![]), None);
        let l = link(&[&a], &LinkInputs::compute(), &p).expect("link");
        assert_eq!(l.layout.push_descriptor_eligible, eligible, "{n}");
        assert_eq!(l.layout.sets.len(), 1);
        assert_eq!(l.layout.sets[0].bindings.len(), n as usize);
    }
    p.fields.descriptors.push_descriptor = false;
    let a = artifact(iface(k, descs(1), vec![]), None);
    let l = link(&[&a], &LinkInputs::compute(), &p).expect("link");
    assert!(!l.layout.push_descriptor_eligible);
}

/// An acceleration structure binding is an `ACCELERATION_STRUCTURE_KHR` descriptor fed by its
/// buffer slot and level, counted against `maxDescriptorSetAccelerationStructures` only.
#[test]
fn acceleration_structures_bind_at_their_buffer_slot() {
    use crate::pipeline::sym::AccelerationStructureLevel as Level;
    let k = Stage::Kernel;
    let descs = |n: u32| {
        (0..n)
            .map(|i| {
                StageDescriptor::new(
                    res(k, ResourceKind::AccelerationStructure(Level::Instance), i),
                    vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                    1,
                )
            })
            .collect::<Vec<_>>()
    };
    let p = profile("lavapipe");
    let a = artifact(iface(k, descs(1), vec![]), None);
    let l = link(&[&a], &LinkInputs::compute(), &p).expect("link");
    let b = &l.layout.sets[0].bindings[0];
    assert_eq!(b.ty, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR);
    assert_eq!(
        b.source,
        BindingSource::AccelerationStructure {
            slot: SlotRef {
                stage: k,
                class: SlotClass::Buffer,
                index: 0,
            },
            level: Level::Instance,
        }
    );
    let mut small = p.clone();
    small
        .fields
        .limits
        .max_descriptor_set_acceleration_structures = 2;
    let a = artifact(iface(k, descs(3), vec![]), None);
    let e = link(&[&a], &LinkInputs::compute(), &small).expect_err("over the set limit");
    let crate::error::Error::Unsupported(u) = e else {
        panic!("{e:?}")
    };
    assert!(
        matches!(
            u.needs,
            crate::error::Needs::AnyOf(ref alts) if alts.iter().flatten().any(|c| matches!(
                c,
                crate::error::Capability::Limit {
                    limit: VkLimit::MaxDescriptorSetAccelerationStructures,
                    required: 3,
                    available: 2,
                }
            ))
        ),
        "{u:?}"
    );
}

/// A 2D float heap class, sampled or storage (formatless).
fn heap_class(access: crate::pipeline::sym::ImageAccess) -> crate::pipeline::sym::HandleClass {
    use crate::pipeline::sym::{HandleClass, ImageDim, ImageHandleClass, SampledType};
    use crate::pipeline::sym::{ImageAccess, StorageFormatClass};
    HandleClass::Image(ImageHandleClass {
        dim: ImageDim::D2,
        arrayed: false,
        multisampled: false,
        sampled_type: SampledType::Float,
        depth_compare: false,
        access,
        format: match access {
            ImageAccess::Storage => StorageFormatClass::Unknown,
            _ => StorageFormatClass::NotStorage,
        },
    })
}

/// Heap arrays are set 1, one binding per class in class order, shared by the stages that declare
/// the class, `PARTIALLY_BOUND | UPDATE_AFTER_BIND` in an `UPDATE_AFTER_BIND_POOL` set, and
/// `UPDATE_UNUSED_WHILE_PENDING` on a profile with the feature (the M3 Max's), all of the heap
/// capacity; set 0 is numbered as before and the handle table is a set-0 storage buffer.
#[test]
fn heap_arrays_are_set_one_at_the_heap_capacity() {
    use crate::pipeline::sym::ImageAccess;
    use vk::DescriptorType as T;
    let p = profile("moltenvk-m3");
    let (v, fr) = (Stage::Vertex, Stage::Fragment);
    let (sampled, storage) = (
        heap_class(ImageAccess::Sampled),
        heap_class(ImageAccess::Storage),
    );
    let table = |s| {
        StageDescriptor::new(
            res(s, ResourceKind::Synthetic(SyntheticBinding::HandleTable), 0),
            T::STORAGE_BUFFER,
            1,
        )
    };
    let vs = iface(
        v,
        vec![
            table(v),
            StageDescriptor::new(res(v, ResourceKind::Heap(sampled), 0), T::SAMPLED_IMAGE, 0),
        ],
        vec![],
    );
    let fs = iface(
        fr,
        vec![
            table(fr),
            StageDescriptor::new(res(fr, ResourceKind::Texture, 1), T::SAMPLED_IMAGE, 1),
            StageDescriptor::new(res(fr, ResourceKind::Heap(sampled), 0), T::SAMPLED_IMAGE, 0),
            StageDescriptor::new(res(fr, ResourceKind::Heap(storage), 0), T::STORAGE_IMAGE, 0),
        ],
        vec![],
    );
    let (va, fa) = (artifact(vs, None), artifact(fs, None));
    let l = link(&[&va, &fa], &render(), &p).expect("link");
    assert_eq!(l.layout.sets.len(), 2);
    let (set0, set1) = (&l.layout.sets[0], &l.layout.sets[1]);
    assert_eq!(set0.bindings.len(), 3);
    assert!(set0.flags.is_empty());
    assert_eq!(set1.set, 1);
    assert_eq!(
        set1.flags,
        vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL
    );
    let order = [sampled, storage];
    assert!(sampled < storage);
    let lim = &p.fields().limits;
    // Fragment set 0 holds the table and one sampled image; the fragment arrays split what
    // remains of each limit after the reserved color attachments.
    let capacity = [
        u64::from(lim.max_descriptor_set_update_after_bind_sampled_images) - 1,
        u64::from(lim.max_per_stage_descriptor_update_after_bind_sampled_images) - 1,
        u64::from(lim.max_descriptor_set_update_after_bind_storage_images),
        u64::from(lim.max_per_stage_descriptor_update_after_bind_storage_images),
        (u64::from(lim.max_per_stage_update_after_bind_resources) - 2 - 8) / 2,
        u64::from(crate::memory::handles::TABLE_CAPACITY),
    ]
    .into_iter()
    .min()
    .unwrap();
    for (n, b) in set1.bindings.iter().enumerate() {
        assert_eq!(b.binding, n as u32);
        assert_eq!(b.source, BindingSource::Heap(order[n]));
        assert_eq!(u64::from(b.count), capacity);
        assert!(super::heap_unused_while_pending(&p));
        assert_eq!(
            b.flags,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
        );
    }
    assert_eq!(set1.bindings[0].ty, T::SAMPLED_IMAGE);
    assert_eq!(
        set1.bindings[0].stages,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
    );
    assert_eq!(set1.bindings[1].stages, vk::ShaderStageFlags::FRAGMENT);
    // Both stages' symbols of the shared class name one binding.
    assert_eq!(
        l.table.bindings[&res(v, ResourceKind::Heap(sampled), 0)],
        (1, 0)
    );
    assert_eq!(
        l.table.bindings[&res(fr, ResourceKind::Heap(sampled), 0)],
        (1, 0)
    );
    assert_eq!(
        l.table.bindings[&res(fr, ResourceKind::Heap(storage), 0)],
        (1, 1)
    );
    // Set-0 bindings keep set 0.
    assert!(l
        .table
        .bindings
        .iter()
        .filter(|(s, _)| !matches!(s.kind, ResourceKind::Heap(_)))
        .all(|(_, (set, _))| *set == 0));
}

/// The heap capacity is bounded by the update-after-bind limits once set 0 is counted: a limit
/// with room for fewer than one element per array is a typed limit refusal, and a profile
/// without the layout's features refuses the heap naming them.
#[test]
fn the_heap_capacity_follows_the_update_after_bind_limits() {
    use crate::pipeline::sym::ImageAccess;
    use vk::DescriptorType as T;
    let k = Stage::Kernel;
    let sampled = heap_class(ImageAccess::Sampled);
    let heap_iface = |extra: u32| {
        let mut d: Vec<StageDescriptor> = (0..extra)
            .map(|i| StageDescriptor::new(res(k, ResourceKind::Texture, i), T::SAMPLED_IMAGE, 1))
            .collect();
        d.push(StageDescriptor::new(
            res(k, ResourceKind::Heap(sampled), 0),
            T::SAMPLED_IMAGE,
            0,
        ));
        artifact(iface(k, d, vec![]), None)
    };
    let mut p = profile("moltenvk-m3");
    p.fields
        .limits
        .max_per_stage_descriptor_update_after_bind_sampled_images = 10;
    let l = link(&[&heap_iface(3)], &LinkInputs::compute(), &p).expect("link");
    assert_eq!(l.layout.sets[1].bindings[0].count, 7);
    p.fields
        .limits
        .max_per_stage_descriptor_update_after_bind_sampled_images = 3;
    assert!(matches!(
        link(&[&heap_iface(3)], &LinkInputs::compute(), &p),
        Err(Error::Unsupported(Unsupported {
            feature: Feature::DescriptorCount,
            needs: crate::error::Needs::AnyOf(_),
            ..
        }))
    ));
    let mut p = profile("moltenvk-m3");
    p.fields.descriptors.update_after_bind_sampled_image = false;
    match link(&[&heap_iface(0)], &LinkInputs::compute(), &p) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::ArgumentBufferTextures);
            assert_eq!(
                u.needs,
                crate::error::Needs::AnyOf(vec![vec![Capability::Feature(
                    VkFeature::DescriptorBindingSampledImageUpdateAfterBind
                )]])
            );
        }
        other => panic!("{other:?}"),
    }
}

/// A combined image sampler whose sampler is the runtime `[[sampler(2)]]`.
fn combined_kind() -> ResourceKind {
    ResourceKind::CombinedTextureSampler(CombinedSampler::new(SamplerRef::Slot(2), false, false))
}

/// A combined image sampler's source names its texture slot and its sampler; a constexpr sampler is
/// baked into the layout for its use, a runtime one is not.
#[test]
fn combined_image_samplers_name_their_texture_and_sampler() {
    let p = profile("moltenvk-m3");
    let k = Stage::Kernel;
    let words = [34_901_797_601_017_929, 0];
    let runtime = CombinedSampler::new(SamplerRef::Slot(2), false, false);
    let constexpr = CombinedSampler::new(SamplerRef::Constexpr(words), true, false);
    let d = |c| {
        StageDescriptor::new(
            res(k, ResourceKind::CombinedTextureSampler(c), 7),
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            1,
        )
    };
    let a = artifact(iface(k, vec![d(runtime), d(constexpr)], vec![]), None);
    let l = link(&[&a], &LinkInputs::compute(), &p).expect("link");
    let texture = SlotRef {
        stage: k,
        class: SlotClass::Texture,
        index: 7,
    };
    let b = &l.layout.sets[0].bindings;
    assert_eq!(b.len(), 2);
    let by = |c: CombinedSampler| {
        b.iter()
            .find(|x| {
                x.source
                    == BindingSource::CombinedTextureSampler {
                        texture,
                        sampler: c,
                    }
            })
            .expect("binding")
    };
    assert_eq!(by(runtime).immutable_sampler, None);
    let baked = by(constexpr).immutable_sampler.expect("immutable sampler");
    assert_eq!(
        Some(baked),
        crate::lower::sampler::constexpr_desc(words, constexpr.sampler_use(), &p).ok()
    );
    assert_eq!(baked.compare.is_some(), constexpr.compare);
}

/// A combined image-sampler descriptor type on any other symbol, or the combined symbol with any
/// other type, is a contract violation.
#[test]
fn combined_descriptors_and_symbols_must_agree() {
    let p = profile("moltenvk-m3");
    let k = Stage::Kernel;
    for d in [
        StageDescriptor::new(
            res(k, ResourceKind::Texture, 0),
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            1,
        ),
        StageDescriptor::new(
            res(k, combined_kind(), 0),
            vk::DescriptorType::SAMPLED_IMAGE,
            1,
        ),
    ] {
        let a = artifact(iface(k, vec![d], vec![]), None);
        assert!(matches!(
            link(&[&a], &LinkInputs::compute(), &p),
            Err(Error::Internal(_))
        ));
    }
}

// ---- Stage-interface locations ----

/// A stage artifact of `io` whose module declares exactly its variables.
pub(super) fn io_stage(stage: Stage, io: StageIo) -> FunctionArtifact {
    let model = match stage {
        Stage::Vertex => ExecModel::Vertex,
        Stage::Fragment => ExecModel::Fragment,
        Stage::Kernel => ExecModel::GLCompute,
        Stage::Object => ExecModel::TaskEXT,
        Stage::Mesh => ExecModel::MeshEXT,
    };
    let m = io_module(model, &io);
    from_module(stage, m, io)
}

/// Attributes are dense in attribute order, colors sit at their attachment, vertex outputs are
/// dense in output order and each fragment input takes its key's location; alone, a fragment
/// function numbers its inputs in declaration order.
#[test]
fn locations_follow_the_stage_io_contract() {
    let p = profile("moltenvk-m3");
    let f4 = io_ty(IoNumeric::Float, 4);
    let u1 = io_ty(IoNumeric::Uint, 1);
    let vs = io_stage(
        Stage::Vertex,
        stage_io(
            vec![attr(7, f4), attr(2, u1)],
            vec![
                varying("a", f4, None),
                varying("b", f4, None),
                varying("c", u1, None),
            ],
        ),
    );
    let fs = io_stage(
        Stage::Fragment,
        stage_io(
            vec![
                varying("c", u1, Some(Interpolation::Flat)),
                varying("a", f4, SMOOTH),
            ],
            vec![color(3, 0, f4), color(0, 1, f4)],
        ),
    );
    let mut d = crate::pipeline::descriptor::VertexDescriptor::new();
    d.layouts.insert(
        0,
        crate::pipeline::descriptor::VertexBufferLayoutDescriptor::new(Some(32)),
    );
    for (index, format, offset) in [
        (2, crate::metal::VertexFormat::UInt, 0),
        (7, crate::metal::VertexFormat::Float4, 16),
    ] {
        d.attributes.insert(
            index,
            crate::pipeline::descriptor::VertexAttributeDescriptor::new(format, offset, 0),
        );
    }
    let mut r = RenderLinkInputs::new();
    r.vertex_descriptor = Some(d);
    let l = link(&[&vs, &fs], &LinkInputs::Render(Box::new(r)), &p).expect("link");
    let located: Vec<(u32, vk::Format)> = l
        .vertex_input
        .attributes
        .iter()
        .map(|a| (a.location, a.format))
        .collect();
    assert_eq!(
        located,
        vec![
            (0, vk::Format::R32_UINT),
            (1, vk::Format::R32G32B32A32_SFLOAT)
        ]
    );
    assert_eq!(
        l.table.locations,
        BTreeMap::from([
            (VaryingKey::attribute(2), (0, 0)),
            (VaryingKey::attribute(7), (1, 0)),
            (key("a"), (0, 0)),
            (key("b"), (1, 0)),
            (key("c"), (2, 0)),
            (VaryingKey::color(3, 0), (3, 0)),
            (VaryingKey::color(0, 1), (0, 0)),
        ])
    );
    for a in [&vs, &fs] {
        let words = crate::pipeline::relocate::relocate(a, &l).expect("relocate");
        let opts = crate::spirv::testing::ValOptions::for_requirements(&a.requirements);
        crate::spirv::testing::spirv_val_with(
            &words,
            crate::spirv::testing::TargetEnv::Vulkan1_2,
            opts,
        )
        .unwrap_or_else(|e| panic!("spirv-val: {e}"));
    }
    let alone = link(&[&fs], &LinkInputs::standalone(), &p).expect("alone");
    assert_eq!(alone.table.locations[&key("c")], (0, 0));
    assert_eq!(alone.table.locations[&key("a")], (1, 0));
    // A missing producer or a type mismatch is Metal's link error.
    for input in [varying("zz", f4, SMOOTH), varying("c", f4, SMOOTH)] {
        let fs = io_stage(Stage::Fragment, stage_io(vec![input], vec![]));
        assert_eq!(
            link(&[&vs, &fs], &render(), &p).unwrap_err(),
            Error::Unsupported(Unsupported::not_representable(Feature::UnmatchedVarying))
        );
    }
}

/// The raw viewport index is a varying of its own key, numbered with the vertex outputs and read
/// flat; a fragment function reads it exactly behind a vertex function that writes the index
/// (`ViewportIndex`). One compiled for the other kind of vertex function, which reads the builtin
/// behind such a function or the varying behind another, is `NeedsInput(ViewportIndex)`.
#[test]
fn the_raw_viewport_index_links_behind_a_vertex_function_that_writes_it() {
    use crate::error::RuntimeInput;
    use crate::pipeline::stage_io::StageBuiltin;
    let p = profile("moltenvk-m3");
    let f4 = io_ty(IoNumeric::Float, 4);
    let raw = IoVariable {
        key: VaryingKey::viewport_index(),
        role: IoRole::ViewportIndex,
        ty: io_ty(IoNumeric::Uint, 1),
    };
    let vertex = |writes: bool| {
        let mut io = stage_io(vec![], vec![varying("a", f4, None)]);
        if writes {
            io.outputs.push(raw.clone());
            io.builtins.push(StageBuiltin::ViewportIndex);
        }
        io_stage(Stage::Vertex, io)
    };
    let fragment = |inputs: Vec<IoVariable>, builtin: bool| {
        let mut io = stage_io(inputs, vec![color(0, 0, f4)]);
        if builtin {
            io.builtins.push(StageBuiltin::ViewportIndex);
        }
        io_stage(Stage::Fragment, io)
    };
    let (writes, plain) = (vertex(true), vertex(false));
    let reads_varying = fragment(vec![raw.clone(), varying("a", f4, SMOOTH)], false);
    let l = link(&[&writes, &reads_varying], &render(), &p).expect("link");
    assert_eq!(l.table.locations[&key("a")], (0, 0));
    assert_eq!(l.table.locations[&VaryingKey::viewport_index()], (1, 0));
    for a in [&writes, &reads_varying] {
        let words = crate::pipeline::relocate::relocate(a, &l).expect("relocate");
        let opts = crate::spirv::testing::ValOptions::for_requirements(&a.requirements);
        crate::spirv::testing::spirv_val_with(
            &words,
            crate::spirv::testing::TargetEnv::Vulkan1_2,
            opts,
        )
        .unwrap_or_else(|e| panic!("spirv-val: {e}"));
    }
    let reads_builtin = fragment(vec![], true);
    let reads_nothing = fragment(vec![], false);
    for (v, f) in [
        (&plain, &reads_builtin),
        (&writes, &reads_nothing),
        (&plain, &reads_nothing),
    ] {
        link(&[v, f], &render(), &p).expect("link");
    }
    for (v, f) in [(&writes, &reads_builtin), (&plain, &reads_varying)] {
        assert_eq!(
            link(&[v, f], &render(), &p).unwrap_err(),
            Error::NeedsInput(RuntimeInput::ViewportIndex)
        );
    }
}

/// Over the interface limits the varyings are packed into shared locations by numeric class and
/// interpolation, fragment reads first; the packed modules pass `spirv-val`. Beyond what packing
/// fits, the pipeline is refused naming the limit.
#[test]
fn varyings_pack_into_components_when_the_limits_need_it() {
    let mut p = profile("moltenvk-m3");
    let f = |n| io_ty(IoNumeric::Float, n);
    let u = |n| io_ty(IoNumeric::Uint, n);
    let flat = Some(Interpolation::Flat);
    let outputs = vec![
        varying("a", f(1), None),
        varying("b", f(2), None),
        varying("c", u(1), None),
        varying("d", f(3), None),
        varying("e", f(1), None),
        varying("f", u(2), None),
    ];
    let vs = io_stage(Stage::Vertex, stage_io(vec![], outputs));
    let inputs = vec![
        varying("a", f(1), SMOOTH),
        varying("c", u(1), flat),
        varying("d", f(3), SMOOTH),
        varying("f", u(2), flat),
    ];
    let fs = io_stage(Stage::Fragment, stage_io(inputs, vec![]));
    // Dense: six locations out, fragment reads up to location 5.
    p.fields.limits.max_vertex_output_components = 6 * 4;
    p.fields.limits.max_fragment_input_components = 6 * 4;
    let dense = link(&[&vs, &fs], &render(), &p).expect("dense");
    assert!(dense.table.locations.values().all(|l| l.1 == 0));
    // Four locations: packed. Reads first by decreasing width: d(3) a(1) | f(2) c(1) (uint,
    // flat); then b(2) e(1) as unread float.
    p.fields.limits.max_vertex_output_components = 4 * 4;
    p.fields.limits.max_fragment_input_components = 2 * 4;
    let l = link(&[&vs, &fs], &render(), &p).expect("packed");
    assert_eq!(
        l.table.locations,
        BTreeMap::from([
            (key("d"), (0, 0)),
            (key("a"), (0, 3)),
            (key("f"), (1, 0)),
            (key("c"), (1, 2)),
            (key("b"), (2, 0)),
            (key("e"), (2, 2)),
        ])
    );
    for a in [&vs, &fs] {
        let words = crate::pipeline::relocate::relocate(a, &l).expect("relocate");
        let opts = crate::spirv::testing::ValOptions::for_requirements(&a.requirements);
        crate::spirv::testing::spirv_val_with(
            &words,
            crate::spirv::testing::TargetEnv::Vulkan1_2,
            opts,
        )
        .unwrap_or_else(|e| panic!("packed spirv-val: {e}"));
    }
    // Even packed, three locations are needed out.
    p.fields.limits.max_vertex_output_components = 2 * 4;
    match link(&[&vs, &fs], &render(), &p) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::InterfaceComponents);
            assert_eq!(
                u.needs,
                crate::error::Needs::AnyOf(vec![vec![Capability::Limit {
                    limit: VkLimit::MaxVertexOutputComponents,
                    required: 12,
                    available: 8
                }]])
            );
        }
        other => panic!("{other:?}"),
    }
    // A module without relocatable components is never packed: the dense refusal stands.
    let mut b = ModuleBuilder::new();
    let f4 = f32_ty(&mut b);
    for name in ["x", "y"] {
        b.io(key(name), IoDir::Output, f4, IoDecor::default())
            .expect("io");
    }
    let unpackable = from_module(
        Stage::Vertex,
        finish(b, ExecModel::Vertex),
        stage_io(
            vec![],
            vec![varying("x", f(1), None), varying("y", f(1), None)],
        ),
    );
    p.fields.limits.max_vertex_output_components = 4;
    match link(&[&unpackable], &LinkInputs::standalone(), &p) {
        Err(Error::Unsupported(u)) => assert_eq!(
            u.needs,
            crate::error::Needs::AnyOf(vec![vec![Capability::Limit {
                limit: VkLimit::MaxVertexOutputComponents,
                required: 8,
                available: 4
            }]])
        ),
        other => panic!("{other:?}"),
    }
}

/// Link packs as little as the limits need. When packing the unread vertex outputs is enough, the
/// varyings the fragment stage reads keep whole locations at component 0 (no interpolated input
/// shares a location), and only the vertex module's unread outputs need a relocatable `Component`.
#[test]
fn unread_outputs_pack_before_read_varyings() {
    let mut p = profile("moltenvk-m3");
    let f = |n| io_ty(IoNumeric::Float, n);
    let u = |n| io_ty(IoNumeric::Uint, n);
    let outputs = vec![
        varying("b", f(1), None),
        varying("a", f(1), None),
        varying("c", f(2), None),
        varying("d", f(1), None),
        varying("e", u(1), None),
        varying("g", f(3), None),
    ];
    let vs = io_stage(Stage::Vertex, stage_io(vec![], outputs));
    let inputs = vec![varying("a", f(1), SMOOTH), varying("g", f(3), SMOOTH)];
    let fs = io_stage(Stage::Fragment, stage_io(inputs.clone(), vec![]));
    // Dense needs six locations out.
    p.fields.limits.max_vertex_output_components = 4 * 4;
    p.fields.limits.max_fragment_input_components = 4 * 4;
    let l = link(&[&vs, &fs], &render(), &p).expect("unread outputs packed");
    // Reads first, whole, in output order: a(0) g(1); then the unread by decreasing width,
    // first fit per numeric class: c(2) b d | e (uint).
    assert_eq!(
        l.table.locations,
        BTreeMap::from([
            (key("a"), (0, 0)),
            (key("g"), (1, 0)),
            (key("c"), (2, 0)),
            (key("b"), (2, 2)),
            (key("d"), (2, 3)),
            (key("e"), (3, 0)),
        ])
    );
    for a in [&vs, &fs] {
        let words = crate::pipeline::relocate::relocate(a, &l).expect("relocate");
        let opts = crate::spirv::testing::ValOptions::for_requirements(&a.requirements);
        crate::spirv::testing::spirv_val_with(
            &words,
            crate::spirv::testing::TargetEnv::Vulkan1_2,
            opts,
        )
        .unwrap_or_else(|e| panic!("packed spirv-val: {e}"));
    }
    // A fragment module whose inputs carry no relocatable `Component` still links: its inputs
    // are never moved off component 0.
    let mut b = ModuleBuilder::new();
    let f32_ = f32_ty(&mut b);
    let f3 = b.ty(Type::Vector { elem: f32_, len: 3 }).expect("vec3");
    b.io(key("a"), IoDir::Input, f32_, IoDecor::default())
        .expect("a");
    b.io(key("g"), IoDir::Input, f3, IoDecor::default())
        .expect("g");
    let fixed = from_module(
        Stage::Fragment,
        finish(b, ExecModel::Fragment),
        stage_io(inputs, vec![]),
    );
    let l2 = link(&[&vs, &fixed], &render(), &p).expect("fixed fragment inputs");
    assert_eq!(l2.table.locations, l.table.locations);
    // Without room for the reads' whole locations, everything packs (the fragment module must
    // then relocate its components too).
    p.fields.limits.max_vertex_output_components = 3 * 4;
    let l3 = link(&[&vs, &fs], &render(), &p).expect("everything packed");
    assert_eq!(l3.table.locations[&key("g")], (0, 0));
    assert_eq!(l3.table.locations[&key("a")], (0, 3));
    assert!(link(&[&vs, &fixed], &render(), &p).is_err());
}

/// A role outside its stage (a kernel with stage IO, a vertex color output) is an invariant
/// failure; too many attributes is refused naming the limit.
#[test]
fn stage_io_roles_and_attribute_limits_are_checked() {
    let mut p = profile("moltenvk-m3");
    let f4 = io_ty(IoNumeric::Float, 4);
    let mut k = iface(Stage::Kernel, vec![], vec![]);
    k.io = stage_io(vec![], vec![varying("x", f4, None)]);
    let mut v = iface(Stage::Vertex, vec![], vec![]);
    v.io = stage_io(vec![], vec![color(0, 0, f4)]);
    for i in [k, v] {
        let a = artifact(i, None);
        assert!(matches!(
            link(&[&a], &LinkInputs::standalone(), &p),
            Err(Error::Internal(_))
        ));
    }
    p.fields.limits.max_vertex_input_attributes = 1;
    let mut v = iface(Stage::Vertex, vec![], vec![]);
    v.io = stage_io(vec![attr(0, f4), attr(5, f4)], vec![]);
    let a = artifact(v, None);
    match link(&[&a], &LinkInputs::standalone(), &p) {
        Err(Error::Unsupported(u)) => assert_eq!(
            u.needs,
            crate::error::Needs::AnyOf(vec![vec![Capability::Limit {
                limit: VkLimit::MaxVertexInputAttributes,
                required: 2,
                available: 1
            }]])
        ),
        other => panic!("{other:?}"),
    }
}

/// Two stages declaring one specialization constant differently were compiled for another
/// pipeline: a descriptor error naming the key.
#[test]
fn specialization_disagreements_are_descriptor_errors() {
    let p = profile("moltenvk-m3");
    let vs = iface(
        Stage::Vertex,
        vec![],
        vec![StageSpec::new(SpecKey::SampleCount, 4, 1)],
    );
    let fs = iface(
        Stage::Fragment,
        vec![],
        vec![StageSpec::new(SpecKey::SampleCount, 4, 4)],
    );
    let (va, fa) = (artifact(vs, None), artifact(fs, None));
    assert_eq!(
        link(&[&va, &fa], &render(), &p).unwrap_err(),
        Error::InvalidDescriptor(DescriptorError::SpecializationConflict {
            key: SpecKey::SampleCount
        })
    );
}

/// Render inputs with Metal's defaults.
pub(super) fn render() -> LinkInputs {
    LinkInputs::Render(Box::default())
}

/// A module of words `body` after the header of SPIR-V `version`.
fn module_words(version: u32, body: &[u32]) -> Vec<u32> {
    [crate::spv::MAGIC_NUMBER, version, 0, 16, 0]
        .into_iter()
        .chain(body.iter().copied())
        .collect()
}

/// A binding declared by two variables, `%5` and `%10`, counts the variables its entry point
/// lists (`DescriptorBinding::variables`): `%10` alone where the entry point uses it alone, both
/// where it uses both, and both in a module before SPIR-V 1.4, whose entry point lists its inputs
/// and outputs alone.
#[test]
fn a_binding_counts_the_variables_its_entry_point_uses() {
    let slot = res(Stage::Fragment, ResourceKind::Buffer, 0);
    let mut table = LinkTable::default();
    table.bindings.insert(slot, (0, 3));
    let entry_point = |interface: &[u32]| {
        // `OpEntryPoint Fragment %1 "main"`, then the interface.
        let len = 5 + interface.len() as u32;
        let mut words = vec![
            len << 16 | crate::spv::Op::EntryPoint as u32,
            4,
            1,
            u32::from_le_bytes(*b"main"),
            0,
        ];
        words.extend_from_slice(interface);
        words
    };
    let decorate = |id: u32| [4 << 16 | crate::spv::Op::Decorate as u32, id, 33, 0];
    let module = |version: u32, interface: &[u32]| {
        let mut body = entry_point(interface);
        body.extend(decorate(5));
        body.extend(decorate(10));
        let words = module_words(version, &body);
        // Each `Binding` literal: the last word of each decoration.
        let at = |n: usize| (words.len() - 4 * (1 - n) - 1) as u32;
        let relocs = vec![
            (Symbol::Binding(slot), at(0)),
            (Symbol::Binding(slot), at(1)),
        ];
        used_variables(&words, &relocs, &table)
    };
    let used = |counts: BTreeMap<(u32, u32), u32>| counts.get(&(0, 3)).copied();
    assert_eq!(used(module(0x0001_0500, &[10, 17])), Some(1));
    assert_eq!(used(module(0x0001_0500, &[5, 10, 17])), Some(2));
    assert_eq!(used(module(0x0001_0300, &[17])), Some(2));
    assert_eq!(used(module(0x0001_0500, &[17])), None);
}
