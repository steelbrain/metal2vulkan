//! `spirv-val` sweep, continued: a mesh pipeline's task and mesh entries and the fragment input
//! that reads a mesh entry's primitive output (`SPV_EXT_mesh_shader`), and the builder's and
//! verifier's refusals of their misuse.

use super::{assert_spirv_valid, cval, float, int, key, vec};
use crate::budget::{Budget, Meter};
use crate::device::requirements::{SpirvCapability, SpirvExtension, VkExtension, VkFeature};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BuildError, EntryRule, GlobalRule};
use crate::spirv::ids::{FuncId, GlobalId, TypeId, Value};
use crate::spirv::ir::{
    BinOp, ExecMode, ExecModel, IoDecor, IoDir, MemAccess, Scope, Semantics, StorageClass, Type,
    TypedModule,
};
use crate::spv::BuiltIn;

/// `[elem; n]`, unstrided.
fn array(b: &mut ModuleBuilder, elem: TypeId, n: u32) -> TypeId {
    let u32_ = int(b, 32, false);
    let len = super::scalar(b, u32_, u64::from(n));
    b.ty(Type::Array {
        elem,
        len,
        stride: None,
    })
    .expect("array")
}

/// A void function around `body`, finished.
fn entry_fn(b: &mut ModuleBuilder, body: impl FnOnce(&mut FunctionBuilder<'_>)) -> FuncId {
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    body(&mut f);
    if f.current_block().is_some() {
        f.ret(None).expect("return");
    }
    f.finish().expect("finish")
}

/// The payload: four `u32` words in `TaskPayloadWorkgroupEXT`.
fn payload(b: &mut ModuleBuilder) -> GlobalId {
    let u32_ = int(b, 32, false);
    let words = array(b, u32_, 4);
    b.plain_global(StorageClass::TaskPayloadWorkgroupEXT, words, None)
        .expect("payload")
}

/// `LocalInvocationIndex`.
fn local_index(f: &mut FunctionBuilder<'_>) -> Value {
    let u32_ = int(f.module(), 32, false);
    let g = f
        .module()
        .builtin(BuiltIn::LocalInvocationIndex, IoDir::Input, u32_)
        .expect("builtin");
    let p = f.module().global_value(g).expect("global");
    f.load(p, MemAccess::default()).expect("load")
}

/// Element `i` of array variable `g`.
fn element(f: &mut FunctionBuilder<'_>, g: GlobalId, i: Value) -> Value {
    let base = f.module().global_value(g).expect("global");
    f.access_chain(base, &[i]).expect("chain")
}

/// A task entry: thread 0 writes the payload, the workgroup synchronizes, and it launches two
/// mesh workgroups that read it.
fn task_module() -> TypedModule {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let words = payload(&mut b);
    let main = entry_fn(&mut b, |f| {
        let i = local_index(f);
        let zero = cval(f, u32_, 0);
        let p = element(f, words, zero);
        f.store(p, i, MemAccess::default()).expect("store");
        f.control_barrier(Scope::Workgroup, Scope::Workgroup, Semantics::RELAXED)
            .expect("barrier");
        let (one, two) = (cval(f, u32_, 1), cval(f, u32_, 2));
        f.emit_mesh_tasks(two, one, one, Some(words))
            .expect("launch");
    });
    b.entry(
        ExecModel::TaskEXT,
        main,
        vec![ExecMode::LocalSize { x: 32, y: 1, z: 1 }],
    )
    .expect("entry");
    b.finalize().expect("finalize")
}

/// The modes of a mesh entry of 3 vertices and 1 triangle.
fn mesh_modes() -> Vec<ExecMode> {
    vec![
        ExecMode::LocalSize { x: 32, y: 1, z: 1 },
        ExecMode::OutputVertices { n: 3 },
        ExecMode::OutputPrimitives { n: 1 },
        ExecMode::OutputTriangles,
    ]
}

/// A mesh entry: it sets its counts first, then thread `i < 3` writes vertex `i`'s position and
/// color, and thread 0 the triangle's indices, primitive id, culled flag, and a per-primitive
/// user value read from the payload.
fn mesh_module() -> TypedModule {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let i32_ = int(&mut b, 32, true);
    let f32_ = float(&mut b, 32);
    let bool_ = b.ty(Type::Bool).expect("bool");
    let vec4 = vec(&mut b, f32_, 4);
    let uvec3 = vec(&mut b, u32_, 3);
    let words = payload(&mut b);
    let (vec4x3, uvec3x1, i32x1, boolx1, u32x1) = (
        array(&mut b, vec4, 3),
        array(&mut b, uvec3, 1),
        array(&mut b, i32_, 1),
        array(&mut b, bool_, 1),
        array(&mut b, u32_, 1),
    );
    let position = b
        .builtin(BuiltIn::Position, IoDir::Output, vec4x3)
        .expect("position");
    let indices = b
        .builtin(BuiltIn::PrimitiveTriangleIndicesEXT, IoDir::Output, uvec3x1)
        .expect("indices");
    let primitive_id = b
        .builtin(BuiltIn::PrimitiveId, IoDir::Output, i32x1)
        .expect("primitive id");
    let culled = b
        .builtin(BuiltIn::CullPrimitiveEXT, IoDir::Output, boolx1)
        .expect("culled");
    let color = b
        .io(key("color"), IoDir::Output, vec4x3, IoDecor::default())
        .expect("color");
    let tag = b
        .io(key("tag"), IoDir::Output, u32x1, IoDecor::default())
        .expect("tag");
    b.per_primitive(tag).expect("per primitive");
    let main = entry_fn(&mut b, |f| {
        let (three, one) = (cval(f, u32_, 3), cval(f, u32_, 1));
        f.set_mesh_outputs(three, one).expect("counts");
        let i = local_index(f);
        let zero = cval(f, u32_, 0);
        let fone = cval(f, f32_, 1f32.to_bits().into());
        let v = f
            .composite_construct(vec4, &[fone, fone, fone, fone])
            .expect("vec4");
        let in_range = f.binary(BinOp::ULessThan, i, three).expect("cmp");
        let mut ctx = ();
        f.if_(
            &mut ctx,
            in_range,
            Box::new(move |fb: &mut FunctionBuilder<'_>, _: &mut ()| {
                let p = element(fb, position, i);
                fb.store(p, v, MemAccess::default())?;
                let c = element(fb, color, i);
                fb.store(c, v, MemAccess::default())?;
                Ok(crate::spirv::builder::constructs::ArmEnd::FallThrough(
                    vec![],
                ))
            }),
            None,
        )
        .expect("if");
        let two = cval(f, u32_, 2);
        let tri = f
            .composite_construct(uvec3, &[zero, one, two])
            .expect("uvec3");
        let p = element(f, indices, zero);
        f.store(p, tri, MemAccess::default()).expect("indices");
        let id = cval(f, i32_, 0);
        let p = element(f, primitive_id, zero);
        f.store(p, id, MemAccess::default()).expect("id");
        let no = cval(f, bool_, 0);
        let p = element(f, culled, zero);
        f.store(p, no, MemAccess::default()).expect("culled");
        let w = element(f, words, zero);
        let word = f.load(w, MemAccess::default()).expect("payload");
        let p = element(f, tag, zero);
        f.store(p, word, MemAccess::default()).expect("tag");
    });
    b.entry(ExecModel::MeshEXT, main, mesh_modes())
        .expect("entry");
    b.finalize().expect("finalize")
}

/// A fragment entry reading the mesh entry's per-primitive `tag` (flat, as an integer input).
fn per_primitive_fragment() -> TypedModule {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let tag = b
        .io(
            key("tag"),
            IoDir::Input,
            u32_,
            IoDecor {
                interpolation: crate::spirv::ir::Interpolation::Flat,
                ..IoDecor::default()
            },
        )
        .expect("tag");
    b.per_primitive(tag).expect("per primitive");
    let out = b
        .io(key("target"), IoDir::Output, u32_, IoDecor::default())
        .expect("target");
    let main = entry_fn(&mut b, |f| {
        let p = f.module().global_value(tag).expect("in");
        let v = f.load(p, MemAccess::default()).expect("load");
        let q = f.module().global_value(out).expect("out");
        f.store(q, v, MemAccess::default()).expect("store");
    });
    b.entry(ExecModel::Fragment, main, vec![]).expect("entry");
    b.finalize().expect("finalize")
}

fn mesh_shading(m: &TypedModule, feature: VkFeature) {
    let r = &m.requirements;
    assert!(r
        .spirv_capabilities
        .contains(&SpirvCapability::of(crate::spv::Capability::MeshShadingEXT)));
    assert!(r.spirv_extensions.contains(&SpirvExtension::ExtMeshShader));
    assert!(r.features.contains(&feature), "{:?}", r.features);
    assert!(r.extensions.contains(&VkExtension::ExtMeshShader));
}

#[test]
fn a_task_entry_launches_its_mesh_workgroups() {
    let m = task_module();
    mesh_shading(&m, VkFeature::TaskShader);
    assert!(!m.requirements.features.contains(&VkFeature::MeshShader));
    assert_spirv_valid(m);
}

#[test]
fn a_mesh_entry_writes_its_vertices_and_primitives() {
    let m = mesh_module();
    mesh_shading(&m, VkFeature::MeshShader);
    // The mesh entry's per-primitive builtins need no `Geometry` or `ShaderLayer`: its
    // `MeshShadingEXT` enables them.
    assert!(!m.requirements.features.contains(&VkFeature::GeometryShader));
    assert_spirv_valid(m);
}

#[test]
fn a_fragment_reads_a_per_primitive_input() {
    let m = per_primitive_fragment();
    mesh_shading(&m, VkFeature::MeshShader);
    assert_spirv_valid(m);
}

/// A mesh entry states its vertex and primitive counts and its primitive kind; a task or mesh
/// entry its local size.
#[test]
fn a_mesh_entry_without_its_counts_or_kind_is_refused() {
    for drop in [1, 2, 3] {
        let mut b = ModuleBuilder::new();
        let main = entry_fn(&mut b, |_| {});
        let mut modes = mesh_modes();
        modes.remove(drop);
        assert_eq!(
            b.entry(ExecModel::MeshEXT, main, modes),
            Err(BuildError::Entry(EntryRule::MeshModes)),
            "without mode {drop}"
        );
    }
    let mut b = ModuleBuilder::new();
    let main = entry_fn(&mut b, |_| {});
    let mut modes = mesh_modes();
    modes[2] = ExecMode::OutputPrimitives { n: 0 };
    assert_eq!(
        b.entry(ExecModel::MeshEXT, main, modes),
        Err(BuildError::Entry(EntryRule::MeshModes))
    );
    let mut b = ModuleBuilder::new();
    let main = entry_fn(&mut b, |_| {});
    assert_eq!(
        b.entry(ExecModel::TaskEXT, main, vec![]),
        Err(BuildError::Entry(EntryRule::MissingLocalSize))
    );
    let mut b = ModuleBuilder::new();
    let main = entry_fn(&mut b, |_| {});
    assert_eq!(
        b.entry(ExecModel::GLCompute, main, mesh_modes()),
        Err(BuildError::Entry(EntryRule::ModeForModel))
    );
}

/// `OpSetMeshOutputsEXT` is a mesh entry's and `OpEmitMeshTasksEXT` a task entry's; a payload is
/// a `TaskPayloadWorkgroupEXT` variable, at most one per entry.
#[test]
fn mesh_instructions_outside_their_models_are_refused() {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let main = entry_fn(&mut b, |f| {
        let one = cval(f, u32_, 1);
        f.set_mesh_outputs(one, one).expect("counts");
    });
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .expect("entry");
    assert_eq!(
        b.finalize().map(|_| ()),
        Err(BuildError::Entry(EntryRule::InstructionForModel(
            crate::spv::Op::SetMeshOutputsEXT
        )))
    );
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let main = entry_fn(&mut b, |f| {
        let one = cval(f, u32_, 1);
        f.emit_mesh_tasks(one, one, one, None).expect("launch");
    });
    b.entry(ExecModel::MeshEXT, main, mesh_modes())
        .expect("entry");
    assert_eq!(
        b.finalize().map(|_| ()),
        Err(BuildError::Entry(EntryRule::InstructionForModel(
            crate::spv::Op::EmitMeshTasksEXT
        )))
    );
    // A payload that is a `Workgroup` variable.
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let shared = b
        .plain_global(StorageClass::Workgroup, u32_, None)
        .expect("shared");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let one = cval(&mut f, u32_, 1);
    assert_eq!(
        f.emit_mesh_tasks(one, one, one, Some(shared)),
        Err(BuildError::Global(GlobalRule::Payload))
    );
    // Two payloads in one entry's interface.
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let (first, second) = (payload(&mut b), payload(&mut b));
    let main = entry_fn(&mut b, |f| {
        let zero = cval(f, u32_, 0);
        let p = element(f, first, zero);
        f.store(p, zero, MemAccess::default()).expect("store");
        let one = cval(f, u32_, 1);
        f.emit_mesh_tasks(one, one, one, Some(second))
            .expect("launch");
    });
    b.entry(
        ExecModel::TaskEXT,
        main,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .expect("entry");
    assert_eq!(
        b.finalize().map(|_| ()),
        Err(BuildError::Global(GlobalRule::Payload))
    );
}

/// A mesh-only builtin exists only as an array over the primitives, and the verifier refuses a
/// mesh entry's per-element form in another model and a payload outside a task or mesh entry.
#[test]
fn mesh_builtins_and_payloads_outside_a_mesh_pipeline_are_refused() {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let uvec3 = vec(&mut b, u32_, 3);
    assert_eq!(
        b.builtin(BuiltIn::PrimitiveTriangleIndicesEXT, IoDir::Output, uvec3),
        Err(BuildError::Global(GlobalRule::BuiltInType(
            BuiltIn::PrimitiveTriangleIndicesEXT
        )))
    );
    let verify = |m: &TypedModule| {
        crate::spirv::verify::check_structure(m, &mut Meter::new(&Budget::default()))
    };
    // A vertex entry writing a per-vertex array of positions.
    let mut b = ModuleBuilder::new();
    let f32_ = float(&mut b, 32);
    let vec4 = vec(&mut b, f32_, 4);
    let vec4x3 = array(&mut b, vec4, 3);
    let position = b
        .builtin(BuiltIn::Position, IoDir::Output, vec4x3)
        .expect("position");
    let main = entry_fn(&mut b, |f| {
        let u32_ = int(f.module(), 32, false);
        let zero = cval(f, u32_, 0);
        let fone = cval(f, f32_, 1f32.to_bits().into());
        let v = f
            .composite_construct(vec4, &[fone, fone, fone, fone])
            .expect("vec4");
        let p = element(f, position, zero);
        f.store(p, v, MemAccess::default()).expect("store");
    });
    b.entry(ExecModel::Vertex, main, vec![]).expect("entry");
    let m = b.finalize().expect("finalize");
    assert!(verify(&m).is_err(), "a vertex entry's arrayed position");
    // A fragment entry with a payload.
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let words = payload(&mut b);
    let main = entry_fn(&mut b, |f| {
        let zero = cval(f, u32_, 0);
        let p = element(f, words, zero);
        f.load(p, MemAccess::default()).expect("load");
    });
    b.entry(ExecModel::Fragment, main, vec![]).expect("entry");
    let m = b.finalize().expect("finalize");
    assert!(verify(&m).is_err(), "a fragment entry's payload");
    // A mesh entry's position that is not per vertex.
    let mut b = ModuleBuilder::new();
    let f32_ = float(&mut b, 32);
    let vec4 = vec(&mut b, f32_, 4);
    let position = b
        .builtin(BuiltIn::Position, IoDir::Output, vec4)
        .expect("position");
    let main = entry_fn(&mut b, |f| {
        let u32_ = int(f.module(), 32, false);
        let (three, one) = (cval(f, u32_, 3), cval(f, u32_, 1));
        f.set_mesh_outputs(three, one).expect("counts");
        let fone = cval(f, f32_, 1f32.to_bits().into());
        let v = f
            .composite_construct(vec4, &[fone, fone, fone, fone])
            .expect("vec4");
        let p = f.module().global_value(position).expect("global");
        f.store(p, v, MemAccess::default()).expect("store");
    });
    b.entry(ExecModel::MeshEXT, main, mesh_modes())
        .expect("entry");
    let m = b.finalize().expect("finalize");
    assert!(verify(&m).is_err(), "a mesh entry's single position");
}
