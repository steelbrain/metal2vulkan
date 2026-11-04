//! Object and mesh functions as `TaskEXT` and `MeshEXT` entries.
//!
//! The owned fixtures `engine/mesh_order` and `engine/mesh_outputs` hold the object, mesh
//! and fragment functions the Metal probes ran; `engine/mesh_gated` and
//! `engine/mesh_amplified` what Apple's VFX particle meshes declare (members and an
//! amplification id under function constants, a primitive's render target slice and viewport).
//! Each mesh function that lowers compiles, on every committed profile whose device lists
//! `VK_EXT_mesh_shader` (which enables it with `taskShader` and `meshShader`), to a `MeshEXT`
//! entry that passes L3 (`spirv-val` and requirements within the profile); the entry states its
//! counts and topology, initializes its outputs and culls past its count, its threadgroup memory
//! is its workgroup's, and its stage interface lists its user outputs, per vertex or per
//! primitive. What does not lower yet is refused typed. No GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use crate::budget::SPIRV_WORDS;
use crate::device::requirements::{VkExtension, VkFeature};
use crate::device::DeviceProfile;
use crate::error::{Capability, Error, Feature, Needs};
use crate::library::{FunctionArtifact, FunctionConstantValue, FunctionConstantValues};
use crate::metal::DataType;
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::stage_io::{IoNumeric, IoRole, StageBuiltin};
use crate::pipeline::sym::{PushField, SpecKey, Stage};
use crate::pipeline::testing::{engine_dir, llvm_assemble, meter, AssembleError};
use crate::spirv::testing::{check_words_valid, committed_profiles};
use std::sync::Arc;

fn fixture(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
    }
}

fn compile(bc: &[u8], entry: &str, p: &DeviceProfile) -> Result<Arc<FunctionArtifact>, Error> {
    compile_with(bc, entry, &FunctionConstantValues::defaults(), p)
}

fn compile_with(
    bc: &[u8],
    entry: &str,
    values: &FunctionConstantValues,
    p: &DeviceProfile,
) -> Result<Arc<FunctionArtifact>, Error> {
    compile_bitcode(
        bc,
        EntryRef::Name(entry),
        values,
        p,
        &LoweringInputs::new(),
        &mut meter(),
        SPIRV_WORDS,
    )
}

/// Whether committed profile `stem`'s device lists `VK_EXT_mesh_shader`.
fn lists_mesh_shader(stem: &str) -> bool {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles");
    std::fs::read_to_string(dir.join(format!("{stem}.json")))
        .is_ok_and(|t| t.contains("\"VK_EXT_mesh_shader\""))
}

/// The committed profiles that enable `taskShader` and `meshShader`: exactly those whose device
/// lists `VK_EXT_mesh_shader`, which each enables with its mesh limits.
fn mesh_profiles() -> Vec<(String, DeviceProfile)> {
    let all = committed_profiles().expect("committed profiles");
    let mut out = Vec::new();
    for (stem, p) in all {
        let enabled = p.enabled();
        let on = enabled.features.contains(&VkFeature::MeshShader)
            && enabled.features.contains(&VkFeature::TaskShader)
            && enabled.extensions.contains(&VkExtension::ExtMeshShader);
        assert_eq!(on, lists_mesh_shader(&stem), "{stem}: mesh shading enabled");
        if on {
            let l = &p.fields().limits.mesh;
            assert!(
                l.max_mesh_output_vertices >= 256 && l.max_mesh_output_primitives >= 256,
                "{stem}: the extension's minimum output limits: {l:?}"
            );
            out.push((stem, p));
        }
    }
    assert!(!out.is_empty(), "no committed profile enables mesh shading");
    out
}

/// The opcodes of a module's instructions.
fn opcodes(words: &[u32]) -> Vec<u32> {
    let mut ops = Vec::new();
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        ops.push(w & 0xFFFF);
        i += (w >> 16).max(1) as usize;
    }
    ops
}

/// A mesh function without a payload is a `MeshEXT` entry on every profile with the extension,
/// valid, with its outputs initialized before its body and culled past its count after it; its
/// stage interface lists its per-vertex and per-primitive user outputs.
#[test]
fn a_mesh_function_is_a_mesh_entry() {
    let (Some(order), Some(outputs)) = (fixture("mesh_order"), fixture("mesh_outputs")) else {
        return;
    };
    for (stem, p) in mesh_profiles() {
        for (bc, entry) in [
            (&order, "mesh_order_solo"),
            (&outputs, "mesh_outputs_lines"),
            (&outputs, "mesh_outputs_points"),
        ] {
            let a = compile(bc, entry, &p).unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            assert_eq!(a.interface.stage, Stage::Mesh, "{stem} {entry}");
            let r = &a.module.requirements;
            assert!(
                r.extensions.contains(&VkExtension::ExtMeshShader),
                "{stem} {entry}"
            );
            assert!(
                r.features.contains(&VkFeature::MeshShader),
                "{stem} {entry}"
            );
            check_words_valid(&a.module.words, r, &p)
                .unwrap_or_else(|e| panic!("{stem} {entry}: L3: {e}"));
            let ops = opcodes(&a.module.words);
            let set = crate::spv::Op::SetMeshOutputsEXT as u32;
            assert_eq!(
                ops.iter().filter(|o| **o == set).count(),
                1,
                "{stem} {entry}: one OpSetMeshOutputsEXT"
            );
        }
        let a = compile(&order, "mesh_order_solo", &p).expect("mesh_order_solo");
        let io = &a.interface.io;
        let roles: Vec<(&IoRole, IoNumeric)> =
            io.outputs.iter().map(|v| (&v.role, v.ty.numeric)).collect();
        assert_eq!(roles.len(), 3, "{stem}: shade, id, weight");
        assert!(
            roles
                .iter()
                .any(|(r, n)| matches!(r, IoRole::Varying { .. }) && *n == IoNumeric::Float),
            "{stem}: the vertex's shade"
        );
        assert_eq!(
            roles
                .iter()
                .filter(|(r, _)| matches!(r, IoRole::PrimitiveVarying { .. }))
                .count(),
            2,
            "{stem}: the primitive's id and weight"
        );
        assert!(io
            .builtins
            .contains(&StageBuiltin::Position { invariant: false }));
        let lines = compile(&outputs, "mesh_outputs_lines", &p).expect("mesh_outputs_lines");
        assert!(
            lines
                .interface
                .io
                .builtins
                .contains(&StageBuiltin::Position { invariant: true }),
            "{stem}: an invariant position"
        );
    }
}

/// The committed profiles whose device does not list `VK_EXT_mesh_shader`: those that emulate a
/// mesh function.
fn emulating_profiles() -> Vec<(String, DeviceProfile)> {
    let out: Vec<(String, DeviceProfile)> = committed_profiles()
        .expect("committed profiles")
        .into_iter()
        .filter(|(stem, _)| !lists_mesh_shader(stem))
        .collect();
    assert!(
        !out.is_empty(),
        "every committed profile lists the extension"
    );
    for (stem, p) in &out {
        assert!(crate::lower::mesh::emulates(p), "{stem}");
    }
    out
}

/// Whether profile `p` lets a vertex stage write every output `mesh_outputs_points` writes: its
/// render target slice, its viewport and its clip distances.
fn writes_points_outputs(p: &DeviceProfile) -> bool {
    let f = p.fields().fragment;
    f.output_layer
        && f.output_viewport_index
        && p.enabled()
            .features
            .contains(&VkFeature::ShaderClipDistance)
}

/// The execution models of a module's entry points.
fn execution_models(words: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let len = (w >> 16).max(1) as usize;
        if w & 0xFFFF == crate::spv::Op::EntryPoint as u32 {
            out.extend(words.get(i + 1).copied());
        }
        i += len;
    }
    out
}

/// Without the extension a mesh function is emulated, a `GLCompute` entry that writes its
/// threadgroup's records to the records buffer, valid, needing no mesh shading, and whose stage
/// interface states the records' layout; an object function and a mesh function reading its payload
/// are `GLCompute` entries too, their payload workgroup memory and their work buffer bound.
#[test]
fn without_the_extension_mesh_and_object_functions_are_compute_entries() {
    use crate::pipeline::interface::RecordValue;
    let (Some(order), Some(outputs)) = (fixture("mesh_order"), fixture("mesh_outputs")) else {
        return;
    };
    for (stem, p) in emulating_profiles() {
        // (bytecode, entry, the indices of a primitive)
        for (bc, entry, corners) in [
            (&order, "mesh_order_solo", 3),
            (&outputs, "mesh_outputs_lines", 2),
            (&outputs, "mesh_outputs_points", 1),
        ] {
            let what = format!("{stem} {entry}");
            // The points choose their slice and viewport and write clip distances, which the
            // synthesized vertex stage writes as a vertex function's: a device that lacks one
            // refuses them as needing it.
            if entry == "mesh_outputs_points" && !writes_points_outputs(&p) {
                match compile(bc, entry, &p) {
                    Err(Error::Unsupported(u))
                        if matches!(
                            u.feature,
                            Feature::LayeredRendering | Feature::ClipDistance
                        ) => {}
                    other => panic!("{what}: {other:?}"),
                }
                continue;
            }
            let a = compile(bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(a.interface.stage, Stage::Mesh, "{what}");
            let r = &a.module.requirements;
            assert!(
                !r.extensions.contains(&VkExtension::ExtMeshShader)
                    && !r.features.contains(&VkFeature::MeshShader),
                "{what}: {r:?}"
            );
            check_words_valid(&a.module.words, r, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_eq!(
                execution_models(&a.module.words),
                [crate::spv::ExecutionModel::GLCompute as u32],
                "{what}"
            );
            let ops = opcodes(&a.module.words);
            assert!(
                !ops.contains(&(crate::spv::Op::SetMeshOutputsEXT as u32)),
                "{what}"
            );
            assert_eq!(
                a.interface.vk_stage(),
                crate::vk::ShaderStage::COMPUTE,
                "{what}"
            );
            assert!(
                a.interface
                    .descriptors
                    .iter()
                    .any(|d| d.slot == crate::lower::mesh::RECORDS_SLOT
                        && d.ty == crate::vk::DescriptorType::STORAGE_BUFFER),
                "{what}: the records binding"
            );
            let records = a.interface.mesh_records.as_ref().expect("the records");
            assert_eq!(records.corners, corners, "{what}");
            // Every field fits its record, the vertex record holds the position, and no two
            // fields of a record overlap.
            assert!(
                records
                    .fields
                    .iter()
                    .any(|f| f.value == RecordValue::Position && !f.per_primitive),
                "{what}: {records:?}"
            );
            for per_primitive in [false, true] {
                let (start, size) = if per_primitive {
                    (corners, records.primitive_words)
                } else {
                    (0, records.vertex_words)
                };
                let mut words: Vec<(u32, u32)> = records
                    .fields
                    .iter()
                    .filter(|f| f.per_primitive == per_primitive)
                    .map(|f| (f.offset, f.offset + f.words))
                    .collect();
                words.sort_unstable();
                let mut at = start;
                for (from, to) in words {
                    assert!(from >= at && to <= size, "{what}: {records:?}");
                    at = to;
                }
            }
            assert_eq!(
                records.group_words(),
                1 + u64::from(records.vertices) * u64::from(records.vertex_words)
                    + u64::from(records.primitives) * u64::from(records.primitive_words),
                "{what}"
            );
        }
        // An object function and a mesh function reading its payload are compute entries too:
        // the payload is workgroup memory, which the draw's work buffer carries between them.
        for entry in ["mesh_order_object", "mesh_order_mesh"] {
            let what = format!("{stem} {entry}");
            let a = compile(&order, entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let r = &a.module.requirements;
            assert!(
                !r.extensions.contains(&VkExtension::ExtMeshShader),
                "{what}: {r:?}"
            );
            check_words_valid(&a.module.words, r, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_eq!(
                execution_models(&a.module.words),
                [crate::spv::ExecutionModel::GLCompute as u32],
                "{what}"
            );
            assert_eq!(
                variables_of(
                    &a.module.words,
                    crate::spv::StorageClass::TaskPayloadWorkgroupEXT
                ),
                0,
                "{what}"
            );
            assert!(
                a.interface.emulated_payload.is_some_and(|words| words > 0),
                "{what}: {:?}",
                a.interface.emulated_payload
            );
            assert_eq!(
                a.interface.vk_stage(),
                crate::vk::ShaderStage::COMPUTE,
                "{what}"
            );
            let work = match a.interface.stage {
                Stage::Object => crate::lower::mesh::OBJECT_WORK_SLOT,
                _ => crate::lower::mesh::WORK_SLOT,
            };
            assert!(
                a.interface.descriptors.iter().any(|d| d.slot == work),
                "{what}: the work buffer"
            );
        }
    }
}

/// The `OpVariable`s of storage class `class` in a module.
fn variables_of(words: &[u32], class: crate::spv::StorageClass) -> usize {
    let mut n = 0;
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let len = (w >> 16).max(1) as usize;
        if w & 0xFFFF == crate::spv::Op::Variable as u32
            && words.get(i + 3) == Some(&(class as u32))
        {
            n += 1;
        }
        i += len;
    }
    n
}

/// The words of each `OpEmitMeshTasksEXT` in a module: its operand count tells whether it
/// passes a payload.
fn emits(words: &[u32]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let len = (w >> 16).max(1) as usize;
        if w & 0xFFFF == crate::spv::Op::EmitMeshTasksEXT as u32 {
            out.push(len);
        }
        i += len;
    }
    out
}

/// An object function's payload is a `TaskPayloadWorkgroupEXT` arena its threads write and its
/// wrapper passes to `OpEmitMeshTasksEXT`; the mesh function it launches declares the same arena
/// and reads it. Each entry is valid on every profile with the extension, the object function's
/// threadgroup size is its own specialization (`SpecKey::ObjectLocalSize`), and its stage's payload
/// limits refuse typed past them.
#[test]
fn an_object_function_passes_its_payload_to_its_mesh_functions() {
    use crate::device::requirements::VkLimit;
    use crate::device::{Device, DeviceDescription, DevicePolicy};
    use crate::pipeline::sym::Axis;
    let (Some(order), Some(draws)) = (fixture("mesh_order"), fixture("mesh_draws")) else {
        return;
    };
    let payload = crate::spv::StorageClass::TaskPayloadWorkgroupEXT;
    for (stem, p) in mesh_profiles() {
        for (bc, object, mesh) in [
            (&order, "mesh_order_object", "mesh_order_mesh"),
            (&draws, "mesh_draws_object", "mesh_draws_mesh"),
        ] {
            let o = compile(bc, object, &p).unwrap_or_else(|e| panic!("{stem} {object}: {e}"));
            assert_eq!(o.interface.stage, Stage::Object, "{stem} {object}");
            check_words_valid(&o.module.words, &o.module.requirements, &p)
                .unwrap_or_else(|e| panic!("{stem} {object}: L3: {e}"));
            assert_eq!(variables_of(&o.module.words, payload), 1, "{stem} {object}");
            assert_eq!(
                emits(&o.module.words),
                [5],
                "{stem} {object}: the grid and payload"
            );
            assert!(
                o.strategies.entries.contains_key("access.payload.store")
                    || o.strategies
                        .entries
                        .contains_key("access.payload.store.checked"),
                "{stem} {object}: {:?}",
                o.strategies.entries.keys().collect::<Vec<_>>()
            );
            let keys: Vec<SpecKey> = o.interface.spec.iter().map(|c| c.key).collect();
            assert!(
                keys.contains(&SpecKey::ObjectLocalSize(Axis::X))
                    && !keys.contains(&SpecKey::LocalSize(Axis::X)),
                "{stem} {object}: {keys:?}"
            );
            let m = compile(bc, mesh, &p).unwrap_or_else(|e| panic!("{stem} {mesh}: {e}"));
            assert_eq!(m.interface.stage, Stage::Mesh, "{stem} {mesh}");
            check_words_valid(&m.module.words, &m.module.requirements, &p)
                .unwrap_or_else(|e| panic!("{stem} {mesh}: L3: {e}"));
            assert_eq!(variables_of(&m.module.words, payload), 1, "{stem} {mesh}");
            assert!(
                m.strategies.entries.contains_key("access.payload.load")
                    || m.strategies
                        .entries
                        .contains_key("access.payload.load.checked"),
                "{stem} {mesh}: {:?}",
                m.strategies.entries.keys().collect::<Vec<_>>()
            );
        }
    }
    // `mesh_draws_object`'s 32-byte payload: within `maxTaskPayloadSize` of 32 and not 31; with
    // the wrapper's 12-byte grid within `maxTaskPayloadAndSharedMemorySize` of 44 and not 43.
    let (stem, _) = &mesh_profiles()[0];
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    // The refusal's feature and needs (its location is the payload parameter's or the entry's).
    let refused = |feature, limit, required, available| {
        let u = crate::error::Unsupported::needs(
            feature,
            Capability::Limit {
                limit,
                required,
                available,
            },
        );
        Err((u.feature, u.needs))
    };
    for (payload_size, together, want) in [
        (32, 44, Ok(())),
        (
            31,
            44,
            refused(Feature::MeshShading, VkLimit::MaxTaskPayloadSize, 32, 31),
        ),
        (
            32,
            43,
            refused(
                Feature::ThreadgroupMemorySize,
                VkLimit::MaxTaskPayloadAndSharedMemorySize,
                44,
                43,
            ),
        ),
    ] {
        let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        desc.limits.mesh.max_task_payload_size = payload_size;
        desc.limits.mesh.max_task_payload_and_shared_memory_size = together;
        let dev = Device::new(desc, DevicePolicy::default()).expect("device");
        let got = compile(&draws, "mesh_draws_object", dev.profile())
            .map(|_| ())
            .map_err(|e| match e {
                Error::Unsupported(u) => (u.feature, u.needs),
                other => panic!("{stem}: {other:?}"),
            });
        assert_eq!(
            got, want,
            "{stem}: {payload_size} payload bytes, {together} together"
        );
    }
}

/// A mesh function's threadgroup memory is its workgroup's, as a kernel's: its threadgroup barrier
/// is a `Workgroup` control barrier, its arena and the wrapper's primitive count share
/// `maxMeshSharedMemorySize`, which refuses them typed past it, and its clip distances ask
/// `maxClipDistances` for their own count, not the vertex count they are arrayed by.
#[test]
fn a_mesh_functions_threadgroup_memory_is_its_workgroups() {
    use crate::device::requirements::VkLimit;
    use crate::device::{Device, DeviceDescription, DevicePolicy};
    let Some(outputs) = fixture("mesh_outputs") else {
        return;
    };
    let profiles = mesh_profiles();
    for (stem, p) in &profiles {
        let a = compile(&outputs, "mesh_outputs_points", p)
            .unwrap_or_else(|e| panic!("{stem} mesh_outputs_points: {e}"));
        assert_eq!(
            a.kernel.static_threadgroup_bytes, 256,
            "{stem}: scratch[64]"
        );
        assert!(
            a.strategies.entries.contains_key("barrier.threadgroup"),
            "{stem}: {:?}",
            a.strategies.entries.keys().collect::<Vec<_>>()
        );
        assert_eq!(
            a.module.requirements.limits.get(&VkLimit::MaxClipDistances),
            Some(&2),
            "{stem}: two clip distances a vertex"
        );
    }
    // The 256-byte arena and the 4-byte primitive count fit 260 bytes, and not 259.
    let (stem, _) = &profiles[0];
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    for (available, fits) in [(260, true), (259, false)] {
        let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        desc.limits.mesh.max_mesh_shared_memory_size = available;
        let dev = Device::new(desc, DevicePolicy::default()).expect("device");
        match compile(&outputs, "mesh_outputs_points", dev.profile()) {
            Ok(_) => assert!(fits, "{stem} at {available}"),
            Err(Error::Unsupported(u)) => {
                assert!(!fits, "{stem} at {available}: {u:?}");
                assert_eq!(u.feature, Feature::ThreadgroupMemorySize, "{stem}");
                assert_eq!(
                    u.needs,
                    Needs::AnyOf(vec![vec![Capability::Limit {
                        limit: VkLimit::MaxMeshSharedMemorySize,
                        required: 260,
                        available: 259,
                    }]]),
                    "{stem}"
                );
            }
            Err(e) => panic!("{stem} at {available}: {e}"),
        }
    }
}

/// A mesh function's `set_point_size` stores a size past Metal's largest point (511 pixels) as 511,
/// as a vertex function's `[[point_size]]` is stored: the `PointSize` store of
/// `mesh_outputs_points` (its `set_vertex`) takes the limit, on every profile with mesh shading.
/// Item 3: the artifact records `PrecisionNote::DevicePointSize` where the device's largest point
/// is below 511 pixels, and only there.
#[test]
fn a_mesh_point_size_past_metals_largest_point_is_stored_as_it() {
    let Some(outputs) = fixture("mesh_outputs") else {
        return;
    };
    for (stem, p) in &mesh_profiles() {
        let a = compile(&outputs, "mesh_outputs_points", p)
            .unwrap_or_else(|e| panic!("{stem} mesh_outputs_points: {e}"));
        // The written size takes the limit; a constant (an output's initial value) is stored
        // as it is, and nothing else reaches `PointSize`.
        let stores = super::engine_stage_io::point_size_stores(&a.module.words);
        assert!(
            stores.contains(&Some(true)) && !stores.contains(&None),
            "{stem}: the size store takes Metal's largest point: {stores:?}"
        );
        let below = p.fields().limits.max_point_size < 511;
        assert_eq!(
            a.precision
                .contains(&crate::pipeline::artifact::PrecisionNote::DevicePointSize),
            below,
            "{stem}: the device's largest point is announced where it is below Metal's"
        );
    }
}

/// A device of committed profile `stem`.
fn device(stem: &str) -> crate::device::Device {
    use crate::device::{Device, DeviceDescription, DevicePolicy};
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// The `OpDecorate ... PerPrimitiveEXT` instructions of a module.
fn per_primitive_decorations(words: &[u32]) -> usize {
    let mut n = 0;
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let len = (w >> 16).max(1) as usize;
        if w & 0xFFFF == crate::spv::Op::Decorate as u32
            && words.get(i + 2) == Some(&(crate::spv::Decoration::PerPrimitiveEXT as u32))
        {
            n += 1;
        }
        i += len;
    }
    n
}

/// A mesh render pipeline links its mesh function with a fragment function compiled for it: the
/// fragment reads per primitive (`PerPrimitiveEXT`, flat) exactly the inputs the mesh function
/// writes per primitive, and its other inputs as varyings. The pipeline's stages are the mesh and
/// fragment stages, each valid under the pipeline's requirements, with no vertex input, and the
/// binder plans its mesh draws ([`mesh_draws`]). A fragment function compiled for another pipeline
/// is `NeedsInput(PrimitiveVaryings)`, one compiled alone under the mesh function's primitive keys
/// links, and a pipeline with an object function passing a payload has its task stage first, each
/// stage valid, and its draws specialize each stage's threadgroup under its own keys
/// ([`object_draws`]).
#[test]
fn a_mesh_pipeline_reads_its_primitive_outputs_per_primitive() {
    use crate::budget::Budget;
    use crate::error::RuntimeInput;
    use crate::library::Library;
    use crate::metal::PixelFormat;
    use crate::pipeline::descriptor::{
        ColorAttachmentDescriptor, FunctionDescriptor, MeshPipelineDescriptor, StageSource,
    };
    use crate::pipeline::lower_inputs::LoweringInputs;
    use crate::vk;
    let (Some(order), Some(outputs)) = (fixture("mesh_order"), fixture("mesh_outputs")) else {
        return;
    };
    let (order, outputs) = (
        Library::from_air(&order).expect("library"),
        Library::from_air(&outputs).expect("library"),
    );
    let function =
        |lib: &Library, name: &str| FunctionDescriptor::new(lib.function(name).expect("function"));
    let stage = |lib: &Library, name: &str| StageSource::Function(function(lib, name));
    let budget = Budget::standard();
    // The mesh function, its fragment function, its color attachments, and the inputs the
    // fragment reads per primitive.
    let pipelines = [
        (
            &order,
            "mesh_order_solo",
            "mesh_order_fragment",
            vec![
                PixelFormat::RGBA32Uint,
                PixelFormat::RGBA32Uint,
                PixelFormat::RGBA32Float,
            ],
            2,
        ),
        (
            &outputs,
            "mesh_outputs_points",
            "mesh_outputs_fragment",
            vec![PixelFormat::RGBA32Float],
            1,
        ),
    ];
    for (stem, _) in mesh_profiles() {
        let dev = device(&stem);
        for (lib, mesh, fragment, formats, per_primitive) in &pipelines {
            let what = format!("{stem} {mesh} {fragment}");
            let mut d = MeshPipelineDescriptor::new(stage(lib, mesh));
            d.fragment = Some(stage(lib, fragment));
            for (i, f) in formats.iter().enumerate() {
                d.color_attachments[i] = Some(ColorAttachmentDescriptor::new(*f));
            }
            let pipeline = dev
                .compile_mesh_pipeline(&d, &budget)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let kinds: Vec<vk::ShaderStage> = pipeline.stages.iter().map(|m| m.stage).collect();
            assert_eq!(
                kinds,
                [vk::ShaderStage::MESH_EXT, vk::ShaderStage::FRAGMENT],
                "{what}"
            );
            assert!(pipeline.vertex_input.bindings.is_empty(), "{what}");
            assert!(pipeline.vertex_input.attributes.is_empty(), "{what}");
            for m in &pipeline.stages {
                check_words_valid(&m.spirv, &pipeline.requirements, dev.profile())
                    .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
            }
            assert_eq!(
                per_primitive_decorations(&pipeline.stages[1].spirv),
                *per_primitive,
                "{what}: the fragment's per-primitive inputs"
            );
            mesh_draws(&dev, &pipeline, &what);
        }
        // The points' fragment function compiled on its own: for a render pipeline it reads the
        // primitive's `pc` per vertex, and under the mesh function's primitive keys it links.
        let mesh = dev
            .compile_function(
                &function(&outputs, "mesh_outputs_points"),
                &LoweringInputs::new(),
                &budget,
            )
            .expect("mesh");
        let mut d = MeshPipelineDescriptor::new(StageSource::Compiled(mesh.clone()));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
        let alone = |inputs: &LoweringInputs| {
            let f = dev
                .compile_function(
                    &function(&outputs, "mesh_outputs_fragment"),
                    inputs,
                    &budget,
                )
                .expect("fragment");
            let mut d = d.clone();
            d.fragment = Some(StageSource::Compiled(f));
            dev.compile_mesh_pipeline(&d, &budget)
        };
        assert_eq!(
            alone(&LoweringInputs::new()).err(),
            Some(Error::NeedsInput(RuntimeInput::PrimitiveVaryings)),
            "{stem}"
        );
        let mut keyed = LoweringInputs::new();
        keyed.primitive_varyings = mesh
            .artifact()
            .interface
            .io
            .outputs
            .iter()
            .filter(|o| matches!(o.role, IoRole::PrimitiveVarying { .. }))
            .map(|o| o.key.clone())
            .collect();
        alone(&keyed).unwrap_or_else(|e| panic!("{stem}: the fragment compiled alone: {e}"));
        // An object function with a payload.
        let mut d = MeshPipelineDescriptor::new(stage(&order, "mesh_order_mesh"));
        d.object = Some(stage(&order, "mesh_order_object"));
        d.fragment = Some(stage(&order, "mesh_order_fragment"));
        for (i, f) in pipelines[0].3.iter().enumerate() {
            d.color_attachments[i] = Some(ColorAttachmentDescriptor::new(*f));
        }
        let pipeline = dev
            .compile_mesh_pipeline(&d, &budget)
            .unwrap_or_else(|e| panic!("{stem}: an object function's payload: {e}"));
        let kinds: Vec<vk::ShaderStage> = pipeline.stages.iter().map(|m| m.stage).collect();
        assert_eq!(
            kinds,
            [
                vk::ShaderStage::TASK_EXT,
                vk::ShaderStage::MESH_EXT,
                vk::ShaderStage::FRAGMENT
            ],
            "{stem}"
        );
        for m in &pipeline.stages {
            check_words_valid(&m.spirv, &pipeline.requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{stem} {:?}: L3: {e}", m.stage));
        }
        object_draws(&dev, &pipeline, &stem);
    }
}

/// `mesh_outputs` with its object function's payload declared 64 bytes, the four `float4`s its
/// threads write, and its mesh functions' 16, one: the object function of a mesh render pipeline
/// whose mesh function (`mesh_outputs_triangles`) reads its payload past its own type.
fn wide_object_payload() -> Option<Vec<u8>> {
    use metal2vulkan_devtools::llvm_assemble_text;
    let text = std::fs::read_to_string(engine_dir().join("mesh_outputs.ll")).expect("fixture");
    let object = "!{i32 0, !\"air.payload\", !\"air.struct_type_info\", !50, \
                  !\"air.arg_type_size\", i32 16,";
    assert_eq!(
        text.matches(object).count(),
        1,
        "the object function's payload"
    );
    let text = text.replace(object, &object.replace("i32 16,", "i32 64,"));
    match llvm_assemble_text(&text) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the wide payload: {why}"),
    }
}

/// A mesh function reads its payload within its pipeline's object function's, as Metal sizes the
/// payload by the type the object function declares: in the mesh render pipeline of
/// `mesh_outputs_object`, its payload declared 64 bytes ([`wide_object_payload`]), and
/// `mesh_outputs_triangles`, which declares 16 and reads 64, the mesh function's payload arena
/// holds the object function's 64 bytes (`LoweringInputs::object_payload`), emulated or not, and
/// each stage is valid. Compiled on its own, the mesh function's arena is its own 32 (the
/// smallest), and the pipeline refuses it as `NeedsInput(ObjectPayload)`; compiled under the object
/// function's bytes it links. An object payload its own arena holds leaves its key as compiled
/// alone.
#[test]
fn a_mesh_function_reads_its_payload_within_its_object_functions() {
    use crate::budget::Budget;
    use crate::error::RuntimeInput;
    use crate::library::Library;
    use crate::lower::mesh::work;
    use crate::pipeline::descriptor::{FunctionDescriptor, MeshPipelineDescriptor, StageSource};
    let Some(wide) = wide_object_payload() else {
        return;
    };
    let wide = Library::from_air(&wide).expect("library");
    let function = |name: &str| FunctionDescriptor::new(wide.function(name).expect("function"));
    let budget = Budget::standard();
    for (stem, p) in mesh_profiles().into_iter().chain(emulating_profiles()) {
        let dev = device(&stem);
        let emulated = crate::lower::mesh::emulates(&p);
        let compiled = |name: &str, inputs: &LoweringInputs| {
            dev.compile_function(&function(name), inputs, &budget)
                .unwrap_or_else(|e| panic!("{stem} {name}: {e}"))
        };
        let object = compiled("mesh_outputs_object", &LoweringInputs::new());
        assert_eq!(object.artifact().interface.payload, Some(64), "{stem}");
        let alone = compiled("mesh_outputs_triangles", &LoweringInputs::new());
        assert_eq!(
            alone.artifact().interface.payload,
            Some(32),
            "{stem}: the mesh function's own arena"
        );
        let mut inputs = LoweringInputs::new();
        inputs.object_payload = Some(64);
        let within = compiled("mesh_outputs_triangles", &inputs);
        let a = within.artifact();
        assert_eq!(a.interface.payload, Some(64), "{stem}");
        assert_eq!(
            a.interface.emulated_payload,
            emulated.then_some(16),
            "{stem}"
        );
        check_words_valid(&a.module.words, &a.module.requirements, &p)
            .unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        // An object payload the mesh function's own arena holds keeps its key.
        inputs.object_payload = Some(32);
        assert_eq!(
            compiled("mesh_outputs_triangles", &inputs).key(),
            alone.key(),
            "{stem}"
        );
        let pipeline = |mesh: StageSource| {
            let mut d = MeshPipelineDescriptor::new(mesh);
            d.object = Some(StageSource::Function(function("mesh_outputs_object")));
            d.rasterization_enabled = false;
            dev.compile_mesh_pipeline(&d, &budget)
        };
        let made = pipeline(StageSource::Function(function("mesh_outputs_triangles")))
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        for m in &made.stages {
            check_words_valid(&m.spirv, &made.requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{stem} {:?}: L3: {e}", m.stage));
        }
        let slot = made.graphics.mesh_records.as_ref().map(|r| r.slot_words);
        assert_eq!(slot, emulated.then_some(work::GRID + 16), "{stem}");
        assert_eq!(
            pipeline(StageSource::Compiled(alone)).err(),
            Some(Error::NeedsInput(RuntimeInput::ObjectPayload)),
            "{stem}"
        );
        pipeline(StageSource::Compiled(within))
            .unwrap_or_else(|e| panic!("{stem}: the mesh function compiled for the payload: {e}"));
    }
}

/// Without the extension a mesh render pipeline is emulated. Its mesh function is a compute stage
/// writing records, a synthesized vertex stage draws them and the fragment reads the primitive's
/// outputs flat (no `PerPrimitiveEXT`); every stage is valid under the pipeline's requirements, the
/// records and work bindings are the compute and vertex stages', and the pipeline's topology is the
/// records'. A draw of `G` threadgroups is the host's: a dispatch of `G` over records of `G`
/// threadgroups' words, its work buffer complete, then a draw of `NP · k` vertices each; a grid
/// whose records pass `maxStorageBufferRange` is refused naming it. An indirect draw is scanned
/// from Metal's arguments into the command buffer's records pool and drawn from the work buffer.
/// With an object function, the object stage is a compute stage first, the push range one hull, and
/// a draw dispatches its object threadgroups and is scanned. A fragment reading `[[primitive_id]]`
/// is not emulated yet.
#[test]
fn without_the_extension_a_mesh_pipeline_draws_its_records() {
    use crate::bind::{
        AccessClass, BindState, BufferRange, DrawCall, IndirectRef, MeshDispatch, MeshGroups,
        MeshRecordsRegion, UseTarget,
    };
    use crate::budget::Budget;
    use crate::device::requirements::VkLimit;
    use crate::error::{BindError, Unsupported};
    use crate::library::Library;
    use crate::lower::mesh::work;
    use crate::metal::PixelFormat;
    use crate::pipeline::descriptor::{
        ColorAttachmentDescriptor, FunctionDescriptor, MeshPipelineDescriptor, StageSource,
    };
    use crate::pipeline::layout::BindingSource;
    use crate::pipeline::sym::SyntheticBinding;
    use crate::vk;
    let (Some(draws), Some(outputs)) = (fixture("mesh_draws"), fixture("mesh_outputs")) else {
        return;
    };
    let (draws, outputs) = (
        Library::from_air(&draws).expect("library"),
        Library::from_air(&outputs).expect("library"),
    );
    let stage = |lib: &Library, name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            lib.function(name).expect("function"),
        ))
    };
    let budget = Budget::standard();
    for (stem, _) in emulating_profiles() {
        let dev = device(&stem);
        let mut d = MeshPipelineDescriptor::new(stage(&draws, "mesh_draws_grid"));
        d.fragment = Some(stage(&draws, "mesh_draws_fragment"));
        for slot in d.color_attachments.iter_mut().take(2) {
            *slot = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
        }
        let pipeline = dev
            .compile_mesh_pipeline(&d, &budget)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let kinds: Vec<vk::ShaderStage> = pipeline.stages.iter().map(|m| m.stage).collect();
        assert_eq!(
            kinds,
            [
                vk::ShaderStage::COMPUTE,
                vk::ShaderStage::VERTEX,
                vk::ShaderStage::FRAGMENT
            ],
            "{stem}"
        );
        assert!(
            !pipeline
                .requirements
                .extensions
                .contains(&VkExtension::ExtMeshShader),
            "{stem}"
        );
        for m in &pipeline.stages {
            check_words_valid(&m.spirv, &pipeline.requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{stem} {:?}: L3: {e}", m.stage));
        }
        assert_eq!(
            per_primitive_decorations(&pipeline.stages[2].spirv),
            0,
            "{stem}: the fragment reads the primitive's outputs flat"
        );
        let records = pipeline
            .graphics
            .mesh_records
            .clone()
            .expect("the pipeline's records");
        assert_eq!(
            pipeline.graphics.topology,
            vk::PrimitiveTopology::TRIANGLE_LIST,
            "{stem}"
        );
        let binding_of = |p: &crate::pipeline::RenderPipeline, stage, binding| {
            p.layout
                .sets
                .iter()
                .flat_map(|s| &s.bindings)
                .find(|b| b.source == BindingSource::Synthetic { stage, binding })
                .cloned()
        };
        for kind in [SyntheticBinding::MeshRecords, SyntheticBinding::MeshWork] {
            let binding = binding_of(&pipeline, Stage::Mesh, kind).expect("the binding");
            assert_eq!(binding.ty, vk::DescriptorType::STORAGE_BUFFER, "{stem}");
            assert_eq!(
                binding.stages,
                vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX,
                "{stem} {kind:?}"
            );
        }
        // A draw of two threadgroups.
        let state = BindState::default();
        let plan = pipeline
            .plan(&dev, &state.shape())
            .unwrap_or_else(|e| panic!("{stem}: plan: {e}"));
        let groups = |groups| DrawCall::MeshThreadgroups {
            groups,
            threads_per_object_group: [1; 3],
            threads_per_mesh_group: [8, 1, 1],
        };
        let cmds = plan
            .instantiate_draw(&state, &groups([2, 1, 1]))
            .unwrap_or_else(|e| panic!("{stem}: a draw: {e}"));
        assert_eq!(
            cmds.draw,
            vk::DrawCommand::Draw {
                vertex_count: 2 * records.primitives * records.corners,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            "{stem}"
        );
        let d = cmds.mesh_dispatch.clone().expect("the draw's dispatch");
        assert_eq!(d.mesh, MeshGroups::Direct([2, 1, 1]), "{stem}");
        assert_eq!(d.object_groups, None, "{stem}");
        assert_eq!(
            d.records,
            MeshRecordsRegion::Exact(2 * 4 * records.group_words()),
            "{stem}"
        );
        // The host's work buffer: one virtual object threadgroup of the draw's grid and no
        // payload, the draw's two mesh threadgroups from word 0 of its records, and the offsets.
        let header = work::HEADER as usize;
        let slot = records.slot_words as usize;
        assert_eq!(records.slot_words, work::GRID, "{stem}: no payload");
        assert_eq!(d.work.len(), header + slot + 2, "{stem}");
        assert_eq!(d.work_bytes, 4 * (header + slot + 2) as u64, "{stem}");
        assert_eq!(d.work[work::OBJECT_GROUPS as usize], 1, "{stem}");
        assert_eq!(d.work[work::MESH_GROUPS as usize], 2, "{stem}");
        assert_eq!(d.work[work::RECORDS_BASE as usize], 0, "{stem}");
        assert_eq!(&d.work[header..header + 4], &[2, 1, 1, 0], "{stem}");
        assert_eq!(&d.work[header + slot..], &[0, 2], "{stem}");
        // A grid Metal draws nothing for (65,535 by 65,535) draws nothing, with no records.
        let none = plan
            .instantiate_draw(&state, &groups([65535, 65535, 1]))
            .unwrap_or_else(|e| panic!("{stem}: a grid past Metal's: {e}"));
        assert!(
            matches!(
                none.draw,
                vk::DrawCommand::Draw {
                    vertex_count: 0,
                    ..
                }
            ),
            "{stem}: {:?}",
            none.draw
        );
        // Records past the device's storage buffer range, of the largest grid Metal draws.
        let l = &dev.profile().fields().limits;
        let x = l.max_compute_work_group_count[0].min(crate::bind::METAL_MESH_GRID_AXIS);
        let y = l.max_compute_work_group_count[1].min(crate::bind::METAL_MESH_GRID_TOTAL / x);
        let bytes = u64::from(x) * u64::from(y) * 4 * records.group_words();
        if bytes > u64::from(l.max_storage_buffer_range) {
            assert_eq!(
                plan.instantiate_draw(&state, &groups([x, y, 1])).err(),
                Some(BindError::Unsupported(Unsupported::needs(
                    Feature::MeshShading,
                    Capability::Limit {
                        limit: VkLimit::MaxStorageBufferRange,
                        required: bytes,
                        available: u64::from(l.max_storage_buffer_range),
                    },
                ))),
                "{stem}"
            );
        }
        // An indirect draw: scanned from Metal's arguments, bound from their offset aligned
        // down, and drawn from the work buffer.
        let indirect = DrawCall::MeshThreadgroupsIndirect {
            indirect: IndirectRef::new(1, 260),
            threads_per_object_group: [1; 3],
            threads_per_mesh_group: [8, 1, 1],
        };
        let cmds = plan
            .instantiate_draw(&state, &indirect)
            .unwrap_or_else(|e| panic!("{stem}: an indirect draw: {e}"));
        assert!(!cmds.indirect_needs_arguments, "{stem}");
        assert_eq!(
            cmds.draw,
            vk::DrawCommand::DrawIndirect {
                offset: MeshDispatch::DRAW_OFFSET,
                draw_count: 1,
                stride: 16,
            },
            "{stem}"
        );
        let d = cmds.mesh_dispatch.clone().expect("the draw's dispatch");
        let a = l.min_storage_buffer_offset_alignment.max(4);
        let start = 260 - 260 % a;
        assert_eq!(
            d.mesh,
            MeshGroups::Scanned {
                arguments: Some(BufferRange {
                    offset: start,
                    range: 260 - start + 12,
                }),
            },
            "{stem}"
        );
        assert_eq!(d.records, MeshRecordsRegion::Pool, "{stem}");
        assert_eq!(d.object_groups, None, "{stem}");
        assert_eq!(
            d.work[work::ARGUMENTS as usize],
            1 + ((260 - start) / 4) as u32,
            "{stem}"
        );
        assert!(
            cmds.uses.iter().any(|u| u.resource == UseTarget::Indirect
                && u.class == AccessClass::Storage
                && u.stages == vk::PipelineStageFlags2::COMPUTE_SHADER),
            "{stem}: the scan's read of the arguments"
        );
        object_records_draws(&dev, &draws, &stem);
        // A fragment reading the primitive's id (behind the points, where the device lets a
        // vertex stage write their outputs).
        if !writes_points_outputs(dev.profile()) {
            continue;
        }
        let mut d = MeshPipelineDescriptor::new(stage(&outputs, "mesh_outputs_points"));
        d.fragment = Some(stage(&outputs, "mesh_outputs_fragment"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
        match dev.compile_mesh_pipeline(&d, &budget) {
            Err(Error::Unsupported(u))
                if u.feature == Feature::MeshShading && u.needs == Needs::NotImplemented => {}
            other => panic!("{stem}: a fragment reading the primitive's id: {other:?}"),
        }
    }
}

/// An emulated pipeline with an object function (`mesh_draws_object`, a 32-byte payload, with
/// `mesh_draws_mesh` and `mesh_draws_fragment`): the object function's compute stage first, one
/// push range of both compute stages' windows, both work bindings; a draw of object threadgroups
/// dispatches them, is scanned into the records pool, and is drawn from the work buffer, whose
/// slots hold the payload; an indirect one runs its object arguments pass first, its slots in the
/// pool.
fn object_records_draws(dev: &crate::device::Device, draws: &crate::library::Library, stem: &str) {
    use crate::bind::{
        BindState, DrawCall, MeshDispatch, MeshGroups, MeshRecordsRegion, UseTarget,
    };
    use crate::budget::Budget;
    use crate::lower::mesh::work;
    use crate::metal::PixelFormat;
    use crate::pipeline::descriptor::{
        ColorAttachmentDescriptor, FunctionDescriptor, MeshPipelineDescriptor, StageSource,
    };
    use crate::pipeline::layout::BindingSource;
    use crate::pipeline::sym::SyntheticBinding;
    use crate::pipeline::AccessSet;
    use crate::vk;
    let stage = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            draws.function(name).expect("function"),
        ))
    };
    let mut d = MeshPipelineDescriptor::new(stage("mesh_draws_mesh"));
    d.object = Some(stage("mesh_draws_object"));
    d.fragment = Some(stage("mesh_draws_fragment"));
    for slot in d.color_attachments.iter_mut().take(2) {
        *slot = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    }
    let pipeline = dev
        .compile_mesh_pipeline(&d, &Budget::standard())
        .unwrap_or_else(|e| panic!("{stem}: an object function: {e}"));
    let kinds: Vec<vk::ShaderStage> = pipeline.stages.iter().map(|m| m.stage).collect();
    assert_eq!(
        kinds,
        [
            vk::ShaderStage::COMPUTE,
            vk::ShaderStage::COMPUTE,
            vk::ShaderStage::VERTEX,
            vk::ShaderStage::FRAGMENT
        ],
        "{stem}"
    );
    for m in &pipeline.stages {
        check_words_valid(&m.spirv, &pipeline.requirements, dev.profile())
            .unwrap_or_else(|e| panic!("{stem} {:?}: L3: {e}", m.stage));
    }
    let compute: Vec<_> = pipeline
        .layout
        .push_constants
        .iter()
        .filter(|r| r.stages.contains(vk::ShaderStageFlags::COMPUTE))
        .collect();
    assert_eq!(
        compute.len(),
        1,
        "{stem}: {:?}",
        pipeline.layout.push_constants
    );
    let stages_of = |stage, binding| {
        pipeline
            .layout
            .sets
            .iter()
            .flat_map(|s| &s.bindings)
            .find(|b| b.source == BindingSource::Synthetic { stage, binding })
            .map(|b| b.stages)
    };
    assert_eq!(
        stages_of(Stage::Object, SyntheticBinding::MeshWork),
        Some(vk::ShaderStageFlags::COMPUTE),
        "{stem}"
    );
    assert_eq!(
        stages_of(Stage::Mesh, SyntheticBinding::MeshWork),
        Some(vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX),
        "{stem}"
    );
    // The object function's slots of an indirect draw lie in the records (`work::SLOTS`).
    assert_eq!(
        stages_of(Stage::Object, SyntheticBinding::MeshRecords),
        Some(vk::ShaderStageFlags::COMPUTE),
        "{stem}"
    );
    let records = pipeline
        .graphics
        .mesh_records
        .clone()
        .expect("the pipeline's records");
    assert_eq!(
        records.slot_words,
        work::GRID + 8,
        "{stem}: the 32-byte payload"
    );
    let state = BindState::default();
    let plan = pipeline
        .plan(dev, &state.shape())
        .unwrap_or_else(|e| panic!("{stem}: plan: {e}"));
    let draw = DrawCall::MeshThreadgroups {
        groups: [2, 3, 1],
        threads_per_object_group: [4, 1, 1],
        threads_per_mesh_group: [8, 1, 1],
    };
    let cmds = plan
        .instantiate_draw(&state, &draw)
        .unwrap_or_else(|e| panic!("{stem}: a draw: {e}"));
    assert_eq!(
        cmds.draw,
        vk::DrawCommand::DrawIndirect {
            offset: MeshDispatch::DRAW_OFFSET,
            draw_count: 1,
            stride: 16,
        },
        "{stem}"
    );
    let d = cmds.mesh_dispatch.expect("the draw's dispatch");
    assert_eq!(d.object_groups, Some([2, 3, 1]), "{stem}");
    assert_eq!(d.mesh, MeshGroups::Scanned { arguments: None }, "{stem}");
    assert_eq!(d.records, MeshRecordsRegion::Pool, "{stem}");
    assert_eq!(d.work.len(), work::HEADER as usize, "{stem}");
    assert_eq!(d.work[work::OBJECT_GROUPS as usize], 6, "{stem}");
    assert_eq!(
        d.work[work::SLOT_WORDS as usize],
        records.slot_words,
        "{stem}"
    );
    assert_eq!(
        d.work_bytes,
        4 * work::words(6, records.slot_words),
        "{stem}"
    );
    // An indirect draw runs its object arguments pass first: Metal's arguments at word 1 plus
    // the offset's residual, the object threadgroup packed, no object count yet, and the slots in
    // the pool, which the pass reserves.
    let indirect = DrawCall::MeshThreadgroupsIndirect {
        indirect: crate::bind::IndirectRef::new(1, 4),
        threads_per_object_group: [4, 1, 1],
        threads_per_mesh_group: [8, 1, 1],
    };
    let cmds = plan
        .instantiate_draw(&state, &indirect)
        .unwrap_or_else(|e| panic!("{stem}: an indirect draw: {e}"));
    let d = cmds.mesh_dispatch.expect("the draw's dispatch");
    assert_eq!(d.object_groups, None, "{stem}");
    let range = d.object_arguments.expect("the object arguments pass");
    assert_eq!(range.offset + range.range, 16, "{stem}");
    assert_eq!(d.mesh, MeshGroups::Scanned { arguments: None }, "{stem}");
    assert_eq!(d.records, MeshRecordsRegion::Pool, "{stem}");
    assert_eq!(
        d.work[work::OBJECT_ARGUMENTS as usize],
        u32::try_from((4 - range.offset) / 4 + 1).expect("a word"),
        "{stem}"
    );
    assert_eq!(d.work[work::OBJECT_GROUPS as usize], 0, "{stem}");
    assert_eq!(
        d.work_bytes,
        4 * work::words(0, records.slot_words),
        "{stem}"
    );
    assert!(
        cmds.uses
            .iter()
            .any(|u| u.resource == UseTarget::Indirect && u.access == AccessSet::READ),
        "{stem}: the pass reads Metal's arguments"
    );
    // A partial object threadgroup is not implemented.
    let threads = DrawCall::MeshThreads {
        threads: [6, 1, 1],
        threads_per_object_group: [4, 1, 1],
        threads_per_mesh_group: [8, 1, 1],
    };
    assert_eq!(
        plan.instantiate_draw(&state, &threads).err(),
        Some(crate::error::BindError::Unsupported(
            crate::error::Unsupported::not_implemented(Feature::MeshShading)
        )),
        "{stem}: {threads:?}"
    );
}

/// The binder's draws of mesh pipeline `p` with an object function: each stage's threadgroup size
/// is a specialization of its own ids, which a draw of object threadgroups states beside each
/// other.
fn object_draws(dev: &crate::device::Device, p: &crate::pipeline::RenderPipeline, what: &str) {
    use crate::bind::{BindState, DrawCall};
    use crate::pipeline::sym::Axis;
    use crate::vk;
    let state = BindState::default();
    let plan = p
        .plan(dev, &state.shape())
        .unwrap_or_else(|e| panic!("{what}: plan: {e}"));
    let constants: Vec<_> = p
        .stages
        .iter()
        .flat_map(|s| s.specialization.constants.iter())
        .collect();
    let id = |key: SpecKey| constants.iter().find(|c| c.key == key).map(|c| c.id);
    let object = [Axis::X, Axis::Y, Axis::Z].map(|a| id(SpecKey::ObjectLocalSize(a)));
    let mesh = [Axis::X, Axis::Y, Axis::Z].map(|a| id(SpecKey::LocalSize(a)));
    assert!(
        object.iter().chain(&mesh).all(Option::is_some) && object != mesh,
        "{what}: {constants:?}"
    );
    let draw = DrawCall::MeshThreadgroups {
        groups: [2, 1, 1],
        threads_per_object_group: [4, 1, 1],
        threads_per_mesh_group: [8, 1, 1],
    };
    let cmds = plan
        .instantiate_draw(&state, &draw)
        .unwrap_or_else(|e| panic!("{what}: an object draw: {e}"));
    assert_eq!(
        cmds.draw,
        vk::DrawCommand::DrawMeshTasks {
            group_count_x: 2,
            group_count_y: 1,
            group_count_z: 1,
        },
        "{what}"
    );
    let key = cmds.specialization.as_ref().expect("a mesh draw's key");
    let value = |id: u32| {
        key.entries.iter().find(|e| e.constant_id == id).map(|e| {
            let at = e.offset as usize;
            u32::from_le_bytes(key.data[at..at + 4].try_into().expect("4 bytes"))
        })
    };
    assert_eq!(
        object.map(|i| i.and_then(value)),
        [Some(4), Some(1), Some(1)],
        "{what}"
    );
    assert_eq!(
        mesh.map(|i| i.and_then(value)),
        [Some(8), Some(1), Some(1)],
        "{what}"
    );
    // Where the device folds object grids, the object function reads its region record, and a
    // grid past the task stage's limits runs in pieces (`bind::mesh_arguments`).
    let folds = crate::lower::mesh::folds_object_grids(dev.profile());
    assert_eq!(
        p.dependencies
            .iter()
            .any(|d| d.fact == crate::pipeline::facts::RuntimeFact::FoldedGroups),
        folds,
        "{what}"
    );
    if !folds {
        return;
    }
    let hole = |c: &crate::bind::DrawCommands| {
        c.push_holes.iter().any(|h| {
            matches!(h, crate::bind::PushHole::ReplayRegionAddress { stages, .. }
                if *stages == vk::ShaderStageFlags::TASK_EXT)
        })
    };
    assert!(hole(&cmds) && cmds.object_fold.is_some(), "{what}");
    let task = dev.profile().fields().limits.mesh.max_task_work_group_count;
    let draw = DrawCall::MeshThreadgroups {
        groups: [task[0].saturating_add(1), 100, 1],
        threads_per_object_group: [4, 1, 1],
        threads_per_mesh_group: [8, 1, 1],
    };
    let cmds = plan
        .instantiate_draw(&state, &draw)
        .unwrap_or_else(|e| panic!("{what}: a folded object draw: {e}"));
    let fold = cmds.object_fold.expect("a fold");
    assert!(
        fold.pieces >= 1 && !fold.past && hole(&cmds),
        "{what}: {fold:?}"
    );
    let indirect = DrawCall::MeshThreadgroupsIndirect {
        indirect: crate::bind::IndirectRef::new(1, 0),
        threads_per_object_group: [4, 1, 1],
        threads_per_mesh_group: [8, 1, 1],
    };
    let cmds = plan
        .instantiate_draw(&state, &indirect)
        .unwrap_or_else(|e| panic!("{what}: a folded indirect object draw: {e}"));
    assert!(
        matches!(
            cmds.draw,
            vk::DrawCommand::DrawMeshTasksIndirectCount { .. }
        ) && hole(&cmds),
        "{what}: {:?}",
        cmds.draw
    );
}

/// The binder's plan of mesh pipeline `p` on `dev` and its mesh draws: the plan leaves the mesh
/// threadgroup's size to each draw, whose specialization states it beside the plan's constants, and
/// whose `DispatchRegion` is the draw's grid where the mesh function reads it; a draw of primitives
/// is `InvalidMeshDraw`; a mesh function that chooses its slice is planned for a pass of up to
/// `maxMeshOutputLayers` layers and no more.
fn mesh_draws(dev: &crate::device::Device, p: &crate::pipeline::RenderPipeline, what: &str) {
    use crate::bind::{BindState, DrawCall};
    use crate::device::requirements::VkLimit;
    use crate::error::BindError;
    use crate::pipeline::sym::Axis;
    use crate::vk;
    let state = BindState::default();
    let plan = p
        .plan(dev, &state.shape())
        .unwrap_or_else(|e| panic!("{what}: plan: {e}"));
    let constants = &p.stages[0].specialization.constants;
    let id = |key: SpecKey| constants.iter().find(|c| c.key == key).map(|c| c.id);
    let local = [Axis::X, Axis::Y, Axis::Z].map(|a| id(SpecKey::LocalSize(a)));
    assert!(local.iter().all(Option::is_some), "{what}: {constants:?}");
    assert!(
        plan.specialization
            .entries
            .iter()
            .all(|e| !local.contains(&Some(e.constant_id))),
        "{what}: the plan's key leaves the threadgroup to the draw"
    );
    let draw = DrawCall::MeshThreadgroups {
        groups: [2, 1, 1],
        threads_per_object_group: [1; 3],
        threads_per_mesh_group: [8, 1, 1],
    };
    let cmds = plan
        .instantiate_draw(&state, &draw)
        .unwrap_or_else(|e| panic!("{what}: a mesh draw: {e}"));
    assert_eq!(
        cmds.draw,
        vk::DrawCommand::DrawMeshTasks {
            group_count_x: 2,
            group_count_y: 1,
            group_count_z: 1,
        },
        "{what}"
    );
    let key = cmds.specialization.as_ref().expect("a mesh draw's key");
    let value = |id: u32| {
        key.entries.iter().find(|e| e.constant_id == id).map(|e| {
            let at = e.offset as usize;
            u32::from_le_bytes(key.data[at..at + 4].try_into().expect("4 bytes"))
        })
    };
    assert_eq!(
        local.map(|i| i.and_then(value)),
        [Some(8), Some(1), Some(1)],
        "{what}"
    );
    if let Some(multi) = id(SpecKey::MultiRegion) {
        assert_eq!(value(multi), Some(0), "{what}");
        let region =
            crate::lower::dispatch::region_bytes([0; 3], [16, 1, 1], [8, 1, 1]).expect("region");
        let mesh = vk::ShaderStageFlags::MESH_EXT;
        let field = plan
            .push
            .fields
            .iter()
            .find(|f| f.stage == Stage::Mesh && f.field == PushField::DispatchRegion)
            .expect("the mesh function's region");
        let crate::bind::FieldLocation::PushWindow { offset } = field.location else {
            panic!("{what}: a region outside the push window");
        };
        let write = cmds
            .push_constants
            .iter()
            .find(|w| w.stages == mesh)
            .expect("the mesh stage's push");
        let at = (offset - write.offset) as usize;
        assert_eq!(write.bytes[at..at + 32], region, "{what}");
    }
    assert_eq!(
        plan.instantiate_draw(
            &state,
            &DrawCall::Primitives {
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            }
        )
        .err(),
        Some(BindError::InvalidMeshDraw { mesh: false }),
        "{what}"
    );
    // A pass of more layers than the mesh entry may write, where the function chooses its slice.
    let chooses = p
        .dependencies
        .iter()
        .any(|d| matches!(d.fact, RuntimeFact::RenderTargetArrayLength { .. }));
    if chooses {
        let layers = dev.profile().fields().limits.mesh.max_mesh_output_layers;
        let mut shape = state.shape();
        shape.render_target_array_length = layers;
        p.plan(dev, &shape)
            .unwrap_or_else(|e| panic!("{what}: a pass of {layers} layers: {e}"));
        shape.render_target_array_length = layers + 1;
        assert_eq!(
            p.plan(dev, &shape).err(),
            Some(BindError::Unsupported(crate::error::Unsupported::needs(
                Feature::MeshShading,
                Capability::Limit {
                    limit: VkLimit::MaxMeshOutputLayers,
                    required: u64::from(layers) + 1,
                    available: u64::from(layers),
                },
            ))),
            "{what}"
        );
    }
}

/// A mesh type's member under `[[function_constant(..)]]` (the VFX framework's particle meshes) is
/// an output only where its constant holds; gated off, the mesh type has no such member, its write
/// stores nothing, and the stage interface omits it.
#[test]
fn a_gated_off_mesh_member_leaves_the_interface() {
    let Some(bc) = fixture("mesh_gated") else {
        return;
    };
    for (stem, p) in mesh_profiles() {
        let mut keys = Vec::new();
        for on in [false, true] {
            let values = FunctionConstantValues::defaults().with_index(
                0,
                FunctionConstantValue::new(DataType::Bool, [u8::from(on)]),
            );
            let a = compile_with(&bc, "mesh_gated", &values, &p)
                .unwrap_or_else(|e| panic!("{stem} has_tint={on}: {e}"));
            check_words_valid(&a.module.words, &a.module.requirements, &p)
                .unwrap_or_else(|e| panic!("{stem} has_tint={on}: L3: {e}"));
            let mut roles: Vec<(bool, String)> = a
                .interface
                .io
                .outputs
                .iter()
                .map(|v| match &v.role {
                    IoRole::Varying { user, .. } => (false, user),
                    IoRole::PrimitiveVarying { user } => (true, user),
                    r => panic!("{stem}: a mesh output role {r:?}"),
                })
                .map(|(prim, user)| (prim, String::from_utf8_lossy(&user.0).into_owned()))
                .collect();
            roles.sort();
            keys.push(roles);
        }
        let [off, on] = &keys[..] else {
            panic!("{stem}: two compiles")
        };
        assert_eq!(off.len(), 2, "{stem}: uv and id, gated off: {off:?}");
        assert_eq!(on.len(), 4, "{stem}: uv, tint, id and weight: {on:?}");
        assert!(off.iter().all(|k| on.contains(k)), "{stem}: {off:?} {on:?}");
        assert_eq!(
            on.iter().filter(|(prim, _)| *prim).count(),
            2,
            "{stem}: {on:?}"
        );
    }
}

/// The push fields of `a`'s stage.
fn push_fields(a: &FunctionArtifact) -> Vec<PushField> {
    a.interface.push.fields.iter().map(|f| f.field).collect()
}

/// A mesh function reads its `[[amplification_id]]` from the push word the binder writes for each
/// amplification it draws (a mesh pipeline draws one amplification at a time, without multiview),
/// and its `[[amplification_count]]` from the specialization constant. Under
/// `[[function_constant(..)]]` the id is a parameter only where its constant holds.
#[test]
fn a_mesh_function_reads_its_amplification_from_the_push_word() {
    let Some(bc) = fixture("mesh_amplified") else {
        return;
    };
    for (stem, p) in mesh_profiles() {
        let a = compile(&bc, "mesh_amplified", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        check_words_valid(&a.module.words, &a.module.requirements, &p)
            .unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        assert!(
            push_fields(&a).contains(&PushField::AmplificationId),
            "{stem}: {:?}",
            push_fields(&a)
        );
        assert!(
            a.interface
                .spec
                .iter()
                .any(|s| s.key == SpecKey::AmplificationCount),
            "{stem}: the count's constant"
        );
        assert!(
            !a.module
                .requirements
                .features
                .contains(&VkFeature::Multiview),
            "{stem}: no ViewIndex"
        );
        for on in [false, true] {
            let values = FunctionConstantValues::defaults().with_index(
                0,
                FunctionConstantValue::new(DataType::Bool, [u8::from(on)]),
            );
            let a = compile_with(&bc, "mesh_amplified_gated", &values, &p)
                .unwrap_or_else(|e| panic!("{stem} amplified={on}: {e}"));
            check_words_valid(&a.module.words, &a.module.requirements, &p)
                .unwrap_or_else(|e| panic!("{stem} amplified={on}: L3: {e}"));
            assert_eq!(
                push_fields(&a).contains(&PushField::AmplificationId),
                on,
                "{stem} amplified={on}"
            );
        }
    }
}

/// A primitive's render target slice index and viewport index are its `Layer` and `ViewportIndex`
/// by the vertex function's rules: each reads its bound from a push field (the pass's length, the
/// draw's viewport count), the function states the facts the binder writes them for, and the raw
/// indices are per-primitive outputs Metal's fragment function reads. Vulkan asks
/// `shaderOutputLayer` and `shaderOutputViewportIndex` of a mesh entry writing them.
#[test]
fn a_primitive_chooses_its_layer_and_viewport_as_a_vertex_does() {
    let Some(bc) = fixture("mesh_amplified") else {
        return;
    };
    for (stem, p) in mesh_profiles() {
        let a = compile(&bc, "mesh_layered", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let r = &a.module.requirements;
        check_words_valid(&a.module.words, r, &p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        for f in [
            VkFeature::ShaderOutputLayer,
            VkFeature::ShaderOutputViewportIndex,
        ] {
            assert!(r.features.contains(&f), "{stem}: {f:?}");
        }
        let fields = push_fields(&a);
        for f in [PushField::RenderTargetArrayLength, PushField::ViewportCount] {
            assert!(fields.contains(&f), "{stem}: {f:?} in {fields:?}");
        }
        assert!(a.facts.contains(&RuntimeFact::ViewportCount), "{stem}");
        assert!(
            a.facts.contains(&RuntimeFact::RenderTargetArrayLength {
                layer_dropped: false
            }),
            "{stem}"
        );
        let io = &a.interface.io;
        for b in [StageBuiltin::Layer, StageBuiltin::ViewportIndex] {
            assert!(io.builtins.contains(&b), "{stem}: {b:?}");
        }
        for role in [IoRole::RenderTargetArrayIndex, IoRole::ViewportIndex] {
            assert!(
                io.outputs.iter().any(|v| v.role == role),
                "{stem}: the raw {role:?}"
            );
        }
        let per_primitive = crate::spv::Decoration::PerPrimitiveEXT as u32;
        let decorate = crate::spv::Op::Decorate as u32;
        let words = &a.module.words;
        let mut i = 5;
        let mut decorated = 0;
        while let Some(&w) = words.get(i) {
            if w & 0xFFFF == decorate && words.get(i + 2) == Some(&per_primitive) {
                decorated += 1;
            }
            i += (w >> 16).max(1) as usize;
        }
        // Culled, primitive id, layer, viewport, their raw varyings, and the view's data.
        assert!(decorated >= 7, "{stem}: {decorated} per-primitive outputs");
    }
}

/// The access chains into a module's primitive index output (`PrimitiveLineIndicesEXT` or
/// `PrimitiveTriangleIndicesEXT`), by their index count: 1 reaches a primitive's whole vector, 2
/// one component of it.
fn index_chains(words: &[u32]) -> [usize; 2] {
    let indices = [
        crate::spv::BuiltIn::PrimitiveLineIndicesEXT as u32,
        crate::spv::BuiltIn::PrimitiveTriangleIndicesEXT as u32,
    ];
    let mut target = None;
    let mut chains = [0; 2];
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let len = (w >> 16).max(1) as usize;
        let op = w & 0xFFFF;
        if op == crate::spv::Op::Decorate as u32
            && words.get(i + 2) == Some(&(crate::spv::Decoration::BuiltIn as u32))
            && words.get(i + 3).is_some_and(|b| indices.contains(b))
        {
            target = words.get(i + 1).copied();
        }
        if op == crate::spv::Op::AccessChain as u32
            && target.is_some()
            && words.get(i + 3) == target.as_ref()
        {
            if let Some(n) = chains.get_mut(len.saturating_sub(5)) {
                *n += 1;
            }
        }
        i += len;
    }
    chains
}

/// On a driver that misplaces a store into one component of an index vector
/// (`DriverWorkaround::MeshIndexComponentStoresMisplaced`, lavapipe before Mesa 26.2) a triangle
/// mesh function keeps its indices in workgroup words and its wrapper writes each primitive's
/// indices whole: no access chain reaches a component of the indices, and the words count among
/// the wrapper's workgroup memory, refused typed one byte short (`mesh_draws_grid`: one
/// triangle, 12 bytes of indices beside the 4-byte count). Every other mesh profile stores each
/// index into its component, as before. Each module passes L3.
#[test]
fn a_mesh_function_stages_its_indices_where_the_driver_misplaces_component_stores() {
    use crate::device::profile::DriverWorkaround;
    use crate::device::requirements::VkLimit;
    use crate::device::{Device, DeviceDescription, DevicePolicy};
    let Some(draws) = fixture("mesh_draws") else {
        return;
    };
    let defect = DriverWorkaround::MeshIndexComponentStoresMisplaced;
    let mut staged = 0;
    for (stem, p) in mesh_profiles() {
        let a = compile(&draws, "mesh_draws_grid", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        check_words_valid(&a.module.words, &a.module.requirements, &p)
            .unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let [whole, component] = index_chains(&a.module.words);
        if p.fields().workarounds.contains(&defect) {
            staged += 1;
            assert_eq!((whole, component), (1, 0), "{stem}: staged indices");
        } else {
            assert_eq!(whole, 1, "{stem}: the wrapper's zeroed slot");
            assert!(component >= 3, "{stem}: {component} component stores");
        }
    }
    assert!(
        staged >= 2,
        "the lavapipe profiles of Mesa 26.0.8 and newer stage their indices"
    );
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/gate-lavapipe-x86_64.json");
    let text = std::fs::read_to_string(path).expect("profile");
    for (available, fits) in [(16, true), (15, false)] {
        let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        desc.limits.mesh.max_mesh_shared_memory_size = available;
        let dev = Device::new(desc, DevicePolicy::default()).expect("device");
        // The refusal's feature and needs (its location is the entry's).
        let got = compile(&draws, "mesh_draws_grid", dev.profile())
            .map(|_| ())
            .map_err(|e| match e {
                Error::Unsupported(u) => (u.feature, u.needs),
                other => panic!("{available} bytes: {other:?}"),
            });
        let want = match fits {
            true => Ok(()),
            false => {
                let u = crate::error::Unsupported::needs(
                    Feature::ThreadgroupMemorySize,
                    Capability::Limit {
                        limit: VkLimit::MaxMeshSharedMemorySize,
                        required: 16,
                        available: 15,
                    },
                );
                Err((u.feature, u.needs))
            }
        };
        assert_eq!(got, want, "{available} bytes");
    }
}

/// A mesh function's buffer writes need no stage store feature: Vulkan's
/// `vertexPipelineStoresAndAtomics` and `fragmentStoresAndAtomics` name no task or mesh stage, and
/// the emulation runs the function as a compute entry. `mesh_barrier_writer`, whose one store is
/// to its device buffer, compiles on every committed profile with both features removed, as a
/// `MeshEXT` entry or the emulation's compute entry, and passes L3 within that profile.
#[test]
fn a_mesh_functions_buffer_writes_need_no_stage_store_feature() {
    use crate::device::{Device, DeviceDescription, DevicePolicy};
    let Some(barrier) = fixture("mesh_barrier") else {
        return;
    };
    for (stem, _) in committed_profiles().expect("committed profiles") {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        for f in [
            VkFeature::VertexPipelineStoresAndAtomics,
            VkFeature::FragmentStoresAndAtomics,
        ] {
            desc.features.remove(&f);
        }
        let dev = Device::new(desc, DevicePolicy::default()).expect("device");
        let p = dev.profile();
        let a =
            compile(&barrier, "mesh_barrier_writer", p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(a.interface.stage, Stage::Mesh, "{stem}");
        let r = &a.module.requirements;
        for f in [
            VkFeature::VertexPipelineStoresAndAtomics,
            VkFeature::FragmentStoresAndAtomics,
        ] {
            assert!(!r.features.contains(&f), "{stem}: {f:?}");
        }
        check_words_valid(&a.module.words, r, p).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
    }
}

/// An object function's `[[threadgroup(n)]]` arguments take the render encoder's lengths
/// (`setObjectThreadgroupMemoryLength:atIndex:`, the object stage's `threadgroup_lengths`): in the
/// pipeline of `mesh_outputs_object` (its payload 64 bytes, [`wide_object_payload`]) and
/// `mesh_outputs_triangles`, a draw specializes `TgLength(0)` to the length rounded up to 16, as
/// a kernel's, and to 0 where it is unset, emulated or not. The artifact states the arena's static
/// part and its payload (`RuntimeFact::ObjectThreadgroupArena`). The plan checks the arena with the
/// lengths against the object stage's workgroup memory limit: the task stage's
/// (`maxTaskSharedMemorySize`, beside the wrapper's 12-byte grid), whose 76 bytes fit a 64-byte
/// length and not a 65-byte one, or, emulated, the compute stage's.
#[test]
fn an_object_functions_threadgroup_memory_is_the_encoders_lengths() {
    use crate::bind::{BindState, DrawCall};
    use crate::budget::Budget;
    use crate::device::requirements::VkLimit;
    use crate::device::{Device, DeviceDescription, DevicePolicy};
    use crate::error::{BindError, Unsupported};
    use crate::library::Library;
    use crate::pipeline::descriptor::{FunctionDescriptor, MeshPipelineDescriptor, StageSource};
    let Some(wide) = wide_object_payload() else {
        return;
    };
    let wide = Library::from_air(&wide).expect("library");
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            wide.function(name).expect("function"),
        ))
    };
    let budget = Budget::standard();
    let pipeline = |dev: &Device| {
        let mut d = MeshPipelineDescriptor::new(function("mesh_outputs_triangles"));
        d.object = Some(function("mesh_outputs_object"));
        d.rasterization_enabled = false;
        dev.compile_mesh_pipeline(&d, &budget)
    };
    let with_length = |len: Option<u32>| {
        let mut state = BindState::default();
        if let Some(len) = len {
            state
                .stages
                .entry(Stage::Object)
                .or_default()
                .threadgroup_lengths
                .insert(0, len);
        }
        state
    };
    let draw = DrawCall::MeshThreadgroups {
        groups: [1, 1, 1],
        threads_per_object_group: [4, 1, 1],
        threads_per_mesh_group: [8, 1, 1],
    };
    let refused_limit = |r: Result<crate::bind::BindingPlan, BindError>| match r {
        Err(BindError::Unsupported(Unsupported {
            feature: Feature::ThreadgroupMemorySize,
            needs: Needs::AnyOf(alternatives),
            ..
        })) => match alternatives.as_slice() {
            [one] => match one.as_slice() {
                [Capability::Limit {
                    limit,
                    required,
                    available,
                }] => Some((*limit, *required, *available)),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    };
    for (stem, p) in mesh_profiles().into_iter().chain(emulating_profiles()) {
        let dev = device(&stem);
        let emulated = crate::lower::mesh::emulates(&p);
        let made = pipeline(&dev).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let arena: Vec<_> = made
            .dependencies
            .iter()
            .filter_map(|d| match d.fact {
                RuntimeFact::ObjectThreadgroupArena {
                    static_bytes,
                    payload,
                } => Some((static_bytes, payload)),
                _ => None,
            })
            .collect();
        assert_eq!(arena, [(0, Some(64))], "{stem}");
        let id = made
            .stages
            .iter()
            .flat_map(|s| s.specialization.constants.iter())
            .find(|c| c.key == SpecKey::TgLength(0))
            .map(|c| c.id)
            .unwrap_or_else(|| panic!("{stem}: the object function's TgLength(0)"));
        for (len, want) in [(None, 0), (Some(20), 32), (Some(64), 64)] {
            let state = with_length(len);
            let plan = made
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{stem} {len:?}: plan: {e}"));
            let cmds = plan
                .instantiate_draw(&state, &draw)
                .unwrap_or_else(|e| panic!("{stem} {len:?}: draw: {e}"));
            let key = cmds.specialization.as_ref().expect("a mesh draw's key");
            let value = key.entries.iter().find(|e| e.constant_id == id).map(|e| {
                let at = e.offset as usize;
                u32::from_le_bytes(key.data[at..at + 4].try_into().expect("4 bytes"))
            });
            assert_eq!(
                value,
                Some(want),
                "{stem} {len:?}: Metal's 16-byte rounding"
            );
        }
        // Past the object stage's limit, refused at plan naming it.
        let l = &p.fields().limits;
        let past = if emulated {
            l.max_compute_shared_memory_size
        } else {
            l.mesh.max_task_shared_memory_size
        };
        let refused = refused_limit(made.plan(&dev, &with_length(Some(past)).shape()));
        let named = refused.map(|(limit, _, _)| limit);
        if emulated {
            assert_eq!(named, Some(VkLimit::MaxComputeSharedMemorySize), "{stem}");
        } else {
            assert!(
                matches!(
                    named,
                    Some(VkLimit::MaxTaskSharedMemorySize)
                        | Some(VkLimit::MaxTaskPayloadAndSharedMemorySize)
                ),
                "{stem}: {refused:?}"
            );
        }
    }
    // The task stage's limit: 76 bytes hold the wrapper's 12 and a 64-byte length, and a 65-byte
    // length, 80 rounded, is refused naming it.
    let (stem, _) = &mesh_profiles()[0];
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    desc.limits.mesh.max_task_shared_memory_size = 76;
    let dev = Device::new(desc, DevicePolicy::default()).expect("device");
    let made = pipeline(&dev).unwrap_or_else(|e| panic!("{stem} at 76: {e}"));
    made.plan(&dev, &with_length(Some(64)).shape())
        .unwrap_or_else(|e| panic!("{stem} at 76: 64 bytes: {e}"));
    assert_eq!(
        refused_limit(made.plan(&dev, &with_length(Some(65)).shape())),
        Some((VkLimit::MaxTaskSharedMemorySize, 92, 76)),
        "{stem} at 76: 65 bytes"
    );
}

/// Under `VariantKey::mesh_last_first` an emulated mesh pipeline's synthesized vertex stage reads
/// each triangle's corner `(c + k - 1) mod k`, so the draw takes each primitive from its last
/// index first: one `OpIAdd` and one `OpUMod` more than the base's, valid, its plan stating the
/// variant (`RuntimeFact::MeshLastFirst`). A draw whose shape asks for it runs it
/// (`BindError::NeedsVariant` from the base), and the variant's plan of the default shape asks
/// for the base again.
#[test]
fn a_last_first_mesh_draw_reads_its_corners_from_the_last() {
    use crate::bind::{BindState, DrawCall};
    use crate::budget::Budget;
    use crate::error::BindError;
    use crate::library::Library;
    use crate::metal::PixelFormat;
    use crate::pipeline::descriptor::{
        ColorAttachmentDescriptor, FunctionDescriptor, MeshPipelineDescriptor, StageSource,
    };
    use crate::vk;
    /// `OpIAdd` and `OpUMod`.
    const ADD_MOD: [u32; 2] = [128, 137];
    let Some(draws) = fixture("mesh_draws") else {
        return;
    };
    let draws = Library::from_air(&draws).expect("library");
    let stage = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            draws.function(name).expect("function"),
        ))
    };
    let budget = Budget::standard();
    let draw = DrawCall::MeshThreadgroups {
        groups: [2, 1, 1],
        threads_per_object_group: [1; 3],
        threads_per_mesh_group: [8, 1, 1],
    };
    let last_first = BindState {
        mesh_last_first: true,
        ..Default::default()
    };
    for (stem, _) in emulating_profiles() {
        let dev = device(&stem);
        let mut d = MeshPipelineDescriptor::new(stage("mesh_draws_grid"));
        d.fragment = Some(stage("mesh_draws_fragment"));
        for slot in d.color_attachments.iter_mut().take(2) {
            *slot = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
        }
        let base = dev
            .compile_mesh_pipeline(&d, &budget)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let plan = base
            .plan(&dev, &last_first.shape())
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert!(plan.variant.mesh_last_first, "{stem}");
        assert_eq!(
            plan.instantiate_draw(&last_first, &draw),
            Err(BindError::NeedsVariant {
                variant: Box::new(plan.variant.clone())
            }),
            "{stem}"
        );
        let key = plan.variant.clone();
        let v = dev
            .compile_mesh_variant(&d, &key, &budget)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        for m in &v.stages {
            check_words_valid(&m.spirv, &v.requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{stem} {:?}: L3: {e}", m.stage));
        }
        assert!(
            v.dependencies
                .iter()
                .any(|x| x.fact == RuntimeFact::MeshLastFirst),
            "{stem}"
        );
        assert!(
            !base
                .dependencies
                .iter()
                .any(|x| x.fact == RuntimeFact::MeshLastFirst),
            "{stem}"
        );
        assert_ne!(v.key, base.key, "{stem}");
        let vertex = |p: &crate::pipeline::RenderPipeline| {
            let m = p
                .stages
                .iter()
                .find(|m| m.stage == vk::ShaderStage::VERTEX)
                .expect("the synthesized vertex stage");
            opcodes(&m.spirv)
        };
        let (a, b) = (vertex(&v), vertex(&base));
        for op in ADD_MOD {
            let n = |ops: &[u32]| ops.iter().filter(|&&o| o == op).count();
            assert_eq!(n(&a), n(&b) + 1, "{stem}: opcode {op}");
        }
        let plan = v
            .plan(&dev, &last_first.shape())
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert_eq!(plan.variant, key, "{stem}");
        plan.instantiate_draw(&last_first, &draw)
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        let plain = BindState::default();
        let plan = v
            .plan(&dev, &plain.shape())
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert!(!plan.variant.mesh_last_first, "{stem}");
        assert!(
            matches!(
                plan.instantiate_draw(&plain, &draw),
                Err(BindError::NeedsVariant { .. })
            ),
            "{stem}"
        );
    }
}
