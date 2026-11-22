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
use crate::spirv::testing::{check_words_valid, committed_profiles, threadgroup_profiles};
use std::sync::Arc;

pub(super) fn fixture(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
    }
}

pub(super) fn compile(
    bc: &[u8],
    entry: &str,
    p: &DeviceProfile,
) -> Result<Arc<FunctionArtifact>, Error> {
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
pub(super) fn mesh_profiles() -> Vec<(String, DeviceProfile)> {
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

/// The committed profiles whose device does not list `VK_EXT_mesh_shader`, which refuse object and
/// mesh functions. Only those that take threadgroup memory, which the object and mesh functions
/// here use (`spirv::testing::threadgroup_profiles`).
fn refusing_profiles() -> Vec<(String, DeviceProfile)> {
    let out: Vec<(String, DeviceProfile)> = threadgroup_profiles()
        .expect("committed profiles")
        .into_iter()
        .filter(|(stem, _)| !lists_mesh_shader(stem))
        .collect();
    assert!(
        !out.is_empty(),
        "every committed profile lists the extension"
    );
    out
}

/// Without the extension a mesh or object function is refused (`Unsupported{MeshShading}`), needing
/// `VK_EXT_mesh_shader` and the stage's feature (`meshShader`, an object function's `taskShader`)
/// where the profile lacks them.
#[test]
fn without_the_extension_mesh_and_object_functions_are_refused() {
    let (Some(order), Some(outputs)) = (fixture("mesh_order"), fixture("mesh_outputs")) else {
        return;
    };
    for (stem, p) in refusing_profiles() {
        for (bc, entry, feature) in [
            (&order, "mesh_order_solo", VkFeature::MeshShader),
            (&outputs, "mesh_outputs_lines", VkFeature::MeshShader),
            (&outputs, "mesh_outputs_points", VkFeature::MeshShader),
            (&order, "mesh_order_object", VkFeature::TaskShader),
            (&order, "mesh_order_mesh", VkFeature::MeshShader),
        ] {
            let what = format!("{stem} {entry}");
            let mut missing = vec![Capability::Extension(VkExtension::ExtMeshShader)];
            if !p.enabled().features.contains(&feature) {
                missing.push(Capability::Feature(feature));
            }
            match compile(bc, entry, &p) {
                Err(Error::Unsupported(u)) => assert_eq!(
                    (u.feature, &u.needs),
                    (Feature::MeshShading, &Needs::AnyOf(vec![missing])),
                    "{what}"
                ),
                other => panic!("{what}: {:?}", other.map(|_| ())),
            }
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
        let dev = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device");
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
        let dev = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device");
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

/// A mesh function's `set_point_size` records `PrecisionNote::DevicePointSize` only where the mesh
/// outputs points: the rasterizer reads a vertex's size for points alone, so a line or triangle
/// mesh's size draws nothing, whatever the device's largest point. Apple's compiler refuses
/// `[[point_size]]` in a triangle mesh's vertex, so the triangle mesh is `mesh_outputs_points`
/// with its type's topology (`air.mesh_type_info`) made `air.triangle`, every other word of the
/// module kept. On every profile with mesh shading, its largest point set below Metal's, both
/// functions store the size and only the point mesh records the note.
#[test]
fn a_mesh_point_size_is_noted_only_where_the_mesh_outputs_points() {
    use crate::device::{Device, DeviceDescription, DevicePolicy};
    use crate::pipeline::artifact::PrecisionNote;
    let text = std::fs::read_to_string(engine_dir().join("mesh_outputs.ll")).expect("fixture");
    let point = "!\"air.point\"}";
    assert_eq!(text.matches(point).count(), 1, "one point mesh type");
    let mut modules = Vec::new();
    for (topology, text) in [
        ("points", text.clone()),
        ("triangles", text.replace(point, "!\"air.triangle\"}")),
    ] {
        match crate::pipeline::testing::llvm_assemble_text(&text) {
            Ok(bitcode) => modules.push((topology, bitcode)),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {topology}: {why}"),
        }
    }
    for (stem, _) in &mesh_profiles() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        desc.limits.max_point_size = 64;
        let dev = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device");
        for (topology, bitcode) in &modules {
            let a = compile(bitcode, "mesh_outputs_points", dev.profile())
                .unwrap_or_else(|e| panic!("{stem} {topology}: {e}"));
            let stores = super::engine_stage_io::point_size_stores(&a.module.words);
            assert!(
                stores.contains(&Some(true)),
                "{stem} {topology}: the written size is stored: {stores:?}"
            );
            assert_eq!(
                a.precision.contains(&PrecisionNote::DevicePointSize),
                *topology == "points",
                "{stem} {topology}: the device's largest point is noted for points alone"
            );
        }
    }
}

/// A device of committed profile `stem`.
pub(super) fn device(stem: &str) -> crate::device::Device {
    use crate::device::{Device, DeviceDescription, DevicePolicy};
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device")
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
/// fragment stages, each valid under the pipeline's requirements, with no vertex input. A fragment
/// function compiled for another pipeline is `NeedsInput(PrimitiveVaryings)`, one compiled alone
/// under the mesh function's primitive keys links, and a pipeline with an object function passing a
/// payload has its task stage first, each stage valid.
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
/// holds the object function's 64 bytes (`LoweringInputs::object_payload`), and each stage is
/// valid. Compiled on its own, the mesh function's arena is its own 32 (the
/// smallest), and the pipeline refuses it as `NeedsInput(ObjectPayload)`; compiled under the object
/// function's bytes it links. An object payload its own arena holds leaves its key as compiled
/// alone.
#[test]
fn a_mesh_function_reads_its_payload_within_its_object_functions() {
    use crate::budget::Budget;
    use crate::error::RuntimeInput;
    use crate::library::Library;
    use crate::pipeline::descriptor::{FunctionDescriptor, MeshPipelineDescriptor, StageSource};
    let Some(wide) = wide_object_payload() else {
        return;
    };
    let wide = Library::from_air(&wide).expect("library");
    let function = |name: &str| FunctionDescriptor::new(wide.function(name).expect("function"));
    let budget = Budget::standard();
    for (stem, p) in mesh_profiles() {
        let dev = device(&stem);
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
        assert_eq!(
            pipeline(StageSource::Compiled(alone)).err(),
            Some(Error::NeedsInput(RuntimeInput::ObjectPayload)),
            "{stem}"
        );
        pipeline(StageSource::Compiled(within))
            .unwrap_or_else(|e| panic!("{stem}: the mesh function compiled for the payload: {e}"));
    }
}

/// Without the extension a mesh render pipeline is refused (`Unsupported{MeshShading}`), with or
/// without an object function.
#[test]
fn without_the_extension_a_mesh_pipeline_is_refused() {
    use crate::budget::Budget;
    use crate::library::Library;
    use crate::pipeline::descriptor::{FunctionDescriptor, MeshPipelineDescriptor, StageSource};
    let Some(draws) = fixture("mesh_draws") else {
        return;
    };
    let draws = Library::from_air(&draws).expect("library");
    let stage = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            draws.function(name).expect("function"),
        ))
    };
    for (stem, _) in refusing_profiles() {
        let dev = device(&stem);
        for object in [None, Some("mesh_draws_object")] {
            let mut d = MeshPipelineDescriptor::new(stage(match object {
                Some(_) => "mesh_draws_mesh",
                None => "mesh_draws_grid",
            }));
            d.object = object.map(stage);
            d.rasterization_enabled = false;
            match dev.compile_mesh_pipeline(&d, &Budget::standard()) {
                Err(Error::Unsupported(u)) => {
                    assert_eq!(u.feature, Feature::MeshShading, "{stem} {object:?}")
                }
                other => panic!("{stem} {object:?}: {:?}", other.map(|_| ())),
            }
        }
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

/// A triangle mesh function stores each index into its component, on every mesh profile: the
/// wrapper's zeroed slot is the one access chain to the whole indices, and at least three reach a
/// component (`mesh_draws_grid`: one triangle). Each module passes L3.
#[test]
fn a_mesh_function_stores_each_index_into_its_component() {
    let Some(draws) = fixture("mesh_draws") else {
        return;
    };
    for (stem, p) in mesh_profiles() {
        let a = compile(&draws, "mesh_draws_grid", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        check_words_valid(&a.module.words, &a.module.requirements, &p)
            .unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let [whole, component] = index_chains(&a.module.words);
        assert_eq!(whole, 1, "{stem}: the wrapper's zeroed slot");
        assert!(component >= 3, "{stem}: {component} component stores");
    }
}

/// A mesh function's buffer writes need no stage store feature: Vulkan's
/// `vertexPipelineStoresAndAtomics` and `fragmentStoresAndAtomics` name no task or mesh stage.
/// `mesh_barrier_writer`, whose one store is to its device buffer, compiles on every committed
/// profile with `VK_EXT_mesh_shader` with both features removed, as a `MeshEXT` entry, and passes
/// L3 within that profile.
#[test]
fn a_mesh_functions_buffer_writes_need_no_stage_store_feature() {
    use crate::device::{Device, DeviceDescription, DevicePolicy};
    let Some(barrier) = fixture("mesh_barrier") else {
        return;
    };
    for (stem, _) in mesh_profiles() {
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
        let dev = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device");
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

/// A mesh render pipeline descriptor Metal refuses is refused as Metal's rule
/// (`InvalidDescriptor(MeshPipeline)`, measured on an M3 Max, `mesh_pipeline_limits.m`), on every
/// committed profile with `VK_EXT_mesh_shader`, and the descriptors beside each are made: a required size with an axis of
/// 0, of more than 1024 threads or more than its stage's maximum, or of an object stage without an
/// object function; a maximum past 1024; a required size of other threads than its stage's
/// maximum, in either stage (Metal's first compile); a function's maximum other than the
/// descriptor's; a payload past 16384 bytes. An amplification count of 0 is refused as a render
/// pipeline's, and one of 2 is made, as Metal makes it.
#[test]
fn a_mesh_descriptor_metal_refuses_is_refused_as_its_rule() {
    use crate::budget::Budget;
    use crate::error::MeshPipelineProblem as P;
    use crate::library::Library;
    use crate::pipeline::descriptor::{FunctionDescriptor, MeshPipelineDescriptor, StageSource};
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
    let refused = |problem| {
        Some(Error::InvalidDescriptor(
            crate::error::DescriptorError::MeshPipeline { problem },
        ))
    };
    for (stem, _) in mesh_profiles() {
        let dev = device(&stem);
        let answer = |d: &MeshPipelineDescriptor| dev.compile_mesh_pipeline(d, &budget).err();
        let mut d = MeshPipelineDescriptor::new(stage(&draws, "mesh_draws_mesh"));
        d.object = Some(stage(&draws, "mesh_draws_object"));
        d.rasterization_enabled = false;
        let mut alone = MeshPipelineDescriptor::new(stage(&draws, "mesh_draws_grid"));
        alone.rasterization_enabled = false;
        let with = |f: &dyn Fn(&mut MeshPipelineDescriptor)| {
            let mut changed = d.clone();
            f(&mut changed);
            changed
        };
        let cases: Vec<(MeshPipelineDescriptor, Option<Error>)> = vec![
            (
                with(&|d| d.required_threads_per_object_threadgroup = Some([4, 0, 1])),
                refused(P::RequiredEmptyAxis {
                    stage: Stage::Object,
                    required: [4, 0, 1],
                }),
            ),
            (
                with(&|d| d.required_threads_per_mesh_threadgroup = Some([2048, 1, 1])),
                refused(P::RequiredOverLimit {
                    stage: Stage::Mesh,
                    threads: 2048,
                    limit: 1024,
                }),
            ),
            (
                with(&|d| {
                    d.max_total_threads_per_object_threadgroup = Some(32);
                    d.required_threads_per_object_threadgroup = Some([64, 1, 1]);
                }),
                refused(P::RequiredOverMax {
                    stage: Stage::Object,
                    threads: 64,
                    max: 32,
                }),
            ),
            (
                {
                    let mut a = alone.clone();
                    a.required_threads_per_object_threadgroup = Some([32, 1, 1]);
                    a
                },
                refused(P::RequiredWithoutObjectFunction {
                    required: [32, 1, 1],
                }),
            ),
            (
                with(&|d| d.max_vertex_amplification_count = 0),
                Some(Error::InvalidDescriptor(
                    crate::error::DescriptorError::VertexAmplificationCount { count: 0 },
                )),
            ),
            (
                with(&|d| d.max_total_threads_per_object_threadgroup = Some(2048)),
                refused(P::MaxThreadsOverLimit {
                    stage: Stage::Object,
                    max: 2048,
                    limit: 1024,
                }),
            ),
            (
                {
                    let mut a = alone.clone();
                    a.max_total_threads_per_mesh_threadgroup = Some(2048);
                    a
                },
                refused(P::MaxThreadsOverLimit {
                    stage: Stage::Mesh,
                    max: 2048,
                    limit: 1024,
                }),
            ),
            (
                with(&|d| {
                    d.max_total_threads_per_mesh_threadgroup = Some(64);
                    d.required_threads_per_mesh_threadgroup = Some([32, 1, 1]);
                }),
                refused(P::MaxThreadsDiffer {
                    stage: Stage::Mesh,
                    specified: 32,
                    max: 64,
                }),
            ),
            (
                with(&|d| {
                    d.max_total_threads_per_object_threadgroup = Some(64);
                    d.required_threads_per_object_threadgroup = Some([32, 1, 1]);
                }),
                refused(P::MaxThreadsDiffer {
                    stage: Stage::Object,
                    specified: 32,
                    max: 64,
                }),
            ),
            (
                with(&|d| {
                    d.max_total_threads_per_object_threadgroup = Some(64);
                    d.required_threads_per_object_threadgroup = Some([16, 2, 1]);
                }),
                refused(P::MaxThreadsDiffer {
                    stage: Stage::Object,
                    specified: 32,
                    max: 64,
                }),
            ),
            (
                with(&|d| d.payload_memory_length = Some(16385)),
                refused(P::PayloadOverLimit {
                    length: 16385,
                    limit: 16384,
                }),
            ),
            (
                {
                    let mut a = alone.clone();
                    a.payload_memory_length = Some(16385);
                    a
                },
                refused(P::PayloadOverLimit {
                    length: 16385,
                    limit: 16384,
                }),
            ),
            // Made as Metal makes them.
            (with(&|d| d.max_vertex_amplification_count = 2), None),
            (
                with(&|d| d.required_threads_per_object_threadgroup = Some([32, 2, 1])),
                None,
            ),
            (
                with(&|d| {
                    d.max_total_threads_per_object_threadgroup = Some(64);
                    d.required_threads_per_object_threadgroup = Some([64, 1, 1]);
                }),
                None,
            ),
            (
                with(&|d| {
                    d.max_total_threads_per_mesh_threadgroup = Some(64);
                    d.required_threads_per_mesh_threadgroup = Some([64, 1, 1]);
                }),
                None,
            ),
            (
                with(&|d| d.max_total_threads_per_object_threadgroup = Some(1024)),
                None,
            ),
        ];
        for (i, (d, want)) in cases.iter().enumerate() {
            assert_eq!(&answer(d), want, "{stem}: case {i}");
        }
        // A function's own maximum beside another of the descriptor's
        // (`mesh_outputs_object`'s 64, `mesh_outputs_triangles`' 32).
        let mut own = MeshPipelineDescriptor::new(stage(&outputs, "mesh_outputs_triangles"));
        own.object = Some(stage(&outputs, "mesh_outputs_object"));
        own.rasterization_enabled = false;
        for (object, mesh, want) in [
            (
                Some(32),
                None,
                refused(P::MaxThreadsDiffer {
                    stage: Stage::Object,
                    specified: 64,
                    max: 32,
                }),
            ),
            (
                Some(128),
                None,
                refused(P::MaxThreadsDiffer {
                    stage: Stage::Object,
                    specified: 64,
                    max: 128,
                }),
            ),
            (
                None,
                Some(64),
                refused(P::MaxThreadsDiffer {
                    stage: Stage::Mesh,
                    specified: 32,
                    max: 64,
                }),
            ),
            (Some(64), Some(32), None),
        ] {
            own.max_total_threads_per_object_threadgroup = object;
            own.max_total_threads_per_mesh_threadgroup = mesh;
            assert_eq!(answer(&own), want, "{stem}: {object:?} {mesh:?}");
        }
        // A required mesh size beside the mesh function's own maximum alone is made.
        own.max_total_threads_per_object_threadgroup = None;
        own.max_total_threads_per_mesh_threadgroup = None;
        own.required_threads_per_mesh_threadgroup = Some([16, 1, 1]);
        assert_eq!(answer(&own), None, "{stem}");
    }
}

/// `payloadMemoryLength` sizes the payload of both stages: in the mesh render pipeline of
/// `mesh_outputs_object`, which declares 16 bytes, and `mesh_outputs_triangles`, which declares 16
/// and reads 64, a length of 64 gives the object function's arena and the mesh function's its 64
/// bytes, as the object function's declaring 64 does ([`wide_object_payload`]);
/// a length the declared type holds leaves the object function's key as without it; and a stage
/// compiled on its own without the length is refused as `NeedsInput(ObjectPayload)`.
#[test]
fn the_descriptors_payload_length_sizes_both_stages_payloads() {
    use crate::budget::Budget;
    use crate::error::RuntimeInput;
    use crate::library::Library;
    use crate::pipeline::descriptor::{FunctionDescriptor, MeshPipelineDescriptor, StageSource};
    let Some(outputs) = fixture("mesh_outputs") else {
        return;
    };
    let outputs = Library::from_air(&outputs).expect("library");
    let function = |name: &str| FunctionDescriptor::new(outputs.function(name).expect("function"));
    let budget = Budget::standard();
    for (stem, _) in mesh_profiles() {
        let dev = device(&stem);
        let mut d =
            MeshPipelineDescriptor::new(StageSource::Function(function("mesh_outputs_triangles")));
        d.object = Some(StageSource::Function(function("mesh_outputs_object")));
        d.rasterization_enabled = false;
        d.payload_memory_length = Some(64);
        let made = dev
            .compile_mesh_pipeline(&d, &budget)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let arena: Vec<_> = made
            .dependencies
            .iter()
            .filter_map(|d| match d.fact {
                RuntimeFact::ObjectThreadgroupArena { payload, .. } => Some(payload),
                _ => None,
            })
            .collect();
        assert_eq!(arena, [Some(64)], "{stem}");
        for m in &made.stages {
            check_words_valid(&m.spirv, &made.requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{stem} {:?}: L3: {e}", m.stage));
        }
        // A length the type holds compiles the object function as without it.
        let compiled = |inputs: &LoweringInputs| {
            dev.compile_function(&function("mesh_outputs_object"), inputs, &budget)
                .unwrap_or_else(|e| panic!("{stem}: {e}"))
        };
        let mut held = LoweringInputs::new();
        held.object_payload = Some(8);
        assert_eq!(
            compiled(&held).key(),
            compiled(&LoweringInputs::new()).key(),
            "{stem}"
        );
        // An object stage compiled without the length is compiled for another pipeline.
        let mut alone = d.clone();
        alone.object = Some(StageSource::Compiled(compiled(&LoweringInputs::new())));
        assert_eq!(
            dev.compile_mesh_pipeline(&alone, &budget).err(),
            Some(Error::NeedsInput(RuntimeInput::ObjectPayload)),
            "{stem}"
        );
    }
}
