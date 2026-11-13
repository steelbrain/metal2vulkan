//! Ray intersection lowers through `VK_KHR_ray_query`.
//!
//! Every module is written here, assembled with `llvm-as` (a missing tool skips visibly), and
//! compiled by the engine for every committed profile. A profile with `rayQuery`,
//! `accelerationStructure`, and `bufferDeviceAddress` translates an intersector over an entry
//! argument's acceleration structure, and `spirv-val` accepts the module for that profile; a
//! profile without them refuses it as `RayTracing`, naming what is missing, at the structure's
//! argument. Refusals the device does not decide (an intersection function table, motion,
//! reading primitive data) are the same everywhere.
//!
//! The CPU interpreter (`spirv::interp`) cannot trace rays: it has no acceleration structure and
//! reports every ray query instruction unmodelled, so what the translation computes is measured
//! on a device (the measurement plan), not here.
//!
//! A structure's instance count is the binder's push field: it translates on every profile, and the
//! interpreter runs it over the binder's instantiated push bytes.

use crate::air::error::IntrinsicError;
use crate::bind::{
    BindState, BoundAccelerationStructure, BoundBuffer, Dispatch, StageTable, WriteSource,
};
use crate::budget::{Budget, Meter, LEDGER_BYTES, SPIRV_WORDS};
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{
    Capability, DiagName, Error, Feature, InputErrorKind, Limit, Location, LowerFamily, Needs,
    Unsupported,
};
use crate::library::{FunctionConstantValues, Library};
use crate::lower::select::LowerInputs;
use crate::pipeline::compile::{compile_bitcode, compile_bitcode_linked, EntryRef, LinkedSource};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::{AccessSet, BufferForm, DescriptorView, SlotInterface, Window};
use crate::pipeline::sym::{
    AccelerationStructureLevel, Axis, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef,
    SpecKey, Stage,
};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, meter, specialized_entry, AssembleError,
    Scratch,
};
use crate::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, PrecisionNote, StageSource};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::testing::{committed_profiles, is_moltenvk_capture, spirv_dis};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// The version metadata every module below ends with.
const VERSIONS: &str = "!air.version = !{!90}\n!air.language_version = !{!91}\n\
                        !90 = !{i32 2, i32 8, i32 0}\n!91 = !{!\"Metal\", i32 4, i32 0, i32 0}\n";

/// The features a ray query translation needs.
const RAY_FEATURES: [VkFeature; 3] = [
    VkFeature::RayQuery,
    VkFeature::AccelerationStructure,
    VkFeature::BufferDeviceAddress,
];

/// `text` (after [`HEADER`]) assembled from a scratch file, or `None` when `llvm-as` is missing.
fn assemble(tag: &str, text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new(&format!("m2v-engine-ray-{tag}"));
    let dir = scratch.path();
    let path = dir.join("module.ll");
    std::fs::write(&path, format!("{HEADER}{text}{VERSIONS}")).expect("write");
    let bc = llvm_assemble(&path);
    match bc {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {tag}: {why}"),
    }
}

/// A device buffer argument at parameter and `[[buffer]]` index `i`.
fn buffer(i: u32, access: &str, name: &str) -> String {
    format!(
        "!{{i32 {i}, !\"air.buffer\", !\"air.location_index\", i32 {i}, i32 1, !\"{access}\", \
         !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, \
         !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", \
         !\"air.arg_name\", !\"{name}\"}}"
    )
}

/// An acceleration-structure argument of `level` (`primitive` or `instance`) at parameter and
/// `[[buffer]]` index `i`.
fn structure(i: u32, level: &str) -> String {
    format!(
        "!{{i32 {i}, !\"air.{level}_acceleration_structure\", !\"air.location_index\", i32 {i}, \
         i32 1, !\"air.read\", !\"air.arg_type_name\", !\"acceleration_structure<>\", \
         !\"air.arg_name\", !\"scene\"}}"
    )
}

/// The primitive-level intersector of `triangle_data` (the shape Metal compiles
/// `intersector<triangle_data>` to) with constant settings.
const TRIANGLE_DATA: &str = "{ i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 }";

/// `k(device uint *out [[buffer(0)]], primitive_acceleration_structure scene [[buffer(1)]])`:
/// one ray through the null table with `settings` (nine `i32`s and `accept_any`), storing the
/// type, distance, primitive, barycentric `u`, and front face.
fn primitive_kernel(settings: &str) -> String {
    format!(
        "define void @k(ptr addrspace(1) %0, ptr addrspace(1) %1) {{\n  %t = call ptr \
         addrspace(1) @air.get_null_intersection_function_table()\n  %r = call {TRIANGLE_DATA} \
         @air.intersect.triangle_data(<3 x float> <float 0.0, float 0.0, float -1.0>, <3 x \
         float> <float 0.0, float 0.0, float 1.0>, float 0.0, float 100.0, ptr addrspace(1) %1, \
         ptr addrspace(1) %t, ptr null, i64 0, {settings})\n  %ty = extractvalue {TRIANGLE_DATA} \
         %r, 0\n  store i32 %ty, ptr addrspace(1) %0, align 4\n  %d = extractvalue \
         {TRIANGLE_DATA} %r, 1\n  %db = bitcast float %d to i32\n  %p1 = getelementptr i32, ptr \
         addrspace(1) %0, i64 1\n  store i32 %db, ptr addrspace(1) %p1, align 4\n  %pr = \
         extractvalue {TRIANGLE_DATA} %r, 2\n  %p2 = getelementptr i32, ptr addrspace(1) %0, \
         i64 2\n  store i32 %pr, ptr addrspace(1) %p2, align 4\n  %b = extractvalue \
         {TRIANGLE_DATA} %r, 5\n  %u = extractelement <2 x float> %b, i64 0\n  %ub = bitcast \
         float %u to i32\n  %p3 = getelementptr i32, ptr addrspace(1) %0, i64 3\n  store i32 \
         %ub, ptr addrspace(1) %p3, align 4\n  %f = extractvalue {TRIANGLE_DATA} %r, 6\n  %fz = \
         zext i1 %f to i32\n  %p4 = getelementptr i32, ptr addrspace(1) %0, i64 4\n  store i32 \
         %fz, ptr addrspace(1) %p4, align 4\n  ret void\n}}\n\ndeclare ptr addrspace(1) \
         @air.get_null_intersection_function_table()\ndeclare {TRIANGLE_DATA} \
         @air.intersect.triangle_data(<3 x float>, <3 x float>, float, float, ptr addrspace(1), \
         ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1)\n\n\
         !air.kernel = !{{!0}}\n!0 = !{{ptr @k, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4}}\n!3 = \
         {}\n!4 = {}\n",
        buffer(0, "air.read_write", "out"),
        structure(1, "primitive"),
    )
}

/// The instanced intersector with world-space data and triangle data.
const INSTANCED: &str = "{ i32, float, i32, i32, ptr addrspace(1), i32, i32, <2 x float>, i1, <3 \
                         x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x \
                         float>, <3 x float>, <3 x float> }";

/// `k(device uint *out [[buffer(0)]], device const uint *s [[buffer(1)]],
/// instance_acceleration_structure scene [[buffer(2)]], uint tid [[thread_position_in_grid]])`:
/// one ray per thread whose mask, winding, culls, forced opacity, and accept-any come from `s`
/// at run time, storing the type, instance and user ids, front face, and one coordinate of each
/// transform.
fn instanced_kernel() -> String {
    let mut body = String::from(
        "define void @k(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, i32 %3) \
         {\n",
    );
    for (k, name) in ["mask", "wind", "tcull", "gcull", "ocull", "force", "any"]
        .iter()
        .enumerate()
    {
        body.push_str(&format!(
            "  %{name}.p = getelementptr i32, ptr addrspace(1) %1, i64 {k}\n  %{name} = load \
             i32, ptr addrspace(1) %{name}.p, align 4\n"
        ));
    }
    body.push_str(&format!(
        "  %anyb = icmp ne i32 %any, 0\n  %x = uitofp i32 %3 to float\n  %o = insertelement <3 \
         x float> <float 0.0, float 0.0, float -1.0>, float %x, i64 0\n  %t = call ptr \
         addrspace(1) @air.get_null_intersection_function_table()\n  %r = call {INSTANCED} \
         @air.intersect.instancing.triangle_data.world_space_data(<3 x float> %o, <3 x float> \
         <float 0.0, float 0.0, float 1.0>, float 0.0, float 100.0, ptr addrspace(1) %2, i32 \
         %mask, ptr addrspace(1) %t, ptr null, i64 0, i32 %wind, i32 %tcull, i32 %gcull, i32 \
         %ocull, i32 %force, i32 3, i32 -1, i32 -1, i32 0, i1 false, i1 %anyb)\n"
    ));
    for (k, field) in [0, 5, 6, 8, 9, 16].iter().enumerate() {
        let value = match field {
            0 | 5 | 6 => format!("  %v{k} = extractvalue {INSTANCED} %r, {field}\n"),
            8 => format!(
                "  %f{k} = extractvalue {INSTANCED} %r, 8\n  %v{k} = zext i1 %f{k} to i32\n"
            ),
            _ => format!(
                "  %c{k} = extractvalue {INSTANCED} %r, {field}\n  %e{k} = extractelement <3 x \
                 float> %c{k}, i64 1\n  %v{k} = bitcast float %e{k} to i32\n"
            ),
        };
        body.push_str(&value);
        body.push_str(&format!(
            "  %p{k} = getelementptr i32, ptr addrspace(1) %0, i64 {k}\n  store i32 %v{k}, ptr \
             addrspace(1) %p{k}, align 4\n"
        ));
    }
    format!(
        "{body}  ret void\n}}\n\ndeclare ptr addrspace(1) \
         @air.get_null_intersection_function_table()\ndeclare {INSTANCED} \
         @air.intersect.instancing.triangle_data.world_space_data(<3 x float>, <3 x float>, \
         float, float, ptr addrspace(1), i32, ptr addrspace(1), ptr, i64, i32, i32, i32, i32, \
         i32, i32, i32, i32, i32, i1, i1)\n\n!air.kernel = !{{!0}}\n!0 = !{{ptr @k, !1, !2}}\n!1 \
         = !{{}}\n!2 = !{{!3, !4, !5, !6}}\n!3 = {}\n!4 = {}\n!5 = {}\n!6 = !{{i32 3, \
         !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", \
         !\"tid\"}}\n",
        buffer(0, "air.read_write", "out"),
        buffer(1, "air.read", "s"),
        structure(2, "instance"),
    )
}

/// Whether `p` enables every feature a ray query translation needs.
fn traces(p: &DeviceProfile) -> bool {
    let enabled = &p.fields().enabled.features;
    RAY_FEATURES.iter().all(|f| enabled.contains(f))
}

/// `entry` of `bitcode` on every committed profile: translated, linked, and valid where the
/// profile traces rays (the ray query instructions present), refused as `RayTracing` naming the
/// missing features at argument `arg` elsewhere. Returns the profiles that translated it.
fn per_profile(bitcode: &[u8], entry: &str, arg: u32) -> Vec<String> {
    let mut translated = Vec::new();
    for (stem, p) in committed_profiles().expect("committed profiles") {
        match compile(bitcode, entry, &p) {
            Ok(a) => {
                assert!(traces(&p), "{stem} translated without ray queries");
                // The device computes the hit; the artifact records it.
                assert!(
                    a.precision.contains(&PrecisionNote::DeviceRayIntersection),
                    "{stem}: {:?}",
                    a.precision
                );
                let (_, words) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
                let dis = spirv_dis(&words);
                for needle in [
                    "OpCapability RayQueryKHR",
                    "OpTypeAccelerationStructureKHR",
                    "OpRayQueryInitializeKHR",
                    "OpRayQueryProceedKHR",
                    "OpRayQueryConfirmIntersectionKHR",
                    "OpRayQueryGetIntersectionTypeKHR",
                ] {
                    assert!(dis.contains(needle), "{stem}: {needle} missing");
                }
                translated.push(stem);
            }
            Err(Error::Unsupported(u)) => {
                assert!(!traces(&p), "{stem} refused: {u}");
                let missing: Vec<Capability> = RAY_FEATURES
                    .into_iter()
                    .filter(|f| !p.fields().enabled.features.contains(f))
                    .map(Capability::Feature)
                    .collect();
                assert_eq!(u.feature, Feature::RayTracing, "{stem}: {u}");
                assert_eq!(u.needs, Needs::AnyOf(vec![missing]), "{stem}: {u}");
                assert_eq!(
                    u.location,
                    Some(Location::metadata(
                        Some(DiagName::new(entry)),
                        Some(arg),
                        None
                    )),
                    "{stem}: {u}"
                );
            }
            Err(other) => panic!("{stem}: {other}"),
        }
    }
    translated
}

/// The refusal of `entry` of `bitcode` on every committed profile that traces rays: the same on
/// each.
fn refusal(bitcode: &[u8], entry: &str) -> Unsupported {
    let mut first: Option<Unsupported> = None;
    for (stem, p) in committed_profiles().expect("committed profiles") {
        if !traces(&p) {
            continue;
        }
        let u = match compile(bitcode, entry, &p) {
            Err(Error::Unsupported(u)) => u,
            Err(other) => panic!("{entry} on {stem}: {other}"),
            Ok(_) => panic!("{entry} on {stem}: compiled"),
        };
        match &first {
            Some(f) => assert_eq!(&u, f, "{entry} on {stem}"),
            None => first = Some(u),
        }
    }
    first.expect("a committed profile that traces rays")
}

/// A primitive-level intersector with constant settings translates where the device traces rays
/// (lavapipe among them) and is refused, naming the missing features, where it does not
/// (MoltenVK among them).
#[test]
fn primitive_intersection_is_a_ray_query() {
    let settings = "i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 -1, i32 -1, i32 0, i1 false";
    let Some(bc) = assemble("primitive", &primitive_kernel(settings)) else {
        return;
    };
    let translated = per_profile(&bc, "k", 1);
    assert!(translated.iter().any(|s| s == "lavapipe"), "{translated:?}");
    assert!(
        !translated.iter().any(|s| is_moltenvk_capture(s)),
        "{translated:?}"
    );
}

/// Every culling the settings can ask for at once still translates and validates:
/// counterclockwise winding with back-face culling, both geometry kinds culled, a forced opacity
/// with its own cull, and accept-any. The flags are computed where the traversal starts, which
/// these settings skip; the flags' exclusivity rules are checked over every combination on the
/// CPU interpreter (`lower::raytracing` unit tests).
#[test]
fn constant_culls_validate() {
    let settings = "i32 1, i32 2, i32 3, i32 1, i32 1, i32 3, i32 -1, i32 -1, i32 0, i1 true";
    let Some(bc) = assemble("culls", &primitive_kernel(settings)) else {
        return;
    };
    let translated = per_profile(&bc, "k", 1);
    assert!(translated.iter().any(|s| s == "lavapipe"), "{translated:?}");
}

/// An instanced intersector with world-space data reads its mask, winding, culls, forced
/// opacity, and accept-any at run time, and its instance ids, front face, and transforms from
/// the committed intersection.
#[test]
fn instanced_intersection_reads_settings_at_run_time() {
    let Some(bc) = assemble("instanced", &instanced_kernel()) else {
        return;
    };
    let translated = per_profile(&bc, "k", 2);
    assert!(translated.iter().any(|s| s == "lavapipe"), "{translated:?}");
    for (stem, p) in committed_profiles().expect("committed profiles") {
        if !translated.contains(&stem) {
            continue;
        }
        let a = compile(&bc, "k", &p).expect("compiled");
        let (_, words) = link_and_validate(&a, &p).expect("valid");
        let dis = spirv_dis(&words);
        for needle in [
            // Runtime flags may skip a primitive kind.
            "OpCapability RayTraversalPrimitiveCullingKHR",
            "OpRayQueryGetIntersectionInstanceIdKHR",
            "OpRayQueryGetIntersectionInstanceCustomIndexKHR",
            "OpRayQueryGetIntersectionFrontFaceKHR",
            "OpRayQueryGetIntersectionObjectToWorldKHR",
            "OpRayQueryGetIntersectionWorldToObjectKHR",
        ] {
            assert!(dis.contains(needle), "{stem}: {needle} missing");
        }
    }
}

/// An intersector whose level is not its structure's (an instanced intersector over a
/// primitive structure) is refused, as ray tracing.
#[test]
fn intersector_level_follows_the_structure() {
    let text =
        instanced_kernel().replace("air.instance_acceleration", "air.primitive_acceleration");
    let Some(bc) = assemble("level", &text) else {
        return;
    };
    let u = refusal(&bc, "k");
    assert_eq!(u.feature, Feature::RayTracing, "{u}");
    assert_eq!(u.needs, Needs::NotImplemented, "{u}");
    assert_eq!(
        u.symbol,
        Some(DiagName::new(
            "air.intersect.instancing.triangle_data.world_space_data"
        )),
        "{u}"
    );
}

/// `k(device uint *out [[buffer(0)]], primitive_acceleration_structure scene [[buffer(1)]],
/// intersection_function_table<triangle_data> t [[buffer(2)]])`: one ray through the table
/// argument, storing the type; `extra` reads the table otherwise.
fn table_kernel(extra: &str) -> String {
    format!(
        "define void @k(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) {{\n  %r \
         = call {TRIANGLE_DATA} @air.intersect.triangle_data(<3 x float> zeroinitializer, <3 x \
         float> <float 0.0, float 0.0, float 1.0>, float 0.0, float 1.0, ptr addrspace(1) %1, \
         ptr addrspace(1) %2, ptr null, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 -1, \
         i32 -1, i32 0, i1 false)\n  %d = extractvalue {TRIANGLE_DATA} %r, 0\n  store i32 %d, \
         ptr addrspace(1) %0, align 4\n{extra}  ret void\n}}\n\ndeclare {TRIANGLE_DATA} \
         @air.intersect.triangle_data(<3 x float>, <3 x float>, float, float, ptr addrspace(1), \
         ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1)\n\n\
         !air.kernel = !{{!0}}\n!0 = !{{ptr @k, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4, !5}}\n!3 = \
         {}\n!4 = {}\n!5 = !{{i32 2, !\"air.intersection_function_table\", \
         !\"air.location_index\", i32 2, i32 1, !\"air.read_write\", !\"air.arg_type_name\", \
         !\"intersection_function_table<triangle_data>\", !\"air.arg_name\", !\"t\"}}\n",
        buffer(0, "air.read_write", "out"),
        structure(1, "primitive"),
    )
}

/// An intersector through an intersection function table argument reads the table's words at
/// each candidate: where the device traces rays it translates, validates, and reads the candidate
/// instance's shader-binding-table record offset and the table's bound range; the table is a
/// read-only `Whole` storage-view buffer slot of its Metal index, which the binder binds like any
/// buffer. Elsewhere the structure's argument is refused, naming what is missing.
#[test]
fn an_intersection_function_table_is_read_per_candidate() {
    let Some(bc) = assemble("table", &table_kernel("")) else {
        return;
    };
    let translated = per_profile(&bc, "k", 1);
    assert!(
        translated.iter().any(|s| s == "lavapipe"),
        "lavapipe traces rays: {translated:?}"
    );
    for (stem, p) in committed_profiles().expect("committed profiles") {
        if !traces(&p) {
            continue;
        }
        let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let slot = SlotRef {
            stage: Stage::Kernel,
            class: SlotClass::Buffer,
            index: 2,
        };
        match a.interface.slots.get(&slot) {
            Some(SlotInterface::Buffer(b)) => {
                assert_eq!(b.access, AccessSet::READ, "{stem}");
                assert_eq!(b.window, Window::Whole, "{stem}");
                assert_eq!(
                    b.form,
                    BufferForm::Descriptor(DescriptorView::Storage),
                    "{stem}"
                );
            }
            other => panic!("{stem}: the table's slot is {other:?}"),
        }
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let dis = spirv_dis(&words);
        for needle in [
            "OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR",
            "OpArrayLength",
        ] {
            assert!(dis.contains(needle), "{stem}: {needle} missing");
        }
    }
}

/// The binder plans a structure argument as the consumer's top-level structure of the Metal
/// structure bound at its slot, at the argument's level, and the table argument as the buffer
/// bound at its slot: the consumer builds the structures (`bind::acceleration`).
#[test]
fn the_binder_writes_the_structure_and_the_table_the_consumer_binds() {
    let Some(bc) = assemble("plan", &table_kernel("")) else {
        return;
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles");
    let text = std::fs::read_to_string(path.join("lavapipe.json")).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let dev = Device::new(desc, DevicePolicy::default()).expect("device");
    let lib = Library::from_air(&bc).expect("library");
    let f = lib.function("k").expect("entry");
    let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    let pipe = dev
        .compile_compute_pipeline(&d, &Budget::standard())
        .expect("compiled");
    let slot = |index| SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    };
    let mut table = StageTable::default();
    for index in [0, 2] {
        table.buffers.insert(
            index,
            BoundBuffer::Buffer {
                length: 16,
                offset: 0,
                gpu_address: 0,
                device_address: None,
                attribute_stride: None,
            },
        );
    }
    let mut state = BindState::default();
    state.stages.insert(Stage::Kernel, table);
    let plan = pipe.plan(&dev, &state.shape()).expect("planned");
    let sources: Vec<&WriteSource> = plan.descriptors.iter().map(|w| &w.source).collect();
    for expected in [
        WriteSource::AccelerationStructure {
            slot: slot(1),
            level: AccelerationStructureLevel::Primitive,
        },
        WriteSource::Buffer { slot: slot(2) },
    ] {
        assert!(sources.contains(&&expected), "{expected:?}: {sources:?}");
    }
}

/// An intersection function table argument read other than as an intersector's table (here
/// compared with null) is refused as an intersection function table at that reader.
#[test]
fn a_table_argument_read_otherwise_is_refused() {
    let extra = "  %q = icmp eq ptr addrspace(1) %2, null\n  %z = zext i1 %q to i32\n  %p = \
                 getelementptr i32, ptr addrspace(1) %0, i64 1\n  store i32 %z, ptr \
                 addrspace(1) %p, align 4\n";
    let Some(bc) = assemble("table-compared", &table_kernel(extra)) else {
        return;
    };
    let u = refusal(&bc, "k");
    assert_eq!(u.feature, Feature::IntersectionFunctionTable, "{u}");
    assert_eq!(u.needs, Needs::NotImplemented, "{u}");
    assert_at_reader(&u);
}

/// `u` is located at an instruction of entry `k`.
fn assert_at_reader(u: &Unsupported) {
    match u.location.as_ref().and_then(Location::in_function) {
        Some((f, Some(_), Some(_))) => assert_eq!(f, &DiagName::new("k"), "{u}"),
        _ => panic!("not at an instruction of k: {u}"),
    }
}

/// A kernel that sets a buffer of its intersection function table argument
/// (`air.set_buffer_intersection_function_table`, the GPU debugger's table-restoring kernels) sets
/// what only an intersection function reads, and linking one is not implemented: an intersection
/// function table, not implemented, at the call on every profile.
#[test]
fn a_table_written_by_set_buffer_is_refused_at_the_call() {
    let text = format!(
        "define void @k(ptr addrspace(1) %0, ptr addrspace(1) %1) {{\n  call void \
         @air.set_buffer_intersection_function_table.p1i8(ptr addrspace(1) %1, ptr addrspace(1) \
         %0, i32 0)\n  ret void\n}}\n\ndeclare void \
         @air.set_buffer_intersection_function_table.p1i8(ptr addrspace(1), ptr addrspace(1), \
         i32)\n\n!air.kernel = !{{!0}}\n!0 = !{{ptr @k, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4}}\n\
         !3 = {}\n!4 = !{{i32 1, !\"air.intersection_function_table\", !\"air.location_index\", \
         i32 1, i32 1, !\"air.read_write\", !\"air.arg_type_name\", \
         !\"intersection_function_table<>\", !\"air.arg_name\", !\"t\"}}\n",
        buffer(0, "air.read", "data"),
    );
    let Some(bc) = assemble("table-set-buffer", &text) else {
        return;
    };
    for (stem, p) in committed_profiles().expect("committed profiles") {
        let u = match compile(&bc, "k", &p) {
            Err(Error::Unsupported(u)) => u,
            Err(other) => panic!("{stem}: {other}"),
            Ok(_) => panic!("{stem}: compiled"),
        };
        assert_eq!(u.feature, Feature::IntersectionFunctionTable, "{stem}: {u}");
        assert_eq!(u.needs, Needs::NotImplemented, "{stem}: {u}");
        assert_at_reader(&u);
    }
}

/// A stage that may link an intersection function (a linked function with a table handle whose
/// module does not decode: a module holding an intersection function does not) refuses an
/// intersector through a table, whose entry could be that function's case, as an intersection
/// function table not implemented, naming the call. A linked module that decodes holds visible
/// functions only, which a table cannot hold, and changes nothing.
#[test]
fn a_table_of_a_stage_that_may_link_an_intersection_function_is_refused() {
    let Some(bc) = assemble("table-linked", &table_kernel("")) else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    let undecodable = b"not an AIR module".to_vec();
    let visible = "define i32 @helper(i32 %x) {\n  ret i32 %x\n}\n\n!air.visible = !{!0}\n!0 = \
                   !{ptr @helper, !1, !3}\n!1 = !{!2}\n!2 = !{!\"air.visible_output\", \
                   !\"air.arg_type_name\", !\"uint\"}\n!3 = !{!4}\n!4 = !{i32 0, \
                   !\"air.visible_input\", !\"air.arg_type_name\", !\"uint\", \
                   !\"air.arg_name\", !\"x\"}\n";
    let Some(visible) = assemble("table-linked-visible", visible) else {
        return;
    };
    for (stem, p) in committed_profiles().expect("committed profiles") {
        if !traces(&p) {
            continue;
        }
        let linked = [LinkedSource::new("intersect_alpha", &undecodable, &none)];
        let compiled = compile_bitcode_linked(
            &bc,
            EntryRef::Name("k"),
            (&none, &linked),
            &p,
            &LowerInputs::new(),
            &mut meter(),
            SPIRV_WORDS,
        );
        let u = match compiled {
            Err(Error::Unsupported(u)) => u,
            Err(other) => panic!("{stem}: {other}"),
            Ok(_) => panic!("{stem}: compiled"),
        };
        assert_eq!(u.feature, Feature::IntersectionFunctionTable, "{stem}: {u}");
        assert_eq!(u.needs, Needs::NotImplemented, "{stem}: {u}");
        assert_eq!(
            u.symbol,
            Some(DiagName::new("air.intersect.triangle_data")),
            "{stem}: {u}"
        );
        let linked = [LinkedSource::new("helper", &visible, &none)];
        compile_bitcode_linked(
            &bc,
            EntryRef::Name("k"),
            (&none, &linked),
            &p,
            &LowerInputs::new(),
            &mut meter(),
            SPIRV_WORDS,
        )
        .unwrap_or_else(|e| panic!("{stem} with a visible function linked: {e}"));
    }
}

/// A motion intersector is not representable (a ray query has no time), naming its callee.
#[test]
fn motion_is_not_representable() {
    let result = "{ i32, float, i32, i32, ptr addrspace(1) }";
    let text = format!(
        "define void @k(ptr addrspace(1) %0, ptr addrspace(1) %1) {{\n  %t = call ptr \
         addrspace(1) @air.get_null_intersection_function_table()\n  %r = call {result} \
         @air.intersect.primitive_motion(<3 x float> zeroinitializer, <3 x float> <float 0.0, \
         float 0.0, float 1.0>, float 0.0, float 1.0, ptr addrspace(1) %1, float 0.5, ptr \
         addrspace(1) %t, ptr null, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 -1, i32 \
         -1, i32 0, i1 false)\n  %d = extractvalue {result} %r, 0\n  store i32 %d, ptr \
         addrspace(1) %0, align 4\n  ret void\n}}\n\ndeclare ptr addrspace(1) \
         @air.get_null_intersection_function_table()\ndeclare {result} \
         @air.intersect.primitive_motion(<3 x float>, <3 x float>, float, float, ptr \
         addrspace(1), float, ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, i32, i32, \
         i32, i32, i1)\n\n!air.kernel = !{{!0}}\n!0 = !{{ptr @k, !1, !2}}\n!1 = !{{}}\n!2 = \
         !{{!3, !4}}\n!3 = {}\n!4 = {}\n",
        buffer(0, "air.read_write", "out"),
        structure(1, "primitive"),
    );
    let Some(bc) = assemble("motion", &text) else {
        return;
    };
    let u = refusal(&bc, "k");
    assert_eq!(u.feature, Feature::RayTracing, "{u}");
    assert_eq!(u.needs, Needs::NotRepresentable, "{u}");
    assert_eq!(
        u.symbol,
        Some(DiagName::new("air.intersect.primitive_motion")),
        "{u}"
    );
}

/// The primitive kernel with `extra` (ending in `ret void`) in place of its `ret void`, after it
/// extracts the primitive-data pointer as `%pd`.
fn data_kernel(extra: &str) -> String {
    let settings = "i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 -1, i32 -1, i32 0, i1 false";
    primitive_kernel(settings).replace(
        "  ret void\n",
        &format!("  %pd = extractvalue {TRIANGLE_DATA} %r, 4\n{extra}"),
    )
}

/// The position of the instruction on the first line of `text`'s `@k` holding `needle`: its
/// block and its index there, counting the body's lines as instructions and labels as blocks.
fn position(text: &str, needle: &str) -> (u32, u32) {
    let body = text
        .split_once("define void @k(")
        .map(|(_, b)| b)
        .expect("a body");
    let (mut block, mut index) = (0, 0);
    for line in body.lines().skip(1) {
        if line.contains(needle) {
            return (block, index);
        }
        if line.starts_with("  ") {
            index += 1;
        } else if line.ends_with(':') {
            (block, index) = (block + 1, 0);
        }
    }
    panic!("{needle} is not in @k");
}

/// The index in the compile's own entry of the instruction at index `at` of `bitcode`'s `@k`:
/// a refusal's location names the entry selection reads (`pipeline::compile`'s
/// `ordered_entry`), whose lockstep barriers renumber the instructions after them. The same on
/// every profile that traces rays (`refusal`).
fn in_ordered_entry(bitcode: &[u8], at: u32) -> u32 {
    let (stem, p) = committed_profiles()
        .expect("committed profiles")
        .into_iter()
        .find(|(_, p)| traces(p))
        .expect("a committed profile that traces rays");
    let e = specialized_entry(
        bitcode,
        "k",
        &p,
        &FunctionConstantValues::defaults(),
        &[],
        &mut meter(),
    )
    .unwrap_or_else(|e| panic!("{stem}: {e}"));
    e.lockstep
        .barriers
        .iter()
        .fold(at, |k, b| if b.raw() <= k { k + 1 } else { k })
}

/// Reading an intersection's primitive data, which no Vulkan structure holds, is refused as ray
/// tracing at the reader: a load or an address computation through the pointer, a compare, a
/// conversion, a merge, a store into a buffer, and a store into thread memory whose bytes are read
/// back or may be (written twice where read back: one store's value is read directly,
/// `canon::forward`).
#[test]
fn reading_primitive_data_is_refused() {
    for (tag, extra, reader) in [
        (
            "data-loaded",
            "  %pw = load i32, ptr addrspace(1) %pd, align 4\n  %p5 = getelementptr i32, ptr \
             addrspace(1) %0, i64 5\n  store i32 %pw, ptr addrspace(1) %p5, align 4\n",
            "%pw = load",
        ),
        (
            "data-indexed",
            "  %pi = getelementptr i32, ptr addrspace(1) %pd, i64 1\n  %pw = load i32, ptr \
             addrspace(1) %pi, align 4\n  store i32 %pw, ptr addrspace(1) %0, align 4\n",
            "%pi = getelementptr",
        ),
        (
            "data-compared",
            "  %q = icmp eq ptr addrspace(1) %pd, null\n  %z = zext i1 %q to i32\n  store i32 \
             %z, ptr addrspace(1) %0, align 4\n",
            "%q = icmp",
        ),
        (
            "data-converted",
            "  %q = ptrtoint ptr addrspace(1) %pd to i64\n  %z = trunc i64 %q to i32\n  store \
             i32 %z, ptr addrspace(1) %0, align 4\n",
            "%q = ptrtoint",
        ),
        (
            "data-selected",
            "  %c = icmp ne i32 %ty, 0\n  %q = select i1 %c, ptr addrspace(1) %pd, ptr \
             addrspace(1) %0\n  store i32 1, ptr addrspace(1) %q, align 4\n",
            "%q = select",
        ),
        (
            "data-in-buffer",
            "  store ptr addrspace(1) %pd, ptr addrspace(1) %0, align 8\n",
            "store ptr addrspace(1) %pd",
        ),
        (
            "data-read-back",
            "  %cell = alloca ptr addrspace(1), align 8\n  store ptr addrspace(1) %pd, ptr \
             %cell, align 8\n  store ptr addrspace(1) %pd, ptr %cell, align 8\n  %back = load \
             ptr addrspace(1), ptr %cell, align 8\n  %bw = load i32, ptr addrspace(1) %back, \
             align 4\n  store i32 %bw, ptr addrspace(1) %0, align 4\n",
            "store ptr addrspace(1) %pd",
        ),
        (
            // A byte offset nothing bounds may read the stored pointer.
            "data-maybe-read",
            "  %cell = alloca { ptr addrspace(1), i32 }, align 8\n  store ptr addrspace(1) %pd, \
             ptr %cell, align 8\n  %o = zext i32 %pr to i64\n  %at = getelementptr i8, ptr \
             %cell, i64 %o\n  %bw = load i32, ptr %at, align 4\n  store i32 %bw, ptr \
             addrspace(1) %0, align 4\n",
            "store ptr addrspace(1) %pd",
        ),
    ] {
        let text = data_kernel(&format!("{extra}  ret void\n"));
        let Some(bc) = assemble(tag, &text) else {
            return;
        };
        let u = refusal(&bc, "k");
        assert_eq!(u.feature, Feature::RayTracing, "{tag}: {u}");
        assert_eq!(u.needs, Needs::NotImplemented, "{tag}: {u}");
        let (block, index) = position(&text, reader);
        assert_eq!(block, 0, "{tag}: the reader is in the entry block");
        assert_eq!(
            u.location,
            Some(Location::function(
                DiagName::new("k"),
                Some(block),
                Some(in_ordered_entry(&bc, index))
            )),
            "{tag}: {u}"
        );
    }
}

/// An intersection's primitive-data pointer that is only moved translates where the device
/// traces rays: extracted and left unread, inserted into an aggregate that a member-wise phi merges
/// (which carries zeros for it) and extracted again, and stored into thread memory whose stored
/// bytes nothing reads (a store that writes nothing). The bytes a read takes are bounded by its
/// constant offset, or by the elements a dynamic array index selects.
#[test]
fn primitive_data_moved_but_never_read_translates() {
    let pair = "{ i32, ptr addrspace(1) }";
    let record = "{ i32, ptr addrspace(1), [4 x i32] }";
    for (tag, extra) in [
        ("data-unread", String::new()),
        (
            // A record whose other member is read back at its constant offset.
            "data-cell",
            format!(
                "  %cell = alloca {pair}, align 8\n  %c0 = getelementptr {pair}, ptr %cell, i64 \
                 0, i32 0\n  store i32 %ty, ptr %c0, align 8\n  %c1 = getelementptr {pair}, \
                 ptr %cell, i64 0, i32 1\n  store ptr addrspace(1) %pd, ptr %c1, align 8\n  \
                 %back = load i32, ptr %c0, align 8\n  %p5 = getelementptr i32, ptr \
                 addrspace(1) %0, i64 5\n  store i32 %back, ptr addrspace(1) %p5, align 4\n"
            ),
        ),
        (
            // A record whose array is written and read at a dynamic index: its elements lie
            // past the stored pointer.
            "data-array",
            format!(
                "  %rec = alloca {record}, align 8\n  %r1 = getelementptr {record}, ptr %rec, \
                 i64 0, i32 1\n  store ptr addrspace(1) %pd, ptr %r1, align 8\n  %k = and i32 \
                 %pr, 3\n  %ke = zext i32 %k to i64\n  %e = getelementptr {record}, ptr %rec, \
                 i64 0, i32 2, i64 %ke\n  store i32 %ty, ptr %e, align 4\n  %e3 = \
                 getelementptr {record}, ptr %rec, i64 0, i32 2, i64 3\n  store i32 7, ptr %e3, \
                 align 4\n  %v = load i32, ptr %e, align 4\n  %p5 = getelementptr i32, ptr \
                 addrspace(1) %0, i64 5\n  store i32 %v, ptr addrspace(1) %p5, align 4\n"
            ),
        ),
        (
            // An aggregate a member-wise phi merges with a constant.
            "data-merged",
            format!(
                "  %hit = icmp ne i32 %ty, 0\n  br i1 %hit, label %a, label %m\na:\n  %x = \
                 insertvalue {pair} undef, i32 %pr, 0\n  %x2 = insertvalue {pair} %x, ptr \
                 addrspace(1) %pd, 1\n  br label %m\nm:\n  %j = phi {pair} [ %x2, %a ], [ \
                 zeroinitializer, %2 ]\n  %j0 = extractvalue {pair} %j, 0\n  %j1 = \
                 extractvalue {pair} %j, 1\n  %y = insertvalue {pair} %j, ptr addrspace(1) %j1, \
                 1\n  %y0 = extractvalue {pair} %y, 0\n  %p5 = getelementptr i32, ptr \
                 addrspace(1) %0, i64 5\n  store i32 %y0, ptr addrspace(1) %p5, align 4\n"
            ),
        ),
    ] {
        let text = data_kernel(&format!("{extra}  ret void\n"));
        let Some(bc) = assemble(tag, &text) else {
            return;
        };
        let translated = per_profile(&bc, "k", 1);
        assert!(
            translated.iter().any(|s| s == "lavapipe"),
            "{tag}: {translated:?}"
        );
    }
}

/// The primitive kernel storing its primitive-data pointer into each of `n` records of an array
/// `%held`, then reading one of the two words beside each record's pointer (clear of it, at a
/// dynamic index) from `reads_from`: `held` itself or `empty`, a second array of the same shape
/// that holds no pointer. Canonicalization keeps every store a read at a dynamic offset into its
/// allocation may reach, so the planning of unread stores sees them all.
fn crowded_kernel(n: u32, reads_from: &str) -> String {
    let array = format!("[{n} x {{ ptr addrspace(1), [2 x i32] }}]");
    let mut extra = format!(
        "  %held = alloca {array}, align 8\n  %empty = alloca {array}, align 8\n  %dz = and i32 \
         %ty, 1\n  %dyn = zext i32 %dz to i64\n"
    );
    for k in 0..n {
        extra.push_str(&format!(
            "  %s{k} = getelementptr inbounds {array}, ptr %held, i64 0, i64 {k}, i32 0\n  store ptr \
             addrspace(1) %pd, ptr %s{k}, align 8\n"
        ));
    }
    for k in 0..n {
        let slot = k + 5;
        extra.push_str(&format!(
            "  %l{k} = getelementptr inbounds {array}, ptr %{reads_from}, i64 0, i64 {k}, i32 1, \
             i64 %dyn\n  %v{k} = load i32, ptr %l{k}, align 4\n  %o{k} = getelementptr i32, ptr \
             addrspace(1) %0, i64 {slot}\n  store i32 %v{k}, ptr addrspace(1) %o{k}, align 4\n"
        ));
    }
    data_kernel(&format!("{extra}  ret void\n"))
}

/// Finding the primitive-data stores nothing reads is metered by the stores each read may reach:
/// `n` reads of an array holding `n` stored pointers charge about `n * n` work units more than the
/// same reads of an array holding none (at least three quarters of it: the two bodies' other work
/// differs a little), so under a cap between the two the first is `LimitExceeded(WorkUnits)`,
/// deterministically, and the second translates.
#[test]
fn unread_store_planning_is_metered_by_the_stores_each_read_reaches() {
    const N: u32 = 64;
    let Some(crowded) = assemble("crowded", &crowded_kernel(N, "held")) else {
        return;
    };
    let Some(apart) = assemble("apart", &crowded_kernel(N, "empty")) else {
        return;
    };
    let (_, p) = committed_profiles()
        .expect("committed profiles")
        .into_iter()
        .find(|(stem, _)| stem == "lavapipe")
        .expect("lavapipe");
    let run = |bitcode: &[u8], m: &mut Meter| {
        compile_bitcode(
            bitcode,
            EntryRef::Name("k"),
            &FunctionConstantValues::defaults(),
            &p,
            &LowerInputs::new(),
            m,
            SPIRV_WORDS,
        )
    };
    let work = |bitcode: &[u8]| {
        let mut m = meter();
        run(bitcode, &mut m).expect("translates");
        m.work()
    };
    let (crowded_work, apart_work) = (work(&crowded), work(&apart));
    let product = u64::from(N * N);
    assert!(
        crowded_work >= apart_work + product * 3 / 4,
        "{crowded_work} work units against {apart_work}: the reads of the crowded array are not \
         charged for the {product} store visits"
    );
    let cap = apart_work + product / 2;
    let mut capped = Meter::with_caps(&Budget::default(), cap, LEDGER_BYTES);
    assert!(run(&apart, &mut capped).is_ok());
    let mut capped = Meter::with_caps(&Budget::default(), cap, LEDGER_BYTES);
    match run(&crowded, &mut capped) {
        Err(Error::LimitExceeded(l)) => assert_eq!(l, Limit::WorkUnits),
        other => panic!("{other:?}"),
    }
}

/// The record [`select_tree_kernel`] stores into and reads from: a pointer, then 64 words.
const TREE_RECORD: &str = "{ ptr addrspace(1), [64 x i32] }";

/// The primitive kernel storing its primitive-data pointer at the start of `%held` (a
/// [`TREE_RECORD`]), then reading one word through a tree of `depth` selects over `tree_in`:
/// `held` itself or `empty`, a second record that holds no pointer. From a dynamic word of the
/// record's array, each level selects the word one or two past the level below, so both arms of
/// every select share the level below; every word read lies past the stored pointer.
fn select_tree_kernel(depth: u32, tree_in: &str) -> String {
    let mut extra = format!(
        "  %held = alloca {TREE_RECORD}, align 8\n  %empty = alloca {TREE_RECORD}, align 8\n  \
         store ptr addrspace(1) %pd, ptr %held, align 8\n  %cw = load i32, ptr addrspace(1) %0, \
         align 4\n  %c = icmp ne i32 %cw, 0\n  %ix = zext i32 %cw to i64\n  %g0 = getelementptr \
         inbounds {TREE_RECORD}, ptr %{tree_in}, i64 0, i32 1, i64 %ix\n"
    );
    for k in 1..=depth {
        let below = k - 1;
        extra.push_str(&format!(
            "  %x{k} = getelementptr i32, ptr %g{below}, i64 1\n  %y{k} = getelementptr i32, ptr \
             %g{below}, i64 2\n  %g{k} = select i1 %c, ptr %x{k}, ptr %y{k}\n"
        ));
    }
    extra.push_str(&format!(
        "  %v = load i32, ptr %g{depth}, align 4\n  %o = getelementptr i32, ptr addrspace(1) %0, \
         i64 5\n  store i32 %v, ptr addrspace(1) %o, align 4\n  ret void\n"
    ));
    data_kernel(&extra)
}

/// The range of a read through a tree of selects that share their arms' definitions is walked
/// once per definition: at depth 30, whether the tree is over the record holding the primitive-data
/// pointer or over the other, the body translates under the default meter, the stored pointer a
/// store nothing reads. Each level of the tree costs at most `PER_LEVEL` work units (about 328
/// measured: 264 before the device lockstep pass, which reads the uniformity of every instruction
/// of a kernel whose buffer accesses may meet, here the stores to `%0` and the read of its first
/// word; 180 before the demanded-bits pass, which reads each instruction of the finished module
/// once per pass, two or three passes here), and the tree over the pointer's record at most
/// `WALK_PER_LEVEL` more per level than the other (about 15): a walk of every path would visit 2^30
/// of them.
#[test]
fn a_shared_select_tree_is_walked_once_per_definition() {
    const PER_LEVEL: u64 = 400;
    const WALK_PER_LEVEL: u64 = 32;
    let (_, p) = committed_profiles()
        .expect("committed profiles")
        .into_iter()
        .find(|(stem, _)| stem == "lavapipe")
        .expect("lavapipe");
    let mut deepest = Vec::new();
    for tree_in in ["held", "empty"] {
        let mut work = Vec::new();
        for depth in [15, 30] {
            let Some(bc) = assemble("select-tree", &select_tree_kernel(depth, tree_in)) else {
                return;
            };
            let mut m = meter();
            let compiled = compile_bitcode(
                &bc,
                EntryRef::Name("k"),
                &FunctionConstantValues::defaults(),
                &p,
                &LowerInputs::new(),
                &mut m,
                SPIRV_WORDS,
            );
            assert!(compiled.is_ok(), "{tree_in} at depth {depth}: {compiled:?}");
            work.push(m.work());
        }
        let [shallow, deep] = work[..] else {
            unreachable!("two depths");
        };
        assert!(
            deep <= shallow + 15 * PER_LEVEL,
            "{tree_in}: {shallow} work units at depth 15, {deep} at depth 30"
        );
        deepest.push(deep);
    }
    let [held, empty] = deepest[..] else {
        unreachable!("two trees");
    };
    assert!(
        held <= empty + 30 * WALK_PER_LEVEL,
        "the range walk of 30 levels: {held} work units against {empty}"
    );
}

/// The primitive kernel storing its primitive-data pointer at the start of `%held` (a
/// [`TREE_RECORD`]), then building a chain of `n` definitions over `reads_in` (`held` itself or
/// `empty`, a record that holds no pointer): from a dynamic word of the record's array, each
/// next word. It reads the deepest word first, then every other one, shallowest first; every
/// word read lies past the stored pointer.
fn chain_kernel(n: u32, reads_in: &str) -> String {
    let mut extra = format!(
        "  %held = alloca {TREE_RECORD}, align 8\n  %empty = alloca {TREE_RECORD}, align 8\n  \
         store ptr addrspace(1) %pd, ptr %held, align 8\n  %cw = load i32, ptr addrspace(1) %0, \
         align 4\n  %ix = zext i32 %cw to i64\n  %g0 = getelementptr inbounds {TREE_RECORD}, \
         ptr %{reads_in}, i64 0, i32 1, i64 %ix\n"
    );
    for k in 1..=n {
        let below = k - 1;
        extra.push_str(&format!(
            "  %g{k} = getelementptr i32, ptr %g{below}, i64 1\n"
        ));
    }
    for k in std::iter::once(n).chain(1..n) {
        extra.push_str(&format!(
            "  %v{k} = load i32, ptr %g{k}, align 4\n  %o{k} = getelementptr i32, ptr \
             addrspace(1) %0, i64 5\n  store i32 %v{k}, ptr addrspace(1) %o{k}, align 4\n"
        ));
    }
    data_kernel(&format!("{extra}  ret void\n"))
}

/// Every read through a chain of 300 definitions shares one range walk, on `lavapipe` under the
/// default meter. Over the record holding the primitive-data pointer the body translates, the
/// stored pointer a store nothing reads: the deepest read is bounded through all 301 definitions
/// (the walk has no depth bound), and the 299 other reads reuse its ranges. Against the same chain
/// over the record holding no pointer, whose reads reach no candidate store, it costs at most
/// `PER_DEFINITION` work units more per definition (about 11 measured, the planner's per-read work
/// included), where walking each read's chain again would visit about 45,000 definitions.
#[test]
fn a_chain_of_reads_shares_one_range_walk() {
    const N: u32 = 300;
    const PER_DEFINITION: u64 = 16;
    let (_, p) = committed_profiles()
        .expect("committed profiles")
        .into_iter()
        .find(|(stem, _)| stem == "lavapipe")
        .expect("lavapipe");
    let mut work = Vec::new();
    for reads_in in ["held", "empty"] {
        let Some(bc) = assemble("read-chain", &chain_kernel(N, reads_in)) else {
            return;
        };
        let mut m = meter();
        let compiled = compile_bitcode(
            &bc,
            EntryRef::Name("k"),
            &FunctionConstantValues::defaults(),
            &p,
            &LowerInputs::new(),
            &mut m,
            SPIRV_WORDS,
        );
        assert!(compiled.is_ok(), "{reads_in}: {compiled:?}");
        work.push(m.work());
    }
    let [held, empty] = work[..] else {
        unreachable!("two chains");
    };
    let definitions = u64::from(N) + 1;
    assert!(
        held <= empty + definitions * PER_DEFINITION,
        "the range walk of {definitions} definitions: {held} work units against {empty}"
    );
}

/// A read through a pointer that a loop carries is bounded by nothing: its range walk meets the
/// loop's phi again and ends, and the primitive-data store it may read is kept and refused at the
/// store, as ray tracing.
#[test]
fn a_pointer_cycle_bounds_nothing() {
    let text = data_kernel(&format!(
        "  %held = alloca {TREE_RECORD}, align 8\n  store ptr addrspace(1) %pd, ptr %held, align \
         8\n  %g0 = getelementptr inbounds {TREE_RECORD}, ptr %held, i64 0, i32 1, i64 0\n  br \
         label %loop\nloop:\n  %q = phi ptr [ %g0, %2 ], [ %qn, %loop ]\n  %v = load i32, ptr \
         %q, align 4\n  %qn = getelementptr i32, ptr %q, i64 1\n  %more = icmp ult i32 %v, 7\n  \
         br i1 %more, label %loop, label %done\ndone:\n  %o = getelementptr i32, ptr \
         addrspace(1) %0, i64 5\n  store i32 %v, ptr addrspace(1) %o, align 4\n  ret void\n"
    ));
    let Some(bc) = assemble("pointer-cycle", &text) else {
        return;
    };
    let u = refusal(&bc, "k");
    assert_eq!(u.feature, Feature::RayTracing, "{u}");
    assert_eq!(u.needs, Needs::NotImplemented, "{u}");
    let (block, index) = position(&text, "store ptr addrspace(1) %pd");
    assert_eq!(
        u.location,
        Some(Location::function(
            DiagName::new("k"),
            Some(block),
            Some(index)
        )),
        "{u}"
    );
}

/// `k(device uint *out [[buffer(0)]], ...)` with `params` after `out` and their ABI metadata
/// `meta` (parameter 1 on), whose body is `body` (ending in `ret void`), with `decls` after it.
fn kernel_with(params: &str, meta: &[String], body: &str, decls: &str) -> String {
    let mut nodes = String::from("!3");
    let mut defs = format!("!3 = {}\n", buffer(0, "air.read_write", "out"));
    for (i, m) in meta.iter().enumerate() {
        nodes.push_str(&format!(", !{}", i + 4));
        defs.push_str(&format!("!{} = {m}\n", i + 4));
    }
    format!(
        "define void @k(ptr addrspace(1) %0{params}) {{\n{body}}}\n\n{decls}\n\n!air.kernel = \
         !{{!0}}\n!0 = !{{ptr @k, !1, !2}}\n!1 = !{{}}\n!2 = !{{{nodes}}}\n{defs}"
    )
}

/// A body calling `callee` of result type `result` with `args` after the null table `%t`, and
/// storing the result's type (member 0).
fn call_body(result: &str, callee: &str, args: &str) -> String {
    format!(
        "  %t = call ptr addrspace(1) @air.get_null_intersection_function_table()\n  %r = call \
         {result} @{callee}({args})\n  %d = extractvalue {result} %r, 0\n  store i32 %d, ptr \
         addrspace(1) %0, align 4\n  ret void\n"
    )
}

/// The declarations of the null table and of `callee`.
fn call_decls(result: &str, callee: &str, operands: &str) -> String {
    format!(
        "declare ptr addrspace(1) @air.get_null_intersection_function_table()\ndeclare {result} \
         @{callee}({operands})"
    )
}

/// The base operands of a primitive intersector over `%1`, before the table.
const RAY: &str = "<3 x float> zeroinitializer, <3 x float> <float 0.0, float 0.0, float 1.0>, \
                   float 0.0, float 1.0";
/// The nine settings (a triangle geometry type) and accept-any.
const SETTINGS: &str = "i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 -1, i32 -1, i32 0, i1 false";
const SETTINGS_TYPES: &str = "i32, i32, i32, i32, i32, i32, i32, i32, i32, i1";

/// The refusal of `text` on every profile that traces rays: `feature`, not implemented, naming
/// `callee` when it is given.
fn assert_ray_refusal(tag: &str, text: &str, feature: Feature, callee: Option<&str>) {
    let Some(bc) = assemble(tag, text) else {
        return;
    };
    let u = refusal(&bc, "k");
    assert_eq!(u.feature, feature, "{tag}: {u}");
    assert_eq!(u.needs, Needs::NotImplemented, "{tag}: {u}");
    if let Some(callee) = callee {
        assert_eq!(u.symbol, Some(DiagName::new(callee)), "{tag}: {u}");
    }
}

/// An intersector with an intersection function buffer (Metal 4) runs intersection functions,
/// so it is an intersection function table refusal, naming the call.
#[test]
fn function_buffers_are_intersection_function_tables() {
    let result = "{ i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 }";
    let callee = "air.intersect.intersection_function_buffer.triangle_data";
    let args = format!(
        "{RAY}, ptr addrspace(1) %1, ptr addrspace(1) %2, i64 64, i64 8, ptr null, i64 0, \
         {SETTINGS}, i32 0, i32 1"
    );
    let operands = format!(
        "<3 x float>, <3 x float>, float, float, ptr addrspace(1), ptr addrspace(1), i64, i64, \
         ptr, i64, {SETTINGS_TYPES}, i32, i32"
    );
    let text = kernel_with(
        ", ptr addrspace(1) %1, ptr addrspace(1) %2",
        &[
            structure(1, "primitive"),
            buffer(2, "air.read", "functions"),
        ],
        &call_body(result, callee, &args),
        &call_decls(result, callee, &operands),
    );
    assert_ray_refusal(
        "function-buffer",
        &text,
        Feature::IntersectionFunctionTable,
        Some(callee),
    );
}

/// Motion decides before an intersection function buffer: implementing the buffer would not make a
/// ray query trace in time, so the call is not representable.
#[test]
fn motion_through_a_function_buffer_is_not_representable() {
    let result = "{ i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 }";
    let callee = "air.intersect.intersection_function_buffer.triangle_data.primitive_motion";
    let args = format!(
        "{RAY}, ptr addrspace(1) %1, float 0.5, ptr addrspace(1) %2, i64 64, i64 8, ptr null, \
         i64 0, {SETTINGS}, i32 0, i32 1"
    );
    let operands = format!(
        "<3 x float>, <3 x float>, float, float, ptr addrspace(1), float, ptr addrspace(1), i64, \
         i64, ptr, i64, {SETTINGS_TYPES}, i32, i32"
    );
    let text = kernel_with(
        ", ptr addrspace(1) %1, ptr addrspace(1) %2",
        &[
            structure(1, "primitive"),
            buffer(2, "air.read", "functions"),
        ],
        &call_body(result, callee, &args),
        &call_decls(result, callee, &operands),
    );
    let Some(bc) = assemble("function-buffer-motion", &text) else {
        return;
    };
    let u = refusal(&bc, "k");
    assert_eq!(u.feature, Feature::RayTracing, "{u}");
    assert_eq!(u.needs, Needs::NotRepresentable, "{u}");
    assert_eq!(u.symbol, Some(DiagName::new(callee)), "{u}");
}

/// Curves are ray tracing not implemented yet, refused at the call.
#[test]
fn curves_are_refused_at_the_call() {
    let result = "{ i32, float, i32, i32, ptr addrspace(1), float }";
    let callee = "air.intersect.curve_data";
    let args =
        format!("{RAY}, ptr addrspace(1) %1, ptr addrspace(1) %t, ptr null, i64 0, {SETTINGS}");
    let operands = format!(
        "<3 x float>, <3 x float>, float, float, ptr addrspace(1), ptr addrspace(1), ptr, i64, \
         {SETTINGS_TYPES}"
    );
    let text = kernel_with(
        ", ptr addrspace(1) %1",
        &[structure(1, "primitive")],
        &call_body(result, callee, &args),
        &call_decls(result, callee, &operands),
    );
    assert_ray_refusal("curves", &text, Feature::RayTracing, Some(callee));
}

/// The multi-level intersector's result: the base fields and its instance count.
const MULTI_LEVEL: &str = "{ i32, float, i32, i32, ptr addrspace(1), i8 }";
const MULTI_LEVEL_CALLEE: &str = "air.intersect.multi_level_instancing";

/// `k(device uint *out [[buffer(0)]], instance_acceleration_structure scene [[buffer(1)]])` with
/// a multi-level intersector of `max_levels` (an `i8` operand, `%m` when computed) writing its
/// instance ids into thread arrays of two elements, storing the type, the instance count, the
/// instance id at a dynamic index (`type & 1`, so the ids' array is indexed at run time), and
/// the first user instance id.
fn multi_level_kernel(max_levels: &str, level: &str) -> String {
    let args = format!(
        "{RAY}, ptr addrspace(1) %1, i32 255, ptr addrspace(1) %t, ptr null, i64 0, i8 \
         {max_levels}, ptr %ids, ptr %users, {SETTINGS}, i1 false"
    );
    let operands = format!(
        "<3 x float>, <3 x float>, float, float, ptr addrspace(1), i32, ptr addrspace(1), ptr, \
         i64, i8, ptr, ptr, {SETTINGS_TYPES}, i1"
    );
    let body = format!(
        "  %ids = alloca [2 x i32], align 4\n  %users = alloca [2 x i32], align 4\n  %mw = load \
         i32, ptr addrspace(1) %0, align 4\n  %m = trunc i32 %mw to i8\n  %t = call ptr \
         addrspace(1) @air.get_null_intersection_function_table()\n  %r = call {MULTI_LEVEL} \
         @{MULTI_LEVEL_CALLEE}({args})\n  %ty = extractvalue {MULTI_LEVEL} %r, 0\n  store i32 \
         %ty, ptr addrspace(1) %0, align 4\n  %n = extractvalue {MULTI_LEVEL} %r, 5\n  %nw = \
         zext i8 %n to i32\n  %p1 = getelementptr i32, ptr addrspace(1) %0, i64 1\n  store i32 \
         %nw, ptr addrspace(1) %p1, align 4\n  %k = and i32 %ty, 1\n  %ke = zext i32 %k to \
         i64\n  %ip = getelementptr [2 x i32], ptr %ids, i64 0, i64 %ke\n  %id = load i32, ptr \
         %ip, align 4\n  %p2 = getelementptr i32, ptr addrspace(1) %0, i64 2\n  store i32 %id, \
         ptr addrspace(1) %p2, align 4\n  %user = load i32, ptr %users, align 4\n  %p3 = \
         getelementptr i32, ptr addrspace(1) %0, i64 3\n  store i32 %user, ptr addrspace(1) \
         %p3, align 4\n  ret void\n"
    );
    kernel_with(
        ", ptr addrspace(1) %1",
        &[structure(1, level)],
        &body,
        &call_decls(MULTI_LEVEL, MULTI_LEVEL_CALLEE, &operands),
    )
}

/// A multi-level intersector traverses the one level of instances a Vulkan top-level structure
/// holds: it translates where the device traces rays, reading the committed instance's index and
/// custom index into the first element of its id arrays (one read back at a run-time index, the
/// other at a constant one), and is refused at the structure's argument, naming the missing
/// features, where it does not (MoltenVK among them).
#[test]
fn multi_level_instancing_reads_one_level() {
    let Some(bc) = assemble("multi-level", &multi_level_kernel("3", "instance")) else {
        return;
    };
    let translated = per_profile(&bc, "k", 1);
    assert!(translated.iter().any(|s| s == "lavapipe"), "{translated:?}");
    assert!(
        !translated.iter().any(|s| is_moltenvk_capture(s)),
        "{translated:?}"
    );
    for (stem, p) in committed_profiles().expect("committed profiles") {
        if !translated.contains(&stem) {
            continue;
        }
        let a = compile(&bc, "k", &p).expect("compiled");
        let (_, words) = link_and_validate(&a, &p).expect("valid");
        let dis = spirv_dis(&words);
        for needle in [
            "OpRayQueryGetIntersectionInstanceIdKHR",
            "OpRayQueryGetIntersectionInstanceCustomIndexKHR",
        ] {
            assert!(dis.contains(needle), "{stem}: {needle} missing");
        }
    }
}

/// One instruction of a module: its opcode, its operand words, and the label of the block it is
/// in (`None` outside a function body).
struct Inst {
    op: u32,
    operands: Vec<u32>,
    block: Option<u32>,
}

/// The instructions of `words` (after the header), each with its block.
fn module_insts(words: &[u32]) -> Vec<Inst> {
    let mut out = Vec::new();
    let mut block = None;
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        assert!(count > 0, "malformed module at word {at}");
        let op = words[at] & 0xffff;
        let operands = words[at + 1..at + count].to_vec();
        if op == crate::spv::Op::Label as u32 {
            block = Some(operands[0]);
        } else if op == crate::spv::Op::FunctionEnd as u32 {
            block = None;
        }
        out.push(Inst {
            op,
            operands,
            block,
        });
        at += count;
    }
    out
}

/// The opcodes [`assert_one_level`] reads a result of: each has a result type, then its id.
const VALUES: [crate::spv::Op; 12] = [
    crate::spv::Op::Constant,
    crate::spv::Op::ConstantFalse,
    crate::spv::Op::ConstantNull,
    crate::spv::Op::Phi,
    crate::spv::Op::Select,
    crate::spv::Op::CompositeConstruct,
    crate::spv::Op::INotEqual,
    crate::spv::Op::AccessChain,
    crate::spv::Op::Load,
    crate::spv::Op::RayQueryGetIntersectionTypeKHR,
    crate::spv::Op::RayQueryGetIntersectionInstanceIdKHR,
    crate::spv::Op::RayQueryGetIntersectionInstanceCustomIndexKHR,
];

/// The multi-level values in a module of [`multi_level_reads`]: the record's instance count is the
/// constant 1 in the record selected when the committed intersection is a hit (the null record
/// otherwise); the committed instance's index and custom index are each stored once, at the element
/// the kernel reads as element 0 of the ids and of the user ids, and nothing else is stored into
/// either array; and both stores are in a block entered only by the hit's branch, whose other edge
/// is the selection's merge, so a miss writes neither.
fn assert_one_level(stem: &str, words: &[u32]) {
    use crate::spv::Op;
    let insts = module_insts(words);
    let defined = |id: u32| {
        insts
            .iter()
            .find(|i| VALUES.iter().any(|&v| i.op == v as u32) && i.operands[1] == id)
    };
    let def = |id: u32| -> &Inst {
        defined(id).unwrap_or_else(|| panic!("{stem}: %{id} is not a value read here"))
    };
    let constant = |id: u32| {
        defined(id)
            .filter(|d| d.op == Op::Constant as u32)
            .map(|d| d.operands[2])
    };
    let is = |op: Op, id: u32| defined(id).is_some_and(|d| d.op == op as u32);
    let only = |op: Op, what: &str, keep: &dyn Fn(&Inst) -> bool| -> &Inst {
        let found: Vec<&Inst> = insts
            .iter()
            .filter(|i| i.op == op as u32 && keep(i))
            .collect();
        assert_eq!(found.len(), 1, "{stem}: {what}");
        found[0]
    };
    // A read of the committed intersection (`RayQueryCommittedIntersectionKHR`, 1).
    let committed = |i: &Inst| constant(i.operands[3]) == Some(1);
    let ty = only(
        Op::RayQueryGetIntersectionTypeKHR,
        "committed type",
        &committed,
    )
    .operands[1];
    let id = only(
        Op::RayQueryGetIntersectionInstanceIdKHR,
        "instance id",
        &|_| true,
    );
    let custom = only(
        Op::RayQueryGetIntersectionInstanceCustomIndexKHR,
        "custom index",
        &|_| true,
    );
    assert!(committed(id) && committed(custom), "{stem}: candidate ids");
    let hit = only(Op::INotEqual, "the hit", &|i| {
        i.operands[2] == ty && constant(i.operands[3]) == Some(0)
    })
    .operands[1];

    // The record: one level of instances on a hit, the null record on a miss.
    let record = only(Op::Select, "the record", &|i| i.operands[2] == hit);
    let made = def(record.operands[3]);
    assert_eq!(made.op, Op::CompositeConstruct as u32, "{stem}: hit record");
    let count = *made.operands.last().expect("members");
    assert_eq!(constant(count), Some(1), "{stem}: instance count");
    assert!(
        is(Op::ConstantNull, record.operands[4]),
        "{stem}: miss record"
    );

    // The value itself or a phi carrying it out of the traced branch.
    let carries = |value: u32, v: u32| {
        v == value
            || insts.iter().any(|i| {
                i.op == Op::Phi as u32
                    && i.operands[1] == v
                    && i.operands[2..].chunks(2).any(|c| c[0] == value)
            })
    };
    let store_of =
        |value: u32, what: &str| only(Op::Store, what, &|i| carries(value, i.operands[1]));
    let id_store = store_of(id.operands[1], "one store of the instance id");
    let custom_store = store_of(custom.operands[1], "one store of the custom index");
    // A pointer's variable and index: an access chain of one constant index into a variable.
    let element = |pointer: u32| {
        let chain = def(pointer);
        assert_eq!(chain.op, Op::AccessChain as u32, "{stem}: element pointer");
        assert_eq!(chain.operands.len(), 4, "{stem}: one index");
        let at = constant(chain.operands[3]).unwrap_or_else(|| panic!("{stem}: dynamic index"));
        (chain.operands[2], at)
    };
    let (id_at, custom_at) = (
        element(id_store.operands[0]),
        element(custom_store.operands[0]),
    );
    let arrays = [id_at.0, custom_at.0];
    // The pointers into the arrays' variables, and what reads and writes through them.
    let into_arrays = |pointer: u32| {
        insts.iter().any(|c| {
            c.op == Op::AccessChain as u32
                && c.operands[1] == pointer
                && arrays.contains(&c.operands[2])
        })
    };
    // The kernel reads ids[0], ids[1], users[0], users[1] in that order.
    let reads: Vec<(u32, u32)> = insts
        .iter()
        .filter(|i| i.op == Op::Load as u32 && into_arrays(i.operands[2]))
        .map(|i| element(i.operands[2]))
        .collect();
    assert_eq!(reads.len(), 4, "{stem}: {reads:?}");
    assert_eq!(id_at, reads[0], "{stem}: ids[0] {reads:?}");
    assert_eq!(custom_at, reads[2], "{stem}: users[0] {reads:?}");
    let writes = insts
        .iter()
        .filter(|i| i.op == Op::Store as u32 && into_arrays(i.operands[0]))
        .count();
    assert_eq!(writes, 2, "{stem}: stores into the id arrays");

    // Both stores in one block, entered only by the hit's branch.
    let block = id_store.block.expect("in a function");
    assert_eq!(custom_store.block, Some(block), "{stem}: one block");
    let entry = only(Op::BranchConditional, "the hit's branch", &|i| {
        i.operands[1] == block
    });
    let condition = entry.operands[0];
    let on_hit = condition == hit
        || is(Op::Phi, condition)
            && def(condition).operands[2..]
                .chunks(2)
                .all(|c| c[0] == hit || is(Op::ConstantFalse, c[0]));
    assert!(on_hit, "{stem}: the store block is entered on the hit");
    let at = insts
        .iter()
        .position(|i| std::ptr::eq(i, entry))
        .expect("the branch");
    assert_eq!(
        insts[at - 1].op,
        Op::SelectionMerge as u32,
        "{stem}: header"
    );
    assert_eq!(
        insts[at - 1].operands[0],
        entry.operands[2],
        "{stem}: a miss goes to the merge"
    );
    let targets = insts
        .iter()
        .filter(|i| {
            (i.op == Op::Branch as u32 && i.operands[0] == block)
                || (i.op == Op::BranchConditional as u32 && i.operands[1..3].contains(&block))
                || (i.op == Op::Switch as u32 && i.operands[1..].contains(&block))
        })
        .count();
    assert_eq!(targets, 1, "{stem}: one way into the store block");
}

/// `k(device uint *out [[buffer(0)]], instance_acceleration_structure scene [[buffer(1)]])` with
/// a multi-level intersector of `max_levels<3>` storing the instance count and then reading
/// both elements of its ids and of its user ids, each at a constant index.
fn multi_level_reads() -> String {
    let args = format!(
        "{RAY}, ptr addrspace(1) %1, i32 255, ptr addrspace(1) %t, ptr null, i64 0, i8 3, ptr \
         %ids, ptr %users, {SETTINGS}, i1 false"
    );
    let operands = format!(
        "<3 x float>, <3 x float>, float, float, ptr addrspace(1), i32, ptr addrspace(1), ptr, \
         i64, i8, ptr, ptr, {SETTINGS_TYPES}, i1"
    );
    let mut body = format!(
        "  %ids = alloca [2 x i32], align 4\n  %users = alloca [2 x i32], align 4\n  %t = call \
         ptr addrspace(1) @air.get_null_intersection_function_table()\n  %r = call \
         {MULTI_LEVEL} @{MULTI_LEVEL_CALLEE}({args})\n  %n = extractvalue {MULTI_LEVEL} %r, 5\n  \
         %nw = zext i8 %n to i32\n  store i32 %nw, ptr addrspace(1) %0, align 4\n"
    );
    for (k, (array, at)) in [("ids", 0), ("ids", 1), ("users", 0), ("users", 1)]
        .into_iter()
        .enumerate()
    {
        body.push_str(&format!(
            "  %e{k} = getelementptr [2 x i32], ptr %{array}, i64 0, i64 {at}\n  %v{k} = load \
             i32, ptr %e{k}, align 4\n  %o{k} = getelementptr i32, ptr addrspace(1) %0, i64 {}\n  \
             store i32 %v{k}, ptr addrspace(1) %o{k}, align 4\n",
            k + 1
        ));
    }
    body.push_str("  ret void\n");
    kernel_with(
        ", ptr addrspace(1) %1",
        &[structure(1, "instance")],
        &body,
        &call_decls(MULTI_LEVEL, MULTI_LEVEL_CALLEE, &operands),
    )
}

/// The multi-level values in SPIR-V: on every profile that traces rays, the instance count is 1 on
/// a hit and 0 on a miss, the committed instance's index and custom index are written to element 0
/// of the ids and user ids only, and a miss writes neither array ([`assert_one_level`]).
#[test]
fn multi_level_values_are_one_level_written_on_a_hit() {
    let Some(bc) = assemble("multi-level-values", &multi_level_reads()) else {
        return;
    };
    let mut checked = 0;
    for (stem, p) in committed_profiles().expect("committed profiles") {
        if !traces(&p) {
            continue;
        }
        let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_one_level(&stem, &words);
        checked += 1;
    }
    assert!(checked > 0, "no profile traces rays");
}

/// The intersection record Metal's compiler keeps a multi-level triangle intersector's result in
/// (`intersection_result<instancing, max_levels<3>, triangle_data>`): the base fields with the
/// primitive-data pointer, the instance count with two levels of instance and user instance
/// ids, and the triangle data, packed.
const RECORD: &str =
    "<{ { i32, float, i32, i32, ptr addrspace(1) }, { i32, [2 x i32], [2 x i32] }, \
                      [4 x i8], <{ <2 x float>, i8 }>, [7 x i8] }>";
const MULTI_LEVEL_TRIANGLE: &str =
    "{ i32, float, i32, i32, ptr addrspace(1), i8, <2 x float>, i1 }";

/// A multi-level intersector in the shape Metal's compiler gives
/// `intersector<instancing, max_levels<3>, triangle_data>` (read from compiling our own MSL): its
/// id arrays live in an intersection record whose primitive-data slot is zeroed, the ids are copied
/// into a second record, and the copy is read at constant offsets and at the last level's run-time
/// index. The result's primitive-data pointer is copied too, as the compiled intersectors the
/// census found do. It translates where the device traces rays, the copied pointer a store nothing
/// reads, and is refused at the structure's argument elsewhere.
#[test]
fn a_compiled_multi_level_record_translates() {
    let callee = "air.intersect.multi_level_instancing.triangle_data";
    let field = |rec: &str, path: &str| {
        format!("getelementptr inbounds {RECORD}, ptr %{rec}, i64 0, {path}")
    };
    let body = format!(
        "  %rec = alloca {RECORD}, align 8\n  %copy = alloca {RECORD}, align 8\n  %t = call ptr \
         addrspace(1) @air.get_null_intersection_function_table()\n  %d0 = {}\n  store ptr \
         addrspace(1) null, ptr %d0, align 8\n  %ids = {}\n  %users = {}\n  %r = call \
         {MULTI_LEVEL_TRIANGLE} @{callee}({RAY}, ptr addrspace(1) %1, i32 255, ptr addrspace(1) \
         %t, ptr null, i64 0, i8 3, ptr %ids, ptr %users, {SETTINGS}, i1 false)\n  %ty = \
         extractvalue {MULTI_LEVEL_TRIANGLE} %r, 0\n  %pd = extractvalue {MULTI_LEVEL_TRIANGLE} \
         %r, 4\n  %n = extractvalue {MULTI_LEVEL_TRIANGLE} %r, 5\n  %nw = zext i8 %n to i32\n  \
         %c0 = {}\n  store i32 %ty, ptr %c0, align 8\n  %c4 = {}\n  store ptr addrspace(1) %pd, \
         ptr %c4, align 8\n  %id0 = load i32, ptr %ids, align 4\n  %ip1 = {}\n  %id1 = load \
         i32, ptr %ip1, align 8\n  %cid0 = {}\n  store i32 %id0, ptr %cid0, align 4\n  %cid1 = \
         {}\n  store i32 %id1, ptr %cid1, align 8\n  store i32 %ty, ptr addrspace(1) %0, align \
         4\n  %p1 = getelementptr i32, ptr addrspace(1) %0, i64 1\n  store i32 %nw, ptr \
         addrspace(1) %p1, align 4\n  %miss = icmp eq i8 %n, 0\n  br i1 %miss, label %done, \
         label %hit\nhit:\n  %last = add nsw i32 %nw, -1\n  %le = zext i32 %last to i64\n  %lp \
         = {}\n  %lid = load i32, ptr %lp, align 4\n  br label %done\ndone:\n  %id = phi i32 [ \
         %lid, %hit ], [ -1, %2 ]\n  %p2 = getelementptr i32, ptr addrspace(1) %0, i64 2\n  store \
         i32 %id, ptr addrspace(1) %p2, align 4\n  %tyb = load i32, ptr %c0, align 8\n  %p3 = \
         getelementptr i32, ptr addrspace(1) %0, i64 3\n  store i32 %tyb, ptr addrspace(1) %p3, \
         align 4\n  ret void\n",
        field("rec", "i32 0, i32 4"),
        field("rec", "i32 1, i32 1, i64 0"),
        field("rec", "i32 1, i32 2, i64 0"),
        field("copy", "i32 0, i32 0"),
        field("copy", "i32 0, i32 4"),
        field("rec", "i32 1, i32 1, i64 1"),
        field("copy", "i32 1, i32 1, i64 0"),
        field("copy", "i32 1, i32 1, i64 1"),
        field("copy", "i32 1, i32 1, i64 %le"),
    );
    let operands = format!(
        "<3 x float>, <3 x float>, float, float, ptr addrspace(1), i32, ptr addrspace(1), ptr, \
         i64, i8, ptr, ptr, {SETTINGS_TYPES}, i1"
    );
    let text = kernel_with(
        ", ptr addrspace(1) %1",
        &[structure(1, "instance")],
        &body,
        &call_decls(MULTI_LEVEL_TRIANGLE, callee, &operands),
    );
    let Some(bc) = assemble("multi-level-record", &text) else {
        return;
    };
    let translated = per_profile(&bc, "k", 1);
    assert!(translated.iter().any(|s| s == "lavapipe"), "{translated:?}");
}

/// A multi-level intersector over a primitive structure is refused at the call, as ray tracing
/// not implemented (its level is not the structure's).
#[test]
fn multi_level_instancing_over_a_primitive_structure_is_refused() {
    assert_ray_refusal(
        "multi-level-primitive",
        &multi_level_kernel("3", "primitive"),
        Feature::RayTracing,
        Some(MULTI_LEVEL_CALLEE),
    );
}

/// `max_levels` is the template parameter of `max_levels<n>`, which Metal's compiler accepts
/// from 2 to 32: a run-time value is `InvalidInput(Intrinsic(ImmediateNotConstant))` and a constant
/// outside that range (0, 1, 33, and the `i8` -1) is `InvalidInput(Intrinsic(SignatureMismatch))`,
/// at the call, on every profile that traces rays; 2 and 32 translate. A value the canonical passes
/// fold to a null `i8` (`freeze i8 undef`, an `extractvalue` of `zeroinitializer`) reads as 0, as
/// `air::intrinsics::sig::Immediates` reads an immediate: a constant outside the range, not a
/// computed one.
#[test]
fn max_levels_outside_metals_range_is_invalid_input() {
    for (tag, max_levels, prelude, kind) in [
        (
            "levels-computed",
            "%m",
            "",
            IntrinsicError::ImmediateNotConstant,
        ),
        ("levels-zero", "0", "", IntrinsicError::SignatureMismatch),
        (
            "levels-freeze-undef",
            "%fz",
            "  %fz = freeze i8 undef\n",
            IntrinsicError::SignatureMismatch,
        ),
        (
            "levels-extracted-zero",
            "%ez",
            "  %ez = extractvalue { i8, i8 } zeroinitializer, 0\n",
            IntrinsicError::SignatureMismatch,
        ),
        ("levels-one", "1", "", IntrinsicError::SignatureMismatch),
        ("levels-33", "33", "", IntrinsicError::SignatureMismatch),
        (
            "levels-minus-one",
            "-1",
            "",
            IntrinsicError::SignatureMismatch,
        ),
    ] {
        let plain = multi_level_kernel(max_levels, "instance");
        let text = plain.replacen("  %r = call", &format!("{prelude}  %r = call"), 1);
        let Some(bc) = assemble(tag, &text) else {
            return;
        };
        // The prelude folds to a constant before lowering, and the load and truncation of a
        // computed count go where the call does not read them, so the call's index is the plain
        // one's without them.
        let unread = "  %mw = load i32, ptr addrspace(1) %0, align 4\n  %m = trunc i32 %mw to i8\n";
        assert!(plain.contains(unread));
        let located = if max_levels == "%m" {
            plain.clone()
        } else {
            plain.replace(unread, "")
        };
        let call = position(&located, "%r = call");
        for (stem, p) in committed_profiles().expect("committed profiles") {
            if !traces(&p) {
                continue;
            }
            match compile(&bc, "k", &p) {
                Err(Error::InvalidInput(e)) => {
                    assert_eq!(
                        e.kind,
                        InputErrorKind::Intrinsic(kind.clone()),
                        "{tag} {stem}"
                    );
                    assert_eq!(
                        e.at,
                        Location::function(DiagName::new("k"), None, Some(call.1)),
                        "{tag} {stem}"
                    );
                }
                other => panic!("{tag} on {stem}: {other:?}"),
            }
        }
    }
    for (tag, max_levels) in [("levels-two", "2"), ("levels-32", "32")] {
        let Some(bc) = assemble(tag, &multi_level_kernel(max_levels, "instance")) else {
            return;
        };
        let translated = per_profile(&bc, "k", 1);
        assert!(
            translated.iter().any(|s| s == "lavapipe"),
            "{tag}: {translated:?}"
        );
    }
}

/// The null table read other than as an intersector's table (compared, converted, or stored) is
/// an intersection function table refusal at the null table.
#[test]
fn a_null_table_read_otherwise_is_refused() {
    let result = "{ i32, float, i32, i32, ptr addrspace(1) }";
    let callee = "air.intersect";
    let args =
        format!("{RAY}, ptr addrspace(1) %1, ptr addrspace(1) %t, ptr null, i64 0, {SETTINGS}");
    let operands = format!(
        "<3 x float>, <3 x float>, float, float, ptr addrspace(1), ptr addrspace(1), ptr, i64, \
         {SETTINGS_TYPES}"
    );
    let body = call_body(result, callee, &args);
    for (tag, read) in [
        (
            "table-compared",
            "  %q = icmp eq ptr addrspace(1) %t, null\n  %z = zext i1 %q to i32\n  store i32 %z, \
             ptr addrspace(1) %0, align 4\n",
        ),
        (
            "table-converted",
            "  %q = ptrtoint ptr addrspace(1) %t to i64\n  %z = trunc i64 %q to i32\n  store i32 \
             %z, ptr addrspace(1) %0, align 4\n",
        ),
        (
            "table-stored",
            "  store ptr addrspace(1) %t, ptr addrspace(1) %0, align 8\n",
        ),
    ] {
        let text = kernel_with(
            ", ptr addrspace(1) %1",
            &[structure(1, "primitive")],
            &body.replace("  ret void\n", &format!("{read}  ret void\n")),
            &call_decls(result, callee, &operands),
        );
        let Some(bc) = assemble(tag, &text) else {
            return;
        };
        let u = refusal(&bc, "k");
        assert_eq!(u.feature, Feature::IntersectionFunctionTable, "{tag}: {u}");
        assert_eq!(u.needs, Needs::NotImplemented, "{tag}: {u}");
        assert_eq!(
            u.location,
            Some(Location::function(DiagName::new("k"), Some(0), Some(0))),
            "{tag}: {u}"
        );
    }
}

/// A structure that is not one entry argument is ray tracing not implemented yet: an array of
/// structures (refused at its argument), a structure chosen at run time (at the choice, as the
/// handle's family), and a constant structure (at the call, naming it).
#[test]
fn structure_arrays_and_computed_structures_are_refused() {
    let result = "{ i32, float, i32, i32, ptr addrspace(1) }";
    let callee = "air.intersect";
    let operands = format!(
        "<3 x float>, <3 x float>, float, float, ptr addrspace(1), ptr addrspace(1), ptr, i64, \
         {SETTINGS_TYPES}"
    );
    let array = structure(1, "primitive")
        .replace("i32 1, i32 1, !\"air.read\"", "i32 1, i32 2, !\"air.read\"")
        .replace(
            "acceleration_structure<>",
            "array<acceleration_structure<>, 2>",
        );
    let body = call_body(
        result,
        callee,
        &format!("{RAY}, ptr addrspace(1) %s, ptr addrspace(1) %t, ptr null, i64 0, {SETTINGS}"),
    )
    .replace(
        "  %t = call",
        "  %e = getelementptr ptr addrspace(1), ptr %1, i64 1\n  %s = load ptr addrspace(1), ptr \
         %e, align 8\n  %t = call",
    );
    let text = kernel_with(
        ", ptr %1",
        &[array],
        &body,
        &call_decls(result, callee, &operands),
    );
    let Some(bc) = assemble("array", &text) else {
        return;
    };
    let u = refusal(&bc, "k");
    assert_eq!(u.feature, Feature::RayTracing, "{u}");
    assert_eq!(u.needs, Needs::NotImplemented, "{u}");
    assert_eq!(
        u.location,
        Some(Location::metadata(Some(DiagName::new("k")), Some(1), None)),
        "{u}"
    );

    let body = call_body(
        result,
        callee,
        &format!("{RAY}, ptr addrspace(1) %s, ptr addrspace(1) %t, ptr null, i64 0, {SETTINGS}"),
    )
    .replace(
        "  %t = call",
        "  %c = load i32, ptr addrspace(1) %0, align 4\n  %b = icmp ne i32 %c, 0\n  %s = select \
         i1 %b, ptr addrspace(1) %1, ptr addrspace(1) %2\n  %t = call",
    );
    let text = kernel_with(
        ", ptr addrspace(1) %1, ptr addrspace(1) %2",
        &[structure(1, "primitive"), structure(2, "primitive")],
        &body,
        &call_decls(result, callee, &operands),
    );
    let Some(bc) = assemble("computed", &text) else {
        return;
    };
    let u = refusal(&bc, "k");
    assert_eq!(u.feature, Feature::RayTracing, "{u}");
    assert_eq!(u.needs, Needs::NotImplemented, "{u}");
    assert_eq!(
        u.location,
        Some(Location::function(DiagName::new("k"), Some(0), Some(2))),
        "{u}"
    );

    let args =
        format!("{RAY}, ptr addrspace(1) null, ptr addrspace(1) %t, ptr null, i64 0, {SETTINGS}");
    let text = kernel_with(
        "",
        &[],
        &call_body(result, callee, &args),
        &call_decls(result, callee, &operands),
    );
    assert_ray_refusal("constant", &text, Feature::RayTracing, Some(callee));
}

/// Filling an intersection function table's buffer slot (the decoded
/// `air.set_buffer_intersection_function_table`) is an intersection function table refusal,
/// naming the call.
#[test]
fn the_table_setter_is_an_intersection_function_table() {
    let callee = "air.set_buffer_intersection_function_table.p1i8";
    let text = kernel_with(
        ", ptr addrspace(1) %1",
        &[buffer(1, "air.read", "source")],
        &format!(
            "  call void @{callee}(ptr addrspace(1) undef, ptr addrspace(1) %1, i32 0)\n  store \
             i32 0, ptr addrspace(1) %0, align 4\n  ret void\n"
        ),
        &format!("declare void @{callee}(ptr addrspace(1), ptr addrspace(1), i32)"),
    );
    let Some(bc) = assemble("setter", &text) else {
        return;
    };
    for (stem, p) in committed_profiles().expect("committed profiles") {
        let u = match compile(&bc, "k", &p) {
            Err(Error::Unsupported(u)) => u,
            Err(other) => panic!("{stem}: {other}"),
            Ok(_) => panic!("{stem}: compiled"),
        };
        assert_eq!(u.feature, Feature::IntersectionFunctionTable, "{stem}: {u}");
        assert_eq!(u.needs, Needs::NotImplemented, "{stem}: {u}");
        assert_eq!(u.symbol, Some(DiagName::new(callee)), "{stem}: {u}");
    }
}

/// `k(device uint *out [[buffer(0)]], instance_acceleration_structure scene [[buffer(1)]])`:
/// `out[0] = scene.get_instance_count()`; with `intersect`, also one ray through the null table
/// whose hit type it stores at `out[1]`.
fn count_kernel(intersect: bool) -> String {
    let (ray, decls) = if intersect {
        (
            format!(
                "  %t = call ptr addrspace(1) @air.get_null_intersection_function_table()\n  %r = \
                 call {INSTANCED} @air.intersect.instancing.triangle_data.world_space_data(<3 x \
                 float> <float 0.0, float 0.0, float -1.0>, <3 x float> <float 0.0, float 0.0, \
                 float 1.0>, float 0.0, float 100.0, ptr addrspace(1) %1, i32 255, ptr \
                 addrspace(1) %t, ptr null, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 3, i32 \
                 -1, i32 -1, i32 0, i1 false, i1 false)\n  %ty = extractvalue {INSTANCED} %r, 0\n  \
                 %p1 = getelementptr i32, ptr addrspace(1) %0, i64 1\n  store i32 %ty, ptr \
                 addrspace(1) %p1, align 4\n"
            ),
            format!(
                "declare ptr addrspace(1) @air.get_null_intersection_function_table()\ndeclare \
                 {INSTANCED} @air.intersect.instancing.triangle_data.world_space_data(<3 x float>, \
                 <3 x float>, float, float, ptr addrspace(1), i32, ptr addrspace(1), ptr, i64, \
                 i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1)\n"
            ),
        )
    } else {
        (String::new(), String::new())
    };
    format!(
        "define void @k(ptr addrspace(1) %0, ptr addrspace(1) %1) {{\n  %n = call i32 \
         @air.get_instance_count_instance_acceleration_structure(ptr addrspace(1) %1)\n  store \
         i32 %n, ptr addrspace(1) %0, align 4\n{ray}  ret void\n}}\n\ndeclare i32 \
         @air.get_instance_count_instance_acceleration_structure(ptr addrspace(1))\n{decls}\n\
         !air.kernel = !{{!0}}\n!0 = !{{ptr @k, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4}}\n!3 = \
         {}\n!4 = {}\n",
        buffer(0, "air.read_write", "out"),
        structure(1, "instance"),
    )
}

/// The buffer slot of the structure the count kernels read.
fn count_slot() -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index: 1,
    }
}

/// An instance count of an entry argument's instance structure reads the binder's push field: no
/// Vulkan instruction reads a structure's instance count, and the consumer knows it. The kernel
/// translates and validates on every committed profile, ray queries or not, with no
/// acceleration-structure descriptor or ray query capability; it states the fact and its push
/// field, and records no precision note (the count is exact).
#[test]
fn an_instance_count_is_the_binders_on_every_profile() {
    let Some(bc) = assemble("count", &count_kernel(false)) else {
        return;
    };
    let slot = count_slot();
    let mut profiles = 0;
    for (stem, p) in committed_profiles().expect("committed profiles") {
        let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            a.facts
                .contains(&RuntimeFact::AccelerationStructureInstanceCount { slot }),
            "{stem}: {:?}",
            a.facts
        );
        assert!(
            a.interface
                .push
                .fields
                .iter()
                .any(|f| f.field == PushField::InstanceCount(slot)),
            "{stem}: {:?}",
            a.interface.push
        );
        assert!(a.precision.is_empty(), "{stem}: {:?}", a.precision);
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let dis = spirv_dis(&words);
        for needle in ["RayQueryKHR", "AccelerationStructureKHR"] {
            assert!(!dis.contains(needle), "{stem}: {needle} present");
        }
        profiles += 1;
    }
    assert!(profiles > 1);
}

/// A structure argument the function both intersects and counts is a descriptor where the device
/// traces rays, with its count still the binder's push field; elsewhere the argument is refused
/// as `RayTracing`, naming the missing features, as any intersected structure is.
#[test]
fn an_intersected_structure_keeps_its_descriptor_beside_its_count() {
    let Some(bc) = assemble("count-intersect", &count_kernel(true)) else {
        return;
    };
    let translated = per_profile(&bc, "k", 1);
    assert!(translated.iter().any(|s| s == "lavapipe"), "{translated:?}");
    let p =
        crate::pipeline::testing::profile("lavapipe", DevicePolicy::default()).expect("lavapipe");
    let a = compile(&bc, "k", &p).expect("lavapipe");
    assert!(a
        .facts
        .contains(&RuntimeFact::AccelerationStructureInstanceCount { slot: count_slot() }));
}

/// The structure's other queries (an instance's primitive structure, the instance descriptors)
/// are not lowered: where the device traces rays they are refused `NotImplemented(RayTracing)`,
/// naming the call. A structure read by one is no count-only structure, so elsewhere its argument
/// is the device's refusal.
#[test]
fn a_structures_other_queries_stay_refused() {
    let callee = "air.get_primitive_acceleration_structure_instance_acceleration_structure";
    let text = format!(
        "define void @k(ptr addrspace(1) %0, ptr addrspace(1) %1) {{\n  %n = call i32 \
         @air.get_instance_count_instance_acceleration_structure(ptr addrspace(1) %1)\n  store \
         i32 %n, ptr addrspace(1) %0, align 4\n  %c = call ptr addrspace(1) @{callee}(ptr \
         addrspace(1) %1, i32 0)\n  ret void\n}}\n\ndeclare i32 \
         @air.get_instance_count_instance_acceleration_structure(ptr addrspace(1))\ndeclare ptr \
         addrspace(1) @{callee}(ptr addrspace(1), i32)\n\n!air.kernel = !{{!0}}\n!0 = !{{ptr @k, \
         !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4}}\n!3 = {}\n!4 = {}\n",
        buffer(0, "air.read_write", "out"),
        structure(1, "instance"),
    );
    let Some(bc) = assemble("other-query", &text) else {
        return;
    };
    let u = refusal(&bc, "k");
    assert_eq!(u.feature, Feature::RayTracing, "{u}");
    assert_eq!(u.needs, Needs::NotImplemented, "{u}");
    assert_eq!(u.symbol, Some(DiagName::new(callee)), "{u}");
    for (stem, p) in committed_profiles().expect("committed profiles") {
        if traces(&p) {
            continue;
        }
        match compile(&bc, "k", &p) {
            Err(Error::Unsupported(u)) => assert!(
                matches!(u.needs, Needs::AnyOf(_)) && u.feature == Feature::RayTracing,
                "{stem}: {u}"
            ),
            Err(other) => panic!("{stem}: {other}"),
            Ok(_) => panic!("{stem}: compiled"),
        }
    }
}

/// The owned fixture's bitcode (`validation/fixtures/public/engine/<stem>.ll`).
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

/// The profiles the count kernels run on: two without ray queries, MoltenVK's, and one with them.
const COUNT_PROFILES: [&str; 4] = [
    "lunarg-desktop-baseline-2023",
    "vk12-floor",
    "moltenvk-m3",
    "lavapipe",
];

/// `entry` of `bc` end to end on `stem`: compiled through the device, planned for a stage table
/// that binds a 16-byte buffer at index 0 and each of `counts` (buffer index, instance count) as
/// an acceleration structure, instantiated for one group of `threads` threads, and run on the
/// interpreter over the instantiated push bytes, with buffer 0 starting as `initial`. Returns
/// buffer 0's final bytes. The plan writes no acceleration-structure descriptor.
fn run_counts(
    bc: &[u8],
    entry: &str,
    stem: &str,
    counts: &[(u32, u32)],
    threads: u32,
    initial: [u8; 16],
) -> Vec<u8> {
    let what = format!("{entry} on {stem} with {counts:?}");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles");
    let text = std::fs::read_to_string(path.join(format!("{stem}.json"))).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let dev = Device::new(desc, DevicePolicy::default()).expect("device");
    let lib = Library::from_air(bc).expect("library");
    let f = lib.function(entry).expect("entry");
    let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
    let pipe = dev
        .compile_compute_pipeline(&d, &Budget::standard())
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let p = crate::pipeline::testing::profile(stem, DevicePolicy::default()).expect("profile");
    let (_, m) = crate::pipeline::testing::typed_module(bc, entry, &p)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let mut table = StageTable::default();
    table.buffers.insert(
        0,
        BoundBuffer::Buffer {
            length: 16,
            offset: 0,
            gpu_address: 0,
            device_address: None,
            attribute_stride: None,
        },
    );
    for &(index, n) in counts {
        table
            .acceleration_structures
            .insert(index, BoundAccelerationStructure::new(n));
    }
    let mut state = BindState::default();
    state.stages.insert(Stage::Kernel, table);
    let plan = pipe.plan(&dev, &state.shape()).expect("planned");
    assert!(
        !plan
            .descriptors
            .iter()
            .any(|w| matches!(w.source, WriteSource::AccelerationStructure { .. })),
        "{what}: {:?}",
        plan.descriptors
    );
    let cmds = plan
        .instantiate_dispatch(
            &state,
            &Dispatch::Threadgroups {
                groups: [1, 1, 1],
                threads_per_group: [threads, 1, 1],
            },
        )
        .expect("instantiated");
    let region = &cmds.regions[0];
    let mut push = Vec::new();
    for w in &region.push_constants {
        let end = w.offset as usize + w.bytes.len();
        if push.len() < end {
            push.resize(end, 0);
        }
        push[w.offset as usize..end].copy_from_slice(&w.bytes);
    }
    let mut spec = std::collections::BTreeMap::new();
    for (axis, n) in [Axis::X, Axis::Y, Axis::Z]
        .into_iter()
        .zip([u64::from(threads), 1, 1])
    {
        spec.insert(SpecKey::LocalSize(axis), n);
    }
    spec.insert(SpecKey::FlatLocalSize, u64::from(threads));
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: initial.to_vec(),
        }],
        push,
        spec,
    };
    let cfg = Config {
        workgroups: [1, 1, 1],
        ..Config::default()
    };
    let mut out = run(&m, &cfg, res).unwrap_or_else(|e| panic!("{what}: {e:?}"));
    out.buffers.swap_remove(0)
}

/// `rt_instance_count` end to end on a device without ray queries and on one with them: the
/// binder plans no descriptor for the structure and pushes the count
/// `StageTable::acceleration_structures` states for its slot, 0 for an unbound slot, and running
/// the module over the instantiated push bytes writes `count * 3 + gid`.
#[test]
fn the_binder_pushes_the_bound_structures_instance_count() {
    let Some(bc) = fixture("rt_instance_count") else {
        return;
    };
    for stem in COUNT_PROFILES {
        for count in [Some(5u32), Some(0), None] {
            let bound: Vec<(u32, u32)> = count.map(|n| (1, n)).into_iter().collect();
            let out = run_counts(&bc, "rt_instance_count", stem, &bound, 4, [0xAB; 16]);
            let n = count.unwrap_or(0);
            let want: Vec<u8> = (0..4u32).flat_map(|g| (n * 3 + g).to_le_bytes()).collect();
            assert_eq!(out, want, "{stem}: {count:?}");
        }
    }
}

/// The declaration of `get_instance_count()`.
const COUNT_DECL: &str =
    "declare i32 @air.get_instance_count_instance_acceleration_structure(ptr addrspace(1))";

/// `%name = structure.get_instance_count()`.
fn count_of(name: &str, structure: &str) -> String {
    format!(
        "  %{name} = call i32 @air.get_instance_count_instance_acceleration_structure(ptr \
         addrspace(1) {structure})\n"
    )
}

/// `k` of `bc` on every committed profile: translated and valid with neither a ray query nor an
/// acceleration structure in the module, stating the count's runtime fact and push field for
/// exactly the structures at `slots`.
fn assert_counts_everywhere(tag: &str, bc: &[u8], slots: &[u32]) {
    let want: Vec<SlotRef> = slots
        .iter()
        .map(|&index| SlotRef {
            stage: Stage::Kernel,
            class: SlotClass::Buffer,
            index,
        })
        .collect();
    for (stem, p) in committed_profiles().expect("committed profiles") {
        let a = compile(bc, "k", &p).unwrap_or_else(|e| panic!("{tag} on {stem}: {e}"));
        let mut facts: Vec<SlotRef> = a
            .facts
            .iter()
            .filter_map(|f| match f {
                RuntimeFact::AccelerationStructureInstanceCount { slot } => Some(*slot),
                _ => None,
            })
            .collect();
        facts.sort();
        assert_eq!(facts, want, "{tag} on {stem}: {:?}", a.facts);
        let mut fields: Vec<SlotRef> = a
            .interface
            .push
            .fields
            .iter()
            .filter_map(|f| match &f.field {
                PushField::InstanceCount(slot) => Some(*slot),
                _ => None,
            })
            .collect();
        fields.sort();
        assert_eq!(fields, want, "{tag} on {stem}: {:?}", a.interface.push);
        let (_, words) =
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{tag} on {stem}: {e}"));
        let dis = spirv_dis(&words);
        for needle in ["RayQueryKHR", "AccelerationStructureKHR"] {
            assert!(!dis.contains(needle), "{tag} on {stem}: {needle} present");
        }
    }
}

/// Two structures, each counted: each count is its own slot's push field, so the binder's two
/// counts reach their own stores, and an unbound one reads 0.
#[test]
fn several_structures_read_their_own_counts() {
    let body = format!(
        "{}{}  store i32 %a, ptr addrspace(1) %0, align 4\n  %p1 = getelementptr i32, ptr \
         addrspace(1) %0, i64 1\n  store i32 %b, ptr addrspace(1) %p1, align 4\n  ret void\n",
        count_of("a", "%1"),
        count_of("b", "%2"),
    );
    let text = kernel_with(
        ", ptr addrspace(1) %1, ptr addrspace(1) %2",
        &[structure(1, "instance"), structure(2, "instance")],
        &body,
        COUNT_DECL,
    );
    let Some(bc) = assemble("two-counts", &text) else {
        return;
    };
    assert_counts_everywhere("two-counts", &bc, &[1, 2]);
    for stem in COUNT_PROFILES {
        for (bound, a, b) in [
            (vec![(1, 7), (2, 9)], 7u32, 9u32),
            (vec![(2, 9)], 0, 9),
            (vec![(1, 7)], 7, 0),
        ] {
            let out = run_counts(&bc, "k", stem, &bound, 1, [0xAB; 16]);
            let mut want = [0xAB; 16];
            want[..4].copy_from_slice(&a.to_le_bytes());
            want[4..8].copy_from_slice(&b.to_le_bytes());
            assert_eq!(out, want, "{stem}: {bound:?}");
        }
    }
}

/// A count read in one arm of a branch and merged by a phi is the binder's count where that arm
/// runs, and the other arm's value where it does not.
#[test]
fn a_count_under_control_flow_is_the_binders() {
    let body = format!(
        "  %v = load i32, ptr addrspace(1) %0, align 4\n  %c = icmp eq i32 %v, 0\n  br i1 %c, \
         label %then, label %join\nthen:\n{}  br label %join\njoin:\n  %r = phi i32 [ %n, %then \
         ], [ 99, %2 ]\n  store i32 %r, ptr addrspace(1) %0, align 4\n  ret void\n",
        count_of("n", "%1"),
    );
    let text = kernel_with(
        ", ptr addrspace(1) %1",
        &[structure(1, "instance")],
        &body,
        COUNT_DECL,
    );
    let Some(bc) = assemble("branch-count", &text) else {
        return;
    };
    assert_counts_everywhere("branch-count", &bc, &[1]);
    for stem in COUNT_PROFILES {
        for (first, want) in [(0u32, 5u32), (1, 99)] {
            let mut initial = [0xAB; 16];
            initial[..4].copy_from_slice(&first.to_le_bytes());
            let out = run_counts(&bc, "k", stem, &[(1, 5)], 1, initial);
            let mut expected = initial;
            expected[..4].copy_from_slice(&want.to_le_bytes());
            assert_eq!(out, expected, "{stem}: out[0] = {first}");
        }
    }
}

/// A count of a structure that is not an entry argument has no slot the binder knows, so it is
/// refused, typed, on every committed profile, never translated: through a `select` or a `phi`
/// of two arguments (where the device traces rays, the merge itself; elsewhere the arguments,
/// which then need their descriptors), through a `getelementptr` (a handle used as a pointer),
/// loaded from memory, and of a primitive structure (the count is an instance structure's).
#[test]
fn a_count_of_anything_but_an_instance_argument_is_refused() {
    let store = "  store i32 %n, ptr addrspace(1) %0, align 4\n  ret void\n";
    let two = [structure(1, "instance"), structure(2, "instance")];
    let two_params = ", ptr addrspace(1) %1, ptr addrspace(1) %2";
    let pick = "  %v = load i32, ptr addrspace(1) %0, align 4\n  %c = icmp eq i32 %v, 0\n";
    let call = "air.get_instance_count_instance_acceleration_structure";
    // (tag, kernel, where the device traces rays, elsewhere); `None` elsewhere: the argument's
    // ray-tracing refusal, naming the features the profile lacks.
    type Expect = (Feature, Needs);
    let cases: Vec<(&str, String, Expect, Option<Expect>)> = vec![
        (
            "select",
            kernel_with(
                two_params,
                &two,
                &format!(
                    "{pick}  %s = select i1 %c, ptr addrspace(1) %1, ptr addrspace(1) %2\n{}{store}",
                    count_of("n", "%s")
                ),
                COUNT_DECL,
            ),
            (Feature::RayTracing, Needs::NotImplemented),
            None,
        ),
        (
            "phi",
            kernel_with(
                two_params,
                &two,
                &format!(
                    "{pick}  br i1 %c, label %a, label %b\na:\n  br label %j\nb:\n  br label \
                     %j\nj:\n  %s = phi ptr addrspace(1) [ %1, %a ], [ %2, %b ]\n{}{store}",
                    count_of("n", "%s")
                ),
                COUNT_DECL,
            ),
            (
                Feature::Lowering(LowerFamily::Access),
                Needs::NotImplemented,
            ),
            None,
        ),
        (
            "gep",
            kernel_with(
                ", ptr addrspace(1) %1",
                &[structure(1, "instance")],
                &format!(
                    "  %s = getelementptr i8, ptr addrspace(1) %1, i64 0\n{}{store}",
                    count_of("n", "%s")
                ),
                COUNT_DECL,
            ),
            (Feature::HandleAsPointer, Needs::NotRepresentable),
            Some((Feature::HandleAsPointer, Needs::NotRepresentable)),
        ),
        (
            "load",
            kernel_with(
                "",
                &[],
                &format!(
                    "  %s = load ptr addrspace(1), ptr addrspace(1) %0, align 8\n{}{store}",
                    count_of("n", "%s")
                ),
                COUNT_DECL,
            ),
            (Feature::RayTracing, Needs::NotImplemented),
            Some((Feature::RayTracing, Needs::NotImplemented)),
        ),
        (
            "primitive",
            kernel_with(
                ", ptr addrspace(1) %1",
                &[structure(1, "primitive")],
                &format!("{}{store}", count_of("n", "%1")),
                COUNT_DECL,
            ),
            (Feature::RayTracing, Needs::NotImplemented),
            Some((Feature::RayTracing, Needs::NotImplemented)),
        ),
    ];
    for (tag, text, tracing, elsewhere) in cases {
        let Some(bc) = assemble(tag, &text) else {
            return;
        };
        for (stem, p) in committed_profiles().expect("committed profiles") {
            let u = match compile(&bc, "k", &p) {
                Err(Error::Unsupported(u)) => u,
                Err(other) => panic!("{tag} on {stem}: {other}"),
                Ok(_) => panic!("{tag} on {stem}: compiled"),
            };
            let want = if traces(&p) {
                Some(tracing.clone())
            } else {
                elsewhere.clone()
            };
            match want {
                Some((feature, needs)) => {
                    assert_eq!(
                        (u.feature.clone(), u.needs.clone()),
                        (feature, needs),
                        "{tag} on {stem}"
                    );
                }
                None => {
                    let missing: Vec<Capability> = RAY_FEATURES
                        .into_iter()
                        .filter(|f| !p.fields().enabled.features.contains(f))
                        .map(Capability::Feature)
                        .collect();
                    assert_eq!(u.feature, Feature::RayTracing, "{tag} on {stem}: {u}");
                    assert_eq!(u.needs, Needs::AnyOf(vec![missing]), "{tag} on {stem}: {u}");
                    assert_eq!(
                        u.location,
                        Some(Location::metadata(Some(DiagName::new("k")), Some(1), None)),
                        "{tag} on {stem}: {u}"
                    );
                }
            }
            if tag == "primitive" {
                assert_eq!(u.symbol, Some(DiagName::new(call)), "{tag} on {stem}: {u}");
            }
        }
    }
}
