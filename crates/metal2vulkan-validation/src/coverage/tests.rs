//! The census of compiled modules and executed runs: what a compile records of each module
//! (its strategies, subgroup shape, and containment) and what a run's census holds.

use super::tier_table::*;
use super::*;
use crate::case::out_of_bounds::{ContainmentForm, OutOfBoundsResource, OutOfBoundsTarget};
use crate::case::AuthoredCase;
use std::collections::BTreeMap;

const MOLTENVK: HostClass = HostClass(vk::DriverId::MOLTENVK);

const FREE: SubgroupShape = SubgroupShape {
    required_size: None,
    full_subgroups: false,
};

fn pinned(size: u32, full_subgroups: bool) -> SubgroupShape {
    SubgroupShape {
        required_size: Some(size),
        full_subgroups,
    }
}

/// The containment of a module with buffers and no image, compiled with robust buffers.
const ROBUST: Containment = Containment {
    buffers: BufferEncoding::Robust,
    images: ImageEncoding::Unused,
};

/// A module compiled under [`ROBUST`].
fn module(stage: Stage, subgroup: SubgroupShape, strategies: &[&str]) -> ModuleCensus {
    ModuleCensus {
        stage,
        subgroup,
        strategies: strategies.iter().map(|name| name.to_string()).collect(),
        precision: BTreeSet::new(),
        containment: Some(ROBUST),
    }
}

#[test]
fn executed_census_round_trips() {
    let census = ExecutedCensus {
        out_of_bounds: None,
        rate_map_pass: None,
        device: Some(ExecutionClass::Host(MOLTENVK)),
        product: "p".into(),
        restricted_to: None,
        leg: Some(Leg::Stated),
        modules: vec![module(Stage::Kernel, pinned(32, true), &["simd.reduce"])],
        artifact: None,
    };
    let json = serde_json::to_value(&census).unwrap();
    assert_eq!(json["device"], serde_json::json!({"host": 14}));
    assert_eq!(
        json["modules"][0]["containment"],
        serde_json::json!({"buffers": "robust", "images": "unused"})
    );
    assert_eq!(
        serde_json::from_value::<ExecutedCensus>(json).unwrap(),
        census
    );
}

/// A rate-mapped pass drawn region by region records its region count and leaves the comparison
/// class the modules'; a row recorded before regions reads without one, and an unknown field is
/// refused.
#[test]
fn a_rate_mapped_pass_records_its_regions() {
    let mut census = ExecutedCensus {
        out_of_bounds: None,
        rate_map_pass: Some(RateMapPassRun {
            screen_size: [37, 23],
            compiled_physical_size: [36, 23],
            regions: 2,
            precision: BTreeSet::new(),
        }),
        device: Some(ExecutionClass::Host(MOLTENVK)),
        product: "p".into(),
        restricted_to: None,
        leg: Some(Leg::Stated),
        modules: vec![module(Stage::Fragment, FREE, &["arith.float"])],
        artifact: None,
    };
    assert_eq!(census.comparison_class(), ComparisonClass::Exact);
    let json = serde_json::to_value(&census).unwrap();
    assert_eq!(
        json["rate_map_pass"],
        serde_json::json!({
            "screen_size": [37, 23],
            "compiled_physical_size": [36, 23],
            "regions": 2
        })
    );
    assert_eq!(
        serde_json::from_value::<ExecutedCensus>(json).unwrap(),
        census
    );
    // The pass's near-equivalent is the row's, recorded beside the regions.
    census.rate_map_pass = Some(RateMapPassRun {
        screen_size: [37, 23],
        compiled_physical_size: [36, 23],
        regions: 2,
        precision: [PrecisionNote::RasterizationRateMap].into(),
    });
    assert_eq!(
        census.comparison_class(),
        ComparisonClass::derive([PrecisionNote::RasterizationRateMap], ["arith.float"])
    );
    let json = serde_json::to_value(&census).unwrap();
    assert_eq!(
        json["rate_map_pass"]["precision"],
        serde_json::json!(["RasterizationRateMap"])
    );
    assert_eq!(
        serde_json::from_value::<ExecutedCensus>(json).unwrap(),
        census
    );
    // A pass over the identity records no note.
    census.rate_map_pass = Some(RateMapPassRun {
        screen_size: [32, 16],
        compiled_physical_size: [32, 16],
        regions: 1,
        precision: BTreeSet::new(),
    });
    let json = serde_json::to_value(&census).unwrap();
    assert_eq!(
        json["rate_map_pass"],
        serde_json::json!({
            "screen_size": [32, 16],
            "compiled_physical_size": [32, 16],
            "regions": 1,
        })
    );
    assert_eq!(
        serde_json::from_value::<ExecutedCensus>(json).unwrap(),
        census
    );
    // Every run states its regions; a field no version wrote is refused.
    let row = serde_json::json!({
        "screen_size": [37, 23],
        "compiled_physical_size": [36, 23],
    });
    assert!(serde_json::from_value::<RateMapPassRun>(row.clone()).is_err());
    for unknown in ["rate", "note"] {
        let mut row = row.clone();
        row["regions"] = serde_json::json!(2);
        assert!(serde_json::from_value::<RateMapPassRun>(row.clone()).is_ok());
        row[unknown] = serde_json::json!("x");
        assert!(serde_json::from_value::<RateMapPassRun>(row).is_err());
    }
}

/// A row of executor v34 states the module-level `device_addresses`, since superseded: it still
/// reads, as the encodings alone (such a row is not current, so it is evidence for nothing), and a
/// field no version wrote is still refused.
#[test]
fn a_v34_containment_reads_without_its_device_addresses() {
    let read = |json| serde_json::from_value::<Containment>(json);
    assert_eq!(
        read(serde_json::json!({
            "buffers": "checked",
            "device_addresses": true,
            "images": "unused"
        }))
        .unwrap(),
        Containment {
            buffers: BufferEncoding::Checked,
            images: ImageEncoding::Unused,
        }
    );
    assert!(read(serde_json::json!({
        "buffers": "checked",
        "images": "unused",
        "device-addresses": true
    }))
    .is_err());
}

/// A tessellation pipeline's census: its evaluation stage is the post-tessellation
/// vertex function's module, as the per-function census counts it, and the synthesized vertex
/// and control stages, which run no Metal function, are left out.
#[test]
fn a_tessellation_pipeline_census_is_its_functions() {
    use metal2vulkan::device::{Device, DevicePolicy};
    use metal2vulkan::library::Library;
    use metal2vulkan::pipeline::{FunctionDescriptor, RenderPipelineDescriptor, StageSource};
    use metal2vulkan::Budget;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};
    let bitcode = match llvm_assemble(&engine_dir().join("tessellation.ll")) {
        Ok(bytes) => bytes,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected tessellation.ll: {why}"),
    };
    let path = crate::engine::profiles_dir().join("moltenvk-m3.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let device = Device::new(
        DeviceDescription::from_khronos_json(&text, None).unwrap(),
        DevicePolicy::default(),
    )
    .unwrap();
    let library = Library::from_air(&bitcode).unwrap();
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        library.function("tess_quad_buffer").unwrap(),
    )));
    d.rasterization_enabled = false;
    let pipeline = device
        .compile_render_pipeline(&d, &Budget::standard())
        .unwrap();
    let stages: Vec<vk::ShaderStage> = pipeline.stages.iter().map(|m| m.stage).collect();
    assert_eq!(
        stages,
        [
            vk::ShaderStage::VERTEX,
            vk::ShaderStage::TESSELLATION_CONTROL,
            vk::ShaderStage::TESSELLATION_EVALUATION
        ]
    );
    let profile = device.profile();
    let census = ModuleCensus::of_pipeline_modules(&pipeline.stages, profile).unwrap();
    assert_eq!(
        census,
        [ModuleCensus::of_stage_module_under(&pipeline.stages[2], profile, None).unwrap()]
    );
    assert_eq!(census[0].stage, Stage::Vertex);
    assert!(census[0].containment.is_some());
    assert_eq!(
        ModuleCensus::of_stage_module_under(&pipeline.stages[1], profile, None),
        None
    );
    // Every other pipeline's modules are each counted.
    assert_eq!(
        ModuleCensus::of_pipeline_modules(&pipeline.stages[..1], profile).unwrap(),
        [ModuleCensus::of_stage_module_under(&pipeline.stages[0], profile, None).unwrap()]
    );
    let error = ModuleCensus::of_pipeline_modules(&pipeline.stages[..2], profile).unwrap_err();
    assert!(error.contains("TESSELLATION_CONTROL"), "{error}");
}

/// A render pipeline's census leaves out its geometry stages, which run no Metal function: a
/// vertex function whose slice and viewport differ among a primitive's vertices
/// (`engine/rt6_first_vertex`) stores them through the synthesized geometry stage of each topology
/// class on a device whose driver does not take Metal's first-vertex indices (the Radeon 780M's
/// committed capture), so the pipeline's modules are its vertex function's, three geometry stages
/// and its fragment function's, and its census is the two functions'.
#[test]
fn a_render_pipeline_census_leaves_out_its_geometry_stages() {
    use metal2vulkan::device::{Device, DevicePolicy};
    use metal2vulkan::library::Library;
    use metal2vulkan::metal::PixelFormat;
    use metal2vulkan::pipeline::{
        ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
    };
    use metal2vulkan::Budget;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};
    let bitcode = match llvm_assemble(&engine_dir().join("rt6_first_vertex.ll")) {
        Ok(bytes) => bytes,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected rt6_first_vertex.ll: {why}"),
    };
    let path = crate::engine::profiles_dir().join("amd-780m-radv.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let device = Device::new(
        DeviceDescription::from_khronos_json(&text, None).unwrap(),
        DevicePolicy::default(),
    )
    .unwrap();
    let library = Library::from_air(&bitcode).unwrap();
    let stage = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(library.function(name).unwrap()))
    };
    let mut d = RenderPipelineDescriptor::new(stage("rt6_first_vertex"));
    d.fragment = Some(stage("rt6_first_vertex_read"));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
    let pipeline = device
        .compile_render_pipeline(&d, &Budget::standard())
        .unwrap();
    let stages: Vec<vk::ShaderStage> = pipeline.stages.iter().map(|m| m.stage).collect();
    assert_eq!(
        stages,
        [
            vk::ShaderStage::VERTEX,
            vk::ShaderStage::GEOMETRY,
            vk::ShaderStage::GEOMETRY,
            vk::ShaderStage::GEOMETRY,
            vk::ShaderStage::FRAGMENT
        ]
    );
    let profile = device.profile();
    let census = ModuleCensus::of_pipeline_modules(&pipeline.stages, profile).unwrap();
    assert_eq!(
        census,
        [
            ModuleCensus::of_stage_module_under(&pipeline.stages[0], profile, None).unwrap(),
            ModuleCensus::of_stage_module_under(&pipeline.stages[4], profile, None).unwrap()
        ]
    );
    assert_eq!(
        census.iter().map(|m| m.stage).collect::<Vec<_>>(),
        [Stage::Vertex, Stage::Fragment]
    );
    for geometry in &pipeline.stages[1..4] {
        assert_eq!(
            ModuleCensus::of_stage_module_under(geometry, profile, None),
            None
        );
    }
}

/// The tier table holds every committed profile as stated and, where it states a robustness
/// feature, in its robustness-off leg (validation runs both): the stated leg compiles the robust
/// encodings the profile states, the off leg the in-shader ones, on the same device class and
/// subgroup width. Every committed profile states `robustBufferAccess`, so every one has both legs;
/// `vk12-floor` states no image robustness, and `moltenvk-m3` states
/// `robustBufferAccess` without `robustBufferAccess2`.
#[test]
fn the_tier_table_derives_a_robustness_off_leg_from_each_profile() {
    use metal2vulkan::device::Robustness;
    let table = committed_profiles().unwrap();
    let stems = crate::engine::committed_profiles().unwrap();
    assert_eq!(table.len(), 2 * stems.len());
    let robust = |profile: &DeviceProfile| {
        (
            profile.checks_buffers_in_shader(),
            profile.images().robust_image_access,
        )
    };
    for stem in &stems {
        let description = crate::engine::load_description(stem).unwrap();
        let legs = table
            .iter()
            .filter(|entry| entry.tier.stem == *stem)
            .collect::<Vec<_>>();
        let [stated, off] = legs[..] else {
            panic!("{stem}: {} legs", legs.len());
        };
        assert_eq!(stated.tier.leg, Leg::Stated);
        assert_eq!(stated.tier.label(), *stem);
        assert_eq!(off.tier.leg, Leg::RobustnessOff);
        assert_eq!(off.tier.label(), format!("{stem}+robustness-off"));
        assert_eq!(
            (off.tier.class, off.tier.subgroup_size),
            (stated.tier.class, stated.tier.subgroup_size)
        );
        let images = description.has(VkFeature::RobustImageAccess)
            || description.has(VkFeature::RobustImageAccess2);
        assert_eq!(robust(&stated.profile), (false, images), "{stem}");
        assert_eq!(robust(&off.profile), (true, false), "{stem}");
    }
    let stated = |stem: &str| {
        table
            .iter()
            .find(|entry| entry.tier.stem == stem && entry.tier.leg == Leg::Stated)
            .unwrap()
    };
    assert!(!stated("vk12-floor").profile.images().robust_image_access);
    assert!(stated("lavapipe").profile.images().robust_image_access);
    assert_eq!(
        stated("moltenvk-m3").profile.containment(),
        ProfileContainment::On {
            robust: Robustness::Rba
        }
    );
    assert_eq!(
        stated("lavapipe").profile.containment(),
        ProfileContainment::On {
            robust: Robustness::Rba2
        }
    );
    // A description that states no robustness feature has no second leg.
    let mut none = crate::engine::load_description("vk12-floor").unwrap();
    none.features
        .retain(|feature| !ROBUSTNESS_FEATURES.contains(feature));
    assert_eq!(robustness_off(&none), None);
}

/// A compile records the containment it compiled under, from its profile and its own module:
/// the robust and the in-shader leg of one copy census the same strategy names (the name hides
/// the encoding) under different encodings; a kernel that declares no image has none; a texture
/// write's image encoding follows the leg's image robustness, and `vk12-floor`, which writes a
/// storage image only through its declared format, needs the texture's format first; and a kernel that turns an integer into a pointer (a dynamic root) reaches that memory by
/// device address under names of its own, checked in the shader in both legs, while its descriptor
/// accesses take each leg's encoding.
#[test]
fn a_compile_records_the_containment_it_compiled_under() {
    use metal2vulkan_devtools::{engine_dir, fixtures_dir, llvm_assemble};
    let table = committed_profiles().unwrap();
    let leg = |stem: &str, leg: Leg| {
        &table
            .iter()
            .find(|entry| entry.tier.stem == stem && entry.tier.leg == leg)
            .unwrap()
            .profile
    };
    let compile = |path: std::path::PathBuf, entry: &str, profile: &DeviceProfile| {
        let bitcode = llvm_assemble(&path).unwrap_or_else(|error| panic!("{error:?}"));
        compile_module(&bitcode, entry, &CaseInputs::defaults(), profile)
            .unwrap_or_else(|error| panic!("{}: {error:?}", path.display()))
            .census
    };
    let encoding = |buffers, images| Some(Containment { buffers, images });
    use BufferEncoding as B;
    use ImageEncoding as I;

    let copy = |leg_of| {
        compile(
            engine_dir().join("kernel_copy_runtime_ulong_length.ll"),
            "kernel_copy_runtime_ulong_length",
            leg("lavapipe", leg_of),
        )
    };
    let (stated, off) = (copy(Leg::Stated), copy(Leg::RobustnessOff));
    assert!(stated.strategies.contains("access.memcpy.dynamic"));
    assert!(off.strategies.contains("access.memcpy.dynamic"));
    assert_eq!(stated.containment, encoding(B::Robust, I::Unused));
    assert_eq!(off.containment, encoding(B::Checked, I::Unused));

    let write = |stem: &str, leg_of| {
        compile(
            engine_dir().join("a4_select_write.ll"),
            "a4_select_write",
            leg(stem, leg_of),
        )
        .containment
        .unwrap()
        .images
    };
    assert_eq!(write("lavapipe", Leg::Stated), I::Robust);
    assert_eq!(write("lavapipe", Leg::RobustnessOff), I::Checked);
    let select_write = llvm_assemble(&engine_dir().join("a4_select_write.ll"))
        .unwrap_or_else(|error| panic!("{error:?}"));
    match compile_module(
        &select_write,
        "a4_select_write",
        &CaseInputs::defaults(),
        leg("vk12-floor", Leg::Stated),
    ) {
        Err(CompileError::Refused(reason)) => assert!(
            reason.starts_with("needs_input StorageTextureFormat"),
            "{reason}"
        ),
        other => panic!("a4_select_write on vk12-floor: {other:?}"),
    }

    let round_trip = |leg_of| {
        let module = compile(
            fixtures_dir().join("kernel_buffer_address_round_trip.ll"),
            "buffer_address_round_trip",
            leg("lavapipe", leg_of),
        );
        let accesses = module
            .strategies
            .iter()
            .filter(|name| {
                ["access.load", "access.store", "access.dynamic.load"]
                    .iter()
                    .any(|base| name.starts_with(base))
            })
            .cloned()
            .collect::<Vec<_>>();
        (module.containment, accesses)
    };
    assert_eq!(
        round_trip(Leg::Stated),
        (
            encoding(B::Robust, I::Unused),
            ["access.dynamic.load", "access.load", "access.store"]
                .map(String::from)
                .into()
        )
    );
    assert_eq!(
        round_trip(Leg::RobustnessOff),
        (
            encoding(B::Checked, I::Unused),
            [
                "access.dynamic.load",
                "access.load.checked",
                "access.store.checked"
            ]
            .map(String::from)
            .into()
        )
    );
}

/// A kernel that reads, through a `select` of two textures by thread parity (a switch-at-use),
/// a `texture_buffer<float, read>` when `texel`, else a `texture2d<float, sample>` at level 0,
/// and stores the texel to `[[buffer(0)]]`, as AIR text.
fn switched_read_kernel(texel: bool) -> String {
    let (ty, access, read, declaration) = if texel {
        (
            "texture_buffer<float, read>",
            "air.read",
            "call { <4 x float>, i8 } @air.read_texture_buffer_1d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, i32 %gid, i32 1)",
            "declare { <4 x float>, i8 } @air.read_texture_buffer_1d.v4f32(ptr addrspace(1), ptr addrspace(2), i32, i32)",
        )
    } else {
        (
            "texture2d<float, sample>",
            "air.sample",
            "call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %xy, <2 x i32> zeroinitializer, i32 0, i32 0)",
            "declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)",
        )
    };
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @switched(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b, i32 %gid) {{
entry:
  %odd = and i32 %gid, 1
  %c = icmp eq i32 %odd, 0
  %t = select i1 %c, ptr addrspace(1) %a, ptr addrspace(1) %b
  %xy = insertelement <2 x i32> zeroinitializer, i32 %gid, i64 0
  %s = call ptr addrspace(2) @air.get_read_sampler()
  %r = {read}
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %gid
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}}

declare ptr addrspace(2) @air.get_read_sampler()
{declaration}

!air.kernel = !{{!0}}
!0 = !{{ptr @switched, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"{access}", !"air.arg_type_name", !"{ty}", !"air.arg_name", !"a"}}
!5 = !{{i32 2, !"air.texture", !"air.location_index", i32 1, i32 1, !"{access}", !"air.arg_type_name", !"{ty}", !"air.arg_name", !"b"}}
!6 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
"#
    )
}

/// An image module's encoding names what its device is left, as the product's texture
/// family decides it (`DeviceProfile::device_reads_texels`, `device_drops_texel_writes`):
/// `robust` under `robustImageAccess2` (reads and writes), `robust_writes` under
/// `robustImageAccess` alone (writes; its reads are clamped in the shader, `read_clamped`), and
/// `checked` without image robustness.
#[test]
fn an_image_encoding_names_what_its_device_is_left() {
    use metal2vulkan_devtools::llvm_assemble;
    let table = committed_profiles().unwrap();
    let scratch = crate::ScratchDir::new("coverage-image-encoding").unwrap();
    let ll = scratch.path().join("image.ll");
    std::fs::write(&ll, switched_read_kernel(false)).unwrap();
    let bitcode = llvm_assemble(&ll).unwrap_or_else(|error| panic!("{error:?}"));
    for (stem, leg, images, read) in [
        (
            "lunarg-desktop-baseline-2023",
            Leg::Stated,
            ImageEncoding::Robust,
            "texture.read.switch",
        ),
        (
            "swiftshader",
            Leg::Stated,
            ImageEncoding::RobustWrites,
            "texture.read_clamped.switch",
        ),
        (
            "swiftshader",
            Leg::RobustnessOff,
            ImageEncoding::Checked,
            "texture.read_clamped.switch",
        ),
    ] {
        let entry = table
            .iter()
            .find(|entry| entry.tier.stem == stem && entry.tier.leg == leg)
            .unwrap_or_else(|| panic!("{stem} {leg:?}"));
        let census = compile_module(
            &bitcode,
            "switched",
            &CaseInputs::defaults(),
            &entry.profile,
        )
        .unwrap_or_else(|error| panic!("{stem}: {error:?}"))
        .census;
        let containment = census.containment.expect("a compiled module's containment");
        assert_eq!(containment.images, images, "{stem} {leg:?}");
        assert!(
            census.strategies.contains(read),
            "{stem} {leg:?}: {:?}",
            census.strategies
        );
    }
    assert_eq!(
        Containment {
            buffers: BufferEncoding::Robust,
            images: ImageEncoding::RobustWrites,
        }
        .label(),
        "buffers:robust,images:robust_writes"
    );
}

/// A vertex case's row compiles under the vertex descriptor its executors draw it with.
/// `vertex_varying_triangle` reads two `float2` attributes and binds no buffer. Under its
/// descriptor the stated leg fetches them fixed function and declares no buffer, while the
/// robustness-off leg, whose buffer robustness no longer bounds fixed-function input, pulls them
/// from a storage buffer checked in the shader: the row the executed module belongs to. A fixture
/// entry's default inputs (no descriptor) declare no buffer on either leg.
#[test]
fn a_vertex_case_compiles_under_its_vertex_descriptor() {
    use crate::case::{AttributeFormat, AttributeInput};
    use metal2vulkan_devtools::{fixtures_dir, llvm_assemble};
    let table = committed_profiles().unwrap();
    let leg = |stem: &str, leg: Leg| {
        &table
            .iter()
            .find(|entry| entry.tier.stem == stem && entry.tier.leg == leg)
            .unwrap()
            .profile
    };
    let bitcode = llvm_assemble(&fixtures_dir().join("vertex_varying_triangle.ll"))
        .unwrap_or_else(|error| panic!("{error:?}"));
    let mut case =
        crate::case::narrow_vertex_test_case("aa".repeat(32), "vertex_varying_triangle".into());
    case.buffers.clear();
    case.vertex_inputs = [0, 1]
        .map(|location| AttributeInput {
            location,
            format: AttributeFormat::Float2,
            stride: 8,
            dynamic_stride: false,
            // Three `float2` elements.
            bytes_b64: "AACAvwAAgL8AAEBAAACAvwAAgL8AAEBA".into(),
        })
        .to_vec();
    let inputs = case_inputs(&case, &bitcode).unwrap();
    let descriptor = inputs
        .vertex_descriptor
        .as_ref()
        .expect("a vertex case's descriptor");
    assert_eq!(descriptor.attributes.len(), 2);
    let buffers = |inputs: &CaseInputs, profile: &DeviceProfile| {
        compile_module(&bitcode, "vertex_varying_triangle", inputs, profile)
            .unwrap_or_else(|error| panic!("{error:?}"))
            .census
            .containment
            .unwrap()
            .buffers
    };
    for stem in ["lavapipe", "moltenvk-m3", "vk12-floor"] {
        assert_eq!(
            buffers(&inputs, leg(stem, Leg::Stated)),
            BufferEncoding::Unused,
            "{stem}"
        );
        assert_eq!(
            buffers(&inputs, leg(stem, Leg::RobustnessOff)),
            BufferEncoding::Checked,
            "{stem}"
        );
        for leg_of in [Leg::Stated, Leg::RobustnessOff] {
            assert_eq!(
                buffers(&CaseInputs::defaults(), leg(stem, leg_of)),
                BufferEncoding::Unused,
                "{stem} {leg_of:?}"
            );
        }
    }
    // A fragment case feeds no vertex buffer, so it compiles without a descriptor.
    let kernel = crate::case::attachmentless_fragment_test_case("bb".repeat(32), "f".into());
    assert_eq!(
        case_inputs(&kernel, b"not decoded")
            .unwrap()
            .vertex_descriptor,
        None
    );
}

/// The resources a module declares are read from its instructions: a buffer variable
/// (`StorageBuffer` or `Uniform`), a texel buffer (a buffer, which buffer robustness contains),
/// an image type of any other `Dim`; words that do not parse declare everything, so no encoding
/// is merged into another. A module whose only image type is a texel buffer's is therefore
/// labelled `images:unused` on every profile, even where `robustImageAccess2` would label an
/// image `robust`.
#[test]
fn a_module_declares_what_its_instructions_declare() {
    const HEADER: [u32; 5] = [0x0723_0203, 0x0001_0500, 0, 100, 0];
    let module = |instructions: &[&[u32]]| {
        HEADER
            .iter()
            .copied()
            .chain(instructions.iter().flat_map(|i| i.iter().copied()))
            .collect::<Vec<_>>()
    };
    let op = |opcode: u32, operands: &[u32]| {
        std::iter::once(((operands.len() as u32 + 1) << 16) | opcode)
            .chain(operands.iter().copied())
            .collect::<Vec<_>>()
    };
    let declared = |buffers, images| Declared { buffers, images };
    // OpTypeImage %1 %2 Dim2D ...; OpTypeImage %3 %2 DimBuffer ...; OpVariable %4 %5 StorageBuffer;
    // OpVariable %6 %7 Uniform; OpVariable %8 %9 Function; OpFunctionEnd.
    let image_2d = op(25, &[1, 2, 1, 0, 0, 0, 2, 0]);
    let texel_buffer = op(25, &[3, 2, 5, 0, 0, 0, 2, 0]);
    let storage = op(59, &[4, 5, 12]);
    let uniform = op(59, &[6, 7, 2]);
    let function = op(59, &[8, 9, 7]);
    let end = op(56, &[]);
    assert_eq!(
        Declared::of(&module(&[&function, &end])),
        declared(false, false)
    );
    assert_eq!(
        Declared::of(&module(&[&image_2d, &end])),
        declared(false, true)
    );
    assert_eq!(
        Declared::of(&module(&[&texel_buffer])),
        declared(true, false)
    );
    assert_eq!(
        Declared::of(&module(&[&texel_buffer, &image_2d, &end])),
        declared(true, true)
    );
    let mut robust_images = 0;
    for entry in committed_profiles().unwrap() {
        robust_images += usize::from(entry.profile.device_reads_texels(false));
        let containment = Containment::of_compile(&entry.profile, &module(&[&texel_buffer, &end]));
        assert_eq!(
            containment.images,
            ImageEncoding::Unused,
            "{}",
            entry.tier.label()
        );
    }
    assert!(robust_images > 0);
    assert_eq!(
        Declared::of(&module(&[&storage, &end])),
        declared(true, false)
    );
    assert_eq!(Declared::of(&module(&[&uniform])), declared(true, false));
    // A word count of zero, an instruction running past the end, or no header.
    assert_eq!(Declared::of(&module(&[&[0]])), declared(true, true));
    assert_eq!(
        Declared::of(&module(&[&storage[..2]])),
        declared(true, true)
    );
    assert_eq!(Declared::of(&[]), declared(true, true));
}

/// A target of `form` at `resource`, with no reach (the report reads only the form).
fn target(resource: OutOfBoundsResource, form: ContainmentForm) -> OutOfBoundsTarget {
    serde_json::from_value(serde_json::json!({ "resource": resource, "form": form })).unwrap()
}

/// The out-of-bounds fixtures (`engine/ob1_*`) compile each target they declare to the form
/// and mechanism of the inventory on lavapipe, MoltenVK, and SwiftShader, in both legs: a
/// descriptor view left to the device's buffer robustness where the leg enables it and checked in
/// the shader where it does not, a device address and the arena checked in the shader on every
/// leg, and an image or texel buffer left to the device's image or buffer robustness where
/// enabled. The form is the compiled structure's: without device addresses the escaping buffer is
/// a storage buffer, and a robustness-off leg views the `constant` buffer as a storage buffer.
#[test]
fn the_out_of_bounds_fixtures_contain_each_target_as_declared() {
    use metal2vulkan_devtools::{engine_dir, llvm_assemble};
    use ContainmentForm as F;
    use OutOfBoundsResource as R;
    let table = committed_profiles().unwrap();
    let fixtures: [(&str, Vec<OutOfBoundsTarget>); 7] = [
        (
            "ob1_storage_buffer",
            vec![
                target(R::Buffer { binding: 1 }, F::StorageBuffer),
                target(R::Buffer { binding: 2 }, F::StorageBuffer),
            ],
        ),
        (
            "ob1_uniform_buffer",
            vec![target(R::Buffer { binding: 1 }, F::UniformBuffer)],
        ),
        (
            "ob1_buffer_atomic",
            vec![target(R::Buffer { binding: 1 }, F::BufferAtomic)],
        ),
        (
            "ob1_device_address",
            vec![target(R::Buffer { binding: 1 }, F::DeviceAddress)],
        ),
        (
            "ob1_threadgroup",
            vec![target(R::ThreadgroupMemory { binding: 0 }, F::Threadgroup)],
        ),
        (
            "ob1_texture",
            vec![
                target(R::Texture { binding: 0 }, F::Texture),
                target(R::Texture { binding: 1 }, F::Texture),
            ],
        ),
        (
            "ob1_texel_buffer",
            vec![
                target(R::Texture { binding: 0 }, F::TexelBuffer),
                target(R::Texture { binding: 1 }, F::TexelBuffer),
            ],
        ),
    ];
    // The mechanism label of each fixture's targets per (stem, leg). SwiftShader states only the
    // first robustness levels and no device addresses: the buffer whose address escapes is a
    // descriptor view there. Its robustness leaves buffer atomics uncontained, so the product
    // checks them in the shader in both legs.
    let expected = |fixture: &str, stem: &str, leg: Leg| -> &str {
        let off = leg == Leg::RobustnessOff;
        let robust1 = ["moltenvk-m3", "swiftshader"].contains(&stem);
        match fixture {
            "ob1_device_address" if stem == "swiftshader" && off => "descriptor-view:shader",
            "ob1_device_address" if stem == "swiftshader" => "descriptor-view:robust",
            "ob1_device_address" => "device-address:shader",
            "ob1_threadgroup" => "arena:shader",
            "ob1_buffer_atomic" if stem == "swiftshader" => "descriptor-view:shader",
            _ if off => match fixture {
                "ob1_texture" => "image:shader",
                "ob1_texel_buffer" => "texel-buffer:shader",
                _ => "descriptor-view:shader",
            },
            // `robustImageAccess` alone discards writes; reads are clamped in the shader.
            "ob1_texture" if stem == "swiftshader" => "image:shader-reads",
            "ob1_texture" => "image:robust2",
            // `robustBufferAccess` alone: the texel buffer is checked in the shader.
            "ob1_texel_buffer" if robust1 => "texel-buffer:shader",
            "ob1_texel_buffer" => "texel-buffer:robust2",
            _ if robust1 => "descriptor-view:robust",
            _ => "descriptor-view:robust2",
        }
    };
    // The form each target is exercised as: the declared one, but a buffer as its compiled slot
    // reaches it. A device address that compiles to a descriptor view is a storage buffer, and so
    // is the `constant` buffer every robustness-off leg views as a storage buffer, checked in the
    // shader: a uniform view is only ever left to the device's robustness.
    let form = |fixture: &str, stem: &str, leg: Leg, declared: ContainmentForm| match fixture {
        "ob1_device_address" if stem == "swiftshader" => F::StorageBuffer,
        "ob1_uniform_buffer" if leg == Leg::RobustnessOff => F::StorageBuffer,
        _ => declared,
    };
    for (fixture, targets) in fixtures {
        let path = engine_dir().join(format!("{fixture}.ll"));
        let bitcode = llvm_assemble(&path).unwrap_or_else(|error| panic!("{error:?}"));
        let inputs = CaseInputs {
            out_of_bounds: targets.clone(),
            ..CaseInputs::defaults()
        };
        for entry in table.iter().filter(|entry| {
            ["lavapipe", "moltenvk-m3", "swiftshader"].contains(&entry.tier.stem.as_str())
        }) {
            let compiled = compile_module(&bitcode, fixture, &inputs, &entry.profile)
                .unwrap_or_else(|error| panic!("{fixture} on {}: {error:?}", entry.tier.label()));
            let want = expected(fixture, &entry.tier.stem, entry.tier.leg);
            assert_eq!(compiled.out_of_bounds.len(), targets.len());
            for (contained, declared) in compiled.out_of_bounds.iter().zip(&targets) {
                assert_eq!(
                    contained.form,
                    form(fixture, &entry.tier.stem, entry.tier.leg, declared.form),
                    "{fixture} on {}",
                    entry.tier.label()
                );
                assert_eq!(
                    contained.mechanism.label(),
                    want,
                    "{fixture} on {}",
                    entry.tier.label()
                );
            }
        }
    }
}

/// The fixtures of the authorable forms (`engine/ob2_*`) compile each target to its form and an
/// in-shader check on every committed profile that compiles them, in both legs: the buffer a loaded
/// pointer reaches is a device address the resident set resolves, the thread-local array a typed
/// object checked per array level, and the argument buffer's texture array a handle table. The rest
/// refuse with what they lack, a typed refusal in both legs: the profiles without device addresses
/// or 64-bit integers the loaded pointer, and those without descriptor indexing the argument
/// buffer's textures.
#[test]
fn the_ob2_fixtures_contain_each_target_in_the_shader_on_every_profile() {
    use metal2vulkan_devtools::{engine_dir, llvm_assemble};
    use ContainmentForm as F;
    use OutOfBoundsResource as R;
    let table = committed_profiles().unwrap();
    let fixtures = [
        (
            "ob2_loaded_pointer",
            target(
                R::ArgumentBufferBuffer {
                    buffer_binding: 1,
                    field_offset: 0,
                },
                F::LoadedPointer,
            ),
            "device-address:shader",
        ),
        (
            "ob2_sized_array",
            target(R::ThreadMemory, F::SizedArray),
            "typed-object:shader",
        ),
        (
            "ob2_handle_array",
            target(
                R::ArgumentBufferTextureArray {
                    buffer_binding: 1,
                    field_offset: 0,
                },
                F::HandleArrayIndex,
            ),
            "handle-table:shader",
        ),
    ];
    let mut refused = BTreeSet::new();
    for (fixture, declared, mechanism) in fixtures {
        let path = engine_dir().join(format!("{fixture}.ll"));
        let bitcode = llvm_assemble(&path).unwrap_or_else(|error| panic!("{error:?}"));
        let inputs = CaseInputs {
            out_of_bounds: vec![declared.clone()],
            ..CaseInputs::defaults()
        };
        for entry in table.iter() {
            let label = entry.tier.label();
            match compile_module(&bitcode, fixture, &inputs, &entry.profile) {
                Ok(compiled) => {
                    let [contained] = compiled.out_of_bounds.as_slice() else {
                        panic!("{fixture} on {label}: {:?}", compiled.out_of_bounds);
                    };
                    assert_eq!(contained.form, declared.form, "{fixture} on {label}");
                    assert_eq!(
                        contained.mechanism.label(),
                        mechanism,
                        "{fixture} on {label}"
                    );
                }
                Err(CompileError::Refused(reason)) => {
                    let lacked = match fixture {
                        "ob2_loaded_pointer" => "DereferenceLoadedPointer",
                        _ => "ArgumentBufferTextures",
                    };
                    assert!(
                        reason.starts_with(&format!("device_capability_missing {lacked} ")),
                        "{fixture} on {label}: {reason}"
                    );
                    refused.insert((fixture, entry.tier.stem.as_str()));
                }
                Err(error) => panic!("{fixture} on {label}: {error:?}"),
            }
        }
    }
    let pointer = [
        "lunarg-desktop-baseline-2023",
        "roadmap-2022",
        "swiftshader",
        "vk12-floor",
    ];
    let handles = ["vk12-floor"];
    assert_eq!(
        refused,
        pointer
            .map(|stem| ("ob2_loaded_pointer", stem))
            .into_iter()
            .chain(handles.map(|stem| ("ob2_handle_array", stem)))
            .collect::<BTreeSet<_>>()
    );
}

/// The case of an OB3 fixture: its committed manifest's literals, as the coverage report reads them
/// (`case_inputs`), with the target's reach and no read declared (the compile reads neither).
fn ob3_case(fixture: &str) -> AuthoredCase {
    use base64::Engine as _;
    use serde_json::json;
    let b64 = |words: &[u32]| {
        base64::engine::general_purpose::STANDARD.encode(
            words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    };
    let attribute = |location: u32, stride: u32, words: &[u32]| json!({"location": location, "format": "uint", "stride": stride, "bytes_b64": b64(words)});
    let target = |resource: serde_json::Value, form: &str| json!({"resource": resource, "form": form, "reach_after": 16});
    let output = |binding: u32, words: usize| json!({"binding": binding, "role": "output", "initial_bytes_b64": b64(&vec![0; words])});
    let attribute_target = |location: u32, form: &str| {
        target(
            json!({"kind": "attribute_buffer", "location": location}),
            form,
        )
    };
    let mut case = json!({
        "air_sha256": "00".repeat(32),
        "case_id": "11".repeat(32),
        "name": fixture,
        "entry": fixture,
        "compare": {"kind": "exact"},
        "execution_safety": "loop_free",
        "output": {"kind": "buffer", "binding": 0, "offset": 0, "length": 4},
    });
    let fields = match fixture {
        "ob3_vertex_input" => json!({
            "stage": "vertex",
            "buffers": [output(1, 8)],
            "vertex_inputs": [attribute(0, 4, &[1, 2, 3, 4])],
            "draw": {"primitive": "point", "vertex_start": 0, "vertex_count": 8,
                     "instance_count": 1},
            "out_of_bounds": {"targets": [attribute_target(0, "vertex_input")]},
        }),
        "ob3_control_point" => json!({
            "stage": "vertex",
            "buffers": [output(2, 16)],
            "tessellation": {
                "factors": vec![json!({"edge_f16": [15360, 15360, 15360], "inside_f16": [15360]}); 4],
                "instance_count": 1,
                "amplification_count": 1,
                "control_points": [attribute(0, 4, &[0; 8])],
                "patch_inputs": [attribute(1, 8, &[0; 4])],
            },
            "out_of_bounds": {"targets": [
                attribute_target(0, "control_point"),
                attribute_target(1, "control_point"),
            ]},
        }),
        "ob3_index_buffer" => json!({
            "stage": "vertex",
            "buffers": [output(0, 8)],
            "draw": {"primitive": "point", "vertex_start": 0, "vertex_count": 16,
                     "instance_count": 1,
                     "index_buffer": {"index_type": "uint32", "offset": 0,
                                      "bytes_b64": b64(&[0, 1, 2, 3])}},
            "out_of_bounds": {"targets": [{"resource": {"kind": "index_buffer"},
                                           "form": "index_buffer", "reach_after": 48}]},
        }),
        "ob3_stage_input" => json!({
            "stage": "kernel",
            "buffers": [output(0, 8)],
            "kernel_stage_inputs": [attribute(0, 4, &[1, 2, 3, 4])],
            "dispatch": {"grid": [8, 1, 1], "threads_per_threadgroup": [8, 1, 1]},
            "out_of_bounds": {"targets": [attribute_target(0, "stage_input")]},
        }),
        "ob3_texture_array" => {
            let texel = |word: u32| json!({"dimensions": [1, 1, 1], "bytes_b64": b64(&[word])});
            let array = |binding: u32| {
                json!({"binding": binding, "role": "input", "texture_type": "d2",
                       "format": "r32_uint", "elements": [texel(1), texel(2)]})
            };
            json!({
                "stage": "kernel",
                "buffers": [
                    {"binding": 0, "role": "input", "bytes_b64": b64(&[0, 1, 2, 3, 1, 0, 3, 2])},
                    output(1, 16),
                ],
                "texture_arrays": [array(0), array(2)],
                "dispatch": {"grid": [8, 1, 1], "threads_per_threadgroup": [8, 1, 1]},
                "out_of_bounds": {"targets": [{"resource": {"kind": "texture_array", "binding": 0},
                                               "form": "handle_array_index",
                                               "reach_after": 2}]},
            })
        }
        other => panic!("no OB3 fixture {other}"),
    };
    for (key, value) in fields.as_object().unwrap() {
        case[key] = value.clone();
    }
    serde_json::from_value(case).unwrap()
}

/// The fixtures of the authorable forms (`engine/ob3_*`) compile, under the inputs their cases
/// imply (a vertex descriptor, a stage-input descriptor), each target to its declared form and the
/// mechanism of its fetch on every committed profile, in both legs: a vertex input is
/// fixed-function under the stated buffer robustness and pulled with the check without it; a
/// control point and a stage input are pulled through a descriptor view, left to the stated
/// robustness or checked in the shader; an index buffer is bounded by `robustBufferAccess2` where
/// the leg enables it and clamped by the binder elsewhere; a top-level texture array's index is
/// clamped in the shader. The profiles without vertex stores refuse the vertex and control-point
/// draws, which write memory, a typed refusal; a control point compiles on every other profile,
/// through the tessellation emulation on one without `tessellationShader`.
#[test]
fn the_ob3_fixtures_contain_each_target_by_its_fetch_on_every_profile() {
    use metal2vulkan_devtools::{engine_dir, llvm_assemble};
    let table = committed_profiles().unwrap();
    // What each profile's description states, by stem: whether it has `robustBufferAccess2` and
    // vertex stores. The stated description decides both legs; the robustness-off leg reads none
    // of the robustness here.
    let described = table
        .iter()
        .map(|entry| entry.tier.stem.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|stem| {
            let description = crate::engine::load_description(stem).unwrap();
            let has = |feature| description.has(feature);
            let facts = (
                has(VkFeature::RobustBufferAccess2),
                has(VkFeature::VertexPipelineStoresAndAtomics),
            );
            (stem, facts)
        })
        .collect::<BTreeMap<_, _>>();
    let mut compiled_rows = 0;
    for fixture in [
        "ob3_vertex_input",
        "ob3_control_point",
        "ob3_index_buffer",
        "ob3_stage_input",
        "ob3_texture_array",
    ] {
        let path = engine_dir().join(format!("{fixture}.ll"));
        let bitcode = llvm_assemble(&path).unwrap_or_else(|error| panic!("{error:?}"));
        let case = ob3_case(fixture);
        let inputs = case_inputs(&case, &bitcode).unwrap();
        for entry in table.iter() {
            let label = entry.tier.label();
            let stem = entry.tier.stem.as_str();
            let off = entry.tier.leg == Leg::RobustnessOff;
            let (robust2, vertex_stores) = described[stem];
            let robust = if robust2 { "robust2" } else { "robust" };
            let refused = match fixture {
                "ob3_vertex_input" | "ob3_index_buffer" | "ob3_control_point" => {
                    (!vertex_stores).then_some("StageStoresAndAtomics")
                }
                _ => None,
            };
            let expected = match fixture {
                "ob3_vertex_input" if off => "descriptor-view:shader".to_string(),
                "ob3_vertex_input" => format!("vertex-input:{robust}"),
                "ob3_control_point" | "ob3_stage_input" if off => "descriptor-view:shader".into(),
                "ob3_control_point" | "ob3_stage_input" => format!("descriptor-view:{robust}"),
                // Every device and leg draws from a zero-padded copy.
                "ob3_index_buffer" => "index-buffer:padded".into(),
                _ => "handle-array:shader".into(),
            };
            match compile_module(&bitcode, fixture, &inputs, &entry.profile) {
                Ok(compiled) => {
                    assert_eq!(refused, None, "{fixture} on {label}");
                    let declared = &case.out_of_bounds.as_ref().unwrap().targets;
                    assert_eq!(compiled.out_of_bounds.len(), declared.len());
                    for (contained, target) in compiled.out_of_bounds.iter().zip(declared) {
                        assert_eq!(contained.form, target.form, "{fixture} on {label}");
                        assert_eq!(
                            contained.mechanism.label(),
                            expected,
                            "{fixture} on {label}"
                        );
                    }
                    compiled_rows += 1;
                }
                Err(CompileError::Refused(reason)) => {
                    let lacked =
                        refused.unwrap_or_else(|| panic!("{fixture} on {label}: {reason}"));
                    assert!(
                        reason.starts_with(&format!("device_capability_missing {lacked} ")),
                        "{fixture} on {label}: {reason}"
                    );
                }
                Err(error) => panic!("{fixture} on {label}: {error:?}"),
            }
        }
    }
    assert!(compiled_rows > 0);
}
