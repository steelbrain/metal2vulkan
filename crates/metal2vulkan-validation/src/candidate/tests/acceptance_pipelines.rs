//! L4 driver acceptance on a device: the unsafe creation path of [`create_acceptance_pipeline`]
//! (set layouts, binding flags, specialization, subgroup size, render pass, graphics pipeline) runs
//! against the lane's Vulkan device for a render row, and kernel rows are made through the runtime
//! on one runtime device ([`RuntimeKernels`]). Nothing is bound or submitted, so no shader
//! executes.
//! `G-linux`'s `m2v_cpu_icd` step runs it on lavapipe.
//!
//! Each render row compiles for the opened device's own `Device` (its queried description
//! restricted to what the context enabled, `platform::opened_worker_device`), never for a committed
//! profile named after the lane, so a lane whose device differs from any committed description
//! still checks what it would create.

use super::*;
use crate::acceptance::{
    compile_row_in_worker, retained, Acceptance, RowCompile, RowPipeline, RowStage, RuntimeRow,
};
use metal2vulkan::library::FunctionConstantValues;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};

/// Compile `entry` of the owned fixture `file` as an L4 row for the device the
/// lane opens, in the device tests' compile worker ([`compile_worker`]), as `corpus-pipeline`
/// compiles its rows. A device lane has `llvm-as`, so a missing assembler fails the test rather
/// than skipping it.
fn compiled(file: &str, entry: &str, stage: RowStage) -> RowPipeline {
    let bitcode = llvm_assemble(&engine_dir().join(file))
        .unwrap_or_else(|error| panic!("assemble {file}: {error:?}"));
    let device = platform::opened_worker_device(crate::gpu::TEST_DEVICE)
        .unwrap_or_else(|error| panic!("open the lane's device: {error:?}"));
    match compile_row_in_worker(
        compile_worker(),
        &device,
        &retained(&bitcode),
        entry,
        stage,
        &metal2vulkan::library::FunctionConstantValues::defaults(),
    ) {
        (RowCompile::Compiled(pipeline), _) => *pipeline,
        (other, _) => panic!("{file} {entry} did not compile for the opened device: {other:?}"),
    }
}

/// The driver created the render pipeline within the ceiling, or the device lacks something it
/// needs; a refusal or a tooling error (including an unavailable device) is a failure.
fn assert_accepted(pipeline: &RowPipeline) {
    let RowPipeline::Render { pipeline, .. } = pipeline else {
        panic!("a kernel row is made through the runtime");
    };
    accepted(create_acceptance_pipeline(
        crate::gpu::TEST_DEVICE,
        pipeline,
    ));
}

#[cfg(target_os = "linux")]
/// What the runtime answered for each kernel row `(file, entry)` of the owned fixtures, made one
/// after another on one runtime device of the lane, as `corpus-pipeline` makes its kernel rows,
/// with no function-constant values; the device finished with nothing logged at its teardown. A
/// device lane has `llvm-as`, so a missing assembler fails the test.
fn made(rows: &[(&str, &str)]) -> Vec<Acceptance> {
    let mut kernels = RuntimeKernels::open(crate::gpu::TEST_DEVICE)
        .unwrap_or_else(|error| panic!("open the lane's runtime device: {error}"));
    let constants = FunctionConstantValues::defaults();
    let made = rows
        .iter()
        .map(|&(file, entry)| {
            let bitcode = llvm_assemble(&engine_dir().join(file))
                .unwrap_or_else(|error| panic!("assemble {file}: {error:?}"));
            let row = RuntimeRow {
                bitcode: &bitcode,
                entry,
                constants: &constants,
            };
            let made = kernels
                .accept(&row)
                .unwrap_or_else(|error| panic!("{entry}: {error}"));
            eprintln!("{entry}: {made:?}");
            made
        })
        .collect();
    kernels
        .finish()
        .unwrap_or_else(|error| panic!("finish the lane's runtime device: {error}"));
    made
}

/// `acceptance` is a pipeline created within the ceiling, or a device that lacks something it
/// needs; anything else fails.
fn accepted(acceptance: Result<Acceptance, crate::acceptance::AcceptanceError>) {
    match acceptance {
        Ok(Acceptance::Created { create }) => {
            assert!(
                crate::budgets::creation_within(
                    "the pipeline",
                    create,
                    crate::budgets::PIPELINE_CREATION_CEILING
                )
                .is_ok(),
                "{create:?}"
            );
        }
        Ok(Acceptance::NotEnabled(missing)) => eprintln!("NOT_ENABLED: {}", missing.join(", ")),
        Ok(Acceptance::Refused(why)) => panic!("the driver refused the pipeline: {why}"),
        Ok(Acceptance::InvalidUsage(messages)) => {
            panic!("the validation layer reported: {}", messages.join(" | "))
        }
        Ok(Acceptance::RuntimeRefused(why)) => panic!("the runtime refused the kernel: {why}"),
        Ok(Acceptance::NoEffect) => panic!("the runtime made no pipeline"),
        Err(error) => panic!("acceptance failed: {error}"),
    }
}

/// Kernel rows are made through the runtime one after another on one runtime device, and the
/// driver creates the pipeline each one's dispatch runs on; a vertex function alone is created
/// from the binder layout.
#[cfg(target_os = "linux")]
#[test]
fn kernel_rows_are_made_through_the_runtime_on_one_device_and_a_vertex_only_pipeline_is_created() {
    let rows = [
        ("store_const.ll", "store_const"),
        ("a1_pack_rounding.ll", "a1_pack_rounding"),
    ];
    for made in made(&rows) {
        accepted(Ok(made));
    }

    let vertex = compiled("stage_io.ll", "stage_io_vertex_position", RowStage::Vertex);
    assert!(matches!(vertex, RowPipeline::Render { .. }));
    assert_accepted(&vertex);
}

/// A kernel whose simd tier can require full subgroups is made, its representative threadgroup
/// whole simdgroups of the state's thread execution width. The 897
/// `VUID-VkPipelineShaderStageCreateInfo-flags-02759` rows were such kernels created at their
/// stage's placeholder local size of 1.
#[cfg(target_os = "linux")]
#[test]
fn a_full_subgroup_kernel_row_is_made_through_the_runtime() {
    for made in made(&[("simd_ops.ll", "simd_shuffle_k")]) {
        accepted(Ok(made));
    }
}

/// A kernel with an explicit imageblock is the runtime's typed refusal, as a guest's is: its
/// compute encoder does not set an imageblock's extent (`setImageblockWidth:height:`).
#[cfg(target_os = "linux")]
#[test]
fn an_imageblock_kernel_row_is_the_runtimes_refusal() {
    let made = made(&[("a4_imageblock_extent.ll", "a4_imageblock_extent")]).remove(0);
    assert!(
        matches!(&made, Acceptance::RuntimeRefused(why) if why.contains("mageblock")),
        "{made:?}"
    );
}

/// A post-tessellation vertex function's row is a tessellation pipeline (the synthesized vertex
/// and control stages before its evaluation stage), created as a patch list of one control point
/// per patch in Metal's default tessellation state, with the patch attributes and control-point
/// fields the row lays out pulled.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn post_tessellation_vertex_rows_are_created() {
    for entry in ["tess_quad_buffer", "tess_triangle_points"] {
        let pipeline = compiled("tessellation.ll", entry, RowStage::Vertex);
        let RowPipeline::Render {
            pipeline: render, ..
        } = &pipeline
        else {
            panic!("{entry}: a vertex row is a render pipeline");
        };
        assert!(render.graphics.tessellation.is_some(), "{entry}");
        assert_eq!(render.stages.len(), 3, "{entry}");
        assert_accepted(&pipeline);
    }
}

/// A vertex function that reads attributes is created with the descriptor the row synthesizes,
/// fixed function, and a fragment function with its generated
/// passthrough companion into the attachments its outputs imply.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn attribute_reading_vertex_and_fragment_rows_are_created() {
    let vertex = compiled("stage_io.ll", "stage_io_vertex", RowStage::Vertex);
    assert_accepted(&vertex);
    let fragment = compiled("stage_io.ll", "stage_io_fragment", RowStage::Fragment);
    let RowPipeline::Render {
        pipeline: render, ..
    } = &fragment
    else {
        panic!("a fragment row compiles to a render pipeline");
    };
    assert!(!render.graphics.rasterizer_discard);
    assert_accepted(&fragment);
}
