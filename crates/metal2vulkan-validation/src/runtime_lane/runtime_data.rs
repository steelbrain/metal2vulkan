//! The runtime-data form: the class of a texture's format reaches a function as a push word the
//! binder writes per dispatch or draw, not as a specialization constant, so one `VkPipeline` of a
//! variant serves every bound format, each answering as Metal's committed observation records. A
//! sampler's words stay specialized: another sampler state takes a `VkPipeline` of its own.
//!
//! The kernel `msf_slot2d_linear_edge` (owned MSL, `fixtures/public/engine/
//! msf_runtime_samplers.metal`) samples its slot texture through a constexpr sampler into its
//! even outputs and through the runtime sampler at slot 0 into its odd ones. Its committed cases
//! bind four formats with a runtime sampler filtering linearly, clamped to the edge: each runs on
//! the one pipeline state, and each output is Metal's, within
//! `argument_linear_samples::DEVICE_FILTER` where the device's own sampler filters
//! (`argument_linear_samples::device_filters`) a format that bound was measured on
//! ([`FILTER_MEASURED`]; another's differences are printed). The kernel `msf_slot2d_nearest_repeat` is the same
//! macro under another constexpr sampler; its cases bind a runtime sampler filtering to the nearest
//! texel, repeating, and their odd outputs are Metal's runtime-sampler answers for those inputs.
//! Run on the first kernel's pipeline state, its odd outputs are its own pipeline state's, and
//! Metal's wherever the device answers the case as Metal on its own state (within its conversion
//! of the codes where it does not sample through Metal's own texture unit; the others, such as a
//! device's sRGB decode, `kernel_textures` judges). Across each sampler's runs the device makes one
//! `VkPipeline` per variant the runs take (an sRGB texture's raw codes take a variant of their
//! own), never one per format.

use super::argument_linear_samples::{alike, device_filters};
use super::pixel_samplers::metal_sampler;
use super::{
    at_plan, converted_holds, layer_reported_nothing_at_all, pipeline_state, public_cases,
    run_with, validated_device_with, Messages, ScratchCache,
};
use crate::candidate::engine::resources::pixel_format;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::ExecutedPipeline;
use std::collections::BTreeSet;
use std::sync::Arc;

/// The kernel whose pipeline state every run uses, with a runtime sampler filtering linearly and
/// clamped to the edge in its cases.
const LINEAR_EDGE: &str = "msf_slot2d_linear_edge";

/// The same kernel under another constexpr sampler, with a runtime sampler filtering to the
/// nearest texel and repeating in its cases.
const NEAREST_REPEAT: &str = "msf_slot2d_nearest_repeat";

/// The formats `PrecisionNote::DeviceSampling`'s recorded bound of a linear filter was measured on.
const FILTER_MEASURED: [PixelFormat; 7] = [
    PixelFormat::R8Unorm,
    PixelFormat::RG8Unorm,
    PixelFormat::RGBA8Unorm,
    PixelFormat::BGRA8Unorm,
    PixelFormat::RGBA8UnormSRGB,
    PixelFormat::RGBA16Unorm,
    PixelFormat::RGBA16Float,
];

/// The bytes of one output element (a `float4`).
const ELEMENT: usize = 16;

/// The odd elements of `output`: the samples through the runtime sampler.
fn runtime_column(output: &[u8]) -> Vec<u8> {
    output
        .chunks(ELEMENT)
        .skip(1)
        .step_by(2)
        .flatten()
        .copied()
        .collect()
}

#[test]
fn one_pipeline_per_sampler_answers_every_bound_format_as_metal() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
    let cases = |entry: &str| {
        let found: Vec<_> = public_cases(|case| case.entry == entry)
            .unwrap()
            .into_iter()
            .map(|(name, case)| case.unwrap_or_else(|why| panic!("{name}: {why}")))
            .collect();
        assert!(found.len() >= 3, "{entry}: {} committed cases", found.len());
        found
    };
    let (edge, repeat) = (cases(LINEAR_EDGE), cases(NEAREST_REPEAT));
    let formats = |cases: &[super::PublicCase]| -> BTreeSet<String> {
        cases
            .iter()
            .flat_map(|case| case.compute.textures.values())
            .map(|texture| format!("{:?}", texture.format))
            .collect()
    };
    assert_eq!(formats(&edge).len(), edge.len(), "one case per format");
    let pipeline = pipeline_state(&device, &edge[0]).unwrap();
    let before = device.work_counts().variant_pipelines_made;
    // Each run's output, the variants it executed added to `variants`.
    let ran = |case: &super::PublicCase, variants: &mut BTreeSet<usize>| {
        let run = run_with(&device, case, &pipeline).unwrap();
        for execution in &run.executions {
            match execution.pipeline() {
                ExecutedPipeline::Compute(variant) => {
                    variants.insert(Arc::as_ptr(variant) as usize)
                }
                ExecutedPipeline::Render(_) => panic!("a render pipeline in a dispatch"),
            };
        }
        run.output
    };
    let mut variants = BTreeSet::new();
    for case in &edge {
        // Where the device's own sampler filters (`argument_linear_samples::device_filters`),
        // each lane within `argument_linear_samples::DEVICE_FILTER`, for a format the bound
        // `PrecisionNote::DeviceSampling` records was measured on; another's differences are
        // printed.
        let within = device_filters(&device);
        let measured = case
            .compute
            .textures
            .values()
            .all(|texture| FILTER_MEASURED.contains(&pixel_format(texture.format)));
        let got = ran(case, &mut variants);
        let lanes = |bytes: &[u8]| -> Vec<f32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let (got_lanes, metal_lanes) = (lanes(&got), lanes(&case.metal_output));
        if within && !measured {
            let differ = (got_lanes.iter().zip(&metal_lanes))
                .filter(|(g, m)| g.to_bits() != m.to_bits())
                .count();
            let most = (got_lanes.iter().zip(&metal_lanes))
                .map(|(g, m)| (g - m).abs())
                .fold(0.0f32, f32::max);
            println!(
                "RUNTIME-DATA-DIFFERS {}: {}: {differ} of {} lanes the device's filter, at most \
                 {most} off Metal's",
                device.name(),
                case.case.name,
                got_lanes.len()
            );
            assert_eq!(got.len(), case.metal_output.len(), "{}", case.case.name);
            continue;
        }
        assert!(
            got.len() == case.metal_output.len()
                && got_lanes
                    .iter()
                    .zip(&metal_lanes)
                    .all(|(&g, &m)| alike(g, m, within)),
            "{} (within the device's filter's bound: {within}): {got_lanes:?}, Metal's \
             {metal_lanes:?}",
            case.case.name
        );
    }
    let made_edge = device.work_counts().variant_pipelines_made - before;
    let variants_edge = variants.len() as u64;
    let mut variants = BTreeSet::new();
    let shared: Vec<Vec<u8>> = repeat
        .iter()
        .map(|case| runtime_column(&ran(case, &mut variants)))
        .collect();
    let made_repeat = device.work_counts().variant_pipelines_made - before - made_edge;
    println!(
        "RUNTIME-DATA {} runs of {} formats under the first sampler: {variants_edge} variants, \
         {made_edge} VkPipelines made; {} runs of {} formats under the second: {} variants, \
         {made_repeat} VkPipelines made",
        edge.len(),
        formats(&edge).len(),
        repeat.len(),
        formats(&repeat).len(),
        variants.len(),
    );
    // The second sampler's answers: on the first kernel's pipeline state, the runtime sampler's
    // samples are those of the case's own pipeline state, and Metal's wherever the device
    // answers the case as Metal does on its own state (`kernel_textures` judges the others: a
    // device's sRGB decode, for one).
    let mut metal = 0;
    for (case, shared) in repeat.iter().zip(&shared) {
        let own = runtime_column(&super::run_case(&device, case).unwrap().output);
        assert_eq!(
            shared, &own,
            "{} on {LINEAR_EDGE}'s pipeline state",
            case.case.name
        );
        // The device's conversion of a texel's codes, where it does not sample through Metal's
        // own texture unit (`converted_holds`).
        let exact = metal_sampler(&device);
        let floats = |bytes: &[u8]| -> Vec<f32> {
            (bytes.as_chunks::<4>().0.iter())
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        let (own_lanes, metal_lanes) = (floats(&own), floats(&runtime_column(&case.metal_output)));
        if own_lanes.len() == metal_lanes.len()
            && (own_lanes.iter().zip(&metal_lanes)).all(|(&g, &m)| converted_holds(g, m, exact))
        {
            metal += 1;
        }
    }
    assert!(
        metal >= 2,
        "{metal} of the second sampler's formats answered as Metal"
    );
    assert_eq!(
        made_edge, variants_edge,
        "one VkPipeline per variant under the first sampler, whatever the bound format"
    );
    assert!(
        made_repeat <= variants.len() as u64,
        "{made_repeat} VkPipelines for {} variants under the second sampler: at most one per \
         variant, whatever the bound format",
        variants.len()
    );
    layer_reported_nothing_at_all(&messages);
}
