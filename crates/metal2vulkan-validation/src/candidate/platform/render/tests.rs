//! The render executor's draws (no device): the whole target, and a rate-mapped pass's regions,
//! flipped as the module docs flip the viewport.

use super::*;
use metal2vulkan::bind::{RasterizationRateMap, RateMapAxis, RegionLimits};

/// A draw as plain values: the viewport's `[x, y, width, height, min_depth, max_depth]` and the
/// scissor's offset and extent (`ash`'s types compare and print nothing).
type Draw = ([f32; 6], [i32; 2], [u32; 2]);

fn plain(draws: Vec<(vk::Viewport, vk::Rect2D)>) -> Vec<Draw> {
    draws
        .into_iter()
        .map(|(v, s)| {
            (
                [v.x, v.y, v.width, v.height, v.min_depth, v.max_depth],
                [s.offset.x, s.offset.y],
                [s.extent.width, s.extent.height],
            )
        })
        .collect()
}

fn extent(width: u32, height: u32) -> vk::Extent2D {
    vk::Extent2D { width, height }
}

#[test]
fn a_pass_without_a_map_draws_once_over_the_whole_target() {
    assert_eq!(
        plain(draws(extent(37, 23), None)),
        [([0.0, 23.0, 37.0, -23.0, 0.0, 1.0], [0, 0], [37, 23])]
    );
}

/// Each region is one recording of the draw: its viewport, Metal's convention flipped to a
/// negative height from its bottom edge, and its scissor in physical pixels.
#[test]
fn a_rate_mapped_pass_draws_once_per_region() {
    let axis = |tiles: &[(u32, u32)]| RateMapAxis::from_tiles(32, tiles).unwrap();
    // Apple M3's compile of 37x23 at [0.5, 1] by [1, 0.5]: 33 screen pixels to 32, then 4 to 4, by
    // one tile of 23.
    let map = RasterizationRateMap::from_layers(
        [37, 23],
        vec![[axis(&[(33, 32), (4, 4)]), axis(&[(23, 23)])]],
    )
    .unwrap();
    let plan = map
        .regions(
            0,
            [0.0, 0.0, 37.0, 23.0],
            None,
            &RegionLimits::new([4096, 4096], [-8192.0, 8191.0]),
        )
        .unwrap();
    let width = (37.0f64 * 32.0 / 33.0) as f32;
    assert_eq!(
        plain(draws(extent(37, 23), Some(&plan))),
        [
            ([0.0, 23.0, width, -23.0, 0.0, 1.0], [0, 0], [32, 23]),
            ([-1.0, 23.0, 37.0, -23.0, 0.0, 1.0], [32, 0], [4, 23]),
        ]
    );
}

/// A fetched color attachment is also an input attachment, kept in `GENERAL` and read by the
/// fragment shader; a plain color attachment is neither.
#[test]
fn a_target_is_bound_for_its_fetch_form() {
    // `ash`'s types compare and print nothing: compare their raw values.
    let usage = |bound: Bound| bound.usage().as_raw();
    assert_eq!(usage(Bound::Color), 0);
    assert_eq!(
        usage(Bound::Fetched),
        vk::ImageUsageFlags::INPUT_ATTACHMENT.as_raw()
    );
    let raw = |bound: Bound| {
        let (layout, access, stages) = bound.attachment_use();
        (layout.as_raw(), access, stages)
    };
    let (layout, access, stages) = raw(Bound::Color);
    assert_eq!(layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL.as_raw());
    assert!(!access.contains(vk::AccessFlags::INPUT_ATTACHMENT_READ));
    assert!(stages == vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);
    let (layout, access, stages) = raw(Bound::Fetched);
    assert_eq!(layout, vk::ImageLayout::GENERAL.as_raw());
    assert!(access.contains(
        vk::AccessFlags::INPUT_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
    ));
    assert!(stages.contains(vk::PipelineStageFlags::FRAGMENT_SHADER));
}

/// Every state a linked render pipeline makes dynamic has a record, so the pipeline's creation
/// takes it and each draw sets it: the primitive restart beside the topology, through the
/// context's extended commands (a pipeline created with a dynamic restart its draw left unset was
/// `VUID-vkCmdDraw-None-04879`).
#[test]
fn every_state_a_render_pipeline_makes_dynamic_is_recorded() {
    use vk::DynamicState as D;
    let base = [
        D::VIEWPORT,
        D::SCISSOR,
        D::DEPTH_BIAS,
        D::BLEND_CONSTANTS,
        D::STENCIL_COMPARE_MASK,
        D::STENCIL_WRITE_MASK,
        D::STENCIL_REFERENCE,
    ];
    let extended = [
        D::PRIMITIVE_TOPOLOGY,
        D::PRIMITIVE_RESTART_ENABLE,
        D::CULL_MODE,
        D::FRONT_FACE,
        D::DEPTH_TEST_ENABLE,
        D::DEPTH_WRITE_ENABLE,
        D::DEPTH_COMPARE_OP,
        D::STENCIL_TEST_ENABLE,
        D::STENCIL_OP,
    ];
    for state in base {
        let record = DynamicRecord::of(state).expect("a recorded state");
        assert!(!record.needs_extended(), "{}", state.as_raw());
    }
    for state in extended {
        assert_eq!(
            DynamicRecord::of(state),
            Some(DynamicRecord::Extended),
            "{}",
            state.as_raw()
        );
    }
    assert_eq!(
        DynamicRecord::of(D::VERTEX_INPUT_BINDING_STRIDE),
        Some(DynamicRecord::Strides)
    );
    assert_eq!(DynamicRecord::of(D::LINE_WIDTH), None);
}
