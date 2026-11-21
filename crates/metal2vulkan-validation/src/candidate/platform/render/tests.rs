//! The render executor's draw area (no device), and its targets' attachment forms.

use super::*;

/// The draw covers the whole target, its viewport flipped as the module docs flip it: compared as
/// plain values (`ash`'s types compare and print nothing).
#[test]
fn a_draw_covers_the_whole_target() {
    let (v, s) = draw_area(vk::Extent2D {
        width: 37,
        height: 23,
    });
    assert_eq!(
        (
            [v.x, v.y, v.width, v.height, v.min_depth, v.max_depth],
            [s.offset.x, s.offset.y],
            [s.extent.width, s.extent.height],
        ),
        ([0.0, 23.0, 37.0, -23.0, 0.0, 1.0], [0, 0], [37, 23])
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
    assert_eq!(DynamicRecord::of(D::LINE_WIDTH), None);
}
