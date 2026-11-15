//! The rule that draws a rate-mapped draw through its region-copy variant into the boxes of the
//! regions its triangles reach rather than fanning it out (`rate_mapped_pass::boxes_cheaper`,
//! `render/fan_out.rs`, **Boxed draws**), on the draws whose times set it.

use crate::rate_mapped_pass::boxes_cheaper;

/// The pieces lane's patches frame (`GEOMETRY_TRIANGLE_WORK`): its 64 commands' 903,450 indices
/// over 2074 regions, a command's 14,116 indices and 4705 triangles, the 61 it boxed about 18
/// regions a box (15,286,908 vertex-regions), took twice their fan-out on an Iris Xe and a
/// Radeon 780M. Such a draw is fanned out; boxed where its box holds 4 regions.
#[test]
fn a_draw_boxed_into_as_many_regions_as_the_patches_frames_is_fanned_out() {
    let (vertices, triangles) = (903_450 / 64, 903_450 / 64 / 3);
    assert!(!boxes_cheaper(18, vertices, triangles));
    assert!(boxes_cheaper(4, vertices, triangles));
}
