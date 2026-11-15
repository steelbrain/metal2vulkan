//! The rule that draws a rate-mapped draw through its region-copy variant into the boxes of the
//! regions its triangles reach rather than fanning it out (`rate_mapped_pass::boxes_cheaper`,
//! `render/fan_out.rs`, **Boxed draws**), on the draws whose times set it.

use crate::rate_mapped_pass::boxes_cheaper;

/// The pieces lane's patches frame (`GEOMETRY_TRIANGLE_WORK`): the 61 of its 64 commands it boxed
/// over 2074 regions, 566,382 indices (9285 indices and 3095 triangles a command), into 15,286,908
/// vertex-regions, 27 regions a box, took about 2.5 and 2.8 times their fan-out on an Iris Xe and a
/// Radeon 780M. Such a draw is fanned out; boxed where its box holds 4 regions.
#[test]
fn a_draw_boxed_into_as_many_regions_as_the_patches_frames_is_fanned_out() {
    let (vertices, triangles) = (566_382 / 61, 566_382 / 61 / 3);
    assert!(!boxes_cheaper(27, vertices, triangles));
    assert!(boxes_cheaper(4, vertices, triangles));
}
