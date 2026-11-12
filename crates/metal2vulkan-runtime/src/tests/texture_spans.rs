//! The runs of a buffer's bytes that textures were made over (`texture_spans.rs`), which the
//! pooled bound reads: a texture made over bytes no run holds is the first over them, one made
//! over bytes a run holds is not, whether or not the textures noted there still live; runs that
//! meet join, and past the most runs kept a new one joins the nearest, which errs toward holding
//! bytes no texture was made over.

use crate::texture_spans::TextureSpans;

#[test]
fn bytes_a_texture_was_made_over_stay_noted() {
    let mut spans = TextureSpans::default();
    assert!(!spans.made_over(&(0..100)), "no texture yet");
    assert!(spans.note(0..100).is_empty(), "no shadow waited");
    assert!(spans.made_over(&(0..100)), "the same bytes");
    assert!(spans.made_over(&(99..200)), "bytes that meet them");
    assert!(!spans.made_over(&(100..200)), "the bytes after them");
    // A second texture over bytes apart from the first's, then one over both: one run.
    spans.note(300..400);
    assert!(!spans.made_over(&(100..300)), "the bytes between two runs");
    spans.note(50..350);
    assert!(spans.made_over(&(100..300)), "the joined run");
    assert!(!spans.made_over(&(400..500)), "past every run");
}

#[test]
fn past_the_most_runs_a_new_run_joins_the_nearest() {
    let mut spans = TextureSpans::default();
    // Runs 1000 bytes apart, more of them than are kept apart.
    for n in 0..20u64 {
        spans.note(n * 1000..n * 1000 + 10);
    }
    for n in 0..20u64 {
        assert!(
            spans.made_over(&(n * 1000..n * 1000 + 10)),
            "every texture's bytes stay noted ({n})"
        );
    }
    // Some gap between runs is now held: the runs err toward the image in place.
    let held_gaps = (0..19u64)
        .filter(|n| spans.made_over(&(n * 1000 + 10..n * 1000 + 1000)))
        .count();
    assert!(held_gaps > 0, "joined runs hold a gap");
}
