//! A draw whose bind shape may have changed compares whole only the shapes its state took lately
//! of its key beside its used resources (`live_bind::RecentShapes`,
//! `BindState::shape_key_beside_used`): it finds the shape a whole comparison of each, the latest
//! first, finds, comparing one where that compared one per shape kept since.

use std::sync::Arc;

use metal2vulkan::bind::BindState;

use crate::live_bind::RecentShapes;

/// States of five shapes, apart in a field the shape reads alone (the render area).
fn states() -> Vec<BindState> {
    (1..=5)
        .map(|side| {
            let mut state = BindState::default();
            state.render_area = [side, side];
            state
        })
        .collect()
}

#[test]
fn a_recent_shape_is_found_comparing_only_those_of_its_key() {
    let states = states();
    let mut recent = RecentShapes::default();
    for state in &states {
        let shape = Arc::new(state.shape());
        recent.keep(shape.key_beside_used(), &shape);
    }
    for (at, state) in states.iter().enumerate() {
        let mut compared = 0;
        let found = recent.find(state, state.shape_key_beside_used(), &mut compared);
        assert_eq!(found.as_deref(), Some(&state.shape()), "state {at}");
        // A whole comparison of each, the latest first, compared one per shape kept after it too.
        let before = states.len() - at;
        assert_eq!(compared, 1, "state {at}: compared whole (before: {before})");
    }
    let mut other = BindState::default();
    other.render_area = [9, 9];
    let mut compared = 0;
    let found = recent.find(&other, other.shape_key_beside_used(), &mut compared);
    assert!(found.is_none(), "a state of another shape found one");
    assert_eq!(compared, 0, "a state of another shape compared whole");
}
