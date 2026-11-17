//! The harness's CPU-side parts: its statistics, its comparisons and its plans.

use crate::authored::WORKLOADS;
use crate::plan::{Binding, BufferPlan, Dispatch, Expect, Grid, Plan, Values};
use crate::run::percentile;
use crate::Storage;

#[test]
fn percentiles_take_the_nearest_rank() {
    let values: Vec<f64> = (1..=10).rev().map(f64::from).collect();
    assert_eq!(percentile(&values, 10), Some(1.0));
    assert_eq!(percentile(&values, 50), Some(5.0));
    assert_eq!(percentile(&values, 90), Some(9.0));
    assert_eq!(percentile(&[7.0], 10), Some(7.0));
    assert_eq!(percentile(&[], 50), None);
}

fn le(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

#[test]
fn a_comparison_names_how_many_values_differ_and_the_first() {
    let want = Values::Words(vec![1, 2, 3, 4]);
    assert!(want.compare(&le(&[1, 2, 3, 4])).is_ok());
    let why = want.compare(&le(&[1, 9, 3, 8])).unwrap_err().to_string();
    assert_eq!(why, "2 values differ; the first, [1], is 0x9, not 0x2");
}

#[test]
fn floats_differ_past_their_tolerance_and_at_a_nan() {
    let want = Values::Floats {
        values: vec![1.0, 2.0],
        tolerance: 0.25,
    };
    let floats = |values: [f32; 2]| le(&values.map(f32::to_bits));
    assert!(want.compare(&floats([1.2, 1.8])).is_ok());
    assert!(want.compare(&floats([1.3, 2.0])).is_err());
    assert!(want.compare(&floats([f32::NAN, 2.0])).is_err());
}

fn one_buffer_plan(length: u64, contents: Option<Vec<u8>>, bound: usize) -> Plan {
    Plan {
        name: "probe".to_owned(),
        buffers: vec![BufferPlan {
            length,
            storage: Storage::Private,
            contents,
        }],
        dispatches: vec![Dispatch {
            function: "copy_words".to_owned(),
            bindings: vec![Binding::Buffer {
                index: 0,
                buffer: bound,
                offset: 0,
            }],
            grid: Grid::groups_of_256(1),
        }],
        expect: vec![Expect {
            buffer: 0,
            values: Values::Words(vec![0; 4]),
        }],
    }
}

#[test]
fn a_plan_with_a_part_length_or_a_missing_buffer_is_refused() {
    assert!(one_buffer_plan(16, None, 0).validate().is_ok());
    assert!(one_buffer_plan(12, None, 0).validate().is_err());
    assert!(one_buffer_plan(16, Some(vec![0; 8]), 0).validate().is_err());
    assert!(one_buffer_plan(16, None, 1).validate().is_err());
    let mut short = one_buffer_plan(16, None, 0);
    short.expect[0].values = Values::Words(vec![0; 5]);
    assert!(short.validate().is_err());
}

/// Each authored plan is whole: its builder's inputs, kernels and references agree in shape.
#[test]
fn every_authored_plan_validates() {
    for (name, build) in WORKLOADS {
        let plan = build();
        assert_eq!(plan.name, name);
        plan.validate().unwrap();
    }
}
