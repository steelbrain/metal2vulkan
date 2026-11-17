//! The harness's CPU-side parts: its statistics, its comparisons and its plans.

use crate::authored::WORKLOADS;
use crate::description;
use crate::plan::{
    Binding, BufferPlan, Comparison, Dispatch, Expect, Grid, Output, Plan, Source, Values,
};
use crate::run::{compare_outputs, percentile, zeroed, Report};
use crate::{Buffer, Storage};
use std::path::Path;
use std::ptr::NonNull;

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
    let nan = Values::Floats {
        values: vec![f32::NAN],
        tolerance: 0.0,
    };
    assert!(nan.compare(&le(&[f32::NAN.to_bits()])).is_ok());
    assert!(nan.compare(&le(&[0])).is_err());
}

#[test]
fn outputs_compare_by_their_comparison() {
    let mut plan = one_buffer_plan(16, None, 0);
    plan.outputs = vec![Output {
        buffer: 0,
        comparison: Comparison::Exact,
    }];
    let report = |words: &[u32]| Report {
        outputs: vec![le(words)],
        ..Report::default()
    };
    let reference = report(&[1, 2, 3, 4]);
    assert!(compare_outputs(&plan, &reference, &report(&[1, 2, 3, 4])).is_ok());
    let why = compare_outputs(&plan, &reference, &report(&[1, 2, 3, 5])).unwrap_err();
    assert_eq!(
        why.to_string(),
        "probe: buffer 0: 1 values differ; the first, [3], is 0x5, not 0x4"
    );
    plan.outputs[0].comparison = Comparison::Floats { tolerance: 0.5 };
    let floats = |values: [f32; 4]| report(&values.map(f32::to_bits));
    let reference = floats([1.0, 2.0, 3.0, 4.0]);
    assert!(compare_outputs(&plan, &reference, &floats([1.0, 2.0, 3.0, 4.25])).is_ok());
    assert!(compare_outputs(&plan, &reference, &floats([1.0, 2.0, 3.0, 5.0])).is_err());
}

fn one_buffer_plan(length: u64, contents: Option<Vec<u8>>, bound: usize) -> Plan {
    Plan {
        name: "probe".to_owned(),
        library: Source::Authored,
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
        outputs: Vec::new(),
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
    for workload in WORKLOADS {
        let plan = (workload.plan)();
        assert_eq!(plan.name, workload.name);
        plan.validate().unwrap();
    }
}

const DESCRIPTION: &str = "\
# Two workloads.
workload first kernels.metallib
buffer 16 shared hex 000102030405060708090a0b0c0d0e0f
buffer 32 private words 1   # trailing comment
set-buffer 0 0
set-buffer 1 1 16
set-bytes 2 4 zero
dispatch k threads 64 2 1 32 2 1 repeat 3
set-bytes 2 4 unit 2
dispatch k threadgroups 2 1 1 32 1 1
check 1 float 0.5

workload second other.metallib
buffer 16 private zero
set-buffer 0 0
dispatch m threadgroups 1 1 1 16 1 1
check 0 exact
";

#[test]
fn a_description_makes_its_plans() {
    let described = description::parse(DESCRIPTION, Path::new("/dir")).unwrap();
    let plans: Vec<Plan> = described
        .iter()
        .map(|workload| workload.plan().unwrap())
        .collect();
    let [first, second] = plans.as_slice() else {
        panic!("two plans, not {}", plans.len());
    };
    assert_eq!(first.name, "first");
    assert_eq!(
        first.library,
        Source::Metallib("/dir/kernels.metallib".into())
    );
    assert_eq!(first.buffers[0].storage, Storage::Shared);
    assert_eq!(first.buffers[0].contents, Some((0..16).collect()));
    assert_eq!(first.buffers[1].storage, Storage::Private);
    let words: Vec<u8> = (crate::plan::seeded_words(1, 8).iter())
        .flat_map(|word| word.to_le_bytes())
        .collect();
    assert_eq!(first.buffers[1].contents, Some(words));
    // `repeat 3`: three dispatches, the bindings before the first alone; then the fourth's own.
    assert_eq!(first.dispatches.len(), 4);
    let bindings: Vec<usize> = first.dispatches.iter().map(|d| d.bindings.len()).collect();
    assert_eq!(bindings, [3, 0, 0, 1]);
    assert!(matches!(
        first.dispatches[0].bindings[1],
        Binding::Buffer {
            index: 1,
            buffer: 1,
            offset: 16
        }
    ));
    assert!(matches!(
        first.dispatches[2].grid,
        Grid::Threads {
            threads: [64, 2, 1],
            per_group: [32, 2, 1]
        }
    ));
    match &first.dispatches[3].bindings[0] {
        Binding::Bytes { index: 2, bytes } => {
            let unit = f32::from_le_bytes(bytes[..4].try_into().unwrap());
            assert!((0.0..1.0).contains(&unit), "{unit}");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(
        first.outputs[0].comparison,
        Comparison::Floats { tolerance: 0.5 }
    );
    assert_eq!(
        second.library,
        Source::Metallib("/dir/other.metallib".into())
    );
    assert_eq!(
        second.buffers[0].contents, None,
        "a zero buffer is left as made"
    );
    assert_eq!(second.outputs[0].comparison, Comparison::Exact);
}

/// The error of the description `text`.
fn refused(text: &str) -> String {
    description::parse(text, Path::new("/dir"))
        .unwrap_err()
        .to_string()
}

/// [`refused`] of `body` after a workload's first two lines.
fn refusal(body: &str) -> String {
    refused(&format!(
        "workload w k.metallib\nbuffer 16 private zero\n{body}"
    ))
}

#[test]
fn a_faulty_description_is_refused_naming_its_line() {
    let dispatch = "set-buffer 0 0\ndispatch k threadgroups 1 1 1 16 1 1";
    assert_eq!(
        refused("buffer 16 private zero"),
        "line 1: buffer before a workload"
    );
    assert_eq!(
        refused("workload w"),
        "line 1: workload takes NAME METALLIB"
    );
    assert_eq!(
        refusal("buffer 16 managed zero"),
        "line 3: storage managed is neither shared nor private"
    );
    assert_eq!(
        refusal("buffer 16 shared hex 00ff"),
        "line 3: the fill has 2 bytes, not 16"
    );
    assert_eq!(
        refusal("buffer 16 shared words x"),
        "line 3: x is not a whole number"
    );
    assert_eq!(
        refusal("buffer 16 shared noise 1"),
        "line 3: fill noise 1 is not zero, words, unit or hex"
    );
    assert_eq!(
        refusal("set-bytes 0 6 words 1"),
        "line 3: a fill of words needs a multiple of 4 bytes, not 6"
    );
    assert_eq!(
        refusal("dispatch k threadgroups 1 1 1 16 1"),
        "line 3: dispatch takes six sizes"
    );
    assert_eq!(
        refusal("dispatch k waves 1 1 1 16 1 1"),
        "line 3: dispatch kind waves is neither threadgroups nor threads"
    );
    assert_eq!(
        refusal(&format!("{dispatch} repeat 0")),
        "line 4: repeat 0 dispatches nothing"
    );
    assert_eq!(
        refusal("check 0 close"),
        "line 3: check 0 close is no directive"
    );
    assert_eq!(refusal("set-buffer 0 0"), "workload w has no dispatch");
    assert_eq!(
        refusal(&format!("{dispatch}\nset-buffer 1 0")),
        "workload w sets a binding after its last dispatch"
    );
    assert_eq!(
        refusal(&format!("{dispatch}\nworkload w k.metallib")),
        "line 5: a second workload w"
    );
    assert_eq!(
        refusal("set-buffer 0 3\ndispatch k threadgroups 1 1 1 16 1 1"),
        "w: dispatch 0 binds buffer 3, which the plan lacks"
    );
}

/// A `Shared` buffer's bytes in host memory.
struct HostBuffer(Vec<u32>);

impl Buffer for HostBuffer {
    fn contents(&self) -> Option<NonNull<u8>> {
        NonNull::new(self.0.as_ptr().cast_mut().cast())
    }
}

#[test]
fn a_new_buffer_is_zero_only_where_every_word_read_is() {
    let mut buffer = HostBuffer(vec![0; 4096]);
    assert!(zeroed(&buffer, 16384, 4).is_ok());
    buffer.0[1500] = 7;
    let why = zeroed(&buffer, 16384, 4).unwrap_err().to_string();
    assert_eq!(why, "a new buffer of 16384 bytes holds 0x7 at byte 6000");
    // One word a page reads the first of each page only.
    assert!(zeroed(&buffer, 16384, 4096).is_ok());
    buffer.0[2048] = 1;
    assert!(zeroed(&buffer, 16384, 4096).is_err());
    assert!(zeroed(&HostBuffer(vec![9; 4]), 16, 4).is_err());
}
