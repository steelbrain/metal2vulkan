//! The harness's CPU-side parts: its statistics, its comparisons and its plans.

use crate::authored::WORKLOADS;
use crate::description;
use crate::plan::{
    Binding, BufferPlan, Comparison, Constant, Dispatch, Expect, Filter, Grid, Output, Plan,
    Source, TextureExpect, TexturePlan, Values,
};
use crate::rt::{self, Ranges};
use crate::run::{
    compare_outputs, copy_dispatches, parts, percentile, run_each, settled, steady, warm_up,
    zeroed, CreateReport, Report, Runs, Session,
};
use crate::{Buffer, Refused, Result, Storage, Waits};
use metal2vulkan_rt::{Error as RtError, Gap, Limit};
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
            constants: Vec::new(),
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
        textures: Vec::new(),
        expect_textures: Vec::new(),
        outputs: Vec::new(),
        per_command_buffer: None,
        per_encoder: None,
    }
}

#[test]
fn a_plan_with_a_part_length_a_missing_buffer_or_no_dispatch_is_refused() {
    assert!(one_buffer_plan(16, None, 0).validate().is_ok());
    assert!(one_buffer_plan(12, None, 0).validate().is_err());
    assert!(one_buffer_plan(16, Some(vec![0; 8]), 0).validate().is_err());
    assert!(one_buffer_plan(16, None, 1).validate().is_err());
    let mut short = one_buffer_plan(16, None, 0);
    short.expect[0].values = Values::Words(vec![0; 5]);
    assert!(short.validate().is_err());
    let mut split = one_buffer_plan(16, None, 0);
    split.per_command_buffer = Some(1);
    assert!(split.validate().is_ok());
    split.per_command_buffer = Some(0);
    assert!(split.validate().is_err());
    let mut passes = one_buffer_plan(16, None, 0);
    passes.per_encoder = Some(1);
    assert!(passes.validate().is_ok());
    passes.per_encoder = Some(0);
    assert!(passes.validate().is_err());
    let mut idle = one_buffer_plan(16, None, 0);
    idle.dispatches.clear();
    assert!(idle.validate().is_err());
}

/// A plan of one `side` by `side` texture, bound at `bound`, with a sampler, and its
/// expectation of `expected` floats.
fn one_texture_plan(side: u32, texels: Option<usize>, bound: usize, expected: usize) -> Plan {
    let mut plan = one_buffer_plan(16, None, 0);
    plan.textures = vec![TexturePlan {
        width: side,
        height: side,
        texels: texels.map(|count| vec![0.0; count]),
    }];
    plan.dispatches[0].bindings = vec![
        Binding::Texture {
            index: 0,
            texture: bound,
        },
        Binding::Sampler {
            index: 0,
            filter: Filter::Linear,
        },
    ];
    plan.expect_textures = vec![TextureExpect {
        texture: 0,
        values: Values::Floats {
            values: vec![0.0; expected],
            tolerance: 0.0,
        },
    }];
    plan
}

#[test]
fn a_plan_with_a_part_tile_texture_a_missing_texture_or_too_many_texels_is_refused() {
    assert!(one_texture_plan(16, Some(4 * 256), 0, 4 * 256)
        .validate()
        .is_ok());
    assert!(one_texture_plan(16, None, 0, 4).validate().is_ok());
    assert!(one_texture_plan(24, None, 0, 4).validate().is_err());
    assert!(one_texture_plan(16, Some(4 * 255), 0, 4)
        .validate()
        .is_err());
    assert!(one_texture_plan(16, None, 1, 4).validate().is_err());
    assert!(one_texture_plan(16, None, 0, 4 * 256 + 1)
        .validate()
        .is_err());
    let mut unknown = one_texture_plan(16, None, 0, 4);
    unknown.expect_textures[0].texture = 1;
    assert!(unknown.validate().is_err());
}

/// A run's dispatches cut into command buffers, and a command buffer's into encoders: whole parts
/// in order, the last one shorter.
#[test]
fn parts_cut_a_range_in_order_the_last_one_shorter() {
    let cut =
        |range, per| (parts(range, per).map(|part| (part.start, part.end))).collect::<Vec<_>>();
    assert_eq!(cut(0..5, Some(2)), [(0, 2), (2, 4), (4, 5)]);
    assert_eq!(cut(10..20, Some(5)), [(10, 15), (15, 20)]);
    assert_eq!(cut(4..10, None), [(4, 10)]);
    assert_eq!(cut(3..4, Some(8)), [(3, 4)]);
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
constant 3 53 01
constant 0 29 ffffffff
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
    // The constants before a dispatch are its kernel's alone.
    assert!(first.dispatches[..3].iter().all(|d| d.constants.is_empty()));
    let constant = |index, data_type, value: &[u8]| Constant {
        index,
        data_type,
        value: value.to_vec(),
    };
    assert_eq!(
        first.dispatches[3].constants,
        [constant(3, 53, &[1]), constant(0, 29, &[0xff; 4])]
    );
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
        "workload w sets a binding or a constant after its last dispatch"
    );
    assert_eq!(
        refusal(&format!("{dispatch}\nconstant 0 53 01")),
        "workload w sets a binding or a constant after its last dispatch"
    );
    assert_eq!(
        refusal("constant 0 53 0x"),
        "line 3: 0x is not whole bytes of hex"
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

    fn did_modify_range(&self, _: u64, _: u64) -> Result<()> {
        Ok(())
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

#[test]
fn ranges_place_at_the_lowest_free_range_and_merge_what_is_given_back() {
    let mut free = Ranges::new(1024);
    assert_eq!(free.take(256), Some(0));
    assert_eq!(free.take(256), Some(256));
    assert_eq!(free.take(256), Some(512));
    // A range given back is the next one taken of its length: the creation workloads' reuse.
    free.give(256..512);
    assert_eq!(free.take(256), Some(256));
    free.give(256..512);
    // Given back between two free ranges, it merges with both.
    free.give(0..256);
    assert_eq!(free, Ranges(vec![0..512, 768..1024]));
    free.give(512..768);
    assert_eq!(free, Ranges::new(1024));
    assert_eq!(free.take(2048), None);
    assert_eq!(free.take(1024), Some(0));
    assert_eq!(free.take(1), None);
}

#[test]
fn a_time_settles_once_its_last_two_windows_agree_within_three_percent() {
    let ramp: Vec<f64> = (0..20).map(|run| 200.0 - run as f64 * 5.0).collect();
    assert!(!settled(&ramp), "a falling time has not settled");
    let mut times = ramp;
    times.extend([100.0; 20]);
    assert!(settled(&times));
    times.extend([
        100.0, 103.0, 99.0, 102.0, 101.0, 100.0, 98.0, 102.0, 100.0, 101.0,
    ]);
    assert!(settled(&times), "within 3%");
    times.extend([110.0; 10]);
    assert!(!settled(&times), "10% apart");
    assert!(!settled(&[100.0; 19]), "two windows are 20 runs");
}

#[test]
fn without_settling_a_warm_up_runs_its_least_count() {
    let mut count = 0;
    let runs = Runs {
        warmup: 7,
        settle: false,
        timed: 0,
    };
    let warmup = warm_up(runs, || {
        count += 1;
        Ok(1.0)
    })
    .unwrap();
    assert_eq!((warmup.runs, warmup.settled, count), (7, false, 7));
}

#[test]
fn a_steady_time_is_the_lowest_median_of_windows_of_ten() {
    assert_eq!(steady(&[1.0; 4]), None, "fewer than 5 runs");
    assert_eq!(
        steady(&[3.0, 1.0, 2.0, 5.0, 4.0]),
        Some(3.0),
        "their median below 10"
    );
    // A clock that drops partway through: the faster windows' level, not the mix.
    let mut times = vec![100.0; 20];
    times.extend([105.0; 30]);
    times[3] = 400.0;
    assert_eq!(steady(&times), Some(100.0));
    assert_eq!(percentile(&times, 50), Some(105.0));
}

/// The new runtime's known gaps and the device's limits are typed refusals; its other errors are
/// failures.
#[test]
fn the_new_runtime_refuses_by_its_gaps_and_limits_alone() {
    let refused = |error: RtError| rt::error(error).downcast::<Refused>().is_ok();
    assert!(refused(RtError::Unsupported(Gap::KernelInput)));
    assert!(refused(RtError::Limit(Limit::BlockIds)));
    assert!(!refused(RtError::Quiesced));
    assert_eq!(rt::size([4, 2, 1]).unwrap(), [4, 2, 1]);
    assert!(rt::size([1 << 32, 1, 1]).unwrap_err().is::<Refused>());
}

/// A session that answers each plan with a report of `outputs`, or with `error`.
struct Answers {
    outputs: Vec<u8>,
    error: Option<fn() -> crate::Error>,
}

impl Session for Answers {
    fn device_name(&self) -> String {
        "answers".to_owned()
    }

    fn run(&mut self, _: &Plan, _: Runs) -> Result<Report> {
        match self.error {
            Some(error) => Err(error()),
            None => Ok(Report {
                outputs: vec![self.outputs.clone()],
                ..Report::default()
            }),
        }
    }

    fn create(&mut self, _: u64, _: Runs) -> Result<CreateReport> {
        Ok(CreateReport::default())
    }

    fn finish(self: Box<Self>) -> Option<Vec<(&'static str, u64)>> {
        None
    }

    fn pipelines_created(&self) -> u64 {
        0
    }

    fn first_use_waits(&self) -> Option<Waits> {
        None
    }
}

/// A runtime's refusal leaves the others' outputs compared among themselves; a failure, or
/// outputs that differ, is an error.
#[test]
fn a_refusal_is_reported_beside_the_runs_and_a_failure_is_not() {
    let mut plan = one_buffer_plan(16, None, 0);
    plan.outputs = vec![Output {
        buffer: 0,
        comparison: Comparison::Exact,
    }];
    let runs = Runs {
        warmup: 0,
        settle: false,
        timed: 1,
    };
    let answers = |outputs: &[u32], error: Option<fn() -> crate::Error>| -> Box<dyn Session> {
        Box::new(Answers {
            outputs: le(outputs),
            error,
        })
    };
    let refuse: fn() -> crate::Error = || Box::new(Refused("a known gap".to_owned()));
    let fail: fn() -> crate::Error = || "a failure".into();
    let mut sessions = vec![
        ("a", answers(&[1, 2, 3, 4], None)),
        ("b", answers(&[], Some(refuse))),
        ("c", answers(&[1, 2, 3, 4], None)),
    ];
    let each = run_each(&plan, &mut sessions, runs).unwrap();
    assert!(each[0].is_ok() && each[2].is_ok());
    assert_eq!(each[1].as_ref().unwrap_err().to_string(), "a known gap");
    sessions[2].1 = answers(&[1, 2, 3, 5], None);
    assert!(run_each(&plan, &mut sessions, runs).is_err());
    sessions[2].1 = answers(&[], Some(fail));
    assert!(run_each(&plan, &mut sessions, runs).is_err());
}

/// The copy kernel's dispatches cover the bytes once, in order, each of whole threadgroups.
#[test]
fn the_copy_moves_every_byte_in_whole_threadgroups() {
    for length in [16, 1024, 4096, 4112, 65535 * 4096, 65535 * 4096 + 4096 + 48] {
        let dispatches = copy_dispatches(length);
        let mut next = 0;
        for &(offset, groups, threads) in &dispatches {
            assert_eq!(offset, next, "{length}: {dispatches:?}");
            assert!(
                groups <= 65535 && (1..=256).contains(&threads),
                "{dispatches:?}"
            );
            next += groups * threads * 16;
        }
        assert_eq!(next, length, "{length}: {dispatches:?}");
    }
    assert_eq!(copy_dispatches(1024), [(0, 1, 64)]);
    assert_eq!(copy_dispatches(4112), [(0, 1, 256), (4096, 1, 1)]);
}
