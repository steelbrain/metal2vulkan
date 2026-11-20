//! The harness's CPU-side parts: its statistics, its comparisons and its plans.

use crate::authored::{Makes, OwnRow, WORKLOADS};
use crate::description;
use crate::plan::{
    Binding, BufferPlan, Comparison, Constant, Dispatch, Expect, Filter, Grid, Output, Plan,
    SamplerPlan, Scaled, Source, TextureExpect, TexturePlan, TextureSource, Values,
};
use crate::rt::{self, Ranges};
use crate::run::{
    compare_outputs, copy_dispatches, create, matrix_time, parts, percentile, run_each, settled,
    steady, warm_up, zeroed, CreateReport, Handled, OwnReport, Report, Runs, Session,
};
use crate::{Buffer, Device, Failed, GaugeReader, Refused, Result, Storage, Waits};
use metal2vulkan::metal::TextureUsage;
use metal2vulkan_runtime::{Error as RtError, Gap, KernelInputKind, Limit};
use std::cell::{Cell, RefCell};
use std::path::Path;
use std::ptr::NonNull;
use std::rc::Rc;

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
            modified: None,
        }],
        dispatches: vec![Dispatch {
            blits: Vec::new(),
            function: "copy_words".to_owned(),
            linked: Vec::new(),
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
        commit_to_complete: false,
        frames: false,
        scaled: None,
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
    plan.textures = vec![TexturePlan::rgba16_float(
        side,
        side,
        texels.map(|count| vec![0.0; count]),
    )];
    plan.dispatches[0].bindings = vec![
        Binding::Texture {
            index: 0,
            texture: bound,
        },
        Binding::Sampler {
            index: 0,
            sampler: SamplerPlan::clamped(Filter::Linear),
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

#[test]
fn a_plan_refuses_a_texture_the_harness_cannot_make_or_fill() {
    let problem = |change: fn(&mut TexturePlan)| {
        let mut plan = one_texture_plan(16, None, 0, 4);
        plan.expect_textures.clear();
        change(&mut plan.textures[0]);
        plan.validate().err().map(|why| why.to_string())
    };
    assert_eq!(problem(|_| {}), None);
    assert_eq!(
        problem(|t| t.texture_type = metal2vulkan::metal::TextureType::Type3D),
        Some(
            "probe: texture 0: a Type3D texture of 1 slices and 1 samples: the harness makes 2D \
             textures and 2D arrays of one sample, and 2D multisample textures of more"
                .to_owned()
        )
    );
    assert_eq!(
        problem(|t| t.array_length = 2),
        Some(
            "probe: texture 0: a Type2D texture of 2 slices and 1 samples: the harness makes 2D \
             textures and 2D arrays of one sample, and 2D multisample textures of more"
                .to_owned()
        )
    );
    assert_eq!(
        problem(|t| {
            t.texture_type = metal2vulkan::metal::TextureType::Type2DArray;
            t.array_length = 2;
            t.source = TextureSource::OverBuffer;
        }),
        Some("probe: texture 0: a texture over a buffer is 2D, as Metal requires".to_owned())
    );
    assert_eq!(
        problem(|t| t.pixel_format = metal2vulkan::metal::PixelFormat::BC1RGBA),
        Some(
            "probe: texture 0: format BC1RGBA is not one the harness knows the size of".to_owned()
        )
    );
    assert_eq!(
        problem(|t| t.bytes = Some(vec![0; 16 * 16 * 8])),
        Some(
            "probe: texture 0: its bytes are written by the CPU, which writes no private texture \
             and none the texel kernel writes"
                .to_owned()
        )
    );
    assert_eq!(
        problem(|t| {
            t.storage = Storage::Shared;
            t.bytes = Some(vec![0; 16 * 16 * 8]);
        }),
        None
    );
    assert_eq!(
        problem(|t| {
            t.storage = Storage::Managed;
            t.bytes = Some(vec![0; 16]);
        }),
        Some("probe: texture 0: it has 16 bytes, not 2048".to_owned())
    );
    // Texels move through the authored texture's kernels alone.
    assert_eq!(
        problem(|t| {
            t.pixel_format = metal2vulkan::metal::PixelFormat::RGBA8Unorm;
            t.texels = Some(vec![0.0; 4 * 256]);
        }),
        Some(
            "probe: texture 0: its texels move through the harness's texel kernels, which take a \
             2D RGBA16Float texture of ShaderRead and ShaderWrite usage that the device made"
                .to_owned()
        )
    );
    let mut expected = one_texture_plan(16, None, 0, 4);
    expected.textures[0].source = TextureSource::Heap;
    assert!(expected.validate().is_err());
}

/// A run's dispatches cut into command buffers, and a command buffer's into encoders: whole parts
/// in order, the last one shorter.
#[test]
fn the_five_buffer_frame_is_the_frame_five_times_over_with_its_handlers() {
    let plan = |name| (WORKLOADS.iter().find(|w| w.name == name).unwrap().plan)();
    let (frame, five) = (plan("frame"), plan("frame_5cb"));
    assert_eq!((frame.command_buffers(), five.command_buffers()), (1, 5));
    assert_eq!(five.dispatches.len(), 5 * frame.dispatches.len());
    // Both are frames, timed by their command buffers' completed handlers: their throughput is
    // the row the matrix judges, their latency from idle a row of its own.
    assert!(five.commit_to_complete && frame.commit_to_complete);
    assert!(five.frames && frame.frames);
    let frames_alone = Plan {
        commit_to_complete: false,
        ..frame.clone()
    };
    assert!(
        frames_alone.validate().is_err(),
        "frames need their completed handlers"
    );
}

/// A frames workload's throughput is the time from each run's last completed handler to the
/// next's, the pairs around the gauges' waits left out; its latency from idle is each run's time
/// from its first commit to its last completed handler.
#[test]
fn frames_take_their_throughput_handler_to_handler_and_their_latency_from_the_first_commit() {
    let runs = 201;
    // Two command buffers a run: the last one's handler 1000 us after the run before's, past the
    // 5 ms waits after runs 100 and 200 (GAUGE_RUNS); each run's first commit 700 us before it.
    let last: Vec<f64> = (0..runs)
        .map(|run| {
            let waits = [100, 200].iter().filter(|&&after| run >= after).count();
            1000.0 * run as f64 + 900.0 + 5000.0 * waits as f64
        })
        .collect();
    let first: Vec<f64> = last.iter().map(|end| end - 700.0).collect();
    let handled = Handled {
        committed: vec![first.clone(), first],
        handled: vec![vec![0.0; runs], last.clone()],
        late: vec![Vec::new(); 2],
    };
    let throughput = handled.throughput_us();
    assert_eq!(
        throughput.len(),
        runs - 1 - 2,
        "every pair but the two around a gauge wait"
    );
    assert!(throughput.iter().all(|&us| us == 1000.0), "{throughput:?}");
    assert!(handled.idle_latency_us().iter().all(|&us| us == 700.0));
    assert_eq!(handled.commit_us()[1], vec![700.0; runs]);
}

#[test]
fn parts_cut_a_range_in_order_the_last_one_shorter() {
    let cut =
        |range, per| (parts(range, per).map(|part| (part.start, part.end))).collect::<Vec<_>>();
    assert_eq!(cut(0..5, Some(2)), [(0, 2), (2, 4), (4, 5)]);
    assert_eq!(cut(10..20, Some(5)), [(10, 15), (15, 20)]);
    assert_eq!(cut(4..10, None), [(4, 10)]);
    assert_eq!(cut(3..4, Some(8)), [(3, 4)]);
}

/// A draw's target is a texture of the plan made with `RenderTarget` usage, and a draw binds
/// vertex buffers and fragment textures alone.
#[test]
fn a_draw_needs_a_render_target_and_binds_buffers_and_textures_alone() {
    let plan = || {
        (WORKLOADS
            .iter()
            .find(|w| w.name == "draw_2000")
            .unwrap()
            .plan)()
    };
    assert!(plan().validate().is_ok());
    let mut plain = plan();
    plain.textures[0].usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    assert!(plain.validate().is_err());
    let mut bound = plan();
    bound.dispatches[1].bindings.push(Binding::Bytes {
        index: 2,
        bytes: vec![0; 16],
    });
    assert!(bound.validate().is_err());
}

/// `draw_full_rt` draws into a target of `RenderTarget` usage alone and checks none of its texels,
/// which the harness's texel kernels cannot read.
#[test]
fn the_render_only_row_draws_into_a_target_of_render_target_usage_alone() {
    let plan = (WORKLOADS
        .iter()
        .find(|w| w.name == "draw_full_rt")
        .unwrap()
        .plan)();
    assert!(plan.validate().is_ok());
    assert_eq!(plan.textures[0].usage, TextureUsage::RENDER_TARGET);
    assert!(plan.expect_textures.is_empty());
}

/// A draw that resolves draws into a multisample target and resolves it into a 1-sample render
/// target of its format and size; a multisample texture takes no texels of a kernel or the CPU.
#[test]
fn a_resolving_draw_needs_a_multisample_target_and_a_matching_resolve_texture() {
    let plan = || {
        (WORKLOADS
            .iter()
            .find(|w| w.name == "draw_full_4x")
            .unwrap()
            .plan)()
    };
    assert!(plan().validate().is_ok());
    let mut one = plan();
    one.textures[1].sample_count = 1;
    assert!(one.validate().is_err(), "a target of one sample");
    let mut smaller = plan();
    smaller.textures[0].width = 256;
    assert!(
        smaller.validate().is_err(),
        "a resolve texture of another size"
    );
    let mut unrendered = plan();
    unrendered.textures[0].usage = TextureUsage::SHADER_READ;
    assert!(
        unrendered.validate().is_err(),
        "a resolve texture without RenderTarget usage"
    );
    let mut filled = plan();
    filled.textures[1].texels = Some(vec![0.0; 4 * 512 * 512]);
    assert!(
        filled.validate().is_err(),
        "texels written into a multisample texture"
    );
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
buffer 64 managed words 3
modify 1 16 32
texture 2 70 4 2 1 1 shared plain words 5
texture 3 20 2 2 3 3 managed heap zero
texture 2 115 16 16 1 0 private over-buffer undefined
set-buffer 0 0
threadgroup-length 0 1024
linked private f g
set-texture 0 0
set-texture 1 2
set-sampler 0 nearest clamp-to-zero pixel
set-sampler 1 linear clamp-to-edge normalized
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
    assert!(matches!(
        second.dispatches[0].bindings[1],
        Binding::ThreadgroupMemory {
            index: 0,
            length: 1024
        }
    ));
    assert_eq!(second.dispatches[0].linked, ["f", "g"]);
    assert!(first.dispatches.iter().all(|d| d.linked.is_empty()));
    assert_eq!(
        (
            second.buffers[1].storage,
            second.buffers[1].modified.clone()
        ),
        (Storage::Managed, Some(16..48))
    );
    assert!(first.buffers.iter().all(|b| b.modified.is_none()));
    let [words, array, undefined] = second.textures.as_slice() else {
        panic!("three textures, not {}", second.textures.len());
    };
    use metal2vulkan::metal::{PixelFormat, SamplerAddressMode, TextureType, TextureUsage};
    assert_eq!(
        (
            words.texture_type,
            words.pixel_format,
            words.width,
            words.height
        ),
        (TextureType::Type2D, PixelFormat::RGBA8Unorm, 4, 2)
    );
    assert_eq!(
        (words.usage, words.storage, words.source),
        (
            TextureUsage::SHADER_READ,
            Storage::Shared,
            TextureSource::Plain
        )
    );
    // Eight texels of 4 bytes: eight seeded words.
    let seeded: Vec<u8> = (crate::plan::seeded_words(5, 8).iter())
        .flat_map(|word| word.to_le_bytes())
        .collect();
    assert_eq!(words.bytes, Some(seeded));
    // Three slices of 2x2 texels of 2 bytes, zero.
    assert_eq!(
        (
            array.texture_type,
            array.array_length,
            array.storage,
            array.source
        ),
        (
            TextureType::Type2DArray,
            3,
            Storage::Managed,
            TextureSource::Heap
        )
    );
    assert_eq!(array.bytes, Some(vec![0; 24]));
    assert_eq!(
        (undefined.usage, undefined.source, &undefined.bytes),
        (TextureUsage::UNKNOWN, TextureSource::OverBuffer, &None)
    );
    let bound = &second.dispatches[0].bindings;
    assert!(matches!(
        bound[2..4],
        [
            Binding::Texture {
                index: 0,
                texture: 0
            },
            Binding::Texture {
                index: 1,
                texture: 2
            }
        ]
    ));
    let pixel = SamplerPlan {
        filter: Filter::Nearest,
        address: SamplerAddressMode::ClampToZero,
        normalized: false,
    };
    assert!(matches!(bound[4], Binding::Sampler { index: 0, sampler } if sampler == pixel));
    assert!(matches!(
        bound[5],
        Binding::Sampler { index: 1, sampler } if sampler == SamplerPlan::clamped(Filter::Linear)
    ));
    assert!(first.textures.is_empty());
    assert_eq!(
        SamplerPlan::clamped(Filter::Linear)
            .descriptor()
            .r_address_mode,
        SamplerAddressMode::ClampToEdge
    );
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
    // Two RGBA8Unorm textures and an R32Uint one, after the workload's first two lines.
    let textures = "texture 2 70 8 8 1 1 shared plain zero\n\
                    texture 3 70 8 8 2 1 shared plain zero\n\
                    texture 2 53 8 8 1 1 shared plain zero\n";
    let copy = |line: &str| refusal(&format!("{textures}{line}\n{dispatch}"));
    assert_eq!(
        copy("copy-texture 0 0 0 0 4 4 3 0 0 0"),
        "line 6: texture 3 is not made before it"
    );
    for (line, why) in [
        ("texture-view 3 2 0 1", "texture 3 is not made before it"),
        (
            "texture-view 1 2 1 2",
            "slices 1..3 are empty or leave texture 1's 2",
        ),
        (
            "texture-view 1 2 0 0",
            "slices 0..0 are empty or leave texture 1's 2",
        ),
        (
            "texture-view 1 7 0 1",
            "the view: a Type3D texture of 1 slices and 1 samples: the harness makes 2D textures \
             and 2D arrays of one sample, and 2D multisample textures of more",
        ),
        (
            "texture-view 1 2",
            "texture-view takes PARENT TYPE SLICE SLICES",
        ),
    ] {
        assert_eq!(copy(line), format!("line 6: {why}"), "{line}");
    }
    assert_eq!(
        copy("copy-texture 0 0 0 0 4 4 1 2 0 0"),
        "line 6: texture 1 has 2 slices, not slice 2"
    );
    assert_eq!(
        copy("copy-texture 0 0 6 0 4 4 1 0 0 0"),
        "line 6: the region at [6, 0] of [4, 4] is outside texture 0"
    );
    assert_eq!(
        copy("copy-texture 0 0 0 0 4 4 2 0 0 0"),
        "line 6: a copy from RGBA8Unorm to R32Uint: Metal copies between textures of one pixel \
         format"
    );
    assert_eq!(
        copy("copy-texture 0 0 0 0 4 4 1"),
        "line 6: copy-texture takes SOURCE SLICE X Y WIDTH HEIGHT DESTINATION SLICE X Y"
    );
    assert_eq!(
        refusal(&format!(
            "{textures}set-buffer 0 0\ncopy-texture 0 0 0 0 4 4 1 1 0 0\n{dispatch}"
        )),
        "line 7: a blit ends the compute encoder, so it goes before the set lines of its dispatch"
    );
    assert_eq!(
        refusal(&format!(
            "{textures}{dispatch}\ncopy-texture 0 0 0 0 4 4 1 1 0 0"
        )),
        "workload w blits after its last dispatch"
    );
    // Buffer 1 of 256 Shared bytes, then the blit on line 7.
    let blit = |line: &str| {
        refusal(&format!(
            "{textures}buffer 256 shared zero\n{line}\n{dispatch}"
        ))
    };
    let refusals = [
        (
            "copy-buffer-to-texture 2 0 32 0 4 4 0 0 0 0",
            "buffer 2 is not made before it",
        ),
        (
            "copy-buffer-to-texture 1 0 32 0 4 4 9 0 0 0",
            "texture 9 is not made before it",
        ),
        (
            "copy-buffer-to-texture 1 0 32 0 0 4 0 0 0 0",
            "the region is empty",
        ),
        (
            "copy-buffer-to-texture 1 2 32 0 4 4 0 0 0 0",
            "an offset of 2 and rows 32 bytes apart are not whole rows of 16 bytes of RGBA8Unorm \
             texels",
        ),
        (
            "copy-texture-to-buffer 0 0 0 0 4 4 1 0 12 0",
            "an offset of 0 and rows 12 bytes apart are not whole rows of 16 bytes of RGBA8Unorm \
             texels",
        ),
        (
            "copy-texture-to-buffer 0 0 0 0 4 4 1 200 32 0",
            "the region's bytes end at 312, past buffer 1's 256",
        ),
        (
            "copy-buffer-to-texture 1 0 32",
            "copy-buffer-to-texture takes BUFFER OFFSET ROW-BYTES IMAGE-BYTES WIDTH HEIGHT TEXTURE \
             SLICE X Y",
        ),
        (
            "copy-texture-to-buffer 0 0 0 0",
            "copy-texture-to-buffer takes TEXTURE SLICE X Y WIDTH HEIGHT BUFFER OFFSET ROW-BYTES \
             IMAGE-BYTES",
        ),
        (
            "fill-buffer 1 250 8 0",
            "a fill of 250..258 is empty or past buffer 1's 256 bytes",
        ),
        (
            "fill-buffer 1 0 0 0",
            "a fill of 0..0 is empty or past buffer 1's 256 bytes",
        ),
        ("fill-buffer 1 0 4 256", "256 is not a whole number"),
        (
            "fill-buffer 1 0",
            "fill-buffer takes BUFFER OFFSET LENGTH VALUE",
        ),
        ("synchronize 2", "buffer 2 is not made before it"),
        (
            "synchronize 1",
            "buffer 1 is not managed, and Metal's validation stops a synchronize of it",
        ),
        ("synchronize", "synchronize takes BUFFER"),
    ];
    for (line, why) in refusals {
        assert_eq!(blit(line), format!("line 7: {why}"), "{line}");
    }
    assert_eq!(
        refused("buffer 16 private zero"),
        "line 1: buffer before a workload"
    );
    assert_eq!(
        refused("workload w"),
        "line 1: workload takes NAME METALLIB"
    );
    assert_eq!(
        refusal("buffer 16 memoryless zero"),
        "line 3: storage memoryless is not shared, managed or private"
    );
    assert_eq!(
        refusal("modify 0 0 16"),
        "line 3: buffer 0: the CPU writes no private buffer"
    );
    assert_eq!(
        refusal("modify 1 0 16"),
        "line 3: modify names buffer 1, which no line before it makes"
    );
    assert_eq!(
        refusal("buffer 32 managed zero\nmodify 1 16 32"),
        "line 4: buffer 1: its modified range 16..48 is not within its 32 bytes"
    );
    assert_eq!(
        refusal("modify 0 0"),
        "line 3: modify takes BUFFER OFFSET LENGTH"
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
    assert_eq!(
        refusal("threadgroup-length 0 1000"),
        "line 3: threadgroup length 1000 is not a multiple of 16"
    );
    assert_eq!(
        refusal("threadgroup-length 0"),
        "line 3: threadgroup-length takes INDEX LENGTH"
    );
    assert_eq!(
        refusal("linked binary f"),
        "line 3: linked takes private FUNCTION..."
    );
    assert_eq!(
        refusal("linked private"),
        "line 3: linked takes private FUNCTION..."
    );
    assert_eq!(
        refusal("texture 2 70 4 4 1 1 private plain zero"),
        "line 3: a private texture takes undefined alone: the CPU writes none"
    );
    assert_eq!(
        refusal("texture 2 70 4 4 1 1 shared plain noise"),
        "line 3: fill noise is not undefined, zero or words SEED"
    );
    assert_eq!(
        refusal("texture 2 70 4 4 1 1 memoryless plain undefined"),
        "line 3: storage memoryless is not shared, managed or private"
    );
    assert_eq!(
        refusal("texture 2 70 4 4 1 1 shared iosurface undefined"),
        "line 3: source iosurface is not plain, heap or over-buffer"
    );
    assert_eq!(
        refusal("texture 3 70 4 4 2 1 shared over-buffer undefined"),
        "line 3: the texture: a texture over a buffer is 2D, as Metal requires"
    );
    assert_eq!(
        refusal("texture 7 70 4 4 1 1 shared plain undefined"),
        "line 3: the texture: a Type3D texture of 1 slices and 1 samples: the harness makes 2D \
         textures and 2D arrays of one sample, and 2D multisample textures of more"
    );
    assert_eq!(
        refusal("texture 2 70 4 0 1 1 shared plain undefined"),
        "line 3: the texture: its size 4x0x1 is empty"
    );
    assert!(refusal("texture 2 9999 4 4 1 1 shared plain undefined").starts_with("line 3: "));
    assert_eq!(
        refusal("texture 2 70 4 4 1 1 shared plain"),
        "line 3: texture takes TYPE FORMAT WIDTH HEIGHT ARRAY USAGE STORAGE SOURCE FILL"
    );
    assert_eq!(
        refusal("set-texture 0"),
        "line 3: set-texture takes INDEX TEXTURE"
    );
    assert_eq!(
        refusal("set-sampler 0 cubic clamp-to-edge normalized"),
        "line 3: filter cubic is neither nearest nor linear"
    );
    assert_eq!(
        refusal("set-sampler 0 linear repeat normalized"),
        "line 3: address mode repeat is neither clamp-to-edge nor clamp-to-zero"
    );
    assert_eq!(
        refusal("set-sampler 0 linear clamp-to-edge texel"),
        "line 3: coordinates texel are neither normalized nor pixel"
    );
    assert_eq!(
        refusal("set-sampler 0 linear"),
        "line 3: set-sampler takes INDEX FILTER ADDRESS COORDS"
    );
    assert_eq!(
        refusal("set-texture 0 0\ndispatch k threadgroups 1 1 1 16 1 1"),
        "w: dispatch 0 binds texture 0, which the plan lacks"
    );
    assert_eq!(refusal("set-buffer 0 0"), "workload w has no dispatch");
    assert_eq!(
        refusal(&format!("{dispatch}\nset-buffer 1 0")),
        "workload w sets a binding, a constant or a linked function after its last dispatch"
    );
    assert_eq!(
        refusal(&format!("{dispatch}\nconstant 0 53 01")),
        "workload w sets a binding, a constant or a linked function after its last dispatch"
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

/// A host buffer that counts the live ones of its device.
struct Counted {
    words: Vec<u32>,
    live: Rc<Cell<usize>>,
}

impl Buffer for Counted {
    fn contents(&self) -> Option<NonNull<u8>> {
        NonNull::new(self.words.as_ptr().cast_mut().cast())
    }

    fn did_modify_range(&self, _: u64, _: u64) -> Result<()> {
        Ok(())
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        self.live.set(self.live.get() - 1);
    }
}

/// A device of host buffers that keeps, at each creation, how many of its buffers lived.
#[derive(Default)]
struct HostDevice {
    live: Rc<Cell<usize>>,
    beside: RefCell<Vec<usize>>,
}

impl Device for HostDevice {
    type Buffer = Counted;

    fn name(&self) -> String {
        "host".to_owned()
    }

    fn new_buffer(&self, length: u64, _: Storage) -> Result<Counted> {
        self.beside.borrow_mut().push(self.live.get());
        self.live.set(self.live.get() + 1);
        Ok(Counted {
            words: vec![0; length.div_ceil(4) as usize],
            live: Rc::clone(&self.live),
        })
    }

    fn gauges(&self) -> Vec<(&'static str, u64)> {
        Vec::new()
    }

    fn gauges_after_drop(&self) -> Option<GaugeReader> {
        None
    }

    fn totals(&self) -> Vec<(&'static str, u64)> {
        Vec::new()
    }

    fn threads(&self) -> Vec<u64> {
        Vec::new()
    }

    fn pipelines_created(&self) -> u64 {
        0
    }

    fn counts(&self) -> Vec<(&'static str, u64)> {
        Vec::new()
    }

    fn submissions(&self) -> u64 {
        0
    }

    fn submit_infos(&self) -> Option<u64> {
        None
    }

    fn first_use_waits(&self) -> Option<Waits> {
        None
    }

    fn allocations(&self) -> Option<Waits> {
        None
    }

    fn own_row(&self, _: OwnRow) -> Option<Result<OwnReport>> {
        None
    }
}

#[test]
fn a_buffer_row_s_first_is_made_beside_its_kept_warm_up() {
    let device = HostDevice::default();
    let runs = Runs {
        warmup: 0,
        settle: false,
        timed: 2,
    };
    let report = create(&device, 4096, runs).unwrap();
    // The warm-up alone; the first, then the next after its release, each beside the warm-up;
    // the timed ones alone once the warm-up is released.
    assert_eq!(*device.beside.borrow(), [0, 1, 1, 0, 0]);
    assert_eq!(device.live.get(), 0);
    assert!(report.device_first_us.is_some());
    assert!(report.first_create_us.is_some() && report.again_us.is_some());
    assert_eq!(report.create_us.len(), 2);
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
fn a_steady_time_is_the_lowest_mean_of_windows_of_ten() {
    assert_eq!(steady(&[1.0; 4]), None, "fewer than 5 runs");
    assert_eq!(
        steady(&[3.0, 1.0, 2.0, 5.0, 4.0]),
        Some(3.0),
        "their mean below 10"
    );
    // A clock that drops partway through: the faster windows' level, not the mix.
    let mut times = vec![100.0; 20];
    times.extend([105.0; 30]);
    times[3] = 400.0;
    assert_eq!(steady(&times), Some(100.0));
    assert_eq!(percentile(&times, 50), Some(105.0));
    // Two runs' completions reported together: each pair's first wait takes both runs' GPU time and
    // its second none (grouped submissions on an Iris Xe, 4367 us of GPU a run). The time a run
    // took, not the short wait's: a median of each window of 10 is 174.
    let paired: Vec<f64> = (0..200)
        .map(|run| if run % 2 == 0 { 9454.0 } else { 174.0 })
        .collect();
    assert_eq!(steady(&paired), Some(4814.0));
    assert_eq!(percentile(&paired[..10], 50), Some(174.0));
}

#[test]
fn a_row_is_judged_by_host_time_never_its_gpu_time() {
    // Twenty runs: GPU times of 10 us, host wall times of 50 us.
    let mut report = Report {
        gpu_us: vec![10.0; 20],
        wall_us: vec![50.0; 20],
        ..Report::default()
    };
    assert_eq!(matrix_time(&report), Some(50.0), "the wall time a run adds");
    // Where the plan times its command buffers' commit to completion, each run's last command
    // buffer's: 100, 200 and 300 us at the three places of every run.
    report.commit_us = vec![vec![100.0; 20], vec![200.0; 20], vec![300.0; 20]];
    assert_eq!(
        matrix_time(&report),
        Some(300.0),
        "the last command buffer's commit-to-complete time"
    );
    // The earlier command buffers' times decide nothing: they are late where a later one's
    // completion reports them, which the lateness column shows.
    report.commit_us[0].fill(290.0);
    assert_eq!(matrix_time(&report), Some(300.0), "the last place alone");
    // A slow window of runs leaves the steady time at the faster windows' level.
    report.commit_us[2][..10].fill(400.0);
    assert_eq!(
        matrix_time(&report),
        Some(300.0),
        "the steady time of the last place"
    );
    // A frames workload's row is its throughput, however its command buffers' times read.
    report.throughput_us = vec![250.0; 20];
    assert_eq!(matrix_time(&report), Some(250.0), "the frames' throughput");
    report.throughput_us.clear();
    report.wall_us.clear();
    report.commit_us.clear();
    assert_eq!(matrix_time(&report), None, "no host time, no matrix time");
}

/// The new runtime's known gaps and the device's limits are typed refusals; its other errors are
/// failures.
#[test]
fn the_new_runtime_refuses_by_its_gaps_and_limits_alone() {
    let refused = |error: RtError| rt::error(error).downcast::<Refused>().is_ok();
    assert!(refused(RtError::Unsupported(Gap::KernelInput {
        kind: KernelInputKind::StageIn
    })));
    assert!(refused(RtError::Limit(Limit::BlockIds)));
    assert!(!refused(RtError::Quiesced));
    assert_eq!(rt::size([4, 2, 1]).unwrap(), [4, 2, 1]);
    assert!(rt::size([1 << 32, 1, 1]).unwrap_err().is::<Refused>());
}

/// A session that answers each plan with a report of `outputs`, or with `error`; its one run
/// takes a tenth of a microsecond of GPU time a byte of the plan's buffers.
struct Answers {
    outputs: Vec<u8>,
    error: Option<fn() -> crate::Error>,
}

impl Session for Answers {
    fn device_name(&self) -> String {
        "answers".to_owned()
    }

    fn run(&mut self, plan: &Plan, _: Runs) -> Result<Report> {
        match self.error {
            Some(error) => Err(error()),
            None => Ok(Report {
                outputs: vec![self.outputs.clone()],
                gpu_us: vec![
                    plan.buffers.iter().map(|buffer| buffer.length).sum::<u64>() as f64 / 10.0,
                ],
                ..Report::default()
            }),
        }
    }

    fn create(&mut self, _: Makes, _: Runs) -> Result<CreateReport> {
        Ok(CreateReport::default())
    }

    fn finish(self: Box<Self>) -> Option<Vec<(&'static str, u64)>> {
        None
    }

    fn totals(&self) -> Vec<(&'static str, u64)> {
        Vec::new()
    }

    fn pipelines_created(&self) -> u64 {
        0
    }

    fn first_use_waits(&self) -> Option<Waits> {
        None
    }

    fn own_row(&mut self, _: OwnRow) -> Option<Result<OwnReport>> {
        None
    }
}

/// A scaled workload's size is its base doubled toward its target, by the time at the base.
#[test]
fn a_scaled_workload_doubles_its_base_toward_its_target() {
    let scaled = Scaled {
        build: |_| one_buffer_plan(16, None, 0),
        base: 1024,
        target_us: 10_000.0,
    };
    // A command buffer of 80 us at the base: 125 times short, 2^7 the nearest power of two.
    assert_eq!(scaled.size(80.0), 1024 << 7);
    // 3,640 us: 2.75 times short, nearer 2 than 4.
    assert_eq!(scaled.size(3640.0), 2048);
    // Longer than the target at the base, or no time measured: the base. At most ten doublings.
    assert_eq!(scaled.size(12_450.0), 1024);
    assert_eq!(scaled.size(0.0), 1024);
    assert_eq!(scaled.size(0.5), 1024 << Scaled::MOST_DOUBLINGS);
    let mut plan = one_buffer_plan(16, None, 0);
    plan.scaled = Some(scaled);
    assert!(plan.validate().is_ok());
    plan.outputs = vec![Output {
        buffer: 0,
        comparison: Comparison::Exact,
    }];
    assert!(plan.validate().is_err());
}

/// A scaled workload is timed at its base on the first runtime, then every runtime runs it as
/// its builder makes it at the size that time gives.
#[test]
fn a_scaled_workload_runs_at_the_size_its_base_time_gives() {
    let runs = Runs {
        warmup: 0,
        settle: false,
        timed: 1,
    };
    // A buffer of `size` bytes: 102.4 us at the base of 1,024, so 2^7 times that is nearest
    // 10,000 us.
    let build = |size| one_buffer_plan(u64::from(size), None, 0);
    let mut plan = build(1024);
    plan.scaled = Some(Scaled {
        build,
        base: 1024,
        target_us: 10_000.0,
    });
    let answers = |error: Option<fn() -> crate::Error>| -> Box<dyn Session> {
        Box::new(Answers {
            outputs: Vec::new(),
            error,
        })
    };
    let refuse: fn() -> crate::Error = || Box::new(Refused("a known gap".to_owned()));
    // The first refuses, so the second sizes it, and the third runs it at that size.
    let mut sessions = vec![
        ("a", answers(Some(refuse))),
        ("b", answers(None)),
        ("c", answers(None)),
    ];
    let each = run_each(&plan, &mut sessions, runs, None).unwrap();
    assert!(each[0].is_err());
    for report in &each[1..] {
        let report = report.as_ref().unwrap();
        let scaled = report.scaled_to.unwrap();
        assert_eq!(
            (scaled.base, scaled.base_us, scaled.size),
            (1024, Some(102.4), 1024 << 7)
        );
        // The report is the run at that size.
        assert_eq!(report.gpu_us, [13_107.2]);
    }
}

/// A scaled workload given its size runs at that size on every runtime, timing no base, so a
/// runtime timed in another process runs the work the first one sized; a size given for a
/// workload of one size is an error.
#[test]
fn a_scaled_workload_given_its_size_runs_at_it_and_times_no_base() {
    let runs = Runs {
        warmup: 0,
        settle: false,
        timed: 1,
    };
    let build = |size| one_buffer_plan(u64::from(size), None, 0);
    let mut plan = build(1024);
    plan.scaled = Some(Scaled {
        build,
        base: 1024,
        target_us: 10_000.0,
    });
    let answers = || -> Box<dyn Session> {
        Box::new(Answers {
            outputs: Vec::new(),
            error: None,
        })
    };
    let mut sessions = vec![("a", answers()), ("b", answers())];
    // Its base time would size it to 2^7 times the base; the size given is 2^3 times.
    let each = run_each(&plan, &mut sessions, runs, Some(1024 << 3)).unwrap();
    for report in &each {
        let report = report.as_ref().unwrap();
        let scaled = report.scaled_to.unwrap();
        assert_eq!(
            (scaled.base, scaled.base_us, scaled.size),
            (1024, None, 1024 << 3)
        );
        assert_eq!(report.gpu_us, [819.2]);
    }
    let one_size = one_buffer_plan(16, None, 0);
    assert!(run_each(&one_size, &mut sessions, runs, Some(1024)).is_err());
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
    let each = run_each(&plan, &mut sessions, runs, None).unwrap();
    assert!(each[0].is_ok() && each[2].is_ok());
    assert_eq!(each[1].as_ref().unwrap_err().to_string(), "a known gap");
    sessions[2].1 = answers(&[1, 2, 3, 5], None);
    assert!(run_each(&plan, &mut sessions, runs, None).is_err());
    // Outputs that differ name no runtime.
    let differ = run_each(&plan, &mut sessions, runs, None).unwrap_err();
    assert!(differ.downcast_ref::<Failed>().is_none());
    sessions[2].1 = answers(&[], Some(fail));
    let failed = run_each(&plan, &mut sessions, runs, None).unwrap_err();
    let failed = failed.downcast_ref::<Failed>().unwrap();
    assert_eq!(
        (failed.runtime.as_str(), failed.why.to_string().as_str()),
        ("c", "a failure")
    );
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

/// The fixture's two cells kernels on lavapipe's `Device::from_queries` profile. Under the
/// resident-table domains both observe their threadgroup partition with
/// partial threadgroups through the lockstep order's barriers, which stand past the bounds test,
/// so they are SIMD-group barriers, as in the row-indexed kernels the fixture stands for. The
/// one-row variant (`LoweringInputs::one_row`) of `row_cells`, whose `int` cell arithmetic
/// compiles to `nsw` steps, does not observe it, so a runtime may run its grids in tiles; that of
/// `row_cells_uint`, the control, whose `uint` steps carry no flags, still does. Under the
/// identity domains (the new runtime's), whose kernels take no lockstep order, neither observes
/// it, plain or with partial threadgroups.
#[test]
fn the_int_cells_kernel_is_apart_in_rows_and_its_uint_control_is_not() {
    use metal2vulkan::device::{
        Device, DeviceDescription, DevicePolicy, HandleDomain, PointerDomain,
    };
    use metal2vulkan::library::Library;
    use metal2vulkan::pipeline::{FunctionDescriptor, LoweringInputs, StageEntries};
    let air = match crate::authored::library_air() {
        Ok(air) => air,
        Err(why) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
    };
    let json = include_str!("../../metal2vulkan-validation/profiles/lavapipe.json");
    let description = DeviceDescription::from_khronos_json(json, None).unwrap();
    let resident = Device::from_queries(description.clone(), DevicePolicy::default()).unwrap();
    let mut policy = DevicePolicy::default();
    policy.pointer_domain = PointerDomain::Identity;
    policy.handle_domain = HandleDomain::Identity;
    let identity = Device::from_queries(description, policy).unwrap();
    let library = Library::from_air(&air).unwrap();
    let budget = metal2vulkan::Budget::standard();
    let mut partial = LoweringInputs::new();
    partial.partial_threadgroups = true;
    let mut rows = partial.clone();
    rows.one_row = true;
    for (name, apart) in [("row_cells", true), ("row_cells_uint", false)] {
        let function = FunctionDescriptor::new(library.function(name).unwrap());
        let observes = |device: &Device, inputs: &LoweringInputs| {
            let entries = StageEntries::new();
            (device.kernel_observes_partition(&function, inputs, &budget, &entries))
                .unwrap_or_else(|why| panic!("{name}: {why}"))
        };
        let plain = (resident.compile_function(&function, &LoweringInputs::new(), &budget))
            .unwrap_or_else(|why| panic!("{name}: {why}"));
        assert!(
            plain.observes_partition_through_simdgroups_alone(),
            "{name}"
        );
        assert!(observes(&resident, &partial), "{name}: the partial variant");
        assert_eq!(
            observes(&resident, &rows),
            !apart,
            "{name}: the one-row variant"
        );
        let own = (identity.compile_function(&function, &LoweringInputs::new(), &budget))
            .unwrap_or_else(|why| panic!("{name}: {why}"));
        assert!(
            !own.observes_threadgroup_partition(),
            "{name}: under the identity domains"
        );
        assert!(
            !observes(&identity, &partial),
            "{name}: the partial variant under the identity domains"
        );
    }
}
