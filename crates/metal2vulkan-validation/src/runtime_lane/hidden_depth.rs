//! Draws whose depth/stencil state uses depth or the stencil their pass has no attachment for on
//! the device the cfg selects, under the validation layer. Metal's validation stops each; plain
//! Metal tests it against a hidden `float` depth, 1.0 at the pass's start, and a hidden stencil of
//! 0, kept through the pass (Metal probes on an M4 with macOS 27.0.1), which the runtime gives the
//! pass as an attachment of its own. The probe's functions (the owned fixture `rt6_render_state`)
//! draw fullscreen triangles into its 4x4 `RGBA8Unorm` target cleared to zero, white unless a case
//! names `rt6_half`.
//!
//! Every answer below is plain Metal's, and each pass stores what the same pass stores with a
//! `Depth32Float` attachment cleared to 1.0 and a `Stencil8` one cleared to 0 (checked):
//!
//! - Each compare at depth 0.5, with writes for `Never` and `Always`: `Less`, `LessEqual`,
//!   `NotEqual` and `Always` draw; `Equal`, `Greater`, `GreaterEqual` and `Never` draw nothing (a
//!   Metal probe).
//! - A write at 0.25, then `Equal` at 0.25, draws (probe 2 `dwa_write_then_equal`); a write at
//!   0.5, then `Equal` at the next `float` above 0.5, draws nothing (probe 3 `dwa_eq_2m24`).
//! - A pass after one that wrote 0.25 starts at 1.0 again: `Less` at 0.5 draws (probe 2
//!   `dwa_across_passes`).
//! - Through a viewport depth range of `[0, 0.5]` depth 1.0 lands at 0.5: `LessEqual` draws,
//!   `Equal` and `GreaterEqual` draw nothing (probe 2 `dwa_range_*`).
//! - The stencil: a compare of `Never` draws nothing, `Always` with `Replace` draws (a Metal
//!   probe); `Replace` of 5 under a small triangle, then `Equal` to 5, draws that triangle alone.
//! - A draw that needs them after a draw that did not (the pass splits) keeps the earlier draw's
//!   texels, and tests against 1.0.
//! - A first draw that needs them takes the place of the pass's first instance before anything
//!   recorded into it: that instance's draws recording ends, and its pool goes back to the
//!   device's spares unreset.
//!
//! A pass with a depth attachment of its own whose draw uses the stencil it lacks, and the other
//! way round, stay refused `NotImplemented(HiddenDepthStencil)` (not built, unmeasured).
//!
//! An indirect command buffer's commands executed in such a pass draw as the encoder's own draws at
//! the call do (an M4 with macOS 27.0.1, plain and validated; an M3 Max, the commands alone):
//! against the hidden depth and stencil under the encoder's state at the call, shared with the
//! encoder's draws before and after them, each command's write seen by
//! the next. Metal's validation stops the encoder's draws among these, not the commands.
//!
//! The pipeline's depth and stencil formats decide nothing (a Metal probe, an M4 with macOS 27.0.1,
//! where Metal's validation stops each at `setRenderPipelineState:`):
//!
//! - A pipeline naming `Depth16Unorm` and `Stencil8` in a pass without them tests the hidden
//!   `float` depth and the hidden stencil: a write at 0.5, then `Equal` at the next `float`
//!   above, draws nothing (`pipe_depth16_extra`), and the stencil draws as without them
//!   (`pipe_stencil_extra`).
//! - A pass `Depth16Unorm` attachment cleared to 0.5 is tested and written by `Less` with writes
//!   at 0.25 whether the pipeline names no depth, `Depth16Unorm` or `Depth32Float`; Metal drew
//!   every texel and stored `0x4000` (`pass_depth16_extra`, `depth_mismatch`). A pass `Stencil8`
//!   attachment is written by `Replace` of 7 though the pipeline names no stencil; Metal drew
//!   every texel and stored 7 (`pass_stencil_extra_write`).

use super::render_state::{rows, stencil, validated, Pipe, Probe, EMPTY, FULL, PASS, WHITE};
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType, StencilOperation};
use metal2vulkan_runtime::{Buffer, Error, RenderCommandEncoder, RuntimeGap, Viewport};

/// A body of draws: the probe, the encoder, the depth buffers it keeps, and what makes a pipe
/// fit the pass (the formats of the attachments the comparing pass has).
type Body<'a> =
    &'a dyn Fn(&Probe, &mut RenderCommandEncoder, &mut Vec<Buffer>, &dyn Fn(Pipe) -> Pipe);

/// `body`'s draws in a pass without a depth/stencil attachment: the texels it stored, after
/// checking they are those of the same draws in a pass with a `Depth32Float` attachment cleared
/// to 1.0 and a `Stencil8` one cleared to 0.
fn hidden(probe: &Probe, case: &str, body: Body<'_>) -> Vec<u8> {
    let without = probe.run(PASS, |p, e, h| body(p, e, h, &|pipe| pipe));
    let with = probe.run(
        PASS.depth(PixelFormat::Depth32Float, 1.0).stencil(0),
        |p, e, h| {
            body(p, e, h, &|pipe| {
                pipe.depth(PixelFormat::Depth32Float).stencil()
            })
        },
    );
    assert_eq!(
        without.color, with.color,
        "{case}: the hidden depth and stencil against attachments"
    );
    without.color
}

#[test]
fn a_draw_without_its_depth_or_stencil_attachment_tests_metal_s_hidden_ones() {
    use CompareFunction::{Always, Equal, Greater, GreaterEqual, Less, LessEqual, Never, NotEqual};
    use StencilOperation::{Keep, Replace};
    let (messages, _cache, probe) = validated();
    let name = probe.device.name().to_owned();
    // Each compare against 1.0, at 0.5.
    for (compare, write, drawn) in [
        (Never, true, false),
        (Always, true, true),
        (Less, false, true),
        (LessEqual, false, true),
        (Equal, false, false),
        (Greater, false, false),
        (GreaterEqual, false, false),
        (NotEqual, false, true),
    ] {
        let texels = hidden(&probe, &format!("{compare:?} {write}"), &|p, e, h, fit| {
            e.set_depth_stencil_state(&p.state(compare, write, None))
                .unwrap();
            p.draw(e, h, fit(WHITE), 0.5, 0.0);
        });
        let expected = rows(if drawn { &FULL } else { &EMPTY });
        assert_eq!(texels, expected, "{compare:?} {write} on {name}");
    }
    // The depth a write leaves, compared exactly as a `float`.
    let half = |fit: &dyn Fn(Pipe) -> Pipe| fit(WHITE.fragment("rt6_half"));
    let texels = hidden(&probe, "a write, then Equal", &|p, e, h, fit| {
        e.set_depth_stencil_state(&p.state(Always, true, None))
            .unwrap();
        p.draw(e, h, fit(WHITE), 0.25, 0.0);
        e.set_depth_stencil_state(&p.state(Equal, false, None))
            .unwrap();
        p.draw(e, h, half(fit), 0.25, 0.0);
    });
    assert_eq!(
        texels,
        [0x80; 4].repeat(16),
        "a write, then Equal on {name}"
    );
    let above = f32::from_bits(0.5f32.to_bits() + 1);
    let texels = hidden(&probe, "a write, then Equal above", &|p, e, h, fit| {
        e.set_depth_stencil_state(&p.state(Always, true, None))
            .unwrap();
        p.draw(e, h, fit(WHITE), 0.5, 0.0);
        e.set_depth_stencil_state(&p.state(Equal, false, None))
            .unwrap();
        p.draw(e, h, half(fit), above, 0.0);
    });
    assert_eq!(texels, rows(&FULL), "a write, then Equal above on {name}");
    // A new pass starts at 1.0.
    hidden(&probe, "the pass before", &|p, e, h, fit| {
        e.set_depth_stencil_state(&p.state(Always, true, None))
            .unwrap();
        p.draw(e, h, fit(WHITE), 0.25, 0.0);
    });
    let texels = hidden(&probe, "the pass after", &|p, e, h, fit| {
        e.set_depth_stencil_state(&p.state(Less, false, None))
            .unwrap();
        p.draw(e, h, fit(WHITE), 0.5, 0.0);
    });
    assert_eq!(texels, rows(&FULL), "the pass after on {name}");
    // Through a viewport depth range of [0, 0.5], depth 1.0 at 0.5.
    for (compare, drawn) in [(LessEqual, true), (Equal, false), (GreaterEqual, false)] {
        let texels = hidden(&probe, &format!("range {compare:?}"), &|p, e, h, fit| {
            e.set_viewport(Viewport::new(0.0, 0.0, 4.0, 4.0, 0.0, 0.5))
                .unwrap();
            e.set_depth_stencil_state(&p.state(compare, false, None))
                .unwrap();
            p.draw(e, h, fit(WHITE), 1.0, 0.0);
        });
        let expected = rows(if drawn { &FULL } else { &EMPTY });
        assert_eq!(texels, expected, "range {compare:?} on {name}");
    }
    // The stencil, 0 at the pass's start.
    for (compare, pass, drawn) in [(Never, Keep, false), (Always, Replace, true)] {
        let face = stencil(compare, pass, 0xff, 0xff);
        let texels = hidden(&probe, &format!("stencil {compare:?}"), &|p, e, h, fit| {
            e.set_depth_stencil_state(&p.state(Always, false, Some(face)))
                .unwrap();
            p.draw(e, h, fit(WHITE), 0.5, 0.0);
        });
        let expected = rows(if drawn { &FULL } else { &EMPTY });
        assert_eq!(texels, expected, "stencil {compare:?} on {name}");
    }
    let small = hidden(&probe, "a replaced stencil, then Equal", &|p, e, h, fit| {
        e.set_stencil_reference_value(5);
        let replace = stencil(Always, Replace, 0xff, 0xff);
        e.set_depth_stencil_state(&p.state(Always, false, Some(replace)))
            .unwrap();
        p.draw(e, h, half(&|pipe| fit(pipe.vertex("rt6_small"))), 0.5, 0.0);
        let equal = stencil(Equal, Keep, 0xff, 0xff);
        e.set_depth_stencil_state(&p.state(Always, false, Some(equal)))
            .unwrap();
        p.draw(e, h, fit(WHITE), 0.5, 0.0);
    });
    let drawn = small.chunks(4).filter(|t| *t == [0xff; 4]).count();
    let rest = small.chunks(4).filter(|t| *t == [0; 4]).count();
    assert!(
        drawn > 0 && drawn + rest == 16,
        "a replaced stencil, then Equal on {name}: {small:02x?}"
    );
    // Arriving after a draw without them: the pass splits, and the earlier draw's texels stay.
    let texels = hidden(&probe, "after a draw without them", &|p, e, h, fit| {
        p.draw(e, h, fit(WHITE.fragment("rt6_half")), 0.5, 0.0);
        e.set_depth_stencil_state(&p.state(Less, true, None))
            .unwrap();
        p.draw(e, h, fit(WHITE.vertex("rt6_small")), 0.25, 0.0);
        e.set_depth_stencil_state(&p.state(Less, false, None))
            .unwrap();
        p.draw(e, h, fit(WHITE.fragment("rt6_half")), 0.5, 0.0);
    });
    let whites = texels.chunks(4).filter(|t| *t == [0xff; 4]).count();
    let halves = texels.chunks(4).filter(|t| *t == [0x80; 4]).count();
    assert!(
        whites > 0 && whites + halves == 16,
        "after a draw without them on {name}: {texels:02x?}"
    );
    eprintln!("draws against the hidden depth and stencil on {name}: Metal's");
    super::render_state::quiet(probe, &messages);
}

/// A pass whose first draw tests its hidden depth runs from that draw on in an instance with the
/// hidden attachment, which takes the place of the pass's first instance before anything recorded
/// into it. Where that first instance began its draws' secondary as the pass opened (a pass with
/// a visibility result buffer, whose first instance never records inline), the runtime ends it
/// unused (`WorkCounts::unrun_instances_ended`); where the first instance may record inline (the
/// probe's color-only pass), it began none and none ends (`render/inline_first.rs`). Either way
/// every command buffer the pass's pool handed out ended, the pool goes back to the device's
/// spares unreset, and no spare is reset as it is taken again
/// (`WorkCounts::spare_command_pools_reset`).
#[test]
fn a_first_draw_s_instance_ends_the_one_it_replaces() {
    const PASSES: u64 = 4;
    let (messages, _cache, probe) = validated();
    let results = super::buffer_with(&probe.device, &[0; 8]).unwrap();
    for (visibility, begun) in [(None, 0), (Some(&results), PASSES)] {
        let before = probe.device.work_counts();
        for _ in 0..PASSES {
            let stored = probe
                .try_run_with(PASS, visibility, |p, e, h| {
                    e.set_depth_stencil_state(&p.state(CompareFunction::Less, false, None))
                        .unwrap();
                    p.draw(e, h, WHITE, 0.5, 0.0);
                })
                .unwrap();
            assert_eq!(stored.color, rows(&FULL), "Less at 0.5 against 1.0");
        }
        let after = probe.device.work_counts();
        let ended = after.unrun_instances_ended - before.unrun_instances_ended;
        let taken = after.spare_command_pools_taken - before.spare_command_pools_taken;
        let reset = after.spare_command_pools_reset - before.spare_command_pools_reset;
        eprintln!(
            "unrun instances on {}, visibility buffer {}: {PASSES} passes ended {ended} replaced \
             instances; {taken} spare pools taken, {reset} of them reset",
            probe.device.name(),
            visibility.is_some(),
        );
        assert_eq!(
            ended, begun,
            "a replaced instance a pass whose first began its secondary"
        );
        assert_eq!(reset, 0, "spare pools reset as taken");
    }
    super::render_state::quiet(probe, &messages);
}

/// A pass with a depth attachment of its own whose draw uses the stencil it lacks, and one with a
/// stencil attachment whose draw uses depth, are refused: the hidden aspect beside an attachment
/// of the other is not built.
#[test]
fn a_hidden_aspect_beside_an_attachment_of_the_other_is_not_built() {
    use CompareFunction::{Always, Less};
    let (messages, _cache, probe) = validated();
    let refused = Err(Error::NotImplemented(RuntimeGap::HiddenDepthStencil));
    probe.run(PASS.depth(PixelFormat::Depth32Float, 1.0), |p, e, h| {
        let face = stencil(Always, StencilOperation::Replace, 0xff, 0xff);
        e.set_depth_stencil_state(&p.state(Always, false, Some(face)))
            .unwrap();
        e.set_render_pipeline_state(&p.pipeline(WHITE.depth(PixelFormat::Depth32Float)))
            .unwrap();
        let depth = p.depth(0.5, 0.0);
        e.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
        assert_eq!(
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0),
            refused
        );
        h.push(depth);
    });
    probe.run(PASS.stencil(0), |p, e, h| {
        e.set_depth_stencil_state(&p.state(Less, true, None))
            .unwrap();
        e.set_render_pipeline_state(&p.pipeline(WHITE.stencil()))
            .unwrap();
        let depth = p.depth(0.5, 0.0);
        e.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
        assert_eq!(
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0),
            refused
        );
        h.push(depth);
    });
    super::render_state::quiet(probe, &messages);
}

/// The pipeline's depth and stencil formats decide nothing: one it names that the pass lacks
/// tests the hidden attachment, and the pass's attachments are tested and written in their own
/// formats whatever the pipeline names (a Metal probe).
#[test]
fn the_pipeline_s_depth_and_stencil_formats_decide_nothing() {
    use CompareFunction::{Always, Equal, Less};
    use StencilOperation::{Keep, Replace};
    let (messages, _cache, probe) = validated();
    let name = probe.device.name().to_owned();
    let named = |fit: &dyn Fn(Pipe) -> Pipe, pipe: Pipe| {
        fit(pipe.depth(PixelFormat::Depth16Unorm).stencil())
    };
    // Named formats the pass lacks: the hidden float depth and stencil.
    let above = f32::from_bits(0.5f32.to_bits() + 1);
    let texels = hidden(
        &probe,
        "Depth16Unorm named, Equal above",
        &|p, e, h, fit| {
            e.set_depth_stencil_state(&p.state(Always, true, None))
                .unwrap();
            p.draw(e, h, named(fit, WHITE), 0.5, 0.0);
            e.set_depth_stencil_state(&p.state(Equal, false, None))
                .unwrap();
            p.draw(e, h, named(fit, WHITE.fragment("rt6_half")), above, 0.0);
        },
    );
    assert_eq!(
        texels,
        rows(&FULL),
        "Depth16Unorm named, Equal above on {name}"
    );
    let small = hidden(&probe, "Stencil8 named", &|p, e, h, fit| {
        e.set_stencil_reference_value(5);
        let replace = stencil(Always, Replace, 0xff, 0xff);
        e.set_depth_stencil_state(&p.state(Always, false, Some(replace)))
            .unwrap();
        let half = WHITE.fragment("rt6_half").vertex("rt6_small");
        p.draw(e, h, named(fit, half), 0.5, 0.0);
        let equal = stencil(Equal, Keep, 0xff, 0xff);
        e.set_depth_stencil_state(&p.state(Always, false, Some(equal)))
            .unwrap();
        p.draw(e, h, named(fit, WHITE), 0.5, 0.0);
    });
    let drawn = small.chunks(4).filter(|t| *t == [0xff; 4]).count();
    let rest = small.chunks(4).filter(|t| *t == [0; 4]).count();
    assert!(
        drawn > 0 && drawn + rest == 16,
        "Stencil8 named on {name}: {small:02x?}"
    );
    // The pass's depth, whatever the pipeline names.
    let depth16 = PASS.depth(PixelFormat::Depth16Unorm, 0.5);
    let written = |pipe: Pipe| {
        probe.run(depth16, |p, e, h| {
            e.set_depth_stencil_state(&p.state(Less, true, None))
                .unwrap();
            p.draw(e, h, pipe, 0.25, 0.0);
        })
    };
    let matched = written(WHITE.depth(PixelFormat::Depth16Unorm));
    assert_eq!(matched.color, rows(&FULL), "the pass's depth on {name}");
    let depth = matched.depth.as_deref().unwrap_or_default();
    assert!(
        depth.len() == 32
            && depth
                .chunks(2)
                .all(|d| u16::from_le_bytes([d[0], d[1]]) < 0x7fff),
        "the pass's depth written on {name}: {depth:02x?}"
    );
    for pipe in [WHITE, WHITE.depth(PixelFormat::Depth32Float)] {
        assert_eq!(written(pipe), matched, "the pass's depth on {name}");
    }
    // The pass's stencil, though the pipeline names none.
    let stored = probe.run(PASS.stencil(0), |p, e, h| {
        e.set_stencil_reference_value(7);
        let replace = stencil(Always, Replace, 0xff, 0xff);
        e.set_depth_stencil_state(&p.state(Always, false, Some(replace)))
            .unwrap();
        p.draw(e, h, WHITE, 0.5, 0.0);
    });
    assert_eq!(stored.color, rows(&FULL), "the pass's stencil on {name}");
    assert_eq!(
        stored.stencil,
        Some(vec![7; 16]),
        "the pass's stencil written on {name}"
    );
    eprintln!("depth and stencil formats the pipeline names otherwise on {name}: Metal's");
    super::render_state::quiet(probe, &messages);
}

/// A step of a pass beside an indirect command buffer's commands: set a depth/stencil state of a
/// compare, a write and a stencil face; draw as the encoder (the pipe at a depth), or draw no
/// vertices; execute commands, each drawing its pipe at its depth, `None` a reset command.
#[derive(Clone, Copy)]
enum Step {
    State(
        CompareFunction,
        bool,
        Option<(CompareFunction, StencilOperation)>,
    ),
    Draw(Pipe, f32),
    NoVertices(Pipe),
    Execute(&'static [Option<(Pipe, f32)>]),
}

/// A case of executed commands: its name, its steps, and the texels Metal stored (`None`: the
/// small triangle's white and the clear elsewhere).
type Case<'a> = (&'a str, &'a [Step], Option<&'a [u8]>);

/// `steps` in a pass without a depth or stencil attachment, the stencil reference 5, each
/// `Execute` running its commands of one indirect command buffer, each command setting its own
/// pipeline and depth buffer: the texels it stored, after checking they are those of the same
/// steps in a pass with a `Depth32Float` attachment cleared to 1.0 and a `Stencil8` one cleared
/// to 0 where each command is drawn as the encoder's own draw, plain Metal's answer.
fn executed(probe: &Probe, case: &str, steps: &[Step]) -> Vec<u8> {
    use metal2vulkan::metal::IndirectCommandType;
    use metal2vulkan_runtime::{IndirectCommandBufferDescriptor, ResourceOptions};
    let commands: Vec<Option<(Pipe, f32)>> = steps
        .iter()
        .flat_map(|step| match step {
            Step::Execute(commands) => commands.to_vec(),
            _ => Vec::new(),
        })
        .collect();
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.max_vertex_buffer_bind_count = 1;
    let icb = probe
        .device
        .new_indirect_command_buffer(
            &descriptor,
            commands.len().max(1) as u64,
            ResourceOptions::shared(),
        )
        .unwrap();
    let mut depths = Vec::new();
    for (index, command) in commands.iter().enumerate() {
        let Some((pipe, z)) = command else {
            continue;
        };
        let at = icb.indirect_render_command_at(index as u64).unwrap();
        at.set_render_pipeline_state(&probe.pipeline(pipe.indirect()))
            .unwrap();
        let depth = probe.depth(*z, 0.0);
        at.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
        at.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        depths.push(depth);
    }
    // The steps, each command executed, or drawn as the encoder's where `fit` fits the pipes to
    // the pass's attachments.
    let run = |p: &Probe,
               e: &mut RenderCommandEncoder,
               h: &mut Vec<Buffer>,
               fit: &dyn Fn(Pipe) -> Pipe,
               executes: bool| {
        e.set_stencil_reference_value(5);
        let mut next = 0;
        for step in steps {
            match *step {
                Step::State(compare, write, face) => {
                    let face = face.map(|(compare, pass)| stencil(compare, pass, 0xff, 0xff));
                    e.set_depth_stencil_state(&p.state(compare, write, face))
                        .unwrap();
                }
                Step::Draw(pipe, z) => p.draw(e, h, fit(pipe), z, 0.0),
                Step::NoVertices(pipe) => {
                    e.set_render_pipeline_state(&p.pipeline(fit(pipe))).unwrap();
                    let depth = p.depth(0.5, 0.0);
                    e.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
                    e.draw_primitives(PrimitiveType::Triangle, 0, 0, 1, 0)
                        .unwrap();
                    h.push(depth);
                }
                Step::Execute(commands) if executes => {
                    let count = commands.len() as u64;
                    e.execute_commands_in_buffer(&icb, next..next + count)
                        .unwrap();
                    next += count;
                }
                Step::Execute(commands) => {
                    for &(pipe, z) in commands.iter().flatten() {
                        p.draw(e, h, fit(pipe), z, 0.0);
                    }
                }
            }
        }
    };
    let without = probe.run(PASS, |p, e, h| run(p, e, h, &|pipe| pipe, true));
    let with = probe.run(
        PASS.depth(PixelFormat::Depth32Float, 1.0).stencil(0),
        |p, e, h| {
            run(
                p,
                e,
                h,
                &|pipe| pipe.depth(PixelFormat::Depth32Float).stencil(),
                false,
            )
        },
    );
    assert_eq!(
        without.color, with.color,
        "{case}: executed commands against attachments"
    );
    drop((depths, icb));
    without.color
}

/// Executed commands in a pass without a depth or stencil attachment test and write Metal's
/// hidden depth and stencil as the encoder's own draws at the call do (an M4
/// with macOS 27.0.1, plain and validated alike, every order of the earlier probes): against 1.0
/// and 0 at the pass's start, under the encoder's state at each call, their writes seen by the
/// commands and the encoder's draws after them, the encoder's by the commands. Metal's validation
/// stops the encoder's draws among these, not the commands. An execution of no draw (a reset
/// command) leaves the pass as it was.
#[test]
fn executed_commands_test_metal_s_hidden_depth_and_stencil() {
    use CompareFunction::{Always, Equal, Less, Never};
    use StencilOperation::{Keep, Replace};
    let (messages, _cache, probe) = validated();
    let name = probe.device.name().to_owned();
    const HALF: Pipe = WHITE.fragment("rt6_half");
    const SMALL: Pipe = HALF.vertex("rt6_small");
    let (white, empty, halves) = (rows(&FULL), rows(&EMPTY), [0x80; 4].repeat(16));
    let cases: [Case<'_>; 12] = [
        // Commands alone: the second fails against the first's write.
        (
            "less, two commands",
            &[
                Step::State(Less, true, None),
                Step::Execute(&[Some((WHITE, 0.5)), Some((HALF, 0.75))]),
            ],
            Some(&white[..]),
        ),
        (
            "less, two executions",
            &[
                Step::State(Less, true, None),
                Step::Execute(&[Some((WHITE, 0.5))]),
                Step::Execute(&[Some((HALF, 0.75))]),
            ],
            Some(&white[..]),
        ),
        // Each execution under the state at its call.
        (
            "less, then equal elsewhere",
            &[
                Step::State(Less, true, None),
                Step::Execute(&[Some((WHITE, 0.25))]),
                Step::State(Equal, false, None),
                Step::Execute(&[Some((HALF, 0.5))]),
            ],
            Some(&white[..]),
        ),
        (
            "never",
            &[
                Step::State(Never, false, None),
                Step::Execute(&[Some((WHITE, 0.5))]),
            ],
            Some(&empty[..]),
        ),
        (
            "stencil, two executions",
            &[
                Step::State(Always, false, Some((Always, Replace))),
                Step::Execute(&[Some((SMALL, 0.5))]),
                Step::State(Always, false, Some((Equal, Keep))),
                Step::Execute(&[Some((WHITE, 0.5))]),
            ],
            None,
        ),
        // Beside the encoder's draws, either order.
        (
            "the encoder's write, then commands",
            &[
                Step::State(Less, true, None),
                Step::Draw(WHITE, 0.5),
                Step::Execute(&[Some((HALF, 0.75))]),
            ],
            Some(&white[..]),
        ),
        (
            "commands' write, then the encoder's equal",
            &[
                Step::State(Less, true, None),
                Step::Execute(&[Some((WHITE, 0.25))]),
                Step::State(Equal, false, None),
                Step::Draw(HALF, 0.25),
            ],
            Some(&halves[..]),
        ),
        (
            "a plain encoder draw, then commands",
            &[
                Step::State(Always, false, None),
                Step::Draw(WHITE, 0.9),
                Step::State(Less, true, None),
                Step::Execute(&[Some((HALF, 0.5))]),
            ],
            Some(&halves[..]),
        ),
        (
            "commands, then a plain encoder draw",
            &[
                Step::State(Less, true, None),
                Step::Execute(&[Some((WHITE, 0.5))]),
                Step::State(Always, false, None),
                Step::Draw(HALF, 0.9),
            ],
            Some(&halves[..]),
        ),
        (
            "the encoder's stencil, then commands",
            &[
                Step::State(Always, false, Some((Always, Replace))),
                Step::Draw(SMALL, 0.5),
                Step::State(Always, false, Some((Equal, Keep))),
                Step::Execute(&[Some((WHITE, 0.5))]),
            ],
            None,
        ),
        (
            "an empty encoder draw, then commands",
            &[
                Step::State(Less, true, None),
                Step::NoVertices(WHITE),
                Step::Execute(&[Some((HALF, 0.5))]),
            ],
            Some(&halves[..]),
        ),
        (
            "the encoder's write, then a reset command",
            &[
                Step::State(Less, true, None),
                Step::Draw(WHITE, 0.5),
                Step::Execute(&[None]),
            ],
            Some(&white[..]),
        ),
    ];
    for (case, steps, expected) in cases {
        let texels = executed(&probe, case, steps);
        match expected {
            Some(expected) => assert_eq!(texels, expected, "{case} on {name}"),
            // The small triangle's texels, white; the rest, the clear.
            None => {
                let drawn = texels.chunks(4).filter(|t| *t == [0xff; 4]).count();
                let rest = texels.chunks(4).filter(|t| *t == [0; 4]).count();
                assert!(
                    drawn > 0 && drawn + rest == 16,
                    "{case} on {name}: {texels:02x?}"
                );
            }
        }
    }
    eprintln!("executed commands against the hidden depth and stencil on {name}: Metal's");
    super::render_state::quiet(probe, &messages);
}
