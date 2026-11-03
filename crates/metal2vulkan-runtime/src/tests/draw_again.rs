//! A render encoder's draw of its last direct draw again (`render/draw_again.rs`): a draw of the
//! same call with the same arguments, which no call of the encoder came between, records its
//! Vulkan draw command alone (`WorkCounts::draws_as_before`), with the same commands, barriers
//! and pixels as the same draws with a call between each that changes nothing; and only then: not
//! a draw of other arguments, an indirect draw, a draw that writes memory, a draw after a no-copy
//! buffer moved (its binding may read elsewhere), nor a draw that meets a notice its device
//! announces at every occurrence, which announces it at every draw. In builds with debug
//! assertions, every second draw again takes every step instead, and checks that it changed
//! nothing the draw again leaves (the module's check), so these tests run that check too.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixtures
//! `validation/fixtures/public/engine/tob_renders.ll` and `engine/cfb_depth_never.ll`
//! (`llvm-as`).

use super::blit_holds::{until, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library, pass, words_of};
use super::rebind_behind_a_wait::{Page, BYTES};
use crate::{
    BlitOption, Buffer, CommandBufferStatus, Degradation, DegradeNotice, Device, DeviceKind,
    DeviceOptions, Error, MessageSink, Origin, PipelineOptions, RenderCommandEncoder,
    RenderPipelineState, ResourceOptions, ShaderCacheOptions, Size, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{BlendFactor, IndexType, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The draws an encoder of a test draws.
const DRAWS: u64 = 9;

/// The side of the target.
const SIDE: u64 = 16;

/// The notices a device announced.
type Heard = Arc<Mutex<Vec<DegradeNotice>>>;

/// A device of the kind the cfg selects that announces the kinds of `each` at every occurrence,
/// its notices into `heard`.
fn device(cache: &Path, each: &[Degradation], heard: &Heard) -> Device {
    let heard = Arc::clone(heard);
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        messages: Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                heard.lock().unwrap().push(notice.clone());
            }
        })),
        announce_each: each.iter().copied().collect(),
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// The draws again of a pass of [`DRAWS`] of the same draw: all but its first two, the first
/// beginning the pass's instance (which keeps no draw), the second kept.
const HITS: u64 = DRAWS - 2;

/// How many of the draws again `hits` of a run recorded their Vulkan draw alone: every second one
/// from the first takes every step in a build with debug assertions (the module docs).
fn drawn_alone(hits: u64) -> u64 {
    match cfg!(debug_assertions) {
        true => hits / 2,
        false => hits,
    }
}

/// The state that draws a rectangle (`tob_rect`, `tob_solid`) into `RGBA8Unorm`, adding its color
/// to the target's, so that a target shows how many draws ran.
fn adding_state(device: &Device) -> RenderPipelineState {
    state(device, "engine/tob_renders.ll", ("tob_rect", "tob_solid"))
}

/// The state of the fixture at `path`'s `vertex` and `fragment` functions that draws into
/// `RGBA8Unorm`, adding its color to the target's.
fn state(device: &Device, path: &str, (vertex, fragment): (&str, &str)) -> RenderPipelineState {
    let library = library(device, path);
    let function = |name| {
        let function = library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
    descriptor.fragment = Some(function(fragment));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    color.blending_enabled = true;
    color.destination_rgb_blend_factor = BlendFactor::One;
    color.destination_alpha_blend_factor = BlendFactor::One;
    descriptor.color_attachments[0] = Some(color);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// What a run recorded and drew: its commands recorded by kind, the target's words, and the
/// device's counts after it, with the counts' change over it.
struct Ran {
    commands: BTreeMap<String, u64>,
    words: Vec<u32>,
    before: WorkCounts,
    after: WorkCounts,
}

impl Ran {
    /// How the count `of` changed over the run.
    fn counted(&self, of: impl Fn(&WorkCounts) -> u64) -> u64 {
        of(&self.after) - of(&self.before)
    }
}

/// The device's commands recorded so far, by kind.
fn commands(device: &Device) -> BTreeMap<String, u64> {
    let recorded = format!("{:?}", device.work_counts().commands_recorded);
    let fields = recorded
        .trim_start_matches("CommandsRecorded {")
        .trim_end_matches('}');
    fields
        .split(',')
        .filter_map(|field| {
            let (name, count) = field.split_once(':')?;
            Some((name.trim().to_owned(), count.trim().parse().unwrap()))
        })
        .collect()
}

/// Encode `encode` in one pass into a new cleared `RGBA8Unorm` target, commit and wait: what it
/// recorded and drew.
fn run(device: &Device, encode: impl Fn(&mut RenderCommandEncoder)) -> Ran {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let target = device.new_texture(&descriptor).unwrap();
    let back = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let (before, recorded) = (device.work_counts(), commands(device));
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&target))
        .unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    let after = device.work_counts();
    let commands = commands(device)
        .into_iter()
        .map(|(name, count)| {
            let before = recorded.get(&name).copied().unwrap_or(0);
            (name, count - before)
        })
        .collect();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        SIDE * 4,
        SIDE * SIDE * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    Ran {
        commands,
        words: words_of(&back),
        before,
        after,
    }
}

/// Run `encode` on `device` until a run draws no key's residual form and the device's pool is
/// idle: every later draw finds its key's own pipeline made, as a draw that is kept to draw again
/// does (a draw recorded against a making, or of a residual form, keeps nothing).
fn warm(device: &Device, encode: impl Fn(&mut RenderCommandEncoder)) {
    until("every key's own pipeline made", || {
        let residual = device.work_counts().residual_form_draws;
        run(device, &encode);
        device.work_counts().residual_form_draws == residual
    });
    until("the pipeline pool idle", || {
        device.work_counts().pipeline_pool_busy == 0
    });
}

/// Set `state`, the rectangle over the whole target and its color, `1 / 64` in each channel.
fn set_rectangle(encoder: &mut RenderCommandEncoder, state: &RenderPipelineState) {
    encoder.set_render_pipeline_state(state).unwrap();
    let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
    encoder.set_vertex_bytes(&corners, 0).unwrap();
    let color = bytes_of(&[1.0f32 / 64.0; 4], f32::to_le_bytes);
    encoder.set_fragment_bytes(&color, 0).unwrap();
}

/// The rectangle's draw: two triangles.
fn rectangle(encoder: &mut RenderCommandEncoder) {
    let drawn = encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0);
    assert_eq!(drawn, Ok(()));
}

/// [`DRAWS`] draws of the rectangle, with `between` called before each but the first.
fn rectangles(
    encoder: &mut RenderCommandEncoder,
    state: &RenderPipelineState,
    between: impl Fn(&mut RenderCommandEncoder),
) {
    set_rectangle(encoder, state);
    for at in 0..DRAWS {
        if at > 0 {
            between(encoder);
        }
        rectangle(encoder);
    }
}

/// Draws of the same rectangle one after another record their Vulkan draws alone, but the first:
/// the same commands of every kind, the same barriers, state commands and noted uses, and the
/// same pixels, as the same draws with a call between each that changes nothing for them
/// (`set_blend_color` of the color the encoder holds), each of which takes every step.
#[test]
fn draws_again_record_what_draws_after_calls_record() {
    let scratch = Scratch::new("draw-again");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = adding_state(&device);
    let same_color =
        |encoder: &mut RenderCommandEncoder| encoder.set_blend_color(0.0, 0.0, 0.0, 0.0);
    warm(&device, |encoder| rectangles(encoder, &state, same_color));
    let heard_in = |ran: &dyn Fn() -> Ran| {
        heard.lock().unwrap().clear();
        let ran = ran();
        let kinds: Vec<Degradation> = (heard.lock().unwrap().iter())
            .map(|notice| notice.degradation)
            .collect();
        (ran, kinds)
    };
    let (again, again_heard) =
        heard_in(&|| run(&device, |encoder| rectangles(encoder, &state, |_| {})));
    let (after_calls, after_calls_heard) =
        heard_in(&|| run(&device, |encoder| rectangles(encoder, &state, same_color)));
    assert_eq!(again_heard, after_calls_heard, "the notices of the draws");
    let as_before = |counts: &WorkCounts| counts.draws_as_before;
    assert_eq!(again.counted(as_before), drawn_alone(HITS));
    assert_eq!(after_calls.counted(as_before), 0);
    assert_eq!(again.commands, after_calls.commands);
    assert_eq!(again.commands["cmd_draw"], DRAWS);
    assert_eq!(again.words, after_calls.words, "the pixels of the draws");
    // Each of the draws added its color: the target holds more than one draw's.
    let one = run(&device, |encoder| {
        set_rectangle(encoder, &state);
        rectangle(encoder);
    });
    assert_ne!(again.words, one.words, "the draws added one draw's color");
    // Their commands, barriers among them, are the same (above), and so are their counts of
    // state commands, uses noted as before and pushes of set 0.
    let counts: [(&str, fn(&WorkCounts) -> u64); 3] = [
        ("draw_states_recorded", |counts| counts.draw_states_recorded),
        ("commands_noted_as_before", |counts| {
            counts.commands_noted_as_before
        }),
        ("draws_pushing_set_zero", |counts| {
            counts.draws_pushing_set_zero
        }),
    ];
    for (name, count) in counts {
        assert_eq!(again.counted(count), after_calls.counted(count), "{name}");
    }
    // Each of the draws but the first noted its uses as before, the draws again among them.
    assert_eq!(again.counted(counts[1].1), DRAWS - 1);
}

/// Indexed draws of the same indices from the same offset draw again too; at another offset they
/// do not.
#[test]
fn indexed_draws_from_the_same_indices_draw_again() {
    let scratch = Scratch::new("draw-again-indexed");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = adding_state(&device);
    let indices = bytes_of(&[0u16, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], u16::to_le_bytes);
    let index = device
        .new_buffer_with_bytes(&indices, ResourceOptions::shared())
        .unwrap();
    let indexed = |encoder: &mut RenderCommandEncoder, offset: u64| {
        let drawn = encoder.draw_indexed_primitives(
            PrimitiveType::Triangle,
            6,
            IndexType::UInt16,
            &index,
            offset,
            1,
            0,
            0,
        );
        assert_eq!(drawn, Ok(()));
    };
    let draws = |encoder: &mut RenderCommandEncoder, offsets: &dyn Fn(u64) -> u64| {
        set_rectangle(encoder, &state);
        for at in 0..DRAWS {
            indexed(encoder, offsets(at));
        }
    };
    warm(&device, |encoder| draws(encoder, &|_| 0));
    let same = run(&device, |encoder| draws(encoder, &|_| 0));
    let as_before = |counts: &WorkCounts| counts.draws_as_before;
    assert_eq!(same.counted(as_before), drawn_alone(HITS));
    assert_eq!(same.commands["cmd_draw_indexed"], DRAWS);
    let moving = run(&device, |encoder| draws(encoder, &|at| (at % 2) * 12));
    assert_eq!(moving.counted(as_before), 0);
    assert_eq!(same.words, moving.words, "the pixels of the draws");
}

/// No draw draws again but the same direct draw after the same: not a draw of other arguments,
/// an indirect draw, nor a draw that writes memory (`cfb_dn_count`, which counts its fragments in
/// a buffer's word 0); each counts what it drew, every draw's.
#[test]
fn other_draws_take_every_step() {
    let scratch = Scratch::new("draw-again-others");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = adding_state(&device);
    let as_before = |counts: &WorkCounts| counts.draws_as_before;
    // Another base instance at each draw, which the functions do not read.
    let others = |encoder: &mut RenderCommandEncoder| {
        set_rectangle(encoder, &state);
        for at in 0..DRAWS as u32 {
            let drawn = encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, at % 2);
            assert_eq!(drawn, Ok(()));
        }
    };
    warm(&device, others);
    assert_eq!(run(&device, others).counted(as_before), 0);
    // Indirect draws of six vertices, one instance.
    let arguments = bytes_of(&[6u32, 1, 0, 0], u32::to_le_bytes);
    let indirect = device
        .new_buffer_with_bytes(&arguments, ResourceOptions::shared())
        .unwrap();
    let indirect_draws = |encoder: &mut RenderCommandEncoder| {
        set_rectangle(encoder, &state);
        for _ in 0..DRAWS {
            let drawn = encoder.draw_primitives_indirect(PrimitiveType::Triangle, &indirect, 0);
            assert_eq!(drawn, Ok(()));
        }
    };
    warm(&device, indirect_draws);
    let ran = run(&device, indirect_draws);
    assert_eq!(ran.counted(as_before), 0);
    assert_eq!(ran.commands["cmd_draw_indirect"], DRAWS);
    // Draws that count their fragments in a buffer: every fragment of every draw counted.
    let counting = state_counting(&device);
    let counter = device.new_buffer(4, ResourceOptions::shared()).unwrap();
    let counted = |encoder: &mut RenderCommandEncoder| {
        encoder.set_render_pipeline_state(&counting).unwrap();
        encoder.set_fragment_buffer(Some(&counter), 0, 0).unwrap();
        for _ in 0..DRAWS {
            let drawn = encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0);
            assert_eq!(drawn, Ok(()));
        }
    };
    warm(&device, counted);
    clear(&counter);
    let ran = run(&device, counted);
    assert_eq!(ran.counted(as_before), 0);
    assert_eq!(words_of(&counter), [(DRAWS * SIDE * SIDE) as u32]);
}

/// The state of `cfb_dn_full`, a triangle over the whole target, and `cfb_dn_count`, which counts
/// its invocations in word 0 of buffer 0, into `RGBA8Unorm`.
fn state_counting(device: &Device) -> RenderPipelineState {
    state(
        device,
        "engine/cfb_depth_never.ll",
        ("cfb_dn_full", "cfb_dn_count"),
    )
}

/// Zero `buffer`, a host-visible buffer no command uses now.
fn clear(buffer: &Buffer) {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's `length` bytes, which no command buffer uses now.
    unsafe { std::ptr::write_bytes(contents.as_ptr(), 0, buffer.length() as usize) };
}

/// A draw after a no-copy buffer moved (a detach, a rebind), on any thread and whatever buffer,
/// takes every step: a binding of the buffer may read elsewhere. The draws after it draw again.
#[test]
fn a_draw_after_a_no_copy_buffer_moved_takes_every_step() {
    let pages = [Page::new(), Page::new()];
    let scratch = Scratch::new("draw-again-moved");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = adding_state(&device);
    // SAFETY: each page outlives every buffer over it (dropped after the device), and only the
    // buffer's commands use it while it lives; no command uses the buffer.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            pages[0].memory(),
            BYTES as u64,
            ResourceOptions::shared(),
            None,
        )
    };
    let no_copy = match made {
        Ok(buffer) => buffer,
        Err(Error::Unsupported(gap)) => {
            eprintln!("DRAW-AGAIN: the device imports no host memory: {gap:?}");
            return;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    let plain = |encoder: &mut RenderCommandEncoder| rectangles(encoder, &state, |_| {});
    warm(&device, plain);
    let as_before = |counts: &WorkCounts| counts.draws_as_before;
    let reference = run(&device, plain);
    assert_eq!(reference.counted(as_before), drawn_alone(HITS));
    // The buffer detached after the third draw, and over the second page again after the sixth.
    let moving = run(&device, |encoder| {
        set_rectangle(encoder, &state);
        for at in 0..DRAWS {
            match at {
                3 => assert_eq!(no_copy.detach_backing(), Ok(true)),
                // SAFETY: as above.
                6 => {
                    unsafe { no_copy.rebind_bytes_no_copy(pages[1].memory(), None, None) }.unwrap()
                }
                _ => {}
            }
            rectangle(encoder);
        }
    });
    // Three runs of three draws: one draw again in the first (its first draw begins the pass's
    // instance), two in each other.
    assert_eq!(
        moving.counted(as_before),
        drawn_alone(1) + 2 * drawn_alone(2)
    );
    assert_eq!(moving.commands, reference.commands);
    assert_eq!(moving.words, reference.words, "the pixels of the draws");
}

/// A draw that meets a notice its device announces at every occurrence (`UnsizedPoints`: points
/// of a vertex function that writes no point size) takes every step at every draw, and announces
/// it at each; on a device that announces it once, the draws after the first draw again.
#[test]
fn a_notice_announced_at_each_occurrence_is_announced_at_every_draw() {
    fn points(state: &RenderPipelineState) -> impl Fn(&mut RenderCommandEncoder) + '_ {
        move |encoder| {
            set_rectangle(encoder, state);
            for _ in 0..DRAWS {
                let drawn = encoder.draw_primitives(PrimitiveType::Point, 0, 4, 1, 0);
                assert_eq!(drawn, Ok(()));
            }
        }
    }
    let as_before = |counts: &WorkCounts| counts.draws_as_before;
    for each in [false, true] {
        let scratch = Scratch::new("draw-again-notices");
        let heard = Heard::default();
        let kinds = match each {
            true => &[Degradation::UnsizedPoints][..],
            false => &[],
        };
        let device = device(&scratch.0, kinds, &heard);
        let state = adding_state(&device);
        warm(&device, points(&state));
        heard.lock().unwrap().clear();
        let ran = run(&device, points(&state));
        let announced = (heard.lock().unwrap().iter())
            .filter(|notice| notice.degradation == Degradation::UnsizedPoints)
            .count() as u64;
        match each {
            true => {
                assert_eq!(announced, DRAWS);
                assert_eq!(ran.counted(as_before), 0);
            }
            false => {
                assert_eq!(announced, 0, "announced in the warm runs");
                assert_eq!(ran.counted(as_before), drawn_alone(HITS));
            }
        }
    }
}
