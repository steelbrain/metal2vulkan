//! A render encoder's draw of its last direct draw again (`render/draw_again.rs`): a draw of the
//! same call with the same arguments, which no call of the encoder came between, records its
//! Vulkan draw command alone (`WorkCounts::draws_as_before`), with the same commands, barriers
//! and pixels as the same draws with a call between each that changes nothing; and only then: not
//! a draw of other arguments, an indirect draw, a draw that writes memory, a draw after a no-copy
//! buffer moved (its binding may read elsewhere), nor a draw that meets a notice its device
//! announces at every occurrence, which announces it at every draw. A draw again after calls that
//! bound other buffers, offsets or bytes at its buffer slots, or other textures or sampler states
//! at its texture and sampler slots, rebinds those alone (`render/draw_again/rebound.rs`,
//! `WorkCounts::draws_rebound`), with the same commands and pixels as the same draws taking every
//! step; so does a draw of other arguments alone, which works out again what they decide
//! (`WorkCounts::draws_rebound_arguments`). In builds with debug assertions, every second draw
//! again takes every step instead, and checks that it changed nothing the draw again leaves and,
//! rebound, worked out what every step works out (the modules' checks), so these tests run those
//! checks too.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixtures
//! `validation/fixtures/public/engine/tob_renders.ll` and `engine/cfb_depth_never.ll`
//! (`llvm-as`).

use super::blit_holds::{until, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library, pass, words_of};
use super::rebind_behind_a_wait::{Page, BYTES};
#[cfg(debug_assertions)]
use crate::render::dynamic_compares;
use crate::{
    BlitOption, Buffer, CommandBufferStatus, CommandQueue, Degradation, DegradeNotice, Device,
    DeviceKind, DeviceOptions, Error, EventRef, MessageSink, Origin, PipelineOptions, Region,
    RenderCommandEncoder, RenderPipelineState, ResourceOptions, SamplerState, ScissorRect,
    ShaderCacheOptions, Size, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    BlendFactor, IndexType, PixelFormat, PrimitiveType, SamplerAddressMode, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The draws an encoder of a test draws.
pub(super) const DRAWS: u64 = 9;

/// The side of the target.
const SIDE: u64 = 16;

/// The notices a device announced.
pub(super) type Heard = Arc<Mutex<Vec<DegradeNotice>>>;

/// A device of the kind the cfg selects that announces the kinds of `each` at every occurrence,
/// its notices into `heard`.
pub(super) fn device(cache: &Path, each: &[Degradation], heard: &Heard) -> Device {
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

/// The draws again of a pass of [`DRAWS`] of the same draw: all but its first, which begins the
/// pass's instance and is kept.
const HITS: u64 = DRAWS - 1;

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
pub(super) fn adding_state(device: &Device) -> RenderPipelineState {
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
pub(super) struct Ran {
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
/// recorded and drew. Its begins (`begin_command_buffer`) are the `VkCommandBuffer`s its encoders
/// began (`WorkCounts::encoder_command_buffers_begun`), its primary among them whether its encoder
/// began it or took it begun ahead (`WorkCounts::encoder_primaries_begun_ahead`), and no other
/// thread's: the device's count of `vkBeginCommandBuffer` also takes the primaries that the threads
/// completing other command buffers begin ahead in their spare pools where no thread waited for
/// those (`spares.rs`), whenever the GPU's completions come, so a begin there may or may not fall
/// while the run encodes.
pub(super) fn run(device: &Device, encode: impl Fn(&mut RenderCommandEncoder)) -> Ran {
    let queue = device.new_command_queue().unwrap();
    run_on(device, &queue, encode, true)
}

/// [`run`] on `queue`, its command buffer's completion waited for where `waited`, else its status
/// read until it completed, no thread waiting for it.
fn run_on(
    device: &Device,
    queue: &CommandQueue,
    encode: impl Fn(&mut RenderCommandEncoder),
    waited: bool,
) -> Ran {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let target = device.new_texture(&descriptor).unwrap();
    let back = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let (before, recorded) = (device.work_counts(), commands(device));
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&target))
        .unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    let after = device.work_counts();
    let begun = after.encoder_command_buffers_begun - before.encoder_command_buffers_begun;
    let commands = commands(device)
        .into_iter()
        .map(|(name, count)| match name == "begin_command_buffer" {
            true => (name, begun),
            false => {
                let before = recorded.get(&name).copied().unwrap_or(0);
                (name, count - before)
            }
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
    let completed = || command_buffer.status() == CommandBufferStatus::Completed;
    match waited {
        true => command_buffer.wait_until_completed_for(WAIT).unwrap(),
        false => until("the command buffer completed, no thread waiting", completed),
    }
    assert!(completed());
    Ran {
        commands,
        words: words_of(&back),
        before,
        after,
    }
}

/// Run `encode` on `device` until a run draws no key's residual form and the device's pool is
/// idle: every later draw finds its key's own pipeline made, as a draw that is kept to draw again
/// does (a draw recorded against a making, or of a residual form, keeps nothing). Then wait until
/// the device's own checks ended ([`checks_ended`]), so that no window a test opens after it meets
/// their command buffers.
pub(super) fn warm(device: &Device, encode: impl Fn(&mut RenderCommandEncoder)) {
    until("every key's own pipeline made", || {
        let residual = device.work_counts().residual_form_draws;
        run(device, &encode);
        device.work_counts().residual_form_draws == residual
    });
    until("the pipeline pool idle", || {
        device.work_counts().pipeline_pool_busy == 0
    });
    checks_ended(device);
}

/// How long a device's own checks may run ([`checks_ended`]).
const CHECKS_WAIT: Duration = Duration::from_secs(120);

/// Wait until `device`'s check of the GPU's idling between `VkQueue`s (`queue_switch_check.rs`,
/// started by its first command queue) and its submission checks (`submit_check.rs`, started as
/// it is made) ended. Each records command buffers on a thread of its own (the queue-switch
/// check's timestamp writes, the submission checks' empty ones), which
/// `WorkCounts::commands_recorded` counts as it counts a run's: a run whose window met one counted
/// its commands too, 32 `cmd_reset_query_pool`, 64 `cmd_write_timestamp` and 32
/// `end_command_buffer` of the queue-switch check under twelve processes' load on an RTX 5080.
fn checks_ended(device: &Device) {
    assert!(
        device.settle_queue_switch_check(CHECKS_WAIT),
        "the device's queue-switch check ended"
    );
    assert!(
        device.settle_submit_checks(CHECKS_WAIT),
        "the device's submission checks ended"
    );
}

/// A run's commands count its primary as begun whether its encoder began it or took it begun
/// ahead ([`run`]): a run after a command buffer that completed while no thread waited for it,
/// whose pool's primary its completing thread began ahead (`WorkCounts::primaries_begun_ahead`)
/// and the run's encoder took, counts the commands of a run whose encoder began its own.
#[test]
fn a_run_counts_its_primary_begun_by_its_encoder_or_ahead_alike() {
    let scratch = Scratch::new("draw-again-begun-ahead");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = adding_state(&device);
    let plain = |encoder: &mut RenderCommandEncoder| rectangles(encoder, &state, |_| {});
    warm(&device, plain);
    let taken = |ran: &Ran| ran.counted(|counts| counts.encoder_primaries_begun_ahead);
    // A run whose encoder began its own primary, as one after a command buffer a thread waited
    // for at its completion does.
    let own = std::cell::RefCell::new(None);
    until("a run whose encoder began its own primary", || {
        let ran = run(&device, plain);
        let began = taken(&ran) == 0;
        if began {
            own.replace(Some(ran));
        }
        began
    });
    let own = own.into_inner().unwrap();
    // A run that completes while no thread waits for it, then the primary its pool's completing
    // thread begins ahead, and a run on the same queue: a pool serves its queue family's command
    // buffers alone, and a new queue may lie in another family.
    let ahead = |counts: &WorkCounts| counts.primaries_begun_ahead;
    let begun = ahead(&device.work_counts());
    let queue = device.new_command_queue().unwrap();
    run_on(&device, &queue, plain, false);
    until("its pool's primary begun ahead", || {
        ahead(&device.work_counts()) > begun
    });
    let after_ahead = run_on(&device, &queue, plain, true);
    assert_eq!(
        taken(&after_ahead),
        1,
        "the encoder took the primary begun ahead"
    );
    assert_eq!(after_ahead.commands, own.commands);
    assert_eq!(after_ahead.commands["begin_command_buffer"], 1);
}

/// A primary another thread begins ahead while a run encodes is no begin of the run's ([`run`]):
/// a command buffer on another queue that waits for an event the run's encoding signals completes
/// while the run encodes, with no thread waiting for it, so the thread that completes it begins
/// its pool's primary ahead then (`spares.rs`); the run counts the begins of a run alone.
#[test]
fn a_primary_begun_ahead_while_a_run_encodes_is_no_begin_of_the_run_s() {
    let scratch = Scratch::new("draw-again-begun-meanwhile");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = adding_state(&device);
    let plain = |encoder: &mut RenderCommandEncoder| rectangles(encoder, &state, |_| {});
    warm(&device, plain);
    let alone = run(&device, plain);
    let gate = device.new_shared_event().unwrap();
    let filled = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let held = device
        .new_command_queue()
        .unwrap()
        .command_buffer()
        .unwrap();
    held.encode_wait_for_event(EventRef::Shared(&gate), 1)
        .unwrap();
    let mut blit = held.blit_command_encoder().unwrap();
    blit.fill_buffer(&filled, 0..4096, 1).unwrap();
    blit.end_encoding().unwrap();
    held.commit().unwrap();
    let ahead = |counts: &WorkCounts| counts.primaries_begun_ahead;
    let meanwhile = run(&device, |encoder| {
        let begun = ahead(&device.work_counts());
        gate.set_signaled_value(1);
        until(
            "the held command buffer's pool's primary begun ahead",
            || {
                held.status() == CommandBufferStatus::Completed
                    && ahead(&device.work_counts()) > begun
            },
        );
        plain(encoder);
    });
    assert!(
        meanwhile.counted(ahead) > 0,
        "a primary begun ahead meanwhile"
    );
    assert_eq!(meanwhile.commands["begin_command_buffer"], 1);
    assert_eq!(
        meanwhile.commands["begin_command_buffer"],
        alone.commands["begin_command_buffer"]
    );
    assert_eq!(meanwhile.words, alone.words, "the pixels of the draws");
}

/// [`warm`] returns once the device's own checks ended ([`checks_ended`]): with its check of the
/// GPU's idling between `VkQueue`s held between two of its rounds, `warm` waits for its verdict
/// (a call that waits for it lets the hold go, below), which is known as `warm` returns. A device
/// that runs no such check is skipped.
#[test]
fn warm_returns_once_the_device_s_own_checks_ended() {
    let scratch = Scratch::new("draw-again-checks-ended");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    if crate::queue_switch_check::pair(&device).is_none()
        || !crate::queue_switch_check::timed(&device)
    {
        println!(
            "SKIPPED {}: the device runs no queue-switch check",
            device.name()
        );
        return;
    }
    let switch = Arc::clone(&device.shared().queue_switch);
    switch.held.hold();
    let releasing = std::thread::spawn({
        let switch = Arc::clone(&switch);
        move || {
            let pending = || switch.verdict() == crate::queue_share::Verdict::Pending;
            while pending() && !switch.awaited() && !switch.stopping() {
                std::thread::sleep(Duration::from_millis(1));
            }
            switch.held.release();
        }
    });
    let state = adding_state(&device);
    warm(&device, |encoder| rectangles(encoder, &state, |_| {}));
    assert_ne!(
        switch.verdict(),
        crate::queue_share::Verdict::Pending,
        "warm returned while the device's queue-switch check ran"
    );
    releasing.join().unwrap();
}

/// Set `state`, the rectangle over the whole target and its color, `1 / 64` in each channel.
pub(super) fn set_rectangle(encoder: &mut RenderCommandEncoder, state: &RenderPipelineState) {
    encoder.set_render_pipeline_state(state).unwrap();
    let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
    encoder.set_vertex_bytes(&corners, 0).unwrap();
    let color = bytes_of(&[1.0f32 / 64.0; 4], f32::to_le_bytes);
    encoder.set_fragment_bytes(&color, 0).unwrap();
}

/// The rectangle's draw: two triangles.
pub(super) fn rectangle(encoder: &mut RenderCommandEncoder) {
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
/// (`set_tessellation_factor_scale`, which no draw of their state reads), each of which takes
/// every step.
#[test]
fn draws_again_record_what_draws_after_calls_record() {
    let scratch = Scratch::new("draw-again");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = adding_state(&device);
    let forgets = |encoder: &mut RenderCommandEncoder| encoder.set_tessellation_factor_scale(1.0);
    warm(&device, |encoder| rectangles(encoder, &state, forgets));
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
        heard_in(&|| run(&device, |encoder| rectangles(encoder, &state, forgets)));
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
    // Three runs of three draws, two drawing again in each: the first run's first draw, which
    // begins the pass's instance, is kept as any.
    assert_eq!(moving.counted(as_before), 3 * drawn_alone(2));
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

/// The rectangles of [`offset_rectangles`]' draws, 16 bytes apart: the `at`th from column `at` of
/// the target to its right edge, so that each column shows how many draws reached it.
pub(super) fn rectangles_from_columns(device: &Device) -> Buffer {
    let rectangles: Vec<f32> = (0..DRAWS)
        .flat_map(|at| [-1.0 + at as f32 * 2.0 / SIDE as f32, -1.0, 1.0, 1.0])
        .collect();
    let bytes = bytes_of(&rectangles, f32::to_le_bytes);
    device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap()
}

/// [`DRAWS`] draws of `state`'s rectangle, the `at`th of the rectangle at `16 * at` bytes of
/// `rectangles` (vertex buffer 0's offset set before each draw but the first) in the color
/// `color(at)` (fragment bytes 0, set before each draw), with `between` called after them.
fn offset_rectangles(
    encoder: &mut RenderCommandEncoder,
    (state, rectangles): (&RenderPipelineState, &Buffer),
    color: impl Fn(u64) -> f32,
    between: impl Fn(&mut RenderCommandEncoder),
) {
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_buffer(Some(rectangles), 0, 0).unwrap();
    for at in 0..DRAWS {
        if at > 0 {
            encoder.set_vertex_buffer_offset(16 * at, 0).unwrap();
            between(encoder);
        }
        let bytes = bytes_of(&[color(at); 4], f32::to_le_bytes);
        encoder.set_fragment_bytes(&bytes, 0).unwrap();
        rectangle(encoder);
    }
}

/// Draws again and rebound draws count themselves as they are recorded: the device's counts, read
/// before each draw but the first with no call of the encoder since the last draw but the binds
/// a rebound draw takes, hold every draw again and every rebound draw before it, not only once
/// another call or the encoder's end came.
#[test]
fn draws_again_and_rebound_count_as_they_record() {
    let scratch = Scratch::new("draw-again-counted");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = adding_state(&device);
    let columns = rectangles_from_columns(&device);
    let drawn = (&state, &columns);
    warm(&device, |encoder| rectangles(encoder, &state, |_| {}));
    warm(&device, |encoder| {
        offset_rectangles(encoder, drawn, |_| 1.0 / 64.0, |_| {})
    });
    // The count `of` read before each draw but the first, over its count before the run.
    let read_between =
        |encode: &dyn Fn(&mut RenderCommandEncoder, &dyn Fn(&mut RenderCommandEncoder)),
         of: fn(&WorkCounts) -> u64| {
            let read = std::cell::RefCell::new(Vec::new());
            let before = of(&device.work_counts());
            run(&device, |encoder| {
                encode(encoder, &|_| {
                    read.borrow_mut().push(of(&device.work_counts()) - before)
                });
            });
            read.into_inner()
        };
    // Before the `at`th draw: the draws again among the first `at` draws, the first beginning the
    // instance and kept.
    let expected: Vec<u64> = (1..DRAWS)
        .map(|at| drawn_alone(at.saturating_sub(1)))
        .collect();
    let again = read_between(
        &|encoder, between| rectangles(encoder, &state, between),
        |counts| counts.draws_as_before,
    );
    assert_eq!(again, expected, "draws again, counted as they draw");
    let rebound = read_between(
        &|encoder, between| offset_rectangles(encoder, drawn, |_| 1.0 / 64.0, between),
        |counts| counts.draws_rebound,
    );
    assert_eq!(rebound, expected, "rebound draws, counted as they draw");
}

/// Draws after calls that bound other offsets and bytes at buffer slots their functions read
/// rebind those slots alone (`WorkCounts::draws_rebound`), with the same commands of every kind,
/// state commands and noted uses, and the same pixels, as the same draws with a call between each
/// that changes nothing for them (`set_tessellation_factor_scale`, which no draw of their state
/// reads), each of which takes every step; at another offset each and in another color each, as each column's
/// count of draws and each draw's color show. Calls that changed more slots than a draw rebound
/// takes between them (samplers 0 to 8, which the functions do not read) take every step.
#[test]
fn draws_after_buffer_changes_rebind_what_every_step_binds() {
    let scratch = Scratch::new("draw-again-rebound");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = adding_state(&device);
    let rectangles = rectangles_from_columns(&device);
    let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
    let forgets = |encoder: &mut RenderCommandEncoder| encoder.set_tessellation_factor_scale(1.0);
    let samplers = |encoder: &mut RenderCommandEncoder| {
        for index in 0..9 {
            encoder
                .set_fragment_sampler_state(Some(&sampler), index)
                .unwrap();
        }
    };
    let one = |_| 1.0 / 64.0;
    let each = |at| (at + 1) as f32 / 64.0;
    let drawn = (&state, &rectangles);
    warm(&device, |encoder| {
        offset_rectangles(encoder, drawn, one, |_| {})
    });
    let rebound_count = |counts: &WorkCounts| counts.draws_rebound;
    let as_before = |counts: &WorkCounts| counts.draws_as_before;
    for color in [&one as &dyn Fn(u64) -> f32, &each] {
        let rebound = run(&device, |encoder| {
            offset_rectangles(encoder, drawn, color, |_| {});
        });
        let every = run(&device, |encoder| {
            offset_rectangles(encoder, drawn, color, forgets);
        });
        let sampled = run(&device, |encoder| {
            offset_rectangles(encoder, drawn, color, samplers);
        });
        assert_eq!(rebound.counted(rebound_count), drawn_alone(HITS));
        assert_eq!(rebound.counted(as_before), 0);
        // Bytes bound at each draw: no draw is moved.
        assert_eq!(rebound.counted(|counts| counts.draws_moved), 0);
        assert_eq!(every.counted(rebound_count), 0);
        assert_eq!(sampled.counted(rebound_count), 0);
        assert_eq!(rebound.commands, every.commands);
        assert_eq!(rebound.commands, sampled.commands);
        assert_eq!(rebound.commands["cmd_draw"], DRAWS);
        assert_eq!(rebound.words, every.words, "the pixels of the draws");
        assert_eq!(rebound.words, sampled.words, "the pixels of the draws");
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
            assert_eq!(rebound.counted(count), every.counted(count), "{name}");
        }
        // Column 0 holds the first draw's color alone; the last column every draw's.
        let last = SIDE as usize - 1;
        assert_ne!(rebound.words[0], rebound.words[last], "the draws' offsets");
    }
}

/// [`DRAWS`] draws of the rectangles of `columns`, each through the next of `states` in turn
/// (set before each draw), the `at`th at `16 * at` bytes of vertex buffer 0 (its offset set before
/// each draw but the first, then `between`) in the color of fragment bytes 0 (set before each
/// draw); `drawn` draws each.
fn switching_rectangles(
    encoder: &mut RenderCommandEncoder,
    (states, columns): (&[&RenderPipelineState], &Buffer),
    between: impl Fn(&mut RenderCommandEncoder),
    drawn: impl Fn(&mut RenderCommandEncoder, usize),
) {
    encoder.set_vertex_buffer(Some(columns), 0, 0).unwrap();
    for at in 0..DRAWS {
        let state = (at as usize) % states.len();
        encoder.set_render_pipeline_state(states[state]).unwrap();
        if at > 0 {
            encoder.set_vertex_buffer_offset(16 * at, 0).unwrap();
            between(encoder);
        }
        let bytes = bytes_of(&[(at + 1) as f32 / 64.0; 4], f32::to_le_bytes);
        encoder.set_fragment_bytes(&bytes, 0).unwrap();
        drawn(encoder, state);
    }
}

/// Draws that switch between two pipeline states at each draw rebind each state's own last draw,
/// its kept draw set aside with its state's bind state while the other state draws
/// (`render/draw_again.rs`): every draw after each state's first is rebound
/// (`WorkCounts::draws_rebound`), with the same commands of every kind, state commands and pixels
/// as the same draws each taking every step, after a call between each that changes nothing for
/// them (`set_tessellation_factor_scale`, which no draw of their states reads), which forgets
/// the kept draws of both states. Every draw keeps itself (`WorkCounts::draws_unkept`), the pass's
/// first, which begins the pass's instance, too, and no draw finds its kept draw unlike it or
/// stops its rebind. And a draw that keeps nothing (one that writes memory,
/// `cfb_dn_count`, counted so) forgets the other state's kept draw: draws switching with it take
/// every step.
#[test]
fn draws_switching_pipeline_states_rebind_each_state_s_last_draw() {
    let scratch = Scratch::new("draw-again-states");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let (first, second) = (adding_state(&device), adding_state(&device));
    let columns = rectangles_from_columns(&device);
    let pair = [&first, &second];
    let states = (&pair[..], &columns);
    let forgets = |encoder: &mut RenderCommandEncoder| encoder.set_tessellation_factor_scale(1.0);
    let drawn = |encoder: &mut RenderCommandEncoder, _| rectangle(encoder);
    warm(&device, |encoder| {
        switching_rectangles(encoder, states, |_| {}, drawn)
    });
    let rebound_count = |counts: &WorkCounts| counts.draws_rebound;
    let states_recorded = |counts: &WorkCounts| counts.draw_states_recorded;
    let rebound = run(&device, |encoder| {
        switching_rectangles(encoder, states, |_| {}, drawn)
    });
    let every = run(&device, |encoder| {
        switching_rectangles(encoder, states, forgets, drawn)
    });
    assert_eq!(rebound.counted(rebound_count), drawn_alone(DRAWS - 2));
    assert_eq!(every.counted(rebound_count), 0);
    assert_eq!(rebound.commands, every.commands);
    assert_eq!(rebound.commands["cmd_draw"], DRAWS);
    assert_eq!(
        rebound.counted(states_recorded),
        every.counted(states_recorded)
    );
    assert_eq!(rebound.words, every.words, "the pixels of the draws");
    let last = SIDE as usize - 1;
    assert_ne!(rebound.words[0], rebound.words[last], "the draws' offsets");
    assert_eq!(rebound.counted(|counts| counts.draws_unkept), 0);
    let missed = |counts: &WorkCounts| counts.draws_unlike_kept + counts.draws_rebind_stopped;
    assert_eq!(rebound.counted(missed), 0);
    // Between the adding state's draws, draws that count their fragments in a buffer.
    let counting = state_counting(&device);
    let counter = device.new_buffer(4, ResourceOptions::shared()).unwrap();
    let with_counting = [&first, &counting];
    let writing = (&with_counting[..], &columns);
    let drawn = |encoder: &mut RenderCommandEncoder, state: usize| match state {
        0 => rectangle(encoder),
        _ => {
            encoder.set_fragment_buffer(Some(&counter), 0, 0).unwrap();
            let drawn = encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0);
            assert_eq!(drawn, Ok(()));
        }
    };
    warm(&device, |encoder| {
        switching_rectangles(encoder, writing, |_| {}, drawn)
    });
    clear(&counter);
    let ran = run(&device, |encoder| {
        switching_rectangles(encoder, writing, |_| {}, drawn)
    });
    assert_eq!(ran.counted(rebound_count), 0);
    assert_eq!(ran.counted(|counts| counts.draws_as_before), 0);
    assert_eq!(ran.counted(|counts| counts.draws_unkept_writing), DRAWS / 2);
    let counted = DRAWS / 2 * SIDE * SIDE;
    assert_eq!(words_of(&counter), [counted as u32]);
}

/// Draws that switch between two pipeline states, with a texture bound before each draw but the
/// first at fragment texture 1, which neither state's functions read (another texture each time):
/// a bind state holds no texture slot its state does not read (`argument_table::ReadSlots`), so
/// every draw after each state's first is still rebound (`WorkCounts::draws_rebound`), none unlike
/// its kept draw or stopped, with the same commands of every kind and pixels as the same draws
/// each taking every step.
#[test]
fn draws_rebind_over_binds_at_a_slot_no_function_reads() {
    let scratch = Scratch::new("draw-again-unread");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let (first, second) = (adding_state(&device), adding_state(&device));
    let columns = rectangles_from_columns(&device);
    let pair = [&first, &second];
    let states = (&pair[..], &columns);
    let textures = textures(&device);
    let bound = std::cell::Cell::new(0);
    let unread = |encoder: &mut RenderCommandEncoder| {
        let texture = &textures[bound.replace(bound.get() + 1) % textures.len()];
        encoder.set_fragment_texture(Some(texture), 1).unwrap();
    };
    let forgets = |encoder: &mut RenderCommandEncoder| {
        unread(encoder);
        encoder.set_tessellation_factor_scale(1.0);
    };
    let drawn = |encoder: &mut RenderCommandEncoder, _| rectangle(encoder);
    warm(&device, |encoder| {
        switching_rectangles(encoder, states, unread, drawn)
    });
    let rebound_count = |counts: &WorkCounts| counts.draws_rebound;
    let rebound = run(&device, |encoder| {
        switching_rectangles(encoder, states, unread, drawn)
    });
    let every = run(&device, |encoder| {
        switching_rectangles(encoder, states, forgets, drawn)
    });
    assert_eq!(rebound.counted(rebound_count), drawn_alone(DRAWS - 2));
    assert_eq!(every.counted(rebound_count), 0);
    assert_eq!(rebound.commands, every.commands);
    assert_eq!(rebound.words, every.words, "the pixels of the draws");
    let missed = |counts: &WorkCounts| counts.draws_unlike_kept + counts.draws_rebind_stopped;
    assert_eq!(rebound.counted(missed), 0);
}

/// [`DRAWS`] rectangles of the columns [`rectangles_from_columns`] holds, each as the six points of
/// its two triangles (`tob_triangle`'s, in `tob_rect`'s order), [`POINTS`] bytes each.
fn rectangle_points(device: &Device) -> Buffer {
    let points: Vec<f32> = (0..DRAWS)
        .flat_map(|at| {
            let (x0, x1) = (-1.0 + at as f32 * 2.0 / SIDE as f32, 1.0);
            [x0, -1.0, x1, -1.0, x0, 1.0, x1, -1.0, x1, 1.0, x0, 1.0]
        })
        .collect();
    let bytes = bytes_of(&points, f32::to_le_bytes);
    device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap()
}

/// The bytes of a rectangle's points ([`rectangle_points`]).
const POINTS: u64 = 48;

/// [`DRAWS`] draws of `state`'s rectangles (`tob_triangle`'s) from `points`, the `at`th at
/// `POINTS * at` bytes of it (vertex buffer 0, its offset moved before each draw but the first),
/// in the color at `16 * at` bytes of `colors` (fragment buffer 0, bound again at that offset
/// before each draw but the first), with `between` called after the binds.
fn moved_rectangles(
    encoder: &mut RenderCommandEncoder,
    (state, points, colors): (&RenderPipelineState, &Buffer, &Buffer),
    between: impl Fn(&mut RenderCommandEncoder),
) {
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_buffer(Some(points), 0, 0).unwrap();
    encoder.set_fragment_buffer(Some(colors), 0, 0).unwrap();
    for at in 0..DRAWS {
        if at > 0 {
            encoder.set_vertex_buffer_offset(POINTS * at, 0).unwrap();
            encoder
                .set_fragment_buffer(Some(colors), 16 * at, 0)
                .unwrap();
            between(encoder);
        }
        rectangle(encoder);
    }
}

/// Draws after calls that moved only the offsets of the buffers bound at their slots, the vertex
/// buffer's and the fragment buffer's (`set_vertex_buffer_offset`, and `set_fragment_buffer` of
/// the buffer bound there), are moved (`WorkCounts::draws_moved`: their tables bind the objects
/// the draw before bound, and their uses differ only at the moved slots'), with the same commands
/// of every kind, state commands, noted uses and pixels as the same draws with a call between each
/// that changes nothing for them, each of which takes every step; each at another offset of both
/// buffers, in another color each, as each column's count of draws and each draw's color show.
/// The vertex function reads its points through a pointer, from the offset to the buffer's end, so
/// each draw's use of them lies inside the draw before's, noted already
/// (`WorkCounts::uses_noted_covered`).
#[test]
fn draws_after_offsets_alone_moved_bind_what_every_step_binds() {
    let scratch = Scratch::new("draw-again-moved");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = state(
        &device,
        "engine/tob_renders.ll",
        ("tob_triangle", "tob_solid"),
    );
    let rectangles = rectangle_points(&device);
    let colors: Vec<f32> = (0..DRAWS)
        .flat_map(|at| [(at + 1) as f32 / 64.0; 4])
        .collect();
    let colors = bytes_of(&colors, f32::to_le_bytes);
    let colors = (device.new_buffer_with_bytes(&colors, ResourceOptions::shared())).unwrap();
    let drawn = (&state, &rectangles, &colors);
    let forgets = |encoder: &mut RenderCommandEncoder| encoder.set_tessellation_factor_scale(1.0);
    warm(&device, |encoder| moved_rectangles(encoder, drawn, |_| {}));
    let rebound = run(&device, |encoder| moved_rectangles(encoder, drawn, |_| {}));
    let every = run(&device, |encoder| moved_rectangles(encoder, drawn, forgets));
    let moved_count = |counts: &WorkCounts| counts.draws_moved;
    assert_eq!(rebound.counted(moved_count), drawn_alone(HITS));
    assert_eq!(
        rebound.counted(|counts| counts.draws_rebound),
        drawn_alone(HITS)
    );
    assert_eq!(every.counted(moved_count), 0);
    assert_eq!(rebound.commands, every.commands);
    assert_eq!(rebound.commands["cmd_draw"], DRAWS);
    assert_eq!(rebound.words, every.words, "the pixels of the draws");
    let counts: [(&str, fn(&WorkCounts) -> u64); 4] = [
        ("draw_states_recorded", |counts| counts.draw_states_recorded),
        ("commands_noted_as_before", |counts| {
            counts.commands_noted_as_before
        }),
        ("draws_pushing_set_zero", |counts| {
            counts.draws_pushing_set_zero
        }),
        ("uses_noted_covered", |counts| counts.uses_noted_covered),
    ];
    // Each draw's vertex buffer use, from its offset to the buffer's end, lies inside the first
    // draw's: noted already (`noted_apart.rs`).
    let covered = rebound.counted(|counts| counts.uses_noted_covered);
    assert!(covered >= DRAWS - 1, "{covered} uses noted covered");
    for (name, count) in counts {
        assert_eq!(rebound.counted(count), every.counted(count), "{name}");
    }
    // Column 0 holds the first draw's color alone; the last column every draw's.
    let last = SIDE as usize - 1;
    assert_ne!(rebound.words[0], rebound.words[last], "the draws' offsets");
}

/// Draws after calls that moved only the vertex buffer's offset along the buffer are moved inside
/// (`WorkCounts::draws_moved_inside`): the vertex function reads its points through a pointer,
/// from the offset to the buffer's end, so each draw's use of them lies inside the draw before's,
/// noted, read and gathered already, and the draw compares, notes, adds and gathers none of it;
/// with the same commands of every kind, state commands and pixels as the same draws with a call
/// between each that changes nothing for them, each of which takes every step and notes the use
/// covered (`WorkCounts::uses_noted_covered`). The fragment buffer stays bound as it was.
#[test]
fn draws_moved_along_their_buffer_note_nothing_again() {
    let scratch = Scratch::new("draw-again-inside");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = state(
        &device,
        "engine/tob_renders.ll",
        ("tob_triangle", "tob_solid"),
    );
    let rectangles = rectangle_points(&device);
    let color = bytes_of(&[0.25f32; 4], f32::to_le_bytes);
    let colors = (device.new_buffer_with_bytes(&color, ResourceOptions::shared())).unwrap();
    let along = |encoder: &mut RenderCommandEncoder,
                 between: &dyn Fn(&mut RenderCommandEncoder)| {
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder.set_vertex_buffer(Some(&rectangles), 0, 0).unwrap();
        encoder.set_fragment_buffer(Some(&colors), 0, 0).unwrap();
        for at in 0..DRAWS {
            if at > 0 {
                encoder.set_vertex_buffer_offset(POINTS * at, 0).unwrap();
                between(encoder);
            }
            rectangle(encoder);
        }
    };
    let nothing = |_: &mut RenderCommandEncoder| {};
    let forgets = |encoder: &mut RenderCommandEncoder| encoder.set_tessellation_factor_scale(1.0);
    warm(&device, |encoder| along(encoder, &nothing));
    let rebound = run(&device, |encoder| along(encoder, &nothing));
    let every = run(&device, |encoder| along(encoder, &forgets));
    let inside = rebound.counted(|counts| counts.draws_moved_inside);
    assert_eq!(
        rebound.counted(|counts| counts.draws_moved),
        drawn_alone(HITS)
    );
    assert_eq!(inside, drawn_alone(HITS), "every draw moved is inside");
    assert_eq!(every.counted(|counts| counts.draws_moved_inside), 0);
    assert_eq!(rebound.commands, every.commands);
    assert_eq!(rebound.commands["cmd_draw"], DRAWS);
    assert_eq!(rebound.words, every.words, "the pixels of the draws");
    // Every step notes each draw's use but the first's covered; a draw inside compares none.
    let covered = |counts: &WorkCounts| counts.uses_noted_covered;
    assert_eq!(every.counted(covered), DRAWS - 1);
    assert_eq!(rebound.counted(covered) + inside, every.counted(covered));
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
        assert_eq!(rebound.counted(count), every.counted(count), "{name}");
    }
    // Column 0 holds the first draw's rectangle alone; the last column every draw's.
    let last = SIDE as usize - 1;
    assert_ne!(rebound.words[0], rebound.words[last], "the draws' offsets");
}

/// [`DRAWS`] 4x4 `RGBA8Unorm` textures, the `at`th's texel in column `x` of the bytes
/// `(2 * (at + 1), 2 * (x + 1), 0, 1)`: what a draw sampled tells which texture and which of its
/// columns.
fn textures(device: &Device) -> Vec<Texture> {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    (0..DRAWS)
        .map(|at| {
            let texture = device.new_texture(&descriptor).unwrap();
            let texels: Vec<u8> = (0..16u8)
                .flat_map(|texel| [2 * (at as u8 + 1), 2 * (texel % 4 + 1), 0, 1])
                .collect();
            let whole = Region::new_2d(0, 0, 4, 4);
            texture.replace_region(whole, 0, 0, &texels, 16, 0).unwrap();
            texture
        })
        .collect()
}

/// A sampler state of nearest filtering that addresses outside a texture as `mode` does, its
/// levels clamped to `[0, lod_max]`.
fn sampler_addressing(device: &Device, mode: SamplerAddressMode, lod_max: f32) -> SamplerState {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.s_address_mode = mode;
    descriptor.t_address_mode = mode;
    descriptor.lod_max_clamp = lod_max;
    device.new_sampler_state(&descriptor).unwrap()
}

/// [`DRAWS`] draws of `state`'s rectangles from `rectangles` as [`offset_rectangles`] draws them,
/// the `at`th sampling `textures[texture(at)]` through `samplers[sampler(at)]` (fragment texture
/// and sampler 0, set before each draw), with `between` called after them.
fn sampled_rectangles(
    encoder: &mut RenderCommandEncoder,
    (state, rectangles): (&RenderPipelineState, &Buffer),
    (textures, samplers): (&[Texture], &[SamplerState]),
    (texture, sampler): (&dyn Fn(u64) -> usize, &dyn Fn(u64) -> usize),
    between: impl Fn(&mut RenderCommandEncoder),
) {
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_buffer(Some(rectangles), 0, 0).unwrap();
    for at in 0..DRAWS {
        if at > 0 {
            encoder.set_vertex_buffer_offset(16 * at, 0).unwrap();
            between(encoder);
        }
        let bound = &textures[texture(at)];
        encoder.set_fragment_texture(Some(bound), 0).unwrap();
        let bound = &samplers[sampler(at)];
        encoder.set_fragment_sampler_state(Some(bound), 0).unwrap();
        rectangle(encoder);
    }
}

/// Draws rebound after their vertex buffer's offset alone moved patch that slot's use in place,
/// looking at no other use (`binding::patch_buffer_uses`, which finds the uses of buffer slots in
/// the list it makes of them at the first patch after `binding::command_uses_into` made them):
/// each draw of `tob_sample`'s rectangles
/// also samples a texture, bound once before them, whose use no patch looks at. Builds with debug
/// assertions, the only builds of this test, count the uses a patch looked at
/// (`binding::patched_slot_uses`) and check the list against the uses.
#[cfg(debug_assertions)]
#[test]
fn draws_rebound_after_an_offset_moved_patch_that_slot_s_use_alone() {
    let scratch = Scratch::new("draw-again-patched-slot-uses");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = state(&device, "engine/tob_renders.ll", ("tob_rect", "tob_sample"));
    let rectangles = rectangles_from_columns(&device);
    let textures = textures(&device);
    let sampler = sampler_addressing(&device, SamplerAddressMode::ClampToEdge, f32::MAX);
    let draws = |encoder: &mut RenderCommandEncoder| {
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder.set_vertex_buffer(Some(&rectangles), 0, 0).unwrap();
        encoder.set_fragment_texture(Some(&textures[0]), 0).unwrap();
        encoder
            .set_fragment_sampler_state(Some(&sampler), 0)
            .unwrap();
        for at in 0..DRAWS {
            if at > 0 {
                encoder.set_vertex_buffer_offset(16 * at, 0).unwrap();
            }
            rectangle(encoder);
        }
    };
    warm(&device, draws);
    let before = (crate::binding::patched_slot_uses(), device.work_counts());
    run(&device, draws);
    let after = (crate::binding::patched_slot_uses(), device.work_counts());
    let (calls, walked) = (after.0 .0 - before.0 .0, after.0 .1 - before.0 .1);
    let rebound = after.1.draws_rebound - before.1.draws_rebound;
    assert_eq!(rebound, drawn_alone(HITS));
    assert!(calls >= rebound, "{calls} patches, {rebound} rebound");
    assert_eq!(
        walked, calls,
        "each patch looks at the vertex buffer's use alone"
    );
}

/// Draws rebound after their fragment texture alone changed, another at each draw, make their
/// uses again (`render/draw_again/rebound.rs`) and so list none of their uses of buffer slots: a
/// draw lists those at the first patch after its uses were made (`binding::list_slot_uses`), which
/// only a draw whose buffer slots alone changed asks for. Listed as the uses are made instead,
/// each of these draws would list them for no patch to read. Builds with debug assertions, the
/// only builds of this test, count the listings (`binding::listed_slot_uses`).
#[cfg(debug_assertions)]
#[test]
fn draws_rebound_after_their_texture_changed_list_no_buffer_slot_uses() {
    let scratch = Scratch::new("draw-again-listed-slot-uses");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = state(&device, "engine/tob_renders.ll", ("tob_rect", "tob_sample"));
    let rectangles = rectangles_from_columns(&device);
    let textures = textures(&device);
    let samplers = [sampler_addressing(
        &device,
        SamplerAddressMode::ClampToEdge,
        f32::MAX,
    )];
    let drawn = (&state, &rectangles);
    let bound = (&textures[..], &samplers[..]);
    let each = |at| at as usize;
    let first = |_| 0;
    let draws = |encoder: &mut RenderCommandEncoder| {
        sampled_rectangles(encoder, drawn, bound, (&each, &first), |_| {})
    };
    warm(&device, draws);
    let before = crate::binding::listed_slot_uses();
    let ran = run(&device, draws);
    let listed = crate::binding::listed_slot_uses();
    let rebound = ran.counted(|counts| counts.draws_rebound);
    assert_eq!(rebound, drawn_alone(HITS));
    assert_eq!(
        listed, before,
        "draws whose uses were made again listed their uses of buffer slots"
    );
}

/// Draws after calls that bound other textures and sampler states at the slots their fragment
/// function samples (`tob_sample`, which samples its 4x4 texture at the pixel's position over the
/// texture's size: inside it in the target's first four columns, outside it in the others) rebind
/// those slots alone (`WorkCounts::draws_rebound`), with the same commands of every kind, state
/// commands and noted uses, and the same pixels, as the same draws with a call between each that
/// changes nothing for them, each of which takes every step: another texture each, as their
/// pixels, unlike those of the same draws of one texture, show; another sampler state each of the
/// same shape (levels clamped to 1000 or to `FLT_MAX`, a `VkSampler` each), whose pixels are
/// those of one; and both. Draws through samplers of another addressing each (clamped to the edge
/// or repeated, which the pipeline's specialization reads) take every step.
#[test]
fn draws_after_texture_and_sampler_changes_rebind_what_every_step_binds() {
    let scratch = Scratch::new("draw-again-rebound-textures");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = state(&device, "engine/tob_renders.ll", ("tob_rect", "tob_sample"));
    let rectangles = rectangles_from_columns(&device);
    let textures = textures(&device);
    let samplers = [
        sampler_addressing(&device, SamplerAddressMode::ClampToEdge, f32::MAX),
        sampler_addressing(&device, SamplerAddressMode::ClampToEdge, 1000.0),
        sampler_addressing(&device, SamplerAddressMode::Repeat, f32::MAX),
    ];
    let forgets = |encoder: &mut RenderCommandEncoder| encoder.set_tessellation_factor_scale(1.0);
    let drawn = (&state, &rectangles);
    let bound = (&textures[..], &samplers[..]);
    let first = |_| 0;
    let each = |at| at as usize;
    let alternate = |at| at as usize % 2;
    let addressing = |at| 2 * (at as usize % 2);
    let warming: [(&dyn Fn(u64) -> usize, &dyn Fn(u64) -> usize); 2] =
        [(&each, &alternate), (&first, &addressing)];
    for chosen in warming {
        warm(&device, |encoder| {
            sampled_rectangles(encoder, drawn, bound, chosen, |_| {})
        });
    }
    let rebound_count = |counts: &WorkCounts| counts.draws_rebound;
    let as_before = |counts: &WorkCounts| counts.draws_as_before;
    let one = run(&device, |encoder| {
        sampled_rectangles(encoder, drawn, bound, (&first, &first), |_| {});
    });
    // Which draws, whether they rebind, and whether their pixels are one texture's.
    let choices: [(&dyn Fn(u64) -> usize, &dyn Fn(u64) -> usize, bool, bool); 4] = [
        (&each, &first, true, false),
        (&first, &alternate, true, true),
        (&each, &alternate, true, false),
        (&first, &addressing, false, false),
    ];
    for (texture, sampler, rebinds, one_texture) in choices {
        let chosen = (texture, sampler);
        let rebound = run(&device, |encoder| {
            sampled_rectangles(encoder, drawn, bound, chosen, |_| {});
        });
        let every = run(&device, |encoder| {
            sampled_rectangles(encoder, drawn, bound, chosen, forgets);
        });
        let expected = match rebinds {
            true => drawn_alone(HITS),
            false => 0,
        };
        assert_eq!(rebound.counted(rebound_count), expected);
        assert_eq!(rebound.counted(as_before), 0);
        assert_eq!(every.counted(rebound_count), 0);
        assert_eq!(rebound.commands, every.commands);
        assert_eq!(rebound.commands["cmd_draw"], DRAWS);
        assert_eq!(rebound.words, every.words, "the pixels of the draws");
        assert_eq!(
            rebound.words == one.words,
            one_texture,
            "the draws' bindings"
        );
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
            assert_eq!(rebound.counted(count), every.counted(count), "{name}");
        }
    }
}

/// What each of `draws` draws of `state`'s rectangles from `rectangles` (as
/// [`offset_rectangles`]' in one color, the `at`th of the rectangle `at` modulo [`DRAWS`], and
/// `between(at)` called before the `at`th draw but the first) changed of the device's counts `of`
/// and of the compares of dynamic states' values the encoding thread made (`render/bound.rs`),
/// the first draw's left out.
#[cfg(debug_assertions)]
fn compared_by_each(
    device: &Device,
    (state, rectangles, draws): (&RenderPipelineState, &Buffer, u64),
    between: impl Fn(&mut RenderCommandEncoder, u64),
    of: fn(&WorkCounts) -> u64,
) -> (Ran, Vec<(u64, u64)>) {
    let drawn = std::cell::RefCell::new(Vec::new());
    let ran = run(device, |encoder| {
        drawn.borrow_mut().clear();
        encoder.set_render_pipeline_state(state).unwrap();
        encoder.set_vertex_buffer(Some(rectangles), 0, 0).unwrap();
        let color = bytes_of(&[1.0f32 / 64.0; 4], f32::to_le_bytes);
        encoder.set_fragment_bytes(&color, 0).unwrap();
        rectangle(encoder);
        for at in 1..draws {
            encoder
                .set_vertex_buffer_offset(16 * (at % DRAWS), 0)
                .unwrap();
            between(encoder, at);
            let before = (of(&device.work_counts()), dynamic_compares());
            rectangle(encoder);
            let after = (of(&device.work_counts()), dynamic_compares());
            drawn
                .borrow_mut()
                .push((after.0 - before.0, after.1 - before.1));
        }
    });
    (ran, drawn.into_inner())
}

/// A rebound draw compares none of the dynamic states a record of its command buffer holds a
/// value of with it: nothing recorded through the record since the kept draw, or the draw rebound
/// before, set them, so it holds them as they set them (`Kept::settled`); a draw that takes every
/// step compares them.
#[test]
#[cfg(debug_assertions)]
fn draws_rebound_compare_no_dynamic_state_their_record_holds() {
    let scratch = Scratch::new("rebound-compares");
    let device = device(&scratch.0, &[], &Heard::default());
    let state = adding_state(&device);
    let columns = rectangles_from_columns(&device);
    let drawn = (&state, &columns);
    let rebound = |counts: &WorkCounts| counts.draws_rebound;
    warm(&device, |encoder| {
        offset_rectangles(encoder, drawn, |_| 1.0 / 64.0, |_| {})
    });
    let all = (&state, &columns, DRAWS);
    let (_, each) = compared_by_each(&device, all, |_, _| {}, rebound);
    let rebound_draws: Vec<_> = each.iter().filter(|(rebound, _)| *rebound == 1).collect();
    assert!(
        rebound_draws.len() as u64 >= (DRAWS - 1) / 2 - 1,
        "draws rebound: {each:?}"
    );
    assert!(
        rebound_draws.iter().all(|(_, compared)| *compared == 0),
        "dynamic states compared by draws rebound (rebound, compared): {each:?}"
    );
    assert!(
        (each.iter()).any(|&(rebound, compared)| rebound == 0 && compared > 0),
        "a draw taking every step compares its dynamic states: {each:?}"
    );
}

/// Draws after something recorded into their command buffer between them, or after a call that
/// changed a dynamic state, set what the same draws each taking every step set: of 15 draws,
/// after another blend color before the 3rd (the state set again), a barrier inside the pass
/// before the 7th, and a barrier that splits the pass before the 11th (another command buffer,
/// whose record begins empty), the draws rebound between them, with the same commands of every
/// kind, state commands and pixels. Each draw after the blend color or the barrier inside the
/// pass compares its dynamic states again, and the one after the blend color records a state
/// command; the draws after the split record after their calls (none is recorded as it is
/// called), so their commands and pixels alone show what they set.
#[test]
#[cfg(debug_assertions)]
fn draws_after_a_state_set_or_a_barrier_set_what_every_step_sets() {
    use metal2vulkan::metal::{BarrierScope, RenderStages};
    let scratch = Scratch::new("rebound-after-recordings");
    let device = device(&scratch.0, &[], &Heard::default());
    let state = adding_state(&device);
    let columns = rectangles_from_columns(&device);
    let drawn = (&state, &columns);
    let (fragment, vertex) = (RenderStages::FRAGMENT, RenderStages::VERTEX);
    // Apart enough that draws rebound come between them in builds with debug assertions too
    // (where a kept draw's first draw again takes every step).
    let came = |encoder: &mut RenderCommandEncoder, at: u64| match at {
        3 => encoder.set_blend_color(0.25, 0.25, 0.25, 0.25),
        7 => encoder.memory_barrier_with_scope(BarrierScope::RENDER_TARGETS, fragment, fragment),
        11 => encoder.memory_barrier_with_scope(BarrierScope::BUFFERS, fragment, vertex),
        _ => {}
    };
    // And a call no draw of the state reads before each draw: every step at each.
    let every_step = |encoder: &mut RenderCommandEncoder, at: u64| {
        came(encoder, at);
        encoder.set_tessellation_factor_scale(1.0);
    };
    let states = |counts: &WorkCounts| counts.draw_states_recorded;
    warm(&device, |encoder| {
        offset_rectangles(encoder, drawn, |_| 1.0 / 64.0, |_| {})
    });
    let fifteen = (&state, &columns, 15);
    let (between, each) = compared_by_each(&device, fifteen, came, states);
    let (every, _) = compared_by_each(&device, fifteen, every_step, states);
    let rebound = |counts: &WorkCounts| counts.draws_rebound;
    assert!(between.counted(rebound) > 0, "no draw rebound");
    assert_eq!(every.counted(rebound), 0);
    assert_eq!(between.commands, every.commands);
    assert_eq!(between.counted(states), every.counted(states));
    assert_eq!(between.words, every.words, "the pixels of the draws");
    // `each[at - 1]` is the `at`th draw's.
    for at in [3, 7] {
        let (_, compared) = each[at - 1];
        assert!(
            compared > 0,
            "the draw after {at}'s call compared nothing: {each:?}"
        );
    }
    assert!(
        each[2].0 > 0,
        "the draw after another blend color set none: {each:?}"
    );
}

/// Indexed draws after the gaps a compositor's draws come after (a guest's window server's): a
/// vertex buffer offset alone, a scissor rectangle and the offset, or a switch to the other of
/// two states of one function pair with another texture and the offset, in turn. Every draw but
/// the pass's first and the second state's first is rebound (`WorkCounts::draws_rebound`): after
/// the scissor rectangle restated (`render/draw_again/restated.rs`), after the switch from the
/// state's own kept draw; with the same commands of every kind, state commands and pixels as the
/// same draws each taking every step, after a call between each that changes nothing for them
/// (`set_tessellation_factor_scale`, which no draw of their states reads). Every draw keeps
/// itself (`WorkCounts::draws_unkept`), the pass's first, which begins its instance, too, and no
/// draw finds its kept draw unlike it or stops its rebind.
#[test]
fn draws_after_a_compositor_s_gaps_rebind_what_every_step_binds() {
    let scratch = Scratch::new("draw-again-gaps");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let sampling = || state(&device, "engine/tob_renders.ll", ("tob_rect", "tob_sample"));
    let pair = [sampling(), sampling()];
    let rectangles = rectangles_from_columns(&device);
    let textures = textures(&device);
    let sampler = sampler_addressing(&device, SamplerAddressMode::ClampToEdge, f32::MAX);
    let indices = bytes_of(&[0u16, 1, 2, 3, 4, 5], u16::to_le_bytes);
    let index = device
        .new_buffer_with_bytes(&indices, ResourceOptions::shared())
        .unwrap();
    let half = |x| ScissorRect {
        x,
        y: 0,
        width: SIDE / 2,
        height: SIDE,
    };
    let halves = [half(0), half(SIDE / 2)];
    let gaps = |encoder: &mut RenderCommandEncoder, between: &dyn Fn(&mut RenderCommandEncoder)| {
        encoder.set_render_pipeline_state(&pair[0]).unwrap();
        encoder.set_vertex_buffer(Some(&rectangles), 0, 0).unwrap();
        encoder.set_fragment_texture(Some(&textures[0]), 0).unwrap();
        encoder
            .set_fragment_sampler_state(Some(&sampler), 0)
            .unwrap();
        let mut drawing = 0;
        for at in 0..DRAWS {
            if at > 0 {
                encoder.set_vertex_buffer_offset(16 * at, 0).unwrap();
                match at % 3 {
                    1 => (encoder.set_scissor_rect(halves[(at / 3) as usize % 2])).unwrap(),
                    2 => {
                        drawing = 1 - drawing;
                        encoder.set_render_pipeline_state(&pair[drawing]).unwrap();
                        let texture = &textures[at as usize];
                        encoder.set_fragment_texture(Some(texture), 0).unwrap();
                    }
                    _ => {}
                }
                between(encoder);
            }
            let drawn = encoder.draw_indexed_primitives(
                PrimitiveType::Triangle,
                6,
                IndexType::UInt16,
                &index,
                0,
                1,
                0,
                0,
            );
            assert_eq!(drawn, Ok(()));
        }
    };
    let forgets = |encoder: &mut RenderCommandEncoder| encoder.set_tessellation_factor_scale(1.0);
    warm(&device, |encoder| gaps(encoder, &|_| {}));
    let rebound = run(&device, |encoder| gaps(encoder, &|_| {}));
    let every = run(&device, |encoder| gaps(encoder, &forgets));
    let rebound_count = |counts: &WorkCounts| counts.draws_rebound;
    let states_recorded = |counts: &WorkCounts| counts.draw_states_recorded;
    assert_eq!(rebound.counted(rebound_count), drawn_alone(DRAWS - 2));
    assert!(rebound.counted(|counts| counts.draws_restated) > 0);
    assert_eq!(every.counted(rebound_count), 0);
    assert_eq!(rebound.commands, every.commands);
    assert_eq!(rebound.commands["cmd_draw_indexed"], DRAWS);
    assert_eq!(
        rebound.counted(states_recorded),
        every.counted(states_recorded)
    );
    assert_eq!(rebound.words, every.words, "the pixels of the draws");
    assert_eq!(rebound.counted(|counts| counts.draws_unkept), 0);
    let missed = |counts: &WorkCounts| counts.draws_unlike_kept + counts.draws_rebind_stopped;
    assert_eq!(rebound.counted(missed), 0);
}

/// A draw unlike the draw its encoder kept is counted in each part it differed by
/// (`WorkCounts::draws_unlike_kept_*`): the pass's first draw kept, a draw of other counts alone,
/// which is rebound with them (`draws_rebound_arguments`), one of another primitive type, which
/// only its assembly sets apart from that one, one of other counts alone that makes no primitive
/// where that one makes some, whose route they change, so its rebind stops
/// (`draws_rebind_stopped_arguments_route`), and one of other counts alone that draws nothing,
/// which takes every step (`draws_unlike_kept_arguments_alone`).
#[test]
fn a_draw_unlike_its_kept_draw_is_counted_by_what_differed() {
    let scratch = Scratch::new("draw-again-unlike");
    let device = device(&scratch.0, &[], &Heard::default());
    let state = adding_state(&device);
    let draws = |encoder: &mut RenderCommandEncoder| {
        set_rectangle(encoder, &state);
        rectangle(encoder);
        // Its first triangle, then the same three vertices as a strip, then a strip of two
        // vertices, which makes no triangle, then one of none.
        for (primitive, count) in [
            (PrimitiveType::Triangle, 3),
            (PrimitiveType::TriangleStrip, 3),
            (PrimitiveType::TriangleStrip, 2),
            (PrimitiveType::TriangleStrip, 0),
        ] {
            let drawn = encoder.draw_primitives(primitive, 0, count, 1, 0);
            assert_eq!(drawn, Ok(()));
        }
    };
    warm(&device, draws);
    let ran = run(&device, draws);
    let parts: [(&str, fn(&WorkCounts) -> u64, u64); 14] = [
        ("unlike", |c| c.draws_unlike_kept, 2),
        ("counts", |c| c.draws_unlike_kept_counts, 1),
        (
            "arguments alone",
            |c| c.draws_unlike_kept_arguments_alone,
            1,
        ),
        ("call", |c| c.draws_unlike_kept_call, 1),
        ("pipeline", |c| c.draws_unlike_kept_pipeline, 0),
        ("index buffer", |c| c.draws_unlike_kept_index_buffer, 0),
        ("index offset", |c| c.draws_unlike_kept_index_offset, 0),
        ("tessellation", |c| c.draws_unlike_kept_tessellation, 0),
        ("command buffer", |c| c.draws_unlike_kept_command_buffer, 0),
        ("under", |c| c.draws_unlike_kept_under, 0),
        (
            "rebound with other arguments",
            |c| c.draws_rebound_arguments,
            drawn_alone(1),
        ),
        (
            "stopped at a step of their arguments",
            |c| c.draws_rebind_stopped_arguments,
            1,
        ),
        (
            "stopped at their route",
            |c| c.draws_rebind_stopped_arguments_route,
            1,
        ),
        ("stopped", |c| c.draws_rebind_stopped, 1),
    ];
    for (part, count, expected) in parts {
        assert_eq!(
            ran.counted(count),
            expected,
            "draws unlike their kept draw: {part}"
        );
    }
}

/// Indexed draws that differ from the draw before only in their first index (their index
/// buffer's offset) are rebound with it (`WorkCounts::draws_rebound_arguments`): their index
/// buffer bound at their offset and their Vulkan draw, with the same commands of every kind, state
/// commands and pixels as the same draws with a call between each that changes nothing for them,
/// each of which takes every step; the `at`th draws the `at`th rectangle's indices
/// ([`rectangle_points`]), as each column's count of draws shows. So are draws that differ only in
/// their base vertex, of the same indices, the `at`th from the `at`th rectangle's first point:
/// their uses are the draw before's, so each is inside (`WorkCounts::draws_moved_inside`) and
/// compares none of them, where every step notes them as before.
#[test]
fn draws_of_another_first_index_alone_are_rebound_with_their_pixels() {
    let scratch = Scratch::new("draw-again-first-index");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = state(
        &device,
        "engine/tob_renders.ll",
        ("tob_triangle", "tob_solid"),
    );
    let rectangles = rectangle_points(&device);
    let color = bytes_of(&[1.0f32 / 64.0; 4], f32::to_le_bytes);
    let colors = (device.new_buffer_with_bytes(&color, ResourceOptions::shared())).unwrap();
    // The `at`th rectangle's six indices, at `12 * at` bytes.
    let indices: Vec<u16> = (0..6 * DRAWS as u16).collect();
    let indices = bytes_of(&indices, u16::to_le_bytes);
    let index = (device.new_buffer_with_bytes(&indices, ResourceOptions::shared())).unwrap();
    type Arguments = (fn(u64) -> u64, fn(u64) -> i32);
    let draws = |encoder: &mut RenderCommandEncoder,
                 (offset, base): Arguments,
                 between: &dyn Fn(&mut RenderCommandEncoder)| {
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder.set_vertex_buffer(Some(&rectangles), 0, 0).unwrap();
        encoder.set_fragment_buffer(Some(&colors), 0, 0).unwrap();
        for at in 0..DRAWS {
            if at > 0 {
                between(encoder);
            }
            let drawn = encoder.draw_indexed_primitives(
                PrimitiveType::Triangle,
                6,
                IndexType::UInt16,
                &index,
                offset(at),
                1,
                base(at),
                0,
            );
            assert_eq!(drawn, Ok(()));
        }
    };
    let nothing = |_: &mut RenderCommandEncoder| {};
    let forgets = |encoder: &mut RenderCommandEncoder| encoder.set_tessellation_factor_scale(1.0);
    let first_indices: Arguments = (|at| 12 * at, |_| 0);
    let base_vertices: Arguments = (|_| 0, |at| 6 * at as i32);
    for (form, arguments, within) in [
        ("first index", first_indices, 0),
        ("base vertex", base_vertices, drawn_alone(HITS)),
    ] {
        warm(&device, |encoder| draws(encoder, arguments, &nothing));
        let rebound = run(&device, |encoder| draws(encoder, arguments, &nothing));
        let every = run(&device, |encoder| draws(encoder, arguments, &forgets));
        let reargued = |counts: &WorkCounts| counts.draws_rebound_arguments;
        assert_eq!(rebound.counted(reargued), drawn_alone(HITS), "{form}");
        assert_eq!(
            rebound.counted(|counts| counts.draws_rebound),
            drawn_alone(HITS),
            "{form}"
        );
        let missed = |counts: &WorkCounts| counts.draws_unlike_kept + counts.draws_rebind_stopped;
        assert_eq!(rebound.counted(missed), 0, "{form}");
        assert_eq!(every.counted(reargued), 0, "{form}");
        assert_eq!(rebound.commands, every.commands, "{form}");
        assert_eq!(rebound.commands["cmd_draw_indexed"], DRAWS, "{form}");
        assert_eq!(
            rebound.words, every.words,
            "the pixels of the draws: {form}"
        );
        let counts: [(&str, fn(&WorkCounts) -> u64); 2] = [
            ("draw_states_recorded", |counts| counts.draw_states_recorded),
            ("draws_pushing_set_zero", |counts| {
                counts.draws_pushing_set_zero
            }),
        ];
        for (name, count) in counts {
            assert_eq!(
                rebound.counted(count),
                every.counted(count),
                "{form}: {name}"
            );
        }
        // A draw inside compares none of its uses; every step notes them as before.
        let inside = rebound.counted(|counts| counts.draws_moved_inside);
        assert_eq!(inside, within, "{form}: draws inside");
        let as_before = |counts: &WorkCounts| counts.commands_noted_as_before;
        assert_eq!(
            rebound.counted(as_before) + inside,
            every.counted(as_before),
            "{form}: commands_noted_as_before"
        );
        // Column 0 holds the first draw's rectangle alone; the last column every draw's.
        let last = SIDE as usize - 1;
        assert_ne!(rebound.words[0], rebound.words[last], "the draws' {form}");
    }
}

/// Draws with indices and without in turn, from two index buffers in turn, each of the next
/// rectangle ([`rectangle_points`]: the `at`th's six indices at `12 * at` bytes, or its six
/// vertices from `6 * at`), are rebound with their call (`WorkCounts::draws_rebound_arguments`):
/// their index buffer bound where they read one and their Vulkan draw, with the same commands of
/// every kind and pixels as the same draws with a call between each that changes nothing for them,
/// each of which takes every step.
#[test]
fn draws_with_indices_and_without_in_turn_are_rebound_with_their_call() {
    let scratch = Scratch::new("draw-again-call");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = state(
        &device,
        "engine/tob_renders.ll",
        ("tob_triangle", "tob_solid"),
    );
    let rectangles = rectangle_points(&device);
    let color = bytes_of(&[1.0f32 / 64.0; 4], f32::to_le_bytes);
    let colors = (device.new_buffer_with_bytes(&color, ResourceOptions::shared())).unwrap();
    let indices: Vec<u16> = (0..6 * DRAWS as u16).collect();
    let indices = bytes_of(&indices, u16::to_le_bytes);
    let index = |_| (device.new_buffer_with_bytes(&indices, ResourceOptions::shared())).unwrap();
    let index: [Buffer; 2] = std::array::from_fn(index);
    let draws = |encoder: &mut RenderCommandEncoder,
                 between: &dyn Fn(&mut RenderCommandEncoder)| {
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder.set_vertex_buffer(Some(&rectangles), 0, 0).unwrap();
        encoder.set_fragment_buffer(Some(&colors), 0, 0).unwrap();
        for at in 0..DRAWS {
            if at > 0 {
                between(encoder);
            }
            let drawn = match at % 3 {
                1 => encoder.draw_primitives(PrimitiveType::Triangle, 6 * at as u32, 6, 1, 0),
                turn => encoder.draw_indexed_primitives(
                    PrimitiveType::Triangle,
                    6,
                    IndexType::UInt16,
                    &index[turn as usize / 2],
                    12 * at,
                    1,
                    0,
                    0,
                ),
            };
            assert_eq!(drawn, Ok(()));
        }
    };
    let nothing = |_: &mut RenderCommandEncoder| {};
    let forgets = |encoder: &mut RenderCommandEncoder| encoder.set_tessellation_factor_scale(1.0);
    warm(&device, |encoder| draws(encoder, &nothing));
    let rebound = run(&device, |encoder| draws(encoder, &nothing));
    let every = run(&device, |encoder| draws(encoder, &forgets));
    let reargued = |counts: &WorkCounts| counts.draws_rebound_arguments;
    assert_eq!(rebound.counted(reargued), drawn_alone(HITS));
    let missed = |counts: &WorkCounts| counts.draws_unlike_kept + counts.draws_rebind_stopped;
    assert_eq!(rebound.counted(missed), 0);
    assert_eq!(every.counted(reargued), 0);
    assert_eq!(rebound.commands, every.commands);
    assert_eq!(rebound.commands["cmd_draw_indexed"], DRAWS - DRAWS / 3);
    assert_eq!(rebound.commands["cmd_draw"], DRAWS / 3);
    assert_eq!(rebound.words, every.words, "the pixels of the draws");
    let last = SIDE as usize - 1;
    assert_ne!(
        rebound.words[0], rebound.words[last],
        "the draws' rectangles"
    );
}
