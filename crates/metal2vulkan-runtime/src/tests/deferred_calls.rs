//! A render encoder's calls lowered after they return (`RenderLoweringPolicy::Worker`,
//! `render/deferral.rs`) against inside each call (`AtCall`): each refusal the encoder checks at
//! the call (`render/call_checks.rs`) returns from the call under both policies, the same error,
//! and the command buffer then runs; a refusal only the lowering finds fails the command buffer,
//! naming the call (`CommandBufferError::CallRefused`), and the command buffer keeps the call's
//! error; a notice announced while a call is lowered names it, and a kind the device refuses while
//! lowering (`DeviceOptions::lowering_refuses`) refuses it; the bytes a call was lent are copied at
//! the call, and the objects it names are held from the call until its command buffer completed;
//! an offset at an index with no buffer bound is taken at the call under both policies, as plain
//! Metal takes it; a buffer detached after the call refuses nothing of it; a program that outruns
//! the device's lowering thread waits for it at the cap of its encoder's batches, and goes on; an
//! end and a commit return before the lowering thread lowered the end, the commit's own work
//! waiting for it, and the commits after it waiting behind it, and one whose own work fails
//! completing with the refusal; a `commit_and_submit` so left is submitted by the lowering thread
//! itself, which runs its submission check and handler, and an encoder opened there lowers its
//! calls at their calls.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{busy, over_host, until, Host, Scratch, FILLED, WAIT};
use super::encoder_holds::{bits, bytes_of, pass, render_state, words_of};
use crate::error::{invalid, vulkan};
use crate::pipeline::pool::held;
use crate::render::{BATCH_CALLS, HANDED_BATCHES};
use crate::StoreAction;
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, Degradation,
    DegradeNotice, Device, DeviceKind, DeviceOptions, EncoderCall, Error, EventRef, FailureOrigin,
    MessageSink, Origin, Problem, RenderCommandEncoder, RenderPipelineState, ResourceOptions,
    ShaderCacheOptions, Size, Submission, Texture, TextureDescriptor,
    VertexAmplificationViewMapping, Viewport,
};
use ash::vk;
use metal2vulkan::device::{PipelineWaitPolicy, RenderLoweringPolicy};
use metal2vulkan::metal::{IndexType, PixelFormat, PrimitiveType, TextureUsage};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

/// The side of the target, and the color the draws draw: `(0.2, 0.4, 0.6, 1)`, whose
/// `RGBA8Unorm` bytes are exact.
const SIDE: u64 = 16;
const COLOR: [f32; 4] = [0.2, 0.4, 0.6, 1.0];
const PACKED: u32 = u32::from_le_bytes([51, 102, 153, 255]);

/// Both policies, the default first.
const POLICIES: [RenderLoweringPolicy; 2] =
    [RenderLoweringPolicy::AtCall, RenderLoweringPolicy::Worker];

/// The notices a device announced.
type Heard = Arc<Mutex<Vec<DegradeNotice>>>;

/// A device of the kind the cfg selects that lowers render calls as `lowering` says, refusing a
/// call for a notice of a kind of `refuses` announced while it is lowered, its notices into
/// `heard`.
fn device(
    cache: &Path,
    lowering: RenderLoweringPolicy,
    refuses: &[Degradation],
    heard: &Heard,
) -> Device {
    device_with(cache, heard, |options| {
        options.lowering_refuses = refuses.iter().copied().collect();
        options.policy.render_lowering = lowering;
    })
}

/// A device of the kind the cfg selects that waits for a draw's pipeline still being made as
/// `waits` says, under `Bounded` [`CREATION_WAIT`] at a state's creation and [`COMMIT_WAIT`] from
/// a command buffer's commit, its notices into `heard`.
fn waiting_device(cache: &Path, waits: PipelineWaitPolicy, heard: &Heard) -> Device {
    device_with(cache, heard, |options| {
        options.policy.pipeline_waits = waits;
        options.policy.pipeline_creation_wait = CREATION_WAIT;
        options.policy.pipeline_commit_wait = COMMIT_WAIT;
    })
}

/// The bounds [`waiting_device`] sets for a state's creation and from a command buffer's commit,
/// apart from the defaults and far enough apart that a test tells which one a wait took: a
/// creation that waits its own bound returns before [`COMMIT_WAIT`], and a command buffer that
/// waits its own completes after it.
const CREATION_WAIT: Duration = Duration::from_millis(250);
const COMMIT_WAIT: Duration = Duration::from_secs(3);

/// A device of the kind the cfg selects, its shader cache in `cache` and its notices into
/// `heard`, of the options `adjust` sets.
fn device_with(cache: &Path, heard: &Heard, adjust: impl FnOnce(&mut DeviceOptions)) -> Device {
    let heard = Arc::clone(heard);
    let mut options = DeviceOptions {
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
        ..DeviceOptions::default()
    };
    adjust(&mut options);
    Device::system_default_with(&options).unwrap()
}

/// A `Private` `RGBA8Unorm` render target.
fn target(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// The state that draws a rectangle over the whole target, whose fragments take their color
/// from fragment buffer 0 (`tob_rect`, `tob_solid`).
fn rectangle_state(device: &Device) -> RenderPipelineState {
    render_state(device, "engine/tob_renders.ll", "tob_rect", "tob_solid")
}

/// Set `state` on `encoder` with the rectangle's corners.
fn rectangle(state: &RenderPipelineState, encoder: &mut RenderCommandEncoder) {
    encoder.set_render_pipeline_state(state).unwrap();
    let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
    encoder.set_vertex_bytes(&corners, 0).unwrap();
}

/// A `Shared` buffer `command_buffer` copies `target` into, encoded after its pass.
fn copy_back(device: &Device, command_buffer: &CommandBuffer, target: &Texture) -> Buffer {
    let back = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        target,
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
    back
}

/// Commit `command_buffer` and wait for it: the words of `back` then.
fn committed(command_buffer: &CommandBuffer, back: &Buffer) -> Vec<u32> {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    words_of(back)
}

/// What a check at the call works on: the device's rectangle state, a small buffer of the device
/// and one of another device.
struct Fixture {
    state: RenderPipelineState,
    buffer: Buffer,
    foreign: Buffer,
}

/// A call refused at its call, after the calls it needs: its result.
type Refused = fn(&Fixture, &mut RenderCommandEncoder) -> Result<(), Error>;

/// One call for each kind of check the encoder makes at the call, each refused.
const REFUSED: &[(&str, Refused)] = &[
    ("an index past the vertex table", |f, e| {
        e.set_vertex_buffer(Some(&f.buffer), 0, 31)
    }),
    ("a buffer of another device", |f, e| {
        e.set_vertex_buffer(Some(&f.foreign), 0, 0)
    }),
    ("an offset past the buffer", |f, e| {
        e.set_fragment_buffer(Some(&f.buffer), f.buffer.length() + 4, 0)
    }),
    ("buffers and offsets of different lengths", |f, e| {
        e.set_vertex_buffers(&[Some(&f.buffer)], &[0, 0], 0)
    }),
    ("an offset past the buffer bound", |f, e| {
        e.set_fragment_buffer(Some(&f.buffer), 0, 2)?;
        e.set_fragment_buffer_offset(f.buffer.length() + 4, 2)
    }),
    ("more inline bytes than Metal takes", |_, e| {
        e.set_fragment_bytes(&[0; 4097], 0)
    }),
    ("a texture index past the table", |_, e| {
        e.set_vertex_texture(None, 128)
    }),
    ("samplers and clamps of different lengths", |_, e| {
        e.set_fragment_sampler_states(&[None], Some((&[0.0, 0.0], &[1.0])), 0)
    }),
    ("more viewports than Metal takes", |_, e| {
        let viewport = Viewport::new(0.0, 0.0, 1.0, 1.0, 0.0, 1.0);
        e.set_viewports(&[viewport; 17])
    }),
    ("fewer view mappings than the count", |_, e| {
        let mapping = VertexAmplificationViewMapping::new(0, 0);
        e.set_vertex_amplification_count(2, Some(&[mapping]))
    }),
    ("a factor buffer of another device", |f, e| {
        e.set_tessellation_factor_buffer(Some(&f.foreign), 0, 0)
    }),
    ("a draw without a pipeline state", |_, e| {
        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
    }),
    ("an indexed draw of another device's indices", |f, e| {
        rectangle(&f.state, e);
        e.draw_indexed_primitives(
            PrimitiveType::Triangle,
            3,
            IndexType::UInt16,
            &f.foreign,
            0,
            1,
            0,
            0,
        )
    }),
    ("indirect arguments past the buffer", |f, e| {
        rectangle(&f.state, e);
        e.draw_primitives_indirect(PrimitiveType::Triangle, &f.buffer, f.buffer.length() - 8)
    }),
    ("indexed indirect arguments past the buffer", |f, e| {
        rectangle(&f.state, e);
        let past = f.buffer.length() - 12;
        e.draw_indexed_primitives_indirect(
            PrimitiveType::Triangle,
            IndexType::UInt16,
            &f.buffer,
            0,
            &f.buffer,
            past,
        )
    }),
    ("an amplification above the pipeline state's", |f, e| {
        rectangle(&f.state, e);
        e.set_vertex_amplification_count(2, None)?;
        e.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
    }),
    ("a resource of another device", |f, e| {
        e.use_resource(
            crate::ResourceRef::Buffer(&f.foreign),
            metal2vulkan::metal::ResourceUsage::READ,
            metal2vulkan::metal::RenderStages::VERTEX,
        )
    }),
];

/// Each refusal the encoder checks at the call returns from the call under both policies, the
/// same error; the command buffer then runs, naming no refusal.
#[test]
fn each_refusal_checked_at_the_call_returns_from_it_under_both_policies() {
    let scratch = Scratch::new("deferred-checks");
    let heard = Heard::default();
    let other = device(&scratch.0, RenderLoweringPolicy::AtCall, &[], &heard);
    let foreign = other.new_buffer(256, ResourceOptions::shared()).unwrap();
    let mut answers: Vec<Vec<Result<(), Error>>> = Vec::new();
    for policy in POLICIES {
        let device = device(&scratch.0, policy, &[], &heard);
        let fixture = Fixture {
            state: rectangle_state(&device),
            buffer: device.new_buffer(64, ResourceOptions::shared()).unwrap(),
            foreign: foreign.clone(),
        };
        let queue = device.new_command_queue().unwrap();
        let target = target(&device);
        let mut each = Vec::new();
        for (what, refused) in REFUSED {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer
                .render_command_encoder(&pass(&target))
                .unwrap();
            let answer = refused(&fixture, &mut encoder);
            assert!(answer.is_err(), "{policy:?}, {what}: {answer:?}");
            // The checks at the call refused it: the encoder still logs its calls.
            let logs = format!("{encoder:?}").contains("lowered_at_call: false");
            assert_eq!(logs, policy == RenderLoweringPolicy::Worker, "{what}");
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(
                (command_buffer.status(), command_buffer.refusal()),
                (CommandBufferStatus::Completed, None),
                "{policy:?}, {what}"
            );
            each.push(answer);
        }
        answers.push(each);
    }
    for (((what, _), at_call), worker) in REFUSED.iter().zip(&answers[0]).zip(&answers[1]) {
        assert_eq!(at_call, worker, "{what}");
    }
}

/// An offset at an index with no buffer bound is taken at the call under both policies, with the
/// same answer, and the draw after it runs: plain Metal keeps no offset there, so refusing it
/// would refuse what Metal takes (`ArgumentTable::take_buffer_offset`, whose doc holds the
/// measurement, and `metal2vulkan-conformance`'s `probes/buffer_offset_at_an_unbound_index.m`).
/// The index past the table is refused in the same call, so the test reads a real check and not a
/// call that answers `Ok` to everything.
#[test]
fn an_offset_at_an_unbound_index_is_taken_at_the_call_under_both_policies() {
    let scratch = Scratch::new("deferred-unbound-offset");
    let mut outcomes = Vec::new();
    for policy in POLICIES {
        let device = device(&scratch.0, policy, &[], &Heard::default());
        let state = rectangle_state(&device);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let target = target(&device);
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        // Nothing is bound at vertex index 3, and nothing in this encoder ever binds it.
        let taken = encoder.set_vertex_buffer_offset(0, 3);
        let past_the_table = encoder.set_vertex_buffer_offset(0, 31).is_err();
        rectangle(&state, &mut encoder);
        let color = bytes_of(&COLOR, f32::to_le_bytes);
        encoder.set_fragment_bytes(&color, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        assert_eq!(encoder.end_encoding(), Ok(()), "{policy:?}");
        let back = copy_back(&device, &command_buffer, &target);
        let words = committed(&command_buffer, &back);
        outcomes.push((
            taken,
            past_the_table,
            command_buffer.status(),
            command_buffer.refusal(),
            words.iter().all(|&word| word == PACKED),
        ));
    }
    assert_eq!(
        outcomes[0],
        (Ok(()), true, CommandBufferStatus::Completed, None, true),
        "{:?}",
        POLICIES[0]
    );
    assert_eq!(outcomes[0], outcomes[1]);
}

/// A draw of points under a state whose vertex function writes no point size, of no point class:
/// its lowering announces `UnsizedPoints` (`ShapeInputs::settings`), at the draw, call 3.
fn unsized_points(state: &RenderPipelineState, encoder: &mut RenderCommandEncoder) {
    rectangle(state, encoder);
    let color = bytes_of(&COLOR, f32::to_le_bytes);
    encoder.set_fragment_bytes(&color, 0).unwrap();
    assert_eq!(
        encoder.draw_primitives(PrimitiveType::Point, 0, 4, 1, 0),
        Ok(())
    );
}

/// The draw of [`unsized_points`].
const POINTS: EncoderCall = EncoderCall {
    selector: "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
    index: 3,
};

/// A notice announced while a call is lowered names it under `Worker`, and none under `AtCall`,
/// where the call lowers itself; with no kind refused (the default) it reaches the sink and the
/// command buffer runs.
#[test]
fn a_notice_announced_as_a_call_is_lowered_names_it() {
    let scratch = Scratch::new("deferred-notice");
    for policy in POLICIES {
        let heard = Heard::default();
        let device = device(&scratch.0, policy, &[], &heard);
        let state = rectangle_state(&device);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let target = target(&device);
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        unsized_points(&state, &mut encoder);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let heard = heard.lock().unwrap();
        let notice = heard
            .iter()
            .find(|notice| notice.degradation == Degradation::UnsizedPoints)
            .unwrap_or_else(|| panic!("{policy:?}: no notice of unsized points"));
        let named = match policy {
            RenderLoweringPolicy::Worker => Some(POINTS),
            _ => None,
        };
        assert_eq!(notice.call, named, "{policy:?}");
    }
}

/// A kind the device refuses while lowering (`lowering_refuses`) refuses the call it is announced
/// at under `Worker`: the call returned, the notice reaches no sink, and the command buffer fails
/// naming it (`CallRefused`), the notice its error, the encoder and the one after it running
/// nothing. Under `AtCall` it is announced as any other kind.
#[test]
fn a_kind_refused_while_lowering_refuses_the_call() {
    let scratch = Scratch::new("deferred-refused");
    for policy in POLICIES {
        let heard = Heard::default();
        let device = device(&scratch.0, policy, &[Degradation::UnsizedPoints], &heard);
        let state = rectangle_state(&device);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let target = target(&device);
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        unsized_points(&state, &mut encoder);
        // A call after the refused one runs nothing, and returns.
        let color = bytes_of(&COLOR, f32::to_le_bytes);
        encoder.set_fragment_bytes(&color, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        let back = copy_back(&device, &command_buffer, &target);
        let words = committed(&command_buffer, &back);
        let heard = heard.lock().unwrap();
        let announced = heard
            .iter()
            .any(|notice| notice.degradation == Degradation::UnsizedPoints);
        match policy {
            RenderLoweringPolicy::Worker => {
                assert!(!announced, "a refused kind announced");
                assert_eq!(
                    command_buffer.error(),
                    Some(CommandBufferError::CallRefused(POINTS))
                );
                let Some(Error::Degraded(notice)) = command_buffer.refusal() else {
                    panic!("{:?}", command_buffer.refusal());
                };
                assert_eq!(notice.degradation, Degradation::UnsizedPoints);
                assert_eq!(notice.call, Some(POINTS));
                // Neither the pass nor the copy after it ran.
                assert!(words.iter().all(|&word| word == 0), "{:x?}", &words[..4]);
            }
            _ => {
                assert!(announced, "{policy:?}: not announced");
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                assert_eq!(command_buffer.refusal(), None);
                assert!(
                    words.iter().all(|&word| word == PACKED),
                    "{:x?}",
                    &words[..4]
                );
            }
        }
    }
}

/// The bytes a call was lent are copied at the call: a later write to them before the encoder
/// ends is not drawn.
#[test]
fn bytes_lent_to_a_call_are_copied_at_the_call() {
    let scratch = Scratch::new("deferred-bytes");
    for policy in POLICIES {
        let device = device(&scratch.0, policy, &[], &Heard::default());
        let state = rectangle_state(&device);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let target = target(&device);
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        rectangle(&state, &mut encoder);
        let mut color = bytes_of(&COLOR, f32::to_le_bytes);
        encoder.set_fragment_bytes(&color, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        color.fill(0);
        encoder.end_encoding().unwrap();
        let back = copy_back(&device, &command_buffer, &target);
        let words = committed(&command_buffer, &back);
        assert!(
            words.iter().all(|&word| word == PACKED),
            "{policy:?}: {:x?}",
            &words[..4]
        );
    }
}

/// An object a call names is held from the call: a no-copy buffer the program lets go of right
/// after binding it lives (its deallocator waits) until the command buffer completed.
#[test]
fn an_object_released_after_its_call_lives_until_its_command_buffer_completes() {
    let scratch = Scratch::new("deferred-holds");
    for policy in POLICIES {
        let host = Host::new();
        let device = device(&scratch.0, policy, &[], &Heard::default());
        let runs = Arc::new(AtomicUsize::new(0));
        let Some(no_copy) = over_host(&device, &host, &runs) else {
            return;
        };
        host.set(0, &bits(&COLOR));
        let state = rectangle_state(&device);
        let queue = device.new_command_queue().unwrap();
        let filled = device
            .new_buffer(FILLED, ResourceOptions::private())
            .unwrap();
        let before = busy(&queue, &filled);
        let command_buffer = queue.command_buffer().unwrap();
        let target = target(&device);
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        rectangle(&state, &mut encoder);
        encoder.set_fragment_buffer(Some(&no_copy), 0, 0).unwrap();
        drop(no_copy);
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        assert_eq!(
            runs.load(Ordering::SeqCst),
            0,
            "{policy:?}: let go of at the call"
        );
        encoder.end_encoding().unwrap();
        let back = copy_back(&device, &command_buffer, &target);
        command_buffer.commit().unwrap();
        if before.status() != CommandBufferStatus::Completed {
            assert_eq!(
                runs.load(Ordering::SeqCst),
                0,
                "{policy:?}: let go of before completion"
            );
        }
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let words = words_of(&back);
        assert!(
            words.iter().all(|&word| word == PACKED),
            "{policy:?}: {:x?}",
            &words[..4]
        );
        until(
            "the deallocator runs once the command buffer completed",
            || runs.load(Ordering::SeqCst) == 1,
        );
    }
}

/// A buffer detached after a call that names it refuses nothing of the call: the encoder ends
/// without a refusal, and a rebind before the command buffer's first submission and the command
/// buffer meet the encoded call under `Worker` as under `AtCall`, where it was lowered before the
/// detach.
#[test]
fn a_buffer_detached_after_its_call_refuses_nothing_of_it() {
    let scratch = Scratch::new("deferred-detach");
    let mut outcomes = Vec::new();
    for policy in POLICIES {
        let (host, again) = (Host::new(), Host::new());
        let device = device(&scratch.0, policy, &[], &Heard::default());
        let runs = Arc::new(AtomicUsize::new(0));
        let Some(no_copy) = over_host(&device, &host, &runs) else {
            return;
        };
        host.fill(0);
        again.set(0, &bits(&COLOR));
        let state = rectangle_state(&device);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let target = target(&device);
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        rectangle(&state, &mut encoder);
        encoder.set_fragment_buffer(Some(&no_copy), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        assert_eq!(no_copy.detach_backing(), Ok(true));
        assert_eq!(encoder.end_encoding(), Ok(()), "{policy:?}");
        let back = copy_back(&device, &command_buffer, &target);
        // SAFETY: the pages outlive the buffer (dropped after it), which alone uses them.
        let rebound = unsafe { no_copy.rebind_bytes_no_copy(again.memory(), None, None) };
        let words = committed(&command_buffer, &back);
        let drawn = words.iter().all(|&word| word == PACKED);
        outcomes.push((
            rebound,
            command_buffer.status(),
            command_buffer.error(),
            drawn,
        ));
    }
    assert_eq!(outcomes[0], outcomes[1]);
}

/// A call made while a buffer of the device is detached is lowered at the call: a draw naming the
/// detached buffer fails its command buffer as one at its call does (`DetachedBacking`).
#[test]
fn a_call_made_while_a_buffer_is_detached_is_lowered_at_the_call() {
    let scratch = Scratch::new("deferred-detached");
    let mut outcomes = Vec::new();
    for policy in POLICIES {
        let host = Host::new();
        let device = device(&scratch.0, policy, &[], &Heard::default());
        let runs = Arc::new(AtomicUsize::new(0));
        let Some(no_copy) = over_host(&device, &host, &runs) else {
            return;
        };
        let state = rectangle_state(&device);
        assert_eq!(no_copy.detach_backing(), Ok(true));
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let target = target(&device);
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        rectangle(&state, &mut encoder);
        encoder.set_fragment_buffer(Some(&no_copy), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        outcomes.push((command_buffer.status(), command_buffer.error()));
    }
    assert_eq!(outcomes[0], outcomes[1]);
    assert_eq!(
        outcomes[0],
        (
            CommandBufferStatus::Error,
            Some(CommandBufferError::DetachedBacking)
        )
    );
}

/// A program that outruns the device's lowering thread waits for it once its encoder has handed
/// the thread `HANDED_BATCHES` batches it has not taken, and goes on as the thread takes them: with
/// the thread held, the program's calls stop at the cap, its log grows no more, and once the hold
/// goes every call is lowered once, in order (the last draw's color left): by the thread, and what
/// it had not taken as the encoder ended on the program's thread.
///
/// The thread takes its first turn late, as on a loaded host: the program reaches the cap before
/// the thread took any batch, and goes on by one batch once it takes one. So the calls made are
/// read only once the thread lowers a batch, which the hold keeps it at, with the cap handed: the
/// program then waits until the hold goes. Read at the cap alone, they may be read before the
/// thread's first turn, and the program's batch after it.
#[test]
fn a_program_that_outruns_the_lowering_thread_waits_for_it() {
    let scratch = Scratch::new("deferred-outrun");
    let device = device(
        &scratch.0,
        RenderLoweringPolicy::Worker,
        &[],
        &Heard::default(),
    );
    // Two calls a draw: past the batches the cap allows, the thread's own and the one filled.
    let draws = (HANDED_BATCHES + 4) * BATCH_CALLS / 2;
    let made = Arc::new(AtomicUsize::new(0));
    let (opened, open) = mpsc::channel();
    let (go, going) = mpsc::channel::<()>();
    let encoding = std::thread::spawn({
        let device = device.clone();
        let made = Arc::clone(&made);
        move || {
            let state = rectangle_state(&device);
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let target = target(&device);
            let mut encoder = command_buffer
                .render_command_encoder(&pass(&target))
                .unwrap();
            rectangle(&state, &mut encoder);
            opened.send(()).unwrap();
            going.recv().unwrap();
            let other = bytes_of(&[1.0f32, 0.0, 0.0, 1.0], f32::to_le_bytes);
            let last = bytes_of(&COLOR, f32::to_le_bytes);
            for draw in 0..draws {
                let color = if draw + 1 == draws { &last } else { &other };
                encoder.set_fragment_bytes(color, 0).unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
                made.fetch_add(1, Ordering::SeqCst);
            }
            encoder.end_encoding().unwrap();
            let back = copy_back(&device, &command_buffer, &target);
            committed(&command_buffer, &back)
        }
    });
    open.recv().unwrap();
    let slots = device.shared().deferred.taken();
    let [slot] = &slots[..] else {
        panic!("{} encoders' slots taken, not one", slots.len());
    };
    slot.turn_late(Duration::from_millis(100));
    let (stopped, after, handed) = slot.while_held(|| {
        go.send(()).unwrap();
        until("the program's calls stop at the cap", || {
            slot.handed() == (HANDED_BATCHES, true)
        });
        let stopped = made.load(Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(200));
        (stopped, made.load(Ordering::SeqCst), slot.handed())
    });
    assert_eq!(stopped, after, "draws made with the thread held at the cap");
    assert!(
        stopped < draws,
        "{stopped} of {draws} draws made at the cap"
    );
    assert_eq!(
        handed,
        (HANDED_BATCHES, true),
        "batches handed with the thread held"
    );
    let words = encoding.join().unwrap();
    assert!(
        words.iter().all(|&word| word == PACKED),
        "{:x?}",
        &words[..4]
    );
    // Every batch lowered once: the thread's, and those it had not taken as the encoder ended.
    let counts = device.work_counts();
    let calls = 2 + 2 * draws;
    let (by_thread, by_program) = (
        counts.render_log_batches_by_thread,
        counts.render_log_batches_by_program,
    );
    assert!(counts.render_log_waits >= 1, "{counts:?}");
    assert!(by_thread >= 1, "{counts:?}");
    assert_eq!(
        by_thread + by_program,
        calls.div_ceil(BATCH_CALLS) as u64,
        "batches lowered"
    );
}

/// An encoder's end leaves the lowering to the device's lowering thread and returns: with the
/// thread held, `end_encoding` returns; the command buffer's next encoder (the copy back) and its
/// commit lower what the thread has not, and the pass ran as drawn. The end was lowered once.
#[test]
fn an_end_returns_before_its_lowering() {
    let scratch = Scratch::new("deferred-end");
    let device = device(
        &scratch.0,
        RenderLoweringPolicy::Worker,
        &[],
        &Heard::default(),
    );
    let (opened, open) = mpsc::channel();
    let (go, going) = mpsc::channel::<()>();
    let (ended, end) = mpsc::channel();
    let encoding = std::thread::spawn({
        let device = device.clone();
        move || {
            let state = rectangle_state(&device);
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let target = target(&device);
            let mut encoder = command_buffer
                .render_command_encoder(&pass(&target))
                .unwrap();
            opened.send(()).unwrap();
            going.recv().unwrap();
            rectangle(&state, &mut encoder);
            encoder
                .set_fragment_bytes(&bytes_of(&COLOR, f32::to_le_bytes), 0)
                .unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
            ended.send(()).unwrap();
            let back = copy_back(&device, &command_buffer, &target);
            committed(&command_buffer, &back)
        }
    });
    open.recv().unwrap();
    let slots = device.shared().deferred.taken();
    let [slot] = &slots[..] else {
        panic!("{} encoders' slots taken, not one", slots.len());
    };
    let returned = slot.while_held(|| {
        go.send(()).unwrap();
        end.recv_timeout(WAIT).is_ok()
    });
    assert!(returned, "end_encoding waited for the held lowering");
    let words = encoding.join().unwrap();
    assert!(
        words.iter().all(|&word| word == PACKED),
        "{:x?}",
        &words[..4]
    );
    let counts = device.work_counts();
    let ends = counts.render_log_ends_by_thread + counts.render_log_ends_by_program;
    assert_eq!(ends, 1, "{counts:?}");
}

/// A kind the device refuses while lowering, met as the end is lowered after its call, refuses
/// the end: `CallRefused` names `endEncoding` and its index, the notice its error, and the pass
/// runs nothing. Where the device announces no such kind at the pass's end under `AtCall` (its
/// `DONT_CARE` keeps one of Metal's answers), there is nothing to refuse.
#[test]
fn a_kind_refused_as_the_end_is_lowered_refuses_the_end() {
    let scratch = Scratch::new("deferred-end-refused");
    let kind = Degradation::DontCareStore;
    for policy in POLICIES {
        let heard = Heard::default();
        let device = device(&scratch.0, policy, &[kind], &heard);
        let state = rectangle_state(&device);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let target = target(&device);
        let mut descriptor = pass(&target);
        descriptor.color_attachments[0].store_action = StoreAction::DontCare;
        let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
        rectangle(&state, &mut encoder);
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let announced = (heard.lock().unwrap().iter()).any(|notice| notice.degradation == kind);
        match policy {
            RenderLoweringPolicy::Worker => {
                let end = EncoderCall {
                    selector: "endEncoding",
                    index: 3,
                };
                assert!(!announced, "a refused kind announced");
                assert_eq!(
                    command_buffer.error(),
                    Some(CommandBufferError::CallRefused(end))
                );
                let Some(Error::Degraded(notice)) = command_buffer.refusal() else {
                    panic!("{:?}", command_buffer.refusal());
                };
                assert_eq!(notice.degradation, kind);
                assert_eq!(notice.call, Some(end));
            }
            _ => {
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                if !announced {
                    eprintln!("{policy:?}: {kind:?} not announced at the pass's end");
                    return;
                }
            }
        }
    }
}

/// What a command buffer's `commit` answers, and the calls made of it right after: its status,
/// a second commit, and a completed handler added.
type AfterCommit = (CommandBufferStatus, Result<(), Error>, Result<(), Error>);

/// A commit after an end the lowering thread has not lowered returns at once, under both
/// policies with the same answers: with the slot's lowering held (Worker), `commit` returns, the
/// status reads `Committed`, and a second commit, a completed handler and an event signal are
/// refused, as after Metal's commit. A command buffer committed after it on the same queue, the
/// copy back, waits behind it: it reads the pass's color, and the completed handlers run in
/// commit order. Under Worker both commits' own work ran once, behind the end.
///
/// The pass commits behind a command buffer of its queue that waits for an unsignalled event,
/// raised once the answers after the commit were read: Metal's commit returns before the command
/// buffer is scheduled, and its status may read `Scheduled` from then on (here, as soon as the
/// scheduler's thread submitted it), but one committed behind a command buffer of its queue an
/// event holds reads `Committed` until the event is raised (conformance case
/// `event-held-command-buffer`, measured on an M3 Max), so the status is read where Metal fixes
/// it under both policies.
#[test]
fn a_commit_after_an_end_in_flight_returns_and_keeps_its_order() {
    let scratch = Scratch::new("deferred-commit");
    for policy in POLICIES {
        let device = device(&scratch.0, policy, &[], &Heard::default());
        let (opened, open) = mpsc::channel();
        let (go, going) = mpsc::channel::<()>();
        let (returned, returns) = mpsc::channel::<AfterCommit>();
        let order = Arc::new(Mutex::new(Vec::new()));
        let encoding = std::thread::spawn({
            let device = device.clone();
            let order = Arc::clone(&order);
            move || {
                let state = rectangle_state(&device);
                let queue = device.new_command_queue().unwrap();
                let gate = device.new_shared_event().unwrap();
                let held = queue.command_buffer().unwrap();
                held.encode_wait_for_event(EventRef::from(&gate), 1)
                    .unwrap();
                let word = device.new_buffer(4, ResourceOptions::shared()).unwrap();
                let mut fill = held.blit_command_encoder().unwrap();
                fill.fill_buffer(&word, 0..4, 0).unwrap();
                fill.end_encoding().unwrap();
                held.commit().unwrap();
                let command_buffer = queue.command_buffer().unwrap();
                let target = target(&device);
                let event = device.new_event().unwrap();
                let mut encoder = command_buffer
                    .render_command_encoder(&pass(&target))
                    .unwrap();
                opened.send(()).unwrap();
                going.recv().unwrap();
                rectangle(&state, &mut encoder);
                encoder
                    .set_fragment_bytes(&bytes_of(&COLOR, f32::to_le_bytes), 0)
                    .unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
                encoder.end_encoding().unwrap();
                let ran = Arc::clone(&order);
                let handler = Box::new(move |_: &CommandBuffer| ran.lock().unwrap().push("pass"));
                command_buffer.add_completed_handler(handler).unwrap();
                command_buffer.commit().unwrap();
                let after = (
                    command_buffer.status(),
                    command_buffer.commit(),
                    command_buffer.add_completed_handler(Box::new(|_| ())),
                );
                gate.set_signaled_value(1);
                let copy = queue.command_buffer().unwrap();
                let back = copy_back(&device, &copy, &target);
                let ran = Arc::clone(&order);
                let handler = Box::new(move |_: &CommandBuffer| ran.lock().unwrap().push("copy"));
                copy.add_completed_handler(handler).unwrap();
                copy.commit().unwrap();
                returned.send(after).unwrap();
                let signal = command_buffer.encode_signal_event(EventRef::from(&event), 1);
                copy.wait_until_completed_for(WAIT).unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                (command_buffer.status(), signal, words_of(&back))
            }
        });
        open.recv().unwrap();
        let after = match policy {
            RenderLoweringPolicy::Worker => {
                let slots = device.shared().deferred.taken();
                let [slot] = &slots[..] else {
                    panic!("{} encoders' slots taken, not one", slots.len());
                };
                slot.while_held(|| {
                    go.send(()).unwrap();
                    returns.recv_timeout(WAIT).ok()
                })
            }
            _ => {
                go.send(()).unwrap();
                returns.recv_timeout(WAIT).ok()
            }
        };
        let Some((status, again, handler)) = after else {
            panic!("{policy:?}: commit waited for the held lowering");
        };
        assert_eq!(status, CommandBufferStatus::Committed, "{policy:?}");
        assert_eq!(again, Err(invalid("commit", Problem::AlreadyCommitted)));
        assert_eq!(
            handler,
            Err(invalid(
                "add_completed_handler",
                Problem::HandlerAfterCommit
            ))
        );
        let (status, signal, words) = encoding.join().unwrap();
        assert_eq!(
            signal,
            Err(invalid("encode_signal_event", Problem::AlreadyCommitted))
        );
        assert_eq!(status, CommandBufferStatus::Completed, "{policy:?}");
        assert!(
            words.iter().all(|&word| word == PACKED),
            "{policy:?}: {:x?}",
            &words[..4]
        );
        assert_eq!(*order.lock().unwrap(), ["pass", "copy"], "{policy:?}");
        let counts = device.work_counts();
        let commits = counts.render_log_commits_by_thread + counts.render_log_commits_by_program;
        let expected = match policy {
            RenderLoweringPolicy::Worker => 2,
            _ => 0,
        };
        assert_eq!(commits, expected, "{policy:?}: {counts:?}");
    }
}

/// The name of the thread that runs this.
fn thread_name() -> Option<String> {
    std::thread::current().name().map(str::to_owned)
}

/// A command buffer's submission check, submission handler and render encoder as a test of
/// [`a_commit_and_submit_after_an_end_in_flight_is_submitted_by_the_lowering_thread`] saw them:
/// which ran on which thread, and the command buffer the handler committed, with whether its
/// encoder lowered its calls at their calls.
#[derive(Default)]
struct RanOn {
    ran: Vec<(&'static str, Option<String>)>,
    inner: Option<(bool, CommandBuffer)>,
}

/// Under `RenderLoweringPolicy::Worker`, a `commit_and_submit` after an end in flight returns
/// before the end is lowered, as `commit` does, and the device's lowering thread, which then runs
/// the commit's own work, makes the call's pass too (`CommandBuffer::commit_queued`): one pass at
/// commit, which runs the command buffer's submission check and then its submission handler on
/// that thread and submits its batch (`WorkCounts::scheduler_passes_at_commit`,
/// `batches_submitted_at_commit`, `batches_checked_at_commit`), with no wake of the scheduler's
/// thread for it; the pass completes. A render encoder the handler opens there lowers its calls
/// at their calls (`render/deferral.rs`, "At the call"), and the command buffer the handler
/// commits completes.
#[test]
fn a_commit_and_submit_after_an_end_in_flight_is_submitted_by_the_lowering_thread() {
    let scratch = Scratch::new("deferred-commit-and-submit");
    let policy = RenderLoweringPolicy::Worker;
    let device = device(&scratch.0, policy, &[], &Heard::default());
    let seen = Arc::new(Mutex::new(RanOn::default()));
    let (opened, open) = mpsc::channel();
    let (go, going) = mpsc::channel::<()>();
    let (returned, returns) = mpsc::channel();
    let (completed, completes) = mpsc::channel();
    let encoding = std::thread::spawn({
        let device = device.clone();
        let seen = Arc::clone(&seen);
        move || {
            let state = rectangle_state(&device);
            let queue = device.new_command_queue().unwrap();
            let (target, inner_target) = (target(&device), target(&device));
            let command_buffer = queue.command_buffer().unwrap();
            let checked = Arc::clone(&seen);
            command_buffer
                .set_submission_check(Arc::new(move |_: &Submission<'_>| {
                    checked.lock().unwrap().ran.push(("check", thread_name()));
                    None
                }))
                .unwrap();
            let (handled, again) = (Arc::clone(&seen), queue.clone());
            command_buffer
                .add_submission_handler(Box::new(move |_: &CommandBuffer| {
                    handled.lock().unwrap().ran.push(("handler", thread_name()));
                    let command_buffer = again.command_buffer().unwrap();
                    let encoder = command_buffer
                        .render_command_encoder(&pass(&inner_target))
                        .unwrap();
                    let at_call = encoder.lowers_at_call();
                    encoder.end_encoding().unwrap();
                    command_buffer.commit().unwrap();
                    handled.lock().unwrap().inner = Some((at_call, command_buffer));
                }))
                .unwrap();
            command_buffer
                .add_completed_handler(Box::new(move |done: &CommandBuffer| {
                    let _ = completed.send((done.status(), done.error()));
                }))
                .unwrap();
            let mut encoder = command_buffer
                .render_command_encoder(&pass(&target))
                .unwrap();
            opened.send(()).unwrap();
            going.recv().unwrap();
            rectangle(&state, &mut encoder);
            encoder
                .set_fragment_bytes(&bytes_of(&COLOR, f32::to_le_bytes), 0)
                .unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit_and_submit().unwrap();
            returned.send(command_buffer.status()).unwrap();
        }
    });
    open.recv().unwrap();
    let slots = device.shared().deferred.taken();
    let [slot] = &slots[..] else {
        panic!("{} encoders' slots taken, not one", slots.len());
    };
    let status = slot.while_held(|| {
        go.send(()).unwrap();
        returns.recv_timeout(WAIT).ok()
    });
    assert_eq!(
        status,
        Some(CommandBufferStatus::Committed),
        "commit_and_submit waited for the held lowering, or its command buffer went on"
    );
    let done = completes.recv_timeout(WAIT);
    assert_eq!(
        done,
        Ok((CommandBufferStatus::Completed, None)),
        "the pass completes"
    );
    encoding.join().unwrap();
    let (ran, inner) = {
        let seen = seen.lock().unwrap();
        (seen.ran.clone(), seen.inner.clone())
    };
    let lowering = Some("metal2vulkan-lowering".to_owned());
    assert_eq!(
        ran,
        [("check", lowering.clone()), ("handler", lowering)],
        "the check and the handler ran on the lowering thread"
    );
    let Some((at_call, inner)) = inner else {
        panic!("the submission handler committed no command buffer");
    };
    assert!(
        at_call,
        "an encoder opened on the lowering thread logged its calls"
    );
    inner.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(inner.status(), CommandBufferStatus::Completed);
    let counts = device.work_counts();
    let passed = (
        counts.scheduler_passes_at_commit,
        counts.batches_submitted_at_commit,
        counts.batches_checked_at_commit,
    );
    assert_eq!(passed, (1, 1, 1), "{counts:?}");
    assert!(
        counts.render_log_commits_by_thread >= 1,
        "the lowering thread ran no commit: {counts:?}"
    );
}

/// A commit after an end in flight takes a command buffer one of whose units of command predication
/// ended with a live block open, under both policies alike: an If whose condition failed, left open
/// in a compute encoder before the pass, fails the command buffer at its completion with Metal's
/// GPU recovery's error (`PredicationOpenAtUnitEnd` of its record 0, code 1, Metal's description), and one whose
/// condition held completes it. Where the device takes no command predication there is nothing to
/// run.
#[test]
fn a_commit_after_an_end_in_flight_fails_a_skipping_predication_block_left_open() {
    let scratch = Scratch::new("deferred-commit-open-block");
    for policy in POLICIES {
        let device = device(&scratch.0, policy, &[], &Heard::default());
        let state = rectangle_state(&device);
        let queue = device.new_command_queue().unwrap();
        let word = device.new_buffer(4, ResourceOptions::shared()).unwrap();
        for (reference, error) in [
            (
                1,
                Some(CommandBufferError::PredicationOpenAtUnitEnd { record: 0 }),
            ),
            (0, None),
        ] {
            let command_buffer = queue.command_buffer().unwrap();
            let mut compute = command_buffer.compute_command_encoder().unwrap();
            if let Err(error) = compute.encode_start_if(Some(&word), 0, 0, reference) {
                assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
                eprintln!("{policy:?}: the device takes no command predication: {error}");
                return;
            }
            compute.end_encoding().unwrap();
            let target = target(&device);
            let mut encoder = command_buffer
                .render_command_encoder(&pass(&target))
                .unwrap();
            rectangle(&state, &mut encoder);
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed().unwrap();
            assert_eq!(command_buffer.error(), error, "{policy:?} {reference}");
            let description = error.map(|_| {
                "Discarded (victim of GPU error/recovery) \
                 (00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)"
                    .to_owned()
            });
            assert_eq!(
                command_buffer.error_description(),
                description,
                "{policy:?}"
            );
            if let Some(error) = error {
                assert_eq!(
                    (error.metal_code(), error.origin()),
                    (1, FailureOrigin::Metal)
                );
            }
        }
    }
}

/// A commit whose own work fails, after it returned, fails its command buffer as Metal's committed
/// command buffers fail: with an out of memory injected into that work twice, under Worker (the
/// slot's lowering held, so the commit's work waits for the end), `commit` returns; the command
/// buffer completes with `CallRefused` of `commit` and the error kept, its completed handler runs
/// and its wait returns; a command buffer committed behind it on the same queue, while the
/// lowering is still held, so with its own commit waiting behind the refused one, completes after
/// it, its fill landed, and both commits' own work ran once. Its second injected failure never
/// ran: the commit ran again making nothing that may fail. Under `AtCall` the commit returns the
/// error.
#[test]
fn a_commit_whose_own_work_fails_completes_with_the_refusal() {
    let scratch = Scratch::new("deferred-commit-fails");
    let error = vulkan(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
    for policy in POLICIES {
        let device = device(&scratch.0, policy, &[], &Heard::default());
        let state = rectangle_state(&device);
        let queue = device.new_command_queue().unwrap();
        let target = target(&device);
        let refused = queue.command_buffer().unwrap();
        let mut encoder = refused.render_command_encoder(&pass(&target)).unwrap();
        rectangle(&state, &mut encoder);
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        let order = Arc::new(Mutex::new(Vec::new()));
        let ran = Arc::clone(&order);
        let handler = Box::new(move |_: &CommandBuffer| ran.lock().unwrap().push("refused"));
        refused.add_completed_handler(handler).unwrap();
        if policy != RenderLoweringPolicy::Worker {
            encoder.end_encoding().unwrap();
            refused.fail_commit_work(1);
            assert_eq!(refused.commit(), Err(error.clone()));
            assert_eq!(refused.status(), CommandBufferStatus::NotEnqueued);
            continue;
        }
        let slots = device.shared().deferred.taken();
        let [slot] = &slots[..] else {
            panic!("{} encoders' slots taken, not one", slots.len());
        };
        refused.fail_commit_work(2);
        let filled = device.new_buffer(4, ResourceOptions::shared()).unwrap();
        let later = queue.command_buffer().unwrap();
        let ran = Arc::clone(&order);
        let handler = Box::new(move |_: &CommandBuffer| ran.lock().unwrap().push("later"));
        // Both commits are made while the lowering is held, so the later one waits behind the
        // refused one, whose own work waits for the end, as every commit made while one waits
        // does (`render/queued_commits.rs`). Made after the hold it is a race with the device's
        // lowering thread: where that thread ran the refused commit and left the queue idle
        // first, the later commit runs inside its own call, which is as right and counts one
        // commit run, not two.
        let (committed, after) = slot.while_held(|| {
            encoder.end_encoding().unwrap();
            let committed = refused.commit();
            let mut blit = later.blit_command_encoder().unwrap();
            blit.fill_buffer(&filled, 0..4, 0x5a).unwrap();
            blit.end_encoding().unwrap();
            later.add_completed_handler(handler).unwrap();
            (committed, later.commit())
        });
        assert_eq!(committed, Ok(()));
        assert_eq!(after, Ok(()));
        refused.wait_until_completed_for(WAIT).unwrap();
        later.wait_until_completed_for(WAIT).unwrap();
        let commit = EncoderCall {
            selector: "commit",
            index: 0,
        };
        assert_eq!(refused.status(), CommandBufferStatus::Error);
        assert_eq!(
            refused.error(),
            Some(CommandBufferError::CallRefused(commit))
        );
        assert_eq!(refused.refusal(), Some(error.clone()));
        assert_eq!(later.status(), CommandBufferStatus::Completed);
        assert_eq!(words_of(&filled), vec![u32::from_le_bytes([0x5a; 4])]);
        assert_eq!(*order.lock().unwrap(), ["refused", "later"]);
        let counts = device.work_counts();
        let commits = counts.render_log_commits_by_thread + counts.render_log_commits_by_program;
        assert_eq!(commits, 2, "{counts:?}");
    }
}
/// A draw whose pipeline the driver did not make after the draw returned (a line over the
/// rectangle state, whose making waits in the held pool, its failure injected as the driver's
/// out of device memory) runs nothing, under both policies: its command buffer completes without
/// error, the pass clears its target and the blit after it copies it, `dropped_draw` names the
/// draw (encoder 0, draw 0, the driver's error) and `failed_pipeline` nothing. Counted:
/// `WorkCounts::draws_dropped_unmade`, and announced `draw-dropped-unmade`, naming the driver's
/// error.
#[test]
fn a_draw_whose_pipeline_was_not_made_runs_nothing_and_its_command_buffer_runs_on() {
    let scratch = Scratch::new("deferred-dropped-draw");
    let failure = vulkan(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
    for policy in POLICIES {
        let heard = Heard::default();
        let device = device(&scratch.0, policy, &[], &heard);
        let state = rectangle_state(&device);
        let queue = device.new_command_queue().unwrap();
        let target = target(&device);
        let counts = device.work_counts();
        let hold = crate::pipeline::pool::held::hold();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer.fail_draw_making(failure.clone());
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        rectangle(&state, &mut encoder);
        let red = bytes_of(&[1.0f32, 0.0, 0.0, 1.0], f32::to_le_bytes);
        encoder.set_fragment_bytes(&red, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Line, 0, 6, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        let back = copy_back(&device, &command_buffer, &target);
        command_buffer.commit().unwrap();
        drop(hold);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None, "{policy:?}");
        assert!(command_buffer.failed_pipeline().is_none(), "{policy:?}");
        let dropped = command_buffer
            .dropped_draw()
            .expect("the draw dropped is named");
        assert_eq!((dropped.encoder, dropped.draw), (0, 0), "{policy:?}");
        assert_eq!(dropped.error, failure, "{policy:?}");
        assert!(words_of(&back).iter().all(|&w| w == 0), "{policy:?}");
        let after = device.work_counts();
        assert!(
            after.pipelines_pooled > counts.pipelines_pooled,
            "the making waited"
        );
        assert_eq!(after.draws_dropped_unmade - counts.draws_dropped_unmade, 1);
        let unmade: Vec<_> = (heard.lock().unwrap().iter())
            .filter(|notice| notice.degradation == Degradation::DrawDroppedUnmade)
            .cloned()
            .collect();
        assert_eq!(unmade.len(), 1, "{policy:?}: {unmade:?}");
        assert!(
            unmade[0].reason.contains(&failure.to_string()),
            "{policy:?}: the notice names the driver's error: {unmade:?}"
        );
    }
}

/// The 10 s a guest's FIFO check gives one of its command buffers' executions, past which it ends
/// the guest's task: no wait for a pipeline's making may hold a command buffer that long.
const GUEST_WATCHDOG: Duration = Duration::from_secs(10);

/// A render pass of `state`'s rectangle in [`COLOR`] into `target`, and a copy of `target` after
/// it, encoded into `command_buffer`, drawn as `primitive`: the buffer the copy fills.
fn rectangle_pass(
    device: &Device,
    command_buffer: &CommandBuffer,
    (state, target): (&RenderPipelineState, &Texture),
    primitive: PrimitiveType,
) -> Buffer {
    let mut encoder = command_buffer
        .render_command_encoder(&pass(target))
        .unwrap();
    rectangle(state, &mut encoder);
    let color = bytes_of(&COLOR, f32::to_le_bytes);
    encoder.set_fragment_bytes(&color, 0).unwrap();
    encoder.draw_primitives(primitive, 0, 6, 1, 0).unwrap();
    encoder.end_encoding().unwrap();
    copy_back(device, command_buffer, target)
}

/// A draw whose pipeline is still being made at its command buffer's bound runs nothing, under
/// `PipelineWaitPolicy::Bounded` (the default), the slow compile the pool's test hold
/// (`pool::held`), which keeps every making waiting, no sleep: the state's creation hands its
/// predicted key's making to the held pool and returns at its own bound, the policy's creation
/// wait as set ([`CREATION_WAIT`], before the commit wait), the making still waiting
/// (`WorkCounts::creations_left_making`), never past the 10 s a guest's FIFO check allows. The
/// rectangle's draw takes that making, and its command buffer, committed with the pool still held,
/// completes without error once its own bound passed, the policy's commit wait as set
/// ([`COMMIT_WAIT`]), within those 10 s: the pass clears its
/// target and the blit after it copies it, `dropped_draw` names the draw (encoder 0, draw 0,
/// `VK_TIMEOUT`), `failed_pipeline` nothing, counted `draws_dropped_still_making` (not
/// `draws_dropped_unmade`) and announced `draw-dropped-still-making`. Once the hold goes the making
/// lands, and a later command buffer draws the rectangle.
#[test]
fn a_draw_whose_pipeline_is_still_being_made_at_its_bound_runs_nothing() {
    let scratch = Scratch::new("deferred-making-bound");
    let heard = Heard::default();
    let device = waiting_device(&scratch.0, PipelineWaitPolicy::Bounded, &heard);
    let queue = device.new_command_queue().unwrap();
    let target = target(&device);
    let counts = device.work_counts();
    let hold = held::hold();
    let creating = Instant::now();
    let state = rectangle_state(&device);
    let created = creating.elapsed();
    let made = device.work_counts();
    assert_eq!(
        made.creations_left_making - counts.creations_left_making,
        1,
        "the creation returned with its predicted making held"
    );
    assert_eq!(made.creation_pipelines_made, counts.creation_pipelines_made);
    assert!(
        (CREATION_WAIT..COMMIT_WAIT).contains(&created),
        "the creation took {created:?}, not its own bound"
    );
    let command_buffer = queue.command_buffer().unwrap();
    let back = rectangle_pass(
        &device,
        &command_buffer,
        (&state, &target),
        PrimitiveType::Triangle,
    );
    let committing = Instant::now();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(GUEST_WATCHDOG)
        .expect("the command buffer completes at its bound with its draw's making held");
    let waited = committing.elapsed();
    assert!(
        (COMMIT_WAIT..GUEST_WATCHDOG).contains(&waited),
        "completed after {waited:?}, not its own bound"
    );
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    assert!(command_buffer.failed_pipeline().is_none());
    let dropped = command_buffer
        .dropped_draw()
        .expect("the draw dropped is named");
    assert_eq!((dropped.encoder, dropped.draw), (0, 0));
    assert_eq!(dropped.error, vulkan(vk::Result::TIMEOUT));
    assert!(words_of(&back).iter().all(|&w| w == 0), "the draw drew");
    let after = device.work_counts();
    assert_eq!(
        after.draws_dropped_still_making - counts.draws_dropped_still_making,
        1
    );
    assert_eq!(after.draws_dropped_unmade, counts.draws_dropped_unmade);
    assert!(
        (heard.lock().unwrap().iter())
            .any(|notice| notice.degradation == Degradation::DrawDroppedStillMaking),
        "no notice of the dropped draw"
    );
    drop(hold);
    // The making lands for the command buffers after it.
    let deadline = Instant::now() + WAIT;
    loop {
        let command_buffer = queue.command_buffer().unwrap();
        let back = rectangle_pass(
            &device,
            &command_buffer,
            (&state, &target),
            PrimitiveType::Triangle,
        );
        let words = committed(&command_buffer, &back);
        if command_buffer.dropped_draw().is_none() {
            assert!(words.iter().all(|&w| w == PACKED), "the rectangle, drawn");
            break;
        }
        assert!(Instant::now() < deadline, "the making never landed");
    }
}

/// A draw of an execution's replay whose pipeline is still being made at its command buffer's
/// bound runs nothing, under `PipelineWaitPolicy::Bounded` (the default), the slow compile the
/// pool's test hold (`pool::held`), no sleep. The commit replays the records the CPU wrote as it
/// reads them, so the draw takes no pending recording: it waits for its making, held in the pool,
/// until the bound, then records nothing. The commit returns once the bound passed, within the
/// 10 s a guest's FIFO check allows, and the command buffer completes without error, its pass
/// clearing its target and the blit after it copying it: `dropped_draw` names the draw (its
/// execution's encoder 0, draw 0, `VK_TIMEOUT`), `failed_pipeline` nothing, counted
/// `draws_dropped_still_making` and announced `draw-dropped-still-making`. Once the hold goes the
/// making lands, and a later command buffer draws the rectangle.
#[test]
fn a_replayed_draw_whose_pipeline_is_still_being_made_at_its_bound_runs_nothing() {
    use super::encoder_holds::library;
    use crate::{IndirectCommandBufferDescriptor, PipelineOptions};
    use metal2vulkan::metal::{IndirectCommandType, RenderStages, ResourceUsage};
    use metal2vulkan::pipeline::{
        ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource,
    };
    let scratch = Scratch::new("deferred-replay-bound");
    let heard = Heard::default();
    let device = waiting_device(&scratch.0, PipelineWaitPolicy::Bounded, &heard);
    let queue = device.new_command_queue().unwrap();
    let target = target(&device);
    let library = library(&device, "engine/tob_renders.ll");
    let vertex = library.new_function_with_name("tob_rect").unwrap();
    let fragment = library.new_function_with_name("tob_solid").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.support_indirect_command_buffers = true;
    let shared = ResourceOptions::shared();
    let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
    let corners = device.new_buffer_with_bytes(&corners, shared).unwrap();
    let color = bytes_of(&COLOR, f32::to_le_bytes);
    let color = device.new_buffer_with_bytes(&color, shared).unwrap();
    let counts = device.work_counts();
    let hold = held::hold();
    let state = device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let mut commands = IndirectCommandBufferDescriptor::new();
    commands.command_types = IndirectCommandType::DRAW;
    commands.max_vertex_buffer_bind_count = 1;
    commands.max_fragment_buffer_bind_count = 1;
    let icb = device
        .new_indirect_command_buffer(&commands, 1, shared)
        .unwrap();
    let command = icb.indirect_render_command_at(0).unwrap();
    command.set_render_pipeline_state(&state).unwrap();
    command.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
    command.set_fragment_buffer(Some(&color), 0, 0).unwrap();
    command
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    let replayed = |command_buffer: &CommandBuffer| {
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        let (read, vertex, fragment) = (
            ResourceUsage::READ,
            RenderStages::VERTEX,
            RenderStages::FRAGMENT,
        );
        encoder
            .use_resource(corners.as_resource(), read, vertex)
            .unwrap();
        encoder
            .use_resource(color.as_resource(), read, fragment)
            .unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        copy_back(&device, command_buffer, &target)
    };
    let command_buffer = queue.command_buffer().unwrap();
    let back = replayed(&command_buffer);
    let committing = Instant::now();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(GUEST_WATCHDOG)
        .expect("the command buffer completes at its bound with its draw's making held");
    let waited = committing.elapsed();
    assert!(
        (COMMIT_WAIT..GUEST_WATCHDOG).contains(&waited),
        "completed after {waited:?}, not its own bound"
    );
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    assert!(command_buffer.failed_pipeline().is_none());
    let dropped = command_buffer
        .dropped_draw()
        .expect("the replayed draw dropped is named");
    assert_eq!((dropped.encoder, dropped.draw), (0, 0));
    assert_eq!(dropped.error, vulkan(vk::Result::TIMEOUT));
    assert!(words_of(&back).iter().all(|&w| w == 0), "the draw drew");
    let after = device.work_counts();
    assert_eq!(
        after.draws_dropped_still_making - counts.draws_dropped_still_making,
        1
    );
    assert!(
        (heard.lock().unwrap().iter())
            .any(|notice| notice.degradation == Degradation::DrawDroppedStillMaking),
        "no notice of the dropped draw"
    );
    drop(hold);
    // The making lands for the command buffers after it.
    let deadline = Instant::now() + WAIT;
    loop {
        let command_buffer = queue.command_buffer().unwrap();
        let back = replayed(&command_buffer);
        let words = committed(&command_buffer, &back);
        if command_buffer.dropped_draw().is_none() {
            assert!(words.iter().all(|&w| w == PACKED), "the rectangle, drawn");
            break;
        }
        assert!(Instant::now() < deadline, "the making never landed");
    }
}

/// A compute dispatch whose pipeline is still being made at its command buffer's bound runs
/// nothing, under `PipelineWaitPolicy::Bounded` (the default), the slow compile the pool's test
/// hold (`pool::held`), no sleep. The state requires no threadgroup size, so its creation predicts
/// nothing; the dispatch hands its region's making to the held pool, and its command buffer,
/// committed with the pool still held, completes without error once its bound passed, within the
/// 10 s a guest's FIFO check allows, its output unwritten: `dropped_draw` names the dispatch
/// (encoder 0, dispatch 0, `VK_TIMEOUT`), `failed_pipeline` nothing, counted
/// `dispatches_dropped_still_making` and announced `dispatch-dropped-still-making`. Once the hold
/// goes the making lands, and a later command buffer writes what Metal's writes.
#[test]
fn a_dispatch_whose_pipeline_is_still_being_made_at_its_bound_runs_nothing() {
    use super::dispatch_pending::{dispatch, state, Buffers, Grid};
    let scratch = Scratch::new("deferred-dispatch-bound");
    let heard = Heard::default();
    let device = waiting_device(&scratch.0, PipelineWaitPolicy::Bounded, &heard);
    let queue = device.new_command_queue().unwrap();
    let state = state(&device);
    let buffers = Buffers::new(&device);
    let one = || Grid::One(Size::new(64, 1, 1));
    let counts = device.work_counts();
    let hold = held::hold();
    let command_buffer = dispatch(&queue, &state, &buffers, one());
    let committing = Instant::now();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(GUEST_WATCHDOG)
        .expect("the command buffer completes at its bound with its dispatch's making held");
    let waited = committing.elapsed();
    assert!(
        (COMMIT_WAIT..GUEST_WATCHDOG).contains(&waited),
        "completed after {waited:?}, not its own bound"
    );
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    assert!(command_buffer.failed_pipeline().is_none());
    let dropped = command_buffer
        .dropped_draw()
        .expect("the dispatch dropped is named");
    assert_eq!((dropped.encoder, dropped.draw), (0, 0));
    assert_eq!(dropped.error, vulkan(vk::Result::TIMEOUT));
    assert!(!buffers.written(2), "the dispatch wrote");
    let after = device.work_counts();
    assert_eq!(
        after.dispatches_dropped_still_making - counts.dispatches_dropped_still_making,
        1
    );
    assert_eq!(
        after.draws_dropped_still_making,
        counts.draws_dropped_still_making
    );
    assert!(
        (heard.lock().unwrap().iter())
            .any(|notice| notice.degradation == Degradation::DispatchDroppedStillMaking),
        "no notice of the dropped dispatch"
    );
    drop(hold);
    // The making lands for the command buffers after it.
    let deadline = Instant::now() + WAIT;
    loop {
        let command_buffer = dispatch(&queue, &state, &buffers, one());
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        if command_buffer.dropped_draw().is_none() {
            assert!(
                buffers.written(64),
                "the dispatch wrote what Metal's writes"
            );
            break;
        }
        assert!(Instant::now() < deadline, "the making never landed");
    }
}

/// A compute state whose descriptor requires a threadgroup size predicts its first dispatch's
/// region at its creation (`pipeline/dispatch_making.rs`). Under `PipelineWaitPolicy::Bounded`
/// (the default) the creation hands that making to the device's pool and, the pool held
/// (`pool::held`), returns at its bound with the making still waiting
/// (`WorkCounts::creations_left_making`), never past the 10 s a guest's FIFO check allows; under
/// `Exact` it makes the region on its own thread before it returns, the pool held or not.
#[test]
fn a_compute_state_s_creation_waits_its_bound_for_its_predicted_making() {
    use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
    for waits in [PipelineWaitPolicy::Bounded, PipelineWaitPolicy::Exact] {
        let scratch = Scratch::new("deferred-compute-creation-bound");
        let heard = Heard::default();
        let device = waiting_device(&scratch.0, waits, &heard);
        let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/fixtures/public/examples/runtime_examples.ll");
        let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
        let function = device
            .new_library_with_compile_library(
                metal2vulkan::library::Library::from_air(&bitcode).unwrap(),
            )
            .new_function_with_name("scale_and_offset")
            .unwrap();
        let mut descriptor =
            ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        descriptor.required_threads_per_threadgroup = Some([32, 1, 1]);
        let counts = device.work_counts();
        let hold = held::hold();
        let creating = Instant::now();
        let _state = device
            .new_compute_pipeline_state(&descriptor, &crate::PipelineOptions::default())
            .unwrap();
        let created = creating.elapsed();
        let after = device.work_counts();
        drop(hold);
        let left = after.creations_left_making - counts.creations_left_making;
        let made = after.creation_dispatch_pipelines_made - counts.creation_dispatch_pipelines_made;
        match waits {
            PipelineWaitPolicy::Bounded => {
                assert_eq!((left, made), (1, 0), "the creation's making, held");
                assert!(
                    (CREATION_WAIT..COMMIT_WAIT).contains(&created),
                    "the creation took {created:?}, not its own bound"
                );
            }
            _ => assert_eq!((left, made), (0, 1), "the creation's making, made"),
        }
    }
}

/// Under `PipelineWaitPolicy::Exact` a held making is waited for until it is made: the state's
/// creation makes its predicted key before it returns, the pool held or not (none left making); a
/// line over it, whose key only its draw makes, waits in the held pool, and its command buffer,
/// committed, is not complete a second past the bounded form's commit wait; once the hold goes, the making lands and the
/// command buffer completes with the line drawn, no draw dropped.
#[test]
fn under_the_exact_wait_a_held_making_holds_its_command_buffer_until_it_lands() {
    let scratch = Scratch::new("deferred-making-exact");
    let heard = Heard::default();
    let device = waiting_device(&scratch.0, PipelineWaitPolicy::Exact, &heard);
    let queue = device.new_command_queue().unwrap();
    let target = target(&device);
    let counts = device.work_counts();
    let hold = held::hold();
    let state = rectangle_state(&device);
    let made = device.work_counts();
    assert_eq!(made.creations_left_making, counts.creations_left_making);
    let command_buffer = queue.command_buffer().unwrap();
    let back = rectangle_pass(
        &device,
        &command_buffer,
        (&state, &target),
        PrimitiveType::Line,
    );
    assert!(
        device.work_counts().pipelines_pooled > made.pipelines_pooled,
        "the line's making waits in the pool"
    );
    command_buffer.commit().unwrap();
    assert!(
        command_buffer
            .wait_until_completed_for(COMMIT_WAIT + Duration::from_secs(1))
            .is_err(),
        "completed while its draw's making was held"
    );
    drop(hold);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    assert!(command_buffer.dropped_draw().is_none());
    assert!(words_of(&back).contains(&PACKED), "the line, drawn");
    let after = device.work_counts();
    assert_eq!(
        after.draws_dropped_still_making,
        counts.draws_dropped_still_making
    );
    assert!(!(heard.lock().unwrap().iter())
        .any(|notice| notice.degradation == Degradation::DrawDroppedStillMaking));
}
