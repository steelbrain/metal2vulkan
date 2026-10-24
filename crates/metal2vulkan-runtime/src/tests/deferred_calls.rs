//! A render encoder's calls lowered after they return (`RenderLoweringPolicy::Worker`,
//! `render/deferral.rs`) against inside each call (`AtCall`): each refusal the encoder checks at
//! the call (`render/call_checks.rs`) returns from the call under both policies, the same error,
//! and the command buffer then runs; a refusal only the lowering finds fails the command buffer,
//! naming the call (`CommandBufferError::CallRefused`), and the command buffer keeps the call's
//! error; a notice announced while a call is lowered names it, and a kind the device refuses while
//! lowering (`DeviceOptions::lowering_refuses`) refuses it; the bytes a call was lent are copied at
//! the call, and the objects it names are held from the call until its command buffer completed;
//! a buffer detached after the call refuses nothing of it; a program that outruns the device's
//! lowering thread waits for it at the cap of its encoder's batches, and goes on; an end and a
//! commit return before the lowering thread lowered the end, the commit's own work waiting for it,
//! and the commits after it waiting behind it, and one whose own work fails completing with the
//! refusal.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{busy, over_host, until, Host, Scratch, FILLED, WAIT};
use super::encoder_holds::{bits, bytes_of, pass, render_state, words_of};
use crate::error::{invalid, vulkan};
use crate::render::{BATCH_CALLS, HANDED_BATCHES};
use crate::StoreAction;
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, Degradation,
    DegradeNotice, Device, DeviceKind, DeviceOptions, EncoderCall, Error, EventRef, FailureOrigin,
    MessageSink, Origin, Problem, RenderCommandEncoder, RenderPipelineState, ResourceOptions,
    ShaderCacheOptions, Size, Texture, TextureDescriptor, VertexAmplificationViewMapping, Viewport,
};
use ash::vk;
use metal2vulkan::device::RenderLoweringPolicy;
use metal2vulkan::metal::{IndexType, PixelFormat, PrimitiveType, TextureUsage};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

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
        lowering_refuses: refuses.iter().copied().collect(),
        ..DeviceOptions::default()
    };
    options.policy.render_lowering = lowering;
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
    ("an offset where no buffer is bound", |_, e| {
        e.set_vertex_buffer_offset(0, 3)
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
    let (stopped, after, handed) = slot.while_held(|| {
        go.send(()).unwrap();
        until("the program's calls stop at the cap", || {
            slot.handed() == HANDED_BATCHES
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
        handed, HANDED_BATCHES,
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
