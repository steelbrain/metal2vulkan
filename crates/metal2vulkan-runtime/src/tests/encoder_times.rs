//! The encoder times (`METAL2VULKAN_ENCODER_TIMES`, a default-off diagnostic, `encoder_times.rs`),
//! switched on in the process for one device as a test switches the queue times on.
//!
//! - **On**: one command buffer of three compute encoders of `scale_and_offset`, a short one, a
//!   long one and a short one, each of a state of its own label, which record on in one
//!   `VkCommandBuffer` (`encoder_fold.rs`, counted) once the same shape ran with the times off,
//!   its pipelines made: its line names the three encoders in order, each under its state's label
//!   hash, the long one of more GPU time than either short one, and the summary the device's
//!   teardown prints ranks the long one's row first of the three.
//! - **Draws again**: a render encoder of [`DRAWS`] draws of one rectangle, warmed, the draws after
//!   the first drawn again as before (`render/draw_again.rs`, `WorkCounts::draws_as_before`) or,
//!   other bytes bound before each, rebound (`render/draw_again/rebound.rs`,
//!   `WorkCounts::draws_rebound`): its command buffer's line names its every draw, the draws
//!   again among them.
//! - **Off**: a command buffer of a compute encoder, a render pass of one draw recorded inline and
//!   a blit encoder, its pipelines made by one run before, makes no timestamp query pool and
//!   begins as many `VkCommandBuffer`s as the same one did before; switched on, the same one
//!   makes the pools and begins one more: the render pass's own, so the commands before the pass
//!   go after the encoder's start (`EncoderCore::hold_pass`).
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects.

use super::blit_holds::{options, Scratch, WAIT};
use super::draw_again::{adding_state, rectangle, run as run_pass, set_rectangle, warm, DRAWS};
use super::encoder_holds::{bytes_of, library, pass, render_state};
use super::plan_room::device_with_room;
use crate::encoder_times::ID_NAME;
use crate::encoder_trail::label_hash;
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, ComputePipelineState, Device, MessageSink,
    PipelineOptions, RenderCommandEncoder, ResourceOptions, Size, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The floats the long encoder's dispatches scale, and its dispatches; a CPU device runs fewer.
const LONG_FLOATS: u64 = 1 << 24;
const LONG_DISPATCHES: usize = match cfg!(m2v_cpu_icd) {
    true => 4,
    false => 16,
};

/// The floats a short encoder scales, in one dispatch.
const SHORT_FLOATS: u64 = 64;

/// The kernel, and the fixture it is in.
const FIXTURE: &str = "examples/runtime_examples.ll";
const KERNEL: &str = "scale_and_offset";

/// The side of the render pass's target.
const SIDE: u64 = 64;

/// A state of the kernel labeled `label`.
fn state(device: &Device, label: &str) -> ComputePipelineState {
    let function = library(device, FIXTURE)
        .new_function_with_name(KERNEL)
        .unwrap();
    let options = PipelineOptions {
        label: Some(label.to_owned()),
        ..PipelineOptions::default()
    };
    (device.new_compute_pipeline_state_with_function(&function, &options)).unwrap()
}

/// A compute encoder of `command_buffer` that scales `count` floats of `input` into `output` with
/// `state`, `times` over.
fn scale(
    command_buffer: &CommandBuffer,
    state: &ComputePipelineState,
    (input, output): (&Buffer, &Buffer),
    count: u64,
    times: usize,
) {
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(input), 0, 0).unwrap();
    encoder.set_buffer(Some(output), 0, 1).unwrap();
    encoder.set_bytes(&1.0f32.to_le_bytes(), 2).unwrap();
    for _ in 0..times {
        (encoder.dispatch_threads(Size::new(count, 1, 1), Size::new(256, 1, 1))).unwrap();
    }
    encoder.end_encoding().unwrap();
}

/// Commit `command_buffer` and hold that it completed.
fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The milliseconds a command buffer's line gives the encoder of label hash `hash`, after
/// "for ".
fn encoder_ms(line: &str, hash: &str) -> f64 {
    let at = line
        .find(&format!("[{hash}]"))
        .unwrap_or_else(|| panic!("no encoder of {hash} in {line}"));
    let rest = &line[at..];
    let rest = rest.split(';').next().unwrap();
    let millis = rest
        .split(" for ")
        .nth(1)
        .and_then(|time| time.strip_suffix(" ms"))
        .unwrap_or_else(|| panic!("no time of {hash} in {line}"));
    millis.parse().unwrap()
}

#[test]
fn a_command_buffer_s_encoders_are_timed_apart_under_their_states_label_hashes() {
    let scratch = Scratch::new("encoder-times");
    let heard = Arc::new(Mutex::new(Vec::new()));
    let hearing = Arc::clone(&heard);
    let mut options = options(&scratch.0);
    options.messages = Some(MessageSink::new(move |message| {
        if message.id_name == ID_NAME {
            hearing.lock().unwrap().push(message.text.clone());
        }
    }));
    let device = Device::system_default_with(&options).unwrap();
    let name = device.name().to_owned();
    let labels = ["short-before", "long", "short-after"];
    let states: Vec<ComputePipelineState> =
        labels.iter().map(|label| state(&device, label)).collect();
    let hashes: Vec<String> = (labels.iter())
        .map(|label| label_hash(Some(label), &[KERNEL]))
        .collect();
    let bytes = LONG_FLOATS * 4;
    let input = device
        .new_buffer(bytes, ResourceOptions::private())
        .unwrap();
    let output = device
        .new_buffer(bytes, ResourceOptions::private())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let buffers = (&input, &output);
    let encode = |dispatches| {
        let command_buffer = queue.command_buffer().unwrap();
        scale(&command_buffer, &states[0], buffers, SHORT_FLOATS, 1);
        scale(
            &command_buffer,
            &states[1],
            buffers,
            LONG_FLOATS,
            dispatches,
        );
        scale(&command_buffer, &states[2], buffers, SHORT_FLOATS, 1);
        command_buffer
    };
    // Warm, with the times off: the shapes' pipelines are made, so no dispatch waits for its
    // making in a segment of its own, which no encoder records on after (`encoder_fold.rs`).
    run(&encode(1));
    // A line for every command buffer from here on.
    (device.shared().timing.encoders()).time_encoders(Some(Duration::ZERO));
    let folded = device.work_counts().encoders_folded;
    let command_buffer = encode(LONG_DISPATCHES);
    // The second and the third record on in the first's `VkCommandBuffer`.
    assert_eq!(
        device.work_counts().encoders_folded,
        folded + 2,
        "{name}: the encoders were not folded"
    );
    run(&command_buffer);
    let id = command_buffer.object_id().get();
    let line = (heard.lock().unwrap().iter())
        .find(|line| line.contains(&format!(" command buffer {id}: ")))
        .cloned()
        .unwrap_or_else(|| panic!("{name}: no line of command buffer {id}"));
    println!("ENCODER-TIMES {name}: {line}");
    assert!(line.contains(" over 3 encoder(s)"), "{line}");
    assert!(!line.contains("untimed"), "{line}");
    // In order, each under its state's hash, one dispatch each but the long one's.
    let places: Vec<usize> = (hashes.iter())
        .map(|hash| line.find(&format!("[{hash}]")).unwrap())
        .collect();
    assert!(places.windows(2).all(|pair| pair[0] < pair[1]), "{line}");
    assert!(
        line.contains(&format!(
            "[{}] {LONG_DISPATCHES} dispatch(es)/draw(s)",
            hashes[1]
        )),
        "{line}"
    );
    let times: Vec<f64> = hashes.iter().map(|hash| encoder_ms(&line, hash)).collect();
    assert!(
        times[1] > times[0] && times[1] > times[2],
        "{name}: the long encoder's time is not the most: {times:?}"
    );
    drop((command_buffer, queue, states, input, output));
    drop(device);
    // The summary the teardown printed (or the completion's, where its window had passed).
    let rows = |heard: &[String]| -> Vec<String> {
        (heard.iter())
            .filter(|line| line.contains(" #") && hashes.iter().any(|hash| line.contains(hash)))
            .cloned()
            .collect()
    };
    let deadline = Instant::now() + WAIT;
    while rows(&heard.lock().unwrap()).len() < 3 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    let heard = heard.lock().unwrap();
    let rows = rows(&heard);
    for row in &rows {
        println!("ENCODER-TIMES {name}: {row}");
    }
    assert_eq!(rows.len(), 3, "{name}: {heard:?}");
    assert!(
        rows[0].contains(&format!(
            "Compute [{}]: 1 encoder(s), {LONG_DISPATCHES}",
            hashes[1]
        )),
        "{name}: the long encoder's row is not first: {rows:?}"
    );
}

/// A render encoder's draws again count among its draws: [`DRAWS`] draws of one rectangle, each
/// after the first drawn again as before, or rebound where `rebound` binds other bytes before
/// each, name `DRAWS` draws in their command buffer's line (the module docs).
#[test]
fn an_encoder_s_draws_again_count_among_its_draws() {
    let scratch = Scratch::new("encoder-times-again");
    let heard = Arc::new(Mutex::new(Vec::new()));
    let hearing = Arc::clone(&heard);
    let mut options = options(&scratch.0);
    options.messages = Some(MessageSink::new(move |message| {
        if message.id_name == ID_NAME {
            hearing.lock().unwrap().push(message.text.clone());
        }
    }));
    let device = device_with_room(&options).unwrap();
    let name = device.name().to_owned();
    let state = adding_state(&device);
    let names = state.function_names();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let hash = label_hash(state.label(), &names);
    let rectangles = |encoder: &mut RenderCommandEncoder, rebound: bool| {
        set_rectangle(encoder, &state);
        for at in 0..DRAWS {
            if rebound && at > 0 {
                let color = bytes_of(&[(at + 1) as f32 / 64.0; 4], f32::to_le_bytes);
                encoder.set_fragment_bytes(&color, 0).unwrap();
            }
            rectangle(encoder);
        }
    };
    for rebound in [false, true] {
        warm(&device, |encoder| rectangles(encoder, rebound));
    }
    (device.shared().timing.encoders()).time_encoders(Some(Duration::ZERO));
    let again: [(&str, fn(&WorkCounts) -> u64); 2] = [
        ("drawn again as before", |counts| counts.draws_as_before),
        ("rebound", |counts| counts.draws_rebound),
    ];
    for (rebound, (how, count)) in [false, true].into_iter().zip(again) {
        heard.lock().unwrap().clear();
        let ran = run_pass(&device, |encoder| rectangles(encoder, rebound));
        // The draws again ran: without them, every draw takes every step and counts there.
        assert!(
            ran.counted(count) > 0,
            "{name}: no draw {how}: {}",
            ran.moved()
        );
        let render = format!("Render [{hash}] ");
        let deadline = Instant::now() + WAIT;
        let line = loop {
            let found = (heard.lock().unwrap().iter())
                .find(|line| line.contains(&render))
                .cloned();
            match found {
                Some(line) => break line,
                None if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
                None => panic!("{name}: no line names the render encoder of {hash}"),
            }
        };
        println!("ENCODER-TIMES {name} ({how}): {line}");
        assert!(
            line.contains(&format!("{render}{DRAWS} dispatch(es)/draw(s)")),
            "{name}: the {} draw(s) {how} are not all among the encoder's {DRAWS}: {line}",
            ran.counted(count),
        );
    }
}

/// The `VkCommandBuffer`s encoders began, the pools' slots, as a command buffer of `device` of a
/// compute encoder, a render pass of one draw and a blit encoder ran.
fn mixed(device: &Device) -> (u64, u64, usize) {
    let scale_state = state(device, "mixed");
    let draw_state = render_state(device, FIXTURE, "colored_vertex", "colored_fragment");
    let input = (device.new_buffer(SHORT_FLOATS * 4, ResourceOptions::private())).unwrap();
    let output = (device.new_buffer(SHORT_FLOATS * 4, ResourceOptions::private())).unwrap();
    let floats = |values: Vec<f32>| {
        let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
        (device.new_buffer_with_bytes(&bytes, ResourceOptions::shared())).unwrap()
    };
    let positions = floats(vec![-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0]);
    let colors = floats((0..4).flat_map(|_| [1.0, 0.2, 0.0, 1.0]).collect());
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let target = device.new_texture(&target).unwrap();
    let queue = device.new_command_queue().unwrap();
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    scale(
        &command_buffer,
        &scale_state,
        (&input, &output),
        SHORT_FLOATS,
        1,
    );
    let mut render = command_buffer
        .render_command_encoder(&pass(&target))
        .unwrap();
    render.set_render_pipeline_state(&draw_state).unwrap();
    render.set_vertex_buffer(Some(&positions), 0, 0).unwrap();
    render.set_vertex_buffer(Some(&colors), 0, 1).unwrap();
    (render.draw_primitives(PrimitiveType::TriangleStrip, 0, 4, 1, 0)).unwrap();
    render.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(&output, 0..SHORT_FLOATS * 4, 7).unwrap();
    blit.end_encoding().unwrap();
    let after = device.work_counts();
    run(&command_buffer);
    let slots = device.shared().timing.encoders().slots();
    (
        after.encoder_command_buffers_begun - before.encoder_command_buffers_begun,
        after.command_buffers_begun_continuations - before.command_buffers_begun_continuations,
        slots,
    )
}

#[test]
fn with_the_times_off_an_encoder_makes_no_query_and_begins_no_command_buffer_more() {
    if metal2vulkan::env_vars::encoder_times().is_some() {
        eprintln!("ENCODER-TIMES: the process asks for the encoder times; nothing to hold off");
        return;
    }
    let scratch = Scratch::new("encoder-times-off");
    let device = Device::system_default_with(&options(&scratch.0)).unwrap();
    let name = device.name().to_owned();
    // Warm: the pipelines are made, so no draw or dispatch waits for its making in a segment.
    mixed(&device);
    let off = mixed(&device);
    // The same shape again: what the off run begins is the command buffer's own.
    assert_eq!(mixed(&device), off, "{name}: two runs off began apart");
    assert_eq!(off.2, 0, "{name}: a query pool made with the times off");
    device.shared().timing.encoders().time_encoders(None);
    let on = mixed(&device);
    println!("ENCODER-TIMES {name}: off began {off:?}, on {on:?} (begun, continuations, slots)");
    assert!(on.2 > 0, "{name}: no query pool made with the times on");
    assert_eq!(
        (on.0, on.1),
        (off.0 + 1, off.1 + 1),
        "{name}: on, the render pass's own command buffer alone is begun more"
    );
}
