//! What an encoder was handed lives until its command buffer completes, on the device the cfg
//! selects, under the validation layer. Metal (with retained references) holds every object set on
//! an encoder until then: one a command used, one replaced or unbound before any command used it,
//! and one bound after the last command or with none at all. A command buffer never committed lets
//! go of them as it is released. (The M3 Max under macOS 27.0.1, probes `dealloc-probe` and
//! `dealloc-probe2`, 3 runs of each case: a no-copy buffer so bound to a compute or render encoder,
//! or under a texture so bound, then released by the program, ran its deallocator only after its
//! command buffer completed, never at the encoder's end nor while the committed command buffer
//! waited for an event; and after an uncommitted command buffer's release.)
//!
//! Each test binds no-copy buffers so, and a texture over one, releases its own handles while the
//! command buffer waits for an event, and checks that no deallocator ran until the command buffer
//! completed, and that each ran once from then on, on a thread of the runtime's; or, never
//! committed, once it was released. A completed command buffer that held a host import last posts
//! its free, which a deallocator waits for, to the device's free thread, where the deallocator runs
//! after it, possibly after the command buffer's waiters returned (the runtime's `free_thread.rs`):
//! so a deallocator that had not run as the wait returned is one whose free was posted
//! (`WorkCounts::host_import_frees_posted`). Metal's own deallocator ran after its command buffer's
//! wait returned now and then on an M4 (`host_memory.rs`).

use super::host_memory::{
    no_copy, not_importable, on_device, recording, Mapping, Seen, GUEST_PAGE,
};
use super::rebind_reach::{deallocated, POINTER_WALKER};
use super::{bound_buffers, encode_case, pipeline_state, public_case, CaseResources, WAIT};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, Device, EventRef, LoadAction, RenderPassDescriptor,
    ResourceOptions, SharedEvent, StoreAction, TextureDescriptor,
};
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Buffer indices the walker's kernel does not read.
const SPARE: u32 = 29;

/// A texture over a handed buffer: `RGBA8Unorm`, [`WIDE`] by [`HIGH`], its rows [`ROW`] bytes
/// apart, within one guest page.
const WIDE: u64 = 64;
const HIGH: u64 = 16;
const ROW: u64 = WIDE * 4;

/// A no-copy buffer over a guest page of its own, its deallocator's runs in its `Seen`; `None`,
/// printed, where the device imports no host memory.
fn handed(device: &Device) -> Option<(Buffer, Arc<Seen>)> {
    let seen = Arc::new(Seen::default());
    let mapping = Mapping::anonymous(GUEST_PAGE);
    let pointer = mapping.pointer();
    // SAFETY: the deallocator releases the mapping, after the buffer's last use.
    let made = unsafe {
        no_copy(
            device,
            pointer,
            GUEST_PAGE,
            ResourceOptions::default(),
            Some(recording(&seen, mapping)),
        )
    };
    match made {
        Ok(buffer) => Some((buffer, seen)),
        Err(error) if not_importable(&error) => {
            eprintln!("HANDED: the device imports no host memory: {error:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

/// No deallocator of `seen` ran yet, `when`.
fn none_ran(seen: &[(&str, &Arc<Seen>)], when: &str) {
    for (what, seen) in seen {
        assert_eq!(
            seen.runs.load(Ordering::SeqCst),
            0,
            "{what}'s deallocator ran {when}"
        );
    }
}

/// Signal `event`, which `command_buffer` of `device` waits for: once the command buffer's wait
/// returned, each deallocator of `seen` runs once, on a thread of the runtime's, not this one; and
/// each that had not run as the wait returned had its free posted to the device's free thread (the
/// module docs).
fn each_runs_once_completed(
    device: &Device,
    event: &SharedEvent,
    seen: &[(&str, &Arc<Seen>)],
    command_buffer: &CommandBuffer,
) {
    let posted = || device.work_counts().host_import_frees_posted;
    let before = posted();
    event.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let late: Vec<&str> = (seen.iter())
        .filter(|(_, seen)| seen.runs.load(Ordering::SeqCst) == 0)
        .map(|(what, _)| *what)
        .collect();
    for (what, seen) in seen {
        deallocated(seen);
        assert_eq!(
            seen.runs.load(Ordering::SeqCst),
            1,
            "{what}'s deallocator ran once"
        );
        assert_ne!(
            *seen.thread.lock().unwrap(),
            Some(std::thread::current().id()),
            "{what}'s deallocator ran on the program's thread"
        );
    }
    // Read once every deallocator ran: each posted free was posted before its deallocator ran.
    let posted = posted() - before;
    eprintln!(
        "HANDED on {}: ran after the wait returned {late:?}, import frees posted {posted}",
        device.name()
    );
    assert!(
        posted >= late.len() as u64,
        "{late:?} ran after the wait returned, {posted} import frees posted"
    );
}

/// A command buffer of a new queue of `device`'s that waits for `event` before its work.
fn waiting(device: &Device, event: &SharedEvent) -> CommandBuffer {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::from(event), 1)
        .unwrap();
    command_buffer
}

/// A compute encoder's bindings: a no-copy buffer replaced before the dispatch, and one bound after
/// it, each released by the program once the command buffer is committed, live until it completed
/// (probe `dealloc-probe` and `dealloc-probe2` mode 3).
#[test]
fn a_compute_encoder_s_bindings_no_dispatch_used_live_until_its_command_buffer_completes() {
    on_device(|device| {
        let walker = public_case(POINTER_WALKER).unwrap();
        let Some((replaced, replaced_seen)) = handed(device) else {
            return;
        };
        let (after, after_seen) = handed(device).unwrap();
        let other = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        let state = pipeline_state(device, &walker).unwrap();
        let holding = CaseResources::new(device, &walker).unwrap();
        let buffers = bound_buffers(device, &walker.compute, &holding).unwrap();
        let event = device.new_shared_event().unwrap();
        let command_buffer = waiting(device, &event);
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_buffer(Some(&replaced), 0, SPARE).unwrap();
        encoder.set_buffer(Some(&other), 0, SPARE).unwrap();
        encode_case(&mut encoder, &state, &walker.compute, &buffers, &holding).unwrap();
        encoder.set_buffer(Some(&after), 0, SPARE + 1).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        drop((replaced, after));
        let seen = [("replaced", &replaced_seen), ("after", &after_seen)];
        none_ran(&seen, "while its command buffer waited");
        each_runs_once_completed(device, &event, &seen, &command_buffer);
    });
}

/// A render encoder's bindings with no draw: a no-copy buffer replaced as the vertex stage's, a
/// texture over one replaced as the fragment stage's, and a no-copy buffer bound last, each
/// released by the program once the command buffer is committed, live until it completed (probe
/// `dealloc-probe2` modes 0 to 2).
#[test]
fn a_render_encoder_s_bindings_no_draw_used_live_until_its_command_buffer_completes() {
    on_device(|device| {
        let Some((vertex, vertex_seen)) = handed(device) else {
            return;
        };
        let (under, under_seen) = handed(device).unwrap();
        let (after, after_seen) = handed(device).unwrap();
        let mut sampled = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDE, HIGH, false);
        sampled.usage = TextureUsage::SHADER_READ;
        sampled.resource_options = under.resource_options();
        let texture = under.new_texture(&sampled, 0, ROW).unwrap();
        let other_texture = device.new_texture(&sampled).unwrap();
        let other = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 8, 8, false);
        target.usage = TextureUsage::RENDER_TARGET;
        let target = device.new_texture(&target).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target);
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(0.0, 0.0, 1.0, 1.0);
        color.store_action = StoreAction::Store;
        let event = device.new_shared_event().unwrap();
        let command_buffer = waiting(device, &event);
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_vertex_buffer(Some(&vertex), 0, 0).unwrap();
        encoder.set_vertex_buffer(Some(&other), 0, 0).unwrap();
        encoder.set_fragment_texture(Some(&texture), 0).unwrap();
        encoder
            .set_fragment_texture(Some(&other_texture), 0)
            .unwrap();
        encoder.set_vertex_buffer(Some(&after), 0, 1).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        drop((vertex, under, texture, after));
        let seen = [
            ("vertex", &vertex_seen),
            ("the texture's", &under_seen),
            ("after", &after_seen),
        ];
        none_ran(&seen, "while its command buffer waited");
        each_runs_once_completed(device, &event, &seen, &command_buffer);
    });
}

/// A command buffer never committed holds a compute encoder's bindings, a no-copy buffer replaced
/// and one bound last, until it is released (probe `dealloc-probe2` mode 4).
#[test]
fn a_command_buffer_never_committed_holds_its_encoder_s_bindings_until_its_release() {
    on_device(|device| {
        let Some((replaced, replaced_seen)) = handed(device) else {
            return;
        };
        let (after, after_seen) = handed(device).unwrap();
        let other = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_buffer(Some(&replaced), 0, 0).unwrap();
        encoder.set_buffer(Some(&other), 0, 0).unwrap();
        encoder.set_buffer(Some(&after), 0, 1).unwrap();
        encoder.end_encoding().unwrap();
        drop((replaced, after));
        let seen = [("replaced", &replaced_seen), ("after", &after_seen)];
        none_ran(&seen, "while its command buffer lived");
        drop(command_buffer);
        for (_, seen) in seen {
            deallocated(seen);
            assert_eq!(seen.runs.load(Ordering::SeqCst), 1);
        }
    });
}
