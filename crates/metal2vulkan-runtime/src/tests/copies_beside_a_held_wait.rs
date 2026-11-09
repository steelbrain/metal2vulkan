//! A `Shared` texture's CPU copies while a command buffer the program committed waits for an
//! event that only a command buffer committed after the copies signals. Metal's `getBytes` and
//! `replaceRegion` of shared storage are CPU copies that wait for no GPU work, so they return, and
//! the waiting command buffer runs once the signal does.
//!
//! The case that matters is a device that copies a texture through a submission of its own on
//! its first `VkQueue` (`device/submit.rs`): one with no `VK_EXT_host_image_copy` and no family
//! of transfers alone for a staging queue (RADV on the RX 550, whose one graphics queue every
//! command queue shares). Such a submission would wait forever behind a batch on that `VkQueue`
//! that waits for the later signal. None is there: the scheduler holds a batch whose event wait no
//! submitted signal and no host set reaches, unsubmitted, so no `VkQueue` holds a wait for a
//! signal not yet submitted (`scheduler/mod.rs`, `event_timeline.rs`). Each case prints the
//! waiting command buffer's status while the copies ran (`Committed`: held on the host) and the
//! internal submissions the copies made, the evidence of the path they took.
//!
//! The copies run on a thread of their own, bounded by [`BOUND`]: where they have not returned by
//! then, the test commits the signal all the same, which releases a copy held behind the wait, and
//! fails, so a hang fails the case rather than wedging the run.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    CommandBufferStatus, CommandQueue, Device, EventRef, Region, ResourceOptions, Texture,
    TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// The texture's side, in texels.
const SIDE: u64 = 64;

/// The longest the copies may take before the case calls them held behind the wait.
const BOUND: Duration = Duration::from_secs(10);

/// The one-texel writes the copies make: one past the copies host calls stage for the device's
/// next submission before one is made for them (`texture/first_move.rs`'s `STAGED_COPIES`, 4096),
/// so where the device stages them, the last write makes that submission and waits for it.
const WRITES: u64 = 4097;

/// A `Shared` `RGBA8Unorm` texture of [`SIDE`] square, written whole and read back, so its first
/// work is done before any wait is committed.
fn texture(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes = vec![7; (4 * SIDE * SIDE) as usize];
    let region = Region::new_2d(0, 0, SIDE, SIDE);
    (texture.replace_region(region, 0, 0, &bytes, 4 * SIDE, 0)).unwrap();
    let mut back = vec![0; bytes.len()];
    (texture.get_bytes(&mut back, 4 * SIDE, 0, region, 0, 0)).unwrap();
    assert_eq!(back, bytes, "the texture's first read");
    texture
}

/// The copies: a read of the whole level, [`WRITES`] writes of one texel each (texel `i` of the
/// first rows to the byte `i`), and a read again, which must hold the writes.
fn copies(texture: &Texture) {
    let region = Region::new_2d(0, 0, SIDE, SIDE);
    let mut back = vec![0; (4 * SIDE * SIDE) as usize];
    (texture.get_bytes(&mut back, 4 * SIDE, 0, region, 0, 0)).unwrap();
    for i in 0..WRITES {
        let at = Region::new_2d(i % SIDE, i / SIDE % SIDE, 1, 1);
        (texture.replace_region(at, 0, 0, &[i as u8; 4], 4, 0)).unwrap();
    }
    (texture.get_bytes(&mut back, 4 * SIDE, 0, region, 0, 0)).unwrap();
    // The last write to each texel is the one past the level's texel count, where `WRITES`
    // wraps.
    let texels = SIDE * SIDE;
    for i in 0..texels {
        let last = (i..WRITES).step_by(texels as usize).last().unwrap_or(i);
        let at = (4 * i) as usize;
        assert_eq!(
            back[at..at + 4],
            [last as u8; 4],
            "texel {i} after the writes"
        );
    }
}

/// Commit a command buffer that waits for `event` at 1 and then fills a word, run [`copies`] on
/// a thread of their own, and only then `release` the wait (given another queue): the copies
/// must return within [`BOUND`], and the waiter must run once released.
fn copies_beside_a_held_wait(
    device: &Device,
    event: EventRef<'_>,
    kind: &str,
    release: impl FnOnce(&CommandQueue),
) {
    let texture = texture(device);
    let (waiting, signalling) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let word = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let waiter = waiting.command_buffer().unwrap();
    waiter.encode_wait_for_event(event, 1).unwrap();
    let mut blit = waiter.blit_command_encoder().unwrap();
    blit.fill_buffer(&word, 0..4096, 1).unwrap();
    blit.end_encoding().unwrap();
    waiter.commit().unwrap();
    let before = device.work_counts();
    let (done, finished) = mpsc::channel();
    let (answer, held) = std::thread::scope(|scope| {
        scope.spawn(|| {
            let started = Instant::now();
            copies(&texture);
            let _ = done.send(started.elapsed());
        });
        let answer = finished.recv_timeout(BOUND);
        let held = waiter.status();
        // After the copies returned, or after the bound to release them.
        release(&signalling);
        (answer, held)
    });
    let internal = device.work_counts().internal_submissions - before.internal_submissions;
    let name = device.name();
    eprintln!(
        "COPIES-BESIDE-A-HELD-WAIT on {name}, {kind}: the copies took {answer:?}, {internal} \
         internal submission(s); the waiting command buffer was {held:?} meanwhile"
    );
    let took = answer.unwrap_or_else(|_| {
        panic!(
            "{kind}: the copies did not return in {BOUND:?} while a command buffer waited for a \
             signal committed after them ({held:?}): held behind the wait"
        )
    });
    assert!(took < BOUND);
    assert!(
        waiter.wait_until_completed_for(WAIT).is_ok(),
        "{kind}: the waiting command buffer did not complete once released"
    );
    assert_eq!(waiter.status(), CommandBufferStatus::Completed);
    assert_eq!(waiter.error(), None);
    let contents = word.contents().unwrap();
    // SAFETY: the buffer's 4096 bytes; the command buffer that wrote them completed, and nothing
    // else writes them.
    let filled = unsafe { std::slice::from_raw_parts(contents.as_ptr(), 4096) };
    assert!(
        filled.iter().all(|&byte| byte == 1),
        "{kind}: the waiter's fill"
    );
}

/// Commit, on `queue`, a command buffer that signals `event` at 1.
fn signal(queue: &CommandQueue, event: EventRef<'_>) {
    let signaller = queue.command_buffer().unwrap();
    signaller.encode_signal_event(event, 1).unwrap();
    signaller.commit().unwrap();
}

/// An `MTLEvent`, whose value no host call sets: its wait chains on the `VkQueue` once a signal
/// that reaches it is submitted, and is held on the host until then.
#[test]
fn cpu_copies_return_beside_a_command_buffer_waiting_for_a_later_event_signal() {
    let scratch = Scratch::new("copies-beside-event-wait");
    let device = device(&scratch.0);
    let event = device.new_event().unwrap();
    let event = EventRef::Event(&event);
    copies_beside_a_held_wait(&device, event, "an event", |queue| signal(queue, event));
}

/// An `MTLSharedEvent`, whose wait the host may release: held on the host until its value is
/// reached.
#[test]
fn cpu_copies_return_beside_a_command_buffer_waiting_for_a_later_shared_event_signal() {
    let scratch = Scratch::new("copies-beside-shared-event-wait");
    let device = device(&scratch.0);
    let event = device.new_shared_event().unwrap();
    let event = EventRef::Shared(&event);
    copies_beside_a_held_wait(&device, event, "a shared event", |queue| {
        signal(queue, event)
    });
}
