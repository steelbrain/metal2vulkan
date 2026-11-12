//! A handle handed out beside another queue's long work: a new `Private` buffer's `VkBuffer`
//! (`Buffer::vk_buffer`), its memory exported (`Buffer::export_memory_fd`) and a new `Private`
//! texture's `VkImage` (`Texture::vk_image`) make their own first work, the buffer's zeros and
//! the image's move, and wait for that alone (`Device::settle_mine`): not for another command
//! queue's command buffer, whose batch took the first work of a resource made before it. A
//! consumer's decoder exports each new picture's memory while the guest's compositor keeps the
//! GPU busy; the hand-out once took every resource's first work and waited for the other
//! queues' batches that did some, holding the decoder for their GPU time.
//!
//! The busy work is a command buffer of fills of a 256 MiB `Private` buffer on a command queue
//! of another `VkQueue` than the device's first (`QueueSharingPolicy::PerQueue`), as many as
//! take about [`BUSY`] of the device's GPU time, timed alone first. Each hand-out is timed and
//! the busy command buffer's status read as it returns. Where the device's queues run side by
//! side for that work, each hand-out returns while the busy work still runs: a buffer's on a
//! device that reserved a staging queue (its fills go there, `device/staging.rs`), every one where
//! the busy `VkQueue`'s family is another than the first's (MoltenVK's families are Metal
//! command queues of their own). Elsewhere (one `VkQueue`, or the graphics family's queues of a
//! GPU that runs them one after another) the times are printed (`HAND-OUT-WAITS`), and the
//! hand-outs' paths: the staging submission or the internal one each made.
//!
//! Its timing assumption: the GPU runs the test's work alone, so that the busy work's fills take
//! the time they took timed alone. Beside another process's GPU work the fills timed slow and the
//! busy work then ended early: on an RTX 5080 Laptop GPU inside the runtime suite (beside the
//! dma-buf leak probe) a fill timed 1.59 ms, the busy work of 378 of them was done within 270 ms
//! of the 601 ms they predicted, and each hand-out's staging submission waited 133 ms for the GPU,
//! so a hand-out returned after the busy work. Where a hand-out found the busy work done before
//! half the time its fills predict, the test says that its assumption failed, not that the
//! hand-out waited. The check rounds run it alone on the GPU.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::blit_holds::{Scratch, WAIT};
use crate::error::{Error, RuntimeGap};
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, DeviceKind, DeviceOptions,
    ResourceOptions, ShaderCacheOptions, TextureDescriptor,
};
use metal2vulkan::device::QueueSharingPolicy;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::time::{Duration, Instant};

/// The busy work's GPU time, about.
const BUSY: Duration = Duration::from_millis(600);

/// The bytes each of its fills writes.
const TARGET: u64 = 256 << 20;

/// The fills timed alone, and the most the busy command buffer takes.
const TIMED: u32 = 4;
const MOST_FILLS: u32 = 4096;

/// A 3840 x 2160 NV12 picture's bytes: what the buffers handed out hold.
const PICTURE: u64 = 3840 * 2160 * 3 / 2;

/// The device the cfg selects, its shader cache in `cache`, each command queue on the `VkQueue`
/// it took at its creation.
fn device(cache: &Scratch) -> Device {
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
            directory: Some(cache.0.clone()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    options.policy.queue_sharing = QueueSharingPolicy::PerQueue;
    Device::system_default_with(&options).unwrap()
}

/// A command queue on another `VkQueue` than the device's first, with the queues made before it
/// (kept, so the round robin moves on); `None` where the device has one `VkQueue`.
fn other_queue(device: &Device) -> (Option<CommandQueue>, Vec<CommandQueue>) {
    let mut made = Vec::new();
    for _ in 0..8 {
        let queue = device.new_command_queue().unwrap();
        if queue.vk_queue_sharing().0 != 0 {
            return (Some(queue), made);
        }
        made.push(queue);
    }
    (None, made)
}

/// A command buffer of `queue`'s of a fill of `marker` where there is one, then `fills` fills of
/// `target`, committed.
fn fills(
    queue: &CommandQueue,
    (target, marker): (&Buffer, Option<&Buffer>),
    fills: u32,
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    if let Some(marker) = marker {
        blit.fill_buffer(marker, 0..marker.length(), 0).unwrap();
    }
    for value in 0..fills {
        blit.fill_buffer(target, 0..TARGET, value as u8).unwrap();
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
}

/// One hand-out's time, whether the busy work still ran as it returned, when it returned after
/// the busy work's submission, and the staging and internal submissions it made.
struct HandOut {
    took: Duration,
    busy_ran: bool,
    returned: Duration,
    staging: u64,
    internal: u64,
}

impl HandOut {
    /// Hand out through `hand_out`, timed, reading `busy`'s status as it returns; `busy` was
    /// submitted at `submitted`.
    fn of(
        device: &Device,
        (busy, submitted): (&CommandBuffer, Instant),
        hand_out: impl FnOnce(),
    ) -> HandOut {
        let before = device.work_counts();
        let start = Instant::now();
        hand_out();
        let took = start.elapsed();
        let busy_ran = busy.status() != CommandBufferStatus::Completed;
        let returned = submitted.elapsed();
        let after = device.work_counts();
        HandOut {
            took,
            busy_ran,
            returned,
            staging: after.staging_submissions - before.staging_submissions,
            internal: after.internal_submissions - before.internal_submissions,
        }
    }

    fn line(&self, what: &str) -> String {
        format!(
            "{what} {:.2} ms, busy work {} as it returned, staging submissions {} internal {}",
            self.took.as_secs_f64() * 1e3,
            match self.busy_ran {
                true => "running",
                false => "done",
            },
            self.staging,
            self.internal
        )
    }
}

/// Beside another queue's long fills, a new `Private` buffer's `VkBuffer` and exported memory,
/// and a new `Private` texture's `VkImage`, are handed out without waiting for them where the
/// device runs that work beside the hand-out's.
#[test]
fn a_hand_out_makes_its_own_first_work_without_waiting_for_another_queue_s_work() {
    let scratch = Scratch::new("hand-out-waits");
    let device = device(&scratch);
    let name = device.name().to_owned();
    // The device's own checks submit beside the program's: settled first, so each hand-out's
    // submissions are its own.
    assert!(
        device.settle_in_place_checks(WAIT),
        "{name}: the in-place checks settle"
    );
    assert!(
        device.settle_submit_checks(WAIT),
        "{name}: the submit checks settle"
    );
    assert!(
        device.settle_queue_switch_check(WAIT),
        "{name}: the queue check settles"
    );
    let (busy_queue, _made) = other_queue(&device);
    let Some(busy_queue) = busy_queue else {
        eprintln!("HAND-OUT-WAITS on {name}: one VkQueue; nothing runs beside the hand-outs");
        return;
    };
    let shared = device.shared();
    let busy_family = shared.queue_family_of(busy_queue.vk_queue_sharing().0);
    let beside_family = busy_family != shared.queue_family_of(0);
    let staging = device.has_staging_queue();
    let target = device
        .new_buffer(TARGET, ResourceOptions::private())
        .unwrap();
    // The fills' GPU time, alone.
    let timed = fills(&busy_queue, (&target, None), TIMED);
    let start = Instant::now();
    timed.wait_until_completed_for(WAIT).unwrap();
    let each = start.elapsed() / TIMED;
    let count = (BUSY.as_secs_f64() / each.as_secs_f64().max(1e-6)).ceil() as u32;
    let count = count.clamp(TIMED, MOST_FILLS);
    // A resource made just before the busy work, which the busy work fills too: its batch, whose
    // commands reach it, takes its zeros (a batch takes the fills of memory its commands may
    // reach alone, `device/staging.rs`), so a hand-out that waits for the batches on the other
    // `VkQueue`s that did first work waits for it.
    let marker = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let submitted = device.work_counts().queue_submissions;
    let busy = fills(&busy_queue, (&target, Some(&marker)), count);
    let started = Instant::now();
    while device.work_counts().queue_submissions == submitted {
        assert!(
            started.elapsed() < WAIT,
            "the busy work was never submitted"
        );
        std::thread::yield_now();
    }
    let firsts = &shared.textures.first_moves;
    assert!(
        !firsts.fills_wait(marker.at(0).0),
        "the busy batch took the marker's zeros"
    );
    let mut lines = Vec::new();
    // A new `Private` buffer's `VkBuffer`, its zeros waiting.
    let buffer = device
        .new_buffer(PICTURE, ResourceOptions::private())
        .unwrap();
    assert!(firsts.fills_wait(buffer.at(0).0), "the buffer's zeros wait");
    let handed = HandOut::of(&device, (&busy, started), || {
        // SAFETY: the handle is not used.
        let _ = unsafe { buffer.vk_buffer() };
    });
    assert!(
        !firsts.fills_wait(buffer.at(0).0),
        "the hand-out made the zeros"
    );
    lines.push(handed.line("vk_buffer"));
    let mut buffers = vec![(handed, "vk_buffer")];
    // A new `Private` exportable buffer's memory exported, its zeros waiting.
    match device.new_exportable_buffer(PICTURE, ResourceOptions::private()) {
        Ok(exportable) => {
            assert!(
                firsts.fills_wait(exportable.at(0).0),
                "the picture's zeros wait"
            );
            let exported = HandOut::of(&device, (&busy, started), || {
                drop(exportable.export_memory_fd().unwrap());
            });
            lines.push(exported.line("export_memory_fd"));
            buffers.push((exported, "export_memory_fd"));
        }
        Err(Error::NotImplemented(RuntimeGap::ExportableMemory)) => {
            lines.push("no exportable buffers".into());
        }
        Err(error) => panic!("{name}: {error:?}"),
    }
    // A new `Private` texture's `VkImage`, its move waiting.
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, 1920, 1080, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let texture = device.new_texture(&descriptor).unwrap();
    let image = HandOut::of(&device, (&busy, started), || {
        // SAFETY: the handle is not used.
        let _ = unsafe { texture.vk_image() };
    });
    lines.push(image.line("vk_image"));
    let ran = started.elapsed();
    busy.wait_until_completed_for(WAIT).unwrap();
    let busy_took = started.elapsed();
    eprintln!(
        "HAND-OUT-WAITS on {name}: busy work of {count} fills of {} MiB on VkQueue {} (family \
         {busy_family}, the first's {}), {:.0} ms ({:.2} ms a fill alone), the hand-outs within \
         its first {:.0} ms; staging queue {staging} | {}",
        TARGET >> 20,
        busy_queue.vk_queue_sharing().0,
        shared.queue_family_of(0),
        busy_took.as_secs_f64() * 1e3,
        each.as_secs_f64() * 1e3,
        ran.as_secs_f64() * 1e3,
        lines.join(" | ")
    );
    // A buffer's fills go to the staging queue where there is one, else internally.
    for (handed, what) in &buffers {
        assert_eq!(
            (handed.staging, handed.internal),
            match staging {
                true => (1, 0),
                false => (0, 1),
            },
            "{name}: {what}'s submission"
        );
    }
    assert_eq!(
        (image.staging, image.internal),
        (0, 1),
        "{name}: vk_image's submission"
    );
    // The timing assumption (the module docs): the busy work runs about as long as its fills
    // timed alone predict, so a hand-out that found it done before half that time found it done
    // early, not by waiting for it.
    let predicted = each * count;
    let early = (buffers.iter().map(|(handed, _)| handed))
        .chain([&image])
        .filter(|handed| !handed.busy_ran)
        .map(|handed| handed.returned)
        .min()
        .filter(|&done_by| done_by < predicted / 2);
    if let Some(done_by) = early {
        panic!(
            "{name}: the busy work was done within {:.0} ms of its submission, where its fills \
             timed alone predict {:.0} ms: other GPU work ran beside the test's timing of the \
             fills (the test's timing assumption: the GPU runs its work alone)",
            done_by.as_secs_f64() * 1e3,
            predicted.as_secs_f64() * 1e3
        );
    }
    if staging || beside_family {
        for (handed, what) in &buffers {
            assert!(handed.busy_ran, "{name}: {what} waited for the busy work");
        }
    }
    if beside_family {
        assert!(image.busy_ran, "{name}: vk_image waited for the busy work");
    }
}
