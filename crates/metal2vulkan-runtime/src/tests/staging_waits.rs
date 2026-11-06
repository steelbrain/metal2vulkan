//! Batches committed while a staging submission has not completed (`device/staging.rs`,
//! `command_buffer/submit.rs`): the first batch on a `VkQueue` waits for it on the GPU, and the
//! batches after it there take no wait of their own, as they follow that wait in the `VkQueue`'s
//! submission order. The staging submission waits on the GPU for a timeline the test signals
//! once every batch was submitted (`Device::hold_staging_submissions`), so the batches meet it
//! not complete whatever the host's timing. Unheld, the host may see a staging copy complete
//! before the first batch is submitted, and then no batch waits for it: the runtime lane's
//! `staging_batches` runs the same calls that way.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. A device with no staging queue, or whose `Managed` textures copy on
//! the host, says so and passes.

use crate::{
    BlitOption, CommandBufferStatus, CpuCacheMode, Device, DeviceKind, DeviceOptions,
    HazardTrackingMode, Origin, Region, ResourceOptions, Size, StorageMode, TextureDescriptor,
};
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::time::{Duration, Instant};

/// The longest a command buffer here may take once the hold is let go.
const WAIT: Duration = Duration::from_secs(60);

/// The texture's width and height.
const SIDE: u64 = 256;

/// Each batch's blit copies a square of the texture this many texels across.
const SQUARE: u64 = 32;

/// The command buffers committed on the one queue while the staging submission is held.
const BATCHES: u64 = 16;

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU.
fn device() -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// One `replace_region` of a `Managed` texture, queued for the staging queue, then [`BATCHES`]
/// command buffers on one queue, each a blit of a square of it into a buffer of its own, all
/// submitted while the staging submission that carries the copy is held on the GPU: the first
/// batch waited for it on the GPU and the others follow that wait, so the staging waits are one,
/// not one per batch (internal submissions wait for it themselves). Each blit reads the call's
/// bytes once the hold is let go.
#[test]
fn batches_submitted_while_a_staging_copy_runs_follow_the_first_one_s_wait() {
    let device = device();
    let name = device.name().to_owned();
    if !device.has_staging_queue() {
        eprintln!("STAGING-WAITS {name}: no staging queue; nothing to test");
        return;
    }
    let raw = device.shared().vk();
    let mut kind = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
    // SAFETY: the device is live and enables `timelineSemaphore`; `info` is live for the call.
    let hold = unsafe { raw.create_semaphore(&info, None) }.unwrap();
    let queue = device.new_command_queue().unwrap();
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    let texture = device.new_texture(&descriptor).unwrap();
    let outs: Vec<_> = (0..BATCHES)
        .map(|_| (device.new_buffer(4 * SQUARE * SQUARE, ResourceOptions::shared())).unwrap())
        .collect();
    let written: Vec<u8> = (0..4 * SIDE * SIDE)
        .map(|i| ((i * 7 + 3) % 251) as u8)
        .collect();
    device.hold_staging_submissions(Some((hold, 1)));
    let before = device.work_counts();
    let whole = Region::new_2d(0, 0, SIDE, SIDE);
    (texture.replace_region(whole, 0, 0, &written, 4 * SIDE, 0)).unwrap();
    let queued = device.work_counts().staging_copies_queued - before.staging_copies_queued;
    let squares: Vec<(u64, u64)> = (0..BATCHES)
        .map(|at| (SQUARE * (at % 8), SQUARE * (at / 8)))
        .collect();
    let mut last = None;
    for (out, &(x, y)) in outs.iter().zip(&squares) {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            Origin::new(x, y, 0),
            Size::new(SQUARE, SQUARE, 1),
            out,
            0,
            4 * SQUARE,
            0,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        last = Some(command_buffer);
    }
    let last = last.unwrap();
    // Every batch submitted, the last one included, while the staging submission waits for the
    // hold: where the copy was queued for it, none can complete. The hold is let go after at
    // most `WAIT` either way, so nothing waits for it for good.
    let deadline = Instant::now() + WAIT;
    while last.status() == CommandBufferStatus::Committed && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    let submitted = device.work_counts();
    let held = last.status();
    let signal = vk::SemaphoreSignalInfo::default().semaphore(hold).value(1);
    // SAFETY: a host signal of the test's timeline, above its value.
    unsafe { raw.signal_semaphore(&signal) }.unwrap();
    last.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(last.status(), CommandBufferStatus::Completed, "{name}");
    device.hold_staging_submissions(None);
    for (at, (out, &(x, y))) in outs.iter().zip(&squares).enumerate() {
        let want: Vec<u8> = (0..SQUARE)
            .flat_map(|row| {
                let from = (4 * ((y + row) * SIDE + x)) as usize;
                written[from..from + (4 * SQUARE) as usize].iter().copied()
            })
            .collect();
        let contents = out.contents().unwrap();
        // SAFETY: the buffer's bytes, which no command buffer uses now.
        let got = unsafe { std::slice::from_raw_parts(contents.as_ptr(), want.len()) };
        assert!(
            got == want,
            "{name}: batch {at}'s blit reads the call's bytes"
        );
    }
    // SAFETY: the staging submission that waited for the timeline completed before the last
    // batch did, and the submissions after it wait for it no more.
    unsafe { raw.destroy_semaphore(hold, None) };
    let waits = submitted.staging_gpu_waits - before.staging_gpu_waits;
    let flushes = submitted.staging_flushes - before.staging_flushes;
    let internal = submitted.internal_submissions - before.internal_submissions;
    eprintln!(
        "STAGING-WAITS {name}: {queued} copies queued; {BATCHES} batches submitted while the \
         staging submission was held (the last {held:?}): {flushes} staging flushes, {waits} GPU \
         waits for it, {internal} internal submissions"
    );
    if queued == 0 {
        eprintln!("STAGING-WAITS {name}: `Managed` textures copy on the host here; nothing more");
        return;
    }
    assert_eq!(
        held,
        CommandBufferStatus::Scheduled,
        "{name}: the last batch completed while the staging submission was held"
    );
    assert_eq!(
        flushes, 1,
        "{name}: one staging submission carried the copy"
    );
    assert!(
        (1..=1 + internal).contains(&waits),
        "{name}: one batch on the queue waited for the staging submission on the GPU, and the \
         batches after it none (internal submissions wait for it themselves): {waits} waits"
    );
}
