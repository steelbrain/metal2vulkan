//! A shadowed texture's image carried across one command buffer's encoders, and a log's hold of
//! logged guest pages, where command buffers of two queues meet the texture (the runtime's
//! `shadow::ShadowCarry` and `dirty_pages.rs`), on the device the cfg selects under the
//! validation layer. The texture is the one `parallel_encodes` uses: 8x8 `RGBA8Unorm` at rows 48
//! bytes apart, 16 bytes past its texels, so it is a shadow on every device.
//!
//! - **Another queue's encoder in between, a CPU write while it runs.** Command buffer X of queue
//!   1 reads the texture, over a `Shared` buffer, in one encoder, spins in a second, and reads it
//!   again in a third. Between X's reads, command buffer Y of queue 2 encodes a read of it whole.
//!   Both are committed, and while X spins the CPU writes the buffer. Metal shows none of a
//!   command buffer's encoders a CPU write made while it ran (`shadow.rs`, **Copy in**): both of
//!   X's reads hold the bytes as of its commit, the second through the image the first left
//!   (counted as carried). Y reads the bytes before or after the write, whole, whichever its run
//!   meets.
//! - **A signal in between.** Command buffer Y of queue 2 waits for an event, then fills a row of
//!   the texture's bytes. Committed after it, command buffer X of queue 1 reads the texture,
//!   signals the event, and reads it again. The scheduler runs Y between X's two batches, as it
//!   runs the ready command buffer committed first, and the tracker orders Y's fill after X's
//!   first read and before its second: the second read holds the fill.
//! - **A CPU write once a signal was seen.** Command buffer X reads the texture, signals a shared
//!   event, then spins and reads it again: in the encoder after the signal, or in one after that.
//!   The CPU waits until the event reads signalled, then writes the buffer. On the M3 Max (macOS
//!   27.0.1) Metal's read after the signal took the write in both shapes, as an encoder after a
//!   batch split does here: it copies in before its read, which runs after the spin.
//! - **A waiting writer under a hold.** Over logged guest pages (`HostMemory::with_dirty_pages`),
//!   command buffer Y of queue 2 waits for an event, fills a row of the texture's bytes, and then
//!   spins, so it completes late. Command buffer X of queue 1, committed after Y, reads the
//!   texture and signals the event: its end holds the texture whole, the hold its commit gives the
//!   log. Command buffer Z of queue 1, committed after X, reads it: the tracker runs it after Y's
//!   fill, and its read holds the fill, though the fill was reported to the log at Y's commit,
//!   before X's hold.
//! - **Both on one `VkQueue`.** The signal in between and the waiting writer again, with no spin
//!   and on a device made under `QueueSharingPolicy::Adaptive` (over the lane's `exact_answers()`)
//!   whose check has answered: queue 2 makes Y while queue 1 keeps short command buffers in
//!   flight, so where the device shares it joins queue 1's `VkQueue` (`queue_share.rs`), and X and
//!   Z run there beside Y, none long. Y is held until X's signal was submitted, so it goes after
//!   X's first batch on that `VkQueue`: the read after the signal holds Y's fill, X's hold is
//!   still withheld while Y has not completed, and Z's read holds the fill. Where the device
//!   shares no `VkQueue` the cases run on two and say so.
//!
//! The spin is the owned fixture `kernel_spin_on_an_unraised_word`, dispatched again and again to
//! a cap of loads timed on the device beforehand ([`Spin`]). The GPU does not see a CPU store to
//! the word while the kernel runs: on the M3 Max a spin waiting for the CPU to raise it ran until
//! Metal ended the command buffer ("Caused GPU Hang Error"), natively and through MoltenVK alike.
//! Metal ended one native dispatch of about 1.5 s the same way, and some of about 0.5 s, so the
//! spin is many short dispatches.

use super::host_memory::{Mapping, GUEST_PAGE};
use super::parallel_encodes::{
    completed, gated, texel_dispatch, texel_kernels, texels_of, GuestLog, Rng, TexelKernels, BYTES,
    ROW, SIDE, TEXEL_ROW, WORDS,
};
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache,
};
use crate::air_library::AirLibrary as _;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::QueueSharingPolicy;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputeCommandEncoder,
    ComputePipelineState, Device, DeviceOptions, EventRef, HostMemory, PipelineOptions,
    ResourceOptions, Size, Texture, TextureDescriptor,
};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// About how long a spin runs on the GPU: far past the CPU's write, made [`SETTLE`] after the
/// spinning command buffer's work was submitted.
const SPIN: Duration = Duration::from_millis(1500);

/// About how long each of a spin's dispatches runs on the GPU: far inside the half a second of one
/// dispatch that Metal took for hung.
const SLICE: Duration = Duration::from_millis(10);

/// The least time a dispatch of the spin's kernel is timed over.
const MEASURED: Duration = Duration::from_millis(50);

/// How long the CPU waits, once the spinning command buffer's work was submitted, before it
/// writes: its first encoder has long read the texture and its spin has started by then.
const SETTLE: Duration = Duration::from_millis(250);

/// The owned fixture `kernel_spin_on_an_unraised_word` (one thread loading a word nothing raises
/// until its cap of loads runs out) at the cap that takes about [`SLICE`] of the device's GPU time,
/// dispatched [`SPIN`] / [`SLICE`] times by a spin.
struct Spin {
    state: ComputePipelineState,
    cap: u32,
    dispatches: u32,
    word: Buffer,
    loads: Buffer,
}

impl Spin {
    /// Time the cap: single dispatches of growing caps in command buffers of their own, from
    /// commit to completion, until one takes [`MEASURED`], scaled to [`SLICE`]. The first, not
    /// timed, makes the pipeline's `VkPipeline`, so no later dispatch of it waits at a host step.
    fn new(device: &Device) -> Spin {
        let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/public/engine/kernel_spin_on_an_unraised_word.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = library
            .new_function_with_name("spin_on_an_unraised_word")
            .unwrap();
        let state = device
            .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
            .unwrap();
        let mut spin = Spin {
            state,
            cap: 1 << 12,
            dispatches: 1,
            word: buffer_with(device, &[0; 4]).unwrap(),
            loads: buffer_with(device, &[0; 4]).unwrap(),
        };
        let queue = device.new_command_queue().unwrap();
        let run = |spin: &Spin| {
            let command_buffer = queue.command_buffer().unwrap();
            spin.encode(&command_buffer);
            let at = Instant::now();
            command_buffer.commit().unwrap();
            completed(&command_buffer);
            at.elapsed()
        };
        run(&spin);
        loop {
            let took = run(&spin);
            if took >= MEASURED || spin.cap >= 1 << 30 {
                let scaled = f64::from(spin.cap) * SLICE.as_secs_f64() / took.as_secs_f64();
                spin.cap = scaled.clamp(1.0, f64::from(u32::MAX)) as u32;
                spin.dispatches = (SPIN.as_millis() / SLICE.as_millis()) as u32;
                return spin;
            }
            spin.cap <<= 2;
        }
    }

    /// Encode, in an encoder of its own, the dispatches of one thread spinning to the cap.
    fn encode(&self, command_buffer: &CommandBuffer) {
        let mut compute = command_buffer.compute_command_encoder().unwrap();
        self.dispatch(&mut compute);
        compute.end_encoding().unwrap();
    }

    /// Encode the dispatches of one thread spinning to the cap in `compute`.
    fn dispatch(&self, compute: &mut ComputeCommandEncoder) {
        compute.set_compute_pipeline_state(&self.state).unwrap();
        compute.set_buffer(Some(&self.word), 0, 0).unwrap();
        compute.set_buffer(Some(&self.loads), 0, 1).unwrap();
        compute.set_bytes(&self.cap.to_le_bytes(), 2).unwrap();
        for _ in 0..self.dispatches {
            compute
                .dispatch_threadgroups(Size::new(1, 1, 1), Size::new(1, 1, 1))
                .unwrap();
        }
    }

    /// The loads the last dispatch made.
    fn loads(&self) -> u32 {
        let bytes = read_buffer(&self.loads, 0, 4).unwrap();
        u32::from_le_bytes(bytes.try_into().unwrap())
    }
}

/// Wait until `command_buffer`'s work was submitted, then [`SETTLE`] more.
fn settled(command_buffer: &CommandBuffer) {
    let deadline = Instant::now() + SPIN;
    while command_buffer.kernel_start_time() == 0.0 {
        assert!(
            Instant::now() < deadline,
            "the command buffer was never submitted"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    std::thread::sleep(SETTLE);
}

/// The test's texture over `truth` from its first byte, rows [`ROW`] bytes apart.
fn shadowed(device: &Device, truth: &Buffer) -> Texture {
    let shadows = device.work_counts().shadowed_textures;
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    let texture = truth.new_texture(&descriptor, 0, ROW).unwrap();
    assert!(
        device.work_counts().shadowed_textures > shadows,
        "a texture whose rows have bytes between them is a shadow"
    );
    texture
}

/// A buffer for a read of the texture's texels.
fn read_target(device: &Device) -> Buffer {
    buffer_with(device, &vec![0xee; (4 * WORDS) as usize]).unwrap()
}

/// `bytes` with row `row`'s texels filled with `value`.
fn filled(bytes: &[u8], row: u64, value: u8) -> Vec<u8> {
    let mut bytes = bytes.to_vec();
    let at = (row * ROW) as usize;
    bytes[at..at + TEXEL_ROW as usize].fill(value);
    bytes
}

/// Fill row `row`'s texels of `truth` with `value`, in a blit of `command_buffer`.
fn fill_row(command_buffer: &CommandBuffer, truth: &Buffer, row: u64, value: u8) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(truth, row * ROW..row * ROW + TEXEL_ROW, value)
        .unwrap();
    blit.end_encoding().unwrap();
}

/// A device under the validation layer, its texel kernels, each run once so that no later dispatch
/// of them waits at a host step for its `VkPipeline`, and the spin.
fn lane(messages: &Messages, cache: &ScratchCache) -> (Device, TexelKernels, Spin) {
    let (device, kernels) = lane_device(messages, cache, |_| {});
    let spin = Spin::new(&device);
    (device, kernels, spin)
}

/// [`lane`]'s device, with `adjust`'s options, and its texel kernels, each run once.
fn lane_device(
    messages: &Messages,
    cache: &ScratchCache,
    adjust: impl FnOnce(&mut DeviceOptions),
) -> (Device, TexelKernels) {
    let device = validated_device_with(TEST_DEVICE, messages, cache.path(), adjust).unwrap();
    let kernels = texel_kernels(&device);
    let scratch = buffer_with(&device, &vec![0; BYTES as usize]).unwrap();
    let texture = shadowed(&device, &scratch);
    let words = read_target(&device);
    let warm = device
        .new_command_queue()
        .unwrap()
        .command_buffer()
        .unwrap();
    texel_dispatch(&warm, &kernels.write, &texture, &words);
    texel_dispatch(&warm, &kernels.read, &texture, &words);
    warm.commit().unwrap();
    completed(&warm);
    (device, kernels)
}

/// [`lane_device`] under `QueueSharingPolicy::Adaptive`, its check answered: whether it shares
/// `VkQueue`s, said where it does not.
fn sharing_lane(messages: &Messages, cache: &ScratchCache) -> (Device, TexelKernels, bool) {
    let (device, kernels) = lane_device(messages, cache, |options| {
        options.policy.queue_sharing = QueueSharingPolicy::Adaptive;
    });
    assert!(
        device.settle_queue_switch_check(Duration::from_secs(20)),
        "the check settles"
    );
    let shares = device.shares_vk_queues();
    if !shares {
        println!(
            "{} shares no VkQueue, so the one-VkQueue cases run on two: {}",
            device.name(),
            device.queue_switch_check_report()
        );
    }
    (device, kernels, shares)
}

/// Keep `busy`'s `VkQueue` busy with short fills of `scratch`, then make `joiner`'s command buffer,
/// which joins that `VkQueue` where the device shares (the module docs): the command buffer, and
/// the fills, to be waited for once the case ran. Where it shares, both queues are on one
/// `VkQueue` and neither is long.
fn joined(
    device: &Device,
    busy: &CommandQueue,
    joiner: &CommandQueue,
    shares: bool,
) -> (CommandBuffer, Vec<CommandBuffer>) {
    let scratch = device
        .new_buffer(16 << 20, ResourceOptions::private())
        .unwrap();
    let fills: Vec<CommandBuffer> = (0..24)
        .map(|_| {
            let command_buffer = busy.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&scratch, 0..(16 << 20), 0x77).unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer
        })
        .collect();
    fills.last().unwrap().wait_until_scheduled().unwrap();
    let joins = device.work_counts().queue_joins;
    let made = joiner.command_buffer().unwrap();
    let (on, with) = (joiner.vk_queue_sharing(), busy.vk_queue_sharing());
    if shares {
        assert!(device.work_counts().queue_joins > joins, "the queue joined");
        assert_eq!(on, with, "both queues on one VkQueue, neither long");
        assert!(!on.1, "neither queue is long");
    }
    (made, fills)
}

/// [`an_encoder_after_a_signal_reads_what_another_queue_wrote_after_it`] with the two queues on
/// one `VkQueue` where the device shares (the module docs).
#[test]
fn an_encoder_after_a_signal_reads_another_queue_s_write_on_one_vk_queue() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let (device, kernels, shares) = sharing_lane(&messages, &cache);
    let (first, second) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let initial = Rng(0x5162).bytes(BYTES);
    let (row, value) = (6, 0x4d);
    let truth = buffer_with(&device, &initial).unwrap();
    let texture = shadowed(&device, &truth);
    let event = device.new_event().unwrap();
    let reads = [(); 2].map(|_| read_target(&device));
    let (y, fills) = joined(&device, &first, &second, shares);
    y.encode_wait_for_event(EventRef::Event(&event), 1).unwrap();
    fill_row(&y, &truth, row, value);
    y.commit().unwrap();
    let x = first.command_buffer().unwrap();
    texel_dispatch(&x, &kernels.read, &texture, &reads[0]);
    x.encode_signal_event(EventRef::Event(&event), 1).unwrap();
    texel_dispatch(&x, &kernels.read, &texture, &reads[1]);
    x.commit().unwrap();
    for command_buffer in fills.iter().chain([&x, &y]) {
        completed(command_buffer);
    }
    println!(
        "ONE-VKQUEUE-SIGNAL on {}: sharing {shares}, X's queue {:?}, Y's {:?}",
        device.name(),
        first.vk_queue_sharing(),
        second.vk_queue_sharing()
    );
    let read = |at: usize| read_buffer(&reads[at], 0, 4 * WORDS).unwrap();
    assert_eq!(
        read(0),
        texels_of(&initial),
        "the encoder before the signal"
    );
    assert_eq!(
        read(1),
        texels_of(&filled(&initial, row, value)),
        "the encoder after the signal, which Y's fill runs before"
    );
    drop((texture, truth, first, second, kernels, event));
    drop(device);
    layer_reported_nothing_at_all(&messages);
}

/// [`a_write_of_logged_pages_after_an_event_wait_reaches_a_copy_in_behind_a_later_hold`] with no
/// spin and the two queues on one `VkQueue` where the device shares (the module docs).
#[test]
fn a_waiting_writer_s_hold_is_withheld_on_one_vk_queue() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let (device, kernels, shares) = sharing_lane(&messages, &cache);
    let (first, second) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let mapping = Mapping::anonymous(GUEST_PAGE);
    let initial = Rng(0x9A6F).bytes(BYTES);
    mapping.write(0, &initial);
    let log = Arc::new(GuestLog::default());
    log.told(0..BYTES);
    // SAFETY: the mapping outlives the buffer, dropped first, and stays writable.
    let memory = unsafe { HostMemory::new(mapping.pointer()) }.with_dirty_pages(Arc::clone(&log));
    // SAFETY: as above.
    let pages = unsafe {
        device.new_buffer_with_bytes_no_copy(
            memory,
            GUEST_PAGE as u64,
            ResourceOptions::default(),
            None,
        )
    }
    .unwrap();
    let texture = shadowed(&device, &pages);
    let event = device.new_event().unwrap();
    let (row, value) = (2, 0x5b);
    let reads = [(); 2].map(|_| read_target(&device));
    let before = gated(&device.work_counts());
    let withheld = device.work_counts().dirty_holds_withheld;
    let (y, fills) = joined(&device, &first, &second, shares);
    y.encode_wait_for_event(EventRef::Event(&event), 1).unwrap();
    fill_row(&y, &pages, row, value);
    y.commit().unwrap();
    let x = first.command_buffer().unwrap();
    texel_dispatch(&x, &kernels.read, &texture, &reads[0]);
    x.encode_signal_event(EventRef::Event(&event), 1).unwrap();
    x.commit().unwrap();
    let z = first.command_buffer().unwrap();
    texel_dispatch(&z, &kernels.read, &texture, &reads[1]);
    z.commit().unwrap();
    for command_buffer in fills.iter().chain([&x, &y, &z]) {
        completed(command_buffer);
    }
    let after = gated(&device.work_counts());
    let counts: [u64; 5] = std::array::from_fn(|i| after[i] - before[i]);
    let withheld = device.work_counts().dirty_holds_withheld - withheld;
    println!(
        "ONE-VKQUEUE-HOLD on {}: sharing {shares}, X's queue {:?}, Y's {:?}, gated copies in \
         (gated, whole, skipped, bytes whole, bytes copied) {counts:?}, holds withheld {withheld}",
        device.name(),
        first.vk_queue_sharing(),
        second.vk_queue_sharing()
    );
    assert!(counts[0] > 0, "the copies in were gated");
    assert!(
        withheld > 0,
        "X's hold was withheld while Y, which waits, had not completed"
    );
    let read = |at: usize| read_buffer(&reads[at], 0, 4 * WORDS).unwrap();
    assert_eq!(read(0), texels_of(&initial), "X's read, before the signal");
    assert_eq!(
        read(1),
        texels_of(&filled(&initial, row, value)),
        "Z's read, after Y's fill"
    );
    drop((texture, pages, first, second, kernels, event));
    drop(device);
    drop(mapping);
    layer_reported_nothing_at_all(&messages);
}

/// Another queue's encoder of the texture ends between a command buffer's two, and the CPU writes
/// the bytes while the command buffer runs: neither of its encoders sees the write (the module
/// docs).
#[test]
fn a_cpu_write_while_a_command_buffer_runs_reaches_none_of_its_encoders() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let (device, kernels, spin) = lane(&messages, &cache);
    let (first, second) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let mut rng = Rng(0xC0FE);
    let (initial, written) = (rng.bytes(BYTES), rng.bytes(BYTES));
    let truth = buffer_with(&device, &initial).unwrap();
    let texture = shadowed(&device, &truth);
    let reads = [(); 3].map(|_| read_target(&device));
    let carried = device.work_counts().shadow_texels_first_copy_in_carried;
    let x = first.command_buffer().unwrap();
    texel_dispatch(&x, &kernels.read, &texture, &reads[0]);
    spin.encode(&x);
    let y = second.command_buffer().unwrap();
    texel_dispatch(&y, &kernels.read, &texture, &reads[2]);
    texel_dispatch(&x, &kernels.read, &texture, &reads[1]);
    x.commit().unwrap();
    y.commit().unwrap();
    settled(&x);
    let at = truth.contents().expect("a Shared buffer");
    // SAFETY: the buffer holds `BYTES` bytes, `Shared`; the GPU reads them only through the
    // texture's copies in, which run before or after this write.
    unsafe { std::ptr::copy_nonoverlapping(written.as_ptr(), at.as_ptr(), written.len()) };
    let running = x.status();
    completed(&x);
    completed(&y);
    let carried = device.work_counts().shadow_texels_first_copy_in_carried - carried;
    let read = |at: usize| read_buffer(&reads[at], 0, 4 * WORDS).unwrap();
    let y_read = match read(2) {
        bytes if bytes == texels_of(&initial) => "before the write",
        bytes if bytes == texels_of(&written) => "after the write",
        _ => "neither whole",
    };
    println!(
        "CROSS-QUEUE-CPU-WRITE on {}: X {running:?} at the write, the spin {} dispatches of {} \
         loads (the last made {}), texels carried {carried}, Y read the bytes {y_read}",
        device.name(),
        spin.dispatches,
        spin.cap,
        spin.loads()
    );
    assert!(
        matches!(
            running,
            CommandBufferStatus::Committed | CommandBufferStatus::Scheduled
        ),
        "the write landed while X ran ({running:?})"
    );
    assert_eq!(read(0), texels_of(&initial), "X's first encoder");
    assert_eq!(
        read(1),
        texels_of(&initial),
        "X's second encoder: a CPU write made while X ran reaches none of its encoders"
    );
    assert_eq!(
        carried, WORDS,
        "X's second read took the image its first left"
    );
    assert_ne!(y_read, "neither whole", "Y's read");
    drop((texture, truth, first, second, kernels, spin));
    drop(device);
    layer_reported_nothing_at_all(&messages);
}

/// Another queue's command buffer that waits for a command buffer's signal and fills the
/// texture's bytes runs between that command buffer's batches: the encoder after the signal reads
/// the fill (the module docs).
#[test]
fn an_encoder_after_a_signal_reads_what_another_queue_wrote_after_it() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let (device, kernels, spin) = lane(&messages, &cache);
    let (first, second) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let initial = Rng(0x5161).bytes(BYTES);
    let (row, value) = (5, 0x3c);
    let truth = buffer_with(&device, &initial).unwrap();
    let texture = shadowed(&device, &truth);
    let event = device.new_event().unwrap();
    let reads = [(); 2].map(|_| read_target(&device));
    let y = second.command_buffer().unwrap();
    y.encode_wait_for_event(EventRef::Event(&event), 1).unwrap();
    fill_row(&y, &truth, row, value);
    y.commit().unwrap();
    let x = first.command_buffer().unwrap();
    texel_dispatch(&x, &kernels.read, &texture, &reads[0]);
    x.encode_signal_event(EventRef::Event(&event), 1).unwrap();
    texel_dispatch(&x, &kernels.read, &texture, &reads[1]);
    x.commit().unwrap();
    completed(&x);
    completed(&y);
    let read = |at: usize| read_buffer(&reads[at], 0, 4 * WORDS).unwrap();
    assert_eq!(
        read(0),
        texels_of(&initial),
        "the encoder before the signal"
    );
    assert_eq!(
        read(1),
        texels_of(&filled(&initial, row, value)),
        "the encoder after the signal, which Y's fill runs before"
    );
    drop((texture, truth, first, second, kernels, spin, event));
    drop(device);
    layer_reported_nothing_at_all(&messages);
}

/// A CPU write made once a command buffer's signal was seen reaches its read after the signal,
/// whether the read is in the encoder after the signal or a later one (the module docs).
#[test]
fn a_cpu_write_once_a_signal_was_seen_reaches_the_reads_after_it() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let (device, kernels, spin) = lane(&messages, &cache);
    let queue = device.new_command_queue().unwrap();
    let name = device.name().to_owned();
    for (shape, encoder_between) in [("a later encoder", true), ("the same encoder", false)] {
        let mut rng = Rng(0x5167 + u64::from(encoder_between));
        let (initial, written) = (rng.bytes(BYTES), rng.bytes(BYTES));
        let truth = buffer_with(&device, &initial).unwrap();
        let texture = shadowed(&device, &truth);
        let event = device.new_shared_event().unwrap();
        let reads = [(); 2].map(|_| read_target(&device));
        let x = queue.command_buffer().unwrap();
        texel_dispatch(&x, &kernels.read, &texture, &reads[0]);
        x.encode_signal_event(EventRef::from(&event), 1).unwrap();
        match encoder_between {
            true => {
                spin.encode(&x);
                texel_dispatch(&x, &kernels.read, &texture, &reads[1]);
            }
            false => {
                let mut compute = x.compute_command_encoder().unwrap();
                spin.dispatch(&mut compute);
                compute.set_compute_pipeline_state(&kernels.read).unwrap();
                compute.set_texture(Some(&texture), 0).unwrap();
                compute.set_buffer(Some(&reads[1]), 0, 0).unwrap();
                compute
                    .dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(SIDE, 1, 1))
                    .unwrap();
                compute.end_encoding().unwrap();
            }
        }
        x.commit().unwrap();
        let deadline = Instant::now() + SPIN;
        while event.signaled_value() < 1 {
            assert!(
                Instant::now() < deadline,
                "{shape}: the signal was never seen"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let at = truth.contents().expect("a Shared buffer");
        // SAFETY: the buffer holds `BYTES` bytes, `Shared`; the GPU reads them only through the
        // texture's copies in, which run before or after this write.
        unsafe { std::ptr::copy_nonoverlapping(written.as_ptr(), at.as_ptr(), written.len()) };
        let running = x.status();
        completed(&x);
        println!("CROSS-QUEUE-SIGNAL-SEEN on {name}, {shape}: X {running:?} at the write");
        assert!(
            matches!(
                running,
                CommandBufferStatus::Committed | CommandBufferStatus::Scheduled
            ),
            "{shape}: the write landed while X ran ({running:?})"
        );
        let read = |at: usize| read_buffer(&reads[at], 0, 4 * WORDS).unwrap();
        assert_eq!(
            read(0),
            texels_of(&initial),
            "{shape}: the read before the signal"
        );
        assert_eq!(
            read(1),
            texels_of(&written),
            "{shape}: the read after the signal, which Metal shows the write"
        );
    }
    drop((queue, kernels, spin));
    drop(device);
    layer_reported_nothing_at_all(&messages);
}

/// A command buffer that waits for an event writes logged pages a shadowed texture lies in, and
/// runs after a command buffer committed after it took the log's hold: a third command buffer's
/// gated copy in still takes the write (the module docs).
#[test]
fn a_write_of_logged_pages_after_an_event_wait_reaches_a_copy_in_behind_a_later_hold() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let (device, kernels, spin) = lane(&messages, &cache);
    let (first, second) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let mapping = Mapping::anonymous(GUEST_PAGE);
    let initial = Rng(0x9A6E).bytes(BYTES);
    mapping.write(0, &initial);
    let log = Arc::new(GuestLog::default());
    log.told(0..BYTES);
    // SAFETY: the mapping outlives the buffer, dropped first, and stays writable.
    let memory = unsafe { HostMemory::new(mapping.pointer()) }.with_dirty_pages(Arc::clone(&log));
    // SAFETY: as above.
    let pages = unsafe {
        device.new_buffer_with_bytes_no_copy(
            memory,
            GUEST_PAGE as u64,
            ResourceOptions::default(),
            None,
        )
    }
    .unwrap();
    let texture = shadowed(&device, &pages);
    let event = device.new_event().unwrap();
    let (row, value) = (3, 0xa5);
    let reads = [(); 2].map(|_| read_target(&device));
    let before = gated(&device.work_counts());
    let withheld = device.work_counts().dirty_holds_withheld;
    let y = second.command_buffer().unwrap();
    y.encode_wait_for_event(EventRef::Event(&event), 1).unwrap();
    fill_row(&y, &pages, row, value);
    spin.encode(&y);
    y.commit().unwrap();
    let x = first.command_buffer().unwrap();
    texel_dispatch(&x, &kernels.read, &texture, &reads[0]);
    x.encode_signal_event(EventRef::Event(&event), 1).unwrap();
    x.commit().unwrap();
    let z = first.command_buffer().unwrap();
    texel_dispatch(&z, &kernels.read, &texture, &reads[1]);
    z.commit().unwrap();
    for command_buffer in [&x, &y, &z] {
        completed(command_buffer);
    }
    let after = gated(&device.work_counts());
    let name = device.name().to_owned();
    let counts: [u64; 5] = std::array::from_fn(|i| after[i] - before[i]);
    let withheld = device.work_counts().dirty_holds_withheld - withheld;
    println!(
        "CROSS-QUEUE-HOLD on {name}: gated copies in (gated, whole, skipped, bytes whole, bytes \
         copied) {counts:?}, holds withheld {withheld}, the spin {} dispatches of {} loads",
        spin.dispatches, spin.cap
    );
    assert!(counts[0] > 0, "the copies in were gated");
    assert!(
        withheld > 0,
        "X's hold was withheld while Y, which waits, had not completed"
    );
    let read = |at: usize| read_buffer(&reads[at], 0, 4 * WORDS).unwrap();
    assert_eq!(read(0), texels_of(&initial), "X's read, before the signal");
    assert_eq!(
        read(1),
        texels_of(&filled(&initial, row, value)),
        "Z's read, after Y's fill"
    );
    drop((texture, pages, first, second, kernels, spin, event));
    drop(device);
    drop(mapping);
    layer_reported_nothing_at_all(&messages);
}
