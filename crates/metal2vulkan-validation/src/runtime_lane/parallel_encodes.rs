//! Command buffers of one queue encoded on several threads out of commit order, then committed in
//! order, on the device the cfg selects under the validation layer, where the runtime keeps
//! state per resource as commands are encoded rather than as they are submitted
//! (`parity::sequences::parallel` holds the sequences' bytes, hazards and notices):
//!
//! - **A new texture's pattern** (`runtime::fresh_unwritten`): a new 4-sample `Depth32Float`
//!   reads zero where Metal reads its pattern (every device notes one: no clear writes a NaN),
//!   announced as `fresh-pattern-zeroed` where a command first reads it unwritten; a
//!   single-sample one reads zero, or holds Metal's signed NaN pattern, announced as
//!   `signed-depth-test` at a load before a write where the device's depth test is not Metal's.
//!   Metal's answer follows the commit order: a pass that loads it after a committed-earlier pass
//!   cleared it reads the clear, and one that loads it before a committed-later clear reads the
//!   pattern. Each case runs its two command buffers serially, then with the later-committed one
//!   encoded whole first on another thread, and requires the same notices of both (the kinds
//!   announced at every occurrence, one fresh texture a run).
//! - **A shadowed texture's carry** (`runtime::shadow`'s `ShadowCarry`): a texture over a
//!   `Private` buffer at rows 48 bytes apart (32 bytes of texels: no linear image aliases it on a
//!   device whose row pitch is another) written by a kernel, its buffer copied out by a blit, then
//!   read by a kernel, in each of eight command buffers on four threads. Every copy and read
//!   holds its own command buffer's words. A command buffer's later encoder takes the image its
//!   earlier one left holding the truth, unless another encoder that used it ended in between,
//!   which encoding on several threads makes likely: only another queue's encoder counts, since
//!   a queue's command buffers run whole in commit order (`shadow::Touched`), so the copies and
//!   the carries on four threads are the serial run's (printed as `PARALLEL-SHADOW` lines).
//! - **Guest pages** (`runtime::dirty_pages`): the same texture's shape over a no-copy buffer of
//!   pages whose memory carries a log of the writes to them ([`GuestLog`],
//!   `HostMemory::with_dirty_pages`), the CPU writing them at each run's start and telling the
//!   log. Eight command buffers of one queue write the texture through a kernel and copy the
//!   pages out, read it, fill a row of its bytes by a blit then read it, and read it in two
//!   encoders, the second carried, so its shadow's gated copies in count on the holds the commits
//!   before them left, which a write of the pages lets go of (`DirtyShadows::committed`). Encoded serially, and on 2, 3 and 4 threads
//!   out of commit order, committed in order: every read and copy holds the bytes of the CPU
//!   model, the pages end holding them, every byte the GPU wrote is one the runtime reported to
//!   the log, the reports and the gated copies in are the serial run's, and the bytes the gated
//!   copies moved are printed (`PARALLEL-GUEST` lines).

use super::host_memory::Mapping;
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary as _;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Degradation,
    Device, DirtyPages, DirtyStamp, HostMemory, LoadAction, PipelineOptions, RenderPassDescriptor,
    ResourceOptions, Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Wait for `command_buffer`, which must complete without an error.
fn completed(command_buffer: &CommandBuffer) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (CommandBufferStatus::Completed, None)
    );
}

/// Encode command buffers `0..encodes.len()` of `queue` (made on this thread in commit order),
/// each by its closure: serially, or with each encoded only once the one committed after it was
/// encoded whole (`reversed`), on a thread of its own; then commit them in order and wait.
fn committed_in_order(
    queue: &CommandQueue,
    encodes: &[&(dyn Fn(&CommandBuffer) + Sync)],
    reversed: bool,
) {
    let command_buffers: Vec<CommandBuffer> = (0..encodes.len())
        .map(|_| queue.command_buffer().unwrap())
        .collect();
    if reversed {
        // The baton: the next command buffer that may encode, from the last down.
        let baton = Mutex::new(encodes.len() - 1);
        let passed = Condvar::new();
        std::thread::scope(|scope| {
            for (index, encode) in encodes.iter().enumerate() {
                let (baton, passed, command_buffer) = (&baton, &passed, &command_buffers[index]);
                scope.spawn(move || {
                    let mut held = baton.lock().unwrap();
                    while *held != index {
                        held = passed.wait_timeout(held, WAIT).unwrap().0;
                    }
                    encode(command_buffer);
                    *held = index.saturating_sub(1);
                    drop(held);
                    passed.notify_all();
                });
            }
        });
    } else {
        for (encode, command_buffer) in encodes.iter().zip(&command_buffers) {
            encode(command_buffer);
        }
    }
    for command_buffer in &command_buffers {
        command_buffer.commit().unwrap();
    }
    for command_buffer in &command_buffers {
        completed(command_buffer);
    }
}

/// A new `Private` `Depth32Float` render target, 16 square, of `samples`.
fn fresh_depth(device: &Device, samples: u64) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, 16, 16, false);
    if samples > 1 {
        descriptor.texture_type = TextureType::Type2DMultisample;
        descriptor.sample_count = samples;
    }
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// A pass of `depth` alone that loads it by `load` (a clear to 0.25) and stores it, encoded
/// with no draw into `command_buffer`.
fn depth_pass(command_buffer: &CommandBuffer, depth: &Texture, load: LoadAction) {
    let mut pass = RenderPassDescriptor::new();
    pass.depth_attachment.texture = Some(depth.clone());
    pass.depth_attachment.load_action = load;
    pass.depth_attachment.clear_depth = 0.25;
    pass.depth_attachment.store_action = StoreAction::Store;
    let encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.end_encoding().unwrap();
}

/// The notices of a new texture's pattern read (`FRESH_KINDS`) `messages` holds, taken with the
/// rest.
fn fresh_notices(messages: &Messages) -> usize {
    (messages.take_notices().iter())
        .filter_map(|message| message.degradation.as_ref())
        .filter(|notice| FRESH_KINDS.contains(&notice.degradation))
        .count()
}

/// What a pass's load of a new texture's unwritten depth pattern announces: `fresh-pattern-zeroed`
/// where the texture reads zero, `signed-depth-test` where it holds Metal's signed NaN and the
/// device's depth test is not Metal's (`runtime::fresh_unwritten`).
const FRESH_KINDS: [Degradation; 2] = [
    Degradation::FreshPatternZeroed,
    Degradation::SignedDepthTest,
];

/// The notice of a new texture's pattern follows the commit order (the module docs): a load
/// after a committed-earlier clear announces nothing, and a load before a committed-later clear
/// announces the pattern it reads, whichever pass was encoded first. A 4-sample `Depth32Float`
/// reads zero on every device (`fresh-pattern-zeroed`, Metal reading the pattern once where its
/// load comes first); a single-sample one reads zero, or holds Metal's signed NaN, announced as
/// `signed-depth-test` where the device's depth test is not Metal's: its serial run, in commit
/// order, is its answer.
#[test]
fn a_fresh_texture_s_pattern_notice_follows_the_commit_order() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.announce_each.extend(FRESH_KINDS);
    })
    .unwrap();
    let name = device.name().to_owned();
    let queue = device.new_command_queue().unwrap();
    messages.take_notices();
    // (samples, the first pass's load, the second's, Metal's reads of the pattern where every
    // device announces them)
    let cases = [
        (4, LoadAction::Clear, LoadAction::Load, Some(0)),
        (4, LoadAction::Load, LoadAction::Clear, Some(1)),
        (1, LoadAction::Clear, LoadAction::Load, Some(0)),
        (1, LoadAction::Load, LoadAction::Clear, None),
    ];
    let mut wrong = Vec::new();
    for (samples, first, second, metal) in cases {
        let mut announced = [0; 2];
        for (reversed, count) in announced.iter_mut().enumerate() {
            let depth = fresh_depth(&device, samples);
            let one = |cb: &CommandBuffer| depth_pass(cb, &depth, first);
            let two = |cb: &CommandBuffer| depth_pass(cb, &depth, second);
            committed_in_order(&queue, &[&one, &two], reversed == 1);
            *count = fresh_notices(&messages);
        }
        println!(
            "PARALLEL-FRESH on {name}: {samples}x, {first:?} then {second:?}: {} notice(s)              serially, {} with the later encoded first; Metal reads the pattern {metal:?} time(s)",
            announced[0], announced[1]
        );
        let expected = metal.unwrap_or(announced[0]);
        if announced != [expected; 2] {
            wrong.push((samples, first, second, announced, expected));
        }
    }
    assert!(
        wrong.is_empty(),
        "on {name}, notices of a new texture's pattern read that differ from the reads in commit          order ((samples, loads, notices serially and with the later encoded first, the          answer)): {wrong:?}"
    );
    drop(queue);
    drop(device);
    layer_reported_nothing_at_all(&messages);
}

/// The texels of the shadowed texture: `SIDE`² `RGBA8Unorm`, rows [`ROW`] bytes apart.
const SIDE: u64 = 8;
const ROW: u64 = 48;
const TEXEL_ROW: u64 = 4 * SIDE;
const BYTES: u64 = ROW * SIDE;
const WORDS: u64 = SIDE * SIDE;

/// The command buffers of the shadow test, its threads, and its runs each way.
const SHADOW_COMMAND_BUFFERS: usize = 8;
const SHADOW_THREADS: usize = 4;
const SHADOW_RUNS: u64 = 16;

/// SplitMix64, for the test's bytes and pauses.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn bytes(&mut self, len: u64) -> Vec<u8> {
        (0..len).map(|_| self.next() as u8).collect()
    }
}

/// The fixture's texel kernels (`seq_programs`): words into texels, and texels into words.
struct TexelKernels {
    write: ComputePipelineState,
    read: ComputePipelineState,
}

fn texel_kernels(device: &Device) -> TexelKernels {
    let bitcode = llvm_assemble(&engine_dir().join("seq_programs.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let kernel = |name| {
        let function = library.new_function_with_name(name).unwrap();
        device
            .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
            .unwrap()
    };
    TexelKernels {
        write: kernel("seq_texel_write"),
        read: kernel("seq_texel_read"),
    }
}

/// One dispatch of `kernel` over the texture's texels, with `words`.
fn texel_dispatch(
    cb: &CommandBuffer,
    kernel: &ComputePipelineState,
    texture: &Texture,
    words: &Buffer,
) {
    let mut compute = cb.compute_command_encoder().unwrap();
    compute.set_compute_pipeline_state(kernel).unwrap();
    compute.set_texture(Some(texture), 0).unwrap();
    compute.set_buffer(Some(words), 0, 0).unwrap();
    compute
        .dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(SIDE, 1, 1))
        .unwrap();
    compute.end_encoding().unwrap();
}

/// One run of the shadow test's command buffers: each one's words, and the buffers its copy and
/// its read fill.
struct ShadowRun {
    words: Vec<Buffer>,
    copies: Vec<Buffer>,
    reads: Vec<Buffer>,
}

/// The shadow test's work counts: shadow copies, texels copied in, texels a carry spared.
fn shadow_counts(counts: &WorkCounts) -> [u64; 3] {
    [
        counts.shadow_copies,
        counts.shadow_texels_copied_in,
        counts.shadow_texels_first_copy_in_carried,
    ]
}

/// What encodes a command buffer, given its index and a pause to take between its encoders.
type Encode<'a> = dyn Fn(usize, &CommandBuffer, &mut dyn FnMut()) + Sync + 'a;

/// Encode `command_buffers` (made in commit order) by `encode`, given each one's index: on this
/// thread in turn, or on `threads` threads from a barrier where `schedule` is some, each taking
/// every `threads`-th command buffer, the latest first, with pauses seeded by it (a yield, a spin
/// or a sleep of up to 60 µs, or none) before each and between its encoders.
fn encoded(command_buffers: &[CommandBuffer], schedule: Option<(usize, u64)>, encode: &Encode<'_>) {
    let Some((threads, seed)) = schedule else {
        for (index, cb) in command_buffers.iter().enumerate() {
            encode(index, cb, &mut || {});
        }
        return;
    };
    let start = Barrier::new(threads);
    let mut rng = Rng(seed);
    let seeds: Vec<u64> = (0..threads).map(|_| rng.next()).collect();
    std::thread::scope(|scope| {
        for (me, &seed) in seeds.iter().enumerate() {
            let start = &start;
            scope.spawn(move || {
                let mut rng = Rng(seed);
                let mut pause = || match rng.next() % 4 {
                    0 => {}
                    1 => std::thread::yield_now(),
                    2 => {
                        let until = Instant::now() + Duration::from_micros(rng.next() % 30);
                        while Instant::now() < until {
                            std::hint::spin_loop();
                        }
                    }
                    _ => std::thread::sleep(Duration::from_micros(rng.next() % 60)),
                };
                start.wait();
                let mine = (0..command_buffers.len())
                    .rev()
                    .filter(|i| i % threads == me);
                for index in mine {
                    pause();
                    encode(index, &command_buffers[index], &mut pause);
                }
            });
        }
    });
}

/// A run of the shadow test on `device` (the module docs), its command buffers encoded on this
/// thread in turn, or on [`SHADOW_THREADS`] threads ([`encoded`]) where `seed` is some; each
/// command buffer's copy and read compared with its words. Its counts.
fn shadow_run(
    device: &Device,
    queue: &CommandQueue,
    (truth, texture, initial): (&Buffer, &Texture, &[u8]),
    kernels: &TexelKernels,
    seed: Option<u64>,
) -> [u64; 3] {
    let mut rng = Rng(seed.unwrap_or(0x5AD0));
    let run = ShadowRun {
        words: (0..SHADOW_COMMAND_BUFFERS)
            .map(|_| buffer_with(device, &rng.bytes(4 * WORDS)).unwrap())
            .collect(),
        copies: (0..SHADOW_COMMAND_BUFFERS)
            .map(|_| buffer_with(device, &vec![0xee; BYTES as usize]).unwrap())
            .collect(),
        reads: (0..SHADOW_COMMAND_BUFFERS)
            .map(|_| buffer_with(device, &vec![0xee; (4 * WORDS) as usize]).unwrap())
            .collect(),
    };
    let encode = |index: usize, cb: &CommandBuffer, pause: &mut dyn FnMut()| {
        texel_dispatch(cb, &kernels.write, texture, &run.words[index]);
        pause();
        let mut blit = cb.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(truth, 0, &run.copies[index], 0, BYTES)
            .unwrap();
        blit.end_encoding().unwrap();
        pause();
        texel_dispatch(cb, &kernels.read, texture, &run.reads[index]);
    };
    let before = shadow_counts(&device.work_counts());
    let command_buffers: Vec<CommandBuffer> = (0..SHADOW_COMMAND_BUFFERS)
        .map(|_| queue.command_buffer().unwrap())
        .collect();
    encoded(
        &command_buffers,
        seed.map(|seed| (SHADOW_THREADS, rng.next() ^ seed)),
        &encode,
    );
    for cb in &command_buffers {
        cb.commit().unwrap();
    }
    for cb in &command_buffers {
        completed(cb);
    }
    let after = shadow_counts(&device.work_counts());
    for index in 0..SHADOW_COMMAND_BUFFERS {
        let words = read_buffer(&run.words[index], 0, 4 * WORDS).unwrap();
        let mut expected = initial.to_vec();
        for row in 0..SIDE {
            let (at, from) = ((row * ROW) as usize, (row * TEXEL_ROW) as usize);
            expected[at..at + TEXEL_ROW as usize]
                .copy_from_slice(&words[from..from + TEXEL_ROW as usize]);
        }
        assert_eq!(
            read_buffer(&run.copies[index], 0, BYTES).unwrap(),
            expected,
            "cb{index}'s copy of the buffer under seed {seed:?}"
        );
        assert_eq!(
            read_buffer(&run.reads[index], 0, 4 * WORDS).unwrap(),
            words,
            "cb{index}'s read of the texture under seed {seed:?}"
        );
    }
    [0, 1, 2].map(|i| after[i] - before[i])
}

/// A shadowed texture's every use holds its own command buffer's bytes however its command
/// buffers' encoders interleave; the copies in a carry spares serially are printed beside those
/// the parallel runs made (the module docs).
#[test]
fn a_shadowed_texture_keeps_each_command_buffer_s_bytes() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
    let name = device.name().to_owned();
    let queue = device.new_command_queue().unwrap();
    let kernels = texel_kernels(&device);
    let initial = Rng(0x1A1A).bytes(BYTES);
    let staging = buffer_with(&device, &initial).unwrap();
    let truth = device
        .new_buffer(BYTES, ResourceOptions::private())
        .unwrap();
    let setup = queue.command_buffer().unwrap();
    let mut blit = setup.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&staging, 0, &truth, 0, BYTES)
        .unwrap();
    blit.end_encoding().unwrap();
    setup.commit().unwrap();
    completed(&setup);
    let shadows = device.work_counts().shadowed_textures;
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    descriptor.resource_options = ResourceOptions::private();
    let texture = truth.new_texture(&descriptor, 0, ROW).unwrap();
    let shadowed = device.work_counts().shadowed_textures > shadows;
    let shadow = (&truth, &texture, &initial[..]);
    let serial: Vec<[u64; 3]> = (0..SHADOW_RUNS)
        .map(|_| shadow_run(&device, &queue, shadow, &kernels, None))
        .collect();
    let parallel: Vec<[u64; 3]> = (0..SHADOW_RUNS)
        .map(|run| {
            let seed = 0x5EED_0000 + run;
            shadow_run(&device, &queue, shadow, &kernels, Some(seed))
        })
        .collect();
    let sum = |runs: &[[u64; 3]]| {
        runs.iter()
            .fold([0; 3], |sum, run| [0, 1, 2].map(|i| sum[i] + run[i]))
    };
    println!(
        "PARALLEL-SHADOW on {name}: shadowed {shadowed}; over {SHADOW_RUNS} runs of \
         {SHADOW_COMMAND_BUFFERS} command buffers, (shadow copies, texels copied in, texels a \
         carry spared) serially {:?}, on {SHADOW_THREADS} threads {:?}",
        sum(&serial),
        sum(&parallel)
    );
    let notices: Vec<String> = (messages.take_notices().iter())
        .filter_map(|message| message.degradation.as_ref())
        .map(|notice| notice.degradation.code().to_owned())
        .collect();
    println!("PARALLEL-SHADOW on {name}: notices {notices:?}");
    // One queue's command buffers run in commit order, so another one's encoder ending between
    // two of a command buffer's encoders leaves its carry standing (`shadow::Touched`).
    assert_eq!(
        sum(&parallel),
        sum(&serial),
        "the copies and the carries on {SHADOW_THREADS} threads"
    );
    drop((texture, truth, queue, kernels));
    drop(device);
    layer_reported_nothing_at_all(&messages);
}

/// The guest pages test's log of the writes to its pages, as a consumer keeps one: each write
/// stamped with the publication it falls in and named with its writer (0 the CPU's and every
/// write reported through `wrote`), and the runtime's reports of its own writes kept apart, the
/// marks the consumer reads.
#[derive(Default)]
struct GuestLog(Mutex<GuestLogState>);

#[derive(Default)]
struct GuestLogState {
    /// The publication writes fall in now.
    now: u64,
    /// Each write's bytes, stamp and writer.
    writes: Vec<(Range<u64>, u64, u64)>,
    /// The runtime's reports since the last [`GuestLog::marks`]: the bytes, and whether a copy
    /// named them its own (`wrote_as`).
    marks: Vec<(Range<u64>, bool)>,
}

impl GuestLog {
    fn state(&self) -> std::sync::MutexGuard<'_, GuestLogState> {
        self.0.lock().unwrap()
    }

    /// Note the CPU's write of the bytes `range`.
    fn told(&self, range: Range<u64>) {
        let mut state = self.state();
        let now = state.now;
        state.writes.push((range, now, 0));
    }

    /// The runtime's reports since the last call, each once, in order of their bytes.
    fn marks(&self) -> Vec<(Range<u64>, bool)> {
        let mut marks = std::mem::take(&mut self.state().marks);
        marks.sort_by_key(|(range, own)| (range.start, range.end, *own));
        marks.dedup();
        marks
    }

    fn changed_by(&self, since: DirtyStamp, but: u64, out: &mut Vec<Range<u64>>) {
        let state = self.state();
        let later = (state.writes.iter())
            .filter(|(_, stamp, writer)| *stamp > since.0 && (but == 0 || *writer != but));
        out.extend(later.map(|(range, _, _)| range.clone()));
    }

    fn reported(&self, range: Range<u64>, writer: u64) {
        let mut state = self.state();
        let now = state.now;
        state.writes.push((range.clone(), now, writer));
        state.marks.push((range, writer != 0));
    }
}

impl DirtyPages for GuestLog {
    type Watch = ();

    fn watch(&self, _: Range<u64>) {}

    fn request(&self) -> DirtyStamp {
        let mut state = self.state();
        let stamp = state.now;
        state.now += 1;
        DirtyStamp(stamp)
    }

    fn wait(&self, _: DirtyStamp) -> bool {
        true
    }

    fn changed(&self, since: DirtyStamp, out: &mut Vec<Range<u64>>) {
        self.changed_by(since, 0, out);
    }

    fn wrote(&self, range: Range<u64>) {
        self.reported(range, 0);
    }

    fn wrote_as(&self, range: Range<u64>, writer: NonZeroU64) {
        self.reported(range, writer.get());
    }

    fn changed_except(&self, since: DirtyStamp, writer: NonZeroU64, out: &mut Vec<Range<u64>>) {
        self.changed_by(since, writer.get(), out);
    }
}

/// The guest pages test's command buffers and runs each way.
const GUEST_COMMAND_BUFFERS: usize = 8;
const GUEST_RUNS: u64 = 12;

/// What a run of the guest pages test saw: the runtime's reports, the gated copies in counted
/// (gated, whole, skipped, bytes whole, bytes copied).
struct GuestRun {
    marks: Vec<(Range<u64>, bool)>,
    gated: [u64; 5],
}

fn gated(counts: &WorkCounts) -> [u64; 5] {
    [
        counts.dirty_copies_in_gated,
        counts.dirty_copies_in_whole,
        counts.dirty_copies_in_skipped,
        counts.dirty_copy_in_bytes_whole,
        counts.dirty_copy_in_bytes_copied,
    ]
}

/// The texels of `bytes` laid out as the texture's buffer holds them, as words in a row.
fn texels_of(bytes: &[u8]) -> Vec<u8> {
    (0..SIDE as usize)
        .flat_map(|row| {
            let at = row * ROW as usize;
            bytes[at..at + TEXEL_ROW as usize].iter().copied()
        })
        .collect()
}

/// A run of the guest pages test (the module docs) on `device`, its command buffers encoded as
/// `schedule` says ([`encoded`]); every read and copy compared with the CPU model's, and the
/// pages after it, and every byte the GPU wrote found among the runtime's reports.
fn guest_run(
    device: &Device,
    queue: &CommandQueue,
    (mapping, log, pages, texture): (&Mapping, &GuestLog, &Buffer, &Texture),
    kernels: &TexelKernels,
    (run, schedule): (u64, Option<(usize, u64)>),
) -> GuestRun {
    let mut rng = Rng(0x6E57 + run);
    // The CPU writes the texture's bytes, as a guest does, and tells the log.
    let initial = rng.bytes(BYTES);
    mapping.write(0, &initial);
    log.told(0..BYTES);
    let words: Vec<Vec<u8>> = (0..GUEST_COMMAND_BUFFERS)
        .map(|_| rng.bytes(4 * WORDS))
        .collect();
    let filled: Vec<(u64, u8)> = (0..GUEST_COMMAND_BUFFERS)
        .map(|_| (rng.next() % SIDE, rng.next() as u8))
        .collect();
    let words_in: Vec<Buffer> = (words.iter())
        .map(|words| buffer_with(device, words).unwrap())
        .collect();
    let reads: Vec<Buffer> = (0..GUEST_COMMAND_BUFFERS)
        .map(|_| buffer_with(device, &vec![0xee; (4 * WORDS) as usize]).unwrap())
        .collect();
    let copies: Vec<Buffer> = (0..GUEST_COMMAND_BUFFERS)
        .map(|_| buffer_with(device, &vec![0xee; BYTES as usize]).unwrap())
        .collect();
    let row_of = |row: u64| row * ROW..row * ROW + TEXEL_ROW;
    let encode = |index: usize, cb: &CommandBuffer, pause: &mut dyn FnMut()| match index % 4 {
        // Write every texel through the texture, then copy the pages out.
        0 => {
            texel_dispatch(cb, &kernels.write, texture, &words_in[index]);
            pause();
            let mut blit = cb.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(pages, 0, &copies[index], 0, BYTES)
                .unwrap();
            blit.end_encoding().unwrap();
        }
        // Fill a row of the texture's bytes in the pages, then read it.
        2 => {
            let (row, value) = filled[index];
            let mut blit = cb.blit_command_encoder().unwrap();
            blit.fill_buffer(pages, row_of(row), value).unwrap();
            blit.end_encoding().unwrap();
            pause();
            texel_dispatch(cb, &kernels.read, texture, &reads[index]);
        }
        // Read it, in one encoder or in two: the second takes the image the first left
        // (`shadow::ShadowCarry`), and the command buffer's end leaves it holding every texel.
        _ => {
            texel_dispatch(cb, &kernels.read, texture, &reads[index]);
            if index % 4 == 3 {
                pause();
                texel_dispatch(cb, &kernels.read, texture, &reads[index]);
            }
        }
    };
    let before = gated(&device.work_counts());
    let command_buffers: Vec<CommandBuffer> = (0..GUEST_COMMAND_BUFFERS)
        .map(|_| queue.command_buffer().unwrap())
        .collect();
    encoded(&command_buffers, schedule, &encode);
    for cb in &command_buffers {
        cb.commit().unwrap();
    }
    for cb in &command_buffers {
        completed(cb);
    }
    let after = gated(&device.work_counts());
    // The CPU model, in commit order; and the bytes each command buffer's GPU wrote.
    let mut model = initial;
    let mut written: Vec<Range<u64>> = Vec::new();
    for index in 0..GUEST_COMMAND_BUFFERS {
        let what = format!("run {run} cb{index} under {schedule:?}");
        match index % 4 {
            0 => {
                for row in 0..SIDE {
                    let (at, from) = (row_of(row).start as usize, (row * TEXEL_ROW) as usize);
                    model[at..at + TEXEL_ROW as usize]
                        .copy_from_slice(&words[index][from..from + TEXEL_ROW as usize]);
                    written.push(row_of(row));
                }
                let copied = read_buffer(&copies[index], 0, BYTES).unwrap();
                assert_eq!(copied, model, "{what}: the pages copied out");
                continue;
            }
            2 => {
                let (row, value) = filled[index];
                let bytes = row_of(row);
                model[bytes.start as usize..bytes.end as usize].fill(value);
                written.push(bytes);
            }
            _ => {}
        }
        let read = read_buffer(&reads[index], 0, 4 * WORDS).unwrap();
        assert_eq!(read, texels_of(&model), "{what}: the texture read");
    }
    assert_eq!(
        mapping.read(0..BYTES as usize),
        model,
        "run {run} under {schedule:?}: the pages"
    );
    let marks = log.marks();
    for byte in written.iter().flat_map(Clone::clone) {
        assert!(
            (marks.iter()).any(|(mark, _)| mark.contains(&byte)),
            "run {run} under {schedule:?}: GPU-written byte {byte} among the reports {marks:?}"
        );
    }
    GuestRun {
        marks,
        gated: std::array::from_fn(|i| after[i] - before[i]),
    }
}

/// One queue's command buffers that write and read a shadowed texture over logged guest pages,
/// encoded on 2 to 4 threads out of commit order, hold the bytes, report the writes and gate the
/// copies in as serially encoded ones do (the module docs).
#[test]
fn guest_pages_follow_the_commit_order_however_the_command_buffers_were_encoded() {
    let cache = ScratchCache::new().unwrap();
    let mut name = String::new();
    let mut runs = |parallel: bool| -> Vec<GuestRun> {
        let messages = Messages::default();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        name = device.name().to_owned();
        let queue = device.new_command_queue().unwrap();
        let kernels = texel_kernels(&device);
        let mapping = Mapping::anonymous(super::host_memory::GUEST_PAGE);
        let log = Arc::new(GuestLog::default());
        // SAFETY: the mapping outlives the buffer, dropped first, and stays writable.
        let memory =
            unsafe { HostMemory::new(mapping.pointer()) }.with_dirty_pages(Arc::clone(&log));
        // SAFETY: as above.
        let pages = unsafe {
            device.new_buffer_with_bytes_no_copy(
                memory,
                super::host_memory::GUEST_PAGE as u64,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap();
        let shadows = device.work_counts().shadowed_textures;
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let texture = pages.new_texture(&descriptor, 0, ROW).unwrap();
        assert!(
            device.work_counts().shadowed_textures > shadows,
            "a texture whose rows have bytes between them is a shadow"
        );
        let guest = (&mapping, &*log, &pages, &texture);
        let runs: Vec<GuestRun> = (0..GUEST_RUNS)
            .map(|run| {
                let schedule = parallel.then(|| (2 + (run % 3) as usize, 0x6E57_0000 + run));
                guest_run(&device, &queue, guest, &kernels, (run, schedule))
            })
            .collect();
        drop((texture, pages, queue, kernels));
        drop(device);
        drop(mapping);
        layer_reported_nothing_at_all(&messages);
        runs
    };
    let serial = runs(false);
    let parallel = runs(true);
    let sum = |runs: &[GuestRun]| {
        (runs.iter()).fold([0; 5], |sum, run| {
            std::array::from_fn(|i| sum[i] + run.gated[i])
        })
    };
    println!(
        "PARALLEL-GUEST on {name}: over {GUEST_RUNS} runs of {GUEST_COMMAND_BUFFERS} command \
         buffers, gated copies in (gated, whole, skipped, bytes whole, bytes copied) serially \
         {:?}, on 2 to 4 threads {:?}; reports a run {:?}",
        sum(&serial),
        sum(&parallel),
        serial.last().map(|run| run.marks.len())
    );
    assert!(sum(&serial)[0] > 0, "the copies in were gated");
    for (run, (serial, parallel)) in serial.iter().zip(&parallel).enumerate() {
        assert_eq!(parallel.marks, serial.marks, "run {run}'s reports");
        assert_eq!(
            parallel.gated[0], serial.gated[0],
            "run {run}'s gated copies in"
        );
    }
}
