//! A fetched render target's storage form on the device the cfg selects, under the validation
//! layer: a runtime texture that a pass fetches through the pixel interlock renders into a scratch
//! image where its image lacks `STORAGE` (MoltenVK on Apple's GPU from 16² up, ANV for some
//! targets), and takes `STORAGE` once, at the commit of the device's only uncommitted command
//! buffer, whatever is in flight. Each test holds the switch to the device's work counts
//! (`Device::work_counts().storage_forms_taken`, not part of the API): deterministic, it counts.
//!
//! - **Once per texture.** A `SIDE`² `R32Float` target cleared to 1 and halved by `fetch_scalar`,
//!   then loaded and halved three times, each pass read back: 0.5, 0.25, 0.125, 0.0625. Where the
//!   first pass renders it through a scratch image (`fetch-through-scratch` names it), it takes
//!   its storage form at that pass's own commit, once, announced once (`fetched-target-storage`);
//!   elsewhere never. Its pipeline is made first, by a pass over an untracked target: a pass whose
//!   draw's `VkPipeline` the device's pool is still making is recorded by a host step, and its
//!   command buffer takes no switch (the first pass's own commit took none once the pool made the
//!   pipelines). A `SIDE`² array of two slices and two levels, its other slice and level cleared to
//!   their own values, keeps them through its switch.
//! - **Another command buffer holds it off.** A command buffer of another queue, made before the
//!   first fetch, holds the switch off: uncommitted, then committed and held by the scheduler for
//!   a shared event. The target's passes give the same answers through the scratch; once the
//!   event is signaled and the command buffer completed, the next commit takes it.
//! - **A steady stream** (the orchestrator's consumer shape): two queues, two frames in flight,
//!   the target fetched every frame on queue A and copied out every frame on queue B, beside an
//!   untracked target fetched every frame (its passes ordered by a fence, as Metal asks). For the
//!   first `OPEN` frames some command buffer is always open (made before the last one commits,
//!   as a consumer encoding on two threads keeps one); from then on each is committed before the
//!   next is made. The switch lands at frame `OPEN`'s commit on queue A, with command buffers of
//!   both queues in flight, among them a copy out of the target on queue B that reads the old
//!   image, held on the GPU by a kernel that spins on a flag the test raises beside the switch's
//!   commit (an event cannot hold it: a command buffer waiting for an event is held by the
//!   scheduler, which holds the switch off). The spin is bounded by construction: its cap is the
//!   loads of about a second at the rate the test measures first, well under any driver's timeout
//!   for one submission (amdgpu resets its ring after 10 s). On the M4 a running kernel never saw a
//!   host store made after it started, nor a release by another queue's kernel (where whether that
//!   kernel ran beside it is not told apart), so there the cap ends the hold. Hazard tracking makes
//!   the switch's copy wait for the held copy, the old image goes only once the switch's command
//!   buffer completed, and every frame's copy holds its frame's value. The untracked target never
//!   switches: its scratch stays. No encoding call waits for another queue's GPU work, as none of
//!   Metal's does (a pass's scratch image was moved to `GENERAL` by an internal submission waited
//!   for on the host, so on MoltenVK, whose one `VkQueue` the held copy is on, the tracked pass's
//!   draw waited the whole hold): the held copy is still in flight as the switch's commit begins,
//!   which the test asserts, naming each encoding call's time where it is not.
//! - **Through its resource ID** (the orchestrator's condition 2): the target's ID, in the
//!   argument buffer of the public case `sample-an-argument-buffer-texture-through-a-bound-sampler`
//!   (thread g samples `texture2d<float>` at (0.25g, 0.25g), nearest), read by a compute pass
//!   after each fetching pass, before and after the switch: `(v, 0, 0, 1)` per thread, the
//!   target's value, Metal's answer for a uniform `R32Float` texture. The ID stays.
//! - **Beside a command buffer encoding on another thread** (the consumer's host, which encodes a
//!   queue's command buffers on several threads and commits them in order): each frame's first
//!   command buffer fetches the target while its second, made before the first commits, is
//!   encoded on another thread, so the first's commit takes no switch and the second's only where
//!   the scheduler holds no batch. Every frame's copy holds its value, as serially on a device of
//!   its own; both take the form once by a lone pass at the end and announce the same kinds.
//! - **A handed-out image.** A target whose `VkImage` was taken (`vk_image`) keeps it: no switch,
//!   the same `VkImage` after.
//! - **Submission checks**. A command buffer with a check takes the switch too, settled at its
//!   submission once its checks ran. One whose check fails its fetching pass's submission (which
//!   runs nothing, the switch's copy included), or a blit's before the pass (the pass and the copy
//!   run), leaves the target on its old image and its scratch: no half-switched state. One whose
//!   checks pass switches it, and the next command buffer made waits for that.
//! - **The stall** (`storage_form_costs`, ignored: a measurement, run in release without
//!   the validation layer): per target, the commit that takes the switch, with and without a
//!   check, the wait of the next command buffer made after the checked one, and a fetching
//!   pass's GPU time through a scratch (an untracked target's) and in place, printed as
//!   `STORAGE-FORM-COST` lines.

use super::{library, loading, pass, pipeline, steps, SIDE, Z};
use crate::runtime_lane::degrade_notices::assert_made_once;
use crate::runtime_lane::resolves::lane_with;
use crate::runtime_lane::resolves::{lane, quiet, Lane, Shape, ONE};
use crate::runtime_lane::{
    bound_buffers, buffer_with, encode_case, pipeline_state, public_case, read_buffer,
    CaseResources, WAIT,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, RenderStages, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferError, CommandBufferStatus,
    CommandQueue, ComputePipelineState, CpuCacheMode, DebugMessage, Degradation, EncoderKind,
    EventRef, Fence, HazardTrackingMode, Origin, PipelineOptions, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, Size, StorageMode, StoreAction, Submission,
    SubmissionCheck, Texture, TextureDescriptor,
};
use std::collections::{BTreeSet, VecDeque};
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

/// The targets' format.
const PF: PixelFormat = PixelFormat::R32Float;

/// The public case whose argument buffer names the target.
const SAMPLER_CASE: &str = "sample-an-argument-buffer-texture-through-a-bound-sampler";

/// `value` in each texel of a `side`² `R32Float` level.
fn level_of(side: u64, value: f32) -> Vec<u8> {
    value.to_le_bytes().repeat((side * side) as usize)
}

/// The textures that took their storage form on `lane`'s device so far.
fn taken(lane: &Lane) -> u64 {
    lane.device.work_counts().storage_forms_taken
}

/// Level `level` of slice `slice` of the `R32Float` `texture`.
fn read(lane: &Lane, texture: &Texture, level: u64, slice: u64) -> Vec<u8> {
    lane.read(texture, (level, slice), 4, BlitOption::None)
}

/// A pass over level `level` of slice `slice` of `texture`, cleared to `value` and stored.
fn cleared(texture: &Texture, level: u64, slice: u64, value: f32) -> RenderPassDescriptor {
    let clear = ClearColor::new(f64::from(value), 0.0, 0.0, 0.0);
    let mut descriptor = pass(&[(texture, clear, StoreAction::Store)], None);
    descriptor.color_attachments[0].level = level;
    descriptor.color_attachments[0].slice = slice;
    descriptor
}

/// The descriptor of [`halving`]'s pipeline.
fn halving_descriptor(lane: &Lane) -> metal2vulkan::pipeline::RenderPipelineDescriptor {
    let vertices = library(lane, "rt6_resolves");
    let fetch = library(lane, "fragment_fetch");
    super::pipeline_descriptor(&vertices, (&fetch, "fetch_scalar"), &[PF], 1)
}

/// The fetching steps: `fetch_scalar` (half what it reads) over the fullscreen triangle.
fn halving(lane: &Lane) -> RenderPipelineState {
    let vertices = library(lane, "rt6_resolves");
    let fetch = library(lane, "fragment_fetch");
    pipeline(lane, &vertices, (&fetch, "fetch_scalar"), &[PF], 1).unwrap()
}

/// Clear slice 0 of level 0 of `texture` to 1 and halve it, then load and halve it three times,
/// reading it after each pass (each read a command buffer of its own): its texels after each.
fn four_passes(lane: &Lane, texture: &Texture, halve: &RenderPipelineState) -> Vec<Vec<u8>> {
    let side = texture.width();
    let first = cleared(texture, 0, 0, 1.0);
    steps(lane, &first, &[halve]).unwrap();
    let mut texels = vec![read(lane, texture, 0, 0)];
    for _ in 0..3 {
        steps(lane, &loading(texture, StoreAction::Store), &[halve]).unwrap();
        texels.push(read(lane, texture, 0, 0));
    }
    let expected: Vec<Vec<u8>> = [0.5, 0.25, 0.125, 0.0625]
        .iter()
        .map(|&value| level_of(side, value))
        .collect();
    assert_eq!(
        texels,
        expected,
        "the fetching passes on {}",
        lane.device.name()
    );
    texels
}

/// Whether the device announced that a fetching pass rendered a texture through a scratch image.
fn scratched(notices: &[DebugMessage]) -> bool {
    notices
        .iter()
        .filter_map(|message| message.degradation.as_ref())
        .any(|notice| notice.degradation == Degradation::FetchThroughScratch)
}

/// The kind [`a_fetched_target_takes_its_storage_form_once`] takes (`degrade_notices.rs`).
pub(in crate::runtime_lane) const STORAGE_FORMS: [Degradation; 1] =
    [Degradation::FetchedTargetStorage];

/// A fetched target takes its storage form once, at its first fetching pass's own commit, where
/// that pass rendered it through a scratch image; its texels, those of its other slice and level
/// included, are kept (the module docs).
#[test]
fn a_fetched_target_takes_its_storage_form_once() {
    let (messages, cache, lane) = lane();
    let device = lane.device.name().to_owned();
    let halve = halving(&lane);
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let plain = lane.texture(PF, Shape { size: SIDE, ..ONE }, usage);
    // The fetching pass's pipeline made first, by a pass over an untracked target (which never
    // takes its storage form): a draw whose `VkPipeline` the device's pool is still making is
    // recorded by a host step, and its command buffer takes no switch (the module docs).
    let warm = untracked(&lane);
    steps(&lane, &cleared(&warm, 0, 0, 1.0), &[&halve]).unwrap();
    let pooled = lane.device.work_counts().pipelines_pooled;
    let before = taken(&lane);
    let first = cleared(&plain, 0, 0, 1.0);
    steps(&lane, &first, &[&halve]).unwrap();
    assert_eq!(
        lane.device.work_counts().pipelines_pooled,
        pooled,
        "the first pass of the target found its pipeline made on {device}"
    );
    // Ended, and announced, by the time the pass's command buffer completed (the scheduler's
    // thread ends it before the command buffer's transition to `Scheduled`), with no other command
    // buffer made since (the module docs).
    let notices = messages.take_notices();
    let through_scratch = scratched(&notices);
    let completed_announced = (notices.iter())
        .filter_map(|message| message.degradation.as_ref())
        .any(|notice| notice.degradation == Degradation::FetchedTargetStorage);
    // The pass's command buffer was the device's only uncommitted one.
    let switched = u64::from(through_scratch);
    assert_eq!(
        (taken(&lane) - before, completed_announced),
        (switched, through_scratch),
        "the switch and its announcement by its first pass's completion on {device}"
    );
    let at = Instant::now();
    drop(lane.queue.command_buffer().unwrap());
    let waited = at.elapsed();
    assert_eq!(
        taken(&lane),
        before + switched,
        "at its first pass's commit on {device} (the next command buffer waited {waited:?})"
    );
    assert_eq!(read(&lane, &plain, 0, 0), level_of(SIDE, 0.5));
    for value in [0.25, 0.125, 0.0625] {
        steps(&lane, &loading(&plain, StoreAction::Store), &[&halve]).unwrap();
        assert_eq!(
            read(&lane, &plain, 0, 0),
            level_of(SIDE, value),
            "on {device}"
        );
    }
    assert_eq!(taken(&lane), before + switched, "once, on {device}");

    // An array of two slices and two levels: its other slice and level keep their texels.
    let shape = Shape {
        size: SIDE,
        slices: 2,
        levels: 2,
        ..ONE
    };
    let array = lane.texture(PF, shape, usage);
    let others = [(0, 1, 0.75), (1, 0, 0.375), (1, 1, 0.1875)];
    for &(level, slice, value) in &others {
        steps(&lane, &cleared(&array, level, slice, value), &[]).unwrap();
    }
    let counted = taken(&lane);
    four_passes(&lane, &array, &halve);
    let array_switched = taken(&lane) - counted;
    assert!(
        array_switched <= 1,
        "the array switched {array_switched} times on {device}"
    );
    for &(level, slice, value) in &others {
        let side = SIDE >> level;
        assert_eq!(
            read(&lane, &array, level, slice),
            level_of(side, value),
            "level {level} of slice {slice} on {device}"
        );
    }
    let notices = [notices, messages.take_notices()].concat();
    // The scratch and the storage form where the first pass took the scratch; the pipeline's
    // making; the passes' and reads' `Shared` buffers.
    let kinds = match through_scratch {
        true => vec![
            Degradation::FetchThroughScratch,
            Degradation::FetchedTargetStorage,
        ],
        false => Vec::new(),
    };
    let first = super::made(&lane, &[halving_descriptor(&lane)]);
    messages.restore_notices(notices);
    let counts = assert_made_once(&messages, &kinds, &first);
    let announced = counts.contains_key(&Degradation::FetchedTargetStorage);
    assert_eq!(
        announced, through_scratch,
        "the notice on {device}: {counts:?}"
    );
    eprintln!(
        "storage form on {device}: the 2D target through a scratch {through_scratch}, took its \
         form {switched} time(s) (ended by its first pass's completion, the next command buffer \
         waited {waited:?}); the array {array_switched} time(s); its texels kept"
    );
    drop(cache);
    quiet(lane, &messages);
}

/// A command buffer of another queue holds the switch off: uncommitted, then committed and held
/// for a shared event; once it completed, the next commit takes it (the module docs).
#[test]
fn an_incomplete_command_buffer_holds_the_storage_form_off() {
    let (messages, cache, lane) = lane();
    let device = lane.device.name().to_owned();
    let halve = halving(&lane);
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let texture = lane.texture(PF, Shape { size: SIDE, ..ONE }, usage);
    let other = lane.device.new_command_queue().unwrap();
    let event = lane.device.new_shared_event().unwrap();
    let held: CommandBuffer = other.command_buffer().unwrap();
    let before = taken(&lane);
    steps(&lane, &cleared(&texture, 0, 0, 1.0), &[&halve]).unwrap();
    let through_scratch = scratched(&messages.take_notices());
    assert_eq!(read(&lane, &texture, 0, 0), level_of(SIDE, 0.5));
    steps(&lane, &loading(&texture, StoreAction::Store), &[&halve]).unwrap();
    assert_eq!(read(&lane, &texture, 0, 0), level_of(SIDE, 0.25));
    assert_eq!(
        taken(&lane),
        before,
        "uncommitted on another queue, on {device}"
    );
    held.encode_wait_for_event(EventRef::from(&event), 1)
        .unwrap();
    held.commit().unwrap();
    steps(&lane, &loading(&texture, StoreAction::Store), &[&halve]).unwrap();
    assert_eq!(read(&lane, &texture, 0, 0), level_of(SIDE, 0.125));
    assert_eq!(taken(&lane), before, "committed and waiting, on {device}");
    event.set_signaled_value(1);
    held.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(held.error(), None);
    drop(held);
    steps(&lane, &loading(&texture, StoreAction::Store), &[&halve]).unwrap();
    assert_eq!(read(&lane, &texture, 0, 0), level_of(SIDE, 0.0625));
    assert_eq!(
        taken(&lane),
        before + u64::from(through_scratch),
        "once the other command buffer completed, on {device}"
    );
    eprintln!(
        "storage form on {device}: held off by an incomplete command buffer of another queue \
         (through a scratch {through_scratch})"
    );
    drop(cache);
    quiet(lane, &messages);
}

/// The target's resource ID, in an argument buffer a compute pass reads after each fetching
/// pass, reaches its texels before and after its storage form: `(v, 0, 0, 1)` in each of the
/// case's four threads, and the ID stays (the module docs).
#[test]
fn an_argument_buffer_reaches_the_target_through_its_storage_form() {
    let (messages, cache, lane) = lane();
    let device = lane.device.name().to_owned();
    let halve = halving(&lane);
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let texture = lane.texture(PF, Shape { size: SIDE, ..ONE }, usage);
    let id = texture.gpu_resource_id();
    let case = public_case(SAMPLER_CASE).unwrap();
    let state = pipeline_state(&lane.device, &case).unwrap();
    let mut resources = CaseResources::new(&lane.device, &case).unwrap();
    let fields: Vec<(u32, u32)> = resources.argument_textures.keys().copied().collect();
    assert_eq!(
        fields,
        [(0, 0)],
        "{SAMPLER_CASE}: one argument-buffer texture"
    );
    resources.argument_textures.insert((0, 0), texture.clone());
    let buffers = bound_buffers(&lane.device, &case.compute, &resources).unwrap();
    let sample = |value: f32| {
        let command_buffer = lane.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encode_case(&mut encoder, &state, &case.compute, &buffers, &resources).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "on {device}");
        let texel: Vec<u8> = [value, 0.0, 0.0, 1.0]
            .iter()
            .flat_map(|c| c.to_le_bytes())
            .collect();
        assert_eq!(
            read_buffer(&buffers[&1], 0, 64).unwrap(),
            texel.repeat(4),
            "through ID {id:?} at {value} on {device}"
        );
    };
    steps(&lane, &cleared(&texture, 0, 0, 1.0), &[]).unwrap();
    sample(1.0);
    let before = taken(&lane);
    steps(&lane, &loading(&texture, StoreAction::Store), &[&halve]).unwrap();
    let through_scratch = scratched(&messages.take_notices());
    // The fetching pass's commit takes the storage form (the sampling command buffer binds the
    // heap sets, so its own never does): the sampling's read is the first after it.
    sample(0.5);
    for value in [0.25, 0.125] {
        steps(&lane, &loading(&texture, StoreAction::Store), &[&halve]).unwrap();
        sample(value);
    }
    assert_eq!(
        taken(&lane),
        before + u64::from(through_scratch),
        "on {device}"
    );
    assert_eq!(texture.gpu_resource_id(), id, "on {device}");
    eprintln!(
        "storage form on {device}: an argument buffer reads the target through its ID before and \
         after (through a scratch {through_scratch})"
    );
    drop(cache);
    quiet(lane, &messages);
}

/// The frames of [`a_target_fetched_beside_an_encoding_command_buffer_keeps_its_answers`].
const BESIDE_FRAMES: usize = 8;

/// What a run of [`beside`] saw: the kinds announced, the forms taken after each frame and after
/// the lone pass, and whether the first pass rendered the target through a scratch.
struct Beside {
    device: String,
    kinds: BTreeSet<Degradation>,
    taken: Vec<u64>,
    through_scratch: bool,
}

/// [`BESIDE_FRAMES`] frames of two command buffers of one queue, the second a fetching pass and a
/// copy out of the target, then one lone fetching pass: on two threads at once from a barrier,
/// both made before either commits, where `parallel`, else each made, encoded and committed in
/// turn; committed in order either way. Every frame's copy holds its value, and the lone pass's.
fn beside(parallel: bool) -> Beside {
    let (messages, cache, lane) = lane();
    let device = lane.device.name().to_owned();
    let halve = halving(&lane);
    // The pipeline made first, by a pass over an untracked target (`a_fetched_target_takes_its_
    // storage_form_once`).
    let warm = untracked(&lane);
    steps(&lane, &cleared(&warm, 0, 0, 1.0), &[&halve]).unwrap();
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let texture = lane.texture(PF, Shape { size: SIDE, ..ONE }, usage);
    let depth = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
    let bytes = 4 * SIDE * SIDE;
    let before = taken(&lane);
    let mut taken_after = Vec::new();
    let mut value = 1.0f32;
    for frame in 0..BESIDE_FRAMES {
        let copy = buffer_with(&lane.device, &vec![0; bytes as usize]).unwrap();
        let first = |cb: &CommandBuffer| {
            frame_pass(cb, (&texture, 2 * frame), &halve, &depth, None);
        };
        let second = |cb: &CommandBuffer| {
            frame_pass(cb, (&texture, 2 * frame + 1), &halve, &depth, None);
            let mut blit = cb.blit_command_encoder().unwrap();
            blit.copy_from_texture_to_buffer(
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(SIDE, SIDE, 1),
                &copy,
                0,
                4 * SIDE,
                bytes,
                BlitOption::None,
            )
            .unwrap();
            blit.end_encoding().unwrap();
        };
        let last = if parallel {
            let one = lane.queue.command_buffer().unwrap();
            let two = lane.queue.command_buffer().unwrap();
            let start = Barrier::new(2);
            std::thread::scope(|scope| {
                let (start, one, two) = (&start, &one, &two);
                scope.spawn(move || {
                    start.wait();
                    first(one);
                });
                scope.spawn(move || {
                    start.wait();
                    second(two);
                });
            });
            one.commit().unwrap();
            two.commit().unwrap();
            two
        } else {
            let one = lane.queue.command_buffer().unwrap();
            first(&one);
            one.commit().unwrap();
            let two = lane.queue.command_buffer().unwrap();
            second(&two);
            two.commit().unwrap();
            two
        };
        last.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(last.error(), None, "frame {frame} on {device}");
        value /= 4.0;
        assert_eq!(
            read_buffer(&copy, 0, bytes).unwrap(),
            level_of(SIDE, value),
            "frame {frame}'s copy on {device}, parallel {parallel}"
        );
        taken_after.push(taken(&lane) - before);
    }
    steps(&lane, &loading(&texture, StoreAction::Store), &[&halve]).unwrap();
    // A new command buffer waits out a switch under way (`texture/storage_form.rs`).
    drop(lane.queue.command_buffer().unwrap());
    taken_after.push(taken(&lane) - before);
    assert_eq!(
        read(&lane, &texture, 0, 0),
        level_of(SIDE, value / 2.0),
        "the lone pass on {device}, parallel {parallel}"
    );
    let notices = messages.take_notices();
    let through_scratch = scratched(&notices);
    let kinds = (notices.iter())
        .filter_map(|message| message.degradation.as_ref())
        .map(|notice| notice.degradation)
        .collect();
    drop(cache);
    quiet(lane, &messages);
    Beside {
        device,
        kinds,
        taken: taken_after,
        through_scratch,
    }
}

/// One queue's command buffers encoded on two threads at once, then committed in order (the
/// consumer's host): each frame's first command buffer fetches the target, halving it, while its
/// second, made before the first commits, is encoded on another thread: a pass that fetches and
/// halves it again, and a copy of it out. So each frame's first commit leaves another command
/// buffer uncommitted and takes no switch (`texture/storage_form.rs`, the commit of the device's
/// only uncommitted command buffer); the second, the only one then, takes it only where the
/// scheduler holds no batch, which the first's commit just gave it. Serially, on a device of its
/// own, each is made, encoded and committed in turn. Both ways every frame's copy holds
/// 2^-2(f+1), both end with a lone fetching pass after which the target took its storage form
/// once (where its first pass rendered it through a scratch), and both announce the same kinds.
/// When each took it is printed (`PARALLEL-STORAGE-FORM`): later in parallel is the cost the
/// deferral has, the scratch's copies in and out at each pass until then.
#[test]
fn a_target_fetched_beside_an_encoding_command_buffer_keeps_its_answers() {
    let serial = beside(false);
    let parallel = beside(true);
    let device = &serial.device;
    println!(
        "PARALLEL-STORAGE-FORM on {device}: through a scratch {}, forms taken after each frame and the lone pass serially {:?}, on two threads {:?}",
        serial.through_scratch, serial.taken, parallel.taken
    );
    assert_eq!(
        serial.through_scratch, parallel.through_scratch,
        "on {device}"
    );
    let once = u64::from(serial.through_scratch);
    assert_eq!(
        (serial.taken.last(), parallel.taken.last()),
        (Some(&once), Some(&once)),
        "by the lone pass, on {device}"
    );
    assert_eq!(
        serial.kinds, parallel.kinds,
        "the kinds announced on {device}"
    );
}

/// A target whose `VkImage` was handed out keeps it: no storage form, the same `VkImage` after
/// its fetching passes (the module docs).
#[test]
fn a_handed_out_image_keeps_its_form() {
    let (messages, cache, lane) = lane();
    let device = lane.device.name().to_owned();
    let halve = halving(&lane);
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let texture = lane.texture(PF, Shape { size: SIDE, ..ONE }, usage);
    // SAFETY: the image is only compared, never used.
    let image = unsafe { texture.vk_image() };
    assert!(image.is_some(), "a runtime texture's image on {device}");
    let before = taken(&lane);
    four_passes(&lane, &texture, &halve);
    assert_eq!(taken(&lane), before, "on {device}");
    // SAFETY: as above.
    assert_eq!(unsafe { texture.vk_image() }, image, "on {device}");
    drop(cache);
    quiet(lane, &messages);
}

/// The stream's frames.
const FRAMES: usize = 12;

/// The stream's first frames, during which some command buffer is always open.
const OPEN: usize = 6;

/// The most GPU time the held copy's kernel spins for its flag before it gives up: about a
/// second, well under any driver's timeout for one submission (amdgpu resets its ring after 10 s),
/// and far longer than the host takes between the held copy's commit and the
/// switch's.
const HOLD: Duration = Duration::from_secs(1);

/// The least time [`spinning`] measures its spin's rate over: long enough that a commit's own
/// cost is a small part of it.
const MEASURED: Duration = Duration::from_millis(50);

/// The word the held copy's kernel publishes once it saw its flag.
const PUBLISHED: u32 = 5;

/// The held copy's kernel and the loads it spins for its flag before it gives up.
struct Spin {
    state: ComputePipelineState,
    cap: u32,
}

/// The held copy's kernel: the owned fixture `rfs_agx_yield`'s `agx_yield_lookback` as one
/// threadgroup that takes ticket 1 and spins, an atomic load of device memory each time, until the
/// word before its own is odd, then publishes its own, [`PUBLISHED`]; after its cap of loads it
/// gives up and publishes nothing. Made, and run once released from the start, before the stream: a
/// command buffer whose `VkPipeline` is still being made waits for a host step, which holds a
/// switch off (the module docs). Then run unreleased, its cap growing until the least of three
/// runs' GPU windows takes [`MEASURED`]: the cap is the loads of [`HOLD`] at that rate. A run's
/// GPU window, from its GPU start time to its GPU end time, bounds its GPU time: the device's
/// timestamps where it writes them, else a window its work ran in (`gpu_time.rs`). So the rate is
/// never overstated, and the hold lasts about [`HOLD`], longer only as far as other work on the
/// GPU slows the spin. The window leaves out the commit's and the wait's own time, which a host
/// loaded by other processes stretches: on an RTX 3090 beside two GPUs' lane suites (a load of 25
/// on 16 hardware threads) a run of one load took 35.8 ms from its commit to its wait's return,
/// its GPU window 0.02 ms, and one such run at the first cap would end the growth with the rate
/// understated as many times over. Where the window comes from the host it holds the
/// completion's latency, which only adds, so the least of three runs is taken.
fn spinning(lane: &Lane, queue: &CommandQueue) -> Spin {
    let device = lane.device.name();
    let function = library(lane, "rfs_agx_yield")
        .new_function_with_name("agx_yield_lookback")
        .unwrap();
    let state = lane
        .device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap();
    // One run of `cap` loads, the flag up from the start where `up`: whether the kernel saw it,
    // and the run's GPU window.
    let run = |cap: u32, up: bool| {
        let command_buffer = queue.command_buffer().unwrap();
        let gate = Gate::encode(lane, &command_buffer, (&state, cap), up);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "the spin on {device}");
        let (start, end) = (
            command_buffer.gpu_start_time(),
            command_buffer.gpu_end_time(),
        );
        let window = match (start, end) {
            (Ok(start), Ok(end)) => Duration::from_secs_f64((end - start).max(0.0)),
            times => panic!("the spin's GPU times on {device}: {times:?}"),
        };
        (gate.published(), window)
    };
    assert!(
        run(1, true).0,
        "the spin publishes once its flag is up, on {device}"
    );
    let mut loads = 1u32 << 12;
    let took = loop {
        let took = (0..3)
            .map(|_| {
                let (published, took) = run(loads, false);
                assert!(
                    !published,
                    "an unreleased spin of {loads} loads publishes nothing, on {device}"
                );
                took
            })
            .min()
            .unwrap();
        if took >= MEASURED || loads >= 1 << 28 {
            break took;
        }
        loads <<= 4;
    };
    let cap = f64::from(loads) * HOLD.as_secs_f64() / took.as_secs_f64();
    Spin {
        state,
        cap: cap.clamp(1.0, f64::from(u32::MAX)) as u32,
    }
}

/// A command buffer's hold on the GPU: [`spinning`]'s kernel, its flag (the word before its own)
/// in a `Shared` buffer the host writes.
struct Gate {
    words: Buffer,
}

impl Gate {
    /// Encode `state`'s spin of at most `cap` loads into `command_buffer`, its flag already up
    /// where `up`.
    fn encode(
        lane: &Lane,
        command_buffer: &CommandBuffer,
        (state, cap): (&ComputePipelineState, u32),
        up: bool,
    ) -> Gate {
        let ticket = buffer_with(&lane.device, &1u32.to_le_bytes()).unwrap();
        let words = [u32::from(up), 0]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<u8>>();
        let words = buffer_with(&lane.device, &words).unwrap();
        let cap = buffer_with(&lane.device, &cap.to_le_bytes()).unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(state).unwrap();
        encoder.set_buffer(Some(&ticket), 0, 0).unwrap();
        encoder.set_buffer(Some(&words), 0, 1).unwrap();
        encoder.set_buffer(Some(&cap), 0, 2).unwrap();
        let one = Size::new(1, 1, 1);
        encoder.dispatch_threadgroups(one, one).unwrap();
        encoder.end_encoding().unwrap();
        Gate { words }
    }

    /// Raise the flag: an atomic store into the buffer the kernel's atomic loads read.
    fn release(&self) {
        let contents = self.words.contents().unwrap();
        // SAFETY: the buffer's first word, `Shared` and 4-aligned, which the host and the
        // kernel's atomics alone touch while the kernel runs.
        let flag = unsafe { AtomicU32::from_ptr(contents.as_ptr().cast::<u32>()) };
        flag.store(1, Ordering::SeqCst);
    }

    /// Raise the flag once `command_buffer`'s commit took its place: its status past
    /// `NotEnqueued`, which the commit sets after it decides on a switch and before it submits.
    /// A commit may wait in the driver for the work its batch waits for (MoltenVK on the M4), so
    /// the flag goes up beside it, never after it returns.
    fn release_once_committed(&self, command_buffer: &CommandBuffer) {
        let deadline = Instant::now() + WAIT;
        while matches!(
            command_buffer.status(),
            CommandBufferStatus::NotEnqueued | CommandBufferStatus::Enqueued
        ) && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_micros(100));
        }
        self.release();
    }

    /// Whether the kernel saw its flag: its own word published.
    fn published(&self) -> bool {
        read_buffer(&self.words, 4, 4).unwrap() == PUBLISHED.to_le_bytes()
    }
}

/// A private `SIDE`² `R32Float` render target with untracked hazards.
fn untracked(lane: &Lane) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PF, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::new(
        StorageMode::Private,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Untracked,
    );
    lane.device.new_texture(&descriptor).unwrap()
}

/// The calls that encode a frame's pass, as [`frame_pass`] times them.
const PASS_CALLS: [&str; 4] = [
    "render_command_encoder",
    "wait_for_fence and the bindings",
    "draw_primitives",
    "update_fence and end_encoding",
];

/// Encode into `command_buffer` frame `frame`'s pass of `texture`: cleared to 1 at frame 0, loaded
/// after, halved by `halve` and stored. With `fence` (an untracked target's), the pass waits for
/// the fence's update by the frame before, and updates it. The time each of [`PASS_CALLS`] took.
fn frame_pass(
    command_buffer: &CommandBuffer,
    (texture, frame): (&Texture, usize),
    halve: &RenderPipelineState,
    depth: &Buffer,
    fence: Option<&Fence>,
) -> [Duration; 4] {
    let descriptor = if frame == 0 {
        cleared(texture, 0, 0, 1.0)
    } else {
        loading(texture, StoreAction::Store)
    };
    let mut at = Instant::now();
    let mut lap = || {
        let now = Instant::now();
        let took = now - at;
        at = now;
        took
    };
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    let begun = lap();
    if let (Some(fence), true) = (fence, frame > 0) {
        encoder
            .wait_for_fence(fence, RenderStages::VERTEX | RenderStages::FRAGMENT)
            .unwrap();
    }
    encoder.set_vertex_buffer(Some(depth), 0, 0).unwrap();
    encoder.set_render_pipeline_state(halve).unwrap();
    let bound = lap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    let drawn = lap();
    if let Some(fence) = fence {
        encoder.update_fence(fence, RenderStages::FRAGMENT).unwrap();
    }
    encoder.end_encoding().unwrap();
    [begun, bound, drawn, lap()]
}

/// A consumer-shaped stream (the module docs): two queues, two frames in flight, a tracked target
/// fetched every frame on queue A and copied out on queue B, an untracked one fetched beside it.
/// The tracked target switches once, at frame `OPEN`'s commit on queue A, the first commit with
/// no other command buffer open, while a copy out of it on queue B is in flight, held on the GPU
/// by a kernel that spins until the test raises its flag beside that commit, or for about
/// [`HOLD`] where the device never shows the running kernel its flag; every frame's copy holds its
/// value, and the untracked target keeps its scratch. The held copy is still in flight as the
/// switch's commit begins: no encoding call before it waited for the GPU.
#[test]
fn a_steady_stream_takes_the_storage_form_at_its_first_lone_commit() {
    let (messages, cache, lane) = lane();
    let device = lane.device.name().to_owned();
    let halve = halving(&lane);
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let tracked = lane.texture(PF, Shape { size: SIDE, ..ONE }, usage);
    let unordered = untracked(&lane);
    let fence = lane.device.new_fence();
    let other = lane.device.new_command_queue().unwrap();
    let spin = spinning(&lane, &other);
    let depth = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
    let bytes = 4 * SIDE * SIDE;
    let copies: Vec<Buffer> = (0..FRAMES)
        .map(|_| buffer_with(&lane.device, &vec![0; bytes as usize]).unwrap())
        .collect();
    let before = taken(&lane);
    let mut flight: VecDeque<(usize, CommandBuffer)> = VecDeque::new();
    // Every frame's command buffers, for the times a wrong copy is reported with.
    let mut frames: Vec<(usize, char, CommandBuffer)> = Vec::new();
    // Queue B's command buffer of the next frame, or queue A's, made before the last one commits.
    let mut next_b = Some(other.command_buffer().unwrap());
    let mut next_a: Option<CommandBuffer> = None;
    let mut through_scratch = None;
    // Queue B's copy of frame `OPEN - 1`, held by its gate, and when it was committed.
    let mut held: Option<(CommandBuffer, Gate, Instant)> = None;
    // Where the switch landed: the frame, the queue, and, as its commit began, the command
    // buffers in flight, whether the held copy was among them, and how long after the held copy's
    // commit. Taken as the commit begins: a driver may hold the commit until the work that the
    // switch's copy waits for completes (MoltenVK on the M4).
    let mut switched: Option<(usize, char, usize, bool, Duration)> = None;
    // The host's steps between the held copy's commit and the switch's: to frame `OPEN`'s start,
    // its wait for frame `OPEN - 2`, and the encoding of its passes, each call's time.
    let mut host_steps: Option<([Duration; 2], [[Duration; 4]; 2])> = None;
    let snapshot = |flight: &VecDeque<(usize, CommandBuffer)>,
                    held: &Option<(CommandBuffer, Gate, Instant)>| {
        let pending = |cb: &CommandBuffer| cb.status() != CommandBufferStatus::Completed;
        let in_flight = flight.iter().filter(|(_, cb)| pending(cb)).count();
        let held_pending = held.as_ref().is_some_and(|(cb, _, _)| pending(cb));
        let after = held
            .as_ref()
            .map_or(Duration::ZERO, |(_, _, at)| at.elapsed());
        (in_flight, held_pending, after)
    };
    // Read once the command buffer just committed is scheduled: a switch riding on it ends by
    // then, on the scheduler's thread (the module docs).
    let mut look = |frame: usize, queue: char, committed: &CommandBuffer, seen| {
        committed.wait_until_scheduled().unwrap();
        let (in_flight, held_pending, after) = seen;
        if switched.is_none() && taken(&lane) > before {
            switched = Some((frame, queue, in_flight, held_pending, after));
        }
    };
    for (frame, copy) in copies.iter().enumerate() {
        let started = Instant::now();
        // Two frames in flight: frame `f` waits for frame `f - 2`.
        while flight.front().is_some_and(|(at, _)| at + 2 <= frame) {
            let (_, done) = flight.pop_front().unwrap();
            done.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(done.error(), None, "on {device}");
        }
        let waited = Instant::now();
        let a = next_a
            .take()
            .unwrap_or_else(|| lane.queue.command_buffer().unwrap());
        let passes = [
            frame_pass(&a, (&tracked, frame), &halve, &depth, None),
            frame_pass(&a, (&unordered, frame), &halve, &depth, Some(&fence)),
        ];
        if let (Some((_, _, at)), true) = (held.as_ref(), frame == OPEN) {
            let steps = [started.duration_since(*at), waited.duration_since(started)];
            host_steps = Some((steps, passes));
        }
        let seen = snapshot(&flight, &held);
        match held.as_ref().filter(|_| frame == OPEN) {
            // The held copy's flag goes up once the switch's commit took its place.
            Some((_, gate, _)) => std::thread::scope(|scope| {
                scope.spawn(|| gate.release_once_committed(&a));
                a.commit().unwrap();
            }),
            None => a.commit().unwrap(),
        }
        through_scratch.get_or_insert_with(|| scratched(&messages.take_notices()));
        look(frame, 'A', &a, seen);
        frames.push((frame, 'A', a.clone()));
        flight.push_back((frame, a));

        let b = next_b
            .take()
            .unwrap_or_else(|| other.command_buffer().unwrap());
        // The next command buffer, made before this one commits: queue B's through frame `OPEN -
        // 1`, then queue A's for frame `OPEN`.
        if frame + 1 < OPEN {
            next_b = Some(other.command_buffer().unwrap());
        } else if frame + 1 == OPEN {
            next_a = Some(lane.queue.command_buffer().unwrap());
        }
        let gate =
            (frame + 1 == OPEN).then(|| Gate::encode(&lane, &b, (&spin.state, spin.cap), false));
        let mut blit = b.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &tracked,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            copy,
            0,
            4 * SIDE,
            bytes,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        let seen = snapshot(&flight, &held);
        b.commit().unwrap();
        if let Some(gate) = gate {
            held = Some((b.clone(), gate, Instant::now()));
        }
        look(frame, 'B', &b, seen);
        frames.push((frame, 'B', b.clone()));
        flight.push_back((frame, b));
    }
    for (_, done) in flight {
        done.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(done.error(), None, "on {device}");
    }
    // A frame's command buffers' kernel start and GPU start and end times, in microseconds from
    // the first's GPU start: a copy that misses its frame's value says how they ran.
    let origin = frames[0].2.gpu_start_time().unwrap();
    let times = |at: usize| {
        let us = |seconds: f64| (seconds - origin) * 1e6;
        frames
            .iter()
            .filter(|(frame, _, _)| *frame == at)
            .map(|(_, queue, cb)| {
                format!(
                    "{queue}: kernel {:.0}, gpu {:.0} to {:.0}",
                    us(cb.kernel_start_time()),
                    us(cb.gpu_start_time().unwrap()),
                    us(cb.gpu_end_time().unwrap())
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    };
    let mut value = 1.0f32;
    for (frame, copy) in copies.iter().enumerate() {
        value *= 0.5;
        let read = read_buffer(copy, 0, bytes).unwrap();
        let mut seen: Vec<u32> = read
            .as_chunks::<4>()
            .0
            .iter()
            .map(|texel| u32::from_le_bytes(*texel))
            .collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            read,
            level_of(SIDE, value),
            "frame {frame}'s copy on {device}: its texels' bits {seen:x?}; the frame's command \
             buffers (us) {}",
            times(frame)
        );
    }
    assert_eq!(
        read(&lane, &tracked, 0, 0),
        level_of(SIDE, value),
        "on {device}"
    );
    assert_eq!(
        read(&lane, &unordered, 0, 0),
        level_of(SIDE, value),
        "the untracked target on {device}"
    );
    let scratch = through_scratch.unwrap_or(false);
    assert_eq!(
        taken(&lane),
        before + u64::from(scratch),
        "the tracked target alone, once, on {device}"
    );
    // How the held copy's hold ended: its flag seen, or its cap of about `HOLD` reached.
    let (held_copy, gate, _) = held.as_ref().unwrap();
    let on_gpu = held_copy
        .gpu_end_time()
        .and_then(|end| held_copy.gpu_start_time().map(|start| end - start))
        .map_or_else(|error| format!("{error}"), |s| format!("{:.0} ms", s * 1e3));
    let ([to_frame, waiting], passes) = host_steps.unwrap();
    let encoding = ["tracked", "untracked"]
        .iter()
        .zip(passes)
        .flat_map(|(target, calls)| {
            PASS_CALLS
                .iter()
                .zip(calls)
                .map(move |(call, took)| format!("the {target} pass's {call} {took:?}"))
        })
        .collect::<Vec<_>>()
        .join(", ");
    let host = format!(
        "the host's steps after the held copy's commit: {to_frame:?} to frame {OPEN}'s start, \
         {waiting:?} waiting for frame {}'s command buffers, then {encoding}",
        OPEN - 2
    );
    let hold = if gate.published() {
        format!("released by its flag, {on_gpu} on the GPU")
    } else {
        format!(
            "its flag, raised beside the switch's commit, unseen: it gave up after its cap of {} \
             loads (about {HOLD:?}), {on_gpu} on the GPU",
            spin.cap
        )
    };
    if scratch {
        let (frame, queue, in_flight, held_in_flight, after) = switched.unwrap();
        assert_eq!(
            (frame, queue),
            (OPEN, 'A'),
            "the switch's commit on {device}"
        );
        assert!(
            held_in_flight,
            "storage form on {device}: the copy of the old image on the other queue had completed \
             as the switch's commit began, {after:?} after its commit: an encoding call waited \
             for it ({hold}; {host})"
        );
        eprintln!(
            "storage form on {device}: a steady stream of two queues switched at frame {frame}'s \
             commit on queue {queue}, the first with no other command buffer open (none in the \
             {OPEN} frames before), with {in_flight} command buffer(s) in flight, the copy of the \
             old image on the other queue (held on the GPU, committed {after:?} before the \
             switch's commit; {hold}) among them; every frame's copy and the untracked target \
             (which kept its scratch) exact; {host}"
        );
    } else {
        assert_eq!(switched, None, "on {device}");
        eprintln!("storage form on {device}: no scratch, no switch, the stream exact ({hold})");
    }
    drop(cache);
    quiet(lane, &messages);
}

/// The code a failing check answers: `OutOfMemory`, as Metal fails a detached resource's.
const CHECK_CODE: u64 = 8;

/// A submission check that fails the submissions of `kind` (none where `None`), counting every
/// submission it sees in `seen`.
fn failing(kind: Option<EncoderKind>, seen: &Arc<AtomicUsize>) -> SubmissionCheck {
    let seen = Arc::clone(seen);
    Arc::new(move |submission: &Submission<'_>| {
        seen.fetch_add(1, Ordering::SeqCst);
        (Some(submission.kind()) == kind).then(|| NonZeroU64::new(CHECK_CODE).unwrap())
    })
}

/// A command buffer with a check takes the switch only where every encoder of it ran: a check
/// that fails its fetching pass's submission, or a blit's before the pass, leaves the target on
/// its old image and its scratch; one whose checks pass switches it, settled at its submission,
/// and the next command buffer made waits for that (the module docs).
#[test]
fn a_checked_command_buffer_takes_the_storage_form_only_where_its_checks_pass() {
    let (messages, cache, lane) = lane();
    let device = lane.device.name().to_owned();
    let halve = halving(&lane);
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let texture = lane.texture(PF, Shape { size: SIDE, ..ONE }, usage);
    let depth = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
    let from = buffer_with(&lane.device, &[7; 16]).unwrap();
    let to = buffer_with(&lane.device, &[0; 16]).unwrap();
    steps(&lane, &cleared(&texture, 0, 0, 1.0), &[]).unwrap();
    let before = taken(&lane);
    // A command buffer of a blit's copy where `blit`, then a pass that loads the target, halves
    // it and stores it, with `check`.
    let checked = |blit: bool, check: SubmissionCheck| {
        let command_buffer = lane.queue.command_buffer().unwrap();
        if blit {
            let mut encoder = command_buffer.blit_command_encoder().unwrap();
            encoder
                .copy_from_buffer_to_buffer(&from, 0, &to, 0, 16)
                .unwrap();
            encoder.end_encoding().unwrap();
        }
        frame_pass(&command_buffer, (&texture, 1), &halve, &depth, None);
        command_buffer.set_submission_check(check).unwrap();
        command_buffer
    };
    let failed_with = |command_buffer: &CommandBuffer| {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(
            command_buffer.status(),
            CommandBufferStatus::Error,
            "on {device}"
        );
        assert_eq!(
            command_buffer.error(),
            Some(CommandBufferError::Injected(CHECK_CODE)),
            "on {device}"
        );
    };
    // The target's texels, read while a command buffer of another queue is open, so that the
    // read's own commit, which a waiting target's switch would ride on, takes none.
    let other = lane.device.new_command_queue().unwrap();
    let held_read = || {
        let open = other.command_buffer().unwrap();
        let texels = read(&lane, &texture, 0, 0);
        drop(open);
        texels
    };
    let seen = Arc::new(AtomicUsize::new(0));

    // The check fails the pass's own submission: it runs nothing, the switch's copy included.
    let refused = checked(false, failing(Some(EncoderKind::Render), &seen));
    refused.commit().unwrap();
    failed_with(&refused);
    let scratch = scratched(&messages.take_notices());
    assert_eq!(taken(&lane), before, "a failed pass, on {device}");
    assert_eq!(held_read(), level_of(SIDE, 1.0), "on {device}");
    assert_eq!(taken(&lane), before, "a failed pass, on {device}");

    // The check fails the blit before the pass: the pass and the copy run, and nothing swaps.
    let partly = checked(true, failing(Some(EncoderKind::Blit), &seen));
    partly.commit().unwrap();
    failed_with(&partly);
    assert_eq!(held_read(), level_of(SIDE, 0.5), "on {device}");
    assert_eq!(read_buffer(&to, 0, 16).unwrap(), [0; 16], "on {device}");
    assert_eq!(
        taken(&lane),
        before,
        "a failed blit before the pass, on {device}"
    );

    // The checks pass: the switch is settled at the submission, and the next command buffer
    // made waits for it.
    let passed = checked(true, failing(None, &seen));
    let at = Instant::now();
    passed.commit().unwrap();
    let commit = at.elapsed();
    let at = Instant::now();
    let next = lane.queue.command_buffer().unwrap();
    let waited = at.elapsed();
    assert_eq!(
        taken(&lane),
        before + u64::from(scratch),
        "once its checks passed, on {device}"
    );
    passed.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(passed.error(), None, "on {device}");
    drop(next);
    assert_eq!(
        read(&lane, &texture, 0, 0),
        level_of(SIDE, 0.25),
        "on {device}"
    );
    assert_eq!(read_buffer(&to, 0, 16).unwrap(), [7; 16], "on {device}");

    // In place from then on, and once.
    steps(&lane, &loading(&texture, StoreAction::Store), &[&halve]).unwrap();
    assert_eq!(
        read(&lane, &texture, 0, 0),
        level_of(SIDE, 0.125),
        "on {device}"
    );
    assert_eq!(
        taken(&lane),
        before + u64::from(scratch),
        "once, on {device}"
    );
    // Each of Metal's submissions checked once: the pass; the blit and the pass, twice.
    assert_eq!(seen.load(Ordering::SeqCst), 5, "on {device}");
    eprintln!(
        "storage form on {device}: a failed check (the pass's, or a blit's before it) left the \
         target on its scratch; passing checks switched it at the submission (through a scratch \
         {scratch}): its commit took {commit:?}, the next command buffer waited {waited:?}"
    );
    drop(cache);
    quiet(lane, &messages);
}

/// The targets [`storage_form_costs`] switches with a check, and without; a side of each.
const COST_TARGETS: usize = 12;
const COST_SIDE: u64 = 2048;

/// The 50th and 90th percentiles of `values`, in microseconds.
fn p50_p90(mut values: Vec<f64>) -> (f64, f64) {
    values.sort_by(f64::total_cmp);
    let at = |p: usize| values[(values.len() * p / 100).min(values.len() - 1)];
    (at(50), at(90))
}

/// The switch's costs, printed (the module docs): `COST_TARGETS` `COST_SIDE`² `R32Float` targets
/// (16 MiB each, the bytes of a 1920x1080 `RGBA16Float` one) switched at their first fetching
/// pass's commit, with a check that passes and without; the next command buffer's wait after
/// the checked commit; and one fetching pass's GPU time through a scratch (an untracked
/// target's, which keeps it) and in place (a switched one's).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all, in release"]
fn storage_form_costs() {
    let (messages, cache, lane) = lane_with(|options| options.layers.clear());
    let device = lane.device.name().to_owned();
    let halve = halving(&lane);
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let shape = Shape {
        size: COST_SIDE,
        ..ONE
    };
    let depth = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
    let gpu_us = |command_buffer: &CommandBuffer| {
        let start = command_buffer.gpu_start_time().unwrap();
        let end = command_buffer.gpu_end_time().unwrap();
        (end - start) * 1e6
    };
    // One fetching pass of `texture` (cleared first where `clear`), committed with `check`:
    // the commit's time, the next command buffer's wait, and its GPU time.
    let fetching = |texture: &Texture, check: Option<SubmissionCheck>, clear: bool| {
        if clear {
            steps(&lane, &cleared(texture, 0, 0, 1.0), &[]).unwrap();
        }
        let command_buffer = lane.queue.command_buffer().unwrap();
        frame_pass(&command_buffer, (texture, 1), &halve, &depth, None);
        if let Some(check) = check {
            command_buffer.set_submission_check(check).unwrap();
        }
        let at = Instant::now();
        command_buffer.commit().unwrap();
        let commit = at.elapsed().as_secs_f64() * 1e6;
        let at = Instant::now();
        let next = lane.queue.command_buffer().unwrap();
        let waited = at.elapsed().as_secs_f64() * 1e6;
        drop(next);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "on {device}");
        (commit, waited, gpu_us(&command_buffer))
    };
    let seen = Arc::new(AtomicUsize::new(0));
    let before = taken(&lane);
    let (mut checked, mut waited, mut unchecked, mut kept) = (vec![], vec![], vec![], vec![]);
    let mut in_place = vec![];
    let mut scratch_gpu = vec![];
    for _ in 0..COST_TARGETS {
        let target = lane.texture(PF, shape, usage);
        let (commit, wait, _) = fetching(&target, Some(failing(None, &seen)), true);
        checked.push(commit);
        waited.push(wait);
        let target = lane.texture(PF, shape, usage);
        let (commit, wait, _) = fetching(&target, None, true);
        unchecked.push(commit);
        kept.push(wait);
        let (commit, _, gpu) = fetching(&target, None, false);
        in_place.push((commit, gpu));
    }
    let switched = taken(&lane) - before;
    let notices = messages.take_notices();
    if scratched(&notices) {
        assert_eq!(switched, 2 * COST_TARGETS as u64, "on {device}");
        let untracked = {
            let mut descriptor = TextureDescriptor::texture_2d(PF, COST_SIDE, COST_SIDE, false);
            descriptor.usage = usage;
            descriptor.resource_options = ResourceOptions::new(
                StorageMode::Private,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Untracked,
            );
            lane.device.new_texture(&descriptor).unwrap()
        };
        steps(&lane, &cleared(&untracked, 0, 0, 1.0), &[]).unwrap();
        for _ in 0..COST_TARGETS {
            scratch_gpu.push(fetching(&untracked, None, false).2);
        }
    }
    let (checked, checked_90) = p50_p90(checked);
    let (waited, waited_90) = p50_p90(waited);
    let (unchecked, unchecked_90) = p50_p90(unchecked);
    let (kept, kept_90) = p50_p90(kept);
    let (plain, plain_90) = p50_p90(in_place.iter().map(|&(commit, _)| commit).collect());
    let (gpu, gpu_90) = p50_p90(in_place.iter().map(|&(_, gpu)| gpu).collect());
    eprintln!(
        "STORAGE-FORM-COST {device}: {COST_TARGETS} targets {COST_SIDE}x{COST_SIDE} R32Float, \
         {switched} switched; checked: commit {checked:.1} (p90 {checked_90:.1}) us, the next \
         command buffer waited {waited:.1} (p90 {waited_90:.1}) us; unchecked: commit \
         {unchecked:.1} (p90 {unchecked_90:.1}) us, the next command buffer {kept:.1} (p90 \
         {kept_90:.1}) us; a fetching pass in place after: commit {plain:.1} (p90 \
         {plain_90:.1}) us, gpu {gpu:.1} (p90 {gpu_90:.1}) us"
    );
    if !scratch_gpu.is_empty() {
        let (through, through_90) = p50_p90(scratch_gpu);
        eprintln!(
            "STORAGE-FORM-COST {device}: a fetching pass through a scratch (an untracked \
             target's): gpu {through:.1} (p90 {through_90:.1}) us"
        );
    }
    drop(cache);
    drop(lane);
    drop(messages);
}
