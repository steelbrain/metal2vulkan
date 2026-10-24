//! Detaching a no-copy buffer or heap from its host pages (`detach_backing`), on the device the cfg
//! selects, under the validation layer. The rule is Metal's as the consumer measured it on the M3
//! Max (macOS 26.7, the rebind probe's tables 1 and 3):
//!
//! - **Pending work** (table 1: committed before the detach, first submitted after). A command
//!   buffer whose command names the buffer (bound and read, case 1; bound and walked, case 2; a
//!   blit's operand, case 5) or whose encoder made it resident (`use_resources`, case 3;
//!   `use_heap`, case 3h) fails `CommandBufferError::DetachedBacking`, Metal's code 8, and runs
//!   nothing; its signal fires, its completed handler runs, and the queue runs on. A resident
//!   buffer counts whether or not a command reaches it, once the encoder encoded a command after
//!   making it resident (the M3 Max under macOS 27.0.1). One that
//!   reaches the buffer unnamed alone (case 4) reads 0 there and completes. No degradation is
//!   announced: the failure is Metal's.
//! - **Submitted work** (table 3, and the M3 Max under macOS 27.0.1). Metal holds what a
//!   submission uses from its submission until its work completed, which for the last submission
//!   is as the command buffer's completion is processed, before its completed handlers run: a
//!   buffer's detach meanwhile blocks until that work completed, then answers `false` and detaches
//!   nothing; a heap's answers `false` at once; a detach in a completed handler detaches, as does
//!   one once `wait_until_completed` returned. Metal's answer to a detach made at the very start of
//!   a completed handler is a race of its own release against the handler and is nothing to pin
//!   (`detach.rs`'s module docs hold the measurement); its answer once the use has retired is
//!   always `true`, which is what this lane holds. A command buffer's later submission behind an
//!   event wait holds nothing until it is submitted: a detach while it waits detaches at once,
//!   though the command buffer's earlier work ran or still runs, and that submission then fails
//!   with code 8.
//! - **The deallocator** of the pages a detach lets go of runs at the buffer's next rebind,
//!   inside the call, on its thread, or at its release, never at the detach (the M3 Max under
//!   macOS 27.0.1, a `newBufferWithBytesNoCopy:` buffer and one of `newBufferWithDescriptor:`,
//!   each with and without GPU use before the detach). A keep-alive goes after the deallocator,
//!   once the import is freed.
//! - **Encoded while detached.** A command that names a detached buffer records nothing, nor does
//!   any after it in its encoder (blit, dispatch, draw), and its command buffer fails
//!   `DetachedBacking` and runs nothing; where a rebind gave the buffer pages again before its
//!   first submission, the rebind encodes a compute or blit encoder again over them, and the
//!   command buffer completes over the new pages, as Metal's (`EncodedUseRebound` before). The
//!   layer reports nothing: no Vulkan command takes the detached buffer's stand-in.
//! - **Racing a commit.** A detach on one thread while another commits a command buffer whose
//!   command names the buffer runs the old pages' deallocator in neither call, wherever the
//!   device's recorder replays the commands, and the buffer's release runs it, on the releasing
//!   thread; nothing is announced.
//! - **No memory.** A detach places no memory of the bytes it detaches: once the device's
//!   stand-in is made, detaching a large heap, and a large buffer, allocates no device memory.
//! - **Textures**: `detach_textures.rs`.

use super::degrade_notices::{
    assert_made_once, assert_once_allowing, on_device as on_device_noticed,
};
use super::host_memory::{gpu_read, on_device, recording, Mapping, Seen};
use super::no_copy_heaps::shared;
use super::rebind_reach::{
    buffer_len, buffer_over, heap_member, kept_member_over, member_over, node, pages,
    rebind_heap_to, rebind_to, walk, Outputs, Uses, Walked, A, B, POINTER_WALKER,
};
use super::shared_kinds;
use super::{buffer_with, public_case, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, ResourceUsage, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferError, CommandBufferStatus, CommandQueue,
    ComputePipelineState, Degradation, Device, Error, EventRef, Heap, HeapDescriptor, HeapType,
    HostMemory, Invalid, LoadAction, PipelineOptions, Problem, RebindProblem, Region,
    RenderPassDescriptor, ResourceOptions, SharedEvent, Size, StoreAction, TextureDescriptor,
};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long a detach that must wait is given to return early, wrongly.
const EARLY: Duration = Duration::from_millis(100);

/// The bytes of the large heap and buffer the memory row detaches.
const LARGE: usize = 64 << 20;

/// `command_buffer` failed `DetachedBacking`, Metal's code 8.
fn detached_backing(command_buffer: &CommandBuffer, name: &str) {
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Error,
        "{name}"
    );
    assert_eq!(
        command_buffer.error(),
        Some(CommandBufferError::DetachedBacking),
        "{name}"
    );
    assert_eq!(CommandBufferError::DetachedBacking.metal_code(), 8);
}

/// A walk failed `DetachedBacking`: its signal fired, its handler ran, and a command buffer
/// behind it on `queue` runs.
fn failed(device: &Device, queue: &CommandQueue, walked: Walked, name: &str) {
    detached_backing(&walked.command_buffer, name);
    assert_eq!(walked.signaled, 1, "{name}: its signal");
    assert!(walked.handled, "{name}: its handler");
    let after = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let behind = queue.command_buffer().unwrap();
    let mut blit = behind.blit_command_encoder().unwrap();
    blit.fill_buffer(&after, 0..16, 0x5a).unwrap();
    blit.end_encoding().unwrap();
    behind.commit().unwrap();
    behind.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(behind.error(), None, "{name}: the command buffer behind it");
    assert_eq!(read_buffer(&after, 0, 16).unwrap(), [0x5a; 16]);
}

/// Detach `buffer`, which must detach at once.
fn detach(buffer: &Buffer) {
    assert_eq!(buffer.detach_backing(), Ok(true));
}

/// Detach `heap`, which must detach at once.
fn detach_heap(heap: &Heap) {
    assert_eq!(heap.detach_backing(), Ok(true));
}

/// Table 1, cases 1, 3, 3h and 5 (detach): a pending command buffer whose command names a
/// buffer detached since (bound and read, a blit's operand), or whose encoder made it resident,
/// fails `DetachedBacking` (code 8) at its first submission and runs nothing; its signal fires,
/// its handler runs, the queue runs on, and nothing is announced. Buffer and heap forms; each
/// given pages again by a rebind. (Case 2 binds the member as the walker's output, which a
/// detached buffer's `None` contents cannot be read back from.)
#[test]
fn table_1_a_pending_command_buffer_holding_a_detached_buffer_fails_with_code_8() {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let Some((member, old)) = member_over(device, A) else {
            return;
        };
        let held = |member: &Buffer, uses: Uses<'_>, between: &dyn Fn()| {
            walk(device, &queue, &walker, member, uses, true, between)
        };
        let blit = held(&member, Uses::Blit, &|| detach(&member));
        failed(device, &queue, blit, "5: a blit's operand");
        // The old pages' deallocator waits for the rebind, which runs it inside the call.
        assert_eq!(old.runs.load(Ordering::SeqCst), 0, "ran at the detach");
        rebind_to(&member, A);
        ran_here(&old, "the rebind after a detach");
        let resident = held(&member, Uses::Resident, &|| detach(&member));
        failed(device, &queue, resident, "3: resident");
        rebind_to(&member, A);
        let steps_seen = Arc::new(Seen::default());
        let steps = buffer_over(
            device,
            pages(buffer_len(), 0, &2u32.to_le_bytes()),
            &steps_seen,
        )
        .unwrap();
        let plain = buffer_with(device, &node(A)).unwrap();
        let bound = held(&plain, Uses::Steps(&steps), &|| detach(&steps));
        failed(device, &queue, bound, "1: bound and read");

        let Some((heap, placed)) = heap_member(device) else {
            return;
        };
        let cases = [
            ("heap 5", &placed, Uses::Blit),
            ("heap 3", &placed, Uses::Resident),
            ("3h", &placed, Uses::Heap(&heap)),
            ("heap 1", &plain, Uses::Steps(&placed)),
        ];
        for (name, member, uses) in cases {
            let walked = held(member, uses, &|| detach_heap(&heap));
            failed(device, &queue, walked, name);
            rebind_heap_to(&heap, A);
        }
        let notices = assert_made_once(messages, &[], &shared_kinds(device));
        assert!(
            !notices.contains_key(&Degradation::EncodedUseRebound),
            "announced EncodedUseRebound for a detach"
        );
        eprintln!("DETACH table 1: named and resident uses fail with code 8, buffer and heap");
    });
}

/// Table 1, case 4 (detach): a pending command buffer that reaches a detached buffer unnamed
/// alone reads 0 there and completes with no error. Buffer and heap forms.
#[test]
fn table_1_a_pending_command_buffer_reaching_a_detached_buffer_unnamed_reads_zero() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let outputs = Outputs::new(device, &queue, &walker);
        let held = |member: &Buffer, between: &dyn Fn()| {
            let walked = walk(
                device,
                &queue,
                &walker,
                member,
                Uses::Unnamed,
                true,
                between,
            );
            assert_eq!(walked.command_buffer.error(), None);
            walked.produced
        };
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        gpu_read(&queue, &member, 0..16);
        assert!(held(&member, &|| {}) == outputs.a, "4: control");
        let detached = held(&member, &|| detach(&member));
        assert!(detached == outputs.zero, "4: detached");

        let Some((heap, placed)) = heap_member(device) else {
            return;
        };
        let other = heap.new_buffer_at_offset(256, shared(), 0).unwrap();
        gpu_read(&queue, &other, 0..4);
        assert!(held(&placed, &|| {}) == outputs.a, "heap 4: control");
        let detached = held(&placed, &|| detach_heap(&heap));
        assert!(detached == outputs.zero, "heap 4: detached");
        eprintln!("DETACH table 1 unnamed: 0 once detached, buffer and heap");
    });
}

/// A command that names a detached buffer records nothing, nor does any after it in its
/// encoder or its command buffer's later encoders: a blit's copy (the fills around it unrun) and
/// a dispatch, each failing `DetachedBacking`; the old pages' deallocator ran in the detach,
/// which nothing held, and `contents` answers `None`. Encoded while detached and given pages
/// again before its first submission, a blit encoder is encoded again by the rebind (It failed
/// `EncodedUseRebound` before) and its copy reads the new pages, with no notice. A command buffer
/// encoded after the rebind reads the new pages. A buffer of the runtime's memory, or a heap's
/// buffer, is not detached (`Invalid`, `NotHostMemory`).
#[test]
fn a_command_naming_a_detached_buffer_records_nothing_and_its_command_buffer_runs_nothing() {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let Some((member, old)) = member_over(device, A) else {
            return;
        };
        gpu_read(&queue, &member, 0..16);
        assert!(member.contents().is_some());
        detach(&member);
        assert_eq!(old.runs.load(Ordering::SeqCst), 0, "ran at the detach");
        assert_eq!(member.contents(), None, "a detached buffer's contents");
        detach(&member);

        let before = buffer_with(device, &[0; 16]).unwrap();
        let copied = buffer_with(device, &[0x5a; 16]).unwrap();
        let after = buffer_with(device, &[0; 16]).unwrap();
        let later = buffer_with(device, &[0; 16]).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&before, 0..16, 0x11).unwrap();
        blit.copy_from_buffer_to_buffer(&member, 0, &copied, 0, 16)
            .unwrap();
        blit.fill_buffer(&after, 0..16, 0x22).unwrap();
        blit.end_encoding().unwrap();
        let mut next = command_buffer.blit_command_encoder().unwrap();
        next.fill_buffer(&later, 0..16, 0x33).unwrap();
        next.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        detached_backing(&command_buffer, "a blit");
        for (name, buffer, held) in [
            ("the fill before", &before, [0; 16]),
            ("the copy", &copied, [0x5a; 16]),
            ("the fill after", &after, [0; 16]),
            ("the later encoder", &later, [0; 16]),
        ] {
            assert_eq!(read_buffer(buffer, 0, 16).unwrap(), held, "{name} ran");
        }

        let plain = buffer_with(device, &node(A)).unwrap();
        let walked = walk(
            device,
            &queue,
            &walker,
            &plain,
            Uses::Steps(&member),
            false,
            || {},
        );
        detached_backing(&walked.command_buffer, "a dispatch");

        // Encoded while detached, given pages before its first submission.
        let event = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&member, 0, &copied, 0, 16)
            .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        let notices = assert_made_once(messages, &[], &shared_kinds(device));
        assert!(!notices.contains_key(&Degradation::EncodedUseRebound));
        let reencoded = device.work_counts().encoders_reencoded;
        rebind_to(&member, B);
        ran_here(&old, "the rebind after the detach");
        // The rebind encodes the blit encoder again over the new pages before its first
        // submission: the copy reads them, as Metal's does.
        assert!(
            device.work_counts().encoders_reencoded > reencoded,
            "the rebind encoded the blit encoder again"
        );
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
        assert_eq!(read_buffer(&copied, 0, 16).unwrap(), node(B), "the copy");
        assert_made_once(messages, &[], &shared_kinds(device));
        assert!(member.contents().is_some());
        assert_eq!(gpu_read(&queue, &member, 0..16), node(B), "attached again");

        // Refused: no host pages of its own.
        let refused = Err(Error::Invalid(Invalid::new(
            "detach_backing",
            Problem::Rebind(RebindProblem::NotHostMemory),
        )));
        let ordinary = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        assert_eq!(ordinary.detach_backing(), refused);
        let Some((heap, placed)) = heap_member(device) else {
            return;
        };
        assert_eq!(placed.detach_backing(), refused);
        let descriptor = HeapDescriptor::new(HeapType::Placement, 1 << 16, shared());
        let own = device.new_heap(&descriptor).unwrap();
        assert_eq!(own.detach_backing(), refused);
        drop(heap);
        eprintln!(
            "DETACH encoded: nothing recorded runs; given pages again, encoded again over them"
        );
    });
}

/// A render pass whose draw takes a detached vertex buffer records none of its draws: its
/// command buffer fails `DetachedBacking`, its target keeps its texels (the pass's clear unrun),
/// and the layer reports nothing. The fixture's pipeline (`rt6_restart.ll`) reads the corners
/// from vertex buffer 0.
#[test]
fn a_render_pass_drawing_from_a_detached_buffer_runs_nothing() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let bitcode = llvm_assemble(&engine_dir().join("rt6_restart.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = |name| {
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor()
        };
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function("rt6_restart_vertex")));
        descriptor.fragment = Some(StageSource::Function(function("rt6_restart_count")));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let side = 8;
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, side, side, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::shared();
        let target = device.new_texture(&target).unwrap();
        let old = vec![0x11; (side * side * 4) as usize];
        let region = Region::new_2d(0, 0, side, side);
        target
            .replace_region(region, 0, 0, &old, side * 4, 0)
            .unwrap();
        let corners: Vec<u8> = [-1.0f32, -1.0, 3.0, -1.0, -1.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let seen = Arc::new(Seen::default());
        let Some(detached) = buffer_over(device, pages(buffer_len(), 0, &corners), &seen) else {
            return;
        };
        let plain = buffer_with(device, &corners).unwrap();
        detach(&detached);
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(0.5, 0.0, 0.0, 1.0);
        color.store_action = StoreAction::Store;
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&pipeline).unwrap();
        encoder.set_vertex_buffer(Some(&plain), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        encoder.set_vertex_buffer(Some(&detached), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        encoder.set_vertex_buffer(Some(&plain), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        detached_backing(&command_buffer, "a draw");
        let mut texels = vec![0; old.len()];
        target
            .get_bytes(&mut texels, side * 4, 0, region, 0, 0)
            .unwrap();
        assert!(texels == old, "the pass ran");
        eprintln!("DETACH render: a draw from a detached buffer ran nothing");
    });
}

/// `seen`'s deallocator ran once, on this thread, before the call that should run it returned.
fn ran_here(seen: &Seen, call: &str) {
    assert_eq!(
        seen.runs.load(Ordering::SeqCst),
        1,
        "{call}: ran once, in the call"
    );
    assert_eq!(
        *seen.thread.lock().unwrap(),
        Some(std::thread::current().id()),
        "{call}: on the calling thread"
    );
}

/// How long a spin runs when nothing ends it early: the duration [`Spin::new`] calibrates its cap
/// to, at the device's own measured rate.
///
/// **A fixed cap cannot hold both ends of this test.** `release` ends a spin at its next load only
/// where the device shows a running kernel the host's store; where it does not
/// (`first_use::release` says so), the cap is the only thing that ends the spin, so the cap's
/// duration is what every assertion here actually rests on. It has to outlast [`EARLY`], or
/// `waits_then_refuses` sees the detach return and reads it as a detach that did not wait; and it
/// has to finish inside the device's hang budget, or the ring is reset, the context is lost
/// "guilty of a hard recovery", and `Spinning::finish` reads `DeviceLost`.
///
/// The constant it replaced, `1 << 25`, was chosen as seconds on the M3 Max (about 135 ns a load)
/// and held neither end anywhere else. Measured on RADV by the lane that diagnosed this: `1 << 25`
/// is about 2 s and fails 5 of 5 runs, each adding a `ring gfx_0.0.0` timeout and a ring reset;
/// `1 << 20` ends inside [`EARLY`], so the other assertion fails instead; `1 << 23`, about half a
/// second, passes. So half a second is the value the failing device itself measured as holding
/// both ends: five times [`EARLY`] on one side, a quarter of the observed hang on the other. A
/// device four times slower than its own fastest measured run still lands inside that gap.
const SPIN: Duration = Duration::from_millis(500);

/// The hang budget the calibration must stay inside, for [`the_spin_cap_holds_both_ends`] to check
/// against. It is the shortest reset observed on this fleet (amdgpu's `ring gfx_0.0.0` timeout at
/// about 2 s), not a limit any device publishes, so it is a floor for the check and never a claim
/// about a device.
const HANG: Duration = Duration::from_secs(2);

/// The cap, in loads, that runs for [`SPIN`] at `rate` loads per second.
///
/// Its own function so [`the_spin_cap_holds_both_ends`] can hold both ends of the calibration
/// over a range of rates with no device: the arithmetic is what decides whether a measured rate
/// becomes a cap that outlasts [`EARLY`] and finishes inside [`HANG`].
fn cap_for(rate: f64) -> u32 {
    (rate * SPIN.as_secs_f64()).clamp(1.0, f64::from(u32::MAX)) as u32
}

/// The calibration holds **both** ends, and the arithmetic is exact over the fleet's range.
///
/// A cap that only fixes one end moves the failure instead of closing it, so the margins are
/// checked rather than assumed. `cap_for(rate) / rate` is [`SPIN`] by construction, so asserting
/// that would prove nothing; what can be wrong is the margin [`SPIN`] leaves at each end and the
/// clamp at the edges of the range.
#[test]
fn the_spin_cap_holds_both_ends() {
    // The lower end, with room: the spin has to be running when `waits_then_refuses` looks, after
    // it slept EARLY, and the spin started before that sleep began.
    assert!(
        SPIN >= EARLY * 2,
        "a spin of {SPIN:?} leaves no room over the {EARLY:?} the test sleeps before it looks"
    );
    // The upper end, with room for a run slower than the fastest rate the calibration measured.
    // The fastest run is the kernel's own rate, so a loaded device runs the same cap for longer,
    // and it is that longer run that has to stay inside the hang budget.
    const SLOWDOWN: u32 = 4;
    assert!(
        SPIN * SLOWDOWN <= HANG,
        "a spin of {SPIN:?} run {SLOWDOWN} times slower than its measured rate is {:?}, past the \
         {HANG:?} budget",
        SPIN * SLOWDOWN
    );
    // The scaling is exact across the fleet: from a CPU ICD's rate to far past any GPU measured
    // here, the cap is neither clamped to one load nor to u32::MAX, either of which would silently
    // stop the cap meaning SPIN. 16.8e6 is RADV's measured rate (1 << 25 loads in about 2 s).
    for rate in [1.0e4, 1.0e5, 1.0e6, 16.8e6, 1.0e8, 1.0e9, 8.0e9] {
        let cap = cap_for(rate);
        assert!(cap > 1, "a rate of {rate} loads/s clamps the cap to {cap}");
        assert!(
            cap < u32::MAX,
            "a rate of {rate} loads/s clamps the cap to u32::MAX"
        );
        let runs = Duration::from_secs_f64(f64::from(cap) / rate);
        assert!(
            runs > EARLY && runs * SLOWDOWN <= HANG,
            "a rate of {rate} loads/s gives {cap} loads, which runs {runs:?}"
        );
    }
    // A rate so low the clamp would bite is below anything that can run this kernel at all: one
    // load every half second. Stated so the floor is read rather than discovered.
    assert_eq!(cap_for(2.0), 1);
}

/// The spin kernel and the cap calibrated for the device it will run on.
///
/// [`first_use::Busy`]'s shape, for this lane's own kernel rather than a second mechanism: the cap
/// grows sixteenfold until a run takes [`first_use::MEASURED`] at the fastest rate measured so
/// far, and the cap that ships is [`cap_for`] of that rate. The fastest run is the kernel's own
/// rate, because the GPU's other work can only slow a run.
struct Spin {
    state: ComputePipelineState,
    /// The loads the spin makes at most.
    cap: u32,
}

impl Spin {
    /// The owned fixture's kernel that loads a word no one raises until its cap
    /// (`kernel_spin_on_an_unraised_word`): its GPU work runs until the host raises the word. Run
    /// once on `queue`, its word raised from the start, so its `VkPipeline` is made: a dispatch
    /// whose pipeline is still being made records at a host step, which its submission waits for.
    /// That first run is also the one the calibration does not measure, for
    /// [`first_use::Busy`]'s reason: it holds the device's first use of the pipeline and of a
    /// command buffer, which on a loaded host take past [`first_use::MEASURED`] by themselves.
    fn new(device: &Device, queue: &CommandQueue) -> Spin {
        let ll = engine_dir().join("kernel_spin_on_an_unraised_word.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = library
            .new_function_with_name("spin_on_an_unraised_word")
            .unwrap();
        let state = device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(StageSource::Function(
                    function.function_descriptor(),
                )),
                &PipelineOptions::default(),
            )
            .unwrap();
        let raised = buffer_with(device, &1u32.to_le_bytes()).unwrap();
        let loads = buffer_with(device, &[0; 4]).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&state).unwrap();
        encoder.set_buffer(Some(&raised), 0, 0).unwrap();
        encoder.set_buffer(Some(&loads), 0, 1).unwrap();
        encoder.set_bytes(&cap_for(0.0).to_le_bytes(), 2).unwrap();
        encoder
            .dispatch_threadgroups(Size::new(1, 1, 1), Size::new(1, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "the spin, raised");
        assert_eq!(
            read_buffer(&loads, 0, 4).unwrap(),
            [0; 4],
            "the spin, raised"
        );
        let mut spin = Spin {
            state,
            cap: 1 << 12,
        };
        // The fastest rate of every measured run, in loads per second of its time on the GPU, and
        // each run's loads and times for the printed line.
        let mut rate = 0.0_f64;
        let mut runs = Vec::new();
        let at_rate =
            |cap: u32, rate: f64| f64::from(cap) / rate >= super::first_use::MEASURED.as_secs_f64();
        loop {
            for _ in 0..super::first_use::RUNS {
                let (took, wall) = spin.measure(device, queue);
                rate = rate.max(f64::from(spin.cap) / took.as_secs_f64());
                runs.push((spin.cap, took, wall));
                if !at_rate(spin.cap, rate) {
                    break;
                }
            }
            if at_rate(spin.cap, rate) || spin.cap >= 1 << 28 {
                break;
            }
            spin.cap <<= 4;
        }
        spin.cap = cap_for(rate);
        let runs: Vec<String> = (runs.iter())
            .map(|(cap, took, wall)| format!("{cap} in {took:.2?} ({wall:.2?})"))
            .collect();
        println!(
            "DETACH SPIN-CAP on {}: {} loads for about {SPIN:?}, at the fastest rate of the runs \
             of loads in GPU time (from commit to completion): {}",
            device.name(),
            spin.cap,
            runs.join(", ")
        );
        spin
    }

    /// Run one spin of the current cap over a word nothing raises, so the cap is what ends it: the
    /// time it took on the GPU, and from its commit to its completion.
    fn measure(&self, device: &Device, queue: &CommandQueue) -> (Duration, Duration) {
        let at = Instant::now();
        let unraised = buffer_with(device, &[0; 4]).unwrap();
        let loads = buffer_with(device, &[0; 4]).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&self.state).unwrap();
        encoder.set_buffer(Some(&unraised), 0, 0).unwrap();
        encoder.set_buffer(Some(&loads), 0, 1).unwrap();
        encoder.set_bytes(&self.cap.to_le_bytes(), 2).unwrap();
        encoder
            .dispatch_threadgroups(Size::new(1, 1, 1), Size::new(1, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let wall = at.elapsed();
        assert_eq!(command_buffer.error(), None, "the spin's calibration");
        (super::first_use::gpu_took(&command_buffer, wall), wall)
    }
}

/// A command buffer whose first batch spins over a buffer until [`Spinning::release`] (the
/// buffer bound as the kernel's output, or made resident), and whose next batch, behind an event
/// wait, fills a buffer of its own; committed, its first batch submitted.
struct Spinning {
    command_buffer: CommandBuffer,
    flag: Buffer,
    output: Buffer,
    gate: SharedEvent,
}

impl Spinning {
    fn new(
        device: &Device,
        queue: &CommandQueue,
        spin: &Spin,
        held: &Buffer,
        resident: bool,
    ) -> Spinning {
        let flag = buffer_with(device, &[0; 16]).unwrap();
        let output = match resident {
            true => buffer_with(device, &[0; 16]).unwrap(),
            false => held.clone(),
        };
        let gate = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&spin.state).unwrap();
        encoder.set_buffer(Some(&flag), 0, 0).unwrap();
        encoder.set_buffer(Some(&output), 0, 1).unwrap();
        encoder.set_bytes(&spin.cap.to_le_bytes(), 2).unwrap();
        if resident {
            encoder
                .use_resource(held.as_resource(), ResourceUsage::READ)
                .unwrap();
        }
        encoder
            .dispatch_threadgroups(Size::new(1, 1, 1), Size::new(1, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&gate), 1)
            .unwrap();
        let filled = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&filled, 0..16, 0x3c).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        let deadline = Instant::now() + WAIT;
        while command_buffer.kernel_start_time() == 0.0 {
            assert!(
                Instant::now() < deadline,
                "the first batch was never submitted"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        Spinning {
            command_buffer,
            flag,
            output,
            gate,
        }
    }

    /// Raise the word: an atomic store into the buffer the kernel's atomic loads read. The spin
    /// ends.
    fn release(&self) {
        let contents = self.flag.contents().unwrap();
        // SAFETY: the buffer's first word, `Shared` and 4-aligned, which the host and the
        // kernel's atomics alone touch while the kernel runs.
        let flag = unsafe { AtomicU32::from_ptr(contents.as_ptr().cast::<u32>()) };
        flag.store(1, Ordering::SeqCst);
    }

    /// Let it run to its end.
    fn finish(&self) {
        self.release();
        self.gate.set_signaled_value(1);
        self.command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(self.command_buffer.error(), None);
    }
}

impl Drop for Spinning {
    /// The GPU is never left spinning, whatever failed.
    fn drop(&mut self) {
        self.release();
        self.gate.set_signaled_value(1);
    }
}

/// `detach`, called on another thread while `spinning`'s first batch spins: it must not return
/// before the spin ends, and answers `false` once it ended, the command buffer still waiting for
/// its event: the hold is the submission's, not the command buffer's.
fn waits_then_refuses(spinning: &Spinning, detach: impl FnOnce() -> Result<bool, Error> + Send) {
    let returned = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let detaching = scope.spawn(|| {
            let answer = detach();
            returned.store(true, Ordering::SeqCst);
            answer
        });
        std::thread::sleep(EARLY);
        if returned.load(Ordering::SeqCst) {
            spinning.release();
            let answer = detaching.join().unwrap();
            panic!(
                "the detach did not wait for the submitted work: {answer:?}, the command buffer \
                 {:?}, the spin's output {:?}",
                spinning.command_buffer.status(),
                read_buffer(&spinning.output, 0, 4)
            );
        }
        spinning.release();
        assert_eq!(detaching.join().unwrap(), Ok(false));
        assert_ne!(
            spinning.command_buffer.status(),
            CommandBufferStatus::Completed,
            "the detach waited for the command buffer, not for the submission"
        );
    });
}

/// Table 3 (work submitted and not yet complete), as Metal answers it on the M3 Max (macOS
/// 27.0.1): a buffer a running submission's command named, or its encoder made resident (the spin
/// reaching no buffer through an address), is detached only once that work completed: the detach blocks until then and answers `false`,
/// detaching nothing, though the command buffer still waits for an event behind it. A heap one of
/// whose buffers that work named answers `false` at once. A buffer it does not take is detached at
/// once.
#[test]
fn table_3_a_detach_waits_for_running_work_then_refuses() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let spin = Spin::new(device, &queue);
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        let named = Spinning::new(device, &queue, &spin, &member, false);
        // A buffer it does not take: at once.
        let (unrelated, _) = member_over(device, B).unwrap();
        detach(&unrelated);
        waits_then_refuses(&named, || member.detach_backing());
        named.finish();
        assert!(member.contents().is_some(), "detached though refused");
        // Once its command buffer completed and the wait returned, it detaches.
        detach(&member);
        rebind_to(&member, A);

        // Resident: its encoder made the buffer resident.
        let resident = Spinning::new(device, &queue, &spin, &member, true);
        waits_then_refuses(&resident, || member.detach_backing());
        resident.finish();
        detach(&member);

        // A heap: at once.
        let Some((heap, placed)) = heap_member(device) else {
            return;
        };
        let in_heap = Spinning::new(device, &queue, &spin, &placed, false);
        assert_eq!(heap.detach_backing(), Ok(false));
        assert_ne!(
            in_heap.command_buffer.status(),
            CommandBufferStatus::Completed,
            "the heap's detach waited"
        );
        in_heap.finish();
        detach_heap(&heap);
        eprintln!(
            "DETACH table 3: a buffer waits for the running submission then refuses, a heap \
             refuses at once"
        );
    });
}

/// A command buffer of `queue`, behind a wait for `gate` where it is given, whose compute encoder
/// makes `resident` resident and runs the spin over a raised word, which reaches no buffer through
/// an address: committed. Its spin writes 0 loads into the returned buffer, which holds `0xff`
/// bytes until then.
fn resident_unreached(
    device: &Device,
    queue: &CommandQueue,
    spin: &Spin,
    resident: &Buffer,
    gate: Option<&SharedEvent>,
) -> (CommandBuffer, Buffer) {
    let raised = buffer_with(device, &1u32.to_le_bytes()).unwrap();
    let loads = buffer_with(device, &[0xff; 4]).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    if let Some(gate) = gate {
        command_buffer
            .encode_wait_for_event(EventRef::from(gate), 1)
            .unwrap();
    }
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder
        .use_resource(resident.as_resource(), ResourceUsage::READ)
        .unwrap();
    encoder.set_compute_pipeline_state(&spin.state).unwrap();
    encoder.set_buffer(Some(&raised), 0, 0).unwrap();
    encoder.set_buffer(Some(&loads), 0, 1).unwrap();
    encoder.set_bytes(&spin.cap.to_le_bytes(), 2).unwrap();
    encoder
        .dispatch_threadgroups(Size::new(1, 1, 1), Size::new(1, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    (command_buffer, loads)
}

/// An encoder's resident buffer that no command reaches, as Metal answers it on the M3 Max (macOS
/// 27.0.1): held and judged as a reached one. Detached before the command buffer is encoded, or
/// after its commit and before its first submission (the detach answering `true` at once), its
/// submission fails `DetachedBacking` (code 8) and runs nothing.
#[test]
fn a_resident_buffer_no_command_reaches_fails_its_submission_once_detached() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let spin = Spin::new(device, &queue);
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        detach(&member);
        let (encoded, loads) = resident_unreached(device, &queue, &spin, &member, None);
        encoded.wait_until_completed_for(WAIT).unwrap();
        detached_backing(&encoded, "encoded detached");
        assert_eq!(
            read_buffer(&loads, 0, 4).unwrap(),
            [0xff; 4],
            "the spin ran"
        );
        rebind_to(&member, A);

        let gate = device.new_shared_event().unwrap();
        let (pending, loads) = resident_unreached(device, &queue, &spin, &member, Some(&gate));
        detach(&member);
        gate.set_signaled_value(1);
        pending.wait_until_completed_for(WAIT).unwrap();
        detached_backing(&pending, "detached before its submission");
        assert_eq!(
            read_buffer(&loads, 0, 4).unwrap(),
            [0xff; 4],
            "the spin ran"
        );
        eprintln!("DETACH resident unreached: held and failed with code 8 as a reached one");
    });
}

/// How long after its signal a split command buffer's first submission is taken as completed: the
/// Metal probe detached 50 ms after the commit, and a device signals an event as its work ends,
/// before the runtime's completion turn for that work runs.
const SETTLE: Duration = Duration::from_millis(50);

/// A command buffer of `queue` whose first submission runs `first` and signals an event, and
/// whose second, behind a wait for another event, runs `second`; committed, and returned once its
/// first submission's work completed (its signal seen, then [`SETTLE`]), with the event that lets
/// the second go.
fn split(
    device: &Device,
    queue: &CommandQueue,
    first: impl FnOnce(&CommandBuffer),
    second: impl FnOnce(&CommandBuffer),
) -> (CommandBuffer, SharedEvent) {
    let done = device.new_shared_event().unwrap();
    let gate = device.new_shared_event().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    first(&command_buffer);
    command_buffer
        .encode_signal_event(EventRef::from(&done), 1)
        .unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::from(&gate), 1)
        .unwrap();
    second(&command_buffer);
    command_buffer.commit().unwrap();
    let deadline = Instant::now() + WAIT;
    while done.signaled_value() < 1 {
        assert!(Instant::now() < deadline, "the first submission never ran");
        std::thread::sleep(Duration::from_millis(1));
    }
    std::thread::sleep(SETTLE);
    (command_buffer, gate)
}

/// A blit of `command_buffer` that fills the first 16 bytes of `buffer` with `value`.
fn fill(command_buffer: &CommandBuffer, buffer: &Buffer, value: u8) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..16, value).unwrap();
    blit.end_encoding().unwrap();
}

/// A detach between two of a command buffer's submissions, as Metal answers it on the M3 Max
/// (macOS 27.0.1): Metal holds what a submission uses only from its submission until its work
/// completed, so a detach while the command buffer waits for an event after a submission that used
/// the buffer, its work done, detaches at once, and the command buffer completes; one before a
/// later submission that uses it detaches at once too, and that submission fails with code 8, the
/// earlier one having run. Buffer and heap forms.
#[test]
fn a_detach_between_submissions_detaches_at_once_and_fails_only_the_later_submission() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        let Some((heap, placed)) = heap_member(device) else {
            return;
        };
        type Forms<'a> = [(
            &'a str,
            &'a Buffer,
            &'a dyn Fn() -> Result<bool, Error>,
            &'a dyn Fn(),
        ); 2];
        let forms: Forms<'_> = [
            ("buffer", &member, &|| member.detach_backing(), &|| {
                rebind_to(&member, A);
            }),
            ("heap", &placed, &|| heap.detach_backing(), &|| {
                rebind_heap_to(&heap, A)
            }),
        ];
        for (form, used, detach, reattach) in forms {
            // The use before the wait: its work done, the detach detaches at once.
            let other = buffer_with(device, &[0; 16]).unwrap();
            let (command_buffer, gate) = split(
                device,
                &queue,
                |cb| fill(cb, used, 0x5c),
                |cb| fill(cb, &other, 0x5d),
            );
            let started = Instant::now();
            assert_eq!(detach(), Ok(true), "{form}: used before the wait");
            assert!(started.elapsed() < EARLY, "{form}: the detach waited");
            gate.set_signaled_value(1);
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None, "{form}: used before the wait");
            assert_eq!(read_buffer(&other, 0, 16).unwrap(), [0x5d; 16]);
            reattach();

            // The use behind the wait: detached at once, and that submission fails.
            let other = buffer_with(device, &[0; 16]).unwrap();
            let (command_buffer, gate) = split(
                device,
                &queue,
                |cb| fill(cb, &other, 0x5c),
                |cb| fill(cb, used, 0x5d),
            );
            let started = Instant::now();
            assert_eq!(detach(), Ok(true), "{form}: used behind the wait");
            assert!(started.elapsed() < EARLY, "{form}: the detach waited");
            gate.set_signaled_value(1);
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            detached_backing(&command_buffer, form);
            assert_eq!(
                read_buffer(&other, 0, 16).unwrap(),
                [0x5c; 16],
                "{form}: the earlier submission ran"
            );
            reattach();
        }
        eprintln!(
            "DETACH split: a detach between submissions detaches at once; the later submission \
             fails with code 8"
        );
    });
}

/// A detach a completed handler calls: a buffer's or a heap's.
type Detaching = Arc<dyn Fn() -> Result<bool, Error> + Send + Sync>;

/// How long a completed handler waits before its detach, so the use it is asked about has retired
/// however the completion was scheduled: far past the 20 us past which Metal detached 684 of 684
/// times on the M3 Max under macOS 27.0.1.
const RETIRED: Duration = Duration::from_millis(20);

/// A detach in a completed handler of the command buffer that used the buffer detaches, as Metal's
/// does once the use has retired, and so does one once its `wait_until_completed` returned.
///
/// The handler waits [`RETIRED`] before it detaches, which is the region where Metal's answer is
/// deterministic. A detach made at the very start of a handler races Metal's own release of what
/// the command buffer used, both answers measured within microseconds of each other
/// (`detach.rs`'s module docs), so that leg is held only to answering at all: either answer is
/// Metal's and nothing may rest on which. Buffer and heap forms.
#[test]
fn a_detach_in_a_completed_handler_detaches() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        for retired in [false, true] {
            let Some((member, _)) = member_over(device, A) else {
                return;
            };
            let Some((heap, placed)) = heap_member(device) else {
                return;
            };
            let (buffer_answer, heap_answer) = (member.clone(), heap.clone());
            let forms: [(&str, &Buffer, Detaching); 2] = [
                (
                    "buffer",
                    &member,
                    Arc::new(move || buffer_answer.detach_backing()),
                ),
                (
                    "heap",
                    &placed,
                    Arc::new(move || heap_answer.detach_backing()),
                ),
            ];
            for (form, used, detach) in forms {
                let answered = Arc::new(std::sync::Mutex::new(None));
                let command_buffer = queue.command_buffer().unwrap();
                let (in_handler, seen) = (Arc::clone(&detach), Arc::clone(&answered));
                command_buffer
                    .add_completed_handler(Box::new(move |_| {
                        if retired {
                            std::thread::sleep(RETIRED);
                        }
                        *seen.lock().unwrap() = Some(in_handler());
                    }))
                    .unwrap();
                fill(&command_buffer, used, 0x5a);
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None, "{form}");
                let answer = answered.lock().unwrap().clone();
                if retired {
                    assert_eq!(
                        answer,
                        Some(Ok(true)),
                        "{form}: in its completed handler, {RETIRED:?} in"
                    );
                } else {
                    // Metal's answer here is a race of its own release against the handler: it is
                    // held to answering, never to an answer.
                    assert!(
                        matches!(answer, Some(Ok(_))),
                        "{form}: at once in its completed handler: {answer:?}"
                    );
                }
                assert_eq!(detach(), Ok(true), "{form}: once the wait returned");
            }
            drop(heap);
        }
        eprintln!(
            "DETACH handler: a detach in a completed handler past the retire detaches, and so \
             does one once the wait returned"
        );
    });
}

/// The deallocator of the pages a detach lets go of runs at the buffer's next rebind, inside the
/// call, on its thread, and, where the buffer is released detached, at the release, on the
/// releasing thread; never at the detach, as Metal runs it (the M3 Max, macOS 27.0.1). A
/// keep-alive goes after the deallocator, which may read the bytes, once the import is freed. With
/// GPU use before the detach and without.
#[test]
fn a_detach_holds_the_deallocator_until_the_rebind_or_the_release() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        for used in [false, true] {
            let Some((member, old)) = member_over(device, A) else {
                return;
            };
            if used {
                gpu_read(&queue, &member, 0..16);
            }
            detach(&member);
            std::thread::sleep(Duration::from_millis(20));
            assert_eq!(
                old.runs.load(Ordering::SeqCst),
                0,
                "used {used}: at the detach"
            );
            let new = rebind_to(&member, B);
            ran_here(&old, "the rebind");
            if used {
                gpu_read(&queue, &member, 0..16);
            }
            detach(&member);
            std::thread::sleep(Duration::from_millis(20));
            assert_eq!(
                new.runs.load(Ordering::SeqCst),
                0,
                "used {used}: the second detach"
            );
            drop(member);
            ran_here(&new, "the release of a detached buffer");
        }
        let Some((member, old, dropped)) = kept_member_over(device, A) else {
            return;
        };
        gpu_read(&queue, &member, 0..16);
        detach(&member);
        assert_eq!(old.runs.load(Ordering::SeqCst), 0, "kept: at the detach");
        // The keep-alive waits for the deallocator, which may read the bytes.
        assert!(
            !dropped.load(Ordering::SeqCst),
            "kept: the keep-alive went first"
        );
        drop(member);
        ran_here(&old, "kept: the release of a detached buffer");
        let deadline = Instant::now() + WAIT;
        while !dropped.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "kept: the keep-alive never went");
            std::thread::sleep(Duration::from_millis(5));
        }
        eprintln!("DETACH deallocator: at the rebind or the release, never at the detach");
    });
}

/// A heap of `LARGE` bytes over fresh host memory, and a buffer placed at its start.
fn large_heap(device: &Device) -> Option<(Heap, Buffer)> {
    let mapping = Mapping::anonymous(LARGE);
    let pointer = mapping.pointer();
    let descriptor = HeapDescriptor::new(HeapType::Placement, LARGE as u64, shared());
    let seen = Arc::new(Seen::default());
    // SAFETY: the deallocator releases the mapping once the heap's pages go.
    let made = unsafe {
        device.new_heap_with_bytes_no_copy(
            HostMemory::new(pointer),
            &descriptor,
            Some(recording(&seen, mapping)),
            None,
        )
    };
    let heap = match made {
        Ok(heap) => heap,
        Err(error) => {
            eprintln!("DETACH memory: no heap over host memory: {error:?}");
            return None;
        }
    };
    let placed = heap.new_buffer_at_offset(1 << 20, shared(), 0).unwrap();
    Some((heap, placed))
}

/// A detach places no memory of the bytes it detaches (the coordinator's lane row): once the
/// device's stand-in is made (a small buffer's detach), detaching a heap of 64 MiB, and a
/// no-copy buffer of 64 MiB, allocates no device memory, and a command buffer that names them
/// fails `DetachedBacking` with no device memory allocated for them either.
#[test]
fn a_detach_of_a_large_heap_allocates_no_device_memory() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let Some((small, _)) = member_over(device, A) else {
            return;
        };
        detach(&small);
        let Some((heap, placed)) = large_heap(device) else {
            return;
        };
        let mapping = Mapping::anonymous(LARGE);
        let pointer = mapping.pointer();
        let seen = Arc::new(Seen::default());
        // SAFETY: the deallocator releases the mapping once the buffer's pages go.
        let large = unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::new(pointer),
                LARGE as u64,
                ResourceOptions::default(),
                Some(recording(&seen, mapping)),
            )
        }
        .unwrap();
        let copied = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        // The copies' own work warmed: a command buffer like the failing ones, over plain bytes.
        let plain = buffer_with(device, &[0; 16]).unwrap();
        let warm = queue.command_buffer().unwrap();
        let mut blit = warm.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&plain, 0, &copied, 0, 16)
            .unwrap();
        blit.end_encoding().unwrap();
        warm.commit().unwrap();
        warm.wait_until_completed_for(WAIT).unwrap();
        let allocated = || device.work_counts().device_memory_allocated;
        let made = allocated();
        detach_heap(&heap);
        detach(&large);
        assert_eq!(allocated(), made, "a detach allocated device memory");
        assert_eq!(seen.runs.load(Ordering::SeqCst), 0, "ran at the detach");
        for (name, buffer) in [("the heap's buffer", &placed), ("the buffer", &large)] {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(buffer, 0, &copied, 0, 16)
                .unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            detached_backing(&command_buffer, name);
        }
        assert_eq!(
            allocated(),
            made,
            "a command buffer naming a detached buffer allocated device memory"
        );
        drop(large);
        ran_here(&seen, "the release of a detached buffer");
        eprintln!(
            "DETACH memory: 0 bytes allocated detaching {} MiB twice",
            LARGE >> 20
        );
    });
}

/// The rounds of [`a_detach_racing_a_commit_leaves_the_deallocator_to_the_release`], for each
/// form: two sweeps of the offsets.
const RACES: u32 = 640;

/// How far apart the detach and the commit of a race start, at most, either first.
const RACE_SPAN: Duration = Duration::from_micros(40);

/// How far one race's offset is from the last's.
const RACE_STEP: Duration = Duration::from_nanos(250);

/// When a race's two calls start, after its other thread is spawned.
const RACE_LEAD: Duration = Duration::from_micros(500);

/// Spin until `at`.
fn spin_until(at: Instant) {
    while Instant::now() < at {
        std::hint::spin_loop();
    }
}

/// A detach on one thread racing the commit, on another, of a command buffer whose blit names the
/// buffer: the old pages' deallocator runs in neither call (the detach or the commit, whichever
/// lets go of the pages last), wherever the device's recorder replays the command buffer's
/// commands, and runs once at the buffer's release, on the releasing thread, as Metal runs it
/// there; nothing is announced: with a keep-alive and without one. The command buffer, which an
/// event holds back until both calls returned, then fails `DetachedBacking`. The races start the
/// two calls up to 40 us apart, either first, swept in steps of 0.25 us, so that many land inside
/// each other.
#[test]
fn a_detach_racing_a_commit_leaves_the_deallocator_to_the_release() {
    for (form, kept) in [("no keep-alive", false), ("keep-alive", true)] {
        on_device_noticed(|device, messages| {
            let queue = device.new_command_queue().unwrap();
            let event = device.new_shared_event().unwrap();
            let made = || match kept {
                true => kept_member_over(device, A).map(|(buffer, seen, _)| (buffer, seen)),
                false => member_over(device, A),
            };
            let (mut early, mut elsewhere) = (0, 0);
            let before = device.work_counts();
            let span = RACE_SPAN.as_nanos() as i64;
            for round in 0..RACES {
                let Some((buffer, seen)) = made() else {
                    return;
                };
                let value = u64::from(round) + 1;
                let command_buffer = queue.command_buffer().unwrap();
                command_buffer
                    .encode_wait_for_event(EventRef::from(&event), value)
                    .unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&buffer, 0..16, 0x5a).unwrap();
                blit.end_encoding().unwrap();
                // The commit's start less the detach's, swept over the span.
                let step = RACE_STEP.as_nanos() as i64;
                let offset = (i64::from(round) * step) % (2 * span) - span;
                let start = Instant::now() + RACE_LEAD;
                let late_by = |ns: i64| Duration::from_nanos(ns.max(0) as u64);
                let (commit_at, detach_at) = (start + late_by(offset), start + late_by(-offset));
                std::thread::scope(|scope| {
                    scope.spawn(|| {
                        spin_until(commit_at);
                        command_buffer.commit().unwrap();
                    });
                    spin_until(detach_at);
                    detach(&buffer);
                });
                early += seen.runs.load(Ordering::SeqCst);
                event.set_signaled_value(value);
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                detached_backing(&command_buffer, form);
                early += seen.runs.load(Ordering::SeqCst);
                drop(command_buffer);
                drop(buffer);
                let here = *seen.thread.lock().unwrap() == Some(std::thread::current().id());
                match (seen.runs.load(Ordering::SeqCst), here) {
                    (1, true) => {}
                    _ => elsewhere += 1,
                }
            }
            let inline = device.work_counts().command_buffers_replayed_inline
                - before.command_buffers_replayed_inline;
            eprintln!(
                "DETACH race {} ({form}): {RACES} races, the deallocator ran before the release \
                 {early}, not once in the release on its thread {elsewhere}; {inline} command \
                 buffers replayed off the recorder's thread",
                device.name()
            );
            assert_eq!(
                (early, elsewhere),
                (0, 0),
                "{form}: the deallocator ran outside the buffer's release"
            );
            assert_once_allowing(&messages.take_notices(), &[], &shared_kinds(device));
        });
    }
}
