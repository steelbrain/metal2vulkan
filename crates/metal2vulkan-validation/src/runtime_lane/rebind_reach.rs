//! A command buffer committed before a rebind of its no-copy buffer or heap
//! (`rebind_bytes_no_copy`) and first submitted after it, on the device the cfg selects, under the
//! validation layer. The rule is Metal's as the consumer measured it on the M3 Max (macOS 26.7, the
//! rebind probe's tables 2 and 3; a detach then a replace of the backing, which the rebind is):
//!
//! - **Resolved at the first submission.** What the command buffer's encoders made resident
//!   (`use_resources`, case 3; `use_heap`, case 3h) reaches the pages the buffer lies in as the
//!   command buffer is first submitted: the new pages, and after two rebinds the last.
//! - **Unnamed reaches.** A buffer the command buffer neither names nor makes resident is reached
//!   at its old listing no more: the walk reads 0, whether an earlier command buffer used it
//!   (case 4) or none did since its pages were given (case 4x).
//! - **Named uses, encoded again**. A command that names the buffer (a binding, case 1; a binding
//!   the walk also reaches, case 2; a blit's operand, case 5) was resolved as it was encoded; the
//!   rebind encodes its compute, blit or render encoder again from its log, so it reaches the new
//!   pages, as Metal's does, and the command buffer completes: a draw reading a vertex buffer over
//!   host pages draws the new pages' corners. A call refused at its encoding is refused again and
//!   the calls after it run.
//! - **Submitted is kept.** A command buffer whose first batch was submitted before the rebind
//!   keeps the table and the pages it took then for every later batch (the split: a wait for an
//!   event after an encoder), named or resident (table 3: a running command buffer reads the old
//!   pages until it completes). The old pages' deallocator runs once it completed; so does a
//!   released buffer's that its table lists, announced once (Metal's runs at the release), unless
//!   the caller gave a keep-alive, which keeps the pages mapped while the deallocator runs at the
//!   release, while a buffer released beside it that no submitted table lists goes at once. One
//!   released after a rebind encoded a pending command buffer again goes at that command buffer's
//!   first submission, before it completes, its deallocator free to read it. One a submitted command buffer's command names goes as that command buffer
//!   completes, as Metal's does, unannounced.
//!
//! The walker (`walk-a-device-pointer-chain-through-a-loop-phi`, an owned case) reaches its
//! argument buffer's member through a pointer it loads; the member holds a node whose value is
//! the page set's ([`A`], [`B`], [`C`]), and its output is calibrated over plain buffers holding
//! each node, and an all-zero one for an unreached member.

use super::degrade_notices::{assert_made_once, made_once, on_device as on_device_noticed};
use super::host_memory::{gpu_read, host_page, no_copy, not_importable, on_device, recording};
use super::host_memory::{Mapping, Seen};
use super::no_copy_heaps::{no_copy_heap, shared, HEAP};
use super::predication::{kernel, EQUAL, NOT_EQUAL};
use super::property_reads::{batches_submitted, first_batch_submitted};
use super::rebinds::{rebind, rebind_heap};
use super::resource_ids::DELAYED_DEALLOCATORS;
use super::shared_kinds;
use super::{
    bound_buffers, buffer_with, encode_case, pipeline_state, public_case, read_buffer,
    validated_device_with, CaseResources, Messages, PublicCase, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::candidate::engine::ComputeOutput;
use crate::gpu::TEST_DEVICE;
use crate::worker::alloc::live_bytes;
use metal2vulkan::metal::{IndexType, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, BufferContents, BufferDescriptor, ClearColor, CommandBuffer, CommandBufferError,
    CommandBufferStatus, CommandQueue, CommitFailure, Deallocator, Degradation, Device, EventRef,
    FailedEncoders, Heap, HeapType, HostMemory, KeepAlive, LoadAction, PipelineOptions, Region,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size,
    StoreAction, Submission, SubmissionCheck, TextureDescriptor,
};
use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

/// Walks a pointer chain from the address its argument buffer (buffer 0) holds at offset 0.
pub(super) const POINTER_WALKER: &str = "walk-a-device-pointer-chain-through-a-loop-phi";

/// The walker's binding of its step count, a word it reads.
const STEPS: u32 = 2;

/// The node values of the old pages, the first new ones and the second, as the probe's sentinels.
pub(super) const A: u32 = 0xa0;
pub(super) const B: u32 = 0xb0;
pub(super) const C: u32 = 0xc0;

/// The kinds the gap's test takes.
pub(super) const REBOUND_USES: [Degradation; 1] = [Degradation::EncodedUseRebound];

/// Where a heap's member lies in it.
pub(super) const IN_HEAP: usize = 64 << 10;

/// A node of the walked chain: no next node, and `value`.
pub(super) fn node(value: u32) -> Vec<u8> {
    let mut bytes = vec![0; 16];
    bytes[8..12].copy_from_slice(&value.to_le_bytes());
    bytes
}

/// The walker's output binding, offset and length.
pub(super) fn output(walker: &PublicCase) -> (u32, u64, u64) {
    let ComputeOutput::Buffer {
        binding,
        offset,
        length,
    } = walker.compute.output
    else {
        panic!("{POINTER_WALKER}: a buffer output");
    };
    (binding, offset, length)
}

/// The walker's outputs over a member holding each node, and over an unreached one.
pub(super) struct Outputs {
    pub(super) a: Vec<u8>,
    pub(super) b: Vec<u8>,
    pub(super) c: Vec<u8>,
    pub(super) zero: Vec<u8>,
}

impl Outputs {
    /// Each walked over a plain buffer its encoder makes resident; distinct.
    pub(super) fn new(device: &Device, queue: &CommandQueue, walker: &PublicCase) -> Outputs {
        let over = |bytes: Vec<u8>| {
            let member = buffer_with(device, &bytes).unwrap();
            let walked = walk(device, queue, walker, &member, Uses::Resident, false, || {});
            assert_eq!(walked.command_buffer.error(), None);
            walked.produced
        };
        let outputs = Outputs {
            a: over(node(A)),
            b: over(node(B)),
            c: over(node(C)),
            zero: over(vec![0; 16]),
        };
        let all = [&outputs.a, &outputs.b, &outputs.c, &outputs.zero];
        for (at, one) in all.iter().enumerate() {
            assert!(
                all[at + 1..].iter().all(|other| other != one),
                "the walker's outputs over the nodes are not distinct"
            );
        }
        outputs
    }
}

/// What the walker's command buffer holds of the member besides the walk's reach of it.
#[derive(Clone, Copy)]
pub(super) enum Uses<'a> {
    /// Its encoder makes the member resident (`use_resources`): case 3.
    Resident,
    /// Its encoder makes the member's heap resident (`use_heap`): case 3h.
    Heap(&'a Heap),
    /// Nothing: cases 4 and 4x.
    Unnamed,
    /// The walker's step count is read from the given buffer, bound: case 1.
    Steps(&'a Buffer),
    /// The walker writes its output into the member, bound: case 2.
    Output,
    /// A blit before the walker copies the member's node out: case 5.
    Blit,
    /// As [`Uses::Output`], after a blit that copies the member's node out ([`Uses::Blit`]),
    /// each encoder with a call refused at its encoding before those: a fill past the member, a
    /// binding past the table.
    Refused,
}

/// A walk's command buffer, once it completed, and what it left.
pub(super) struct Walked {
    pub(super) command_buffer: CommandBuffer,
    /// The walker's output.
    pub(super) produced: Vec<u8>,
    /// The event the command buffer signals as its last command, to 1.
    pub(super) signaled: u64,
    /// Whether its completed handler ran.
    pub(super) handled: bool,
    /// What the blit before the walker copied out of the member ([`Uses::Blit`]).
    pub(super) copied: Option<Vec<u8>>,
}

/// The walker run over `member`, held by `uses`, in a command buffer that waits for an event
/// before it where `wait` (`between`, a rebind, running once it is committed, before the event),
/// else at once; the command buffer signals an event of its own last.
pub(super) fn walk(
    device: &Device,
    queue: &CommandQueue,
    walker: &PublicCase,
    member: &Buffer,
    uses: Uses<'_>,
    wait: bool,
    between: impl FnOnce(),
) -> Walked {
    let (binding, offset, length) = output(walker);
    let state = pipeline_state(device, walker).unwrap();
    let mut holding = CaseResources::new(device, walker).unwrap();
    let [field] = holding.argument_buffers.keys().copied().collect::<Vec<_>>()[..] else {
        panic!("{POINTER_WALKER}: one argument-buffer buffer");
    };
    holding.argument_buffers.insert(field, member.clone());
    let mut buffers = bound_buffers(device, &walker.compute, &holding).unwrap();
    match uses {
        Uses::Steps(steps) => {
            buffers.insert(STEPS, steps.clone());
        }
        Uses::Output | Uses::Refused => {
            buffers.insert(binding, member.clone());
        }
        _ => {}
    }
    if !matches!(uses, Uses::Resident) {
        // The encoder makes no member resident: the walk reaches it unnamed.
        drop(holding.argument_buffers.remove(&field));
    }
    let event = device.new_shared_event().unwrap();
    let done = device.new_shared_event().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    if wait {
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
    }
    let copied = match uses {
        Uses::Blit | Uses::Refused => {
            let copied = device.new_buffer(16, ResourceOptions::shared()).unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            if let Uses::Refused = uses {
                let past = member.length() + 16;
                assert!(blit.fill_buffer(member, 0..past, 0).is_err());
            }
            blit.copy_from_buffer_to_buffer(member, 0, &copied, 0, 16)
                .unwrap();
            blit.end_encoding().unwrap();
            Some(copied)
        }
        _ => None,
    };
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    if let Uses::Heap(heap) = uses {
        encoder.use_heap(heap).unwrap();
    }
    if let Uses::Refused = uses {
        assert!(encoder.set_buffer(Some(member), 0, u32::MAX).is_err());
    }
    encode_case(&mut encoder, &state, &walker.compute, &buffers, &holding).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer
        .encode_signal_event(EventRef::from(&done), 1)
        .unwrap();
    let handled = Arc::new(AtomicBool::new(false));
    {
        let handled = Arc::clone(&handled);
        command_buffer
            .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                handled.store(true, Ordering::SeqCst);
            }))
            .unwrap();
    }
    let before = batches_submitted(device);
    command_buffer.commit().unwrap();
    if wait {
        between();
        assert_eq!(
            batches_submitted(device),
            before,
            "a batch was submitted before the event"
        );
        event.set_signaled_value(1);
    }
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let produced = read_buffer(&buffers[&binding], offset, length).unwrap();
    let deadline = Instant::now() + WAIT;
    while !handled.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "the completed handler never ran");
        std::thread::sleep(Duration::from_millis(1));
    }
    Walked {
        command_buffer,
        produced,
        signaled: done.signaled_value(),
        handled: handled.load(Ordering::SeqCst),
        copied: copied.map(|copied| read_buffer(&copied, 0, 16).unwrap()),
    }
}

/// `len` bytes of fresh host memory holding `bytes` at `at`, zero elsewhere.
pub(super) fn pages(len: usize, at: usize, bytes: &[u8]) -> Mapping {
    let mapping = Mapping::anonymous(len);
    mapping.write(at, bytes);
    mapping
}

/// The buffers of these tests: two of any host's pages.
pub(super) fn buffer_len() -> usize {
    2 * host_page().max(16 << 10)
}

/// A no-copy buffer over `mapping`, whose deallocator records into `seen`; `None`, printed, where
/// the device imports no host memory.
pub(super) fn buffer_over(device: &Device, mapping: Mapping, seen: &Arc<Seen>) -> Option<Buffer> {
    buffer_of(device, mapping, buffer_len(), seen)
}

/// [`buffer_over`] of `len` bytes, `mapping`'s length.
pub(super) fn buffer_of(
    device: &Device,
    mapping: Mapping,
    len: usize,
    seen: &Arc<Seen>,
) -> Option<Buffer> {
    let pointer = mapping.pointer();
    // SAFETY: the deallocator releases the mapping, after the buffer's last use of it.
    let made = unsafe {
        no_copy(
            device,
            pointer,
            len,
            ResourceOptions::default(),
            Some(recording(seen, mapping)),
        )
    };
    match made {
        Ok(buffer) => Some(buffer),
        Err(error) if not_importable(&error) => {
            eprintln!("REBIND-REACH: the device imports no host memory: {error:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

/// A no-copy buffer over a node of `value`, its deallocator's runs in a `Seen` of its own.
pub(super) fn member_over(device: &Device, value: u32) -> Option<(Buffer, Arc<Seen>)> {
    let seen = Arc::new(Seen::default());
    let buffer = buffer_over(device, pages(buffer_len(), 0, &node(value)), &seen)?;
    Some((buffer, seen))
}

/// Sets its flag as it drops.
struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// A no-copy buffer over a node of `value`, its deallocator's runs in a `Seen` of its own, whose
/// keep-alive holds the mapping and sets the flag returned as it drops; `None`, printed, where the
/// device imports no host memory.
pub(super) fn kept_member_over(
    device: &Device,
    value: u32,
) -> Option<(Buffer, Arc<Seen>, Arc<AtomicBool>)> {
    let seen = Arc::new(Seen::default());
    let dropped = Arc::new(AtomicBool::new(false));
    let mapping = pages(buffer_len(), 0, &node(value));
    let pointer = mapping.pointer();
    // The mapping goes first, then the flag is set.
    let keep_alive: KeepAlive = Box::new((mapping, DropFlag(Arc::clone(&dropped))));
    // SAFETY: the keep-alive releases the mapping, after the buffer's last use of it.
    let memory = unsafe { HostMemory::new(pointer) };
    let contents = BufferContents::NoCopy {
        memory,
        deallocator: Some(recording(&seen, ())),
        keep_alive: Some(keep_alive),
    };
    let descriptor =
        BufferDescriptor::new(buffer_len() as u64, ResourceOptions::default(), contents);
    match device.new_buffer_with_descriptor(descriptor) {
        Ok(buffer) => Some((buffer, seen, dropped)),
        Err(error) if not_importable(&error) => {
            eprintln!("REBIND-REACH: the device imports no host memory: {error:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_descriptor: {error:?}"),
    }
}

/// Rebind `buffer` to fresh pages holding a node of `value`.
pub(super) fn rebind_to(buffer: &Buffer, value: u32) -> Arc<Seen> {
    let seen = Arc::new(Seen::default());
    rebind(buffer, pages(buffer_len(), 0, &node(value)), &seen).unwrap();
    seen
}

/// A placement heap over the caller's memory holding a node of `A` at [`IN_HEAP`], and its member
/// there; `None` where the device imports no host memory.
pub(super) fn heap_member(device: &Device) -> Option<(Heap, Buffer)> {
    let mapping = pages(HEAP, IN_HEAP, &node(A));
    let pointer = mapping.pointer();
    // SAFETY: the deallocator releases the mapping once the heap's pages go.
    let heap = unsafe {
        no_copy_heap(
            device,
            pointer,
            HeapType::Placement,
            Some(recording(&Arc::new(Seen::default()), mapping)),
        )
    }?;
    let member = heap
        .new_buffer_at_offset(256, shared(), IN_HEAP as u64)
        .unwrap();
    Some((heap, member))
}

/// Rebind `heap` to fresh pages holding a node of `value` at [`IN_HEAP`].
pub(super) fn rebind_heap_to(heap: &Heap, value: u32) {
    let seen = Arc::new(Seen::default());
    rebind_heap(heap, pages(HEAP, IN_HEAP, &node(value)), &seen).unwrap();
}

/// Wait until the deallocator `seen` records ran.
pub(super) fn deallocated(seen: &Seen) {
    let deadline = Instant::now() + WAIT;
    while seen.runs.load(Ordering::SeqCst) == 0 {
        assert!(Instant::now() < deadline, "the deallocator never ran");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Table 2, case 3 and 3h (detach+replace, 6a/6b): what a pending command buffer's encoder made
/// resident reaches the pages its buffer lies in at the first submission, B after one rebind, C
/// after two; with no rebind, A. Buffer and heap forms, `use_resources` and `use_heap`.
#[test]
fn a_pending_command_buffer_s_resident_uses_reach_the_pages_of_its_first_submission() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let outputs = Outputs::new(device, &queue, &walker);
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        let held = |member: &Buffer, uses: Uses<'_>, between: &dyn Fn()| {
            let walked = walk(device, &queue, &walker, member, uses, true, between);
            assert_eq!(walked.command_buffer.error(), None);
            walked.produced
        };
        assert!(
            held(&member, Uses::Resident, &|| {}) == outputs.a,
            "3: control"
        );
        let once = held(&member, Uses::Resident, &|| drop(rebind_to(&member, B)));
        assert!(once == outputs.b, "3: one rebind");
        let twice = held(&member, Uses::Resident, &|| {
            rebind_to(&member, C);
            rebind_to(&member, B);
        });
        assert!(twice == outputs.b, "3: two rebinds, the last's pages");
        let to_c = held(&member, Uses::Resident, &|| {
            rebind_to(&member, A);
            rebind_to(&member, C);
        });
        assert!(to_c == outputs.c, "3: two rebinds, to C");

        let Some((heap, placed)) = heap_member(device) else {
            return;
        };
        for (name, uses) in [("3", Uses::Resident), ("3h", Uses::Heap(&heap))] {
            rebind_heap_to(&heap, A);
            assert!(
                held(&placed, uses, &|| {}) == outputs.a,
                "heap {name}: control"
            );
            let once = held(&placed, uses, &|| rebind_heap_to(&heap, B));
            assert!(once == outputs.b, "heap {name}: one rebind");
            let twice = held(&placed, uses, &|| {
                rebind_heap_to(&heap, B);
                rebind_heap_to(&heap, C);
            });
            assert!(twice == outputs.c, "heap {name}: two rebinds");
        }
        eprintln!("REBIND-REACH resident: B after one rebind, C after two, buffer and heap");
    });
}

/// Table 2, cases 4 and 4x: a buffer a pending command buffer neither names nor makes resident
/// is reached at no pages once rebound, its walk reading 0 and its command buffer completing with
/// no error; with no rebind, one an earlier command buffer used reads A (case 4) and one none
/// used reads 0 (4x). Buffer and heap forms.
#[test]
fn a_pending_command_buffer_reaches_a_buffer_rebound_unnamed_at_no_pages() {
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
        let Some((never, _)) = member_over(device, A) else {
            return;
        };
        assert!(held(&never, &|| {}) == outputs.zero, "4x: control");
        let rebound = held(&never, &|| drop(rebind_to(&never, B)));
        assert!(rebound == outputs.zero, "4x: rebound");

        let (used, _) = member_over(device, A).unwrap();
        gpu_read(&queue, &used, 0..16);
        assert!(held(&used, &|| {}) == outputs.a, "4: control");
        let rebound = held(&used, &|| drop(rebind_to(&used, B)));
        assert!(rebound == outputs.zero, "4: rebound");
        gpu_read(&queue, &used, 0..16);
        assert!(held(&used, &|| {}) == outputs.b, "4: used again");
        let rebound = held(&used, &|| {
            rebind_to(&used, C);
            rebind_to(&used, A);
        });
        assert!(rebound == outputs.zero, "4: two rebinds");

        let Some((heap, placed)) = heap_member(device) else {
            return;
        };
        assert!(held(&placed, &|| {}) == outputs.zero, "heap 4x: control");
        let rebound = held(&placed, &|| rebind_heap_to(&heap, B));
        assert!(rebound == outputs.zero, "heap 4x: rebound");
        let other = heap.new_buffer_at_offset(256, shared(), 0).unwrap();
        gpu_read(&queue, &other, 0..4);
        assert!(held(&placed, &|| {}) == outputs.b, "heap 4: control");
        let rebound = held(&placed, &|| rebind_heap_to(&heap, C));
        assert!(rebound == outputs.zero, "heap 4: rebound");
        eprintln!("REBIND-REACH unnamed: 0 once rebound, buffer and heap");
    });
}

/// Table 2 cases 1, 2 and 5, as Metal runs them: a pending command buffer whose command names a
/// buffer rebound since it was encoded (bound and read, bound and walked, a blit's operand; a
/// heap's member bound or blitted) reaches the pages the buffer lies in at its first submission.
/// The rebind encodes the command buffer's compute and blit encoders that named it again from
/// their logs (`replay.rs`): the blit copies the new node, the walk reads it, the bound output
/// lands in the new pages, and the command buffer completes with no error, its signal fired and
/// its handler run; nothing of its own is announced (a device without unified memory announces
/// its first `Shared` buffer in system memory). Each once failed `EncodedUseRebound`. The device's
/// counts show the encoders encoded again. A rebind before the encoding takes the new pages as
/// before.
#[test]
fn a_pending_command_buffer_naming_a_buffer_rebound_since_reaches_the_new_pages() {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let outputs = Outputs::new(device, &queue, &walker);
        let ran = |walked: &Walked, name: &str| {
            let command_buffer = &walked.command_buffer;
            assert_eq!(command_buffer.error(), None, "{name}");
            assert_eq!(
                command_buffer.status(),
                CommandBufferStatus::Completed,
                "{name}"
            );
            assert_eq!(walked.signaled, 1, "{name}: its signal");
            assert!(walked.handled, "{name}: its handler");
        };
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        let held = |member: &Buffer, uses: Uses<'_>, between: &dyn Fn()| {
            walk(device, &queue, &walker, member, uses, true, between)
        };
        let before = device.work_counts();
        let blit = held(&member, Uses::Blit, &|| drop(rebind_to(&member, B)));
        ran(&blit, "5: a blit's operand");
        assert_eq!(blit.copied.as_deref(), Some(&node(B)[..]), "5: the copy");
        assert!(blit.produced == outputs.b, "5: the walk");
        let output = held(&member, Uses::Output, &|| drop(rebind_to(&member, C)));
        ran(&output, "2: bound and walked");
        assert!(
            output.produced == outputs.c,
            "2: the output in the new pages"
        );
        let steps_seen = Arc::new(Seen::default());
        let steps = buffer_over(
            device,
            pages(buffer_len(), 0, &2u32.to_le_bytes()),
            &steps_seen,
        )
        .unwrap();
        let plain = buffer_with(device, &node(A)).unwrap();
        let bound = held(&plain, Uses::Steps(&steps), &|| {
            let seen = Arc::new(Seen::default());
            rebind(&steps, pages(buffer_len(), 0, &2u32.to_le_bytes()), &seen).unwrap();
        });
        ran(&bound, "1: bound and read");
        assert!(bound.produced == outputs.a, "1: the walk");
        // Rebound before the encoding: the commands take the new pages.
        rebind_to(&member, A);
        let before_encoding = held(&member, Uses::Blit, &|| {});
        ran(&before_encoding, "5: rebound before the encoding");
        assert!(
            before_encoding.produced == outputs.a,
            "5: rebound before the encoding"
        );

        let Some((heap, placed)) = heap_member(device) else {
            return;
        };
        let blit = held(&placed, Uses::Blit, &|| rebind_heap_to(&heap, B));
        ran(&blit, "heap 5: a blit's operand");
        assert_eq!(
            blit.copied.as_deref(),
            Some(&node(B)[..]),
            "heap 5: the copy"
        );
        let output = held(&placed, Uses::Output, &|| rebind_heap_to(&heap, C));
        ran(&output, "heap 2: bound and walked");
        assert!(output.produced == outputs.c, "heap 2: the output");
        // Nothing of its own: a device without unified memory announces its first `Shared`
        // buffer in system memory (An RTX 3080 announced it here).
        let notices = assert_made_once(messages, &[], &shared_kinds(device));
        assert!(!notices.contains_key(&Degradation::EncodedUseRebound));
        let after = device.work_counts();
        let reencoded = after.encoders_reencoded - before.encoders_reencoded;
        assert!(reencoded >= 5, "encoders encoded again: {reencoded}");
        assert_eq!(after.reencodes_refused, before.reencodes_refused);
        eprintln!(
            "CBF-MARK {}: named uses rebound before the first submission reached the new pages; \
             {reencoded} encoders encoded again",
            device.name()
        );
    });
}

/// An encoder one of whose calls was refused at its encoding is encoded again at a rebind
/// (`replay.rs`; it once kept its first recording and failed `EncodedUseRebound`): the call is
/// refused again and the calls after it run, as they ran after its refusal. A blit encoder's fill
/// past the member, refused, then its copy of the member, which copies the new pages' node; a
/// compute encoder's binding past its table, refused, then the walk, whose output lands in the new
/// pages. The command buffer completes, its signal fired and its handler run, nothing is
/// announced, and the device counts both encoders encoded again and no encoding refused. The
/// render encoder's case is [`a_render_pass_with_calls_refused_at_its_encoding_is_encoded_again`].
#[test]
fn an_encoder_with_a_call_refused_at_its_encoding_is_encoded_again_with_it_refused() {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let outputs = Outputs::new(device, &queue, &walker);
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        let before = device.work_counts();
        let walked = walk(
            device,
            &queue,
            &walker,
            &member,
            Uses::Refused,
            true,
            || drop(rebind_to(&member, B)),
        );
        let command_buffer = &walked.command_buffer;
        assert_eq!(command_buffer.error(), None);
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(walked.signaled, 1, "its signal");
        assert!(walked.handled, "its handler");
        assert_eq!(
            walked.copied.as_deref(),
            Some(&node(B)[..]),
            "the copy after the refused fill"
        );
        assert!(
            walked.produced == outputs.b,
            "the walk after the refused binding"
        );
        let notices = assert_made_once(messages, &[], &shared_kinds(device));
        assert!(!notices.contains_key(&Degradation::EncodedUseRebound));
        let after = device.work_counts();
        assert_eq!(after.encoders_reencoded, before.encoders_reencoded + 2);
        assert_eq!(after.reencodes_refused, before.reencodes_refused);
        eprintln!(
            "CBF-MARK {}: encoders with a call refused at their encoding were encoded again",
            device.name()
        );
    });
}

/// A unit of command predication a rebind encodes again: an encoder of one keeps its log, its
/// Ifs, Elses and Ends among its calls, where its unit laid no loop and left no work out and, of
/// its unit's predication, nothing reached in over its open and nothing of its own reaches out
/// over its end (`Predication::alone`). A plain compute encoder that dispatches `prd_mark` into
/// the member's word 0, and next in its unit a predicated one whose failing If holds a dispatch
/// into its word 1, both naming the member rebound after the commit: the plain encoder laid no
/// step of the unit, and the predicated one laid the unit's first step and the unit ended with
/// it, so both are encoded again, their dispatches reach the new pages, the held dispatch runs
/// nothing and the command buffer completes with no error, its signal fired. Both once failed
/// `EncodedUseRebound` for sharing one unit. The blit encoder before them, its own submission
/// (`checks::metal_submissions`: consecutive compute encoders share one), copies the new pages
/// too.
#[test]
fn a_unit_s_plain_encoder_before_its_predicated_one_is_encoded_again() {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        let copied = buffer_with(device, &[0x5a; 16]).unwrap();
        let word = buffer_with(device, &[0; 4]).unwrap();
        let mark = kernel(device, "prd_mark", false);
        let event = device.new_shared_event().unwrap();
        let done = device.new_shared_event().unwrap();
        let one = Size::new(1, 1, 1);
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&member, 0, &copied, 0, 16)
            .unwrap();
        blit.end_encoding().unwrap();
        // The unit's first encoder, which lays no step of it: its dispatch runs ungated.
        let mut compute = command_buffer.compute_command_encoder().unwrap();
        compute.set_compute_pipeline_state(&mark).unwrap();
        compute.set_bytes(&0u32.to_le_bytes(), 0).unwrap();
        compute.set_buffer(Some(&member), 0, 1).unwrap();
        compute.dispatch_threads(one, one).unwrap();
        compute.end_encoding().unwrap();
        // The unit's last encoder, which lays its every step: `0 != 0` fails, so its block runs
        // nothing.
        let mut compute = command_buffer.compute_command_encoder().unwrap();
        compute.set_compute_pipeline_state(&mark).unwrap();
        compute.set_bytes(&1u32.to_le_bytes(), 0).unwrap();
        compute.set_buffer(Some(&member), 0, 1).unwrap();
        compute
            .encode_start_if(Some(&word), 0, NOT_EQUAL, 0)
            .unwrap();
        compute.dispatch_threads(one, one).unwrap();
        assert!(compute.encode_end_if().unwrap());
        compute.end_encoding().unwrap();
        command_buffer
            .encode_signal_event(EventRef::from(&done), 1)
            .unwrap();
        command_buffer.commit().unwrap();
        let before = device.work_counts();
        drop(rebind_to(&member, B));
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let mut marked = node(B);
        marked[..4].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(
            read_buffer(&member, 0, 16).unwrap(),
            marked,
            "the ungated dispatch ran over the new pages, and the held one ran nothing"
        );
        assert_eq!(
            read_buffer(&copied, 0, 16).unwrap(),
            node(B),
            "the blit's copy of the new pages"
        );
        assert_eq!(done.signaled_value(), 1, "its signal");
        let notices = assert_made_once(messages, &[], &shared_kinds(device));
        assert!(!notices.contains_key(&Degradation::EncodedUseRebound));
        let after = device.work_counts();
        // The blit encoder and the unit's two, each naming the member.
        assert_eq!(after.encoders_reencoded, before.encoders_reencoded + 3);
        assert_eq!(after.reencodes_refused, before.reencodes_refused);
        eprintln!(
            "REBIND-REACH {}: a unit's plain encoder before its predicated one, both encoded \
             again",
            device.name()
        );
    });
}

/// What a rebind cannot encode again keeps the gap: a unit of command predication whose first
/// step lies in an earlier encoder than its last keeps no log from that encoder on
/// (`Predication::alone`). The commands of the unit's later encoders run from gates the step
/// before them writes, which lie in that encoder's recording; an encoding again lays steps and
/// gates of its own, leaving those written by nothing -- a gate holds the group counts only
/// where its step writes them -- so such a command would run nothing where Metal runs it.
/// Encoding a run of a unit's encoders again together, which that shape needs, is not done.
///
/// The unit's first encoder dispatches `prd_mark` into the member rebound after the commit and
/// then opens and closes an If, so the member is named by a dispatch before the unit's first
/// record, which runs ungated; next in its unit a plain encoder dispatches into a buffer of its
/// own, from a gate that first encoder's step writes. The first encoder laid the unit's first
/// step and the unit went on past it, so it keeps no log and keeps its first recording, which
/// named the old pages: its submission (the unit's two encoders) runs nothing, the member keeps
/// the new pages' node, the command buffer fails `EncodedUseRebound`, announced once, the notice
/// naming the encoder and why, and its signal fires. The blit encoder before it, which copies
/// the member, is encoded again, and its own submission
/// (`checks::metal_submissions`: consecutive compute encoders share one) copies the new pages.
#[test]
fn a_unit_s_encoder_that_laid_its_first_step_fails_encoded_use_rebound() {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        let copied = buffer_with(device, &[0x5a; 16]).unwrap();
        let word = buffer_with(device, &[0; 4]).unwrap();
        // The later encoder's own log, over no host pages: its gated dispatch names nothing the
        // rebind moves.
        let log = buffer_with(device, &[0; 16]).unwrap();
        let mark = kernel(device, "prd_mark", false);
        let event = device.new_shared_event().unwrap();
        let done = device.new_shared_event().unwrap();
        let one = Size::new(1, 1, 1);
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&member, 0, &copied, 0, 16)
            .unwrap();
        blit.end_encoding().unwrap();
        // The unit's first encoder: its dispatch comes before the unit's first record, so it
        // runs ungated, and its empty If lays that first step.
        let mut compute = command_buffer.compute_command_encoder().unwrap();
        compute.set_compute_pipeline_state(&mark).unwrap();
        compute.set_bytes(&0u32.to_le_bytes(), 0).unwrap();
        compute.set_buffer(Some(&member), 0, 1).unwrap();
        compute.dispatch_threads(one, one).unwrap();
        compute.encode_start_if(Some(&word), 0, EQUAL, 0).unwrap();
        assert!(compute.encode_end_if().unwrap());
        compute.end_encoding().unwrap();
        // Next in the unit, after that step: its dispatch runs from a gate the step writes.
        let mut compute = command_buffer.compute_command_encoder().unwrap();
        compute.set_compute_pipeline_state(&mark).unwrap();
        compute.set_bytes(&0u32.to_le_bytes(), 0).unwrap();
        compute.set_buffer(Some(&log), 0, 1).unwrap();
        compute.dispatch_threads(one, one).unwrap();
        compute.end_encoding().unwrap();
        command_buffer
            .encode_signal_event(EventRef::from(&done), 1)
            .unwrap();
        command_buffer.commit().unwrap();
        let before = device.work_counts();
        drop(rebind_to(&member, B));
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
        assert_eq!(
            command_buffer.error(),
            Some(CommandBufferError::EncodedUseRebound)
        );
        assert_eq!(
            read_buffer(&copied, 0, 16).unwrap(),
            node(B),
            "the blit's copy of the new pages"
        );
        assert_eq!(
            read_buffer(&member, 0, 16).unwrap(),
            node(B),
            "the unit's dispatches ran nothing"
        );
        assert_eq!(
            read_buffer(&log, 0, 16).unwrap(),
            vec![0; 16],
            "the gated dispatch of the unit's later encoder ran nothing"
        );
        assert_eq!(done.signaled_value(), 1, "its signal");
        let notices = made_once(messages, &REBOUND_USES, &shared_kinds(device));
        // The notice names where the first submission found the use, and why no rebind
        // encoded it again.
        let reason = (notices.iter())
            .filter_map(|message| message.degradation.as_ref())
            .find(|notice| notice.degradation == Degradation::EncodedUseRebound)
            .map(|notice| notice.reason.clone());
        let found = "(found in its compute encoder 1: a command of its own named it, not encoded \
                     again as its unit of command predication laid a step before it opened, or \
                     one of its own that the unit's later encoders read, or a loop, or left work \
                     out";
        assert!(
            reason
                .as_deref()
                .is_some_and(|reason| reason.contains(found)),
            "{reason:?}"
        );
        let after = device.work_counts();
        // The blit encoder, alone, encoded again.
        assert_eq!(after.encoders_reencoded, before.encoders_reencoded + 1);
        eprintln!("REBIND-REACH not encoded again: EncodedUseRebound, announced once");
    });
}

/// What the logs cost: one compute encoder of [`COST_DISPATCHES`] walker dispatches, each with
/// its pipeline, buffers and resident members set again (`encode_case`), encoded three ways on
/// one device without the validation layer (whose own cost per call would hide the log's): with
/// no buffer over host pages on the device (nothing logged), with one the encoder does not name
/// (each call logged, the log let go at its end), and naming it (the log kept until the first
/// submission). Each way is encoded [`COST_ROUNDS`] times, the logged two alternating, and its
/// fastest round kept. `CBF-COST` prints each way's encode time per dispatch; `CBF-MEMORY` the
/// command buffer's memory once its encoder ended, unlogged and with the log kept: the heap bytes
/// the process's counting allocator saw it take (the runtime's own, not the driver's) and the
/// process's resident growth (Linux), beside the log's own bytes. The log is kept only where a
/// command named the buffer. The numbers are printed, not asserted: a release build gives those
/// records.
#[test]
fn a_dispatch_heavy_encoder_s_log_costs_a_push_per_call() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    let device = &device;
    let queue = device.new_command_queue().unwrap();
    let walker = public_case(POINTER_WALKER).unwrap();
    let state = pipeline_state(device, &walker).unwrap();
    let holding = CaseResources::new(device, &walker).unwrap();
    let plain = bound_buffers(device, &walker.compute, &holding).unwrap();
    let encode = |buffers: &BTreeMap<u32, Buffer>| {
        let command_buffer = queue.command_buffer().unwrap();
        let (heap, resident) = (live_bytes(), resident_bytes());
        let start = Instant::now();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        for _ in 0..COST_DISPATCHES {
            encode_case(&mut encoder, &state, &walker.compute, buffers, &holding).unwrap();
        }
        encoder.end_encoding().unwrap();
        let elapsed = start.elapsed();
        let held = Held {
            nanos: elapsed.as_secs_f64() * 1e9 / COST_DISPATCHES as f64,
            heap: live_bytes().saturating_sub(heap),
            resident: resident_bytes()
                .zip(resident)
                .map(|(now, then)| now.saturating_sub(then)),
        };
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        held
    };
    let before = device.work_counts();
    let off: Vec<Held> = (0..COST_ROUNDS).map(|_| encode(&plain)).collect();
    assert_eq!(
        device.work_counts().replay_logs_kept,
        before.replay_logs_kept,
        "a log kept on a device with no buffer over host pages"
    );
    let Some((member, _)) = member_over(device, A) else {
        return;
    };
    let mut named = plain.clone();
    let (binding, _, _) = output(&walker);
    named.insert(binding, member.clone());
    let (mut unnamed, mut kept, mut bytes) = (Vec::new(), Vec::new(), 0);
    for _ in 0..COST_ROUNDS {
        let logged = device.work_counts();
        unnamed.push(encode(&plain));
        let let_go = device.work_counts();
        assert_eq!(
            let_go.replay_logs_kept, logged.replay_logs_kept,
            "a log kept where no command named a buffer over host pages"
        );
        kept.push(encode(&named));
        let after = device.work_counts();
        assert_eq!(
            after.replay_logs_kept,
            let_go.replay_logs_kept + 1,
            "the log kept"
        );
        bytes = bytes.max(after.replay_log_bytes - let_go.replay_log_bytes);
    }
    let fastest = |held: &[Held]| held.iter().map(|h| h.nanos).fold(f64::INFINITY, f64::min);
    let least = |held: &[Held]| held.iter().map(|h| h.heap).min().unwrap_or(0);
    let resident = |held: &[Held]| {
        held.iter()
            .filter_map(|h| h.resident)
            .min()
            .map_or_else(|| "n/a".to_owned(), |bytes| bytes.to_string())
    };
    let name = device.name();
    eprintln!(
        "CBF-COST {name}: {COST_DISPATCHES} dispatches, fastest of {COST_ROUNDS}: {:.0} ns per \
         dispatch unlogged, {:.0} ns logged and let go, {:.0} ns logged and kept",
        fastest(&off),
        fastest(&unnamed),
        fastest(&kept),
    );
    eprintln!(
        "CBF-MEMORY {name}: the command buffer once its encoder ended, least of {COST_ROUNDS}: \
         heap {} bytes unlogged, {} with the log kept; resident growth {} unlogged, {} kept; the \
         log {bytes} bytes, {} per dispatch",
        least(&off),
        least(&kept),
        resident(&off),
        resident(&kept),
        bytes / COST_DISPATCHES as u64
    );
}

/// What [`a_dispatch_heavy_encoder_s_log_costs_a_push_per_call`] measures of one command buffer:
/// its encode time per dispatch, the heap bytes the process took while it encoded, and the
/// process's resident growth (where the system says).
struct Held {
    nanos: f64,
    heap: usize,
    resident: Option<usize>,
}

/// The process's resident bytes, from `/proc/self/statm` (Linux; `None` elsewhere).
fn resident_bytes() -> Option<usize> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: usize = statm.split_whitespace().nth(1)?.parse().ok()?;
    // SAFETY: `sysconf` reads a system constant.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    Some(pages * usize::try_from(page).ok()?)
}

/// The dispatches [`a_dispatch_heavy_encoder_s_log_costs_a_push_per_call`] encodes each way.
const COST_DISPATCHES: usize = 4000;

/// The rounds [`a_dispatch_heavy_encoder_s_log_costs_a_push_per_call`] encodes each way.
const COST_ROUNDS: usize = 5;

/// Bytes written into a buffer's pages, each run at its offset.
type Placed = [(usize, Vec<u8>)];

/// The texels a pass drew into an 8 by 8 `RGBA8Unorm` target it clears blue, its draws `encode`d
/// with their corners from `member`, a vertex buffer over host pages that hold `old`, the command
/// buffer committed behind an event wait; where `rebound`, the buffer is rebound to pages that
/// hold `new` before the first submission, which encodes the render encoder again. `None` where
/// the device imports no host memory.
fn rebound_pass(
    device: &Device,
    (old, new): (&Placed, &Placed),
    rebound: bool,
    encode: &dyn Fn(&mut RenderCommandEncoder, &Buffer),
) -> Option<Vec<u8>> {
    let queue = device.new_command_queue().unwrap();
    let side = 8;
    let filled = |placed: &Placed| {
        let mapping = Mapping::anonymous(buffer_len());
        for (at, bytes) in placed {
            mapping.write(*at, bytes);
        }
        mapping
    };
    let seen = Arc::new(Seen::default());
    let member = buffer_over(device, filled(old), &seen)?;
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, side, side, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&target).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.clear_color = ClearColor::new(0.0, 0.0, 1.0, 1.0);
    color.store_action = StoreAction::Store;
    let event = device.new_shared_event().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::from(&event), 1)
        .unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encode(&mut encoder, &member);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    let before = device.work_counts();
    if rebound {
        let seen = Arc::new(Seen::default());
        rebind(&member, filled(new), &seen).unwrap();
        let after = device.work_counts();
        assert_eq!(
            after.encoders_reencoded,
            before.encoders_reencoded + 1,
            "the rebind encoded the render encoder again"
        );
    }
    event.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let mut texels = vec![0; (side * side * 4) as usize];
    let region = Region::new_2d(0, 0, side, side);
    target
        .get_bytes(&mut texels, side * 4, 0, region, 0, 0)
        .unwrap();
    Some(texels)
}

/// The bytes of a triangle's three corners, `x` then `y` of each.
fn corners(points: [f32; 6]) -> Vec<u8> {
    points.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A render pass committed behind an event wait, whose draw reads its corners from a vertex
/// buffer over host pages, is encoded again by a rebind before its first submission: the draw
/// reads the new pages, as Metal's does. The old pages hold corners off the target, so with no
/// rebind the draw covers nothing and the target keeps the pass's clear (blue); the new pages
/// hold a triangle over the whole target, which the fragment function writes (1/255, 0, 0, 1).
/// The fixture's pipeline (`rt6_restart.ll`) reads the corners from vertex buffer 0.
#[test]
fn a_render_pass_drawing_from_a_buffer_rebound_before_its_first_submission_reads_the_new_pages() {
    on_device(|device| {
        let pipeline = corner_pipeline(device);
        let off = [(0, corners([4.0, 4.0, 5.0, 4.0, 4.0, 5.0]))];
        let whole = [(0, corners([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]))];
        let encode = |encoder: &mut RenderCommandEncoder, member: &Buffer| {
            encoder.set_render_pipeline_state(&pipeline).unwrap();
            encoder.set_vertex_buffer(Some(member), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        };
        let Some(kept) = rebound_pass(device, (&off, &whole), false, &encode) else {
            return;
        };
        assert!(
            kept.chunks(4).all(|texel| texel == [0, 0, 255, 255]),
            "the old pages' draw covered texels: {kept:?}"
        );
        let rebound = rebound_pass(device, (&off, &whole), true, &encode).unwrap();
        assert!(
            rebound.chunks(4).all(|texel| texel == [1, 0, 0, 255]),
            "the draw did not read the new pages: {rebound:?}"
        );
        eprintln!(
            "CBF-MARK {}: a render pass drawing from a buffer rebound before its first submission \
             drew the new pages",
            device.name()
        );
    });
}

/// As [`a_render_pass_drawing_from_a_buffer_rebound_before_its_first_submission_reads_the_new_pages`],
/// with two indexed draws, each setting the pipeline state and the rebound vertex buffer again,
/// the second at another offset: the log holds each handle once for every call that names it
/// (`replay.rs`), and its indexed draws' arguments in one list (`render/replay.rs`), and the
/// encoding again runs every call with its own arguments. The new pages hold a triangle over the
/// target's left half at the first offset and one over its right half at the second, so the
/// target is drawn whole only where both draws ran again at their own offsets.
#[test]
fn indexed_draws_setting_a_rebound_buffer_again_are_each_encoded_again() {
    on_device(|device| {
        let pipeline = corner_pipeline(device);
        let indices: Vec<u8> = [0u16, 1, 2]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        let indices = buffer_with(device, &indices).unwrap();
        // The second draw's corners, past the first's 24 bytes.
        const SECOND: usize = 32;
        let away = corners([4.0, 4.0, 5.0, 4.0, 4.0, 5.0]);
        let off = [(0, away.clone()), (SECOND, away)];
        let halves = [
            (0, corners([0.0, -5.0, 0.0, 5.0, -5.0, 0.0])),
            (SECOND, corners([0.0, -5.0, 0.0, 5.0, 5.0, 0.0])),
        ];
        let encode = |encoder: &mut RenderCommandEncoder, member: &Buffer| {
            for at in [0, SECOND] {
                encoder.set_render_pipeline_state(&pipeline).unwrap();
                encoder
                    .set_vertex_buffer(Some(member), at as u64, 0)
                    .unwrap();
                encoder
                    .draw_indexed_primitives(
                        PrimitiveType::Triangle,
                        3,
                        IndexType::UInt16,
                        &indices,
                        0,
                        1,
                        0,
                        0,
                    )
                    .unwrap();
            }
        };
        let Some(kept) = rebound_pass(device, (&off, &halves), false, &encode) else {
            return;
        };
        assert!(
            kept.chunks(4).all(|texel| texel == [0, 0, 255, 255]),
            "the old pages' draws covered texels: {kept:?}"
        );
        let rebound = rebound_pass(device, (&off, &halves), true, &encode).unwrap();
        assert!(
            rebound.chunks(4).all(|texel| texel == [1, 0, 0, 255]),
            "the draws did not both read the new pages at their offsets: {rebound:?}"
        );
    });
}

/// As [`a_render_pass_drawing_from_a_buffer_rebound_before_its_first_submission_reads_the_new_pages`],
/// with calls refused at their encoding before the draw (a draw with no pipeline state set, a
/// vertex buffer past the table): the rebind encodes the render encoder again, each refused call
/// refused again and the calls after it run (`render/replay.rs`), so the draw reads the new pages
/// and the command buffer completes.
#[test]
fn a_render_pass_with_calls_refused_at_its_encoding_is_encoded_again() {
    on_device(|device| {
        let pipeline = corner_pipeline(device);
        let off = [(0, corners([4.0, 4.0, 5.0, 4.0, 4.0, 5.0]))];
        let whole = [(0, corners([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]))];
        let encode = |encoder: &mut RenderCommandEncoder, member: &Buffer| {
            assert!(encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .is_err());
            assert!(encoder
                .set_vertex_buffer(Some(member), 0, u32::MAX)
                .is_err());
            encoder.set_render_pipeline_state(&pipeline).unwrap();
            encoder.set_vertex_buffer(Some(member), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        };
        let Some(kept) = rebound_pass(device, (&off, &whole), false, &encode) else {
            return;
        };
        assert!(
            kept.chunks(4).all(|texel| texel == [0, 0, 255, 255]),
            "the old pages' draw covered texels: {kept:?}"
        );
        let rebound = rebound_pass(device, (&off, &whole), true, &encode).unwrap();
        assert!(
            rebound.chunks(4).all(|texel| texel == [1, 0, 0, 255]),
            "the draw did not read the new pages: {rebound:?}"
        );
    });
}

/// The pipeline of `rt6_restart.ll`: corners from vertex buffer 0, each fragment
/// (1/255, 0, 0, 1) into an `RGBA8Unorm` target.
fn corner_pipeline(device: &Device) -> RenderPipelineState {
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
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A pending command buffer that a submission check fails (the consumer's emulated detach,
/// `Injected(8)`, as Metal's code 8) keeps that code where its encoder also named a buffer
/// rebound since: the encoder runs nothing, and the device announces no
/// `Degradation::EncodedUseRebound`, which only a command buffer that fails with it announces.
#[test]
fn a_command_buffer_a_check_failed_keeps_its_code_and_announces_no_rebound_use() {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        let copied = buffer_with(device, &[0x5a; 16]).unwrap();
        let event = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&member, 0, &copied, 0, 16)
            .unwrap();
        blit.end_encoding().unwrap();
        let check: SubmissionCheck = Arc::new(|_: &Submission<'_>| NonZeroU64::new(8));
        command_buffer.set_submission_check(check).unwrap();
        command_buffer.commit().unwrap();
        drop(rebind_to(&member, B));
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
        assert_eq!(
            command_buffer.error(),
            Some(CommandBufferError::Injected(8))
        );
        assert_eq!(CommandBufferError::Injected(8).metal_code(), 8);
        assert_eq!(
            read_buffer(&copied, 0, 16).unwrap(),
            [0x5a; 16],
            "the blit ran"
        );
        let notices = assert_made_once(messages, &[], &shared_kinds(device));
        assert!(
            !notices.contains_key(&Degradation::EncodedUseRebound),
            "announced EncodedUseRebound for a command buffer failed with code 8"
        );
        eprintln!("REBIND-REACH checked: code 8 kept, no rebound-use notice");
    });
}

/// A command buffer split by an event signal into Metal's submissions A, a blit that uses no
/// member, and B, a blit that names the member, committed behind a wait; the member is rebound,
/// then detached as the consumer detaches (its submission check fails a submission that uses it
/// with code 8), while the command buffer waits. A runs; B runs nothing and the command buffer
/// keeps B's check's code 8, with no `Degradation::EncodedUseRebound` notice (Metal: A runs, B
/// fails with code 8). Rebound and not detached, B's check passes it: A runs, and B, encoded again
/// at the rebind (It failed `EncodedUseRebound` before), copies the new node, and the command
/// buffer completes, with no notice. Either way the pages are fixed at the first submission for
/// every later batch, and B's submission is decided at its own batch, after its check.
#[test]
fn a_split_command_buffer_decides_each_rebound_submission_after_its_own_check() {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        let split = |detached: bool| {
            let ran = device.new_buffer(16, ResourceOptions::shared()).unwrap();
            let copied = buffer_with(device, &[0x5a; 16]).unwrap();
            let event = device.new_shared_event().unwrap();
            let between = device.new_shared_event().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            command_buffer
                .encode_wait_for_event(EventRef::from(&event), 1)
                .unwrap();
            let mut first = command_buffer.blit_command_encoder().unwrap();
            first.fill_buffer(&ran, 0..16, 0xa5).unwrap();
            first.end_encoding().unwrap();
            command_buffer
                .encode_signal_event(EventRef::from(&between), 1)
                .unwrap();
            let mut second = command_buffer.blit_command_encoder().unwrap();
            second
                .copy_from_buffer_to_buffer(&member, 0, &copied, 0, 16)
                .unwrap();
            second.end_encoding().unwrap();
            let gone = Arc::new(AtomicBool::new(false));
            let check: SubmissionCheck = {
                let gone = Arc::clone(&gone);
                let id = member.object_id();
                Arc::new(move |submission: &Submission<'_>| {
                    let uses = submission
                        .used_resources()
                        .buffers
                        .iter()
                        .any(|buffer| buffer.object_id() == id);
                    (uses && gone.load(Ordering::SeqCst)).then(|| NonZeroU64::new(8).unwrap())
                })
            };
            command_buffer.set_submission_check(check).unwrap();
            command_buffer.commit().unwrap();
            drop(rebind_to(&member, B));
            gone.store(detached, Ordering::SeqCst);
            event.set_signaled_value(1);
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(read_buffer(&ran, 0, 16).unwrap(), [0xa5; 16], "A ran");
            assert_eq!(between.signaled_value(), 1, "the signal between them");
            let copied = read_buffer(&copied, 0, 16).unwrap();
            (command_buffer.status(), command_buffer.error(), copied)
        };
        assert_eq!(
            split(true),
            (
                CommandBufferStatus::Error,
                Some(CommandBufferError::Injected(8)),
                vec![0x5a; 16]
            ),
            "B ran nothing"
        );
        let notices = assert_made_once(messages, &[], &shared_kinds(device));
        assert!(
            !notices.contains_key(&Degradation::EncodedUseRebound),
            "announced EncodedUseRebound for a submission its check failed with code 8"
        );
        assert_eq!(
            split(false),
            (CommandBufferStatus::Completed, None, node(B)),
            "B encoded again copied the new node"
        );
        assert_made_once(messages, &[], &shared_kinds(device));
        eprintln!("REBIND-REACH split: A ran, B's check's code 8 kept; unchecked, B encoded again");
    });
}

/// How [`rebound_after_a_failure`]'s command buffer fails X's submission with code 8.
#[derive(Clone, Copy)]
enum FirstFailed {
    /// As the consumer's emulated detach does: a submission check fails what names X. X's
    /// submission comes first, the fill's after it.
    Check,
    /// As `commit_failed` does, from X's encoder on (`FailedEncoders::From`). The fill's
    /// submission comes first, before that encoder.
    CommitFailed,
}

/// A command buffer split by an event signal into two of Metal's submissions, a blit that copies
/// out of X and a fill of Y, in the order `how` says, committed behind a wait; X's submission fails
/// with code 8 as `how` says, and Y is rebound while it waits. Metal fails X's submission with code
/// 8 and runs the fill over Y's new pages. The runtime keeps code 8 for the command buffer, and the
/// fill, encoded again at the rebind, writes Y's new pages, with no notice: it once ran nothing,
/// announced (`Degradation::EncodedUseRebound`), and before that a command buffer that had failed
/// already failed the fill silently.
fn rebound_after_a_failure(how: FirstFailed) {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let Some((x, _)) = member_over(device, A) else {
            return;
        };
        let Some((y, _)) = member_over(device, A) else {
            return;
        };
        let copied = buffer_with(device, &[0x5a; 16]).unwrap();
        let event = device.new_shared_event().unwrap();
        let between = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let copy = || {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(&x, 0, &copied, 0, 16)
                .unwrap();
            blit.end_encoding().unwrap();
        };
        let fill = || {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&y, 0..16, 0x77).unwrap();
            blit.end_encoding().unwrap();
        };
        let signal = || {
            command_buffer
                .encode_signal_event(EventRef::from(&between), 1)
                .unwrap();
        };
        match how {
            FirstFailed::Check => {
                copy();
                signal();
                fill();
                let id = x.object_id();
                let check: SubmissionCheck = Arc::new(move |submission: &Submission<'_>| {
                    let uses = submission
                        .used_resources()
                        .buffers
                        .iter()
                        .any(|buffer| buffer.object_id() == id);
                    uses.then(|| NonZeroU64::new(8).unwrap())
                });
                command_buffer.set_submission_check(check).unwrap();
                command_buffer.commit().unwrap();
            }
            FirstFailed::CommitFailed => {
                fill();
                signal();
                copy();
                let mut failure = CommitFailure::new(8);
                failure.encoders = FailedEncoders::From(1);
                command_buffer.commit_failed(failure).unwrap();
            }
        }
        let notices = assert_made_once(messages, &[], &shared_kinds(device));
        assert!(
            !notices.contains_key(&Degradation::EncodedUseRebound),
            "announced before the rebind"
        );
        drop(rebind_to(&y, B));
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
        assert_eq!(
            command_buffer.error(),
            Some(CommandBufferError::Injected(8)),
            "the first code kept"
        );
        assert_eq!(
            read_buffer(&copied, 0, 16).unwrap(),
            [0x5a; 16],
            "X's copy ran"
        );
        assert_eq!(between.signaled_value(), 1, "the signal between them");
        assert_eq!(
            read_buffer(&y, 0, 16).unwrap(),
            [0x77; 16],
            "the fill ran over Y's new pages"
        );
        let notices = assert_made_once(messages, &[], &shared_kinds(device));
        assert!(!notices.contains_key(&Degradation::EncodedUseRebound));
    });
}

/// [`rebound_after_a_failure`], X's submission failed by the consumer's submission check.
#[test]
fn a_rebound_submission_after_a_checks_code_8_runs_over_the_new_pages() {
    rebound_after_a_failure(FirstFailed::Check);
    eprintln!("REBIND-REACH after a check: code 8 kept; the rebound fill ran over the new pages");
}

/// [`rebound_after_a_failure`], X's encoder failed by `commit_failed`, the fill before it.
#[test]
fn a_rebound_submission_before_commit_faileds_encoders_runs_over_the_new_pages() {
    rebound_after_a_failure(FirstFailed::CommitFailed);
    eprintln!(
        "REBIND-REACH after commit_failed: code 8 kept; the rebound fill ran over the new pages"
    );
}

/// A rebind between two submissions of one command buffer (its wait for an event after the
/// walker's encoder splits it), as Metal answers it on the M3 Max (macOS 27.0.1): the later
/// submission, behind the wait, runs over the new pages with no error. Its blit, which names the
/// buffer, is encoded again by the rebind and copies the new node. Its walker, which made the
/// buffer resident, reaches it through the table the first submission fixed, and reads the old
/// node where Metal's reads the new one (`docs/FAST-PATHS.md`, known gaps), the old pages' import
/// held, and their deallocator waiting, until the command buffer completed. A command buffer
/// encoded after the rebind reads the new pages.
#[test]
fn a_rebind_between_submissions_reaches_the_new_pages_in_the_later_named_use() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let outputs = Outputs::new(device, &queue, &walker);
        let Some((member, old)) = member_over(device, A) else {
            return;
        };
        let (binding, offset, length) = output(&walker);
        let state = pipeline_state(device, &walker).unwrap();
        let mut holding = CaseResources::new(device, &walker).unwrap();
        let [field] = holding.argument_buffers.keys().copied().collect::<Vec<_>>()[..] else {
            panic!("{POINTER_WALKER}: one argument-buffer buffer");
        };
        holding.argument_buffers.insert(field, member.clone());
        let first = bound_buffers(device, &walker.compute, &holding).unwrap();
        let mut later = first.clone();
        let fresh = read_buffer(&first[&binding], 0, first[&binding].length()).unwrap();
        later.insert(binding, buffer_with(device, &fresh).unwrap());
        let copied = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        let event = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encode_case(&mut encoder, &state, &walker.compute, &first, &holding).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&member, 0, &copied, 0, 16)
            .unwrap();
        blit.end_encoding().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encode_case(&mut encoder, &state, &walker.compute, &later, &holding).unwrap();
        encoder.end_encoding().unwrap();
        let before = batches_submitted(device);
        command_buffer.commit().unwrap();
        first_batch_submitted(device, before);
        drop(rebind_to(&member, B));
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(
            old.runs.load(Ordering::SeqCst),
            0,
            "the old pages went while a submitted command buffer held them"
        );
        assert_eq!(command_buffer.status(), CommandBufferStatus::Committed);
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let read = |buffers: &std::collections::BTreeMap<u32, Buffer>| {
            read_buffer(&buffers[&binding], offset, length).unwrap()
        };
        assert!(read(&first) == outputs.a, "the first batch's walk");
        // The known gap: Metal's later walk reads the new node.
        assert!(read(&later) == outputs.a, "the later batch's walk");
        assert_eq!(read_buffer(&copied, 0, 16).unwrap(), node(B), "its blit");
        deallocated(&old);
        drop(holding);
        let after = walk(
            device,
            &queue,
            &walker,
            &member,
            Uses::Resident,
            false,
            || {},
        );
        assert_eq!(after.command_buffer.error(), None);
        assert!(
            after.produced == outputs.b,
            "a command buffer encoded after"
        );
        eprintln!(
            "REBIND-REACH between submissions: the later blit encoded again over the new pages"
        );
    });
}

/// A no-copy buffer released while a command buffer whose first batch was submitted holds the
/// table that lists it keeps its pages, its deallocator waiting for that command buffer to
/// complete (the submitted batches hold their table), which the device announces once
/// (`host-deallocator-delayed`: Metal's runs at the release, probe `dealloc-probe4`); one
/// released beside it, which no submitted table lists, has its deallocator run while that command
/// buffer still waits. (The G344 re-land's
/// tests marked a table by hand and ran a spinning kernel on a host-written flag, which an Apple
/// GPU never saw; an event wait between two batches holds the command buffer here.)
#[test]
fn a_released_buffer_s_pages_wait_for_the_submitted_command_buffer_whose_table_lists_it() {
    released_while_listed(false);
}

/// A no-copy buffer given a keep-alive, released while a command buffer whose first batch was
/// submitted holds the table that lists it, runs its deallocator at the release, as Metal's runs
/// (probe `dealloc-probe4`), unannounced: the import of its pages keeps the keep-alive, which
/// keeps them mapped for that command buffer's walk, until it is freed after the command buffer
/// completes.
#[test]
fn a_released_buffer_s_keep_alive_holds_its_listed_pages_while_its_deallocator_runs_at_once() {
    released_while_listed(true);
}

/// The release of a no-copy buffer a submitted command buffer's table lists, with a keep-alive
/// or without one, and of one beside it that no submitted table lists.
fn released_while_listed(keep_alive: bool) {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let outputs = Outputs::new(device, &queue, &walker);
        let made = match keep_alive {
            true => kept_member_over(device, A),
            false => member_over(device, A)
                .map(|(buffer, seen)| (buffer, seen, Arc::new(AtomicBool::new(false)))),
        };
        let Some((listed, listed_seen, kept_dropped)) = made else {
            return;
        };
        gpu_read(&queue, &listed, 0..16);
        let (binding, offset, length) = output(&walker);
        let state = pipeline_state(device, &walker).unwrap();
        let mut holding = CaseResources::new(device, &walker).unwrap();
        let [field] = holding.argument_buffers.keys().copied().collect::<Vec<_>>()[..] else {
            panic!("{POINTER_WALKER}: one argument-buffer buffer");
        };
        holding.argument_buffers.insert(field, listed.clone());
        let buffers = bound_buffers(device, &walker.compute, &holding).unwrap();
        drop(holding.argument_buffers.remove(&field));
        let unrelated = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        let event = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encode_case(&mut encoder, &state, &walker.compute, &buffers, &holding).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&unrelated, 0..16, 0x3c).unwrap();
        blit.end_encoding().unwrap();
        let before = batches_submitted(device);
        command_buffer.commit().unwrap();
        first_batch_submitted(device, before);
        let before = device.work_counts();
        drop(listed);
        let after = device.work_counts();
        let delayed = after.host_deallocators_delayed - before.host_deallocators_delayed;
        let kept = after.host_keep_alives_kept - before.host_keep_alives_kept;
        match keep_alive {
            // The deallocator runs in the release; the import keeps the keep-alive.
            true => {
                assert_eq!(delayed, 0, "a deallocator with a keep-alive waited");
                assert_eq!(kept, 1, "the import keeps the keep-alive");
                assert_eq!(
                    listed_seen.runs.load(Ordering::SeqCst),
                    1,
                    "the deallocator did not run at the release, as Metal's runs"
                );
            }
            false => {
                assert_eq!(delayed, 1, "the release waits for the submitted table");
                std::thread::sleep(Duration::from_millis(50));
                assert_eq!(
                    listed_seen.runs.load(Ordering::SeqCst),
                    0,
                    "ran though a submitted command buffer's table listed it"
                );
            }
        }
        // Beside it: a buffer listed after that submission, released; read on another queue,
        // since this one's next command buffer waits behind the held one.
        let Some((beside, beside_seen)) = member_over(device, B) else {
            return;
        };
        let other = device.new_command_queue().unwrap();
        gpu_read(&other, &beside, 0..16);
        drop(beside);
        deallocated(&beside_seen);
        assert_eq!(command_buffer.status(), CommandBufferStatus::Committed);
        let ran = usize::from(keep_alive);
        assert_eq!(listed_seen.runs.load(Ordering::SeqCst), ran);
        assert!(
            !kept_dropped.load(Ordering::SeqCst),
            "the keep-alive went while a submitted command buffer's table listed its pages"
        );
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let produced = read_buffer(&buffers[&binding], offset, length).unwrap();
        assert!(produced == outputs.a, "its walk read the listed pages");
        assert_eq!(read_buffer(&unrelated, 0, 16).unwrap(), [0x3c; 16]);
        deallocated(&listed_seen);
        assert_eq!(listed_seen.runs.load(Ordering::SeqCst), 1);
        if keep_alive {
            // The import is freed as the command buffer lets go of its table, and drops the
            // keep-alive (a runtime thread may free it).
            let deadline = Instant::now() + WAIT;
            while !kept_dropped.load(Ordering::SeqCst) {
                assert!(
                    Instant::now() < deadline,
                    "the keep-alive was never dropped"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            assert_eq!(
                device.work_counts().host_keep_alives_kept,
                before.host_keep_alives_kept
            );
            // No wait past Metal's release, nothing announced.
            assert_made_once(messages, &[], &shared_kinds(device));
            eprintln!("REBIND-REACH released: the keep-alive held the listed pages");
            return;
        }
        // The wait past Metal's release, announced once; the buffer beside it waited for none.
        assert_made_once(messages, &DELAYED_DEALLOCATORS, &shared_kinds(device));
        eprintln!("REBIND-REACH released: the pages waited for the submitted table's holder");
    });
}

/// A no-copy buffer that a submitted command buffer's command names, released while that command
/// buffer waits, goes as it completes, as Metal's goes (Metal holds what an encoder was handed
/// until its command buffer completes: probe `dealloc-probe`, the M3 Max under macOS 27.0.1). That
/// wait is no longer than Metal's, so it is not announced, though the command buffer's own table
/// lists the buffer too: the command buffer lets go of its tables before what it retained, so only
/// another command buffer's table counts as holding the pages past the release.
#[test]
fn a_named_buffer_released_while_its_command_buffer_waits_goes_unannounced_as_it_completes() {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let Some((named, named_seen)) = member_over(device, A) else {
            return;
        };
        // Used, so the command buffer's table lists it (`live_buffers.rs`).
        gpu_read(&queue, &named, 0..16);
        let (binding, ..) = output(&walker);
        let state = pipeline_state(device, &walker).unwrap();
        let holding = CaseResources::new(device, &walker).unwrap();
        let mut buffers = bound_buffers(device, &walker.compute, &holding).unwrap();
        buffers.insert(binding, named.clone());
        let event = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encode_case(&mut encoder, &state, &walker.compute, &buffers, &holding).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let before = batches_submitted(device);
        command_buffer.commit().unwrap();
        first_batch_submitted(device, before);
        let before = device.work_counts();
        drop(buffers);
        drop(named);
        assert_eq!(
            named_seen.runs.load(Ordering::SeqCst),
            0,
            "ran while its command buffer held it"
        );
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        deallocated(&named_seen);
        assert_eq!(
            device.work_counts().host_deallocators_delayed - before.host_deallocators_delayed,
            0,
            "counted a wait no longer than Metal's"
        );
        assert_made_once(messages, &[], &shared_kinds(device));
    });
}

/// A no-copy buffer released after a rebind encoded a pending command buffer again, before that
/// command buffer's first submission, goes at that submission, before the command buffer
/// completes, as Metal lets go at once of a buffer no submitted command buffer holds (its
/// deallocator runs in the release, while such a command buffer waits committed: probe
/// `dealloc-probe3`, the M3 Max under macOS 27.0.1). Its deallocator may read that command
/// buffer's status. The encoding again of the walker's encoder, which names the rebound member,
/// held the live buffers' table that lists the released buffer from then on (`reencode.rs`); the
/// first submission makes the resident set again of the list without it and lets go of that
/// table, its pages' last hold. It let go of it on the thread that submits, under its locks and
/// the command buffer's: the deallocator ran there, and its read of the status never returned.
/// The wait from the release to that submission, past Metal's, is announced once
/// (`host-deallocator-delayed`).
#[test]
fn a_buffer_released_after_a_rebind_before_the_first_submission_goes_at_once() {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        // Its pipelines made, so the commit records the walker's encoder, its dispatch at once.
        let plain = buffer_with(device, &node(A)).unwrap();
        walk(
            device,
            &queue,
            &walker,
            &plain,
            Uses::Resident,
            false,
            || {},
        );
        let (binding, ..) = output(&walker);
        let state = pipeline_state(device, &walker).unwrap();
        let holding = CaseResources::new(device, &walker).unwrap();
        let mut buffers = bound_buffers(device, &walker.compute, &holding).unwrap();
        buffers.insert(binding, member.clone());
        let command_buffer = queue.command_buffer().unwrap();
        let mapping = pages(buffer_len(), 0, &node(C));
        let pointer = mapping.pointer();
        let (sender, receiver) = mpsc::channel();
        let deallocator: Deallocator = {
            let command_buffer = command_buffer.clone();
            Box::new(move |_: NonNull<u8>, _: u64| {
                let status = command_buffer.status();
                drop(command_buffer);
                drop(mapping);
                sender.send(status).unwrap();
            })
        };
        // SAFETY: the deallocator releases the mapping, after the buffer's last use.
        let made = unsafe {
            no_copy(
                device,
                pointer,
                buffer_len(),
                ResourceOptions::default(),
                Some(deallocator),
            )
        };
        let released = match made {
            Ok(buffer) => buffer,
            Err(error) if not_importable(&error) => return,
            Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
        };
        // Used, so every later command buffer's commands reach it unnamed (`live_buffers.rs`).
        gpu_read(&queue, &released, 0..16);
        let (started, finished) = (
            device.new_shared_event().unwrap(),
            device.new_shared_event().unwrap(),
        );
        command_buffer
            .encode_wait_for_event(EventRef::from(&started), 1)
            .unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encode_case(&mut encoder, &state, &walker.compute, &buffers, &holding).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&finished), 1)
            .unwrap();
        command_buffer.commit().unwrap();
        let before = device.work_counts();
        rebind_to(&member, B);
        assert_eq!(
            device.work_counts().encoders_reencoded - before.encoders_reencoded,
            1,
            "the walker's encoder encoded again"
        );
        drop(released);
        started.set_signaled_value(1);
        let status = receiver
            .recv_timeout(WAIT)
            .expect("the deallocator ran and returned");
        assert_eq!(status, CommandBufferStatus::Committed);
        finished.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        assert_made_once(messages, &DELAYED_DEALLOCATORS, &shared_kinds(device));
    });
}

/// The cases of [`a_dispatch_counts_into_the_pages_its_buffer_lies_in_at_the_first_submission`]:
/// whether the dispatch lies inside an If, and whether it runs (the If's condition holds).
const REBOUND_DISPATCHES: [(bool, bool); 3] = [(false, true), (true, true), (true, false)];

/// A compute dispatch counting into a buffer over host pages, plain or inside an If (its encoder's
/// unit of command predication alone), in a command buffer committed behind an event wait, its
/// encoder ended, then the buffer rebound to other pages before the wait's signal: the dispatch
/// counts into the new pages and the command buffer completes, as Metal's does (the consumer's
/// `rebound_dispatch`, plain and inside an If whose condition holds, the M3 Max under macOS 27.0.1:
/// status 4, no error, the old pages' word 0, the new pages' 1); inside an If whose condition
/// fails it runs nothing, the new pages' word 0. Each case with the dispatch's pipeline still being
/// made at its encoding (a state's first dispatch hands its making to the device's pool:
/// `WorkCounts::dispatch_pipelines_pooled`), and made. The rebind encodes the encoder again
/// (`WorkCounts::encoders_reencoded`), its If among its calls, and refuses nothing.
#[test]
fn a_dispatch_counts_into_the_pages_its_buffer_lies_in_at_the_first_submission() {
    // A device of its own each, whose first state's first dispatch hands its making to the pool.
    for (predicated, runs) in REBOUND_DISPATCHES {
        on_device(|device| {
            let state = kernel(device, "prd_mark", false);
            for cold in [true, false] {
                let what =
                    format!("predicated {predicated}, runs {runs}, pipeline being made {cold}");
                // Each mapping lives with its deallocator, which notes its first bytes as it runs.
                let seen = Arc::new(Seen::default());
                let Some(buffer) = buffer_over(device, pages(buffer_len(), 0, &[0; 4]), &seen)
                else {
                    return;
                };
                let condition = buffer_with(device, &u32::from(runs).to_le_bytes()).unwrap();
                let queue = device.new_command_queue().unwrap();
                let gate = device.new_shared_event().unwrap();
                let command_buffer = queue.command_buffer().unwrap();
                command_buffer
                    .encode_wait_for_event(EventRef::from(&gate), 1)
                    .unwrap();
                let before = device.work_counts();
                let mut encoder = command_buffer.compute_command_encoder().unwrap();
                encoder.set_compute_pipeline_state(&state).unwrap();
                encoder.set_bytes(&0u32.to_le_bytes(), 0).unwrap();
                encoder.set_buffer(Some(&buffer), 0, 1).unwrap();
                if predicated {
                    encoder.encode_start_if(Some(&condition), 0, 0, 1).unwrap();
                }
                let one = Size::new(1, 1, 1);
                encoder.dispatch_threads(one, one).unwrap();
                if predicated {
                    assert!(encoder.encode_end_if().unwrap(), "{what}");
                }
                encoder.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                let pooled = device.work_counts().dispatch_pipelines_pooled
                    - before.dispatch_pipelines_pooled;
                assert_eq!(
                    pooled,
                    u64::from(cold),
                    "{what}: its making handed to the pool"
                );
                let new_seen = Arc::new(Seen::default());
                let rebound = rebind(&buffer, pages(buffer_len(), 0, &[0; 4]), &new_seen);
                assert_eq!(rebound, Ok(()), "{what}: the rebind");
                gate.set_signaled_value(1);
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                let after = device.work_counts();
                let word = |bytes: &[u8]| u32::from_le_bytes(bytes[..4].try_into().unwrap());
                let new_word = word(&read_buffer(&buffer, 0, 4).unwrap());
                // The old pages go once nothing holds them, their word as the GPU left it.
                deallocated(&seen);
                let old_word = word(&seen.bytes.lock().unwrap());
                let seen_now = (
                    command_buffer.status(),
                    command_buffer.error(),
                    old_word,
                    new_word,
                    after.encoders_reencoded - before.encoders_reencoded,
                    after.reencodes_refused - before.reencodes_refused,
                );
                let counted = u32::from(runs);
                assert_eq!(
                    seen_now,
                    (CommandBufferStatus::Completed, None, 0, counted, 1, 0),
                    "{what}: (status, error, the old pages' word, the new pages', encoders encoded \
                     again, refused)"
                );
            }
            eprintln!(
                "REBOUND-DISPATCH {}: a dispatch (predicated {predicated}, runs {runs}) rebound \
                 before the first submission counted into the new pages as its block said, its \
                 pipeline being made or made",
                device.name()
            );
        });
    }
}
