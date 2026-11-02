//! Device tests of buffers in the slots of an address pool (`Device::reserve_address_pool`,
//! `BufferDescriptor::with_pool_slot`; the consumer's ask "sub-page placement") on the device the
//! cfg selects, under the validation layer:
//!
//! - A pool holds a pin's span of its length; buffers in its slots answer the pool's address
//!   plus their offsets and hold their lengths rounded up to 256 bytes, so several share a page,
//!   as Metal's buffers shorter than a page do (the consumer's `vaprobe smallpool`, an M3 Max);
//!   slots that meet a live one, are off a 256-byte offset or pass the pool's end are refused
//!   typed; a dropped pool's span stays held until its last slot's buffer is gone.
//! - A pointer into a slot reaches that slot's buffer, whose neighbours in the page are other
//!   buffers: a kernel walks a pointer chain from an argument buffer through slots of one page,
//!   the slots named resident or reached unnamed, and indirect commands bind slots of one page.
//! - A slot freed while a committed command buffer uses its buffer is taken again once the
//!   buffer's range token drops, from its drop on, and the new buffer there holds its own bytes.

use super::host_memory::{not_importable, on_device, run, Mapping, GUEST_PAGE};
use super::{
    bound_buffers, compute_descriptor, encode_case, pipeline_state, public_case, read_buffer,
    CaseResources, PublicCase,
};
use crate::air_library::AirLibrary;
use crate::candidate::engine::ComputeOutput;
use metal2vulkan::metal::{IndirectCommandType, ResourceUsage};
use metal2vulkan::pipeline::LinkedFunctions;
use metal2vulkan_runtime::{
    AddressPlacement, AddressPool, AddressRange, Buffer, BufferContents, BufferDescriptor,
    CommandBufferStatus, ComputePipelineState, Device, Error, GpuAddress, HostMemory,
    IndirectCommandBufferDescriptor, PipelineOptions, PlacementConflict, RangeToken,
    ResourceOptions, Size,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// Metal's page.
const PAGE: u64 = 16 << 10;

/// The length of Metal's pools, and the span of the space one holds (a pin's: its pages, one
/// guard page, rounded up to 32 KiB).
const POOL: u64 = 0x20000;
const SPAN: u64 = 0x28000;

/// An address high in the space, `[2^32, 2^47)`, far from the ranges the runtime picks first.
const HIGH: u64 = 0x7000_0000_0000;

/// Walks a pointer chain from the address its argument buffer (buffer 0) holds at offset 0:
/// `struct Node { device uint *next; uint value; }`, two steps, the sum of the values it visits
/// and the last node's again, in each of its 4 threads' words.
const POINTER_WALKER: &str = "walk-a-device-pointer-chain-through-a-loop-phi";

/// One thread copies a 32-bit word from its buffer 0 into its buffer 1.
const WORD_COPIER: &str = "single-thread-copy-word";

fn at(address: u64) -> AddressPlacement {
    AddressPlacement::At(GpuAddress(address))
}

/// The Metal address space, `[2^32, 2^47)` (the runtime's `address.rs`).
const SPACE: std::ops::Range<u64> = (1 << 32)..(1 << 47);

/// A pool holding the whole Metal address space of `device` (its pages and guard page end at the
/// space's end), checked: a buffer the runtime places in the space is refused `OutOfMemory` while
/// it lives, and one outside the space (`AddressPlacement::Outside`) is made.
pub(super) fn holding_the_space(device: &Device) -> AddressPool {
    let pool = device
        .reserve_address_pool(at(SPACE.start), SPACE.end - SPACE.start - PAGE)
        .unwrap();
    let zero = |placement| {
        device.new_buffer_with_descriptor(
            BufferDescriptor::new(PAGE, ResourceOptions::shared(), BufferContents::Zero)
                .with_placement(placement),
        )
    };
    assert!(
        matches!(zero(AddressPlacement::Any), Err(Error::OutOfMemory(_))),
        "a buffer in the space while it is held"
    );
    zero(AddressPlacement::Outside).unwrap();
    pool
}

/// A `Shared` buffer holding `bytes` in the slot at `offset` of `pool`.
fn slot(pool: &AddressPool, offset: u64, bytes: &[u8]) -> Result<Buffer, Error> {
    pool.device().new_buffer_with_descriptor(
        BufferDescriptor::new(
            bytes.len() as u64,
            ResourceOptions::shared(),
            BufferContents::Copy(bytes),
        )
        .with_pool_slot(pool, offset),
    )
}

/// A `Shared` buffer of `length` zero bytes in the slot at `offset` of `pool`.
fn zero_slot(pool: &AddressPool, offset: u64, length: u64) -> Result<Buffer, Error> {
    pool.device().new_buffer_with_descriptor(
        BufferDescriptor::new(length, ResourceOptions::shared(), BufferContents::Zero)
            .with_pool_slot(pool, offset),
    )
}

/// A buffer of `length` zero bytes pinned at `address`.
fn pinned(device: &Device, address: u64, length: u64) -> Result<Buffer, Error> {
    device.new_buffer_with_descriptor(
        BufferDescriptor::new(length, ResourceOptions::shared(), BufferContents::Zero)
            .with_placement(at(address)),
    )
}

fn conflict(error: Error) -> PlacementConflict {
    match error {
        Error::PlacementConflict(conflict) => conflict,
        other => panic!("not a placement conflict: {other:?}"),
    }
}

/// A pool holds a pin's span of its length, and buffers in its slots answer its address plus
/// their offsets, holding their lengths rounded up to 256 bytes (Metal's measured sizes: 1 and
/// 0x100 bytes 0x100, 0x101 bytes 0x200, 0x1001 bytes 0x1100, 0x3fff bytes 0x4000), their bytes
/// their own though they share pages. What a slot cannot hold is refused typed, and a dropped
/// pool's span is held until its last slot's buffer is gone.
#[test]
fn buffers_in_a_pool_s_slots_share_pages_at_their_addresses() {
    on_device(|device| {
        let pool = device.reserve_address_pool(at(HIGH), POOL).unwrap();
        assert_eq!(
            (pool.gpu_address(), pool.length()),
            (GpuAddress(HIGH), POOL)
        );
        // The span holds pins and picks off: its guard page refuses a pin, the page after it
        // takes one, and a buffer the runtime places lands outside it.
        assert!(matches!(
            conflict(pinned(device, HIGH + POOL, PAGE).unwrap_err()),
            PlacementConflict::Held { .. }
        ));
        drop(pinned(device, HIGH + SPAN, PAGE).unwrap());
        let picked = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        let address = picked.gpu_address().0;
        assert!(
            address + PAGE <= HIGH || address >= HIGH + SPAN,
            "{address:#x}"
        );
        // Slots one after another, each holding its own bytes.
        let mut offset = 0;
        let mut slots = Vec::new();
        for (length, size) in [
            (1u64, 0x100),
            (0x40, 0x100),
            (0x100, 0x100),
            (0x101, 0x200),
            (0x1001, 0x1100),
            (0x3fff, 0x4000),
        ] {
            let bytes: Vec<u8> = (0..length).map(|i| (i + offset / 0x100) as u8).collect();
            let buffer = slot(&pool, offset, &bytes).unwrap();
            assert_eq!(
                buffer.gpu_address(),
                GpuAddress(HIGH + offset),
                "{length:#x}"
            );
            slots.push((buffer, bytes));
            offset += size;
        }
        for (buffer, bytes) in &slots {
            assert_eq!(&read_buffer(buffer, 0, buffer.length()).unwrap(), bytes);
        }
        // A no-copy buffer in a slot: the slot's address, its own bytes, and no more of the pool.
        let memory = Mapping::anonymous(GUEST_PAGE);
        // SAFETY: the mapping outlives the buffer made over it, which is shorter than it.
        let host = unsafe { HostMemory::new(memory.pointer()) };
        let contents = BufferContents::NoCopy {
            memory: host,
            deallocator: None,
            keep_alive: None,
        };
        let over = BufferDescriptor::new(0x100, ResourceOptions::shared(), contents)
            .with_pool_slot(&pool, 0x8000);
        match device.new_buffer_with_descriptor(over) {
            Ok(over) => {
                assert_eq!(over.gpu_address(), GpuAddress(HIGH + 0x8000));
                assert_eq!(over.contents(), Some(memory.pointer()));
                drop(zero_slot(&pool, 0x8100, 0x100).unwrap());
                drop(over);
            }
            Err(error) => {
                assert!(not_importable(&error), "{error:?}");
                println!("POOL SLOTS {}: no-copy refused {error:?}", device.name());
            }
        }
        // The next slot right after the last one's rounded bytes; one that meets the run of
        // slots held, one off a 256-byte offset, one placed elsewhere and one past the pool's
        // end are refused.
        let next = zero_slot(&pool, offset, 0x100).unwrap();
        let PlacementConflict::Held { requested, held } =
            conflict(zero_slot(&pool, offset - 0x100, 1).unwrap_err())
        else {
            panic!("a slot meeting held ones: not Held");
        };
        assert!(is(requested, HIGH + offset - 0x100, 0x100), "{requested:?}");
        assert!(is(held, HIGH, offset + 0x100), "{held:?}");
        assert!(matches!(
            conflict(zero_slot(&pool, 0x10080, 16).unwrap_err()),
            PlacementConflict::OffSlot { .. }
        ));
        let elsewhere = BufferDescriptor::new(16, ResourceOptions::shared(), BufferContents::Zero)
            .with_pool_slot(&pool, 0x10000)
            .with_placement(at(HIGH + 0x10100));
        assert!(matches!(
            conflict(device.new_buffer_with_descriptor(elsewhere).unwrap_err()),
            PlacementConflict::OffSlot { .. }
        ));
        let PlacementConflict::OutsidePool {
            requested,
            pool: whole,
        } = conflict(zero_slot(&pool, POOL - 0x100, 0x101).unwrap_err())
        else {
            panic!("a slot past the pool's end: not OutsidePool");
        };
        assert!(is(requested, HIGH + POOL - 0x100, 0x200), "{requested:?}");
        assert!(is(whole, HIGH, POOL), "{whole:?}");
        let last = zero_slot(&pool, POOL - 0x100, 0x100).unwrap();
        // Dropped while its slots' buffers live: the span is held until the last is gone.
        drop(pool);
        drop((slots, next));
        assert!(pinned(device, HIGH, PAGE).is_err(), "a slot's buffer lives");
        drop(last);
        drop(pinned(device, HIGH, POOL).unwrap());
    });
}

/// Whether `range` is the `length` bytes at `start`.
fn is(range: AddressRange, start: u64, length: u64) -> bool {
    range.start == GpuAddress(start) && range.length == length
}

/// A walker node: `next`, then `value`, 16 bytes.
fn node(next: u64, value: u32) -> Vec<u8> {
    let mut bytes = next.to_le_bytes().to_vec();
    bytes.extend(u64::from(value).to_le_bytes());
    bytes
}

/// The walker's output from a root of value 7 whose `next` is a node of value `first`, whose own
/// `next` is a node of value `second` (`first` again where null): in each thread's word.
fn walked(first: u32, second: u32) -> Vec<u8> {
    (0..4)
        .flat_map(|_| (7 + first + second).to_le_bytes())
        .collect()
}

/// A kernel walks a pointer chain from its argument buffer through buffers in slots of one page:
/// each pointer reaches its own slot's buffer, a slot's end being the next one's start, whether
/// the slots are made resident or reached unnamed (buffers of their own), and with the argument
/// buffer itself in a slot of the page.
#[test]
fn pointers_into_slots_of_one_page_reach_their_own_buffers() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let ComputeOutput::Buffer {
            binding,
            offset,
            length,
        } = walker.compute.output
        else {
            panic!("{POINTER_WALKER}: a buffer output");
        };
        let walk = pipeline_state(device, &walker).unwrap();
        let base = HIGH + 0x100_0000;
        let pool = device.reserve_address_pool(at(base), POOL).unwrap();
        // A (0x100 bytes, its end B's start) -> B; C -> D; each a buffer of its own in the page.
        let (a, b, c, d) = (0, 0x100, 0x200, 0x300);
        let mut a_bytes = node(base + b, 21);
        a_bytes.resize(0x100, 0);
        let nodes = [
            slot(&pool, a, &a_bytes).unwrap(),
            slot(&pool, b, &node(0, 34)).unwrap(),
            slot(&pool, c, &node(base + d, 55)).unwrap(),
            slot(&pool, d, &node(0, 89)).unwrap(),
        ];
        let mut owner_at = 0x400;
        for named in [true, false] {
            for (start, expected) in [
                (0, walked(21, 34)),
                (1, walked(34, 34)),
                (2, walked(55, 89)),
            ] {
                let how = format!("from node {start}, named {named}");
                let mut resources = CaseResources::new(device, &walker).unwrap();
                let [(&field, _)] = resources.argument_buffers.iter().collect::<Vec<_>>()[..]
                else {
                    panic!("{POINTER_WALKER}: one argument-buffer buffer");
                };
                resources
                    .argument_buffers
                    .insert(field, nodes[start].clone());
                let mut buffers = bound_buffers(device, &walker.compute, &resources).unwrap();
                // The argument buffer itself in a slot of the page.
                let owner = &buffers[&field.0];
                let owner_bytes = read_buffer(owner, 0, owner.length()).unwrap();
                let in_slot = slot(&pool, owner_at, &owner_bytes).unwrap();
                owner_at += 0x100;
                buffers.insert(field.0, in_slot);
                let command_buffer = queue.command_buffer().unwrap();
                let mut encoder = command_buffer.compute_command_encoder().unwrap();
                if named {
                    let resident: Vec<_> = nodes.iter().map(Buffer::as_resource).collect();
                    encoder
                        .use_resources(&resident, ResourceUsage::READ)
                        .unwrap();
                }
                encode_case(&mut encoder, &walk, &walker.compute, &buffers, &resources).unwrap();
                encoder.end_encoding().unwrap();
                run(&command_buffer);
                let output = read_buffer(&buffers[&binding], offset, length).unwrap();
                assert_eq!(output, expected, "{how}");
                let report = command_buffer.execution_report().unwrap();
                assert!(!report.resident_miss, "{how}");
            }
        }
    });
}

/// `case`'s pipeline state, made to run from indirect commands.
fn indirect_state(device: &Device, case: &PublicCase) -> ComputePipelineState {
    let library = device.new_library_with_air(&case.bitcode).unwrap();
    let function = library
        .new_function_with_name_constant_values(&case.compute.entry, &case.compute.constants)
        .unwrap();
    let mut descriptor = compute_descriptor(&function, LinkedFunctions::new(), &case.compute);
    descriptor.support_indirect_command_buffers = true;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Indirect commands bind buffers in slots of one page: each copies its input slot's word into
/// its output slot, crosswise, whether the slots are made resident or reached unnamed.
#[test]
fn indirect_commands_bind_slots_of_one_page() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let copier = public_case(WORD_COPIER).unwrap();
        let copy = indirect_state(device, &copier);
        let base = HIGH + 0x200_0000;
        let pool = device.reserve_address_pool(at(base), POOL).unwrap();
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH
            .union(IndirectCommandType::CONCURRENT_DISPATCH_THREADS);
        descriptor.max_kernel_buffer_bind_count = 2;
        for named in [true, false] {
            let words: Vec<u32> = (0..4)
                .map(|i| 0x1234_5600 + i + u32::from(named) * 16)
                .collect();
            let inputs: Vec<Buffer> = (0..4)
                .map(|i| slot(&pool, i * 0x100, &words[i as usize].to_le_bytes()).unwrap())
                .collect();
            let outputs: Vec<Buffer> = (0..4)
                .map(|i| slot(&pool, 0x400 + i * 0x100, &[0xab; 4]).unwrap())
                .collect();
            let icb = device
                .new_indirect_command_buffer(&descriptor, 4, ResourceOptions::shared())
                .unwrap();
            for i in 0..4 {
                let command = icb.indirect_compute_command_at(i as u64).unwrap();
                command.set_compute_pipeline_state(&copy).unwrap();
                command.set_kernel_buffer(Some(&inputs[i]), 0, 0).unwrap();
                command
                    .set_kernel_buffer(Some(&outputs[3 - i]), 0, 1)
                    .unwrap();
                let one = Size::new(1, 1, 1);
                command.concurrent_dispatch_threads(one, one).unwrap();
            }
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            if named {
                let resident: Vec<_> = inputs
                    .iter()
                    .chain(&outputs)
                    .map(Buffer::as_resource)
                    .collect();
                encoder
                    .use_resources(&resident, ResourceUsage::READ | ResourceUsage::WRITE)
                    .unwrap();
            }
            encoder.execute_commands_in_buffer(&icb, 0..4).unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
            for i in 0..4 {
                assert_eq!(
                    read_buffer(&outputs[3 - i], 0, 4).unwrap(),
                    words[i].to_le_bytes(),
                    "named {named}: command {i}"
                );
            }
            let report = command_buffer.execution_report().unwrap();
            assert!(!report.resident_miss, "named {named}");
        }
    });
}

/// What a slot's range token saw: whether it dropped, and the buffer made in its slot from its
/// drop.
#[derive(Default)]
struct Seen {
    dropped: AtomicBool,
    remade: Mutex<Option<Result<Buffer, Error>>>,
}

/// A range token that makes a buffer holding `bytes` in the slot at `offset` of `pool` as it
/// drops, keeping it in `seen`.
struct Token {
    pool: Arc<AddressPool>,
    offset: u64,
    bytes: Vec<u8>,
    seen: Arc<Seen>,
}

impl Drop for Token {
    fn drop(&mut self) {
        let made = slot(&self.pool, self.offset, &self.bytes);
        *self.seen.remade.lock().unwrap() = Some(made);
        self.seen.dropped.store(true, Ordering::SeqCst);
    }
}

/// A slot's buffer released while a committed command buffer fills it keeps its slot until the
/// command buffer completed, its range token dropping then; a buffer made in the slot from the
/// token's drop on is made, at the slot's address, and indirect commands copying through it read
/// its own bytes, not the fill.
#[test]
fn a_freed_slot_is_taken_again_once_its_token_drops() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let base = HIGH + 0x300_0000;
        let pool = Arc::new(device.reserve_address_pool(at(base), POOL).unwrap());
        let offset = 0x100;
        let seen = Arc::new(Seen::default());
        let token: RangeToken = Box::new(Token {
            pool: Arc::clone(&pool),
            offset,
            bytes: 0x600d_f00du32.to_le_bytes().to_vec(),
            seen: Arc::clone(&seen),
        });
        let freed = device
            .new_buffer_with_descriptor(
                BufferDescriptor::new(16, ResourceOptions::shared(), BufferContents::Zero)
                    .with_pool_slot(&pool, offset)
                    .with_range_token(token),
            )
            .unwrap();
        let beside = zero_slot(&pool, 0, 0x100).unwrap();
        let event = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_wait_for_event((&event).into(), 1)
            .unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&freed, 0..16, 0x5a).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        drop(freed);
        assert!(!seen.dropped.load(Ordering::SeqCst), "in flight");
        assert!(matches!(
            conflict(zero_slot(&pool, offset, 1).unwrap_err()),
            PlacementConflict::Held { .. }
        ));
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert!(
            seen.dropped.load(Ordering::SeqCst),
            "dropped before the wait returns"
        );
        let again = seen.remade.lock().unwrap().take().unwrap().unwrap();
        assert_eq!(again.gpu_address(), GpuAddress(base + offset));
        // Commands copy through the buffer taking the slot again: its own word, not the fill.
        let copier = public_case(WORD_COPIER).unwrap();
        let copy = indirect_state(device, &copier);
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH_THREADS;
        descriptor.max_kernel_buffer_bind_count = 2;
        let icb = device
            .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
            .unwrap();
        let command = icb.indirect_compute_command_at(0).unwrap();
        command.set_compute_pipeline_state(&copy).unwrap();
        command.set_kernel_buffer(Some(&again), 0, 0).unwrap();
        command.set_kernel_buffer(Some(&beside), 0, 1).unwrap();
        let one = Size::new(1, 1, 1);
        command.concurrent_dispatch_threads(one, one).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        run(&command_buffer);
        assert_eq!(
            read_buffer(&beside, 0, 4).unwrap(),
            0x600d_f00du32.to_le_bytes()
        );
    });
}
