//! Device tests of the runtime's blits, hazard tracking, fences, managed storage and where shared
//! storage lies, on the device the cfg selects, under the validation layer.
//!
//! The expected bytes are Metal's: copies and fills of any range, overlapping copies within one
//! buffer as `memmove`, and every refusal, as measured on Metal; a chain of compute, blit and
//! compute reproduces a public case's committed Metal output.

use super::degrade_notices::{assert_once, counts};
use super::{
    buffer_with, buffer_with_options, encode_case, layer_reported_nothing,
    layer_reported_nothing_at_all, pipeline_state, public_case, read_buffer, validated_device,
    CaseResources, Messages, ScratchCache,
};
use crate::gpu::TEST_DEVICE;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandQueue, CpuCacheMode, Degradation, Device, DeviceKind, Error,
    EventRef, HazardTrackingMode, Invalid, Problem, ResourceOptions, SharedEvent, StorageMode,
};
use std::collections::BTreeMap;
use std::time::Duration;

/// The longest a command buffer of these tests may take.
const WAIT: Duration = Duration::from_secs(60);

/// Large enough that a missing barrier between two transfers of it would show on a GPU.
const BIG: u64 = 8 << 20;

/// Large enough that a fill of it takes milliseconds on the fastest GPU of the lanes (about
/// 1 ms at the RTX 3090's 936 GB/s), well over the time between two submissions.
const RACE: u64 = 1 << 30;

fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

fn options(storage: StorageMode, hazard: HazardTrackingMode) -> ResourceOptions {
    ResourceOptions::new(storage, CpuCacheMode::DefaultCache, hazard)
}

/// `f` of `buffer`'s contents, its `length` bytes.
fn with_bytes<R>(buffer: &Buffer, f: impl FnOnce(&mut [u8]) -> R) -> R {
    let contents = buffer
        .contents()
        .expect("a host-visible buffer has contents");
    // SAFETY: the buffer holds `length` bytes at `contents`, alive with the buffer, and no
    // command buffer of these tests uses them while `f` runs.
    f(unsafe { std::slice::from_raw_parts_mut(contents.as_ptr(), buffer.length() as usize) })
}

/// A Shared buffer of `length` bytes, byte `i` holding `i % 251` (a period no power of two
/// divides, so a misplaced copy shows).
fn counting(device: &Device, length: u64) -> Buffer {
    let buffer = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    with_bytes(&buffer, |bytes| {
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = (index % 251) as u8;
        }
    });
    buffer
}

/// The bytes of `counting` from `range`.
fn counted(range: std::ops::Range<usize>) -> Vec<u8> {
    range.map(|index| (index % 251) as u8).collect()
}

/// Commit `command_buffer`, wait for it, and check it completed without error.
fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// A Shared copy of `buffer`'s bytes, read back after a blit command buffer of `queue`.
fn read_back(queue: &CommandQueue, buffer: &Buffer) -> Vec<u8> {
    let device = queue.device();
    let shared = device
        .new_buffer(buffer.length(), ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(buffer, 0, &shared, 0, buffer.length())
        .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    with_bytes(&shared, |bytes| bytes.to_vec())
}

/// Run `test` on a validated device, then check that once everything is dropped (the device
/// last) the layer reported nothing, so no Vulkan object leaked and no command was invalid.
fn on_device(test: impl FnOnce(&Device)) {
    on_device_with(|device, _| test(device));
}

/// [`on_device`], with the messages reported so far for `test` to take itself, to fail at the
/// step that caused them.
fn on_device_with(test: impl FnOnce(&Device, &Messages)) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        test(&device, &messages);
    }
    layer_reported_nothing_at_all(&messages);
}

/// Copies and fills of any offsets and sizes land exactly where Metal puts them, within one
/// encoder in encoded order.
#[test]
fn blits_copy_and_fill_any_range() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let source = counting(device, 64);
        let private = device.new_buffer(64, ResourceOptions::private()).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        // A fill of unaligned ends, then a copy into its middle at odd offsets, then a fill of
        // bytes inside one word and one of a lone tail.
        blit.fill_buffer(&private, 1..63, 0x5a).unwrap();
        blit.copy_from_buffer_to_buffer(&source, 3, &private, 21, 10)
            .unwrap();
        blit.fill_buffer(&private, 41..43, 0xee).unwrap();
        blit.fill_buffer(&private, 48..53, 0x11).unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        let mut expected = vec![0x5a; 64];
        expected[0] = 0;
        expected[63] = 0;
        expected[21..31].copy_from_slice(&counted(3..13));
        expected[41..43].fill(0xee);
        expected[48..53].fill(0x11);
        assert_eq!(read_back(&queue, &private), expected);
    });
}

/// Fills of every size, past 64 KiB, 128 KiB and 256 KiB among them, land exactly, each its words
/// as one `vkCmdFillBuffer` (`word_fill.rs`): overlapping fills of one encoder land in encoded
/// order, and the work counts show every fill's words recorded once.
#[test]
fn fills_of_every_size_land_as_one_fill_each() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let len = 600 * 1024;
        let private = device.new_buffer(len, ResourceOptions::private()).unwrap();
        let fills = [
            (0..len, 0x11),
            (3..65_540, 0x22),
            (70_001..70_001 + 3 * 65_536 + 6, 0x33),
            (300_000..300_000 + 256 * 1024 + 4, 0x44),
            (len - 9..len, 0x55),
        ];
        let before = device.work_counts();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for (range, value) in &fills {
            blit.fill_buffer(&private, range.clone(), *value).unwrap();
        }
        blit.end_encoding().unwrap();
        run(&command_buffer);
        let after = device.work_counts();
        let mut expected = vec![0u8; len as usize];
        for (range, value) in &fills {
            expected[range.start as usize..range.end as usize].fill(*value);
        }
        assert_eq!(read_back(&queue, &private), expected);
        let ran = after.fills_recorded - before.fills_recorded;
        let every = fills.len() as u64;
        eprintln!("PATH fills: {ran} recorded of {every}");
        assert_eq!(ran, every);
    });
}

/// Overlapping ranges of one buffer move as `memmove` moves them, both ways, in Shared and
/// Private memory.
#[test]
fn overlapping_copies_move_as_memmove() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let forward = counting(device, 4099);
        let backward = counting(device, 4099);
        let private = device.new_buffer(4099, ResourceOptions::private()).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&forward, 1, &forward, 7, 4000)
            .unwrap();
        blit.copy_from_buffer_to_buffer(&backward, 9, &backward, 2, 4000)
            .unwrap();
        blit.copy_from_buffer_to_buffer(&forward, 0, &private, 0, 4099)
            .unwrap();
        blit.copy_from_buffer_to_buffer(&private, 5, &private, 6, 4093)
            .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);

        let mut moved = counted(0..4099);
        moved.copy_within(1..4001, 7);
        assert_eq!(with_bytes(&forward, |bytes| bytes.to_vec()), moved);
        let mut twice = moved.clone();
        twice.copy_within(5..4098, 6);
        assert_eq!(read_back(&queue, &private), twice);
        let mut moved = counted(0..4099);
        moved.copy_within(9..4009, 2);
        assert_eq!(with_bytes(&backward, |bytes| bytes.to_vec()), moved);
    });
}

/// Tracked buffers are ordered between encoders of one command buffer and between command
/// buffers of one queue: reads after writes, writes after reads, and writes after writes.
#[test]
fn encoders_and_command_buffers_order_tracked_buffers() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let private = || device.new_buffer(BIG, ResourceOptions::private()).unwrap();
        let (a, b, c, d) = (private(), private(), private(), private());

        let command_buffer = queue.command_buffer().unwrap();
        for step in 0..4 {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            match step {
                0 => blit.fill_buffer(&a, 0..BIG, 1).unwrap(),
                1 => blit.copy_from_buffer_to_buffer(&a, 0, &b, 0, BIG).unwrap(),
                // A write after the read before it, then a read after it.
                2 => blit.fill_buffer(&a, 0..BIG, 2).unwrap(),
                _ => blit.copy_from_buffer_to_buffer(&a, 0, &c, 0, BIG).unwrap(),
            }
            blit.end_encoding().unwrap();
        }
        // Committed back to back, without a wait between them.
        let fill = queue.command_buffer().unwrap();
        let mut blit = fill.blit_command_encoder().unwrap();
        blit.fill_buffer(&c, 0..BIG / 2, 3).unwrap();
        blit.end_encoding().unwrap();
        let copy = queue.command_buffer().unwrap();
        let mut blit = copy.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&c, 0, &d, 0, BIG).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        fill.commit().unwrap();
        run(&copy);
        assert_eq!(fill.error(), None);
        assert_eq!(command_buffer.error(), None);

        assert!(read_back(&queue, &b).iter().all(|&byte| byte == 1));
        let mut expected = vec![3; BIG as usize];
        expected[BIG as usize / 2..].fill(2);
        assert_eq!(read_back(&queue, &d), expected);
    });
}

/// Untracked buffers are ordered by fences between encoders and command buffers; a wait after
/// an update in one encoder waits for the update before the encoder, as plain Metal's does: here
/// none, so it orders nothing.
#[test]
fn fences_order_untracked_buffers() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let untracked = options(StorageMode::Private, HazardTrackingMode::Untracked);
        let a = device.new_buffer(BIG, untracked).unwrap();
        let b = device.new_buffer(BIG, untracked).unwrap();
        let fence = device.new_fence();
        fence.set_label("blits");
        assert_eq!(fence.label().as_deref(), Some("blits"));
        assert_eq!(fence, fence.clone());
        assert_ne!(fence, device.new_fence());

        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&a, 0..BIG, 7).unwrap();
        blit.update_fence(&fence).unwrap();
        blit.wait_for_fence(&fence).unwrap();
        blit.end_encoding().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.wait_for_fence(&fence).unwrap();
        blit.copy_from_buffer_to_buffer(&a, 0, &b, 0, BIG).unwrap();
        blit.update_fence(&fence).unwrap();
        blit.end_encoding().unwrap();
        // A second command buffer, committed without a wait: its wait follows the update of
        // the first command buffer's last encoder.
        let next = queue.command_buffer().unwrap();
        let mut blit = next.blit_command_encoder().unwrap();
        blit.wait_for_fence(&fence).unwrap();
        blit.fill_buffer(&b, 0..BIG / 2, 8).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        run(&next);
        assert_eq!(command_buffer.error(), None);

        let mut expected = vec![8; BIG as usize];
        expected[BIG as usize / 2..].fill(7);
        // The read-back copy waits for the last update too, the program's job for untracked
        // memory.
        let shared = device.new_buffer(BIG, ResourceOptions::shared()).unwrap();
        let read = queue.command_buffer().unwrap();
        let mut blit = read.blit_command_encoder().unwrap();
        blit.wait_for_fence(&fence).unwrap();
        blit.copy_from_buffer_to_buffer(&b, 0, &shared, 0, BIG)
            .unwrap();
        blit.end_encoding().unwrap();
        run(&read);
        with_bytes(&shared, |bytes| assert!(bytes == expected.as_slice()));

        // The compute encoder takes fences and resource barriers too.
        let command_buffer = queue.command_buffer().unwrap();
        let mut compute = command_buffer.compute_command_encoder().unwrap();
        compute.wait_for_fence(&fence).unwrap();
        compute
            .memory_barrier_with_resources(&[a.as_resource(), (&b).into()])
            .unwrap();
        compute.memory_barrier_with_resources(&[]).unwrap();
        compute.update_fence(&fence).unwrap();
        compute.end_encoding().unwrap();
        run(&command_buffer);
    });
}

/// A managed buffer's CPU writes reach the GPU and its GPU writes reach the CPU, with
/// `did_modify_range` refusing a managed buffer's range past it, and both calls doing nothing on
/// a buffer of another storage mode, as plain Metal does.
#[test]
fn managed_buffers_round_trip() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let managed_options = options(StorageMode::Managed, HazardTrackingMode::Default);
        let managed = device.new_buffer(1001, managed_options).unwrap();
        assert_eq!(managed.storage_mode(), StorageMode::Managed);
        with_bytes(&managed, |bytes| {
            assert!(bytes.iter().all(|&byte| byte == 0));
            for (index, byte) in bytes.iter_mut().enumerate() {
                *byte = (index % 251) as u8;
            }
        });
        managed.did_modify_range(0..1001).unwrap();
        let private = device.new_buffer(1001, ResourceOptions::private()).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&managed, 0, &private, 0, 1001)
            .unwrap();
        blit.copy_from_buffer_to_buffer(&private, 0, &managed, 500, 501)
            .unwrap();
        blit.synchronize_resource(managed.as_resource()).unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        let mut expected = counted(0..1001);
        expected[500..].copy_from_slice(&counted(0..501));
        assert_eq!(with_bytes(&managed, |bytes| bytes.to_vec()), expected);

        let shared = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        with_bytes(&shared, |bytes| bytes.fill(7));
        shared.did_modify_range(0..16).unwrap();
        shared.did_modify_range(8..64).unwrap();
        private.did_modify_range(0..1001).unwrap();
        assert_eq!(
            managed.did_modify_range(1000..1002),
            Err(invalid(
                "did_modify_range",
                Problem::RangeOutOfBounds {
                    offset: 1000,
                    size: 2,
                    length: 1001
                }
            ))
        );
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.synchronize_resource(shared.as_resource()).unwrap();
        blit.synchronize_resource((&private).into()).unwrap();
        blit.fill_buffer(&shared, 0..4, 9).unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        let mut expected = [7; 16];
        expected[..4].fill(9);
        assert_eq!(with_bytes(&shared, |bytes| bytes.to_vec()), expected);
    });
}

/// A managed buffer follows Metal's contract for the device's memory model, which
/// `has_unified_memory` answers from the device's kind. With unified memory (an integrated GPU,
/// lavapipe) its CPU writes reach the GPU and its GPU writes the CPU without the calls, as on
/// Apple silicon. Without (a discrete GPU) they do only through the calls, as on a Mac with a
/// discrete GPU: the GPU reads what creation and the last `did_modify_range` gave it, and the CPU
/// what the last `synchronize_resource` gave it.
#[test]
fn managed_buffers_follow_the_memory_model() {
    on_device(|device| {
        let properties = device.properties();
        let unified = properties.has_unified_memory;
        println!(
            "runtime_lane: {} is {:?}, has_unified_memory {unified}",
            properties.name, properties.kind
        );
        assert_eq!(
            unified,
            matches!(properties.kind, DeviceKind::Integrated | DeviceKind::Cpu),
            "{:?}",
            properties.kind
        );
        let queue = device.new_command_queue().unwrap();
        let managed_options = options(StorageMode::Managed, HazardTrackingMode::Default);
        let made = counted(0..1001);
        let managed = device
            .new_buffer_with_bytes(&made, managed_options)
            .unwrap();
        assert_eq!(read_back(&queue, &managed), made);
        // A CPU write, then the GPU's read without `did_modify_range`, then with it.
        with_bytes(&managed, |bytes| bytes[..16].fill(0xAB));
        let mut written = made.clone();
        written[..16].fill(0xAB);
        let unsynchronized = if unified { &written } else { &made };
        assert_eq!(&read_back(&queue, &managed), unsynchronized);
        managed.did_modify_range(0..16).unwrap();
        assert_eq!(read_back(&queue, &managed), written);
        // A GPU write, then the CPU's read without `synchronize_resource`, then with it.
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&managed, 100..200, 0x5C).unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        let mut filled = written.clone();
        filled[100..200].fill(0x5C);
        let unsynchronized = if unified { &filled } else { &written };
        assert_eq!(
            &with_bytes(&managed, |bytes| bytes.to_vec()),
            unsynchronized
        );
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.synchronize_resource(managed.as_resource()).unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        assert_eq!(with_bytes(&managed, |bytes| bytes.to_vec()), filled);
    });
}

/// The kind [`shared_buffers_follow_the_memory_model`] takes on a device without unified memory,
/// and expects absent with it.
pub(super) const SYSTEM_MEMORY: [Degradation; 1] = [Degradation::SharedInSystemMemory];

/// `Shared` storage lies where a Mac places it (docs/API.md "The runtime's memory model"): on a
/// device without unified memory the first buffer of the default CPU cache mode the program makes
/// announces `shared-in-system-memory`, once, and neither a buffer of the write-combined mode nor
/// `Private` or `Managed` storage does; with unified memory none does. In either mode the host
/// reads the GPU's bytes and the GPU the host's.
#[test]
fn shared_buffers_follow_the_memory_model() {
    on_device_with(|device, messages| {
        let unified = device.properties().has_unified_memory;
        let shared =
            |cache| ResourceOptions::new(StorageMode::Shared, cache, HazardTrackingMode::Default);
        let queue = device.new_command_queue().unwrap();
        messages.take_notices();
        let combined = device
            .new_buffer(4096, shared(CpuCacheMode::WriteCombined))
            .unwrap();
        let _private = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        let _managed = device
            .new_buffer(
                4096,
                options(StorageMode::Managed, HazardTrackingMode::Default),
            )
            .unwrap();
        assert_eq!(counts(&messages.take_notices()), BTreeMap::new());
        let cached = device
            .new_buffer(4096, shared(CpuCacheMode::DefaultCache))
            .unwrap();
        let _again = device
            .new_buffer(64, shared(CpuCacheMode::DefaultCache))
            .unwrap();
        let notices = messages.take_notices();
        let expected: &[Degradation] = if unified { &[] } else { &SYSTEM_MEMORY };
        let announced = assert_once(&notices, expected);
        assert_eq!(announced.into_keys().collect::<Vec<_>>(), expected);
        for message in &notices {
            let notice = message.degradation.as_ref().unwrap();
            assert_eq!(notice.object, "a `Shared` buffer of 4096 bytes");
        }
        // The host's bytes reach the GPU from either mode, and the GPU's the host in the default
        // one.
        with_bytes(&combined, |bytes| {
            for (index, byte) in bytes.iter_mut().enumerate() {
                *byte = (index % 251) as u8;
            }
        });
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&combined, 0, &cached, 0, 4096)
            .unwrap();
        blit.fill_buffer(&cached, 100..200, 0x5C).unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        let mut expected = counted(0..4096);
        expected[100..200].fill(0x5C);
        assert_eq!(with_bytes(&cached, |bytes| bytes.to_vec()), expected);
        assert_eq!(read_back(&queue, &combined), counted(0..4096));
    });
}

/// What Metal's validation refuses is refused typed, and records nothing; an empty copy or fill,
/// which plain Metal skips wherever it lies, records nothing either.
#[test]
fn blits_refuse_what_metal_refuses() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let a = counting(device, 32);
        let b = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let copy = "copy_from_buffer_to_buffer";
        blit.copy_from_buffer_to_buffer(&a, 0, &b, 0, 0).unwrap();
        blit.copy_from_buffer_to_buffer(&a, 0, &b, 64, 0).unwrap();
        assert_eq!(
            blit.copy_from_buffer_to_buffer(&a, 0, &b, 1, 16),
            Err(invalid(
                copy,
                Problem::RangeOutOfBounds {
                    offset: 1,
                    size: 16,
                    length: 16
                }
            ))
        );
        assert_eq!(
            blit.copy_from_buffer_to_buffer(&a, u64::MAX, &b, 0, 2),
            Err(invalid(
                copy,
                Problem::RangeOutOfBounds {
                    offset: u64::MAX,
                    size: 2,
                    length: 32
                }
            ))
        );
        blit.fill_buffer(&b, 4..4, 1).unwrap();
        blit.fill_buffer(&b, 64..64, 1).unwrap();
        assert_eq!(
            blit.fill_buffer(&b, 8..17, 1),
            Err(invalid(
                "fill_buffer",
                Problem::RangeOutOfBounds {
                    offset: 8,
                    size: 9,
                    length: 16
                }
            ))
        );
        blit.end_encoding().unwrap();
        run(&command_buffer);
        with_bytes(&b, |bytes| assert!(bytes.iter().all(|&byte| byte == 0)));
    });
}

/// The barrier that starts every command buffer waits for nothing: two command buffers that fill
/// one untracked buffer, committed back to back without a fence, race, and the validation layer's
/// synchronization validation reports the hazard. (A start barrier that waited for earlier work
/// would order them, and the layer would report nothing.) Ignored by default: only a run with
/// synchronization validation on (`METAL2VULKAN_LANE_VALIDATION=sync`, `LaneChecks`, or the
/// layer's own `VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true`) checks hazards; a lavapipe run takes it.
///
/// The layer reports the hazard only when the second command buffer is submitted before the
/// completion turn has seen the first complete; once it has, the layer knows the first fill
/// done and reports nothing. Two things keep that window open on a fast GPU:
/// - The first command buffer waits for a shared event the host signals only once the second is
///   committed. The scheduler holds the first until then and submits the two in one pass. The
///   wait orders the first command buffer after the host's signal alone, not the second after
///   the first.
/// - The fills cover [`RACE`] bytes, so the first takes milliseconds on the GPU while the second
///   is submitted. With 8 MiB fills, the M3, the M4 and the RTX 3090 often finished the first
///   before the second went: on the M3, 1 run of 5 reported the hazard both without the hold
///   and with it.
#[test]
#[ignore = "needs the validation layer's synchronization validation"]
fn the_start_barrier_waits_for_nothing() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let untracked = options(StorageMode::Private, HazardTrackingMode::Untracked);
        let buffer = device.new_buffer(RACE, untracked).unwrap();
        let hold: SharedEvent = device.new_shared_event().unwrap();
        let fills: Vec<CommandBuffer> = (1..=2)
            .map(|value| {
                let command_buffer = queue.command_buffer().unwrap();
                if value == 1 {
                    command_buffer
                        .encode_wait_for_event(EventRef::from(&hold), 1)
                        .unwrap();
                }
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&buffer, 0..RACE, value).unwrap();
                blit.end_encoding().unwrap();
                command_buffer
            })
            .collect();
        for command_buffer in &fills {
            command_buffer.commit().unwrap();
        }
        hold.set_signaled_value(1);
        for command_buffer in &fills {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
        }
    }
    let messages = messages.take();
    eprintln!("the layer reported: {messages:#?}");
    assert!(
        messages
            .iter()
            .any(|message| message.id_name.contains("SYNC-HAZARD-WRITE-AFTER-WRITE")),
        "no write-after-write hazard was reported"
    );
}

/// Chains of the compute-blit-compute test, per leg.
const CHAINS: usize = 8;

/// A compute encoder, a blit encoder and a compute encoder chained through buffers: the first
/// dispatch writes `b`, a blit copies `b` into a Private `c`, and the second dispatch reads `c`.
/// Tracked buffers are ordered by the tracker, untracked ones by a fence each encoder waits for
/// and updates; each in one command buffer and across three committed back to back. The second
/// dispatch's output must equal a run of it alone over Metal's output of the first, and the
/// layer (with synchronization validation in the lanes) must report nothing for the leg.
#[test]
fn compute_blit_compute_chains_read_what_was_written() {
    on_device_with(|device, messages| {
        let case =
            public_case("saturating-halving-and-wide-product-integers-at-their-edges").unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        let queue = device.new_command_queue().unwrap();
        let output = case.compute.buffers[&1].len() as u64;
        let case_buffers = |options: ResourceOptions| -> BTreeMap<u32, Buffer> {
            case.compute
                .buffers
                .iter()
                .map(|(&index, bytes)| {
                    (index, buffer_with_options(device, bytes, options).unwrap())
                })
                .collect()
        };
        // The second dispatch alone, over the first one's output (Metal's).
        let mut alone = case_buffers(ResourceOptions::shared());
        alone.insert(0, buffer_with(device, &case.metal_output).unwrap());
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encode_case(
            &mut encoder,
            &pipeline,
            &case.compute,
            &alone,
            &CaseResources::default(),
        )
        .unwrap();
        encoder.end_encoding().unwrap();
        run(&command_buffer);
        let expected = read_buffer(&alone[&1], 0, output).unwrap();

        for split in [false, true] {
            for hazard in [HazardTrackingMode::Tracked, HazardTrackingMode::Untracked] {
                let leg = format!("split {split}, {hazard:?}");
                let fenced = hazard == HazardTrackingMode::Untracked;
                let fence = device.new_fence();
                let chains: Vec<_> = (0..CHAINS)
                    .map(|_| {
                        let first = case_buffers(options(StorageMode::Shared, hazard));
                        let staged = device
                            .new_buffer(first[&1].length(), options(StorageMode::Private, hazard))
                            .unwrap();
                        let mut second = case_buffers(options(StorageMode::Shared, hazard));
                        second.insert(0, staged);
                        (first, second)
                    })
                    .collect();
                let mut command_buffers = vec![queue.command_buffer().unwrap()];
                let next = |command_buffers: &mut Vec<CommandBuffer>| {
                    if split {
                        command_buffers.push(queue.command_buffer().unwrap());
                    }
                    command_buffers.last().unwrap().clone()
                };
                let current = command_buffers[0].clone();
                let mut compute = current.compute_command_encoder().unwrap();
                for (first, _) in &chains {
                    encode_case(
                        &mut compute,
                        &pipeline,
                        &case.compute,
                        first,
                        &CaseResources::default(),
                    )
                    .unwrap();
                }
                if fenced {
                    compute.update_fence(&fence).unwrap();
                }
                compute.end_encoding().unwrap();
                let current = next(&mut command_buffers);
                let mut blit = current.blit_command_encoder().unwrap();
                if fenced {
                    blit.wait_for_fence(&fence).unwrap();
                }
                for (first, second) in &chains {
                    let length = first[&1].length();
                    blit.copy_from_buffer_to_buffer(&first[&1], 0, &second[&0], 0, length)
                        .unwrap();
                }
                if fenced {
                    blit.update_fence(&fence).unwrap();
                }
                blit.end_encoding().unwrap();
                let current = next(&mut command_buffers);
                let mut compute = current.compute_command_encoder().unwrap();
                if fenced {
                    compute.wait_for_fence(&fence).unwrap();
                }
                for (_, second) in &chains {
                    encode_case(
                        &mut compute,
                        &pipeline,
                        &case.compute,
                        second,
                        &CaseResources::default(),
                    )
                    .unwrap();
                }
                compute.end_encoding().unwrap();
                for command_buffer in &command_buffers {
                    command_buffer.commit().unwrap();
                }
                for command_buffer in &command_buffers {
                    command_buffer.wait_until_completed_for(WAIT).unwrap();
                    assert_eq!(command_buffer.error(), None, "{leg}");
                }
                layer_reported_nothing(messages, &leg);
                for (chain, (first, second)) in chains.iter().enumerate() {
                    let written = read_buffer(&first[&1], 0, output).unwrap();
                    assert_eq!(written, case.metal_output, "{leg}, chain {chain}: first");
                    let chained = read_buffer(&second[&1], 0, output).unwrap();
                    assert_eq!(chained, expected, "{leg}, chain {chain}: second");
                }
            }
        }
    });
}
