//! Heaps with buffers: the runtime's heaps answer as Metal's measured ones do (an M3 Max on macOS
//! 26.7 and an M4 on 27.0.1, the same on both), public kernel cases run over buffers placed in
//! heaps, placed buffers alias byte for byte from the CPU and the GPU, a kernel's accesses past a
//! placed buffer stay in it, whatever is placed beside it, where the device contains them, a kernel
//! reaches a buffer placed in a heap its encoder made resident with `use_heap(s)` through an
//! address held in memory, and the case runner's guarded buffer (an out-of-bounds target's heap of
//! its own) counts each byte a run changed around its target.

use super::degrade_notices::assert_once;
use super::host_memory::{
    gpu_fill, gpu_read, host_page, no_copy, not_importable, on_device, run, Mapping,
};
use super::no_copy_heaps::{no_copy_heap, HEAP};
use super::{
    bound_buffers, encode_case, layer_reported_nothing_at_all, pipeline_state, public_case,
    read_buffer, run_case, validated_device, CaseResources, Messages, PublicCase, ScratchCache,
    WAIT,
};
use crate::candidate::device_facts::DeviceFacts;
use crate::candidate::engine::guards::{GuardCount, GuardReach};
use crate::candidate::engine::ComputeOutput;
use crate::candidate::runtime_case::{guard_count, guarded_buffer, open_device, validated_options};
use crate::case::out_of_bounds::{OutOfBoundsResource, OutOfBoundsTarget};
use crate::coverage::RobustBufferForm;
use crate::executor_contract::out_of_bounds::{GUARD_FILL, PAD_FILL};
use crate::gpu::TEST_DEVICE;
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, CpuCacheMode, Degradation, Device, Error,
    HazardTrackingMode, Heap, HeapDescriptor, HeapProblem, HeapType, HostMemory, Invalid,
    MemoryKind, Nil, Problem, PurgeableState, ResourceOptions, RuntimeGap, SizeAndAlign,
    StorageMode,
};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

fn heap_problem(call: &'static str, problem: HeapProblem) -> Error {
    invalid(call, Problem::Heap(problem))
}

/// Plain Metal's nil for `call`'s heap `problem`.
fn heap_nil(call: &'static str, problem: HeapProblem) -> Error {
    Error::Nil(Nil::new(call, Problem::Heap(problem)))
}

fn options(
    storage: StorageMode,
    cache: CpuCacheMode,
    hazard: HazardTrackingMode,
) -> ResourceOptions {
    ResourceOptions::new(storage, cache, hazard)
}

fn shared(hazard: HazardTrackingMode) -> ResourceOptions {
    options(StorageMode::Shared, CpuCacheMode::DefaultCache, hazard)
}

fn private(hazard: HazardTrackingMode) -> ResourceOptions {
    options(StorageMode::Private, CpuCacheMode::DefaultCache, hazard)
}

/// A heap of `heap_type`, `size` bytes, and `options`.
fn heap(device: &Device, heap_type: HeapType, size: u64, options: ResourceOptions) -> Heap {
    device
        .new_heap(&HeapDescriptor::new(heap_type, size, options))
        .unwrap()
}

/// Write `bytes` at the start of a host-visible `buffer`.
fn write(buffer: &Buffer, bytes: &[u8]) {
    assert!(bytes.len() as u64 <= buffer.length());
    let contents = buffer.contents().expect("a host-visible buffer");
    // SAFETY: the buffer holds `bytes.len()` bytes at `contents`, and no command uses them now.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
}

/// The words `0x{base}0000 + i` of `count` words, little-endian.
fn words(base: u32, count: u32) -> Vec<u8> {
    (0..count)
        .flat_map(|i| ((base << 16) + i).to_le_bytes())
        .collect()
}

/// Metal's heap answers, each as measured, and the device's limits a heap meets: `HEAP` lines
/// print them for each venue.
#[test]
fn heaps_answer_as_metal_s_do() {
    on_device(|device| {
        let limits = device.compile_device().profile().limits();
        for alignment in [
            limits.min_storage_buffer_offset_alignment,
            limits.min_uniform_buffer_offset_alignment,
            limits.min_texel_buffer_offset_alignment,
        ] {
            assert_eq!(256 % alignment, 0, "an offset alignment of {alignment}");
        }
        let probe = |size| {
            device
                .new_heap(&HeapDescriptor::new(
                    HeapType::Placement,
                    size,
                    ResourceOptions::private(),
                ))
                .err()
        };
        eprintln!(
            "HEAP limits: maxBufferSize {}, maxMemoryAllocationSize {}, maxStorageBufferRange {}, \
             max_buffer_length {}; a heap past both: {:?}",
            device.compile_device().allocation().max_buffer_size,
            device
                .compile_device()
                .allocation()
                .max_memory_allocation_size,
            limits.max_storage_buffer_range,
            device.properties().max_buffer_length,
            probe(u64::MAX - (1 << 20)),
        );

        // heapBufferSizeAndAlign: the length, at 256, whatever the options.
        for length in [0, 1, 255, 257, 4097, 65537] {
            for options in [
                ResourceOptions::shared(),
                ResourceOptions::private(),
                shared(HazardTrackingMode::Tracked),
            ] {
                let answer = device.heap_buffer_size_and_align(length, options);
                assert_eq!((answer.size, answer.align), (length, 256));
            }
        }
        let _: SizeAndAlign = device.heap_buffer_size_and_align(1, ResourceOptions::shared());

        // Descriptors: the defaults' heap, sizes rounded to 16 KiB, and the refusals.
        let mut descriptor = HeapDescriptor::default();
        descriptor.size = 1;
        let default = device.new_heap(&descriptor).unwrap();
        assert_eq!(default.heap_type(), HeapType::Automatic);
        assert_eq!(default.size(), 16384);
        assert_eq!(default.current_allocated_size(), 16384);
        assert_eq!(default.storage_mode(), StorageMode::Private);
        assert_eq!(
            default.hazard_tracking_mode(),
            HazardTrackingMode::Untracked
        );
        assert_eq!(default.resource_options().raw(), 288);
        assert_eq!(default.max_available_size(16), Ok(16384));
        for (size, rounded) in [(4095, 16384), (65537, 81920), (1_048_577, 1_064_960)] {
            let made = heap(device, HeapType::Placement, size, ResourceOptions::shared());
            assert_eq!(made.size(), rounded);
            assert_eq!(made.max_available_size(16), Ok(0));
            assert_eq!(made.gpu_address().0 % 16384, 0);
        }
        let refused = |descriptor: HeapDescriptor| device.new_heap(&descriptor).err();
        assert_eq!(
            refused(HeapDescriptor::new(
                HeapType::Automatic,
                0,
                ResourceOptions::shared()
            )),
            Some(heap_nil("new_heap", HeapProblem::Size))
        );
        for storage in [StorageMode::Managed, StorageMode::Memoryless] {
            let options = options(
                storage,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Default,
            );
            assert_eq!(
                refused(HeapDescriptor::new(HeapType::Placement, 65536, options)),
                Some(invalid("new_heap", Problem::StorageMode))
            );
        }
        // A sparse heap: made where the device binds sparse memory, else refused: the runtime's gap
        // of the tile map's ordinary images, or what it lacks.
        match device.new_heap(&HeapDescriptor::new(
            HeapType::Sparse,
            65536,
            ResourceOptions::private(),
        )) {
            Ok(sparse) => {
                assert_eq!(sparse.heap_type(), HeapType::Sparse);
                assert_eq!(sparse.size(), 65536);
                assert_eq!(sparse.used_size(), 0);
            }
            Err(Error::Unsupported(unsupported)) => assert!(
                matches!(
                    unsupported.feature,
                    metal2vulkan::error::Feature::SparseTextures { .. }
                ),
                "{unsupported:?}"
            ),
            Err(Error::NotImplemented(RuntimeGap::SparseTileMaps)) => {}
            Err(other) => panic!("a sparse heap: {other:?}"),
        }
        let tracked = heap(
            device,
            HeapType::Placement,
            65536,
            shared(HazardTrackingMode::Tracked),
        );
        assert_eq!(tracked.resource_options().raw(), 512);
        let write_combined = options(
            StorageMode::Shared,
            CpuCacheMode::WriteCombined,
            HazardTrackingMode::Default,
        );
        let combined = heap(device, HeapType::Placement, 65536, write_combined);
        assert_eq!(combined.resource_options().raw(), 257);
        let buffer = combined
            .new_buffer_at_offset(16, write_combined, 0)
            .unwrap();
        assert_eq!(buffer.resource_options().raw(), 257);

        placement_rules(device);
        automatic_rules(device);
        option_rules(device);
        aliasable_rules(device);
        purgeable_rules(device);
    });
}

/// The largest heap the device takes (the lower of one allocation and one `VkBuffer`, in whole
/// 16 KiB) lies in one allocation of at most `maxMemoryAllocationSize` bytes, whose block's buffer
/// of addresses, as long as the block, is at most `maxBufferSize` (the validation layer reports
/// nothing): the runtime's memory grows by the heap's size and no more than one allocation. RADV's
/// largest heap is 4 GiB less 16 KiB, of an allocation of at most 4 GiB less 4 bytes. Made where
/// it is at most 4 GiB (RADV, ANV, lavapipe), else not; out of device memory is an answer too. A
/// GPU hazard (`m2v_gpu_hazards`): the heap is the device's largest allocation, up to 4 GiB.
#[cfg(any(m2v_cpu_icd, m2v_gpu_hazards))]
#[test]
fn the_largest_heap_lies_in_one_allocation() {
    on_device(|device| {
        let limits = device.compile_device().allocation();
        let buffer = match limits.max_buffer_size {
            0 => u64::MAX,
            size => size,
        };
        let largest = limits.max_memory_allocation_size.min(buffer) / 16384 * 16384;
        if largest > 4 << 30 {
            eprintln!("HEAP the largest heap, {largest} bytes, is past 4 GiB: not made");
            return;
        }
        // A heap first, so that the pools the runtime keeps room in have it before the count.
        let _first = heap(
            device,
            HeapType::Placement,
            16384,
            ResourceOptions::private(),
        );
        let before = device.work_counts().device_memory_allocated;
        let made = device.new_heap(&HeapDescriptor::new(
            HeapType::Placement,
            largest,
            ResourceOptions::private(),
        ));
        let grown = device.work_counts().device_memory_allocated - before;
        match made {
            Ok(made) => {
                eprintln!("HEAP the largest heap, {largest} bytes, took {grown} bytes of memory");
                assert_eq!(made.size(), largest);
                assert!(
                    (largest..=limits.max_memory_allocation_size).contains(&grown),
                    "the largest heap, {largest} bytes, took {grown} bytes, past one allocation \
                     of {}",
                    limits.max_memory_allocation_size
                );
            }
            Err(Error::OutOfMemory(MemoryKind::Device)) => {
                eprintln!("HEAP the largest heap, {largest} bytes: out of device memory");
            }
            Err(other) => panic!("the largest heap, {largest} bytes: {other:?}"),
        }
    });
}

/// A placement heap: offsets at multiples of 256 inside the heap, a buffer's address the heap's
/// plus its offset, and its bytes the heap's there, a buffer of length 0 too (probes 1 and 2).
fn placement_rules(device: &Device) {
    const CALL: &str = "new_buffer_at_offset";
    let heap = heap(
        device,
        HeapType::Placement,
        65536,
        ResourceOptions::shared(),
    );
    assert_eq!(heap.resource_options().raw(), 256);
    let base = heap.gpu_address().0;
    let mut placed = Vec::new();
    for (offset, length) in [
        (0, 100),
        (256, 16),
        (4096, 4096),
        (65280, 256),
        (256, 65280),
        // A buffer of no bytes, made where Metal's validation stops it.
        (0, 0),
        (256, 0),
    ] {
        let buffer = heap
            .new_buffer_at_offset(length, ResourceOptions::shared(), offset)
            .unwrap();
        assert_eq!(buffer.length(), length);
        assert_eq!(buffer.heap_offset(), offset);
        assert_eq!(buffer.gpu_address().0, base + offset);
        assert_eq!(buffer.heap(), Some(&heap));
        assert_eq!(buffer.resource_options().raw(), 256);
        assert_eq!(buffer.hazard_tracking_mode(), HazardTrackingMode::Untracked);
        placed.push(buffer);
    }
    let delta = placed[1].contents().unwrap().as_ptr() as usize
        - placed[0].contents().unwrap().as_ptr() as usize;
    assert_eq!(delta, 256);
    assert_eq!(heap.used_size(), 0, "a placement heap's used size is 0");
    let whole = heap
        .new_buffer_at_offset(65536, ResourceOptions::shared(), 0)
        .unwrap();
    assert!(read_buffer(&whole, 0, 65536)
        .unwrap()
        .iter()
        .all(|&b| b == 0));
    for offset in [1, 4, 16, 128, 255] {
        assert_eq!(
            heap.new_buffer_at_offset(16, ResourceOptions::shared(), offset)
                .err(),
            Some(heap_nil(CALL, HeapProblem::Offset { offset }))
        );
    }
    for (offset, length) in [(65280, 257), (65536, 1), (0, 65537), (u64::MAX - 255, 256)] {
        assert_eq!(
            heap.new_buffer_at_offset(length, ResourceOptions::shared(), offset)
                .err(),
            Some(heap_nil(
                CALL,
                HeapProblem::PastEnd {
                    offset,
                    length,
                    heap_size: 65536
                }
            ))
        );
    }
    assert_eq!(
        heap.new_buffer(16, ResourceOptions::shared()).err(),
        Some(heap_problem(
            "new_buffer",
            HeapProblem::HeapType {
                heap_type: HeapType::Placement
            }
        ))
    );
}

/// An automatic heap: first fit at 256-byte steps, the used size the lengths' sum, and nil when
/// full (probes 1 to 4).
fn automatic_rules(device: &Device) {
    let heap = heap(
        device,
        HeapType::Automatic,
        65536,
        ResourceOptions::shared(),
    );
    let mut held = Vec::new();
    for (length, delta, used, available) in [
        (1, 0, 1, 65520),
        (255, 256, 256, 65024),
        (256, 512, 512, 64768),
        (257, 768, 769, 64496),
        (16, 1280, 785, 64240),
        (4096, 1536, 4881, 59904),
    ] {
        let buffer = heap.new_buffer(length, ResourceOptions::shared()).unwrap();
        assert_eq!(buffer.gpu_address().0, heap.gpu_address().0 + delta);
        assert_eq!(
            buffer.heap_offset(),
            0,
            "Metal answers 0 in an automatic heap"
        );
        assert_eq!(heap.used_size(), used);
        assert_eq!(heap.max_available_size(16), Ok(available));
        held.push(buffer);
    }
    assert_eq!(heap.max_available_size(0), heap.max_available_size(1));
    for alignment in [3, 100] {
        assert_eq!(
            heap.max_available_size(alignment),
            Err(heap_problem(
                "max_available_size",
                HeapProblem::Alignment { alignment }
            ))
        );
    }
    drop(held);
    assert_eq!(heap.used_size(), 0);
    let first = heap.new_buffer(100, ResourceOptions::shared()).unwrap();
    let second = heap.new_buffer(100, ResourceOptions::shared()).unwrap();
    let filled: Vec<Buffer> =
        std::iter::from_fn(|| heap.new_buffer(4096, ResourceOptions::shared()).ok())
            .take(16)
            .collect();
    assert_eq!(filled.len(), 15);
    assert_eq!(
        heap.new_buffer(4096, ResourceOptions::shared()).err(),
        Some(Error::OutOfMemory(MemoryKind::Heap))
    );
    assert_eq!(heap.used_size(), 61640);
    drop((first, second, filled));
    for length in [0, 65537] {
        let refused = match length {
            0 => Error::Nil(Nil::new("new_buffer", Problem::BufferLength { length })),
            _ => heap_nil(
                "new_buffer",
                HeapProblem::PastEnd {
                    offset: 0,
                    length,
                    heap_size: 65536,
                },
            ),
        };
        assert_eq!(
            heap.new_buffer(length, ResourceOptions::shared()).err(),
            Some(refused)
        );
    }
    let whole = heap.new_buffer(65536, ResourceOptions::shared()).unwrap();
    assert_eq!(heap.used_size(), 65536);
    drop(whole);
    assert_eq!(
        heap.new_buffer_at_offset(16, ResourceOptions::shared(), 0)
            .err(),
        Some(heap_problem(
            "new_buffer_at_offset",
            HeapProblem::HeapType {
                heap_type: HeapType::Automatic
            }
        ))
    );
}

/// A buffer takes its heap's modes: other storage or cache modes are refused (probe 2's
/// mismatches, one per process on Metal), and a tracking mode named other than the heap's becomes
/// the heap's, as plain Metal makes it.
fn option_rules(device: &Device) {
    const CALL: &str = "new_buffer_at_offset";
    let untracked = heap(
        device,
        HeapType::Placement,
        65536,
        ResourceOptions::shared(),
    );
    let tracked = heap(
        device,
        HeapType::Placement,
        65536,
        shared(HazardTrackingMode::Tracked),
    );
    let in_private = heap(
        device,
        HeapType::Placement,
        65536,
        ResourceOptions::private(),
    );
    let place = |heap: &Heap, options| heap.new_buffer_at_offset(16, options, 0);
    assert_eq!(
        place(&untracked, ResourceOptions::private()).err(),
        Some(heap_problem(
            CALL,
            HeapProblem::StorageMode {
                heap: StorageMode::Shared,
                resource: StorageMode::Private
            }
        ))
    );
    assert_eq!(
        place(&in_private, ResourceOptions::shared()).err(),
        Some(heap_problem(
            CALL,
            HeapProblem::StorageMode {
                heap: StorageMode::Private,
                resource: StorageMode::Shared
            }
        ))
    );
    let combined = options(
        StorageMode::Shared,
        CpuCacheMode::WriteCombined,
        HazardTrackingMode::Default,
    );
    assert_eq!(
        place(&untracked, combined).err(),
        Some(heap_problem(
            CALL,
            HeapProblem::CpuCacheMode {
                heap: CpuCacheMode::DefaultCache,
                resource: CpuCacheMode::WriteCombined
            }
        ))
    );
    // A tracking mode named other than the heap's becomes the heap's, as plain Metal makes it.
    for heap in [&untracked, &tracked] {
        for named in [
            HazardTrackingMode::Tracked,
            HazardTrackingMode::Untracked,
            HazardTrackingMode::Default,
        ] {
            let buffer = place(heap, shared(named)).unwrap();
            assert_eq!(buffer.hazard_tracking_mode(), heap.hazard_tracking_mode());
            assert_eq!(buffer.resource_options(), heap.resource_options());
        }
    }
    let private = place(&in_private, private(HazardTrackingMode::Default)).unwrap();
    assert!(private.contents().is_none());
}

/// `make_aliasable` (probes 1, 3 and 4): an automatic heap gives the range to the next buffer,
/// which reads the bytes as they are; a placement heap's buffer is aliasable from the start; a
/// buffer outside a heap stays as it is.
fn aliasable_rules(device: &Device) {
    let heap = heap(
        device,
        HeapType::Automatic,
        65536,
        ResourceOptions::shared(),
    );
    let a = heap.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let b = heap.new_buffer(256, ResourceOptions::shared()).unwrap();
    write(&a, &words(0x5000, 2));
    assert!(!a.is_aliasable());
    a.make_aliasable().unwrap();
    a.make_aliasable().unwrap();
    assert!(a.is_aliasable() && !b.is_aliasable());
    assert_eq!(heap.used_size(), 256);
    let c = heap.new_buffer(512, ResourceOptions::shared()).unwrap();
    assert_eq!(c.gpu_address(), a.gpu_address());
    assert_eq!(read_buffer(&c, 0, 8).unwrap(), words(0x5000, 2));
    assert_eq!(heap.used_size(), 768);
    drop(a);
    assert_eq!(
        heap.used_size(),
        768,
        "an aliasable buffer's range was given back already"
    );
    drop((b, c));
    assert_eq!(heap.used_size(), 0);

    let placement = self::heap(
        device,
        HeapType::Placement,
        65536,
        ResourceOptions::shared(),
    );
    let placed = placement
        .new_buffer_at_offset(100, ResourceOptions::shared(), 0)
        .unwrap();
    // Aliasable from the start, as Metal's.
    assert!(placed.is_aliasable());
    placed.make_aliasable().unwrap();
    assert!(placed.is_aliasable());
    assert_eq!(placement.used_size(), 0);

    // Outside a heap, the call changes nothing, as plain Metal.
    let outside = device.new_buffer(100, ResourceOptions::shared()).unwrap();
    write(&outside, &words(0x5100, 25));
    assert_eq!(outside.make_aliasable(), Ok(()));
    assert!(!outside.is_aliasable());
    assert_eq!(read_buffer(&outside, 0, 100).unwrap(), words(0x5100, 25));
    assert_eq!((outside.heap(), outside.heap_offset()), (None, 0));
}

/// Purgeable states (probes 1 and 3): a heap's answer as a buffer's; emptying a `Shared` heap
/// zeroes its bytes until a committed command used the heap and keeps them from then on, and a
/// `Private` one keeps them; every call on a heap's buffer answers `NonVolatile` and changes
/// nothing. A buffer holds its heap: the heap's handle dropped, the GPU still writes the buffer.
fn purgeable_rules(device: &Device) {
    use PurgeableState::{Empty, KeepCurrent, NonVolatile, Volatile};
    let queue = device.new_command_queue().unwrap();
    let heap = heap(
        device,
        HeapType::Placement,
        65536,
        ResourceOptions::shared(),
    );
    let buffer = heap
        .new_buffer_at_offset(1024, ResourceOptions::shared(), 0)
        .unwrap();
    for (state, answer) in [
        (KeepCurrent, NonVolatile),
        (Volatile, NonVolatile),
        (KeepCurrent, Volatile),
        (NonVolatile, Volatile),
        (KeepCurrent, NonVolatile),
    ] {
        assert_eq!(heap.set_purgeable_state(state), Ok(answer));
    }
    // A heap's buffer answers `NonVolatile` to every call and changes nothing: `Empty` keeps
    // its bytes, and the heap's state stays.
    write(&buffer, &words(0x7000, 256));
    assert_eq!(heap.set_purgeable_state(Volatile), Ok(NonVolatile));
    for state in [KeepCurrent, NonVolatile, Volatile, Empty] {
        assert_eq!(buffer.set_purgeable_state(state), Ok(NonVolatile));
        assert_eq!(heap.set_purgeable_state(KeepCurrent), Ok(Volatile));
    }
    assert_eq!(read_buffer(&buffer, 0, 1024).unwrap(), words(0x7000, 256));
    assert_eq!(heap.set_purgeable_state(NonVolatile), Ok(Volatile));
    assert_eq!(heap.set_purgeable_state(Empty), Ok(NonVolatile));
    assert_eq!(heap.set_purgeable_state(NonVolatile), Ok(Empty));
    assert!(read_buffer(&buffer, 0, 1024)
        .unwrap()
        .iter()
        .all(|&b| b == 0));
    assert_eq!(gpu_read(&queue, &buffer, 0..1024), vec![0; 1024]);

    let in_private = self::heap(
        device,
        HeapType::Placement,
        65536,
        ResourceOptions::private(),
    );
    let kept = in_private
        .new_buffer_at_offset(1024, ResourceOptions::private(), 0)
        .unwrap();
    gpu_fill(&queue, &kept, 0..1024, 0x71);
    assert_eq!(in_private.set_purgeable_state(Empty), Ok(NonVolatile));
    assert_eq!(in_private.set_purgeable_state(NonVolatile), Ok(Empty));
    assert_eq!(gpu_read(&queue, &kept, 0..1024), vec![0x71; 1024]);

    // A blit encoded and never committed, and `use_heap` in an encoder that encodes nothing
    // after it, do not use the heap: `Empty` zeroes its bytes. A committed blit does: `Empty`
    // keeps them, every buffer's, from then on.
    let used = self::heap(
        device,
        HeapType::Placement,
        65536,
        ResourceOptions::shared(),
    );
    let first = used
        .new_buffer_at_offset(1024, ResourceOptions::shared(), 0)
        .unwrap();
    let second = used
        .new_buffer_at_offset(1024, ResourceOptions::shared(), 1024)
        .unwrap();
    write(&first, &words(0x7100, 256));
    let uncommitted = queue.command_buffer().unwrap();
    let mut blit = uncommitted.blit_command_encoder().unwrap();
    blit.fill_buffer(&second, 0..1024, 0x22).unwrap();
    blit.end_encoding().unwrap();
    drop(uncommitted);
    let resident_only = queue.command_buffer().unwrap();
    let mut compute = resident_only.compute_command_encoder().unwrap();
    compute.use_heap(&used).unwrap();
    compute.end_encoding().unwrap();
    run(&resident_only);
    assert_eq!(used.set_purgeable_state(Empty), Ok(NonVolatile));
    assert_eq!(used.set_purgeable_state(NonVolatile), Ok(Empty));
    assert!(read_buffer(&first, 0, 1024)
        .unwrap()
        .iter()
        .all(|&b| b == 0));
    write(&first, &words(0x7100, 256));
    gpu_fill(&queue, &second, 0..1024, 0x22);
    assert_eq!(used.set_purgeable_state(Empty), Ok(NonVolatile));
    assert_eq!(used.set_purgeable_state(NonVolatile), Ok(Empty));
    assert_eq!(read_buffer(&first, 0, 1024).unwrap(), words(0x7100, 256));
    assert_eq!(read_buffer(&second, 0, 1024).unwrap(), vec![0x22; 1024]);

    drop((heap, in_private, used));
    gpu_fill(&queue, &buffer, 16..32, 0x4d);
    assert_eq!(read_buffer(&buffer, 16, 16).unwrap(), vec![0x4d; 16]);
    assert!(buffer.heap().is_some());
}

/// Where each of `case`'s buffers is placed in a placement heap: offsets at multiples of 256, each
/// at least `gap` bytes past the end of the one before.
fn offsets(case: &PublicCase, gap: u64) -> BTreeMap<u32, u64> {
    let mut at = 0;
    case.compute
        .buffers
        .iter()
        .map(|(&index, bytes)| {
            let offset = at;
            at = (offset + bytes.len().max(1) as u64 + gap).next_multiple_of(256);
            (index, offset)
        })
        .collect()
}

/// Run `case` over `buffers` in one command buffer of `queue`: first a blit encoder of `copies`
/// (each whole source into its destination), then the case's compute encoder, then a blit of
/// `back` (the output buffer into a buffer the CPU reads), and wait for it.
fn run_over(
    queue: &CommandQueue,
    case: &PublicCase,
    buffers: &BTreeMap<u32, Buffer>,
    copies: &[(&Buffer, &Buffer)],
    back: Option<(&Buffer, &Buffer)>,
) {
    let device = queue.device();
    let pipeline = pipeline_state(device, case).unwrap();
    let resources = CaseResources::new(device, case).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    if !copies.is_empty() {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for (from, to) in copies {
            blit.copy_from_buffer_to_buffer(from, 0, to, 0, from.length())
                .unwrap();
        }
        blit.end_encoding().unwrap();
    }
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encode_case(&mut encoder, &pipeline, &case.compute, buffers, &resources).unwrap();
    encoder.end_encoding().unwrap();
    if let Some((from, to)) = back {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(from, 0, to, 0, from.length())
            .unwrap();
        blit.end_encoding().unwrap();
    }
    run(&command_buffer);
}

/// The output binding, offset and length of `case`.
fn output(case: &PublicCase) -> (u32, u64, u64) {
    match case.compute.output {
        ComputeOutput::Buffer {
            binding,
            offset,
            length,
        } => (binding, offset, length),
        ref other => panic!("an output {other:?}"),
    }
}

/// Whether `device` leaves loads and stores through buffer descriptors uncontained: its
/// containment relies on the device's buffer robustness (no
/// `DeviceProfile::checks_buffers_in_shader`), which its driver does not keep inside the buffer
/// (`ExecutionClass::robust_buffers_uncontained`).
fn loads_and_stores_uncontained(device: &Device) -> bool {
    // SAFETY: only physical-device queries of the device's own instance; nothing is created,
    // destroyed, or submitted, and the handles are not kept past the borrow.
    let facts = unsafe {
        let handles = device.raw();
        DeviceFacts::query(handles.entry, handles.instance, handles.physical_device)
    }
    .unwrap();
    let robust = !device.compile_device().profile().checks_buffers_in_shader();
    robust
        && facts
            .execution_class()
            .robust_buffers_uncontained()
            .contains(&RobustBufferForm::LoadStore)
}

/// A public kernel case runs over buffers placed in heaps and matches Metal: in a `Shared`
/// placement heap and a `Shared` automatic heap, its inputs written by the CPU, and in a tracked
/// `Private` placement heap, its inputs copied in and its output copied out by blits in the same
/// command buffer, which the runtime orders as Metal orders a tracked heap's uses: as one
/// resource's (the layer's synchronization validation checks the barriers).
#[test]
fn a_public_case_runs_over_buffers_placed_in_heaps() {
    on_device(|device| {
        let case =
            public_case("saturating-halving-and-wide-product-integers-at-their-edges").unwrap();
        let queue = device.new_command_queue().unwrap();
        let (binding, offset, length) = output(&case);
        let at = offsets(&case, 0);

        let placement = heap(
            device,
            HeapType::Placement,
            65536,
            ResourceOptions::shared(),
        );
        let automatic = heap(
            device,
            HeapType::Automatic,
            65536,
            ResourceOptions::shared(),
        );
        for in_heap in [&placement, &automatic] {
            let buffers: BTreeMap<u32, Buffer> = case
                .compute
                .buffers
                .iter()
                .map(|(&index, bytes)| {
                    let length = bytes.len().max(1) as u64;
                    let buffer = match in_heap.heap_type() {
                        HeapType::Placement => in_heap
                            .new_buffer_at_offset(length, ResourceOptions::shared(), at[&index])
                            .unwrap(),
                        _ => in_heap
                            .new_buffer(length, ResourceOptions::shared())
                            .unwrap(),
                    };
                    write(&buffer, bytes);
                    (index, buffer)
                })
                .collect();
            run_over(&queue, &case, &buffers, &[], None);
            let produced = read_buffer(&buffers[&binding], offset, length).unwrap();
            assert!(
                produced == case.metal_output,
                "{:?} heap: the output differs from Metal's",
                in_heap.heap_type()
            );
        }

        let in_private = heap(
            device,
            HeapType::Placement,
            65536,
            private(HazardTrackingMode::Tracked),
        );
        let staged: Vec<(Buffer, Buffer)> = case
            .compute
            .buffers
            .iter()
            .map(|(&index, bytes)| {
                let length = bytes.len().max(1) as u64;
                let from = device
                    .new_buffer(length, ResourceOptions::shared())
                    .unwrap();
                write(&from, bytes);
                let to = in_private
                    .new_buffer_at_offset(length, ResourceOptions::private(), at[&index])
                    .unwrap();
                (from, to)
            })
            .collect();
        let buffers: BTreeMap<u32, Buffer> = case
            .compute
            .buffers
            .keys()
            .zip(&staged)
            .map(|(&index, (_, to))| (index, to.clone()))
            .collect();
        let copies: Vec<(&Buffer, &Buffer)> = staged.iter().map(|(from, to)| (from, to)).collect();
        let back = device
            .new_buffer(buffers[&binding].length(), ResourceOptions::shared())
            .unwrap();
        run_over(
            &queue,
            &case,
            &buffers,
            &copies,
            Some((&buffers[&binding], &back)),
        );
        let produced = read_buffer(&back, offset, length).unwrap();
        assert!(
            produced == case.metal_output,
            "the tracked private heap's output differs from Metal's"
        );
    });
}

/// Placed buffers over the same bytes alias them: a CPU write through one reads through the other,
/// and a GPU fill through one is what a GPU copy through the other reads, in two command buffers
/// of an untracked heap and in two encoders of one command buffer of a tracked heap, which the
/// runtime orders as one resource (probe 2's `alias_*`).
#[test]
fn placed_buffers_over_the_same_bytes_alias_them() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let untracked = heap(
            device,
            HeapType::Placement,
            65536,
            ResourceOptions::shared(),
        );
        let a = untracked
            .new_buffer_at_offset(512, ResourceOptions::shared(), 0)
            .unwrap();
        let b = untracked
            .new_buffer_at_offset(512, ResourceOptions::shared(), 256)
            .unwrap();
        assert_eq!(read_buffer(&b, 0, 16).unwrap(), vec![0; 16]);
        write(&a, &words(0xa000, 128));
        assert_eq!(
            read_buffer(&b, 0, 16).unwrap(),
            words(0xa000, 128)[256..272]
        );
        write(&b, &0x1122_3344u32.to_le_bytes());
        assert_eq!(
            read_buffer(&a, 256, 4).unwrap(),
            0x1122_3344u32.to_le_bytes()
        );
        gpu_fill(&queue, &a, 256..512, 0x3c);
        assert_eq!(gpu_read(&queue, &b, 0..256), vec![0x3c; 256]);

        for storage in [StorageMode::Shared, StorageMode::Private] {
            let options = options(
                storage,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Tracked,
            );
            let tracked = heap(device, HeapType::Placement, 65536, options);
            let a = tracked.new_buffer_at_offset(1024, options, 0).unwrap();
            let b = tracked.new_buffer_at_offset(1024, options, 512).unwrap();
            let back = device.new_buffer(512, ResourceOptions::shared()).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&a, 0..1024, 0x5e).unwrap();
            blit.end_encoding().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(&b, 0, &back, 0, 512)
                .unwrap();
            blit.end_encoding().unwrap();
            run(&command_buffer);
            assert_eq!(
                read_buffer(&back, 0, 512).unwrap(),
                vec![0x5e; 512],
                "{storage:?}"
            );
        }
    });
}

/// RT13's out-of-bounds targets on placed buffers: `loads-and-stores-past-a-device-buffer` loads
/// past a 16-byte buffer and stores past a 64-byte one. Each runs at its exact length in a `Shared`
/// placement heap with a buffer placed right after it at the next multiple of 256, the rest of the
/// heap filled with the guard byte through a buffer over the whole heap. The in-bounds loads and
/// stores give Metal's words. Where the device contains the targets, no access reaches past a
/// target's bound bytes (its length padded to 16): every other byte of the heap, the neighbours'
/// included, is unchanged. Where it relies on a robustness its driver leaves uncontained (MoltenVK,
/// Metal bounds no buffer access either), the stores land in the heap past the target, as Metal's
/// do, and stay within the reach the case declares past it.
#[test]
fn accesses_past_a_placed_buffer_stay_in_the_reach_the_device_allows() {
    on_device(|device| {
        let case = public_case("loads-and-stores-past-a-device-buffer").unwrap();
        let uncontained = loads_and_stores_uncontained(device);
        let queue = device.new_command_queue().unwrap();
        let heap = heap(
            device,
            HeapType::Placement,
            65536,
            ResourceOptions::shared(),
        );
        let whole = heap
            .new_buffer_at_offset(65536, ResourceOptions::shared(), 0)
            .unwrap();
        write(&whole, &[GUARD_FILL; 65536]);
        let at = offsets(&case, 256);
        let declared: BTreeMap<u32, &OutOfBoundsTarget> = case
            .case
            .out_of_bounds
            .iter()
            .flat_map(|declaration| &declaration.targets)
            .filter_map(|target| match target.resource {
                OutOfBoundsResource::Buffer { binding } => Some((binding, target)),
                _ => None,
            })
            .collect();
        let mut bound = Vec::new();
        let mut reach = Vec::new();
        let mut neighbours = Vec::new();
        let buffers: BTreeMap<u32, Buffer> = case
            .compute
            .buffers
            .iter()
            .map(|(&index, bytes)| {
                let length = bytes.len().max(1) as u64;
                let buffer = heap
                    .new_buffer_at_offset(length, ResourceOptions::shared(), at[&index])
                    .unwrap();
                write(&buffer, bytes);
                bound.push(at[&index]..at[&index] + length.next_multiple_of(16));
                if let Some(target) = declared.get(&index) {
                    let start = at[&index].saturating_sub(u64::from(target.reach_before));
                    reach.push(start..at[&index] + length + u64::from(target.reach_after));
                }
                let next = (at[&index] + length).next_multiple_of(256);
                let neighbour = heap
                    .new_buffer_at_offset(256, ResourceOptions::shared(), next)
                    .unwrap();
                write(&neighbour, &words(0xc000 + index, 64));
                neighbours.push(neighbour);
                (index, buffer)
            })
            .collect();
        let before = read_buffer(&whole, 0, 65536).unwrap();
        run_over(&queue, &case, &buffers, &[], None);
        let after = read_buffer(&whole, 0, 65536).unwrap();
        let changed: Vec<u64> = (0..65536u64)
            .filter(|&i| before[i as usize] != after[i as usize])
            .filter(|i| !bound.iter().any(|range| range.contains(i)))
            .collect();
        let outside_reach = changed
            .iter()
            .filter(|i| !reach.iter().any(|range| range.contains(i)))
            .count();
        eprintln!(
            "HEAP out-of-bounds case: loads and stores {}contained; {} bytes outside the targets \
             changed ({:?} to {:?}), {outside_reach} past the declared reach",
            if uncontained { "un" } else { "" },
            changed.len(),
            changed.first(),
            changed.last()
        );
        assert_eq!(outside_reach, 0, "an access past the case's declared reach");
        assert!(
            uncontained || changed.is_empty(),
            "{} bytes outside the targets changed, from {:?}",
            changed.len(),
            changed.first()
        );
        let (binding, offset, length) = output(&case);
        let produced = read_buffer(&buffers[&binding], offset, length).unwrap();
        // Loads 0 to 3 lie in the source buffer; stores 0 to 3 (words 8 to 11) in the output.
        for words in [0..4, 8..12] {
            let bytes = words.start * 4..words.end * 4;
            assert!(
                produced[bytes.clone()] == case.metal_output[bytes],
                "words {words:?} differ from Metal's"
            );
        }
        drop(neighbours);
    });
}

/// The case runner's guarded buffer, an out-of-bounds target's: it lies in a tracked placement heap
/// of its own, at the first multiple of 256 at least its reach before it (and 256), its bytes where
/// it lies; the bytes past its length to its length padded to 16 hold the padding fill, and every
/// other byte of the heap the guard fill, at least its reach after it past the padding. Each
/// changed guard and padding byte is counted apart, and the target's own bytes are neither. A
/// buffer of its own has no guard to count.
#[test]
fn a_guarded_buffer_counts_each_changed_guard_and_padding_byte() {
    on_device(|device| {
        let bytes: Vec<u8> = (0..20).collect();
        let reach = GuardReach {
            before: 300,
            after: 1024,
        };
        let buffer = guarded_buffer(device, &bytes, reach).unwrap();
        let heap = buffer.heap().expect("a guarded buffer lies in a heap");
        assert_eq!(heap.heap_type(), HeapType::Placement);
        assert_eq!(heap.hazard_tracking_mode(), HazardTrackingMode::Tracked);
        assert_eq!((buffer.length(), buffer.heap_offset()), (20, 512));
        assert!(heap.size() >= 512 + 32 + 1024);
        let whole = heap
            .new_buffer_at_offset(heap.size(), ResourceOptions::shared(), 0)
            .unwrap();
        let mut held = read_buffer(&whole, 0, heap.size()).unwrap();
        assert_eq!(held[512..532], bytes[..]);
        assert!(held[532..544].iter().all(|&byte| byte == PAD_FILL));
        assert!(held[..512]
            .iter()
            .chain(&held[544..])
            .all(|&byte| byte == GUARD_FILL));
        let count = |escaped, past_end| GuardCount { escaped, past_end };
        assert_eq!(guard_count(&buffer).unwrap(), count(0, 0));
        held[0] = 0;
        held[544 + 3] = 1;
        held[532] = 0;
        // The target's own byte is neither guard nor padding.
        held[512] = 0xff;
        write(&whole, &held);
        assert_eq!(guard_count(&buffer).unwrap(), count(2, 1));
        let own = device.new_buffer(20, ResourceOptions::shared()).unwrap();
        assert!(guard_count(&own).is_err());
    });
}

/// Walks a pointer chain from the address its argument buffer (buffer 0) holds at offset 0.
const POINTER_WALKER: &str = "walk-a-device-pointer-chain-through-a-loop-phi";

/// The pointer walker's argument-buffer buffer placed in a heap that the encoder made resident
/// with `use_heap(s)`, never with `use_resource`: the kernel loads the buffer's address from its
/// argument buffer and reaches it through the heap's range, its output Metal's. In a `Shared`
/// placement heap at an offset; in a `Shared` automatic heap after another buffer, the second of
/// two heaps `use_heaps` makes resident; and in a tracked `Private` placement heap whose bytes a
/// blit of the same command buffer copied in, which the runtime orders before the kernel's read
/// as one resource's use (the layer's synchronization validation checks the barrier). Another
/// device's encoder refuses the heap.
#[test]
fn a_kernel_reaches_buffers_placed_in_heaps_it_uses() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let (binding, offset, length) = output(&walker);
        let walk = pipeline_state(device, &walker).unwrap();
        let resources = CaseResources::new(device, &walker).unwrap();
        let [(&field, member)] = resources.argument_buffers.iter().collect::<Vec<_>>()[..] else {
            panic!("{POINTER_WALKER}: one argument-buffer buffer");
        };
        let bytes = read_buffer(member, 0, member.length()).unwrap();
        let size = bytes.len() as u64;
        // Run the walker with its argument buffer holding `placed`'s address, after a blit of
        // `copy` in the same command buffer, its encoder making `heaps` resident and nothing else.
        let walk_over = |placed: &Buffer, heaps: &[&Heap], copy: Option<(&Buffer, &Buffer)>| {
            let mut holding = CaseResources::new(device, &walker).unwrap();
            holding.argument_buffers.insert(field, placed.clone());
            let buffers = bound_buffers(device, &walker.compute, &holding).unwrap();
            assert_eq!(
                read_buffer(&buffers[&field.0], u64::from(field.1), 8).unwrap(),
                placed.gpu_address().0.to_le_bytes(),
                "the argument buffer holds the placed buffer's address"
            );
            let unused = CaseResources {
                argument_buffers: Default::default(),
                ..holding
            };
            let command_buffer = queue.command_buffer().unwrap();
            if let Some((from, to)) = copy {
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_buffer_to_buffer(from, 0, to, 0, from.length())
                    .unwrap();
                blit.end_encoding().unwrap();
            }
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            match heaps {
                [heap] => encoder.use_heap(heap).unwrap(),
                heaps => encoder.use_heaps(heaps).unwrap(),
            }
            encode_case(&mut encoder, &walk, &walker.compute, &buffers, &unused).unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
            read_buffer(&buffers[&binding], offset, length).unwrap()
        };

        let placement = heap(
            device,
            HeapType::Placement,
            65536,
            ResourceOptions::shared(),
        );
        let placed = placement
            .new_buffer_at_offset(size, ResourceOptions::shared(), 4096)
            .unwrap();
        write(&placed, &bytes);
        let produced = walk_over(&placed, &[&placement], None);
        assert!(produced == walker.metal_output, "a placement heap");

        let automatic = heap(
            device,
            HeapType::Automatic,
            65536,
            ResourceOptions::shared(),
        );
        let before = automatic
            .new_buffer(4096, ResourceOptions::shared())
            .unwrap();
        let placed = automatic
            .new_buffer(size, ResourceOptions::shared())
            .unwrap();
        assert_eq!(
            placed.gpu_address().0,
            automatic.gpu_address().0 + 4096,
            "the second buffer follows the first"
        );
        write(&placed, &bytes);
        let produced = walk_over(&placed, &[&placement, &automatic], None);
        assert!(produced == walker.metal_output, "the second of two heaps");
        drop(before);

        let in_private = heap(
            device,
            HeapType::Placement,
            65536,
            private(HazardTrackingMode::Tracked),
        );
        let placed = in_private
            .new_buffer_at_offset(size, ResourceOptions::private(), 256)
            .unwrap();
        let staged = device.new_buffer(size, ResourceOptions::shared()).unwrap();
        write(&staged, &bytes);
        let produced = walk_over(&placed, &[&in_private], Some((&staged, &placed)));
        assert!(produced == walker.metal_output, "a tracked private heap");

        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        {
            let other = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
            let command_buffer = other.new_command_queue().unwrap().command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            assert_eq!(
                encoder.use_heap(&placement),
                Err(invalid("use_heap", Problem::ForeignObject))
            );
            let own = heap(
                &other,
                HeapType::Placement,
                65536,
                ResourceOptions::shared(),
            );
            assert_eq!(
                encoder.use_heaps(&[&own, &placement]),
                Err(invalid("use_heaps", Problem::ForeignObject))
            );
            encoder.end_encoding().unwrap();
        }
        layer_reported_nothing_at_all(&messages);
    });
}

/// Where the pointer walker's argument buffer points at `placed`, and its encoder makes nothing
/// resident: the walker run in a command buffer whose later compute encoder uses `later` (if
/// any) and dispatches nothing; its output.
fn walk_unnamed(
    device: &Device,
    queue: &CommandQueue,
    walker: &PublicCase,
    placed: &Buffer,
    later: Option<&Heap>,
) -> Vec<u8> {
    let (binding, offset, length) = output(walker);
    let walk = pipeline_state(device, walker).unwrap();
    let mut holding = CaseResources::new(device, walker).unwrap();
    let [field] = holding.argument_buffers.keys().copied().collect::<Vec<_>>()[..] else {
        panic!("{POINTER_WALKER}: one argument-buffer buffer");
    };
    holding.argument_buffers.insert(field, placed.clone());
    let buffers = bound_buffers(device, &walker.compute, &holding).unwrap();
    let unnamed = CaseResources {
        argument_buffers: Default::default(),
        ..holding
    };
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encode_case(&mut encoder, &walk, &walker.compute, &buffers, &unnamed).unwrap();
    encoder.end_encoding().unwrap();
    if let Some(heap) = later {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.use_heap(heap).unwrap();
        encoder.end_encoding().unwrap();
    }
    run(&command_buffer);
    read_buffer(&buffers[&binding], offset, length).unwrap()
}

/// A pointer loaded from memory reaches what Metal's reaches (the M3 Max under macOS 26.7, and an
/// M4, every run): every live buffer of its own, though no encoder of its command buffer named it;
/// a buffer placed in a heap that a later encoder of its command buffer used, or one whose other
/// buffer an earlier command buffer's blit read. A buffer placed in a heap that no command buffer's
/// commands used is not reached: the walk reads zero and its output is not Metal's.
#[test]
fn a_loaded_pointer_reaches_every_buffer_of_its_own_and_the_heaps_a_command_buffer_used() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let resources = CaseResources::new(device, &walker).unwrap();
        let [member] = resources.argument_buffers.values().collect::<Vec<_>>()[..] else {
            panic!("{POINTER_WALKER}: one argument-buffer buffer");
        };
        let bytes = read_buffer(member, 0, member.length()).unwrap();
        let size = bytes.len() as u64;

        let own = device.new_buffer(size, ResourceOptions::shared()).unwrap();
        write(&own, &bytes);
        let produced = walk_unnamed(device, &queue, &walker, &own, None);
        assert!(produced == walker.metal_output, "a buffer of its own");

        let used = heap(
            device,
            HeapType::Placement,
            65536,
            ResourceOptions::shared(),
        );
        let placed = used
            .new_buffer_at_offset(size, ResourceOptions::shared(), 4096)
            .unwrap();
        write(&placed, &bytes);
        let produced = walk_unnamed(device, &queue, &walker, &placed, Some(&used));
        assert!(
            produced == walker.metal_output,
            "a heap a later encoder used"
        );

        let unused = heap(
            device,
            HeapType::Placement,
            65536,
            ResourceOptions::shared(),
        );
        let placed = unused
            .new_buffer_at_offset(size, ResourceOptions::shared(), 4096)
            .unwrap();
        write(&placed, &bytes);
        let produced = walk_unnamed(device, &queue, &walker, &placed, None);
        assert!(produced != walker.metal_output, "a heap no command used");
        // Another buffer of the heap, which an earlier command buffer's blit reads.
        let other = unused
            .new_buffer_at_offset(256, ResourceOptions::shared(), 0)
            .unwrap();
        gpu_read(&queue, &other, 0..4);
        write(&placed, &bytes);
        let produced = walk_unnamed(device, &queue, &walker, &placed, None);
        assert!(
            produced == walker.metal_output,
            "a heap whose other buffer an earlier command buffer used"
        );
    });
}

/// A pointer loaded from memory reaches a no-copy buffer of its own only once a command used it,
/// as Metal's (the M3 Max and an M4, every run): before, the walk reads zero and its output is not
/// Metal's; after an earlier command buffer's blit read it, the walk is Metal's. A venue that
/// refuses the import has nothing to run.
#[test]
fn a_loaded_pointer_reaches_a_no_copy_buffer_once_a_command_used_it() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let resources = CaseResources::new(device, &walker).unwrap();
        let [member] = resources.argument_buffers.values().collect::<Vec<_>>()[..] else {
            panic!("{POINTER_WALKER}: one argument-buffer buffer");
        };
        let bytes = read_buffer(member, 0, member.length()).unwrap();
        let len = bytes.len().next_multiple_of(host_page());
        let memory = Mapping::anonymous(len);
        memory.write(0, &bytes);
        // SAFETY: the mapping outlives the buffer, which is dropped first.
        let pages = unsafe {
            no_copy(
                device,
                memory.pointer(),
                len,
                ResourceOptions::shared(),
                None,
            )
        };
        let pages = match pages {
            Err(error) if not_importable(&error) => {
                eprintln!("NO-COPY-REFUSED {error:?}: nothing to run");
                return;
            }
            pages => pages.unwrap(),
        };
        let produced = walk_unnamed(device, &queue, &walker, &pages, None);
        assert!(
            produced != walker.metal_output,
            "a no-copy buffer no command used"
        );
        gpu_read(&queue, &pages, 0..4);
        memory.write(0, &bytes);
        let produced = walk_unnamed(device, &queue, &walker, &pages, None);
        assert!(
            produced == walker.metal_output,
            "a no-copy buffer an earlier command buffer used"
        );
        drop(pages);
    });
}

/// Of two no-copy buffers over one memory, which lie in one import, a loaded pointer reaches the
/// one a command used and not the other until a command uses it, as Metal's (an M4 under macOS
/// 27.0.1 and the M3 Max under 26.7, plain and validated: a pointer loaded from memory or an
/// argument buffer's reaches B only after a command used B; an earlier command buffer's, an earlier
/// encoder's or the dispatch's own use of A leaves B's accesses dropped).
#[test]
fn a_loaded_pointer_reaches_only_the_buffer_over_a_memory_a_command_used() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let resources = CaseResources::new(device, &walker).unwrap();
        let [member] = resources.argument_buffers.values().collect::<Vec<_>>()[..] else {
            panic!("{POINTER_WALKER}: one argument-buffer buffer");
        };
        let bytes = read_buffer(member, 0, member.length()).unwrap();
        let len = bytes.len().next_multiple_of(host_page());
        let memory = Mapping::anonymous(len);
        let import = || {
            // SAFETY: the mapping outlives both buffers, which are dropped first.
            unsafe {
                no_copy(
                    device,
                    memory.pointer(),
                    len,
                    ResourceOptions::shared(),
                    None,
                )
            }
        };
        let used = match import() {
            Err(error) if not_importable(&error) => {
                eprintln!("NO-COPY-REFUSED {error:?}: nothing to run");
                return;
            }
            used => used.unwrap(),
        };
        let other = import().unwrap();
        gpu_read(&queue, &used, 0..4);
        memory.write(0, &bytes);
        let produced = walk_unnamed(device, &queue, &walker, &used, None);
        assert!(produced == walker.metal_output, "the buffer a command used");
        memory.write(0, &bytes);
        let produced = walk_unnamed(device, &queue, &walker, &other, None);
        assert!(
            produced != walker.metal_output,
            "the other buffer over its memory, which no command used"
        );
        gpu_read(&queue, &other, 0..4);
        memory.write(0, &bytes);
        let produced = walk_unnamed(device, &queue, &walker, &other, None);
        assert!(
            produced == walker.metal_output,
            "the other buffer once a command used it"
        );
        drop((used, other));
    });
}

/// The bytes the pointer walker's argument-buffer buffer holds: what a buffer it reaches unnamed
/// must hold for its walk to give Metal's output.
fn walked_bytes(device: &Device, walker: &PublicCase) -> Vec<u8> {
    let resources = CaseResources::new(device, walker).unwrap();
    let [member] = resources.argument_buffers.values().collect::<Vec<_>>()[..] else {
        panic!("{POINTER_WALKER}: one argument-buffer buffer");
    };
    read_buffer(member, 0, member.length()).unwrap()
}

/// A no-copy buffer over the `len` bytes of `memory`; `None`, printed, where the venue refuses
/// the import.
fn no_copy_over(device: &Device, memory: &Mapping, len: usize) -> Option<Buffer> {
    // SAFETY: the caller keeps the mapping until the buffer is gone.
    let made = unsafe {
        no_copy(
            device,
            memory.pointer(),
            len,
            ResourceOptions::shared(),
            None,
        )
    };
    match made {
        Err(error) if not_importable(&error) => {
            eprintln!("NO-COPY-REFUSED {error:?}: nothing to run");
            None
        }
        made => Some(made.unwrap()),
    }
}

/// A rebind of a no-copy buffer's pages ends its use, as Metal's M3 Max ends it (the consumer's
/// measurement, three runs alike, through a loaded pointer and an indirect command's kernel
/// buffer: a buffer used, then detached and replaced, reads 0 and drops writes): a pointer loaded
/// from memory reaches neither the old pages nor the new until a command uses the buffer again,
/// and then reaches the new pages.
#[test]
fn a_loaded_pointer_reaches_a_rebound_no_copy_buffer_once_a_command_used_it_again() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let bytes = walked_bytes(device, &walker);
        let len = bytes.len().next_multiple_of(host_page());
        let old = Mapping::anonymous(len);
        old.write(0, &bytes);
        let Some(pages) = no_copy_over(device, &old, len) else {
            return;
        };
        gpu_read(&queue, &pages, 0..4);
        old.write(0, &bytes);
        let produced = walk_unnamed(device, &queue, &walker, &pages, None);
        assert!(produced == walker.metal_output, "a buffer a command used");
        let new = Mapping::anonymous(len);
        new.write(0, &bytes);
        // SAFETY: `new` outlives the buffer, which is dropped first, and no other thread uses it.
        unsafe { pages.rebind_bytes_no_copy(HostMemory::new(new.pointer()), None, None) }.unwrap();
        old.write(0, &bytes);
        let produced = walk_unnamed(device, &queue, &walker, &pages, None);
        assert!(
            produced != walker.metal_output,
            "rebound since a command used it"
        );
        gpu_read(&queue, &pages, 0..4);
        old.fill(0);
        new.write(0, &bytes);
        let produced = walk_unnamed(device, &queue, &walker, &pages, None);
        assert!(
            produced == walker.metal_output,
            "the new pages, once a command used it again"
        );
        drop(pages);
    });
}

/// A no-copy buffer rebound before any command used it is reached unnamed once a command uses it
/// after the rebind, over the new pages, and not before, as Metal's M3 Max (the consumer's
/// measurement: never used, replaced, then used, reads and writes the new pages; never used and
/// replaced, 0).
#[test]
fn a_loaded_pointer_reaches_a_no_copy_buffer_rebound_before_its_use_once_a_command_used_it() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let bytes = walked_bytes(device, &walker);
        let len = bytes.len().next_multiple_of(host_page());
        let old = Mapping::anonymous(len);
        old.write(0, &bytes);
        let Some(pages) = no_copy_over(device, &old, len) else {
            return;
        };
        let new = Mapping::anonymous(len);
        new.write(0, &bytes);
        // SAFETY: `new` outlives the buffer, which is dropped first, and no other thread uses it.
        unsafe { pages.rebind_bytes_no_copy(HostMemory::new(new.pointer()), None, None) }.unwrap();
        let produced = walk_unnamed(device, &queue, &walker, &pages, None);
        assert!(produced != walker.metal_output, "no command used it");
        gpu_read(&queue, &pages, 0..4);
        old.fill(0);
        new.write(0, &bytes);
        let produced = walk_unnamed(device, &queue, &walker, &pages, None);
        assert!(
            produced == walker.metal_output,
            "the new pages, once a command used it"
        );
        drop(pages);
    });
}

/// A rebind of a heap's pages ends the heap's use, as Metal's M3 Max ends it (the consumer's
/// measurement: a heap used, then detached and replaced, reads 0 through its buffers and drops
/// their writes): a pointer loaded from memory reaches a buffer placed in it only once a command
/// uses the heap again (here through its other buffer), and then the new pages.
#[test]
fn a_loaded_pointer_reaches_a_rebound_heap_s_buffer_once_a_command_used_the_heap_again() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let bytes = walked_bytes(device, &walker);
        const AT: usize = 4096;
        let old = Mapping::anonymous(HEAP);
        // SAFETY: the mapping outlives the heap and its buffers, which are dropped first.
        let Some(heap) =
            (unsafe { no_copy_heap(device, old.pointer(), HeapType::Placement, None) })
        else {
            return;
        };
        let placed = heap
            .new_buffer_at_offset(
                bytes.len() as u64,
                shared(HazardTrackingMode::Tracked),
                AT as u64,
            )
            .unwrap();
        let other = heap
            .new_buffer_at_offset(256, shared(HazardTrackingMode::Tracked), 0)
            .unwrap();
        gpu_read(&queue, &other, 0..4);
        old.write(AT, &bytes);
        let produced = walk_unnamed(device, &queue, &walker, &placed, None);
        assert!(produced == walker.metal_output, "a heap a command used");
        let new = Mapping::anonymous(HEAP);
        new.write(AT, &bytes);
        // SAFETY: `new` outlives the heap, which is dropped first, and no other thread uses it.
        unsafe { heap.rebind_bytes_no_copy(HostMemory::new(new.pointer()), None, None) }.unwrap();
        old.write(AT, &bytes);
        let produced = walk_unnamed(device, &queue, &walker, &placed, None);
        assert!(
            produced != walker.metal_output,
            "rebound since a command used the heap"
        );
        gpu_read(&queue, &other, 0..4);
        old.fill(0);
        new.write(AT, &bytes);
        let produced = walk_unnamed(device, &queue, &walker, &placed, None);
        assert!(
            produced == walker.metal_output,
            "the new pages, once a command used the heap again"
        );
        drop((placed, other, heap));
    });
}

/// Of two no-copy buffers over one memory, which lie in one import, a rebind of one ends only its
/// own use: a loaded pointer still reaches the other, over the memory they shared, as each keeps
/// its own use on Metal and a rebind clears the rebound one's (the consumer's measurement on the M3
/// Max).
#[test]
fn a_rebind_of_one_of_two_buffers_over_a_memory_ends_only_its_use() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let bytes = walked_bytes(device, &walker);
        let len = bytes.len().next_multiple_of(host_page());
        let memory = Mapping::anonymous(len);
        memory.write(0, &bytes);
        let Some(rebound) = no_copy_over(device, &memory, len) else {
            return;
        };
        let kept = no_copy_over(device, &memory, len).unwrap();
        gpu_read(&queue, &rebound, 0..4);
        gpu_read(&queue, &kept, 0..4);
        memory.write(0, &bytes);
        let produced = walk_unnamed(device, &queue, &walker, &kept, None);
        assert!(produced == walker.metal_output, "the kept buffer, used");
        let new = Mapping::anonymous(len);
        new.write(0, &bytes);
        // SAFETY: `new` outlives the buffer, which is dropped first, and no other thread uses it.
        unsafe { rebound.rebind_bytes_no_copy(HostMemory::new(new.pointer()), None, None) }
            .unwrap();
        memory.write(0, &bytes);
        let produced = walk_unnamed(device, &queue, &walker, &rebound, None);
        assert!(
            produced != walker.metal_output,
            "the rebound buffer, rebound since its use"
        );
        memory.write(0, &bytes);
        let produced = walk_unnamed(device, &queue, &walker, &kept, None);
        assert!(
            produced == walker.metal_output,
            "the other buffer over the memory keeps its use"
        );
        drop((rebound, kept));
    });
}

/// The entries the root tables' search covered before, its unrolled steps' reach since.
const OLD_SEARCH_CAPACITY: usize = 1 << 16;

/// Buffers of their own past what the root tables' search covered, 2^16 entries after
/// coalescing, none inside another: a pointer into a buffer of its own that no encoder made
/// resident reaches it as Metal's does (the search's loop past its unrolled steps), and nothing is
/// announced. Before, it read zero there, announced (`unnamed-buffers-past-capacity`).
#[test]
fn a_resident_set_past_the_old_search_reaches_every_live_buffer() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let live: Vec<Buffer> = (0..=OLD_SEARCH_CAPACITY)
            .map(|_| device.new_buffer(16, ResourceOptions::shared()).unwrap())
            .collect();
        let resident = run_case(&device, &walker).unwrap();
        assert!(
            resident.output == walker.metal_output,
            "the buffer its encoder made resident"
        );
        let resources = CaseResources::new(&device, &walker).unwrap();
        let [member] = resources.argument_buffers.values().collect::<Vec<_>>()[..] else {
            panic!("{POINTER_WALKER}: one argument-buffer buffer");
        };
        let bytes = read_buffer(member, 0, member.length()).unwrap();
        let own = device
            .new_buffer(bytes.len() as u64, ResourceOptions::shared())
            .unwrap();
        write(&own, &bytes);
        let produced = walk_unnamed(&device, &queue, &walker, &own, None);
        assert!(
            produced == walker.metal_output,
            "a buffer no encoder named, among {} live buffers",
            live.len() + 1
        );
        // On a device without unified memory, the `Shared` buffers lie in system memory,
        // announced as the first is made (`blits::SYSTEM_MEMORY`).
        let expected: &[Degradation] = match device.properties().has_unified_memory {
            true => &[],
            false => &[Degradation::SharedInSystemMemory],
        };
        assert_once(&messages.take_notices(), expected);
        println!(
            "CBF-MARK {}: a walk through a buffer no encoder named, among {} live buffers, \
             read Metal's output",
            device.name(),
            live.len() + 1
        );
        drop(live);
    }
    layer_reported_nothing_at_all(&messages);
}

/// The counts of buffers of their own [`the_resident_set_s_cost`] adds, in turn.
const LIVE: [usize; 4] = [0, 1024, 8192, 32768];

/// The commits each of its times is the median of.
const COMMITS: usize = 15;

/// The cost of the command buffer's resident set, on a device without the validation layer, printed
/// as `RESIDENT-COST` lines. With [`LIVE`]'s count of buffers of their own (256 bytes each,
/// `Shared`) live beside the pointer walker's, the median wall time of the `commit` of a command
/// buffer of one walk, whose plan reads a root table (the commit's other work included): `steady`,
/// no buffer made or gone since the last commit, so the device's table is reused; `churn`, a buffer
/// dropped and one made before each commit, so it is made again. And the table's bytes: 32 for each
/// buffer, at most (adjacent ranges coalesce). Ignored: a cost run takes it in release mode on an
/// Apple and a Linux venue.
#[test]
#[ignore = "a measurement: a cost run takes it in release mode"]
fn the_resident_set_s_cost() {
    let cache = ScratchCache::new().unwrap();
    let mut options = validated_options(&Messages::default(), cache.path());
    options.layers.clear();
    let device = open_device(TEST_DEVICE, options).unwrap();
    let queue = device.new_command_queue().unwrap();
    let walker = public_case(POINTER_WALKER).unwrap();
    let walk = pipeline_state(&device, &walker).unwrap();
    let resources = CaseResources::new(&device, &walker).unwrap();
    let buffers = bound_buffers(&device, &walker.compute, &resources).unwrap();
    let commit = || {
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encode_case(&mut encoder, &walk, &walker.compute, &buffers, &resources).unwrap();
        encoder.end_encoding().unwrap();
        let start = Instant::now();
        command_buffer.commit().unwrap();
        let took = start.elapsed();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        took
    };
    let median = |mut times: Vec<Duration>| {
        times.sort();
        times[times.len() / 2]
    };
    let own = || device.new_buffer(256, ResourceOptions::shared()).unwrap();
    let mut live = Vec::new();
    for count in LIVE {
        live.extend((live.len()..count).map(|_| own()));
        commit();
        let steady = median((0..COMMITS).map(|_| commit()).collect());
        let churn = median(
            (0..COMMITS)
                .map(|_| {
                    live.pop();
                    live.push(own());
                    commit()
                })
                .collect(),
        );
        println!(
            "RESIDENT-COST {}: live={count} steady_us={} churn_us={} table_bytes<={}",
            device.name(),
            steady.as_micros(),
            churn.as_micros(),
            32 * (count + buffers.len() + resources.argument_buffers.len())
        );
    }
}
