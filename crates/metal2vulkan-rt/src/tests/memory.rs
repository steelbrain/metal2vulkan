//! Memory blocks and buffers: the plan of a zero fill, a block's length and alignment, the idle
//! clock; and, under a device cfg, on every device of the cfg's kind, that the device makes an
//! allocation of each class as it is made and the next only while a class's free bytes run low,
//! that blocks are handed out and buffers view them, zeroed from their `zero_from` (a Private
//! buffer's fill ahead of a copy into it in the next command buffer), that the consumer's hook
//! maps each allocation before any block of it goes out and a take whose allocation it does not
//! map is refused, that a block given back is a free range at once, its bytes as written, the
//! first fit of a later take, that each idle allocation is freed at its idle age, under steady
//! commits too, that seeded takes and releases of mixed lengths all return, that a Private range
//! is zeroed when the consumer asks, that a take answers whether its bytes are the driver's zeros
//! and the runtime zeroes what it is asked over bytes written outside its buffers, and that
//! teardown frees everything with the gauges at zero.

use crate::fill::{plan, Plan};
use crate::memory::{base_align, idle_due, round, IDLE_AGE_NANOS};
use crate::{Error, Limit};

#[test]
fn a_zero_fill_fills_whole_words_and_copies_the_bytes_outside_them() {
    let fill = |fill: Option<(u64, u64)>, copies: &[(u64, u64)]| Plan {
        fill,
        copies: copies.to_vec(),
    };
    let cases = [
        // Whole words: one fill.
        ((0, 256), fill(Some((0, 256)), &[])),
        ((4096, 4100), fill(Some((4096, 4100)), &[])),
        // Words inside, with bytes before and after them.
        (
            (4097, 4110),
            fill(Some((4100, 4108)), &[(4097, 4100), (4108, 4110)]),
        ),
        ((1, 8), fill(Some((4, 8)), &[(1, 4)])),
        ((0, 7), fill(Some((0, 4)), &[(4, 7)])),
        // No whole word inside: one copy, at most 6 bytes.
        ((1, 7), fill(None, &[(1, 7)])),
        ((5, 7), fill(None, &[(5, 7)])),
        ((4099, 4100), fill(None, &[(4099, 4100)])),
        ((3, 3), fill(None, &[])),
    ];
    for ((offset, end), expected) in cases {
        let got = plan(offset, end);
        assert_eq!(got, expected, "{offset}..{end}");
        for &(start, end) in &got.copies {
            assert!(
                end - start <= 6,
                "a copy of {start}..{end} passes the zero source"
            );
        }
    }
}

/// A block's length is the length asked (at least 1) rounded up to the granule, within the
/// device's longest allocation and buffer.
#[test]
fn a_block_is_the_length_rounded_up_to_the_granule_within_the_device() {
    let max = 4 << 30;
    let granule = 64 << 10;
    assert_eq!(round(0, granule, max).unwrap(), granule);
    assert_eq!(round(1, granule, max).unwrap(), granule);
    assert_eq!(round(257 << 20, granule, max).unwrap(), 257 << 20);
    assert_eq!(
        round((256 << 20) + 1, granule, max).unwrap(),
        (256 << 20) + granule
    );
    assert_eq!(round(max, granule, max).unwrap(), max);
    assert_eq!(round(1, 128 << 10, max).unwrap(), 128 << 10);
    for (len, reported) in [(max + 1, max + granule), (u64::MAX, u64::MAX)] {
        match round(len, granule, max) {
            Err(Error::Limit(Limit::BlockLen { len, max: limit })) => {
                assert_eq!((len, limit), (reported, max))
            }
            other => panic!("{len} bytes: {other:?}"),
        }
    }
}

/// A block's first byte lies at a multiple of its length rounded up to a power of two, at most
/// 2 MiB, and at least its buffer's alignment and the granule.
#[test]
fn a_block_starts_at_its_length_rounded_to_a_power_of_two_up_to_2_mib() {
    const KIB: u64 = 1 << 10;
    const MIB: u64 = 1 << 20;
    let granule = 64 * KIB;
    for (len, alignment, expected) in [
        (64 * KIB, 256, 64 * KIB),
        (192 * KIB, 256, 256 * KIB),
        (MIB, 256, MIB),
        (3 * MIB, 256, 2 * MIB),
        (16 * MIB, 256, 2 * MIB),
        (16 * MIB, 4 * MIB, 4 * MIB),
        (64 * KIB, 128 * KIB, 128 * KIB),
    ] {
        assert_eq!(base_align(len, alignment, granule), expected, "{len}");
    }
}

/// The idle clock: what became idle is due once the idle age has passed, and before then names
/// the host time it becomes due.
#[test]
fn the_idle_clock_is_due_at_the_idle_age() {
    let since = 1_000;
    let due = since + IDLE_AGE_NANOS;
    assert_eq!(idle_due(since, since), Some(due));
    assert_eq!(idle_due(since, due - 1), Some(due));
    assert_eq!(idle_due(since, due), None);
    assert_eq!(idle_due(since, u64::MAX), None);
    assert_eq!(idle_due(u64::MAX, 0), Some(u64::MAX), "no overflow");
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::device::lock;
    use crate::memory::IDLE_AGE_NANOS;
    use crate::tests::command::on_device::finished;
    use crate::tests::compute::on_device::{add, WORDS};
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{
        AllocationId, AllocationInfo, Block, BlockId, BlockInfo, BlockMappings, Buffer,
        CommandBufferStatus, ComputePipelineState, Device, DeviceOptions, DispatchType, Error,
        Gauges, Limit, MemoryClass, PixelFormat, Refusal, TextureDescriptor, TextureType,
        MAX_BUFFER_LENGTH,
    };
    use ash::vk;
    use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    /// The runtime's own allocation besides the pool's: the zero source of Private fills.
    const ZERO_SOURCE: u64 = 1;

    /// The consumer's hooks as a test sees them: each allocation mapped, as the runtime described
    /// it and the thread the map ran on, and each unmap, the thread it ran on and the
    /// allocation's first byte read through the mapping the consumer was told of. While `refuse`
    /// is set, it maps nothing.
    #[derive(Default)]
    struct Hook {
        mapped: Mutex<Vec<Mapped>>,
        unmapped: Mutex<Vec<Unmapped>>,
        refuse: AtomicBool,
    }

    #[derive(Clone, Debug)]
    struct Mapped {
        allocation: AllocationId,
        class: MemoryClass,
        len: u64,
        gpu_address: u64,
        contents: Option<usize>,
        thread: Option<String>,
    }

    #[derive(Clone, Debug)]
    struct Unmapped {
        allocation: AllocationId,
        thread: Option<String>,
        first: Option<u8>,
    }

    impl BlockMappings for Hook {
        fn map(&self, allocation: &AllocationInfo) -> bool {
            if self.refuse.load(Relaxed) {
                return false;
            }
            lock(&self.mapped).push(Mapped {
                allocation: allocation.id,
                class: allocation.class,
                len: allocation.len,
                gpu_address: allocation.gpu_address,
                contents: allocation.contents.map(|c| c.as_ptr() as usize),
                thread: thread::current().name().map(str::to_string),
            });
            true
        }

        fn unmap(&self, allocation: AllocationId) {
            let contents = (lock(&self.mapped).iter())
                .find(|mapped| mapped.allocation == allocation)
                .and_then(|mapped| mapped.contents);
            let first = contents.map(|address| {
                // SAFETY: the runtime keeps the memory mapped until this returns.
                unsafe { (address as *const u8).read_volatile() }
            });
            lock(&self.unmapped).push(Unmapped {
                allocation,
                thread: thread::current().name().map(str::to_string),
                first,
            });
        }
    }

    impl Hook {
        fn mapped(&self) -> Vec<Mapped> {
            lock(&self.mapped).clone()
        }

        fn unmapped(&self) -> Vec<Unmapped> {
            lock(&self.unmapped).clone()
        }

        /// The block's allocation was mapped before the block went out, as the runtime described
        /// it: its class, the block inside it, at the allocation's address and mapping plus the
        /// block's offset.
        fn assert_mapped(&self, name: &str, block: &BlockInfo) {
            let mapped = self.mapped();
            let allocation = (mapped.iter())
                .find(|mapped| mapped.allocation == block.allocation)
                .unwrap_or_else(|| {
                    panic!("{name}: block {:?} of an allocation not mapped", block.id)
                });
            assert_eq!(allocation.class, block.class, "{name}");
            assert!(
                block.offset + block.len <= allocation.len,
                "{name}: block {:?} at {} of {} bytes in {} bytes",
                block.id,
                block.offset,
                block.len,
                allocation.len
            );
            assert_eq!(
                (
                    block.gpu_address,
                    block.contents.map(|c| c.as_ptr() as usize)
                ),
                (
                    allocation.gpu_address + block.offset,
                    allocation.contents.map(|c| c + block.offset as usize)
                ),
                "{name}: block {:?}'s address and mapping",
                block.id
            );
        }
    }

    fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(120);
        while !done() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// A Shared block's bytes.
    fn bytes(block: &BlockInfo) -> &'static mut [u8] {
        let contents = block.contents.expect("a Shared block's contents");
        // SAFETY: the mapping spans the block, which the test holds while it uses the slice.
        unsafe { std::slice::from_raw_parts_mut(contents.as_ptr(), block.len as usize) }
    }

    /// The `VkBuffer` commands reach a held block through, its allocation's, and the offset of the
    /// block's first byte in it.
    fn vk_buffer(device: &Device, block: BlockId) -> (vk::Buffer, u64) {
        device.block(block).expect("a held block").memory.range(0)
    }

    /// Transfers the test records on the runtime's queue, each waited for, with every earlier
    /// write visible to it and its writes visible to the host.
    struct Gpu {
        device: Device,
        pool: vk::CommandPool,
        command: vk::CommandBuffer,
        fence: vk::Fence,
    }

    impl Gpu {
        fn new(device: &Device) -> Gpu {
            let raw = device.raw();
            let pool =
                vk::CommandPoolCreateInfo::default().queue_family_index(raw.queue_family_index);
            // SAFETY: the device is live; the objects are destroyed in `drop`.
            unsafe {
                let pool = raw.device.create_command_pool(&pool, None).expect("a pool");
                let command = raw
                    .device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                    .expect("a command buffer")[0];
                let fence = raw
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .expect("a fence");
                Gpu {
                    device: device.clone(),
                    pool,
                    command,
                    fence,
                }
            }
        }

        fn run(&self, record: impl FnOnce(&ash::Device, vk::CommandBuffer)) {
            let raw = self.device.raw();
            let device = raw.device;
            let command = self.command;
            let barrier = |dst_stage, dst_access| {
                let barrier = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                    .dst_stage_mask(dst_stage)
                    .dst_access_mask(dst_access)];
                // SAFETY: the command buffer is recording.
                unsafe {
                    device.cmd_pipeline_barrier2(
                        command,
                        &vk::DependencyInfo::default().memory_barriers(&barrier),
                    )
                };
            };
            // SAFETY: the pool and fence are this test's; each run waits for its work.
            unsafe {
                device
                    .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                    .expect("a reset");
                device
                    .begin_command_buffer(
                        command,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .expect("a begin");
            }
            barrier(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            );
            record(device, command);
            barrier(vk::PipelineStageFlags2::HOST, vk::AccessFlags2::HOST_READ);
            let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
            let submit = vk::SubmitInfo2::default().command_buffer_infos(&commands);
            // SAFETY: as above; the queue is held while it is used.
            unsafe {
                device.end_command_buffer(command).expect("an end");
                self.device
                    .with_runtime_queue(|queue| device.queue_submit2(queue, &[submit], self.fence))
                    .expect("a submit");
                device
                    .wait_for_fences(&[self.fence], true, u64::MAX)
                    .expect("the transfer completes");
                device.reset_fences(&[self.fence]).expect("a fence reset");
            }
        }

        /// Fills `size` bytes from `offset` of the block that `(buffer, base)` reaches
        /// ([`vk_buffer`]).
        fn fill(&self, (buffer, base): (vk::Buffer, u64), offset: u64, size: u64, word: u32) {
            // SAFETY: recorded in `run`; the range is in the block and word-aligned.
            self.run(|device, command| unsafe {
                device.cmd_fill_buffer(command, buffer, base + offset, size, word)
            });
        }
    }

    impl Drop for Gpu {
        fn drop(&mut self) {
            let raw = self.device.raw();
            // SAFETY: no work of the test is pending.
            unsafe {
                raw.device.destroy_fence(self.fence, None);
                raw.device.destroy_command_pool(self.pool, None);
            }
        }
    }

    fn device_with(hook: &Arc<Hook>, physical: &crate::PhysicalDevice, name: &str) -> Device {
        let options = DeviceOptions {
            block_mappings: Some(Arc::clone(hook) as Arc<dyn BlockMappings>),
            ..DeviceOptions::default()
        };
        Device::new(physical, options).unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// The matrix's leak records: each kind's live count after the device dropped.
    fn print_gauges_at_drop(test: &str, gauges: Gauges) {
        for (kind, value) in [("buffers", gauges.buffers), ("blocks", gauges.blocks)] {
            println!("M2V-MATRIX gauge {test}/{kind}@drop {value}");
        }
    }

    /// Every allocation the hook mapped was unmapped exactly once by the end, and no other.
    fn assert_each_unmapped_once(name: &str, hook: &Hook) {
        let mut unmapped: Vec<AllocationId> =
            hook.unmapped().iter().map(|u| u.allocation).collect();
        unmapped.sort_unstable();
        let mut mapped: Vec<AllocationId> = hook.mapped().iter().map(|m| m.allocation).collect();
        mapped.sort_unstable();
        assert_eq!(unmapped, mapped, "{name}: the allocations unmapped");
    }

    /// A buffer of exactly [`MAX_BUFFER_LENGTH`] (2 GiB) is made in a Private block of that
    /// length, and one a byte longer is refused with a typed refusal, before its range is looked
    /// at; with the block released and the device dropped, every gauge is zero. A device whose
    /// largest block or memory is smaller skips the 2 GiB buffer visibly.
    #[test]
    fn a_buffer_of_2_gib_is_made_and_a_longer_one_refused() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(
            !devices.is_empty(),
            "the loader lists no device of this cfg's kind"
        );
        for (name, physical) in &devices {
            let hook = Arc::new(Hook::default());
            let device = device_with(&hook, physical, name);
            let counters = device.counters();
            let block = match device.take_block(MemoryClass::Private, MAX_BUFFER_LENGTH) {
                Ok(block) => block,
                Err(
                    error @ (Error::Limit(Limit::BlockLen { .. })
                    | Error::Vulkan {
                        result:
                            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                            | vk::Result::ERROR_OUT_OF_HOST_MEMORY,
                        ..
                    }),
                ) => {
                    eprintln!("SKIPPED: {name}: no Private block of 2 GiB: {error}");
                    continue;
                }
                Err(error) => panic!("{name}: {error}"),
            };
            assert_eq!(block.len, MAX_BUFFER_LENGTH, "{name}");
            let buffer = device
                .new_buffer(block.id, 0, MAX_BUFFER_LENGTH, None)
                .expect("a buffer of 2 GiB");
            assert_eq!(
                (buffer.gpu_address(), buffer.length()),
                (block.gpu_address, MAX_BUFFER_LENGTH),
                "{name}"
            );
            match device.new_buffer(block.id, 0, MAX_BUFFER_LENGTH + 1, None) {
                Err(Error::Refused(Refusal::BufferTooLong { length, max })) => {
                    assert_eq!((length, max), (MAX_BUFFER_LENGTH + 1, MAX_BUFFER_LENGTH))
                }
                other => panic!("{name}: a buffer of 2 GiB and a byte: {:?}", other.err()),
            }
            assert_eq!(
                counters.gauges().buffers,
                1,
                "{name}: only the 2 GiB buffer"
            );
            drop(buffer);
            device.release_block(block).expect("a release");
            drop(device);
            let gauges = counters.gauges();
            print_gauges_at_drop("buffer_2_gib", gauges);
            assert_eq!(gauges, Gauges::default(), "{name}: gauges after teardown");
            println!("memory: {name}: a buffer of 2 GiB made, 2 GiB and a byte refused");
        }
    }

    /// Blocks of both classes are handed out, buffers view them, a zeroed buffer zeroes exactly
    /// its bytes from its `zero_from`, counted as the buffer's bytes and not the block's, the
    /// refusals are Metal's, each block's allocation was mapped through the hook before the block
    /// went out, a block longer than an allocation has an allocation of its own, a released block
    /// lives until its last buffer is gone and its allocation is then idle, its bytes as written,
    /// and freed at its idle age after the hook's unmap on the block-maker thread, still mapped;
    /// and teardown unmaps and frees the rest on the dropping thread, with every gauge at zero.
    #[test]
    fn every_device_hands_out_blocks_and_buffers_and_frees_them_through_the_hook() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(
            !devices.is_empty(),
            "the loader lists no device of this cfg's kind"
        );
        let this_thread = thread::current().name().map(str::to_string);
        for (name, physical) in &devices {
            let hook = Arc::new(Hook::default());
            let device = device_with(&hook, physical, name);
            let core = device.core();
            let counters = device.counters();
            let gpu = Gpu::new(&device);

            // A new Shared block, mapped.
            let shared = device
                .take_block(MemoryClass::Shared, MemoryClass::Shared.block_len())
                .expect("a Shared block");
            assert_eq!(
                (shared.class, shared.len),
                (MemoryClass::Shared, MemoryClass::Shared.block_len())
            );
            assert_ne!(shared.gpu_address, 0, "{name}: a Shared block's address");
            let host = bytes(&shared);

            // A new Private block, unmapped.
            let private = device
                .take_block(MemoryClass::Private, MemoryClass::Private.block_len())
                .expect("a Private block");
            assert_eq!(
                (private.class, private.len, private.contents),
                (MemoryClass::Private, MemoryClass::Private.block_len(), None)
            );
            assert_ne!(private.gpu_address, 0, "{name}: a Private block's address");
            assert_ne!(private.id, shared.id);
            hook.assert_mapped(name, &shared);
            hook.assert_mapped(name, &private);

            // Buffers are views: the block's address and mapping plus the offset.
            let view = device
                .new_buffer(shared.id, 4096, 256, None)
                .expect("a Shared buffer");
            let base = shared.contents.unwrap().as_ptr() as usize;
            assert_eq!(view.gpu_address(), shared.gpu_address + 4096);
            assert_eq!(
                view.contents().map(|c| c.as_ptr() as usize),
                Some(base + 4096)
            );
            assert_eq!(
                (view.block(), view.offset(), view.length()),
                (shared.id, 4096, 256)
            );
            let whole = device
                .new_buffer(private.id, 0, MemoryClass::Private.block_len(), None)
                .expect("a Private buffer");
            assert_eq!(
                (whole.gpu_address(), whole.contents()),
                (private.gpu_address, None)
            );
            let clone = view.clone();
            let live = counters.gauges();
            assert_eq!(live.buffers, 2, "{name}: a clone is the same buffer");
            assert!(live.blocks >= ZERO_SOURCE + 2, "{name}: {live:?}");
            assert!(counters.metrics().blocks_made >= ZERO_SOURCE + 2);
            drop(clone);
            assert_eq!(counters.gauges().buffers, 2);

            // A zeroed buffer zeroes exactly its bytes from its `zero_from`, counted as the
            // buffer's bytes: in a Shared block through the mapping, in a Private one by a fill of
            // its whole words and copies of the bytes beside them.
            for zero_from in [Some(0), Some(3), Some(7), Some(9), None] {
                host[..8192].fill(0xFF);
                let before = counters.metrics().bytes_zeroed;
                let zeroed = device
                    .new_buffer(shared.id, 4097, 7, zero_from)
                    .expect("a zeroed Shared buffer");
                let from = 4097 + zero_from.unwrap_or(7).min(7) as usize;
                for (index, &byte) in host[..8192].iter().enumerate() {
                    let inside = (from..4104).contains(&index);
                    assert_eq!(
                        byte,
                        if inside { 0 } else { 0xFF },
                        "{name}: {zero_from:?}, Shared byte {index}"
                    );
                }
                assert_eq!(
                    counters.metrics().bytes_zeroed - before,
                    (4104 - from) as u64,
                    "{name}: {zero_from:?}, the Shared bytes zeroed"
                );
                drop(zeroed);
            }
            // A zeroed Private range is filled ahead of the next command buffer committed, whose
            // commands see the fill.
            let queue = device.new_command_queue(1).expect("a queue");
            for (offset, length, zero_from) in [
                (4097, 13, 0),
                (8001, 2, 0),
                (4096, 64, 0),
                (4099, 1, 0),
                (1, 8000, 0),
                (4096, 64, 5),
                (4097, 13, 12),
                (1, 8000, 4000),
            ] {
                gpu.fill(vk_buffer(&device, private.id), 0, 8192, u32::MAX);
                let before = counters.metrics().bytes_zeroed;
                let zeroed = device
                    .new_buffer(private.id, offset, length, Some(zero_from))
                    .expect("a zeroed Private buffer");
                assert_eq!(
                    counters.metrics().bytes_zeroed - before,
                    length - zero_from,
                    "{name}: {offset}+{length} from {zero_from}, the Private bytes zeroed"
                );
                let ((from, src_offset), (to, dst_offset)) = (
                    vk_buffer(&device, private.id),
                    vk_buffer(&device, shared.id),
                );
                let copy = queue.command_buffer().expect("a command buffer");
                copy.record_for_test(&[], |device, command| {
                    let region = [vk::BufferCopy {
                        src_offset,
                        dst_offset,
                        size: 8192,
                    }];
                    // SAFETY: the command buffer is recording; both blocks are held and span the
                    // region.
                    unsafe { device.cmd_copy_buffer(command, from, to, &region) };
                })
                .expect("a copy");
                copy.commit().expect("a commit");
                copy.wait_until_completed()
                    .expect("a wait outside a scheduled handler");
                assert_eq!(copy.status(), CommandBufferStatus::Completed, "{name}");
                let range = (offset + zero_from) as usize..(offset + length) as usize;
                for (index, &byte) in host[..8192].iter().enumerate() {
                    let expected = if range.contains(&index) { 0 } else { 0xFF };
                    assert_eq!(
                        byte, expected,
                        "{name}: {offset}+{length} from {zero_from}, Private byte {index}"
                    );
                }
                drop(zeroed);
            }

            // Metal's refusals, and the device's limit.
            assert!(matches!(
                device.new_buffer(shared.id, 0, 0, None),
                Err(Error::Refused(Refusal::EmptyBuffer))
            ));
            // Past 2 GiB, whatever the block: the length is refused before the range.
            for length in [MAX_BUFFER_LENGTH + 1, u64::MAX] {
                match device.new_buffer(shared.id, 0, length, None) {
                    Err(Error::Refused(Refusal::BufferTooLong { length: l, max })) => {
                        assert_eq!((l, max), (length, MAX_BUFFER_LENGTH), "{name}")
                    }
                    other => panic!("{name}: a buffer of {length}: {:?}", other.err()),
                }
            }
            let shared_len = MemoryClass::Shared.block_len();
            for (offset, length) in [(shared_len - 4, 8), (u64::MAX, 2), (shared_len, 1)] {
                match device.new_buffer(shared.id, offset, length, None) {
                    Err(Error::Refused(Refusal::BufferPastBlock {
                        offset: o,
                        length: l,
                        block_len,
                    })) => assert_eq!((o, l, block_len), (offset, length, shared_len)),
                    other => panic!("{name}: {offset}+{length}: {:?}", other.err()),
                }
            }
            assert!(matches!(
                device.take_block(MemoryClass::Shared, u64::MAX),
                Err(Error::Limit(Limit::BlockLen { len: u64::MAX, .. }))
            ));

            // A block longer than an allocation: an allocation of its own length, rounded up to
            // the granule.
            let asked = 2 * MemoryClass::Shared.block_len() + 1;
            let long = device
                .take_block(MemoryClass::Shared, asked)
                .expect("a long block");
            assert!(
                long.len >= asked && long.len.is_multiple_of(64 << 10),
                "{name}: a long block of {} bytes",
                long.len
            );
            hook.assert_mapped(name, &long);
            assert_eq!(long.offset, 0, "{name}: the first block of its allocation");
            let long_allocation = long.allocation;
            let long_host = bytes(&long);
            long_host[0] = 0x5A;
            let last_view: Buffer = device
                .new_buffer(long.id, 0, 16, None)
                .expect("a buffer of the long block");

            // Released, it lives while its buffer does, then its allocation is idle.
            let long_id = long.id;
            device.release_block(long).expect("a release");
            assert!(matches!(
                device.new_buffer(long_id, 0, 16, None),
                Err(Error::NoSuchBlock(id)) if id == long_id
            ));
            assert!(
                hook.unmapped().is_empty(),
                "{name}: unmapped with a buffer alive"
            );
            // SAFETY: the buffer keeps the block mapped.
            let first = unsafe { last_view.contents().unwrap().as_ptr().read_volatile() };
            assert_eq!(first, 0x5A);
            drop(last_view);
            wait_until("the long block given back", || core.pool.holds(long_id));
            assert!(
                hook.unmapped().is_empty(),
                "{name}: unmapped as it went idle"
            );

            // At its idle age, the block-maker thread frees its allocation through the hook,
            // still mapped, its bytes as written.
            wait_until("the long block's unmap", || !hook.unmapped().is_empty());
            let unmapped = hook.unmapped();
            assert_eq!(unmapped.len(), 1, "{name}: {unmapped:?}");
            assert_eq!(unmapped[0].allocation, long_allocation);
            assert_eq!(unmapped[0].thread.as_deref(), Some("m2v-blocks"));
            assert_eq!(
                unmapped[0].first,
                Some(0x5A),
                "{name}: unmapped while still mapped, its bytes as written"
            );

            // Teardown unmaps the rest on this thread, and frees every block.
            drop((view, whole, gpu, queue));
            drop(device);
            let gauges = counters.gauges();
            print_gauges_at_drop("blocks_teardown", gauges);
            assert_eq!(gauges, Gauges::default(), "{name}: gauges after teardown");
            assert_each_unmapped_once(name, &hook);
            for unmapped in &hook.unmapped()[1..] {
                assert_eq!(unmapped.thread, this_thread, "{name}: {unmapped:?}");
            }
            println!(
                "memory: {name}: blocks and buffers; metrics {:?}",
                counters.metrics()
            );
        }
    }

    /// The device's making makes one allocation of each class before it returns, before the
    /// block-maker thread started, which then makes none while small takes leave 8 MiB or more
    /// free: a Shared take of a byte is a range of the first allocation. A take that leaves fewer
    /// free sets the block-maker thread off, which makes the class's next allocation off the
    /// taker's thread.
    #[test]
    fn every_device_makes_an_allocation_only_while_a_class_s_free_bytes_run_low() {
        const MIB: u64 = 1 << 20;
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::new(physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let core = device.core();
            let counters = device.counters();
            let before = counters.metrics();
            let made = before.blocks_made;
            let len = MemoryClass::Shared.block_len();
            assert!(
                core.pool.stocked(MemoryClass::Private) && core.pool.stocked(MemoryClass::Shared),
                "{name}: the first allocations as the device returns"
            );
            let gauges = counters.gauges();
            assert_eq!(
                (
                    gauges.blocks,
                    gauges.shared_idle_bytes,
                    gauges.private_idle_bytes
                ),
                (ZERO_SOURCE + 2, len, len),
                "{name}: the zero source and the first allocations"
            );
            let small = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|error| panic!("{name}: a small block: {error}"));
            thread::sleep(Duration::from_millis(200));
            assert_eq!(
                counters.metrics().blocks_made,
                made,
                "{name}: an allocation beside a small take"
            );
            assert_eq!(counters.gauges().shared_idle_bytes, 0, "{name}: in use");
            let half = (device.take_block(MemoryClass::Shared, 8 * MIB))
                .unwrap_or_else(|error| panic!("{name}: a block of 8 MiB: {error}"));
            wait_until("the next Shared allocation", || {
                counters.metrics().blocks_made == made + 1 && core.pool.stocked(MemoryClass::Shared)
            });
            let metrics = counters.metrics();
            assert_eq!(
                (metrics.block_bytes_made, metrics.consumer_allocations),
                (3 * len, before.consumer_allocations),
                "{name}: three allocations of 16 MiB, the third off this thread"
            );
            assert!(
                metrics.maker_allocation_nanos > before.maker_allocation_nanos,
                "{name}: the block-maker thread's allocation time"
            );
            drop((small, half));
            drop(device);
            assert_eq!(
                counters.gauges(),
                Gauges::default(),
                "{name}: gauges after teardown"
            );
            println!(
                "memory: {name}: an allocation made only below 8 MiB free; metrics {:?}",
                counters.metrics()
            );
        }
    }

    /// A block given back is a free range of its allocation at once, its bytes as written, and the
    /// first fit of a later take: a Shared block of 16 MiB, the device's first Shared allocation
    /// whole, is written and given back once the next allocation is made ahead of need; then of
    /// two takes of 16 MiB the first lands in the first block's bytes, a new number at its address
    /// and mapping, its bytes as written, and the second in that next allocation. Both are takes
    /// from free ranges: neither allocates nor calls the hook. A Private block the GPU wrote,
    /// given back, holds the GPU's bytes when a take of its length lands there again, the first
    /// fit; a buffer zeroed over a range of them reads zero there, ahead of the next command buffer
    /// committed, and nowhere else.
    #[test]
    fn every_device_takes_given_back_bytes_as_written_at_the_first_fit() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let hook = Arc::new(Hook::default());
            let device = device_with(&hook, physical, name);
            let core = device.core();
            let counters = device.counters();
            let len = MemoryClass::Shared.block_len();
            let first = (device.take_block(MemoryClass::Shared, len))
                .unwrap_or_else(|error| panic!("{name}: a Shared block: {error}"));
            assert_eq!(first.offset, 0, "{name}: the first Shared allocation whole");
            let info = *first;
            let host = bytes(&first);
            host[0] = 0x5A;
            host[host.len() - 1] = 0x5A;
            // The class's free bytes ran out: the block-maker thread makes the next allocation.
            wait_until("the next Shared allocation", || {
                core.pool.allocations(MemoryClass::Shared).len() == 2
            });
            let next = core.pool.allocations(MemoryClass::Shared)[1].0;
            device.release_block(first).expect("a release");
            wait_until("the block given back", || core.pool.holds(info.id));
            let before = counters.metrics();
            let again = (device.take_block(MemoryClass::Shared, len))
                .unwrap_or_else(|error| panic!("{name}: a block: {error}"));
            assert_eq!(
                (
                    again.allocation,
                    again.offset,
                    again.gpu_address,
                    again.contents
                ),
                (info.allocation, 0, info.gpu_address, info.contents),
                "{name}: the given-back bytes, the first fit"
            );
            assert_ne!(again.id, info.id, "{name}: each take is a new number");
            let host = bytes(&again);
            assert_eq!(
                (host[0], host[host.len() - 1]),
                (0x5A, 0x5A),
                "{name}: the given-back bytes, as written"
            );
            let second = (device.take_block(MemoryClass::Shared, len))
                .unwrap_or_else(|error| panic!("{name}: a block: {error}"));
            assert_eq!(
                (second.allocation, second.offset),
                (next, 0),
                "{name}: the allocation made ahead of need, next"
            );
            let after = counters.metrics();
            assert_eq!(
                (
                    after.takes_from_free_ranges - before.takes_from_free_ranges,
                    after.consumer_allocations - before.consumer_allocations,
                    after.allocations_mapped_on_takes - before.allocations_mapped_on_takes,
                ),
                (2, 0, 0),
                "{name}: two takes from free ranges, with no allocation and no hook"
            );
            assert!(hook.unmapped().is_empty(), "{name}: {:?}", hook.unmapped());

            // A Private block written by the GPU and given back holds the GPU's bytes when a take
            // lands there again; a buffer zeroed over a range of them reads zero there, ahead of
            // the next command buffer committed, and nowhere else.
            let gpu = Gpu::new(&device);
            let private_len = MemoryClass::Private.block_len();
            let private = (device.take_block(MemoryClass::Private, private_len))
                .unwrap_or_else(|error| panic!("{name}: a Private block: {error}"));
            let private_info = *private;
            gpu.fill(vk_buffer(&device, private.id), 0, private_len, u32::MAX);
            device.release_block(private).expect("a release");
            wait_until("the Private block given back", || {
                core.pool.holds(private_info.id)
            });
            let back = (device.take_block(MemoryClass::Private, private_len))
                .unwrap_or_else(|error| panic!("{name}: the Private bytes again: {error}"));
            assert_eq!(
                (back.allocation, back.gpu_address),
                (private_info.allocation, private_info.gpu_address),
                "{name}: the given-back Private bytes, the first fit"
            );
            let zeroed = (device.new_buffer(back.id, 4096, 4096, Some(0)))
                .unwrap_or_else(|error| panic!("{name}: a zeroed Private buffer: {error}"));
            host[..3 * 4096].fill(0x11);
            let queue = device.new_command_queue(1).expect("a queue");
            let ((from, src_offset), (to, dst_offset)) =
                (vk_buffer(&device, back.id), vk_buffer(&device, again.id));
            let copy = queue.command_buffer().expect("a command buffer");
            copy.record_for_test(&[], |device, command| {
                let region = [vk::BufferCopy {
                    src_offset,
                    dst_offset,
                    size: 3 * 4096,
                }];
                // SAFETY: the command buffer is recording; both blocks are held and span the
                // region.
                unsafe { device.cmd_copy_buffer(command, from, to, &region) };
            })
            .expect("a copy");
            copy.commit().expect("a commit");
            copy.wait_until_completed()
                .expect("a wait outside a scheduled handler");
            assert_eq!(copy.status(), CommandBufferStatus::Completed, "{name}");
            for (index, &byte) in host[..3 * 4096].iter().enumerate() {
                let expected = if (4096..8192).contains(&index) {
                    0
                } else {
                    0xFF
                };
                assert_eq!(
                    byte, expected,
                    "{name}: the Private bytes taken again, byte {index}"
                );
            }
            drop((zeroed, copy, queue));
            drop((second, again, back, gpu));
            drop(device);
            assert_eq!(
                counters.gauges(),
                Gauges::default(),
                "{name}: gauges after teardown"
            );
            assert_each_unmapped_once(name, &hook);
            println!(
                "memory: {name}: given-back bytes taken as written at the first fit; metrics {:?}",
                counters.metrics()
            );
        }
    }

    /// The consumer's hook maps each allocation of the pool before any block of it goes out, on
    /// the thread that made it, counted where it ran: each class's first in the device's making,
    /// on the thread that made the device; one a take that no free range fits makes, inside the
    /// take, on the taker's thread; one made ahead of need, on the block-maker thread. While the
    /// hook maps nothing, a take that no free range fits is refused with
    /// `Limit::AllocationNotMapped`, its allocation freed and counted, the gauges as they were;
    /// once it maps again, the same take is a block. Teardown unmaps each allocation mapped once,
    /// and never the one refused.
    #[test]
    fn every_device_maps_each_allocation_before_its_blocks_go_out_and_refuses_one_not_mapped() {
        const MIB: u64 = 1 << 20;
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        let this_thread = thread::current().name().map(str::to_string);
        for (name, physical) in &devices {
            let hook = Arc::new(Hook::default());
            let device = device_with(&hook, physical, name);
            let core = device.core();
            let counters = device.counters();
            let made = hook.mapped();
            assert_eq!(
                (made.iter())
                    .map(|m| (m.class, m.len, m.contents.is_some(), m.thread.clone()))
                    .collect::<Vec<_>>(),
                [(MemoryClass::Private, false), (MemoryClass::Shared, true)]
                    .map(|(class, mapped)| (class, class.block_len(), mapped, this_thread.clone())),
                "{name}: each class's first allocation, mapped in the device's making"
            );
            let metrics = counters.metrics();
            assert_eq!(
                (
                    metrics.allocations_mapped,
                    metrics.allocations_mapped_on_takes,
                    metrics.allocation_map_refusals
                ),
                (2, 0, 0),
                "{name}"
            );

            // A take that no free range fits: its allocation mapped inside it, on its thread.
            let taker = device.clone();
            let long = (thread::Builder::new().name("m2v-test-taker".to_string()))
                .spawn(move || taker.take_block(MemoryClass::Shared, 24 * MIB))
                .expect("a thread")
                .join()
                .expect("the taker")
                .unwrap_or_else(|error| panic!("{name}: a block of 24 MiB: {error}"));
            hook.assert_mapped(name, &long);
            let mapped = hook.mapped();
            assert_eq!(
                (
                    mapped.len(),
                    mapped[2].allocation,
                    mapped[2].len,
                    mapped[2].thread.as_deref()
                ),
                (3, long.allocation, 24 * MIB, Some("m2v-test-taker")),
                "{name}: the take's allocation, mapped on the taker's thread"
            );

            // A take that leaves fewer than 8 MiB free: the block-maker thread makes and maps the
            // next allocation, ahead of need.
            let low = (device.take_block(MemoryClass::Shared, 12 * MIB))
                .unwrap_or_else(|error| panic!("{name}: a block of 12 MiB: {error}"));
            hook.assert_mapped(name, &low);
            wait_until("the allocation made ahead", || {
                hook.mapped().len() == 4 && core.pool.stocked(MemoryClass::Shared)
            });
            let ahead = hook.mapped()[3].clone();
            assert_eq!(
                (ahead.class, ahead.len, ahead.thread.as_deref()),
                (
                    MemoryClass::Shared,
                    MemoryClass::Shared.block_len(),
                    Some("m2v-blocks")
                ),
                "{name}: the allocation made ahead of need"
            );
            let metrics = counters.metrics();
            assert_eq!(
                (
                    metrics.allocations_mapped,
                    metrics.allocations_mapped_on_takes
                ),
                (3, 1),
                "{name}"
            );

            // While the hook maps nothing, a take that no free range fits is refused, and its
            // allocation freed.
            hook.refuse.store(true, Relaxed);
            let gauges = counters.gauges();
            let before = counters.metrics();
            match device.take_block(MemoryClass::Shared, 40 * MIB) {
                Err(Error::Limit(Limit::AllocationNotMapped {
                    class: MemoryClass::Shared,
                    len,
                })) => assert_eq!(len, 40 * MIB, "{name}"),
                other => panic!(
                    "{name}: a take not mapped: {:?}",
                    other.map(|block| block.id)
                ),
            }
            let after = counters.metrics();
            assert_eq!(
                (
                    after.allocation_map_refusals - before.allocation_map_refusals,
                    after.consumer_allocations - before.consumer_allocations,
                    after.allocations_mapped_on_takes - before.allocations_mapped_on_takes
                ),
                (1, 1, 0),
                "{name}: the take's allocation made, not mapped, and freed"
            );
            let refused = counters.gauges();
            assert_eq!(
                (refused.blocks, refused.shared_idle_bytes),
                (gauges.blocks, gauges.shared_idle_bytes),
                "{name}: the gauges as they were"
            );
            hook.refuse.store(false, Relaxed);
            let mapped = (device.take_block(MemoryClass::Shared, 40 * MIB))
                .unwrap_or_else(|error| panic!("{name}: a block of 40 MiB: {error}"));
            hook.assert_mapped(name, &mapped);
            drop((long, low, mapped));
            drop(device);
            assert_eq!(
                counters.gauges(),
                Gauges::default(),
                "{name}: gauges after teardown"
            );
            assert_each_unmapped_once(name, &hook);
            println!(
                "memory: {name}: each allocation mapped before its blocks went out, one not mapped \
                 refused; metrics {:?}",
                counters.metrics()
            );
        }
    }

    /// A long block given back is a free range of its long allocation at once, which the next take
    /// of its length lands in: a Shared block of 64 MiB, which takes an allocation of its own, is
    /// taken, given a zeroed buffer, and given back 20 times in a row, each take straight after
    /// the give-back before. Only the first take allocates; the other 19 are long reuses, the
    /// allocation's bytes counted as long reused while a block of a later take is in it, and
    /// nothing is freed.
    #[test]
    fn every_device_takes_a_long_block_given_back_again_at_once() {
        const LEN: u64 = 64 << 20;
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let hook = Arc::new(Hook::default());
            let device = device_with(&hook, physical, name);
            let counters = device.counters();
            let before = counters.metrics();
            let mut made = Vec::new();
            for round in 0..20 {
                let allocations = counters.metrics().consumer_allocations;
                let block = (device.take_block(MemoryClass::Shared, LEN))
                    .unwrap_or_else(|error| panic!("{name}: round {round}: {error}"));
                let reused = counters.gauges().shared_long_reused_bytes;
                assert_eq!(
                    reused,
                    if round == 0 { 0 } else { LEN },
                    "{name}: round {round}: the long allocation's bytes, reused"
                );
                let buffer = (device.new_buffer(block.id, 0, LEN, Some(0)))
                    .unwrap_or_else(|error| panic!("{name}: round {round}: {error}"));
                drop(buffer);
                device.release_block(block).expect("a release");
                made.push(counters.metrics().consumer_allocations - allocations);
            }
            let after = counters.metrics();
            assert_eq!(
                (made[0], made.iter().skip(1).sum::<u64>()),
                (1, 0),
                "{name}: allocations each take made: {made:?}"
            );
            assert_eq!(
                (
                    after.shared_long_reuses - before.shared_long_reuses,
                    after.allocations_freed - before.allocations_freed
                ),
                (19, 0),
                "{name}: takes of the long allocation after its first, and allocations freed"
            );
            assert_eq!(
                counters.gauges().shared_long_reused_bytes,
                0,
                "{name}: no block in it"
            );
            drop(device);
            assert_each_unmapped_once(name, &hook);
        }
    }

    /// Checks each free of `frees` (when its allocation became idle, when its free began and
    /// ended) began at its idle age or after, and after the free before it ended: one at a time.
    /// How long past its idle age a free begins is a time, which this suite does not judge: the
    /// harness's own rows `idle_frees` and `idle_frees_under_commits` judge it, each alone.
    fn assert_freed_at_the_idle_age(name: &str, frees: &[(u64, u64, u64)]) {
        let mut ended = 0;
        for (index, &(since, began, end)) in frees.iter().enumerate() {
            let due = since + IDLE_AGE_NANOS;
            assert!(
                began >= due,
                "{name}: free {index} began {} us idle",
                (began - since) / 1000
            );
            assert!(
                began >= ended && end >= began,
                "{name}: free {index} began {} us before the free before it ended",
                ended.saturating_sub(began) / 1000
            );
            ended = end;
        }
    }

    /// Each idle allocation is freed at its idle age, oldest first: four Shared blocks of 16 MiB,
    /// each with an allocation of its own, taken beside a small block and given back leave their
    /// allocations idle, the oldest idle age gauge no older than the first give-back. A take of
    /// their length right then lands in the oldest, the first fit, counted in the bucket of its
    /// idle time, and holds it for 300 ms: the other three are freed at their idle age, one at a
    /// time, after the hook's unmap of each, on the block-maker thread. Given back again, the
    /// first is freed at its idle age from then. The small block's allocation, once its block is
    /// given back, is the reserve that keeps 8 MiB free, and stays past the idle age, unweighed:
    /// the oldest idle age gauge reads 0. Nothing commits.
    #[test]
    fn every_device_frees_each_idle_allocation_at_its_idle_age() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let hook = Arc::new(Hook::default());
            let device = device_with(&hook, physical, name);
            let core = device.core();
            let counters = device.counters();
            let len = MemoryClass::Shared.block_len();
            let small = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|error| panic!("{name}: a small block: {error}"));
            let taken: Vec<Block> = (0..4)
                .map(|_| {
                    (device.take_block(MemoryClass::Shared, len))
                        .unwrap_or_else(|error| panic!("{name}: a block: {error}"))
                })
                .collect();
            let four: Vec<AllocationId> = taken.iter().map(|block| block.allocation).collect();
            let ids: Vec<BlockId> = taken.iter().map(|block| block.id).collect();
            let small_id = small.id;
            let before = counters.metrics();
            let releasing = crate::host_time();
            for block in taken {
                device.release_block(block).expect("a release");
            }
            wait_until("the four given back", || {
                ids.iter().all(|&id| core.pool.holds(id))
            });
            let gauges = counters.gauges();
            let read = crate::host_time();
            assert_eq!(gauges.shared_idle_bytes, 4 * len, "{name}: four idle");
            assert!(
                gauges.shared_idle_age_nanos > 0
                    && gauges.shared_idle_age_nanos <= read - releasing,
                "{name}: the oldest idle age, {} us, {} us after the first give-back",
                gauges.shared_idle_age_nanos / 1000,
                (read - releasing) / 1000
            );

            // The oldest taken again: its idle time ends, counted by how long it was idle. The
            // take must come within the idle age of the give-back, which the test's own clock
            // reads state: past it, the block-maker may have freed the allocation.
            let again = (device.take_block(MemoryClass::Shared, len))
                .unwrap_or_else(|error| panic!("{name}: a block: {error}"));
            let taken = crate::host_time();
            assert!(
                taken - releasing < IDLE_AGE_NANOS,
                "{name}: this test's thread took the block {} us after the give-back, past the \
                 idle age: the test, not the runtime, was late",
                (taken - releasing) / 1000
            );
            assert_eq!(
                (again.allocation, again.offset),
                (four[0], 0),
                "{name}: the first fit, the oldest"
            );
            let reuses = counters.metrics().idle_reuses_by_age;
            let new = |at: usize| reuses[at] - before.idle_reuses_by_age[at];
            let ages = 0..=reuse_bucket(taken - releasing);
            assert_eq!(
                (
                    ages.clone().map(new).sum::<u64>(),
                    (0..7).map(new).sum::<u64>()
                ),
                (1, 1),
                "{name}: taken again at most {} us idle, buckets {ages:?}: {reuses:?}",
                (taken - releasing) / 1000
            );

            // The other three freed at their idle age, oldest first.
            wait_until("three idle Shared allocations freed", || {
                core.pool.frees().len() == 3 && !core.pool.freeing()
            });
            assert_freed_at_the_idle_age(name, &core.pool.frees());
            let unmapped = hook.unmapped();
            assert_eq!(unmapped.len(), 3, "{name}: {unmapped:?}");
            for (unmapped, allocation) in unmapped.iter().zip(&four[1..]) {
                assert_eq!(unmapped.allocation, *allocation, "{name}: the oldest first");
                assert_eq!(unmapped.thread.as_deref(), Some("m2v-blocks"), "{name}");
            }
            thread::sleep(Duration::from_millis(250));
            assert_eq!(
                core.pool.frees().len(),
                3,
                "{name}: the block taken again is not freed while held"
            );

            // Given back again: freed at its idle age from now.
            let released = crate::host_time();
            device.release_block(again).expect("a release");
            wait_until("the fourth freed", || {
                core.pool.frees().len() == 4 && !core.pool.freeing()
            });
            let frees = core.pool.frees();
            assert!(
                frees[3].0 >= released,
                "{name}: the fourth's idle time starts at its second give-back"
            );
            assert_freed_at_the_idle_age(name, &frees[3..]);
            assert_eq!(hook.unmapped()[3].allocation, four[0], "{name}: the first");

            // The small block's allocation is the reserve: kept past the idle age, unweighed.
            device.release_block(small).expect("a release");
            wait_until("the small block given back", || core.pool.holds(small_id));
            thread::sleep(Duration::from_nanos(2 * IDLE_AGE_NANOS));
            let gauges = counters.gauges();
            assert_eq!(
                (
                    gauges.blocks,
                    gauges.shared_idle_bytes,
                    gauges.shared_idle_age_nanos
                ),
                (ZERO_SOURCE + 2, len, 0),
                "{name}: the reserve of each class, kept and unweighed"
            );
            let metrics = counters.metrics();
            let freed = |at: usize| metrics.idle_frees_by_age[at] - before.idle_frees_by_age[at];
            assert_eq!(
                (
                    metrics.allocations_freed - before.allocations_freed,
                    freed(0)
                ),
                (4, 0),
                "{name}: four frees, none before the idle age: {:?}",
                metrics.idle_frees_by_age
            );
            drop(device);
            assert_eq!(
                counters.gauges(),
                Gauges::default(),
                "{name}: gauges after teardown"
            );
            assert_each_unmapped_once(name, &hook);
            println!(
                "memory: {name}: idle allocations freed at their idle age, frees {:?}; metrics {:?}",
                free_times(&frees),
                counters.metrics()
            );
        }
    }

    /// An allocation a take longer than the unit made for itself is never the reserve: a Shared
    /// block of 64 MiB, given back, leaves its allocation idle with free bytes, newer than the
    /// class's first one, and nothing else of the class live. The long allocation is freed at its
    /// idle age, and the class keeps its unit allocation as its reserve, unweighed. Nothing
    /// commits.
    #[test]
    fn every_device_frees_a_long_allocation_whose_block_s_bytes_are_free_at_its_idle_age() {
        const MIB: u64 = 1 << 20;
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let hook = Arc::new(Hook::default());
            let device = device_with(&hook, physical, name);
            let core = device.core();
            let counters = device.counters();
            let unit = MemoryClass::Shared.block_len();
            let long = (device.take_block(MemoryClass::Shared, 64 * MIB))
                .unwrap_or_else(|error| panic!("{name}: a block of 64 MiB: {error}"));
            let allocation = long.allocation;
            device.release_block(long).expect("a release");
            wait_until("the long allocation freed", || {
                core.pool.frees().len() == 1 && !core.pool.freeing()
            });
            let unmapped = hook.unmapped();
            assert_eq!(
                unmapped.iter().map(|u| u.allocation).collect::<Vec<_>>(),
                [allocation],
                "{name}: the long allocation unmapped, and no other"
            );
            thread::sleep(Duration::from_nanos(2 * IDLE_AGE_NANOS));
            let gauges = counters.gauges();
            assert_eq!(
                (
                    gauges.blocks,
                    gauges.shared_idle_bytes,
                    gauges.shared_reserve_bytes,
                    gauges.shared_idle_age_nanos
                ),
                (ZERO_SOURCE + 2, unit, unit, 0),
                "{name}: one unit allocation of each class, the Shared one the reserve"
            );
            assert_eq!(
                core.pool.frees().len(),
                1,
                "{name}: no second free: the reserve stays"
            );
            drop(device);
            assert_eq!(
                counters.gauges(),
                Gauges::default(),
                "{name}: gauges after teardown"
            );
            assert_each_unmapped_once(name, &hook);
            println!(
                "memory: {name}: a long allocation with free bytes freed at its idle age; \
                 metrics {:?}",
                counters.metrics()
            );
        }
    }

    /// The bucket of [`Metrics::idle_reuses_by_age`](crate::Metrics::idle_reuses_by_age) an idle
    /// time of `nanos` counts in: the runtime's upper bounds, 50, 100, 250, 500, 1000 and
    /// 2000 ms.
    fn reuse_bucket(nanos: u64) -> usize {
        ([50, 100, 250, 500, 1000, 2000].iter())
            .position(|&ms: &u64| nanos < ms * 1_000_000)
            .unwrap_or(6)
    }

    /// Each free's idle time when it began and its length, in microseconds.
    fn free_times(frees: &[(u64, u64, u64)]) -> Vec<(u64, u64)> {
        (frees.iter())
            .map(|&(since, began, end)| ((began - since) / 1000, (end - began) / 1000))
            .collect()
    }

    /// Steady commits hold no idle allocation, and a submission during a free is counted. A
    /// command buffer is committed every 2 ms; 0.5 seconds in, a Shared block of 16 MiB that fills
    /// an allocation is given back, which makes that allocation the class's reserve and the one
    /// made ahead before it one too many; a second is given back once the first free ended and the
    /// Shared idle bytes were seen back at the made-ahead amount, which makes the first's
    /// allocation one too many in turn; the commits stop once the second free ended. Each free
    /// begins after the give-back that set it off, at its idle age or after, one at a time, while
    /// the commits go on, and once they stop the Shared idle bytes are the made-ahead amount they
    /// were before. Then the free a third give-back sets off is held open: a command buffer
    /// committed with a fill during it is counted as a submission during a free, and one committed
    /// once the free ended is not.
    #[test]
    fn every_device_frees_idle_allocations_at_their_idle_age_under_steady_commits() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::new(physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let core = device.core();
            let counters = device.counters();
            let queue = device.new_command_queue(1).expect("a queue");
            let len = MemoryClass::Shared.block_len();
            let target = (device.take_block(MemoryClass::Private, 4096))
                .unwrap_or_else(|error| panic!("{name}: a block: {error}"));
            let buffer = (device.new_buffer(target.id, 0, 256, None)).expect("a buffer");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            command_buffer.commit().expect("a commit");
            command_buffer
                .wait_until_completed()
                .expect("a wait outside a scheduled handler");
            let mut blocks: Vec<Block> = (0..3)
                .map(|_| {
                    (device.take_block(MemoryClass::Shared, len))
                        .unwrap_or_else(|error| panic!("{name}: a block: {error}"))
                })
                .collect();
            wait_until("the Shared reserve", || {
                core.pool.stocked(MemoryClass::Shared)
            });
            let made_ahead = counters.gauges().shared_idle_bytes;
            assert_eq!(made_ahead, len, "{name}: one Shared allocation made ahead");
            let before = counters.metrics();

            // Commits every 2 ms: a block given back 0.5 seconds in, the second once the first's
            // free ended, until the second's free ended.
            let started = Instant::now();
            let mut given = Vec::new();
            let mut between = None;
            let mut commits = 0;
            let ended = |count: usize| core.pool.frees().len() == count && !core.pool.freeing();
            while !(given.len() == 2 && ended(2)) {
                assert!(
                    started.elapsed() < Duration::from_secs(120),
                    "{name}: the frees"
                );
                if given.is_empty() && started.elapsed() >= Duration::from_millis(500) {
                    given.push(crate::host_time());
                    device.release_block(blocks.remove(0)).expect("a release");
                } else if given.len() == 1 && ended(1) {
                    between = Some(counters.gauges().shared_idle_bytes);
                    given.push(crate::host_time());
                    device.release_block(blocks.remove(0)).expect("a release");
                }
                let command_buffer = queue.command_buffer().expect("a command buffer");
                command_buffer.commit().expect("a commit");
                commits += 1;
                thread::sleep(Duration::from_millis(2));
            }
            let stopped = crate::host_time();
            let frees = core.pool.frees();
            assert_eq!(frees.len(), 2, "{name}: both freed during the commits");
            assert_freed_at_the_idle_age(name, &frees);
            for ((_, began, end), given) in frees.iter().zip(&given) {
                assert!(
                    began >= given && *end < stopped,
                    "{name}: freed after the give-back that set it off, while the commits went on"
                );
            }
            assert_eq!(
                (between, counters.gauges().shared_idle_bytes),
                (Some(made_ahead), made_ahead),
                "{name}: the idle bytes back to the made-ahead amount"
            );
            let steady = counters.metrics();
            println!(
                "memory: {name}: {commits} commits, {} of them during the frees; frees {:?}",
                steady.submissions_during_frees - before.submissions_during_frees,
                free_times(&frees)
            );

            // A free held open: a submission during it is counted, one after it is not.
            let fill = |buffer: &crate::Buffer| {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let (to, at) = vk_buffer(&device, target.id);
                command_buffer
                    .record_for_test(&[buffer], |device, command| {
                        // SAFETY: the command buffer is recording; the block is held.
                        unsafe { device.cmd_fill_buffer(command, to, at, 256, 7) };
                    })
                    .expect("a fill");
                command_buffer.commit().expect("a commit");
                command_buffer
                    .wait_until_completed()
                    .expect("a wait outside a scheduled handler");
            };
            core.pool.hold_frees(true);
            device.release_block(blocks.remove(0)).expect("a release");
            wait_until("a free held open", || core.pool.freeing());
            fill(&buffer);
            assert_eq!(
                counters.metrics().submissions_during_frees,
                steady.submissions_during_frees + 1,
                "{name}: the submission during the free"
            );
            core.pool.hold_frees(false);
            wait_until("the third freed", || {
                core.pool.frees().len() == 3 && !core.pool.freeing()
            });
            fill(&buffer);
            assert_eq!(
                counters.metrics().submissions_during_frees,
                steady.submissions_during_frees + 1,
                "{name}: the submission after the free"
            );
            drop((buffer, target, queue));
            drop(device);
            wait_until("the teardown", || counters.gauges() == Gauges::default());
            println!(
                "memory: {name}: frees at the idle age under steady commits; metrics {:?}",
                counters.metrics()
            );
        }
    }

    /// Seeded takes and releases of mixed lengths across both classes, with the block-maker
    /// thread running: every take and release returns within a minute, every block's allocation
    /// was mapped before the take returned it, every allocation mapped is unmapped once by the
    /// end, on the block-maker thread or the dropping one, and every gauge reads zero after
    /// teardown.
    #[test]
    fn every_device_returns_every_seeded_take_and_release_of_mixed_lengths() {
        use std::sync::mpsc::RecvTimeoutError;
        const OPS: usize = 300;
        /// The most live bytes of a class the run keeps: past it, it releases.
        const MOST: u64 = 192 << 20;
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        let this_thread = thread::current().name().map(str::to_string);
        for (name, physical) in &devices {
            let hook = Arc::new(Hook::default());
            let device = device_with(&hook, physical, name);
            let counters = device.counters();
            let worker = device.clone();
            let mappings = Arc::clone(&hook);
            let (sender, receiver) = std::sync::mpsc::channel();
            let run = thread::spawn(move || -> Result<Vec<BlockId>, String> {
                const MIB: u64 = 1 << 20;
                let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
                let mut next = move || {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    seed
                };
                let lens = [64 << 10, MIB, 3 * MIB, 16 * MIB, 20 * MIB, 40 * MIB];
                let index = |class: MemoryClass| usize::from(class == MemoryClass::Shared);
                let mut live: Vec<Block> = Vec::new();
                let mut bytes = [0u64; 2];
                let mut handed = Vec::new();
                for op in 0..OPS {
                    let full = bytes.iter().any(|&bytes| bytes > MOST);
                    if !live.is_empty() && (full || next() % 3 == 0) {
                        let block = live.swap_remove((next() % live.len() as u64) as usize);
                        bytes[index(block.class)] -= block.len;
                        (worker.release_block(block)).map_err(|error| format!("{op}: {error}"))?;
                    } else {
                        let class = match next() % 2 {
                            0 => MemoryClass::Shared,
                            _ => MemoryClass::Private,
                        };
                        let len = lens[(next() % lens.len() as u64) as usize];
                        let block = (worker.take_block(class, len))
                            .map_err(|error| format!("{op}: {class:?} {len}: {error}"))?;
                        if !(mappings.mapped().iter()).any(|m| m.allocation == block.allocation) {
                            return Err(format!(
                                "{op}: {class:?} {len}: its allocation not mapped"
                            ));
                        }
                        bytes[index(class)] += block.len;
                        handed.push(block.id);
                        live.push(block);
                    }
                    if sender.send(op).is_err() {
                        break;
                    }
                }
                drop(live);
                drop(worker);
                Ok(handed)
            });
            for op in 0..OPS {
                match receiver.recv_timeout(Duration::from_secs(60)) {
                    Ok(_) => {}
                    Err(RecvTimeoutError::Timeout) => {
                        panic!("{name}: operation {op} did not return within a minute")
                    }
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }
            let mut handed = (run.join())
                .expect("the run's thread")
                .unwrap_or_else(|error| panic!("{name}: operation {error}"));
            drop(device);
            assert_eq!(
                counters.gauges(),
                Gauges::default(),
                "{name}: gauges after teardown"
            );
            handed.sort_unstable();
            handed.dedup();
            assert_each_unmapped_once(name, &hook);
            for unmapped in hook.unmapped() {
                assert!(
                    unmapped.thread.as_deref() == Some("m2v-blocks")
                        || unmapped.thread == this_thread,
                    "{name}: unmapped on {:?}",
                    unmapped.thread
                );
            }
            println!(
                "memory: {name}: {OPS} seeded takes and releases, {} blocks; metrics {:?}",
                handed.len(),
                counters.metrics()
            );
        }
    }

    /// With the device's `maxMemoryAllocationCount` answered as the zero source's, the first two
    /// allocations' and one more (`Device::open`), the count is reached once a texture's block
    /// takes the first Private allocation whole and the block-maker thread makes the next. Past
    /// it, a texture longer than an allocation, whose block takes an allocation of its own, and
    /// the staging region of a first copy are `Limit::MemoryAllocations` with the device's count,
    /// refused before the driver is called, the gauges unchanged; textures placed in the texture
    /// block still come and go; and every gauge reads zero after teardown.
    #[test]
    fn every_device_refuses_an_allocation_past_its_count() {
        use crate::{IndexClass, PixelFormat, Region, TextureDescriptor, TextureType};
        const MAX: u32 = ZERO_SOURCE as u32 + 3;
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::open(physical, DeviceOptions::default(), |limits| {
                limits.max_memory_allocation_count = MAX;
            })
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let counters = device.counters();
            wait_until("the first allocation of each class", || {
                counters.gauges().blocks == ZERO_SOURCE + 2
            });
            device
                .grant_indices(IndexClass::Texture, 4096, 16)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 4, 4);
            let first = (device.new_texture(&d, Some(4096)))
                .unwrap_or_else(|error| panic!("{name}: the first texture: {error}"));
            wait_until("the next Private allocation", || {
                counters.gauges().blocks == u64::from(MAX)
            });
            // Three layers of 4096 x 4096 RGBA16Float, 384 MiB: longer than an allocation.
            let mut large = TextureDescriptor::new(
                TextureType::Type2DArray,
                PixelFormat::RGBA16Float,
                4096,
                4096,
            );
            large.array_length = 3;
            match device.new_texture(&large, Some(4097)) {
                Err(Error::Limit(Limit::MemoryAllocations { max: MAX })) => {}
                Err(error) => panic!("{name}: a long texture past the count: {error}"),
                Ok(_) => panic!("{name}: a long texture past the count was made"),
            }
            let gauges = counters.gauges();
            assert_eq!(
                (gauges.blocks, gauges.textures, gauges.descriptor_slots),
                (u64::from(MAX), 1, 1),
                "{name}: after the refused texture"
            );
            let whole = Region {
                origin: [0; 3],
                size: [4, 4, 1],
            };
            match first.replace_region(whole, 0, 0, crate::HostBytes::from(&[0x5A; 64][..]), 16, 64)
            {
                Err(Error::Limit(Limit::MemoryAllocations { max: MAX })) => {}
                other => panic!("{name}: the staging region past the count: {other:?}"),
            }
            assert_eq!(
                counters.gauges().blocks,
                u64::from(MAX),
                "{name}: after the refused staging region"
            );
            // Placed in the texture block: no allocation.
            let placed: Vec<_> = (4097..4105)
                .map(|index| {
                    (device.new_texture(&d, Some(index)))
                        .unwrap_or_else(|error| panic!("{name}: texture {index}: {error}"))
                })
                .collect();
            assert_eq!(counters.gauges().blocks, u64::from(MAX), "{name}");
            drop(placed);
            drop(first);
            drop(device);
            wait_until("the teardown", || counters.gauges() == Gauges::default());
            print_gauges_at_drop("memory_allocation_count", counters.gauges());
            println!("memory: {name}: refused past {MAX} allocations");
        }
    }

    /// A zeroed Private buffer's fill goes ahead of the next command buffer committed, before a
    /// copy that command buffer makes into the buffer: each copied range holds the copy, the rest
    /// of the buffer from its `zero_from` reads zero, a copy into the bytes before `zero_from`
    /// holds as well, and the block's bytes outside the buffer are as they were. A `zero_from` at
    /// the length zeroes nothing. A texture placed in a block zeroes nothing.
    #[test]
    fn every_device_fills_a_zeroed_private_buffer_before_a_copy_into_it_in_the_next_command_buffer()
    {
        use crate::IndexClass;
        const LEN: u64 = 4096;
        const AT: u64 = 1024;
        const SPAN: u64 = 2 * AT + LEN;
        // The copies' source, at 64 KiB in the Shared block: a pattern with no zero byte.
        const SOURCE: u64 = 64 << 10;
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::new(physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let counters = device.counters();
            let gpu = Gpu::new(&device);
            let queue = device.new_command_queue(1).expect("a queue");
            let private = (device
                .take_block(MemoryClass::Private, MemoryClass::Private.block_len()))
            .unwrap_or_else(|error| panic!("{name}: a Private block: {error}"));
            let shared = (device.take_block(MemoryClass::Shared, MemoryClass::Shared.block_len()))
                .unwrap_or_else(|error| panic!("{name}: a Shared block: {error}"));
            let host = bytes(&shared);
            let source = SOURCE as usize..(SOURCE + LEN) as usize;
            for (index, byte) in host[source].iter_mut().enumerate() {
                *byte = (index % 251) as u8 + 1;
            }
            let (to_private, to_shared) = (
                vk_buffer(&device, private.id),
                vk_buffer(&device, shared.id),
            );
            let ((private_buffer, private_base), (shared_buffer, shared_base)) =
                (to_private, to_shared);
            // Each case: the buffer's `zero_from`, and the ranges of it copied into.
            for (zero_from, copies) in [
                (0, vec![(100, 300)]),
                (1000, vec![(0, 1000)]),
                (1000, vec![(0, 1000), (2000, 2100)]),
                (LEN, vec![(0, LEN)]),
            ] {
                gpu.fill(to_private, 0, SPAN, u32::MAX);
                let before = counters.metrics().bytes_zeroed;
                let buffer = (device.new_buffer(private.id, AT, LEN, Some(zero_from)))
                    .unwrap_or_else(|error| panic!("{name}: a zeroed Private buffer: {error}"));
                assert_eq!(
                    counters.metrics().bytes_zeroed - before,
                    LEN - zero_from,
                    "{name}: from {zero_from}, the bytes zeroed"
                );
                let regions: Vec<vk::BufferCopy> = (copies.iter())
                    .map(|&(start, end)| vk::BufferCopy {
                        src_offset: shared_base + SOURCE + start,
                        dst_offset: private_base + AT + start,
                        size: end - start,
                    })
                    .collect();
                let back = [vk::BufferCopy {
                    src_offset: private_base,
                    dst_offset: shared_base,
                    size: SPAN,
                }];
                let command_buffer = queue.command_buffer().expect("a command buffer");
                command_buffer
                    .record_for_test(&[&buffer], |device, command| {
                        let barrier = [vk::MemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)];
                        let dependency = vk::DependencyInfo::default().memory_barriers(&barrier);
                        // SAFETY: the command buffer is recording; both blocks are held and span
                        // the regions.
                        unsafe {
                            device.cmd_copy_buffer(
                                command,
                                shared_buffer,
                                private_buffer,
                                &regions,
                            );
                            device.cmd_pipeline_barrier2(command, &dependency);
                            device.cmd_copy_buffer(command, private_buffer, shared_buffer, &back);
                        }
                    })
                    .expect("the copies");
                command_buffer.commit().expect("a commit");
                command_buffer
                    .wait_until_completed()
                    .expect("a wait outside a scheduled handler");
                assert_eq!(
                    command_buffer.status(),
                    CommandBufferStatus::Completed,
                    "{name}"
                );
                for (index, &byte) in host[..SPAN as usize].iter().enumerate() {
                    let expected = match (index as u64).checked_sub(AT).filter(|&at| at < LEN) {
                        None => 0xFF,
                        Some(at)
                            if copies
                                .iter()
                                .any(|&(start, end)| (start..end).contains(&at)) =>
                        {
                            (at % 251) as u8 + 1
                        }
                        Some(at) if at >= zero_from => 0,
                        Some(_) => 0xFF,
                    };
                    assert_eq!(
                        byte, expected,
                        "{name}: from {zero_from}, copies {copies:?}, byte {index}"
                    );
                }
                drop(buffer);
            }

            // A texture placed in a block zeroes nothing.
            device
                .grant_indices(IndexClass::Texture, 4096, 1)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let before = counters.metrics().bytes_zeroed;
            let d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 64, 64);
            let texture = (device.new_texture(&d, Some(4096)))
                .unwrap_or_else(|error| panic!("{name}: a texture: {error}"));
            assert_eq!(
                counters.metrics().bytes_zeroed,
                before,
                "{name}: a texture placed in a block"
            );
            drop((texture, gpu, queue, private, shared));
            drop(device);
            wait_until("the teardown", || counters.gauges() == Gauges::default());
            println!(
                "memory: {name}: zeroed Private buffers filled before copies into them; metrics {:?}",
                counters.metrics()
            );
        }
    }

    /// The add kernel over `a`, `b` and `out`, committed on `queue` and waited for.
    fn add_into(
        name: &str,
        queue: &crate::CommandQueue,
        state: &ComputePipelineState,
        [a, b, out]: [&Buffer; 3],
    ) {
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        encoder.set_compute_pipeline_state(state).expect("a state");
        encoder.set_buffer(Some(a), 0, 0).expect("a buffer");
        encoder.set_buffer(Some(b), 0, 1).expect("a buffer");
        encoder.set_buffer(Some(out), 0, 2).expect("a buffer");
        (encoder.dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1])).expect("a dispatch");
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// A Shared buffer's words.
    fn words(buffer: &Buffer) -> Vec<u32> {
        let contents = buffer.contents().expect("a Shared buffer").cast::<u32>();
        // SAFETY: the buffer is alive and holds `WORDS` words, and its work completed.
        (0..WORDS)
            .map(|index| unsafe { contents.as_ptr().add(index).read_volatile() })
            .collect()
    }

    /// A zero fill the consumer asks of a Private block ([`Device::zero_block_range`]): a block a
    /// kernel wrote, given back, and its bytes taken again, holds the kernel's words; a fill of its
    /// first half reads zero to a kernel of the next command buffer, the rest as the first kernel
    /// wrote it, counted once with its bytes. A range past the block, a Shared block and a block
    /// the consumer does not hold are refused, and a fill of no bytes does nothing.
    #[test]
    fn every_device_zeroes_a_range_of_a_private_block_ahead_of_the_next_commit() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let hook = Arc::new(Hook::default());
            let device = device_with(&hook, physical, name);
            let core = device.core();
            let counters = device.counters();
            let len = (WORDS * 4) as u64;
            let shared = (device.take_block(MemoryClass::Shared, 3 * len))
                .unwrap_or_else(|error| panic!("{name}: a Shared block: {error}"));
            let buffer = |block: BlockId, at: u64| {
                (device.new_buffer(block, at, len, Some(0)))
                    .unwrap_or_else(|error| panic!("{name}: a buffer: {error}"))
            };
            let (ones, zero, out) = (
                buffer(shared.id, 0),
                buffer(shared.id, len),
                buffer(shared.id, 2 * len),
            );
            let contents = ones.contents().expect("a Shared buffer").cast::<u32>();
            for index in 0..WORDS {
                // SAFETY: the buffer is alive and holds `WORDS` words, and no GPU work uses it.
                unsafe { (contents.as_ptr().add(index)).write_volatile(index as u32 + 1) };
            }
            let state = add(&device);
            let queue = device.new_command_queue(2).expect("a queue");

            let private = (device.take_block(MemoryClass::Private, len))
                .unwrap_or_else(|error| panic!("{name}: a Private block: {error}"));
            let first = *private;
            let target = (device.new_buffer(first.id, 0, len, None))
                .unwrap_or_else(|error| panic!("{name}: a Private buffer: {error}"));
            add_into(name, &queue, &state, [&ones, &zero, &target]);
            drop(target);
            device.release_block(private).expect("a release");
            wait_until("the Private block given back", || core.pool.holds(first.id));
            let again = (device.take_block(MemoryClass::Private, len))
                .unwrap_or_else(|error| panic!("{name}: the Private bytes again: {error}"));
            assert_eq!(
                again.gpu_address, first.gpu_address,
                "{name}: the given-back bytes, the first fit"
            );
            let id = again.id;

            let before = counters.metrics();
            device
                .zero_block_range(id, 0, len / 2)
                .unwrap_or_else(|error| panic!("{name}: a zero fill: {error}"));
            device
                .zero_block_range(id, len, 0)
                .unwrap_or_else(|error| panic!("{name}: a zero fill of no bytes: {error}"));
            let after = counters.metrics();
            assert_eq!(
                (
                    after.block_zero_fills - before.block_zero_fills,
                    after.block_zero_fill_bytes - before.block_zero_fill_bytes,
                ),
                (1, len / 2),
                "{name}: the fills counted"
            );
            let source = (device.new_buffer(id, 0, len, None))
                .unwrap_or_else(|error| panic!("{name}: a Private buffer: {error}"));
            add_into(name, &queue, &state, [&source, &zero, &out]);
            let expected: Vec<u32> = (0..WORDS as u32)
                .map(|index| {
                    if index < WORDS as u32 / 2 {
                        0
                    } else {
                        index + 1
                    }
                })
                .collect();
            assert_eq!(
                words(&out),
                expected,
                "{name}: zero in the range, as written past it"
            );

            let past = device.zero_block_range(id, again.len - 4, 8);
            assert!(
                matches!(
                    past,
                    Err(Error::Refused(Refusal::ZeroRangePastBlock { offset, length: 8, block_len }))
                        if offset == again.len - 4 && block_len == again.len
                ),
                "{name}: {past:?}"
            );
            let overflow = device.zero_block_range(id, u64::MAX, 2);
            assert!(
                matches!(
                    overflow,
                    Err(Error::Refused(Refusal::ZeroRangePastBlock { .. }))
                ),
                "{name}: {overflow:?}"
            );
            let host = device.zero_block_range(shared.id, 0, 4);
            assert!(
                matches!(
                    host,
                    Err(Error::Refused(Refusal::ZeroRangeShared { block })) if block == shared.id
                ),
                "{name}: {host:?}"
            );
            let gone = shared.id;
            drop((ones, zero, out, source));
            device.release_block(shared).expect("a release");
            let missing = device.zero_block_range(gone, 0, 4);
            assert!(
                matches!(missing, Err(Error::NoSuchBlock(block)) if block == gone),
                "{name}: {missing:?}"
            );
            println!(
                "memory: {name}: a zero fill of a Private block ahead of the next commit; metrics {:?}",
                counters.metrics()
            );
            drop((again, queue, state));
            drop(device);
            wait_until("the teardown", || counters.gauges() == Gauges::default());
        }
    }

    /// The driver's zeros at a take (`BlockInfo::zeroed`), on both forms. Where the device
    /// enabled `zeroInitializeDeviceMemory` (its floor check's `zero_initialize_memory`), a take
    /// that no block of its allocation reached before answers true and is counted: a long Shared
    /// block reads zero through its mapping, whole, and a long Private one to a kernel, with no
    /// zeroing asked of the runtime, and a block carved past another in a unit allocation answers
    /// true as well. A range given back after a write and taken again answers false, its bytes as
    /// written, and the consumer's zeroing of it reads zero; so does a unit allocation's range
    /// taken again below a later block. Elsewhere every take answers false and none is counted.
    /// The consumer's one path, zero unless `zeroed`, reads zero on both forms.
    #[test]
    fn every_device_answers_at_a_take_whether_its_bytes_are_the_driver_s_zeros() {
        const LONG: u64 = 32 << 20;
        const SMALL: u64 = 64 << 10;
        let len = (WORDS * 4) as u64;
        let deep = LONG - len;
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let capabilities = (physical.check()).unwrap_or_else(|below| panic!("{name}: {below}"));
            let form = capabilities.zero_initialize_memory;
            let device = Device::new(physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let core = device.core();
            let counters = device.counters();
            assert_eq!(core.zero_initialize, form, "{name}: the device's form");
            let listed = (device.extensions().iter())
                .any(|extension| extension.as_c_str() == c"VK_EXT_zero_initialize_device_memory");
            assert!(listed || !form, "{name}: the form without its extension");
            let start = counters.metrics();
            let zero_unless = |block: &Block| (!block.zeroed).then_some(0);

            // The kernel's other input, words 1 to WORDS, and its output.
            let words_block = (device.take_block(MemoryClass::Shared, 2 * len))
                .unwrap_or_else(|error| panic!("{name}: a Shared block: {error}"));
            assert_eq!(words_block.zeroed, form, "{name}: the first take");
            let ones = (device.new_buffer(words_block.id, 0, len, zero_unless(&words_block)))
                .unwrap_or_else(|error| panic!("{name}: a buffer: {error}"));
            let out = (device.new_buffer(words_block.id, len, len, None))
                .unwrap_or_else(|error| panic!("{name}: a buffer: {error}"));
            let contents = ones.contents().expect("a Shared buffer").cast::<u32>();
            for index in 0..WORDS {
                // SAFETY: the buffer is alive and holds `WORDS` words, and no GPU work uses it.
                unsafe { (contents.as_ptr().add(index)).write_volatile(index as u32 + 1) };
            }
            let state = add(&device);
            let queue = device.new_command_queue(2).expect("a queue");
            let ones_words: Vec<u32> = (1..=WORDS as u32).collect();

            // A long Shared take: an allocation of its own, made in the take.
            let before = counters.metrics();
            let long = (device.take_block(MemoryClass::Shared, LONG))
                .unwrap_or_else(|error| panic!("{name}: a long Shared block: {error}"));
            assert_eq!(long.zeroed, form, "{name}: a long Shared take");
            let host = bytes(&long);
            if long.zeroed {
                assert!(
                    host.iter().all(|&byte| byte == 0),
                    "{name}: a Shared take that answered zeroed reads zero through its mapping"
                );
            }
            let whole = (device.new_buffer(long.id, 0, LONG, zero_unless(&long)))
                .unwrap_or_else(|error| panic!("{name}: a buffer: {error}"));
            assert!(
                host.iter().all(|&byte| byte == 0),
                "{name}: the consumer's path reads zero"
            );
            assert_eq!(
                counters.metrics().bytes_zeroed - before.bytes_zeroed,
                if form { 0 } else { LONG },
                "{name}: the runtime's zeroing of the long Shared buffer"
            );

            // A long Private take, read by a kernel deep in it.
            let private = (device.take_block(MemoryClass::Private, LONG))
                .unwrap_or_else(|error| panic!("{name}: a long Private block: {error}"));
            assert_eq!(private.zeroed, form, "{name}: a long Private take");
            let before = counters.metrics();
            let read = (device.new_buffer(private.id, deep, len, zero_unless(&private)))
                .unwrap_or_else(|error| panic!("{name}: a Private buffer: {error}"));
            assert_eq!(
                counters.metrics().bytes_zeroed - before.bytes_zeroed,
                if form { 0 } else { len },
                "{name}: the runtime's zeroing of the Private buffer"
            );
            add_into(name, &queue, &state, [&read, &ones, &out]);
            assert_eq!(
                words(&out),
                ones_words,
                "{name}: a kernel reads the Private take's zeros"
            );

            // A carve past another in a unit allocation.
            let first = (device.take_block(MemoryClass::Shared, SMALL))
                .unwrap_or_else(|error| panic!("{name}: a Shared block: {error}"));
            let second = (device.take_block(MemoryClass::Shared, SMALL))
                .unwrap_or_else(|error| panic!("{name}: a Shared block: {error}"));
            assert_eq!(second.zeroed, form, "{name}: a carve past another");
            let taken = counters.metrics();
            let zeroed_bytes = words_block.len + 2 * LONG + first.len + second.len;
            assert_eq!(
                (
                    taken.zeroed_takes - start.zeroed_takes,
                    taken.zeroed_take_bytes - start.zeroed_take_bytes
                ),
                if form { (5, zeroed_bytes) } else { (0, 0) },
                "{name}: the takes counted"
            );

            // Written, given back and taken again: false, as written, and zeroed when asked.
            host.fill(0xA5);
            drop(whole);
            let (address, id) = (long.gpu_address, long.id);
            device.release_block(long).expect("a release");
            wait_until("the long Shared block given back", || core.pool.holds(id));
            let again = (device.take_block(MemoryClass::Shared, LONG))
                .unwrap_or_else(|error| panic!("{name}: the long Shared block again: {error}"));
            assert_eq!(
                again.gpu_address, address,
                "{name}: the long allocation again"
            );
            assert!(!again.zeroed, "{name}: a long Shared range taken again");
            let host = bytes(&again);
            assert!(
                host.iter().all(|&byte| byte == 0xA5),
                "{name}: a range taken again, as written"
            );
            let before = counters.metrics().bytes_zeroed;
            let whole = (device.new_buffer(again.id, 0, LONG, zero_unless(&again)))
                .unwrap_or_else(|error| panic!("{name}: a buffer: {error}"));
            assert!(
                host.iter().all(|&byte| byte == 0),
                "{name}: the consumer's zeroing of a range taken again"
            );
            assert_eq!(counters.metrics().bytes_zeroed - before, LONG, "{name}");

            add_into(name, &queue, &state, [&ones, &ones, &read]);
            drop(read);
            let (address, id) = (private.gpu_address, private.id);
            device.release_block(private).expect("a release");
            wait_until("the long Private block given back", || core.pool.holds(id));
            let private = (device.take_block(MemoryClass::Private, LONG))
                .unwrap_or_else(|error| panic!("{name}: the long Private block again: {error}"));
            assert_eq!(
                private.gpu_address, address,
                "{name}: the long allocation again"
            );
            assert!(!private.zeroed, "{name}: a long Private range taken again");
            let written = (device.new_buffer(private.id, deep, len, None))
                .unwrap_or_else(|error| panic!("{name}: a Private buffer: {error}"));
            add_into(name, &queue, &state, [&written, &ones, &out]);
            let thrice: Vec<u32> = ones_words.iter().map(|word| 3 * word).collect();
            assert_eq!(
                words(&out),
                thrice,
                "{name}: a Private range taken again, as written"
            );
            let read = (device.new_buffer(private.id, deep, len, zero_unless(&private)))
                .unwrap_or_else(|error| panic!("{name}: a Private buffer: {error}"));
            add_into(name, &queue, &state, [&read, &ones, &out]);
            assert_eq!(
                words(&out),
                ones_words,
                "{name}: the consumer's zeroing of a Private range taken again"
            );

            // A unit allocation's range given back and taken again below a later block.
            let (address, id) = (first.gpu_address, first.id);
            device.release_block(first).expect("a release");
            wait_until("the Shared block given back", || core.pool.holds(id));
            let third = (device.take_block(MemoryClass::Shared, SMALL))
                .unwrap_or_else(|error| panic!("{name}: a Shared block: {error}"));
            assert_eq!(third.gpu_address, address, "{name}: the first fit");
            assert!(
                !third.zeroed,
                "{name}: a unit allocation's range taken again"
            );
            let end = counters.metrics();
            assert_eq!(
                end.zeroed_takes - taken.zeroed_takes,
                0,
                "{name}: no range taken again is counted"
            );
            println!(
                "memory: {name}: zeroed takes, form {form} (extension listed {listed}): zeroed_takes {} zeroed_take_bytes {} bytes_zeroed {} block_zero_fill_bytes {}",
                end.zeroed_takes - start.zeroed_takes,
                end.zeroed_take_bytes - start.zeroed_take_bytes,
                end.bytes_zeroed - start.bytes_zeroed,
                end.block_zero_fill_bytes - start.block_zero_fill_bytes,
            );
            drop((whole, read, written, ones, out, queue, state));
            drop((again, private, second, third, words_block));
            drop(device);
            wait_until("the teardown", || counters.gauges() == Gauges::default());
        }
    }

    /// The runtime zeroes what the consumer asks over bytes it cannot see written: the consumer's
    /// writes through a new Shared block's mapping, outside every buffer, then a buffer zeroed
    /// over some of them, read zero there and as written elsewhere; a new Private block's bytes
    /// written by a copy outside every buffer, then zeroed by `zero_block_range`, read zero to a
    /// kernel. Each zeroing is counted whole, on every device.
    #[test]
    fn every_device_zeroes_what_it_is_asked_over_bytes_written_outside_its_buffers() {
        const BLOCK: u64 = 4 << 20;
        let len = (WORDS * 4) as u64;
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::new(physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let counters = device.counters();
            let gpu = Gpu::new(&device);
            let state = add(&device);
            let queue = device.new_command_queue(1).expect("a queue");
            let words_block = (device.take_block(MemoryClass::Shared, 2 * len))
                .unwrap_or_else(|error| panic!("{name}: a Shared block: {error}"));
            let ones = (device.new_buffer(words_block.id, 0, len, Some(0)))
                .unwrap_or_else(|error| panic!("{name}: a buffer: {error}"));
            let out = (device.new_buffer(words_block.id, len, len, None))
                .unwrap_or_else(|error| panic!("{name}: a buffer: {error}"));
            let contents = ones.contents().expect("a Shared buffer").cast::<u32>();
            for index in 0..WORDS {
                // SAFETY: the buffer is alive and holds `WORDS` words, and no GPU work uses it.
                unsafe { (contents.as_ptr().add(index)).write_volatile(index as u32 + 1) };
            }

            // Shared: written through the mapping, no buffer there yet.
            let shared = (device.take_block(MemoryClass::Shared, BLOCK))
                .unwrap_or_else(|error| panic!("{name}: a Shared block: {error}"));
            let host = bytes(&shared);
            host.fill(0x5A);
            let before = counters.metrics().bytes_zeroed;
            let zeroed = (device.new_buffer(shared.id, 4096, 8192, Some(0)))
                .unwrap_or_else(|error| panic!("{name}: a zeroed buffer: {error}"));
            for (index, &byte) in host.iter().enumerate() {
                let inside = (4096..4096 + 8192).contains(&index);
                assert_eq!(
                    byte,
                    if inside { 0 } else { 0x5A },
                    "{name}: Shared byte {index} after the consumer's write and a zeroed buffer"
                );
            }
            assert_eq!(
                counters.metrics().bytes_zeroed - before,
                8192,
                "{name}: the Shared bytes zeroed"
            );

            // Private: written by a copy, no buffer there.
            let private = (device.take_block(MemoryClass::Private, BLOCK))
                .unwrap_or_else(|error| panic!("{name}: a Private block: {error}"));
            gpu.fill(vk_buffer(&device, private.id), 0, private.len, 0xA5A5_A5A5);
            let before = counters.metrics().block_zero_fill_bytes;
            device
                .zero_block_range(private.id, 0, len)
                .unwrap_or_else(|error| panic!("{name}: a zero fill: {error}"));
            assert_eq!(
                counters.metrics().block_zero_fill_bytes - before,
                len,
                "{name}: the Private bytes filled"
            );
            let source = (device.new_buffer(private.id, 0, len, None))
                .unwrap_or_else(|error| panic!("{name}: a Private buffer: {error}"));
            add_into(name, &queue, &state, [&source, &ones, &out]);
            let expected: Vec<u32> = (1..=WORDS as u32).collect();
            assert_eq!(
                words(&out),
                expected,
                "{name}: a kernel reads the zero fill over the copy's bytes"
            );
            println!(
                "memory: {name}: zeroes what it is asked over bytes written outside its buffers; metrics {:?}",
                counters.metrics()
            );
            drop((zeroed, source, ones, out, gpu, queue, state));
            drop((shared, private, words_block));
            drop(device);
            wait_until("the teardown", || counters.gauges() == Gauges::default());
        }
    }
}
