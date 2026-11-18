//! Memory blocks and buffers: the plan of a zero fill, a block's length and alignment, the idle
//! rule's window and surplus; and, under a device cfg, on every device of the cfg's kind, that the
//! device makes an allocation of each class as it is made and the next only while a class's free
//! bytes run low, that blocks are handed out and buffers view them, zeroed from their
//! `zero_from` (a Private buffer's fill ahead of a copy into it in the next command buffer), that
//! a block given back is kept for a take of its length, its bytes as written, that a take beside kept blocks of other lengths allocates and
//! their bytes are given back after the hook, and taken where its allocation fails, that idle
//! allocations follow the window's peak down, that seeded takes and releases of mixed lengths all
//! return, and that teardown frees everything with the gauges at zero.

use crate::fill::{plan, Plan};
use crate::memory::{base_align, round, surplus, Peaks};
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

/// The window holds the most live bytes of each second with a take or a give-back for 3 seconds:
/// the peak falls as the clock passes it, and the idle rule frees the oldest idle allocations while
/// the rest pass the peak less the live bytes.
#[test]
fn the_idle_rule_keeps_idle_allocations_under_the_window_s_peak() {
    const MIB: u64 = 1 << 20;
    let mut peaks = Peaks::default();
    // Four blocks of 16 MiB live in second 0, then given back.
    peaks.note(0, 0);
    peaks.note(0, 64 * MIB);
    peaks.note(0, 0);
    assert_eq!(peaks.most(), 64 * MIB);
    let idle = [16 * MIB; 4];
    assert_eq!(surplus(&idle, peaks.most(), 0), 0, "all within the peak");
    // A second on, the peak stands.
    peaks.note(1, 0);
    assert_eq!(surplus(&idle, peaks.most(), 0), 0, "within the window");
    // A block of 16 MiB live in second 2: the peak of second 0 still stands.
    peaks.note(2, 16 * MIB);
    assert_eq!(surplus(&idle, peaks.most(), 16 * MIB), 1);
    // Second 3 forgets second 0: the peak is second 2's.
    peaks.note(3, 16 * MIB);
    assert_eq!(peaks.most(), 16 * MIB);
    assert_eq!(surplus(&idle, peaks.most(), 16 * MIB), 4);
    assert_eq!(surplus(&idle, peaks.most(), 0), 3);
    // Second 6 forgets seconds 2 and 3: only what is live now.
    peaks.note(6, 0);
    assert_eq!(peaks.most(), 0);
    assert_eq!(surplus(&idle, peaks.most(), 0), 4);
    assert_eq!(surplus(&[], 64 * MIB, 0), 0);
    // Oldest first: allocations of other lengths are counted by their bytes.
    assert_eq!(surplus(&[48 * MIB, 16 * MIB], 20 * MIB, 0), 1);
    assert_eq!(surplus(&[16 * MIB, 48 * MIB], 20 * MIB, 0), 2);
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::device::lock;
    use crate::memory::{free_allocation, make_allocation};
    use crate::tests::command::on_device::each_device;
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{
        Block, BlockId, BlockInfo, BlockMappings, Buffer, CommandBufferStatus, Device,
        DeviceOptions, Error, Gauges, Instance, Limit, MemoryClass, PixelFormat, Refusal,
        TextureDescriptor, TextureType, MAX_BUFFER_LENGTH,
    };
    use ash::vk;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    /// The runtime's own allocation besides the pool's: the zero source of Private fills.
    const ZERO_SOURCE: u64 = 1;

    /// The consumer's hook as a test sees it: each unmap, the thread it ran on, and the first byte
    /// of the block read through the consumer's record of its mapping.
    #[derive(Default)]
    struct Hook {
        /// Block number to the address of its first byte, for the Shared blocks a test records.
        contents: Mutex<HashMap<u32, usize>>,
        unmapped: Mutex<Vec<Unmapped>>,
    }

    #[derive(Clone, Debug)]
    struct Unmapped {
        block: u32,
        thread: Option<String>,
        first: Option<u8>,
    }

    impl BlockMappings for Hook {
        fn unmap(&self, block: BlockId) {
            let first = lock(&self.contents).get(&block.get()).map(|&address| {
                // SAFETY: the runtime keeps the memory mapped until this returns.
                unsafe { (address as *const u8).read_volatile() }
            });
            lock(&self.unmapped).push(Unmapped {
                block: block.get(),
                thread: thread::current().name().map(str::to_string),
                first,
            });
        }
    }

    impl Hook {
        fn record(&self, block: &BlockInfo) {
            let address = block.contents.expect("a Shared block's contents").as_ptr() as usize;
            lock(&self.contents).insert(block.id.get(), address);
        }

        fn unmapped(&self) -> Vec<Unmapped> {
            lock(&self.unmapped).clone()
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

    fn vk_buffer(device: &Device, block: BlockId) -> vk::Buffer {
        device.block(block).expect("a held block").memory.buffer
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

        fn fill(&self, buffer: vk::Buffer, offset: u64, size: u64, word: u32) {
            // SAFETY: recorded in `run`; the range is in the buffer and word-aligned.
            self.run(|device, command| unsafe {
                device.cmd_fill_buffer(command, buffer, offset, size, word)
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

    /// Every block's handed-out number was unmapped exactly once by the end, and no other.
    fn assert_each_unmapped_once(name: &str, hook: &Hook, handed: &[BlockId]) {
        let mut unmapped: Vec<u32> = hook.unmapped().iter().map(|u| u.block).collect();
        unmapped.sort_unstable();
        let mut handed: Vec<u32> = handed.iter().map(|id| id.get()).collect();
        handed.sort_unstable();
        assert_eq!(unmapped, handed, "{name}: the blocks unmapped");
    }

    /// A buffer of exactly [`MAX_BUFFER_LENGTH`] (2 GiB) is made in a Private block of that
    /// length, and one a byte longer is refused with a typed refusal, before its range is looked
    /// at; with the block released and the device dropped, every gauge is zero. A device whose
    /// largest block or memory is smaller skips the 2 GiB buffer visibly.
    #[test]
    fn a_buffer_of_2_gib_is_made_and_a_longer_one_refused() {
        let instance = Instance::new(&[]).expect("an instance at the floor");
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
    /// refusals are Metal's, a block longer than an allocation has an allocation of its own, a
    /// released block lives until its last buffer is gone and is then kept, its bytes as written,
    /// and freed once the window no longer holds its bytes, through the hook on the block-maker
    /// thread while still mapped; and teardown frees the rest through the hook on the dropping
    /// thread, with every gauge at zero.
    #[test]
    fn every_device_hands_out_blocks_and_buffers_and_frees_them_through_the_hook() {
        let instance = Instance::new(&[]).expect("an instance at the floor");
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
                let (from, to) = (
                    vk_buffer(&device, private.id),
                    vk_buffer(&device, shared.id),
                );
                let copy = queue.command_buffer().expect("a command buffer");
                copy.record_for_test(&[], |device, command| {
                    let region = [vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: 8192,
                    }];
                    // SAFETY: the command buffer is recording; both blocks are held and span the
                    // region.
                    unsafe { device.cmd_copy_buffer(command, from, to, &region) };
                })
                .expect("a copy");
                copy.commit().expect("a commit");
                copy.wait_until_completed();
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
            let long_host = bytes(&long);
            long_host[0] = 0x5A;
            hook.record(&long);
            let last_view: Buffer = device
                .new_buffer(long.id, 0, 16, None)
                .expect("a buffer of the long block");

            // Released, it lives while its buffer does, then is kept, its allocation idle.
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
            wait_until("the long block kept", || {
                core.pool.kept(MemoryClass::Shared) == 1
            });
            assert!(hook.unmapped().is_empty(), "{name}: unmapped while kept");

            // Eleven seconds on, the window no longer holds its allocation: the next take, of a
            // small Private block, frees it through the hook on the block-maker thread, still
            // mapped, its bytes as written.
            core.pool.advance_clock(Duration::from_secs(11));
            let nudge = device
                .take_block(MemoryClass::Private, 1)
                .expect("a small Private block");
            let nudge_id = nudge.id;
            device.release_block(nudge).expect("a release");
            wait_until("the long block's unmap", || !hook.unmapped().is_empty());
            let unmapped = hook.unmapped();
            assert_eq!(unmapped.len(), 1, "{name}: {unmapped:?}");
            assert_eq!(unmapped[0].block, long_id.get());
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
            assert_each_unmapped_once(name, &hook, &[shared.id, private.id, long_id, nudge_id]);
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
        let instance = Instance::new(&[]).expect("an instance at the floor");
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

    /// A block given back is kept, its bytes as written, and a take of its length takes it again
    /// with its number, address and mapping, allocating nothing and running no hook: Shared blocks
    /// of 16 MiB, 1 MiB and 40 MiB (an allocation of its own), and a Private block the GPU wrote,
    /// whose bytes are the GPU's when taken again, and read zero in a buffer zeroed over a range of
    /// them, there alone.
    #[test]
    fn every_device_keeps_given_back_blocks_for_takes_of_their_length() {
        const MIB: u64 = 1 << 20;
        let instance = Instance::new(&[]).expect("an instance at the floor");
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let hook = Arc::new(Hook::default());
            let device = device_with(&hook, physical, name);
            let core = device.core();
            let counters = device.counters();
            let lens = [MemoryClass::Shared.block_len(), MIB, 40 * MIB];
            let taken: Vec<Block> = (lens.iter())
                .map(|&len| {
                    (device.take_block(MemoryClass::Shared, len))
                        .unwrap_or_else(|error| panic!("{name}: a block of {len}: {error}"))
                })
                .collect();
            let infos: Vec<BlockInfo> = taken.iter().map(|block| **block).collect();
            for info in &infos {
                assert!(lens.contains(&info.len), "{name}: {} bytes", info.len);
                let host = bytes(info);
                host[0] = 0x5A;
                host[host.len() - 1] = 0x5A;
                hook.record(info);
            }
            for block in taken {
                device.release_block(block).expect("a release");
            }
            wait_until("the three blocks kept", || {
                core.pool.kept(MemoryClass::Shared) == 3 && core.pool.stocked(MemoryClass::Shared)
            });
            let made = counters.metrics().blocks_made;
            let reused = counters.metrics().blocks_reused;
            let again: Vec<Block> = (infos.iter().rev())
                .map(|info| {
                    let block = (device.take_block(MemoryClass::Shared, info.len))
                        .unwrap_or_else(|error| panic!("{name}: {} again: {error}", info.len));
                    assert_eq!(
                        (block.id, block.gpu_address, block.contents, block.len),
                        (info.id, info.gpu_address, info.contents, info.len),
                        "{name}: the kept block of {} bytes",
                        info.len
                    );
                    let host = bytes(&block);
                    assert_eq!(
                        (host[0], host[host.len() - 1]),
                        (0x5A, 0x5A),
                        "{name}: a kept block taken again, its bytes as written"
                    );
                    block
                })
                .collect();
            assert_eq!(
                counters.metrics().blocks_made,
                made,
                "{name}: an allocation"
            );
            assert_eq!(counters.metrics().blocks_reused, reused + 3, "{name}");
            assert!(hook.unmapped().is_empty(), "{name}: {:?}", hook.unmapped());

            // A Private block written by the GPU and given back holds the GPU's bytes when taken
            // again; a buffer zeroed over a range of them reads zero there, ahead of the next
            // command buffer committed, and nowhere else.
            let gpu = Gpu::new(&device);
            let private_len = MemoryClass::Private.block_len();
            let private = (device.take_block(MemoryClass::Private, private_len))
                .unwrap_or_else(|error| panic!("{name}: a Private block: {error}"));
            let private_info = *private;
            gpu.fill(vk_buffer(&device, private.id), 0, private_len, u32::MAX);
            device.release_block(private).expect("a release");
            wait_until("the Private block kept", || {
                core.pool.kept(MemoryClass::Private) == 1
            });
            let back = (device.take_block(MemoryClass::Private, private_len))
                .unwrap_or_else(|error| panic!("{name}: the Private block again: {error}"));
            assert_eq!(
                (back.id, back.gpu_address),
                (private_info.id, private_info.gpu_address),
                "{name}: the kept Private block"
            );
            let shared = (again.iter())
                .find(|block| block.len == MemoryClass::Shared.block_len())
                .expect("the 16 MiB block");
            let zeroed = (device.new_buffer(back.id, 4096, 4096, Some(0)))
                .unwrap_or_else(|error| panic!("{name}: a zeroed Private buffer: {error}"));
            let host = bytes(shared);
            host[..3 * 4096].fill(0x11);
            let queue = device.new_command_queue(1).expect("a queue");
            let (from, to) = (vk_buffer(&device, back.id), vk_buffer(&device, shared.id));
            let copy = queue.command_buffer().expect("a command buffer");
            copy.record_for_test(&[], |device, command| {
                let region = [vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: 3 * 4096,
                }];
                // SAFETY: the command buffer is recording; both blocks are held and span the
                // region.
                unsafe { device.cmd_copy_buffer(command, from, to, &region) };
            })
            .expect("a copy");
            copy.commit().expect("a commit");
            copy.wait_until_completed();
            assert_eq!(copy.status(), CommandBufferStatus::Completed, "{name}");
            for (index, &byte) in host[..3 * 4096].iter().enumerate() {
                let expected = if (4096..8192).contains(&index) {
                    0
                } else {
                    0xFF
                };
                assert_eq!(
                    byte, expected,
                    "{name}: the Private block taken again, byte {index}"
                );
            }
            let mut handed: Vec<BlockId> = infos.iter().map(|info| info.id).collect();
            handed.push(private_info.id);
            drop((zeroed, copy, queue));
            drop((again, back, gpu));
            drop(device);
            assert_eq!(
                counters.gauges(),
                Gauges::default(),
                "{name}: gauges after teardown"
            );
            assert_each_unmapped_once(name, &hook, &handed);
            println!(
                "memory: {name}: four blocks kept and taken again; metrics {:?}",
                counters.metrics()
            );
        }
    }

    /// A take that no free range fits while the class keeps only blocks of other lengths makes its
    /// allocation at once, counted as a taker's allocation, and the block-maker thread then gives
    /// the kept blocks' bytes back to their allocations, after each one's hook on that thread,
    /// still mapped, their bytes as written; a later take is a range of those bytes, allocating nothing: three
    /// kept Shared blocks of 24 MiB and takes of 20 MiB, each returning within 30 seconds.
    #[test]
    fn every_device_allocates_beside_kept_blocks_of_other_lengths_and_gives_their_bytes_back() {
        const MIB: u64 = 1 << 20;
        let instance = Instance::new(&[]).expect("an instance at the floor");
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let hook = Arc::new(Hook::default());
            let device = device_with(&hook, physical, name);
            let core = device.core();
            let counters = device.counters();
            let kept: Vec<Block> = (0..3)
                .map(|_| {
                    (device.take_block(MemoryClass::Shared, 24 * MIB))
                        .unwrap_or_else(|error| panic!("{name}: a block of 24 MiB: {error}"))
                })
                .collect();
            let first = *kept[0];
            let mut handed: Vec<BlockId> = kept.iter().map(|block| block.id).collect();
            for block in &kept {
                bytes(block)[0] = 0x5A;
                hook.record(block);
            }
            for block in kept {
                device.release_block(block).expect("a release");
            }
            wait_until("the three kept", || {
                core.pool.kept(MemoryClass::Shared) == 3
            });
            let before = counters.metrics();
            let take = |what: &str| {
                let (sender, receiver) = std::sync::mpsc::channel();
                let taker = device.clone();
                thread::spawn(move || {
                    let block = taker.take_block(MemoryClass::Shared, 20 * MIB);
                    drop(taker);
                    let _ = sender.send(block);
                });
                (receiver.recv_timeout(Duration::from_secs(30)))
                    .unwrap_or_else(|_| panic!("{name}: {what} waited 30 s"))
                    .unwrap_or_else(|error| panic!("{name}: {what}: {error}"))
            };
            let own = take("the first take");
            let after = counters.metrics();
            assert_eq!(
                (after.blocks_made, after.consumer_allocations),
                (before.blocks_made + 1, before.consumer_allocations + 1),
                "{name}: the first take's allocation, on its thread"
            );
            assert!(
                after.consumer_allocation_nanos > before.consumer_allocation_nanos
                    && after.consumer_allocation_max_nanos > 0,
                "{name}: the taker's allocation time"
            );
            wait_until("the kept blocks' bytes given back", || {
                hook.unmapped().len() == 3 && core.pool.kept(MemoryClass::Shared) == 0
            });
            for unmapped in hook.unmapped() {
                assert_eq!(unmapped.thread.as_deref(), Some("m2v-blocks"), "{name}");
                assert_eq!(
                    unmapped.first,
                    Some(0x5A),
                    "{name}: still mapped, its bytes as written"
                );
            }
            let shared = take("the second take");
            assert_eq!(
                shared.contents.map(|c| c.as_ptr() as usize),
                first.contents.map(|c| c.as_ptr() as usize),
                "{name}: a range of the first kept block's bytes"
            );
            assert_eq!(
                bytes(&shared)[0],
                0x5A,
                "{name}: a block of a kept block's bytes, as written"
            );
            assert_eq!(
                counters.metrics().blocks_made,
                before.blocks_made + 1,
                "{name}: an allocation for the second take"
            );
            handed.extend([own.id, shared.id]);
            drop((own, shared));
            drop(device);
            assert_eq!(
                counters.gauges(),
                Gauges::default(),
                "{name}: gauges after teardown"
            );
            assert_each_unmapped_once(name, &hook, &handed);
            println!(
                "memory: {name}: takes beside kept blocks of other lengths; metrics {:?}",
                counters.metrics()
            );
        }
    }

    /// Where a take's own allocation fails while the class keeps blocks of other lengths, the
    /// take waits for the block-maker thread to give their bytes back and takes a range of them:
    /// with the device's `maxMemoryAllocationCount` answered as its live allocations once a
    /// 24 MiB Shared block has one of its own, that block kept, a take of 20 MiB returns within
    /// 30 seconds in its bytes, and a take of 40 MiB, which no free range fits, returns the
    /// allocation's `Limit::MemoryAllocations`.
    #[test]
    fn every_device_takes_kept_blocks_bytes_where_its_allocation_fails() {
        const MIB: u64 = 1 << 20;
        const MAX: u32 = ZERO_SOURCE as u32 + 3;
        let instance = Instance::new(&[]).expect("an instance at the floor");
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let hook = Arc::new(Hook::default());
            let options = DeviceOptions {
                block_mappings: Some(Arc::clone(&hook) as Arc<dyn BlockMappings>),
                ..DeviceOptions::default()
            };
            let device = Device::open(physical, options, |limits| {
                limits.max_memory_allocation_count = MAX;
            })
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let core = device.core();
            let counters = device.counters();
            let kept = (device.take_block(MemoryClass::Shared, 24 * MIB))
                .unwrap_or_else(|error| panic!("{name}: a block of 24 MiB: {error}"));
            assert_eq!(counters.gauges().blocks, u64::from(MAX), "{name}");
            let kept_info = *kept;
            device.release_block(kept).expect("a release");
            wait_until("the block kept", || {
                core.pool.kept(MemoryClass::Shared) == 1
            });
            let (sender, receiver) = std::sync::mpsc::channel();
            let taker = device.clone();
            thread::spawn(move || {
                let block = taker.take_block(MemoryClass::Shared, 20 * MIB);
                drop(taker);
                let _ = sender.send(block);
            });
            let block = (receiver.recv_timeout(Duration::from_secs(30)))
                .unwrap_or_else(|_| panic!("{name}: the take waited 30 s"))
                .unwrap_or_else(|error| panic!("{name}: the take: {error}"));
            assert_eq!(
                block.gpu_address, kept_info.gpu_address,
                "{name}: the kept block's bytes"
            );
            let unmapped = hook.unmapped();
            assert_eq!(unmapped.len(), 1, "{name}: {unmapped:?}");
            assert_eq!(
                (unmapped[0].block, unmapped[0].thread.as_deref()),
                (kept_info.id.get(), Some("m2v-blocks")),
                "{name}"
            );
            match device.take_block(MemoryClass::Shared, 40 * MIB) {
                Err(Error::Limit(Limit::MemoryAllocations { max: MAX })) => {}
                other => panic!("{name}: a take past the count: {other:?}"),
            }
            assert_eq!(counters.gauges().blocks, u64::from(MAX), "{name}");
            let handed = [kept_info.id, block.id];
            drop(block);
            drop(device);
            assert_eq!(
                counters.gauges(),
                Gauges::default(),
                "{name}: gauges after teardown"
            );
            assert_each_unmapped_once(name, &hook, &handed);
            println!("memory: {name}: a failed allocation took the kept block's bytes");
        }
    }

    /// A long block given back is taken again at once by the next take of its length: a Shared
    /// block of 64 MiB, which takes an allocation of its own, is taken, given a zeroed buffer, and
    /// given back 20 times in a row, each take straight after the give-back before. Whether the
    /// block-maker thread has kept the block yet or not (it may be inside a driver's allocation or
    /// free), the take finds it: only the first take allocates, and nothing is freed.
    #[test]
    fn every_device_takes_a_long_block_given_back_again_at_once() {
        const LEN: u64 = 64 << 20;
        let instance = Instance::new(&[]).expect("an instance at the floor");
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
                    after.blocks_reused - before.blocks_reused,
                    after.allocations_freed - before.allocations_freed
                ),
                (19, 0),
                "{name}: takes of the block given back, and allocations freed"
            );
        }
    }

    /// Idle allocations follow the window's peak down: four Shared blocks of 16 MiB taken beside
    /// a small one, each with an allocation of its own, and given back are kept, their
    /// allocations idle, while the most bytes in use over the last 3 seconds hold them, through a
    /// Private take and give-back and the small block's give-back 2 seconds on. Once the clock has
    /// moved 4 seconds, the small block's allocation is the reserve that keeps 8 MiB free, and the
    /// window holds one allocation more than the bytes in use: the three oldest idle allocations
    /// are freed, one at a time, after the hook of the block kept in each, on the block-maker
    /// thread. Two seconds later, with no take or give-back, the window's peak falls to nothing
    /// and the fourth is freed; the reserve stays. Nothing commits, so the device is quiet
    /// throughout.
    #[test]
    fn every_device_frees_idle_allocations_as_the_window_s_peak_falls() {
        let instance = Instance::new(&[]).expect("an instance at the floor");
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
            let four: Vec<u32> = taken.iter().map(|block| block.id.get()).collect();
            let mut handed: Vec<BlockId> = taken.iter().map(|block| block.id).collect();
            handed.push(small.id);
            for block in taken {
                device.release_block(block).expect("a release");
            }
            wait_until("the four kept", || core.pool.kept(MemoryClass::Shared) == 4);
            assert_eq!(
                counters.gauges().shared_idle_bytes,
                4 * len,
                "{name}: four idle allocations"
            );

            // Two seconds on, a Private take and give-back, and the small block's give-back: the
            // window still holds the four.
            core.pool.advance_clock(Duration::from_secs(2));
            let nudge = (device.take_block(MemoryClass::Private, 1))
                .unwrap_or_else(|error| panic!("{name}: a Private block: {error}"));
            handed.push(nudge.id);
            device.release_block(nudge).expect("a release");
            device.release_block(small).expect("a release");
            wait_until("the Private and small blocks kept", || {
                core.pool.kept(MemoryClass::Private) == 1
                    && core.pool.kept(MemoryClass::Shared) == 5
            });
            assert_eq!(
                (counters.gauges().shared_idle_bytes, hook.unmapped().len()),
                (5 * len, 0),
                "{name}: the four and the small block's within the window"
            );

            // Four seconds on: the three oldest are freed. The gauge of allocations falls after
            // each one's free, which follows its kept block's hook and the idle gauge's fall: once
            // it reads the zero source, the reserve, the fourth and the Private allocation, the
            // three hooks and the idle gauge are final.
            core.pool.advance_clock(Duration::from_secs(2));
            wait_until("three idle Shared allocations freed", || {
                counters.gauges().blocks == ZERO_SOURCE + 3
            });
            assert_eq!(
                (hook.unmapped().len(), counters.gauges().shared_idle_bytes),
                (3, 2 * len),
                "{name}: three unmapped, the reserve and the fourth idle"
            );
            for (unmapped, block) in hook.unmapped().iter().zip(&four) {
                assert_eq!(unmapped.block, *block, "{name}: the oldest first");
                assert_eq!(unmapped.thread.as_deref(), Some("m2v-blocks"), "{name}");
                assert_eq!(unmapped.first, None, "{name}: no block recorded");
            }

            // Two seconds later, with no take or give-back: the window's peak is gone, and the
            // fourth is freed.
            core.pool.advance_clock(Duration::from_secs(2));
            wait_until("the fourth freed", || {
                counters.gauges().blocks == ZERO_SOURCE + 2
            });
            assert_eq!(
                (hook.unmapped().len(), counters.gauges().shared_idle_bytes),
                (4, len),
                "{name}: the fourth unmapped, the reserve idle"
            );
            assert_eq!(hook.unmapped()[3].block, four[3], "{name}: the fourth");
            assert_eq!(
                core.pool.kept(MemoryClass::Shared),
                1,
                "{name}: the small block, kept in the reserve"
            );
            assert_eq!(counters.metrics().allocations_freed, 4, "{name}: the frees");
            drop(device);
            assert_eq!(
                counters.gauges(),
                Gauges::default(),
                "{name}: gauges after teardown"
            );
            assert_each_unmapped_once(name, &hook, &handed);
            println!(
                "memory: {name}: idle allocations freed as the peak fell; metrics {:?}",
                counters.metrics()
            );
        }
    }

    /// A commit is not held for a free of host-visible memory beside it: a command buffer
    /// committed 1 ms into the free of a 64 MiB Shared allocation the GPU used completes in less
    /// than half the free's time. NVIDIA's Linux driver held the submission of a command buffer
    /// from a pool reset keeping its memory for the whole of such a free (20 to 70 ms on an RTX
    /// 2080 and 3090); the recorders' pools are reset releasing it. A device whose free takes
    /// under 10 ms only prints its times.
    #[test]
    fn every_device_completes_a_commit_beside_a_free_of_shared_memory_unheld() {
        each_device("commit beside a free", |name, _, device| {
            let core = device.core();
            let queue = device.new_command_queue(4).expect("a queue");
            let target = device
                .take_block(MemoryClass::Private, 4096)
                .expect("a block");
            let kept = device
                .new_buffer(target.id, 0, 256, None)
                .expect("a buffer");
            let to = device.block(target.id).expect("a held block").memory.buffer;
            let commit = |fill: vk::Buffer| {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                command_buffer
                    .record_for_test(&[&kept], |device, command| unsafe {
                        device.cmd_fill_buffer(command, fill, 0, 256, 7)
                    })
                    .expect("a fill");
                let started = Instant::now();
                command_buffer.commit().expect("a commit");
                command_buffer.wait_until_completed();
                started.elapsed()
            };
            // The queue's recorders reused, as a program's are.
            for _ in 0..3 {
                commit(to);
            }
            for _ in 0..3 {
                let allocation =
                    make_allocation(core, MemoryClass::Shared, 64 << 20).expect("an allocation");
                let info = vk::BufferCreateInfo::default()
                    .size(256)
                    .usage(crate::floor::BUFFER_USAGE)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE);
                // SAFETY: the device is live; the buffer is bound once, to memory of its usage's
                // type, and destroyed once the fill that uses it completed.
                let used = unsafe {
                    let used = core.device.create_buffer(&info, None).expect("a buffer");
                    (core.device)
                        .bind_buffer_memory(used, allocation.memory, 0)
                        .expect("its memory");
                    used
                };
                commit(used);
                // SAFETY: the fill that used it completed.
                unsafe { core.device.destroy_buffer(used, None) };
                let (freed, completed) = thread::scope(|scope| {
                    let committed = scope.spawn(|| {
                        thread::sleep(Duration::from_millis(1));
                        commit(to)
                    });
                    let started = Instant::now();
                    free_allocation(core, allocation);
                    let freed = started.elapsed();
                    (freed, committed.join().expect("the commit"))
                });
                println!(
                    "{name}: a 64 MiB Shared free took {freed:?}; a commit 1 ms into it completed \
                     in {completed:?}"
                );
                if freed >= Duration::from_millis(10) {
                    assert!(
                        completed < freed / 2,
                        "{name}: the commit waited for the free: {completed:?} of {freed:?}"
                    );
                }
                thread::sleep(Duration::from_millis(100));
            }
            drop((kept, target, queue));
        });
    }

    /// Idle allocations are freed only in quiet gaps, and a submission during a free is counted.
    /// Four Shared blocks of 16 MiB are given back while command buffers are committed every
    /// 2 ms; the clock moves past the window, so the idle rule frees all four but the reserve, yet
    /// every free begins at least 20 ms after the last commit before it, whatever the test
    /// thread's own gaps were. Once the commits stop, the idle allocations are freed. Then the
    /// free of one of two more idle allocations is held open: a command buffer committed with a
    /// fill during it is counted as a submission during a free, the other's free begins no sooner
    /// than 20 ms after the held one ended, though the hold outlasted the fill by more than that,
    /// and a command buffer committed once both are freed is not counted.
    #[test]
    fn every_device_frees_idle_allocations_in_quiet_gaps() {
        const QUIET: u64 = 20_000_000;
        let instance = Instance::new(&[]).expect("an instance at the floor");
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::new(physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let core = device.core();
            let counters = device.counters();
            let queue = device.new_command_queue(1).expect("a queue");
            let len = MemoryClass::Shared.block_len();
            let commit = || {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let before = crate::host_time();
                command_buffer.commit().expect("a commit");
                (before, crate::host_time())
            };
            let give_back = |count: usize| {
                let taken: Vec<Block> = (0..count)
                    .map(|_| {
                        (device.take_block(MemoryClass::Shared, len))
                            .unwrap_or_else(|error| panic!("{name}: a block: {error}"))
                    })
                    .collect();
                for block in taken {
                    device.release_block(block).expect("a release");
                }
                wait_until("the blocks kept", || {
                    core.pool.kept(MemoryClass::Shared) == count
                });
            };
            give_back(4);
            let mut commits = vec![commit()];

            // Commits every 2 ms, the four idle allocations past the window from the third on.
            for step in 0..150 {
                if step == 2 {
                    core.pool.advance_clock(Duration::from_secs(11));
                }
                thread::sleep(Duration::from_millis(2));
                commits.push(commit());
            }
            let busy_frees = counters.metrics().allocations_freed;
            // The count rises once each free has returned.
            wait_until("the idle allocations freed once quiet", || {
                counters.metrics().allocations_freed == 4 && !core.pool.freeing()
            });
            let quiet_frees = core.pool.quiet_frees();
            for (index, &quiet) in quiet_frees.iter().enumerate() {
                for &(before, after) in &commits {
                    assert!(
                        after > quiet || before + QUIET <= quiet,
                        "{name}: free {index} began {} us after a commit",
                        quiet.saturating_sub(before) / 1000
                    );
                }
            }
            let gaps = (commits.windows(2))
                .filter(|pair| pair[1].0 - pair[0].1 >= QUIET)
                .count();
            println!(
                "memory: {name}: {} frees in quiet gaps, {busy_frees} of them while the test \
                 committed, with {gaps} gaps of 20 ms or more between its commits",
                quiet_frees.len()
            );

            // A free held open: a submission during it is counted, one after it is not.
            let target = (device.take_block(MemoryClass::Shared, 4096))
                .unwrap_or_else(|error| panic!("{name}: a block: {error}"));
            let buffer = (device.new_buffer(target.id, 0, 256, None)).expect("a buffer");
            let fill = |buffer: &crate::Buffer| {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let to = vk_buffer(&device, target.id);
                command_buffer
                    .record_for_test(&[buffer], |device, command| {
                        // SAFETY: the command buffer is recording; the block is held.
                        unsafe { device.cmd_fill_buffer(command, to, 0, 256, 7) };
                    })
                    .expect("a fill");
                command_buffer.commit().expect("a commit");
                command_buffer.wait_until_completed();
            };
            let freed = counters.metrics().allocations_freed;
            core.pool.hold_frees(true);
            give_back(2);
            core.pool.advance_clock(Duration::from_secs(11));
            wait_until("a free held open", || core.pool.freeing());
            fill(&buffer);
            assert_eq!(
                counters.metrics().submissions_during_frees,
                1,
                "{name}: the submission during the free"
            );
            // Held past the quiet time after the submission, as a driver may hold a submission for
            // a whole free: the next free still waits the quiet time from this free's end.
            thread::sleep(Duration::from_nanos(QUIET + 10_000_000));
            let released = crate::host_time();
            core.pool.hold_frees(false);
            wait_until("both freed", || {
                counters.metrics().allocations_freed == freed + 2 && !core.pool.freeing()
            });
            let next = *core.pool.quiet_frees().last().expect("the second free");
            assert!(
                next >= released + QUIET,
                "{name}: the free after a held submission began {} us after the hold ended",
                next.saturating_sub(released) / 1000
            );
            fill(&buffer);
            assert_eq!(
                counters.metrics().submissions_during_frees,
                1,
                "{name}: the submission after the free"
            );
            drop((buffer, target, queue));
            drop(device);
            wait_until("the teardown", || counters.gauges() == Gauges::default());
            println!(
                "memory: {name}: frees only in quiet gaps; metrics {:?}",
                counters.metrics()
            );
        }
    }

    /// Seeded takes and releases of mixed lengths across both classes, with the block-maker
    /// thread running: every take and release returns within a minute, every block is unmapped
    /// once by the end, on the block-maker thread or the dropping one, and every gauge reads zero
    /// after teardown.
    #[test]
    fn every_device_returns_every_seeded_take_and_release_of_mixed_lengths() {
        use std::sync::mpsc::RecvTimeoutError;
        const OPS: usize = 300;
        /// The most live bytes of a class the run keeps: past it, it releases.
        const MOST: u64 = 192 << 20;
        let instance = Instance::new(&[]).expect("an instance at the floor");
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        let this_thread = thread::current().name().map(str::to_string);
        for (name, physical) in &devices {
            let hook = Arc::new(Hook::default());
            let device = device_with(&hook, physical, name);
            let counters = device.counters();
            let worker = device.clone();
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
            assert_each_unmapped_once(name, &hook, &handed);
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
        let instance = Instance::new(&[]).expect("an instance at the floor");
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
            match first.replace_region(whole, 0, 0, &[0x5A; 64], 16, 64) {
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
        let instance = Instance::new(&[]).expect("an instance at the floor");
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
                        src_offset: SOURCE + start,
                        dst_offset: AT + start,
                        size: end - start,
                    })
                    .collect();
                let back = [vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
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
                            device.cmd_copy_buffer(command, to_shared, to_private, &regions);
                            device.cmd_pipeline_barrier2(command, &dependency);
                            device.cmd_copy_buffer(command, to_private, to_shared, &back);
                        }
                    })
                    .expect("the copies");
                command_buffer.commit().expect("a commit");
                command_buffer.wait_until_completed();
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
}
