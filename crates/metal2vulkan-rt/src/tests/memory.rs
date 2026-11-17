//! Memory blocks and buffers: the plan of a zero fill and a dedicated block's length; and, under a
//! device cfg, on every device of the cfg's kind, that blocks are handed out zeroed and buffers
//! view them, that a block is freed through the consumer's hook only once nothing uses it, that
//! the pool keeps two idle blocks of a class and frees the rest, and that teardown frees every
//! block with the gauges at zero.

use crate::fill::{plan, Plan};
use crate::memory::dedicated_len;
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

#[test]
fn a_dedicated_block_is_the_length_rounded_up_to_64_kib_within_the_device() {
    let max = 4 << 30;
    let granule = 64 << 10;
    assert_eq!(dedicated_len(257 << 20, max).unwrap(), 257 << 20);
    assert_eq!(
        dedicated_len((256 << 20) + 1, max).unwrap(),
        (256 << 20) + granule
    );
    assert_eq!(dedicated_len(max, max).unwrap(), max);
    for (len, reported) in [(max + 1, max + granule), (u64::MAX, u64::MAX)] {
        match dedicated_len(len, max) {
            Err(Error::Limit(Limit::BlockLen { len, max: limit })) => {
                assert_eq!((len, limit), (reported, max))
            }
            other => panic!("{len} bytes: {other:?}"),
        }
    }
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::device::lock;
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{
        BlockId, BlockInfo, BlockMappings, Buffer, Device, DeviceOptions, Error, Gauges, Instance,
        Limit, MemoryClass, Refusal, BLOCK_LEN,
    };
    use ash::vk;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    /// The runtime's own allocation besides blocks: the zero source of Private fills.
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
    fn bytes(block: BlockInfo) -> &'static mut [u8] {
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

        fn copy(&self, from: vk::Buffer, to: vk::Buffer, offset: u64, size: u64) {
            let region = [vk::BufferCopy {
                src_offset: offset,
                dst_offset: offset,
                size,
            }];
            // SAFETY: recorded in `run`; the range is in both buffers.
            self.run(|device, command| unsafe {
                device.cmd_copy_buffer(command, from, to, &region)
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

    /// Blocks of both classes are handed out zeroed, buffers view them, a zeroed buffer zeroes
    /// exactly its bytes, the refusals are Metal's, a released block lives until its last buffer
    /// is gone and is then freed through the hook while still mapped, and teardown frees the rest
    /// through the hook on the dropping thread, with every gauge at zero.
    #[test]
    fn every_device_hands_out_zeroed_blocks_and_buffers_and_frees_them_through_the_hook() {
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
            let counters = device.counters();
            let gpu = Gpu::new(&device);

            // A new Shared block reads zero, through its mapping.
            let shared = device
                .take_block(MemoryClass::Shared, 1)
                .expect("a Shared block");
            assert_eq!((shared.class, shared.len), (MemoryClass::Shared, BLOCK_LEN));
            assert_ne!(shared.gpu_address, 0, "{name}: a Shared block's address");
            let host = bytes(shared);
            assert!(
                host.iter().all(|&byte| byte == 0),
                "{name}: a new Shared block"
            );

            // A new Private block reads zero, copied into the Shared block.
            let private = device
                .take_block(MemoryClass::Private, BLOCK_LEN)
                .expect("a Private block");
            assert_eq!(
                (private.class, private.len, private.contents),
                (MemoryClass::Private, BLOCK_LEN, None)
            );
            assert_ne!(private.gpu_address, 0, "{name}: a Private block's address");
            assert_ne!(private.id, shared.id);
            host.fill(0xFF);
            gpu.copy(
                vk_buffer(&device, private.id),
                vk_buffer(&device, shared.id),
                0,
                BLOCK_LEN,
            );
            assert!(
                host.iter().all(|&byte| byte == 0),
                "{name}: a new Private block"
            );

            // Buffers are views: the block's address and mapping plus the offset.
            let view = device
                .new_buffer(shared.id, 4096, 256, false)
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
                .new_buffer(private.id, 0, BLOCK_LEN, false)
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

            // A zeroed buffer zeroes exactly its bytes: in a Shared block through the mapping, in a
            // Private one by a fill of its whole words and copies of the bytes beside them.
            host[..8192].fill(0xFF);
            let zeroed = device
                .new_buffer(shared.id, 4097, 7, true)
                .expect("a zeroed Shared buffer");
            for (index, &byte) in host[..8192].iter().enumerate() {
                let inside = (4097..4104).contains(&index);
                assert_eq!(
                    byte,
                    if inside { 0 } else { 0xFF },
                    "{name}: Shared byte {index}"
                );
            }
            drop(zeroed);
            for (offset, length) in [(4097, 13), (8001, 2), (4096, 64), (4099, 1), (1, 8000)] {
                gpu.fill(vk_buffer(&device, private.id), 0, 8192, u32::MAX);
                let zeroed = device
                    .new_buffer(private.id, offset, length, true)
                    .expect("a zeroed Private buffer");
                gpu.copy(
                    vk_buffer(&device, private.id),
                    vk_buffer(&device, shared.id),
                    0,
                    8192,
                );
                let range = offset as usize..(offset + length) as usize;
                for (index, &byte) in host[..8192].iter().enumerate() {
                    let expected = if range.contains(&index) { 0 } else { 0xFF };
                    assert_eq!(
                        byte, expected,
                        "{name}: {offset}+{length}, Private byte {index}"
                    );
                }
                drop(zeroed);
            }

            // Metal's refusals, and the device's limit.
            assert!(matches!(
                device.new_buffer(shared.id, 0, 0, false),
                Err(Error::Refused(Refusal::EmptyBuffer))
            ));
            for (offset, length) in [(BLOCK_LEN - 4, 8), (u64::MAX, 2), (BLOCK_LEN, 1)] {
                match device.new_buffer(shared.id, offset, length, false) {
                    Err(Error::Refused(Refusal::BufferPastBlock {
                        offset: o,
                        length: l,
                        block_len,
                    })) => assert_eq!((o, l, block_len), (offset, length, BLOCK_LEN)),
                    other => panic!("{name}: {offset}+{length}: {:?}", other.err()),
                }
            }
            assert!(matches!(
                device.take_block(MemoryClass::Shared, u64::MAX),
                Err(Error::Limit(Limit::BlockLen { len: u64::MAX, .. }))
            ));

            // A dedicated block: past BLOCK_LEN, rounded up to 64 KiB, zeroed.
            let dedicated = device
                .take_block(MemoryClass::Shared, BLOCK_LEN + 1)
                .expect("a dedicated block");
            assert_eq!(dedicated.len, BLOCK_LEN + (64 << 10));
            let dedicated_host = bytes(dedicated);
            assert!(
                dedicated_host.iter().all(|&byte| byte == 0),
                "{name}: a dedicated block"
            );
            dedicated_host[0] = 0x5A;
            hook.record(&dedicated);
            let last_view: Buffer = device
                .new_buffer(dedicated.id, 0, 16, false)
                .expect("a buffer of the dedicated block");

            // Released, it lives while its buffer does, then is freed through the hook on the
            // block-maker thread, still mapped.
            device.release_block(dedicated.id).expect("a release");
            assert!(matches!(
                device.release_block(dedicated.id),
                Err(Error::NoSuchBlock(id)) if id == dedicated.id
            ));
            assert!(matches!(
                device.new_buffer(dedicated.id, 0, 16, false),
                Err(Error::NoSuchBlock(_))
            ));
            assert!(
                hook.unmapped().is_empty(),
                "{name}: unmapped with a buffer alive"
            );
            // SAFETY: the buffer keeps the block mapped.
            let first = unsafe { last_view.contents().unwrap().as_ptr().read_volatile() };
            assert_eq!(first, 0x5A);
            drop(last_view);
            wait_until("the dedicated block's unmap", || {
                !hook.unmapped().is_empty()
            });
            let unmapped = hook.unmapped();
            assert_eq!(unmapped.len(), 1, "{name}: {unmapped:?}");
            assert_eq!(unmapped[0].block, dedicated.id.get());
            assert_eq!(unmapped[0].thread.as_deref(), Some("m2v-blocks"));
            assert_eq!(
                unmapped[0].first,
                Some(0x5A),
                "{name}: unmapped while still mapped"
            );

            // Teardown unmaps the rest on this thread, and frees every block.
            drop((view, whole, gpu));
            drop(device);
            let gauges = counters.gauges();
            print_gauges_at_drop("blocks_teardown", gauges);
            assert_eq!(gauges, Gauges::default(), "{name}: gauges after teardown");
            assert_each_unmapped_once(name, &hook, &[shared.id, private.id, dedicated.id]);
            for unmapped in &hook.unmapped()[1..] {
                assert_eq!(unmapped.thread, this_thread, "{name}: {unmapped:?}");
            }
            println!(
                "memory: {name}: blocks and buffers; metrics {:?}",
                counters.metrics()
            );
        }
    }

    /// Past two idle blocks of a class, the pool frees what is given back through the hook; a
    /// block it keeps is zeroed and handed out again with its number, address and mapping, with no
    /// block made; and a Private block taken again reads zero too.
    #[test]
    fn every_device_keeps_two_idle_blocks_per_class_and_frees_the_rest() {
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
            let mut handed = Vec::new();

            let taken: Vec<BlockInfo> = (0..3)
                .map(|_| {
                    device
                        .take_block(MemoryClass::Shared, BLOCK_LEN)
                        .expect("a block")
                })
                .collect();
            for block in &taken {
                handed.push(block.id);
                let host = bytes(*block);
                host[0] = 0x5A;
                host[host.len() - 1] = 0x5A;
                hook.record(block);
            }
            // The three, one ready block of each class, and the zero source.
            wait_until("a ready block of each class", || {
                counters.gauges().blocks == ZERO_SOURCE + 2 + 3
            });
            for block in &taken {
                device.release_block(block.id).expect("a release");
            }
            wait_until("two blocks freed", || hook.unmapped().len() == 2);
            wait_until("the pool settled", || {
                counters.gauges().blocks == ZERO_SOURCE + 1 + 2
            });
            let unmapped = hook.unmapped();
            for unmapped in &unmapped {
                assert_eq!(unmapped.thread.as_deref(), Some("m2v-blocks"));
                assert_eq!(
                    unmapped.first,
                    Some(0x5A),
                    "{name}: unmapped while still mapped"
                );
            }
            let kept: Vec<&BlockInfo> = taken
                .iter()
                .filter(|block| unmapped.iter().all(|u| u.block != block.id.get()))
                .collect();
            assert_eq!(kept.len(), 1, "{name}: {unmapped:?}");

            // The kept block comes back first, zeroed, as it was, with nothing made.
            let made = counters.metrics().blocks_made;
            let again = device
                .take_block(MemoryClass::Shared, 1)
                .expect("a block again");
            assert_eq!(again.id, kept[0].id, "{name}: the latest idle block first");
            assert_eq!(again.gpu_address, kept[0].gpu_address);
            assert_eq!(again.contents, kept[0].contents);
            let host = bytes(again);
            assert!(
                host.iter().all(|&byte| byte == 0),
                "{name}: a block taken again"
            );
            assert_eq!(counters.metrics().blocks_made, made, "{name}: a block made");

            // A Private block written by the GPU and released reads zero when taken again.
            let gpu = Gpu::new(&device);
            let private = device
                .take_block(MemoryClass::Private, 1)
                .expect("a Private block");
            handed.push(private.id);
            gpu.fill(vk_buffer(&device, private.id), 0, BLOCK_LEN, u32::MAX);
            device.release_block(private.id).expect("a release");
            // The pool zeroes and keeps it before it makes anything, so it is one of the next two.
            let next: Vec<BlockInfo> = (0..2)
                .map(|_| device.take_block(MemoryClass::Private, 1).expect("a block"))
                .collect();
            handed.extend(next.iter().map(|block| block.id));
            let back = next
                .iter()
                .find(|block| block.id == private.id)
                .expect("the released Private block among the next two");
            assert_eq!(back.gpu_address, private.gpu_address);
            host.fill(0xFF);
            gpu.copy(
                vk_buffer(&device, private.id),
                vk_buffer(&device, again.id),
                0,
                BLOCK_LEN,
            );
            assert!(
                host.iter().all(|&byte| byte == 0),
                "{name}: a Private block taken again"
            );

            drop(gpu);
            drop(device);
            assert_eq!(
                counters.gauges(),
                Gauges::default(),
                "{name}: gauges after teardown"
            );
            handed.sort_unstable();
            handed.dedup();
            assert_each_unmapped_once(name, &hook, &handed);
            println!(
                "memory: {name}: the pool kept {} and freed 2; metrics {:?}",
                kept[0].id.get(),
                counters.metrics()
            );
        }
    }
}
