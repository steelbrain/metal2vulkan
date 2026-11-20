//! Managed buffers: changed ranges merge where they overlap or touch; and, under a device cfg, on
//! every device of the cfg's kind, in the form Vulkan's device type picks: with a device copy
//! where the device is neither integrated nor a CPU, the copy holds the consumer's first bytes and
//! zeros after them, a changed range reaches it at the next commit and a range not marked changed
//! does not, a change submits nothing, and the copy-in at a commit adds one barrier; without one,
//! shaders read the consumer's bytes as they are. In both forms a command buffer reads the bytes
//! the consumer wrote and named before its commit (Metal's `didModifyRange:` contract), though the
//! consumer's buffer and block handles go before it runs; nothing writes under a pending command
//! buffer, where Metal leaves the range undefined for the GPU. A range of three 16 MiB units named
//! arrives whole at one commit, which takes no staging and allocates nothing. A Managed buffer over
//! an imported dma-buf, a block the runtime does not map, has no `contents`, and a kernel reads the
//! bytes the consumer wrote through its own mapping and named, in either form. Metal's two-copy
//! rule holds where `Device::has_unified_memory` is false: a kernel's write stays in the device
//! copy, and a range named with no host write puts the consumer's bytes back over it. A range past
//! the buffer and a Managed buffer over a Private block are refused, and every gauge reads zero
//! after teardown. The consumer places the device copy: at an odd offset of its Private block it
//! holds the bytes there and nowhere else, the consumer's slack after it and the bytes before it
//! as the consumer left them; a missing copy, a copy on a unified device, one in a Shared block
//! and one past its block are refused.
//!
//! Managed textures, on every device: the image takes the named bytes of a region at its upload
//! and nothing outside it, `get_bytes` reads the named bytes, `replace_region` writes them and
//! uploads them, and places that do not lie as a copy reads them are refused. The upload keeps the
//! named buffer's block it reads until it retires, though the texture, the buffer and the block's
//! handle go right after the call.

use crate::capture::merge;

#[test]
fn changed_ranges_merge_where_they_overlap_or_touch() {
    let mut ranges = Vec::new();
    merge(&mut ranges, 100, 200);
    merge(&mut ranges, 300, 400);
    assert_eq!(ranges, [(100, 200), (300, 400)]);
    merge(&mut ranges, 0, 50);
    assert_eq!(ranges, [(0, 50), (100, 200), (300, 400)], "before, apart");
    merge(&mut ranges, 200, 250);
    assert_eq!(
        ranges,
        [(0, 50), (100, 250), (300, 400)],
        "touching the end"
    );
    merge(&mut ranges, 290, 300);
    assert_eq!(
        ranges,
        [(0, 50), (100, 250), (290, 400)],
        "touching the start"
    );
    merge(&mut ranges, 120, 130);
    assert_eq!(ranges, [(0, 50), (100, 250), (290, 400)], "inside one");
    merge(&mut ranges, 40, 295);
    assert_eq!(ranges, [(0, 400)], "across three");
    merge(&mut ranges, 500, 600);
    merge(&mut ranges, 0, 1000);
    assert_eq!(ranges, [(0, 1000)], "over all");
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(crate) mod on_device {
    use crate::buffer::BufferObject;
    use crate::tests::command::on_device::{each_device, fill, finished, within, Arena};
    use crate::tests::compute::on_device::{add, step, WORDS};
    use crate::{
        Block, BlockId, Buffer, BytesPlace, CommandBuffer, CommandBufferStatus,
        ComputePipelineState, Device, DispatchType, Error, MemoryClass, PixelFormat, Refusal,
        Region, StorageMode, TextureDescriptor, TextureType, TextureUsage,
    };
    use ash::vk;
    use std::time::Instant;

    const LEN: u64 = 64 << 10;

    /// [`Device::new_managed_buffer`] of `length` bytes at `offset` in Shared block `bytes`, placed
    /// as a consumer places it: its device copy, where the device keeps one, at the start of a
    /// Private block taken for it, which the buffer holds once the handle here is gone.
    pub(crate) fn new_managed(
        device: &Device,
        bytes: BlockId,
        offset: u64,
        length: u64,
        copied: u64,
    ) -> Result<Buffer, Error> {
        if device.has_unified_memory() {
            return device.new_managed_buffer((bytes, offset), None, length, copied);
        }
        let copy = device.take_block(MemoryClass::Private, length)?;
        device.new_managed_buffer((bytes, offset), Some((copy.id, 0)), length, copied)
    }

    /// The bytes shaders reach at `managed`'s address, `len` of them, copied by the GPU into
    /// `into`, a Shared buffer as long.
    fn device_bytes(device: &Device, managed: &Buffer, into: &Buffer, len: u64) -> Vec<u8> {
        let queue = device.new_command_queue(1).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let (source, from) = managed.device_range();
        let (target, to) = into.device_range();
        command_buffer
            .record_for_test(&[managed, into], |device, command| {
                let region = vk::BufferCopy {
                    src_offset: from,
                    dst_offset: to,
                    size: len,
                };
                // SAFETY: the command buffer is recording; both ranges are the buffers', which it
                // keeps.
                unsafe { device.cmd_copy_buffer(command, source, target, &[region]) };
            })
            .expect("a copy");
        command_buffer.commit().expect("a commit");
        finished("the copy", &command_buffer);
        let contents = into.contents().expect("a Shared buffer");
        // SAFETY: the copy into the buffer completed, and its writes reach the host.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), len as usize) }.to_vec()
    }

    /// Writes `bytes` at `at` in the consumer's bytes of `managed`.
    fn write(managed: &Buffer, at: u64, bytes: &[u8]) {
        let contents = managed.contents().expect("the consumer's bytes");
        // SAFETY: `at..at + bytes.len()` lies in the buffer, and nothing reads it meanwhile.
        unsafe {
            bytes
                .as_ptr()
                .copy_to_nonoverlapping(contents.as_ptr().add(at as usize), bytes.len())
        };
    }

    /// The first `len` bytes of a Shared buffer.
    fn read(buffer: &Buffer, len: usize) -> Vec<u8> {
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the buffer is at least `len` long, and no GPU work writes it meanwhile.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), len) }.to_vec()
    }

    #[test]
    fn every_device_copies_a_managed_buffer_s_changed_ranges() {
        each_device("managed", |name, _, device| {
            let core = device.core();
            let copied = !core.unified_memory();
            let counters = device.counters();
            let cpu = (device.take_block(MemoryClass::Shared, 2 * LEN))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let into = (device.new_buffer(cpu.id, LEN, LEN, Some(0)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let first: Vec<u8> = (0..LEN).map(|i| (i % 251) as u8).collect();
            let start = (device.new_buffer(cpu.id, 0, LEN, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            write(&start, 0, &first);
            drop(start);

            // The first half copied, the rest zero.
            let managed = (new_managed(&device, cpu.id, 0, LEN, LEN / 2))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut expected = first.clone();
            if copied {
                expected[LEN as usize / 2..].fill(0);
            }
            assert!(
                device_bytes(&device, &managed, &into, LEN) == expected,
                "{name}: the bytes at creation (device copy: {copied})"
            );

            // Changed ranges, two of them overlapping; and a change not marked.
            for (at, len) in [(1000, 2000), (2000, 3000), (40_000, 1)] {
                write(&managed, at, &vec![0xA5; len as usize]);
                managed
                    .did_modify_range(at, len)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                expected[at as usize..(at + len) as usize].fill(0xA5);
            }
            write(&managed, 20_000, &[0x5A; 100]);
            if !copied {
                expected[20_000..20_100].fill(0x5A);
            }
            let before = counters.metrics().barriers;
            assert!(
                device_bytes(&device, &managed, &into, LEN) == expected,
                "{name}: the bytes after the changes (device copy: {copied})"
            );
            let named = counters.metrics().barriers - before;
            assert!(
                device_bytes(&device, &managed, &into, LEN) == expected,
                "{name}: the bytes with nothing named (device copy: {copied})"
            );
            // The copy-in adds one barrier, after its copies: the one ending each earlier command
            // buffer on the queue orders it after them.
            assert_eq!(
                named - (counters.metrics().barriers - before - named),
                u64::from(copied),
                "{name}: the barriers a copy-in adds (device copy: {copied})"
            );

            // A change only names its range: nothing is submitted until a commit.
            let before = counters.metrics().submissions;
            for change in 0..10_000 {
                managed
                    .did_modify_range((change * 8) % LEN, 4)
                    .unwrap_or_else(|error| panic!("{name}: change {change}: {error}"));
            }
            assert_eq!(
                counters.metrics().submissions - before,
                0,
                "{name}: changes submit nothing"
            );

            match managed.did_modify_range(LEN - 4, 8) {
                Err(Error::Refused(Refusal::ModifiedRange {
                    start,
                    length: 8,
                    buffer_length: LEN,
                })) if start == LEN - 4 => {}
                other => panic!("{name}: a range past the buffer: {other:?}"),
            }
            let private = (device.take_block(MemoryClass::Private, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            match device.new_managed_buffer((private.id, 0), None, 256, 0) {
                Err(Error::Refused(Refusal::ManagedBlockPrivate { block })) => {
                    assert_eq!(block, private.id, "{name}")
                }
                Err(error) => panic!("{name}: a Managed buffer over a Private block: {error}"),
                Ok(_) => panic!("{name}: a Managed buffer over a Private block was made"),
            }
            println!(
                "managed: {name}: device copy {copied}; metrics {:?}",
                counters.metrics()
            );
            drop((managed, into));
            for block in [cpu, private] {
                (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
            }
        });
    }

    /// The bytes of the whole of Private block `block`, `len` of them, copied by the GPU into
    /// `into`, a Shared buffer as long; `through` is a buffer in the block for the command buffer
    /// to keep.
    fn block_bytes(device: &Device, through: &Buffer, into: &Buffer, len: u64) -> Vec<u8> {
        let queue = device.new_command_queue(1).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let (source, _) = through.device_range();
        let (target, to) = into.device_range();
        command_buffer
            .record_for_test(&[through, into], |device, command| {
                let region = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: to,
                    size: len,
                };
                // SAFETY: the command buffer is recording; both ranges are the buffers' blocks',
                // which it keeps.
                unsafe { device.cmd_copy_buffer(command, source, target, &[region]) };
            })
            .expect("a copy");
        command_buffer.commit().expect("a commit");
        finished("the copy", &command_buffer);
        read(into, len as usize)
    }

    /// The consumer places a Managed buffer's device copy, at any byte offset. Where the device
    /// keeps a copy: placed 4099 bytes into a Private block the consumer filled with `0xA5`,
    /// `gpu_address` is the block's address plus 4099; the copy's first 600 bytes take the
    /// consumer's bytes and the rest of its 1000 zeros, and nothing else of the block changes; a
    /// GPU write into the consumer's slack past the copy (at 5100 to 5200) stays through a later
    /// copy-in of the whole buffer, which lands at 4099 to 5099 alone. Without one the buffer is
    /// its bytes, at their block's address plus their offset. A missing copy, a copy on a unified
    /// device, a copy in a Shared block and a copy past its block are refused.
    #[test]
    fn every_device_places_a_managed_buffer_s_device_copy_where_the_consumer_says() {
        const BLOCK: u64 = 64 << 10;
        const AT: u64 = 4099;
        const LENGTH: u64 = 1000;
        const COPIED: u64 = 600;
        each_device("managed placed", |name, _, device| {
            let cpu = (device.take_block(MemoryClass::Shared, 2 * BLOCK))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let first: Vec<u8> = (0..LENGTH).map(|i| (i * 7 % 251 + 1) as u8).collect();
            let start = (device.new_buffer(cpu.id, 0, LENGTH, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            write(&start, 0, &first);
            drop(start);
            if device.has_unified_memory() {
                let managed = (device.new_managed_buffer((cpu.id, AT), None, LENGTH, COPIED))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                assert_eq!(
                    managed.gpu_address(),
                    cpu.gpu_address + AT,
                    "{name}: one copy"
                );
                let private = (device.take_block(MemoryClass::Private, BLOCK))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                match device.new_managed_buffer((cpu.id, 0), Some((private.id, 0)), 256, 0) {
                    Err(Error::Refused(Refusal::ManagedCopyOnUnified)) => {}
                    other => panic!(
                        "{name}: a copy on a unified device: {:?}",
                        other.map(|_| ())
                    ),
                }
                println!("managed placed: {name}: no device copy");
                drop(managed);
                for block in [cpu, private] {
                    (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
                }
                return;
            }
            let private = (device.take_block(MemoryClass::Private, BLOCK))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let whole = (device.new_buffer(private.id, 0, BLOCK, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let into = (device.new_buffer(cpu.id, BLOCK, BLOCK, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (target, base) = whole.device_range();
            let fill = |range: std::ops::Range<u64>, word: u32| {
                let queue = device.new_command_queue(1).expect("a queue");
                let command_buffer = queue.command_buffer().expect("a command buffer");
                command_buffer
                    .record_for_test(&[&whole], |device, command| {
                        // SAFETY: the command buffer is recording; the range is the block's, which
                        // it keeps, and whole words.
                        unsafe {
                            device.cmd_fill_buffer(
                                command,
                                target,
                                base + range.start,
                                range.end - range.start,
                                word,
                            )
                        }
                    })
                    .expect("a fill");
                command_buffer.commit().expect("a commit");
                finished("the fill", &command_buffer);
            };
            fill(0..BLOCK, 0xA5A5_A5A5);

            let managed =
                (device.new_managed_buffer((cpu.id, 0), Some((private.id, AT)), LENGTH, COPIED))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(
                managed.gpu_address(),
                private.gpu_address + AT,
                "{name}: the copy where the consumer placed it"
            );
            // The consumer's write into its slack, in the commit that takes the first bytes in.
            fill(5100..5200, 0x7777_7777);
            let mut want = vec![0xA5; BLOCK as usize];
            want[AT as usize..(AT + COPIED) as usize].copy_from_slice(&first[..COPIED as usize]);
            want[(AT + COPIED) as usize..(AT + LENGTH) as usize].fill(0);
            want[5100..5200].fill(0x77);
            assert!(
                block_bytes(&device, &whole, &into, BLOCK) == want,
                "{name}: the block after the copy's creation and the slack write"
            );

            // The whole buffer named: the copy-in lands on the copy alone.
            let second: Vec<u8> = (0..LENGTH).map(|i| (i * 13 % 241 + 3) as u8).collect();
            write(&managed, 0, &second);
            (managed.did_modify_range(0, LENGTH)).unwrap_or_else(|error| panic!("{name}: {error}"));
            want[AT as usize..(AT + LENGTH) as usize].copy_from_slice(&second);
            assert!(
                block_bytes(&device, &whole, &into, BLOCK) == want,
                "{name}: the block after a copy-in of the whole buffer"
            );

            let refused = |copy: Option<(BlockId, u64)>| {
                device
                    .new_managed_buffer((cpu.id, 0), copy, 256, 0)
                    .map(|_| ())
            };
            match refused(None) {
                Err(Error::Refused(Refusal::ManagedCopyMissing)) => {}
                other => panic!("{name}: no copy: {other:?}"),
            }
            match refused(Some((cpu.id, BLOCK))) {
                Err(Error::Refused(Refusal::ManagedCopyNotPrivate { block })) => {
                    assert_eq!(block, cpu.id, "{name}")
                }
                other => panic!("{name}: a copy in a Shared block: {other:?}"),
            }
            match refused(Some((private.id, private.len - 255))) {
                Err(Error::Refused(Refusal::BufferPastBlock {
                    offset,
                    length: 256,
                    block_len,
                })) => {
                    assert_eq!(
                        (offset, block_len),
                        (private.len - 255, private.len),
                        "{name}"
                    )
                }
                other => panic!("{name}: a copy past its block: {other:?}"),
            }
            println!("managed placed: {name}: device copy at {AT} of its block");
            drop((managed, whole, into));
            for block in [cpu, private] {
                (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
            }
        });
    }

    /// The add kernel over `managed`, a zeroed buffer and `out`, encoded into `command_buffer`:
    /// `out` takes the bytes the kernel reads of `managed`.
    fn kernel(
        command_buffer: &CommandBuffer,
        state: &ComputePipelineState,
        [managed, zero, out]: [&Buffer; 3],
    ) {
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        encoder.set_compute_pipeline_state(state).expect("a state");
        encoder.set_buffer(Some(managed), 0, 0).expect("a buffer");
        encoder.set_buffer(Some(zero), 0, 1).expect("a buffer");
        encoder.set_buffer(Some(out), 0, 2).expect("a buffer");
        (encoder.dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1])).expect("a dispatch");
        encoder.end_encoding();
    }

    /// Metal's two-copy rule on `device`, its answer to `has_unified_memory` deciding which form
    /// holds. A Managed buffer of [`WORDS`] words, written and named; a kernel takes each word `x`
    /// to `x * 2 + 1`; the whole range is named again with no host write; a second kernel reads
    /// the buffer. With two copies (`has_unified_memory` false) the kernel's write stays in the
    /// device copy, the consumer's bytes keep the host's words, the second name puts them back
    /// over the copy and the reader reads the host's words, and the buffer's address is not its
    /// Shared range's: no such device makes a Managed buffer of one copy. With one copy the
    /// kernel's write is the buffer's bytes, which the reader reads. `alive` runs while the
    /// buffer lives, its kernels done.
    pub(crate) fn two_copy_rule(name: &str, device: &Device, alive: impl FnOnce(&Buffer)) {
        let two = !device.has_unified_memory();
        let len = (WORDS * 4) as u64;
        let mut arena = Arena::new(device);
        let (zero, out) = (arena.buffer(), arena.buffer());
        let cpu = (device.take_block(MemoryClass::Shared, 1))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        // The device copy where the consumer places it: 4 KiB into a Private block it takes.
        let copy = two.then(|| {
            (device.take_block(MemoryClass::Private, len + 4096))
                .unwrap_or_else(|error| panic!("{name}: {error}"))
        });
        let at = copy.as_ref().map(|copy| (copy.id, 4096));
        let managed = (device.new_managed_buffer((cpu.id, 0), at, len, 0))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        match &copy {
            Some(copy) => assert_eq!(
                managed.gpu_address(),
                copy.gpu_address + 4096,
                "{name}: a device that is not unified keeps a device copy where it is placed"
            ),
            None => assert_eq!(managed.gpu_address(), cpu.gpu_address, "{name}: one copy"),
        }
        let host: Vec<u32> = (0..WORDS as u32).map(|i| i * 3 + 7).collect();
        let bytes =
            |words: &[u32]| -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() };
        write(&managed, 0, &bytes(&host));
        (managed.did_modify_range(0, len)).unwrap_or_else(|error| panic!("{name}: {error}"));
        let queue = device.new_command_queue(2).expect("a queue");

        let stepper = step(device);
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        encoder
            .set_compute_pipeline_state(&stepper)
            .expect("a state");
        encoder.set_buffer(Some(&managed), 0, 0).expect("a buffer");
        (encoder.dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1])).expect("a dispatch");
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished("the step", &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
        let stepped: Vec<u32> = host.iter().map(|x| x * 2 + 1).collect();
        assert!(
            read(&managed, len as usize) == bytes(if two { &host } else { &stepped }),
            "{name}: the consumer's bytes after the kernel's write (two copies: {two})"
        );

        // Named with no host write: the consumer's bytes go back over the device copy.
        (managed.did_modify_range(0, len)).unwrap_or_else(|error| panic!("{name}: {error}"));
        let reader = add(device);
        let command_buffer = queue.command_buffer().expect("a command buffer");
        kernel(&command_buffer, &reader, [&managed, &zero, &out]);
        command_buffer.commit().expect("a commit");
        finished("the reader", &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
        assert!(
            read(&out, len as usize) == bytes(if two { &host } else { &stepped }),
            "{name}: the reader reads {} (two copies: {two})",
            if two {
                "the host's words"
            } else {
                "the kernel's write"
            }
        );
        println!(
            "managed two copies: {name}: two copies {two}, the copy's block device-local {:?}",
            copy.as_ref().map(|copy| copy.device_local)
        );
        alive(&managed);
        drop((managed, zero, out, arena));
        for block in [Some(cpu), copy].into_iter().flatten() {
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        }
    }

    #[test]
    fn every_device_keeps_metal_s_two_copy_rule_for_a_managed_buffer() {
        each_device("managed two copies", |name, _, device| {
            two_copy_rule(name, &device, |_| {});
        });
    }

    /// A command buffer reads the bytes the consumer wrote and named before its commit, in either
    /// form (Metal's `didModifyRange:` contract): written and named, a kernel committed after reads
    /// them; written and named again, a kernel committed behind an event wait reads the second
    /// bytes once the event lets it run, though the consumer's buffer and block handles went
    /// before that, as what the commit took holds the consumer's block until its copy ran. Nothing
    /// here writes under a pending command buffer, where Metal leaves the range undefined for the
    /// GPU.
    #[test]
    fn every_device_takes_a_managed_buffer_s_bytes_at_the_commit() {
        each_device("managed at commit", |name, _, device| {
            let copied = !device.core().unified_memory();
            let counters = device.counters();
            let mut arena = Arena::new(&device);
            let (zero, out) = (arena.buffer(), arena.buffer());
            let cpu = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let managed = (new_managed(&device, cpu.id, 0, 256, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let state = add(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let (a, b) = (vec![0xA1; 256], vec![0xB2; 256]);
            let before = counters.metrics();

            write(&managed, 0, &a);
            (managed.did_modify_range(0, 256)).unwrap_or_else(|error| panic!("{name}: {error}"));
            let command_buffer = queue.command_buffer().expect("a command buffer");
            kernel(&command_buffer, &state, [&managed, &zero, &out]);
            command_buffer.commit().expect("a commit");
            finished("the kernel", &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            assert!(
                read(&out, 256) == a,
                "{name}: the bytes named before the commit (device copy: {copied})"
            );

            write(&managed, 0, &b);
            (managed.did_modify_range(0, 256)).unwrap_or_else(|error| panic!("{name}: {error}"));
            let event = device.new_shared_event().expect("an event");
            let held = queue.command_buffer().expect("a command buffer");
            (held.encode_wait_for_event(event.as_event(), 1)).expect("a wait");
            kernel(&held, &state, [&managed, &zero, &out]);
            held.commit().expect("a commit");
            drop(managed);
            (device.release_block(cpu)).unwrap_or_else(|error| panic!("{name}: {error}"));
            event.set_signaled_value(1);
            finished("the held kernel", &held);
            assert_eq!(held.error(), None, "{name}");
            assert!(
                read(&out, 256) == b,
                "{name}: the bytes named before the held commit, the consumer's handles gone \
                 (device copy: {copied})"
            );

            // Each commit took its range once.
            let metrics = counters.metrics();
            let captured = metrics.managed_bytes_captured - before.managed_bytes_captured;
            assert_eq!(captured, if copied { 512 } else { 0 }, "{name}: captured");
            println!("managed at commit: {name}: device copy {copied}; metrics {metrics:?}");
            drop((command_buffer, held, event, queue, zero, out, arena));
        });
    }

    /// The copy-in waits at its capture's number in the commit order, not on the command buffer
    /// whose commit captured it (`crate::capture`): with that one held, waiting on an event of its
    /// own or behind a held command buffer of its queue, a kernel committed after it on another
    /// queue reads the bytes named before, at once, in either form. The capturing command buffer
    /// is still held when the reader completes.
    #[test]
    fn every_device_reads_bytes_named_before_its_commit_while_the_capturing_one_waits() {
        each_device("managed capture held", |name, _, device| {
            let copied = !device.core().unified_memory();
            let mut arena = Arena::new(&device);
            let (zero, out, scratch) = (arena.buffer(), arena.buffer(), arena.buffer());
            let cpu = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let managed = (new_managed(&device, cpu.id, 0, 256, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let state = add(&device);
            let (queue, other) = (
                device.new_command_queue(4).expect("a queue"),
                device.new_command_queue(4).expect("a queue"),
            );
            for (behind, byte) in [(false, 0xB2), (true, 0xC3)] {
                let event = device.new_shared_event().expect("an event");
                // Behind: an older command buffer of the queue, committed before the bytes are
                // named, waits on the event; the capturing one waits on nothing of its own.
                let older = behind.then(|| {
                    let older = queue.command_buffer().expect("a command buffer");
                    (older.encode_wait_for_event(event.as_event(), 1)).expect("a wait");
                    fill(&older, &scratch, 1);
                    older.commit().expect("a commit");
                    older
                });
                let bytes = vec![byte; 256];
                write(&managed, 0, &bytes);
                (managed.did_modify_range(0, 256))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let capturing = queue.command_buffer().expect("a command buffer");
                if !behind {
                    (capturing.encode_wait_for_event(event.as_event(), 1)).expect("a wait");
                }
                fill(&capturing, &scratch, 2);
                capturing.commit().expect("a commit");

                let reader = other.command_buffer().expect("a command buffer");
                kernel(&reader, &state, [&managed, &zero, &out]);
                reader.commit().expect("a commit");
                finished("the reader", &reader);
                assert_eq!(reader.error(), None, "{name}");
                assert_eq!(
                    capturing.status(),
                    CommandBufferStatus::Committed,
                    "{name}: the capturing command buffer is held (behind: {behind})"
                );
                assert!(
                    read(&out, 256) == bytes,
                    "{name}: a later commit of another queue reads the bytes named before it, \
                     the capturing one held (behind: {behind}, device copy: {copied})"
                );
                event.set_signaled_value(1);
                finished("the capturing one", &capturing);
                if let Some(older) = &older {
                    finished("the older one", older);
                }
                drop((older, capturing, reader, event));
            }
            println!("managed capture held: {name}: device copy {copied}");
            drop((managed, state, queue, other, zero, out, scratch, arena));
            (device.release_block(cpu)).unwrap_or_else(|error| panic!("{name}: {error}"));
        });
    }

    /// The copy-in never goes ahead of an older command buffer of its queue (`crate::capture`): one
    /// that writes the bytes shaders reach of the range, unsent behind an event wait, then the
    /// consumer's bytes named, then a reader on the same queue. With two copies the copy-in goes
    /// after the older one's write, and the reader reads the consumer's bytes; with one copy the
    /// older one's write is the buffer's bytes, written after the consumer's, which the reader
    /// reads.
    #[test]
    fn every_device_keeps_a_copy_in_behind_its_queue_s_older_command_buffers() {
        each_device("managed capture order", |name, _, device| {
            let copied = !device.core().unified_memory();
            let mut arena = Arena::new(&device);
            let (zero, out) = (arena.buffer(), arena.buffer());
            let cpu = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let managed = (new_managed(&device, cpu.id, 0, 256, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let state = add(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let event = device.new_shared_event().expect("an event");

            let writer = queue.command_buffer().expect("a command buffer");
            (writer.encode_wait_for_event(event.as_event(), 1)).expect("a wait");
            fill(&writer, &managed, 0x7777_7777);
            writer.commit().expect("a commit");
            let bytes = vec![0xD4; 256];
            write(&managed, 0, &bytes);
            (managed.did_modify_range(0, 256)).unwrap_or_else(|error| panic!("{name}: {error}"));
            let reader = queue.command_buffer().expect("a command buffer");
            kernel(&reader, &state, [&managed, &zero, &out]);
            reader.commit().expect("a commit");
            event.set_signaled_value(1);
            finished("the writer", &writer);
            finished("the reader", &reader);
            assert_eq!((writer.error(), reader.error()), (None, None), "{name}");
            let expected = if copied { bytes } else { vec![0x77; 256] };
            assert!(
                read(&out, 256) == expected,
                "{name}: the reader reads {} (device copy: {copied})",
                if copied {
                    "the consumer's bytes, copied in after the older write"
                } else {
                    "the older write, made after the consumer's"
                }
            );
            println!("managed capture order: {name}: device copy {copied}");
            drop((
                writer, reader, event, queue, managed, state, zero, out, arena,
            ));
            (device.release_block(cpu)).unwrap_or_else(|error| panic!("{name}: {error}"));
        });
    }

    /// Records into `command_buffer` a copy of `at..at + len` of the bytes shaders reach at
    /// `managed`'s address into the same range of `into`, a Shared buffer.
    fn copy_out(
        command_buffer: &CommandBuffer,
        managed: &Buffer,
        into: &Buffer,
        at: u64,
        len: u64,
    ) {
        let (source, from) = managed.device_range();
        let (target, to) = into.device_range();
        command_buffer
            .record_for_test(&[managed, into], |device, command| {
                let region = vk::BufferCopy {
                    src_offset: from + at,
                    dst_offset: to + at,
                    size: len,
                };
                // SAFETY: the command buffer is recording; both ranges are the buffers', which it
                // keeps.
                unsafe { device.cmd_copy_buffer(command, source, target, &[region]) };
            })
            .expect("a copy");
    }

    /// A Managed buffer of `len` bytes with a device copy, its bytes `bytes` and named whole, and a
    /// Shared buffer as long to copy its device copy into, with their blocks.
    fn named_whole(
        device: &Device,
        name: &str,
        len: u64,
        bytes: &[u8],
    ) -> (Buffer, Buffer, [Block; 2]) {
        let cpu = (device.take_block(MemoryClass::Shared, len))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let side = (device.take_block(MemoryClass::Shared, len))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let into = (device.new_buffer(side.id, 0, len, None))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let managed = (new_managed(device, cpu.id, 0, len, 0))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        write(&managed, 0, bytes);
        (managed.did_modify_range(0, len)).unwrap_or_else(|error| panic!("{name}: {error}"));
        (managed, into, [cpu, side])
    }

    /// A range of three 16 MiB units named arrives whole at one commit, copied on the GPU from the
    /// consumer's block: the commit takes no staging and allocates nothing. It prints the commit's
    /// time on the host.
    #[test]
    fn every_device_copies_a_long_named_range_with_no_staging() {
        each_device("managed long range", |name, _, device| {
            if device.core().unified_memory() {
                println!("managed long range: {name}: no device copy");
                return;
            }
            let counters = device.counters();
            device.core().pool.await_internals();
            let len = 3 * MemoryClass::Shared.block_len();
            let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            let (managed, into, blocks) = named_whole(&device, name, len, &bytes);
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            copy_out(&command_buffer, &managed, &into, 0, len);
            let before = (counters.metrics(), counters.gauges().staging_bytes);
            let began = Instant::now();
            command_buffer.commit().expect("a commit");
            let commit = began.elapsed();
            let after = (counters.metrics(), counters.gauges().staging_bytes);
            finished("the copy", &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            assert!(
                read(&into, len as usize) == bytes,
                "{name}: the named bytes"
            );
            assert_eq!(
                after.0.managed_bytes_captured - before.0.managed_bytes_captured,
                len,
                "{name}: captured"
            );
            assert_eq!(
                (after.0.consumer_allocations, after.1),
                (before.0.consumer_allocations, before.1),
                "{name}: the commit takes no staging and allocates nothing"
            );
            println!(
                "managed long range: {name}: {} MiB named, the commit {:.3} ms on the host",
                len >> 20,
                commit.as_secs_f64() * 1e3
            );
            drop((command_buffer, queue, managed, into));
            for block in blocks {
                (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
            }
        });
    }

    /// A Managed buffer over an imported dma-buf, a block the runtime does not map: the buffer
    /// has no `contents`, and the consumer writes the pages through its own mapping (here the
    /// memfd behind the dma-buf). Written and named, a kernel committed after reads the bytes,
    /// with a device copy or without; written and named again, the next one reads the new bytes.
    /// A device that imports no such dma-buf is passed over, as the import tests check its
    /// refusal.
    #[cfg(target_os = "linux")]
    #[test]
    fn every_device_makes_a_managed_buffer_over_an_imported_dma_buf() {
        use crate::tests::import::on_device::{memfd, udmabuf};
        use crate::{ImportHandle, Limit};
        use std::os::unix::fs::FileExt;
        each_device("managed over a dma-buf", |name, _, device| {
            const LEN: u64 = 64 << 10;
            let copied = !device.core().unified_memory();
            let memfd = memfd(LEN);
            let Some(fd) = udmabuf(&memfd, 0, LEN) else {
                println!("managed over a dma-buf: {name}: no /dev/udmabuf here; not run");
                return;
            };
            let block = match device.import_dma_buf(fd, LEN, None) {
                Ok((block, _)) => block,
                Err(
                    error @ Error::Limit(
                        Limit::NoImport {
                            handle: ImportHandle::DmaBuf,
                        }
                        | Limit::ImportType {
                            handle: ImportHandle::DmaBuf,
                            ..
                        },
                    ),
                ) => {
                    println!(
                        "managed over a dma-buf: {name}: not imported here ({error}); not run"
                    );
                    return;
                }
                Err(error) => panic!("{name}: the import: {error}"),
            };
            assert_eq!(
                (block.class, block.contents),
                (MemoryClass::Shared, None),
                "{name}: an imported dma-buf's block, Shared and not mapped by the runtime"
            );
            let mut arena = Arena::new(&device);
            let (zero, out) = (arena.buffer(), arena.buffer());
            let managed = (new_managed(&device, block.id, 0, 256, 0)).unwrap_or_else(|error| {
                panic!("{name}: a Managed buffer over the import: {error}")
            });
            assert!(
                managed.contents().is_none(),
                "{name}: no mapping of the runtime's"
            );
            let state = add(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let pages = std::fs::File::from(memfd);
            for word in [0xA1u8, 0xB2] {
                let bytes = vec![word; 256];
                (pages.write_all_at(&bytes, 0)).expect("the consumer's write");
                (managed.did_modify_range(0, 256))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let command_buffer = queue.command_buffer().expect("a command buffer");
                kernel(&command_buffer, &state, [&managed, &zero, &out]);
                command_buffer.commit().expect("a commit");
                finished("the kernel", &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
                assert!(
                    read(&out, 256) == bytes,
                    "{name}: the bytes written through the consumer's mapping, {word:#x} \
                     (device copy: {copied})"
                );
            }
            println!("managed over a dma-buf: {name}: read twice (device copy: {copied})");
            drop((managed, queue, zero, out, arena));
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        });
    }

    fn whole(side: u32) -> Region {
        Region {
            origin: [0; 3],
            size: [side, side, 1],
        }
    }

    /// The square RGBA8 level `level` of a texture of `SIDE`, as it lies in `named`: rows `PITCH`
    /// apart at the level's place, packed.
    fn level_bytes(named: &[u8], places: &[BytesPlace], level: u32) -> Vec<u8> {
        let side = (SIDE >> level) as usize;
        let place = places[level as usize];
        (0..side)
            .flat_map(|row| {
                let at = (place.offset + row as u64 * place.bytes_per_row) as usize;
                named[at..at + side * 4].iter().copied()
            })
            .collect()
    }

    /// Writes `value` over `region` of the RGBA8 level at `place` in `named`.
    fn fill_region(named: &mut [u8], place: BytesPlace, region: Region, value: u8) {
        let [x, y, _] = region.origin.map(u64::from);
        let [width, height, _] = region.size.map(u64::from);
        for row in y..y + height {
            let at = (place.offset + row * place.bytes_per_row + x * 4) as usize;
            named[at..at + width as usize * 4].fill(value);
        }
    }

    const SIDE: u32 = 64;
    const PITCH: u64 = 256;
    const NAMED: u64 = 24_576;

    #[test]
    fn every_device_copies_a_managed_texture_s_named_regions() {
        each_device("managed texture", |name, _, device| {
            let cpu = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            // Level 0's rows packed at 0, level 1's 128-byte rows 256 apart after it.
            let places = [
                BytesPlace {
                    offset: 0,
                    bytes_per_row: PITCH,
                    bytes_per_image: 0,
                },
                BytesPlace {
                    offset: 16_384,
                    bytes_per_row: PITCH,
                    bytes_per_image: 0,
                },
            ];
            let named = (device.new_buffer(cpu.id, 4096, NAMED, Some(0)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut now: Vec<u8> = (0..NAMED).map(|i| (i % 251) as u8).collect();
            write(&named, 0, &now);
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, SIDE, SIDE);
            d.mipmap_level_count = 2;
            d.usage = TextureUsage::RENDER_TARGET;
            let texture = (device.new_managed_texture(&d, None, &named, &places))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(texture.descriptor().storage_mode, StorageMode::Managed);
            let image = |level: u32| {
                let side = SIDE >> level;
                let row = u64::from(side) * 4;
                let mut bytes = vec![0; (row * u64::from(side)) as usize];
                (texture.image_bytes(&mut bytes, row, whole(side), level, 0))
                    .unwrap_or_else(|error| panic!("{name}: level {level}: {error}"));
                bytes
            };

            // Both levels uploaded whole.
            for level in 0..2 {
                (texture.upload_region(whole(SIDE >> level), level, 0))
                    .unwrap_or_else(|error| panic!("{name}: level {level}: {error}"));
            }
            for level in 0..2 {
                assert!(
                    image(level) == level_bytes(&now, &places, level),
                    "{name}: level {level} after its upload"
                );
            }

            // A region changed and uploaded; a change outside it, not uploaded.
            let mut seen = now.clone();
            let region = Region {
                origin: [8, 4, 0],
                size: [16, 8, 1],
            };
            fill_region(&mut now, places[0], region, 0xA5);
            fill_region(&mut seen, places[0], region, 0xA5);
            let outside = Region {
                origin: [0, 40, 0],
                size: [SIDE, 1, 1],
            };
            fill_region(&mut now, places[0], outside, 0x5A);
            write(&named, 0, &now);
            (texture.upload_region(region, 0, 0)).unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(
                image(0) == level_bytes(&seen, &places, 0),
                "{name}: level 0 after the region's upload"
            );
            // get_bytes reads the named bytes, the change not uploaded too.
            let mut read = vec![0; 16_384];
            (texture.get_bytes(&mut read, PITCH, 16_384, whole(SIDE), 0, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(
                read == level_bytes(&now, &places, 0),
                "{name}: get_bytes reads the named bytes"
            );

            // replace_region writes the named bytes, and the image takes them.
            let small = Region {
                origin: [4, 4, 0],
                size: [4, 4, 1],
            };
            (texture.replace_region(small, 1, 0, crate::HostBytes::from(&[0x11; 64][..]), 16, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            fill_region(&mut now, places[1], small, 0x11);
            fill_region(&mut seen, places[1], small, 0x11);
            let contents = named.contents().expect("Shared");
            // SAFETY: the buffer's bytes, which nothing writes meanwhile.
            let held = unsafe { std::slice::from_raw_parts(contents.as_ptr(), NAMED as usize) };
            assert!(
                held == now.as_slice(),
                "{name}: replace_region wrote the named bytes"
            );
            assert!(
                image(1) == level_bytes(&seen, &places, 1),
                "{name}: level 1 after replace_region"
            );

            // Places a copy does not read, and a texture without named bytes.
            let refused = |places: &[BytesPlace], bytes: &Buffer, what: &str| -> (u32, u32) {
                match device.new_managed_texture(&d, None, bytes, places) {
                    Err(Error::Refused(Refusal::ManagedTextureBytes { level, slice })) => {
                        (level, slice)
                    }
                    Err(error) => panic!("{name}: {what}: {error}"),
                    Ok(_) => panic!("{name}: {what}: made"),
                }
            };
            let with = |offset: u64, bytes_per_row: u64| {
                let mut wrong = places;
                wrong[1] = BytesPlace {
                    offset,
                    bytes_per_row,
                    bytes_per_image: 0,
                };
                wrong
            };
            assert_eq!(refused(&places[..1], &named, "one place"), (2, 0));
            assert_eq!(refused(&with(16_386, PITCH)[..], &named, "offset"), (1, 0));
            assert_eq!(refused(&with(16_384, 130)[..], &named, "pitch"), (1, 0));
            assert_eq!(refused(&with(16_384, 64)[..], &named, "short rows"), (1, 0));
            assert_eq!(
                refused(&with(NAMED - 4096, PITCH)[..], &named, "past"),
                (1, 0)
            );
            let private = (device.take_block(MemoryClass::Private, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let unmapped = (device.new_buffer(private.id, 0, NAMED, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(refused(&places, &unmapped, "Private"), (0, 0));
            d.storage_mode = StorageMode::Managed;
            let plain =
                (device.new_texture(&d, None)).unwrap_or_else(|error| panic!("{name}: {error}"));
            match plain.upload_region(whole(SIDE), 0, 0) {
                Err(Error::Refused(Refusal::ManagedTextureBytes { level: 0, slice: 0 })) => {}
                Err(error) => panic!("{name}: an upload without named bytes: {error}"),
                Ok(()) => panic!("{name}: an upload without named bytes went"),
            }
            match texture.upload_region(whole(1), 2, 0) {
                Err(Error::Refused(Refusal::TextureRegion { level: 2, slice: 0 })) => {}
                Err(error) => panic!("{name}: a level past the texture: {error}"),
                Ok(()) => panic!("{name}: a level past the texture went"),
            }
            drop((texture, plain, named, unmapped));
            for block in [cpu, private] {
                device
                    .release_block(block)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            }
        });
    }

    /// The upload of a Managed texture's region keeps the bytes it reads until it retires: with
    /// the texture, its named buffer and the block's handle gone right after the call, and the
    /// upload not yet submitted, the named buffer's block stays out of the pool until the upload
    /// has run. Named bytes lie in Shared memory with no device copy, on every device.
    #[test]
    fn every_device_keeps_the_bytes_a_managed_texture_s_upload_reads_until_it_retires() {
        each_device("managed texture upload hold", |name, _, device| {
            let cpu = (device.take_block(MemoryClass::Shared, NAMED))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let read = cpu.id;
            let named = (device.new_buffer(cpu.id, 0, NAMED, Some(0)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let places = [BytesPlace {
                offset: 0,
                bytes_per_row: PITCH,
                bytes_per_image: 0,
            }];
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, SIDE, SIDE);
            d.usage = TextureUsage::RENDER_TARGET;
            let texture = (device.new_managed_texture(&d, None, &named, &places))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            (texture.upload_region(whole(SIDE), 0, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            drop((texture, named, cpu));
            let core = device.core();
            assert!(
                !core.pool.holds(read),
                "{name}: the bytes the pending upload reads went back to the pool"
            );
            let value = core.work.flush();
            (core.done.wait(core, value)).unwrap_or_else(|error| panic!("{name}: {error}"));
            within("the read bytes' return", || core.pool.holds(read));
        });
    }
}
