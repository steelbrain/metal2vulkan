//! Managed buffers: changed ranges merge where they overlap or touch; and, under a device cfg, on
//! every device of the cfg's kind, in the form Vulkan's device type picks: with a device copy
//! where the device is neither integrated nor a CPU, the copy holds the consumer's first bytes and
//! zeros after them, a changed range reaches it at the next commit and a range not marked changed
//! does not, a change submits nothing, and the copy-in at a commit adds one barrier; without one,
//! shaders read the consumer's bytes as they are. In both forms a command buffer reads the bytes
//! the consumer wrote and named before its commit (Metal's `didModifyRange:` contract), though the
//! consumer's buffer and block handles go before it runs; nothing writes under a pending command
//! buffer, where Metal leaves the range undefined for the GPU. A commit with nothing for the GPU (no
//! commands, or one dispatch of a kernel without effects) takes the ranges named before it: with a
//! device copy its one submission is the copy-in, it completes after it, and the next commit
//! captures nothing. A range of three 16 MiB units named
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
//!
//! A texture over a Managed buffer in an image of its own, on every device: a command buffer of
//! another queue that runs ahead of the first of its own queue's to read it, that one in a group
//! held or held for room, reads the buffer's bytes, as does that one, and so over a Shared buffer
//! with that one in a group held; and a command buffer
//! committed before a capture of a range over it, in a group held or held for room, reads the
//! buffer's bytes outside that range, and the one that takes the capture the new bytes. A reader
//! of the buffer and a write of the texture on one queue run in the order they are committed,
//! whichever was encoded first, and a reader of another queue encoded before the write and
//! committed once it completed reads what it wrote.

use crate::capture::{merge, taker, Captured};
use crate::submit::put_ahead;

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

/// A copy-in waits at its capture's number in the commit order: a command buffer committed before
/// it never takes it, and the first submitted at or after it does, ahead of its own commands,
/// whichever queue it is of and wherever the capturing one waits, so it waits for that one, which
/// takes a submission for it with nothing for the GPU; in a group it lands behind the members
/// before its taker and ahead of the taker (`crate::capture`, `crate::submit`).
#[test]
fn a_copy_in_goes_ahead_of_the_first_command_buffer_committed_at_or_after_its_capture() {
    assert_eq!(
        taker(5, &[3, 4]),
        None,
        "only older command buffers go first"
    );
    assert_eq!(taker(5, &[]), None, "nothing goes first");
    assert_eq!(
        taker(5, &[3, 5, 7]),
        Some(1),
        "the capturing one, after an older one"
    );
    assert_eq!(taker(5, &[6, 8]), Some(0), "a later one, its capturer held");

    let mut captured = Captured::default();
    captured.push(5, "five");
    captured.push(9, "nine");
    // A submission of older command buffers, of the capturer's queue or another, takes nothing.
    assert!(captured.take(&[2, 4]).is_empty(), "older ones leave both");
    // One that holds an older command buffer and a later one: the first copy-in goes ahead of the
    // later one, after the older; the second stays.
    assert_eq!(
        captured.take(&[4, 6]),
        [(1, "five")],
        "ahead of the later one"
    );
    // The second's capturer held: a later command buffer of another queue takes it.
    assert!(
        captured.take(&[8]).is_empty(),
        "committed before the second"
    );
    assert_eq!(
        captured.take(&[10, 11]),
        [(0, "nine")],
        "the first at or after it"
    );
    assert!(captured.take(&[12]).is_empty(), "each goes once");
    // A command buffer with nothing for the GPU takes a submission only where one waits for it.
    assert!(!captured.waits_for(12), "none waits");
    captured.push(13, "thirteen");
    assert!(!captured.waits_for(12), "not for an older one");
    assert!(captured.waits_for(13), "for its capturer");
    assert!(captured.waits_for(15), "for a later one, its capturer held");
    assert_eq!(captured.take(&[13]), [(0, "thirteen")]);
    // Two pending go ahead of the same command buffer, in their order.
    captured.push(13, "thirteen");
    captured.push(14, "fourteen");
    assert_eq!(
        captured.take(&[12, 15]),
        [(1, "thirteen"), (1, "fourteen")],
        "both ahead of the later one, in commit order"
    );

    // A group of three, the capturer in the middle: its copy-in lands after the first member's
    // segments and ahead of the capturer's first; the last member begins where it did, one on.
    let mut captured = Captured::default();
    captured.push(5, "copy-in 5");
    let mut segments = vec!["3a", "3b", "5a", "7a"];
    let mut starts = vec![0, 2, 3];
    let taken = captured.take(&[3, 5, 7]);
    let kept = put_ahead(&mut segments, &mut starts, taken, |copy_in| *copy_in);
    assert_eq!(kept, ["copy-in 5"]);
    assert_eq!(
        segments,
        ["3a", "3b", "copy-in 5", "5a", "7a"],
        "after 3, ahead of 5"
    );
    assert_eq!(starts, [0, 2, 4], "5 begins with its copy-in, 7 one on");

    // Two captures in one group: each ahead of its own taker, the group's order kept.
    captured.push(4, "copy-in 4");
    captured.push(6, "copy-in 6");
    let mut segments = vec!["3a", "3b", "5a", "7a"];
    let mut starts = vec![0, 2, 3];
    let taken = captured.take(&[3, 5, 7]);
    let kept = put_ahead(&mut segments, &mut starts, taken, |copy_in| *copy_in);
    assert_eq!(kept, ["copy-in 4", "copy-in 6"]);
    assert_eq!(
        segments,
        ["3a", "3b", "copy-in 4", "5a", "copy-in 6", "7a"],
        "4 ahead of 5, 6 ahead of 7"
    );
    assert_eq!(starts, [0, 2, 4]);
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(crate) mod on_device {
    use crate::buffer::BufferObject;
    use crate::submit::UNSEEN_JOINS;
    use crate::tests::command::on_device::{
        behind_busy, each_device, fill, finished, two_queues, within, Arena,
    };
    use crate::tests::compute::on_device::{add, quiet_step, step, WORDS};
    use crate::texture::TextureObject;
    use crate::{
        Block, BlockId, Buffer, BufferLayout, BytesPlace, CommandBuffer, CommandBufferStatus,
        ComputePipelineState, Device, DispatchType, Error, IndexClass, MemoryClass, PixelFormat,
        Refusal, Region, StorageMode, Texture, TextureDescriptor, TexturePlace, TextureType,
        TextureUsage,
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
                    cpu.gpu_address.expect("a taken block's address") + AT,
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
                private.gpu_address.expect("a taken block's address") + AT,
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
                copy.gpu_address.expect("a taken block's address") + 4096,
                "{name}: a device that is not unified keeps a device copy where it is placed"
            ),
            None => assert_eq!(
                Some(managed.gpu_address()),
                cpu.gpu_address,
                "{name}: one copy"
            ),
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

    /// A commit takes the ranges named since the last, with commands or none (`crate::capture`):
    /// a Managed buffer written and named, then a command buffer that `encode` leaves with nothing
    /// for the GPU, committed and waited for. With a device copy, its commit captured the range,
    /// its one submission was the copy-in, and it completed after it, its GPU start and end one
    /// time as for any command buffer with nothing for the GPU; the next commit captures nothing
    /// and reads the bytes in the device copy. With one copy nothing is captured or submitted.
    /// A discrete Mac GPU (an AMD Radeon Pro 5300M) takes a named range at the first commit whose
    /// encoder binds its buffer: the same commit where the command buffer binds it and records
    /// nothing, and a later one than here where it binds nothing.
    fn empty_commit_takes_named_ranges(
        name: &str,
        device: &Device,
        encode: impl FnOnce(&CommandBuffer, &Buffer),
    ) {
        let copied = !device.has_unified_memory();
        let counters = device.counters();
        let take = || {
            (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"))
        };
        let (cpu, side) = (take(), take());
        let into = (device.new_buffer(side.id, 0, 256, None))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let managed = (new_managed(device, cpu.id, 0, 256, 0))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        // The creation's upload goes first, with a commit of its own.
        device_bytes(device, &managed, &into, 256);
        let bytes = vec![0xE7; 256];
        write(&managed, 0, &bytes);
        (managed.did_modify_range(0, 256)).unwrap_or_else(|error| panic!("{name}: {error}"));
        let queue = device.new_command_queue(4).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        encode(&command_buffer, &managed);
        let before = counters.metrics();
        command_buffer.commit().expect("a commit");
        let committed = counters.metrics();
        finished("the empty one", &command_buffer);
        let after = counters.metrics();
        let captured = committed.managed_bytes_captured - before.managed_bytes_captured;
        let one = u64::from(copied);
        assert_eq!(
            (
                captured,
                after.submissions - before.submissions,
                after.runtime_command_buffers - before.runtime_command_buffers,
            ),
            (256 * one, one, one),
            "{name}: captured at its commit, its submissions and the runtime's command buffers \
             in them (device copy: {copied})"
        );
        assert_eq!(
            (command_buffer.status(), command_buffer.error()),
            (CommandBufferStatus::Completed, None),
            "{name}"
        );
        let (start, end) = (
            command_buffer.gpu_start_time(),
            command_buffer.gpu_end_time(),
        );
        assert!(
            start == end && start > 0,
            "{name}: GPU times {start}, {end}"
        );
        let read = device_bytes(device, &managed, &into, 256);
        let next = counters.metrics().managed_bytes_captured - after.managed_bytes_captured;
        assert_eq!(next, 0, "{name}: the next commit captures nothing");
        assert!(
            read == bytes,
            "{name}: the bytes shaders reach (device copy: {copied})"
        );
        println!("managed empty commit: {name}: device copy {copied}, {captured} bytes captured");
        drop((command_buffer, queue, managed, into));
        for block in [cpu, side] {
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        }
    }

    #[test]
    fn every_device_takes_named_ranges_at_a_commit_with_no_commands() {
        each_device("managed empty commit", |name, _, device| {
            empty_commit_takes_named_ranges(name, &device, |_, _| {});
        });
    }

    /// [`empty_commit_takes_named_ranges`] with a command buffer whose one dispatch, over the
    /// Managed buffer, is of a kernel that writes no memory, and so records nothing.
    #[test]
    fn every_device_takes_named_ranges_at_a_commit_whose_dispatch_has_no_effects() {
        each_device("managed quiet commit", |name, _, device| {
            let quiet = quiet_step(&device);
            let before = device.counters().metrics().dispatches_without_effects;
            empty_commit_takes_named_ranges(name, &device, |command_buffer, managed| {
                let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&quiet).expect("a state");
                encoder.set_buffer(Some(managed), 0, 0).expect("a buffer");
                (encoder.dispatch_threadgroups([1, 1, 1], [16, 1, 1])).expect("a dispatch");
                encoder.end_encoding();
            });
            let elided = device.counters().metrics().dispatches_without_effects - before;
            assert_eq!(elided, 1, "{name}: the dispatch recorded nothing");
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
            // A colour render target takes an index.
            (device.grant_indices(IndexClass::Texture, 4096, 1)).expect("a grant");
            let texture = (device.new_managed_texture(&d, Some(4096), &named, &places))
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
                // Refused before the index is placed.
                match device.new_managed_texture(&d, Some(4097), bytes, places) {
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
            let plain = (device.new_texture_for_views(&d))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
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
    /// upload not yet submitted, the block's release sends the upload, and the block goes back to
    /// the pool once it has run, with no later call. Named bytes lie in Shared memory with no
    /// device copy, on every device.
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
            // A colour render target takes an index.
            (device.grant_indices(IndexClass::Texture, 4096, 1)).expect("a grant");
            let texture = (device.new_managed_texture(&d, Some(4096), &named, &places))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            (texture.upload_region(whole(SIDE), 0, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            drop((texture, named, cpu));
            let core = device.core();
            within("the read bytes' return with no later call", || {
                core.pool.holds(read)
            });
        });
    }

    /// The texels of the texture over a Managed buffer the tests below read, RGBA8Unorm.
    const OVER_TEXELS: [u32; 2] = [16, 4];
    const OVER_ROW: u64 = 64;
    /// A texel past the row: no device lays a linear image's rows there, so the texture is an
    /// image of its own.
    const OVER_PITCH: u64 = OVER_ROW + 4;
    const OVER_LEN: u64 = 512;

    /// A buffer of [`OVER_LEN`] bytes in Shared block `cpu`, its bytes written, and a texture over
    /// it at [`OVER_PITCH`] made after. Where `managed`, a Managed buffer whose bytes are named and
    /// taken by a commit that has run before the texture is made, so the texture's first copy in
    /// is owed by its making alone: with two copies (`Device::has_unified_memory` false) its
    /// buffer's entry owes it (`crate::over::ManagedOvers`); with one, as over a Shared buffer
    /// (else), the Shared rules copy it in for each submission (`crate::over`). Returns the
    /// buffer, its bytes, the texture and its form printed.
    fn over_texture(
        name: &str,
        device: &Device,
        cpu: BlockId,
        managed: bool,
    ) -> (Buffer, Vec<u8>, Texture, String) {
        let buffer = match managed {
            true => new_managed(device, cpu, 0, OVER_LEN, 0),
            false => device.new_buffer(cpu, 0, OVER_LEN, None),
        }
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let bytes: Vec<u8> = (0..OVER_LEN).map(|i| (i % 251) as u8 + 1).collect();
        write(&buffer, 0, &bytes);
        if managed {
            (buffer.did_modify_range(0, OVER_LEN))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let queue = device.new_command_queue(1).expect("a queue");
            let capture = queue.command_buffer().expect("a command buffer");
            capture.commit().expect("a commit");
            finished(name, &capture);
        }
        let first = IndexClass::Texture.range().start;
        (device.grant_indices(IndexClass::Texture, first, 1)).expect("a grant");
        let mut d = TextureDescriptor::new(
            TextureType::Type2D,
            PixelFormat::RGBA8Unorm,
            OVER_TEXELS[0],
            OVER_TEXELS[1],
        );
        d.usage = TextureUsage::SHADER_READ;
        let texture = (buffer.new_texture(&d, Some(first), 0, OVER_PITCH))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let (two, own, entry) = (
            !device.has_unified_memory(),
            texture.copied_over().is_some(),
            texture.managed_over().is_some(),
        );
        assert!(own, "{name}: an image of its own");
        assert_eq!(
            entry,
            managed && two,
            "{name}: owed by its buffer's entry over a Managed buffer with two copies"
        );
        let kind = if managed { "Managed" } else { "Shared" };
        let form = format!(
            "over a {kind} buffer, two copies {two}, an image of its own {own}, owed by its \
             buffer's entry {entry}"
        );
        (buffer, bytes, texture, form)
    }

    /// Encodes a copy of the texture's texels into `into`, packed.
    fn read_over(command_buffer: &CommandBuffer, texture: &Texture, into: &Buffer) {
        let mut blit = command_buffer.blit_command_encoder().expect("an encoder");
        let at = TexturePlace {
            slice: 0,
            level: 0,
            origin: [0; 3],
        };
        let layout = BufferLayout {
            offset: 0,
            bytes_per_row: OVER_ROW,
            bytes_per_image: 0,
        };
        let size = [OVER_TEXELS[0], OVER_TEXELS[1], 1];
        (blit.copy_from_texture_to_buffer(texture, at, size, into, layout)).expect("a copy");
        blit.end_encoding();
    }

    /// The texture's texels as they lie in `bytes`, packed.
    fn over_rows(bytes: &[u8]) -> Vec<u8> {
        (0..u64::from(OVER_TEXELS[1]))
            .flat_map(|row| {
                let at = (row * OVER_PITCH) as usize;
                bytes[at..at + OVER_ROW as usize].iter().copied()
            })
            .collect()
    }

    /// A texture over a buffer in an image of its own ([`over_texture`], over a Managed buffer
    /// where `managed`, else a Shared one) is read by two queues behind the busy gate
    /// ([`two_queues`]): by A's command buffer numbered `reader`, the first of A's to read it, and
    /// by B1, committed after it and run before it. Both read the buffer's bytes: Metal's texture
    /// over a buffer shares the buffer's storage, and a command buffer committed after a
    /// `didModifyRange:` and a commit reads what it named (over a Shared buffer, what the host
    /// wrote before the commit). A3 is in A's
    /// group held when B1 is submitted, A4 held for room in no group. The form, A's submissions
    /// in flight and group held as B1 commits, and the order they ran in are printed.
    fn over_read_from_two_queues(test: &str, reader: u32, managed: bool) {
        each_device(test, |name, _, device| {
            let cpu = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (buffer, bytes, texture, form) = over_texture(name, &device, cpu.id, managed);
            let mut arena = Arena::new(&device);
            let (a_read, b_read) = (arena.buffer(), arena.buffer());
            let encode =
                |number: u32, command_buffer: &CommandBuffer, after: bool| match (number, after) {
                    (6, false) => read_over(command_buffer, &texture, &b_read),
                    (number, false) if number == reader => {
                        read_over(command_buffer, &texture, &a_read)
                    }
                    _ => {}
                };
            let mut held = None;
            let log = two_queues(name, &device, (true, 2), encode, |_, a_work| {
                held = Some(device.core().work.in_flight(a_work[0].queue()));
            });
            let rows = over_rows(&bytes);
            let (a_bytes, b_bytes) = (read(&a_read, rows.len()), read(&b_read, rows.len()));
            println!(
                "{test}: {name}: {form}; A's submissions in flight and group held as B1 commits \
                 {held:?}; log {log:?}; A{reader} read the bytes {}, B1 {} (its first texels \
                 {:?}, the bytes' {:?})",
                a_bytes == rows,
                b_bytes == rows,
                &b_bytes[..8],
                &rows[..8]
            );
            assert_eq!(
                held,
                Some((2, true)),
                "{name}: A's group held as B1 commits"
            );
            assert_eq!(log, [0, 1, 6, 3, 4, 2], "{name}: B1 ran ahead of A3 to A5");
            assert!(
                a_bytes == rows,
                "{name}: A{reader} reads the buffer's bytes"
            );
            assert!(b_bytes == rows, "{name}: B1 reads the buffer's bytes");
            drop((texture, buffer, a_read, b_read, arena));
            (device.release_block(cpu)).unwrap_or_else(|error| panic!("{name}: {error}"));
        });
    }

    /// [`over_read_from_two_queues`] with A3, in A's group held when B1 passes it, the first of
    /// A's to read the texture.
    #[test]
    fn every_device_reads_a_managed_texture_over_on_a_queue_passing_a_group_that_reads_it() {
        over_read_from_two_queues("managed over passing a group", 3, true);
    }

    /// [`over_read_from_two_queues`] with A3, in A's group held when B1 passes it, over a Shared
    /// buffer: each submission copies the texture in for its own commands (`crate::over`).
    #[test]
    fn every_device_reads_a_shared_texture_over_on_a_queue_passing_a_group_that_reads_it() {
        over_read_from_two_queues("shared over passing a group", 3, false);
    }

    /// [`over_read_from_two_queues`] with A4, held for room in no group when B1 passes it, the
    /// first of A's to read the texture.
    #[test]
    fn every_device_reads_a_managed_texture_over_on_a_queue_passing_one_held_for_room() {
        over_read_from_two_queues("managed over passing one held for room", 4, true);
    }

    /// A copy in a texture over a Managed buffer in an image of its own owes ([`over_texture`])
    /// stays owed to a command buffer committed before a capture that copies the texture in, as
    /// that capture's copy in runs ahead of a later command buffer only (`crate::capture`). Behind
    /// the busy gate ([`behind_busy`]), R reads the texture; then the consumer writes and names
    /// the texture's last row, and C, committed after on the same queue, takes the capture and
    /// reads the texture. R runs first and reads the buffer's bytes in the other rows (Metal
    /// leaves the named row undefined for it, `MTLBuffer.h` `didModifyRange:`); C reads the new
    /// bytes in every row. With `held`, [`UNSEEN_JOINS`] command buffers join the queue's group
    /// held first, so R is held for room in no group when C commits; else R is in the group held.
    fn over_read_before_a_capture(test: &str, held: bool) {
        each_device(test, |name, _, device| {
            let counters = device.counters();
            let cpu = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (managed, mut bytes, texture, form) = over_texture(name, &device, cpu.id, true);
            let before = over_rows(&bytes);
            let mut arena = Arena::new(&device);
            let (r_read, c_read) = (arena.buffer(), arena.buffer());
            let queue = device.new_command_queue(16).expect("a queue");
            let mut kept = Vec::new();
            let mut counted = 0;
            behind_busy(name, &device, &queue, |_, _, _| {
                for _ in 0..UNSEEN_JOINS * usize::from(held) {
                    let joins = queue.command_buffer().expect("a command buffer");
                    (joins.record_for_test(&[], |_, _| {})).expect("a command buffer's commands");
                    joins.commit().expect("a commit");
                    kept.push(joins);
                }
                let room = counters.metrics().held_for_room;
                let reader = queue.command_buffer().expect("a command buffer");
                read_over(&reader, &texture, &r_read);
                reader.commit().expect("a commit");
                within(name, || reader.status() == CommandBufferStatus::Scheduled);
                counted = counters.metrics().held_for_room - room;
                let at = 3 * OVER_PITCH;
                let row = at as usize..(at + OVER_ROW) as usize;
                bytes[row.clone()].fill(0xC5);
                write(&managed, at, &bytes[row]);
                (managed.did_modify_range(at, OVER_ROW))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let capturer = queue.command_buffer().expect("a command buffer");
                read_over(&capturer, &texture, &c_read);
                capturer.commit().expect("a commit");
                within(name, || capturer.status() == CommandBufferStatus::Scheduled);
                kept.extend([reader, capturer]);
            });
            for command_buffer in &kept {
                finished(name, command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
            }
            let (after, unnamed) = (over_rows(&bytes), 3 * OVER_ROW as usize);
            let (r_bytes, c_bytes) = (read(&r_read, after.len()), read(&c_read, after.len()));
            println!(
                "{test}: {name}: {form}; R held for room {counted}; R read the bytes outside the \
                 named row {} (its first texels {:?}, the bytes' {:?}), C the new bytes {}",
                r_bytes[..unnamed] == before[..unnamed],
                &r_bytes[..8],
                &before[..8],
                c_bytes == after
            );
            assert_eq!(counted, u64::from(held), "{name}: R held for room");
            assert!(
                r_bytes[..unnamed] == before[..unnamed],
                "{name}: R reads the buffer's bytes outside the named row"
            );
            assert!(c_bytes == after, "{name}: C reads the new bytes");
            drop((texture, managed, r_read, c_read, arena, queue, kept));
            (device.release_block(cpu)).unwrap_or_else(|error| panic!("{name}: {error}"));
        });
    }

    /// [`over_read_before_a_capture`] with R in the group held when C commits.
    #[test]
    fn every_device_reads_a_managed_texture_over_in_a_group_ahead_of_a_later_capture() {
        over_read_before_a_capture("managed over before a capture", false);
    }

    /// [`over_read_before_a_capture`] with R held for room in no group when C commits.
    #[test]
    fn every_device_reads_a_managed_texture_over_held_for_room_ahead_of_a_later_capture() {
        over_read_before_a_capture("managed over held before a capture", true);
    }

    /// Encodes a copy of `from`'s bytes, packed, into the texture's texels.
    fn write_over(command_buffer: &CommandBuffer, texture: &Texture, from: &Buffer) {
        let mut blit = command_buffer.blit_command_encoder().expect("an encoder");
        let layout = BufferLayout {
            offset: 0,
            bytes_per_row: OVER_ROW,
            bytes_per_image: 0,
        };
        let size = [OVER_TEXELS[0], OVER_TEXELS[1], 1];
        let at = TexturePlace {
            slice: 0,
            level: 0,
            origin: [0; 3],
        };
        (blit.copy_from_buffer(from, layout, size, texture, at)).expect("a copy");
        blit.end_encoding();
    }

    /// A GPU write of a texture over a Managed buffer in an image of its own ([`over_texture`]),
    /// by a command buffer committed before a capture over part of the texture's bytes and not
    /// yet submitted as the capture is taken, reaches a command buffer committed after the
    /// capture outside the range named: Metal's device copy keeps a GPU write outside a
    /// `didModifyRange:`'s range, and the range's new bytes reach a command buffer committed after
    /// it (`MTLBuffer.h`). Behind the busy gate ([`behind_busy`]), W copies a pattern into the
    /// whole texture; then the consumer writes and names the texture's last row, and R, committed
    /// after on the same queue, takes the capture and reads the texture. R reads W's pattern in
    /// the other rows and, with two copies, the new bytes in the named row; with one, W's write
    /// there races the consumer's, which Metal leaves undefined. With `held`, [`UNSEEN_JOINS`]
    /// command buffers join the queue's group held first, so W is held for room in no group when
    /// R commits; else W is in the group held. There, with two copies, R, which takes the copy-in,
    /// leads its own group, so W's group goes first (`crate::capture::Captured::names_written`):
    /// W's submission is counted before R is scheduled, while the busy work still holds the GPU.
    /// Held for room, or with one copy, nothing goes before the busy work ends.
    fn over_write_before_a_capture(test: &str, held: bool) {
        each_device(test, |name, _, device| {
            let counters = device.counters();
            let cpu = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (managed, bytes, texture, form) = over_texture(name, &device, cpu.id, true);
            let len = over_rows(&bytes).len();
            let mut arena = Arena::new(&device);
            let (pattern, r_read) = (arena.buffer(), arena.buffer());
            write(&pattern, 0, &vec![0x3C; len]);
            let queue = device.new_command_queue(16).expect("a queue");
            let mut kept = Vec::new();
            let (mut counted, mut unsent, mut in_flight, mut split) = (0, false, (0, false), false);
            behind_busy(name, &device, &queue, |_, _, _| {
                for _ in 0..UNSEEN_JOINS * usize::from(held) {
                    let joins = queue.command_buffer().expect("a command buffer");
                    (joins.record_for_test(&[], |_, _| {})).expect("a command buffer's commands");
                    joins.commit().expect("a commit");
                    kept.push(joins);
                }
                let metrics = counters.metrics();
                let (room, sent) = (metrics.held_for_room, metrics.submissions);
                let writer = queue.command_buffer().expect("a command buffer");
                write_over(&writer, &texture, &pattern);
                writer.commit().expect("a commit");
                within(name, || writer.status() == CommandBufferStatus::Scheduled);
                counted = counters.metrics().held_for_room - room;
                let at = 3 * OVER_PITCH;
                write(&managed, at, &[0xC5; OVER_ROW as usize]);
                (managed.did_modify_range(at, OVER_ROW))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let reader = queue.command_buffer().expect("a command buffer");
                read_over(&reader, &texture, &r_read);
                // Read before R's commit, which takes the capture first: W's group may go as R
                // is staged after it (`crate::capture`).
                unsent = counters.metrics().submissions == sent;
                in_flight = device.core().work.in_flight(writer.queue());
                reader.commit().expect("a commit");
                within(name, || reader.status() == CommandBufferStatus::Scheduled);
                split = counters.metrics().submissions > sent;
                kept.extend([writer, reader]);
            });
            for command_buffer in &kept {
                finished(name, command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
            }
            let (r_bytes, named) = (read(&r_read, len), 3 * OVER_ROW as usize);
            let (outside, inside) = (
                r_bytes[..named].iter().all(|&byte| byte == 0x3C),
                r_bytes[named..].iter().all(|&byte| byte == 0xC5),
            );
            println!(
                "{test}: {name}: {form}; W held for room {counted}, the queue's submissions in \
                 flight and group held as R commits {in_flight:?}, W not yet submitted as R's \
                 commit took the capture {unsent}, W's group sent ahead of R as R came, the busy \
                 work still holding the GPU {split}; R read W's bytes outside the named row \
                 {outside} (its first texels {:?}), the new bytes in it {inside} (its first \
                 texels there {:?})",
                &r_bytes[..8],
                &r_bytes[named..named + 8]
            );
            assert_eq!(counted, u64::from(held), "{name}: W held for room");
            assert!(
                unsent,
                "{name}: W not yet submitted as the capture is taken"
            );
            let two = !device.has_unified_memory();
            assert_eq!(
                split,
                two && !held,
                "{name}: W's group goes ahead of R as R comes only in the group held, with two copies"
            );
            assert!(outside, "{name}: R reads W's bytes outside the named row");
            if two {
                assert!(inside, "{name}: R reads the new bytes in the named row");
            }
            drop((texture, managed, pattern, r_read, arena, queue, kept));
            (device.release_block(cpu)).unwrap_or_else(|error| panic!("{name}: {error}"));
        });
    }

    /// [`over_write_before_a_capture`] with W in the group held when R commits.
    #[test]
    fn every_device_keeps_a_managed_texture_over_s_write_in_a_group_outside_a_later_capture() {
        over_write_before_a_capture("managed over written before a capture", false);
    }

    /// [`over_write_before_a_capture`] with W held for room in no group when R commits.
    #[test]
    fn every_device_keeps_a_managed_texture_over_s_write_held_for_room_outside_a_later_capture() {
        over_write_before_a_capture("managed over written held before a capture", true);
    }

    /// A write of a texture over a Managed buffer in an image of its own ([`over_texture`]) by A3,
    /// in A's group held, and a capture of the texture's last row taken by B1, committed after on
    /// another queue and run ahead of A3 ([`two_queues`]): B1 takes the copy-in, which runs before
    /// the write and copies nothing back for it (`crate::capture`). B1 reads the buffer's bytes
    /// outside the named row and, with two copies, the new bytes in it; A4, committed before the
    /// capture and run after A3, reads A3's bytes outside the named row; and a blit that reads the
    /// buffer once all completed, taking the copy back A3 owes, reads A3's bytes there too.
    #[test]
    fn every_device_keeps_a_managed_texture_over_s_write_held_behind_a_passing_capture() {
        let test = "managed over written behind a passing capture";
        each_device(test, |name, _, device| {
            let cpu = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (managed, bytes, texture, form) = over_texture(name, &device, cpu.id, true);
            let len = over_rows(&bytes).len();
            let mut arena = Arena::new(&device);
            let (pattern, a_read, b_read, later) = (
                arena.buffer(),
                arena.buffer(),
                arena.buffer(),
                arena.buffer(),
            );
            write(&pattern, 0, &vec![0x3C; len]);
            let encode =
                |number: u32, command_buffer: &CommandBuffer, after: bool| match (number, after) {
                    (3, false) => write_over(command_buffer, &texture, &pattern),
                    (4, false) => read_over(command_buffer, &texture, &a_read),
                    (6, false) => read_over(command_buffer, &texture, &b_read),
                    _ => {}
                };
            let at = 3 * OVER_PITCH;
            let mut held = None;
            let log = two_queues(name, &device, (true, 2), encode, |_, a_work| {
                held = Some(device.core().work.in_flight(a_work[0].queue()));
                write(&managed, at, &[0xC5; OVER_ROW as usize]);
                (managed.did_modify_range(at, OVER_ROW))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            });
            let queue = device.new_command_queue(1).expect("a queue");
            let reader = queue.command_buffer().expect("a command buffer");
            let mut blit = reader.blit_command_encoder().expect("an encoder");
            let whole = later.length();
            (blit.copy_from_buffer_to_buffer(&managed, 0, &later, 0, whole)).expect("a copy");
            blit.end_encoding();
            reader.commit().expect("a commit");
            finished(name, &reader);
            let (rows, named) = (over_rows(&bytes), 3 * OVER_ROW as usize);
            let (a_bytes, b_bytes) = (read(&a_read, len), read(&b_read, len));
            let held_bytes = read(&later, whole as usize);
            let (b_outside, b_inside, a_outside) = (
                b_bytes[..named] == rows[..named],
                b_bytes[named..].iter().all(|&byte| byte == 0xC5),
                a_bytes[..named].iter().all(|&byte| byte == 0x3C),
            );
            let kept = (0..3).all(|row| {
                let at = (row * OVER_PITCH) as usize;
                held_bytes[at..at + OVER_ROW as usize]
                    .iter()
                    .all(|&byte| byte == 0x3C)
            });
            println!(
                "{test}: {name}: {form}; A's submissions in flight and group held as B1 commits \
                 {held:?}; log {log:?}; B1 read the buffer's bytes outside the named row \
                 {b_outside} (its first texels {:?}, the bytes' {:?}), the new bytes in it \
                 {b_inside}; A4 read A3's bytes outside it {a_outside}; the buffer read after \
                 holds A3's bytes outside it {kept}",
                &b_bytes[..8],
                &rows[..8]
            );
            assert_eq!(
                held,
                Some((2, true)),
                "{name}: A's group held as B1 commits"
            );
            assert_eq!(log, [0, 1, 6, 3, 4, 2], "{name}: B1 ran ahead of A3 to A5");
            assert!(
                b_outside,
                "{name}: B1 reads the buffer's bytes outside the named row"
            );
            if !device.has_unified_memory() {
                assert!(b_inside, "{name}: B1 reads the new bytes in the named row");
            }
            assert!(
                a_outside,
                "{name}: A4 reads A3's bytes outside the named row"
            );
            assert!(
                kept,
                "{name}: the buffer holds A3's bytes outside the named row"
            );
            drop((texture, managed, pattern, a_read, b_read, later, arena));
            drop((reader, queue));
            (device.release_block(cpu)).unwrap_or_else(|error| panic!("{name}: {error}"));
        });
    }

    /// `bytes`, a buffer's, with the texels of the texture over it at [`OVER_PITCH`] all `value`.
    fn over_filled(bytes: &[u8], value: u8) -> Vec<u8> {
        let mut filled = bytes.to_vec();
        for row in 0..u64::from(OVER_TEXELS[1]) {
            let at = (row * OVER_PITCH) as usize;
            filled[at..at + OVER_ROW as usize].fill(value);
        }
        filled
    }

    /// Encodes a copy of the whole of `buffer`, [`OVER_LEN`] bytes, into `into`.
    fn read_whole(command_buffer: &CommandBuffer, buffer: &Buffer, into: &Buffer) {
        let mut blit = command_buffer.blit_command_encoder().expect("an encoder");
        (blit.copy_from_buffer_to_buffer(buffer, 0, into, 0, OVER_LEN)).expect("a copy");
        blit.end_encoding();
    }

    /// A reader of a Managed buffer and a write of a texture over it in an image of its own
    /// ([`over_texture`]) run in the order Metal gives them, whichever was encoded first: on one
    /// queue, the order they are committed, a queue being "a serial queue of command buffers"
    /// (`MTLCommandQueue.h`) to whose end a commit appends a command buffer not yet enqueued, as
    /// `enqueue` does (`MTLCommandBuffer.h`); on two, a reader committed once the write completed
    /// runs after it. P reads the texture and completes, so its image holds the buffer's bytes. W
    /// copies a pattern into the whole texture and R copies the whole buffer into a Shared buffer,
    /// W encoded first where `writer_first`, else R; the one encoded later is committed first.
    /// With `other_queue`, R is of another queue, committed once W completed. Once both completed,
    /// L copies the whole buffer again. R reads the buffer's bytes where it was committed first,
    /// else W's pattern in the texture's rows; L reads W's pattern there; both read the buffer's
    /// bytes outside them. The form, the copies R recorded as it was encoded, those made as R and
    /// W went, and what R and L read are printed.
    fn over_read_in_commit_order(test: &str, writer_first: bool, other_queue: bool) {
        each_device(test, |name, _, device| {
            let counters = device.counters();
            let cpu = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (managed, bytes, texture, form) = over_texture(name, &device, cpu.id, true);
            let mut arena = Arena::new(&device);
            let (pattern, p_read) = (arena.buffer(), arena.buffer());
            write(&pattern, 0, &vec![0x3C; over_rows(&bytes).len()]);
            let whole = |at: u64| {
                (device.new_buffer(cpu.id, at * OVER_LEN, OVER_LEN, Some(0)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            let (r_read, l_read) = (whole(1), whole(2));
            let queue = device.new_command_queue(16).expect("a queue");
            let other = device.new_command_queue(16).expect("a queue");
            let filler = queue.command_buffer().expect("a command buffer");
            read_over(&filler, &texture, &p_read);
            filler.commit().expect("a commit");
            finished(name, &filler);
            let writer = queue.command_buffer().expect("a command buffer");
            let reads_on = if other_queue { &other } else { &queue };
            let reader = reads_on.command_buffer().expect("a command buffer");
            if writer_first {
                write_over(&writer, &texture, &pattern);
            }
            let copies = counters.metrics().over_copies;
            read_whole(&reader, &managed, &r_read);
            let recorded = counters.metrics().over_copies - copies;
            if !writer_first {
                write_over(&writer, &texture, &pattern);
            }
            let written = over_filled(&bytes, 0x3C);
            let (first, second, r_owed, owed) = match writer_first {
                true => (
                    &reader,
                    &writer,
                    &bytes,
                    "R first, R read the buffer's bytes",
                ),
                false => (&writer, &reader, &written, "W first, R read W's pattern"),
            };
            let copies = counters.metrics().over_copies;
            first.commit().expect("a commit");
            if other_queue {
                finished(name, first);
            }
            second.commit().expect("a commit");
            for command_buffer in [&writer, &reader] {
                finished(name, command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
            }
            let went = counters.metrics().over_copies - copies;
            let later = queue.command_buffer().expect("a command buffer");
            read_whole(&later, &managed, &l_read);
            later.commit().expect("a commit");
            finished(name, &later);
            let len = OVER_LEN as usize;
            let (r_bytes, l_bytes) = (read(&r_read, len), read(&l_read, len));
            println!(
                "{test}: {name}: {form}; W encoded first {writer_first}, R of another queue \
                 {other_queue}; copies R recorded as it was encoded {recorded}, made as R and W \
                 went {went}; committed {owed} {} (its first texels {:?}); L read W's pattern {} \
                 (its first texels {:?})",
                r_bytes == *r_owed,
                &r_bytes[..8],
                l_bytes == written,
                &l_bytes[..8]
            );
            assert!(
                r_bytes == *r_owed,
                "{name}: R reads what Metal's order leaves"
            );
            assert!(
                l_bytes == written,
                "{name}: L reads W's pattern in the texture's rows"
            );
            drop((texture, managed, pattern, p_read, r_read, l_read, arena));
            drop((filler, writer, reader, later, queue, other));
            (device.release_block(cpu)).unwrap_or_else(|error| panic!("{name}: {error}"));
        });
    }

    /// [`over_read_in_commit_order`] with W encoded first and R committed first.
    #[test]
    fn every_device_reads_a_managed_buffer_ahead_of_a_write_over_it_encoded_before_it() {
        over_read_in_commit_order(
            "managed read ahead of a write encoded before it",
            true,
            false,
        );
    }

    /// [`over_read_in_commit_order`] with R encoded first and W committed first.
    #[test]
    fn every_device_reads_a_managed_buffer_after_a_write_over_it_encoded_after_it() {
        over_read_in_commit_order("managed read after a write encoded after it", false, false);
    }

    /// [`over_read_in_commit_order`] with R encoded first on another queue and committed once W
    /// completed.
    #[test]
    fn every_device_reads_a_managed_buffer_on_another_queue_after_a_completed_write_over_it() {
        over_read_in_commit_order(
            "managed read on another queue after a completed write",
            false,
            true,
        );
    }
}
