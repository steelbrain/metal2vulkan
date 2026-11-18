//! Managed buffers: changed ranges merge where they overlap or touch; and, under a device cfg, on
//! every device of the cfg's kind, in the form the device's memory-type query picks: with a device
//! copy where Shared memory is not device-local, the copy holds the consumer's first bytes and
//! zeros after them, a changed range reaches it at the next submission and a range not marked
//! changed does not, and changes past the upload's calls submit it; without one, shaders read the
//! consumer's bytes as they are and a changed range submits nothing. A range past the buffer and a
//! Managed buffer over a Private block are refused, and every gauge reads zero after teardown.

use crate::submit::merge;

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
mod on_device {
    use crate::submit::UPLOAD_ENTRIES;
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{Buffer, Device, Error, MemoryClass, Refusal};
    use ash::vk;

    const LEN: u64 = 64 << 10;

    /// The bytes shaders reach at `managed`'s address, copied by the GPU into `into`, a Shared
    /// buffer as long.
    fn device_bytes(device: &Device, managed: &Buffer, into: &Buffer) -> Vec<u8> {
        let queue = device.new_command_queue(1).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let (source, from) = managed.device_range();
        let (target, to) = into.device_range();
        command_buffer
            .record_for_test(&[managed, into], |device, command| {
                let region = vk::BufferCopy {
                    src_offset: from,
                    dst_offset: to,
                    size: LEN,
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
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), LEN as usize) }.to_vec()
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

    #[test]
    fn every_device_copies_a_managed_buffer_s_changed_ranges() {
        each_device("managed", |name, _, device| {
            let core = device.core();
            let copied = !core.shared_is_device_local();
            let counters = device.counters();
            let cpu = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let into = (device.new_buffer(cpu.id, LEN, LEN, true))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let first: Vec<u8> = (0..LEN).map(|i| (i % 251) as u8).collect();
            let start = (device.new_buffer(cpu.id, 0, LEN, false))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            write(&start, 0, &first);
            drop(start);

            // The first half copied, the rest zero.
            let managed = (device.new_managed_buffer(cpu.id, 0, LEN, LEN / 2))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut expected = first.clone();
            if copied {
                expected[LEN as usize / 2..].fill(0);
            }
            assert!(
                device_bytes(&device, &managed, &into) == expected,
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
            assert!(
                device_bytes(&device, &managed, &into) == expected,
                "{name}: the bytes after the changes (device copy: {copied})"
            );

            // Changes past the upload's calls submit it; without a copy nothing is submitted.
            let before = counters.metrics().submissions;
            for change in 0..UPLOAD_ENTRIES as u64 + 8 {
                managed
                    .did_modify_range((change * 8) % LEN, 4)
                    .unwrap_or_else(|error| panic!("{name}: change {change}: {error}"));
            }
            let submitted = counters.metrics().submissions - before;
            match copied {
                true => assert!(submitted >= 1, "{name}: the changes went"),
                false => assert_eq!(submitted, 0, "{name}: nothing to copy"),
            }

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
            match device.new_managed_buffer(private.id, 0, 256, 0) {
                Err(Error::Refused(Refusal::ManagedBlockNotShared { block })) => {
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
            for block in [cpu.id, private.id] {
                device
                    .release_block(block)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            }
        });
    }
}
